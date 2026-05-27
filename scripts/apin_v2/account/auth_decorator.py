"""API Console `@require_scope` decorator and its dependent helpers.

Spec contract: spec_v7.md §5.4 (lines 1784-1866). The decorator structure
is copied verbatim from the spec (all 11 ApiError raises preserved in
order). Helpers are split across phases:

  Implemented now (Phase 2.2):
    _was_bearer_or_xapikey       header inspection — pure function
    _resolve_key_from_request    DB lookup via auth_db.lookup_api_key_full
    now_utc                      datetime.now(timezone.utc)
    _seconds_to_midnight_utc     deterministic math
    _parse_iso_utc               normalised parser for the timestamps in DB

  Deferred to later phases (stubbed permissive HERE with WARNING logs):
    _ip_in_allowlist             Phase 11 / §11.5 — returns True for now
    _origin_in_allowlist         Phase 11 / §11.7 — returns True for now
    _rate_limit_check            Phase 10 / §10  — returns "ok" for now
    _quota_ok                    Phase 10 / §10  — returns True for now
    _record_usage                Phase 10 / §12  — no-op for now

Why permissive stubs (not NotImplementedError):
    The decorator must be USABLE end-to-end as soon as Phase 2 ships, so
    Phase 2.4 routes can mount it. Each stub logs a WARNING at module
    import time so the operator can see what's still deferred. When the
    real Phase 10/11 implementations land, they OVERRIDE these stubs by
    re-binding the module-level symbol (or by direct replacement in the
    routes layer).

Audit trail for the §5.4 implementation:
    - REV-R5-I01: rate_limited raise uses a single parenthesised expression
      with headers as a kwarg (no orphan `headers={...}` line).
    - REV-R5-I02: `ApiError` is imported at the top of the file.
    - REV-R5-I03: every raise uses `ApiError`, not `HTTPException`. The
      §3 envelope handler in `api_envelope.py:_failure` merges
      `err.headers` into the JSONResponse.
    - REV-R5-I06: no HTTPException → all errors return the conforming
      §3 envelope shape (not the legacy `{"detail": "..."}` form).
    - PDA-F04: console-only routes refuse Bearer/X-API-Key auth EARLY.
    - PDA-F11: rate-limit backend failure is FAIL-CLOSED (503), not open.
    - PDA-F49: revoked/expired/deleted statuses each map to a distinct
      canonical error code ("key_disabled" / "key_expired" / "key_deleted"
      etc., depending on `key['status']`).
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Optional

# Ensure project root is importable when this module is loaded from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import Request

from scripts.apin_v2.api_envelope import ApiError   # §3 envelope-conforming exception (REV-R5-I02)
from scripts.apin_v2 import auth_db                 # lookup_api_key_full lives here

log = logging.getLogger("apin_v2.account.auth_decorator")


# PDA-P2-R1-F09 production-gate. ALL five Phase 10/11 helpers below are
# permissive stubs. Running this module in APIN_ENV=production with the
# stubs still active would silently trust unverified IP/origin/rate/quota
# checks — a deployment-time disaster waiting to happen. Fail-closed at
# import time so the misconfiguration is caught before the first request.
#
# Operator opt-in: set APIN_ALLOW_DEFERRED_STUBS=1 to acknowledge running
# with permissive defaults (e.g. during a phased rollout where Phase 10/11
# helpers will land before public traffic).
def _check_stub_production_gate() -> None:
    env = (os.environ.get("APIN_ENV") or "").strip().lower()
    if env != "production":
        return
    if (os.environ.get("APIN_ALLOW_DEFERRED_STUBS") or "").strip() == "1":
        log.critical(
            "auth_decorator.py loaded in PRODUCTION with five Phase 10/11 "
            "stubs still active (APIN_ALLOW_DEFERRED_STUBS=1 acknowledges). "
            "Stubs: _ip_in_allowlist, _origin_in_allowlist, _rate_limit_check, "
            "_quota_ok, _record_usage. All return permissive defaults. "
            "This is a DEPLOYMENT MISCONFIGURATION; fix before public traffic."
        )
        return
    raise RuntimeError(
        "auth_decorator.py: refusing to load in APIN_ENV=production while "
        "the Phase 10/11 helper stubs are still in place. The five stubs "
        "(_ip_in_allowlist, _origin_in_allowlist, _rate_limit_check, "
        "_quota_ok, _record_usage) return permissive defaults — running "
        "them in production silently trusts unverified auth signals. "
        "Either deploy the Phase 10/11 implementations or set "
        "APIN_ALLOW_DEFERRED_STUBS=1 in the environment to explicitly "
        "acknowledge the risk."
    )


_check_stub_production_gate()


# ── Spec constants ─────────────────────────────────────────────────────────

# Spec §5.4 line 1805: console-only paths reject API-key auth.
CONSOLE_ONLY_PATH = "/api/account/"

# Status values that the decorator treats as authenticatable.
ACTIVE_STATUSES = frozenset({"active", "rotating"})


# ── Time helpers ───────────────────────────────────────────────────────────

def now_utc() -> datetime:
    """UTC `datetime` (tz-aware). Used by the expiration check in the
    decorator. Kept as a module-level function so tests can monkey-patch it
    deterministically (e.g. `with patch('...now_utc', lambda: fixed_dt): ...`).
    """
    return datetime.now(timezone.utc)


def _parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 UTC timestamp from the DB. Accepts both `Z`
    suffix and `+00:00`. Returns None on parse failure (defensive — a
    malformed DB timestamp should never crash the auth path).
    """
    if not isinstance(s, str) or not s:
        return None
    try:
        # datetime.fromisoformat accepts `+00:00` natively; convert `Z`.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # Naive timestamp — assume UTC.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _seconds_to_midnight_utc(*, now: Optional[datetime] = None) -> int:
    """Seconds remaining until the next UTC midnight. Used as the
    `Retry-After` header value for `quota_exceeded` (spec §5.4 line 1858).

    The `now` parameter is for deterministic testing — production calls
    pass nothing and get the live `now_utc()` value.
    """
    n = now or now_utc()
    # Next midnight = today's midnight + 1 day
    today_midnight = n.replace(hour=0, minute=0, second=0, microsecond=0)
    next_midnight = today_midnight.replace(day=today_midnight.day) \
        if False else _add_one_day(today_midnight)
    delta = next_midnight - n
    # Round UP to the nearest whole second (clients shouldn't undershoot).
    return max(1, int(delta.total_seconds()) + (1 if delta.microseconds > 0 else 0))


def _add_one_day(dt: datetime) -> datetime:
    """Add exactly one calendar day, avoiding the deprecation pitfall of
    `dt.replace(day=dt.day + 1)` at month boundaries."""
    from datetime import timedelta
    return dt + timedelta(days=1)


# ── Request-inspection helpers ─────────────────────────────────────────────

def _was_bearer_or_xapikey(request: Request) -> bool:
    """True if the request presented an `Authorization: Bearer …` header
    OR an `X-API-Key: …` header (any value, including empty).

    Spec §5.4 line 1813: this is the PDA-F04 console-only check — if a
    caller hits `/api/account/*` with API-key-style auth, we reject with
    `console_only_route` BEFORE looking the key up.

    Header lookup is case-insensitive (FastAPI's `Request.headers` already
    handles that), so we don't double-fold.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return False
    # Authorization header — only "Bearer" prefix counts; "Basic" / others
    # are session-cookie auth's territory (or unauthenticated).
    auth = headers.get("authorization") or ""
    if auth.strip().lower().startswith("bearer "):
        return True
    # X-API-Key is the alternative header form for API-key auth.
    if "x-api-key" in headers:
        return True
    return False


def _extract_bearer_token(request: Request) -> Optional[str]:
    """Pull the raw token out of `Authorization: Bearer …` OR `X-API-Key`.

    Returns the raw token string (without prefix) or None. Does NOT validate
    format here — that's `_resolve_key_from_request`'s job via the regex.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[len("Bearer "):].strip()
        return token or None
    api_key = headers.get("x-api-key")
    if api_key:
        return api_key.strip() or None
    return None


def _resolve_key_from_request(request: Request) -> Optional[dict]:
    """Find the API key row that authenticated this request.

    Returns the FULL key dict (per `auth_db.lookup_api_key_full`) on hit,
    or None if no valid key was presented. The decorator's `key is None`
    branch maps None → `invalid_or_missing_token` (401).

    NOTE: this does NOT enforce status/expiration — that's the decorator.
    Hard-deleted keys (`deleted_at IS NOT NULL`) return None here because
    `lookup_api_key_full` filters them out at the DB layer.
    """
    raw = _extract_bearer_token(request)
    if raw is None:
        return None
    return auth_db.lookup_api_key_full(raw)


# ── Deferred helpers (Phase 10 + 11 work) ──────────────────────────────────
#
# These are PERMISSIVE STUBS — they let the decorator function end-to-end
# during Phases 2-9. Each logs a one-time WARNING at first use so the
# operator can see what's still mocked.

_STUB_WARNED: set[str] = set()


def _warn_once(name: str, message: str) -> None:
    """One-time WARNING per stub name. Avoids log spam in tight loops.

    Module-import-time `_check_stub_production_gate()` already enforces
    the production opt-in (PDA-P2-R1-F09). This helper just surfaces
    a one-time WARNING per stub for dev observability.
    """
    if name in _STUB_WARNED:
        return
    _STUB_WARNED.add(name)
    log.warning("%s — STUB (Phase 2.2 placeholder): %s", name, message)


def _ip_in_allowlist(request: Request, key: dict) -> bool:
    """STUB (Phase 11 / §11.5): always returns True.

    Real implementation will:
      1. Read client IP via `apin_server._client_ip_rightmost(request)`
      2. If `key['ip_allowlist']` is None or empty → True (no restriction)
      3. Parse each entry as ipaddress.ip_network (handles CIDR + bare IPs)
      4. Return True if client IP is in ANY allowed network
    """
    if key.get("ip_allowlist"):
        _warn_once(
            "_ip_in_allowlist",
            f"Key {key.get('public_id', '?')} has ip_allowlist set but "
            "the Phase-11 enforcement helper is not yet wired. "
            "Treating as ALLOWED."
        )
    return True


def _origin_in_allowlist(request: Request, key: dict) -> bool:
    """STUB (Phase 11 / §11.7): always returns True.

    Real implementation will:
      1. Read `Origin` header
      2. If `key['enforce_origin_for_non_browser']==0` AND no Origin → True
         (CLI / server-to-server calls have no Origin; legacy keys exempt)
      3. If `key['origin_allowlist']` is None → True (no restriction)
      4. Reject `null` literal Origin (PDA-F08 — sandboxed iframes)
      5. Return True if Origin matches any entry in the allowlist
    """
    if key.get("origin_allowlist"):
        _warn_once(
            "_origin_in_allowlist",
            f"Key {key.get('public_id', '?')} has origin_allowlist set but "
            "the Phase-11 enforcement helper is not yet wired. "
            "Treating as ALLOWED."
        )
    return True


def _rate_limit_check(key: dict) -> str:
    """STUB (Phase 10 / §10): always returns "ok".

    Real implementation will return one of:
      - "ok"          — under the per-minute cap
      - "limited"     — over the cap; decorator raises rate_limited (429)
      - "unavailable" — rate-limit backend (Redis-or-equivalent) is down;
                        decorator raises rate_limit_unavailable (503,
                        fail-CLOSED per PDA-F11)
    """
    return "ok"


def _quota_ok(key: dict) -> bool:
    """STUB (Phase 10 / §10): always returns True.

    Real implementation will atomically increment `quota_used_today` and
    return False if the daily cap was hit (decorator raises quota_exceeded
    with `Retry-After` set to seconds-to-midnight-UTC).
    """
    return True


def _record_usage(key: dict, request: Request,
                  response_status: int = 200,
                  *, error_code: Optional[str] = None) -> None:
    """Phase 9.A — REAL usage telemetry (replaces the long-standing stub).

    Writes one row to the in-memory `usage_recorder` buffer. A background
    flusher (started by `apin_server._ensure_heartbeat`) drains the buffer
    every ~2 s into:
      - api_key_request_log   (one row per request, raw log)
      - api_key_usage_minute  (per-minute aggregate: requests/errors/etc)
      - api_keys              (request_count / error_count / last_used_*)

    Alert producers still run here (5xx / high-latency / first-IP) — they
    are independent of the buffer flush and must fire on this tick.

    Always called from the decorator's `finally:` — must NEVER raise.
    """
    import time as _time
    from scripts.apin_v2 import auth_db as _adb
    from scripts.apin_v2 import usage_recorder as _ur

    user_id = int(key.get("user_id") or 0)
    public_id = key.get("public_id")
    key_name = key.get("name", "?")
    request_id = getattr(request.state, "request_id", None) or "?"

    # ── Latency ──────────────────────────────────────────────────
    latency_ms = None
    started = getattr(request.state, "_apin_started_at", None)
    if started is not None:
        latency_ms = int((_time.monotonic() - started) * 1000)

    # ── Buffer the row for the flusher ──────────────────────────
    try:
        # IP — best-effort. Production should swap in
        # apin_server._client_ip_rightmost so a proxy chain is honoured,
        # but importing it here would create a circular import. The
        # immediate-client IP is correct for the common case.
        client_ip = ""
        try:
            client_ip = (request.client.host if request.client else "") or ""
        except Exception:
            pass
        ua = ""
        try:
            ua = request.headers.get("user-agent", "") or ""
        except Exception:
            pass
        # Auth source — bearer / x_api_key / session. The request only
        # gets here through @require_scope, so it was bearer or x-api-key.
        via = "bearer" if _was_bearer_or_xapikey(request) else "session"
        # bytes_in — Content-Length if the client sent it. Multipart
        # uploads don't always include it; that's fine, we store NULL.
        bytes_in = None
        try:
            cl = request.headers.get("content-length")
            if cl is not None and cl.isdigit():
                bytes_in = int(cl)
        except Exception:
            pass

        method = (request.method or "GET").upper()
        # URL path, not full URL — query string is *not* part of the path
        # for analytics purposes (queryStrings vary per call).
        try:
            path = request.url.path or "/"
        except Exception:
            path = "/"

        if public_id:
            _ur.record_request(
                key_id=public_id,
                user_id=user_id,
                method=method,
                path=path,
                status_code=int(response_status),
                latency_ms=latency_ms,
                ip=client_ip or None,
                ua=ua or None,
                bytes_in=bytes_in,
                bytes_out=None,           # response not yet finalised here
                error_code=error_code,
                via=via,
                rate_limited=(int(response_status) == 429),
                quota_blocked=(error_code == "quota_exceeded"),
            )
    except Exception as e:
        log.debug("buffer.record_request failed: %s", e)

    # ── 5xx — fire only on server errors (4xx is the caller's fault) ──
    if response_status >= 500:
        try:
            _adb.emit_alert(
                user_id, "request.error_5xx",
                key_id=public_id,
                action={"kind": "view_request",
                        "public_id": public_id,
                        "request_id": request_id},
                key_name=key_name,
                request_id=request_id,
                status=response_status,
            )
        except Exception:
            pass

    # ── High latency — absolute threshold (5 s) ──
    HIGH_LATENCY_MS = 5000
    if latency_ms is not None and latency_ms >= HIGH_LATENCY_MS:
        try:
            _adb.emit_alert(
                user_id, "request.high_latency",
                key_id=public_id,
                action={"kind": "view_request",
                        "public_id": public_id,
                        "request_id": request_id},
                request_id=request_id,
                latency_ms=latency_ms,
                p99_ms=HIGH_LATENCY_MS,
            )
        except Exception:
            pass

    # ── First-use-from-new-IP detection ──
    # Source-of-truth: api_key_request_log if it has rows for this key.
    # While that log is still a stub, fall back to an in-memory set keyed
    # on (user_id, public_id) on app.state. This means the alert may
    # double-fire across process restarts but never misses.
    try:
        client_ip = (request.client.host if request.client else "") or ""
        if client_ip and public_id:
            seen_key = (user_id, public_id)
            app_state = getattr(request, "app", None)
            seen_map = None
            if app_state is not None and hasattr(app_state, "state"):
                seen_map = getattr(app_state.state, "_apin_seen_ips", None)
                if seen_map is None:
                    seen_map = {}
                    setattr(app_state.state, "_apin_seen_ips", seen_map)
            if seen_map is not None:
                prev = seen_map.get(seen_key)
                if prev is None:
                    seen_map[seen_key] = {client_ip}
                    is_new_ip = True   # first request ever for this key
                elif client_ip in prev:
                    is_new_ip = False
                else:
                    prev.add(client_ip)
                    is_new_ip = True
                # Only alert on subsequent new IPs — the very first request
                # of a key's life is its "home" IP, not a security event.
                # Heuristic: if `prev` was None we just seeded; alert only
                # when we ADD to an existing set.
                if is_new_ip and prev is not None:
                    _adb.emit_alert(
                        user_id, "key.first_use_from_new_ip",
                        key_id=public_id,
                        action={"kind": "view_key", "public_id": public_id},
                        key_name=key_name,
                        ip=client_ip,
                    )
    except Exception:
        # Never let new-IP detection break a successful request.
        pass


# ── The decorator (verbatim per spec §5.4, R6-corrected) ───────────────────

def require_scope(*needed: str):
    """Decorator that wraps a route handler with the full auth pipeline.

    Usage:
        @app.get("/api/account/keys")
        @require_scope()                     # session-cookie-only route
        async def list_keys(request: Request):
            user_id = request.state.session.user_id  # set by session middleware
            ...

        @app.post("/api/predict")
        @require_scope("predict:write")     # API-key route requiring scope
        async def predict(request: Request, file: UploadFile = File(...)):
            ...

    The decorator runs (in order):
      1. PDA-F04 — console-only-route check (reject API-key auth on /account/*)
      2. Resolve the key from headers (Bearer / X-API-Key)
      3. Status checks: legacy_pending / non-active / expired
      4. IP allowlist check  (Phase 11)
      5. Origin allowlist check  (Phase 11)
      6. Scope check
      7. Rate-limit check  (Phase 10)
      8. Quota check  (Phase 10)
      9. Call the wrapped handler
      10. Record usage in a `finally:` block (Phase 10)

    Every failure raises `ApiError` with the canonical code from §26.
    The §9.1 ApiError handler in `api_envelope._failure` produces the
    9-key envelope response with the right HTTP status and any merged
    headers (Retry-After, X-Missing-Scope, etc.).
    """
    def deco(fn):
        @wraps(fn)
        async def wrapper(*args, request: Request, **kwargs):
            # ── 1. PDA-F04: console-only paths reject Bearer/X-API-Key auth ──
            if request.url.path.startswith(CONSOLE_ONLY_PATH):
                if _was_bearer_or_xapikey(request):
                    raise ApiError(
                        "console_only_route",
                        "This route accepts session-cookie auth only.",
                        hint=("Use the /account UI; do not send Bearer or "
                              "X-API-Key headers here."),
                    )

            # ── 2. Resolve the key (returns None if no/invalid token) ──
            key = _resolve_key_from_request(request)
            if key is None:
                raise ApiError(
                    "invalid_or_missing_token",
                    "Authentication token is missing or invalid.",
                )

            # ── 3a. Sentinel status: legacy keys awaiting re-issue ──
            if key["status"] == "legacy_pending":
                raise ApiError(
                    "key_pending_migration",
                    "This key requires re-issuance before use.",
                    hint="Visit /account/api/keys to mint a new key.",
                )

            # ── 3b. Non-active status (revoked / expired / deleted / etc.) ──
            if key["status"] not in ACTIVE_STATUSES:
                # PDA-F49: codes like key_revoked, key_expired, key_deleted
                # are formed from the status — all registered in §26.
                code = f"key_{key['status']}" if key["status"] else "key_disabled"
                raise ApiError(
                    code,
                    f"Key is {key['status']!r} — cannot be used.",
                )

            # ── 3c. Expiration check (key has expires_at and now > that) ──
            exp = _parse_iso_utc(key.get("expires_at"))
            if exp is not None and now_utc() > exp:
                raise ApiError(
                    "key_expired",
                    "Key has expired.",
                )

            # ── 4. IP allowlist (STUB — Phase 11) ──
            if not _ip_in_allowlist(request, key):
                raise ApiError(
                    "ip_not_allowed",
                    "Request source IP is not in this key's allowlist.",
                )

            # ── 5. Origin allowlist (STUB — Phase 11) ──
            if not _origin_in_allowlist(request, key):
                raise ApiError(
                    "origin_not_allowed",
                    "Request Origin is not in this key's allowlist.",
                )

            # ── 6. Scope check ──
            scopes = set(key.get("scopes") or [])
            for required_scope in needed:
                if required_scope not in scopes:
                    raise ApiError(
                        "missing_scope",
                        f"This route requires scope {required_scope!r}.",
                        details={"required_scope": required_scope},
                        headers={"X-Missing-Scope": required_scope},
                    )

            # ── 7. Rate-limit check (STUB — Phase 10) ──
            rl = _rate_limit_check(key)
            if rl == "limited":
                # Phase 8.H · quota.rate_limit_hit. Stays usable once the
                # stub becomes real — the emit happens regardless.
                try:
                    from scripts.apin_v2 import auth_db as _adb
                    _adb.emit_alert(
                        int(key.get("user_id") or 0), "quota.rate_limit_hit",
                        key_id=key.get("public_id"),
                        action={"kind": "view_key",
                                "public_id": key.get("public_id")},
                        key_name=key.get("name", "?"),
                        rpm=key.get("rate_limit_per_min", "?"),
                    )
                except Exception:
                    pass
                # REV-R5-I01: single parenthesised expression, no orphan line.
                raise ApiError(
                    "rate_limited",
                    "Rate limit exceeded.",
                    hint="Wait per Retry-After and retry.",
                    headers={"Retry-After": "60"},
                )
            if rl == "unavailable":
                # PDA-F11: fail-CLOSED. We don't know if we're safe.
                raise ApiError(
                    "rate_limit_unavailable",
                    "Rate-limit backend unavailable; failing closed.",
                    headers={"Retry-After": "5"},
                )

            # ── 8. Quota check (STUB — Phase 10) ──
            if not _quota_ok(key):
                # Phase 8.H · quota.daily_exceeded.
                try:
                    from scripts.apin_v2 import auth_db as _adb
                    _adb.emit_alert(
                        int(key.get("user_id") or 0), "quota.daily_exceeded",
                        key_id=key.get("public_id"),
                        action={"kind": "adjust_quota"},
                        cap=key.get("quota_per_day", "?"),
                    )
                except Exception:
                    pass
                raise ApiError(
                    "quota_exceeded",
                    "Daily quota for this key has been exhausted.",
                    headers={"Retry-After": str(_seconds_to_midnight_utc())},
                )

            # ── 9. Stash the key on request.state for the handler ──
            request.state.api_key = key
            # Phase 8.H · stash arrival timestamp for high-latency detection.
            import time as _time
            request.state._apin_started_at = _time.monotonic()

            # ── 10. Call the handler, record usage in finally ──
            response_status = 200
            error_code: Optional[str] = None
            try:
                result = await fn(*args, request=request, **kwargs)
                return result
            except ApiError as ae:
                # ApiError shapes into a 4xx/5xx — capture status for alerts.
                error_code = ae.code
                try:
                    from scripts.apin_v2 import api_envelope as _env
                    response_status = int(
                        _env.ERROR_STATUS.get(ae.code, 500))
                except Exception:
                    response_status = 500
                raise
            except Exception:
                response_status = 500
                error_code = "internal_error"
                raise
            finally:
                # Must not raise — the request has already succeeded /
                # responded; an exception here would corrupt the response.
                try:
                    _record_usage(key, request,
                                   response_status=response_status,
                                   error_code=error_code)
                except Exception as e:
                    log.warning("_record_usage failed (non-fatal): %s", e)

        return wrapper
    return deco
