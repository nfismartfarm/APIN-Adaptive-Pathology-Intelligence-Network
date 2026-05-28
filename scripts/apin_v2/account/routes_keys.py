"""API Console — key CRUD routes.

Spec contract: spec_v7.md §7.1 (lines 2672-2758).

Six endpoints exposed under `/api/account/keys`:
    GET    /api/account/keys                       list (cursor pagination)
    POST   /api/account/keys                       mint (one-time view)
    GET    /api/account/keys/{public_id}           fetch one
    PATCH  /api/account/keys/{public_id}           edit
    POST   /api/account/keys/{public_id}/rotate    rotate
    DELETE /api/account/keys/{public_id}           hard-delete

Auth model:
    - Session-cookie auth (apin_v2_session). Bearer/X-API-Key REJECTED by
      TokenFormatMiddleware (slot 4 / PDA-F04).
    - POST/PATCH/DELETE additionally gated by SudoMiddleware (slot 7) —
      handled at the middleware layer, not here.
    - CSRF header check (X-Console-Csrf) is the route's job — see
      `_require_csrf` below. Deferred to Phase 4 sudo-cookie integration
      for full session.csrf_token rotation flow; current Phase 2.4
      implementation enforces presence + non-empty value as a basic
      defence.

What's STUBBED (deferred to later phases):
    - Audit emission (every mutation should write to api_key_audit with
      hash chain) — Phase 4. We log a structured WARNING for now so the
      operator can see what would have been audited.
    - Idempotency cache (POST should honour Idempotency-Key per §6.9) —
      Phase 4. For now, idempotent retries create fresh keys; the spec's
      `idempotency_keys` table is wired in Phase 1 but the cache helper
      isn't.
    - Scope catalogue validation — uses a static allowlist; the full
      Appendix A enforcement is in §25's catalogue helper (Phase 4).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import tokens as T

log = logging.getLogger("apin_v2.account.routes_keys")

router = APIRouter(prefix="/api/account/keys", tags=["account/keys"])


# ── Scope catalogue (subset for Phase 2.4 — full catalogue in Phase 4) ────

# Console-only scopes — CANNOT be assigned to user-minted keys
# (§5.2 lines 1761-1766; POST returns 400 invalid_scope).
_CONSOLE_ONLY_SCOPES = frozenset({"keys:read", "keys:write", "keys:admin"})

# Subset of Appendix A allowable for user keys. Full enumeration in Phase 4.
_USER_ASSIGNABLE_SCOPES = frozenset({
    "predict:write", "predict:read", "predict:delete",
    "models:read", "models:benchmarks",
    "feedback:write",
    "reports:read", "reports:write",
    "usage:read",
    "alerts:read", "alerts:write",
    "webhooks:read", "webhooks:write",
    "account:read", "account:write",
})


# ── Auth helpers ──────────────────────────────────────────────────────────
#
# Phase 5.1 (WI-P4-DEDUP-SESS): the previous in-file definitions of
# `_get_session_user`, `_require_csrf`, and the `_SESSION_COOKIE_NAME`
# constant are now hosted in `_session_helpers` (canonical single source,
# also used by `routes_sudo`). We re-export under the underscored names
# for backward compatibility with internal call sites.
from scripts.apin_v2.account import _session_helpers as _sh

_SESSION_COOKIE_NAME = _sh.SESSION_COOKIE_NAME
_get_session_user = _sh.get_session_user
_require_csrf = _sh.require_csrf


# ── Field validators ──────────────────────────────────────────────────────

# Spec §7.1 line 2686: name 1-80 chars, no control chars.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")


def _validate_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ApiError("invalid_name", "Key name must be a string.")
    name = name.strip()
    if not (1 <= len(name) <= 80):
        raise ApiError(
            "invalid_name",
            "Key name must be 1-80 chars after stripping whitespace.",
            details={"received_length": len(name)},
        )
    if _CONTROL_CHARS_RE.search(name):
        raise ApiError(
            "invalid_name",
            "Key name contains control characters (forbidden).",
        )
    return name


def _validate_environment(env: Any) -> str:
    if env not in ("live", "test"):
        raise ApiError(
            "invalid_parameter",
            "environment must be 'live' or 'test'.",
            details={"field": "environment", "received": env},
        )
    return env


def _validate_scopes(scopes: Any) -> list:
    if not isinstance(scopes, list):
        raise ApiError(
            "invalid_scope",
            "scopes must be an array of strings.",
            details={"field": "scopes"},
        )
    if not scopes:
        raise ApiError(
            "invalid_scope",
            "scopes must contain at least one entry.",
        )
    seen = set()
    out = []
    for s in scopes:
        if not isinstance(s, str):
            raise ApiError("invalid_scope",
                           f"each scope must be a string; got {type(s).__name__}")
        if s in _CONSOLE_ONLY_SCOPES:
            raise ApiError(
                "invalid_scope",
                f"scope {s!r} is console-only and cannot be assigned to API keys.",
                details={"console_only_scope": s},
            )
        if s not in _USER_ASSIGNABLE_SCOPES:
            raise ApiError(
                "invalid_scope",
                f"unknown scope {s!r}. See /docs#scopes for the catalogue.",
                details={"unknown_scope": s},
            )
        if s in seen:
            continue   # dedup silently
        seen.add(s)
        out.append(s)
    return out


def _validate_quota(value: Any, *, field: str) -> Optional[int]:
    """Spec §10.6.1 — quota_per_day / rate_limit_per_min validation.

    Phase 0 spec mandated bool-guard (Python bool ⊂ int) + range check.
    """
    if value is None:
        return None
    # bool guard (bool ⊂ int)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(
            "invalid_quota",
            f"{field} must be an integer in (0, 1_000_000].",
            details={"field": field, "received": repr(value)},
        )
    if value <= 0 or value > 1_000_000:
        raise ApiError(
            "invalid_quota",
            f"{field} must be in (0, 1_000_000].",
            details={"field": field, "received": value},
        )
    return value


# PDA-P2-R1-F07 fix: charset-restrict ip_allowlist entries to defeat
# stored-XSS via the Console UI. Full CIDR parsing is Phase 11; for now,
# accept only chars that COULD appear in a valid CIDR / bare IP:
#   - hex digits (IPv6)
#   - decimal digits (both)
#   - `.` (IPv4 separator)
#   - `:` (IPv6 separator)
#   - `/` (CIDR mask separator)
#   - `[` `]` (RFC 3986 bracketed IPv6 — uncommon but legal in some forms)
# Length cap: 45 chars covers the longest legal IPv6 + /128 form.
_IP_ALLOWLIST_ENTRY_RE = re.compile(r"^[0-9a-fA-F\.:/\[\]]+$")
_IP_ALLOWLIST_MAX_LEN = 45


def _validate_ip_allowlist(value: Any) -> Optional[list]:
    """Charset-restricted format check. Full CIDR parsing in Phase 11.

    PDA-P2-R1-F07: rejects payloads like `<script>alert(1)</script>`,
    Unicode characters, and anything outside the IP/CIDR character class.
    Phase 11's ipaddress-module parser will do semantic validation.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ApiError("invalid_ip_cidr",
                       "ip_allowlist must be an array of strings.")
    for entry in value:
        if not isinstance(entry, str):
            raise ApiError("invalid_ip_cidr",
                           "ip_allowlist entries must be strings.")
        stripped = entry.strip()
        if not stripped:
            raise ApiError("invalid_ip_cidr",
                           "ip_allowlist entries must be non-empty after strip.")
        if len(stripped) > _IP_ALLOWLIST_MAX_LEN:
            raise ApiError(
                "invalid_ip_cidr",
                f"ip_allowlist entry too long (max {_IP_ALLOWLIST_MAX_LEN} chars).",
                details={"received_length": len(stripped)},
            )
        if not _IP_ALLOWLIST_ENTRY_RE.match(stripped):
            raise ApiError(
                "invalid_ip_cidr",
                "ip_allowlist entry contains characters outside the IP/CIDR "
                "charset [0-9a-fA-F.:/[]]. Phase-11 CIDR parsing will do "
                "stricter validation; this check defends against stored-XSS.",
            )
    return [e.strip() for e in value]


def _validate_origin_allowlist(value: Any) -> Optional[list]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ApiError("invalid_origin",
                       "origin_allowlist must be an array of strings.")
    for entry in value:
        if not isinstance(entry, str):
            raise ApiError("invalid_origin",
                           "origin_allowlist entries must be strings.")
        # PDA-P2-R1-F06 fix: reject literal "null" Origin case-insensitively
        # and after whitespace strip. Previously only the exact lowercase
        # string "null" was rejected, so " null ", "NULL", "Null" all
        # passed through. The browser Origin spec is case-sensitive on the
        # token but tolerant of leading/trailing whitespace from operator
        # error — defensive reject either form.
        if entry.strip().lower() == "null":
            raise ApiError("invalid_origin",
                           "literal 'null' Origin is forbidden (PDA-F08).")
        if _CONTROL_CHARS_RE.search(entry):
            raise ApiError("invalid_origin",
                           "origin_allowlist entry contains control chars.")
    return [e.strip() for e in value]


def _validate_expires_at(value: Any) -> Optional[str]:
    """Must be a future ISO-8601 timestamp, or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError("invalid_parameter", "expires_at must be a string ISO-8601 timestamp.")
    from datetime import datetime, timezone
    try:
        s = value
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt <= datetime.now(timezone.utc):
            raise ApiError("invalid_parameter",
                           "expires_at must be in the future.",
                           details={"field": "expires_at"})
    except ApiError:
        raise
    except Exception:
        raise ApiError("invalid_parameter",
                       "expires_at is not a valid ISO-8601 timestamp.",
                       details={"field": "expires_at"})
    return value


# ── Audit emission stub (Phase 4) ─────────────────────────────────────────

def _audit_log_stub(*, action: str, user_id: int, key_id: Optional[str],
                   before: Optional[dict] = None,
                   after: Optional[dict] = None) -> None:
    """Phase 8 Wave C (WI-P8-AUDITREC-SLOT5): no longer a stub. Writes a
    hash-chained row to api_key_audit. Best-effort: if the write raises
    for any reason (DB locked, disk full, etc.), we log a WARNING but the
    request continues — losing one audit row should not break the user's
    workflow. Operators should monitor the WARNING and treat it as a
    correctness incident.

    The function name is kept (`_audit_log_stub`) to avoid churn at all 6
    call sites in this module. Despite the name, it now actually emits.
    """
    try:
        from scripts.apin_v2 import auth_db as _adb
        _adb.append_audit_log(
            user_id=int(user_id),
            action=action,
            key_id=key_id,
            after=after,
            before=before,
            key_name_at_time=(after.get("name", "") if after else ""),
        )
    except Exception as e:
        log.warning(
            "AUDIT-EMIT FAILED action=%r user_id=%d key_id=%r — %s: %s",
            action, user_id, key_id, type(e).__name__, e,
        )


# ── Route handlers ────────────────────────────────────────────────────────

@router.get("")
@api_endpoint("/api/account/keys")
async def list_keys(
    request: Request,
    # PVA-P2-R1 note: migrate `regex=` to `pattern=` (FastAPI deprecation).
    env: str = Query("all", pattern="^(live|test|all)$"),
    # PDA-P2-R1-F08: status now pattern-restricted (was arbitrary string).
    status: str = Query("all",
                        pattern="^(active|rotating|disabled|expired|compromised|all)$"),
    search: Optional[str] = Query(None, max_length=80),
    cursor: Optional[int] = Query(None, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """GET /api/account/keys — paginated list of the caller's keys."""
    user = _get_session_user(request)
    result = auth_db.list_console_api_keys(
        user_id=int(user["id"]),
        env=env,
        status=status,
        search=search,
        cursor=cursor,
        limit=limit,
    )
    return result


@router.post("", status_code=201)
@api_endpoint("/api/account/keys", success_status=201)
async def create_key(request: Request):
    """POST /api/account/keys — mint a new key with one-time-view payload."""
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")

    name = _validate_name(body.get("name"))
    environment = _validate_environment(body.get("environment"))
    scopes = _validate_scopes(body.get("scopes"))
    ip_allowlist = _validate_ip_allowlist(body.get("ip_allowlist"))
    origin_allowlist = _validate_origin_allowlist(body.get("origin_allowlist"))
    rate_limit_per_min = _validate_quota(
        body.get("rate_limit_per_min"), field="rate_limit_per_min")
    quota_per_day = _validate_quota(
        body.get("quota_per_day"), field="quota_per_day")
    expires_at = _validate_expires_at(body.get("expires_at"))
    note = body.get("note")
    if note is not None and not isinstance(note, str):
        raise ApiError("invalid_parameter", "note must be a string.")

    try:
        new_key = auth_db.create_console_api_key(
            user_id=int(user["id"]),
            name=name,
            environment=environment,
            scopes=scopes,
            ip_allowlist=ip_allowlist,
            origin_allowlist=origin_allowlist,
            rate_limit_per_min=rate_limit_per_min,
            quota_per_day=quota_per_day,
            expires_at=expires_at,
            note=note,
            created_ip=request.client.host if request.client else None,
            created_ua=(request.headers.get("user-agent") or "")[:1024],
        )
    except auth_db.DuplicateKeyNameError as e:
        raise ApiError("duplicate_name", str(e),
                       details={"field": "name"}) from e

    _audit_log_stub(action="created", user_id=int(user["id"]),
                    key_id=new_key["public_id"], after={"name": name})
    # Phase 8.H · emit alert (gated by user prefs · default ON).
    auth_db.emit_alert(
        int(user["id"]), "key.created",
        key_id=new_key["public_id"],
        action={"kind": "view_key", "public_id": new_key["public_id"]},
        key_name=name, environment=environment,
    )

    # Envelope wraps this in `data`. The one-time-view client must
    # extract plaintext_token from data.plaintext_token and IMMEDIATELY
    # clear it from the response object (PDA-F10 — see §4.4).
    #
    # PDA-P2-R1-F01 fix: the response body contains a literal token
    # (plaintext_token) that MUST NOT be stripped by TokenRedactionMiddleware,
    # or the one-time-view ceremony would never deliver the token to the user.
    # Set the `X-APIN-No-Redact: 1` opt-out header — the middleware honours
    # this and strips the header before forwarding to the client.
    return (new_key, {
        "warnings": [{
            "code": "token_one_time_display",
            "message": "Save this token now. It will not be shown again.",
        }],
    }, {"X-APIN-No-Redact": "1"})


@router.get("/{public_id}")
@api_endpoint("/api/account/keys/{public_id}")
async def get_key(request: Request, public_id: str):
    """GET /api/account/keys/{public_id} — fetch one key."""
    user = _get_session_user(request)
    key = auth_db.get_console_api_key(
        user_id=int(user["id"]), public_id=public_id)
    if key is None:
        raise ApiError("not_found", f"key {public_id!r} not found.")
    return key


@router.patch("/{public_id}")
@api_endpoint("/api/account/keys/{public_id}")
async def patch_key(request: Request, public_id: str):
    """PATCH /api/account/keys/{public_id} — edit a subset of fields."""
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")

    # Build a validated update dict
    updates: dict = {}
    if "name" in body:
        updates["name"] = _validate_name(body["name"])
    if "scopes" in body:
        updates["scopes"] = _validate_scopes(body["scopes"])
    if "ip_allowlist" in body:
        updates["ip_allowlist"] = _validate_ip_allowlist(body["ip_allowlist"])
    if "origin_allowlist" in body:
        updates["origin_allowlist"] = _validate_origin_allowlist(body["origin_allowlist"])
    if "rate_limit_per_min" in body:
        updates["rate_limit_per_min"] = _validate_quota(
            body["rate_limit_per_min"], field="rate_limit_per_min")
    if "quota_per_day" in body:
        updates["quota_per_day"] = _validate_quota(
            body["quota_per_day"], field="quota_per_day")
    if "expires_at" in body:
        updates["expires_at"] = _validate_expires_at(body["expires_at"])
    if "note" in body:
        if body["note"] is not None and not isinstance(body["note"], str):
            raise ApiError("invalid_parameter", "note must be a string.")
        updates["note"] = body["note"]

    if not updates:
        raise ApiError(
            "invalid_parameter",
            "Request body has no editable fields. Supply at least one of: "
            "name, scopes, ip_allowlist, origin_allowlist, rate_limit_per_min, "
            "quota_per_day, expires_at, note.",
        )

    try:
        updated = auth_db.patch_console_api_key(
            user_id=int(user["id"]), public_id=public_id, **updates)
    except auth_db.DuplicateKeyNameError as e:
        raise ApiError("duplicate_name", str(e),
                       details={"field": "name"}) from e

    if updated is None:
        raise ApiError(
            "not_found",
            f"key {public_id!r} not found or not editable (status must be "
            f"'active' or 'rotating').",
        )

    _audit_log_stub(action="patched", user_id=int(user["id"]),
                    key_id=public_id, after=updates)
    # Phase 8.H · key.patched is default-OFF (chatty for power users).
    auth_db.emit_alert(
        int(user["id"]), "key.patched",
        key_id=public_id,
        action={"kind": "view_key", "public_id": public_id},
        key_name=(updated.get("name") if isinstance(updated, dict) else public_id),
        fields_changed=", ".join(sorted(updates.keys())) if updates else "metadata",
    )
    return updated


@router.post("/{public_id}/rotate", status_code=201)
@api_endpoint("/api/account/keys/{public_id}/rotate", success_status=201)
async def rotate_key(request: Request, public_id: str):
    """POST /api/account/keys/{public_id}/rotate — rotate to a new plaintext.

    FX-P5-1 (PDA-P5-R1 F01 + PVA-P5-R1 NOTE A): explicitly returns 201
    matching the mint endpoint. Both produce a one-time-view of a fresh
    plaintext_token; both warrant the same "created new credential
    representation" status. Spec §7.1 PDA-R3-F42 pins mint to 201; spec
    is silent on rotate (VER-P5-R1 confirms 200 was also conforming);
    we pick 201 for client/server consistency and to avoid the
    `r.status === 201` mismatch in the wizard JS.
    """
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    grace = body.get("grace_seconds", 172_800)
    if isinstance(grace, bool) or not isinstance(grace, int) \
            or grace < 0 or grace > 30 * 24 * 3600:
        raise ApiError(
            "invalid_parameter",
            "grace_seconds must be an integer in [0, 30 days].",
            details={"field": "grace_seconds", "received": repr(grace)},
        )

    try:
        new_key = auth_db.rotate_console_api_key(
            user_id=int(user["id"]),
            public_id=public_id,
            grace_seconds=int(grace),
        )
    except auth_db.KeyAlreadyRotatingError as e:
        raise ApiError("already_rotating", str(e)) from e
    except auth_db.InvalidKeyStateError as e:
        raise ApiError("not_found", str(e)) from e
    except auth_db.DuplicateKeyNameError as e:
        # PDA-P2-R1-F04 fix: the inner create_console_api_key call uses
        # `name + " (rotated)"` as the new key name to avoid UNIQUE
        # collision with the predecessor. If the user already has another
        # key with that exact rotated name (e.g. they previously rotated
        # AND then re-named the new one to free the original), surface
        # 409 duplicate_name instead of letting it bubble to 500.
        raise ApiError(
            "duplicate_name",
            "Cannot rotate: another key already uses the rotation name. "
            "Rename the existing rotated key first.",
            details={"field": "name"},
        ) from e

    if new_key is None:
        raise ApiError("not_found", f"key {public_id!r} not found.")

    _audit_log_stub(action="rotated", user_id=int(user["id"]),
                    key_id=public_id,
                    after={"new_public_id": new_key["public_id"]})
    # Phase 8.H · default-ON, info severity.
    auth_db.emit_alert(
        int(user["id"]), "key.rotated",
        key_id=new_key["public_id"],
        action={"kind": "view_key", "public_id": new_key["public_id"]},
        key_name=new_key.get("name") or public_id,
    )

    # PDA-P2-R1-F01 fix: same opt-out as POST /keys — rotate's plaintext
    # is a one-time-view payload that must not be redacted.
    return (new_key, {
        "warnings": [{
            "code": "token_one_time_display",
            "message": "Save this rotated token now. It will not be shown again.",
        }],
    }, {"X-APIN-No-Redact": "1"})


@router.get("/{public_id}/usage")
@api_endpoint("/api/account/keys/{public_id}/usage")
async def get_key_usage(request: Request, public_id: str,
                         minutes: int = Query(60, ge=1, le=1440)):
    """Phase 8 Wave D: per-minute usage rollup for the detail page.
    Returns oldest-first so the client can render a left-to-right sparkline
    with no resorting."""
    user = _get_session_user(request)
    items = auth_db.list_key_usage_minute(
        user_id=int(user["id"]), public_id=public_id, minutes=minutes)
    return {"items": items, "count": len(items),
            "window_minutes": minutes, "public_id": public_id}


# ════════════════════════════════════════════════════════════════════════
# 9.N.9 · Per-key OVERVIEW — single endpoint feeding all 6 bento widgets.
#
# One round-trip returns: health score (4-pillar), KPIs (with prev-period
# deltas), request ribbon (last 120), spark-grid (top-6 endpoints), key
# personality (derived behaviour tags), and narrated insights. The live
# ribbon updates separately via SSE; this is the initial + periodic snapshot.
# ════════════════════════════════════════════════════════════════════════

_WINDOW_MINUTES = {"1h": 60, "24h": 1440, "7d": 10080}


def _build_key_overview(user_id: int, public_id: str, key: dict,
                        window: str) -> dict:
    """Gather all overview data in one DB pass + compute the health score.
    Runs inside asyncio.to_thread (blocking DB I/O)."""
    import time as _t
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    from collections import Counter as _Counter
    from scripts.apin_v2 import key_health as _kh

    win_min = _WINDOW_MINUTES.get(window, 1440)
    now = _dt.now(_tz.utc)
    cut_cur  = (now - _td(minutes=win_min)).strftime("%Y-%m-%d %H:%M:%S.%f")
    cut_prev = (now - _td(minutes=win_min * 2)).strftime("%Y-%m-%d %H:%M:%S.%f")

    out = {"public_id": public_id, "window": window,
           "key": {"name": key.get("name"), "environment": key.get("environment"),
                   "status": key.get("status"), "token_prefix": key.get("token_prefix"),
                   "created_at": key.get("created_at"), "expires_at": key.get("expires_at"),
                   "scopes": key.get("scopes"), "last_used_at": key.get("last_used_at")}}

    with auth_db.get_conn() as c:
        def _rows(sql, args=()):
            try:
                return [dict(r) for r in c.execute(sql, args).fetchall()]
            except Exception as e:
                log.warning("overview query failed: %s", e)
                return []

        # ── Pull window rows (status, latency, path, method, ip) ──────────
        cur_rows = _rows(
            "SELECT timestamp, method, path, status_code, latency_ms, ip, "
            "       bytes_in, bytes_out, error_code "
            "FROM api_key_request_log WHERE key_id = ? AND timestamp >= ? "
            "ORDER BY id", (public_id, cut_cur))
        prev_rows = _rows(
            "SELECT status_code, latency_ms FROM api_key_request_log "
            "WHERE key_id = ? AND timestamp >= ? AND timestamp < ?",
            (public_id, cut_prev, cut_cur))

        # ── Status buckets ───────────────────────────────────────────────
        def _bucket(rows):
            n2 = n4 = n5 = 0
            for r in rows:
                s = int(r.get("status_code") or 0)
                if 200 <= s < 400: n2 += 1
                elif 400 <= s < 500: n4 += 1
                elif s >= 500: n5 += 1
            return n2, n4, n5
        n2, n4, n5 = _bucket(cur_rows)
        total = len(cur_rows)
        p_n2, p_n4, p_n5 = _bucket(prev_rows)
        prev_total = len(prev_rows)

        # ── Latencies by endpoint class (for Apdex) ──────────────────────
        # COLD-START GRACE: the first inference request after an idle gap
        # > 5 min pays the model-backbone lazy-load tax (the container is
        # kept warm by the uptime probes, but the model unloads). That
        # cold latency is infrastructure, not the key's behaviour, so we
        # exclude it from the Apdex calc — the score reflects WARM
        # performance, which is what a user on always-warm infra sees.
        # Cold requests are still counted in traffic/ribbon, just not Apdex.
        COLD_GAP_S = 300
        lat_by_class: dict = {}
        all_lat = []
        methods = set()
        ips = set()
        cold_start_excluded = 0
        _last_ts = None
        for r in cur_rows:
            cls = _kh.classify_endpoint(r.get("path") or "")
            lm = r.get("latency_ms")
            # parse ts for gap detection
            cur_ts = None
            try:
                cur_ts = _dt.fromisoformat(str(r["timestamp"]).replace(" ", "T")).timestamp()
            except Exception:
                pass
            is_cold = False
            if (cur_ts is not None and _last_ts is not None
                    and (cur_ts - _last_ts) > COLD_GAP_S
                    and cls in ("quick_inference", "heavy_inference")):
                is_cold = True
            if cur_ts is not None:
                _last_ts = cur_ts
            if lm is not None:
                if is_cold:
                    cold_start_excluded += 1   # skip from Apdex
                else:
                    lat_by_class.setdefault(cls, []).append(float(lm))
                all_lat.append(float(lm))      # still counts in overall p50/p95
            if r.get("method"): methods.add(str(r["method"]).upper())
            if r.get("ip"): ips.add(r["ip"])

        def _pct(vals, q):
            if not vals: return None
            s = sorted(vals)
            return s[min(len(s) - 1, int(len(s) * q))]
        p50 = _pct(all_lat, 0.50)
        p95 = _pct(all_lat, 0.95)
        prev_lat = [float(r["latency_ms"]) for r in prev_rows if r.get("latency_ms") is not None]
        p50_prev = _pct(prev_lat, 0.50)
        p95_prev = _pct(prev_lat, 0.95)

        # ── usage_minute: rate-limited count in window ───────────────────
        um = _rows(
            "SELECT SUM(rate_limited) AS rl, SUM(quota_blocked) AS qb "
            "FROM api_key_usage_minute WHERE key_id = ? AND minute_ts >= ?",
            (public_id, cut_cur[:16]))  # minute_ts is 'YYYY-MM-DD HH:MM'
        rate_limited = int((um[0].get("rl") if um else 0) or 0)

        # ── IP baseline (distinct IPs in the prev window) ────────────────
        prev_ip_rows = _rows(
            "SELECT DISTINCT ip FROM api_key_request_log "
            "WHERE key_id = ? AND timestamp >= ? AND timestamp < ? AND ip IS NOT NULL",
            (public_id, cut_prev, cut_cur))
        ip_baseline = float(len(prev_ip_rows))

        # ── HEALTH SCORE ─────────────────────────────────────────────────
        quota = key.get("quota_per_day")
        health = _kh.compute_health_score(
            window_label=window, total_requests=total,
            n_2xx=n2, n_4xx=n4, n_5xx=n5,
            latencies_by_class=lat_by_class,
            p95_current=p95, p95_prev=p95_prev,
            rate_limited=rate_limited,
            quota_per_day=int(quota) if quota else None,
            quota_consumed=total,  # approximation: requests today
            created_at=key.get("created_at"), expires_at=key.get("expires_at"),
            scopes=key.get("scopes"), observed_methods=methods,
            distinct_ips=len(ips), ip_baseline=ip_baseline,
        )
        health["cold_start_excluded"] = cold_start_excluded
        out["health"] = health

        # ── KPIs with prev-period deltas ─────────────────────────────────
        def _delta(cur, prev):
            if prev in (None, 0): return None
            return round((cur - prev) / prev * 100, 1)
        succ = round(100.0 * n2 / total, 1) if total else None
        prev_succ = round(100.0 * p_n2 / prev_total, 1) if prev_total else None
        out["kpis"] = {
            "requests":    {"value": total, "prev": prev_total, "delta_pct": _delta(total, prev_total)},
            "success_rate":{"value": succ, "prev": prev_succ,
                            "delta_pct": (round(succ - prev_succ, 1) if (succ is not None and prev_succ is not None) else None)},
            "p50_ms":      {"value": p50, "prev": p50_prev, "delta_pct": _delta(p50, p50_prev)},
            "rate_limited":{"value": rate_limited, "prev": None, "delta_pct": None},
        }

        # ── Request ribbon: last 120 (newest last for left→right flow) ───
        ribbon = _rows(
            "SELECT id, timestamp, method, path, status_code, latency_ms "
            "FROM api_key_request_log WHERE key_id = ? "
            "ORDER BY id DESC LIMIT 120", (public_id,))
        out["ribbon"] = list(reversed(ribbon))

        # ── Spark-grid: top-6 endpoints, per-bucket counts ──────────────
        path_counter = _Counter(r.get("path") for r in cur_rows if r.get("path"))
        top6 = [p for p, _ in path_counter.most_common(6)]
        n_buckets = 24
        bucket_ms = (win_min * 60_000) / n_buckets
        t0 = (now - _td(minutes=win_min)).timestamp() * 1000
        spark = []
        for p in top6:
            buckets = [0] * n_buckets              # request count per bucket
            blat_sum = [0.0] * n_buckets           # latency sum per bucket
            blat_n = [0] * n_buckets               # latency count per bucket
            berr = [0] * n_buckets                 # error count per bucket
            lats = []
            for r in cur_rows:
                if r.get("path") != p: continue
                ts = _dt.fromisoformat(str(r["timestamp"]).replace(" ", "T")).timestamp() * 1000 \
                     if r.get("timestamp") else None
                bi = None
                if ts is not None:
                    bi = min(n_buckets - 1, max(0, int((ts - t0) / bucket_ms)))
                    buckets[bi] += 1
                    if int(r.get("status_code") or 0) >= 400:
                        berr[bi] += 1
                lm = r.get("latency_ms")
                if lm is not None:
                    lats.append(float(lm))
                    if bi is not None:
                        blat_sum[bi] += float(lm); blat_n[bi] += 1
            # 9.N.9(d) · per-bucket avg latency for the metric toggle
            buckets_lat = [round(blat_sum[i] / blat_n[i]) if blat_n[i] else 0
                           for i in range(n_buckets)]
            spark.append({"path": p, "count": path_counter[p],
                          "buckets": buckets, "buckets_lat": buckets_lat,
                          "buckets_err": berr, "p95": _pct(lats, 0.95)})
        out["spark_grid"] = spark

        # ── Personality (derived behaviour tags) ─────────────────────────
        predict_n = sum(1 for r in cur_rows if (r.get("path") or "").startswith("/api/predict"))
        get_n = sum(1 for r in cur_rows if (r.get("method") or "").upper() == "GET")
        # burstiness: coefficient of variation of inter-request gaps
        ts_list = []
        for r in cur_rows:
            try:
                ts_list.append(_dt.fromisoformat(str(r["timestamp"]).replace(" ", "T")).timestamp())
            except Exception:
                pass
        gaps = [b - a for a, b in zip(ts_list, ts_list[1:])] if len(ts_list) > 2 else []
        burst = 0.0
        if gaps:
            mean_g = sum(gaps) / len(gaps)
            if mean_g > 0:
                var = sum((g - mean_g) ** 2 for g in gaps) / len(gaps)
                cv = (var ** 0.5) / mean_g
                burst = min(1.0, cv / 3.0)   # CV of 3+ = maximally bursty
        out["personality"] = {
            "tags": [
                {"name": "predict-heavy", "value": round(predict_n / total, 2) if total else 0,
                 "signal": f"{round(100*predict_n/total) if total else 0}% of calls hit /predict/*"},
                {"name": "bursty", "value": round(burst, 2),
                 "signal": "inter-request timing variance (coefficient of variation)"},
                {"name": "read-mostly", "value": round(get_n / total, 2) if total else 0,
                 "signal": f"{round(100*get_n/total) if total else 0}% GET requests"},
            ]
        }

        # ── Insights (narrated, key-scoped) ──────────────────────────────
        insights = []
        if total == 0:
            insights.append({"tone": "info", "text": "No traffic in this window yet."})
        else:
            if n5 == 0 and n4 == 0:
                insights.append({"tone": "great", "text": f"No errors in the last {window}."})
            if health.get("composite") and health["composite"] >= 90:
                insights.append({"tone": "great", "text": f"Healthy — grade {health['grade']}."})
            pp = health["pillars"]["performance"]
            if pp.get("trend_pct") is not None and pp["trend_pct"] < -3:
                insights.append({"tone": "great",
                                 "text": f"p95 latency improved {abs(pp['trend_pct'])}% vs previous {window}."})
            elif pp.get("trend_pct") is not None and pp["trend_pct"] > 5:
                insights.append({"tone": "warn",
                                 "text": f"p95 latency degraded {pp['trend_pct']}% vs previous {window}."})
            if predict_n and total and predict_n / total > 0.5:
                insights.append({"tone": "info",
                                 "text": f"Predict-heavy key — {round(100*predict_n/total)}% of calls are inference."})
            if len(ips) == 1:
                insights.append({"tone": "info",
                                 "text": "All traffic from a single IP — likely one integration."})
            elif len(ips) > 8 and (ip_baseline <= 0 or len(ips) > ip_baseline * 4):
                insights.append({"tone": "warn",
                                 "text": f"IP fan-out: {len(ips)} distinct IPs (baseline ~{ip_baseline:.0f}) — verify the key isn't shared."})
        out["insights"] = insights[:6]

    return out


@router.get("/{public_id}/overview")
@api_endpoint("/api/account/keys/{public_id}/overview")
async def get_key_overview(request: Request, public_id: str,
                           window: str = Query("24h")):
    """9.N.9 · One-shot overview payload for the bento Overview tab."""
    import asyncio as _aio
    user = _get_session_user(request)
    key = auth_db.get_console_api_key(user_id=int(user["id"]), public_id=public_id)
    if key is None:
        raise ApiError("not_found", f"key {public_id!r} not found.")
    if window not in _WINDOW_MINUTES:
        window = "24h"
    data = await _aio.to_thread(_build_key_overview, int(user["id"]), public_id, key, window)
    return data


@router.get("/{public_id}/requests")
@api_endpoint("/api/account/keys/{public_id}/requests")
async def get_key_requests(request: Request, public_id: str,
                            limit: int = Query(50, ge=1, le=200),
                            cursor: Optional[int] = Query(None, ge=1)):
    """Phase 8 Wave D: per-request log for the detail page. Newest first."""
    user = _get_session_user(request)
    items = auth_db.list_key_requests(
        user_id=int(user["id"]), public_id=public_id,
        limit=limit, cursor=cursor)
    return {"items": items, "count": len(items), "public_id": public_id}


@router.get("/{public_id}/audit")
@api_endpoint("/api/account/keys/{public_id}/audit")
async def get_key_audit(request: Request, public_id: str,
                         limit: int = Query(100, ge=1, le=500),
                         cursor: Optional[int] = Query(None, ge=1)):
    """Phase 8 Wave D: per-key hash-chained audit-log timeline."""
    user = _get_session_user(request)
    # ownership check is implicit: list_audit_log filters by user_id, and
    # the key_id filter is scoped to this user's audit rows only.
    if auth_db.get_console_api_key(
            user_id=int(user["id"]), public_id=public_id) is None:
        raise ApiError("not_found", f"key {public_id!r} not found.")
    items = auth_db.list_audit_log(
        user_id=int(user["id"]), key_id=public_id,
        limit=limit, cursor=cursor)
    return {"items": items, "count": len(items), "public_id": public_id}


@router.post("/{public_id}/disable")
@api_endpoint("/api/account/keys/{public_id}/disable")
async def disable_key(request: Request, public_id: str):
    """POST /api/account/keys/{public_id}/disable — Phase 8 WI-P8-KEY-DISABLE-VERB.

    Idempotent. Moves the key from 'active' to 'disabled'. Disabled keys
    cannot authenticate. Use POST .../enable to reverse. Sudo + CSRF.

    Why a dedicated verb and not PATCH {status:...}: spec §7.1 wants the
    state machine for keys (active → rotating → disabled → deleted) to
    be controlled by explicit RPC verbs, not by sneaking a `status`
    field through the generic PATCH. Makes audit log clean and lets us
    add per-transition side-effects later (alert emission, webhook firing).
    """
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        updated = auth_db.disable_console_api_key(
            user_id=int(user["id"]), public_id=public_id)
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    if updated is None:
        raise ApiError("not_found", f"key {public_id!r} not found.")
    _audit_log_stub(action="disabled", user_id=int(user["id"]),
                    key_id=public_id)
    # Phase 8.H · default-ON.
    auth_db.emit_alert(
        int(user["id"]), "key.disabled",
        key_id=public_id,
        action={"kind": "view_key", "public_id": public_id},
        key_name=(updated.get("name") if isinstance(updated, dict) else public_id),
    )
    return updated


@router.post("/{public_id}/enable")
@api_endpoint("/api/account/keys/{public_id}/enable")
async def enable_key(request: Request, public_id: str):
    """POST /api/account/keys/{public_id}/enable — reverse of disable.
    Only valid for keys currently in 'disabled' status. Sudo + CSRF."""
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        updated = auth_db.enable_console_api_key(
            user_id=int(user["id"]), public_id=public_id)
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    if updated is None:
        raise ApiError("not_found", f"key {public_id!r} not found.")
    _audit_log_stub(action="enabled", user_id=int(user["id"]),
                    key_id=public_id)
    # Phase 8.H · default-ON.
    auth_db.emit_alert(
        int(user["id"]), "key.enabled",
        key_id=public_id,
        action={"kind": "view_key", "public_id": public_id},
        key_name=(updated.get("name") if isinstance(updated, dict) else public_id),
    )
    return updated


@router.delete("/{public_id}")
@api_endpoint("/api/account/keys/{public_id}")
async def delete_key(request: Request, public_id: str):
    """DELETE /api/account/keys/{public_id} — hard-delete (must be disabled first)."""
    _require_csrf(request)
    user = _get_session_user(request)
    ok = auth_db.hard_delete_console_api_key(
        user_id=int(user["id"]), public_id=public_id)
    if not ok:
        raise ApiError(
            "not_found",
            f"key {public_id!r} not found, already deleted, or not in a "
            f"deletable status (must be 'disabled' or 'expired').",
        )

    _audit_log_stub(action="hard_deleted", user_id=int(user["id"]),
                    key_id=public_id)
    # Phase 8.H · default-ON. No view_key action — the key is gone; the
    # alert links back to the keys list instead.
    auth_db.emit_alert(
        int(user["id"]), "key.deleted",
        key_id=None,
        action={"kind": "view_keys_list"},
        key_name=public_id,
    )
    return {"deleted": True, "public_id": public_id}
