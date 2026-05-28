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

# Personality axes (order is the radar spoke order).
_PERSONALITY_DIMS = ["predict_heavy", "read_mostly", "write_heavy",
                     "bursty", "error_tolerant", "steady"]


def _personality_vector(rows: list) -> dict:
    """Derive a 6-dim behavioural vector in [0,1] from request-log rows.
    Pure function reused for this key, the account average, and each peer
    key (for cosine-similarity 'similar keys'). Rows must carry path, method,
    status_code, timestamp; rows are assumed chronological (query ORDER BY id).
    """
    from datetime import datetime as _dt
    total = len(rows)
    if not total:
        return {k: 0.0 for k in _PERSONALITY_DIMS}
    parsed = []
    for r in rows:
        try:
            parsed.append(_dt.fromisoformat(str(r["timestamp"]).replace(" ", "T")).timestamp())
        except Exception:
            parsed.append(None)
    predict_n = sum(1 for r in rows if (r.get("path") or "").startswith("/api/predict"))
    get_n = sum(1 for r in rows if (r.get("method") or "").upper() == "GET")
    # bursty — coefficient of variation of inter-request gaps
    tser = [t for t in parsed if t is not None]
    gaps = [b - a for a, b in zip(tser, tser[1:])] if len(tser) > 2 else []
    bursty = 0.0
    if gaps:
        mg = sum(gaps) / len(gaps)
        if mg > 0:
            var = sum((g - mg) ** 2 for g in gaps) / len(gaps)
            bursty = min(1.0, ((var ** 0.5) / mg) / 3.0)
    # error_tolerant — share of 4xx that were followed by another request <=120s
    four = followed = 0
    for i, r in enumerate(rows):
        if 400 <= int(r.get("status_code") or 0) < 500:
            four += 1
            ti = parsed[i]
            if ti is not None:
                for j in range(i + 1, min(i + 60, total)):
                    tj = parsed[j]
                    if tj is not None and 0 <= tj - ti <= 120:
                        followed += 1
                        break
    error_tolerant = (followed / four) if four else 0.0
    # steady — 1 − CV of per-bucket request counts across the observed span
    steady = 0.0
    if len(tser) > 2:
        lo, hi = tser[0], tser[-1]
        span = hi - lo
        if span > 0:
            nb = 24
            counts = [0] * nb
            for t in tser:
                counts[min(nb - 1, int((t - lo) / span * nb))] += 1
            mc = sum(counts) / nb
            if mc > 0:
                v = sum((c - mc) ** 2 for c in counts) / nb
                steady = max(0.0, 1.0 - min(1.0, (v ** 0.5) / mc))
    return {
        "predict_heavy": predict_n / total,
        "read_mostly": get_n / total,
        "write_heavy": (total - get_n) / total,
        "bursty": bursty,
        "error_tolerant": error_tolerant,
        "steady": steady,
    }


def _cosine(a: dict, b: dict) -> float:
    """Cosine similarity of two personality vectors over _PERSONALITY_DIMS."""
    import math
    dot = sum(a[k] * b[k] for k in _PERSONALITY_DIMS)
    na = math.sqrt(sum(a[k] ** 2 for k in _PERSONALITY_DIMS))
    nb = math.sqrt(sum(b[k] ** 2 for k in _PERSONALITY_DIMS))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _build_key_overview(user_id: int, public_id: str, window: str) -> dict:
    """Gather all overview data in one DB pass + compute the health score.
    Runs inside asyncio.to_thread (blocking DB I/O). The key row is fetched
    inside the same batched read (no separate ownership round-trip); a missing
    key returns {"not_found": True} for the route to translate to a 404."""
    import time as _t
    import json as _json
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    from collections import Counter as _Counter
    from scripts.apin_v2 import key_health as _kh

    win_min = _WINDOW_MINUTES.get(window, 1440)
    now = _dt.now(_tz.utc)
    cut_cur  = (now - _td(minutes=win_min)).strftime("%Y-%m-%d %H:%M:%S.%f")
    cut_prev = (now - _td(minutes=win_min * 2)).strftime("%Y-%m-%d %H:%M:%S.%f")

    out = {"public_id": public_id, "window": window}

    with auth_db.get_conn() as c:
        def _rows(sql, args=()):
            try:
                return [dict(r) for r in c.execute(sql, args).fetchall()]
            except Exception as e:
                log.warning("overview query failed: %s", e)
                return []

        # ── One pipelined round-trip for every window read. The Turso HTTP
        #    shim bills one remote request per statement, so the previous ~7
        #    sequential reads cost ~5 s; batched they cost ~1 s. Falls back to
        #    sequential for the plain-sqlite backend (no batch_read). ──
        cut_min = cut_cur[:16]   # api_key_usage_minute.minute_ts is 'YYYY-MM-DD HH:MM'
        _READS = [
            ("SELECT timestamp, method, path, status_code, latency_ms, ip, "
             "bytes_in, bytes_out, error_code FROM api_key_request_log "
             "WHERE key_id = ? AND timestamp >= ? ORDER BY id", (public_id, cut_cur)),
            ("SELECT status_code, latency_ms FROM api_key_request_log "
             "WHERE key_id = ? AND timestamp >= ? AND timestamp < ?",
             (public_id, cut_prev, cut_cur)),
            ("SELECT minute_ts, rate_limited FROM api_key_usage_minute "
             "WHERE key_id = ? AND minute_ts >= ? ORDER BY minute_ts", (public_id, cut_min)),
            ("SELECT DISTINCT ip FROM api_key_request_log WHERE key_id = ? "
             "AND timestamp >= ? AND timestamp < ? AND ip IS NOT NULL",
             (public_id, cut_prev, cut_cur)),
            ("SELECT id, timestamp, method, path, status_code, latency_ms "
             "FROM api_key_request_log WHERE key_id = ? ORDER BY id DESC LIMIT 240",
             (public_id,)),
            ("SELECT public_id, name, environment, status, last_four, created_at, "
             "expires_at, scopes, last_used_at, quota_per_day FROM api_keys "
             "WHERE user_id = ? AND deleted_at IS NULL", (user_id,)),
            ("SELECT day, composite FROM api_key_health_snapshot WHERE key_id = ? "
             "ORDER BY day DESC LIMIT 30", (public_id,)),
        ]
        if hasattr(c, "batch_read"):
            try:
                _curs = c.batch_read(_READS)
                _res = [[dict(r) for r in cur.fetchall()] for cur in _curs]
            except Exception as e:
                log.warning("overview batch failed, sequential fallback: %s", e)
                _res = [_rows(sql, args) for sql, args in _READS]
        else:
            _res = [_rows(sql, args) for sql, args in _READS]
        (cur_rows, prev_rows, um_rows, prev_ip_rows,
         ribbon_rows, key_rows, snap_rows) = _res

        # ── Resolve THIS key from key_rows (no separate ownership round-trip).
        #    user_id scoping is in the query, so a missing row == not-this-
        #    user's key → 404. ──
        key = next((r for r in key_rows if r.get("public_id") == public_id), None)
        if key is None:
            return {"not_found": True, "public_id": public_id, "window": window}
        # scopes is a JSON column — parse to a list (health hygiene tolerates
        # both, but the UI + out["key"] want the list form).
        try:
            _scopes = _json.loads(key.get("scopes")) if key.get("scopes") else []
            if not isinstance(_scopes, list):
                _scopes = []
        except Exception:
            _scopes = []
        key["scopes"] = _scopes
        out["key"] = {"name": key.get("name"), "environment": key.get("environment"),
                      "status": key.get("status"), "token_prefix": key.get("last_four"),
                      "created_at": key.get("created_at"), "expires_at": key.get("expires_at"),
                      "scopes": _scopes, "last_used_at": key.get("last_used_at")}
        out["ribbon"] = list(reversed(ribbon_rows))

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

        # ── rate-limit: total + per-minute events timeline (from um_rows) ──
        rate_limited = sum(int(r.get("rate_limited") or 0) for r in um_rows)
        out["rate_limit_events"] = [
            {"minute": r["minute_ts"], "count": int(r["rate_limited"] or 0)}
            for r in um_rows if int(r.get("rate_limited") or 0) > 0][:240]

        # ── IP baseline (distinct IPs in the prev window) ────────────────
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
            # Guard cur is None too: a window with no latency rows (p50=None)
            # while the previous window had some would otherwise crash on the
            # None - float subtraction. Common once traffic ages past the
            # current window but the prior window still has rows.
            if cur is None or prev in (None, 0): return None
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

        # ── Request ribbon (last 240) already fetched in the batch above and
        #    stored in out["ribbon"]; bento shows the most-recent 120, expanded
        #    shows all 240 with a brush-scrubber. ──

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
            bbytes = [0] * n_buckets               # bytes_out sum per bucket
            ep_bytes_total = 0
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
                bo = r.get("bytes_out")
                if bo is not None and bi is not None:
                    bbytes[bi] += int(bo); ep_bytes_total += int(bo)
            # per-bucket avg latency for the metric toggle
            buckets_lat = [round(blat_sum[i] / blat_n[i]) if blat_n[i] else 0
                           for i in range(n_buckets)]
            err_n = sum(1 for r in cur_rows
                        if r.get("path") == p and int(r.get("status_code") or 0) >= 400)
            spark.append({"path": p, "count": path_counter[p],
                          "buckets": buckets, "buckets_lat": buckets_lat,
                          "buckets_err": berr, "buckets_bytes": bbytes,
                          "p50": _pct(lats, 0.50), "p95": _pct(lats, 0.95),
                          "err_count": err_n,
                          "err_pct": round(100.0 * err_n / path_counter[p], 1) if path_counter[p] else 0,
                          "bytes_total": ep_bytes_total})
        out["spark_grid"] = spark

        # ── 9.N.9 · Extra aggregates for the rich KPI + ribbon expands ────
        # status_counts (SUCCESS donut), overall timeseries by status
        # (REQUESTS expand + ribbon density), latency histogram (p50 expand),
        # slowest endpoint.
        ts_req = [0] * n_buckets
        ts_2xx = [0] * n_buckets
        ts_4xx = [0] * n_buckets
        ts_5xx = [0] * n_buckets
        ts_lat = [[] for _ in range(n_buckets)]   # per-bucket latencies (fan)
        for r in cur_rows:
            sc = int(r.get("status_code") or 0)
            tsx = _dt.fromisoformat(str(r["timestamp"]).replace(" ", "T")).timestamp() * 1000 \
                  if r.get("timestamp") else None
            if tsx is None:
                continue
            bi = min(n_buckets - 1, max(0, int((tsx - t0) / bucket_ms)))
            ts_req[bi] += 1
            if 200 <= sc < 400: ts_2xx[bi] += 1
            elif 400 <= sc < 500: ts_4xx[bi] += 1
            elif sc >= 500: ts_5xx[bi] += 1
            lm = r.get("latency_ms")
            if lm is not None:
                ts_lat[bi].append(float(lm))
        # percentile fan: p50/p95/p99 per bucket (0 where the bucket is empty)
        fan_p50 = [round(_pct(b, 0.50) or 0) for b in ts_lat]
        fan_p95 = [round(_pct(b, 0.95) or 0) for b in ts_lat]
        fan_p99 = [round(_pct(b, 0.99) or 0) for b in ts_lat]
        out["status_counts"] = {"n_2xx": n2, "n_4xx": n4, "n_5xx": n5, "total": total}
        out["timeseries"] = {"n_buckets": n_buckets, "bucket_ms": bucket_ms,
                             "t0_ms": t0, "req": ts_req, "s2xx": ts_2xx,
                             "s4xx": ts_4xx, "s5xx": ts_5xx,
                             "lat_p50": fan_p50, "lat_p95": fan_p95, "lat_p99": fan_p99}
        # latency histogram — log-spaced bins from 1ms to 30s
        import math as _math
        hist_edges = [0, 10, 30, 100, 300, 1000, 3000, 8000, 20000, 60000]
        hist = [0] * (len(hist_edges) - 1)
        for lm in all_lat:
            for bi in range(len(hist_edges) - 1):
                if hist_edges[bi] <= lm < hist_edges[bi + 1]:
                    hist[bi] += 1; break
            else:
                if lm >= hist_edges[-1]: hist[-1] += 1
        out["latency_hist"] = {"edges": hist_edges, "bins": hist,
                               "p50": p50, "p95": p95,
                               "p99": _pct(all_lat, 0.99)}
        # slowest endpoint by p95
        slowest = max(spark, key=lambda s: (s.get("p95") or 0), default=None) if spark else None
        out["slowest_endpoint"] = ({"path": slowest["path"], "p95": slowest["p95"]}
                                   if slowest else None)

        # ── Personality (6-dim behavioural vector) ───────────────────────
        pv = _personality_vector(cur_rows)
        predict_n = round(pv["predict_heavy"] * total) if total else 0
        _sig = {
            "predict-heavy":  f"{round(100*pv['predict_heavy'])}% of calls hit /predict/*",
            "bursty":         "inter-request timing variance (coefficient of variation)",
            "read-mostly":    f"{round(100*pv['read_mostly'])}% GET requests",
            "write-heavy":    f"{round(100*pv['write_heavy'])}% non-GET (write) calls",
            "error-tolerant": "share of 4xx retried within 2 min",
            "steady":         "uniformity of request volume across the window",
        }
        _order = [("predict-heavy", "predict_heavy"), ("bursty", "bursty"),
                  ("read-mostly", "read_mostly"), ("write-heavy", "write_heavy"),
                  ("error-tolerant", "error_tolerant"), ("steady", "steady")]
        out["personality"] = {
            "tags": [{"name": n, "value": round(pv[k], 2), "signal": _sig[n]}
                     for n, k in _order],
            "vector": {k: round(pv[k], 3) for k in _PERSONALITY_DIMS},
        }

        # ── Account average + similar keys (cross-key, cosine) ───────────
        # key_rows came from the batched read. For a single-key account the
        # account == this key (reuse pv / cur_rows; no extra cross-key query).
        # Only when there ARE peer keys do we spend one more round-trip to pull
        # their rows for the average + cosine-similarity ranking.
        id_name = {r["public_id"]: r.get("name") for r in key_rows}
        other_ids = [r["public_id"] for r in key_rows if r["public_id"] != public_id]
        acct_vec = dict(pv)
        acct_total = total
        keys_with_traffic = 1 if total else 0
        similar = []
        if other_ids:
            ph = ",".join("?" for _ in other_ids)
            arows = _rows(
                f"SELECT key_id, method, path, status_code, timestamp "
                f"FROM api_key_request_log WHERE key_id IN ({ph}) AND timestamp >= ? "
                f"ORDER BY id", other_ids + [cut_cur])
            by_key: dict = {}
            for r in arows:
                by_key.setdefault(r["key_id"], []).append(r)
            # account aggregate spans THIS key (cur_rows) + every peer
            acct_total = total + len(arows)
            keys_with_traffic = (1 if total else 0) + sum(1 for v in by_key.values() if v)
            combined = list(cur_rows) + arows
            if combined:
                acct_vec = _personality_vector(combined)
            for kid, rws in by_key.items():
                if not rws:
                    continue
                similar.append({"public_id": kid, "name": id_name.get(kid) or kid,
                                "match": round(_cosine(pv, _personality_vector(rws)) * 100)})
            similar.sort(key=lambda s: s["match"], reverse=True)
            similar = similar[:3]
        out["personality"]["account_vector"] = {k: round(acct_vec[k], 3) for k in _PERSONALITY_DIMS}
        out["personality"]["similar_keys"] = similar
        avg_per_key = (acct_total / keys_with_traffic) if keys_with_traffic else 0
        out["account"] = {
            "total": acct_total, "keys_with_traffic": keys_with_traffic,
            "avg_requests_per_key": round(avg_per_key, 1),
            "this_vs_avg": round(total / avg_per_key, 2) if avg_per_key else None,
        }

        # ── 30-day composite-health trend ────────────────────────────────
        # Read came from the batch (snap_rows). Write today's snapshot only on
        # the canonical 24h window so 1h/7d toggles stay on the fast single
        # round-trip; the daily point still accrues whenever 24h is viewed.
        health["trend"] = list(reversed(
            [{"day": r["day"], "composite": r["composite"]} for r in (snap_rows or [])]))
        if window == "24h" and health.get("composite") is not None:
            try:
                P = health.get("pillars", {})
                c.execute(
                    "INSERT OR REPLACE INTO api_key_health_snapshot"
                    "(key_id, day, composite, reliability, performance, capacity,"
                    " hygiene, sample_size, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (public_id, now.strftime("%Y-%m-%d"), health.get("composite"),
                     (P.get("reliability") or {}).get("score"),
                     (P.get("performance") or {}).get("score"),
                     (P.get("capacity") or {}).get("score"),
                     (P.get("hygiene") or {}).get("score"),
                     total, now.isoformat()))
                try: c.commit()
                except Exception: pass
                # reflect today's point immediately (the batch read predates it)
                _today = now.strftime("%Y-%m-%d")
                _tr = [t for t in health["trend"] if t["day"] != _today]
                _tr.append({"day": _today, "composite": health.get("composite")})
                health["trend"] = _tr
            except Exception as e:
                log.warning("health snapshot write failed: %s", e)

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


# ════════════════════════════════════════════════════════════════════════
# 9.N.T · TRAFFIC tab — hero (status-stacked over time), stats rail, GitHub
# calendar, traffic clock (local hour-of-day), bytes-flow mirror. One batched
# round-trip. Buckets are computed in the VIEWER's timezone (tz_off minutes)
# by shifting the UTC timestamp inside SQLite — storage stays UTC.
# ════════════════════════════════════════════════════════════════════════
_TRAFFIC_GRAN = {
    "hour":  {"win_min": 24 * 60,        "sub": 13, "n": 24},
    "day":   {"win_min": 30 * 24 * 60,   "sub": 10, "n": 30},
    "week":  {"win_min": 12 * 7 * 24 * 60, "sub": 10, "n": 12},  # daily rows → 12 weeks
    "month": {"win_min": 372 * 24 * 60,  "sub": 7,  "n": 12},
}


def _build_key_traffic(user_id: int, public_id: str, granularity: str,
                       tz_off: int) -> dict:
    """All Traffic-tab data in one batched read. tz_off = viewer UTC offset in
    minutes (IST → 330); used to bucket hero/calendar/clock in local time."""
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    if granularity not in _TRAFFIC_GRAN:
        granularity = "hour"
    cfg = _TRAFFIC_GRAN[granularity]
    sub, nb, win_min = cfg["sub"], cfg["n"], cfg["win_min"]
    try:
        tz_off = int(tz_off)
    except Exception:
        tz_off = 0
    tz_off = max(-840, min(840, tz_off))          # clamp ±14h
    modifier = f"{tz_off:+d} minutes"             # SQLite datetime() modifier
    now = _dt.now(_tz.utc)
    now_local = now + _td(minutes=tz_off)
    cut     = (now - _td(minutes=win_min)).strftime("%Y-%m-%d %H:%M:%S.%f")
    cut_min = (now - _td(minutes=win_min)).strftime("%Y-%m-%d %H:%M")
    cut_cal = (now - _td(days=371)).strftime("%Y-%m-%d %H:%M:%S.%f")
    out = {"public_id": public_id, "granularity": granularity, "tz_off": tz_off}

    def _utc_ms(local_naive):
        # local wall-clock (naive) → real UTC epoch ms
        return int((local_naive.replace(tzinfo=_tz.utc).timestamp() - tz_off * 60) * 1000)

    with auth_db.get_conn() as c:
        def _rows(sql, args=()):
            try:
                return [dict(r) for r in c.execute(sql, args).fetchall()]
            except Exception as e:
                log.warning("traffic query failed: %s", e)
                return []
        # tz-shifted bucket expressions (strip microseconds first so datetime()
        # parses reliably, then apply the viewer offset, then slice the key).
        bexpr = f"substr(datetime(substr(timestamp,1,19), ?), 1, {sub})"
        dexpr = "substr(datetime(substr(timestamp,1,19), ?), 1, 10)"
        hexpr = "substr(datetime(substr(timestamp,1,19), ?), 12, 2)"
        READS = [
            (f"SELECT {bexpr} AS b, "
             "SUM(CASE WHEN status_code>=200 AND status_code<400 THEN 1 ELSE 0 END) AS n2, "
             "SUM(CASE WHEN status_code>=400 AND status_code<500 THEN 1 ELSE 0 END) AS n4, "
             "SUM(CASE WHEN status_code>=500 THEN 1 ELSE 0 END) AS n5, "
             "SUM(COALESCE(bytes_in,0)) AS bin, SUM(COALESCE(bytes_out,0)) AS bout, COUNT(*) AS n "
             "FROM api_key_request_log WHERE key_id=? AND timestamp>=? GROUP BY b ORDER BY b",
             (modifier, public_id, cut)),
            (f"SELECT {dexpr} AS d, COUNT(*) AS n, "
             "SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) AS e "
             "FROM api_key_request_log WHERE key_id=? AND timestamp>=? GROUP BY d ORDER BY d",
             (modifier, public_id, cut_cal)),
            (f"SELECT {hexpr} AS h, COUNT(*) AS n, "
             "SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) AS e "
             "FROM api_key_request_log WHERE key_id=? AND timestamp>=? GROUP BY h ORDER BY h",
             (modifier, public_id, cut)),
            ("SELECT path, SUM(COALESCE(bytes_out,0)) AS bout, "
             "SUM(COALESCE(bytes_in,0)) AS bin, COUNT(*) AS n "
             "FROM api_key_request_log WHERE key_id=? AND timestamp>=? "
             "GROUP BY path ORDER BY bout DESC LIMIT 12", (public_id, cut)),
            ("SELECT minute_ts, requests FROM api_key_usage_minute "
             "WHERE key_id=? AND minute_ts>=? ORDER BY requests DESC LIMIT 1",
             (public_id, cut_min)),
            # method × status-bucket matrix (the honeycomb hive)
            ("SELECT method, "
             "CASE WHEN status_code>=500 THEN 5 WHEN status_code>=400 THEN 4 ELSE 2 END AS sb, "
             "COUNT(*) AS n FROM api_key_request_log WHERE key_id=? AND timestamp>=? "
             "GROUP BY method, sb", (public_id, cut)),
            ("SELECT public_id, name FROM api_keys WHERE user_id=? AND deleted_at IS NULL",
             (user_id,)),
        ]
        if hasattr(c, "batch_read"):
            try:
                res = [[dict(r) for r in cur.fetchall()] for cur in c.batch_read(READS)]
            except Exception as e:
                log.warning("traffic batch failed, sequential: %s", e)
                res = [_rows(s, a) for s, a in READS]
        else:
            res = [_rows(s, a) for s, a in READS]
        bucket_rows, cal_rows, clock_rows, ep_rows, peak_rows, matrix_rows, key_rows = res

        key = next((r for r in key_rows if r.get("public_id") == public_id), None)
        if key is None:
            return {"not_found": True, "public_id": public_id, "granularity": granularity}
        out["key"] = {"name": key.get("name")}

        # ── ordered local bucket keys + two-line labels (main + sub) ──
        keys, labels, subs, tms = [], [], [], []
        MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dur_ms = {"hour": 3600_000, "day": 86400_000,
                  "week": 7 * 86400_000, "month": 31 * 86400_000}[granularity]
        if granularity == "hour":
            base = now_local.replace(minute=0, second=0, microsecond=0)
            for i in range(nb - 1, -1, -1):
                d = base - _td(hours=i)
                keys.append(d.strftime("%Y-%m-%d %H"))
                h12 = (d.hour % 12) or 12
                labels.append(f"{h12} {'AM' if d.hour < 12 else 'PM'}")
                subs.append(WD[d.weekday()] if d.hour == 0 else "")
                tms.append(_utc_ms(d))
        elif granularity == "day":
            base = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            for i in range(nb - 1, -1, -1):
                d = base - _td(days=i)
                keys.append(d.strftime("%Y-%m-%d"))
                labels.append(f"{d.month}/{d.day}"); subs.append(WD[d.weekday()])
                tms.append(_utc_ms(d))
        elif granularity == "month":
            seq = []
            y, m = now_local.year, now_local.month
            for i in range(nb):
                mm, yy = m - i, y
                while mm <= 0:
                    mm += 12; yy -= 1
                seq.append((yy, mm))
            seq.reverse()
            for yy, mm in seq:
                keys.append(f"{yy:04d}-{mm:02d}")
                labels.append(MON[mm - 1]); subs.append(str(yy) if mm == 1 else "")
                tms.append(_utc_ms(_dt(yy, mm, 1)))
        else:  # week — SQL returns local daily keys; fold into 12 Monday-weeks
            base = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            monday = base - _td(days=base.weekday())
            for i in range(nb - 1, -1, -1):
                wk = monday - _td(weeks=i)
                wke = wk + _td(days=6)
                keys.append(wk.strftime("%Y-%m-%d"))
                labels.append(f"{MON[wk.month-1]} {wk.day}–{wke.day}")
                subs.append(f"wk {nb - i}")
                tms.append(_utc_ms(wk))
        idx = {k: i for i, k in enumerate(keys)}
        out["bucket_ms"] = dur_ms

        def _wk_key(daykey):
            try:
                dd = _dt.strptime(daykey, "%Y-%m-%d")
                return (dd - _td(days=dd.weekday())).strftime("%Y-%m-%d")
            except Exception:
                return None

        n2a = [0] * nb; n4a = [0] * nb; n5a = [0] * nb
        bina = [0] * nb; bouta = [0] * nb
        for r in bucket_rows:
            b = r.get("b")
            if granularity == "week":
                b = _wk_key(b)
            i = idx.get(b)
            if i is None:
                continue
            n2a[i] += int(r.get("n2") or 0); n4a[i] += int(r.get("n4") or 0)
            n5a[i] += int(r.get("n5") or 0)
            bina[i] += int(r.get("bin") or 0); bouta[i] += int(r.get("bout") or 0)

        hero = [{"label": labels[i], "sub": subs[i], "t_ms": tms[i], "n2": n2a[i],
                 "n4": n4a[i], "n5": n5a[i], "total": n2a[i] + n4a[i] + n5a[i]}
                for i in range(nb)]
        out["hero"] = {"buckets": hero, "max": max((h["total"] for h in hero), default=0)}

        # method × status-bucket matrix (honeycomb hive)
        mtx = {}
        for r in matrix_rows:
            m = (r.get("method") or "?").upper()
            sbk = "n" + str(r.get("sb") or 2)
            mtx.setdefault(m, {"n2": 0, "n4": 0, "n5": 0})[sbk] += int(r.get("n") or 0)
        out["matrix"] = [{"method": m, "n2": v["n2"], "n4": v["n4"], "n5": v["n5"],
                          "total": v["n2"] + v["n4"] + v["n5"]}
                         for m, v in sorted(mtx.items(), key=lambda kv: -(kv[1]["n2"] + kv[1]["n4"] + kv[1]["n5"]))]

        bytes_buckets = [{"label": labels[i], "sub": subs[i], "t_ms": tms[i], "bin": bina[i], "bout": bouta[i]}
                         for i in range(nb)]
        total_in, total_out = sum(bina), sum(bouta)
        total_req = sum(h["total"] for h in hero)
        out["bytes"] = {
            "buckets": bytes_buckets, "total_in": total_in, "total_out": total_out,
            "avg_in": round(total_in / total_req) if total_req else 0,
            "avg_out": round(total_out / total_req) if total_req else 0,
            "ratio": round(total_out / total_in, 1) if total_in else None,
            "by_endpoint": [{"path": r["path"], "bout": int(r.get("bout") or 0),
                             "bin": int(r.get("bin") or 0), "n": int(r.get("n") or 0)}
                            for r in ep_rows],
        }

        n2t, n4t, n5t = sum(n2a), sum(n4a), sum(n5a)
        tot = n2t + n4t + n5t
        busiest_i = max(range(nb), key=lambda i: hero[i]["total"]) if nb else 0
        out["stats"] = {
            "total": tot,
            "error_pct": round(100 * (n4t + n5t) / tot, 1) if tot else 0,
            "busiest_label": hero[busiest_i]["label"] if tot else "—",
            "busiest_count": hero[busiest_i]["total"] if tot else 0,
            "peak_per_min": int(peak_rows[0].get("requests") or 0) if peak_rows else 0,
        }

        # ── clock: 24 local hours ──
        ch = [0] * 24; ce = [0] * 24
        for r in clock_rows:
            try:
                h = int(r.get("h"))
            except Exception:
                continue
            if 0 <= h < 24:
                ch[h] = int(r.get("n") or 0); ce[h] = int(r.get("e") or 0)
        out["clock"] = {
            "hours": [{"h": h, "n": ch[h], "e": ce[h],
                       "err_pct": round(100 * ce[h] / ch[h], 1) if ch[h] else 0}
                      for h in range(24)],
            "max": max(ch) if ch else 0,
            "busiest_h": (max(range(24), key=lambda h: ch[h]) if any(ch) else None),
            "now_h": now_local.hour, "now_min": now_local.minute,
        }

        # ── calendar: last ~52 weeks, daily (local dates) ──
        out["calendar"] = {
            "days": [{"date": r["d"], "n": int(r.get("n") or 0), "e": int(r.get("e") or 0)}
                     for r in cal_rows if r.get("d")],
            "max": max((int(r.get("n") or 0) for r in cal_rows), default=0),
        }

    return out


@router.get("/{public_id}/overview")
@api_endpoint("/api/account/keys/{public_id}/overview")
async def get_key_overview(request: Request, public_id: str,
                           window: str = Query("24h")):
    """9.N.9 · One-shot overview payload for the bento Overview tab."""
    import asyncio as _aio
    user = _get_session_user(request)
    if window not in _WINDOW_MINUTES:
        window = "24h"
    # Ownership + existence are resolved inside the batched read (the key row
    # is fetched alongside the window data — no separate lookup round-trip).
    data = await _aio.to_thread(_build_key_overview, int(user["id"]), public_id, window)
    if data.get("not_found"):
        raise ApiError("not_found", f"key {public_id!r} not found.")
    return data


@router.get("/{public_id}/traffic")
@api_endpoint("/api/account/keys/{public_id}/traffic")
async def get_key_traffic(request: Request, public_id: str,
                          granularity: str = Query("hour"),
                          tz_off: int = Query(0)):
    """9.N.T · One-shot Traffic-tab payload (hero / stats / calendar / clock /
    bytes). tz_off = viewer UTC offset in minutes so buckets render in local
    time."""
    import asyncio as _aio
    user = _get_session_user(request)
    if granularity not in _TRAFFIC_GRAN:
        granularity = "hour"
    data = await _aio.to_thread(_build_key_traffic, int(user["id"]),
                                public_id, granularity, tz_off)
    if data.get("not_found"):
        raise ApiError("not_found", f"key {public_id!r} not found.")
    return data


@router.get("/{public_id}/requests")
@api_endpoint("/api/account/keys/{public_id}/requests")
async def get_key_requests(request: Request, public_id: str,
                            limit: int = Query(50, ge=1, le=200),
                            cursor: Optional[int] = Query(None, ge=1),
                            since: Optional[str] = Query(None),
                            until: Optional[str] = Query(None)):
    """Phase 8 Wave D: per-request log for the detail page. Newest first.
    9.N.T: optional since/until (UTC 'YYYY-MM-DD HH:MM:SS' or ms epoch) time
    filter so Traffic drill-downs land on a specific bucket/day."""
    user = _get_session_user(request)
    items = auth_db.list_key_requests(
        user_id=int(user["id"]), public_id=public_id,
        limit=limit, cursor=cursor, since_iso=since, until_iso=until)
    return {"items": items, "count": len(items), "public_id": public_id,
            "since": since, "until": until}


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
