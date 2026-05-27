"""API Console — sudo mode endpoints (spec §7.6).

Three endpoints exposed under `/api/account/sudo`:
    POST   /api/account/sudo           verify password, mint sudo cookie
    POST   /api/account/sudo/revoke    invalidate active sudo
    GET    /api/account/sudo           probe sudo state

Auth model:
    - Session-cookie auth (apin_v2_session) required on ALL three.
    - X-Console-Csrf header required on ALL three (per PDA-R2-F25 +
      §7.6: CSRF on GET defends against cross-origin probe via
      <img>/<link> tags. Starlette default treats GET as CSRF-exempt
      so the handler MUST opt in.)
    - These routes are EXEMPT from SudoMiddleware (`/api/account/sudo` and
      `/api/account/sudo/revoke` listed at spec §9.2 line 3252 — chicken-
      and-egg avoidance: you need to mint sudo before you can use it).

Cookie shape (per spec §7.6):
    Set-Cookie: apin_sudo=<base64url(secret)>; HttpOnly; Secure;
                SameSite=Strict; Path=/api/account;
                Max-Age=<sudo_session_length_seconds>

What's STUBBED (deferred to later phases):
    - settings.sudo_session_length_seconds — using fixed 1800 s (30 min)
      until the account_settings table is wired. Spec §7.6 wants this
      to be per-user configurable.
    - Rate-limit on POST /sudo — spec §5791 mentions a rate-limiter on
      the sudo POST. Currently absent; add in Phase 4 alongside the
      general rate-limit table.
    - Audit emission to api_key_audit hash chain — currently writes to
      audit_log via auth_db.audit(). Phase 4 wires the hash-chained
      api_key_audit table.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db

log = logging.getLogger("apin_v2.account.routes_sudo")

router = APIRouter(prefix="/api/account/sudo", tags=["account/sudo"])


# ── Constants ─────────────────────────────────────────────────────────────

_SESSION_COOKIE_NAME = "apin_v2_session"
_SUDO_COOKIE_NAME = "apin_sudo"
# WI-P4-ACCT-SETTINGS: TTL is now per-user via account_settings.
# `_SUDO_TTL_DEFAULT` is the fallback when `get_account_settings` returns
# the static default dict (e.g. tests that mock users without going through
# trg_user_insert_default_settings). The schema's CHECK constraint enforces
# the 60-900 range, so this default is always valid.
_SUDO_TTL_DEFAULT = 300


# ── Auth helpers ─────────────────────────────────────────────────────────
#
# Phase 5.1 (WI-P4-DEDUP-SESS): canonical helpers live in
# `_session_helpers`. The previous in-file copies were verbatim duplicates
# (after the Phase 4 CSRF upgrade applied to both files). Re-export under
# the underscored names so the call sites in this file don't need to change.
from scripts.apin_v2.account import _session_helpers as _sh

_get_session_with_id = _sh.get_session_with_id
_require_csrf = _sh.require_csrf


def _client_ip(request: Request) -> Optional[str]:
    """Extract client IP (best-effort)."""
    # Prefer X-Forwarded-For rightmost-untrusted (REV-R2-I03), but in dev
    # this header isn't usually set; fall back to request.client.host.
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else None


def _build_sudo_cookie_header(raw_value: str, ttl_seconds: int) -> str:
    """Build the Set-Cookie header value per spec §7.6 line 2881."""
    parts = [
        f"{_SUDO_COOKIE_NAME}={raw_value}",
        "HttpOnly",
        "Secure",
        "SameSite=Strict",
        "Path=/api/account",
        f"Max-Age={ttl_seconds}",
    ]
    return "; ".join(parts)


def _build_sudo_clear_header() -> str:
    """Build the Set-Cookie header value that clears the sudo cookie."""
    parts = [
        f"{_SUDO_COOKIE_NAME}=",
        "HttpOnly",
        "Secure",
        "SameSite=Strict",
        "Path=/api/account",
        "Max-Age=0",
    ]
    return "; ".join(parts)


# ── Routes ────────────────────────────────────────────────────────────────

@router.post("", status_code=200)
@api_endpoint("/api/account/sudo")
async def sudo_start(request: Request):
    """POST /api/account/sudo — verify password, mint sudo cookie.

    Spec §7.6 (lines 2875-2884).
    """
    _require_csrf(request)
    user, session_id = _get_session_with_id(request)

    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter",
                       "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter",
                       "Request body must be a JSON object.")

    password = body.get("password")
    if not isinstance(password, str) or not password:
        raise ApiError("invalid_parameter",
                       "password (string) is required.",
                       details={"field": "password"})

    client_ip = _client_ip(request)

    # WI-P4-RATE-SUDO (spec line 5791): rate-limit POST /sudo per
    # (user_id, ip) to defeat brute-force grinding through argon2id.
    # 5 attempts per 5-min sliding window. On limit, return 429 with
    # spec-canonical `rate_limited` code.
    if not auth_db.check_sudo_rate_limit(int(user["id"]), client_ip):
        try:
            auth_db.audit(
                "sudo_rate_limited",
                user_id=int(user["id"]),
                detail={"session_id": session_id, "ip": client_ip},
                ip_addr=client_ip,
            )
        except Exception as e:
            log.warning("sudo_start: rate-limit audit emission failed: %s", e)
        raise ApiError(
            "rate_limited",
            "Too many sudo attempts. Wait a few minutes and try again.",
            hint="The rate-limiter resets after 5 minutes.",
        )

    # Verify against stored hash (argon2/bcrypt)
    pw_hash = user.get("password_hash")
    if not pw_hash or not auth_db.verify_password(password, pw_hash):
        # FIX-T3 (PDA-P3.3-3.4-R1 F1): emit an audit row for failed sudo
        # attempts. Without this, brute-force probing through a stolen
        # session cookie is invisible in the canonical audit trail
        # (STRIDE A3 repudiation gap). The audit row records the user_id
        # and IP — enough for a Phase 4 rate-limiter or alert system to
        # observe patterns.
        try:
            auth_db.audit(
                "sudo_failed",
                user_id=int(user["id"]),
                detail={"session_id": session_id, "reason": "bad_password"},
                ip_addr=client_ip,
            )
        except Exception as e:
            log.warning("sudo_start: failed-attempt audit emission failed: %s", e)
        # Generic message — never disclose whether the password was wrong
        # vs the account locked. Code choice: `invalid_credentials` is
        # NOT in the spec §26 canonical 44 codes (auth_routes.py:303 uses
        # raw HTTPException 401 for the same case, bypassing the envelope
        # contract entirely). Within the canonical pool, `invalid_parameter`
        # (400) is the closest semantically — the password parameter was
        # rejected. Phase 4 spec update can introduce a dedicated code.
        log.warning("sudo_start: password verify failed for user_id=%s",
                    user.get("id"))
        raise ApiError("invalid_parameter",
                       "Password verification failed.",
                       details={"field": "password",
                                "reason": "verification_failed"})

    # FIX-T4 (PDA-P3.3-3.4-R1 F2): rotate session CSRF BEFORE creating
    # sudo. If the session row vanished between password verify and now
    # (admin revoke, concurrent logout, etc.), `rotate_session_csrf_token`
    # returns None — in that case we MUST NOT create a sudo cookie bound
    # to a dead session (would leave an orphan sudo row with no rotated
    # CSRF). Original ordering created the sudo first and returned
    # `csrf_token: None` on rotate failure — leaving the orphan + a half-
    # baked response. Reordered + fail-closed:
    new_csrf = auth_db.rotate_session_csrf_token(session_id)
    if new_csrf is None:
        # Session disappeared between password verify and CSRF rotate.
        # Treat as session-expired; the client should re-auth from scratch.
        try:
            auth_db.audit(
                "sudo_failed",
                user_id=int(user["id"]),
                detail={"session_id": session_id, "reason": "session_vanished"},
                ip_addr=client_ip,
            )
        except Exception as e:
            log.warning("sudo_start: vanished-session audit emission failed: %s", e)
        raise ApiError(
            "invalid_or_missing_token",
            "Session expired between password verify and sudo issue. "
            "Please sign in again.",
        )

    # WI-P4-ACCT-SETTINGS: read TTL from per-user settings (spec §6.10).
    # Falls back to the schema default (300 s) if no row exists.
    settings = auth_db.get_account_settings(int(user["id"]))
    ttl = int(settings.get("sudo_session_length_seconds", _SUDO_TTL_DEFAULT)
              or _SUDO_TTL_DEFAULT)
    # Defensive clamp matching the schema CHECK (60-900). Should be
    # unnecessary because the column has the CHECK constraint, but if
    # the fallback default is ever tweaked outside the range, this
    # prevents Set-Cookie Max-Age from drifting out of spec.
    ttl = max(60, min(900, ttl))

    # Mint the sudo token AFTER successful CSRF rotation.
    raw_sudo, expires_at = auth_db.create_sudo_token(
        user_id=int(user["id"]),
        session_id=session_id,
        client_ip=client_ip,
        ttl_seconds=ttl,
    )

    # FX-P4-4 (PDA-P4-R1 F4): clear rate-limit history for this
    # (user, IP) on success. Symmetric with the login flow's
    # `clear_rate_limit(ip)` at auth_routes.py:311. Without this,
    # a legitimate user who fat-fingers their password 5 times in
    # 5 minutes stays locked out for the FULL window even after
    # they finally type it right.
    try:
        auth_db.clear_sudo_rate_limit(int(user["id"]), client_ip)
    except Exception as e:
        # Cleanup failure is non-fatal — the lockout will lift
        # naturally when the window expires.
        log.warning("sudo_start: clear_sudo_rate_limit failed: %s", e)

    # Audit success
    try:
        auth_db.audit(
            "sudo_started",
            user_id=int(user["id"]),
            detail={"session_id": session_id},
            ip_addr=client_ip,
        )
    except Exception as e:
        # Audit failure must not block the route — log + carry on.
        log.warning("sudo_start: audit failed: %s", e)

    # Build cookie header + envelope tuple (data, meta, extra_headers)
    cookie_header = _build_sudo_cookie_header(raw_sudo, ttl)
    return (
        {"expires_at": expires_at, "csrf_token": new_csrf},
        {},
        {"Set-Cookie": cookie_header,
         # The plaintext sudo cookie value is the response Set-Cookie line.
         # Without the opt-out header, TokenRedactionMiddleware would scan
         # the body looking for tokens, but apin_sudo is a base64url string
         # not matching the apin_* regex — so it's incidentally safe. Even
         # so, we opt out to be explicit (one-time-handoff ceremony).
         "X-APIN-No-Redact": "1"},
    )


@router.post("/revoke", status_code=200)
@api_endpoint("/api/account/sudo/revoke")
async def sudo_revoke(request: Request):
    """POST /api/account/sudo/revoke — invalidate active sudo.

    Spec §7.6 (lines 2886-2890).
    """
    _require_csrf(request)
    user, session_id = _get_session_with_id(request)

    revoked = auth_db.revoke_active_sudo_for_session(session_id)
    client_ip = _client_ip(request)

    try:
        auth_db.audit(
            "sudo_revoked",
            user_id=int(user["id"]),
            detail={"session_id": session_id, "revoked_count": revoked},
            ip_addr=client_ip,
        )
    except Exception as e:
        log.warning("sudo_revoke: audit failed: %s", e)

    # Clear the cookie
    clear_header = _build_sudo_clear_header()
    return (
        {"revoked": revoked},
        {},
        {"Set-Cookie": clear_header},
    )


@router.get("")
@api_endpoint("/api/account/sudo")
async def sudo_state(request: Request):
    """GET /api/account/sudo — probe sudo state.

    Spec §7.6 (lines 2892-2895). CSRF required per PDA-F25 (defends
    against cross-origin probe via <img>/<link>).
    """
    _require_csrf(request)
    user, session_id = _get_session_with_id(request)

    cookie_value = request.cookies.get(_SUDO_COOKIE_NAME, "")
    if not cookie_value:
        return {"active": False}

    state = auth_db.get_sudo_state_for_cookie(
        cookie_value=cookie_value,
        session_id=session_id,
        client_ip=_client_ip(request),
    )
    return state
