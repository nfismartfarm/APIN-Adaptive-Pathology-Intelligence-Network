"""API Console — account settings GET + PATCH (Phase 6.B.1).

Endpoints under /api/account/settings:
  GET    — return current account_settings row (with schema defaults
           filled in for any missing field)
  PATCH  — update a subset of the editable fields. Requires sudo.

CSRF presence + real-token compare (FX-P4-CSRF) applies to both.
Sudo gate (SudoMiddleware) applies to PATCH only — GET is read-only,
spec §9.2 read-method bypass.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import _session_helpers as _sh

log = logging.getLogger("apin_v2.account.routes_settings")

router = APIRouter(prefix="/api/account/settings", tags=["account/settings"])


@router.get("")
@api_endpoint("/api/account/settings")
async def get_settings(request: Request):
    """GET /api/account/settings — return the caller's account_settings."""
    user = _sh.get_session_user(request)
    return auth_db.get_account_settings(int(user["id"]))


@router.patch("")
@api_endpoint("/api/account/settings")
async def patch_settings(request: Request):
    """PATCH /api/account/settings — update editable fields. Requires sudo.

    Phase 8 Wave C (WI-P8-SUDO-AMPLIFY): if this PATCH RAISES sudo_max_uses
    (the cap on how many mutating requests a sudo session can issue), we
    auto-revoke the current sudo session. Reason: if an attacker compromises
    a sudo session, they could amplify its blast radius by PATCHing the cap
    upward, then continuing to mutate. Forcing re-auth on raise neutralises
    that. Lowering the cap is safe and does not trigger revocation.
    """
    _sh.require_csrf(request)
    user, session_id = _sh.get_session_with_id(request)

    # Snapshot the BEFORE state for amplify detection + audit
    before = auth_db.get_account_settings(int(user["id"]))

    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")

    try:
        updated = auth_db.update_account_settings(int(user["id"]), **body)
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e

    # FX-P8-C2 amplify guard: detect a raise of sudo_max_uses and revoke.
    revoked = False
    try:
        before_cap = int(before.get("sudo_max_uses") or 0)
        after_cap  = int(updated.get("sudo_max_uses") or 0)
        if after_cap > before_cap:
            n_revoked = auth_db.revoke_active_sudo_for_session(session_id)
            revoked = (n_revoked > 0)
            log.info("FX-P8-C2: sudo_max_uses raised %d -> %d for user_id=%d; "
                     "revoked %d active sudo token(s) for session %r",
                     before_cap, after_cap, int(user["id"]),
                     n_revoked, session_id[:8] + "...")
    except Exception as e:
        log.warning("amplify-guard check failed: %s: %s", type(e).__name__, e)

    # Audit-log the change (FX-P8-C1)
    try:
        from scripts.apin_v2 import auth_db as _adb
        _adb.append_audit_log(
            user_id=int(user["id"]),
            action="settings_patched",
            actor_session_id=session_id,
            before={"sudo_max_uses": before.get("sudo_max_uses")},
            after={"sudo_max_uses": updated.get("sudo_max_uses"),
                   "_patched_keys": sorted(body.keys()),
                   "_sudo_revoked_amplify_guard": revoked},
        )
    except Exception as e:
        log.warning("audit emit failed: %s: %s", type(e).__name__, e)

    if revoked:
        updated = dict(updated)
        updated["sudo_revoked_amplify_guard"] = True
        updated["sudo_revoked_message"] = (
            "Sudo session was auto-revoked because you raised "
            "sudo_max_uses. Re-authenticate before making more "
            "mutating requests.")
        # Phase 8.H · default-ON (warn severity) — security event.
        try:
            auth_db.emit_alert(
                int(user["id"]), "account.sudo_amplify_guard",
                action={"kind": "view_settings"},
            )
        except Exception:
            pass

    # Phase 8.H · account.settings_changed default-OFF.
    try:
        if body:
            auth_db.emit_alert(
                int(user["id"]), "account.settings_changed",
                action={"kind": "view_settings"},
                fields_changed=", ".join(sorted(body.keys())),
            )
    except Exception:
        pass
    return updated
