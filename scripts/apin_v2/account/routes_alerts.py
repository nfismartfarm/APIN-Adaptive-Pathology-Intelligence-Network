"""API Console — alert routes.

Phase 7 Wave 1.

Spec contract: spec_v7.md §7.5 (alert endpoints), §6.7 (api_key_alerts),
§14.1 (alert code enum), §15.17 (alerts UI).

Six endpoints under /api/account/alerts:
    GET    /api/account/alerts                         list (with filters)
    GET    /api/account/alerts/unread-count            for the nav bell badge
    GET    /api/account/alerts/{id}                    fetch one
    PATCH  /api/account/alerts/{id}/read               mark read (idempotent)
    DELETE /api/account/alerts/{id}                    dismiss (soft-delete)
    POST   /api/account/alerts/{id}/restore            un-dismiss

Auth model:
    - Session-cookie auth (apin_v2_session).
    - NO sudo gating — alerts are user-state housekeeping (read flags,
      dismiss) and per spec §9.2 these are not in the sudo_required_for
      list. CSRF required on all mutating verbs.
    - Spec §7.5 also describes POST /api/account/alerts/{id}/snooze
      with `{until: ISO}` body. Filed as WI-P8-ALERT-SNOOZE — not built
      in Wave 1 because the snooze surfacing UX (bell-badge skip until
      snooze expires) is non-trivial.

Honesty note: alert creation is purely SERVER-INITIATED (background
events). There is no POST /api/account/alerts. Callers use
auth_db.create_alert() from event handlers (webhook failure, quota
warnings, new-IP detection, etc).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request, Query

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import _session_helpers as _sh

log = logging.getLogger("apin_v2.account.routes_alerts")

router = APIRouter(prefix="/api/account/alerts", tags=["account/alerts"])


# ─── Routes ───────────────────────────────────────────────────────────────

@router.get("")
@api_endpoint("/api/account/alerts")
async def list_alerts_route(
    request: Request,
    severity: Optional[str] = Query(None, regex=r"^(info|warn|critical)$"),
    code: Optional[str] = Query(None, max_length=80),
    key_id: Optional[str] = Query(None, max_length=80),
    only_unread: bool = Query(False),
    include_dismissed: bool = Query(False),
    limit: int = Query(50, ge=1, le=100),
    cursor: Optional[str] = Query(None),
):
    """List alerts with filters. Cursor pagination by (updated_at DESC, id DESC)."""
    user = _sh.get_session_user(request)
    try:
        page = auth_db.list_alerts(
            int(user["id"]),
            severity=severity, code=code, key_id=key_id,
            only_unread=only_unread,
            include_dismissed=include_dismissed,
            limit=limit, cursor=cursor,
        )
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    return page


# ─── Phase 8.H.D · alert preferences ──────────────────────────────────
# GET   /api/account/alert-prefs  → returns the registry catalogue + the
#                                   user's current overrides
# PATCH /api/account/alert-prefs  → upsert categories + per-code overrides
#                                   into account_settings.notify_prefs_json
@router.get("/prefs/registry")
@api_endpoint("/api/account/alerts/prefs/registry")
async def alert_prefs_registry(request: Request):
    """Return the catalogue of every alert code, grouped by category, with
    metadata (severity, default_on, title template). Used by the Settings
    UI to render the toggle list. Doesn't include user values."""
    user = _sh.get_session_user(request)
    _ = user  # cookie session required, value unused
    return auth_db.alert_registry_snapshot()


@router.get("/prefs")
@api_endpoint("/api/account/alerts/prefs")
async def alert_prefs_get(request: Request):
    """Return the user's current alert preference overrides as the JSON
    structure stored in account_settings.notify_prefs_json."""
    user = _sh.get_session_user(request)
    return auth_db._resolve_notify_prefs(int(user["id"]))


@router.patch("/prefs")
@api_endpoint("/api/account/alerts/prefs")
async def alert_prefs_patch(request: Request):
    """Upsert alert preferences. Body shape:
        {"categories": {"key_lifecycle": false, ...},
         "codes":      {"key.created": true, ...}}
    Unknown categories/codes are silently dropped (defensive). The merge
    is REPLACE — clients should send the full desired prefs object.
    """
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")
    # Sanitise: keep only known categories + codes.
    known_cats = set(auth_db._ALERT_CATEGORIES)
    known_codes = set(auth_db._ALERT_REGISTRY.keys())
    cats = {k: bool(v) for k, v in (body.get("categories") or {}).items()
            if k in known_cats}
    codes = {k: bool(v) for k, v in (body.get("codes") or {}).items()
             if k in known_codes}
    prefs = {"categories": cats, "codes": codes}
    import json as _json
    auth_db.update_account_settings(
        int(user["id"]),
        notify_prefs_json=_json.dumps(prefs),
    )
    return prefs


@router.get("/unread-count")
@api_endpoint("/api/account/alerts/unread-count")
async def unread_count_route(request: Request,
                              since: Optional[int] = None):
    """For the nav bell badge + the toast detector poll.

    Phase 8.H · `?since=<id>` — when the client tells us the highest alert
    id it has already seen, we additionally return:
      - `new_unread`         (count of unread alerts with id > since)
      - `new_alerts`         (up to 10 of the newest unread rows, oldest-first
                              so the toast layer can slide them in in order)
      - `latest_id`          (the highest id in the user's inbox, including
                              read+dismissed — so the client can advance its
                              `since` cursor monotonically and never re-toast)
    Without `since`, only the global `unread` count is returned (cheap).
    """
    user = _sh.get_session_user(request)
    n = auth_db.count_unread_alerts(int(user["id"]))
    out: dict = {"unread": int(n)}
    if since is not None:
        try:
            since_i = int(since)
        except (TypeError, ValueError):
            since_i = 0
        new_alerts = auth_db.list_alerts_since(int(user["id"]), since_i)
        out["new_unread"] = len([a for a in new_alerts if not a.get("read_at")])
        out["new_alerts"] = new_alerts
        out["latest_id"] = auth_db.latest_alert_id(int(user["id"]))
    return out


@router.get("/{alert_id}")
@api_endpoint("/api/account/alerts/{alert_id}")
async def get_alert_route(alert_id: int, request: Request):
    user = _sh.get_session_user(request)
    a = auth_db.get_alert(int(alert_id), int(user["id"]))
    if a is None:
        raise ApiError("not_found", "Alert not found.")
    return a


@router.patch("/{alert_id}/read")
@api_endpoint("/api/account/alerts/{alert_id}/read")
async def mark_read_route(alert_id: int, request: Request):
    """Idempotent. Marking an already-read alert is a no-op."""
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    a = auth_db.mark_alert_read(int(alert_id), int(user["id"]))
    if a is None:
        raise ApiError("not_found", "Alert not found.")
    return a


@router.delete("/{alert_id}")
@api_endpoint("/api/account/alerts/{alert_id}")
async def dismiss_route(alert_id: int, request: Request):
    """Soft-delete: sets dismissed_at. Reversible via /restore."""
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    a = auth_db.dismiss_alert(int(alert_id), int(user["id"]))
    if a is None:
        raise ApiError("not_found", "Alert not found.")
    return a


@router.post("/{alert_id}/restore")
@api_endpoint("/api/account/alerts/{alert_id}/restore")
async def restore_route(alert_id: int, request: Request):
    """Reverses a prior dismiss."""
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    a = auth_db.restore_alert(int(alert_id), int(user["id"]))
    if a is None:
        raise ApiError("not_found", "Alert not found.")
    return a


@router.post("/{alert_id}/snooze")
@api_endpoint("/api/account/alerts/{alert_id}/snooze")
async def snooze_route(alert_id: int, request: Request):
    """Phase 8 Wave E (WI-P8-ALERTS-SNOOZE): mark an alert as snoozed
    until `body.until` (ISO-8601). Until that timestamp the alert is
    excluded from the unread-count badge and from the default list view.

    Body: { until: "2026-05-27T10:00:00Z" }  OR  { hours: 24 }
    """
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Body must be a JSON object.")
    until_iso = body.get("until")
    hours = body.get("hours")
    if until_iso is None and hours is None:
        raise ApiError("invalid_parameter",
                       "Pass either `until` (ISO timestamp) or `hours` (int).")
    if hours is not None:
        try:
            h = int(hours)
        except (TypeError, ValueError):
            raise ApiError("invalid_parameter", "hours must be an integer.")
        if h < 1 or h > 720:
            raise ApiError("invalid_parameter",
                           "hours must be in [1, 720] (30 days).")
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        until_iso = (_dt.now(_tz.utc) + _td(hours=h)).replace(
            microsecond=0).isoformat()
    a = auth_db.snooze_alert(int(alert_id), int(user["id"]), until_iso)
    if a is None:
        raise ApiError("not_found", "Alert not found.")
    return a
