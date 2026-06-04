"""API Console — admin endpoints (Phase A foundation).

Mounted under ``/api/account/admin``. Because the prefix lives inside
``/api/account/*``, every route here automatically inherits the full console
middleware stack: SessionMiddleware (cookie auth), SudoMiddleware (central
CSRF on unsafe methods + sudo step-up enforcement), token-format rejection,
and token redaction. Admin routes therefore get the same hardening as the
rest of the console "for free", and add ``require_admin`` on top.

Phase A surface (intentionally minimal — metrics, users, and the DB mirror
arrive in later phases):

    GET  /api/account/admin/whoami   report the CALLER's own admin status.

``whoami`` is the one admin route that is NOT gated by ``require_admin``: any
signed-in user may ask "am I an admin?" and receive an honest boolean. It is
used by the account chip to decide whether to surface an "Admin console"
link, and by the admin page's own client to confirm access. It returns only
the caller's own identity — never anyone else's — so it discloses nothing
sensitive. It is a safe (GET) method, so SudoMiddleware does not require a
CSRF header for it; we deliberately do not call ``require_csrf`` so the
endpoint is callable from any authenticated context.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request

from scripts.apin_v2.api_envelope import ApiError, api_endpoint, paginated
from scripts.apin_v2.account._session_helpers import get_session_user, require_csrf
from scripts.apin_v2.account.admin_guard import (
    is_admin, require_admin, require_admin_verified,
)
from scripts.apin_v2.account import admin_queries
from scripts.apin_v2.account import admin_db
from scripts.apin_v2.account import admin_drill
from scripts.apin_v2.account import admin_traffic as admin_traffic_mod  # module (route fn 'admin_traffic' shadows the bare name)

log = logging.getLogger("apin_v2.account.routes_admin")

router = APIRouter(prefix="/api/account/admin", tags=["account/admin"])


def _client_ip(request: Request):
    return request.client.host if request.client else None


@router.get("/whoami")
@api_endpoint("/api/account/admin/whoami")
async def admin_whoami(request: Request):
    """Report whether the signed-in caller is an administrator.

    Requires a valid session (401 otherwise). Returns ONLY the caller's own
    identity + admin flag — no other user is observable through this route.
    """
    user = get_session_user(request)  # 401 invalid_or_missing_token if no session
    # Least-disclosure: return ONLY what the client UI needs (the boolean + a
    # display name/avatar seed). Deliberately omit email and role so the
    # response never reveals WHICH admin signal fired (DB role vs env allowlist)
    # — that internal mechanism detail is not the caller's business.
    return {
        "is_admin": is_admin(user),
        "user_id": user.get("id"),
        "username": user.get("username"),
        "display_name": user.get("display_name"),
        # Public avatar seed (0-5) so the console renders the SAME pressed-leaf
        # identity the rest of the app uses. Not sensitive; email/role still omitted.
        "pressed_leaf_seed": user.get("pressed_leaf_seed"),
    }


# ── Pulse / overview (admin read) ──────────────────────────────────────────
@router.get("/overview")
@api_endpoint("/api/account/admin/overview")
async def admin_overview(request: Request):
    """Org-wide vital signs for the Pulse section. Admin + verified + CSRF."""
    require_csrf(request)
    require_admin_verified(request)   # 404 non-admins · 401 un-elevated admins
    return admin_queries.admin_overview()


# ── Drill: list containers + detail drawers ────────────────────────────────
@router.get("/list/{kind}")
@api_endpoint("/api/account/admin/list/{kind}")
async def admin_list(kind: str, request: Request):
    """Filtered item list for the drill container. kind = requests|keys|scans."""
    require_csrf(request)
    require_admin_verified(request)
    q = request.query_params
    def _i(name, dflt):
        try:
            return int(q.get(name) or dflt)
        except (TypeError, ValueError):
            return dflt
    common = dict(window=(q.get("window") or "all"), bucket=(q.get("bucket") or None),
                  user_id=q.get("user_id") or None, limit=_i("limit", 50), offset=_i("offset", 0))
    if kind == "requests":
        return admin_drill.list_requests(
            endpoint=q.get("endpoint") or None, status=q.get("status") or None,
            key_id=q.get("key_id") or None, ip=q.get("ip") or None, **common)
    if kind == "keys":
        return admin_drill.list_keys(window=common["window"], bucket=common["bucket"],
                                     user_id=common["user_id"], limit=common["limit"], offset=common["offset"])
    if kind == "scans":
        return admin_drill.list_scans(window=common["window"], bucket=common["bucket"],
                                      diagnosis=q.get("diagnosis") or None,
                                      district=q.get("district") or None, crop=q.get("crop") or None,
                                      user_id=common["user_id"], limit=common["limit"], offset=common["offset"])
    if kind == "predictions":
        return admin_drill.list_predictions(window=common["window"], bucket=common["bucket"],
                                            diagnosis=q.get("diagnosis") or None, crop=q.get("crop") or None,
                                            user_id=common["user_id"], limit=common["limit"], offset=common["offset"])
    if kind == "users":
        return admin_drill.list_users(window=common["window"], bucket=common["bucket"],
                                      limit=common["limit"], offset=common["offset"])
    raise ApiError("invalid_parameter", "Unknown list kind.")


@router.get("/detail/request/{row_id}")
@api_endpoint("/api/account/admin/detail/request/{row_id}")
async def admin_detail_request(row_id: int, request: Request):
    require_csrf(request)
    require_admin_verified(request)
    d = admin_drill.detail_request(int(row_id))
    if d is None:
        raise ApiError("not_found", "No such request.")
    return d


@router.get("/detail/key/{public_id}")
@api_endpoint("/api/account/admin/detail/key/{public_id}")
async def admin_detail_key(public_id: str, request: Request):
    require_csrf(request)
    require_admin_verified(request)
    d = admin_drill.detail_key(public_id)
    if d is None:
        raise ApiError("not_found", "No such key.")
    return d


@router.get("/detail/scan/{scan_uid}")
@api_endpoint("/api/account/admin/detail/scan/{scan_uid}")
async def admin_detail_scan(scan_uid: str, request: Request):
    require_csrf(request)
    require_admin_verified(request)
    d = admin_drill.detail_scan(scan_uid)
    if d is None:
        raise ApiError("not_found", "No such scan.")
    return d


@router.get("/detail/user/{user_id}")
@api_endpoint("/api/account/admin/detail/user/{user_id}")
async def admin_detail_user(user_id: int, request: Request):
    require_csrf(request)
    require_admin_verified(request)
    d = admin_drill.detail_user(int(user_id))
    if d is None:
        raise ApiError("not_found", "No such user.")
    return d


@router.get("/detail/prediction/{pred_id}")
@api_endpoint("/api/account/admin/detail/prediction/{pred_id}")
async def admin_detail_prediction(pred_id: str, request: Request):
    require_csrf(request)
    require_admin_verified(request)
    d = admin_drill.detail_prediction(pred_id)
    if d is None:
        raise ApiError("not_found", "No such inference.")
    return d


@router.get("/prediction-image/{pred_id}")
async def admin_prediction_image(pred_id: str, request: Request):
    """Raw uploaded leaf image bytes for a website inference (admin)."""
    from fastapi.responses import Response as _Resp
    require_admin_verified(request)
    data, mime = admin_drill.prediction_image_bytes(pred_id)
    if not data:
        return _Resp(status_code=404)
    return _Resp(content=data, media_type=mime or "image/jpeg",
                 headers={"Cache-Control": "private, max-age=300"})


@router.get("/prediction-heatmap/{pred_id}")
async def admin_prediction_heatmap(pred_id: str, request: Request):
    """Stored Grad-CAM heatmap PNG for a website inference (admin)."""
    from fastapi.responses import Response as _Resp
    require_admin_verified(request)
    data, mime = admin_drill.prediction_heatmap_bytes(pred_id)
    if not data:
        return _Resp(status_code=404)
    return _Resp(content=data, media_type=mime or "image/png",
                 headers={"Cache-Control": "private, max-age=300"})


@router.get("/scan-region/{scan_uid}")
@api_endpoint("/api/account/admin/scan-region/{scan_uid}")
async def admin_scan_region(scan_uid: str, request: Request):
    """District-highlighted region map (SVG-projected paths) for a scan."""
    require_csrf(request)
    require_admin_verified(request)
    d = admin_drill.scan_region(scan_uid)
    if d is None:
        raise ApiError("not_found", "No such scan.")
    return d


@router.get("/scan-image/{scan_uid}")
async def admin_scan_image(scan_uid: str, request: Request):
    """Raw scan image bytes (admin). Not envelope-wrapped — returns the image."""
    from fastapi.responses import Response as _Resp
    require_admin_verified(request)
    data, mime = admin_drill.scan_image_bytes(scan_uid)
    if not data:
        return _Resp(status_code=404)
    return _Resp(content=data, media_type=mime or "image/jpeg",
                 headers={"Cache-Control": "private, max-age=300"})


# ── Per-metric detail (clickable Overview tiles) ───────────────────────────
@router.get("/metric/{metric}")
@api_endpoint("/api/account/admin/metric/{metric}")
async def admin_metric(metric: str, request: Request):
    """Time series + breakdown + headline for one Overview metric. Admin + verified."""
    require_csrf(request)
    require_admin_verified(request)
    window = (request.query_params.get("window") or "all").strip().lower()
    return admin_queries.admin_metric_detail(metric, window=window)


# ── Live activity feed (admin read) ────────────────────────────────────────
@router.get("/events")
@api_endpoint("/api/account/admin/events")
async def admin_events(request: Request):
    """Normalised cross-source event stream for the live feed. Admin + verified."""
    require_csrf(request)
    require_admin_verified(request)
    q = request.query_params
    cat = (q.get("category") or "").strip().lower() or None
    try:
        limit = int(q.get("limit") or 40)
    except (TypeError, ValueError):
        limit = 40
    return admin_queries.admin_events(limit=limit, category=cat)


# ── Database mirror (admin read) ───────────────────────────────────────────
@router.get("/db/tables")
@api_endpoint("/api/account/admin/db/tables")
async def admin_db_tables(request: Request):
    """List every table with row + column counts. Admin + verified + CSRF."""
    require_csrf(request)
    require_admin_verified(request)
    return admin_db.db_list_tables()


@router.get("/db/table")
@api_endpoint("/api/account/admin/db/table")
async def admin_db_table(request: Request):
    """Schema + masked, paginated rows for one table. Admin + verified + CSRF."""
    require_csrf(request)
    require_admin_verified(request)
    q = request.query_params
    name = (q.get("name") or "").strip()
    if not name:
        raise ApiError("invalid_parameter", "name is required.")
    try:
        limit = int(q.get("limit") or 50)
        offset = int(q.get("offset") or 0)
    except (TypeError, ValueError):
        limit, offset = 50, 0
    data = admin_db.db_table(
        name, search=(q.get("search") or "").strip() or None,
        sort=q.get("sort") or None, order=q.get("order") or "asc",
        limit=limit, offset=offset)
    if data is None:
        raise ApiError("not_found", "No such table.")
    return data


# ── Traffic / Website / Geography aggregation (ADM-T · P1) ──────────────────
# One gated dispatch endpoint for the whole Traffic + Geography data layer.
# /api/account/admin/traffic?widget=<name>&window=<24h|7d|30d|all>[&route=…]
# Every widget is org-wide and fail-open in the aggregator, so a bad widget
# name 404s but a data hiccup degrades to an empty shape rather than 500ing.
_TRAFFIC_WIDGETS = None


def _traffic_dispatch():
    global _TRAFFIC_WIDGETS
    if _TRAFFIC_WIDGETS is not None:
        return _TRAFFIC_WIDGETS
    from scripts.apin_v2.account import admin_traffic as T
    _TRAFFIC_WIDGETS = {
        # Traffic › API
        "api_overview":    lambda w, r: T.traffic_api_overview(w),
        "api_terrain":     lambda w, r: T.traffic_api_terrain("7d"),
        "api_methods":     lambda w, r: T.traffic_method_status(w),
        "api_latency":     lambda w, r: T.traffic_latency(w if w != "all" else "30d"),
        "api_bandwidth":   lambda w, r: T.traffic_bandwidth(w if w != "all" else "24h"),
        "api_top":         lambda w, r: T.traffic_top(w),
        "api_endpoints":   lambda w, r: T.traffic_endpoints(w),
        "api_sequences":   lambda w, r: T.traffic_sequences(w if w != "all" else "30d"),
        # Traffic › Website
        "web_overview":    lambda w, r: T.website_overview(w),
        "web_pages":       lambda w, r: T.website_pages(w),
        "web_heatmap":     lambda w, r: T.website_heatmap(r, w),
        "web_devices":     lambda w, r: T.website_devices(w),
        "web_journey":     lambda w, r: T.website_journey(w if w != "all" else "30d"),
        "web_scroll":      lambda w, r: T.website_scroll(w),
        "web_acquisition": lambda w, r: T.website_acquisition(w),
        "web_vitals":      lambda w, r: T.website_vitals(w),
        # Geography (globe-centric: 3 lenses on real scan GPS + API origins)
        "geo":             lambda w, r: T.admin_geo(w),
        "inference_geo":   lambda w, r: T.admin_inference_geo(w),
        "origins":         lambda w, r: T.admin_origins(w),
        # Stats deck — r carries the dataset (api|website) via the route param
        "deck_api":        lambda w, r: T.admin_deck("api", w),
        "deck_website":    lambda w, r: T.admin_deck("website", w),
    }
    return _TRAFFIC_WIDGETS


# Short-TTL in-process cache for traffic widgets. Aggregations over the request
# log / telemetry tables are read-heavy (each widget fans out to several Turso
# round-trips); the underlying data changes slowly relative to a 20s window. This
# makes the live-poll, API↔Website re-entry, and window revisits effectively
# instant while keeping numbers fresh enough to feel live.
_TRAFFIC_CACHE: dict = {}
_TRAFFIC_TTL_S = 20.0


def _traffic_cached(widget: str, window: str, route, fn):
    import time as _t
    key = (widget, window, route or "")
    hit = _TRAFFIC_CACHE.get(key)
    now = _t.monotonic()
    if hit is not None and (now - hit[0]) < _TRAFFIC_TTL_S:
        return hit[1]
    val = fn(window, route)
    _TRAFFIC_CACHE[key] = (now, val)
    # opportunistic prune so the dict can't grow unbounded across windows/routes
    if len(_TRAFFIC_CACHE) > 256:
        for k in [k for k, v in _TRAFFIC_CACHE.items() if (now - v[0]) >= _TRAFFIC_TTL_S]:
            _TRAFFIC_CACHE.pop(k, None)
    return val


@router.get("/traffic")
@api_endpoint("/api/account/admin/traffic")
async def admin_traffic(request: Request):
    require_csrf(request)
    require_admin_verified(request)
    q = request.query_params
    widget = (q.get("widget") or "").strip()
    window = q.get("window") or "all"
    if window not in ("24h", "7d", "30d", "all"):
        window = "all"
    route = (q.get("route") or "").strip() or None
    fn = _traffic_dispatch().get(widget)
    if fn is None:
        raise ApiError("not_found", "Unknown traffic widget.")
    return _traffic_cached(widget, window, route, fn)


@router.get("/traffic/deck-layout")
@api_endpoint("/api/account/admin/traffic/deck-layout")
async def admin_deck_layout_get(request: Request):
    """[STATS DECK F2] Return this admin's saved card order + pins for a dataset."""
    require_csrf(request)
    admin = require_admin_verified(request)
    dataset = (request.query_params.get("dataset") or "api").strip()
    return admin_traffic_mod.get_deck_layout(int(admin["id"]), dataset)


@router.post("/traffic/deck-layout")
@api_endpoint("/api/account/admin/traffic/deck-layout")
async def admin_deck_layout_set(request: Request):
    """[STATS DECK F2] Persist this admin's card order + pins. CSRF via middleware."""
    admin = require_admin_verified(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    dataset = (body.get("dataset") or "api")
    saved = admin_traffic_mod.set_deck_layout(
        int(admin["id"]), dataset, body.get("order"), body.get("pins"))
    return {"ok": True, **saved}


@router.get("/traffic/stream")
async def admin_traffic_stream(request: Request):
    """Org-wide live request feed (Server-Sent Events) for the admin Traffic
    section. Subscribes to the usage bus' BROADCAST channel so it sees every
    recorded request across all keys/users with no extra DB work.

    Auth: require_admin_verified only — the session cookie rides along on the
    EventSource connection. NO CSRF: EventSource can't set request headers, GET
    is read-only and exempt from the central CSRF net (middlewares §846). Each
    frame is its own JSON object (NOT the standard envelope); clients use
    EventSource. Heartbeats every 15s keep proxies from dropping the stream.
    """
    require_admin_verified(request)

    from fastapi.responses import StreamingResponse
    from scripts.apin_v2 import usage_recorder
    import json as _json
    import asyncio as _asyncio

    bus = usage_recorder.get_stream_bus()
    q = bus.subscribe(bus.BROADCAST)

    async def gen():
        try:
            yield "event: ready\ndata: " + _json.dumps({"type": "ready"}) + "\n\n"
            while True:
                try:
                    ev = await _asyncio.wait_for(q.get(), timeout=15.0)
                    yield "data: " + _json.dumps({
                        "type": "request",
                        "path": ev.get("path"),
                        "method": ev.get("method"),
                        "status": ev.get("status_code"),
                        "latency": ev.get("latency_ms"),
                        "bytes_out": ev.get("bytes_out"),
                        "key_id": ev.get("key_id"),
                        "ts": ev.get("timestamp"),
                    }) + "\n\n"
                except _asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except _asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            bus.unsubscribe(bus.BROADCAST, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Users directory (admin read) ───────────────────────────────────────────
@router.get("/users")
@api_endpoint("/api/account/admin/users")
async def admin_users(request: Request):
    require_csrf(request)
    require_admin_verified(request)
    q = request.query_params
    data = admin_queries.admin_list_users(
        search=(q.get("search") or "").strip() or None,
        sort=q.get("sort") or "created",
        order=q.get("order") or "desc",
        limit=int(q.get("limit") or 25),
        offset=int(q.get("offset") or 0),
    )
    page_size = data["limit"] or 25
    page = (data["offset"] // page_size) + 1 if page_size else 1
    return paginated(data["items"], page=page, page_size=page_size, total=data["total"],
                     summary=data.get("summary") or {})


@router.get("/users/{user_id}")
@api_endpoint("/api/account/admin/users/{user_id}")
async def admin_user_detail(user_id: int, request: Request):
    require_csrf(request)
    require_admin_verified(request)
    d = admin_queries.admin_get_user(int(user_id))
    if d is None:
        raise ApiError("not_found", "No such user.")
    return d


# ── Mutations — sudo + CSRF enforced by SudoMiddleware on unsafe methods ────
@router.post("/users/{user_id}/role")
@api_endpoint("/api/account/admin/users/{user_id}/role")
async def admin_set_role(user_id: int, request: Request):
    """Promote / revoke admin. Sudo + CSRF enforced upstream by middleware;
    require_admin_verified here. Blocks any change that would leave zero admins."""
    admin = require_admin_verified(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    new_role = (body or {}).get("role", "")
    try:
        updated = admin_queries.admin_set_user_role(
            target_user_id=int(user_id), new_role=new_role,
            actor_user_id=int(admin["id"]), ip=_client_ip(request))
    except ValueError as e:
        code = str(e)
        if code == "not_found":
            raise ApiError("not_found", "No such user.")
        if code == "last_admin":
            raise ApiError(
                "invalid_parameter",
                "Cannot remove the last administrator — promote another admin first.")
        raise ApiError("invalid_parameter", "role must be 'admin' or 'collector'.")
    return {"ok": True, "user": updated}


@router.post("/maintenance/backfill-telemetry")
@api_endpoint("/api/account/admin/maintenance/backfill-telemetry")
async def admin_backfill_telemetry(request: Request):
    """One-time idempotent attribution of historical anonymous telemetry.
    Sudo + CSRF enforced upstream by middleware; require_admin_verified here."""
    require_admin_verified(request)
    # Exposed route is ADDITIVE-ONLY (attributes NULL→user via guest conversion).
    # The destructive reset=True variant is intentionally NOT reachable over
    # HTTP — it stays a dev-only function argument.
    from scripts.apin_v2 import auth_db as _adb
    counts = _adb.backfill_telemetry_attribution(reset=False)
    return {"ok": True, "attributed": counts}


@router.post("/users/{user_id}/logout-all")
@api_endpoint("/api/account/admin/users/{user_id}/logout-all")
async def admin_force_logout(user_id: int, request: Request):
    """Revoke every active session for the user (force-logout)."""
    admin = require_admin_verified(request)
    n = admin_queries.admin_revoke_user_sessions(
        target_user_id=int(user_id), actor_user_id=int(admin["id"]),
        ip=_client_ip(request))
    return {"ok": True, "sessions_revoked": n}
