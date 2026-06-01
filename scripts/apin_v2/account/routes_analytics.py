"""9.N.T31 · Per-key "Analytics" tab endpoints under /api/account/analytics/*.

These power the inference-observatory widgets (geographic origin map, 3D
inference field, severity spectrum, disease bloom, request stream, quota,
response payloads). They read the scans table (GPS + diagnosis + precomputed
geo_*) and the request log, scoped to the user's keys (+ a pinned key on the
per-key tab). Auth is session-cookie only, same as routes_usage.

Every handler delegates to a fail-open auth_db.compute_analytics_* function,
so an empty database returns a valid empty shape rather than a 500.
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
from fastapi.responses import Response

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import _session_helpers as _sh

log = logging.getLogger("apin_v2.account.routes_analytics")

router = APIRouter(prefix="/api/account/analytics", tags=["account/analytics"])

_get_session_user = _sh.get_session_user

_RANGE_SECONDS = {
    "15m": 15 * 60, "1h": 60 * 60, "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60, "7d": 7 * 24 * 60 * 60, "30d": 30 * 24 * 60 * 60,
    "90d": 90 * 24 * 60 * 60,
}


def _resolve_range(range_str: str) -> int:
    rs = (range_str or "24h").strip().lower()
    if rs not in _RANGE_SECONDS:
        raise ApiError("invalid_parameter",
                       f"unknown range {rs!r}; allowed: {sorted(_RANGE_SECONDS)}")
    return _RANGE_SECONDS[rs]


def _uid(request: Request) -> int:
    return int(_get_session_user(request)["id"])


@router.get("/geo")
@api_endpoint("/api/account/analytics/geo")
async def analytics_geo(request: Request,
                        range: str = Query("24h"),
                        key_id: Optional[str] = Query(None),
                        env: Optional[str] = Query(None),
                        tier: str = Query("state"),
                        source: str = Query("requests"),
                        compare: int = Query(0)):
    # source=requests -> geolocate the request-log client IP (where the API was
    # called from). source=scans -> the GPS attached to /api/scan uploads.
    # compare=1 attaches a previous equal-length window's count per region.
    uid = _uid(request)
    rs = _resolve_range(range)
    cmp = bool(int(compare or 0))
    if source == "scans":
        return auth_db.compute_analytics_geo(
            user_id=uid, key_id=key_id, env=env, range_seconds=rs, tier=tier,
            compare=cmp)
    return auth_db.compute_analytics_geo_requests(
        user_id=uid, key_id=key_id, env=env, range_seconds=rs, tier=tier,
        compare=cmp)


@router.get("/geo/region")
@api_endpoint("/api/account/analytics/geo/region")
async def analytics_geo_region(request: Request,
                               cc: str = Query(...),
                               state: Optional[str] = Query(None),
                               district: Optional[str] = Query(None),
                               range: str = Query("24h"),
                               key_id: Optional[str] = Query(None),
                               env: Optional[str] = Query(None),
                               tier: str = Query("state"),
                               source: str = Query("requests"),
                               compare: int = Query(0)):
    # Single-region dossier for the expanded console's side panel.
    # compare=1 additionally returns prev-window trend + metrics for Compare mode.
    return auth_db.compute_analytics_geo_region(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), tier=tier, source=source,
        cc=cc, state=state, district=district, compare=bool(int(compare or 0)))


@router.get("/geo/region/requests")
@api_endpoint("/api/account/analytics/geo/region/requests")
async def analytics_geo_region_requests(request: Request,
                                        cc: str = Query(...),
                                        state: Optional[str] = Query(None),
                                        district: Optional[str] = Query(None),
                                        range: str = Query("24h"),
                                        key_id: Optional[str] = Query(None),
                                        env: Optional[str] = Query(None),
                                        tier: str = Query("state"),
                                        limit: int = Query(200)):
    # Recent request rows from a single region → the click-a-dot request list.
    return auth_db.compute_analytics_geo_region_requests(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), tier=tier,
        cc=cc, state=state, district=district, limit=max(1, min(300, limit)))


@router.get("/geo/replay")
@api_endpoint("/api/account/analytics/geo/replay")
async def analytics_geo_replay(request: Request,
                               range: str = Query("24h"),
                               key_id: Optional[str] = Query(None),
                               env: Optional[str] = Query(None),
                               tier: str = Query("state"),
                               source: str = Query("requests"),
                               frames: int = Query(120)):
    # Time-bucketed origin events for the replay scrubber.
    return auth_db.compute_analytics_geo_replay(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), tier=tier, source=source,
        frames=max(20, min(240, frames)))


@router.get("/field")
@api_endpoint("/api/account/analytics/field")
async def analytics_field(request: Request,
                          range: str = Query("24h"),
                          key_id: Optional[str] = Query(None),
                          env: Optional[str] = Query(None)):
    return auth_db.compute_analytics_field(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range))


@router.get("/field/scans")
@api_endpoint("/api/account/analytics/field/scans")
async def analytics_field_scans(request: Request,
                                range: str = Query("24h"),
                                key_id: Optional[str] = Query(None),
                                env: Optional[str] = Query(None),
                                crop: Optional[str] = Query(None),
                                disease: Optional[str] = Query(None),
                                limit: int = Query(400)):
    # Individual scan rows for the farm plants + prediction feed.
    return auth_db.compute_analytics_field_scans(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), crop=crop, disease=disease,
        limit=max(1, min(1000, limit)))


@router.get("/scan/{scan_uid}")
@api_endpoint("/api/account/analytics/scan/{scan_uid}")
async def analytics_scan_detail(request: Request, scan_uid: str):
    # Prediction Inspector dossier (click a plant). Ownership-scoped.
    return auth_db.compute_analytics_scan_detail(
        user_id=_uid(request), scan_uid=scan_uid)


@router.get("/scan/{scan_uid}/image")
async def analytics_scan_image(request: Request, scan_uid: str):
    # Raw scan image bytes (ownership-checked) for the inspector <img>. Not
    # JSON-enveloped — returns the image directly so the browser can cache it.
    try:
        uid = _uid(request)
    except Exception:
        return Response(status_code=401)
    data, mime = auth_db.get_scan_image(user_id=uid, scan_uid=scan_uid)
    if not data:
        return Response(status_code=404)
    return Response(content=data, media_type=mime,
                    headers={"Cache-Control": "private, max-age=300"})


@router.get("/severity")
@api_endpoint("/api/account/analytics/severity")
async def analytics_severity(request: Request,
                             range: str = Query("24h"),
                             key_id: Optional[str] = Query(None),
                             env: Optional[str] = Query(None)):
    return auth_db.compute_analytics_severity(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range))


@router.get("/bloom")
@api_endpoint("/api/account/analytics/bloom")
async def analytics_bloom(request: Request,
                          range: str = Query("7d"),
                          key_id: Optional[str] = Query(None),
                          env: Optional[str] = Query(None),
                          buckets: int = Query(28)):
    return auth_db.compute_analytics_bloom(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), buckets=buckets)


@router.get("/streams")
@api_endpoint("/api/account/analytics/streams")
async def analytics_streams(request: Request,
                            range: str = Query("24h"),
                            key_id: Optional[str] = Query(None),
                            env: Optional[str] = Query(None),
                            buckets: int = Query(48)):
    return auth_db.compute_analytics_streams(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), buckets=buckets)


@router.get("/clients")
@api_endpoint("/api/account/analytics/clients")
async def analytics_clients(request: Request,
                            range: str = Query("7d"),
                            key_id: Optional[str] = Query(None),
                            env: Optional[str] = Query(None),
                            buckets: int = Query(24)):
    return auth_db.compute_analytics_clients(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), buckets=buckets)


@router.get("/confidence")
@api_endpoint("/api/account/analytics/confidence")
async def analytics_confidence(request: Request,
                               range: str = Query("7d"),
                               key_id: Optional[str] = Query(None),
                               env: Optional[str] = Query(None),
                               buckets: int = Query(24)):
    return auth_db.compute_analytics_confidence(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), buckets=buckets)


@router.get("/quota")
@api_endpoint("/api/account/analytics/quota")
async def analytics_quota(request: Request,
                          key_id: Optional[str] = Query(None),
                          env: Optional[str] = Query(None)):
    return auth_db.compute_analytics_quota(
        user_id=_uid(request), key_id=key_id, env=env)


@router.get("/payloads")
@api_endpoint("/api/account/analytics/payloads")
async def analytics_payloads(request: Request,
                             range: str = Query("24h"),
                             key_id: Optional[str] = Query(None),
                             env: Optional[str] = Query(None),
                             sample: int = Query(20)):
    return auth_db.compute_analytics_payloads(
        user_id=_uid(request), key_id=key_id, env=env,
        range_seconds=_resolve_range(range), sample=max(1, min(100, sample)))
