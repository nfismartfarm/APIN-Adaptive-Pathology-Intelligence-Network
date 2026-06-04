"""Admin drill-down data layer (admin only) — list containers + rich detail.

Every Overview tile drills:  tile → metric lightbox → LIST container (filtered
items) → DETAIL drawer (one request / key / scan / user).

Design tenet — IDENTICAL CONTENT, ORG-WIDE. The per-user console already has
battle-tested builders (request detail, scan dossier, key usage, …) but they are
all ``user_id``-scoped. Rather than re-implement them (and risk drift), each
admin detail here first resolves the entity's OWNER, then calls the very same
builder. So an admin sees exactly what the user would, for any entity.

Everything is fail-open (returns an empty-but-valid shape) so the console never
500s on a malformed drill.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import admin_queries

log = logging.getLogger("apin_v2.account.admin_drill")


def _scalar(c, sql, args=()):
    row = c.execute(sql, args).fetchone()
    if not row:
        return 0
    return int(list(dict(row).values())[0] or 0)


def _bucket_bounds(bucket: Optional[str]):
    """A chart bucket key → [start_iso, end_iso). Daily 'YYYY-MM-DD' or hourly
    'YYYY-MM-DDTHH'. Returns (None, None) if not a bucket (caller uses window)."""
    if not bucket:
        return None, None
    try:
        if len(bucket) >= 13:
            start = datetime.strptime(bucket[:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            end = start + timedelta(hours=1)
        else:
            start = datetime.strptime(bucket[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = start + timedelta(days=1)
        return start.isoformat(), end.isoformat()
    except Exception:
        return None, None


def _pctile(c, where, args, col="latency_ms", p=0.5):
    """Percentile of a column over a filtered set, via ORDER BY + OFFSET."""
    n = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log r LEFT JOIN api_keys k "
                   "ON k.public_id = r.key_id" + where + (" AND " if where else " WHERE ")
                   + ("r.%s IS NOT NULL" % col), args)
    if n <= 0:
        return None
    off = min(n - 1, int(n * p))
    row = c.execute(
        "SELECT r.%s v FROM api_key_request_log r LEFT JOIN api_keys k ON k.public_id = r.key_id"
        % col + where + (" AND " if where else " WHERE ") + "r.%s IS NOT NULL ORDER BY r.%s LIMIT 1 OFFSET ?"
        % (col, col), tuple(args) + (off,)).fetchone()
    return int(dict(row)["v"]) if row else None


# ── LIST · requests ─────────────────────────────────────────────────────────
def list_requests(*, window="all", bucket=None, endpoint=None, status=None,
                  key_id=None, ip=None, user_id=None, limit=50, offset=0) -> dict:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    out = {"items": [], "total": 0, "stats": {}, "title": "", "filters": {}}
    try:
        where, args = [], []
        bstart, bend = _bucket_bounds(bucket)
        if bstart:
            where.append("r.timestamp >= ? AND r.timestamp < ?"); args += [bstart, bend]
        else:
            cut = admin_queries._window_cutoff(window)
            if cut:
                where.append("r.timestamp >= ?"); args.append(cut)
        if endpoint:
            where.append("r.path = ?"); args.append(endpoint)
        if key_id:
            where.append("r.key_id = ?"); args.append(key_id)
        if ip:
            where.append("r.ip = ?"); args.append(ip)
        if user_id:
            where.append("k.user_id = ?"); args.append(int(user_id))
        if status:
            st = str(status).lower()
            if st in ("2xx", "3xx", "4xx", "5xx"):
                lo = int(st[0]) * 100
                where.append("r.status_code >= ? AND r.status_code < ?"); args += [lo, lo + 100]
            elif st == "error":
                where.append("r.status_code >= 400")
            elif st.isdigit():
                where.append("r.status_code = ?"); args.append(int(st))
        wsql = (" WHERE " + " AND ".join(where)) if where else ""
        with auth_db.get_conn() as c:
            out["total"] = _scalar(
                c, "SELECT COUNT(*) FROM api_key_request_log r "
                   "LEFT JOIN api_keys k ON k.public_id = r.key_id" + wsql, args)
            errs = _scalar(
                c, "SELECT COUNT(*) FROM api_key_request_log r "
                   "LEFT JOIN api_keys k ON k.public_id = r.key_id" + wsql
                   + (" AND " if wsql else " WHERE ") + "r.status_code >= 400", args)
            avg = c.execute("SELECT AVG(r.latency_ms) a FROM api_key_request_log r "
                            "LEFT JOIN api_keys k ON k.public_id = r.key_id" + wsql, args).fetchone()
            out["stats"] = {
                "total": out["total"], "errors": errs,
                "error_rate": round(100.0 * errs / out["total"], 1) if out["total"] else 0.0,
                "avg_latency": int(dict(avg)["a"] or 0) if avg and dict(avg)["a"] is not None else 0,
                "p50": _pctile(c, wsql, args, "latency_ms", 0.5),
                "p95": _pctile(c, wsql, args, "latency_ms", 0.95),
            }
            for r in c.execute(
                "SELECT r.id, r.timestamp, r.method, r.path, r.status_code, r.latency_ms, "
                "r.bytes_in, r.bytes_out, r.key_id, k.name kn, k.user_id uid, "
                "COALESCE(u.display_name, u.username) un "
                "FROM api_key_request_log r LEFT JOIN api_keys k ON k.public_id = r.key_id "
                "LEFT JOIN users u ON u.id = k.user_id" + wsql
                + " ORDER BY r.timestamp DESC LIMIT ? OFFSET ?", tuple(args) + (limit, offset)):
                d = dict(r)
                out["items"].append({
                    "id": d["id"], "ts": d.get("timestamp"), "method": d.get("method"),
                    "path": d.get("path"), "status": d.get("status_code"),
                    "latency_ms": d.get("latency_ms"),
                    "bytes_in": d.get("bytes_in"), "bytes_out": d.get("bytes_out"),
                    "key_name": d.get("kn") or d.get("key_id"), "owner": d.get("un"),
                })
        out["filters"] = {"window": window, "bucket": bucket, "endpoint": endpoint,
                          "status": status, "key_id": key_id, "user_id": user_id}
    except Exception as e:  # noqa: BLE001
        log.warning("list_requests failed: %s", e)
    return out


# ── LIST · keys ─────────────────────────────────────────────────────────────
def list_keys(*, window="all", bucket=None, user_id=None, limit=60, offset=0) -> dict:
    limit = max(1, min(int(limit or 60), 200))
    offset = max(0, int(offset or 0))
    out = {"items": [], "total": 0, "stats": {}}
    cut = admin_queries._window_cutoff(window)
    bstart, bend = _bucket_bounds(bucket)
    try:
        with auth_db.get_conn() as c:
            uwhere = " WHERE k.deleted_at IS NULL"
            uargs = []
            if user_id:
                uwhere += " AND k.user_id = ?"; uargs.append(int(user_id))
            out["total"] = _scalar(c, "SELECT COUNT(*) FROM api_keys k" + uwhere, uargs)
            # per-key request filter
            if bstart:
                rfilter = "AND r.timestamp >= '%s' AND r.timestamp < '%s'" % (bstart, bend)
            elif cut:
                rfilter = "AND r.timestamp >= '%s'" % cut
            else:
                rfilter = ""
            for k in c.execute(
                "SELECT k.public_id, k.name, k.status, k.user_id, k.created_at, "
                "k.group_id, g.name gname, COALESCE(u.display_name,u.username) owner, "
                "(SELECT COUNT(*) FROM api_key_request_log r WHERE r.key_id = k.public_id %s) reqs, "
                "(SELECT COUNT(*) FROM api_key_request_log r WHERE r.key_id = k.public_id AND r.status_code>=400 %s) errs, "
                "(SELECT MAX(r.timestamp) FROM api_key_request_log r WHERE r.key_id = k.public_id) last_used "
                "FROM api_keys k LEFT JOIN api_key_groups g ON g.id = k.group_id "
                "LEFT JOIN users u ON u.id = k.user_id" % (rfilter, rfilter)
                + uwhere + " ORDER BY reqs DESC LIMIT ? OFFSET ?", tuple(uargs) + (limit, offset)):
                d = dict(k)
                reqs = int(d.get("reqs") or 0)
                out["items"].append({
                    "public_id": d["public_id"], "name": d.get("name") or d["public_id"],
                    "status": d.get("status"), "group": d.get("gname"), "owner": d.get("owner"),
                    "requests": reqs, "errors": int(d.get("errs") or 0),
                    "error_rate": round(100.0 * int(d.get("errs") or 0) / reqs, 1) if reqs else 0.0,
                    "last_used": d.get("last_used"), "created_at": d.get("created_at"),
                })
    except Exception as e:  # noqa: BLE001
        log.warning("list_keys failed: %s", e)
    return out


# ── LIST · scans ────────────────────────────────────────────────────────────
def list_scans(*, window="all", bucket=None, diagnosis=None, district=None, crop=None, user_id=None, limit=60, offset=0) -> dict:
    limit = max(1, min(int(limit or 60), 200))
    offset = max(0, int(offset or 0))
    out = {"items": [], "total": 0}
    try:
        with auth_db.get_conn() as c:
            stc = admin_queries._scans_time_col(c)
            where = ["s.deleted_at IS NULL"]
            args = []
            bstart, bend = _bucket_bounds(bucket)
            if bstart:
                where.append('s."%s" >= ? AND s."%s" < ?' % (stc, stc)); args += [bstart, bend]
            else:
                cut = admin_queries._window_cutoff(window)
                if cut:
                    where.append('s."%s" >= ?' % stc); args.append(cut)
            if diagnosis:
                where.append("s.diagnosis = ?"); args.append(diagnosis)
            if district:
                where.append("s.geo_district = ?"); args.append(district)
            if crop:
                # crop bucket → diagnosis prefix (okra_* / brassica_* / tomato_*)
                where.append("s.diagnosis LIKE ?"); args.append(str(crop).lower() + "%")
            if user_id:
                where.append("s.user_id = ?"); args.append(int(user_id))
            wsql = " WHERE " + " AND ".join(where)
            out["total"] = _scalar(
                c, "SELECT COUNT(*) FROM scans s" + wsql, args)
            for r in c.execute(
                'SELECT s.scan_uid, s.diagnosis, s.confidence, s.severity, s.tier, s.is_ood, '
                's.image_sha256, s.processing_ms, s."%s" pts, s.user_id, '
                'COALESCE(u.display_name,u.username) owner '
                'FROM scans s LEFT JOIN users u ON u.id = s.user_id' % stc + wsql
                + ' ORDER BY s."%s" DESC LIMIT ? OFFSET ?' % stc, tuple(args) + (limit, offset)):
                d = dict(r)
                out["items"].append({
                    "scan_uid": d["scan_uid"], "diagnosis": d.get("diagnosis"),
                    "confidence": d.get("confidence"), "severity": d.get("severity"),
                    "tier": d.get("tier"), "is_ood": int(d.get("is_ood") or 0) == 1,
                    "has_image": bool(d.get("image_sha256")),
                    "processing_ms": d.get("processing_ms"), "ts": d.get("pts"),
                    "owner": d.get("owner"),
                })
    except Exception as e:  # noqa: BLE001
        log.warning("list_scans failed: %s", e)
    return out


# ── LIST · users ────────────────────────────────────────────────────────────
def list_users(*, window="all", bucket=None, limit=60, offset=0) -> dict:
    """Users, optionally filtered to those who SIGNED UP in a window/bucket
    (the Users tile's bars are daily signups). Reuses admin_list_users' shape
    where possible but adds the date filter."""
    limit = max(1, min(int(limit or 60), 200))
    offset = max(0, int(offset or 0))
    out = {"items": [], "total": 0}
    try:
        where, args = [], []
        bstart, bend = _bucket_bounds(bucket)
        if bstart:
            where.append("u.created_at >= ? AND u.created_at < ?"); args += [bstart, bend]
        else:
            cut = admin_queries._window_cutoff(window)
            if cut:
                where.append("u.created_at >= ?"); args.append(cut)
        wsql = (" WHERE " + " AND ".join(where)) if where else ""
        with auth_db.get_conn() as c:
            out["total"] = _scalar(c, "SELECT COUNT(*) FROM users u" + wsql, args)
            for r in c.execute(
                "SELECT u.id, u.display_name, u.username, u.email, u.role, u.created_at, "
                "u.last_seen_at, u.pressed_leaf_seed, "
                "(SELECT COUNT(*) FROM api_keys k WHERE k.user_id = u.id AND k.deleted_at IS NULL) keys, "
                "(SELECT COUNT(*) FROM api_key_request_log r WHERE r.key_id IN "
                "  (SELECT public_id FROM api_keys k2 WHERE k2.user_id = u.id)) reqs "
                "FROM users u" + wsql + " ORDER BY u.created_at DESC LIMIT ? OFFSET ?",
                tuple(args) + (limit, offset)):
                d = dict(r)
                out["items"].append({
                    "id": d["id"], "display_name": d.get("display_name"),
                    "username": d.get("username"), "email": d.get("email"),
                    "role": d.get("role"), "created_at": d.get("created_at"),
                    "last_seen_at": d.get("last_seen_at"),
                    "pressed_leaf_seed": d.get("pressed_leaf_seed"),
                    "keys": int(d.get("keys") or 0), "requests": int(d.get("reqs") or 0),
                })
    except Exception as e:  # noqa: BLE001
        log.warning("list_users failed: %s", e)
    return out


# ── DETAIL · request (reuse console builder, org-wide) ──────────────────────
def detail_request(row_id: int) -> Optional[dict]:
    try:
        with auth_db.get_conn() as c:
            o = c.execute("SELECT k.user_id FROM api_key_request_log r "
                          "JOIN api_keys k ON k.public_id = r.key_id WHERE r.id = ?",
                          (int(row_id),)).fetchone()
        if not o:
            return None
        uid = int(dict(o)["user_id"])
        row = auth_db.get_request_log_row(user_id=uid, row_id=int(row_id))
        if not row:
            return None
        from scripts.apin_v2.account.routes_usage import build_request_detail_payload
        payload = build_request_detail_payload(uid, row)
        # admin context: who owns the key that made this call
        try:
            u = admin_queries.admin_get_user(uid)
            payload["owner"] = {"id": uid, "display_name": (u or {}).get("display_name"),
                                "username": (u or {}).get("username"), "email": (u or {}).get("email")}
        except Exception:
            payload["owner"] = {"id": uid}
        return payload
    except Exception as e:  # noqa: BLE001
        log.warning("detail_request(%s) failed: %s", row_id, e)
        return None


# ── DETAIL · scan (reuse console dossier, org-wide) ─────────────────────────
def detail_scan(scan_uid: str) -> Optional[dict]:
    try:
        with auth_db.get_conn() as c:
            o = c.execute("SELECT user_id FROM scans WHERE scan_uid = ?", (scan_uid,)).fetchone()
        if not o:
            return None
        uid = int(dict(o)["user_id"])
        d = auth_db.compute_analytics_scan_detail(user_id=uid, scan_uid=scan_uid)
        if not d or not d.get("found"):
            return d or {"found": False}
        try:
            u = admin_queries.admin_get_user(uid)
            d["owner"] = {"id": uid, "display_name": (u or {}).get("display_name"),
                          "username": (u or {}).get("username")}
        except Exception:
            d["owner"] = {"id": uid}
        return d
    except Exception as e:  # noqa: BLE001
        log.warning("detail_scan(%s) failed: %s", scan_uid, e)
        return None


def scan_image_bytes(scan_uid: str):
    """(bytes, mime) for a scan image, owner resolved. (None, None) if missing."""
    try:
        with auth_db.get_conn() as c:
            o = c.execute("SELECT user_id FROM scans WHERE scan_uid = ?", (scan_uid,)).fetchone()
        if not o:
            return None, None
        return auth_db.get_scan_image(user_id=int(dict(o)["user_id"]), scan_uid=scan_uid)
    except Exception:
        return None, None


# ── DETAIL · scan REGION map (district highlighted within its locale) ────────
# The scan dossier's Location pane shows the STATE/region with the scan's
# DISTRICT highlighted (not the whole country). We render this server-side from
# geo_district_IN.json so the 735-feature, multi-MB boundary file never reaches
# the browser: we pick the target district + its near neighbours, decimate the
# rings, project them to a fixed SVG viewBox, and ship a few ready-to-draw paths.
import json as _json_geo
import os as _os_geo

_GEO_DISTRICTS = None          # [{name, lower, rings:[[(lon,lat)..]], clon, clat}]
_GEO_PROJ_W = 480
_GEO_PROJ_H = 360


def _load_districts():
    global _GEO_DISTRICTS
    if _GEO_DISTRICTS is not None:
        return _GEO_DISTRICTS
    out = []
    try:
        path = _os_geo.path.join(_os_geo.path.dirname(_os_geo.path.dirname(_os_geo.path.abspath(__file__))),
                                 "geo_district_IN.json")
        with open(path, "r", encoding="utf-8") as f:
            gj = _json_geo.load(f)
        for ft in gj.get("features", []):
            nm = (ft.get("properties", {}) or {}).get("shapeName") or ""
            geom = ft.get("geometry", {}) or {}
            gt, coords = geom.get("type"), geom.get("coordinates")
            rings = []
            if gt == "Polygon":
                rings = coords or []
            elif gt == "MultiPolygon":
                # flatten outer rings of each polygon part
                for poly in (coords or []):
                    if poly:
                        rings.append(poly[0])
            if not rings:
                continue
            # centroid from first ring (good enough for neighbour search)
            r0 = rings[0]
            cx = sum(p[0] for p in r0) / len(r0)
            cy = sum(p[1] for p in r0) / len(r0)
            out.append({"name": nm, "lower": nm.lower().strip(),
                        "rings": rings, "clon": cx, "clat": cy})
    except Exception as e:  # noqa: BLE001
        log.warning("district geojson load failed: %s", e)
        out = []
    _GEO_DISTRICTS = out
    return out


def _decimate(ring, max_pts=70):
    n = len(ring)
    if n <= max_pts:
        return ring
    step = n / float(max_pts)
    return [ring[int(i * step)] for i in range(max_pts)]


def scan_region(scan_uid: str) -> Optional[dict]:
    """District-highlighted region map for a scan. Returns SVG-projected paths.
    Fail-open: returns {found:False, ...labels} so the pane shows a graceful
    fallback (labels + GPS) when boundaries are unavailable."""
    try:
        with auth_db.get_conn() as c:
            row = c.execute(
                "SELECT geo_state, geo_district, geo_cc, latitude, longitude "
                "FROM scans WHERE scan_uid = ?", (scan_uid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        state = d.get("geo_state"); district = d.get("geo_district")
        lat = d.get("latitude"); lon = d.get("longitude")
        base = {"found": False, "state": state, "district": district,
                "cc": d.get("geo_cc"), "lat": lat, "lon": lon}
        feats = _load_districts()
        if not feats or not district:
            return base
        dl = str(district).lower().strip()
        target = next((f for f in feats if f["lower"] == dl), None)
        if target is None:  # fuzzy: contains / startswith
            target = next((f for f in feats if dl in f["lower"] or f["lower"] in dl), None)
        if target is None:
            return base
        # neighbours within a box around the target centroid
        span = 1.7
        neigh = [f for f in feats
                 if f is not target
                 and abs(f["clon"] - target["clon"]) <= span
                 and abs(f["clat"] - target["clat"]) <= span]
        # bbox over target + neighbours
        allrings = [target["rings"]] + [f["rings"] for f in neigh]
        xs, ys = [], []
        for rings in allrings:
            for ring in rings:
                for p in ring:
                    xs.append(p[0]); ys.append(p[1])
        if not xs:
            return base
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        # pad 4%
        padx = (maxx - minx) * 0.04 or 0.1
        pady = (maxy - miny) * 0.04 or 0.1
        minx -= padx; maxx += padx; miny -= pady; maxy += pady
        sx = _GEO_PROJ_W / (maxx - minx) if maxx > minx else 1
        sy = _GEO_PROJ_H / (maxy - miny) if maxy > miny else 1
        sc = min(sx, sy)
        ox = (_GEO_PROJ_W - (maxx - minx) * sc) / 2
        oy = (_GEO_PROJ_H - (maxy - miny) * sc) / 2

        def proj(lonp, latp):
            x = ox + (lonp - minx) * sc
            y = oy + (maxy - latp) * sc      # flip y (north up)
            return round(x, 1), round(y, 1)

        def path_of(rings, decim):
            parts = []
            for ring in rings:
                rr = _decimate(ring, decim) if decim else ring
                if len(rr) < 3:
                    continue
                pts = [proj(p[0], p[1]) for p in rr]
                parts.append("M" + " L".join("%s,%s" % (x, y) for x, y in pts) + "Z")
            return " ".join(parts)

        marker = None
        if lat is not None and lon is not None:
            mx, my = proj(lon, lat)
            marker = {"x": mx, "y": my}
        return {
            "found": True, "state": state, "district": target["name"], "cc": d.get("geo_cc"),
            "lat": lat, "lon": lon, "viewbox": "0 0 %d %d" % (_GEO_PROJ_W, _GEO_PROJ_H),
            "target_path": path_of(target["rings"], 90),
            "neighbour_paths": [path_of(f["rings"], 38) for f in neigh][:40],
            "neighbour_count": len(neigh), "marker": marker,
        }
    except Exception as e:  # noqa: BLE001
        log.warning("scan_region(%s) failed: %s", scan_uid, e)
        return {"found": False}


# ── DETAIL · key (reuse usage builders, org-wide) ───────────────────────────
_RANGE_ALL = 366 * 24 * 3600


def detail_key(public_id: str) -> Optional[dict]:
    import json as _json
    try:
        with auth_db.get_conn() as c:
            k = c.execute(
                "SELECT k.public_id, k.name, k.status, k.user_id, k.environment, "
                "k.scopes, k.created_at, k.group_id, g.name gname, "
                "COALESCE(u.display_name,u.username) owner, u.id ouid "
                "FROM api_keys k LEFT JOIN api_key_groups g ON g.id = k.group_id "
                "LEFT JOIN users u ON u.id = k.user_id WHERE k.public_id = ?", (public_id,)).fetchone()
            if not k:
                return None
            kd = dict(k)
            uid = int(kd["user_id"])
            # scopes (json or csv)
            scopes = []
            sc = kd.get("scopes")
            if sc:
                try:
                    scopes = _json.loads(sc) if sc.strip().startswith("[") else [s for s in sc.split(",") if s]
                except Exception:
                    scopes = [s for s in str(sc).split(",") if s]
            # lifecycle (audit) + recent requests
            lifecycle = []
            for r in c.execute("SELECT action, timestamp, key_name_at_time FROM api_key_audit "
                               "WHERE key_id = ? ORDER BY id DESC LIMIT 12", (public_id,)):
                d = dict(r); lifecycle.append({"action": d.get("action"), "ts": d.get("timestamp")})
            recent = []
            for r in c.execute("SELECT id, timestamp, method, path, status_code, latency_ms "
                               "FROM api_key_request_log WHERE key_id = ? ORDER BY id DESC LIMIT 8", (public_id,)):
                d = dict(r); recent.append({"id": d["id"], "ts": d.get("timestamp"), "method": d.get("method"),
                                            "path": d.get("path"), "status": d.get("status_code"),
                                            "latency_ms": d.get("latency_ms")})
            total = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log WHERE key_id = ?", (public_id,))
            errs = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log WHERE key_id = ? AND status_code >= 400", (public_id,))
            # daily series (45d) + status mix + top endpoints
            series = admin_queries._series_for(c, "api_key_request_log", "timestamp", "30d",
                                               "AND key_id = '%s'" % public_id.replace("'", "''"))
            status_mix = []
            for r in c.execute("SELECT (status_code/100)||'xx' band, COUNT(*) v FROM api_key_request_log "
                               "WHERE key_id = ? GROUP BY band ORDER BY v DESC", (public_id,)):
                d = dict(r); status_mix.append({"label": d.get("band"), "value": int(d.get("v") or 0)})
            top_eps = []
            for r in c.execute("SELECT path label, COUNT(*) v FROM api_key_request_log WHERE key_id = ? "
                               "GROUP BY path ORDER BY v DESC LIMIT 6", (public_id,)):
                d = dict(r); top_eps.append({"label": d.get("label"), "value": int(d.get("v") or 0)})
            p50 = _pctile(c, " WHERE r.key_id = ?", (public_id,), "latency_ms", 0.5)
            p95 = _pctile(c, " WHERE r.key_id = ?", (public_id,), "latency_ms", 0.95)
            org_total = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log") or 1
            _lu = c.execute("SELECT MAX(timestamp) lu FROM api_key_request_log WHERE key_id = ?",
                            (public_id,)).fetchone()
            last_used = dict(_lu).get("lu") if _lu else None
        return {
            "public_id": public_id, "name": kd.get("name") or public_id, "status": kd.get("status"),
            "environment": kd.get("environment"), "group": kd.get("gname"), "scopes": scopes,
            "owner": {"id": uid, "display_name": kd.get("owner")},
            "created_at": kd.get("created_at"), "last_used": last_used,
            "in_use": bool(total),
            "stats": {"requests": total, "errors": errs,
                      "error_rate": round(100.0 * errs / total, 1) if total else 0.0,
                      "p50": p50, "p95": p95,
                      "org_share": round(100.0 * total / org_total, 1)},
            "series": series, "status_mix": status_mix, "top_endpoints": top_eps,
            "recent": recent, "lifecycle": lifecycle,
        }
    except Exception as e:  # noqa: BLE001
        log.warning("detail_key(%s) failed: %s", public_id, e)
        return None


# ── DETAIL · user (account + engagement + usage + inferences + location) ────
def user_engagement(user_id: int) -> dict:
    """Product-analytics rollup for one user from page_views + clicks + events."""
    uid = int(user_id)
    out = {"total_active_ms": 0, "total_idle_ms": 0, "sessions": 0, "page_views": 0,
           "avg_scroll": 0.0, "bounce_rate": 0.0, "clicks": 0, "rage_clicks": 0,
           "dead_clicks": 0, "pages": [], "heatmap": [], "timeline": [],
           "web_vitals": {}, "engagement_score": None}
    try:
        with auth_db.get_conn() as c:
            pv_cols = {dict(r)["name"] for r in c.execute("PRAGMA table_info(page_views)")}
            if "user_id" not in pv_cols:
                return out
            agg = c.execute(
                "SELECT COUNT(*) pv, COUNT(DISTINCT browser_session_id) sess, "
                "SUM(active_duration_ms) act, SUM(idle_duration_ms) idle, "
                "AVG(max_scroll_depth_pct) scroll, AVG(bounce) bounce, SUM(click_count) clk, "
                "AVG(engagement_score) eng, AVG(lcp_ms) lcp, AVG(cls) cls, AVG(inp_ms) inp "
                "FROM page_views WHERE user_id = ? AND deleted_at IS NULL", (uid,)).fetchone()
            a = dict(agg) if agg else {}
            out["page_views"] = int(a.get("pv") or 0)
            out["sessions"] = int(a.get("sess") or 0)
            out["total_active_ms"] = int(a.get("act") or 0)
            out["total_idle_ms"] = int(a.get("idle") or 0)
            out["avg_scroll"] = round(float(a.get("scroll") or 0), 1)
            out["bounce_rate"] = round(100.0 * float(a.get("bounce") or 0), 1)
            out["clicks"] = int(a.get("clk") or 0)
            out["engagement_score"] = round(float(a["eng"]), 1) if a.get("eng") is not None else None
            out["web_vitals"] = {
                "lcp_ms": int(a["lcp"]) if a.get("lcp") is not None else None,
                "cls": round(float(a["cls"]), 3) if a.get("cls") is not None else None,
                "inp_ms": int(a["inp"]) if a.get("inp") is not None else None,
            }
            # per-page rollup
            for r in c.execute(
                "SELECT page_route route, page_title title, COUNT(*) visits, "
                "SUM(active_duration_ms) act, SUM(click_count) clk, AVG(max_scroll_depth_pct) scroll "
                "FROM page_views WHERE user_id = ? AND deleted_at IS NULL "
                "GROUP BY page_route ORDER BY act DESC LIMIT 12", (uid,)):
                d = dict(r)
                out["pages"].append({"route": d.get("route"), "title": d.get("title"),
                                     "visits": int(d.get("visits") or 0),
                                     "active_ms": int(d.get("act") or 0),
                                     "clicks": int(d.get("clk") or 0),
                                     "scroll": round(float(d.get("scroll") or 0), 0)})
            # activity heatmap hour(0-23) × weekday(0-6) from entered_at
            grid = [[0] * 24 for _ in range(7)]
            for r in c.execute("SELECT entered_at FROM page_views WHERE user_id = ? AND deleted_at IS NULL", (uid,)):
                ts = dict(r).get("entered_at")
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    grid[dt.weekday()][dt.hour] += 1
                except Exception:
                    pass
            out["heatmap"] = grid
            # recent activity timeline
            for r in c.execute(
                "SELECT page_route route, entered_at, active_duration_ms act, click_count clk "
                "FROM page_views WHERE user_id = ? AND deleted_at IS NULL "
                "ORDER BY entered_at DESC LIMIT 14", (uid,)):
                d = dict(r)
                out["timeline"].append({"route": d.get("route"), "ts": d.get("entered_at"),
                                        "active_ms": int(d.get("act") or 0), "clicks": int(d.get("clk") or 0)})
            # rage / dead clicks
            try:
                cl = c.execute("SELECT SUM(was_rage_click) rage, SUM(was_dead_click) dead "
                               "FROM clicks WHERE user_id = ?", (uid,)).fetchone()
                cd = dict(cl) if cl else {}
                out["rage_clicks"] = int(cd.get("rage") or 0)
                out["dead_clicks"] = int(cd.get("dead") or 0)
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        log.warning("user_engagement(%s) failed: %s", user_id, e)
    return out


# ── PREDICTIONS · real website inferences (predictions + guest_predictions) ──
# The admin "Inferences" surface reads these — the actual inferences run on the
# site (`/`), each carrying its stored Grad-CAM heatmap + full monograph result —
# NOT the geo `scans` table (that stays for Geography/Observatory). Logged-in
# inferences live in `predictions` (id prefixed "p"); guest inferences in
# `guest_predictions` (prefixed "g") so the two id-spaces never collide.
import json as _json_pred


def _pred_owner(c, uid):
    try:
        if not uid:
            return None
        u = admin_queries.admin_get_user(int(uid))
        return {"id": int(uid), "display_name": (u or {}).get("display_name"),
                "username": (u or {}).get("username")}
    except Exception:
        return {"id": uid}


def _pred_table(pid):
    """('predictions'|'guest_predictions', numeric_id) from a prefixed id."""
    s = str(pid or "")
    if s[:1] == "g":
        return "guest_predictions", s[1:]
    if s[:1] == "p":
        return "predictions", s[1:]
    return "predictions", s  # legacy bare-numeric → predictions


def _pj(v):
    if v is None or v == "":
        return None
    if not isinstance(v, str):
        return v
    try:
        return _json_pred.loads(v)
    except Exception:
        return None


def list_predictions(*, window="all", bucket=None, diagnosis=None, crop=None,
                     user_id=None, limit=60, offset=0) -> dict:
    limit = max(1, min(int(limit or 60), 200))
    offset = max(0, int(offset or 0))
    out = {"items": [], "total": 0, "stats": {}}
    bstart, bend = _bucket_bounds(bucket)
    cut = admin_queries._window_cutoff(window)
    try:
        with auth_db.get_conn() as c:
            # per-user → predictions only (guests have no user_id);
            # org-wide → predictions ∪ guest_predictions.
            sources = [("predictions", True)]
            if not user_id:
                sources.append(("guest_predictions", False))
            merged, total, ood_n, conf_sum, conf_n = [], 0, 0, 0.0, 0
            for table, has_user in sources:
                where = ["deleted_at IS NULL"]
                args = []
                if bstart:
                    where.append("created_at >= ? AND created_at < ?"); args += [bstart, bend]
                elif cut:
                    where.append("created_at >= ?"); args.append(cut)
                if diagnosis:
                    where.append("predicted_class = ?"); args.append(diagnosis)
                if crop:
                    where.append("crop = ?"); args.append(crop)
                if user_id and has_user:
                    where.append("user_id = ?"); args.append(int(user_id))
                wsql = " WHERE " + " AND ".join(where)
                total += _scalar(c, "SELECT COUNT(*) FROM %s%s" % (table, wsql), args)
                ood_n += _scalar(c, "SELECT COUNT(*) FROM %s%s AND ood_flag = 1" % (table, wsql), args)
                cs = c.execute("SELECT AVG(confidence) a, COUNT(confidence) n FROM %s%s" % (table, wsql), args).fetchone()
                if cs:
                    cd = dict(cs)
                    if cd.get("a") is not None:
                        conf_sum += float(cd["a"]) * int(cd.get("n") or 0); conf_n += int(cd.get("n") or 0)
                uid_sel = "user_id" if has_user else "NULL user_id"
                pre = "p" if has_user else "g"
                for r in c.execute(
                        "SELECT id, predicted_class, confidence, tier, crop, ood_flag, created_at, "
                        "(heatmap_b64 IS NOT NULL) hm, %s FROM %s%s "
                        "ORDER BY created_at DESC LIMIT ?" % (uid_sel, table, wsql),
                        tuple(args) + (limit + offset,)):
                    d = dict(r)
                    merged.append({
                        "id": pre + str(d["id"]), "diagnosis": d.get("predicted_class"),
                        "confidence": d.get("confidence"), "tier": d.get("tier"),
                        "crop": d.get("crop"), "is_ood": int(d.get("ood_flag") or 0),
                        "ts": d.get("created_at"), "has_heatmap": bool(d.get("hm")),
                        "guest": not has_user,
                        "owner": _pred_owner(c, d.get("user_id")),
                    })
            merged.sort(key=lambda x: x.get("ts") or "", reverse=True)
            out["items"] = merged[offset:offset + limit]
            out["total"] = total
            out["stats"] = {
                "total": total, "ood": ood_n,
                "ood_rate": round(100.0 * ood_n / total, 1) if total else 0.0,
                "avg_confidence": round(100.0 * conf_sum / conf_n, 1) if conf_n else None,
            }
    except Exception as e:  # noqa: BLE001
        log.warning("list_predictions failed: %s", e)
    return out


def detail_prediction(pid: str) -> Optional[dict]:
    table, raw = _pred_table(pid)
    try:
        with auth_db.get_conn() as c:
            row = c.execute("SELECT * FROM %s WHERE id = ?" % table, (int(raw),)).fetchone()
            if not row:
                return None
            d = dict(row)
            result = _pj(d.get("response_json")) or {}
            signal = _pj(d.get("signal_predictions"))
            top3 = _pj(d.get("predicted_top3")) or result.get("predicted_top3") or result.get("top3")
            conformal = _pj(d.get("conformal_set")) or result.get("conformal_set") or result.get("conformal_prediction_set")
            gate = _pj(d.get("gate_decision_path")) or result.get("gate_decision_path") or result.get("gate_weights")
            all_probs = (result.get("all_class_probabilities") or result.get("class_probabilities")
                         or result.get("probabilities") or result.get("all_probs"))
            uid = d.get("user_id")
            return {
                "id": str(pid), "guest": table == "guest_predictions",
                "diagnosis": d.get("predicted_class"), "crop": d.get("crop"),
                "confidence": d.get("confidence"), "tier": d.get("tier"),
                "is_ood": int(d.get("ood_flag") or 0),
                "severity": result.get("severity"),
                "calibration_warning": d.get("calibration_warning"),
                "confidence_outlier": d.get("confidence_outlier"),
                "conformal_set": conformal, "conformal_set_size": d.get("conformal_set_size"),
                "predicted_top3": top3, "all_class_probabilities": all_probs,
                "signal_predictions": signal, "gate_decision_path": gate,
                "timings": {"validation": d.get("validation_ms"), "router": d.get("router_ms"),
                            "specialist": d.get("specialist_ms"), "calibration": d.get("calibration_ms"),
                            "total": d.get("total_ms")},
                "has_heatmap": bool(d.get("heatmap_b64")),
                "created_at": d.get("created_at"),
                "device": {"ua": d.get("user_agent_family"), "country": d.get("client_country"),
                           "region": d.get("client_region"), "city": d.get("client_city"),
                           "camera": d.get("exif_camera_model"),
                           "lat": d.get("exif_gps_lat"), "lon": d.get("exif_gps_lon"),
                           "gps_accuracy_m": d.get("exif_gps_accuracy_m"),
                           "width": d.get("image_width"), "height": d.get("image_height"),
                           "mimetype": d.get("image_mimetype"), "n_bytes": d.get("image_n_bytes"),
                           "phash": d.get("image_perceptual_hash")},
                "model": {"deployment_version": d.get("deployment_version"),
                          "weights_hash": d.get("model_weights_hash"), "gpu_used": d.get("gpu_used"),
                          "cold_start": d.get("cold_start"), "peak_vram_mb": d.get("peak_vram_mb"),
                          "fallback_to_cpu": d.get("fallback_to_cpu")},
                "request": {"endpoint": d.get("endpoint"), "api_version": d.get("api_version"),
                            "request_id": d.get("request_id"), "trace_id": d.get("trace_id"),
                            "status_code": d.get("status_code")},
                "owner": _pred_owner(c, uid),
                "result": result,
            }
    except Exception as e:  # noqa: BLE001
        log.warning("detail_prediction(%s) failed: %s", pid, e)
        return None


def prediction_image_bytes(pid: str):
    """(bytes, mime) of the uploaded leaf. (None, None) if absent."""
    table, raw = _pred_table(pid)
    try:
        with auth_db.get_conn() as c:
            r = c.execute("SELECT image_bytes, image_mimetype FROM %s WHERE id = ?" % table,
                          (int(raw),)).fetchone()
        if not r:
            return None, None
        d = dict(r)
        return d.get("image_bytes"), (d.get("image_mimetype") or "image/jpeg")
    except Exception:
        return None, None


def prediction_heatmap_bytes(pid: str):
    """(png_bytes, 'image/png') decoded from the stored Grad-CAM base64."""
    import base64 as _b64
    table, raw = _pred_table(pid)
    try:
        with auth_db.get_conn() as c:
            r = c.execute("SELECT heatmap_b64 FROM %s WHERE id = ?" % table, (int(raw),)).fetchone()
        if not r:
            return None, None
        b64 = dict(r).get("heatmap_b64")
        if not b64:
            return None, None
        if isinstance(b64, str) and "," in b64[:40] and b64[:5].lower() == "data:":
            b64 = b64.split(",", 1)[1]   # strip a data-URL prefix if present
        return _b64.b64decode(b64), "image/png"
    except Exception:
        return None, None


# ── ADM-U · User-360 enrichment (persona/health · sessions · reliability ·
#            event timeline · keys sub-panel) ─────────────────────────────────
import math as _math_u
from datetime import datetime as _dtU, timezone as _tzU, timedelta as _tdU


def _days_since(iso):
    """Whole+fractional days since an ISO timestamp (UTC-naive safe). None if unparseable."""
    if not iso:
        return None
    try:
        t = _dtU.fromisoformat(str(iso).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=_tzU.utc)
        return max(0.0, (_dtU.now(_tzU.utc) - t).total_seconds() / 86400.0)
    except Exception:
        return None


def _persona_and_health(*, keys, api_requests, attributed_pv, web_sessions,
                        inferences, error_rate, last_seen, created):
    """Classify an account's primary usage mode (persona) and a 0-100 health score.

    Persona is descriptive (how they use the product); health is operational
    (should an admin worry). They are independent: an API-first account can be
    perfectly healthy. The persona text is what makes an empty engagement panel
    read as *expected* ("integration account") rather than *broken*.
    """
    d_seen = _days_since(last_seen)
    d_join = _days_since(created)
    has_api = bool(keys and api_requests)
    has_web = bool(attributed_pv or web_sessions)
    total_activity = int(api_requests or 0) + int(inferences or 0) + int(attributed_pv or 0)

    # ── persona ──
    if has_api and has_web:
        persona = ("hybrid", "Hybrid", "Drives traffic through BOTH the API and the web app.")
    elif has_api:
        persona = ("api_first", "API-first",
                   "Integration account — traffic flows through API keys, not the browser. "
                   "On-page telemetry isn't expected here.")
    elif has_web:
        persona = ("web", "Web user", "Engages primarily through the browser app.")
    elif total_activity == 0 and d_join is not None and d_join < 7:
        persona = ("new", "New", "Signed up recently — no activity recorded yet.")
    else:
        persona = ("observer", "Observer", "No API or web activity recorded.")

    # ── health (recency 40% · reliability 30% · volume 30%) ──
    recency = 0.0 if d_seen is None else max(0.0, 100.0 * (1.0 - min(d_seen, 60.0) / 60.0))
    reliability = max(0.0, 100.0 - float(error_rate or 0.0) * 2.5)           # 40% err → 0
    volume = min(100.0, 100.0 * (_math_u.log10(total_activity + 1) / 3.0))   # ~1000 acts → 100
    score = int(round(0.40 * recency + 0.30 * reliability + 0.30 * volume))

    if total_activity == 0 and d_join is not None and d_join < 7:
        state, reason = "new", "Joined %d day%s ago — settling in." % (int(d_join), "" if int(d_join) == 1 else "s")
    elif d_seen is not None and d_seen > 30:
        state, reason = "dormant", "No activity for %d days." % int(d_seen)
    elif float(error_rate or 0) >= 20:
        state, reason = "at_risk", "%.0f%% of API calls are failing." % float(error_rate)
    elif d_seen is not None and 7 <= d_seen <= 30 and total_activity > 0:
        state, reason = "at_risk", "Quieter lately — last active %d days ago." % int(d_seen)
    elif score < 50:
        state, reason = "at_risk", "Low overall health signal."
    else:
        state, reason = "healthy", "Active and stable."

    return {
        "persona": {"type": persona[0], "label": persona[1], "desc": persona[2]},
        "health": {"state": state, "score": score, "reason": reason,
                   "factors": {"recency": int(round(recency)),
                               "reliability": int(round(reliability)),
                               "volume": int(round(volume))}},
    }


def _user_sessions(c, uid):
    """Honest session accounting. Distinguishes:
       live        — browser tabs heartbeating within 45s (truly present)
       idle_open   — browser sessions not ended but gone quiet
       valid_tokens— non-revoked, non-expired auth tokens (what the old UI called
                     'active' — inflated by every login/restart on localhost)
    """
    now = auth_db._now_iso()
    hb = (_dtU.now(_tzU.utc) - _tdU(seconds=45)).isoformat()
    valid_tokens = _scalar(
        c, "SELECT COUNT(*) FROM sessions WHERE user_id=? AND revoked_at IS NULL AND expires_at>?",
        (uid, now))
    bs_open = _scalar(
        c, "SELECT COUNT(*) FROM browser_sessions WHERE user_id=? AND session_end_at IS NULL "
           "AND deleted_at IS NULL", (uid,))
    bs_live = _scalar(
        c, "SELECT COUNT(*) FROM browser_sessions WHERE user_id=? AND session_end_at IS NULL "
           "AND deleted_at IS NULL AND last_heartbeat_at>=?", (uid, hb))
    idle_open = max(0, int(bs_open or 0) - int(bs_live or 0))
    devices = []
    for r in c.execute(
            "SELECT device_browser, device_os, device_type, ip_city, ip_country, "
            "last_heartbeat_at, session_end_at FROM browser_sessions "
            "WHERE user_id=? AND deleted_at IS NULL ORDER BY last_heartbeat_at DESC LIMIT 6", (uid,)):
        d = dict(r)
        loc = " · ".join([x for x in (d.get("ip_city"), d.get("ip_country")) if x])
        devices.append({
            "browser": d.get("device_browser") or "—",
            "os": d.get("device_os") or "—",
            "type": d.get("device_type") or "—",
            "loc": loc or None,
            "last": d.get("last_heartbeat_at"),
            "open": d.get("session_end_at") is None,
        })
    if not devices:
        # API-first / token-only accounts have no browser sessions — fall back to
        # the auth-token user-agent so the panel still shows *something* truthful.
        for r in c.execute(
                "SELECT user_agent, ip_addr, created_at FROM sessions "
                "WHERE user_id=? AND revoked_at IS NULL ORDER BY created_at DESC LIMIT 4", (uid,)):
            d = dict(r)
            ua = (d.get("user_agent") or "").strip()
            devices.append({"browser": (ua[:42] + "…") if len(ua) > 42 else (ua or "unknown client"),
                            "os": None, "type": "token", "loc": None,
                            "last": d.get("created_at"), "open": True})
    return {"valid_tokens": int(valid_tokens or 0), "live": int(bs_live or 0),
            "idle_open": idle_open, "browser_open": int(bs_open or 0), "devices": devices}


def _user_reliability(c, uid):
    """Status-code breakdown that EXPLAINS the headline error rate, plus the
    specific endpoints generating the errors."""
    breakdown = []
    for r in c.execute(
            "SELECT status_code st, COUNT(*) n FROM api_key_request_log r "
            "JOIN api_keys k ON k.public_id=r.key_id WHERE k.user_id=? "
            "GROUP BY status_code ORDER BY n DESC", (uid,)):
        d = dict(r)
        code = int(d.get("st") or 0)
        cls = "ok" if code < 400 else ("warn" if code < 500 else "err")
        breakdown.append({"code": code, "n": int(d.get("n") or 0), "cls": cls})
    top_err = []
    for r in c.execute(
            "SELECT r.path label, r.status_code st, COUNT(*) n FROM api_key_request_log r "
            "JOIN api_keys k ON k.public_id=r.key_id WHERE k.user_id=? AND r.status_code>=400 "
            "GROUP BY r.path, r.status_code ORDER BY n DESC LIMIT 6", (uid,)):
        d = dict(r)
        top_err.append({"label": d.get("label"), "code": int(d.get("st") or 0),
                        "n": int(d.get("n") or 0)})
    return {"status_breakdown": breakdown, "top_errors": top_err}


def _user_events(c, uid, limit=16):
    """Unified, namespaced event timeline merged across sources (audit_log for
    identity/security, predictions for inferences). Inference rows carry a `ref`
    so the frontend can cross-drill into the prediction dossier."""
    events = []
    amap = getattr(admin_queries, "_AUDIT_MAP", {})
    try:
        for r in c.execute(
                "SELECT id, event, created_at FROM audit_log WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?", (uid, limit)):
            d = dict(r)
            cat, sev, title = amap.get(d["event"],
                                       ("identity", "info", str(d["event"]).replace("_", " ")))
            events.append({"id": "al-" + str(d["id"]), "kind": cat, "sev": sev,
                           "title": title, "ts": d.get("created_at"), "ref": None})
    except Exception:
        pass
    try:
        for r in c.execute(
                "SELECT id, predicted_class, created_at FROM predictions "
                "WHERE user_id=? AND deleted_at IS NULL ORDER BY created_at DESC LIMIT ?", (uid, limit)):
            d = dict(r)
            dx = str(d.get("predicted_class") or "").replace("_", " ")
            events.append({"id": "p" + str(d["id"]), "kind": "inference", "sev": "info",
                           "title": "Inference · " + (dx or "—"), "ts": d.get("created_at"),
                           "ref": {"kind": "prediction", "id": "p" + str(d["id"])}})
    except Exception:
        pass
    events.sort(key=lambda e: e.get("ts") or "", reverse=True)
    return events[:limit]


def _user_keys(c, uid):
    """Per-key rollup for the dossier's keys sub-panel. Each row cross-drills
    into the existing key drawer via public_id."""
    out = []
    for r in c.execute(
            "SELECT public_id, name, status, scopes, last_used_at, request_count, error_count, "
            "quota_per_day, quota_period, rate_limit_per_min, group_id, created_at "
            "FROM api_keys WHERE user_id=? AND deleted_at IS NULL ORDER BY created_at DESC", (uid,)):
        d = dict(r)
        rc = int(d.get("request_count") or 0)
        ec = int(d.get("error_count") or 0)
        # scopes may be JSON array or comma list — derive a count defensively.
        raw_scopes = d.get("scopes") or ""
        try:
            parsed = _json_pred.loads(raw_scopes) if raw_scopes.strip().startswith("[") else None
            scope_count = len(parsed) if isinstance(parsed, list) else len([s for s in raw_scopes.split(",") if s.strip()])
        except Exception:
            scope_count = len([s for s in str(raw_scopes).split(",") if s.strip()])
        out.append({
            "public_id": d.get("public_id"), "name": d.get("name"),
            "status": (d.get("status") or "active"), "scope_count": scope_count,
            "last_used": d.get("last_used_at"), "requests": rc,
            "error_rate": round(100.0 * ec / rc, 1) if rc else 0.0,
            "quota_per_day": d.get("quota_per_day"), "quota_period": d.get("quota_period"),
            "group_id": d.get("group_id"), "created_at": d.get("created_at"),
        })
    return out


def detail_user(user_id: int) -> Optional[dict]:
    uid = int(user_id)
    try:
        base = admin_queries.admin_get_user(uid)
        if base is None:
            return None
        out = dict(base)
        out["engagement"] = user_engagement(uid)
        with auth_db.get_conn() as c:
            # API usage rollup
            total = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log r "
                               "JOIN api_keys k ON k.public_id = r.key_id WHERE k.user_id = ?", (uid,))
            errs = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log r "
                              "JOIN api_keys k ON k.public_id = r.key_id "
                              "WHERE k.user_id = ? AND r.status_code >= 400", (uid,))
            top_eps = []
            for r in c.execute("SELECT r.path label, COUNT(*) v FROM api_key_request_log r "
                               "JOIN api_keys k ON k.public_id = r.key_id WHERE k.user_id = ? "
                               "GROUP BY r.path ORDER BY v DESC LIMIT 6", (uid,)):
                d = dict(r); top_eps.append({"label": d.get("label"), "value": int(d.get("v") or 0)})
            req_series = admin_queries._series_for(
                c, "api_key_request_log", "timestamp", "30d",
                "AND key_id IN (SELECT public_id FROM api_keys WHERE user_id = %d)" % uid)
            out["api"] = {"requests": total, "errors": errs,
                          "error_rate": round(100.0 * errs / total, 1) if total else 0.0,
                          "top_endpoints": top_eps, "series": req_series}
            # inferences rollup — REAL website inferences (predictions), repointed
            # from the geo `scans` table (ADM-X). Recent rows carry the prefixed
            # prediction id ("p…") so they cross-drill into the inference dossier.
            stc = admin_queries._scans_time_col(c)
            crop_mix, disease_mix, districts = [], [], []
            n_pred = _scalar(c, "SELECT COUNT(*) FROM predictions WHERE user_id = ? AND deleted_at IS NULL", (uid,))
            for r in c.execute("SELECT predicted_class label, COUNT(*) v FROM predictions WHERE user_id = ? "
                               "AND deleted_at IS NULL GROUP BY predicted_class ORDER BY v DESC LIMIT 8", (uid,)):
                d = dict(r); disease_mix.append({"label": d.get("label"), "value": int(d.get("v") or 0)})
            recent_scans = []
            for r in c.execute("SELECT id, predicted_class, confidence, tier, "
                               "(heatmap_b64 IS NOT NULL) hm, created_at FROM predictions "
                               "WHERE user_id = ? AND deleted_at IS NULL ORDER BY created_at DESC LIMIT 8", (uid,)):
                d = dict(r); recent_scans.append({"scan_uid": "p" + str(d["id"]), "diagnosis": d.get("predicted_class"),
                                                  "confidence": d.get("confidence"), "severity": ("tier " + str(d.get("tier"))) if d.get("tier") else "",
                                                  "has_heatmap": bool(d.get("hm")), "ts": d.get("created_at")})
            # location — districts this user scans from (geo columns vary; best-effort)
            geo_cols = {dict(r)["name"] for r in c.execute("PRAGMA table_info(scans)")}
            dcol = next((x for x in ("geo_district", "district", "admin2") if x in geo_cols), None)
            scol = next((x for x in ("geo_state", "state", "admin1") if x in geo_cols), None)
            if dcol:
                q = ('SELECT "%s" label, %s state, COUNT(*) v FROM scans WHERE user_id = ? '
                     'AND deleted_at IS NULL AND "%s" IS NOT NULL GROUP BY "%s" ORDER BY v DESC LIMIT 8'
                     % (dcol, ('"%s"' % scol) if scol else "NULL", dcol, dcol))
                for r in c.execute(q, (uid,)):
                    d = dict(r); districts.append({"district": d.get("label"), "state": d.get("state"),
                                                   "count": int(d.get("v") or 0)})
            out["inferences"] = {"count": n_pred, "disease_mix": disease_mix,
                                 "recent": recent_scans}
            out["location"] = {"districts": districts}
            # ── ADM-U enrichments (all read from the SAME open connection) ──
            out["sessions_detail"] = _user_sessions(c, uid)
            out["reliability"] = _user_reliability(c, uid)
            out["events"] = _user_events(c, uid)
            out["keys"] = _user_keys(c, uid)
            # persona + health depend on the rollups computed above
            ph = _persona_and_health(
                keys=out.get("key_count", 0),
                api_requests=out["api"]["requests"],
                attributed_pv=out["engagement"].get("page_views", 0),
                web_sessions=out["engagement"].get("sessions", 0),
                inferences=out["inferences"]["count"],
                error_rate=out["api"]["error_rate"],
                last_seen=out.get("last_seen_at"),
                created=out.get("created_at"))
            out["persona"] = ph["persona"]
            out["health"] = ph["health"]
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("detail_user(%s) failed: %s", user_id, e)
        return None
