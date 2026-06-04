"""ADM-T · P1 — org-wide aggregation data layer for the admin Traffic + Geography
sections (API sub-section, Website sub-section, and Geography).

Design notes
------------
* All functions are **org-wide** (no key/user filter) and **fail-open**: any
  query error degrades to an empty/zero shape rather than raising, mirroring the
  rest of the admin data layer.
* Time bucketing reuses ``admin_queries._buckets`` for window→key alignment.
  ``api_key_request_log.timestamp`` is space-separated ("YYYY-MM-DD HH:..."),
  while predictions / page_views use ISO "T". ``_DT`` below normalises both with
  ``replace(ts,' ','T')`` so a single substr bucketing scheme works everywhere.
* Percentiles: SQLite has no percentile_cont, so ``_percentiles`` pulls the
  latency column (capped) and computes p50/p90/p95/p99 in Python — fine at this
  scale and exact, not an approximation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import admin_queries as aq

log = logging.getLogger("apin_v2.account.admin_traffic")

_scalar = aq._scalar          # int-coercing scalar (COUNT/SUM) — NEVER for text
_buckets = aq._buckets
_window_cutoff = aq._window_cutoff


def _scalar_raw(c, sql, args=()):
    """Scalar that returns the value verbatim (no int coercion) — for MIN(text)
    / MAX(timestamp) etc. where aq._scalar would raise on a non-numeric value."""
    try:
        r = c.execute(sql, args).fetchone()
        if not r:
            return None
        return list(dict(r).values())[0]
    except Exception:
        return None


def _bucket_series(c, table: str, ts_expr: str, window: str, extra: str = "", args=None):
    """Zero-filled count-per-bucket series using a normalised ts *expression*
    (not a column name — so it never gets quoted as an identifier the way
    aq._series_for would). ts_expr must already be SQL like
    ``replace(timestamp,' ','T')``."""
    keys, sub, _hourly = _buckets(window)
    cutoff = keys[0]
    counts = {}
    try:
        sql = ("SELECT substr(%s,1,%d) b, COUNT(*) c FROM %s "
               "WHERE %s >= ? %s GROUP BY b" % (ts_expr, sub, table, ts_expr, extra))
        for r in c.execute(sql, [cutoff] + list(args or [])):
            d = dict(r); counts[d["b"]] = int(d["c"] or 0)
    except Exception:
        pass
    return [{"t": k, "c": counts.get(k, 0)} for k in keys]

REQ = "api_key_request_log"
# normalised timestamp expression for the request log (space → T)
_RDT = "replace(%s.timestamp,' ','T')" % REQ
_RDT_BARE = "replace(timestamp,' ','T')"


def _win_clause(window: str, col_expr: str = _RDT_BARE):
    """Return (sql_fragment, args) restricting `col_expr` to the window.
    'all' → no restriction. col_expr is an already-normalised ISO expression."""
    cut = _window_cutoff(window)
    if not cut:
        return "", []
    return (" AND %s >= ? " % col_expr), [cut]


def _percentiles(values, ps=(50, 90, 95, 99)):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {f"p{p}": None for p in ps}
    out = {}
    n = len(vals)
    for p in ps:
        # nearest-rank
        idx = max(0, min(n - 1, int(round((p / 100.0) * n + 0.5)) - 1))
        out[f"p{p}"] = vals[idx]
    return out


def _status_class(code: int) -> str:
    if code < 300: return "2xx"
    if code < 400: return "3xx"
    if code < 500: return "4xx"
    return "5xx"


# ════════════════════════════════════════════════════════════════════════════
# TRAFFIC › API
# ════════════════════════════════════════════════════════════════════════════
def traffic_api_overview(window: str = "all") -> dict:
    """KPI ribbon + request/error series + status mix + latency percentiles."""
    out = {"window": window, "requests": 0, "errors": 0, "error_rate": 0.0,
           "bytes_in": 0, "bytes_out": 0, "throughput_per_min": 0.0,
           "latency": {}, "series": [], "series_label": "", "status_mix": []}
    try:
        wc, wa = _win_clause(window)
        with auth_db.get_conn() as c:
            row = c.execute(
                "SELECT COUNT(*) n, "
                "SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) errs, "
                "COALESCE(SUM(bytes_in),0) bi, COALESCE(SUM(bytes_out),0) bo "
                "FROM %s WHERE 1=1 %s" % (REQ, wc), wa).fetchone()
            d = dict(row) if row else {}
            n = int(d.get("n") or 0); errs = int(d.get("errs") or 0)
            out["requests"] = n; out["errors"] = errs
            out["error_rate"] = round(100.0 * errs / n, 1) if n else 0.0
            out["bytes_in"] = int(d.get("bi") or 0); out["bytes_out"] = int(d.get("bo") or 0)
            # throughput over the window's minutes
            mins = {"24h": 1440, "7d": 10080, "30d": 43200}.get(window)
            if mins is None:
                first = _scalar_raw(c, "SELECT MIN(%s) FROM %s" % (_RDT_BARE, REQ))
                try:
                    t0 = datetime.fromisoformat(str(first));
                    if t0.tzinfo is None: t0 = t0.replace(tzinfo=timezone.utc)
                    mins = max(1.0, (datetime.now(timezone.utc) - t0).total_seconds() / 60.0)
                except Exception:
                    mins = 1.0
            out["throughput_per_min"] = round(n / mins, 2) if mins else 0.0
            # latency percentiles (capped pull)
            lats = [int(dict(r)["latency_ms"]) for r in c.execute(
                "SELECT latency_ms FROM %s WHERE latency_ms IS NOT NULL %s "
                "ORDER BY RANDOM() LIMIT 20000" % (REQ, wc), wa)]
            out["latency"] = _percentiles(lats)
            # status mix
            mix = {}
            for r in c.execute(
                "SELECT status_code, COUNT(*) v FROM %s WHERE 1=1 %s GROUP BY status_code" % (REQ, wc), wa):
                dd = dict(r); mix[_status_class(int(dd["status_code"] or 0))] = mix.get(_status_class(int(dd["status_code"] or 0)), 0) + int(dd["v"] or 0)
            out["status_mix"] = [{"label": k, "value": mix[k]} for k in ("2xx", "3xx", "4xx", "5xx") if mix.get(k)]
            # series
            out["series"] = _bucket_series(c, REQ, _RDT_BARE, window)
            out["series_label"] = "requests / " + ("hour" if window == "24h" else "day")
    except Exception as e:
        log.warning("traffic_api_overview failed: %s", e)
    return out


def traffic_api_terrain(window: str = "7d") -> dict:
    """7-day × 24-hour grid for the 3D Traffic Terrain. Each cell: count, error
    rate, avg latency. Always returns a full 7×24 matrix (zero-filled)."""
    out = {"days": [], "grid": []}
    try:
        now = datetime.now(timezone.utc)
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        days = [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
        out["days"] = days
        grid = {d: {h: {"n": 0, "e": 0, "lat": 0.0} for h in range(24)} for d in days}
        cut = days[0]
        with auth_db.get_conn() as c:
            for r in c.execute(
                "SELECT substr(%s,1,10) d, CAST(substr(%s,12,2) AS INT) h, "
                "COUNT(*) n, SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) e, "
                "AVG(latency_ms) lat FROM %s WHERE %s >= ? GROUP BY d, h"
                % (_RDT_BARE, _RDT_BARE, REQ, _RDT_BARE), (cut,)):
                d = dict(r); day = d["d"]; h = int(d["h"] or 0)
                if day in grid and 0 <= h < 24:
                    grid[day][h] = {"n": int(d["n"] or 0), "e": int(d["e"] or 0),
                                    "lat": round(float(d["lat"] or 0), 1)}
        out["grid"] = [[grid[day][h] for h in range(24)] for day in days]
    except Exception as e:
        log.warning("traffic_api_terrain failed: %s", e)
    return out


def traffic_method_status(window: str = "all") -> dict:
    """Method × status-class matrix for the honeycomb hive."""
    out = {"methods": [], "classes": ["2xx", "3xx", "4xx", "5xx"], "cells": []}
    try:
        wc, wa = _win_clause(window)
        agg = {}
        with auth_db.get_conn() as c:
            for r in c.execute(
                "SELECT method, status_code, COUNT(*) v FROM %s WHERE 1=1 %s "
                "GROUP BY method, status_code" % (REQ, wc), wa):
                d = dict(r); m = (d["method"] or "?").upper(); cls = _status_class(int(d["status_code"] or 0))
                agg.setdefault(m, {}).setdefault(cls, 0)
                agg[m][cls] += int(d["v"] or 0)
        methods = sorted(agg.keys(), key=lambda m: -sum(agg[m].values()))
        out["methods"] = methods
        out["cells"] = [{"method": m, "cls": cls, "value": agg[m].get(cls, 0)}
                        for m in methods for cls in out["classes"]]
    except Exception as e:
        log.warning("traffic_method_status failed: %s", e)
    return out


def traffic_latency(window: str = "7d") -> dict:
    """Per-day p50/p90/p99 ridgeline + recent slow requests for the seismograph."""
    out = {"ridges": [], "slow": []}
    try:
        keys, _sub, _hourly = _buckets(window)
        with auth_db.get_conn() as c:
            by_day = {}
            wc, wa = _win_clause(window)
            for r in c.execute(
                "SELECT substr(%s,1,10) d, latency_ms FROM %s "
                "WHERE latency_ms IS NOT NULL %s" % (_RDT_BARE, REQ, wc), wa):
                d = dict(r); by_day.setdefault(d["d"], []).append(int(d["latency_ms"]))
            day_keys = [k for k in keys if len(k) == 10] or sorted(by_day.keys())
            for k in day_keys:
                vals = by_day.get(k, [])
                p = _percentiles(vals, (50, 90, 99))
                out["ridges"].append({"t": k, "n": len(vals), **p})
            # recent slow requests (seismograph spikes) → drill to request drawer
            for r in c.execute(
                "SELECT id, path, status_code, latency_ms, %s ts FROM %s "
                "WHERE latency_ms IS NOT NULL %s ORDER BY %s DESC LIMIT 40"
                % (_RDT_BARE, REQ, wc, _RDT_BARE), wa):
                d = dict(r)
                out["slow"].append({"id": d["id"], "path": d["path"],
                                    "status": d["status_code"], "latency_ms": int(d["latency_ms"]),
                                    "ts": d["ts"]})
    except Exception as e:
        log.warning("traffic_latency failed: %s", e)
    return out


def traffic_bandwidth(window: str = "24h") -> dict:
    """Per-bucket bytes in/out for the bandwidth ring + 24h clock dial volume."""
    out = {"buckets": [], "clock": [0] * 24}
    try:
        keys, sub, hourly = _buckets(window)
        agg = {}
        clock = [0] * 24
        wc, wa = _win_clause(window)
        with auth_db.get_conn() as c:
            for r in c.execute(
                "SELECT substr(%s,1,%d) b, COALESCE(SUM(bytes_in),0) bi, "
                "COALESCE(SUM(bytes_out),0) bo, COUNT(*) n FROM %s WHERE 1=1 %s GROUP BY b"
                % (_RDT_BARE, sub, REQ, wc), wa):
                d = dict(r); agg[d["b"]] = {"in": int(d["bi"]), "out": int(d["bo"]), "n": int(d["n"])}
            for r in c.execute(
                "SELECT CAST(substr(%s,12,2) AS INT) h, COUNT(*) n FROM %s "
                "WHERE 1=1 %s GROUP BY h" % (_RDT_BARE, REQ, wc), wa):
                d = dict(r); h = int(d["h"] or 0)
                if 0 <= h < 24: clock[h] += int(d["n"] or 0)
        out["buckets"] = [{"t": k, **agg.get(k, {"in": 0, "out": 0, "n": 0})} for k in keys]
        out["clock"] = clock
    except Exception as e:
        log.warning("traffic_bandwidth failed: %s", e)
    return out


def traffic_top(window: str = "all", limit: int = 8) -> dict:
    """Top API keys + top origin IPs (with sparkline-able recent counts)."""
    out = {"keys": [], "ips": []}
    try:
        wc, wa = _win_clause(window)
        with auth_db.get_conn() as c:
            for r in c.execute(
                "SELECT r.key_id, k.name, COUNT(*) v, "
                "SUM(CASE WHEN r.status_code>=400 THEN 1 ELSE 0 END) e "
                "FROM %s r LEFT JOIN api_keys k ON k.public_id=r.key_id "
                "WHERE 1=1 %s GROUP BY r.key_id ORDER BY v DESC LIMIT ?"
                % (REQ, wc.replace(_RDT_BARE, "replace(r.timestamp,' ','T')")),
                    wa + [limit]):
                d = dict(r); v = int(d["v"] or 0); e = int(d["e"] or 0)
                out["keys"].append({"public_id": d["key_id"], "name": d.get("name") or d["key_id"],
                                    "requests": v, "error_rate": round(100.0 * e / v, 1) if v else 0.0})
            for r in c.execute(
                "SELECT ip, COUNT(*) v FROM %s WHERE ip IS NOT NULL %s "
                "GROUP BY ip ORDER BY v DESC LIMIT ?" % (REQ, wc), wa + [limit]):
                d = dict(r); out["ips"].append({"ip": d["ip"], "requests": int(d["v"] or 0)})
    except Exception as e:
        log.warning("traffic_top failed: %s", e)
    return out


def traffic_endpoints(window: str = "all", limit: int = 60) -> dict:
    """Per-endpoint (path) rollup for the galaxy / genome / treemap / needs-attention.
    volume · avg+p95 latency · error rate · method mix · last-seen."""
    out = {"endpoints": []}
    try:
        wc, wa = _win_clause(window)
        with auth_db.get_conn() as c:
            rows = c.execute(
                "SELECT path, COUNT(*) v, AVG(latency_ms) lat, "
                "SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) e, "
                "MAX(%s) last FROM %s WHERE path IS NOT NULL %s "
                "GROUP BY path ORDER BY v DESC LIMIT ?" % (_RDT_BARE, REQ, wc), wa + [limit]).fetchall()
            for r in rows:
                d = dict(r); v = int(d["v"] or 0); e = int(d["e"] or 0)
                # p95 + method mix per path (bounded)
                lats = [int(dict(x)["latency_ms"]) for x in c.execute(
                    "SELECT latency_ms FROM %s WHERE path=? AND latency_ms IS NOT NULL %s "
                    "ORDER BY RANDOM() LIMIT 4000" % (REQ, wc), [d["path"]] + wa)]
                p = _percentiles(lats, (50, 95, 99))
                methods = {}
                for mr in c.execute(
                    "SELECT method, COUNT(*) n FROM %s WHERE path=? %s GROUP BY method"
                    % (REQ, wc), [d["path"]] + wa):
                    md = dict(mr); methods[(md["method"] or "?").upper()] = int(md["n"] or 0)
                err_rate = round(100.0 * e / v, 1) if v else 0.0
                out["endpoints"].append({
                    "path": d["path"], "requests": v, "errors": e, "error_rate": err_rate,
                    "avg_latency": round(float(d["lat"] or 0), 1), "p50": p["p50"],
                    "p95": p["p95"], "p99": p["p99"], "methods": methods, "last": d["last"],
                    # a simple health 0-100: penalise error rate + slow p95
                    "health": max(0, min(100, int(100 - err_rate * 2 - min(40, (p["p95"] or 0) / 50)))),
                })
    except Exception as e:
        log.warning("traffic_endpoints failed: %s", e)
    return out


def traffic_sequences(window: str = "7d", gap_s: int = 1800, limit_edges: int = 60) -> dict:
    """Endpoint call-sequence graph (galaxy arcs): for each key, order requests by
    time and count consecutive (path→path) transitions within a `gap_s` session
    gap. Returns weighted directed edges between endpoints."""
    out = {"edges": []}
    try:
        wc, wa = _win_clause(window)
        edges = {}
        with auth_db.get_conn() as c:
            rows = c.execute(
                "SELECT key_id, path, %s ts FROM %s WHERE path IS NOT NULL %s "
                "ORDER BY key_id, %s ASC" % (_RDT_BARE, REQ, wc, _RDT_BARE), wa).fetchall()
        prev_key = None; prev_path = None; prev_t = None
        for r in rows:
            d = dict(r); k = d["key_id"]; path = d["path"]
            try:
                t = datetime.fromisoformat(str(d["ts"]))
            except Exception:
                t = None
            if prev_key == k and prev_path and path != prev_path and prev_t and t:
                if (t - prev_t).total_seconds() <= gap_s:
                    key = (prev_path, path)
                    edges[key] = edges.get(key, 0) + 1
            prev_key, prev_path, prev_t = k, path, t
        ranked = sorted(edges.items(), key=lambda kv: -kv[1])[:limit_edges]
        out["edges"] = [{"from": a, "to": b, "weight": w} for (a, b), w in ranked]
    except Exception as e:
        log.warning("traffic_sequences failed: %s", e)
    return out


# ════════════════════════════════════════════════════════════════════════════
# TRAFFIC › WEBSITE  (page_views · clicks · browser_sessions — ISO 'T' stamps)
# ════════════════════════════════════════════════════════════════════════════
PV = "page_views"
BS = "browser_sessions"
CL = "clicks"
_PV_LIVE = "deleted_at IS NULL"


def website_overview(window: str = "all") -> dict:
    """Top-of-section KPIs + visits/day series + new-vs-returning.
    NOTE: avg_active_s / scroll are sparse until the W0.2 beacon-flush lands;
    they surface real depth automatically once those columns populate."""
    out = {"window": window, "visits": 0, "sessions": 0, "avg_active_s": 0.0,
           "bounce_rate": 0.0, "avg_scroll": 0.0, "new": 0, "returning": 0,
           "series": [], "series_label": ""}
    try:
        pvc, pva = _win_clause(window, "entered_at")
        bsc, bsa = _win_clause(window, "session_start_at")
        with auth_db.get_conn() as c:
            row = c.execute(
                "SELECT COUNT(*) v, COUNT(DISTINCT browser_session_id) s, "
                "AVG(NULLIF(active_duration_ms,0)) act, AVG(bounce) bnc, "
                "AVG(NULLIF(max_scroll_depth_pct,0)) scr "
                "FROM %s WHERE %s %s" % (PV, _PV_LIVE, pvc), pva).fetchone()
            d = dict(row) if row else {}
            out["visits"] = int(d.get("v") or 0)
            out["sessions"] = int(d.get("s") or 0)
            out["avg_active_s"] = round(float(d.get("act") or 0) / 1000.0, 1)
            out["bounce_rate"] = round(100.0 * float(d.get("bnc") or 0), 1)
            out["avg_scroll"] = round(float(d.get("scr") or 0), 0)
            for r in c.execute(
                "SELECT COALESCE(is_returning_user,0) ret, COUNT(*) n FROM %s "
                "WHERE deleted_at IS NULL %s GROUP BY ret" % (BS, bsc), bsa):
                dd = dict(r)
                if int(dd.get("ret") or 0): out["returning"] = int(dd["n"] or 0)
                else: out["new"] = int(dd["n"] or 0)
            out["series"] = _bucket_series(c, PV, "entered_at", window, extra="AND " + _PV_LIVE)
            out["series_label"] = "visits / " + ("hour" if window == "24h" else "day")
    except Exception as e:
        log.warning("website_overview failed: %s", e)
    return out


def website_pages(window: str = "all", limit: int = 30) -> dict:
    """Per-route rollup for the Site-Flow City + Top Pages table."""
    out = {"pages": []}
    try:
        pvc, pva = _win_clause(window, "entered_at")
        with auth_db.get_conn() as c:
            for r in c.execute(
                "SELECT page_route route, MAX(page_title) title, COUNT(*) visits, "
                "COUNT(DISTINCT browser_session_id) sessions, "
                "AVG(NULLIF(active_duration_ms,0)) act, AVG(NULLIF(idle_duration_ms,0)) idle, "
                "AVG(NULLIF(max_scroll_depth_pct,0)) scroll, SUM(click_count) clicks, "
                "SUM(api_call_count) api, AVG(bounce) bounce "
                "FROM %s WHERE %s %s GROUP BY page_route ORDER BY visits DESC LIMIT ?"
                % (PV, _PV_LIVE, pvc), pva + [limit]):
                d = dict(r)
                out["pages"].append({
                    "route": d["route"], "title": d.get("title"),
                    "visits": int(d["visits"] or 0), "sessions": int(d["sessions"] or 0),
                    "avg_active_s": round(float(d["act"] or 0) / 1000.0, 1),
                    "avg_idle_s": round(float(d["idle"] or 0) / 1000.0, 1),
                    "scroll": round(float(d["scroll"] or 0), 0),
                    "clicks": int(d["clicks"] or 0), "api_calls": int(d["api"] or 0),
                    "bounce": round(100.0 * float(d["bounce"] or 0), 1),
                })
    except Exception as e:
        log.warning("website_pages failed: %s", e)
    return out


def website_heatmap(route: Optional[str] = None, window: str = "all", limit: int = 1500) -> dict:
    """Real click coordinates (already fully captured) for the click heatmap,
    plus dead/rage flags and the ranked clicked-elements. Optionally scoped to a
    single route."""
    out = {"route": route, "points": [], "dead": 0, "rage": 0, "total": 0,
           "elements": [], "routes": []}
    try:
        clc, cla = _win_clause(window, "occurred_at")
        rcl = ""
        ra = []
        if route:
            # join to the page_view to resolve the route of each click
            rcl = " AND cl.page_view_id IN (SELECT id FROM %s WHERE page_route = ?)" % PV
            ra = [route]
        with auth_db.get_conn() as c:
            # routes available (for the route toggle)
            for r in c.execute(
                "SELECT page_route, COUNT(*) n FROM %s WHERE %s %s GROUP BY page_route "
                "ORDER BY n DESC LIMIT 12" % (PV, _PV_LIVE, _win_clause(window, "entered_at")[0]),
                    _win_clause(window, "entered_at")[1]):
                d = dict(r); out["routes"].append({"route": d["page_route"], "visits": int(d["n"] or 0)})
            rows = c.execute(
                "SELECT cl.click_x_page x, cl.click_y_page y, cl.viewport_width_at_click vw, "
                "cl.viewport_height_at_click vh, cl.was_rage_click rage, cl.was_dead_click dead "
                "FROM %s cl WHERE cl.click_x_page IS NOT NULL %s %s ORDER BY cl.occurred_at DESC LIMIT ?"
                % (CL, clc.replace("occurred_at", "cl.occurred_at"), rcl), cla + ra + [limit]).fetchall()
            dead = rage = 0
            for r in rows:
                d = dict(r)
                isr = int(d.get("rage") or 0); isd = int(d.get("dead") or 0)
                dead += isd; rage += isr
                out["points"].append({
                    "x": int(d["x"]), "y": int(d["y"]),
                    "vw": int(d["vw"] or 0), "vh": int(d["vh"] or 0),
                    "rage": bool(isr), "dead": bool(isd)})
            out["dead"] = dead; out["rage"] = rage; out["total"] = len(rows)
            # ranked clicked elements
            for r in c.execute(
                "SELECT COALESCE(NULLIF(cl.target_text,''), NULLIF(cl.target_id,''), cl.target_tag) el, "
                "COUNT(*) n, SUM(cl.was_dead_click) dead FROM %s cl "
                "WHERE 1=1 %s %s GROUP BY el ORDER BY n DESC LIMIT 12"
                % (CL, clc.replace("occurred_at", "cl.occurred_at"), rcl), cla + ra):
                d = dict(r)
                out["elements"].append({"label": (d["el"] or "—")[:48], "clicks": int(d["n"] or 0),
                                        "dead": int(d["dead"] or 0)})
    except Exception as e:
        log.warning("website_heatmap failed: %s", e)
    return out


def website_devices(window: str = "all") -> dict:
    """Device-type / browser / OS / screen-size / capability / locale breakdown
    for the Device Orbits widget. (browser/os populate via W0.1 going forward.)"""
    out = {"types": [], "browsers": [], "os": [], "screens": [], "locales": [],
           "capability": {"cpu": [], "memory": []}}
    try:
        bsc, bsa = _win_clause(window, "session_start_at")
        base = "FROM %s WHERE deleted_at IS NULL %s" % (BS, bsc)

        def _grp(col, cap=8):
            res = []
            try:
                for r in c.execute(
                    "SELECT COALESCE(%s,'unknown') k, COUNT(*) n %s GROUP BY %s "
                    "ORDER BY n DESC LIMIT %d" % (col, base, col, cap), bsa):
                    d = dict(r); res.append({"label": d["k"], "value": int(d["n"] or 0)})
            except Exception:
                pass
            return res
        with auth_db.get_conn() as c:
            out["types"] = _grp("device_type")
            out["browsers"] = _grp("device_browser")
            out["os"] = _grp("device_os")
            out["locales"] = _grp("locale", 10)
            for r in c.execute(
                "SELECT screen_width w, screen_height h, COUNT(*) n %s "
                "AND screen_width IS NOT NULL GROUP BY w,h ORDER BY n DESC LIMIT 12" % base, bsa):
                d = dict(r); out["screens"].append({"w": int(d["w"] or 0), "h": int(d["h"] or 0),
                                                    "value": int(d["n"] or 0)})
            out["capability"]["cpu"] = _grp("cpu_cores", 8)
            out["capability"]["memory"] = _grp("memory_gb", 8)
    except Exception as e:
        log.warning("website_devices failed: %s", e)
    return out


def website_journey(window: str = "7d", gap_s: int = 1800, limit_edges: int = 40) -> dict:
    """Page→page journey flows (sankey) sessionized by browser_session_id."""
    out = {"edges": [], "entries": [], "exits": []}
    try:
        pvc, pva = _win_clause(window, "entered_at")
        edges = {}; entries = {}; exits = {}
        with auth_db.get_conn() as c:
            rows = c.execute(
                "SELECT browser_session_id sid, page_route route, entered_at FROM %s "
                "WHERE %s %s ORDER BY browser_session_id, entered_at ASC"
                % (PV, _PV_LIVE, pvc), pva).fetchall()
        prev_sid = None; prev_route = None
        for r in rows:
            d = dict(r); sid = d["sid"]; route = d["route"]
            if sid != prev_sid:
                if prev_route is not None:
                    exits[prev_route] = exits.get(prev_route, 0) + 1
                entries[route] = entries.get(route, 0) + 1
            elif prev_route and route != prev_route:
                edges[(prev_route, route)] = edges.get((prev_route, route), 0) + 1
            prev_sid, prev_route = sid, route
        if prev_route is not None:
            exits[prev_route] = exits.get(prev_route, 0) + 1
        out["edges"] = [{"from": a, "to": b, "weight": w}
                        for (a, b), w in sorted(edges.items(), key=lambda kv: -kv[1])[:limit_edges]]
        out["entries"] = [{"route": k, "value": v} for k, v in sorted(entries.items(), key=lambda kv: -kv[1])[:8]]
        out["exits"] = [{"route": k, "value": v} for k, v in sorted(exits.items(), key=lambda kv: -kv[1])[:8]]
    except Exception as e:
        log.warning("website_journey failed: %s", e)
    return out


def website_scroll(window: str = "all") -> dict:
    """Per-route scroll-depth distribution (25/50/75/100 reefs).
    Sparse until W0.2; structure is correct so it lights up on capture."""
    out = {"pages": []}
    try:
        pvc, pva = _win_clause(window, "entered_at")
        with auth_db.get_conn() as c:
            for r in c.execute(
                "SELECT page_route route, COUNT(*) total, "
                "SUM(CASE WHEN max_scroll_depth_pct>=25 THEN 1 ELSE 0 END) d25, "
                "SUM(CASE WHEN max_scroll_depth_pct>=50 THEN 1 ELSE 0 END) d50, "
                "SUM(CASE WHEN max_scroll_depth_pct>=75 THEN 1 ELSE 0 END) d75, "
                "SUM(CASE WHEN max_scroll_depth_pct>=100 THEN 1 ELSE 0 END) d100, "
                "AVG(NULLIF(max_scroll_depth_pct,0)) avg "
                "FROM %s WHERE %s AND max_scroll_depth_pct IS NOT NULL %s "
                "GROUP BY page_route ORDER BY total DESC LIMIT 12" % (PV, _PV_LIVE, pvc), pva):
                d = dict(r); t = int(d["total"] or 0)
                if not t: continue
                out["pages"].append({
                    "route": d["route"], "total": t,
                    "d25": int(d["d25"] or 0), "d50": int(d["d50"] or 0),
                    "d75": int(d["d75"] or 0), "d100": int(d["d100"] or 0),
                    "avg": round(float(d["avg"] or 0), 0)})
    except Exception as e:
        log.warning("website_scroll failed: %s", e)
    return out


def website_acquisition(window: str = "all") -> dict:
    """Referrer / UTM / direct sources feeding the site (Acquisition Constellation)."""
    out = {"referrers": [], "utm_sources": [], "direct": 0}
    try:
        bsc, bsa = _win_clause(window, "session_start_at")
        base = "FROM %s WHERE deleted_at IS NULL %s" % (BS, bsc)
        with auth_db.get_conn() as c:
            for r in c.execute(
                "SELECT CASE WHEN referrer_host IS NULL OR referrer_host='' THEN '(direct)' "
                "ELSE referrer_host END ref, COUNT(*) n %s GROUP BY ref ORDER BY n DESC LIMIT 10" % base, bsa):
                d = dict(r)
                if d["ref"] == "(direct)": out["direct"] = int(d["n"] or 0)
                else: out["referrers"].append({"label": d["ref"], "value": int(d["n"] or 0)})
            for r in c.execute(
                "SELECT utm_source s, COUNT(*) n %s AND utm_source IS NOT NULL AND utm_source<>'' "
                "GROUP BY s ORDER BY n DESC LIMIT 8" % base, bsa):
                d = dict(r); out["utm_sources"].append({"label": d["s"], "value": int(d["n"] or 0)})
    except Exception as e:
        log.warning("website_acquisition failed: %s", e)
    return out


def website_vitals(window: str = "all") -> dict:
    """Core Web Vitals distributions (LCP/CLS/INP/FCP/TTFB). Populates after
    W0.3 capture; returns coverage so the UI can show an honest 'measuring…'
    state rather than a fake green gauge."""
    metrics = {"lcp_ms": "LCP", "cls": "CLS", "inp_ms": "INP", "fcp_ms": "FCP", "ttfb_ms": "TTFB"}
    out = {"metrics": [], "coverage": 0, "total": 0}
    try:
        pvc, pva = _win_clause(window, "entered_at")
        with auth_db.get_conn() as c:
            total = _scalar(c, "SELECT COUNT(*) FROM %s WHERE %s %s" % (PV, _PV_LIVE, pvc), pva)
            out["total"] = total
            for col, label in metrics.items():
                vals = [dict(r)[col] for r in c.execute(
                    "SELECT %s FROM %s WHERE %s IS NOT NULL %s" % (col, PV, col, pvc), pva)]
                vals = [float(v) for v in vals if v is not None]
                if vals:
                    vals.sort()
                    p = _percentiles(vals, (50, 75, 90, 99))
                    out["metrics"].append({"key": col, "label": label, "n": len(vals),
                                           "p50": p["p50"], "p75": p["p75"], "p99": p["p99"]})
                else:
                    out["metrics"].append({"key": col, "label": label, "n": 0})
            measured = sum(1 for m in out["metrics"] if m["n"])
            out["coverage"] = round(100.0 * measured / max(1, len(metrics)))
    except Exception as e:
        log.warning("website_vitals failed: %s", e)
    return out


# ════════════════════════════════════════════════════════════════════════════
# GEOGRAPHY  (scans.latitude/longitude + geo_* — the ONLY real client locations)
# ════════════════════════════════════════════════════════════════════════════
# API request IPs in this deployment are all private/localhost (dev + LAN) and
# browser_sessions geo is NULL, so neither can be honestly geolocated. The scans
# table, however, carries genuine GPS lat/lon resolved to country/state/district
# at capture time. That is the accurate "where the product is used" footprint —
# what the Living Globe plots. admin_origins() states the IP truth plainly.
SCANS = "scans"


def _crop_of(diagnosis: Optional[str]) -> str:
    d = (diagnosis or "").lower()
    if d.startswith("okra"):
        return "okra"
    if d.startswith("brassica") or "cabbage" in d or "broccoli" in d:
        return "brassica"
    if d.startswith("tomato"):
        return "tomato"
    return "other"


def admin_geo(window: str = "all") -> dict:
    """Org-wide scan-origin geography: REAL lat/lon grouped by district, with a
    dominant diagnosis/crop per district and honest geolocation coverage. Window
    is ignored for now (scan corpus is small); kept for signature symmetry."""
    out = {"districts": [], "total_scans": 0, "geolocated": 0, "coverage_pct": 0.0,
           "countries": [], "crops": []}
    try:
        with auth_db.get_conn() as c:
            row = c.execute(
                "SELECT COUNT(*) total, SUM(CASE WHEN latitude IS NOT NULL "
                "AND geo_district IS NOT NULL THEN 1 ELSE 0 END) geo "
                "FROM %s WHERE deleted_at IS NULL" % SCANS).fetchone()
            d = dict(row); total = int(d.get("total") or 0); geo = int(d.get("geo") or 0)
            out["total_scans"] = total; out["geolocated"] = geo
            out["coverage_pct"] = round(100.0 * geo / total, 1) if total else 0.0

            rows = c.execute(
                "SELECT geo_cc, geo_state, geo_district, AVG(latitude) lat, "
                "AVG(longitude) lon, COUNT(*) n, AVG(confidence) conf "
                "FROM %s WHERE deleted_at IS NULL AND latitude IS NOT NULL "
                "AND geo_district IS NOT NULL "
                "GROUP BY geo_cc, geo_state, geo_district ORDER BY n DESC" % SCANS).fetchall()
            districts = []
            for r in rows:
                d = dict(r)
                dd = c.execute(
                    "SELECT diagnosis, COUNT(*) n FROM %s WHERE deleted_at IS NULL "
                    "AND geo_district=? AND geo_cc=? AND diagnosis IS NOT NULL "
                    "GROUP BY diagnosis ORDER BY n DESC LIMIT 1" % SCANS,
                    (d["geo_district"], d["geo_cc"])).fetchone()
                topdx = dict(dd)["diagnosis"] if dd else None
                districts.append({
                    "cc": d.get("geo_cc"), "state": d.get("geo_state"),
                    "district": d.get("geo_district"),
                    "lat": round(float(d["lat"]), 5), "lon": round(float(d["lon"]), 5),
                    "count": int(d["n"] or 0),
                    "avg_confidence": round(float(d["conf"] or 0), 3),
                    "top_diagnosis": topdx, "crop": _crop_of(topdx),
                })
            out["districts"] = districts

            cc_agg, crop_agg = {}, {}
            for dd in districts:
                cc_agg[dd["cc"]] = cc_agg.get(dd["cc"], 0) + dd["count"]
                crop_agg[dd["crop"]] = crop_agg.get(dd["crop"], 0) + dd["count"]
            out["countries"] = [{"cc": k, "count": v}
                                for k, v in sorted(cc_agg.items(), key=lambda x: -x[1])]
            out["crops"] = [{"crop": k, "count": v}
                            for k, v in sorted(crop_agg.items(), key=lambda x: -x[1])]
    except Exception as e:
        log.warning("admin_geo failed: %s", e)
    return out


def admin_inference_geo(window: str = "all") -> dict:
    """Inference (model-output) geography — the SAME real scan GPS as admin_geo,
    but the disease lens: per district the dominant diagnosis, severity mix, avg
    confidence and OOD count, plus overall disease / severity / OOD / confidence
    distributions. Powers the Inference globe + its side widgets."""
    out = {"districts": [], "diseases": [], "severities": [], "ood_total": 0,
           "total": 0, "geolocated": 0, "avg_confidence": 0.0, "coverage_pct": 0.0}
    try:
        with auth_db.get_conn() as c:
            row = c.execute(
                "SELECT COUNT(*) total, AVG(confidence) conf, "
                "SUM(CASE WHEN is_ood=1 OR ood_flag=1 THEN 1 ELSE 0 END) ood, "
                "SUM(CASE WHEN latitude IS NOT NULL AND geo_district IS NOT NULL THEN 1 ELSE 0 END) geo "
                "FROM %s WHERE deleted_at IS NULL" % SCANS).fetchone()
            d = dict(row); total = int(d.get("total") or 0); geo = int(d.get("geo") or 0)
            out["total"] = total; out["geolocated"] = geo
            out["ood_total"] = int(d.get("ood") or 0)
            out["avg_confidence"] = round(float(d.get("conf") or 0), 3)
            out["coverage_pct"] = round(100.0 * geo / total, 1) if total else 0.0

            # overall disease + severity distributions
            for r in c.execute(
                "SELECT diagnosis, COUNT(*) n, AVG(confidence) conf FROM %s "
                "WHERE deleted_at IS NULL AND diagnosis IS NOT NULL "
                "GROUP BY diagnosis ORDER BY n DESC" % SCANS):
                dd = dict(r)
                out["diseases"].append({"diagnosis": dd["diagnosis"], "count": int(dd["n"] or 0),
                                        "crop": _crop_of(dd["diagnosis"]),
                                        "avg_confidence": round(float(dd["conf"] or 0), 3)})
            for r in c.execute(
                "SELECT severity, COUNT(*) n FROM %s WHERE deleted_at IS NULL "
                "AND severity IS NOT NULL GROUP BY severity ORDER BY n DESC" % SCANS):
                dd = dict(r)
                out["severities"].append({"label": dd["severity"], "count": int(dd["n"] or 0)})

            # per-district disease lens
            rows = c.execute(
                "SELECT geo_cc, geo_state, geo_district, AVG(latitude) lat, AVG(longitude) lon, "
                "COUNT(*) n, AVG(confidence) conf, "
                "SUM(CASE WHEN is_ood=1 OR ood_flag=1 THEN 1 ELSE 0 END) ood "
                "FROM %s WHERE deleted_at IS NULL AND latitude IS NOT NULL "
                "AND geo_district IS NOT NULL "
                "GROUP BY geo_cc, geo_state, geo_district ORDER BY n DESC" % SCANS).fetchall()
            for r in rows:
                dd = dict(r)
                dx = c.execute(
                    "SELECT diagnosis, COUNT(*) n FROM %s WHERE deleted_at IS NULL "
                    "AND geo_district=? AND geo_cc=? AND diagnosis IS NOT NULL "
                    "GROUP BY diagnosis ORDER BY n DESC LIMIT 1" % SCANS,
                    (dd["geo_district"], dd["geo_cc"])).fetchone()
                topdx = dict(dx)["diagnosis"] if dx else None
                sev = {}
                for sr in c.execute(
                    "SELECT severity, COUNT(*) n FROM %s WHERE deleted_at IS NULL "
                    "AND geo_district=? AND geo_cc=? AND severity IS NOT NULL "
                    "GROUP BY severity" % SCANS, (dd["geo_district"], dd["geo_cc"])):
                    sd = dict(sr); sev[sd["severity"]] = int(sd["n"] or 0)
                out["districts"].append({
                    "cc": dd.get("geo_cc"), "state": dd.get("geo_state"),
                    "district": dd.get("geo_district"),
                    "lat": round(float(dd["lat"]), 5), "lon": round(float(dd["lon"]), 5),
                    "count": int(dd["n"] or 0), "avg_confidence": round(float(dd["conf"] or 0), 3),
                    "ood": int(dd["ood"] or 0), "severity_mix": sev,
                    "top_diagnosis": topdx, "crop": _crop_of(topdx),
                })
    except Exception as e:
        log.warning("admin_inference_geo failed: %s", e)
    return out


def admin_origins(window: str = "all") -> dict:
    """The honest API client-origin truth: classify every request-log IP as
    localhost / private-LAN / public-internet. In this deployment everything is
    private (local dev + 10.16.x.x LAN), so origins are NOT geolocatable — the
    Traffic UI says so plainly instead of inventing map pins.

    Enriched for the API origin-network page: `hosts` (per-IP count + error rate
    + top endpoints) and `series` (origin-class composition over time)."""
    import ipaddress
    out = {"total": 0, "private": 0, "public": 0, "private_pct": 0.0,
           "buckets": [], "ips": [], "hosts": [], "series": []}

    def _classify(ip):
        try:
            a = ipaddress.ip_address(ip)
            return "loopback" if a.is_loopback else ("lan" if (a.is_private or a.is_link_local) else "public")
        except Exception:
            return "lan"

    try:
        wc, wa = _win_clause(window)
        agg = {"loopback": 0, "lan": 0, "public": 0}
        ips = []
        host_err = {}; host_paths = {}
        with auth_db.get_conn() as c:
            rows = c.execute(
                "SELECT ip, COUNT(*) n, SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) e "
                "FROM %s WHERE ip IS NOT NULL %s GROUP BY ip ORDER BY n DESC" % (REQ, wc), wa).fetchall()
            # per-host top endpoints (single pass, aggregated in Python)
            for r in c.execute(
                "SELECT ip, path, COUNT(*) n FROM %s WHERE ip IS NOT NULL AND path IS NOT NULL %s "
                "GROUP BY ip, path" % (REQ, wc), wa):
                d = dict(r); ip = (d["ip"] or "").split(",")[0].strip()
                host_paths.setdefault(ip, []).append((d["path"], int(d["n"] or 0)))
            # composition over time
            keys, sub, _hourly = _buckets(window)
            comp = {k: {"loopback": 0, "lan": 0, "public": 0} for k in keys}
            for r in c.execute(
                "SELECT substr(%s,1,%d) b, ip, COUNT(*) n FROM %s WHERE ip IS NOT NULL %s "
                "GROUP BY b, ip" % (_RDT_BARE, sub, REQ, wc), wa):
                d = dict(r); b = d["b"]; ip = (d["ip"] or "").split(",")[0].strip()
                if b in comp:
                    comp[b][_classify(ip)] += int(d["n"] or 0)
            out["series"] = [{"t": k, "loopback": comp[k]["loopback"], "lan": comp[k]["lan"],
                              "public": comp[k]["public"]} for k in keys]

        total = 0
        for r in rows:
            d = dict(r); ip = (d["ip"] or "").split(",")[0].strip()
            n = int(d["n"] or 0); e = int(d["e"] or 0); total += n
            cls = _classify(ip); agg[cls] += n
            if len(ips) < 12:
                ips.append({"ip": ip, "count": n, "class": cls})
            paths = sorted(host_paths.get(ip, []), key=lambda x: -x[1])[:3]
            out["hosts"].append({
                "ip": ip, "count": n, "class": cls,
                "error_rate": round(100.0 * e / n, 1) if n else 0.0,
                "top_paths": [{"path": p, "count": pn} for p, pn in paths],
            })
        out["hosts"] = out["hosts"][:14]
        out["total"] = total
        out["private"] = agg["loopback"] + agg["lan"]
        out["public"] = agg["public"]
        out["private_pct"] = round(100.0 * out["private"] / total, 1) if total else 0.0
        out["buckets"] = [
            {"label": "localhost", "count": agg["loopback"]},
            {"label": "private LAN", "count": agg["lan"]},
            {"label": "public internet", "count": agg["public"]},
        ]
        out["ips"] = ips
    except Exception as e:
        log.warning("admin_origins failed: %s", e)
    return out


# ════════════════════════════════════════════════════════════════════════════
# STATS DECK  (18-card observability deck — one real-data bundle per dataset)
# ════════════════════════════════════════════════════════════════════════════
# Each card gets a focused real payload, or {"state":"calibrating","reason":..}
# when the analytic genuinely needs more signal than the data carries. Nothing
# is fabricated — a quiet metric returns an honest empty/low-confidence shape.
import statistics as _stats


def _pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 4:
        return None
    xs, ys = xs[:n], ys[:n]
    try:
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
        if dx == 0 or dy == 0:
            return None
        return round(num / (dx * dy), 3)
    except Exception:
        return None


def _deck_api(window):
    cards = {}
    ov = traffic_api_overview(window)
    eps = (traffic_endpoints(window).get("endpoints") or [])
    seq = (traffic_sequences(window if window != "all" else "30d").get("edges") or [])
    lat = traffic_latency(window if window != "all" else "30d")
    series = ov.get("series") or []
    counts = [float(s.get("c") or 0) for s in series]

    changes = []
    for i in range(1, len(series)):
        prev, cur = counts[i - 1], counts[i]
        if prev > 0:
            pct = round(100.0 * (cur - prev) / prev, 1)
            if abs(pct) >= 25 and cur >= 3:
                changes.append({"t": series[i].get("t"), "from": int(prev), "to": int(cur), "pct": pct})
    changes.sort(key=lambda c: -abs(c["pct"]))
    cards["change_detection"] = {"series": series, "changes": changes[:8], "biggest": changes[0] if changes else None}

    rid = [r for r in (lat.get("ridges") or []) if r.get("n")]
    if len(rid) >= 4 and len(counts) >= 4:
        vol = [float(r.get("n") or 0) for r in rid]
        p95s = [float(r.get("p95") or 0) for r in rid]
        r = _pearson(vol, p95s)
        cards["correlation"] = {"pairs": [{"a": "volume", "b": "p95 latency", "r": r}]} if r is not None else {"state": "calibrating", "reason": "flat signal"}
    else:
        cards["correlation"] = {"state": "calibrating", "reason": "need 4+ days of latency"}

    ranked = sorted(eps, key=lambda e: (-(e.get("error_rate") or 0), -(e.get("p95") or 0)))
    cards["regressions"] = {"items": [{"path": e["path"], "error_rate": e.get("error_rate", 0), "p95": e.get("p95", 0), "n": e.get("requests", 0)} for e in ranked[:6] if (e.get("error_rate") or 0) > 0],
                            "baseline": "current standings; prior-window baseline calibrating"}
    healthy = sorted([e for e in eps if (e.get("error_rate") or 0) == 0], key=lambda e: (e.get("p95") or 0))
    cards["improvements"] = {"items": [{"path": e["path"], "p95": e.get("p95", 0), "n": e.get("requests", 0)} for e in healthy[:6]]}

    seg = {"production": 0, "testing": 0, "automation": 0}
    try:
        with auth_db.get_conn() as c:
            wc, wa = _win_clause(window)
            for r in c.execute("SELECT ua, key_id, COUNT(*) n FROM %s WHERE 1=1 %s GROUP BY ua, key_id" % (REQ, wc), wa):
                d = dict(r); ua = (d.get("ua") or "").lower(); n = int(d["n"] or 0)
                if any(k in ua for k in ("bot", "curl", "python", "wget", "spider", "scan")):
                    seg["automation"] += n
                elif "test" in (d.get("key_id") or "").lower() or "test" in ua:
                    seg["testing"] += n
                else:
                    seg["production"] += n
    except Exception:
        pass
    cards["segments"] = {"segments": [{"label": k, "count": v} for k, v in sorted(seg.items(), key=lambda x: -x[1]) if v]}

    fs, ls = [], []
    try:
        with auth_db.get_conn() as c:
            wc, wa = _win_clause(window)
            for r in c.execute("SELECT path, MIN(%s) f, MAX(%s) l, COUNT(*) n FROM %s WHERE path IS NOT NULL %s GROUP BY path"
                               % (_RDT_BARE, _RDT_BARE, REQ, wc), wa):
                d = dict(r); fs.append({"path": d["path"], "ts": d["f"], "n": int(d["n"] or 0)}); ls.append({"path": d["path"], "ts": d["l"], "n": int(d["n"] or 0)})
    except Exception:
        pass
    fs.sort(key=lambda x: x["ts"] or "", reverse=True)
    ls.sort(key=lambda x: x["ts"] or "")
    cards["first_seen"] = {"items": fs[:8]}
    cards["last_seen"] = {"items": ls[:8]}

    if len(counts) >= 3 and sum(counts) > 0:
        m = sum(counts) / len(counts); sd = _stats.pstdev(counts); cv = sd / m if m else 0
        cards["volatility"] = {"index": round(cv, 2), "series": series, "level": ("high" if cv > 1 else ("medium" if cv > 0.4 else "low"))}
    else:
        cards["volatility"] = {"state": "calibrating", "reason": "need 3+ buckets"}

    deg = {}
    for e in seq:
        deg[e["from"]] = deg.get(e["from"], 0) + e.get("weight", 0); deg[e["to"]] = deg.get(e["to"], 0) + e.get("weight", 0)
    risk = []; maxn = max([e.get("requests", 0) for e in eps] or [1]); maxd = max(deg.values() or [1])
    for e in eps:
        d = deg.get(e["path"], 0); score = round((d / maxd) * 0.6 + (e.get("requests", 0) / maxn) * 0.4, 2)
        risk.append({"path": e["path"], "degree": d, "n": e.get("requests", 0), "risk": score})
    risk.sort(key=lambda x: -x["risk"])
    cards["dependency_risk"] = {"items": risk[:8]}

    fp = []
    try:
        with auth_db.get_conn() as c:
            wc, wa = _win_clause(window)
            for r in c.execute("SELECT path, status_code, COUNT(*) n FROM %s WHERE status_code>=400 %s GROUP BY path, status_code ORDER BY n DESC LIMIT 12" % (REQ, wc), wa):
                d = dict(r); fp.append({"path": d["path"], "status": int(d["status_code"] or 0), "count": int(d["n"] or 0)})
    except Exception:
        pass
    cards["error_fingerprints"] = {"items": fp}

    if len(counts) >= 4:
        a = 0.4; ew = counts[0]
        for v in counts[1:]:
            ew = a * v + (1 - a) * ew
        trend = "rising" if counts[-1] > ew else ("falling" if counts[-1] < ew else "steady")
        cards["forecast"] = {"series": series, "projection": round(ew, 1), "trend": trend, "note": "EWMA projection, not a guarantee"}
    else:
        cards["forecast"] = {"state": "calibrating", "reason": "need 4+ buckets"}

    if len(counts) >= 3:
        base = round(sum(counts) / len(counts), 1)
        cards["baseline"] = {"series": series, "baseline": base, "now": int(counts[-1]), "delta_pct": round(100.0 * (counts[-1] - base) / base, 1) if base else 0}
    else:
        cards["baseline"] = {"state": "calibrating", "reason": "need 3+ buckets"}

    routes = sorted(seq, key=lambda e: -e.get("weight", 0))[:8]
    cards["journeys"] = {"edges": seq[:30], "routes": [{"from": e["from"], "to": e["to"], "weight": e.get("weight", 0)} for e in routes]}

    rec = []
    try:
        keys, subn, _h = _buckets(window)
        with auth_db.get_conn() as c:
            wc, wa = _win_clause(window)
            agg = {k: [0, 0] for k in keys}
            for r in c.execute("SELECT substr(%s,1,%d) b, COUNT(*) n, SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) e FROM %s WHERE 1=1 %s GROUP BY b" % (_RDT_BARE, subn, REQ, wc), wa):
                d = dict(r); b = d["b"]
                if b in agg:
                    agg[b] = [int(d["n"] or 0), int(d["e"] or 0)]
            rec = [{"t": k, "err_rate": round(100.0 * agg[k][1] / agg[k][0], 1) if agg[k][0] else 0} for k in keys]
    except Exception:
        pass
    cards["recovery"] = {"series": rec}

    ltc = ov.get("latency") or {}
    cards["latency_attribution"] = {"percentiles": {"p50": ltc.get("p50", 0), "p90": ltc.get("p90", 0), "p95": ltc.get("p95", 0), "p99": ltc.get("p99", 0)}, "ridges": (lat.get("ridges") or [])[-7:]}

    cards["alert_simulator"] = {"metrics": [
        {"key": "error_rate", "label": "Error rate", "value": ov.get("error_rate", 0), "unit": "%", "suggested": 5},
        {"key": "p95", "label": "p95 latency", "value": ltc.get("p95", 0), "unit": "ms", "suggested": max(500, int((ltc.get("p95", 0) or 0) * 1.3))},
        {"key": "throughput", "label": "Throughput", "value": ov.get("throughput_per_min", 0), "unit": "/min", "suggested": 0}]}

    def _conf(n):
        return round(max(0.0, min(1.0, 1 - 1.0 / (1 + (n / 30.0)))), 2) if n else 0.0
    latn = sum(int(r.get("n") or 0) for r in (lat.get("ridges") or []))
    cards["metric_confidence"] = {"metrics": [
        {"label": "Requests", "n": ov.get("requests", 0), "confidence": _conf(ov.get("requests", 0))},
        {"label": "Error rate", "n": ov.get("requests", 0), "confidence": _conf(ov.get("requests", 0))},
        {"label": "Latency", "n": latn, "confidence": _conf(latn)}]}

    heads = []
    if cards["change_detection"]["biggest"]:
        b = cards["change_detection"]["biggest"]
        heads.append({"kind": ("surge" if b["pct"] > 0 else "drop"), "text": "Traffic %s %s%% at %s" % (("surged" if b["pct"] > 0 else "fell"), abs(b["pct"]), (b["t"] or "")[-5:])})
    if eps:
        busiest = max(eps, key=lambda e: e.get("requests", 0))
        heads.append({"kind": "info", "text": "%s is the busiest endpoint (%d requests)" % (busiest["path"], busiest.get("requests", 0))})
    if cards["regressions"]["items"]:
        w = cards["regressions"]["items"][0]
        heads.append({"kind": "alert", "text": "%s has the highest error rate at %s%%" % (w["path"], w["error_rate"])})
    heads.append({"kind": "info", "text": "Org-wide error rate is %s%% over %s requests" % (ov.get("error_rate", 0), ov.get("requests", 0))})
    cards["narrative"] = {"headlines": heads}
    return cards


def _deck_website(window):
    cards = {}
    ov = website_overview(window)
    pages = (website_pages(window).get("pages") or [])
    jr = website_journey(window if window != "all" else "30d")
    cards["change_detection"] = {"state": "calibrating", "reason": "per-bucket visit series pending"}
    cards["correlation"] = {"state": "calibrating", "reason": "needs web vitals (0% coverage)"}
    cards["regressions"] = {"items": [{"path": p["route"], "error_rate": 0, "p95": p.get("avg_active_s", 0), "n": p.get("visits", 0)} for p in sorted(pages, key=lambda x: -(x.get("bounce_rate") or 0))[:6]], "baseline": "by bounce; latency n/a for web"}
    cards["improvements"] = {"items": [{"path": p["route"], "p95": p.get("avg_active_s", 0), "n": p.get("visits", 0)} for p in sorted(pages, key=lambda x: -(x.get("avg_active_s") or 0))[:6]]}
    cards["segments"] = {"segments": []}
    cards["first_seen"] = {"items": [{"path": p["route"], "ts": None, "n": p.get("visits", 0)} for p in pages[:8]]}
    cards["last_seen"] = {"items": [{"path": p["route"], "ts": None, "n": p.get("visits", 0)} for p in pages[:8]]}
    cards["volatility"] = {"state": "calibrating", "reason": "per-bucket visit series pending"}
    cards["dependency_risk"] = {"items": [{"path": p["route"], "degree": 0, "n": p.get("visits", 0), "risk": 0} for p in sorted(pages, key=lambda x: -(x.get("visits") or 0))[:8]]}
    cards["error_fingerprints"] = {"items": []}
    cards["forecast"] = {"state": "calibrating", "reason": "per-bucket visit series pending"}
    cards["baseline"] = {"state": "calibrating", "reason": "per-bucket visit series pending"}
    cards["journeys"] = {"edges": (jr.get("edges") or [])[:30], "routes": [{"from": e.get("from"), "to": e.get("to"), "weight": e.get("weight", e.get("count", 0))} for e in (jr.get("edges") or [])[:8]]}
    cards["recovery"] = {"series": []}
    cards["latency_attribution"] = {"percentiles": {"p50": ov.get("avg_active_s", 0)}, "ridges": [], "note": "web = time-on-page, not latency"}
    cards["alert_simulator"] = {"metrics": [{"key": "bounce", "label": "Bounce rate", "value": ov.get("bounce_rate", 0), "unit": "%", "suggested": 60}]}
    cards["metric_confidence"] = {"metrics": [{"label": "Visits", "n": ov.get("visits", 0), "confidence": round(max(0.0, min(1.0, 1 - 1.0 / (1 + (ov.get("visits", 0) / 30.0)))), 2)}]}
    heads = []
    if pages:
        top = max(pages, key=lambda p: p.get("visits", 0))
        heads.append({"kind": "info", "text": "%s is the most-visited page (%d views)" % (top["route"], top.get("visits", 0))})
    heads.append({"kind": "info", "text": "Bounce rate is %s%% across %s sessions" % (ov.get("bounce_rate", 0), ov.get("sessions", 0))})
    cards["narrative"] = {"headlines": heads}
    return cards


def admin_deck(dataset: str = "api", window: str = "all") -> dict:
    """One real-data bundle for the 18-card Stats deck. Real where computable,
    {state:'calibrating'} where the analytic needs more signal. Never fabricates."""
    out = {"dataset": dataset, "cards": {}}
    try:
        out["cards"] = _deck_website(window) if dataset == "website" else _deck_api(window)
    except Exception as e:
        log.warning("admin_deck failed: %s", e)
    return out


# ── Deck layout persistence (card order + pins, per admin account) ──────────────
# [STATS DECK F2] User chose backend persistence. One row per (user_id, dataset).
import json as _json

_DECK_N = 18

def _ensure_deck_layout_table():
    with auth_db.get_conn() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS admin_deck_layout ("
            "  user_id INTEGER NOT NULL,"
            "  dataset TEXT NOT NULL,"
            "  order_json TEXT,"
            "  pins_json TEXT,"
            "  updated_at TEXT,"
            "  PRIMARY KEY (user_id, dataset)"
            ")"
        )

def get_deck_layout(user_id: int, dataset: str) -> dict:
    """Return {'order': [int]|None, 'pins': [int]} for this admin + dataset."""
    dataset = "website" if dataset == "website" else "api"
    _ensure_deck_layout_table()
    with auth_db.get_conn() as c:
        r = c.execute(
            "SELECT order_json, pins_json FROM admin_deck_layout "
            "WHERE user_id = ? AND dataset = ? LIMIT 1",
            [int(user_id), dataset],
        ).fetchone()
    if not r:
        return {"order": None, "pins": []}
    try:
        order = _json.loads(r[0]) if r[0] else None
    except Exception:
        order = None
    try:
        pins = _json.loads(r[1]) if r[1] else []
    except Exception:
        pins = []
    return {"order": _valid_order(order), "pins": _valid_pins(pins)}

def set_deck_layout(user_id: int, dataset: str, order, pins) -> dict:
    """Persist order + pins. Validates order is a permutation of 0..17 and pins a subset."""
    dataset = "website" if dataset == "website" else "api"
    order = _valid_order(order)
    pins = _valid_pins(pins)
    _ensure_deck_layout_table()
    now = datetime.now(timezone.utc).isoformat()
    with auth_db.get_conn() as c:
        c.execute(
            "INSERT INTO admin_deck_layout (user_id, dataset, order_json, pins_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, dataset) DO UPDATE SET "
            "  order_json = excluded.order_json, pins_json = excluded.pins_json, "
            "  updated_at = excluded.updated_at",
            [int(user_id), dataset,
             _json.dumps(order) if order is not None else None,
             _json.dumps(pins), now],
        )
    return {"order": order, "pins": pins}

def _valid_order(order):
    if not isinstance(order, list):
        return None
    try:
        ints = [int(x) for x in order]
    except Exception:
        return None
    if sorted(ints) != list(range(_DECK_N)):
        return None
    return ints

def _valid_pins(pins):
    if not isinstance(pins, list):
        return []
    out = []
    for x in pins:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < _DECK_N and i not in out:
            out.append(i)
    return out[:4]
