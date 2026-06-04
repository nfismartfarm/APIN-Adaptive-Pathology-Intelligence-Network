"""Phase 9.B — Account-wide + per-key usage observability endpoints.

Five new endpoints under `/api/account/usage/*`:

    GET /api/account/usage/summary
        KPIs over a range: requests, errors, rate_limited, bytes_in/out,
        avg/p50/p95/p99 latency, plus a `delta_pct` comparing the previous
        window (e.g. last 24 h vs 24-48 h ago). Account-wide by default;
        narrowable by ?key_id, ?endpoint, ?status, ?env.

    GET /api/account/usage/timeseries
        Bucketed series for charting. Granularity auto-picks based on range:
        1m for ≤1h, 5m for ≤24h, 1h for ≤7d, 1d for >7d. Mode controls the
        shape: total (one series), by_status (stacked 2xx/4xx/5xx/429),
        by_endpoint (top 5 endpoints + Other), latency (p50/p95/p99 lines),
        errors (5xx + 429 rate).

    GET /api/account/usage/top
        Top-N ranking over a range. dim ∈ {keys, endpoints, ips, statuses,
        error_codes, methods}. Returns rows with .label / .count / .pct.

    GET /api/account/usage/requests
        Paginated raw request log across ALL of the user's keys, with the
        same filter params used everywhere else. Used by both the
        "Recent requests" panel and the CSV export.

    GET /api/account/usage/minute-detail
        Drill-down for one specific (key_id, minute) cell on the chart.
        Returns the raw request rows that landed in that minute plus the
        aggregate row for context. Used by the click-time-series drawer.

The per-key routes added in Phase 8.D (`/api/account/keys/{public_id}/usage`
and `/requests`) are unchanged in URL but extended with the same filter
params via `list_key_usage_minute` / `list_key_requests` enhancements.

Auth: every endpoint is session-cookie (not Bearer/X-API-Key) per
PDA-F04 — TokenFormatMiddleware will reject API-key auth at the
account/* prefix.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request, Query

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import _session_helpers as _sh

log = logging.getLogger("apin_v2.account.routes_usage")

router = APIRouter(prefix="/api/account/usage", tags=["account/usage"])

_get_session_user = _sh.get_session_user


# ── Range / granularity helpers ───────────────────────────────────────────

# Allowed range strings. Map → seconds. "24h" is the default everywhere.
_RANGE_SECONDS: dict[str, int] = {
    "15m":   15 * 60,
    "1h":    60 * 60,
    "6h":    6 * 60 * 60,
    "24h":   24 * 60 * 60,
    "7d":    7 * 24 * 60 * 60,
    "30d":   30 * 24 * 60 * 60,
}

# Default granularity picker — pick the smallest bucket that yields at most
# ~360 buckets across the range so charts stay readable.
def _auto_granularity_seconds(range_sec: int) -> int:
    if range_sec <= 60 * 60:           # ≤1h → 1m buckets
        return 60
    if range_sec <= 6 * 60 * 60:       # ≤6h → 5m
        return 5 * 60
    if range_sec <= 24 * 60 * 60:      # ≤24h → 5m
        return 5 * 60
    if range_sec <= 7 * 24 * 60 * 60:  # ≤7d → 1h
        return 60 * 60
    return 24 * 60 * 60                # >7d → 1d


_GRANULARITY_LABEL_TO_SECONDS: dict[str, int] = {
    "1m":  60,
    "5m":  5 * 60,
    "15m": 15 * 60,
    "1h":  60 * 60,
    "6h":  6 * 60 * 60,
    "1d":  24 * 60 * 60,
}


def _resolve_range(range_str: str) -> int:
    """range_str → seconds. Raises ApiError on invalid input."""
    rs = (range_str or "24h").strip().lower()
    if rs not in _RANGE_SECONDS:
        raise ApiError("invalid_parameter",
                        f"unknown range {rs!r}; allowed: "
                        f"{sorted(_RANGE_SECONDS)}")
    return _RANGE_SECONDS[rs]


def _resolve_granularity(label: Optional[str], range_sec: int) -> int:
    if not label:
        return _auto_granularity_seconds(range_sec)
    g = label.strip().lower()
    if g not in _GRANULARITY_LABEL_TO_SECONDS:
        raise ApiError("invalid_parameter",
                        f"unknown granularity {g!r}; allowed: "
                        f"{sorted(_GRANULARITY_LABEL_TO_SECONDS)}")
    return _GRANULARITY_LABEL_TO_SECONDS[g]


def _bucket_format(granularity_sec: int) -> str:
    """SQLite strftime mask matching the bucket size."""
    if granularity_sec < 60 * 60:
        # 1m / 5m / 15m — round down to nearest multiple of granularity
        # using `unixepoch / N * N`. Computed inline in SQL below.
        return "%Y-%m-%d %H:%M:00"
    if granularity_sec < 24 * 60 * 60:
        # 1h / 6h — hour-floor; 6h floor by hour mod 6 (handled in SQL).
        return "%Y-%m-%d %H:00:00"
    return "%Y-%m-%d 00:00:00"


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/summary")
@api_endpoint("/api/account/usage/summary")
async def usage_summary(request: Request,
                         range: str = Query("24h"),
                         key_id: Optional[str] = Query(None),
                         endpoint: Optional[str] = Query(None),
                         status: Optional[str] = Query(None),
                         env: Optional[str] = Query(None)):
    """Account-wide KPIs over the requested window, with delta-vs-previous.

    Returns:
      {
        "range": "24h", "window_seconds": 86400,
        "kpis": {
          "requests":       {"current": 1234, "previous": 1100, "delta_pct": 12.2},
          "errors":         {"current": 12,   "previous": 8,    "delta_pct": 50.0},
          "rate_limited":   {"current": 3,    "previous": 0,    "delta_pct": None},
          "quota_blocked":  {"current": 0,    "previous": 0,    "delta_pct": 0.0},
          "bytes_in":       {"current": ..., "previous": ..., "delta_pct": ...},
          "bytes_out":      {"current": ..., "previous": ..., "delta_pct": ...},
          "latency_p50_ms": {"current": 110, "previous": 105, "delta_pct": 4.8},
          "latency_p95_ms": {"current": 320, "previous": 290, "delta_pct": 10.3},
          "latency_p99_ms": {"current": 510, "previous": 470, "delta_pct": 8.5},
          "active_keys":    {"current": 4,   "previous": 3,    "delta_pct": 33.3},
          "error_rate":     {"current": 0.97, "previous": 0.72, "delta_pct": 34.7},
        },
        "computed_at": "2026-...Z"
      }

    The previous window has the same length as the current and immediately
    precedes it (i.e. last 24h vs the 24h before that, not vs yesterday).
    """
    user = _get_session_user(request)
    user_id = int(user["id"])
    range_sec = _resolve_range(range)

    # NOTE on filters: `endpoint` (path prefix) and `status` (HTTP class)
    # require scanning the raw log. `key_id` and `env` narrow at the
    # aggregate layer (api_key_usage_minute is per-key; env is on api_keys).
    # We do the full-log scan only when an endpoint/status filter is set,
    # falling back to the per-minute aggregate otherwise — orders of
    # magnitude cheaper.
    summary = auth_db.compute_usage_summary(
        user_id=user_id,
        range_seconds=range_sec,
        key_id=key_id,
        env=env,
        endpoint=endpoint,
        status=status,
    )
    # 9.N.T29 · lifetime probe — lets the empty state distinguish a brand-new
    # key (never any traffic) from a window that just happens to be quiet.
    # Scoped to key/env (same as summary), not endpoint/status.
    summary["lifetime"] = auth_db.compute_usage_lifetime(
        user_id=user_id, key_id=key_id, env=env)
    return summary


@router.get("/timeseries")
@api_endpoint("/api/account/usage/timeseries")
async def usage_timeseries(request: Request,
                            range: str = Query("24h"),
                            granularity: Optional[str] = Query(None),
                            mode: str = Query("total"),
                            compare: Optional[str] = Query(None),
                            key_id: Optional[str] = Query(None),
                            endpoint: Optional[str] = Query(None),
                            status: Optional[str] = Query(None),
                            env: Optional[str] = Query(None)):
    """Bucketed time-series for charting.

    `mode` ∈ {total, by_status, by_endpoint, latency, errors, bytes}.

    Returns:
      {
        "range": "24h", "granularity_seconds": 300,
        "mode": "by_status",
        "buckets": [
          {"t": "2026-05-26 13:00:00", "values": {"2xx": 120, "4xx": 3, "5xx": 1, "429": 0}},
          ...
        ],
        "series_meta": [
          {"key": "2xx", "label": "2xx", "color_token": "ok"},
          {"key": "4xx", "label": "4xx", "color_token": "warn"},
          ...
        ]
      }
    """
    user = _get_session_user(request)
    user_id = int(user["id"])
    range_sec = _resolve_range(range)
    gran_sec = _resolve_granularity(granularity, range_sec)

    allowed_modes = {"total", "by_status", "by_endpoint",
                      "latency", "errors", "bytes"}
    if mode not in allowed_modes:
        raise ApiError("invalid_parameter",
                        f"unknown mode {mode!r}; allowed: {sorted(allowed_modes)}")

    result = auth_db.compute_usage_timeseries(
        user_id=user_id, range_seconds=range_sec,
        granularity_seconds=gran_sec, mode=mode,
        key_id=key_id, endpoint=endpoint, status=status, env=env,
    )
    # 9.N.10 · Compare mode — fetch the SAME-length window from the
    # previous period and inline as `prev_buckets`. Frontend overlays
    # it as a dashed series. Honours all the same filters.
    if compare and compare in ("prev", "prev_24h", "prev_7d", "prev_period"):
        prev = auth_db.compute_usage_timeseries(
            user_id=user_id, range_seconds=range_sec,
            granularity_seconds=gran_sec, mode=mode,
            key_id=key_id, endpoint=endpoint, status=status, env=env,
            offset_seconds=range_sec,  # shift backwards by one window
        )
        result["prev_buckets"] = prev.get("buckets", [])
        result["compare"] = compare
    return result


@router.get("/top")
@api_endpoint("/api/account/usage/top")
async def usage_top(request: Request,
                     range: str = Query("24h"),
                     dim: str = Query("keys"),
                     limit: int = Query(10, ge=1, le=50),
                     key_id: Optional[str] = Query(None),
                     env: Optional[str] = Query(None),
                     status: Optional[str] = Query(None),
                     endpoint: Optional[str] = Query(None)):
    """Top-N ranking across `dim`. Returns rows sorted DESC by count.

    dim ∈ {keys, endpoints, ips, statuses, error_codes, methods}.

    Returns:
      {
        "range": "24h", "dim": "endpoints", "limit": 10,
        "items": [
          {"label": "/api/predict/full", "count": 980, "pct": 79.4,
           "extra": {"p50_ms": 120, "avg_latency_ms": 135}},
          ...
        ],
        "total_for_pct": 1234
      }
    """
    user = _get_session_user(request)
    user_id = int(user["id"])
    range_sec = _resolve_range(range)
    allowed_dims = {"keys", "endpoints", "ips", "statuses",
                     "error_codes", "methods"}
    if dim not in allowed_dims:
        raise ApiError("invalid_parameter",
                        f"unknown dim {dim!r}; allowed: {sorted(allowed_dims)}")
    return auth_db.compute_usage_top(
        user_id=user_id, range_seconds=range_sec,
        dim=dim, limit=int(limit), key_id=key_id, env=env,
        status=status, endpoint=endpoint,
    )


@router.get("/requests")
@api_endpoint("/api/account/usage/requests")
async def usage_requests(request: Request,
                          range: str = Query("24h"),
                          limit: int = Query(50, ge=1, le=200),
                          cursor: Optional[int] = Query(None, ge=1),
                          key_id: Optional[str] = Query(None),
                          method: Optional[str] = Query(None),
                          status: Optional[str] = Query(None),
                          endpoint: Optional[str] = Query(None),
                          env: Optional[str] = Query(None),
                          format: Optional[str] = Query(None)):
    """Paginated raw request log across ALL of the user's keys.

    `status` accepts a class shorthand (2xx/3xx/4xx/5xx/429) OR a literal
    code (e.g. "404"). `endpoint` is a path substring match. `method` is
    exact (GET/POST/etc.). `key_id` narrows to one key.

    `format=csv` returns text/csv instead of JSON — used by the Export
    button on the Usage page (per U5 "all data and logs, fully detailed").
    """
    user = _get_session_user(request)
    user_id = int(user["id"])
    range_sec = _resolve_range(range)

    items = auth_db.list_user_request_log(
        user_id=user_id, range_seconds=range_sec,
        limit=(limit if format != "csv" else 10_000),
        cursor=cursor, key_id=key_id, method=method,
        status=status, endpoint=endpoint, env=env,
    )

    if format == "csv":
        # Stream a CSV. Stdlib csv module + a StringIO buffer — small
        # enough for our 10k row cap that we don't need streaming.
        import csv
        import io
        from fastapi.responses import Response
        buf = io.StringIO()
        # Columns chosen for forensic completeness per user U5.
        cols = ["id", "timestamp", "key_public_id", "key_name", "method",
                 "path", "status_code", "error_code", "latency_ms",
                 "bytes_in", "bytes_out", "ip", "ua", "via"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in items:
            w.writerow(r)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition":
                    'attachment; filename="apin-usage-requests.csv"',
            },
        )

    return {"items": items, "count": len(items),
            "range": range, "has_more": len(items) == limit}


@router.get("/minute-detail")
@api_endpoint("/api/account/usage/minute-detail")
async def usage_minute_detail(request: Request,
                               minute_ts: str = Query(...),
                               key_id: Optional[str] = Query(None),
                               endpoint: Optional[str] = Query(None),
                               status: Optional[str] = Query(None),
                               limit: int = Query(100, ge=1, le=500)):
    """Drill-down for one (key_id, minute) cell.

    Returns the aggregate row for that minute (from api_key_usage_minute)
    plus the raw requests that landed in it (up to `limit`), filtered the
    same way as the timeseries. Used by the click-time-series drawer.

    minute_ts format: "YYYY-MM-DD HH:MM:00" UTC.
    """
    user = _get_session_user(request)
    user_id = int(user["id"])

    if not minute_ts or len(minute_ts) < 19:
        raise ApiError("invalid_parameter",
                        "minute_ts must be 'YYYY-MM-DD HH:MM:00'")

    return auth_db.compute_minute_detail(
        user_id=user_id, minute_ts=minute_ts,
        key_id=key_id, endpoint=endpoint, status=status,
        limit=int(limit),
    )


@router.get("/latency-drill")
@api_endpoint("/api/account/usage/latency-drill")
async def usage_latency_drill(request: Request,
                                range: str = Query("24h"),
                                min_ms: int = Query(..., ge=0),
                                max_ms: Optional[int] = Query(None, ge=0),
                                key_id: Optional[str] = Query(None),
                                env: Optional[str] = Query(None),
                                limit: int = Query(100, ge=1, le=500)):
    """9.I — drill into a clicked latency-histogram bar."""
    user = _get_session_user(request)
    user_id = int(user["id"])
    range_sec = _resolve_range(range)
    return auth_db.compute_latency_bucket_drill(
        user_id=user_id, range_seconds=range_sec,
        min_ms=int(min_ms), max_ms=(int(max_ms) if max_ms is not None else None),
        key_id=key_id, env=env, limit=int(limit),
    )


@router.get("/request/{request_id}")
@api_endpoint("/api/account/usage/request/{request_id}")
async def usage_request_detail(request: Request, request_id: int):
    """9.I — Full request detail card.

    Returns one row from `api_key_request_log` (joined with `api_keys` for
    name + env), enriched with client-origin inference (parses UA →
    detects sandbox / docs / curl / python / node / browser) and a
    redacted-curl reconstruction the user can paste back to reproduce
    the call.

    Ownership check: the row's key_id must belong to the calling user.
    Otherwise return not_found (don't leak existence).
    """
    user = _get_session_user(request)
    user_id = int(user["id"])
    row = auth_db.get_request_log_row(user_id=user_id, row_id=int(request_id))
    if row is None:
        raise ApiError("not_found",
                        f"Request log entry {request_id} not found.")
    return build_request_detail_payload(user_id, row)


def build_request_detail_payload(user_id: int, row: dict) -> dict:
    """Assemble the FULL request-detail payload (UA inference, redacted curl +
    snippets, endpoint baselines, burst context, endpoint health, decoded
    payload + stage timings) from an already-fetched, owner-scoped request-log
    row. Extracted from ``usage_request_detail`` so the admin console can render
    the IDENTICAL drawer for ANY request (it resolves the owner first)."""
    # Infer client origin from UA + Referer
    ua = (row.get("ua") or "").lower()
    inferred = {"category": "unknown", "label": "Unknown client",
                 "icon": "i-flask", "language": None,
                 "via_page": None}
    if "sandbox" in ua or (row.get("referer") and "sandbox" in (row.get("referer") or "")):
        inferred = {"category": "sandbox", "label": "APIN Sandbox",
                     "icon": "i-flask", "language": "browser",
                     "via_page": "/account/api/sandbox"}
    elif "docs" in ua or (row.get("referer") and "docs" in (row.get("referer") or "")):
        inferred = {"category": "docs", "label": "Docs page (try-it-out)",
                     "icon": "i-book", "language": "browser",
                     "via_page": "/docs"}
    elif "node-fetch" in ua or "undici" in ua or "axios" in ua or "got/" in ua:
        # axios + got are popular Node HTTP clients; both set distinctive UAs
        client_label = "Node.js"
        if "axios" in ua: client_label = "Node.js (axios)"
        elif "got/" in ua: client_label = "Node.js (got)"
        elif "node-fetch" in ua: client_label = "Node.js (node-fetch)"
        elif "undici" in ua: client_label = "Node.js (undici)"
        inferred = {"category": "node", "label": client_label,
                     "icon": "i-route", "language": "javascript",
                     "via_page": None}
    elif "python-requests" in ua or "httpx" in ua or "aiohttp" in ua or "python-urllib" in ua:
        # 9.N.7.f · added python-urllib (stdlib default; was falling through
        # to "Unknown client" even though we clearly know it's Python).
        py_label = "Python"
        if "python-requests" in ua: py_label = "Python (requests)"
        elif "httpx" in ua:         py_label = "Python (httpx)"
        elif "aiohttp" in ua:       py_label = "Python (aiohttp)"
        elif "python-urllib" in ua: py_label = "Python (urllib)"
        inferred = {"category": "python", "label": py_label,
                     "icon": "i-flask", "language": "python",
                     "via_page": None}
    elif ua.startswith("curl/"):
        inferred = {"category": "curl", "label": "curl",
                     "icon": "i-funnel", "language": "shell",
                     "via_page": None}
    elif ua.startswith("postmanruntime/") or "postman" in ua:
        # Postman is one of the top-3 API testing clients — give it its own
        # branch so users see it identified correctly (was matching the
        # browser branch via the embedded "mozilla" string in some versions).
        inferred = {"category": "postman", "label": "Postman",
                     "icon": "i-flask", "language": "json",
                     "via_page": None}
    elif "insomnia" in ua:
        inferred = {"category": "insomnia", "label": "Insomnia",
                     "icon": "i-flask", "language": "json",
                     "via_page": None}
    elif "go-http-client" in ua:
        inferred = {"category": "go", "label": "Go (net/http)",
                     "icon": "i-route", "language": "go",
                     "via_page": None}
    elif "java/" in ua or "okhttp" in ua:
        inferred = {"category": "java", "label": "Java" + (" (OkHttp)" if "okhttp" in ua else ""),
                     "icon": "i-route", "language": "java",
                     "via_page": None}
    elif "ruby" in ua:
        inferred = {"category": "ruby", "label": "Ruby",
                     "icon": "i-route", "language": "ruby",
                     "via_page": None}
    elif "mozilla/" in ua or "chrome/" in ua or "firefox/" in ua or "safari/" in ua:
        # Generic browser fetch — distinguish docs vs sandbox via referer
        inferred = {"category": "browser", "label": "Browser",
                     "icon": "i-eye", "language": "javascript",
                     "via_page": None}

    # Build redacted curl
    method = row.get("method") or "GET"
    path = row.get("path") or "/"
    redacted_token = "apin_<your_token>"
    curl_lines = [
        f"curl -X {method} \\",
        f"  -H 'Authorization: Bearer {redacted_token}' \\",
    ]
    if method in ("POST", "PUT", "PATCH"):
        ct_hint = "multipart/form-data" if "predict" in path or "scan" in path \
                    else "application/json"
        if ct_hint == "multipart/form-data":
            curl_lines.append(f"  -F 'file=@/path/to/leaf.jpg' \\")
        else:
            curl_lines.append(f"  -H 'Content-Type: application/json' \\")
            curl_lines.append(f"  -d '{{}}' \\")
    curl_lines.append(f"  http://localhost:8888{path}")
    curl_str = "\n".join(curl_lines)

    # 9.N.7.f · Real endpoint latency baselines (not hardcoded).
    baselines = _compute_endpoint_baselines(user_id=user_id, path=path)

    # 9.N.8 · Burst context — which other requests landed near this one?
    burst = {}
    try:
        burst = auth_db.get_burst_context(
            user_id=user_id,
            key_id=row.get("key_public_id") or "",
            timestamp=row.get("timestamp") or "",
            row_id=int(row.get("id") or 0),
            window_seconds=1.0,
            max_neighbours=30,
        )
    except Exception:
        burst = {}

    # 9.N.8 · Endpoint health buckets for the mini-sparklines.
    health = {}
    try:
        health = auth_db.get_endpoint_health_buckets(
            user_id=user_id, path=path, bucket_count=20, total_seconds=3600,
        )
    except Exception:
        health = {}

    # 9.N.8 · Decode payload + stage_timings JSON (stored as text in DB).
    payload = _decode_payload_from_row(row)
    stage_timings = _decode_stage_timings_from_row(row)

    # 9.N.8 · JS-side syntax-highlighting hint maps to apin_syntax.js lexers.
    return {
        "row": row,
        "inferred": inferred,
        "curl": curl_str,
        "as_python": _python_snippet(method, path),
        "as_node": _node_snippet(method, path),
        "as_js":     _js_fetch_snippet(method, path),
        "baselines": baselines,
        "burst":     burst,
        "health":    health,
        "payload":   payload,
        "stage_timings": stage_timings,
    }


def _decode_payload_from_row(row: dict) -> dict:
    """Parse the JSON-stored payload columns into structured form for the
    drawer. Always returns a dict with the same shape; missing fields
    appear as nulls so the UI's 'not recorded' fallback can engage."""
    import json as _json
    def _safe_json(s):
        if not s: return None
        try: return _json.loads(s)
        except Exception: return None
    return {
        "headers_in":         _safe_json(row.get("headers_in_json")),
        "headers_out":        _safe_json(row.get("headers_out_json")),
        "body_in_preview":    row.get("body_in_preview"),
        "body_out_preview":   row.get("body_out_preview"),
        "body_in_ctype":      row.get("body_in_ctype"),
        "body_out_ctype":     row.get("body_out_ctype"),
        "body_in_truncated":  bool(row.get("body_in_truncated") or 0),
        "body_out_truncated": bool(row.get("body_out_truncated") or 0),
    }


def _decode_stage_timings_from_row(row: dict):
    import json as _json
    s = row.get("stage_timings_json")
    if not s: return None
    try: return _json.loads(s)
    except Exception: return None


def _js_fetch_snippet(method: str, path: str) -> str:
    """Browser-style fetch() snippet. Mirrors the Node template but without
    the import; suitable for paste into a browser console or React handler.
    """
    is_multipart = "predict" in path or "scan" in path
    if is_multipart and method in ("POST", "PUT", "PATCH"):
        return (
            "const fd = new FormData();\n"
            "fd.append('file', fileInput.files[0]);\n"
            "const r = await fetch('http://localhost:8888{p}', {{\n"
            "  method: '{m}',\n"
            "  headers: {{ Authorization: 'Bearer apin_<your_token>' }},\n"
            "  body: fd,\n"
            "}});\n"
            "console.log(r.status, await r.json());".format(m=method, p=path)
        )
    return (
        "const r = await fetch('http://localhost:8888{p}', {{\n"
        "  method: '{m}',\n"
        "  headers: {{ Authorization: 'Bearer apin_<your_token>' }},\n"
        "}});\n"
        "console.log(r.status, await r.json());".format(m=method, p=path)
    )


def _compute_endpoint_baselines(*, user_id: int, path: str) -> dict:
    """Compute p50/p95 latency for the given endpoint from the last 200
    successful requests by this user. Returns a small dict with the
    percentile values plus the sample size used. Empty/zero values if
    not enough data — the UI handles that gracefully."""
    try:
        rows = auth_db.get_recent_endpoint_latencies(
            user_id=user_id, path=path, limit=200,
        )
        lats = sorted([int(r) for r in rows if r is not None and r >= 0])
        n = len(lats)
        if n == 0:
            return {"p50_ms": None, "p95_ms": None, "sample_size": 0}
        def pct(p):
            idx = max(0, min(n - 1, int(p * (n - 1))))
            return lats[idx]
        return {"p50_ms": pct(0.50), "p95_ms": pct(0.95), "sample_size": n}
    except Exception:
        # Defensive: baselines are nice-to-have; never break the detail view.
        return {"p50_ms": None, "p95_ms": None, "sample_size": 0}


def _python_snippet(method: str, path: str) -> str:
    is_multipart = "predict" in path or "scan" in path
    if is_multipart and method in ("POST", "PUT", "PATCH"):
        return (
            "import requests\n"
            "r = requests.{m}(\n"
            "    'http://localhost:8888{p}',\n"
            "    headers={{'Authorization': 'Bearer apin_<your_token>'}},\n"
            "    files={{'file': open('leaf.jpg', 'rb')}},\n"
            ")\n"
            "print(r.status_code, r.json())".format(m=method.lower(), p=path)
        )
    return (
        "import requests\n"
        "r = requests.{m}(\n"
        "    'http://localhost:8888{p}',\n"
        "    headers={{'Authorization': 'Bearer apin_<your_token>'}},\n"
        ")\n"
        "print(r.status_code, r.json())".format(m=method.lower(), p=path)
    )


def _node_snippet(method: str, path: str) -> str:
    is_multipart = "predict" in path or "scan" in path
    if is_multipart and method in ("POST", "PUT", "PATCH"):
        return (
            "import fs from 'node:fs';\n"
            "const fd = new FormData();\n"
            "fd.append('file', new Blob([fs.readFileSync('leaf.jpg')]), 'leaf.jpg');\n"
            "const r = await fetch('http://localhost:8888{p}', {{\n"
            "  method: '{m}',\n"
            "  headers: {{ Authorization: 'Bearer apin_<your_token>' }},\n"
            "  body: fd,\n"
            "}});\n"
            "console.log(r.status, await r.json());".format(m=method, p=path)
        )
    return (
        "const r = await fetch('http://localhost:8888{p}', {{\n"
        "  method: '{m}',\n"
        "  headers: {{ Authorization: 'Bearer apin_<your_token>' }},\n"
        "}});\n"
        "console.log(r.status, await r.json());".format(m=method, p=path)
    )


# ─── 9.N.5 · New endpoints for the 6 new chart types ────────────────────────
# /heatmap-calendar     → day-of-week × hour-of-day activity grid (7×24 cells)
# /per-endpoint-detail  → mega-endpoint: per-endpoint count + error_rate +
#                         p50/p95/p99 + sparkline buckets. Feeds spark-grid /
#                         treemap / boxplot / quadrant.

@router.get("/heatmap-calendar")
@api_endpoint("/api/account/usage/heatmap-calendar")
async def usage_heatmap_calendar(request: Request,
                                  mode: str = Query("week"),
                                  key_id: Optional[str] = Query(None),
                                  env: Optional[str] = Query(None)):
    """Returns activity cells for the chosen calendar mode:
      - week   → 7×24 grid (day_of_week × hour_of_day), last 7 days
      - month  → 1×N grid (day-of-month, last 30 days)
      - year   → 1×12 grid (month-of-year, last 365 days)
      - years  → 1×N grid (year, last 5 years)
    Returns: { mode, cells: [{row, col, count, label?}, ...], rows, cols }
    """
    user = _get_session_user(request)
    user_id = int(user["id"])
    valid = {"week", "month", "year", "years"}
    if mode not in valid:
        raise ApiError("invalid_parameter",
                        f"unknown calendar mode {mode!r}; allowed: {sorted(valid)}")
    return auth_db.compute_usage_heatmap_calendar_multi(
        user_id=user_id, mode=mode, key_id=key_id, env=env)


# ─── 9.N.7 · SSE live stream — per-account real-time request feed ──────────
# GET /api/account/usage/stream emits one server-sent-event per request that
# UsageRecordingMiddleware records for any key belonging to the current user.
# Heartbeat every 15s keeps connections alive through proxies that drop idle
# streams. Authenticated via the session cookie.

@router.get("/stream")
async def usage_stream(request: Request):
    """Server-Sent Events feed. Streams every request_log row for this
    user's keys, in real time, as they're recorded by the middleware.

    Frame format (per W3C SSE):
        data: <JSON object>\\n\\n

    Plus periodic heartbeats `: heartbeat\\n\\n` (comment lines) every 15s.

    NOT wrapped in the standard envelope — this is a streaming endpoint
    where each frame is its own JSON object. Clients use EventSource.
    """
    from fastapi.responses import StreamingResponse
    from scripts.apin_v2 import usage_recorder
    import json as _json
    import asyncio as _asyncio

    user = _get_session_user(request)
    user_id = int(user["id"])
    bus = usage_recorder.get_stream_bus()
    q = bus.subscribe(user_id)

    async def gen():
        # First frame: stream-ready acknowledgement so the client knows
        # the connection is live and authenticated. Avoids the "is it
        # working?" ambiguity on a stream that may legitimately be quiet.
        try:
            yield (
                "event: ready\n"
                "data: " + _json.dumps({
                    "type": "ready",
                    "user_id": user_id,
                    "ts": _dt.now(_tz.utc).isoformat(),
                }) + "\n\n"
            )
            last_heartbeat = _asyncio.get_event_loop().time()
            while True:
                # Wait up to 15s for the next event; on timeout, emit a
                # heartbeat comment so intermediaries don't close the connection.
                try:
                    event = await _asyncio.wait_for(q.get(), timeout=15.0)
                    yield "data: " + _json.dumps(event) + "\n\n"
                except _asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                # Send a periodic ping comment regardless every 15s (in case
                # of bursty traffic that resets the wait_for loop).
                now = _asyncio.get_event_loop().time()
                if now - last_heartbeat > 15:
                    yield ": ping\n\n"
                    last_heartbeat = now
        except _asyncio.CancelledError:
            raise
        except Exception:
            # Stream died — let it close. Browser EventSource will auto-reconnect.
            pass
        finally:
            bus.unsubscribe(user_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",      # Disable nginx response buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/per-endpoint-detail")
@api_endpoint("/api/account/usage/per-endpoint-detail")
async def usage_per_endpoint_detail(request: Request,
                                     range: str = Query("24h"),
                                     limit: int = Query(20, ge=1, le=50),
                                     key_id: Optional[str] = Query(None),
                                     env: Optional[str] = Query(None),
                                     status: Optional[str] = Query(None),
                                     endpoint: Optional[str] = Query(None),
                                     spark_buckets: int = Query(20, ge=4, le=60)):
    """Returns per-endpoint stats: count, error_rate, p50, p95, p99 plus
    a small sparkline-shape (last N buckets of activity).
    Used by: spark-grid, treemap, boxplot, quadrant.

    9.N.6.f · Now accepts status+endpoint filters so a global drill cascades
    to these charts too (treemap/sparkgrid/boxplot/quadrant).
    """
    user = _get_session_user(request)
    user_id = int(user["id"])
    range_sec = _resolve_range(range)
    items = auth_db.compute_per_endpoint_detail(
        user_id=user_id, range_seconds=range_sec, limit=int(limit),
        spark_buckets=int(spark_buckets), key_id=key_id, env=env,
        status=status, endpoint=endpoint)
    return {"items": items, "range": range, "count": len(items)}
