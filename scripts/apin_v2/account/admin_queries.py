"""Org-wide admin queries — Phase B (Pulse + Users).

Read aggregations and the user-directory + role/session mutations behind the
admin gate. These are the ONLY cross-user queries in the codebase; every other
`compute_*` is hard-scoped to one `user_id`. Reads are fail-open (return an
empty-but-valid shape on error) so the dashboard never 500s. Mutations raise
``ValueError(<code>)`` which the route layer maps to a clean ApiError.

Security: every caller is already past ``require_admin`` (and, for mutations,
the SudoMiddleware central sudo+CSRF gate). Nothing here re-derives trust from
client input. The "last admin" guard reasons about EFFECTIVE admins
(``role='admin'`` ∪ env allowlist) so the system can never reach zero admins.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import admin_guard

log = logging.getLogger("apin_v2.account.admin_queries")

_VALID_ROLES = ("admin", "collector")


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _iso_hours_ago(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _scalar(c, sql, args=()):
    row = c.execute(sql, args).fetchone()
    if not row:
        return 0
    v = list(dict(row).values())[0]
    return int(v or 0)


def _scans_time_col(c) -> str:
    """scans uses processed_at; fall back to captured_at/created_at if absent."""
    try:
        cols = {dict(r)["name"] for r in c.execute("PRAGMA table_info(scans)")}
    except Exception:
        cols = set()
    for cand in ("processed_at", "captured_at", "created_at"):
        if cand in cols:
            return cand
    return "processed_at"


def _window_metrics(c, cutoff: Optional[str]) -> dict:
    """EVERY time-based metric for one window. ``cutoff=None`` = lifetime. A
    single overview call carries all four windows so the toggle re-skins the
    whole page instantly with no re-fetch."""
    stc = _scans_time_col(c)
    if cutoff:
        req = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log WHERE timestamp >= ?", (cutoff,))
        err = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log "
                         "WHERE timestamp >= ? AND status_code >= 400", (cutoff,))
        act = _scalar(c, "SELECT COUNT(DISTINCT k.user_id) FROM api_key_request_log r "
                         "JOIN api_keys k ON k.public_id = r.key_id WHERE r.timestamp >= ?", (cutoff,))
        keys_used = _scalar(c, "SELECT COUNT(DISTINCT key_id) FROM api_key_request_log "
                               "WHERE timestamp >= ?", (cutoff,))
        new_users = _scalar(c, "SELECT COUNT(*) FROM users WHERE created_at >= ?", (cutoff,))
        # inferences = real website inferences (predictions ∪ guest_predictions),
        # repointed (ADM-X) from the geo `scans` table.
        infer = (_scalar(c, "SELECT COUNT(*) FROM predictions WHERE deleted_at IS NULL AND created_at >= ?", (cutoff,))
                 + _scalar(c, "SELECT COUNT(*) FROM guest_predictions WHERE deleted_at IS NULL AND created_at >= ?", (cutoff,)))
    else:
        req = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log")
        err = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log WHERE status_code >= 400")
        act = _scalar(c, "SELECT COUNT(DISTINCT k.user_id) FROM api_key_request_log r "
                         "JOIN api_keys k ON k.public_id = r.key_id")
        keys_used = _scalar(c, "SELECT COUNT(DISTINCT key_id) FROM api_key_request_log")
        new_users = _scalar(c, "SELECT COUNT(*) FROM users")
        infer = (_scalar(c, "SELECT COUNT(*) FROM predictions WHERE deleted_at IS NULL")
                 + _scalar(c, "SELECT COUNT(*) FROM guest_predictions WHERE deleted_at IS NULL"))
    rate = round(100.0 * err / req, 2) if req > 0 else 0.0
    return {"requests": req, "errors": err, "error_rate": rate, "active_users": act,
            "keys_used": keys_used, "new_users": new_users, "inferences": infer}


# ── Time-bucketed series + per-metric detail (clickable tiles) ──────────────
def _window_cutoff(window: str) -> Optional[str]:
    return {"24h": _iso_hours_ago(24), "7d": _iso_days_ago(7),
            "30d": _iso_days_ago(30)}.get(window)   # 'all' → None


def _buckets(window: str):
    """(bucket_keys, substr_len, hourly?) for a window. 24h → 24 hourly buckets,
    else daily ('all' is bounded to the last 45 days so the chart stays legible).
    Keys are the leading substring of an ISO timestamp so a GROUP BY lines up."""
    now = datetime.now(timezone.utc)
    if window == "24h":
        base = now.replace(minute=0, second=0, microsecond=0)
        keys = [(base - timedelta(hours=i)).strftime("%Y-%m-%dT%H") for i in range(23, -1, -1)]
        return keys, 13, True
    n = 7 if window == "7d" else (30 if window == "30d" else 45)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    keys = [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n - 1, -1, -1)]
    return keys, 10, False


def _series_for(c, table: str, ts_col: str, window: str, extra: str = "") -> list:
    """Zero-filled count-per-bucket series over the window."""
    keys, sub, _hourly = _buckets(window)
    cutoff = keys[0]  # ISO-lexicographic: stored 'YYYY-..T..' >= 'YYYY-..' works
    counts: dict = {}
    try:
        sql = ('SELECT substr("%s",1,%d) AS b, COUNT(*) AS c FROM "%s" '
               'WHERE "%s" >= ? %s GROUP BY b' % (ts_col, sub, table, ts_col, extra))
        for r in c.execute(sql, (cutoff,)):
            d = dict(r)
            counts[d["b"]] = int(d["c"] or 0)
    except Exception:
        pass
    return [{"t": k, "c": counts.get(k, 0)} for k in keys]


def _breakdown(c, sql: str, args=()) -> list:
    """Rows of {label, value, [id]}. A query may SELECT an extra ``id`` column
    (e.g. a key public_id or user id) so the drill can open that entity directly
    instead of a filtered list."""
    out = []
    try:
        for r in c.execute(sql, args):
            d = dict(r)
            row = {"label": "" if d.get("label") is None else str(d["label"]),
                   "value": int(d.get("v") or 0)}
            if "id" in d and d.get("id") is not None:
                row["id"] = d["id"]
            out.append(row)
    except Exception:
        pass
    return out


def admin_metric_detail(metric: str, window: str = "all") -> dict:
    """Detailed drill for a single Overview tile: a time series + a top-N
    breakdown + a headline, all honouring the selected window."""
    metric = (metric or "").lower()
    window = window if window in ("24h", "7d", "30d", "all") else "all"
    out = {"metric": metric, "window": window, "series": [], "series_label": "",
           "breakdown": [], "breakdown_label": "", "headline": None, "unit": "", "hint": ""}
    cut = _window_cutoff(window)
    tw = "WHERE timestamp >= ?" if cut else ""
    targs = (cut,) if cut else ()
    try:
        with auth_db.get_conn() as c:
            stc = _scans_time_col(c)
            if metric in ("requests", "total_requests"):
                out["series"] = _series_for(c, "api_key_request_log", "timestamp", window)
                out["series_label"] = "requests / " + ("hour" if window == "24h" else "day")
                out["breakdown_label"] = "top endpoints"
                out["breakdown"] = _breakdown(
                    c, "SELECT path AS label, COUNT(*) AS v FROM api_key_request_log %s "
                       "GROUP BY path ORDER BY v DESC LIMIT 8" % tw, targs)
                out["headline"] = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log %s" % tw, targs)
                out["unit"] = "requests"
            elif metric in ("error_rate", "errors"):
                out["series"] = _series_for(c, "api_key_request_log", "timestamp", window, "AND status_code >= 400")
                out["series_label"] = "errors / " + ("hour" if window == "24h" else "day")
                ew = (tw + " AND status_code >= 400") if tw else "WHERE status_code >= 400"
                out["breakdown_label"] = "by status code"
                out["breakdown"] = _breakdown(
                    c, "SELECT CAST(status_code AS TEXT) AS label, COUNT(*) AS v "
                       "FROM api_key_request_log %s GROUP BY status_code ORDER BY v DESC LIMIT 8" % ew, targs)
                req = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log %s" % tw, targs)
                err = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log %s" % ew, targs)
                out["headline"] = round(100.0 * err / req, 2) if req else 0.0
                out["unit"] = "% error rate"
                out["hint"] = "%s errors of %s requests" % (err, req)
            elif metric in ("active_users", "users", "total_users", "new_users"):
                out["series"] = _series_for(c, "users", "created_at", window)
                out["series_label"] = "signups / " + ("hour" if window == "24h" else "day")
                out["breakdown_label"] = "top callers"
                bw = (tw.replace("timestamp", "r.timestamp")) if tw else ""
                out["breakdown"] = _breakdown(
                    c, "SELECT COALESCE(u.display_name, u.username, 'user '||u.id) AS label, "
                       "u.id AS id, COUNT(*) AS v FROM api_key_request_log r "
                       "JOIN api_keys k ON k.public_id = r.key_id "
                       "JOIN users u ON u.id = k.user_id %s "
                       "GROUP BY u.id ORDER BY v DESC LIMIT 8" % bw, targs)
                out["headline"] = _scalar(
                    c, "SELECT COUNT(DISTINCT k.user_id) FROM api_key_request_log r "
                       "JOIN api_keys k ON k.public_id = r.key_id %s" % bw, targs)
                out["unit"] = "active users"
            elif metric in ("keys", "active_keys", "keys_used"):
                out["series"] = _series_for(c, "api_key_request_log", "timestamp", window)
                out["series_label"] = "requests / " + ("hour" if window == "24h" else "day")
                out["breakdown_label"] = "busiest keys"
                out["breakdown"] = _breakdown(
                    c, "SELECT COALESCE(k.name, r.key_id) AS label, r.key_id AS id, COUNT(*) AS v "
                       "FROM api_key_request_log r LEFT JOIN api_keys k ON k.public_id = r.key_id %s "
                       "GROUP BY r.key_id ORDER BY v DESC LIMIT 8"
                       % (tw.replace("timestamp", "r.timestamp") if tw else ""), targs)
                out["headline"] = _scalar(c, "SELECT COUNT(DISTINCT key_id) FROM api_key_request_log %s" % tw, targs)
                out["unit"] = "keys used"
            elif metric in ("inferences", "scans"):
                # Repointed (ADM-X) from the geo `scans` table to the REAL website
                # inferences: predictions (logged-in) ∪ guest_predictions (guests),
                # each carrying its stored Grad-CAM + full monograph result. `scans`
                # stays for Geography/Observatory only.
                out["series_label"] = "inferences / " + ("hour" if window == "24h" else "day")
                out["breakdown_label"] = "top diagnoses"
                pw = "WHERE deleted_at IS NULL" + (" AND created_at >= ?" if cut else "")
                pargs = (cut,) if cut else ()
                # series — sum the two stores bucket-for-bucket (same window grid)
                s1 = _series_for(c, "predictions", "created_at", window, "AND deleted_at IS NULL")
                s2 = _series_for(c, "guest_predictions", "created_at", window, "AND deleted_at IS NULL")
                m2 = {x.get("t"): x.get("c", 0) for x in (s2 or [])}
                out["series"] = ([{"t": x.get("t"), "c": (x.get("c", 0) + m2.get(x.get("t"), 0))} for x in s1]
                                 if s1 else (s2 or []))
                out["breakdown"] = _breakdown(
                    c, "SELECT predicted_class AS label, COUNT(*) AS v FROM ("
                       "  SELECT predicted_class, created_at, deleted_at FROM predictions "
                       "  UNION ALL "
                       "  SELECT predicted_class, created_at, deleted_at FROM guest_predictions"
                       ") %s GROUP BY label ORDER BY v DESC LIMIT 8" % pw, pargs)
                out["headline"] = (
                    _scalar(c, "SELECT COUNT(*) FROM predictions %s" % pw, pargs)
                    + _scalar(c, "SELECT COUNT(*) FROM guest_predictions %s" % pw, pargs))
                out["unit"] = "inferences"
    except Exception as e:  # noqa: BLE001
        log.warning("admin_metric_detail(%s) failed: %s", metric, e)
    return out


# ── Effective-admin set (role ∪ allowlist) ────────────────────────────────
def _effective_admin_ids(c) -> set:
    """Set of user ids that are admins right now (role='admin' OR email in the
    env allowlist). Full scan of `users` — fine at this scale; the only place
    we need the allowlist∪role union materialised."""
    allow = admin_guard.admin_allowlist()
    ids = set()
    for r in c.execute("SELECT id, email, role FROM users"):
        d = dict(r)
        role = (d.get("role") or "").strip().lower()
        email = (d.get("email") or "").strip().lower()
        if role == "admin" or (email and email in allow):
            ids.add(int(d["id"]))
    return ids


# ── Overview / Pulse counts ────────────────────────────────────────────────
def admin_overview() -> dict:
    """Org-wide vital signs for the Pulse section. All fail-open."""
    out = {
        "total_users": 0, "admins": 0, "new_users_7d": 0, "new_users_30d": 0,
        "signups_series": [], "total_keys": 0, "active_keys": 0,
        "total_requests": 0, "requests_24h": 0, "errors_24h": 0,
        "error_rate_24h": 0.0, "active_users_24h": 0,
        "total_inferences": 0, "guests_total": 0, "conversion_pct": 0.0,
        # Per-window metrics for the Overview window toggle (everything follows it).
        "windows": {
            w: {"requests": 0, "errors": 0, "error_rate": 0.0, "active_users": 0,
                "keys_used": 0, "new_users": 0, "inferences": 0}
            for w in ("24h", "7d", "30d", "all")
        },
    }
    try:
        with auth_db.get_conn() as c:
            out["total_users"] = _scalar(c, "SELECT COUNT(*) FROM users")
            out["admins"] = len(_effective_admin_ids(c))
            out["new_users_7d"] = _scalar(
                c, "SELECT COUNT(*) FROM users WHERE created_at >= ?", (_iso_days_ago(7),))
            out["new_users_30d"] = _scalar(
                c, "SELECT COUNT(*) FROM users WHERE created_at >= ?", (_iso_days_ago(30),))

            # signups per day over the last 30 days (sparkline)
            series = []
            for r in c.execute(
                "SELECT substr(created_at,1,10) AS d, COUNT(*) AS n FROM users "
                "WHERE created_at >= ? GROUP BY d ORDER BY d", (_iso_days_ago(30),)):
                dd = dict(r); series.append({"date": dd["d"], "count": int(dd["n"] or 0)})
            out["signups_series"] = series

            out["total_keys"] = _scalar(
                c, "SELECT COUNT(*) FROM api_keys WHERE deleted_at IS NULL")
            out["active_keys"] = _scalar(
                c, "SELECT COUNT(*) FROM api_keys WHERE deleted_at IS NULL AND status = 'active'")

            out["total_requests"] = _scalar(c, "SELECT COUNT(*) FROM api_key_request_log")
            out["requests_24h"] = _scalar(
                c, "SELECT COUNT(*) FROM api_key_request_log WHERE timestamp >= ?",
                (_iso_hours_ago(24),))
            out["errors_24h"] = _scalar(
                c, "SELECT COUNT(*) FROM api_key_request_log "
                   "WHERE timestamp >= ? AND status_code >= 400", (_iso_hours_ago(24),))
            if out["requests_24h"] > 0:
                out["error_rate_24h"] = round(
                    100.0 * out["errors_24h"] / out["requests_24h"], 2)
            out["active_users_24h"] = _scalar(
                c, "SELECT COUNT(DISTINCT k.user_id) FROM api_key_request_log r "
                   "JOIN api_keys k ON k.public_id = r.key_id WHERE r.timestamp >= ?",
                (_iso_hours_ago(24),))

            # All four windows in one shot so the Pulse toggle is instant.
            out["windows"] = {
                "24h": _window_metrics(c, _iso_hours_ago(24)),
                "7d":  _window_metrics(c, _iso_days_ago(7)),
                "30d": _window_metrics(c, _iso_days_ago(30)),
                "all": _window_metrics(c, None),
            }

            # inferences = real website inferences (predictions ∪ guest_predictions)
            out["total_inferences"] = (
                _scalar(c, "SELECT COUNT(*) FROM predictions WHERE deleted_at IS NULL")
                + _scalar(c, "SELECT COUNT(*) FROM guest_predictions WHERE deleted_at IS NULL"))

            # rough guest→user conversion proxy
            try:
                out["guests_total"] = _scalar(c, "SELECT COUNT(*) FROM guest_sessions")
            except Exception:
                out["guests_total"] = 0
            denom = out["total_users"] + out["guests_total"]
            out["conversion_pct"] = round(100.0 * out["total_users"] / denom, 1) if denom else 0.0
    except Exception as e:
        log.warning("admin_overview failed: %s", e)
    return out


# ── User directory ─────────────────────────────────────────────────────────
_SORT_COLUMNS = {
    "created": "u.created_at", "last_seen": "u.last_seen_at",
    "email": "u.email", "username": "u.username", "role": "u.role",
    "keys": "key_count", "requests": "request_count",
}


def admin_list_users(*, search: Optional[str] = None, sort: str = "created",
                     order: str = "desc", limit: int = 25, offset: int = 0) -> dict:
    """Paginated, sortable, filterable user directory. Fail-open."""
    limit = max(1, min(int(limit or 25), 100))
    offset = max(0, int(offset or 0))
    sort_col = _SORT_COLUMNS.get(str(sort), "u.created_at")
    order_sql = "ASC" if str(order).lower() == "asc" else "DESC"

    where, args = "", []
    if search:
        where = ("WHERE (u.email LIKE ? OR u.username LIKE ? OR u.display_name LIKE ?)")
        like = f"%{search}%"; args = [like, like, like]

    allow = admin_guard.admin_allowlist()
    out = {"items": [], "total": 0, "limit": limit, "offset": offset, "summary": {}}
    try:
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        hb_cut = (_dt.now(_tz.utc) - _td(seconds=45)).isoformat()   # "live now" window
        spark_cut = (_dt.now(_tz.utc) - _td(days=14)).isoformat()
        with auth_db.get_conn() as c:
            out["total"] = _scalar(c, f"SELECT COUNT(*) FROM users u {where}", args)
            rows = c.execute(
                "SELECT u.id, u.username, u.display_name, u.email, u.role, "
                "       u.created_at, u.last_seen_at, "
                "  (SELECT COUNT(*) FROM api_keys k WHERE k.user_id = u.id "
                "     AND k.deleted_at IS NULL) AS key_count, "
                "  (SELECT COUNT(*) FROM api_key_request_log r WHERE r.key_id IN "
                "     (SELECT public_id FROM api_keys k2 WHERE k2.user_id = u.id)) AS request_count, "
                "  (SELECT COUNT(*) FROM predictions p WHERE p.user_id = u.id "
                "     AND p.deleted_at IS NULL) AS inference_count, "
                "  (SELECT COUNT(*) FROM guest_sessions g WHERE g.converted_to_user_id = u.id) AS guest_count, "
                "  (SELECT COUNT(*) FROM browser_sessions bs WHERE bs.user_id = u.id "
                "     AND bs.session_end_at IS NULL AND bs.last_heartbeat_at >= '%s') AS active_now "
                % hb_cut.replace("'", "")
                + f"FROM users u {where} ORDER BY {sort_col} {order_sql} LIMIT ? OFFSET ?",
                args + [limit, offset]).fetchall()
            items = [dict(r) for r in rows]
            page_ids = [int(d["id"]) for d in items]
            # one batched 14-day request sparkline for all page users
            spark = {}
            if page_ids:
                idlist = ",".join(str(i) for i in page_ids)
                for sr in c.execute(
                        "SELECT k.user_id uid, substr(r.timestamp,1,10) d, COUNT(*) c "
                        "FROM api_key_request_log r JOIN api_keys k ON k.public_id = r.key_id "
                        "WHERE k.user_id IN (%s) AND r.timestamp >= ? "
                        "GROUP BY k.user_id, d" % idlist, (spark_cut,)):
                    sd = dict(sr); spark.setdefault(int(sd["uid"]), {})[sd["d"]] = int(sd["c"] or 0)
            days = [(_dt.now(_tz.utc) - _td(days=13 - i)).strftime("%Y-%m-%d") for i in range(14)]
            for d in items:
                email = (d.get("email") or "").strip().lower()
                is_role_admin = (d.get("role") or "").strip().lower() == "admin"
                via_allow = bool(email and email in allow)
                an = int(d.get("active_now") or 0) > 0
                um = spark.get(int(d["id"]), {})
                out["items"].append({
                    "id": d["id"], "username": d["username"],
                    "display_name": d["display_name"], "email": d["email"],
                    "role": d["role"], "created_at": d["created_at"],
                    "last_seen_at": d["last_seen_at"],
                    "key_count": int(d["key_count"] or 0),
                    "request_count": int(d["request_count"] or 0),
                    "inference_count": int(d.get("inference_count") or 0),
                    "guest_count": int(d.get("guest_count") or 0),
                    "active_now": an,
                    "spark": [um.get(dd, 0) for dd in days],
                    "is_admin": is_role_admin or via_allow,
                    "admin_via": ("role" if is_role_admin else ("allowlist" if via_allow else None)),
                })
            # header summary (whole directory, not just this page)
            try:
                admins_total = len(_effective_admin_ids(c))
            except Exception:
                admins_total = _scalar(c, "SELECT COUNT(*) FROM users WHERE LOWER(role)='admin'")
            # directory-wide "active now": distinct users with a live (un-ended,
            # recently-heartbeat) browser session — counted across ALL users, not
            # just the current page, so the header reconciles regardless of paging.
            active_total = _scalar(
                c,
                "SELECT COUNT(DISTINCT bs.user_id) FROM browser_sessions bs "
                "WHERE bs.user_id IS NOT NULL AND bs.user_id > 0 "
                "  AND bs.session_end_at IS NULL AND bs.last_heartbeat_at >= '%s'"
                % hb_cut.replace("'", ""))
            out["summary"] = {
                "total_users": out["total"],
                "admins": admins_total,
                "active_now": active_total,
                "guests_converted": _scalar(c, "SELECT COUNT(*) FROM guest_sessions WHERE converted_to_user_id IS NOT NULL"),
            }
    except Exception as e:
        log.warning("admin_list_users failed: %s", e)
    return out


def admin_get_user(user_id: int) -> Optional[dict]:
    """Single-user dossier for the drawer. None if not found. Fail-open."""
    try:
        with auth_db.get_conn() as c:
            r = c.execute(
                "SELECT id, username, display_name, email, role, created_at, "
                "       last_seen_at, mobile_e164 FROM users WHERE id = ?",
                (int(user_id),)).fetchone()
            if not r:
                return None
            d = dict(r)
            email = (d.get("email") or "").strip().lower()
            allow = admin_guard.admin_allowlist()
            is_role_admin = (d.get("role") or "").strip().lower() == "admin"
            via_allow = bool(email and email in allow)
            d["is_admin"] = is_role_admin or via_allow
            d["admin_via"] = "role" if is_role_admin else ("allowlist" if via_allow else None)
            d["key_count"] = _scalar(
                c, "SELECT COUNT(*) FROM api_keys WHERE user_id = ? AND deleted_at IS NULL",
                (int(user_id),))
            d["request_count"] = _scalar(
                c, "SELECT COUNT(*) FROM api_key_request_log WHERE key_id IN "
                   "(SELECT public_id FROM api_keys WHERE user_id = ?)", (int(user_id),))
            d["active_sessions"] = _scalar(
                c, "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND revoked_at IS NULL "
                   "AND expires_at > ?", (int(user_id), auth_db._now_iso()))
            # never leak the hash; mobile is partially masked
            mob = d.get("mobile_e164") or ""
            d["mobile_masked"] = (mob[:3] + "•••" + mob[-2:]) if len(mob) > 5 else None
            d.pop("mobile_e164", None)
            return d
    except Exception as e:
        log.warning("admin_get_user failed: %s", e)
        return None


# ── Mutations (promote / revoke role · force-logout) ───────────────────────
def admin_set_user_role(*, target_user_id: int, new_role: str,
                        actor_user_id: int, ip: Optional[str] = None) -> dict:
    """Set a user's role. Raises ValueError('<code>') on:
       invalid_role · not_found · last_admin (would leave zero effective admins).
    Self-demotion IS allowed (the only block is reaching zero admins)."""
    new_role = (new_role or "").strip().lower()
    if new_role not in _VALID_ROLES:
        raise ValueError("invalid_role")

    with auth_db._write_lock, auth_db.get_conn() as c:
        row = c.execute(
            "SELECT id, email, role FROM users WHERE id = ?", (int(target_user_id),)).fetchone()
        if not row:
            raise ValueError("not_found")
        target = dict(row)
        from_role = (target.get("role") or "").strip().lower()
        email = (target.get("email") or "").strip().lower()
        allow = admin_guard.admin_allowlist()

        # Simulate the effective-admin set AFTER the change; block if it empties.
        future = _effective_admin_ids(c)
        if new_role == "admin":
            future.add(int(target_user_id))
        else:
            if email and email in allow:
                future.add(int(target_user_id))   # still admin via allowlist
            else:
                future.discard(int(target_user_id))
        if not future:
            raise ValueError("last_admin")

        c.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, int(target_user_id)))

    auth_db.audit("admin.role_changed", user_id=actor_user_id, ip_addr=ip,
                  detail={"target_user_id": int(target_user_id), "target_email": email,
                          "from": from_role, "to": new_role,
                          "self": int(actor_user_id) == int(target_user_id)})
    return admin_get_user(int(target_user_id))


def admin_revoke_user_sessions(*, target_user_id: int, actor_user_id: int,
                               ip: Optional[str] = None) -> int:
    """Force-logout: revoke every active session for the target. Returns count."""
    n = 0
    with auth_db._write_lock, auth_db.get_conn() as c:
        cur = c.execute(
            "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
            (auth_db._now_iso(), int(target_user_id)))
        n = cur.rowcount or 0
    auth_db.audit("admin.force_logout", user_id=actor_user_id, ip_addr=ip,
                  detail={"target_user_id": int(target_user_id), "sessions_revoked": n})
    return n


# ── Live activity feed (R5) ────────────────────────────────────────────────
# A normalised, cross-source event stream for the admin "what's happening now"
# container. Four lanes the operator cares about:
#   identity   — sign-ins, sign-ups, sign-outs, admin MFA, role changes
#   keys       — API-key lifecycle (mint / rotate / disable / delete / groups)
#   inference  — field scans (diagnosis, confidence, severity)
#   anomaly    — failed sign-ins, rejected admin codes, request errors (4xx/5xx)
# Sources are heterogeneous tables with independent id spaces, so each event id
# is namespaced (e.g. "al-1421"). The endpoint returns the most-recent N across
# all sources; the client dedupes by id and pops in anything it hasn't shown,
# which makes the feed "live" without a stateful cursor.

# How a raw audit_log `event` maps to (category, severity, human title).
_AUDIT_MAP = {
    "login_success":             ("identity",  "success", "Signed in"),
    "login_failed":              ("anomaly",   "danger",  "Failed sign-in"),
    "signup":                    ("identity",  "success", "New account"),
    "signup_success":            ("identity",  "success", "New account"),
    "logout":                    ("identity",  "info",    "Signed out"),
    "guest_started":             ("identity",  "info",    "Guest session started"),
    "admin.otp_requested":       ("identity",  "info",    "Admin code requested"),
    "admin.otp_verified":        ("identity",  "success", "Admin verified"),
    "admin.otp_failed":          ("anomaly",   "warn",    "Admin code rejected"),
    "admin.device_trusted":      ("identity",  "info",    "Device trusted (7 days)"),
    "admin.elevation_via_device":("identity",  "success", "Admin elevated · trusted device"),
    "admin.role_changed":        ("identity",  "warn",    "Role changed"),
    "admin.force_logout":        ("identity",  "warn",    "Force sign-out"),
    "sudo_failed":               ("anomaly",   "warn",    "Step-up auth failed"),
}

# Pure-mechanics audit events that would flood the operator feed — never shown.
_AUDIT_SKIP = {"csrf_rotated", "sudo_started", "sudo_used"}

# How an api_key_audit `action` (stored past-tense) maps to (severity, title).
_KEY_ACTION_MAP = {
    "created":       ("success", "API key minted"),
    "rotated":       ("info",    "API key rotated"),
    "disabled":      ("warn",    "API key disabled"),
    "enabled":       ("info",    "API key enabled"),
    "reactivated":   ("info",    "API key re-enabled"),
    "deleted":       ("danger",  "API key deleted"),
    "hard_deleted":  ("danger",  "API key deleted"),
    "patched":       ("info",    "API key updated"),
    "updated":       ("info",    "API key updated"),
    "group_created": ("success", "Key group created"),
    "group_updated": ("info",    "Key group updated"),
    "group_deleted": ("danger",  "Key group deleted"),
    "assigned":      ("info",    "Key assigned to group"),
    "unassigned":    ("info",    "Key removed from group"),
}


def _humanise(token: str) -> str:
    return (token or "").replace("admin.", "").replace("_", " ").replace(".", " ").strip().capitalize()


def _short(s, n=120):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def admin_events(limit: int = 40, category: Optional[str] = None) -> dict:
    """Newest events across all sources, normalised + actor-resolved. Fail-open."""
    limit = max(1, min(int(limit or 40), 120))
    per = limit  # pull `limit` from each source, then merge + trim
    events: list = []
    try:
        import json as _json
        with auth_db.get_conn() as c:
            # ── identity & anomalies — audit_log (pull extra to absorb skips) ─
            for r in c.execute(
                "SELECT id, user_id, event, detail, ip_addr, created_at "
                "FROM audit_log ORDER BY id DESC LIMIT ?", (min(per * 3, 200),)):
                d = dict(r)
                ev = d.get("event") or ""
                if ev in _AUDIT_SKIP:
                    continue
                cat, sev, title = _AUDIT_MAP.get(ev, ("identity", "info", _humanise(ev)))
                try:
                    det = _json.loads(d.get("detail") or "{}")
                except Exception:
                    det = {}
                sub = ""
                if ev == "admin.role_changed":
                    sub = "%s → %s · %s" % (det.get("from", "?"), det.get("to", "?"),
                                            det.get("target_email", "user"))
                elif ev == "login_failed":
                    sub = det.get("reason", "") or "bad credentials"
                elif ev == "admin.force_logout":
                    sub = "%s session(s) revoked" % det.get("sessions_revoked", "?")
                elif det:
                    sub = ", ".join("%s=%s" % (k, det[k]) for k in list(det)[:3])
                events.append({
                    "id": "al-%s" % d["id"], "ts": d.get("created_at") or "",
                    "category": cat, "severity": sev, "title": title,
                    "detail": _short(sub), "ip": d.get("ip_addr") or "",
                    "actor_id": d.get("user_id"), "raw": ev,
                })

            # ── API keys — api_key_audit ──────────────────────────────────
            for r in c.execute(
                "SELECT id, key_id, user_id, action, timestamp, key_name_at_time, details "
                "FROM api_key_audit ORDER BY id DESC LIMIT ?", (per,)):
                d = dict(r)
                act = (d.get("action") or "").lower()
                sev, title = _KEY_ACTION_MAP.get(act, ("info", _humanise(act) or "Key action"))
                kn = d.get("key_name_at_time") or (d.get("key_id") or "")
                events.append({
                    "id": "ka-%s" % d["id"], "ts": d.get("timestamp") or "",
                    "category": "keys", "severity": sev, "title": title,
                    "detail": _short(kn), "ip": "",
                    "actor_id": d.get("user_id"), "raw": act,
                })

            # ── Inference — scans ─────────────────────────────────────────
            try:
                scan_rows = c.execute(
                    "SELECT id, user_id, diagnosis, confidence, severity, tier, is_ood, processed_at "
                    "FROM scans WHERE deleted_at IS NULL ORDER BY id DESC LIMIT ?", (per,))
            except Exception:
                scan_rows = c.execute(
                    "SELECT id, user_id, diagnosis, confidence, severity, tier, is_ood, processed_at "
                    "FROM scans ORDER BY id DESC LIMIT ?", (per,))
            for r in scan_rows:
                d = dict(r)
                ood = int(d.get("is_ood") or 0) == 1
                diag = d.get("diagnosis") or "unknown"
                conf = d.get("confidence")
                conf_s = ("%d%%" % round(float(conf) * 100)) if conf is not None else ""
                sev = "warn" if ood else "success"
                title = "Out-of-distribution scan" if ood else "Leaf diagnosed"
                bits = [b for b in [diag.replace("_", " "), conf_s,
                                    (d.get("severity") or ""), (d.get("tier") or "")] if b]
                events.append({
                    "id": "sc-%s" % d["id"], "ts": d.get("processed_at") or "",
                    "category": "inference", "severity": sev, "title": title,
                    "detail": _short(" · ".join(bits)), "ip": "",
                    "actor_id": d.get("user_id"), "raw": diag,
                })

            # ── Anomalies — request errors (4xx/5xx) ──────────────────────
            for r in c.execute(
                "SELECT id, method, path, status_code, error_code, timestamp "
                "FROM api_key_request_log WHERE status_code >= 400 "
                "ORDER BY id DESC LIMIT ?", (per,)):
                d = dict(r)
                sc = int(d.get("status_code") or 0)
                sev = "danger" if sc >= 500 else "warn"
                title = "%s %s" % (sc, "server error" if sc >= 500 else "client error")
                detail = "%s %s%s" % (d.get("method") or "", _short(d.get("path") or "", 60),
                                      (" · " + d["error_code"]) if d.get("error_code") else "")
                events.append({
                    "id": "rq-%s" % d["id"], "ts": d.get("timestamp") or "",
                    "category": "anomaly", "severity": sev, "title": title,
                    "detail": _short(detail), "ip": "", "actor_id": None, "raw": str(sc),
                })

            # ── Resolve actor display names in one batch ──────────────────
            ids = sorted({e["actor_id"] for e in events if e.get("actor_id")})
            names = {}
            if ids:
                qmarks = ",".join("?" * len(ids))
                for r in c.execute(
                    "SELECT id, display_name, username FROM users WHERE id IN (%s)" % qmarks,
                    tuple(int(i) for i in ids)):
                    dd = dict(r)
                    names[int(dd["id"])] = dd.get("display_name") or dd.get("username") or ("user " + str(dd["id"]))
            for e in events:
                e["actor"] = names.get(int(e["actor_id"]), None) if e.get("actor_id") else None
                e.pop("actor_id", None)

        # newest first, optional lane filter, trim
        events.sort(key=lambda e: (e.get("ts") or ""), reverse=True)
        if category and category != "all":
            events = [e for e in events if e["category"] == category]
        events = events[:limit]
    except Exception as e:  # noqa: BLE001
        log.warning("admin_events failed: %s", e)
        events = []
    return {"events": events}
