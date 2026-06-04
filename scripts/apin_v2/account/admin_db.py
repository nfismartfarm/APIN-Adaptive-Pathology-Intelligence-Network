"""Admin database mirror — a read-only, Turso-style table browser (Phase D).

The admin "Database" section is a faithful mirror of the live SQLite/libSQL
schema: list every table with its row count, inspect a table's columns + types,
and page/search/sort the rows. Editing is intentionally NOT in this module — a
mutation surface is a separate, sudo-gated, audited, denylist-guarded pass.

Security posture (this is cross-table read of the WHOLE database, so it is the
most sensitive admin surface):

  • IDENTIFIER SAFETY. Table and column names are NEVER taken from client input
    verbatim. The table must be a member of the live ``sqlite_master`` set; the
    sort column must be a member of that table's ``PRAGMA table_info``. Anything
    else falls back to a safe default. Validated identifiers are double-quoted.
    Row VALUES are always passed as bound parameters.

  • SECRET MASKING. Columns whose name looks like a credential (password, token,
    secret, csrf, *_hash, otp code) are masked to "••••••" + a length hint and
    are excluded from search — the browser can confirm a hash EXISTS and its
    size, but never read it.

  • INTERNAL TABLES HIDDEN. The shim's ``_probe`` / ``_shim_probe`` scratch
    tables and SQLite internals are filtered out.

Everything is fail-open (returns an empty-but-valid shape) so the console never
500s on a malformed query.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from scripts.apin_v2 import auth_db

log = logging.getLogger("apin_v2.account.admin_db")

# Tables that are never exposed (shim scratch + anything SQLite-internal).
_HIDDEN_TABLES = {"_probe", "_shim_probe", "sqlite_sequence", "sqlite_stat1"}

# A column whose name matches this is a credential → masked + unsearchable.
_SECRET_RE = re.compile(
    r"(password|secret|csrf|token|code_hash|key_hash|api_secret|otp_code|recovery)",
    re.IGNORECASE,
)
# A column that is an opaque integrity hash → shown but truncated (not a secret).
_HASHISH_RE = re.compile(r"(_hash$|sha256|sha1|checksum)", re.IGNORECASE)

_MAX_CELL = 200          # display truncation for ordinary cell values
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200


def _live_tables(c) -> list:
    """Ordered list of user-visible table names from the live schema."""
    rows = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")
    return [dict(r)["name"] for r in rows if dict(r)["name"] not in _HIDDEN_TABLES]


def _columns(c, table: str) -> list:
    """[{name, type, pk, notnull, secret}] for a validated table name."""
    out = []
    for r in c.execute('PRAGMA table_info("%s")' % table):
        d = dict(r)
        name = d.get("name") or ""
        out.append({
            "name": name,
            "type": (d.get("type") or "").upper() or "—",
            "pk": int(d.get("pk") or 0) == 1,
            "notnull": int(d.get("notnull") or 0) == 1,
            "secret": bool(_SECRET_RE.search(name)),
        })
    return out


def _scalar(c, sql, args=()):
    row = c.execute(sql, args).fetchone()
    if not row:
        return 0
    return int(list(dict(row).values())[0] or 0)


def _mask_cell(col: dict, value):
    """Render a single cell for transport: secrets masked, hashes/longs trimmed."""
    if value is None:
        return None
    if col["secret"]:
        n = len(str(value))
        return "•••••• (%d)" % n
    s = str(value)
    if _HASHISH_RE.search(col["name"]) and len(s) > 18:
        return s[:10] + "…" + s[-4:]
    if len(s) > _MAX_CELL:
        return s[: _MAX_CELL - 1] + "…"
    return s


# ── Public API ──────────────────────────────────────────────────────────────
def db_list_tables() -> dict:
    """Every user-visible table with a live row count + column count.

    Performance: a naive version did 1 + N + N round-trips (a COUNT and a PRAGMA
    per table) — ~17s over the Turso HTTP shim for 43 tables. This collapses it
    to THREE queries: the table list, all columns via the ``pragma_table_info``
    table-valued function in one join, and all counts in one UNION-ALL.
    """
    out = {"tables": [], "total_rows": 0}
    try:
        with auth_db.get_conn() as c:
            names = _live_tables(c)
            if not names:
                return out

            # All columns for all tables in ONE query (TVF join).
            colmap: dict = {}
            try:
                for r in c.execute(
                    "SELECT m.name AS tbl, p.name AS col FROM sqlite_master m "
                    "JOIN pragma_table_info(m.name) p "
                    "WHERE m.type='table' AND m.name NOT LIKE 'sqlite_%'"):
                    d = dict(r)
                    colmap.setdefault(d["tbl"], []).append(d["col"])
            except Exception:
                colmap = {}

            # All row counts in ONE query (single-quoted name literals).
            counts: dict = {}
            try:
                union = " UNION ALL ".join(
                    "SELECT '%s' AS t, COUNT(*) AS c FROM \"%s\""
                    % (n.replace("'", "''"), n) for n in names)
                for r in c.execute(union):
                    d = dict(r)
                    counts[d["t"]] = int(d["c"] or 0)
            except Exception:
                counts = {}

            for n in names:
                cols = colmap.get(n)
                if cols is None:                       # TVF missed → fall back
                    cols = [col["name"] for col in _columns(c, n)]
                rows = counts.get(n)
                if rows is None:
                    try:
                        rows = _scalar(c, 'SELECT COUNT(*) FROM "%s"' % n)
                    except Exception:
                        rows = 0
                out["tables"].append({
                    "name": n, "rows": rows, "columns": len(cols),
                    "has_secrets": any(_SECRET_RE.search(col) for col in cols),
                })
                out["total_rows"] += rows
    except Exception as e:  # noqa: BLE001
        log.warning("db_list_tables failed: %s", e)
    return out


def db_table(name: str, *, search: Optional[str] = None, sort: Optional[str] = None,
             order: str = "asc", limit: int = _DEFAULT_LIMIT, offset: int = 0) -> Optional[dict]:
    """Schema + a masked, paginated page of rows for one validated table.

    Returns None if ``name`` is not a real, visible table.
    """
    limit = max(1, min(int(limit or _DEFAULT_LIMIT), _MAX_LIMIT))
    offset = max(0, int(offset or 0))
    order_sql = "DESC" if str(order).lower() == "desc" else "ASC"
    try:
        with auth_db.get_conn() as c:
            if name not in _live_tables(c):
                return None
            cols = _columns(c, name)
            col_names = [col["name"] for col in cols]
            # sort column: must be real; default to pk, else first column
            sort_col = sort if (sort in col_names) else None
            if not sort_col:
                pk = next((col["name"] for col in cols if col["pk"]), None)
                sort_col = pk or (col_names[0] if col_names else None)

            # search: across non-secret columns only
            where, args = "", []
            term = (search or "").strip()
            if term:
                searchable = [col["name"] for col in cols if not col["secret"]]
                if searchable:
                    clause = " OR ".join('CAST("%s" AS TEXT) LIKE ?' % cn for cn in searchable)
                    where = " WHERE (" + clause + ")"
                    args = ["%" + term + "%"] * len(searchable)

            total = _scalar(c, 'SELECT COUNT(*) FROM "%s"%s' % (name, where), tuple(args))

            order_clause = (' ORDER BY "%s" %s' % (sort_col, order_sql)) if sort_col else ""
            sql = 'SELECT * FROM "%s"%s%s LIMIT ? OFFSET ?' % (name, where, order_clause)
            rows = []
            for r in c.execute(sql, tuple(args) + (limit, offset)):
                d = dict(r)
                rows.append([_mask_cell(col, d.get(col["name"])) for col in cols])

            return {
                "name": name,
                "columns": cols,
                "rows": rows,
                "pagination": {"total": total, "limit": limit, "offset": offset,
                               "returned": len(rows)},
                "sort": sort_col, "order": order_sql.lower(),
                "search": term,
            }
    except Exception as e:  # noqa: BLE001
        log.warning("db_table(%s) failed: %s", name, e)
        return {"name": name, "columns": [], "rows": [],
                "pagination": {"total": 0, "limit": limit, "offset": offset, "returned": 0},
                "sort": None, "order": "asc", "search": search or "", "error": "query_failed"}
