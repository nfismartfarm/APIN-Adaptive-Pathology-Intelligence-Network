"""APIN v2 auth — SQLite-backed user accounts + sessions.

Day 3 deliverable. Provides:
    * Database schema bootstrap (idempotent)
    * argon2id password hashing
    * User CRUD helpers (create / lookup by handle)
    * Session lifecycle (create, validate, revoke)
    * Uniqueness checks used by /auth/check
    * Next-accession-number reader used by /auth/next-accession

Design notes:
    - Pure stdlib sqlite3 (no SQLModel/SQLAlchemy ORM overhead for ~3 tables).
    - SQLite is single-writer; fine for current scale. The schema avoids
      SQLite-specific SQL so it migrates to Postgres cleanly when needed.
    - Argon2id parameters follow OWASP 2024 guidance:
        time_cost=3, memory_cost=64 MiB, parallelism=4, hash_len=32.
    - Session tokens are 32 bytes from secrets.token_urlsafe;
      DB stores only sha256(token) so a DB leak can't impersonate users.
    - All timestamps stored as ISO 8601 UTC; rendered as IST at the UI.
    - Connection is created per-request via get_conn() with check_same_thread=False
      (FastAPI runs sync route handlers in a threadpool).
"""
from __future__ import annotations

import base64
import os
import sqlite3
import secrets
import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR  = PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "apin_v2.db"

# Argon2id — OWASP 2024 minimums
_ph = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,   # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)

# Session lifetime: 30 days
SESSION_LIFETIME = timedelta(days=30)

# Guest sessions: short-lived, anonymous, hard inference cap.
GUEST_SESSION_LIFETIME = timedelta(days=7)
GUEST_INFERENCE_LIMIT  = 3      # free inference checks before sign-up is required

# Connection lock — sqlite3 module is thread-safe with check_same_thread=False
# but we serialize writes to avoid "database is locked" under bursty load.
_write_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE BACKEND — local SQLite (dev) OR Turso/libSQL (production)
# ═══════════════════════════════════════════════════════════════════════════
#
# The app is built on SQLite. For a deployment where the host filesystem is
# ephemeral (e.g. a free Hugging Face Space), the database must live OFF the
# host on a durable service. Turso is hosted libSQL — a SQLite fork — so the
# entire schema, every BLOB column, and all the dashboard SQL migrate
# UNCHANGED.
#
# Backend selection is purely by environment:
#   • TURSO_DATABASE_URL unset            → stdlib sqlite3 against a local
#                                           file (dev + the whole test suite)
#   • TURSO_DATABASE_URL = libsql://…      → Turso (production)
#   • TURSO_DATABASE_URL = file:/path.db   → local libSQL — lets the EXACT
#                                           production code path be tested
#                                           without a Turso account
#
# `_ShimConn` is a minimal sqlite3.Connection-compatible adapter over a
# libsql ClientSync. It exposes exactly the surface auth_db.py uses, so the
# ~40 helper functions and all their SQL are untouched by this migration.

_TURSO_URL   = os.environ.get("TURSO_DATABASE_URL", "").strip()
_TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
_USE_TURSO   = bool(_TURSO_URL)

_LibsqlError = Exception   # replaced with the real class in Turso mode


def _turso_http_url(url: str) -> str:
    """Force the Hrana-over-HTTP transport for a remote Turso URL.

    libsql-client picks its transport from the URL scheme: libsql:// and
    wss:// use a WebSocket, https:// uses HTTP. The pure-Python client's
    WebSocket upgrade is rejected by current Turso edge servers
    (`WSServerHandshakeError: 400`), so rewrite ws-style schemes to https.
    Hrana-over-HTTP is also what Turso recommends for short-lived /
    serverless processes. `file:` URLs (local libSQL, used by the tests)
    pass through untouched."""
    if url.startswith("libsql://"):
        return "https://" + url[len("libsql://"):]
    if url.startswith("wss://"):
        return "https://" + url[len("wss://"):]
    if url.startswith("ws://"):
        return "http://" + url[len("ws://"):]
    return url


def _split_sql(script: str) -> list[str]:
    """Split a multi-statement SQL script into individual statements.

    libSQL's batch() runs one statement at a time. Chunks that contain
    only blank lines or `--` comments are dropped.

    Two corruption modes have been observed historically and are both
    defended against here:

    1. PDA-3 (Stage 1.6) — `;` inside a `--` line comment would split the
       script in the wrong place. The previous fix stripped `--` line
       comments BEFORE splitting on `;`.

    2. PDA-P1-R1-F01 (Stage 7 Phase 1) — `;` inside a `CREATE TRIGGER …
       BEGIN … END;` body would split the trigger into unparseable
       fragments. The libsql path then fed each fragment to `batch()`,
       which rejected the half-statement. CI didn't catch this because
       the stdlib sqlite3 path uses `executescript()` which handles
       trigger bodies natively. Stage-7's two new triggers
       (`api_key_audit_no_update`, `trg_user_insert_default_settings`)
       are the first triggers in the codebase, so this surfaced now.

    The fix is a single-pass state machine over the cleaned script that:
      - Tracks single-quoted string literals (so `;` inside `'…'` is safe)
      - Tracks `BEGIN…END` nesting (case-insensitive, word-boundary
        matched — so `BEGINNING` in a comment-stripped string does not
        increment the counter)
      - Only splits on `;` when both flags are zero (outside strings AND
        outside trigger bodies)

    Note: SQLite's lexer is more permissive than this (it accepts
    nested BEGIN inside compound statements, recognises `[ident]` and
    `` `ident` `` quoting), but our scripts use single-quoted strings
    only and a single BEGIN per trigger, so this simple machine is
    sufficient AND deterministic.
    """
    # Step 1: strip `--` line comments while respecting single-quoted strings.
    cleaned_lines: list[str] = []
    for ln in script.splitlines():
        in_str = False
        cut = -1
        i = 0
        while i < len(ln):
            ch = ln[i]
            if ch == "'":
                in_str = not in_str
            elif (not in_str
                  and ch == '-'
                  and i + 1 < len(ln)
                  and ln[i + 1] == '-'):
                cut = i
                break
            i += 1
        cleaned = ln[:cut].rstrip() if cut >= 0 else ln
        if cleaned.strip():
            cleaned_lines.append(cleaned)
    cleaned_script = "\n".join(cleaned_lines)

    # Step 2: state-machine split honouring strings AND BEGIN...END bodies.
    #
    # PDA-P1-R2-F03 (time bomb fix): SQLite allows `CASE expr WHEN … END`
    # expressions inside trigger bodies. The naive state machine treats
    # the trailing `END` of a CASE as closing the BEGIN block — prematurely
    # decrements depth to 0, then the next `;` splits the trigger
    # mid-body. The fix tracks CASE nesting independently of BEGIN nesting
    # and only decrements `begin_depth` when the matched `END` is NOT
    # closing a CASE. Today's Stage-7 triggers don't use CASE, but Phase 2+
    # webhook-retry triggers almost certainly will.
    out: list[str] = []
    current: list[str] = []
    in_str = False
    begin_depth = 0   # 0 = outside any BEGIN block; >0 = inside one or more
    case_depth = 0    # CASE/END pairs are independent of BEGIN/END
    i = 0
    n = len(cleaned_script)

    def _is_word_char(ch: str) -> bool:
        # Identifier characters that would invalidate a keyword word match.
        return ch.isalnum() or ch == '_'

    def _word_at(idx: int, word: str) -> bool:
        """Case-insensitive word match at position `idx` with word boundaries
        on both sides (prevents `BEGINNING` from matching `BEGIN`)."""
        wlen = len(word)
        if idx + wlen > n:
            return False
        if cleaned_script[idx:idx + wlen].upper() != word:
            return False
        # Left boundary
        if idx > 0 and _is_word_char(cleaned_script[idx - 1]):
            return False
        # Right boundary
        if idx + wlen < n and _is_word_char(cleaned_script[idx + wlen]):
            return False
        return True

    while i < n:
        ch = cleaned_script[i]
        if ch == "'":
            in_str = not in_str
            current.append(ch)
            i += 1
            continue
        if in_str:
            current.append(ch)
            i += 1
            continue
        # Outside a string — check for BEGIN / CASE / END keywords.
        if _word_at(i, "BEGIN"):
            begin_depth += 1
            current.append(cleaned_script[i:i + 5])
            i += 5
            continue
        if _word_at(i, "CASE"):
            case_depth += 1
            current.append(cleaned_script[i:i + 4])
            i += 4
            continue
        if _word_at(i, "END"):
            # `END` could close either a CASE (innermost wins by SQL
            # grammar) or a BEGIN. Prefer CASE: SQLite's CASE expression
            # MUST end with `END`, but a BEGIN block ends with `END;`
            # (terminator). So if any CASE is open, this END closes the
            # innermost one. If no CASE is open AND we're inside a BEGIN,
            # this END closes that. If neither, treat literally.
            if case_depth > 0:
                case_depth -= 1
            elif begin_depth > 0:
                begin_depth -= 1
            # else: malformed script — treat END literally and move on.
            current.append(cleaned_script[i:i + 3])
            i += 3
            continue
        if ch == ";" and begin_depth == 0 and case_depth == 0:
            # Statement terminator outside any trigger body or CASE expr.
            chunk = "".join(current).strip()
            if chunk:
                out.append(chunk)
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1

    # Tail (script that doesn't end with `;`)
    tail = "".join(current).strip()
    if tail:
        out.append(tail)
    return out


class _ShimRow:
    """sqlite3.Row-compatible row: supports row['col'], row[idx], dict(row),
    iteration over values, and len()."""
    __slots__ = ("_cols", "_vals")

    def __init__(self, cols, vals):
        self._cols = cols
        self._vals = vals

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._vals[key]
        return self._vals[self._cols.index(key)]

    def keys(self):
        return list(self._cols)

    def __iter__(self):
        return iter(self._vals)      # sqlite3.Row iterates VALUES

    def __len__(self):
        return len(self._vals)


class _ShimCursor:
    """Cursor-like wrapper around a libsql ResultSet. Supports fetchone /
    fetchall / iteration / lastrowid / rowcount."""

    def __init__(self, rs):
        cols = tuple(rs.columns)
        self._rows = [_ShimRow(cols, list(r)) for r in rs.rows]
        self._idx = 0
        self.lastrowid = rs.last_insert_rowid
        self.rowcount = rs.rows_affected if rs.rows_affected is not None else -1

    def fetchone(self):
        if self._idx < len(self._rows):
            r = self._rows[self._idx]
            self._idx += 1
            return r
        return None

    def fetchall(self):
        rest = self._rows[self._idx:]
        self._idx = len(self._rows)
        return rest

    def __iter__(self):
        rest = self._rows[self._idx:]
        self._idx = len(self._rows)
        return iter(rest)


class _ShimConn:
    """Minimal sqlite3.Connection-compatible adapter over a libsql
    ClientSync. Every call is serialised through `lock` because a single
    shared ClientSync is reused across FastAPI's request threadpool."""

    def __init__(self, client, lock):
        self._client = client
        self._lock = lock

    def execute(self, sql, params=()):
        # libsql wants a plain list; sqlite3.Binary is a memoryview, which
        # we normalise to bytes so BLOB inserts (image_bytes) work.
        args = [bytes(p) if isinstance(p, memoryview) else p
                for p in (params or ())]
        with self._lock:
            try:
                rs = self._client.execute(sql, args) if args \
                    else self._client.execute(sql)
            except _LibsqlError as e:
                msg = str(e)
                # Translate libSQL constraint errors so callers that
                # `except sqlite3.IntegrityError` keep working unchanged.
                if "constraint failed" in msg.lower() or "UNIQUE" in msg:
                    raise sqlite3.IntegrityError(msg) from e
                raise
            except KeyError as ke:
                # FX-Phase-8-libsql · libsql_client/http.py blindly does
                # `response["result"]` and crashes with KeyError when the
                # Turso server returns an error envelope ({"error": ...})
                # instead of a success envelope ({"result": ...}). The
                # library's own error path is broken on certain HTTP
                # status codes / transient failures.
                #
                # We can't fix the upstream library, but we CAN translate
                # the bare KeyError into a sqlite3.OperationalError so the
                # caller sees a useful message and FastAPI surfaces a real
                # 503-style envelope instead of a generic 500.
                #
                # Multi-attempt retry with exponential-ish backoff — Turso
                # transients can extend to 10-15 s in real outages. Backoffs:
                # 0.5, 1.0, 2.0, 3.5, 5.0, 7.0 s → total ≈19s of retry window.
                if str(ke) in ("'result'", "result"):
                    import time as _time
                    backoffs = (0.5, 1.0, 2.0, 3.5, 5.0, 7.0)
                    rs = None
                    last_err = ke
                    for attempt, delay in enumerate(backoffs):
                        _time.sleep(delay)
                        try:
                            rs = self._client.execute(sql, args) if args \
                                else self._client.execute(sql)
                            last_err = None
                            break
                        except Exception as e2:
                            last_err = e2
                    if last_err is not None:
                        raise sqlite3.OperationalError(
                            "libsql/Turso transient error: "
                            "missing 'result' in response, "
                            f"{len(backoffs)} retries all failed: "
                            f"{type(last_err).__name__}: {last_err}"
                        ) from last_err
                else:
                    raise
        return _ShimCursor(rs)

    def batch_read(self, stmts):
        """Pipeline multiple read statements in ONE HTTP round-trip and return
        a list of _ShimCursor (same surface as execute()). `stmts` is a list of
        (sql, params) tuples. Used by the per-key Overview to collapse ~7
        sequential Turso round-trips (≈5 s) into one (≈1 s). Read-only by
        convention — writes stay on execute() so they aren't bundled into a
        single atomic batch with the reads."""
        norm = []
        for s in stmts:
            if isinstance(s, (tuple, list)) and len(s) == 2:
                sql, params = s
                args = [bytes(p) if isinstance(p, memoryview) else p
                        for p in (params or ())]
                norm.append((sql, args))
            else:
                norm.append((str(s), []))
        with self._lock:
            results = self._client.batch(norm)
        return [_ShimCursor(rs) for rs in results]

    def executescript(self, script):
        stmts = _split_sql(script)
        with self._lock:
            # Multi-attempt retry for the SAME class of Turso transient
            # we handle in execute() — libsql_client can raise KeyError
            # 'result' on batch() during a flaky window.
            import time as _time
            last_err = None
            for attempt in range(4):
                try:
                    if attempt > 0:
                        _time.sleep(0.5 * attempt)
                    self._client.batch(stmts)
                    return
                except KeyError as ke:
                    if str(ke) not in ("'result'", "result"):
                        raise
                    last_err = ke
                except Exception as e:
                    # Don't swallow other classes of errors — re-raise.
                    raise
            raise sqlite3.OperationalError(
                "libsql/Turso transient on batch(): missing 'result' in "
                f"response across 4 attempts: {last_err}"
            ) from last_err

    def executemany(self, sql, seq_of_params):
        """sqlite3.Connection.executemany compatibility.

        libsql_client (and our HTTP shim) don't have a true bulk INSERT
        path — they accept a list of (sql, args) tuples via `batch()`.
        Build the per-statement list once and pass it through.

        Phase 9.A's usage_recorder relies on this to bulk-INSERT the
        request log on every flush.
        """
        # Materialize once — `seq_of_params` may be a generator.
        rows = list(seq_of_params)
        if not rows:
            return
        # libsql_client.Statement accepts (sql, args); the HTTP shim's
        # batch() accepts (sql, args) tuples directly. Build tuples.
        try:
            from libsql_client import Statement  # type: ignore[no-redef]
            stmts = [Statement(sql, list(p) if not isinstance(p, list) else p)
                     for p in rows]
        except Exception:
            # Fallback for the HTTP shim (no Statement class needed).
            stmts = [(sql, list(p) if not isinstance(p, list) else p)
                     for p in rows]
        with self._lock:
            import time as _time
            last_err = None
            for attempt in range(4):
                try:
                    if attempt > 0:
                        _time.sleep(0.5 * attempt)
                    self._client.batch(stmts)
                    return
                except KeyError as ke:
                    if str(ke) not in ("'result'", "result"):
                        raise
                    last_err = ke
                except Exception:
                    raise
            raise sqlite3.OperationalError(
                "libsql/Turso transient on executemany(): missing 'result' "
                f"in response across 4 attempts: {last_err}"
            ) from last_err

    def commit(self):
        # REV-R2-I02 + PDA-R2-F01 (§3.3): Direct multi-statement transactions
        # against the libsql shim are unsafe — libsql autocommits each execute()
        # so there is no transactional grouping if callers do
        # `c.execute(...); c.execute(...); c.commit()`. The new pattern is:
        #
        #     result = _txn(c, lambda c: (c.execute(...), c.execute(...))[-1])
        #
        # which buffers writes into _BatchProxy and atomically applies them via
        # client.batch() on success. See §3.3 of the API Console spec for the
        # full rationale.
        raise NotImplementedError(
            "_ShimConn.commit() is intentionally disabled. Use _txn(c, fn) "
            "for any multi-statement transaction — see scripts/apin_v2/auth_db.py "
            "module docstring or §3.3 of the API Console spec."
        )

    def close(self):
        pass   # the ClientSync is shared — never closed per request


# ── §3.3 — atomic transactions across SQLite WAL + libsql/Turso ─────────────
#
# Why this exists: libsql's ClientSync autocommits every execute(), and its
# batch() applies multiple statements atomically. Native sqlite3 supports
# BEGIN IMMEDIATE / COMMIT / ROLLBACK explicitly. _txn() bridges both — the
# caller writes `_txn(c, lambda c: ...)` and gets the same semantics either
# way. Module-level scope so any helper (auth_db functions or downstream
# Console handlers) can import and use it.

class AuditChainBatchError(RuntimeError):
    """Raised when more than one audit-table INSERT is buffered in a single
    `_txn()` on the libsql backend. See §3.3.0.1 for the read-isolation
    contract: buffered writes are not visible to subsequent reads inside the
    same batch, so a second audit insert would compute prev_hash against
    stale state and silently corrupt the hash chain.

    Resolution: chain `_mutate_with_audit` calls across SEPARATE `_txn()`s.
    """


# Audit tables — INSERT INTO any of these inside a _txn counts toward the
# audit-insert limit (§3.3.0.1). Both `audit_log` (existing, pre-Console)
# and `api_key_audit` (Phase 1 — API Console hash-chain audit table) are
# registered here in advance so the audit-insert detection is consistent
# the moment the Phase-1 schema lands.
_AUDIT_TABLES = frozenset({"audit_log", "api_key_audit"})


def _is_audit_insert(sql: str) -> bool:
    """Heuristic: 'INSERT INTO <table>' where <table> is in _AUDIT_TABLES.
    Robust to leading whitespace, mixed case, and quoted identifiers."""
    s = sql.lstrip().upper()
    if not s.startswith("INSERT"):
        return False
    # Strip 'INSERT' and find what comes after 'INTO'.
    after_insert = s[6:].lstrip()
    if after_insert.startswith("OR "):
        # INSERT OR REPLACE / INSERT OR IGNORE — skip the modifier.
        after_insert = after_insert.split(" ", 2)[-1].lstrip()
    if not after_insert.startswith("INTO"):
        return False
    rest = after_insert[4:].lstrip()
    # First word after INTO is the table name; strip quotes/backticks.
    tbl_token = rest.split(None, 1)[0] if rest else ""
    tbl = tbl_token.strip('"`[](),').upper()
    return tbl in {t.upper() for t in _AUDIT_TABLES}


class _SyntheticWriteCursor:
    """Returned by `_BatchProxy.execute()` for buffered WRITE statements.

    Callers inside `_txn(c, op)` MUST NOT branch on `rowcount` or `lastrowid`
    of a WRITE issued inside the batch — those values are not known until the
    batch flushes on commit. If the caller needs the result of a write, the
    correct pattern is a follow-up SELECT (which round-trips and sees the
    in-progress state on SQLite WAL; on libsql, the SELECT in question
    typically happens in a SEPARATE `_txn` post-commit). See PDA-R2-F09.
    """
    rowcount = -1
    lastrowid = None

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __iter__(self):
        return iter(())


class _BatchProxy:
    """Read-through, write-buffered cursor wrapper for libsql.

    READ statements (SELECT/PRAGMA/EXPLAIN/WITH) round-trip immediately
    because the caller needs their result.

    WRITE statements (INSERT/UPDATE/DELETE/REPLACE) are buffered as libsql
    Statement objects and flushed atomically via `client.batch()` on
    `_BatchProxy.flush()`. This gives multi-statement atomicity on a backend
    that has no explicit BEGIN/COMMIT/ROLLBACK.

    Audit-insert tracking (§3.3.0.1): at most ONE `INSERT INTO {audit_table}`
    per `_txn()`. The proxy raises `AuditChainBatchError` on `flush()` if
    `self._audit_insert_count > 1`. This is the read-isolation defence — the
    second audit insert would build its prev_hash from a STALE read because
    the first insert is still buffered.
    """
    _READ_VERBS = ("SELECT", "PRAGMA", "EXPLAIN", "WITH")

    def __init__(self, conn):
        self._conn = conn
        self._client = conn._client
        self._buffered: list = []   # libsql Statement objects
        self._aborted = False
        self._audit_insert_count = 0   # §3.3.0.1 enforcement

    def execute(self, sql, params=()):
        if self._aborted:
            raise RuntimeError("_BatchProxy used after rollback")
        verb = sql.lstrip().split(" ", 1)[0].upper() if sql else ""
        if verb in self._READ_VERBS:
            # Round-trip immediately — caller wants the result.
            return self._conn.execute(sql, params)
        # WRITE — buffer for atomic flush.
        if _is_audit_insert(sql):
            self._audit_insert_count += 1
        args = [bytes(p) if isinstance(p, memoryview) else p
                for p in (params or ())]
        try:
            from libsql_client import Statement
        except ImportError as e:
            raise RuntimeError(
                "_BatchProxy used outside libsql backend — _txn() should "
                "have dispatched to the sqlite3 path."
            ) from e
        self._buffered.append(Statement(sql, args))
        return _SyntheticWriteCursor()

    def flush(self):
        """Apply buffered writes atomically. Enforces audit-insert limit."""
        # §3.3.0.1: at most one audit insert per _txn on libsql.
        if self._audit_insert_count > 1:
            raise AuditChainBatchError(
                f"multiple audit inserts ({self._audit_insert_count}) "
                f"in single _txn forbidden on libsql backend — "
                f"chain _mutate_with_audit across separate _txn()s "
                f"(see §3.3.0.1)"
            )
        if self._buffered:
            with self._conn._lock:
                self._client.batch(self._buffered)
        self._buffered = []

    def abort(self):
        self._aborted = True
        self._buffered = []


def _txn(conn, op):
    """Atomic transaction across SQLite (WAL) and libsql/Turso.

    Args:
      conn: connection-like — either `_ShimConn` (libsql) or `sqlite3.Connection`.
      op:   callable `(cursor) -> result`. Receives a cursor-like object that
            shares the same `.execute()` surface as `conn`.

    Returns: whatever `op` returns.

    Semantics:
      - libsql:  writes buffered into `_BatchProxy`; on `op` success, all
                 writes flushed atomically via `client.batch()`. On exception,
                 buffered writes are dropped (libsql has nothing to roll back
                 because nothing was applied yet).
      - sqlite3: BEGIN IMMEDIATE / COMMIT / ROLLBACK explicitly via cursor.

    Constraint: `op` MUST NOT branch on `rowcount` of an INSERT/UPDATE/DELETE
    issued INSIDE the batch — on libsql, that value is not known until flush.
    Use a follow-up SELECT or a post-commit re-read instead. See PDA-R2-F09.

    Constraint: at most ONE audit-table INSERT per `_txn` on libsql. Enforced
    by `_BatchProxy.flush()` raising `AuditChainBatchError`.
    """
    if isinstance(conn, _ShimConn):
        proxy = _BatchProxy(conn)
        try:
            result = op(proxy)
            proxy.flush()
            return result
        except Exception:
            proxy.abort()
            raise
    # Native sqlite3.Connection path
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        result = op(conn)
        conn.commit()
        return result
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def _smoke_test_txn(conn) -> None:
    """Optional boot-time check. Call from setup scripts when APIN_ENV !=
    'production' to verify _txn round-trips correctly on the configured
    backend. Fails loudly on mismatch. See §3.3 last paragraph."""
    result = _txn(conn, lambda c: c.execute("SELECT 1").fetchone())
    # _ShimRow / sqlite3.Row both index-accessible; native sqlite3 returns
    # a plain tuple here, libsql returns _ShimRow. Normalise via int().
    val = int(result[0]) if result is not None else None
    if val != 1:
        raise RuntimeError(f"_txn smoke test failed: got {result!r}")


# The shim has its OWN lock — distinct from _write_lock. Write helpers do
# `with _write_lock, get_conn() as c: c.execute(...)`, so if the shim
# reused _write_lock it would try to acquire a non-reentrant lock twice
# on the same thread → deadlock. Lock order is always write_lock → shim
# lock (reads take only the shim lock), so there is no inversion.
_libsql_lock = threading.Lock()

# Build the Turso client + shared shim connection (once, at import).
#
# APIN_HTTP_TURSO=1 → use our in-house HTTP shim (scripts/apin_v2/_libsql_http_shim.py)
# instead of libsql_client. We added this because libsql_client 0.3.1's
# `_send` blows up with `KeyError: 'result'` on certain valid Turso
# responses (sustained issue, not transient). The HTTP shim talks to
# Turso's v2/pipeline endpoint directly via requests + has built-in
# retries, and works reliably on the same DB libsql_client chokes on.
_libsql_client = None
_libsql_conn = None
if _USE_TURSO:
    _USE_HTTP_SHIM = (os.environ.get("APIN_HTTP_TURSO") or "").strip() == "1"
    if _USE_HTTP_SHIM:
        from scripts.apin_v2 import _libsql_http_shim as _libsql_client  # type: ignore[no-redef]
        from scripts.apin_v2._libsql_http_shim import LibsqlError as _LibsqlError  # noqa: F811
        import logging as _logging
        _logging.getLogger("apin_v2.auth_db").warning(
            "Turso: using HTTP shim (APIN_HTTP_TURSO=1) "
            "— bypasses libsql_client 0.3.1 KeyError bug.")
    else:
        import libsql_client
        from libsql_client import LibsqlError as _LibsqlError  # noqa: F811
    if _TURSO_URL.startswith("file:"):
        # Local libSQL file — used by _turso_shim_test.py to exercise the
        # exact production code path without a Turso account. No auth, no
        # scheme rewrite. (The HTTP shim does not support file: URLs;
        # APIN_HTTP_TURSO=1 + file: is a misconfiguration — fall back.)
        if _USE_HTTP_SHIM:
            import libsql_client as _real
            _libsql_client_obj = _real.create_client_sync(_TURSO_URL)
        else:
            _libsql_client_obj = _libsql_client.create_client_sync(_TURSO_URL)
    else:
        # Remote Turso — force the HTTP transport.
        _conn_url = _turso_http_url(_TURSO_URL)
        _libsql_client_obj = _libsql_client.create_client_sync(
            _conn_url, auth_token=(_TURSO_TOKEN or None))
    _libsql_conn = _ShimConn(_libsql_client_obj, _libsql_lock)


# ─── Schema bootstrap ─────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL COLLATE NOCASE,
    display_name    TEXT NOT NULL COLLATE NOCASE,
    email           TEXT NOT NULL COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    mobile_e164     TEXT NOT NULL,
    pressed_leaf_seed INTEGER NOT NULL,
    role            TEXT NOT NULL DEFAULT 'collector',
    preferred_language TEXT NOT NULL DEFAULT 'en',
    profile         TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    last_seen_at    TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username COLLATE NOCASE);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_display_name ON users(display_name COLLATE NOCASE);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      TEXT NOT NULL UNIQUE,
    user_agent      TEXT,
    ip_addr         TEXT,
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    revoked_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash);

-- Guest sessions — anonymous "try it out" access with a hard inference cap.
-- A guest is NOT a user: no account, no dashboard, no persisted predictions.
-- inference_count is incremented server-side on each /predict/full so the
-- quota cannot be bypassed by clearing localStorage.  token_hash stores
-- sha256(raw_token) — the raw token lives only in the apin_v2_guest cookie.
CREATE TABLE IF NOT EXISTS guest_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash      TEXT NOT NULL UNIQUE,
    inference_count INTEGER NOT NULL DEFAULT 0,
    user_agent      TEXT,
    ip_addr         TEXT,
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_guest_token_hash ON guest_sessions(token_hash);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    event           TEXT NOT NULL,
    detail          TEXT NOT NULL DEFAULT '{}',
    ip_addr         TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    crop            TEXT,
    predicted_class TEXT,
    confidence      REAL,
    tier            TEXT,
    image_sha256    TEXT,
    -- Phase 3.5 — real image + heatmap storage.
    --   image_bytes : raw JPEG/PNG bytes exactly as uploaded by the user.
    --   heatmap_b64 : Grad-CAM PNG base64 (decoded server-side when served).
    -- Both NULL for rows recorded before this migration.  See _migrate().
    image_bytes     BLOB,
    heatmap_b64     TEXT,
    response_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_date ON predictions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop);
CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class);

-- ── Phase 2: Margin Notes ─────────────────────────────────────────────────
-- Each row is a small free-form annotation the user attaches either to a
-- specific date (attached_date, YYYY-MM-DD) or to a specific prediction
-- (attached_prediction_id) — exactly one of those two should be set.
-- Mood is optional (0–3 maps to four moods displayed as filled circles).
CREATE TABLE IF NOT EXISTS margin_notes (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text                       TEXT NOT NULL,
    attached_date              TEXT,            -- YYYY-MM-DD or NULL
    attached_prediction_id     INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    mood                       INTEGER,         -- 0..3 or NULL
    created_at                 TEXT NOT NULL,
    updated_at                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_notes_user_date  ON margin_notes(user_id, attached_date);
CREATE INDEX IF NOT EXISTS idx_notes_user_pred  ON margin_notes(user_id, attached_prediction_id);
CREATE INDEX IF NOT EXISTS idx_notes_user_created ON margin_notes(user_id, created_at DESC);

-- ── Phase 3: Treatment Log ────────────────────────────────────────────────
-- Each row records a treatment the user applied — e.g. "neem oil on okra
-- YVMV in Plot A on May 14". Optional `target_prediction_id` ties it to a
-- specific prediction so the dashboard can show "🌿 treated 3 days ago" on
-- future predictions of the same crop.
CREATE TABLE IF NOT EXISTS treatment_log (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    treatment                TEXT NOT NULL,             -- "neem oil", "copper hydroxide", "removed plant", etc.
    crop                     TEXT,                       -- okra | brassica | tomato | chilli
    disease                  TEXT,                       -- canonical class string (matches DISEASE_TAXONOMY keys)
    plot                     TEXT,                       -- user free-text plot label (e.g. "Plot A", "south field")
    notes                    TEXT,                       -- free-text observations
    target_prediction_id     INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    applied_date             TEXT NOT NULL,              -- YYYY-MM-DD
    created_at               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_treat_user_date  ON treatment_log(user_id, applied_date DESC);
CREATE INDEX IF NOT EXISTS idx_treat_user_pred  ON treatment_log(user_id, target_prediction_id);
CREATE INDEX IF NOT EXISTS idx_treat_user_dis   ON treatment_log(user_id, disease);

-- ── Phase 3: Public Share Tokens ──────────────────────────────────────────
-- Each row is a short-lived public share link for ONE prediction.
-- DB stores SHA-256 hash of the token (raw token only ever returned at
-- creation time). Revocation is soft (revoked_at) so we can audit.
CREATE TABLE IF NOT EXISTS share_tokens (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    prediction_id            INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    token_hash               TEXT NOT NULL UNIQUE,
    label                    TEXT,                       -- optional user label "for Krishi officer"
    created_at               TEXT NOT NULL,
    expires_at               TEXT,                       -- nullable = no expiry
    revoked_at               TEXT,
    last_viewed_at           TEXT,
    view_count               INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_shares_user      ON share_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_shares_token     ON share_tokens(token_hash);

-- ── Weekly PDF reports ────────────────────────────────────────────────────
-- One row per generated weekly report. The rendered PDF is stored as a BLOB
-- so it is reused instead of re-rendered and survives a Space restart.
-- Delete is soft (deleted_at) so the undo toast can restore it. The partial
-- unique index keeps at most one ACTIVE report per user per week while
-- allowing a deleted week to be generated again.
CREATE TABLE IF NOT EXISTS reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    week_start    TEXT NOT NULL,
    week_end      TEXT NOT NULL,
    pdf_bytes     BLOB NOT NULL,
    summary_json  TEXT,
    generated_at  TEXT NOT NULL,
    deleted_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_user ON reports(user_id, week_start DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_active ON reports(user_id, week_start) WHERE deleted_at IS NULL;

-- Status monitoring heartbeats
-- A background task writes one row here every 60 seconds. Each row is a
-- snapshot of every component check at that moment. The raw table is kept
-- small (pruned to roughly the last 48 hours) and powers the live "recent
-- pulse" strip plus the health-check latency KPI. The longer 90-day history
-- the status page draws its uptime bars from lives in status_days below,
-- which is a compact daily rollup.
--   overall     : operational | degraded | down  (worst component that tick)
--   components  : JSON object, component-key to up|degraded|down
--   response_ms : wall time the component sweep itself took, in ms
CREATE TABLE IF NOT EXISTS heartbeats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at  TEXT NOT NULL,
    overall      TEXT NOT NULL,
    components   TEXT NOT NULL DEFAULT '{}',
    response_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_heartbeats_time ON heartbeats(recorded_at DESC);

-- Daily status rollup
-- One row per calendar day (UTC). The heartbeat task upserts today's row on
-- every tick, accumulating per-component up/degraded/down counters. The
-- status page reads at most 90 of these rows to draw its uptime bars, so the
-- 90-day view costs one tiny query regardless of how many heartbeats exist.
--   components : JSON object, component-key to {up,deg,down} counters
--   overall    : worst overall status observed across the whole day
CREATE TABLE IF NOT EXISTS status_days (
    day          TEXT PRIMARY KEY,
    overall      TEXT NOT NULL,
    components   TEXT NOT NULL DEFAULT '{}',
    samples      INTEGER NOT NULL DEFAULT 0,
    op_count     INTEGER NOT NULL DEFAULT 0,
    resp_ms_sum  INTEGER NOT NULL DEFAULT 0,
    resp_ms_n    INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);

-- ═════════════════════════════════════════════════════════════════════════
-- 9.N.8i · External availability probes.
--
-- Records the result of every external monitor probe (GitHub Action +
-- Cloudflare Worker, every 2 min / 1 min respectively).  These probes
-- hit /api/probe/external — a dedicated endpoint that is *excluded* from
-- api_key_request_log so monitoring traffic never inflates user usage
-- stats.  The probe runner writes results DIRECTLY to Turso, so probe
-- failures (including HF Space hibernation) are still recorded — the
-- Space being asleep doesn't break the recording pipeline.
--
-- This is the ground-truth source for the /status page's "External
-- Availability" section.  Heartbeats above (status_days) are the
-- inside-the-container view; this table is the from-outside view.
--
-- Industry-grade column design — every probe captures:
--   · Identity   : probe_id (UUIDv4 for dedup on retries) + schema_version
--   · Timing     : 4 timestamps (issued, server-recv, server-send, recorded)
--                  + 5-stage breakdown (dns, tcp, tls, ttfb, download)
--   · Outcome    : success flag + overall + error_class enum + error_detail
--   · HTTP       : status code, body size, sha256 prefix, target_url
--   · Components : full JSON map + denormalised up/total counts + failures
--   · Resources  : memory, cpu, gpu vram, disk, fds, event-loop lag
--   · App stats  : last-5min request count, error rate, p50, p95
--   · Build      : version, git sha, deployed_at  (drift detection across deploys)
--   · Provenance : probe_source, runner, region, version, user-agent, depth
--   · Gates      : 7 boolean gates the probe checks (HTTP 2xx, JSON parse,
--                  schema match, overall ok, all components up, latency SLO,
--                  no resource alerts) — for fine-grained failure analysis
-- ═════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS external_probes (
    -- ── Identity ────────────────────────────────────────────────────
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    probe_id                 TEXT    NOT NULL,
    schema_version           TEXT    NOT NULL DEFAULT 'probe.v1',

    -- ── Timing (ISO 8601, UTC, microsecond precision) ───────────────
    issued_at_utc            TEXT    NOT NULL,
    server_recv_at_utc       TEXT,
    server_send_at_utc       TEXT,
    recorded_at_utc          TEXT    NOT NULL,

    -- ── Outcome ─────────────────────────────────────────────────────
    success                  INTEGER NOT NULL,
    overall                  TEXT    NOT NULL,
    error_class              TEXT,
    error_detail             TEXT,
    sla_breach               INTEGER NOT NULL DEFAULT 0,

    -- ── HTTP level ──────────────────────────────────────────────────
    http_status              INTEGER,
    target_url               TEXT    NOT NULL,
    response_bytes           INTEGER,
    response_checksum_sha256 TEXT,

    -- ── Timing breakdown (probe-perspective, ms) ────────────────────
    dns_ms                   INTEGER,
    tcp_connect_ms           INTEGER,
    tls_handshake_ms         INTEGER,
    ttfb_ms                  INTEGER,
    download_ms              INTEGER,
    total_ms                 INTEGER NOT NULL,

    -- ── Component health (parsed from response body) ────────────────
    components_json          TEXT,
    components_up_count      INTEGER,
    components_total_count   INTEGER,
    component_failures_json  TEXT,

    -- ── Server-side resource snapshot ───────────────────────────────
    process_uptime_s         INTEGER,
    memory_rss_mb            INTEGER,
    memory_pct               REAL,
    cpu_pct_1m               REAL,
    gpu_vram_used_mb         INTEGER,
    gpu_vram_total_mb        INTEGER,
    disk_free_gb             REAL,
    open_fds                 INTEGER,
    event_loop_lag_ms        INTEGER,

    -- ── Rolling 5-min app stats (from api_key_request_log) ──────────
    request_count_5m         INTEGER,
    error_count_5m           INTEGER,
    error_rate_5m_pct        REAL,
    p50_latency_5m_ms        INTEGER,
    p95_latency_5m_ms        INTEGER,

    -- ── Build identity ──────────────────────────────────────────────
    build_version            TEXT,
    build_git_sha            TEXT,
    build_deployed_at_utc    TEXT,

    -- ── Probe provenance ────────────────────────────────────────────
    probe_source             TEXT    NOT NULL,
    probe_runner             TEXT,
    probe_region             TEXT,
    probe_version            TEXT,
    probe_depth              TEXT    NOT NULL DEFAULT 'shallow',
    probe_user_agent         TEXT,

    -- ── Validation gates ────────────────────────────────────────────
    gate_http_2xx            INTEGER,
    gate_json_parseable      INTEGER,
    gate_schema_match        INTEGER,
    gate_overall_ok          INTEGER,
    gate_all_components_up   INTEGER,
    gate_latency_under_slo   INTEGER,
    gate_no_resource_alerts  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_external_probes_issued_at
    ON external_probes(issued_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_external_probes_success_issued
    ON external_probes(success, issued_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_external_probes_source_issued
    ON external_probes(probe_source, issued_at_utc DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_external_probes_probe_id
    ON external_probes(probe_id);

-- ═════════════════════════════════════════════════════════════════════════
-- 9.N.8i · Daily rollup of external_probes.
--
-- Populated by .github/workflows/external-uptime-daily-rollup.yml at 00:05 UTC
-- every day.  Lets /status render the 90-day uptime chart in 1 query
-- instead of scanning ~64k raw rows (90d × ~720 2-min slots).  Raw rows
-- are retained for 90 days; rollup rows are retained forever.
-- ═════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS external_probes_daily (
    day                      TEXT    PRIMARY KEY,
    probes_total             INTEGER NOT NULL DEFAULT 0,
    probes_success           INTEGER NOT NULL DEFAULT 0,
    probes_failed            INTEGER NOT NULL DEFAULT 0,
    uptime_pct               REAL    NOT NULL DEFAULT 0,
    p50_ms                   INTEGER,
    p95_ms                   INTEGER,
    p99_ms                   INTEGER,
    max_ms                   INTEGER,
    error_breakdown_json     TEXT,
    incident_count           INTEGER NOT NULL DEFAULT 0,
    longest_outage_minutes   INTEGER,
    mttr_avg_minutes         INTEGER,
    first_failure_utc        TEXT,
    last_failure_utc         TEXT,
    sources_breakdown_json   TEXT,
    updated_at_utc           TEXT    NOT NULL
);

-- Machine API keys (Bearer tokens) tied to a user account.
-- token_hash    : sha256(raw_token) hex; the raw token is shown to the
--                 caller ONCE on creation and is never recoverable after.
-- token_prefix  : first 12 chars of the raw token (e.g. apin_3f8a1c7d) for
--                 list / display so the user can recognise a key without
--                 ever exposing its full secret.
-- name          : human label ("drone-fleet-prod", "ci-runner").
-- last_used_at  : updated lazily; lets the user retire unused keys.
-- revoked_at    : soft delete. A revoked key never authenticates again.
CREATE TABLE IF NOT EXISTS api_keys (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    token_hash    TEXT NOT NULL UNIQUE,
    token_prefix  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT,
    revoked_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(token_hash);

-- Drone perception scans (Phase 2 batch 1 of the public /api/).
-- One row per inferred frame. Each scan carries the GPS where the frame
-- was taken, the diagnosis APIN produced, and the full per-frame audit
-- JSON. The image_bytes column is optional — operators who don't want
-- raw frames stored just set image_bytes = NULL.
--   scan_uid     : public stable id (scn_ + 16 hex). Shown to callers
--                  instead of the rowid so a database migration cannot
--                  ever shift the visible ids.
--   flight_id    : free-form string the caller supplies to group scans
--                  into a sortie. Not a foreign key — the flights table
--                  is deferred to a later phase; until then the column
--                  is just an index for fast aggregation queries.
--   image_sha256 : sha256 of the raw uploaded bytes; lets the API
--                  dedup a re-uploaded frame at the same GPS.
--   result_json  : the full APINResult dataclass, json-encoded. Lets a
--                  client retrieve the rich payload (decision trace,
--                  per-signal distributions, heatmap) later without
--                  re-running inference.
--   deleted_at   : soft-delete marker so a caller can DELETE without
--                  the row being lost for audit. The list / get
--                  endpoints filter it out; an admin can restore.
CREATE TABLE IF NOT EXISTS scans (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_uid       TEXT NOT NULL UNIQUE,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    api_key_id     INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
    flight_id      TEXT,

    -- inference summary (denormalised from result_json for filter speed)
    diagnosis      TEXT,
    confidence     REAL,
    tier           TEXT,
    severity       TEXT,
    is_ood         INTEGER NOT NULL DEFAULT 0,

    -- geo (REQUIRED for /api/scan — that's the entire point of the route)
    latitude       REAL NOT NULL,
    longitude      REAL NOT NULL,
    altitude_m     REAL,
    heading_deg    REAL,
    accuracy_m     REAL,
    captured_at    TEXT NOT NULL,

    -- image
    image_sha256   TEXT NOT NULL,
    image_bytes    BLOB,
    image_n_bytes  INTEGER,

    -- timing
    processed_at   TEXT NOT NULL,
    processing_ms  INTEGER,

    -- audit
    result_json    TEXT NOT NULL,
    deleted_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_user        ON scans(user_id);
CREATE INDEX IF NOT EXISTS idx_scans_flight      ON scans(flight_id);
CREATE INDEX IF NOT EXISTS idx_scans_captured    ON scans(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_diagnosis   ON scans(diagnosis);
CREATE INDEX IF NOT EXISTS idx_scans_user_active ON scans(user_id, deleted_at);

-- ─── Stage-1 telemetry + analytics tables (v2 schema additions) ─────────────
-- Purpose: industry-grade event logging covering inference, browser sessions,
-- page views, clicks, impressions, API calls, errors, goals, feature usage,
-- A/B experiments. All FK relationships use ON DELETE CASCADE or SET NULL so
-- erasing a user cleans up their telemetry without orphaning anything.
--
-- Design notes:
--  - All "id" PKs on telemetry tables are TEXT (UUID v7-style) so client-side
--    code can pre-generate ids and batch-insert without round-trip.
--  - All timestamps are ISO 8601 TEXT (UTC). Same convention as existing tables.
--  - JSON-typed columns (signal_predictions, properties, etc.) are stored as
--    TEXT — libsql does not have a true JSONB type, and TEXT is portable.
--  - Polymorphic ownership tables (inference_telemetry, inference_feedback,
--    inference_reviews) carry an inference_type discriminator + three nullable
--    FKs (prediction_id / guest_prediction_id / scan_id) where exactly one
--    is set. The discriminator is enforced at the app layer; we don't add a
--    CHECK constraint because libsql parses CHECK constraints conservatively
--    and a typo here would brick the schema deploy.

-- Guest predictions · mirrors the predictions table shape but owned by a
-- guest session (no user). Guest inferences previously went to /dev/null
-- (only inference_count was incremented). Now we keep the full row so
-- "Top disease this week" and "Inferences served" KPIs are accurate.
CREATE TABLE IF NOT EXISTS guest_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_session_id    INTEGER NOT NULL REFERENCES guest_sessions(id) ON DELETE CASCADE,
    crop                TEXT,
    predicted_class     TEXT,
    confidence          REAL,
    tier                TEXT,
    image_sha256        TEXT,
    image_bytes         BLOB,
    heatmap_b64         TEXT,
    response_json       TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    -- Same extension columns as predictions ADD COLUMN below
    api_key_id          INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
    browser_session_id  TEXT,
    client_ip_hash      TEXT,
    user_agent_family   TEXT,
    client_country      TEXT,
    client_region       TEXT,
    client_city         TEXT,
    exif_camera_model   TEXT,
    exif_capture_timestamp TEXT,
    exif_gps_lat        REAL,
    exif_gps_lon        REAL,
    exif_gps_accuracy_m REAL,
    image_perceptual_hash TEXT,
    image_n_bytes       INTEGER,
    image_width         INTEGER,
    image_height        INTEGER,
    image_mimetype      TEXT,
    signal_predictions  TEXT,
    gate_decision_path  TEXT,
    deployment_version  TEXT,
    model_weights_hash  TEXT,
    cold_start          INTEGER NOT NULL DEFAULT 0,
    fallback_to_cpu     INTEGER NOT NULL DEFAULT 0,
    gpu_used            INTEGER NOT NULL DEFAULT 1,
    peak_vram_mb        INTEGER,
    conformal_set       TEXT,
    conformal_set_size  INTEGER,
    ood_flag            INTEGER NOT NULL DEFAULT 0,
    calibration_warning INTEGER NOT NULL DEFAULT 0,
    predicted_top3      TEXT,
    validation_ms       INTEGER,
    router_ms           INTEGER,
    specialist_ms       INTEGER,
    calibration_ms      INTEGER,
    total_ms            INTEGER,
    endpoint            TEXT,
    api_version         TEXT,
    request_id          TEXT,
    trace_id            TEXT,
    status_code         INTEGER,
    error_class         TEXT,
    error_message       TEXT,
    deleted_at          TEXT,
    review_status       TEXT,
    sampled_for_review  INTEGER NOT NULL DEFAULT 0,
    confidence_outlier  INTEGER NOT NULL DEFAULT 0,
    consent_to_research INTEGER NOT NULL DEFAULT 1,
    consent_to_share    INTEGER NOT NULL DEFAULT 1,
    data_residency_region TEXT,
    user_pseudoid       TEXT,
    treatment_advice_shown TEXT,
    grad_cam_generated  INTEGER NOT NULL DEFAULT 0,
    pdf_report_generated INTEGER NOT NULL DEFAULT 0,
    experiment_exposures TEXT
);
CREATE INDEX IF NOT EXISTS idx_gpredictions_guest_date ON guest_predictions(guest_session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gpredictions_class      ON guest_predictions(predicted_class, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_gpredictions_sha        ON guest_predictions(image_sha256);
CREATE INDEX IF NOT EXISTS idx_gpredictions_phash      ON guest_predictions(image_perceptual_hash);
CREATE INDEX IF NOT EXISTS idx_gpredictions_alive      ON guest_predictions(created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_gpredictions_browser    ON guest_predictions(browser_session_id);

-- Browser sessions · DIFFERENT from auth `sessions` table. One row per
-- browser session (idle timeout 30 min). Covers logged-in users, guests,
-- and pure-anonymous visitors equally.
CREATE TABLE IF NOT EXISTS browser_sessions (
    id                       TEXT PRIMARY KEY,
    user_id                  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id         INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    user_pseudoid            TEXT,
    session_start_at         TEXT NOT NULL,
    session_end_at           TEXT,
    last_heartbeat_at        TEXT,
    total_active_ms          INTEGER NOT NULL DEFAULT 0,
    total_idle_ms            INTEGER NOT NULL DEFAULT 0,
    total_hidden_ms          INTEGER NOT NULL DEFAULT 0,
    page_count               INTEGER NOT NULL DEFAULT 0,
    click_count              INTEGER NOT NULL DEFAULT 0,
    inference_count          INTEGER NOT NULL DEFAULT 0,
    error_count              INTEGER NOT NULL DEFAULT 0,
    api_call_count           INTEGER NOT NULL DEFAULT 0,
    device_type              TEXT,
    device_os                TEXT,
    device_os_version        TEXT,
    device_model             TEXT,
    device_browser           TEXT,
    device_browser_version   TEXT,
    user_agent_family        TEXT,
    user_agent_raw           TEXT,
    screen_width             INTEGER,
    screen_height            INTEGER,
    viewport_width           INTEGER,
    viewport_height          INTEGER,
    pixel_ratio              REAL,
    timezone                 TEXT,
    locale                   TEXT,
    connection_type          TEXT,
    network_effective_type   TEXT,
    cpu_cores                INTEGER,
    memory_gb                REAL,
    is_pwa_installed         INTEGER NOT NULL DEFAULT 0,
    referrer_host            TEXT,
    referrer_path            TEXT,
    referrer_url             TEXT,
    entry_url                TEXT,
    exit_url                 TEXT,
    utm_source               TEXT,
    utm_medium               TEXT,
    utm_campaign             TEXT,
    utm_term                 TEXT,
    utm_content              TEXT,
    gclid                    TEXT,
    fbclid                   TEXT,
    ip_country               TEXT,
    ip_region                TEXT,
    ip_city                  TEXT,
    client_ip_hash           TEXT,
    is_returning_user        INTEGER NOT NULL DEFAULT 0,
    deleted_at               TEXT
);
CREATE INDEX IF NOT EXISTS idx_bsessions_user    ON browser_sessions(user_id, session_start_at DESC);
CREATE INDEX IF NOT EXISTS idx_bsessions_guest   ON browser_sessions(guest_session_id, session_start_at DESC);
CREATE INDEX IF NOT EXISTS idx_bsessions_start   ON browser_sessions(session_start_at DESC);
CREATE INDEX IF NOT EXISTS idx_bsessions_alive   ON browser_sessions(session_start_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_bsessions_pseudo  ON browser_sessions(user_pseudoid);
CREATE INDEX IF NOT EXISTS idx_bsessions_ip_geo  ON browser_sessions(ip_country, ip_region);
CREATE INDEX IF NOT EXISTS idx_bsessions_utm     ON browser_sessions(utm_source, utm_campaign);

-- Page views · one row per route entry. SPA navigations count too.
CREATE TABLE IF NOT EXISTS page_views (
    id                            TEXT PRIMARY KEY,
    browser_session_id            TEXT NOT NULL REFERENCES browser_sessions(id) ON DELETE CASCADE,
    user_id                       INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id              INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    page_url                      TEXT NOT NULL,
    page_title                    TEXT,
    page_route                    TEXT NOT NULL,
    navigation_type               TEXT,
    referrer_url                  TEXT,
    referrer_host                 TEXT,
    entered_at                    TEXT NOT NULL,
    left_at                       TEXT,
    active_duration_ms            INTEGER NOT NULL DEFAULT 0,
    idle_duration_ms              INTEGER NOT NULL DEFAULT 0,
    hidden_duration_ms            INTEGER NOT NULL DEFAULT 0,
    max_scroll_depth_pct          REAL NOT NULL DEFAULT 0,
    scroll_milestones_reached     TEXT,
    scroll_pause_points           TEXT,
    ttfb_ms                       INTEGER,
    fcp_ms                        INTEGER,
    lcp_ms                        INTEGER,
    cls                           REAL,
    tti_ms                        INTEGER,
    inp_ms                        INTEGER,
    click_count                   INTEGER NOT NULL DEFAULT 0,
    error_count                   INTEGER NOT NULL DEFAULT 0,
    api_call_count                INTEGER NOT NULL DEFAULT 0,
    bounce                        INTEGER NOT NULL DEFAULT 0,
    engagement_score              REAL,
    deleted_at                    TEXT
);
CREATE INDEX IF NOT EXISTS idx_pageviews_session ON page_views(browser_session_id, entered_at);
CREATE INDEX IF NOT EXISTS idx_pageviews_user    ON page_views(user_id, entered_at DESC);
CREATE INDEX IF NOT EXISTS idx_pageviews_route   ON page_views(page_route, entered_at DESC);
CREATE INDEX IF NOT EXISTS idx_pageviews_alive   ON page_views(entered_at DESC) WHERE deleted_at IS NULL;

-- Clicks · specialised (most-queried event type)
CREATE TABLE IF NOT EXISTS clicks (
    id                                   TEXT PRIMARY KEY,
    browser_session_id                   TEXT NOT NULL REFERENCES browser_sessions(id) ON DELETE CASCADE,
    page_view_id                         TEXT REFERENCES page_views(id) ON DELETE CASCADE,
    user_id                              INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id                     INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    target_tag                           TEXT,
    target_id                            TEXT,
    target_classes                       TEXT,
    target_text                          TEXT,
    target_xpath                         TEXT,
    target_data_attrs                    TEXT,
    click_x_viewport                     INTEGER,
    click_y_viewport                     INTEGER,
    click_x_page                         INTEGER,
    click_y_page                         INTEGER,
    viewport_width_at_click              INTEGER,
    viewport_height_at_click             INTEGER,
    viewport_y_pct                       REAL,
    modifier_keys                        TEXT,
    click_type                           TEXT,
    was_rage_click                       INTEGER NOT NULL DEFAULT 0,
    was_dead_click                       INTEGER NOT NULL DEFAULT 0,
    element_visible_seconds_before_click REAL,
    ms_since_page_view                   INTEGER,
    ms_since_session_start               INTEGER,
    led_to_navigation                    INTEGER NOT NULL DEFAULT 0,
    led_to_modal_open                    INTEGER NOT NULL DEFAULT 0,
    triggered_api_call_ids               TEXT,
    occurred_at                          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clicks_session ON clicks(browser_session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_clicks_pv      ON clicks(page_view_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_clicks_user    ON clicks(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_clicks_target  ON clicks(target_id);
CREATE INDEX IF NOT EXISTS idx_clicks_rage    ON clicks(was_rage_click, occurred_at) WHERE was_rage_click = 1;
CREATE INDEX IF NOT EXISTS idx_clicks_dead    ON clicks(was_dead_click, occurred_at) WHERE was_dead_click = 1;

-- Impressions · element entered viewport (Meta-style exposure)
CREATE TABLE IF NOT EXISTS impressions (
    id                                    TEXT PRIMARY KEY,
    browser_session_id                    TEXT NOT NULL REFERENCES browser_sessions(id) ON DELETE CASCADE,
    page_view_id                          TEXT REFERENCES page_views(id) ON DELETE CASCADE,
    user_id                               INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id                      INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    target_id                             TEXT,
    target_classes                        TEXT,
    target_text                           TEXT,
    target_xpath                          TEXT,
    intersection_ratio_at_first_visible   REAL,
    visibility_duration_ms                INTEGER,
    ms_since_page_view                    INTEGER,
    led_to_interaction                    INTEGER NOT NULL DEFAULT 0,
    occurred_at                           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_impressions_session ON impressions(browser_session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_impressions_pv      ON impressions(page_view_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_impressions_target  ON impressions(target_id);

-- Events · generic high-volume stream (hovers, focus, scroll milestones, settings, custom)
CREATE TABLE IF NOT EXISTS events (
    id                       TEXT PRIMARY KEY,
    browser_session_id       TEXT NOT NULL REFERENCES browser_sessions(id) ON DELETE CASCADE,
    page_view_id             TEXT REFERENCES page_views(id) ON DELETE CASCADE,
    user_id                  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id         INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    event_type               TEXT NOT NULL,
    event_name               TEXT NOT NULL,
    event_version            TEXT NOT NULL DEFAULT 'v1',
    properties               TEXT,
    ms_since_page_view       INTEGER,
    ms_since_session_start   INTEGER,
    occurred_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(browser_session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type, event_name, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_pv      ON events(page_view_id, occurred_at);

-- API calls · client-side fetch log (companion to server-side predictions)
CREATE TABLE IF NOT EXISTS api_calls (
    id                          TEXT PRIMARY KEY,
    browser_session_id          TEXT NOT NULL REFERENCES browser_sessions(id) ON DELETE CASCADE,
    page_view_id                TEXT REFERENCES page_views(id) ON DELETE CASCADE,
    user_id                     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id            INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    endpoint                    TEXT NOT NULL,
    method                      TEXT NOT NULL,
    request_body_size_bytes     INTEGER,
    response_body_size_bytes    INTEGER,
    status_code                 INTEGER,
    client_latency_ms           INTEGER,
    server_latency_ms           INTEGER,
    network_latency_ms          INTEGER,
    error_type                  TEXT,
    retry_count                 INTEGER NOT NULL DEFAULT 0,
    triggered_by                TEXT,
    cache_hit                   INTEGER NOT NULL DEFAULT 0,
    idempotency_key             TEXT,
    request_id                  TEXT,
    occurred_at                 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_apicalls_session  ON api_calls(browser_session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_apicalls_endpoint ON api_calls(endpoint, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_apicalls_user     ON api_calls(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_apicalls_errors   ON api_calls(occurred_at) WHERE status_code >= 400;

-- Inference telemetry · client-side perceived timings + post-result actions.
-- Polymorphic FK: exactly one of (prediction_id, guest_prediction_id, scan_id) is set.
CREATE TABLE IF NOT EXISTS inference_telemetry (
    id                          TEXT PRIMARY KEY,
    browser_session_id          TEXT REFERENCES browser_sessions(id) ON DELETE CASCADE,
    user_id                     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id            INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    inference_type              TEXT NOT NULL,
    prediction_id               INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    guest_prediction_id         INTEGER REFERENCES guest_predictions(id) ON DELETE CASCADE,
    scan_id                     INTEGER REFERENCES scans(id) ON DELETE CASCADE,
    file_selected_at            TEXT,
    file_size_at_select         INTEGER,
    upload_started_at           TEXT,
    upload_completed_at         TEXT,
    upload_duration_ms          INTEGER,
    api_request_sent_at         TEXT,
    api_response_received_at    TEXT,
    result_rendered_at          TEXT,
    perceived_total_ms          INTEGER,
    preview_shown               INTEGER NOT NULL DEFAULT 0,
    result_expanded             INTEGER NOT NULL DEFAULT 0,
    gradcam_viewed              INTEGER NOT NULL DEFAULT 0,
    pdf_exported                INTEGER NOT NULL DEFAULT 0,
    shared                      INTEGER NOT NULL DEFAULT 0,
    feedback_given              INTEGER NOT NULL DEFAULT 0,
    cancelled_by_user           INTEGER NOT NULL DEFAULT 0,
    user_next_action            TEXT,
    occurred_at                 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inftel_prediction      ON inference_telemetry(prediction_id);
CREATE INDEX IF NOT EXISTS idx_inftel_gprediction     ON inference_telemetry(guest_prediction_id);
CREATE INDEX IF NOT EXISTS idx_inftel_scan            ON inference_telemetry(scan_id);
CREATE INDEX IF NOT EXISTS idx_inftel_session         ON inference_telemetry(browser_session_id, occurred_at);

-- Inference feedback · DB-backed user "was this correct?" responses.
CREATE TABLE IF NOT EXISTS inference_feedback (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_type              TEXT NOT NULL,
    prediction_id               INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    guest_prediction_id         INTEGER REFERENCES guest_predictions(id) ON DELETE CASCADE,
    scan_id                     INTEGER REFERENCES scans(id) ON DELETE CASCADE,
    submitter_user_id           INTEGER REFERENCES users(id) ON DELETE SET NULL,
    submitter_guest_session_id  INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    verdict                     TEXT NOT NULL,
    corrected_class             TEXT,
    comment                     TEXT,
    submitted_at                TEXT NOT NULL,
    agronomist_verified         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_infb_prediction ON inference_feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_infb_gpred      ON inference_feedback(guest_prediction_id);
CREATE INDEX IF NOT EXISTS idx_infb_scan       ON inference_feedback(scan_id);
CREATE INDEX IF NOT EXISTS idx_infb_submitted  ON inference_feedback(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_infb_verdict    ON inference_feedback(verdict, submitted_at DESC);

-- Inference reviews · human-in-the-loop verification queue.
CREATE TABLE IF NOT EXISTS inference_reviews (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_type      TEXT NOT NULL,
    prediction_id       INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    guest_prediction_id INTEGER REFERENCES guest_predictions(id) ON DELETE CASCADE,
    scan_id             INTEGER REFERENCES scans(id) ON DELETE CASCADE,
    reviewer_user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    sampled_at          TEXT NOT NULL,
    reviewed_at         TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    verdict             TEXT,
    actual_class        TEXT,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_infrev_status   ON inference_reviews(status, sampled_at);
CREATE INDEX IF NOT EXISTS idx_infrev_pred     ON inference_reviews(prediction_id);
CREATE INDEX IF NOT EXISTS idx_infrev_gpred    ON inference_reviews(guest_prediction_id);
CREATE INDEX IF NOT EXISTS idx_infrev_scan     ON inference_reviews(scan_id);

-- Errors · client + server errors that bubbled to UI.
CREATE TABLE IF NOT EXISTS errors (
    id                  TEXT PRIMARY KEY,
    browser_session_id  TEXT REFERENCES browser_sessions(id) ON DELETE CASCADE,
    page_view_id        TEXT REFERENCES page_views(id) ON DELETE CASCADE,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    guest_session_id    INTEGER REFERENCES guest_sessions(id) ON DELETE SET NULL,
    error_type          TEXT NOT NULL,
    error_message       TEXT,
    error_stack         TEXT,
    source_file         TEXT,
    source_line         INTEGER,
    source_column       INTEGER,
    url                 TEXT,
    user_agent          TEXT,
    shown_to_user       INTEGER NOT NULL DEFAULT 0,
    recovery_action     TEXT,
    occurred_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_errors_session  ON errors(browser_session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_errors_type     ON errors(error_type, occurred_at DESC);

-- Goals · named conversion events
CREATE TABLE IF NOT EXISTS goals (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    guest_session_id                INTEGER REFERENCES guest_sessions(id) ON DELETE CASCADE,
    browser_session_id              TEXT REFERENCES browser_sessions(id) ON DELETE CASCADE,
    goal_name                       TEXT NOT NULL,
    achieved_at                     TEXT NOT NULL,
    time_from_session_start_ms      INTEGER,
    time_from_signup_ms             INTEGER,
    goal_value                      REAL,
    utm_source_at_first_visit       TEXT,
    utm_medium_at_first_visit       TEXT,
    utm_campaign_at_first_visit     TEXT,
    touchpoint_count                INTEGER,
    attribution_path                TEXT
);
CREATE INDEX IF NOT EXISTS idx_goals_user      ON goals(user_id, achieved_at DESC);
CREATE INDEX IF NOT EXISTS idx_goals_guest     ON goals(guest_session_id, achieved_at DESC);
CREATE INDEX IF NOT EXISTS idx_goals_name      ON goals(goal_name, achieved_at DESC);

-- Feature usage · per-user-per-feature rollup.
CREATE TABLE IF NOT EXISTS feature_usage (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES users(id) ON DELETE CASCADE,
    guest_session_id    INTEGER REFERENCES guest_sessions(id) ON DELETE CASCADE,
    feature_name        TEXT NOT NULL,
    first_used_at       TEXT,
    last_used_at        TEXT,
    use_count           INTEGER NOT NULL DEFAULT 0,
    total_dwell_ms      INTEGER NOT NULL DEFAULT 0,
    first_session_id    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_featuse_user_feat   ON feature_usage(user_id, feature_name) WHERE user_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_featuse_guest_feat  ON feature_usage(guest_session_id, feature_name) WHERE guest_session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_featuse_feature ON feature_usage(feature_name, last_used_at DESC);

-- Experiments exposures · A/B test variant assignments.
CREATE TABLE IF NOT EXISTS experiments_exposures (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES users(id) ON DELETE CASCADE,
    guest_session_id    INTEGER REFERENCES guest_sessions(id) ON DELETE CASCADE,
    browser_session_id  TEXT REFERENCES browser_sessions(id) ON DELETE CASCADE,
    experiment_name     TEXT NOT NULL,
    variant             TEXT NOT NULL,
    exposed_at          TEXT NOT NULL,
    page_view_id        TEXT REFERENCES page_views(id) ON DELETE SET NULL,
    properties          TEXT
);
CREATE INDEX IF NOT EXISTS idx_expexp_name      ON experiments_exposures(experiment_name, variant);
CREATE INDEX IF NOT EXISTS idx_expexp_user      ON experiments_exposures(user_id, exposed_at DESC);
CREATE INDEX IF NOT EXISTS idx_expexp_session   ON experiments_exposures(browser_session_id);

-- Predictions daily rollup · pre-aggregated for fast KPI queries.
CREATE TABLE IF NOT EXISTS predictions_daily (
    date                TEXT PRIMARY KEY,
    total_count         INTEGER NOT NULL DEFAULT 0,
    user_count          INTEGER NOT NULL DEFAULT 0,
    guest_count         INTEGER NOT NULL DEFAULT 0,
    scan_count          INTEGER NOT NULL DEFAULT 0,
    success_count       INTEGER NOT NULL DEFAULT 0,
    error_count         INTEGER NOT NULL DEFAULT 0,
    mean_confidence     REAL,
    median_latency_ms   INTEGER,
    p95_latency_ms      INTEGER,
    per_crop_counts     TEXT,
    per_class_counts    TEXT,
    per_tier_counts     TEXT,
    unique_users        INTEGER NOT NULL DEFAULT 0,
    unique_sessions     INTEGER NOT NULL DEFAULT 0,
    unique_guests       INTEGER NOT NULL DEFAULT 0,
    computed_at         TEXT NOT NULL
);

-- ─── FK indexes · PDA Round-1 audit finding · index every FK column ──────
-- Industry standard: every FK column gets an index, otherwise JOIN and
-- ON DELETE CASCADE queries become full table scans at scale.
CREATE INDEX IF NOT EXISTS idx_gpredictions_apikey       ON guest_predictions(api_key_id);
CREATE INDEX IF NOT EXISTS idx_pageviews_guest           ON page_views(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_clicks_guest              ON clicks(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_impressions_user          ON impressions(user_id);
CREATE INDEX IF NOT EXISTS idx_impressions_guest         ON impressions(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_events_user               ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_guest              ON events(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_apicalls_guest            ON api_calls(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_apicalls_pv               ON api_calls(page_view_id);
CREATE INDEX IF NOT EXISTS idx_inftel_user               ON inference_telemetry(user_id);
CREATE INDEX IF NOT EXISTS idx_inftel_guest              ON inference_telemetry(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_infb_submitter_user       ON inference_feedback(submitter_user_id);
CREATE INDEX IF NOT EXISTS idx_infb_submitter_guest      ON inference_feedback(submitter_guest_session_id);
CREATE INDEX IF NOT EXISTS idx_infrev_reviewer           ON inference_reviews(reviewer_user_id);
CREATE INDEX IF NOT EXISTS idx_errors_user               ON errors(user_id);
CREATE INDEX IF NOT EXISTS idx_errors_guest              ON errors(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_errors_pv                 ON errors(page_view_id);
CREATE INDEX IF NOT EXISTS idx_goals_session             ON goals(browser_session_id);
CREATE INDEX IF NOT EXISTS idx_expexp_guest              ON experiments_exposures(guest_session_id);
CREATE INDEX IF NOT EXISTS idx_expexp_pv                 ON experiments_exposures(page_view_id);

-- Unified read-only VIEW · UNIONs user + guest predictions for analytics.
-- Scans stay separate because they have a different shape (GPS/flight context).
-- Every KPI query on "Inferences served" reads from this view, not the base
-- tables, so adding a new ownership type later (e.g. an embedded-edge mode)
-- is a one-line VIEW change rather than touching every analytics query.
CREATE VIEW IF NOT EXISTS all_predictions AS
SELECT
    'user' AS owner_type,
    id, user_id, NULL AS guest_session_id,
    crop, predicted_class, confidence, tier,
    image_sha256, created_at, deleted_at
FROM predictions
UNION ALL
SELECT
    'guest' AS owner_type,
    id, NULL AS user_id, guest_session_id,
    crop, predicted_class, confidence, tier,
    image_sha256, created_at, deleted_at
FROM guest_predictions;
"""


def _migrate_predictions_blob_columns(c: sqlite3.Connection) -> None:
    """Idempotent ALTER TABLE migration for Phase-3.5 image storage.

    SQLite has no `ADD COLUMN IF NOT EXISTS`, so we introspect the schema
    first via `pragma_table_info` and only run each ALTER if the column
    is missing.  Safe to run on every startup; safe on a fresh DB where
    the columns are already present from the CREATE TABLE.

    Pre-existing prediction rows simply get NULL for both columns —
    serving routes return 404 for those rows and the frontend renders
    an honest "image not captured (pre-upgrade)" placeholder rather
    than a misleading emoji.
    """
    existing = {row[1] for row in c.execute("PRAGMA table_info(predictions)")}
    if "image_bytes" not in existing:
        c.execute("ALTER TABLE predictions ADD COLUMN image_bytes BLOB")
    if "heatmap_b64" not in existing:
        c.execute("ALTER TABLE predictions ADD COLUMN heatmap_b64 TEXT")


# ─── Stage-1 v2 schema migrations · additive ADD COLUMN, idempotent ────────
# Every ALTER TABLE here is wrapped in a PRAGMA introspect so it can run on
# every server boot without error. The same pattern auth_db.py already uses
# for _migrate_predictions_blob_columns. Safe on both libsql (Turso) and
# local SQLite. Each individual ALTER is wrapped in try/except so one bad
# add doesn't block the rest of the migration.
def _add_columns(c, table: str, columns: list[tuple[str, str]]) -> None:
    """Idempotently ADD COLUMN. `columns` is a list of (name, "SQL TYPE DEFAULT ...").

    REV-R2-I04 (§22.3 row 4): each individual ALTER is still wrapped in
    try/except so one transient failure (e.g. a column added by a concurrent
    boot, an autoincrement-class limit) doesn't cascade. BUT after the loop
    we re-introspect the table and raise `RuntimeError` if any EXPECTED
    column is still missing — fail loudly on misconfiguration instead of
    silently continuing with a broken schema.

    This catches:
      - libsql backends that genuinely lack ALTER TABLE ADD COLUMN
      - typos in column declarations the catch-Exception swallowed
      - permission errors where the user can read but not ALTER

    Without the post-loop check, the server boots with missing columns and
    fails at first query with an obscure `OperationalError: no such column`.
    """
    import logging as _l
    log = _l.getLogger("apin_v2.auth")

    try:
        existing = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
    except Exception as e:
        # If we can't even read the schema, the table likely doesn't exist —
        # let the caller see this as a real error (this used to silently
        # return — now we surface it).
        raise RuntimeError(
            f"_add_columns: cannot introspect table {table!r}: {e}"
        ) from e

    for name, decl in columns:
        if name in existing:
            continue
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        except Exception as e:
            # Don't fail the whole loop here — some ALTER errors are
            # idempotent (concurrent migrations racing). The post-loop
            # verification will catch genuine omissions.
            log.warning(
                "ALTER TABLE %s ADD COLUMN %s failed (will verify post-loop): %s",
                table, name, e)

    # Post-loop verification (REV-R2-I04): every requested column MUST exist
    # now, regardless of whether the ALTER raised or no-op'd.
    try:
        final = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
    except Exception as e:
        raise RuntimeError(
            f"_add_columns: post-loop introspection failed for {table!r}: {e}"
        ) from e

    expected = {name for name, _ in columns}
    missing = expected - final
    if missing:
        raise RuntimeError(
            f"_add_columns failed: missing {sorted(missing)} from table "
            f"{table!r}. Schema is incomplete — check the warning log above "
            f"for the underlying ALTER TABLE error."
        )


def _migrate_v2_extensions(c) -> None:
    """Idempotent · additive · safe to re-run · v2 schema fields for
    predictions / users / guest_sessions / scans. Existing rows get NULL or
    the column default. No existing read path queries these columns yet, so
    NULL on old rows is invisible until callers explicitly opt in.
    """
    # predictions table · extension columns
    _add_columns(c, "predictions", [
        ("api_key_id",              "INTEGER REFERENCES api_keys(id) ON DELETE SET NULL"),
        ("browser_session_id",      "TEXT"),
        ("client_ip_hash",          "TEXT"),
        ("user_agent_family",       "TEXT"),
        ("client_country",          "TEXT"),
        ("client_region",           "TEXT"),
        ("client_city",             "TEXT"),
        ("exif_camera_model",       "TEXT"),
        ("exif_capture_timestamp",  "TEXT"),
        ("exif_gps_lat",            "REAL"),
        ("exif_gps_lon",            "REAL"),
        ("exif_gps_accuracy_m",     "REAL"),
        ("image_perceptual_hash",   "TEXT"),
        ("image_n_bytes",           "INTEGER"),
        ("image_width",             "INTEGER"),
        ("image_height",            "INTEGER"),
        ("image_mimetype",          "TEXT"),
        ("signal_predictions",      "TEXT"),
        ("gate_decision_path",      "TEXT"),
        ("deployment_version",      "TEXT"),
        ("model_weights_hash",      "TEXT"),
        ("cold_start",              "INTEGER NOT NULL DEFAULT 0"),
        ("fallback_to_cpu",         "INTEGER NOT NULL DEFAULT 0"),
        ("gpu_used",                "INTEGER NOT NULL DEFAULT 1"),
        ("peak_vram_mb",            "INTEGER"),
        ("conformal_set",           "TEXT"),
        ("conformal_set_size",      "INTEGER"),
        ("ood_flag",                "INTEGER NOT NULL DEFAULT 0"),
        ("calibration_warning",     "INTEGER NOT NULL DEFAULT 0"),
        ("predicted_top3",          "TEXT"),
        ("validation_ms",           "INTEGER"),
        ("router_ms",               "INTEGER"),
        ("specialist_ms",           "INTEGER"),
        ("calibration_ms",          "INTEGER"),
        ("total_ms",                "INTEGER"),
        ("endpoint",                "TEXT"),
        ("api_version",             "TEXT"),
        ("request_id",              "TEXT"),
        ("trace_id",                "TEXT"),
        ("status_code",             "INTEGER"),
        ("error_class",             "TEXT"),
        ("error_message",           "TEXT"),
        ("deleted_at",              "TEXT"),
        ("review_status",           "TEXT"),
        ("sampled_for_review",      "INTEGER NOT NULL DEFAULT 0"),
        ("confidence_outlier",      "INTEGER NOT NULL DEFAULT 0"),
        ("consent_to_research",     "INTEGER NOT NULL DEFAULT 1"),
        ("consent_to_share",        "INTEGER NOT NULL DEFAULT 1"),
        ("data_residency_region",   "TEXT"),
        ("user_pseudoid",           "TEXT"),
        ("treatment_advice_shown",  "TEXT"),
        ("grad_cam_generated",      "INTEGER NOT NULL DEFAULT 0"),
        ("pdf_report_generated",    "INTEGER NOT NULL DEFAULT 0"),
        ("experiment_exposures",    "TEXT"),
    ])
    # users table · extension columns
    _add_columns(c, "users", [
        ("consent_to_research",             "INTEGER NOT NULL DEFAULT 1"),
        ("consent_to_share",                "INTEGER NOT NULL DEFAULT 1"),
        ("data_residency_region",           "TEXT"),
        ("first_seen_country",              "TEXT"),
        ("utm_source_at_signup",            "TEXT"),
        ("utm_medium_at_signup",            "TEXT"),
        ("utm_campaign_at_signup",          "TEXT"),
        ("utm_term_at_signup",              "TEXT"),
        ("utm_content_at_signup",           "TEXT"),
        ("referrer_at_signup",              "TEXT"),
        ("converted_from_guest_session_id", "INTEGER"),
    ])
    # guest_sessions table · extension columns
    _add_columns(c, "guest_sessions", [
        ("user_agent_family",       "TEXT"),
        ("ip_country",              "TEXT"),
        ("ip_region",               "TEXT"),
        ("ip_city",                 "TEXT"),
        ("utm_source",              "TEXT"),
        ("utm_medium",              "TEXT"),
        ("utm_campaign",            "TEXT"),
        ("utm_term",                "TEXT"),
        ("utm_content",             "TEXT"),
        ("entry_url",               "TEXT"),
        ("referrer_host",           "TEXT"),
        ("converted_to_user_id",    "INTEGER REFERENCES users(id) ON DELETE SET NULL"),
        ("converted_at",            "TEXT"),
    ])
    # scans table · extension columns (drone path inherits the same telemetry surface)
    _add_columns(c, "scans", [
        ("signal_predictions",      "TEXT"),
        ("gate_decision_path",      "TEXT"),
        ("deployment_version",      "TEXT"),
        ("model_weights_hash",      "TEXT"),
        ("cold_start",              "INTEGER NOT NULL DEFAULT 0"),
        ("ood_flag",                "INTEGER NOT NULL DEFAULT 0"),
        ("calibration_warning",     "INTEGER NOT NULL DEFAULT 0"),
        ("client_country",          "TEXT"),
    ])

    # Stage 6 · browser_sessions gets a denormalised `current_route` so
    # the "Live now" KPI tile can do a single GROUP BY against the table
    # instead of a per-row subquery into page_views. The telemetry library
    # stamps this on every flush; live_sessions_by_route() reads it.
    _add_columns(c, "browser_sessions", [
        ("current_route",           "TEXT"),
    ])

    # 9.N.8 · Pass 2 — request-detail drawer columns on api_key_request_log.
    # Idempotent ADD COLUMN; old rows get NULL and the drawer renders a
    # "not recorded for older requests" fallback per section.
    _add_columns(c, "api_key_request_log", [
        ("headers_in_json",      "TEXT"),
        ("headers_out_json",     "TEXT"),
        ("body_in_preview",      "TEXT"),
        ("body_out_preview",     "TEXT"),
        ("body_in_ctype",        "TEXT"),
        ("body_out_ctype",       "TEXT"),
        ("body_in_truncated",    "INTEGER"),
        ("body_out_truncated",   "INTEGER"),
        ("stage_timings_json",   "TEXT"),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 7 — API CONSOLE — Phase 1 schema (spec_v7 §6, all 12 tables/ALTERs)
#
# Additive: never drops or renames an existing column. Idempotent across:
#   - First boot on fresh DB           (CREATE … IF NOT EXISTS handles it)
#   - Re-run on an already-migrated DB (sentinels + _add_columns skip)
#   - Mid-migration crash and restart  (each statement is independently safe)
#
# Run order (CRITICAL):
#   1. _migrate_stage7_alter_api_keys(c)      adds 24 columns + idx + backfill
#   2. _migrate_stage7_alter_sessions(c)      adds csrf_token + backfill
#   3. STAGE7_SCHEMA_SQL executed             creates the 8 new tables/triggers
#
# Why this order: the new tables FK to api_keys.public_id. public_id is added
# (and the unique index created) by step 1. Step 3 will fail if step 1 didn't
# run first.
# ═════════════════════════════════════════════════════════════════════════════

# Legacy-key detection uses the `status='legacy_pending'` sentinel rather
# than a created_at cutoff (REV-R3-I04). The sentinel is the column DEFAULT
# applied to every existing api_keys row by the §6.1 ALTER, BEFORE any
# reclassification UPDATE fires. While the sentinel is still in place we can
# safely mark rows as "legacy" — once Step 4 reclassifies to 'active' or
# 'disabled', the marker disappears and the work cannot be re-run by
# accident on new keys (PDA-P1-R1-F02 / F03 / F05 fixes).
#
# `_STAGE7_LEGACY_CUTOFF` REMOVED entirely. Previous form used lexicographic
# string compare against ISO-8601 timestamps; `+00:00` (0x2B) sorts before
# `Z` (0x5A), so a key minted one microsecond past midnight was mis-tagged
# as legacy (CSRF defence loss, silent new-key-alert suppression).

# Default scopes for legacy keys (PDA-F26): full non-console scope set so
# existing CLI users keep working. The Console-only scopes (account:*) are
# excluded — legacy keys cannot manage the Console.
_STAGE7_LEGACY_SCOPES = (
    '["predict:write","predict:read","predict:delete",'
    '"models:read","models:benchmarks","feedback:write",'
    '"reports:read","reports:write","usage:read",'
    '"alerts:read","alerts:write","webhooks:read",'
    '"webhooks:write","account:read","account:write"]'
)

# The 8 new tables + indexes + triggers. Wrapped as a single string so we
# can fire it through either `executescript` (SQLite) or `_split_sql + batch`
# (libsql via the shim's executescript path).
STAGE7_SCHEMA_SQL = """
-- ── §6.2 api_key_audit ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_key_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id      TEXT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    actor_ip    TEXT,
    actor_ua    TEXT    CHECK (length(actor_ua) <= 1024),
    actor_session_id TEXT,
    sudo_token_id TEXT,
    before_json TEXT,
    after_json  TEXT,
    details     TEXT    CHECK (length(details) <= 4096),
    key_name_at_time TEXT NOT NULL DEFAULT '',
    prev_hash   TEXT NOT NULL DEFAULT '0000000000000000000000000000000000000000000000000000000000000000',
    row_hash    TEXT NOT NULL,
    FOREIGN KEY (key_id) REFERENCES api_keys(public_id) ON DELETE SET NULL,
    CHECK (COALESCE(length(before_json),0) + COALESCE(length(after_json),0) <= 8192)
);
CREATE INDEX IF NOT EXISTS idx_api_key_audit_key    ON api_key_audit(key_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_key_audit_user   ON api_key_audit(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_key_audit_action ON api_key_audit(action, timestamp DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_key_audit_user_prev ON api_key_audit(user_id, prev_hash);
CREATE TRIGGER IF NOT EXISTS api_key_audit_no_update
  BEFORE UPDATE ON api_key_audit
  BEGIN
    SELECT RAISE(ABORT, 'audit rows are append-only');
  END;

-- ── §6.3 api_key_usage_minute ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_key_usage_minute (
    key_id          TEXT    NOT NULL REFERENCES api_keys(public_id) ON DELETE CASCADE,
    minute_ts       TEXT    NOT NULL,
    requests        INTEGER NOT NULL DEFAULT 0,
    errors          INTEGER NOT NULL DEFAULT 0,
    rate_limited    INTEGER NOT NULL DEFAULT 0,
    quota_blocked   INTEGER NOT NULL DEFAULT 0,
    bytes_in        INTEGER NOT NULL DEFAULT 0,
    bytes_out       INTEGER NOT NULL DEFAULT 0,
    latency_sum_ms  INTEGER NOT NULL DEFAULT 0,
    latency_count   INTEGER NOT NULL DEFAULT 0,
    latency_p50_ms  REAL,
    latency_p95_ms  REAL,
    latency_p99_ms  REAL,
    PRIMARY KEY (key_id, minute_ts)
);
CREATE INDEX IF NOT EXISTS idx_usage_key_time
  ON api_key_usage_minute(key_id, minute_ts DESC);

-- ── §6.4 api_key_request_log ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_key_request_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id      TEXT NOT NULL REFERENCES api_keys(public_id) ON DELETE CASCADE,
    timestamp   TEXT NOT NULL,
    method      TEXT NOT NULL,
    path        TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms  INTEGER,
    ip          TEXT,
    ua          TEXT,
    bytes_in    INTEGER,
    bytes_out   INTEGER,
    error_code  TEXT,
    via         TEXT,
    -- 9.N.8 · Pass 2 columns for the request-detail drawer.
    -- These are nullable so older rows (pre-migration) still load.
    -- Bodies are stored as previews (first 4 KB max); full bodies are
    -- never persisted server-side per privacy rules.
    headers_in_json   TEXT,     -- JSON object of redacted request headers
    headers_out_json  TEXT,     -- JSON object of response headers
    body_in_preview   TEXT,     -- first 4 KB of request body (text-decoded if possible)
    body_out_preview  TEXT,     -- first 4 KB of response body
    body_in_ctype     TEXT,     -- content-type of the request body
    body_out_ctype    TEXT,     -- content-type of the response body
    body_in_truncated INTEGER,  -- 1 if the preview is truncated (full was larger)
    body_out_truncated INTEGER, -- 1 if response preview is truncated
    stage_timings_json TEXT     -- JSON {"auth":3,"validate":1,"handler":38,"serialize":3,"send":2}
);
CREATE INDEX IF NOT EXISTS idx_req_log_key_time_status
  ON api_key_request_log(key_id, timestamp DESC, status_code);

-- ── §6.5 webhooks ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS webhooks (
    id              TEXT PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url             TEXT NOT NULL CHECK (length(url) <= 2048),
    secret_hash     TEXT NOT NULL,
    secret_encrypted BLOB NOT NULL,
    secret_encrypted_old BLOB,
    secret_old_until TEXT,
    events          TEXT NOT NULL CHECK (length(events) <= 4096),
    description     TEXT CHECK (length(description) <= 280),
    active          INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_delivery_at TEXT,
    last_delivery_status INTEGER,
    consecutive_failure_count INTEGER NOT NULL DEFAULT 0,
    consecutive_gave_up_count INTEGER NOT NULL DEFAULT 0,
    auto_disabled_at TEXT,
    allow_self_signed INTEGER NOT NULL DEFAULT 0,
    allow_internal_target INTEGER NOT NULL DEFAULT 0,
    pinned_ip       TEXT
);
CREATE INDEX IF NOT EXISTS idx_webhooks_user ON webhooks(user_id, active);

-- ── §6.6 webhook_deliveries ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    webhook_id      TEXT NOT NULL REFERENCES webhooks(id) ON DELETE CASCADE,
    event_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    signature       TEXT NOT NULL,
    queued_at       TEXT NOT NULL,
    next_attempt_at TEXT NOT NULL,
    first_attempt_at TEXT,
    last_attempt_at  TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0 CHECK (attempts <= 8),
    status          TEXT NOT NULL DEFAULT 'queued',
    response_status INTEGER,
    response_body   TEXT,
    response_truncated INTEGER NOT NULL DEFAULT 0,
    response_headers TEXT,
    error_text      TEXT,
    replay_of       INTEGER REFERENCES webhook_deliveries(id) ON DELETE SET NULL,
    claimed_by      TEXT,
    claimed_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_deliveries_webhook
  ON webhook_deliveries(webhook_id, queued_at DESC);
CREATE INDEX IF NOT EXISTS idx_deliveries_pending
  ON webhook_deliveries(status, next_attempt_at)
  WHERE status IN ('queued','in_flight');

-- ── §6.7 api_key_alerts ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_key_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_id      TEXT REFERENCES api_keys(public_id) ON DELETE SET NULL,
    severity    TEXT NOT NULL CHECK (severity IN ('info','warn','critical')),
    code        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    details_json TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    read_at     TEXT,
    dismissed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_user_unread
  ON api_key_alerts(user_id, read_at, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_dedup
  ON api_key_alerts(user_id, code, key_id, updated_at);

-- ── §6.8 sudo_tokens ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sudo_tokens (
    id          TEXT PRIMARY KEY,
    token_hash  TEXT NOT NULL UNIQUE,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id  TEXT NOT NULL,
    bound_ip    TEXT,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    used_count  INTEGER NOT NULL DEFAULT 0,
    revoked_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sudo_active   ON sudo_tokens(user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_sudo_by_hash  ON sudo_tokens(token_hash);

-- ── §6.9 idempotency_keys ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS idempotency_keys (
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key             TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    request_hash    TEXT NOT NULL,
    response_status INTEGER NOT NULL,
    response_body   TEXT NOT NULL,
    plaintext_stripped_at TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
CREATE INDEX IF NOT EXISTS idx_idem_expires ON idempotency_keys(expires_at);

-- ── §6.10 account_settings ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS account_settings (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    default_key_expiry_days INTEGER,
    default_scope_template  TEXT NOT NULL DEFAULT 'inference_only'
                            CHECK (default_scope_template IN
                                   ('inference_only','read_only',
                                    'full_inference','webhooks')),
    require_ip_allowlist    INTEGER NOT NULL DEFAULT 0,
    sudo_required_for       TEXT NOT NULL DEFAULT
                            '["create_key","revoke_key","rotate_key","change_scope","change_allowlist","create_webhook","revoke_webhook","change_settings"]'
                            CHECK (json_valid(sudo_required_for)),
    notify_on_key_created   INTEGER NOT NULL DEFAULT 1,
    notify_on_first_use_from_new_ip INTEGER NOT NULL DEFAULT 1,
    notify_on_quota_exceeded INTEGER NOT NULL DEFAULT 1,
    notify_on_auth_failures_threshold INTEGER,
    sudo_session_length_seconds INTEGER NOT NULL DEFAULT 300
                                CHECK (sudo_session_length_seconds BETWEEN 60 AND 900),
    sudo_max_uses INTEGER NOT NULL DEFAULT 50
                  CHECK (sudo_max_uses BETWEEN 10 AND 200),
    max_webhooks_per_user INTEGER NOT NULL DEFAULT 50
                          CHECK (max_webhooks_per_user BETWEEN 1 AND 200),
    request_log_retention_days INTEGER NOT NULL DEFAULT 30
                                CHECK (request_log_retention_days BETWEEN 1 AND 90),
    audit_log_retention_days INTEGER NOT NULL DEFAULT 365
                              CHECK (audit_log_retention_days BETWEEN 30 AND 1825),
    webhook_delivery_retention_days INTEGER NOT NULL DEFAULT 90
                                     CHECK (webhook_delivery_retention_days BETWEEN 7 AND 365),
    ip_truncation_enabled INTEGER NOT NULL DEFAULT 0,
    auto_revoke_on_partner_detection INTEGER NOT NULL DEFAULT 1,
    org_aggregate_rate_limit INTEGER NOT NULL DEFAULT 600,
    org_aggregate_quota_per_day INTEGER NOT NULL DEFAULT 100000,
    sandbox_rate_per_min INTEGER NOT NULL DEFAULT 600,
    sandbox_max_upload_mb_per_min INTEGER NOT NULL DEFAULT 50,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- ── 9.N.9 api_key_health_snapshot — one composite-health row per key per
-- UTC day, upserted lazily on each Overview load (window='24h'). Powers the
-- 30-day composite-trend line in the Health-score expanded view. No separate
-- scheduler: the read path writes today's row, so the series accrues over time.
CREATE TABLE IF NOT EXISTS api_key_health_snapshot (
    key_id      TEXT NOT NULL REFERENCES api_keys(public_id) ON DELETE CASCADE,
    day         TEXT NOT NULL,          -- 'YYYY-MM-DD' (UTC)
    composite   REAL,
    reliability REAL,
    performance REAL,
    capacity    REAL,
    hygiene     REAL,
    sample_size INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (key_id, day)
);
CREATE INDEX IF NOT EXISTS idx_health_snap_key_day
  ON api_key_health_snapshot(key_id, day DESC);

-- Auto-create account_settings for every new user (PDA-R2-F12).
CREATE TRIGGER IF NOT EXISTS trg_user_insert_default_settings
  AFTER INSERT ON users
  BEGIN
    INSERT OR IGNORE INTO account_settings(user_id, created_at, updated_at)
    VALUES (NEW.id, datetime('now'), datetime('now'));
  END;
"""


def _migrate_stage7_alter_api_keys(c) -> None:
    """§6.1 — ADD 24 columns to api_keys, UNIQUE INDEX on public_id, then
    backfill legacy rows. Idempotent (sentinel-gated) and re-runnable."""
    import logging as _l
    log = _l.getLogger("apin_v2.auth.stage7")

    # The 24 new columns. _add_columns will raise RuntimeError if any
    # cannot be added (post-loop verification — PDA-P0-R1-F04 wiring).
    # Note: last_used_at already exists on the legacy api_keys table; the
    # post-loop check tolerates pre-existing columns.
    _add_columns(c, "api_keys", [
        ("environment",           "TEXT NOT NULL DEFAULT 'live'"),
        ("scopes",                "TEXT NOT NULL DEFAULT '[]'"),
        ("last_four",             "TEXT"),
        ("ip_allowlist",          "TEXT"),
        ("origin_allowlist",      "TEXT"),
        ("rate_limit_per_min",    "INTEGER"),
        ("quota_per_day",         "INTEGER"),
        ("expires_at",            "TEXT"),
        ("created_ip",            "TEXT"),
        ("created_ua",            "TEXT"),
        ("last_used_ip",          "TEXT"),
        ("last_used_ua",          "TEXT"),
        ("request_count",         "INTEGER NOT NULL DEFAULT 0"),
        ("error_count",           "INTEGER NOT NULL DEFAULT 0"),
        ("status",                "TEXT NOT NULL DEFAULT 'legacy_pending'"),
        ("predecessor_id",        "TEXT"),
        ("successor_id",          "TEXT"),
        ("rotation_grace_until",  "TEXT"),
        ("restore_blocked",       "INTEGER NOT NULL DEFAULT 0"),
        ("note",                  "TEXT"),
        ("public_id",             "TEXT"),
        ("deleted_at",            "TEXT"),
        ("enforce_origin_for_non_browser", "INTEGER NOT NULL DEFAULT 1"),
        ("legacy_alert_emitted",  "INTEGER NOT NULL DEFAULT 0"),
        # Phase 8.H · FIX-MIGRATION-GAP: disable_console_api_key /
        # enable_console_api_key / rotate / patch all write to updated_at,
        # but the original Phase-1 migration omitted it. On Turso this
        # caused every state-changing write to crash with
        # `SQLite error: no such column: updated_at` (surfacing as a bare
        # KeyError because libsql_client mishandles error responses).
        # Adding it now backfills NULL on existing rows — read paths
        # don't reference the column directly, only writes do.
        ("updated_at",            "TEXT"),
    ])

    # Indexes — created BEFORE backfill so any UNIQUE-collision retries
    # (public_id) work during the backfill loop.
    #
    # PDA-P1-R1-F04 fix: index-creation errors are NO LONGER silently
    # swallowed. The UNIQUE index on `public_id` is load-bearing for the
    # backfill loop (the retry-on-collision logic depends on it firing
    # IntegrityError). If we silently warn-and-continue when the index
    # fails to create, we'd silently lose collision detection — two rows
    # could end up with the same public_id and ON DELETE SET NULL would
    # spread collateral damage to webhooks/usage/audit.
    #
    # Partial-index failures (the two `WHERE …` indexes) are tolerated on
    # libsql backends that lack partial-index support. The main UNIQUE
    # index on public_id MUST succeed.
    # PDA-P1-R2-F02 fix: collapse the two tuples into one explicit list so
    # the `(stmt, required)` flag carries the full semantic. The previous
    # split into `REQUIRED` vs `OPTIONAL_PARTIAL` was misleading because
    # `idx_api_keys_status` was non-partial AND required=True yet lived in
    # the OPTIONAL_PARTIAL bucket. Tuple order matters: public_id UNIQUE
    # index MUST be created first (the backfill loop relies on it firing
    # IntegrityError on collision).
    STAGE7_INDEXES = (
        # Required (load-bearing) — failure must raise:
        ("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_public_id_unique "
         "ON api_keys(public_id)",
         True),
        ("CREATE INDEX IF NOT EXISTS idx_api_keys_status "
         "ON api_keys(user_id, status)",
         True),
        # Optional (partial indexes) — failure tolerated on backends
        # that lack partial-index support:
        ("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_user_name_active "
         "ON api_keys(user_id, name) WHERE status IN ('active','rotating')",
         False),
        ("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at "
         "ON api_keys(expires_at) WHERE expires_at IS NOT NULL",
         False),
    )
    for stmt, required in STAGE7_INDEXES:
        try:
            c.execute(stmt)
        except Exception as e:
            if required:
                # Load-bearing index — fail loud, don't degrade silently.
                raise RuntimeError(
                    f"Required Stage-7 index creation failed: {e}. "
                    f"Statement: {stmt[:120]}..."
                ) from e
            log.warning(
                "Stage 7 optional partial index skipped (backend may lack "
                "partial-index support): %s — error: %s", stmt[:80], e
            )

    # ── Backfill (idempotent, sentinel-gated) ────────────────────────
    # 1. Generate public_id for legacy rows missing one.
    import secrets as _secrets
    rows = c.execute("SELECT id FROM api_keys WHERE public_id IS NULL").fetchall()
    for row in rows:
        rid = row[0] if not hasattr(row, '__getitem__') or isinstance(row, tuple) else row[0]
        for attempt in range(8):
            try:
                c.execute(
                    "UPDATE api_keys SET public_id = ? WHERE id = ?",
                    ("k_" + _secrets.token_hex(8), rid)
                )
                break
            except sqlite3.IntegrityError:
                continue
        else:
            raise RuntimeError(
                f"_migrate_stage7_alter_api_keys: could not assign unique "
                f"public_id for api_keys.id={rid} after 8 attempts"
            )

    # 2. Backfill last_four from token_prefix (best-effort).
    c.execute(
        "UPDATE api_keys SET last_four = substr(token_prefix, -4) "
        "WHERE last_four IS NULL AND token_prefix IS NOT NULL"
    )

    # 3. Legacy keys get the default non-console scope set.
    #    Identified by the still-sentinel status (Step 4 has NOT run yet).
    c.execute(
        "UPDATE api_keys SET scopes = ? "
        "WHERE scopes = '[]' AND status = 'legacy_pending'",
        (_STAGE7_LEGACY_SCOPES,)
    )

    # 4. Legacy keys get enforce_origin_for_non_browser=0 (backward compat
    #    — they predate the column and may be used by CLI without Origin).
    #    PDA-P1-R1-F02/F03/F05 fix: gate on the sentinel status, NOT a
    #    created_at cutoff. Sentinel-gated guarantees:
    #      (a) lexicographic-compare bug eliminated entirely;
    #      (b) the UPDATE is single-shot — once Step 5 reclassifies status,
    #          no row matches and the UPDATE is a no-op on re-run;
    #      (c) new keys minted post-migration with status='active' are
    #          untouched, so their enforce_origin DEFAULT 1 stands.
    c.execute(
        "UPDATE api_keys SET enforce_origin_for_non_browser = 0 "
        "WHERE status = 'legacy_pending'"
    )

    # 5. Reclassify status sentinel: revoked → 'disabled', else 'active'.
    #    This MUST run AFTER the legacy-detection updates above, because
    #    those use status='legacy_pending' as the sentinel marker.
    #
    # Forgiving variant: if this UPDATE would violate a UNIQUE constraint
    # (e.g. a legacy_pending row and an active row share the same name on
    # the same user_id — a real-world data state we've seen on Turso),
    # skip the batch and migrate row-by-row, marking conflicting rows as
    # 'disabled' instead of failing the whole server start. The original
    # batched UPDATE is preserved as the happy path.
    try:
        c.execute(
            "UPDATE api_keys SET status = CASE "
            "  WHEN revoked_at IS NOT NULL THEN 'disabled' "
            "  ELSE 'active' END "
            "WHERE status = 'legacy_pending'"
        )
    except sqlite3.IntegrityError as _ie:
        import logging as _lg
        _lg.getLogger("apin_v2.auth_db").warning(
            "Stage7 migration batch UPDATE hit a UNIQUE constraint "
            "(legacy_pending + existing row share name on same user); "
            "falling back to per-row + conflict→disabled. err=%s", _ie)
        rows = c.execute(
            "SELECT id, user_id, name, revoked_at FROM api_keys "
            "WHERE status = 'legacy_pending'"
        ).fetchall()
        for r in rows:
            d = dict(r)
            new_status = 'disabled' if d.get('revoked_at') else 'active'
            try:
                c.execute(
                    "UPDATE api_keys SET status = ? WHERE id = ?",
                    (new_status, d['id'])
                )
            except sqlite3.IntegrityError:
                # Conflicting name → mark as disabled (safer; loses the
                # `active` claim but keeps the row addressable for hard-
                # delete later by the user via the console).
                try:
                    c.execute(
                        "UPDATE api_keys SET status = 'disabled' WHERE id = ?",
                        (d['id'],)
                    )
                except Exception:
                    pass

    # 6. Set environment for any legacy rows missing it (after the sentinel
    #    is gone — this UPDATE is no-op-on-retry because future rows have
    #    a non-empty environment from the column DEFAULT 'live').
    c.execute(
        "UPDATE api_keys SET environment = 'live' "
        "WHERE environment IS NULL OR environment = ''"
    )

    # 7. legacy_alert_emitted is DELIBERATELY left at its column DEFAULT (0)
    #    for legacy keys. The actual alert emission lives in the alert
    #    helpers (future phase). When those land, the alert emitter sets
    #    the flag to 1 only AFTER the alert successfully fires — that
    #    preserves the crash-recovery idempotency guarantee (PDA-R2-F11):
    #    if the alert helpers crash mid-emit, the flag stays 0 and the
    #    next migration run still emits. VER-P1-F01 + PDA-P1-R1-F02 fix:
    #    do NOT pre-emptively flip the flag here.

    # 8. Post-loop verification — every expected column MUST exist.
    cols = {r[1] for r in c.execute("PRAGMA table_info(api_keys)").fetchall()}
    EXPECTED = {
        "environment", "scopes", "last_four", "ip_allowlist", "origin_allowlist",
        "rate_limit_per_min", "quota_per_day", "expires_at", "created_ip",
        "created_ua", "last_used_ip", "last_used_ua", "request_count",
        "error_count", "status", "predecessor_id", "successor_id",
        "rotation_grace_until", "restore_blocked", "note", "public_id",
        "deleted_at", "enforce_origin_for_non_browser", "legacy_alert_emitted",
    }
    missing = EXPECTED - cols
    if missing:
        raise RuntimeError(
            f"_migrate_stage7_alter_api_keys: incomplete — missing {sorted(missing)}"
        )


def _migrate_stage7_alter_sessions(c) -> None:
    """§6.11 — ADD csrf_token to sessions, backfill existing rows."""
    _add_columns(c, "sessions", [
        ("csrf_token", "TEXT"),
    ])
    # Backfill: every session without a csrf_token gets a fresh 64-hex.
    # randomblob(32)|hex() = 64 lowercase hex chars = 256 bits entropy.
    c.execute(
        "UPDATE sessions SET csrf_token = lower(hex(randomblob(32))) "
        "WHERE csrf_token IS NULL"
    )


def _check_fk_enforcement(c) -> bool:
    """Probe whether ON DELETE CASCADE actually fires on this backend.

    Spec §6.13 (VER-P1-R1-F02): libsql/Turso sometimes silently ignores
    `PRAGMA foreign_keys = ON`. Without enforcement, every `ON DELETE
    CASCADE` clause in our schema becomes a no-op — `delete_user()` leaves
    orphaned api_keys, webhooks, alerts, sessions, etc. That's a GDPR
    right-to-erasure problem (the data persists after the user deletes
    their account).

    This probe inserts a sentinel user + child session, deletes the user,
    and checks whether the session was cascade-deleted. The result is
    cached at module level so subsequent calls are free.

    Returns:
        True  — FK CASCADE is honoured; no application-level cleanup needed
        False — FK CASCADE is NOT honoured; application MUST cascade-delete
                in code (see _APPLICATION_LEVEL_CASCADE flag below)

    Side effect: any sentinel rows left from a prior aborted probe are
    cleaned up regardless of outcome.
    """
    SENTINEL_USER_ID = -999999   # negative — guaranteed not a real user

    def _full_cleanup():
        # PDA-P1-R2-F01: the `trg_user_insert_default_settings` trigger
        # fires on EVERY users INSERT — including the probe's seed row —
        # creating an account_settings(user_id=-999999) row as a side
        # effect. The R1 cleanup only deleted from `sessions` and `users`,
        # leaving the trigger-induced account_settings orphan in place
        # when FK enforcement was off (CASCADE didn't fire). Clean ALL
        # tables touched directly OR via trigger.
        for tbl in ("sessions", "account_settings", "users"):
            try:
                c.execute(f"DELETE FROM {tbl} WHERE user_id = ?",
                          (SENTINEL_USER_ID,))
            except Exception:
                # account_settings might not exist yet on first migration
                # run if probe is called before Stage-7 CREATE TABLEs land
                # (defensive — current code calls probe AFTER tables exist).
                pass
        # users uses id, not user_id
        try:
            c.execute("DELETE FROM users WHERE id = ?", (SENTINEL_USER_ID,))
        except Exception:
            pass

    try:
        # Belt-and-suspenders cleanup in case a previous probe crashed.
        _full_cleanup()

        # Seed the probe.
        c.execute(
            "INSERT INTO users(id, username, display_name, email, "
            "password_hash, mobile_e164, pressed_leaf_seed, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (SENTINEL_USER_ID, '_fk_probe', '_fk_probe',
             '_fk_probe@example.invalid', '_', '+10000000000', 0)
        )
        c.execute(
            "INSERT INTO sessions(user_id, token_hash, created_at, expires_at) "
            "VALUES(?, ?, datetime('now'), datetime('now', '+1 hour'))",
            (SENTINEL_USER_ID, '_fk_probe_hash')
        )

        # Delete the parent.
        c.execute("DELETE FROM users WHERE id = ?", (SENTINEL_USER_ID,))

        # Did the child cascade-delete?
        row = c.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id = ?",
            (SENTINEL_USER_ID,)
        ).fetchone()
        remaining = row[0] if row is not None else 0

        if remaining > 0:
            # FK enforcement is OFF — clean up all orphans manually,
            # including the trigger-induced account_settings row.
            _full_cleanup()
            return False

        # FK enforcement IS on for `sessions` (sessions CASCADE-deleted).
        # The trigger-induced account_settings row may or may not have
        # CASCADE-deleted depending on its own FK CASCADE — defensively
        # clean it up here regardless. (PDA-P1-R2-F01: this DELETE is a
        # no-op when CASCADE fired, harmless when it didn't.)
        try:
            c.execute(
                "DELETE FROM account_settings WHERE user_id = ?",
                (SENTINEL_USER_ID,)
            )
        except Exception:
            pass
        return True
    except Exception as e:
        # If the probe itself raised (e.g. backend lacks INSERT support
        # in this context), assume FK enforcement is broken and require
        # application-level cascade. Log and return False.
        import logging as _l
        _l.getLogger("apin_v2.auth.stage7").warning(
            "_check_fk_enforcement probe failed: %s — assuming FK off", e
        )
        _full_cleanup()
        return False


# Module-level cache. Set by _migrate_stage7_extensions on first migration
# run; consulted by delete_user() and any other code path that needs to
# know whether to manually cascade. None = not yet probed.
_APPLICATION_LEVEL_CASCADE: bool | None = None


def _enforce_fk_or_fail(fk_works: bool) -> None:
    """Production hard-fail gate (VER-P1-R2-F01 / spec §6.13 lines 2619-2636).

    Refuse to start the server in production when FK enforcement is OFF,
    unless the operator has explicitly opted in via `APIN_ALLOW_NO_FK=1`.

    The probe (`_check_fk_enforcement`) tells us WHETHER CASCADE works;
    this function decides WHAT TO DO about it.

    Behaviour matrix:

    | APIN_ENV       | fk_works | APIN_ALLOW_NO_FK | Action                   |
    |----------------|----------|------------------|---------------------------|
    | production     | True     | (any)            | proceed                   |
    | production     | False    | (unset/empty/0)  | raise RuntimeError        |
    | production     | False    | "1"              | proceed + CRITICAL log    |
    | dev / unset    | True     | (any)            | proceed                   |
    | dev / unset    | False    | (any)            | proceed + WARNING log     |

    In dev mode we accept FK-off silently (with a WARNING) so local
    development on backends that don't support CASCADE — e.g. some
    libsql-on-file configurations — doesn't block work. In production
    the cost of an unenforced CASCADE is GDPR liability, so we refuse
    to start unless the operator acknowledges the risk.
    """
    import logging as _l
    import os as _os
    log = _l.getLogger("apin_v2.auth.stage7")

    if fk_works:
        return   # No-op — everything fine.

    env = (_os.environ.get("APIN_ENV") or "").strip().lower()
    allow = (_os.environ.get("APIN_ALLOW_NO_FK") or "").strip() == "1"

    if env == "production":
        if not allow:
            raise RuntimeError(
                "FK CASCADE enforcement is DISABLED on this backend "
                "(probe at module import returned False). Refusing to "
                "start in production mode because cascading deletes of "
                "user data (sessions, api_keys, webhooks, audit rows) "
                "would silently leave orphans — a GDPR right-to-erasure "
                "liability. To override (e.g. for a backend you know "
                "handles cascade in application code), set "
                "APIN_ALLOW_NO_FK=1 in the environment. See spec §6.13."
            )
        # Opted in — log at CRITICAL so it cannot be missed in alerting.
        log.critical(
            "FK CASCADE enforcement is OFF in PRODUCTION; "
            "APIN_ALLOW_NO_FK=1 acknowledges this. All deletions of "
            "user data MUST cascade in application code; check "
            "_APPLICATION_LEVEL_CASCADE before relying on FK CASCADE."
        )
        return

    # Dev / staging / unset env — log at WARNING (already done at the
    # caller, but cite it here for completeness).
    log.warning(
        "FK CASCADE enforcement is OFF on this backend (APIN_ENV=%r); "
        "dev mode tolerates this but production will refuse to start "
        "unless APIN_ALLOW_NO_FK=1 is set.", env or "(unset)"
    )


def _migrate_stage7_extensions(c) -> None:
    """Phase-1 schema landing — call after _migrate_v2_extensions.

    Run order is critical: ALTERs first so public_id exists, then the new
    tables that FK to it. Per-step verification raises RuntimeError on
    incomplete migration (PDA-P0-R1-F04 enforcement pattern).
    """
    global _APPLICATION_LEVEL_CASCADE

    # 1. ALTER api_keys (24 cols + indexes + backfill).
    _migrate_stage7_alter_api_keys(c)
    # 2. ALTER sessions (csrf_token + backfill).
    _migrate_stage7_alter_sessions(c)
    # 3. CREATE the 8 new tables / 11 indexes / 2 triggers.
    c.executescript(STAGE7_SCHEMA_SQL)
    # 4. Backfill account_settings for any pre-existing users that predate
    #    the trigger. INSERT OR IGNORE keeps it idempotent.
    c.execute(
        "INSERT OR IGNORE INTO account_settings(user_id, created_at, updated_at) "
        "SELECT id, datetime('now'), datetime('now') FROM users"
    )
    # 4.5 Phase 8.G · session TTL columns on account_settings.
    #     idempotent + additive — existing rows get NULL which means
    #     "use the global default" in get_session_ttls().
    _add_columns(c, "account_settings", [
        ("session_absolute_ttl_seconds",
            "INTEGER"),
        ("session_idle_ttl_seconds",
            "INTEGER"),
        ("remember_me_ttl_seconds",
            "INTEGER"),
    ])
    # 4.6 Phase 8.H · notification preferences. ONE JSON column rather
    #     than 30+ booleans — easy to extend with new alert codes, easy
    #     for the prefs UI to round-trip. Shape:
    #       {"categories": {"key_lifecycle": true, ...},
    #        "codes":      {"key.created": false, ...}}   ← per-code overrides
    #     NULL → use code defaults from _NOTIFY_DEFAULTS in this module.
    _add_columns(c, "account_settings", [
        ("notify_prefs_json", "TEXT"),
    ])
    # 5. Probe FK enforcement and cache the result for delete_user()
    #    and other cascade-dependent code paths (VER-P1-R1-F02 / §6.13).
    #    The probe is idempotent — repeated calls are safe and return the
    #    same answer for a given backend (it's a backend property, not a
    #    per-call state).
    fk_works = _check_fk_enforcement(c)
    _APPLICATION_LEVEL_CASCADE = not fk_works

    # 6. Production hard-fail gate (VER-P1-R2-F01 / §6.13 lines 2619-2636).
    #    Refuse to start in production when FK is off, unless the
    #    operator has explicitly opted in via APIN_ALLOW_NO_FK=1. Dev
    #    mode tolerates FK-off with a WARNING.
    _enforce_fk_or_fail(fk_works)


def _ensure_db():
    """Apply the schema + idempotent migrations. Idempotent across both
    backends — `CREATE TABLE IF NOT EXISTS` and the introspected
    ADD COLUMN make re-running safe.

    Turso path: wraps the whole init in a top-level retry. A flaky
    Turso window can hit any one of the dozens of queries fired during
    init; if the inner per-query retry burns through its budget, we
    take a longer breath here and try the entire sequence again. This
    is safe because every statement in the init is idempotent."""
    if _USE_TURSO:
        # Turso/libSQL — run each schema statement, then migrate.
        # journal_mode=WAL is a no-op on a remote DB, so it is skipped.
        import time as _time
        last_err = None
        # Top-level retry windows: 0s (first try), then 5s, 15s, 30s
        # between attempts. Gives Turso up to ~50s total to clear.
        for attempt, pre_wait in enumerate((0, 5, 15, 30)):
            if pre_wait > 0:
                import logging as _lg
                _lg.getLogger("apin_v2.auth_db").warning(
                    "Turso init attempt %d failed, retrying in %ds...",
                    attempt, pre_wait
                )
                _time.sleep(pre_wait)
            try:
                _libsql_conn.executescript(SCHEMA_SQL)
                try:
                    _libsql_conn.execute("PRAGMA foreign_keys = ON")
                except Exception:
                    pass   # not all libSQL deployments honour this PRAGMA
                _migrate_predictions_blob_columns(_libsql_conn)
                _migrate_v2_extensions(_libsql_conn)
                _migrate_stage7_extensions(_libsql_conn)   # API Console — Phase 1
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            msg = str(last_err)
            hint = ""
            if any(s in msg for s in ("JWT", "nauthorized", "401", "400")):
                hint = ("\n  -> Turso rejected the credentials. Check the "
                        "TURSO_AUTH_TOKEN secret — it must be a current, "
                        "non-expired token for this database (regenerate "
                        "with: turso db tokens create <db-name>).")
            raise RuntimeError(
                f"Could not initialise the Turso database at "
                f"{_TURSO_URL!r} after 4 attempts: {msg}{hint}"
            ) from last_err
    else:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as c:
            c.executescript(SCHEMA_SQL)
            c.execute("PRAGMA journal_mode = WAL;")  # better concurrency
            c.execute("PRAGMA foreign_keys = ON;")
            _migrate_predictions_blob_columns(c)
            _migrate_v2_extensions(c)
            _migrate_stage7_extensions(c)              # API Console — Phase 1
            c.commit()


_ensure_db()


@contextmanager
def get_conn():
    """Yield a database connection.

    Turso mode  → the single shared libSQL shim connection.
    Local mode  → a fresh stdlib sqlite3 connection (row_factory=Row,
                  foreign keys on), closed when the block exits.
    Either way the caller sees the same .execute / .fetchone / dict(row)
    surface, so every helper in this module is backend-agnostic.
    """
    if _USE_TURSO:
        yield _libsql_conn
        return
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        isolation_level=None,  # autocommit; we'll BEGIN explicitly
        # Stage 2.5 [PDA-3 C1] · default busy_timeout is 0 ms which means
        # SQLite returns SQLITE_BUSY (or, in WAL mode, occasionally
        # "attempt to write a readonly database") immediately on transient
        # write-lock contention. Under 30+ concurrent telemetry POSTs we
        # observed ~3% page_view loss because of this. 5000 ms gives the
        # filesystem lock plenty of time to settle without making the
        # request latency visibly worse.
        timeout=5.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    try:
        yield conn
    finally:
        conn.close()


# ─── Time helpers ─────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """UTC now in ISO 8601 (used for created_at, expires_at, etc.)."""
    return datetime.now(timezone.utc).isoformat()


# ─── Password hashing ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Return argon2id encoded hash string (contains salt + params)."""
    return _ph.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    """Verify password against stored hash. Returns True on match."""
    try:
        _ph.verify(encoded, password)
        return True
    except (VerifyMismatchError, InvalidHashError):
        return False


# ─── Token helpers ────────────────────────────────────────────────────────────

def _new_session_token() -> tuple[str, str]:
    """Generate (raw_token, sha256(token_hash)) pair.

    The raw token is sent to the client as the session cookie value.
    The hash is what we store in the DB so a DB read-only leak cannot
    impersonate users.
    """
    raw = secrets.token_urlsafe(32)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return raw, h


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ─── User helpers ─────────────────────────────────────────────────────────────

def is_taken(field: str, value: str) -> bool:
    """Case-insensitive existence check on username / display_name / email."""
    if field not in {"username", "display_name", "email"}:
        return False
    v = value.strip()
    if not v:
        return False
    with get_conn() as c:
        sql = f"SELECT 1 FROM users WHERE {field} = ? COLLATE NOCASE LIMIT 1"
        row = c.execute(sql, (v,)).fetchone()
        return row is not None


def next_accession() -> int:
    """Return what the next-created user's id will be (max(id)+1)."""
    with get_conn() as c:
        row = c.execute("SELECT IFNULL(MAX(id), 0) AS m FROM users").fetchone()
        return int(row["m"]) + 1


def create_user(
    *,
    username: str,
    display_name: str,
    email: str,
    password: str,
    mobile_e164: str,
    pressed_leaf_seed: Optional[int] = None,
) -> dict:
    """Create a user row. Raises ValueError("taken:<field>") on UNIQUE violation."""
    import random as _r
    if pressed_leaf_seed is None:
        pressed_leaf_seed = _r.randint(0, 5)

    pw_hash = hash_password(password)
    now = _now_iso()

    with _write_lock, get_conn() as c:
        try:
            cur = c.execute(
                """INSERT INTO users
                   (username, display_name, email, password_hash, mobile_e164,
                    pressed_leaf_seed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (username.strip(), display_name.strip(), email.strip().lower(),
                 pw_hash, mobile_e164, pressed_leaf_seed, now),
            )
            new_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "users.username" in msg:
                raise ValueError("taken:username")
            if "users.display_name" in msg:
                raise ValueError("taken:display_name")
            if "users.email" in msg:
                raise ValueError("taken:email")
            raise

    return get_user_by_id(new_id)


def get_user_by_id(uid: int) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        return dict(row) if row else None


def get_user_by_handle(handle: str) -> Optional[dict]:
    """Find user by username OR email (case-insensitive)."""
    h = handle.strip()
    if not h:
        return None
    with get_conn() as c:
        # Try email first if it looks like one (contains @), else username
        if "@" in h:
            row = c.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE LIMIT 1",
                (h,),
            ).fetchone()
            if row:
                return dict(row)
        row = c.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE LIMIT 1",
            (h,),
        ).fetchone()
        return dict(row) if row else None


def touch_last_seen(user_id: int):
    with _write_lock, get_conn() as c:
        c.execute("UPDATE users SET last_seen_at = ? WHERE id = ?",
                  (_now_iso(), user_id))


# ─── Session helpers ──────────────────────────────────────────────────────────

def create_session(
    user_id: int,
    *,
    user_agent: Optional[str] = None,
    ip_addr: Optional[str] = None,
) -> str:
    """Create a session row, return the raw token (to be sent as cookie value).

    WI-P4-CSRF fix: seed `csrf_token` at row-insert time. Previously the
    column was added by §6.11 migration and BACKFILLED for existing rows
    (`UPDATE sessions SET csrf_token = lower(hex(randomblob(32))) WHERE
    csrf_token IS NULL`), but fresh inserts left it NULL until something
    rotated it. PVA-P3.3-3.4-R1 caught this latent gap. Real CSRF check
    in `_require_csrf` (Phase 4) compares header against this row value,
    so NULL would always reject — every freshly-logged-in user would fail
    their first mutation.

    The seeded token uses the same 32-byte base64url shape as
    `rotate_session_csrf_token` so any downstream code that consumes the
    field has a uniform format.
    """
    raw, h = _new_session_token()
    csrf_initial = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = (now + SESSION_LIFETIME).isoformat()
    with _write_lock, get_conn() as c:
        c.execute(
            """INSERT INTO sessions
               (user_id, token_hash, user_agent, ip_addr,
                created_at, expires_at, csrf_token)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, h, user_agent, ip_addr,
             now.isoformat(), expires, csrf_initial),
        )
    return raw


def get_session_user(raw_token: str) -> Optional[dict]:
    """Resolve a raw session token to the underlying user, or None if invalid."""
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    now = _now_iso()
    with get_conn() as c:
        row = c.execute(
            """SELECT u.* FROM sessions s
               JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = ?
                 AND s.revoked_at IS NULL
                 AND s.expires_at > ?
               LIMIT 1""",
            (h, now),
        ).fetchone()
        return dict(row) if row else None


def rotate_session_csrf_token(session_id: str) -> Optional[str]:
    """Generate a fresh csrf_token for the session and return the raw value.

    Spec §7.6 PDA-F44: rotate session.csrf_token on sudo_started so the
    pre-sudo CSRF token can't be replayed. Returns the new token, or None
    if no session row matched.

    FX-P4-7 (VER-P4-R1 C8): emit canonical `csrf_rotated` audit event per
    spec §12.1 enum. Audit emission is non-fatal (best-effort) — a DB
    glitch on audit must not block the CSRF rotation itself.
    """
    if not session_id:
        return None
    new_token = secrets.token_urlsafe(32)
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "UPDATE sessions SET csrf_token = ? WHERE id = ?",
            (new_token, session_id),
        )
        if cur.rowcount == 0:
            return None
    # FX-P4-7: emit canonical audit row
    try:
        audit("csrf_rotated", detail={"session_id": str(session_id)})
    except Exception:
        pass
    return new_token


# ── §6.10 account_settings helpers (WI-P4-ACCT-SETTINGS) ────────────────

# Defaults match the schema DEFAULTs at line 1834-1862. Used when a user's
# settings row is missing (shouldn't happen — the trg_user_insert_default_settings
# trigger backfills on every INSERT — but defensive fallback for tests that
# mock users without going through the trigger).
_ACCOUNT_SETTINGS_DEFAULTS = {
    "sudo_session_length_seconds": 300,
    "sudo_max_uses": 50,
    "max_webhooks_per_user": 50,
    "request_log_retention_days": 30,
    "audit_log_retention_days": 365,
    "webhook_delivery_retention_days": 90,
    "require_ip_allowlist": 0,
    "notify_on_key_created": 1,
    "notify_on_first_use_from_new_ip": 1,
    "notify_on_quota_exceeded": 1,
    "default_scope_template": "inference_only",
}


def get_account_settings(user_id: int) -> dict:
    """Return the account_settings row for `user_id` as a dict, or schema
    defaults if no row exists.

    WI-P4-ACCT-SETTINGS: replaces hardcoded constants throughout the codebase
    (e.g. routes_sudo.py's `_SUDO_TTL_SECONDS = 300`, the future webhook cap,
    etc.). Each caller reads the field it needs from the returned dict.

    The CHECK constraints on the schema columns guarantee values are within
    spec-mandated ranges, so callers don't need to re-validate.
    """
    with get_conn() as c:
        row = c.execute(
            "SELECT * FROM account_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        d = dict(_ACCOUNT_SETTINGS_DEFAULTS)
    else:
        d = dict(row)
        for k, v in _ACCOUNT_SETTINGS_DEFAULTS.items():
            if d.get(k) is None:
                d[k] = v
    # FX-P7-QA1: parse JSON-encoded fields so callers get real lists.
    # sudo_required_for is stored as TEXT containing a JSON array (spec §6.10).
    # Pre-fix, the client received the raw string and iterated it per-char,
    # producing 126-element char sets instead of 8-element action lists.
    raw = d.get("sudo_required_for")
    if isinstance(raw, str):
        try:
            d["sudo_required_for"] = json.loads(raw)
        except Exception:
            d["sudo_required_for"] = []
    return d


def update_account_settings(user_id: int, **fields) -> dict:
    """Update editable fields in account_settings. Returns the full
    updated row as a dict (or raises ValueError on out-of-range value
    that the schema CHECK would reject — we pre-validate to give a
    cleaner error than a bare SQLite IntegrityError).

    Phase 6.B.1 (WI-P5+-SETTINGS-UI). Allowed fields match the schema
    at §6.10. Unknown fields are silently dropped.
    """
    # Spec §6.10 CHECK ranges. Phase 7 Wave 2: expanded from 12 to all
    # 20 editable fields to support the industry-grade Settings page.
    _BOOL_FIELDS = {
        # Defaults for new keys
        "require_ip_allowlist",
        # Notifications
        "notify_on_key_created",
        "notify_on_first_use_from_new_ip",
        "notify_on_quota_exceeded",
        # Privacy & hardening
        "ip_truncation_enabled",
        "auto_revoke_on_partner_detection",
    }
    _INT_RANGES = {
        # Sudo
        "sudo_session_length_seconds":   (60, 900),
        "sudo_max_uses":                 (10, 200),
        # Defaults for new keys
        "default_key_expiry_days":       (0, 3650),
        # Caps
        "max_webhooks_per_user":         (1, 200),
        # Retention windows (days)
        "request_log_retention_days":    (1, 90),
        "audit_log_retention_days":      (30, 1825),
        # FX-P7-F02: align with schema CHECK (BETWEEN 7 AND 365). A user
        # PATCH of 1-6 would have raised raw sqlite3.IntegrityError → 500.
        "webhook_delivery_retention_days": (7, 365),
        # Org-wide rate / quota envelope
        "org_aggregate_rate_limit":      (60, 100_000),
        "org_aggregate_quota_per_day":   (1, 10_000_000),
        # Sandbox limits
        "sandbox_rate_per_min":          (1, 10_000),
        "sandbox_max_upload_mb_per_min": (1, 1000),
    }
    # Spec §9.2: 8 valid action classes for sudo_required_for.
    _SUDO_ACTION_CLASSES = {
        "create_key", "revoke_key", "rotate_key",
        "change_scope", "change_allowlist",
        "create_webhook", "revoke_webhook", "change_settings",
    }
    _SCOPE_TEMPLATES = {"inference_only", "read_only",
                        "full_inference", "webhooks"}
    # notify_on_auth_failures_threshold is nullable: int >= 1 enables; None
    # disables. Handled as a special case below (NOT in _INT_RANGES).
    _NULLABLE_INT_RANGES = {
        "notify_on_auth_failures_threshold": (1, 100),
    }

    sets, args = [], []
    for k, v in fields.items():
        if k in _BOOL_FIELDS:
            sets.append(f"{k} = ?")
            args.append(1 if v else 0)
        elif k in _INT_RANGES:
            lo, hi = _INT_RANGES[k]
            try:
                vi = int(v)
            except Exception:
                raise ValueError(f"{k} must be an integer")
            if not (lo <= vi <= hi):
                raise ValueError(f"{k} must be in [{lo}, {hi}]")
            sets.append(f"{k} = ?")
            args.append(vi)
        elif k in _NULLABLE_INT_RANGES:
            # null / None disables the feature; an int in range enables.
            if v is None:
                sets.append(f"{k} = ?"); args.append(None)
            else:
                lo, hi = _NULLABLE_INT_RANGES[k]
                try:
                    vi = int(v)
                except Exception:
                    raise ValueError(f"{k} must be an integer or null")
                if not (lo <= vi <= hi):
                    raise ValueError(f"{k} must be in [{lo}, {hi}] or null")
                sets.append(f"{k} = ?"); args.append(vi)
        elif k == "default_scope_template":
            if v not in _SCOPE_TEMPLATES:
                raise ValueError(
                    f"default_scope_template must be one of {sorted(_SCOPE_TEMPLATES)}")
            sets.append("default_scope_template = ?")
            args.append(v)
        elif k == "sudo_required_for":
            # Spec §6.10: stored as JSON; spec §9.2: 8 valid action classes.
            if not isinstance(v, list):
                raise ValueError("sudo_required_for must be a list of action-class strings")
            normalized = []
            for action in v:
                if not isinstance(action, str):
                    raise ValueError("sudo_required_for items must be strings")
                if action not in _SUDO_ACTION_CLASSES:
                    raise ValueError(
                        f"invalid action class '{action}'. "
                        f"Valid: {sorted(_SUDO_ACTION_CLASSES)}")
                if action not in normalized:
                    normalized.append(action)
            # Note: this allows REMOVING sudo from any action — the user is
            # trusting their own session. UI should warn before saving an
            # empty list (effective: no sudo gate anywhere).
            sets.append("sudo_required_for = ?")
            args.append(json.dumps(normalized))
        elif k == "notify_prefs_json":
            # Phase 8.H.D · the alert prefs JSON column. Caller is
            # expected to have already serialised + validated the shape
            # (routes_alerts.py::alert_prefs_patch does this). Accept any
            # JSON string here, or NULL to clear.
            if v is None:
                sets.append("notify_prefs_json = ?"); args.append(None)
            elif isinstance(v, str):
                try:
                    json.loads(v)   # validate it parses
                except Exception:
                    raise ValueError("notify_prefs_json must be valid JSON")
                sets.append("notify_prefs_json = ?"); args.append(v)
            elif isinstance(v, dict):
                sets.append("notify_prefs_json = ?")
                args.append(json.dumps(v))
            else:
                raise ValueError("notify_prefs_json must be a dict or JSON string")
        # Silently drop unknown fields (defensive)

    if sets:
        # Phase 6 FX-P6-3: ensure the account_settings row exists before
        # UPDATE. The trg_user_insert_default_settings trigger created
        # rows for users registered AFTER the trigger was added (PDA-R2-F12),
        # but legacy users may have no row. Without this INSERT-OR-IGNORE
        # the UPDATE would match zero rows and the caller would receive
        # the defaults dict back, falsely indicating the save succeeded.
        now = _now_iso()
        args.extend([now, int(user_id)])
        with _write_lock, get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO account_settings "
                "(user_id, created_at, updated_at) VALUES (?, ?, ?)",
                (int(user_id), now, now),
            )
            c.execute(
                f"UPDATE account_settings SET {', '.join(sets)}, "
                f"updated_at = ? WHERE user_id = ?",
                args,
            )
    return get_account_settings(int(user_id))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 Wave 1 — Webhook + Alert helpers
#
# Spec refs:
#   §6.5  webhooks table
#   §6.6  webhook_deliveries table
#   §6.7  api_key_alerts table
#   §7.4  webhook endpoints
#   §13.x webhook signing + delivery worker (delivery worker is Phase 8;
#         this module provides synchronous test-ping + enqueue helpers)
#   §18.11 AES-GCM secret encryption ceremony
#
# Scope honesty: the FULL spec includes a background delivery worker with
# retry-with-backoff (1m, 5m, 25m, 2h, 12h, 24h, ...), dead-letter handling,
# multi-secret overlap during rotation, exactly-once delivery via claimed_at
# leasing, and HMAC signing with rolled secrets. Phase 7 implements:
#   - Secret generation + AES-GCM encryption at rest (spec §18.11)
#   - HMAC-SHA256 signature computation (spec §13.2)
#   - Synchronous test-ping (single POST, no retry)
#   - CRUD on webhooks + alerts
#   - Delivery enqueue + log query (UI surface for "what has been queued")
# The actual delivery worker is filed as WI-P8-DELIVERY-WORKER.
# ═══════════════════════════════════════════════════════════════════════════

import hmac as _hmac
import hashlib as _hashlib
import secrets as _secrets


def _get_apin_secret_key() -> bytes:
    """Return the 32-byte AES-GCM key from `APIN_SECRET_KEY` env var.

    Spec §18.11.1: this env var is REQUIRED even in development for any
    webhook endpoint. Boot-time should log CRITICAL if unset. At request
    time, callers should catch RuntimeError and return 503
    `service_unavailable` with the documented hint.
    """
    raw = os.environ.get("APIN_SECRET_KEY")
    if not raw:
        raise RuntimeError(
            "APIN_SECRET_KEY env var is required for webhook secret "
            "encryption. Generate with `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"` and place in your .env."
        )
    # Accept either raw 32-byte hex/b64 or any string → SHA-256 to 32 bytes
    try:
        if len(raw) == 64:
            return bytes.fromhex(raw)
    except ValueError:
        pass
    # Derive 32-byte key via SHA-256 (deterministic; safe for KDF role here
    # because the env var is already a high-entropy secret per spec §18.11).
    return _hashlib.sha256(raw.encode("utf-8")).digest()


def encrypt_webhook_secret(plaintext: bytes, webhook_id: str) -> bytes:
    """AES-GCM encrypt a webhook secret per spec §18.11.

    Wire format: b'\\x01' || nonce(12) || ciphertext || tag(16)
    AAD = webhook_id (UTF-8 bytes) — binds the ciphertext to the row so
    swapping secret_encrypted between rows fails decryption.

    Returns the wire-format BLOB suitable for the `secret_encrypted` column.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = _get_apin_secret_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    aad = webhook_id.encode("utf-8")
    ct_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
    return b"\x01" + nonce + ct_and_tag


def decrypt_webhook_secret(blob: bytes, webhook_id: str) -> bytes:
    """Inverse of encrypt_webhook_secret. Returns plaintext secret bytes.

    Raises ValueError on version-byte mismatch (future-proof for v2 format)
    or cryptography.exceptions.InvalidTag on tamper / wrong AAD / wrong key.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not blob or blob[0:1] != b"\x01":
        raise ValueError("webhook secret blob version mismatch")
    nonce = blob[1:13]
    ct_and_tag = blob[13:]
    key = _get_apin_secret_key()
    aesgcm = AESGCM(key)
    aad = webhook_id.encode("utf-8")
    return aesgcm.decrypt(nonce, ct_and_tag, aad)


def _generate_webhook_secret() -> tuple[bytes, str]:
    """Spec §4.2 + §6.5 (PDA-R2-F54): secret = `whsec_` + 32 base62 chars
    from 24 random bytes. Returns (plaintext_bytes, last4_for_display).

    The `whsec_` prefix is part of the plaintext that the customer sees and
    that signs the webhook bodies. last4 is for the redacted listing UI.
    """
    raw = _secrets.token_bytes(24)
    b62_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # Convert 24 bytes to a base-62 string of exactly 32 chars
    n = int.from_bytes(raw, "big")
    chars: list[str] = []
    while n:
        n, rem = divmod(n, 62)
        chars.append(b62_alphabet[rem])
    s = "".join(reversed(chars)).rjust(32, "0")[-32:]
    plaintext = ("whsec_" + s).encode("utf-8")
    last4 = s[-4:]
    return plaintext, last4


def compute_webhook_signature(secret: bytes, timestamp: str,
                              event_id: str, body: bytes) -> str:
    """HMAC-SHA256 signature per spec §13.2.

    Signed bytes: `t.event_id.body` where `t` is the integer Unix timestamp
    as ASCII, then `.`, then event_id, then `.`, then the raw body bytes
    WITHOUT a trailing newline (server enforces this on _insert_delivery_row).

    Returns the hex digest (lowercase) for the `v1=` portion of the
    `APIN-Signature` header.
    """
    msg = (timestamp + "." + event_id + ".").encode("ascii") + body
    return _hmac.new(secret, msg, _hashlib.sha256).hexdigest()


# ── Webhook CRUD ──────────────────────────────────────────────────────────

# Reasonable per-spec defaults — spec §13.x events list is large; we accept
# any string here and validate against the catalogue only at the route layer.
_WEBHOOK_MAX_EVENTS = 50
_WEBHOOK_DEFAULT_CAP = 50   # account_settings.max_webhooks_per_user default


def list_webhooks(user_id: int) -> list[dict]:
    """List all webhooks for a user. Excludes encrypted secret BLOBs."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, url, events, description, active, created_at, "
            "       updated_at, last_delivery_at, last_delivery_status, "
            "       consecutive_failure_count, consecutive_gave_up_count, "
            "       auto_disabled_at, allow_self_signed, allow_internal_target "
            "FROM webhooks WHERE user_id = ? ORDER BY created_at DESC",
            (int(user_id),),
        ).fetchall()
    return [_webhook_row_to_dict(r) for r in rows]


def _webhook_row_to_dict(row) -> dict:
    d = dict(row)
    # Parse events JSON list
    try:
        d["events"] = json.loads(d.get("events") or "[]")
    except Exception:
        d["events"] = []
    # Coerce booleans
    for k in ("active", "allow_self_signed", "allow_internal_target"):
        if k in d:
            d[k] = bool(d[k])
    return d


def count_user_webhooks(user_id: int) -> int:
    with get_conn() as c:
        return int(c.execute(
            "SELECT COUNT(*) AS n FROM webhooks WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()["n"])


def get_webhook(webhook_id: str, user_id: int) -> Optional[dict]:
    """Returns the webhook dict or None if not found / not owned by user."""
    with get_conn() as c:
        row = c.execute(
            "SELECT id, url, events, description, active, created_at, "
            "       updated_at, last_delivery_at, last_delivery_status, "
            "       consecutive_failure_count, consecutive_gave_up_count, "
            "       auto_disabled_at, allow_self_signed, allow_internal_target "
            "FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, int(user_id)),
        ).fetchone()
    return _webhook_row_to_dict(row) if row else None


def create_webhook(
    user_id: int,
    url: str,
    events: list[str],
    *,
    description: Optional[str] = None,
    allow_self_signed: bool = False,
    allow_internal_target: bool = False,
) -> tuple[dict, str]:
    """Mint a new webhook + generate + encrypt its signing secret.

    Returns (webhook_dict, plaintext_secret_string). The plaintext is shown
    ONCE in the UI; subsequent listings show only `whsec_******_<last4>`.
    Raises ValueError on validation failures (URL too long, too many events,
    invalid events list shape, over-cap).
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url is required")
    if len(url) > 2048:
        raise ValueError("url must be <= 2048 chars")
    if not isinstance(events, list) or not events:
        raise ValueError("events must be a non-empty list of strings")
    if len(events) > _WEBHOOK_MAX_EVENTS:
        raise ValueError(f"too many events (max {_WEBHOOK_MAX_EVENTS})")
    for e in events:
        if not isinstance(e, str) or not e:
            raise ValueError("each event must be a non-empty string")
    if description is not None and len(str(description)) > 280:
        raise ValueError("description must be <= 280 chars")

    # Per-user cap (spec §6.10 max_webhooks_per_user, default 50)
    cap = int(get_account_settings(int(user_id)).get(
        "max_webhooks_per_user", _WEBHOOK_DEFAULT_CAP))
    if count_user_webhooks(int(user_id)) >= cap:
        raise ValueError(
            f"webhook cap reached ({cap}). Delete an existing webhook first.")

    # Generate secret + ID
    webhook_id = "wh_" + _new_id()
    plaintext, _last4 = _generate_webhook_secret()
    secret_hash = _hashlib.sha256(plaintext).hexdigest()
    secret_encrypted = encrypt_webhook_secret(plaintext, webhook_id)

    now = _now_iso()
    events_json = json.dumps(sorted(set(events)))

    with _write_lock, get_conn() as c:
        c.execute(
            "INSERT INTO webhooks "
            "(id, user_id, url, secret_hash, secret_encrypted, events, "
            " description, active, created_at, updated_at, "
            " allow_self_signed, allow_internal_target) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
            (webhook_id, int(user_id), url, secret_hash, secret_encrypted,
             events_json, description, now, now,
             1 if allow_self_signed else 0,
             1 if allow_internal_target else 0),
        )

    wh = get_webhook(webhook_id, int(user_id))
    return wh, plaintext.decode("utf-8")


def update_webhook(webhook_id: str, user_id: int, **fields) -> dict:
    """Update a subset of editable webhook fields. Returns updated dict.
    Editable: url, events, description, active, allow_self_signed,
    allow_internal_target. Raises ValueError on validation failure.
    """
    if get_webhook(webhook_id, int(user_id)) is None:
        raise ValueError("webhook not found")

    sets: list[str] = []
    args: list = []
    if "url" in fields:
        url = fields["url"]
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        if len(url) > 2048:
            raise ValueError("url must be <= 2048 chars")
        sets.append("url = ?"); args.append(url)
    if "events" in fields:
        ev = fields["events"]
        if not isinstance(ev, list) or not ev:
            raise ValueError("events must be a non-empty list")
        if len(ev) > _WEBHOOK_MAX_EVENTS:
            raise ValueError(f"too many events (max {_WEBHOOK_MAX_EVENTS})")
        for e in ev:
            if not isinstance(e, str) or not e:
                raise ValueError("each event must be a non-empty string")
        sets.append("events = ?"); args.append(json.dumps(sorted(set(ev))))
    if "description" in fields:
        d = fields["description"]
        if d is not None and len(str(d)) > 280:
            raise ValueError("description must be <= 280 chars")
        sets.append("description = ?"); args.append(d)
    if "active" in fields:
        sets.append("active = ?"); args.append(1 if fields["active"] else 0)
    if "allow_self_signed" in fields:
        sets.append("allow_self_signed = ?")
        args.append(1 if fields["allow_self_signed"] else 0)
    if "allow_internal_target" in fields:
        sets.append("allow_internal_target = ?")
        args.append(1 if fields["allow_internal_target"] else 0)

    if not sets:
        return get_webhook(webhook_id, int(user_id))

    args.extend([_now_iso(), webhook_id, int(user_id)])
    with _write_lock, get_conn() as c:
        c.execute(
            f"UPDATE webhooks SET {', '.join(sets)}, updated_at = ? "
            f"WHERE id = ? AND user_id = ?",
            args,
        )
    return get_webhook(webhook_id, int(user_id))


def delete_webhook(webhook_id: str, user_id: int) -> bool:
    """Hard-delete a webhook. Cascades to webhook_deliveries via FK.
    Returns True if deleted, False if not found."""
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "DELETE FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, int(user_id)),
        )
        return cur.rowcount > 0


def rotate_webhook_secret(webhook_id: str, user_id: int,
                          grace_seconds: int = 86400) -> tuple[dict, str]:
    """Generate a new secret, keep old one valid for `grace_seconds` so
    receivers can swap in the new secret without dropping in-flight events.

    Returns (webhook_dict, new_plaintext_secret). Grace window is enforced
    by the delivery worker; this helper only sets the columns.
    """
    if grace_seconds < 0 or grace_seconds > 30 * 86400:
        raise ValueError("grace_seconds must be in [0, 30 days]")
    wh = get_webhook(webhook_id, int(user_id))
    if wh is None:
        raise ValueError("webhook not found")

    new_plaintext, _last4 = _generate_webhook_secret()
    new_hash = _hashlib.sha256(new_plaintext).hexdigest()
    new_encrypted = encrypt_webhook_secret(new_plaintext, webhook_id)

    grace_until = datetime.now(timezone.utc).replace(microsecond=0)
    grace_until_iso = (grace_until + timedelta(seconds=grace_seconds)).isoformat() \
        if grace_seconds > 0 else None

    with _write_lock, get_conn() as c:
        # Move current secret to *_old slot
        c.execute(
            "UPDATE webhooks SET "
            "  secret_encrypted_old = secret_encrypted, "
            "  secret_old_until = ?, "
            "  secret_encrypted = ?, "
            "  secret_hash = ?, "
            "  updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (grace_until_iso, new_encrypted, new_hash, _now_iso(),
             webhook_id, int(user_id)),
        )

    return get_webhook(webhook_id, int(user_id)), new_plaintext.decode("utf-8")


def list_webhook_deliveries(webhook_id: str, user_id: int,
                             limit: int = 50) -> list[dict]:
    """Recent delivery rows for a webhook. Ordered newest first.
    Strips claimed_by / response_body to control payload size."""
    if get_webhook(webhook_id, int(user_id)) is None:
        return []
    limit = max(1, min(int(limit), 200))
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, event_id, event_type, queued_at, first_attempt_at, "
            "       last_attempt_at, attempts, status, response_status, "
            "       response_truncated, error_text "
            "FROM webhook_deliveries WHERE webhook_id = ? "
            "ORDER BY queued_at DESC LIMIT ?",
            (webhook_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def enqueue_webhook_delivery(webhook_id: str, event_type: str,
                              payload: dict) -> int:
    """Insert a webhook_deliveries row in 'queued' state.

    Spec §13.6: body MUST NOT end in '\\n' (server-side assertion). We
    enforce that here by using compact JSON. The actual delivery is the
    worker's job (WI-P8-DELIVERY-WORKER); this helper just builds the row.
    """
    wh_row = None
    with get_conn() as c:
        wh_row = c.execute(
            "SELECT id, secret_encrypted FROM webhooks WHERE id = ?",
            (webhook_id,),
        ).fetchone()
    if wh_row is None:
        raise ValueError("webhook not found")

    event_id = "evt_" + _new_id()
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    assert not payload_bytes.endswith(b"\n"), \
        "webhook body MUST NOT end in newline (spec §13.6 PDA-R2-F02)"

    secret_plaintext = decrypt_webhook_secret(wh_row["secret_encrypted"], webhook_id)
    ts = str(int(datetime.now(timezone.utc).timestamp()))
    sig = compute_webhook_signature(secret_plaintext, ts, event_id, payload_bytes)
    sig_header = f"t={ts},v1={sig}"

    now = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "INSERT INTO webhook_deliveries "
            "(webhook_id, event_id, event_type, payload, signature, "
            " queued_at, next_attempt_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (webhook_id, event_id, event_type, payload_bytes.decode("utf-8"),
             sig_header, now, now),
        )
        delivery_id = int(cur.lastrowid)
    return delivery_id


def test_ping_webhook(webhook_id: str, user_id: int,
                       timeout_seconds: float = 10.0) -> dict:
    """Synchronously fire ONE test event at the webhook URL. Updates the
    webhook's last_delivery_* columns. Returns a dict with the attempt
    outcome (status, response_status, error_text, elapsed_ms).

    Spec §7.4 PDA-R2-F46: this endpoint is state-changing (real outbound
    POST) and the route layer enforces sudo. Here we just do the network
    call. No retry — that's the worker's job.
    """
    import time as _time
    wh = get_webhook(webhook_id, int(user_id))
    if wh is None:
        raise ValueError("webhook not found")
    if not wh.get("active"):
        raise ValueError("webhook is disabled; enable it before testing")

    # Build a representative test event
    test_payload = {
        "type": "webhook.test",
        "webhook_id": webhook_id,
        "sent_at": _now_iso(),
        "message": "Test ping from APIN Console. If you can read this, "
                   "your endpoint is reachable and HMAC verification is "
                   "working on this side. Verify the APIN-Signature header.",
    }
    delivery_id = enqueue_webhook_delivery(webhook_id, "webhook.test", test_payload)

    with get_conn() as c:
        d = c.execute(
            "SELECT event_id, payload, signature FROM webhook_deliveries "
            "WHERE id = ?", (delivery_id,),
        ).fetchone()

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "APIN-Webhook/1.0 (synchronous-test-ping)",
        "APIN-Event-Id": d["event_id"],
        "APIN-Delivery-Id": str(delivery_id),
        "APIN-Signature": d["signature"],
        "APIN-Webhook-Id": webhook_id,
    }
    body = d["payload"].encode("utf-8")

    import urllib.request, urllib.error

    # FX-P7-F03: disable redirect-following entirely. Python's default
    # HTTPRedirectHandler follows 301/302/303/307/308 up to 10 hops and does
    # NOT re-check the redirected URL against the original allowlist. A
    # malicious or compromised receiver could redirect APIN to
    # http://169.254.169.254/latest/meta-data/ (AWS IMDS) or any internal
    # service, turning the sudo-gated test-ping into an SSRF primitive.
    # By installing a no-op redirect handler we surface 3xx as the actual
    # response status, and the user sees "your endpoint returned 302; we
    # do not follow redirects on test-ping" — the right behavior anyway
    # because spec §13.7 says webhooks SHOULD respond with 2xx directly.
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # do not follow

    opener = urllib.request.build_opener(_NoRedirect())
    req = urllib.request.Request(wh["url"], data=body, headers=headers,
                                   method="POST")
    start = _time.time()
    err_text = None
    response_status = None
    try:
        with opener.open(req, timeout=timeout_seconds) as resp:
            response_status = int(resp.status)
            _ = resp.read(2048)  # consume to free socket; body not stored here
        if 300 <= response_status < 400:
            outcome = "failed_attempt"
            err_text = (f"receiver returned {response_status} redirect; "
                        "test-ping does not follow redirects "
                        "(SSRF defence). Have your endpoint return 2xx directly.")
        else:
            outcome = "delivered" if 200 <= response_status < 300 else "failed_attempt"
    except urllib.error.HTTPError as e:
        response_status = int(e.code)
        outcome = "failed_attempt"
        err_text = f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        outcome = "failed_attempt"
        err_text = f"URL error: {e.reason}"
    except Exception as e:
        outcome = "failed_attempt"
        err_text = f"{type(e).__name__}: {e}"
    elapsed_ms = int((_time.time() - start) * 1000)

    final_status = "delivered" if outcome == "delivered" else "gave_up"
    now = _now_iso()
    with _write_lock, get_conn() as c:
        c.execute(
            "UPDATE webhook_deliveries SET "
            "  first_attempt_at = ?, last_attempt_at = ?, attempts = 1, "
            "  status = ?, response_status = ?, error_text = ? "
            "WHERE id = ?",
            (now, now, final_status, response_status, err_text, delivery_id),
        )
        c.execute(
            "UPDATE webhooks SET "
            "  last_delivery_at = ?, last_delivery_status = ?, "
            "  updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (now, response_status, now, webhook_id, int(user_id)),
        )

    return {
        "delivery_id": delivery_id,
        "status": final_status,
        "response_status": response_status,
        "elapsed_ms": elapsed_ms,
        "error_text": err_text,
        "event_id": d["event_id"],
    }


# ── Alert CRUD ────────────────────────────────────────────────────────────

_ALERT_SEVERITIES = ("info", "warn", "critical")


def list_alerts(user_id: int, *, severity: Optional[str] = None,
                code: Optional[str] = None,
                key_id: Optional[str] = None,
                only_unread: bool = False, include_dismissed: bool = False,
                limit: int = 50, cursor: Optional[str] = None) -> dict:
    """List alerts with filters. Returns {"items": [...], "next_cursor": ...}.

    Sort: updated_at DESC, then id DESC (so 'bumped' dedups resurface).
    Cursor = "<updated_at_iso>|<id>" of the last row in the previous page.
    """
    limit = max(1, min(int(limit), 100))
    wheres = ["user_id = ?"]
    args: list = [int(user_id)]
    if severity is not None:
        if severity not in _ALERT_SEVERITIES:
            raise ValueError(f"severity must be one of {_ALERT_SEVERITIES}")
        wheres.append("severity = ?"); args.append(severity)
    if code:
        wheres.append("code = ?"); args.append(code)
    if key_id:
        wheres.append("key_id = ?"); args.append(key_id)
    if only_unread:
        wheres.append("read_at IS NULL")
    if not include_dismissed:
        wheres.append("dismissed_at IS NULL")
    if cursor:
        try:
            c_upd, c_id = cursor.split("|", 1)
            c_id = int(c_id)
            wheres.append("(updated_at < ? OR (updated_at = ? AND id < ?))")
            args.extend([c_upd, c_upd, c_id])
        except Exception:
            raise ValueError("invalid cursor format")

    sql = (
        "SELECT id, user_id, key_id, severity, code, title, body, "
        "       details_json, occurrence_count, created_at, updated_at, "
        "       read_at, dismissed_at "
        "FROM api_key_alerts "
        f"WHERE {' AND '.join(wheres)} "
        "ORDER BY updated_at DESC, id DESC "
        "LIMIT ?"
    )
    args.append(limit + 1)
    with get_conn() as c:
        rows = c.execute(sql, args).fetchall()

    items = []
    for r in rows[:limit]:
        d = dict(r)
        if d.get("details_json"):
            try: d["details"] = json.loads(d["details_json"])
            except Exception: d["details"] = None
        else:
            d["details"] = None
        d.pop("details_json", None)
        items.append(d)

    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = f"{last['updated_at']}|{last['id']}"

    return {"items": items, "next_cursor": next_cursor}


def list_alerts_since(user_id: int, since_id: int,
                       limit: int = 10) -> list[dict]:
    """Phase 8.H · for the toast detector poll. Returns alerts with
    `id > since_id` for this user, OLDEST-FIRST so the client can slide
    them in chronologically. Caps at `limit` rows.

    Includes both read and unread rows — the client uses `read_at` to
    decide whether to toast (only unread → toast). Dismissed rows are
    excluded since they shouldn't bubble back up.
    """
    limit = max(1, min(int(limit), 50))
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, key_id, severity, code, title, body, "
            "       details_json, created_at, updated_at, read_at "
            "FROM api_key_alerts "
            "WHERE user_id = ? AND id > ? AND dismissed_at IS NULL "
            "ORDER BY id ASC LIMIT ?",
            (int(user_id), int(since_id), limit),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        raw = d.pop("details_json", None)
        if raw:
            try:
                d["details"] = json.loads(raw)
            except Exception:
                d["details"] = None
        out.append(d)
    return out


def latest_alert_id(user_id: int) -> int:
    """Phase 8.H · highest alert id in this user's inbox, regardless of
    state. Used by the client to bootstrap its `since` cursor on first
    page load (so we don't replay the entire history as fresh toasts).
    Returns 0 when the user has no alerts."""
    with get_conn() as c:
        row = c.execute(
            "SELECT MAX(id) AS m FROM api_key_alerts WHERE user_id = ?",
            (int(user_id),)
        ).fetchone()
    if not row or row["m"] is None:
        return 0
    return int(row["m"])


def count_unread_alerts(user_id: int) -> int:
    """For the nav bell badge. Excludes dismissed AND snoozed alerts whose
    snooze hasn't yet expired (WI-P8-ALERTS-SNOOZE).

    Snooze is stored as `details_json._snoozed_until` (ISO-8601). When
    that key is in the future we exclude the row. Comparing ISO strings
    lexically works because all timestamps are UTC-normalised with same
    precision.
    """
    now_iso = _now_iso()
    with get_conn() as c:
        # Use a JSON-functions query when available, else fall back to a
        # rows-then-Python filter. SQLite ships json_extract by default
        # in modern builds (3.38+). For Turso/libsql it's always present.
        try:
            n = c.execute(
                "SELECT COUNT(*) AS n FROM api_key_alerts "
                "WHERE user_id = ? AND read_at IS NULL "
                "  AND dismissed_at IS NULL "
                "  AND (details_json IS NULL "
                "       OR json_extract(details_json, '$._snoozed_until') IS NULL "
                "       OR json_extract(details_json, '$._snoozed_until') < ?)",
                (int(user_id), now_iso),
            ).fetchone()
            return int(n["n"])
        except Exception:
            # Fallback (very old SQLite): scan + filter in Python
            rows = c.execute(
                "SELECT details_json FROM api_key_alerts "
                "WHERE user_id = ? AND read_at IS NULL AND dismissed_at IS NULL",
                (int(user_id),),
            ).fetchall()
        n = 0
        for r in rows:
            dj = r["details_json"]   # sqlite3.Row dict-indexing only
            if not dj:
                n += 1; continue
            try:
                d = json.loads(dj)
                snooze = d.get("_snoozed_until") if isinstance(d, dict) else None
            except Exception:
                snooze = None
            if not snooze or snooze < now_iso:
                n += 1
        return n


def get_alert(alert_id: int, user_id: int) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute(
            "SELECT id, user_id, key_id, severity, code, title, body, "
            "       details_json, occurrence_count, created_at, updated_at, "
            "       read_at, dismissed_at "
            "FROM api_key_alerts WHERE id = ? AND user_id = ?",
            (int(alert_id), int(user_id)),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("details_json"):
        try: d["details"] = json.loads(d["details_json"])
        except Exception: d["details"] = None
    else:
        d["details"] = None
    d.pop("details_json", None)
    return d


def mark_alert_read(alert_id: int, user_id: int) -> Optional[dict]:
    """Idempotent: re-marking an already-read alert is a no-op."""
    with _write_lock, get_conn() as c:
        c.execute(
            "UPDATE api_key_alerts SET read_at = COALESCE(read_at, ?), "
            "   updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (_now_iso(), _now_iso(), int(alert_id), int(user_id)),
        )
    return get_alert(int(alert_id), int(user_id))


def dismiss_alert(alert_id: int, user_id: int) -> Optional[dict]:
    """Soft-delete: sets dismissed_at. Use restore_alert to undismiss."""
    with _write_lock, get_conn() as c:
        c.execute(
            "UPDATE api_key_alerts SET dismissed_at = COALESCE(dismissed_at, ?), "
            "   read_at = COALESCE(read_at, ?), updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (_now_iso(), _now_iso(), _now_iso(),
             int(alert_id), int(user_id)),
        )
    return get_alert(int(alert_id), int(user_id))


def snooze_alert(alert_id: int, user_id: int, until_iso: str) -> Optional[dict]:
    """Phase 8 Wave E (WI-P8-ALERTS-SNOOZE): mark an alert as snoozed until
    `until_iso`. Until then, alert is hidden from the unread-count badge
    and from default-filter list views.

    Uses the existing `details_json` column with a `_snoozed_until` key,
    so no schema migration is needed. count_unread_alerts + list_alerts
    consult this on read.
    """
    if not isinstance(until_iso, str) or not until_iso:
        raise ValueError("until_iso must be an ISO-8601 string")
    with _write_lock, get_conn() as c:
        row = c.execute(
            "SELECT details_json FROM api_key_alerts "
            "WHERE id = ? AND user_id = ?",
            (int(alert_id), int(user_id))
        ).fetchone()
        if row is None:
            return None
        details: dict = {}
        raw = row["details_json"]  # sqlite3.Row uses dict-style indexing only
        if raw:
            try: details = json.loads(raw)
            except Exception: details = {}
        if not isinstance(details, dict):
            details = {"_legacy_details": details}
        details["_snoozed_until"] = until_iso
        c.execute(
            "UPDATE api_key_alerts SET details_json = ?, "
            "  updated_at = ?, "
            "  read_at = COALESCE(read_at, ?) "
            "WHERE id = ? AND user_id = ?",
            (json.dumps(details), _now_iso(), _now_iso(),
             int(alert_id), int(user_id))
        )
    return get_alert(int(alert_id), int(user_id))


def restore_alert(alert_id: int, user_id: int) -> Optional[dict]:
    """Reverse of dismiss_alert."""
    with _write_lock, get_conn() as c:
        c.execute(
            "UPDATE api_key_alerts SET dismissed_at = NULL, "
            "   updated_at = ? WHERE id = ? AND user_id = ?",
            (_now_iso(), int(alert_id), int(user_id)),
        )
    return get_alert(int(alert_id), int(user_id))


def create_alert(user_id: int, severity: str, code: str,
                 title: str, body: str, *,
                 key_id: Optional[str] = None,
                 details: Optional[dict] = None) -> int:
    """Insert a new alert. Used by background events (webhook failure,
    quota exceeded, new-IP detection, etc.). Returns the alert id.

    Dedup behavior is handled by callers via the idx_alerts_dedup index;
    they may UPDATE occurrence_count + updated_at instead of inserting
    when (user_id, code, key_id) already has a recent row.
    """
    if severity not in _ALERT_SEVERITIES:
        raise ValueError(f"severity must be one of {_ALERT_SEVERITIES}")
    now = _now_iso()
    details_json = json.dumps(details) if details is not None else None
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "INSERT INTO api_key_alerts "
            "(user_id, key_id, severity, code, title, body, "
            " details_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (int(user_id), key_id, severity, code, title, body,
             details_json, now, now),
        )
        return int(cur.lastrowid)


# ─── Phase 8.H · alert producer registry ──────────────────────────────────
# Each alert code has a default category, default-on flag, severity, and
# a tiny title/body template. Producers call `emit_alert(user_id, code,
# **format_args)` and the helper looks up severity + writes the row,
# gated by the user's prefs.
#
# Schema (per code):
#   code             stable identifier (e.g. "key.created")
#   category         one of 9 categories — controls the category toggle
#   default_on       bool — included in "all categories on" defaults
#   severity         "info" | "warn" | "critical"
#   title_tmpl       Python str.format() template with named args
#   body_tmpl        same
#   action           {kind: "view_key" | "view_request" | "extend_session"
#                          | "approve_block_ip" | "re_enable_webhook"
#                          | "adjust_quota" | "view_settings" | None,
#                     ... kind-specific args ...}
_ALERT_REGISTRY: dict[str, dict] = {
    # ── Category 1 · Key lifecycle ────────────────────────────────────
    "key.created": {
        "category": "key_lifecycle", "default_on": True, "severity": "info",
        "title_tmpl": "New API key created",
        "body_tmpl": "Key '{key_name}' was minted ({environment}).",
    },
    "key.rotated": {
        "category": "key_lifecycle", "default_on": True, "severity": "info",
        "title_tmpl": "Key rotated",
        "body_tmpl": "Key '{key_name}' was rotated; the old secret is now in "
                     "the rotation overlap window.",
    },
    "key.disabled": {
        "category": "key_lifecycle", "default_on": True, "severity": "info",
        "title_tmpl": "Key disabled",
        "body_tmpl": "Key '{key_name}' is now disabled; new requests will 401.",
    },
    "key.enabled": {
        "category": "key_lifecycle", "default_on": True, "severity": "info",
        "title_tmpl": "Key re-enabled",
        "body_tmpl": "Key '{key_name}' is active again.",
    },
    "key.deleted": {
        "category": "key_lifecycle", "default_on": True, "severity": "info",
        "title_tmpl": "Key deleted",
        "body_tmpl": "Key '{key_name}' is permanently deleted; this cannot be "
                     "undone.",
    },
    "key.patched": {
        "category": "key_lifecycle", "default_on": False, "severity": "info",
        "title_tmpl": "Key settings updated",
        "body_tmpl": "Key '{key_name}' had {fields_changed} updated.",
    },

    # ── Category 2 · Key security ─────────────────────────────────────
    "key.first_use_from_new_ip": {
        "category": "key_security", "default_on": True, "severity": "warn",
        "title_tmpl": "Key used from a new IP",
        "body_tmpl": "Key '{key_name}' was used from {ip} for the first time.",
    },
    "key.compromised": {
        "category": "key_security", "default_on": True, "severity": "critical",
        "title_tmpl": "Key marked compromised",
        "body_tmpl": "Key '{key_name}' was flagged as compromised and revoked.",
    },
    "key.repeated_401s": {
        "category": "key_security", "default_on": True, "severity": "warn",
        "title_tmpl": "Repeated auth failures on a key",
        "body_tmpl": "Key '{key_name}' got {count} auth failures in {window}.",
    },

    # ── Category 3 · Webhook delivery ─────────────────────────────────
    "webhook.delivery_gave_up": {
        "category": "webhook_delivery", "default_on": True, "severity": "critical",
        "title_tmpl": "Webhook delivery failed",
        "body_tmpl": "Delivery of event {event} to {url} failed after "
                     "{attempts} attempts and was given up.",
    },
    "webhook.auto_disabled": {
        "category": "webhook_delivery", "default_on": True, "severity": "critical",
        "title_tmpl": "Webhook auto-disabled",
        "body_tmpl": "Webhook {url} was auto-disabled after {giveups} delivery "
                     "give-ups in 24h.",
    },
    "webhook.delivery_recovered": {
        "category": "webhook_delivery", "default_on": False, "severity": "info",
        "title_tmpl": "Webhook delivery succeeded after retry",
        "body_tmpl": "Delivery to {url} succeeded on attempt {attempt}.",
    },

    # ── Category 4 · Webhook config ───────────────────────────────────
    "webhook.created": {
        "category": "webhook_config", "default_on": False, "severity": "info",
        "title_tmpl": "Webhook endpoint added",
        "body_tmpl": "New webhook for {events} → {url}.",
    },
    "webhook.patched": {
        "category": "webhook_config", "default_on": False, "severity": "info",
        "title_tmpl": "Webhook updated",
        "body_tmpl": "Webhook {url} had {fields_changed} updated.",
    },
    "webhook.deleted": {
        "category": "webhook_config", "default_on": False, "severity": "info",
        "title_tmpl": "Webhook deleted",
        "body_tmpl": "Webhook {url} was removed.",
    },
    "webhook.secret_rotated": {
        "category": "webhook_config", "default_on": False, "severity": "info",
        "title_tmpl": "Webhook secret rotated",
        "body_tmpl": "Webhook {url} has a new HMAC signing secret.",
    },

    # ── Category 5 · Quotas & rate limits ─────────────────────────────
    "quota.daily_50": {
        "category": "quota_rate", "default_on": False, "severity": "info",
        "title_tmpl": "Daily quota 50% used",
        "body_tmpl": "{used} of {cap} requests used today.",
    },
    "quota.daily_80": {
        "category": "quota_rate", "default_on": True, "severity": "warn",
        "title_tmpl": "Daily quota 80% used",
        "body_tmpl": "{used} of {cap} requests used today — approaching cap.",
    },
    "quota.daily_exceeded": {
        "category": "quota_rate", "default_on": True, "severity": "critical",
        "title_tmpl": "Daily quota hit",
        "body_tmpl": "Daily cap of {cap} requests reached; subsequent calls "
                     "return 429 until rollover.",
    },
    "quota.rate_limit_hit": {
        "category": "quota_rate", "default_on": False, "severity": "warn",
        "title_tmpl": "Per-minute rate limit triggered",
        "body_tmpl": "Key '{key_name}' hit its per-minute cap of {rpm}.",
    },

    # ── Category 6 · Per-request anomalies ────────────────────────────
    "request.error_5xx": {
        "category": "per_request", "default_on": False, "severity": "warn",
        "title_tmpl": "Request returned 5xx",
        "body_tmpl": "Request {request_id} on '{key_name}' failed with "
                     "status {status}.",
    },
    "request.ood_rejected": {
        "category": "per_request", "default_on": False, "severity": "info",
        "title_tmpl": "Request was OOD-rejected",
        "body_tmpl": "Request {request_id} on '{key_name}' was rejected as "
                     "out-of-distribution (m={mahalanobis:.1f}).",
    },
    "request.high_latency": {
        "category": "per_request", "default_on": False, "severity": "info",
        "title_tmpl": "High-latency request",
        "body_tmpl": "Request {request_id} took {latency_ms}ms (your p99 is "
                     "{p99_ms}ms).",
    },
    "request.live_stream": {
        "category": "per_request", "default_on": False, "severity": "info",
        "title_tmpl": "Live request",
        "body_tmpl": "Request {request_id}: {summary}",
    },

    # ── Category 7 · Session ──────────────────────────────────────────
    "session.expiring_soon": {
        "category": "session", "default_on": True, "severity": "warn",
        "title_tmpl": "Session expires soon",
        "body_tmpl": "Your session ends in {minutes} minutes.",
    },
    "session.new_device": {
        "category": "session", "default_on": False, "severity": "warn",
        "title_tmpl": "New device signed in",
        "body_tmpl": "Signed in from {device} at {ip}.",
    },

    # ── Category 8 · Account changes ──────────────────────────────────
    "account.settings_changed": {
        "category": "account_changes", "default_on": False, "severity": "info",
        "title_tmpl": "Account settings updated",
        "body_tmpl": "Fields changed: {fields_changed}.",
    },
    "account.password_changed": {
        "category": "account_changes", "default_on": True, "severity": "warn",
        "title_tmpl": "Password changed",
        "body_tmpl": "Your account password was changed.",
    },
    "account.email_changed": {
        "category": "account_changes", "default_on": True, "severity": "warn",
        "title_tmpl": "Email changed",
        "body_tmpl": "Your account email was changed to {email}.",
    },
    "account.sudo_amplify_guard": {
        "category": "account_changes", "default_on": True, "severity": "warn",
        "title_tmpl": "Sudo session auto-revoked",
        "body_tmpl": "Sudo was revoked after sudo_max_uses was raised "
                     "(amplify guard).",
    },
    "account.locked": {
        "category": "account_changes", "default_on": True, "severity": "critical",
        "title_tmpl": "Account temporarily locked",
        "body_tmpl": "{count} auth failures in {window} triggered a lock.",
    },

    # ── Category 9 · System ───────────────────────────────────────────
    "system.maintenance_scheduled": {
        "category": "system", "default_on": True, "severity": "info",
        "title_tmpl": "Maintenance scheduled",
        "body_tmpl": "{window} — {description}",
    },
    "system.deprecation": {
        "category": "system", "default_on": True, "severity": "warn",
        "title_tmpl": "API deprecation notice",
        "body_tmpl": "{detail}",
    },
}

_ALERT_CATEGORIES = (
    "key_lifecycle", "key_security",
    "webhook_delivery", "webhook_config",
    "quota_rate", "per_request",
    "session", "account_changes", "system",
)


def _resolve_notify_prefs(user_id: int) -> dict:
    """Parse the user's notify_prefs_json column. Returns:
        {"categories": {cat: bool, ...}, "codes": {code: bool, ...}}
    Missing keys default to the registry's default_on per code.
    """
    s = get_account_settings(int(user_id))
    raw = s.get("notify_prefs_json")
    if isinstance(raw, str) and raw:
        try:
            d = json.loads(raw)
            if not isinstance(d, dict):
                d = {}
        except Exception:
            d = {}
    else:
        d = {}
    cats = d.get("categories") or {}
    codes = d.get("codes") or {}
    if not isinstance(cats, dict):  cats = {}
    if not isinstance(codes, dict): codes = {}
    return {"categories": cats, "codes": codes}


def is_alert_enabled(user_id: int, code: str) -> bool:
    """Check whether the user has alert `code` enabled.

    Precedence (highest → lowest):
      1. Explicit per-code override in `codes`        → wins absolutely
      2. Category-level toggle in `categories`        → applies to all
                                                       codes in that category
      3. Registry default_on                          → fallback
    Unknown code → False (defensive — don't fire unrecognised events).
    """
    meta = _ALERT_REGISTRY.get(code)
    if meta is None:
        return False
    prefs = _resolve_notify_prefs(int(user_id))
    if code in prefs["codes"]:
        return bool(prefs["codes"][code])
    cat = meta["category"]
    if cat in prefs["categories"]:
        return bool(prefs["categories"][cat])
    return bool(meta["default_on"])


def emit_alert(user_id: int, code: str, *,
               key_id: Optional[str] = None,
               action: Optional[dict] = None,
               **fmt) -> Optional[int]:
    """Phase 8.H · the canonical producer entry-point.

    Looks up `code` in the registry, formats the title/body templates with
    the provided **fmt kwargs, checks the user's prefs, and (if enabled)
    inserts a new row via create_alert. Returns the alert id, or None
    when the user has the code disabled.

    `action` is an optional dict like {"kind": "view_key", "public_id":
    "abc123"} that the toast renderer uses to draw the inline action
    button. Stored in details_json alongside the format args (so a user
    rereading the alert weeks later can still click through).

    All exceptions are swallowed — alert emission must NEVER break the
    producer's primary path. A failed insert is logged but not raised.
    """
    import logging as _log
    log = _log.getLogger("apin_v2.alerts")
    meta = _ALERT_REGISTRY.get(code)
    if meta is None:
        log.warning("emit_alert: unknown code %r — skipped", code)
        return None
    try:
        if not is_alert_enabled(int(user_id), code):
            return None
        try:
            title = meta["title_tmpl"].format(**fmt)
        except Exception:
            title = meta["title_tmpl"]
        try:
            body = meta["body_tmpl"].format(**fmt)
        except Exception:
            body = meta["body_tmpl"]
        details = {"args": fmt}
        if action:
            details["action"] = action
        return create_alert(
            int(user_id), meta["severity"], code, title, body,
            key_id=key_id, details=details,
        )
    except Exception as exc:
        log.exception("emit_alert(%s) failed: %s", code, exc)
        return None


def alert_registry_snapshot() -> dict:
    """Return the registry shape the prefs UI needs: per-category list of
    codes with metadata. Used by GET /api/account/alert-prefs."""
    out: dict[str, dict] = {c: {"codes": [], "label": c.replace("_", " ")}
                            for c in _ALERT_CATEGORIES}
    for code, meta in _ALERT_REGISTRY.items():
        out[meta["category"]]["codes"].append({
            "code": code,
            "severity": meta["severity"],
            "default_on": meta["default_on"],
            "title": meta["title_tmpl"],
            "body":  meta["body_tmpl"],
        })
    return out


def consume_sudo_use(sudo_token_id: str, max_uses: int) -> bool:
    """Atomically increment `sudo_tokens.used_count` if under the cap.

    WI-P4-SUDO-USED-COUNT (spec §6.8 + §7.6 PDA-R2-F33): each successful
    sudo-gated mutation MUST increment used_count. On the (max_uses + 1)th
    attempt the increment fails and the caller should revoke the sudo.

    Returns:
        True  — used_count was incremented; mutation may proceed
        False — used_count already at or above cap; caller should reject
                with 403 sudo_required and revoke the token

    Atomicity: the `UPDATE ... WHERE used_count < max_uses` is a single
    statement, so two concurrent mutations cannot both succeed past the
    cap (only the first one finds rowcount=1; the second finds 0 and
    returns False). No application-level lock needed.

    FX-P4-7 (VER-P4-R1 C8): emit canonical `sudo_used` audit event on
    success per spec §12.1 enum + STRIDE §8363 repudiation control.
    Audit is best-effort — a DB glitch must not undo the increment.
    """
    if not sudo_token_id or max_uses <= 0:
        return False
    with _write_lock, get_conn() as c:
        cur = c.execute(
            """UPDATE sudo_tokens
               SET used_count = used_count + 1
               WHERE id = ? AND used_count < ? AND revoked_at IS NULL""",
            (sudo_token_id, max_uses),
        )
        ok = cur.rowcount > 0
    if ok:
        # FX-P4-7: canonical sudo_used audit event
        try:
            audit("sudo_used", detail={"sudo_id": str(sudo_token_id)})
        except Exception:
            pass
    return ok


def create_sudo_token(
    user_id: int,
    session_id: str,
    client_ip: Optional[str],
    ttl_seconds: int = 1800,
) -> tuple[str, str]:
    """Mint a sudo token bound to (user_id, session_id, client_ip).

    Spec §7.6. Returns (raw_cookie_value, expires_at_iso). The raw value
    is sent to the client as the `apin_sudo` HttpOnly cookie; the hash is
    what we store in sudo_tokens.token_hash.

    Why client_ip-bound (and not user-agent-bound): REV-I14 removed UA
    binding because user-agent strings are too brittle (browser updates,
    extensions, etc.). IP binding is left in place but `bound_ip=None`
    is acceptable for any-IP sudo (explicit opt-in by passing None).

    Idempotency: this function does NOT revoke prior sudo tokens for the
    same (user, session). Multiple active sudos for the same session are
    legal. The route handler MAY choose to revoke prior tokens before
    minting a new one — this helper does not.
    """
    raw = secrets.token_urlsafe(32)
    h = _hash_token(raw)
    sudo_id = "su_" + secrets.token_hex(8)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)
    now_iso = now.isoformat()
    expires_iso = expires_at.isoformat()
    # PDA-P3.3-LATENT-1 fix: sudo_tokens.session_id is TEXT NOT NULL in the
    # schema; sessions.id is INTEGER. SQLite coerces the int to string on
    # INSERT, so a follow-up `== session_id_from_sessions_int` comparison
    # in _verify_sudo_cookie fails (type mismatch). Pinning to str here
    # ensures both stored value AND any future direct comparison use the
    # same type. _verify_sudo_cookie's read side is also string-normalized
    # defensively (see middlewares.py).
    session_id_str = str(session_id) if session_id is not None else None
    with _write_lock, get_conn() as c:
        c.execute(
            """INSERT INTO sudo_tokens
               (id, token_hash, user_id, session_id, bound_ip,
                created_at, expires_at, used_count, revoked_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (sudo_id, h, user_id, session_id_str, client_ip,
             now_iso, expires_iso),
        )
    return raw, expires_iso


def revoke_active_sudo_for_session(session_id: str) -> int:
    """Mark all active sudo tokens for this session as revoked.

    Spec §7.6 POST /api/account/sudo/revoke. Returns number of rows
    affected (0 if no active sudo, 1+ if revocation succeeded).
    """
    if not session_id:
        return 0
    # PDA-P3.3-LATENT-1: stringify for storage-type consistency
    session_id_str = str(session_id)
    now_iso = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            """UPDATE sudo_tokens
               SET revoked_at = ?
               WHERE session_id = ? AND revoked_at IS NULL""",
            (now_iso, session_id_str),
        )
        return cur.rowcount


def get_sudo_state_for_cookie(
    cookie_value: str,
    session_id: str,
    client_ip: Optional[str],
) -> dict:
    """Inspect the sudo cookie and report its state.

    Spec §7.6 GET /api/account/sudo response shape:
      {active: bool, expires_at?: str, expires_in_seconds?: int}

    Returns the matching shape. `active=True` only if the cookie maps
    to a row that's not revoked, not expired, AND binding (session +
    optional IP) matches. Same logic as `_verify_sudo_cookie` in
    middlewares.py — we deliberately duplicate (per Rule 11 cross-
    layer-import policy) rather than import across the boundary.
    """
    if not cookie_value or not session_id:
        return {"active": False}
    h = _hash_token(cookie_value)
    with get_conn() as c:
        # FIX-T2 (VER-P3.3-3.4-R1 C4): include used_count in SELECT — spec
        # PDA-R2-F33 (line 156) mandates exposing it via GET /sudo for UI
        # display (so the user can see how many uses remain against
        # account_settings.sudo_max_uses when that lands in Phase 4).
        row = c.execute(
            """SELECT user_id, session_id, bound_ip, expires_at, revoked_at,
                      used_count
               FROM sudo_tokens WHERE token_hash = ? LIMIT 1""",
            (h,),
        ).fetchone()
    if not row:
        return {"active": False}
    d = dict(row)
    if d.get("revoked_at"):
        return {"active": False}
    # PDA-P3.3-LATENT-1: stringify both sides — see create_sudo_token comment.
    if str(d.get("session_id")) != str(session_id):
        return {"active": False}
    bound = d.get("bound_ip")
    if bound not in (None, "") and bound != client_ip:
        return {"active": False}
    try:
        exp_dt = datetime.fromisoformat(d["expires_at"])
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if now >= exp_dt:
            return {"active": False}
        expires_in = int((exp_dt - now).total_seconds())
    except Exception:
        return {"active": False}
    return {
        "active": True,
        "expires_at": d["expires_at"],
        "expires_in_seconds": expires_in,
        # FIX-T2: surface used_count per spec PDA-R2-F33.
        "used_count": int(d.get("used_count", 0) or 0),
    }


def lookup_session_by_token(raw_token: str) -> Optional[dict]:
    """Resolve a raw session token to {session_id, user_id, csrf_token, expires_at}.

    Lighter-weight sibling of `get_session_user` — returns the SESSION row
    (no user JOIN) for middleware that needs the session_id to bind sudo,
    audit logs, etc. to the originating session.

    Used by SessionMiddleware (spec §9.1 slot 6) to populate
    `scope.state.session` for SudoMiddleware (slot 7) to consume.

    Returns:
        {"session_id": str, "user_id": int, "csrf_token": str|None,
         "expires_at": str} on hit
        None on miss / expired / revoked
    """
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    now = _now_iso()
    with get_conn() as c:
        row = c.execute(
            """SELECT id, user_id, csrf_token, expires_at
               FROM sessions
               WHERE token_hash = ?
                 AND revoked_at IS NULL
                 AND expires_at > ?
               LIMIT 1""",
            (h, now),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        return {
            "session_id": d["id"],
            "user_id":    d["user_id"],
            "csrf_token": d.get("csrf_token"),
            "expires_at": d.get("expires_at"),
        }


# ── Phase 8.G · session TTL defaults + introspect + extend ─────────────────
# Per-user TTLs override these via account_settings.
_DEFAULT_SESSION_ABSOLUTE_TTL_S = 7 * 24 * 3600           # 7 days
_DEFAULT_SESSION_IDLE_TTL_S     = 2 * 24 * 3600           # 48 hours
_DEFAULT_REMEMBER_ME_TTL_S      = 30 * 24 * 3600          # 30 days
# Warning lead: 5 minutes before expiry the client shows the modal.
_SESSION_WARN_LEAD_S            = 5 * 60


def _resolve_session_ttls(user_id: int) -> dict:
    """Return per-user TTLs, falling back to the module defaults if the
    user's account_settings row has NULL for a given column."""
    s = get_account_settings(int(user_id))
    return {
        "absolute_ttl_s": int(s.get("session_absolute_ttl_seconds")
                              or _DEFAULT_SESSION_ABSOLUTE_TTL_S),
        "idle_ttl_s":     int(s.get("session_idle_ttl_seconds")
                              or _DEFAULT_SESSION_IDLE_TTL_S),
        "remember_me_s":  int(s.get("remember_me_ttl_seconds")
                              or _DEFAULT_REMEMBER_ME_TTL_S),
    }


def session_introspect(raw_token: str) -> Optional[dict]:
    """Phase 8.G · resolve a raw session token to expiry info for the
    Console account chip. Returns None on miss/expired/revoked.

    Shape (matches what console_account_chip.js expects):
      {
        "session_id":     str,
        "user_id":        int,
        "created_at":     ISO timestamp,
        "expires_at":     ISO timestamp,
        "absolute_cap_at": ISO timestamp (created_at + absolute TTL),
        "idle_warning_at": ISO timestamp (expires_at - 5 min),
        "ttls": { absolute_ttl_s, idle_ttl_s, remember_me_s }
      }
    """
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    now = _now_iso()
    with get_conn() as c:
        row = c.execute(
            """SELECT id, user_id, created_at, expires_at
               FROM sessions
               WHERE token_hash = ?
                 AND revoked_at IS NULL
                 AND expires_at > ?""",
            (h, now),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        ttls = _resolve_session_ttls(int(d["user_id"]))
    except Exception:
        ttls = {
            "absolute_ttl_s": _DEFAULT_SESSION_ABSOLUTE_TTL_S,
            "idle_ttl_s":     _DEFAULT_SESSION_IDLE_TTL_S,
            "remember_me_s":  _DEFAULT_REMEMBER_ME_TTL_S,
        }
    # Compute the absolute cap (created_at + absolute_ttl).
    try:
        created = datetime.fromisoformat(d["created_at"].replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        abs_cap = created + timedelta(seconds=ttls["absolute_ttl_s"])
        abs_cap_iso = abs_cap.isoformat()
    except Exception:
        abs_cap_iso = d.get("expires_at")
    # Compute the warning lead-time stamp.
    try:
        exp_dt = datetime.fromisoformat(
            str(d["expires_at"]).replace("Z", "+00:00"))
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        warn_iso = (exp_dt - timedelta(seconds=_SESSION_WARN_LEAD_S)).isoformat()
    except Exception:
        warn_iso = None
    return {
        "session_id":      d["id"],
        "user_id":         d["user_id"],
        "created_at":      d["created_at"],
        "expires_at":      d["expires_at"],
        "absolute_cap_at": abs_cap_iso,
        "idle_warning_at": warn_iso,
        "ttls":            ttls,
    }


def extend_session(raw_token: str) -> Optional[dict]:
    """Phase 8.G · sliding renewal. Push the session's expires_at forward
    by `idle_ttl_s`, capped at `created_at + absolute_ttl_s`.

    Returns:
      None              on miss / expired / revoked
      {"error": "absolute_cap_reached"}  if the absolute cap is in the past
      {"expires_at": ISO, "absolute_cap_at": ISO, ...} on success

    Idempotent — caller can spam this and we'll always serialize the new
    expiry. The cap check is per-session, not per-user, so different sessions
    of the same user are independent."""
    if not raw_token:
        return None
    info = session_introspect(raw_token)
    if info is None:
        return None
    # Compute the new candidate expiry: now + idle_ttl, capped at absolute.
    now = datetime.now(timezone.utc)
    new_expiry = now + timedelta(seconds=info["ttls"]["idle_ttl_s"])
    try:
        abs_cap = datetime.fromisoformat(
            info["absolute_cap_at"].replace("Z", "+00:00"))
        if abs_cap.tzinfo is None:
            abs_cap = abs_cap.replace(tzinfo=timezone.utc)
    except Exception:
        abs_cap = new_expiry
    if abs_cap <= now:
        return {"error": "absolute_cap_reached"}
    if new_expiry > abs_cap:
        new_expiry = abs_cap
    new_expiry_iso = new_expiry.isoformat()
    h = _hash_token(raw_token)
    with _write_lock, get_conn() as c:
        c.execute(
            "UPDATE sessions SET expires_at = ? "
            "WHERE token_hash = ? AND revoked_at IS NULL",
            (new_expiry_iso, h),
        )
    warn_iso = (new_expiry - timedelta(seconds=_SESSION_WARN_LEAD_S)).isoformat()
    return {
        "expires_at":      new_expiry_iso,
        "absolute_cap_at": info["absolute_cap_at"],
        "idle_warning_at": warn_iso,
        "ttls":            info["ttls"],
    }


def revoke_session(raw_token: str) -> bool:
    if not raw_token:
        return False
    h = _hash_token(raw_token)
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "UPDATE sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (_now_iso(), h),
        )
        return cur.rowcount > 0


# ─── Guest session helpers ────────────────────────────────────────────────────
#
# Guests are deliberately a SEPARATE concept from users — no row in `users`,
# no dashboard, no persisted predictions.  The only state a guest carries is
# an inference counter, enforced server-side so the free-check quota cannot
# be cheated by clearing browser storage.

def create_guest_session(
    *,
    user_agent: Optional[str] = None,
    ip_addr: Optional[str] = None,
) -> str:
    """Create a guest-session row; return the raw token for the cookie."""
    raw, h = _new_session_token()
    now = datetime.now(timezone.utc)
    expires = (now + GUEST_SESSION_LIFETIME).isoformat()
    with _write_lock, get_conn() as c:
        c.execute(
            """INSERT INTO guest_sessions
               (token_hash, inference_count, user_agent, ip_addr,
                created_at, expires_at)
               VALUES (?, 0, ?, ?, ?, ?)""",
            (h, user_agent, ip_addr, now.isoformat(), expires),
        )
    return raw


def get_guest_session(raw_token: str) -> Optional[dict]:
    """Resolve a raw guest token to its row, or None if invalid/expired.

    Returns a dict with: id, inference_count, remaining, exhausted.
    """
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    now = _now_iso()
    with get_conn() as c:
        row = c.execute(
            """SELECT id, inference_count FROM guest_sessions
               WHERE token_hash = ? AND expires_at > ? LIMIT 1""",
            (h, now),
        ).fetchone()
    if row is None:
        return None
    used = int(row["inference_count"])
    remaining = max(0, GUEST_INFERENCE_LIMIT - used)
    return {
        "id":              int(row["id"]),
        "inference_count": used,
        "remaining":       remaining,
        "exhausted":       remaining <= 0,
    }


def consume_guest_inference(raw_token: str) -> Optional[dict]:
    """Atomically increment a guest's inference counter IF quota remains.

    Returns the post-increment state {inference_count, remaining, exhausted}
    on success, or None when the token is invalid/expired, or a dict with
    `"denied": True` when the quota was already exhausted.

    The increment is guarded by a WHERE clause on inference_count so two
    concurrent requests cannot both slip past the cap (no check-then-act
    race — the DB does the compare and the update in one statement).
    """
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    now = _now_iso()
    with _write_lock, get_conn() as c:
        row = c.execute(
            """SELECT id, inference_count FROM guest_sessions
               WHERE token_hash = ? AND expires_at > ? LIMIT 1""",
            (h, now),
        ).fetchone()
        if row is None:
            return None
        used = int(row["inference_count"])
        if used >= GUEST_INFERENCE_LIMIT:
            return {"denied": True, "inference_count": used,
                    "remaining": 0, "exhausted": True}
        cur = c.execute(
            """UPDATE guest_sessions
               SET inference_count = inference_count + 1
               WHERE token_hash = ? AND inference_count < ?""",
            (h, GUEST_INFERENCE_LIMIT),
        )
        if cur.rowcount == 0:
            # Lost the race — another request consumed the last slot.
            return {"denied": True, "inference_count": GUEST_INFERENCE_LIMIT,
                    "remaining": 0, "exhausted": True}
    new_used   = used + 1
    remaining  = max(0, GUEST_INFERENCE_LIMIT - new_used)
    return {"inference_count": new_used, "remaining": remaining,
            "exhausted": remaining <= 0}


# ─── Audit log ────────────────────────────────────────────────────────────────

def audit(
    event: str,
    *,
    user_id: Optional[int] = None,
    detail: Optional[dict] = None,
    ip_addr: Optional[str] = None,
):
    """Append an audit-log row. Safe to call from any thread."""
    import json
    with _write_lock, get_conn() as c:
        c.execute(
            "INSERT INTO audit_log (user_id, event, detail, ip_addr, created_at) VALUES (?,?,?,?,?)",
            (user_id, event, json.dumps(detail or {}), ip_addr, _now_iso()),
        )


# ─── Rate limiting (in-memory; sufficient for single-server deployment) ───────

_login_attempts: dict[str, list[float]] = {}
_login_attempts_lock = threading.Lock()


def check_rate_limit(ip: str, *, max_per_window: int = 5, window_seconds: int = 60) -> bool:
    """Return True if this IP is allowed to attempt another login.

    Sliding-window: count attempts within last `window_seconds`. If under limit,
    record this attempt and return True. Else return False.
    """
    import time
    now = time.time()
    cutoff = now - window_seconds
    with _login_attempts_lock:
        history = _login_attempts.get(ip, [])
        history = [t for t in history if t > cutoff]
        if len(history) >= max_per_window:
            _login_attempts[ip] = history
            return False
        history.append(now)
        _login_attempts[ip] = history
        return True


def clear_rate_limit(ip: str):
    """Clear rate-limit history for an IP (called after successful login)."""
    with _login_attempts_lock:
        _login_attempts.pop(ip, None)


# ── WI-P4-RATE-SUDO: sudo POST rate-limit (spec line 5791 + 8363) ──────
#
# Separate in-memory window from `_login_attempts` so the two limits don't
# interact. Key shape: `"u{user_id}:i{ip}"` — caps a stolen-session attacker
# at one IP (sliding-window window) AND prevents distributed brute-force
# (same user_id from multiple IPs sums to the same per-user counter via
# the helper below). Phase 4 production-ready will swap this for a Redis
# token bucket; in-memory is fine for the single-server deploy today.

_sudo_attempts: dict[str, list[float]] = {}
_sudo_attempts_lock = threading.Lock()
# FX-P4-2 (PDA-P4-R1 F2): cap dict size to prevent memory DoS via
# (user, IP) rotation. Each entry is ~64 bytes of dict overhead + 8 bytes
# per timestamp; cap at 10 000 entries = ~1 MB worst case. FIFO eviction
# on add (oldest key removed when full).
_SUDO_ATTEMPTS_MAX = 10_000


# ── Phase 6.C.2: pluggable rate-limit backend ──────────────────────────
#
# The two functions below (`check_sudo_rate_limit`, `clear_sudo_rate_limit`)
# delegate to a swappable backend. The in-memory default is fine for the
# single-server deploy today; a multi-worker production deploy can swap in
# a Redis backend without changing call sites.
#
# Contract: a backend must expose:
#   - check(key: str, max_per_window: int, window_seconds: int) -> bool
#     Atomically: count attempts for `key` in the last `window_seconds`,
#     return False if >= max_per_window; else record this attempt + True.
#   - clear(key: str) -> None
#     Remove all history for `key`.
#   - clear_prefix(prefix: str) -> None
#     Remove all history for keys starting with `prefix`.
#
# Swap by setting `_SUDO_RATE_BACKEND = MyRedisBackend(redis_url)`
# before any sudo route is exercised. The InMemoryBackend implementation
# below is the historical behavior preserved as-is.


class _InMemoryRateBackend:
    """Single-process in-memory rate-limit. The historical behavior; fine
    for `--workers 1` deploys, NOT safe for multi-worker uvicorn (each
    worker has its own counter, so effective limit = max × num_workers).

    Wraps the existing `_sudo_attempts` dict + lock so the swap-in for a
    Redis backend is a one-line `_SUDO_RATE_BACKEND = RedisBackend(...)`.
    """

    def check(self, key: str, max_per_window: int, window_seconds: int) -> bool:
        import time
        now = time.time()
        cutoff = now - window_seconds
        with _sudo_attempts_lock:
            history = _sudo_attempts.get(key, [])
            history = [t for t in history if t > cutoff]
            if len(history) >= max_per_window:
                _sudo_attempts[key] = history
                return False
            history.append(now)
            _sudo_attempts[key] = history
            if len(_sudo_attempts) > _SUDO_ATTEMPTS_MAX:
                try:
                    oldest = next(iter(_sudo_attempts))
                    _sudo_attempts.pop(oldest, None)
                except StopIteration:
                    pass
            return True

    def clear(self, key: str) -> None:
        with _sudo_attempts_lock:
            _sudo_attempts.pop(key, None)

    def clear_prefix(self, prefix: str) -> None:
        with _sudo_attempts_lock:
            for k in [k for k in _sudo_attempts.keys() if k.startswith(prefix)]:
                _sudo_attempts.pop(k, None)


# Module-level singleton. Swap in a different backend (e.g. RedisBackend)
# at process startup before any sudo route runs.
_SUDO_RATE_BACKEND: _InMemoryRateBackend = _InMemoryRateBackend()


def set_sudo_rate_backend(backend) -> None:
    """Override the default in-memory rate-limit backend.

    Phase 6.C.2 (WI-P4+-RATE-SHARED): for production multi-worker deploys,
    swap in a shared backend (e.g. Redis) at startup before the first
    request. The backend must implement `check(key, max, window) -> bool`,
    `clear(key)`, and `clear_prefix(prefix)`.
    """
    global _SUDO_RATE_BACKEND
    _SUDO_RATE_BACKEND = backend


def check_sudo_rate_limit(
    user_id: int,
    ip: Optional[str],
    *,
    max_per_window: int = 5,
    window_seconds: int = 300,
) -> bool:
    """Return True if this (user, IP) pair may attempt another sudo.

    Spec line 5791 mandates a rate-limiter on POST /api/account/sudo.
    Defaults: 5 attempts per 5-minute window — argon2id takes ~150 ms
    per verify, so even at the cap an attacker gets ~33 guesses/hour
    sustained per (user, IP).

    KNOWN LIMITATIONS (PDA-P4-R1 F2, F3):
      1. **Single-process state**: the `_sudo_attempts` dict is in-memory
         per Python process. Multi-worker uvicorn (e.g. `--workers 4` as
         documented in README) gives EACH worker its own counter, so the
         effective limit is `max_per_window * num_workers` (20/window
         under the documented 4-worker production config). For a true
         per-(user, IP) limit across workers, swap for Redis/DB-backed
         in Phase 4+ (WI-P4+-RATE-SHARED in decisions.md).
      2. **Dict-size cap (FX-P4-2)**: `_SUDO_ATTEMPTS_MAX = 10_000`
         entries; oldest key is evicted on overflow. An attacker rotating
         IPs can churn old entries out, but cannot blow up memory.

    Returns False without recording the attempt when the cap is reached
    (so a rejected attempt doesn't extend the window).

    Phase 6.C.2: this function is now a thin wrapper that delegates to the
    swappable `_SUDO_RATE_BACKEND` singleton. The keyword API is preserved
    so routes_sudo.py and existing tests remain untouched. The KNOWN
    LIMITATIONS above describe the *default* in-memory backend; a Redis
    backend swapped in via `set_sudo_rate_backend()` resolves limitation 1.
    """
    if user_id is None:
        return True   # defensive: anonymous attempts shouldn't reach this
    key = "u{}:i{}".format(int(user_id), ip or "?")
    # Read the global at call time so a Phase-7 backend swap takes effect.
    return _SUDO_RATE_BACKEND.check(key, max_per_window, window_seconds)


def clear_sudo_rate_limit(user_id: int, ip: Optional[str] = None) -> None:
    """Clear sudo rate-limit history for a (user, IP) pair after success.

    Phase 6.C.2: delegates to `_SUDO_RATE_BACKEND` (in-memory by default,
    Redis-swappable for production). When `ip` is None, clears every entry
    for this user across all IPs via `clear_prefix()` — used by the admin
    revoke-all path.
    """
    if ip is not None:
        _SUDO_RATE_BACKEND.clear("u{}:i{}".format(int(user_id), ip))
    else:
        _SUDO_RATE_BACKEND.clear_prefix("u{}:i".format(int(user_id)))


# ─── Prediction history ───────────────────────────────────────────────────────

def _extract_prediction_summary(response: dict) -> dict:
    """Pull `crop / predicted_class / confidence / tier` from a /predict/full
    response. Robust to the three different response shapes:
      * TomatoPipeline   — has 'best_class' / 'top_class' / 'top_confidence' / 'tier_label'
      * APIN (okra/brassica) — has 'best_class' / 'overall_confidence' / 'tier_label'
      * Router-rejected / tomato-unavailable — has 'tier' but no class
    Missing fields return None; we never crash a background write.
    """
    if not isinstance(response, dict):
        return {"crop": None, "predicted_class": None, "confidence": None, "tier": None}

    routing = response.get("routing") or {}
    crop = routing.get("router_crop") if isinstance(routing, dict) else None

    # Top class candidates across shapes
    predicted_class = (
        response.get("best_class")
        or response.get("top_class")
        or response.get("diagnosis")
        or None
    )
    # If diagnosis is a dict, pull its inner label
    if isinstance(predicted_class, dict):
        predicted_class = predicted_class.get("class") or predicted_class.get("name")

    # Confidence candidates
    conf = (
        response.get("top_confidence")
        or response.get("overall_confidence")
        or response.get("confidence")
    )
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None

    # Tier label varies between systems
    tier = response.get("tier_label") or response.get("tier")
    if not isinstance(tier, (str, type(None))):
        tier = str(tier)

    return {
        "crop": str(crop) if crop is not None else None,
        "predicted_class": str(predicted_class) if predicted_class is not None else None,
        "confidence": conf,
        "tier": tier,
    }


_HEAVY_KEY_HINTS = ("heatmap", "image_b64", "raw_image", "cam_overlay",
                     "gradcam", "image_base64", "thumbnail",
                     # Phase-3.5 — strip the stage-debug visualization dict
                     # *entirely* (it contains 5 PNGs, each up to ~340 KB,
                     # which pushed response_json well over the 200 KB cap
                     # and caused the whole payload to be replaced with a
                     # "_truncated" envelope — wiping out signal_predictions
                     # and breaking the Day Detail bars in the process).
                     "pipeline_visualizations",
                     "pipeline_viz")


def _strip_heavy_keys(obj):
    """Recursively replace heatmap-like keys with '<stripped>'.

    Heatmap base64 strings can be hundreds of KB each; stripping them keeps
    the stored response_json small enough to query usefully later, while
    preserving all the diagnostic structure (tier, signals, votes, etc.).

    NOTE: when called from record_prediction(), _extract_heatmap_b64() has
    ALREADY run on the unscrubbed response and captured a heatmap (or a
    suitable fallback from pipeline_visualizations).  This function only
    decides what survives into response_json.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and any(h in k.lower() for h in _HEAVY_KEY_HINTS):
                out[k] = "<stripped>"
            elif isinstance(v, (dict, list)):
                out[k] = _strip_heavy_keys(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_strip_heavy_keys(x) for x in obj]
    return obj


_RESPONSE_JSON_MAX = 200_000  # ~200 KB DB row cap


def _extract_heatmap_b64(response: dict) -> Optional[str]:
    """Pull the Grad-CAM heatmap base64 out of a fresh inference response
    BEFORE _strip_heavy_keys scrubs it.  Returns None when no heatmap is
    present.  The pipeline conventionally puts it at a handful of paths;
    we search them in order of preference.
    """
    if not isinstance(response, dict):
        return None
    # Common locations the tomato/okra inference pipelines use.
    # `gradcam_b64_png` is the okra/brassica pipeline's final overlay key;
    # it can be None when confidence is too low to draw a meaningful CAM.
    for path in (("heatmap_b64",),
                 ("gradcam_b64",),
                 ("gradcam_b64_png",),
                 ("explain", "heatmap_b64"),
                 ("explain", "gradcam_b64"),
                 ("explain", "gradcam_b64_png"),
                 ("model2", "heatmap_b64"),
                 ("efficientnet", "heatmap_b64"),
                 # Phase-3.5 fallback: when the proper Grad-CAM is None
                 # (low-confidence predictions), the okra/brassica
                 # pipeline still emits pipeline_visualizations.*  pngs.
                 # The leaf-isolation mask is the most semantically
                 # appropriate "where the model looked" fallback; the
                 # CLAHE-preproc visualization is next.
                 ("pipeline_visualizations", "gate_zero_leaf_mask"),
                 ("pipeline_visualizations", "preproc_rgb_clahe"),
                 ("pipeline_visualizations", "preproc_lab_clahe"),
                 ("pipeline_visualizations", "gate_zero_lap"),
                 ("pipeline_visualizations", "gate_zero_lab_l")):
        cur = response
        ok  = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and isinstance(cur, str) and len(cur) > 100:
            return cur
    return None


# Cap stored heatmap base64 at 1 MB to prevent a runaway heatmap from
# bloating a single DB row beyond reason.  Real Grad-CAM PNGs at 224×224
# encode to ~30–80 KB so this is a generous ceiling.
_HEATMAP_MAX_CHARS = 1_000_000


def record_prediction(
    user_id: int,
    response: dict,
    *,
    image_bytes: Optional[bytes] = None,
    extras: Optional[dict] = None,
) -> Optional[int]:
    """Insert a prediction row. Returns the new row id, or None on failure.

    Designed to be safe to call from a BackgroundTask — never raises.
    Always stores VALID json — if a response is still too large after
    stripping heatmaps, we store a compact summary instead of truncated
    invalid JSON.

    Phase-3.5 — also persists the raw `image_bytes` and the Grad-CAM
    `heatmap_b64` (captured BEFORE _strip_heavy_keys scrubs the response)
    so every downstream surface can render the actual specimen, not a
    placeholder.

    Stage 2 — accepts an optional `extras` dict to populate v2 telemetry
    columns added by _migrate_v2_extensions (browser_session_id, EXIF,
    client_country, signal_predictions, latency breakdowns, etc.).
    Unknown keys are silently ignored so callers can pass a shared dict
    of all known telemetry without worrying about which column exists.
    Existing call sites that pass no extras get byte-identical behaviour.
    """
    import json as _json
    try:
        summary = _extract_prediction_summary(response)
        img_hash = None
        if image_bytes:
            img_hash = hashlib.sha256(image_bytes).hexdigest()

        # Capture the heatmap BEFORE we strip it out of response_json.
        heatmap_b64 = None
        if isinstance(response, dict):
            heatmap_b64 = _extract_heatmap_b64(response)
            if heatmap_b64 and len(heatmap_b64) > _HEATMAP_MAX_CHARS:
                heatmap_b64 = None  # too large — drop rather than truncate

        if isinstance(response, dict):
            slim = _strip_heavy_keys(response)
            response_json = _json.dumps(slim, default=str)
        else:
            response_json = _json.dumps({"_": str(response)[:500]})

        # If still oversize after stripping, fall back to a minimal envelope
        # rather than truncating into invalid JSON.
        if len(response_json) > _RESPONSE_JSON_MAX:
            response_json = _json.dumps({
                "_truncated": True,
                "_original_size_bytes": len(response_json),
                "summary": summary,
            })

        # Base columns — present on every predictions row (legacy + v2)
        cols = ["user_id", "crop", "predicted_class", "confidence", "tier",
                "image_sha256", "image_bytes", "heatmap_b64",
                "response_json", "created_at"]
        vals = [user_id, summary["crop"], summary["predicted_class"],
                summary["confidence"], summary["tier"], img_hash,
                sqlite3.Binary(image_bytes) if image_bytes else None,
                heatmap_b64, response_json, _now_iso()]

        # Optional extension columns — same allowlist as record_guest_prediction
        # so the two writes are symmetric and ingest_telemetry_batch can union them.
        if extras:
            _supported = {
                "api_key_id", "browser_session_id", "client_ip_hash",
                "user_agent_family", "client_country", "client_region", "client_city",
                "exif_camera_model", "exif_capture_timestamp",
                "exif_gps_lat", "exif_gps_lon", "exif_gps_accuracy_m",
                "image_perceptual_hash", "image_n_bytes", "image_width",
                "image_height", "image_mimetype",
                "signal_predictions", "gate_decision_path",
                "deployment_version", "model_weights_hash", "cold_start",
                "fallback_to_cpu", "gpu_used", "peak_vram_mb",
                "conformal_set", "conformal_set_size", "ood_flag",
                "calibration_warning", "predicted_top3",
                "validation_ms", "router_ms", "specialist_ms",
                "calibration_ms", "total_ms",
                "endpoint", "api_version", "request_id", "trace_id",
                "status_code", "error_class", "error_message",
                "review_status", "sampled_for_review", "confidence_outlier",
                "consent_to_research", "consent_to_share", "data_residency_region",
                "user_pseudoid", "treatment_advice_shown",
                "grad_cam_generated", "pdf_report_generated", "experiment_exposures",
            }
            for k, v in extras.items():
                if k in _supported and v is not None:
                    cols.append(k)
                    vals.append(v)

        placeholders = ",".join(["?"] * len(cols))
        sql = ("INSERT INTO predictions (" + ",".join(cols) + ") VALUES ("
               + placeholders + ")")
        with _write_lock, get_conn() as c:
            cur = c.execute(sql, tuple(vals))
            return cur.lastrowid
    except Exception:
        # Background tasks should never raise — log and continue
        import logging as _l
        _l.getLogger("apin_v2.auth").exception("record_prediction failed")
        return None


# ─── Owner-gated image retrieval ──────────────────────────────────────────────

def get_prediction_image(prediction_id: int, *,
                         user_id: int) -> Optional[bytes]:
    """Return the raw uploaded JPEG/PNG bytes for prediction `prediction_id`
    if AND ONLY IF it belongs to `user_id`.  Returns None when:
      - the row does not exist
      - the row exists but is owned by another user (IDOR shield)
      - the row exists but predates Phase-3.5 (image_bytes IS NULL)

    The frontend distinguishes "not found" from "no image stored" by
    using a HEAD request first, but the API surface is the same: 404.
    """
    with get_conn() as c:
        row = c.execute(
            "SELECT image_bytes FROM predictions "
            "WHERE id = ? AND user_id = ?",
            (int(prediction_id), int(user_id)),
        ).fetchone()
    if row is None or row["image_bytes"] is None:
        return None
    return bytes(row["image_bytes"])


def get_prediction_heatmap(prediction_id: int, *,
                           user_id: int) -> Optional[bytes]:
    """Return the Grad-CAM PNG bytes (b64-decoded) for the prediction,
    enforcing ownership.  Returns None when missing or wrong owner.
    """
    import base64 as _b64
    with get_conn() as c:
        row = c.execute(
            "SELECT heatmap_b64 FROM predictions "
            "WHERE id = ? AND user_id = ?",
            (int(prediction_id), int(user_id)),
        ).fetchone()
    if row is None or row["heatmap_b64"] is None:
        return None
    try:
        return _b64.b64decode(row["heatmap_b64"])
    except Exception:
        return None


# ─── Share-token-gated image retrieval ────────────────────────────────────────

def _resolve_share_pid(token: str) -> Optional[int]:
    """Validate a share token and return the underlying prediction_id, or
    None when the token is invalid/revoked/expired.  Used by the public
    /share/{token}/image and /heatmap routes — does NOT increment the
    share view counter (only the JSON-data route should do that).
    """
    if not token:
        return None
    token_hash = _hash_token(token)
    with get_conn() as c:
        row = c.execute(
            "SELECT prediction_id, revoked_at, expires_at "
            "FROM share_tokens WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
    if row is None:
        return None
    if row["revoked_at"]:
        return None
    if row["expires_at"]:
        try:
            from datetime import datetime as _dt
            if _dt.fromisoformat(row["expires_at"]) < _dt.now(timezone.utc):
                return None
        except Exception:
            pass
    return int(row["prediction_id"])


def resolve_share_image(token: str) -> Optional[bytes]:
    """Public: return raw image bytes for a valid share token, or None."""
    pid = _resolve_share_pid(token)
    if pid is None:
        return None
    with get_conn() as c:
        row = c.execute(
            "SELECT image_bytes FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()
    if row is None or row["image_bytes"] is None:
        return None
    return bytes(row["image_bytes"])


def resolve_share_heatmap(token: str) -> Optional[bytes]:
    """Public: return Grad-CAM PNG bytes for a valid share token, or None."""
    import base64 as _b64
    pid = _resolve_share_pid(token)
    if pid is None:
        return None
    with get_conn() as c:
        row = c.execute(
            "SELECT heatmap_b64 FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()
    if row is None or row["heatmap_b64"] is None:
        return None
    try:
        return _b64.b64decode(row["heatmap_b64"])
    except Exception:
        return None


def get_user_predictions(
    user_id: int,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """Return rows from `predictions` for this user, newest first.

    Excludes `response_json` blob from the listing for size; use
    get_prediction_full(id) to retrieve a single row's full response.
    """
    limit = max(1, min(100, int(limit)))
    offset = max(0, int(offset))
    with get_conn() as c:
        rows = c.execute(
            """SELECT id, crop, predicted_class, confidence, tier,
                      image_sha256, created_at,
                      (image_bytes IS NOT NULL) AS has_image,
                      (heatmap_b64 IS NOT NULL) AS has_heatmap
               FROM predictions
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            # Coerce SQLite integers to clean booleans for the frontend
            d["has_image"]   = bool(d.get("has_image"))
            d["has_heatmap"] = bool(d.get("has_heatmap"))
            out.append(d)
        return out


def count_user_predictions(user_id: int) -> int:
    with get_conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row["n"]) if row else 0


def get_prediction_full(prediction_id: int, *, user_id: int) -> Optional[dict]:
    """Return full row scoped to one user.  Explicitly enumerates columns
    so that the new `image_bytes` BLOB and `heatmap_b64` (both potentially
    multi-MB) are NEVER returned in this generic getter — those flow only
    through the dedicated /image and /heatmap routes that stream them as
    binary responses.
    """
    with get_conn() as c:
        row = c.execute(
            "SELECT id, user_id, crop, predicted_class, confidence, tier, "
            "       image_sha256, response_json, created_at, "
            "       (image_bytes IS NOT NULL) AS has_image, "
            "       (heatmap_b64 IS NOT NULL) AS has_heatmap "
            "FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["has_image"]   = bool(d.get("has_image"))
        d["has_heatmap"] = bool(d.get("has_heatmap"))
        return d


def prediction_has_image(prediction_id: int, *, user_id: int) -> bool:
    """Lightweight existence probe used by the frontend before requesting
    bytes — lets surfaces decide between rendering `<img src=/image>` and
    rendering the honest "image not captured" placeholder without paying
    a full BLOB transfer.  Ownership-gated.
    """
    with get_conn() as c:
        row = c.execute(
            "SELECT image_bytes IS NOT NULL AS has_img, "
            "       heatmap_b64 IS NOT NULL AS has_cam "
            "FROM predictions WHERE id = ? AND user_id = ?",
            (int(prediction_id), int(user_id)),
        ).fetchone()
    return bool(row) and bool(row["has_img"])


def prediction_image_flags(prediction_id: int, *,
                           user_id: int) -> Optional[dict]:
    """Returns {has_image, has_heatmap} for the row, ownership-gated.
    None when row not found or wrong owner.  Used by listing endpoints to
    pre-flight image availability in a single SQL hit.
    """
    with get_conn() as c:
        row = c.execute(
            "SELECT image_bytes IS NOT NULL AS has_img, "
            "       heatmap_b64 IS NOT NULL AS has_cam "
            "FROM predictions WHERE id = ? AND user_id = ?",
            (int(prediction_id), int(user_id)),
        ).fetchone()
    if not row:
        return None
    return {"has_image":   bool(row["has_img"]),
            "has_heatmap": bool(row["has_cam"])}


# ─── Dashboard aggregation ────────────────────────────────────────────────────
#
# get_dashboard_data() runs the handful of SELECTs the dashboard needs in one
# connection and returns a single dict the frontend can consume. Doing the
# aggregation in SQL (instead of Python) keeps the wire payload small even
# for users with thousands of predictions.

_HEALTHY_TOKENS = ("healthy", "Healthy", "HEALTHY")


def _looks_healthy(predicted_class: Optional[str]) -> bool:
    """True if a predicted_class string clearly denotes a healthy leaf.
    Robust to capitalisation and the various crop-prefixed forms we use
    (okra_healthy, tomato_healthy, brassica_healthy, etc.)."""
    if not predicted_class:
        return False
    p = str(predicted_class).lower()
    return ("healthy" in p) or (p in {"none", "no_disease"})


def get_dashboard_data(user_id: int) -> dict:
    """One-shot fetch that powers the entire dashboard JSON payload.

    Returns a dict shaped for the frontend widgets:
        user        - basic profile (display_name, pressed_leaf_seed, etc.)
        hero        - {total, healthy_pct, avg_confidence, streak_days}
        calendar    - list of {date, count} for the last 28 days (oldest first)
        ledger      - top predicted_class counts (max 8)
        crop_mix    - per-crop counts (full distribution)
        dominant_crop - the crop the user has logged most (drives avatar leaf)
        recent      - last 6 predictions (slim columns only, no response_json)
    """
    import json as _json
    from datetime import datetime, timezone, timedelta

    user = get_user_by_id(user_id) or {}

    with get_conn() as c:
        # ── 1. Total count + healthy% + average confidence
        row = c.execute(
            """SELECT
                  COUNT(*)                                          AS total,
                  SUM(CASE WHEN LOWER(predicted_class) LIKE '%healthy%'
                            OR LOWER(predicted_class) = 'no_disease'
                           THEN 1 ELSE 0 END)                       AS healthy_n,
                  AVG(confidence)                                   AS avg_conf
               FROM predictions WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        total      = int(row["total"]) if row else 0
        healthy_n  = int(row["healthy_n"] or 0)
        avg_conf   = float(row["avg_conf"] or 0.0)
        healthy_pct = (healthy_n / total) if total else 0.0

        # ── 2. Calendar: counts per day for the last 28 days (UTC).
        #     SQLite stores created_at as ISO 8601 strings; substr(.,1,10) extracts YYYY-MM-DD.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=27))\
                    .replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        cal_rows = c.execute(
            """SELECT SUBSTR(created_at, 1, 10) AS d, COUNT(*) AS n
               FROM predictions
               WHERE user_id = ? AND created_at >= ?
               GROUP BY d ORDER BY d ASC""",
            (user_id, cutoff),
        ).fetchall()
        by_day = {r["d"]: int(r["n"]) for r in cal_rows}

        # Build a complete 28-day series even for days the user didn't log
        # (zero-fill).  Oldest first → newest last.
        today = datetime.now(timezone.utc).date()
        calendar = []
        for back in range(27, -1, -1):
            d = today - timedelta(days=back)
            key = d.isoformat()
            calendar.append({"date": key, "count": by_day.get(key, 0)})

        # ── 3. Streak: consecutive days ending today (or yesterday) with ≥1 log
        streak = 0
        cur = today
        for back in range(0, 60):  # cap streak search to 60 days
            d = today - timedelta(days=back)
            if by_day.get(d.isoformat(), 0) > 0:
                streak += 1
            else:
                # Allow the streak to start "yesterday" if today is empty,
                # but break the first time we hit a zero after counting started.
                if streak > 0:
                    break
                if back > 0:
                    break

        # ── 4. Disease ledger: top predicted_class for this user
        ledger_rows = c.execute(
            """SELECT predicted_class AS cls, crop, COUNT(*) AS n
               FROM predictions
               WHERE user_id = ? AND predicted_class IS NOT NULL
               GROUP BY predicted_class
               ORDER BY n DESC LIMIT 8""",
            (user_id,),
        ).fetchall()
        ledger = [
            {"class": r["cls"], "crop": r["crop"], "count": int(r["n"])}
            for r in ledger_rows
        ]

        # ── 5. Crop distribution (drives avatar selection)
        crop_rows = c.execute(
            """SELECT crop, COUNT(*) AS n
               FROM predictions
               WHERE user_id = ? AND crop IS NOT NULL AND crop != ''
               GROUP BY crop ORDER BY n DESC""",
            (user_id,),
        ).fetchall()
        crop_mix = [{"crop": r["crop"], "count": int(r["n"])} for r in crop_rows]
        dominant_crop = crop_mix[0]["crop"] if crop_mix else None

        # ── 6. Recent: last 6 predictions, slim columns + image flags
        # has_image / has_heatmap tell the frontend whether to render a
        # real <img> tag (pointing at /dashboard/predictions/{id}/image)
        # or the honest "image not captured" pre-upgrade placeholder.
        recent_rows = c.execute(
            """SELECT id, crop, predicted_class, confidence, tier, created_at,
                      (image_bytes IS NOT NULL) AS has_image,
                      (heatmap_b64 IS NOT NULL) AS has_heatmap
               FROM predictions
               WHERE user_id = ?
               ORDER BY created_at DESC LIMIT 6""",
            (user_id,),
        ).fetchall()
        recent = []
        for r in recent_rows:
            d = dict(r)
            d["has_image"]   = bool(d.get("has_image"))
            d["has_heatmap"] = bool(d.get("has_heatmap"))
            recent.append(d)

    return {
        "user": {
            "id": user.get("id"),
            "username": user.get("username"),
            "display_name": user.get("display_name"),
            "pressed_leaf_seed": user.get("pressed_leaf_seed"),
            "created_at": user.get("created_at"),
        },
        "hero": {
            "total_predictions": total,
            "healthy_pct": round(healthy_pct, 3),
            "avg_confidence": round(avg_conf, 3),
            "streak_days": int(streak),
        },
        "calendar": calendar,
        "ledger": ledger,
        "crop_mix": crop_mix,
        "dominant_crop": dominant_crop,
        "recent": recent,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase-1 widget queries
# ───────────────────────
# These helpers power the multi-widget dashboard (Daily Brief, Crop Almanac,
# Confidence Distribution histogram) + the /dashboard/history page (Ledger
# view with filters / pagination / export). Each function is *standalone* —
# do NOT couple them to get_dashboard_data() above so that future widgets can
# pull just the slice they need without recomputing the whole dashboard blob.
# ═══════════════════════════════════════════════════════════════════════════

# ─── Allowed filter tokens (defensive whitelist) ──────────────────────────────
_ALLOWED_CROPS = {"okra", "brassica", "tomato", "chilli"}
_ALLOWED_TIERS = {
    "FIELD_GRADE", "LAB_GRADE", "UNCERTAIN", "OOD",
    # We also accept lower-case variants since tier values in older payloads
    # are not consistently cased.
    "field_grade", "lab_grade", "uncertain", "ood",
}


def _build_history_where(
    *,
    user_id: int,
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    tier: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
) -> tuple[str, list]:
    """Construct a parameterised WHERE clause from optional filters.

    Returns (where_sql, params). The WHERE always starts with "WHERE user_id = ?"
    so subsequent clauses can append " AND ..." safely. All user-supplied
    values are passed as bound params; column names and operators are fixed
    literal strings so this is SQL-injection-safe.
    """
    where = ["user_id = ?"]
    params: list = [user_id]

    if crop and crop in _ALLOWED_CROPS:
        where.append("crop = ?")
        params.append(crop)

    if disease:
        # Disease is a class string like "tomato_early_blight" — exact match.
        where.append("predicted_class = ?")
        params.append(disease)

    if tier and tier in _ALLOWED_TIERS:
        # Tier is stored as the canonical (upper-case) form; normalise on read.
        where.append("UPPER(tier) = ?")
        params.append(tier.upper())

    if date_from:
        # Accepts YYYY-MM-DD; the substring comparison on created_at (ISO 8601)
        # is monotonic so this works without parsing.
        where.append("SUBSTR(created_at, 1, 10) >= ?")
        params.append(date_from)

    if date_to:
        where.append("SUBSTR(created_at, 1, 10) <= ?")
        params.append(date_to)

    if search:
        # Free-text LIKE on predicted_class for now (e.g. "blight" matches
        # tomato_early_blight + tomato_late_blight). Anchored with %; we cap
        # the search string to 60 chars at the route layer to avoid abuse.
        where.append("LOWER(predicted_class) LIKE ?")
        params.append("%" + search.lower() + "%")

    return ("WHERE " + " AND ".join(where), params)


_HISTORY_SORTS = {
    "newest":  "created_at DESC",
    "oldest":  "created_at ASC",
    "highest": "confidence DESC",
    "lowest":  "confidence ASC",
}


def list_predictions(
    user_id: int,
    *,
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    tier: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
    sort: str = "newest",
    page: int = 1,
    page_size: int = 25,
    max_page_size: int = 200,
) -> list[dict]:
    """Paginated, filterable predictions list for the Field History page.

    Returns slim rows (no response_json); use `get_prediction` for full record.
    All filter params are optional and are silently dropped if they aren't in
    the whitelist (defensive — invalid input becomes "no filter").

    sort:           one of "newest" | "oldest" | "highest" | "lowest"
    page:           1-based page number
    page_size:      capped to [1, max_page_size]
    max_page_size:  hard ceiling (default 200 — Field History UI uses 25,
                    but the Disease Drill-down modal asks for 200 because
                    the mini-timeline needs every dot, not just one page).
    """
    page = max(1, int(page))
    page_size = max(1, min(int(max_page_size), int(page_size)))
    offset = (page - 1) * page_size

    order_by = _HISTORY_SORTS.get(sort, _HISTORY_SORTS["newest"])
    where_sql, params = _build_history_where(
        user_id=user_id, crop=crop, disease=disease, tier=tier,
        date_from=date_from, date_to=date_to, search=search,
    )

    sql = (
        "SELECT id, crop, predicted_class, confidence, tier, "
        "       image_sha256, created_at, "
        "       (image_bytes IS NOT NULL) AS has_image, "
        "       (heatmap_b64 IS NOT NULL) AS has_heatmap "
        "FROM predictions "
        f"{where_sql} "
        f"ORDER BY {order_by} "
        "LIMIT ? OFFSET ?"
    )
    params2 = list(params) + [page_size, offset]
    with get_conn() as c:
        rows = c.execute(sql, params2).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["has_image"]   = bool(d.get("has_image"))
            d["has_heatmap"] = bool(d.get("has_heatmap"))
            out.append(d)
        return out


def count_predictions(
    user_id: int,
    *,
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    tier: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
) -> int:
    """Count predictions matching the same filter shape as list_predictions().
    Used for pagination math (total / page_size = page count)."""
    where_sql, params = _build_history_where(
        user_id=user_id, crop=crop, disease=disease, tier=tier,
        date_from=date_from, date_to=date_to, search=search,
    )
    sql = f"SELECT COUNT(*) AS n FROM predictions {where_sql}"
    with get_conn() as c:
        row = c.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0


def get_prediction(prediction_id: int, *, user_id: int) -> Optional[dict]:
    """Single full prediction record (including response_json).
    Scoped to user_id so users can't read each other's predictions even
    by ID guessing. Returns None if not found or not owned."""
    # Re-exported as a clean alias for the existing helper to keep the public
    # API of this module aligned with the widget naming.
    return get_prediction_full(prediction_id, user_id=user_id)


def aggregate_by_disease(
    user_id: int,
    *,
    crop: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Per-disease counts for this user.

    Returns [{class, crop, count, first_seen, last_seen, avg_confidence}, ...]
    ordered by count DESC.  Used by the Disease Ledger, Field Notebook Index,
    Disease Drill-down modal, and Daily Brief generator.

    `crop` filter: only classes belonging to that crop.
    `date_from` / `date_to` filters: restricts the time window for the count;
    useful for "this week's top disease" calculations (date strings YYYY-MM-DD).

    Each row's `crop` field is the user's MOST FREQUENT crop for that class
    (not the lexicographic max — SQLite's MAX(crop) used to pick 'tomato'
    over 'brassica' even when brassica was more common for that class)."""
    limit = max(1, min(200, int(limit)))
    # Build the WHERE clause directly with the p1. prefix so we don't have
    # to retrofit prefixes later (which would clash with crop = ? after a
    # naive replace).
    where = ["p1.user_id = ?", "p1.predicted_class IS NOT NULL"]
    params: list = [user_id]
    if crop and crop in _ALLOWED_CROPS:
        where.append("p1.crop = ?")
        params.append(crop)
    if date_from:
        where.append("SUBSTR(p1.created_at, 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        where.append("SUBSTR(p1.created_at, 1, 10) <= ?")
        params.append(date_to)
    where_sql = "WHERE " + " AND ".join(where)

    # Use a correlated subquery to fetch the most-frequent crop per class
    # rather than MAX(crop) which returns the alphabetically largest value.
    # SQLite's query planner handles the inner GROUP BY efficiently because
    # we filter on (user_id, predicted_class) which is covered by the
    # idx_predictions_user_date index.
    sql = (
        "SELECT p1.predicted_class AS class, "
        "       (SELECT p2.crop FROM predictions p2 "
        "         WHERE p2.user_id = p1.user_id "
        "           AND p2.predicted_class = p1.predicted_class "
        "           AND p2.crop IS NOT NULL "
        "         GROUP BY p2.crop ORDER BY COUNT(*) DESC LIMIT 1) AS crop, "
        "       COUNT(*)            AS count, "
        "       MIN(p1.created_at)  AS first_seen, "
        "       MAX(p1.created_at)  AS last_seen, "
        "       AVG(p1.confidence)  AS avg_confidence "
        "FROM predictions p1 "
        f"{where_sql} "
        "GROUP BY p1.predicted_class "
        "ORDER BY count DESC "
        "LIMIT ?"
    )
    with get_conn() as c:
        rows = c.execute(sql, list(params) + [limit]).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["count"] = int(d["count"])
            d["avg_confidence"] = round(float(d["avg_confidence"] or 0.0), 3)
            out.append(d)
        return out


def aggregate_by_day(user_id: int, *, window_days: int = 28) -> list[dict]:
    """Per-day prediction counts ending today, oldest first, zero-filled.

    Identical shape to the `calendar` array in get_dashboard_data() but
    parameterised by window length so a future "Comparison Spread" widget
    can request 7 or 60 days. Zero-filling keeps the array length stable
    regardless of how active the user has been."""
    from datetime import datetime, timezone, timedelta
    window_days = max(1, min(365, int(window_days)))
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=window_days - 1)
    ).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    with get_conn() as c:
        rows = c.execute(
            "SELECT SUBSTR(created_at, 1, 10) AS d, COUNT(*) AS n "
            "FROM predictions "
            "WHERE user_id = ? AND created_at >= ? "
            "GROUP BY d ORDER BY d ASC",
            (user_id, cutoff),
        ).fetchall()
        by_day = {r["d"]: int(r["n"]) for r in rows}

    today = datetime.now(timezone.utc).date()
    out = []
    for back in range(window_days - 1, -1, -1):
        d = today - timedelta(days=back)
        key = d.isoformat()
        out.append({"date": key, "count": by_day.get(key, 0)})
    return out


def confidence_histogram(
    user_id: int,
    *,
    bins: int = 10,
) -> list[dict]:
    """Histogram of prediction confidences for this user.

    Bins span [0, 1] uniformly. Each entry is {lo, hi, count, label}.
    `label` is a pre-formatted "0.70–0.80" string for the chart legend.
    `bins` capped to [4, 20] — narrower bins look noisy on small datasets,
    wider bins lose resolution."""
    bins = max(4, min(20, int(bins)))
    edges = [round(i / bins, 4) for i in range(bins + 1)]
    out = [
        {"lo": edges[i], "hi": edges[i + 1], "count": 0,
         "label": f"{edges[i]:.2f}–{edges[i + 1]:.2f}"}
        for i in range(bins)
    ]
    with get_conn() as c:
        rows = c.execute(
            "SELECT confidence FROM predictions "
            "WHERE user_id = ? AND confidence IS NOT NULL",
            (user_id,),
        ).fetchall()
    for r in rows:
        c_val = float(r["confidence"])
        if c_val < 0.0: c_val = 0.0
        if c_val > 1.0: c_val = 1.0
        # Map to bin index; the upper edge (1.0) goes into the last bin.
        idx = min(bins - 1, int(c_val * bins))
        out[idx]["count"] += 1
    return out


def first_sightings(user_id: int) -> list[dict]:
    """First-sighting date per disease class for this user.

    Returns [{class, crop, first_seen, count}, ...] ordered by first_seen ASC.
    Powers the First Sightings phenology timeline (Container B mode 2). Only
    classes the user has actually logged are returned — empty list if the
    user has no predictions yet.

    `crop` is the most-frequent crop for the class (not MAX(crop) which would
    pick the alphabetically largest value)."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT p1.predicted_class AS class, "
            "       (SELECT p2.crop FROM predictions p2 "
            "         WHERE p2.user_id = p1.user_id "
            "           AND p2.predicted_class = p1.predicted_class "
            "           AND p2.crop IS NOT NULL "
            "         GROUP BY p2.crop ORDER BY COUNT(*) DESC LIMIT 1) AS crop, "
            "       MIN(p1.created_at)    AS first_seen, "
            "       COUNT(*)              AS count "
            "FROM predictions p1 "
            "WHERE p1.user_id = ? AND p1.predicted_class IS NOT NULL "
            "GROUP BY p1.predicted_class "
            "ORDER BY first_seen ASC",
            (user_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "class":      r["class"],
            "crop":       r["crop"],
            "first_seen": r["first_seen"],
            "count":      int(r["count"]),
        })
    return out


def list_user_crops(user_id: int) -> list[str]:
    """Return distinct crops the user has logged, alphabetical.

    Used by /dashboard/history/data to populate the crop-filter dropdown
    without paying for the 6-query `get_dashboard_data()` round-trip."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT DISTINCT crop FROM predictions "
            "WHERE user_id = ? AND crop IS NOT NULL AND crop <> '' "
            "ORDER BY crop",
            (user_id,),
        ).fetchall()
    return [r["crop"] for r in rows]


# ═══════════════════════════════════════════════════════════════════════════
# Phase-2 helpers
# ───────────────
# Margin notes CRUD (Container F), full-prediction-with-signals fetch
# (Signal Vote modal), and disease taxonomy lookup (Family Tree widget).
# ═══════════════════════════════════════════════════════════════════════════

# ─── Margin Notes CRUD ────────────────────────────────────────────────────

def list_margin_notes(
    user_id: int,
    *,
    attached_date: Optional[str] = None,
    attached_prediction_id: Optional[int] = None,
    limit: int = 200,
) -> list[dict]:
    """Return this user's notes, newest first. Optional filters narrow to
    notes attached to a specific date or specific prediction."""
    limit = max(1, min(500, int(limit)))
    where = ["user_id = ?"]
    params: list = [user_id]
    if attached_date:
        where.append("attached_date = ?")
        params.append(attached_date)
    if attached_prediction_id is not None:
        where.append("attached_prediction_id = ?")
        params.append(int(attached_prediction_id))
    where_sql = "WHERE " + " AND ".join(where)
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, text, attached_date, attached_prediction_id, "
            "       mood, created_at, updated_at "
            f"FROM margin_notes {where_sql} "
            "ORDER BY created_at DESC LIMIT ?",
            list(params) + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def create_margin_note(
    user_id: int,
    *,
    text: str,
    attached_date: Optional[str] = None,
    attached_prediction_id: Optional[int] = None,
    mood: Optional[int] = None,
) -> dict:
    """Insert a new note for this user. Validates that text is non-empty
    and that at most one attachment target is provided (date XOR prediction).
    Returns the inserted row as a dict."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Note text cannot be empty")
    if len(text) > 2000:
        text = text[:2000]  # generous cap — UI limits to ~280
    if attached_date and attached_prediction_id is not None:
        raise ValueError(
            "Note must attach to either a date OR a prediction, not both"
        )
    if mood is not None:
        try: mood = int(mood)
        except (TypeError, ValueError): mood = None
        if mood is not None and (mood < 0 or mood > 3):
            mood = None
    now = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "INSERT INTO margin_notes "
            "  (user_id, text, attached_date, attached_prediction_id, "
            "   mood, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, text, attached_date, attached_prediction_id, mood, now),
        )
        new_id = cur.lastrowid
        row = c.execute(
            "SELECT id, text, attached_date, attached_prediction_id, "
            "       mood, created_at, updated_at "
            "FROM margin_notes WHERE id = ?",
            (new_id,),
        ).fetchone()
    return dict(row) if row else {}


def update_margin_note(
    note_id: int,
    *,
    user_id: int,
    text: Optional[str] = None,
    mood: Optional[int] = None,
) -> Optional[dict]:
    """Update an existing note. Scoped to user_id so users can't edit
    each other's notes. Returns the updated row, or None if not found."""
    sets = []
    params: list = []
    if text is not None:
        t = text.strip()
        if not t:
            raise ValueError("Note text cannot be empty")
        sets.append("text = ?")
        params.append(t[:2000])
    if mood is not None:
        try: mood = int(mood)
        except (TypeError, ValueError): mood = None
        if mood is not None and (mood < 0 or mood > 3):
            mood = None
        # Only persist mood if the value is still valid after sanitisation.
        # Out-of-range / non-numeric input on PATCH must NOT silently
        # overwrite the existing stored mood with NULL.
        if mood is not None:
            sets.append("mood = ?")
            params.append(mood)
    if not sets:
        return get_margin_note(note_id, user_id=user_id)
    sets.append("updated_at = ?")
    params.append(_now_iso())
    params.extend([note_id, user_id])
    with _write_lock, get_conn() as c:
        c.execute(
            f"UPDATE margin_notes SET {', '.join(sets)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
    return get_margin_note(note_id, user_id=user_id)


def delete_margin_note(note_id: int, *, user_id: int) -> bool:
    """Delete a note scoped to user_id. Returns True if a row was removed."""
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "DELETE FROM margin_notes WHERE id = ? AND user_id = ?",
            (note_id, user_id),
        )
        return cur.rowcount > 0


def get_margin_note(note_id: int, *, user_id: int) -> Optional[dict]:
    """Single note fetch scoped to user."""
    with get_conn() as c:
        row = c.execute(
            "SELECT id, text, attached_date, attached_prediction_id, "
            "       mood, created_at, updated_at "
            "FROM margin_notes WHERE id = ? AND user_id = ?",
            (note_id, user_id),
        ).fetchone()
    return dict(row) if row else None


# ─── Full prediction with parsed signal votes (for Signal Vote modal) ────

def get_prediction_with_signals(prediction_id: int, *, user_id: int) -> Optional[dict]:
    """Return a prediction row with the response_json parsed into structured
    signal votes for display in the Signal Vote Card modal.

    The response_json stored at prediction time is the raw APIN ensemble
    output. It may contain different shapes depending on the model versions
    active when the prediction was made; we try to extract the four-signal
    breakdown (Model 2 ConvNeXt / EfficientNet / DINOv2 / PSV) plus the
    final ensemble decision. Missing fields are returned as None so the UI
    can show a placeholder gracefully."""
    import json as _json
    pred = get_prediction_full(prediction_id, user_id=user_id)
    if not pred:
        return None
    # Parse response_json safely
    parsed = {}
    try:
        if pred.get("response_json"):
            parsed = _json.loads(pred["response_json"])
            if not isinstance(parsed, dict):
                parsed = {}
    except (TypeError, ValueError):
        parsed = {}
    # Try multiple shape variants because different model generations
    # used different field names. We never crash — missing → None.
    # Phase-3.5 — the okra/brassica pipeline emits `signal_predictions`
    # with shape {model2: {argmax, top_prob}, efficientnet: {…}, dinov2: …}
    # — neither `signals` nor `signal_votes` is the canonical key.  We try
    # all known variants below.
    signals = (parsed.get("signal_predictions")
               or parsed.get("signals")
               or parsed.get("signal_votes")
               or {})
    def _sig(key_candidates):
        for k in key_candidates:
            if isinstance(signals, dict) and k in signals:
                return _normalize_signal_payload(signals[k])
        return None
    pred["parsed_signals"] = {
        "model2":       _sig(["model2", "m2", "convnext", "model2_convnext"]),
        "efficientnet": _sig(["efficientnet", "effnet", "eff_b0"]),
        "dinov2":       _sig(["dinov2_head", "dinov2", "dino", "psv_dinov2"]),
        "psv":          _sig(["psv", "prototype_similarity", "psv_score"]),
        "moe_gate":     parsed.get("moe_gate") or parsed.get("gate"),
        "ensemble":     _normalize_signal_payload(
                          parsed.get("ensemble") or parsed.get("final")
                        ),
    }
    return pred


def _normalize_signal_payload(v):
    """Map any of the per-signal shapes the inference pipeline emits to
    the {confidence, vote} shape the Day-Detail bars renderer reads.

    Accepted inputs:
      None                                     → None
      number                                   → {confidence: n, vote: None}
      {argmax, top_prob}     (okra/brassica)   → {vote: argmax, confidence: top_prob}
      {class, confidence}    (legacy)          → unchanged
      {predicted_class, score}                 → mapped
      anything else                            → returned as-is (UI tolerates)
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return {"confidence": float(v), "vote": None}
    if isinstance(v, dict):
        out = dict(v)  # don't mutate the caller's object
        # Confidence aliases
        if "confidence" not in out:
            for k in ("top_prob", "score", "prob", "probability", "top_p"):
                if k in out and isinstance(out[k], (int, float)):
                    out["confidence"] = float(out[k])
                    break
        # Vote aliases — covers argmax, class, predicted_class, label
        if "vote" not in out:
            for k in ("argmax", "class", "predicted_class", "label",
                      "winning_class", "top_class"):
                if k in out and isinstance(out[k], str):
                    out["vote"] = out[k]
                    break
        return out
    return v


# ─── Disease taxonomy lookup (for Disease Family Tree widget) ────────────
# Static lookup table — pathogen kingdom → genus → species.
# We use the same disease class names as DISEASE_ENCYCLOPEDIA in
# dashboard.html so cross-references work without translation.

DISEASE_TAXONOMY = {
    # Okra
    "okra_yvmv":                     {"kingdom": "virus",  "genus": "Begomovirus",       "species": "yellow-vein mosaic strain"},
    "okra_enation":                  {"kingdom": "virus",  "genus": "Begomovirus",       "species": "enation leaf-curl strain"},
    "okra_powdery_mildew":           {"kingdom": "fungi",  "genus": "Erysiphe",          "species": "cichoracearum"},
    "okra_cercospora":               {"kingdom": "fungi",  "genus": "Cercospora",        "species": "abelmoschi"},
    "okra_healthy":                  {"kingdom": "n/a",    "genus": "Abelmoschus",       "species": "esculentus"},
    # Brassica
    "brassica_black_rot":            {"kingdom": "bacteria","genus": "Xanthomonas",       "species": "campestris"},
    "brassica_downy_mildew":         {"kingdom": "oomycota","genus": "Hyaloperonospora",  "species": "parasitica"},
    "brassica_alternaria":           {"kingdom": "fungi",  "genus": "Alternaria",        "species": "brassicicola"},
    "brassica_clubroot":             {"kingdom": "protist","genus": "Plasmodiophora",    "species": "brassicae"},
    "brassica_healthy":              {"kingdom": "n/a",    "genus": "Brassica",          "species": "oleracea"},
    # Tomato
    "tomato_early_blight":           {"kingdom": "fungi",  "genus": "Alternaria",        "species": "solani"},
    "tomato_late_blight":            {"kingdom": "oomycota","genus": "Phytophthora",      "species": "infestans"},
    "tomato_septoria_leaf_spot":     {"kingdom": "fungi",  "genus": "Septoria",          "species": "lycopersici"},
    "tomato_septoria":               {"kingdom": "fungi",  "genus": "Septoria",          "species": "lycopersici"},
    "tomato_target_spot":            {"kingdom": "fungi",  "genus": "Corynespora",       "species": "cassiicola"},
    "tomato_bacterial_spot":         {"kingdom": "bacteria","genus": "Xanthomonas",       "species": "perforans"},
    "tomato_leaf_mold":              {"kingdom": "fungi",  "genus": "Passalora",         "species": "fulva"},
    "tomato_yellow_leaf_curl_virus": {"kingdom": "virus",  "genus": "Begomovirus",       "species": "tomato yellow leaf-curl virus"},
    "tomato_yellow_leaf_curl":       {"kingdom": "virus",  "genus": "Begomovirus",       "species": "tomato yellow leaf-curl virus"},
    "tomato_mosaic_virus":           {"kingdom": "virus",  "genus": "Tobamovirus",       "species": "tomato mosaic virus"},
    "tomato_foliar_spot":            {"kingdom": "fungi",  "genus": "(various)",         "species": "foliar pathogen complex"},
    "tomato_healthy":                {"kingdom": "n/a",    "genus": "Solanum",           "species": "lycopersicum"},
    # Chilli
    "chilli_anthracnose":            {"kingdom": "fungi",  "genus": "Colletotrichum",    "species": "gloeosporioides"},
    "chilli_cercospora_leaf_spot":   {"kingdom": "fungi",  "genus": "Cercospora",        "species": "capsici"},
    "chilli_leaf_curl":              {"kingdom": "virus",  "genus": "Begomovirus",       "species": "chilli leaf-curl virus"},
    "chilli_healthy":                {"kingdom": "n/a",    "genus": "Capsicum",          "species": "annuum"},
}


def build_taxonomy_tree(user_id: int) -> dict:
    """Return a tree {kingdom: {genus: {species: [class_keys...]}}} restricted
    to disease classes the user has actually logged at least once.
    Each leaf class carries its count so the dendrogram can size labels."""
    agg = aggregate_by_disease(user_id, limit=200)
    counts = {d["class"]: d["count"] for d in agg}
    if not counts:
        return {}
    tree: dict = {}
    for cls, n in counts.items():
        tax = DISEASE_TAXONOMY.get(cls)
        if not tax:
            continue
        k = tax["kingdom"]; g = tax["genus"]; s = tax["species"]
        tree.setdefault(k, {}).setdefault(g, {}).setdefault(s, [])
        tree[k][g][s].append({"class": cls, "count": n})
    return tree


# ═══════════════════════════════════════════════════════════════════════════
# Phase-3 helpers — Treatment Log + Public Share Links
# ═══════════════════════════════════════════════════════════════════════════

# ─── Treatment Log CRUD ───────────────────────────────────────────────────

def list_treatments(
    user_id: int,
    *,
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    target_prediction_id: Optional[int] = None,
    limit: int = 500,
) -> list[dict]:
    """Return treatments for this user, newest applied_date first.

    Filter params follow the same defensive pattern as the rest of the
    module — invalid values are silently dropped (whitelist on crop;
    free-text disease/plot pass through to parameterised SQL)."""
    limit = max(1, min(2000, int(limit)))
    where = ["user_id = ?"]
    params: list = [user_id]
    if crop and crop in _ALLOWED_CROPS:
        where.append("crop = ?")
        params.append(crop)
    if disease:
        where.append("disease = ?")
        params.append(disease[:60])
    if date_from:
        where.append("applied_date >= ?")
        params.append(date_from)
    if date_to:
        where.append("applied_date <= ?")
        params.append(date_to)
    if target_prediction_id is not None:
        where.append("target_prediction_id = ?")
        params.append(int(target_prediction_id))
    where_sql = "WHERE " + " AND ".join(where)
    with get_conn() as c:
        rows = c.execute(
            f"SELECT id, treatment, crop, disease, plot, notes, "
            f"       target_prediction_id, applied_date, created_at "
            f"FROM treatment_log {where_sql} "
            f"ORDER BY applied_date DESC, created_at DESC LIMIT ?",
            list(params) + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def create_treatment(
    user_id: int,
    *,
    treatment: str,
    applied_date: str,
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    plot: Optional[str] = None,
    notes: Optional[str] = None,
    target_prediction_id: Optional[int] = None,
) -> dict:
    """Insert a treatment record."""
    treatment = (treatment or "").strip()
    if not treatment:
        raise ValueError("treatment text cannot be empty")
    if len(treatment) > 200:
        treatment = treatment[:200]
    applied_date = (applied_date or "").strip()[:10]
    if not applied_date:
        raise ValueError("applied_date is required (YYYY-MM-DD)")
    if crop and crop not in _ALLOWED_CROPS:
        crop = None
    if disease:
        disease = disease.strip()[:60] or None
    if plot:
        plot = plot.strip()[:60] or None
    if notes:
        notes = notes.strip()[:1000] or None
    now = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "INSERT INTO treatment_log "
            "(user_id, treatment, crop, disease, plot, notes, "
            " target_prediction_id, applied_date, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, treatment, crop, disease, plot, notes,
             target_prediction_id, applied_date, now),
        )
        new_id = cur.lastrowid
        row = c.execute(
            "SELECT * FROM treatment_log WHERE id = ?", (new_id,),
        ).fetchone()
    return dict(row) if row else {}


def update_treatment(
    treatment_id: int, *, user_id: int,
    treatment: Optional[str] = None,
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    plot: Optional[str] = None,
    notes: Optional[str] = None,
    applied_date: Optional[str] = None,
) -> Optional[dict]:
    sets, params = [], []
    if treatment is not None:
        t = treatment.strip()
        if not t: raise ValueError("treatment text cannot be empty")
        sets.append("treatment = ?"); params.append(t[:200])
    if crop is not None:
        # Allow empty string to clear; whitelist guard otherwise
        v = crop or None
        if v and v not in _ALLOWED_CROPS: v = None
        sets.append("crop = ?"); params.append(v)
    if disease is not None:
        sets.append("disease = ?"); params.append((disease or "").strip()[:60] or None)
    if plot is not None:
        sets.append("plot = ?"); params.append((plot or "").strip()[:60] or None)
    if notes is not None:
        sets.append("notes = ?"); params.append((notes or "").strip()[:1000] or None)
    if applied_date is not None:
        sets.append("applied_date = ?"); params.append((applied_date or "").strip()[:10])
    if not sets:
        return get_treatment(treatment_id, user_id=user_id)
    params.extend([treatment_id, user_id])
    with _write_lock, get_conn() as c:
        c.execute(
            f"UPDATE treatment_log SET {', '.join(sets)} "
            f"WHERE id = ? AND user_id = ?",
            params,
        )
    return get_treatment(treatment_id, user_id=user_id)


def delete_treatment(treatment_id: int, *, user_id: int) -> bool:
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "DELETE FROM treatment_log WHERE id = ? AND user_id = ?",
            (treatment_id, user_id),
        )
        return cur.rowcount > 0


def get_treatment(treatment_id: int, *, user_id: int) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute(
            "SELECT * FROM treatment_log WHERE id = ? AND user_id = ?",
            (treatment_id, user_id),
        ).fetchone()
    return dict(row) if row else None


# ─── Public Share Tokens ──────────────────────────────────────────────────
# A share token is a 32-byte url-safe random string. The DB stores only the
# sha256(token) — the raw token is returned to the caller ONCE at creation
# time. Public viewers hit /share/{token} which hashes the token, looks it
# up, increments view_count, and returns the prediction (if not revoked /
# not expired).

def create_share_token(
    user_id: int,
    prediction_id: int,
    *,
    label: Optional[str] = None,
    expires_at: Optional[str] = None,
) -> dict:
    """Generate a new share token for the given prediction.

    Returns {"id", "token", "url_suffix", "created_at", ...} — `token` is
    the RAW token visible exactly once. Caller must verify the prediction
    is owned by user_id BEFORE calling this.
    """
    # Reuse the session-token plumbing — 32 bytes urlsafe + sha256 stored
    raw, h = _new_session_token()
    label = (label or "").strip()[:120] or None
    if expires_at is not None:
        expires_at = expires_at.strip()[:32] or None
    now = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "INSERT INTO share_tokens "
            "(user_id, prediction_id, token_hash, label, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, prediction_id, h, label, now, expires_at),
        )
        new_id = cur.lastrowid
    return {
        "id":          new_id,
        "token":       raw,        # ONLY time the raw token is exposed
        "url_suffix":  raw,        # convenience alias
        "label":       label,
        "prediction_id": prediction_id,
        "created_at":  now,
        "expires_at":  expires_at,
        "revoked_at":  None,
        "view_count":  0,
    }


def list_share_tokens(user_id: int) -> list[dict]:
    """Return all share-token records for a user. Does NOT include token
    hashes — those are write-only. Used to populate the Reports page's
    share-link management list."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, prediction_id, label, created_at, expires_at, "
            "       revoked_at, last_viewed_at, view_count "
            "FROM share_tokens WHERE user_id = ? "
            "ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_share_token(token_id: int, *, user_id: int) -> bool:
    """Soft-revoke a share token. Returns True if the row was updated."""
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "UPDATE share_tokens SET revoked_at = ? "
            "WHERE id = ? AND user_id = ? AND revoked_at IS NULL",
            (_now_iso(), token_id, user_id),
        )
        return cur.rowcount > 0


def resolve_share_token(raw_token: str,
                        count_view: bool = True) -> Optional[dict]:
    """Public-facing token lookup. Hashes the raw token, validates not
    revoked / not expired, and returns the full prediction record so the
    public view can render. Returns None if the token is invalid, revoked,
    or expired.

    view_count + last_viewed_at are bumped only when `count_view` is True.
    The share-data route passes count_view=False on same-browser refreshes
    so a viewer reloading the page does not inflate the owner's count."""
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    now = _now_iso()
    with _write_lock, get_conn() as c:
        row = c.execute(
            "SELECT id, user_id, prediction_id, label, created_at, "
            "       expires_at, revoked_at, view_count "
            "FROM share_tokens WHERE token_hash = ?",
            (h,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d["revoked_at"]:                          return None
        if d["expires_at"] and d["expires_at"] < now: return None
        # Increment view counters only for counted views (best-effort).
        if count_view:
            c.execute(
                "UPDATE share_tokens SET view_count = view_count + 1, "
                "       last_viewed_at = ? WHERE id = ?",
                (now, d["id"]),
            )
    # Fetch the prediction WITHOUT signal parsing — the public share view
    # only renders class/crop/tier/confidence/date, never per-signal votes.
    # Using get_prediction_full + popping response_json keeps the wire
    # payload minimal and prevents leaking parsed_signals (which would
    # otherwise be derived from response_json's internals).  Code-review
    # finding #2: a previous version used get_prediction_with_signals and
    # only popped response_json, leaving parsed_signals exposed.
    pred = get_prediction_full(d["prediction_id"], user_id=d["user_id"])
    if pred is None:
        return None
    pred["share"] = {
        "label":      d["label"],
        "created_at": d["created_at"],
        "view_count": (d["view_count"] or 0) + (1 if count_view else 0),
    }
    # Strip server-only fields before returning to the public viewer.
    # Round-2 PDA finding: a public share should expose ONLY agronomy facts.
    # `user_id` enables account enumeration; `id` leaks the monotonic
    # prediction primary key (lets an outsider count predictions in the
    # system); `image_sha256` is an internal storage handle. None of these
    # are read by share.html's renderer, so popping them is purely additive.
    pred.pop("response_json",  None)
    pred.pop("parsed_signals", None)  # defensive — should not be present
    pred.pop("user_id",        None)
    pred.pop("id",             None)
    pred.pop("image_sha256",   None)
    return pred


# ═══════════════════════════════════════════════════════════════════════════
# Weekly report persistence
# A generated weekly PDF is stored once and reused. Delete is soft so the
# undo toast can restore it. The partial unique index on (user_id,
# week_start) WHERE deleted_at IS NULL keeps at most one ACTIVE report per
# week while allowing a deleted week to be generated again.
# ═══════════════════════════════════════════════════════════════════════════

def save_report(user_id: int, *, week_start: str, week_end: str,
                pdf_bytes: bytes, summary: Optional[dict] = None) -> int:
    """Persist a generated weekly PDF. Any existing ACTIVE report for the
    same (user, week_start) is soft-deleted first, so a generate always
    yields exactly one fresh active row. Returns the new report id."""
    if not pdf_bytes:
        raise ValueError("pdf_bytes cannot be empty")
    week_start = (week_start or "").strip()[:10]
    week_end = (week_end or "").strip()[:10]
    if not week_start or not week_end:
        raise ValueError("week_start and week_end are required")
    blob = bytes(pdf_bytes)
    summary_json = json.dumps(summary) if summary is not None else None
    now = _now_iso()
    with _write_lock, get_conn() as c:
        c.execute(
            "UPDATE reports SET deleted_at = ? "
            "WHERE user_id = ? AND week_start = ? AND deleted_at IS NULL",
            (now, int(user_id), week_start),
        )
        cur = c.execute(
            "INSERT INTO reports (user_id, week_start, week_end, pdf_bytes, "
            "                     summary_json, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (int(user_id), week_start, week_end, blob, summary_json, now),
        )
        return int(cur.lastrowid)


def list_reports(user_id: int) -> list[dict]:
    """Active (non-deleted) reports for a user, newest week first.
    Excludes the heavy pdf_bytes BLOB; parses summary_json into `summary`."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, week_start, week_end, summary_json, generated_at "
            "FROM reports WHERE user_id = ? AND deleted_at IS NULL "
            "ORDER BY week_start DESC, id DESC",
            (int(user_id),),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        raw = d.pop("summary_json", None)
        try:
            d["summary"] = json.loads(raw) if raw else None
        except Exception:
            d["summary"] = None
        out.append(d)
    return out


def get_report_meta(report_id: int, *, user_id: int) -> Optional[dict]:
    """Lightweight metadata for one ACTIVE report (no BLOB). Ownership-gated."""
    with get_conn() as c:
        row = c.execute(
            "SELECT id, week_start, week_end, generated_at FROM reports "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            (int(report_id), int(user_id)),
        ).fetchone()
    return dict(row) if row else None


def get_report_pdf(report_id: int, *, user_id: int) -> Optional[bytes]:
    """Return the stored PDF bytes for an ACTIVE report owned by user_id.
    None if not found, wrong owner, or deleted. The SQL's
    `id = ? AND user_id = ?` makes this IDOR-safe."""
    with get_conn() as c:
        row = c.execute(
            "SELECT pdf_bytes FROM reports "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            (int(report_id), int(user_id)),
        ).fetchone()
    if row is None:
        return None
    data = row["pdf_bytes"]
    if isinstance(data, memoryview):
        data = bytes(data)
    return data


def soft_delete_report(report_id: int, *, user_id: int) -> bool:
    """Soft-delete a report (set deleted_at). Returns True if an active
    report was found and deleted, False otherwise. Ownership-gated."""
    now = _now_iso()
    with _write_lock, get_conn() as c:
        row = c.execute(
            "SELECT id FROM reports "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            (int(report_id), int(user_id)),
        ).fetchone()
        if row is None:
            return False
        c.execute(
            "UPDATE reports SET deleted_at = ? "
            "WHERE id = ? AND user_id = ?",
            (now, int(report_id), int(user_id)))
        return True


def restore_report(report_id: int, *, user_id: int) -> bool:
    """Undo a soft delete. Returns False if the report is not found, is not
    deleted, or if another active report for the same week now exists
    (which would violate the partial unique index)."""
    with _write_lock, get_conn() as c:
        row = c.execute(
            "SELECT week_start FROM reports "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NOT NULL",
            (int(report_id), int(user_id)),
        ).fetchone()
        if row is None:
            return False
        clash = c.execute(
            "SELECT 1 FROM reports WHERE user_id = ? AND week_start = ? "
            "AND deleted_at IS NULL LIMIT 1",
            (int(user_id), row["week_start"]),
        ).fetchone()
        if clash is not None:
            return False
        c.execute(
            "UPDATE reports SET deleted_at = NULL "
            "WHERE id = ? AND user_id = ?",
            (int(report_id), int(user_id)))
        return True


def predictions_in_range(user_id: int, start_iso: str,
                         end_iso: str) -> list[dict]:
    """Predictions for a user with created_at in [start_iso, end_iso).
    Carries response_json (for severity / urgency parsing) and the
    has_image / has_heatmap flags, but NOT the heavy BLOBs. Newest first."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, crop, predicted_class, confidence, tier, "
            "       response_json, created_at, "
            "       (image_bytes IS NOT NULL) AS has_image, "
            "       (heatmap_b64 IS NOT NULL) AS has_heatmap "
            "FROM predictions "
            "WHERE user_id = ? AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at DESC",
            (int(user_id), start_iso, end_iso),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["has_image"] = bool(d.get("has_image"))
        d["has_heatmap"] = bool(d.get("has_heatmap"))
        out.append(d)
    return out


def weekly_prediction_counts(user_id: int) -> dict:
    """Map ISO Monday-week-start (YYYY-MM-DD) -> prediction count for a
    user. Buckets in Python so week alignment is portable across the
    sqlite and libSQL backends. Used by the Reports list."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT created_at FROM predictions WHERE user_id = ?",
            (int(user_id),),
        ).fetchall()
    counts: dict = {}
    for r in rows:
        ts = r["created_at"]
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            try:
                dt = datetime.strptime(str(ts)[:10], "%Y-%m-%d")
            except Exception:
                continue
        monday = (dt - timedelta(days=dt.weekday())).date()
        key = monday.isoformat()
        counts[key] = counts.get(key, 0) + 1
    return counts


# ─── Status monitoring — heartbeats + daily rollup ────────────────────────────
#
# These power the public /status and /health pages. Every helper here is
# written so a failure can NEVER take down a page or the monitor loop: the
# functions swallow their own exceptions and degrade to empty / safe results.

# Worst-of ranking. Lower number = worse. Used to fold many samples into one.
_STATUS_RANK = {"down": 0, "degraded": 1, "deg": 1, "up": 2, "operational": 2}

# Retention: raw heartbeats kept ~48h, daily rollups kept ~95 days.
_HEARTBEAT_RAW_HOURS = 48
_STATUS_DAYS_KEEP    = 95


def _worse(a: str, b: str) -> str:
    """Return whichever of two status strings is the worse one."""
    return a if _STATUS_RANK.get(a, 1) <= _STATUS_RANK.get(b, 1) else b


def record_heartbeat(overall: str, components: dict,
                     response_ms: Optional[int] = None) -> None:
    """Persist one monitoring snapshot.

    Writes a raw row to `heartbeats`, folds the snapshot into today's
    `status_days` rollup, and prunes stale rows. Meant to be called once a
    minute by a background task. Never raises — any failure is swallowed so
    the monitor loop keeps ticking.
    """
    try:
        now = _now_iso()
        day = now[:10]
        comp_json = json.dumps(components or {}, separators=(",", ":"))
        rms = int(response_ms) if response_ms is not None else None
        with _write_lock, get_conn() as c:
            # The raw insert + rollup upsert + prunes are one logical unit.
            # On local SQLite (autocommit) wrap them in an explicit
            # transaction so a crash mid-sequence cannot leave a heartbeat
            # row without its matching rollup update. If anything raises, the
            # connection is closed by get_conn()'s contextmanager, which
            # rolls the open transaction back. Turso keeps its per-statement
            # behaviour (no transaction shim dependency).
            _txn = not _USE_TURSO
            if _txn:
                c.execute("BEGIN")
            c.execute(
                "INSERT INTO heartbeats "
                "(recorded_at, overall, components, response_ms) "
                "VALUES (?, ?, ?, ?)",
                (now, overall, comp_json, rms))

            # Upsert today's rollup — read-merge-write, portable across the
            # sqlite and libSQL backends (no ON CONFLICT dependency).
            row = c.execute(
                "SELECT overall, components, samples, op_count, "
                "resp_ms_sum, resp_ms_n "
                "FROM status_days WHERE day = ?", (day,)).fetchone()
            if row is None:
                roll, day_overall = {}, overall
                samples = op_count = rsum = rn = 0
            else:
                try:
                    roll = json.loads(row["components"] or "{}")
                except Exception:
                    roll = {}
                day_overall = _worse(row["overall"] or "operational", overall)
                samples  = int(row["samples"] or 0)
                op_count = int(row["op_count"] or 0)
                rsum     = int(row["resp_ms_sum"] or 0)
                rn       = int(row["resp_ms_n"] or 0)

            for key, st in (components or {}).items():
                bucket = roll.get(key) or {"up": 0, "deg": 0, "down": 0}
                if st == "up":
                    bucket["up"] = int(bucket.get("up", 0)) + 1
                elif st == "down":
                    bucket["down"] = int(bucket.get("down", 0)) + 1
                else:
                    bucket["deg"] = int(bucket.get("deg", 0)) + 1
                roll[key] = bucket

            samples += 1
            if overall == "operational":
                op_count += 1
            if rms is not None:
                rsum += rms
                rn   += 1
            roll_json = json.dumps(roll, separators=(",", ":"))

            if row is None:
                c.execute(
                    "INSERT INTO status_days (day, overall, components, "
                    "samples, op_count, resp_ms_sum, resp_ms_n, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (day, day_overall, roll_json, samples, op_count,
                     rsum, rn, now))
            else:
                c.execute(
                    "UPDATE status_days SET overall = ?, components = ?, "
                    "samples = ?, op_count = ?, resp_ms_sum = ?, "
                    "resp_ms_n = ?, updated_at = ? WHERE day = ?",
                    (day_overall, roll_json, samples, op_count,
                     rsum, rn, now, day))

            # Prune — bounded, cheap deletes.
            cutoff_raw = (datetime.now(timezone.utc)
                          - timedelta(hours=_HEARTBEAT_RAW_HOURS)).isoformat()
            c.execute("DELETE FROM heartbeats WHERE recorded_at < ?",
                      (cutoff_raw,))
            cutoff_day = (datetime.now(timezone.utc)
                          - timedelta(days=_STATUS_DAYS_KEEP)).date().isoformat()
            c.execute("DELETE FROM status_days WHERE day < ?", (cutoff_day,))

            if _txn:
                c.execute("COMMIT")
    except Exception:
        # Monitoring must never crash the app.
        pass


def get_status_days(days: int = 90) -> list[dict]:
    """Return up to `days` daily-rollup rows, oldest first.

    Each row: {day, overall, components(dict), samples, op_count,
    resp_ms_avg}. Returns [] on any failure.
    """
    try:
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=days)).date().isoformat()
        with get_conn() as c:
            rows = c.execute(
                "SELECT day, overall, components, samples, op_count, "
                "resp_ms_sum, resp_ms_n FROM status_days "
                "WHERE day >= ? ORDER BY day ASC", (cutoff,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["components"] = json.loads(d.get("components") or "{}")
            except Exception:
                d["components"] = {}
            rn   = int(d.pop("resp_ms_n", 0) or 0)
            rsum = int(d.pop("resp_ms_sum", 0) or 0)
            d["resp_ms_avg"] = round(rsum / rn) if rn else None
            d["op_count"] = int(d.get("op_count", 0) or 0)
            out.append(d)
        return out
    except Exception:
        return []


def get_recent_heartbeats(hours: int = 24, limit: int = 600) -> list[dict]:
    """Return raw heartbeat rows from the last `hours`, newest first.
    Returns [] on any failure."""
    try:
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=hours)).isoformat()
        with get_conn() as c:
            rows = c.execute(
                "SELECT recorded_at, overall, components, response_ms "
                "FROM heartbeats WHERE recorded_at >= ? "
                "ORDER BY recorded_at DESC LIMIT ?",
                (cutoff, int(limit))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["components"] = json.loads(d.get("components") or "{}")
            except Exception:
                d["components"] = {}
            out.append(d)
        return out
    except Exception:
        return []


def count_all_predictions() -> int:
    """Global prediction count across every user. Returns 0 on failure."""
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM predictions").fetchone()
            return int(row["n"]) if row else 0
    except Exception:
        return 0


def status_db_probe() -> dict:
    """Cheap liveness probe for the database. Returns
    {ok, latency_ms, backend}. Never raises."""
    backend = "turso" if _USE_TURSO else "sqlite"
    t0 = datetime.now(timezone.utc)
    try:
        with get_conn() as c:
            c.execute("SELECT 1").fetchone()
        dt = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
        return {"ok": True, "latency_ms": round(dt), "backend": backend}
    except Exception as e:
        return {"ok": False, "latency_ms": None, "backend": backend,
                "error": str(e)[:160]}


# ─────────────────────────────────────────────────────────────────────────
# Machine API keys (Bearer tokens for /predict/quick, /predict/batch, ...)
# ─────────────────────────────────────────────────────────────────────────
# Threat model: a leaked key gives the bearer the same crop-diagnosis
# capability the user has. Keys are scoped to one user account; the user
# can revoke at any time. The raw key is shown to the caller ONCE on
# creation and is never stored in the DB — only a sha256 hash. A leaked
# DB therefore cannot be used to mint working tokens. Token format:
# `apin_` + 32 hex chars (40 chars total) — easy to spot in logs.

import hashlib as _hashlib
import secrets as _secrets

_API_KEY_PREFIX     = "apin_"
# PDA F1: cap active keys per user so a single account cannot exhaust disk
# or balloon the api_keys table. Tunable; 20 is generous (most users mint 1–3).
_MAX_KEYS_PER_USER  = 20


def _hash_api_token(raw_token: str) -> str:
    """sha256 hex of the raw token, deterministic per token."""
    return _hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_api_key(user_id: int, name: str) -> dict:
    """Mint a new API key for `user_id` and return the one-time secret.

    The returned dict is the ONLY time the raw `token` is ever exposed.
    On every subsequent read the caller can see only `token_prefix`.

    Raises ValueError if name is empty or > 64 chars.
    Raises sqlite3.IntegrityError if the (vanishingly rare) sha256 collision
    occurs — call sites should let that bubble up so we never silently
    deduplicate two different keys.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("API key name is required.")
    if len(name) > 64:
        raise ValueError("API key name must be 64 characters or fewer.")

    # 32 hex chars = 128 bits of entropy. Prefixed with `apin_` so a leaked
    # key in a log file is grep-able without ambiguity (industry practice;
    # GitHub uses ghp_, OpenAI uses sk-, Stripe uses sk_live_, ...).
    raw_token    = _API_KEY_PREFIX + _secrets.token_hex(16)
    token_hash   = _hash_api_token(raw_token)
    token_prefix = raw_token[:12]
    created_at   = _now_iso()

    with _write_lock, get_conn() as c:
        # PDA F1: enforce per-user key cap inside the write lock so two
        # concurrent POST /keys cannot both pass the check and exceed the
        # cap by one.
        row = c.execute(
            "SELECT COUNT(*) AS n FROM api_keys "
            "WHERE user_id = ? AND revoked_at IS NULL",
            (int(user_id),)).fetchone()
        active = int(row["n"] if row is not None else 0)
        if active >= _MAX_KEYS_PER_USER:
            raise ValueError(
                f"Limit of {_MAX_KEYS_PER_USER} active API keys per "
                "account reached. Revoke an existing key first "
                "(DELETE /keys/{id}) before minting a new one.")

        cur = c.execute(
            "INSERT INTO api_keys (user_id, name, token_hash, token_prefix, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (int(user_id), name, token_hash, token_prefix, created_at))
        # sqlite3 cursors expose lastrowid; the libsql shim exposes it too.
        key_id = int(getattr(cur, "lastrowid", 0) or 0)

    return {
        "id":            key_id,
        "name":          name,
        "token":         raw_token,    # ONE-TIME — caller MUST store it.
        "token_prefix":  token_prefix,
        "created_at":    created_at,
    }


def list_api_keys(user_id: int, include_revoked: bool = False) -> list[dict]:
    """List the user's keys (newest first). Never returns the raw token.

    Fields: id, name, token_prefix, created_at, last_used_at, revoked_at.
    """
    sql = ("SELECT id, name, token_prefix, created_at, last_used_at, "
           "revoked_at FROM api_keys WHERE user_id = ?")
    params: list = [int(user_id)]
    if not include_revoked:
        sql += " AND revoked_at IS NULL"
    sql += " ORDER BY id DESC"
    try:
        with get_conn() as c:
            rows = c.execute(sql, tuple(params)).fetchall()
    except Exception:
        return []
    return [dict(r) for r in rows]


def revoke_api_key(user_id: int, key_id: int) -> bool:
    """Revoke (soft-delete) a key the caller owns. Returns True if a row was
    updated, False if no matching active key exists (already revoked or
    not theirs)."""
    now = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "UPDATE api_keys SET revoked_at = ? "
            "WHERE id = ? AND user_id = ? AND revoked_at IS NULL",
            (now, int(key_id), int(user_id)))
        # Both backends expose rowcount on the cursor. PDA: report
        # failure on rowcount-read errors rather than fabricating
        # success — the caller (v1_keys_revoke) translates False to a
        # 404 "no active key found", which is the honest answer.
        try:
            return int(getattr(cur, "rowcount", 0) or 0) > 0
        except Exception:
            return False


def find_api_key(raw_token: str) -> Optional[dict]:
    """Look up a raw token. Returns {user_id, key_id, name} on hit, else None.

    Touches last_used_at lazily (best-effort, swallowed on failure). Returns
    None for revoked keys, malformed tokens, or any DB error — callers can
    treat None uniformly as "no valid auth".
    """
    if not isinstance(raw_token, str):
        return None
    raw_token = raw_token.strip()
    if not raw_token.startswith(_API_KEY_PREFIX):
        return None
    token_hash = _hash_api_token(raw_token)
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT id, user_id, name, revoked_at FROM api_keys "
                "WHERE token_hash = ?", (token_hash,)).fetchone()
    except Exception:
        return None
    if row is None or row["revoked_at"] is not None:
        return None
    # Lazy last-used touch (best-effort).
    try:
        with _write_lock, get_conn() as c:
            c.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                      (_now_iso(), int(row["id"])))
    except Exception:
        pass
    return {"key_id":  int(row["id"]),
            "user_id": int(row["user_id"]),
            "name":    row["name"]}


def lookup_api_key_full(raw_token: str) -> Optional[dict]:
    """Stage-7 / API Console key lookup. Returns the FULL row dict needed by
    `@require_scope` (scopes, status, ip_allowlist, etc.), or None.

    Differs from `find_api_key` (the legacy lookup) in three ways:
      1. Returns all Stage-7 columns the decorator inspects (scopes parsed
         from JSON, status string, expires_at, ip_allowlist, origin_allowlist,
         enforce_origin_for_non_browser, environment, public_id).
      2. Does NOT touch last_used_at — that's deferred to the buffered
         batch update from §10 (REV-R2-I07 — Phase-10 work).
      3. Returns None on any DB error so callers treat it as "no auth".

    NOTE: this function intentionally does NOT enforce expiration / status
    rules — that's the decorator's job. Returns the raw row; the decorator
    branches on status='legacy_pending' / 'rotating' / 'disabled' / etc.

    Raises nothing. Returns None on:
      - non-string input
      - prefix doesn't match `apin_`
      - DB error
      - row not found
      - row has deleted_at set (hard-delete)
    """
    if not isinstance(raw_token, str):
        return None
    raw_token = raw_token.strip()
    if not raw_token.startswith(_API_KEY_PREFIX):
        return None
    token_hash = _hash_api_token(raw_token)
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT id, user_id, name, public_id, "
                "       environment, scopes, status, expires_at, "
                "       ip_allowlist, origin_allowlist, "
                "       enforce_origin_for_non_browser, "
                "       rate_limit_per_min, quota_per_day, "
                "       revoked_at, deleted_at "
                "FROM api_keys WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    # Hard-deleted keys never authenticate.
    if row["deleted_at"] is not None:
        return None
    # Parse the JSON-encoded scopes array. Empty list on parse failure
    # (safer than returning None — the decorator's `missing_scope` check
    # will then reject if any scope is required).
    import json as _json
    try:
        scopes = _json.loads(row["scopes"]) if row["scopes"] else []
        if not isinstance(scopes, list):
            scopes = []
    except Exception:
        scopes = []
    # IP / origin allowlists are JSON arrays of strings, or NULL.
    def _parse_json_list(s):
        if s is None or s == "":
            return None
        try:
            v = _json.loads(s)
            return v if isinstance(v, list) else None
        except Exception:
            return None
    return {
        "key_id":           int(row["id"]),
        "user_id":          int(row["user_id"]),
        "public_id":        row["public_id"],
        "name":             row["name"],
        "environment":      row["environment"],
        "scopes":           scopes,
        "status":           row["status"],
        "expires_at":       row["expires_at"],
        "ip_allowlist":     _parse_json_list(row["ip_allowlist"]),
        "origin_allowlist": _parse_json_list(row["origin_allowlist"]),
        "enforce_origin_for_non_browser":
            bool(row["enforce_origin_for_non_browser"]),
        "rate_limit_per_min": row["rate_limit_per_min"],
        "quota_per_day":    row["quota_per_day"],
        # `revoked_at` is preserved for callers that want to know WHEN the
        # key was revoked. Status='disabled' is set by Phase-1 backfill
        # whenever revoked_at is non-NULL.
        "revoked_at":       row["revoked_at"],
    }


# ═════════════════════════════════════════════════════════════════════════
# STAGE 7 — API CONSOLE — Phase 2.4 CRUD helpers (spec §7.1)
#
# Each helper here is scoped to a single (user_id, public_id) operation. The
# session-cookie auth (slot 6 middleware) populates request.state.session
# with the user_id; routes pass that user_id in so a user can never act on
# someone else's keys even with a valid session.
#
# All helpers return either a dict (success), None (not-found / wrong user),
# or raise a typed exception for the conflict cases (duplicate name, rotating
# already-in-progress, etc.). Routes map these to the §26 error codes.
# ═════════════════════════════════════════════════════════════════════════

class DuplicateKeyNameError(ValueError):
    """409 duplicate_name — UNIQUE INDEX collision on (user_id, name) for
    active/rotating keys."""


class KeyAlreadyRotatingError(ValueError):
    """409 already_rotating — rotate called on a key already in rotation."""


class InvalidKeyStateError(ValueError):
    """400-class — wrong status for the requested operation (e.g. trying
    to hard-delete an active key)."""


def _new_public_id() -> str:
    """16 lowercase hex chars prefixed with `k_`. Caller retries on UNIQUE
    collision (~1 in 2^64 chance, but we still defend against it)."""
    return "k_" + secrets.token_hex(8)


def _b62_43() -> str:
    """Generate 43 base62 chars via the same path tokens.py uses.

    Importing tokens here avoids re-duplicating the entropy logic. The
    function is called at api-key creation time only — not in any hot path.
    """
    from scripts.apin_v2.account.tokens import _b62_fixed_43
    return _b62_fixed_43(secrets.token_bytes(32))


def create_console_api_key(
    *, user_id: int, name: str, environment: str, scopes: list,
    ip_allowlist: Optional[list] = None,
    origin_allowlist: Optional[list] = None,
    rate_limit_per_min: Optional[int] = None,
    quota_per_day: Optional[int] = None,
    expires_at: Optional[str] = None,
    note: Optional[str] = None,
    created_ip: Optional[str] = None,
    created_ua: Optional[str] = None,
) -> dict:
    """Create an API key via the Console.

    Returns the full dict (same shape as lookup_api_key_full) PLUS the
    plaintext token under key `plaintext_token` — that's the one-time-view
    payload (§4.4). Caller MUST strip it from any cached/log copy.

    Raises:
        DuplicateKeyNameError — (user_id, name) collides with an existing
            active/rotating key (UNIQUE INDEX idx_api_keys_user_name_active).
    """
    if environment not in ("live", "test"):
        raise ValueError(f"environment must be 'live' or 'test', got {environment!r}")

    plaintext = "apin_" + environment + "_" + _b62_43()
    tok_hash = _hash_api_token(plaintext)
    last_four = plaintext[-4:]
    scopes_json = json.dumps(scopes)
    ipal_json = json.dumps(ip_allowlist) if ip_allowlist is not None else None
    oral_json = json.dumps(origin_allowlist) if origin_allowlist is not None else None
    now = _now_iso()

    # Retry on public_id collision (effectively never happens, but defensive).
    for _attempt in range(8):
        pid = _new_public_id()
        try:
            with _write_lock, get_conn() as c:
                cur = c.execute(
                    "INSERT INTO api_keys ("
                    "  user_id, name, token_hash, token_prefix, created_at, "
                    "  environment, scopes, last_four, "
                    "  ip_allowlist, origin_allowlist, "
                    "  rate_limit_per_min, quota_per_day, expires_at, "
                    "  created_ip, created_ua, "
                    "  status, public_id, "
                    "  enforce_origin_for_non_browser, note, legacy_alert_emitted"
                    ") VALUES ("
                    "  ?, ?, ?, ?, ?, "
                    "  ?, ?, ?, "
                    "  ?, ?, "
                    "  ?, ?, ?, "
                    "  ?, ?, "
                    "  'active', ?, "
                    "  1, ?, 0"
                    ")",
                    (
                        user_id, name, tok_hash, plaintext[:10], now,
                        environment, scopes_json, last_four,
                        ipal_json, oral_json,
                        rate_limit_per_min, quota_per_day, expires_at,
                        created_ip, created_ua,
                        pid, note,
                    ),
                )
            key_id = int(cur.lastrowid) if hasattr(cur, "lastrowid") else None
            return {
                "key_id": key_id,
                "user_id": user_id,
                "public_id": pid,
                "name": name,
                "environment": environment,
                "scopes": scopes,
                "status": "active",
                "expires_at": expires_at,
                "ip_allowlist": ip_allowlist,
                "origin_allowlist": origin_allowlist,
                "enforce_origin_for_non_browser": True,
                "rate_limit_per_min": rate_limit_per_min,
                "quota_per_day": quota_per_day,
                "last_four": last_four,
                "created_at": now,
                "note": note,
                "plaintext_token": plaintext,
            }
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            # `idx_api_keys_user_name_active` UNIQUE collision → duplicate name
            if "user_name_active" in msg or "name" in msg:
                raise DuplicateKeyNameError(
                    f"a key named {name!r} already exists in {environment!r}"
                ) from e
            # `idx_api_keys_public_id_unique` collision → retry with new id
            if "public_id" in msg:
                continue
            # `idx_api_keys_hash` (token_hash UNIQUE) collision → retry, regen
            if "token_hash" in msg:
                plaintext = "apin_" + environment + "_" + _b62_43()
                tok_hash = _hash_api_token(plaintext)
                last_four = plaintext[-4:]
                continue
            raise
    raise RuntimeError(
        "create_console_api_key: 8 public_id retries exhausted — improbable"
    )


def list_console_api_keys(
    *, user_id: int,
    env: str = "all",          # 'live' | 'test' | 'all'
    status: str = "all",       # 'active' | 'rotating' | 'disabled' | 'all'
    search: Optional[str] = None,
    cursor: Optional[int] = None,
    limit: int = 20,
) -> dict:
    """Paginated list of a user's API keys for the Console.

    Returns {items: [...], next_cursor: int | None, total_filter: int}.
    `items` are the same shape as lookup_api_key_full but without the
    plaintext_token (never persisted). Excludes hard-deleted rows.
    """
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    where = ["user_id = ?", "deleted_at IS NULL"]
    args: list = [user_id]
    if env in ("live", "test"):
        where.append("environment = ?")
        args.append(env)
    if status != "all":
        where.append("status = ?")
        args.append(status)
    if search:
        # Substring on name. SQLite LIKE is case-insensitive on ASCII by
        # default; OK for our use case (names are user-typed strings).
        where.append("name LIKE ?")
        args.append(f"%{search}%")
    if cursor is not None:
        where.append("id < ?")   # descending id pagination
        args.append(int(cursor))

    where_sql = " AND ".join(where)

    with get_conn() as c:
        rows = c.execute(
            f"SELECT id, public_id, name, environment, scopes, status, "
            f"       last_four, expires_at, created_at, last_used_at, "
            f"       ip_allowlist, origin_allowlist, rate_limit_per_min, "
            f"       quota_per_day, note, enforce_origin_for_non_browser "
            f"FROM api_keys WHERE {where_sql} "
            f"ORDER BY id DESC LIMIT ?",
            args + [limit + 1]
        ).fetchall()

    items: list = []
    has_more = len(rows) > limit
    for row in rows[:limit]:
        try:
            scopes = json.loads(row["scopes"]) if row["scopes"] else []
            if not isinstance(scopes, list):
                scopes = []
        except Exception:
            scopes = []
        ipal = None
        if row["ip_allowlist"]:
            try:
                ipal = json.loads(row["ip_allowlist"])
                if not isinstance(ipal, list):
                    ipal = None
            except Exception:
                ipal = None
        oral = None
        if row["origin_allowlist"]:
            try:
                oral = json.loads(row["origin_allowlist"])
                if not isinstance(oral, list):
                    oral = None
            except Exception:
                oral = None
        items.append({
            "public_id": row["public_id"],
            "name": row["name"],
            "environment": row["environment"],
            "scopes": scopes,
            "status": row["status"],
            "last_four": row["last_four"],
            "expires_at": row["expires_at"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "ip_allowlist": ipal,
            "origin_allowlist": oral,
            "rate_limit_per_min": row["rate_limit_per_min"],
            "quota_per_day": row["quota_per_day"],
            "note": row["note"],
            "enforce_origin_for_non_browser": bool(
                row["enforce_origin_for_non_browser"]),
        })
    next_cursor = int(rows[limit - 1]["id"]) if has_more else None
    return {"items": items, "next_cursor": next_cursor}


def get_console_api_key(*, user_id: int, public_id: str) -> Optional[dict]:
    """Fetch one key by (user_id, public_id). None if not found or
    belongs to a different user (the user_id filter is part of the WHERE
    clause for cross-user safety)."""
    with get_conn() as c:
        row = c.execute(
            "SELECT id, public_id, name, environment, scopes, status, "
            "       last_four, expires_at, created_at, last_used_at, "
            "       ip_allowlist, origin_allowlist, rate_limit_per_min, "
            "       quota_per_day, note, enforce_origin_for_non_browser, "
            "       revoked_at, deleted_at, predecessor_id, successor_id, "
            "       rotation_grace_until "
            "FROM api_keys WHERE user_id = ? AND public_id = ? AND deleted_at IS NULL",
            (user_id, public_id)
        ).fetchone()
    if row is None:
        return None
    try:
        scopes = json.loads(row["scopes"]) if row["scopes"] else []
        if not isinstance(scopes, list):
            scopes = []
    except Exception:
        scopes = []

    def _parse(s):
        if not s:
            return None
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else None
        except Exception:
            return None

    return {
        "key_id": int(row["id"]),
        "public_id": row["public_id"],
        "name": row["name"],
        "environment": row["environment"],
        "scopes": scopes,
        "status": row["status"],
        "last_four": row["last_four"],
        "expires_at": row["expires_at"],
        "created_at": row["created_at"],
        "last_used_at": row["last_used_at"],
        "ip_allowlist": _parse(row["ip_allowlist"]),
        "origin_allowlist": _parse(row["origin_allowlist"]),
        "rate_limit_per_min": row["rate_limit_per_min"],
        "quota_per_day": row["quota_per_day"],
        "note": row["note"],
        "enforce_origin_for_non_browser":
            bool(row["enforce_origin_for_non_browser"]),
        "revoked_at": row["revoked_at"],
        "predecessor_id": row["predecessor_id"],
        "successor_id": row["successor_id"],
        "rotation_grace_until": row["rotation_grace_until"],
    }


def patch_console_api_key(
    *, user_id: int, public_id: str, **fields
) -> Optional[dict]:
    """Update editable fields on a key. Returns the updated dict, or None
    if the key doesn't belong to user_id. Raises DuplicateKeyNameError
    on name-collision with another active key.

    Accepted fields:
        name, scopes (list -> JSON), ip_allowlist (list -> JSON),
        origin_allowlist (list -> JSON), rate_limit_per_min, quota_per_day,
        expires_at, note.

    Unknown fields are silently dropped (defensive — handler should have
    validated before calling, but we don't trust the kwargs blindly).
    """
    ALLOWED = {
        "name", "scopes", "ip_allowlist", "origin_allowlist",
        "rate_limit_per_min", "quota_per_day", "expires_at", "note",
    }
    JSON_FIELDS = {"scopes", "ip_allowlist", "origin_allowlist"}

    sets: list = []
    args: list = []
    for k, v in fields.items():
        if k not in ALLOWED:
            continue
        if v is None and k not in ("note", "expires_at", "ip_allowlist",
                                    "origin_allowlist", "rate_limit_per_min",
                                    "quota_per_day"):
            # name, scopes can't be set to None
            continue
        if k in JSON_FIELDS:
            sets.append(f"{k} = ?")
            args.append(json.dumps(v) if v is not None else None)
        else:
            sets.append(f"{k} = ?")
            args.append(v)
    if not sets:
        # Nothing to update — return current state.
        return get_console_api_key(user_id=user_id, public_id=public_id)

    args.extend([user_id, public_id])
    try:
        with _write_lock, get_conn() as c:
            cur = c.execute(
                f"UPDATE api_keys SET {', '.join(sets)} "
                f"WHERE user_id = ? AND public_id = ? "
                f"AND deleted_at IS NULL AND status IN ('active','rotating')",
                args
            )
            rowcount = cur.rowcount if hasattr(cur, "rowcount") else None
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "user_name_active" in msg or "name" in msg:
            raise DuplicateKeyNameError(
                f"a key with the new name already exists"
            ) from e
        raise
    if rowcount == 0:
        # Either not found, wrong user, deleted, or non-editable status
        return None
    return get_console_api_key(user_id=user_id, public_id=public_id)


def rotate_console_api_key(
    *, user_id: int, public_id: str, grace_seconds: int = 172_800
) -> Optional[dict]:
    """Rotate a key: mint a new one with identical metadata, mark the old
    as 'rotating' until grace expiry. Returns the NEW key dict with
    plaintext_token (one-time-view). None if old key not found.

    Spec §7.1 line 2711-2722:
      - status must currently be 'active' (cannot rotate a rotating key)
      - new key inherits scopes/limits/expiration
      - old key.successor_id = new.public_id, status='rotating',
        rotation_grace_until = now + grace_seconds
      - both rows get a 'rotated' audit row (audit emission deferred)

    Raises KeyAlreadyRotatingError if status='rotating' already.
    """
    old = get_console_api_key(user_id=user_id, public_id=public_id)
    if old is None:
        return None
    if old["status"] == "rotating":
        raise KeyAlreadyRotatingError(
            f"key {public_id} is already rotating; wait for grace period"
        )
    if old["status"] != "active":
        raise InvalidKeyStateError(
            f"cannot rotate key in status {old['status']!r}"
        )

    # Mint the new key inheriting metadata
    new = create_console_api_key(
        user_id=user_id,
        name=old["name"] + " (rotated)",   # avoid UNIQUE collision
        environment=old["environment"],
        scopes=old["scopes"],
        ip_allowlist=old["ip_allowlist"],
        origin_allowlist=old["origin_allowlist"],
        rate_limit_per_min=old["rate_limit_per_min"],
        quota_per_day=old["quota_per_day"],
        expires_at=old["expires_at"],
        note=(old.get("note") or ""),
    )

    # Mark old as rotating with grace
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    grace_until = (_dt.now(_tz.utc) + _td(seconds=int(grace_seconds))) \
        .isoformat(timespec="milliseconds").replace("+00:00", "Z")
    with _write_lock, get_conn() as c:
        c.execute(
            "UPDATE api_keys SET status = 'rotating', successor_id = ?, "
            "rotation_grace_until = ? "
            "WHERE user_id = ? AND public_id = ? AND status = 'active'",
            (new["public_id"], grace_until, user_id, public_id)
        )
        # Also stamp predecessor on the new row
        c.execute(
            "UPDATE api_keys SET predecessor_id = ? "
            "WHERE user_id = ? AND public_id = ?",
            (public_id, user_id, new["public_id"])
        )
    new["predecessor_id"] = public_id
    return new


def disable_console_api_key(*, user_id: int, public_id: str) -> Optional[dict]:
    """Phase 8 WI-P8-KEY-DISABLE-VERB: move a key from 'active' to
    'disabled'. Disabled keys cannot authenticate; can be re-enabled or
    hard-deleted. Idempotent: re-disabling an already-disabled key is a
    no-op and returns the current row.

    Returns the updated row dict, or None if not found / not owned.
    Raises ValueError if the key is in a status that cannot be disabled
    (currently: 'rotating', 'deleted', 'expired' — those need their own
    transition paths).
    """
    with _write_lock, get_conn() as c:
        row = c.execute(
            "SELECT id, status FROM api_keys "
            "WHERE user_id = ? AND public_id = ? AND deleted_at IS NULL",
            (int(user_id), public_id)
        ).fetchone()
        if row is None:
            return None
        if row["status"] == "disabled":
            return get_console_api_key(user_id=int(user_id), public_id=public_id)
        if row["status"] not in ("active", "legacy_pending"):
            raise ValueError(
                f"cannot disable a key in status {row['status']!r}; only "
                f"'active' keys can be disabled directly"
            )
        c.execute(
            "UPDATE api_keys SET status = 'disabled', revoked_at = ?, "
            "  updated_at = ? "
            "WHERE id = ?",
            (_now_iso(), _now_iso(), row["id"])
        )
    return get_console_api_key(user_id=int(user_id), public_id=public_id)


def enable_console_api_key(*, user_id: int, public_id: str) -> Optional[dict]:
    """Phase 8 WI-P8-KEY-DISABLE-VERB: reverse of disable. Moves a
    'disabled' key back to 'active' and clears revoked_at.

    Returns updated row dict, or None if not found. Raises ValueError if
    the key is in a non-re-enableable status (rotating / deleted / expired).
    """
    with _write_lock, get_conn() as c:
        row = c.execute(
            "SELECT id, status FROM api_keys "
            "WHERE user_id = ? AND public_id = ? AND deleted_at IS NULL",
            (int(user_id), public_id)
        ).fetchone()
        if row is None:
            return None
        if row["status"] == "active":
            return get_console_api_key(user_id=int(user_id), public_id=public_id)
        if row["status"] != "disabled":
            raise ValueError(
                f"cannot enable a key in status {row['status']!r}; only "
                f"'disabled' keys can be re-activated"
            )
        c.execute(
            "UPDATE api_keys SET status = 'active', revoked_at = NULL, "
            "  updated_at = ? "
            "WHERE id = ?",
            (_now_iso(), row["id"])
        )
    return get_console_api_key(user_id=int(user_id), public_id=public_id)


def append_audit_log(
    *, user_id: int, action: str, key_id: Optional[str] = None,
    before: Optional[dict] = None, after: Optional[dict] = None,
    actor_ip: Optional[str] = None, actor_ua: Optional[str] = None,
    actor_session_id: Optional[str] = None,
    sudo_token_id: Optional[str] = None,
    details: Optional[dict] = None,
    key_name_at_time: str = "",
) -> int:
    """Phase 8 Wave C (WI-P8-AUDITREC-SLOT5): append a row to api_key_audit
    with the hash-chained prev_hash → row_hash linkage so the operator can
    later prove a row wasn't deleted/edited.

    The hash chain is per-user: `prev_hash` is the previous row's `row_hash`
    for the same user_id (or 64 zeros for the first row). Spec §12 contract.

    Returns the new audit row id.
    """
    import json as _json
    if not action:
        raise ValueError("action is required")
    if len(action) > 80:
        raise ValueError("action too long (max 80 chars)")

    before_json = _json.dumps(before, separators=(",", ":")) if before else None
    after_json  = _json.dumps(after,  separators=(",", ":")) if after  else None
    details_str = _json.dumps(details, separators=(",", ":")) if details else None
    if actor_ua and len(actor_ua) > 1024:
        actor_ua = actor_ua[:1024]

    with _write_lock, get_conn() as c:
        # Look up prev_hash: the latest row's row_hash for this user, or
        # the all-zeros initial value.
        prev_row = c.execute(
            "SELECT row_hash FROM api_key_audit "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (int(user_id),)
        ).fetchone()
        prev_hash = (prev_row["row_hash"] if prev_row
                     else "0" * 64)

        now_iso = _now_iso()

        # Compute the row hash: SHA-256 over the canonical row tuple.
        # Field order is fixed forever — never reorder without bumping
        # a hash version (spec §12.4 PDA-R3-F02).
        digest_input = "|".join([
            str(int(user_id)),
            action,
            now_iso,
            str(key_id or ""),
            actor_ip or "",
            actor_ua or "",
            actor_session_id or "",
            sudo_token_id or "",
            before_json or "",
            after_json or "",
            details_str or "",
            key_name_at_time,
            prev_hash,
        ])
        row_hash = _hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

        cur = c.execute(
            "INSERT INTO api_key_audit "
            "(key_id, user_id, action, timestamp, actor_ip, actor_ua, "
            " actor_session_id, sudo_token_id, before_json, after_json, "
            " details, key_name_at_time, prev_hash, row_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key_id, int(user_id), action, now_iso,
             actor_ip, actor_ua, actor_session_id, sudo_token_id,
             before_json, after_json, details_str, key_name_at_time,
             prev_hash, row_hash)
        )
        return int(cur.lastrowid)


def list_audit_log(
    *, user_id: int, key_id: Optional[str] = None,
    limit: int = 100, cursor: Optional[int] = None,
) -> list[dict]:
    """List audit rows, newest first. Filter by key_id when given."""
    limit = max(1, min(int(limit), 500))
    wheres = ["user_id = ?"]
    args: list = [int(user_id)]
    if key_id:
        wheres.append("key_id = ?")
        args.append(key_id)
    if cursor:
        wheres.append("id < ?")
        args.append(int(cursor))
    sql = (
        "SELECT id, key_id, action, timestamp, actor_ip, actor_session_id, "
        "       sudo_token_id, before_json, after_json, details, "
        "       key_name_at_time, prev_hash, row_hash "
        "FROM api_key_audit "
        f"WHERE {' AND '.join(wheres)} "
        "ORDER BY id DESC LIMIT ?"
    )
    args.append(limit)
    import json as _json
    with get_conn() as c:
        rows = c.execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("before_json", "after_json", "details"):
            v = d.get(k)
            if v:
                try: d[k] = _json.loads(v)
                except Exception: pass
        out.append(d)
    return out


def list_key_usage_minute(*, user_id: int, public_id: str,
                            minutes: int = 60) -> list[dict]:
    """Phase 8 Wave D: per-minute usage rollup for the detail page sparkline.
    Returns rows for the LAST `minutes` minutes (default 60). Ordered oldest
    → newest so the client can plot left-to-right without re-sorting.

    Verifies key ownership before reading (don't leak another user's data).

    Phase 9.B: also returns bytes_in/out + latency_p99 + latency_avg so
    the Usage tab and Overview tab can render full KPI tiles from this
    single endpoint when granularity_seconds == 60.
    """
    if get_console_api_key(user_id=int(user_id), public_id=public_id) is None:
        return []
    minutes = max(1, min(int(minutes), 10080))   # cap at 7d
    with get_conn() as c:
        rows = c.execute(
            "SELECT minute_ts, requests, errors, rate_limited, "
            "       quota_blocked, bytes_in, bytes_out, latency_sum_ms, "
            "       latency_count, latency_p50_ms, latency_p95_ms, "
            "       latency_p99_ms "
            "FROM api_key_usage_minute "
            "WHERE key_id = ? "
            "  AND minute_ts >= datetime('now', ?) "
            "ORDER BY minute_ts ASC",
            (public_id, f"-{minutes} minutes")
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        lc = int(d.get("latency_count") or 0)
        ls = int(d.get("latency_sum_ms") or 0)
        d["latency_avg_ms"] = round(ls / lc, 1) if lc > 0 else None
        out.append(d)
    return out


def list_key_requests(*, user_id: int, public_id: str,
                       limit: int = 50, cursor: Optional[int] = None,
                       method: Optional[str] = None,
                       status_min: Optional[int] = None,
                       status_max: Optional[int] = None,
                       path_contains: Optional[str] = None,
                       since_iso: Optional[str] = None,
                       until_iso: Optional[str] = None,
                       local_hour: Optional[int] = None,
                       local_weekday: Optional[int] = None,
                       tz_off: int = 0,
                       ) -> list[dict]:
    """Phase 9.A — recent request log for the detail page. Newest first.

    Schema is §6.4 api_key_request_log: id, key_id, timestamp, method, path,
    status_code, latency_ms, ip, ua, bytes_in, bytes_out, error_code, via.

    Optional filter args support Phase 9.B endpoint filters (method,
    status range, path substring, since/until). The `cursor` is the
    `id` of the last row from the previous page (newer ID == newer row).
    """
    if get_console_api_key(user_id=int(user_id), public_id=public_id) is None:
        return []
    limit = max(1, min(int(limit), 200))
    wheres = ["key_id = ?"]
    args: list = [public_id]
    if cursor:
        wheres.append("id < ?")
        args.append(int(cursor))
    if method:
        wheres.append("method = ?")
        args.append(method.upper())
    if status_min is not None:
        wheres.append("status_code >= ?")
        args.append(int(status_min))
    if status_max is not None:
        wheres.append("status_code < ?")
        args.append(int(status_max))
    if path_contains:
        wheres.append("path LIKE ?")
        # Limit substring length to avoid catastrophic patterns.
        sub = str(path_contains)[:128].replace("%", r"\%").replace("_", r"\_")
        args.append(f"%{sub}%")
    if since_iso:
        wheres.append("timestamp >= ?")
        args.append(str(since_iso))
    if until_iso:
        wheres.append("timestamp < ?")
        args.append(str(until_iso))
    # local hour-of-day / weekday filters (viewer tz). Shift the UTC timestamp
    # by tz_off minutes, then match the hour / weekday in that local frame.
    if local_hour is not None or local_weekday is not None:
        try:
            tzo = max(-840, min(840, int(tz_off)))
        except Exception:
            tzo = 0
        modifier = f"{tzo:+d} minutes"
        if local_hour is not None:
            wheres.append(
                "CAST(substr(datetime(substr(timestamp,1,19), ?), 12, 2) AS INTEGER) = ?")
            args.append(modifier)
            args.append(int(local_hour))
        if local_weekday is not None:
            # incoming local_weekday is Mon=0..Sun=6; SQLite %w is Sun=0..Sat=6
            sqlite_wd = (int(local_weekday) + 1) % 7
            wheres.append(
                "CAST(strftime('%w', datetime(substr(timestamp,1,19), ?)) AS INTEGER) = ?")
            args.append(modifier)
            args.append(sqlite_wd)
    sql = (
        "SELECT id, timestamp, method, path, status_code, latency_ms, "
        "       bytes_in, bytes_out, ip, ua, error_code, via "
        "FROM api_key_request_log "
        f"WHERE {' AND '.join(wheres)} "
        "ORDER BY id DESC LIMIT ?"
    )
    args.append(limit)
    with get_conn() as c:
        try:
            rows = c.execute(sql, args).fetchall()
            return [dict(r) for r in rows]
        except Exception as _e:
            # Defensive: if a future migration adds columns the SELECT
            # references but they're missing, return [] rather than crash.
            return []


# ─────────────────────────────────────────────────────────────────────────
# Phase 9.B — Account-wide usage observability helpers.
#
# These all read `api_key_request_log` + `api_key_usage_minute` filtered by
# the user's owned set of `api_keys.public_id`. Ownership is enforced once
# via the `_user_key_ids()` subquery — the request log has no user_id of
# its own, so we always join through api_keys.public_id.
#
# Filter semantics (used consistently across summary / timeseries / top /
# requests / minute-detail):
#   - key_id        : single public_id (must belong to user; otherwise 0)
#   - endpoint      : path substring match  (case-sensitive LIKE %x%)
#   - status        : "2xx" | "3xx" | "4xx" | "5xx" | "429" | "200" | etc.
#   - env           : "live" | "test"  (joins api_keys.environment)
#   - method        : "GET" | "POST" | ...
# ─────────────────────────────────────────────────────────────────────────

def _status_clause(status: Optional[str], alias: str = '') -> tuple[str, list]:
    """Translate a status shorthand or literal to a SQL fragment + params.
    Returns ('', []) when status is None/empty (no filter).

    9.N.6.f · `alias` lets callers prefix the column (e.g. 'r' → 'r.status_code')
    so this helper works with JOINed queries that need a qualified column ref.
    """
    if not status:
        return "", []
    s = str(status).strip().lower()
    col = (alias + '.' if alias else '') + 'status_code'
    if s == "2xx":
        return f"({col} >= 200 AND {col} < 300)", []
    if s == "3xx":
        return f"({col} >= 300 AND {col} < 400)", []
    if s == "4xx":
        return f"({col} >= 400 AND {col} < 500)", []
    if s == "5xx":
        return f"({col} >= 500 AND {col} < 600)", []
    if s == "429":
        return f"({col} = 429)", []
    if s.isdigit():
        return f"({col} = ?)", [int(s)]
    return "(1=0)", []


def _build_user_keys_where(user_id: int,
                             key_id: Optional[str],
                             env: Optional[str]) -> tuple[str, list]:
    """Return (sub-SELECT for `key_id IN (...)`, params)."""
    wheres = ["user_id = ?", "deleted_at IS NULL"]
    args: list = [int(user_id)]
    if key_id:
        wheres.append("public_id = ?")
        args.append(str(key_id))
    if env:
        wheres.append("environment = ?")
        args.append(str(env))
    sub = "(SELECT public_id FROM api_keys WHERE " + " AND ".join(wheres) + ")"
    return sub, args


def compute_usage_lifetime(*, user_id: int, key_id: Optional[str] = None,
                            env: Optional[str] = None) -> dict:
    """'Has this scope ever seen a request?' probe — powers the Usage empty
    states (new-key vs dormant-window). Scoped to the user's keys (+ optional
    key_id/env), deliberately NOT to endpoint/status (those are content
    filters, a separate 'no matches' case). One indexed lookup + a count."""
    sub, args = _build_user_keys_where(int(user_id), key_id, env)
    out = {"ever": False, "total": 0, "last_ts": None, "last_id": None,
           "last_method": None, "last_path": None, "last_status": None}
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT id, timestamp, method, path, status_code "
                "FROM api_key_request_log WHERE key_id IN " + sub +
                " ORDER BY id DESC LIMIT 1", args).fetchone()
            if not row:
                return out
            d = dict(row)
            cnt = c.execute(
                "SELECT COUNT(*) AS n FROM api_key_request_log WHERE key_id IN " + sub,
                args).fetchone()
            out.update({
                "ever": True,
                "total": int(dict(cnt).get("n", 0) or 0),
                "last_ts": d.get("timestamp"),
                "last_id": d.get("id"),
                "last_method": d.get("method"),
                "last_path": d.get("path"),
                "last_status": d.get("status_code"),
            })
    except Exception as e:
        _logging.getLogger("apin_v2.auth_db").warning(
            "compute_usage_lifetime failed: %s", e)
    return out


def _percentile(sorted_lats: list[int], p: float) -> Optional[float]:
    if not sorted_lats:
        return None
    import math
    n = len(sorted_lats)
    idx = max(0, min(n - 1, math.ceil(p * n / 100.0) - 1))
    return float(sorted_lats[idx])


def compute_usage_summary(*, user_id: int, range_seconds: int,
                           key_id: Optional[str] = None,
                           env: Optional[str] = None,
                           endpoint: Optional[str] = None,
                           status: Optional[str] = None) -> dict:
    """Compute the KPI block for the Usage summary endpoint.

    9.H · D1 fix: ALWAYS scan the raw log (`api_key_request_log`) — single
    source of truth shared with `compute_usage_timeseries`. Previously the
    no-filter path summed from `api_key_usage_minute` which could drift
    from the raw log (different write paths, different bucket alignment).
    On a 10k-req/day DB the scan is still <100 ms; consistency wins.

    Returns a 4xx + 5xx breakdown so the frontend can show "errors total"
    plus a sub-line "5xx: N · 4xx: M".
    """
    user_id = int(user_id)
    range_seconds = max(60, int(range_seconds))
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)

    def _empty_block() -> dict:
        return {"current": 0, "previous": 0, "delta_pct": None}

    kpis = {
        "requests":      _empty_block(),
        "errors":        _empty_block(),   # 4xx + 5xx combined (D2)
        "errors_4xx":    _empty_block(),
        "errors_5xx":    _empty_block(),
        "rate_limited":  _empty_block(),
        "quota_blocked": _empty_block(),
        "bytes_in":      _empty_block(),
        "bytes_out":     _empty_block(),
        "latency_p50_ms": {"current": None, "previous": None, "delta_pct": None},
        "latency_p95_ms": {"current": None, "previous": None, "delta_pct": None},
        "latency_p99_ms": {"current": None, "previous": None, "delta_pct": None},
        "latency_avg_ms": {"current": None, "previous": None, "delta_pct": None},
        "active_keys":   _empty_block(),  # keys with traffic in window
        "total_keys":    _empty_block(),  # total non-deleted keys (context)
        "error_rate":    {"current": 0.0, "previous": 0.0, "delta_pct": None},
    }

    s_clause, s_args = _status_clause(status)

    def _delta(curr, prev):
        if prev in (0, 0.0, None):
            return None
        try:
            return round(((curr - prev) / prev) * 100.0, 1)
        except Exception:
            return None

    with get_conn() as c:
        # Always raw-scan over current + previous windows. Two queries.
        for label, offset in (("current", 0), ("previous", range_seconds)):
            low_offset = offset + range_seconds
            wheres = [
                f"key_id IN {sub_keys}",
                "timestamp >= datetime('now', '-' || ? || ' seconds')",
                "timestamp <  datetime('now', '-' || ? || ' seconds')",
            ]
            args = list(sub_args) + [low_offset, offset]
            if endpoint:
                wheres.append("path LIKE ?")
                args.append(f"%{endpoint}%")
            if s_clause:
                wheres.append(s_clause)
                args.extend(s_args)
            sql = (
                "SELECT COUNT(*) AS requests, "
                "       SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS errors_5xx, "
                "       SUM(CASE WHEN status_code >= 400 AND status_code < 500 THEN 1 ELSE 0 END) AS errors_4xx, "
                "       SUM(CASE WHEN status_code = 429 THEN 1 ELSE 0 END) AS rate_limited, "
                "       SUM(CASE WHEN error_code = 'rate_limited' OR status_code = 429 THEN 1 ELSE 0 END) AS rate_limited_e, "
                "       SUM(CASE WHEN error_code = 'quota_exceeded' THEN 1 ELSE 0 END) AS quota_blocked, "
                "       COALESCE(SUM(bytes_in), 0)  AS bytes_in, "
                "       COALESCE(SUM(bytes_out), 0) AS bytes_out, "
                "       COALESCE(AVG(latency_ms), 0) AS latency_avg_ms "
                "FROM api_key_request_log "
                f"WHERE {' AND '.join(wheres)}"
            )
            row = dict(c.execute(sql, args).fetchone() or {})
            req_n = int(row.get("requests") or 0)
            err_5xx = int(row.get("errors_5xx") or 0)
            err_4xx = int(row.get("errors_4xx") or 0)
            kpis["requests"][label]      = req_n
            kpis["errors"][label]        = err_4xx + err_5xx
            kpis["errors_4xx"][label]    = err_4xx
            kpis["errors_5xx"][label]    = err_5xx
            kpis["rate_limited"][label]  = int(row.get("rate_limited") or 0)
            kpis["quota_blocked"][label] = int(row.get("quota_blocked") or 0)
            kpis["bytes_in"][label]      = int(row.get("bytes_in") or 0)
            kpis["bytes_out"][label]     = int(row.get("bytes_out") or 0)
            kpis["latency_avg_ms"][label] = (
                round(row.get("latency_avg_ms") or 0, 1) or None
            )
            # Active keys (distinct keys with ≥1 request in window).
            row2 = c.execute(
                "SELECT COUNT(DISTINCT key_id) AS n FROM api_key_request_log "
                f"WHERE key_id IN {sub_keys} "
                "  AND timestamp >= datetime('now', '-' || ? || ' seconds') "
                "  AND timestamp <  datetime('now', '-' || ? || ' seconds')",
                list(sub_args) + [low_offset, offset]
            ).fetchone()
            kpis["active_keys"][label] = int(dict(row2 or {}).get("n") or 0)
            # Percentiles for this window
            if req_n > 0:
                lat_args = list(args)
                lat_sql = (
                    "SELECT latency_ms FROM api_key_request_log "
                    f"WHERE {' AND '.join(wheres)} AND latency_ms IS NOT NULL "
                    "ORDER BY latency_ms ASC"
                )
                lats = [int(r["latency_ms"])
                         for r in c.execute(lat_sql, lat_args).fetchall()]
                kpis["latency_p50_ms"][label] = _percentile(lats, 50)
                kpis["latency_p95_ms"][label] = _percentile(lats, 95)
                kpis["latency_p99_ms"][label] = _percentile(lats, 99)

        # ── Total non-deleted key count (context for "keys with traffic" tile) ──
        try:
            row_tk = c.execute(
                "SELECT COUNT(*) AS n FROM api_keys "
                "WHERE user_id = ? AND deleted_at IS NULL "
                "  AND status != 'deleted'",
                [user_id]
            ).fetchone()
            kpis["total_keys"]["current"] = int(dict(row_tk or {}).get("n") or 0)
            kpis["total_keys"]["previous"] = kpis["total_keys"]["current"]
        except Exception:
            pass

    # ── Derived fields: error_rate + delta_pct ─────────────────────────
    for label in ("current", "previous"):
        req = kpis["requests"][label] or 0
        err = kpis["errors"][label] or 0
        kpis["error_rate"][label] = (
            round((err / req) * 100.0, 2) if req > 0 else 0.0
        )

    for k, block in kpis.items():
        block["delta_pct"] = _delta(block["current"], block["previous"])

    from datetime import datetime as _dt, timezone as _tz
    return {
        "range_seconds": range_seconds,
        "key_id": key_id, "env": env,
        "endpoint": endpoint, "status": status,
        "kpis": kpis,
        "computed_at": _dt.now(_tz.utc).isoformat(),
    }


def compute_usage_timeseries(*, user_id: int, range_seconds: int,
                               granularity_seconds: int, mode: str,
                               key_id: Optional[str] = None,
                               env: Optional[str] = None,
                               endpoint: Optional[str] = None,
                               status: Optional[str] = None,
                               offset_seconds: int = 0) -> dict:
    """Return bucketed series. See routes_usage docstring for shape.

    SQL strategy: GROUP BY a bucket expression. For minute-aligned
    granularity (60/300/900/3600/21600/86400 seconds) we floor the
    `timestamp` by integer-division on epoch seconds — cross-backend.

    9.N.10 · `offset_seconds` shifts the window back in time. Used by
    compare mode: pass offset=range_seconds to fetch the previous-period
    window with the same length as the current view.
    """
    user_id = int(user_id)
    range_seconds = max(60, int(range_seconds))
    offset_seconds = max(0, int(offset_seconds))
    g = max(60, int(granularity_seconds))
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)

    # Choose source: if endpoint/status/by_status/by_endpoint filters
    # or mode requires per-status break-out, use raw request log.
    # 9.N.10 · Also force-raw when offset_seconds > 0 (compare mode) —
    # the aggregate path doesn't support arbitrary time-window offsets.
    use_raw = bool(endpoint or status or offset_seconds > 0 or
                    mode in ("by_status", "by_endpoint",
                             "errors", "latency", "bytes"))

    # 9.N.10 · Time window: [now - (range+offset), now - offset]
    # When offset_seconds == 0 → standard "last N seconds"
    # When offset_seconds == range_seconds → "the previous period"
    wheres_raw = [
        f"key_id IN {sub_keys}",
        "timestamp >= datetime('now', '-' || ? || ' seconds')",
        "timestamp <  datetime('now', '-' || ? || ' seconds')",
    ]
    args_raw: list = list(sub_args) + [range_seconds + offset_seconds, offset_seconds]
    if endpoint:
        wheres_raw.append("path LIKE ?")
        args_raw.append(f"%{endpoint}%")
    s_clause, s_args = _status_clause(status)
    if s_clause:
        wheres_raw.append(s_clause)
        args_raw.extend(s_args)

    # bucket_floor = strftime('%s', timestamp) / granularity * granularity
    bucket_expr = (
        "strftime('%Y-%m-%d %H:%M:%S', "
        "    (CAST(strftime('%s', timestamp) AS INTEGER) / ?) * ?, "
        "    'unixepoch'"
        ")"
    )

    series_meta: list[dict] = []
    buckets_out: list[dict] = []

    with get_conn() as c:
        if mode == "total" or not use_raw:
            # Aggregate path — sum from api_key_usage_minute. Granularity
            # must be ≥60s here, which we enforce above. Re-bucket if
            # granularity > 60s by integer-dividing the minute epoch.
            min_bucket_expr = (
                "strftime('%Y-%m-%d %H:%M:%S', "
                "    (CAST(strftime('%s', minute_ts) AS INTEGER) / ?) * ?, "
                "    'unixepoch'"
                ")"
            )
            sql = (
                f"SELECT {min_bucket_expr} AS bucket, "
                "       COALESCE(SUM(requests), 0)      AS requests, "
                "       COALESCE(SUM(errors), 0)        AS errors, "
                "       COALESCE(SUM(rate_limited), 0)  AS rate_limited, "
                "       COALESCE(SUM(quota_blocked), 0) AS quota_blocked, "
                "       COALESCE(SUM(bytes_in), 0)      AS bytes_in, "
                "       COALESCE(SUM(bytes_out), 0)     AS bytes_out, "
                "       COALESCE(SUM(latency_sum_ms), 0) AS latency_sum_ms, "
                "       COALESCE(SUM(latency_count), 0) AS latency_count "
                "FROM api_key_usage_minute "
                f"WHERE key_id IN {sub_keys} "
                "  AND minute_ts >= datetime('now', '-' || ? || ' seconds') "
                f"GROUP BY {min_bucket_expr} "
                "ORDER BY bucket ASC"
            )
            args = [g, g] + list(sub_args) + [range_seconds, g, g]
            rows = c.execute(sql, args).fetchall()
            for r in rows:
                d = dict(r)
                buckets_out.append({
                    "t": d["bucket"],
                    "values": {
                        "requests":      int(d.get("requests") or 0),
                        "errors":        int(d.get("errors") or 0),
                        "rate_limited":  int(d.get("rate_limited") or 0),
                        "quota_blocked": int(d.get("quota_blocked") or 0),
                        "bytes_in":      int(d.get("bytes_in") or 0),
                        "bytes_out":     int(d.get("bytes_out") or 0),
                        "latency_avg_ms": (
                            round(float(d.get("latency_sum_ms") or 0)
                                   / max(int(d.get("latency_count") or 1), 1), 1)
                            if (d.get("latency_count") or 0) > 0 else None
                        ),
                    },
                })
            series_meta = [
                {"key": "requests",     "label": "Requests",     "color_token": "ink"},
                {"key": "errors",       "label": "5xx errors",   "color_token": "danger"},
                {"key": "rate_limited", "label": "429 rate-limited", "color_token": "warn"},
            ]
        elif mode == "by_status":
            sql = (
                f"SELECT {bucket_expr} AS bucket, "
                "       SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) AS s2xx, "
                "       SUM(CASE WHEN status_code >= 300 AND status_code < 400 THEN 1 ELSE 0 END) AS s3xx, "
                "       SUM(CASE WHEN status_code >= 400 AND status_code < 500 AND status_code <> 429 THEN 1 ELSE 0 END) AS s4xx, "
                "       SUM(CASE WHEN status_code = 429 THEN 1 ELSE 0 END) AS s429, "
                "       SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS s5xx "
                "FROM api_key_request_log "
                f"WHERE {' AND '.join(wheres_raw)} "
                f"GROUP BY {bucket_expr} "
                "ORDER BY bucket ASC"
            )
            args = [g, g] + args_raw + [g, g]
            rows = c.execute(sql, args).fetchall()
            for r in rows:
                d = dict(r)
                buckets_out.append({
                    "t": d["bucket"],
                    "values": {
                        "2xx": int(d.get("s2xx") or 0),
                        "3xx": int(d.get("s3xx") or 0),
                        "4xx": int(d.get("s4xx") or 0),
                        "429": int(d.get("s429") or 0),
                        "5xx": int(d.get("s5xx") or 0),
                    },
                })
            series_meta = [
                {"key": "2xx", "label": "Success (2xx)",   "color_token": "ok"},
                {"key": "3xx", "label": "Redirect (3xx)",  "color_token": "info"},
                {"key": "4xx", "label": "Client (4xx)",    "color_token": "warn"},
                {"key": "429", "label": "Rate-limited (429)", "color_token": "amber"},
                {"key": "5xx", "label": "Server (5xx)",    "color_token": "danger"},
            ]
        elif mode == "by_endpoint":
            # Top 5 endpoints + "Other". Two-pass:
            # 1) find top 5 paths in the window
            # 2) bucket: those 5 + everything else under "other"
            top_sql = (
                "SELECT path, COUNT(*) AS cnt FROM api_key_request_log "
                f"WHERE {' AND '.join(wheres_raw)} "
                "GROUP BY path ORDER BY cnt DESC LIMIT 5"
            )
            top_rows = c.execute(top_sql, args_raw).fetchall()
            top_paths = [dict(r)["path"] for r in top_rows]
            # Build case expressions:
            case_parts = []
            for i, p in enumerate(top_paths):
                case_parts.append(
                    f"SUM(CASE WHEN path = ? THEN 1 ELSE 0 END) AS p{i}"
                )
            if top_paths:
                paths_in = ",".join(["?"] * len(top_paths))
                case_parts.append(
                    f"SUM(CASE WHEN path NOT IN ({paths_in}) THEN 1 ELSE 0 END) AS other"
                )
            else:
                case_parts.append("0 AS other")
            sql = (
                f"SELECT {bucket_expr} AS bucket, " +
                ", ".join(case_parts) + " "
                "FROM api_key_request_log "
                f"WHERE {' AND '.join(wheres_raw)} "
                f"GROUP BY {bucket_expr} "
                "ORDER BY bucket ASC"
            )
            args = [g, g] + top_paths + (top_paths if top_paths else []) + args_raw + [g, g]
            rows = c.execute(sql, args).fetchall()
            for r in rows:
                d = dict(r)
                values = {top_paths[i]: int(d.get(f"p{i}") or 0)
                            for i in range(len(top_paths))}
                values["other"] = int(d.get("other") or 0)
                buckets_out.append({"t": d["bucket"], "values": values})
            series_meta = [
                {"key": p, "label": p, "color_token": f"series-{i}"}
                for i, p in enumerate(top_paths)
            ] + [{"key": "other", "label": "Other", "color_token": "muted"}]
        elif mode == "latency":
            # p50 / p95 / p99 / avg per bucket. We MUST read raw and
            # compute in Python — SQLite has no PERCENTILE_CONT.
            sql = (
                f"SELECT {bucket_expr} AS bucket, latency_ms "
                "FROM api_key_request_log "
                f"WHERE {' AND '.join(wheres_raw)} AND latency_ms IS NOT NULL "
                "ORDER BY bucket ASC, latency_ms ASC"
            )
            args = [g, g] + args_raw
            rows = c.execute(sql, args).fetchall()
            from collections import defaultdict as _dd
            grouped: dict[str, list[int]] = _dd(list)
            for r in rows:
                d = dict(r)
                grouped[d["bucket"]].append(int(d["latency_ms"]))
            for bucket in sorted(grouped.keys()):
                lats = grouped[bucket]
                avg = sum(lats) / len(lats) if lats else None
                buckets_out.append({
                    "t": bucket,
                    "values": {
                        "p50":   _percentile(lats, 50),
                        "p95":   _percentile(lats, 95),
                        "p99":   _percentile(lats, 99),
                        "avg":   round(avg, 1) if avg is not None else None,
                        "count": len(lats),
                    },
                })
            series_meta = [
                {"key": "p50", "label": "p50",  "color_token": "ink"},
                {"key": "p95", "label": "p95",  "color_token": "warn"},
                {"key": "p99", "label": "p99",  "color_token": "danger"},
                {"key": "avg", "label": "avg",  "color_token": "muted"},
            ]
        elif mode == "errors":
            # 9.N.5 fix · ALL error classes (4xx + 5xx + 429) so the
            # chart isn't blank when only client errors exist. 429 is
            # split out from the 4xx bucket because it deserves its own
            # series — rate-limit pressure is a different signal.
            sql = (
                f"SELECT {bucket_expr} AS bucket, "
                "       COUNT(*) AS total, "
                "       SUM(CASE WHEN status_code = 429 THEN 1 ELSE 0 END) AS s429, "
                "       SUM(CASE WHEN status_code >= 400 AND status_code < 500 AND status_code != 429 THEN 1 ELSE 0 END) AS s4xx, "
                "       SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS s5xx "
                "FROM api_key_request_log "
                f"WHERE {' AND '.join(wheres_raw)} "
                f"GROUP BY {bucket_expr} "
                "ORDER BY bucket ASC"
            )
            args = [g, g] + args_raw + [g, g]
            rows = c.execute(sql, args).fetchall()
            for r in rows:
                d = dict(r)
                total = int(d.get("total") or 0)
                s429 = int(d.get("s429") or 0)
                s4xx = int(d.get("s4xx") or 0)
                s5xx = int(d.get("s5xx") or 0)
                err_count = s429 + s4xx + s5xx
                err_rate = round((err_count / total) * 100.0, 2) if total > 0 else 0.0
                buckets_out.append({
                    "t": d["bucket"],
                    "values": {
                        "4xx": s4xx, "429": s429, "5xx": s5xx,
                        "error_rate_pct": err_rate, "total": total,
                    },
                })
            series_meta = [
                {"key": "4xx", "label": "4xx (client)", "color_token": "amber"},
                {"key": "5xx", "label": "5xx (server)", "color_token": "danger"},
                {"key": "429", "label": "429 (limit)",  "color_token": "warn"},
            ]
        elif mode == "bytes":
            sql = (
                f"SELECT {bucket_expr} AS bucket, "
                "       COALESCE(SUM(bytes_in), 0) AS bytes_in, "
                "       COALESCE(SUM(bytes_out), 0) AS bytes_out "
                "FROM api_key_request_log "
                f"WHERE {' AND '.join(wheres_raw)} "
                f"GROUP BY {bucket_expr} "
                "ORDER BY bucket ASC"
            )
            args = [g, g] + args_raw + [g, g]
            rows = c.execute(sql, args).fetchall()
            for r in rows:
                d = dict(r)
                buckets_out.append({
                    "t": d["bucket"],
                    "values": {
                        "bytes_in":  int(d.get("bytes_in") or 0),
                        "bytes_out": int(d.get("bytes_out") or 0),
                    },
                })
            series_meta = [
                {"key": "bytes_in",  "label": "In",  "color_token": "info"},
                {"key": "bytes_out", "label": "Out", "color_token": "ok"},
            ]

    from datetime import datetime as _dt, timezone as _tz
    return {
        "range_seconds": range_seconds,
        "granularity_seconds": g,
        "mode": mode,
        "buckets": buckets_out,
        "series_meta": series_meta,
        "computed_at": _dt.now(_tz.utc).isoformat(),
    }


def compute_usage_top(*, user_id: int, range_seconds: int, dim: str,
                       limit: int, key_id: Optional[str] = None,
                       env: Optional[str] = None,
                       status: Optional[str] = None,
                       endpoint: Optional[str] = None) -> dict:
    """Top-N ranking. Returns rows sorted DESC by count. See routes_usage.

    9.N.6.f · Now accepts `status` and `endpoint` filters so a global drill
    (e.g. "status: 4xx" from the donut lightbox) actually narrows every
    top panel. Previously these filters were silently dropped here.
    """
    user_id = int(user_id)
    range_seconds = max(60, int(range_seconds))
    limit = max(1, min(int(limit), 50))
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)

    wheres = [
        f"r.key_id IN {sub_keys}",
        "r.timestamp >= datetime('now', '-' || ? || ' seconds')",
    ]
    args: list = list(sub_args) + [range_seconds]
    # 9.N.6.f · status / endpoint filters wired through to all dims
    if endpoint:
        wheres.append("r.path LIKE ?")
        args.append(f"%{endpoint}%")
    s_clause, s_args = _status_clause(status, alias='r')
    if s_clause:
        wheres.append(s_clause)
        args.extend(s_args)

    items: list[dict] = []
    total = 0

    with get_conn() as c:
        # Total for percentage normalisation.
        total_row = c.execute(
            "SELECT COUNT(*) AS n FROM api_key_request_log r "
            f"WHERE {' AND '.join(wheres)}",
            args
        ).fetchone()
        total = int(dict(total_row or {}).get("n") or 0)

        if dim == "keys":
            # Group by key_id; join api_keys for friendly name + env.
            sql = (
                "SELECT r.key_id AS public_id, k.name AS name, "
                "       k.environment AS env, COUNT(*) AS cnt, "
                "       SUM(CASE WHEN r.status_code >= 500 THEN 1 ELSE 0 END) AS errors, "
                "       COALESCE(AVG(r.latency_ms), 0) AS avg_latency_ms "
                "FROM api_key_request_log r "
                "LEFT JOIN api_keys k ON k.public_id = r.key_id "
                f"WHERE {' AND '.join(wheres)} "
                "GROUP BY r.key_id ORDER BY cnt DESC LIMIT ?"
            )
            for r in c.execute(sql, args + [limit]).fetchall():
                d = dict(r)
                items.append({
                    "label": d.get("name") or d.get("public_id"),
                    "public_id": d.get("public_id"),
                    "env": d.get("env"),
                    "count": int(d.get("cnt") or 0),
                    "pct": (round((d.get("cnt") or 0) / total * 100.0, 1)
                            if total > 0 else 0.0),
                    "extra": {
                        "errors": int(d.get("errors") or 0),
                        "avg_latency_ms": round(d.get("avg_latency_ms") or 0, 1),
                    },
                })
        elif dim == "endpoints":
            sql = (
                "SELECT r.path AS path, COUNT(*) AS cnt, "
                "       SUM(CASE WHEN r.status_code >= 500 THEN 1 ELSE 0 END) AS errors, "
                "       COALESCE(AVG(r.latency_ms), 0) AS avg_latency_ms "
                "FROM api_key_request_log r "
                f"WHERE {' AND '.join(wheres)} "
                "GROUP BY r.path ORDER BY cnt DESC LIMIT ?"
            )
            for r in c.execute(sql, args + [limit]).fetchall():
                d = dict(r)
                items.append({
                    "label": d.get("path") or "?",
                    "path": d.get("path"),
                    "count": int(d.get("cnt") or 0),
                    "pct": (round((d.get("cnt") or 0) / total * 100.0, 1)
                            if total > 0 else 0.0),
                    "extra": {
                        "errors": int(d.get("errors") or 0),
                        "avg_latency_ms": round(d.get("avg_latency_ms") or 0, 1),
                    },
                })
        elif dim == "ips":
            sql = (
                "SELECT COALESCE(r.ip, '(unknown)') AS ip, COUNT(*) AS cnt "
                "FROM api_key_request_log r "
                f"WHERE {' AND '.join(wheres)} "
                "GROUP BY r.ip ORDER BY cnt DESC LIMIT ?"
            )
            for r in c.execute(sql, args + [limit]).fetchall():
                d = dict(r)
                items.append({
                    "label": d.get("ip") or "(unknown)",
                    "count": int(d.get("cnt") or 0),
                    "pct": (round((d.get("cnt") or 0) / total * 100.0, 1)
                            if total > 0 else 0.0),
                    "extra": {},
                })
        elif dim == "statuses":
            sql = (
                "SELECT r.status_code AS status, COUNT(*) AS cnt "
                "FROM api_key_request_log r "
                f"WHERE {' AND '.join(wheres)} "
                "GROUP BY r.status_code ORDER BY cnt DESC LIMIT ?"
            )
            for r in c.execute(sql, args + [limit]).fetchall():
                d = dict(r)
                items.append({
                    "label": str(d.get("status") or "?"),
                    "status": d.get("status"),
                    "count": int(d.get("cnt") or 0),
                    "pct": (round((d.get("cnt") or 0) / total * 100.0, 1)
                            if total > 0 else 0.0),
                    "extra": {},
                })
        elif dim == "methods":
            sql = (
                "SELECT r.method AS method, COUNT(*) AS cnt "
                "FROM api_key_request_log r "
                f"WHERE {' AND '.join(wheres)} "
                "GROUP BY r.method ORDER BY cnt DESC LIMIT ?"
            )
            for r in c.execute(sql, args + [limit]).fetchall():
                d = dict(r)
                items.append({
                    "label": d.get("method") or "?",
                    "method": d.get("method"),
                    "count": int(d.get("cnt") or 0),
                    "pct": (round((d.get("cnt") or 0) / total * 100.0, 1)
                            if total > 0 else 0.0),
                    "extra": {},
                })
        elif dim == "error_codes":
            # 9.H · D7 — include status-derived pseudo-codes when error_code
            # is empty. The alias name must NOT collide with the source
            # column name `error_code` or Turso/SQLite groups by the column
            # (not the CASE), collapsing all 4xx into one bucket. Use
            # `display_code` as the unambiguous alias.
            sql = (
                "SELECT "
                "  CASE "
                "    WHEN r.error_code IS NOT NULL AND r.error_code != '' "
                "      THEN r.error_code "
                "    WHEN r.status_code = 429 THEN 'rate_limited' "
                "    WHEN r.status_code = 404 THEN 'not_found' "
                "    WHEN r.status_code = 401 THEN 'auth_invalid' "
                "    WHEN r.status_code = 403 THEN 'forbidden' "
                "    WHEN r.status_code = 422 THEN 'unprocessable' "
                "    WHEN r.status_code >= 500 THEN 'server_error' "
                "    WHEN r.status_code >= 400 THEN 'bad_request' "
                "    ELSE 'http_' || r.status_code "
                "  END AS display_code, "
                "  MAX(r.status_code) AS sample_status, "
                "  COUNT(*) AS cnt "
                "FROM api_key_request_log r "
                f"WHERE {' AND '.join(wheres)} "
                "  AND r.status_code >= 400 "
                "GROUP BY display_code "
                "ORDER BY cnt DESC LIMIT ?"
            )
            for r in c.execute(sql, args + [limit]).fetchall():
                d = dict(r)
                items.append({
                    "label": d.get("display_code") or "unknown",
                    "error_code": d.get("display_code"),
                    "count": int(d.get("cnt") or 0),
                    "pct": (round((d.get("cnt") or 0) / total * 100.0, 1)
                            if total > 0 else 0.0),
                    "extra": {"sample_status": d.get("sample_status")},
                })

    from datetime import datetime as _dt, timezone as _tz
    return {
        "range_seconds": range_seconds, "dim": dim, "limit": limit,
        "items": items, "total_for_pct": total,
        "computed_at": _dt.now(_tz.utc).isoformat(),
    }


def list_user_request_log(*, user_id: int, range_seconds: int,
                            limit: int = 50,
                            cursor: Optional[int] = None,
                            key_id: Optional[str] = None,
                            method: Optional[str] = None,
                            status: Optional[str] = None,
                            endpoint: Optional[str] = None,
                            env: Optional[str] = None) -> list[dict]:
    """Account-wide raw request log with filters.

    Same shape as `list_key_requests` but joined to api_keys for friendly
    name + env. Ordered by id DESC (newest first).
    """
    user_id = int(user_id)
    range_seconds = max(60, int(range_seconds))
    limit = max(1, min(int(limit), 10000))
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)

    wheres = [
        f"r.key_id IN {sub_keys}",
        "r.timestamp >= datetime('now', '-' || ? || ' seconds')",
    ]
    args: list = list(sub_args) + [range_seconds]
    if cursor:
        wheres.append("r.id < ?"); args.append(int(cursor))
    if method:
        wheres.append("r.method = ?"); args.append(method.upper())
    if endpoint:
        sub = str(endpoint)[:128].replace("%", r"\%").replace("_", r"\_")
        wheres.append("r.path LIKE ?"); args.append(f"%{sub}%")
    s_clause, s_args = _status_clause(status)
    if s_clause:
        wheres.append(s_clause.replace("status_code", "r.status_code"))
        args.extend(s_args)

    sql = (
        "SELECT r.id, r.timestamp, r.key_id AS key_public_id, "
        "       k.name AS key_name, k.environment AS env, "
        "       r.method, r.path, r.status_code, r.error_code, "
        "       r.latency_ms, r.bytes_in, r.bytes_out, r.ip, r.ua, r.via "
        "FROM api_key_request_log r "
        "LEFT JOIN api_keys k ON k.public_id = r.key_id "
        f"WHERE {' AND '.join(wheres)} "
        "ORDER BY r.id DESC LIMIT ?"
    )
    args.append(limit)
    with get_conn() as c:
        try:
            return [dict(r) for r in c.execute(sql, args).fetchall()]
        except Exception:
            return []


def compute_minute_detail(*, user_id: int, minute_ts: str,
                           key_id: Optional[str] = None,
                           endpoint: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: int = 100) -> dict:
    """One-minute drill-down. Returns:
      {
        "minute_ts": "...",
        "aggregate": {... row from api_key_usage_minute or computed ...},
        "requests": [... up to `limit` raw rows ...],
        "computed_at": "..."
      }
    """
    user_id = int(user_id)
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, None)
    limit = max(1, min(int(limit), 500))

    # The minute window in raw log is [minute_ts, minute_ts+60s).
    # We use SQL date arithmetic so timezones don't drift.
    next_minute_sql = "datetime(?, '+1 minutes')"

    wheres = [
        f"r.key_id IN {sub_keys}",
        "r.timestamp >= ?",
        f"r.timestamp <  {next_minute_sql}",
    ]
    args: list = list(sub_args) + [minute_ts, minute_ts]
    if endpoint:
        sub = str(endpoint)[:128].replace("%", r"\%").replace("_", r"\_")
        wheres.append("r.path LIKE ?"); args.append(f"%{sub}%")
    s_clause, s_args = _status_clause(status)
    if s_clause:
        wheres.append(s_clause.replace("status_code", "r.status_code"))
        args.extend(s_args)

    with get_conn() as c:
        # Aggregate row (one per key matching the minute_ts; if key_id is
        # not given we sum across keys for the user).
        agg_sql = (
            "SELECT COALESCE(SUM(requests), 0)        AS requests, "
            "       COALESCE(SUM(errors), 0)          AS errors, "
            "       COALESCE(SUM(rate_limited), 0)    AS rate_limited, "
            "       COALESCE(SUM(quota_blocked), 0)   AS quota_blocked, "
            "       COALESCE(SUM(bytes_in), 0)        AS bytes_in, "
            "       COALESCE(SUM(bytes_out), 0)       AS bytes_out, "
            "       COALESCE(SUM(latency_sum_ms), 0)  AS latency_sum_ms, "
            "       COALESCE(SUM(latency_count), 0)   AS latency_count, "
            "       MAX(latency_p50_ms)               AS p50, "
            "       MAX(latency_p95_ms)               AS p95, "
            "       MAX(latency_p99_ms)               AS p99 "
            "FROM api_key_usage_minute "
            f"WHERE key_id IN {sub_keys} AND minute_ts = ?"
        )
        agg = dict(c.execute(agg_sql, list(sub_args) + [minute_ts]).fetchone() or {})

        # Raw rows that landed in this minute.
        list_sql = (
            "SELECT r.id, r.timestamp, r.key_id AS key_public_id, "
            "       k.name AS key_name, r.method, r.path, "
            "       r.status_code, r.error_code, r.latency_ms, "
            "       r.bytes_in, r.bytes_out, r.ip, r.ua, r.via "
            "FROM api_key_request_log r "
            "LEFT JOIN api_keys k ON k.public_id = r.key_id "
            f"WHERE {' AND '.join(wheres)} "
            "ORDER BY r.id DESC LIMIT ?"
        )
        try:
            rows = [dict(r) for r in c.execute(list_sql, args + [limit]).fetchall()]
        except Exception:
            rows = []

    from datetime import datetime as _dt, timezone as _tz
    return {
        "minute_ts": minute_ts,
        "key_id": key_id, "endpoint": endpoint, "status": status,
        "aggregate": agg,
        "requests": rows,
        "request_count_shown": len(rows),
        "computed_at": _dt.now(_tz.utc).isoformat(),
    }


def get_request_log_row(*, user_id: int, row_id: int) -> Optional[dict]:
    """Single request log row, joined with api_keys for friendly fields.
    Returns None if not found or doesn't belong to this user.

    9.I — used by the request-detail drawer.
    9.N.8 — extended to return payload + stage_timings columns.
    """
    user_id = int(user_id)
    row_id = int(row_id)
    sub_keys, sub_args = _build_user_keys_where(user_id, None, None)
    sql = (
        "SELECT r.id, r.timestamp, r.key_id AS key_public_id, "
        "       k.name AS key_name, k.environment AS env, "
        "       r.method, r.path, r.status_code, r.error_code, "
        "       r.latency_ms, r.bytes_in, r.bytes_out, r.ip, r.ua, r.via, "
        "       r.headers_in_json, r.headers_out_json, "
        "       r.body_in_preview, r.body_out_preview, "
        "       r.body_in_ctype, r.body_out_ctype, "
        "       r.body_in_truncated, r.body_out_truncated, "
        "       r.stage_timings_json "
        "FROM api_key_request_log r "
        "LEFT JOIN api_keys k ON k.public_id = r.key_id "
        f"WHERE r.id = ? AND r.key_id IN {sub_keys}"
    )
    with get_conn() as c:
        try:
            row = c.execute(sql, [row_id] + list(sub_args)).fetchone()
            return dict(row) if row else None
        except Exception:
            return None


def get_burst_context(*, user_id: int, key_id: str, timestamp: str,
                       row_id: int, window_seconds: float = 1.0,
                       max_neighbours: int = 30) -> dict:
    """9.N.8 · Burst-context query for the request-detail drawer.

    Returns:
      {
        'cluster_size': int,   # total requests by this key within ±window_seconds
        'neighbours': [        # the actual rows in time order, capped at max_neighbours
          {id, timestamp, method, path, status_code, latency_ms, is_current},
          ...
        ],
        'is_burst': bool,      # True if cluster_size >= 5
      }
    """
    user_id = int(user_id)
    row_id = int(row_id)
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, None)
    # Use SQLite's strftime to compute a time window around `timestamp`.
    # timestamp comes in as "YYYY-MM-DD HH:MM:SS.ffffff" UTC.
    sql_count = (
        "SELECT COUNT(*) as n "
        "FROM api_key_request_log r "
        f"WHERE r.key_id IN {sub_keys} "
        "  AND ABS((julianday(r.timestamp) - julianday(?)) * 86400.0) <= ?"
    )
    sql_rows = (
        "SELECT r.id, r.timestamp, r.method, r.path, r.status_code, r.latency_ms "
        "FROM api_key_request_log r "
        f"WHERE r.key_id IN {sub_keys} "
        "  AND ABS((julianday(r.timestamp) - julianday(?)) * 86400.0) <= ? "
        "ORDER BY r.timestamp ASC "
        "LIMIT ?"
    )
    with get_conn() as c:
        try:
            n_row = c.execute(sql_count, list(sub_args) + [timestamp, window_seconds]).fetchone()
            cluster_size = int(n_row[0]) if n_row else 0
            rows = c.execute(
                sql_rows,
                list(sub_args) + [timestamp, window_seconds, max_neighbours],
            ).fetchall()
            neighbours = []
            for r in rows:
                d = dict(r)
                d["is_current"] = (int(d.get("id") or 0) == row_id)
                neighbours.append(d)
            return {
                "cluster_size": cluster_size,
                "neighbours": neighbours,
                "is_burst": cluster_size >= 5,
                "window_seconds": float(window_seconds),
            }
        except Exception:
            return {"cluster_size": 0, "neighbours": [],
                    "is_burst": False, "window_seconds": float(window_seconds)}


def get_endpoint_health_buckets(*, user_id: int, path: str,
                                  bucket_count: int = 20,
                                  total_seconds: int = 3600) -> dict:
    """9.N.8 · Endpoint health mini-chart data — sparklines for requests /
    errors / p50 / p95 over the last `total_seconds`, divided into
    `bucket_count` time buckets.

    Returns:
      {
        'buckets': [
          {t_start, t_end, count, errors, p50, p95},
          ...   # length == bucket_count, oldest first
        ],
        'totals': {requests, errors, p50, p95},
      }
    """
    user_id = int(user_id)
    bucket_count = max(4, min(int(bucket_count), 60))
    total_seconds = max(60, int(total_seconds))
    bucket_seconds = total_seconds / bucket_count
    sub_keys, sub_args = _build_user_keys_where(user_id, None, None)

    # We fetch all matching latencies in the window, then bucketize in
    # Python — simpler than SQL window functions and SQLite-portable.
    sql = (
        "SELECT r.timestamp, r.status_code, r.latency_ms "
        "FROM api_key_request_log r "
        f"WHERE r.key_id IN {sub_keys} "
        "  AND r.path = ? "
        "  AND r.timestamp >= datetime('now', '-' || ? || ' seconds') "
        "ORDER BY r.timestamp ASC"
    )
    buckets = [
        {"i": i, "count": 0, "errors": 0, "_lats": []}
        for i in range(bucket_count)
    ]
    all_lats: list = []
    n_total = 0
    n_errs = 0
    try:
        from datetime import datetime as _dt
        with get_conn() as c:
            rows = c.execute(sql, list(sub_args) + [path, total_seconds]).fetchall()
        if not rows:
            # No data: return empty buckets so the UI still renders the
            # axes + zero baseline.
            for b in buckets:
                b["p50"] = 0
                b["p95"] = 0
                del b["_lats"]
                del b["i"]
            return {
                "buckets": buckets,
                "totals": {"requests": 0, "errors": 0, "p50": 0, "p95": 0},
                "sample_size": 0,
            }
        # Anchor: the OLDEST row's wall-clock relative to "now" determines
        # bucket boundaries. We use server-clock now() — simplest.
        now_unix = _dt.utcnow().timestamp()
        window_start = now_unix - total_seconds
        for r in rows:
            try:
                ts = r["timestamp"]
                # Parse "YYYY-MM-DD HH:MM:SS.ffffff" → unix
                t_unix = _dt.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp()
            except Exception:
                continue
            if t_unix < window_start:
                continue
            offset = t_unix - window_start
            bucket_i = int(offset / bucket_seconds)
            if bucket_i < 0: bucket_i = 0
            if bucket_i >= bucket_count: bucket_i = bucket_count - 1
            b = buckets[bucket_i]
            b["count"] += 1
            sc = int(r["status_code"] or 0)
            if sc >= 400:
                b["errors"] += 1
                n_errs += 1
            n_total += 1
            lat = r["latency_ms"]
            if lat is not None:
                lat = int(lat)
                b["_lats"].append(lat)
                all_lats.append(lat)
        # Compute per-bucket percentiles
        def _pct(arr, p):
            if not arr: return 0
            s = sorted(arr)
            idx = max(0, min(len(s) - 1, int(p * (len(s) - 1))))
            return s[idx]
        for b in buckets:
            b["p50"] = _pct(b["_lats"], 0.50)
            b["p95"] = _pct(b["_lats"], 0.95)
            del b["_lats"]
            del b["i"]
        # Totals
        totals = {
            "requests": n_total,
            "errors": n_errs,
            "p50": _pct(all_lats, 0.50),
            "p95": _pct(all_lats, 0.95),
        }
        return {
            "buckets": buckets,
            "totals": totals,
            "sample_size": n_total,
            "bucket_seconds": bucket_seconds,
            "total_seconds": total_seconds,
        }
    except Exception:
        return {"buckets": [], "totals": {"requests":0,"errors":0,"p50":0,"p95":0},
                "sample_size": 0}


def get_recent_endpoint_latencies(*, user_id: int, path: str,
                                    limit: int = 200) -> list:
    """9.N.7.f · Returns a list of recent latency_ms values for the given
    endpoint path across all keys owned by this user. Used by the request-
    detail drawer to compute p50/p95 baselines for context — so the user
    can see how their inspected request compares to typical performance
    on that same endpoint.
    """
    user_id = int(user_id)
    limit = max(1, min(int(limit), 1000))
    sub_keys, sub_args = _build_user_keys_where(user_id, None, None)
    sql = (
        "SELECT latency_ms FROM api_key_request_log r "
        f"WHERE r.key_id IN {sub_keys} "
        "  AND r.path = ? "
        "  AND r.latency_ms IS NOT NULL "
        "ORDER BY r.timestamp DESC "
        "LIMIT ?"
    )
    with get_conn() as c:
        try:
            rows = c.execute(sql, list(sub_args) + [path, limit]).fetchall()
            return [int(r[0]) for r in rows if r[0] is not None]
        except Exception:
            return []


def compute_latency_bucket_drill(*, user_id: int, range_seconds: int,
                                   min_ms: int, max_ms: Optional[int],
                                   key_id: Optional[str] = None,
                                   env: Optional[str] = None,
                                   limit: int = 100) -> dict:
    """9.I — Drill from a clicked latency-histogram bar.

    Returns rows within [min_ms, max_ms) — `max_ms=None` means open-ended
    (the >5000ms bar). Same envelope shape as compute_minute_detail.
    """
    user_id = int(user_id)
    range_seconds = max(60, int(range_seconds))
    limit = max(1, min(int(limit), 500))
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)

    wheres = [
        f"r.key_id IN {sub_keys}",
        "r.timestamp >= datetime('now', '-' || ? || ' seconds')",
        "r.latency_ms IS NOT NULL",
        "r.latency_ms >= ?",
    ]
    args: list = list(sub_args) + [range_seconds, int(min_ms)]
    if max_ms is not None:
        wheres.append("r.latency_ms < ?")
        args.append(int(max_ms))

    sql = (
        "SELECT r.id, r.timestamp, r.key_id AS key_public_id, "
        "       k.name AS key_name, r.method, r.path, "
        "       r.status_code, r.error_code, r.latency_ms, "
        "       r.bytes_in, r.bytes_out, r.ip, r.ua, r.via "
        "FROM api_key_request_log r "
        "LEFT JOIN api_keys k ON k.public_id = r.key_id "
        f"WHERE {' AND '.join(wheres)} "
        "ORDER BY r.latency_ms DESC LIMIT ?"
    )
    with get_conn() as c:
        rows = [dict(r) for r in c.execute(sql, args + [limit]).fetchall()]
    from datetime import datetime as _dt, timezone as _tz
    return {
        "min_ms": int(min_ms), "max_ms": max_ms,
        "requests": rows, "request_count_shown": len(rows),
        "computed_at": _dt.now(_tz.utc).isoformat(),
    }


# ─── 9.N.5 · per-endpoint detail (feeds 4 of 6 new charts) ──────────────────
def compute_per_endpoint_detail(*, user_id: int, range_seconds: int,
                                  limit: int = 20,
                                  spark_buckets: int = 20,
                                  key_id: Optional[str] = None,
                                  env: Optional[str] = None,
                                  status: Optional[str] = None,
                                  endpoint: Optional[str] = None) -> list:
    """Returns one row per endpoint with count, error_rate, p50/p95/p99,
    and a sparkline-shape (small N-bucket time-binned activity).
    Sorted by count desc.

    Powers spark-grid, treemap, boxplot, and quadrant charts on the dashboard.

    9.N.6.f · Honours status + endpoint filters so a global drill cascades.
    """
    user_id = int(user_id)
    range_seconds = max(60, int(range_seconds))
    limit = max(1, min(int(limit), 50))
    spark_buckets = max(4, min(int(spark_buckets), 60))
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)

    # Aggregate stats per path within window
    args: list = list(sub_args) + [range_seconds]
    base_where = (
        f"r.key_id IN {sub_keys} "
        f"AND r.timestamp >= datetime('now', '-' || ? || ' seconds')"
    )
    # 9.N.6.f · Append optional status / endpoint filter
    if endpoint:
        base_where += " AND r.path LIKE ?"
        args.append(f"%{endpoint}%")
    s_clause, s_args = _status_clause(status, alias='r')
    if s_clause:
        base_where += " AND " + s_clause
        args.extend(s_args)

    # Top-N paths by request count
    sql_top = (
        "SELECT r.path AS path, COUNT(*) AS count, "
        "       SUM(CASE WHEN r.status_code >= 400 THEN 1 ELSE 0 END) AS errors, "
        "       SUM(r.bytes_in) AS bytes_in, "
        "       SUM(r.bytes_out) AS bytes_out, "
        "       AVG(r.latency_ms) AS avg_lat "
        "FROM api_key_request_log r "
        f"WHERE {base_where} "
        "GROUP BY r.path "
        "ORDER BY count DESC LIMIT ?"
    )
    with get_conn() as c:
        top_rows = [dict(r) for r in c.execute(sql_top, args + [limit]).fetchall()]

    if not top_rows:
        return []

    # For each endpoint: compute percentiles (p50, p90, p95, p99, p25, p10)
    # and a per-bucket sparkline.
    bucket_sec = max(60, int(range_seconds // spark_buckets))
    out: list = []
    with get_conn() as c:
        for tr in top_rows:
            path = tr["path"]
            # Percentile sort
            lat_sql = (
                "SELECT r.latency_ms FROM api_key_request_log r "
                f"WHERE {base_where} AND r.path = ? AND r.latency_ms IS NOT NULL "
                "ORDER BY r.latency_ms ASC"
            )
            lats = [int(r[0] if isinstance(r, tuple) else r["latency_ms"]) for r in
                    c.execute(lat_sql, args + [path]).fetchall()]
            def pctl(p):
                if not lats:
                    return 0
                i = int(max(0, min(len(lats) - 1, round(p * (len(lats) - 1)))))
                return lats[i]
            # Outliers: > p95 * 1.5
            p95 = pctl(0.95)
            p90 = pctl(0.90)
            outliers = [v for v in lats if v > p95 * 1.5 and v != p95][:6]

            # Sparkline: per-bucket count
            spark_sql = (
                "SELECT "
                "  CAST((strftime('%s','now') - strftime('%s', r.timestamp)) / ? AS INTEGER) AS bidx, "
                "  COUNT(*) AS c "
                "FROM api_key_request_log r "
                f"WHERE {base_where} AND r.path = ? "
                "GROUP BY bidx ORDER BY bidx DESC LIMIT ?"
            )
            spark_rows = c.execute(spark_sql, [bucket_sec] + args + [path, spark_buckets]).fetchall()
            spark = [0] * spark_buckets
            for sr in spark_rows:
                bi = int(sr[0] if isinstance(sr, tuple) else sr["bidx"])
                cv = int(sr[1] if isinstance(sr, tuple) else sr["c"])
                if 0 <= bi < spark_buckets:
                    spark[bi] = cv
            spark.reverse()  # oldest → newest left→right

            count = int(tr["count"])
            errors = int(tr["errors"] or 0)
            err_rate = (errors / count) if count else 0.0

            out.append({
                "label": path,
                "path": path,
                "count": count,
                "errors": errors,
                "error_rate": round(err_rate, 4),
                "pct": round((count / sum(int(r["count"]) for r in top_rows)) * 100, 2) if top_rows else 0,
                "bytes_in": int(tr.get("bytes_in") or 0),
                "bytes_out": int(tr.get("bytes_out") or 0),
                "avg_lat_ms": int(tr.get("avg_lat") or 0),
                "p10": pctl(0.10),
                "p25": pctl(0.25),
                "p50": pctl(0.50),
                "p75": pctl(0.75),
                "p90": p90,
                "p95": p95,
                "p99": pctl(0.99),
                "max": (lats[-1] if lats else 0),
                "outliers": outliers,
                "sparkline": spark,
            })
    return out


# ─── 9.N.5 · heatmap calendar (day-of-week × hour-of-day) ────────────────────
def compute_usage_heatmap_calendar(*, user_id: int, range_seconds: int,
                                     key_id: Optional[str] = None,
                                     env: Optional[str] = None) -> list:
    """Returns activity cells grouped by (day_of_week × hour_of_day).
    `day_idx`: 0=Monday … 6=Sunday (ISO).
    """
    user_id = int(user_id)
    range_seconds = max(60, int(range_seconds))
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)
    # SQLite strftime: %w gives 0=Sunday..6=Saturday. We remap to ISO 0=Mon..6=Sun.
    sql = (
        "SELECT "
        "  ((CAST(strftime('%w', r.timestamp) AS INTEGER) + 6) % 7) AS day_idx, "
        "  CAST(strftime('%H', r.timestamp) AS INTEGER) AS hour, "
        "  COUNT(*) AS count "
        "FROM api_key_request_log r "
        f"WHERE r.key_id IN {sub_keys} "
        "  AND r.timestamp >= datetime('now', '-' || ? || ' seconds') "
        "GROUP BY day_idx, hour "
        "ORDER BY day_idx ASC, hour ASC"
    )
    args = list(sub_args) + [range_seconds]
    with get_conn() as c:
        rows = c.execute(sql, args).fetchall()
    cells = []
    for r in rows:
        try:
            d = int(r["day_idx"] if not isinstance(r, tuple) else r[0])
            h = int(r["hour"]    if not isinstance(r, tuple) else r[1])
            v = int(r["count"]   if not isinstance(r, tuple) else r[2])
            cells.append({"day_idx": d, "hour": h, "count": v})
        except (KeyError, ValueError, TypeError):
            continue
    return cells


# ─── 9.N.5.f · multi-mode calendar (week/month/year/years) ──────────────────
def compute_usage_heatmap_calendar_multi(*, user_id: int, mode: str,
                                           key_id: Optional[str] = None,
                                           env: Optional[str] = None) -> dict:
    """Heatmap calendar in 4 zoom modes. Returns cells in a generic
    {row, col, count, label} shape the renderer can lay out.

    week  → 7 rows (Mon..Sun) × 24 cols (hours), window = last 7 days
    month → 5 rows × 7 cols (days of week × week-of-month), window = last 30 days
    year  → 1 row × 12 cols (months), window = last 365 days
    years → 1 row × 5 cols (years), window = last 5 years
    """
    user_id = int(user_id)
    sub_keys, sub_args = _build_user_keys_where(user_id, key_id, env)
    out_cells: list = []
    rows, cols = 1, 1
    col_labels: list = []
    row_labels: list = []

    if mode == "week":
        # Last 7 days × 24 hours. SQLite strftime %w: 0=Sun..6=Sat; remap to ISO.
        sql = (
            "SELECT "
            "  ((CAST(strftime('%w', r.timestamp) AS INTEGER) + 6) % 7) AS day_idx, "
            "  CAST(strftime('%H', r.timestamp) AS INTEGER) AS hour, "
            "  COUNT(*) AS count "
            "FROM api_key_request_log r "
            f"WHERE r.key_id IN {sub_keys} "
            "  AND r.timestamp >= datetime('now', '-7 days') "
            "GROUP BY day_idx, hour "
            "ORDER BY day_idx ASC, hour ASC"
        )
        args = list(sub_args)
        rows, cols = 7, 24
        row_labels = ["MON","TUE","WED","THU","FRI","SAT","SUN"]
        col_labels = [str(h).zfill(2) for h in range(24)]
        with get_conn() as c:
            for r in c.execute(sql, args).fetchall():
                try:
                    out_cells.append({
                        "row":   int(r["day_idx"] if not isinstance(r, tuple) else r[0]),
                        "col":   int(r["hour"]    if not isinstance(r, tuple) else r[1]),
                        "count": int(r["count"]   if not isinstance(r, tuple) else r[2]),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
        # back-compat: also expose day_idx + hour aliases
        for cl in out_cells:
            cl["day_idx"] = cl["row"]; cl["hour"] = cl["col"]

    elif mode == "month":
        # Last 30 days, one cell per day, laid out as 5 rows × 7 cols
        # (each row = a "week" anchored to today). row 0 = current week.
        sql = (
            "SELECT "
            "  CAST(julianday('now') - julianday(r.timestamp) AS INTEGER) AS days_ago, "
            "  COUNT(*) AS count "
            "FROM api_key_request_log r "
            f"WHERE r.key_id IN {sub_keys} "
            "  AND r.timestamp >= datetime('now', '-35 days') "
            "GROUP BY days_ago "
            "ORDER BY days_ago ASC"
        )
        args = list(sub_args)
        rows, cols = 5, 7
        row_labels = [f"week -{i}" for i in range(5)]
        col_labels = ["MON","TUE","WED","THU","FRI","SAT","SUN"]
        with get_conn() as c:
            for r in c.execute(sql, args).fetchall():
                try:
                    da = int(r["days_ago"] if not isinstance(r, tuple) else r[0])
                    if da >= 35 or da < 0:
                        continue
                    out_cells.append({
                        "row":   da // 7,
                        "col":   6 - (da % 7),  # 6=today-aligned right edge
                        "count": int(r["count"] if not isinstance(r, tuple) else r[1]),
                        "days_ago": da,
                    })
                except (KeyError, ValueError, TypeError):
                    continue

    elif mode == "year":
        # Last 365 days, one cell per month
        sql = (
            "SELECT "
            "  strftime('%Y-%m', r.timestamp) AS ym, "
            "  COUNT(*) AS count "
            "FROM api_key_request_log r "
            f"WHERE r.key_id IN {sub_keys} "
            "  AND r.timestamp >= datetime('now', '-365 days') "
            "GROUP BY ym "
            "ORDER BY ym ASC"
        )
        args = list(sub_args)
        rows, cols = 1, 12
        # column labels for last 12 months ending at now
        from datetime import datetime as _dt, timedelta as _td
        now = _dt.utcnow()
        col_labels = []
        ym_to_col = {}
        for i in range(11, -1, -1):
            # month i months back
            m = now.month - i
            y = now.year
            while m <= 0:
                m += 12; y -= 1
            col_labels.append(f"{y:04d}-{m:02d}")
            ym_to_col[f"{y:04d}-{m:02d}"] = 11 - i
        with get_conn() as c:
            for r in c.execute(sql, args).fetchall():
                try:
                    ym = r["ym"] if not isinstance(r, tuple) else r[0]
                    col = ym_to_col.get(ym)
                    if col is None:
                        continue
                    out_cells.append({
                        "row": 0, "col": col,
                        "count": int(r["count"] if not isinstance(r, tuple) else r[1]),
                        "label": ym,
                    })
                except (KeyError, ValueError, TypeError):
                    continue

    elif mode == "years":
        # Last 5 years, one cell per year
        sql = (
            "SELECT "
            "  strftime('%Y', r.timestamp) AS yr, "
            "  COUNT(*) AS count "
            "FROM api_key_request_log r "
            f"WHERE r.key_id IN {sub_keys} "
            "  AND r.timestamp >= datetime('now', '-5 years') "
            "GROUP BY yr "
            "ORDER BY yr ASC"
        )
        args = list(sub_args)
        rows, cols = 1, 5
        from datetime import datetime as _dt
        now = _dt.utcnow()
        col_labels = [str(now.year - i) for i in range(4, -1, -1)]
        yr_to_col = {col_labels[i]: i for i in range(5)}
        with get_conn() as c:
            for r in c.execute(sql, args).fetchall():
                try:
                    yr = r["yr"] if not isinstance(r, tuple) else r[0]
                    col = yr_to_col.get(yr)
                    if col is None:
                        continue
                    out_cells.append({
                        "row": 0, "col": col,
                        "count": int(r["count"] if not isinstance(r, tuple) else r[1]),
                        "label": yr,
                    })
                except (KeyError, ValueError, TypeError):
                    continue

    return {
        "mode": mode,
        "rows": rows, "cols": cols,
        "row_labels": row_labels, "col_labels": col_labels,
        "cells": out_cells,
    }


def hard_delete_console_api_key(*, user_id: int, public_id: str) -> bool:
    """Hard-delete a key. Spec §7.1 line 2738: requires status IN
    ('disabled','expired'). Sets deleted_at + status='deleted'.

    Returns True on success, False if not found or in wrong status.
    """
    now = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "UPDATE api_keys SET deleted_at = ?, status = 'deleted' "
            "WHERE user_id = ? AND public_id = ? "
            "AND deleted_at IS NULL "
            "AND status IN ('disabled','expired')",
            (now, user_id, public_id)
        )
        rowcount = cur.rowcount if hasattr(cur, "rowcount") else None
    return rowcount == 1


# ─────────────────────────────────────────────────────────────────────────
# Drone perception scans (Phase 2 batch 1 of the public /api/)
# ─────────────────────────────────────────────────────────────────────────
# One scan = one drone frame, persisted with its GPS + diagnosis + the
# full APINResult JSON for audit. The scan_uid is the stable public
# identifier; the integer `id` is internal. Helpers below never expose
# the integer `id` to callers — only `scan_uid`.

_SCAN_UID_PREFIX = "scn_"
# Image-bytes persistence: if drone operators set persist_image=True we
# store the raw frame for replay; otherwise the column is NULL and the
# image is forgotten as soon as inference returns (privacy-preserving
# default, mirrors the /predict/full website behavior).
_SCAN_MAX_IMAGE_BYTES = 12_000_000


def _new_scan_uid() -> str:
    # PDA Phase-2: was token_hex(8) = 64 bits, birthday-bound to
    # collisions at ~4 billion rows. token_hex(16) = 128 bits is
    # collision-proof at any realistic scale (RFC 4122 UUID-equivalent
    # entropy). 32 hex chars + 4 prefix = 36-char total uid.
    return _SCAN_UID_PREFIX + _secrets.token_hex(16)


def _scan_row_to_dict(row, *, include_image: bool = False,
                      include_result: bool = False) -> dict:
    """Project a sqlite Row to a clean dict. By default omits the heavy
    image_bytes BLOB and the full result_json (the list endpoint never
    wants them — they bloat the response by ~150 KB per row)."""
    d = {
        "scan_uid":     row["scan_uid"],
        "flight_id":    row["flight_id"],
        "diagnosis":    row["diagnosis"],
        "confidence":   row["confidence"],
        "tier":         row["tier"],
        "severity":     row["severity"],
        "is_ood":       bool(row["is_ood"]),
        "geo": {
            "latitude":    row["latitude"],
            "longitude":   row["longitude"],
            "altitude_m":  row["altitude_m"],
            "heading_deg": row["heading_deg"],
            "accuracy_m":  row["accuracy_m"],
        },
        "captured_at":   row["captured_at"],
        "processed_at":  row["processed_at"],
        "processing_ms": row["processing_ms"],
        "image": {
            "sha256":      row["image_sha256"],
            "n_bytes":     row["image_n_bytes"],
            "persisted":   row["image_bytes"] is not None,
        },
    }
    if include_image and row["image_bytes"] is not None:
        d["image_bytes_b64"] = base64.b64encode(row["image_bytes"]
                                                ).decode("ascii")
    if include_result:
        try:
            d["result"] = json.loads(row["result_json"])
        except Exception:
            d["result"] = None
    return d


def create_scan(*, user_id: int, api_key_id: Optional[int],
                flight_id: Optional[str],
                geo: dict,
                captured_at: str,
                image_sha256: str,
                image_bytes: Optional[bytes],
                result: dict,
                processing_ms: int) -> dict:
    """Persist one drone scan and return the public-facing record.

    `geo` MUST be pre-validated by the caller (range checks belong at
    the API boundary, not here — the DB just enforces NOT NULL on the
    two required fields). `result` is the full APINResult dict.

    Returns the projection produced by `_scan_row_to_dict` (no image
    blob, no full result_json — the route handler can re-merge those
    when wanted).
    """
    if not isinstance(geo, dict):
        raise ValueError("geo must be a dict")
    lat = geo.get("latitude")
    lon = geo.get("longitude")
    if lat is None or lon is None:
        raise ValueError("geo.latitude and geo.longitude are required")
    if image_bytes is not None and len(image_bytes) > _SCAN_MAX_IMAGE_BYTES:
        # Refuse to persist absurd blobs in the row — the API boundary
        # already enforces this same limit; this is belt-and-braces.
        raise ValueError(
            f"image_bytes exceeds the {_SCAN_MAX_IMAGE_BYTES} byte cap")

    now = _now_iso()
    severity = (result or {}).get("severity") or _derive_severity(result)
    diagnosis = (result or {}).get("diagnosis")
    confidence = float((result or {}).get("confidence", 0.0))
    tier = (result or {}).get("tier")
    is_ood = 1 if (result or {}).get("is_ood") else 0
    n_bytes = len(image_bytes) if image_bytes is not None else None
    result_json = json.dumps(result, separators=(",", ":"), default=str)

    # PDA Phase-2: retry on the (vanishingly rare) sha256-on-uid
    # collision so a single unlucky scan can never propagate as an
    # unhandled 500. 3 attempts is far more than 128-bit entropy
    # would ever need.
    with _write_lock, get_conn() as c:
        last_err = None
        for _attempt in range(3):
            scan_uid = _new_scan_uid()
            try:
                c.execute(
                    "INSERT INTO scans (scan_uid, user_id, api_key_id, "
                    "flight_id, diagnosis, confidence, tier, severity, "
                    "is_ood, latitude, longitude, altitude_m, "
                    "heading_deg, accuracy_m, captured_at, "
                    "image_sha256, image_bytes, image_n_bytes, "
                    "processed_at, processing_ms, result_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?)",
                    (scan_uid, int(user_id),
                     int(api_key_id) if api_key_id is not None else None,
                     flight_id,
                     diagnosis, confidence, tier, severity, is_ood,
                     float(lat), float(lon),
                     float(geo["altitude_m"])  if geo.get("altitude_m")  is not None else None,
                     float(geo["heading_deg"]) if geo.get("heading_deg") is not None else None,
                     float(geo["accuracy_m"])  if geo.get("accuracy_m")  is not None else None,
                     captured_at,
                     image_sha256, image_bytes, n_bytes,
                     now, int(processing_ms), result_json))
                break
            except sqlite3.IntegrityError as e:
                # UNIQUE constraint failed (extreme bad luck on uid).
                # Loop once more with a fresh uid.
                last_err = e
                continue
            except _LibsqlError as e:
                # Same in Turso mode.
                last_err = e
                continue
        else:
            # All retries exhausted — surface the failure to the route
            # handler, which will return service_unavailable / 503.
            raise RuntimeError(
                f"create_scan: 3 uid retries all hit UNIQUE constraint; "
                f"last_err={last_err!r}")
        row = c.execute(
            "SELECT * FROM scans WHERE scan_uid = ?", (scan_uid,)
        ).fetchone()
    return _scan_row_to_dict(row)


def _derive_severity(result: Optional[dict]) -> Optional[str]:
    """If the inference result didn't carry an explicit severity (the
    APIN ensemble doesn't always produce one), derive a reasonable
    proxy from confidence + tier. Returns one of mild | moderate |
    severe | None.

    PDA Phase-2: previously this returned "moderate" for confident
    tier-1 predictions REGARDLESS of whether the diagnosis was a
    healthy class. That meant a confident healthy-leaf prediction
    showed up as `severity = "moderate"` in the GeoJSON — colour-coded
    as a problem on the map. Now we short-circuit healthy diagnoses
    to None ("not applicable") before applying the confidence ladder.
    """
    if not isinstance(result, dict):
        return None
    if result.get("severity"):
        return str(result["severity"])
    if result.get("is_ood"):
        return None
    diag = str(result.get("diagnosis") or "")
    # Healthy classes have no severity. The canonical APIN class names
    # for healthy leaves all end in "_healthy" — okra_healthy and
    # brassica_healthy at present, plus any future healthy classes that
    # follow the same naming convention.
    if diag.endswith("_healthy"):
        return None
    conf = float(result.get("confidence", 0.0) or 0.0)
    tier = str(result.get("tier", "") or "")
    # Tier 1A/1B/1C: clean diagnoses — severity follows the confidence.
    # Tier 2/3: weaker; report as moderate. Tier 4/5: abstain.
    if tier.startswith("1"):
        if conf >= 0.85: return "moderate"   # confident but no severity head
        return "mild"
    if tier.startswith("2") or tier.startswith("3"):
        return "moderate"
    return None


def get_scan(user_id: int, scan_uid: str,
              *, include_image: bool = False,
              include_result: bool = False) -> Optional[dict]:
    """Look up a scan by uid. Returns None for not-found, not-owned,
    or soft-deleted scans — never reveals to a caller that someone
    ELSE owns a scan with that uid."""
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT * FROM scans WHERE scan_uid = ? AND user_id = ? "
                "AND deleted_at IS NULL",
                (scan_uid, int(user_id))).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return _scan_row_to_dict(row, include_image=include_image,
                              include_result=include_result)


def list_scans(user_id: int, *, page: int = 1, page_size: int = 50,
                flight_id: Optional[str] = None,
                diagnosis: Optional[str] = None,
                since_iso: Optional[str] = None,
                until_iso: Optional[str] = None) -> tuple[list[dict], int]:
    """Page through the caller's scans, newest first. Returns
    `(rows, total)`. Soft-deleted scans are filtered out.

    Filters are AND-ed. `diagnosis` is a canonical class name; mismatches
    return an empty list rather than an error (consistent with REST list
    semantics).
    """
    page      = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))

    where = ["user_id = ?", "deleted_at IS NULL"]
    args: list = [int(user_id)]
    if flight_id:
        where.append("flight_id = ?");  args.append(flight_id)
    if diagnosis:
        where.append("diagnosis = ?");  args.append(diagnosis)
    if since_iso:
        where.append("captured_at >= ?"); args.append(since_iso)
    if until_iso:
        where.append("captured_at <= ?"); args.append(until_iso)
    wsql = " AND ".join(where)

    try:
        with get_conn() as c:
            total_row = c.execute(
                f"SELECT COUNT(*) AS n FROM scans WHERE {wsql}",
                tuple(args)).fetchone()
            total = int(total_row["n"] if total_row else 0)
            offset = (page - 1) * page_size
            rows = c.execute(
                f"SELECT * FROM scans WHERE {wsql} "
                f"ORDER BY captured_at DESC, id DESC "
                f"LIMIT ? OFFSET ?",
                tuple(args) + (page_size, offset)).fetchall()
    except Exception:
        return [], 0
    return [_scan_row_to_dict(r) for r in rows], total


def delete_scan(user_id: int, scan_uid: str) -> bool:
    """Soft-delete a scan the caller owns. Returns True if a row was
    actually marked, False if the scan didn't exist, isn't theirs, was
    already deleted, OR if we couldn't read rowcount cleanly.

    PDA Phase-2: previously returned True on rowcount-read failure,
    which made `v1_scan_delete` emit a 200 'deleted' response for a
    scan that may not have been deleted at all. Conservative fallback
    is False — the caller then sees 404 and can retry, and the DB
    state matches the response.
    """
    now = _now_iso()
    with _write_lock, get_conn() as c:
        cur = c.execute(
            "UPDATE scans SET deleted_at = ? "
            "WHERE scan_uid = ? AND user_id = ? AND deleted_at IS NULL",
            (now, scan_uid, int(user_id)))
        try:
            return int(getattr(cur, "rowcount", 0) or 0) > 0
        except Exception:
            return False


# ═════════════════════════════════════════════════════════════════════════════
# Stage-1 v2 helpers · guest predictions + telemetry ingest + KPI aggregates
# ═════════════════════════════════════════════════════════════════════════════
#
# All helpers in this block follow the same conventions as the rest of auth_db:
#   - Use `with _write_lock, get_conn() as c:` for writes
#   - Use `with get_conn() as c:` for reads
#   - Return None / 0 / [] on failure, never raise into hot paths
#   - Idempotent where it matters (rollups, conversion event)
#
# Hot-path callers (record_guest_prediction, ingest_telemetry_batch) MUST be
# safe to call from a FastAPI BackgroundTask: log on failure, never raise.


def record_guest_prediction(
    guest_session_id: int,
    response: dict,
    *,
    image_bytes: Optional[bytes] = None,
    extras: Optional[dict] = None,
) -> Optional[int]:
    """Insert a guest_predictions row · mirrors record_prediction() for guests.

    `extras` is a dict of any v2 extension columns the caller already knows
    (browser_session_id, client_ip_hash, exif_*, signal_predictions, etc.).
    Unknown keys are silently ignored. NULL fields are fine.

    Returns the inserted row id, or None on failure. Never raises.
    """
    import json as _json
    try:
        summary = _extract_prediction_summary(response)
        img_hash = None
        if image_bytes:
            img_hash = hashlib.sha256(image_bytes).hexdigest()
        heatmap_b64 = None
        if isinstance(response, dict):
            heatmap_b64 = _extract_heatmap_b64(response)
            if heatmap_b64 and len(heatmap_b64) > _HEATMAP_MAX_CHARS:
                heatmap_b64 = None
        if isinstance(response, dict):
            slim = _strip_heavy_keys(response)
            response_json = _json.dumps(slim, default=str)
        else:
            response_json = _json.dumps({"_": str(response)[:500]})
        if len(response_json) > _RESPONSE_JSON_MAX:
            response_json = _json.dumps({
                "_truncated": True,
                "_original_size_bytes": len(response_json),
                "summary": summary,
            })

        e = extras or {}
        supported = {
            "api_key_id", "browser_session_id", "client_ip_hash",
            "user_agent_family", "client_country", "client_region", "client_city",
            "exif_camera_model", "exif_capture_timestamp",
            "exif_gps_lat", "exif_gps_lon", "exif_gps_accuracy_m",
            "image_perceptual_hash", "image_n_bytes", "image_width",
            "image_height", "image_mimetype",
            "signal_predictions", "gate_decision_path",
            "deployment_version", "model_weights_hash", "cold_start",
            "fallback_to_cpu", "gpu_used", "peak_vram_mb",
            "conformal_set", "conformal_set_size", "ood_flag",
            "calibration_warning", "predicted_top3",
            "validation_ms", "router_ms", "specialist_ms",
            "calibration_ms", "total_ms",
            "endpoint", "api_version", "request_id", "trace_id",
            "status_code", "error_class", "error_message",
            "review_status", "sampled_for_review", "confidence_outlier",
            "consent_to_research", "consent_to_share", "data_residency_region",
            "user_pseudoid", "treatment_advice_shown",
            "grad_cam_generated", "pdf_report_generated", "experiment_exposures",
        }
        cols = ["guest_session_id", "crop", "predicted_class", "confidence",
                "tier", "image_sha256", "image_bytes", "heatmap_b64",
                "response_json", "created_at"]
        vals = [int(guest_session_id), summary["crop"],
                summary["predicted_class"], summary["confidence"],
                summary["tier"], img_hash,
                sqlite3.Binary(image_bytes) if image_bytes else None,
                heatmap_b64, response_json, _now_iso()]
        for k, v in e.items():
            if k in supported:
                cols.append(k); vals.append(v)

        placeholders = ",".join(["?"] * len(cols))
        sql = "INSERT INTO guest_predictions (" + ",".join(cols) + ") VALUES (" + placeholders + ")"
        with _write_lock, get_conn() as c:
            cur = c.execute(sql, tuple(vals))
            return cur.lastrowid
    except Exception:
        import logging as _l
        _l.getLogger("apin_v2.auth").exception("record_guest_prediction failed")
        return None


# ─── KPI aggregates · backs the /api/stats/summary endpoint ──────────────────

def count_total_inferences() -> dict:
    """All-time inference count: user + guest + scan, alive only.
    Returns dict with keys total, user, guest, scan."""
    out = {"total": 0, "user": 0, "guest": 0, "scan": 0}
    try:
        with get_conn() as c:
            r = c.execute(
                "SELECT COUNT(*) AS n FROM predictions WHERE deleted_at IS NULL"
            ).fetchone()
            out["user"] = int(r["n"]) if r else 0
            r = c.execute(
                "SELECT COUNT(*) AS n FROM guest_predictions WHERE deleted_at IS NULL"
            ).fetchone()
            out["guest"] = int(r["n"]) if r else 0
            r = c.execute(
                "SELECT COUNT(*) AS n FROM scans WHERE deleted_at IS NULL"
            ).fetchone()
            out["scan"] = int(r["n"]) if r else 0
            out["total"] = out["user"] + out["guest"] + out["scan"]
    except Exception:
        pass
    return out


def top_disease_in_window(days: int = 7) -> Optional[dict]:
    """Most-predicted disease in the last N days across predictions + guest_predictions.
    Returns dict {class, count, share, window_days} or None if no traffic."""
    try:
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        cutoff = (now - _dt.timedelta(days=days)).isoformat()
        with get_conn() as c:
            row = c.execute(
                "SELECT predicted_class, COUNT(*) AS n FROM all_predictions "
                "WHERE deleted_at IS NULL AND created_at >= ? "
                "AND predicted_class IS NOT NULL "
                "GROUP BY predicted_class ORDER BY n DESC LIMIT 1",
                (cutoff,)
            ).fetchone()
            if not row or not row["n"]:
                return None
            total = c.execute(
                "SELECT COUNT(*) AS n FROM all_predictions "
                "WHERE deleted_at IS NULL AND created_at >= ?",
                (cutoff,)
            ).fetchone()
            total_n = int(total["n"]) if total else 0
            top_n = int(row["n"])
            return {
                "class": row["predicted_class"],
                "count": top_n,
                "share": round(top_n / total_n, 4) if total_n else 0.0,
                "window_days": days,
            }
    except Exception:
        return None


def live_sessions_by_route(window_s: int = 30) -> dict:
    """Stage 6 · powers the "Live now" KPI tile.

    Returns the count of browser_sessions whose last_heartbeat_at falls
    inside the last `window_s` seconds AND have not been explicitly
    closed (session_end_at is NULL), plus a per-route breakdown.

    The telemetry library flushes every 10 s and stamps last_heartbeat_at
    on each flush. 30 s (3× the flush interval) tolerates one missed
    flush without dropping a visitor; closed tabs are filtered out
    immediately by the session_end_at check.

    Stage 6.1 · the window was 90 s (9× tolerance) which made closed
    tabs linger in the count for up to a minute and a half — the user
    reported tiles still showing 5 sessions three minutes after closing
    everything. The explicit session_end_at filter handles graceful
    close + the 30 s window is the safety net for crash / unplug.

    Returns
    -------
    {
      'active_count': int,
      'by_route': [{'route': '/pipeline', 'count': 2}, …],
      'as_of':       <iso utc>,
    }

    Routes with zero active sessions are omitted (user requirement: "if
    0 we do not need to mention").  Private dashboard routes ARE included
    in the breakdown (user said: not a privacy risk to surface the count).
    """
    out = {"active_count": 0, "by_route": [], "as_of": _now_iso()}
    try:
        import datetime as _dt
        cutoff = (
            _dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(seconds=int(window_s))
        ).isoformat()
        with get_conn() as c:
            # Stage 6.1 · "active" = heartbeat in the window AND not
            # explicitly ended. The explicit `session_end_at` filter
            # makes graceful closes (pagehide) disappear instantly; the
            # 30 s window handles ungraceful drops (network loss, crash,
            # browser tab kill where the beacon never landed).
            # Total active count
            r = c.execute(
                "SELECT COUNT(*) AS n FROM browser_sessions "
                "WHERE last_heartbeat_at IS NOT NULL "
                "AND last_heartbeat_at >= ? "
                "AND (session_end_at IS NULL OR session_end_at = '' "
                "     OR session_end_at < last_heartbeat_at)",
                (cutoff,),
            ).fetchone()
            out["active_count"] = int(r["n"]) if r else 0
            # Per-route breakdown · same filter set
            rows = c.execute(
                "SELECT current_route AS route, COUNT(*) AS n "
                "FROM browser_sessions "
                "WHERE last_heartbeat_at IS NOT NULL "
                "AND last_heartbeat_at >= ? "
                "AND current_route IS NOT NULL "
                "AND current_route != '' "
                "AND (session_end_at IS NULL OR session_end_at = '' "
                "     OR session_end_at < last_heartbeat_at) "
                "GROUP BY current_route "
                "ORDER BY n DESC, current_route ASC",
                (cutoff,),
            ).fetchall()
            out["by_route"] = [
                {"route": r["route"], "count": int(r["n"])}
                for r in rows if int(r["n"]) > 0
            ]
    except Exception:
        import logging as _l
        _l.getLogger("apin_v2.auth").exception("live_sessions_by_route failed")
    return out


def live_activity_summary() -> dict:
    """Last-60s + last-5min + last-hour activity from all_predictions VIEW.
    Powers the Live activity KPI tile."""
    out = {
        "last_60s_count": 0, "last_5min_count": 0, "last_hour_count": 0,
        "median_latency_ms": None, "error_rate_pct": None, "as_of": _now_iso(),
    }
    try:
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        c60 = (now - _dt.timedelta(seconds=60)).isoformat()
        c5m = (now - _dt.timedelta(minutes=5)).isoformat()
        c1h = (now - _dt.timedelta(hours=1)).isoformat()
        with get_conn() as c:
            r = c.execute(
                "SELECT COUNT(*) AS n FROM all_predictions "
                "WHERE deleted_at IS NULL AND created_at >= ?",
                (c60,)).fetchone()
            out["last_60s_count"] = int(r["n"]) if r else 0
            r = c.execute(
                "SELECT COUNT(*) AS n FROM all_predictions "
                "WHERE deleted_at IS NULL AND created_at >= ?",
                (c5m,)).fetchone()
            out["last_5min_count"] = int(r["n"]) if r else 0
            r = c.execute(
                "SELECT COUNT(*) AS n FROM all_predictions "
                "WHERE deleted_at IS NULL AND created_at >= ?",
                (c1h,)).fetchone()
            out["last_hour_count"] = int(r["n"]) if r else 0
            rows = c.execute(
                "SELECT total_ms FROM predictions WHERE deleted_at IS NULL "
                "AND created_at >= ? AND total_ms IS NOT NULL ORDER BY total_ms",
                (c5m,)).fetchall()
            if rows:
                vals = [int(r["total_ms"]) for r in rows]
                out["median_latency_ms"] = vals[len(vals) // 2]
            row = c.execute(
                "SELECT COUNT(*) AS n, SUM(CASE WHEN error_class IS NOT NULL "
                "THEN 1 ELSE 0 END) AS errs FROM predictions "
                "WHERE deleted_at IS NULL AND created_at >= ?",
                (c1h,)).fetchone()
            if row and row["n"]:
                errs = int(row["errs"] or 0)
                out["error_rate_pct"] = round(errs * 100.0 / int(row["n"]), 1)
    except Exception:
        pass
    return out


# ─── Telemetry ingest (batch endpoint backend) ───────────────────────────────

_NEW_ID_LOCK = threading.Lock()
_NEW_ID_LAST_TS = 0
_NEW_ID_COUNTER = 0


def _new_id() -> str:
    """Time-ordered sortable id · 22 chars URL-safe base64.

    Audit finding (PDA-2.2): same-ms calls weren't strictly sortable because
    we used 10 random bytes after the ms timestamp. Two ids generated in the
    same ms had random tails that could swap order.

    Fix: prepend a 2-byte monotonic counter (per-ms reset) before the random
    tail. This guarantees strict lex sortability across all calls.

      bytes: [ 6 ts_ms | 2 counter | 8 random ] → 16 bytes → 22 base64 chars
    """
    import secrets, datetime as _dt, base64
    global _NEW_ID_LAST_TS, _NEW_ID_COUNTER
    ts_ms = int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)
    with _NEW_ID_LOCK:
        if ts_ms == _NEW_ID_LAST_TS:
            _NEW_ID_COUNTER += 1
            # Wrap counter at 65535; in practice we won't exceed 100/ms
            if _NEW_ID_COUNTER > 65535:
                _NEW_ID_COUNTER = 0
        else:
            _NEW_ID_LAST_TS = ts_ms
            _NEW_ID_COUNTER = 0
        counter = _NEW_ID_COUNTER
    ts_b = ts_ms.to_bytes(6, "big")
    cnt_b = counter.to_bytes(2, "big")
    rnd_b = secrets.token_bytes(8)
    return base64.urlsafe_b64encode(ts_b + cnt_b + rnd_b).decode("ascii").rstrip("=")


def _retry_transient_oe(fn, *, attempts=4, base_sleep=0.02):
    """Stage 2.5 [PDA-3] · Retry a callable on transient sqlite3
    OperationalError messages that occur under WAL+concurrent-writer
    contention. busy_timeout only retries SQLITE_BUSY; this catches the
    "attempt to write a readonly database" (SQLITE_READONLY_RECOVERY)
    and "database is locked" variants too.

    Backoff is short-and-linear (20/40/60 ms) — enough for the WAL to
    settle but not long enough to make the inference response visibly
    slower."""
    import time as _time
    _RETRYABLE = (
        "database is locked",
        "attempt to write a readonly database",
        "database is busy",
    )
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if i < attempts - 1 and any(m in msg for m in _RETRYABLE):
                _time.sleep(base_sleep * (i + 1))
                continue
            raise


def upsert_browser_session(session_id: str, fields: dict) -> bool:
    """Insert browser_sessions row if missing, else UPDATE dynamic fields.
    Idempotent + safe to call from every batch.

    Stage 2.5 [PDA-3] · wrapped in _retry_transient_oe so a single
    SQLITE_READONLY_RECOVERY / SQLITE_BUSY does not lose a session row
    (and, via FK, the page_view + event children of that batch)."""
    def _go():
        with _write_lock, get_conn() as c:
            r = c.execute("SELECT id FROM browser_sessions WHERE id = ?",
                          (session_id,)).fetchone()
            if r is None:
                cols = ["id", "session_start_at"]
                vals = [session_id, fields.get("session_start_at") or _now_iso()]
                static_fields = (
                    "user_id", "guest_session_id", "user_pseudoid",
                    "device_type", "device_os", "device_os_version",
                    "device_model", "device_browser", "device_browser_version",
                    "user_agent_family", "user_agent_raw",
                    "screen_width", "screen_height", "viewport_width",
                    "viewport_height", "pixel_ratio", "timezone", "locale",
                    "connection_type", "network_effective_type",
                    "cpu_cores", "memory_gb", "is_pwa_installed",
                    "referrer_host", "referrer_path", "referrer_url",
                    "entry_url", "utm_source", "utm_medium", "utm_campaign",
                    "utm_term", "utm_content", "gclid", "fbclid",
                    "ip_country", "ip_region", "ip_city", "client_ip_hash",
                    "is_returning_user",
                    # Stage 6 · these were only in dyn_fields, which meant a
                    # FRESH session's first upsert left last_heartbeat_at +
                    # current_route NULL — and live_sessions_by_route filters
                    # `WHERE last_heartbeat_at IS NOT NULL`, so brand-new
                    # sessions were invisible to the "Live now" tile until
                    # the second flush. Including them in the INSERT path
                    # too means the first heartbeat is immediately visible.
                    "last_heartbeat_at", "current_route",
                    # Stage 6.1c · session_end_at was also dyn_fields-only,
                    # which silently dropped the end signal on any FIRST-flush
                    # beacon (e.g. tab closed before its 300ms first heartbeat
                    # completed the INSERT — the pagehide beacon then races,
                    # whichever runs first triggers INSERT, and if the close
                    # beacon wins it carries session_end_at but it was lost
                    # because the static_fields list didn't accept it). The
                    # symptom: closed tabs lingered on Live Now until the 30 s
                    # window aged them out, instead of disappearing on the
                    # explicit end signal.
                    "session_end_at",
                )
                for k in static_fields:
                    if k in fields:
                        cols.append(k); vals.append(fields[k])
                placeholders = ",".join(["?"] * len(cols))
                sql = ("INSERT INTO browser_sessions (" + ",".join(cols)
                       + ") VALUES (" + placeholders + ")")
                c.execute(sql, tuple(vals))
            else:
                set_clauses = []
                params: list = []
                dyn_fields = (
                    "session_end_at", "last_heartbeat_at",
                    "total_active_ms", "total_idle_ms", "total_hidden_ms",
                    "page_count", "click_count", "inference_count",
                    "error_count", "api_call_count", "exit_url",
                    "user_id", "user_pseudoid",
                    "current_route",   # Stage 6 · live-now per-route
                )
                for k in dyn_fields:
                    if k in fields:
                        set_clauses.append(k + " = ?"); params.append(fields[k])
                if set_clauses:
                    params.append(session_id)
                    c.execute(
                        "UPDATE browser_sessions SET " + ", ".join(set_clauses)
                        + " WHERE id = ?", tuple(params))
            return True

    try:
        return _retry_transient_oe(_go)
    except Exception:
        import logging as _l
        _l.getLogger("apin_v2.auth").exception("upsert_browser_session failed")
        return False


_PV_COLS = {
    "id", "browser_session_id", "user_id", "guest_session_id",
    "page_url", "page_title", "page_route", "navigation_type",
    "referrer_url", "referrer_host",
    "entered_at", "left_at",
    "active_duration_ms", "idle_duration_ms", "hidden_duration_ms",
    "max_scroll_depth_pct", "scroll_milestones_reached", "scroll_pause_points",
    "ttfb_ms", "fcp_ms", "lcp_ms", "cls", "tti_ms", "inp_ms",
    "click_count", "error_count", "api_call_count",
    "bounce", "engagement_score",
}
_CLICK_COLS = {
    "id", "browser_session_id", "page_view_id", "user_id", "guest_session_id",
    "target_tag", "target_id", "target_classes", "target_text",
    "target_xpath", "target_data_attrs",
    "click_x_viewport", "click_y_viewport", "click_x_page", "click_y_page",
    "viewport_width_at_click", "viewport_height_at_click", "viewport_y_pct",
    "modifier_keys", "click_type",
    "was_rage_click", "was_dead_click", "element_visible_seconds_before_click",
    "ms_since_page_view", "ms_since_session_start",
    "led_to_navigation", "led_to_modal_open", "triggered_api_call_ids",
    "occurred_at",
}
_IMPR_COLS = {
    "id", "browser_session_id", "page_view_id", "user_id", "guest_session_id",
    "target_id", "target_classes", "target_text", "target_xpath",
    "intersection_ratio_at_first_visible", "visibility_duration_ms",
    "ms_since_page_view", "led_to_interaction", "occurred_at",
}
_EVT_COLS = {
    "id", "browser_session_id", "page_view_id", "user_id", "guest_session_id",
    "event_type", "event_name", "event_version", "properties",
    "ms_since_page_view", "ms_since_session_start", "occurred_at",
}
_API_COLS = {
    "id", "browser_session_id", "page_view_id", "user_id", "guest_session_id",
    "endpoint", "method",
    "request_body_size_bytes", "response_body_size_bytes", "status_code",
    "client_latency_ms", "server_latency_ms", "network_latency_ms",
    "error_type", "retry_count", "triggered_by", "cache_hit",
    "idempotency_key", "request_id", "occurred_at",
}
_INFTEL_COLS = {
    "id", "browser_session_id", "user_id", "guest_session_id",
    "inference_type", "prediction_id", "guest_prediction_id", "scan_id",
    "file_selected_at", "file_size_at_select",
    "upload_started_at", "upload_completed_at", "upload_duration_ms",
    "api_request_sent_at", "api_response_received_at", "result_rendered_at",
    "perceived_total_ms", "preview_shown", "result_expanded",
    "gradcam_viewed", "pdf_exported", "shared", "feedback_given",
    "cancelled_by_user", "user_next_action", "occurred_at",
}
_ERR_COLS = {
    "id", "browser_session_id", "page_view_id", "user_id", "guest_session_id",
    "error_type", "error_message", "error_stack",
    "source_file", "source_line", "source_column", "url", "user_agent",
    "shown_to_user", "recovery_action", "occurred_at",
}
_GOAL_COLS = {
    "user_id", "guest_session_id", "browser_session_id",
    "goal_name", "achieved_at", "time_from_session_start_ms",
    "time_from_signup_ms", "goal_value",
    "utm_source_at_first_visit", "utm_medium_at_first_visit",
    "utm_campaign_at_first_visit", "touchpoint_count", "attribution_path",
}
_EXP_COLS = {
    "user_id", "guest_session_id", "browser_session_id",
    "experiment_name", "variant", "exposed_at",
    "page_view_id", "properties",
}


def _bulk_insert(table: str, items: list, allowed: set) -> int:
    """Insert list-of-dicts into table, filtering to allowed cols.
    Tolerates per-row failures (logs + continues).

    Stage 2.5 [PDA-3] · the connection acquire + per-row inserts each
    retry once on a transient OperationalError (SQLITE_BUSY /
    SQLITE_READONLY_RECOVERY) so brief WAL contention doesn't silently
    drop rows."""
    n = 0
    if not items:
        return 0

    def _connect_and_insert_all():
        nonlocal n
        with _write_lock, get_conn() as c:
            for it in items:
                if not isinstance(it, dict):
                    continue
                if "id" in allowed and not it.get("id"):
                    it["id"] = _new_id()
                cols = [k for k in it.keys() if k in allowed]
                if not cols:
                    continue
                vals = [it[k] for k in cols]
                placeholders = ",".join(["?"] * len(cols))
                sql = ("INSERT INTO " + table + " (" + ",".join(cols)
                       + ") VALUES (" + placeholders + ")")

                def _do_insert():
                    c.execute(sql, tuple(vals))

                try:
                    _retry_transient_oe(_do_insert)
                    n += 1
                except sqlite3.IntegrityError:
                    # FK / UNIQUE failure — log but keep going so a single
                    # bad row doesn't kill the rest of the batch.
                    import logging as _l
                    _l.getLogger("apin_v2.auth").warning(
                        "_bulk_insert(%s) skipped row: IntegrityError "
                        "(probably FK to missing parent)",
                        table,
                    )
                except Exception:
                    import logging as _l
                    _l.getLogger("apin_v2.auth").exception(
                        "_bulk_insert(%s) row failed", table)

    try:
        _retry_transient_oe(_connect_and_insert_all)
    except Exception:
        import logging as _l
        _l.getLogger("apin_v2.auth").exception(
            "_bulk_insert(%s) failed at the outer scope", table)
    return n


def ingest_telemetry_batch(batch: dict) -> dict:
    """Validate + insert a batch of telemetry events.
    See _qa_tmp/_telemetry_batch_shape.md for the expected payload shape.
    Returns counts per table inserted. Never raises."""
    counts = {
        "page_views": 0, "clicks": 0, "impressions": 0, "events": 0,
        "api_calls": 0, "inference_telemetry": 0, "errors": 0,
        "goals": 0, "experiments_exposures": 0, "session_upserted": 0,
    }
    if not isinstance(batch, dict):
        return counts
    try:
        s = batch.get("session")
        # Stage 2.5 [PDA-3] · if the batch declares a session and the upsert
        # fails (even after retries), child rows that FK to this session
        # would all fail with FOREIGN KEY constraint failed. Skip them
        # rather than silently dropping each one inside _bulk_insert.
        session_upsert_failed = False
        if isinstance(s, dict) and s.get("id"):
            if upsert_browser_session(str(s["id"]), s):
                counts["session_upserted"] = 1
            else:
                session_upsert_failed = True
        if session_upsert_failed:
            return counts
        for table_name, items_key, col_set in [
            ("page_views", "page_views", _PV_COLS),
            ("clicks", "clicks", _CLICK_COLS),
            ("impressions", "impressions", _IMPR_COLS),
            ("events", "events", _EVT_COLS),
            ("api_calls", "api_calls", _API_COLS),
            ("inference_telemetry", "inference_telemetry", _INFTEL_COLS),
            ("errors", "errors", _ERR_COLS),
            ("goals", "goals", _GOAL_COLS),
            ("experiments_exposures", "experiments_exposures", _EXP_COLS),
        ]:
            items = batch.get(items_key) or []
            if not isinstance(items, list):
                continue
            counts[table_name] = _bulk_insert(table_name, items, col_set)
    except Exception:
        import logging as _l
        _l.getLogger("apin_v2.auth").exception("ingest_telemetry_batch failed")
    return counts


# ─── Conversion tracking · guest → registered user ───────────────────────────

def mark_guest_converted(guest_session_id: int, user_id: int) -> bool:
    """Called from /auth/signup when the user had an active guest cookie.
    Sets converted_to_user_id + converted_from_guest_session_id AND inserts
    a 'guest_to_user_conversion' goal row.

    Stage 2.5 [F-6] · idempotent against re-application. If this guest
    session has already been attributed to a different user, we do NOT
    overwrite — the first conversion wins. The function returns False in
    that case so the caller can log the attempted double-attribution
    without raising. Re-applying the same (guest_id, user_id) pair is a
    safe no-op and returns True.
    """
    try:
        with _write_lock, get_conn() as c:
            existing = c.execute(
                "SELECT converted_to_user_id FROM guest_sessions WHERE id = ?",
                (int(guest_session_id),),
            ).fetchone()
            if existing is None:
                # No such guest session — caller passed a stale id.
                return False
            already = existing["converted_to_user_id"]
            if already is not None and int(already) != int(user_id):
                # Already attributed to a different user; refuse to overwrite.
                import logging as _l
                _l.getLogger("apin_v2.auth").warning(
                    "mark_guest_converted refused: guest %s already "
                    "converted to user %s (asked for %s)",
                    guest_session_id, already, user_id,
                )
                return False
            if already is not None and int(already) == int(user_id):
                # Idempotent re-application — nothing to do, but report success.
                return True
            now = _now_iso()
            c.execute(
                "UPDATE guest_sessions SET converted_to_user_id = ?, "
                "converted_at = ? WHERE id = ? AND converted_to_user_id IS NULL",
                (int(user_id), now, int(guest_session_id)))
            c.execute(
                "UPDATE users SET converted_from_guest_session_id = ? "
                "WHERE id = ? AND converted_from_guest_session_id IS NULL",
                (int(guest_session_id), int(user_id)))
            c.execute(
                "INSERT INTO goals (user_id, guest_session_id, goal_name, "
                "achieved_at) VALUES (?, ?, ?, ?)",
                (int(user_id), int(guest_session_id),
                 "guest_to_user_conversion", now))
            return True
    except Exception:
        import logging as _l
        _l.getLogger("apin_v2.auth").exception("mark_guest_converted failed")
        return False
