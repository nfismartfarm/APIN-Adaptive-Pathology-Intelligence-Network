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
    libSQL's execute() runs one statement at a time. Chunks that contain
    only blank lines or `--` comments are dropped."""
    out = []
    for chunk in script.split(";"):
        body = [ln for ln in chunk.splitlines()
                if ln.strip() and not ln.strip().startswith("--")]
        if body:
            out.append(chunk.strip())
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
        return _ShimCursor(rs)

    def executescript(self, script):
        stmts = _split_sql(script)
        with self._lock:
            self._client.batch(stmts)

    def commit(self):
        pass   # libsql autocommits each execute()

    def close(self):
        pass   # the ClientSync is shared — never closed per request


# The shim has its OWN lock — distinct from _write_lock. Write helpers do
# `with _write_lock, get_conn() as c: c.execute(...)`, so if the shim
# reused _write_lock it would try to acquire a non-reentrant lock twice
# on the same thread → deadlock. Lock order is always write_lock → shim
# lock (reads take only the shim lock), so there is no inversion.
_libsql_lock = threading.Lock()

# Build the Turso client + shared shim connection (once, at import).
_libsql_client = None
_libsql_conn = None
if _USE_TURSO:
    import libsql_client
    from libsql_client import LibsqlError as _LibsqlError  # noqa: F811
    if _TURSO_URL.startswith("file:"):
        # Local libSQL file — used by _turso_shim_test.py to exercise the
        # exact production code path without a Turso account. No auth, no
        # scheme rewrite.
        _libsql_client = libsql_client.create_client_sync(_TURSO_URL)
    else:
        # Remote Turso — force the HTTP transport (see _turso_http_url).
        _conn_url = _turso_http_url(_TURSO_URL)
        _libsql_client = libsql_client.create_client_sync(
            _conn_url, auth_token=(_TURSO_TOKEN or None))
    _libsql_conn = _ShimConn(_libsql_client, _libsql_lock)


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


def _ensure_db():
    """Apply the schema + idempotent migrations. Idempotent across both
    backends — `CREATE TABLE IF NOT EXISTS` and the introspected
    ADD COLUMN make re-running safe."""
    if _USE_TURSO:
        # Turso/libSQL — run each schema statement, then migrate.
        # journal_mode=WAL is a no-op on a remote DB, so it is skipped.
        try:
            _libsql_conn.executescript(SCHEMA_SQL)
            try:
                _libsql_conn.execute("PRAGMA foreign_keys = ON")
            except Exception:
                pass   # not all libSQL deployments honour this PRAGMA
            _migrate_predictions_blob_columns(_libsql_conn)
        except Exception as e:
            msg = str(e)
            hint = ""
            if any(s in msg for s in ("JWT", "nauthorized", "401", "400")):
                hint = ("\n  -> Turso rejected the credentials. Check the "
                        "TURSO_AUTH_TOKEN secret — it must be a current, "
                        "non-expired token for this database (regenerate "
                        "with: turso db tokens create <db-name>).")
            raise RuntimeError(
                f"Could not initialise the Turso database at "
                f"{_TURSO_URL!r}: {msg}{hint}") from e
    else:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as c:
            c.executescript(SCHEMA_SQL)
            c.execute("PRAGMA journal_mode = WAL;")  # better concurrency
            c.execute("PRAGMA foreign_keys = ON;")
            _migrate_predictions_blob_columns(c)
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
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
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
    """Create a session row, return the raw token (to be sent as cookie value)."""
    raw, h = _new_session_token()
    now = datetime.now(timezone.utc)
    expires = (now + SESSION_LIFETIME).isoformat()
    with _write_lock, get_conn() as c:
        c.execute(
            """INSERT INTO sessions
               (user_id, token_hash, user_agent, ip_addr, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, h, user_agent, ip_addr, now.isoformat(), expires),
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

        with _write_lock, get_conn() as c:
            cur = c.execute(
                """INSERT INTO predictions
                   (user_id, crop, predicted_class, confidence, tier,
                    image_sha256, image_bytes, heatmap_b64,
                    response_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (user_id, summary["crop"], summary["predicted_class"],
                 summary["confidence"], summary["tier"],
                 img_hash,
                 sqlite3.Binary(image_bytes) if image_bytes else None,
                 heatmap_b64,
                 response_json, _now_iso()),
            )
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
