"""Phase 9.A — Real usage telemetry recorder.

Replaces the long-standing `_record_usage` stub in
`scripts.apin_v2.account.auth_decorator`. This module is the data-plumbing
layer for the §6.3 / §6.4 tables (api_key_usage_minute + api_key_request_log)
and §4.2 api_keys lifecycle counters (last_used_at / request_count /
error_count / last_used_ip / last_used_ua).

Design
======
A per-request DB write would be O(N) inserts + updates per request which is
brutal for SQLite under any concurrency. The standard fix is a two-tier
buffer:

  1. In-process buffer  — `_UsageBuffer`
     `record()` is called from the decorator's `finally:` block. It is
     non-blocking, lock-protected, and never touches the DB. Worst case
     on crash: we lose <flush_interval> seconds of telemetry. The decorator
     swallows any exception this raises so a buggy buffer can never break
     a successful request.

  2. Background flusher — `UsageFlusher`
     Started once at app boot (lives next to `webhook_worker` in
     `apin_server._ensure_heartbeat`). It wakes every ~2 seconds, drains
     the buffer, and writes three things in one transaction:
       - BULK INSERT api_key_request_log (one row per request)
       - UPSERT api_key_usage_minute (one row per minute, accumulating)
       - UPDATE api_keys SET last_used_*, request_count+=, error_count+=

     Every 60 s it also runs a percentile rollup pass that selects the
     last 5 minutes of `api_key_request_log` (so freshly-flushed rows
     are picked up even if the previous tick raced past them) and writes
     exact p50 / p95 / p99 into `api_key_usage_minute`. The rollup is
     idempotent — repeated runs over the same minute produce the same
     result.

Concurrency
===========
The buffer uses a single `threading.Lock`. `record()` does O(1) work
under the lock — append a small dict and bump in-memory counters. The
flusher acquires the lock only to swap out the buffer (atomic move-then-
release), then writes to the DB without holding it. This means a hot
request never waits more than a few µs on flush contention.

What we DON'T do
================
- No t-digest streaming percentiles. The rollup approach is simpler and
  exact for short windows. Trade-off: percentiles for the current minute
  appear ~60 s after the requests happened. Acceptable for an
  observability dashboard.
- No external queue (Redis, Kafka). Single process, single SQLite/Turso
  connection.
- No write-amplification for buffered minute aggs. We merge in-process
  before flushing so a 100-req minute on one key is one UPSERT, not 100.

Crash recovery
==============
If the process dies between `record()` and the next flush, those few
seconds of requests are LOST. This matches the trade-off documented for
the webhook worker. For an MVP that's fine; if we need durability later,
write-ahead to a small append-only file before the flush window.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("apin_v2.usage_recorder")


# Flush every N seconds. 2 s = small loss window, low DB load.
_FLUSH_INTERVAL_SECONDS = 2.0

# Percentile rollup pass cadence — runs after each flush every Nth tick.
# 30 ticks * 2 s = 60 s. So percentiles trail by up to 60 s.
_ROLLUP_EVERY_N_TICKS = 30

# Window the rollup pass looks back through (in minutes) when computing
# percentiles. Wider than 1 to catch minutes that straddle the boundary.
_ROLLUP_WINDOW_MINUTES = 5

# Hard cap on the buffer size — if the flusher dies and the buffer keeps
# growing, drop oldest rows past this cap. This is purely a safety valve.
_MAX_BUFFERED_REQUESTS = 50000

# Truncate path/UA strings to avoid blowing up rows on malicious clients.
_MAX_PATH_LEN = 512
_MAX_UA_LEN = 512
_MAX_IP_LEN = 64


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _minute_bucket(dt: datetime) -> str:
    """Round a datetime down to its UTC minute and return as ISO-Z text.
    Matches the format `api_key_usage_minute.minute_ts` is queried with
    elsewhere ("YYYY-MM-DD HH:MM:00")."""
    return dt.strftime("%Y-%m-%d %H:%M:00")


def _truncate(s: Optional[str], limit: int) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) > limit:
        return s[:limit]
    return s


@dataclass
class _RequestRow:
    """One row destined for api_key_request_log + the index info needed to
    update the per-minute aggregate, the api_keys row, and re-rank the
    auth lifecycle (last_used_*)."""
    key_id: str
    user_id: int
    timestamp: str           # ISO with microsecond precision
    minute_ts: str           # bucket
    method: str
    path: str
    status_code: int
    latency_ms: Optional[int]
    ip: Optional[str]
    ua: Optional[str]
    bytes_in: Optional[int]
    bytes_out: Optional[int]
    error_code: Optional[str]
    via: Optional[str]       # bearer | x_api_key | session
    # 9.N.8 · Pass 2 payload-capture fields. All nullable; old/missing data
    # surfaces as "not recorded" in the drawer.
    headers_in_json: Optional[str] = None
    headers_out_json: Optional[str] = None
    body_in_preview: Optional[str] = None
    body_out_preview: Optional[str] = None
    body_in_ctype: Optional[str] = None
    body_out_ctype: Optional[str] = None
    body_in_truncated: Optional[int] = None    # 1 if full body was larger than preview cap
    body_out_truncated: Optional[int] = None
    stage_timings_json: Optional[str] = None   # {"auth":3,"validate":1,"handler":38,...}


@dataclass
class _MinuteAgg:
    """In-memory accumulator that gets UPSERTed into api_key_usage_minute."""
    requests: int = 0
    errors: int = 0
    rate_limited: int = 0
    quota_blocked: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    latency_sum_ms: int = 0
    latency_count: int = 0


@dataclass
class _KeyDelta:
    """Per-key accumulated lifecycle deltas. Folded into one UPDATE per
    key at flush time."""
    requests_delta: int = 0
    errors_delta: int = 0
    last_used_at: Optional[str] = None
    last_used_ip: Optional[str] = None
    last_used_ua: Optional[str] = None


class _UsageBuffer:
    """Thread-safe in-memory buffer. Owned by the flusher; mutated by
    `record()` calls from request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_buckets()

    def _reset_buckets(self) -> None:
        self._rows: List[_RequestRow] = []
        # (key_id, minute_ts) -> _MinuteAgg
        self._minute_aggs: Dict[Tuple[str, str], _MinuteAgg] = defaultdict(_MinuteAgg)
        # key_id -> _KeyDelta
        self._key_deltas: Dict[str, _KeyDelta] = defaultdict(_KeyDelta)

    def record(self, row: _RequestRow,
               *, rate_limited: bool = False,
               quota_blocked: bool = False) -> None:
        """Append one request to the buffer. Fast path — O(1) work,
        protected by a single short-held lock."""
        with self._lock:
            # Safety valve — if the flusher is dead, don't keep allocating
            # forever. Drop oldest rows past the cap (FIFO).
            if len(self._rows) >= _MAX_BUFFERED_REQUESTS:
                drop = len(self._rows) - _MAX_BUFFERED_REQUESTS + 1
                del self._rows[:drop]

            self._rows.append(row)

            agg = self._minute_aggs[(row.key_id, row.minute_ts)]
            agg.requests += 1
            if row.status_code >= 500:
                agg.errors += 1
            if rate_limited:
                agg.rate_limited += 1
            if quota_blocked:
                agg.quota_blocked += 1
            if row.bytes_in:
                agg.bytes_in += int(row.bytes_in)
            if row.bytes_out:
                agg.bytes_out += int(row.bytes_out)
            if row.latency_ms is not None:
                agg.latency_sum_ms += int(row.latency_ms)
                agg.latency_count += 1

            delta = self._key_deltas[row.key_id]
            delta.requests_delta += 1
            if row.status_code >= 500:
                delta.errors_delta += 1
            # last_used_at — keep the most recent across the batch.
            if delta.last_used_at is None or row.timestamp > delta.last_used_at:
                delta.last_used_at = row.timestamp
                delta.last_used_ip = row.ip
                delta.last_used_ua = row.ua

    def drain(self) -> Tuple[List[_RequestRow],
                              Dict[Tuple[str, str], _MinuteAgg],
                              Dict[str, _KeyDelta]]:
        """Atomically swap out the buffers and return them. Caller writes
        to DB outside the lock."""
        with self._lock:
            rows = self._rows
            mins = self._minute_aggs
            keys = self._key_deltas
            self._reset_buckets()
        return rows, mins, keys

    def size(self) -> int:
        with self._lock:
            return len(self._rows)


# Module-level singleton — the decorator pushes into this; the flusher
# pulls out of it. Always-available; if the flusher worker never starts
# we silently accumulate up to _MAX_BUFFERED_REQUESTS rows then drop the
# oldest. (The flusher is started by apin_server._ensure_heartbeat alongside
# the webhook worker.)
_BUFFER = _UsageBuffer()


def record_request(*, key_id: str, user_id: int,
                    method: str, path: str, status_code: int,
                    latency_ms: Optional[int],
                    ip: Optional[str], ua: Optional[str],
                    bytes_in: Optional[int] = None,
                    bytes_out: Optional[int] = None,
                    error_code: Optional[str] = None,
                    via: Optional[str] = None,
                    rate_limited: bool = False,
                    quota_blocked: bool = False,
                    # 9.N.8 · Pass 2 payload capture (all optional)
                    headers_in: Optional[Dict[str, str]] = None,
                    headers_out: Optional[Dict[str, str]] = None,
                    body_in_preview: Optional[str] = None,
                    body_out_preview: Optional[str] = None,
                    body_in_ctype: Optional[str] = None,
                    body_out_ctype: Optional[str] = None,
                    body_in_truncated: bool = False,
                    body_out_truncated: bool = False,
                    stage_timings: Optional[Dict[str, int]] = None) -> None:
    """Public entry point. Called from `_record_usage` in the decorator.

    Never raises. If anything goes wrong, log + swallow (this runs inside
    the decorator's `finally:` block and a raise here would corrupt the
    response).
    """
    try:
        now = _utcnow()
        # ISO with microseconds, no `+00:00` suffix — matches `datetime('now')`
        # SQLite emits. Storing UTC always.
        ts_iso = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        minute_ts = _minute_bucket(now)

        # 9.N.8 · Serialize payload helpers to JSON. Always bounded.
        headers_in_json = json.dumps(headers_in, separators=(",", ":"))[:8192] if headers_in else None
        headers_out_json = json.dumps(headers_out, separators=(",", ":"))[:8192] if headers_out else None
        stage_timings_json = json.dumps(stage_timings, separators=(",", ":"))[:512] if stage_timings else None

        row = _RequestRow(
            key_id=key_id,
            user_id=int(user_id),
            timestamp=ts_iso,
            minute_ts=minute_ts,
            method=(method or "GET")[:8].upper(),
            path=_truncate(path, _MAX_PATH_LEN) or "/",
            status_code=int(status_code),
            latency_ms=(int(latency_ms) if latency_ms is not None else None),
            ip=_truncate(ip, _MAX_IP_LEN),
            ua=_truncate(ua, _MAX_UA_LEN),
            bytes_in=(int(bytes_in) if bytes_in is not None else None),
            bytes_out=(int(bytes_out) if bytes_out is not None else None),
            error_code=_truncate(error_code, 64),
            via=_truncate(via, 32),
            headers_in_json=headers_in_json,
            headers_out_json=headers_out_json,
            # 9.N.8e · 96 KB cap on body previews — large enough to fit a
            # multipart JSON with embedded base64 image thumbnails
            # (~64 KB raw → ~88 KB b64 + part metadata). Plain text previews
            # are typically < 4 KB; multipart-with-images is the outlier.
            body_in_preview=_truncate(body_in_preview, 96 * 1024),
            body_out_preview=_truncate(body_out_preview, 96 * 1024),
            body_in_ctype=_truncate(body_in_ctype, 128),
            body_out_ctype=_truncate(body_out_ctype, 128),
            body_in_truncated=(1 if body_in_truncated else 0) if body_in_preview is not None else None,
            body_out_truncated=(1 if body_out_truncated else 0) if body_out_preview is not None else None,
            stage_timings_json=stage_timings_json,
        )
        _BUFFER.record(row,
                        rate_limited=rate_limited,
                        quota_blocked=quota_blocked)
        # 9.N.7 · After the request is buffered for DB write, fan out to any
        # SSE subscribers watching this user's stream. The publish is sync +
        # non-blocking (uses put_nowait + drop-oldest backpressure) so this
        # cannot slow down the request path.
        _publish_stream_event(row, rate_limited, quota_blocked)
    except Exception as e:
        log.debug("usage_recorder.record_request failed (non-fatal): %s", e)


def buffer_size() -> int:
    """For /api/_diag or health: how many rows are sitting in memory."""
    return _BUFFER.size()


# ─── 9.N.7 · Per-user live-stream pub-sub for SSE ──────────────────────────
# Each authenticated user can subscribe to /api/account/usage/stream and
# receive every recorded request for any of their keys as it happens.
#
# Architecture (mirrors the Stage-2 _LiveNowBus in apin_server.py):
#   · One bus instance per process; thread-safe queue per subscriber.
#   · subscribe(user_id) returns a queue; the SSE endpoint reads from it.
#   · publish(user_id, event) fans out to all of that user's subscribers.
#   · Slow consumers get drop-oldest backpressure (no head-of-line blocking).
#   · Bounded queue size — one bad client can't balloon process memory.
#
# Why per-user (not global): privacy. User A must NOT see user B's traffic.
# We key the subscriber set by user_id and only fan to matching subscribers.

import asyncio as _asyncio
import threading as _threading


class _AccountStreamBus:
    """Per-user pub-sub for usage events. Subscribers receive every new
    request_log row published for keys belonging to their user_id.
    """
    def __init__(self):
        # Map: user_id (int) -> set of asyncio.Queue
        self._subs: dict[int, set] = {}
        self._lock = _threading.Lock()

    def subscribe(self, user_id: int, queue_max: int = 64) -> "_asyncio.Queue":
        q: _asyncio.Queue = _asyncio.Queue(maxsize=queue_max)
        with self._lock:
            self._subs.setdefault(int(user_id), set()).add(q)
        return q

    def unsubscribe(self, user_id: int, q: "_asyncio.Queue") -> None:
        with self._lock:
            s = self._subs.get(int(user_id))
            if s is not None:
                s.discard(q)
                if not s:
                    self._subs.pop(int(user_id), None)

    def publish(self, user_id: int, event: dict) -> None:
        """Synchronous publish — callable from any thread / from the
        usage_recorder buffer flush. Best-effort: never raises."""
        with self._lock:
            subs = list(self._subs.get(int(user_id), ()))
        for q in subs:
            try:
                q.put_nowait(event)
            except _asyncio.QueueFull:
                # Drop-oldest backpressure: discard the head, retry
                try: q.get_nowait()
                except Exception: pass
                try: q.put_nowait(event)
                except Exception: pass
            except Exception:
                pass

    def subscriber_count(self, user_id: Optional[int] = None) -> int:
        with self._lock:
            if user_id is None:
                return sum(len(s) for s in self._subs.values())
            return len(self._subs.get(int(user_id), ()))


# Single process-wide bus. Imported by the SSE endpoint + by record_request.
_STREAM_BUS = _AccountStreamBus()


def get_stream_bus() -> _AccountStreamBus:
    """Public accessor for the SSE bus. Used by apin_server.py to mount
    the /api/account/usage/stream endpoint."""
    return _STREAM_BUS


def _publish_stream_event(row: "_RequestRow",
                            rate_limited: bool, quota_blocked: bool) -> None:
    """Fire-and-forget SSE publish. Called after a successful record_request
    so dashboard subscribers see the row in near-real-time. Failures are
    silently dropped — recording always succeeds even if streaming is dead.
    """
    try:
        event = {
            "type": "request",
            # Slim payload — UI only needs these fields for the row entry.
            # Full row is fetchable via /api/account/usage/request/{id}.
            "timestamp": row.timestamp,
            "key_id": row.key_id,
            "method": row.method,
            "path": row.path,
            "status_code": row.status_code,
            "latency_ms": row.latency_ms,
            "bytes_in": row.bytes_in,
            "bytes_out": row.bytes_out,
            "error_code": row.error_code,
            "ip": row.ip,
            "ua": row.ua,
            "via": row.via,
            "rate_limited": bool(rate_limited),
            "quota_blocked": bool(quota_blocked),
        }
        _STREAM_BUS.publish(int(row.user_id), event)
    except Exception as e:
        log.debug("stream publish (non-fatal): %s", e)


# ─── Flusher ────────────────────────────────────────────────────────────────

def _flush_once(conn) -> Dict[str, int]:
    """One drain → bulk INSERT → UPSERT → UPDATE pass. Returns counts for
    logging. Pass the already-open `auth_db.get_conn()` connection."""
    rows, mins, keys = _BUFFER.drain()
    if not rows and not mins and not keys:
        return {"rows": 0, "mins": 0, "keys": 0}

    # ── 1. bulk INSERT into api_key_request_log ──────────────────────
    if rows:
        # Validate that the referenced key still exists in api_keys — a
        # hard-deleted key would FAIL the FK and abort the whole batch.
        # Cheap check: build the set of seen key_ids and prune missing.
        try:
            seen = {r.key_id for r in rows}
            existing_set = set()
            if seen:
                placeholders = ",".join(["?"] * len(seen))
                existing = conn.execute(
                    f"SELECT public_id FROM api_keys "
                    f"WHERE public_id IN ({placeholders})",
                    list(seen)
                ).fetchall()
                existing_set = {dict(r)["public_id"] for r in existing}
            kept = [r for r in rows if r.key_id in existing_set]
        except Exception:
            kept = rows  # fall back to writing all; DB layer will validate

        if kept:
            # 9.N.8 · INSERT now writes the payload + stage timings columns too.
            # All are nullable; rows without payload (sandbox/docs/internal) just
            # store NULL and the drawer falls back to "not recorded".
            payload = [(r.key_id, r.timestamp, r.method, r.path,
                         r.status_code, r.latency_ms, r.ip, r.ua,
                         r.bytes_in, r.bytes_out, r.error_code, r.via,
                         r.headers_in_json, r.headers_out_json,
                         r.body_in_preview, r.body_out_preview,
                         r.body_in_ctype, r.body_out_ctype,
                         r.body_in_truncated, r.body_out_truncated,
                         r.stage_timings_json)
                        for r in kept]
            conn.executemany(
                "INSERT INTO api_key_request_log "
                "(key_id, timestamp, method, path, status_code, latency_ms, "
                " ip, ua, bytes_in, bytes_out, error_code, via, "
                " headers_in_json, headers_out_json, "
                " body_in_preview, body_out_preview, "
                " body_in_ctype, body_out_ctype, "
                " body_in_truncated, body_out_truncated, "
                " stage_timings_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                payload
            )

    # ── 2. UPSERT into api_key_usage_minute ──────────────────────────
    if mins:
        for (key_id, minute_ts), agg in mins.items():
            # SQLite UPSERT — accumulate counters on conflict.
            conn.execute(
                "INSERT INTO api_key_usage_minute "
                "(key_id, minute_ts, requests, errors, rate_limited, "
                " quota_blocked, bytes_in, bytes_out, latency_sum_ms, "
                " latency_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(key_id, minute_ts) DO UPDATE SET "
                "  requests       = requests       + excluded.requests, "
                "  errors         = errors         + excluded.errors, "
                "  rate_limited   = rate_limited   + excluded.rate_limited, "
                "  quota_blocked  = quota_blocked  + excluded.quota_blocked, "
                "  bytes_in       = bytes_in       + excluded.bytes_in, "
                "  bytes_out      = bytes_out      + excluded.bytes_out, "
                "  latency_sum_ms = latency_sum_ms + excluded.latency_sum_ms, "
                "  latency_count  = latency_count  + excluded.latency_count",
                (key_id, minute_ts, agg.requests, agg.errors, agg.rate_limited,
                 agg.quota_blocked, agg.bytes_in, agg.bytes_out,
                 agg.latency_sum_ms, agg.latency_count)
            )

    # ── 3. UPDATE api_keys lifecycle counters ────────────────────────
    if keys:
        for key_id, delta in keys.items():
            conn.execute(
                "UPDATE api_keys SET "
                "  request_count = COALESCE(request_count, 0) + ?, "
                "  error_count   = COALESCE(error_count, 0)   + ?, "
                "  last_used_at  = ?, "
                "  last_used_ip  = ?, "
                "  last_used_ua  = ? "
                "WHERE public_id = ?",
                (delta.requests_delta, delta.errors_delta,
                 delta.last_used_at, delta.last_used_ip,
                 delta.last_used_ua, key_id)
            )

    return {"rows": len(rows), "mins": len(mins), "keys": len(keys)}


def _rollup_percentiles(conn) -> int:
    """Compute exact p50/p95/p99 from `api_key_request_log` rows over the
    last `_ROLLUP_WINDOW_MINUTES` minutes and write them back to
    `api_key_usage_minute`.

    Returns number of (key_id, minute_ts) buckets updated.

    Algorithm:
      1. SELECT key_id, minute(timestamp), latency_ms FROM request_log
         WHERE timestamp >= now - 5 minutes AND latency_ms IS NOT NULL
      2. Group by (key_id, minute) in Python (SQLite doesn't have
         PERCENTILE_CONT cross-backend; libSQL same).
      3. For each group, compute p50/p95/p99 via simple nearest-rank.
      4. UPDATE api_key_usage_minute SET latency_pXX_ms = ?
         WHERE key_id = ? AND minute_ts = ?
    """
    try:
        rows = conn.execute(
            "SELECT key_id, "
            "       strftime('%Y-%m-%d %H:%M:00', timestamp) AS minute_ts, "
            "       latency_ms "
            "FROM api_key_request_log "
            "WHERE timestamp >= datetime('now', '-' || ? || ' minutes') "
            "  AND latency_ms IS NOT NULL",
            (_ROLLUP_WINDOW_MINUTES,)
        ).fetchall()
    except Exception as e:
        log.debug("rollup: select failed: %s", e)
        return 0

    if not rows:
        return 0

    # Group: (key_id, minute_ts) -> sorted list of latencies
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in rows:
        d = dict(r)
        grouped[(d["key_id"], d["minute_ts"])].append(int(d["latency_ms"]))

    updates = 0
    for (key_id, minute_ts), lats in grouped.items():
        lats.sort()
        n = len(lats)
        if n == 0:
            continue
        # Nearest-rank percentile. For tiny n (<10) p99 == p95 == max,
        # which is fine — the dashboard renders them all.
        def _pct(p: float) -> float:
            # 0-indexed nearest-rank: ceil(p * n / 100) - 1
            import math
            idx = max(0, min(n - 1, math.ceil(p * n / 100.0) - 1))
            return float(lats[idx])

        p50 = _pct(50)
        p95 = _pct(95)
        p99 = _pct(99)
        try:
            conn.execute(
                "UPDATE api_key_usage_minute SET "
                "  latency_p50_ms = ?, "
                "  latency_p95_ms = ?, "
                "  latency_p99_ms = ? "
                "WHERE key_id = ? AND minute_ts = ?",
                (p50, p95, p99, key_id, minute_ts)
            )
            updates += 1
        except Exception as e:
            log.debug("rollup: update failed for %s/%s: %s",
                      key_id, minute_ts, e)
            continue

    return updates


class UsageFlusher:
    """Background coroutine. Owned by the app; started once at boot in
    `apin_server._ensure_heartbeat`."""

    def __init__(self) -> None:
        self._stop_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._ticks = 0
        self._flushed_rows = 0
        self._flushed_mins = 0
        self._flushed_keys = 0
        self._rollup_updates = 0
        self._last_flush_iso: Optional[str] = None
        self._last_error: Optional[str] = None

    @classmethod
    def start_in_background(cls) -> "UsageFlusher":
        f = cls()
        f._stop_event = asyncio.Event()
        f._task = asyncio.create_task(f._run())
        log.info("Usage flusher started (interval=%.1fs, rollup every %ds)",
                  _FLUSH_INTERVAL_SECONDS,
                  _FLUSH_INTERVAL_SECONDS * _ROLLUP_EVERY_N_TICKS)
        return f

    async def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    def stats(self) -> Dict[str, Any]:
        return {
            "ticks": self._ticks,
            "flushed_rows": self._flushed_rows,
            "flushed_mins": self._flushed_mins,
            "flushed_keys": self._flushed_keys,
            "rollup_updates": self._rollup_updates,
            "buffer_size": buffer_size(),
            "last_flush_iso": self._last_flush_iso,
            "last_error": self._last_error,
        }

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            self._ticks += 1
            try:
                # The flush itself runs in a thread so we don't block the
                # event loop on the SQLite write lock.
                counts = await asyncio.get_event_loop().run_in_executor(
                    None, _safe_flush_pass, self._ticks)
                self._flushed_rows += counts.get("rows", 0)
                self._flushed_mins += counts.get("mins", 0)
                self._flushed_keys += counts.get("keys", 0)
                self._rollup_updates += counts.get("rollup", 0)
                if counts.get("rows") or counts.get("mins"):
                    self._last_flush_iso = _utcnow().isoformat()
            except Exception as e:
                self._last_error = repr(e)
                log.warning("usage flusher tick failed: %s", e)

            try:
                await asyncio.wait_for(self._stop_event.wait(),
                                        timeout=_FLUSH_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
        log.info("Usage flusher stopped (ticks=%d)", self._ticks)


def _safe_flush_pass(tick: int) -> Dict[str, int]:
    """Single flush pass, exception-safe, runs inside an executor thread.
    Acquires its own auth_db connection so it doesn't share with the loop."""
    # Lazy import to avoid circular dependency at module load.
    from scripts.apin_v2 import auth_db as adb

    counts: Dict[str, int] = {"rows": 0, "mins": 0, "keys": 0, "rollup": 0}
    try:
        with adb.get_conn() as conn:
            counts.update(_flush_once(conn))
            # Every _ROLLUP_EVERY_N_TICKS-th tick, also compute percentiles
            # over the recently-flushed window. Run AFTER the flush so the
            # current tick's rows are picked up.
            if tick % _ROLLUP_EVERY_N_TICKS == 0:
                counts["rollup"] = _rollup_percentiles(conn)
    except Exception as e:
        log.warning("_safe_flush_pass: %s", e)
    return counts


# ─── Manual flush for tests + graceful shutdown ────────────────────────────

def flush_now() -> Dict[str, int]:
    """Synchronous flush. Used by tests and shutdown hooks."""
    return _safe_flush_pass(tick=_ROLLUP_EVERY_N_TICKS)  # also runs rollup
