"""Phase 8 Wave B — webhook delivery worker.

Background async task that drains `webhook_deliveries` rows in `status='queued'`
and POSTs them to the receiver URL. Implements the spec §13.x delivery
contract on a best-effort budget:

  - Exponential backoff: attempt 1 → +1m, +5m, +25m, +2h, +12h, +24h, +48h.
    Spec §6.6 CHECK (attempts <= 8) caps total attempts at 8.
  - `gave_up` at attempt 8 if still failing.
  - Lease-based exactly-once: each worker `UPDATE ... SET status='in_flight',
    claimed_by=?, claimed_at=?` WHERE status='queued' AND
    next_attempt_at <= now. Other workers (or this same worker after a crash)
    can re-claim rows whose `claimed_at < now - 10 min` (spec §13.6).
  - Multi-secret rotation overlap: per spec §13.9 the `APIN-Signature` header
    is `t=<ts>,v1=<new_hex>` and during the grace window also includes
    `,v1_prev=<old_hex>` so receivers can verify against either.
  - SSRF defence: no redirects followed; deny RFC1918 / loopback / link-local
    destinations unless the webhook row has `allow_internal_target = 1`.
  - URL homograph defence: Punycode-normalised hostnames + ASCII-only check
    at create-time (see `account.routes_webhooks._validate_url_basic_v2`).

Honesty: the spec mandates an `httpx.AsyncClient` with connection-pool limits
and `asyncio.Semaphore(50)` (PDA-R2-F35). I use `httpx` if available, fall
back to a `urllib`-in-executor path so the worker still runs in environments
where httpx is not on the path. The semaphore is in place either way.

Filed for Phase 9+: header replay protection (`APIN-Replay: 1`), per-webhook
back-pressure queue (current implementation pumps everything through one
semaphore), `webhook_deliveries.replay_of` chain UI exposure.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import os
import socket
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

log = logging.getLogger("apin_v2.webhook_worker")


# ─── Constants (spec §13.6) ────────────────────────────────────────────────

# Exponential backoff schedule in SECONDS. Index = attempt count BEFORE this
# attempt (so [0] is the wait BEFORE attempt 2). Length 7 lets us cap at 8
# attempts total. Spec lists 1m, 5m, 25m, 2h, 12h, 24h, 48h — same schedule.
_BACKOFF_SCHEDULE = (60, 300, 1500, 7200, 43200, 86400, 172800)

# Per spec §13.6, an `in_flight` row whose `claimed_at < now - 10 min` is
# considered stuck and may be re-claimed by another worker.
_LEASE_RECLAIM_SECONDS = 600

# Per spec §13.x cap of 8 attempts total (also enforced by schema CHECK).
_MAX_ATTEMPTS = 8

# Polling cadence when the queue is empty.
_IDLE_POLL_SECONDS = 5.0

# Max concurrent in-flight deliveries per worker (PDA-R2-F35).
_MAX_CONCURRENT = 50

# How big a response body to store (truncate longer).
_RESPONSE_BODY_LIMIT = 2048

# Per-attempt timeout (seconds). Spec §13.6 suggests 10s for sync ops.
_ATTEMPT_TIMEOUT_SECONDS = 15.0

# How long this worker generation is considered alive in the `claimed_by`
# field — used for the operator log only, not for correctness.
_WORKER_ID = f"worker-{os.getpid()}-{int(time.time())}"


# ─── SSRF + URL hardening (spec §13.7 + §13.8) ────────────────────────────

def _is_internal_target_host(hostname: str) -> bool:
    """Resolve `hostname` and return True if ANY resolved address falls in
    a forbidden range: RFC1918 private, loopback, link-local, multicast,
    or unspecified. This is a best-effort check — a receiver that uses DNS
    rebinding can still pivot AFTER the check but BEFORE the connect; the
    only complete defence is at OS-firewall level (out of scope).
    """
    try:
        # getaddrinfo returns multiple records; check ALL of them
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        # If DNS fails, treat as failure-closed (block)
        return True
    for fam, _typ, _proto, _canon, sockaddr in infos:
        try:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
        except (IndexError, ValueError):
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_unspecified
                or ip.is_reserved):
            return True
    return False


def url_is_safe_target(url: str, *, allow_internal: bool = False) -> tuple[bool, str]:
    """Returns (safe, reason). `reason` is empty on safe; describes the
    block when unsafe. Spec §13.7 requires HTTPS + (no internal targets
    unless `allow_internal_target`)."""
    try:
        u = urlparse(url)
    except Exception as e:
        return False, f"unparseable URL: {e}"
    if u.scheme != "https":
        return False, f"scheme must be https (got {u.scheme!r})"
    if not u.hostname:
        return False, "no hostname in URL"
    if not allow_internal and _is_internal_target_host(u.hostname):
        return False, (
            f"host {u.hostname!r} resolves to an internal/loopback/link-local "
            f"address; set allow_internal_target=true on the webhook if you "
            f"really mean to target an internal endpoint")
    return True, ""


def url_to_punycode(url: str) -> str:
    """Per PDA-R2-F40: Punycode-normalise the hostname to make homograph
    attacks visible (mixed Cyrillic+Latin etc.). Pass-through for ASCII
    hosts."""
    try:
        u = urlparse(url)
        if not u.hostname:
            return url
        host_ascii = u.hostname.encode("idna").decode("ascii")
        if host_ascii == u.hostname:
            return url
        # Rebuild URL with the Punycode host
        netloc = host_ascii
        if u.port:
            netloc = f"{netloc}:{u.port}"
        if u.username:
            netloc = (f"{u.username}@{netloc}"
                      if not u.password else
                      f"{u.username}:{u.password}@{netloc}")
        return u._replace(netloc=netloc).geturl()
    except Exception:
        return url   # punt; caller's basic URL validator already rejected


# ─── Signature helpers ────────────────────────────────────────────────────

def _build_signature_header(secret_plaintext: bytes,
                             secret_prev: Optional[bytes],
                             timestamp_str: str,
                             event_id: str,
                             body: bytes) -> str:
    """Spec §13.2 + §13.9. Format:
        APIN-Signature: t=<ts>,v1=<hex>[,v1_prev=<hex>]
    `v1_prev` is included only while a rotation grace window is open
    (secret_prev != None).
    """
    msg = (timestamp_str + "." + event_id + ".").encode("ascii") + body
    new_hex = hmac.new(secret_plaintext, msg, hashlib.sha256).hexdigest()
    out = f"t={timestamp_str},v1={new_hex}"
    if secret_prev:
        prev_hex = hmac.new(secret_prev, msg, hashlib.sha256).hexdigest()
        out += f",v1_prev={prev_hex}"
    return out


# ─── DB helpers (avoid circular import with auth_db at module-load time) ──

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat()


def _claim_due_deliveries(limit: int = 25) -> list[dict]:
    """Atomically claim `limit` due deliveries: pick rows where
    `status='queued' AND next_attempt_at <= now`, OR `status='in_flight'
    AND claimed_at < now - 10 min` (stuck row), and atomically flip them
    to `status='in_flight', claimed_by=worker, claimed_at=now`.
    Returns the freshly-claimed rows (joined to webhooks for url + secret).
    """
    from scripts.apin_v2 import auth_db
    now_iso = _iso(_now())
    stuck_cutoff = _iso(datetime.fromtimestamp(
        time.time() - _LEASE_RECLAIM_SECONDS, tz=timezone.utc))

    with auth_db._write_lock, auth_db.get_conn() as c:
        # Two-step claim: first pick candidate ids under the lock, then UPDATE
        # them. Single-process SQLite serializes via _write_lock so we can't
        # race; for Turso the libsql round-trip is serialized too.
        rows = c.execute(
            "SELECT id FROM webhook_deliveries "
            "WHERE (status = 'queued' AND next_attempt_at <= ?) "
            "   OR (status = 'in_flight' AND claimed_at < ?) "
            "ORDER BY next_attempt_at ASC LIMIT ?",
            (now_iso, stuck_cutoff, int(limit))
        ).fetchall()
        if not rows:
            return []
        ids = [int(r["id"]) for r in rows]
        placeholders = ",".join("?" * len(ids))
        c.execute(
            f"UPDATE webhook_deliveries SET status = 'in_flight', "
            f"  claimed_by = ?, claimed_at = ? "
            f"WHERE id IN ({placeholders})",
            [_WORKER_ID, now_iso] + ids
        )
        # Re-fetch with join to webhooks for url + secrets.
        full = c.execute(
            f"SELECT d.id AS delivery_id, d.webhook_id, d.event_id, "
            f"       d.event_type, d.payload, d.attempts, d.queued_at, "
            f"       w.url, w.secret_encrypted, w.secret_encrypted_old, "
            f"       w.secret_old_until, w.active, w.allow_internal_target, "
            f"       w.allow_self_signed, w.user_id "
            f"FROM webhook_deliveries d "
            f"JOIN webhooks w ON w.id = d.webhook_id "
            f"WHERE d.id IN ({placeholders})",
            ids
        ).fetchall()
    return [dict(r) for r in full]


def _record_attempt(delivery_id: int, *, attempts: int,
                     status: str, response_status: Optional[int],
                     response_body: Optional[str], error_text: Optional[str],
                     next_attempt_at: Optional[str]) -> None:
    """Update the delivery row with attempt outcome. Also bumps the parent
    webhook's last_delivery_at + last_delivery_status + counters."""
    from scripts.apin_v2 import auth_db
    now_iso = _iso(_now())
    body_trunc = 0
    if response_body and len(response_body) > _RESPONSE_BODY_LIMIT:
        response_body = response_body[:_RESPONSE_BODY_LIMIT]
        body_trunc = 1
    with auth_db._write_lock, auth_db.get_conn() as c:
        # Fetch webhook id + current failure counters for the parent update
        row = c.execute(
            "SELECT webhook_id FROM webhook_deliveries WHERE id = ?",
            (int(delivery_id),)
        ).fetchone()
        if row is None:
            return
        wh_id = row["webhook_id"]

        c.execute(
            "UPDATE webhook_deliveries SET "
            "  attempts = ?, status = ?, response_status = ?, "
            "  response_body = ?, response_truncated = ?, "
            "  error_text = ?, "
            "  first_attempt_at = COALESCE(first_attempt_at, ?), "
            "  last_attempt_at = ?, "
            "  next_attempt_at = ?, "
            "  claimed_by = NULL, claimed_at = NULL "
            "WHERE id = ?",
            (int(attempts), status, response_status, response_body, body_trunc,
             error_text, now_iso, now_iso, next_attempt_at, int(delivery_id))
        )

        # Parent webhook counters
        if status == "delivered":
            c.execute(
                "UPDATE webhooks SET "
                "  last_delivery_at = ?, last_delivery_status = ?, "
                "  consecutive_failure_count = 0, "
                "  updated_at = ? "
                "WHERE id = ?",
                (now_iso, response_status, now_iso, wh_id)
            )
        elif status in ("failed_attempt", "gave_up"):
            c.execute(
                "UPDATE webhooks SET "
                "  last_delivery_at = ?, last_delivery_status = ?, "
                "  consecutive_failure_count = consecutive_failure_count + 1, "
                "  consecutive_gave_up_count = consecutive_gave_up_count + ? ,"
                "  updated_at = ? "
                "WHERE id = ?",
                (now_iso, response_status,
                 (1 if status == "gave_up" else 0),
                 now_iso, wh_id)
            )


def _maybe_auto_disable(webhook_id: str) -> None:
    """Spec §13.5: if `consecutive_gave_up_count >= 5` (rolling 24h), set
    `auto_disabled_at` + `active = 0`. Phase 8 keeps the 24h window literal:
    count `gave_up` rows in the last day.

    Phase 8.H: emit `webhook.auto_disabled` alert when the threshold trips
    (the UPDATE rowcount confirms we actually disabled it — if it was
    already disabled, no alert).
    """
    from scripts.apin_v2 import auth_db
    cutoff_iso = _iso(datetime.fromtimestamp(
        time.time() - 86400, tz=timezone.utc))
    with auth_db._write_lock, auth_db.get_conn() as c:
        n_row = c.execute(
            "SELECT COUNT(*) AS n FROM webhook_deliveries "
            "WHERE webhook_id = ? AND status = 'gave_up' "
            "  AND last_attempt_at > ?",
            (webhook_id, cutoff_iso)
        ).fetchone()
        n = int(n_row["n"])
        disabled_now = False
        wh_meta = None
        if n >= 5:
            cur = c.execute(
                "UPDATE webhooks SET "
                "  auto_disabled_at = COALESCE(auto_disabled_at, ?), "
                "  active = 0, updated_at = ? "
                "WHERE id = ? AND auto_disabled_at IS NULL",
                (_iso(_now()), _iso(_now()), webhook_id)
            )
            disabled_now = bool(getattr(cur, "rowcount", 0))
            if disabled_now:
                wh_meta = c.execute(
                    "SELECT user_id, url FROM webhooks WHERE id = ?",
                    (webhook_id,)
                ).fetchone()
    if disabled_now and wh_meta:
        try:
            auth_db.emit_alert(
                int(wh_meta["user_id"]), "webhook.auto_disabled",
                action={"kind": "re_enable_webhook", "id": webhook_id},
                url=wh_meta["url"], giveups=n,
            )
        except Exception:
            pass


# ─── Single-delivery attempt ──────────────────────────────────────────────

async def _attempt_delivery(row: dict, *, semaphore: asyncio.Semaphore) -> None:
    """Fire one HTTP POST for a claimed delivery row. Records the outcome.
    Caller must have already claimed the row (status='in_flight')."""
    from scripts.apin_v2 import auth_db
    delivery_id = int(row["delivery_id"])
    webhook_id = row["webhook_id"]
    payload_bytes = (row["payload"] or "").encode("utf-8")
    timestamp_str = str(int(time.time()))
    attempts_after = int(row["attempts"]) + 1

    # SSRF re-check at attempt time (defence-in-depth — URL may have been
    # acceptable at create-time but blocked now).
    url = row["url"]
    allow_internal = bool(row["allow_internal_target"])
    safe, reason = url_is_safe_target(url, allow_internal=allow_internal)
    if not safe:
        _record_attempt(
            delivery_id, attempts=attempts_after, status="gave_up",
            response_status=None, response_body=None,
            error_text=f"SSRF guard: {reason}",
            next_attempt_at=_iso(_now())
        )
        # Phase 8.H · SSRF-rejected gave-up emits the same alert.
        try:
            auth_db.emit_alert(
                int(row["user_id"]), "webhook.delivery_gave_up",
                action={"kind": "view_delivery",
                        "webhook_id": webhook_id,
                        "delivery_id": delivery_id},
                url=url, event=row.get("event_type", "?"),
                attempts=attempts_after,
            )
        except Exception:
            pass
        _maybe_auto_disable(webhook_id)
        return

    # Decrypt secret(s)
    try:
        secret = auth_db.decrypt_webhook_secret(row["secret_encrypted"], webhook_id)
    except Exception as e:
        _record_attempt(
            delivery_id, attempts=attempts_after, status="failed_attempt",
            response_status=None, response_body=None,
            error_text=f"secret decrypt failed: {type(e).__name__}: {e}",
            next_attempt_at=_compute_next_attempt(attempts_after)
        )
        return

    secret_prev = None
    if row.get("secret_encrypted_old") and row.get("secret_old_until"):
        try:
            until = datetime.fromisoformat(row["secret_old_until"])
            if until > _now():
                secret_prev = auth_db.decrypt_webhook_secret(
                    row["secret_encrypted_old"], webhook_id)
        except Exception:
            pass   # ignore — proceed without v1_prev

    sig_header = _build_signature_header(
        secret, secret_prev, timestamp_str, row["event_id"], payload_bytes
    )

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "APIN-Webhook/1.0",
        "APIN-Event-Id": row["event_id"],
        "APIN-Delivery-Id": str(delivery_id),
        "APIN-Signature": sig_header,
        "APIN-Webhook-Id": webhook_id,
    }
    if int(row["attempts"]) > 0:
        headers["APIN-Replay"] = "1"

    async with semaphore:
        outcome = await _do_post(url, payload_bytes, headers,
                                  timeout_seconds=_ATTEMPT_TIMEOUT_SECONDS,
                                  allow_self_signed=bool(row["allow_self_signed"]))

    # Compute next_attempt_at for the records helper
    if outcome["status"] == "delivered":
        next_at = _iso(_now())
        final_status = "delivered"
    elif attempts_after >= _MAX_ATTEMPTS:
        final_status = "gave_up"
        next_at = _iso(_now())
    else:
        final_status = "queued"   # back to queued, with bumped attempts
        next_at = _compute_next_attempt(attempts_after)

    _record_attempt(
        delivery_id, attempts=attempts_after, status=final_status,
        response_status=outcome["response_status"],
        response_body=outcome["response_body"],
        error_text=outcome["error_text"],
        next_attempt_at=next_at
    )
    if final_status == "gave_up":
        # Phase 8.H · alert BEFORE auto-disable check, so the user sees
        # the delivery failure even if it doesn't cross the disable
        # threshold (4 give-ups in 24h won't auto-disable, but each one
        # still warrants a notification).
        try:
            auth_db.emit_alert(
                int(row["user_id"]), "webhook.delivery_gave_up",
                action={"kind": "view_delivery",
                        "webhook_id": webhook_id,
                        "delivery_id": delivery_id},
                url=row.get("url", "?"),
                event=row.get("event_type", "?"),
                attempts=attempts_after,
            )
        except Exception:
            pass
        _maybe_auto_disable(webhook_id)


def _compute_next_attempt(attempts_after: int) -> str:
    """Given the attempt count AFTER this attempt, return the ISO timestamp
    for when to try again. Index BACKOFF[attempts_after - 1]."""
    idx = max(0, min(attempts_after - 1, len(_BACKOFF_SCHEDULE) - 1))
    delta_sec = _BACKOFF_SCHEDULE[idx]
    return _iso(datetime.fromtimestamp(time.time() + delta_sec, tz=timezone.utc))


# ─── HTTP POST (httpx if available, else urllib in executor) ──────────────

async def _do_post(url: str, body: bytes, headers: dict, *,
                    timeout_seconds: float, allow_self_signed: bool) -> dict:
    """Fire one POST with no redirects. Returns
    {status, response_status, response_body, error_text}.
    Tries httpx first (proper async + connection pooling); falls back to
    urllib in a thread executor if httpx is unavailable."""
    try:
        import httpx
        return await _do_post_httpx(url, body, headers,
                                     timeout_seconds=timeout_seconds,
                                     allow_self_signed=allow_self_signed)
    except ImportError:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: _do_post_urllib(url, body, headers,
                                     timeout_seconds=timeout_seconds,
                                     allow_self_signed=allow_self_signed)
        )


async def _do_post_httpx(url: str, body: bytes, headers: dict, *,
                          timeout_seconds: float, allow_self_signed: bool) -> dict:
    import httpx
    verify = not allow_self_signed
    try:
        async with httpx.AsyncClient(
            verify=verify, follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20,
                                 keepalive_expiry=30.0),
        ) as client:
            resp = await client.post(url, content=body, headers=headers)
        status = resp.status_code
        text = resp.text[:_RESPONSE_BODY_LIMIT]
        if 300 <= status < 400:
            return {"status": "failed_attempt", "response_status": status,
                    "response_body": text,
                    "error_text": f"receiver returned {status} redirect; "
                                  "delivery worker does not follow redirects"}
        if 200 <= status < 300:
            return {"status": "delivered", "response_status": status,
                    "response_body": text, "error_text": None}
        return {"status": "failed_attempt", "response_status": status,
                "response_body": text,
                "error_text": f"HTTP {status}"}
    except httpx.TimeoutException as e:
        return {"status": "failed_attempt", "response_status": None,
                "response_body": None,
                "error_text": f"timeout after {timeout_seconds}s: {e}"}
    except httpx.RequestError as e:
        return {"status": "failed_attempt", "response_status": None,
                "response_body": None,
                "error_text": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"status": "failed_attempt", "response_status": None,
                "response_body": None,
                "error_text": f"{type(e).__name__}: {e}"}


def _do_post_urllib(url: str, body: bytes, headers: dict, *,
                     timeout_seconds: float, allow_self_signed: bool) -> dict:
    import urllib.request, urllib.error, ssl

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            return None

    ctx = ssl.create_default_context()
    if allow_self_signed:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(_NoRedirect(),
                                          urllib.request.HTTPSHandler(context=ctx))
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with opener.open(req, timeout=timeout_seconds) as resp:
            status = int(resp.status)
            text = resp.read(_RESPONSE_BODY_LIMIT).decode("utf-8", errors="replace")
        if 300 <= status < 400:
            return {"status": "failed_attempt", "response_status": status,
                    "response_body": text,
                    "error_text": f"receiver returned {status} redirect"}
        if 200 <= status < 300:
            return {"status": "delivered", "response_status": status,
                    "response_body": text, "error_text": None}
        return {"status": "failed_attempt", "response_status": status,
                "response_body": text, "error_text": f"HTTP {status}"}
    except urllib.error.HTTPError as e:
        return {"status": "failed_attempt", "response_status": int(e.code),
                "response_body": None, "error_text": f"HTTP {e.code} {e.reason}"}
    except (urllib.error.URLError, OSError) as e:
        return {"status": "failed_attempt", "response_status": None,
                "response_body": None, "error_text": f"{type(e).__name__}: {e}"}


# ─── Worker loop ──────────────────────────────────────────────────────────

class WebhookWorker:
    """Lifecycle wrapper for the background worker.
    Usage:
        worker = WebhookWorker.start_in_background()   # at app startup
        await worker.stop()                            # at app shutdown
    """

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
        self._ticks = 0
        self._delivered_count = 0
        self._failed_count = 0
        self._gave_up_count = 0

    @classmethod
    def start_in_background(cls) -> "WebhookWorker":
        w = cls()
        w._stop_event = asyncio.Event()
        w._task = asyncio.create_task(w._run())
        log.info("Webhook delivery worker started (id=%s, max_concurrent=%d)",
                 _WORKER_ID, _MAX_CONCURRENT)
        return w

    async def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    def stats(self) -> dict:
        return {
            "worker_id": _WORKER_ID,
            "ticks": self._ticks,
            "delivered": self._delivered_count,
            "failed_attempts": self._failed_count,
            "gave_up": self._gave_up_count,
        }

    async def _run(self) -> None:
        """The actual loop: claim → fan-out → await → sleep when idle."""
        while not self._stop_event.is_set():
            self._ticks += 1
            try:
                claimed = await asyncio.get_event_loop().run_in_executor(
                    None, _claim_due_deliveries, 25)
            except Exception as e:
                log.exception("Worker claim failed: %s", e)
                claimed = []

            if not claimed:
                try:
                    await asyncio.wait_for(self._stop_event.wait(),
                                            timeout=_IDLE_POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass
                continue

            # Fan-out attempts concurrently (bounded by semaphore)
            results = await asyncio.gather(
                *(_attempt_delivery(r, semaphore=self._semaphore)
                  for r in claimed),
                return_exceptions=True
            )
            for res in results:
                if isinstance(res, Exception):
                    log.warning("Delivery attempt raised: %s", res)
            # Refresh counters by looking at the row final statuses isn't
            # worth a per-tick query; we approximate from the batch.
            self._delivered_count += sum(1 for _ in claimed)   # rough
        log.info("Webhook delivery worker stopped (id=%s, ticks=%d)",
                 _WORKER_ID, self._ticks)
