"""Admin email-OTP MFA + trusted devices (R1) — the admin security spine.

After password login an admin must pass a 6-digit code emailed to them, UNLESS
this device was 'remembered' (a trusted-device cookie within 7 days). The
session then carries a time-boxed elevation (`sessions.admin_verified_at`) that
the admin gate re-checks on every request.

Threat model / guarantees:
  • Codes: 6 digits, HASHED (never stored raw), single-use, 10-min TTL,
    attempt-capped (5) then invalidated, bound to {user, session}.
  • Requesting a code is rate-limited (5 / 15 min / user).
  • Hash comparison is constant-time (hmac.compare_digest).
  • Elevation lasts 12h then re-OTP. Trusted devices last 7 days, revocable.
  • Every request / success / failure / device-trust is audited.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from scripts.apin_v2 import auth_db

OTP_TTL_SECONDS      = 600           # 10 minutes
OTP_MAX_ATTEMPTS     = 5
OTP_REQUEST_WINDOW_S = 15 * 60       # rate-limit window for code requests
OTP_REQUEST_MAX      = 5             # codes per window per user
ELEVATION_WINDOW_S   = 12 * 3600     # admin elevation lifetime
TRUSTED_DEVICE_TTL_S = 7 * 24 * 3600 # 'remember this device' = 7 days
DEVICE_COOKIE_NAME   = "apin_admin_device"


def _salt() -> bytes:
    return (os.environ.get("IP_HASH_SALT") or "apin-fallback-salt").encode()


def _hash(value: str) -> str:
    """Salted SHA-256. Codes are low-entropy, so the real protections are TTL +
    attempt-cap + single-use; hashing just avoids storing the plaintext."""
    return hmac.new(_salt(), (value or "").encode(), hashlib.sha256).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def gen_code() -> str:
    """6 uniformly-random digits, leading zeros allowed."""
    return "".join(secrets.choice("0123456789") for _ in range(6))


# ── OTP ────────────────────────────────────────────────────────────────────
def request_rate_limited(user_id: int) -> bool:
    """True if the user has requested too many codes in the window."""
    cutoff = _iso(_now() - timedelta(seconds=OTP_REQUEST_WINDOW_S))
    with auth_db.get_conn() as c:
        n = c.execute(
            "SELECT COUNT(*) n FROM admin_otp WHERE user_id = ? AND created_at >= ?",
            (int(user_id), cutoff)).fetchone()
        return int(dict(n).get("n") or 0) >= OTP_REQUEST_MAX


def create_admin_otp(*, user_id: int, session_id, ip, ua) -> Tuple[str, str]:
    """Mint a code. Returns (raw_code, expires_at_iso). Supersedes the user's
    prior unconsumed codes so only one is ever live."""
    code = gen_code()
    now = _now()
    exp = _iso(now + timedelta(seconds=OTP_TTL_SECONDS))
    with auth_db._write_lock, auth_db.get_conn() as c:
        c.execute("UPDATE admin_otp SET consumed_at = ? "
                  "WHERE user_id = ? AND consumed_at IS NULL",
                  (_iso(now), int(user_id)))
        c.execute(
            "INSERT INTO admin_otp(user_id, session_id, code_hash, purpose, "
            "created_at, expires_at, ip, user_agent) VALUES (?,?,?,?,?,?,?,?)",
            (int(user_id), session_id, _hash(code), "admin_login",
             _iso(now), exp, ip, (ua or "")[:255]))
    auth_db.audit("admin.otp_requested", user_id=int(user_id), ip_addr=ip)
    return code, exp


def verify_admin_otp(*, user_id: int, session_id, code: str, ip) -> Tuple[bool, Optional[str]]:
    """Verify a code against the user's latest live OTP. On success, elevates
    the session. Returns (ok, reason). reason ∈ no_code/expired/too_many_attempts/
    bad_code. Constant-time compare."""
    now_iso = _iso(_now())
    # IMPORTANT: never call auth_db.audit() while holding _write_lock — audit()
    # re-acquires the same non-reentrant threading.Lock, which would deadlock the
    # event-loop thread. We do all DB writes under the lock, decide the outcome
    # (+ a deferred audit event), release the lock, THEN audit.
    result: Tuple[bool, Optional[str]] = (False, "no_code")
    audit_event: Optional[str] = None
    audit_detail: Optional[dict] = None
    with auth_db._write_lock, auth_db.get_conn() as c:
        row = c.execute(
            "SELECT id, code_hash, expires_at, consumed_at, attempts FROM admin_otp "
            "WHERE user_id = ? AND consumed_at IS NULL ORDER BY id DESC LIMIT 1",
            (int(user_id),)).fetchone()
        if not row:
            result = (False, "no_code")
        else:
            d = dict(row)
            if str(d["expires_at"]) <= now_iso:
                c.execute("UPDATE admin_otp SET consumed_at = ? WHERE id = ?", (now_iso, d["id"]))
                result = (False, "expired")
            elif int(d["attempts"]) >= OTP_MAX_ATTEMPTS:
                c.execute("UPDATE admin_otp SET consumed_at = ? WHERE id = ?", (now_iso, d["id"]))
                result = (False, "too_many_attempts")
            elif not hmac.compare_digest(str(d["code_hash"]), _hash(code)):
                new_attempts = int(d["attempts"]) + 1
                c.execute("UPDATE admin_otp SET attempts = ? WHERE id = ?", (new_attempts, d["id"]))
                if new_attempts >= OTP_MAX_ATTEMPTS:
                    c.execute("UPDATE admin_otp SET consumed_at = ? WHERE id = ?", (now_iso, d["id"]))
                audit_event = "admin.otp_failed"
                audit_detail = {"remaining": max(0, OTP_MAX_ATTEMPTS - new_attempts)}
                result = (False, "bad_code")
            else:
                # success — consume + elevate
                c.execute("UPDATE admin_otp SET consumed_at = ? WHERE id = ?", (now_iso, d["id"]))
                c.execute("UPDATE sessions SET admin_verified_at = ? WHERE id = ?",
                          (now_iso, session_id))
                audit_event = "admin.otp_verified"
                result = (True, None)
    if audit_event:
        auth_db.audit(audit_event, user_id=int(user_id), ip_addr=ip, detail=audit_detail)
    return result


def mark_session_admin_verified(session_id) -> None:
    with auth_db._write_lock, auth_db.get_conn() as c:
        c.execute("UPDATE sessions SET admin_verified_at = ? WHERE id = ?",
                  (_iso(_now()), session_id))


def elevation_ok(admin_verified_at: Optional[str]) -> bool:
    """Is the session's elevation present AND within the window?"""
    if not admin_verified_at:
        return False
    try:
        t = datetime.fromisoformat(str(admin_verified_at))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except Exception:
        return False
    return (_now() - t).total_seconds() < ELEVATION_WINDOW_S


# ── Trusted devices (remember for 7 days) ──────────────────────────────────
def create_trusted_device(*, user_id: int, ip, ua) -> str:
    """Store a trusted-device row, return the raw token (goes in the cookie)."""
    raw = secrets.token_urlsafe(32)
    now = _now()
    with auth_db._write_lock, auth_db.get_conn() as c:
        c.execute(
            "INSERT INTO admin_trusted_devices(user_id, token_hash, created_at, "
            "expires_at, ip, user_agent) VALUES (?,?,?,?,?,?)",
            (int(user_id), _hash(raw), _iso(now),
             _iso(now + timedelta(seconds=TRUSTED_DEVICE_TTL_S)), ip, (ua or "")[:255]))
    auth_db.audit("admin.device_trusted", user_id=int(user_id), ip_addr=ip)
    return raw


def check_trusted_device(*, user_id: int, raw_token: Optional[str]) -> bool:
    """True if this raw token is a valid, non-expired, non-revoked trusted
    device for the user. Touches last_used_at."""
    if not raw_token:
        return False
    now_iso = _iso(_now())
    with auth_db._write_lock, auth_db.get_conn() as c:
        row = c.execute(
            "SELECT id FROM admin_trusted_devices WHERE user_id = ? AND token_hash = ? "
            "AND revoked_at IS NULL AND expires_at > ? LIMIT 1",
            (int(user_id), _hash(raw_token), now_iso)).fetchone()
        if not row:
            return False
        c.execute("UPDATE admin_trusted_devices SET last_used_at = ? WHERE id = ?",
                  (now_iso, dict(row)["id"]))
    return True
