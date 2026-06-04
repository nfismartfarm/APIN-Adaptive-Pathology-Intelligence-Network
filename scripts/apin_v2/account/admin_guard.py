"""Admin access control for the APIN console — Phase A foundation.

This module is the SINGLE SOURCE OF TRUTH for "is this user an admin?" and
for the route guard that protects every admin surface. It is deliberately
small, pure, and heavily tested, because every other admin route trusts it.

Design tenets
-------------
1. NON-DESTRUCTIVE. Introduces zero schema changes. Admin status is derived
   at request time from two independent, OR-combined signals:

       (a) Bootstrap allowlist — emails listed in the ``APIN_ADMIN_EMAILS``
           environment variable (comma / semicolon / whitespace separated,
           case-insensitive). This seeds the very first admin(s) without
           touching the database, and is the recovery path if the DB role
           is ever lost.

       (b) Role column — ``users.role == 'admin'`` (the pre-existing,
           previously-unused free-text column on ``users``). Set via the
           in-console "promote" action in a later phase (sudo-gated +
           audited). The default for every existing/new user is
           ``'collector'``, so NO current account becomes an admin by
           accident when this code ships.

2. NEVER TRUST THE CLIENT. Admin status is always re-derived server-side
   from the authenticated user row resolved from the session cookie. No
   header, query param, body field, or cookie can assert admin-ness.

3. HIDE THE SURFACE. An authenticated non-admin hitting an admin *API* gets
   ``not_found`` (HTTP 404) — the admin surface does not acknowledge its own
   existence to people who shouldn't see it. (There is intentionally no
   generic "forbidden" canonical error code; 404 is both available and the
   more secure choice.) An *unauthenticated* caller gets the normal 401 from
   ``get_session_user`` so the client knows to sign in. The admin *page*
   route handles non-admins with a redirect (see apin_server.py), not a 404
   body, for friendlier UX — disclosure is equivalent either way.

This module depends only on ``api_envelope.ApiError`` and the canonical
``_session_helpers.get_session_user``; it is imported by ``routes_admin`` and
(for the pure ``is_admin`` check) by ``auth_routes`` to stamp the login
response. No circular-import risk: neither ``auth_db`` nor ``api_envelope``
imports this module.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Request

from scripts.apin_v2.api_envelope import ApiError

# NOTE: ``get_session_user`` is imported LAZILY inside ``require_admin`` (not at
# module top). The pure helpers below (``is_admin``, ``_parse_allowlist``,
# ``admin_allowlist``) carry no auth/session/DB dependency, so this keeps the
# module trivially importable and unit-testable in isolation — importing it must
# not pull in the entire ``_session_helpers → auth_db`` chain.


# Environment variable that seeds the bootstrap admin allowlist.
ADMIN_EMAILS_ENV_VAR = "APIN_ADMIN_EMAILS"


def _parse_allowlist(raw: Optional[str]) -> frozenset[str]:
    """Parse the raw env value into a normalised set of admin emails.

    Accepts commas, semicolons, and any whitespace (incl. newlines) as
    separators so the value is forgiving to format. Each entry is stripped
    and lower-cased. Blank entries are dropped. Returns a frozenset for
    cheap, immutable membership tests.
    """
    if not raw:
        return frozenset()
    # Normalise every plausible separator to a comma, then split.
    normalised = (
        raw.replace(";", ",")
           .replace("\n", ",")
           .replace("\r", ",")
           .replace("\t", ",")
           .replace(" ", ",")
    )
    out: set[str] = set()
    for chunk in normalised.split(","):
        email = chunk.strip().lower()
        if email:
            out.add(email)
    return frozenset(out)


def admin_allowlist() -> frozenset[str]:
    """The current bootstrap admin allowlist, read LIVE from the environment.

    Read on every call (not cached at import) so that changing the env var
    — and, crucially, unit tests that monk-patch ``os.environ`` — take effect
    without a process restart. The parse is trivial and called only on
    explicit admin checks, so the cost is negligible.
    """
    return _parse_allowlist(os.environ.get(ADMIN_EMAILS_ENV_VAR))


def is_admin(user: Optional[dict]) -> bool:
    """Return True iff ``user`` is an administrator.

    Pure function over a user dict (the shape returned by
    ``auth_db.get_user_by_id`` / ``get_session_user``). Safe to call with
    ``None`` (returns False). The two signals are OR-combined:

        role == 'admin'  OR  email ∈ APIN_ADMIN_EMAILS

    Comparison is case-insensitive on both email and role, and tolerant of
    surrounding whitespace, so a stray space in the env var or a mixed-case
    email can never silently lock out a legitimate admin.
    """
    if not user:
        return False
    role = (user.get("role") or "").strip().lower()
    if role == "admin":
        return True
    email = (user.get("email") or "").strip().lower()
    if email and email in admin_allowlist():
        return True
    return False


def require_admin(request: Request) -> dict:
    """Resolve the request to an authenticated ADMIN user, or raise.

    Returns the full user dict on success. Failure modes:

      * No / invalid / expired session  → ``ApiError('invalid_or_missing_token',
        401)`` (raised by ``get_session_user``). Tells the client to sign in.

      * Authenticated, but NOT an admin → ``ApiError('not_found', 404)``. The
        admin surface stays invisible to ordinary users — we do not confirm
        that an admin API exists.

    Every admin API handler should call this first, before any work.
    """
    # Lazy import: only needed at call time, and only inside the running server
    # where the auth/session/DB stack is already loaded.
    from scripts.apin_v2.account._session_helpers import get_session_user
    user = get_session_user(request)  # 401 invalid_or_missing_token if no session
    if not is_admin(user):
        # Deliberately indistinguishable from "this route does not exist".
        raise ApiError("not_found", "Not found.")
    return user


def require_admin_verified(request: Request) -> dict:
    """Like ``require_admin`` PLUS the email-OTP elevation (R1).

    An admin who has NOT passed the email code (or whose elevation expired, and
    who isn't on a trusted device) is rejected with ``invalid_or_missing_token``
    (401) — which the client treats as "re-authenticate". This is what stops a
    logged-in admin from cur‑ing the admin APIs directly without ever passing
    the second factor. A valid trusted-device cookie auto-elevates the session.

    Use this on EVERY admin data/mutation route and the admin page. (``whoami``
    deliberately uses the lighter ``require_admin`` so the login page can detect
    admin-ness BEFORE elevation.)
    """
    from scripts.apin_v2.account._session_helpers import (
        get_session_with_id, SESSION_COOKIE_NAME,
    )
    from scripts.apin_v2.account import admin_auth
    from scripts.apin_v2 import auth_db

    user, session_id = get_session_with_id(request)   # 401 if no session
    if not is_admin(user):
        raise ApiError("not_found", "Not found.")

    raw = request.cookies.get(SESSION_COOKIE_NAME)
    sess = auth_db.lookup_session_by_token(raw) if raw else None
    verified = bool(sess and admin_auth.elevation_ok(sess.get("admin_verified_at")))
    if not verified:
        dev = request.cookies.get(admin_auth.DEVICE_COOKIE_NAME)
        if dev and admin_auth.check_trusted_device(user_id=user["id"], raw_token=dev):
            admin_auth.mark_session_admin_verified(session_id)
            verified = True
    if not verified:
        raise ApiError(
            "invalid_or_missing_token",
            "Admin verification required.",
            hint="Complete email verification to access the admin console.")
    return user
