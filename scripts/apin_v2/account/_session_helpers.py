"""Shared session + CSRF helpers for the Console route modules.

Resolves WI-P4-DEDUP-SESS (PDA-P3.3-3.4-R1 F6): `routes_keys.py` and
`routes_sudo.py` previously each had a private copy of `_get_session_user`-
shaped logic and a byte-identical `_require_csrf` (after the Phase 4 CSRF
upgrade landed in both copies). This module is the canonical single
source.

Design notes:

- The functions are PUBLIC-named (no leading underscore) because they're
  imported across module boundaries. Within a single route file the
  callers may still alias them as `_get_session_user` etc. for cosmetic
  continuity with the older code, but the canonical names live here.

- `get_session_user` is a thin wrapper over `get_session_with_id` — both
  paths share the same resolution logic; the difference is just whether
  the caller needs the session_id alongside the user. The wrapper avoids
  duplicating the SessionMiddleware-state-first + cookie-fallback chain.

- `require_csrf` does constant-time comparison via `secrets.compare_digest`
  against `sessions.csrf_token` (seeded at `create_session` time by the
  Phase 4 fix, rotated on `sudo_started` per spec §7.6 PDA-F44).

- `SESSION_COOKIE_NAME` is the public canonical constant. The two route
  files re-import it (or alias to `_SESSION_COOKIE_NAME`) for backward
  compatibility — the actual string lives here.

This module deliberately depends only on `api_envelope.ApiError` and
`auth_db.*` (lookup_session_by_token, get_user_by_id). No circular import
risk because neither caller (`routes_keys`, `routes_sudo`) is imported
by `auth_db` or `api_envelope`.
"""
from __future__ import annotations

import secrets as _secrets
from typing import Optional

from fastapi import Request

from scripts.apin_v2.api_envelope import ApiError
from scripts.apin_v2 import auth_db


SESSION_COOKIE_NAME = "apin_v2_session"


def get_session_with_id(request: Request) -> tuple[dict, str]:
    """Resolve (user_dict, session_id) from the apin_v2_session cookie.

    Prefers `request.state.session` populated by SessionMiddleware
    (slot 6); falls back to a direct DB lookup if state is missing
    (e.g. during unit tests that don't mount the full stack).

    Raises ApiError("invalid_or_missing_token", 401) if the session is
    missing / expired / revoked.
    """
    # First try scope.state.session (SessionMiddleware-populated)
    state = getattr(request, "state", None)
    sess = getattr(state, "session", None) if state else None
    if sess is not None:
        session_id = getattr(sess, "session_id", None)
        user_id = getattr(sess, "user_id", None)
        if session_id and user_id:
            user = auth_db.get_user_by_id(int(user_id))
            if user:
                return user, session_id
            # User row deleted but session still cached in middleware
            # state — fall through to cookie path which will fail cleanly.

    # Fallback: direct DB lookup from cookie
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        raise ApiError(
            "invalid_or_missing_token",
            "Session cookie missing. Please sign in.",
            hint="POST /api/auth/login to obtain a session cookie.",
        )
    row = auth_db.lookup_session_by_token(raw)
    if row is None:
        raise ApiError(
            "invalid_or_missing_token",
            "Session expired or revoked. Please sign in again.",
        )
    user = auth_db.get_user_by_id(int(row["user_id"]))
    if user is None:
        raise ApiError(
            "invalid_or_missing_token",
            "Session expired or revoked. Please sign in again.",
        )
    return user, row["session_id"]


def get_session_user(request: Request) -> dict:
    """Resolve the session cookie to a user dict, or raise 401.

    Thin wrapper over `get_session_with_id` for callers that don't need
    the session_id. Same resolution chain, same error semantics.
    """
    user, _session_id = get_session_with_id(request)
    return user


def require_csrf(request: Request) -> None:
    """Verify X-Console-Csrf header matches the session's csrf_token.

    Spec §7.6 PDA-F44 + WI-P4-CSRF. Constant-time comparison via
    `secrets.compare_digest`. The expected value comes from
    `sessions.csrf_token` (seeded at `create_session` time, rotated on
    `sudo_started`), resolved via either the SessionMiddleware-populated
    scope.state OR a direct cookie lookup.

    Raises ApiError("invalid_or_missing_token", 401) on any failure
    (missing header, missing session, mismatch).
    """
    incoming = (request.headers.get("x-console-csrf") or "").strip()
    if not incoming:
        raise ApiError(
            "invalid_or_missing_token",
            "X-Console-Csrf header required.",
            hint="Read the CSRF token from <meta name=\"csrf-token\"> on "
                 "Console pages.",
        )

    # Resolve session — prefer SessionMiddleware-populated scope.state
    state = getattr(request, "state", None)
    sess = getattr(state, "session", None) if state else None
    expected: Optional[str] = None
    if sess is not None:
        expected = getattr(sess, "csrf_token", None)
        if expected is None and isinstance(sess, dict):
            expected = sess.get("csrf_token")
    if expected is None:
        raw = request.cookies.get(SESSION_COOKIE_NAME)
        if raw:
            row = auth_db.lookup_session_by_token(raw)
            if row:
                expected = row.get("csrf_token")

    if not expected:
        # Session row exists but csrf_token is unset — should never happen
        # after WI-P4-CSRF seeds at create_session time, but defensive:
        # treat as auth failure to force re-auth.
        raise ApiError(
            "invalid_or_missing_token",
            "Session has no CSRF token. Please sign in again.",
        )

    if not _secrets.compare_digest(incoming, expected):
        raise ApiError(
            "invalid_or_missing_token",
            "CSRF token mismatch.",
            hint="The token may have rotated (after sudo step-up) — re-read "
                 "from the response body or <meta> tag.",
        )
