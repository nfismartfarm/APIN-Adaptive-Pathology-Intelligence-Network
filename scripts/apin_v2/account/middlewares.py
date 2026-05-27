"""API Console middleware stack — TokenRedaction · TokenFormat · Sudo.

Spec contract: spec_v7.md §9 (lines 3103-3275)

Middleware slots (runtime execution order — outermost first):
    1 GZip                        (existing)
    2 CORS                        (existing)
    3 TokenRedactionMiddleware    (this module — REV-R2-I25)
    4 TokenFormatMiddleware       (this module — REV-I06 + PDA-R2-F06 + PDA-R2-F13)
    5 AuditRecentMiddleware       (existing)
    6 SessionMiddleware           (existing — populates request.state.session)
    7 SudoMiddleware              (this module — runs just before handler)

Registration in apin_server.py MUST be in REVERSE order (slot 7 FIRST, slot 1
LAST) because FastAPI/Starlette's `add_middleware()` inserts at index 0.
See spec §9.1 for the full ordering rationale.

ASGI pattern: each middleware here is an ASGI app (callable with
`scope, receive, send`), NOT a `BaseHTTPMiddleware` subclass. This lets us:
  1. Inspect headers before FastAPI parses the body (DoS defence for
     TokenFormatMiddleware — REV-I06).
  2. Wrap `send` events streaming-friendly for redaction (no body
     materialization for non-JSON responses).

Audit IDs touched here:
  - REV-I06   — early header-only auth check (TokenFormat)
  - REV-R2-I08 — TokenFormatMiddleware impl phase
  - REV-R2-I25 — TokenRedactionMiddleware impl phase
  - PDA-R2-F06 — sticky-Bearer + session cookie tolerance (TokenFormat)
  - PDA-R2-F13 — VALID_TOKEN_REGEX shared with tokens.py
  - PDA-F04    — console_only_route enforcement (TokenFormat)
  - PDA-F14    — sudo HttpOnly cookie (SudoMiddleware)
  - PDA-R2-F33 — sudo_tokens.used_count cap (Phase 4 enforcement; stubbed here)
  - REV-I14    — UA binding removed from sudo verification
  - REV-I24    — Session before SudoMiddleware (slot 6 < slot 7)
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import tokens as T

log = logging.getLogger("apin_v2.account.middlewares")


# ──────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────

def _scope_headers_dict(scope) -> dict[bytes, bytes]:
    """ASGI scope['headers'] is a list of (bytes, bytes) tuples (header
    names are pre-lowercased by the server). Convert to dict for cheap
    lookup. Multi-valued headers (cookie, set-cookie) keep ONLY the last
    value — acceptable here because our checks (cookie presence,
    Bearer token) only need a single value."""
    d: dict[bytes, bytes] = {}
    for k, v in scope.get("headers", []):
        d[k] = v
    return d


def _parse_cookie_header(cookie_header: bytes, name: str) -> Optional[str]:
    """Extract a single cookie value from the raw Cookie header.

    Avoids importing http.cookies (overkill for one cookie). Handles the
    common cases: `name=value`, `; ` separator, optional `=`-padding.
    Returns None on miss.
    """
    if not cookie_header:
        return None
    needle = (name + "=").encode("latin1")
    for part in cookie_header.split(b"; "):
        if part.startswith(needle):
            return part[len(needle):].decode("latin1")
    return None


def _collect_cookie_headers(scope) -> bytes:
    """WI-P4-PRE-1: join ALL Cookie header values into a single bytes.

    HTTP/2 + HTTP/1.1 both allow multiple `Cookie:` headers (one cookie
    per line, or grouped by the user agent). PDA-P3.2-R2 PRE-1 caught
    the previous `for k, v in scope.headers: if k == b"cookie":
    cookie_header = v; break` pattern that picked up only the FIRST
    Cookie header and missed cookies in subsequent ones.

    This helper joins all of them with `"; "` (same separator used by
    `_parse_cookie_header` internally) so the existing parse logic
    works without further changes.

    FX-P4-5 (PDA-P4-R1 F5): strip trailing whitespace + semicolons from
    each individual header value before joining. RFC 6265 allows clients
    to emit `Cookie: name=value;` (trailing separator), which would
    otherwise survive the join-then-split round-trip as a trailing `;`
    glued onto the next cookie's name, breaking lookups.
    """
    parts = []
    for k, v in scope.get("headers", []):
        if k == b"cookie" and v:
            # Strip trailing whitespace + trailing semicolons + trailing
            # whitespace before the trailing semicolon, in case the
            # header looks like `name=value;  ` or `name=value ;`.
            cleaned = v.rstrip(b" \t").rstrip(b";").rstrip(b" \t")
            if cleaned:
                parts.append(cleaned)
    if not parts:
        return b""
    if len(parts) == 1:
        return parts[0]
    return b"; ".join(parts)


async def _send_json_error(send, status: int, code: str, message: str,
                           *, extra_headers: list[tuple[bytes, bytes]] = None) -> None:
    """Write a minimal §3-conforming error envelope directly to the ASGI
    `send` channel. Used by middlewares that need to short-circuit BEFORE
    FastAPI's exception-handler chain runs.

    Body shape matches `api_envelope._failure()`'s 9-key envelope to keep
    downstream clients consistent.

    PDA-P3.2-R1-F04 fix: generate a unique request_id per call (was the
    literal sentinel "req_middleware" which broke correlation across
    concurrent failures), and emit `X-Request-Id` response header so the
    client can echo it in support requests.
    """
    # PDA-F04: per-call request_id. uuid4-hex-prefix is plenty unique
    # (122 bits of randomness; collision odds astronomically lower than
    # the rate at which we generate errors).
    import uuid as _uuid
    request_id = "req_" + _uuid.uuid4().hex[:16]

    body = json.dumps({
        "api_version": "1.0",
        "request_id": request_id,
        "endpoint": "(middleware)",
        "ok": False,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "processing_time_ms": 0,
        "status_code": status,
        "error": {
            "code": code,
            "message": message,
            "docs_url": "https://dxv-404-apin.hf.space/docs#errors",
        },
        "meta": {},
    }).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"x-api-version", b"1.0"),
        (b"cache-control", b"no-store"),
        # PDA-F04: emit request_id as a response header too (PDA-R5-I03).
        (b"x-request-id", request_id.encode("ascii")),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": headers,
    })
    await send({
        "type": "http.response.body",
        "body": body,
        "more_body": False,
    })


# ──────────────────────────────────────────────────────────────────────────
# Slot 4 — TokenFormatMiddleware  (REV-I06 + PDA-R2-F06 + PDA-R2-F13)
# ──────────────────────────────────────────────────────────────────────────

class TokenFormatMiddleware:
    """Reject malformed Bearer/X-API-Key headers EARLY — before FastAPI
    parses the request body. Prevents a 10 MB multipart upload from an
    unauthenticated client from DoS-ing the worker.

    Behaviour:
      1. Only inspects requests on `/api/*` paths (out-of-scope: static
         assets, HTML pages, dashboard routes).
      2. If a Bearer token or X-API-Key header is PRESENT but doesn't
         match `tokens.VALID_TOKEN_REGEX`, respond 401
         `invalid_or_missing_token` immediately.
      3. For `/api/account/*` paths: reject API-key auth with 403
         `console_only_route` ONLY IF there is no `apin_v2_session`
         cookie. (PDA-R2-F06 — users with a session cookie + stale
         Bearer header from prior tools still authenticate via the
         session.)

    What this middleware does NOT do (responsibility of @require_scope):
      - DB lookup (key resolution)
      - Status checks (active / rotating / expired / etc.)
      - Scope enforcement
      - Rate limit / quota
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if not path.startswith("/api/"):
            return await self.app(scope, receive, send)

        headers = _scope_headers_dict(scope)
        # Header values arrive as bytes — decode with latin1 (lossless for
        # ASCII; we don't accept non-ASCII tokens anyway, but the decode
        # itself can't fail on arbitrary bytes).
        auth = headers.get(b"authorization", b"").decode("latin1")
        xapi = headers.get(b"x-api-key", b"").decode("latin1")

        candidate = ""
        if auth.lower().startswith("bearer "):
            candidate = auth[7:].strip()
        elif xapi:
            candidate = xapi.strip()

        # PDA-P2-R1-F03 fix: on console paths, a session cookie is the
        # authoritative auth signal. A user with a valid session cookie AND
        # a sticky / malformed Bearer header from prior tools should still
        # authenticate via the session. The previous order rejected the
        # malformed Bearer FIRST, which breaks PDA-R2-F06.
        #
        # New order on `/api/account/*`:
        #   if session cookie present → pass through (session middleware
        #     will handle auth; Bearer header is ignored)
        #   else if Bearer/X-API-Key present → 403 console_only_route
        #     (PDA-F04: API-key auth on console paths is forbidden)
        #
        # On non-console `/api/*` paths the original logic stands: malformed
        # token → 401 immediately.
        on_console = path.startswith("/api/account/")

        if on_console:
            if _has_session_cookie(scope):
                # Session cookie is authoritative — let session middleware
                # decide auth. We do not validate Bearer format here.
                return await self.app(scope, receive, send)
            if candidate:
                # API-key auth on console path with no session cookie →
                # PDA-F04 console_only_route. (Whether or not the candidate
                # is well-formed — auth-method enforcement, not format.)
                return await _send_json_error(
                    send, 403, "console_only_route",
                    "This route accepts session-cookie auth only.",
                    extra_headers=[(b"x-apin-hint",
                                    b"use the /account UI; do not send Bearer or X-API-Key")],
                )
            # No Bearer + no session → let downstream auth (e.g. the
            # session middleware or @require_scope) handle it.
            return await self.app(scope, receive, send)

        # Non-console /api/* path: reject malformed token early (REV-I06).
        if candidate and not T.VALID_TOKEN_REGEX.match(candidate):
            return await _send_json_error(
                send, 401, "invalid_or_missing_token",
                "Authentication token is missing or invalid.",
            )

        return await self.app(scope, receive, send)


def _has_session_cookie(scope) -> bool:
    """True if the request carries an `apin_v2_session=...` cookie.

    WI-P4-PRE-1: use `_collect_cookie_headers` to handle multi-Cookie-
    header requests correctly. The previous first-Cookie-wins pattern
    would miss the session cookie if a client sent it in a non-first
    Cookie header (legal per RFC 6265).
    """
    cookie = _collect_cookie_headers(scope)
    return b"apin_v2_session=" in cookie


# ──────────────────────────────────────────────────────────────────────────
# Slot 3 — TokenRedactionMiddleware  (REV-R2-I25)
# ──────────────────────────────────────────────────────────────────────────

# Match-anywhere version of VALID_TOKEN_REGEX. The redaction middleware
# scans response bodies for token-shaped substrings (no leading/trailing
# anchors). This MUST stay in sync with `tokens.VALID_TOKEN_REGEX` —
# any future grammar change updates both.
_REDACT_TOKEN_RE = re.compile(
    rb"apin_(?:live|test)_[0-9A-Za-z]{43}|apin_[0-9a-f]{32}"
)
_REDACT_WHSEC_RE = re.compile(rb"whsec_[0-9A-Za-z]{32}")
_REDACT_REPLACEMENT = b"apin_<redacted>"
_REDACT_WHSEC_REPLACEMENT = b"whsec_<redacted>"


# Content types where redaction is safe to apply. Binary types (images,
# octet-stream, etc.) are passed through untouched — they can't contain
# textual tokens, and our regex on raw bytes might match by chance.
_REDACT_OK_MIMES = (
    b"application/json",
    b"text/plain",
    b"text/html",
    b"text/event-stream",       # SSE — redact streaming token leaks
    b"application/xml",
    b"text/xml",
)


def _redact_body_chunk(chunk: bytes) -> bytes:
    """Apply both token + webhook-secret redactions to a body chunk."""
    chunk = _REDACT_TOKEN_RE.sub(_REDACT_REPLACEMENT, chunk)
    chunk = _REDACT_WHSEC_RE.sub(_REDACT_WHSEC_REPLACEMENT, chunk)
    return chunk


class TokenRedactionMiddleware:
    """Strip token-shaped substrings from response bodies before they
    reach the client OR any logging middleware that runs OUTSIDE us
    (slots 1-2).

    Defence-in-depth: tokens should never appear in responses to begin
    with (the §4.4 one-time view ceremony deliberately does NOT cache
    the plaintext, etc.). But operator error happens — a misconfigured
    debug endpoint might dump request.state to the response. This
    middleware catches that.

    Wraps the `send` channel: every `http.response.body` event is
    inspected. Non-textual content types pass through untouched.

    NOTE: redaction happens on the OUTBOUND path. We do NOT redact
    request bodies — those carry plaintext tokens by design (Bearer
    headers). Inbound request redaction belongs in the logging layer.
    """

    # Opt-out header (PDA-P2-R1-F01 fix). Route handlers that legitimately
    # emit one-time plaintext (e.g. POST /api/account/keys + .../rotate)
    # set this header on the response; the middleware:
    #   1. Detects the header
    #   2. Skips redaction for that response
    #   3. Strips the header before forwarding to the client (so the
    #      operator-side signal doesn't leak as a public-API hint)
    OPT_OUT_HEADER = b"x-apin-no-redact"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # State that survives across multiple body events.
        # WI-P4-PRE-2 (PDA-P3.2-R2 PRE-2): `tail` carries the last N bytes
        # of the previous chunk so a token straddling the chunk boundary
        # (e.g. 20 chars in chunk N, 23 chars in chunk N+1 — totalling
        # the 43-char base62 body of a live token) is caught by the regex
        # on the concatenated (tail + current) buffer. N = 64 is larger
        # than any APIN token length (the longest is "apin_live_" + 43 =
        # 53 chars, well under 64), so the regex window cannot miss a
        # token solely because of a chunk split.
        state = {"redact": False, "tail": b""}
        _TAIL_KEEP = 64

        async def wrapped_send(message):
            mtype = message.get("type", "")
            if mtype == "http.response.start":
                # Decide once, based on content-type, whether to redact.
                ct = b""
                opt_out = False
                stripped_headers: list[tuple[bytes, bytes]] = []
                for k, v in message.get("headers", []):
                    if k == b"content-type":
                        ct = v.split(b";", 1)[0].strip().lower()
                        stripped_headers.append((k, v))
                    elif k == self.OPT_OUT_HEADER:
                        # PDA-P2-R1-F01: route declared one-time-view content.
                        # Skip redaction AND strip the marker header so the
                        # client never sees the internal signal.
                        opt_out = True
                    else:
                        stripped_headers.append((k, v))
                state["redact"] = (
                    (not opt_out) and any(ct == m for m in _REDACT_OK_MIMES)
                )
                if opt_out:
                    log.info(
                        "TokenRedactionMiddleware: opt-out header present "
                        "for path=%r — skipping redaction (one-time-view "
                        "ceremony)", scope.get("path", "?"),
                    )
                    # Rebuild message with header stripped
                    message = {**message, "headers": stripped_headers}
                elif state["redact"]:
                    # PDA-P3.2-R1-F05 fix: if we plan to redact this
                    # response, strip Content-Length from the headers so
                    # transport falls back to chunked encoding. If the
                    # handler set Content-Length=N and redaction shortens
                    # the body, the previous behaviour was "let uvicorn
                    # surface a transport-level error" — but in practice
                    # uvicorn silently truncates, masking leaks. Chunked
                    # encoding sidesteps the mismatch entirely.
                    stripped_no_cl = [
                        (k, v) for (k, v) in stripped_headers
                        if k != b"content-length"
                    ]
                    if len(stripped_no_cl) != len(stripped_headers):
                        log.debug(
                            "TokenRedactionMiddleware: stripped Content-Length "
                            "on redactable response for path=%r (will use "
                            "chunked encoding)", scope.get("path", "?"),
                        )
                    message = {**message, "headers": stripped_no_cl}
                else:
                    message = {**message, "headers": stripped_headers}
                return await send(message)

            if mtype == "http.response.body" and state["redact"]:
                body = message.get("body", b"")
                more_body = message.get("more_body", False)
                # WI-P4-PRE-2: streaming redaction with tail-hold-back.
                # Algorithm:
                #   combined = previous-held-tail + this-chunk
                #   if more chunks coming: hold back the last N bytes
                #       (might be the START of a token that completes
                #       in the next chunk). Emit only what's safe.
                #   else: emit everything (no more chunks to wait for).
                #   Redact the emit-portion against the regex.
                combined = state["tail"] + body
                if more_body:
                    if len(combined) > _TAIL_KEEP:
                        emit_now = combined[:-_TAIL_KEEP]
                        state["tail"] = combined[-_TAIL_KEEP:]
                    else:
                        # Combined is small enough that the entire buffer
                        # might be the start of a token — withhold all.
                        emit_now = b""
                        state["tail"] = combined
                else:
                    # Last chunk — emit everything, no need to hold tail.
                    emit_now = combined
                    state["tail"] = b""

                if emit_now:
                    redacted = _redact_body_chunk(emit_now)
                    if redacted != emit_now:
                        # Tokens leaked! Log a WARNING (no body content
                        # because that would defeat redaction's purpose).
                        log.warning(
                            "TokenRedactionMiddleware redacted %d byte(s) of "
                            "token-shaped content from response (path=%r). "
                            "This indicates a handler leaked a token — find "
                            "and fix the source.",
                            len(emit_now) - len(redacted) + len(_REDACT_REPLACEMENT),
                            scope.get("path", "?"),
                        )
                    message = {**message, "body": redacted}
                else:
                    # Nothing to emit this round — but the original
                    # message may still have more_body=True signal that
                    # downstream needs. Send an empty-body keepalive.
                    message = {**message, "body": b""}
                # Content-Length was stripped above (PDA-F05); transport
                # uses chunked encoding so body-length mismatches are
                # impossible by construction.
            return await send(message)

        return await self.app(scope, receive, wrapped_send)


# ──────────────────────────────────────────────────────────────────────────
# Slot 7 — SudoMiddleware  (§7.6 + PDA-F14 + REV-I14)
# ──────────────────────────────────────────────────────────────────────────

SUDO_COOKIE_NAME = "apin_sudo"

# Methods that require sudo confirmation. Read-only (GET, HEAD, OPTIONS)
# is always exempt — see §9.2.
_SUDO_REQUIRED_METHODS = frozenset({"POST", "PATCH", "DELETE", "PUT"})

# Path-prefix exempts (§9.2). The sudo endpoints themselves can't require
# sudo, or you'd have a chicken-and-egg problem.
_SUDO_EXEMPT_PATHS = (
    "/api/account/sudo",          # POST/GET — includes /sudo/revoke
    "/api/account/alerts/stream", # SSE — read-only
    "/api/account/docs/try",      # ergonomics — see §9.2
    # Phase 8.H · alert inbox-state operations are NOT security mutations.
    # Marking notifications read / dismissed / snoozed is the user's own
    # inbox bookkeeping; gating it behind sudo silently kills the toast
    # interaction (toast's mark-read fetch 403'd, badge never refreshed,
    # dashboard stayed stale). The four routes covered:
    #   PATCH  /api/account/alerts/{id}/read
    #   PATCH  /api/account/alerts/{id}/restore
    #   POST   /api/account/alerts/{id}/snooze
    #   DELETE /api/account/alerts/{id}        (soft-dismiss)
    # AND the new prefs routes:
    #   PATCH  /api/account/alerts/prefs       (notification preferences)
    "/api/account/alerts",        # NOTE: matches /alerts AND /alerts/...
)


def _is_sudo_exempt(path: str, method: str) -> bool:
    """Return True if this request bypasses sudo enforcement.

    Spec §9.2: read-only methods exempt unconditionally. POST/PATCH/DELETE
    on path prefixes in `_SUDO_EXEMPT_PATHS` are also exempt.
    """
    if method not in _SUDO_REQUIRED_METHODS:
        return True
    for prefix in _SUDO_EXEMPT_PATHS:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _verify_sudo_cookie(cookie_value: str,
                       session_id: Optional[str],
                       client_ip: Optional[str]) -> Optional[dict]:
    """Look up + validate a sudo cookie value. Returns the sudo_tokens row
    dict on success, None on any verification failure.

    Spec §7.6 verification chain:
      1. SHA-256 the cookie value -> token_hash
      2. SELECT FROM sudo_tokens WHERE token_hash=?
      3. Check session_id matches (binding)
      4. Check bound_ip matches client_ip
      5. Check expires_at > now AND revoked_at IS NULL

    UA binding removed per REV-I14.

    PDA-R2-F33 used_count cap is enforced at INCREMENT time (in the
    consume_sudo_use helper, not here). This middleware only verifies
    the cookie is currently usable; the increment is the handler's job
    in Phase 4+.

    For Phase 2.3 the function tolerates `session_id is None` (means
    "session middleware didn't populate request.state.session yet") —
    returns None which the middleware treats as "no valid sudo." This
    keeps Phase 2.3 functional without requiring the full session layer
    to be wired up yet.
    """
    if not isinstance(cookie_value, str) or not cookie_value:
        return None
    try:
        token_hash = hashlib.sha256(cookie_value.encode("ascii")).hexdigest()
    except (UnicodeEncodeError, AttributeError):
        return None

    try:
        with auth_db.get_conn() as c:
            row = c.execute(
                "SELECT id, user_id, session_id, bound_ip, expires_at, "
                "revoked_at, used_count "
                "FROM sudo_tokens WHERE token_hash = ?",
                (token_hash,)
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None

    # Tolerate _ShimRow vs sqlite3.Row vs tuple
    def _g(r, k):
        try:
            return r[k]
        except Exception:
            return None
    rev = _g(row, "revoked_at")
    if rev is not None:
        return None
    exp = _g(row, "expires_at")
    if not exp:
        return None
    try:
        if exp.endswith("Z"):
            exp = exp[:-1] + "+00:00"
        exp_dt = datetime.fromisoformat(exp)
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) >= exp_dt:
            return None
    except Exception:
        return None

    # PDA-P2-R1-F02 fix: Session binding is REQUIRED. The previous code
    # said "if session_id is not None and …" — meaning when session_id
    # was None (e.g. session middleware hadn't yet populated state), the
    # check was SKIPPED, fail-open. Now: missing session_id means we
    # cannot verify binding → reject the cookie. Sudo MUST be tied to
    # the session that minted it.
    if session_id is None:
        return None
    # PDA-P3.3-LATENT-1 fix: stringify both sides. sudo_tokens.session_id
    # is TEXT (so SQLite stores "236") but sessions.id is INTEGER (so the
    # value passed in is int 236). Defensive normalization here means even
    # if a future caller forgets to stringify, verification still works.
    if str(_g(row, "session_id")) != str(session_id):
        return None
    # IP binding (PDA-P2-R1-F02 fix: same logic — None client_ip
    # means we can't verify; reject. The exception is keys minted
    # with NULL bound_ip in the DB, which means "any IP allowed"
    # — that's an explicit opt-in at sudo-create time, not a
    # missing-data fallback.)
    if client_ip is None:
        return None
    bound = _g(row, "bound_ip")
    # Treat empty-string bound_ip the same as NULL (any-IP allowed).
    if bound not in (None, "") and bound != client_ip:
        return None

    return {
        "id":           _g(row, "id"),
        "user_id":      _g(row, "user_id"),
        "session_id":   _g(row, "session_id"),
        "bound_ip":     _g(row, "bound_ip"),
        "expires_at":   _g(row, "expires_at"),
        "used_count":   _g(row, "used_count"),
    }


# Cookie name passed to `_parse_cookie_header` (which accepts str and
# encodes internally to latin-1). Variable name dropped the `_BYTES`
# suffix per PDA-P3.2-R2 P3 finding — it was misleading (the value is
# a str, the helper encodes).
_SESSION_COOKIE_NAME = "apin_v2_session"


class SessionMiddleware:
    """Parse the `apin_v2_session` cookie and populate `scope.state.session`.

    Spec §9.1 slot 6. Runs AFTER TokenFormatMiddleware (slot 4) and
    BEFORE SudoMiddleware (slot 7) in the ASGI stack. SudoMiddleware
    reads `scope.state.session.session_id` to bind sudo to the
    originating session (`_verify_sudo_cookie` fail-CLOSED on None).

    Added in Phase 3.2 R1 fix bundle to resolve PDA-P3.2-R1-F01: the
    previous deployment never populated `scope.state.session`, so
    SudoMiddleware always 403'd even with a valid sudo cookie.

    Behaviour:
      - Only fires on `/api/account/*` (where SudoMiddleware needs the
        session info). Other paths short-circuit to avoid the DB query
        tax — they don't depend on `scope.state.session`.
      - Reads `apin_v2_session` cookie via Cookie header.
      - Calls `auth_db.lookup_session_by_token` (session-row query, no
        user JOIN — lighter than `get_session_user`).
      - On hit: populates `scope.state.session` with a SimpleNamespace
        exposing `session_id`, `user_id`, `csrf_token`.
      - On miss / no cookie / expired: leaves `scope.state.session = None`.
        Does NOT reject — handlers + SudoMiddleware decide what None means.
      - DB lookup failures are logged but non-fatal (None populated).

    Why SimpleNamespace and not dict: SudoMiddleware's read side at
    `middlewares.py:574-581` accepts both shapes but prefers attribute
    access via `getattr(session_state, "session_id", None)`. Using
    SimpleNamespace gives the cleanest read path and matches Starlette's
    own `request.state` convention.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        # Only Console API paths need session pre-population for downstream
        # middleware (Sudo). HTML page handlers and inference paths handle
        # their own auth at the handler level.
        if not path.startswith("/api/account/"):
            return await self.app(scope, receive, send)

        # Read apin_v2_session cookie from raw headers
        # WI-P4-PRE-1: collect ALL Cookie headers, not just the first.
        cookie_header = _collect_cookie_headers(scope)
        raw_token = _parse_cookie_header(cookie_header,
                                         _SESSION_COOKIE_NAME)

        session_info = None
        if raw_token:
            # Import here to avoid module-load circular dependency
            # (account.middlewares is imported BY apin_v2.auth_db's
            # consumers, not the other way around — but the runtime
            # import isolates us from any future re-ordering).
            try:
                from scripts.apin_v2 import auth_db as _auth_db
                row = _auth_db.lookup_session_by_token(raw_token)
                if row:
                    from types import SimpleNamespace
                    session_info = SimpleNamespace(
                        session_id=row["session_id"],
                        user_id=row["user_id"],
                        csrf_token=row.get("csrf_token"),
                    )
            except Exception as e:
                # DB failure is non-fatal — leave session_info=None so
                # downstream middleware handles it as "no session".
                log.warning(
                    "SessionMiddleware DB lookup failed for path=%r: %s",
                    path, e,
                )

        # Populate scope.state.session — handle dict / SimpleNamespace shapes
        if "state" not in scope:
            scope["state"] = {}
        if isinstance(scope["state"], dict):
            scope["state"]["session"] = session_info
        else:
            # SimpleNamespace or similar
            try:
                scope["state"].session = session_info
            except Exception:
                # Last-resort fallback: replace state with a dict
                scope["state"] = {"session": session_info}

        return await self.app(scope, receive, send)


class SudoMiddleware:
    """Verify the sudo HttpOnly cookie on protected `/api/account/*`
    mutation routes. Reads `request.state.session` (set by slot-6 session
    middleware) for the session-id + user-id binding.

    Behaviour:
      - Only fires for POST/PATCH/DELETE/PUT on `/api/account/*`
        (with §9.2 path exemptions).
      - Reads `apin_sudo` cookie via Cookie header.
      - Calls `_verify_sudo_cookie` against `sudo_tokens` table.
      - Failure → 403 `sudo_required` with `X-APIN-Sudo-Hint` header.
      - Success → stash sudo row on `request.state.sudo` and let the
        request proceed.

    What's deferred:
      - PDA-R2-F33 used_count cap enforcement at increment time
        (consume_sudo_use helper, Phase 4)
      - Sudo session-length / max-uses configuration via account_settings
        (Phase 4)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()

        if not path.startswith("/api/account/"):
            return await self.app(scope, receive, send)

        if _is_sudo_exempt(path, method):
            return await self.app(scope, receive, send)

        # Read the sudo cookie
        # WI-P4-PRE-1: collect ALL Cookie headers, not just the first.
        cookie_header = _collect_cookie_headers(scope)
        cookie_value = _parse_cookie_header(cookie_header, SUDO_COOKIE_NAME)

        # Read session binding from request.state if populated by session
        # middleware. In raw ASGI we don't have a Request object; the
        # session middleware also stashes a snapshot in scope (the
        # convention used elsewhere in this project — see apin_server.py).
        #
        # PDA-P2-R2-F01 fix: scope["state"] may be either a dict OR a
        # SimpleNamespace, depending on which session middleware populated
        # it (Starlette's `request.state` is SimpleNamespace-backed; our
        # ad-hoc test fixtures use plain dict). The write side of THIS
        # middleware (lines below) already handles both shapes; the read
        # side must too. Normalise to a uniform "get .session-like field"
        # operation that works for either.
        _scope_state = scope.get("state")
        if _scope_state is None:
            session_state = None
        elif isinstance(_scope_state, dict):
            session_state = _scope_state.get("session")
        else:
            # SimpleNamespace or any object exposing attributes
            session_state = getattr(_scope_state, "session", None)
        session_id = None
        if session_state is not None:
            session_id = getattr(session_state, "session_id", None) \
                       or getattr(session_state, "id", None) \
                       or (session_state.get("session_id")
                           if isinstance(session_state, dict) else None) \
                       or (session_state.get("id")
                           if isinstance(session_state, dict) else None)

        # Read client IP — we don't have the apin_server._client_ip_rightmost
        # helper directly here (would create a circular import), so accept
        # the X-Forwarded-For + scope.client fallback inline. Phase 4 can
        # refactor this if the helper is needed elsewhere.
        client_ip = _client_ip_from_scope(scope)

        if cookie_value is None:
            return await _send_json_error(
                send, 403, "sudo_required",
                "This action requires sudo. POST /api/account/sudo with your password.",
                extra_headers=[(b"x-apin-sudo-hint",
                                b"POST /api/account/sudo")],
            )

        sudo = _verify_sudo_cookie(cookie_value, session_id, client_ip)
        if sudo is None:
            # Distinguish "no/expired/wrong sudo" from "binding mismatch"
            # if we have enough info. Per §7.6 step 6: if we matched a
            # token but binding failed, use sudo_invalid_for_context.
            # Without a "did we match?" signal from _verify_sudo_cookie
            # we conservatively report sudo_required — the more general
            # error code. A future refactor can split the return shape.
            return await _send_json_error(
                send, 403, "sudo_required",
                "Sudo cookie missing, expired, or revoked.",
                extra_headers=[(b"x-apin-sudo-hint",
                                b"POST /api/account/sudo")],
            )

        # WI-P4-SUDO-USED-COUNT (spec §7.6 step 7 + PDA-R2-F33): the
        # mutation is about to proceed; consume one sudo use. If the
        # token has hit its cap (account_settings.sudo_max_uses), the
        # atomic UPDATE in consume_sudo_use returns False — we MUST
        # reject AND revoke the spent token so its cookie value can't
        # be replayed by a leaked-cookie attacker. Defer the helper
        # import to runtime to avoid circular module dependencies.
        try:
            from scripts.apin_v2 import auth_db as _adb
            settings = _adb.get_account_settings(int(sudo.get("user_id") or 0))
            max_uses = int(settings.get("sudo_max_uses", 50) or 50)
            ok = _adb.consume_sudo_use(str(sudo.get("id") or ""), max_uses)
            if not ok:
                # Cap exceeded — revoke the token so the cookie is dead,
                # then 403. The hint message tells the user to re-auth.
                try:
                    with _adb._write_lock, _adb.get_conn() as _c:
                        _c.execute(
                            "UPDATE sudo_tokens SET revoked_at = ? "
                            "WHERE id = ? AND revoked_at IS NULL",
                            (_adb._now_iso(), sudo.get("id")),
                        )
                except Exception as _e:
                    log.warning(
                        "Sudo cap-exceeded auto-revoke failed for sudo_id=%r: %s",
                        sudo.get("id"), _e,
                    )
                return await _send_json_error(
                    send, 403, "sudo_required",
                    "Sudo session reached its use limit. Re-authenticate.",
                    extra_headers=[(b"x-apin-sudo-hint",
                                    b"POST /api/account/sudo")],
                )
        except Exception as e:
            # Defensive: if consume_sudo_use raises, log + fail-CLOSED
            # (don't quietly let the mutation through without counting).
            # FX-P4-6 (PDA-P4-R1 F6): emit audit row on the 503 path —
            # STRIDE A3 repudiation gap if DB issues are induced under
            # load and we have no audit trail of refused mutations.
            log.warning("consume_sudo_use raised: %s — failing closed", e)
            try:
                from scripts.apin_v2 import auth_db as _adb_audit
                _adb_audit.audit(
                    "sudo_accounting_failed",
                    user_id=sudo.get("user_id") if sudo else None,
                    detail={"sudo_id": sudo.get("id") if sudo else None,
                            "error": str(e)[:200]},
                )
            except Exception:
                pass   # never let the audit failure mask the 503
            return await _send_json_error(
                send, 503, "service_unavailable",
                "Sudo accounting temporarily unavailable. Try again.",
            )

        # Stash on scope.state so downstream handlers can read it.
        if "state" not in scope:
            scope["state"] = {}
        if not isinstance(scope["state"], dict):
            # Older ASGI middleware may use SimpleNamespace; convert defensively.
            from types import SimpleNamespace
            if isinstance(scope["state"], SimpleNamespace):
                scope["state"].sudo = sudo
            else:
                scope["state"] = {"sudo": sudo}
        else:
            scope["state"]["sudo"] = sudo

        return await self.app(scope, receive, send)


def _client_ip_from_scope(scope) -> Optional[str]:
    """Inline IP extraction from ASGI scope. Mirrors the rightmost-
    untrusted logic of `apin_server._client_ip_rightmost` (REV-R2-I03)
    but works on raw ASGI scope instead of a `Request` object.

    Avoids a circular import by reimplementing the small algorithm here.
    """
    import os
    try:
        hops = int(os.environ.get("APIN_TRUSTED_PROXY_HOPS", "1"))
    except Exception:
        hops = 1
    if hops < 0:
        hops = 0

    xff = b""
    for k, v in scope.get("headers", []):
        if k == b"x-forwarded-for":
            xff = v
            break

    if xff:
        entries = [
            e.strip().decode("latin1")
            for e in xff.split(b",")
            if e.strip()
        ]
        if entries:
            idx = -(hops + 1)
            if -len(entries) <= idx < 0:
                return entries[idx]
            # Chain too short for trusted-hop window — fall through to
            # the direct TCP peer (matches apin_server.py REV-R6-I04).

    client = scope.get("client")
    if client and len(client) >= 1:
        return client[0]
    return None


# ──────────────────────────────────────────────────────────────────────────
# Phase 9.F · UsageRecordingMiddleware
#
# Phase 9.A wired _record_usage() inside @require_scope. But almost every
# `/api/*` route in apin_server.py authenticates with a local
# `_require_api_key()` helper instead of the decorator — which means
# the recorder never fires for /api/predict/full, /api/predict/quick,
# /api/scan, /api/predict/batch, etc. This middleware closes that gap
# uniformly: it sniffs Bearer / X-API-Key headers, resolves the key
# once via auth_db.find_api_key, and buffers a usage row keyed on the
# real response status code captured from the ASGI `send` stream.
#
# Why a middleware instead of decorating each helper:
#   - Single instrumentation point (DRY)
#   - Captures the FINAL response status (not what the handler thought
#     before envelope wrapping)
#   - Latency timing wraps the entire response lifecycle
#   - Bytes_in captured from Content-Length, bytes_out from the actual
#     wire bytes (Content-Length on the response, or summed body chunks)
#
# What this middleware does NOT do:
#   - Auth validation (handler still calls _require_api_key / decorator).
#     We trust the handler to reject invalid keys with 401 — we'll still
#     record that 401 as a "tried to use this key, failed" telemetry row.
#   - Rate-limit / quota enforcement (Phase 10).
# ──────────────────────────────────────────────────────────────────────────


class UsageRecordingMiddleware:
    """Bearer/X-API-Key-aware usage recorder.

    Algorithm:
      1. Early-out if no Bearer/X-API-Key header present (most requests).
      2. Resolve the key once via `find_api_key` (cheap — hashed lookup).
      3. Wrap `send` to capture:
         - The `http.response.start` message → final status_code.
         - Each `http.response.body` chunk → bytes_out tally.
      4. After the response completes, call `usage_recorder.record_request`
         with all the fields. Buffer flush happens in the worker.

    Failures are swallowed — recording is never allowed to break a request.

    Slot: registered AFTER all the security middlewares (TokenFormat,
    Redaction, etc.) but BEFORE handler — so we see the post-redaction
    response (correct bytes_out for the redacted body), and we don't
    record requests TokenFormat/CORS would reject.
    """

    def __init__(self, app):
        self.app = app
        # Skip listing — paths that should not be recorded as usage even
        # if a key is presented. Tightens the signal so dashboard noise
        # doesn't drown out real API traffic.
        self._skip_prefixes = (
            "/account/api/",   # Console pages + console API routes
            "/static/",
            "/auth/",
            "/api/auth/",
            "/dashboard",
            "/health",
            "/status",
            "/favicon",
            "/landing",
            "/pipeline",
            "/docs",
            "/share/",
            "/robots.txt",
            "/sitemap.xml",
        )

    async def __call__(self, scope, receive, send):
        # Pass non-HTTP traffic (websocket / lifespan) straight through.
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "") or ""

        # Skip non-API paths (the bulk of traffic to this app: the
        # inference UI's static assets, dashboard HTML, etc).
        if any(path.startswith(p) for p in self._skip_prefixes):
            return await self.app(scope, receive, send)

        # 9.N.8 · Payload capture constants. Two-tier strategy:
        #   _SIPHON_CAP — how much we buffer from the wire per direction.
        #                 Must be large enough to capture useful JSON
        #                 structure AROUND huge embedded blobs (gradcam
        #                 base64, vectors, etc.).
        #   _PREVIEW_CAP — target output size after smart-trimming JSON.
        # Binary / image content uses the dedicated 32 KB capture in
        # _body_to_preview's image branch.
        # 256KB siphon — large enough to cover the ML response bodies that
        # embed full-resolution base64 PNGs (~50–200KB gradcam blobs).
        # We never store this directly — the preview goes through the
        # smart-trim step which collapses huge strings before persisting.
        _SIPHON_CAP = 256 * 1024
        _PREVIEW_CAP = 4096
        # Header names whose VALUES must be redacted before storage.
        # Names themselves are kept (useful to know auth method was used).
        _REDACT_HEADER_NAMES = {
            b"authorization", b"cookie", b"x-api-key",
            b"x-auth-token", b"x-csrf-token", b"x-session-token",
            b"proxy-authorization", b"set-cookie",
        }
        # Content types we DON'T try to decode-as-text; we just record the
        # type + size and skip the body preview (binary placeholder).
        _BINARY_CTYPE_PREFIXES = (
            "image/", "audio/", "video/",
            "application/octet-stream", "application/pdf",
            "application/zip", "application/gzip",
        )

        def _redact_header_value(name_lower: bytes, raw: bytes) -> str:
            """Redact sensitive header values; keep first 4 + last 4 chars
            for traceability of token rotation, etc."""
            s = raw.decode("latin1", errors="replace")
            if name_lower in _REDACT_HEADER_NAMES:
                if len(s) > 12:
                    return s[:4] + "•" * 6 + s[-4:]
                return "•" * min(8, len(s))
            return s[:512]   # cap per-value length

        def _headers_to_json_dict(headers_list) -> dict:
            """Convert ASGI header list to JSON-safe dict with redaction.
            Header names are lowercased; values are redacted per the rules."""
            out = {}
            for k, v in headers_list:
                kl = k.lower()
                try:
                    name = k.decode("ascii", errors="replace").lower()
                except Exception:
                    continue
                out[name] = _redact_header_value(kl, v)
            return out

        def _body_to_preview(body: bytes, ctype: str) -> tuple:
            """Convert a body bytes blob to a JSON-safe preview string.
            Returns (preview_text, truncated_bool). Binary content yields
            a placeholder describing size + type. Multipart content yields
            a JSON-encoded structured part list so the drawer can render
            it properly (with image thumbnails)."""
            if not body:
                return ("", False)
            ctype_lower = (ctype or "").lower()
            # ── 9.N.8e · Multipart parser ─────────────────────────────
            # Parse boundary out of ctype, then split body into parts and
            # capture each part's headers + filename + a small base64
            # preview if it's an image. The output is a JSON structure
            # the client can render as a structured list (with image
            # thumb hover popovers), NOT as garbled mojibake text.
            if ctype_lower.startswith("multipart/"):
                try:
                    return (_parse_multipart_for_drawer(body, ctype_lower), False)
                except Exception:
                    # Fall through to binary handling if parsing fails
                    pass
            # Binary content → placeholder marker (with base64 preview
            # for images so the client can show a hover thumbnail).
            if any(ctype_lower.startswith(p) for p in _BINARY_CTYPE_PREFIXES):
                if ctype_lower.startswith("image/") and len(body) > 0:
                    # For raw image bodies (not multipart), embed a base64
                    # preview directly into the JSON structure so the client
                    # can render <img src="data:image/...;base64,...">.
                    import base64 as _b64
                    cap = 32 * 1024   # 32KB cap on stored thumbnail bytes
                    head = body[:cap]
                    try:
                        b64 = _b64.b64encode(head).decode("ascii")
                        import json as _json
                        return (
                            _json.dumps({
                                "kind": "image",
                                "ctype": ctype_lower,
                                "size_bytes": len(body),
                                "preview_b64": b64,
                                "truncated": len(body) > cap,
                            }, separators=(",", ":")),
                            len(body) > cap,
                        )
                    except Exception:
                        pass
                return (
                    f"[binary · {len(body)} bytes · {ctype_lower or 'unknown type'}]",
                    False,
                )
            # Text-ish content → decode + cap
            try:
                txt = body.decode("utf-8", errors="replace")
            except Exception:
                return (f"[undecodable · {len(body)} bytes]", False)
            # 9.N.8e · For JSON, smart-trim huge string values (typically
            # base64-encoded images, vectors, etc.) so the structure stays
            # visible inside the preview cap. Without this, a response like
            # /api/predict/full with a 50 KB gradcam_b64_png field eats the
            # entire 4 KB preview before the useful keys (status, confidence,
            # diagnosis, ...) can appear.
            if "application/json" in ctype_lower or "+json" in ctype_lower:
                trimmed_txt, was_trimmed = _trim_json_for_preview(txt, _PREVIEW_CAP)
                if was_trimmed:
                    return (trimmed_txt, True)
                # Even if not trimmed, the JSON might still be over cap
                if len(trimmed_txt) > _PREVIEW_CAP:
                    return (trimmed_txt[:_PREVIEW_CAP], True)
                return (trimmed_txt, False)
            if len(txt) > _PREVIEW_CAP:
                return (txt[:_PREVIEW_CAP], True)
            return (txt, False)

        def _trim_json_for_preview(text: str, target_size: int) -> tuple:
            """Smart-trim a JSON document so its STRUCTURE survives a small
            preview window. Any string value > 200 chars is replaced with a
            short marker. Returns (trimmed_text, was_trimmed_bool).
            Handles truncated JSON via _recover_truncated_json — important
            for ML responses with embedded base64 PNGs that exceed the wire
            siphon cap."""
            import json as _json
            obj = None
            recovered_from_truncation = False
            try:
                obj = _json.loads(text)
            except Exception:
                # Try to recover a partial JSON object from a truncated tail.
                # Walks backward from the cut to find a valid object/array
                # boundary, then closes any unbalanced brackets.
                recovered = _recover_truncated_json(text)
                if recovered is not None:
                    try:
                        obj = _json.loads(recovered)
                        recovered_from_truncation = True
                    except Exception:
                        obj = None
            if obj is None:
                # JSON irrecoverable — return raw text. The caller will
                # truncate it to the preview cap.
                return (text, False)
            trimmed_flag = [recovered_from_truncation]
            STR_VALUE_CAP = 200
            def _walk(o):
                if isinstance(o, str):
                    if len(o) > STR_VALUE_CAP:
                        trimmed_flag[0] = True
                        head = o[:32]
                        tail_info = "[" + str(len(o)) + " chars truncated]"
                        return head + "…" + tail_info
                    return o
                if isinstance(o, list):
                    if len(o) > 50:
                        trimmed_flag[0] = True
                        return [_walk(x) for x in o[:50]] + [
                            "[" + str(len(o) - 50) + " more items]"]
                    return [_walk(x) for x in o]
                if isinstance(o, dict):
                    return {k: _walk(v) for k, v in o.items()}
                return o
            trimmed = _walk(obj)
            try:
                out_text = _json.dumps(trimmed, indent=2, ensure_ascii=False)
                if recovered_from_truncation:
                    # Add a clear marker so the UI knows the structure was
                    # recovered from a truncated body.
                    if isinstance(trimmed, dict):
                        trimmed["__truncated__"] = "response body was larger than siphon cap; some keys may be missing"
                        out_text = _json.dumps(trimmed, indent=2, ensure_ascii=False)
                return (out_text, trimmed_flag[0])
            except Exception:
                return (text, False)

        def _recover_truncated_json(text: str):
            """Recover a valid-JSON prefix from a truncated body.

            Bracket-balance state machine. Tracks string boundaries and
            escape sequences. Critically, only marks a "safe truncation
            point" after we've actually closed a key/value pair — NOT
            after we close a string (because the string might be a key
            with no value yet).

            Algorithm:
              · Walk char-by-char, maintain depth + stack of open brackets
              · Track expecting-key vs expecting-value vs after-value state
              · Update `last_safe_idx` only at points where the JSON is
                in a "between values at some depth" state — i.e., just
                after a `,` or just inside an empty `{` or `[`.
              · At end of input, truncate to last_safe_idx, strip trailing
                comma, then close all open brackets in stack order.
            """
            if not text or text[0] not in "{[":
                return None
            in_string = False
            escape = False
            # State at each depth: True means "expecting a key (or first key)"
            # i.e. we're inside an object that has no entries OR just saw a
            # comma. False means we're inside an array (no keys) or have just
            # seen a `:` (expecting a value).
            #
            # We track these as a parallel stack to `bracket_stack`.
            bracket_stack = []         # of '{' or '['
            expecting_key = []         # parallel to bracket_stack; True if obj-key-mode
            value_just_closed = False  # set True after a complete value (string/num/lit/obj/arr)
            last_safe_idx = 1          # default: just after the opening `{` or `[`
            i = 0
            n = len(text)
            while i < n:
                ch = text[i]
                if escape:
                    escape = False
                    i += 1
                    continue
                if ch == "\\":
                    escape = True
                    i += 1
                    continue
                if ch == '"':
                    if in_string:
                        in_string = False
                        # A string just closed. Whether it's a key or value:
                        if bracket_stack and bracket_stack[-1] == '{' and expecting_key[-1]:
                            # That string was a key. Wait for `:` then value.
                            expecting_key[-1] = False   # now expecting value
                        else:
                            # That string was a value (either in array or after :)
                            value_just_closed = True
                            last_safe_idx = i + 1
                    else:
                        in_string = True
                    i += 1
                    continue
                if in_string:
                    i += 1
                    continue
                if ch in " \t\n\r":
                    i += 1
                    continue
                if ch == "{":
                    bracket_stack.append("{")
                    expecting_key.append(True)
                    value_just_closed = False
                    last_safe_idx = i + 1   # just after `{` is safe (empty obj)
                    i += 1
                    continue
                if ch == "[":
                    bracket_stack.append("[")
                    expecting_key.append(False)   # arrays don't have keys
                    value_just_closed = False
                    last_safe_idx = i + 1   # just after `[` is safe (empty arr)
                    i += 1
                    continue
                if ch == "}" or ch == "]":
                    if bracket_stack:
                        bracket_stack.pop()
                        expecting_key.pop()
                    value_just_closed = True
                    last_safe_idx = i + 1
                    if not bracket_stack:
                        # Top-level closed — full doc parsed.
                        return text[: last_safe_idx]
                    i += 1
                    continue
                if ch == ":":
                    # We just had a key; now expecting a value.
                    value_just_closed = False
                    i += 1
                    continue
                if ch == ",":
                    # A complete value/pair ends here. Inside an object, we
                    # now expect another key.
                    if bracket_stack and bracket_stack[-1] == '{':
                        expecting_key[-1] = True
                    value_just_closed = False
                    last_safe_idx = i   # cut at the comma so we drop it next
                    i += 1
                    continue
                # Number / true / false / null token. Scan ahead until a
                # JSON-structural character.
                start = i
                while i < n and text[i] not in ",{}[]: \t\n\r":
                    i += 1
                # We just consumed a value token.
                value_just_closed = True
                last_safe_idx = i
            # Walked the whole input without closing the root.
            if not bracket_stack:
                return text
            # Truncate to last_safe_idx, strip trailing comma, close all open brackets.
            recovered = text[:last_safe_idx].rstrip()
            if recovered.endswith(","):
                recovered = recovered[:-1]
            for openc in reversed(bracket_stack):
                recovered += "}" if openc == "{" else "]"
            return recovered

        def _parse_multipart_for_drawer(body: bytes, ctype_lower: str) -> str:
            """Parse a multipart body into a structured JSON for the drawer.
            Returns a JSON string with kind=multipart and a parts[] list,
            each containing name/filename/ctype/size/preview_b64 (for image
            parts only, capped at 32KB)."""
            import base64 as _b64
            import json as _json
            import re as _re
            # Extract boundary parameter
            m = _re.search(r'boundary=("?)([^";]+)\1', ctype_lower)
            if not m:
                return _json.dumps({
                    "kind": "multipart",
                    "ctype": ctype_lower,
                    "size_bytes": len(body),
                    "parts": [],
                    "error": "no boundary",
                }, separators=(",", ":"))
            boundary = m.group(2).strip()
            # Boundary in body is preceded by `--` and lines end with \r\n
            sep = ("--" + boundary).encode("ascii", errors="ignore")
            parts_bytes = body.split(sep)
            # Drop the preamble + trailing closer
            parts_bytes = [p for p in parts_bytes if p.strip(b"\r\n-") != b""]
            parts_out = []
            IMAGE_PREVIEW_CAP = 32 * 1024     # 32 KB per image
            TOTAL_PREVIEW_CAP = 64 * 1024     # never exceed 64 KB across all images
            total_b64_bytes = 0
            for p in parts_bytes:
                # Each part: headers \r\n\r\n body \r\n
                if p.startswith(b"\r\n"):
                    p = p[2:]
                if p.endswith(b"\r\n"):
                    p = p[:-2]
                hdr_split = p.split(b"\r\n\r\n", 1)
                if len(hdr_split) != 2:
                    continue
                hdr_bytes, body_part = hdr_split
                try:
                    hdr_text = hdr_bytes.decode("utf-8", errors="replace")
                except Exception:
                    continue
                # Parse Content-Disposition + Content-Type
                part_info = {"size_bytes": len(body_part)}
                name_m = _re.search(r'name="([^"]*)"', hdr_text)
                if name_m:
                    part_info["name"] = name_m.group(1)
                fn_m = _re.search(r'filename="([^"]*)"', hdr_text)
                if fn_m:
                    part_info["filename"] = fn_m.group(1)
                ct_m = _re.search(r'(?im)^Content-Type:\s*([^\r\n]+)', hdr_text)
                if ct_m:
                    part_info["ctype"] = ct_m.group(1).strip().lower()
                pct = part_info.get("ctype", "")
                # For image parts, embed base64 preview (up to cap)
                if pct.startswith("image/") and len(body_part) > 0 \
                        and total_b64_bytes < TOTAL_PREVIEW_CAP:
                    remaining = TOTAL_PREVIEW_CAP - total_b64_bytes
                    take = min(IMAGE_PREVIEW_CAP, remaining, len(body_part))
                    try:
                        part_info["preview_b64"] = _b64.b64encode(body_part[:take]).decode("ascii")
                        part_info["preview_truncated"] = (take < len(body_part))
                        total_b64_bytes += take
                    except Exception:
                        pass
                parts_out.append(part_info)
            return _json.dumps({
                "kind": "multipart",
                "ctype": ctype_lower,
                "size_bytes": len(body),
                "parts": parts_out,
            }, separators=(",", ":"))

        # Look for an API key in the headers BEFORE running the handler.
        # `headers` is a list of (bytes, bytes) tuples in ASGI scope.
        raw_token = None
        ua = None
        bytes_in = None
        req_ctype = None
        for k, v in scope.get("headers", []):
            kl = k.lower()
            if kl == b"authorization":
                vd = v.decode("latin1", errors="replace")
                if vd.lower().startswith("bearer "):
                    raw_token = vd[7:].strip()
            elif kl == b"x-api-key" and raw_token is None:
                raw_token = v.decode("latin1", errors="replace").strip()
            elif kl == b"user-agent":
                ua = v.decode("latin1", errors="replace")
            elif kl == b"content-length":
                try:
                    bytes_in = int(v.decode("ascii"))
                except Exception:
                    bytes_in = None
            elif kl == b"content-type":
                req_ctype = v.decode("latin1", errors="replace")

        # Cheap exit — no API auth on this request, nothing to record.
        if not raw_token:
            return await self.app(scope, receive, send)

        # Resolve the key.
        key_record = None
        try:
            from scripts.apin_v2 import auth_db as _adb
            key_record = _adb.lookup_api_key_full(raw_token)
        except Exception:
            key_record = None

        if key_record is None:
            return await self.app(scope, receive, send)

        # ── 9.N.8 · Stage timings + payload capture wrappers ────────────────
        import time as _time
        started = _time.monotonic()
        t_auth_done = _time.monotonic()   # auth resolution finished here
        t_first_body = None               # set on first response chunk
        t_finished = None

        # Pre-redacted request headers snapshot (taken once before handler runs).
        req_headers_dict = _headers_to_json_dict(scope.get("headers", []))

        # Buffer for request body capture (up to PREVIEW_CAP bytes).
        req_body_buf = bytearray()
        req_body_full_size = 0
        req_body_done = False

        async def wrapped_receive():
            """Wrap ASGI `receive()` to siphon the request body for preview
            while passing every chunk through to the handler unchanged."""
            nonlocal req_body_done, req_body_full_size
            msg = await receive()
            if msg.get("type") == "http.request":
                body = msg.get("body") or b""
                if body:
                    req_body_full_size += len(body)
                    # Only buffer up to the cap; further bytes flow through
                    # but don't get stored.
                    space = _SIPHON_CAP - len(req_body_buf)
                    if space > 0:
                        req_body_buf.extend(body[:space])
                if not msg.get("more_body", False):
                    req_body_done = True
            return msg

        # Response capture state.
        status_holder = {
            "code": 200,
            "bytes_out": 0,
            "headers": [],       # list of (bytes, bytes) tuples
            "ctype": None,       # decoded text content-type
        }
        resp_body_buf = bytearray()
        resp_body_full_size = 0

        async def wrapped_send(message):
            nonlocal t_first_body, resp_body_full_size
            mt = message.get("type")
            if mt == "http.response.start":
                status_holder["code"] = int(message.get("status", 200))
                hdrs = message.get("headers") or []
                status_holder["headers"] = list(hdrs)
                for k, v in hdrs:
                    if k.lower() == b"content-type":
                        status_holder["ctype"] = v.decode("latin1", "replace")
                        break
            elif mt == "http.response.body":
                if t_first_body is None:
                    t_first_body = _time.monotonic()
                body = message.get("body") or b""
                if body:
                    status_holder["bytes_out"] += len(body)
                    resp_body_full_size += len(body)
                    space = _SIPHON_CAP - len(resp_body_buf)
                    if space > 0:
                        resp_body_buf.extend(body[:space])
            await send(message)

        # Run the handler. Whether it succeeds or raises, we want to
        # record the attempt.
        client_ip = _client_ip_from_scope(scope) or ""
        t_handler_start = _time.monotonic()
        try:
            # 9.N.8 · Use wrapped_receive (siphons request body) +
            # wrapped_send (siphons response body + headers + timings).
            await self.app(scope, wrapped_receive, wrapped_send)
        finally:
            t_finished = _time.monotonic()
            latency_ms = int((t_finished - started) * 1000)
            try:
                from scripts.apin_v2 import usage_recorder as _ur
                final_status = int(status_holder["code"])
                # 9.H · D7 fix — capture an error_code for every non-2xx so
                # the top-error-codes panel and request detail card have
                # something meaningful to show. Mapping:
                #   200/204/...  → None (success, no error to label)
                #   3xx          → `http_3xx`
                #   429          → `rate_limited`
                #   401          → `auth_invalid`
                #   403          → `forbidden`
                #   404          → `not_found`
                #   422          → `unprocessable`
                #   400/4xx      → `bad_request`
                #   5xx          → `server_error`
                _SPECIFIC = {
                    400: "bad_request", 401: "auth_invalid", 403: "forbidden",
                    404: "not_found",   408: "timeout",      409: "conflict",
                    410: "gone",        413: "payload_too_large",
                    415: "unsupported_media_type",
                    422: "unprocessable", 429: "rate_limited",
                    500: "internal_error", 502: "bad_gateway",
                    503: "service_unavailable", 504: "gateway_timeout",
                }
                if final_status < 300:
                    error_code = None
                elif final_status in _SPECIFIC:
                    error_code = _SPECIFIC[final_status]
                elif 300 <= final_status < 400:
                    error_code = "http_3xx"
                elif 400 <= final_status < 500:
                    error_code = "bad_request"
                else:
                    error_code = "server_error"

                # 9.N.8 · Compute the payload + stage-timing extras to pass.
                # All inside try/except so any single bad value can't break
                # the recording pipeline.
                try:
                    req_body_preview, req_body_truncated = _body_to_preview(
                        bytes(req_body_buf), req_ctype or "",
                    )
                except Exception:
                    req_body_preview, req_body_truncated = None, False
                try:
                    resp_body_preview, resp_body_truncated = _body_to_preview(
                        bytes(resp_body_buf), status_holder["ctype"] or "",
                    )
                except Exception:
                    resp_body_preview, resp_body_truncated = None, False
                try:
                    resp_headers_dict = _headers_to_json_dict(status_holder["headers"])
                except Exception:
                    resp_headers_dict = None

                # Stage breakdown in ms. Some stages are merged because the
                # middleware can't see inside the handler — that's OK; the
                # waterfall just shows what we DO know.
                stage_timings_map = None
                try:
                    auth_ms      = max(0, int((t_auth_done   - started)         * 1000))
                    pre_handler  = max(0, int((t_handler_start - t_auth_done)   * 1000))
                    if t_first_body is not None:
                        handler_ms = max(0, int((t_first_body  - t_handler_start) * 1000))
                        send_ms    = max(0, int((t_finished     - t_first_body)   * 1000))
                    else:
                        handler_ms = max(0, int((t_finished - t_handler_start) * 1000))
                        send_ms    = 0
                    stage_timings_map = {
                        "auth":     auth_ms,
                        "validate": pre_handler,
                        "handler":  handler_ms,
                        "send":     send_ms,
                    }
                except Exception:
                    stage_timings_map = None

                _ur.record_request(
                    key_id=key_record.get("public_id") or "",
                    user_id=int(key_record.get("user_id") or 0),
                    method=scope.get("method", "GET"),
                    path=path,
                    status_code=final_status,
                    latency_ms=latency_ms,
                    ip=client_ip or None,
                    ua=ua,
                    bytes_in=(req_body_full_size or bytes_in),
                    bytes_out=status_holder["bytes_out"] or None,
                    error_code=error_code,
                    via="bearer",
                    rate_limited=(final_status == 429),
                    quota_blocked=False,
                    # 9.N.8 payload + timings
                    headers_in=req_headers_dict,
                    headers_out=resp_headers_dict,
                    body_in_preview=req_body_preview,
                    body_out_preview=resp_body_preview,
                    body_in_ctype=req_ctype,
                    body_out_ctype=status_holder["ctype"],
                    body_in_truncated=req_body_truncated,
                    body_out_truncated=resp_body_truncated,
                    stage_timings=stage_timings_map,
                    # 9.N.8g · Forward key name into SSE event so recent-
                    # requests live rows show "test-app" instead of "·".
                    key_name=key_record.get("name") or None,
                )
            except Exception as e:
                log.debug("UsageRecordingMiddleware swallow: %s", e)
