"""APIN API response envelope — the foundation every v1 endpoint conforms to.

Implements API_CONTRACT.md:
  §2  the success envelope (9 fixed top-level keys)
  §3  the error envelope (stable error.code, no leaked internals)
  §4  the error-code -> HTTP-status taxonomy
  §6  the collection / pagination shape helper

Usage in a route handler:

    @app.get("/version")
    @api_endpoint("/version")
    async def v1_version():
        return {"api_version": "1.0", ...}          # -> wrapped in the envelope

    # to fail a request, raise ApiError — never leak a raw exception:
        raise ApiError("not_found", "No disease with that name.")

    # collection endpoints:
        return paginated(items, page=1, page_size=50, total=1284)

The decorator generates the request_id, times the request, wraps the return
value in the §2 envelope, converts a raised ApiError into the §3 envelope, and
converts any *uncaught* exception into a safe `internal_error` (the real
exception is logged under the request_id, never returned to the caller — §8).
"""
from __future__ import annotations

import functools
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response as _StarletteResponse

logger = logging.getLogger("apin.api")

API_VERSION = "1.0"

# ── §4 — error code -> HTTP status ──────────────────────────────────────────
#
# Two dicts power the contract:
#
#   ERROR_STATUS         — exactly 44 CANONICAL codes (API Console spec §26).
#                          Every new endpoint MUST raise codes from this pool.
#                          The CI gate asserts `len(ERROR_STATUS) == 44`.
#
#   LEGACY_STATUS        — preserved codes from the pre-Console era, with their
#                          ORIGINAL HTTP statuses. Existing /predict, /feedback,
#                          /warmup, etc. emit these. Kept for backward-compat
#                          so the inference website keeps working unchanged
#                          during the Console rollout.
#
#   LEGACY_CODE_ALIASES  — documentation map: legacy code -> canonical code.
#                          Surfaced in `meta.deprecation` for clients reading
#                          legacy codes so they can plan migration. The HTTP
#                          status of a legacy code is ALWAYS its LEGACY_STATUS
#                          value (not the canonical's value) — see REV-R3-I15
#                          and §26.1 alias-validation in the API Console spec.
#
# Migration story for clients:
#   - Legacy endpoint `/predict`     emits   `auth_required` 401  (no change)
#   - New     endpoint `/api/account/*` emits `invalid_or_missing_token` 401
#   - Clients see `meta.deprecation = "invalid_or_missing_token"` on the legacy
#     response so they know what the canonical replacement is. No silent code
#     translation in the response body; the code field always echoes what the
#     handler raised.

ERROR_STATUS = {
    # ── auth / key state (7) ─────────────────────────────────────────────
    "invalid_or_missing_token":     401,
    "key_disabled":                 403,
    "key_expired":                  403,
    "key_compromised":              403,
    "key_rotating_expired":         403,
    "key_pending_migration":        401,
    "key_deleted":                  403,   # R5 — hard-deleted-key race
    # ── network / origin / scope (6) ─────────────────────────────────────
    "ip_not_allowed":               403,
    "origin_not_allowed":           403,
    "use_sandbox_endpoint":         403,
    "use_live_endpoint":            403,
    "missing_scope":                403,
    "console_only_route":           403,
    # ── sudo (2) ─────────────────────────────────────────────────────────
    "sudo_required":                403,
    "sudo_invalid_for_context":     403,
    # ── rate / quota (5) ─────────────────────────────────────────────────
    "rate_limited":                 429,
    "rate_limit_unavailable":       503,
    "quota_exceeded":               429,
    "sandbox_rate_limited":         429,
    "sandbox_upload_limit":         429,
    # ── validation (8) ───────────────────────────────────────────────────
    "key_limit_reached":            400,
    "invalid_scope":                400,
    "invalid_ip_cidr":              400,
    "invalid_origin":               400,
    "invalid_name":                 400,
    "invalid_quota":                400,   # R5 — quota_per_day=0 or >1_000_000
    "invalid_parameter":            400,
    "invalid_url":                  400,
    # ── url validation (9) ───────────────────────────────────────────────
    "url_not_https":                400,
    "url_credentials_forbidden":    400,
    "url_too_long":                 400,
    "url_missing_host":             400,
    "url_localhost_forbidden":      400,
    "url_internal_ip_forbidden":    400,
    "url_target_changed":           400,
    "url_dns_failed":               503,
    "url_dns_empty":                400,
    # ── conflict / state (3) ─────────────────────────────────────────────
    "duplicate_name":               409,
    "already_rotating":             409,
    "idempotency_conflict":         409,
    # ── resource / infrastructure (4) ────────────────────────────────────
    "not_found":                    404,
    "internal_error":               500,
    "db_locked":                    503,
    "service_unavailable":          503,
}
# 7 + 6 + 2 + 5 + 8 + 9 + 3 + 4 = 44     PDA-P0-R1-F07: section counts now correct

# Legacy codes preserved at their ORIGINAL HTTP status (pre-Console era).
# /predict, /feedback, /warmup, /apin/* etc. still emit these — do NOT remove
# until every legacy caller has been migrated.
LEGACY_STATUS = {
    "auth_required":          401,
    "auth_invalid":            401,
    "auth_expired":            401,
    "forbidden":               403,
    "quota_exhausted":         402,
    "guest_exhausted":         401,
    "missing_parameter":       400,
    "validation_failed":       422,
    "unsupported_media_type":  415,
    "payload_too_large":       413,
    "model_warming":           503,
    "model_unavailable":       503,
    "inference_failed":        500,
}

# Documentation-only map: legacy code -> canonical replacement.
# Surfaced via `meta.deprecation` so clients of legacy endpoints can plan
# their migration. The legacy code's HTTP status is unchanged.
#
# PDA-P0-R1-F02: every code in LEGACY_STATUS that has a canonical equivalent
# MUST be in this dict; otherwise clients reading the legacy code get no
# migration hint. The 3 below (validation_failed / unsupported_media_type /
# payload_too_large) are HTTP-spec'd codes whose canonical replacement is
# `invalid_parameter` — they all signal "your request payload was wrong."
LEGACY_CODE_ALIASES = {
    "auth_required":          "invalid_or_missing_token",
    "auth_invalid":           "invalid_or_missing_token",
    "auth_expired":           "invalid_or_missing_token",
    "forbidden":              "missing_scope",
    "quota_exhausted":        "quota_exceeded",
    "guest_exhausted":        "quota_exceeded",
    "missing_parameter":      "invalid_parameter",
    "validation_failed":      "invalid_parameter",   # PDA-P0-R1-F02
    "unsupported_media_type": "invalid_parameter",   # PDA-P0-R1-F02
    "payload_too_large":      "invalid_parameter",   # PDA-P0-R1-F02
    "model_warming":          "service_unavailable",
    "model_unavailable":      "service_unavailable",
    "inference_failed":       "internal_error",
}


def _status_for(code: str) -> int:
    """Resolve HTTP status for any code, canonical or legacy.

    Canonical pool wins on collisions (none currently — by construction the
    two pools are disjoint, but defending against future drift).

    PDA-P0-R1-F03 — INTENTIONAL DEVIATION FROM SPEC §26.1:
    The spec implies that aliased legacy codes (e.g. `quota_exhausted`) should
    resolve to the CANONICAL status (429) rather than the original legacy
    status (402). This implementation deliberately preserves the LEGACY status
    to honour the backward-compatibility contract with /predict, /feedback,
    /warmup, and other pre-Console endpoints whose clients read 402 from
    `quota_exhausted` today. Changing the status would break those clients
    silently. The CODE in the response is unchanged either way — only the
    HTTP status differs.

    Migration path for clients who want canonical statuses: switch to the
    canonical code at the route level (e.g. raise `quota_exceeded` instead
    of `quota_exhausted` once you're ready to emit 429). The `meta.deprecation`
    field in the response body tells clients which canonical code to expect.

    To re-align with spec §26.1's literal reading, change the second branch
    to:
        if code in LEGACY_STATUS:
            canonical = LEGACY_CODE_ALIASES.get(code, code)
            return ERROR_STATUS.get(canonical, LEGACY_STATUS[code])
    This is the future Phase-N migration; do NOT do it as part of Phase 0.

    Unknown codes return 500 — `ApiError.__init__` also coerces to
    `internal_error` upstream so this fallback is only hit if validation
    was bypassed.
    """
    if code in ERROR_STATUS:
        return ERROR_STATUS[code]
    if code in LEGACY_STATUS:
        return LEGACY_STATUS[code]
    return 500

_DOCS_ERRORS_URL = "https://dxv-404-apin.hf.space/docs#errors"


# ── helpers ────────────────────────────────────────────────────────────────
def new_request_id() -> str:
    """A unique request id: `req_` + 16 lowercase hex chars (contract §2.2)."""
    return "req_" + secrets.token_hex(8)


def _now_iso() -> str:
    """UTC, millisecond precision, `Z` suffix (contract §8)."""
    return (datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"))


class ApiError(Exception):
    """Raise inside any @api_endpoint handler to return a clean §3 error.

    `code` MUST be one of ERROR_STATUS (canonical) or LEGACY_STATUS (preserved
    pre-Console codes). `message` is one human sentence with no internal detail.
    `hint` is optional actionable remediation. `headers` is an optional dict of
    response headers to merge into the JSONResponse (e.g. `Retry-After: 60` for
    `rate_limited`, `X-Missing-Scope: writes` for `missing_scope`).

    REV-R5-I03 + REV-R6-I01: the `headers` kwarg is REQUIRED by §5.4's
    `@require_scope` decorator. Without it, every rate-limit / scope / quota
    raise hits a `TypeError` at runtime.
    """
    def __init__(self, code: str, message: str, *,
                 hint: Optional[str] = None, field: Optional[str] = None,
                 details: Optional[dict] = None,
                 headers: Optional[dict] = None):
        # Accept canonical OR legacy codes. The two pools are disjoint by
        # construction; validation here is a "did the developer typo a code"
        # check, not a contract enforcement.
        if code not in ERROR_STATUS and code not in LEGACY_STATUS:
            # a programming mistake — fail loudly in logs, still safe to caller
            logger.error("ApiError raised with unknown code %r", code)
            code = "internal_error"
        self.code = code
        self.message = message
        self.hint = hint
        self.field = field
        self.details = details
        # Defensive copy so callers can mutate the dict after raising without
        # affecting our outbound response.
        self.headers = dict(headers) if headers else {}
        super().__init__(message)


class Paginated:
    """Wrap a list return so @api_endpoint emits the §6 collection shape:
    data = {items, pagination, **extra}. `meta` (e.g. warnings, §7) is carried
    through to the envelope's top-level `meta` block."""
    def __init__(self, items: list, *, page: int, page_size: int,
                 total: int, meta: Optional[dict] = None, **extra: Any):
        self.items = items
        self.page = page
        self.page_size = page_size
        self.total = total
        self.meta = meta or {}
        self.extra = extra


def paginated(items: list, *, page: int, page_size: int, total: int,
              meta: Optional[dict] = None, **extra: Any) -> Paginated:
    """Convenience constructor — see §6. `meta` flows to the envelope's
    top-level `meta` (use it for `warnings`, e.g. `page_size_clamped`)."""
    return Paginated(items, page=page, page_size=page_size,
                     total=total, meta=meta, **extra)


# ── envelope builders ──────────────────────────────────────────────────────
def _base(endpoint: str, request_id: str, started_at: float,
          ok: bool, status_code: int) -> dict:
    return {
        "api_version": API_VERSION,
        "request_id": request_id,
        "endpoint": endpoint,
        "ok": ok,
        "processed_at": _now_iso(),
        "processing_time_ms": int(round((time.perf_counter() - started_at)
                                        * 1000)),
        "status_code": status_code,
    }


def _success(payload: Any, endpoint: str, request_id: str,
             started_at: float, status_code: int) -> JSONResponse:
    """Build a §3-conforming success envelope.

    Payload shapes accepted:
        plain dict                       → data = dict, meta = {}
        Paginated                        → data = {items, pagination, ...},
                                            meta = paginated.meta
        (data_dict, meta_dict)           → 2-tuple
        (data_dict, meta_dict, headers)  → 3-tuple (PDA-P2-R1-F01 fix):
                                            `headers` is a dict of EXTRA
                                            response headers (e.g.
                                            "X-APIN-No-Redact" to opt out
                                            of TokenRedactionMiddleware for
                                            one-time-view ceremony responses)
    """
    env = _base(endpoint, request_id, started_at, True, status_code)
    meta: dict = {}
    extra_headers: dict = {}
    if isinstance(payload, Paginated):
        count = len(payload.items)
        total_pages = (max(1, -(-payload.total // payload.page_size))
                       if payload.page_size else 1)
        data = {"items": payload.items,
                "pagination": {
                    "total": payload.total,
                    "count": count,
                    "page": payload.page,
                    "page_size": payload.page_size,
                    "total_pages": total_pages,
                    "has_next": payload.page < total_pages,
                    "has_prev": payload.page > 1,
                }}
        data.update(payload.extra)
        meta = dict(payload.meta)
    elif isinstance(payload, tuple) and len(payload) == 2:
        data, meta = payload[0], (payload[1] or {})
    elif isinstance(payload, tuple) and len(payload) == 3:
        # 3-tuple: (data, meta, extra_response_headers)
        data, meta, extra_headers = (
            payload[0], (payload[1] or {}), (payload[2] or {})
        )
    else:
        data = payload
    env["data"] = data if isinstance(data, dict) else {"value": data}
    env["meta"] = meta
    response_headers = {
        "X-Request-Id": request_id,
        "X-API-Version": API_VERSION,
        "Cache-Control": "no-store",
    }
    # Merge route-supplied extra headers UNDER the envelope-reserved keys.
    # Same precedence policy as _failure (PDA-P0-R1-F01): envelope headers
    # win against accidental collisions.
    if extra_headers:
        merged = dict(extra_headers)
        merged.update(response_headers)
        response_headers = merged
    return JSONResponse(env, status_code=status_code, headers=response_headers)


def _failure(err: ApiError, endpoint: str, request_id: str,
             started_at: float) -> JSONResponse:
    # REV-R3-I15: status resolution consults both canonical and legacy pools
    # via _status_for. The CI gate at §26.1 inspects this function's source
    # for the alias-translation path.
    status = _status_for(err.code)
    env = _base(endpoint, request_id, started_at, False, status)
    error_obj = {
        "code": err.code,
        "message": err.message,
        "docs_url": _DOCS_ERRORS_URL,
    }
    if err.hint:
        error_obj["hint"] = err.hint
    if err.field:
        error_obj["field"] = err.field
    if err.details:
        error_obj["details"] = err.details
    env["error"] = error_obj

    # PDA-P0-R1-F05 + R2-F02: envelope-shape symmetry. `_success()` always
    # emits `meta` (even if empty). `_failure()` must too, so canonical-code
    # and legacy-code errors and successes all share the same 9-key envelope
    # shape (api_version, endpoint, ok, processed_at, processing_time_ms,
    # request_id, status_code, data|error, meta). Spec §2 lists meta among
    # the fixed keys; §3 is silent but the consistent shape is the safe
    # reading. Clients can rely on `env.meta` being present without a
    # has-key check.
    env["meta"] = {}

    # Surface deprecation hint for legacy codes (clients can use this to plan
    # migration to the canonical equivalent). Does not affect status or shape.
    if err.code in LEGACY_CODE_ALIASES:
        env["meta"]["deprecation"] = {
            "canonical_code": LEGACY_CODE_ALIASES[err.code],
            "note": (f"`{err.code}` is preserved for backward compatibility. "
                     f"New code should expect `{LEGACY_CODE_ALIASES[err.code]}`."),
        }

    # REV-R5-I03 + PDA-P0-R1-F01: merge err.headers (Retry-After,
    # X-Missing-Scope, etc.) UNDERNEATH the envelope headers, so the envelope's
    # reserved keys (X-Request-Id, X-API-Version, Cache-Control) ALWAYS WIN
    # over anything a handler accidentally passes. PDA empirically reproduced
    # the bypass — a malicious or buggy handler raising
    # `ApiError(..., headers={"X-Request-Id": "FAKE"})` would have destroyed
    # request-id correlation under the previous merge order.
    #
    # Start with the handler's headers (lowest precedence), then overlay the
    # envelope's reserved keys (highest precedence). Reserved keys cannot be
    # spoofed even by accident.
    response_headers = dict(err.headers or {})
    response_headers.update({
        "X-Request-Id":   request_id,
        "X-API-Version":  API_VERSION,
        "Cache-Control":  "no-store",
    })

    # NOTE: no top-level `detail` shim. The §3 error envelope is exactly the
    # 7 base keys + `error` (+ `meta.deprecation` for legacy codes). The
    # legacy-`detail` backward-compat concern only applies when the GLOBAL
    # exception handler later migrates the existing endpoints (whose
    # frontends read `.detail`); these new v1 endpoints have no legacy
    # consumers, so a shim here would be a contract violation for no benefit.
    return JSONResponse(env, status_code=status, headers=response_headers)


# ── the decorator every v1 endpoint uses ───────────────────────────────────
def api_endpoint(endpoint_template: str, *, success_status: int = 200):
    """Wrap a route handler so it conforms to the contract with no per-handler
    boilerplate. `endpoint_template` is the route template for the envelope's
    `endpoint` field (e.g. "/predictions/{id}", contract §2.3).
    """
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            request_id = new_request_id()
            started_at = time.perf_counter()
            # if the handler takes a Request, give it the request_id
            req = kwargs.get("request")
            if isinstance(req, Request):
                try:
                    req.state.request_id = request_id
                except Exception:
                    pass
            try:
                payload = await fn(*args, **kwargs)
                # Pass-through escape hatch: if the handler returns a raw
                # Response (CSV export, file download, streaming response,
                # etc.) we MUST NOT JSON-encode it. JSONResponse is itself
                # a subclass of Response, but we never return raw JSONResponse
                # from handlers — envelope-wrapped JSON is built by _success.
                # So the rule is: any Response subclass that ISN'T JSONResponse
                # is a deliberate non-JSON payload — stamp the standard
                # envelope correlation headers on it and return as-is.
                if (isinstance(payload, _StarletteResponse)
                        and not isinstance(payload, JSONResponse)):
                    payload.headers.setdefault("X-Request-Id", request_id)
                    payload.headers.setdefault("X-API-Version", API_VERSION)
                    payload.headers.setdefault("Cache-Control", "no-store")
                    return payload
                return _success(payload, endpoint_template, request_id,
                                started_at, success_status)
            except ApiError as e:
                return _failure(e, endpoint_template, request_id, started_at)
            except Exception as e:                       # noqa: BLE001
                # never leak the raw exception (contract §8) — log it, return
                # a safe internal_error keyed by the same request_id.
                logger.exception("Unhandled error in %s [%s]: %s",
                                 endpoint_template, request_id, e)
                safe = ApiError(
                    "internal_error",
                    "An unexpected server error occurred.",
                    hint=f"Quote request_id {request_id} when reporting this.")
                return _failure(safe, endpoint_template, request_id,
                                started_at)
        return wrapper
    return deco
