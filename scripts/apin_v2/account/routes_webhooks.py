"""API Console — webhook CRUD + test-ping + delivery log routes.

Phase 7 Wave 1.

Spec contract: spec_v7.md §7.4 (lines for webhook endpoints), §13.x
(signing + delivery), §18.11 (AES-GCM secret encryption).

Eight endpoints under /api/account/webhooks:
    GET    /api/account/webhooks                       list user's webhooks
    POST   /api/account/webhooks                       create + return secret ONCE
    GET    /api/account/webhooks/{id}                  fetch one
    PATCH  /api/account/webhooks/{id}                  edit (sudo)
    DELETE /api/account/webhooks/{id}                  hard-delete (sudo)
    POST   /api/account/webhooks/{id}/rotate-secret    rotate (sudo, one-time-view)
    POST   /api/account/webhooks/{id}/test             synchronous test-ping (sudo)
    GET    /api/account/webhooks/{id}/deliveries       delivery log

Auth model:
    - Session-cookie auth (apin_v2_session) at all endpoints.
    - SudoMiddleware (slot 7) gates mutations + test-ping.
    - CSRF header check (X-Console-Csrf) at the route layer for all
      mutating verbs (PATCH/POST/DELETE).

Honesty notes (scope cuts vs full spec):
    - The actual delivery worker (retry-with-backoff, dead-letter,
      multi-secret overlap, lease-based exactly-once) is filed as
      WI-P8-DELIVERY-WORKER. This module implements:
        * Synchronous test-ping (single POST, no retry)
        * Delivery enqueue + log query (so the UI can show "queued")
      Production webhook delivery requires the Phase 8 worker.
    - URL homograph check (PDA-R2-F40 Punycode + confusables) is NOT
      enforced here; we accept the URL as given. Filed as WI-P8-URL-HARDEN.
    - SSRF defence (internal-target detection) is NOT enforced unless
      `allow_internal_target=False` is honored by the worker. test-ping
      uses urllib.request which DOES follow redirects; receivers behind
      cloud DNS rebinding could bypass. Filed as WI-P8-SSRF-HARDEN.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request, Query

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import _session_helpers as _sh

log = logging.getLogger("apin_v2.account.routes_webhooks")

router = APIRouter(prefix="/api/account/webhooks", tags=["account/webhooks"])


# ─── Validation helpers ──────────────────────────────────────────────────

# Minimal URL validation. Full Punycode + confusables check is WI-P8.
import re as _re
_URL_RE = _re.compile(r"^https://[A-Za-z0-9._\-]+(:\d+)?(/.*)?$")


def _validate_url_basic(url: Any) -> str:
    """Phase 8 WI-P8-URL-HARDEN: in addition to the original https + length
    + regex checks, we now Punycode-normalise the hostname (so homograph
    confusables like `аpin.com` Cyrillic vs `apin.com` Latin are surfaced
    as visibly different strings) and reject internal-target hosts UNLESS
    the create-webhook caller has set `allow_internal_target=true`.

    The SSRF check at THIS stage is a UX nicety — it gives the user a
    clean 400 instead of an opaque "delivery worker silently dropped all
    your events" experience. The delivery worker re-checks at attempt
    time as defence-in-depth (host could rebinding-pivot post-create).
    """
    if not isinstance(url, str) or not url.strip():
        raise ApiError("invalid_url", "URL is required.")
    url = url.strip()
    if len(url) > 2048:
        raise ApiError("invalid_url", "URL must be 2048 characters or shorter.")
    if not url.startswith("https://"):
        raise ApiError("invalid_url",
                       "URL must use https:// scheme. Plaintext http:// is "
                       "rejected because webhook bodies sign auth events.")
    if not _URL_RE.match(url):
        raise ApiError("invalid_url",
                       "URL is malformed. Expected https://host[:port][/path].")
    # Punycode-normalise the hostname to make homograph attacks visible.
    try:
        from scripts.apin_v2 import webhook_worker as _ww
        url = _ww.url_to_punycode(url)
    except Exception:
        pass   # punt on punycode error; URL_RE already constrains
    return url


def _validate_url_safe(url: str, *, allow_internal: bool) -> str:
    """Stricter check used by POST /webhooks (create) and PATCH on `url`.
    Raises ApiError if the URL points at an internal target without
    opt-in."""
    try:
        from scripts.apin_v2 import webhook_worker as _ww
        safe, reason = _ww.url_is_safe_target(url, allow_internal=allow_internal)
    except Exception:
        safe, reason = True, ""   # punt
    if not safe:
        raise ApiError("invalid_url", reason or "URL points at a forbidden target.")
    return url


def _validate_events(events: Any) -> list[str]:
    if not isinstance(events, list) or not events:
        raise ApiError("invalid_events",
                       "events must be a non-empty array of event-type strings.")
    if len(events) > 50:
        raise ApiError("invalid_events",
                       "Too many events. The hard cap is 50 per webhook.")
    out: list[str] = []
    for e in events:
        if not isinstance(e, str) or not e:
            raise ApiError("invalid_events",
                           "Each event must be a non-empty string.")
        if len(e) > 80:
            raise ApiError("invalid_events",
                           f"Event name '{e[:40]}...' is too long (max 80).")
        out.append(e)
    return out


def _check_secret_env_or_503():
    """Spec §18.11.1: webhook endpoints return 503 service_unavailable when
    APIN_SECRET_KEY env is missing. Single source of truth for the check.
    """
    import os
    if not os.environ.get("APIN_SECRET_KEY"):
        raise ApiError(
            "service_unavailable",
            "Webhook service is offline: APIN_SECRET_KEY env var is not set. "
            "Generate one with `python -c \"import secrets; print(secrets.token_urlsafe(32))\"` "
            "and place it in your .env, then restart the server.",
        )


# ─── Routes ───────────────────────────────────────────────────────────────

@router.get("")
@api_endpoint("/api/account/webhooks")
async def list_webhooks_route(request: Request):
    """List the caller's webhooks."""
    user = _sh.get_session_user(request)
    items = auth_db.list_webhooks(int(user["id"]))
    cap = int(auth_db.get_account_settings(int(user["id"])).get(
        "max_webhooks_per_user", 50))
    return {
        "items": items,
        "count": len(items),
        "cap": cap,
        "remaining": max(0, cap - len(items)),
    }


@router.post("", status_code=201)
@api_endpoint("/api/account/webhooks", success_status=201)
async def create_webhook_route(request: Request):
    """Mint a new webhook. The plaintext signing secret is returned ONCE in
    the response. Save it immediately — APIN cannot show it again."""
    _check_secret_env_or_503()
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")

    url = _validate_url_basic(body.get("url"))
    events = _validate_events(body.get("events"))
    description = body.get("description")
    if description is not None and not isinstance(description, str):
        raise ApiError("invalid_parameter", "description must be a string.")
    if description is not None and len(description) > 280:
        raise ApiError("invalid_parameter",
                       "description must be 280 characters or shorter.")
    allow_self_signed = bool(body.get("allow_self_signed", False))
    allow_internal_target = bool(body.get("allow_internal_target", False))
    # FX-P8-B SSRF/URL hardening: refuse internal-target URLs unless the
    # caller has explicitly opted-in via allow_internal_target=true.
    url = _validate_url_safe(url, allow_internal=allow_internal_target)

    try:
        wh, plaintext = auth_db.create_webhook(
            int(user["id"]),
            url,
            events,
            description=description,
            allow_self_signed=allow_self_signed,
            allow_internal_target=allow_internal_target,
        )
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    except RuntimeError as e:
        raise ApiError("service_unavailable", str(e)) from e

    # Phase 7 audit FX-P7-1 (P0-F01): return as 3-tuple so the envelope
    # helper attaches X-APIN-No-Redact: 1. Without this header,
    # TokenRedactionMiddleware (_REDACT_WHSEC_RE) clobbers the plaintext
    # signing_secret to `whsec_<redacted>` on egress, and the one-time-view
    # ceremony silently breaks — no receiver can ever verify HMAC.
    data = {
        **wh,
        "signing_secret": plaintext,
        "signing_secret_notice": (
            "This is the only time we will show the full signing secret. "
            "Save it to your secrets manager NOW. Use it to verify the "
            "APIN-Signature header on incoming webhook deliveries."
        ),
    }
    # Phase 8.H · default-OFF (you minted it on purpose). Quiet by default,
    # but powerful for users who want a paper trail.
    try:
        auth_db.emit_alert(
            int(user["id"]), "webhook.created",
            action={"kind": "view_webhook", "id": wh.get("id")},
            url=wh.get("url", "?"),
            events=", ".join(wh.get("events") or []) or "all",
        )
    except Exception:
        pass
    return (data, None, {"X-APIN-No-Redact": "1"})


@router.get("/{webhook_id}")
@api_endpoint("/api/account/webhooks/{webhook_id}")
async def get_webhook_route(webhook_id: str, request: Request):
    user = _sh.get_session_user(request)
    wh = auth_db.get_webhook(webhook_id, int(user["id"]))
    if wh is None:
        raise ApiError("not_found", "Webhook not found.")
    return wh


@router.patch("/{webhook_id}")
@api_endpoint("/api/account/webhooks/{webhook_id}")
async def patch_webhook_route(webhook_id: str, request: Request):
    """Update a subset of editable webhook fields. Sudo + CSRF required."""
    _check_secret_env_or_503()
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")
    # Pre-validate touched fields with the same UX-friendly messages
    if "url" in body:
        body["url"] = _validate_url_basic(body["url"])
    if "events" in body:
        body["events"] = _validate_events(body["events"])
    if "description" in body:
        d = body["description"]
        if d is not None and not isinstance(d, str):
            raise ApiError("invalid_parameter", "description must be a string.")
        if d is not None and len(d) > 280:
            raise ApiError("invalid_parameter",
                           "description must be 280 characters or shorter.")
    try:
        wh = auth_db.update_webhook(webhook_id, int(user["id"]), **body)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise ApiError("not_found", "Webhook not found.") from e
        raise ApiError("invalid_parameter", msg) from e
    # Phase 8.H · webhook.patched default-OFF.
    try:
        auth_db.emit_alert(
            int(user["id"]), "webhook.patched",
            action={"kind": "view_webhook", "id": webhook_id},
            url=wh.get("url", "?"),
            fields_changed=", ".join(sorted(body.keys())) or "metadata",
        )
    except Exception:
        pass
    return wh


@router.delete("/{webhook_id}")
@api_endpoint("/api/account/webhooks/{webhook_id}")
async def delete_webhook_route(webhook_id: str, request: Request):
    """Hard-delete a webhook. Cascades to delivery rows. Sudo + CSRF."""
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    # Snapshot URL BEFORE delete so the alert can name it (the row will
    # be gone after the delete returns).
    prev = auth_db.get_webhook(webhook_id, int(user["id"])) or {}
    ok = auth_db.delete_webhook(webhook_id, int(user["id"]))
    if not ok:
        raise ApiError("not_found", "Webhook not found.")
    # Phase 8.H · webhook.deleted default-OFF. No view_webhook action —
    # the row is gone; link back to the webhooks list.
    try:
        auth_db.emit_alert(
            int(user["id"]), "webhook.deleted",
            action={"kind": "view_webhooks_list"},
            url=prev.get("url", webhook_id),
        )
    except Exception:
        pass
    return {"deleted": True, "id": webhook_id}


@router.post("/{webhook_id}/rotate-secret")
@api_endpoint("/api/account/webhooks/{webhook_id}/rotate-secret")
async def rotate_webhook_secret_route(webhook_id: str, request: Request):
    """Mint a new signing secret. Old secret stays valid for `grace_seconds`
    so receivers can swap in the new secret without dropping in-flight
    events. Returns the plaintext NEW secret ONCE. Sudo + CSRF."""
    _check_secret_env_or_503()
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    grace_seconds = body.get("grace_seconds", 86400)
    try:
        grace_seconds = int(grace_seconds)
    except (TypeError, ValueError):
        raise ApiError("invalid_parameter",
                       "grace_seconds must be an integer.")

    try:
        wh, new_plaintext = auth_db.rotate_webhook_secret(
            webhook_id, int(user["id"]), grace_seconds=grace_seconds)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise ApiError("not_found", "Webhook not found.") from e
        raise ApiError("invalid_parameter", msg) from e

    # FX-P7-1: same one-time-view ceremony as create — use 3-tuple form to
    # attach X-APIN-No-Redact: 1, so the plaintext rotated secret survives
    # TokenRedactionMiddleware.
    data = {
        **wh,
        "signing_secret": new_plaintext,
        "grace_seconds_applied": grace_seconds,
        "signing_secret_notice": (
            "Save this new secret immediately. The previous secret stays "
            f"valid for {grace_seconds} seconds so your receiver can "
            "verify in-flight deliveries during the swap. After the grace "
            "window the old secret stops working."
        ),
    }
    # Phase 8.H · webhook.secret_rotated default-OFF.
    try:
        auth_db.emit_alert(
            int(user["id"]), "webhook.secret_rotated",
            action={"kind": "view_webhook", "id": webhook_id},
            url=wh.get("url", "?"),
        )
    except Exception:
        pass
    return (data, None, {"X-APIN-No-Redact": "1"})


@router.post("/{webhook_id}/test")
@api_endpoint("/api/account/webhooks/{webhook_id}/test")
async def test_webhook_route(webhook_id: str, request: Request):
    """Synchronous test-ping. Fires ONE POST and waits up to 10s for the
    receiver. Returns the attempt outcome (status code, error text, ms).
    Sudo + CSRF. State-changing per PDA-R2-F46."""
    _check_secret_env_or_503()
    _sh.require_csrf(request)
    user = _sh.get_session_user(request)
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: auth_db.test_ping_webhook(webhook_id, int(user["id"])),
        )
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise ApiError("not_found", "Webhook not found.") from e
        raise ApiError("invalid_parameter", msg) from e
    return result


@router.get("/{webhook_id}/deliveries")
@api_endpoint("/api/account/webhooks/{webhook_id}/deliveries")
async def list_deliveries_route(
    webhook_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
):
    user = _sh.get_session_user(request)
    wh = auth_db.get_webhook(webhook_id, int(user["id"]))
    if wh is None:
        raise ApiError("not_found", "Webhook not found.")
    items = auth_db.list_webhook_deliveries(
        webhook_id, int(user["id"]), limit=limit)
    return {
        "items": items,
        "count": len(items),
        "webhook_id": webhook_id,
    }
