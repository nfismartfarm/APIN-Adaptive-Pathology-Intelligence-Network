"""APIN v2 — FastAPI routes for user accounts.

Endpoints:
    POST /auth/signup          → create account, set session cookie
    POST /auth/login           → verify credentials, set session cookie
    POST /auth/logout          → revoke session, clear cookie
    GET  /auth/me              → current user (or 401)
    GET  /auth/check           → uniqueness lookup (replaces Day-1 stub)
    GET  /auth/next-accession  → next available account id (replaces Day-1 stub)

Cookie: HttpOnly, SameSite=Lax, Secure when not localhost, 30-day expiry.
Rate limit: 5 login attempts per IP per minute. Cleared on success.
"""
from __future__ import annotations

import re
import random
import logging
from typing import Optional

from fastapi import APIRouter, Request, Response, HTTPException, Query
from pydantic import BaseModel, Field, EmailStr, field_validator

from scripts.apin_v2 import auth_db

logger = logging.getLogger("apin_v2.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "apin_v2_session"           # HttpOnly real session token
MARKER_COOKIE = "apin_v2_signed_in"       # non-HttpOnly presence flag (no secrets)
GUEST_COOKIE = "apin_v2_guest"            # HttpOnly guest-session token
COOKIE_MAX_AGE = 30 * 24 * 3600           # 30 days
GUEST_COOKIE_MAX_AGE = 7 * 24 * 3600      # 7 days


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
PASSWORD_MIN_LEN = 12


def _validate_password(pw: str) -> Optional[str]:
    """Return error string if invalid, None if OK. Matches frontend rules."""
    if len(pw) < PASSWORD_MIN_LEN:
        return f"password must be at least {PASSWORD_MIN_LEN} characters"
    if not re.search(r"[A-Z]", pw): return "password needs an uppercase letter"
    if not re.search(r"[a-z]", pw): return "password needs a lowercase letter"
    if not re.search(r"[0-9]", pw): return "password needs a number"
    if not re.search(r"[^A-Za-z0-9]", pw): return "password needs a special character"
    if re.search(r"\s", pw): return "password cannot contain whitespace"
    return None


def _normalize_mobile(raw: str) -> Optional[str]:
    """Strip non-digits, return canonical E.164 (+91XXXXXXXXXX) or None if invalid."""
    digits = re.sub(r"\D", "", raw or "")
    if not digits:
        return None
    if digits.startswith("91"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    if not re.match(r"^[6-9]", digits):
        return None
    return "+91" + digits


class SignupPayload(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    display_name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=3, max_length=255)
    password: str
    mobile: str

    @field_validator("username")
    @classmethod
    def _v_username(cls, v):
        v = v.strip()
        if not USERNAME_RE.match(v):
            raise ValueError("username must be 3-32 chars, letters/numbers/_")
        return v

    @field_validator("email")
    @classmethod
    def _v_email(cls, v):
        v = v.strip()
        if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", v):
            raise ValueError("invalid email format")
        return v.lower()


class LoginPayload(BaseModel):
    handle: str = Field(..., min_length=1, max_length=255)
    password: str


# ─── Cookie helper ────────────────────────────────────────────────────────────

def _set_session_cookie(response: Response, token: str, *, request: Request):
    """Set the session cookie with safe defaults, plus a non-HttpOnly marker
    cookie so the frontend can detect 'maybe signed in' without calling
    /auth/me (and creating console-log noise) on every page load."""
    host = request.url.hostname or ""
    is_local = host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    # The app is embedded in an iframe on huggingface.co/spaces/... — a
    # cross-site context. A SameSite=Lax cookie is not sent there, so the
    # session never sticks and auth loops. SameSite=None (which requires
    # Secure) lets the cookie work inside the Space iframe. On localhost
    # there is no iframe and no HTTPS, so keep Lax + non-secure.
    samesite = "lax" if is_local else "none"
    secure = not is_local
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,        # secret, never readable from JS
        samesite=samesite,
        secure=secure,
        path="/",
    )
    response.set_cookie(
        key=MARKER_COOKIE,
        value="1",
        max_age=COOKIE_MAX_AGE,
        httponly=False,       # readable from JS, but holds no secret
        samesite=samesite,
        secure=secure,
        path="/",
    )


def _clear_session_cookie(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    response.delete_cookie(MARKER_COOKIE, path="/")


def _clear_guest_cookie(response: Response):
    """Drop the guest-session cookie.  Called when a guest upgrades to a
    real account (signup/login) so a later logout returns cleanly to the
    anonymous state instead of resurfacing a stale guest session, and
    called on logout for symmetry."""
    response.delete_cookie(GUEST_COOKIE, path="/")


def _set_guest_cookie(response: Response, token: str, *, request: Request):
    """Set the HttpOnly guest-session cookie. Separate from the user
    session cookie so a guest upgrading to a real account is clean."""
    host = request.url.hostname or ""
    is_local = host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    # SameSite=None so the guest session also works inside the Space
    # iframe (see _set_session_cookie for the full rationale).
    samesite = "lax" if is_local else "none"
    secure = not is_local
    response.set_cookie(
        key=GUEST_COOKIE,
        value=token,
        max_age=GUEST_COOKIE_MAX_AGE,
        httponly=True,
        samesite=samesite,
        secure=secure,
        path="/",
    )


# ─── Public endpoints ─────────────────────────────────────────────────────────

@router.get("/check")
async def check_uniqueness(
    field: str = Query(..., regex="^(username|email|display_name)$"),
    value: str = Query(..., min_length=1, max_length=255),
):
    """Live uniqueness probe. Returns {available, reason?, suggestions?}."""
    v = (value or "").strip()
    if not v:
        return {"available": False, "reason": "empty"}

    # Format guard mirrors signup-time validation (defence in depth)
    if field == "username":
        if not USERNAME_RE.match(v):
            return {"available": False, "reason": "format"}
    if field == "display_name":
        if len(v) < 2 or len(v) > 80:
            return {"available": False, "reason": "length"}
    if field == "email":
        if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", v):
            return {"available": False, "reason": "format"}

    if auth_db.is_taken(field, v):
        # Generate gentle suggestions
        base = v.lower().strip()
        sugg = []
        if field == "username":
            n = random.randint(2, 99)
            sugg = [f"{base}_kerala", f"{base}{n}", f"{base}_lab"]
            # Filter out any that themselves are taken
            sugg = [s for s in sugg if not auth_db.is_taken("username", s)][:3]
        elif field == "display_name":
            sugg = [f"{v} (Kerala)", f"{v} {random.randint(2, 99)}"]
            sugg = [s for s in sugg if not auth_db.is_taken("display_name", s)][:2]
        return {"available": False, "reason": "taken", "suggestions": sugg}

    return {"available": True}


@router.get("/next-accession")
async def next_accession_endpoint():
    """Return the next-to-be-assigned user id."""
    return {"accession": auth_db.next_accession()}


# ─── Authentication endpoints ─────────────────────────────────────────────────

@router.post("/signup")
async def signup(payload: SignupPayload, request: Request, response: Response):
    # Password rules (frontend already checked, but server is authoritative)
    pw_err = _validate_password(payload.password)
    if pw_err:
        raise HTTPException(status_code=400, detail=pw_err)
    # Mobile
    mob = _normalize_mobile(payload.mobile)
    if not mob:
        raise HTTPException(status_code=400,
                            detail="mobile must be a valid Indian number (+91, 10 digits, starts 6-9)")
    # Display name format
    if not payload.display_name.strip():
        raise HTTPException(status_code=400, detail="display_name cannot be blank")

    # Create
    try:
        user = auth_db.create_user(
            username=payload.username,
            display_name=payload.display_name.strip(),
            email=payload.email,
            password=payload.password,
            mobile_e164=mob,
        )
    except ValueError as e:
        # 'taken:<field>'
        msg = str(e)
        if msg.startswith("taken:"):
            field = msg.split(":", 1)[1]
            raise HTTPException(status_code=409, detail=f"{field} already taken")
        raise HTTPException(status_code=400, detail=msg)

    # Issue session
    ua = request.headers.get("user-agent", "")[:255]
    ip = request.client.host if request.client else None
    token = auth_db.create_session(user["id"], user_agent=ua, ip_addr=ip)
    _set_session_cookie(response, token, request=request)
    _clear_guest_cookie(response)   # guest → real account: retire the guest session
    auth_db.audit("signup", user_id=user["id"], ip_addr=ip,
                  detail={"username": user["username"]})

    return {
        "ok": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "email": user["email"],
            "pressed_leaf_seed": user["pressed_leaf_seed"],
            "created_at": user["created_at"],
        },
        "redirect": "/",
    }


@router.post("/login")
async def login(payload: LoginPayload, request: Request, response: Response):
    ip = request.client.host if request.client else "unknown"
    if not auth_db.check_rate_limit(ip):
        raise HTTPException(status_code=429,
                            detail="too many attempts; wait a minute and try again")

    user = auth_db.get_user_by_handle(payload.handle)
    # Constant-ish-time mismatch: if user not found, still hash the input
    # to mitigate username enumeration via response time.
    if user is None:
        auth_db.verify_password(payload.password,
                                "$argon2id$v=19$m=65536,t=3,p=4$"
                                "AAAAAAAAAAAAAAAAAAAAAA$"
                                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        auth_db.audit("login_failed", ip_addr=ip,
                      detail={"handle": payload.handle, "reason": "no_user"})
        raise HTTPException(status_code=401, detail="invalid credentials")

    if not auth_db.verify_password(payload.password, user["password_hash"]):
        auth_db.audit("login_failed", user_id=user["id"], ip_addr=ip,
                      detail={"reason": "bad_password"})
        raise HTTPException(status_code=401, detail="invalid credentials")

    # Success
    auth_db.clear_rate_limit(ip)
    auth_db.touch_last_seen(user["id"])
    ua = request.headers.get("user-agent", "")[:255]
    token = auth_db.create_session(user["id"], user_agent=ua, ip_addr=ip)
    _set_session_cookie(response, token, request=request)
    _clear_guest_cookie(response)   # guest → real account: retire the guest session
    auth_db.audit("login_success", user_id=user["id"], ip_addr=ip)

    return {
        "ok": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "email": user["email"],
            "pressed_leaf_seed": user["pressed_leaf_seed"],
        },
        "redirect": "/",
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        auth_db.revoke_session(token)
        user = auth_db.get_session_user(token)  # will be None after revoke
        auth_db.audit("logout", user_id=user["id"] if user else None,
                      ip_addr=request.client.host if request.client else None)
    _clear_session_cookie(response)
    _clear_guest_cookie(response)   # symmetry — logout returns to a clean anonymous state
    return {"ok": True}


@router.post("/guest")
async def guest(request: Request, response: Response):
    """Start (or resume) an anonymous guest session.

    A guest gets GUEST_INFERENCE_LIMIT free inference checks and no
    dashboard/account access. If the caller is already a signed-in user,
    this is a no-op that just reports their authenticated state. If they
    already hold a non-expired guest cookie, that session is reused so
    refreshing the page doesn't reset the quota.
    """
    # Already a real user? Nothing to do.
    utoken = request.cookies.get(COOKIE_NAME)
    if utoken and auth_db.get_session_user(utoken):
        return {"ok": True, "mode": "user"}

    # Reuse an existing valid guest session if present.
    gtoken = request.cookies.get(GUEST_COOKIE)
    existing = auth_db.get_guest_session(gtoken) if gtoken else None
    if existing is not None:
        return {
            "ok": True, "mode": "guest",
            "remaining": existing["remaining"],
            "limit": auth_db.GUEST_INFERENCE_LIMIT,
        }

    # Fresh guest session.
    ua = request.headers.get("user-agent", "")[:255]
    ip = request.client.host if request.client else None
    token = auth_db.create_guest_session(user_agent=ua, ip_addr=ip)
    _set_guest_cookie(response, token, request=request)
    return {
        "ok": True, "mode": "guest",
        "remaining": auth_db.GUEST_INFERENCE_LIMIT,
        "limit": auth_db.GUEST_INFERENCE_LIMIT,
    }


@router.get("/state")
async def state(request: Request):
    """Single source of truth for the frontend's auth state. Always 200.

    Returns one of:
      {"mode": "user",      "user": {...}}
      {"mode": "guest",     "guest": {"remaining": N, "limit": M}}
      {"mode": "anonymous"}
    The inference page calls this once on load to decide whether to show
    the account chip, the guest counter, or nothing.
    """
    utoken = request.cookies.get(COOKIE_NAME)
    user = auth_db.get_session_user(utoken) if utoken else None
    if user is not None:
        return {
            "mode": "user",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "display_name": user["display_name"],
                "pressed_leaf_seed": user["pressed_leaf_seed"],
            },
        }
    gtoken = request.cookies.get(GUEST_COOKIE)
    guest_row = auth_db.get_guest_session(gtoken) if gtoken else None
    if guest_row is not None:
        return {
            "mode": "guest",
            "guest": {
                "remaining": guest_row["remaining"],
                "limit": auth_db.GUEST_INFERENCE_LIMIT,
                "exhausted": guest_row["exhausted"],
            },
        }
    return {"mode": "anonymous"}


@router.get("/me")
async def me(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    user = auth_db.get_session_user(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "email": user["email"],
        "pressed_leaf_seed": user["pressed_leaf_seed"],
        "created_at": user["created_at"],
        "last_seen_at": user["last_seen_at"],
        "predictions_count": auth_db.count_user_predictions(user["id"]),
    }


@router.get("/history")
async def history(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Return paginated prediction history for the current user."""
    token = request.cookies.get(COOKIE_NAME)
    user = auth_db.get_session_user(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    rows = auth_db.get_user_predictions(user["id"], limit=limit, offset=offset)
    total = auth_db.count_user_predictions(user["id"])
    return {
        "user_id": user["id"],
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": rows,
    }


@router.get("/history/{prediction_id}")
async def history_detail(prediction_id: int, request: Request):
    """Full response_json for one prediction, scoped to current user."""
    token = request.cookies.get(COOKIE_NAME)
    user = auth_db.get_session_user(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    row = auth_db.get_prediction_full(prediction_id, user_id=user["id"])
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    import json
    try:
        row["response"] = json.loads(row["response_json"])
    except Exception:
        row["response"] = None
    row.pop("response_json", None)
    return row


# ─── FastAPI helper used elsewhere ────────────────────────────────────────────

def get_current_user(request: Request) -> Optional[dict]:
    """Use as a dependency on protected routes:
        @app.get(...) async def x(user=Depends(get_current_user)): ...
    Returns None if not authenticated; protected routes should raise 401 manually
    or use a `require_user` wrapper.
    """
    token = request.cookies.get(COOKIE_NAME)
    return auth_db.get_session_user(token) if token else None


def attach(app):
    """Mount this router onto a FastAPI app."""
    app.include_router(router)
    logger.info("v2 /auth/* routes registered (real DB-backed)")
