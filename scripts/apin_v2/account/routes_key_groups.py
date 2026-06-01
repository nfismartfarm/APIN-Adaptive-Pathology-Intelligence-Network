"""API Console — API key GROUP routes (Phase 2).

Endpoints under `/api/account/key-groups`:
    GET    /api/account/key-groups                  list (with member counts)
    POST   /api/account/key-groups                  create  {name, scopes}
    GET    /api/account/key-groups/{group_id}       fetch one (+ members)
    PATCH  /api/account/key-groups/{group_id}       edit    {name?, scopes?}
    DELETE /api/account/key-groups/{group_id}       delete  {member_policy, keep?}
    POST   /api/account/key-groups/{group_id}/members   assign {public_id, role, ceiling?}
    DELETE /api/account/key-groups/{group_id}/members/{public_id}  remove
    PATCH  /api/account/key-groups/{group_id}/members/{public_id}  role {role, ceiling?}

Auth: session-cookie. Mutations are gated by Sudo + central CSRF middleware
(same as the keys routes); handlers also call `_require_csrf`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, Request

from scripts.apin_v2.api_envelope import ApiError, api_endpoint
from scripts.apin_v2 import auth_db
from scripts.apin_v2.account import _session_helpers as _sh

log = logging.getLogger("apin_v2.account.routes_key_groups")

router = APIRouter(prefix="/api/account/key-groups", tags=["account/key-groups"])

_get_session_user = _sh.get_session_user
_require_csrf = _sh.require_csrf

_ROLES = ("locked", "special")
_DELETE_POLICIES = ("ungroup", "delete_all", "keep_special", "choose")


def _validate_scopes(raw) -> list:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        raise ApiError("invalid_parameter", "scopes must be an array of strings.")
    # de-dup, preserve order
    seen, out = set(), []
    for s in raw:
        s = s.strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out


def _validate_name(raw) -> str:
    if not isinstance(raw, str):
        raise ApiError("invalid_parameter", "name must be a string.")
    name = raw.strip()
    if not (1 <= len(name) <= 80):
        raise ApiError("invalid_parameter", "name must be 1-80 characters.")
    return name


@router.get("")
@api_endpoint("/api/account/key-groups")
async def list_groups(request: Request):
    user = _get_session_user(request)
    return {"groups": auth_db.list_key_groups(user_id=int(user["id"]))}


@router.post("", status_code=201)
@api_endpoint("/api/account/key-groups")
async def create_group(request: Request):
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")
    name = _validate_name(body.get("name"))
    scopes = _validate_scopes(body.get("scopes"))
    try:
        g = auth_db.create_key_group(user_id=int(user["id"]), name=name, scopes=scopes)
    except auth_db.DuplicateKeyNameError as e:
        raise ApiError("duplicate_name", str(e), details={"field": "name"}) from e
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    auth_db.emit_alert(int(user["id"]), "group.created",
                       group_name=name, scope_count=len(scopes))
    return g


@router.get("/{group_id}")
@api_endpoint("/api/account/key-groups/{group_id}")
async def get_group(request: Request, group_id: int):
    user = _get_session_user(request)
    g = auth_db.get_key_group(user_id=int(user["id"]), group_id=int(group_id))
    if g is None:
        raise ApiError("not_found", f"group {group_id} not found.")
    return g


@router.patch("/{group_id}")
@api_endpoint("/api/account/key-groups/{group_id}")
async def patch_group(request: Request, group_id: int):
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError("invalid_parameter", "Request body must be a JSON object.")
    name = _validate_name(body["name"]) if "name" in body else None
    scopes = _validate_scopes(body["scopes"]) if "scopes" in body else None
    if name is None and scopes is None:
        raise ApiError("invalid_parameter", "Supply at least one of: name, scopes.")
    try:
        g = auth_db.patch_key_group(user_id=int(user["id"]), group_id=int(group_id),
                                    name=name, scopes=scopes)
    except auth_db.DuplicateKeyNameError as e:
        raise ApiError("duplicate_name", str(e), details={"field": "name"}) from e
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    if g is None:
        raise ApiError("not_found", f"group {group_id} not found.")
    _change = "renamed" if name is not None else "permissions changed"
    if name is not None and scopes is not None:
        _change = "renamed + permissions changed"
    auth_db.emit_alert(int(user["id"]), "group.updated",
                       group_name=g.get("name", "?"), change=_change)
    return g


@router.delete("/{group_id}")
@api_endpoint("/api/account/key-groups/{group_id}")
async def delete_group(request: Request, group_id: int):
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    policy = (body or {}).get("member_policy", "ungroup")
    if policy not in _DELETE_POLICIES:
        raise ApiError("invalid_parameter",
                       f"member_policy must be one of: {', '.join(_DELETE_POLICIES)}.")
    keep = (body or {}).get("keep_public_ids") or []
    if not isinstance(keep, list):
        raise ApiError("invalid_parameter", "keep_public_ids must be an array.")
    # capture the name before the row is gone, for the alert
    _pre = auth_db.get_key_group(user_id=int(user["id"]), group_id=int(group_id))
    _gname = _pre.get("name", "?") if _pre else "?"
    res = auth_db.delete_key_group(user_id=int(user["id"]), group_id=int(group_id),
                                   member_policy=policy, keep_public_ids=keep)
    if res is None:
        raise ApiError("not_found", f"group {group_id} not found.")
    auth_db.emit_alert(int(user["id"]), "group.deleted",
                       group_name=_gname,
                       members_kept=res.get("members_kept", 0),
                       members_deleted=res.get("members_deleted", 0))
    return res


@router.post("/{group_id}/members", status_code=201)
@api_endpoint("/api/account/key-groups/{group_id}/members")
async def add_member(request: Request, group_id: int):
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    pid = (body or {}).get("public_id")
    if not isinstance(pid, str) or not pid:
        raise ApiError("invalid_parameter", "public_id is required.")
    role = (body or {}).get("role", "locked")
    if role not in _ROLES:
        raise ApiError("invalid_parameter", "role must be 'locked' or 'special'.")
    ceiling = body.get("scope_ceiling")
    if ceiling is not None:
        ceiling = _validate_scopes(ceiling)
    try:
        k = auth_db.assign_key_to_group(user_id=int(user["id"]), public_id=pid,
                                        group_id=int(group_id), role=role,
                                        scope_ceiling=ceiling)
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    if k is None:
        raise ApiError("not_found", "key or group not found.")
    return k


@router.patch("/{group_id}/members/{public_id}")
@api_endpoint("/api/account/key-groups/{group_id}/members/{public_id}")
async def set_member_role(request: Request, group_id: int, public_id: str):
    _require_csrf(request)
    user = _get_session_user(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError("invalid_parameter", "Request body must be valid JSON.")
    role = (body or {}).get("role")
    if role not in _ROLES:
        raise ApiError("invalid_parameter", "role must be 'locked' or 'special'.")
    ceiling = body.get("scope_ceiling")
    if ceiling is not None:
        ceiling = _validate_scopes(ceiling)
    try:
        k = auth_db.set_key_group_role(user_id=int(user["id"]), public_id=public_id,
                                       role=role, scope_ceiling=ceiling)
    except ValueError as e:
        raise ApiError("invalid_parameter", str(e)) from e
    if k is None:
        raise ApiError("not_found", "key not found.")
    return k


@router.delete("/{group_id}/members/{public_id}")
@api_endpoint("/api/account/key-groups/{group_id}/members/{public_id}")
async def remove_member(request: Request, group_id: int, public_id: str):
    _require_csrf(request)
    user = _get_session_user(request)
    k = auth_db.remove_key_from_group(user_id=int(user["id"]), public_id=public_id)
    if k is None:
        raise ApiError("not_found", "key not found.")
    return k
