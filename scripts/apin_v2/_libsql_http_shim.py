"""HTTP-only Turso client that bypasses libsql_client 0.3.1.

The official libsql_client 0.3.1 has a parsing bug where its `_send` method
sometimes receives a response that's not in the expected `{"result": ...}`
shape, then blindly does `response["result"]` and blows up with KeyError.
The Turso HTTP API itself works fine — both v1 and v2 endpoints return
correct responses for all our queries. The bug is purely client-side.

This module provides a drop-in replacement (`HttpClientSync`) that mirrors
just enough of libsql_client's `ClientSync` surface for our `_ShimConn`
to work: `.execute(sql, args)`, `.batch(stmts)`, and exception types.

Uses `requests` (sync, blocking) over the v2/pipeline endpoint. v2 returns
a richer response shape that handles batches, errors, and execute in one
HTTP call.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable, List, Optional, Tuple, Union

import requests


class LibsqlError(Exception):
    """Mirrors libsql_client.LibsqlError so callers don't need to change."""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        super().__init__(message)
        self.message = message
        self.code = code


# ─── Response normalization ────────────────────────────────────────────────

def _decode_value(v: dict) -> Any:
    """Turso protobuf-over-JSON value → Python scalar."""
    if v is None:
        return None
    t = v.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(v["value"])
    if t == "float":
        return float(v["value"])
    if t == "text":
        return v["value"]
    if t == "blob":
        import base64
        # Hrana returns blob bytes under `base64`; tolerate legacy `value` too.
        b64 = v.get("base64")
        if b64 is None:
            b64 = v.get("value")
        if b64 is None:
            return b""
        # Hrana emits UNPADDED base64 — re-pad to a multiple of 4 before decode.
        pad = (-len(b64)) % 4
        return base64.b64decode(b64 + ("=" * pad))
    return v.get("value")


class _Row:
    """Row that supports both index access and column-name access (sqlite3.Row-ish)."""

    __slots__ = ("_cols", "_values")

    def __init__(self, cols: List[str], values: List[Any]):
        self._cols = cols
        self._values = values

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        try:
            return self._values[self._cols.index(key)]
        except ValueError:
            raise KeyError(key)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return list(self._cols)

    def __repr__(self):
        return f"Row({dict(zip(self._cols, self._values))})"


class _ResultSet:
    """Mirrors libsql_client.ResultSet enough for our code path."""

    def __init__(self, cols: List[str], rows: List[_Row], last_insert_rowid: Optional[int],
                 rows_affected: int):
        self.columns = cols
        self.rows = rows
        self.last_insert_rowid = last_insert_rowid
        self.rows_affected = rows_affected

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _row_set_from_proto(result: dict) -> _ResultSet:
    cols = [c["name"] for c in result.get("cols", [])]
    rows = []
    for raw_row in result.get("rows", []):
        rows.append(_Row(cols, [_decode_value(v) for v in raw_row]))
    last_id = result.get("last_insert_rowid")
    rows.sort  # noop, but keeps the linter happy
    return _ResultSet(
        cols=cols,
        rows=rows,
        last_insert_rowid=int(last_id) if last_id is not None else None,
        rows_affected=int(result.get("affected_row_count") or 0),
    )


def _encode_arg(v: Any) -> dict:
    """Python scalar → Turso protobuf-over-JSON value."""
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, (bytes, bytearray, memoryview)):
        import base64
        # Hrana protocol expects the field name `base64` for blob values.
        return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode("ascii")}
    return {"type": "text", "value": str(v)}


def _stmt_to_proto(sql: str, args: Optional[Iterable] = None) -> dict:
    out: dict = {"sql": sql}
    if args:
        if isinstance(args, dict):
            out["named_args"] = [
                {"name": k, "value": _encode_arg(v)} for k, v in args.items()
            ]
        else:
            out["args"] = [_encode_arg(v) for v in args]
    return out


# ─── HTTP client ──────────────────────────────────────────────────────────

class HttpClientSync:
    """Sync Turso HTTP client. Drop-in for libsql_client.ClientSync for our
    `_ShimConn`'s narrow needs (execute + batch + close)."""

    def __init__(self, url: str, auth_token: str, *, timeout: float = 30.0):
        # libsql://X → https://X for the HTTP API
        if url.startswith("libsql://"):
            url = "https://" + url[len("libsql://"):]
        elif url.startswith("wss://"):
            url = "https://" + url[len("wss://"):]
        elif url.startswith("ws://"):
            url = "http://" + url[len("ws://"):]
        self._url = url.rstrip("/")
        self._token = auth_token
        self._timeout = timeout
        # One persistent session for keep-alive + connection pooling.
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "apin-libsql-http-shim/1.0",
        })

    def execute(self, sql, args=None) -> _ResultSet:
        # `sql` may be a Statement instance (libsql_client style); unwrap.
        if hasattr(sql, "sql"):
            actual_sql = sql.sql
            actual_args = getattr(sql, "args", None) or args
        else:
            actual_sql = sql
            actual_args = args
        body = {"requests": [
            {"type": "execute", "stmt": _stmt_to_proto(actual_sql, actual_args)},
        ]}
        result = self._post_pipeline(body)
        # result["results"][0]["response"]["result"] is our row set
        entry = result["results"][0]
        if entry.get("type") == "error":
            err = entry.get("error", {})
            raise LibsqlError(err.get("message", "unknown error"),
                              err.get("code", "UNKNOWN"))
        resp = entry["response"]
        if resp.get("type") == "error":
            err = resp.get("error", {})
            raise LibsqlError(err.get("message", "unknown error"),
                              err.get("code", "UNKNOWN"))
        return _row_set_from_proto(resp["result"])

    def batch(self, stmts: List[Union[str, Any]]) -> List[_ResultSet]:
        # Each stmt can be a string or have .sql/.args
        steps = []
        for s in stmts:
            if hasattr(s, "sql"):
                steps.append({"stmt": _stmt_to_proto(s.sql, getattr(s, "args", None))})
            elif isinstance(s, tuple) and len(s) == 2:
                steps.append({"stmt": _stmt_to_proto(s[0], s[1])})
            else:
                steps.append({"stmt": _stmt_to_proto(str(s))})
        body = {"requests": [{"type": "batch", "batch": {"steps": steps}}]}
        result = self._post_pipeline(body)
        entry = result["results"][0]
        if entry.get("type") == "error":
            err = entry.get("error", {})
            raise LibsqlError(err.get("message", "unknown error"),
                              err.get("code", "UNKNOWN"))
        resp = entry["response"]
        if resp.get("type") == "error":
            err = resp.get("error", {})
            raise LibsqlError(err.get("message", "unknown error"),
                              err.get("code", "UNKNOWN"))
        step_results = resp["result"]["step_results"]
        step_errors = resp["result"].get("step_errors") or [None] * len(step_results)
        out = []
        for sr, se in zip(step_results, step_errors):
            if se is not None:
                raise LibsqlError(
                    se.get("message", "batch step failed"),
                    se.get("code", "BATCH_STEP_FAILED"),
                )
            out.append(_row_set_from_proto(sr))
        return out

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass

    # ─── private ───────────────────────────────────────────────────────

    def _post_pipeline(self, body: dict, *, attempts: int = 5) -> dict:
        """POST /v2/pipeline with retry on transient errors."""
        last = None
        backoffs = (0.3, 0.8, 1.8, 3.0)
        for i in range(attempts):
            try:
                r = self._session.post(
                    self._url + "/v2/pipeline",
                    json=body, timeout=self._timeout,
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (502, 503, 504):
                    last = LibsqlError(f"transient HTTP {r.status_code}", "TRANSIENT")
                else:
                    # Authoritative error from Turso
                    try:
                        j = r.json()
                        raise LibsqlError(j.get("message", r.text), j.get("code", "HTTP_ERR"))
                    except (ValueError, json.JSONDecodeError):
                        raise LibsqlError(
                            f"HTTP {r.status_code}: {r.text[:200]}", "HTTP_ERR",
                        )
            except requests.exceptions.RequestException as e:
                last = LibsqlError(f"network: {e}", "NETWORK")
            if i < attempts - 1:
                time.sleep(backoffs[min(i, len(backoffs) - 1)])
        raise last or LibsqlError("pipeline post failed", "UNKNOWN")


def create_client_sync(url: str, auth_token: Optional[str] = None, **kw) -> HttpClientSync:
    """Drop-in for libsql_client.create_client_sync()."""
    if auth_token is None:
        raise LibsqlError("auth_token is required", "AUTH_REQUIRED")
    return HttpClientSync(url, auth_token, **{k: v for k, v in kw.items() if k == "timeout"})
