"""Phase-1 smoke test for the dashboard build. Run after server start.
Validates every new route exists, returns sane payloads, and that filters
+ pagination + exports actually narrow / paginate / export correctly."""
import json
import sys
import urllib.parse
import urllib.request
import urllib.error

# Force UTF-8 on Windows so unicode arrows in check labels don't crash
# the cp1252 console (matching what _phase1_browser_qa.py already does).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = "http://127.0.0.1:8766"

# ── Authenticated session ─────────────────────────────────────────────────
# Since the dev-mode auth bypass was removed, every /dashboard/* route now
# requires a real session.  The suite mints a session for the `dashtester`
# account (which carries the seeded prediction/treatment data the dashboard
# assertions were written against) and sends it as a cookie on every request.
_SESSION_COOKIE = ""   # "apin_v2_session=<token>"


def _setup_session():
    """Mint a DB-level session for the dashtester account and return the
    cookie header value.  Direct DB insert (not /auth/login) because the
    account password isn't known to the suite."""
    global _SESSION_COOKIE
    import os, sqlite3, secrets, hashlib
    from datetime import datetime, timezone, timedelta
    db = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "apin_v2.db")
    raw = secrets.token_urlsafe(32)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    c = sqlite3.connect(db)
    row = c.execute("SELECT id FROM users WHERE username = 'dashtester'").fetchone()
    if not row:
        c.close()
        raise RuntimeError("dashtester account not found — seed it first")
    c.execute("INSERT INTO sessions (user_id, token_hash, created_at, expires_at) "
              "VALUES (?,?,?,?)",
              (row[0], h, now.isoformat(), (now + timedelta(days=1)).isoformat()))
    c.commit()
    c.close()
    _SESSION_COOKIE = "apin_v2_session=" + raw
    return _SESSION_COOKIE


def get(path, expect_json=True, expect_status=200):
    url = BASE + path
    req = urllib.request.Request(url)
    if _SESSION_COOKIE:
        req.add_header("Cookie", _SESSION_COOKIE)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            ct = resp.headers.get("content-type", "")
            body = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read()
        ct = e.headers.get("content-type", "") if hasattr(e, "headers") else ""
    if status != expect_status:
        return {"ok": False, "status": status, "path": path,
                "body_head": body[:300].decode("utf-8", "replace")}
    if expect_json and "json" in ct:
        return {"ok": True, "status": status, "path": path,
                "json": json.loads(body.decode("utf-8"))}
    return {"ok": True, "status": status, "path": path,
            "bytes": len(body), "content_type": ct,
            "body_head": body[:120].decode("utf-8", "replace")}


def red(s):   return f"\x1b[31m{s}\x1b[0m"
def green(s): return f"\x1b[32m{s}\x1b[0m"
def yel(s):   return f"\x1b[33m{s}\x1b[0m"


checks = []
def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    mark = green("PASS") if ok else red("FAIL")
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


print("\n=== AUTH SETUP ===")
try:
    _setup_session()
    check("minted dashtester session for authenticated probes", bool(_SESSION_COOKIE))
except Exception as e:
    check("minted dashtester session", False, str(e))

print("\n=== AUTH GATING (security fix — dev-mode bypass removed) ===")
# These probes carry NO session cookie — they verify the bypass is gone.
def _anon(method, path, allow_redirects=True):
    """Cookie-less request. Returns (status, json_or_None, redirect_location)."""
    req = urllib.request.Request(BASE + path, method=method)
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k): return None
    opener = urllib.request.build_opener(_NoRedirect) if not allow_redirects \
        else urllib.request.build_opener()
    try:
        with opener.open(req, timeout=15) as resp:
            txt = resp.read().decode("utf-8", "replace")
            try: j = json.loads(txt)
            except Exception: j = None
            return resp.status, j, resp.headers.get("Location", "")
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try: j = json.loads(txt)
        except Exception: j = None
        return e.code, j, e.headers.get("Location", "") if hasattr(e, "headers") else ""

_st, _j, _ = _anon("GET", "/auth/state")
check("anonymous /auth/state -> mode=anonymous",
      _st == 200 and _j and _j.get("mode") == "anonymous", f"got={_j}")

_st, _j, _loc = _anon("GET", "/dashboard", allow_redirects=False)
check("anonymous /dashboard -> 303 redirect to /",
      _st in (302, 303) and (_loc == "/" or _loc.rstrip("/").endswith("8766")),
      f"status={_st} loc={_loc}")

_st, _j, _loc = _anon("GET", "/dashboard/settings", allow_redirects=False)
check("anonymous /dashboard/settings -> redirect",
      _st in (302, 303), f"status={_st}")

_st, _j, _ = _anon("GET", "/dashboard/data")
check("anonymous /dashboard/data -> 401 (no data leak)",
      _st == 401, f"status={_st}")
check("401 body marks auth_required",
      bool(_j) and _j.get("auth_required") is True, f"body={_j}")

_st, _j, _ = _anon("POST", "/auth/guest")
check("POST /auth/guest -> creates guest session",
      _st == 200 and _j and _j.get("mode") == "guest"
      and _j.get("remaining") == 3, f"got={_j}")

print("\n=== ROUTE PROBE ===")
for path in ["/dashboard", "/dashboard/history",
             "/dashboard/data", "/dashboard/history/data",
             "/dashboard/history/export.csv",
             "/dashboard/history/export.json"]:
    r = get(path, expect_json=False)
    check(f"GET {path}", r.get("ok") and r["status"] == 200,
          f"status={r.get('status')} bytes={r.get('bytes') or '-'}")

print("\n=== /dashboard/data shape ===")
r = get("/dashboard/data")
d = r.get("json") or {}
check("payload has 'user'",         "user"    in d)
check("payload has 'hero'",         "hero"    in d)
check("payload has 'calendar'",     "calendar" in d and len(d.get("calendar", [])) == 28)
check("payload has 'ledger'",       "ledger"  in d)
check("payload has 'crop_mix'",     "crop_mix" in d)
check("payload has 'dominant_crop'","dominant_crop" in d)
check("payload has 'recent'",       "recent"  in d and len(d.get("recent", [])) <= 6)
check("payload has 'confidence_histogram'", "confidence_histogram" in d,
      f"bins={len(d.get('confidence_histogram', []))}")
check("confidence_histogram has 10 bins",
      isinstance(d.get("confidence_histogram"), list)
      and len(d.get("confidence_histogram", [])) == 10)
check("payload has 'daily_brief'",  "daily_brief" in d,
      f"text len={len((d.get('daily_brief') or {}).get('text', ''))}")
check("daily_brief.text non-empty", bool((d.get("daily_brief") or {}).get("text")))
check("daily_brief.tokens has n_today",
      "n_today" in (d.get("daily_brief") or {}).get("tokens", {}))

print("\n=== /dashboard/history/data — unfiltered page 1 ===")
r = get("/dashboard/history/data?page=1&page_size=10")
d = r.get("json") or {}
check("has results",     isinstance(d.get("results"), list))
check("has total",       "total" in d)
check("has total_unfiltered", "total_unfiltered" in d)
check("has n_pages",     "n_pages" in d)
check("has available_filters.diseases", "diseases" in d.get("available_filters", {}))
check("returned <= 10 rows", len(d.get("results", [])) <= 10,
      f"got {len(d.get('results', []))}")
total_unfiltered_a = d.get("total_unfiltered")

print("\n=== /dashboard/history/data — filtered by crop=tomato ===")
r = get("/dashboard/history/data?crop=tomato")
d = r.get("json") or {}
check("filtered total > 0",          d.get("total", 0) > 0)
check("filtered total <= unfiltered", d.get("total", 0) <= total_unfiltered_a,
      f"{d.get('total')} <= {total_unfiltered_a}")
check("filters_applied.crop == tomato",
      d.get("filters_applied", {}).get("crop") == "tomato")
check("every result has crop=tomato",
      all(r.get("crop") == "tomato" for r in d.get("results", [])))

print("\n=== /dashboard/history/data — search='blight' ===")
r = get("/dashboard/history/data?q=blight")
d = r.get("json") or {}
check("search matches some rows", d.get("total", 0) > 0)
check("every result contains 'blight' in predicted_class",
      all("blight" in (r.get("predicted_class") or "").lower()
          for r in d.get("results", [])))

print("\n=== /dashboard/history/data — tier=FIELD_GRADE ===")
r = get("/dashboard/history/data?tier=FIELD_GRADE")
d = r.get("json") or {}
check("tier filter returns some rows", d.get("total", 0) > 0)
check("every result has tier=FIELD_GRADE",
      all((r.get("tier") or "").upper() == "FIELD_GRADE"
          for r in d.get("results", [])))

print("\n=== /dashboard/history/data — sort=highest ===")
r = get("/dashboard/history/data?sort=highest&page_size=20")
d = r.get("json") or {}
rows = d.get("results", [])
sorted_ok = all(
    (rows[i].get("confidence") or 0) >= (rows[i+1].get("confidence") or 0)
    for i in range(len(rows) - 1)
)
check("rows monotonically decreasing in confidence", sorted_ok)

print("\n=== Pagination consistency ===")
r1 = get("/dashboard/history/data?page=1&page_size=5")
r2 = get("/dashboard/history/data?page=2&page_size=5")
ids1 = [x["id"] for x in r1["json"]["results"]]
ids2 = [x["id"] for x in r2["json"]["results"]]
check("page 1 + 2 are disjoint", not (set(ids1) & set(ids2)),
      f"overlap = {sorted(set(ids1) & set(ids2))}")
check("page 1 size matches page_size=5", len(ids1) <= 5)

print("\n=== Export CSV ===")
r = get("/dashboard/history/export.csv?crop=tomato", expect_json=False)
check("export csv 200", r.get("ok"))
check("export csv content-type is csv",
      "csv" in (r.get("content_type") or "").lower())

print("\n=== Export JSON ===")
r = get("/dashboard/history/export.json?crop=tomato")
check("export json 200", r.get("ok"))
ej = r.get("json") or {}
check("export json has n_rows", "n_rows" in ej)
check("export json has results array", isinstance(ej.get("results"), list))


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2 backend smoke
# ═══════════════════════════════════════════════════════════════════════

print("\n=== PHASE 2 / /dashboard/prediction/{id} ===")
r = get("/dashboard/prediction/10")
d = r.get("json") or {}
check("prediction detail returns id, class, confidence",
      d.get("id") == 10 and d.get("predicted_class") and d.get("confidence") is not None)
check("prediction detail includes parsed_signals dict",
      isinstance(d.get("parsed_signals"), dict))
check("parsed_signals has all 4 signal keys",
      all(k in (d.get("parsed_signals") or {})
          for k in ("model2","efficientnet","dinov2","psv")))
check("response_json blob stripped from wire payload",
      "response_json" not in d)

print("\n=== PHASE 2 / /dashboard/prediction/{id} (not found) ===")
r = get("/dashboard/prediction/999999", expect_status=404)
check("missing prediction returns 404", r.get("status") == 404)

print("\n=== PHASE 2 / /dashboard/disease/{class}/predictions ===")
r = get("/dashboard/disease/tomato_early_blight/predictions")
d = r.get("json") or {}
check("disease drill-down returns class", d.get("class") == "tomato_early_blight")
check("disease drill-down returns taxonomy block",
      isinstance(d.get("taxonomy"), dict))
check("disease drill-down has results array",
      isinstance(d.get("results"), list) and len(d.get("results", [])) > 0)
check("disease drill-down total > 0", d.get("total", 0) > 0)
check("every result has predicted_class == tomato_early_blight",
      all(r.get("predicted_class") == "tomato_early_blight" for r in d.get("results", [])))

print("\n=== PHASE 2 / /dashboard/taxonomy ===")
r = get("/dashboard/taxonomy")
d = r.get("json") or {}
check("taxonomy returns tree object", isinstance(d.get("tree"), dict))
check("taxonomy has at least one kingdom",
      len(d.get("tree", {})) >= 1, f"kingdoms={list((d.get('tree') or {}).keys())}")

print("\n=== PHASE 2 / /dashboard/notes (CRUD round-trip) ===")
import urllib.request, json as _json

# initial baseline
r = get("/dashboard/notes")
initial_total = (r.get("json") or {}).get("total", 0)

def _http(method, path, body=None):
    data = _json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if _SESSION_COOKIE:
        headers["Cookie"] = _SESSION_COOKIE
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            txt = resp.read().decode("utf-8")
            try: j = _json.loads(txt) if txt else None
            except _json.JSONDecodeError: j = None
            return {"status": resp.status, "json": j}
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8","replace")
        try: j = _json.loads(txt) if txt else None
        except _json.JSONDecodeError: j = None
        return {"status": e.code, "json": j}

created = _http("POST", "/dashboard/notes",
                {"text": "smoke-test note", "attached_date": "2026-05-14", "mood": 2})
check("POST /dashboard/notes returns 201",
      created.get("status") == 201, f"status={created.get('status')}")
note_id = (created.get("json") or {}).get("id")
check("created note has integer id", isinstance(note_id, int))

r = get("/dashboard/notes")
after_total = (r.get("json") or {}).get("total", 0)
check("note list total grew by 1",
      after_total == initial_total + 1,
      f"before={initial_total} after={after_total}")

r = get("/dashboard/notes?date=2026-05-14")
date_notes = (r.get("json") or {}).get("results", [])
check("date filter returns the created note",
      any(n.get("id") == note_id for n in date_notes))

if note_id:
    patched = _http("PATCH", f"/dashboard/notes/{note_id}",
                    {"text": "edited", "mood": 3})
    check("PATCH /dashboard/notes/{id} returns 200",
          patched.get("status") == 200)
    check("PATCH applied — text updated",
          (patched.get("json") or {}).get("text") == "edited")
    check("PATCH applied — mood updated to 3",
          (patched.get("json") or {}).get("mood") == 3)

    deleted = _http("DELETE", f"/dashboard/notes/{note_id}")
    check("DELETE /dashboard/notes/{id} returns 200",
          deleted.get("status") == 200)

    r = get("/dashboard/notes")
    final_total = (r.get("json") or {}).get("total", 0)
    check("note count returned to initial after delete",
          final_total == initial_total,
          f"initial={initial_total} final={final_total}")

print("\n=== PHASE 2 / validation guards ===")
bad = _http("POST", "/dashboard/notes", {"text": ""})
check("empty-text note rejected with 400",
      bad.get("status") == 400)
bad2 = _http("POST", "/dashboard/notes",
             {"text":"oops","attached_date":"2026-05-14","attached_prediction_id":10})
check("note attaching to BOTH date+prediction rejected with 400",
      bad2.get("status") == 400)


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3 backend smoke
# ═══════════════════════════════════════════════════════════════════════

print("\n=== PHASE 3 / new HTML pages serve ===")
for path in ["/dashboard/reports", "/dashboard/loupe",
             "/dashboard/gallery", "/dashboard/settings"]:
    r = get(path, expect_json=False)
    check(f"GET {path}", r.get("ok") and r["status"] == 200,
          f"bytes={r.get('bytes', '-')}")

print("\n=== PHASE 3 / Treatment Log CRUD ===")
r = get("/dashboard/treatments")
init_count = (r.get("json") or {}).get("total", 0)

t = _http("POST", "/dashboard/treatments",
          {"treatment": "neem oil spray (smoke)",
           "applied_date": "2026-05-14",
           "crop": "okra",
           "disease": "okra_yvmv",
           "plot": "Plot A",
           "notes": "smoke test"})
check("POST /dashboard/treatments returns 201",
      t.get("status") == 201)
tid = (t.get("json") or {}).get("id")
check("created treatment has integer id", isinstance(tid, int))

r = get("/dashboard/treatments?crop=okra")
check("filter by crop=okra returns the treatment",
      any(x.get("id") == tid for x in (r.get("json") or {}).get("results", [])))

r = get("/dashboard/treatments?disease=okra_yvmv")
check("filter by disease=okra_yvmv returns the treatment",
      any(x.get("id") == tid for x in (r.get("json") or {}).get("results", [])))

if tid:
    u = _http("PATCH", f"/dashboard/treatments/{tid}",
              {"notes": "smoke test — updated"})
    check("PATCH /dashboard/treatments/{id} returns 200",
          u.get("status") == 200)
    check("PATCH applied — notes updated",
          (u.get("json") or {}).get("notes", "").endswith("updated"))

# Bad PATCH: empty treatment text
bad = _http("PATCH", f"/dashboard/treatments/{tid}", {"treatment": "   "})
check("PATCH with empty treatment text rejected with 400",
      bad.get("status") == 400)

# CSV export
r = get("/dashboard/treatments/export.csv", expect_json=False)
check("export.csv 200", r.get("ok"))
check("export.csv content-type is csv",
      "csv" in (r.get("content_type") or "").lower())

# DELETE
if tid:
    d = _http("DELETE", f"/dashboard/treatments/{tid}")
    check("DELETE /dashboard/treatments/{id} returns 200",
          d.get("status") == 200)

print("\n=== PHASE 3 / Share tokens ===")
# Create a share for prediction id=10 (known seeded prediction)
r = _http("POST", "/dashboard/shares",
          {"prediction_id": 10, "label": "smoke share"})
check("POST /dashboard/shares returns 201", r.get("status") == 201)
share = r.get("json") or {}
token = share.get("token")
sid = share.get("id")
check("share response includes raw token", isinstance(token, str) and len(token) > 16)
check("share response includes share id", isinstance(sid, int))

# Public viewer — HTML page
r = get(f"/share/{token}", expect_json=False)
check(f"GET /share/{{token}} HTML returns 200", r.get("ok"))

# Public data endpoint
r = get(f"/share/{token}/data")
share_data = r.get("json") or {}
check("share data returns predicted_class",
      share_data.get("predicted_class"))
check("share data does NOT leak response_json blob",
      "response_json" not in share_data)
# Round-2 PDA finding: ensure all server-only fields are stripped
check("share data does NOT leak parsed_signals dict",
      "parsed_signals" not in share_data)
check("share data does NOT leak user_id",
      "user_id" not in share_data)
check("share data does NOT leak internal prediction id",
      "id" not in share_data)
check("share data does NOT leak image_sha256",
      "image_sha256" not in share_data)
# Whitelist of fields that SHOULD be in the public payload.
# has_image / has_heatmap are boolean flags the share-viewer JS uses to
# decide whether to render an <img> tag or the honest pre-upgrade
# placeholder; they leak no sensitive data (just a yes/no for each).
_allowed_share_keys = {"crop", "predicted_class", "confidence",
                       "tier", "created_at", "share",
                       "has_image", "has_heatmap"}
_unexpected = set(share_data.keys()) - _allowed_share_keys
check("share data only contains whitelisted public keys",
      not _unexpected,
      f"unexpected keys leaked: {sorted(_unexpected)}")
check("share data includes share metadata",
      isinstance(share_data.get("share"), dict))
check("view_count incremented on viewer GET",
      share_data.get("share", {}).get("view_count", 0) >= 1)

# Bad token — should 404
r = get("/share/clearly-not-real-token-xxxx/data", expect_status=404)
check("invalid share token → 404", r.get("status") == 404)

# POST share for unowned prediction id → must 404
r = _http("POST", "/dashboard/shares",
          {"prediction_id": 999999, "label": "should fail"})
check("share for non-owned prediction → 404", r.get("status") == 404)

# Bad POST: missing prediction_id
r = _http("POST", "/dashboard/shares", {"label": "no prediction id"})
check("share without prediction_id → 400", r.get("status") == 400)

# Revoke
if sid:
    r = _http("DELETE", f"/dashboard/shares/{sid}")
    check("DELETE /dashboard/shares/{id} returns 200",
          r.get("status") == 200)
    # Confirm revoked: public viewer must now 404
    r = get(f"/share/{token}/data", expect_status=404)
    check("revoked share → 404 on public viewer", r.get("status") == 404)

print("\n=== PHASE 3.5 / Image storage routes ===")
# Drive at least one fresh prediction through /predict/full so an image
# is guaranteed to exist for the dev-fallback user (`dashtester`).
import io as _io_for_jpg, time as _time_for_image
try:
    import numpy as _np
    from PIL import Image as _PIL
except Exception as _e:
    print(f"  [SKIP] image-route tests need numpy + Pillow ({_e})")
    _np = _PIL = None

def _multipart_jpeg_post(path, jpeg_bytes):
    """Tiny multipart POST using urllib (no `requests` dep)."""
    boundary = "----smokeboundary" + str(int(_time_for_image.time()*1000))
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="smoke.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + jpeg_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    _mh = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if _SESSION_COOKIE:
        _mh["Cookie"] = _SESSION_COOKIE
    req = urllib.request.Request(
        BASE + path, data=body, method="POST", headers=_mh,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

def _raw_get(path):
    """GET that returns (status, content_bytes, content_type)."""
    req = urllib.request.Request(BASE + path, method="GET")
    if _SESSION_COOKIE:
        req.add_header("Cookie", _SESSION_COOKIE)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type","")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), ""

if _np is not None and _PIL is not None:
    _rng = _np.random.default_rng(seed=99)
    _h, _w = 320, 320
    _base = _np.zeros((_h, _w, 3), dtype=_np.uint8)
    _y, _x = _np.indices((_h, _w))
    _base[:, :, 1] = 100 + ((_x // 4 + _y // 5) % 80).astype(_np.uint8)
    _noise = _rng.integers(-30, 30, (_h, _w, 3), dtype=_np.int16)
    _img  = _np.clip(_base.astype(_np.int16) + _noise, 0, 255).astype(_np.uint8)
    _buf  = _io_for_jpg.BytesIO()
    _PIL.fromarray(_img).save(_buf, format="JPEG", quality=88)
    _jpg = _buf.getvalue()

    _st, _body = _multipart_jpeg_post("/predict/full", _jpg)
    check("POST /predict/full (smoke leaf) returns 200", _st == 200,
          f"status={_st}")
    # Wait for the background record_prediction task to flush. Poll instead
    # of a fixed sleep so a slow flush under load doesn't flake the suite.
    _pid_fresh = None
    for _ in range(20):                       # up to ~10s
        _time_for_image.sleep(0.5)
        _newest = get("/dashboard/history/data?sort=newest&page_size=1")
        _rows = (_newest.get("json") or {}).get("results", [])
        if _rows and _rows[0].get("has_image") is True:
            break

    _newest = get("/dashboard/history/data?sort=newest&page_size=1")
    _rows = (_newest.get("json") or {}).get("results", [])
    check("history listing exposes has_image / has_heatmap flags",
          bool(_rows) and "has_image" in _rows[0])
    _pid_fresh = _rows[0]["id"] if _rows else None
    check("freshly-uploaded row has has_image=True",
          bool(_rows) and _rows[0].get("has_image") is True)

    if _pid_fresh:
        # Phase-3.5 regression: pipeline_visualizations must be stripped
        # from response_json so we don't blow past the 200-KB cap and
        # silently truncate (which would wipe signal_predictions and
        # break the Day Detail signal bars).
        _detail = get(f"/dashboard/prediction/{_pid_fresh}")
        _full   = (_detail.get("json") or {})
        _ps     = _full.get("parsed_signals") or {}
        check("response_json was NOT truncated (signal_predictions survived)",
              isinstance(_ps.get("model2"), dict),
              f"parsed_signals.model2={_ps.get('model2')!r}")
        # All four signals normalized into {confidence, vote} shape
        for _sig_name in ("model2", "efficientnet", "dinov2", "psv"):
            _sv = _ps.get(_sig_name)
            check(f"parsed_signals.{_sig_name} has confidence + vote",
                  isinstance(_sv, dict) and
                  isinstance(_sv.get("confidence"), (int, float)) and
                  isinstance(_sv.get("vote"), str),
                  f"value={_sv!r}")
        # Heatmap should be captured via the fallback (gate_zero_leaf_mask)
        # because the okra/brassica pipeline's gradcam_b64_png is often
        # None for low-confidence predictions.
        check("freshly-uploaded row has has_heatmap=True (gradcam OR fallback)",
              _full.get("has_heatmap") is True,
              f"has_heatmap={_full.get('has_heatmap')}")

        _st, _body, _ct = _raw_get(f"/dashboard/predictions/{_pid_fresh}/image")
        check("GET /dashboard/predictions/{id}/image returns 200 (owned row)",
              _st == 200, f"status={_st}")
        check("image response is JPEG/PNG/WebP",
              (_body[:3] == b"\xff\xd8\xff") or
              (_body[:8] == b"\x89PNG\r\n\x1a\n") or
              (_body[:4] == b"RIFF"),
              f"first bytes: {_body[:8]!r}")
        check("image content-type is image/*",
              _ct.startswith("image/"), f"got={_ct}")

        # IDOR proxy — pre-upgrade row should 404 even for the dev user.
        _oldest = get("/dashboard/history/data?sort=oldest&page_size=1")
        _oldrow = ((_oldest.get("json") or {}).get("results") or [{}])[0]
        _pid_old = _oldrow.get("id")
        if _pid_old and _oldrow.get("has_image") is False:
            _st2, _, _ = _raw_get(f"/dashboard/predictions/{_pid_old}/image")
            check("pre-upgrade row image route returns 404", _st2 == 404,
                  f"status={_st2}")

        _st3, _, _ = _raw_get("/dashboard/predictions/99999999/image")
        check("nonexistent prediction image returns 404", _st3 == 404)

        _st4, _, _ = _raw_get(f"/dashboard/predictions/{_pid_fresh}/heatmap")
        check("heatmap route returns 200 or 404 (never 5xx)",
              _st4 in (200, 404), f"status={_st4}")

        # Public share image flow
        _share = _http("POST", "/dashboard/shares",
                       {"prediction_id": _pid_fresh, "label": "smoke share image"})
        if _share.get("status") in (200, 201):
            _stok = (_share.get("json") or {}).get("token", "")
            _sid  = (_share.get("json") or {}).get("id")
            _stat, _body2, _ct2 = _raw_get(f"/share/{_stok}/image")
            check("public /share/{token}/image returns 200 with real bytes",
                  _stat == 200 and len(_body2) > 1000,
                  f"status={_stat} size={len(_body2)}")
            check("public share image content-type is image/*",
                  _ct2.startswith("image/"))
            check("public share image is JPEG/PNG/WebP",
                  (_body2[:3] == b"\xff\xd8\xff") or
                  (_body2[:8] == b"\x89PNG\r\n\x1a\n"),
                  f"first bytes: {_body2[:8]!r}")
            if _sid is not None:
                _http("DELETE", f"/dashboard/shares/{_sid}")
                _stat2, _, _ = _raw_get(f"/share/{_stok}/image")
                check("revoked share image returns 404", _stat2 == 404)

        # Garbage token → 404
        _stat3, _, _ = _raw_get("/share/totally-bogus-token-xxx/image")
        check("garbage share token image returns 404", _stat3 == 404)

print("\n=== PHASE 3 / Weekly PDF ===")
r = get("/dashboard/report/weekly.pdf", expect_json=False)
check("GET /dashboard/report/weekly.pdf returns 200",
      r.get("ok") and r.get("status") == 200)
check("PDF content-type is application/pdf",
      "pdf" in (r.get("content_type") or "").lower(),
      f"got={r.get('content_type')}")
# Ensure it's a real PDF (starts with %PDF-)
check("PDF body starts with %PDF- magic bytes",
      (r.get("body_head") or "").startswith("%PDF-"),
      f"head={(r.get('body_head') or '')[:20]!r}")

# Custom week_start
r = get("/dashboard/report/weekly.pdf?week_start=2026-05-08", expect_json=False)
check("PDF with custom week_start returns 200", r.get("ok"))

# Malformed week_start should not crash (falls back to default)
r = get("/dashboard/report/weekly.pdf?week_start=garbage", expect_json=False)
check("PDF with garbage week_start does not crash", r.get("ok"))

print("\n=== SUMMARY ===")
n_pass = sum(1 for _, ok, _ in checks if ok)
n_fail = sum(1 for _, ok, _ in checks if not ok)
print(f"  {green(f'{n_pass} passed')}    " +
      (red(f'{n_fail} failed') if n_fail else green('0 failed')))
sys.exit(0 if n_fail == 0 else 2)
