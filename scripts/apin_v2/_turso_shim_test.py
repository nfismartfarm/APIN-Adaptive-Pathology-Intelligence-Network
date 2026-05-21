"""Verify the libSQL/Turso backend of auth_db.py.

Runs auth_db with TURSO_DATABASE_URL pointed at a LOCAL libSQL file —
the exact production code path (the _ShimConn adapter over libsql-client),
differing from real Turso only in the connection string. Exercises every
category of helper, with emphasis on BLOB round-trips (image storage).

Run standalone:  python scripts/apin_v2/_turso_shim_test.py
"""
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# Point auth_db at a fresh local libSQL file BEFORE importing it.
_DBFILE = os.path.join(tempfile.gettempdir(), "turso_shim_test.db")
for ext in ("", "-shm", "-wal"):
    try:
        os.remove(_DBFILE + ext)
    except OSError:
        pass
os.environ["TURSO_DATABASE_URL"] = "file:" + _DBFILE.replace("\\", "/")
os.environ.pop("TURSO_AUTH_TOKEN", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.apin_v2 import auth_db  # noqa: E402

checks = []


def check(name, ok, detail=""):
    checks.append((name, ok))
    mark = "\x1b[32mPASS\x1b[0m" if ok else "\x1b[31mFAIL\x1b[0m"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


print("\n=== BACKEND SELECTION ===")
check("auth_db is running in Turso/libSQL mode", auth_db._USE_TURSO is True)
check("shim connection object is built", auth_db._libsql_conn is not None)

print("\n=== SCHEMA BOOTSTRAP (executescript via shim batch) ===")
with auth_db.get_conn() as c:
    tbls = {r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
for t in ("users", "sessions", "guest_sessions", "predictions",
          "treatment_log", "margin_notes", "share_tokens", "audit_log"):
    check(f"table '{t}' created", t in tbls)
with auth_db.get_conn() as c:
    pcols = {r[1] for r in c.execute("PRAGMA table_info(predictions)")}
check("predictions.image_bytes column present (BLOB migration)",
      "image_bytes" in pcols)
check("predictions.heatmap_b64 column present", "heatmap_b64" in pcols)

print("\n=== USER CRUD ===")
import time as _t
_sfx = str(int(_t.time()))[-6:]
uname = "turso_u" + _sfx
check("username not taken before create",
      not auth_db.is_taken("username", uname))
user = auth_db.create_user(
    username=uname, display_name="Turso Tester " + _sfx,
    email=uname + "@example.com", password="TursoTest!2026",
    mobile_e164="+919876500000")
check("create_user returns a row with integer id",
      isinstance(user, dict) and isinstance(user.get("id"), int))
check("username taken after create", auth_db.is_taken("username", uname))
check("get_user_by_handle by username works",
      (auth_db.get_user_by_handle(uname) or {}).get("id") == user["id"])
check("get_user_by_handle by email works",
      (auth_db.get_user_by_handle(uname + "@example.com") or {}).get("id")
      == user["id"])
check("verify_password round-trips through argon2",
      auth_db.verify_password("TursoTest!2026", user["password_hash"]))
try:
    auth_db.create_user(username=uname, display_name="dup " + _sfx,
                        email="dup" + _sfx + "@example.com",
                        password="TursoTest!2026", mobile_e164="+919876500001")
    check("duplicate username raises (UNIQUE -> IntegrityError translated)", False)
except ValueError:
    check("duplicate username raises (UNIQUE -> IntegrityError translated)", True)
except Exception as e:
    check("duplicate username raises", False, type(e).__name__)

print("\n=== SESSIONS ===")
tok = auth_db.create_session(user["id"], user_agent="qa", ip_addr="127.0.0.1")
check("create_session returns a raw token string",
      isinstance(tok, str) and len(tok) > 16)
check("get_session_user resolves the token to the user",
      (auth_db.get_session_user(tok) or {}).get("id") == user["id"])
check("revoke_session returns True", auth_db.revoke_session(tok) is True)
check("revoked token no longer resolves",
      auth_db.get_session_user(tok) is None)

print("\n=== GUEST SESSIONS (quota enforced via shim) ===")
gtok = auth_db.create_guest_session(user_agent="qa")
g = auth_db.get_guest_session(gtok)
check("new guest has full quota remaining",
      g and g["remaining"] == auth_db.GUEST_INFERENCE_LIMIT)
consumed = [auth_db.consume_guest_inference(gtok)
            for _ in range(auth_db.GUEST_INFERENCE_LIMIT)]
check("guest can consume exactly its quota",
      all(c and not c.get("denied") for c in consumed))
denied = auth_db.consume_guest_inference(gtok)
check("guest blocked after quota exhausted",
      denied and denied.get("denied") is True)

print("\n=== PREDICTIONS + IMAGE BLOB ROUND-TRIP ===")
# A small but non-trivial binary payload — the thing that has to survive
# the DB on every restart.
img_bytes = bytes(range(256)) * 40          # 10 KB of every byte value
heatmap_png = b"\x89PNG\r\n\x1a\n" + bytes(range(200)) * 5
import base64 as _b64
# Use the real APIN response keys: the summary extractor reads `diagnosis`
# (not `predicted_class`) and the router crop from `routing.router_crop`.
fake_response = {
    "routing": {"router_crop": "tomato"},
    "diagnosis": "tomato_early_blight",
    "confidence": 0.88, "tier": "FIELD_GRADE",
    "gradcam_b64_png": _b64.b64encode(heatmap_png).decode(),
}
pid = auth_db.record_prediction(user["id"], fake_response,
                                image_bytes=img_bytes)
check("record_prediction returns a new row id", isinstance(pid, int))
got_img = auth_db.get_prediction_image(pid, user_id=user["id"])
check("image BLOB round-trips byte-for-byte through libSQL",
      got_img == img_bytes,
      f"in={len(img_bytes)}B out={len(got_img) if got_img else 0}B")
got_cam = auth_db.get_prediction_heatmap(pid, user_id=user["id"])
check("heatmap round-trips and decodes to PNG bytes",
      got_cam == heatmap_png)
full = auth_db.get_prediction_full(pid, user_id=user["id"])
check("get_prediction_full reports has_image=True",
      full and full.get("has_image") is True)
check("get_prediction_full does NOT carry the raw BLOB",
      full and "image_bytes" not in full)
check("IDOR: another user cannot read the image",
      auth_db.get_prediction_image(pid, user_id=user["id"] + 99999) is None)

print("\n=== DASHBOARD AGGREGATION (the widget queries) ===")
dash = auth_db.get_dashboard_data(user["id"])
check("get_dashboard_data returns hero/calendar/ledger/recent",
      all(k in dash for k in ("hero", "calendar", "ledger", "recent")))
check("hero total_predictions counts the recorded prediction",
      dash["hero"]["total_predictions"] == 1)
check("calendar is a 28-day series",
      isinstance(dash["calendar"], list) and len(dash["calendar"]) == 28)
check("ledger reflects the recorded class",
      any(l["class"] == "tomato_early_blight" for l in dash["ledger"]))
lp = auth_db.list_predictions(user["id"], page=1, page_size=25)
check("list_predictions returns the row with has_image flag",
      len(lp) == 1 and lp[0].get("has_image") is True)
check("count_user_predictions == 1",
      auth_db.count_user_predictions(user["id"]) == 1)
fs = auth_db.first_sightings(user["id"])
check("first_sightings returns the disease lineage",
      any(r["class"] == "tomato_early_blight" for r in fs))

print("\n=== TREATMENT LOG ===")
tr = auth_db.create_treatment(user["id"], treatment="Copper spray",
                              crop="tomato", disease="tomato_early_blight",
                              plot="Plot A", notes="shim test",
                              applied_date="2026-05-21")
check("create_treatment returns a row", isinstance(tr, dict) and tr.get("id"))
trs = auth_db.list_treatments(user["id"])
check("list_treatments returns the new treatment", len(trs) == 1)
upd = auth_db.update_treatment(tr["id"], user_id=user["id"],
                               notes="updated via shim")
check("update_treatment persists the change",
      upd and upd.get("notes") == "updated via shim")
check("delete_treatment removes it",
      auth_db.delete_treatment(tr["id"], user_id=user["id"]) is True
      and len(auth_db.list_treatments(user["id"])) == 0)

print("\n=== SHARE TOKENS ===")
raw_share = auth_db.create_share_token(user["id"], pid, label="for officer")
check("create_share_token returns a raw token",
      isinstance(raw_share, dict) and raw_share.get("token"))
resolved = auth_db.resolve_share_token(raw_share["token"])
check("resolve_share_token returns the shared specimen",
      resolved and resolved.get("predicted_class") == "tomato_early_blight")
check("resolved share does NOT leak user_id / response_json",
      resolved and "user_id" not in resolved
      and "response_json" not in resolved)

print("\n" + "=" * 60)
npass = sum(1 for _, ok in checks if ok)
nfail = sum(1 for _, ok in checks if not ok)
colour = "\x1b[32m" if nfail == 0 else "\x1b[31m"
print(f"  TURSO SHIM TEST: {colour}{npass} passed   {nfail} failed\x1b[0m")

# The libsql ClientSync runs a non-daemon background event-loop thread;
# close it so the interpreter can exit cleanly.
try:
    auth_db._libsql_client.close()
except Exception:
    pass
sys.exit(0 if nfail == 0 else 1)
