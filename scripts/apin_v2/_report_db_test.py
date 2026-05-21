"""Phase A test: the reports table + auth_db report helpers.

Runs auth_db against a LOCAL libSQL file (the production _ShimConn code
path) and exercises save / list / get / soft-delete / restore plus the
week-range helpers. Run:  python scripts/apin_v2/_report_db_test.py
"""
import os
import sys
import tempfile
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

_DBFILE = os.path.join(tempfile.gettempdir(), "report_db_test.db")
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
    checks.append(ok)
    mark = "\x1b[32mPASS\x1b[0m" if ok else "\x1b[31mFAIL\x1b[0m"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


print("\n=== SCHEMA ===")
with auth_db.get_conn() as c:
    tbls = {r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
check("reports table created", "reports" in tbls)
with auth_db.get_conn() as c:
    idx = {r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
check("partial unique index present", "idx_reports_active" in idx)

print("\n=== SETUP: user + predictions ===")
sfx = str(int(time.time()))[-6:]
user = auth_db.create_user(username="rep" + sfx,
                           display_name="Report Tester " + sfx,
                           email="rep" + sfx + "@example.com",
                           password="ReportTest!2026",
                           mobile_e164="+919000000000")
uid = user["id"]
other = auth_db.create_user(username="oth" + sfx, display_name="Other " + sfx,
                            email="oth" + sfx + "@example.com",
                            password="ReportTest!2026",
                            mobile_e164="+919000000001")
resp = {"routing": {"router_crop": "okra"}, "diagnosis": "okra_yvmv",
        "confidence": 0.97, "tier": "5"}
p1 = auth_db.record_prediction(uid, resp)
p2 = auth_db.record_prediction(uid, resp)
check("two predictions recorded", isinstance(p1, int) and isinstance(p2, int))

print("\n=== save_report / list_reports ===")
pdf = b"%PDF-1.4\n" + bytes(range(256)) * 20  # ~5 KB stand-in
rid = auth_db.save_report(uid, week_start="2026-05-18", week_end="2026-05-24",
                          pdf_bytes=pdf, summary={"specimens": 2, "diseases": 1})
check("save_report returns an int id", isinstance(rid, int))
lst = auth_db.list_reports(uid)
check("list_reports returns the report", len(lst) == 1 and lst[0]["id"] == rid)
check("list_reports excludes the pdf BLOB", "pdf_bytes" not in (lst[0] if lst else {}))
check("summary_json parsed into summary dict",
      lst and lst[0].get("summary", {}).get("specimens") == 2)

print("\n=== get_report_pdf + IDOR ===")
got = auth_db.get_report_pdf(rid, user_id=uid)
check("get_report_pdf round-trips bytes exactly", got == pdf,
      f"in={len(pdf)} out={len(got) if got else 0}")
check("IDOR: another user cannot read the pdf",
      auth_db.get_report_pdf(rid, user_id=other["id"]) is None)

print("\n=== regenerate replaces the active report ===")
pdf2 = b"%PDF-1.4\n" + b"second" * 200
rid2 = auth_db.save_report(uid, week_start="2026-05-18", week_end="2026-05-24",
                           pdf_bytes=pdf2, summary={"specimens": 9})
lst = auth_db.list_reports(uid)
check("only one active report for the week after regenerate",
      len(lst) == 1 and lst[0]["id"] == rid2)
check("old report pdf no longer reachable",
      auth_db.get_report_pdf(rid, user_id=uid) is None)

print("\n=== soft delete + restore (undo) ===")
check("soft_delete_report returns True",
      auth_db.soft_delete_report(rid2, user_id=uid) is True)
check("deleted report drops out of list_reports",
      len(auth_db.list_reports(uid)) == 0)
check("deleted report pdf is unreachable",
      auth_db.get_report_pdf(rid2, user_id=uid) is None)
check("restore_report returns True",
      auth_db.restore_report(rid2, user_id=uid) is True)
check("restored report is back in list_reports",
      len(auth_db.list_reports(uid)) == 1)
check("IDOR: another user cannot delete the report",
      auth_db.soft_delete_report(rid2, user_id=other["id"]) is False)

print("\n=== restore conflict guard ===")
auth_db.soft_delete_report(rid2, user_id=uid)
rid3 = auth_db.save_report(uid, week_start="2026-05-18", week_end="2026-05-24",
                           pdf_bytes=pdf, summary=None)
check("restoring a deleted report is refused when the week is active again",
      auth_db.restore_report(rid2, user_id=uid) is False)
check("the freshly generated report is the active one",
      len(auth_db.list_reports(uid)) == 1
      and auth_db.list_reports(uid)[0]["id"] == rid3)

print("\n=== predictions_in_range + weekly_prediction_counts ===")
from datetime import datetime, timezone, timedelta
now = datetime.now(timezone.utc)
start = (now - timedelta(days=1)).isoformat()
end = (now + timedelta(days=1)).isoformat()
rng = auth_db.predictions_in_range(uid, start, end)
check("predictions_in_range finds this week's predictions", len(rng) == 2)
check("range rows carry response_json + flags",
      all("response_json" in r and "has_image" in r for r in rng))
past = auth_db.predictions_in_range(uid, "2020-01-01", "2020-01-08")
check("predictions_in_range is empty for an unrelated window", len(past) == 0)
wk = auth_db.weekly_prediction_counts(uid)
check("weekly_prediction_counts totals 2 across the weeks",
      sum(wk.values()) == 2, str(wk))

print("\n" + "=" * 60)
npass = sum(1 for ok in checks if ok)
nfail = sum(1 for ok in checks if not ok)
colour = "\x1b[32m" if nfail == 0 else "\x1b[31m"
print(f"  PHASE A DB TEST: {colour}{npass} passed   {nfail} failed\x1b[0m")
try:
    auth_db._libsql_client.close()
except Exception:
    pass
sys.exit(0 if nfail == 0 else 1)
