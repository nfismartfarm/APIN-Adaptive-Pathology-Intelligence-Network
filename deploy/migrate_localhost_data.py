"""Migrate accounts + predictions from the localhost SQLite DB to Turso.

During local development the app wrote to data/apin_v2.db (a plain SQLite
file). Production uses the external Turso database. This one-off script
copies selected user accounts and their predictions (dashboard data —
including the uploaded-image BLOBs and Grad-CAM heatmaps) into Turso so
nothing from local testing is lost.

What it copies:  users + predictions (the dashboard's data).
What it skips:   sessions / guest_sessions / audit_log (ephemeral),
                 share_tokens (regenerable), treatment_log / margin_notes
                 (empty locally).

Idempotent: a user already present in Turso (same email or username) is
skipped, so the script is safe to re-run.

──────────────────────────────────────────────────────────────────────────
USAGE
  # 1. Point it at the production Turso database (same values as the
  #    Hugging Face Space secrets):
  export TURSO_DATABASE_URL="***REDACTED-TURSO-URL***"
  export TURSO_AUTH_TOKEN="<your-turso-token>"
  #    (Windows PowerShell:  $env:TURSO_DATABASE_URL="..."  etc.)

  # 2. List the local accounts (does nothing else):
  python deploy/migrate_localhost_data.py

  # 3. Migrate the ones you want, by id:
  python deploy/migrate_localhost_data.py --users 5
  python deploy/migrate_localhost_data.py --users 5,11
  python deploy/migrate_localhost_data.py --all
──────────────────────────────────────────────────────────────────────────
"""
import argparse
import os
import sqlite3
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB = os.path.join(ROOT, "data", "apin_v2.db")

# A local account is treated as a QA / throwaway account (skipped by --real)
# if its email is on the test domain or any of these markers appears in its
# username / email / display name.
TEST_MARKERS = ("test", "qa", "demo", "debug", "browserqa", "dbg",
                "dashtester", "fielduser")


def _is_test_account(username, email, display_name):
    blob = f"{username or ''} {email or ''} {display_name or ''}".lower()
    if "@example.com" in (email or "").lower():
        return True
    return any(m in blob for m in TEST_MARKERS)


USER_COLS = ["username", "display_name", "email", "password_hash",
             "mobile_e164", "pressed_leaf_seed", "role",
             "preferred_language", "profile", "created_at", "last_seen_at"]
PRED_COLS = ["crop", "predicted_class", "confidence", "tier", "image_sha256",
             "response_json", "created_at", "image_bytes", "heatmap_b64"]


def turso_client():
    """Open a Turso client over the Hrana-HTTP transport (the WS endpoint
    is retired — see auth_db._turso_http_url)."""
    url = os.environ.get("TURSO_DATABASE_URL", "").strip()
    token = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
    if not url or not token:
        sys.exit("ERROR: set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN first "
                 "(the same values as the Hugging Face Space secrets).")
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://"):]
    elif url.startswith("wss://"):
        url = "https://" + url[len("wss://"):]
    import libsql_client
    return libsql_client.create_client_sync(url, auth_token=token)


def list_accounts(local):
    print(f"\nLocal accounts in {os.path.relpath(LOCAL_DB, ROOT)}:\n")
    rows = local.execute(
        "SELECT u.id, u.username, u.display_name, u.email, "
        "       (SELECT COUNT(*) FROM predictions p WHERE p.user_id=u.id) n "
        "FROM users u ORDER BY u.id").fetchall()
    for r in rows:
        print(f"  id={r[0]:<3} {r[1]:<20} {r[3]:<32} {r[4]:>3} predictions"
              f"  ({r[2]})")
    print(f"\n  {len(rows)} accounts total.")
    print("\nRe-run with --users <id,id,...> to migrate, e.g.:")
    print("  python deploy/migrate_localhost_data.py --users 5\n")


def migrate(local, turso, user_ids):
    local.row_factory = sqlite3.Row
    total_u = total_p = 0

    for uid in user_ids:
        urow = local.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if urow is None:
            print(f"  user id={uid}: not found locally — skipped")
            continue

        # Idempotency: skip if this account already exists in Turso.
        existing = turso.execute(
            "SELECT id FROM users WHERE email=? OR username=?",
            [urow["email"], urow["username"]])
        if existing.rows:
            print(f"  user id={uid} ({urow['username']}): already in Turso "
                  f"— skipped")
            continue

        # Insert the user (all columns except the local auto-increment id).
        placeholders = ",".join("?" * len(USER_COLS))
        ins = turso.execute(
            f"INSERT INTO users ({','.join(USER_COLS)}) VALUES ({placeholders})",
            [urow[c] for c in USER_COLS])
        new_uid = ins.last_insert_rowid
        total_u += 1

        # Insert that user's predictions, remapped onto the new user id.
        preds = local.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY id",
            (uid,)).fetchall()
        p_placeholders = ",".join("?" * (len(PRED_COLS) + 1))
        n_img = 0
        for p in preds:
            vals = [new_uid] + [p[c] for c in PRED_COLS]
            turso.execute(
                f"INSERT INTO predictions (user_id,{','.join(PRED_COLS)}) "
                f"VALUES ({p_placeholders})", vals)
            if p["image_bytes"] is not None:
                n_img += 1
        total_p += len(preds)
        print(f"  user id={uid} ({urow['username']}) -> Turso id={new_uid}: "
              f"{len(preds)} predictions ({n_img} with images)")

    print(f"\nMigrated {total_u} account(s) and {total_p} prediction(s) "
          f"into Turso.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="",
                    help="comma-separated local user ids to migrate")
    ap.add_argument("--real", action="store_true",
                    help="migrate only genuine accounts (skip every QA / "
                         "test / demo account), recommended")
    ap.add_argument("--all", action="store_true",
                    help="migrate every local account (not recommended, "
                         "includes QA throwaway accounts)")
    args = ap.parse_args()

    if not os.path.exists(LOCAL_DB):
        sys.exit(f"ERROR: local database not found at {LOCAL_DB}")
    local = sqlite3.connect(LOCAL_DB)

    if not args.users and not args.all and not args.real:
        list_accounts(local)
        return

    if args.all:
        ids = [r[0] for r in local.execute("SELECT id FROM users ORDER BY id")]
    elif args.real:
        ids = [r[0] for r in local.execute(
            "SELECT id, username, email, display_name FROM users ORDER BY id")
            if not _is_test_account(r[1], r[2], r[3])]
        if not ids:
            sys.exit("No genuine accounts found to migrate.")
    else:
        try:
            ids = [int(x) for x in args.users.split(",") if x.strip()]
        except ValueError:
            sys.exit("ERROR: --users must be comma-separated integers, "
                     "e.g. --users 5,11")

    turso = turso_client()
    print(f"\nMigrating {len(ids)} account(s) -> Turso ...\n")
    try:
        migrate(local, turso, ids)
    finally:
        try:
            turso.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
