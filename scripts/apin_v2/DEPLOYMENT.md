# APIN v2 — Deployment Guide

Hosting target: a **free Hugging Face Space** (the app) + a **free Turso
database** (durable storage). The Space's disk is ephemeral; Turso holds
every account, prediction, uploaded image, treatment and note, so nothing
is lost when the Space restarts.

```
visitor ──HTTPS──> Hugging Face Space (free, CPU)
                     │  FastAPI app — auth, inference, dashboard
                     └──libsql──> Turso database (free, durable)
                                   accounts · predictions · image BLOBs
```

The app already supports this. `auth_db.py` switches backend purely by
environment variable — no code change is needed to go from local SQLite
to Turso:

| `TURSO_DATABASE_URL`            | Backend used                         |
|---------------------------------|--------------------------------------|
| *unset*                         | local `sqlite3` file (dev + tests)   |
| `libsql://….turso.io`           | Turso (production)                   |
| `file:/path.db`                 | local libSQL (used to test the path) |

---

## Status — LIVE

The app is deployed and running: **https://dxv-404-apin.hf.space**

- [x] **Database migration** — `auth_db.py` runs on Turso/libSQL. Verified:
      46/46 direct backend tests (incl. byte-for-byte image-BLOB round-trip),
      full server integration on libSQL, and 127+56+17 regression on the
      local path.
- [x] **Part A — Turso database** — `apin` database created.
- [x] **Part B — Hugging Face Space** — `dxv-404/Apin` (Docker SDK), the
      `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` secrets are set.
- [x] **Part C — Deployment build** — `deploy/space_files/` (Dockerfile,
      requirements, download_weights.py, README) + `deploy/stage_space.py`
      assemble the bundle; weights live in `dxv-404/apin-models` and are
      pulled at Docker-build time.
- [x] **Part D — First-run verification** — build + boot confirmed; live
      smoke test passed: `/health`, `/`, `/auth/state`, a guest session,
      and an end-to-end `/predict/full` on a tomato leaf (diagnosis +
      Grad-CAM heatmap returned, router routed to the tomato pipeline).

### Turso transport note

Turso has retired the bare Hrana-over-WebSocket endpoint, so a `libsql://`
URL (which `libsql-client` maps to `wss://`) fails the upgrade with
`WSServerHandshakeError: 400`. `auth_db._turso_http_url()` rewrites the URL
to `https://`, forcing the Hrana-over-HTTP transport, which works. No change
to the `TURSO_DATABASE_URL` secret is needed — keep it as `libsql://…`.

### Redeploying after a code change

```bash
python deploy/stage_space.py          # restage deploy/apin-space/
hf upload dxv-404/Apin deploy/apin-space . --repo-type space
```

The Space rebuilds automatically. If model weights change, rerun
`python scripts/apin_v2/collect_deployment.py` and re-upload
`deploy/apin-models` to `dxv-404/apin-models` first.

---

## Part A — Create the Turso database  *(you)*

1. Sign up at **turso.tech** (free).
2. Install the Turso CLI, or use the web dashboard. With the CLI:
   ```bash
   turso auth login
   turso db create apin-v2
   ```
3. Get the two values the app needs:
   ```bash
   turso db show apin-v2 --url        # -> TURSO_DATABASE_URL  (libsql://…)
   turso db tokens create apin-v2     # -> TURSO_AUTH_TOKEN
   ```
4. Keep both strings — they go into the Space as secrets (Part B). **Do not
   commit them to git.**

Notes:
- You do **not** create any tables. On first boot the app runs its schema
  bootstrap (`_ensure_db`) against Turso automatically.
- Verify the free tier's current storage limit on Turso's pricing page.
  Each prediction row is ~255 KB (image + heatmap + metadata), so a few GB
  is ~10–20k predictions — ample for a demo.

---

## Part B — Create the Hugging Face Space  *(you)*

1. Sign up at **huggingface.co** (free).
2. **New → Space**. Choose **SDK: Docker**, hardware **CPU basic (free)**.
3. In the Space's **Settings → Variables and secrets**, add two **secrets**:
   - `TURSO_DATABASE_URL` = the `libsql://…` URL from Part A
   - `TURSO_AUTH_TOKEN`   = the token from Part A
4. The Space gets a free public URL: `https://<you>-<space>.hf.space`.
   That is your shareable link (HTTPS, stable). A custom domain is optional
   later (~$10/yr domain + Cloudflare in front).

---

## Part C — Deployment build  *(we do together)*

This is the next build step after Parts A & B. It produces:

1. **A curated deployment bundle.** The repo is 50+ GB of training data and
   checkpoints; only ~722 MB of runtime weights ship. A collector script
   copies just the traced runtime files (see the model-footprint analysis):
   `model3_production_v3.pt`, `sp_lora_epoch13…PRESERVED.pt`,
   `model2_production.pt`, `best_model.pt`, the OOD detector, the DINOv2
   probe head, the APIN caches, the router — plus the `scripts/apin*`,
   `scripts/apin_v2`, `scripts/ladi_net`, `scripts/model3_training`,
   `diagnosis/` code.
2. **A `Dockerfile`** for the Space — Python base, **CPU torch** (the
   `+cpu` wheel — ~190 MB, not the 2.5 GB CUDA build), the deps from
   `requirements.txt` (which now includes `libsql-client`), the bundled
   weights, and `CMD uvicorn`/the apin server entrypoint on port 7860
   (HF Spaces expose 7860).
3. **HF model upload.** The ~722 MB of weights go to the Space repo via
   git-LFS (free Spaces allow large LFS repos).

Pre-launch hardening still recommended (from earlier reviews):
- Gate `/predict/apin` and `/predict` server-side (the UI already gates
  all uploads; these two alternate API endpoints are not yet gated).
- Add a `/predict/full` rate limit — cheap insurance against abuse and the
  guest-cookie-reset loophole.

---

## Part D — First-run verification

When the Space boots:

1. Check the Space **logs** — you should see the schema bootstrap run and
   `APIN inference ready`. A bad Turso URL/token fails fast here (correct
   behaviour — you want to know immediately).
2. Open the Space URL → the inference page loads.
3. Upload a leaf → the auth modal appears → **Continue as guest** → a
   diagnosis renders.
4. **Create an account**, log in, run a prediction, open the dashboard.
5. The real test of persistence: in the Space settings, **Restart** the
   Space. Log in again with the same account — it should still exist,
   with its prediction and dashboard intact. That confirms Turso is
   holding the data, not the ephemeral Space disk.

---

## How the backend switch works (reference)

`scripts/apin_v2/auth_db.py`:

- Reads `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` at import.
- If a Turso URL is present, it builds one shared `libsql-client`
  connection and wraps it in `_ShimConn` — a minimal, `sqlite3`-compatible
  adapter (`.execute` / `.executescript` / cursor-style `.fetchone` /
  `.fetchall` / `.lastrowid` / `.rowcount`, and rows that support
  `dict(row)`, name- and index-access).
- Every one of the ~40 DB helper functions and all SQL is **unchanged** —
  Turso is SQLite-dialect, so the schema, the `image_bytes BLOB` column,
  and all dashboard widget queries port verbatim.
- `_turso_shim_test.py` exercises the whole backend against a local
  libSQL file — run it any time with
  `python scripts/apin_v2/_turso_shim_test.py`.
