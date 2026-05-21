# APIN — Plant Pathology Journal

A leaf-disease diagnosis web app for **tomato**, **okra (ladies finger)** and
**brassica (broccoli / cabbage)**. Photograph a leaf and get a calibrated
diagnosis with a Grad-CAM heatmap, severity, treatment and prevention advice —
kept in a personal field-notebook dashboard.

**Live demo:** https://dxv-404-apin.hf.space

---

## What it does

- **Diagnose** — upload a leaf photo, get the disease, a calibrated confidence,
  a severity reading and a Grad-CAM heatmap showing where the model looked.
- **Advise** — treatment, prevention and urgency for the detected (crop, disease).
- **Track** — a private dashboard per account: prediction history, a disease
  ledger, an activity calendar and a treatment log.
- **Guests** — 3 free diagnoses without an account; sign up for the dashboard.

## Architecture

The diagnosis is an ensemble: a crop **router**, a **Model 2** specialist, an
EfficientNet head, a **DINOv2** probe with an out-of-distribution detector, a
tomato single-pass-LoRA specialist (**Model 3**), and a **PSV** signal stack
fused through a calibrated stacking layer. Temperature scaling keeps the
reported confidence honest.

```
scripts/apin/             APIN ensemble server + inference (okra/brassica/chilli)
scripts/apin_v2/          FastAPI app — auth, dashboard, Field-Notes UI
scripts/ladi_net/         LADI-Net tomato pipeline (Model 3 + LoRA)
scripts/model3_training/  Model 3 architecture
scripts/dinov2_probe/     DINOv2 feature probe + OOD detector
scripts/psv/              Plant Signal Vector feature stack
app/                      Model 2 / Model 3 / router config modules
deploy/                   Hugging Face Space deployment tooling
```

## Running it

Model weights (~722 MB) are **not** in this repo — they live in the public
[`dxv-404/apin-models`](https://huggingface.co/dxv-404/apin-models) model repo
and are pulled automatically at build time.

### Docker (recommended — mirrors the live deployment)

```bash
git clone https://github.com/Dxv-404/Plant-disease-detection-for-brocolli-and-ladies-finger.git
cd Plant-disease-detection-for-brocolli-and-ladies-finger

python deploy/stage_space.py          # assemble deploy/apin-space/
docker build -t apin deploy/apin-space
docker run -p 7860:7860 \
  -e TURSO_DATABASE_URL="libsql://<your-db>.turso.io" \
  -e TURSO_AUTH_TOKEN="<your-token>" \
  apin
```

Open http://localhost:7860.

### Local (development)

```bash
python -m venv venv && source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# Fetch the model weights into the working tree:
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('dxv-404/apin-models', repo_type='model', local_dir='.')"

python scripts/apin_v2/apin_server.py --host 0.0.0.0 --port 8766
```

Open http://localhost:8766.

## Storage

Accounts, predictions and uploaded-image BLOBs persist in an external
**Turso** (libSQL) database, selected by the `TURSO_DATABASE_URL` /
`TURSO_AUTH_TOKEN` environment variables. With them unset, the app falls back
to a local SQLite file — handy for development. See
[`scripts/apin_v2/DEPLOYMENT.md`](scripts/apin_v2/DEPLOYMENT.md) for the full
Hugging Face Space + Turso setup.

## Deployment

`deploy/` holds everything for the free Hugging Face Space deployment:
`stage_space.py` assembles the bundle, `space_files/Dockerfile` builds the
image, `space_files/download_weights.py` pulls the weights, and
`watch_space.py` monitors the build. Full guide in
[`scripts/apin_v2/DEPLOYMENT.md`](scripts/apin_v2/DEPLOYMENT.md).
