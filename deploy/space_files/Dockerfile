# APIN v2 — Plant Pathology Journal · Hugging Face Space (Docker SDK)
# ---------------------------------------------------------------------------
# Runs the FastAPI app on :7860 (the port HF Spaces expose).
#   · Model weights (~722 MB) are pulled at BUILD time from the public
#     dxv-404/apin-models repo — they are not committed to this Space.
#   · Accounts, predictions and uploaded-image BLOBs persist in an external
#     Turso database, configured via the TURSO_DATABASE_URL / TURSO_AUTH_TOKEN
#     Space secrets. The Space's own filesystem is ephemeral.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# ── System libraries OpenCV links against at runtime + build toolchain ─────
# build-essential (gcc/g++/make) is required because some transitive deps
# (e.g. stringzilla, pulled in by albumentations→albucore) publish only an
# sdist for newer releases and must be compiled from C source at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (Hugging Face Spaces convention: uid 1000) ───────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
# CPU-only torch FIRST: the +cpu wheel is ~190 MB, not the ~2.5 GB CUDA build.
# Everything else then resolves against it (pip sees torch already satisfied).
COPY --chown=user requirements.txt .
# torch/torchvision pinned to the dev environment (2.11.0 / 0.26.0) for a
# reproducible build. NOTE: the determinant of CORRECT Model-2 output is the
# transformers version (see requirements.txt — must stay on 4.57.x, NOT 5.x);
# torch version does not change the tier/heatmap. Kept pinned to match dev.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Model weights — pulled from the public model repo at build time ────────
COPY --chown=user download_weights.py .
RUN python download_weights.py

# ── Pre-warm the DINOv2 backbone so the first request is not slow ──────────
# timm fetches the pretrained ViT-S/14 once; bake it into the image layer.
RUN python -c "import timm; timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224)" \
    || echo "DINOv2 pre-warm skipped — it will download on the first request"

# ── Application code (last layers — this is what changes most often) ───────
# `app/` carries the Model 2/3 + router config modules that scripts/ imports
# as `from app.config_* import ...`; `scripts/` is the rest of the codebase.
# `logo.png` lands at /app/logo.png — the weekly PDF report reads it from
# the project root to draw the report cover logo.
# `_qa_tmp/` carries the 5 pipeline-atlas JSON files read at runtime by
# /api/pipeline_data/{slug} (router, tomato, chilli, forward_pass, phase_d).
# The folder is in the Space's git repo but was previously not copied into
# the Docker image — that made every module deep-dive on /pipeline 404.
COPY --chown=user logo.png .
COPY --chown=user app/ app/
COPY --chown=user scripts/ scripts/
COPY --chown=user _qa_tmp/ _qa_tmp/

EXPOSE 7860
CMD ["python", "scripts/apin_v2/apin_server.py", "--host", "0.0.0.0", "--port", "7860"]
