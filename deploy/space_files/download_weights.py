"""Build-time step: pull the APIN runtime weights into the image.

The dxv-404/apin-models repo holds exactly the ~722 MB of artifacts the
deployed app loads (the model3/specialist/router/DINOv2 checkpoints, the
APIN signal caches, the calibration JSONs, the diagnosis lookup). Each
file is stored under its path relative to the project root, so downloading
with local_dir="/app" drops every file exactly where the inference code
expects it (e.g. /app/models/best_model.pt, /app/diagnosis/...).

The repo is public, so no token is needed; if it is ever made private,
set an HF_TOKEN build secret and it is picked up automatically.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "dxv-404/apin-models"

print(f"Downloading runtime weights from {REPO} -> /app ...", flush=True)
path = snapshot_download(
    repo_id=REPO,
    repo_type="model",
    local_dir="/app",
    token=os.environ.get("HF_TOKEN"),  # None == anonymous (repo is public)
)
print(f"APIN model weights staged into {path}", flush=True)

# Fail the build loudly if the headline checkpoint is missing — better to
# know at build time than to boot a Space that 500s on every request.
must_exist = [
    "/app/models/best_model.pt",
    "/app/models/model2_specialist/model2_production.pt",
    "/app/scripts/model3_training/checkpoints/model3_production_v3.pt",
    "/app/diagnosis/diagnosis_lookup.json",
]
missing = [p for p in must_exist if not os.path.exists(p)]
if missing:
    print("ERROR: expected weight files are missing after download:", flush=True)
    for m in missing:
        print(f"  {m}", flush=True)
    sys.exit(1)
print("All headline weight files present.", flush=True)
