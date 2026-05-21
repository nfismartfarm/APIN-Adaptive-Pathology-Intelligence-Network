"""Stage the APIN v2 runtime model weights for upload to the HF model repo.

The repo holds 50+ GB of training data and dated checkpoint history; the
deployed app loads exactly one checkpoint per model. This script copies
only the traced runtime artifacts into `deploy/apin-models/`, preserving
each file's path relative to the project root so the Dockerfile can place
them exactly where the inference code expects.

Run:  python scripts/apin_v2/collect_deployment.py
Then: hf upload dxv-404/apin-models deploy/apin-models . --repo-type model
"""
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE = os.path.join(ROOT, "deploy", "apin-models")

# ── Exact runtime weight files (traced from the inference code) ───────────
#   tomato_pipeline.py : V3_CKPT, SP_LORA_CKPT, SP_BANK + 4 calibration JSONs
#   apin/inference.py  : model2_production, best_model, router, dinov2 head
#                        + scaler + config, OOD detector
RUNTIME_FILES = [
    # APIN ensemble
    "models/model2_specialist/model2_production.pt",
    "models/best_model.pt",
    "models/router/router_best.pt",
    # Tomato specialist + Model 3
    "models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt",
    "scripts/model3_training/checkpoints/model3_production_v3.pt",
    # DINOv2 probe head + OOD detector
    "scripts/dinov2_probe/results/dinov2_nonlinear_head_20260416_204427.pt",
    "scripts/dinov2_probe/results/dinov2_nonlinear_head_scaler_20260416_204427.pkl",
    "scripts/dinov2_probe/results/dinov2_nonlinear_head_config_20260416_204427.json",
    "scripts/dinov2_probe/results/ood_detector_min_over_all_20260417_111933.pkl",
    # Model 3 / specialist calibration artifacts
    "data/specialist/model3/prototype_bank_sp_lora_ep13.pt",
    "data/specialist/model3/phase3_calibration_sp_lora_ep13.json",
    "data/specialist/model3/phase3_tier_thresholds_sp_lora_ep13.json",
    "data/specialist/model3/phase3_calibration_v3_tomato.json",
    "data/specialist/model3/phase3_tier_thresholds_v3_tomato.json",
    # Diagnosis content
    "diagnosis/diagnosis_lookup.json",
]

# The APIN caches dir holds the canonical (non-timestamped) signal artifacts
# the ensemble loads at runtime — apin_stacking_mlp.pt, apin_calibration.json,
# psv_calibration.json, reliability matrices, signal caches. Every dated
# duplicate (filename contains '_2026…') is training history and is skipped.
CACHES_DIR = "scripts/apin/caches"


def _copy(rel_path):
    src = os.path.join(ROOT, rel_path)
    dst = os.path.join(STAGE, rel_path)
    if not os.path.exists(src):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return os.path.getsize(dst)


def main():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE, exist_ok=True)

    total = 0
    missing = []
    print("=" * 64)
    print("Staging runtime weights -> deploy/apin-models/")
    print("=" * 64)

    for rel in RUNTIME_FILES:
        size = _copy(rel)
        if size is None:
            missing.append(rel)
            print(f"  MISSING   {rel}")
        else:
            total += size
            print(f"  {size/1048576:8.1f} MB  {rel}")

    # Caches dir — canonical files only (skip dated duplicates).
    caches_src = os.path.join(ROOT, CACHES_DIR)
    n_cache = 0
    if os.path.isdir(caches_src):
        for fname in sorted(os.listdir(caches_src)):
            if "_2026" in fname:            # dated duplicate — training history
                continue
            fpath = os.path.join(caches_src, fname)
            if not os.path.isfile(fpath):
                continue
            size = _copy(os.path.join(CACHES_DIR, fname))
            total += size
            n_cache += 1
        print(f"  {'':8s} +  {n_cache} canonical files from {CACHES_DIR}/")
    else:
        missing.append(CACHES_DIR)
        print(f"  MISSING   {CACHES_DIR}/")

    print("-" * 64)
    print(f"  TOTAL staged: {total/1048576:.1f} MB")
    print(f"  Location:     {STAGE}")
    if missing:
        print()
        print(f"  ⚠ {len(missing)} MISSING item(s) — deployment will be incomplete:")
        for m in missing:
            print(f"      {m}")
        sys.exit(1)
    print()
    print("  All runtime weights staged. Next:")
    print("    hf upload dxv-404/apin-models deploy/apin-models . --repo-type model")
    sys.exit(0)


if __name__ == "__main__":
    main()
