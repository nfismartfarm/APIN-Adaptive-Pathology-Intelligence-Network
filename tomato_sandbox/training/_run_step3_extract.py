"""
T-PHASE-F0-CLASSIFIER Step 3 — main-thread feature extraction over 259 images.

Composition:
  - 160 train_subset + 43 held_out_subset (source-stratified carve of 203 field_val, seed=42)
  - 36 OOD okra/brassica from data/specialist/model2/cleaned/ (seed=43, 4 per folder × 9)
  - 20 synthetic noise (seed=44, 7 Gaussian + 7 solid + 6 scrambled, 224×224)

Outputs:
  tomato_sandbox/phase_f0_calibration/_classifier_training/features.npz
  tomato_sandbox/phase_f0_calibration/_classifier_training/extraction_log.json

spec: section 12.9 lines 3408-3442 — training procedure context
spec: section 4.5 — model3_unified_source_map.csv as authoritative source map
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root on sys.path for absolute imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tomato_sandbox.api.model_loaders import load_v3_model, load_lora_model
from tomato_sandbox.orchestrator.pipeline import PipelineContext
from tomato_sandbox.training.extract_features import (
    extract_features_for_training,
    extract_features_no_iqa,
)

# ── Paths ────────────────────────────────────────────────────────────────
SPLIT_INDICES = _PROJECT_ROOT / "data" / "specialist" / "model3" / "split_indices.json"
SOURCE_MAP_CSV = _PROJECT_ROOT / "data" / "specialist" / "model3" / "model3_unified_source_map.csv"
MODEL2_CLEANED = _PROJECT_ROOT / "data" / "specialist" / "model2" / "cleaned"

OUT_DIR = _PROJECT_ROOT / "tomato_sandbox" / "phase_f0_calibration" / "_classifier_training"
OUT_NPZ = OUT_DIR / "features.npz"
OUT_LOG = OUT_DIR / "extraction_log.json"

# ── Constants ────────────────────────────────────────────────────────────
SEED_SPLIT = 42
SEED_OOD = 43
SEED_NOISE = 44

OOD_FOLDERS = [
    "brassica_alternaria",
    "brassica_black_rot",
    "brassica_downy_mildew",
    "brassica_healthy",
    "okra_cercospora",
    "okra_enation",
    "okra_healthy",
    "okra_powdery_mildew",
    "okra_yvmv",
]
OOD_PER_FOLDER = 4
OOD_TOTAL = OOD_PER_FOLDER * len(OOD_FOLDERS)  # 36
OOD_BYPASS_CAP = 7  # 20% of 36 (Refinement 2)

NOISE_GAUSSIAN = 7
NOISE_SOLID = 7
NOISE_SCRAMBLED = 6
NOISE_TOTAL = NOISE_GAUSSIAN + NOISE_SOLID + NOISE_SCRAMBLED  # 20
NOISE_SIZE = 224

# Stage 1 labels
STAGE1_HEALTHY = 0
STAGE1_DISEASED = 1
STAGE1_OOD = 2

# Stage 2 labels (canonical 5-disease index space per S12.10)
STAGE2_FOLIAR = 0
STAGE2_SEPTORIA = 1
STAGE2_LATE_BLIGHT = 2
STAGE2_YLCV = 3
STAGE2_MOSAIC = 4
STAGE2_NA = -1  # sentinel for non-diseased rows

CLASS_TO_STAGE2 = {
    "tomato_foliar_spot": STAGE2_FOLIAR,
    "tomato_septoria_leaf_spot": STAGE2_SEPTORIA,
    "tomato_late_blight": STAGE2_LATE_BLIGHT,
    "tomato_yellow_leaf_curl_virus": STAGE2_YLCV,
    "tomato_mosaic_virus": STAGE2_MOSAIC,
}


def carve_field_val(records: list[dict], rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """160 train_subset + 43 held_out_subset, source-stratified by (class, source)."""
    by_cell = defaultdict(list)
    for r in records:
        by_cell[(r["class_name"], r["source_dataset"])].append(r)
    train_records, held_records = [], []
    for cell, items in by_cell.items():
        rng.shuffle(items)
        n = len(items)
        # proportional allocation: round(n * 160/203) to train, rest to held
        n_train = int(round(n * 160 / 203))
        # ensure at least 1 in each side if cell size >= 2
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        else:
            # cell of size 1: assign to train (preserving training class coverage)
            n_train = 1
        train_records.extend(items[:n_train])
        held_records.extend(items[n_train:])
    # Adjust to exact 160/43: if off, transfer between sides
    while len(train_records) > 160:
        held_records.append(train_records.pop())
    while len(train_records) < 160 and held_records:
        train_records.append(held_records.pop())
    return train_records, held_records


def select_ood_images(rng: np.random.Generator) -> list[dict]:
    """4 images per OOD folder × 9 folders = 36, deterministic."""
    selections = []
    for folder in OOD_FOLDERS:
        folder_path = MODEL2_CLEANED / folder
        files = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.JPG")) \
              + sorted(folder_path.glob("*.png")) + sorted(folder_path.glob("*.jpeg"))
        if len(files) < OOD_PER_FOLDER * 3:
            # ensure pool large enough for IQA-failure replacements (3x oversample)
            if len(files) < OOD_PER_FOLDER:
                raise RuntimeError(f"OOD folder {folder} has only {len(files)} images")
        # deterministic shuffle, take first OOD_PER_FOLDER as primary draw + rest as fallback pool
        idx = rng.permutation(len(files))
        primary = [files[i] for i in idx[:OOD_PER_FOLDER]]
        fallback_pool = [files[i] for i in idx[OOD_PER_FOLDER:OOD_PER_FOLDER * 4]]  # up to 12 fallbacks
        for p in primary:
            selections.append({
                "image_path": str(p),
                "folder": folder,
                "source_dataset": f"model2_cleaned_{folder}",
                "fallback_pool": [str(x) for x in fallback_pool],
            })
    return selections


def generate_noise_images(rng: np.random.Generator) -> list[dict]:
    """20 synthetic noise: 7 Gaussian + 7 solid + 6 scrambled, 224×224."""
    images = []
    # Gaussian RGB
    for i in range(NOISE_GAUSSIAN):
        arr = rng.normal(loc=128, scale=64, size=(NOISE_SIZE, NOISE_SIZE, 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        images.append({
            "pil": Image.fromarray(arr, "RGB"),
            "path_sentinel": f"synthetic_noise_gaussian_{i:02d}",
            "source_dataset": "synthetic_noise_gaussian",
        })
    # Solid color
    for i in range(NOISE_SOLID):
        color = rng.integers(0, 256, size=3).astype(np.uint8)
        arr = np.broadcast_to(color, (NOISE_SIZE, NOISE_SIZE, 3)).copy()
        images.append({
            "pil": Image.fromarray(arr, "RGB"),
            "path_sentinel": f"synthetic_noise_solid_{i:02d}",
            "source_dataset": "synthetic_noise_solid",
        })
    # Scrambled: Gaussian seed + 16x16 block permutation
    for i in range(NOISE_SCRAMBLED):
        arr = rng.normal(loc=128, scale=80, size=(NOISE_SIZE, NOISE_SIZE, 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        # 14x14 blocks of 16x16
        blocks = arr.reshape(14, 16, 14, 16, 3).transpose(0, 2, 1, 3, 4)  # (14,14,16,16,3)
        flat = blocks.reshape(-1, 16, 16, 3)
        perm = rng.permutation(flat.shape[0])
        flat = flat[perm]
        arr = flat.reshape(14, 14, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(NOISE_SIZE, NOISE_SIZE, 3)
        images.append({
            "pil": Image.fromarray(arr, "RGB"),
            "path_sentinel": f"synthetic_noise_scrambled_{i:02d}",
            "source_dataset": "synthetic_noise_scrambled",
        })
    return images


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.perf_counter()

    print("=" * 60)
    print("Step 3: feature extraction over 259 images")
    print("=" * 60)

    # ── Load split + source map ─────────────────────────────────────────
    split = json.loads(SPLIT_INDICES.read_text())
    field_val_paths = split["field_val"]
    src_df = pd.read_csv(SOURCE_MAP_CSV)
    fv_df = src_df[src_df["image_path"].isin(field_val_paths)].copy()
    print(f"Field_val matched in source_map: {len(fv_df)}/{len(field_val_paths)}")
    if len(fv_df) != 203:
        raise RuntimeError(f"field_val join lost rows: {len(fv_df)} vs 203")

    fv_records = fv_df[["image_path", "class_name", "source_dataset"]].to_dict("records")

    # ── Build 160/43 carve ───────────────────────────────────────────────
    rng_split = np.random.default_rng(SEED_SPLIT)
    train_recs, held_recs = carve_field_val(fv_records, rng_split)
    print(f"Carve: {len(train_recs)} train_subset + {len(held_recs)} held_out_subset")

    # ── Select OOD images ───────────────────────────────────────────────
    rng_ood = np.random.default_rng(SEED_OOD)
    ood_selections = select_ood_images(rng_ood)
    print(f"OOD okra/brassica selected: {len(ood_selections)} (4 per × 9 folders)")

    # ── Generate noise ──────────────────────────────────────────────────
    rng_noise = np.random.default_rng(SEED_NOISE)
    noise_images = generate_noise_images(rng_noise)
    print(f"Synthetic noise generated: {len(noise_images)} (G/S/Scr = {NOISE_GAUSSIAN}/{NOISE_SOLID}/{NOISE_SCRAMBLED})")

    # ── Load models + build PipelineContext ─────────────────────────────
    print("\nLoading v3 model...")
    v3_model, _v3_meta = load_v3_model()
    print("Loading LoRA model...")
    lora_model, _lora_meta = load_lora_model()
    ctx = PipelineContext(v3_model=v3_model, lora_model=lora_model)

    # ── Extract features ─────────────────────────────────────────────────
    total = len(train_recs) + len(held_recs) + len(ood_selections) + len(noise_images)
    assert total == 259, f"Expected 259 total, got {total}"

    features = np.zeros((total, 19), dtype=np.float32)
    y_stage1 = np.full(total, -1, dtype=np.int64)
    y_stage2 = np.full(total, STAGE2_NA, dtype=np.int64)
    source_per_image = np.empty(total, dtype=object)
    partition = np.empty(total, dtype=object)
    image_path = np.empty(total, dtype=object)
    forward_succeeded = np.zeros((total, 3), dtype=bool)
    iqa_bypassed = np.zeros(total, dtype=bool)

    audit_log = []
    failures = []
    bypass_count = 0

    def _stage1_label(class_name: str) -> int:
        return STAGE1_HEALTHY if class_name == "tomato_healthy" else STAGE1_DISEASED

    def _process_image(idx: int, pil: Image.Image, partition_label: str,
                       class_name: str | None, src: str, path_label: str,
                       allow_bypass: bool, force_no_iqa: bool):
        """Run extraction via uniform IQA-bypass path (DEC-060 sub-decision).

        DEC-060 sub-decision: training-time feature extraction bypasses IQA
        for all 259 samples uniformly.  IQA gate is calibrated for
        inference-time user-photo quality protection, rejecting ~44% of
        available training data including images the classifier needs to
        learn marginal-quality decision boundaries from.  Spec S12.7
        degraded-mode training design requires full-distribution training
        data.  Refinement-2 OOD bypass cap retired under this uniform policy.

        The allow_bypass / force_no_iqa flags are retained for signature
        stability with the audit log; both paths now route through
        extract_features_no_iqa.
        """
        nonlocal bypass_count
        vec, err = extract_features_no_iqa(pil, ctx)
        used_bypass = True  # uniform bypass: always True per DEC-060
        return vec, err, used_bypass

    def _populate(idx: int, vec: np.ndarray, signals_succeeded_guess: tuple,
                  partition_label: str, class_name: str | None, src: str,
                  path_label: str, used_bypass: bool):
        features[idx] = vec
        partition[idx] = partition_label
        source_per_image[idx] = src
        image_path[idx] = path_label
        iqa_bypassed[idx] = used_bypass
        # forward_succeeded inferred from non-zero blocks (best-effort; not exact)
        # build_classifier_input zeros entire block on signal failure (0:6 v3, 6:12 lora)
        # For PSV (12:15+17), zeros-block ⇒ failed
        forward_succeeded[idx, 0] = bool((vec[0:6] != 0).any())
        forward_succeeded[idx, 1] = bool((vec[6:12] != 0).any())
        forward_succeeded[idx, 2] = bool((vec[12:18] != 0).any())  # incl. JSD slot inferred separately
        if class_name is None:
            y_stage1[idx] = STAGE1_OOD
            y_stage2[idx] = STAGE2_NA
        else:
            y_stage1[idx] = _stage1_label(class_name)
            y_stage2[idx] = CLASS_TO_STAGE2.get(class_name, STAGE2_NA)

    idx = 0
    print("\nExtracting train_subset...")
    t0 = time.perf_counter()
    for r in train_recs:
        path = r["image_path"]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            failures.append({"idx": idx, "path": path, "stage": "decode", "err": str(e)})
            idx += 1
            continue
        vec, err, used = _process_image(idx, pil, "train_subset", r["class_name"],
                                          r["source_dataset"], path, allow_bypass=False,
                                          force_no_iqa=False)
        if vec is None:
            failures.append({"idx": idx, "path": path, "stage": "extract", "err": err,
                              "partition": "train_subset"})
        else:
            _populate(idx, vec, None, "train_subset", r["class_name"],
                      r["source_dataset"], path, used)
        audit_log.append({"idx": idx, "partition": "train_subset",
                          "class_name": r["class_name"], "source": r["source_dataset"],
                          "success": vec is not None, "err": err, "iqa_bypassed": used})
        idx += 1
    print(f"  train_subset done in {time.perf_counter() - t0:.1f}s")

    print("Extracting held_out_subset...")
    t0 = time.perf_counter()
    for r in held_recs:
        path = r["image_path"]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            failures.append({"idx": idx, "path": path, "stage": "decode", "err": str(e)})
            idx += 1
            continue
        vec, err, used = _process_image(idx, pil, "held_out_subset", r["class_name"],
                                          r["source_dataset"], path, allow_bypass=False,
                                          force_no_iqa=False)
        if vec is None:
            failures.append({"idx": idx, "path": path, "stage": "extract", "err": err,
                              "partition": "held_out_subset"})
        else:
            _populate(idx, vec, None, "held_out_subset", r["class_name"],
                      r["source_dataset"], path, used)
        audit_log.append({"idx": idx, "partition": "held_out_subset",
                          "class_name": r["class_name"], "source": r["source_dataset"],
                          "success": vec is not None, "err": err, "iqa_bypassed": used})
        idx += 1
    print(f"  held_out_subset done in {time.perf_counter() - t0:.1f}s")

    print("Extracting OOD okra/brassica...")
    t0 = time.perf_counter()
    for sel in ood_selections:
        path = sel["image_path"]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            failures.append({"idx": idx, "path": path, "stage": "decode", "err": str(e)})
            idx += 1
            continue
        vec, err, used = _process_image(idx, pil, "ood", None, sel["source_dataset"],
                                          path, allow_bypass=True, force_no_iqa=False)
        # If IQA reject and replacement attempts fail, vec=None; try fallback pool
        attempt = 0
        while vec is None and err and err.startswith("iqa_reject") and attempt < len(sel["fallback_pool"]) and bypass_count < OOD_BYPASS_CAP:
            fb_path = sel["fallback_pool"][attempt]
            attempt += 1
            try:
                pil_fb = Image.open(fb_path).convert("RGB")
                vec, err, used = _process_image(idx, pil_fb, "ood", None, sel["source_dataset"],
                                                  fb_path, allow_bypass=True, force_no_iqa=False)
                if vec is not None:
                    path = fb_path
                    pil = pil_fb
            except Exception:
                continue
        if vec is None:
            failures.append({"idx": idx, "path": path, "stage": "extract", "err": err,
                              "partition": "ood_okra_brassica"})
        else:
            _populate(idx, vec, None, "ood", None, sel["source_dataset"], path, used)
        audit_log.append({"idx": idx, "partition": "ood",
                          "class_name": None, "source": sel["source_dataset"],
                          "success": vec is not None, "err": err, "iqa_bypassed": used})
        idx += 1
    print(f"  OOD okra/brassica done in {time.perf_counter() - t0:.1f}s; bypass_count={bypass_count}")

    print("Extracting synthetic noise...")
    t0 = time.perf_counter()
    for n_img in noise_images:
        path_label = n_img["path_sentinel"]
        # noise always uses no-IQA path
        vec, err, used = _process_image(idx, n_img["pil"], "ood", None,
                                          n_img["source_dataset"], path_label,
                                          allow_bypass=False, force_no_iqa=True)
        if vec is None:
            failures.append({"idx": idx, "path": path_label, "stage": "extract", "err": err,
                              "partition": "ood_noise"})
        else:
            _populate(idx, vec, None, "ood", None, n_img["source_dataset"], path_label, used)
        audit_log.append({"idx": idx, "partition": "ood",
                          "class_name": None, "source": n_img["source_dataset"],
                          "success": vec is not None, "err": err, "iqa_bypassed": used})
        idx += 1
    print(f"  noise done in {time.perf_counter() - t0:.1f}s")

    elapsed = time.perf_counter() - t_global
    print(f"\nTotal extraction time: {elapsed:.1f}s")
    print(f"Failures: {len(failures)}")
    print(f"OOD IQA-bypasses used: {bypass_count}/{OOD_BYPASS_CAP}")

    # ── Save outputs ────────────────────────────────────────────────────
    metadata = {
        "seeds": {"split": SEED_SPLIT, "ood": SEED_OOD, "noise": SEED_NOISE},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_citations": ["S12.2:3169-3244", "S12.7:3348-3373", "S12.9:3408-3442"],
        "counts": {
            "train_subset": len(train_recs),
            "held_out_subset": len(held_recs),
            "ood_okra_brassica": len(ood_selections),
            "ood_noise": len(noise_images),
            "total": total,
            "failures": len(failures),
            "iqa_bypassed_total": int(iqa_bypassed.sum()),
        },
        "ood_folders": OOD_FOLDERS,
        "elapsed_seconds": round(elapsed, 2),
        # DEC-060 sub-decision audit fields
        "training_iqa_policy": "uniform_bypass",
        "inference_iqa_policy": "S6.6_filter",
        "distribution_shift_note": (
            "Training feature distribution includes IQA-rejected images "
            "(low resolution / wet leaf / etc.); inference classifier sees "
            "only IQA-passing images. Mild shift; acceptable per DEC-060 "
            "rationale: full-distribution training data is required for "
            "spec S12.7 degraded-mode robustness, and ylcv (n=3) and "
            "mosaic (n=8) classes would become statistically unfit under "
            "44% IQA-induced rejection."
        ),
        "refinement_2_status": (
            "Refinement-2 OOD okra/brassica IQA-bypass cap (max 7 of 36) "
            "retired under uniform training-time IQA bypass."
        ),
    }

    np.savez_compressed(
        OUT_NPZ,
        features=features,
        y_stage1=y_stage1,
        y_stage2=y_stage2,
        source_per_image=source_per_image,
        partition=partition,
        image_path=image_path,
        forward_succeeded_per_signal=forward_succeeded,
        iqa_bypassed=iqa_bypassed,
        metadata=np.array(json.dumps(metadata), dtype=object),
    )
    print(f"\nSaved: {OUT_NPZ}")

    OUT_LOG.write_text(json.dumps({"metadata": metadata, "failures": failures,
                                     "audit_log": audit_log}, indent=2))
    print(f"Saved: {OUT_LOG}")

    # Sanity checks (Refinement 3)
    print("\n=== Sanity checks ===")
    n_nan = int(np.isnan(features).any(axis=1).sum())
    n_zero = int((features == 0).all(axis=1).sum())
    print(f"NaN rows: {n_nan} (target: 0)")
    print(f"All-zero rows: {n_zero} (target: < 5% = {int(0.05 * total)})")
    print(f"y_stage1 distribution: healthy={int((y_stage1==0).sum())}, "
          f"diseased={int((y_stage1==1).sum())}, ood={int((y_stage1==2).sum())}, "
          f"unset={int((y_stage1==-1).sum())}")
    print(f"y_stage2 (diseased only): {dict(zip(*np.unique(y_stage2[y_stage2 >= 0], return_counts=True)))}")
    print(f"partition: train={int((partition=='train_subset').sum())}, "
          f"held={int((partition=='held_out_subset').sum())}, "
          f"ood={int((partition=='ood').sum())}")
    print(f"forward_succeeded: v3={forward_succeeded[:,0].sum()}, "
          f"lora={forward_succeeded[:,1].sum()}, psv={forward_succeeded[:,2].sum()} (out of {total})")

    if n_nan > 0:
        print("STOP: NaN rows present")
        return 1
    if n_zero > int(0.05 * total):
        print(f"STOP: too many zero rows ({n_zero} > {int(0.05 * total)})")
        return 1
    if forward_succeeded[:, 0].sum() == 0 or forward_succeeded[:, 0].sum() == total:
        print("STOP: v3 succeed-rate degenerate (all 0 or all 1)")
        return 1

    print("\nAll sanity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
