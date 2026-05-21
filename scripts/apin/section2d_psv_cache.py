"""Section 2D — PSV Raw Features Cache for all 9006 Model 2 training images.

Extracts the 66 PSV features per image and stores them in a cache.
After this completes, Section 3C reads from this cache to compute
psv_calibration.json on the training subset, then a separate downstream
step applies the calibration to convert raw features → 9 calibrated
disease scores per image (Signal 3 prediction cache).

PSV is CPU-only and slow (~1 img/s). Total runtime: ~2.5 hours for 9006
images. Designed to run unattended.

Output:
  scripts/apin/caches/psv_raw_features_cache.pkl  (all 9006 raw feature dicts)
  scripts/apin/caches/psv_raw_features_fingerprint_{ts}.json
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section2d_psv_cache_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section2d")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "model2_unified_source_map.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "split_indices.json"
OUT_CACHE = CACHE_DIR / f"psv_raw_features_cache_{TIMESTAMP}.pkl"
OUT_LATEST = CACHE_DIR / "psv_raw_features_cache.pkl"
OUT_FP = CACHE_DIR / f"psv_raw_features_fingerprint_{TIMESTAMP}.json"

# Resume support: if a partial cache exists, load it and skip already-done rows
RESUME_FROM_LATEST = True


def load_artifacts():
    """Load CSV + splits, build row->split map."""
    df = pd.read_csv(CSV_PATH)
    assert "is_recomposed" in df.columns
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    row_to_split = {}
    for k, idxs in splits.items():
        for idx in idxs:
            row_to_split[int(idx)] = k
    return df, row_to_split


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 2D -- PSV Raw Features Cache (9006 images, ~2.5h)")
    logger.info("=" * 70)

    from scripts.apin.constants import MODEL2_CLASS_ORDER

    df, row_to_split = load_artifacts()
    class_to_idx = {c: i for i, c in enumerate(MODEL2_CLASS_ORDER)}
    logger.info(f"CSV: {len(df)} rows")

    # Resume: load existing cache if present
    cache = {}
    if RESUME_FROM_LATEST and OUT_LATEST.exists():
        logger.info(f"Found existing cache at {OUT_LATEST.name}, loading for resume")
        try:
            with open(OUT_LATEST, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"  Loaded {len(cache)} existing entries")
        except Exception as e:
            logger.warning(f"  Could not load: {e} — starting fresh")
            cache = {}

    rows_to_process = [i for i in range(len(df)) if i not in cache]
    logger.info(f"Rows to process: {len(rows_to_process)}/{len(df)}")
    if not rows_to_process:
        logger.info("Nothing to do — cache is complete")
        return 0

    from scripts.psv.feature_extractor import extract_all_features

    # Process loop
    failed = 0
    save_every = 500  # checkpoint every N images
    last_save = time.time()
    save_interval_sec = 300  # also checkpoint every 5 minutes

    t_start = time.time()
    pbar = tqdm(rows_to_process, desc="PSV cache")
    for i in pbar:
        row = df.iloc[i]
        path = row.get("clahe_path", row["image_path"])
        if not isinstance(path, str) or not Path(path).exists():
            path = row["image_path"]
        try:
            img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            result = extract_all_features(img)
            cache[int(i)] = {
                "features": result.features,
                "extraction_success": True,
                "n_failed_features": len(result.failed_features) if hasattr(result, "failed_features") else 0,
                "class_name": str(row["class_name"]),
                "true_class_idx": class_to_idx[row["class_name"]],
                "source_dataset": str(row["source_dataset"]),
                "is_field_photo": bool(row["is_field_photo"]),
                "split": row_to_split[int(i)],
                "is_recomposed": bool(row["is_recomposed"]),
            }
        except Exception as e:
            failed += 1
            cache[int(i)] = {
                "features": {},
                "extraction_success": False,
                "n_failed_features": 66,
                "class_name": str(row["class_name"]),
                "true_class_idx": class_to_idx[row["class_name"]],
                "source_dataset": str(row["source_dataset"]),
                "is_field_photo": bool(row["is_field_photo"]),
                "split": row_to_split[int(i)],
                "is_recomposed": bool(row["is_recomposed"]),
                "error": str(e),
            }

        # Periodic checkpoint
        elapsed = time.time() - last_save
        if (len(cache) % save_every == 0) or (elapsed > save_interval_sec):
            with open(OUT_LATEST, "wb") as f:
                pickle.dump(cache, f)
            last_save = time.time()

    elapsed = time.time() - t_start
    n_done = len(cache)
    logger.info(f"\nExtraction complete: {n_done}/{len(df)} entries "
                f"({failed} failed), {elapsed:.0f}s ({n_done/elapsed:.2f} img/s)")

    # Final save
    with open(OUT_CACHE, "wb") as f:
        pickle.dump(cache, f)
    with open(OUT_LATEST, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"Saved cache: {OUT_LATEST.name} "
                f"({OUT_LATEST.stat().st_size / 1e6:.2f} MB)")

    # Fingerprint
    n_success = sum(1 for e in cache.values() if e["extraction_success"])
    n_fail = sum(1 for e in cache.values() if not e["extraction_success"])
    n_per_class = {}
    for e in cache.values():
        n_per_class[e["class_name"]] = n_per_class.get(e["class_name"], 0) + 1

    fp = {
        "section": "2D",
        "timestamp": TIMESTAMP,
        "n_total_rows": len(df),
        "n_cached": len(cache),
        "n_extraction_success": n_success,
        "n_extraction_failed": n_fail,
        "n_per_class": n_per_class,
        "feature_extractor": "scripts.psv.feature_extractor.extract_all_features",
        "n_features_per_image": 66,
        "preprocessing_branch": "A (LAB-CLAHE pre-applied via clahe_path; same as Signal 1)",
        "cache_path": str(OUT_LATEST.relative_to(PROJECT_ROOT)),
        "elapsed_seconds": elapsed,
        "throughput_img_per_sec": n_done / elapsed if elapsed > 0 else 0,
    }
    with open(OUT_FP, "w") as f:
        json.dump(fp, f, indent=2)
    logger.info(f"Fingerprint: {OUT_FP.name}")
    logger.info("=" * 70)
    logger.info("APIN SECTION 2D -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
