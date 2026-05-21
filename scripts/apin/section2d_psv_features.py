"""
Section 2D (Phase 1) -- Extract 66 PSV features for all 9006 Model 2 images.

Runs on CPU (~70 min unattended at ~461ms/image). Saves RAW feature dicts to
a pickle cache; calibration and 9-class score computation happen in a later
phase (section2d_psv_scores.py) once psv_calibration.json is built.

Output:
  scripts/apin/caches/psv_raw_features_cache.pkl
    Keyed by CSV row index -> dict with:
      'features': dict[str, float] — 66 raw feature values
      'psv_confidence': float
      'psv_flags': dict — any IQA flags that fired
      'extraction_time_ms': float
      'class_name', 'source_dataset', 'is_field_photo', 'split',
      'is_recomposed', 'true_class_idx'

  scripts/apin/caches/psv_raw_features_fingerprint.json
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section2d_psv_features_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section2d")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "model2_unified_source_map.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "split_indices.json"
OUTPUT_CACHE = CACHE_DIR / f"psv_raw_features_cache_{TIMESTAMP}.pkl"
OUTPUT_CACHE_LATEST = CACHE_DIR / "psv_raw_features_cache.pkl"
OUTPUT_FINGERPRINT = CACHE_DIR / f"psv_raw_features_fingerprint_{TIMESTAMP}.json"

# Incremental save every N images so we don't lose 70 min of work on a crash
INCREMENTAL_SAVE_EVERY = 500


def load_psv():
    """Import PSV feature extractor and IQA module."""
    from scripts.psv.feature_extractor import extract_all_features
    from scripts.psv.image_quality import assess_image_quality
    return extract_all_features, assess_image_quality


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 2D Phase 1 -- PSV raw feature extraction (9006 images)")
    logger.info("=" * 70)

    df = pd.read_csv(CSV_PATH)
    assert "is_recomposed" in df.columns, "Run Section 1 first"
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    row_to_split = {int(i): k for k, idxs in splits.items() for i in idxs}

    from scripts.apin.constants import MODEL2_CLASS_ORDER
    class_to_idx = {c: i for i, c in enumerate(MODEL2_CLASS_ORDER)}

    extract_all_features, assess_image_quality = load_psv()

    # Resume support: if a cache already exists, load it and skip completed rows
    cache = {}
    if OUTPUT_CACHE_LATEST.exists():
        logger.info(f"Found existing cache at {OUTPUT_CACHE_LATEST.name}")
        try:
            with open(OUTPUT_CACHE_LATEST, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"  Loaded {len(cache)} existing entries -- resuming")
        except Exception as e:
            logger.warning(f"  Could not load existing cache ({e}) -- starting fresh")
            cache = {}

    n_total = len(df)
    n_to_process = n_total - len(cache)
    logger.info(f"Total rows: {n_total}  |  Already cached: {len(cache)}  |  To process: {n_to_process}")

    if n_to_process == 0:
        logger.info("All rows already cached. Nothing to do.")
        return 0

    t_start = time.time()
    pbar = tqdm(total=n_total, initial=len(cache), desc="PSV features")
    failures = 0
    ext_time_total = 0.0

    for i in range(n_total):
        if int(i) in cache:
            continue
        row = df.iloc[i]
        # Prefer clahe_path (LAB-CLAHE pre-processed). Fall back to raw.
        path = row.get("clahe_path")
        if not isinstance(path, str) or not Path(path).exists():
            path = row["image_path"]
        try:
            img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            t0 = time.time()
            result = extract_all_features(img)
            iqa = assess_image_quality(img)
            ext_ms = (time.time() - t0) * 1000.0
            ext_time_total += ext_ms

            cls = row["class_name"]
            cache[int(i)] = {
                "features": result.features,  # dict[name -> float]
                # PSV confidence and flags live on the IQAResult, not FeatureResult
                "psv_confidence": float(getattr(iqa, "psv_confidence", 1.0)),
                "iqa_flags": dict(getattr(iqa, "quality_flags", {}) or {}),
                "iqa_confidence_mult": float(getattr(iqa, "psv_confidence", 1.0)),
                "extraction_time_ms": round(ext_ms, 2),
                "class_name": cls,
                "source_dataset": str(row["source_dataset"]),
                "is_field_photo": bool(row["is_field_photo"]),
                "split": row_to_split[int(i)],
                "is_recomposed": bool(row["is_recomposed"]),
                "true_class_idx": class_to_idx[cls],
                "extraction_success": True,
            }
        except Exception as e:
            failures += 1
            cache[int(i)] = {
                "features": {},
                "psv_confidence": 0.0,
                "iqa_flags": {},
                "iqa_confidence_mult": 0.0,
                "extraction_time_ms": 0.0,
                "class_name": str(row["class_name"]),
                "source_dataset": str(row["source_dataset"]),
                "is_field_photo": bool(row["is_field_photo"]),
                "split": row_to_split[int(i)],
                "is_recomposed": bool(row["is_recomposed"]),
                "true_class_idx": class_to_idx[str(row["class_name"])],
                "extraction_success": False,
                "error": str(e)[:200],
            }
            if failures < 10:
                logger.warning(f"  Row {i} failed: {str(e)[:200]}")

        pbar.update(1)

        # Incremental save
        if len(cache) % INCREMENTAL_SAVE_EVERY == 0:
            with open(OUTPUT_CACHE_LATEST, "wb") as f:
                pickle.dump(cache, f)

    pbar.close()
    elapsed = time.time() - t_start
    avg_ms = ext_time_total / max(1, (len(cache) - failures))
    logger.info(f"\nExtraction complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Average extraction time: {avg_ms:.1f}ms/image")
    logger.info(f"  Successful: {len(cache) - failures}/{len(cache)}")
    logger.info(f"  Failures:   {failures}")

    # Final save
    with open(OUTPUT_CACHE, "wb") as f:
        pickle.dump(cache, f)
    with open(OUTPUT_CACHE_LATEST, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"  Cache: {OUTPUT_CACHE.name} ({OUTPUT_CACHE.stat().st_size / 1e6:.2f} MB)")

    # Fingerprint
    feature_names = set()
    for e in cache.values():
        if e.get("extraction_success"):
            feature_names.update(e["features"].keys())

    fingerprint = {
        "phase": "2D Phase 1 -- raw features only, no calibration or scoring yet",
        "timestamp": TIMESTAMP,
        "total_rows": len(cache),
        "successful_extractions": int(sum(1 for e in cache.values() if e.get("extraction_success"))),
        "failures": int(failures),
        "mean_extraction_time_ms": round(avg_ms, 2),
        "total_feature_names": len(feature_names),
        "feature_names_alphabetical": sorted(feature_names),
        "cache_path": str(OUTPUT_CACHE_LATEST.relative_to(PROJECT_ROOT)),
        "csv_path": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        "class_order_model2": list(MODEL2_CLASS_ORDER),
        "notes": [
            "RAW features — no calibration applied yet (see Section 3C).",
            "9-class PSV score computation happens in section2d_psv_scores.py "
            "after calibration is fitted.",
            "Features dict per image contains ~66 named keys.",
        ],
    }
    with open(OUTPUT_FINGERPRINT, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"  Fingerprint: {OUTPUT_FINGERPRINT.name}")

    logger.info("=" * 70)
    logger.info("APIN SECTION 2D Phase 1 -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
