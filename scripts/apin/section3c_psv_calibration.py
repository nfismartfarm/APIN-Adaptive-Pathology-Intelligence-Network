"""
Section 3C -- PSV Calibration.

Computes per-feature percentile statistics from the PSV raw features cache
(is_recomposed=False + split=train subset only, to avoid val leakage).
Saves psv_calibration.json in the format compatible with
scripts/psv/calibration.py:save_calibration / load_calibration.

Output:
  scripts/apin/caches/psv_calibration.json (same file path PSV code
    already loads via PSV_CFG.CALIBRATION_PATH)
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section3c_calibration_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section3c")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

FEATURES_CACHE = CACHE_DIR / "psv_raw_features_cache.pkl"

# PSV loads calibration from this default path — point our output there
def get_psv_default_calibration_path():
    from scripts.psv.config import PSV_CFG
    path = Path(PSV_CFG.ROOT) / PSV_CFG.CALIBRATION_PATH
    return path


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 3C -- PSV calibration from training features")
    logger.info("=" * 70)

    with open(FEATURES_CACHE, "rb") as f:
        cache = pickle.load(f)
    logger.info(f"Loaded cache: {len(cache)} entries")

    # Build feature dicts + labels for calibration training pool:
    #   is_recomposed=False AND split=train AND extraction_success=True
    feature_dicts = []
    labels = []
    excluded_recomposed = 0
    excluded_val = 0
    excluded_fail = 0
    for idx in sorted(cache.keys()):
        e = cache[idx]
        if not e.get("extraction_success"):
            excluded_fail += 1; continue
        if e["is_recomposed"]:
            excluded_recomposed += 1; continue
        if e["split"] != "train":
            excluded_val += 1; continue
        feature_dicts.append(e["features"])
        labels.append(e["class_name"])

    logger.info(f"  Included for calibration: {len(feature_dicts)}")
    logger.info(f"  Excluded (not train split): {excluded_val}")
    logger.info(f"  Excluded (is_recomposed=True): {excluded_recomposed} "
                f"[should be 0 for Model 2]")
    logger.info(f"  Excluded (extraction failed): {excluded_fail}")

    if len(feature_dicts) < 100:
        logger.error("Too few training images for reliable calibration.")
        return 1

    # Use PSV's own compute_calibration function so the output format matches
    from scripts.psv.calibration import compute_calibration, save_calibration
    calibration = compute_calibration(feature_dicts, labels)
    logger.info(f"  Computed per-feature percentiles for {len(calibration)} features")

    # Sanity: check a few feature ranges
    sample = list(calibration.keys())[:5]
    for name in sample:
        stats = calibration[name]
        logger.info(f"    {name}: p5={stats['p5']:.4f}, p50={stats['p50']:.4f}, "
                    f"p95={stats['p95']:.4f}, mean={stats['mean']:.4f}")

    # Save to PSV's default path AND to our audit location
    psv_default = get_psv_default_calibration_path()
    psv_default.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(calibration, path=str(psv_default))
    logger.info(f"  PSV default calibration: {psv_default}")

    our_copy_ts = CACHE_DIR / f"psv_calibration_{TIMESTAMP}.json"
    our_copy_latest = CACHE_DIR / "psv_calibration.json"
    with open(our_copy_ts, "w") as f:
        json.dump(calibration, f, indent=2)
    with open(our_copy_latest, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"  APIN audit copy: {our_copy_latest.name}")

    logger.info("=" * 70)
    logger.info("APIN SECTION 3C -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
