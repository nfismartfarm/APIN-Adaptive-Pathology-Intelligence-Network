"""
Section 2D Phase 2 -- Apply calibration to raw PSV features and generate
9-class PSV score vectors for all 9006 Model 2 images.

Runs AFTER section2d_psv_features.py (raw features) AND
section3c_psv_calibration.py (calibration JSON).

Also optionally consumes section3a_supervised_feature_importance.json --
if the supervised fix passed (separation >= 0.10), the rebuilt black_rot
formula is used; otherwise PSV abstention is enabled for the
black_rot/alternaria pair.

Output:
  scripts/apin/caches/signal3_psv_predictions_cache.pkl
  scripts/apin/caches/signal3_psv_predictions_fingerprint.json

Cache format (per row):
  {
    'predictions': np.array(9,) float32,      # 9-class PSV scores in
                                                MODEL2_CLASS_ORDER
    'psv_abstains_for_pair': bool,           # True if abstention enabled AND
                                                the top-2 scores differ by < 0.05
    'psv_reliability_per_class': dict,
    'class_name', 'source_dataset', 'is_field_photo', 'split',
    'is_recomposed', 'true_class_idx', 'extraction_success'
  }
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
LOG_PATH = APIN_DIR / f"section2d_phase2_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section2d_p2")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

FEATURES_CACHE = CACHE_DIR / "psv_raw_features_cache.pkl"
CALIBRATION_PATH = CACHE_DIR / "psv_calibration.json"
IMPORTANCE_PATH = CACHE_DIR / "psv_black_rot_alternaria_importance.json"
OUTPUT_CACHE = CACHE_DIR / f"signal3_psv_predictions_cache_{TIMESTAMP}.pkl"
OUTPUT_CACHE_LATEST = CACHE_DIR / "signal3_psv_predictions_cache.pkl"
OUTPUT_FINGERPRINT = CACHE_DIR / f"signal3_psv_predictions_fingerprint_{TIMESTAMP}.json"

ABSTENTION_GAP_THRESHOLD = 0.05  # top-2 PSV diff below this -> abstain


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 2D Phase 2 -- Apply calibration, generate 9-class PSV scores")
    logger.info("=" * 70)

    from scripts.apin.constants import MODEL2_CLASS_ORDER, NUM_CLASSES
    from scripts.psv.disease_scores import compute_disease_scores
    from scripts.psv.calibration import load_calibration

    assert FEATURES_CACHE.exists(), f"{FEATURES_CACHE} missing"
    assert CALIBRATION_PATH.exists(), f"{CALIBRATION_PATH} missing"

    with open(FEATURES_CACHE, "rb") as f:
        features_cache = pickle.load(f)
    logger.info(f"  Loaded raw features cache: {len(features_cache)} entries")

    # Load calibration via PSV's own loader (handles format)
    calibration = load_calibration(str(CALIBRATION_PATH))
    logger.info(f"  Loaded calibration: {len(calibration)} features")

    # Load importance-based abstention decision (if Section 3A ran)
    abstention_enabled = False
    psv_reliability_override = None
    if IMPORTANCE_PATH.exists():
        with open(IMPORTANCE_PATH) as f:
            imp = json.load(f)
        abstention_enabled = bool(
            imp.get("psv_abstention_for_black_rot_alternaria_pair", False)
        )
        psv_reliability_override = imp.get("psv_reliability_black_rot_recommendation")
        logger.info(
            f"  Importance decision: "
            f"abstention_enabled={abstention_enabled}, "
            f"reliability={psv_reliability_override}"
        )
    else:
        logger.info("  Section 3A importance file missing — using default PSV formulas")

    class_to_idx = {c: i for i, c in enumerate(MODEL2_CLASS_ORDER)}
    br_idx = class_to_idx["brassica_black_rot"]
    alt_idx = class_to_idx["brassica_alternaria"]

    output_cache = {}
    n_abstain = 0
    n_success = 0
    n_fail = 0

    for idx in sorted(features_cache.keys()):
        e = features_cache[idx]
        if not e.get("extraction_success"):
            output_cache[idx] = {
                "predictions": np.zeros(NUM_CLASSES, dtype=np.float32),
                "psv_abstains_for_pair": False,
                "class_name": e["class_name"],
                "source_dataset": e["source_dataset"],
                "is_field_photo": e["is_field_photo"],
                "split": e["split"],
                "is_recomposed": e["is_recomposed"],
                "true_class_idx": e["true_class_idx"],
                "extraction_success": False,
            }
            n_fail += 1; continue

        features = e["features"]
        # PSV disease_scores consumes the raw features + loaded calibration
        # internally. Returns dict[class_name -> score in [0,1]].
        try:
            scores_dict = compute_disease_scores(features, calibration)
        except Exception as ex:
            scores_dict = {c: 0.0 for c in MODEL2_CLASS_ORDER}
            logger.warning(f"  Row {idx} score compute failed: {ex}")

        # Reorder to MODEL2_CLASS_ORDER exactly (in case PSV returns in
        # different order or includes brassica_clubroot)
        scores_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for i, cn in enumerate(MODEL2_CLASS_ORDER):
            scores_vec[i] = float(scores_dict.get(cn, 0.0))

        # Check abstention for black_rot/alternaria pair
        abstains = False
        if abstention_enabled:
            pair_diff = abs(float(scores_vec[br_idx] - scores_vec[alt_idx]))
            if pair_diff < ABSTENTION_GAP_THRESHOLD:
                abstains = True
                n_abstain += 1

        output_cache[idx] = {
            "predictions": scores_vec,
            "psv_abstains_for_pair": abstains,
            "class_name": e["class_name"],
            "source_dataset": e["source_dataset"],
            "is_field_photo": e["is_field_photo"],
            "split": e["split"],
            "is_recomposed": e["is_recomposed"],
            "true_class_idx": e["true_class_idx"],
            "extraction_success": True,
        }
        n_success += 1

    logger.info(f"\nOutput cache summary:")
    logger.info(f"  Total rows: {len(output_cache)}")
    logger.info(f"  Successful: {n_success}")
    logger.info(f"  Failed:     {n_fail}")
    logger.info(f"  Abstained on black_rot/alternaria pair: {n_abstain}")

    # Save
    with open(OUTPUT_CACHE, "wb") as f:
        pickle.dump(output_cache, f)
    with open(OUTPUT_CACHE_LATEST, "wb") as f:
        pickle.dump(output_cache, f)
    logger.info(f"\n  Cache: {OUTPUT_CACHE.name} "
                f"({OUTPUT_CACHE.stat().st_size/1e6:.2f} MB)")

    # Per-class summary
    probs = np.stack([output_cache[i]["predictions"] for i in sorted(output_cache.keys())
                       if output_cache[i]["extraction_success"]], axis=0)
    y_true = np.array([output_cache[i]["true_class_idx"]
                        for i in sorted(output_cache.keys())
                        if output_cache[i]["extraction_success"]])

    per_class_stats = {}
    for c_idx, c_name in enumerate(MODEL2_CLASS_ORDER):
        mask = y_true == c_idx
        n = int(mask.sum())
        if n > 0:
            mean_true = float(probs[mask, c_idx].mean())
            mean_all = [round(float(v), 6) for v in probs[mask].mean(axis=0)]
        else:
            mean_true = 0.0
            mean_all = [0.0] * 9
        per_class_stats[c_name] = {
            "n_images": n,
            "mean_psv_score_on_true_class": round(mean_true, 6),
            "mean_all_class_scores": mean_all,
        }

    fingerprint = {
        "signal": 3,
        "signal_name": "PSV Physics-based Symptom Verification",
        "timestamp": TIMESTAMP,
        "raw_features_cache": str(FEATURES_CACHE.relative_to(PROJECT_ROOT)),
        "calibration_path": str(CALIBRATION_PATH.relative_to(PROJECT_ROOT)),
        "importance_file_used": str(IMPORTANCE_PATH.relative_to(PROJECT_ROOT))
            if IMPORTANCE_PATH.exists() else None,
        "abstention_enabled_for_black_rot_alternaria": bool(abstention_enabled),
        "abstention_gap_threshold": ABSTENTION_GAP_THRESHOLD,
        "psv_reliability_override": psv_reliability_override,
        "class_order_model2": list(MODEL2_CLASS_ORDER),
        "total_rows": len(output_cache),
        "successful_rows": n_success,
        "failed_rows": n_fail,
        "abstention_rows": n_abstain,
        "per_class_stats": per_class_stats,
        "cache_path": str(OUTPUT_CACHE_LATEST.relative_to(PROJECT_ROOT)),
    }
    with open(OUTPUT_FINGERPRINT, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"  Fingerprint: {OUTPUT_FINGERPRINT.name}")

    logger.info("=" * 70)
    logger.info("APIN SECTION 2D Phase 2 -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
