"""Section 3A (multi-class) — supervised PSV feature importance for all 9 classes.

Extends the original binary black_rot-vs-alternaria analysis to ALL 9 classes
via one-vs-rest L2-penalised logistic regression on the 66 PSV features.

Output: scripts/apin/section3a_psv_multiclass_importance.json with one entry
per class containing:
  - classifier params (penalty, C, cv_5fold_accuracy, train_accuracy)
  - top_15_features (sorted by |coef|)
  - top_k_separation_test for k in {5, 10, 15, 20, 30}
  - decision (best_k, best_separation, recommendation)

PSV's `disease_scores.py` reads this JSON when present and falls back to the
binary BR/ALT artifact for those two classes if the multi-class file is
unavailable. Intent: every class formula gets data-derived weights, not just
the documented BR/ALT failure pair (Gap 1 audit fix).
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.apin.constants import MODEL2_CLASS_ORDER

CACHE_DIR = PROJECT_ROOT / "scripts" / "apin" / "caches"
PSV_RAW_CACHE = CACHE_DIR / "psv_raw_features_cache.pkl"
PSV_CALIBRATION = CACHE_DIR / "psv_calibration.json"
OUT_PATH = (PROJECT_ROOT / "scripts" / "apin"
             / "section3a_psv_multiclass_importance.json")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = PROJECT_ROOT / "scripts" / "apin" / f"section3a_multiclass_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section3a_multiclass")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

K_VALUES = [5, 10, 15, 20, 30]
SEPARATION_THRESHOLD = 0.10  # minimum mean-score gap to consider class viable


def _calibrate_features_dict(features: dict, calibration: dict) -> dict:
    out = {}
    for name, val in features.items():
        if name in calibration:
            cal = calibration[name]
            p5 = cal.get("p5", 0); p95 = cal.get("p95", 1)
            if p95 > p5:
                out[name] = float(np.clip((val - p5) / (p95 - p5), 0, 1))
            else:
                out[name] = 0.5
        else:
            out[name] = float(val)
    return out


def _build_xy(raw_cache: dict, calibration: dict, target_class: str,
                feature_names: List[str]) -> tuple:
    """Returns (X, y) where y is 1 for target_class, 0 otherwise.

    Restricted to images with extraction_success=True to avoid garbage.
    Uses calibrated features (percentile-normalized) to match how
    disease_scores.py consumes them at inference."""
    target_idx = MODEL2_CLASS_ORDER.index(target_class)
    X_rows = []
    y_rows = []
    for entry in raw_cache.values():
        if not entry.get("extraction_success", True):
            continue
        feat_cal = _calibrate_features_dict(entry["features"], calibration)
        row = [feat_cal.get(n, 0.0) for n in feature_names]
        X_rows.append(row)
        y_rows.append(1 if entry["true_class_idx"] == target_idx else 0)
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64)


def _fit_lr_for_class(X: np.ndarray, y: np.ndarray,
                       feature_names: List[str]) -> dict:
    """Fit one-vs-rest L2-penalised LR; pick best C via 5-fold CV."""
    # Class balance: positives are typically 5-15% of pool — use balanced weights
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr = LogisticRegression(
                penalty="l2", C=C, max_iter=2000, class_weight="balanced",
                solver="lbfgs",
            )
            scores = cross_val_score(lr, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_mean = float(scores.mean())
            if best is None or cv_mean > best["cv_5fold_accuracy"]:
                # Refit on full data for the coefficients
                lr.fit(X, y)
                train_acc = float(lr.score(X, y))
                best = {
                    "key": f"l2_C{C}",
                    "penalty": "l2",
                    "C": float(C),
                    "cv_5fold_accuracy": cv_mean,
                    "train_accuracy": train_acc,
                    "coefs": lr.coef_[0].tolist(),
                    "intercept": float(lr.intercept_[0]),
                }
        except Exception as e:
            logger.warning(f"  LR fit C={C} failed: {e}")
    return best


def _separation_test(X: np.ndarray, y: np.ndarray,
                       feature_indices: List[int],
                       feature_coefs: List[float]) -> dict:
    """For top-k features, compute the LR z-score on positives vs negatives;
    separation = mean(z|y=1) - mean(z|y=0). Higher = more discriminative."""
    coefs = np.asarray(feature_coefs, dtype=np.float32)
    sub_X = X[:, feature_indices]
    z = sub_X @ coefs
    z_pos = z[y == 1]
    z_neg = z[y == 0]
    if len(z_pos) == 0 or len(z_neg) == 0:
        return {"separation": None}
    # Sigmoid-normalize the z scores to compare on a [0, 1] scale
    p_pos = 1.0 / (1.0 + np.exp(-z_pos))
    p_neg = 1.0 / (1.0 + np.exp(-z_neg))
    return {
        "top_k": len(feature_indices),
        "mean_positive_score": float(p_pos.mean()),
        "mean_negative_score": float(p_neg.mean()),
        "separation": float(p_pos.mean() - p_neg.mean()),
    }


def main():
    logger.info("=" * 70)
    logger.info("APIN SECTION 3A (MULTI-CLASS) — Supervised PSV importance")
    logger.info("=" * 70)

    if not PSV_RAW_CACHE.exists():
        logger.error(f"PSV raw cache missing: {PSV_RAW_CACHE}")
        sys.exit(1)
    if not PSV_CALIBRATION.exists():
        logger.error(f"PSV calibration missing: {PSV_CALIBRATION}")
        sys.exit(1)

    raw = pickle.load(open(PSV_RAW_CACHE, "rb"))
    cal = json.load(open(PSV_CALIBRATION))
    logger.info(f"Loaded {len(raw)} raw PSV entries, {len(cal)} calibrated features")

    # Get the canonical feature ordering from the first entry
    sample = next(iter(raw.values()))
    feature_names = sorted(sample["features"].keys())
    logger.info(f"PSV feature dimensionality: {len(feature_names)}")

    results: Dict[str, dict] = {}
    for target in MODEL2_CLASS_ORDER:
        logger.info(f"\n--- Class: {target} ---")
        X, y = _build_xy(raw, cal, target, feature_names)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        logger.info(f"  positives={n_pos}, negatives={n_neg}")
        if n_pos < 30:
            logger.warning(f"  too few positives ({n_pos}) — skipping LR fit")
            results[target] = {"status": "skipped_too_few_positives",
                                "n_positives": n_pos, "n_negatives": n_neg}
            continue

        best = _fit_lr_for_class(X, y, feature_names)
        if best is None:
            logger.warning(f"  no LR fit succeeded — skipping")
            results[target] = {"status": "skipped_no_fit"}
            continue

        coefs = np.asarray(best["coefs"])
        # Top features by |coef|
        idx_sorted = np.argsort(-np.abs(coefs))
        top15 = []
        for rank, idx in enumerate(idx_sorted[:15], start=1):
            top15.append({
                "rank": int(rank),
                "feature": feature_names[idx],
                "coef": float(coefs[idx]),
                "abs_coef": float(abs(coefs[idx])),
            })
        # Top-K separation tests
        sep_tests = {}
        for k in K_VALUES:
            top_k_idx = idx_sorted[:k].tolist()
            top_k_coefs = coefs[top_k_idx].tolist()
            top_k_names = [feature_names[i] for i in top_k_idx]
            sep_tests[str(k)] = {
                **_separation_test(X, y, top_k_idx, top_k_coefs),
                "feature_names_used": top_k_names,
                "feature_coefs_used": top_k_coefs,
            }
        # Best k
        best_k_entry = max(
            sep_tests.items(),
            key=lambda kv: kv[1].get("separation") or -1.0,
        )
        best_k_int = int(best_k_entry[0])
        best_sep = best_k_entry[1].get("separation") or 0.0
        passes = best_sep >= SEPARATION_THRESHOLD
        decision = {
            "best_k": best_k_int,
            "best_separation": float(best_sep),
            "separation_threshold": SEPARATION_THRESHOLD,
            "passes_threshold": bool(passes),
            "recommendation": (
                f"PASS — Use top-{best_k_int} features for {target}. "
                f"Separation {best_sep:.4f} > threshold."
                if passes else
                f"FAIL — Best separation {best_sep:.4f} below threshold {SEPARATION_THRESHOLD}. "
                f"Hand-engineered formula recommended for {target}."
            ),
        }

        logger.info(f"  best_classifier: cv_acc={best['cv_5fold_accuracy']:.3f} "
                    f"train_acc={best['train_accuracy']:.3f}")
        for k_str, t in sep_tests.items():
            logger.info(f"  k={k_str}: separation={t.get('separation'):.4f}")
        logger.info(f"  decision: {decision['recommendation']}")

        results[target] = {
            "section": "3A_multiclass",
            "timestamp": TIMESTAMP,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "feature_names_all_66": feature_names,
            "best_classifier": {k: v for k, v in best.items() if k != "coefs"},
            "intercept": best["intercept"],
            "top_15_features": top15,
            "top_k_separation_test": sep_tests,
            "decision": decision,
        }

    payload = {
        "timestamp": TIMESTAMP,
        "n_classes": len(results),
        "feature_names_all_66": feature_names,
        "per_class": results,
        "notes": [
            "One-vs-rest L2-penalised logistic regression on calibrated PSV features.",
            "Sign convention: positive coef predicts the target class.",
            "PSV disease_scores.py uses these weights when 'passes_threshold' is True.",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nSaved: {OUT_PATH.name}")
    logger.info("=" * 70)
    logger.info("APIN SECTION 3A MULTI-CLASS — COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
