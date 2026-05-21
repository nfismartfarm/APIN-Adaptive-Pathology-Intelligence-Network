"""Section 3A — PSV Supervised Feature Importance for black_rot vs alternaria.

The PSV separation score for brassica_black_rot vs brassica_alternaria
was -0.043 (alternaria scored HIGHER on the black_rot formula). This
means manual formula tuning has failed.

Approach: fit sklearn LogisticRegression on the 66 PSV features as a
binary classifier (black_rot=1 vs alternaria=0). Coefficient magnitudes
become the empirical feature importance scores. Use top-K features
(determined by CV) to rebuild the black_rot disease score formula.

If the supervised fit can achieve >0.10 separation between the two classes
(measured as |mean(black_rot_score) - mean(alternaria_score)|), use the
new formula. Otherwise, document PSV reliability for black_rot as 0.30
in the reliability matrix and enable PSV abstention for this pair.

Output:
  scripts/apin/section3a_psv_blackrot_alternaria_importance_{ts}.json
    {feature_importances, top_k, fit_metrics, separation_score, decision}
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section3a_psv_importance_{TIMESTAMP}.log"
OUT_PATH = APIN_DIR / f"section3a_psv_blackrot_alternaria_importance_{TIMESTAMP}.json"
OUT_LATEST = APIN_DIR / "section3a_psv_blackrot_alternaria_importance.json"

logger = logging.getLogger("apin.section3a")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "model2_unified_source_map.csv"
SEPARATION_THRESHOLD = 0.10
SAMPLES_PER_CLASS = 200  # subsample for speed; if class has fewer, use all


def extract_features_for_class(df_class, label):
    """Extract PSV features for a subset of images of one class.
    Returns (X np.array Nx66, feature_names, success_count)."""
    from scripts.psv.feature_extractor import extract_all_features

    X = []
    feature_names = None
    success = 0
    failed = 0

    for _, row in tqdm(df_class.iterrows(), total=len(df_class),
                       desc=f"PSV[{label}]"):
        # Use clahe_path if present, else image_path
        path = row.get("clahe_path", row["image_path"])
        if not isinstance(path, str) or not Path(path).exists():
            path = row["image_path"]
        try:
            img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            result = extract_all_features(img)
            if feature_names is None:
                feature_names = sorted(result.features.keys())
            row_feats = [float(result.features.get(fn, 0.0)) for fn in feature_names]
            # Clean NaN/Inf
            row_feats = [v if (v is not None and np.isfinite(v)) else 0.0
                         for v in row_feats]
            X.append(row_feats)
            success += 1
        except Exception as e:
            failed += 1

    logger.info(f"  {label}: {success} success, {failed} failed")
    return np.array(X, dtype=np.float32), feature_names, success


def main():
    logger.info("=" * 70)
    logger.info("APIN SECTION 3A -- PSV Supervised Feature Importance")
    logger.info("=" * 70)
    logger.info("Goal: fit LogisticRegression on 66 PSV features for black_rot")
    logger.info("vs alternaria. Use coefficient magnitudes as feature importance.")
    logger.info(f"Separation threshold for accept: {SEPARATION_THRESHOLD}")

    # Load CSV — restrict to is_recomposed=False (always True for Model 2 anyway)
    df = pd.read_csv(CSV_PATH)
    df = df[df["is_recomposed"] == False]
    logger.info(f"Loaded CSV: {len(df)} non-recomposed rows")

    # Get black_rot and alternaria images (subsample for speed)
    df_br = df[df["class_name"] == "brassica_black_rot"]
    df_alt = df[df["class_name"] == "brassica_alternaria"]
    logger.info(f"black_rot images available: {len(df_br)}")
    logger.info(f"alternaria images available: {len(df_alt)}")

    # Subsample with seed for reproducibility
    n_per = min(SAMPLES_PER_CLASS, len(df_br), len(df_alt))
    df_br_sample = df_br.sample(n=n_per, random_state=42)
    df_alt_sample = df_alt.sample(n=n_per, random_state=42)
    logger.info(f"Using {n_per} images per class")

    # Extract PSV features
    logger.info("\nExtracting PSV features (CPU-only, ~3-5 min per 200 images)...")
    t0 = time.time()
    X_br, feat_names, n_br = extract_features_for_class(df_br_sample, "black_rot")
    X_alt, feat_names_alt, n_alt = extract_features_for_class(df_alt_sample, "alternaria")
    logger.info(f"Feature extraction took {time.time() - t0:.1f}s")

    # Verify same feature names
    assert feat_names == feat_names_alt, "Feature name mismatch between classes"
    assert X_br.shape[1] == X_alt.shape[1] == len(feat_names) == 66

    # Build labeled dataset: black_rot=1, alternaria=0
    X = np.vstack([X_br, X_alt])
    y = np.concatenate([np.ones(len(X_br)), np.zeros(len(X_alt))])
    logger.info(f"Combined dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Standardize features (per-feature)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit LogisticRegression with L1 (sparse) and L2 (smooth) variants
    logger.info("\nFitting LogisticRegression variants:")
    results = {}
    for penalty, C, solver in [("l1", 1.0, "liblinear"),
                                ("l2", 1.0, "lbfgs"),
                                ("l2", 0.1, "lbfgs"),
                                ("l2", 0.01, "lbfgs")]:
        clf = LogisticRegression(penalty=penalty, C=C, solver=solver,
                                  max_iter=1000, random_state=42)
        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(clf, X_scaled, y, cv=skf,
                                  scoring="accuracy").mean()
        # Fit on full data for coefficients
        clf.fit(X_scaled, y)
        train_acc = clf.score(X_scaled, y)
        key = f"{penalty}_C{C}"
        results[key] = {
            "penalty": penalty, "C": C, "solver": solver,
            "cv_5fold_accuracy": float(cv_acc),
            "train_accuracy": float(train_acc),
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "n_nonzero_coef": int(np.sum(clf.coef_[0] != 0)),
        }
        logger.info(f"  {key:<10}  CV5_acc={cv_acc:.4f}  train_acc={train_acc:.4f}  "
                    f"nonzero_coef={results[key]['n_nonzero_coef']}/66")

    # Pick best model by CV accuracy
    best_key = max(results.keys(), key=lambda k: results[k]["cv_5fold_accuracy"])
    best = results[best_key]
    logger.info(f"\nBest model: {best_key} (CV accuracy: {best['cv_5fold_accuracy']:.4f})")

    # Feature importance ranking
    coef = np.array(best["coef"])
    importances = np.abs(coef)
    sorted_idx = np.argsort(-importances)

    logger.info(f"\nTop 15 most important features (by |coefficient|):")
    logger.info(f"  {'Feature':<40} {'|coef|':>10} {'sign':>6} {'direction':>15}")
    logger.info(f"  {'-'*75}")
    top_features = []
    for rank, idx in enumerate(sorted_idx[:15]):
        fname = feat_names[idx]
        c = float(coef[idx])
        sign = "+" if c > 0 else "-"
        direction = "favors black_rot" if c > 0 else "favors alternaria"
        logger.info(f"  {fname:<40} {abs(c):>10.4f} {sign:>6} {direction:>15}")
        top_features.append({"rank": rank + 1, "feature": fname,
                              "coef": c, "abs_coef": float(abs(c))})

    # Build a new black_rot vs alternaria score using top-K features
    # Test K = 5, 10, 15, 20 by separation score
    logger.info("\nTesting top-K formulas for separation score:")
    logger.info(f"  {'K':>3} {'mean_BR_score':>14} {'mean_ALT_score':>15} "
                f"{'separation':>12}")
    logger.info(f"  {'-'*48}")

    best_k = None
    best_sep = -1.0
    best_k_results = {}

    for K in [5, 10, 15, 20, 30]:
        top_idx = sorted_idx[:K]
        top_coef = coef[top_idx]
        # Score each image as: sum(top_coef * top_feature_value_normalized)
        # Use scaled features so coefficients are comparable
        X_br_top = X_scaled[:n_br][:, top_idx]
        X_alt_top = X_scaled[n_br:][:, top_idx]
        score_br = X_br_top @ top_coef
        score_alt = X_alt_top @ top_coef
        # Normalize to [0,1] using sigmoid for separation comparison
        score_br_n = 1.0 / (1.0 + np.exp(-score_br))
        score_alt_n = 1.0 / (1.0 + np.exp(-score_alt))
        mean_br = float(score_br_n.mean())
        mean_alt = float(score_alt_n.mean())
        separation = mean_br - mean_alt
        logger.info(f"  {K:>3} {mean_br:>14.4f} {mean_alt:>15.4f} {separation:>12.4f}")
        best_k_results[K] = {
            "top_k": K,
            "mean_black_rot_score": mean_br,
            "mean_alternaria_score": mean_alt,
            "separation": separation,
            "feature_names_used": [feat_names[i] for i in top_idx],
            "feature_coefs_used": [float(coef[i]) for i in top_idx],
        }
        if separation > best_sep:
            best_sep = separation
            best_k = K

    # Decision
    decision = {
        "best_k": best_k,
        "best_separation": best_sep,
        "separation_threshold": SEPARATION_THRESHOLD,
        "passes_threshold": best_sep > SEPARATION_THRESHOLD,
    }

    if best_sep > SEPARATION_THRESHOLD:
        decision["recommendation"] = (
            f"PASS — Use top-{best_k} features with logistic-regression-derived "
            f"weights as the new black_rot score formula. Separation {best_sep:.4f} "
            f"> threshold {SEPARATION_THRESHOLD}."
        )
        decision["psv_reliability_for_blackrot"] = "use_normally"
    else:
        decision["recommendation"] = (
            f"FAIL — Best separation {best_sep:.4f} below threshold "
            f"{SEPARATION_THRESHOLD}. Document PSV reliability for black_rot "
            f"as 0.30 in the reliability matrix and enable PSV abstention "
            f"for the black_rot/alternaria pair (Section 4 Layer 4 conflict "
            f"detection should treat PSV vote as 'abstained' when these two "
            f"classes are the top candidates)."
        )
        decision["psv_reliability_for_blackrot"] = 0.30
        decision["enable_psv_abstention_blackrot_alternaria"] = True

    logger.info("\n" + "=" * 70)
    logger.info("DECISION")
    logger.info("=" * 70)
    for k, v in decision.items():
        logger.info(f"  {k}: {v}")

    # Save full results
    payload = {
        "section": "3A",
        "timestamp": TIMESTAMP,
        "n_images_per_class": n_per,
        "feature_names_all_66": feat_names,
        "best_classifier": {
            "key": best_key,
            "penalty": best["penalty"],
            "C": best["C"],
            "cv_5fold_accuracy": best["cv_5fold_accuracy"],
            "train_accuracy": best["train_accuracy"],
        },
        "all_classifier_variants": {k: {
            "cv_5fold_accuracy": v["cv_5fold_accuracy"],
            "train_accuracy": v["train_accuracy"],
            "n_nonzero_coef": v["n_nonzero_coef"],
        } for k, v in results.items()},
        "top_15_features": top_features,
        "top_k_separation_test": best_k_results,
        "decision": decision,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    with open(OUT_LATEST, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nSaved: {OUT_PATH.name}")
    logger.info(f"Latest: {OUT_LATEST.name}")
    logger.info("=" * 70)
    logger.info("APIN SECTION 3A -- COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
