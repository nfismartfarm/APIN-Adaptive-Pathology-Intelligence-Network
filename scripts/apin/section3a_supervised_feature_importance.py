"""
Section 3A -- Supervised feature importance for black_rot vs alternaria.

After the PSV raw feature cache exists (section2d_psv_features.py), this
script:
  1. Extracts features for all is_recomposed=False training images of
     brassica_black_rot and brassica_alternaria.
  2. Fits sklearn LogisticRegression for the binary black_rot vs alternaria
     classification.
  3. Reports top 10 most important features by |coefficient magnitude|.
  4. Validates: separation = mean(p_black_rot | true=black_rot)
                            - mean(p_black_rot | true=alternaria)
     must be > 0.10 to declare PSV reliable for this pair.
  5. Saves rebuilt black_rot scoring formula as a dict of
     {feature_name: weight} that disease_scores.py can consume.

Output:
  scripts/apin/caches/psv_black_rot_alternaria_importance.json
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section3a_supervised_importance_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section3a")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

FEATURES_CACHE = CACHE_DIR / "psv_raw_features_cache.pkl"
OUTPUT_PATH = CACHE_DIR / f"psv_black_rot_alternaria_importance_{TIMESTAMP}.json"
OUTPUT_LATEST = CACHE_DIR / "psv_black_rot_alternaria_importance.json"

SEPARATION_THRESHOLD = 0.10


def build_feature_matrix(cache, feature_names, target_class1, target_class2):
    """Build (X, y) from cache for the two target classes, excluding
    is_recomposed=True and excluding rows in val/final_val/conformal
    (only use train split to avoid leakage).
    """
    rows_x, rows_y = [], []
    meta = []
    for idx in sorted(cache.keys()):
        entry = cache[idx]
        if not entry.get("extraction_success", False):
            continue
        if entry["is_recomposed"]:
            continue
        if entry["split"] != "train":
            continue
        cls = entry["class_name"]
        if cls not in (target_class1, target_class2):
            continue
        feat_dict = entry["features"]
        row_vec = [float(feat_dict.get(name, 0.0)) for name in feature_names]
        rows_x.append(row_vec)
        rows_y.append(1 if cls == target_class1 else 0)
        meta.append({"idx": idx, "class_name": cls})
    return np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.int64), meta


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 3A -- Supervised feature importance (black_rot vs alternaria)")
    logger.info("=" * 70)

    assert FEATURES_CACHE.exists(), (
        f"{FEATURES_CACHE} missing. Run section2d_psv_features.py first."
    )
    with open(FEATURES_CACHE, "rb") as f:
        cache = pickle.load(f)
    logger.info(f"Loaded cache: {len(cache)} entries")

    # Collect all unique feature names across all cache entries
    feature_names = set()
    for e in cache.values():
        if e.get("extraction_success"):
            feature_names.update(e["features"].keys())
    feature_names = sorted(feature_names)
    logger.info(f"  Feature names: {len(feature_names)}")

    X, y, meta = build_feature_matrix(
        cache, feature_names, "brassica_black_rot", "brassica_alternaria"
    )
    n_br = int((y == 1).sum()); n_alt = int((y == 0).sum())
    logger.info(f"  brassica_black_rot train samples: {n_br}")
    logger.info(f"  brassica_alternaria train samples: {n_alt}")

    if n_br < 30 or n_alt < 30:
        logger.error("Not enough samples for reliable LogReg fit.")
        return 1

    # Standard-scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit LogReg with L2 + cross-validation for robust coefficient estimates
    clf = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    # 5-fold CV on accuracy
    cv_acc = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
    cv_auc = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")
    logger.info(
        f"  5-fold CV: mean accuracy={cv_acc.mean():.4f} (std {cv_acc.std():.4f}), "
        f"mean AUC={cv_auc.mean():.4f} (std {cv_auc.std():.4f})"
    )

    # Fit on full data for coefficient inspection
    clf.fit(X_scaled, y)
    coefs = clf.coef_[0]  # shape (n_features,)
    importances = np.abs(coefs)

    # Rank and report
    order = np.argsort(-importances)
    logger.info("\nTop 20 features by |coefficient| (positive = black_rot, negative = alternaria):")
    logger.info(f"  {'Feature':<40} {'coef':>10}  {'|coef|':>10}")
    logger.info(f"  {'-' * 60}")
    top = []
    for k in order[:20]:
        name = feature_names[k]
        c = float(coefs[k]); a = float(importances[k])
        logger.info(f"  {name:<40} {c:>10.4f}  {a:>10.4f}")
        top.append({"feature": name, "coef": c, "abs_coef": a})

    # Rebuild black_rot scoring formula: positive-coef features added with their
    # coefficient, normalized so the top weight = 2.0 (matches existing PSV
    # formula scales). Only keep positive-coef features (those pushing toward
    # black_rot), top K.
    K_FEATURES = 8
    pos_features = [t for t in top if t["coef"] > 0][:K_FEATURES]
    neg_features = [t for t in top if t["coef"] < 0][:K_FEATURES]
    max_pos = max((f["coef"] for f in pos_features), default=1.0)
    new_formula = {
        f["feature"]: round(2.0 * f["coef"] / max_pos, 4)
        for f in pos_features
    }
    logger.info(
        f"\nRebuilt black_rot score formula (top {len(new_formula)} positive features):"
    )
    for name, w in new_formula.items():
        logger.info(f"  + {w:>6.3f} * {name}")
    logger.info("\nTop features pointing AWAY from black_rot (toward alternaria):")
    for f in neg_features:
        logger.info(f"  {f['feature']}: coef={f['coef']:.4f}")

    # Compute separation using the LogReg probabilities (out-of-fold would be
    # cleaner but given the already-small sample we use in-sample here, with
    # the CV accuracy/AUC above providing generalization evidence).
    probs = clf.predict_proba(X_scaled)[:, 1]  # P(black_rot)
    sep = float(probs[y == 1].mean() - probs[y == 0].mean())
    logger.info(f"\nSeparation (P(black_rot|true=black_rot) - P(black_rot|true=alternaria)):")
    logger.info(f"  {sep:.4f} "
                f"({'>= 0.10 PASS' if sep >= SEPARATION_THRESHOLD else '< 0.10 FAIL — abstention recommended'})")

    # Out-of-fold check for a more honest separation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros_like(probs)
    for tr_idx, vl_idx in skf.split(X_scaled, y):
        c2 = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
        c2.fit(X_scaled[tr_idx], y[tr_idx])
        oof_probs[vl_idx] = c2.predict_proba(X_scaled[vl_idx])[:, 1]
    oof_sep = float(oof_probs[y == 1].mean() - oof_probs[y == 0].mean())
    logger.info(f"\nOut-of-fold separation: {oof_sep:.4f}")

    # Decision
    passes = (oof_sep >= SEPARATION_THRESHOLD)
    logger.info(
        f"\nDecision: PSV black_rot vs alternaria separation "
        f"{'PASSES' if passes else 'FAILS'} threshold {SEPARATION_THRESHOLD}"
    )
    if not passes:
        logger.info(
            "  -> PSV reliability for brassica_black_rot will be set to 0.30\n"
            "  -> Abstention enabled: PSV contributes 0 weight when top-2 PSV\n"
            "     scores' difference < 0.05 for this pair"
        )

    # Save
    payload = {
        "timestamp": TIMESTAMP,
        "target_class_1_positive": "brassica_black_rot",
        "target_class_0_negative": "brassica_alternaria",
        "n_black_rot_train": n_br,
        "n_alternaria_train": n_alt,
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 5000,
            "cv_5fold_accuracy_mean": float(cv_acc.mean()),
            "cv_5fold_accuracy_std": float(cv_acc.std()),
            "cv_5fold_auc_mean": float(cv_auc.mean()),
            "cv_5fold_auc_std": float(cv_auc.std()),
        },
        "separation_in_sample": sep,
        "separation_out_of_fold": oof_sep,
        "separation_threshold": SEPARATION_THRESHOLD,
        "passes_threshold": bool(passes),
        "top_20_features": top,
        "top_features_toward_black_rot": pos_features,
        "top_features_toward_alternaria": neg_features,
        "rebuilt_black_rot_score_formula": new_formula,
        "psv_reliability_black_rot_recommendation": 0.95 if passes else 0.30,
        "psv_abstention_for_black_rot_alternaria_pair": bool(not passes),
        "notes": [
            "The rebuilt formula uses only positive-coef features "
            "(features that push PSV score UP for black_rot).",
            "Weights are normalized so the top feature has weight 2.0 "
            "(matching existing PSV formula scales).",
            "If separation >= 0.10 -> use the new formula in disease_scores.py; "
            "if < 0.10 -> trigger abstention path instead (do not overwrite formula).",
            "When PSV abstention is active, PSV contributes zero weight to the "
            "stacking MLP's decision for the black_rot/alternaria pair specifically.",
        ],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    with open(OUTPUT_LATEST, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nSaved: {OUTPUT_PATH.name}")

    logger.info("=" * 70)
    logger.info("APIN SECTION 3A -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
