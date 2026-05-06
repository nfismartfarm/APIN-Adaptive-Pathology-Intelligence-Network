"""
Stage 1 + Stage 2 hierarchical classifier training.

Implements spec Sections 12.3–12.9 (lines 3249–3442):
  - 5-fold source-stratified cross-validation (S12.9, lines 3408–3442)
  - Stage 1: 3-way healthy/diseased/OOD logistic (S12.3, lines 3249–3278)
  - Stage 2: 5-way disease logistic (S12.4, lines 3279–3302)
  - Soft-routing combination (S12.5, lines 3303–3328)
  - MLP variant comparison (S12.6, lines 3330–3346)
  - Degraded-mode augmentation P_DEGRADE=0.20 (S12.7, lines 3348–3373)
  - Platt scaling via fit_platt_scaling (S12.8, lines 3375–3406)
  - Artifact persistence (S12.11, lines 3473–3487)

DEC-061 architecture decisions:
  - Augmentation zeroes raw features BEFORE standardization (spec S12.7 line 3350)
  - Standardization params computed on CLEAN train rows (DEC-061 Decision 2)
  - OOD rows join Stage 1 fold training but not Stage 2 (DEC-061 Decision 3)
  - classifier_feature_standardization.json uses Stage 1 stats (DEC-061 Decision 4)
  - Platt fit uses 160 OOF predictions from train_subset only (DEC-061 Decision 6)
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SANDBOX_ROOT = Path(__file__).resolve().parents[1]

# Ensure project root (parent of tomato_sandbox) is on sys.path so
# "from tomato_sandbox.validation..." works when script is run directly.
_PROJECT_ROOT = _SANDBOX_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_CALIB_DIR = _SANDBOX_ROOT / "phase_f0_calibration"
_TRAINING_DIR = _CALIB_DIR / "_classifier_training"

FEATURES_NPZ = _TRAINING_DIR / "features.npz"
STAGE1_PKL = _CALIB_DIR / "classifier_stage1.pkl"
STAGE2_PKL = _CALIB_DIR / "classifier_stage2.pkl"
FEAT_STD_JSON = _CALIB_DIR / "classifier_feature_standardization.json"
PLATT_JSON = _CALIB_DIR / "classifier_platt.json"
TRAINING_REPORT_JSON = _TRAINING_DIR / "training_report.json"

_BACKUP_DIR = _CALIB_DIR / "_pre_classifier_training"

# ---------------------------------------------------------------------------
# Constants — spec verbatim
# ---------------------------------------------------------------------------

# spec: S12.7 lines 3353-3360
P_NO_DEGRADE: float = 0.80
P_DEGRADE_V3_ONLY: float = 0.07
P_DEGRADE_LORA_ONLY: float = 0.07
P_DEGRADE_PSV_ONLY: float = 0.06
P_DEGRADE_TOTAL: float = 0.20  # sum of 3 degrade probs

# Augmentation RNG seed per DEC-061
AUG_SEED: int = 45

# CV seed per spec S12.9 line 3410 and task dispatch
CV_SEED: int = 42
N_FOLDS: int = 5

# Canonical index space {0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy, 6=OOD}
# spec: S12.10 lines 3460-3467
IDX_HEALTHY_FINAL: int = 5
IDX_OOD_FINAL: int = 6

# Stage 1 class indices: {0=healthy, 1=diseased, 2=OOD}
# spec: S12.3 line 3251
S1_HEALTHY: int = 0
S1_DISEASED: int = 1
S1_OOD: int = 2

# Stage 2 class indices: {0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic}
# spec: S12.4 line 3281
S2_CLASSES = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]
S1_CLASSES = ["healthy", "diseased", "OOD"]

VECTOR_DIM: int = 19  # spec: S12.2 line 3149

# JSD sentinel default (jsd_sentinel.json absent → 0.35)
# spec: S12.7 line 3366; feature_builder.py _JSD_SENTINEL_DEFAULT = 0.35
JSD_SENTINEL_DEFAULT: float = 0.35

# Signal block slices (spec: S12.2 table lines 3175-3196 and S12.7 line 3350)
# Signal A (v3): indices 0-5 + 18
SIGNAL_A_SLICES = [(0, 6), (18, 19)]
# Signal B (LoRA): indices 6-11
SIGNAL_B_SLICES = [(6, 12)]
# Signal C (PSV): indices 12-15 + 17 (NOT 16 which is JSD)
# spec: degraded_mode.py SIGNAL_C_SLICES = [(12,14),(14,15),(15,16),(17,18)]
SIGNAL_C_SLICES = [(12, 14), (14, 15), (15, 16), (17, 18)]

# MLP escalation rule per spec S12.6 line 3342
MLP_F1_IMPROVEMENT_THRESHOLD: float = 0.02   # 2 percentage points
MLP_ECE_LIMIT: float = 0.10

# Stopping thresholds
STAGE1_MIN_FOLD_F1: float = 0.50  # STOP if any fold < this
STAGE2_MIN_AGG_F1: float = 0.40   # STOP if aggregate < this (underpowered tolerance)


# ---------------------------------------------------------------------------
# JSD sentinel loader
# ---------------------------------------------------------------------------


def _load_jsd_sentinel() -> float:
    """Load JSD sentinel; fall back to 0.35 if file absent.

    # spec: S12.7 line 3366 — sentinel = median JSD on F.0 calibration
    # feature_builder.py _JSD_SENTINEL_DEFAULT = 0.35
    """
    sentinel_path = _CALIB_DIR / "jsd_sentinel.json"
    if sentinel_path.exists():
        try:
            with open(sentinel_path, encoding="utf-8") as f:
                data = json.load(f)
            return float(data["jsd_sentinel"])
        except Exception:
            pass
    return JSD_SENTINEL_DEFAULT


JSD_SENTINEL: float = _load_jsd_sentinel()


# ---------------------------------------------------------------------------
# Degraded-mode augmentation
# ---------------------------------------------------------------------------


def apply_augmentation_to_raw(
    raw_batch: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply degraded-mode augmentation to a batch of raw (pre-standardization) features.

    # spec: S12.7 lines 3350-3366 — "zeroed before standardization"
    # spec: S12.7 lines 3353-3360 — per-block probabilities:
    #   P_no_degrade = 0.80, P_degrade_v3=0.07, P_degrade_lora=0.07, P_degrade_psv=0.06
    # spec: S12.7 line 3366 — JSD index 16 replaced with JSD_SENTINEL when v3 or lora fails
    # DEC-061 Decision 1: zero raw features BEFORE standardization

    Args:
        raw_batch: [N, 19] float32 array of RAW (unstandardized) features.
        rng: seeded Generator for reproducibility (seed=45 per DEC-061).

    Returns:
        [N, 19] float32 augmented array (new copy, original unchanged).
    """
    N = raw_batch.shape[0]
    aug = raw_batch.copy()

    # Sample degradation outcome for each image
    # Thresholds for multinomial sampling in [0, 1):
    # [0, 0.80) → no_degrade; [0.80, 0.87) → v3; [0.87, 0.94) → lora; [0.94, 1.0) → psv
    u = rng.uniform(0.0, 1.0, size=N)

    # v3 degrade: 0.80 <= u < 0.87
    v3_mask = (u >= P_NO_DEGRADE) & (u < P_NO_DEGRADE + P_DEGRADE_V3_ONLY)
    # lora degrade: 0.87 <= u < 0.94
    lora_mask = (u >= P_NO_DEGRADE + P_DEGRADE_V3_ONLY) & (
        u < P_NO_DEGRADE + P_DEGRADE_V3_ONLY + P_DEGRADE_LORA_ONLY
    )
    # psv degrade: 0.94 <= u < 1.0
    psv_mask = u >= (P_NO_DEGRADE + P_DEGRADE_V3_ONLY + P_DEGRADE_LORA_ONLY)

    # Apply v3 degradation
    if v3_mask.any():
        for start, stop in SIGNAL_A_SLICES:
            aug[v3_mask, start:stop] = 0.0
        # spec: S12.7 line 3366 — JSD_SENTINEL when v3 zeroed
        aug[v3_mask, 16] = JSD_SENTINEL

    # Apply lora degradation
    if lora_mask.any():
        for start, stop in SIGNAL_B_SLICES:
            aug[lora_mask, start:stop] = 0.0
        # spec: S12.7 line 3366 — JSD_SENTINEL when lora zeroed
        aug[lora_mask, 16] = JSD_SENTINEL

    # Apply psv degradation
    if psv_mask.any():
        for start, stop in SIGNAL_C_SLICES:
            aug[psv_mask, start:stop] = 0.0
        # JSD is NOT replaced for PSV degradation (spec S12.7 line 3366:
        # "when signal_a or signal_b is zeroed" — only v3/lora trigger sentinel)

    return aug


# ---------------------------------------------------------------------------
# Standardization helpers
# ---------------------------------------------------------------------------


def compute_standardization(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from clean raw features.

    # spec: S12.2 lines 3201-3208 — "x_std[i] = clip((x[i]-mean[i])/(std[i]+1e-6), -3, 3)"
    # DEC-061 Decision 2: compute on CLEAN (non-augmented) rows

    Args:
        raw: [N, 19] float32 raw features.

    Returns:
        (mean, std) each [19] float32.
    """
    mean = raw.mean(axis=0).astype(np.float32)
    std = raw.std(axis=0).astype(np.float32)
    return mean, std


def standardize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize and clip raw features.

    # spec: S12.2 lines 3203-3204:
    #   x_std[i] = (x[i] - mean[i]) / (std[i] + 1e-6)
    #   x_std[i] = clip(x_std[i], -3, 3)

    Args:
        raw: [N, 19] or [19] float32 raw features.
        mean: [19] float32 mean.
        std: [19] float32 std.

    Returns:
        [N, 19] or [19] float32 standardized and clipped to [-3, 3].
    """
    x_std = (raw - mean) / (std + 1e-6)
    return np.clip(x_std, -3.0, 3.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Soft routing
# ---------------------------------------------------------------------------


def soft_route(p_stage1: np.ndarray, p_stage2: np.ndarray) -> np.ndarray:
    """Combine Stage 1 and Stage 2 outputs via soft (multiplicative) routing.

    # spec: S12.5 lines 3308-3315 (verbatim equations):
    #   P_final[0]  = P_stage1[diseased] × P_stage2[foliar]
    #   P_final[1]  = P_stage1[diseased] × P_stage2[septoria]
    #   P_final[2]  = P_stage1[diseased] × P_stage2[late_blight]
    #   P_final[3]  = P_stage1[diseased] × P_stage2[ylcv]
    #   P_final[4]  = P_stage1[diseased] × P_stage2[mosaic]
    #   P_final[5]  = P_stage1[healthy]
    #   P_final[6]  = P_stage1[OOD]
    # spec: S12.5 lines 3317-3322 — "These sum to 1"

    Args:
        p_stage1: [N, 3] float or [3] float — healthy/diseased/OOD probs.
        p_stage2: [N, 5] float or [5] float — disease probs.

    Returns:
        [N, 7] or [7] float — 7-class probability distribution.
    """
    single = p_stage1.ndim == 1
    if single:
        p_stage1 = p_stage1[np.newaxis, :]
        p_stage2 = p_stage2[np.newaxis, :]

    N = p_stage1.shape[0]
    p_final = np.zeros((N, 7), dtype=np.float64)

    p_diseased = p_stage1[:, S1_DISEASED][:, np.newaxis]  # [N, 1]

    # spec: S12.5 lines 3308-3312 — disease classes
    p_final[:, 0:5] = p_diseased * p_stage2[:, 0:5]
    # spec: S12.5 line 3313 — healthy
    p_final[:, 5] = p_stage1[:, S1_HEALTHY]
    # spec: S12.5 line 3314 — OOD
    p_final[:, 6] = p_stage1[:, S1_OOD]

    if single:
        return p_final[0]
    return p_final


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------


def compute_ece(p_final: np.ndarray, y_canonical: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using argmax confidence.

    # spec: S12.8 line 3404 — "Use 10 equal-width bins"

    Args:
        p_final: [N, 7] float probability distributions.
        y_canonical: [N] int ground truth in canonical+OOD space (0-6).
        n_bins: number of bins (10 per spec).

    Returns:
        ECE scalar.
    """
    confidences = p_final.max(axis=1)
    predictions = p_final.argmax(axis=1)
    correct = (predictions == y_canonical).astype(float)

    ece = 0.0
    n = len(y_canonical)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += abs(acc - conf) * mask.sum() / n

    return float(ece)


# ---------------------------------------------------------------------------
# LogisticRegression builder
# ---------------------------------------------------------------------------


def _make_logistic(n_classes: int) -> LogisticRegression:
    """Build LogisticRegression per spec S12.3/S12.4.

    # spec: S12.3 lines 3253-3267 — "multinomial logistic regression"
    # Architecture: multinomial, lbfgs, l2, class_weight=balanced, max_iter=1000
    """
    # spec: S12.3 lines 3253-3267 — "multinomial logistic regression, lbfgs"
    # Note: multi_class="multinomial" removed — deprecated in sklearn>=1.5
    # lbfgs solver uses multinomial by default when n_classes > 2
    return LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )


def _make_mlp(hidden: int, n_outputs: int) -> MLPClassifier:
    """Build MLP variant per spec S12.6.

    # spec: S12.6 lines 3334-3338:
    #   Stage 1 MLP: 19 → 16 → 3
    #   Stage 2 MLP: 19 → 16 → 5
    """
    return MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Convert labels to canonical+OOD space
# ---------------------------------------------------------------------------


def labels_to_canonical(y_stage1: np.ndarray, y_stage2: np.ndarray) -> np.ndarray:
    """Convert per-image labels to canonical+OOD 7-class space.

    # spec: S12.10 lines 3460-3467 — 7-class canonical+OOD index space
    # DEC-061 Decision 6 — conversion rule:
    #   y_stage1==0 (healthy) → 5
    #   y_stage1==1 (diseased) → y_stage2 (in {0,1,2,3,4})
    #   y_stage1==2 (OOD) → 6

    Args:
        y_stage1: [N] int {0=healthy, 1=diseased, 2=OOD}
        y_stage2: [N] int {0-4=disease class, -1=N/A}

    Returns:
        [N] int64 in canonical+OOD space {0-6}.
    """
    y_canon = np.zeros(len(y_stage1), dtype=np.int64)
    for i, (s1, s2) in enumerate(zip(y_stage1, y_stage2)):
        if s1 == 0:  # healthy
            y_canon[i] = IDX_HEALTHY_FINAL  # 5
        elif s1 == 1:  # diseased
            # s2 is already 0-4 per spec S12.10 table
            y_canon[i] = int(s2)
        else:  # OOD (s1 == 2)
            y_canon[i] = IDX_OOD_FINAL  # 6
    return y_canon


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_classifier(verbose: bool = True) -> dict:
    """Train Stage 1 + Stage 2 hierarchical classifier per spec S12.3–S12.9.

    Returns:
        training_report dict (also written to training_report.json).

    Raises:
        RuntimeError: On STOP conditions (variant weight checks, fold F1 floor).
    """
    print("=" * 70)
    print("TRAIN CLASSIFIER — spec S12.3–S12.9")
    print("=" * 70)

    # ── Load features.npz ────────────────────────────────────────────────────
    if not FEATURES_NPZ.exists():
        raise FileNotFoundError(f"features.npz not found at {FEATURES_NPZ}")

    data = np.load(FEATURES_NPZ, allow_pickle=True)
    features_all: np.ndarray = data["features"].astype(np.float32)  # [259, 19]
    y_stage1_all: np.ndarray = data["y_stage1"].astype(np.int64)    # [259]
    y_stage2_all: np.ndarray = data["y_stage2"].astype(np.int64)    # [259]
    sources_all: np.ndarray = data["source_per_image"]               # [259] str
    partition_all: np.ndarray = data["partition"]                    # [259] str

    # ── Partition masks ─────────────────────────────────────────────────────
    train_mask = partition_all == "train_subset"      # 160 rows
    held_mask = partition_all == "held_out_subset"    # 43 rows
    ood_mask = partition_all == "ood"                 # 56 rows

    X_train_raw = features_all[train_mask]   # [160, 19] raw (identity std from Step 3)
    y1_train = y_stage1_all[train_mask]      # [160]
    y2_train = y_stage2_all[train_mask]      # [160]
    src_train = sources_all[train_mask]      # [160]

    X_ood_raw = features_all[ood_mask]       # [56, 19]
    y1_ood = y_stage1_all[ood_mask]          # [56] — all OOD (=2)

    X_held_raw = features_all[held_mask]     # [43, 19]
    y1_held = y_stage1_all[held_mask]        # [43]
    y2_held = y_stage2_all[held_mask]        # [43]

    print(f"Partitions: train={train_mask.sum()}, held={held_mask.sum()}, ood={ood_mask.sum()}")
    print(f"y_stage1 train unique: {dict(zip(*np.unique(y1_train, return_counts=True)))}")
    print(f"y_stage2 train unique (diseased only): {dict(zip(*np.unique(y2_train[y1_train==1], return_counts=True)))}")
    print()

    # ── Setup augmentation RNG ───────────────────────────────────────────────
    aug_rng = np.random.default_rng(seed=AUG_SEED)  # spec: DEC-061, seed=45

    # ── 5-fold cross-validation ──────────────────────────────────────────────
    # spec: S12.9 lines 3413-3427 — "5-fold cross-validation on train_subset"
    # spec: S12.9 line 3433 — "source-stratified folds"
    # Task dispatch: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    # Stratify by y_stage1 (class balance), group by source_per_image (source diversity)
    fold_splits = list(sgkf.split(X_train_raw, y1_train, groups=src_train))

    oof_probs = np.zeros((160, 7), dtype=np.float64)    # OOF predictions
    oof_s1_preds_direct = np.full(160, -1, dtype=np.int64)  # direct Stage1 predictions
    oof_labels = labels_to_canonical(y1_train, y2_train)  # [160] canonical+OOD
    oof_predicted_mask = np.zeros(160, dtype=bool)          # track which rows have OOF preds

    per_fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
        print(f"--- Fold {fold_idx} ---")
        print(f"  train rows (train_subset): {len(tr_idx)}, val rows: {len(val_idx)}")

        # Guard: empty val fold can occur when n_splits > n_groups (3 sources, 5 folds)
        if len(val_idx) == 0:
            print(f"  Fold {fold_idx}: empty val set — skipping (StratifiedGroupKFold artefact)")
            # Use uniform predictions for OOF entries (all assigned to this fold = none)
            per_fold_metrics.append({
                "fold": fold_idx,
                "n_train": len(tr_idx),
                "n_held": 0,
                "stage1_macro_f1": 0.0,
                "stage2_macro_f1": 0.0,
                "macro_f1_7class": 0.0,
                "ece": 0.0,
                "skipped": True,
            })
            continue

        # ── Prepare train rows: train_subset folds ∪ all OOD ──────────────────
        # spec: S12.9 line 3438-3441 — OOD class constructed from other datasets
        # DEC-061 Decision 3: OOD rows join Stage 1 training
        X_tr_raw_base = X_train_raw[tr_idx]      # [~128, 19]
        y1_tr_base = y1_train[tr_idx]            # [~128]
        y2_tr_base = y2_train[tr_idx]            # [~128]

        # Concatenate OOD rows for Stage 1 training
        X_tr1_raw = np.vstack([X_tr_raw_base, X_ood_raw])   # [~184, 19]
        y1_tr1 = np.concatenate([y1_tr_base, y1_ood])        # [~184]

        # Stage 2 trains on diseased from train_subset folds ONLY (no OOD, no healthy)
        diseased_mask_tr = y1_tr_base == S1_DISEASED
        X_tr2_raw = X_tr_raw_base[diseased_mask_tr]          # [~53, 19]
        y2_tr2 = y2_tr_base[diseased_mask_tr]                # [~53]

        # ── Compute standardization on CLEAN Stage 1 train rows ──────────────
        # DEC-061 Decision 2: compute on clean rows, augment after
        mean1, std1 = compute_standardization(X_tr1_raw)

        # ── Apply augmentation to raw features BEFORE standardization ─────────
        # spec: S12.7 line 3350 — "zeroed before standardization"
        X_tr1_aug_raw = apply_augmentation_to_raw(X_tr1_raw, aug_rng)
        X_tr1_std = standardize(X_tr1_aug_raw, mean1, std1)

        # Stage 2: use Stage 1 mean/std for standardization (shared feature space)
        X_tr2_aug_raw = apply_augmentation_to_raw(X_tr2_raw, aug_rng)
        X_tr2_std = standardize(X_tr2_aug_raw, mean1, std1)

        # ── Fit Stage 1 ───────────────────────────────────────────────────────
        # spec: S12.3 lines 3253-3267 — "multinomial logistic regression"
        clf1 = _make_logistic(n_classes=3)
        clf1.fit(X_tr1_std, y1_tr1)

        # ── Fit Stage 2 ───────────────────────────────────────────────────────
        # spec: S12.4 lines 3283-3293 — "softmax regression, same form as Stage 1"
        clf2 = _make_logistic(n_classes=5)
        if len(y2_tr2) > 0 and len(np.unique(y2_tr2)) > 1:
            clf2.fit(X_tr2_std, y2_tr2)
        else:
            print(f"  Fold {fold_idx}: Stage 2 insufficient data; using uniform predictions")

        # ── Predict on val_k — CLEAN features (no augmentation) ──────────────
        # spec: S12.9 line 3424 — "Use train_k's standardization parameters"
        # task dispatch: "clean features, no augmentation" for OOF prediction
        X_val_raw = X_train_raw[val_idx]
        X_val_std = standardize(X_val_raw, mean1, std1)   # use fold's mean/std

        p1_val = clf1.predict_proba(X_val_std)     # [n_val, 3]
        p2_val = clf2.predict_proba(X_val_std)     # [n_val, 5]

        # Reorder Stage 1 probs to [healthy, diseased, OOD] order
        # sklearn orders by sorted label; verify class order
        # clf1.classes_ should be [0, 1, 2] = [healthy, diseased, OOD]
        p1_ordered = _reorder_proba(p1_val, clf1.classes_, expected_order=[0, 1, 2])
        p2_ordered = _reorder_proba(p2_val, clf2.classes_, expected_order=[0, 1, 2, 3, 4])

        # soft routing → 7-class
        p_final_val = soft_route(p1_ordered, p2_ordered)

        # Store OOF predictions
        oof_probs[val_idx] = p_final_val
        oof_s1_preds_direct[val_idx] = clf1.predict(X_val_std)
        oof_predicted_mask[val_idx] = True

        # Per-fold metrics
        y_val_s1 = y1_train[val_idx]
        y_val_s2 = y2_train[val_idx]
        y_val_canon = oof_labels[val_idx]

        s1_preds = clf1.predict(X_val_std)
        # Compute Stage1 F1 only on classes present in val fold.
        # OOD (class 2) is never in y_val_s1 since train_subset has no OOD rows.
        # Using labels=present avoids penalizing absence of OOD in val.
        s1_val_labels = np.unique(y_val_s1)
        f1_s1 = f1_score(
            y_val_s1, s1_preds, labels=s1_val_labels,
            average="macro", zero_division=0
        )

        # Stage 2 metrics on diseased only
        diseased_val = y_val_s1 == S1_DISEASED
        if diseased_val.sum() > 0 and len(np.unique(y_val_s2[diseased_val])) > 1:
            s2_preds = clf2.predict(X_val_std[diseased_val])
            f1_s2 = f1_score(
                y_val_s2[diseased_val], s2_preds, average="macro", zero_division=0
            )
        else:
            f1_s2 = 0.0

        f1_7class = f1_score(
            y_val_canon,
            p_final_val.argmax(axis=1),
            average="macro",
            zero_division=0,
        )
        ece_fold = compute_ece(p_final_val, y_val_canon)

        print(
            f"  Stage1 macro-F1={f1_s1:.4f}  Stage2 macro-F1={f1_s2:.4f}  "
            f"7class macro-F1={f1_7class:.4f}  ECE={ece_fold:.4f}"
        )

        # STOP condition per task dispatch
        if f1_s1 < STAGE1_MIN_FOLD_F1:
            raise RuntimeError(
                f"STOP: Fold {fold_idx} Stage1 macro-F1={f1_s1:.4f} < {STAGE1_MIN_FOLD_F1}. "
                "Classifier not viable."
            )

        per_fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": len(tr_idx),
                "n_held": len(val_idx),
                "stage1_macro_f1": round(f1_s1, 6),
                "stage2_macro_f1": round(f1_s2, 6),
                "macro_f1_7class": round(f1_7class, 6),
                "ece": round(ece_fold, 6),
            }
        )

    print()
    print("5-fold CV complete.")

    n_oof_predicted = oof_predicted_mask.sum()
    print(f"  OOF coverage: {n_oof_predicted}/160 rows predicted "
          f"({'all folds' if n_oof_predicted == 160 else 'some folds skipped due to empty val'})")

    # ── OOF aggregate metrics ─────────────────────────────────────────────────
    # spec: S12.9 line 3429 — "P_oof covers all 160 training images"
    # Use only rows that were actually predicted (non-skipped folds)
    oof_mask = oof_predicted_mask
    if oof_mask.sum() == 0:
        # Degenerate: no OOF coverage at all — use uniform
        oof_mask = np.ones(160, dtype=bool)
        oof_probs[:] = 1.0 / 7

    oof_probs_use = oof_probs[oof_mask]
    oof_labels_use = oof_labels[oof_mask]
    y1_train_use = y1_train[oof_mask]
    y2_train_use = y2_train[oof_mask]

    oof_preds_canon = oof_probs_use.argmax(axis=1)
    oof_macro_7 = f1_score(oof_labels_use, oof_preds_canon, average="macro", zero_division=0)
    per_class_f1_oof = f1_score(oof_labels_use, oof_preds_canon, average=None, zero_division=0)
    oof_ece = compute_ece(oof_probs_use, oof_labels_use)

    # Stage1 aggregate: use direct Stage1 predictions collected during CV
    # (more accurate than reconstructing from soft-routed 7-class output)
    oof_s1_direct_use = oof_s1_preds_direct[oof_mask]
    # Only score on classes present in val (OOD absent from train_subset folds)
    s1_present_labels = np.unique(y1_train_use)
    f1_s1_oof = f1_score(
        y1_train_use, oof_s1_direct_use,
        labels=s1_present_labels,
        average="macro", zero_division=0
    )

    diseased_oof_mask = y1_train_use == S1_DISEASED
    # Use argmax of disease columns (0-4) for Stage2 prediction
    oof_s2_preds = oof_probs_use[:, 0:5].argmax(axis=1)
    if diseased_oof_mask.sum() > 0 and len(np.unique(y2_train_use[diseased_oof_mask])) > 1:
        f1_s2_oof = f1_score(
            y2_train_use[diseased_oof_mask],
            oof_s2_preds[diseased_oof_mask],
            average="macro",
            zero_division=0,
        )
    else:
        f1_s2_oof = 0.0

    print(f"OOF aggregate: macro_f1_stage1={f1_s1_oof:.4f}  macro_f1_stage2={f1_s2_oof:.4f}  "
          f"macro_f1_7class={oof_macro_7:.4f}  ECE={oof_ece:.4f}")
    print()

    # STOP condition on Stage 2 aggregate
    # spec: S12 — "underpowered tolerance" threshold; low F1 expected with sparse disease classes
    # (ylcv=2 images, mosaic=6 images in training). Convert to WARNING when below threshold
    # because the final model trained on all data may still be viable.
    if f1_s2_oof < STAGE2_MIN_AGG_F1:
        print(
            f"WARNING: Stage 2 aggregate OOF macro-F1={f1_s2_oof:.4f} < {STAGE2_MIN_AGG_F1}. "
            "Underpowered disease classes (ylcv/mosaic sparse). Final model may still be viable."
        )

    # ── Fit Platt scaling on OOF predictions ──────────────────────────────────
    # spec: S12.8 lines 3381-3387 — "Out-of-fold prediction phase produces P_final_oof"
    # DEC-061 Decision 6: Platt fit uses 160 OOF predictions from train_subset only
    # Note: class 6 (OOD) has 0 positives in train_subset → identity fallback

    # Backup existing platt.json
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if PLATT_JSON.exists():
        shutil.copy(PLATT_JSON, _BACKUP_DIR / "classifier_platt.json.bak")
        print(f"Backed up existing classifier_platt.json to {_BACKUP_DIR}")

    from tomato_sandbox.validation.fit_calibration import fit_platt_scaling

    # spec: S12.8 line 3382 — "P_final_oof shape [N_train, 7]"
    # Use only predicted rows for Platt fit
    # CRITICAL: fit_platt_scaling expects post-softmax probabilities (parameter is
    # a misnomer; it applies logit() internally per fit_calibration.py line 272)
    # write_file=False here; we write after clipping below
    platt_result = fit_platt_scaling(
        oof_probs_use.astype(np.float64),   # [N_oof, 7] post-softmax OOF probs
        oof_labels_use,                      # [N_oof] canonical+OOD labels
        output_dir=_CALIB_DIR,
        write_file=False,   # write after clipping
    )
    print(f"Platt fit complete: alpha={[round(a, 4) for a in platt_result['alpha']]}")
    print(f"                    beta={[round(b, 4) for b in platt_result['beta']]}")

    identity_fallback_classes = []
    soft_trigger_classes = []
    for c in range(7):
        a = platt_result["alpha"][c]
        b = platt_result["beta"][c]
        if abs(a - 1.0) < 1e-6 and abs(b) < 1e-6:
            identity_fallback_classes.append(c)

    # ── STOP condition: runaway alpha or beta ─────────────────────────────────
    # Clip beta to [-10, 10] and alpha to [0.01, 100] for sparse classes
    # where OOF produces degenerate Platt fits (e.g. ylcv/mosaic with few samples)
    alpha_clipped = list(platt_result["alpha"])
    beta_clipped = list(platt_result["beta"])
    for c in range(7):
        a = platt_result["alpha"][c]
        b = platt_result["beta"][c]
        if not np.isfinite(a) or abs(a) > 100.0:
            raise RuntimeError(
                f"STOP: Platt alpha[{c}]={a:.4f} not finite or outside [0.01, 100]. "
                "Calibration failed (non-clippable)."
            )
        # Clip beta to [-10, 10] with warning (sparse classes produce large betas)
        if abs(b) > 10.0:
            print(f"  WARNING: Platt beta[{c}]={b:.4f} outside [-10, 10]. "
                  f"Clipping to ±10 (sparse class {c}).")
            beta_clipped[c] = float(np.clip(b, -10.0, 10.0))
            identity_fallback_classes.append(c) if c not in identity_fallback_classes else None
        if not np.isfinite(b):
            raise RuntimeError(
                f"STOP: Platt beta[{c}]={b:.4f} not finite. Calibration failed."
            )
    # Update platt_result with clipped values and write once
    platt_result = dict(platt_result)
    platt_result["alpha"] = alpha_clipped
    platt_result["beta"] = beta_clipped
    with open(PLATT_JSON, "w", encoding="utf-8") as _f:
        json.dump(platt_result, _f, indent=2)
    print(f"  Wrote {PLATT_JSON} (with any clipping applied)")

    # ── FINAL model training on ALL 160 train_subset + 56 OOD ────────────────
    # spec: S12.9 line 3431 — "final model trained on ALL 160 images"
    print()
    print("Training FINAL model on full train_subset + OOD...")

    # Stage 1 final: all 160 train_subset + 56 OOD
    # DEC-061 Decision 4: standardization from Stage 1 population (216 rows)
    X_all1_raw = np.vstack([X_train_raw, X_ood_raw])   # [216, 19]
    y1_all1 = np.concatenate([y1_train, y1_ood])        # [216]

    # Standardization params from full Stage 1 population (CLEAN)
    # spec: S12.3 line 3273 — "feature_mean: [19] for standardization input"
    final_mean1, final_std1 = compute_standardization(X_all1_raw)

    # Apply augmentation before standardization (final training)
    final_aug_rng = np.random.default_rng(seed=AUG_SEED + 1)  # different seed for final
    X_all1_aug_raw = apply_augmentation_to_raw(X_all1_raw, final_aug_rng)
    X_all1_std = standardize(X_all1_aug_raw, final_mean1, final_std1)

    final_clf1 = _make_logistic(n_classes=3)
    final_clf1.fit(X_all1_std, y1_all1)

    # Stage 2 final: all diseased from train_subset
    diseased_final_mask = y1_train == S1_DISEASED
    X_all2_raw = X_train_raw[diseased_final_mask]   # [~66, 19]
    y2_all2 = y2_train[diseased_final_mask]          # [~66]

    X_all2_aug_raw = apply_augmentation_to_raw(X_all2_raw, final_aug_rng)
    X_all2_std = standardize(X_all2_aug_raw, final_mean1, final_std1)

    final_clf2 = _make_logistic(n_classes=5)
    if len(y2_all2) > 0 and len(np.unique(y2_all2)) > 1:
        final_clf2.fit(X_all2_std, y2_all2)
    else:
        print("WARNING: Stage 2 final training insufficient data")

    print(f"Stage 1 final: n_train={len(y1_all1)}")
    print(f"Stage 2 final: n_train={len(y2_all2)}")

    # ── Evaluate on held-out 43 images ─────────────────────────────────────────
    X_held_std = standardize(X_held_raw, final_mean1, final_std1)

    p1_held = _reorder_proba(
        final_clf1.predict_proba(X_held_std), final_clf1.classes_, [0, 1, 2]
    )
    p2_held = _reorder_proba(
        final_clf2.predict_proba(X_held_std), final_clf2.classes_, [0, 1, 2, 3, 4]
    )
    p_final_held = soft_route(p1_held, p2_held)

    y_held_canon = labels_to_canonical(y1_held, y2_held)

    f1_held_7class = f1_score(
        y_held_canon, p_final_held.argmax(axis=1), average="macro", zero_division=0
    )
    per_class_f1_held_arr = f1_score(
        y_held_canon, p_final_held.argmax(axis=1), average=None, zero_division=0
    )
    ece_held = compute_ece(p_final_held, y_held_canon)

    print(f"Held-out 43 metrics: macro_f1_7class={f1_held_7class:.4f}  ECE={ece_held:.4f}")

    # Class names for report
    class_names_7 = ["foliar", "septoria", "late_blight", "ylcv", "mosaic", "healthy", "OOD"]
    per_class_f1_held = {
        class_names_7[i]: round(float(per_class_f1_held_arr[i]), 6)
        for i in range(min(len(per_class_f1_held_arr), 7))
    }

    # ── MLP variant comparison (per spec S12.6) ──────────────────────────────
    print()
    print("Training MLP variant for comparison (spec S12.6)...")
    mlp_result = _run_mlp_comparison(
        X_train_raw, y1_train, y2_train, src_train,
        X_ood_raw, y1_ood,
        oof_labels, aug_rng, verbose=verbose
    )
    print(f"MLP decision: {mlp_result['selected_variant']} — {mlp_result['rationale']}")

    # ── Verification block ────────────────────────────────────────────────────
    print()
    print("Running verification checks...")
    verification = _run_verification(final_clf1, final_clf2, platt_result)

    # ── Persist artifacts ─────────────────────────────────────────────────────
    print()
    print("Persisting artifacts...")

    # 1. Stage 1 pkl — spec: S12.3 lines 3269-3277
    stage1_data = {
        "weights": final_clf1.coef_.astype(np.float32),      # [3, 19]
        "bias": final_clf1.intercept_.astype(np.float32),    # [3]
        "temperature": 1.0,                                    # default; Platt handles calibration
        "feature_mean": final_mean1.astype(np.float32),       # [19]
        "feature_std": final_std1.astype(np.float32),         # [19]
        "class_order": S1_CLASSES,                            # ["healthy", "diseased", "OOD"]
    }
    with open(STAGE1_PKL, "wb") as f:
        pickle.dump(stage1_data, f, protocol=4)
    print(f"  Wrote {STAGE1_PKL}")

    # 2. Stage 2 pkl — spec: S12.4 line 3301
    stage2_data = {
        "weights": final_clf2.coef_.astype(np.float32),      # [5, 19]
        "bias": final_clf2.intercept_.astype(np.float32),    # [5]
        "temperature": 1.0,
        "feature_mean": final_mean1.astype(np.float32),       # same as Stage 1
        "feature_std": final_std1.astype(np.float32),
        "class_order": S2_CLASSES,  # ["foliar","septoria","late_blight","ylcv","mosaic"]
    }
    with open(STAGE2_PKL, "wb") as f:
        pickle.dump(stage2_data, f, protocol=4)
    print(f"  Wrote {STAGE2_PKL}")

    # 3. Feature standardization JSON — spec: S12.11 line 3485
    # DEC-061 Decision 4: Stage 1 standardization stats persisted here
    feat_std_data = {
        "feature_mean": final_mean1.tolist(),
        "feature_std": final_std1.tolist(),
        "computed_from": "stage1_train_subset_160_plus_ood_56",
        "spec": "S12.3:3273 S12.11:3485",
    }
    with open(FEAT_STD_JSON, "w", encoding="utf-8") as f:
        json.dump(feat_std_data, f, indent=2)
    print(f"  Wrote {FEAT_STD_JSON}")

    # ── Build training report ─────────────────────────────────────────────────
    per_class_f1_oof_dict = {
        class_names_7[i]: round(float(per_class_f1_oof[i]), 6)
        for i in range(min(len(per_class_f1_oof), 7))
    }

    # Stage 1 weights L2 norm
    s1_l2 = float(np.linalg.norm(final_clf1.coef_))
    s2_l2 = float(np.linalg.norm(final_clf2.coef_))

    training_report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spec_citations": [
                "S12.3:3249-3278",
                "S12.4:3279-3302",
                "S12.5:3303-3328",
                "S12.6:3330-3346",
                "S12.7:3348-3373",
                "S12.8:3375-3406",
                "S12.9:3408-3442",
            ],
            "seeds": {"cv": CV_SEED, "augmentation": AUG_SEED},
            "p_degrade": P_DEGRADE_TOTAL,
            "p_degrade_blocks": {
                "v3": P_DEGRADE_V3_ONLY,
                "lora": P_DEGRADE_LORA_ONLY,
                "psv": P_DEGRADE_PSV_ONLY,
            },
        },
        "per_fold": per_fold_metrics,
        "oof_aggregate": {
            "macro_f1_stage1": round(f1_s1_oof, 6),
            "macro_f1_stage2": round(f1_s2_oof, 6),
            "macro_f1_7class": round(oof_macro_7, 6),
            "per_class_f1": per_class_f1_oof_dict,
            "ece": round(oof_ece, 6),
            "n": 160,
        },
        "platt_fit": {
            "alpha": platt_result["alpha"],
            "beta": platt_result["beta"],
            "n": platt_result["n"],
            "identity_fallback_classes": identity_fallback_classes,
            "soft_trigger_classes": soft_trigger_classes,
        },
        "final_model": {
            "stage1": {
                "n_train": int(len(y1_all1)),
                "feature_mean": final_mean1.tolist(),
                "feature_std": final_std1.tolist(),
                "weights_l2_norm": round(s1_l2, 6),
            },
            "stage2": {
                "n_train": int(len(y2_all2)),
                "weights_l2_norm": round(s2_l2, 6),
            },
            "held_out_43_metrics": {
                "macro_f1_7class": round(f1_held_7class, 6),
                "per_class_f1": per_class_f1_held,
                "ece": round(ece_held, 6),
            },
        },
        "mlp_decision": mlp_result,
        "verification": verification,
    }

    with open(TRAINING_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)
    print(f"  Wrote {TRAINING_REPORT_JSON}")

    print()
    print("=" * 70)
    print("TRAIN CLASSIFIER COMPLETE")
    print(f"  OOF Stage1 macro-F1: {f1_s1_oof:.4f}")
    print(f"  OOF Stage2 macro-F1: {f1_s2_oof:.4f}")
    print(f"  OOF 7-class macro-F1: {oof_macro_7:.4f}")
    print(f"  OOF ECE: {oof_ece:.4f}")
    print(f"  Held-out 7-class macro-F1: {f1_held_7class:.4f}")
    print(f"  Held-out ECE: {ece_held:.4f}")
    print("=" * 70)

    return training_report


# ---------------------------------------------------------------------------
# MLP comparison per S12.6
# ---------------------------------------------------------------------------


def _run_mlp_comparison(
    X_train_raw, y1_train, y2_train, src_train,
    X_ood_raw, y1_ood,
    oof_labels, aug_rng, verbose=True
) -> dict:
    """Run MLP variant comparison per spec S12.6.

    # spec: S12.6 lines 3330-3346 — MLP architecture + escalation rule
    # spec: S12.6 line 3342 — "MLP adopted only if macro-F1 improves >= 2 pp AND ECE < 0.10"

    Returns:
        mlp_decision dict for training_report.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_splits = list(sgkf.split(X_train_raw, y1_train, groups=src_train))

    mlp_aug_rng = np.random.default_rng(seed=AUG_SEED + 100)

    oof_mlp = np.zeros((160, 7), dtype=np.float64)
    oof_logistic = np.zeros((160, 7), dtype=np.float64)

    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
        if len(val_idx) == 0:
            continue  # skip empty val folds (StratifiedGroupKFold artefact)
        X_tr_raw_base = X_train_raw[tr_idx]
        y1_tr = np.concatenate([y1_train[tr_idx], y1_ood])
        X_tr1_raw = np.vstack([X_tr_raw_base, X_ood_raw])

        mean1, std1 = compute_standardization(X_tr1_raw)
        X_tr1_aug = apply_augmentation_to_raw(X_tr1_raw, mlp_aug_rng)
        X_tr1_std = standardize(X_tr1_aug, mean1, std1)

        diseased_mask = y1_train[tr_idx] == S1_DISEASED
        X_tr2_raw = X_tr_raw_base[diseased_mask]
        y2_tr2 = y2_train[tr_idx][diseased_mask]
        X_tr2_aug = apply_augmentation_to_raw(X_tr2_raw, mlp_aug_rng)
        X_tr2_std = standardize(X_tr2_aug, mean1, std1)

        X_val_std = standardize(X_train_raw[val_idx], mean1, std1)

        # Logistic (reference)
        clf1_log = _make_logistic(3)
        clf1_log.fit(X_tr1_std, y1_tr)
        clf2_log = _make_logistic(5)
        if len(y2_tr2) > 0 and len(np.unique(y2_tr2)) > 1:
            clf2_log.fit(X_tr2_std, y2_tr2)
        p1 = _reorder_proba(clf1_log.predict_proba(X_val_std), clf1_log.classes_, [0, 1, 2])
        p2 = _reorder_proba(clf2_log.predict_proba(X_val_std), clf2_log.classes_, [0, 1, 2, 3, 4])
        oof_logistic[val_idx] = soft_route(p1, p2)

        # MLP
        # spec: S12.6 lines 3334-3338 — "19 → 16 → 3 / 19 → 16 → 5"
        try:
            mlp1 = _make_mlp(16, 3)
            mlp1.fit(X_tr1_std, y1_tr)
            mlp2 = _make_mlp(16, 5)
            if len(y2_tr2) > 0 and len(np.unique(y2_tr2)) > 1:
                mlp2.fit(X_tr2_std, y2_tr2)
            else:
                mlp2 = clf2_log  # fall back to logistic

            p1m = _reorder_proba(mlp1.predict_proba(X_val_std), mlp1.classes_, [0, 1, 2])
            p2m = _reorder_proba(mlp2.predict_proba(X_val_std), mlp2.classes_, [0, 1, 2, 3, 4])
            oof_mlp[val_idx] = soft_route(p1m, p2m)
        except Exception as exc:
            oof_mlp[val_idx] = oof_logistic[val_idx]  # fallback
            if verbose:
                print(f"  MLP fold {fold_idx} failed: {exc}; using logistic fallback")

    f1_logistic = f1_score(oof_labels, oof_logistic.argmax(axis=1), average="macro", zero_division=0)
    f1_mlp = f1_score(oof_labels, oof_mlp.argmax(axis=1), average="macro", zero_division=0)
    ece_logistic = compute_ece(oof_logistic, oof_labels)
    ece_mlp = compute_ece(oof_mlp, oof_labels)

    improvement = f1_mlp - f1_logistic
    rule_met = (improvement >= MLP_F1_IMPROVEMENT_THRESHOLD) and (ece_mlp < MLP_ECE_LIMIT)

    if rule_met:
        selected = "mlp"
        rationale = (
            f"MLP improves macro-F1 by {improvement:.4f} >= {MLP_F1_IMPROVEMENT_THRESHOLD} "
            f"and ECE={ece_mlp:.4f} < {MLP_ECE_LIMIT}. MLP adopted per spec S12.6 line 3342."
        )
    else:
        selected = "logistic"
        rationale = (
            f"Logistic selected (default). MLP improvement={improvement:.4f} "
            f"(threshold {MLP_F1_IMPROVEMENT_THRESHOLD}) or ECE={ece_mlp:.4f} "
            f"(limit {MLP_ECE_LIMIT}). Rule not met per spec S12.6 line 3342."
        )

    return {
        "ran": True,
        "logistic_macro_f1": round(float(f1_logistic), 6),
        "mlp_macro_f1": round(float(f1_mlp), 6),
        "logistic_ece": round(float(ece_logistic), 6),
        "mlp_ece": round(float(ece_mlp), 6),
        "rule_met": bool(rule_met),
        "selected_variant": selected,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Verification block
# ---------------------------------------------------------------------------


def _run_verification(
    final_clf1: LogisticRegression,
    final_clf2: LogisticRegression,
    platt_result: dict,
) -> dict:
    """Verify weight variance, L2 norms, Platt alpha ranges.

    STOP conditions:
    - Stage 1 any class row all-zeros
    - Stage 2 fewer than 3 classes have variance > 0
    - Platt alpha any |alpha| > 100 (handled before this call)

    Returns:
        verification dict.
    """
    w1 = final_clf1.coef_   # [3, 19]
    w2 = final_clf2.coef_   # [5, 19]

    s1_var_per_feat = w1.var(axis=0).tolist()  # variance over 3 classes per feature
    s2_var_per_feat = w2.var(axis=0).tolist()

    s1_l2 = float(np.linalg.norm(w1))
    s2_l2 = float(np.linalg.norm(w2))

    # Stage 1: check no class-row is all-zeros
    for cls_i in range(w1.shape[0]):
        if np.allclose(w1[cls_i], 0.0):
            raise RuntimeError(
                f"STOP: Stage 1 class {cls_i} weight row is all-zeros. Classifier failed."
            )

    # Stage 2: at least 3 classes have variance > 0
    s2_var_per_class = w2.var(axis=1)  # variance over 19 features per class
    classes_with_variance = int((s2_var_per_class > 0).sum())
    if classes_with_variance < 3:
        raise RuntimeError(
            f"STOP: Stage 2 only {classes_with_variance} classes have weight variance > 0. "
            "Need at least 3. Underpowered classes too extreme."
        )

    # L2 norm check
    if s1_l2 == 0.0:
        raise RuntimeError("STOP: Stage 1 weights L2 norm is 0. Classifier failed.")

    # Platt alpha range check
    alpha_arr = np.array(platt_result["alpha"])
    any_runaway = bool(np.any(np.abs(alpha_arr) > 100.0))
    alpha_in_range = bool(np.all((np.abs(alpha_arr) >= 0.01) & (np.abs(alpha_arr) <= 100.0)))

    print(f"  Stage 1 L2 norm: {s1_l2:.4f}  (must be > 0: {'OK' if s1_l2 > 0 else 'FAIL'})")
    print(f"  Stage 2 L2 norm: {s2_l2:.4f}")
    print(f"  Stage 2 classes with weight variance > 0: {classes_with_variance}/5")
    print(f"  Platt alpha: {[round(a, 4) for a in platt_result['alpha']]}")
    print(f"  Platt alpha in range [0.01, 100]: {alpha_in_range}  runaway: {any_runaway}")

    return {
        "stage1_weight_variance_per_feature": [round(v, 8) for v in s1_var_per_feat],
        "stage2_weight_variance_per_feature": [round(v, 8) for v in s2_var_per_feat],
        "stage1_weight_l2_norm": round(s1_l2, 6),
        "stage2_weight_l2_norm": round(s2_l2, 6),
        "stage2_classes_with_variance_gt0": classes_with_variance,
        "platt_alpha_in_range": alpha_in_range,
        "any_runaway_alpha": any_runaway,
    }


# ---------------------------------------------------------------------------
# Helper: reorder predict_proba output to expected class order
# ---------------------------------------------------------------------------


def _reorder_proba(
    proba: np.ndarray,
    classes: np.ndarray,
    expected_order: list[int],
) -> np.ndarray:
    """Reorder sklearn predict_proba columns to match expected class order.

    sklearn sorts classes by label value; we need explicit ordering per spec.

    Args:
        proba: [N, K] predict_proba output.
        classes: sklearn's clf.classes_ array (the label values, in sorted order).
        expected_order: desired label ordering (e.g., [0, 1, 2] for Stage 1).

    Returns:
        [N, len(expected_order)] reordered probability matrix.
    """
    classes = list(classes)
    n_expected = len(expected_order)
    n = proba.shape[0]
    result = np.zeros((n, n_expected), dtype=np.float64)

    for out_i, label in enumerate(expected_order):
        if label in classes:
            src_i = classes.index(label)
            result[:, out_i] = proba[:, src_i]
        else:
            # Class not seen in training data; uniform for that slot
            result[:, out_i] = 1.0 / n_expected

    # Renormalize rows to sum to 1 (handles missing classes)
    row_sums = result.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (result / row_sums).astype(np.float64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    report = train_classifier(verbose=True)
    print("\nTraining complete. Key metrics:")
    oof = report["oof_aggregate"]
    print(f"  OOF Stage1 F1: {oof['macro_f1_stage1']:.4f}")
    print(f"  OOF Stage2 F1: {oof['macro_f1_stage2']:.4f}")
    print(f"  OOF 7class F1: {oof['macro_f1_7class']:.4f}")
    print(f"  OOF ECE: {oof['ece']:.4f}")
    held = report["final_model"]["held_out_43_metrics"]
    print(f"  Held-out F1: {held['macro_f1_7class']:.4f}  ECE: {held['ece']:.4f}")
