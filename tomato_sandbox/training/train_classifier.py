"""
Hierarchical classifier training for the Tomato 3-Signal sandbox.
Dispatch: Step 4 V3 (DEC-061 sub-decision). V2 quarantined for S12.7 threshold misses.

Spec citations (read verbatim per Fix-42):
  spec: section 12.3 lines 3249-3278  -- Stage 1 architecture / pkl schema
  spec: section 12.4 lines 3279-3302  -- Stage 2 architecture / pkl schema
  spec: section 12.5 lines 3303-3328  -- Soft-routing combination
  spec: section 12.6 lines 3330-3346  -- Logistic default; MLP escalation
  spec: section 12.7 lines 3348-3373  -- Degraded-mode P_DEGRADE design + verification
  spec: section 12.8 lines 3375-3406  -- Platt scaling
  spec: section 12.9 lines 3408-3442  -- OOF training procedure
  spec: section 12.10 lines 3444-3471 -- canonical+OOD index space
  spec: section 12.11 lines 3473-3487 -- pkl/JSON paths

V3 vs V2 delta (SINGLE CHANGE per DEC-061 sub-decision):
  P_DEGRADE increased from 0.20 to 0.35 per spec S12.7:3373 remediation rule.
  Per-block ratios scaled proportionally with lora>=psv preserved:
    P_NO_DEGRADE   = 0.65  (was 0.80)
    P_DEGRADE_V3   = 0.12  (was 0.07)
    P_DEGRADE_LORA = 0.12  (was 0.07)
    P_DEGRADE_PSV  = 0.11  (was 0.06)
  RNG seed unchanged: seed=45 (per DEC-060 sub-decision).

V3 additions (M: built-in degraded-mode verification):
  After FINAL model training + Platt fit, simulate each signal failure on
  held_out_subset (43 rows) and verify S12.7:3368-3373 thresholds.
  STOP if any threshold misses — EXCEPT lora_off + psv_off which are
  pre-adjudicated per DEC-061 sub-decision / BLK-017 (see below).

BLK-017 adjudication (main-thread decision, this run):
  lora_off F1=0.528 < 0.55 and psv_off F1=0.536 < 0.65 are DOCUMENTED
  LIMITATIONS. v3 is accepted architecturally. pass: false is recorded
  verbatim in training_report_v3.json with bypassed_per_blk_017: true.
  Production mitigation: signal failures fire Rule 1 → Tier 4B → retake-prompt.

STOP discipline (NON-NEGOTIABLE):
  - STOP conditions raise ValueError with clear messages
  - Do NOT clip to fit threshold
  - Do NOT convert STOP to WARNING
  - Partial state saved to quarantine directory before raising
  - Main thread must adjudicate

STOP thresholds:
  - Stage 1 per-fold macro-F1 < 0.50
  - Stage 2 OOF aggregate macro-F1 < 0.30
  - Platt alpha NaN OR beta NaN OR beta outside [-50, 50]
  - Weight variance failure: > 2 classes with variance <= 0
  - Degraded-mode v3_off macro-F1 < 0.55  (NEW in V3; still STOP)
  - Degraded-mode lora_off macro-F1 < 0.55 (NEW in V3; BYPASSED per BLK-017)
  - Degraded-mode psv_off macro-F1 < 0.65  (NEW in V3; BYPASSED per BLK-017)
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SANDBOX_ROOT = Path(__file__).resolve().parents[1]
_CALIBRATION_DIR = _SANDBOX_ROOT / "phase_f0_calibration"
_TRAINING_DIR = _CALIBRATION_DIR / "_classifier_training"

# Production output paths per spec S12.11 lines 3481-3485
_STAGE1_PKL = _CALIBRATION_DIR / "classifier_stage1.pkl"
_STAGE2_PKL = _CALIBRATION_DIR / "classifier_stage2.pkl"
_FEATURE_STD_JSON = _CALIBRATION_DIR / "classifier_feature_standardization.json"
_PLATT_JSON = _CALIBRATION_DIR / "classifier_platt.json"
_TRAINING_REPORT = _TRAINING_DIR / "training_report_v3.json"

# Quarantine directory for STOP conditions
_QUARANTINE_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
_QUARANTINE_DIR = _CALIBRATION_DIR / f"_quarantined_step4_v3_{_QUARANTINE_TS}"

# JSD sentinel (spec S12.2 line 3247)
_JSD_SENTINEL_FILE = _CALIBRATION_DIR / "jsd_sentinel.json"
_JSD_SENTINEL_DEFAULT = 0.35

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# V3 P_DEGRADE values per DEC-061 sub-decision (S12.7:3373 remediation).
# Verification: P_DEGRADE_V3 + P_DEGRADE_LORA + P_DEGRADE_PSV = 0.35 = P_DEGRADE
# Per-block ratios (lora>psv preserved): 0.12:0.12:0.11 = 1.09:1.09:1.00
# spec: section 12.7 lines 3348-3373
P_NO_DEGRADE   = 0.65   # v3: was 0.80
P_DEGRADE_V3   = 0.12   # v3: was 0.07
P_DEGRADE_LORA = 0.12   # v3: was 0.07
P_DEGRADE_PSV  = 0.11   # v3: was 0.06
_P_DEGRADE_SUM = P_DEGRADE_V3 + P_DEGRADE_LORA + P_DEGRADE_PSV  # must be 0.35
assert abs(_P_DEGRADE_SUM - 0.35) < 1e-10, f"P_DEGRADE sum {_P_DEGRADE_SUM} != 0.35"
assert abs(P_NO_DEGRADE + _P_DEGRADE_SUM - 1.0) < 1e-10, "Probability mass != 1.0"

# Augmentation RNG seed: unchanged from v2 per DEC-060 sub-decision
_AUG_SEED = 45

# CV seed
_CV_SEED = 42

# OOD heldout selection seed
_OOD_HELDOUT_SEED = 46

# 7-class canonical+OOD index space per spec S12.10 lines 3460-3467
CLASS_NAMES_7 = ["foliar", "septoria", "late_blight", "ylcv", "mosaic", "healthy", "OOD"]
NUM_CLASSES = 7
IDX_OOD = 6
IDX_HEALTHY = 5

# Stage 1 class order per spec S12.3 line 3275
# spec: section 12.3 line 3275 — "class_order: ['healthy', 'diseased', 'OOD']"
STAGE1_CLASS_ORDER = ["healthy", "diseased", "OOD"]
S1_HEALTHY_IDX = 0
S1_DISEASED_IDX = 1
S1_OOD_IDX = 2

# Stage 2 class order per spec S12.4 line 3301
# spec: section 12.4 line 3301 — "class_order = ['foliar', 'septoria', 'late_blight', 'ylcv', 'mosaic']"
STAGE2_CLASS_ORDER = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]

# Signal block indices per spec S12.7 + degraded_mode.py
# spec: SIGNAL_A_SLICES=[(0,6),(18,19)], SIGNAL_B_SLICES=[(6,12)],
#       SIGNAL_C_SLICES=[(12,14),(14,15),(15,16),(17,18)]
_V3_BLOCK_PROBS = slice(0, 6)    # v3 probability features
_V3_CHILLI_IDX = 18              # chilli_leakage
_LORA_BLOCK = slice(6, 12)       # lora probability features
# PSV block: indices 12-15 + 17 (excl idx 16=JSD, per SIGNAL_C_SLICES)
_PSV_BLOCK_SLICES = [slice(12, 16), slice(17, 18)]
_JSD_IDX = 16                    # JSD feature index

# S12.7:3368-3373 degraded-mode verification thresholds
_DEGRADE_THRESH = {
    "v3_off":   0.55,   # spec S12.7 line 3369
    "lora_off": 0.55,   # spec S12.7 line 3370
    "psv_off":  0.65,   # spec S12.7 line 3371
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsd_sentinel() -> float:
    """Load JSD sentinel; default 0.35 if file absent.
    # spec: section 12.2 line 3247
    """
    if _JSD_SENTINEL_FILE.exists():
        try:
            with open(_JSD_SENTINEL_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return float(data["jsd_sentinel"])
        except Exception:
            pass
    return _JSD_SENTINEL_DEFAULT


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error over n_bins equal-width bins.
    # spec: section 12.8 line 3404 — "10 equal-width bins"
    probs: [N, 7] post-Platt probabilities
    labels: [N] integer labels 0-6
    """
    n = len(labels)
    if n == 0:
        return 0.0
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (pred == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece_val += abs(acc - conf) * mask.sum() / n
    return float(ece_val)


def _soft_route(p_stage1: np.ndarray, p_stage2: np.ndarray) -> np.ndarray:
    """Soft routing from Stage 1 (3-class) + Stage 2 (5-class) to 7-class joint.
    # spec: section 12.5 lines 3307-3315
    p_stage1: [3] — [P(healthy), P(diseased), P(OOD)]
    p_stage2: [5] — [P(foliar), P(septoria), P(lb), P(ylcv), P(mosaic)]
    Returns: [7] — [foliar, septoria, lb, ylcv, mosaic, healthy, OOD]
    """
    p_final = np.zeros(NUM_CLASSES, dtype=np.float64)
    # spec S12.5 lines 3308-3314
    p_final[0:5] = p_stage1[S1_DISEASED_IDX] * p_stage2[0:5]  # disease classes
    p_final[IDX_HEALTHY] = p_stage1[S1_HEALTHY_IDX]            # healthy
    p_final[IDX_OOD] = p_stage1[S1_OOD_IDX]                    # OOD
    return p_final


def _soft_route_batch(p_stage1: np.ndarray, p_stage2: np.ndarray) -> np.ndarray:
    """Batch soft routing.
    # spec: section 12.5 lines 3307-3315
    p_stage1: [N, 3]; p_stage2: [N, 5] -> [N, 7]
    """
    n = p_stage1.shape[0]
    p_final = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    p_final[:, 0:5] = p_stage1[:, S1_DISEASED_IDX : S1_DISEASED_IDX + 1] * p_stage2[:, 0:5]
    p_final[:, IDX_HEALTHY] = p_stage1[:, S1_HEALTHY_IDX]
    p_final[:, IDX_OOD] = p_stage1[:, S1_OOD_IDX]
    return p_final


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize and clip per spec S12.2 lines 3202-3205.
    # spec: section 12.2 lines 3202-3205
    x_std[i] = clip((x[i] - mean[i]) / (std[i] + 1e-6), -3, 3)
    """
    return np.clip((X - mean) / (std + 1e-6), -3.0, 3.0)


def _to_canonical(y_stage1: int, y_stage2: int) -> int:
    """Convert stage labels to canonical+OOD index space.
    # spec: section 12.10 lines 3460-3467
    y_stage1: 0=healthy, 1=diseased, 2=OOD
    y_stage2: 0-4 for diseased; -1 sentinel for non-diseased
    Returns: canonical+OOD index {0=foliar,1=septoria,2=lb,3=ylcv,4=mosaic,5=healthy,6=OOD}
    """
    if y_stage1 == S1_HEALTHY_IDX:    # 0 = healthy
        return IDX_HEALTHY             # 5
    if y_stage1 == S1_DISEASED_IDX:   # 1 = diseased
        return int(y_stage2)           # 0-4 disease indices
    if y_stage1 == S1_OOD_IDX:        # 2 = OOD
        return IDX_OOD                 # 6
    raise ValueError(f"Unexpected y_stage1={y_stage1}")


def _to_canonical_batch(y_stage1: np.ndarray, y_stage2: np.ndarray) -> np.ndarray:
    """Batch conversion to canonical+OOD labels."""
    result = np.full(len(y_stage1), -1, dtype=np.int64)
    for i, (s1, s2) in enumerate(zip(y_stage1, y_stage2)):
        result[i] = _to_canonical(int(s1), int(s2))
    return result


def _apply_degraded_augmentation(
    X: np.ndarray,
    rng: np.random.Generator,
    jsd_sentinel: float,
) -> np.ndarray:
    """Apply V3 degraded-mode augmentation to a batch during FIT (not eval).

    V3 probabilities (DEC-061 sub-decision):
      P_no_degrade = 0.65, P_v3 = 0.12, P_lora = 0.12, P_psv = 0.11

    # spec: section 12.7 lines 3348-3373 — training-time augmentation
    # spec: section 12.7 line 3366 — JSD replaced with JSD_SENTINEL when
    #   v3 OR lora is zeroed (NOT when psv is zeroed)

    Applied per sample AFTER per-fold standardization.
    RNG seed unchanged from v2: seed=45 per DEC-060 sub-decision.
    """
    X_aug = X.copy()
    n = X.shape[0]
    # Draw uniform [0,1) per sample to assign degrade bucket
    draws = rng.uniform(0.0, 1.0, size=n)
    # Bucket boundaries (V3):
    #   [0, 0.12)          = degrade_v3
    #   [0.12, 0.24)       = degrade_lora
    #   [0.24, 0.35)       = degrade_psv
    #   [0.35, 1.0)        = no_degrade
    for i in range(n):
        d = draws[i]
        if d < P_DEGRADE_V3:
            # Zero v3 block: indices 0-5 + 18
            # spec S12.7 + SIGNAL_A_SLICES from degraded_mode.py
            X_aug[i, _V3_BLOCK_PROBS] = 0.0
            X_aug[i, _V3_CHILLI_IDX] = 0.0
            # Replace JSD with sentinel (v3 failed)
            # spec: S12.7 line 3366 — "When signal_a or signal_b is zeroed out,
            #   JSD feature (index 16) is replaced with JSD_SENTINEL"
            X_aug[i, _JSD_IDX] = jsd_sentinel
        elif d < P_DEGRADE_V3 + P_DEGRADE_LORA:
            # Zero lora block: indices 6-11
            # spec S12.7 + SIGNAL_B_SLICES from degraded_mode.py
            X_aug[i, _LORA_BLOCK] = 0.0
            # Replace JSD with sentinel (lora failed)
            # spec: S12.7 line 3366
            X_aug[i, _JSD_IDX] = jsd_sentinel
        elif d < P_DEGRADE_V3 + P_DEGRADE_LORA + P_DEGRADE_PSV:
            # Zero PSV block: indices 12-15 + 17 (excl idx 16=JSD)
            # spec S12.7 + SIGNAL_C_SLICES from degraded_mode.py
            for slc in _PSV_BLOCK_SLICES:
                X_aug[i, slc] = 0.0
            # JSD stays as-is when PSV fails (v3 and lora both present)
            # spec: S12.7 line 3366 — only zeroed when signal_a OR signal_b fails
        # else: no degradation (0.65 probability)
    return X_aug


def _fit_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
) -> LogisticRegression:
    """Fit LogisticRegression per spec S12.3/S12.4.
    # spec: section 12.3 lines 3253 — "multinomial logistic regression"
    # spec: section 12.9 line 3435 — "class_weight='balanced'"
    """
    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=_CV_SEED,
        # lbfgs with 3+ classes uses multinomial cross-entropy by default.
        # spec S12.3: multinomial logistic — satisfied by lbfgs default.
    )
    clf.fit(X_train, y_train)
    return clf


def _predict_proba_stage1(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Predict probabilities for Stage 1 (3-class).
    Returns [N, 3] with columns in STAGE1_CLASS_ORDER = ['healthy', 'diseased', 'OOD'].
    sklearn may reorder classes; we reorder to match our ordering.
    """
    proba = clf.predict_proba(X)  # [N, n_classes_seen]
    n = X.shape[0]
    result = np.zeros((n, 3), dtype=np.float64)
    for out_idx, label in enumerate(clf.classes_):
        # label values: 0=healthy, 1=diseased, 2=OOD (from y_stage1)
        result[:, int(label)] = proba[:, out_idx]
    return result


def _predict_proba_stage2(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Predict probabilities for Stage 2 (5-class).
    Returns [N, 5] with columns in STAGE2_CLASS_ORDER.
    """
    proba = clf.predict_proba(X)  # [N, n_classes_seen]
    n = X.shape[0]
    result = np.zeros((n, 5), dtype=np.float64)
    for out_idx, label in enumerate(clf.classes_):
        # label values: 0=foliar, 1=septoria, 2=lb, 3=ylcv, 4=mosaic
        result[:, int(label)] = proba[:, out_idx]
    return result


def _apply_platt_batch(
    p_final: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Apply Platt calibration per spec S12.8 lines 3391-3397.
    # spec: section 12.8 lines 3391-3397 — apply_platt implementation
    p_final: [N, 7]; alpha, beta: [7]
    Returns [N, 7] calibrated probabilities summing to 1.
    """
    eps = 1e-12
    p = np.clip(p_final, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))  # [N, 7]
    p_cal = 1.0 / (1.0 + np.exp(-(alpha * logits + beta)))  # [N, 7]
    # Renormalize: spec S12.8 line 3396
    sums = p_cal.sum(axis=1, keepdims=True)
    sums = np.where(sums < eps, 1.0, sums)
    return p_cal / sums


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _stage_forward_raw(X: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Raw multinomial logistic forward pass for degraded-mode sim.
    Returns (n, k) probabilities via softmax.
    """
    return _softmax(X @ weights.T + bias)


def _save_quarantine(data: dict[str, Any], label: str) -> Path:
    """Save partial state to quarantine directory for STOP conditions."""
    _QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    path = _QUARANTINE_DIR / f"{label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def _save_pkl_quarantine(obj: Any, label: str) -> Path:
    """Save pickle to quarantine directory for STOP conditions."""
    _QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    path = _QUARANTINE_DIR / f"{label}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return path


# ---------------------------------------------------------------------------
# OOD re-partitioning
# ---------------------------------------------------------------------------


def _repartition_ood(
    source: np.ndarray,
    partition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-partition the 56 OOD rows into ood_heldout (14) + ood_oof (42).

    ood_heldout: 14 rows (9 model2 folders x 1 each + 5 synthetic)
    ood_oof: 42 rows (distributed across 3 folds via StratifiedKFold)
    Selection reproducible via seed=46.
    """
    rng = np.random.default_rng(_OOD_HELDOUT_SEED)
    ood_global_indices = np.where(partition == "ood")[0]
    ood_sources = source[ood_global_indices]

    heldout_local = []  # indices into ood_global_indices

    # 9 model2 folders: pick 1 image per folder (4 images per folder)
    model2_folders = sorted(set(s for s in ood_sources if "model2" in s))
    assert len(model2_folders) == 9, f"Expected 9 model2 folders, got {len(model2_folders)}"
    for folder in model2_folders:
        local_idxs = np.where(ood_sources == folder)[0]
        assert len(local_idxs) == 4, f"Expected 4 images for {folder}, got {len(local_idxs)}"
        pick = rng.integers(0, 4)
        heldout_local.append(local_idxs[pick])

    # Synthetic noise: pick 2 Gaussian + 2 solid + 1 scrambled = 5
    gaussian_local = np.where(ood_sources == "synthetic_noise_gaussian")[0]
    scrambled_local = np.where(ood_sources == "synthetic_noise_scrambled")[0]
    solid_local = np.where(ood_sources == "synthetic_noise_solid")[0]

    gaussian_picks = rng.choice(gaussian_local, size=2, replace=False)
    solid_picks = rng.choice(solid_local, size=2, replace=False)
    scrambled_picks = rng.choice(scrambled_local, size=1, replace=False)

    heldout_local.extend(gaussian_picks.tolist())
    heldout_local.extend(solid_picks.tolist())
    heldout_local.extend(scrambled_picks.tolist())

    heldout_local = np.array(sorted(heldout_local), dtype=np.int64)
    assert len(heldout_local) == 14, f"Expected 14 ood_heldout, got {len(heldout_local)}"

    # ood_oof = all other OOD rows
    all_local = np.arange(len(ood_global_indices))
    oof_local_mask = np.ones(len(all_local), dtype=bool)
    oof_local_mask[heldout_local] = False
    oof_local = all_local[oof_local_mask]
    assert len(oof_local) == 42, f"Expected 42 ood_oof, got {len(oof_local)}"

    ood_heldout_global = ood_global_indices[heldout_local]
    ood_oof_global = ood_global_indices[oof_local]

    return ood_heldout_global, ood_oof_global


# ---------------------------------------------------------------------------
# MLP variant (optional, S12.6)
# ---------------------------------------------------------------------------


def _fit_mlp_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    hidden: int = 16,
) -> Any:
    """Fit MLP classifier (19->16->n_classes).
    # spec: section 12.6 lines 3334-3338 -- MLP architecture
    """
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="lbfgs",
        max_iter=2000,
        random_state=_CV_SEED,
        alpha=1e-4,  # L2 regularization
    )
    clf.fit(X_train, y_train)
    return clf


def _predict_proba_mlp_s1(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    n = X.shape[0]
    result = np.zeros((n, 3), dtype=np.float64)
    for out_idx, label in enumerate(clf.classes_):
        result[:, int(label)] = proba[:, out_idx]
    return result


def _predict_proba_mlp_s2(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    n = X.shape[0]
    result = np.zeros((n, 5), dtype=np.float64)
    for out_idx, label in enumerate(clf.classes_):
        result[:, int(label)] = proba[:, out_idx]
    return result


# ---------------------------------------------------------------------------
# Degraded-mode simulation (M: built-in V3 check)
# ---------------------------------------------------------------------------


def _run_degraded_mode_verification(
    X_held: np.ndarray,
    y_s1_held: np.ndarray,
    y_s2_held: np.ndarray,
    final_mean: np.ndarray,
    final_std: np.ndarray,
    s1_weights: np.ndarray,
    s1_bias: np.ndarray,
    s2_weights: np.ndarray,
    s2_bias: np.ndarray,
    alpha_np: np.ndarray,
    beta_np: np.ndarray,
    jsd_sentinel: float,
) -> dict[str, Any]:
    """Run S12.7 degraded-mode verification on held_out_subset (43 rows).

    Simulation algorithm per _run_step8_degraded_mode.py reference:
      For each scenario {all_on, v3_off, lora_off, psv_off}:
        1. Copy raw held_out features
        2. Apply spec S12.2 zeroing (+ JSD sentinel when v3 or lora fails)
        3. Standardize per S12.2:3203-3204
        4. Forward Stage 1 + Stage 2 + soft-route + Platt
        5. Compute macro-F1 on canonical labels
      Verify against spec S12.7:3368-3373 thresholds.

    Returns dict with all scenario results.
    Does NOT modify any artifact files.
    STOP conditions handled by caller.
    """
    n = len(X_held)

    # Build canonical ground-truth labels per S12.10
    y_canonical = np.zeros(n, dtype=np.int64)
    for i in range(n):
        y_canonical[i] = _to_canonical(int(y_s1_held[i]), int(y_s2_held[i]))

    # Labels present in held_out_subset (no OOD expected here)
    present_labels = sorted(set(y_canonical.tolist()))

    def _run_scenario(scenario: str) -> dict[str, Any]:
        X = X_held.copy()
        for i in range(n):
            if scenario == "v3_off":
                # Zero v3: indices 0-5 + 18; JSD -> sentinel
                X[i, _V3_BLOCK_PROBS] = 0.0
                X[i, _V3_CHILLI_IDX] = 0.0
                X[i, _JSD_IDX] = jsd_sentinel
            elif scenario == "lora_off":
                # Zero lora: indices 6-11; JSD -> sentinel
                X[i, _LORA_BLOCK] = 0.0
                X[i, _JSD_IDX] = jsd_sentinel
            elif scenario == "psv_off":
                # Zero PSV: indices 12-15 + 17 (not JSD)
                for slc in _PSV_BLOCK_SLICES:
                    X[i, slc] = 0.0
            elif scenario == "all_on":
                pass
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

        # Standardize per S12.2
        X_std = np.clip((X - final_mean) / (final_std + 1e-6), -3.0, 3.0)

        # Stage 1 forward
        p_s1 = _stage_forward_raw(X_std, s1_weights, s1_bias)  # (n, 3)
        # Stage 2 forward
        p_s2 = _stage_forward_raw(X_std, s2_weights, s2_bias)  # (n, 5)
        # Soft route
        p_final = _soft_route_batch(p_s1, p_s2)                # (n, 7)
        # Platt calibration
        p_cal = _apply_platt_batch(p_final, alpha_np, beta_np)  # (n, 7)

        y_pred = p_cal.argmax(axis=1)

        macro_f1 = float(f1_score(
            y_canonical, y_pred,
            labels=present_labels,
            average="macro",
            zero_division=0,
        ))
        accuracy = float((y_pred == y_canonical).mean())

        per_class_f1 = f1_score(
            y_canonical, y_pred,
            labels=present_labels,
            average=None,
            zero_division=0,
        )
        per_class_dict = {
            CLASS_NAMES_7[lab]: round(float(per_class_f1[j]), 6)
            for j, lab in enumerate(present_labels)
        }

        return {
            "scenario": scenario,
            "n": int(n),
            "macro_f1": round(macro_f1, 6),
            "accuracy": round(accuracy, 6),
            "per_class_f1": per_class_dict,
        }

    results = {}
    for scenario in ["all_on", "v3_off", "lora_off", "psv_off"]:
        r = _run_scenario(scenario)
        thresh = _DEGRADE_THRESH.get(scenario)
        if thresh is not None:
            r["threshold"] = thresh
            r["pass"] = r["macro_f1"] >= thresh
        results[scenario] = r

    return results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_classifier() -> dict[str, Any]:
    """Full training pipeline per DEC-061 V3 dispatch.

    Single delta from V2: P_DEGRADE increased to 0.35 (from 0.20).
    Built-in S12.7 degraded-mode verification added (M).

    Returns training report dict.
    Persists artifacts to production paths on success.
    Raises ValueError (with quarantine save) on any STOP condition.

    # spec: section 12.9 lines 3408-3442 -- training procedure
    """
    print("=" * 70)
    print("TRAIN CLASSIFIER V3 — DEC-061 (P_DEGRADE=0.35)")
    print(f"  P_DEGRADE_V3={P_DEGRADE_V3}, P_DEGRADE_LORA={P_DEGRADE_LORA}, "
          f"P_DEGRADE_PSV={P_DEGRADE_PSV}, sum={_P_DEGRADE_SUM}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # A. Load features.npz
    # ------------------------------------------------------------------
    features_path = _TRAINING_DIR / "features.npz"
    if not features_path.exists():
        raise FileNotFoundError(f"features.npz not found at {features_path}")

    data = np.load(features_path, allow_pickle=True)
    X_all = data["features"].astype(np.float64)        # (259, 19)
    y_s1_all = data["y_stage1"].astype(np.int64)       # (259,) — 0=healthy,1=diseased,2=OOD
    y_s2_all = data["y_stage2"].astype(np.int64)       # (259,) — -1=N/A, 0-4=disease
    source_all = data["source_per_image"]               # (259,) object
    partition_all = data["partition"]                   # (259,) object

    assert X_all.shape == (259, 19), f"Expected (259, 19), got {X_all.shape}"
    print(f"Loaded features: {X_all.shape}")

    # Load JSD sentinel
    jsd_sentinel = _load_jsd_sentinel()
    print(f"JSD sentinel: {jsd_sentinel}")

    # ------------------------------------------------------------------
    # B. Re-partition OOD
    # ------------------------------------------------------------------
    ood_heldout_idx, ood_oof_idx = _repartition_ood(source_all, partition_all)
    print(f"OOD re-partitioned: ood_heldout={len(ood_heldout_idx)}, ood_oof={len(ood_oof_idx)}")

    train_subset_mask = partition_all == "train_subset"
    held_out_mask = partition_all == "held_out_subset"
    train_subset_idx = np.where(train_subset_mask)[0]    # 160 rows
    held_out_subset_idx = np.where(held_out_mask)[0]     # 43 rows

    assert len(train_subset_idx) == 160, f"Expected 160 train_subset, got {len(train_subset_idx)}"
    assert len(held_out_subset_idx) == 43, f"Expected 43 held_out_subset, got {len(held_out_subset_idx)}"

    # Stage 1 OOF pool: train_subset (160) + ood_oof (42) = 202 rows
    oof_pool_idx = np.concatenate([train_subset_idx, ood_oof_idx])
    assert len(oof_pool_idx) == 202, f"Expected 202 OOF pool, got {len(oof_pool_idx)}"

    X_oof_pool = X_all[oof_pool_idx]       # (202, 19)
    y_s1_oof = y_s1_all[oof_pool_idx]      # (202,) stage1 labels
    y_s2_oof = y_s2_all[oof_pool_idx]      # (202,)
    y_canon_oof = _to_canonical_batch(y_s1_oof, y_s2_oof)  # (202,) canonical+OOD

    # ------------------------------------------------------------------
    # C. 3-fold CV — StratifiedKFold(n_splits=3) per Fix 1
    # spec: dispatch note — "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)"
    # spec: section 12.9 line 3433 — source-stratified folds
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=_CV_SEED)

    # OOF prediction arrays — initialize
    oof_probabilities = np.zeros((202, NUM_CLASSES), dtype=np.float64)
    oof_labels = y_canon_oof.copy()

    # Augmentation RNG: fixed seed for reproducibility (seed=45 per DEC-060)
    aug_rng = np.random.default_rng(_AUG_SEED)

    per_fold_report = []

    # Stratify by y_stage1 for Stage 1 OOF (spec dispatch note)
    for fold_idx, (train_local, held_local) in enumerate(skf.split(X_oof_pool, y_s1_oof)):
        print(f"\nFold {fold_idx}: n_train={len(train_local)}, n_held={len(held_local)}")

        X_fold_train = X_oof_pool[train_local]
        y_s1_fold_train = y_s1_oof[train_local]
        y_s2_fold_train = y_s2_oof[train_local]

        X_fold_held = X_oof_pool[held_local]
        y_s1_fold_held = y_s1_oof[held_local]
        y_s2_fold_held = y_s2_oof[held_local]
        y_canon_held = y_canon_oof[held_local]

        # Per-fold feature standardization — fitted on train_local ONLY
        # spec: section 12.9 line 3423 — "Use train_k's standardization parameters"
        feat_mean = X_fold_train.mean(axis=0).astype(np.float64)
        feat_std = X_fold_train.std(axis=0).astype(np.float64)

        X_fold_train_std = _standardize(X_fold_train, feat_mean, feat_std)
        X_fold_held_std = _standardize(X_fold_held, feat_mean, feat_std)

        # V3 degraded-mode augmentation on training fold ONLY (not eval)
        # spec: section 12.7 lines 3348-3373 — "during training, with probability P_DEGRADE"
        X_fold_train_aug = _apply_degraded_augmentation(X_fold_train_std, aug_rng, jsd_sentinel)

        # Fit Stage 1 on all train_local
        clf_s1 = _fit_stage(X_fold_train_aug, y_s1_fold_train, n_classes=3)

        # Fit Stage 2 on diseased subset of train_local
        diseased_local_mask = y_s1_fold_train == S1_DISEASED_IDX
        n_diseased = diseased_local_mask.sum()
        if n_diseased < 5:
            print(f"  WARNING: Only {n_diseased} diseased samples in fold {fold_idx} train; using uniform Stage 2")
            p_s1_held = _predict_proba_stage1(clf_s1, X_fold_held_std)
            p_s2_held = np.full((len(X_fold_held), 5), 0.2, dtype=np.float64)
        else:
            X_dis_train = X_fold_train_aug[diseased_local_mask]
            y_dis_train = y_s2_fold_train[diseased_local_mask]
            clf_s2 = _fit_stage(X_dis_train, y_dis_train, n_classes=5)

            # Predict on held (CLEAN — no augmentation, spec S12.9 line 3423)
            p_s1_held = _predict_proba_stage1(clf_s1, X_fold_held_std)
            p_s2_held = _predict_proba_stage2(clf_s2, X_fold_held_std)

        # Soft route to 7-class
        p_final_held = _soft_route_batch(p_s1_held, p_s2_held)

        # Verify partition-of-unity per spec S12.5 line 3317
        row_sums = p_final_held.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"Fold {fold_idx}: p_final doesn't sum to 1, "
            f"max deviation={np.abs(row_sums - 1.0).max()}"
        )

        # Store OOF predictions
        oof_probabilities[held_local] = p_final_held

        # Per-fold metrics
        y_pred_s1 = np.argmax(p_s1_held, axis=1)
        stage1_f1 = float(f1_score(y_s1_fold_held, y_pred_s1, average="macro", zero_division=0))

        y_pred_7 = np.argmax(p_final_held, axis=1)
        macro_f1_7 = float(f1_score(y_canon_held, y_pred_7, average="macro", zero_division=0))

        # Stage 2 metrics (on diseased subset of held)
        held_diseased_mask = y_s1_fold_held == S1_DISEASED_IDX
        if held_diseased_mask.sum() > 0 and n_diseased >= 5:
            y_s2_held_dis = y_s2_fold_held[held_diseased_mask]
            p_s2_dis = p_s2_held[held_diseased_mask]
            y_pred_s2_dis = np.argmax(p_s2_dis, axis=1)
            stage2_f1 = float(f1_score(y_s2_held_dis, y_pred_s2_dis, average="macro", zero_division=0))
        else:
            stage2_f1 = 0.0

        ece_fold = _ece(p_final_held, y_canon_held)

        fold_report = {
            "fold": fold_idx,
            "n_train": int(len(train_local)),
            "n_held": int(len(held_local)),
            "stage1_macro_f1": round(stage1_f1, 6),
            "stage2_macro_f1": round(stage2_f1, 6),
            "macro_f1_7class": round(macro_f1_7, 6),
            "ece": round(ece_fold, 6),
        }
        per_fold_report.append(fold_report)
        print(f"  Stage1 F1={stage1_f1:.4f}  Stage2 F1={stage2_f1:.4f}  "
              f"7-class F1={macro_f1_7:.4f}  ECE={ece_fold:.4f}")

        # STOP: Stage 1 per-fold macro-F1 < 0.50
        if stage1_f1 < 0.50:
            _save_quarantine({
                "stop_reason": "Stage 1 per-fold macro-F1 below threshold",
                "fold": fold_idx,
                "stage1_macro_f1": float(stage1_f1),
                "threshold": 0.50,
                "per_fold_report": per_fold_report,
                "dispatch": "DEC-061 V3",
            }, "stop_stage1_f1")
            raise ValueError(
                f"STOP: Stage 1 fold {fold_idx} macro-F1={stage1_f1:.4f} < 0.50 threshold. "
                f"Partial state saved to {_QUARANTINE_DIR}. "
                "Main thread must adjudicate per dispatch governance language."
            )

    # ------------------------------------------------------------------
    # D. OOF aggregate metrics
    # ------------------------------------------------------------------
    y_pred_oof_7 = np.argmax(oof_probabilities, axis=1)

    # Map 7-class back to stage1 labels for S1 metric
    oof_macro_s1 = float(f1_score(y_s1_oof, [
        S1_HEALTHY_IDX if c == IDX_HEALTHY else
        S1_OOD_IDX if c == IDX_OOD else
        S1_DISEASED_IDX
        for c in y_pred_oof_7
    ], average="macro", zero_division=0))

    # Stage 2: diseased subset only
    diseased_oof_mask = y_s1_oof == S1_DISEASED_IDX
    if diseased_oof_mask.sum() > 0:
        y_s2_dis_oof = y_s2_oof[diseased_oof_mask]
        pred_7_oof_dis = np.argmax(oof_probabilities[diseased_oof_mask], axis=1)
        oof_macro_s2 = float(f1_score(y_s2_dis_oof, pred_7_oof_dis, average="macro", zero_division=0))
    else:
        oof_macro_s2 = 0.0

    oof_macro_7 = float(f1_score(oof_labels, y_pred_oof_7, average="macro", zero_division=0))
    per_class_f1_oof = f1_score(
        oof_labels, y_pred_oof_7,
        average=None, zero_division=0,
        labels=list(range(NUM_CLASSES))
    )
    oof_ece = _ece(oof_probabilities, oof_labels)

    print(f"\nOOF aggregate: S1 F1={oof_macro_s1:.4f}  S2 F1={oof_macro_s2:.4f}  "
          f"7-class F1={oof_macro_7:.4f}  ECE={oof_ece:.4f}")

    # STOP: Stage 2 OOF aggregate macro-F1 < 0.30
    if oof_macro_s2 < 0.30:
        _save_quarantine({
            "stop_reason": "Stage 2 OOF aggregate macro-F1 below threshold",
            "oof_macro_s2": float(oof_macro_s2),
            "threshold": 0.30,
            "per_fold_report": per_fold_report,
            "dispatch": "DEC-061 V3",
        }, "stop_stage2_oof_f1")
        raise ValueError(
            f"STOP: Stage 2 OOF macro-F1={oof_macro_s2:.4f} < 0.30 threshold. "
            f"Partial state saved to {_QUARANTINE_DIR}. "
            "Main thread must adjudicate per dispatch governance language."
        )

    # ------------------------------------------------------------------
    # E. Platt scaling on OOF predictions
    # spec: section 12.8 lines 3375-3406
    # ------------------------------------------------------------------
    print("\nFitting Platt scaling...")

    # Import fit_platt_scaling from validation module
    sys.path.insert(0, str(_SANDBOX_ROOT.parent))
    from tomato_sandbox.validation.fit_calibration import fit_platt_scaling

    platt_result = fit_platt_scaling(
        oof_probabilities,   # [202, 7] post-softmax probs
        oof_labels,          # [202] canonical+OOD labels
        output_dir=_CALIBRATION_DIR,
        write_file=True,
    )

    alpha_arr = platt_result["alpha"]   # list of 7 floats
    beta_arr = platt_result["beta"]     # list of 7 floats

    # Verify no NaN and beta in [-50, 50]. DO NOT CLIP. STOP on violation.
    for c, (a, b) in enumerate(zip(alpha_arr, beta_arr)):
        if math.isnan(a) or math.isnan(b):
            _save_quarantine({
                "stop_reason": "Platt fit produced NaN",
                "class_idx": c,
                "class_name": CLASS_NAMES_7[c],
                "alpha": a,
                "beta": b,
                "alpha_all": alpha_arr,
                "beta_all": beta_arr,
                "dispatch": "DEC-061 V3",
            }, "stop_platt_nan")
            raise ValueError(
                f"STOP: Platt fit produced NaN for class {c} ({CLASS_NAMES_7[c]}): "
                f"α={a}, β={b}. "
                f"Saved to {_QUARANTINE_DIR}. "
                "Main thread must adjudicate per dispatch governance language."
            )
        if not (-50.0 <= b <= 50.0):
            _save_quarantine({
                "stop_reason": "Platt beta outside [-50, 50]",
                "class_idx": c,
                "class_name": CLASS_NAMES_7[c],
                "beta": b,
                "alpha": a,
                "alpha_all": alpha_arr,
                "beta_all": beta_arr,
                "dispatch": "DEC-061 V3",
            }, "stop_platt_beta_range")
            raise ValueError(
                f"STOP: Platt β[{c}] ({CLASS_NAMES_7[c]})={b} outside [-50, 50]. "
                f"Saved to {_QUARANTINE_DIR}. "
                "Main thread must adjudicate per dispatch governance language."
            )

    # Identify soft-trigger classes (α==1.0 AND β==0.0 — identity fallback)
    soft_trigger_classes = [
        c for c, (a, b) in enumerate(zip(alpha_arr, beta_arr))
        if a == 1.0 and b == 0.0
    ]
    identity_fallback_classes = soft_trigger_classes
    print(f"  Platt alpha: {[round(a, 4) for a in alpha_arr]}")
    print(f"  Platt beta:  {[round(b, 4) for b in beta_arr]}")
    print(f"  Soft-trigger/identity classes: {[CLASS_NAMES_7[c] for c in soft_trigger_classes]}")

    alpha_np = np.array(alpha_arr, dtype=np.float64)
    beta_np = np.array(beta_arr, dtype=np.float64)

    oof_calibrated = _apply_platt_batch(oof_probabilities, alpha_np, beta_np)
    oof_ece_calibrated = _ece(oof_calibrated, oof_labels)
    print(f"  OOF ECE post-Platt: {oof_ece_calibrated:.4f}")

    # ------------------------------------------------------------------
    # F. Final model training on full data (train_subset + ood_oof + ood_heldout)
    # spec: section 12.9 lines 3431 — "final model trained on ALL 160 images"
    # dispatch note (I): train on train_subset + ood_oof + ood_heldout = 216 rows for Stage 1
    # ------------------------------------------------------------------
    print("\nTraining final model...")
    final_stage1_idx = np.concatenate([train_subset_idx, ood_oof_idx, ood_heldout_idx])
    assert len(final_stage1_idx) == 216, f"Expected 216 final train, got {len(final_stage1_idx)}"

    X_final = X_all[final_stage1_idx]
    y_s1_final = y_s1_all[final_stage1_idx]
    y_s2_final = y_s2_all[final_stage1_idx]

    # Compute final feature mean/std on Stage 1's full training set (216 rows)
    final_mean = X_final.mean(axis=0).astype(np.float64)
    final_std = X_final.std(axis=0).astype(np.float64)

    X_final_std = _standardize(X_final, final_mean, final_std)

    # V3 degraded augmentation on final training set (fresh rng same seed)
    aug_rng_final = np.random.default_rng(_AUG_SEED)
    X_final_aug = _apply_degraded_augmentation(X_final_std, aug_rng_final, jsd_sentinel)

    # Fit Stage 1 (all 216 rows)
    final_clf_s1 = _fit_stage(X_final_aug, y_s1_final, n_classes=3)

    # Fit Stage 2 (diseased subset of train_subset only)
    train_diseased_mask = (
        (partition_all[final_stage1_idx] == "train_subset")
        & (y_s1_final == S1_DISEASED_IDX)
    )
    X_s2_train = X_final_aug[train_diseased_mask]
    y_s2_train = y_s2_final[train_diseased_mask]
    print(f"  Stage 2 train n={len(X_s2_train)}")
    final_clf_s2 = _fit_stage(X_s2_train, y_s2_train, n_classes=5)

    # ------------------------------------------------------------------
    # G. Held-out 57 evaluation with final model + Platt
    # ------------------------------------------------------------------
    print("\nEvaluating on held-out 57...")
    held_57_idx = np.concatenate([held_out_subset_idx, ood_heldout_idx])
    assert len(held_57_idx) == 57, f"Expected 57 held-out, got {len(held_57_idx)}"

    X_held57 = X_all[held_57_idx]
    y_s1_held57 = y_s1_all[held_57_idx]
    y_s2_held57 = y_s2_all[held_57_idx]
    y_canon_held57 = _to_canonical_batch(y_s1_held57, y_s2_held57)

    X_held57_std = _standardize(X_held57, final_mean, final_std)
    p_s1_h57 = _predict_proba_stage1(final_clf_s1, X_held57_std)
    p_s2_h57 = _predict_proba_stage2(final_clf_s2, X_held57_std)
    p_final_h57 = _soft_route_batch(p_s1_h57, p_s2_h57)
    p_cal_h57 = _apply_platt_batch(p_final_h57, alpha_np, beta_np)

    pred_h57 = np.argmax(p_cal_h57, axis=1)
    macro_f1_h57 = float(f1_score(y_canon_held57, pred_h57, average="macro", zero_division=0))
    per_class_f1_h57 = f1_score(
        y_canon_held57, pred_h57,
        average=None, zero_division=0,
        labels=list(range(NUM_CLASSES))
    )
    ece_h57 = _ece(p_cal_h57, y_canon_held57)
    ood_f1_h57 = float(per_class_f1_h57[IDX_OOD])

    print(f"  Held-out 57: macro_F1={macro_f1_h57:.4f}  OOD_F1={ood_f1_h57:.4f}  ECE={ece_h57:.4f}")

    # ------------------------------------------------------------------
    # H. MLP variant comparison (optional, S12.6)
    # ------------------------------------------------------------------
    print("\nTraining MLP variant for comparison...")
    mlp_decision = _run_mlp_comparison(
        X_oof_pool, y_s1_oof, y_s2_oof, y_canon_oof,
        oof_macro_7, oof_ece_calibrated,
        jsd_sentinel, alpha_np, beta_np,
    )
    print(f"  MLP decision: {mlp_decision['selected_variant']} — {mlp_decision['rationale']}")

    # ------------------------------------------------------------------
    # I. Variance verification
    # ------------------------------------------------------------------
    s1_weights = np.array(final_clf_s1.coef_, dtype=np.float64)   # [3, 19]
    s2_weights = np.array(final_clf_s2.coef_, dtype=np.float64)   # [5, 19]

    s1_var_per_feature = np.var(s1_weights, axis=0)   # variance across 3 Stage 1 classes
    s2_var_per_feature = np.var(s2_weights, axis=0)   # variance across 5 Stage 2 classes

    s1_l2 = float(np.linalg.norm(s1_weights))
    s2_l2 = float(np.linalg.norm(s2_weights))

    s1_zero_var = int(np.sum(s1_var_per_feature <= 0))
    s2_zero_var = int(np.sum(s2_var_per_feature <= 0))

    # S2 class-level variance (across features)
    s2_class_var = np.var(s2_weights, axis=1)   # [5] — variance per class across features
    n_zero_class_var = int(np.sum(s2_class_var <= 0))
    s2_classes_var_gt0 = int(np.sum(s2_class_var > 0))

    print(f"\nVerification: S1 L2={s1_l2:.4f}  S2 L2={s2_l2:.4f}")
    print(f"  S1 zero-variance features: {s1_zero_var}")
    print(f"  S2 zero-variance features: {s2_zero_var}")
    print(f"  S2 classes with variance > 0: {s2_classes_var_gt0}/5")

    # STOP: > 2 classes with variance <= 0
    if n_zero_class_var > 2:
        _save_quarantine({
            "stop_reason": "Stage 2 weight variance failure: > 2 classes with variance <= 0",
            "n_zero_class_var": n_zero_class_var,
            "s2_class_var": s2_class_var.tolist(),
            "dispatch": "DEC-061 V3",
        }, "stop_variance")
        raise ValueError(
            f"STOP: Stage 2 has {n_zero_class_var} classes with weight variance <= 0 (threshold: 2). "
            f"Saved to {_QUARANTINE_DIR}. "
            "Main thread must adjudicate per dispatch governance language."
        )

    # ------------------------------------------------------------------
    # M. Built-in degraded-mode verification (NEW in V3)
    # spec: section 12.7 lines 3368-3373
    # ------------------------------------------------------------------
    print("\nRunning built-in S12.7 degraded-mode verification...")

    X_held_43 = X_all[held_out_subset_idx]
    y_s1_held_43 = y_s1_all[held_out_subset_idx]
    y_s2_held_43 = y_s2_all[held_out_subset_idx]

    degrade_results = _run_degraded_mode_verification(
        X_held=X_held_43,
        y_s1_held=y_s1_held_43,
        y_s2_held=y_s2_held_43,
        final_mean=final_mean,
        final_std=final_std,
        s1_weights=s1_weights,
        s1_bias=final_clf_s1.intercept_.astype(np.float64),
        s2_weights=s2_weights,
        s2_bias=final_clf_s2.intercept_.astype(np.float64),
        alpha_np=alpha_np,
        beta_np=beta_np,
        jsd_sentinel=jsd_sentinel,
    )

    for scenario in ["all_on", "v3_off", "lora_off", "psv_off"]:
        r = degrade_results[scenario]
        thresh = _DEGRADE_THRESH.get(scenario)
        mark = "BASELINE" if thresh is None else ("PASS" if r["macro_f1"] >= thresh else "FAIL")
        print(f"  {scenario:>10s}: macro_f1={r['macro_f1']:.4f}  ({mark})")

    # Pre-adjudicated BLK-017 bypass scenarios.
    # lora_off and psv_off threshold misses are DOCUMENTED LIMITATIONS per DEC-061
    # sub-decision / BLK-017 (data-imposed plateau; spec-prescribed iteration cap
    # honored at v2→v3; production mitigation: signal failures fire Rule 1 →
    # Tier 4B → retake-prompt). CONTINUE past these with a logged warning.
    # v3_off is NOT pre-adjudicated — still STOP on miss.
    _BLK017_BYPASS_SCENARIOS = {"lora_off", "psv_off"}

    # Annotate degrade_results with bypass field for pre-adjudicated scenarios
    for scenario in _BLK017_BYPASS_SCENARIOS:
        thresh = _DEGRADE_THRESH.get(scenario)
        if thresh is not None and degrade_results[scenario]["macro_f1"] < thresh:
            degrade_results[scenario]["bypassed_per_blk_017"] = True
        else:
            degrade_results[scenario]["bypassed_per_blk_017"] = False

    # Check thresholds and STOP if any miss (unless pre-adjudicated by BLK-017)
    for scenario, thresh in _DEGRADE_THRESH.items():
        f1_val = degrade_results[scenario]["macro_f1"]
        if f1_val < thresh:
            if scenario in _BLK017_BYPASS_SCENARIOS:
                # Pre-adjudicated: log warning and continue, do NOT raise
                # Per DEC-061 sub-decision / BLK-017: data-imposed plateau;
                # pass: false is recorded in training_report_v3.json;
                # bypassed_per_blk_017: true field added per task dispatch.
                print(
                    f"  WARNING [BLK-017 BYPASS]: {scenario} macro_f1={f1_val:.4f} "
                    f"< threshold {thresh}. "
                    f"Pre-adjudicated per DEC-061 sub-decision / BLK-017. "
                    f"Continuing — artifact save will proceed. "
                    f"pass: false recorded in training_report_v3.json with "
                    f"bypassed_per_blk_017: true."
                )
            else:
                _save_quarantine({
                    "stop_reason": f"S12.7 degraded-mode threshold miss: {scenario}",
                    "scenario": scenario,
                    "observed_f1": f1_val,
                    "threshold": thresh,
                    "all_scenarios": {
                        s: {"macro_f1": degrade_results[s]["macro_f1"]}
                        for s in ["all_on", "v3_off", "lora_off", "psv_off"]
                    },
                    "dispatch": "DEC-061 V3",
                    "spec_citation": "S12.7:3368-3373",
                }, f"stop_degraded_{scenario}")
                raise ValueError(
                    f"STOP: S12.7 degraded-mode threshold miss.\n"
                    f"  Scenario: {scenario}\n"
                    f"  Observed F1: {f1_val:.4f}\n"
                    f"  Required:    {thresh}\n"
                    f"  Spec: S12.7:3373 — if targets not met, increase P_DEGRADE and retrain.\n"
                    f"Partial state saved to {_QUARANTINE_DIR}.\n"
                    "Main thread must adjudicate per dispatch governance language."
                )

    # Summary line reflects which scenarios passed vs bypassed
    _passed = [s for s in ["v3_off", "lora_off", "psv_off"]
               if degrade_results[s].get("pass", False)]
    _bypassed = [s for s in _BLK017_BYPASS_SCENARIOS
                 if degrade_results[s].get("bypassed_per_blk_017", False)]
    if _bypassed:
        print(f"  S12.7 degraded-mode: PASS={_passed}  "
              f"BYPASSED_BLK017={_bypassed} (documented limitations)")
    else:
        print("  All S12.7 degraded-mode thresholds PASSED.")

    # ------------------------------------------------------------------
    # J. Persist artifacts
    # ------------------------------------------------------------------
    print("\nPersisting artifacts...")
    _CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1 pkl — spec S12.3 lines 3269-3275
    stage1_pkl = {
        "weights": s1_weights.astype(np.float32),                    # [3, 19]
        "bias": final_clf_s1.intercept_.astype(np.float32),          # [3]
        "temperature": 1.0,
        "feature_mean": final_mean.astype(np.float32),               # [19]
        "feature_std": final_std.astype(np.float32),                 # [19]
        "class_order": STAGE1_CLASS_ORDER,
    }
    with open(_STAGE1_PKL, "wb") as f:
        pickle.dump(stage1_pkl, f, protocol=4)
    print(f"  Written: {_STAGE1_PKL}")

    # Stage 2 pkl — spec S12.4 line 3301
    stage2_pkl = {
        "weights": s2_weights.astype(np.float32),                    # [5, 19]
        "bias": final_clf_s2.intercept_.astype(np.float32),          # [5]
        "temperature": 1.0,
        "feature_mean": final_mean.astype(np.float32),
        "feature_std": final_std.astype(np.float32),
        "class_order": STAGE2_CLASS_ORDER,
    }
    with open(_STAGE2_PKL, "wb") as f:
        pickle.dump(stage2_pkl, f, protocol=4)
    print(f"  Written: {_STAGE2_PKL}")

    # Feature standardization JSON
    # spec: section 12.2 line 3206
    std_json = {
        "feature_mean": final_mean.astype(np.float32).tolist(),
        "feature_std": final_std.astype(np.float32).tolist(),
        "n_features": 19,
        "n_train_samples": 216,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(_FEATURE_STD_JSON, "w", encoding="utf-8") as f:
        json.dump(std_json, f, indent=2)
    print(f"  Written: {_FEATURE_STD_JSON}")

    # ------------------------------------------------------------------
    # K. Build training report
    # ------------------------------------------------------------------
    oof_post_platt_f1 = float(f1_score(
        oof_labels,
        np.argmax(oof_calibrated, axis=1),
        average="macro",
        zero_division=0,
    ))

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dispatch": "DEC-061 Step 4 V3",
            "v3_delta": "P_DEGRADE increased from 0.20 to 0.35 per S12.7:3373",
            "spec_citations": [
                "S12.3:3249-3278",
                "S12.4:3279-3302",
                "S12.5:3303-3328",
                "S12.6:3330-3346",
                "S12.7:3348-3373",
                "S12.8:3375-3406",
                "S12.9:3408-3442",
                "S12.10:3444-3471",
                "S12.11:3473-3487",
            ],
            "seeds": {
                "cv": _CV_SEED,
                "augmentation": _AUG_SEED,
                "ood_heldout": _OOD_HELDOUT_SEED,
            },
            "p_degrade": round(_P_DEGRADE_SUM, 6),
            "p_degrade_blocks": {
                "no_degrade": P_NO_DEGRADE,
                "v3": P_DEGRADE_V3,
                "lora": P_DEGRADE_LORA,
                "psv": P_DEGRADE_PSV,
            },
            "n_splits": 3,
            "cv_method": "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)",
            "ood_repartition": {
                "ood_heldout": 14,
                "ood_oof": 42,
                "oof_pool_size": 202,
                "final_stage1_train_size": 216,
            },
        },
        "per_fold": per_fold_report,
        "oof_aggregate": {
            "macro_f1_stage1": round(oof_macro_s1, 6),
            "macro_f1_stage2": round(oof_macro_s2, 6),
            "macro_f1_7class": round(oof_macro_7, 6),
            "macro_f1_7class_post_platt": round(oof_post_platt_f1, 6),
            "per_class_f1": {
                CLASS_NAMES_7[i]: round(float(per_class_f1_oof[i]), 6)
                for i in range(NUM_CLASSES)
            },
            "ece_pre_platt": round(oof_ece, 6),
            "ece_post_platt": round(oof_ece_calibrated, 6),
            "n": 202,
        },
        "final_model": {
            "stage1": {
                "n_train": 216,
                "weights_l2_norm": round(s1_l2, 6),
                "weight_variance_per_feature": s1_var_per_feature.tolist(),
            },
            "stage2": {
                "n_train": int(len(X_s2_train)),
                "weights_l2_norm": round(s2_l2, 6),
            },
            "held_out_57_metrics": {
                "macro_f1_7class": round(macro_f1_h57, 6),
                "per_class_f1": {
                    CLASS_NAMES_7[i]: round(float(per_class_f1_h57[i]), 6)
                    for i in range(NUM_CLASSES)
                },
                "ood_f1": round(ood_f1_h57, 6),
                "ece": round(ece_h57, 6),
            },
        },
        "platt_fit": {
            "alpha": alpha_arr,
            "beta": beta_arr,
            "n": platt_result["n"],
            "method": platt_result["method"],
            "computed_at": platt_result["computed_at"],
            "identity_fallback_classes": identity_fallback_classes,
            "soft_trigger_classes": soft_trigger_classes,
        },
        "mlp_decision": mlp_decision,
        "verification": {
            "stage1_weight_variance_per_feature": s1_var_per_feature.tolist(),
            "stage2_weight_variance_per_feature": s2_var_per_feature.tolist(),
            "stage1_weight_l2_norm": round(s1_l2, 6),
            "stage2_weight_l2_norm": round(s2_l2, 6),
            "stage2_classes_with_variance_gt0": s2_classes_var_gt0,
            "s2_n_zero_class_variance": n_zero_class_var,
            "platt_alpha_in_range": all(-50.0 <= b <= 50.0 for b in beta_arr),
            "any_runaway_beta": any(not (-50.0 <= b <= 50.0) for b in beta_arr),
        },
        "degraded_mode_verification": {
            "spec_citation": "S12.7:3368-3373",
            "thresholds": _DEGRADE_THRESH,
            "blk_017_adjudication": (
                "lora_off and psv_off threshold misses are pre-adjudicated per "
                "DEC-061 sub-decision / BLK-017 (data-imposed plateau; "
                "production mitigation: signal failures fire Rule 1 → Tier 4B → "
                "retake-prompt). pass: false is recorded verbatim; "
                "bypassed_per_blk_017: true indicates the miss was pre-adjudicated."
            ),
            "results": {
                scenario: {
                    "macro_f1": degrade_results[scenario]["macro_f1"],
                    "threshold": _DEGRADE_THRESH.get(scenario, None),
                    "pass": degrade_results[scenario].get("pass", None),
                    # bypassed_per_blk_017 present only for BLK-017 scenarios;
                    # true iff pass==false AND scenario in _BLK017_BYPASS_SCENARIOS
                    **({
                        "bypassed_per_blk_017": degrade_results[scenario].get(
                            "bypassed_per_blk_017", False
                        )
                    } if scenario in _BLK017_BYPASS_SCENARIOS else {}),
                }
                for scenario in ["all_on", "v3_off", "lora_off", "psv_off"]
            },
            "all_pass": all(
                degrade_results[s].get("pass", True)
                for s in ["v3_off", "lora_off", "psv_off"]
            ),
            "all_pass_or_bypassed_blk017": all(
                degrade_results[s].get("pass", True)
                or degrade_results[s].get("bypassed_per_blk_017", False)
                for s in ["v3_off", "lora_off", "psv_off"]
            ),
        },
    }

    # Write training report
    _TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with open(_TRAINING_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nTraining report: {_TRAINING_REPORT}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — V3 (P_DEGRADE=0.35)")
    print(f"  Stage 1 OOF F1: {oof_macro_s1:.4f}")
    print(f"  Stage 2 OOF F1: {oof_macro_s2:.4f}")
    print(f"  7-class OOF F1: {oof_macro_7:.4f}")
    print(f"  Held-out 57 F1: {macro_f1_h57:.4f}  OOD F1: {ood_f1_h57:.4f}")
    print(f"  ECE (post-Platt): {oof_ece_calibrated:.4f}")
    print(f"  Degraded-mode: v3_off={degrade_results['v3_off']['macro_f1']:.4f}  "
          f"lora_off={degrade_results['lora_off']['macro_f1']:.4f}  "
          f"psv_off={degrade_results['psv_off']['macro_f1']:.4f}")
    print("=" * 70)

    return report


def _run_mlp_comparison(
    X_oof_pool: np.ndarray,
    y_s1_oof: np.ndarray,
    y_s2_oof: np.ndarray,
    y_canon_oof: np.ndarray,
    logistic_macro_f1: float,
    logistic_ece: float,
    jsd_sentinel: float,
    alpha_np: np.ndarray,
    beta_np: np.ndarray,
) -> dict[str, Any]:
    """Train MLP variant and decide per spec S12.6.
    # spec: section 12.6 lines 3330-3346
    Adopt MLP only if improvement >= 2pp AND ECE < 0.10.
    """
    try:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=_CV_SEED)
        aug_rng = np.random.default_rng(_AUG_SEED + 1000)

        mlp_oof_probs = np.zeros((202, NUM_CLASSES), dtype=np.float64)

        for fold_idx, (train_local, held_local) in enumerate(skf.split(X_oof_pool, y_s1_oof)):
            X_ft = X_oof_pool[train_local]
            y_s1_ft = y_s1_oof[train_local]
            y_s2_ft = y_s2_oof[train_local]

            feat_mean = X_ft.mean(axis=0)
            feat_std = X_ft.std(axis=0)
            X_ft_std = _standardize(X_ft, feat_mean, feat_std)
            X_fh_std = _standardize(X_oof_pool[held_local], feat_mean, feat_std)
            X_ft_aug = _apply_degraded_augmentation(X_ft_std, aug_rng, jsd_sentinel)

            mlp_s1 = _fit_mlp_stage(X_ft_aug, y_s1_ft, n_classes=3)
            dis_mask = y_s1_ft == S1_DISEASED_IDX
            if dis_mask.sum() >= 5:
                mlp_s2 = _fit_mlp_stage(X_ft_aug[dis_mask], y_s2_ft[dis_mask], n_classes=5)
                p_s1_h = _predict_proba_mlp_s1(mlp_s1, X_fh_std)
                p_s2_h = _predict_proba_mlp_s2(mlp_s2, X_fh_std)
            else:
                p_s1_h = _predict_proba_mlp_s1(mlp_s1, X_fh_std)
                p_s2_h = np.full((len(X_fh_std), 5), 0.2)
            mlp_oof_probs[held_local] = _soft_route_batch(p_s1_h, p_s2_h)

        mlp_cal = _apply_platt_batch(mlp_oof_probs, alpha_np, beta_np)
        mlp_pred = np.argmax(mlp_cal, axis=1)
        mlp_macro_f1 = float(f1_score(y_canon_oof, mlp_pred, average="macro", zero_division=0))
        mlp_ece = _ece(mlp_cal, y_canon_oof)

        improvement = mlp_macro_f1 - logistic_macro_f1
        # spec: S12.6 line 3342 — "at least 2 percentage points" AND "ECE remains under 0.10"
        rule_met = (improvement >= 0.02) and (mlp_ece < 0.10)
        selected = "mlp" if rule_met else "logistic"

        return {
            "ran": True,
            "logistic_macro_f1": round(logistic_macro_f1, 6),
            "mlp_macro_f1": round(mlp_macro_f1, 6),
            "logistic_ece": round(logistic_ece, 6),
            "mlp_ece": round(mlp_ece, 6),
            "rule_met": rule_met,
            "selected_variant": selected,
            "rationale": (
                f"MLP selected (improvement={improvement:.4f} >= 0.02 and ECE={mlp_ece:.4f} < 0.10)."
                if rule_met else
                f"Logistic selected (default). MLP improvement={improvement:.4f} (threshold 0.02) "
                f"or ECE={mlp_ece:.4f} (limit 0.1). Rule not met per spec S12.6 line 3342."
            ),
        }
    except Exception as e:
        return {
            "ran": False,
            "error": str(e),
            "selected_variant": "logistic",
            "rationale": f"MLP training failed: {e}; defaulting to logistic per spec S12.6.",
        }


if __name__ == "__main__":
    report = train_classifier()
    print("\nDone.")
