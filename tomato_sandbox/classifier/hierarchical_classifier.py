"""
Hierarchical stacking classifier: Stage 1 (healthy/diseased/OOD) +
Stage 2 (5-way disease) + soft-routing combination + Platt calibration.

Spec section: 12 (Hierarchical classifier), lines 3145–3505.

Defines:
  - ``ClassifierResult``          dataclass (spec S12.10 lines 3446–3458)
  - ``compute_classifier(sa, sb, sc) -> ClassifierResult``
  - ``_stage1_forward(x) -> np.ndarray``   (internal, exposed for testing)
  - ``_stage2_forward(x) -> np.ndarray``   (internal, exposed for testing)
  - ``_soft_route(p1, p2) -> np.ndarray``  (internal, exposed for testing)
  - ``_apply_platt(p) -> np.ndarray``      (internal, exposed for testing)

No GPU locking: classifier is post-signal, CPU-only numpy.
# spec: section 12.12 lines 3491–3504 — "all numpy, no GPU"
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.nan_guards import guard_array
from tomato_sandbox.classifier.feature_builder import build_classifier_input

if TYPE_CHECKING:
    from tomato_sandbox.signals.v3_signal import SignalAResult
    from tomato_sandbox.signals.lora_signal import SignalBResult
    from tomato_sandbox.signals.psv.psv import SignalCResult

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_SANDBOX_ROOT = Path(__file__).resolve().parents[1]
_CALIBRATION_DIR = _SANDBOX_ROOT / "phase_f0_calibration"

# Calibration file paths
# spec: section 12.11 lines 3481–3485
_STAGE1_PKL = _CALIBRATION_DIR / "classifier_stage1.pkl"
_STAGE2_PKL = _CALIBRATION_DIR / "classifier_stage2.pkl"
_PLATT_JSON = _CALIBRATION_DIR / "classifier_platt.json"

# ---------------------------------------------------------------------------
# Canonical class indices
# ---------------------------------------------------------------------------

# Final 7-class canonical+OOD ordering
# spec: section 12.1 lines 3151–3159; section 12.10 lines 3460–3467
NUM_FINAL_CLASSES: int = 7   # spec: 12.1 — "7 canonical classes"
IDX_FOLIAR: int = 0           # spec: 12.1 line 3152
IDX_SEPTORIA: int = 1         # spec: 12.1 line 3153
IDX_LATE_BLIGHT: int = 2      # spec: 12.1 line 3154
IDX_YLCV: int = 3             # spec: 12.1 line 3155
IDX_MOSAIC: int = 4           # spec: 12.1 line 3156
IDX_HEALTHY: int = 5          # spec: 12.1 line 3157
IDX_OOD: int = 6              # spec: 12.1 line 3158

# Stage 1 class ordering: ["healthy", "diseased", "OOD"]
# spec: section 12.3 lines 3251 — "3-class distribution: [P(healthy), P(diseased), P(OOD)]"
# spec: section 12.11 / section 12.3 storage: "class_order: ['healthy', 'diseased', 'OOD']"
_S1_HEALTHY_IDX: int = 0   # spec: 12.3 line 3251
_S1_DISEASED_IDX: int = 1  # spec: 12.3 line 3251
_S1_OOD_IDX: int = 2       # spec: 12.3 line 3251

# Stage 2 class ordering: ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]
# spec: section 12.4 lines 3281, 12.11 — "class_order = ['foliar', 'septoria',
# 'late_blight', 'ylcv', 'mosaic']"
_S2_FOLIAR_IDX: int = 0     # spec: 12.4 line 3281
_S2_SEPTORIA_IDX: int = 1   # spec: 12.4 line 3281
_S2_LB_IDX: int = 2         # spec: 12.4 line 3281
_S2_YLCV_IDX: int = 3       # spec: 12.4 line 3281
_S2_MOSAIC_IDX: int = 4     # spec: 12.4 line 3281

NUM_STAGE1_CLASSES: int = 3  # spec: 12.3 — "3-class distribution"
NUM_STAGE2_CLASSES: int = 5  # spec: 12.4 — "5-class distribution"


# ---------------------------------------------------------------------------
# ClassifierResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClassifierResult:
    """Output of ``compute_classifier``.

    Implements the dataclass from spec S12.10 verbatim (9 fields, in order).
    # spec: section 12.10 lines 3446–3458

    Downstream consumers:
    - TTA (S11.2): reads ``combined_max_prob``
    - Conformal (S13): reads ``p_final_calibrated``
    - Tier assignment (S14): reads ``combined_argmax``, ``combined_max_prob``,
      ``combined_margin``
    """

    p_final_calibrated: np.ndarray
    """[7] float32, post-Platt calibrated probabilities, sum ≈ 1.
    # spec: section 12.10 line 3449 — "post-Platt, sums to 1"
    """

    combined_argmax: int
    """Index 0-6 in canonical+OOD order (argmax of p_final_calibrated).
    # spec: section 12.10 line 3450 — "0-6 in canonical+OOD order"
    """

    combined_max_prob: float
    """Max of p_final_calibrated.  TTA reads this field.
    # spec: section 12.10 line 3451; section 12.10 line 3471 —
    # "`combined_max_prob` is the field TTA reads"
    """

    combined_margin: float
    """Max minus second-max of p_final_calibrated.
    # spec: section 12.10 line 3452 — "max minus second-max"
    """

    p_final_uncalibrated: np.ndarray
    """[7] float32, pre-Platt probabilities (soft-routed, sums to 1).
    # spec: section 12.10 line 3453 — "pre-Platt, sums to 1 (for monitoring)"
    """

    p_stage1: np.ndarray
    """[3] float32, Stage 1 output: [P(healthy), P(diseased), P(OOD)].
    # spec: section 12.10 line 3454 — "healthy/diseased/OOD probs"
    """

    p_stage2: np.ndarray
    """[5] float32, Stage 2 output: [P(foliar), P(septoria), P(lb), P(ylcv), P(mosaic)].
    Meaningful only when stage1[diseased] is high.
    # spec: section 12.10 line 3455 — "only meaningful when stage1[diseased] is high"
    """

    classifier_succeeded: bool
    """False only if input was malformed (NaN vector, wrong shape, etc.).
    # spec: section 12.10 line 3456 — "False only if input was malformed"
    """

    failure_reason: Optional[str]
    """None on success; short description on failure.
    # spec: section 12.10 line 3457
    """


# ---------------------------------------------------------------------------
# Calibration weight loading
# ---------------------------------------------------------------------------


def _load_stage_weights(pkl_path: Path, expected_class_order: list[str]) -> dict:
    """Load Stage 1 or Stage 2 weights from pickle.

    Returns dict with keys: weights [n_classes, 19], bias [n_classes],
    temperature float, class_order list[str].

    Falls back to sentinel weights if file absent or malformed.
    Sentinel: uniform logits → softmax uniform distribution.

    # spec: section 12.3 lines 3269–3277 — Stage 1 pkl schema
    # spec: section 12.4 lines 3301 — Stage 2 pkl schema
    """
    n_classes = len(expected_class_order)
    sentinel = {
        "weights": np.zeros((n_classes, 19), dtype=np.float32),
        "bias": np.zeros(n_classes, dtype=np.float32),
        "temperature": 1.0,
        "class_order": expected_class_order,
    }

    if not pkl_path.exists():
        _logger.warning(
            "stage pkl absent (pre-F.0); using sentinel weights",
            step="load_stage_weights",
            path=str(pkl_path),
        )
        return sentinel

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301 — trusted internal calibration file

        # Validate class_order (spec: section 12.3 line 3277 — "mismatch is fatal")
        # At inference level we log WARNING rather than crashing; startup checker
        # (spec S4.4 / S12.11) handles the fatal-at-startup contract.
        loaded_order = data.get("class_order", [])
        if loaded_order != expected_class_order:
            _logger.warning(
                "stage pkl class_order mismatch; using sentinel",
                step="load_stage_weights",
                expected=expected_class_order,
                got=loaded_order,
            )
            return sentinel

        return {
            "weights": np.asarray(data["weights"], dtype=np.float32),
            "bias": np.asarray(data["bias"], dtype=np.float32),
            "temperature": float(data.get("temperature", 1.0)),
            "class_order": loaded_order,
        }
    except Exception as exc:
        _logger.warning(
            "stage pkl load failed; using sentinel",
            step="load_stage_weights",
            path=str(pkl_path),
            error=str(exc),
        )
        return sentinel


def _load_platt_params() -> tuple[np.ndarray, np.ndarray]:
    """Load Platt scaling alpha and beta from calibration JSON.

    Returns:
        (alpha, beta) each of shape [7] float32.
        Falls back to identity (alpha=1, beta=0) if file absent.

    # spec: section 12.8 lines 3387 — "store α and β arrays of shape [7]
    # each in phase_f0_calibration/classifier_platt.json"
    # spec: section 12.8 lines 3391–3397 — apply_platt implementation
    """
    n = NUM_FINAL_CLASSES
    sentinel_alpha = np.ones(n, dtype=np.float32)
    sentinel_beta = np.zeros(n, dtype=np.float32)

    if not _PLATT_JSON.exists():
        _logger.warning(
            "classifier_platt.json absent (pre-F.0); using identity Platt",
            step="load_platt_params",
        )
        return sentinel_alpha, sentinel_beta

    try:
        with open(_PLATT_JSON, encoding="utf-8") as f:
            data = json.load(f)
        alpha = np.asarray(data["alpha"], dtype=np.float32)
        beta = np.asarray(data["beta"], dtype=np.float32)
        if alpha.shape != (n,) or beta.shape != (n,):
            raise ValueError(
                f"Expected shape ({n},); got alpha={alpha.shape} beta={beta.shape}"
            )
        _logger.info("classifier_platt.json loaded", step="load_platt_params")
        return alpha, beta
    except Exception as exc:
        _logger.warning(
            "classifier_platt.json load failed; using identity",
            step="load_platt_params",
            error=str(exc),
        )
        return sentinel_alpha, sentinel_beta


# Stage class orderings (used for assertions in loading)
# spec: section 12.3 line 3275; section 12.4 line 3301
_STAGE1_CLASS_ORDER = ["healthy", "diseased", "OOD"]
_STAGE2_CLASS_ORDER = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]

# Module-level loaded weights (loaded once at import time)
_stage1: dict = _load_stage_weights(_STAGE1_PKL, _STAGE1_CLASS_ORDER)
_stage2: dict = _load_stage_weights(_STAGE2_PKL, _STAGE2_CLASS_ORDER)
_platt_alpha: np.ndarray  # [7]
_platt_beta: np.ndarray   # [7]
_platt_alpha, _platt_beta = _load_platt_params()


# ---------------------------------------------------------------------------
# Stage forward functions
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    # spec: section 12.3 lines 3259-3260 — "probs = softmax(logits / T_stage1)"
    # spec: section 12.4 lines 3291 — "probs = softmax(logits / T_stage2)"
    """
    x = logits / (temperature + 1e-12)
    x = x - x.max()  # numerical stability
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum()).astype(np.float32)


def _stage1_forward(x: np.ndarray) -> np.ndarray:
    """Stage 1 forward pass: 19-dim → [P(healthy), P(diseased), P(OOD)].

    # spec: section 12.3 lines 3255-3261:
    #   Input: x ∈ R^19  (standardized)
    #   W_stage1: shape [3, 19]  (learned)
    #   b_stage1: shape [3]      (learned)
    #   logits = W_stage1 @ x + b_stage1  → shape [3]
    #   probs = softmax(logits / T_stage1) → shape [3]

    Args:
        x: [19] float32 standardized feature vector.

    Returns:
        [3] float32 probability distribution: [P(healthy), P(diseased), P(OOD)].
    """
    w = _stage1["weights"]   # [3, 19]
    b = _stage1["bias"]      # [3]
    t = _stage1["temperature"]  # scalar
    logits = w @ x + b          # [3]
    return _softmax(logits, t)  # [3]


def _stage2_forward(x: np.ndarray) -> np.ndarray:
    """Stage 2 forward pass: 19-dim → [P(foliar), P(septoria), P(lb), P(ylcv), P(mosaic)].

    # spec: section 12.4 lines 3285-3291:
    #   Input: x ∈ R^19  (same standardized vector)
    #   W_stage2: shape [5, 19]
    #   b_stage2: shape [5]
    #   logits = W_stage2 @ x + b_stage2
    #   probs = softmax(logits / T_stage2) → shape [5]

    Args:
        x: [19] float32 standardized feature vector.

    Returns:
        [5] float32 disease probability distribution.
    """
    w = _stage2["weights"]   # [5, 19]
    b = _stage2["bias"]      # [5]
    t = _stage2["temperature"]
    logits = w @ x + b       # [5]
    return _softmax(logits, t)  # [5]


# ---------------------------------------------------------------------------
# Soft routing
# ---------------------------------------------------------------------------


def _soft_route(p_stage1: np.ndarray, p_stage2: np.ndarray) -> np.ndarray:
    """Combine Stage 1 and Stage 2 via soft (multiplicative) routing.

    Produces a 7-class probability distribution in canonical+OOD order.

    # spec: section 12.5 lines 3308-3315 (verbatim equations):
    #   P_final[0]  = P_stage1[diseased] × P_stage2[foliar]
    #   P_final[1]  = P_stage1[diseased] × P_stage2[septoria]
    #   P_final[2]  = P_stage1[diseased] × P_stage2[late_blight]
    #   P_final[3]  = P_stage1[diseased] × P_stage2[ylcv]
    #   P_final[4]  = P_stage1[diseased] × P_stage2[mosaic]
    #   P_final[5]  = P_stage1[healthy]
    #   P_final[6]  = P_stage1[OOD]
    # spec: section 12.5 lines 3317-3322 — "These sum to 1"

    Args:
        p_stage1: [3] float32 [P(healthy), P(diseased), P(OOD)].
        p_stage2: [5] float32 [P(foliar), P(septoria), P(lb), P(ylcv), P(mosaic)].

    Returns:
        [7] float32 probability distribution summing to 1.
    """
    p_diseased = float(p_stage1[_S1_DISEASED_IDX])  # spec: 12.5 line 3308

    p_final = np.zeros(NUM_FINAL_CLASSES, dtype=np.float32)

    # spec: section 12.5 lines 3308-3315
    p_final[IDX_FOLIAR]      = p_diseased * float(p_stage2[_S2_FOLIAR_IDX])    # line 3308
    p_final[IDX_SEPTORIA]    = p_diseased * float(p_stage2[_S2_SEPTORIA_IDX])  # line 3309
    p_final[IDX_LATE_BLIGHT] = p_diseased * float(p_stage2[_S2_LB_IDX])        # line 3310
    p_final[IDX_YLCV]        = p_diseased * float(p_stage2[_S2_YLCV_IDX])      # line 3311
    p_final[IDX_MOSAIC]      = p_diseased * float(p_stage2[_S2_MOSAIC_IDX])    # line 3312
    p_final[IDX_HEALTHY]     = float(p_stage1[_S1_HEALTHY_IDX])                # line 3313
    p_final[IDX_OOD]         = float(p_stage1[_S1_OOD_IDX])                    # line 3314

    return p_final


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------


def _apply_platt(p_uncal: np.ndarray) -> np.ndarray:
    """Apply Platt scaling to uncalibrated 7-class probabilities.

    # spec: section 12.8 lines 3391-3397:
    #   logits = log(p / (1 - p + 1e-12) + 1e-12)
    #   p_cal_per_class = sigmoid(alpha * logits + beta)
    #   renormalize so calibrated probs sum to 1

    # spec: section 12.8 lines 3395-3400 — "renormalization is necessary"

    Args:
        p_uncal: [7] float32, pre-Platt distribution (sums to 1).

    Returns:
        [7] float32, post-Platt calibrated distribution (sums to 1).
    """
    eps = 1e-12
    p = np.clip(p_uncal.astype(np.float64), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p + eps) + eps)
    p_cal = 1.0 / (1.0 + np.exp(-(_platt_alpha.astype(np.float64) * logits + _platt_beta.astype(np.float64))))
    p_sum = p_cal.sum()
    if p_sum > eps:
        p_cal = p_cal / p_sum  # spec: section 12.8 line 3395-3396 — renormalize
    else:
        p_cal = np.ones(NUM_FINAL_CLASSES, dtype=np.float64) / NUM_FINAL_CLASSES
    return p_cal.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_classifier(
    sa: "SignalAResult",
    sb: "SignalBResult",
    sc: "SignalCResult",
) -> ClassifierResult:
    """Run the full hierarchical classifier pipeline.

    Steps (spec S12.2 + S12.3 + S12.4 + S12.5 + S12.8):
    1. Build 19-dim standardized feature vector from signal outputs.
    2. Run Stage 1 forward → p_stage1 [3].
    3. Run Stage 2 forward → p_stage2 [5].
    4. Soft-route combine → p_final_uncalibrated [7].
    5. Apply Platt calibration → p_final_calibrated [7].
    6. Compute combined_argmax, combined_max_prob, combined_margin.

    Args:
        sa: Output of Signal A (v3).
        sb: Output of Signal B (LoRA).
        sc: Output of Signal C (PSV).

    Returns:
        ``ClassifierResult`` with all 9 fields populated.
        # spec: section 12.10 lines 3446-3458
    """
    # ── Step 1: feature vector ─────────────────────────────────────────────
    try:
        x = build_classifier_input(sa, sb, sc)   # [19] standardized + clipped
    except Exception as exc:
        _logger.warning(
            "build_classifier_input failed",
            step="compute_classifier",
            error=str(exc),
        )
        # Return a failure result with uniform distribution
        uniform = np.full(NUM_FINAL_CLASSES, 1.0 / NUM_FINAL_CLASSES, dtype=np.float32)
        return ClassifierResult(
            p_final_calibrated=uniform.copy(),
            combined_argmax=0,
            combined_max_prob=float(1.0 / NUM_FINAL_CLASSES),
            combined_margin=0.0,
            p_final_uncalibrated=uniform.copy(),
            p_stage1=np.full(NUM_STAGE1_CLASSES, 1.0 / NUM_STAGE1_CLASSES, dtype=np.float32),
            p_stage2=np.full(NUM_STAGE2_CLASSES, 1.0 / NUM_STAGE2_CLASSES, dtype=np.float32),
            classifier_succeeded=False,
            failure_reason=f"feature_vector_build_failed: {type(exc).__name__}: {exc}",
        )

    # Guard: feature vector should be finite after clipping
    if not np.isfinite(x).all():
        _logger.warning(
            "feature vector contains non-finite values after clipping",
            step="compute_classifier",
        )
        uniform = np.full(NUM_FINAL_CLASSES, 1.0 / NUM_FINAL_CLASSES, dtype=np.float32)
        return ClassifierResult(
            p_final_calibrated=uniform.copy(),
            combined_argmax=0,
            combined_max_prob=float(1.0 / NUM_FINAL_CLASSES),
            combined_margin=0.0,
            p_final_uncalibrated=uniform.copy(),
            p_stage1=np.full(NUM_STAGE1_CLASSES, 1.0 / NUM_STAGE1_CLASSES, dtype=np.float32),
            p_stage2=np.full(NUM_STAGE2_CLASSES, 1.0 / NUM_STAGE2_CLASSES, dtype=np.float32),
            classifier_succeeded=False,
            failure_reason="non_finite_feature_vector",
        )

    # ── Step 2: Stage 1 forward ────────────────────────────────────────────
    # spec: section 12.3 lines 3255-3261
    p_stage1 = _stage1_forward(x)   # [3]: [P(healthy), P(diseased), P(OOD)]

    # ── Step 3: Stage 2 forward ────────────────────────────────────────────
    # spec: section 12.4 lines 3285-3291 — "Stage 2 always runs (cheap)"
    p_stage2 = _stage2_forward(x)   # [5]: [P(foliar)..P(mosaic)]

    # ── Step 4: soft routing ───────────────────────────────────────────────
    # spec: section 12.5 lines 3308-3315
    p_final_uncalibrated = _soft_route(p_stage1, p_stage2)   # [7]

    # ── Step 5: Platt calibration ──────────────────────────────────────────
    # spec: section 12.8 lines 3391-3400
    p_final_calibrated = _apply_platt(p_final_uncalibrated)  # [7]

    # Guard outputs
    p_final_calibrated = guard_array(p_final_calibrated, default_value=0.0, expected_len=NUM_FINAL_CLASSES)
    if not np.isfinite(p_final_calibrated).all() or p_final_calibrated.sum() < 1e-6:
        # Renormalize or fall back to uniform
        total = p_final_calibrated.sum()
        if total > 1e-6:
            p_final_calibrated = (p_final_calibrated / total).astype(np.float32)
        else:
            p_final_calibrated = np.full(NUM_FINAL_CLASSES, 1.0 / NUM_FINAL_CLASSES, dtype=np.float32)

    # ── Step 6: summary statistics ────────────────────────────────────────
    combined_argmax = int(np.argmax(p_final_calibrated))
    # spec: section 12.10 line 3451 — "max of p_final_calibrated"
    combined_max_prob = float(p_final_calibrated[combined_argmax])
    # spec: section 12.10 line 3452 — "max minus second-max"
    sorted_probs = np.sort(p_final_calibrated)[::-1]
    if len(sorted_probs) >= 2:
        combined_margin = float(sorted_probs[0] - sorted_probs[1])
    else:
        combined_margin = combined_max_prob

    _logger.info(
        "compute_classifier complete",
        step="compute_classifier",
        combined_argmax=combined_argmax,
        combined_max_prob=round(combined_max_prob, 4),
        combined_margin=round(combined_margin, 4),
        succeeded=True,
    )

    return ClassifierResult(
        p_final_calibrated=p_final_calibrated,
        combined_argmax=combined_argmax,
        combined_max_prob=combined_max_prob,
        combined_margin=combined_margin,
        p_final_uncalibrated=p_final_uncalibrated,
        p_stage1=p_stage1,
        p_stage2=p_stage2,
        classifier_succeeded=True,
        failure_reason=None,
    )
