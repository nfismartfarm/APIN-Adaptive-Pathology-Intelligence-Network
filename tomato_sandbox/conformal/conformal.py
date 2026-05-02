"""
Conformal prediction sets for the Tomato 3-Signal sandbox.

Implements split conformal prediction (inductive conformal) per spec Section 13.

Algorithm overview (spec 13.2 lines 3521-3553):
  Calibration phase (one-shot, in F.0):
    1. Hold out a calibration set (40-image held_out_subset).
    2. For each calibration image i, compute nonconformity score:
           s_i = 1 - P_final_calibrated[i, y_true_i]
    3. Compute threshold τ at the (n+1)*(1-α)/n quantile of {s_i}.

  Inference phase (per request):
    1. For each class c, compute s_c = 1 - P_final_calibrated[c].
    2. Prediction set = {c : s_c <= τ} = {c : P_final_calibrated[c] >= 1 - τ}.

Coverage target: 90% (α = 0.10).
# spec: section 13.4 lines 3566-3581 — "Coverage target: 90%"
# spec: section 13.1 lines 3512-3519 — "The 90% target is a deliberate choice"

Input: p_final_calibrated [7] — post-Platt calibrated probabilities over
  {foliar, septoria, late_blight, ylcv, mosaic, healthy, OOD}.
  Consumed from ClassifierResult.p_final_calibrated.
  # spec: section 12.10 lines 3448-3449 — "p_final_calibrated: np.ndarray [7]"
  # BLK-010.2: field name is spec-pinned as p_final_calibrated

τ source: tomato_sandbox/phase_f0_calibration/conformal_tau.json
  # spec: section 13.5 lines 3602-3619 — τ stored at that path

DEC-040: module layout is sub-package (tomato_sandbox/conformal/conformal.py)
  per DEC-033 pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.nan_guards import guard_array

# ---------------------------------------------------------------------------
# Module logger
# spec: section 26.7 lines 7758 — "Use structlog; never print()"
# ---------------------------------------------------------------------------
_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# spec: section 13.4 lines 3566-3570 — "Coverage target: 90%. α = 0.10"
# spec: section 13.2 line 3538 — "n = 40, α = 0.10"
# ---------------------------------------------------------------------------

# spec: section 13.4 line 3567 — "Coverage target: 90%."
# spec: section 13.2 line 3538 — "α = 0.10 (for 90% coverage)"
CONFORMAL_ALPHA: float = 0.10  # spec: section 13.2 line 3538

# spec: section 13.3 line 3557 — "The calibration set is the 40-image held_out_subset"
# spec: section 13.2 line 3538 — "n = 40"
CONFORMAL_N_CALIBRATION: int = 40  # spec: section 13.2 line 3538

# spec: section 12.10 lines 3460-3467 — class index space
# spec: section 13.7 lines 3641-3647 — "[7], canonical+OOD indices"
NUM_CLASSES: int = 7  # 6 tomato classes + 1 OOD

# spec: section 13.5 lines 3602-3611 — τ file path
# "The threshold τ is stored at tomato_sandbox/phase_f0_calibration/conformal_tau.json"
_SANDBOX_ROOT: Path = Path(__file__).parent.parent  # tomato_sandbox/
CONFORMAL_TAU_PATH: Path = _SANDBOX_ROOT / "phase_f0_calibration" / "conformal_tau.json"


# ---------------------------------------------------------------------------
# Output dataclass
# spec: section 13.7 lines 3639-3647
# ---------------------------------------------------------------------------


@dataclass
class ConformalResult:
    """Output of compute_conformal_set().

    # spec: section 13.7 lines 3639-3647
    Fields:
        prediction_set: canonical+OOD indices (0-6) that are in the set.
            # spec: section 13.7 line 3642 — "canonical+OOD indices in the set"
        prediction_set_size: len(prediction_set).
            # spec: section 13.7 line 3643 — "len(prediction_set)"
        threshold_tau_used: τ value that produced this set.
            # spec: section 13.7 line 3644 — "the τ that produced this set"
        nonconformity_per_class: [7] float array, s_c = 1 - p_calibrated[c].
            # spec: section 13.7 line 3645 — "[7], 1 - p_calibrated[c] for each c"
        inside_set_per_class: [7] bool array, True if class is in the set.
            # spec: section 13.7 line 3646 — "[7] bool, True if class is in the set"
    """
    prediction_set: list[int]
    prediction_set_size: int
    threshold_tau_used: float
    nonconformity_per_class: np.ndarray  # [7] float
    inside_set_per_class: np.ndarray     # [7] bool


# ---------------------------------------------------------------------------
# τ computation (calibration phase)
# spec: section 13.5 lines 3583-3600
# ---------------------------------------------------------------------------


def compute_conformal_tau(
    p_final_calibrated_holdout: np.ndarray,
    y_true: np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
) -> float:
    """Compute the conformal threshold τ from held_out_subset predictions.

    This is the calibration-phase computation (one-shot, run at F.0 and
    monthly thereafter per spec 13.6). At inference, τ is loaded from disk
    by load_tau(); this function is not called per-request.

    Args:
        p_final_calibrated_holdout: [N, 7] array of calibrated probabilities
            on the held_out_subset.
            # spec: section 13.5 line 3587-3588 — "[N, 7] array"
        y_true: [N] integer labels (0-6).
            # spec: section 13.5 line 3589 — "y_true: [N] integer labels (0-6)"
        alpha: Coverage significance level. Default 0.10 for 90% coverage.
            # spec: section 13.2 line 3538 — "α = 0.10 (for 90% coverage)"

    Returns:
        τ ∈ [0, 1] — the nonconformity threshold.
        # spec: section 13.5 line 3590 — "Returns: τ ∈ [0, 1]"

    Raises:
        ValueError: If array shapes are inconsistent or y_true contains
            out-of-range class indices.

    # spec: section 13.5 lines 3585-3600 — verbatim algorithm
    """
    p = np.asarray(p_final_calibrated_holdout, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)

    if p.ndim != 2 or p.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"compute_conformal_tau: expected p shape [N, {NUM_CLASSES}], "
            f"got {p.shape}"
        )
    if y.ndim != 1 or len(y) != len(p):
        raise ValueError(
            f"compute_conformal_tau: y_true length {len(y)} != N={len(p)}"
        )
    if np.any(y < 0) or np.any(y >= NUM_CLASSES):
        raise ValueError(
            f"compute_conformal_tau: y_true contains out-of-range class indices "
            f"(valid: 0-{NUM_CLASSES - 1})"
        )

    N = len(y)

    # Nonconformity score: s_i = 1 - p_true_class
    # spec: section 13.2 lines 3529-3532 — "s_i = 1 - P_final_calibrated[i, y_true_i]"
    nonconformity_scores = 1.0 - p[np.arange(N), y]

    # Guard NaN from log/exp operations in upstream Platt calibration
    # spec: section 13 — conformal scores from 1-p can propagate NaN if p is NaN
    if np.any(~np.isfinite(nonconformity_scores)):
        _log.warning(
            "compute_conformal_tau: non-finite nonconformity scores detected; "
            "clipping to [0, 1]"
        )
        nonconformity_scores = np.clip(
            np.nan_to_num(nonconformity_scores, nan=1.0, posinf=1.0, neginf=0.0),
            0.0, 1.0
        )

    # Quantile level q = ceil((N+1) * (1-α)) / N
    # spec: section 13.2 lines 3533-3537 — "q = ceil((n+1)×(1-α)) / n"
    # spec: section 13.5 line 3594 — "q = np.ceil((N + 1) * (1 - alpha)) / N"
    q = np.ceil((N + 1) * (1.0 - alpha)) / N
    # spec: section 13.5 line 3595 — "q = min(q, 1.0)  # clip in case of edge case"
    q = min(q, 1.0)

    # Use "higher" interpolation for conservative (≥ 1-α) coverage guarantee
    # spec: section 13.5 lines 3596-3600 — "method='higher'": upper interpolation,
    # "ensuring the empirical coverage is at LEAST 1-α, not just approximately"
    tau = float(np.quantile(nonconformity_scores, q, method="higher"))

    _log.info(
        "compute_conformal_tau",
        N=N,
        alpha=alpha,
        q=round(q, 6),
        tau=round(tau, 6),
    )
    return tau


# ---------------------------------------------------------------------------
# τ loading from disk
# spec: section 13.5 lines 3602-3619
# ---------------------------------------------------------------------------


def load_tau(tau_path: Path = CONFORMAL_TAU_PATH) -> float:
    """Load the conformal threshold τ from conformal_tau.json.

    If the file is missing (F.0 calibration not yet run), returns a
    conservative fallback τ = 1.0, which produces the all-class set for
    every input.  This is the safe failure mode — the prediction set is
    never empty, and the coverage guarantee holds trivially (every true
    class is in a set that contains all classes).

    Args:
        tau_path: Path to conformal_tau.json.
            # spec: section 13.5 lines 3602-3611 — stored at
            # "tomato_sandbox/phase_f0_calibration/conformal_tau.json"

    Returns:
        τ as float ∈ [0, 1].

    # spec: section 13.5 lines 3602-3619
    """
    if not tau_path.exists():
        _log.warning(
            "load_tau: conformal_tau.json not found; using fallback tau=1.0",
            tau_path=str(tau_path),
        )
        return 1.0  # conservative: all-class prediction set

    try:
        with tau_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        tau = float(data["tau"])
        if not (0.0 <= tau <= 1.0):
            _log.warning(
                "load_tau: tau out of [0, 1]; clamping",
                tau_raw=tau,
            )
            tau = float(np.clip(tau, 0.0, 1.0))
        _log.info(
            "load_tau: loaded tau from disk",
            tau=tau,
            alpha=data.get("alpha"),
            calibration_set_size=data.get("calibration_set_size"),
            model_version=data.get("model_version"),
        )
        return tau
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        _log.error(
            "load_tau: failed to parse conformal_tau.json; using fallback tau=1.0",
            exc_info=exc,
        )
        return 1.0


# ---------------------------------------------------------------------------
# Inference-phase conformal set construction
# spec: section 13.2 lines 3541-3553
# ---------------------------------------------------------------------------


def compute_conformal_set(
    p_final_calibrated: np.ndarray,
    tau: Optional[float] = None,
    *,
    tau_path: Path = CONFORMAL_TAU_PATH,
) -> ConformalResult:
    """Build the conformal prediction set for a single inference request.

    Implements the inference-phase algorithm from spec Section 13.2:
      1. For each class c: s_c = 1 - P_final_calibrated[c]
      2. PredSet = {c : s_c <= τ} = {c : P_final_calibrated[c] >= 1 - τ}

    Args:
        p_final_calibrated: [7] float array of calibrated probabilities over
            {foliar, septoria, late_blight, ylcv, mosaic, healthy, OOD}.
            Consumed from ClassifierResult.p_final_calibrated.
            # spec: section 12.10 lines 3448-3449 — "p_final_calibrated: np.ndarray [7]"
            # spec: section 13.2 line 3543 — "get P_final_calibrated[c] for c = 0..6"
        tau: If provided, use this τ directly (e.g. for testing).
            If None (default), τ is loaded from tau_path.
            # spec: section 13.5 — τ loaded at startup
        tau_path: Path to conformal_tau.json. Only used when tau=None.

    Returns:
        ConformalResult with prediction_set, prediction_set_size,
        threshold_tau_used, nonconformity_per_class, inside_set_per_class.
        # spec: section 13.7 lines 3639-3647

    Notes:
        NaN guard applied to p_final_calibrated before use:
        conformal scores from 1 - p can propagate NaN if Platt calibration
        produced non-finite probabilities.  guard_array zeros-out the
        vector in that case, producing a uniform-ish nonconformity distribution
        and a large prediction set (conservative failure mode).
        # per tomato_sandbox.utils.nan_guards.guard_array usage policy

        The prediction set may be empty if the classifier assigns all
        probability mass below 1-τ.  This is rare with proper calibration
        but spec 13.2 line 3553 notes it: "Empty set is rare with proper
        calibration but possible."
    """
    # ── Guard NaN / non-finite inputs ──────────────────────────────────────
    # guard_array: if any element is NaN or length != NUM_CLASSES, returns zeros.
    # Zeros produce s_c = 1.0 for all classes (max nonconformity), which means
    # no class passes the threshold → empty set (or all-class set if τ=1.0).
    # This is the conservative failure mode per spec Section 13 intent.
    p = guard_array(p_final_calibrated, expected_len=NUM_CLASSES, default_value=0.0)

    # ── Load τ if not supplied ──────────────────────────────────────────────
    if tau is None:
        tau_val = load_tau(tau_path)
    else:
        tau_val = float(tau)
        if not np.isfinite(tau_val):
            _log.warning(
                "compute_conformal_set: non-finite tau supplied; using fallback 1.0",
                tau_raw=tau_val,
            )
            tau_val = 1.0

    # ── Nonconformity scores: s_c = 1 - p_c ────────────────────────────────
    # spec: section 13.2 lines 3543-3546 —
    # "For each class c, compute nonconformity: s_c = 1 - P_final_calibrated[c]"
    nonconformity = 1.0 - p.astype(np.float64)  # [7]

    # ── Prediction set: {c : s_c <= τ} ─────────────────────────────────────
    # spec: section 13.2 lines 3547-3550 —
    # "PredSet = {c : s_c <= τ} = {c : P_final_calibrated[c] >= 1 - τ}"
    inside = nonconformity <= tau_val  # [7] bool
    prediction_set = [int(c) for c in range(NUM_CLASSES) if inside[c]]

    result = ConformalResult(
        prediction_set=prediction_set,
        prediction_set_size=len(prediction_set),
        threshold_tau_used=tau_val,
        nonconformity_per_class=nonconformity.astype(np.float64),
        inside_set_per_class=inside,
    )

    _log.debug(
        "compute_conformal_set",
        prediction_set=prediction_set,
        prediction_set_size=len(prediction_set),
        tau=tau_val,
    )
    return result
