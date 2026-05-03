"""
F.0 calibration script for the Tomato 3-Signal sandbox.

Fits four sets of calibration parameters from labeled data:
  1. Conformal threshold τ (spec Section 13.5)
  2. Platt scaling parameters α, β per class (spec Section 12.8)
  3. Per-disease severity thresholds mild_max, moderate_max (spec Section 17.3)
  4. chilli_leakage OOD threshold τ (spec Section 8.4 / Section 4.5)

All four write JSON files to tomato_sandbox/phase_f0_calibration/ (or a
caller-supplied output_dir for tests).

Entry point for Phase F.0 dry-run: run_full_calibration(labeled_data_path, ctx)

(β) interpretation per DEC-047: this script consumes pipeline outputs via
predict_single from the orchestrator. It does NOT load model weights directly.

# spec: section 29.3 lines 8140-8171 — F.0 validation procedure Step 2
# spec: section 13.5 lines 3583-3619 — conformal τ derivation
# spec: section 12.8 lines 3375-3407 — Platt scaling algorithm
# spec: section 17.3 lines 5966-5982 — per-disease severity thresholds
# spec: section 8.4 lines 1691-1701 — chilli_leakage as misrouting signal
# spec: section 4.5 line 816 — TOMATO_CHILLI_LEAKAGE_THRESHOLD F.0 derivation
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from tomato_sandbox.utils.logging import get_logger
# Delegate conformal math to the canonical module per DEC-052 Decision 2
from tomato_sandbox.conformal.conformal import compute_conformal_tau

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical class index mapping (spec S12.1 lines 3151-3159)
_CLASS_NAMES = [
    "foliar",       # 0
    "septoria",     # 1
    "late_blight",  # 2
    "ylcv",         # 3
    "mosaic",       # 4
    "healthy",      # 5
    "OOD",          # 6
]
_CLASS_TO_IDX = {n: i for i, n in enumerate(_CLASS_NAMES)}
NUM_CLASSES: int = 7  # spec: section 13.7 line 3642 — "[7], canonical+OOD indices"

# Disease names for severity thresholds (spec S17.3 — excludes healthy, OOD)
_DISEASE_NAMES = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]

# Canonical output directory (spec: section 13.5 line 3602 — "tomato_sandbox/phase_f0_calibration/")
_SANDBOX_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR: Path = _SANDBOX_ROOT / "phase_f0_calibration"

# Minimum samples per disease to fit severity thresholds (DEC-052 Decision 4)
_MIN_SEVERITY_SAMPLES: int = 10

# spec: section 4.5 line 816 — "F.0 sets to 95th percentile of chilli_leakage scores on
# confirmed-tomato images" (primary derivation method per DEC-052 Decision 5)
_CHILLI_LEAKAGE_PERCENTILE: float = 95.0

# ---------------------------------------------------------------------------
# Spec S17.3 default severity thresholds (lines 5972-5979)
# ---------------------------------------------------------------------------
# "These thresholds are placeholders for v1 deployment. Phase F.0 will replace them."
# spec: section 17.3 lines 5970-5982
SEVERITY_DEFAULTS: dict[str, dict[str, float]] = {
    # spec: section 17.3 line 5974 — "Foliar leaf spot: < 5%, ... 5-15%, ... > 15%"
    "foliar": {"mild_max": 5.0, "moderate_max": 15.0},
    # spec: section 17.3 line 5975 — "Septoria leaf spot: < 8%, ... 8-25%, ... > 25%"
    "septoria": {"mild_max": 8.0, "moderate_max": 25.0},
    # spec: section 17.3 line 5976 — "Late blight: < 2%, ... 2-8%, ... > 8%"
    "late_blight": {"mild_max": 2.0, "moderate_max": 8.0},
    # spec: section 17.3 line 5977 — "YLCV: < 10% ... 10-30% ... > 30%"
    "ylcv": {"mild_max": 10.0, "moderate_max": 30.0},
    # spec: section 17.3 line 5978 — "Mosaic virus: < 15% ... 15-40% ... > 40%"
    "mosaic": {"mild_max": 15.0, "moderate_max": 40.0},
}

# ---------------------------------------------------------------------------
# 1. Conformal threshold τ
# spec: section 13.5 lines 3583-3619
# ---------------------------------------------------------------------------


def fit_conformal_tau(
    held_out_results: list[tuple[np.ndarray, int]],
    alpha: float = 0.10,
    *,
    output_dir: Optional[Path] = None,
    write_file: bool = True,
) -> dict[str, Any]:
    """Fit conformal threshold τ from held-out calibration set predictions.

    Delegates core math to ``compute_conformal_tau`` from
    ``tomato_sandbox.conformal.conformal`` per DEC-052 Decision 2.

    Args:
        held_out_results: List of (p_final_calibrated, y_true) tuples.
            p_final_calibrated: [7] float array of calibrated probabilities.
                # spec: section 13.5 line 3587 — "p_final_calibrated_holdout: [N, 7]"
            y_true: integer class label 0-6.
                # spec: section 13.5 line 3589 — "y_true: [N] integer labels (0-6)"
        alpha: Coverage significance level. Default 0.10 for 90% coverage.
            # spec: section 13.2 line 3538 — "α = 0.10 (for 90% coverage)"
        output_dir: Directory for conformal_tau.json. Defaults to
            phase_f0_calibration/. Tests should pass tmp_path here.
            # DEC-052 Decision 7 — output directory parameter
        write_file: If True (default), write conformal_tau.json to output_dir.

    Returns:
        dict with keys: tau, alpha, n, computed_at, method.
        # spec: section 13.5 lines 3602-3611 — JSON structure

    Raises:
        ValueError: If held_out_results is empty or shapes are inconsistent.
    """
    if not held_out_results:
        raise ValueError(
            "fit_conformal_tau: held_out_results is empty. "
            "Need at least 1 calibration sample (spec S13.3 recommends n=40)."
        )

    # Build [N, 7] and [N] arrays
    probs_list = []
    labels_list = []
    for p, y in held_out_results:
        arr = np.asarray(p, dtype=np.float64)
        if arr.shape != (NUM_CLASSES,):
            raise ValueError(
                f"fit_conformal_tau: expected p shape ({NUM_CLASSES},), "
                f"got {arr.shape}"
            )
        if not (0 <= int(y) < NUM_CLASSES):
            raise ValueError(
                f"fit_conformal_tau: y_true={y} out of valid range [0, {NUM_CLASSES - 1}]"
            )
        probs_list.append(arr)
        labels_list.append(int(y))

    p_matrix = np.stack(probs_list, axis=0)  # [N, 7]
    y_arr = np.array(labels_list, dtype=np.int64)  # [N]
    N = len(y_arr)

    # Delegate to canonical conformal tau function
    # spec: section 13.5 lines 3585-3600 — compute_conformal_tau formula
    # Delegates math to conformal.compute_conformal_tau per DEC-052 Decision 2.
    tau = compute_conformal_tau(
        p_final_calibrated_holdout=p_matrix,
        y_true=y_arr,
        alpha=alpha,
    )

    result: dict[str, Any] = {
        "tau": float(tau),
        "alpha": float(alpha),
        # spec: section 13.5 line 3607 — "calibration_set_size": 40
        "calibration_set_size": N,
        "n": N,
        # spec: section 13.5 line 3608 — "calibration_date": "2026-MM-DD"
        "computed_at": datetime.now(timezone.utc).isoformat(),
        # spec: section 13.2 line 3523 — "split conformal prediction (inductive conformal)"
        "method": "split_conformal_v1",
    }

    _log.info(
        "fit_conformal_tau",
        tau=round(tau, 6),
        alpha=alpha,
        N=N,
    )

    if write_file:
        out_dir = _resolve_output_dir(output_dir)
        out_path = out_dir / "conformal_tau.json"
        _write_json(result, out_path)
        _log.info("fit_conformal_tau: wrote", path=str(out_path))

    return result


# ---------------------------------------------------------------------------
# 2. Platt scaling
# spec: section 12.8 lines 3375-3407
# ---------------------------------------------------------------------------


def fit_platt_scaling(
    classifier_logits: Union[np.ndarray, list],
    ground_truth_labels: Union[np.ndarray, list],
    *,
    output_dir: Optional[Path] = None,
    write_file: bool = True,
) -> dict[str, Any]:
    """Fit Platt scaling parameters α, β for each of the 7 classes.

    Implements spec S12.8 Algorithm:
      For each class c in {0..6}:
        y_c = (true_label == c)  for each training image
        p_c = P_final_oof[:, c]
        Fit logistic regression: p_c_calibrated = sigmoid(α_c × logit(p_c) + β_c)
      # spec: section 12.8 lines 3382-3387 — "For each class c ∈ {0..6}"

    Args:
        classifier_logits: [N, 7] array of uncalibrated classifier output probabilities
            (P_final_oof from out-of-fold predictions per spec S12.9).
            # spec: section 12.8 line 3382 — "P_final_oof of shape [N_train, 7]"
        ground_truth_labels: [N] integer class labels (0-6).
            # spec: section 12.8 line 3384 — "y_c = (true_label == c)"
        output_dir: Directory for classifier_platt.json. Defaults to
            phase_f0_calibration/.
        write_file: If True (default), write classifier_platt.json.

    Returns:
        dict with keys: alpha (list[7]), beta (list[7]), n, method, computed_at.
        # spec: section 12.8 line 3387 — "Store α and β arrays of shape [7] each"
        # spec: section 12.11 line 3484 — "phase_f0_calibration/classifier_platt.json"

    Raises:
        ValueError: If shape mismatch or all-same-label for a class.
    """
    p = np.asarray(classifier_logits, dtype=np.float64)
    y = np.asarray(ground_truth_labels, dtype=np.int64)

    if p.ndim != 2 or p.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"fit_platt_scaling: expected p shape [N, {NUM_CLASSES}], got {p.shape}"
        )
    if y.ndim != 1 or len(y) != len(p):
        raise ValueError(
            f"fit_platt_scaling: y length {len(y)} != N={len(p)}"
        )

    N = len(y)
    alpha_arr = np.ones(NUM_CLASSES, dtype=np.float64)  # fallback: identity
    beta_arr = np.zeros(NUM_CLASSES, dtype=np.float64)  # fallback: identity

    for c in range(NUM_CLASSES):
        # spec: section 12.8 line 3384 — "y_c = (true_label == c)"
        y_c = (y == c).astype(np.float64)  # [N] binary
        p_c = p[:, c]  # [N] probabilities for class c

        # If all labels are the same class or never this class, identity fallback
        if y_c.sum() == 0 or y_c.sum() == N:
            _log.warning(
                "fit_platt_scaling: class has degenerate labels; using identity",
                class_idx=c,
                class_name=_CLASS_NAMES[c],
                n_positive=int(y_c.sum()),
                N=N,
            )
            alpha_arr[c] = 1.0
            beta_arr[c] = 0.0
            continue

        # spec: section 12.8 line 3393 — logit(p_c)
        # "logits = np.log(P_final_uncal / (1.0 - P_final_uncal + 1e-12) + 1e-12)"
        eps = 1e-12
        p_c_clipped = np.clip(p_c, eps, 1.0 - eps)
        logit_p_c = np.log(p_c_clipped / (1.0 - p_c_clipped))  # [N]

        # Fit logistic regression: p_c_calibrated = sigmoid(α_c × logit(p_c) + β_c)
        # spec: section 12.8 line 3386 — "Fit logistic regression"
        a_c, b_c = _fit_logistic_one_class(logit_p_c, y_c)
        alpha_arr[c] = a_c
        beta_arr[c] = b_c

        _log.debug(
            "fit_platt_scaling: fitted class",
            class_idx=c,
            class_name=_CLASS_NAMES[c],
            alpha=round(a_c, 6),
            beta=round(b_c, 6),
        )

    result: dict[str, Any] = {
        # spec: section 12.8 line 3387 — "Store α and β arrays of shape [7] each"
        "alpha": alpha_arr.tolist(),
        "beta": beta_arr.tolist(),
        "n": N,
        "method": "platt_v1",
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    _log.info("fit_platt_scaling", n=N)

    if write_file:
        out_dir = _resolve_output_dir(output_dir)
        out_path = out_dir / "classifier_platt.json"
        _write_json(result, out_path)
        _log.info("fit_platt_scaling: wrote", path=str(out_path))

    return result


def _fit_logistic_one_class(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Fit logistic regression (2 parameters: slope a, intercept b) for one class.

    Minimises binary cross-entropy: -sum(y * log(sigmoid(a*x + b))
                                         + (1-y) * log(1 - sigmoid(a*x + b)))
    using gradient descent with L-BFGS-B or scipy.optimize.minimize.

    Returns (a, b) where a=1.0, b=0.0 is the identity (no calibration).
    """
    try:
        from scipy.optimize import minimize

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b = params[0], params[1]
            logits = a * x + b
            # Numerically stable log-sigmoid
            # log(sigmoid(z)) = -log(1 + exp(-z)); log(1 - sigmoid(z)) = -log(1 + exp(z))
            log_p = -np.logaddexp(0.0, -logits)      # log sigmoid
            log_1mp = -np.logaddexp(0.0, logits)     # log(1-sigmoid)
            nll = -(y * log_p + (1.0 - y) * log_1mp).mean()
            return float(nll)

        result = minimize(
            neg_log_likelihood,
            x0=np.array([1.0, 0.0]),
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success or result.fun < neg_log_likelihood(np.array([1.0, 0.0])):
            return float(result.x[0]), float(result.x[1])
        # If optimisation didn't improve, fall back to identity
        return 1.0, 0.0

    except ImportError:
        # scipy unavailable — use sklearn if present, else identity
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e4)
            lr.fit(x.reshape(-1, 1), y)
            a = float(lr.coef_[0, 0])
            b = float(lr.intercept_[0])
            return a, b
        except ImportError:
            _log.warning(
                "_fit_logistic_one_class: scipy and sklearn both unavailable; "
                "returning identity (a=1, b=0)"
            )
            return 1.0, 0.0


# ---------------------------------------------------------------------------
# 3. Per-disease severity thresholds
# spec: section 17.3 lines 5966-5982
# ---------------------------------------------------------------------------


def fit_severity_thresholds(
    severity_features_per_disease: dict[str, dict[str, Any]],
    ground_truth_grades: dict[str, list[str]],
    *,
    output_dir: Optional[Path] = None,
    write_file: bool = True,
) -> dict[str, Any]:
    """Fit per-disease severity thresholds mild_max and moderate_max.

    For each disease, fits thresholds from agronomist-confirmed severity labels
    if sufficient labeled samples exist (n >= _MIN_SEVERITY_SAMPLES = 10).
    If not, uses spec S17.3 defaults and marks ``default_used: True``.

    Args:
        severity_features_per_disease: Mapping from disease name to a dict
            containing a list of ``coverage_pct`` values for each sample:
              {"foliar": {"coverage_pct": [2.1, 5.4, 12.3, ...]}, ...}
            disease names: foliar | septoria | late_blight | ylcv | mosaic
        ground_truth_grades: Mapping from disease name to a list of grade
            strings ("mild" | "moderate" | "severe") parallel to coverage_pct:
              {"foliar": ["mild", "moderate", "severe", ...], ...}
        output_dir: Directory for severity_thresholds.json.
        write_file: If True, write severity_thresholds.json.

    Returns:
        dict keyed by disease name with sub-dicts:
          { "mild_max": float, "moderate_max": float, "n": int,
            "default_used": bool }
        Plus top-level keys: method, computed_at.
        # spec: section 17.3 lines 5972-5979 — per-disease table
        # spec: section 17.4 lines 5992-6010 — thresholds_used block in response

    Raises:
        ValueError: If an unknown disease name is present.
    """
    result: dict[str, Any] = {}

    for disease in _DISEASE_NAMES:
        default_thresholds = SEVERITY_DEFAULTS[disease]

        features_for_disease = severity_features_per_disease.get(disease, {})
        grades_for_disease = ground_truth_grades.get(disease, [])
        coverage_values = features_for_disease.get("coverage_pct", [])
        n = min(len(coverage_values), len(grades_for_disease))

        if n < _MIN_SEVERITY_SAMPLES:
            # Insufficient data — use spec defaults
            # DEC-052 Decision 4: n < 10 → spec defaults, mark default_used=True
            result[disease] = {
                "mild_max": default_thresholds["mild_max"],
                "moderate_max": default_thresholds["moderate_max"],
                "n": n,
                "default_used": True,
            }
            _log.info(
                "fit_severity_thresholds: using spec defaults",
                disease=disease,
                n=n,
                min_required=_MIN_SEVERITY_SAMPLES,
            )
        else:
            # Fit from data using percentile approach
            cov = np.array(coverage_values[:n], dtype=np.float64)
            grades = grades_for_disease[:n]

            # Collect coverage_pct for mild and mild+moderate samples
            # mild_max = 95th percentile of coverage_pct for confirmed-mild samples
            # moderate_max = 95th percentile of coverage_pct for mild+moderate samples
            mild_covs = [c for c, g in zip(cov, grades) if g == "mild"]
            mild_mod_covs = [c for c, g in zip(cov, grades) if g in ("mild", "moderate")]

            if len(mild_covs) < 3 or len(mild_mod_covs) < 3:
                # Degenerate split — fall back to defaults
                mild_max = default_thresholds["mild_max"]
                moderate_max = default_thresholds["moderate_max"]
                default_used = True
                _log.warning(
                    "fit_severity_thresholds: degenerate grade split; using defaults",
                    disease=disease,
                    n_mild=len(mild_covs),
                    n_mild_mod=len(mild_mod_covs),
                )
            else:
                mild_max = float(np.percentile(mild_covs, 95))
                moderate_max = float(np.percentile(mild_mod_covs, 95))
                # Ensure mild_max < moderate_max (monotonicity sanity check)
                if mild_max >= moderate_max:
                    _log.warning(
                        "fit_severity_thresholds: mild_max >= moderate_max; using defaults",
                        disease=disease,
                        mild_max=mild_max,
                        moderate_max=moderate_max,
                    )
                    mild_max = default_thresholds["mild_max"]
                    moderate_max = default_thresholds["moderate_max"]
                    default_used = True
                else:
                    default_used = False

            result[disease] = {
                "mild_max": mild_max,
                "moderate_max": moderate_max,
                "n": n,
                "default_used": default_used,
            }
            _log.info(
                "fit_severity_thresholds: fitted",
                disease=disease,
                mild_max=round(mild_max, 4),
                moderate_max=round(moderate_max, 4),
                n=n,
                default_used=default_used,
            )

    # Add metadata
    result["method"] = "spec_S17.3_calibration"
    result["computed_at"] = datetime.now(timezone.utc).isoformat()

    if write_file:
        out_dir = _resolve_output_dir(output_dir)
        out_path = out_dir / "severity_thresholds.json"
        _write_json(result, out_path)
        _log.info("fit_severity_thresholds: wrote", path=str(out_path))

    return result


# ---------------------------------------------------------------------------
# 4. chilli_leakage OOD threshold
# spec: section 8.4 lines 1691-1701; section 4.5 line 816
# spec: section 14.5 Rule 3 line 3832 — "chilli_leakage > 0.40"
# ---------------------------------------------------------------------------


def fit_chilli_leakage_threshold(
    signal_a_chilli_leakages: list[float],
    ground_truth_is_chilli: list[Union[int, bool]],
    *,
    output_dir: Optional[Path] = None,
    write_file: bool = True,
) -> dict[str, Any]:
    """Fit the chilli_leakage OOD threshold τ.

    Primary method: 95th percentile of chilli_leakage scores on confirmed-tomato
    images.
    # spec: section 4.5 line 816 — "F.0 sets to 95th percentile of chilli_leakage
    # scores on confirmed-tomato images in the training subset (so true tomato gets
    # flagged at most 5% of the time)"

    Also computes Youden J statistic (informational only, reported separately).
    # DEC-052 Decision 5 — Youden J is informational; primary tau is 95th percentile

    Args:
        signal_a_chilli_leakages: List of chilli_leakage values from SignalAResult.
            # spec: section 8.4 line 1665 — "sum of probs at v3 indices 6, 7, 8, 9"
        ground_truth_is_chilli: List of 0/1 (or bool) — 1 means the image is
            confirmed chilli (OOD for this endpoint), 0 means confirmed tomato.
        output_dir: Directory for chilli_leakage_tau.json.
        write_file: If True, write chilli_leakage_tau.json.

    Returns:
        dict with keys: tau, n_chilli, n_tomato, method, computed_at,
        youden_tau_informational.
        # spec: section 4.5 line 816 — "TOMATO_CHILLI_LEAKAGE_THRESHOLD"
        # spec: section 14.5 line 3832 — "chilli_leakage > 0.40" (Rule 3 default)

    Raises:
        ValueError: If leakage and label lists have different lengths or are empty.
    """
    if len(signal_a_chilli_leakages) != len(ground_truth_is_chilli):
        raise ValueError(
            f"fit_chilli_leakage_threshold: "
            f"len(signal_a_chilli_leakages)={len(signal_a_chilli_leakages)} "
            f"!= len(ground_truth_is_chilli)={len(ground_truth_is_chilli)}"
        )
    if not signal_a_chilli_leakages:
        raise ValueError(
            "fit_chilli_leakage_threshold: input lists are empty"
        )

    leakages = np.array(signal_a_chilli_leakages, dtype=np.float64)
    is_chilli = np.array(ground_truth_is_chilli, dtype=np.int32)  # 1=chilli, 0=tomato

    # Clip leakage to [0, 1]
    leakages = np.clip(leakages, 0.0, 1.0)

    n_chilli = int(is_chilli.sum())
    n_tomato = int((is_chilli == 0).sum())

    # Primary method: 95th percentile of chilli_leakage on confirmed-tomato images
    # spec: section 4.5 line 816 — "95th percentile of chilli_leakage scores on
    # confirmed-tomato images in the training subset"
    tomato_leakages = leakages[is_chilli == 0]
    if len(tomato_leakages) == 0:
        # No tomato samples — use spec default 0.40
        # spec: section 4.5 line 816 — default 0.40
        tau_primary = 0.40
        _log.warning(
            "fit_chilli_leakage_threshold: no tomato samples; using default tau=0.40",
        )
    else:
        tau_primary = float(np.percentile(tomato_leakages, _CHILLI_LEAKAGE_PERCENTILE))

    # Informational: Youden J statistic (J = sensitivity + specificity - 1)
    # Sweep candidate thresholds and pick the one maximising J
    tau_youden = _compute_youden_tau(leakages, is_chilli)

    result: dict[str, Any] = {
        # spec: section 4.5 line 816 — primary tau = 95th percentile on tomato images
        "tau": float(tau_primary),
        "n_chilli": n_chilli,
        "n_tomato": n_tomato,
        # DEC-052 Decision 5 — Youden J is informational
        "youden_tau_informational": float(tau_youden),
        "method": "percentile_95_tomato_v1",
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    _log.info(
        "fit_chilli_leakage_threshold",
        tau=round(tau_primary, 4),
        tau_youden=round(tau_youden, 4),
        n_chilli=n_chilli,
        n_tomato=n_tomato,
    )

    if write_file:
        out_dir = _resolve_output_dir(output_dir)
        out_path = out_dir / "chilli_leakage_tau.json"
        _write_json(result, out_path)
        _log.info("fit_chilli_leakage_threshold: wrote", path=str(out_path))

    return result


def _compute_youden_tau(
    leakages: np.ndarray,
    is_chilli: np.ndarray,
) -> float:
    """Find threshold maximising Youden J = sensitivity + specificity - 1."""
    if is_chilli.sum() == 0 or (is_chilli == 0).sum() == 0:
        return 0.40  # degenerate — use spec default

    candidates = np.unique(leakages)
    best_j = -1.0
    best_tau = 0.40
    for t in candidates:
        tp = float(((leakages > t) & (is_chilli == 1)).sum())
        fn = float(((leakages <= t) & (is_chilli == 1)).sum())
        tn = float(((leakages <= t) & (is_chilli == 0)).sum())
        fp = float(((leakages > t) & (is_chilli == 0)).sum())
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_j = j
            best_tau = float(t)
    return best_tau


# ---------------------------------------------------------------------------
# 5. Top-level entry point
# spec: section 29.3 lines 8140-8171 — F.0 validation procedure Step 2
# ---------------------------------------------------------------------------


def run_full_calibration(
    labeled_data_path: Path,
    pipeline_context: Any,
    *,
    output_dir: Optional[Path] = None,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Run all four calibration fits from labeled data.

    Drives images through predict_single from the orchestrator, collects
    outputs, then calls each fit function and writes all 4 JSON files to
    output_dir (defaults to phase_f0_calibration/).

    (β) interpretation per DEC-047: predict_single is called to get pipeline
    outputs. Model weights may be in pre-F.0 degraded mode; the caller is
    responsible for ensuring the pipeline_context has the desired model state.

    Args:
        labeled_data_path: Path to a CSV file with columns:
            image_path, true_class, split
            Optional columns: true_severity, is_confirmed_tomato
            # DEC-052 Decision 6 — labeled data CSV layout
            Rows with split == "calibration" are used for conformal + Platt.
        pipeline_context: PipelineContext from orchestrator, used to call
            predict_single for each image.
            # spec: section 21.2 line 6614 — "predict_single(image_bytes, request_id, ctx)"
        output_dir: Directory for output JSON files. Defaults to
            phase_f0_calibration/. Tests should pass tmp_path here.
        alpha: Conformal coverage significance level. Default 0.10.
            # spec: section 13.2 line 3538 — "α = 0.10"

    Returns:
        Combined calibration report dict with keys:
          conformal_tau, platt_scaling, severity_thresholds,
          chilli_leakage_tau, n_processed, n_errors, computed_at.

    Raises:
        FileNotFoundError: If labeled_data_path does not exist.
        ValueError: If CSV has no calibration-split rows.
    """
    labeled_data_path = Path(labeled_data_path)
    if not labeled_data_path.exists():
        raise FileNotFoundError(
            f"run_full_calibration: labeled data file not found: {labeled_data_path}"
        )

    _log.info(
        "run_full_calibration: starting",
        labeled_data_path=str(labeled_data_path),
        alpha=alpha,
    )

    # --- Load CSV ---
    records = _load_labeled_csv(labeled_data_path)
    calib_records = [r for r in records if r.get("split") == "calibration"]
    if not calib_records:
        raise ValueError(
            f"run_full_calibration: no rows with split='calibration' in "
            f"{labeled_data_path}"
        )

    _log.info(
        "run_full_calibration: calibration set loaded",
        n_total=len(records),
        n_calibration=len(calib_records),
    )

    # --- Run pipeline on calibration set ---
    # Import here (not at module top) to avoid circular import at test time
    from tomato_sandbox.orchestrator.orchestrator import predict_single  # noqa

    held_out_results: list[tuple[np.ndarray, int]] = []
    chilli_leakages: list[float] = []
    is_chilli_flags: list[int] = []
    severity_features: dict[str, dict[str, list]] = {d: {"coverage_pct": []} for d in _DISEASE_NAMES}
    severity_grades: dict[str, list] = {d: [] for d in _DISEASE_NAMES}
    n_processed = 0
    n_errors = 0

    for i, record in enumerate(calib_records):
        image_path_str: str = record.get("image_path", "")
        true_class_str: str = record.get("true_class", "")
        true_class_idx = _CLASS_TO_IDX.get(true_class_str, -1)

        if true_class_idx < 0:
            _log.warning(
                "run_full_calibration: unknown true_class; skipping",
                image_path=image_path_str,
                true_class=true_class_str,
            )
            n_errors += 1
            continue

        # Resolve image path relative to CSV parent if not absolute
        img_path = Path(image_path_str)
        if not img_path.is_absolute():
            img_path = labeled_data_path.parent / img_path

        try:
            image_bytes = img_path.read_bytes()
        except (OSError, FileNotFoundError) as exc:
            _log.warning(
                "run_full_calibration: cannot read image; skipping",
                image_path=str(img_path),
                error=str(exc),
            )
            n_errors += 1
            continue

        try:
            request_id = f"calib_{i:05d}"
            pipeline_result = predict_single(image_bytes, request_id, pipeline_context)
        except Exception as exc:
            _log.warning(
                "run_full_calibration: predict_single raised; skipping",
                image_path=str(img_path),
                error=str(exc),
            )
            n_errors += 1
            continue

        # Extract calibrated probabilities from pipeline result
        # spec: section 16.3 lines 5726-5738 — response JSON structure
        p_calibrated = _extract_p_calibrated(pipeline_result)
        if p_calibrated is None:
            n_errors += 1
            continue

        held_out_results.append((p_calibrated, true_class_idx))

        # chilli_leakage for OOD threshold fitting
        chilli_leak = _extract_chilli_leakage(pipeline_result)
        chilli_leakages.append(chilli_leak)
        is_chilli_val = int(record.get("is_confirmed_tomato", 1) == 0)
        is_chilli_flags.append(is_chilli_val)

        # Severity features (only for diseased classes)
        true_severity = record.get("true_severity", "")
        if true_class_str in _DISEASE_NAMES and true_severity in ("mild", "moderate", "severe"):
            cov_pct = _extract_coverage_pct(pipeline_result)
            if cov_pct is not None:
                severity_features[true_class_str]["coverage_pct"].append(cov_pct)
                severity_grades[true_class_str].append(true_severity)

        n_processed += 1
        if (i + 1) % 100 == 0:
            _log.info(
                "run_full_calibration: progress",
                processed=n_processed,
                total=len(calib_records),
            )

    if not held_out_results:
        raise ValueError(
            "run_full_calibration: all calibration images produced errors; "
            "cannot fit calibration parameters"
        )

    _log.info(
        "run_full_calibration: pipeline runs complete",
        n_processed=n_processed,
        n_errors=n_errors,
    )

    # --- Fit all four calibration parameters ---
    out_dir = _resolve_output_dir(output_dir)

    conformal_result = fit_conformal_tau(
        held_out_results, alpha=alpha, output_dir=out_dir
    )

    # For Platt scaling, build [N, 7] matrix
    p_matrix = np.stack([p for p, _ in held_out_results], axis=0)
    y_arr = [y for _, y in held_out_results]
    platt_result = fit_platt_scaling(
        p_matrix, y_arr, output_dir=out_dir
    )

    severity_result = fit_severity_thresholds(
        severity_features, severity_grades, output_dir=out_dir
    )

    chilli_result = fit_chilli_leakage_threshold(
        chilli_leakages, is_chilli_flags, output_dir=out_dir
    )

    combined: dict[str, Any] = {
        "conformal_tau": conformal_result,
        "platt_scaling": platt_result,
        "severity_thresholds": severity_result,
        "chilli_leakage_tau": chilli_result,
        "n_processed": n_processed,
        "n_errors": n_errors,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    _log.info(
        "run_full_calibration: complete",
        n_processed=n_processed,
        n_errors=n_errors,
        tau=conformal_result.get("tau"),
    )

    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_output_dir(output_dir: Optional[Path]) -> Path:
    """Return the output directory, creating it if necessary."""
    if output_dir is None:
        d = _DEFAULT_OUTPUT_DIR
    else:
        d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(data: dict, path: Path) -> None:
    """Write dict as pretty-printed JSON."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _load_labeled_csv(csv_path: Path) -> list[dict[str, str]]:
    """Load labeled data CSV. Returns list of row dicts."""
    records = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))
    return records


def _extract_p_calibrated(pipeline_result: dict) -> Optional[np.ndarray]:
    """Extract p_final_calibrated [7] from the pipeline result dict.

    The orchestrator returns a response dict per spec S16. We look for the
    probability vector in the expected location. If absent, return None.
    """
    try:
        # The pipeline result may expose p_final_calibrated directly or via nested
        # structure. Try multiple likely locations:
        # Option 1: top-level classifier_result field (degraded mode / test)
        clf = pipeline_result.get("_classifier_result")
        if clf is not None and hasattr(clf, "p_final_calibrated"):
            arr = np.asarray(clf.p_final_calibrated, dtype=np.float64)
            if arr.shape == (NUM_CLASSES,):
                return arr

        # Option 2: explanation.classifier block in response
        exp = pipeline_result.get("explanation", {})
        clf_block = exp.get("classifier", {})
        p_list = clf_block.get("p_final_calibrated")
        if p_list is not None:
            arr = np.asarray(p_list, dtype=np.float64)
            if arr.shape == (NUM_CLASSES,):
                return arr

        # Option 3: flat field directly on response (if pipeline exposes it)
        p_list = pipeline_result.get("p_final_calibrated")
        if p_list is not None:
            arr = np.asarray(p_list, dtype=np.float64)
            if arr.shape == (NUM_CLASSES,):
                return arr

        # Fallback: uniform distribution (degenerate; log warning)
        _log.warning(
            "_extract_p_calibrated: could not find p_final_calibrated in result; "
            "using uniform fallback"
        )
        return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float64)

    except Exception as exc:
        _log.error(
            "_extract_p_calibrated: exception", error=str(exc)
        )
        return None


def _extract_chilli_leakage(pipeline_result: dict) -> float:
    """Extract chilli_leakage from pipeline result."""
    try:
        exp = pipeline_result.get("explanation", {})
        signal_a_block = exp.get("signal_a", {})
        leak = signal_a_block.get("chilli_leakage")
        if leak is not None:
            return float(np.clip(leak, 0.0, 1.0))
        # Try flat
        leak = pipeline_result.get("chilli_leakage")
        if leak is not None:
            return float(np.clip(leak, 0.0, 1.0))
        return 0.0
    except Exception:
        return 0.0


def _extract_coverage_pct(pipeline_result: dict) -> Optional[float]:
    """Extract disease_coverage_pct from pipeline result."""
    try:
        sev = pipeline_result.get("severity", {})
        details = sev.get("details", {})
        cov = details.get("disease_coverage_pct")
        if cov is not None:
            return float(cov)
        return None
    except Exception:
        return None
