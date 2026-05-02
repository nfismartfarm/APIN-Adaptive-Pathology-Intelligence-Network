"""
Feature vector construction for the hierarchical classifier.

Spec section: 12.2 (The 19-dimensional feature vector), lines 3169–3245.

Defines:
  - ``build_classifier_input(sa, sb, sc) -> np.ndarray``   (19-dim feature vector)
  - ``jensen_shannon_divergence(p, q) -> float``            (JSD with natural log)
  - ``JSD_SENTINEL``                                         (fallback when signals fail)
  - ``load_feature_standardization() -> tuple[np.ndarray, np.ndarray]``

The 19-dim vector layout (spec: section 12.2 table, lines 3175–3196):
  Index 0-5   : v3 tomato_probs_canonical [foliar, septoria, late_blight, ylcv, mosaic, healthy]
  Index 6-11  : LoRA tomato_probs_canonical [foliar, septoria, late_blight, ylcv, mosaic, healthy]
  Index 12    : psv compatibility_max
  Index 13    : psv compatibility_margin
  Index 14    : agree_v3  (1.0 if PSV.argmax == v3 argmax, else 0.0)
  Index 15    : agree_lora (1.0 if PSV.argmax == LoRA argmax, else 0.0)
  Index 16    : JSD between v3 and LoRA distributions (natural log)
  Index 17    : psv_reliability
  Index 18    : chilli_leakage (from v3)

No GPU locking: classifier is post-signal, CPU-only numpy.
# spec: section 12.12 lines 3491–3504 — "all numpy, no GPU"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.nan_guards import guard_array
from tomato_sandbox.utils.degraded_mode import (
    VECTOR_DIM,
    SIGNAL_A_SLICES,
    SIGNAL_B_SLICES,
    SIGNAL_C_SLICES,
    apply_degraded_mode,
)

if TYPE_CHECKING:
    from tomato_sandbox.signals.v3_signal import SignalAResult
    from tomato_sandbox.signals.lora_signal import SignalBResult
    from tomato_sandbox.signals.psv.psv import SignalCResult

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to phase_f0_calibration directory (relative to sandbox root)
_SANDBOX_ROOT = Path(__file__).resolve().parents[1]
_CALIBRATION_DIR = _SANDBOX_ROOT / "phase_f0_calibration"

# JSD sentinel: median JSD observed on F.0 calibration set.
# Loaded from phase_f0_calibration/jsd_sentinel.json; default if file absent.
# spec: section 12.2 lines 3247 — "JSD_SENTINEL loaded from
# phase_f0_calibration/jsd_sentinel.json (Section 11.5)"
_JSD_SENTINEL_FILE = _CALIBRATION_DIR / "jsd_sentinel.json"
_JSD_SENTINEL_DEFAULT: float = 0.35  # reasonable prior; overridden by F.0

# Feature standardization file
# spec: section 12.2 lines 3206 — "stored in
# tomato_sandbox/phase_f0_calibration/classifier_feature_standardization.json"
_STANDARDIZATION_FILE = _CALIBRATION_DIR / "classifier_feature_standardization.json"


def _load_jsd_sentinel() -> float:
    """Load JSD sentinel from calibration file; fall back to default.

    # spec: section 12.2 lines 3247 — "JSD_SENTINEL is loaded from
    # phase_f0_calibration/jsd_sentinel.json (Section 11.5) — the median
    # JSD observed on the F.0 calibration set"
    """
    if _JSD_SENTINEL_FILE.exists():
        try:
            with open(_JSD_SENTINEL_FILE, encoding="utf-8") as f:
                data = json.load(f)
            sentinel = float(data["jsd_sentinel"])
            _logger.info(
                "jsd_sentinel loaded",
                step="load_jsd_sentinel",
                value=sentinel,
            )
            return sentinel
        except Exception as exc:
            _logger.warning(
                "jsd_sentinel load failed; using default",
                step="load_jsd_sentinel",
                error=str(exc),
                default=_JSD_SENTINEL_DEFAULT,
            )
    else:
        _logger.warning(
            "jsd_sentinel.json absent (pre-F.0); using default",
            step="load_jsd_sentinel",
            default=_JSD_SENTINEL_DEFAULT,
        )
    return _JSD_SENTINEL_DEFAULT


def load_feature_standardization() -> tuple[np.ndarray, np.ndarray]:
    """Load per-feature mean and std for classifier input standardization.

    Returns:
        (mean, std) each of shape [VECTOR_DIM == 19].

    Falls back to identity standardization (mean=0, std=1) if file absent.

    # spec: section 12.2 lines 3201–3208 — standardization formula:
    #   x_std[i] = clip((x[i] - mean[i]) / (std[i] + 1e-6), -3, 3)
    # spec: section 12.11 lines 3484–3485 — stored in
    # classifier_feature_standardization.json
    """
    if _STANDARDIZATION_FILE.exists():
        try:
            with open(_STANDARDIZATION_FILE, encoding="utf-8") as f:
                data = json.load(f)
            mean = np.array(data["feature_mean"], dtype=np.float32)
            std = np.array(data["feature_std"], dtype=np.float32)
            if mean.shape != (VECTOR_DIM,) or std.shape != (VECTOR_DIM,):
                raise ValueError(
                    f"Expected shape ({VECTOR_DIM},), got mean={mean.shape} std={std.shape}"
                )
            _logger.info(
                "classifier_feature_standardization loaded",
                step="load_feature_standardization",
            )
            return mean, std
        except Exception as exc:
            _logger.warning(
                "classifier_feature_standardization load failed; using identity",
                step="load_feature_standardization",
                error=str(exc),
            )
    else:
        _logger.warning(
            "classifier_feature_standardization.json absent (pre-F.0); using identity",
            step="load_feature_standardization",
        )
    # Identity standardization: subtract 0, divide by 1
    return np.zeros(VECTOR_DIM, dtype=np.float32), np.ones(VECTOR_DIM, dtype=np.float32)


# Module-level loaded constants (loaded once at import time)
JSD_SENTINEL: float = _load_jsd_sentinel()
CLASSIFIER_FEATURE_MEAN: np.ndarray  # [19]
CLASSIFIER_FEATURE_STD: np.ndarray   # [19]
CLASSIFIER_FEATURE_MEAN, CLASSIFIER_FEATURE_STD = load_feature_standardization()


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence (natural log)
# ---------------------------------------------------------------------------


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between p and q using natural log.

    Bounded in [0, ln(2)] ≈ [0, 0.693] when both inputs are proper
    probability distributions summing to 1.  When p or q do NOT sum to 1
    (e.g. v3's 6 tomato probs sum to 1 - chilli_leakage), the result is still
    computed as-is — the classifier standardizes it regardless.

    # spec: section 12.2 line 3193 — "JSD between v3 and LoRA distributions
    # (Section 11.5)"; natural log per spec section 11.5 lines 3061 —
    # "Range and log base: JSD computed with natural log (used here)"

    Args:
        p: Probability-like array [6], may not sum to 1 (v3 case).
        q: Probability-like array [6], sums to 1 (LoRA case).

    Returns:
        JSD scalar (float).  Returns JSD_SENTINEL if any input contains
        non-finite values or has length 0.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        return JSD_SENTINEL

    # Avoid log(0) by adding a small epsilon; clip negatives from float error
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    # Renormalize for JSD computation (JSD is defined on probability distributions)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    jsd = 0.5 * (kl_pm + kl_qm)

    if not np.isfinite(jsd):
        return JSD_SENTINEL

    return float(jsd)


# ---------------------------------------------------------------------------
# 19-dim feature vector construction
# ---------------------------------------------------------------------------


def build_classifier_input(
    sa: "SignalAResult",
    sb: "SignalBResult",
    sc: "SignalCResult",
) -> np.ndarray:
    """Assemble and standardize the 19-dimensional classifier input vector.

    Implements the ``build_classifier_input`` function from spec verbatim,
    including degraded-mode zero-filling and per-feature standardization.

    # spec: section 12.2 lines 3212–3244 — full implementation (authoritative)

    Args:
        sa: Output of Signal A (v3).  Uses ``tomato_probs_canonical``,
            ``tomato_argmax_canonical``, ``chilli_leakage``, ``forward_succeeded``.
        sb: Output of Signal B (LoRA).  Uses ``tomato_probs_canonical``,
            ``tomato_argmax_canonical``, ``forward_succeeded``.
        sc: Output of Signal C (PSV).  Uses ``compatibility_max``,
            ``compatibility_margin``, ``compatibility_argmax``,
            ``psv_reliability``, ``forward_succeeded``.

    Returns:
        Float32 numpy array of shape ``[19]``, standardized and clipped to [-3, 3].
        # spec: section 12.2 line 3204 — "clip(x_std[i], -3, 3)"
    """
    raw = np.zeros(VECTOR_DIM, dtype=np.float32)

    # ── Indices 0-5: v3 tomato canonical probs ─────────────────────────────
    # spec: section 12.2 table lines 3177-3182
    # Note: these do NOT sum to 1 (sum = 1 - chilli_leakage); do NOT renormalize
    # spec: section 12.2 lines 3197 — "The classifier sees both forms; it does
    # not re-normalize either"
    raw[0:6] = guard_array(
        np.asarray(sa.tomato_probs_canonical, dtype=np.float32),
        default_value=0.0,
        expected_len=6,
    )

    # ── Indices 6-11: LoRA tomato canonical probs ──────────────────────────
    # spec: section 12.2 table lines 3183-3188
    # LoRA probs sum to 1 (6-class softmax output)
    raw[6:12] = guard_array(
        np.asarray(sb.tomato_probs_canonical, dtype=np.float32),
        default_value=0.0,
        expected_len=6,
    )

    # ── Index 12: PSV top-1 compatibility score ────────────────────────────
    # spec: section 12.2 table line 3189 — "psv_top1: SignalCResult.compatibility_max"
    raw[12] = float(sc.compatibility_max)

    # ── Index 13: PSV compatibility margin ────────────────────────────────
    # spec: section 12.2 table line 3190 — "psv_margin: SignalCResult.compatibility_margin"
    raw[13] = float(sc.compatibility_margin)

    # ── Index 14: PSV argmax agrees with v3 argmax ────────────────────────
    # spec: section 12.2 table line 3191 — "agree_v3: 1.0 if PSV.argmax == v3.argmax"
    raw[14] = float(sc.compatibility_argmax == sa.tomato_argmax_canonical)

    # ── Index 15: PSV argmax agrees with LoRA argmax ─────────────────────
    # spec: section 12.2 table line 3192 — "agree_lora: 1.0 if PSV.argmax == LoRA.argmax"
    raw[15] = float(sc.compatibility_argmax == sb.tomato_argmax_canonical)

    # ── Index 16: JSD between v3 and LoRA ────────────────────────────────
    # spec: section 12.2 table line 3193 — "jsd_v3_lora: JSD(v3, LoRA) in [0, log 2]"
    # spec: section 12.2 lines 3225-3228 — JSD_SENTINEL when either signal failed
    if sa.forward_succeeded and sb.forward_succeeded:
        raw[16] = jensen_shannon_divergence(
            sa.tomato_probs_canonical, sb.tomato_probs_canonical
        )
    else:
        raw[16] = JSD_SENTINEL  # spec: section 12.2 line 3227 — "else JSD_SENTINEL"

    # ── Index 17: PSV reliability score ──────────────────────────────────
    # spec: section 12.2 table line 3194 — "psv_reliability: SignalCResult.psv_reliability"
    raw[17] = float(sc.psv_reliability)

    # ── Index 18: chilli leakage from v3 ─────────────────────────────────
    # spec: section 12.2 table line 3195 — "chilli_leakage: SignalAResult.chilli_leakage"
    raw[18] = float(sa.chilli_leakage)

    # ── Degraded-mode zero-filling ────────────────────────────────────────
    # spec: section 12.2 lines 3231-3242 — zero failed signal blocks
    # spec: section 12.7 lines 3364 — "At inference, signal failures are handled
    # directly in build_classifier_input: corresponding feature block zeroed
    # before standardization"
    apply_degraded_mode(
        raw,
        sa_ok=sa.forward_succeeded,
        sb_ok=sb.forward_succeeded,
        sc_ok=sc.forward_succeeded,
    )

    # ── Standardization ───────────────────────────────────────────────────
    # spec: section 12.2 lines 3203-3204:
    #   x_std[i] = (x[i] - mean[i]) / (std[i] + 1e-6)
    #   x_std[i] = clip(x_std[i], -3, 3)
    standardized = (raw - CLASSIFIER_FEATURE_MEAN) / (CLASSIFIER_FEATURE_STD + 1e-6)
    clipped = np.clip(standardized, -3.0, 3.0).astype(np.float32)

    return clipped
