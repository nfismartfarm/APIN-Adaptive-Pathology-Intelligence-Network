"""
PSV orchestrator — ties the five PSV stages together.

Defines:
  - SignalCResult: output dataclass (spec: 10.9 lines 2839-2854)
  - compute_signal_c: public entry point (spec: 10.2 lines 2086-2171)

CPU-only: no GPU API, no gpu_lock. All stages use OpenCV and NumPy.

# spec: 10.2 lines 2016-2176, 10.9 lines 2837-2854, 10.10 lines 2866-2880
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.signals.psv.leaf_segmentation import segment_leaf
from tomato_sandbox.signals.psv.disease_detection import detect_disease_regions
from tomato_sandbox.signals.psv.features import compute_26_features
from tomato_sandbox.signals.psv.compatibility import (
    F0_FEATURE_MEAN,
    F0_FEATURE_STD,
    standardize_features,
    compute_compatibility_scores,
)
from tomato_sandbox.signals.psv.reliability import compute_psv_reliability, fallback_reliability

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# spec: 10.9 lines 2839-2854
# ---------------------------------------------------------------------------

@dataclass
class SignalCResult:
    """Output of compute_signal_c. All arrays are CPU numpy arrays.

    # spec: 10.9 lines 2840-2854
    """
    # [6] in canonical order: foliar, septoria, late_blight, ylcv, mosaic, healthy
    compatibility: np.ndarray
    # Index of the highest compatibility score (0-5)
    compatibility_argmax: int
    # Value of the highest compatibility score
    compatibility_max: float
    # max - second-max; used as a classifier feature
    compatibility_margin: float
    # Trustworthiness estimate in [0, 1] for this image's PSV output
    psv_reliability: float
    # [26] raw feature values before standardization; for monitoring/debug
    raw_features: np.ndarray
    # [26] post-standardization values, capped at ±3
    standardized_features: np.ndarray
    # [H, W] bool — kept for response builder UI overlays
    leaf_mask: np.ndarray
    # [H, W] bool — kept for response builder UI overlays
    disease_mask: np.ndarray
    # Number of connected lesion components
    n_lesions: int
    # True if Stage 5 fallback (IQA mask substituted for PSV mask) fired
    fallback_used: bool
    # True for any non-exceptional code path; False only on hard exception
    forward_succeeded: bool
    # None on success; exception description on failure
    failure_reason: Optional[str]


def _empty_psv_result(
    image_shape: tuple[int, ...],
    fallback_used: bool,
) -> SignalCResult:
    """Return an all-zero PSV result for both-empty-mask fallback.

    # spec: 10.8 lines 2835-2836
    """
    h = image_shape[0]
    w = image_shape[1]
    n_classes = 6
    uniform = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    return SignalCResult(
        compatibility=uniform,
        compatibility_argmax=0,
        compatibility_max=float(uniform[0]),
        compatibility_margin=0.0,
        psv_reliability=0.05,
        raw_features=np.zeros(26, dtype=np.float32),
        standardized_features=np.zeros(26, dtype=np.float32),
        leaf_mask=np.zeros((h, w), dtype=bool),
        disease_mask=np.zeros((h, w), dtype=bool),
        n_lesions=0,
        fallback_used=fallback_used,
        forward_succeeded=True,  # not an exception; controlled empty-mask path
        failure_reason=None,
    )


def compute_signal_c(
    rgb_cc: np.ndarray,                  # [H, W, 3] from preprocess_for_psv
    iqa_green_mask: "np.ndarray | None", # from IQAResult.green_mask
    iqa_aggregate_score: float,          # from IQAResult.aggregate_score
) -> SignalCResult:
    """Run the full 5-stage PSV pipeline on one image.

    All stages run on CPU. No GPU or gpu_lock used.

    Args:
        rgb_cc: ``[H, W, 3]`` uint8 color-constancy-applied RGB image.
        iqa_green_mask: ``[H, W]`` bool rough leaf mask from IQA, or None.
        iqa_aggregate_score: float ∈ [0, 1] overall image quality from IQA.

    Returns:
        SignalCResult. Never raises — exceptions are caught and produce a
        ``forward_succeeded=False`` result with uniform compatibility scores.

    # spec: 10.2 lines 2086-2171 (verbatim orchestrator wiring)
    """
    try:
        # ------------------------------------------------------------------
        # Stage 1 — Leaf segmentation
        # spec: 10.2 lines 2093-2108
        # ------------------------------------------------------------------
        leaf_mask = segment_leaf(rgb_cc)
        fallback_used = False

        if leaf_mask.sum() == 0:
            # Fallback: use IQA's mask if available
            # spec: 10.2 lines 2095-2108
            if iqa_green_mask is not None and iqa_green_mask.sum() > 0:
                leaf_mask = iqa_green_mask
                if leaf_mask.shape != rgb_cc.shape[:2]:
                    leaf_mask = cv2.resize(
                        leaf_mask.astype(np.uint8),
                        (rgb_cc.shape[1], rgb_cc.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                fallback_used = True
                _log.debug("compute_signal_c: Stage 1 empty; using IQA fallback mask")
            else:
                # Both empty: produce zero-features result
                # spec: 10.2 line 2108, 10.8 lines 2835-2836
                _log.debug("compute_signal_c: both PSV and IQA masks empty → zero result")
                return _empty_psv_result(rgb_cc.shape, fallback_used=False)

        # ------------------------------------------------------------------
        # Stage 2 — Disease region detection
        # spec: 10.2 line 2111
        # ------------------------------------------------------------------
        disease_mask, lesion_stats = detect_disease_regions(rgb_cc, leaf_mask)

        # ------------------------------------------------------------------
        # Stage 3 — 26 features (index 21 = placeholder 0)
        # spec: 10.2 lines 2113-2117
        # ------------------------------------------------------------------
        raw_features = compute_26_features(
            rgb_cc, leaf_mask, disease_mask, lesion_stats, iqa_aggregate_score
        )
        # raw_features[21] is currently 0 (placeholder); overwritten by Stage 5

        # ------------------------------------------------------------------
        # Stage 4 — Compatibility scoring
        # spec: 10.2 lines 2119-2121
        # (zero weight on feature 21 means placeholder value is OK here)
        # ------------------------------------------------------------------
        standardized = standardize_features(raw_features)
        compatibility = compute_compatibility_scores(standardized)

        # ------------------------------------------------------------------
        # Stage 5 — Reliability; updates raw_features[21]
        # spec: 10.2 lines 2123-2134
        # ------------------------------------------------------------------
        reliability = compute_psv_reliability(
            leaf_mask,
            disease_mask,
            iqa_green_mask,
            iqa_aggregate_score,
            n_lesions=lesion_stats["n_lesions"],
        )
        if fallback_used:
            # spec: 10.8 line 2831 "reliability = max(0.1, 0.3 * iqa_aggregate_score)"
            reliability = fallback_reliability(iqa_aggregate_score)

        # Write real reliability into index 21 of both raw and standardized vectors
        # spec: 10.2 lines 2130-2134
        raw_features[21] = float(reliability)
        standardized[21] = float(
            np.clip(
                (reliability - float(F0_FEATURE_MEAN[21])) / (float(F0_FEATURE_STD[21]) + 1e-6),
                -3.0,
                3.0,
            )
        )

        # ------------------------------------------------------------------
        # Build summary statistics
        # spec: 10.2 lines 2136-2139
        # ------------------------------------------------------------------
        argmax = int(np.argmax(compatibility))
        max_val = float(compatibility[argmax])
        sorted_desc = np.sort(compatibility)[::-1]
        margin = float(sorted_desc[0] - sorted_desc[1])

        _log.debug(
            "compute_signal_c succeeded",
            argmax=argmax,
            max_val=round(max_val, 4),
            margin=round(margin, 4),
            reliability=round(reliability, 4),
            n_lesions=lesion_stats["n_lesions"],
            fallback_used=fallback_used,
        )

        return SignalCResult(
            compatibility=compatibility,
            compatibility_argmax=argmax,
            compatibility_max=max_val,
            compatibility_margin=margin,
            psv_reliability=reliability,
            raw_features=raw_features,
            standardized_features=standardized,
            leaf_mask=leaf_mask,
            disease_mask=disease_mask,
            n_lesions=lesion_stats["n_lesions"],
            fallback_used=fallback_used,
            forward_succeeded=True,
            failure_reason=None,
        )

    except Exception as exc:
        # ------------------------------------------------------------------
        # Exception path — uniform distribution, forward_succeeded=False
        # spec: 10.2 lines 2156-2171
        # ------------------------------------------------------------------
        _log.error(
            "compute_signal_c exception",
            exc_type=type(exc).__name__,
            exc=str(exc),
        )
        # Safely extract shape — rgb_cc may itself be None or non-array
        try:
            h, w = int(rgb_cc.shape[0]), int(rgb_cc.shape[1])
        except Exception:
            h, w = 1, 1
        return SignalCResult(
            compatibility=np.full(6, 1.0 / 6, dtype=np.float32),
            compatibility_argmax=0,
            compatibility_max=1.0 / 6,
            compatibility_margin=0.0,
            psv_reliability=0.05,
            raw_features=np.zeros(26, dtype=np.float32),
            standardized_features=np.zeros(26, dtype=np.float32),
            leaf_mask=np.zeros((h, w), dtype=bool),
            disease_mask=np.zeros((h, w), dtype=bool),
            n_lesions=0,
            fallback_used=False,
            forward_succeeded=False,
            failure_reason=f"exception: {type(exc).__name__}",
        )
