"""
PSV Stage 5 — Reliability assessment.

Produces psv_aggregate_reliability ∈ [0, 1], combining three components via
geometric mean: mask agreement with IQA, disease-coverage sanity, and IQA quality.

# spec: 10.7 lines 2753-2834
"""

from __future__ import annotations

import cv2
import numpy as np

from tomato_sandbox.utils.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fallback constants — spec: 10.8 lines 2831-2834
# "Set psv_aggregate_reliability = max(0.1, 0.3 * iqa_aggregate_score)"
# These are design-time constants, NOT F.0-calibrated.
# ---------------------------------------------------------------------------
_FALLBACK_FLOOR: float = 0.1
_FALLBACK_SCALE: float = 0.3

# IoU mapping breakpoints — spec: 10.7 lines 2786-2787
# "IoU > 0.7 → 1.0; IoU < 0.3 → 0.0; linear in between"
_IOU_LOW: float = 0.3
_IOU_HIGH: float = 0.7


def compute_psv_reliability(
    leaf_mask: np.ndarray,
    disease_mask: np.ndarray,
    iqa_green_mask: "np.ndarray | None",
    iqa_aggregate_score: float,
    n_lesions: int,
) -> float:
    """Combine reliability signals into a single score in [0, 1].

    Args:
        leaf_mask: ``[H, W]`` bool from Stage 1.
        disease_mask: ``[H, W]`` bool from Stage 2.
        iqa_green_mask: ``[H, W]`` bool or None from IQAResult.green_mask.
        iqa_aggregate_score: float ∈ [0, 1] from IQAResult.aggregate_score.
        n_lesions: number of lesion connected components (informational only).

    Returns:
        Reliability float ∈ [0, 1].

    # spec: 10.7 lines 2757-2811
    """
    # -----------------------------------------------------------------------
    # Guard: no leaf → reliability = 0.0
    # spec: 10.7 lines 2771-2772
    # -----------------------------------------------------------------------
    if leaf_mask.sum() == 0:
        _log.debug("compute_psv_reliability: empty leaf_mask → 0.0")
        return 0.0

    # -----------------------------------------------------------------------
    # Component 1 — Mask agreement: IoU between PSV leaf_mask and IQA mask
    # spec: 10.7 lines 2768-2787
    # -----------------------------------------------------------------------
    if iqa_green_mask is None or iqa_green_mask.sum() == 0:
        # No IQA mask to compare; neutral score
        # spec: 10.7 lines 2773-2774 "neutral score"
        mask_agreement = 0.5
    else:
        # Resize IQA mask if shapes differ (PSV uses post-resize-cap shape)
        # spec: 10.7 lines 2777-2782
        if iqa_green_mask.shape != leaf_mask.shape:
            iqa_green_mask = cv2.resize(
                iqa_green_mask.astype(np.uint8),
                (leaf_mask.shape[1], leaf_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        intersection = int((leaf_mask & iqa_green_mask).sum())
        union = int((leaf_mask | iqa_green_mask).sum())
        iou = intersection / max(union, 1)
        # Map IoU to reliability factor: linear in [0.3, 0.7]
        # spec: 10.7 lines 2786-2787
        mask_agreement = float(np.clip((iou - _IOU_LOW) / (_IOU_HIGH - _IOU_LOW), 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Component 2 — Coverage sanity
    # Disease coverage > 90 % → likely segmentation failure
    # spec: 10.7 lines 2789-2801
    # -----------------------------------------------------------------------
    leaf_area = int(leaf_mask.sum())
    if leaf_area == 0:
        coverage_sanity = 0.0
    else:
        coverage = disease_mask.sum() / leaf_area
        if coverage > 0.90:
            coverage_sanity = 0.2   # likely segmentation failure
        elif coverage > 0.70:
            coverage_sanity = 0.6   # very heavy disease, possible but rare
        else:
            coverage_sanity = 1.0

    # -----------------------------------------------------------------------
    # Component 3 — IQA quality factor
    # PSV reliability cannot exceed IQA's quality
    # spec: 10.7 lines 2803-2805
    # -----------------------------------------------------------------------
    iqa_factor = float(iqa_aggregate_score)

    # -----------------------------------------------------------------------
    # Geometric mean combination — any one low pulls result down
    # spec: 10.7 lines 2807-2810
    # -----------------------------------------------------------------------
    components = [mask_agreement, coverage_sanity, iqa_factor]
    components = [max(c, 0.05) for c in components]   # floor avoids log(0)
    reliability = float(np.exp(np.mean(np.log(components))))

    _log.debug(
        "compute_psv_reliability",
        mask_agreement=round(mask_agreement, 4),
        coverage_sanity=round(coverage_sanity, 4),
        iqa_factor=round(iqa_factor, 4),
        reliability=round(reliability, 4),
        n_lesions=n_lesions,
    )
    return reliability


def fallback_reliability(iqa_aggregate_score: float) -> float:
    """Reliability score for fallback-mode PSV (IQA mask used as leaf_mask).

    spec: 10.8 lines 2831 "reliability = max(0.1, 0.3 * iqa_aggregate_score)"

    Args:
        iqa_aggregate_score: IQA aggregate quality in [0, 1].

    Returns:
        Reliability float ∈ [0.1, 0.3].
    """
    return float(max(_FALLBACK_FLOOR, _FALLBACK_SCALE * iqa_aggregate_score))
