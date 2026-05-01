"""
PSV Stage 2 — Disease region detection.

Identifies pixels within the leaf mask that deviate from the healthy-green median
in HSV space, then segments them into connected lesion components.

# spec: 10.4 lines 2233-2305
"""

from __future__ import annotations

import cv2
import numpy as np

from tomato_sandbox.utils.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# PSV_DISEASE_DEVIATION_THRESHOLD — placeholder; F.0 calibrates this.
# spec: 10.4 lines 2268-2270:
#   "PSV_DISEASE_DEVIATION_THRESHOLD is loaded from F.0 calibration at startup
#    (phase_f0_calibration/psv_disease_threshold.json). Placeholder value 35.0 is
#    used when F.0 has not yet produced the calibrated value."
# spec: 10.4 lines 2305 — "Threshold of 35.0: placeholder."
# ---------------------------------------------------------------------------
PSV_DISEASE_DEVIATION_THRESHOLD: float = 35.0


def detect_disease_regions(
    rgb_cc: np.ndarray,
    leaf_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Detect disease pixels within the leaf mask.

    Args:
        rgb_cc: ``[H, W, 3]`` uint8 color-constancy-applied RGB image.
        leaf_mask: ``[H, W]`` bool mask; ``True`` for leaf pixels.

    Returns:
        A tuple ``(disease_mask, lesion_stats)`` where:
          - ``disease_mask``: ``[H, W]`` bool; ``True`` for disease pixels.
          - ``lesion_stats``: dict with connected-component info:
              - ``n_lesions``: int — number of lesion components (excl. background)
              - ``labels``: ``[H, W]`` int32 CC label image
              - ``stats``: ``[N+1, 5]`` CC stats (x, y, w, h, area)
              - ``centroids``: ``[N+1, 2]`` CC centroids
              - ``leaf_area_px``: int — pixel count of leaf
              - ``disease_area_px``: int — pixel count of disease mask

    # spec: 10.4 lines 2238-2294
    """
    # Edge case: no leaf pixels — return empty results
    # spec: 10.4 lines 2246-2247
    if leaf_mask.sum() == 0:
        _log.debug("detect_disease_regions: leaf_mask is empty; returning zero masks")
        empty = np.zeros_like(leaf_mask, dtype=bool)
        return empty, {"n_lesions": 0, "labels": None, "stats": None,
                       "centroids": None, "leaf_area_px": 0, "disease_area_px": 0}

    # Compute HSV and leaf statistics
    # spec: 10.4 lines 2249-2255
    hsv = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2HSV)
    leaf_hsv = hsv[leaf_mask]
    hue_median = float(np.median(leaf_hsv[:, 0]))
    sat_median = float(np.median(leaf_hsv[:, 1]))
    val_median = float(np.median(leaf_hsv[:, 2]))

    # Compute deviation from the healthy median in HSV space
    # spec: 10.4 lines 2257-2263  weights: H×2, S×1, V×0.5
    H_dev = np.abs(hsv[:, :, 0].astype(np.int32) - int(hue_median))
    H_dev = np.minimum(H_dev, 180 - H_dev)   # circular hue distance
    S_dev = np.abs(hsv[:, :, 1].astype(np.int32) - int(sat_median))
    V_dev = np.abs(hsv[:, :, 2].astype(np.int32) - int(val_median))

    deviation = 2.0 * H_dev + 1.0 * S_dev + 0.5 * V_dev  # [H, W] float64

    # Threshold deviation inside leaf
    # spec: 10.4 lines 2265-2270
    disease_candidate = (deviation > PSV_DISEASE_DEVIATION_THRESHOLD) & leaf_mask

    # Morphological cleanup: open then close with a 3×3 kernel
    # spec: 10.4 lines 2272-2278
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(
        disease_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
    disease_mask = disease_mask.astype(bool)

    # Connected components for per-lesion stats
    # spec: 10.4 lines 2280-2293
    nb, labels, stats, centroids = cv2.connectedComponentsWithStats(
        disease_mask.astype(np.uint8)
    )
    n_lesions = nb - 1  # component 0 is background

    lesion_stats: dict = {
        "n_lesions": n_lesions,
        "labels": labels,
        "stats": stats,         # [N+1, 5]: x, y, w, h, area
        "centroids": centroids,  # [N+1, 2]
        "leaf_area_px": int(leaf_mask.sum()),
        "disease_area_px": int(disease_mask.sum()),
    }
    return disease_mask, lesion_stats
