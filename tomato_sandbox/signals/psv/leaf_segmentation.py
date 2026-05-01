"""
PSV Stage 1 — Leaf segmentation.

Produces a binary mask of leaf pixels from a color-constancy-applied RGB image.

# spec: 10.3 lines 2178-2231
"""

from __future__ import annotations

import cv2
import numpy as np

from tomato_sandbox.utils.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Kernel sizes derived from PSV_RESIZE_CAP per spec 10.3 lines 2220-2224:
#   PSV_OPEN_KERNEL_SIZE  = max(3, PSV_RESIZE_CAP // 240)
#   PSV_CLOSE_KERNEL_SIZE = max(9, PSV_RESIZE_CAP // 80)
# Default cap: 1200 → open=5, close=15
# spec: 10.3 lines 2220-2224
# ---------------------------------------------------------------------------
_PSV_RESIZE_CAP: int = 1200
PSV_OPEN_KERNEL_SIZE: int = max(3, _PSV_RESIZE_CAP // 240)   # 5 at cap=1200
PSV_CLOSE_KERNEL_SIZE: int = max(9, _PSV_RESIZE_CAP // 80)   # 15 at cap=1200


def segment_leaf(rgb_cc: np.ndarray) -> np.ndarray:
    """Segment the dominant leaf from a color-constancy-applied RGB image.

    Args:
        rgb_cc: ``[H, W, 3]`` uint8 array; output of ``preprocess_for_psv``
            (color constancy applied per Section 7.4).

    Returns:
        ``leaf_mask``: boolean ``[H, W]`` array; ``True`` for leaf pixels.
        Returns an all-False mask if no leaf-like connected component is found.

    Algorithm (spec 10.3 verbatim):
        1. Convert to HSV.
        2. Otsu threshold on saturation channel.
        3. Restrict to green-ish hue range [25, 95] (drops red/blue objects).
        4. Morphological open (small kernel) then close (large kernel).
        5. Keep only the largest connected component.

    # spec: 10.3 lines 2183-2216
    """
    # Step 1 — HSV conversion
    # spec: 10.3 lines 2188-2190
    hsv = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2HSV)

    # Step 2 — Otsu threshold on saturation
    # spec: 10.3 lines 2190-2194
    sat = hsv[:, :, 1]
    _thresh, sat_mask = cv2.threshold(
        sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    sat_mask = sat_mask.astype(bool)

    # Step 3 — Restrict to green-ish hue range
    # spec: 10.3 lines 2196-2199  hue range [25, 95]
    hue = hsv[:, :, 0]
    green_hue = (hue >= 25) & (hue <= 95)
    leaf_candidate = sat_mask & green_hue

    # Step 4 — Morphological open then close
    # spec: 10.3 lines 2201-2207
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PSV_OPEN_KERNEL_SIZE, PSV_OPEN_KERNEL_SIZE)
    )
    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PSV_CLOSE_KERNEL_SIZE, PSV_CLOSE_KERNEL_SIZE)
    )
    cleaned = cv2.morphologyEx(
        leaf_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel_small
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)

    # Step 5 — Keep only the largest connected component
    # spec: 10.3 lines 2209-2214
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if nb <= 1:
        _log.debug(
            "segment_leaf: no connected components found after morphology",
            n_components=nb,
        )
        return np.zeros_like(cleaned, dtype=bool)

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf_mask = labels == largest_idx
    return leaf_mask
