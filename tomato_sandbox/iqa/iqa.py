"""
Image Quality Assessment (IQA) for the Tomato 3-Signal pipeline.

Spec reference: Section 6, lines 1053-1388.

IQA runs AFTER input validation (Section 5) and BEFORE any GPU work or PSV.
It is CPU-only with a ~40 ms median budget (spec 6.7).

Public API (spec 6.6 line 1374):
    compute_iqa(validated_image) -> IQAResult

All seven dimensions are implemented verbatim from spec Section 6.2.
Aggregation is geometric mean (spec 6.3).
Decision is four-way: REJECT / DEGRADED / ACCEPTABLE / HIGH (spec 6.4).

No print() anywhere in this module. Uses get_logger from utils.logging.
No nan_guards import: all IQA arithmetic is bounded by construction (see DEC-030).
No degraded_mode import: IQA is a precondition gate, not a signal block (DEC-030).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from tomato_sandbox.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# BAD thresholds (placeholder defaults; F.0 calibration overwrites via
# iqa_thresholds.json per spec 6.4 lines 1346-1351).
# spec: 6.4 lines 1304-1315
# ---------------------------------------------------------------------------
BAD_THRESHOLDS: dict[str, float] = {
    "sharpness": 0.20,              # spec: 6.4 line 1305
    "exposure": 0.20,               # spec: 6.4 line 1306
    "leaf_presence": 0.30,          # spec: 6.4 line 1307
    "leaf_fill": 0.30,              # spec: 6.4 line 1308
    "background_contamination": 0.30,  # spec: 6.4 line 1309
    "resolution": 0.20,             # spec: 6.4 line 1310
    "wetness": 0.30,                # spec: 6.4 line 1311
}

# Aggregate four-way decision boundaries (placeholder; spec 6.4 lines 1324-1331)
_AGG_REJECT = 0.40      # spec: 6.4 line 1324
_AGG_DEGRADED = 0.60    # spec: 6.4 line 1326
_AGG_ACCEPTABLE = 0.80  # spec: 6.4 line 1328

# Retake messages per dimension (spec 6.2.x)
# exposure has two variants stored in a tuple; others are plain strings.
_RETAKE_MESSAGES: dict[str, Any] = {
    "sharpness": (
        "Image is blurry. Hold the phone steady, tap on the leaf to focus, and re-take."
    ),  # spec: 6.2.1 line 1095
    "exposure_dark": (
        "Image is too dark. Move into more even lighting and re-take."
    ),  # spec: 6.2.2 line 1124
    "exposure_bright": (
        "Image is overexposed (too bright). Move out of direct sunlight or shade"
        " the leaf and re-take."
    ),  # spec: 6.2.2 line 1125
    "leaf_presence": (
        "I cannot find a tomato leaf in this image. Re-take with the leaf centered"
        " in the frame."
    ),  # spec: 6.2.3 line 1157
    "leaf_fill": (
        "The leaf is too far away in the frame. Move closer (10-15 cm) so the leaf"
        " fills most of the image."
    ),  # spec: 6.2.4 line 1192
    "background_contamination": (
        "Multiple leaves or strong distractors are in the frame. Isolate one leaf"
        " against a plain background and re-take."
    ),  # spec: 6.2.5 line 1224
    "resolution": (
        "Image resolution is too low. Use the phone's main camera (not zoom or"
        " screenshot) and re-take."
    ),  # spec: 6.2.6 line 1250
    "wetness": (
        "The leaf appears wet, which can be confused with disease symptoms. Wait for"
        " the leaf to dry and re-take."
    ),  # spec: 6.2.7 line 1278
}


# ---------------------------------------------------------------------------
# Output data structure
# spec: 6.5 lines 1357-1365
# ---------------------------------------------------------------------------

@dataclass
class IQAResult:
    """
    Output of compute_iqa().

    Fields (verbatim from spec 6.5 lines 1357-1365):
        decision          : "REJECT" | "DEGRADED" | "ACCEPTABLE" | "HIGH"
        aggregate_score   : float in [0, 1]
        per_dimension     : dict[str, float]  — 7 entries
        failing_dimensions: list[str]          — names where score < BAD_THRESHOLD
        retake_message    : str | None         — if REJECT, the user-facing message
        green_mask        : np.ndarray | None  — rough HSV green mask; passed to PSV
    """

    decision: str                          # spec: 6.5 line 1359
    aggregate_score: float                 # spec: 6.5 line 1360
    per_dimension: dict[str, float]        # spec: 6.5 line 1361  (7 entries)
    failing_dimensions: list[str] = field(default_factory=list)   # spec: 6.5 line 1362
    retake_message: str | None = None      # spec: 6.5 line 1363
    green_mask: np.ndarray | None = None   # spec: 6.5 line 1364


# ---------------------------------------------------------------------------
# Per-dimension measurement functions
# Each function is verbatim from spec Section 6.2.x.
# ---------------------------------------------------------------------------


def _sharpness(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian (ksize=3) normalized to [0, 1].
    Saturates at variance = 1000.

    spec: 6.2.1 lines 1081-1086
    """
    # spec: 6.2.1 line 1083 — Laplacian with CV_64F, ksize=3
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    raw_variance = lap.var()
    # spec: 6.2.1 lines 1085-1086 — normalize; variance > 1000 → 1.0
    return min(raw_variance / 1000.0, 1.0)


def _exposure(hsv: np.ndarray) -> tuple[float, str]:
    """
    Tent-function score based on HSV V-channel mean.
    Returns (score, retake_key) where retake_key is "exposure_dark" or "exposure_bright".

    spec: 6.2.2 lines 1104-1115
    """
    v_mean = float(hsv[:, :, 2].mean())  # spec: 6.2.2 line 1106 — V mean in [0, 255]

    if v_mean < 50:                          # spec: 6.2.2 line 1108
        return 0.0, "exposure_dark"
    elif v_mean > 220:                       # spec: 6.2.2 line 1110
        return 0.0, "exposure_bright"
    elif v_mean < 130:                       # spec: 6.2.2 line 1112
        return (v_mean - 50) / 80, "exposure_dark"
    else:                                    # spec: 6.2.2 line 1114
        return 1.0 - (v_mean - 130) / 90, "exposure_bright"


def _leaf_presence(hsv: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Fraction of green/yellow-green pixels (H in [25,95], S >= 40).
    Ramp from 5% to 30%.

    Returns (score, green_mask_bool).

    spec: 6.2.3 lines 1134-1148
    """
    # spec: 6.2.3 lines 1137-1140 — hue range [25, 95] AND saturation >= 40
    green_mask: np.ndarray = (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) &  # H
        (hsv[:, :, 1] >= 40)                             # S
    )
    pct_green = float(green_mask.mean())  # spec: 6.2.3 line 1141

    if pct_green < 0.05:                                  # spec: 6.2.3 line 1143
        return 0.0, green_mask
    elif pct_green > 0.30:                                # spec: 6.2.3 line 1145
        return 1.0, green_mask
    else:                                                  # spec: 6.2.3 line 1147
        return (pct_green - 0.05) / 0.25, green_mask


def _leaf_fill(green_mask: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """
    Largest connected component bounding box / image area.
    Ramp from 5% to 40%.

    spec: 6.2.4 lines 1166-1183
    """
    H, W = image_shape[:2]
    # spec: 6.2.4 line 1169 — connectedComponentsWithStats on uint8 mask
    nb, _, stats, _ = cv2.connectedComponentsWithStats(green_mask.astype(np.uint8))
    if nb <= 1:                               # spec: 6.2.4 line 1170
        return 0.0
    # spec: 6.2.4 line 1173 — largest component (ignore background label 0)
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_bbox_w = stats[largest_idx, cv2.CC_STAT_WIDTH]   # spec: 6.2.4 line 1174
    largest_bbox_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]  # spec: 6.2.4 line 1175
    fill = (largest_bbox_w * largest_bbox_h) / (W * H)       # spec: 6.2.4 line 1176

    if fill < 0.05:              # spec: 6.2.4 line 1178
        return 0.0
    elif fill > 0.40:            # spec: 6.2.4 line 1180
        return 1.0
    else:                        # spec: 6.2.4 line 1182
        return (fill - 0.05) / 0.35


def _background_contamination(green_mask: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """
    Number of significant (> 5% of image area) green connected components.
    1 → 1.0 (clean), 2 → 0.5 (two leaves), 3+ → 0.0 (cluttered).

    spec: 6.2.5 lines 1201-1215
    """
    H, W = image_shape[:2]
    nb, _, stats, _ = cv2.connectedComponentsWithStats(green_mask.astype(np.uint8))
    if nb <= 1:
        # spec: 6.2.5 line 1205 — no green at all; defer to leaf_presence check
        return 1.0
    sizes = stats[1:, cv2.CC_STAT_AREA]                      # spec: 6.2.5 line 1206
    image_area = H * W
    significant = sizes[sizes > image_area * 0.05]            # spec: 6.2.5 line 1208 — >5% of image
    n_significant = len(significant)

    if n_significant <= 1:    # spec: 6.2.5 line 1210
        return 1.0
    elif n_significant == 2:  # spec: 6.2.5 line 1212
        return 0.5
    else:                     # spec: 6.2.5 line 1214
        return 0.0


def _resolution(image_shape: tuple[int, ...]) -> float:
    """
    Smaller image dimension normalized between 224 (minimum) and 800 (full score).

    spec: 6.2.6 lines 1233-1241
    """
    H, W = image_shape[:2]
    smaller = min(W, H)            # spec: 6.2.6 line 1235
    if smaller < 224:              # spec: 6.2.6 line 1236 — defensive; validation already caught this
        return 0.0
    elif smaller >= 800:           # spec: 6.2.6 line 1238
        return 1.0
    else:                          # spec: 6.2.6 line 1240
        return (smaller - 224) / (800 - 224)


def _wetness(hsv: np.ndarray) -> float:
    """
    Fraction of specular highlight pixels (V > 220 AND S < 30).
    1.0 if fraction < 0.5%; 0.0 if fraction > 5%; linear ramp between.

    spec: 6.2.7 lines 1259-1269
    """
    # spec: 6.2.7 line 1261 — bright (V>220) AND desaturated (S<30)
    spec_mask: np.ndarray = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 30)
    pct_spec = float(spec_mask.mean())

    if pct_spec < 0.005:                            # spec: 6.2.7 line 1264
        return 1.0
    elif pct_spec > 0.05:                           # spec: 6.2.7 line 1266
        return 0.0
    else:                                           # spec: 6.2.7 line 1268
        return 1.0 - (pct_spec - 0.005) / 0.045


# ---------------------------------------------------------------------------
# Aggregation
# spec: 6.3 lines 1287-1292
# ---------------------------------------------------------------------------

def _aggregate_quality(
    scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Weighted geometric mean of per-dimension scores.
    Equal weights by default.

    spec: 6.3 lines 1287-1292
    """
    if weights is None:
        weights = {k: 1.0 for k in scores}  # spec: 6.3 line 1289 — equal weighting default
    total_weight = sum(weights.values())
    # spec: 6.3 line 1291 — weighted log-sum then exp
    log_sum = sum(weights[k] * math.log(max(scores[k], 1e-6)) for k in scores)
    return math.exp(log_sum / total_weight)


# ---------------------------------------------------------------------------
# Four-way decision
# spec: 6.4 lines 1318-1331
# ---------------------------------------------------------------------------

def _iqa_decide(
    aggregate: float,
    per_dim: dict[str, float],
    bad_thresholds: dict[str, float] | None = None,
) -> str:
    """
    REJECT / DEGRADED / ACCEPTABLE / HIGH decision.

    spec: 6.4 lines 1318-1331
    """
    thresholds = bad_thresholds if bad_thresholds is not None else BAD_THRESHOLDS

    # spec: 6.4 lines 1319-1322 — hard rejections from individual dimensions trump aggregate
    for dim_name, score in per_dim.items():
        if score < thresholds[dim_name]:
            return "REJECT"

    # spec: 6.4 lines 1324-1331 — decide by aggregate
    if aggregate < _AGG_REJECT:        # spec: 6.4 line 1324
        return "REJECT"
    elif aggregate < _AGG_DEGRADED:    # spec: 6.4 line 1326
        return "DEGRADED"
    elif aggregate < _AGG_ACCEPTABLE:  # spec: 6.4 line 1328
        return "ACCEPTABLE"
    else:
        return "HIGH"


# ---------------------------------------------------------------------------
# Main entry point
# spec: 6.6 line 1374 — compute_iqa(validated_image: ValidatedImage) -> IQAResult
# ---------------------------------------------------------------------------

def compute_iqa(validated_image: Any) -> IQAResult:
    """
    Compute all seven IQA dimensions and return an IQAResult.

    Parameters
    ----------
    validated_image:
        Any object with a ``pil_image`` attribute that is a PIL Image in RGB
        mode (or convertible to RGB). The real type is ``ValidatedImage`` from
        ``tomato_sandbox.input_validation`` (spec Section 5.2, lines 960-970).
        T-IMPL-2a runs in parallel, so the hard import is deferred (DEC-030).

    Returns
    -------
    IQAResult
        Per spec Section 6.5, lines 1357-1365.

    Notes
    -----
    - HSV conversion is computed ONCE and shared across dimensions
      (spec 6.7 performance budget: "HSV conversion computed once, reused").
    - green_mask from leaf_presence is reused in leaf_fill and
      background_contamination (spec 6.7: "Leaf presence + fill (mask +
      connected components): ~10 ms").
    - On any unexpected exception, a REJECT result is returned with an
      explanatory retake message and the error logged at ERROR level.

    spec: 6.6 line 1374
    """
    try:
        pil_image = validated_image.pil_image
        rgb: np.ndarray = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        _logger.error(
            "iqa_input_conversion_failed",
            succeeded=False,
            error=str(exc),
        )
        return IQAResult(
            decision="REJECT",
            aggregate_score=0.0,
            per_dimension={k: 0.0 for k in BAD_THRESHOLDS},
            failing_dimensions=list(BAD_THRESHOLDS.keys()),
            retake_message=(
                "Could not read the image. Please re-take and upload again."
            ),
            green_mask=None,
        )

    image_shape = rgb.shape  # (H, W, 3)

    # -- Convert to grayscale (for sharpness) and HSV (shared for all others) --
    # spec: 6.7 "HSV conversion (computed once, reused): ~5 ms"
    gray: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # ------------------------------------------------------------------ #
    # 1. Sharpness                                                         #
    # spec: 6.2.1 lines 1081-1086                                          #
    # ------------------------------------------------------------------ #
    sharpness_score = _sharpness(gray)

    # ------------------------------------------------------------------ #
    # 2. Exposure                                                          #
    # spec: 6.2.2 lines 1104-1115                                          #
    # ------------------------------------------------------------------ #
    exposure_score, exposure_retake_key = _exposure(hsv)

    # ------------------------------------------------------------------ #
    # 3. Leaf presence (also produces green_mask for reuse)                #
    # spec: 6.2.3 lines 1134-1148                                          #
    # ------------------------------------------------------------------ #
    leaf_presence_score, green_mask = _leaf_presence(hsv)

    # ------------------------------------------------------------------ #
    # 4. Leaf fill (reuses green_mask)                                     #
    # spec: 6.2.4 lines 1166-1183                                          #
    # ------------------------------------------------------------------ #
    leaf_fill_score = _leaf_fill(green_mask, image_shape)

    # ------------------------------------------------------------------ #
    # 5. Background contamination (reuses green_mask)                      #
    # spec: 6.2.5 lines 1201-1215                                          #
    # ------------------------------------------------------------------ #
    bg_contamination_score = _background_contamination(green_mask, image_shape)

    # ------------------------------------------------------------------ #
    # 6. Resolution                                                        #
    # spec: 6.2.6 lines 1233-1241                                          #
    # ------------------------------------------------------------------ #
    resolution_score = _resolution(image_shape)

    # ------------------------------------------------------------------ #
    # 7. Wetness                                                           #
    # spec: 6.2.7 lines 1259-1269                                          #
    # ------------------------------------------------------------------ #
    wetness_score = _wetness(hsv)

    # ------------------------------------------------------------------ #
    # Assemble per_dimension dict                                          #
    # ------------------------------------------------------------------ #
    per_dimension: dict[str, float] = {
        "sharpness": sharpness_score,
        "exposure": exposure_score,
        "leaf_presence": leaf_presence_score,
        "leaf_fill": leaf_fill_score,
        "background_contamination": bg_contamination_score,
        "resolution": resolution_score,
        "wetness": wetness_score,
    }

    # ------------------------------------------------------------------ #
    # Aggregate                                                            #
    # spec: 6.3 lines 1287-1292                                            #
    # ------------------------------------------------------------------ #
    aggregate_score = _aggregate_quality(per_dimension)

    # ------------------------------------------------------------------ #
    # Decision                                                             #
    # spec: 6.4 lines 1318-1331                                            #
    # ------------------------------------------------------------------ #
    decision = _iqa_decide(aggregate_score, per_dimension)

    # ------------------------------------------------------------------ #
    # Failing dimensions                                                   #
    # ------------------------------------------------------------------ #
    failing_dimensions = [
        dim for dim, score in per_dimension.items()
        if score < BAD_THRESHOLDS[dim]
    ]

    # ------------------------------------------------------------------ #
    # Retake message                                                       #
    # spec: 6.4 line 1335 — "the most-failing dimension's retake message;  #
    # if multiple dimensions fail, the worst is shown"                     #
    # "Worst" = lowest score among failing dimensions.                     #
    # ------------------------------------------------------------------ #
    retake_message: str | None = None
    green_mask_out: np.ndarray | None = green_mask

    if decision == "REJECT":
        if failing_dimensions:
            # Find the dimension with the lowest score
            worst_dim = min(failing_dimensions, key=lambda d: per_dimension[d])
            if worst_dim == "exposure":
                # spec: 6.2.2 — two separate messages for dark vs bright
                retake_message = _RETAKE_MESSAGES[exposure_retake_key]
            else:
                retake_message = _RETAKE_MESSAGES[worst_dim]
        else:
            # Aggregate-based reject (no single dim failed individually)
            retake_message = (
                "Image quality is too low overall. Please re-take with better"
                " lighting, focus, and leaf framing."
            )
        # spec: 6.5 line 1364 — green_mask passed to PSV even on REJECT
        # (PSV uses it as fallback; set to None on REJECT so PSV knows no mask available)
        green_mask_out = None

    _logger.info(
        "iqa_complete",
        succeeded=True,
        decision=decision,
        aggregate_score=round(aggregate_score, 4),
        failing_dimensions=failing_dimensions,
        sharpness=round(sharpness_score, 4),
        exposure=round(exposure_score, 4),
        leaf_presence=round(leaf_presence_score, 4),
        leaf_fill=round(leaf_fill_score, 4),
        background_contamination=round(bg_contamination_score, 4),
        resolution=round(resolution_score, 4),
        wetness=round(wetness_score, 4),
    )

    return IQAResult(
        decision=decision,
        aggregate_score=aggregate_score,
        per_dimension=per_dimension,
        failing_dimensions=failing_dimensions,
        retake_message=retake_message,
        green_mask=green_mask_out,
    )
