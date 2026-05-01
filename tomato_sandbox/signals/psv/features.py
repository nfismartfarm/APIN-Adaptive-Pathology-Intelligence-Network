"""
PSV Stage 3 — The 26 features.

Computes all 26 hand-engineered features organised into 8 groups (G1–G8).
Feature vector layout is FIXED at the indices in Section 10.5.9.

BLK-007 resolution: every feature has an inline spec traceability comment.

# spec: 10.5 lines 2307-2641
"""

from __future__ import annotations

import cv2
import numpy as np

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.nan_guards import guard_array

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature catalog — canonical index order per spec 10.5.9 lines 2609-2640
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "disease_coverage_pct",      # G1 idx 0   spec: 10.5.1 lines 2317-2321
    "largest_lesion_pct",        # G1 idx 1   spec: 10.5.1 lines 2323-2328
    "lesion_count",              # G1 idx 2   spec: 10.5.1 lines 2330-2334
    "mean_lesion_size",          # G2 idx 3   spec: 10.5.2 lines 2340-2345
    "lesion_size_std",           # G2 idx 4   spec: 10.5.2 lines 2347-2351
    "mean_lesion_circularity",   # G2 idx 5   spec: 10.5.2 lines 2353-2376
    "edge_sharpness",            # G2 idx 6   spec: 10.5.2 lines 2379-2397
    "yellow_pixel_fraction",     # G3 idx 7   spec: 10.5.3 lines 2404-2409
    "brown_pixel_fraction",      # G3 idx 8   spec: 10.5.3 lines 2412-2418
    "necrotic_pixel_fraction",   # G3 idx 9   spec: 10.5.3 lines 2420-2425
    "leaf_color_variance",       # G3 idx 10  spec: 10.5.3 lines 2427-2432
    "yellow_marginality_ratio",  # G4 idx 11  spec: 10.5.4 lines 2438-2453
    "disease_centroid_offset",   # G4 idx 12  spec: 10.5.4 lines 2457-2466
    "disease_spatial_dispersion",# G4 idx 13  spec: 10.5.4 lines 2469-2478
    "GLCM_contrast",             # G5 idx 14  spec: 10.5.5 lines 2485-2497
    "GLCM_homogeneity",          # G5 idx 15  spec: 10.5.5 lines 2500-2504
    "high_freq_energy_ratio",    # G5 idx 16  spec: 10.5.5 lines 2507-2522
    "leaf_compactness",          # G6 idx 17  spec: 10.5.6 lines 2529-2534
    "leaf_aspect_ratio",         # G6 idx 18  spec: 10.5.6 lines 2537-2543
    "sharpness",                 # G7 idx 19  spec: 10.5.7 lines 2550-2553
    "aggregate_quality",         # G7 idx 20  spec: 10.5.7 lines 2556-2560
    "psv_aggregate_reliability", # G7 idx 21  spec: 10.5.7 lines 2562-2567  (Stage 5 fills this)
    "ExG",                       # G8 idx 22  spec: 10.5.8 lines 2577-2581
    "GLI",                       # G8 idx 23  spec: 10.5.8 lines 2584-2589
    "MGRVI",                     # G8 idx 24  spec: 10.5.8 lines 2592-2596
    "VARI",                      # G8 idx 25  spec: 10.5.8 lines 2599-2603
]

assert len(FEATURE_NAMES) == 26, "FEATURE_NAMES must have exactly 26 entries"

NUM_FEATURES: int = 26


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safe division; returns *default* when denominator is zero."""
    return float(num / den) if den != 0.0 else default


# ---------------------------------------------------------------------------
# G1 — Coverage features (indices 0-2)
# spec: 10.5.1 lines 2313-2334
# ---------------------------------------------------------------------------

def _g1_disease_coverage_pct(lesion_stats: dict) -> float:
    """G1.1 disease_coverage_pct ∈ [0, 100].

    100 * disease_area_px / leaf_area_px

    # spec: 10.5.1 lines 2317-2321 — "percentage of the leaf area covered by disease pixels"
    """
    return _safe_div(
        100.0 * lesion_stats["disease_area_px"],
        lesion_stats["leaf_area_px"],
    )


def _g1_largest_lesion_pct(lesion_stats: dict) -> float:
    """G1.2 largest_lesion_pct ∈ [0, 100].

    100 * max lesion area / leaf_area_px

    # spec: 10.5.1 lines 2323-2328 — "size of the single biggest lesion as a fraction"
    """
    n = lesion_stats["n_lesions"]
    if n == 0:
        return 0.0
    stats = lesion_stats["stats"]  # [N+1, 5]; row 0 is background
    largest_area = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return _safe_div(100.0 * largest_area, lesion_stats["leaf_area_px"])


def _g1_lesion_count(lesion_stats: dict) -> float:
    """G1.3 lesion_count ∈ ℕ capped at 200.

    # spec: 10.5.1 lines 2330-2334 — "number of distinct disease regions; cap at 200"
    """
    return float(min(lesion_stats["n_lesions"], 200))


# ---------------------------------------------------------------------------
# G2 — Lesion shape features (indices 3-6)
# spec: 10.5.2 lines 2336-2398
# ---------------------------------------------------------------------------

def _g2_mean_lesion_size(lesion_stats: dict) -> float:
    """G2.1 mean_lesion_size (px²; clipped at 50000).

    # spec: 10.5.2 lines 2340-2345 — "average lesion size; clipped at 50000"
    """
    n = lesion_stats["n_lesions"]
    if n == 0:
        return 0.0
    stats = lesion_stats["stats"]
    mean_sz = float(np.mean(stats[1:, cv2.CC_STAT_AREA]))
    return float(np.clip(mean_sz, 0.0, 50_000.0))


def _g2_lesion_size_std(lesion_stats: dict) -> float:
    """G2.2 lesion_size_std (px²; clipped at 50000).

    # spec: 10.5.2 lines 2347-2351 — "variance in lesion sizes; clipped at 50000"
    """
    n = lesion_stats["n_lesions"]
    if n <= 1:
        return 0.0
    stats = lesion_stats["stats"]
    std_sz = float(np.std(stats[1:, cv2.CC_STAT_AREA]))
    return float(np.clip(std_sz, 0.0, 50_000.0))


def _g2_mean_lesion_circularity(disease_mask: np.ndarray, lesion_stats: dict) -> float:
    """G2.3 mean_lesion_circularity ∈ [0, 1].

    4π * area / perimeter² averaged over all lesions.

    # spec: 10.5.2 lines 2353-2376 — isoperimetric-inequality circularity formula
    """
    n = lesion_stats["n_lesions"]
    if n == 0:
        return 0.0
    labels = lesion_stats["labels"]
    stats = lesion_stats["stats"]
    circularities: list[float] = []

    for i in range(1, n + 1):
        component_i = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        # Take the longest-perimeter contour for this component
        # spec: 10.5.2 line 2367 — "Take the longest contour (largest perimeter)"
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter <= 0.0:
            continue
        area = float(stats[i, cv2.CC_STAT_AREA])
        circ = 4.0 * np.pi * area / (perimeter ** 2)
        circularities.append(min(circ, 1.0))  # isoperimetric bound

    return float(np.mean(circularities)) if circularities else 0.0


def _g2_edge_sharpness(rgb_cc: np.ndarray, disease_mask: np.ndarray) -> float:
    """G2.4 edge_sharpness ∈ [0, 1].

    Mean Sobel gradient at lesion edge pixels normalized by 255.

    # spec: 10.5.2 lines 2379-2397 — "Sobel on L channel of LAB at edge pixels"
    """
    if disease_mask.sum() == 0:
        return 0.0
    # Edge pixels = disease pixels minus its eroded version
    eroded = cv2.erode(disease_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    edge_mask = disease_mask & (~eroded.astype(bool))
    if edge_mask.sum() == 0:
        return 0.0
    lab = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.clip(grad_mag[edge_mask].mean() / 255.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# G3 — Color statistics features (indices 7-10)
# spec: 10.5.3 lines 2400-2432
# ---------------------------------------------------------------------------

def _g3_color_features(rgb_cc: np.ndarray, leaf_mask: np.ndarray) -> tuple[float, float, float, float]:
    """G3 features 7-10: yellow_fraction, brown_fraction, necrotic_fraction, color_variance.

    Returns (yellow_pixel_fraction, brown_pixel_fraction, necrotic_pixel_fraction,
             leaf_color_variance).

    # spec: 10.5.3 lines 2404-2432
    """
    hsv = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2HSV)
    leaf_area = int(leaf_mask.sum())

    if leaf_area == 0:
        return 0.0, 0.0, 0.0, 0.0

    leaf_hsv = hsv[leaf_mask]  # [N, 3]

    # G3.1 yellow_pixel_fraction — hue [15, 35], saturation >= 50
    # spec: 10.5.3 lines 2404-2409
    yellow_mask = (
        (leaf_hsv[:, 0] >= 15) & (leaf_hsv[:, 0] <= 35) & (leaf_hsv[:, 1] >= 50)
    )
    yellow_frac = float(yellow_mask.sum()) / leaf_area

    # G3.2 brown_pixel_fraction — hue [5, 20], saturation >= 50, value < 150
    # spec: 10.5.3 lines 2412-2418
    brown_mask = (
        (leaf_hsv[:, 0] >= 5)
        & (leaf_hsv[:, 0] <= 20)
        & (leaf_hsv[:, 1] >= 50)
        & (leaf_hsv[:, 2] < 150)
    )
    brown_frac = float(brown_mask.sum()) / leaf_area

    # G3.3 necrotic_pixel_fraction — value < 50, saturation < 60
    # spec: 10.5.3 lines 2420-2425
    necrotic_mask = (leaf_hsv[:, 2] < 50) & (leaf_hsv[:, 1] < 60)
    necrotic_frac = float(necrotic_mask.sum()) / leaf_area

    # G3.4 leaf_color_variance — variance of L channel in LAB, range [0, 6500]
    # spec: 10.5.3 lines 2427-2432
    lab = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2LAB)
    lab_leaf = lab[leaf_mask]
    color_var = float(np.var(lab_leaf[:, 0]))

    return yellow_frac, brown_frac, necrotic_frac, color_var


# ---------------------------------------------------------------------------
# G4 — Spatial pattern features (indices 11-13)
# spec: 10.5.4 lines 2434-2479
# ---------------------------------------------------------------------------

def _g4_yellow_marginality_ratio(
    rgb_cc: np.ndarray,
    leaf_mask: np.ndarray,
) -> float:
    """G4.1 yellow_marginality_ratio ∈ [0, 1].

    Fraction of yellow pixels that sit near the leaf margin.

    # spec: 10.5.4 lines 2438-2453 — "margin = leaf pixels within 15% of longer bbox dim"
    """
    if leaf_mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2HSV)
    leaf_hsv = hsv[leaf_mask]
    # Recompute yellow mask on full image, masked to leaf
    hsv_full = hsv
    yellow_full = (
        (hsv_full[:, :, 0] >= 15)
        & (hsv_full[:, :, 0] <= 35)
        & (hsv_full[:, :, 1] >= 50)
        & leaf_mask
    )
    yellow_total = int(yellow_full.sum())
    if yellow_total == 0:
        return 0.0

    # Margin = pixels with distance-to-boundary < 15% of longer bbox dimension
    x_bb, y_bb, bbox_w, bbox_h, _ = cv2.boundingRect(leaf_mask.astype(np.uint8))
    margin_dist_threshold = 0.15 * max(bbox_w, bbox_h)

    leaf_uint8 = leaf_mask.astype(np.uint8)
    dist = cv2.distanceTransform(leaf_uint8, cv2.DIST_L2, 3)
    margin_mask = (dist > 0) & (dist < margin_dist_threshold)

    yellow_in_margin = int((yellow_full & margin_mask).sum())
    return float(yellow_in_margin) / max(yellow_total, 1)


def _g4_disease_centroid_offset(leaf_mask: np.ndarray, disease_mask: np.ndarray) -> float:
    """G4.2 disease_centroid_offset ∈ [0, 1].

    Distance from disease centroid to leaf centroid, normalized by leaf radius.

    # spec: 10.5.4 lines 2457-2466 — centroid offset normalized by equivalent radius
    """
    if disease_mask.sum() == 0:
        return 0.0
    leaf_area = float(leaf_mask.sum())
    if leaf_area == 0:
        return 0.0

    # Centroids computed as mean of pixel coordinates
    leaf_yx = np.argwhere(leaf_mask)
    disease_yx = np.argwhere(disease_mask)
    leaf_centroid = leaf_yx.mean(axis=0)     # [y, x]
    disease_centroid = disease_yx.mean(axis=0)  # [y, x]

    offset = float(np.linalg.norm(disease_centroid - leaf_centroid))
    leaf_radius = float(np.sqrt(leaf_area / np.pi))  # equivalent circle radius
    return float(np.clip(offset / max(leaf_radius, 1.0), 0.0, 1.0))


def _g4_disease_spatial_dispersion(leaf_mask: np.ndarray, lesion_stats: dict) -> float:
    """G4.3 disease_spatial_dispersion ∈ [0, 1].

    Mean pairwise distance between lesion centroids, normalized by leaf diagonal.

    # spec: 10.5.4 lines 2469-2478 — "pdist on centroids / leaf diagonal"
    """
    n = lesion_stats["n_lesions"]
    if n <= 1:
        return 0.0

    centroids = lesion_stats["centroids"][1:]  # skip background centroid; shape [n, 2]

    # Pairwise distances
    try:
        from scipy.spatial.distance import pdist
        pairwise = pdist(centroids)
        mean_dist = float(np.mean(pairwise)) if len(pairwise) > 0 else 0.0
    except ImportError:
        # Fallback: manual pairwise (slower)
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                d = float(np.linalg.norm(centroids[i] - centroids[j]))
                dists.append(d)
        mean_dist = float(np.mean(dists)) if dists else 0.0

    # Leaf bounding box diagonal
    x_bb, y_bb, bbox_w, bbox_h = cv2.boundingRect(leaf_mask.astype(np.uint8))[:4]
    leaf_diagonal = float(np.sqrt(bbox_w ** 2 + bbox_h ** 2))
    return float(np.clip(_safe_div(mean_dist, leaf_diagonal), 0.0, 1.0))


# ---------------------------------------------------------------------------
# G5 — Texture features (indices 14-16)
# spec: 10.5.5 lines 2481-2522
# ---------------------------------------------------------------------------

def _g5_texture_features(rgb_cc: np.ndarray, leaf_mask: np.ndarray) -> tuple[float, float, float]:
    """G5 features 14-16: GLCM_contrast, GLCM_homogeneity, high_freq_energy_ratio.

    # spec: 10.5.5 lines 2481-2522
    """
    lab = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2LAB)
    gray_leaf = lab[:, :, 0].copy()
    gray_leaf[~leaf_mask] = 0  # zero out non-leaf — see Known bias note, spec 10.5.5 line 2498

    # G5.1/G5.2 — GLCM contrast and homogeneity
    # spec: 10.5.5 lines 2485-2503
    # graycomatrix from scikit-image
    try:
        from skimage.feature import graycomatrix, graycoprops  # type: ignore[import]

        # Reduce to 32 levels by scaling [0,255] → [0,31]
        # spec: 10.5.5 line 2490 — "levels=32"
        gray_32 = (gray_leaf.astype(np.float32) / 255.0 * 31).astype(np.uint8)

        glcm = graycomatrix(
            gray_32,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=32,
            symmetric=True,
            normed=True,
        )  # [32, 32, 1, 4]

        contrast_raw = float(graycoprops(glcm, "contrast").mean())
        # spec: 10.5.5 line 2494 — "normalize: 100 saturates"
        glcm_contrast = float(np.clip(contrast_raw / 100.0, 0.0, 1.0))

        # spec: 10.5.5 lines 2500-2503 — "homogeneity_raw already in [0, 1]"
        glcm_homogeneity = float(np.clip(graycoprops(glcm, "homogeneity").mean(), 0.0, 1.0))

    except ImportError:
        # scikit-image not installed — produce neutral values
        _log.warning("skimage not available; GLCM features set to 0.5 (neutral)")
        glcm_contrast = 0.5
        glcm_homogeneity = 0.5

    # G5.3 — High-frequency energy ratio via FFT
    # spec: 10.5.5 lines 2507-2522
    gray = lab[:, :, 0]
    gray_leaf_only = np.where(leaf_mask, gray, 0).astype(np.float64)
    fft_mag = np.abs(np.fft.fft2(gray_leaf_only))
    total_energy = float((fft_mag ** 2).sum())

    H, W = gray.shape
    center_y, center_x = H // 2, W // 2
    y_grid, x_grid = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
    # Low frequency = central 10% radius
    low_freq_mask = dist_from_center < min(H, W) * 0.1

    fft_mag_shifted = np.fft.fftshift(fft_mag)
    high_freq_energy = float((fft_mag_shifted ** 2)[~low_freq_mask].sum())
    high_freq_energy_ratio = float(
        np.clip(_safe_div(high_freq_energy, max(total_energy, 1.0)), 0.0, 1.0)
    )

    return glcm_contrast, glcm_homogeneity, high_freq_energy_ratio


# ---------------------------------------------------------------------------
# G6 — Leaf geometry features (indices 17-18)
# spec: 10.5.6 lines 2524-2544
# ---------------------------------------------------------------------------

def _g6_geometry_features(leaf_mask: np.ndarray) -> tuple[float, float]:
    """G6 features 17-18: leaf_compactness, leaf_aspect_ratio.

    # spec: 10.5.6 lines 2524-2544
    """
    leaf_area = int(leaf_mask.sum())
    if leaf_area == 0:
        return 0.0, 1.0

    contours, _ = cv2.findContours(
        leaf_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0, 1.0
    contour = max(contours, key=cv2.contourArea)

    # G6.1 leaf_compactness ∈ [0, 1]
    # spec: 10.5.6 lines 2529-2534 — "4π*area/perimeter² clipped to [0, 1]"
    leaf_perimeter = cv2.arcLength(contour, closed=True)
    if leaf_perimeter > 0:
        compactness = float(
            np.clip(4.0 * np.pi * leaf_area / max(leaf_perimeter ** 2, 1.0), 0.0, 1.0)
        )
    else:
        compactness = 0.0

    # G6.2 leaf_aspect_ratio ∈ [0.1, 10] — clipped
    # spec: 10.5.6 lines 2537-2543 — "minAreaRect; long axis / short axis"
    rect = cv2.minAreaRect(contour)  # (center, (w, h), angle)
    w, h = rect[1]
    long_side = float(max(w, h))
    short_side = float(max(min(w, h), 1.0))
    aspect_ratio = float(np.clip(long_side / short_side, 0.1, 10.0))

    return compactness, aspect_ratio


# ---------------------------------------------------------------------------
# G7 — Quality and reliability features (indices 19-21)
# spec: 10.5.7 lines 2546-2567
# G7.1 sharpness and G7.2 aggregate_quality are passed in; G7.3 is Stage 5.
# ---------------------------------------------------------------------------

def _g7_sharpness(rgb_cc: np.ndarray) -> float:
    """G7.1 sharpness ∈ [0, 1] — same formula as IQA sharpness, recomputed.

    Laplacian variance on L channel, normalized to [0, 1].

    # spec: 10.5.7 lines 2550-2553 — "recomputed on the color-constancy-applied image"
    # spec: 6.2.1 — IQA sharpness definition (Laplacian variance)
    """
    lab = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    lap_var = float(cv2.Laplacian(l_channel, cv2.CV_64F).var())
    # Normalize: 500 saturates (empirical cap from IQA sharpness definition)
    return float(np.clip(lap_var / 500.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# G8 — Vegetation indices (indices 22-25)
# spec: 10.5.8 lines 2568-2604
# ---------------------------------------------------------------------------

def _g8_vegetation_indices(rgb_cc: np.ndarray, leaf_mask: np.ndarray) -> tuple[float, float, float, float]:
    """G8 features 22-25: ExG, GLI, MGRVI, VARI.

    Computed on leaf pixels only in sRGB space.

    # spec: 10.5.8 lines 2568-2604 — "sRGB used directly; linearization deferred to F.0"
    """
    if leaf_mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0

    leaf_rgb = rgb_cc[leaf_mask].astype(np.float32)  # [N, 3]
    r = leaf_rgb[:, 0]
    g = leaf_rgb[:, 1]
    b = leaf_rgb[:, 2]

    # G8.1 ExG (Excess Green; standardized to [-1, 1])
    # spec: 10.5.8 lines 2577-2581 — "2g - r - b normalized by 255"
    exg_per_pixel = 2.0 * g - r - b  # in [-510, 510]
    exg = float(np.clip(np.mean(exg_per_pixel) / 255.0, -1.0, 1.0))

    # G8.2 GLI (Green Leaf Index; in [-1, 1])
    # spec: 10.5.8 lines 2584-2589 — "(2g - r - b) / (2g + r + b)"
    denom_gli = 2.0 * g + r + b
    gli_per_pixel = np.where(
        denom_gli > 0,
        (2.0 * g - r - b) / np.maximum(denom_gli, 1.0),
        0.0,
    )
    gli = float(np.mean(gli_per_pixel))

    # G8.3 MGRVI (Modified Green-Red Vegetation Index; in [-1, 1])
    # spec: 10.5.8 lines 2592-2596 — "(g²-r²)/(g²+r²)"
    denom_mgrvi = g ** 2 + r ** 2
    mgrvi_per_pixel = np.where(
        denom_mgrvi > 0,
        (g ** 2 - r ** 2) / np.maximum(denom_mgrvi, 1.0),
        0.0,
    )
    mgrvi = float(np.mean(mgrvi_per_pixel))

    # G8.4 VARI (Visible Atmospherically Resistant Index; clipped to [-1, 1])
    # spec: 10.5.8 lines 2599-2603 — "(g-r)/(g+r-b)"
    denom_vari = g + r - b
    vari_per_pixel = np.where(
        denom_vari != 0,
        (g - r) / np.where(denom_vari != 0, denom_vari, 1.0),
        0.0,
    )
    vari = float(np.clip(np.mean(vari_per_pixel), -1.0, 1.0))

    return exg, gli, mgrvi, vari


# ---------------------------------------------------------------------------
# Public orchestrator for Stage 3
# spec: 10.2 lines 2113-2116 — "compute_26_features(rgb_cc, leaf_mask, disease_mask, ...)"
# ---------------------------------------------------------------------------

def compute_26_features(
    rgb_cc: np.ndarray,
    leaf_mask: np.ndarray,
    disease_mask: np.ndarray,
    lesion_stats: dict,
    iqa_aggregate_score: float,
) -> np.ndarray:
    """Compute all 26 PSV features in the canonical index order.

    Args:
        rgb_cc: ``[H, W, 3]`` uint8 color-constancy RGB image.
        leaf_mask: ``[H, W]`` bool leaf mask from Stage 1.
        disease_mask: ``[H, W]`` bool disease mask from Stage 2.
        lesion_stats: Dict from Stage 2 with CC info.
        iqa_aggregate_score: IQA aggregate score ∈ [0, 1] (passes through as G7.2).

    Returns:
        ``raw_features``: float32 array of shape ``[26]``. Feature at index 21
        (``psv_aggregate_reliability``) is a placeholder **zero** — Stage 5 fills it.

    # spec: 10.5 lines 2307-2641
    # spec: 10.2 lines 2113-2117 — "raw_features[21] is currently 0 (placeholder)"
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # G1 — Coverage (indices 0-2)
    # spec: 10.5.1 lines 2313-2334
    features[0] = _g1_disease_coverage_pct(lesion_stats)
    features[1] = _g1_largest_lesion_pct(lesion_stats)
    features[2] = _g1_lesion_count(lesion_stats)

    # G2 — Lesion shape (indices 3-6)
    # spec: 10.5.2 lines 2336-2398
    features[3] = _g2_mean_lesion_size(lesion_stats)
    features[4] = _g2_lesion_size_std(lesion_stats)
    features[5] = _g2_mean_lesion_circularity(disease_mask, lesion_stats)
    features[6] = _g2_edge_sharpness(rgb_cc, disease_mask)

    # G3 — Color statistics (indices 7-10)
    # spec: 10.5.3 lines 2400-2432
    yf, bf, nf, cv = _g3_color_features(rgb_cc, leaf_mask)
    features[7] = yf
    features[8] = bf
    features[9] = nf
    features[10] = cv

    # G4 — Spatial pattern (indices 11-13)
    # spec: 10.5.4 lines 2434-2479
    features[11] = _g4_yellow_marginality_ratio(rgb_cc, leaf_mask)
    features[12] = _g4_disease_centroid_offset(leaf_mask, disease_mask)
    features[13] = _g4_disease_spatial_dispersion(leaf_mask, lesion_stats)

    # G5 — Texture (indices 14-16)
    # spec: 10.5.5 lines 2481-2522
    gc, gh, hfe = _g5_texture_features(rgb_cc, leaf_mask)
    features[14] = gc
    features[15] = gh
    features[16] = hfe

    # G6 — Geometry (indices 17-18)
    # spec: 10.5.6 lines 2524-2544
    comp, asp = _g6_geometry_features(leaf_mask)
    features[17] = comp
    features[18] = asp

    # G7 — Quality/reliability (indices 19-21)
    # spec: 10.5.7 lines 2546-2567
    features[19] = _g7_sharpness(rgb_cc)
    features[20] = float(iqa_aggregate_score)  # aggregate_quality — passed through
    features[21] = 0.0  # psv_aggregate_reliability — placeholder; Stage 5 fills it

    # G8 — Vegetation indices (indices 22-25)
    # spec: 10.5.8 lines 2568-2604
    exg, gli, mgrvi, vari = _g8_vegetation_indices(rgb_cc, leaf_mask)
    features[22] = exg
    features[23] = gli
    features[24] = mgrvi
    features[25] = vari

    # Guard: replace any NaN/Inf with 0.0 (should not occur for uint8 inputs)
    features = guard_array(features, NUM_FEATURES, default_value=0.0)

    assert len(features) == 26, f"Feature vector must be length 26; got {len(features)}"
    return features
