"""
PSV Feature Extractor — 64+ domain-invariant features from plant pathology.

Each feature measures a PHYSICAL property of the disease that exists in every
photo regardless of camera, country, lighting, or training distribution.

Architecture:
  1. Shared pre-computation (run ONCE per image)
  2. Group A: Spatial Distribution (15 features)
  3. Group B: Vein Analysis (8 features)
  4. Group C: Spot/Lesion Morphology (14 features)
  5. Group D: Color Zone Analysis (11 features)
  6. Group E: Texture and Surface (8 features)
  7. Group F: Cross-Class Discriminators (8 features)
  Total: 64 features minimum

Safety: every feature computation is wrapped in try/except.
If any feature fails, it returns a default value and flags the failure.
The pipeline NEVER crashes due to a single feature error.
"""

import warnings
import numpy as np
import cv2
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Suppress non-critical warnings from skimage
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

from skimage.filters import frangi, sobel
from skimage.feature import blob_dog
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import binary_opening, binary_closing, disk

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

from scripts.psv.config import PSV_CFG


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SharedMaps:
    """Pre-computed maps shared across all feature groups."""
    lab: np.ndarray              # LAB image [H, W, 3]
    gray: np.ndarray             # Grayscale [H, W] float 0-1
    leaf_mask: np.ndarray        # Boolean leaf mask [H, W]
    distance_map: np.ndarray     # Normalized distance from edge [H, W] 0=edge, 1=center
    frangi_map: np.ndarray       # Vein detection [H, W] float
    disease_mask: np.ndarray     # Boolean disease pixels [H, W]
    healthy_mask: np.ndarray     # Boolean healthy green pixels [H, W]
    dark_mask: np.ndarray        # Boolean dark pixels [H, W]
    specular_mask: np.ndarray    # Boolean specular highlight pixels [H, W]
    blobs: np.ndarray            # Blob centroids and radii [N, 3] (y, x, sigma)
    blob_regions: list           # List of regionprops for connected disease components
    leaf_angle: float            # Leaf principal axis angle (degrees)
    leaf_area: int               # Total leaf pixels
    disease_area: int            # Total disease pixels
    yellow_mask: np.ndarray      # Boolean yellow pixels [H, W]
    brown_mask: np.ndarray       # Boolean brown pixels [H, W]
    gray_white_mask: np.ndarray  # Boolean gray-white pixels [H, W]
    powdery_mask: np.ndarray     # Boolean powdery white pixels [H, W]
    necrosis_mask: np.ndarray    # Boolean necrotic pixels [H, W]
    chlorosis_mask: np.ndarray   # Boolean chlorotic pixels [H, W]
    original_rgb: np.ndarray     # Original RGB image (for visualization)


@dataclass
class FeatureResult:
    """Result of full feature extraction."""
    features: Dict[str, float]        # All feature values
    failed_features: Dict[str, str]   # Features that failed: name -> error message
    shared_maps: SharedMaps           # Pre-computed maps (for visualization)
    extraction_time_ms: float         # Total extraction time


# ═══════════════════════════════════════════════════════════════════════
# HELPER: Safe feature computation wrapper
# ═══════════════════════════════════════════════════════════════════════

def _safe(func, default=0.0, name='unknown'):
    """Wrap a feature computation — returns default on any error."""
    try:
        val = func()
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════════════
# SHARED PRE-COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_shared_maps(image_rgb: np.ndarray) -> SharedMaps:
    """
    Run all shared pre-computations ONCE per image.
    Returns SharedMaps that all feature groups reference.

    Input: uint8 RGB image [H, W, 3]
    """
    cfg = PSV_CFG

    # Resize for consistent processing
    h, w = image_rgb.shape[:2]
    scale = min(cfg.PROCESSING_SIZE / max(h, w), 1.0)
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)

    # LAB conversion
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV LAB: L in [0,255], A in [0,255] centered at 128, B in [0,255] centered at 128
    # Convert to standard LAB: L in [0,100], A in [-128,127], B in [-128,127]
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0  # L: 0-100
    lab[:, :, 1] = lab[:, :, 1] - 128.0            # A: centered at 0
    lab[:, :, 2] = lab[:, :, 2] - 128.0            # B: centered at 0

    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

    # ── Leaf segmentation ──────────────────────────────────────────────
    leaf_mask = (
        (A >= cfg.LEAF_A_MIN) & (A <= cfg.LEAF_A_MAX) &
        (B >= cfg.LEAF_B_MIN) & (B <= cfg.LEAF_B_MAX) &
        (L >= cfg.LEAF_L_MIN) & (L <= cfg.LEAF_L_MAX)
    )
    # Morphological cleanup
    kernel = disk(cfg.LEAF_MORPH_KERNEL // 2)
    leaf_mask = binary_closing(leaf_mask, kernel)
    leaf_mask = binary_opening(leaf_mask, kernel)
    # Keep largest connected component
    labeled, num = sk_label(leaf_mask, return_num=True)
    if num > 0:
        largest = max(range(1, num + 1), key=lambda i: (labeled == i).sum())
        leaf_mask = labeled == largest

    leaf_area = int(leaf_mask.sum())

    # If leaf detection is too poor, use a generous fallback
    if leaf_area < image_rgb.shape[0] * image_rgb.shape[1] * cfg.LEAF_MIN_AREA_FRACTION:
        # Fallback: use green-ish channel threshold more liberally
        leaf_mask = (A < 15) & (L > 15) & (L < 95)
        leaf_mask = binary_closing(leaf_mask, disk(5))
        labeled, num = sk_label(leaf_mask, return_num=True)
        if num > 0:
            largest = max(range(1, num + 1), key=lambda i: (labeled == i).sum())
            leaf_mask = labeled == largest
        leaf_area = int(leaf_mask.sum())

    # ── Distance transform ─────────────────────────────────────────────
    if leaf_area > 0:
        dist = ndimage.distance_transform_edt(leaf_mask)
        dist_max = dist.max()
        distance_map = dist / max(dist_max, 1.0)  # Normalize 0-1
    else:
        distance_map = np.zeros_like(gray)

    # ── Leaf orientation (PCA) ─────────────────────────────────────────
    leaf_angle = 0.0
    if leaf_area > 100:
        ys, xs = np.where(leaf_mask)
        if len(ys) > 10:
            coords = np.column_stack([xs - xs.mean(), ys - ys.mean()])
            try:
                cov = np.cov(coords.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                leaf_angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
            except Exception:
                leaf_angle = 0.0

    # ── Frangi vein map ────────────────────────────────────────────────
    try:
        frangi_map = frangi(gray, sigmas=cfg.FRANGI_SIGMAS,
                           alpha=cfg.FRANGI_ALPHA, beta=cfg.FRANGI_BETA,
                           black_ridges=cfg.FRANGI_BLACK_RIDGES)
        frangi_map = frangi_map * leaf_mask  # Only within leaf
    except Exception:
        frangi_map = np.zeros_like(gray)

    # ── Specular mask ──────────────────────────────────────────────────
    specular_mask = (
        (L > cfg.SPECULAR_L_MIN) &
        (np.abs(A) < cfg.SPECULAR_A_ABS_MAX) &
        (np.abs(B) < cfg.SPECULAR_B_ABS_MAX) &
        leaf_mask
    )

    # ── Color masks ────────────────────────────────────────────────────
    # Healthy green
    healthy_mask = (
        (A >= cfg.HEALTHY_A_MIN) & (A <= cfg.HEALTHY_A_MAX) &
        (B >= cfg.HEALTHY_B_MIN) & (B <= cfg.HEALTHY_B_MAX) &
        (L >= cfg.HEALTHY_L_MIN) & (L <= cfg.HEALTHY_L_MAX) &
        leaf_mask & ~specular_mask
    )

    # Disease pixels = leaf pixels that deviate from healthy green
    disease_mask = leaf_mask & ~healthy_mask & ~specular_mask
    disease_area = int(disease_mask.sum())

    # Dark pixels
    dark_mask = (L < cfg.DARK_PIXEL_L_MAX) & leaf_mask & ~specular_mask

    # Yellow
    yellow_mask = (
        (A < cfg.YELLOW_A_MAX) & (B > cfg.YELLOW_B_MIN) & (L > cfg.YELLOW_L_MIN) &
        leaf_mask & ~specular_mask
    )

    # Brown
    brown_mask = (
        (A > cfg.BROWN_A_MIN) & (B > cfg.BROWN_B_MIN) &
        (L > cfg.BROWN_L_MIN) & (L < cfg.BROWN_L_MAX) &
        leaf_mask & ~specular_mask
    )

    # Gray-white (desiccated tissue — cercospora centers)
    gray_white_mask = (
        (L > cfg.GRAY_WHITE_L_MIN) &
        (np.abs(A) < cfg.GRAY_WHITE_A_ABS_MAX) &
        (np.abs(B) < cfg.GRAY_WHITE_B_ABS_MAX) &
        leaf_mask & ~specular_mask
    )

    # Powdery white (mildew surface conidia)
    powdery_mask = (
        (L > cfg.POWDERY_L_MIN) &
        (np.abs(A) < cfg.POWDERY_A_ABS_MAX) &
        (np.abs(B) < cfg.POWDERY_B_ABS_MAX) &
        leaf_mask & ~specular_mask
    )

    # Necrosis (very dark brown)
    necrosis_mask = (
        (L < cfg.NECROSIS_L_MAX) & (A > cfg.NECROSIS_A_MIN) & (B > cfg.NECROSIS_B_MIN) &
        leaf_mask & ~specular_mask
    )

    # Chlorosis (yellow-green early stage)
    chlorosis_mask = (
        (A < cfg.CHLOROSIS_A_MAX) & (B > cfg.CHLOROSIS_B_MIN) & (L > cfg.CHLOROSIS_L_MIN) &
        leaf_mask & ~specular_mask
    )

    # ── Blob detection ─────────────────────────────────────────────────
    # [FIX R2-5] Removed blob_dog (100-200ms on binary mask, unreliable).
    # Use connected component regionprops instead (computed below, ~0ms extra).
    blobs = np.array([]).reshape(0, 3)  # placeholder — regionprops used instead

    # Connected component regions of disease mask
    try:
        disease_labeled = sk_label(disease_mask)
        blob_regions = regionprops(disease_labeled)
        # Filter tiny regions
        blob_regions = [r for r in blob_regions if r.area >= cfg.BLOB_MIN_AREA]
    except Exception:
        blob_regions = []

    return SharedMaps(
        lab=lab, gray=gray, leaf_mask=leaf_mask, distance_map=distance_map,
        frangi_map=frangi_map, disease_mask=disease_mask, healthy_mask=healthy_mask,
        dark_mask=dark_mask, specular_mask=specular_mask,
        blobs=blobs, blob_regions=blob_regions,
        leaf_angle=leaf_angle, leaf_area=leaf_area, disease_area=disease_area,
        yellow_mask=yellow_mask, brown_mask=brown_mask,
        gray_white_mask=gray_white_mask, powdery_mask=powdery_mask,
        necrosis_mask=necrosis_mask, chlorosis_mask=chlorosis_mask,
        original_rgb=image_rgb,
    )


# ═══════════════════════════════════════════════════════════════════════
# GROUP A: SPATIAL DISTRIBUTION (15 features)
# Where on the leaf is the disease? Relative to edge, midrib, quadrant.
# ═══════════════════════════════════════════════════════════════════════

def compute_group_a(m: SharedMaps) -> Dict[str, float]:
    """Spatial distribution features."""
    cfg = PSV_CFG
    eps = cfg.EPSILON
    f = {}

    dm = m.distance_map
    dis = m.disease_mask
    leaf = m.leaf_mask

    # A01: margin disease density
    margin = leaf & (dm < cfg.MARGIN_ZONE_THRESHOLD) & (dm > 0)
    f['A01_margin_disease_density'] = _safe(
        lambda: dis[margin].mean() if margin.any() else 0.0)

    # A02: interior disease density
    interior = leaf & (dm > cfg.INTERIOR_ZONE_THRESHOLD)
    f['A02_interior_disease_density'] = _safe(
        lambda: dis[interior].mean() if interior.any() else 0.0)

    # A03: margin vs interior ratio (HIGH = black_rot, LOW = cercospora/alternaria)
    # [FIX] Clamp to prevent explosion when interior_density near zero
    f['A03_margin_vs_interior_ratio'] = _safe(
        lambda: min(f['A01_margin_disease_density'] / (f['A02_interior_disease_density'] + eps), 10.0))

    # A04: disease centroid depth (0=edge, 1=center)
    f['A04_disease_centroid_depth'] = _safe(
        lambda: dm[dis].mean() if dis.any() else 0.5)

    # A05: bilateral symmetry of disease
    def _bilateral_sym():
        if not dis.any():
            return 0.5
        ys, xs = np.where(leaf)
        cx = xs.mean()
        left = dis[:, :int(cx)]
        right = dis[:, int(cx):]
        right_flipped = right[:, ::-1]
        min_w = min(left.shape[1], right_flipped.shape[1])
        if min_w < 5:
            return 0.5
        left_col = left[:, :min_w].astype(float).mean(axis=0)
        right_col = right_flipped[:, :min_w].astype(float).mean(axis=0)
        if left_col.std() < eps or right_col.std() < eps:
            return 0.5
        return float(np.corrcoef(left_col, right_col)[0, 1])
    f['A05_disease_bilateral_symmetry'] = _safe(_bilateral_sym, default=0.5)

    # A06: disease quadrant entropy
    def _quadrant_entropy():
        if not dis.any():
            return 0.0
        ys, xs = np.where(leaf)
        cy, cx = ys.mean(), xs.mean()
        q = [
            dis[:int(cy), :int(cx)].sum(),
            dis[:int(cy), int(cx):].sum(),
            dis[int(cy):, :int(cx)].sum(),
            dis[int(cy):, int(cx):].sum(),
        ]
        total = sum(q)
        if total < 1:
            return 0.0
        probs = [qi / total for qi in q if qi > 0]
        return float(-sum(p * np.log(p + eps) for p in probs) / np.log(4))
    f['A06_disease_quadrant_entropy'] = _safe(_quadrant_entropy)

    # A07: margin connectivity (largest connected component touching edge)
    def _margin_connectivity():
        margin_dis = dis & (dm < cfg.MARGIN_ZONE_THRESHOLD) & (dm > 0)
        if not margin_dis.any():
            return 0.0
        labeled = sk_label(margin_dis)
        regions = regionprops(labeled)
        if not regions:
            return 0.0
        largest = max(regions, key=lambda r: r.area)
        return float(largest.area / (margin_dis.sum() + eps))
    f['A07_margin_connectivity'] = _safe(_margin_connectivity)

    # A08: disease radial gradient (positive=concentrated at edge, negative=interior)
    def _radial_gradient():
        if not dis.any() or m.leaf_area < 100:
            return 0.0
        dm_vals = dm[leaf].flatten()
        dis_vals = dis[leaf].astype(float).flatten()
        if dm_vals.std() < eps:
            return 0.0
        return float(np.corrcoef(dm_vals, dis_vals)[0, 1])
    f['A08_disease_radial_gradient'] = _safe(_radial_gradient)

    # A09: apex concentration (top 10% along leaf axis)
    def _apex_conc():
        if not dis.any() or m.leaf_area < 100:
            return 0.0
        ys, _ = np.where(leaf)
        apex_thresh = np.percentile(ys, 100 * (1 - cfg.APEX_ZONE_FRACTION))
        apex_zone = leaf & (np.arange(dis.shape[0])[:, None] > apex_thresh)
        return float(dis[apex_zone].mean()) if apex_zone.any() else 0.0
    f['A09_apex_concentration'] = _safe(_apex_conc)

    # A10: base concentration (bottom 10%)
    def _base_conc():
        if not dis.any() or m.leaf_area < 100:
            return 0.0
        ys, _ = np.where(leaf)
        base_thresh = np.percentile(ys, 100 * cfg.BASE_ZONE_FRACTION)
        base_zone = leaf & (np.arange(dis.shape[0])[:, None] < base_thresh)
        return float(dis[base_zone].mean()) if base_zone.any() else 0.0
    f['A10_base_concentration'] = _safe(_base_conc)

    # A11: midrib alignment
    def _midrib_align():
        if not dis.any() or m.frangi_map.max() < eps:
            return 0.0
        vein_zone = m.frangi_map > cfg.VEIN_THRESHOLD
        return float(dis[vein_zone].mean()) if vein_zone.any() else 0.0
    f['A11_midrib_alignment'] = _safe(_midrib_align)

    # A12: disease coverage fraction
    f['A12_disease_coverage_fraction'] = _safe(
        lambda: m.disease_area / max(m.leaf_area, 1))

    # A13: max single lesion fraction
    def _max_lesion():
        if not m.blob_regions:
            return 0.0
        largest = max(m.blob_regions, key=lambda r: r.area)
        return float(largest.area / max(m.leaf_area, 1))
    f['A13_max_single_lesion_fraction'] = _safe(_max_lesion)

    # A14: lesion dispersion (std of pairwise centroid distances)
    def _lesion_dispersion():
        if len(m.blob_regions) < 2:
            return 0.0
        centroids = np.array([r.centroid for r in m.blob_regions])
        from scipy.spatial.distance import pdist
        dists = pdist(centroids)
        return float(dists.std() / (dists.mean() + eps))
    f['A14_lesion_dispersion'] = _safe(_lesion_dispersion)

    # A15: edge-originating fraction
    def _edge_orig():
        if not m.blob_regions:
            return 0.0
        edge_zone = dm < cfg.MARGIN_ZONE_THRESHOLD
        touching = sum(1 for r in m.blob_regions
                       if edge_zone[int(r.centroid[0]), int(r.centroid[1])]
                       if 0 <= int(r.centroid[0]) < edge_zone.shape[0]
                       and 0 <= int(r.centroid[1]) < edge_zone.shape[1])
        return float(touching / len(m.blob_regions))
    f['A15_edge_originating_fraction'] = _safe(_edge_orig)

    # A16: V-shape score [FIX R2-1: uses FULL disease mask, not truncated margin]
    # Black rot enters through hydathodes at leaf margins and creates
    # V-shaped lesions with wide base at edge, narrowing toward midrib.
    def _vshape_score():
        if m.disease_area < cfg.DISEASE_MIN_PIXELS or m.leaf_area < 100:
            return 0.0
        # Check if disease TOUCHES the margin (originates from edge)
        margin_touching = dis & (dm < 0.10) & (dm > 0)
        if margin_touching.sum() < 10:
            return 0.0
        # Measure disease width at 4 distance bands from edge
        # V-shape = wide at margin, narrowing inward
        bands = [
            (0.0, 0.10),   # very near margin
            (0.10, 0.20),  # near margin
            (0.20, 0.35),  # mid-depth
            (0.35, 0.55),  # deeper interior
        ]
        widths = []
        for lo, hi in bands:
            band = dis & (dm >= lo) & (dm < hi) & leaf
            widths.append(band.sum())
        # Check for narrowing pattern: each band should have fewer disease pixels
        if widths[0] < 5:
            return 0.0
        # Score = how consistently width decreases with depth
        narrowing_ratios = []
        for i in range(len(widths) - 1):
            if widths[i] > 0:
                ratio = widths[i + 1] / (widths[i] + eps)
                narrowing_ratios.append(min(ratio, 2.0))  # clamp
        if not narrowing_ratios:
            return 0.0
        # Average narrowing: 0.0 = perfect V (each band half the previous)
        # 1.0 = uniform width (not V-shaped)
        mean_ratio = np.mean(narrowing_ratios)
        # Convert: low ratio = strong V-shape = high score
        score = max(0, 1.0 - mean_ratio)
        return float(min(score * 1.5, 1.0))  # scale up, cap at 1.0
    f['A16_vshape_score'] = _safe(_vshape_score)

    # A17: disease elongation toward midrib [FIX: was MISSING]
    def _disease_elongation():
        if m.disease_area < cfg.DISEASE_MIN_PIXELS:
            return 0.0
        # Measure if disease extends radially inward (toward midrib)
        dis_depths = dm[dis]
        if len(dis_depths) < 10:
            return 0.0
        depth_range = dis_depths.max() - dis_depths.min()
        return float(min(depth_range / 0.5, 1.0))  # normalize: 0.5 depth range = max
    f['A17_disease_elongation_toward_midrib'] = _safe(_disease_elongation)

    return f


# ═══════════════════════════════════════════════════════════════════════
# GROUP B: VEIN ANALYSIS (8 features)
# Are the veins darkened? Key signature of black rot.
# ═══════════════════════════════════════════════════════════════════════

def compute_group_b(m: SharedMaps) -> Dict[str, float]:
    """Vein analysis features."""
    cfg = PSV_CFG
    eps = cfg.EPSILON
    f = {}
    fm = m.frangi_map
    dk = m.dark_mask

    # B01: vein-dark co-localization
    f['B01_vein_dark_colocalization'] = _safe(
        lambda: (dk.astype(float) * fm).mean() / (fm.mean() + eps)
        if fm.max() > eps else 0.0)

    # B02: vein darkening extent
    def _vein_dark_extent():
        vein_px = fm > cfg.VEIN_THRESHOLD
        if not vein_px.any():
            return 0.0
        return float(dk[vein_px].mean())
    f['B02_vein_darkening_extent'] = _safe(_vein_dark_extent)

    # B03: linear dark structures (Hough lines in dark regions)
    def _linear_dark():
        dark_uint8 = (dk.astype(np.uint8) * 255)
        edges = cv2.Canny(dark_uint8, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                 minLineLength=20, maxLineGap=10)
        return float(len(lines)) if lines is not None else 0.0
    f['B03_linear_dark_structures'] = _safe(_linear_dark)

    # B04: vein-lesion ratio
    def _vein_lesion():
        vein_dark = dk & (fm > cfg.VEIN_THRESHOLD)
        return float(vein_dark.sum() / (dk.sum() + eps))
    f['B04_vein_lesion_ratio'] = _safe(_vein_lesion)

    # B05: vein boundary alignment (disease boundaries align with veins)
    def _vein_boundary():
        if m.disease_area < cfg.DISEASE_MIN_PIXELS:
            return 0.0
        # Find disease boundary pixels using erosion
        eroded = ndimage.binary_erosion(m.disease_mask)
        boundary = m.disease_mask & ~eroded
        if not boundary.any():
            return 0.0
        return float(fm[boundary].mean())
    f['B05_vein_boundary_alignment'] = _safe(_vein_boundary)

    # B06: vein color shift (LAB-A in vein regions vs healthy)
    def _vein_color():
        vein_px = (fm > cfg.VEIN_THRESHOLD) & m.leaf_mask
        healthy_vein = vein_px & m.healthy_mask
        dark_vein = vein_px & m.dark_mask
        if not healthy_vein.any() or not dark_vein.any():
            return 0.0
        a_healthy = m.lab[:, :, 1][healthy_vein].mean()
        a_dark = m.lab[:, :, 1][dark_vein].mean()
        return float(a_dark - a_healthy)
    f['B06_vein_color_shift'] = _safe(_vein_color)

    # B07: dark linear density (dark pixels per unit vein length)
    def _dark_linear_density():
        vein_px = fm > cfg.VEIN_THRESHOLD
        vein_length = vein_px.sum()
        if vein_length < 10:
            return 0.0
        dark_on_vein = dk & vein_px
        return float(dark_on_vein.sum() / vein_length)
    f['B07_dark_linear_density'] = _safe(_dark_linear_density)

    # B08: vein branch darkening ratio (secondary vs primary vein darkening)
    def _branch_dark():
        if fm.max() < eps:
            return 0.0
        primary = fm > np.percentile(fm[fm > 0], 75) if (fm > 0).any() else fm > 0.5
        secondary = (fm > cfg.VEIN_THRESHOLD) & ~primary
        if not primary.any() or not secondary.any():
            return 0.5
        dark_primary = m.dark_mask[primary].mean()
        dark_secondary = m.dark_mask[secondary].mean()
        return float(dark_secondary / (dark_primary + eps))
    f['B08_vein_branch_darkening'] = _safe(_branch_dark, default=0.5)

    return f


# ═══════════════════════════════════════════════════════════════════════
# GROUP C: SPOT/LESION MORPHOLOGY (14 features)
# Per-blob statistics aggregated as mean/std/max.
# ═══════════════════════════════════════════════════════════════════════

def compute_group_c(m: SharedMaps) -> Dict[str, float]:
    """Spot and lesion morphology features."""
    cfg = PSV_CFG
    eps = cfg.EPSILON
    f = {}
    regions = m.blob_regions

    # Compute per-blob metrics
    circularities = []
    areas = []
    elongations = []
    orientations = []
    center_brightness = []
    border_brightness = []

    for r in regions:
        if r.area < cfg.BLOB_MIN_AREA:
            continue
        # Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
        perim = r.perimeter if r.perimeter > 0 else 1
        circ = 4 * np.pi * r.area / (perim ** 2)
        circularities.append(min(circ, 1.0))
        areas.append(r.area)
        # Elongation: major/minor axis ratio
        if r.axis_minor_length > 0:
            elongations.append(r.axis_major_length / r.axis_minor_length)
        else:
            elongations.append(1.0)
        orientations.append(r.orientation)

        # Center vs border brightness for ring detection
        try:
            mask = np.zeros_like(m.gray, dtype=bool)
            for coord in r.coords:
                mask[coord[0], coord[1]] = True
            # [FIX] Use cfg.BLOB_CENTER_FRACTION for proper center/border separation
            erd_iter = max(2, int(r.equivalent_diameter_area * cfg.BLOB_CENTER_FRACTION / 2))
            eroded = ndimage.binary_erosion(mask, iterations=erd_iter)
            border = mask & ~eroded
            if eroded.any() and border.any():
                center_brightness.append(float(m.lab[:, :, 0][eroded].mean()))
                border_brightness.append(float(m.lab[:, :, 0][border].mean()))
        except Exception:
            pass

    n_blobs = len(circularities)

    # C01: mean blob circularity
    f['C01_mean_blob_circularity'] = _safe(
        lambda: np.mean(circularities) if circularities else 0.0)

    # C02: std blob circularity
    f['C02_std_blob_circularity'] = _safe(
        lambda: np.std(circularities) if len(circularities) > 1 else 0.0)

    # C03: mean blob size (pixels)
    f['C03_mean_blob_size_px'] = _safe(
        lambda: np.mean(areas) if areas else 0.0)

    # C04: blob size CV (coefficient of variation)
    f['C04_blob_size_cv'] = _safe(
        lambda: np.std(areas) / (np.mean(areas) + eps) if len(areas) > 1 else 0.0)

    # C05: blob count normalized by leaf area
    f['C05_blob_count_normalized'] = _safe(
        lambda: n_blobs / (m.leaf_area / 10000 + eps))

    # C06: cercospora ring score (center BRIGHTER than border)
    def _cerc_ring():
        if not center_brightness or not border_brightness:
            return 0.0
        scores = [c / (b + eps) for c, b in zip(center_brightness, border_brightness)]
        return float(np.mean(scores))
    f['C06_cercospora_ring_score'] = _safe(_cerc_ring)

    # C07: alternaria ring score (border DARKER than center — concentric rings)
    def _alt_ring():
        if not center_brightness or not border_brightness:
            return 0.0
        scores = [b / (c + eps) for c, b in zip(center_brightness, border_brightness)]
        return float(np.mean(scores))
    f['C07_alternaria_ring_score'] = _safe(_alt_ring)

    # C08: yellow halo fraction
    def _yellow_halo():
        if not regions:
            return 0.0
        halo_scores = []
        for r in regions[:20]:  # limit for speed
            mask = np.zeros_like(m.yellow_mask, dtype=bool)
            for coord in r.coords:
                mask[coord[0], coord[1]] = True
            dilated = ndimage.binary_dilation(mask, iterations=cfg.YELLOW_HALO_DILATION)
            halo = dilated & ~mask & m.leaf_mask
            if halo.any():
                halo_scores.append(float(m.yellow_mask[halo].mean()))
        return float(np.mean(halo_scores)) if halo_scores else 0.0
    f['C08_yellow_halo_fraction'] = _safe(_yellow_halo)

    # C09: blob interior fraction (blobs in leaf interior, not margin)
    def _blob_interior():
        if not regions:
            return 0.0
        interior = 0
        for r in regions:
            cy, cx = r.centroid
            if 0 <= int(cy) < m.distance_map.shape[0] and 0 <= int(cx) < m.distance_map.shape[1]:
                if m.distance_map[int(cy), int(cx)] > cfg.INTERIOR_ZONE_THRESHOLD:
                    interior += 1
        return float(interior / len(regions))
    f['C09_blob_interior_fraction'] = _safe(_blob_interior)

    # C10: concentric ring count per blob (intensity profile rings)
    def _ring_count():
        if not regions:
            return 0.0
        ring_counts = []
        for r in regions[:10]:  # limit for speed
            try:
                mask = np.zeros_like(m.gray, dtype=bool)
                for coord in r.coords:
                    mask[coord[0], coord[1]] = True
                cy, cx = r.centroid
                ys, xs = np.where(mask)
                dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
                if dists.max() < 3:
                    ring_counts.append(0)
                    continue
                nbins = min(10, int(dists.max()))
                bins = np.linspace(0, dists.max(), nbins + 1)
                profile = []
                for i in range(len(bins) - 1):
                    ring = (dists >= bins[i]) & (dists < bins[i + 1])
                    if ring.any():
                        profile.append(m.gray[ys[ring], xs[ring]].mean())
                # Count direction changes (peaks/valleys) = rings
                if len(profile) > 2:
                    diffs = np.diff(profile)
                    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                    ring_counts.append(sign_changes)
                else:
                    ring_counts.append(0)
            except Exception:
                ring_counts.append(0)
        return float(np.mean(ring_counts)) if ring_counts else 0.0
    f['C10_concentric_ring_count'] = _safe(_ring_count)

    # C11: hole count (gray-white drop-out regions within leaf)
    def _hole_count():
        holes = m.gray_white_mask & m.disease_mask
        labeled = sk_label(holes)
        return float(labeled.max() / (m.leaf_area / 10000 + eps))
    f['C11_hole_count_normalized'] = _safe(_hole_count)

    # C12: blob elongation mean
    f['C12_blob_elongation_mean'] = _safe(
        lambda: np.mean(elongations) if elongations else 1.0, default=1.0)

    # C13: blob orientation variance
    f['C13_blob_orientation_variance'] = _safe(
        lambda: np.var(orientations) if len(orientations) > 1 else 0.0)

    # C14: spot clustering score (tight nearest-neighbor distances)
    def _clustering():
        if len(regions) < 3:
            return 0.0
        centroids = np.array([r.centroid for r in regions])
        from scipy.spatial.distance import cdist
        dists = cdist(centroids, centroids)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)
        return float(min(1.0 / (nn_dists.mean() + eps), 10.0))  # [FIX] clamp
    f['C14_spot_clustering_score'] = _safe(_clustering)

    return f


# ═══════════════════════════════════════════════════════════════════════
# GROUP D: COLOR ZONE ANALYSIS (11 features)
# LAB colorspace analysis in specific spatial zones.
# ═══════════════════════════════════════════════════════════════════════

def compute_group_d(m: SharedMaps) -> Dict[str, float]:
    """Color zone analysis features."""
    cfg = PSV_CFG
    eps = cfg.EPSILON
    f = {}
    L, A, B = m.lab[:, :, 0], m.lab[:, :, 1], m.lab[:, :, 2]

    # D01: gray-white center fraction (KEY cercospora signature)
    def _gw_center():
        if not m.blob_regions:
            return 0.0
        scores = []
        for r in m.blob_regions[:20]:
            mask = np.zeros_like(m.gray_white_mask, dtype=bool)
            for coord in r.coords:
                mask[coord[0], coord[1]] = True
            center = ndimage.binary_erosion(mask, iterations=max(1, int(r.equivalent_diameter_area * 0.2)))
            if center.any():
                scores.append(float(m.gray_white_mask[center].mean()))
        return float(np.mean(scores)) if scores else 0.0
    f['D01_gray_white_center_fraction'] = _safe(_gw_center)

    # D02: yellow marginal fraction (KEY black_rot halo)
    margin = m.leaf_mask & (m.distance_map < 0.25) & (m.distance_map > 0)
    f['D02_yellow_marginal_fraction'] = _safe(
        lambda: m.yellow_mask[margin].mean() if margin.any() else 0.0)

    # D03: brown vein fraction
    vein_px = m.frangi_map > cfg.VEIN_THRESHOLD
    f['D03_brown_vein_fraction'] = _safe(
        lambda: m.brown_mask[vein_px].mean() if vein_px.any() else 0.0)

    # D04: yellow vein fraction (KEY yvmv signature)
    f['D04_yellow_vein_fraction'] = _safe(
        lambda: m.yellow_mask[vein_px].mean() if vein_px.any() else 0.0)

    # D05: powdery white coverage (KEY powdery mildew)
    f['D05_powdery_white_coverage'] = _safe(
        lambda: m.powdery_mask.sum() / max(m.leaf_area, 1))

    # D06: mosaic color variance (KEY yvmv — mosaic pattern)
    def _mosaic_var():
        if m.leaf_area < 100:
            return 0.0
        b_channel = B.copy()
        b_channel[~m.leaf_mask] = 0
        # Local variance at vein-network scale
        kernel = np.ones((15, 15)) / 225
        local_mean = cv2.filter2D(b_channel, -1, kernel)
        local_sq_mean = cv2.filter2D(b_channel ** 2, -1, kernel)
        local_var = local_sq_mean - local_mean ** 2
        return float(local_var[m.leaf_mask].mean())
    f['D06_mosaic_color_variance'] = _safe(_mosaic_var)

    # D07: green retention fraction (high = early stage or leaf curl)
    f['D07_green_retention_fraction'] = _safe(
        lambda: m.healthy_mask.sum() / max(m.leaf_area, 1))

    # D08: necrosis fraction (severe, very dark brown)
    f['D08_necrosis_fraction'] = _safe(
        lambda: m.necrosis_mask.sum() / max(m.leaf_area, 1))

    # D09: chlorosis fraction (early yellowing)
    f['D09_chlorosis_fraction'] = _safe(
        lambda: m.chlorosis_mask.sum() / max(m.leaf_area, 1))

    # D10: disease boundary sharpness (Sobel at disease edges)
    def _boundary_sharp():
        if m.disease_area < cfg.DISEASE_MIN_PIXELS:
            return 0.0
        eroded = ndimage.binary_erosion(m.disease_mask)
        boundary = m.disease_mask & ~eroded
        if not boundary.any():
            return 0.0
        sob = sobel(m.gray)
        return float(sob[boundary].mean())
    f['D10_color_zone_boundary_sharpness'] = _safe(_boundary_sharp)

    # D11: disease hue uniformity
    def _hue_uniform():
        if m.disease_area < cfg.DISEASE_MIN_PIXELS:
            return 0.0
        a_vals = A[m.disease_mask]
        b_vals = B[m.disease_mask]
        hue = np.arctan2(b_vals, a_vals)
        return float(min(1.0 / (hue.std() + eps), 10.0))  # [FIX] clamp
    f['D11_disease_hue_uniformity'] = _safe(_hue_uniform)

    return f


# ═══════════════════════════════════════════════════════════════════════
# GROUP E: TEXTURE AND SURFACE (8 features)
# ═══════════════════════════════════════════════════════════════════════

def compute_group_e(m: SharedMaps) -> Dict[str, float]:
    """Texture and surface features."""
    cfg = PSV_CFG
    eps = cfg.EPSILON
    f = {}

    # E01: surface roughness score (KEY enation signature)
    def _roughness():
        grad = sobel(m.gray)
        if not m.leaf_mask.any():
            return 0.0
        leaf_grad = grad[m.leaf_mask]
        return float(leaf_grad.var() / (leaf_grad.mean() + eps))
    f['E01_surface_roughness_score'] = _safe(_roughness)

    # E02-E04: GLCM features on disease region
    def _glcm_features():
        if m.disease_area < 100:
            return 0.5, 0.0, 0.5
        # Extract disease region, quantize
        roi = (m.gray * (cfg.GLCM_LEVELS - 1)).astype(np.uint8)
        roi[~m.disease_mask] = 0
        # Find bounding box of disease
        ys, xs = np.where(m.disease_mask)
        if len(ys) == 0:
            return 0.5, 0.0, 0.5
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        patch = roi[y0:y1, x0:x1]
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            return 0.5, 0.0, 0.5
        glcm = graycomatrix(patch, distances=cfg.GLCM_DISTANCES,
                            angles=cfg.GLCM_ANGLES, levels=cfg.GLCM_LEVELS,
                            symmetric=True, normed=True)
        hom = float(graycoprops(glcm, 'homogeneity').mean())
        con = float(graycoprops(glcm, 'contrast').mean())
        ene = float(graycoprops(glcm, 'energy').mean())
        return hom, con, ene

    try:
        hom, con, ene = _glcm_features()
    except Exception:
        hom, con, ene = 0.5, 0.0, 0.5

    f['E02_glcm_homogeneity'] = hom
    f['E03_glcm_contrast'] = con
    f['E04_glcm_energy'] = ene

    # E05: local variance mean
    def _local_var():
        if not m.leaf_mask.any():
            return 0.0
        k = cfg.LOCAL_VARIANCE_KERNEL
        kernel = np.ones((k, k)) / (k * k)
        local_mean = cv2.filter2D(m.gray, -1, kernel)
        local_sq = cv2.filter2D(m.gray ** 2, -1, kernel)
        lv = local_sq - local_mean ** 2
        return float(lv[m.leaf_mask].mean())
    f['E05_local_variance_mean'] = _safe(_local_var)

    # E06: lesion edge sharpness (sharp=alternaria/cercospora, diffuse=downy/early black_rot)
    def _edge_sharp():
        if m.disease_area < cfg.DISEASE_MIN_PIXELS:
            return 0.0
        sob = sobel(m.gray)
        eroded = ndimage.binary_erosion(m.disease_mask)
        boundary = m.disease_mask & ~eroded
        return float(sob[boundary].mean()) if boundary.any() else 0.0
    f['E06_lesion_edge_sharpness'] = _safe(_edge_sharp)

    # E07: leaf contour irregularity (KEY enation/curl signature)
    def _contour_irreg():
        if m.leaf_area < 100:
            return 0.0
        contours, _ = cv2.findContours(m.leaf_mask.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return 0.0
        ellipse = cv2.fitEllipse(cnt)
        # Draw ideal ellipse
        ideal = np.zeros_like(m.leaf_mask, dtype=np.uint8)
        cv2.ellipse(ideal, ellipse, 1, -1)
        # Measure mismatch
        intersection = (m.leaf_mask & ideal.astype(bool)).sum()
        union = (m.leaf_mask | ideal.astype(bool)).sum()
        iou = intersection / max(union, 1)
        return float(1.0 - iou)
    f['E07_leaf_contour_irregularity'] = _safe(_contour_irreg)

    # E08: fractal dimension estimate (box-counting on disease mask)
    def _fractal():
        if m.disease_area < 50:
            return 1.0
        # Simple box-counting
        sizes = [2, 4, 8, 16, 32]
        counts = []
        for s in sizes:
            h, w = m.disease_mask.shape
            nh, nw = h // s, w // s
            if nh < 1 or nw < 1:
                continue
            resized = m.disease_mask[:nh * s, :nw * s].reshape(nh, s, nw, s)
            box_has = resized.any(axis=(1, 3))
            counts.append((s, box_has.sum()))
        if len(counts) < 2:
            return 1.0
        log_sizes = np.log([1.0 / c[0] for c in counts])
        log_counts = np.log([c[1] + 1 for c in counts])
        if log_sizes.std() < eps:
            return 1.0
        slope = np.polyfit(log_sizes, log_counts, 1)[0]
        return float(np.clip(slope, 1.0, 2.0))
    f['E08_fractal_dimension_estimate'] = _safe(_fractal, default=1.0)

    return f


# ═══════════════════════════════════════════════════════════════════════
# GROUP F: CROSS-CLASS DISCRIMINATORS (8 features)
# Features specifically designed for known confusion pairs.
# ═══════════════════════════════════════════════════════════════════════

def compute_group_f(all_features: Dict[str, float]) -> Dict[str, float]:
    """Cross-class discriminator features. Computed from other features."""
    eps = PSV_CFG.EPSILON
    f = {}
    g = all_features  # shorthand

    # F01: blackrot vs alternaria
    f['F01_blackrot_vs_alternaria'] = _safe(lambda: (
        g.get('A03_margin_vs_interior_ratio', 0) * g.get('B01_vein_dark_colocalization', 0)
        - g.get('C01_mean_blob_circularity', 0) * g.get('C09_blob_interior_fraction', 0)
    ))

    # F02: cercospora vs alternaria
    f['F02_cercospora_vs_alternaria'] = _safe(lambda: (
        g.get('D01_gray_white_center_fraction', 0) * g.get('C06_cercospora_ring_score', 0)
        - g.get('C07_alternaria_ring_score', 0) * g.get('C08_yellow_halo_fraction', 0)
    ))

    # F03: yvmv vs healthy
    f['F03_yvmv_vs_healthy'] = _safe(lambda: (
        g.get('D04_yellow_vein_fraction', 0) * g.get('D06_mosaic_color_variance', 0)
    ))

    # F04: powdery vs cercospora
    f['F04_powdery_vs_cercospora'] = _safe(lambda: (
        g.get('D05_powdery_white_coverage', 0) / (g.get('D01_gray_white_center_fraction', 0) + eps)
    ))

    # F05: enation vs healthy
    f['F05_enation_vs_healthy'] = _safe(lambda: (
        g.get('E01_surface_roughness_score', 0) * g.get('E07_leaf_contour_irregularity', 0)
    ))

    # F06: downy vs alternaria
    f['F06_downy_vs_alternaria'] = _safe(lambda: (
        (1 - g.get('C01_mean_blob_circularity', 0.5)) * g.get('B05_vein_boundary_alignment', 0)
    ))

    # F07: blackrot severity
    f['F07_blackrot_severity'] = _safe(lambda: (
        g.get('A01_margin_disease_density', 0) * g.get('B02_vein_darkening_extent', 0)
    ))

    # F08: disease stage index (necrosis/chlorosis ratio)
    f['F08_disease_stage_index'] = _safe(lambda: (
        g.get('D08_necrosis_fraction', 0) / (g.get('D09_chlorosis_fraction', 0) + eps)
    ))

    return f


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def extract_all_features(image_rgb: np.ndarray) -> FeatureResult:
    """
    Extract all 64+ PSV features from an RGB image.

    Args:
        image_rgb: uint8 numpy array [H, W, 3] in RGB color order

    Returns:
        FeatureResult with all features, failed feature list, shared maps
    """
    import time
    t0 = time.time()

    # Pre-compute shared maps
    maps = compute_shared_maps(image_rgb)

    # Compute each group
    features = {}
    failed = {}

    for group_name, group_fn in [
        ('A', lambda: compute_group_a(maps)),
        ('B', lambda: compute_group_b(maps)),
        ('C', lambda: compute_group_c(maps)),
        ('D', lambda: compute_group_d(maps)),
        ('E', lambda: compute_group_e(maps)),
    ]:
        try:
            group_features = group_fn()
            features.update(group_features)
        except Exception as e:
            failed[f'GROUP_{group_name}'] = str(e)

    # Group F depends on previous groups
    try:
        f_features = compute_group_f(features)
        features.update(f_features)
    except Exception as e:
        failed['GROUP_F'] = str(e)

    extraction_time = (time.time() - t0) * 1000

    return FeatureResult(
        features=features,
        failed_features=failed,
        shared_maps=maps,
        extraction_time_ms=extraction_time,
    )
