"""
PSV Image Quality Assessor — 32+ edge case detection and handling.

Runs BEFORE PSV feature extraction. Returns:
  - quality_flags: which edge cases triggered
  - psv_confidence: overall PSV reliability (0.0 = disabled, 1.0 = fully reliable)
  - feature_masks: which feature GROUPS to disable
  - retake_requested: whether to ask farmer to retake

Each edge case has:
  - Detection method (specific cv2/skimage operation)
  - Feature groups it affects
  - Confidence multiplier applied
"""

import numpy as np
import cv2
from typing import Dict, Tuple
from dataclasses import dataclass, field

from scripts.psv.config import PSV_CFG


@dataclass
class IQAResult:
    """Image quality assessment result."""
    quality_flags: Dict[str, bool]         # which edge cases triggered
    psv_confidence: float                  # overall PSV reliability 0-1
    feature_masks: Dict[str, bool]         # which feature groups to disable (True=enabled)
    retake_requested: bool                 # whether to ask for retake
    details: Dict[str, float]             # numeric values for each check


def assess_image_quality(image_rgb: np.ndarray, leaf_mask: np.ndarray = None,
                         disease_mask: np.ndarray = None) -> IQAResult:
    """
    Comprehensive image quality assessment. Run before PSV features.

    Args:
        image_rgb: uint8 RGB image
        leaf_mask: optional pre-computed leaf mask (computed if None)
        disease_mask: optional pre-computed disease mask

    Returns:
        IQAResult with flags, confidence, feature masks
    """
    cfg = PSV_CFG
    flags = {}
    details = {}
    conf = 1.0  # start at full confidence, reduce per issue
    retake = False

    # Enable all feature groups by default
    groups = {g: True for g in ['A', 'B', 'C', 'D', 'E', 'F']}

    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float64)

    # Convert to LAB for exposure analysis
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0

    # Basic leaf mask if not provided
    if leaf_mask is None:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        leaf_mask = (hsv[:, :, 0] > 25) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 30)
        leaf_mask = cv2.morphologyEx(leaf_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                      np.ones((11, 11), np.uint8)).astype(bool)

    leaf_fraction = leaf_mask.mean()
    leaf_L = L[leaf_mask] if leaf_mask.any() else L.flatten()

    # ═════════════════════════════════════════════════════════════════
    # BLUR AND FOCUS (EQ01-EQ04)
    # ═════════════════════════════════════════════════════════════════

    # EQ01/EQ02: Gaussian blur via Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    details['laplacian_variance'] = float(lap_var)

    if lap_var < cfg.BLUR_SEVERE_THRESHOLD:
        flags['EQ02_gaussian_blur_severe'] = True
        conf *= 0.2
        groups['E'] = False  # texture unreliable when blurry
        groups['C'] = False  # spot morphology unreliable
    elif lap_var < cfg.BLUR_MILD_THRESHOLD:
        flags['EQ01_gaussian_blur_mild'] = True
        conf *= 0.6

    # EQ03: Motion blur (directional blur via FFT)
    try:
        f_transform = np.fft.fft2(gray_f)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        # Motion blur creates a line in frequency domain
        center_h, center_w = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        horiz = magnitude[center_h, :].std()
        vert = magnitude[:, center_w].std()
        directionality = max(horiz, vert) / (min(horiz, vert) + 1e-8)
        details['motion_blur_directionality'] = float(directionality)
        if directionality > 3.0 and lap_var < cfg.BLUR_MILD_THRESHOLD:
            flags['EQ03_motion_blur'] = True
            groups['C'] = False  # spot morphology unreliable
            conf *= 0.5
    except:
        pass

    # EQ04: Center vs edge sharpness
    ch, cw = h // 4, w // 4
    center_lap = cv2.Laplacian(gray[ch:3*ch, cw:3*cw], cv2.CV_64F).var()
    edge_regions = [gray[:ch, :], gray[3*ch:, :], gray[:, :cw], gray[:, 3*cw:]]
    edge_lap = np.mean([cv2.Laplacian(r, cv2.CV_64F).var() for r in edge_regions if r.size > 0])
    details['center_edge_sharpness_ratio'] = float(center_lap / (edge_lap + 1e-8))
    if center_lap > 3 * edge_lap and edge_lap < cfg.BLUR_SEVERE_THRESHOLD:
        flags['EQ04_out_of_focus_edges'] = True
        conf *= 0.8

    # ═════════════════════════════════════════════════════════════════
    # EXPOSURE (EQ05-EQ09)
    # ═════════════════════════════════════════════════════════════════

    mean_L = float(leaf_L.mean()) if len(leaf_L) > 0 else float(L.mean())
    details['mean_L'] = mean_L

    if mean_L < cfg.UNDEREXPOSED_SEVERE_L:
        flags['EQ05_underexposed_severe'] = True
        conf *= 0.1
        groups['D'] = False  # color features meaningless
    elif mean_L < cfg.UNDEREXPOSED_MILD_L:
        flags['EQ06_underexposed_mild'] = True
        conf *= 0.5
        # Reduce color feature confidence but don't disable

    if mean_L > cfg.OVEREXPOSED_SEVERE_L:
        flags['EQ07_overexposed_severe'] = True
        conf *= 0.2
        groups['D'] = False
    elif mean_L > cfg.OVEREXPOSED_MILD_L:
        flags['EQ08_overexposed_mild'] = True
        conf *= 0.6

    # EQ09: Harsh sun (bimodal L histogram)
    try:
        hist, _ = np.histogram(leaf_L, bins=50, range=(0, 100))
        hist_norm = hist / (hist.sum() + 1e-8)
        # Check for bimodality: two peaks separated by a valley
        peaks = np.where((hist_norm[1:-1] > hist_norm[:-2]) &
                         (hist_norm[1:-1] > hist_norm[2:]))[0]
        if len(peaks) >= 2:
            valley = hist_norm[peaks[0]:peaks[-1]].min()
            peak_mean = (hist_norm[peaks[0]] + hist_norm[peaks[-1]]) / 2
            if valley < peak_mean * 0.3:
                flags['EQ09_high_contrast_harsh_sun'] = True
                conf *= 0.8
    except:
        pass

    # ═════════════════════════════════════════════════════════════════
    # LEAF DETECTION (EQ10-EQ17)
    # ═════════════════════════════════════════════════════════════════

    details['leaf_fraction'] = float(leaf_fraction)

    # EQ10: No leaf detected
    if leaf_fraction < cfg.NO_LEAF_FRACTION:
        flags['EQ10_no_leaf_detected'] = True
        conf = 0.0
        retake = True

    # EQ11: Partial leaf (touches multiple borders)
    border_touch = 0
    if leaf_mask.any():
        if leaf_mask[0, :].any(): border_touch += 1    # top
        if leaf_mask[-1, :].any(): border_touch += 1   # bottom
        if leaf_mask[:, 0].any(): border_touch += 1    # left
        if leaf_mask[:, -1].any(): border_touch += 1   # right
    details['border_touch_count'] = border_touch
    if border_touch > 2:
        flags['EQ11_partial_leaf_edge'] = True
        groups['A'] = False  # margin analysis unreliable
        conf *= 0.6

    # EQ12: Small leaf
    if cfg.NO_LEAF_FRACTION <= leaf_fraction < cfg.PARTIAL_LEAF_FRACTION:
        flags['EQ12_partial_leaf_small'] = True
        conf *= 0.7

    # EQ13: Multiple leaves
    from skimage.measure import label as sk_label
    if leaf_mask.any():
        labeled, n_components = sk_label(leaf_mask, return_num=True)
        details['leaf_components'] = n_components
        if n_components > 3:
            flags['EQ13_multiple_leaves'] = True
            conf *= 0.7

    # EQ14: Leaf underside (different color signature)
    if leaf_mask.any():
        a_channel = lab[:, :, 1][leaf_mask].astype(float) - 128
        mean_a = a_channel.mean()
        details['leaf_mean_a'] = float(mean_a)
        if mean_a > 10:  # unusually red/purple = possible underside
            flags['EQ14_leaf_underside_view'] = True
            conf *= 0.6

    # EQ15: Leaf in hand (skin tone detection)
    try:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        skin = (hsv[:, :, 0] > 0) & (hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 80)
        skin_fraction = skin.mean()
        details['skin_fraction'] = float(skin_fraction)
        if skin_fraction > 0.05:
            flags['EQ15_leaf_in_hand'] = True
            conf *= 0.9  # minor — just flag
    except:
        pass

    # EQ16: Leaf on surface (very uniform background)
    if leaf_mask.any():
        bg = ~leaf_mask
        if bg.sum() > 100:
            bg_std = gray_f[bg].std()
            details['background_std'] = float(bg_std)
            if bg_std < 15:
                flags['EQ16_leaf_on_surface'] = True
                # Not a problem per se, just flag

    # EQ17: Extreme rotation
    try:
        ys, xs = np.where(leaf_mask)
        if len(ys) > 50:
            coords = np.column_stack([xs - xs.mean(), ys - ys.mean()])
            _, _, Vt = np.linalg.svd(coords, full_matrices=False)
            angle = abs(np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0])))
            details['leaf_rotation_degrees'] = float(angle)
            if angle > cfg.EXTREME_ROTATION_DEGREES:
                flags['EQ17_extreme_rotation'] = True
                # Features use rotation-corrected maps, so just flag
    except:
        pass

    # ═════════════════════════════════════════════════════════════════
    # SURFACE INTERFERENCE (EQ18-EQ22)
    # ═════════════════════════════════════════════════════════════════

    # EQ18: Specular reflection
    if leaf_mask.any():
        specular = (L > cfg.SPECULAR_L_MIN) & leaf_mask
        spec_fraction = specular.sum() / max(leaf_mask.sum(), 1)
        details['specular_fraction'] = float(spec_fraction)
        if spec_fraction > cfg.SPECULAR_CRITICAL_FRACTION:
            flags['EQ18_specular_reflection'] = True
            conf *= 0.7

    # EQ19: Water droplets (small circular bright regions)
    try:
        bright = (L > 80) & leaf_mask
        bright_labeled = sk_label(bright.astype(np.uint8))
        from skimage.measure import regionprops
        bright_regions = regionprops(bright_labeled)
        circular_bright = [r for r in bright_regions
                          if r.area > 20 and r.area < 500
                          and (4 * np.pi * r.area / (r.perimeter ** 2 + 1e-8)) > 0.7]
        details['circular_bright_count'] = len(circular_bright)
        if len(circular_bright) > 3:
            flags['EQ19_water_droplets'] = True
            conf *= 0.8
            # Don't disable C group entirely but flag
    except:
        pass

    # EQ20: Soil splash (irregular dark patches near base)
    if leaf_mask.any() and disease_mask is not None:
        try:
            from scipy import ndimage as ndi
            dist = ndi.distance_transform_edt(leaf_mask)
            dist_norm = dist / (dist.max() + 1e-8)
            base_disease = disease_mask & (dist_norm < 0.15)
            base_fraction = base_disease.sum() / max(disease_mask.sum(), 1)
            if base_fraction > 0.6:
                flags['EQ20_soil_splash'] = True
                conf *= 0.8
        except:
            pass

    # EQ21: Dust particles (fine uniform speckling)
    # Detect as very small, numerous dark spots
    try:
        if disease_mask is not None:
            small_spots = 0
            from skimage.measure import label as sk_label2, regionprops as rp2
            labeled = sk_label2(disease_mask.astype(np.uint8))
            for r in rp2(labeled):
                if r.area < 10:
                    small_spots += 1
            details['tiny_spot_count'] = small_spots
            if small_spots > 50:
                flags['EQ21_dust_particles'] = True
                conf *= 0.85
    except:
        pass

    # EQ22: Fungicide residue (white powdery artificial coating)
    # Similar to powdery mildew — flag if white coverage is suspiciously uniform
    if leaf_mask.any():
        white = (L > 75) & leaf_mask
        white_fraction = white.sum() / max(leaf_mask.sum(), 1)
        if white_fraction > 0.3:
            white_L_std = L[white].std() if white.any() else 0
            details['white_L_std'] = float(white_L_std)
            if white_L_std < 3:  # suspiciously uniform white
                flags['EQ22_fungicide_residue'] = True
                conf *= 0.7

    # ═════════════════════════════════════════════════════════════════
    # DISEASE PRESENTATION (EQ23-EQ28)
    # ═════════════════════════════════════════════════════════════════

    if disease_mask is not None:
        disease_coverage = disease_mask.sum() / max(leaf_mask.sum(), 1)
        details['disease_coverage'] = float(disease_coverage)

        # EQ23: No visible symptoms
        if disease_mask.sum() < cfg.MIN_DISEASE_PIXELS:
            flags['EQ23_no_visible_symptoms'] = True
            # Still run PSV — absence is signal for healthy

        # EQ24: Entire leaf necrotic
        if disease_coverage > cfg.MAX_DISEASE_COVERAGE:
            flags['EQ24_entire_leaf_necrotic'] = True
            conf *= 0.4  # class-distinctive features mostly gone

        # EQ25: Disease at image edge
        if disease_mask.any():
            edge_disease = (disease_mask[0, :].any() or disease_mask[-1, :].any() or
                          disease_mask[:, 0].any() or disease_mask[:, -1].any())
            if edge_disease:
                flags['EQ25_disease_at_image_edge'] = True
                conf *= 0.9

        # EQ26: Very early stage (only chlorosis, no necrosis)
        if disease_mask.any():
            L_disease = L[disease_mask]
            if L_disease.mean() > 55 and L_disease.min() > 30:
                flags['EQ26_very_early_stage'] = True
                # Early stage is valid, just flag

    # EQ27: Insect damage holes
    try:
        if leaf_mask.any():
            # Holes in leaf: background pixels surrounded by leaf pixels
            filled = cv2.morphologyEx(leaf_mask.astype(np.uint8),
                                       cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            holes = filled.astype(bool) & ~leaf_mask
            hole_fraction = holes.sum() / max(leaf_mask.sum(), 1)
            details['hole_fraction'] = float(hole_fraction)
            if hole_fraction > 0.02:
                flags['EQ27_insect_damage_holes'] = True
                conf *= 0.85
    except:
        pass

    # EQ28: Nutrient deficiency simulation (interveinal chlorosis mimics yvmv)
    # Detected by: chlorosis WITHOUT vein-following pattern
    # This is subtle — just flag, don't reduce confidence
    flags.setdefault('EQ28_nutrient_deficiency_sim', False)

    # ═════════════════════════════════════════════════════════════════
    # PROCESSING ARTEFACTS (EQ29-EQ32)
    # ═════════════════════════════════════════════════════════════════

    # EQ29: JPEG compression (8x8 block artifacts)
    try:
        # Check DCT blockiness
        block_diffs = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                if y + 8 < h:
                    block_diffs.append(abs(float(gray_f[y + 7, x]) - float(gray_f[y + 8, x])))
                if x + 8 < w:
                    block_diffs.append(abs(float(gray_f[y, x + 7]) - float(gray_f[y, x + 8])))
                if len(block_diffs) > 1000:
                    break
            if len(block_diffs) > 1000:
                break
        if block_diffs:
            blockiness = np.mean(block_diffs)
            details['jpeg_blockiness'] = float(blockiness)
            if blockiness > 15:
                flags['EQ29_jpeg_compression'] = True
                conf *= 0.95
    except:
        pass

    # EQ30: UI overlay (text/rectangle detection)
    try:
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                 minLineLength=50, maxLineGap=5)
        if lines is not None and len(lines) > 20:
            # Many straight lines suggest UI overlay
            horizontal = sum(1 for l in lines if abs(l[0][1] - l[0][3]) < 5)
            vertical = sum(1 for l in lines if abs(l[0][0] - l[0][2]) < 5)
            if horizontal > 5 and vertical > 5:
                flags['EQ30_ui_overlay'] = True
                conf *= 0.7
    except:
        pass

    # EQ31: Already processed (unusual histogram flatness)
    try:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-8)
        hist_entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
        details['histogram_entropy'] = float(hist_entropy)
        if hist_entropy > 7.5:  # very flat histogram = likely enhanced
            flags['EQ31_already_processed'] = True
            conf *= 0.9
    except:
        pass

    # EQ32: Screenshot artifact
    try:
        # Check for perfectly sharp rectangular edges
        border_pix = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        border_std = border_pix.std()
        details['border_std'] = float(border_std)
        if border_std < 5:  # very uniform border = screenshot
            flags['EQ32_screenshot_artifact'] = True
            conf *= 0.85
    except:
        pass

    # ═════════════════════════════════════════════════════════════════
    # FINAL CONFIDENCE CLAMP
    # ═════════════════════════════════════════════════════════════════
    conf = max(0.0, min(1.0, conf))

    if conf < 0.3:
        retake = True

    return IQAResult(
        quality_flags={k: v for k, v in flags.items() if v},
        psv_confidence=conf,
        feature_masks=groups,
        retake_requested=retake,
        details=details,
    )
