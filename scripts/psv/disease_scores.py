"""
PSV Disease Score Computation — per-class scores from extracted features.

Each score is a weighted combination of features that encode the pathological
hallmarks of that disease. Scores are contrastive: features that contribute
to one class subtract from competing classes.

Scores are NOT classification outputs — they are inputs to the MLP.
Return: Dict[class_name, float] normalized 0-1.

[2026-04-17 v1.7 update — Gap 1 audit fix]
The black_rot and alternaria score formulas were originally hand-engineered
based on biology heuristics. Section 3A (`scripts/apin/section3a_psv_*`)
ran supervised feature importance via L2-penalised logistic regression on
the binary black_rot-vs-alternaria task and found top-30 features that
achieve a +0.5133 mean-score separation (vs the −0.043 the hand formulas
achieved). The new supervised path is loaded from
`scripts/apin/section3a_psv_blackrot_alternaria_importance.json` and used
*in addition to* the hand formulas: the supervised LR score replaces the
original `br_raw` and `alt_raw` formulas. The hand-crafted formulas remain
for the remaining 7 classes (yvmv, powdery, cercospora, enation, downy,
two healthy) where no supervised analysis was run.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scripts.psv.config import PSV_CFG


# ════════════════════════════════════════════════════════════════════════════
# Supervised black_rot vs alternaria scoring — Section 3A LR coefficients
# ════════════════════════════════════════════════════════════════════════════
_SUPERVISED_BR_ALT_PATH = (
    Path(__file__).resolve().parent.parent / "apin"
    / "section3a_psv_blackrot_alternaria_importance.json"
)
_MULTICLASS_PATH = (
    Path(__file__).resolve().parent.parent / "apin"
    / "section3a_psv_multiclass_importance.json"
)
_SUPERVISED_CACHE: Optional[Tuple[List[str], np.ndarray]] = None
# Multi-class cache: per-class (feature_names, coefs) for all 9 classes
# (Gap 1 audit fix). Populated on first call by _load_multiclass_weights.
_MULTICLASS_CACHE: Optional[Dict[str, Tuple[List[str], np.ndarray]]] = None


def _load_multiclass_weights() -> Optional[Dict[str, Tuple[List[str], np.ndarray]]]:
    """Load Section 3A multi-class supervised LR weights for ALL 9 classes.

    Returns dict mapping class_name → (feature_names, coefs) for the best-k
    feature subset per class. Each class's separation_threshold check must
    have PASSed for it to be included.
    """
    global _MULTICLASS_CACHE
    if _MULTICLASS_CACHE is not None:
        return _MULTICLASS_CACHE
    if not _MULTICLASS_PATH.exists():
        return None
    try:
        with open(_MULTICLASS_PATH) as f:
            data = json.load(f)
        out: Dict[str, Tuple[List[str], np.ndarray]] = {}
        for cls_name, entry in data.get("per_class", {}).items():
            decision = entry.get("decision", {})
            if not decision.get("passes_threshold", False):
                continue
            best_k = decision["best_k"]
            test_block = entry["top_k_separation_test"][str(best_k)]
            names = list(test_block["feature_names_used"])
            coefs = np.asarray(test_block["feature_coefs_used"], dtype=np.float32)
            out[cls_name] = (names, coefs)
        _MULTICLASS_CACHE = out if out else None
        return _MULTICLASS_CACHE
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def _supervised_class_score(features: Dict[str, float],
                              cls_name: str) -> Optional[float]:
    """Compute the LR z-score for a specific class using its top-k
    supervised feature weights. Returns None if multi-class artifact
    missing or this class wasn't included (failed separation threshold)."""
    weights = _load_multiclass_weights()
    if weights is None or cls_name not in weights:
        return None
    names, coefs = weights[cls_name]
    vec = np.asarray([float(features.get(n, 0.0)) for n in names],
                       dtype=np.float32)
    return float(np.dot(coefs, vec))


def _load_supervised_br_alt_weights() -> Optional[Tuple[List[str], np.ndarray]]:
    """Load Section 3A's top-30 (feature_name, coef) pairs.

    Sign convention from Section 3A: positive coef → predicts brassica_black_rot,
    negative coef → predicts brassica_alternaria. (Verified against
    `top_15_features` ranks 1–15 — `E01_surface_roughness_score` has
    coef -0.717 and indeed alternaria has higher surface-roughness than
    black_rot per the LR fit.)
    """
    global _SUPERVISED_CACHE
    if _SUPERVISED_CACHE is not None:
        return _SUPERVISED_CACHE
    if not _SUPERVISED_BR_ALT_PATH.exists():
        return None
    try:
        with open(_SUPERVISED_BR_ALT_PATH) as f:
            data = json.load(f)
        # Use the top-30 set (best separation 0.5133 in Section 3A)
        top30 = data["top_k_separation_test"]["30"]
        names = list(top30["feature_names_used"])
        coefs = np.asarray(top30["feature_coefs_used"], dtype=np.float32)
        _SUPERVISED_CACHE = (names, coefs)
        return _SUPERVISED_CACHE
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def _supervised_blackrot_alternaria_score(
    features: Dict[str, float],
) -> Optional[Tuple[float, float]]:
    """Compute (br_score_raw, alt_score_raw) using the top-30 supervised LR
    weights from Section 3A. Returns None if weights file is unavailable.

    The LR returns a single signed score `z = sum(coef * feature)`. We map
    z to two raw scores: br_raw = z (positive z → black_rot), alt_raw = -z
    (negative z → alternaria). They are subsequently put through the same
    raw → [0,1] normalisation as all other classes (clip / max_raw).
    """
    weights = _load_supervised_br_alt_weights()
    if weights is None:
        return None
    names, coefs = weights
    # Build the feature vector; missing features default to 0.0 (consistent
    # with how other formulas use g(name, default=0.0))
    vec = np.asarray([float(features.get(n, 0.0)) for n in names],
                       dtype=np.float32)
    z = float(np.dot(coefs, vec))
    return z, -z


def compute_disease_scores(features: Dict[str, float],
                           calibration: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
    """
    Compute per-class disease scores from extracted features.

    Args:
        features: dict of feature_name -> value (from feature_extractor)
        calibration: optional percentile normalization params (from calibration.py)

    Returns:
        Dict of class_name -> score (0-1, normalized via sigmoid)
    """
    eps = PSV_CFG.EPSILON

    def g(name, default=0.0):
        """Get feature value with fallback."""
        return features.get(name, default)

    # Apply calibration if available (percentile normalization)
    if calibration:
        features = _calibrate_features(features, calibration)

    scores = {}

    # ════════════════════════════════════════════════════════════════
    # brassica_black_rot AND brassica_alternaria
    # [v1.7] Use Section 3A supervised LR top-30 weights when available.
    # Hand-engineered formulas had separation -0.043 on the failure pair;
    # supervised path achieves +0.5133 on the same val subset.
    # ════════════════════════════════════════════════════════════════
    sup_pair = _supervised_blackrot_alternaria_score(features)
    if sup_pair is not None:
        br_raw, alt_raw = sup_pair
    else:
        # Fallback to the original hand-engineered formulas if Section 3A
        # artifact is missing — preserves backward compatibility.
        br_raw = (
            3.0 * g('A16_vshape_score', 0) +
            2.5 * g('A03_margin_vs_interior_ratio') +
            2.5 * g('B01_vein_dark_colocalization') +
            2.0 * g('B02_vein_darkening_extent') +
            1.5 * g('A01_margin_disease_density') +
            1.5 * g('A17_disease_elongation_toward_midrib', 0) +
            1.0 * g('D02_yellow_marginal_fraction') +
            1.0 * g('A07_margin_connectivity') +
            0.8 * g('F07_blackrot_severity') +
            0.5 * g('A15_edge_originating_fraction') -
            2.0 * g('C01_mean_blob_circularity') -
            1.5 * g('C09_blob_interior_fraction')
        )
        alt_raw = (
            2.0 * g('C07_alternaria_ring_score') +
            1.5 * g('C01_mean_blob_circularity') +
            1.5 * g('C09_blob_interior_fraction') +
            1.0 * g('C08_yellow_halo_fraction') +
            0.3 * min(g('C05_blob_count_normalized'), 5.0) +
            0.8 * g('E06_lesion_edge_sharpness') +
            0.5 * g('C10_concentric_ring_count') -
            2.0 * g('A03_margin_vs_interior_ratio') -
            1.5 * g('B02_vein_darkening_extent')
        )

    # ════════════════════════════════════════════════════════════════
    # brassica_downy_mildew
    # Biology: Angular vein-bounded lesions, yellow patches, diffuse
    # ════════════════════════════════════════════════════════════════
    dm_raw = (
        2.0 * (1 - g('C01_mean_blob_circularity', 0.5)) +  # angular, not round
        1.5 * g('B05_vein_boundary_alignment') +
        1.5 * g('D09_chlorosis_fraction') +
        1.0 * g('D06_mosaic_color_variance') +
        0.8 * g('A12_disease_coverage_fraction') -
        1.5 * g('C01_mean_blob_circularity') -  # penalize round spots
        1.0 * g('E06_lesion_edge_sharpness')    # penalize sharp edges (alternaria)
    )

    # ════════════════════════════════════════════════════════════════
    # brassica_healthy
    # Biology: Uniform green, no lesions, smooth outline
    # ════════════════════════════════════════════════════════════════
    bh_raw = (
        2.5 * g('D07_green_retention_fraction') +
        1.5 * (1 - g('A12_disease_coverage_fraction')) +
        1.0 * max(0, 1 - g('C05_blob_count_normalized') * 0.05) +
        0.5 * (1 - g('E01_surface_roughness_score'))
    )

    # ════════════════════════════════════════════════════════════════
    # okra_yvmv (Yellow Vein Mosaic Virus)
    # Biology: Yellow vein network, mosaic pattern, green mesophyll
    # ════════════════════════════════════════════════════════════════
    yv_raw = (
        2.5 * g('D04_yellow_vein_fraction') +
        2.0 * g('D06_mosaic_color_variance') +
        1.5 * g('F03_yvmv_vs_healthy') +
        1.0 * g('D07_green_retention_fraction') +  # mesophyll preserved
        0.5 * g('D09_chlorosis_fraction') -
        1.5 * g('D01_gray_white_center_fraction') -  # penalize cercospora-like
        1.0 * g('A01_margin_disease_density')         # penalize margin (black_rot)
    )

    # ════════════════════════════════════════════════════════════════
    # okra_powdery_mildew
    # Biology: White powdery surface, uniform, high GLCM homogeneity
    # ════════════════════════════════════════════════════════════════
    pm_raw = (
        2.5 * g('D05_powdery_white_coverage') +
        2.0 * g('E02_glcm_homogeneity') +
        1.5 * g('D11_disease_hue_uniformity') +
        1.0 * g('F04_powdery_vs_cercospora') -
        1.5 * g('D01_gray_white_center_fraction') -  # penalize spot centers
        1.0 * g('C01_mean_blob_circularity')          # penalize discrete spots
    )

    # ════════════════════════════════════════════════════════════════
    # okra_cercospora
    # Biology: Gray-white centers, dark borders, circular spots in interior
    # ════════════════════════════════════════════════════════════════
    cc_raw = (
        2.5 * g('D01_gray_white_center_fraction') +
        2.0 * g('C06_cercospora_ring_score') +
        1.5 * g('C09_blob_interior_fraction') +
        1.0 * g('C01_mean_blob_circularity') +
        0.8 * (1 - g('C04_blob_size_cv')) +  # regular spot sizes
        0.5 * g('C11_hole_count_normalized') -
        2.0 * g('A03_margin_vs_interior_ratio') -  # penalize margin (black_rot)
        1.0 * g('D05_powdery_white_coverage')       # penalize uniform white
    )

    # ════════════════════════════════════════════════════════════════
    # okra_enation
    # Biology: Surface bumps, leaf curl, rough texture, shape distortion
    # ════════════════════════════════════════════════════════════════
    en_raw = (
        2.0 * g('E01_surface_roughness_score') +
        2.0 * g('E07_leaf_contour_irregularity') +
        1.5 * g('F05_enation_vs_healthy') +
        1.5 * g('E05_local_variance_mean') +
        1.0 * g('D07_green_retention_fraction') -  # color relatively normal
        1.5 * g('D01_gray_white_center_fraction') -
        1.0 * g('A12_disease_coverage_fraction')
    )

    # ════════════════════════════════════════════════════════════════
    # okra_healthy
    # Biology: Uniform green, smooth, no lesions
    # ════════════════════════════════════════════════════════════════
    oh_raw = (
        2.5 * g('D07_green_retention_fraction') +
        1.5 * (1 - g('A12_disease_coverage_fraction')) +
        1.0 * max(0, 1 - g('C05_blob_count_normalized') * 0.05) +
        0.5 * (1 - g('E07_leaf_contour_irregularity')) +
        0.5 * (1 - g('E01_surface_roughness_score'))
    )

    # ════════════════════════════════════════════════════════════════
    # Normalize all scores via sigmoid to [0, 1]
    # [FIX] Per-class denominators instead of single 5.0
    # Each class has different raw score range due to different
    # numbers of contributing features and penalty terms.
    # ════════════════════════════════════════════════════════════════
    raw_scores = {
        'okra_yvmv': yv_raw,
        'okra_powdery_mildew': pm_raw,
        'okra_cercospora': cc_raw,
        'okra_enation': en_raw,
        'okra_healthy': oh_raw,
        'brassica_black_rot': br_raw,
        'brassica_downy_mildew': dm_raw,
        'brassica_alternaria': alt_raw,
        'brassica_healthy': bh_raw,
    }

    # [Gap 1 audit fix] Multi-class supervised override for ALL classes.
    # When the multi-class LR artifact is present and a class PASSED its
    # separation threshold (Section 3A multi-class), use its LR z-score
    # passed through sigmoid (= LR class probability) INSTEAD of the
    # hand-crafted formula's raw score. Falls back per-class to the
    # binary BR/ALT supervised path (legacy) and finally to the hand
    # formula clip(raw/12) if no supervised weights exist.
    max_raw = 12.0  # empirical: typical max for hand-crafted formulas

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    using_binary_pair = sup_pair is not None
    multiclass_weights = _load_multiclass_weights()
    multiclass_available = multiclass_weights is not None

    for cls, raw in raw_scores.items():
        # 1. Multi-class supervised path (preferred)
        if multiclass_available and cls in multiclass_weights:
            z = _supervised_class_score(features, cls)
            if z is not None:
                scores[cls] = float(_sigmoid(z))
                continue
        # 2. Legacy binary BR/ALT supervised pair
        if using_binary_pair and cls in ("brassica_black_rot", "brassica_alternaria"):
            scores[cls] = float(_sigmoid(raw))
            continue
        # 3. Hand-engineered formula fallback
        scores[cls] = float(np.clip(raw / max_raw, 0.0, 1.0))

    return scores


def _calibrate_features(features: Dict[str, float],
                        calibration: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Apply percentile normalization from calibration data.
    Transforms each feature to its percentile rank in the training distribution.
    """
    calibrated = {}
    for name, val in features.items():
        if name in calibration:
            cal = calibration[name]
            p5 = cal.get('p5', 0)
            p95 = cal.get('p95', 1)
            if p95 > p5:
                calibrated[name] = np.clip((val - p5) / (p95 - p5), 0, 1)
            else:
                calibrated[name] = 0.5
        else:
            calibrated[name] = val
    return calibrated
