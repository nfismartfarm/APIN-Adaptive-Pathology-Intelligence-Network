"""
PSV Configuration — ALL tunable parameters.
Zero hardcoded thresholds anywhere else in the codebase.

Two types of parameters:
  1. Architecture parameters (this file) — human-tunable design choices
  2. Calibration parameters (psv_calibration.json) — data-fitted normalization

This file contains ONLY architecture parameters.
Calibration parameters are fitted by calibration.py and saved separately.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os


@dataclass
class PSVConfig:
    """All PSV tunable parameters. Modify here, not in feature code."""

    # ═══════════════════════════════════════════════════════════════════
    # CLASS DEFINITIONS (must match Model 2 config exactly)
    # ═══════════════════════════════════════════════════════════════════
    CLASS_NAMES: List[str] = field(default_factory=lambda: [
        'okra_yvmv', 'okra_powdery_mildew', 'okra_cercospora', 'okra_enation',
        'okra_healthy', 'brassica_black_rot', 'brassica_downy_mildew',
        'brassica_alternaria', 'brassica_healthy',
    ])
    NUM_CLASSES: int = 9
    OKRA_INDICES: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    BRASSICA_INDICES: List[int] = field(default_factory=lambda: [5, 6, 7, 8])

    # Known failure classes (adversarial augmentation targets)
    FAILURE_CLASSES: List[str] = field(default_factory=lambda: [
        'brassica_black_rot', 'okra_cercospora',
    ])

    # ═══════════════════════════════════════════════════════════════════
    # IMAGE PREPROCESSING
    # ═══════════════════════════════════════════════════════════════════
    # Resize for consistent processing (features are scale-invariant via normalization)
    PROCESSING_SIZE: int = 384  # [FIX Perf] reduced from 512 for speed
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)

    # ═══════════════════════════════════════════════════════════════════
    # LEAF SEGMENTATION (pure CV, no neural network)
    # ═══════════════════════════════════════════════════════════════════
    # In LAB space: A channel <0 = green, >0 = red/magenta
    LEAF_A_MIN: float = -40.0   # minimum A value for leaf (very green)
    LEAF_A_MAX: float = 8.0     # maximum A value (slightly greenish still leaf)
    LEAF_B_MIN: float = -10.0   # minimum B value
    LEAF_B_MAX: float = 50.0    # maximum B value
    LEAF_L_MIN: float = 20.0    # minimum lightness (exclude very dark)
    LEAF_L_MAX: float = 95.0    # maximum lightness (exclude specular highlights)
    LEAF_MORPH_KERNEL: int = 15  # morphological opening/closing kernel size
    LEAF_MIN_AREA_FRACTION: float = 0.05  # minimum leaf area as fraction of image

    # ═══════════════════════════════════════════════════════════════════
    # DISEASE PIXEL DETECTION
    # ═══════════════════════════════════════════════════════════════════
    # Healthy green in LAB: A in [-15, 5], B in [-5, 25], L in [35, 80]
    HEALTHY_A_MIN: float = -18.0
    HEALTHY_A_MAX: float = 12.0   # [FIX] was 5.0, too narrow for okra green
    HEALTHY_B_MIN: float = -8.0
    HEALTHY_B_MAX: float = 35.0   # [FIX] was 30.0, okra has wider B range
    HEALTHY_L_MIN: float = 30.0   # [FIX] was 35.0, slightly too restrictive
    HEALTHY_L_MAX: float = 85.0
    DISEASE_MIN_PIXELS: int = 50  # minimum disease pixels to consider

    # ═══════════════════════════════════════════════════════════════════
    # SPATIAL ZONES (Group A parameters)
    # ═══════════════════════════════════════════════════════════════════
    MARGIN_ZONE_THRESHOLD: float = 0.15    # distance_map < this = margin zone
    INTERIOR_ZONE_THRESHOLD: float = 0.40  # distance_map > this = interior zone
    APEX_ZONE_FRACTION: float = 0.10       # top 10% along leaf axis = apex
    BASE_ZONE_FRACTION: float = 0.10       # bottom 10% = base
    MIDRIB_WIDTH_FACTOR: float = 2.0       # vein width multiplier for midrib zone

    # ═══════════════════════════════════════════════════════════════════
    # VEIN DETECTION (Group B parameters)
    # ═══════════════════════════════════════════════════════════════════
    # Frangi filter for vein-like structures
    FRANGI_SIGMAS: Tuple[float, ...] = (2.0, 4.0)  # [FIX Perf] reduced from 4 to 2 sigmas
    FRANGI_ALPHA: float = 0.5
    FRANGI_BETA: float = 0.5
    FRANGI_BLACK_RIDGES: bool = True  # dark veins on light background
    VEIN_THRESHOLD: float = 0.1      # frangi response > this = vein pixel
    DARK_PIXEL_L_MAX: float = 40.0   # L channel < this = dark pixel

    # Black top-hat for vein detection (alternative/supplementary)
    TOPHAT_KERNEL_LENGTH: int = 15   # elongated kernel for vein morphology
    TOPHAT_DARK_THRESHOLD: int = 20  # minimum top-hat response

    # ═══════════════════════════════════════════════════════════════════
    # BLOB DETECTION (Group C parameters)
    # ═══════════════════════════════════════════════════════════════════
    BLOB_MIN_SIGMA: float = 3.0
    BLOB_MAX_SIGMA: float = 30.0
    BLOB_NUM_SIGMA: int = 8
    BLOB_THRESHOLD: float = 0.05     # detection sensitivity
    BLOB_OVERLAP: float = 0.5
    BLOB_CENTER_FRACTION: float = 0.3  # inner 30% of blob = center
    BLOB_BORDER_FRACTION: float = 0.3  # outer 30% ring = border
    BLOB_MIN_AREA: int = 20           # minimum blob area in pixels
    YELLOW_HALO_DILATION: int = 10    # pixels to dilate blob for yellow halo check

    # ═══════════════════════════════════════════════════════════════════
    # COLOR ZONE THRESHOLDS (Group D parameters)
    # ═══════════════════════════════════════════════════════════════════
    # Gray-white (dead desiccated tissue — cercospora centers)
    GRAY_WHITE_L_MIN: float = 65.0
    GRAY_WHITE_A_ABS_MAX: float = 12.0
    GRAY_WHITE_B_ABS_MAX: float = 12.0

    # Yellow (chlorosis, halos)
    YELLOW_A_MAX: float = 5.0
    YELLOW_B_MIN: float = 20.0
    YELLOW_L_MIN: float = 50.0

    # Brown (necrosis on veins)
    BROWN_A_MIN: float = 5.0
    BROWN_B_MIN: float = 10.0
    BROWN_L_MIN: float = 20.0
    BROWN_L_MAX: float = 55.0

    # Powdery white (mildew surface conidia)
    POWDERY_L_MIN: float = 75.0
    POWDERY_A_ABS_MAX: float = 10.0
    POWDERY_B_ABS_MAX: float = 10.0

    # Necrosis (severe, very dark brown)
    NECROSIS_L_MAX: float = 35.0
    NECROSIS_A_MIN: float = 5.0
    NECROSIS_B_MIN: float = 5.0

    # Chlorosis (early yellowing)
    CHLOROSIS_A_MAX: float = 0.0
    CHLOROSIS_B_MIN: float = 15.0
    CHLOROSIS_L_MIN: float = 55.0

    # ═══════════════════════════════════════════════════════════════════
    # TEXTURE (Group E parameters)
    # ═══════════════════════════════════════════════════════════════════
    GLCM_DISTANCES: List[int] = field(default_factory=lambda: [1, 3])
    GLCM_ANGLES: List[float] = field(default_factory=lambda: [0.0, 1.5708])  # 0, pi/2
    GLCM_LEVELS: int = 64  # quantize to 64 levels for speed
    LOCAL_VARIANCE_KERNEL: int = 9  # kernel for local variance map
    SOBEL_THRESHOLD: float = 0.05  # minimum edge strength

    # ═══════════════════════════════════════════════════════════════════
    # SPECULAR REFLECTION
    # ═══════════════════════════════════════════════════════════════════
    SPECULAR_L_MIN: float = 92.0
    SPECULAR_A_ABS_MAX: float = 8.0
    SPECULAR_B_ABS_MAX: float = 8.0
    SPECULAR_CRITICAL_FRACTION: float = 0.08  # if >8% specular, flag and mask

    # ═══════════════════════════════════════════════════════════════════
    # IMAGE QUALITY ASSESSMENT THRESHOLDS
    # ═══════════════════════════════════════════════════════════════════
    # Blur
    BLUR_SEVERE_THRESHOLD: float = 80.0    # Laplacian variance < this = severe
    BLUR_MILD_THRESHOLD: float = 200.0     # < this = mild blur

    # Exposure
    UNDEREXPOSED_SEVERE_L: float = 20.0
    UNDEREXPOSED_MILD_L: float = 35.0
    OVEREXPOSED_SEVERE_L: float = 92.0
    OVEREXPOSED_MILD_L: float = 85.0

    # Leaf detection
    NO_LEAF_FRACTION: float = 0.10    # leaf < this = no leaf
    PARTIAL_LEAF_FRACTION: float = 0.25
    EXTREME_ROTATION_DEGREES: float = 60.0

    # Disease visibility
    MIN_DISEASE_PIXELS: int = 50
    MAX_DISEASE_COVERAGE: float = 0.85  # > this = entire leaf necrotic

    # ═══════════════════════════════════════════════════════════════════
    # DISEASE SCORE WEIGHTS (per-class feature contributions)
    # These are the INITIAL weights. Calibration adjusts them.
    # ═══════════════════════════════════════════════════════════════════
    # Format: feature_name -> weight (positive = contributes, negative = penalizes)
    # Defined per class in disease_scores.py, loaded from config

    # ═══════════════════════════════════════════════════════════════════
    # MLP ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════
    MLP_INPUT_DIM: int = 36   # 9 (Model2) + 9 (EfficientNet) + 9 (PSV) + 9 (DINOv2)
    MLP_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [64, 32])  # [FIX] spec says 36->64->32->9
    MLP_DROPOUT: float = 0.35
    MLP_OUTPUT_DIM: int = 9

    # Training
    MLP_LR: float = 1e-3
    MLP_WEIGHT_DECAY: float = 1e-4
    MLP_EPOCHS: int = 100
    MLP_BATCH_SIZE: int = 64
    MLP_EARLY_STOP_PATIENCE: int = 10
    MLP_EARLY_STOP_DELTA: float = 0.001
    MLP_K_FOLDS: int = 5

    # ═══════════════════════════════════════════════════════════════════
    # SAMPLE WEIGHTS FOR MLP TRAINING
    # ═══════════════════════════════════════════════════════════════════
    FIELD_PHOTO_WEIGHT: float = 5.0
    DIVERSE_LAB_WEIGHT: float = 2.0
    RECOMPOSED_WEIGHT: float = 1.5
    STANDARD_LAB_WEIGHT: float = 1.0
    ADVERSARIAL_WEIGHT: float = 6.0

    # ═══════════════════════════════════════════════════════════════════
    # ADVERSARIAL AUGMENTATION
    # ═══════════════════════════════════════════════════════════════════
    # Known Model 2 failure patterns (observed on real-world photos)
    ADVERSARIAL_PATTERNS: Dict[str, List[float]] = field(default_factory=lambda: {
        'brassica_black_rot': [
            # Model 2 outputs: alternaria 77%, black_rot 8%, rest distributed
            0.0, 0.0, 0.0, 0.0, 0.0,  # okra classes = 0
            0.08,  # black_rot (true class, but model gives low)
            0.04,  # downy_mildew
            0.77,  # alternaria (model's wrong answer)
            0.11,  # brassica_healthy
        ],
        'okra_cercospora': [
            # Model 2: okra_healthy 82%, cercospora 5%
            0.03, 0.02, 0.05, 0.03, 0.82,  # healthy dominates
            0.0, 0.0, 0.0, 0.05,  # brassica = low
        ],
    })

    # ═══════════════════════════════════════════════════════════════════
    # ONLINE LEARNING
    # ═══════════════════════════════════════════════════════════════════
    MIN_FEEDBACK_FOR_RETRAIN: int = 10
    MIN_FAILURE_CLASS_FEEDBACK: int = 3
    RETRAIN_REGRESSION_TOLERANCE: float = 0.005
    RECENCY_WEIGHT_7DAYS: float = 5.0
    RECENCY_WEIGHT_30DAYS: float = 3.0
    RECENCY_WEIGHT_OLDER: float = 1.5

    # ═══════════════════════════════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════════════════════════════
    ROOT: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    CALIBRATION_PATH: str = 'psv_calibration.json'
    FEATURES_CACHE_PATH: str = 'psv_features_cache.pkl'
    MODEL_PREDICTIONS_CACHE_PATH: str = 'model_predictions_cache.pkl'
    FEEDBACK_BUFFER_PATH: str = 'psv_feedback_buffer.json'
    MLP_CHECKPOINT_PATH: str = 'psv_mlp_best.pt'
    DEBUG_DIR: str = 'psv_debug'

    # ═══════════════════════════════════════════════════════════════════
    # PERFORMANCE TARGETS
    # ═══════════════════════════════════════════════════════════════════
    MAX_PSV_TIME_MS: float = 150.0  # target: <150ms per image on CPU
    MIN_BLACK_ROT_F1: float = 0.70
    MIN_CERCOSPORA_F1: float = 0.65
    MIN_MACRO_F1: float = 0.80

    EPSILON: float = 1e-8  # numerical stability constant


# Global config instance
PSV_CFG = PSVConfig()
