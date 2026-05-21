"""
DINOv2 Linear Probe Experiment — Configuration

All paths and hyperparameters. No hardcoded values in any other file.

Research-informed implementation choices:
  - Feature aggregation: DINOv2 paper recommends CLS + mean_patch concat (768d)
    for linear evaluation. We test CLS (384d), mean_patch (384d), and cls_mean
    (768d) and compare. Register tokens at indices 1-4 are EXCLUDED from mean.
    Source: DINOv2 paper (Oquab et al., 2024, TMLR), Section 4.1
  - Normalization: DINOv2 evaluation uses L2 normalization (unit sphere), not
    StandardScaler. ViT features have varying norms; L2 makes cosine similarity
    equivalent to dot product in the linear head.
    Source: DINOv2 model card, facebookresearch/dinov2 GitHub
  - Solver: lbfgs with L2 regularization, max_iter=2000.
    lbfgs handles multinomial logistic regression efficiently for n_features < n_samples.
    Source: sklearn documentation, standard practice for ViT linear probes
  - C values: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] — DINOv2 features are
    well-separated, so higher C (less regularization) often works best.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Model 2 data
MODEL2_CSV = PROJECT_ROOT / 'data' / 'specialist' / 'model2' / 'model2_unified_source_map.csv'
SPLIT_INDICES = PROJECT_ROOT / 'data' / 'specialist' / 'model2' / 'split_indices.json'

# Router checkpoint (head-only, 8.6KB — backbone is timm pretrained)
ROUTER_CHECKPOINT = PROJECT_ROOT / 'models' / 'router' / 'router_best.pt'

# CLAHE images root
CLAHE_IMAGES_ROOT = PROJECT_ROOT / 'data' / 'specialist' / 'model2' / 'cleaned_clahe'

# Output paths
RESULTS_DIR = Path(__file__).parent / 'results'
FEATURES_CACHE_PATH = RESULTS_DIR / 'dinov2_features_cache.pkl'
CACHE_FINGERPRINT_PATH = RESULTS_DIR / 'feature_cache_fingerprint.json'
EXTRACTION_FAILURES_PATH = RESULTS_DIR / 'feature_extraction_failures.txt'


# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

# CRITICAL: This is the exact same backbone used in the router.
# The router trained only a linear head on top of this frozen backbone.
# Using timm pretrained directly = using the router backbone.
TIMM_MODEL_NAME = 'vit_small_patch14_reg4_dinov2.lvd142m'

FEATURE_DIM_CLS = 384       # CLS token dimension
FEATURE_DIM_MEAN_PATCH = 384  # Mean patch token dimension (same as CLS)
FEATURE_DIM_CLS_MEAN = 768  # CLS + mean_patch concatenated

# Helper: maps aggregation strategy -> feature dimension
FEATURE_DIM = {
    'cls': 384,
    'mean_patch': 384,
    'cls_mean': 768,
}

# Feature aggregation strategy — tested options:
#   'cls'       : CLS token only (384d) — simplest, fastest
#   'mean_patch': mean of patch tokens excluding CLS and registers (384d)
#   'cls_mean'  : CLS + mean_patch concatenated (768d) — DINOv2 paper recommended
FEATURE_AGGREGATION = 'cls_mean'  # Default: paper's recommended protocol

# Number of prefix tokens to skip when computing mean_patch
# DINOv2-Small-Registers: 1 CLS + 4 register tokens = 5 prefix tokens
NUM_PREFIX_TOKENS = 5

BATCH_SIZE = 64
# CRITICAL: Must pass img_size to timm.create_model().
# Default is 518px which changes token count from 261 to 1374, silently
# producing wrong-dimension features. Always use: timm.create_model(..., img_size=IMG_SIZE)
IMG_SIZE = 224  # DINOv2 standard evaluation size

# Number of DataLoader workers
# NOTE: On Windows, >0 workers can cause multiprocessing issues.
# Set to 0 for safety, increase if on Linux.
NUM_WORKERS = 0


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════
# CRITICAL: This must match the router training preprocessing exactly.
# If this differs, features will be on a different distribution.
#
# Router training (scripts/train_router_simple.py) uses:
#   get_eval_transform(img_size=224) from train_utils.py which does:
#     A.Resize(224, 224)
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#     ToTensorV2()
#
# For feature caching, we use the equivalent torchvision transforms
# since we process with torch DataLoader, not albumentations:
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Transform pipeline: Resize(224) -> ToTensor -> Normalize(ImageNet)
# NOTE: Images loaded from clahe_path are ALREADY LAB-CLAHE processed.
# Do NOT apply CLAHE again. The preprocessing here is just resize + normalize.


# ═══════════════════════════════════════════════════════════════════════
# CLASS MAPPING (must match Model 2 config exactly)
# ═══════════════════════════════════════════════════════════════════════

CLASS_NAMES: List[str] = [
    'okra_yvmv',              # 0
    'okra_powdery_mildew',    # 1
    'okra_cercospora',        # 2
    'okra_enation',           # 3
    'okra_healthy',           # 4
    'brassica_black_rot',     # 5
    'brassica_downy_mildew',  # 6
    'brassica_alternaria',    # 7
    'brassica_healthy',       # 8
]
NUM_CLASSES = len(CLASS_NAMES)  # 9

CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

CROP_TO_CLASSES: Dict[str, List[str]] = {
    'okra': ['okra_yvmv', 'okra_powdery_mildew', 'okra_cercospora',
             'okra_enation', 'okra_healthy'],
    'brassica': ['brassica_black_rot', 'brassica_downy_mildew',
                 'brassica_alternaria', 'brassica_healthy'],
}

# Classes with known domain shift problems (real-world failure)
FAILURE_CLASSES: List[str] = ['brassica_black_rot', 'okra_cercospora']

# Classes with limited training data diversity
THIN_CLASSES: List[str] = [
    'brassica_black_rot',      # 94% from original_pool
    'okra_cercospora',         # 89% from original_pool
    'brassica_downy_mildew',   # thin: only 338 total
    'okra_enation',            # thin: only 288 total
]

# Split names as they appear in split_indices.json
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val_and_soup'
FINAL_VAL_SPLIT = 'final_val'
CONFORMAL_SPLIT = 'conformal'


# ═══════════════════════════════════════════════════════════════════════
# LINEAR PROBE HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════

# Regularization strengths to grid-search
C_VALUES: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Solver for sklearn LogisticRegression
# lbfgs: supports L2 regularization, handles multinomial, efficient for d<n
SOLVER = 'lbfgs'
MAX_ITER = 2000

# Feature normalization strategies to test
# 'standard': StandardScaler (zero mean, unit variance per feature)
# 'l2_norm':  L2 normalize each sample to unit sphere (DINOv2 paper protocol)
# 'none':     raw features (baseline)
SCALER_TYPES: List[str] = ['l2_norm', 'standard', 'none']

# Cross-validation folds for hyperparameter search (on training data only)
CV_FOLDS = 5


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════

# Minimum samples to report field-only or per-source metrics
FIELD_PHOTO_MIN_SAMPLES = 5
SOURCE_MIN_SAMPLES = 3

# OOD detection
OOD_DISTANCE_PERCENTILE = 95  # Mahalanobis threshold calibration percentile

# Reference Model 2 F1 scores (from PHASE0_LOG.md LOG ENTRY 036, val_and_soup split)
MODEL2_VAL_F1 = {
    'macro': 0.9443,
    'okra_yvmv': 0.9665,
    'okra_powdery_mildew': 0.9282,
    'okra_cercospora': 0.9423,
    'okra_enation': 0.9318,
    'okra_healthy': 0.9752,
    'brassica_black_rot': 0.9630,
    'brassica_downy_mildew': 0.8909,
    'brassica_alternaria': 0.9384,
    'brassica_healthy': 0.9623,
}


# ═══════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
