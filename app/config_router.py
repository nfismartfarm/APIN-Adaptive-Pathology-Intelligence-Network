"""
Crop Router Config: 4-class crop classifier (okra / brassica / tomato / chilli)

Role in the 3-model pipeline:
  Router (this model, DINOv2-Small-with-Registers, frozen backbone) → decides which specialist to invoke
     -> if crop == okra or brassica: forward to Model 2 specialist
     -> if crop == tomato or chilli: forward to Model 3 specialist

The router does NOT predict diseases. It only predicts the crop identity so
that the correct downstream specialist can be called. Every training image
from Models 2 and 3 is also a valid router training image (the leaf shows
the crop regardless of what disease is present).

Data source of truth: data/specialist/router/router_unified_source_map.csv
Training image folder: data/specialist/router/cleaned/{crop}/
"""

import os

# ── CLASS DEFINITIONS ──────────────────────────────────────────────────────
# 4 crops, densely indexed 0-3. Order matches Models 2 and 3 conventions.
CLASS_NAMES = ['okra', 'brassica', 'tomato', 'chilli']
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: n for i, n in enumerate(CLASS_NAMES)}

# Routing table: which specialist model handles which crop
CROP_TO_SPECIALIST = {
    'okra':     'model2',
    'brassica': 'model2',
    'tomato':   'model3',
    'chilli':   'model3',
}

# ── TRAINING STRATEGY — IMBALANCE HANDLING ─────────────────────────────────
# Current router pool counts (from router_unified_source_map.csv):
#   okra:      7,366
#   brassica:  4,269
#   tomato:   24,055 (90% lab — dominated by scidb_data_merged)
#   chilli:    9,468
# Imbalance ratio: 5.6:1 (tomato:brassica)
#
# Strategy: WeightedRandomSampler with per-crop caps PLUS tomato undersampling
# for the lab-heavy scidb_data_merged source. This prevents the model from
# defaulting to "tomato" when confidence is low, and reduces lab-background
# shortcut learning.

SAMPLER_STRATEGY = 'weighted_with_tomato_undersample'
UNDERSAMPLE_TOMATO_TO = 7300         # per epoch cap for tomato (matches okra pool size)
TOMATO_UNDERSAMPLE_SOURCE = 'scidb_data_merged'  # which source gets capped first
SAMPLER_MAX_WEIGHT_RATIO = 5.0       # prevent extreme thin-class oversampling

# ── FIELD PHOTO EMPHASIS ───────────────────────────────────────────────────
# Router needs to learn field-condition crop morphology, not lab artifacts.
# Current field photo ratios:
#   okra:     45.9% field
#   brassica: 92.3% field
#   tomato:   10.7% field  <-- CRITICAL — dominant source bias
#   chilli:   97.1% field
#
# Strategy: during training, up-weight field photos by FIELD_PHOTO_WEIGHT_MULTIPLIER
# so the sampler preferentially draws them. Lab images still contribute but
# at lower frequency.
FIELD_PHOTO_WEIGHT_MULTIPLIER = 5.0  # field images sampled 5x more often than lab (per architecture_convo.md L612)

# ── IMAGE / TRAINING HYPERPARAMETERS ───────────────────────────────────────
# DINOv2-Small-Registers (frozen backbone, linear head only — fast inference).
IMG_SIZE        = (224, 224)         # DINOv2-Small native (matches DINOV2_IMG_SIZE)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
# DINOv2-Small with Registers (verified in timm 1.0.26, Phase 0 Step 0.4)
# Backbone is FROZEN — only linear head (384 -> 4) trains
BACKBONE_NAME   = 'vit_small_patch14_reg4_dinov2.lvd142m'
BACKBONE_FREEZE = True
DINOV2_IMG_SIZE = 224           # MUST pass to timm.create_model (default 518 crashes)
DINOV2_EMBED_DIM = 384
DROPOUT_P       = 0.2
LABEL_SMOOTHING = 0.1
RANDOM_SEED     = 42

BATCH_SIZE              = 64
NUM_EPOCHS              = 20   # per MASTER_PLAN Section 4.2
BASE_LR                 = 1e-3
WEIGHT_DECAY            = 1e-2  # per MASTER_PLAN Section 4.2 (applied to head only, not bias)
GRAD_CLIP_NORM          = 1.0
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# ── INFERENCE — CROP CONFIDENCE GATE ───────────────────────────────────────
# Production inference uses crop confidence to decide whether to trust the
# specialist's disease prediction. Below this threshold, return OOD warning.
CROP_CONFIDENCE_MIN = 0.60        # < this -> flag as OOD / ambiguous

# Per-crop confidence floors (some crops are harder than others)
PER_CROP_MIN_CONFIDENCE = {
    'okra':     0.55,
    'brassica': 0.55,
    'tomato':   0.60,
    'chilli':   0.55,
}

# ── MONITORING TARGETS ─────────────────────────────────────────────────────
# Expected test accuracy per crop (based on imbalance + data quality)
TARGET_CROP_ACC = {
    'okra':     0.90,
    'brassica': 0.92,   # high field % supports high target
    'tomato':   0.85,   # domain gap penalty
    'chilli':   0.92,   # high field % supports high target
}
TARGET_OVERALL_MACRO_F1 = 0.88

# ── FILE PATHS ─────────────────────────────────────────────────────────────
ROOT              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTER_CLEANED    = os.path.join(ROOT, 'data', 'specialist', 'router', 'cleaned')
ROUTER_CSV        = os.path.join(ROOT, 'data', 'specialist', 'router',
                                 'router_unified_source_map.csv')
ROUTER_CHECKPOINTS = os.path.join(ROOT, 'models', 'router')

# ── VALIDATION HELPERS ─────────────────────────────────────────────────────
def assert_config_consistency():
    assert len(CLASS_NAMES) == 4
    assert NUM_CLASSES == 4
    assert CLASS_NAMES == ['okra', 'brassica', 'tomato', 'chilli']
    for cls, idx in CLASS_TO_IDX.items():
        assert IDX_TO_CLASS[idx] == cls
    for crop, specialist in CROP_TO_SPECIALIST.items():
        assert crop in CLASS_NAMES
        assert specialist in ('model2', 'model3')
    assert UNDERSAMPLE_TOMATO_TO > 0
    assert 0 < CROP_CONFIDENCE_MIN < 1
    return True


if __name__ == '__main__':
    assert_config_consistency()
    print('Router config consistency check PASSED')
    print(f'NUM_CLASSES = {NUM_CLASSES}')
    print(f'Crops: {CLASS_NAMES}')
    print(f'Routing table:')
    for crop, specialist in CROP_TO_SPECIALIST.items():
        print(f'  {crop:<10} -> {specialist}')
    print(f'Tomato undersample cap: {UNDERSAMPLE_TOMATO_TO}')
    print(f'Field photo weight multiplier: {FIELD_PHOTO_WEIGHT_MULTIPLIER}x')
