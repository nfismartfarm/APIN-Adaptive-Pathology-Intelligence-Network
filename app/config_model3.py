"""
Model 3 Specialist Config: Tomato + Chilli (10 classes)

IMPORTANT:
- This file is SEPARATE from app/config.py (23-class Swin production model).
- tomato_target_spot has been PERMANENTLY DROPPED from this specialist model.
  Reason: label reliability suspect (visual inspection found class 5 annotations
  were dark spots identical to black_spot, no concentric ring pattern), 539 images
  80% PlantVillage, 97% lab. Training on these labels actively degrades macro F1.
  Quarantined at data/specialist/model3/cleaned/tomato_target_spot_QUARANTINED/
- Model 3 uses a narrower tomato class set than the Swin model:
    Swin has: bacterial_spot, early_blight, late_blight, leaf_mold, septoria_leaf_spot,
              target_spot, mosaic_virus, yellow_leaf_curl_virus, healthy (9 classes)
    Model 3 has: foliar_spot (unified), late_blight, septoria_leaf_spot,
                 yellow_leaf_curl_virus, mosaic_virus, healthy (6 tomato classes)
  The Model 3 schema MERGES bacterial_spot + early_blight + leaf_mold into
  `tomato_foliar_spot` because (1) visually confusable, (2) inconsistent labelling
  across sources, (3) farmer treatment action is identical (fungicide + sanitation).

Data source of truth: data/specialist/model3/model3_unified_source_map.csv
Destination folder:   data/specialist/model3/cleaned/{class_name}/
"""

import os

# ── CLASS DEFINITIONS ──────────────────────────────────────────────────────
# 10 classes, densely indexed 0-9.
# tomato_target_spot is EXCLUDED (quarantined).
CLASS_NAMES = [
    # Tomato (indices 0-5)
    'tomato_foliar_spot',                 # 0 — merged bacterial + early blight + leaf mold
    'tomato_late_blight',                 # 1
    'tomato_septoria_leaf_spot',          # 2
    'tomato_yellow_leaf_curl_virus',      # 3
    'tomato_mosaic_virus',                # 4
    'tomato_healthy',                     # 5
    # Chilli (indices 6-9)
    'chilli_leaf_curl',                   # 6
    'chilli_healthy',                     # 7
    'chilli_cercospora_leaf_spot',        # 8
    'chilli_anthracnose',                 # 9 — thinnest chilli class
]
NUM_CLASSES  = len(CLASS_NAMES)           # 10
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: n for i, n in enumerate(CLASS_NAMES)}

# Crop indices within Model 3 (tomato=0, chilli=1)
NUM_CROPS        = 2
CROP_NAMES       = ['tomato', 'chilli']
TOMATO_INDICES   = [0, 1, 2, 3, 4, 5]
CHILLI_INDICES   = [6, 7, 8, 9]
HEALTHY_INDICES  = [5, 7]                 # tomato_healthy=5, chilli_healthy=7

CROP_FROM_IDX = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,   # tomato -> crop 0
    6: 1, 7: 1, 8: 1, 9: 1,                # chilli -> crop 1
}

CROP_TO_DISEASE_INDICES = {
    0: [0, 1, 2, 3, 4, 5],     # tomato diseases
    1: [6, 7, 8, 9],           # chilli diseases
}

HEALTHY_CLASSES = {'tomato_healthy', 'chilli_healthy'}

# ── EXCLUSION RECORD ───────────────────────────────────────────────────────
EXCLUDED_CLASSES = {'tomato_target_spot'}
EXCLUSION_REASON = (
    'Label reliability suspect: visual inspection of Tomato Leaf Multiclass '
    'dataset showed class 5 (target_spot) annotations were dark spots (37x35px avg) '
    'visually identical to black_spot, with no concentric ring pattern. '
    '539 images, 80% PlantVillage, 97% lab, 2 sources only. '
    'Quarantined at data/specialist/model3/cleaned/tomato_target_spot_QUARANTINED/'
)

# ── LABEL HARMONISATION ────────────────────────────────────────────────────
# target_spot mappings REMOVED — any row labelled as target_spot will fail
# label resolution and must be either dropped or reclassified during data prep.
LABEL_MAP = {
    # TOMATO FOLIAR SPOT (merged class)
    'bacterial_spot':               'tomato_foliar_spot',
    'tomato_bacterial_spot':        'tomato_foliar_spot',
    'early_blight':                 'tomato_foliar_spot',
    'tomato_early_blight':          'tomato_foliar_spot',
    'leaf_mold':                    'tomato_foliar_spot',
    'tomato_leaf_mold':             'tomato_foliar_spot',
    'tomato_foliar_spot':           'tomato_foliar_spot',
    # TOMATO LATE BLIGHT
    'late_blight':                  'tomato_late_blight',
    'tomato_late_blight':           'tomato_late_blight',
    # TOMATO SEPTORIA
    'septoria_leaf_spot':           'tomato_septoria_leaf_spot',
    'tomato_septoria_leaf_spot':    'tomato_septoria_leaf_spot',
    'gray_spot':                    'tomato_septoria_leaf_spot',   # Taiwan 'Gray spot'
    # TOMATO YELLOW LEAF CURL VIRUS
    'yellow_leaf_curl':             'tomato_yellow_leaf_curl_virus',
    'tomato_yellow_leaf_curl_virus':'tomato_yellow_leaf_curl_virus',
    'tylcv':                        'tomato_yellow_leaf_curl_virus',
    # TOMATO MOSAIC VIRUS
    'mosaic_virus':                 'tomato_mosaic_virus',
    'tomato_mosaic_virus':          'tomato_mosaic_virus',
    'tomv':                         'tomato_mosaic_virus',
    # TOMATO HEALTHY
    'tomato_healthy':               'tomato_healthy',
    'healthy_tomato':               'tomato_healthy',
    # CHILLI LEAF CURL
    'chilli_leaf_curl':             'chilli_leaf_curl',
    'leaf_curl':                    'chilli_leaf_curl',
    'chili_leaf_curl':              'chilli_leaf_curl',
    # CHILLI HEALTHY
    'chilli_healthy':               'chilli_healthy',
    'chili_healthy':                'chilli_healthy',
    'healthy_chilli':               'chilli_healthy',
    'capsicum_healthy':             'chilli_healthy',  # accepted lab images
    # CHILLI CERCOSPORA
    'chilli_cercospora_leaf_spot':  'chilli_cercospora_leaf_spot',
    'cercospora_capsici':           'chilli_cercospora_leaf_spot',
    'chili_leaf_spot':              'chilli_cercospora_leaf_spot',
    # CHILLI ANTHRACNOSE
    'chilli_anthracnose':           'chilli_anthracnose',
    'anthracnose':                  'chilli_anthracnose',
    'colletotrichum':               'chilli_anthracnose',
}

# ── IMAGE / TRAINING HYPERPARAMETERS (from MASTER_PLAN.md) ────────────────
# DINOv2-Small backbone (NOT EfficientNetV2-S — that was the old 23-class config)
DINOV2_BACKBONE  = 'vit_small_patch14_dinov2.lvd142m'
DINOV2_IMG_SIZE  = 224         # CRITICAL: must pass to timm.create_model (default 518 crashes)
DINOV2_EMBED_DIM = 384
IMG_SIZE         = (224, 224)
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
DROPOUT_P        = 0.3
LABEL_SMOOTHING  = 0.1
RANDOM_SEED      = 42

# LoRA configuration
LORA_RANK           = 8
LORA_ALPHA          = 16
LORA_TARGET_MODULES = ['qkv']   # timm DINOv2 uses fused qkv, not separate q/v
LORA_DROPOUT        = 0.1

# FiLM conditioning on LoRA adapter outputs
FILM_CROP_EMBEDDING_DIM = 4    # sufficient for binary tomato/chilli signal

# Training schedule
# Batch 32 verified safe: 0.27 GB peak VRAM, 304 img/s throughput (Phase 0 test)
# Batch 64 also fits (0.49 GB) but throughput drops to 280 img/s (GPU saturated on LoRA backward)
BATCH_SIZE              = 32
GRAD_ACCUM_STEPS        = 2   # effective batch 64 (same as previous 16*4)
STAGE1_EPOCHS           = 25
STAGE2_EPOCHS           = 7
STAGE1_LR               = 1e-4
STAGE2_LR               = 5e-5
WEIGHT_DECAY            = 1e-2
GRAD_CLIP_NORM          = 1.0
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001
DATALOADER_NUM_WORKERS  = 0     # Windows — test 2 in Phase 0, use if stable

# Curriculum learning (Model 3 only)
CURRICULUM_PHASE1_EPOCHS = 8    # epochs 1-8: field + diverse-source only
CURRICULUM_PHASE2_START  = 9    # epochs 9-25: all images

# Sampling
SCIDB_CAP_PER_CLASS      = 1000  # max scidb images per class per epoch
FIELD_PHOTO_MULTIPLIER   = 4.0
# ENS beta=0.999 (not 0.9999) for Model 3's 13:1 pre-cap imbalance
# CVPR 2019 CBLoss paper: beta=0.9999 optimal for ~10:1 (CIFAR-10),
# but for higher imbalance ratios, lower beta gives better calibration.
# Model 2 keeps 0.9999 (10:1 ratio fits the CIFAR-10 finding).
ENS_BETA                 = 0.999

# CutMix (chilli_anthracnose only, after epoch 12)
CUTMIX_CLASSES           = ['chilli_anthracnose']
CUTMIX_PROBABILITY       = 0.3
CUTMIX_START_EPOCH       = 12

# Curl disease augmentation
CURL_DISEASE_CLASSES = ['chilli_leaf_curl', 'tomato_yellow_leaf_curl_virus']

# Self-distillation
# T=3.0 softens teacher output distribution to expose inter-class similarity
# (Born-Again Networks convention). T=2.0 is standard for CE-trained teachers;
# T=3.0 used here because it better captures the merged foliar_spot sub-class structure.
DISTILLATION_TEMPERATURE       = 3.0
DISTILLATION_AGREEMENT_THRESH  = 0.70
DISTILLATION_MIN_FIRST_PASS_F1 = 0.70
DISTILLATION_PER_CLASS_MIN_AGR = 0.50

# EMA
EMA_DECAY_STAGE1 = 0.9999
EMA_DECAY_STAGE2 = 0.999    # faster for shorter stage

# Rollback
ROLLBACK_SAVE_EPOCH    = 3
ROLLBACK_TRIGGER_EPOCH = 8
ROLLBACK_THRESHOLD     = 0.95   # trigger if val_f1 < rollback_f1 * threshold

# Soup checkpoints
SOUP_CHECKPOINT_EPOCHS = [17, 19, 21, 23, 25]  # Stage 1 near convergence

# ── INFERENCE THRESHOLDS ───────────────────────────────────────────────────
DISEASE_THRESHOLDS = {
    'tomato_foliar_spot':            0.30,
    'tomato_late_blight':            0.30,
    'tomato_septoria_leaf_spot':     0.30,
    'tomato_yellow_leaf_curl_virus': 0.30,
    'tomato_mosaic_virus':           0.30,
    'tomato_healthy':                0.35,
    'chilli_leaf_curl':              0.30,
    'chilli_healthy':                0.35,
    'chilli_cercospora_leaf_spot':   0.30,
    'chilli_anthracnose':            0.30,
}
DISEASE_THRESH = 0.30  # global fallback

# ── NEEDS VERIFICATION FLAGS ──────────────────────────────────────────────
NEEDS_VERIFICATION_CLASSES = ['tomato_yellow_leaf_curl_virus']
# tomato_yellow_leaf_curl_virus: only 16 field photos in 3,612 total (0.4% field)

# ── MONITORING FLAGS ───────────────────────────────────────────────────────
WATCH_CLASSES = {
    'chilli_healthy':     '928 images are lab Capsicum — risk of lab-background shortcut learning',
    'chilli_anthracnose': 'Thinnest chilli class at 653 images',
}

# Capsicum shortcut monitoring
CAPSICUM_GAP_THRESHOLD = 0.20   # if F1(capsicum) - F1(real_chilli) > this, halve weight
CAPSICUM_SOURCE_DATASET = 'multi_D'

# ── FILE PATHS ─────────────────────────────────────────────────────────────
ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL3_CLEANED   = os.path.join(ROOT, 'data', 'specialist', 'model3', 'cleaned')
MODEL3_CSV       = os.path.join(ROOT, 'data', 'specialist', 'model3',
                                'model3_unified_source_map.csv')
MODEL3_CHECKPOINTS = os.path.join(ROOT, 'models', 'model3_specialist')
QUARANTINE_DIR   = os.path.join(MODEL3_CLEANED, 'tomato_target_spot_QUARANTINED')

# ── ACCEPTANCE CRITERIA ───────────────────────────────────────────────────
MIN_MACRO_F1             = 0.72
MIN_TOMATO_PER_CLASS_F1  = 0.58
MIN_CHILLI_PER_CLASS_F1  = 0.76
MAX_CAPSICUM_GAP         = 0.15

# ── VALIDATION HELPERS ─────────────────────────────────────────────────────
def assert_config_consistency():
    """Sanity-check that CLASS_NAMES, indices, and crop mappings are consistent.
    Called by training scripts at startup to catch config bugs early."""
    assert len(CLASS_NAMES) == 10, f'Expected 10 classes, got {len(CLASS_NAMES)}'
    assert NUM_CLASSES == 10
    assert 'tomato_target_spot' not in CLASS_NAMES, \
        'tomato_target_spot must be excluded from Model 3 specialist'
    assert len(TOMATO_INDICES) == 6
    assert len(CHILLI_INDICES) == 4
    assert set(TOMATO_INDICES) | set(CHILLI_INDICES) == set(range(NUM_CLASSES))
    assert max(CROP_FROM_IDX.keys()) == NUM_CLASSES - 1
    assert all(0 <= v < NUM_CROPS for v in CROP_FROM_IDX.values())
    for cls, idx in CLASS_TO_IDX.items():
        assert IDX_TO_CLASS[idx] == cls
    # Verify merged foliar_spot mapping
    assert LABEL_MAP['bacterial_spot'] == 'tomato_foliar_spot'
    assert LABEL_MAP['early_blight']   == 'tomato_foliar_spot'
    assert LABEL_MAP['leaf_mold']      == 'tomato_foliar_spot'
    # Verify target_spot is NOT in LABEL_MAP
    assert 'target_spot' not in LABEL_MAP, \
        'target_spot must not be in LABEL_MAP (quarantined class)'
    assert 'tomato_target_spot' not in LABEL_MAP
    # Verify thresholds match class names
    for cls in CLASS_NAMES:
        assert cls in DISEASE_THRESHOLDS, f'{cls} missing from DISEASE_THRESHOLDS'
    # Verify no threshold exists for quarantined class
    assert 'tomato_target_spot' not in DISEASE_THRESHOLDS
    return True


if __name__ == '__main__':
    assert_config_consistency()
    print('Model 3 config consistency check PASSED')
    print(f'NUM_CLASSES = {NUM_CLASSES}')
    for i, name in enumerate(CLASS_NAMES):
        crop = CROP_NAMES[CROP_FROM_IDX[i]]
        is_healthy = '(healthy)' if i in HEALTHY_INDICES else ''
        watch = f'  [WATCH: {WATCH_CLASSES[name]}]' if name in WATCH_CLASSES else ''
        needs_v = '  [NEEDS_VERIFICATION]' if name in NEEDS_VERIFICATION_CLASSES else ''
        print(f'  {i}: {name:<35} crop={crop:<8} {is_healthy}{watch}{needs_v}')
    print(f'\nEXCLUDED: {EXCLUDED_CLASSES}')
    print(f'  Reason: {EXCLUSION_REASON[:80]}...')
