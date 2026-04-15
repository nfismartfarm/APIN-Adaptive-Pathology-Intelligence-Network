"""
Model 2 Specialist Config: Okra + Brassica (9 classes)

IMPORTANT:
- This file is SEPARATE from app/config.py, which remains the source of truth
  for the existing 23-class Swin-Tiny production model (swin_best_model.pt).
- brassica_clubroot has been PERMANENTLY DROPPED from this specialist model.
  Its images are quarantined at data/specialist/model2/cleaned/brassica_clubroot_QUARANTINED/
  and excluded from training.
- The 9 classes are re-indexed 0-8 (dense, no gaps). Do NOT try to preserve the
  original 10-class indices from app/config.py — that would leave a dead index.
- Training scripts for Model 2 MUST import from here, not from app.config.

Data source of truth: data/specialist/model2/model2_unified_source_map.csv
Destination folder:   data/specialist/model2/cleaned/{class_name}/
"""

import os

# ── CLASS DEFINITIONS ──────────────────────────────────────────────────────
# 9 classes, densely indexed 0-8. brassica_clubroot is EXCLUDED.
CLASS_NAMES = [
    # Okra (indices 0-4)
    'okra_yvmv',
    'okra_powdery_mildew',
    'okra_cercospora',
    'okra_enation',
    'okra_healthy',
    # Brassica (indices 5-8) — clubroot removed, so brassica_healthy moves from 9 to 8
    'brassica_black_rot',
    'brassica_downy_mildew',
    'brassica_alternaria',
    'brassica_healthy',
]
NUM_CLASSES      = len(CLASS_NAMES)                # 9
CLASS_TO_IDX     = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS     = {i: n for i, n in enumerate(CLASS_NAMES)}

# Crop indices within Model 2 (okra=0, brassica=1)
NUM_CROPS        = 2
CROP_NAMES       = ['okra', 'brassica']
OKRA_INDICES     = [0, 1, 2, 3, 4]
BRASSICA_INDICES = [5, 6, 7, 8]                    # 4 classes now, not 5
HEALTHY_INDICES  = [4, 8]                          # okra_healthy=4, brassica_healthy=8

CROP_FROM_IDX = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0,  # okra → crop 0
    5: 1, 6: 1, 7: 1, 8: 1,        # brassica → crop 1
}

CROP_TO_DISEASE_INDICES = {
    0: [0, 1, 2, 3, 4],      # okra diseases
    1: [5, 6, 7, 8],         # brassica diseases (no clubroot)
}

HEALTHY_CLASSES = {'okra_healthy', 'brassica_healthy'}

# ── EXCLUSION RECORD ───────────────────────────────────────────────────────
# brassica_clubroot was dropped because:
#   1. Root-disease symptoms are not reliably visible on leaves
#   2. Above-ground symptoms overlap with nitrogen deficiency, fusarium wilt, water stress
#   3. Only 304 training images available, all from single source (single-source risk)
#   4. Quarantined at data/specialist/model2/cleaned/brassica_clubroot_QUARANTINED/
#      (100 new images from Dataset B + 204 from original source_map.csv)
EXCLUDED_CLASSES = {'brassica_clubroot'}
EXCLUSION_REASON = 'Root disease not reliably diagnosable from leaf images'

# ── LABEL HARMONISATION ────────────────────────────────────────────────────
# Maps dataset-specific raw labels to our canonical Model 2 class names.
# brassica_clubroot mappings are DELIBERATELY removed — any dataset row labelled
# as clubroot will fail the assert_all_labels_mapped() check and either needs
# to be dropped or reassigned to brassica_healthy during data prep.
LABEL_MAP = {
    # OKRA YVMV
    'okra_yellow_vein': 'okra_yvmv',
    'yvmv': 'okra_yvmv',
    'yellow vein mosaic': 'okra_yvmv',
    'yellow_vein_mosaic': 'okra_yvmv',
    'yellow_vein_mosaic_virus': 'okra_yvmv',
    'bhindi_mosaic': 'okra_yvmv',
    'okra_yvmv': 'okra_yvmv',
    'yellow_vein_mosaic_disease': 'okra_yvmv',
    'yellowveinmosaic': 'okra_yvmv',
    # NOTE: 'leaf curl' is NOT mapped globally — it's source-dependent.
    # sabbir_okra 'leaf curl' → okra_enation (per CLAUDE.md SOURCE_LABEL_OVERRIDES)
    # iubat_okra 'leaf curl' → okra_enation (per CLAUDE.md SOURCE_LABEL_OVERRIDES)
    # Global mapping removed to prevent mislabelling okra_enation as okra_yvmv.
    # OKRA POWDERY MILDEW
    'okra_powdery_mildew': 'okra_powdery_mildew',
    'powdery_mildew_okra': 'okra_powdery_mildew',
    # OKRA CERCOSPORA
    'okra_leaf_spot': 'okra_cercospora',
    'cercospora_leaf_spot': 'okra_cercospora',
    'okra_cercospora': 'okra_cercospora',
    'cercospora_abelmoschi': 'okra_cercospora',
    # OKRA ENATION
    'enation_leaf_curl': 'okra_enation',
    'okra_enation': 'okra_enation',
    # OKRA HEALTHY
    'okra_healthy': 'okra_healthy',
    'healthy_okra': 'okra_healthy',
    # BRASSICA BLACK ROT
    'black_rot': 'brassica_black_rot',
    'brassica_black_rot': 'brassica_black_rot',
    'cabbage_black_rot': 'brassica_black_rot',
    'xanthomonas': 'brassica_black_rot',
    # BRASSICA DOWNY MILDEW
    'downy_mildew_brassica': 'brassica_downy_mildew',
    'brassica_downy_mildew': 'brassica_downy_mildew',
    'cabbage_downy_mildew': 'brassica_downy_mildew',
    'hyaloperonospora': 'brassica_downy_mildew',
    # BRASSICA ALTERNARIA
    'alternaria_brassicae': 'brassica_alternaria',
    'alternaria_leaf_spot_brassica': 'brassica_alternaria',
    'brassica_alternaria': 'brassica_alternaria',
    'cabbage_alternaria': 'brassica_alternaria',
    'dark_leaf_spot': 'brassica_alternaria',
    'alternaria_leaf_spot': 'brassica_alternaria',
    # BRASSICA HEALTHY
    'brassica_healthy': 'brassica_healthy',
    'cabbage_healthy': 'brassica_healthy',
    'cauliflower_healthy': 'brassica_healthy',
    'healthy_brassica': 'brassica_healthy',
    'broccoli_healthy': 'brassica_healthy',
}

# ── IMAGE / TRAINING HYPERPARAMETERS ───────────────────────────────────────
IMG_SIZE        = (384, 384)       # ConvNeXt-Small native (matches CONVNEXT_BACKBONE pretrain resolution)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
# DINOv3-ConvNeXt-Small: self-supervised pretraining on 1.7B images,
# distilled from 7B ViT teacher. Superior to IN22k for domain generalisation.
# Loaded via transformers library (not timm — HF model config incompatible with timm).
# Access granted: user has HuggingFace access to facebook/dinov3-* model family.
# Verified: 49.5M params, 768-dim pooler_output, GradCAM++ compatible via stages.3
DINOV3_BACKBONE   = 'facebook/dinov3-convnext-small-pretrain-lvd1689m'
BACKBONE_LIBRARY  = 'transformers'     # NOT timm — use AutoModel.from_pretrained()
BACKBONE_EMBED_DIM = 768               # pooler_output dimension

# Fallback if DINOv3 access revoked or download fails:
FALLBACK_BACKBONE = 'convnext_small.fb_in22k_ft_in1k_384'  # timm, IN22k pretrained
FALLBACK_LIBRARY  = 'timm'
DROPOUT_P       = 0.3
LABEL_SMOOTHING = 0.1

# ── TRAINING SCHEDULE ─────────────────────────────────────────────────────
# Stage 1: progressive resize 128->224->384, SupCon + ASAM (epochs 1-25)
# Stage 2: head-only retraining, CutMix, Focal Loss (epochs 26-32)
STAGE1_EPOCHS           = 25
STAGE2_EPOCHS           = 7

# Progressive resize batch sizes — Stage 1 (measured VRAM: batch 16 @ 384px = 4.62 GB)
BATCH_SIZES_STAGE1 = {128: 32, 224: 16, 384: 16}
GRAD_ACCUM_STEPS_STAGE1 = 2     # effective batch = BATCH * GRAD_ACCUM

# Stage 2 batch sizes (head-only, no ASAM, more VRAM headroom)
BATCH_SIZE_STAGE2       = 16    # at 384px, head-only
GRAD_ACCUM_STEPS_STAGE2 = 2     # effective batch = 32
DATALOADER_NUM_WORKERS  = 0     # Windows — >0 causes crashes

# Learning rates
STAGE1_BASE_LR          = 1e-3
STAGE2_HEAD_LR          = 1e-4
LLRD_DECAY              = 0.90  # per-block LR decay for ConvNeXt stages
WEIGHT_DECAY            = 1e-2  # applied to non-bias, non-LayerNorm params
GRAD_CLIP_NORM          = 1.0
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001
RANDOM_SEED             = 42

# ── ASAM (Sharpness-Aware Minimisation) ───────────────────────────────────
# Model 2 Stage 1 only. Disabled for 2 warmup epochs after each resolution transition.
# Not used in Router (linear head is convex) or Model 3 (LoRA too few params).
ASAM_RHO_BY_RESOLUTION = {224: 0.10, 384: 0.20}
ASAM_WARMUP_EPOCHS_PER_RESOLUTION = 2  # disable ASAM for this many epochs after res change

# ── SupCon (Supervised Contrastive Loss) ──────────────────────────────────
# Stage 1 only (epochs 1-15). NOT at 384px (batch too small for class-balanced pairs).
# NOT compatible with CutMix — disabled when CutMix is active in Stage 2.
SUPCON_TEMPERATURE      = 0.10  # Khosla et al. 2020 recommends 0.1 for >5 class fine-grained
SUPCON_LAMBDA           = 0.10  # weight of SupCon loss vs CE: total = CE + 0.1*SupCon
SUPCON_MAX_EPOCH        = 15    # SupCon disabled after this epoch (384px starts at 16)

# ── CutMix (Stage 2 thin classes only) ────────────────────────────────────
# Applied only when thin classes appear in batch. Incompatible with SupCon.
CUTMIX_CLASSES_STAGE2   = ['okra_enation', 'okra_cercospora']
CUTMIX_PROBABILITY      = 0.3
CUTMIX_ALPHA            = 1.0

# ── EMA (Exponential Moving Average) ─────────────────────────────────────
EMA_DECAY_STAGE1        = 0.9999  # slow averaging during 25-epoch Stage 1
EMA_DECAY_STAGE2        = 0.999   # faster averaging for 7-epoch Stage 2

# ── Rollback ─────────────────────────────────────────────────────────────
# Resolution-aware: do NOT compare F1 across resolutions (always drops at transition)
ROLLBACK_SAVE_EPOCHS    = [5, 15, 25]  # end of each resolution stage (128px, 224px, 384px)
ROLLBACK_THRESHOLD      = 0.95   # trigger if val_f1 < rollback_f1 * threshold
# Check at epoch 8 (within 128px), 18 (within 224px), 28 (within 384px)

# ── Sampling ─────────────────────────────────────────────────────────────
FIELD_PHOTO_MULTIPLIER  = 4.0    # field photos sampled 4x more often (specialist, not router's 5x)
ENS_BETA                = 0.9999  # Effective Number of Samples weighting
STAGE2_BALANCED_COUNT   = 432    # 1.5x okra_enation count, per class per epoch

# ── Soup checkpoints ─────────────────────────────────────────────────────
SOUP_CHECKPOINT_EPOCHS  = [25, 27, 29, 31, 32]  # near convergence, greedy selection

# ── Needs verification (inference layer flag) ─────────────────────────────
NEEDS_VERIFICATION_CLASSES = ['okra_enation']  # 288 images, single source, F1 ceiling 0.55-0.68

# ── GradCAM++ heatmap target ─────────────────────────────────────────────
# DINOv3-ConvNeXt-Small via transformers: target is last conv stage
GRADCAM_TARGET_LAYER    = 'stages.3'  # 768-ch, 12x12 at 384px input

# ── Acceptance criteria ──────────────────────────────────────────────────
MIN_MACRO_F1            = 0.82
MIN_OKRA_ENATION_F1     = 0.55   # honest ceiling given 288 single-source images

# ── INFERENCE THRESHOLDS (copied from production config, clubroot excluded) ─
DISEASE_THRESHOLDS = {
    'okra_yvmv':              0.30,
    'okra_powdery_mildew':    0.30,
    'okra_cercospora':        0.30,
    'okra_enation':            0.30,
    'okra_healthy':           0.35,
    'brassica_black_rot':     0.30,
    'brassica_downy_mildew':  0.30,
    'brassica_alternaria':    0.30,
    'brassica_healthy':       0.35,
}
DISEASE_THRESH = 0.30  # global fallback

# ── FILE PATHS ─────────────────────────────────────────────────────────────
ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL2_CLEANED   = os.path.join(ROOT, 'data', 'specialist', 'model2', 'cleaned')
MODEL2_CSV       = os.path.join(ROOT, 'data', 'specialist', 'model2',
                                'model2_unified_source_map.csv')
MODEL2_CHECKPOINTS = os.path.join(ROOT, 'models', 'model2_specialist')
os.makedirs(MODEL2_CHECKPOINTS, exist_ok=True) if False else None  # don't side-effect on import

QUARANTINE_DIR   = os.path.join(MODEL2_CLEANED, 'brassica_clubroot_QUARANTINED')

# ── VALIDATION HELPERS ─────────────────────────────────────────────────────
def assert_config_consistency():
    """Sanity-check that CLASS_NAMES, indices, and crop mappings are consistent.
    Called by training scripts at startup to catch config bugs early."""
    assert len(CLASS_NAMES) == 9, f'Expected 9 classes, got {len(CLASS_NAMES)}'
    assert NUM_CLASSES == 9
    assert 'brassica_clubroot' not in CLASS_NAMES, \
        'brassica_clubroot must be excluded from Model 2 specialist'
    assert len(OKRA_INDICES) == 5
    assert len(BRASSICA_INDICES) == 4, \
        f'Brassica should have 4 classes (clubroot dropped), got {len(BRASSICA_INDICES)}'
    assert set(OKRA_INDICES) | set(BRASSICA_INDICES) == set(range(NUM_CLASSES))
    assert max(CROP_FROM_IDX.keys()) == NUM_CLASSES - 1
    assert all(0 <= v < NUM_CROPS for v in CROP_FROM_IDX.values())
    for cls, idx in CLASS_TO_IDX.items():
        assert IDX_TO_CLASS[idx] == cls
    return True


if __name__ == '__main__':
    assert_config_consistency()
    print('Model 2 config consistency check PASSED')
    print(f'NUM_CLASSES = {NUM_CLASSES}')
    for i, name in enumerate(CLASS_NAMES):
        crop = CROP_NAMES[CROP_FROM_IDX[i]]
        is_healthy = '(healthy)' if i in HEALTHY_INDICES else ''
        print(f'  {i}: {name:<25} crop={crop:<10} {is_healthy}')
