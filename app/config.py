# app/config.py
# Single source of truth for ALL constants. Every other module imports from here.
# No magic numbers anywhere else in the codebase.

import os
import torch

# ── CLASS DEFINITIONS ──────────────────────────────────────────────────────
CLASS_NAMES = [
    'okra_yvmv','okra_powdery_mildew','okra_cercospora','okra_enation',
    'okra_healthy','brassica_black_rot','brassica_downy_mildew',
    'brassica_alternaria','brassica_clubroot','brassica_healthy',
]
NUM_CLASSES      = len(CLASS_NAMES)
CLASS_TO_IDX     = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS     = {i: n for i, n in enumerate(CLASS_NAMES)}
OKRA_INDICES     = [0, 1, 2, 3, 4]
BRASSICA_INDICES = [5, 6, 7, 8, 9]
HEALTHY_INDICES  = [4, 9]
NUM_CROPS        = 2
CROP_FROM_IDX    = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1}
CROP_NAMES       = {0: 'okra', 1: 'brassica'}
HEALTHY_CLASSES  = {'okra_healthy', 'brassica_healthy'}

# ── PLANTDOC CLASS MAP ──────────────────────────────────────────────────────
# [FIX GAP 52] Exact PlantDoc folder name -> canonical class mapping.
# Used by download_plantdoc.py and 08_evaluate_tier2_plantdoc.py.
PLANTDOC_CLASS_MAP = {
    'Cabbage__Black_Rot'              : 'brassica_black_rot',
    'Cabbage__Downy_Mildew'           : 'brassica_downy_mildew',
    'Cabbage__Alternaria_leaf_spot'   : 'brassica_alternaria',
    'Cabbage__healthy'                : 'brassica_healthy',
    'cabbage black rot'               : 'brassica_black_rot',
    'cabbage downy mildew'            : 'brassica_downy_mildew',
    'cabbage alternaria leaf spot'    : 'brassica_alternaria',
    'cabbage healthy'                 : 'brassica_healthy',
}

# ── LABEL HARMONISATION MAPS ───────────────────────────────────────────────
# [FIX GAP 22] Defined here (not in 01_prepare_data.py) so both training
# scripts and agent scripts can import from app.config without circular imports.
LABEL_MAP = {
    # OKRA YVMV
    'okra_yellow_vein':'okra_yvmv','yvmv':'okra_yvmv',
    'yellow vein mosaic':'okra_yvmv','yellow_vein_mosaic':'okra_yvmv',
    'yellow_vein_mosaic_virus':'okra_yvmv','bhindi_mosaic':'okra_yvmv',
    'yellow vein mosaic virus':'okra_yvmv','okra_yvmv':'okra_yvmv',
    'yellowveinmosaic':'okra_yvmv','yellow vein':'okra_yvmv',
    'mosaic virus':'okra_yvmv','yvm':'okra_yvmv',
    'yellow_mosaic':'okra_yvmv','yellowmosaic':'okra_yvmv',
    # OKRA POWDERY MILDEW
    'okra_powdery_mildew':'okra_powdery_mildew',
    'powdery_mildew_okra':'okra_powdery_mildew',
    'powdery mildew okra':'okra_powdery_mildew',
    # OKRA CERCOSPORA
    'okra_leaf_spot':'okra_cercospora','cercospora':'okra_cercospora',
    'cercospora_leaf_spot':'okra_cercospora','okra_cercospora':'okra_cercospora',
    'cercospora_abelmoschi':'okra_cercospora','leaf spot okra':'okra_cercospora',
    # OKRA ENATION
    'enation_leaf_curl':'okra_enation','okra_leaf_curl':'okra_enation',
    'enation leaf curl':'okra_enation','okra_enation':'okra_enation',
    'leaf_curl_okra':'okra_enation','okra leaf curl':'okra_enation',
    'enation':'okra_enation',
    # OKRA HEALTHY
    'okra_healthy':'okra_healthy','healthy_okra':'okra_healthy',
    'okra healthy':'okra_healthy','okra_normal':'okra_healthy',
    'healthy okra':'okra_healthy',
    # BRASSICA BLACK ROT
    'black_rot':'brassica_black_rot','brassica_black_rot':'brassica_black_rot',
    'blackrot':'brassica_black_rot','black rot':'brassica_black_rot',
    'cabbage_black_rot':'brassica_black_rot','xanthomonas':'brassica_black_rot',
    'bacterial_black_rot':'brassica_black_rot',
    # BRASSICA DOWNY MILDEW
    'downy_mildew_brassica':'brassica_downy_mildew',
    'brassica_downy_mildew':'brassica_downy_mildew',
    'cabbage_downy_mildew':'brassica_downy_mildew',
    'downy mildew brassica':'brassica_downy_mildew',
    'hyaloperonospora':'brassica_downy_mildew',
    'downy mildew cabbage':'brassica_downy_mildew',
    # BRASSICA ALTERNARIA
    'alternaria_brassicae':'brassica_alternaria',
    'alternaria_leaf_spot_brassica':'brassica_alternaria',
    'brassica_alternaria':'brassica_alternaria',
    'cabbage_alternaria':'brassica_alternaria',
    'dark_leaf_spot':'brassica_alternaria',
    'alternaria leaf spot':'brassica_alternaria',
    'alternaria brassica':'brassica_alternaria',
    # BRASSICA CLUBROOT
    'clubroot':'brassica_clubroot','brassica_clubroot':'brassica_clubroot',
    'club root':'brassica_clubroot','club_root':'brassica_clubroot',
    'plasmodiophora':'brassica_clubroot',
    # BRASSICA HEALTHY
    'brassica_healthy':'brassica_healthy','cabbage_healthy':'brassica_healthy',
    'cauliflower_healthy':'brassica_healthy','healthy_brassica':'brassica_healthy',
    'healthy cabbage':'brassica_healthy','healthy_cabbage':'brassica_healthy',
    'broccoli_healthy':'brassica_healthy','healthy_broccoli':'brassica_healthy',
    'healthy brassica':'brassica_healthy',
}

SOURCE_LABEL_OVERRIDES = {
    # 'powdery mildew' — different crops use this string
    ('sabbir_okra',   'powdery mildew'):'okra_powdery_mildew',
    ('iubat_okra',    'powdery mildew'):'okra_powdery_mildew',
    ('faruk_okra',    'powdery mildew'):'okra_powdery_mildew',
    ('kareem_cabbage','powdery mildew'):'brassica_downy_mildew',
    ('ghose_cabbage', 'powdery mildew'):'brassica_downy_mildew',
    ('misrak_veg',    'powdery mildew'):'brassica_downy_mildew',
    # 'leaf spot' — okra=cercospora, brassica=alternaria
    ('sabbir_okra',   'leaf_spot'):'okra_cercospora',
    ('iubat_okra',    'leaf_spot'):'okra_cercospora',
    ('faruk_okra',    'leaf_spot'):'okra_cercospora',
    ('kareem_cabbage','leaf_spot'):'brassica_alternaria',
    ('ghose_cabbage', 'leaf_spot'):'brassica_alternaria',
    ('misrak_veg',    'leaf_spot'):'brassica_alternaria',
    ('plantdoc',      'leaf_spot'):'brassica_alternaria',
    # 'leaf curl' context
    ('sabbir_okra',   'leaf curl'):'okra_enation',
    ('iubat_okra',    'leaf curl'):'okra_enation',
    # 'downy mildew' — only brassica datasets
    ('kareem_cabbage','downy mildew'):'brassica_downy_mildew',
    ('ghose_cabbage', 'downy mildew'):'brassica_downy_mildew',
    ('plantdoc',      'downy mildew'):'brassica_downy_mildew',
    # 'healthy' without crop qualifier
    ('sabbir_okra',   'healthy'):'okra_healthy',
    ('iubat_okra',    'healthy'):'okra_healthy',
    ('faruk_okra',    'healthy'):'okra_healthy',
    ('kareem_cabbage','healthy'):'brassica_healthy',
    ('ghose_cabbage', 'healthy'):'brassica_healthy',
    ('misrak_veg',    'healthy'):'brassica_healthy',
    ('plantdoc',      'healthy'):'brassica_healthy',
    # 'alternaria' — only brassica datasets have it
    ('kareem_cabbage','alternaria'):'brassica_alternaria',
    ('ghose_cabbage', 'alternaria'):'brassica_alternaria',
    ('plantdoc',      'alternaria'):'brassica_alternaria',
}

# ── MODEL ARCHITECTURE ─────────────────────────────────────────────────────
BACKBONE_NAME   = 'efficientnetv2_s'
# timm: features_only=True, out_indices=(2,3,4)
# Stage 2: 48ch  28×28 (H/8)   at 224px input = P3
# Stage 3: 160ch 14×14 (H/16)  at 224px input = P4
# Stage 4: 256ch  7×7  (H/32)  at 224px input = P5
FPN_IN_CH       = [48, 160, 256]   # must match timm stage outputs
FPN_OUT_CH      = 256              # all FPN levels projected to this
POOLED_DIM      = 256              # after GlobalAvgPool on FPN P3 output
CROP_EMB_DIM    = 64               # crop classifier embedding dimension
HEAD_HIDDEN_DIM = 256              # hidden layer in disease and severity heads
DROPOUT_P       = 0.3
IMG_H = IMG_W   = 224
IMG_SIZE        = (224, 224)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ── TRAINING ───────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
PHASE1_EPOCHS    = 10
PHASE2_EPOCHS    = 7
PHASE1_LR        = 1e-3
PHASE2_BASE_LR   = 1e-4
LLRD_DECAY       = 0.85
GRAD_CLIP_NORM   = 1.0
BATCH_SIZE       = 32     # use 16 + GRAD_ACCUM_STEPS=2 if VRAM OOM
GRAD_ACCUM_STEPS = 1
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTH     = 0.1
LOSS_W_CROP      = 0.4
LOSS_W_DISEASE   = 0.4
LOSS_W_SEVERITY  = 0.2
MAX_POS_WEIGHT   = 10.0   # cap to prevent loss destabilisation
EARLY_STOP_PAT   = 5
EARLY_STOP_DELTA = 0.001
KEEP_CKPTS       = 3
# [FIX GAP 35] OneCycleLR constants — used in 05_train_phase2.py.
# Import these; do NOT hardcode 0.1, 10, 1000 in training scripts.
ONE_CYCLE_PCT    = 0.1    # pct_start — warmup fraction of total steps
ONE_CYCLE_DIV    = 10     # div_factor — initial LR = max_lr / div_factor
ONE_CYCLE_FDIV   = 1000   # final_div_factor — final LR = max_lr / final_div

# ── SEVERITY PROXY GENERATION ──────────────────────────────────────────────
SEVERITY_PROXY_THRESHOLD = 0.30   # top 30% activations = lesion region
SEVERITY_MILD_MAX        = 0.15   # coverage < 0.15 = mild
SEVERITY_MOD_MAX         = 0.50   # coverage 0.15-0.50 = moderate, else severe

# ── DATA PIPELINE ──────────────────────────────────────────────────────────
# [FIX GAP 60] HEIC removed from VALID_EXT — pillow-heif not installed.
VALID_EXT        = {'.jpg', '.jpeg', '.png', '.webp',
                    '.JPG', '.JPEG', '.PNG', '.WEBP'}
SPLIT_TRAIN      = 0.70
SPLIT_VAL        = 0.15
SPLIT_TEST       = 0.15
MIN_IMGS_CLASS   = 150
CLUBROOT_OVERSAMPLE = 2.0

# ── INPUT VALIDATION ───────────────────────────────────────────────────────
MAX_FILE_MB      = 10
MIN_BLUR_VAR     = 80
MIN_PIXEL_MEAN   = 40
MAX_PIXEL_MEAN   = 220
MIN_IMG_DIM      = 150
MAX_CH_RATIO     = 0.65   # no single channel > 65% of total (non-plant check)

# ── INFERENCE ──────────────────────────────────────────────────────────────
DISEASE_THRESH   = 0.50
OOD_CONF_THRESH  = 0.60
OOD_UNC_THRESH   = 0.40
MC_PASSES        = 5
TEMP_INIT        = 1.5    # LBFGS starting value for temperature scaling

# ── EVALUATION THRESHOLDS ──────────────────────────────────────────────────
TIER2_MIN_F1    = 0.55
TIER3_MIN_ACC   = 0.70
TIER3_MIN_IMGS  = 50
TIER3_MIN_CLS   = 5      # minimum images per class to evaluate that class

# ── FILE PATHS (all relative to project ROOT — [FIX GAP 30]) ───────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA        = os.path.join(ROOT, 'data')
RAW         = os.path.join(ROOT, 'data', 'raw')
PROC        = os.path.join(ROOT, 'data', 'processed')  # created but empty
TRAIN_DIR   = os.path.join(ROOT, 'data', 'processed', 'train')   # not used
VAL_DIR     = os.path.join(ROOT, 'data', 'processed', 'val')     # not used
KERALA_DIR  = os.path.join(ROOT, 'data', 'kerala')
PLANTDOC_DIR= os.path.join(ROOT, 'data', 'plantdoc')
META        = os.path.join(ROOT, 'data', 'metadata')
SOURCE_MAP  = os.path.join(ROOT, 'data', 'metadata', 'source_map.csv')
SEV_LABELS  = os.path.join(ROOT, 'data', 'metadata', 'severity_labels.csv')
# [FIX GAP 62] CLASS_COUNTS_PATH was missing from v5:
CLASS_COUNTS_PATH = os.path.join(ROOT, 'data', 'metadata', 'class_counts.csv')
MODELS      = os.path.join(ROOT, 'models')
CKPT_DIR    = os.path.join(ROOT, 'models', 'checkpoints')
BEST_MODEL  = os.path.join(ROOT, 'models', 'best_model.pt')
TEMP_PATH   = os.path.join(ROOT, 'models', 'temperature.pt')
CACHE       = os.path.join(ROOT, 'cache')
TRAIN_CACHE = os.path.join(ROOT, 'cache', 'train_features.pt')
VAL_CACHE   = os.path.join(ROOT, 'cache', 'val_features.pt')
REPORTS     = os.path.join(ROOT, 'reports')
DIAG_JSON   = os.path.join(ROOT, 'diagnosis', 'diagnosis_lookup.json')

# ── DEVICE ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── WANDB ──────────────────────────────────────────────────────────────────
WANDB_PROJECT = 'plant-disease-kerala'
WANDB_CONFIG  = {
    'backbone'         : BACKBONE_NAME,
    'img_size'         : IMG_SIZE,
    'batch_size'       : BATCH_SIZE,
    'phase1_epochs'    : PHASE1_EPOCHS,
    'phase2_epochs'    : PHASE2_EPOCHS,
    'phase1_lr'        : PHASE1_LR,
    'phase2_base_lr'   : PHASE2_BASE_LR,
    'llrd_decay'       : LLRD_DECAY,
    'dropout_p'        : DROPOUT_P,
    'loss_w_crop'      : LOSS_W_CROP,
    'loss_w_disease'   : LOSS_W_DISEASE,
    'loss_w_severity'  : LOSS_W_SEVERITY,
    'grad_clip_norm'   : GRAD_CLIP_NORM,
    'weight_decay'     : WEIGHT_DECAY,
    'label_smooth'     : LABEL_SMOOTH,
}
