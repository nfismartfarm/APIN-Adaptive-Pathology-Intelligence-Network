"""
scripts/model3_training/model3_config.py
Trainer-side constants for the Model 3 final training run.

Design decisions:
- CLASS_NAMES / CLASS_TO_IDX / CROP_FROM_IDX are RE-EXPORTED from
  app/config_model3.py (not re-declared) to enforce a single source of truth.
  Per user clarification (D9): app/config_model3.py ordering is canonical
  because the APIN system's Signal 2 subsampling map (EN_TO_M2_INDEX_MAP)
  already depends on it. The original spec's CLASSES_10 ordering (with
  septoria at index 3, ylcv at index 2) must NOT be used — it would silently
  break APIN integration.
- The BACKBONE constant here is DIFFERENT from app/config_model3.py's:
  app/config_model3.py:     'vit_small_patch14_dinov2.lvd142m' (no registers, old Model3DINOLoRA)
  this file:                'vit_small_patch14_reg4_dinov2'    (with 4 registers, new spec)
  Do NOT confuse them — they are different pretrained checkpoints.

All hyperparameter values are the empirically-grounded choices from the
April 2026 diagnostics (probe + 5-epoch mixed baseline). See docs/
pre_training_checklist.md for the rationale per value.
"""
from __future__ import annotations

from pathlib import Path

# ── Canonical class ordering (imported, NOT re-declared) ──────────────────
from app.config_model3 import (
    CLASS_NAMES,       # verbose CSV names; septoria at idx 2, ylcv at idx 3
    NUM_CLASSES,       # 10
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CROP_FROM_IDX,
    CROP_NAMES,        # ['tomato', 'chilli']
    TOMATO_INDICES,    # [0..5]
    CHILLI_INDICES,    # [6..9]
    HEALTHY_CLASSES,
    EXCLUDED_CLASSES,  # {'tomato_target_spot'}
    IMG_SIZE,          # (224, 224)
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Convenience: short alias + crop-by-class map
CLASSES_10 = CLASS_NAMES                                # alias
CROP_BY_CLASS = {c: ('tomato' if 'tomato' in c else 'chilli') for c in CLASS_NAMES}

# ── Project paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
SPLIT_PATH = ROOT / 'data' / 'specialist' / 'model3' / 'split_indices.json'
CHECKPOINT_DIR = ROOT / 'scripts' / 'model3_training' / 'checkpoints'
LOG_DIR = ROOT / 'scripts' / 'model3_training' / 'logs'
PROBE_RESULTS = ROOT / 'scripts' / 'model3_training' / 'probe' / 'probe_results.json'

# ── Backbone ──────────────────────────────────────────────────────────────
# WITH-registers variant per spec. Different from app/config_model3's backbone.
BACKBONE = 'vit_small_patch14_reg4_dinov2'
PROBE_IMG_SIZE = 224         # timm default is 518 — must override explicitly
DINOV2_EMBED_DIM = 384       # per-token dim
FEAT_DIM = 768               # CLS(384) + mean(non-CLS)(384) concat

# ── LoRA (rank=4 per probe diagnostic, Part 3 of spec) ────────────────────
LORA_RANK = 4
LORA_ALPHA = 8               # alpha = 2 * rank convention
LORA_TARGET_MODULES = ['qkv']  # timm fused QKV projection; ['query','value'] attaches ZERO
LORA_DROPOUT = 0.1
LORA_EXPECTED_PARAMS_MIN = 60_000   # rank=4 -> ~73,728 expected
LORA_EXPECTED_PARAMS_MAX = 100_000

# ── Head architecture ─────────────────────────────────────────────────────
SE_REDUCTION = 16             # 768 -> 48 -> 768
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1
MIXSTYLE_EPS = 1e-6
N_FILM_MODES = 3              # 0=tomato, 1=chilli, 2=uncertain
SUPCON_PROJ_HIDDEN = 128
SUPCON_PROJ_OUT = 64

# ── Training hyperparameters ──────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4               # Step 2 empirical winner (188.8 img/s)
GRAD_ACCUM_STEPS = 1          # effective batch = 32; no scaling fix needed
MAX_EPOCHS = 25                        # was 20 — retrain Change 2 (more budget)
EARLY_STOP_PATIENCE = 5                # was 3 — retrain Change 2 (more tolerance)
MIN_EPOCHS_BEFORE_STOP = 12            # NEW — retrain Change 2 (hard floor; CutMix from epoch 2 needs ≥10 epochs of pressure)
STOPPING_SMOOTHING_WINDOW = 3          # NEW — retrain Change 1 (3-epoch rolling mean of stopping_metric)
STOP_METRIC_WEIGHTS = (0.6, 0.4)       # (field_f1, overall_f1) — UNCHANGED

LR_LORA = 3e-4                # LoRA adapters
LR_HEAD = 8e-4                # SE, MixStyle, SupCon proj, FiLM, classifier
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LR_SCHEDULE_T_MAX = 20
LR_SCHEDULE_ETA_MIN = 1e-6

# ── Loss weights ──────────────────────────────────────────────────────────
LOSS_W_CE = 1.0
LOSS_W_SUPCON = 0.30
LOSS_W_FILM_IDENTITY_REG = 0.03  # raised from 0.01 (full-run spec, mandatory correction — verify FiLM beta drift 0.26→0.33→0.41)
LABEL_SMOOTHING = 0.1         # used in CE component

# ── SupCon (per user decision, temp=0.1 not 0.07 per Reb-SupCon 2025) ────
SUPCON_TEMPERATURE = 0.1
# Per-class weights — anthracnose downgraded (probe shows 0.897 already),
# septoria upweighted (primary target, gap 0.41).
SUPCON_CLASS_WEIGHTS = {
    'tomato_septoria_leaf_spot':        0.50,   # PRIMARY — largest field gap
    'tomato_foliar_spot':               0.35,   # confusable pair member
    'tomato_late_blight':               0.25,
    'tomato_mosaic_virus':              0.25,
    'tomato_yellow_leaf_curl_virus':    0.20,   # already 1.0 in baseline
    'tomato_healthy':                   0.20,
    'chilli_anthracnose':               0.20,   # DOWNGRADED from 0.50
    'chilli_cercospora_leaf_spot':      0.25,
    'chilli_leaf_curl':                 0.20,
    'chilli_healthy':                   0.20,
}

# ── ENS class weights ─────────────────────────────────────────────────────
ENS_BETA = 0.999              # NOT 0.9999 (Decision 9)
ENS_RATIO_CAP = 5.0           # cap max:min ratio at 5:1 (Model 2 had 9.4:1 -> collapse)

# ── AmpMix (2 classes only) ───────────────────────────────────────────────
AMPMIX_CLASSES = {'tomato_foliar_spot', 'tomato_septoria_leaf_spot'}
AMPMIX_BETA_ALPHA = 0.25       # Beta(0.25, 0.75) -> mean 0.25 lab amplitude
AMPMIX_BETA_BETA = 0.75
AMPMIX_PROBABILITY_LAB = 0.80  # for lab images of AMPMIX_CLASSES
AMPMIX_PROBABILITY_RECOMPOSED = 0.40  # for scidb/capsicum_recomposed
AMPMIX_FIELD_NEVER = True      # field images of AMPMIX_CLASSES are NEVER processed

# Adaptive intensification: if gap[class] > THIS for N consecutive epochs,
# raise probability to AMPMIX_PROBABILITY_HIGH.
AMPMIX_ADAPTIVE_GAP_THRESHOLD = 0.15  # lowered from 0.25 (full-run Change 1 — would have engaged septoria at verify-epoch-2)
AMPMIX_ADAPTIVE_CONSECUTIVE = 2
AMPMIX_PROBABILITY_HIGH = 0.95

# ── CutMix (one pair only) ────────────────────────────────────────────────
CUTMIX_PAIR = frozenset({'tomato_foliar_spot', 'tomato_septoria_leaf_spot'})
CUTMIX_EXCLUDED_CLASSES = {
    'chilli_anthracnose',            # thin — every example is precious
    'chilli_cercospora_leaf_spot',   # thin
    'tomato_healthy',                # thin
}
CUTMIX_START_EPOCH = 2           # lowered from 3 (full-run Change 3 — verify showed overall F1 stable at epoch 2, +1 epoch of foliar↔septoria pressure)
CUTMIX_BATCH_FRACTION = 0.15      # max 15% of batch slots

# ── DomainBalancedSampler ─────────────────────────────────────────────────
FIELD_FLOOR_TOMATO = 0.50
FIELD_FLOOR_CHILLI = 0.70
FIGSHARE_CAP_CHILLI_LEAF_CURL = 0.60

# ── Diagnostic thresholds ─────────────────────────────────────────────────
# Per-class halt: septoria < 0.50 at epoch 8 -> STOP
SEPTORIA_DIAGNOSTIC_HALT_EPOCH = 8
SEPTORIA_DIAGNOSTIC_HALT_F1 = 0.50
# Collapse monitor: any class < 0.05 for N consecutive epochs -> STOP
COLLAPSE_F1_THRESHOLD = 0.05
# COLLAPSE_CONSECUTIVE_EPOCHS moved to Round B section below — value is 5 (not 3)

# MixStyle regression guard (user clarification, 2026-04-18):
# If epoch 2 field_val_f1 < STEP3_BASELINE_EPOCH2_FIELD_F1 - 0.05, flag MixStyle.
STEP3_BASELINE_EPOCH2_FIELD_F1 = 0.8765
MIXSTYLE_REGRESSION_THRESHOLD = STEP3_BASELINE_EPOCH2_FIELD_F1 - 0.05  # 0.827

# ── LoRA verification test (Part 8) ───────────────────────────────────────
LORA_VERIFY_EPOCHS = 3
LORA_VERIFY_SEPTORIA_PROCEED = 0.70
LORA_VERIFY_SEPTORIA_DISCUSS_LOW = 0.60
LORA_VERIFY_SEPTORIA_STOP = 0.55

# ── Acceptance criteria (Part 11 final report) ────────────────────────────
ACCEPT_OVERALL_F1_MIN = 0.80
ACCEPT_OVERALL_F1_TARGET = 0.85
ACCEPT_FIELD_F1_MIN = 0.72
ACCEPT_FIELD_F1_TARGET = 0.80
ACCEPT_MEAN_GAP_MAX = 0.18
ACCEPT_MEAN_GAP_TARGET = 0.12
ACCEPT_SEPTORIA_FIELD_F1_MIN = 0.65
ACCEPT_PER_CLASS_FIELD_F1_MIN = 0.50

# ── Model soup (Proposed Change #4 / Part 6 of full_run_prompt) ───────────
# Greedy soup over top-N checkpoints by stopping_metric. Selection is done
# on the `soup_selection` split (never used for training or val). Selection
# metric is field_f1. A candidate is added to the soup if its inclusion does
# not drop field_f1 on soup_selection by more than SOUP_TOLERANCE.
SOUP_TOP_N = 5
SOUP_SELECTION_SPLIT = 'soup_selection'
SOUP_SELECTION_METRIC = 'field_f1'
SOUP_TOLERANCE = 0.01   # drop tolerance when evaluating candidates
PRODUCTION_CHECKPOINT_NAME = 'model3_production.pt'
PRODUCTION_V2_CHECKPOINT_NAME = 'model3_production_v2.pt'  # safety-net = full_v2_epoch09 copy
PRODUCTION_V3_CHECKPOINT_NAME = 'model3_production_v3.pt'  # Round B (full_v3) soup output

# ── Round B: tomato crop-mode dropout (YLCV oscillation fix) ─────────────
# YLCV's binary 0.000↔1.000 oscillation in v2 was caused by the model relying
# on FiLM crop conditioning as a shortcut. Raising the per-class uncertainty
# dropout for tomato images forces the model to learn YLCV's visual features
# (upward leaf curl, interveinal yellowing, crumpled appearance) rather than
# routing through the "told this is tomato → look for curl" shortcut.
# Chilli stays at 15% (no oscillation issue observed there).
TOMATO_CROP_MODE_UNCERTAIN_PROB = 0.25  # was effectively 0.15 globally
TOMATO_CROP_MODE_WRONG_PROB = 0.10      # adversarial wrong-crop, only after epoch 5 (unchanged)
CHILLI_CROP_MODE_UNCERTAIN_PROB = 0.15  # unchanged
CHILLI_CROP_MODE_WRONG_PROB = 0.10      # unchanged

# ── Round B: tightened CLASS_COLLAPSE criterion ──────────────────────────
# v2 halted at epoch 12 because YLCV had 3 consecutive 0.000s, but YLCV had
# previously hit 1.000 at epoch 9 — clearly not a permanent collapse, just
# oscillation. New criterion requires BOTH: (a) ≥5 consecutive epochs below
# 0.05, AND (b) the class has never exceeded 0.50 field F1 in this run.
# This catches genuinely-never-learned classes while letting oscillating
# classes recover. The smoothed stopping metric still protects aggregate
# patience from oscillation noise.
COLLAPSE_CONSECUTIVE_EPOCHS = 5    # was 3
COLLAPSE_RECOVERY_F1_FLOOR = 0.50  # NEW — class must have NEVER exceeded this
