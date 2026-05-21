"""
LADI-Net canonical configuration.

This module is the single source of truth for every hyperparameter referenced
by Decisions 15-38. Both Phase 1 (head training) and Phase 2 (LoRA fine-tune)
import from here. The training scripts assert these values at startup to
prevent silent drift between decision documents and implementation.

Corresponds to ladi_decisions.md Decisions 15-38.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL3_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
MASK_LOG_CSV = MODEL3_DIR / "mask_precompute_log.csv"
SPLIT_JSON = MODEL3_DIR / "split_indices.json"
CORAL_TARGET_PT = MODEL3_DIR / "coral_target_cov.pt"
PHASE1_CKPT_DIR = PROJECT_ROOT / "models" / "specialist" / "ladinet_checkpoints"
PHASE1_HEADS_PT = PROJECT_ROOT / "models" / "specialist" / "ladinet_phase1_heads.pt"
LOGS_DIR = PROJECT_ROOT / "logs" / "ladinet"

# ---------------------------------------------------------------------------
# Classes (canonical names per CSV — see Decision 3)
# ---------------------------------------------------------------------------
TOMATO_CLASSES = [
    "tomato_foliar_spot",
    "tomato_septoria_leaf_spot",
    "tomato_late_blight",
    "tomato_yellow_leaf_curl_virus",
    "tomato_mosaic_virus",
    "tomato_healthy",
]
NUM_CLASSES = 6
CLASS_TO_IDX = {c: i for i, c in enumerate(TOMATO_CLASSES)}
CONFUSABLE_CLASSES = {"tomato_foliar_spot", "tomato_septoria_leaf_spot"}

# ---------------------------------------------------------------------------
# Backbone + LoRA (Decision 17 §17.1)
# ---------------------------------------------------------------------------
BACKBONE = "vit_base_patch14_reg4_dinov2"
EMBED_DIM = 768
PATCH_SIZE = 14
RESOLUTION = 392                                 # Decision 15
NUM_PATCHES = (RESOLUTION // PATCH_SIZE) ** 2    # 784
NUM_REGISTERS = 4
PREFIX_TOKENS = 1 + NUM_REGISTERS                # 5 (CLS + 4 registers)
TOTAL_TOKENS = PREFIX_TOKENS + NUM_PATCHES       # 789

# LoRA (only relevant in Phase 2 — listed here for Phase 2 to pick up)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_SCALE = LORA_ALPHA / LORA_RANK              # 2.0
LORA_DROPOUT = 0.1
LORA_TARGET_BLOCKS = [4, 5, 6, 7, 8, 9, 10, 11]  # top 8 of 12 (Decision 15, 17)
LORA_IMPL = "manual_fused_qkv"                   # Decision 35

# ---------------------------------------------------------------------------
# Heads (Decision 17 §17.2, Decision 18)
# ---------------------------------------------------------------------------
ABMIL_HIDDEN = 256
FUSION_HIDDEN = 512
FUSION_MID = 256
FUSION_INPUT_DIM = EMBED_DIM + EMBED_DIM + 1     # 768 spatial + 768 global + 1 fallback_flag = 1537
SUPCON_PROJ_HIDDEN = 256
SUPCON_PROJ_DIM = 128                             # L2-normed 128-d projection

# ---------------------------------------------------------------------------
# Loss weights + temperatures (Decision 17 §17.4, Decision 37)
# ---------------------------------------------------------------------------
LOSS_W_CE = 1.0
LOSS_W_SUPCON_PHASE2 = 0.30
LOSS_W_CORAL_PHASE2 = 0.50
SUPCON_TAU = 0.07
FIELD_LOSS_WEIGHT = 8.0                           # CE + SupCon only (Decision 17 §17.4)

def phase1_supcon_weight(epoch: int) -> float:
    """Decision 37: ramp 0 → 0.30 across epochs 0-4."""
    return min(0.30, max(0.0, 0.30 * (epoch - 1) / 3.0))


# ---------------------------------------------------------------------------
# Decision 49: Deferred Pass-1 LoRA Scale Injection
# Pass-1 LoRA is BYPASSED (scale=0.0) for epochs 0-6 to let LoRA converge via Pass 2.
# Epochs 7-9 linearly ramp to 1.0. Epoch 10+ is full LoRA.
# ---------------------------------------------------------------------------
PASS1_LORA_RAMP_START = 7
PASS1_LORA_RAMP_END = 10


def compute_pass1_lora_scale(epoch: int | None,
                              ramp_start: int = PASS1_LORA_RAMP_START,
                              ramp_end: int = PASS1_LORA_RAMP_END) -> float:
    """Decision 49: Pass-1 LoRA scale schedule.

    epoch=None (inference): return 1.0 (LoRA merged at inference per Decision 31 §31.7).
    epoch < ramp_start: 0.0 (bypass LoRA in Pass 1).
    ramp_start <= epoch < ramp_end: linear ramp 0.0 → 1.0.
    epoch >= ramp_end: 1.0 (full Pass-1 LoRA).
    """
    if epoch is None:
        return 1.0
    if epoch < ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return 1.0
    return float(epoch - ramp_start) / float(ramp_end - ramp_start)

# ---------------------------------------------------------------------------
# Optimizer + scheduler (Decision 17 §17.3, Decision 30, Decision 31)
# ---------------------------------------------------------------------------
LR_LORA = 1e-4
WD_LORA = 0.01
LR_HEADS = 5e-4
WD_HEADS = 0.0
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 2
PATIENCE_STARTS_EPOCH = 4                         # Decision 30.5
LR_COSINE_MIN_RATIO = 0.1                         # cosine anneal to 0.1 × peak

# ---------------------------------------------------------------------------
# Fallback flag (Decision 23, 28)
# ---------------------------------------------------------------------------
FALLBACK_MAX_ATTN_THRESHOLD = 0.15
FALLBACK_ENTROPY_THRESHOLD_FRAC = 0.90            # × log(784) ≈ 5.97 nats
import math
FALLBACK_ENTROPY_THRESHOLD = FALLBACK_ENTROPY_THRESHOLD_FRAC * math.log(NUM_PATCHES)
FALLBACK_FLAG_COL_FREEZE_UNTIL_EPOCH = 3          # Decision 30.2
FALLBACK_FLAG_LR_MULT_EPOCH_4 = 0.1               # Decision 30.2

# ---------------------------------------------------------------------------
# Data pipeline (Decisions 16, 21, 25, 27, 30, 34)
# ---------------------------------------------------------------------------
PHASE1_BATCH_SIZE = 32                            # Decision 31 §31.6
PHASE2_BATCH_SIZE = 16                            # Decision 15
PHASE1_CLASS_SLOTS = [8, 8, 4, 4, 4, 4]           # 2× Phase 2 slots: total 32
PHASE2_CLASS_SLOTS = [4, 4, 2, 2, 2, 2]           # Decision 19: total 16
RECOMPOSE_PROB_NON_FLAGGED = 0.70                 # Decision 16
STOCHASTIC_TIGHT_CROP_PROB = 0.30                 # Decision 25
TIGHT_CROP_PAD = 0.15                             # 15% padding around bbox
LETTERBOX_PAD_VALUE = 114                         # Decision 30.1 grey
FIELD_SAMPLE_WEIGHT = 8.0                         # Decision 17 §17.5
FIELD_SAMPLE_WEIGHT_THIN = 4.0                    # Decision 24 cap
THIN_CLASS_THRESHOLD = 30                         # field_train_count threshold for thin class
AMPMIX_PROB = 0.45                                # Decision 38

# Standard augmentation (Decision 17 §17.5 / 30.3)
AUG_HFLIP_P = 0.5
AUG_AFFINE_ROTATE_DEG = 15
AUG_AFFINE_P = 0.5
AUG_COLOR_JITTER_BCS = 0.10                       # brightness/contrast/saturation
AUG_COLOR_JITTER_P = 0.5
AUG_RANDOM_RESIZED_CROP_SCALE = (0.82, 1.0)
AUG_RANDOM_RESIZED_CROP_RATIO = (0.95, 1.05)
AUG_RANDOM_RESIZED_CROP_P = 0.5

# Mixed precision + Windows (Decision 17 §17.3, CLAUDE.md)
MIXED_PRECISION = "bf16"                           # bf16 autocast, NO permanent cast
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = 0                                    # Windows + CUDA constraint

# ---------------------------------------------------------------------------
# Reproducibility (Decision 31 §31.4)
# ---------------------------------------------------------------------------
SEED = 42
PYTHONHASHSEED = "0"

# ---------------------------------------------------------------------------
# Stopping criterion (Decision 17 §17.6)
# ---------------------------------------------------------------------------
STOPPING_WEIGHTS = {
    "tomato_foliar_spot": 0.210886,
    "tomato_septoria_leaf_spot": 0.133376,
    "tomato_late_blight": 0.141466,
    "tomato_yellow_leaf_curl_virus": 0.057753,
    "tomato_mosaic_virus": 0.094311,
    "tomato_healthy": 0.362208,
}
DISEASE_F1_FLOOR = 0.30                           # Decision 17 §17.6
PHASE2_MIN_EPOCHS = 12
PHASE2_MAX_EPOCHS = 25
PHASE2_PATIENCE = 5
PHASE2_ROLLING_MEAN_EPOCHS = 3

# Phase 1 (head-only, fixed-duration per Decision 17 §17.6)
PHASE1_NUM_EPOCHS = 5                             # can extend to 8 if attention gate doesn't pass
PHASE1_MAX_EPOCHS = 8

# ---------------------------------------------------------------------------
# CORAL (Decision 17 §17.4, Decisions 26, 29, 31)
# ---------------------------------------------------------------------------
CORAL_EMA_DECAY = 0.9
CORAL_MIN_LAB_COUNT_IN_BATCH = 6                  # Decision 26
CORAL_TARGET_REFRESH_EPOCHS = 5                   # every 5 epochs
CORAL_WARMUP_STEPS = 2000                         # Decision 31 §31.5

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (timm DINOv2 default)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Config hash (for checkpoint audit — Decision 31 §31.1 / 32.2)
# ---------------------------------------------------------------------------
def config_hash() -> str:
    """md5 of the configuration's hyperparameter values for checkpoint drift detection."""
    items = [
        f"backbone={BACKBONE}",
        f"resolution={RESOLUTION}",
        f"num_classes={NUM_CLASSES}",
        f"lora_rank={LORA_RANK}",
        f"lora_alpha={LORA_ALPHA}",
        f"lora_target_blocks={LORA_TARGET_BLOCKS}",
        f"lora_impl={LORA_IMPL}",
        f"abmil_hidden={ABMIL_HIDDEN}",
        f"fusion_input_dim={FUSION_INPUT_DIM}",
        f"fusion_hidden={FUSION_HIDDEN}",
        f"supcon_proj_dim={SUPCON_PROJ_DIM}",
        f"loss_weights=ce1.0_supcon0.30_coral0.50",
        f"lr_lora={LR_LORA}_wd{WD_LORA}",
        f"lr_heads={LR_HEADS}_wd{WD_HEADS}",
        f"grad_clip={GRAD_CLIP_NORM}",
        f"warmup_epochs={WARMUP_EPOCHS}",
        f"p1_bs={PHASE1_BATCH_SIZE}_slots={PHASE1_CLASS_SLOTS}",
        f"p2_bs={PHASE2_BATCH_SIZE}_slots={PHASE2_CLASS_SLOTS}",
        f"recompose_p={RECOMPOSE_PROB_NON_FLAGGED}",
        f"tight_crop_p={STOCHASTIC_TIGHT_CROP_PROB}",
        f"letterbox_pad={LETTERBOX_PAD_VALUE}",
        f"field_weight={FIELD_SAMPLE_WEIGHT}_thin{FIELD_SAMPLE_WEIGHT_THIN}_thresh{THIN_CLASS_THRESHOLD}",
        f"ampmix_p={AMPMIX_PROB}",
        f"coral_ema_decay={CORAL_EMA_DECAY}_min_lab{CORAL_MIN_LAB_COUNT_IN_BATCH}_refresh{CORAL_TARGET_REFRESH_EPOCHS}_warmup{CORAL_WARMUP_STEPS}",
        f"fallback_attn_thresh={FALLBACK_MAX_ATTN_THRESHOLD}",
        f"fallback_entropy_thresh_frac={FALLBACK_ENTROPY_THRESHOLD_FRAC}",
        f"fallback_freeze_until_epoch={FALLBACK_FLAG_COL_FREEZE_UNTIL_EPOCH}",
        f"seed={SEED}",
    ]
    blob = "|".join(items).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


CONFIG_HASH = config_hash()

if __name__ == "__main__":
    print(f"LADI-Net config hash: {CONFIG_HASH}")
    print(f"  resolution={RESOLUTION}  bs_p1={PHASE1_BATCH_SIZE}  bs_p2={PHASE2_BATCH_SIZE}")
    print(f"  fusion_input_dim={FUSION_INPUT_DIM}")
    print(f"  fallback_entropy_threshold={FALLBACK_ENTROPY_THRESHOLD:.4f} nats")
