# MASTER PLAN — Plant Disease Detection System for Kerala Agriculture
# Version 5.1 (Post-Verification — 6 errors corrected)
# Generated: April 2026
# Source: Complete architecture conversation + pessimistic debug analysis

---

## HOW TO READ THIS DOCUMENT

This document is the single source of truth for the entire 3-model specialist
pipeline. Read every section before writing any code. Every decision is
justified. Every technique explains what it does, why it is in the plan, and
what happens if it fails.

Two sacred rules that override everything in this document:
1. NEVER modify: models/best_model.pt, models/swin_best_model.pt, app/config.py,
   data/metadata/source_map.csv
2. When in doubt between data quality and data quantity, choose quality.

---

## SECTION 1: PROJECT OVERVIEW

### 1.1 Goal

Build a farmer-facing plant disease detection web app for Kerala, India.
A farmer photographs a diseased leaf with any smartphone, uploads through a
browser, and receives: crop identification, disease diagnosis with calibrated
confidence, severity assessment, heatmap showing where the disease evidence is,
treatment recommendations, and an honest uncertainty flag when the model isn't
confident.

Four crops supported: okra (ladies finger), brassica (cabbage/broccoli/cauliflower),
tomato, and chilli.

Target deployment: FastAPI server + plain HTML/CSS/JS frontend on localhost.
No React, no cloud, no accounts. One machine: RTX 4060 Laptop GPU with 8GB VRAM.

### 1.2 System Architecture

Two operational modes exist:

**Mode 1 (Legacy):** Existing Swin-Tiny 23-class model (swin_best_model.pt, 114.9MB).
  Single model handles all 23 disease classes across all 4 crops.
  Validated at ~0.94 F1 on internal test but ~0.32 F1 on PlantDoc (real-world gap).
  This is the fallback if the specialist pipeline fails to meet acceptance criteria.

**Mode 2 (Specialist Pipeline — this plan):** Three specialised models in sequence:
  Step 1: **Crop Router** identifies which crop (4-class: okra/brassica/tomato/chilli)
  Step 2a: If okra or brassica → **Model 2 Specialist** (9-class disease diagnosis)
  Step 2b: If tomato or chilli → **Model 3 Specialist** (10-class disease diagnosis)

Why Mode 2 over Mode 1: The Swin model's 0.32 PlantDoc F1 shows that a single
monolithic model overfits to lab training data. The specialist pipeline allows
each model to be tuned to its specific data quality: Model 2 has 68% field data
(can fine-tune aggressively), Model 3 has 97% lab data (must preserve domain-
invariant features via LoRA). A monolithic model cannot have both strategies.

### 1.3 Hardware Constraints

```
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  VRAM: 8,188 MiB (8.0 GB)
  Compute capability: sm_89 (Ada Lovelace)
  BF16 tensor core performance: ~27 TFLOPS measured
  Steady-state 4096×4096 BF16 matmul: 5.08ms

CPU: Windows 11, Python 3.13 (miniconda3)
torch: 2.11.0+cu130 (CUDA 13.0), cold-start import: 2.3s (Defender exclusions applied)
DINOv2-Small + LoRA forward pass: 12.1ms at batch 4, 224px BF16 (330 img/s)

Effective batch sizes given VRAM:
  Router (frozen backbone + linear head): batch 64 at 224px = ~1.5GB
  Model 2 at 128px (full fine-tune ConvNeXt-S): batch 32 = ~4GB
  Model 2 at 224px (full fine-tune + ASAM 2x fwd-bwd): batch 16 = ~6GB
  Model 2 at 384px (full fine-tune + ASAM 2x fwd-bwd): batch 8 = ~7.5GB
  Model 3 at 224px (DINOv2 frozen + LoRA + grad ckpt): batch 16 = ~5GB
```

### 1.4 Sacred Files (Never Modify)

```
models/best_model.pt         — 84.2MB, 23-class EfficientNetV2-S teacher model
models/swin_best_model.pt    — 114.9MB, 23-class Swin-Tiny production model
app/config.py                — 23-class production config (used by both sacred models)
data/metadata/source_map.csv — 21,783-row historical record of original training data
```

All specialist training uses separate config files (config_model2.py, config_model3.py,
config_router.py) and separate unified CSV files. The sacred files are never read by
specialist training scripts.

---

## SECTION 2: DATA INVENTORY AND KNOWN ISSUES

### 2.1 Model 2 — Okra + Brassica (9 classes, 9,006 images)

| Class | Count | Field% | Top Source | Top% | Known Issues |
|---|---|---|---|---|---|
| okra_yvmv | 1,612 | 58% | gadde_okra | 40% | Good diversity (5 sources) |
| okra_powdery_mildew | 602 | 72% | yeesi | 72% | Single-source risk |
| okra_cercospora | 335 | 89% | bangladesh_okra | 89% | Single-source risk, marginal count |
| okra_enation | 288 | 100% | bangladesh_okra | 90% | PERMANENT THIN CLASS — epistemic ceiling |
| okra_healthy | 2,965 | 39% | okra_100 | 32% | Lab-heavy but well-diversified |
| brassica_black_rot | 1,080 | 96% | mendeley_caul_leaf | 70% | Single-source risk |
| brassica_downy_mildew | 338 | 93% | cauliflower_noam | 40% | Just crossed 300 threshold |
| brassica_alternaria | 723 | 82% | plantwild_C | 22% | Best diversity (9 sources) |
| brassica_healthy | 1,063 | 100% | mendeley_caul_leaf | 70% | Single-source risk |
| **TOTAL** | **9,006** | **68.2%** | | | |
| ~~brassica_clubroot~~ | ~~304~~ | | | | **QUARANTINED — root gall, not leaf** |

**Critical limitation: okra_enation (288 images, 90% single-source Bangladesh)**

This is an epistemic constraint, not an algorithmic one. The Bayesian posterior for
okra_enation as a disease class is genuinely wide given this evidence. No training
technique — SupCon, CutMix, ASAM, curriculum — can manufacture disease biology
information that does not exist in the training data. The model will learn what
Bangladeshi okra_enation looks like, which may differ from Kerala okra_enation.

Honest target F1: 0.55-0.68. Not 0.82.

The correct response is transparent uncertainty: conformal prediction sets will
naturally widen for this class, and the UI returns needs_verification=true with
a note recommending local agronomist comparison.

### 2.2 Model 3 — Tomato + Chilli (10 classes, ~32,243 after target_spot removal)

| Class | Count | Field% | Top Source | Top% | Known Issues |
|---|---|---|---|---|---|
| tomato_foliar_spot | 8,485 | 2.3% | scidb_data_merged | 63% | LAB DOMINATED — needs bg recomposition |
| tomato_late_blight | 4,133 | 2.2% | scidb_data_merged | 73% | LAB DOMINATED |
| tomato_septoria_leaf_spot | 3,279 | 2.4% | scidb_data_merged | 77% | LAB DOMINATED |
| tomato_yellow_leaf_curl_virus | 3,612 | 0.4% | scidb_data_merged | 63% | NEAR-ZERO FIELD — needs_verification always |
| tomato_mosaic_virus | 2,290 | 1.7% | scidb_data_merged | 77% | LAB DOMINATED |
| tomato_healthy | 1,657 | 35.8% | plantvillage_existing | 38% | Best tomato diversity |
| chilli_leaf_curl | 3,317 | 87.8% | figshare_leaf_curl | 68% | Single-source risk but field-heavy |
| chilli_healthy | 3,385 | 96.9% | multi_D | 27% | 928 Capsicum lab images (27%) |
| chilli_cercospora_leaf_spot | 1,432 | 45.7% | plantvillage_existing | 54% | Mixed field/lab |
| chilli_anthracnose | 653 | 58.2% | plantvillage_existing | 42% | Thinnest chilli class |
| **TOTAL** | **~32,243** | **~25%** | | | |
| ~~tomato_target_spot~~ | ~~539~~ | | | | **QUARANTINED — labelling suspect** |

**Why tomato_target_spot was removed:**
Visual inspection of the Tomato Leaf Multiclass dataset revealed that class 5
("target spot") annotations were dark spots (37×35px average) visually identical
to class 1 ("black spot"), with no concentric ring pattern characteristic of actual
target spot. 37 images were reclassified to foliar_spot. The remaining 432 original
images (80% PlantVillage, 97% lab, 2 sources only) have insufficient label reliability.
Training on these images actively degrades macro F1 by introducing label noise.
Quarantined at model3/cleaned/tomato_target_spot_QUARANTINED/.

**Capsicum contamination in chilli_healthy:**
928 of 3,385 chilli_healthy images are Capsicum annuum (bell pepper) lab photos from
Dataset D. While Capsicum is in the same genus as chilli, the leaf morphology differs
and the lab backgrounds create shortcut learning risk. Background recomposition (Phase 0)
converts these from contamination into augmentation by pasting Capsicum leaves onto
chilli field backgrounds. Per-subsource F1 monitoring during training detects if
shortcut learning persists.

### 2.3 Router (4 crops, ~45,158 images)

| Crop | Count | Field% | Top Source | Top% | Notes |
|---|---|---|---|---|---|
| okra | 7,366 | 45.9% | original_pool | 42% | Adequate |
| brassica | 4,269 | 92.3% | original_pool | 62% | Best field data |
| tomato | 24,055 | 10.7% | scidb_data_merged | 77% | CRITICALLY LAB-DOMINATED |
| chilli | 9,468 | 97.1% | model3_cleaned | 38% | Excellent field data |
| **TOTAL** | **45,158** | **42.2%** | | | Imbalance: 5.6:1 tomato:brassica |

### 2.4 Data Quality Events (Historical Record)

- PlantVillage duplicate datasets deleted (5 folders, 3.54 GB freed):
  tomato_cookiefinder, tomato_hakim, tomato_kaustubh, tomato_luisolazo, tomato_ashish
- 37 target_spot images reclassified to foliar_spot after visual verification
- 4 brassica_alternaria infographics deleted (red background, text overlays, cauliflower grocery photo)
- brassica_clubroot quarantined (304 images, root disease not diagnosable from leaves)
- Taiwan tomato integrated: 482 images (39% field, 279 disease + 203 router)
- iNaturalist tomato: already in pool (79 images, 100% field)
- Capsicum healthy images: 928 lab photos accepted with monitoring plan

---

## SECTION 3: PREPROCESSING PIPELINE (OFFLINE — RUN ONCE BEFORE TRAINING)

### 3.1 LAB-CLAHE (Contrast-Limited Adaptive Histogram Equalisation in LAB Colorspace)

**What it does:** Normalises brightness differences between photos taken in Kerala's
bright monsoon sun versus shade or indoor conditions, WITHOUT changing leaf colours.

**Why LAB not RGB:** Standard CLAHE applied to all three RGB channels independently
causes hue shifts — a green leaf can become bluish or yellowish because the R, G, B
adjustments are uncoordinated. LAB colorspace separates brightness (L channel) from
colour information (A=green-red axis, B=blue-yellow axis). Applying CLAHE only to
the L channel adjusts contrast and brightness while preserving the exact green-yellow
colour signature that distinguishes healthy tissue from disease lesions.

**Implementation:**
```python
import cv2
import numpy as np

def apply_lab_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # L channel only
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

**Output:** For each image in the unified CSVs, save CLAHE-processed version alongside
original. Add `clahe_path` column to CSVs. Training scripts use `clahe_path` by default,
fall back to `image_path` if CLAHE file missing.

### 3.2 Background Recomposition

**What it does:** Segments diseased tomato leaves from lab backgrounds using U2-Net
(via rembg library), pastes them onto randomly selected field backgrounds from
chilli and brassica datasets.

**Why this is the single highest-leverage technique in the plan:** Tomato disease
classes are 97-99% lab photos. A model fine-tuned on this data learns "gray background
= diseased tomato." Background recomposition bridges the lab-to-field domain gap by
keeping the exact disease signature (the segmented leaf) while replacing the confounding
variable (the background). Expected impact: tomato effective field% from 2% to ~40%
without collecting a single new photo.

**Which images:**
- All scidb_data_merged tomato disease class images (capped at 2,000 per class)
- All 928 Capsicum lab images in chilli_healthy (convert contamination to augmentation)
- Total: ~12,000 recompositions (10,000 tomato + 2,000 Capsicum, with some skipped for quality)

**Background pool:** Field images from:
- chilli_healthy (is_field_photo=True): outdoor chilli plant backgrounds
- chilli_leaf_curl (is_field_photo=True): outdoor backgrounds with foliage
- brassica_healthy (is_field_photo=True): outdoor vegetable garden backgrounds
- ~2,000 unique backgrounds; each recomposition randomly selects one

**Quality filter:** If U2-Net foreground mask covers <15% or >85% of the image, the
segmentation likely failed (too little leaf or too much background included). Skip
that image — keep original lab version in training. Log count of skipped images.

**Output:**
- Saved to `data/specialist/model3/recomposed/{class_name}/`
- `source_dataset='scidb_recomposed'` (or `capsicum_recomposed`)
- `is_field_photo=True`
- NOT counted as scidb for sampling cap purposes (separate category)
- Training split ONLY — never in validation or conformal calibration sets

### 3.3 Data Split Strategy

**Why 5-way split for specialists:**
Greedy model soup needs a held-out evaluation set separate from the validation set used
for early stopping (otherwise, model selection optimises for val distribution and soup
doubles down on it). Conformal prediction requires its own held-out set that the model
has never influenced. Using the same set for both early stopping and conformal calibration
violates conformal theory's exchangeability assumption, reducing the 95% coverage
guarantee to ~85% actual coverage.

**Strict exclusion rules:**
- Rule 1: Recomposed images → training split ONLY (validation measures real performance)
- Rule 2: Conformal indices → excluded from ALL other uses (including self-distillation soft label generation)

**Model 2 split (thin classes require combined eval sets for statistical power):**
```
68% training (includes recomposed images)
15% val_and_soup (for early stopping AND greedy soup selection)
12% final_validation (for final reported metrics)
 5% conformal_calibration (approximate — uses val set instead; see Known Limitations)
```
NOTE: Model 2 conformal calibration uses the val set (not a separate 5% split) because
okra_enation has only 288 images; 5% = 14 images is statistically meaningless for APS.
This is documented as a known limitation — the 95% coverage guarantee is weakened to
~92-94% for thin classes.

**Model 3 split (enough data for full 5-way):**
```
68% training (includes recomposed)
10% validation (for early stopping)
 7% soup_selection (for greedy soup)
10% final_validation (for final metrics)
 5% conformal_calibration (strict held-out, excluded from self-distillation)
```

**Router split (simpler):**
```
75% training
15% validation (for early stopping)
10% conformal_calibration (for routing thresholds)
```

All split indices are saved to `data/specialist/{model}/split_indices.json` BEFORE
any training begins.

**Source-aware stratification (CRITICAL — added from research analysis):**
Standard `StratifiedShuffleSplit(stratify=class_name)` only ensures class balance across
splits. It says nothing about SOURCE composition — a random split could accidentally put
85 of 89 field tomato photos in training, leaving validation entirely lab-based. This makes
val F1 meaningless for deployment performance estimation.

The correct approach uses a composite stratification key:
```python
df['source_bucket'] = assign_source_bucket(df)  # scidb | field | lab | recomposed
df['strat_key'] = df['class_name'] + '_' + df['source_bucket']
StratifiedShuffleSplit(stratify=df['strat_key'], ...)
```

Source buckets per model:
- Model 2: field_high_quality (>75% field class), field_mixed (25-75%), lab_heavy (<25%)
- Model 3: scidb_original, field_verified, lab_non_scidb, recomposed
- Router: per-crop source splits

For classes with 3+ distinct sources (≥10 images each), consider StratifiedGroupKFold
with groups=source_dataset for genuinely source-disjoint validation.

For thin single-source classes (okra_enation at 90% Bangladesh), source-disjoint splitting
is impossible. Val F1 for these classes measures within-distribution performance only.

**Source-aware conformal calibration:**
The conformal calibration split must ALSO use source-aware stratification. A conformal set
that is 97% lab produces thresholds calibrated on lab distribution, not field deployment.
Force proportional field/lab representation in conformal splits where possible.

**Gradient accumulation loss scaling (CRITICAL — confirmed bug):**
Naive gradient accumulation with `loss.backward()` and `grad_accum_steps=G` produces
gradients G times too large. The fix: `(loss / grad_accum_steps).backward()`.
Confirmed by: Unsloth blog, HuggingFace blog, multiple PyTorch framework patches.
Without this fix, effective LR is 4x too high with grad_accum=4, undermining ASAM
and potentially causing LoRA divergence. Implemented in train_utils.py Section X.

**Model wrapper classes (scripts/models.py):**
Three reusable model wrapper classes encapsulate backbone loading, head attachment,
forward pass, feature extraction, and stage transitions:
- `RouterDINO`: DINOv2-Registers (timm), frozen, Linear(384→4)
- `Model2ConvNeXt`: DINOv3-ConvNeXt-Small (transformers), full FT, Linear(768→9)
- `Model3DINOLoRA`: DINOv2-Small (timm) + LoRA (peft) + FiLM, Linear(384→10)
Training scripts import from scripts/models.py — never build models inline.

### 3.4 Sampling Weight Computation

**Why target-based, not multiplicative:**
The original multiplicative formula `weight = (7300/total_crop) × field_mult × scidb_cap`
produces wrong distributions because WeightedRandomSampler normalises all weights to
a probability distribution. The interaction between three multiplied factors is non-obvious.
Actual measurement showed 912 scidb images/epoch instead of the intended 3,000.

**Correct approach: explicit per-bucket target counts.**

```python
# Router sampling targets per epoch (29,200 total = 7,300 per crop)
ROUTER_TARGETS = {
    'okra':     {'total': 7300, 'field': 4500, 'lab': 2800, 'scidb': 0},
    'brassica': {'total': 7300, 'field': 6700, 'lab':  600, 'scidb': 0},
    'tomato':   {'total': 7300, 'field': 3000, 'lab': 1300, 'scidb': 3000},
    'chilli':   {'total': 7300, 'field': 7100, 'lab':  200, 'scidb': 0},
}
# Per-image weight = target_count_for_my_bucket / actual_count_in_my_bucket
```

**Recomposed images:** Categorised as field, NOT scidb. They get the field multiplier
and are not subject to the scidb cap. This is intentional — recomposed images are the
highest-value synthetic data.

**Monte Carlo verification:** Before any training run, simulate 30 epochs of sampling
and print mean±std per bucket. All buckets must be within 10% of their targets. This
verification takes ~30 seconds and is MANDATORY.

---

## SECTION 4: CROP ROUTER — COMPLETE SPECIFICATION

### 4.1 Architecture

**Backbone:** DINOv2-Small with Registers (22M parameters, FROZEN)
  - If available in timm: `vit_small_patch14_reg4_dinov2.lvd142m`
  - If not available in timm: `transformers.AutoModel.from_pretrained('facebook/dinov2-small-with-registers')`
  - Phase 0 verifies which library provides this model
  - CRITICAL: `img_size=224` must be passed if using timm (default 518 crashes)

**Head:** Linear classifier (384 → 4)
  - Only 1,536 trainable parameters
  - Backbone contributes 22M frozen parameters

**Why frozen DINOv2 for the router:**
Tomato is 89% lab images. Full fine-tuning would teach the backbone 'gray background =
tomato'. Frozen DINOv2 features are domain-invariant from 142M image self-supervised
pretraining. The linear head must learn crop morphology from the frozen features only —
it cannot learn background-based shortcuts because the backbone isn't adapting to them.

### 4.2 Training Recipe

```
Optimizer: AdamW (not ASAM — linear head is convex, SAM adds no value)
Learning rate: 1e-3, cosine schedule with 5-epoch warmup
Weight decay: 0.01 (applied to linear head weights only, not bias)
Batch size: 64 effective (32 actual × grad_accum=2)
Epochs: 20
Precision: BF16 autocast
EMA: decay=0.9999
Loss: CrossEntropyLoss with ENS class weights + label_smoothing=0.1
Sampler: WeightedRandomSampler with target-based weights (Section 3.4)
Augmentation: AugMix (standard — no GridDistortion for router)
num_workers: 0 (Windows)
Rollback: save at epoch 5, trigger at epoch 8 if val F1 < 95% of epoch 5 F1
Soup checkpoints: epochs 10, 12, 14, 16, 18 (save EMA, greedy select)
```

**Not included:** FiLM (no crop conditioning needed — router IS the crop classifier),
SupCon (only 4 classes, CE sufficient), MC Dropout (router uses TTA only), CutMix
(4 classes all have 4,000+ images, no thin classes).

### 4.3 Inference

- TTA: 5 views (original, h-flip, v-flip, brightness±15% jitter, ±15° rotation)
- Average softmax across 5 views
- Apply per-crop routing threshold (from conformal calibration)
- If max probability < threshold[crop]: abstain ("unable to identify crop")
- NOT prediction sets (different from specialist conformal)

### 4.4 Expected Performance

- Target macro F1: ≥ 0.85
- Per-crop: okra ~0.87, brassica ~0.90, tomato ~0.82 (lab dominance), chilli ~0.93
- Abstention rate: 2-5% (calibrated to 95% correct routing on non-abstained)
- Training time: ~1 hour

---

## SECTION 5: MODEL 2 — OKRA+BRASSICA — COMPLETE SPECIFICATION

### 5.1 Architecture

**Backbone:** DINOv3-ConvNeXt-Small, full fine-tune (49.5M parameters trainable)
  Loaded via transformers: `AutoModel.from_pretrained('facebook/dinov3-convnext-small-pretrain-lvd1689m')`
  NOT via timm (HF config incompatible). Fallback: `convnext_small.fb_in22k_ft_in1k_384` (timm)
  - Primary: `convnext_small.fb_in22k_ft_in1k_384` (IN22k pretrained, timm verified)
  - Fallback 1: `convnext_small.in12k_ft_in1k_384` (IN12k pretrained)
  - Fallback 2: `convnext_small` (IN1k pretrained, least powerful but guaranteed)
  - Phase 0 verifies which pretrained variant is available with `pretrained=True`
  - Use `timm.create_model(name, pretrained=True, num_classes=9)` — timm handles
    the GAP + linear head internally. Do NOT use features_only=True (unnecessary
    complexity since FPN is not used).
  - FPN channels [192, 384, 768] documented for reference only — NOT used in pipeline.
  - GradCAM++ targets the last convolutional stage output (before GAP).

**Head:** Built-in by timm: Global average pooling → Linear (768 → 9)

**Why full fine-tune ConvNeXt (not frozen DINOv2+LoRA):**
Model 2 has 68% field data — high enough quality for full fine-tuning without
destroying pretrained features. Full fine-tuning allows the backbone to learn
Kerala-specific disease morphology: the exact texture, shape, and colour profiles
of each crop's disease symptoms. LoRA would preserve more pretraining but miss
these domain-specific features.

**Why ConvNeXt not ViT:**
ConvNeXt supports GradCAM++ natively (convolutional feature maps produce spatial
activation gradients). GradCAM++ handles multiple disease spots on the same leaf,
which standard GradCAM misses. ViT-based models need DINO attention maps instead
(different implementation, less tested for disease localisation).

**GradCAM++ specification:**
Target layer: final ConvNeXt stage (stage 3, output channels=768)
Handles multiple lesion instances per leaf (pixel-wise gradient weighting)
Returns 12×12 spatial heatmap at 384px input, upsampled to original resolution

### 5.2 Training Recipe — Stage 1 (Epochs 1-25)

**Progressive resizing (coarse-to-fine learning):**
```
Epochs 1-5:   128px, batch 32, ASAM disabled (warmup)
Epochs 6-7:   224px, batch 16, ASAM disabled (resolution transition warmup)
Epochs 8-15:  224px, batch 16, ASAM enabled (ρ=0.10)
Epochs 16-17: 384px, batch 8,  ASAM disabled (resolution transition warmup)
Epochs 18-25: 384px, batch 8,  ASAM enabled (ρ=0.20), grad_accum=4 (effective batch 32)
```

**ASAM (Adaptive Sharpness-Aware Minimisation):**
Does two forward-backward passes per step: first finds which direction would make
the loss worse, second takes a step avoiding that direction. Result: the model
converges to flatter minima in the loss landscape, which generalise better to
unseen Kerala field photos. Disabled for 2 epochs after each resolution transition
because the loss landscape shifts dramatically when image resolution changes.

**LLRD (Layer-wise Learning Rate Decay):**
decay_rate=0.90 per ConvNeXt block. Early layers (already excellent from pretraining)
get lower LR; later layers (task-specific) get higher LR; classification head gets
the base LR. Prevents disturbing useful low-level feature detectors.

**SupCon auxiliary loss (epochs 1-15 ONLY, NOT at 384px):**
λ=0.10, temperature=0.10. Pulls all okra_enation images together in embedding space,
pushes them away from other classes. For a class with only 288 images, this contrastive
signal shapes the feature space. Requires class-balanced batches (≥2 per class = 18
minimum), which is feasible at batch 32 (128px) and batch 16 (224px) but NOT at
batch 8 (384px). Therefore SupCon is disabled at 384px epochs.

**Other training details:**
- Loss: CrossEntropyLoss + label_smoothing=0.1 + ENS class weights (β=0.9999)
- Augmentation: AugMix (standard for all classes)
- EMA: decay=0.9999
- BF16 autocast
- Rollback: save checkpoint at end of EACH resolution stage (epoch 5, 15, 25).
  CRITICAL: Do NOT compare F1 values across resolutions (128px F1 is always lower than
  384px F1 due to information loss — comparing them falsely triggers rollback).
  Use `resolution_aware_rollback_check()` from train_utils.py:
    Within 128px stage: check at epoch 8 vs epoch 5 baseline (same resolution)
    Within 224px stage: check at epoch 18 vs epoch 15 baseline (same resolution)
    Within 384px stage: check at epoch 28 vs epoch 25 baseline (same resolution)
  At resolution TRANSITIONS (epoch 5→6, 15→16): no rollback check — F1 drop expected

### 5.3 Training Recipe — Stage 2 (Epochs 26-32)

**Freeze backbone. Train classification head only.**

```
Resolution: 384px (no change from end of Stage 1)
Batch size: 8 + grad_accum=4
Optimizer: NEW AdamW (reinitialised — no stale Stage 1 momentum)
LR: 1e-4 (lower than Stage 1 base LR)
EMA: decay=0.999 (faster than Stage 1 — shorter training horizon)
Epochs: 7
```

**CutMix for thin classes (okra_enation + okra_cercospora ONLY):**
Probability 0.3. When these classes appear in a batch, 30% of the time paste a patch
from the thin-class image onto another okra image with area-weighted mixed labels.
NOT compatible with SupCon (mixed labels break positive pair definition) — this is
why CutMix is Stage 2 only.

**Balanced sampling:** ~432 images per class per epoch (1.5× okra_enation count).
Light oversampling of the thinnest class (each image seen ~1.5× per epoch).

**Loss:** Focal Loss γ=2 (concentrates training effort on hard examples)

### 5.4 Expected Performance

- Target macro F1: ≥ 0.82
- okra_enation: F1 = 0.55-0.68 (honest ceiling, needs_verification=true always)
- All other classes: F1 ≥ 0.78
- Training time: ~4-5 hours (progressive resize + ASAM doubles compute)

---

## SECTION 6: MODEL 3 — TOMATO+CHILLI — COMPLETE SPECIFICATION

### 6.1 Architecture

**Backbone:** DINOv2-Small (22M parameters, FROZEN)
  - timm name: `vit_small_patch14_dinov2.lvd142m`
  - CRITICAL: `img_size=224` MUST be passed to timm.create_model (default 518 crashes)
  - Patch size: 14×14
  - Token count: 224/14 = 16 → 16×16 = 256 patch tokens + 1 CLS token = 257 total
  - Embedding dimension: 384

**LoRA adapters:** rank=8, alpha=16, target modules=['qkv']
  NOTE: timm DINOv2 uses fused QKV projection (single nn.Linear named 'qkv'),
  NOT separate query/value modules. Using ['query','value'] matches zero modules
  and trains no LoRA parameters — a critical silent failure.
  - 230K trainable parameters (1.07% of 21.87M total)
  - Preserves DINOv2's domain-invariant features while learning disease boundaries
  - Applied via peft library: `get_peft_model(backbone, LoraConfig(...))`

**FiLM conditioning on LoRA adapter OUTPUTS:**
  - crop_embedding_dim=4 (sufficient for binary tomato/chilli signal)
  - Wraps peft LoRA layer output (not injected into peft internals)
  - gamma initialised near 1 (near-identity), beta near 0
  - Tells LoRA layers which direction to specialise (tomato-specific vs chilli-specific)

**Classification head:** Linear (384 → 10) trained alongside LoRA

**Why LoRA on frozen DINOv2 (not full fine-tune):**
Five tomato disease classes have 97-99.6% lab photos. Full fine-tuning any backbone on
this data destroys the domain-invariant features that make DINOv2 useful on real farmer
photos. LoRA trains only 800K parameters: learns disease classification boundaries
without corrupting the 22M parameters of domain-invariant visual representation.

**Heatmaps: DINO self-attention maps (NOT GradCAM):**
GradCAM requires convolutional feature maps — ViT doesn't have them. DINOv2's
self-attention heads spontaneously learn foreground segmentation during pretraining.
Extract from last transformer block (block 11 of 12), CLS token attention to
256 patch tokens, averaged across all 6 attention heads, reshaped to 16×16 spatial
map, upsampled to original image dimensions with bilinear interpolation. No gradient
computation needed.

**Gradient checkpointing:** Enabled to reduce VRAM from ~7GB to ~5GB. Trades 25-30%
more compute per step for 40% less memory. Allows batch 16 at 224px.

### 6.2 Training Recipe — Stage 1 First Pass (Epochs 1-25)

**Curriculum learning:**
```
Epochs 1-8 (Phase 1):  Only field photos + images from classes with ≥3 unique sources
                       (~70% of data). Clearest, most reliable training examples.
Epochs 9-25 (Phase 2): All images including scidb lab and recomposed synthetic field.
```

**Training details:**
```
Optimizer: AdamW (not ASAM — only 800K params, ASAM's perturbation is noise at this scale)
LR: 1e-4, cosine schedule with 5-epoch warmup
Weight decay: 0.01 (excluded from bias and LayerNorm via parameter groups)
Batch size: 16, grad_accum=4 (effective batch 64)
Precision: BF16 autocast
EMA: decay=0.9999
Loss: Focal Loss γ=2 with ENS class weights
Sampler: WeightedRandomSampler (scidb cap 1000/class, field 4x, recomposed uncapped)
Augmentation: AugMix (all classes) + GridDistortion+ElasticTransform (curl classes only)
CutMix: for chilli_anthracnose only (p=0.3, thinnest class at 653 images), enabled after epoch 12
gradient_checkpointing: enabled
num_workers: 0 (test 2 in Phase 0; use if stable)
```

**Capsicum shortcut monitoring:**
At every validation epoch, compute F1 separately for Capsicum-source chilli_healthy
vs real-chilli-source chilli_healthy. If gap > 0.20, halve Capsicum sampling weight
and continue training (adaptive intervention, not restart).

**Rollback:** Save at epoch 3, trigger at epoch 8 if val F1 < 95% of epoch 3 F1.

### 6.3 Training Recipe — Stage 2 First Pass (Epochs 26-32)

**Freeze backbone AND LoRA. Train classification head + FiLM only.**

```
Optimizer: NEW AdamW (reinitialised)
LR: 5e-5
EMA: decay=0.999 (faster for shorter stage)
Epochs: 7
Loss: Focal Loss γ=1.5 (slightly less aggressive)
Sampling: Balanced at 653/class (chilli_anthracnose count)
No CutMix in Stage 2 (head-only retraining, don't need it)
```

### 6.4 Self-Distillation (Second Training Pass)

**Decision gate:** Only run if first pass macro F1 ≥ 0.70 AND per-class agreement_rate:
  - Overall agreement_rate ≥ 0.70
  - Minimum per-class agreement_rate ≥ 0.50
  If any class has <50% agreement, use hard labels for ALL images in that class.

**Soft label generation:**
Run first-pass model on all training images. For each image:
- If first-pass prediction agrees with hard label AND confidence > 0.70:
  use soft probability distribution as label (preserves inter-class relationships)
- If first-pass prediction disagrees OR confidence ≤ 0.70:
  use one-hot hard label (prevents propagating first-pass errors)
- EXCLUDE conformal calibration indices from soft label generation entirely

**Temperature:** 3.0 (not 2.0 — Focal Loss sharpens first-pass logits more than CE)

**Loss for second pass:** KL divergence for soft-label images, CrossEntropyLoss
for hard-label images, combined in a mixed loss function.

**Second pass training:** Identical recipe to first pass (same LR, same augmentation,
same curriculum). Additional time: ~1.5 hours.

### 6.5 Expected Performance

- Target macro F1: ≥ 0.72
- Tomato classes: F1 ≥ 0.58/class (lab dominance limits ceiling)
- Chilli classes: F1 ≥ 0.76/class
- tomato_yellow_leaf_curl_virus: needs_verification=true always (16 field photos)
- Capsicum gap: < 0.15 throughout training
- Self-distillation lift: +1-3% macro F1 if agreement conditions met
- Training time: ~1.5 hours first pass + ~1.5 hours self-distillation = ~3 hours total
  (Based on verified throughput: 330 img/s DINOv2+LoRA at batch 4 on RTX 4060.
   32,243 images × 25 epochs ÷ ~200 effective img/s with backward = ~67 min/pass.
   Previous "3 hours first pass" estimate was pre-benchmark and incorrect.)

---

## SECTION 7: POST-TRAINING PIPELINE

### 7.1 Greedy Model Soup

Save EMA checkpoints at 5 intervals near convergence:
- Model 2: epochs 25, 27, 29, 31, 32
- Model 3 first pass: epochs 17, 19, 21, 23, 25
- Model 3 second pass: same relative epochs

Selection algorithm: start with best single checkpoint, add each candidate only if
it improves val F1 by > 1e-4. Zero inference cost — one model, merged weights.
  Model 2: soup selection uses the SAME val_and_soup set as early stopping (by design
    — thin classes need statistical power, a separate soup split would have <15 images
    per thin class). This is a known approximation.
  Model 3: soup selection uses the DEDICATED 7% soup_selection split (separate from
    the 10% val set used for early stopping). This is methodologically proper.

### 7.2 Test-Time Augmentation (TTA)

Router: 5 views (original, h-flip, v-flip, brightness±15% jitter, ±15° rotation)
Specialists: 8 views (above 5 + 3 random crops at different positions)
Weighted average softmax probabilities across views (not uniform — original view
gets higher weight because it hasn't been geometrically distorted):
`weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05]`
(Original 25%, flips 15% each, random crops 5% each)
"Free accuracy" — 1-3% improvement, 1-line implementation.
NOTE: The input image is ALREADY LAB-CLAHE processed at Step 1 of inference.
A "CLAHE-enhanced" TTA view would apply double-CLAHE, degrading disease texture.
Use brightness jitter instead (multiply pixel values by uniform(0.85, 1.15)).

### 7.3 MC Dropout Uncertainty (Specialists Only)

5 forward passes with model.train() (dropout active, no gradients).
Returns mean prediction (more reliable) + std deviation (uncertainty estimate).
Confidence tier assignment: HIGH if std < 0.05, MODERATE if < 0.15, LOW if ≥ 0.15.

### 7.4 Conformal Prediction

**Specialists — APS (Adaptive Prediction Sets):**
Produces prediction SETS with 95% coverage guarantee. Sorts classes by probability,
accumulates until threshold q_hat. q_hat calibrated on held-out conformal set.
Farmer sees: {tomato_late_blight, tomato_foliar_spot} — honest uncertainty.

**Router — Abstention thresholds:**
Per-crop scalar confidence threshold. If max probability < threshold: abstain.
Farmer sees: "unable to identify crop, please retake photo from closer distance."

### 7.5 Kerala Tier-3 Evaluation

30-50 photos from Kerala geography, photographed in real field conditions.
Sources: iNaturalist Kerala GPS box, local markets, agricultural extension service.
Collected BEFORE Model 3 training (mandatory prerequisite).
The ONLY measure that tells you if the model works for actual Kerala farmers.

---

## SECTION 8: DEPLOYMENT LAYER

### 8.1 Inference Pipeline (app/inference.py)

```
On startup:
  Load router.pt, model2_specialist.pt, model3_specialist.pt
  Load conformal threshold files

Per farmer request:
  Step 1: LAB-CLAHE the uploaded image
  Step 2: Router TTA (5 views) → crop probabilities → top-1 crop
  Step 3: Check routing threshold → if below: return abstention
  Step 4: Dispatch to Model 2 (okra/brassica) or Model 3 (tomato/chilli)
  Step 5: Specialist TTA (8 views) + MC Dropout (5 passes) → mean + uncertainty
  Step 6: APS conformal → prediction set
  Step 7: Assign confidence_tier (HIGH/MODERATE/LOW)
  Step 8: Set needs_verification flag per class config
  Step 9: Generate heatmap (GradCAM++ for M2, DINO attention for M3)
  Step 10: Return structured JSON response
```

### 8.2 Farmer Response JSON

```json
{
  "crop_identified": "tomato",
  "crop_confidence": 0.91,
  "diagnosis": "tomato_late_blight",
  "top1_confidence": 0.73,
  "prediction_set": ["tomato_late_blight", "tomato_foliar_spot"],
  "prediction_set_size": 2,
  "uncertainty": 0.18,
  "needs_verification": false,
  "confidence_tier": "MODERATE",
  "heatmap_url": "/heatmap/{request_id}",
  "reasoning_note": null,
  "routing_note": null
}
```

### 8.3 API Endpoints

- POST /predict — accepts image file, returns diagnosis JSON
- GET /heatmap/{request_id} — returns heatmap image
- POST /feedback — farmer submits correction (logged for retraining)

### 8.4 Feedback Loop

When a farmer submits a correction or an agronomist verifies/corrects a diagnosis,
that image-label pair is logged to a database. When 50+ verified Kerala images are
accumulated, the models can be retrained on locally-grounded data. This is how
production agricultural AI actually improves — through deployment feedback, not
pre-training algorithmic tricks.

---

## SECTION 9: train_utils.py — FUNCTION REFERENCE

### Section A: Checkpointing
- `save_checkpoint(epoch, model, ema_model, optimizer, scheduler, scaler, best_f1, path)`
  Saves full training state including RNG states for reproducible resume. Sampler seed
  saved as epoch-dependent value.
- `load_checkpoint(path, model, ema_model, optimizer, scheduler, scaler)`
  Restores state, calls torch._dynamo.reset() to clear compile cache on resume.
- `find_latest_checkpoint(ckpt_dir, prefix)`
  Finds most recent checkpoint by epoch number for auto-resume.

### Section B: EMA
- `setup_ema(model, decay=0.9999)` — returns timm ModelEmaV2 instance
- `reset_ema(ema_model, model, new_decay=0.999)` — reinitialise for Stage 2

### Section C: Model Soup
- `greedy_soup(checkpoints, val_loader, device, improvement_threshold=1e-4)`
  Greedy selection: adds checkpoint only if it improves val F1 over current soup.

### Section D: Conformal Prediction
- `compute_aps_thresholds(model, cal_loader, alpha=0.05)` — specialists
- `compute_routing_thresholds(model, cal_loader, alpha=0.05)` — router
- `predict_with_aps(probs, threshold)` — returns prediction set
- `predict_with_routing(probs, thresholds)` — returns top-1 + abstain decision

### Section E: Augmentation
- `get_augmentation_pipeline(class_name, curl_classes, img_size)`
  Returns AugMix + GridDistortion if class is in curl_classes, else AugMix only.

### Section F: CutMix
- `apply_cutmix(images, labels, alpha=1.0, thin_class_indices=None, probability=0.3)`
  Only applies to specified thin classes. Returns mixed images + soft labels.

### Section G: Capsicum Monitoring
- `track_subsource_f1(model, val_loader, source_column, target_class)` — per-source F1
- `adaptive_capsicum_intervention(current_gap, sampling_weights, capsicum_indices)`
  Halves Capsicum weight if gap > 0.20.

### Section H: Rollback
- `should_rollback(current_f1, rollback_f1, threshold=0.95)` — returns bool
- `apply_rollback(model, optimizer, ema, rollback_path, lr_reduction=0.2)`
- `resolution_aware_rollback_check(current_f1, stage_f1s, current_stage)` — Model 2

### Section I: Sampling Verification
- `verify_sampling_weights(df, weights, targets, epochs=30, samples_per_epoch=29200)`
  Monte Carlo simulation. Prints mean±std per bucket. MANDATORY before training.

### Section J: Data Loading
- `load_split(csv_path, split_name, exclude_indices=None, exclude_sources=None)`
  Returns filtered DataFrame. Respects conformal exclusion rules.

### Section K: Self-Distillation
- `generate_soft_labels(model, dataset, device, agreement_threshold=0.70, temperature=3.0)`
  Agreement mask filter. Returns soft labels + use_soft boolean array.
  Per-class agreement check: classes below 50% get hard labels entirely.

### Section L: Evaluation
- `evaluate(model, val_loader, device, class_names)` — per-class F1, macro F1
- `evaluate_with_subsource(model, val_loader, device, source_column)` — per-source F1

### Section M: Compilation
- `compile_model_safe(model, mode='default')` — torch.compile with error handling
  Falls back to uncompiled if compile fails. Calls torch._dynamo.reset() on resume.

### Section N-O: TTA and MC Dropout
- `predict_with_tta(model, image, views=5)` — Router TTA
- `predict_with_mc_dropout(model, image, passes=5)` — Specialist uncertainty

### Section P: DINO Attention Maps
- `extract_dino_attention_map(model, image, block_idx=11, head_average=True)`
  CLS attention to 256 patches, averaged across 6 heads, 16×16 → upsampled.

### Section Q: SupCon Loss
- `SupConLoss(temperature=0.10)` — nn.Module, supports class-balanced batches

### Section R: ASAM Optimizer + Parameter Groups
- `ASAMWrapper(base_optimizer, model, rho=0.10)` — Adaptive Sharpness-Aware Minimisation
  Wraps any base optimizer (AdamW). Two-pass per step: (1) perturb weights by rho
  in steepest gradient direction, (2) compute loss at perturbed point, (3) step in
  direction that avoids sharp minima. Used in Model 2 Stage 1 at 224px (rho=0.10)
  and 384px (rho=0.20). NOT used in Router or Model 3. Implementation: ~50 lines
  if `asam-optimizer` pip package unavailable. Install: `pip install asam-optimizer`
- `get_param_groups_no_decay(model, weight_decay=0.01)` — excludes bias + LayerNorm
- `get_llrd_param_groups(model, base_lr, decay_rate=0.90)` — ConvNeXt LLRD

### Section S: Stage Transitions
- `freeze_backbone(model)`, `unfreeze_all(model)`, `switch_to_stage2(model, head_only=True)`

### Section T: FiLM Module
- `FiLMWrapper(nn.Module)` — wraps LoRA layer output with crop-conditioned scale+shift

### Section U: Mixed Loss
- `soft_hard_mixed_loss(logits, soft_targets, is_soft, temperature=3.0)`
  KL for soft targets, CE for hard targets, combined.

### Section V: ENS Weights
- `compute_ens_class_weights(class_counts, beta=0.9999)` — returns per-class tensor

---

## SECTION 10: EXECUTION CHECKLIST AND ORDER

```
Phase 0 — Prerequisites (estimated: 6-8 hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] 0.1  Update config_model3.py: remove tomato_target_spot, NUM_CLASSES=10
         Verify: assert_config_consistency() passes
[ ] 0.2  Quarantine tomato_target_spot/ → tomato_target_spot_QUARANTINED/
         Verify: directory renamed, no active tomato_target_spot in cleaned/
[ ] 0.3  Rerun rebuild_unified_csvs.py to regenerate all 3 unified CSVs
         Verify: model3 CSV has 10 classes, no tomato_target_spot rows
[ ] 0.4  Verify DINOv2-with-registers availability (timm vs transformers)
         Verify: model creation succeeds with pretrained weights
[ ] 0.5  Verify ConvNeXt-Small pretrained download works
         Verify: timm.create_model('convnext_small.fb_in22k_ft_in1k_384', pretrained=True)
[ ] 0.6  Test torch.compile + gradient_checkpointing compatibility
         Verify: DINOv2+LoRA forward+backward pass succeeds with both enabled
[ ] 0.7  Install rembg and test background segmentation on 5 images
         Verify: rembg produces reasonable foreground masks
[ ] 0.7b Install ASAM optimizer: pip install asam-optimizer
         Verify: python -c "from asam import ASAM; print('ASAM OK')"
         If asam-optimizer unavailable: implement ASAM wrapper in train_utils.py
         Section R (~50 lines, wraps any base optimizer with sharpness-aware step)
[ ] 0.8  Test num_workers=2 DataLoader on Windows
         Verify: 100-image training loop completes without crash
[ ] 0.9  Run LAB-CLAHE offline on all training images
         Verify: clahe_path column added to all unified CSVs
[ ] 0.10 Run background recomposition for tomato scidb + Capsicum
         Verify: recomposed images saved, source_dataset='scidb_recomposed'
[ ] 0.11 Compute 5-way data splits, save indices to JSON
         Verify: all splits sum to total, conformal indices disjoint from all others
[ ] 0.12 Compute sampling weights + Monte Carlo verification
         Verify: all buckets within 10% of targets over 30-epoch simulation
[ ] 0.13 Write acceptance_criteria.json
         Verify: all models have min F1, per-class mins, action_below_minimum
[ ] 0.14 Write train_utils.py and test all functions
         Verify: import succeeds, checkpoint round-trip, EMA creation, soup on dummy
[ ] 0.15 Write simple fallback scripts and run 3-epoch test
         Verify: each fallback produces a saved model file
[ ] 0.16 Collect 30+ Kerala tier-3 images (iNaturalist + markets)
         Verify: images saved to data/kerala/ with class labels

Phase 1 — Router Training (~1 hour)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] 1.1  Run router training (20 epochs)
         Verify: macro F1 ≥ 0.85 on val set
[ ] 1.2  Greedy model soup
         Verify: soup F1 ≥ single best checkpoint F1
[ ] 1.3  Compute routing conformal thresholds
         Verify: thresholds saved to JSON
[ ] 1.4  Save router.pt
         If FAIL (F1 < 0.85): investigate, adjust weights, retrain

Phase 2 — Model 2 Training (~4-5 hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] 2.1  Run Model 2 simple fallback first (2 hours)
         Verify: macro F1 ≥ 0.78, okra_enation ≥ 0.50
[ ] 2.2  If fallback acceptable: run full Model 2 (Stage 1 + Stage 2)
         Verify: macro F1 ≥ 0.82, okra_enation ≥ 0.55
[ ] 2.3  Greedy model soup
[ ] 2.4  Compute APS conformal thresholds (using val set, approximate)
[ ] 2.5  Save model2_specialist.pt
         If FAIL: use simple fallback model

Phase 3 — Model 3 Training (~4.5 hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] 3.1  Run Model 3 simple fallback first (1.5 hours)
         Verify: macro F1 ≥ 0.68, Capsicum gap < 0.15
[ ] 3.2  If fallback acceptable: run full Model 3 first pass (1.5 hours)
         Verify: macro F1 ≥ 0.72, agreement_rate ≥ 0.70
[ ] 3.3  If decision gate passes: run self-distillation second pass (1.5 hours)
         Verify: second pass F1 > first pass F1
[ ] 3.4  Greedy model soup
[ ] 3.5  Compute APS conformal thresholds (proper 5% split)
[ ] 3.6  Save model3_specialist.pt
         If FAIL: use simple fallback model
         If BOTH fail: deploy Mode 1 (Swin-Tiny 23-class)

Phase 4 — Deployment (~3 days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] 4.1  Write app/inference.py (routing + dispatch + TTA + MC Dropout)
[ ] 4.2  Write app/api.py (FastAPI endpoints)
[ ] 4.3  Write app/feedback.py (farmer submission logging)
[ ] 4.4  Write confidence tier UI (HTML/CSS/JS)
[ ] 4.5  Integration test: upload test image, verify full pipeline response

Phase 5 — Tier-3 Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] 5.1  Run all 3 models on Kerala tier-3 images
[ ] 5.2  Document val F1 vs tier-3 F1 gap
[ ] 5.3  If tier-3 F1 acceptable: deploy to 10 test farmers
[ ] 5.4  If tier-3 F1 unacceptable: deploy Mode 1, collect more Kerala data
```

---

## SECTION 11: KNOWN LIMITATIONS AND HONEST EXPECTATIONS

### Permanent Data Limitations
1. **okra_enation** (288 images, single source): F1 ceiling 0.55-0.68. No fix without Kerala data collection.
2. **tomato_yellow_leaf_curl_virus** (16 field photos): needs_verification always true.
3. **Tomato disease classes 97-99% lab**: background recomposition mitigates but cannot fully close the domain gap.
4. **Capsicum in chilli_healthy** (27%): monitored and recomposed, but genuine chilli-vs-capsicum morphological differences remain.

### Methodological Limitations
5. **Model 2 conformal calibration** uses val set (not separate split): 95% coverage weakened to ~92-94% for thin classes.
6. **Self-distillation** propagates first-pass biases despite agreement mask filter: majority-class soft labels are more informative than thin-class soft labels.
7. **Background recomposition** introduces synthetic artefacts at segmentation boundaries: some recomposed images will have visible cut-and-paste seams.

### Infrastructure Limitations
8. **num_workers=0 on Windows**: CPU data loading is the training bottleneck for Model 3 (32K images). GPU utilisation ~60-70% instead of >90%.
9. **8GB VRAM**: Phase 0 batch testing showed batch 16 at 384px = 4.62 GB (fits with ASAM).
   Batch 8 was overly conservative. However, VRAM fragmentation can still cause OOM.
10. **PlantDoc generalisation gap**: Multiple 2025 papers show 20-40% accuracy drop from
    PlantVillage lab conditions to PlantDoc field conditions. Our specialist Model 3 with
    97-99% lab tomato data will show a similar gap. The Kerala tier-3 F1 is the ONLY
    honest measure of deployment performance — internal val F1 is optimistic by 20-40%.
11. **Source-disjoint validation impossible for thin classes**: okra_enation (90% Bangladesh)
    and brassica_black_rot (70% mendeley_caul_leaf) cannot have source-disjoint validation.
    Val F1 for these classes measures within-distribution performance, not generalisation.

### What farmer-facing uncertainty looks like:
- **okra_enation**: "Leaf enation detected (moderate confidence). Limited reference data for this disease in your region. Compare with photos from your local agricultural extension office."
- **tomato_yellow_leaf_curl_virus**: "Possible yellow leaf curl virus (limited confidence). Recommend agronomist verification before applying treatment."
- **Wide prediction sets**: "The leaf shows symptoms consistent with either late blight or foliar spot. Both respond to copper-based fungicide — apply that while awaiting expert verification."

---

## SECTION 12: DEBATE SUMMARY — ISSUES FOUND AND RESOLVED

### Round 1 (11 issues: 1 fatal, 6 significant, 4 minor)
1. [FATAL] config_model3.py has 11 classes but plan says 10 → Fixed: Phase 0 Step 1 updates config
2. [SIGNIFICANT] DINOv3-ConvNeXt-Small may not exist in timm → Fixed: 3-tier backbone fallback
3. [SIGNIFICANT] SupCon batch requirement (18 min) vs 384px batch size (8) → Fixed: SupCon only at 128/224px
4. [SIGNIFICANT] torch.compile + gradient checkpointing untested → Fixed: Phase 0 compatibility test
5. [SIGNIFICANT] rembg not installed → Fixed: Phase 0 installation
6. [SIGNIFICANT] DINOv2 registers: timm vs transformers ambiguity → Fixed: Phase 0 verification
7. [SIGNIFICANT] Stage 2 balanced at 288/class causes overfitting → Fixed: 432/class, 7 epochs
8. [MINOR] ASAM ρ=0.05 at 128px is dead config → Fixed: removed from config
9. [MINOR] Router soup on converged linear head is wasted → Fixed: 20 epochs, soup at 10-18
10. [MINOR] num_workers=0 makes data loading bottleneck → Fixed: test num_workers=2 in Phase 0
11. [MINOR] EMA decay too slow for Stage 2 → Fixed: 0.999 for Stage 2

### Round 2 (7 issues: 1 fatal, 3 significant, 3 minor)
12. [FATAL] target_spot quarantine requires CSV rebuild → Fixed: Phase 0 rebuild step
13. [SIGNIFICANT] FPN channel mismatch for DINOv3 variant → Fixed: FPN documented as unused
14. [SIGNIFICANT] DINO attention extraction unspecified → Fixed: last block, CLS, 6-head avg, 16×16
15. [SIGNIFICANT] Self-distillation agreement_rate hides per-class variation → Fixed: per-class min ≥ 0.50
16. [MINOR] Patch token count undocumented → Fixed: 257 tokens documented
17. [MINOR] Class-conditional augmentation non-trivial → Fixed: implementation pattern documented
18. [MINOR] SupCon temperature 0.07 too aggressive → Fixed: 0.10 with fallback to 0.20

### Round 3 (6 issues: 0 fatal, 3 significant, 3 minor)
19. [SIGNIFICANT] ConvNeXt pretrained=True untested → Fixed: Phase 0 download test
20. [SIGNIFICANT] 14 conformal images for okra_enation useless → Fixed: val set approximation
21. [SIGNIFICANT] Recomposed image categorisation in sampler ambiguous → Fixed: separate source, uncapped
22. [MINOR] num_workers per-model differentiation → Fixed: Model 3 only
23. [MINOR] Weight decay on LayerNorm → Fixed: no-decay parameter groups
24. [MINOR] Optimizer state at Stage 1→2 transition → Fixed: reinitialise

### Round 4 (4 issues: 0 fatal, 1 significant, 3 minor)
25. [SIGNIFICANT] Model 2 conformal uses val set (multi-purpose) → Accepted as known limitation
26. [MINOR] rembg quality filter → Fixed: 15-85% foreground mask check
27. [MINOR] Soup checkpoint epochs unspecified → Fixed: explicit epoch numbers
28. [MINOR] FiLM integration with peft internals fragile → Fixed: output-wrapper pattern

**TOTAL: 28 issues found across 4 rounds**
- 2 FATAL → both resolved
- 13 SIGNIFICANT → 12 resolved, 1 accepted as known limitation
- 13 MINOR → all resolved or documented

---

## SECTION 13: REMAINING OPEN QUESTIONS (For Human Decision)

1. **DINOv2-with-registers availability:** Phase 0 Step 0.4 verifies this. If unavailable in timm, the router uses `transformers.AutoModel` which has a different API. The training script structure depends on this decision. Cannot be resolved without running the verification.

2. **ConvNeXt-Small pretrained variant:** Phase 0 Step 0.5 verifies which pretrained weights download successfully. Model 2's starting point quality depends on this (DINOv3 > IN22k > IN1k).

3. **torch.compile + gradient_checkpointing:** Phase 0 Step 0.6 tests compatibility. If incompatible, Model 3 training is 25-30% slower (no compile) but still functional.

4. **num_workers=2 on Windows:** Phase 0 Step 0.8 tests this. If it crashes, Model 3 data loading is the bottleneck (~60% GPU utilisation instead of 90%).

5. **Kerala tier-3 data collection:** Minimum 30 images, ideally 50+. Collection strategy (iNaturalist API, market visits, extension service contacts) requires human effort outside the training pipeline. This gates Model 3 training — without it, all performance numbers are blind to deployment reality.

6. **Self-distillation go/no-go:** Depends on first pass Model 3 results. If macro F1 < 0.70 or agreement_rate < 0.70, skip self-distillation and ship the simple model. Cannot be decided until first pass completes.

---

---

## ERRATA — Version 5.1 (Post-Verification Corrections)

Six errors identified during independent verification of v5.0:

| # | Severity | Error | Fix Applied |
|---|---|---|---|
| E1 | CRITICAL | ASAM package not installed, missing from Phase 0, missing from train_utils Section R | Added Phase 0 Step 0.7b (pip install asam-optimizer), Section R now includes ASAM wrapper specification |
| E2 | SIGNIFICANT | TTA "CLAHE-enhanced" view applies double-CLAHE (input already CLAHE'd at Step 1) | Replaced with "brightness ±15% jitter" across all TTA references. Added NOTE explaining why |
| E3 | SIGNIFICANT | Section 7.1 says soup uses separate held-out set, Section 3.3 says Model 2 combines val+soup | Clarified: Model 2 uses combined val_and_soup (by design), Model 3 uses separate 7% soup split |
| E4 | SIGNIFICANT | Section 6.5 says 3 hours first pass, Section 10 says 1.5 hours | Corrected to 1.5 hours (verified from 330 img/s benchmark). Pre-benchmark estimate was wrong |
| E5 | MINOR | Rollback trigger "3 epochs later" compares across resolutions (always false-triggers) | Replaced with resolution_aware_rollback_check() specification. Added epoch 25 checkpoint |
| E6 | MINOR | features_only=True unnecessary since FPN not used | Changed to num_classes=9 (timm builds head internally). Simpler, less error-prone |

---

*End of MASTER_PLAN.md — Version 5.1 (Post-Verification)*
*28 debate issues + 6 verification errors = 34 total issues found and resolved*
*Estimated total execution time: 2-3 weeks of focused work*
*Minimum viable path (simple fallbacks + deployment): 11.5 days*
*Full plan path: 16.5 days*
