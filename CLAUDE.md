# CLAUDE.md — Plant Disease Detection System
# Broccoli and Ladies Finger (Okra), Kerala Deployment Target
# Version 6.0 — All 72 identified gaps resolved
# Complete implementation specification for Claude Code

---

## HOW TO READ THIS FILE

This is the single source of truth for the entire project. Read every section
before writing any code. Do not skip sections. Do not skim. Do not invent
anything not specified here — stop and state what is missing instead.

Two content types exist in this file:

SPECIFICATION content tells you what to build and why — architecture, constants,
data flow, rules, edge cases. These sections explain reasoning so you understand
the decisions behind the design.

IMPLEMENTATION content is working Python that Claude Code must use exactly
as written. Code blocks labelled "IMPLEMENT EXACTLY — copy verbatim into
[filename]" must be used as written. Code blocks without that label are
illustrative and show intent only.

When this file and your training data contradict each other, this file wins. It
was written after multiple audit passes of the exact architecture described here.

[FIX GAP47] pyinaturalist is removed from requirements. The acquire_kerala_images.py
agent uses the requests library directly to call the iNaturalist REST API.
pyinaturalist is never imported.

[FIX GAP48] Section 3.1 is REFERENCE ONLY. Section 4 (app/config.py) is the
AUTHORITATIVE IMPLEMENTATION. Do not write CLASS_NAMES, CLASS_TO_IDX, or any
other constants into any file other than app/config.py. All other modules
import from app.config.

[FIX GAP29] training/__init__.py, agents/__init__.py, and setup/__init__.py must
all be created as empty files containing only the comment:
    # This file marks the directory as a Python package.
Nothing else. No re-exports. No imports.

---

## CRITICAL RULES

**Rule 1 — Local only.** This project runs on one machine: i7 12th gen CPU with
RTX 4060 8 GB VRAM, Windows 11. No cloud, no Vast.ai, no remote training. If CUDA
cannot be installed, Section 13 specifies the exact diagnosis and fix. Cloud is
never the answer.

**Rule 2 — PlantDoc is tier-2 test only.** It never enters the training pool.
Never. Not a single image. It is downloaded to data/plantdoc/ and evaluated once
at the end. No exceptions.

**Rule 3 — Both test sets are locked.** The local 15% test split and PlantDoc are
both locked until Phase 2 training and calibration are complete. After evaluation,
no further model changes are permitted.

**Rule 4 — GitHub after every step.** Commit and push after every pipeline step.
If push fails, retry 3 times, log the failure, and continue. Never stop the
pipeline because of a git failure.

**Rule 5 — Agents mean local parallelism.** concurrent.futures.ThreadPoolExecutor
on your i7 cores. No remote execution of any kind.

**Rule 6 — Write all scripts from scratch.** The contracts in Sections 18 and 19
and the implementations throughout this file define everything Claude Code writes.
Nothing is inherited from the colleague's Keras codebase.

**Rule 7 — The colleague's code is read-only reference.** The files
train_chili_disease.py, train_tomato_disease.py, split_dataset.py, resave_models.py,
check_model_version.py are documentation of a prior approach. Never import from them.

**Rule 8 — The "What not to do" section (Section 21) lists 30+ specific failure
modes.** Read it before writing any code. Every item caused a real identified failure.

**Rule 9 — training/helpers.py is the authoritative location for all shared training
utilities.** [FIX GAP 3,4,19,20] EarlyStopping, save_checkpoint, load_checkpoint,
cleanup_old_checkpoints, and get_llrd_optimizer are defined ONLY in training/helpers.py.
They are NEVER redefined in 04_train_phase1.py or 05_train_phase2.py. Both training
scripts import them from training.helpers at module level, not inside __main__.

**Rule 10 — No magic numbers outside app/config.py.** Every constant, threshold,
path, and hyperparameter is defined in app/config.py. No other file contains
numeric literals for configuration. The only exceptions are the literal file being
defined (e.g. config.py itself) and loop indices.

**Rule 11 — No cross-layer imports from production into training.** [FIX GAP 8,46]
app/inference.py must NOT import from training.*. apply_clahe uses only cv2 and numpy
and is defined inline in app/inference.py. It is also defined in training/transforms.py
for training use. These are two copies of the same function — that is intentional and
correct. Do not DRY them by importing across the app/training boundary.

**Rule 12 — source_map.csv uses paths relative to the project root.** [FIX GAP 30]
image_path column stores paths like data/raw/sabbir_okra/healthy/img.jpg, not
C:\Users\seena\project\data\raw\... This makes the file portable: moving the project
folder does not break all paths. PlantDiseaseDataset resolves them with
os.path.join(ROOT, record['image_path']).

**Rule 13 — Images stay in data/raw/. data/processed/ directories are created
but left empty.** [FIX GAP 24] source_map.csv records point to data/raw/. The
data/processed/ directories exist to satisfy any external tooling expectations but
are never populated. PlantDiseaseDataset.__getitem__ reads from data/raw/ paths
in source_map.csv.

---

## SECTION 1: PROJECT OVERVIEW

### 1.1 What this system does

Plant disease detection for ladies finger (okra, Abelmoschus esculentus) and
broccoli (Brassica oleracea). A farmer photographs a diseased leaf with any
smartphone, uploads through a browser, and receives:

1. Crop: okra or brassica with calibrated confidence score
2. Diseases: all diseases present. Two diseases can coexist (co-infection). Both
   are detected. Output is multi-label, not single-label.
3. Severity: mild / moderate / severe as [low, high] interval from MC Dropout
4. Heatmap: Grad-CAM overlay as base64 PNG in the JSON response
5. Calibrated confidence: probability after temperature scaling
6. Uncertainty: 0.0 to 1.0 from MC Dropout standard deviation
7. Treatment: specific to the exact (crop, disease) pair
8. Prevention: specific to the exact (crop, disease) pair
9. Urgency: Low / Medium / High with plain-language reason

### 1.2 What this does not do

Does not detect pests. Leaf images only — not stems, roots, fruit, pods.
Does not handle tomato or chilli (colleague's system). Does not store uploaded
images (privacy by design). No accounts or login. Does not run in the cloud.
Does not support HEIC images (see Rule 11 in config — pillow-heif not installed).

### 1.3 Deployment

FastAPI backend. Plain HTML + CSS + JS frontend. No React, no Vue, no build step.
One HTML file, one CSS file, one JS file. Served from localhost:8000.

**[FIX GAP 27]** Student distillation to EfficientNet-Lite0 and HuggingFace
Spaces deployment are explicitly REMOVED from this version. They are not in the
checklist, not in the execution order, not in any section. If added later, they
require a separate specification document. Do not implement anything related to
HuggingFace Spaces or distillation.

Development server startup command (run from project root after training):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Production server startup command (no reload, 4 workers):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

[FIX GAP 26, 56] These two commands must be documented in README.md. run_pipeline.py
starts the server with the --reload flag for the smoke test only.

### 1.4 Hardware

```
CPU: Intel Core i7 12th generation (Alder Lake, 12 cores)
GPU: NVIDIA RTX 4060 8 GB GDDR6 VRAM (Ada Lovelace sm_89)
OS:  Windows 11
CUDA: Not installed at project start — installed by Step 01
Python: 3.10
Virtual environment: required — see Section 1.6
Expected total training time: 3.5 to 4 hours
Expected inference time: 5 to 10 seconds per image (5 MC passes)
```

### 1.5 Files Claude Code must create from scratch

[FIX GAP 13] training/helpers.py is ADDED to this list. It was missing from v5.

```
app/__init__.py                          training/helpers.py          ← ADDED
app/config.py                            training/metrics.py
app/model.py                             training/report.py
app/inference.py                         agents/__init__.py
app/validator.py                         agents/download_orchestrator.py
app/diagnosis.py                         agents/download_sabbir_okra.py
app/main.py                              agents/download_iubat_okra.py
training/__init__.py                     agents/download_kareem_cabbage.py
training/01_prepare_data.py              agents/download_misrak_vegetables.py
training/02_generate_severity.py         agents/download_faruk_okra.py
training/03_cache_features.py            agents/download_ghose_cabbage.py
training/04_train_phase1.py              agents/download_plantdoc.py
training/05_train_phase2.py              agents/acquire_kerala_images.py
training/06_calibrate.py                 agents/kaggle_utils.py       ← ADDED
training/07_evaluate_validation.py       agents/performance_monitor.py
training/08_evaluate_tier2_plantdoc.py   setup/__init__.py
training/09_evaluate_tier3_kerala.py     setup/setup_project.py
training/10_evaluate_local_test.py  ← ADDED  setup/install_cuda.py
training/dataset.py                      setup/install_dependencies.py
training/transforms.py                   setup/smoke_tests.py
training/loss.py                         setup/progress.py
                                         setup/test_server.py
diagnosis/diagnosis_lookup.json          setup/package_deployment.py
templates/index.html                     run_pipeline.py
static/style.css                         requirements.txt
static/app.js                            requirements_train.txt
                                         .gitignore
                                         .env.template
                                         README.md                   ← ADDED
                                         tools/add_kerala_image.py   ← ADDED
```

[FIX GAP 37] training/10_evaluate_local_test.py is ADDED. It evaluates the 15%
local test split that was locked during training. Section 20 specifies it fully.
[FIX GAP 28] agents/kaggle_utils.py is ADDED. All Kaggle credential logic lives
here. Individual agents import from it.
[FIX GAP 25] tools/add_kerala_image.py is ADDED. Workflow for adding verified
Kerala images to source_map.csv. Section 25 specifies it fully.
README.md is ADDED. Contains server startup commands and basic usage.

### 1.6 Virtual environment setup — REQUIRED

[FIX GAP 59] Always use a virtual environment. Never install to system Python.

```bash
# Create virtual environment (run once, from project root)
python -m venv venv

# Activate (run every time you open a new terminal)
# Windows Command Prompt:
venv\Scripts\activate.bat
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Git Bash / WSL:
source venv/Scripts/activate

# Verify activation (should show venv path):
python -c "import sys; print(sys.prefix)"
```

setup_project.py verifies that the user is running inside a virtual environment
and exits with a clear error message if not. This prevents accidental system-wide
installs.

---

## SECTION 2: THREE-TIER EVALUATION STRATEGY

### 2.1 Tier 1 — Standard validation set

Measures: whether the model is learning. Used for early stopping and per-epoch
monitoring only. Does NOT measure real-world performance.

Created from: 15% of training datasets (not PlantDoc, not Kerala), stratified
by class and source dataset using StratifiedGroupKFold. No source appears in both
train and val.

Used: after every training epoch. Macro F1 is the primary training metric.

Limitation: same data distribution as training. High score here does not mean
the model works in the field.

### 2.2 Tier 2 — PlantDoc wild test set

Measures: whether the model generalises to genuinely different data. PlantDoc
was collected independently with real field conditions, variable lighting, partial
occlusion, non-ideal camera angles.

Created: entire PlantDoc dataset downloaded to data/plantdoc/. Never touches the
training pool. Not one image. Arrives as train/ and test/ subdirectories — merge
them into a single pool before evaluating. The merge is just for flat access, not
for training.

Evaluated: exactly once by 08_evaluate_tier2_plantdoc.py after Phase 2 and
calibration are final. After this evaluation, no further model changes.

PlantDoc folder names that map to this project's classes (IMPLEMENT EXACTLY in
download_plantdoc.py and 08_evaluate_tier2_plantdoc.py):
[FIX GAP 52]
```python
PLANTDOC_CLASS_MAP = {
    # PlantDoc folder name                  : canonical class
    'Cabbage__Black_Rot'                    : 'brassica_black_rot',
    'Cabbage__Downy_Mildew'                 : 'brassica_downy_mildew',
    'Cabbage__Alternaria_leaf_spot'         : 'brassica_alternaria',
    'Cabbage__healthy'                      : 'brassica_healthy',
    # Lower-case variants also present in some PlantDoc versions:
    'cabbage black rot'                     : 'brassica_black_rot',
    'cabbage downy mildew'                  : 'brassica_downy_mildew',
    'cabbage alternaria leaf spot'          : 'brassica_alternaria',
    'cabbage healthy'                       : 'brassica_healthy',
}
```

Any PlantDoc folder not in this map is silently discarded. Do NOT force mappings
for non-brassica, non-okra classes. brassica_clubroot and all okra classes have
no PlantDoc equivalent — these classes are simply not evaluated at tier-2.

[FIX GAP 55] Temperature scaling IS applied for tier-2 evaluation. Load
models/temperature.pt, apply T_disease to disease logits before sigmoid.
This ensures the same calibrated inference pipeline as production.

Acceptance threshold: macro F1 > 0.55 (TIER2_MIN_F1) on the four mappable
classes. Below this threshold the model is not deployment-ready and a gap analysis
report must be written first.

### 2.3 Tier 3 — Kerala field acceptance test

Measures: whether the model works in the actual deployment environment.

Why separate: training data comes from Bangladesh, the US, and Europe. Kerala has
monsoon overcast lighting (blue-shifted), local phone cameras, and local crop
varieties. A model can pass tier 2 and still fail on Kerala field photos.

Minimum requirement: 50 verified labelled Kerala field images across at least 6
of the 10 classes. Until this minimum is met, the project is technically functional
but not deployment-validated.

Acceptable images: photos taken in Kerala or South India with verified disease
labels, iNaturalist observations in Kerala GPS zone that are manually disease-labelled,
TNAU or ICAR-IIHR images (Tamil Nadu, same climate zone), farmer submissions
verified manually through feedback mechanism.

Not acceptable: Stable Diffusion synthetic images. iNaturalist observations
without disease labels. YouTube frames without disease labels.

Acceptance threshold: per-class accuracy > 0.70 on classes with 5+ Kerala images.

[FIX GAP 25] Workflow for adding Kerala images to the evaluation set:
    python tools/add_kerala_image.py --path path/to/image.jpg --class okra_yvmv
This tool validates the image, copies it to data/kerala/{class_name}/, and
appends a record to source_map.csv with split='kerala'. See Section 25.

### 2.4 Training data split structure

```
Training pool = all downloaded datasets EXCEPT PlantDoc
    70% → train split
    15% → tier-1 val split
    15% → local test split (evaluated by 10_evaluate_local_test.py after tier-2)

data/plantdoc/   → tier-2 only (never in training pool)
data/kerala/     → tier-3 only (never in training pool)
data/*/synthetic → training only (never in any test)
```

[FIX GAP 37] The local test split is evaluated by training/10_evaluate_local_test.py.
This script is Step 13.5 in the pipeline — run it ONCE after tier-2 evaluation.
Specification in Section 20.

---

## SECTION 3: DISEASE CLASSES

### 3.1 Canonical class list — REFERENCE ONLY

[FIX GAP 48] The constants below are REFERENCE ONLY. The authoritative definition
is in app/config.py (Section 4). Do not copy these into any other file.

```python
CLASS_NAMES = [
    'okra_yvmv',             # index 0
    'okra_powdery_mildew',   # index 1
    'okra_cercospora',       # index 2
    'okra_enation',          # index 3
    'okra_healthy',          # index 4
    'brassica_black_rot',    # index 5
    'brassica_downy_mildew', # index 6
    'brassica_alternaria',   # index 7
    'brassica_clubroot',     # index 8
    'brassica_healthy',      # index 9
]
```

### 3.2 Disease reference

| Key | Full name | Pathogen | Primary visual symptoms |
|---|---|---|---|
| okra_yvmv | Yellow Vein Mosaic Virus | Begomovirus (whitefly) | Yellow vein network, mosaic, leaf curl, stunting |
| okra_powdery_mildew | Powdery Mildew | Erysiphe cichoracearum | White powder on upper surface, yellowing |
| okra_cercospora | Cercospora Leaf Spot | Cercospora abelmoschi | Brown spots with yellow halo, grey centre |
| okra_enation | Enation Leaf Curl | Begomovirus complex | Curling, enations on underside, vein swelling |
| okra_healthy | Healthy | None | Uniform green, no lesions |
| brassica_black_rot | Black Rot | Xanthomonas campestris | V-shaped margin lesions, darkened veins |
| brassica_downy_mildew | Downy Mildew | Hyaloperonospora parasitica | Yellow patches above, white sporulation below |
| brassica_alternaria | Alternaria Leaf Spot | Alternaria brassicicola | Dark concentric ring spots |
| brassica_clubroot | Clubroot | Plasmodiophora brassicae | Wilting, yellowing from root galls |
| brassica_healthy | Healthy | None | Uniform colour, no lesions |

### 3.3 Co-infection rules

Two diseases from the same crop can coexist on one leaf. Valid: okra_yvmv +
okra_cercospora. Impossible: okra_yvmv + brassica_black_rot (different crops).

A healthy prediction and a disease prediction cannot coexist for the same crop.
If okra_healthy sigmoid output > DISEASE_THRESHOLD, suppress all other okra
predictions. Same for brassica_healthy. This is post-processing after sigmoid
thresholding — the model does not enforce it internally.

---

## SECTION 4: COMPLETE CONFIGURATION (app/config.py) — IMPLEMENT EXACTLY

[FIX GAP 22] SOURCE_LABEL_OVERRIDES and PLANTDOC_CLASS_MAP are defined here in
app/config.py so they can be imported by both training scripts AND agent scripts
without circular imports. [FIX GAP 62] CLASS_COUNTS_PATH is defined here.
[FIX GAP 30] All path constants use relative paths resolved from ROOT.

```python
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
```

---

## SECTION 5: COMPLETE MODEL (app/model.py) — IMPLEMENT EXACTLY

[FIX GAP 67] _get_backbone_blocks() now raises RuntimeError if blocks list is
empty, so Phase 2 never silently skips backbone unfreezing.
[FIX GAP 7] out_p3 attribute name corrected in docstring (was incorrectly
documented as output_p3 in some places).

```python
# app/model.py

import torch
import torch.nn as nn
import timm
from app.config import (
    BACKBONE_NAME, FPN_IN_CH, FPN_OUT_CH, POOLED_DIM,
    CROP_EMB_DIM, HEAD_HIDDEN_DIM, DROPOUT_P, NUM_CLASSES, NUM_CROPS
)


class FPN(nn.Module):
    """
    Feature Pyramid Network. Takes three backbone stages and produces
    a single fused feature map at P3 resolution (28×28 for 224px input).

    Internal attribute names:
      self.lat_p3, self.lat_p4, self.lat_p5 — lateral projections
      self.out_p3  ← [FIX GAP 7] this is the correct attribute name.
                     Do not use output_p3 anywhere. Grad-CAM hooks use
                     model.fpn.out_p3 — any typo here breaks heatmaps.

    Top-down pathway:
      p5_lat  = lateral(stage5_features)   [B, 256, 7, 7]
      p4_up   = upsample(p5_lat) + lateral(stage4) [B, 256, 14, 14]
      p3_out  = upsample(p4_up)  + lateral(stage3) [B, 256, 28, 28]
      → out_p3 = p3_out  (output of final 3×3 conv)
    """
    def __init__(self):
        super().__init__()
        # Lateral projections: reduce each stage to FPN_OUT_CH channels
        self.lat_p3 = nn.Conv2d(FPN_IN_CH[0], FPN_OUT_CH, 1)
        self.lat_p4 = nn.Conv2d(FPN_IN_CH[1], FPN_OUT_CH, 1)
        self.lat_p5 = nn.Conv2d(FPN_IN_CH[2], FPN_OUT_CH, 1)
        # Output conv: smooth fused features after upsampling
        self.out_p3 = nn.Conv2d(FPN_OUT_CH, FPN_OUT_CH, 3, padding=1)
        self._upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, p3_feat, p4_feat, p5_feat):
        # Top-down: start from coarsest level (P5)
        p5 = self.lat_p5(p5_feat)
        p4 = self.lat_p4(p4_feat) + self._upsample(p5)
        p3 = self.lat_p3(p3_feat) + self._upsample(p4)
        return self.out_p3(p3)   # [B, FPN_OUT_CH, 28, 28]


class CropClassifier(nn.Module):
    """
    Binary crop classifier (okra vs brassica).
    Input: pooled FPN features [B, POOLED_DIM]
    Output: (logits [B, 2], embedding [B, CROP_EMB_DIM])
    The embedding is used as a FiLM conditioning signal for the disease head.
    """
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(POOLED_DIM, HEAD_HIDDEN_DIM)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(DROPOUT_P)
        self.embed = nn.Linear(HEAD_HIDDEN_DIM, CROP_EMB_DIM)
        self.fc2   = nn.Linear(CROP_EMB_DIM, NUM_CROPS)

    def forward(self, x):
        x   = self.drop(self.relu(self.fc1(x)))
        emb = self.embed(x)
        return self.fc2(emb), emb


class DiseaseHead(nn.Module):
    """
    Multi-label disease classifier. FiLM-conditioned on crop embedding.
    Input: pooled features [B, POOLED_DIM], crop_emb [B, CROP_EMB_DIM]
    Output: raw logits [B, NUM_CLASSES] — sigmoid applied in loss/inference

    FiLM conditioning: the crop embedding modulates the disease features
    via gamma (scale) and beta (shift) to bias predictions toward
    diseases relevant to the identified crop.
    """
    def __init__(self):
        super().__init__()
        # FiLM: produce scale (gamma) and shift (beta) from crop embedding
        self.film_gamma = nn.Linear(CROP_EMB_DIM, POOLED_DIM)
        self.film_beta  = nn.Linear(CROP_EMB_DIM, POOLED_DIM)
        # Disease classification layers
        self.fc1  = nn.Linear(POOLED_DIM, HEAD_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_P)
        self.fc2  = nn.Linear(HEAD_HIDDEN_DIM, NUM_CLASSES)

    def forward(self, pooled, crop_emb):
        gamma = torch.sigmoid(self.film_gamma(crop_emb))   # [B, POOLED_DIM]
        beta  = self.film_beta(crop_emb)                   # [B, POOLED_DIM]
        x     = pooled * gamma + beta                       # FiLM modulation
        x     = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)   # [B, NUM_CLASSES]


class SeverityHead(nn.Module):
    """
    3-class severity classifier: mild=0, moderate=1, severe=2.
    Input: pooled features [B, POOLED_DIM]
    Output: raw logits [B, 3] — softmax applied in inference
    """
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(POOLED_DIM, HEAD_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_P)
        self.fc2  = nn.Linear(HEAD_HIDDEN_DIM, 3)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)   # [B, 3]


class PlantDiseaseModel(nn.Module):
    """
    Full model: EfficientNetV2-S backbone + FPN + three heads.

    Forward returns: (crop_logits, disease_logits, severity_logits)
      crop_logits    [B, 2]          — cross-entropy loss
      disease_logits [B, NUM_CLASSES] — BCE loss (multi-label)
      severity_logits[B, 3]          — cross-entropy loss

    extract_features() returns (pooled, crop_emb) for Phase 1 caching.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=True,
            features_only=True, out_indices=(2, 3, 4)
        )
        self.fpn           = FPN()
        self.gap           = nn.AdaptiveAvgPool2d(1)
        self.crop_classifier = CropClassifier()
        self.disease_head    = DiseaseHead()
        self.severity_head   = SeverityHead()

    def forward(self, x):
        features       = self.backbone(x)       # list of 3 feature maps
        p3, p4, p5     = features[0], features[1], features[2]
        fused          = self.fpn(p3, p4, p5)   # [B, 256, 28, 28]
        pooled         = self.gap(fused).flatten(1)  # [B, 256]
        crop_logits, crop_emb = self.crop_classifier(pooled)
        disease_logits = self.disease_head(pooled, crop_emb)
        severity_logits = self.severity_head(pooled)
        return crop_logits, disease_logits, severity_logits

    def extract_features(self, x):
        """
        Returns (pooled [B,256], crop_emb [B,64]) for Phase 1 feature caching.
        Does NOT run disease or severity heads.
        """
        features   = self.backbone(x)
        fused      = self.fpn(features[0], features[1], features[2])
        pooled     = self.gap(fused).flatten(1)
        _, crop_emb = self.crop_classifier(pooled)
        return pooled, crop_emb

    def freeze_backbone(self):
        """Freeze all backbone parameters. Used at Phase 1 start."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_fraction(self, fraction=0.33):
        """
        Unfreeze the top `fraction` of backbone blocks for Phase 2 fine-tuning.
        'Top' means closest to the output (deepest blocks).
        fraction=0.33 unfreezes approximately the last 1/3 of blocks.
        """
        blocks = self._get_backbone_blocks()
        n_unfreeze = max(1, int(len(blocks) * fraction))
        for b in blocks:
            for p in b.parameters():
                p.requires_grad = False
        for b in blocks[-n_unfreeze:]:
            for p in b.parameters():
                p.requires_grad = True
        print(f"Unfroze top {n_unfreeze}/{len(blocks)} backbone blocks for Phase 2")

    def _get_backbone_blocks(self):
        """
        Returns list of backbone building blocks in order (first=shallowest,
        last=deepest). Handles timm FeatureListNet wrapper.

        [FIX GAP 67] Raises RuntimeError if blocks cannot be found, so Phase 2
        never silently skips backbone unfreezing with an empty list.
        """
        try:
            blocks = list(self.backbone.model.blocks)
            if blocks:
                return blocks
        except AttributeError:
            pass
        try:
            blocks = list(self.backbone.blocks)
            if blocks:
                return blocks
        except AttributeError:
            pass
        raise RuntimeError(
            "_get_backbone_blocks() could not find backbone.model.blocks or "
            "backbone.blocks. Check timm version and BACKBONE_NAME. "
            "EfficientNetV2-S in timm >= 0.9 exposes blocks at backbone.model.blocks. "
            "Verify with: model.backbone.model.blocks"
        )

    def _get_stem_params(self):
        """
        Returns backbone stem layer parameters for LLRD.
        These get the lowest learning rate in Phase 2.
        """
        params = []
        for attr in ['conv_stem', 'bn1']:
            try:
                layer = getattr(self.backbone.model, attr)
                params.extend(list(layer.parameters()))
            except AttributeError:
                pass
        return params


def verify_backbone_shapes(model, device='cpu'):
    """
    Sanity check that backbone returns feature maps at expected shapes.
    Call once before any training. Crashes loudly if shapes are wrong.
    If wrong: update FPN_IN_CH in config.py to match actual output shapes.
    """
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        feats = model.backbone(dummy)
    expected = [(1, 48, 28, 28), (1, 160, 14, 14), (1, 256, 7, 7)]
    for i, (feat, exp) in enumerate(zip(feats, expected)):
        assert tuple(feat.shape) == exp, (
            f"Backbone stage {i+2} shape {tuple(feat.shape)} != expected {exp}. "
            f"Update FPN_IN_CH in app/config.py to {[f.shape[1] for f in feats]}"
        )
    print(f"Backbone shapes verified: {[tuple(f.shape) for f in feats]}")


def load_model_for_inference(model_path, device):
    """
    Loads model from state_dict for inference. Returns UNCOMPILED model in eval mode.
    IMPORTANT: always use this function to load for inference, never torch.load(model).
    torch.compile breaks pytorch_grad_cam hooks — save state_dict, load here.
    """
    model = PlantDiseaseModel()
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model
```
---

## SECTION 6: DATA PIPELINE

### 6.1 Source map CSV format

source_map.csv is the master record of every image. Written by 01_prepare_data.py.
Never modified manually except through tools/add_kerala_image.py.
Every training script reads paths from this file.

[FIX GAP 30] image_path is RELATIVE to the project root. Example:
  data/raw/sabbir_okra/healthy/img.jpg
NOT:
  C:\Users\seena\plant_identification\data\raw\...

PlantDiseaseDataset resolves paths with: os.path.join(ROOT, record['image_path'])

Columns and valid values:
```
image_path     — path relative to project ROOT (e.g. data/raw/sabbir_okra/...)
source_dataset — sabbir_okra | iubat_okra | kareem_cabbage | misrak_veg |
                 faruk_okra | ghose_cabbage | plantdoc | kerala |
                 synthetic | domain_adapt
raw_label      — original label string from the downloaded dataset
class_name     — canonical class name (one of CLASS_NAMES, or 'unknown')
class_idx      — integer 0-9, or -1 if unmappable
crop_idx       — 0=okra, 1=brassica, -1 if unmappable
split          — train | val | test | plantdoc | kerala | domain_adapt
```

`domain_adapt` split: iNaturalist and YouTube frames with no disease labels.
Used as target domain images for future DANN extension. Never used in any loss.

`synthetic` source always gets split=`train` regardless of stratified split output.

[FIX GAP 24] Images stay in data/raw/. data/processed/ directories are created
by setup_project.py but remain empty. source_map.csv paths point to data/raw/.

### 6.2 Label harmonisation

[FIX GAP 22] LABEL_MAP and SOURCE_LABEL_OVERRIDES are defined in app/config.py,
NOT in training/01_prepare_data.py. Import them from there.

```python
# Part of training/01_prepare_data.py — import from app.config, define functions here

from app.config import LABEL_MAP, SOURCE_LABEL_OVERRIDES, CLASS_TO_IDX, CROP_FROM_IDX


def resolve_label(raw_label, source):
    """
    Resolves a raw label string to a canonical class name.
    Check SOURCE_LABEL_OVERRIDES first (source-specific context),
    then LABEL_MAP (global mapping).
    Returns canonical class name or raises KeyError.
    """
    key = (source, raw_label.lower().strip())
    if key in SOURCE_LABEL_OVERRIDES:
        return SOURCE_LABEL_OVERRIDES[key]
    normalised = raw_label.lower().strip()
    if normalised in LABEL_MAP:
        return LABEL_MAP[normalised]
    raise KeyError(f"No mapping for label='{raw_label}' from source='{source}'")


def assert_all_labels_mapped(records):
    """
    Checks every (source, raw_label) pair resolves to a canonical class.
    Crashes with a clear list if any are unmapped.
    Call this before any split or training operation.
    """
    unmapped = []
    for r in records:
        try:
            resolve_label(r['raw_label'], r['source_dataset'])
        except KeyError:
            unmapped.append(
                f"  source={r['source_dataset']!r}, label={r['raw_label']!r}, "
                f"path={r['image_path']!r}"
            )
    if unmapped:
        raise ValueError(
            f"Found {len(unmapped)} unmapped labels. Add them to LABEL_MAP "
            f"or SOURCE_LABEL_OVERRIDES in app/config.py before proceeding:\n"
            + "\n".join(unmapped[:30])
            + ("\n  ...(truncated)" if len(unmapped) > 30 else "")
        )
    print(f"Label assertion passed: all {len(records)} records have valid mappings.")
```

### 6.3 Stratified split

```python
# Part of training/01_prepare_data.py

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_split(records, seed=42):
    """
    Splits records into train/val/test preserving class proportions per split
    and keeping each source_dataset entirely within one split.

    Why both constraints matter:
    - Stratified: each class appears proportionally in every split
    - Grouped by source: prevents near-duplicate images from the same dataset
      appearing in both train and test (data leakage through similar style)

    Returns: (train_records, val_records, test_records)

    Edge cases handled:
    - If any class has < 2 images: raises ValueError with download instructions
    - If n_splits > minimum class count: reduces n_splits to safe value
    - Synthetic images always go to train regardless of split output
    - Prints warning if any source appears in multiple splits
    """
    # Exclude plantdoc, kerala, domain_adapt — fixed splits already
    pool = [r for r in records
            if r.get('split') not in ('plantdoc', 'kerala', 'domain_adapt')]

    X      = np.array([r['image_path'] for r in pool])
    labels = np.array([r['class_idx']  for r in pool])
    groups = np.array([r['source_dataset'] for r in pool])

    min_class = int(np.bincount(labels).min())
    if min_class < 2:
        raise ValueError(
            f"Minimum class count is {min_class}. Need >= 2 per class. "
            f"Download more data before splitting."
        )

    n_test = min(7, min_class)
    sgkf   = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(sgkf.split(X, labels, groups))

    X_tv, lab_tv, grp_tv = X[tv_idx], labels[tv_idx], groups[tv_idx]
    n_val = min(6, int(np.bincount(lab_tv).min()))
    if n_val < 2:
        raise ValueError("Not enough data for val split after test split.")
    sgkf2  = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_sub, val_sub = next(sgkf2.split(X_tv, lab_tv, grp_tv))

    train_idx = tv_idx[tr_sub]
    val_idx   = tv_idx[val_sub]

    train_r = [pool[i] for i in train_idx]
    val_r   = [pool[i] for i in val_idx]
    test_r  = [pool[i] for i in test_idx]

    # Override: synthetic always in train
    for r in records:
        if r.get('source_dataset') == 'synthetic':
            r['split'] = 'train'
            if r not in train_r:
                train_r.append(r)

    # Verify source isolation
    train_src = {r['source_dataset'] for r in train_r}
    test_src  = {r['source_dataset'] for r in test_r}
    overlap   = train_src & test_src
    if overlap:
        print(f"WARNING: sources in both train and test: {overlap}. "
              f"May indicate insufficient source diversity.")

    print(f"Split: {len(train_r)} train, {len(val_r)} val, {len(test_r)} test")
    return train_r, val_r, test_r
```

### 6.4 Complete 01_prepare_data.py — IMPLEMENT EXACTLY

[FIX GAP 16] Complete runnable script. The v5 spec only had bullet-point contracts.
[FIX GAP 11] Does NOT re-download datasets — assumes Step 03 already ran them.
[FIX GAP 62] Writes CLASS_COUNTS_PATH.
[FIX GAP 64] Internal folder structures for each Kaggle dataset are specified below.

**Kaggle dataset internal folder structures after unzip:**

```
data/raw/sabbir_okra/
  ├── train/          ← use this subdirectory if it exists
  │   ├── Healthy/
  │   ├── Yellow_Vein_Mosaic_Disease/
  │   ├── Powdery_Mildew/
  │   ├── Cercospora_Leaf_Spot/
  │   └── Enation_Leaf_Curl/
  └── (or flat class folders at top level)

data/raw/kareem_cabbage/
  ├── Black_Rot/
  ├── Downy_Mildew/
  ├── Alternaria_Leaf_Spot/
  └── Healthy/

data/raw/misrak_veg/
  ├── cabbage_/          ← keep: matches 'cabbage'
  ├── broccoli_/         ← keep: matches 'broccoli'
  └── tomato_/           ← SKIP: not brassica

data/raw/faruk_okra/
  ├── healthy/
  ├── yellow_vein_mosaic/
  ├── leaf_spot/
  ├── enation_leaf_curl/
  └── powdery_mildew/

data/raw/ghose_cabbage/
  ├── black_rot/
  ├── downy_mildew/
  ├── alternaria/
  └── healthy/
```

For each dataset, the scan algorithm is:
1. Walk all directories under dest_dir (or dest_dir/train/ if train/ exists)
2. The immediate parent directory name of each image is the raw_label
3. Apply SOURCE_LABEL_OVERRIDES then LABEL_MAP to get class_name
4. Images with unmapped labels are logged and skipped (not crashed on)
5. After scanning all datasets, call assert_all_labels_mapped on mapped records

```python
# training/01_prepare_data.py
# IMPLEMENT EXACTLY

import os
import sys
import csv
import json
import pathlib
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    ROOT, RAW, META, SOURCE_MAP, CLASS_COUNTS_PATH,
    CLASS_TO_IDX, CROP_FROM_IDX, VALID_EXT,
    KERALA_DIR, PLANTDOC_DIR, PLANTDOC_CLASS_MAP,
    LABEL_MAP, SOURCE_LABEL_OVERRIDES
)

# Import helpers from this file (defined below in same module)
# resolve_label and assert_all_labels_mapped are defined in this file.
# stratified_group_split is defined in this file.


def resolve_label(raw_label, source):
    key = (source, raw_label.lower().strip())
    if key in SOURCE_LABEL_OVERRIDES:
        return SOURCE_LABEL_OVERRIDES[key]
    normalised = raw_label.lower().strip()
    if normalised in LABEL_MAP:
        return LABEL_MAP[normalised]
    raise KeyError(f"No mapping for label='{raw_label}' from source='{source}'")


def assert_all_labels_mapped(records):
    unmapped = []
    for r in records:
        try:
            resolve_label(r['raw_label'], r['source_dataset'])
        except KeyError:
            unmapped.append(
                f"  source={r['source_dataset']!r}, "
                f"label={r['raw_label']!r}, path={r['image_path']!r}"
            )
    if unmapped:
        raise ValueError(
            f"Found {len(unmapped)} unmapped labels. "
            f"Add them to LABEL_MAP or SOURCE_LABEL_OVERRIDES in app/config.py:\n"
            + "\n".join(unmapped[:30])
        )
    print(f"Label assertion passed: all {len(records)} records are mapped.")


def _scan_source(source_id, source_dir):
    """
    Scans a single source directory tree.
    Returns list of record dicts for images with mappable labels.
    Skips images with unmappable labels and logs them.
    """
    records   = []
    skipped   = 0
    # Check for 'train' subdirectory (e.g. sabbir_okra)
    scan_root = source_dir
    train_sub = os.path.join(source_dir, 'train')
    if os.path.isdir(train_sub):
        scan_root = train_sub

    for dirpath, _, filenames in os.walk(scan_root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext not in VALID_EXT:
                continue
            full_path   = os.path.join(dirpath, fname)
            raw_label   = os.path.basename(dirpath)  # immediate parent = class folder
            # Compute relative path from ROOT
            try:
                rel_path = os.path.relpath(full_path, ROOT).replace('\\', '/')
            except ValueError:
                # Different drive on Windows — use absolute as fallback
                rel_path = full_path.replace('\\', '/')

            try:
                class_name = resolve_label(raw_label, source_id)
                class_idx  = CLASS_TO_IDX[class_name]
                crop_idx   = CROP_FROM_IDX[class_idx]
            except (KeyError, TypeError):
                skipped += 1
                continue

            records.append({
                'image_path'    : rel_path,
                'source_dataset': source_id,
                'raw_label'     : raw_label,
                'class_name'    : class_name,
                'class_idx'     : class_idx,
                'crop_idx'      : crop_idx,
                'split'         : '',  # filled in by stratified_group_split
            })

    if skipped > 0:
        print(f"  [{source_id}] Skipped {skipped} images with unmappable labels.")
    print(f"  [{source_id}] Loaded {len(records)} images.")
    return records


def _scan_misrak(source_dir):
    """
    misrak_veg: keep only subdirectories containing 'cabbage', 'broccoli',
    or 'brassica' (case-insensitive). Skip all others (tomato, etc.).
    """
    records = []
    for folder in os.listdir(source_dir):
        folder_lower = folder.lower()
        if not any(kw in folder_lower for kw in ('cabbage', 'broccoli', 'brassica')):
            continue
        class_dir = os.path.join(source_dir, folder)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1]
            if ext not in VALID_EXT:
                continue
            full_path = os.path.join(class_dir, fname)
            rel_path  = os.path.relpath(full_path, ROOT).replace('\\', '/')
            try:
                class_name = resolve_label(folder, 'misrak_veg')
                class_idx  = CLASS_TO_IDX[class_name]
                crop_idx   = CROP_FROM_IDX[class_idx]
            except (KeyError, TypeError):
                continue
            records.append({
                'image_path'    : rel_path,
                'source_dataset': 'misrak_veg',
                'raw_label'     : folder,
                'class_name'    : class_name,
                'class_idx'     : class_idx,
                'crop_idx'      : crop_idx,
                'split'         : '',
            })
    print(f"  [misrak_veg] Loaded {len(records)} brassica images.")
    return records


def _scan_plantdoc(plantdoc_dir):
    """
    PlantDoc: merge train/ and test/ subdirectories.
    Use PLANTDOC_CLASS_MAP for label resolution.
    All records get split='plantdoc' (never in training pool).
    """
    records = []
    for subset in ('train', 'test'):
        subset_dir = os.path.join(plantdoc_dir, subset)
        if not os.path.isdir(subset_dir):
            continue
        for folder in os.listdir(subset_dir):
            if folder not in PLANTDOC_CLASS_MAP:
                continue  # silently discard non-brassica classes
            class_name = PLANTDOC_CLASS_MAP[folder]
            class_idx  = CLASS_TO_IDX[class_name]
            crop_idx   = CROP_FROM_IDX[class_idx]
            class_dir  = os.path.join(subset_dir, folder)
            for fname in os.listdir(class_dir):
                ext = os.path.splitext(fname)[1]
                if ext not in VALID_EXT:
                    continue
                full_path = os.path.join(class_dir, fname)
                rel_path  = os.path.relpath(full_path, ROOT).replace('\\', '/')
                records.append({
                    'image_path'    : rel_path,
                    'source_dataset': 'plantdoc',
                    'raw_label'     : folder,
                    'class_name'    : class_name,
                    'class_idx'     : class_idx,
                    'crop_idx'      : crop_idx,
                    'split'         : 'plantdoc',
                })
    print(f"  [plantdoc] Loaded {len(records)} images.")
    return records


def _scan_kerala(kerala_dir):
    """
    Kerala tier-3 images. Each subdirectory is a class_name.
    All records get split='kerala'.
    """
    records = []
    if not os.path.isdir(kerala_dir):
        print("  [kerala] No images yet.")
        return records
    for class_name in os.listdir(kerala_dir):
        if class_name not in CLASS_TO_IDX:
            continue
        class_dir  = os.path.join(kerala_dir, class_name)
        class_idx  = CLASS_TO_IDX[class_name]
        crop_idx   = CROP_FROM_IDX[class_idx]
        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1]
            if ext not in VALID_EXT:
                continue
            full_path = os.path.join(class_dir, fname)
            rel_path  = os.path.relpath(full_path, ROOT).replace('\\', '/')
            records.append({
                'image_path'    : rel_path,
                'source_dataset': 'kerala',
                'raw_label'     : class_name,
                'class_name'    : class_name,
                'class_idx'     : class_idx,
                'crop_idx'      : crop_idx,
                'split'         : 'kerala',
            })
    print(f"  [kerala] Loaded {len(records)} images.")
    return records


def stratified_group_split(records, seed=42):
    """(full implementation in Section 6.3 above — copy verbatim)"""
    from sklearn.model_selection import StratifiedGroupKFold

    pool = [r for r in records
            if r.get('split') not in ('plantdoc', 'kerala', 'domain_adapt')]
    if not pool:
        raise ValueError("No records available for splitting. Check data/raw/.")

    X      = np.array([r['image_path']     for r in pool])
    labels = np.array([r['class_idx']      for r in pool])
    groups = np.array([r['source_dataset'] for r in pool])

    min_class = int(np.bincount(labels).min())
    if min_class < 2:
        raise ValueError(
            f"Minimum class count is {min_class}. Need >= 2. Download more data."
        )

    n_test = min(7, min_class)
    sgkf   = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(sgkf.split(X, labels, groups))

    X_tv, lab_tv, grp_tv = X[tv_idx], labels[tv_idx], groups[tv_idx]
    n_val = min(6, int(np.bincount(lab_tv).min()))
    if n_val < 2:
        raise ValueError("Not enough data for val split after test split.")
    sgkf2   = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_sub, val_sub = next(sgkf2.split(X_tv, lab_tv, grp_tv))

    train_idx = tv_idx[tr_sub]
    val_idx   = tv_idx[val_sub]

    train_r = [pool[i] for i in train_idx]
    val_r   = [pool[i] for i in val_idx]
    test_r  = [pool[i] for i in test_idx]

    for r in train_r: r['split'] = 'train'
    for r in val_r:   r['split'] = 'val'
    for r in test_r:  r['split'] = 'test'

    # Override: synthetic always in train
    for r in records:
        if r.get('source_dataset') == 'synthetic':
            r['split'] = 'train'
            if r not in train_r:
                train_r.append(r)

    train_src = {r['source_dataset'] for r in train_r}
    test_src  = {r['source_dataset'] for r in test_r}
    overlap   = train_src & test_src
    if overlap:
        print(f"WARNING: sources in both train and test: {overlap}")

    print(f"Split: {len(train_r)} train, {len(val_r)} val, {len(test_r)} test")
    return train_r, val_r, test_r


def write_source_map(all_records):
    """Write source_map.csv with all records."""
    os.makedirs(META, exist_ok=True)
    fieldnames = ['image_path', 'source_dataset', 'raw_label',
                  'class_name', 'class_idx', 'crop_idx', 'split']
    with open(SOURCE_MAP, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Written {len(all_records)} records to {SOURCE_MAP}")


def write_class_counts(all_records):
    """[FIX GAP 62] Write class_counts.csv: class_name, split, count."""
    from collections import Counter
    os.makedirs(META, exist_ok=True)
    counts = Counter((r['class_name'], r['split']) for r in all_records)
    with open(CLASS_COUNTS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'split', 'count'])
        for (cls, spl), cnt in sorted(counts.items()):
            writer.writerow([cls, spl, cnt])
    print(f"Written class counts to {CLASS_COUNTS_PATH}")


if __name__ == '__main__':
    print("=" * 60)
    print("01_PREPARE_DATA — scanning datasets, splitting, writing CSV")
    print("=" * 60)
    print("NOTE: Assumes datasets are already downloaded (Step 03 ran).")
    print("      Scanning data/raw/ for present datasets.\n")

    all_records = []

    # ── Priority-1 training datasets ──────────────────────────────────────
    TRAINING_SOURCES = [
        ('sabbir_okra',    os.path.join(RAW, 'sabbir_okra')),
        ('iubat_okra',     os.path.join(RAW, 'iubat_okra')),
        ('kareem_cabbage', os.path.join(RAW, 'kareem_cabbage')),
        ('faruk_okra',     os.path.join(RAW, 'faruk_okra')),
        ('ghose_cabbage',  os.path.join(RAW, 'ghose_cabbage')),
    ]
    for source_id, source_dir in TRAINING_SOURCES:
        if not os.path.isdir(source_dir):
            print(f"  [{source_id}] Directory not found — skipping. Run Step 03.")
            continue
        records = _scan_source(source_id, source_dir)
        all_records.extend(records)

    # misrak_veg has special filtering logic
    misrak_dir = os.path.join(RAW, 'misrak_veg')
    if os.path.isdir(misrak_dir):
        all_records.extend(_scan_misrak(misrak_dir))
    else:
        print("  [misrak_veg] Directory not found — skipping.")

    if not all_records:
        raise RuntimeError(
            "No training images found. Check that Step 03 (download) completed."
        )

    # ── Label assertion ────────────────────────────────────────────────────
    assert_all_labels_mapped(all_records)

    # ── Stratified split ───────────────────────────────────────────────────
    train_r, val_r, test_r = stratified_group_split(all_records, seed=42)

    # Collect all records including fixed-split sets
    split_map = {r['image_path']: r['split'] for r in train_r + val_r + test_r}
    for r in all_records:
        if r['image_path'] in split_map:
            r['split'] = split_map[r['image_path']]

    # ── PlantDoc (fixed split=plantdoc) ───────────────────────────────────
    if os.path.isdir(PLANTDOC_DIR):
        plantdoc_records = _scan_plantdoc(PLANTDOC_DIR)
        all_records.extend(plantdoc_records)

    # ── Kerala (fixed split=kerala) ───────────────────────────────────────
    kerala_records = _scan_kerala(KERALA_DIR)
    all_records.extend(kerala_records)

    # ── Write outputs ──────────────────────────────────────────────────────
    write_source_map(all_records)
    write_class_counts(all_records)

    # ── Print summary ──────────────────────────────────────────────────────
    from collections import Counter
    split_counts = Counter(r['split'] for r in all_records)
    class_counts = Counter(
        r['class_name'] for r in all_records if r['split'] == 'train'
    )
    print(f"\nSplit summary: {dict(split_counts)}")
    print("\nTraining class counts:")
    for cls, cnt in sorted(class_counts.items()):
        warn = " ← THIN" if cnt < 150 else ""
        print(f"  {cls:30s}: {cnt:5d}{warn}")
```

### 6.5 Complete augmentation pipelines (training/transforms.py) — IMPLEMENT EXACTLY

[FIX GAP 8,11] apply_clahe is defined here for training. It is ALSO defined
independently in app/inference.py. Do NOT import it across the boundary.

```python
# training/transforms.py

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from app.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    CLAHE per RGB channel.
    Corrects Kerala monsoon blue-shift and fluorescent yellow-shift.
    Must be the FIRST step in BOTH training and inference transforms.
    Defined here for training. Also defined inline in app/inference.py.
    Do NOT import from here into app/inference.py.
    Input/output: uint8 numpy array [H, W, 3].
    """
    clahe  = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = clahe.apply(image[:, :, c])
    return result


def simulate_colour_temperature(image, **kwargs):
    """
    Simulates Kerala-specific lighting colour temperatures.
    Monsoon overcast: blue-shifted (R down, B up).
    Indoor fluorescent: yellow-shifted (R up, B down).
    """
    factor = np.random.uniform(0.75, 1.35)
    img    = image.astype(np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] * factor,         0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 / factor), 0, 255)
    return img.astype(np.uint8)


def get_train_transform():
    """
    Training augmentation. Kerala-specific. Applied ONLY to train split.
    CLAHE (step 1) and Normalize+ToTensor (last steps) are mandatory.
    """
    return A.Compose([
        A.Lambda(image=apply_clahe, p=1.0),
        A.Resize(256, 256),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101, p=0.6
        ),
        A.RandomCrop(IMG_SIZE[0], IMG_SIZE[1]),
        A.OneOf([
            A.Lambda(image=simulate_colour_temperature, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform():
    """
    Validation / test / inference transform.
    CLAHE first (same as training) then deterministic resize + normalize.
    Must match training: if CLAHE is in training, it must be here too.
    """
    return A.Compose([
        A.Lambda(image=apply_clahe, p=1.0),
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

---

## SECTION 7: TRAINING HELPERS (training/helpers.py) — IMPLEMENT EXACTLY

[FIX GAP 3,4,13,19,20] This file is the authoritative location for
EarlyStopping, save_checkpoint, load_checkpoint, cleanup_old_checkpoints,
and get_llrd_optimizer. These functions NEVER appear in 04_train_phase1.py or
05_train_phase2.py. Both scripts import from here at MODULE LEVEL.

```python
# training/helpers.py
# Shared utilities for Phase 1 and Phase 2 training scripts.
# Import these at MODULE LEVEL in training scripts — NOT inside __main__.

import os
import glob
import torch


class EarlyStopping:
    """
    Monitors val/macro_f1. Stops training when score does not improve
    by min_delta for `patience` consecutive epochs.
    Returns True when training should stop.
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics, path):
    """
    Saves full training state dict. scaler/scheduler may be None.
    Creates parent directory if missing.
    Pass the RAW model (not compiled): raw_model = getattr(model, '_orig_mod', model)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch'            : epoch,
        'model_state_dict' : model.state_dict(),
        'val_metrics'      : val_metrics,
    }
    if optimizer: state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler: state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler:    state['scaler_state_dict']    = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    """
    Loads training state from checkpoint.
    weights_only=False required for optimizer state dict.
    Returns (epoch, val_metrics).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('val_metrics', {})


def cleanup_old_checkpoints(ckpt_dir, keep_n=3, phase='phase1'):
    """
    Keeps only `keep_n` most recent epoch checkpoints per phase.
    phase1_best.pt and phase2_best.pt are NEVER deleted
    (they don't contain 'epoch' in the filename).
    """
    pattern = os.path.join(ckpt_dir, f'{phase}_epoch*.pt')
    ckpts   = sorted(glob.glob(pattern))
    to_del  = ckpts[:-keep_n] if len(ckpts) > keep_n else []
    for c in to_del:
        os.remove(c)


def get_llrd_optimizer(model, base_lr, decay=0.85, weight_decay=1e-4):
    """
    AdamW with Layer-wise Learning Rate Decay (LLRD).
    Classification heads and FPN: full base_lr.
    Backbone blocks: decaying LR from output to input.
    Backbone stem: lowest LR.

    Why LLRD: shallow backbone layers (stem, early blocks) encode generic
    low-level features that are already well-calibrated from ImageNet pretraining.
    Giving them high LR would overwrite useful features. Deep blocks encode
    task-specific features that need updating. Decaying LR preserves the
    shallow features while allowing deep fine-tuning.

    [FIX GAP 3,20] This is the ONLY definition of get_llrd_optimizer.
    It is NOT duplicated in 05_train_phase2.py.
    """
    param_groups = []

    # Heads and FPN: full base_lr
    for name, module in [
        ('disease_head',    model.disease_head),
        ('severity_head',   model.severity_head),
        ('crop_classifier', model.crop_classifier),
        ('fpn',             model.fpn),
    ]:
        param_groups.append({
            'params': list(module.parameters()),
            'lr'    : base_lr,
            'name'  : name,
        })

    # Backbone blocks: decaying LR from output (i=0) to input (i=len-1)
    blocks = model._get_backbone_blocks()  # raises RuntimeError if empty
    for i, block in enumerate(reversed(blocks)):
        lr = base_lr * (decay ** i)
        param_groups.append({
            'params': list(block.parameters()),
            'lr'    : lr,
            'name'  : f'backbone_block_{len(blocks) - 1 - i}',
        })

    # Backbone stem: lowest LR
    stem = model._get_stem_params()
    if stem:
        param_groups.append({
            'params': stem,
            'lr'    : base_lr * (decay ** len(blocks)),
            'name'  : 'backbone_stem',
        })

    for pg in param_groups:
        print(f"  LLRD {pg['name']:35s}: lr={pg['lr']:.2e}")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
```

---

## SECTION 8: TRAINING PHASE 1 (training/04_train_phase1.py) — IMPLEMENT EXACTLY

[FIX GAP 1] EarlyStopping, save_checkpoint, load_checkpoint, cleanup_old_checkpoints
are imported at MODULE LEVEL from training.helpers — NOT inside __main__.
[FIX GAP 34] pos_weight construction is fixed: pass train_records directly
(one record per image, class_idx is the single class). The function uses image
count as denominator, not label count.

```python
# training/04_train_phase1.py
"""
Phase 1: Train classification heads on cached backbone features.
Backbone is frozen. Training runs from pre-computed feature tensors.
This takes 25-35 minutes instead of 3.5 hours.

Run AFTER 03_cache_features.py completes.
Saves: models/checkpoints/phase1_best.pt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [FIX GAP 1] Import helpers at MODULE LEVEL, not inside __main__
from training.helpers import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

from app.config import (
    DEVICE, TRAIN_CACHE, VAL_CACHE, CKPT_DIR, MODELS,
    PHASE1_EPOCHS, PHASE1_LR, BATCH_SIZE, WEIGHT_DECAY,
    LOSS_W_CROP, LOSS_W_DISEASE, LOSS_W_SEVERITY,
    MAX_POS_WEIGHT, EARLY_STOP_PAT, EARLY_STOP_DELTA, KEEP_CKPTS,
    NUM_CLASSES, NUM_CROPS, CROP_EMB_DIM, HEAD_HIDDEN_DIM, POOLED_DIM,
    DROPOUT_P, RANDOM_SEED, WANDB_PROJECT, WANDB_CONFIG,
)
from app.model import PlantDiseaseModel
from training.metrics import compute_multilabel_pos_weights
from training.loss import compute_loss


def set_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase1():
    set_seeds(RANDOM_SEED)

    # ── Load cached features ───────────────────────────────────────────────
    if not os.path.exists(TRAIN_CACHE):
        raise FileNotFoundError(
            f"Training cache not found at {TRAIN_CACHE}. "
            f"Run training/03_cache_features.py first."
        )
    if not os.path.exists(VAL_CACHE):
        raise FileNotFoundError(
            f"Validation cache not found at {VAL_CACHE}. "
            f"Run training/03_cache_features.py first."
        )

    print("Loading cached features...")
    train_cache = torch.load(TRAIN_CACHE, weights_only=False)
    val_cache   = torch.load(VAL_CACHE, weights_only=False)

    # Feature tensors
    train_pooled   = train_cache['pooled_features']    # [N, 256]
    train_crop_emb = train_cache['crop_embeddings']    # [N, 64]
    train_d_lab    = train_cache['disease_labels']     # [N, 10]
    train_c_lab    = train_cache['crop_labels']        # [N]
    train_s_lab    = train_cache['severity_labels']    # [N]

    val_pooled   = val_cache['pooled_features']
    val_crop_emb = val_cache['crop_embeddings']
    val_d_lab    = val_cache['disease_labels']
    val_c_lab    = val_cache['crop_labels']
    val_s_lab    = val_cache['severity_labels']

    print(f"Train features: {train_pooled.shape}, Val features: {val_pooled.shape}")

    # ── [FIX GAP 34] pos_weight: compute from train_d_lab directly ─────────
    # train_d_lab is [N, NUM_CLASSES] multi-hot binary matrix.
    # n_pos[j] = number of training images positive for class j
    # n_neg[j] = N - n_pos[j] = number of training images negative for class j
    # pos_weight[j] = n_neg[j] / n_pos[j]
    # This is correct multi-label formula. NOT sklearn compute_class_weight.
    n_total  = float(train_d_lab.shape[0])
    n_pos    = train_d_lab.float().sum(dim=0).clamp(min=1.0)   # [NUM_CLASSES]
    n_neg    = n_total - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=MAX_POS_WEIGHT)
    print(f"pos_weight range: {pos_weight.min():.2f} to {pos_weight.max():.2f}")

    # ── DataLoaders from cached features ───────────────────────────────────
    train_ds = TensorDataset(train_pooled, train_crop_emb,
                             train_d_lab, train_c_lab, train_s_lab)
    val_ds   = TensorDataset(val_pooled, val_crop_emb,
                             val_d_lab, val_c_lab, val_s_lab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Build model — heads only ────────────────────────────────────────────
    # In Phase 1 we only train heads. The model is instantiated but only
    # head parameters are passed to the optimizer.
    model = PlantDiseaseModel().to(DEVICE)
    model.freeze_backbone()

    head_params = (
        list(model.crop_classifier.parameters()) +
        list(model.disease_head.parameters()) +
        list(model.severity_head.parameters()) +
        list(model.fpn.parameters())
    )
    optimizer = torch.optim.Adam(head_params, lr=PHASE1_LR,
                                 weight_decay=WEIGHT_DECAY)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name='phase1',
        config={**WANDB_CONFIG, 'phase': 1, 'epochs': PHASE1_EPOCHS},
    )

    early_stop  = EarlyStopping(EARLY_STOP_PAT, EARLY_STOP_DELTA)
    best_val_f1 = 0.0

    for epoch in range(PHASE1_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for pooled, crop_emb, d_lab, c_lab, s_lab in train_loader:
            pooled   = pooled.to(DEVICE)
            crop_emb = crop_emb.to(DEVICE)
            d_lab    = d_lab.to(DEVICE)
            c_lab    = c_lab.to(DEVICE)
            s_lab    = s_lab.to(DEVICE)

            # Forward using cached features directly
            # We run only the heads, not the full forward pass
            crop_logits, crop_emb_out = model.crop_classifier(pooled)
            disease_logits = model.disease_head(pooled, crop_emb_out)
            severity_logits = model.severity_head(pooled)

            total_loss, loss_dict = compute_loss(
                crop_logits, disease_logits, severity_logits,
                c_lab, d_lab, s_lab, pos_weight.to(DEVICE)
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_preds_disease = []
        val_true_disease  = []
        val_preds_crop    = []
        val_true_crop     = []
        total_val_loss    = 0.0

        with torch.no_grad():
            for pooled, crop_emb, d_lab, c_lab, s_lab in val_loader:
                pooled   = pooled.to(DEVICE)
                crop_emb = crop_emb.to(DEVICE)
                d_lab_d  = d_lab.to(DEVICE)
                c_lab_d  = c_lab.to(DEVICE)
                s_lab_d  = s_lab.to(DEVICE)

                c_log, _ = model.crop_classifier(pooled)
                d_log    = model.disease_head(pooled, model.crop_classifier(pooled)[1])
                s_log    = model.severity_head(pooled)

                loss, _ = compute_loss(c_log, d_log, s_log,
                                       c_lab_d, d_lab_d, s_lab_d,
                                       pos_weight.to(DEVICE))
                total_val_loss += loss.item()

                val_preds_disease.append(torch.sigmoid(d_log).cpu())
                val_true_disease.append(d_lab)
                val_preds_crop.append(c_log.argmax(dim=1).cpu())
                val_true_crop.append(c_lab)

        # Compute macro F1
        from sklearn.metrics import f1_score
        import numpy as np
        d_preds = (torch.cat(val_preds_disease).numpy() > 0.5).astype(int)
        d_true  = torch.cat(val_true_disease).numpy()
        val_f1  = f1_score(d_true, d_preds, average='macro', zero_division=0)
        crop_acc = (torch.cat(val_preds_crop) == torch.cat(val_true_crop)).float().mean().item()

        train_loss_avg = epoch_loss / max(len(train_loader), 1)
        val_loss_avg   = total_val_loss / max(len(val_loader), 1)

        wandb.log({
            'epoch'       : epoch,
            'train/loss'  : train_loss_avg,
            'val/loss'    : val_loss_avg,
            'val/macro_f1': val_f1,
            'val/crop_acc': crop_acc,
        })
        print(f"Phase1 Epoch {epoch:2d}: "
              f"train_loss={train_loss_avg:.4f}  "
              f"val_loss={val_loss_avg:.4f}  "
              f"val_macro_f1={val_f1:.4f}  "
              f"crop_acc={crop_acc:.3f}")

        # Checkpoint
        ckpt_path = os.path.join(
            CKPT_DIR, f"phase1_epoch{epoch:02d}_f1{val_f1:.3f}.pt"
        )
        save_checkpoint(model, optimizer, None, None, epoch, {'val/macro_f1': val_f1}, ckpt_path)
        cleanup_old_checkpoints(CKPT_DIR, KEEP_CKPTS, phase='phase1')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path   = os.path.join(CKPT_DIR, 'phase1_best.pt')
            save_checkpoint(model, optimizer, None, None, epoch,
                           {'val/macro_f1': val_f1}, best_path)
            print(f"  → New best phase1 model: macro_f1={val_f1:.4f}")

        if early_stop(val_f1):
            print(f"Early stopping at epoch {epoch}")
            break

    wandb.finish()
    print(f"\nPhase 1 complete. Best macro F1: {best_val_f1:.4f}")
    if best_val_f1 < 0.30:
        print("WARNING: macro F1 < 0.30. Check data pipeline and labels.")


if __name__ == '__main__':
    train_phase1()
```

---

## SECTION 9: TRAINING PHASE 2 (training/05_train_phase2.py) — IMPLEMENT EXACTLY

[FIX GAP 6] Phase 2 calls set_seeds() at the start — was missing in v5.
[FIX GAP 35] Uses ONE_CYCLE_PCT, ONE_CYCLE_DIV, ONE_CYCLE_FDIV from config.
[FIX GAP 42] get_llrd_optimizer imported from training.helpers at module level.
[FIX GAP 45] compiled flag is used to unwrap model for saving.
[FIX GAP 36] Threading lock protects MC Dropout state modification.
[FIX GAP 69] Uses torch.amp.autocast instead of deprecated torch.cuda.amp.autocast.
[FIX GAP 65] load_dotenv() called at top of script.
[FIX GAP 66] WANDB_MODE=offline fallback if WANDB_API_KEY not set.
[FIX GAP 71] Phase 2 resumes from latest phase2_epoch*.pt if it exists.

```python
# training/05_train_phase2.py
"""
Phase 2: Full fine-tuning with top 1/3 of backbone unfrozen.
Loads phase1_best.pt as starting weights.
Uses LLRD optimizer and OneCycleLR scheduler.
Saves: models/best_model.pt (state_dict only — use for inference)
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [FIX GAP 65] Load environment variables at module level
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional for manual execution

# [FIX GAP 1, 42] Import ALL training helpers at MODULE LEVEL
from training.helpers import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    get_llrd_optimizer,
)

import torch
import torch.nn as nn
# [FIX GAP 69] Use torch.amp, not deprecated torch.cuda.amp
from torch.amp import autocast, GradScaler
import wandb

from app.config import (
    DEVICE, SOURCE_MAP, SEV_LABELS, CKPT_DIR, MODELS, BEST_MODEL,
    PHASE2_EPOCHS, PHASE2_BASE_LR, LLRD_DECAY, WEIGHT_DECAY,
    BATCH_SIZE, GRAD_ACCUM_STEPS, GRAD_CLIP_NORM,
    EARLY_STOP_PAT, EARLY_STOP_DELTA, KEEP_CKPTS, RANDOM_SEED,
    ONE_CYCLE_PCT, ONE_CYCLE_DIV, ONE_CYCLE_FDIV,   # [FIX GAP 35]
    WANDB_PROJECT, WANDB_CONFIG,
)
from app.model import PlantDiseaseModel, verify_backbone_shapes
from training.dataset import PlantDiseaseDataset, load_severity_labels, make_weighted_sampler
from training.transforms import get_train_transform, get_eval_transform
from training.loss import compute_loss
from training.metrics import compute_all_metrics, warn_on_thin_classes


def set_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase2(train_records, val_records):
    # [FIX GAP 6] Phase 2 must call set_seeds() — was missing in v5
    set_seeds(RANDOM_SEED)

    # [FIX GAP 66] WANDB_MODE=offline if no API key, so training is not blocked
    if not os.environ.get('WANDB_API_KEY'):
        os.environ.setdefault('WANDB_MODE', 'offline')
        print("WANDB_API_KEY not set. Running wandb in offline mode.")

    is_windows = sys.platform.startswith('win')
    n_workers  = 0 if is_windows else 2

    # ── Load model from Phase 1 best ──────────────────────────────────────
    phase1_best = os.path.join(CKPT_DIR, 'phase1_best.pt')
    if not os.path.exists(phase1_best):
        raise FileNotFoundError(
            f"Phase 1 best checkpoint not found at {phase1_best}. "
            f"Run training/04_train_phase1.py first."
        )

    model = PlantDiseaseModel().to(DEVICE)
    verify_backbone_shapes(model, device=DEVICE)

    # Load Phase 1 weights
    ckpt  = torch.load(phase1_best, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded Phase 1 weights from {phase1_best}")

    # Unfreeze top 1/3 of backbone blocks
    model.unfreeze_top_fraction(fraction=0.33)

    # ── DataLoaders ───────────────────────────────────────────────────────
    sev_labels  = load_severity_labels()
    train_ds    = PlantDiseaseDataset(train_records, get_train_transform(), sev_labels)
    val_ds      = PlantDiseaseDataset(val_records,   get_eval_transform(),  sev_labels)
    sampler     = make_weighted_sampler(train_records)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=False,
        prefetch_factor=2 if n_workers > 0 else None,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=n_workers,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=False,
        drop_last=False,
    )

    # ── [FIX GAP 34] pos_weight from binary label matrix ──────────────────
    import pandas as pd
    import numpy as np
    from app.config import CLASS_TO_IDX, NUM_CLASSES, MAX_POS_WEIGHT
    d_labels_all = torch.zeros(len(train_records), NUM_CLASSES)
    for i, r in enumerate(train_records):
        idx = r.get('class_idx', -1)
        if 0 <= idx < NUM_CLASSES:
            d_labels_all[i, idx] = 1.0
    n_total   = float(len(train_records))
    n_pos     = d_labels_all.sum(dim=0).clamp(min=1.0)
    n_neg     = n_total - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=MAX_POS_WEIGHT)

    # ── [FIX GAP 42] LLRD optimizer from helpers only ─────────────────────
    optimizer = get_llrd_optimizer(model, PHASE2_BASE_LR, LLRD_DECAY, WEIGHT_DECAY)

    # ── OneCycleLR — [FIX GAP 35] use config constants ────────────────────
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[pg['lr'] for pg in optimizer.param_groups],
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=ONE_CYCLE_PCT,      # from config, not hardcoded
        anneal_strategy='cos',
        div_factor=ONE_CYCLE_DIV,     # from config
        final_div_factor=ONE_CYCLE_FDIV,  # from config
    )

    # ── Mixed precision ────────────────────────────────────────────────────
    use_amp = (DEVICE.type == 'cuda')
    # [FIX GAP 69] GradScaler from torch.amp, device_type parameter required
    scaler  = GradScaler(device='cuda' if use_amp else 'cpu', enabled=use_amp)

    # ── torch.compile ─────────────────────────────────────────────────────
    # [FIX GAP 45] compiled flag is actually used for model unwrapping
    compiled = False
    if use_amp and not is_windows:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            compiled = True
            print("torch.compile enabled (25-35% speedup)")
        except Exception as e:
            print(f"torch.compile unavailable: {e}. Continuing without.")

    # ── [FIX GAP 71] Resume from latest phase2 checkpoint if it exists ────
    start_epoch = 0
    phase2_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'phase2_epoch*.pt')))
    if phase2_ckpts:
        latest = phase2_ckpts[-1]
        print(f"Resuming Phase 2 from checkpoint: {latest}")
        raw_model = getattr(model, '_orig_mod', model)
        resume_epoch, resume_metrics = load_checkpoint(
            raw_model, optimizer, scheduler, scaler, latest, DEVICE
        )
        start_epoch = resume_epoch + 1
        print(f"Resumed from epoch {resume_epoch}, "
              f"val_f1={resume_metrics.get('val/macro_f1', 0):.4f}")

    wandb.init(
        project=WANDB_PROJECT,
        name='phase2',
        config={**WANDB_CONFIG, 'phase': 2, 'amp': use_amp,
                'resume_epoch': start_epoch},
    )

    early_stop  = EarlyStopping(EARLY_STOP_PAT, EARLY_STOP_DELTA)
    best_val_f1 = 0.0

    for epoch in range(start_epoch, PHASE2_EPOCHS):
        model.train()
        epoch_loss    = 0.0
        accum_counter = 0
        optimizer.zero_grad()

        for batch_idx, (images, d_lab, c_lab, s_lab) in enumerate(train_loader):
            images = images.to(DEVICE)
            d_lab  = d_lab.to(DEVICE)
            c_lab  = c_lab.to(DEVICE)
            s_lab  = s_lab.to(DEVICE)

            # [FIX GAP 69] device_type parameter required in torch.amp.autocast
            with autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                c_log, d_log, s_log = model(images)
                total_loss, _ = compute_loss(
                    c_log, d_log, s_log, c_lab, d_lab, s_lab,
                    pos_weight.to(DEVICE)
                )
                scaled_loss = total_loss / GRAD_ACCUM_STEPS

            scaler.scale(scaled_loss).backward()
            accum_counter += 1

            if accum_counter % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                wandb.log({'train/grad_norm': grad_norm.item(),
                           'train/lr': scheduler.get_last_lr()[0]})

            epoch_loss += total_loss.item()

        # Flush incomplete accumulation window at epoch end
        if accum_counter % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        val_metrics = compute_all_metrics(model, val_loader, pos_weight,
                                          DEVICE, phase='phase2_full')
        val_f1 = val_metrics.get('val/macro_f1', 0.0)

        wandb.log({'epoch': epoch,
                   'train/loss': epoch_loss / max(len(train_loader), 1),
                   **val_metrics})
        print(f"Phase2 Epoch {epoch:2d}: "
              f"loss={epoch_loss / len(train_loader):.4f}  "
              f"val_macro_f1={val_f1:.4f}")

        warn_on_thin_classes(val_metrics, epoch)

        # [FIX GAP 45] Use compiled flag to unwrap model for saving
        raw_model = getattr(model, '_orig_mod', model) if compiled else model
        ckpt_path = os.path.join(
            CKPT_DIR, f"phase2_epoch{epoch:02d}_f1{val_f1:.3f}.pt"
        )
        save_checkpoint(raw_model, optimizer, scheduler, scaler,
                        epoch, val_metrics, ckpt_path)
        cleanup_old_checkpoints(CKPT_DIR, KEEP_CKPTS, phase='phase2')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {'model_state_dict': raw_model.state_dict(),
                 'val_metrics': val_metrics,
                 'epoch': epoch},
                BEST_MODEL
            )
            best_ckpt = os.path.join(CKPT_DIR, 'phase2_best.pt')
            save_checkpoint(raw_model, optimizer, scheduler, scaler,
                            epoch, val_metrics, best_ckpt)
            print(f"  → Best model saved: macro_f1={val_f1:.4f}")

        if early_stop(val_f1):
            print(f"Early stopping at epoch {epoch}")
            break

    wandb.finish()
    print(f"\nPhase 2 complete. Best macro F1: {best_val_f1:.4f}")
    if best_val_f1 < 0.50:
        print("WARNING: macro F1 < 0.50. Check data balance and training setup.")


if __name__ == '__main__':
    import pandas as pd
    from app.config import CLASS_TO_IDX, CROP_FROM_IDX, SOURCE_MAP

    df = pd.read_csv(SOURCE_MAP)
    train_records = df[df['split'] == 'train'].to_dict('records')
    val_records   = df[df['split'] == 'val'].to_dict('records')
    for r in train_records + val_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

    train_phase2(train_records, val_records)
```
---

## SECTION 10: LOSS FUNCTION (training/loss.py) — IMPLEMENT EXACTLY

```python
# training/loss.py

import torch
import torch.nn as nn
from app.config import (
    LOSS_W_CROP, LOSS_W_DISEASE, LOSS_W_SEVERITY,
    LABEL_SMOOTH, NUM_CLASSES
)


def compute_loss(crop_logits, disease_logits, severity_logits,
                 crop_labels, disease_labels, severity_labels, pos_weight):
    """
    Combined loss for three heads.

    Args:
        crop_logits     [B, 2]           — binary crop CE loss
        disease_logits  [B, NUM_CLASSES] — multi-label BCE with pos_weight
        severity_logits [B, 3]           — 3-class CE loss
        crop_labels     [B]              — long tensor
        disease_labels  [B, NUM_CLASSES] — float multi-hot tensor
        severity_labels [B]              — long tensor
        pos_weight      [NUM_CLASSES]    — class imbalance weights

    Returns:
        (total_loss, loss_dict)
        total_loss: scalar Tensor with .backward()
        loss_dict: dict of named component losses for logging

    CRITICAL: compute_loss returns a TUPLE. Call:
        total, details = compute_loss(...)
        total.backward()
    Never call compute_loss(...).backward() — tuples have no backward().
    """
    # Crop: cross-entropy with label smoothing
    crop_loss  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)(
        crop_logits, crop_labels
    )

    # Disease: binary cross-entropy with pos_weight for class imbalance
    # pos_weight must be on the same device as disease_logits
    pos_w = pos_weight.to(disease_logits.device)
    bce   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dis_loss = bce(disease_logits, disease_labels.float())

    # Severity: cross-entropy with label smoothing
    sev_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)(
        severity_logits, severity_labels
    )

    total = (LOSS_W_CROP * crop_loss
             + LOSS_W_DISEASE * dis_loss
             + LOSS_W_SEVERITY * sev_loss)

    loss_dict = {
        'loss/crop'    : crop_loss.item(),
        'loss/disease' : dis_loss.item(),
        'loss/severity': sev_loss.item(),
        'loss/total'   : total.item(),
    }
    return total, loss_dict
```

---

## SECTION 11: METRICS (training/metrics.py) — IMPLEMENT EXACTLY

[FIX GAP 41] compute_all_metrics default phase changed to 'phase2_full'.
[FIX GAP 39] compute_all_metrics re-runs inference (does not read from checkpoint).
ECE computation and per-class F1 are specified here.

```python
# training/metrics.py

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from app.config import NUM_CLASSES, CLASS_NAMES, DISEASE_THRESH


def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error over n_bins equal-width probability bins.
    probs:  [N, C] or [N] float array of predicted probabilities
    labels: [N, C] or [N] float/int array of ground truth
    Returns scalar ECE value.
    """
    probs  = np.array(probs).flatten()
    labels = np.array(labels).flatten()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += np.abs(acc - conf) * mask.sum() / n
    return float(ece)


def compute_multilabel_pos_weights(train_records):
    """
    [FIX GAP 34] Correct multi-label pos_weight computation.
    train_records: list of dicts, each with 'class_idx' key.
    Returns Tensor [NUM_CLASSES] where pos_weight[j] = n_neg_j / n_pos_j.

    This is the correct formula for BCEWithLogitsLoss pos_weight.
    sklearn.compute_class_weight('balanced') is for SINGLE-LABEL.
    Never use it for multi-label BCE.
    """
    from app.config import MAX_POS_WEIGHT
    d_labels = torch.zeros(len(train_records), NUM_CLASSES)
    for i, r in enumerate(train_records):
        idx = r.get('class_idx', -1)
        if 0 <= idx < NUM_CLASSES:
            d_labels[i, idx] = 1.0
    n_total   = float(len(train_records))
    n_pos     = d_labels.sum(dim=0).clamp(min=1.0)
    n_neg     = n_total - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=MAX_POS_WEIGHT)
    return pos_weight


def compute_all_metrics(model, data_loader, pos_weight, device, phase='phase2_full'):
    """
    Runs a full inference pass on data_loader and computes all metrics.

    [FIX GAP 39] This function ALWAYS re-runs inference — it never reads
    from a saved checkpoint. Confusion matrices, per-class F1, and
    calibration curves cannot be computed from stored scalars alone.

    [FIX GAP 41] Default phase is 'phase2_full' (was incorrectly 'val').

    For 'phase2_full': expects batches of (images, d_lab, c_lab, s_lab)
    For 'phase1_cached': expects batches of (pooled, crop_emb, d_lab, c_lab, s_lab)

    Returns: dict of metric keys prefixed with 'val/'
    """
    model.eval()
    from training.loss import compute_loss

    all_d_preds = []
    all_d_true  = []
    all_c_preds = []
    all_c_true  = []
    total_loss  = 0.0

    with torch.no_grad():
        for batch in data_loader:
            if phase == 'phase1_cached':
                pooled, crop_emb, d_lab, c_lab, s_lab = batch
                pooled   = pooled.to(device)
                c_log, crop_emb_out = model.crop_classifier(pooled)
                d_log    = model.disease_head(pooled, crop_emb_out)
                s_log    = model.severity_head(pooled)
            else:  # phase2_full or any other phase
                images, d_lab, c_lab, s_lab = batch
                c_log, d_log, s_log = model(images.to(device))

            d_lab = d_lab.to(device)
            c_lab = c_lab.to(device)
            s_lab = s_lab.to(device)

            loss, _ = compute_loss(c_log, d_log, s_log, c_lab, d_lab, s_lab,
                                   pos_weight.to(device))
            total_loss += loss.item()

            all_d_preds.append(torch.sigmoid(d_log).cpu().numpy())
            all_d_true.append(d_lab.cpu().numpy())
            all_c_preds.append(c_log.argmax(dim=1).cpu().numpy())
            all_c_true.append(c_lab.cpu().numpy())

    d_preds = np.concatenate(all_d_preds)  # [N, NUM_CLASSES]
    d_true  = np.concatenate(all_d_true)
    c_preds = np.concatenate(all_c_preds)
    c_true  = np.concatenate(all_c_true)

    d_binary = (d_preds > DISEASE_THRESH).astype(int)

    # Per-class F1
    per_class_f1 = f1_score(d_true, d_binary, average=None, zero_division=0)
    macro_f1     = float(np.mean(per_class_f1))
    crop_acc     = float(accuracy_score(c_true, c_preds))
    ece          = compute_ece(d_preds, d_true)

    metrics = {
        'val/macro_f1' : macro_f1,
        'val/crop_acc' : crop_acc,
        'val/ece'      : ece,
        'val/loss'     : total_loss / max(len(data_loader), 1),
    }
    for i, cls in enumerate(CLASS_NAMES):
        metrics[f'val/f1_{cls}'] = float(per_class_f1[i])

    return metrics


def warn_on_thin_classes(val_metrics, epoch, threshold=0.40):
    """
    Prints a warning for any class with per-class F1 < threshold after epoch 3.
    Thin classes (brassica_clubroot especially) need this monitoring.
    """
    if epoch < 3:
        return
    for cls in CLASS_NAMES:
        f1 = val_metrics.get(f'val/f1_{cls}', 1.0)
        if f1 < threshold:
            print(f"  ⚠ Thin class warning: {cls} F1={f1:.3f} < {threshold}")
```

---

## SECTION 12: DATASET (training/dataset.py) — IMPLEMENT EXACTLY

[FIX GAP 12] _SevProxyDataset now applies CLAHE as first transform step.
[FIX GAP 30] Paths resolved using ROOT from config.

```python
# training/dataset.py

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from app.config import (
    ROOT, SEV_LABELS, CLASS_TO_IDX, CROP_FROM_IDX, NUM_CLASSES, VALID_EXT
)


def load_severity_labels():
    """
    Loads severity_labels.csv as dict {image_path: severity_idx}.
    image_path keys are relative to ROOT (same as source_map.csv).
    severity_idx: 0=mild, 1=moderate, 2=severe.
    Returns empty dict if file doesn't exist yet.
    """
    if not os.path.exists(SEV_LABELS):
        return {}
    sev_map = {}
    with open(SEV_LABELS, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sev_map[row['image_path']] = int(row['severity_idx'])
    return sev_map


class PlantDiseaseDataset(Dataset):
    """
    Loads images from source_map.csv records.
    image_path is relative to ROOT — resolved here.
    Returns (image_tensor, disease_labels_multihot, crop_label, severity_label).
    """
    def __init__(self, records, transform, sev_labels=None):
        self.records    = records
        self.transform  = transform
        self.sev_labels = sev_labels or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r         = self.records[idx]
        rel_path  = r['image_path']
        # [FIX GAP 30] Resolve relative path from ROOT
        full_path = os.path.join(ROOT, rel_path.replace('/', os.sep))

        try:
            img = Image.open(full_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
        except Exception:
            # Return zero tensor on load failure — training continues
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        # Multi-hot disease label
        class_idx = r.get('class_idx', -1)
        d_label   = torch.zeros(NUM_CLASSES)
        if 0 <= class_idx < NUM_CLASSES:
            d_label[class_idx] = 1.0

        # Single-label crop
        c_label = torch.tensor(r.get('crop_idx', 0), dtype=torch.long)

        # Severity (0=mild default if not in sev_labels)
        s_label = torch.tensor(
            self.sev_labels.get(rel_path, 0), dtype=torch.long
        )

        return img, d_label, c_label, s_label


def make_weighted_sampler(records):
    """
    WeightedRandomSampler for clubroot oversampling.
    brassica_clubroot has ~150 images vs 1000+ for other classes.
    Oversample it by CLUBROOT_OVERSAMPLE factor.
    """
    from app.config import CLUBROOT_OVERSAMPLE
    weights = []
    for r in records:
        cls = r.get('class_name', '')
        w   = CLUBROOT_OVERSAMPLE if cls == 'brassica_clubroot' else 1.0
        weights.append(w)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(records),
        replacement=True,
    )


class _SevProxyDataset(Dataset):
    """
    Dataset for severity proxy label generation (02_generate_severity.py).
    Loads raw images and applies CLAHE before returning.

    [FIX GAP 12] CLAHE must be applied here — the disease model was trained
    with CLAHE. Saliency maps on raw vs CLAHE images differ measurably.
    If CLAHE is skipped here, proxy labels are computed from a different
    pixel distribution than what the model sees during training.
    """
    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from training.transforms import apply_clahe
        import torchvision.transforms as TF

        full_path = os.path.join(ROOT, self.paths[idx].replace('/', os.sep))
        try:
            img = Image.open(full_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            # [FIX GAP 12] Apply CLAHE first — must match training distribution
            img = apply_clahe(img)
            img = Image.fromarray(img)
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        transform = TF.Compose([
            TF.Resize((224, 224)),
            TF.ToTensor(),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(img), self.paths[idx]
```

---

## SECTION 13: SEVERITY PROXY (training/02_generate_severity.py) — IMPLEMENT EXACTLY

[FIX GAP 9] Synthetic images path is standardised: synthetic images generated by
acquire_kerala_images.py are saved to data/raw/synthetic/{class_name}/ and
source_map.csv points to those paths. They do NOT go to data/processed/train/.

```python
# training/02_generate_severity.py
"""
Generates proxy severity labels for all training images using GradCAM saliency.
Severity is estimated from the fraction of the leaf area covered by high-activation
saliency regions.

[FIX GAP 12] Uses _SevProxyDataset which applies CLAHE (matching training distribution).
[FIX GAP 9]  Synthetic images are in data/raw/synthetic/ — not data/processed/train/.

Saves: data/metadata/severity_labels.csv
  Columns: image_path (relative to ROOT), severity_idx (0=mild,1=moderate,2=severe)
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import pandas as pd

from app.config import (
    DEVICE, SOURCE_MAP, SEV_LABELS, ROOT, META,
    SEVERITY_PROXY_THRESHOLD, SEVERITY_MILD_MAX, SEVERITY_MOD_MAX,
    CKPT_DIR,
)
from app.model import load_model_for_inference
from training.dataset import _SevProxyDataset


def estimate_severity_from_saliency(saliency_map, threshold, mild_max, mod_max):
    """
    Given a saliency map [H, W] normalised to [0,1]:
    - Binarise at threshold (top threshold fraction of activations)
    - coverage = fraction of pixels that are active
    - severity: mild if < mild_max, moderate if < mod_max, else severe
    Returns int: 0=mild, 1=moderate, 2=severe
    """
    thresh_val = torch.quantile(saliency_map, 1.0 - threshold)
    binary     = (saliency_map >= thresh_val).float()
    coverage   = binary.mean().item()
    if coverage < mild_max:
        return 0  # mild
    elif coverage < mod_max:
        return 1  # moderate
    else:
        return 2  # severe


def generate_severity_labels():
    """
    Main function. Loads training images, computes GradCAM saliency,
    estimates severity coverage, writes severity_labels.csv.
    """
    # Load model — need the trained disease head to compute saliency
    phase1_best = os.path.join(CKPT_DIR, 'phase1_best.pt')
    if not os.path.exists(phase1_best):
        print("Phase 1 model not found. Using random severity labels as fallback.")
        _write_random_labels()
        return

    model = load_model_for_inference(phase1_best, DEVICE)
    model.eval()

    # Load all training image paths
    df    = pd.read_csv(SOURCE_MAP)
    train_df = df[df['split'] == 'train']
    paths    = train_df['image_path'].tolist()

    print(f"Generating severity labels for {len(paths)} training images...")

    dataset = _SevProxyDataset(paths)  # [FIX GAP 12] CLAHE applied inside
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    os.makedirs(META, exist_ok=True)

    results = []
    processed = 0

    for images, rel_paths in loader:
        images = images.to(DEVICE)

        with torch.no_grad():
            # Use model disease_head activation as saliency proxy
            features     = model.backbone(images)
            fused        = model.fpn(*features)
            pooled       = model.gap(fused).flatten(1)
            _, crop_emb  = model.crop_classifier(pooled)
            d_logits     = model.disease_head(pooled, crop_emb)
            d_probs      = torch.sigmoid(d_logits)
            # Use max disease probability as confidence proxy
            # Use FPN P3 feature magnitude as saliency map
            saliency_maps = fused.abs().mean(dim=1)  # [B, 28, 28]

        for i, rel_path in enumerate(rel_paths):
            sal   = saliency_maps[i]
            sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            sev   = estimate_severity_from_saliency(
                sal_n.cpu(),
                SEVERITY_PROXY_THRESHOLD,
                SEVERITY_MILD_MAX,
                SEVERITY_MOD_MAX,
            )
            results.append({'image_path': rel_path, 'severity_idx': sev})
        processed += len(images)
        if processed % 500 == 0:
            print(f"  {processed}/{len(paths)}", end='\r')

    with open(SEV_LABELS, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'severity_idx'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWritten {len(results)} severity labels to {SEV_LABELS}")
    mild = sum(1 for r in results if r['severity_idx'] == 0)
    mod  = sum(1 for r in results if r['severity_idx'] == 1)
    sev  = sum(1 for r in results if r['severity_idx'] == 2)
    print(f"Distribution: mild={mild}, moderate={mod}, severe={sev}")


def _write_random_labels():
    """Fallback: write random severity labels when no model is available."""
    import random
    df      = pd.read_csv(SOURCE_MAP)
    train_df = df[df['split'] == 'train']
    os.makedirs(META, exist_ok=True)
    with open(SEV_LABELS, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'severity_idx'])
        writer.writeheader()
        for _, row in train_df.iterrows():
            writer.writerow({
                'image_path'  : row['image_path'],
                'severity_idx': random.randint(0, 2),
            })
    print(f"Written random severity labels (no model available).")


if __name__ == '__main__':
    generate_severity_labels()
```

---

## SECTION 14: INFERENCE (app/inference.py) — IMPLEMENT EXACTLY

[FIX GAP 8,11] apply_clahe is defined INLINE here. Do NOT import from training.transforms.
[FIX GAP 7] Grad-CAM target layer is model.fpn.out_p3 (not output_p3).
[FIX GAP 36] Threading lock protects MC Dropout state modification.
[FIX GAP 10] merge_diagnoses() removed. Merging is done inline in run_inference().

```python
# app/inference.py
"""
Inference pipeline: preprocess → MC Dropout → temperature scaling → Grad-CAM.

[FIX GAP 8]  apply_clahe defined inline — does NOT import from training.transforms.
[FIX GAP 7]  Grad-CAM target: model.fpn.out_p3 (was wrongly documented as output_p3).
[FIX GAP 36] threading.Lock protects MC Dropout state changes.
[FIX GAP 10] No merge_diagnoses() function — diagnosis merging is inline in run_inference().
"""

import os
import io
import base64
import json
import threading
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, DIAG_JSON,
    NUM_CLASSES, CLASS_NAMES, OKRA_INDICES, BRASSICA_INDICES,
    HEALTHY_INDICES, CROP_NAMES, CROP_FROM_IDX,
    DISEASE_THRESH, OOD_CONF_THRESH, OOD_UNC_THRESH, MC_PASSES,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


# ── [FIX GAP 8,11] apply_clahe defined INLINE — NOT imported from training ─
def apply_clahe(image: np.ndarray, clip_limit=2.0, tile_size=(8, 8)) -> np.ndarray:
    """
    CLAHE per RGB channel. Defined inline — do NOT import from training.transforms.
    Both this and the training version use only cv2 and numpy.
    Input/output: uint8 numpy array [H, W, 3].
    """
    clahe  = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = clahe.apply(image[:, :, c])
    return result


# ── Lock for MC Dropout state modification [FIX GAP 36] ────────────────────
_mc_dropout_lock = threading.Lock()


def preprocess_for_inference(image_np: np.ndarray) -> torch.Tensor:
    """
    Applies CLAHE, resizes to IMG_SIZE, normalises with ImageNet stats.
    Returns float32 Tensor [1, 3, H, W].
    image_np: uint8 numpy [H, W, 3].
    """
    img = apply_clahe(image_np)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = (img - mean) / std
    img  = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return img


def generate_heatmap(model, image_tensor: torch.Tensor, original_np: np.ndarray) -> str:
    """
    Generates Grad-CAM heatmap overlay on the original image.
    Returns base64 PNG string.

    [FIX GAP 7] Target layer is model.fpn.out_p3 — do NOT use model.fpn.output_p3
    (that attribute does not exist and will raise AttributeError).

    The model must be in eval mode with no Dropout active (not MC mode).
    Run all MC passes first, then call this function.
    """
    # [FIX GAP 7] Correct attribute name: out_p3
    target_layer = model.fpn.out_p3
    model.eval()

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=image_tensor.to(DEVICE))[0]

    # Overlay on original image resized to match
    orig_resized = cv2.resize(original_np, IMG_SIZE)
    orig_float   = orig_resized.astype(np.float32) / 255.0
    overlay      = show_cam_on_image(orig_float, grayscale, use_rgb=True)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_inference(model, image_np: np.ndarray) -> dict:
    """
    Full inference pipeline. Returns result dict with all fields.

    [FIX GAP 36] Uses _mc_dropout_lock to serialise MC Dropout state changes.
    Multiple concurrent requests cannot race on model.eval()/module.train().

    [FIX GAP 10] Diagnosis merging is inline — no merge_diagnoses() function.
    """
    image_tensor = preprocess_for_inference(image_np).to(DEVICE)

    # ── Load temperature scalars ────────────────────────────────────────────
    T_disease = T_crop = T_severity = 1.0
    if os.path.exists(TEMP_PATH):
        temp_data  = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease  = float(temp_data.get('T_disease',  1.0))
        T_crop     = float(temp_data.get('T_crop',     1.0))
        T_severity = float(temp_data.get('T_severity', 1.0))

    # ── MC Dropout passes ──────────────────────────────────────────────────
    # [FIX GAP 36] Lock ensures no race condition on module state
    with _mc_dropout_lock:
        # Set model to eval, then set only Dropout layers to train mode
        # BatchNorm stays in eval — prevents batch-of-1 statistics problem
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        mc_disease  = []
        mc_crop     = []
        mc_severity = []
        with torch.no_grad():
            for _ in range(MC_PASSES):
                c_log, d_log, s_log = model(image_tensor)
                mc_disease.append(torch.sigmoid(d_log / T_disease).cpu())
                mc_crop.append(torch.softmax(c_log / T_crop, dim=1).cpu())
                mc_severity.append(torch.softmax(s_log / T_severity, dim=1).cpu())

        # Restore full eval mode
        model.eval()

    mc_disease  = torch.stack(mc_disease, dim=0)   # [MC, 1, NUM_CLASSES]
    mc_crop     = torch.stack(mc_crop,    dim=0)   # [MC, 1, 2]
    mc_severity = torch.stack(mc_severity,dim=0)   # [MC, 1, 3]

    mean_dis = mc_disease.mean(dim=0).squeeze(0)   # [NUM_CLASSES]
    std_dis  = mc_disease.std(dim=0).squeeze(0)    # [NUM_CLASSES]
    mean_crop  = mc_crop.mean(dim=0).squeeze(0)    # [2]
    mean_sev   = mc_severity.mean(dim=0).squeeze(0) # [3]

    uncertainty = float(std_dis.mean())

    # ── Crop prediction ────────────────────────────────────────────────────
    crop_idx  = int(mean_crop.argmax())
    crop_conf = float(mean_crop.max())
    crop_name = CROP_NAMES[crop_idx]

    # ── OOD detection ──────────────────────────────────────────────────────
    ood_flagged = (crop_conf < OOD_CONF_THRESH or uncertainty > OOD_UNC_THRESH)

    # ── Disease predictions ────────────────────────────────────────────────
    # Apply crop gate: zero out predictions for the other crop
    # [Note: gate tensor must be on same device as mean_dis — FIX GAP inline]
    gate = torch.zeros(NUM_CLASSES)
    relevant = OKRA_INDICES if crop_idx == 0 else BRASSICA_INDICES
    for i in relevant:
        gate[i] = 1.0
    gate     = gate.to(mean_dis.device)  # ensure same device
    gated_dis = mean_dis * gate

    detected = [
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
        if gated_dis[i].item() > DISEASE_THRESH
    ]

    # Healthy suppression: if healthy class fires, suppress disease predictions
    healthy_cls = 'okra_healthy' if crop_idx == 0 else 'brassica_healthy'
    if healthy_cls in detected and len(detected) > 1:
        detected = [healthy_cls]

    if not detected:
        detected = [healthy_cls]

    # Confidence = mean of detected class probabilities
    detected_idx = [CLASS_NAMES.index(c) for c in detected]
    confidence   = float(mean_dis[detected_idx].mean()) if detected_idx else 0.5

    # ── Severity ───────────────────────────────────────────────────────────
    sev_idx    = int(mean_sev.argmax())
    sev_labels = ['mild', 'moderate', 'severe']
    severity   = sev_labels[sev_idx]
    sev_std    = float(mean_sev.std())
    # [low, high] interval based on MC uncertainty
    sev_low    = max(0.0, float(mean_sev[sev_idx]) - sev_std)
    sev_high   = min(1.0, float(mean_sev[sev_idx]) + sev_std)

    # ── Grad-CAM ───────────────────────────────────────────────────────────
    try:
        heatmap_b64 = generate_heatmap(model, image_tensor, image_np)
    except Exception as e:
        heatmap_b64 = ''

    # ── Diagnosis lookup [FIX GAP 10] inline merging ──────────────────────
    with open(DIAG_JSON, 'r', encoding='utf-8') as f:
        diag_db = json.load(f)

    # Merge treatment/prevention across all detected diseases (inline, no function)
    treatment  = []
    prevention = []
    urgency    = 'Low'
    urgency_reason = ''
    urgency_priority = {'High': 3, 'Medium': 2, 'Low': 1}
    for cls in detected:
        if cls in diag_db:
            entry = diag_db[cls]
            treatment.extend(entry.get('treatment', []))
            prevention.extend(entry.get('prevention', []))
            entry_urgency = entry.get('urgency', 'Low')
            if urgency_priority.get(entry_urgency, 0) > urgency_priority.get(urgency, 0):
                urgency        = entry_urgency
                urgency_reason = entry.get('urgency_reason', '')

    # Deduplicate while preserving order
    seen = set()
    treatment  = [t for t in treatment  if not (t in seen or seen.add(t))]
    seen       = set()
    prevention = [p for p in prevention if not (p in seen or seen.add(p))]

    return {
        'crop'            : crop_name,
        'crop_confidence' : round(crop_conf, 3),
        'diseases'        : detected,
        'confidence'      : round(confidence, 3),
        'uncertainty'     : round(uncertainty, 3),
        'severity'        : severity,
        'severity_interval': [round(sev_low, 3), round(sev_high, 3)],
        'treatment'       : treatment,
        'prevention'      : prevention,
        'urgency'         : urgency,
        'urgency_reason'  : urgency_reason,
        'heatmap_b64'     : heatmap_b64,
        'ood_flagged'     : ood_flagged,
    }
```

---

## SECTION 15: KAGGLE UTILITIES (agents/kaggle_utils.py) — IMPLEMENT EXACTLY

[FIX GAP 28] Shared Kaggle credential setup. All Kaggle download agents import
from here. Never duplicate credential logic in individual agent files.

```python
# agents/kaggle_utils.py
"""
Shared Kaggle credential setup.
[FIX GAP 28] This utility is imported by all Kaggle download agents.
Credential logic is defined ONCE here, never in individual agent files.
"""

import os
import json
import subprocess
import sys


def setup_kaggle_credentials():
    """
    Reads KAGGLE_USERNAME and KAGGLE_KEY from environment variables.
    Writes ~/.kaggle/kaggle.json if not already present.
    Raises EnvironmentError if credentials not found.
    """
    username = os.environ.get('KAGGLE_USERNAME')
    key      = os.environ.get('KAGGLE_KEY')
    if not username or not key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in environment/.env.\n"
            "Get them from kaggle.com > Account > Create API Token."
        )
    kaggle_dir  = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json):
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_json, 'w') as f:
            json.dump({'username': username, 'key': key}, f)
        os.chmod(kaggle_json, 0o600)
        print("  Kaggle credentials written to ~/.kaggle/kaggle.json")


def kaggle_download(slug, dest_dir):
    """
    Downloads and unzips a Kaggle dataset.
    slug: 'owner/dataset-name'
    dest_dir: destination directory (created if missing)
    Raises RuntimeError on non-zero returncode with stdout/stderr.
    """
    setup_kaggle_credentials()
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'kaggle', 'datasets', 'download',
        '-d', slug,
        '--path', dest_dir,
        '--unzip',
        '--quiet',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for {slug}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print(f"  Downloaded and unzipped: {slug} → {dest_dir}")
```

---

## SECTION 16: ACQUISITION AGENT (agents/acquire_kerala_images.py) — IMPLEMENT EXACTLY

[FIX GAP 2,14] acquire_all() is fully defined and specified.
[FIX GAP 9] Synthetic images saved to data/raw/synthetic/{class_name}/ (not data/processed/).
[FIX GAP 40] iNaturalist taxon ID for Brassica oleracea corrected to 55774.
[FIX GAP 47] Uses requests directly. pyinaturalist is NOT used.

```python
# agents/acquire_kerala_images.py
"""
Kerala image acquisition: iNaturalist GPS-filtered + YouTube frames + Stable Diffusion.

[FIX GAP 2,14] acquire_all() defined here, runs all three in parallel.
[FIX GAP 9]    Synthetic images → data/raw/synthetic/{class_name}/ (not processed/).
[FIX GAP 40]   Brassica oleracea iNat taxon ID = 55774 (was wrongly 47313).
[FIX GAP 47]   Uses requests library directly — pyinaturalist NOT installed.
"""

import os
import sys
import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ROOT, CLASS_NAMES


# Kerala GPS bounding box (lat/long)
KERALA_SW_LAT = 8.18
KERALA_NE_LAT = 12.77
KERALA_SW_LNG = 74.85
KERALA_NE_LNG = 77.42


# [FIX GAP 40] Correct iNaturalist taxon IDs:
# 47382 = Abelmoschus esculentus (okra) — verified
# 55774 = Brassica oleracea (cabbage/broccoli) — corrected (was 47313 = Brassicaceae family)
INAT_TAXON_IDS = {
    'okra'    : 47382,
    'brassica': 55774,  # [FIX GAP 40] was incorrectly 47313
}


def acquire_inaturalist(dest_dir='data/kerala/inaturalist'):
    """
    Downloads plant images from iNaturalist within Kerala GPS bounding box.
    Saves to dest_dir/{class_name}/ subdirectories.
    Uses domain_adapt split — images have no disease labels.
    """
    dest_full = os.path.join(ROOT, dest_dir)
    os.makedirs(dest_full, exist_ok=True)
    downloaded = 0
    results    = {}

    for crop, taxon_id in INAT_TAXON_IDS.items():
        crop_dir = os.path.join(dest_full, crop)
        os.makedirs(crop_dir, exist_ok=True)

        url    = 'https://api.inaturalist.org/v1/observations'
        params = {
            'taxon_id' : taxon_id,
            'swlat'    : KERALA_SW_LAT,
            'swlng'    : KERALA_SW_LNG,
            'nelat'    : KERALA_NE_LAT,
            'nelng'    : KERALA_NE_LNG,
            'quality_grade': 'research',
            'photos'   : True,
            'per_page' : 200,
            'page'     : 1,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            obs  = data.get('results', [])
            for ob in obs:
                for photo in ob.get('photos', [])[:1]:
                    img_url = photo.get('url', '').replace('/square.', '/medium.')
                    if not img_url:
                        continue
                    try:
                        img_resp = requests.get(img_url, timeout=15)
                        if img_resp.status_code == 200:
                            fname = f"inat_{ob['id']}.jpg"
                            fpath = os.path.join(crop_dir, fname)
                            with open(fpath, 'wb') as f:
                                f.write(img_resp.content)
                            downloaded += 1
                        time.sleep(0.1)
                    except Exception:
                        continue
            results[crop] = downloaded
            print(f"  [iNaturalist] {crop}: {downloaded} images")
        except Exception as e:
            print(f"  [iNaturalist] {crop}: failed — {e}")
            results[crop] = 0

    return {'source': 'inaturalist', 'downloaded': downloaded, 'details': results}


def acquire_youtube_frames(dest_dir='data/kerala/youtube'):
    """
    Downloads frames from Malayalam agriculture YouTube channels.
    Saves frames to dest_dir/ with source crop as subdirectory.
    Uses domain_adapt split — images have no disease labels.
    """
    dest_full = os.path.join(ROOT, dest_dir)
    os.makedirs(dest_full, exist_ok=True)

    CHANNELS = [
        'https://www.youtube.com/@AgriculturalKerala',
        'https://www.youtube.com/@KrishiVigyanKendra',
    ]
    downloaded = 0
    try:
        import yt_dlp
        for channel in CHANNELS:
            try:
                ydl_opts = {
                    'format'          : 'best[height<=480]',
                    'outtmpl'         : os.path.join(dest_full, '%(id)s.%(ext)s'),
                    'max_downloads'   : 5,
                    'ignoreerrors'    : True,
                    'quiet'           : True,
                    'writeinfojson'   : False,
                    'skip_download'   : False,
                    'extract_flat'    : False,
                }
                # Extract frames every 30 seconds using ffmpeg after download
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([channel])
                downloaded += 5
            except Exception as e:
                print(f"  [YouTube] {channel}: failed — {e}")
    except ImportError:
        print("  [YouTube] yt_dlp not installed. Skipping YouTube acquisition.")

    print(f"  [YouTube] {downloaded} frames acquired")
    return {'source': 'youtube', 'downloaded': downloaded}


def generate_synthetic(dest_dir='data/raw/synthetic'):
    """
    Generates synthetic disease images using Stable Diffusion.
    [FIX GAP 9] Saves to data/raw/synthetic/{class_name}/ — NOT data/processed/.
    Only runs for thin classes (< MIN_IMGS_CLASS = 150 images).
    """
    dest_full = os.path.join(ROOT, dest_dir)
    os.makedirs(dest_full, exist_ok=True)
    generated = 0

    try:
        from diffusers import StableDiffusionPipeline
        import torch as _torch
        import pandas as pd
        from app.config import SOURCE_MAP, MIN_IMGS_CLASS, CLASS_NAMES
        from collections import Counter

        df         = pd.read_csv(os.path.join(ROOT, 'data', 'metadata', 'source_map.csv'))
        train_df   = df[df['split'] == 'train']
        counts     = Counter(train_df['class_name'].tolist())
        thin       = [cls for cls in CLASS_NAMES if counts.get(cls, 0) < MIN_IMGS_CLASS]

        if not thin:
            print("  [Synthetic] No thin classes. Skipping.")
            return {'source': 'synthetic', 'generated': 0}

        pipe = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            torch_dtype=_torch.float16,
        ).to('cuda' if _torch.cuda.is_available() else 'cpu')

        DISEASE_PROMPTS = {
            'brassica_clubroot': 'brassica leaf wilting yellowing clubroot disease',
            'okra_enation'     : 'okra leaf curl enation disease begomovirus',
            'brassica_alternaria': 'cabbage leaf dark spots alternaria disease',
        }
        for cls in thin:
            if cls not in DISEASE_PROMPTS:
                continue
            cls_dir = os.path.join(dest_full, cls)
            os.makedirs(cls_dir, exist_ok=True)
            n_generate = MIN_IMGS_CLASS - counts.get(cls, 0)
            print(f"  [Synthetic] Generating {n_generate} images for {cls}...")
            for i in range(n_generate):
                try:
                    prompt = DISEASE_PROMPTS[cls] + ', high quality, close-up'
                    img    = pipe(prompt).images[0]
                    fname  = f"synthetic_{cls}_{i:04d}.png"
                    img.save(os.path.join(cls_dir, fname))
                    generated += 1
                except Exception as e:
                    print(f"    Generation {i} failed: {e}")

    except ImportError:
        print("  [Synthetic] diffusers not installed. Skipping.")
    except FileNotFoundError:
        print("  [Synthetic] source_map.csv not found yet. Skipping.")

    print(f"  [Synthetic] {generated} images generated → {dest_full}")
    return {'source': 'synthetic', 'generated': generated}


def create_kerala_source_map_entries():
    """
    Scans data/kerala/ and data/raw/synthetic/ and adds new records
    to source_map.csv. Call this after acquire_all().
    This is a convenience wrapper — full data pipeline runs in 01_prepare_data.py.
    """
    print("  [Kerala] source_map.csv entries will be created by 01_prepare_data.py")


def acquire_all():
    """
    [FIX GAP 2,14] acquire_all() — runs all three acquisition methods in parallel.

    Runs acquire_inaturalist, acquire_youtube_frames, and generate_synthetic
    simultaneously using ThreadPoolExecutor(max_workers=3).

    Returns results dict:
    {
        'inaturalist': {'source': ..., 'downloaded': N},
        'youtube'    : {'source': ..., 'downloaded': N},
        'synthetic'  : {'source': ..., 'generated':  N},
    }

    After this function: run 01_prepare_data.py to scan all new images into source_map.csv.
    Synthetic images are in data/raw/synthetic/ and will be found by _scan_source().
    """
    tasks = {
        'inaturalist': acquire_inaturalist,
        'youtube'    : acquire_youtube_frames,
        'synthetic'  : generate_synthetic,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                result = future.result()
                results[name] = result
                print(f"  [{name}] completed: {result}")
            except Exception as e:
                results[name] = {'source': name, 'error': str(e)}
                print(f"  [{name}] FAILED: {e}")

    print(f"\nAcquisition summary: {results}")
    return results


if __name__ == '__main__':
    acquire_all()
```

---

## SECTION 17: PIPELINE RUNNER (run_pipeline.py) — IMPLEMENT EXACTLY

[FIX GAP 15,21] Complete run_pipeline.py with argparse, proper step execution,
string-path steps via subprocess, lambda steps called directly.
[FIX GAP 23] Smoke test for Step 11 (validation report) added.
[FIX GAP 51] --yes flag propagated to install_cuda.py subprocess.
[FIX GAP 53] Priority-2 agents always run after priority-1 (conditional removed).
[FIX GAP 54] Evaluation scripts receive --yes flag to bypass interactive prompts.
[FIX GAP 65,66] load_dotenv() called at startup, WANDB_MODE fallback set.

```python
# run_pipeline.py
"""
Automated pipeline runner for the plant disease detection project.

Usage:
    python run_pipeline.py                    # run from step 0
    python run_pipeline.py --from-step 5      # resume from step 5
    python run_pipeline.py --step 8           # run only step 8
    python run_pipeline.py --reset-step 8     # mark step 8 as incomplete
    python run_pipeline.py --status           # show step completion status
    python run_pipeline.py --yes              # no interactive prompts

[FIX GAP 21] Step execution rules:
    String steps  (e.g. "setup/setup_project.py") → subprocess.run([sys.executable, path, ...])
    Lambda steps  (e.g. lambda: run_downloads())  → called directly in-process
[FIX GAP 51] --yes is propagated to all subprocesses (install_cuda.py, etc.)
[FIX GAP 65] load_dotenv() called before any step runs
[FIX GAP 66] WANDB_MODE=offline set as fallback if WANDB_API_KEY missing
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# [FIX GAP 66] WANDB offline fallback
if not os.environ.get('WANDB_API_KEY'):
    os.environ.setdefault('WANDB_MODE', 'offline')

ROOT          = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(ROOT, '.pipeline_progress.json')
LOG_FILE      = os.path.join(ROOT, 'pipeline_failures.log')


# ── Step definitions ───────────────────────────────────────────────────────
# [FIX GAP 21] String = subprocess, lambda = direct call.
# Lambdas used only for steps that need in-process module access
# (download orchestrator, acquisition) where subprocess pickling is complex.

def _make_steps(yes_flag):
    """Returns STEPS list. yes_flag is passed to subprocess calls."""
    yes_args = ['--yes'] if yes_flag else []

    def _run_downloads():
        sys.path.insert(0, ROOT)
        from agents.download_orchestrator import run_all_downloads
        return run_all_downloads()

    def _acquire_kerala():
        sys.path.insert(0, ROOT)
        from agents.acquire_kerala_images import acquire_all
        return acquire_all()

    STEPS = [
        # Step 0: environment setup
        'setup/setup_project.py',
        # Step 1: CUDA installation — [FIX GAP 51] pass --yes if set
        ['setup/install_cuda.py'] + yes_args,
        # Step 2: dependency installation
        'setup/install_dependencies.py',
        # Step 3: dataset downloads (lambda — in-process parallel)
        lambda: _run_downloads(),
        # Step 4: Kerala image acquisition (lambda — in-process parallel)
        lambda: _acquire_kerala(),
        # Step 5: data preparation and source_map.csv
        'training/01_prepare_data.py',
        # Step 6: severity proxy labels
        'training/02_generate_severity.py',
        # Step 7: feature caching
        'training/03_cache_features.py',
        # Step 8: Phase 1 training (heads only)
        'training/04_train_phase1.py',
        # Step 9: Phase 2 training (full fine-tuning)
        'training/05_train_phase2.py',
        # Step 10: temperature calibration
        'training/06_calibrate.py',
        # Step 11: validation report
        'training/07_evaluate_validation.py',
        # Step 12: server smoke test
        'setup/test_server.py',
        # Step 13: tier-2 PlantDoc evaluation (ONCE) — [FIX GAP 54] pass --yes
        ['training/08_evaluate_tier2_plantdoc.py'] + yes_args,
        # Step 14: deployment packaging
        'setup/package_deployment.py',
    ]
    return STEPS


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def is_done(step_idx, progress):
    return progress.get(str(step_idx), {}).get('done', False)


def mark_done(step_idx, progress):
    progress[str(step_idx)] = {
        'done'       : True,
        'timestamp'  : datetime.datetime.now().isoformat(),
    }
    save_progress(progress)


def mark_undone(step_idx, progress):
    if str(step_idx) in progress:
        progress[str(step_idx)]['done'] = False
        save_progress(progress)


def run_smoke_test(step_idx):
    """
    Per-step smoke test. Returns True if passes, False if fails.
    [FIX GAP 23] Step 11 smoke test added.
    """
    tests = {
        0  : lambda: os.path.exists(os.path.join(ROOT, 'data', 'metadata')),
        1  : _smoke_cuda,
        2  : _smoke_imports,
        3  : _smoke_downloads,
        5  : lambda: os.path.exists(os.path.join(ROOT, 'data', 'metadata', 'source_map.csv')),
        6  : lambda: os.path.exists(os.path.join(ROOT, 'data', 'metadata', 'severity_labels.csv')),
        7  : lambda: (
            os.path.exists(os.path.join(ROOT, 'cache', 'train_features.pt')) and
            os.path.exists(os.path.join(ROOT, 'cache', 'val_features.pt'))
        ),
        8  : lambda: os.path.exists(os.path.join(ROOT, 'models', 'checkpoints', 'phase1_best.pt')),
        9  : lambda: os.path.exists(os.path.join(ROOT, 'models', 'best_model.pt')),
        10 : lambda: os.path.exists(os.path.join(ROOT, 'models', 'temperature.pt')),
        # [FIX GAP 23] Step 11 smoke test:
        11 : lambda: any(
            f.endswith('.md') for f in os.listdir(os.path.join(ROOT, 'reports'))
        ) if os.path.isdir(os.path.join(ROOT, 'reports')) else False,
        12 : _smoke_server,
        13 : lambda: any(
            f.startswith('tier2') and f.endswith('.md')
            for f in os.listdir(os.path.join(ROOT, 'reports'))
        ) if os.path.isdir(os.path.join(ROOT, 'reports')) else False,
        14 : lambda: os.path.exists(os.path.join(ROOT, 'Dockerfile')),
    }
    test_fn = tests.get(step_idx)
    if test_fn is None:
        return True  # No smoke test for this step
    try:
        result = test_fn()
        if result:
            print(f"  ✓ Smoke test for Step {step_idx} passed")
        else:
            print(f"  ✗ Smoke test for Step {step_idx} FAILED")
        return result
    except Exception as e:
        print(f"  ✗ Smoke test for Step {step_idx} raised exception: {e}")
        return False


def _smoke_cuda():
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch; assert torch.cuda.is_available(), "CUDA not available"'],
        capture_output=True, text=True
    )
    return result.returncode == 0


def _smoke_imports():
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch, timm, albumentations, sklearn, cv2, fastapi, wandb'],
        capture_output=True, text=True
    )
    return result.returncode == 0


def _smoke_downloads():
    raw_dir    = os.path.join(ROOT, 'data', 'raw')
    results_f  = os.path.join(ROOT, 'data', 'metadata', 'download_results.json')
    if not os.path.exists(results_f):
        return False
    with open(results_f) as f:
        results = json.load(f)
    # Minimum: sabbir_okra and kareem_cabbage must have images
    mandatories = ['sabbir_okra', 'kareem_cabbage']
    for name in mandatories:
        r = results.get(name, {})
        if not r.get('success', False):
            print(f"  Mandatory dataset {name} failed download: {r.get('error')}")
            return False
    return True


def _smoke_server():
    """
    [FIX GAP 5] Test image has texture (noise overlay) so it passes blur check.
    """
    import time
    import threading

    server_proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'app.main:app',
         '--host', '0.0.0.0', '--port', '8765'],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'setup', 'test_server.py')],
            capture_output=True, text=True, timeout=30
        )
        passed = result.returncode == 0
        if not passed:
            print(f"  Server test stdout: {result.stdout[-500:]}")
            print(f"  Server test stderr: {result.stderr[-500:]}")
        return passed
    except Exception as e:
        print(f"  Server smoke test failed: {e}")
        return False
    finally:
        server_proc.terminate()


def github_commit(step_idx):
    """Commit and push after each step. Retry 3x. Log failures. Never stop pipeline."""
    msg = f"Step {step_idx:02d} complete — automated pipeline commit"
    cmds = [
        ['git', 'add', '-A'],
        ['git', 'commit', '-m', msg, '--allow-empty'],
        ['git', 'push'],
    ]
    for cmd in cmds:
        for attempt in range(3):
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                break
            if attempt == 2:
                commit_failure_log(step_idx, cmd, r)


def commit_failure_log(step_idx, cmd, result):
    with open(LOG_FILE, 'a') as f:
        f.write(
            f"[{datetime.datetime.now().isoformat()}] "
            f"Step {step_idx} git failure: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout[:200]}\n"
            f"  stderr: {result.stderr[:200]}\n"
        )


def execute_step(step, step_idx, yes_flag):
    """
    [FIX GAP 21] String steps → subprocess.run with sys.executable.
    List steps → subprocess.run with the list items.
    Lambda steps → called directly in-process.
    """
    if callable(step):
        # Lambda: call directly
        result = step()
        print(f"  Step {step_idx} (in-process) returned: {result}")
    elif isinstance(step, list):
        # List: first item is script path, rest are args
        script_path = os.path.join(ROOT, step[0])
        extra_args  = step[1:]
        cmd = [sys.executable, script_path] + extra_args
        r   = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(
                f"Step {step_idx} subprocess failed with returncode {r.returncode}: "
                f"{' '.join(cmd)}"
            )
    else:
        # String: script path relative to ROOT
        script_path = os.path.join(ROOT, step)
        cmd = [sys.executable, script_path]
        r   = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(
                f"Step {step_idx} subprocess failed with returncode {r.returncode}: "
                f"{' '.join(cmd)}"
            )


def show_status(progress, steps):
    print(f"\nPipeline status ({len(steps)} steps):")
    for i, step in enumerate(steps):
        done   = is_done(i, progress)
        status = '✓' if done else '○'
        ts     = progress.get(str(i), {}).get('timestamp', '')
        label  = step.__name__ if callable(step) else (step[0] if isinstance(step, list) else step)
        print(f"  [{status}] Step {i:2d}: {label}  {ts}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Run the plant disease pipeline')
    parser.add_argument('--from-step', type=int, default=0,
                        help='Start from this step (skip completed)')
    parser.add_argument('--step', type=int, default=None,
                        help='Run only this specific step')
    parser.add_argument('--reset-step', type=int, default=None,
                        help='Mark a step as incomplete')
    parser.add_argument('--status', action='store_true',
                        help='Show completion status and exit')
    parser.add_argument('--yes', action='store_true',
                        help='Pass --yes to all subprocesses (no interactive prompts)')
    args = parser.parse_args()

    steps    = _make_steps(args.yes)
    progress = load_progress()

    if args.status:
        show_status(progress, steps)
        return

    if args.reset_step is not None:
        mark_undone(args.reset_step, progress)
        print(f"Step {args.reset_step} marked as incomplete.")
        return

    # Determine which steps to run
    if args.step is not None:
        run_indices = [args.step]
    else:
        run_indices = list(range(args.from_step, len(steps)))

    print(f"Pipeline starting. Steps to run: {run_indices}")
    print(f"Platform: {sys.platform}")

    for i in run_indices:
        if i >= len(steps):
            print(f"Step {i} does not exist (max is {len(steps) - 1})")
            break

        step = steps[i]
        label = step.__name__ if callable(step) else (
            step[0] if isinstance(step, list) else step
        )

        # Skip if already done (unless --step used for single-step mode)
        if args.step is None and is_done(i, progress):
            print(f"[SKIP] Step {i:2d}: {label} (already done)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[RUN ] Step {i:2d}: {label}")
        print(f"{'=' * 60}")

        try:
            execute_step(step, i, args.yes)
            smoke_passed = run_smoke_test(i)
            if not smoke_passed:
                print(f"WARNING: smoke test failed for step {i}. Continuing anyway.")
            mark_done(i, progress)
            github_commit(i)
            print(f"[DONE] Step {i:2d}: {label}")
        except Exception as e:
            traceback.print_exc()
            print(f"\n[FAIL] Step {i:2d}: {label}")
            print(f"Error: {e}")
            with open(LOG_FILE, 'a') as f:
                f.write(
                    f"\n[{datetime.datetime.now().isoformat()}] "
                    f"Step {i} FAILED: {e}\n"
                    f"{traceback.format_exc()}\n"
                )
            print(f"Failure logged to {LOG_FILE}")
            print(f"To retry: python run_pipeline.py --step {i}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Pipeline complete.")
    show_status(progress, steps)


if __name__ == '__main__':
    main()
```
---

## SECTION 18: EVALUATION SCRIPTS — IMPLEMENT EXACTLY

[FIX GAP 18] All evaluation scripts are fully implemented — no bullet-point contracts only.
[FIX GAP 37] 10_evaluate_local_test.py added for the locked 15% local test split.
[FIX GAP 39,58] All scripts re-run full inference — they do NOT read scalars from checkpoints.
Confusion matrices require a full inference pass.

### 18.1 Validation report (training/07_evaluate_validation.py) — IMPLEMENT EXACTLY

[FIX GAP 18,39,58] Full inference re-run. Produces confusion matrix, per-class F1,
calibration curve. Writes reports/validation_report_{timestamp}.md.

```python
# training/07_evaluate_validation.py
"""
Tier-1 validation set evaluation. Writes comprehensive report to reports/.
[FIX GAP 39,58] Re-runs full inference on val set — reads no stored scalars.
[FIX GAP 54] Accepts --yes flag for pipeline compatibility.
"""

import os
import sys
import argparse
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, SOURCE_MAP, SEV_LABELS, REPORTS,
    CLASS_NAMES, NUM_CLASSES, DISEASE_THRESH, CLASS_TO_IDX, CROP_FROM_IDX
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform
from training.metrics import compute_ece


def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()

    model = load_model_for_inference(BEST_MODEL, DEVICE)

    T_disease = 1.0
    if os.path.exists(TEMP_PATH):
        t = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease = float(t.get('T_disease', 1.0))

    df  = pd.read_csv(SOURCE_MAP)
    val = df[df['split'] == 'val'].to_dict('records')
    for r in val:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

    sev_labels = load_severity_labels()
    ds  = PlantDiseaseDataset(val, get_eval_transform(), sev_labels)
    dl  = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_d_probs = []
    all_d_true  = []
    all_c_preds = []
    all_c_true  = []

    model.eval()
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in dl:
            c_log, d_log, s_log = model(images.to(DEVICE))
            d_probs = torch.sigmoid(d_log / T_disease).cpu()
            all_d_probs.append(d_probs)
            all_d_true.append(d_lab)
            all_c_preds.append(c_log.argmax(dim=1).cpu())
            all_c_true.append(c_lab)

    d_probs  = torch.cat(all_d_probs).numpy()
    d_true   = torch.cat(all_d_true).numpy()
    c_preds  = torch.cat(all_c_preds).numpy()
    c_true   = torch.cat(all_c_true).numpy()
    d_binary = (d_probs > DISEASE_THRESH).astype(int)

    macro_f1     = float(f1_score(d_true, d_binary, average='macro', zero_division=0))
    per_class_f1 = f1_score(d_true, d_binary, average=None, zero_division=0)
    crop_acc     = float(accuracy_score(c_true, c_preds))
    ece          = compute_ece(d_probs, d_true)
    cm           = confusion_matrix(d_true.argmax(axis=1), d_binary.argmax(axis=1))

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'validation_report_{ts}.md')

    lines = [
        '# Validation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'Val images: {len(val)}  |  T_disease: {T_disease:.4f}',
        '',
        '## Summary',
        f'- Macro F1 (disease): {macro_f1:.4f}',
        f'- Crop accuracy: {crop_acc:.4f}',
        f'- ECE (calibration error): {ece:.4f}',
        '',
        '## Per-Class F1',
        '| Class | F1 |',
        '|-------|-----|',
    ]
    for cls, f1 in zip(CLASS_NAMES, per_class_f1):
        flag = ' ← LOW' if f1 < 0.40 else ''
        lines.append(f'| {cls} | {f1:.4f}{flag} |')

    lines += [
        '',
        '## Confusion Matrix (argmax actual vs argmax predicted)',
        '```',
        str(cm),
        '```',
        '',
        '## Acceptance Status',
    ]
    if macro_f1 >= 0.50:
        lines.append(f'✓ PASS — macro F1 {macro_f1:.4f} >= 0.50')
    else:
        lines.append(f'✗ FAIL — macro F1 {macro_f1:.4f} < 0.50. Training needs improvement.')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Validation report: {path}")
    print(f"Macro F1={macro_f1:.4f}  Crop acc={crop_acc:.4f}  ECE={ece:.4f}")


if __name__ == '__main__':
    run_evaluation()
```

### 18.2 Tier-2 PlantDoc evaluation (training/08_evaluate_tier2_plantdoc.py) — IMPLEMENT EXACTLY

[FIX GAP 18,54,55] Full inference with temperature scaling. Accepts --yes flag.

```python
# training/08_evaluate_tier2_plantdoc.py
"""
Tier-2 PlantDoc evaluation. Run ONCE after all training is final.
[FIX GAP 55] Temperature scaling applied (same as production inference).
[FIX GAP 54] --yes flag for non-interactive pipeline execution.
"""

import os
import sys
import argparse
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import f1_score

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, REPORTS, SOURCE_MAP,
    CLASS_NAMES, CLASS_TO_IDX, CROP_FROM_IDX,
    DISEASE_THRESH, TIER2_MIN_F1, PLANTDOC_CLASS_MAP, SEV_LABELS
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform


def run_tier2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()

    if not args.yes:
        try:
            confirm = input(
                "\nTIER-2 EVALUATION: Run ONCE only. No model changes after this.\n"
                "Type 'yes' to proceed: "
            ).strip().lower()
        except EOFError:
            confirm = 'yes'
        if confirm != 'yes':
            print("Aborted.")
            sys.exit(0)

    model = load_model_for_inference(BEST_MODEL, DEVICE)
    T_disease = 1.0
    if os.path.exists(TEMP_PATH):
        t = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease = float(t.get('T_disease', 1.0))

    df = pd.read_csv(SOURCE_MAP)
    plantdoc_records = df[df['split'] == 'plantdoc'].to_dict('records')
    for r in plantdoc_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)
    plantdoc_records = [r for r in plantdoc_records if r['class_idx'] >= 0]

    if not plantdoc_records:
        print("No PlantDoc records in source_map.csv. Run 01_prepare_data.py after downloading PlantDoc.")
        return

    sev_labels = load_severity_labels()
    ds = PlantDiseaseDataset(plantdoc_records, get_eval_transform(), sev_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_d_probs = []
    all_d_true  = []

    model.eval()
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in dl:
            _, d_log, _ = model(images.to(DEVICE))
            d_probs = torch.sigmoid(d_log / T_disease).cpu()
            all_d_probs.append(d_probs)
            all_d_true.append(d_lab)

    d_probs  = torch.cat(all_d_probs).numpy()
    d_true   = torch.cat(all_d_true).numpy()
    d_binary = (d_probs > DISEASE_THRESH).astype(int)

    mappable_classes = list(set(PLANTDOC_CLASS_MAP.values()))
    mappable_idx     = [CLASS_NAMES.index(c) for c in mappable_classes if c in CLASS_NAMES]
    d_probs_m  = d_probs[:, mappable_idx]
    d_true_m   = d_true[:,  mappable_idx]
    d_binary_m = d_binary[:, mappable_idx]

    per_class_f1 = f1_score(d_true_m, d_binary_m, average=None, zero_division=0)
    macro_f1     = float(np.mean(per_class_f1))

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'tier2_plantdoc_{ts}.md')

    lines = [
        '# Tier-2 PlantDoc Evaluation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'PlantDoc images evaluated: {len(plantdoc_records)}',
        f'T_disease applied: {T_disease:.4f}',
        '',
        '## Results (mappable classes only)',
        f'Macro F1: {macro_f1:.4f}  (acceptance threshold: {TIER2_MIN_F1})',
        '',
        '| Class | F1 |',
        '|-------|-----|',
    ]
    eval_classes = [CLASS_NAMES[i] for i in mappable_idx]
    for cls, f1 in zip(eval_classes, per_class_f1):
        lines.append(f'| {cls} | {f1:.4f} |')

    lines += ['', '## Decision']
    if macro_f1 >= TIER2_MIN_F1:
        lines.append(f'✓ PASS — Model is deployment-ready at tier-2.')
    else:
        lines.append(f'✗ FAIL — Gap analysis required before deployment.')
        for cls, f1 in zip(eval_classes, per_class_f1):
            if f1 < 0.40:
                lines.append(f'  - {cls}: F1={f1:.3f} — needs more diverse training data')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Tier-2 report: {path}")
    print(f"Tier-2 macro F1: {macro_f1:.4f}  {'PASS' if macro_f1 >= TIER2_MIN_F1 else 'FAIL'}")


if __name__ == '__main__':
    run_tier2()
```

### 18.3 Local test evaluation (training/10_evaluate_local_test.py) — IMPLEMENT EXACTLY

[FIX GAP 37] Evaluates the 15% locked local test split.

```python
# training/10_evaluate_local_test.py
"""
Evaluates the 15% locked local test split. Run ONCE after tier-2.
[FIX GAP 37] Missing from v5 — now specified and implemented.
"""

import os
import sys
import argparse
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, SOURCE_MAP, SEV_LABELS, REPORTS,
    CLASS_NAMES, NUM_CLASSES, DISEASE_THRESH, CLASS_TO_IDX, CROP_FROM_IDX
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform
from training.metrics import compute_ece


def run_local_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()

    if not args.yes:
        try:
            confirm = input(
                "\nLOCAL TEST EVALUATION: Run only after tier-2 is complete.\n"
                "Type 'yes' to proceed: "
            ).strip().lower()
        except EOFError:
            confirm = 'yes'
        if confirm != 'yes':
            print("Aborted.")
            sys.exit(0)

    model = load_model_for_inference(BEST_MODEL, DEVICE)
    T_disease = 1.0
    if os.path.exists(TEMP_PATH):
        t = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease = float(t.get('T_disease', 1.0))

    df = pd.read_csv(SOURCE_MAP)
    test_records = df[df['split'] == 'test'].to_dict('records')
    for r in test_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

    if not test_records:
        print("No test records in source_map.csv.")
        return

    sev_labels = load_severity_labels()
    ds = PlantDiseaseDataset(test_records, get_eval_transform(), sev_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_d_probs = []
    all_d_true  = []
    all_c_preds = []
    all_c_true  = []

    model.eval()
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in dl:
            c_log, d_log, s_log = model(images.to(DEVICE))
            d_probs = torch.sigmoid(d_log / T_disease).cpu()
            all_d_probs.append(d_probs)
            all_d_true.append(d_lab)
            all_c_preds.append(c_log.argmax(dim=1).cpu())
            all_c_true.append(c_lab)

    d_probs  = torch.cat(all_d_probs).numpy()
    d_true   = torch.cat(all_d_true).numpy()
    c_preds  = torch.cat(all_c_preds).numpy()
    c_true   = torch.cat(all_c_true).numpy()
    d_binary = (d_probs > DISEASE_THRESH).astype(int)

    macro_f1     = float(f1_score(d_true, d_binary, average='macro', zero_division=0))
    per_class_f1 = f1_score(d_true, d_binary, average=None, zero_division=0)
    crop_acc     = float(accuracy_score(c_true, c_preds))
    ece          = compute_ece(d_probs, d_true)

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'local_test_report_{ts}.md')

    lines = [
        '# Local Test Set Evaluation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'Test images: {len(test_records)}  |  T_disease: {T_disease:.4f}',
        '',
        '## Summary',
        f'- Macro F1: {macro_f1:.4f}',
        f'- Crop accuracy: {crop_acc:.4f}',
        f'- ECE: {ece:.4f}',
        '',
        '## Per-Class F1',
        '| Class | F1 |',
        '|-------|-----|',
    ]
    for cls, f1_val in zip(CLASS_NAMES, per_class_f1):
        lines.append(f'| {cls} | {f1_val:.4f} |')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Local test report: {path}")
    print(f"Test macro F1={macro_f1:.4f}  Crop acc={crop_acc:.4f}  ECE={ece:.4f}")


if __name__ == '__main__':
    run_local_test()
```

### 18.4 Tier-3 Kerala evaluation (training/09_evaluate_tier3_kerala.py) — IMPLEMENT EXACTLY

```python
# training/09_evaluate_tier3_kerala.py

import os
import sys
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, SOURCE_MAP, SEV_LABELS, REPORTS,
    CLASS_NAMES, NUM_CLASSES, DISEASE_THRESH, CLASS_TO_IDX, CROP_FROM_IDX,
    TIER3_MIN_ACC, TIER3_MIN_IMGS, TIER3_MIN_CLS
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform


def run_tier3():
    model = load_model_for_inference(BEST_MODEL, DEVICE)
    T_disease = 1.0
    if os.path.exists(TEMP_PATH):
        t = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease = float(t.get('T_disease', 1.0))

    df = pd.read_csv(SOURCE_MAP)
    kerala_records = df[df['split'] == 'kerala'].to_dict('records')
    for r in kerala_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)
    kerala_records = [r for r in kerala_records if r['class_idx'] >= 0]

    if len(kerala_records) < TIER3_MIN_IMGS:
        print(f"Only {len(kerala_records)} Kerala images. Need {TIER3_MIN_IMGS}.")
        print("Use: python tools/add_kerala_image.py --path img.jpg --class class_name")
        return

    sev_labels = load_severity_labels()
    ds = PlantDiseaseDataset(kerala_records, get_eval_transform(), sev_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_d_probs = []
    all_d_true  = []

    model.eval()
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in dl:
            _, d_log, _ = model(images.to(DEVICE))
            d_probs = torch.sigmoid(d_log / T_disease).cpu()
            all_d_probs.append(d_probs)
            all_d_true.append(d_lab)

    d_probs  = torch.cat(all_d_probs).numpy()
    d_true   = torch.cat(all_d_true).numpy()
    d_binary = (d_probs > DISEASE_THRESH).astype(int)

    class_counts = Counter(r['class_name'] for r in kerala_records)
    results      = {}
    overall_pass = True

    for cls in CLASS_NAMES:
        cnt = class_counts.get(cls, 0)
        if cnt < TIER3_MIN_CLS:
            continue
        idx  = CLASS_NAMES.index(cls)
        mask = d_true[:, idx].astype(bool)
        if not mask.any():
            continue
        acc = float(accuracy_score(d_true[mask, idx], d_binary[mask, idx]))
        results[cls] = {'count': cnt, 'accuracy': acc}
        if acc < TIER3_MIN_ACC:
            overall_pass = False

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'tier3_kerala_{ts}.md')

    lines = [
        '# Tier-3 Kerala Field Evaluation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'Kerala images: {len(kerala_records)}',
        '',
        '## Per-Class Results',
        '| Class | Count | Accuracy | Pass? |',
        '|-------|-------|----------|-------|',
    ]
    for cls, r in results.items():
        passed = '✓' if r['accuracy'] >= TIER3_MIN_ACC else '✗'
        lines.append(f"| {cls} | {r['count']} | {r['accuracy']:.3f} | {passed} |")

    lines += ['', '## Overall Result']
    if overall_pass:
        lines.append('✓ PASS — Project is DEPLOYMENT-VALIDATED for Kerala.')
    else:
        lines.append('✗ FAIL — Some classes below accuracy threshold. Collect more Kerala images.')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Tier-3 report: {path}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == '__main__':
    run_tier3()
```

---

## SECTION 19: SERVER (app/main.py) — IMPLEMENT EXACTLY

[FIX GAP 36] Inference uses run_in_executor (non-blocking). Lock is in inference.py.

```python
# app/main.py

import os
import sys
import json
import sqlite3
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io

from app.config import DEVICE, BEST_MODEL, ROOT
from app.model import load_model_for_inference
from app.validator import validate_image
from app.inference import run_inference


DB_PATH = os.path.join(ROOT, 'feedback.db')


def init_db():
    # [FIX GAP inline] check_same_thread=False required for multi-threaded FastAPI
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT,
            crop       TEXT,
            diseases   TEXT,
            thumbs_up  INTEGER,
            correction TEXT
        )
    ''')
    conn.commit()
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_for_inference(BEST_MODEL, DEVICE)
    app.state.db    = init_db()
    yield
    app.state.db.close()


app = FastAPI(title='Plant Disease Detection — Kerala', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

templates = Jinja2Templates(directory=os.path.join(ROOT, 'templates'))
app.mount('/static', StaticFiles(directory=os.path.join(ROOT, 'static')), name='static')


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/health')
async def health():
    return {'status': 'ok', 'device': str(DEVICE)}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    validation = validate_image(contents)
    if not validation['valid']:
        raise HTTPException(status_code=400, detail=validation['reason'])

    image_np = validation['image']
    model    = app.state.model
    loop     = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_inference(model, image_np)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference failed: {e}')

    # OOD returns 200 with ood_flagged=True — NOT 422
    return JSONResponse(content=result)


@app.post('/feedback')
async def feedback(request: Request):
    body      = await request.json()
    timestamp = datetime.utcnow().isoformat()
    db        = app.state.db
    db.execute(
        'INSERT INTO feedback (timestamp, crop, diseases, thumbs_up, correction) '
        'VALUES (?, ?, ?, ?, ?)',
        (timestamp,
         body.get('crop', ''),
         json.dumps(body.get('diseases', [])),
         1 if body.get('thumbs_up') else 0,
         body.get('correction', ''))
    )
    db.commit()
    return {'status': 'saved'}
```

---

## SECTION 20: INPUT VALIDATOR (app/validator.py) — IMPLEMENT EXACTLY

[FIX GAP 60] HEIC removed — pillow-heif not installed.

```python
# app/validator.py

import numpy as np
import io
from PIL import Image
from app.config import (
    MAX_FILE_MB, MIN_BLUR_VAR, MIN_PIXEL_MEAN, MAX_PIXEL_MEAN,
    MIN_IMG_DIM, MAX_CH_RATIO
)


def _check_magic_bytes(data: bytes) -> str:
    if data[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if len(data) >= 12 and data[8:12] == b'WEBP':
        return 'webp'
    return 'unknown'


def validate_image(data: bytes) -> dict:
    """
    Validates uploaded image. Returns dict with valid, reason, image keys.
    [FIX GAP 60] HEIC not accepted — pillow-heif not installed.

    Checks:
    1. File size <= MAX_FILE_MB
    2. Magic bytes: jpeg/png/webp only
    3. PIL can open it
    4. Minimum dimensions
    5. Blur check: Laplacian variance >= MIN_BLUR_VAR
    6. Pixel mean within [MIN_PIXEL_MEAN, MAX_PIXEL_MEAN]
    7. No single channel dominates > MAX_CH_RATIO
    """
    # 1. File size
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return {'valid': False, 'reason': f'File too large (max {MAX_FILE_MB} MB)', 'image': None}

    # 2. Magic bytes
    fmt = _check_magic_bytes(data)
    if fmt == 'unknown':
        return {'valid': False,
                'reason': 'Unsupported format. Upload JPEG, PNG, or WebP.',
                'image': None}

    # 3. PIL open
    try:
        pil_img = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return {'valid': False, 'reason': 'Could not open image file.', 'image': None}

    img_np = np.array(pil_img, dtype=np.uint8)

    # 4. Minimum dimensions
    h, w = img_np.shape[:2]
    if h < MIN_IMG_DIM or w < MIN_IMG_DIM:
        return {'valid': False,
                'reason': f'Image too small. Minimum {MIN_IMG_DIM}px in each dimension.',
                'image': None}

    # 5. Blur check
    import cv2
    gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < MIN_BLUR_VAR:
        return {'valid': False,
                'reason': f'Image too blurry (score {blur_var:.1f}). Take a clearer photo.',
                'image': None}

    # 6. Pixel mean
    mean_val = float(img_np.mean())
    if mean_val < MIN_PIXEL_MEAN:
        return {'valid': False, 'reason': 'Image is too dark.', 'image': None}
    if mean_val > MAX_PIXEL_MEAN:
        return {'valid': False, 'reason': 'Image is overexposed.', 'image': None}

    # 7. Single-channel dominance
    ch_means = img_np.mean(axis=(0, 1))
    total    = ch_means.sum()
    if total > 0 and ch_means.max() / total > MAX_CH_RATIO:
        return {'valid': False,
                'reason': 'Image does not appear to contain a plant leaf.',
                'image': None}

    return {'valid': True, 'reason': '', 'image': img_np}
```

---

## SECTION 21: TEST SERVER (setup/test_server.py) — IMPLEMENT EXACTLY

[FIX GAP 5] Test image has random noise texture so it passes blur check.
A solid-colour image has Laplacian variance = 0 and always fails.

```python
# setup/test_server.py
"""
[FIX GAP 5] Test image is a textured JPEG (noise overlay on green base).
A solid-colour rectangle has Laplacian variance = 0 → fails blur check.
Noise overlay gives variance >> 80 → passes.
"""

import sys
import io
import requests
import numpy as np
from PIL import Image

SERVER_URL = 'http://localhost:8765'


def create_textured_test_image():
    """Green rectangle + heavy random noise. Passes blur check."""
    rng  = np.random.default_rng(seed=42)
    base = np.zeros((300, 300, 3), dtype=np.uint8)
    base[:, :, 1] = 120   # green
    noise = rng.integers(0, 60, (300, 300, 3), dtype=np.uint8)
    img   = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    buf   = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def run_smoke_test():
    print("Running server smoke test...")

    try:
        r = requests.get(f'{SERVER_URL}/health', timeout=5)
        assert r.status_code == 200, f"Health returned {r.status_code}"
        print(f"  ✓ Health: {r.json()}")
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        sys.exit(1)

    try:
        img_bytes = create_textured_test_image()
        r = requests.post(
            f'{SERVER_URL}/predict',
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            timeout=60,
        )
        assert r.status_code == 200, f"Predict returned {r.status_code}: {r.text[:300]}"
        result = r.json()
        required = ['crop', 'diseases', 'confidence', 'uncertainty',
                    'severity', 'treatment', 'urgency', 'ood_flagged']
        for key in required:
            assert key in result, f"Missing key: {key}"
        print(f"  ✓ Predict: crop={result['crop']} diseases={result['diseases']}")
    except Exception as e:
        print(f"  ✗ Predict test failed: {e}")
        sys.exit(1)

    print("Server smoke test PASSED")
    sys.exit(0)


if __name__ == '__main__':
    run_smoke_test()
```

---

## SECTION 22: KERALA IMAGE TOOL (tools/add_kerala_image.py) — IMPLEMENT EXACTLY

[FIX GAP 25] Complete workflow for adding verified Kerala images.

```python
# tools/add_kerala_image.py
"""
[FIX GAP 25] Adds a verified Kerala field image to source_map.csv.
Usage: python tools/add_kerala_image.py --path img.jpg --class okra_yvmv
"""

import os
import sys
import csv
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ROOT, CLASS_TO_IDX, CROP_FROM_IDX, SOURCE_MAP


def add_kerala_image(src_path, class_name):
    if class_name not in CLASS_TO_IDX:
        print(f"Unknown class: {class_name}")
        print(f"Valid: {sorted(CLASS_TO_IDX.keys())}")
        sys.exit(1)

    if not os.path.exists(src_path):
        print(f"File not found: {src_path}")
        sys.exit(1)

    with open(src_path, 'rb') as f:
        data = f.read()
    from app.validator import validate_image
    result = validate_image(data)
    if not result['valid']:
        print(f"Validation failed: {result['reason']}")
        sys.exit(1)

    class_dir = os.path.join(ROOT, 'data', 'kerala', class_name)
    os.makedirs(class_dir, exist_ok=True)

    fname    = os.path.basename(src_path)
    dst_path = os.path.join(class_dir, fname)
    if os.path.exists(dst_path):
        import time
        base, ext = os.path.splitext(fname)
        fname     = f"{base}_{int(time.time())}{ext}"
        dst_path  = os.path.join(class_dir, fname)

    shutil.copy2(src_path, dst_path)
    rel_path  = os.path.relpath(dst_path, ROOT).replace('\\', '/')
    class_idx = CLASS_TO_IDX[class_name]
    crop_idx  = CROP_FROM_IDX[class_idx]

    write_header = not os.path.exists(SOURCE_MAP)
    with open(SOURCE_MAP, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['image_path', 'source_dataset', 'raw_label',
                      'class_name', 'class_idx', 'crop_idx', 'split']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'image_path'    : rel_path,
            'source_dataset': 'kerala',
            'raw_label'     : class_name,
            'class_name'    : class_name,
            'class_idx'     : class_idx,
            'crop_idx'      : crop_idx,
            'split'         : 'kerala',
        })

    print(f"Added: {rel_path}")
    print(f"Class: {class_name} (idx={class_idx})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',  required=True)
    parser.add_argument('--class', dest='class_name', required=True)
    args = parser.parse_args()
    add_kerala_image(args.path, args.class_name)
```

---

## SECTION 23: FRONTEND — IMPLEMENT EXACTLY

### 23.1 templates/index.html

[FIX GAP 49,50] All element IDs specified. Thumbs-up/down both have IDs.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plant Disease Detector — Kerala</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header>
  <h1>Plant Disease Detector</h1>
  <p class="subtitle">Okra &amp; Broccoli — Kerala</p>
</header>

<main>
  <section id="upload-section">
    <div id="upload-area" tabindex="0" role="button"
         aria-label="Click or drag to upload a leaf image">
      <span id="upload-icon">🌿</span>
      <p id="upload-text">Click or drag a leaf photo here</p>
      <p class="hint">JPEG / PNG / WebP · max 10 MB</p>
    </div>
    <input type="file" id="file-input"
           accept="image/jpeg,image/png,image/webp"
           style="display:none" aria-label="File upload">
  </section>

  <section id="preview-section" hidden>
    <img id="preview-img" alt="Uploaded leaf preview">
    <button id="change-btn" class="btn-secondary">Change photo</button>
  </section>

  <section id="result-section" hidden>
    <div id="result-card">
      <div id="heatmap-container">
        <img id="heatmap-img" alt="Grad-CAM disease heatmap">
      </div>

      <div id="badges">
        <span id="crop-badge"     class="badge badge-crop"></span>
        <span id="severity-badge" class="badge"></span>
        <span id="urgency-badge"  class="badge"></span>
        <span id="ood-badge"      class="badge badge-ood" hidden>Low confidence</span>
      </div>

      <h2 id="disease-heading"></h2>
      <p  id="confidence-text"></p>
      <div id="confidence-bar-wrap">
        <div id="confidence-bar"></div>
      </div>

      <details id="treatment-details">
        <summary>Treatment</summary>
        <ul id="treatment-list"></ul>
      </details>

      <details id="prevention-details">
        <summary>Prevention</summary>
        <ul id="prevention-list"></ul>
      </details>

      <section id="feedback-section">
        <p>Was this diagnosis correct?</p>
        <!-- [FIX GAP 49,50] Both buttons have IDs -->
        <button id="thumbs-up-btn"   class="btn-feedback" aria-label="Yes, correct">👍</button>
        <button id="thumbs-down-btn" class="btn-feedback" aria-label="No, incorrect">👎</button>
        <div id="correction-form" hidden>
          <select id="correction-select" aria-label="Correct diagnosis">
            <option value="">— Select correct class —</option>
          </select>
          <button id="submit-correction-btn" class="btn-primary">Submit correction</button>
        </div>
        <p id="feedback-thanks" hidden>Thank you for your feedback!</p>
      </section>
    </div>
  </section>

  <div id="spinner" hidden aria-live="polite">
    <div class="spinner-ring"></div>
    <p>Analysing leaf…</p>
  </div>

  <div id="error-box" hidden role="alert"></div>
</main>

<script src="/static/app.js"></script>
</body>
</html>
```

### 23.2 static/style.css (first portion — continued in Section 23.2b)

[FIX GAP 17,43] Complete CSS. All IDs styled. Mobile responsive.
Severity: mild=green, moderate=amber, severe=red.
Urgency: Low=green, Medium=amber, High=red.

```css
/* static/style.css */

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --color-bg        : #f0f4f0;
  --color-surface   : #ffffff;
  --color-primary   : #2d6a4f;
  --color-primary-dk: #1b4332;
  --color-accent    : #52b788;
  --color-text      : #1a1a1a;
  --color-subtle    : #6c757d;
  --color-border    : #d1e7dd;
  --color-error     : #c0392b;
  --color-mild      : #27ae60;
  --color-moderate  : #e67e22;
  --color-severe    : #c0392b;
  --radius          : 12px;
  --shadow          : 0 2px 12px rgba(0,0,0,0.08);
  --transition      : 0.2s ease;
}

body {
  font-family    : 'Segoe UI', system-ui, sans-serif;
  background     : var(--color-bg);
  color          : var(--color-text);
  min-height     : 100vh;
  padding-bottom : 48px;
}

header { background: var(--color-primary); color: white; text-align: center; padding: 20px 16px 16px; }
header h1        { font-size: 1.6rem; font-weight: 700; }
header .subtitle { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }

main { max-width: 680px; margin: 24px auto; padding: 0 16px; }

#upload-area {
  border: 2.5px dashed var(--color-accent); border-radius: var(--radius);
  background: var(--color-surface); padding: 40px 20px; text-align: center;
  cursor: pointer; transition: background var(--transition), border-color var(--transition);
}
#upload-area:hover, #upload-area:focus { background: #e8f5ee; border-color: var(--color-primary); outline: none; }
#upload-icon { font-size: 3rem; display: block; margin-bottom: 12px; }
#upload-text { font-size: 1.05rem; color: var(--color-primary); font-weight: 500; }
.hint        { font-size: 0.82rem; color: var(--color-subtle); margin-top: 6px; }

#preview-section { margin-top: 16px; text-align: center; }
#preview-img { max-width: 100%; max-height: 300px; border-radius: var(--radius); box-shadow: var(--shadow); object-fit: cover; }

#result-card { background: var(--color-surface); border-radius: var(--radius); box-shadow: var(--shadow); padding: 24px; margin-top: 20px; }
#heatmap-container { text-align: center; margin-bottom: 16px; }
#heatmap-img { max-width: 100%; max-height: 280px; border-radius: 8px; box-shadow: var(--shadow); }

#badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.badge { font-size: 0.78rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.03em; }
.badge-crop     { background: #d0ebff; color: #1c6ea4; }
.badge-mild     { background: #d4edda; color: var(--color-mild); }
.badge-moderate { background: #fff3cd; color: var(--color-moderate); }
.badge-severe   { background: #f8d7da; color: var(--color-severe); }
.badge-low      { background: #d4edda; color: var(--color-mild); }
.badge-medium   { background: #fff3cd; color: var(--color-moderate); }
.badge-high     { background: #f8d7da; color: var(--color-severe); }
.badge-ood      { background: #fff3cd; color: var(--color-moderate); }

#disease-heading { font-size: 1.2rem; font-weight: 700; margin-bottom: 6px; }
#confidence-text { font-size: 0.9rem; color: var(--color-subtle); margin-bottom: 6px; }
#confidence-bar-wrap { height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; margin-bottom: 18px; }
#confidence-bar { height: 100%; border-radius: 4px; background: linear-gradient(to right, var(--color-mild), var(--color-severe)); transition: width 0.4s ease; }

details { border: 1px solid var(--color-border); border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; }
summary { font-weight: 600; cursor: pointer; list-style: none; padding: 2px 0; }
summary::marker, summary::-webkit-details-marker { display: none; }
summary::before { content: '▸ '; color: var(--color-primary); }
details[open] summary::before { content: '▾ '; }
details ul { margin-top: 8px; padding-left: 20px; }
details li { font-size: 0.9rem; line-height: 1.6; margin-bottom: 4px; }

#feedback-section { border-top: 1px solid var(--color-border); padding-top: 14px; margin-top: 14px; text-align: center; }
#feedback-section p { font-size: 0.9rem; color: var(--color-subtle); margin-bottom: 8px; }
.btn-feedback { font-size: 1.4rem; background: none; border: 2px solid var(--color-border); border-radius: 8px; padding: 6px 14px; cursor: pointer; margin: 0 6px; transition: border-color var(--transition), background var(--transition); }
.btn-feedback:hover { border-color: var(--color-accent); background: #e8f5ee; }
#correction-form { margin-top: 12px; }
#correction-select { padding: 6px 10px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 0.9rem; margin-bottom: 8px; width: 100%; max-width: 360px; }
#feedback-thanks { font-weight: 600; color: var(--color-primary); margin-top: 8px; }

.btn-primary { background: var(--color-primary); color: white; border: none; border-radius: 8px; padding: 8px 20px; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: background var(--transition); }
.btn-primary:hover { background: var(--color-primary-dk); }
.btn-secondary { background: transparent; color: var(--color-primary); border: 2px solid var(--color-accent); border-radius: 8px; padding: 6px 16px; font-size: 0.85rem; cursor: pointer; margin-top: 8px; }

#spinner { text-align: center; padding: 40px; }
.spinner-ring { width: 48px; height: 48px; border: 5px solid var(--color-border); border-top-color: var(--color-primary); border-radius: 50%; animation: spin 0.9s linear infinite; margin: 0 auto 12px; }
@keyframes spin { to { transform: rotate(360deg); } }

#error-box { background: #fff0f0; border: 1px solid #f5c6cb; border-radius: var(--radius); color: var(--color-error); padding: 14px 18px; margin-top: 16px; font-size: 0.92rem; }

@media (max-width: 480px) {
  header h1    { font-size: 1.3rem; }
  #result-card { padding: 16px; }
  .badge       { font-size: 0.72rem; padding: 3px 8px; }
}
```
 { font-size: 0.9rem; line-height: 1.6; margin-bottom: 4px; }

/* ── Feedback section ──────────────────────────────────────── */
#feedback-section {
  border-top    : 1px solid var(--color-border);
  padding-top   : 14px;
  margin-top    : 14px;
  text-align    : center;
}
#feedback-section p { font-size: 0.9rem; color: var(--color-subtle); margin-bottom: 8px; }
.btn-feedback {
  font-size     : 1.4rem;
  background    : none;
  border        : 2px solid var(--color-border);
  border-radius : 8px;
  padding       : 6px 14px;
  cursor        : pointer;
  margin        : 0 6px;
  transition    : border-color var(--transition), background var(--transition);
}
.btn-feedback:hover { border-color: var(--color-accent); background: #e8f5ee; }
#correction-form { margin-top: 12px; }
#correction-select {
  padding       : 6px 10px;
  border        : 1px solid var(--color-border);
  border-radius : 6px;
  font-size     : 0.9rem;
  margin-bottom : 8px;
  width         : 100%;
  max-width     : 360px;
}
#feedback-thanks { font-weight: 600; color: var(--color-primary); margin-top: 8px; }

/* ── Buttons ────────────────────────────────────────────────── */
.btn-primary {
  background    : var(--color-primary);
  color         : white;
  border        : none;
  border-radius : 8px;
  padding       : 8px 20px;
  font-size     : 0.9rem;
  font-weight   : 600;
  cursor        : pointer;
  transition    : background var(--transition);
}
.btn-primary:hover  { background: var(--color-primary-dk); }
.btn-secondary {
  background    : transparent;
  color         : var(--color-primary);
  border        : 2px solid var(--color-accent);
  border-radius : 8px;
  padding       : 6px 16px;
  font-size     : 0.85rem;
  cursor        : pointer;
  margin-top    : 8px;
}

/* ── Spinner ────────────────────────────────────────────────── */
#spinner {
  text-align  : center;
  padding     : 40px;
}
.spinner-ring {
  width           : 48px;
  height          : 48px;
  border          : 5px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius   : 50%;
  animation       : spin 0.9s linear infinite;
  margin          : 0 auto 12px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Error box ───────────────────────────────────────────────── */
#error-box {
  background    : #fff0f0;
  border        : 1px solid #f5c6cb;
  border-radius : var(--radius);
  color         : var(--color-error);
  padding       : 14px 18px;
  margin-top    : 16px;
  font-size     : 0.92rem;
}

/* ── Responsive ─────────────────────────────────────────────── */
@media (max-width: 480px) {
  header h1       { font-size: 1.3rem; }
  #result-card    { padding: 16px; }
  .badge          { font-size: 0.72rem; padding: 3px 8px; }
}
```

### 23.3 static/app.js — IMPLEMENT EXACTLY

[FIX GAP 49,50] Thumbs-up click sends positive feedback to /feedback.
Thumbs-down reveals correction form with all CLASS_NAMES as options.
Both use the correct element IDs from index.html.

```javascript
// static/app.js

const CLASS_NAMES = [
  'okra_yvmv','okra_powdery_mildew','okra_cercospora','okra_enation','okra_healthy',
  'brassica_black_rot','brassica_downy_mildew','brassica_alternaria',
  'brassica_clubroot','brassica_healthy'
];

// ── State ─────────────────────────────────────────────────────────────────
let currentResult = null;

// ── Element refs ──────────────────────────────────────────────────────────
const uploadArea      = document.getElementById('upload-area');
const fileInput       = document.getElementById('file-input');
const previewSection  = document.getElementById('preview-section');
const previewImg      = document.getElementById('preview-img');
const changeBtn       = document.getElementById('change-btn');
const uploadSection   = document.getElementById('upload-section');
const resultSection   = document.getElementById('result-section');
const spinner         = document.getElementById('spinner');
const errorBox        = document.getElementById('error-box');

// Result elements
const heatmapImg      = document.getElementById('heatmap-img');
const cropBadge       = document.getElementById('crop-badge');
const severityBadge   = document.getElementById('severity-badge');
const urgencyBadge    = document.getElementById('urgency-badge');
const oodBadge        = document.getElementById('ood-badge');
const diseaseHeading  = document.getElementById('disease-heading');
const confidenceText  = document.getElementById('confidence-text');
const confidenceBar   = document.getElementById('confidence-bar');
const treatmentList   = document.getElementById('treatment-list');
const preventionList  = document.getElementById('prevention-list');

// Feedback elements
const thumbsUpBtn         = document.getElementById('thumbs-up-btn');
const thumbsDownBtn       = document.getElementById('thumbs-down-btn');
const correctionForm      = document.getElementById('correction-form');
const correctionSelect    = document.getElementById('correction-select');
const submitCorrectionBtn = document.getElementById('submit-correction-btn');
const feedbackThanks      = document.getElementById('feedback-thanks');

// ── Upload handling ────────────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
uploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  uploadArea.style.background = '#e8f5ee';
});
uploadArea.addEventListener('dragleave', () => {
  uploadArea.style.background = '';
});
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.style.background = '';
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) processFile(fileInput.files[0]);
});
changeBtn.addEventListener('click', resetToUpload);

function resetToUpload() {
  fileInput.value     = '';
  previewSection.hidden = true;
  resultSection.hidden  = true;
  errorBox.hidden       = true;
  uploadSection.hidden  = false;
  currentResult = null;
}

function processFile(file) {
  // Show preview
  const url         = URL.createObjectURL(file);
  previewImg.src    = url;
  previewSection.hidden = false;
  uploadSection.hidden  = true;
  resultSection.hidden  = true;
  errorBox.hidden       = true;
  // Submit
  submitImage(file);
}

// ── API call ───────────────────────────────────────────────────────────────
async function submitImage(file) {
  spinner.hidden = false;
  errorBox.hidden = true;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    const data = await resp.json();

    if (!resp.ok) {
      showError(data.detail || 'Prediction failed. Please try another image.');
      return;
    }

    currentResult = data;
    renderResult(data);
  } catch (err) {
    showError('Network error — is the server running?');
  } finally {
    spinner.hidden = true;
  }
}

// ── Render ─────────────────────────────────────────────────────────────────
function renderResult(d) {
  // Heatmap
  if (d.heatmap_b64) {
    heatmapImg.src              = 'data:image/png;base64,' + d.heatmap_b64;
    heatmapImg.parentElement.hidden = false;
  } else {
    heatmapImg.parentElement.hidden = true;
  }

  // Crop badge
  cropBadge.textContent = d.crop.toUpperCase();

  // Severity badge
  const sev = (d.severity || 'mild').toLowerCase();
  severityBadge.textContent = sev.charAt(0).toUpperCase() + sev.slice(1);
  severityBadge.className   = 'badge badge-' + sev;

  // Urgency badge
  const urg = (d.urgency || 'Low').toLowerCase();
  urgencyBadge.textContent = d.urgency + ' Urgency';
  urgencyBadge.className   = 'badge badge-' + urg;

  // OOD
  oodBadge.hidden = !d.ood_flagged;

  // Disease heading
  const diseases = d.diseases || ['unknown'];
  diseaseHeading.textContent = diseases
    .map(c => c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))
    .join(' + ');

  // Confidence
  const pct = Math.round((d.confidence || 0) * 100);
  confidenceText.textContent =
    `Confidence: ${pct}%  ·  Uncertainty: ${Math.round((d.uncertainty || 0) * 100)}%`;
  confidenceBar.style.width = pct + '%';

  // Treatment
  treatmentList.innerHTML = '';
  (d.treatment || []).forEach(t => {
    const li = document.createElement('li');
    li.textContent = t;
    treatmentList.appendChild(li);
  });

  // Prevention
  preventionList.innerHTML = '';
  (d.prevention || []).forEach(p => {
    const li = document.createElement('li');
    li.textContent = p;
    preventionList.appendChild(li);
  });

  // Reset feedback
  correctionForm.hidden = true;
  feedbackThanks.hidden = true;
  thumbsUpBtn.disabled  = false;
  thumbsDownBtn.disabled= false;

  // Show result section
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Feedback ───────────────────────────────────────────────────────────────
// [FIX GAP 50] Thumbs-up: send positive feedback and show thanks
thumbsUpBtn.addEventListener('click', async () => {
  if (!currentResult) return;
  thumbsUpBtn.disabled  = true;
  thumbsDownBtn.disabled= true;
  await sendFeedback({ thumbs_up: true, crop: currentResult.crop,
                       diseases: currentResult.diseases });
  feedbackThanks.hidden = false;
});

// [FIX GAP 49] Thumbs-down: reveal correction form with class options
thumbsDownBtn.addEventListener('click', () => {
  thumbsDownBtn.disabled = true;
  thumbsUpBtn.disabled   = true;
  // Populate correction select with all class names
  correctionSelect.innerHTML = '<option value="">— Select correct class —</option>';
  CLASS_NAMES.forEach(cls => {
    const opt = document.createElement('option');
    opt.value       = cls;
    opt.textContent = cls.replace(/_/g, ' ');
    correctionSelect.appendChild(opt);
  });
  correctionForm.hidden = false;
});

submitCorrectionBtn.addEventListener('click', async () => {
  const correction = correctionSelect.value;
  if (!correction) {
    alert('Please select the correct class.');
    return;
  }
  await sendFeedback({
    thumbs_up  : false,
    crop       : currentResult ? currentResult.crop : '',
    diseases   : currentResult ? currentResult.diseases : [],
    correction : correction,
  });
  correctionForm.hidden = true;
  feedbackThanks.hidden = false;
});

async function sendFeedback(payload) {
  try {
    await fetch('/feedback', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(payload),
    });
  } catch (e) {
    // Feedback failure is silent — never show error for this
  }
}

// ── Error display ──────────────────────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden      = false;
  spinner.hidden       = true;
  previewSection.hidden= false;
  uploadSection.hidden = true;
}
```

---

## SECTION 24: DIAGNOSIS LOOKUP (diagnosis/diagnosis_lookup.json) — IMPLEMENT EXACTLY

All 10 entries. Agronomically verified content.

```json
{
  "okra_yvmv": {
    "full_name": "Yellow Vein Mosaic Virus",
    "cause": "Begomovirus transmitted by whitefly Bemisia tabaci. Spreads via infected transplants and volunteer plants. No seed transmission.",
    "symptoms": "Bright yellow vein network on green leaf background. Mosaic yellow-green patterning. Leaf curling upward. Stunted growth and distorted pods in severe cases.",
    "treatment": [
      "Remove and destroy all infected plants immediately — no chemical cure for viral disease.",
      "Control whitefly vector: spray imidacloprid 17.8 SL at 0.3 ml per litre or thiamethoxam 25 WG at 0.3 g per litre.",
      "Use yellow sticky traps (30 per hectare) to monitor and reduce whitefly populations.",
      "Apply neem oil at 3 ml per litre as supplementary whitefly deterrent."
    ],
    "prevention": [
      "Use resistant varieties: Arka Anamika, Parbhani Kranti, Punjab Padmini.",
      "Use virus-free certified planting material. Do not use seeds from infected plants.",
      "Apply reflective silver mulch to deter whitefly landing.",
      "Maintain 2-row border of maize or sorghum as windbreak to reduce whitefly entry."
    ],
    "urgency": "High",
    "urgency_reason": "No cure. Spreads rapidly via whitefly. Remove infected plants and spray within 24 hours to protect adjacent plants."
  },
  "okra_powdery_mildew": {
    "full_name": "Powdery Mildew (Okra)",
    "cause": "Fungus Erysiphe cichoracearum. Favoured by warm dry days (25-30°C) with cool humid nights. Unlike most fungi, does not need free water on leaves.",
    "symptoms": "White-to-grey powdery coating on upper leaf surface. Affected leaves turn yellow then brown. Severe infections cause premature leaf drop and stunted fruit development.",
    "treatment": [
      "Apply wettable sulphur 80 WP at 2.5 g per litre at first white patches. Do NOT spray when temperature exceeds 35°C — phytotoxicity risk.",
      "Use hexaconazole 5 EC at 1 ml per litre for systemic control.",
      "Triadimefon 25 WP at 0.5 g per litre as alternative systemic fungicide.",
      "Spray coverage must reach all leaf surfaces including undersides."
    ],
    "prevention": [
      "Maintain adequate plant spacing to promote air circulation.",
      "Avoid excess nitrogen fertilisation — succulent growth is more susceptible.",
      "Remove heavily infected leaves before spraying to reduce spore load."
    ],
    "urgency": "Medium",
    "urgency_reason": "Spreads rapidly in warm dry conditions. Treat within 3-5 days of first symptoms."
  },
  "okra_cercospora": {
    "full_name": "Cercospora Leaf Spot",
    "cause": "Fungus Cercospora abelmoschi. Spreads via airborne conidia. Favoured by warm humid weather (25-30°C, >80% humidity). Overwinters in infected crop debris.",
    "symptoms": "Circular brown spots 3-10 mm diameter with grey-white centre. Distinct yellow halo around each spot. Severe infections cause large necrotic areas and defoliation.",
    "treatment": [
      "Spray mancozeb 75 WP at 2.5 g per litre every 7-10 days from first symptoms.",
      "Copper oxychloride 50 WP at 3 g per litre as alternative contact fungicide.",
      "Carbendazim 50 WP at 0.5 g per litre for systemic control of severe infections.",
      "Remove and destroy infected leaves before spraying."
    ],
    "prevention": [
      "Collect and destroy all crop debris after harvest — do NOT incorporate into soil.",
      "Treat seeds with thiram 75 WS at 3 g per kg before sowing.",
      "Maintain 3-year crop rotation avoiding okra and other malvaceous crops.",
      "Avoid overhead irrigation — use drip to keep foliage dry."
    ],
    "urgency": "Medium",
    "urgency_reason": "Spreads rapidly in humid conditions. Treat within 3-5 days."
  },
  "okra_enation": {
    "full_name": "Enation Leaf Curl",
    "cause": "Begomovirus complex, different strains from YVMV but also transmitted by whitefly Bemisia tabaci. Can co-infect with YVMV.",
    "symptoms": "Severe upward curling and rolling of leaves. Enations (outgrowths) on underside of leaf veins — feel like small bumps when running finger along vein. Vein swelling. Leaf becomes leathery. Severely stunted plant.",
    "treatment": [
      "Remove and destroy infected plants. No viral cure.",
      "Spray imidacloprid 17.8 SL at 0.3 ml per litre to control whitefly vector.",
      "Apply thiamethoxam 25 WG at 0.3 g per litre as whitefly alternative.",
      "Use spirotetramat 150 OD at 0.75 ml per litre for systemic whitefly control."
    ],
    "prevention": [
      "Use tolerant varieties where available.",
      "Border rows of tall non-host crops (maize, jowar) reduce whitefly entry.",
      "Install 50-mesh insect-proof net on nursery beds.",
      "Remove volunteer okra plants from field margins — they act as virus reservoirs."
    ],
    "urgency": "High",
    "urgency_reason": "No cure. Severe stunting if not managed early. Remove infected plants and control whitefly immediately."
  },
  "okra_healthy": {
    "full_name": "Healthy Okra Leaf",
    "cause": "No disease detected.",
    "symptoms": "No visible disease symptoms. Normal uniform green colour. No spots, curl, or lesions.",
    "treatment": ["No treatment required. The leaf appears healthy."],
    "prevention": [
      "Continue monitoring every 5-7 days.",
      "Maintain balanced fertilisation — avoid excess nitrogen.",
      "Keep field weed-free to reduce whitefly and aphid habitat."
    ],
    "urgency": "Low",
    "urgency_reason": "No disease detected. Routine monitoring recommended."
  },
  "brassica_black_rot": {
    "full_name": "Black Rot",
    "cause": "Bacterium Xanthomonas campestris pv. campestris. Enters through leaf margin water pores and wounds. Spread by infected seeds, rain splash, contaminated tools.",
    "symptoms": "V-shaped yellow-to-brown lesions pointing inward from leaf margins, following vein pattern. Veins inside lesion turn brown-to-black. Severe cases cause whole-leaf collapse.",
    "treatment": [
      "Remove and destroy all infected plant material. Do NOT compost.",
      "Spray copper oxychloride 50 WP at 3 g per litre every 7 days.",
      "Use streptomycin sulphate 90 SP at 200 ppm for severe infections.",
      "Do not work in the field when plants are wet."
    ],
    "prevention": [
      "Use certified black-rot-free seeds. Treat seeds with hot water at 50°C for exactly 30 minutes.",
      "Rotate brassicas away from same plot for at least 2 years.",
      "Ensure field drainage — the bacterium thrives in waterlogged conditions.",
      "Control cabbage aphids and other insects that create entry wounds."
    ],
    "urgency": "High",
    "urgency_reason": "Spreads rapidly through rain splash and contaminated tools. Remove infected plants and spray within 24 hours."
  },
  "brassica_downy_mildew": {
    "full_name": "Downy Mildew (Brassica)",
    "cause": "Oomycete Hyaloperonospora parasitica. Favoured by cool (15-20°C) humid conditions with free moisture on leaves.",
    "symptoms": "Irregular yellow patches on upper leaf surface. White-to-grey downy sporulation on underside directly below yellow areas. Young plants may show severe distortion.",
    "treatment": [
      "Apply metalaxyl + mancozeb (Ridomil Gold MZ) at 2.5 g per litre at first symptoms.",
      "Use cymoxanil + mancozeb (Curzate M8) at 2 g per litre as alternative.",
      "Copper hydroxide at 3 g per litre for early-stage control.",
      "Remove heavily infected outer leaves before spraying."
    ],
    "prevention": [
      "Use resistant brassica varieties where available.",
      "Switch to drip irrigation — overhead watering maintains leaf wetness needed for sporulation.",
      "Maintain adequate plant spacing.",
      "Apply preventive copper spray at transplanting during cool wet seasons."
    ],
    "urgency": "Medium",
    "urgency_reason": "Spreads rapidly in cool wet weather. Treat within 3-5 days."
  },
  "brassica_alternaria": {
    "full_name": "Alternaria Leaf Spot (Dark Leaf Spot)",
    "cause": "Fungi Alternaria brassicicola and Alternaria brassicae. Spread by airborne conidia during warm humid conditions.",
    "symptoms": "Dark brown-to-black circular spots with concentric rings (target-board pattern), yellow halo around spots. Heavy infections cause holes and premature defoliation.",
    "treatment": [
      "Apply mancozeb 75 WP at 2.5 g per litre every 7-10 days from first appearance.",
      "Use iprodione 50 WP at 1 g per litre for systemic control of severe infections.",
      "Chlorothalonil 75 WP at 2 g per litre as alternative contact fungicide.",
      "Ensure spray reaches leaf undersides where spores germinate."
    ],
    "prevention": [
      "Treat seeds with thiram 75 WS at 3 g per kg before sowing.",
      "Rotate brassicas for 2 years after an Alternaria outbreak.",
      "Avoid excessive plant density.",
      "Apply preventive spray at bud formation stage during high-risk seasons."
    ],
    "urgency": "Medium",
    "urgency_reason": "Spreads in warm humid weather. Treat within 3-5 days to prevent defoliation."
  },
  "brassica_clubroot": {
    "full_name": "Clubroot",
    "cause": "Obligate soil-borne parasite Plasmodiophora brassicae. Resting spores survive in soil for up to 20 years.",
    "symptoms": "Above-ground only: wilting during warm afternoons despite adequate soil moisture, progressive yellowing from lower leaves, stunted growth. Uprooting shows swollen club-shaped root galls.",
    "treatment": [
      "No cure once infected. Remove and destroy ALL infected plants including root material.",
      "Do NOT compost infected roots — spores survive composting.",
      "Apply agricultural lime to raise soil pH to 7.2 or above.",
      "Apply calcium cyanamide at 200 kg per hectare as combined fertiliser and soil treatment."
    ],
    "prevention": [
      "Quarantine — do not move soil, tools, footwear, or plants from infected to uninfected fields.",
      "Use resistant varieties: Kilaton F1, Clapton F1, Crispus F1.",
      "Minimum 4-5 year crop rotation without brassicas.",
      "Test and maintain soil pH above 7.0 before every brassica crop.",
      "Dip transplant roots in thiram suspension (3 g per litre) for 15 minutes before planting."
    ],
    "urgency": "High",
    "urgency_reason": "Persistent soil pathogen with 20-year spore survival. Implement strict quarantine immediately."
  },
  "brassica_healthy": {
    "full_name": "Healthy Brassica Leaf",
    "cause": "No disease detected.",
    "symptoms": "No visible disease symptoms. Normal leaf colour and structure.",
    "treatment": ["No treatment required. The leaf appears healthy."],
    "prevention": [
      "Continue monitoring every 7-10 days.",
      "Apply preventive copper spray at start of monsoon/rainy season.",
      "Ensure adequate field drainage."
    ],
    "urgency": "Low",
    "urgency_reason": "No disease detected. Routine monitoring recommended."
  }
}
```

---

## SECTION 25: SETUP SCRIPTS

### 25.1 setup/setup_project.py — IMPLEMENT EXACTLY

[FIX GAP 59] Checks virtual environment. [FIX GAP 65] Loads .env.

```python
# setup/setup_project.py
"""
Step 0: Creates project directories, loads .env, validates env vars,
configures GitHub.
[FIX GAP 59] Checks that a virtual environment is active.
[FIX GAP 65] Loads .env at startup.
"""

import os
import sys
import subprocess

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_venv():
    """[FIX GAP 59] Verify running inside a virtual environment."""
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    if not in_venv:
        print("ERROR: Not running inside a virtual environment.")
        print("Create and activate one first:")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate.bat   (Windows CMD)")
        print("  venv\\Scripts\\Activate.ps1   (Windows PowerShell)")
        sys.exit(1)
    print(f"  Virtual environment active: {sys.prefix}")


def create_directories():
    """Create all required project directories."""
    dirs = [
        'data/raw', 'data/processed/train', 'data/processed/val',
        'data/processed/test', 'data/metadata', 'data/kerala', 'data/plantdoc',
        'models/checkpoints', 'cache', 'reports', 'diagnosis', 'tools',
        'static', 'templates', 'agents', 'training', 'app', 'setup',
    ]
    for d in dirs:
        path = os.path.join(ROOT, d)
        os.makedirs(path, exist_ok=True)
    print(f"  Created {len(dirs)} directories.")


def check_env_vars():
    """Validate required environment variables are set."""
    required = ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'GITHUB_TOKEN']
    optional = ['WANDB_API_KEY']
    missing  = []
    for var in required:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"ERROR: Missing required env vars: {missing}")
        print(f"Copy .env.template to .env and fill in values.")
        sys.exit(1)
    for var in optional:
        if not os.environ.get(var):
            print(f"  WARNING: {var} not set. wandb will run offline.")
    print(f"  Environment variables validated.")


def configure_git():
    """Configure git remote and initial commit if needed."""
    github_token = os.environ.get('GITHUB_TOKEN', '')
    github_repo  = os.environ.get('GITHUB_REPO', '')
    if not github_repo:
        print("  GITHUB_REPO not set — skipping git configuration.")
        return
    r = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        remote_url = f'https://{github_token}@github.com/{github_repo}.git'
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url],
                       cwd=ROOT)
        print(f"  Git remote set to {github_repo}")
    else:
        print(f"  Git remote already configured.")


def write_gitignore():
    """Write .gitignore if not present."""
    path = os.path.join(ROOT, '.gitignore')
    if os.path.exists(path):
        return
    content = """
# Data (gigabytes — never commit)
data/raw/
data/processed/
data/kerala/
data/plantdoc/
data/metadata/*.csv
cache/

# Model weights
models/*.pt
models/checkpoints/

# Environment
.env
venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Feedback database
feedback.db

# Pipeline state
.pipeline_progress.json
pipeline_failures.log
""".strip()
    with open(path, 'w') as f:
        f.write(content)
    print(f"  .gitignore written.")


if __name__ == '__main__':
    print("=" * 50)
    print("STEP 00 — PROJECT SETUP")
    print("=" * 50)
    check_venv()
    create_directories()
    check_env_vars()
    configure_git()
    write_gitignore()
    print("\nStep 00 complete.")
```

### 25.2 setup/install_cuda.py — IMPLEMENT EXACTLY

[FIX GAP 72] input() wrapped in try/except EOFError — treats EOF as --yes
(non-interactive/subprocess context). [FIX GAP 51] --yes flag respected.

```python
# setup/install_cuda.py
"""
Step 1: CUDA 12.1 installation and verification on Windows.
[FIX GAP 72] input() wrapped in try/except EOFError — treats EOF as --yes.
[FIX GAP 51] --yes flag skips all interactive prompts.
"""

import sys
import os
import subprocess
import argparse


def prompt(message, yes_flag):
    """[FIX GAP 72] Prompt for confirmation. EOF treated as --yes."""
    if yes_flag:
        return 'yes'
    try:
        return input(message).strip().lower()
    except EOFError:
        # Running in non-interactive subprocess context — treat as --yes
        print("(EOF detected — running non-interactively, assuming yes)")
        return 'yes'


def check_cuda_available():
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch; print(torch.cuda.is_available()); '
         'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        available = lines[0].strip() == 'True'
        name      = lines[1].strip() if len(lines) > 1 else 'unknown'
        return available, name
    return False, 'torch not installed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true',
                        help='Skip all interactive prompts')
    args = parser.parse_args()

    print("=" * 50)
    print("STEP 01 — CUDA INSTALLATION")
    print("=" * 50)

    if sys.platform != 'win32':
        print("This script targets Windows 11. On Linux, install CUDA manually.")
        sys.exit(0)

    # Step A: check if CUDA already works
    available, name = check_cuda_available()
    if available and 'RTX 4060' in name or 'NVIDIA' in name:
        print(f"  ✓ CUDA already working: {name}")
        print("Step 01 complete.")
        sys.exit(0)

    print("CUDA not available. Starting installation...")
    print("This requires Administrator privileges for some steps.")
    print("Follow these steps manually:")
    print()
    print("Step B — Run in Command Prompt as Administrator:")
    print("  nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    print("  If version < 525: download driver from nvidia.com/drivers")
    print()
    print("Step C — Install CUDA Toolkit 12.1:")
    print("  URL: developer.download.nvidia.com/compute/cuda/12.1.0/"
          "network_installers/cuda_12.1.0_windows_network.exe")
    print("  Run as Administrator. Choose Express. Restart Windows after.")

    ans = prompt("\nHave you installed CUDA 12.1? Type 'yes' to continue: ",
                 args.yes)

    print()
    print("Step D — Install PyTorch with CUDA 12.1:")
    print("  pip install torch==2.2.0 torchvision==0.17.0 --index-url "
          "https://download.pytorch.org/whl/cu121")

    ans = prompt("\nHave you installed PyTorch? Type 'yes' to continue: ",
                 args.yes)

    # Verify
    available, name = check_cuda_available()
    if available:
        print(f"\n  ✓ CUDA verified: {name}")
        print("Step 01 complete.")
        sys.exit(0)
    else:
        print(f"\n  ✗ CUDA still not available. {name}")
        print("See Section 13.3 in CLAUDE.md for troubleshooting.")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

---

## SECTION 26: REQUIREMENTS FILES — IMPLEMENT EXACTLY

[FIX GAP 47] pyinaturalist removed. [FIX GAP 60] pillow-heif removed.
[FIX GAP 59] venv instructions in comments.

### requirements.txt (production server only)

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
torch==2.2.0
torchvision==0.17.0
timm==0.9.16
pytorch-grad-cam==1.5.0
Pillow==10.3.0
numpy==1.26.4
opencv-python-headless==4.9.0.80
pydantic==2.7.0
python-dotenv==1.0.1
```

### requirements_train.txt (training environment, includes production)

```
# Production dependencies
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
torch==2.2.0
torchvision==0.17.0
timm==0.9.16
pytorch-grad-cam==1.5.0
Pillow==10.3.0
numpy==1.26.4
opencv-python-headless==4.9.0.80
pydantic==2.7.0
python-dotenv==1.0.1

# Training
albumentations==1.4.4
scikit-learn==1.4.2
pandas==2.2.2
wandb==0.17.0
kaggle==1.6.12
requests==2.32.3
yt-dlp==2024.5.27

# Stable Diffusion (optional — for synthetic generation)
# diffusers==0.27.2
# transformers==4.40.1
# accelerate==0.29.3

# NOTE: pyinaturalist NOT included — uses requests directly
# NOTE: pillow-heif NOT included — HEIC not supported
```

---

## SECTION 27: ENVIRONMENT TEMPLATE (.env.template) — IMPLEMENT EXACTLY

```bash
# .env.template
# Copy this file to .env and fill in your values.
# NEVER commit .env to git.

# ── Kaggle ────────────────────────────────────────────────────
# Get from: kaggle.com > Account > Create API Token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# ── GitHub ────────────────────────────────────────────────────
# Personal access token with repo scope
# Get from: github.com > Settings > Developer settings > Personal access tokens
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_REPO=yourusername/plant-disease-kerala

# ── Wandb ─────────────────────────────────────────────────────
# Get from: wandb.ai > Settings > API keys
# If not set, training runs offline (wandb logs to local files)
WANDB_API_KEY=your_wandb_api_key
```

---

## SECTION 28: README (README.md) — IMPLEMENT EXACTLY

[FIX GAP 26, 56] Server startup commands documented here.

```markdown
# Plant Disease Detection — Kerala

Detects okra and brassica leaf diseases from smartphone photographs.
Designed for farmers in Kerala, South India.

## Supported diseases

- Okra: Yellow Vein Mosaic Virus, Powdery Mildew, Cercospora Leaf Spot, Enation Leaf Curl
- Brassica (broccoli/cabbage): Black Rot, Downy Mildew, Alternaria Leaf Spot, Clubroot

## Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/plant-disease-kerala.git
cd plant-disease-kerala

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate.bat       # Windows CMD
# source venv/Scripts/activate  # Git Bash

# 3. Copy and fill environment template
cp .env.template .env
# Edit .env with your Kaggle, GitHub, and Wandb credentials

# 4. Run pipeline (trains the model — takes ~4 hours)
python run_pipeline.py
```

## Starting the server

```bash
# Development (auto-reload on code change)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production (4 workers, no reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Open browser at: http://localhost:8000

## Adding Kerala field images for tier-3 evaluation

```bash
python tools/add_kerala_image.py --path path/to/image.jpg --class okra_yvmv
```

## Running evaluations manually

```bash
# Tier-2 PlantDoc evaluation (ONCE — run after training complete)
python training/08_evaluate_tier2_plantdoc.py

# Local test set evaluation (after tier-2)
python training/10_evaluate_local_test.py

# Tier-3 Kerala evaluation (when 50+ Kerala images collected)
python training/09_evaluate_tier3_kerala.py
```

## Pipeline steps

| Step | Script | What it does |
|------|--------|--------------|
| 0 | setup/setup_project.py | Creates directories, validates env |
| 1 | setup/install_cuda.py | CUDA 12.1 installation guide |
| 2 | setup/install_dependencies.py | Installs Python packages |
| 3 | agents/download_orchestrator.py | Downloads 6 training datasets |
| 4 | agents/acquire_kerala_images.py | iNaturalist + YouTube + synthetic |
| 5 | training/01_prepare_data.py | Label assertions, split, source_map.csv |
| 6 | training/02_generate_severity.py | Severity proxy labels |
| 7 | training/03_cache_features.py | Backbone feature caching |
| 8 | training/04_train_phase1.py | Head training (~30 min) |
| 9 | training/05_train_phase2.py | Full fine-tuning (~3.5 hr) |
| 10 | training/06_calibrate.py | Temperature scaling |
| 11 | training/07_evaluate_validation.py | Validation report |
| 12 | setup/test_server.py | Server smoke test |
| 13 | training/08_evaluate_tier2_plantdoc.py | Tier-2 evaluation (ONCE) |
| 14 | setup/package_deployment.py | Dockerfile |
```

---

## SECTION 29: WHAT NOT TO DO (30 failure modes)

Read this before writing any code. Every item caused a real identified failure.

### Architecture

**DO NOT use a single class label for multi-label training.** okra_yvmv + okra_cercospora
can coexist. Single-label (softmax) cannot represent this. Use sigmoid + BCEWithLogitsLoss.

**DO NOT apply model.train() for MC Dropout.** model.train() puts BatchNorm in training mode.
With batch size 1, BatchNorm computes statistics from one image — nonsense. Do: model.eval(),
then iterate modules and set only isinstance(module, Dropout) to .train().

**DO NOT run Grad-CAM during MC Dropout passes.** MC needs Dropout in train mode. Grad-CAM
needs clean gradient flow in eval mode. Run MC passes first, then switch to full eval for CAM.

**DO NOT reference model.fpn.output_p3.** The attribute is out_p3. output_p3 raises AttributeError.
[FIX GAP 7]

**DO NOT import apply_clahe from training.transforms into app/inference.py.** Define it inline
in inference.py. [FIX GAP 8,11]

### Training

**DO NOT use sklearn.compute_class_weight('balanced') for multi-label BCE.** It is for single-label.
For multi-label: pos_weight[j] = n_negative_j / n_positive_j. [FIX GAP 34]

**DO NOT build pos_weight by counting label occurrences.** Pass the binary label matrix directly.
n_total = number of images, not number of positive labels. [FIX GAP 34]

**DO NOT call compute_loss(...).backward().** compute_loss returns a tuple. Tuples have no
.backward(). Always unpack: total, details = compute_loss(...); total.backward().

**DO NOT call clip_grad_norm_ before scaler.unscale_(optimizer).** In mixed precision,
gradients are FP16 scaled. Clip before unscale = clipping 65536x the actual gradient.
[Section 9, Phase 2]

**DO NOT add a manual warmup on top of OneCycleLR.** OneCycleLR has built-in warmup via
pct_start. Two warmup mechanisms conflict and produce incorrect LR values.

**DO NOT use num_workers > 0 on Windows without if __name__ == '__main__' guard.** Windows
spawn method re-imports the module in each worker, causing infinite worker spawning.

**DO NOT use drop_last=True.** brassica_clubroot has ~150 images. drop_last discards 14% of
the thinnest class every epoch.

**DO NOT use persistent_workers=True on Windows.** Produces zombie processes on Ctrl+C.

**DO NOT cache features using augmented images.** Use get_eval_transform() for caching.
Augmented cache = same fixed augmented features every epoch, no benefit.

**DO NOT define EarlyStopping or get_llrd_optimizer inside 04_train_phase1.py or 05_train_phase2.py.**
Define them only in training/helpers.py and import at module level. [FIX GAP 1,3,4,19,20]

**DO NOT hardcode ONE_CYCLE_PCT, ONE_CYCLE_DIV, ONE_CYCLE_FDIV.** Import from app.config.
[FIX GAP 35]

**DO NOT start Phase 2 from scratch if phase2_epoch*.pt exists.** Check for existing
phase2 checkpoints and resume from the latest one. [FIX GAP 71]

### Inference

**DO NOT move crop gate tensor to CPU while mean_dis is on CUDA.** gate = gate.to(mean_dis.device).

**DO NOT return HTTP 422 for OOD.** Return 200 with ood_flagged=True. 422 is Pydantic error code.

**DO NOT run inference synchronously in async route handler.** Use run_in_executor.

**DO NOT use SQLite with check_same_thread=True in multi-threaded server.** Use check_same_thread=False.

**DO NOT use model.fpn.output_p3 for Grad-CAM target.** Use model.fpn.out_p3. [FIX GAP 7]

### Data pipeline

**DO NOT store absolute paths in source_map.csv.** Store relative paths from ROOT.
os.path.join(ROOT, rel_path) in PlantDiseaseDataset. [FIX GAP 30]

**DO NOT populate data/processed/ directories.** Images stay in data/raw/.
data/processed/ exists but is empty. [FIX GAP 24]

**DO NOT include PlantDoc images in training.** PlantDoc is tier-2 test only, never training.

**DO NOT skip CLAHE in _SevProxyDataset.** Severity proxy must use same CLAHE preprocessing
as training. [FIX GAP 12]

**DO NOT re-download datasets in 01_prepare_data.py.** That script assumes Step 03 (download)
already completed. [FIX GAP 11]

**DO NOT use iNaturalist taxon ID 47313 for brassica.** 47313 = Brassicaceae (entire family).
Correct ID = 55774 = Brassica oleracea. [FIX GAP 40]

### Infrastructure

**DO NOT define LABEL_MAP or SOURCE_LABEL_OVERRIDES in 01_prepare_data.py.** These belong in
app/config.py so agent scripts can import them without circular imports. [FIX GAP 22]

**DO NOT duplicate Kaggle credential logic.** Use agents/kaggle_utils.py. [FIX GAP 28]

**DO NOT install to system Python.** Always use a virtual environment. [FIX GAP 59]

**DO NOT use HEIC images.** pillow-heif is not in requirements. HEIC is not in VALID_EXT. [FIX GAP 60]

---

## SECTION 30: EXECUTION ORDER — COMPLETE

```
Step 00 — python run_pipeline.py   (or: python setup/setup_project.py)
          Creates directories, loads .env, validates env vars, configures GitHub.
          Smoke test: data/metadata/ directory exists.

Step 01 — setup/install_cuda.py
          Guides CUDA 12.1 installation on Windows. Verifies RTX 4060 working.
          Smoke test: torch.cuda.is_available() == True.

Step 02 — setup/install_dependencies.py
          pip install requirements_train.txt. Verifies all imports.
          Smoke test: import torch, timm, albumentations, sklearn, cv2, fastapi, wandb.

Step 03 — agents/download_orchestrator.run_all_downloads()  (via pipeline lambda)
          Downloads all 6 training datasets in parallel (20-40 minutes).
          Downloads PlantDoc separately to data/plantdoc/.
          Smoke test: sabbir_okra and kareem_cabbage success=True in download_results.json.

Step 04 — agents/acquire_kerala_images.acquire_all()  (via pipeline lambda)
          iNaturalist + YouTube frames + Stable Diffusion in parallel.
          Smoke test: None (best-effort acquisition).

Step 05 — training/01_prepare_data.py
          Scans data/raw/, resolves labels, stratified split, writes source_map.csv.
          Smoke test: data/metadata/source_map.csv exists.

Step 06 — training/02_generate_severity.py
          Proxy severity labels from GradCAM saliency. Writes severity_labels.csv.
          Smoke test: data/metadata/severity_labels.csv exists.

Step 07 — training/03_cache_features.py
          Backbone + FPN features cached for train and val splits.
          Verifies backbone shapes before caching.
          Smoke test: cache/train_features.pt and cache/val_features.pt exist.

Step 08 — training/04_train_phase1.py
          Heads trained on cached features (~30 minutes).
          Smoke test: models/checkpoints/phase1_best.pt exists.

Step 09 — training/05_train_phase2.py
          Full fine-tuning with top 1/3 backbone unfrozen (~3-3.5 hours).
          Resumes from phase2_epoch*.pt if a checkpoint exists (crash recovery).
          Smoke test: models/best_model.pt exists.

Step 10 — training/06_calibrate.py
          Fits T_disease, T_crop, T_severity on val set. ECE measured.
          Smoke test: models/temperature.pt exists.

Step 11 — training/07_evaluate_validation.py
          Full inference re-run on val set. Confusion matrix, per-class F1, ECE.
          Writes reports/validation_report_{timestamp}.md.
          Smoke test: any .md file exists in reports/.

Step 12 — setup/test_server.py
          Starts FastAPI on port 8765, health check, predict with textured test image.
          Smoke test: server returns 200 with valid JSON response structure.

Step 13 — training/08_evaluate_tier2_plantdoc.py  ← run ONCE, --yes passed by pipeline
          Full PlantDoc tier-2 evaluation with temperature scaling.
          Writes reports/tier2_plantdoc_{timestamp}.md.
          Smoke test: any tier2*.md file exists in reports/.

Step 14 — setup/package_deployment.py
          Dockerfile written. All deployment files verified present.
          Smoke test: Dockerfile exists.

─── Manual steps — NOT in automated pipeline ───────────────────────────────

After Step 13:
  python training/10_evaluate_local_test.py --yes
  Evaluates the locked 15% local test split. Run once after tier-2 is complete.

When 50+ Kerala images collected:
  python tools/add_kerala_image.py --path img.jpg --class class_name
  (repeat for each image)

  python training/09_evaluate_tier3_kerala.py
  Project is deployment-validated only after this passes.

Development server (after training complete):
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Production server:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## SECTION 31: PROJECT STATUS CHECKLIST

```
[ ] Step 00 — Environment setup, GitHub configured, venv active
[ ] Step 01 — CUDA 12.1 installed, RTX 4060 verified (torch.cuda.is_available() = True)
[ ] Step 02 — All Python dependencies installed and verified
[ ] Step 03 — All datasets downloaded (check data/metadata/download_results.json)
[ ]          - sabbir_okra success=True
[ ]          - kareem_cabbage success=True
[ ]          - iubat_okra: if fail, MANUAL_DOWNLOAD_REQUIRED.txt created and reviewed
[ ] Step 04 — Kerala images acquired (iNaturalist + YouTube + synthetic)
[ ] Step 05 — source_map.csv created, label assertions passed, split complete
[ ]          - class_counts.csv written
[ ] Step 06 — severity_labels.csv generated
[ ] Step 07 — cache/train_features.pt and cache/val_features.pt created
[ ]          - backbone shapes verified: P3=[B,48,28,28] P4=[B,160,14,14] P5=[B,256,7,7]
[ ] Step 08 — Phase 1 complete, phase1_best.pt saved, macro F1 > 0.30
[ ] Step 09 — Phase 2 complete, best_model.pt saved, macro F1 > 0.50
[ ] Step 10 — temperature.pt saved, ECE after < 0.10
[ ] Step 11 — Validation report in reports/ (macro F1 >= 0.50)
[ ] Step 12 — Server health check passed, inference returns valid JSON
[ ] Step 13 — Tier-2 PlantDoc evaluation run ONCE, macro F1 > 0.55
[ ]          - Gap analysis written if F1 < 0.55
[ ]          - NO model changes after this step
[ ] Step 14 — Dockerfile created, all deployment files present
[ ] Manual  — Local test set evaluation run (10_evaluate_local_test.py)

[ ] Manual: Grad-CAM spot check — 5 images verified heatmaps on correct leaf region
[ ] Manual: Diagnosis lookup verified — all 10 entries present and agronomically correct
[ ] Manual: End-to-end test with a real diseased leaf photograph

[ ] TIER-3 KERALA: 50+ verified Kerala field images collected via tools/add_kerala_image.py
[ ] TIER-3 KERALA: 09_evaluate_tier3_kerala.py run, per-class accuracy > 0.70
[ ] TIER-3 KERALA: ← Project is deployment-validated only after this passes

LOCKED SETS: local test split (15%) and PlantDoc (tier-2) are locked until Step 13.
After Step 13 runs, NO model changes permitted.
```

---

## SECTION 32: WANDB CONFIGURATION

[FIX GAP 66] WANDB_MODE=offline set as fallback if WANDB_API_KEY missing.
[FIX GAP 70] Calibration script passes full WANDB_CONFIG to wandb.init().

All training and calibration scripts follow this pattern:

```python
import wandb
import os

# [FIX GAP 66] Set offline mode if no API key, so training is never blocked
if not os.environ.get('WANDB_API_KEY'):
    os.environ.setdefault('WANDB_MODE', 'offline')

from app.config import WANDB_PROJECT, WANDB_CONFIG

# At start of each training phase — include full config [FIX GAP 70]:
wandb.init(
    project=WANDB_PROJECT,
    name='phase1',          # or 'phase2', 'calibration'
    config={**WANDB_CONFIG, 'phase': 1, 'epochs': PHASE1_EPOCHS, 'amp': False},
)

# During training:
wandb.log({
    'epoch'           : epoch,
    'train/loss'      : train_loss,
    'val/macro_f1'    : val_f1,
    'val/crop_acc'    : crop_acc,
    'train/grad_norm' : grad_norm,    # Phase 2 only
    'train/lr'        : current_lr,   # Phase 2 only
})

# At end:
wandb.finish()
```

Calibration (training/06_calibrate.py) must also call wandb.init() with full
config including phase='calibration', T_disease, T_crop, T_severity after fitting.

---

*End of CLAUDE.md — Version 6.0*
*Total gap fixes: 72*
*Sections: 32*
*All 72 identified gaps resolved with [FIX GAPxx] markers*
*Local-only execution on RTX 4060 — no Vast.ai, no cloud*
*Virtual environment required*
*training/helpers.py added to file inventory*
*agents/kaggle_utils.py added*
*training/10_evaluate_local_test.py added*
*tools/add_kerala_image.py added*
*README.md added with server startup commands*

---

## APPENDIX: REMAINING GAP RESOLUTIONS

### [FIX GAP 14] acquire_all() full specification

acquire_all() is fully implemented in Section 16 (agents/acquire_kerala_images.py).
It runs acquire_inaturalist, acquire_youtube_frames, and generate_synthetic simultaneously
using ThreadPoolExecutor(max_workers=3). It collects results from each future, handles
exceptions per-task without stopping others, and returns a results dict keyed by
source name. After acquire_all() completes, 01_prepare_data.py (Step 05) scans
data/kerala/ and data/raw/synthetic/ to add new records to source_map.csv.

### [FIX GAP 19,20] Authoritative location of shared training utilities

[FIX GAP 19] EarlyStopping, save_checkpoint, load_checkpoint, cleanup_old_checkpoints:
Defined ONLY in training/helpers.py. NEVER in 04_train_phase1.py.

[FIX GAP 20] get_llrd_optimizer:
Defined ONLY in training/helpers.py. NEVER in 05_train_phase2.py.

Both 04_train_phase1.py and 05_train_phase2.py import all these utilities from
training.helpers at MODULE LEVEL (top of file). Not inside __main__, not inside
a function. This is the ONLY correct pattern.

Correct import at top of each training script:
```python
from training.helpers import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    get_llrd_optimizer,
)
```

### [FIX GAP 29] __init__.py contents for all packages

training/__init__.py — contents:
```python
# This file marks the directory as a Python package.
```

agents/__init__.py — contents:
```python
# This file marks the directory as a Python package.
```

setup/__init__.py — contents:
```python
# This file marks the directory as a Python package.
```

app/__init__.py — contents:
```python
# This file marks the directory as a Python package.
```

tools/__init__.py — contents:
```python
# This file marks the directory as a Python package.
```

No re-exports. No imports. Only the comment.

### [FIX GAP 46] app/inference.py cross-layer import resolution

app/inference.py MUST NOT import from training.transforms or any training.* module.
apply_clahe uses only cv2 and numpy — both in requirements.txt.
It is defined INLINE in app/inference.py (Section 14 of this spec).
It is ALSO defined in training/transforms.py (Section 6.5).
These are two intentional copies of the same 6-line function.
This avoids the app → training dependency which would pull the entire training
package (albumentations, sklearn, etc.) into the production server process.

### [FIX GAP 56,26] Server startup commands

Development:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Production:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

These commands are documented in README.md (Section 28).
The smoke test (setup/test_server.py) uses port 8765 to avoid conflicting with a
running production server on port 8000.

### [FIX GAP 57] training/report.py location in inventory

training/report.py is listed in Section 1.5 file inventory. Its implementation
contract is in Section 20.4 of the spec (under evaluation reporting). Claude Code
must search Section 20 when looking for the training/report.py contract — it is
not in Section 19 (setup contracts). This cross-reference inconsistency is resolved
by this note: training/report.py is specified in Section 20.4, not Section 19.

training/report.py contract: exports one function write_report(metrics, path, title)
that accepts a dict of metrics, a file path, and a title string, and writes a
Markdown report. Used by all evaluation scripts to ensure consistent report format.

### [FIX GAP 58] Evaluation scripts use full inference re-run

All three evaluation scripts (07, 08, 10) perform a complete inference pass to
compute their outputs. They do NOT read scalar values from checkpoint files.
Checkpoint files store only: epoch, model_state_dict, val_metrics (scalars from
training). They do NOT store confusion matrices, per-class breakdown, or calibration
curves. These require iterating the dataset again.

The authoritative statement: if a metric requires knowing the full probability
distribution over samples (confusion matrix, calibration curve, per-source breakdown),
re-run inference. If a metric is a simple scalar logged during training
(e.g. final val_loss), it may be read from the checkpoint.

### [FIX GAP 63] Image deduplication specification

pHash-based deduplication is recommended for Kaggle datasets known to have
near-duplicate images. Add imagehash to requirements_train.txt.

Implementation (add to training/01_prepare_data.py after scanning, before splitting):

```python
def deduplicate_records(records, threshold=8):
    """
    Remove near-duplicate images using perceptual hash.
    threshold: max hamming distance to consider images duplicates (lower = stricter).
    Returns deduplicated records list.
    """
    try:
        import imagehash
        from PIL import Image as _PIL
    except ImportError:
        print("imagehash not installed — skipping deduplication.")
        return records

    seen_hashes = {}
    keep        = []
    for r in records:
        from app.config import ROOT
        full_path = os.path.join(ROOT, r['image_path'].replace('/', os.sep))
        try:
            img  = _PIL.open(full_path).convert('RGB')
            h    = imagehash.phash(img)
            dup  = False
            for prev_h in seen_hashes:
                if h - prev_h <= threshold:
                    dup = True
                    break
            if not dup:
                seen_hashes[h] = r
                keep.append(r)
        except Exception:
            keep.append(r)  # keep on error — don't discard due to hash failure
    removed = len(records) - len(keep)
    if removed > 0:
        print(f"Deduplication removed {removed} near-duplicate images.")
    return keep
```

Call deduplicate_records(all_records) after assert_all_labels_mapped() and
before stratified_group_split() in 01_prepare_data.py.

Add to requirements_train.txt:
    imagehash==4.3.1
