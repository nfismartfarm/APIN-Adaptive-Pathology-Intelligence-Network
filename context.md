# context.md — Plant Disease Detection: Kerala
# Living document. Updated by Claude Code after every significant event.
# Read this at the start of every new session before touching any code.
# Last updated: 2026-03-23

---

## HOW TO USE THIS FILE

This file is your memory across sessions. Claude Code has no memory between
conversations. Without this file, every new session starts from zero — making
decisions that conflict with decisions already made, re-discovering problems
that were already solved, re-downloading data that was already downloaded.

**At the start of every session:** Read this entire file first. Do not skip sections.
Do not assume the project is in the state CLAUDE.md describes — assume the actual
state is what this file says it is.

**After every significant action:** Update the relevant section. "Significant" means:
any step completed, any error encountered and resolved, any metric measured, any
decision made that differs from what CLAUDE.md specified, any file structure that
differs from what was expected.

**What this file is NOT:** It is not a tutorial. It is not a spec. CLAUDE.md is
the spec. This file is the ground truth about what has actually happened on this
specific machine, with this specific data, in this specific run of the project.

---

## SECTION 1: ENVIRONMENT STATUS

### 1.1 Machine

```
OS               : Windows 11
CPU              : Intel Core i7 12th gen
GPU              : NVIDIA RTX 4060 8 GB VRAM
Python version   : 3.13.11 (NOTE: spec says 3.10 — may need version-compatible packages)
Virtual env path : C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger\venv
Project root     : C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger
```

### 1.2 CUDA status

```
CUDA installed   : [YES / NO / NOT YET]
CUDA version     : [FILL IN — run: nvcc --version]
PyTorch version  : [FILL IN — run: python -c "import torch; print(torch.__version__)"]
CUDA available   : [FILL IN — run: python -c "import torch; print(torch.cuda.is_available())"]
GPU name         : [FILL IN — run: python -c "import torch; print(torch.cuda.get_device_name(0))"]
VRAM total       : [FILL IN — e.g. 8.0 GB]
Driver version   : [FILL IN — run: nvidia-smi]
torch.compile    : [WORKS / FAILS — requires MSVC Build Tools]
```

If CUDA is not working, record the exact error here:
```
CUDA error encountered: [PASTE ERROR]
Resolution attempted : [WHAT WAS DONE]
Resolution status    : [RESOLVED / PENDING]
```

### 1.3 Dependencies

```
requirements_train.txt installed : [YES / NO / PARTIAL]
All imports verified             : [YES / NO]
Failed imports (if any)          : [LIST ANY THAT FAILED]
albumentations version           : [FILL IN]
timm version                     : [FILL IN]
pytorch-grad-cam version         : [FILL IN]
wandb version                    : [FILL IN]
```

### 1.4 Credentials

```
KAGGLE_USERNAME set  : YES
KAGGLE_KEY set       : YES (new KGAT_ token format)
WANDB_API_KEY set    : YES
GITHUB_TOKEN set     : YES
GITHUB_REPO set      : YES
kaggle.json written  : NO — will be written by kaggle_utils.py at Step 03
```

---

## SECTION 2: PIPELINE STEP STATUS

Each step has three fields: Done (yes/no), Outcome (pass/fail/partial), Notes.
Update this table as steps complete. Do NOT mark a step Done until its smoke test passes.

| Step | Script | Done | Outcome | Notes |
|------|--------|------|---------|-------|
| 00 | setup/setup_project.py | NO | — | |
| 01 | setup/install_cuda.py | NO | — | |
| 02 | setup/install_dependencies.py | NO | — | |
| 03 | download_orchestrator.run_all_downloads() | NO | — | |
| 04 | acquire_kerala_images.acquire_all() | NO | — | |
| 05 | training/01_prepare_data.py | NO | — | |
| 06 | training/02_generate_severity.py | NO | — | |
| 07 | training/03_cache_features.py | NO | — | |
| 08 | training/04_train_phase1.py | NO | — | |
| 09 | training/05_train_phase2.py | NO | — | |
| 10 | training/06_calibrate.py | NO | — | |
| 11 | training/07_evaluate_validation.py | NO | — | |
| 12 | setup/test_server.py | NO | — | |
| 13 | training/08_evaluate_tier2_plantdoc.py | NO | — | |
| 14 | setup/package_deployment.py | NO | — | |
| — | training/10_evaluate_local_test.py | NO | — | Manual — run after step 13 |
| — | training/09_evaluate_tier3_kerala.py | NO | — | Manual — run when 50+ Kerala images collected |

---

## SECTION 3: DATA STATUS

### 3.1 Downloaded datasets

Update each row after Step 03 completes.

| Dataset | Expected min | Actual count | Status | Notes |
|---------|-------------|--------------|--------|-------|
| sabbir_okra | 3000 | [FILL] | [OK/FAIL] | |
| iubat_okra | 2000 | [FILL] | [OK/FAIL/MANUAL] | Roboflow may require manual download |
| kareem_cabbage | 3800 | [FILL] | [OK/FAIL] | |
| misrak_veg (brassica only) | 1500 | [FILL] | [OK/FAIL] | After filtering non-brassica |
| faruk_okra | 1600 | [FILL] | [OK/FAIL] | |
| ghose_cabbage | 1800 | [FILL] | [OK/FAIL] | |
| plantdoc | 2000 | [FILL] | [OK/FAIL] | Tier-2 test only — never training |

**iubat_okra manual download status:**
If Roboflow API failed and MANUAL_DOWNLOAD_REQUIRED.txt was created:
```
Manual download required : [YES / NO]
URL provided in txt file : [PASTE URL]
Manual download completed: [YES / NO / PENDING]
Images manually placed   : [YES / NO]
```

**Unexpected folder structures encountered:**
If any dataset had a folder structure different from what CLAUDE.md specified:
```
Dataset   : [NAME]
Expected  : [WHAT CLAUDE.md SAID]
Actual    : [WHAT WAS FOUND]
Resolution: [HOW IT WAS HANDLED — e.g. added subfolder scan logic]
```

### 3.2 Label harmonisation results

Update after Step 05 (01_prepare_data.py) completes.

```
Total records scanned        : [FILL IN]
Records with mapped labels   : [FILL IN]
Records skipped (unmapped)   : [FILL IN]
Label assertion result       : [PASSED / FAILED]
```

**Labels that were NOT in LABEL_MAP and had to be added to app/config.py:**
```
source=, raw_label=, mapped_to=
source=, raw_label=, mapped_to=
[Add rows as needed. If none, write: None — all labels mapped on first run]
```

### 3.3 Split results

Update after Step 05 completes.

```
Total training images        : [FILL IN]
Total val images             : [FILL IN]
Total test images            : [FILL IN]
PlantDoc images (plantdoc)   : [FILL IN]
Kerala images (kerala)       : [FILL IN]
```

**Class counts in training split:**

| Class | Count | Thin? (< 150) |
|-------|-------|---------------|
| okra_yvmv | [FILL] | [YES/NO] |
| okra_powdery_mildew | [FILL] | [YES/NO] |
| okra_cercospora | [FILL] | [YES/NO] |
| okra_enation | [FILL] | [YES/NO] |
| okra_healthy | [FILL] | [YES/NO] |
| brassica_black_rot | [FILL] | [YES/NO] |
| brassica_downy_mildew | [FILL] | [YES/NO] |
| brassica_alternaria | [FILL] | [YES/NO] |
| brassica_clubroot | [FILL] | [YES/NO] |
| brassica_healthy | [FILL] | [YES/NO] |

**Thin classes and what was done:**
```
[If brassica_clubroot < 150: did synthetic generation run? How many generated?]
[If okra_enation < 150: same.]
[Write: None — all classes met 150 minimum, if that is the case]
```

### 3.4 Severity label distribution

Update after Step 06 (02_generate_severity.py) completes.

```
Mild (0)      : [FILL IN count and %]
Moderate (1)  : [FILL IN count and %]
Severe (2)    : [FILL IN count and %]
Total labelled: [FILL IN]
Phase 1 model used for saliency: [YES — phase1_best.pt / NO — random fallback]
```

If random fallback was used (phase1_best.pt did not exist when severity ran):
```
Ran severity again after Phase 1: [YES / NO / PENDING]
```

### 3.5 Feature cache

Update after Step 07 (03_cache_features.py) completes.

```
train_features.pt size  : [FILL IN — e.g. 450 MB]
val_features.pt size    : [FILL IN]
Backbone shapes verified: [YES / NO]
Actual P3 shape         : [FILL IN — expected (B,48,28,28)]
Actual P4 shape         : [FILL IN — expected (B,160,14,14)]
Actual P5 shape         : [FILL IN — expected (B,256,7,7)]
```

If backbone shapes were WRONG (required FPN_IN_CH update):
```
Actual channels found : [FILL IN]
FPN_IN_CH updated to  : [FILL IN]
Config updated        : [YES / NO]
```

### 3.6 Kerala acquisition results

Update after Step 04 (acquire_kerala_images.acquire_all()) completes.

```
iNaturalist images downloaded : [FILL IN] (domain_adapt, no disease labels)
YouTube frames acquired       : [FILL IN] (domain_adapt, no disease labels)
Synthetic images generated    : [FILL IN] (training, labelled)
  ├── brassica_clubroot       : [FILL IN]
  ├── okra_enation            : [FILL IN]
  └── other classes           : [FILL IN]
yt_dlp installed              : [YES / NO — affects YouTube acquisition]
diffusers installed           : [YES / NO — affects synthetic generation]
```

---

## SECTION 4: TRAINING STATUS

### 4.1 Phase 1 — Head training

Update after Step 08 (04_train_phase1.py) completes.

```
Status               : [COMPLETE / IN PROGRESS / NOT STARTED / FAILED]
Epochs completed     : [FILL IN] / 10
Stopped by           : [EARLY STOPPING at epoch X / EPOCH LIMIT]
Best macro F1        : [FILL IN]
Best epoch           : [FILL IN]
phase1_best.pt saved : [YES / NO]
Training duration    : [FILL IN — e.g. 28 minutes]
wandb run URL        : [PASTE wandb URL or "offline"]
```

**Per-class F1 at best epoch (from wandb or terminal output):**

| Class | F1 |
|-------|-----|
| okra_yvmv | [FILL] |
| okra_powdery_mildew | [FILL] |
| okra_cercospora | [FILL] |
| okra_enation | [FILL] |
| okra_healthy | [FILL] |
| brassica_black_rot | [FILL] |
| brassica_downy_mildew | [FILL] |
| brassica_alternaria | [FILL] |
| brassica_clubroot | [FILL] |
| brassica_healthy | [FILL] |

**Phase 1 issues encountered:**
```
[Write any problems here: OOM, import errors, unexpected loss behaviour, etc.]
[Write: None — Phase 1 completed without issues, if applicable]
```

**If macro F1 < 0.30 after Phase 1:**
```
F1 achieved : [FILL IN]
Root cause  : [FILL IN — e.g. bad label mappings, class imbalance not handled]
Action taken: [FILL IN]
```

### 4.2 Phase 2 — Full fine-tuning

Update after Step 09 (05_train_phase2.py) completes.

```
Status                : [COMPLETE / IN PROGRESS / NOT STARTED / FAILED / RESUMED]
Resumed from ckpt     : [YES — phase2_epoch{N}.pt / NO — started fresh]
Epochs completed      : [FILL IN] / 7
Stopped by            : [EARLY STOPPING at epoch X / EPOCH LIMIT]
Best macro F1         : [FILL IN]
Best epoch            : [FILL IN]
best_model.pt saved   : [YES / NO]
Training duration     : [FILL IN — e.g. 3h 22min]
torch.compile enabled : [YES / NO — NO on Windows is normal]
Mixed precision (AMP) : [YES / NO]
VRAM peak usage       : [FILL IN — check nvidia-smi during training]
Batch size used       : [32 / 16 — was 16 used due to OOM?]
GRAD_ACCUM_STEPS      : [1 / 2 — was 2 used to compensate for batch 16?]
wandb run URL         : [PASTE wandb URL or "offline"]
```

**Phase 2 issues encountered:**
```
[OOM errors and resolutions]
[torch.compile failures]
[Gradient explosion incidents (loss went to inf)]
[Any step where checkpoint resume was needed]
[Write: None — Phase 2 completed without issues, if applicable]
```

**If machine was restarted mid-Phase 2:**
```
Restart at epoch     : [FILL IN]
Checkpoint resumed   : [phase2_epoch{N}.pt]
Epochs lost          : [FILL IN]
```

### 4.3 Temperature calibration

Update after Step 10 (06_calibrate.py) completes.

```
Status          : [COMPLETE / NOT STARTED]
T_disease       : [FILL IN — expected range 0.8 to 2.5]
T_crop          : [FILL IN]
T_severity      : [FILL IN]
ECE before      : [FILL IN — lower is better]
ECE after       : [FILL IN — should be < 0.10]
temperature.pt  : [SAVED / NOT SAVED]
```

If ECE after calibration is >= 0.10:
```
ECE achieved : [FILL IN]
Possible cause: [model overconfident / underconfident / miscalibrated LBFGS]
Action taken  : [FILL IN]
```

---

## SECTION 5: EVALUATION RESULTS

### 5.1 Tier-1 validation results

Update after Step 11 (07_evaluate_validation.py) completes.
Report file saved at: [FILL IN — e.g. reports/validation_report_20250601_142033.md]

```
Macro F1 (disease) : [FILL IN] — acceptance >= 0.50
Crop accuracy      : [FILL IN] — expected > 0.90
ECE post-calibration: [FILL IN] — expected < 0.10
Val images evaluated: [FILL IN]
Report path        : [FILL IN]
```

**Weakest classes (F1 < 0.40):**
```
[List any class with F1 below 0.40 here]
[Write: None — all classes above 0.40, if applicable]
```

### 5.2 Tier-2 PlantDoc results

**DO NOT run 08_evaluate_tier2_plantdoc.py until all training is finalised.**
Update after Step 13 runs (run ONCE only).

```
Run status          : [NOT YET RUN / RUN ON date]
Macro F1 (4 mappable classes): [FILL IN] — acceptance >= 0.55
Acceptance threshold: 0.55
Result              : [PASS / FAIL]
Report path         : [FILL IN — e.g. reports/tier2_plantdoc_20250601_160000.md]
PlantDoc images used: [FILL IN]
```

**Per-class F1 (PlantDoc-mappable only):**

| Class | F1 |
|-------|-----|
| brassica_black_rot | [FILL] |
| brassica_downy_mildew | [FILL] |
| brassica_alternaria | [FILL] |
| brassica_healthy | [FILL] |

**If FAIL — gap analysis:**
```
Failing classes : [LIST]
Likely cause    : [domain shift / insufficient training data / label mismatch]
Action planned  : [FILL IN — or "No action — accept current performance for now"]
```

**LOCKED: After tier-2 runs, no further model changes permitted.**
```
Tier-2 lock status: [UNLOCKED — not yet run / LOCKED — run on date]
```

### 5.3 Local test set results (15% split)

Run training/10_evaluate_local_test.py ONCE after tier-2.

```
Run status   : [NOT YET RUN / RUN ON date]
Macro F1     : [FILL IN]
Crop accuracy: [FILL IN]
ECE          : [FILL IN]
Report path  : [FILL IN]
```

### 5.4 Tier-3 Kerala field results

Run training/09_evaluate_tier3_kerala.py when 50+ verified Kerala images collected.

```
Kerala images collected : [FILL IN] / 50 minimum
Classes represented     : [FILL IN — e.g. 6 of 10]
Run status              : [NOT YET RUN / RUN ON date]
Overall result          : [PASS / FAIL / NOT RUN]
Report path             : [FILL IN]
```

**Per-class results (classes with >= 5 Kerala images):**
```
[Leave blank until tier-3 is run]
```

---

## SECTION 6: SERVER AND DEPLOYMENT STATUS

### 6.1 Server smoke test

Update after Step 12 (setup/test_server.py) runs.

```
Health check       : [PASS / FAIL]
Predict endpoint   : [PASS / FAIL]
OOD flag working   : [YES / NO / NOT TESTED]
Heatmap returned   : [YES / NO]
Response time      : [FILL IN — e.g. 6.2 seconds per image]
Server error (if)  : [PASTE ANY ERRORS]
```

### 6.2 Development server

To start the server after training:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```
Server tested manually : [YES / NO]
URL                    : http://localhost:8000
Any runtime errors     : [PASTE ANY ERRORS or "None"]
```

### 6.3 Deployment packaging

Update after Step 14 (setup/package_deployment.py) completes.

```
Dockerfile created    : [YES / NO]
All deployment files  : [VERIFIED / NOT YET]
```

---

## SECTION 7: ERRORS ENCOUNTERED AND RESOLUTIONS

This is a running log. Append new entries at the bottom. Never delete old entries.

Format for each entry:
```
DATE       : [date]
STEP       : [which pipeline step]
ERROR      : [exact error message or description]
CAUSE      : [root cause identified]
RESOLUTION : [what was done to fix it]
STATUS     : [RESOLVED / WORKAROUND / PENDING]
```

**Entry 1:**
```
DATE       : [FILL IN]
STEP       : [FILL IN]
ERROR      : [FILL IN]
CAUSE      : [FILL IN]
RESOLUTION : [FILL IN]
STATUS     : [FILL IN]
```

[Add more entries as needed. Leave the template entry above for reference.]

---

## SECTION 8: ACTUAL VS PREDICTED DIFFERENCES

This is the most important section for catching spec drift. Record any time the
actual outcome differs from what CLAUDE.md predicted. These differences inform
whether the spec needs updating and prevent the same surprises in future sessions.

**Training time:**
```
Phase 1 predicted : 25-35 minutes
Phase 1 actual    : [FILL IN]
Phase 2 predicted : 3-3.5 hours
Phase 2 actual    : [FILL IN]
```

**VRAM usage:**
```
Phase 2 predicted peak: ~6.1 GB at batch 32
Phase 2 actual peak   : [FILL IN]
OOM occurred          : [YES — had to reduce batch size / NO]
Batch size ultimately used: [32 / 16]
```

**Model performance:**
```
Phase 1 macro F1 predicted: > 0.30
Phase 1 macro F1 actual   : [FILL IN]
Phase 2 macro F1 predicted: > 0.50
Phase 2 macro F1 actual   : [FILL IN]
ECE predicted             : < 0.10
ECE actual                : [FILL IN]
Tier-2 F1 predicted       : > 0.55
Tier-2 F1 actual          : [FILL IN]
```

**Data pipeline:**
```
Total images expected: ~13,700 (rough sum of all dataset minimums)
Total images actual  : [FILL IN]
Any dataset that had unexpected folder structure: [FILL IN or None]
Any labels that were not in LABEL_MAP: [FILL IN or None]
```

**torch.compile:**
```
Expected: enabled on Windows if MSVC present
Actual   : [ENABLED / DISABLED — reason if disabled]
```

**Backbone shapes:**
```
Expected P3: (B, 48, 28, 28)
Actual P3  : [FILL IN]
Expected P4: (B, 160, 14, 14)
Actual P4  : [FILL IN]
Expected P5: (B, 256, 7, 7)
Actual P5  : [FILL IN]
FPN_IN_CH updated: [YES — to [x, y, z] / NO — matched expected]
```

---

## SECTION 9: KERALA IMAGE COLLECTION

Track progress toward the 50-image tier-3 minimum.

```
Total Kerala images in data/kerala/ : [FILL IN]
Target                              : 50 minimum, across >= 6 classes
```

**Images per class:**

| Class | Count | Verified? |
|-------|-------|-----------|
| okra_yvmv | [FILL] | [YES/NO] |
| okra_powdery_mildew | [FILL] | [YES/NO] |
| okra_cercospora | [FILL] | [YES/NO] |
| okra_enation | [FILL] | [YES/NO] |
| okra_healthy | [FILL] | [YES/NO] |
| brassica_black_rot | [FILL] | [YES/NO] |
| brassica_downy_mildew | [FILL] | [YES/NO] |
| brassica_alternaria | [FILL] | [YES/NO] |
| brassica_clubroot | [FILL] | [YES/NO] |
| brassica_healthy | [FILL] | [YES/NO] |

**Image sources used:**
```
iNaturalist GPS-filtered (disease-labelled) : [FILL IN count]
KVK / ICAR-IIHR publications               : [FILL IN count]
Farmer submissions via feedback             : [FILL IN count]
Other (specify)                             : [FILL IN]
```

To add a new Kerala image:
  python tools/add_kerala_image.py --path path/to/image.jpg --class okra_yvmv

---

## SECTION 10: WANDB RUNS

Running log of all wandb training runs. Useful for comparing hyperparameter changes.

| Run name | Phase | Epochs | Best macro F1 | URL | Notes |
|----------|-------|--------|---------------|-----|-------|
| phase1 | 1 | [FILL] | [FILL] | [FILL] | |
| phase2 | 2 | [FILL] | [FILL] | [FILL] | |
| calibration | — | — | — | [FILL] | T values logged |

If wandb ran offline (no API key):
```
Local wandb logs at: [FILL IN — e.g. wandb/offline-run-*/]
Synced to cloud    : [YES / NO / PENDING]
```

---

## SECTION 11: CURRENT SESSION CONTEXT

Update this section at the START of each new Claude Code session to orient quickly.

```
Session date          : 2026-03-23
Last step completed   : None — project not started
Next step to run      : Step 00 — setup/setup_project.py
Anything blocking     : None — creating all project files now
State of working tree : clean (only spec files present, untracked)
Last GitHub commit    : None — no commits yet
```

**What was happening when the last session ended:**
```
First session. Read all 5 spec files. Venv confirmed active (Python 3.13.11). All 5 env vars
confirmed set in .env. Kaggle key uses new KGAT_ format. Now creating all 56 project files
from CLAUDE.md spec, followed by pessimistic review loop.
```

**Unresolved questions from last session:**
```
1. Python 3.13.11 vs spec's 3.10 — pinned package versions may need adjustment at Step 02
2. Kaggle key uses new KGAT_ format — kaggle_utils.py updated to handle both formats
```

---

## SECTION 12: PROJECT HEALTH SUMMARY

Quick-reference card. Update after each major milestone.

```
Overall status       : NOT STARTED
Blocking issues      : None
Model ready to serve : [YES / NO]
Deployment-validated : [YES — tier-3 passed / NO — tier-3 not yet run]

Phase 1 macro F1 : [FILL IN or "—"]
Phase 2 macro F1 : [FILL IN or "—"]
Tier-2 macro F1  : [FILL IN or "—" — run ONCE, then locked]
Tier-3 result    : [FILL IN or "Not yet run"]

Key risks remaining:
  □ Tier-3 not yet validated (50+ Kerala images needed)
  □ [Add any other specific risks here]
```
