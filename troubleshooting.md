# troubleshooting.md — Plant Disease Detection: Kerala
# Pre-populated with every predictable failure mode for this project.
# Platform: Windows 11, RTX 4060 8GB, Python 3.10, local-only execution.
# Claude Code reads this before attempting to diagnose any error.
# Append new entries at the bottom as new issues are discovered.
# NEVER delete entries — a solved problem may recur.

---

## HOW TO USE THIS FILE

When something fails:
1. Copy the exact error message.
2. Search this file (Ctrl+F) for key words from the error.
3. Read the full entry — especially CAUSE and EXACT FIX.
4. Follow the fix steps exactly, in order.
5. If the fix works, note it in context.md Section 7.
6. If the fix does not work, append a new entry at the bottom with what was found.

This file is organised by pipeline stage. Jump directly to the stage where
the failure occurred. If the stage is unclear, search by error message keyword.

---

## SECTION 1: ENVIRONMENT AND SETUP FAILURES

---

### T-001: "Not running inside a virtual environment"

```
SYMPTOM    : setup_project.py exits immediately with:
             "ERROR: Not running inside a virtual environment."

CAUSE      : The virtual environment was not activated before running the script,
             or the script was called via a system Python installation.

EXACT FIX  :
  Step 1 — Open a new Command Prompt (not PowerShell, not Git Bash for this step).
  Step 2 — Navigate to the project root:
             cd C:\Users\seena\plant_identification
  Step 3 — Create the venv if it does not exist yet:
             python -m venv venv
  Step 4 — Activate it:
             venv\Scripts\activate.bat
  Step 5 — Verify activation (should show venv path in prompt prefix):
             python -c "import sys; print(sys.prefix)"
  Step 6 — Re-run the failing script.

  If using PowerShell instead of CMD:
             venv\Scripts\Activate.ps1
  If PowerShell blocks execution policy:
             Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
             venv\Scripts\Activate.ps1

VERIFY     : (venv) prefix appears in the terminal prompt before the path.
PREVENTION : Always activate venv before any python command. Add activation to
             your terminal startup or project README habit.
```

---

### T-002: "KAGGLE_USERNAME and KAGGLE_KEY must be set"

```
SYMPTOM    : Any download agent or setup_project.py fails with:
             "EnvironmentError: KAGGLE_USERNAME and KAGGLE_KEY must be set"

CAUSE      : The .env file was not created, or load_dotenv() failed, or the
             venv does not have python-dotenv installed yet.

EXACT FIX  :
  Step 1 — Check .env exists:
             dir .env
             (Should show the file. If not: copy .env.template .env)
  Step 2 — Open .env in a text editor and verify KAGGLE_USERNAME and KAGGLE_KEY
             are filled in (not the template placeholder values).
  Step 3 — Get Kaggle credentials if you do not have them:
             Go to kaggle.com → Account → API → Create New API Token.
             This downloads kaggle.json containing your username and key.
  Step 4 — Install python-dotenv if not yet installed:
             pip install python-dotenv
  Step 5 — Verify the .env is being loaded:
             python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('KAGGLE_USERNAME', 'NOT SET'))"
  Step 6 — If it shows NOT SET, check that .env is in the project ROOT (same
             directory as run_pipeline.py and CLAUDE.md).

VERIFY     : python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('KAGGLE_USERNAME'))"
             Should print your Kaggle username.
PREVENTION : Run setup_project.py (Step 00) before any other step. It validates
             all environment variables and exits with clear instructions.
```

---

### T-003: "ModuleNotFoundError: No module named 'app'"

```
SYMPTOM    : Any training script fails with:
             "ModuleNotFoundError: No module named 'app'"
             or
             "ModuleNotFoundError: No module named 'training'"

CAUSE      : The script was run from the wrong directory, or sys.path.insert()
             at the top of the script is missing or incorrect.

EXACT FIX  :
  Step 1 — Always run scripts from the project ROOT directory:
             cd C:\Users\seena\plant_identification
             python training/04_train_phase1.py
             NOT from inside the training/ subdirectory.
  Step 2 — If running from root still fails, check the sys.path.insert at the top
             of the failing script. It should be:
             sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             This inserts the project root (two levels up from the script) into sys.path.
  Step 3 — Verify app/__init__.py exists (even if empty):
             dir app\__init__.py
             If missing: create it with content: # This file marks the directory as a Python package.
  Step 4 — Same check for training/__init__.py, agents/__init__.py, setup/__init__.py.

VERIFY     : python -c "import app.config; print('OK')"
             Should print OK.
PREVENTION : Always run from project root. run_pipeline.py sets cwd=ROOT for all
             subprocess calls, so this issue only occurs when running manually.
```

---

### T-004: "python-dotenv is not installed" or dotenv ImportError

```
SYMPTOM    : try/except block silently swallows dotenv import and env vars are not loaded,
             causing downstream credential failures.

CAUSE      : requirements_train.txt was not yet installed when a script was run.

EXACT FIX  :
  Step 1 — Install dependencies first (this is Step 02 in the pipeline):
             pip install -r requirements_train.txt
  Step 2 — Verify dotenv is installed:
             pip show python-dotenv
  Step 3 — Re-run the failing script.

VERIFY     : python -c "from dotenv import load_dotenv; print('dotenv OK')"
PREVENTION : Never skip Step 02 (install_dependencies.py). The pipeline enforces
             this order, but if running manually, always install deps first.
```

---

## SECTION 2: CUDA AND GPU FAILURES

---

### T-005: "torch.cuda.is_available() returns False"

```
SYMPTOM    : Training starts but uses CPU, taking 10× longer than expected.
             Or: python -c "import torch; print(torch.cuda.is_available())" prints False.

CAUSE A    : PyTorch was installed without the CUDA index URL (CPU-only build).
CAUSE B    : CUDA Toolkit 12.1 is not installed or not in PATH.
CAUSE C    : NVIDIA driver version < 525.

DIAGNOSE   :
  Run: nvidia-smi
  - If "nvidia-smi is not recognized": driver not installed → CAUSE C
  - If shows driver version < 525: update driver → CAUSE C
  - If shows driver >= 525: check PyTorch build → CAUSE A

EXACT FIX FOR CAUSE A (wrong PyTorch build):
  Step 1 — Uninstall current PyTorch:
             pip uninstall torch torchvision torchaudio -y
  Step 2 — Reinstall with CUDA 12.1 index URL:
             pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
             --index-url https://download.pytorch.org/whl/cu121
  Step 3 — Verify:
             python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
             Should print: True and 12.1

EXACT FIX FOR CAUSE B (CUDA toolkit not in PATH):
  Step 1 — Add to Windows PATH:
             C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
  Step 2 — Open a new Command Prompt (PATH changes need new terminal).
  Step 3 — Verify: nvcc --version (should show release 12.1)

EXACT FIX FOR CAUSE C (old driver):
  Step 1 — Download driver: nvidia.com/drivers
             Product: GeForce → RTX 4060 → Windows 11 → Game Ready Driver
  Step 2 — Run installer as Administrator. Choose Express.
  Step 3 — Restart Windows.
  Step 4 — Verify: nvidia-smi (should show driver >= 525)
  Step 5 — Reinstall PyTorch (CAUSE A steps).

VERIFY     : python -c "import torch; assert torch.cuda.is_available(); x = torch.randn(100,100,device='cuda'); print('GPU OK:', torch.cuda.get_device_name(0))"
PREVENTION : Step 01 of the pipeline (install_cuda.py) walks through this diagnosis.
             Run it before any training.
```

---

### T-006: CUDA Out of Memory (OOM) during Phase 2 training

```
SYMPTOM    : RuntimeError: CUDA out of memory. Tried to allocate X GiB.
             Occurs during the first or second batch of Phase 2 training.

CAUSE      : The default BATCH_SIZE=32 requires ~6.1 GB VRAM. The RTX 4060 has
             8 GB but other processes (browser, other apps) may consume 1-2 GB.
             The actual available VRAM at training time may be only 6-7 GB.

EXACT FIX  :
  Step 1 — Close all other GPU-using applications (browser tabs with GPU acceleration,
             games, other ML processes). Check with nvidia-smi.
  Step 2 — If OOM persists after clearing other processes:
             Open app/config.py and change:
               BATCH_SIZE = 32  →  BATCH_SIZE = 16
               GRAD_ACCUM_STEPS = 1  →  GRAD_ACCUM_STEPS = 2
             This is mathematically equivalent to batch 32 but uses half the VRAM.
  Step 3 — Clear CUDA cache before restarting training:
             python -c "import torch; torch.cuda.empty_cache(); print('cache cleared')"
  Step 4 — Delete any incomplete phase2_epoch*.pt checkpoints from models/checkpoints/
             (they may occupy disk space and confuse the resume logic if corrupted).
  Step 5 — Restart Phase 2: python training/05_train_phase2.py

  If OOM persists at batch 16:
  Step 6 — Further reduce: BATCH_SIZE=8, GRAD_ACCUM_STEPS=4

RECORD IN context.md:
  - Whether OOM occurred
  - Final BATCH_SIZE and GRAD_ACCUM_STEPS used
  - Actual VRAM peak (monitor with nvidia-smi dmon -s u during training)

VERIFY     : Monitor VRAM during first 5 batches: nvidia-smi dmon -s u
             Should stay below 7.5 GB.
PREVENTION : Check available VRAM before starting Phase 2:
             python -c "import torch; print(torch.cuda.mem_get_info())"
             Returns (free, total). Free should be > 6.5 GB before starting.
```

---

### T-007: "RuntimeError: Failed to find C compiler" (torch.compile)

```
SYMPTOM    : Phase 2 training prints:
             "RuntimeError: Failed to find C compiler. Please install MSVC."
             or similar torch.compile failure.

CAUSE      : torch.compile on Windows requires MSVC Build Tools (Visual C++ compiler).
             They are not installed by default.

EXACT FIX  :
  Step 1 — Download MSVC Build Tools:
             https://aka.ms/vs/17/release/vs_BuildTools.exe
  Step 2 — Run the installer. Select "Desktop development with C++".
             Install takes 5-15 minutes and requires ~4 GB disk space.
  Step 3 — Restart Windows (PATH must be updated).
  Step 4 — Re-run Phase 2.

  IMPORTANT: If you cannot install MSVC or choose not to:
  - torch.compile failure is caught and handled gracefully in 05_train_phase2.py.
  - Training continues without compile (compiled=False).
  - This adds ~25-35% to Phase 2 training time (3.5 hours → 4.5 hours).
  - This is acceptable. Do NOT try to workaround the compile failure in code.

RECORD IN context.md:
  - Whether torch.compile was enabled or disabled
  - Actual Phase 2 training time

VERIFY     : After fixing, test: python -c "import torch; m = torch.nn.Linear(10,10); mc = torch.compile(m); print('compile OK')"
PREVENTION : Install MSVC Build Tools as part of Step 01 (install_cuda.py guides this).
```

---

### T-008: CUDA version mismatch warnings

```
SYMPTOM    : UserWarning: CUDA version mismatch: PyTorch was compiled with CUDA X.X
             but CUDA Y.Y is found.

CAUSE      : The installed CUDA Toolkit version does not match the version PyTorch
             was compiled against. Common when CUDA Toolkit is updated after
             PyTorch is installed.

EXACT FIX  :
  Step 1 — Check what CUDA version PyTorch was compiled with:
             python -c "import torch; print(torch.version.cuda)"
  Step 2 — Check installed CUDA Toolkit:
             nvcc --version
  Step 3 — If they do not match (e.g. PyTorch says 12.1 but nvcc says 12.3):
             Option A: Reinstall PyTorch to match nvcc version.
               Check https://pytorch.org/get-started for the correct --index-url.
             Option B: PyTorch 2.2.0 is compiled for CUDA 12.1. If nvcc is 12.3,
               PyTorch usually still works (minor version compatible). The warning
               is usually safe to ignore if training runs without errors.
  Step 4 — If training runs without actual errors (no crash, reasonable loss values),
             the mismatch is minor and can be ignored.

VERIFY     : python -c "import torch; x = torch.randn(1000,1000,device='cuda'); y = torch.mm(x,x); print('matmul OK')"
```

---

## SECTION 3: DATA DOWNLOAD FAILURES

---

### T-009: Kaggle download fails — "403 Forbidden" or "404 Not Found"

```
SYMPTOM    : download agent raises RuntimeError:
             "Kaggle download failed for owner/dataset-name.
              stderr: 403: Forbidden"
             or
             "404: Not Found"

CAUSE A    : Kaggle credentials are wrong or expired.
CAUSE B    : The dataset slug changed or the dataset was deleted by its owner.
CAUSE C    : You have not accepted the dataset's terms on Kaggle.com.

DIAGNOSE   :
  Run: kaggle datasets list -s "owner/dataset-name"
  - If "401 Unauthorized": credentials wrong → CAUSE A
  - If "404": dataset gone → CAUSE B
  - If shows dataset but download fails: terms not accepted → CAUSE C

EXACT FIX FOR CAUSE A:
  Step 1 — Go to kaggle.com → Account → Create New API Token.
             This invalidates old tokens.
  Step 2 — Update .env with new KAGGLE_KEY.
  Step 3 — Delete ~/.kaggle/kaggle.json (will be rewritten by kaggle_utils.py).
  Step 4 — Re-run the download agent.

EXACT FIX FOR CAUSE B:
  Step 1 — Search kaggle.com for the dataset by its subject (e.g. "okra disease").
  Step 2 — Find a replacement dataset with similar content.
  Step 3 — Update the slug in the relevant download agent file.
  Step 4 — Update the minimum expected image count if different.
  Step 5 — Add a new entry to LABEL_MAP in app/config.py for any new label strings.
  Step 6 — Document the change in decisions.md (new DECISION-XX entry).

EXACT FIX FOR CAUSE C:
  Step 1 — Go to the dataset's Kaggle page in your browser.
  Step 2 — Click "Download" — if there is a terms acceptance dialog, accept it.
  Step 3 — Re-run the download agent.

RECORD IN context.md: Which dataset failed, which fix was used.
```

---

### T-010: iubat_okra Roboflow download fails — MANUAL_DOWNLOAD_REQUIRED.txt created

```
SYMPTOM    : After Step 03, data/raw/iubat_okra/ contains only
             MANUAL_DOWNLOAD_REQUIRED.txt and no images.
             The orchestrator logs: "[WARN] iubat_okra: Only 0 images, expected >= 2000"

CAUSE      : The Roboflow public API is unreliable for unauthenticated access.
             The IUBAT okra disease dataset requires a Roboflow API key or
             manual download from the dataset's Roboflow Universe page.

EXACT FIX  :
  Step 1 — Open MANUAL_DOWNLOAD_REQUIRED.txt in data/raw/iubat_okra/.
             It contains the exact Roboflow URL for the dataset.
  Step 2 — Go to that URL in your browser.
  Step 3 — Click "Download Dataset" → select format "Folder Structure"
             (NOT YOLO, NOT COCO — Folder Structure gives class-named subdirectories).
  Step 4 — Unzip the downloaded file into data/raw/iubat_okra/.
  Step 5 — Verify folder structure:
             data/raw/iubat_okra/
               ├── train/
               │   ├── Yellow_Vein_Mosaic/
               │   ├── Healthy/
               │   └── ...
             or flat class folders at top level.
  Step 6 — Verify image count:
             python -c "
             import os
             count = sum(1 for r,d,f in os.walk('data/raw/iubat_okra') for fn in f if fn.lower().endswith(('.jpg','.jpeg','.png')))
             print(f'iubat_okra images: {count}')
             "
             Should be >= 2000.
  Step 7 — Re-run 01_prepare_data.py (Step 05) — it will scan the newly added images.

  ALTERNATIVE: If iubat download remains impossible:
  - The pipeline continues without it (sabbir_okra has 3000+ images and covers same classes).
  - Note in context.md that iubat_okra was not included.
  - Do NOT force a fake iubat entry — the class counts will still meet minimums
    from the other okra datasets.

RECORD IN context.md: Whether manual download was completed or skipped.
```

---

### T-011: Unexpected folder structure in downloaded Kaggle dataset

```
SYMPTOM    : 01_prepare_data.py scans a source directory and finds 0 images,
             or scans subdirectories with unexpected names.
             Example: sabbir_okra has an extra nesting level not expected.

CAUSE      : Kaggle dataset owners occasionally restructure their datasets after
             the spec was written. The expected folder structure may have changed.

DIAGNOSE   :
  python -c "
  import os
  for root, dirs, files in os.walk('data/raw/sabbir_okra'):
      depth = root.replace('data/raw/sabbir_okra', '').count(os.sep)
      print(' ' * depth + os.path.basename(root) + '/')
      if depth > 3: break
  "

EXACT FIX  :
  Step 1 — Inspect the actual folder structure using the command above.
  Step 2 — Identify the level where class-named folders appear.
  Step 3 — Update _scan_source() in training/01_prepare_data.py:
             The current logic checks for a 'train' subdirectory and uses it
             if found, otherwise uses the root directory.
             If the structure has a different intermediate folder (e.g. 'dataset/train/'),
             add logic to detect and descend into it:
               for extra_dir in ['train', 'dataset', 'data']:
                   candidate = os.path.join(source_dir, extra_dir)
                   if os.path.isdir(candidate):
                       scan_root = candidate
                       break
  Step 4 — Re-run 01_prepare_data.py.
  Step 5 — Document the actual folder structure in context.md Section 3.1.

PREVENTION : verify_backbone_shapes() is not relevant here, but a similar
             "verify structure" scan should be added to the download agent
             for any newly encountered dataset variant.
```

---

### T-012: Label strings not in LABEL_MAP — label assertion fails

```
SYMPTOM    : 01_prepare_data.py raises ValueError:
             "Found X unmapped labels. Add them to LABEL_MAP or SOURCE_LABEL_OVERRIDES"
             followed by a list of offending labels.

CAUSE      : A downloaded dataset uses label strings not anticipated in LABEL_MAP
             or SOURCE_LABEL_OVERRIDES in app/config.py.
             Common: typos in dataset labels, different language variants,
             version updates to label names by dataset owners.

EXACT FIX  :
  Step 1 — Read the full list of unmapped labels printed by the error.
  Step 2 — For each unmapped label, determine the correct canonical class:
             - "okra_leaf_blight" → maps to okra_cercospora (Cercospora is a leaf blight)
             - "late_blight_broccoli" → maps to brassica_alternaria
             - "cabbage_mosaic" → no valid mapping, mark as unknown
  Step 3 — For source-specific mappings (where the same label string means different
             things in different datasets), add to SOURCE_LABEL_OVERRIDES in app/config.py:
               ('new_source_name', 'new_label_string'): 'canonical_class_name',
  Step 4 — For globally unambiguous labels, add to LABEL_MAP in app/config.py:
               'new_label_string': 'canonical_class_name',
  Step 5 — If a label genuinely has no valid mapping (not okra or brassica disease),
             it is correctly skipped by _scan_source() — no action needed.
  Step 6 — Re-run 01_prepare_data.py.
  Step 7 — Document all added mappings in context.md Section 3.2.

VERIFY     : 01_prepare_data.py prints "Label assertion passed: all X records are mapped."
```

---

### T-013: PlantDoc git clone fails or is incomplete

```
SYMPTOM    : download_plantdoc.py fails with git clone error, or
             data/plantdoc/ is empty or only partially populated.

CAUSE A    : git is not installed or not in PATH.
CAUSE B    : Network connection is slow — the clone timed out.
CAUSE C    : PlantDoc repository URL changed.

EXACT FIX FOR CAUSE A:
  Step 1 — Install Git for Windows: https://git-scm.com/download/win
  Step 2 — Open a new terminal (PATH must be updated).
  Step 3 — Verify: git --version
  Step 4 — Re-run the download agent.

EXACT FIX FOR CAUSE B:
  Step 1 — Run the clone manually with depth 1 to minimise download size:
             git clone --depth=1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/plantdoc
  Step 2 — If the clone is interrupted, delete data/plantdoc and retry.
  Step 3 — Verify after clone:
             python -c "
             import os
             count = sum(1 for r,d,f in os.walk('data/plantdoc') for fn in f if fn.lower().endswith(('.jpg','.jpeg','.png')))
             print(f'plantdoc images: {count}')
             "

EXACT FIX FOR CAUSE C:
  Step 1 — Search GitHub for "PlantDoc dataset" to find the current repository URL.
  Step 2 — Update the URL in agents/download_plantdoc.py.
  Step 3 — Re-run the download agent.

RECORD IN context.md: Actual PlantDoc image count.
```

---

## SECTION 4: DATA PREPARATION FAILURES (Step 05)

---

### T-014: Backbone shape mismatch — verify_backbone_shapes() fails

```
SYMPTOM    : 03_cache_features.py raises AssertionError:
             "Backbone stage X shape (B, Y, W, H) != expected (B, Z, W', H').
              Update FPN_IN_CH in app/config.py to [a, b, c]"

CAUSE      : The timm version installed returns different channel counts than
             expected for EfficientNetV2-S with out_indices=(2,3,4).
             Expected: [48, 160, 256] channels at stages [2, 3, 4].
             Some timm versions may return different values.

EXACT FIX  :
  Step 1 — Run the diagnostic to get actual shapes:
             python -c "
             import torch, timm
             from app.config import BACKBONE_NAME
             m = timm.create_model(BACKBONE_NAME, pretrained=False, features_only=True, out_indices=(2,3,4))
             x = torch.zeros(1,3,224,224)
             feats = m(x)
             print('Actual shapes:', [tuple(f.shape) for f in feats])
             print('Actual channels:', [f.shape[1] for f in feats])
             "
  Step 2 — Open app/config.py.
  Step 3 — Update FPN_IN_CH to match the actual channel values printed above:
             FPN_IN_CH = [actual_ch_stage2, actual_ch_stage3, actual_ch_stage4]
  Step 4 — Re-run 03_cache_features.py.

  IMPORTANT: If you change FPN_IN_CH, you must also delete any existing cache
             files (cache/train_features.pt, cache/val_features.pt) and rebuild
             the model from scratch. Old caches were built with wrong channel dims.

  Step 5 — Delete old cache: del cache\train_features.pt cache\val_features.pt
  Step 6 — Re-run 03_cache_features.py.
  Step 7 — Document the actual FPN_IN_CH in context.md Section 3.5.

VERIFY     : verify_backbone_shapes() prints "Backbone shapes verified: [(1,X,28,28), ...]"
             without raising an AssertionError.
```

---

### T-015: Severity generation uses random labels (phase1_best.pt not found)

```
SYMPTOM    : 02_generate_severity.py prints:
             "Phase 1 model not found. Using random severity labels as fallback."
             And writes random 0/1/2 values to severity_labels.csv.

CAUSE      : The severity proxy is generated before Phase 1 training completes,
             or Phase 1 training failed and phase1_best.pt was not saved.

EXACT FIX  :
  Step 1 — Check if phase1_best.pt exists:
             dir models\checkpoints\phase1_best.pt
  Step 2 — If it does NOT exist: Run Phase 1 first (Step 08), then re-run
             Step 06 (02_generate_severity.py).
  Step 3 — If it DOES exist but severity ran with random fallback anyway:
             Delete severity_labels.csv and re-run 02_generate_severity.py.
             del data\metadata\severity_labels.csv
             python training/02_generate_severity.py

  NOTE: Using random severity labels for training is acceptable for the first
  training run since severity labels are proxy labels (not ground truth).
  Random labels will cause the severity head to perform at chance (~33%).
  After Phase 1 completes, re-generate severity labels using the trained model
  for better proxy label quality before Phase 2.

RECORD IN context.md: Whether severity was generated with real model or random fallback.
```

---

### T-016: source_map.csv has 0 records for a dataset

```
SYMPTOM    : After 01_prepare_data.py, class_counts.csv shows 0 images for
             a dataset that was downloaded successfully.

CAUSE A    : The dataset folder structure does not match what _scan_source expects.
CAUSE B    : All label strings from that dataset are unmapped (filtered out as unknown).
CAUSE C    : The download landed in the wrong directory.

DIAGNOSE   :
  python -c "
  import os
  src = 'data/raw/sabbir_okra'  # replace with failing dataset
  count = sum(1 for r,d,f in os.walk(src) for fn in f if fn.lower().endswith('.jpg'))
  print(f'Images found in {src}: {count}')
  for root, dirs, files in os.walk(src):
      if files: print(root, ':', len(files))
      if root.count(os.sep) > src.count(os.sep) + 3: break
  "

EXACT FIX  :
  If images exist but 0 were scanned → folder structure issue → see T-011.
  If images exist and folder structure is correct → label mapping issue → see T-012.
  If 0 images in the raw directory → download failed → re-run Step 03.
```

---

## SECTION 5: TRAINING FAILURES

---

### T-017: Phase 1 — "Training cache not found"

```
SYMPTOM    : 04_train_phase1.py raises FileNotFoundError:
             "Training cache not found at cache/train_features.pt.
              Run training/03_cache_features.py first."

CAUSE      : Step 07 (03_cache_features.py) was not run, or it failed partway through.

EXACT FIX  :
  Step 1 — Check cache directory: dir cache\
  Step 2 — If cache files are missing or 0 bytes: Re-run Step 07:
             python training/03_cache_features.py
  Step 3 — If 03_cache_features.py fails, check:
             - Does source_map.csv exist? (Step 05 must complete first)
             - Is phase1_best.pt needed? (No — 03_cache_features.py uses
               a fresh model for feature extraction, not a trained one)
             - Is there enough disk space? Cache is ~400-500 MB total.
               python -c "import shutil; print(shutil.disk_usage('.').free // 1e9, 'GB free')"
  Step 4 — After successful caching, re-run Phase 1.

VERIFY     : dir cache\ should show train_features.pt and val_features.pt,
             each at least 50 MB.
```

---

### T-018: Training loss is NaN or Inf from the start

```
SYMPTOM    : Phase 1 or Phase 2 training prints loss=nan or loss=inf from the
             first batch. Training immediately stops being useful.

CAUSE A    : pos_weight contains NaN or Inf (division by zero in class weight computation).
CAUSE B    : Input images contain NaN values (corrupted images loaded as 0/255 but
             normalisation produces NaN).
CAUSE C    : Learning rate is too high (only for Phase 2 — Phase 1 uses cached features
             which cannot cause NaN in weights).

DIAGNOSE   :
  After one batch failure, run:
  python -c "
  import torch
  cache = torch.load('cache/train_features.pt')
  p = cache['pooled_features']
  print('NaN in features:', torch.isnan(p).any())
  print('Inf in features:', torch.isinf(p).any())
  print('Feature range:', p.min().item(), p.max().item())
  "

EXACT FIX FOR CAUSE A (NaN pos_weight):
  Step 1 — Check class counts: look at class_counts.csv in data/metadata/
  Step 2 — If any class has count=0, it will produce pos_weight=inf (n_total/0).
  Step 3 — The .clamp(min=1.0) in compute_multilabel_pos_weights should prevent this,
             but verify it is present in training/metrics.py.
  Step 4 — If a class has 0 training images, that is a data problem — download more
             data or add synthetic images for that class.

EXACT FIX FOR CAUSE B (corrupted images):
  Step 1 — The try/except in PlantDiseaseDataset.__getitem__ returns a zero tensor
             on load failure. Verify this is in training/dataset.py.
  Step 2 — Check normalisation: after apply_clahe and Normalize, values should be
             roughly in [-2.5, 2.5]. A zero image gives [-2.12, -1.80, -1.65] after
             ImageNet normalisation, which is valid (not NaN).
  Step 3 — If NaN appears in features, delete the feature cache and rebuild it.

EXACT FIX FOR CAUSE C (LR too high — Phase 2):
  Step 1 — Reduce PHASE2_BASE_LR in app/config.py from 1e-4 to 5e-5.
  Step 2 — Delete phase2 checkpoints and restart Phase 2.

VERIFY     : Loss should be finite (typically 2.0-3.5) at batch 1 for a fresh model.
```

---

### T-019: Training loss converges but macro F1 stays near 0

```
SYMPTOM    : Phase 1 or Phase 2 training shows decreasing loss but val/macro_f1
             stays at 0.05-0.10 through all epochs. Early stopping triggers.

CAUSE A    : Label mapping error — most images are assigned the wrong class,
             so the model learns to predict incorrect classes.
CAUSE B    : The model is predicting all negatives (never predicts any disease above
             threshold), which gives F1=0 even with low loss.
CAUSE C    : pos_weight is near zero for all classes (the bug where n_total was
             computed as number of positive labels, not number of images — fixed
             in v6 but verify it is correctly implemented).

DIAGNOSE   :
  Step 1 — After a few epochs, check what the model is predicting:
  python -c "
  import torch
  from app.config import DEVICE, TRAIN_CACHE, DISEASE_THRESH
  cache = torch.load(TRAIN_CACHE, weights_only=False)
  from app.model import PlantDiseaseModel
  from training.helpers import load_checkpoint
  import os
  ckpt_path = 'models/checkpoints/phase1_best.pt'
  if os.path.exists(ckpt_path):
      model = PlantDiseaseModel().to(DEVICE)
      load_checkpoint(model, None, None, None, ckpt_path, DEVICE)
      model.eval()
      pooled = cache['pooled_features'][:32].to(DEVICE)
      with torch.no_grad():
          c_log, _ = model.crop_classifier(pooled)
          d_log = model.disease_head(pooled, model.crop_classifier(pooled)[1])
          d_probs = torch.sigmoid(d_log)
      print('Disease prob range:', d_probs.min().item(), d_probs.max().item())
      print('Any above threshold:', (d_probs > DISEASE_THRESH).any().item())
  "

EXACT FIX FOR CAUSE B (always predicts negatives):
  Step 1 — Check pos_weight values: they should be > 1 for all classes.
  Step 2 — Verify compute_multilabel_pos_weights in training/metrics.py uses
             n_total = float(len(train_records)) (number of images), NOT the
             sum of positive labels. See DECISION-15.
  Step 3 — If pos_weight is correct but predictions are still all negative:
             Increase MAX_POS_WEIGHT from 10.0 to 15.0 as a temporary measure.
  Step 4 — Check DISEASE_THRESH — if it is accidentally set to 0.99, almost
             nothing will pass. Verify it is 0.50 in app/config.py.

EXACT FIX FOR CAUSE A (label mapping error):
  Step 1 — Sample 20 records from source_map.csv and verify class_name values
             make sense for the source_dataset.
  Step 2 — Look for cases where brassica labels were mapped to okra classes or
             vice versa — these would be in SOURCE_LABEL_OVERRIDES.
  Step 3 — Correct the mapping in app/config.py and regenerate source_map.csv.
```

---

### T-020: Phase 2 does not improve over Phase 1 (F1 does not increase)

```
SYMPTOM    : Phase 2 macro F1 after epoch 0 equals or is less than Phase 1 best F1.
             No improvement even after several epochs.

CAUSE A    : Backbone blocks were not actually unfrozen.
CAUSE B    : LLRD optimizer has wrong learning rates (all near zero).
CAUSE C    : OneCycleLR scheduler parameters are wrong.

DIAGNOSE   :
  Step 1 — Verify backbone unfreezing worked:
  python -c "
  from app.model import PlantDiseaseModel
  import torch
  m = PlantDiseaseModel()
  m.freeze_backbone()
  m.unfreeze_top_fraction(0.33)
  trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
  frozen    = sum(p.numel() for p in m.parameters() if not p.requires_grad)
  print(f'Trainable: {trainable:,}  Frozen: {frozen:,}')
  "
  Trainable should be > 1,000,000 (the unfrozen backbone blocks + heads).
  If Trainable ≈ 500,000 (heads only), unfreezing did not work.

  Step 2 — If _get_backbone_blocks() raised RuntimeError instead of returning empty:
             The fix in DECISION-34/T-UNK should have caught this. Check the
             RuntimeError message for which backbone attribute was not found.

EXACT FIX FOR CAUSE A:
  Step 1 — Verify _get_backbone_blocks() returns a non-empty list:
  python -c "
  from app.model import PlantDiseaseModel
  m = PlantDiseaseModel()
  blocks = m._get_backbone_blocks()
  print(f'Backbone blocks found: {len(blocks)}')
  "
  Step 2 — If 0 blocks: check timm version. EfficientNetV2-S in timm >= 0.9 exposes
             blocks at backbone.model.blocks. Run:
  python -c "import timm; print(timm.__version__)"
             Expected: 0.9.x
  Step 3 — If timm version is correct but blocks still not found, inspect manually:
  python -c "
  from app.model import PlantDiseaseModel
  m = PlantDiseaseModel()
  print(dir(m.backbone))
  print(dir(m.backbone.model))
  "
             Find where the block list lives and update _get_backbone_blocks() accordingly.

EXACT FIX FOR CAUSE C (wrong OneCycleLR params):
  Step 1 — Verify ONE_CYCLE_PCT, ONE_CYCLE_DIV, ONE_CYCLE_FDIV are imported from
             app.config (not hardcoded). If hardcoded values are wrong, update config.
```

---

### T-021: Phase 2 training crashes mid-epoch (not OOM)

```
SYMPTOM    : Phase 2 crashes with a non-OOM error partway through an epoch.
             Examples: "RuntimeError: element 0 of tensors does not require grad",
             "RuntimeError: grad can be implicitly created only for scalar outputs",
             "ValueError: optimizer got an empty parameter list"

CAUSE A    : "grad can be implicitly created" — scaled_loss.backward() is called
             outside the autocast context, or scaler.scale() is not used.
CAUSE B    : "empty parameter list" — LLRD optimizer received a parameter group
             with no parameters (can happen if _get_backbone_blocks() returns
             an unexpected structure).
CAUSE C    : get_llrd_optimizer is defined in the wrong file and imported incorrectly.

EXACT FIX FOR CAUSE A:
  Step 1 — Verify the training loop structure:
             with autocast(device_type='cuda', enabled=use_amp):
                 c_log, d_log, s_log = model(images)
                 total_loss, _ = compute_loss(...)
                 scaled_loss = total_loss / GRAD_ACCUM_STEPS
             scaler.scale(scaled_loss).backward()  ← must be OUTSIDE autocast
  Step 2 — The backward() must be on scaler.scale(loss), not on loss directly.

EXACT FIX FOR CAUSE B:
  Step 1 — Check get_llrd_optimizer in training/helpers.py.
             After blocks = model._get_backbone_blocks(), it should raise RuntimeError
             if blocks is empty (DECISION-34). If this was not implemented, add:
             if not blocks: raise RuntimeError("backbone blocks not found")
  Step 2 — Filter out any param groups with empty params before passing to AdamW:
             param_groups = [pg for pg in param_groups if len(pg['params']) > 0]

EXACT FIX FOR CAUSE C:
  Step 1 — Verify get_llrd_optimizer is imported from training.helpers at the top
             of 05_train_phase2.py, not defined locally.
             Check: grep "def get_llrd_optimizer" training/05_train_phase2.py
             Should return nothing (not defined there).
```

---

### T-022: Gradient clipping does not work (grad_norm is huge every batch)

```
SYMPTOM    : wandb shows train/grad_norm consistently > 100 even after unscaling.
             Loss may become unstable or NaN after many steps.

CAUSE      : scaler.unscale_(optimizer) is not being called before clip_grad_norm_.
             Without unscaling, gradients are at FP16 scale (~65536× the true value)
             and clip_grad_norm_(model.parameters(), 1.0) effectively clips at
             65536.0, which is meaningless.

EXACT FIX  :
  Step 1 — Verify the exact order in the Phase 2 training loop:
             scaler.unscale_(optimizer)          ← MUST be first
             grad_norm = clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
             scaler.step(optimizer)
             scaler.update()
  Step 2 — If the order is wrong, fix it. This is a correctness requirement.
  Step 3 — Expected grad_norm after fixing: typically 0.1 to 3.0 per batch.
             Values consistently > 10 suggest LR is too high.
             Values consistently < 0.01 suggest LR is too low or model is not learning.

VERIFY     : After fixing, check wandb train/grad_norm. Should be in range 0.1-5.0.
```

---

### T-023: Early stopping triggers at epoch 0 or 1

```
SYMPTOM    : Phase 1 or Phase 2 early stopping triggers immediately.
             Prints "Early stopping at epoch 0" or "Early stopping at epoch 1".

CAUSE A    : val/macro_f1 is 0.0 at epoch 0, then 0.0001 at epoch 1 — technically
             improvement, but patience counter is wrong because the first call
             sets best_score=None and immediately returns False.
             Then the second call: score=0.0001 > 0.0 + 0.001 = False (no improvement).
             This is expected behaviour — 0.0001 does not exceed 0.001 delta.
CAUSE B    : The EarlyStopping instance is being recreated every epoch (loses state).
CAUSE C    : val/macro_f1 is genuinely not improving — see T-019.

DIAGNOSE   :
  Step 1 — Check EarlyStopping instantiation:
             early_stop = EarlyStopping(EARLY_STOP_PAT, EARLY_STOP_DELTA)
             This must be OUTSIDE the epoch loop, not inside it.
  Step 2 — Check that early_stop is not reset to None inside the loop.

EXACT FIX FOR CAUSE B:
  Move early_stop = EarlyStopping(...) to before the for epoch in range(...): loop.
  It must persist across epochs to track consecutive non-improvement.

EXACT FIX FOR CAUSE A:
  The first few epochs of Phase 1 often have F1 near 0 while the model calibrates.
  Wait until epoch 3-5 before worrying about slow improvement.
  If F1 is still 0 at epoch 5, see T-019.
```

---

## SECTION 6: INFERENCE AND SERVER FAILURES

---

### T-024: Grad-CAM returns empty or all-black heatmap

```
SYMPTOM    : The heatmap_b64 in the inference response is empty string, or
             the rendered heatmap image is entirely black or entirely one colour.

CAUSE A    : Wrong target layer — model.fpn.output_p3 used instead of model.fpn.out_p3
             (AttributeError silently caught, heatmap_b64 set to '').
CAUSE B    : Model was torch.compiled — pytorch_grad_cam hooks don't work on
             compiled models. Inference should always use an uncompiled model.
CAUSE C    : Model is in MC Dropout mode (Dropout in train mode) when Grad-CAM runs.
             Stochastic activations produce inconsistent gradients and poor heatmaps.

EXACT FIX FOR CAUSE A:
  Step 1 — Search inference.py for output_p3. Must not exist.
  Step 2 — Verify: target_layer = model.fpn.out_p3 (not output_p3).
  Step 3 — Test directly:
  python -c "
  from app.model import load_model_for_inference
  from app.config import DEVICE, BEST_MODEL
  m = load_model_for_inference(BEST_MODEL, DEVICE)
  print(m.fpn.out_p3)  # should print Conv2d layer info, not AttributeError
  "

EXACT FIX FOR CAUSE B:
  Step 1 — Verify load_model_for_inference() in app/model.py does NOT call
             torch.compile(). It should load state_dict into a fresh PlantDiseaseModel().
  Step 2 — The compiled model is used ONLY during training. The saved best_model.pt
             contains the raw state_dict (via getattr(model, '_orig_mod', model).state_dict()).
             Verify this is how best_model.pt was saved in 05_train_phase2.py.

EXACT FIX FOR CAUSE C:
  Step 1 — Verify the MC Dropout / Grad-CAM sequence in run_inference():
             1. MC passes: model.eval() → set Dropout to train → run MC_PASSES passes
             2. After MC passes: model.eval() → Grad-CAM
             Grad-CAM MUST run after model.eval() is restored, NOT during MC passes.
  Step 2 — Check that model.eval() is called after the MC loop, before generate_heatmap().
```

---

### T-025: FastAPI server fails to start — "Address already in use"

```
SYMPTOM    : Running uvicorn raises:
             "ERROR: [Errno 10048] error while attempting to bind on address
              ('0.0.0.0', 8000): only one usage of each socket address..."

CAUSE      : Another process is already listening on port 8000 (or 8765 for tests).

EXACT FIX  :
  Step 1 — Find and kill the process using port 8000:
             netstat -ano | findstr :8000
             (Note the PID in the last column)
             taskkill /PID <PID> /F
  Step 2 — Or use a different port:
             uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
  Step 3 — Update CLAUDE.md reference if permanently changing the port.

  For test server (port 8765):
             netstat -ano | findstr :8765
             taskkill /PID <PID> /F
```

---

### T-026: Server returns 500 on /predict — "best_model.pt not found"

```
SYMPTOM    : POST /predict returns HTTP 500 with:
             "Inference failed: [Errno 2] No such file or directory: 'models/best_model.pt'"

CAUSE      : Training has not completed, or best_model.pt was not saved correctly.

EXACT FIX  :
  Step 1 — Check: dir models\best_model.pt
  Step 2 — If file does not exist: Phase 2 training must complete successfully first.
  Step 3 — If file exists but is 0 bytes: Phase 2 was interrupted before save.
             Delete it and re-run Phase 2 from the latest phase2_best.pt checkpoint.
  Step 4 — Verify the server loads the model at startup (in the lifespan function)
             and stores it in app.state.model. It should not reload per request.

VERIFY     : GET /health returns {"status": "ok", "device": "cuda"} (or "cpu").
             The model loads at startup — if health check passes, the model is loaded.
```

---

### T-027: "/predict returns 400 — Image too blurry" for a clear image

```
SYMPTOM    : A clearly in-focus image is rejected with:
             "Image too blurry (score X.X < 80). Take a clearer photo."
             But the image looks sharp to the human eye.

CAUSE A    : The image is a screenshot or digitally generated (no natural texture
             grain) — Laplacian variance is genuinely low for synthetic images.
CAUSE B    : The image was resized heavily before upload (downsampling removes
             high-frequency content, reducing Laplacian variance).
CAUSE C    : MIN_BLUR_VAR is too aggressive for the specific image type.

EXACT FIX FOR CAUSE A (testing with synthetic/screenshot images):
  This is correct behaviour. The smoke test uses a noise-overlay image specifically
  because synthetic solid-colour images have zero Laplacian variance. For production
  testing, use a real photograph.

EXACT FIX FOR CAUSE B (user uploads heavily resized image):
  This is expected — the validation is working correctly. Real field photos from
  modern smartphones have variance >> 80. If a real field photo fails, the farmer
  should retake the photo with better focus.

EXACT FIX FOR CAUSE C (threshold too strict):
  Step 1 — Check the actual variance of the failing image:
  python -c "
  import cv2, numpy as np
  img = cv2.imread('path/to/image.jpg', cv2.IMREAD_GRAYSCALE)
  print('Laplacian variance:', cv2.Laplacian(img, cv2.CV_64F).var())
  "
  Step 2 — If the image is genuinely a sharp field photo with variance = 60-80,
             consider reducing MIN_BLUR_VAR from 80 to 50 in app/config.py.
             Document this change in decisions.md.
  Step 3 — Do NOT reduce MIN_BLUR_VAR below 30 — this would allow genuinely blurry
             images that produce poor Grad-CAM heatmaps and unreliable predictions.
```

---

### T-028: Inference returns wrong crop prediction (okra vs brassica)

```
SYMPTOM    : An okra leaf image returns crop='brassica', or vice versa.
             crop_confidence may be high (> 0.85) or low (< 0.60).

CAUSE A    : If low confidence (< 0.60): OOD_FLAGGED=True should be set.
             The model is uncertain — this may be a genuinely ambiguous image.
CAUSE B    : If high confidence (> 0.85): The model has learned a systematic
             misclassification, likely from label errors in training data where
             some okra images were labelled as brassica or vice versa.
CAUSE C    : Crop classifier was trained with very imbalanced crop representation
             in the training set.

DIAGNOSE   :
  Step 1 — Check class distribution in training split:
  python -c "
  import pandas as pd
  from app.config import SOURCE_MAP, CROP_FROM_IDX
  df = pd.read_csv(SOURCE_MAP)
  train = df[df['split']=='train']
  train['crop'] = train['class_idx'].map(CROP_FROM_IDX)
  print(train['crop'].value_counts())
  "
  Expect roughly 50/50 okra (0) vs brassica (1). Major imbalance → CAUSE C.

EXACT FIX FOR CAUSE B:
  Step 1 — Manually inspect 50 source_map.csv records from the incorrectly classified source.
  Step 2 — Look for label mapping errors (brassica labels mapped to okra idx or vice versa).
  Step 3 — Fix the mapping in app/config.py SOURCE_LABEL_OVERRIDES.
  Step 4 — Regenerate source_map.csv and retrain.
```

---

## SECTION 7: EVALUATION FAILURES

---

### T-029: Tier-2 evaluation — no PlantDoc records in source_map.csv

```
SYMPTOM    : 08_evaluate_tier2_plantdoc.py prints:
             "No PlantDoc records in source_map.csv. Run 01_prepare_data.py after downloading PlantDoc."

CAUSE      : PlantDoc was not downloaded before 01_prepare_data.py ran, or
             01_prepare_data.py was run before the PlantDoc download completed.

EXACT FIX  :
  Step 1 — Verify PlantDoc exists: dir data\plantdoc
  Step 2 — If empty or missing: re-run download_plantdoc.py (see T-013).
  Step 3 — After PlantDoc is in place, re-run 01_prepare_data.py:
             python training/01_prepare_data.py
             This will add plantdoc records with split='plantdoc' to source_map.csv.
  Step 4 — Re-run 08_evaluate_tier2_plantdoc.py.

RECORD IN context.md: Actual PlantDoc record count after fix.
```

---

### T-030: Tier-2 macro F1 < 0.55 (FAIL)

```
SYMPTOM    : 08_evaluate_tier2_plantdoc.py writes a FAIL report.
             One or more mappable PlantDoc classes have F1 < 0.40.

CAUSE      : This is not necessarily a bug — it is a performance gap. Common causes:
CAUSE A    : Domain shift — PlantDoc images have different lighting/background than training.
CAUSE B    : The specific failing class has very few training examples from diverse sources.
CAUSE C    : Temperature scaling was not applied (T_disease=1.0 leaves logits uncalibrated).

DIAGNOSE   :
  Step 1 — Check the tier-2 report for which classes are failing.
  Step 2 — Check if temperature.pt was loaded:
             python -c "import torch; t=torch.load('models/temperature.pt',weights_only=False); print(t)"
  Step 3 — Check training class counts for the failing classes from class_counts.csv.

ACTIONS (these must be documented in decisions.md before taking):
  Action A — Add more diverse training data for the failing class (new Kaggle search).
  Action B — Adjust CLAHE parameters to better handle PlantDoc lighting conditions.
  Action C — If macro F1 is 0.45-0.55 and only 1-2 classes fail: write gap analysis
              and proceed to deployment with known limitations documented.
  Action D — Lower TIER2_MIN_F1 threshold (requires strong justification and
              documentation in decisions.md).

IMPORTANT: After tier-2 runs, record the result in context.md Section 5.2 and
           set the lock status to LOCKED. No model changes are permitted after
           the first tier-2 evaluation run.
```

---

## SECTION 8: GIT AND GITHUB FAILURES

---

### T-031: git push fails — "Authentication failed"

```
SYMPTOM    : After a pipeline step, git push returns:
             "remote: Support for password authentication was removed"
             or
             "fatal: Authentication failed for 'https://github.com/...'"

CAUSE      : GITHUB_TOKEN in .env is expired, wrong, or not being passed to the
             git remote URL.

EXACT FIX  :
  Step 1 — Generate a new GitHub Personal Access Token:
             github.com → Settings → Developer settings → Personal access tokens
             → Generate new token (classic) → select "repo" scope → generate.
  Step 2 — Update .env: GITHUB_TOKEN=new_token_value
  Step 3 — Update the git remote URL with the new token:
             git remote set-url origin https://<TOKEN>@github.com/<USER>/<REPO>.git
             Replace <TOKEN>, <USER>, <REPO> with actual values.
  Step 4 — Test: git push
  Step 5 — The pipeline logs push failures to pipeline_failures.log but continues —
             a failed push does NOT stop the pipeline.

PREVENTION : GitHub tokens can be set to never expire. Use a token with no expiry
             for automated pipeline use.
```

---

### T-032: Large files blocked by GitHub — "File exceeds 100MB limit"

```
SYMPTOM    : git push is rejected with:
             "remote: error: File models/best_model.pt is X MB; this exceeds GitHub's file size limit of 100 MB"

CAUSE      : A model weight file, cache file, or dataset was accidentally staged for commit.
             .gitignore should prevent this but may have been misconfigured.

EXACT FIX  :
  Step 1 — Unstage the large file:
             git rm --cached models/best_model.pt
             (or whatever large file was staged)
  Step 2 — Add the file pattern to .gitignore if not already there:
             echo "models/*.pt" >> .gitignore
             echo "cache/" >> .gitignore
             echo "data/" >> .gitignore
  Step 3 — If the file was already committed (not just staged), you need to remove it
             from git history. This is complex — ask for help before proceeding.
             The simplest fix for a new project: delete the repo and start fresh with
             correct .gitignore.
  Step 4 — After fixing, commit and push again.

VERIFY     : git status should show no large files staged.
```

---

## SECTION 9: WANDB FAILURES

---

### T-033: wandb.init() blocks waiting for API key input

```
SYMPTOM    : Training script hangs at wandb.init() waiting for user input.
             Or: "wandb: ERROR You are not authenticated. Please run 'wandb login'"

CAUSE      : WANDB_API_KEY is not set in the environment and WANDB_MODE was not
             set to 'offline' as a fallback.

EXACT FIX  :
  Option A — Set WANDB_API_KEY in .env (preferred):
             WANDB_API_KEY=your_key_from_wandb.ai/settings
  Option B — Run wandb offline and sync later:
             Set in .env: WANDB_MODE=offline
             Or in terminal before running: set WANDB_MODE=offline
  Option C — Disable wandb entirely for one run:
             set WANDB_MODE=disabled

  The pipeline already sets WANDB_MODE=offline as a fallback if WANDB_API_KEY
  is not set. If the script still hangs, check that load_dotenv() runs before
  wandb.init() in the training script.

VERIFY     : Run: python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.environ.get('WANDB_MODE', 'not set'))"
             Should print 'offline' if WANDB_API_KEY is missing.
```

---

### T-034: wandb run shows no config parameters (empty Config tab)

```
SYMPTOM    : The wandb run shows empty Config section, making it impossible to
             compare runs by hyperparameters.

CAUSE      : wandb.init() was called without the config parameter, or only with
             phase name but not WANDB_CONFIG.

EXACT FIX  :
  Step 1 — Verify wandb.init() call includes **WANDB_CONFIG:
             wandb.init(
                 project=WANDB_PROJECT,
                 name='phase1',
                 config={**WANDB_CONFIG, 'phase': 1, 'epochs': PHASE1_EPOCHS}
             )
  Step 2 — WANDB_CONFIG is defined in app/config.py with all hyperparameters.
             Verify it is imported at the top of the training script.
  Step 3 — For the calibration script: config must include T_disease, T_crop,
             T_severity after fitting:
             wandb.log({'T_disease': T_disease, 'T_crop': T_crop, 'T_severity': T_severity})
```

---

## SECTION 10: WINDOWS-SPECIFIC ISSUES

---

### T-035: PowerShell execution policy blocks venv activation

```
SYMPTOM    : Running venv\Scripts\Activate.ps1 in PowerShell returns:
             "cannot be loaded because running scripts is disabled on this system"

CAUSE      : Windows PowerShell has script execution restricted by default.

EXACT FIX  :
  Option A — Change policy for current user (recommended):
             Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
             Then re-run: venv\Scripts\Activate.ps1
  Option B — Use Command Prompt instead of PowerShell:
             venv\Scripts\activate.bat
             This always works without policy changes.
  Option C — Use Git Bash:
             source venv/Scripts/activate

PREVENTION : Use CMD (Command Prompt) for all project work. It is more reliable
             on Windows for Python development than PowerShell.
```

---

### T-036: "PermissionError: [WinError 32] file is being used by another process"

```
SYMPTOM    : Training script crashes with PermissionError when trying to write or
             delete a checkpoint file.
             Or: 01_prepare_data.py cannot write source_map.csv.

CAUSE A    : A zombie DataLoader worker process (from persistent_workers=True or
             a previous crash) is holding a file lock on the data directory.
CAUSE B    : Another Python process (e.g. Jupyter notebook) has the file open.
CAUSE C    : Windows Defender is scanning the newly written file.

EXACT FIX FOR CAUSE A:
  Step 1 — Open Task Manager (Ctrl+Shift+Esc).
  Step 2 — Go to Details tab. Sort by Name. Find all python.exe processes.
  Step 3 — Kill any python.exe processes that are not the current terminal session.
  Step 4 — Re-run the script.

  Or from terminal:
  tasklist | findstr python
  taskkill /IM python.exe /F   ← WARNING: kills ALL python processes including this terminal

EXACT FIX FOR CAUSE C (Windows Defender):
  Step 1 — Add the project root to Windows Defender exclusions:
             Windows Security → Virus & threat protection → Manage settings
             → Add or remove exclusions → Add a folder → select project root
  Step 2 — This prevents Defender from holding write locks on newly created files.
```

---

### T-037: Long path errors — "FileNotFoundError: path too long"

```
SYMPTOM    : Kaggle download or file operations fail with:
             "FileNotFoundError: [WinError 3] The system cannot find the path specified"
             even though the path looks correct.

CAUSE      : Windows has a default MAX_PATH limit of 260 characters. Deep Kaggle
             dataset folder hierarchies can exceed this when combined with the
             project path.

EXACT FIX  :
  Step 1 — Enable long paths in Windows:
             Open Group Policy Editor (gpedit.msc) or run in PowerShell as admin:
             New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
               -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  Step 2 — Also enable in Python (Python 3.6+ handles this automatically, but verify):
             python -c "import os; print(os.stat('.'))"
  Step 3 — Restart the terminal after enabling long paths.
  Step 4 — If the project is in a deeply nested directory (e.g. C:\Users\username\
             Documents\Projects\ML\Semester4\Assignments\...), move it closer to root:
             C:\projects\plant-disease\ (shorter path = less risk)
```

---

## SECTION 11: PERFORMANCE ISSUES

---

### T-038: Feature caching takes longer than expected (> 60 minutes)

```
SYMPTOM    : 03_cache_features.py runs for more than 60 minutes for the train split.
             Expected time: 20-30 minutes for ~10,000 images.

CAUSE A    : num_workers > 0 is set on Windows, causing worker spawn overhead.
CAUSE B    : CUDA is not available — caching runs on CPU at 10× slower speed.
CAUSE C    : The dataset is larger than expected (> 15,000 images).

EXACT FIX FOR CAUSE A:
  Verify 03_cache_features.py uses num_workers=0 in the DataLoader.
  The comment in the spec: "num_workers=0: caching is a one-time pass, no benefit from workers"

EXACT FIX FOR CAUSE B:
  Verify CUDA is available: python -c "import torch; print(torch.cuda.is_available())"
  If False: fix CUDA first (see T-005).
  CPU caching of 10,000 images at 224×224 takes approximately 45-90 minutes.
  GPU caching takes approximately 15-25 minutes.

FOR CAUSE C:
  If the actual dataset is larger (> 15,000 images), caching time scales linearly.
  20,000 images ≈ 35-45 minutes on GPU. This is expected — no fix needed.
  Record the actual dataset size and caching time in context.md.
```

---

### T-039: Phase 2 training is slower than 3.5 hours

```
SYMPTOM    : Phase 2 training is projected to take > 5 hours based on per-epoch time.

CAUSE A    : torch.compile is disabled (no MSVC Build Tools) — adds ~25-35% time.
CAUSE B    : Mixed precision (AMP) is disabled — adds ~30-50% time.
CAUSE C    : BATCH_SIZE was reduced to 16 due to OOM — roughly 2× the per-epoch time.
CAUSE D    : The dataset is larger than the ~10,000 images estimated.

DIAGNOSIS  : Check the Phase 2 training output for:
             "torch.compile enabled" (CAUSE A if missing)
             "use_amp: True" (CAUSE B if False)
             The BATCH_SIZE in app/config.py (CAUSE C if 16)

EXACT FIX FOR CAUSE A:
  Install MSVC Build Tools (see T-007). If not possible, accept the slower time.
  Record in context.md that compile was disabled.

FOR CAUSE B:
  Verify DEVICE is 'cuda' (not 'cpu'). AMP is disabled on CPU because it provides
  no benefit. If DEVICE='cpu', fix CUDA first (T-005).

FOR ALL CAUSES:
  Record the actual Phase 2 training time in context.md Section 4.2.
  If training exceeds 8 hours, consider reducing PHASE2_EPOCHS from 7 to 5.
  Update PHASE2_EPOCHS in app/config.py and document in decisions.md.
```

---

## SECTION 12: NEW ISSUES DISCOVERED DURING IMPLEMENTATION

[Claude Code appends new troubleshooting entries here as issues are discovered
during the project. Use the same format as above.]

```
### T-040: [TITLE]

SYMPTOM    :
CAUSE      :
DIAGNOSE   :
EXACT FIX  :
VERIFY     :
PREVENTION :
```
