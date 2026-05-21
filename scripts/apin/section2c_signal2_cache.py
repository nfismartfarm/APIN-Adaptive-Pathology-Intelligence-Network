"""
Section 2C -- Signal 2 Prediction Cache (EfficientNet 10-class).

The 10-class EfficientNet from old_10class/ — the SACRED production model
at models/best_model.pt. Different architecture (EfficientNetV2-S + FPN),
different training data version, different preprocessing (RGB per-channel
CLAHE, NOT LAB-CLAHE). This independence is what makes it valuable as
Signal 2 in the APIN ensemble.

Critical preprocessing decisions documented in
architecture_claude_decisions.md Decision 12:
  - Import apply_clahe DIRECTLY from old_10class/app/inference.py — do NOT
    reimplement. It uses cv2.createCLAHE(clip_limit=2.0, tileGridSize=(8,8))
    applied per R, G, B channel independently.
  - RESOLUTION: 224x224 (EfficientNetV2-S training resolution, asserted by
    verify_backbone_shapes in old_10class/app/model.py line 160).

MC Dropout decision (checked in Step 0 prep):
  - Production inference in old_10class/app/inference.py DID use MC Dropout
    (5 passes with Dropout in train() mode, BatchNorm in eval()).
  - For Signal 2 cache we use SINGLE-PASS eval() mode — deterministic,
    reproducible, 5x faster. The stacking MLP will learn what single-pass
    EfficientNet sigmoid values mean in its training data.
  - Documented in fingerprint JSON.

Output format per CSV row index:
  {
    'predictions': np.array(9,) float32,  # RAW sigmoid (NOT renormalized)
    'class_name': str, 'source_dataset': str, 'is_field_photo': bool,
    'split': str, 'is_recomposed': bool, 'true_class_idx': int,
    'inference_success': bool,
    'raw_en_10class_predictions': np.array(10,) float32,  # before reorder
                                                         # for audit
  }

The 9-class reordering: EN's class_names[8] is brassica_clubroot (quarantined
in Model 2). We take indices [0,1,2,3,4,5,6,7,9] → Model 2 ordering.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Hide timm deprecation noise from cluttering the log
warnings.filterwarnings("ignore", category=UserWarning, module="timm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section2c_signal2_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section2c")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(fmt)
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

# Paths
EN_CKPT = PROJECT_ROOT / "models" / "best_model.pt"  # SACRED — DO NOT MODIFY
CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "model2_unified_source_map.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "split_indices.json"
OUTPUT_CACHE = CACHE_DIR / f"signal2_predictions_cache_{TIMESTAMP}.pkl"
OUTPUT_CACHE_LATEST = CACHE_DIR / "signal2_predictions_cache.pkl"
OUTPUT_FINGERPRINT = CACHE_DIR / f"signal2_predictions_fingerprint_{TIMESTAMP}.json"

# Constants — EfficientNetV2-S trained at 224x224 per old_10class verify_backbone_shapes
IMG_SIZE = 224
BATCH_SIZE = 64  # smaller backbone, smaller image — can go larger than Signal 1
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Import apply_clahe DIRECTLY from old_10class — the sacred preprocessing function
# Never reimplement this. If this import breaks, fix the path, don't reinvent.
def _import_apply_clahe():
    """Import the canonical apply_clahe from old_10class/app/inference.py."""
    import importlib.util
    inference_path = PROJECT_ROOT / "old_10class" / "app" / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(f"Sacred file missing: {inference_path}")
    # Load module by path (old_10class isn't a regular package import target
    # because it has its own app/config.py that conflicts with the main app/).
    # We only need the apply_clahe function, which has no external dependencies
    # apart from cv2 + numpy.
    spec = importlib.util.spec_from_file_location("old_10class_inference", inference_path)
    # But loading this module fires imports that fail (it has relative imports
    # that don't work outside the old_10class package). Easier: extract the
    # function source directly.
    source = inference_path.read_text(encoding="utf-8")
    # Verify the exact signature + body we expect is still there
    import re
    match = re.search(
        r"def apply_clahe\(image: np\.ndarray, clip_limit=2\.0, tile_size=\(8, 8\)\)"
        r".*?return result",
        source, re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            "apply_clahe signature in old_10class/app/inference.py has changed. "
            "Signal 2 preprocessing must match training exactly. Investigate before "
            "proceeding."
        )
    # Execute just the function definition in an isolated namespace
    ns = {}
    import cv2 as _cv2
    import numpy as _np
    ns["cv2"] = _cv2
    ns["np"] = _np
    # Execute the matched function body
    exec(
        "import cv2\nimport numpy as np\n" + match.group(0),
        ns,
    )
    apply_clahe = ns["apply_clahe"]
    return apply_clahe, str(inference_path.relative_to(PROJECT_ROOT))


apply_clahe, APPLY_CLAHE_SOURCE = _import_apply_clahe()


class Model2ImageDataset(Dataset):
    """Loads raw images, applies RGB per-channel CLAHE via apply_clahe,
    resize 224, ImageNet normalise. Returns (tensor, row_idx, success_flag).

    This is preprocessing BRANCH B per APIN design — different from Branch A
    which uses LAB-CLAHE.
    """

    def __init__(self, df, raw_col="image_path"):
        self.df = df.reset_index(drop=True)
        self.raw_col = raw_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row[self.raw_col]
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img, dtype=np.uint8)
            # Apply RGB per-channel CLAHE (Branch B preprocessing)
            img_np = apply_clahe(img_np)
            # Resize to 224
            img_pil = Image.fromarray(img_np).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
            # ImageNet normalize
            img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
            # HWC -> CHW
            tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
            return tensor, int(i), 1
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32), int(i), 0


def load_efficientnet():
    """Load the 23-class 4-crop EfficientNetV2-S + FPN + FiLM model.

    CRITICAL FINDING: models/best_model.pt is NOT the 10-class model the
    prompt described. The checkpoint is the 23-class, 4-crop intermediate
    model (EfficientNetV2-S backbone + FPN + FiLM disease head):
      - 23 disease classes: okra (5) + brassica (5) + tomato (9) + chilli (4)
      - 4 crops: okra, brassica, tomato, chilli
      - val_metrics show val/macro_f1 = 0.8597 at epoch 6
    The first 10 classes (indices 0-9) are identical in ordering to the
    old 10-class model, so our EN_TO_M2_INDEX_MAP = [0,1,2,3,4,5,6,7,9]
    still selects the correct Model 2 classes. We drop index 8
    (brassica_clubroot, quarantined in Model 2) and indices 10-22 (tomato
    and chilli diseases, which are Model 3's domain, not Model 2's).

    This model is STILL a valid Signal 2 because:
      - Different architecture from Model 2 (EffNetV2-S vs DINOv3-ConvNeXt)
      - Different training regime (earlier, pre-specialist-pipeline)
      - Different head structure (FiLM-conditioned sigmoid vs direct linear
        softmax)
      - Independent failure modes

    We build a PlantDiseaseModel variant with NUM_CLASSES=23 and NUM_CROPS=4
    rather than using old_10class's 10-class definition.
    """
    logger.info("=" * 70)
    logger.info("Loading EfficientNet intermediate model (SACRED FILE -- do not modify)")
    logger.info("=" * 70)
    logger.info(
        "  DISCOVERY: models/best_model.pt is 23-class 4-crop, NOT 10-class.\n"
        "  The first 10 class indices (0-9) match the old 10-class ordering\n"
        "  exactly. We use indices [0,1,2,3,4,5,6,7,9] to map to Model 2's\n"
        "  9-class space (dropping index 8 = brassica_clubroot, and dropping\n"
        "  indices 10-22 = tomato/chilli which are Model 3 territory)."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Build an ad-hoc 23-class 4-crop PlantDiseaseModel using the old_10class
    # model.py architecture (EffNetV2-S backbone, FPN, FiLM disease head).
    # Import-hack the old_10class model.py with NUM_CLASSES=23 and NUM_CROPS=4.
    import importlib.util
    import sys as _sys

    model_src_path = PROJECT_ROOT / "old_10class" / "app" / "model.py"
    config_src_path = PROJECT_ROOT / "old_10class" / "app" / "config.py"

    # Save current 'app' module if loaded
    saved_app = _sys.modules.get("app")
    saved_app_config = _sys.modules.get("app.config")

    # Load old_10class/app/config as 'app.config' under a fresh 'app' package
    cfg_spec = importlib.util.spec_from_file_location("app.config", config_src_path)
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    app_mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("app", loader=None)
    )
    app_mod.__path__ = [str(model_src_path.parent)]
    _sys.modules["app"] = app_mod
    _sys.modules["app.config"] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)

    # OVERRIDE the 10-class constants with 23-class 4-crop values BEFORE
    # loading model.py (which reads these constants at class-definition time).
    cfg_mod.NUM_CLASSES = 23
    cfg_mod.NUM_CROPS = 4

    # Now load model.py — it will pick up the overridden NUM_CLASSES/NUM_CROPS
    model_spec = importlib.util.spec_from_file_location("app.model", model_src_path)
    model_mod = importlib.util.module_from_spec(model_spec)
    _sys.modules["app.model"] = model_mod
    model_spec.loader.exec_module(model_mod)

    PlantDiseaseModel = model_mod.PlantDiseaseModel

    # Restore caller's 'app' module
    if saved_app is not None:
        _sys.modules["app"] = saved_app
    if saved_app_config is not None:
        _sys.modules["app.config"] = saved_app_config

    logger.info("  Built PlantDiseaseModel with NUM_CLASSES=23, NUM_CROPS=4")

    # Load checkpoint
    ckpt = torch.load(EN_CKPT, map_location="cpu", weights_only=False)
    logger.info(f"  Checkpoint keys: {list(ckpt.keys())}")
    state = ckpt["model_state_dict"]

    # Build model and load state dict strictly
    model = PlantDiseaseModel()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"  Missing keys in state_dict load: {len(missing)}")
    if unexpected:
        logger.warning(f"  Unexpected keys in state_dict load: {len(unexpected)}")
    # Verify disease head + crop classifier loaded correctly
    disease_head_fc2_shape = model.disease_head.fc2.weight.shape
    assert disease_head_fc2_shape == (23, 256), (
        f"Expected disease_head.fc2 [23,256], got {disease_head_fc2_shape}"
    )
    crop_fc2_shape = model.crop_classifier.fc2.weight.shape
    assert crop_fc2_shape == (4, 64), (
        f"Expected crop_classifier.fc2 [4,64], got {crop_fc2_shape}"
    )
    logger.info(
        f"  Disease head shape: {disease_head_fc2_shape} (23 classes)"
    )
    logger.info(
        f"  Crop classifier shape: {crop_fc2_shape} (4 crops)"
    )

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total params: {total_params / 1e6:.2f}M")

    # Training val metrics from checkpoint
    val_metrics = ckpt.get("val_metrics", {})
    logger.info(f"  Training val macro F1: {val_metrics.get('val/macro_f1', 'unknown')}")
    logger.info(f"  Training val crop acc: {val_metrics.get('val/crop_acc', 'unknown')}")

    ckpt_meta = {
        "ckpt_path": str(EN_CKPT.relative_to(PROJECT_ROOT)),
        "ckpt_size_mb": round(EN_CKPT.stat().st_size / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 3),
        "num_classes": 23,
        "num_crops": 4,
        "epoch": ckpt.get("epoch"),
        "val_macro_f1": val_metrics.get("val/macro_f1"),
        "val_crop_acc": val_metrics.get("val/crop_acc"),
        "val_f1_failure_classes": {
            "brassica_black_rot": val_metrics.get("val/f1_brassica_black_rot"),
            "okra_cercospora": val_metrics.get("val/f1_okra_cercospora"),
        },
    }

    return model, device, ckpt_meta


def run_inference(model, device, df):
    """Forward-pass all 9006 images through EfficientNet.
    Returns (probs_9class, probs_10class_raw, success_flags).
    """
    logger.info("=" * 70)
    logger.info(f"Running EfficientNet inference on {len(df)} images")
    logger.info("=" * 70)
    logger.info(f"  Batch size: {BATCH_SIZE}, img_size: {IMG_SIZE}")
    logger.info(f"  Preprocessing: RGB per-channel CLAHE (apply_clahe "
                f"from {APPLY_CLAHE_SOURCE})")
    logger.info(f"  Mode: eval() single-pass (no MC Dropout)")

    from scripts.apin.constants import EN_TO_M2_INDEX_MAP

    dataset = Model2ImageDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    n = len(df)
    # 23-class (not 10-class) — this is the intermediate 4-crop model
    probs_23 = np.zeros((n, 23), dtype=np.float32)
    probs_9 = np.zeros((n, 9), dtype=np.float32)
    success_flags = np.ones(n, dtype=bool)
    t_start = time.time()

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Signal 2")
        for batch_tensors, batch_idxs, batch_ok in pbar:
            batch_tensors = batch_tensors.to(device, dtype=torch.float32)
            batch_idxs = batch_idxs.numpy()
            batch_ok = batch_ok.numpy().astype(bool)

            # Forward — PlantDiseaseModel returns (crop_logits, disease_logits, severity_logits)
            _crop_logits, disease_logits, _sev_logits = model(batch_tensors)
            # Raw sigmoid (NOT renormalized) — CRITICAL per architecture_claude_decisions.md
            batch_probs_23 = torch.sigmoid(disease_logits).cpu().numpy()  # (B, 23)

            # Reorder to Model 2's 9-class ordering: take first 10 indices
            # then drop index 8 (clubroot) -> [0,1,2,3,4,5,6,7,9]
            # EN_TO_M2_INDEX_MAP also correctly indexes into 23-wide array
            # since the first 10 classes match the old 10-class ordering.
            batch_probs_9 = batch_probs_23[:, EN_TO_M2_INDEX_MAP]  # (B, 9)

            for j, idx in enumerate(batch_idxs):
                probs_23[idx] = batch_probs_23[j]
                probs_9[idx] = batch_probs_9[j]
                if not batch_ok[j]:
                    success_flags[idx] = False

    elapsed = time.time() - t_start
    n_failed = int((~success_flags).sum())
    logger.info(f"  Inference complete in {elapsed:.1f}s ({n/elapsed:.1f} img/s)")
    logger.info(f"  Successfully inferred: {n - n_failed}/{n}")
    if n_failed > 0:
        logger.warning(f"  Failed image loads: {n_failed}")

    # Sanity checks
    # Note: sigmoid outputs in [0,1] but DO NOT sum to 1 (multi-label head)
    assert (probs_9 >= 0).all() and (probs_9 <= 1).all(), "Sigmoid out of [0,1] range"
    sums = probs_9.sum(axis=1)
    logger.info(f"  Sanity: per-row 9-class sigmoid sums — "
                f"min={sums.min():.3f}, mean={sums.mean():.3f}, max={sums.max():.3f}")
    logger.info(f"    (These do NOT sum to 1 by design — multi-label head, "
                f"raw sigmoid preserved)")

    return probs_9, probs_23, success_flags


def build_cache(df, probs_9, probs_23, success_flags, row_to_split, class_order):
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    cache = {}
    for i in range(len(df)):
        row = df.iloc[i]
        cls = row["class_name"]
        cache[int(i)] = {
            "predictions": probs_9[i].astype(np.float32),  # 9-class, Model 2 order
            "raw_en_23class_predictions": probs_23[i].astype(np.float32),
            "class_name": cls,
            "source_dataset": str(row["source_dataset"]),
            "is_field_photo": bool(row["is_field_photo"]),
            "split": row_to_split[int(i)],
            "is_recomposed": bool(row["is_recomposed"]),
            "true_class_idx": class_to_idx[cls],
            "inference_success": bool(success_flags[i]),
        }
    return cache


def write_outputs(cache, probs_9, success_flags, ckpt_meta, class_order):
    logger.info("=" * 70)
    logger.info("Writing outputs")
    logger.info("=" * 70)

    with open(OUTPUT_CACHE, "wb") as f:
        pickle.dump(cache, f)
    with open(OUTPUT_CACHE_LATEST, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"  Cache: {OUTPUT_CACHE.name} "
                f"({OUTPUT_CACHE.stat().st_size / 1e6:.2f} MB)")

    y_true = np.array([cache[i]["true_class_idx"] for i in sorted(cache.keys())])
    per_class_stats = {}
    for c_idx, c_name in enumerate(class_order):
        mask = y_true == c_idx
        n = int(mask.sum())
        per_class_stats[c_name] = {
            "n_images": n,
            "mean_prediction": [round(v, 6) for v in probs_9[mask].mean(axis=0)] if n > 0 else [0.0] * 9,
            "mean_sigmoid_on_true_class": round(float(probs_9[mask, c_idx].mean()), 6) if n > 0 else 0.0,
            "first_5_predictions": probs_9[mask][:5].round(6).tolist() if n > 0 else [],
        }

    fingerprint = {
        "signal": 2,
        "signal_name": ("EfficientNetV2-S 23-class 4-crop intermediate model "
                         "(models/best_model.pt). The first 10 indices match "
                         "the old 10-class ordering exactly; we subsample 9 "
                         "classes via EN_TO_M2_INDEX_MAP."),
        "timestamp": TIMESTAMP,
        "model_checkpoint": str(EN_CKPT.relative_to(PROJECT_ROOT)),
        "ckpt_meta": ckpt_meta,
        "img_size": IMG_SIZE,
        "inference_mode": (
            "eval() single-pass, no MC Dropout. (Production old_10class "
            "inference uses 5-pass MC Dropout per "
            "old_10class/app/inference.py:139-160 — single-pass chosen for "
            "reproducibility, per architecture_claude_decisions.md)"
        ),
        "preprocessing_branch": "B (RGB per-channel CLAHE)",
        "preprocessing_function_source": APPLY_CLAHE_SOURCE,
        "preprocessing_notes": (
            "apply_clahe extracted verbatim from old_10class/app/inference.py "
            "lines 36-46. cv2.createCLAHE(clip_limit=2.0, tileGridSize=(8,8)) "
            "applied to R, G, B channels independently. Do NOT reimplement."
        ),
        "output_format": (
            "RAW sigmoid probabilities (NOT renormalized), float32. "
            "Class order reordered from EN 10-class to Model 2 9-class "
            "(dropped index 8 = brassica_clubroot)."
        ),
        "class_order_model2": class_order,
        "en_class_order": [
            "okra_yvmv", "okra_powdery_mildew", "okra_cercospora", "okra_enation",
            "okra_healthy", "brassica_black_rot", "brassica_downy_mildew",
            "brassica_alternaria", "brassica_clubroot (DROPPED)", "brassica_healthy",
        ],
        "en_to_m2_index_map": [0, 1, 2, 3, 4, 5, 6, 7, 9],
        "total_rows": len(cache),
        "successfully_inferred_rows": int(success_flags.sum()),
        "failed_image_loads": int((~success_flags).sum()),
        "per_class_stats": per_class_stats,
        "cache_path": str(OUTPUT_CACHE_LATEST.relative_to(PROJECT_ROOT)),
        "verification": {
            "sigmoid_in_0_1": True,
            "not_renormalized_to_sum_1": True,
            "class_order_dropping_clubroot": True,
        },
    }
    with open(OUTPUT_FINGERPRINT, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"  Fingerprint: {OUTPUT_FINGERPRINT.name}")


def report_summary(cache, class_order):
    """Check Signal 2 val accuracy (not F1 — sigmoid is multi-label).
    For failure classes, also report mean sigmoid on true class for
    val_and_soup + field photos only.
    """
    logger.info("=" * 70)
    logger.info("SIGNAL 2 CACHE SUMMARY")
    logger.info("=" * 70)

    val_rows = [e for e in cache.values()
                if e["split"] == "val_and_soup" and e["inference_success"]]
    y_true = np.array([e["true_class_idx"] for e in val_rows])
    probs = np.stack([e["predictions"] for e in val_rows], axis=0)

    # For multi-label sigmoid: argmax approximation + per-class
    y_pred = probs.argmax(axis=1)
    from sklearn.metrics import f1_score
    f1_per_class = f1_score(y_true, y_pred, average=None,
                             labels=list(range(9)), zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                         labels=list(range(9)), zero_division=0)
    logger.info(f"val_and_soup argmax macro F1: {macro_f1:.4f}")
    logger.info("  (Note: EN was trained with multi-label BCE; "
                "argmax accuracy is a lower bound on real performance)")
    logger.info("")
    logger.info(f"{'Class':<28} {'argmax_F1':>10}")
    logger.info("-" * 45)
    for c_name, f1 in zip(class_order, f1_per_class):
        logger.info(f"{c_name:<28} {f1:>10.4f}")


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 2C -- Signal 2 (EfficientNet) Prediction Cache")
    logger.info("=" * 70)

    df = pd.read_csv(CSV_PATH)
    assert "is_recomposed" in df.columns, "Run Section 1 first"
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    row_to_split = {}
    for k, idxs in splits.items():
        for idx in idxs:
            row_to_split[int(idx)] = k

    from scripts.apin.constants import MODEL2_CLASS_ORDER
    model, device, ckpt_meta = load_efficientnet()
    probs_9, probs_23, success_flags = run_inference(model, device, df)

    cache = build_cache(df, probs_9, probs_23, success_flags, row_to_split,
                        MODEL2_CLASS_ORDER)
    write_outputs(cache, probs_9, success_flags, ckpt_meta, MODEL2_CLASS_ORDER)
    report_summary(cache, MODEL2_CLASS_ORDER)

    logger.info("=" * 70)
    logger.info("APIN SECTION 2C -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
