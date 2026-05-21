"""
Section 2B -- Signal 1 Prediction Cache (Model 2: DINOv3-ConvNeXt-Small).

Load models/model2_specialist/model2_production.pt. Single-pass eval() mode
(no MC Dropout). Preprocessing Branch A: LAB-CLAHE (images already
pre-processed during Phase 0 — use the clahe_path column). Resize 384
(Model 2 was trained at 384 per ckpt). ImageNet normalise. Softmax 9 classes.

After cache generation, also compute the ACTUAL failure distributions from
Model 2's misclassified val_and_soup images for brassica_black_rot and
okra_cercospora — the user requested these measured values in place of
the hand-specified prompt defaults.

Output cache format (per row): same as Signal 4.
Output: scripts/apin/caches/signal1_predictions_cache.pkl + fingerprint JSON
        + failure_distributions.json
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section2b_signal1_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section2b")
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
MODEL2_CKPT = PROJECT_ROOT / "models" / "model2_specialist" / "model2_production.pt"
CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "model2_unified_source_map.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "split_indices.json"
OUTPUT_CACHE = CACHE_DIR / f"signal1_predictions_cache_{TIMESTAMP}.pkl"
OUTPUT_CACHE_LATEST = CACHE_DIR / "signal1_predictions_cache.pkl"
OUTPUT_FINGERPRINT = CACHE_DIR / f"signal1_predictions_fingerprint_{TIMESTAMP}.json"
OUTPUT_FAILURE_DIST = CACHE_DIR / f"signal1_measured_failure_distributions_{TIMESTAMP}.json"
OUTPUT_FAILURE_DIST_LATEST = CACHE_DIR / "signal1_measured_failure_distributions.json"

# Constants — Model 2 trained at 384px per checkpoint
IMG_SIZE = 384
BATCH_SIZE = 32  # ConvNeXt-Small at 384 is heavier than at 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Model2ImageDataset(Dataset):
    """Loads LAB-CLAHE'd images at Model 2 training resolution (384).
    Returns (tensor, row_idx, success_flag).
    """

    def __init__(self, df, csv_to_path_col="clahe_path", fallback_col="image_path"):
        self.df = df.reset_index(drop=True)
        self.clahe_col = csv_to_path_col
        self.fallback_col = fallback_col
        # Standard preprocessing matching Model 2 training
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row[self.clahe_col]
        if not isinstance(path, str) or not Path(path).exists():
            path = row[self.fallback_col]

        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, int(i), 1  # success
        except Exception:
            # Return zero tensor on failure — record the failure downstream
            return torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32), int(i), 0


def load_model2():
    """Construct Model2ConvNeXt and load weights. Return (model, device, ckpt_meta)."""
    logger.info("=" * 70)
    logger.info("Loading Model 2 checkpoint")
    logger.info("=" * 70)

    from scripts.models import Model2ConvNeXt
    from scripts.apin.constants import MODEL2_CLASS_ORDER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    ckpt = torch.load(MODEL2_CKPT, map_location="cpu", weights_only=False)
    assert ckpt["num_classes"] == 9
    assert list(ckpt["class_names"]) == MODEL2_CLASS_ORDER, (
        f"Checkpoint class order mismatch!\n"
        f"  Ckpt: {ckpt['class_names']}\n"
        f"  APIN: {MODEL2_CLASS_ORDER}"
    )
    logger.info(f"  Class order: verified against MODEL2_CLASS_ORDER")
    logger.info(f"  Training img_size: {ckpt['img_size']}")
    logger.info(f"  Training val F1:   {ckpt['val_f1']:.4f}")
    logger.info(f"  Backbone:          {ckpt['backbone_name']}")

    # Build model — allow network fetch for first-time backbone download
    model = Model2ConvNeXt(num_classes=9, pretrained=True)

    # State-dict key remapping for transformers version skew.
    # Checkpoint was saved with keys like 'backbone.stages.X...' (older
    # transformers). The current transformers version wraps DINOv3 such
    # that keys are 'backbone.model.stages.X...' inside AutoModel.
    # Remap by inserting '.model' after 'backbone.' for backbone keys only.
    current_keys = set(model.state_dict().keys())
    ckpt_sd = ckpt["model_state_dict"]
    remapped_sd = {}
    renamed = 0
    for k, v in ckpt_sd.items():
        if k.startswith("backbone.") and not k.startswith("backbone.model."):
            new_key = "backbone.model." + k[len("backbone."):]
            if new_key in current_keys:
                remapped_sd[new_key] = v
                renamed += 1
                continue
        remapped_sd[k] = v
    if renamed > 0:
        logger.info(f"  Remapped {renamed} backbone keys "
                    f"(transformers version skew fix: 'backbone.' -> 'backbone.model.')")

    missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
    if missing:
        logger.warning(f"  Missing keys in state_dict load: {len(missing)} "
                       f"(first 3: {missing[:3]})")
    if unexpected:
        logger.warning(f"  Unexpected keys in state_dict load: {len(unexpected)} "
                       f"(first 3: {unexpected[:3]})")
    # Head keys MUST be present — the training-learned classifier
    head_keys = [k for k in current_keys if k.startswith("head.")]
    for hk in head_keys:
        assert hk not in missing, (
            f"CRITICAL: head key '{hk}' missing from checkpoint. "
            f"The trained classifier is lost — aborting."
        )
    logger.info("  Head weights loaded successfully (trained classifier preserved)")

    model = model.to(device)
    model.eval()

    # Float32 master weights — autocast(dtype=bfloat16) for forward pass only.
    # Do NOT call model.to(torch.bfloat16) (that's the documented BF16 bug).
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total params: {total_params / 1e6:.2f}M")

    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"  VRAM allocated after load: {vram_used:.2f} GB")

    return model, device, ckpt


def run_inference(model, device, df):
    """Forward-pass all 9006 images through Model 2. Returns (probs, failures)."""
    logger.info("=" * 70)
    logger.info(f"Running Model 2 inference on {len(df)} images")
    logger.info("=" * 70)
    logger.info(f"  Batch size: {BATCH_SIZE}, img_size: {IMG_SIZE}")

    dataset = Model2ImageDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows: avoid deadlocks
        pin_memory=(device == "cuda"),
    )

    n = len(df)
    probs = np.zeros((n, 9), dtype=np.float32)
    success_flags = np.ones(n, dtype=bool)
    t_start = time.time()

    use_amp = (device == "cuda")
    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Signal 1")
        for batch_tensors, batch_idxs, batch_ok in pbar:
            batch_tensors = batch_tensors.to(device, dtype=torch.float32)
            batch_idxs = batch_idxs.numpy()
            batch_ok = batch_ok.numpy().astype(bool)

            with torch.autocast(device_type=("cuda" if use_amp else "cpu"),
                                dtype=torch.bfloat16, enabled=use_amp):
                logits = model(batch_tensors)  # (B, 9)

            # Softmax in float32 for numerical stability
            batch_probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

            for j, idx in enumerate(batch_idxs):
                probs[idx] = batch_probs[j]
                if not batch_ok[j]:
                    success_flags[idx] = False

    elapsed = time.time() - t_start
    n_failed = int((~success_flags).sum())
    logger.info(f"  Inference complete in {elapsed:.1f}s "
                f"({n/elapsed:.1f} img/s)")
    logger.info(f"  Successfully inferred: {n - n_failed}/{n}")
    if n_failed > 0:
        logger.warning(f"  Failed image loads: {n_failed} "
                       f"(rows returned all-zero predictions)")

    # Sanity check
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), (
        "Softmax normalization check failed"
    )
    return probs, success_flags


def build_cache(df, probs, success_flags, row_to_split, class_order):
    """Build the dict-of-dicts cache keyed by CSV row index."""
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    cache = {}
    for i in range(len(df)):
        row = df.iloc[i]
        cls = row["class_name"]
        cache[int(i)] = {
            "predictions": probs[i].astype(np.float32),
            "class_name": cls,
            "source_dataset": str(row["source_dataset"]),
            "is_field_photo": bool(row["is_field_photo"]),
            "split": row_to_split[int(i)],
            "is_recomposed": bool(row["is_recomposed"]),
            "true_class_idx": class_to_idx[cls],
            "inference_success": bool(success_flags[i]),
        }
    return cache


def measure_failure_distributions(cache, class_order):
    """Compute the ACTUAL failure distribution for brassica_black_rot and
    okra_cercospora from Model 2's misclassified val_and_soup images.

    A "misclassified" image is one where the true class's probability is not
    the argmax. Take the mean of Model 2's 9-dim probability vector across
    all such misclassified val images of the target class.
    """
    logger.info("=" * 70)
    logger.info("Measuring ACTUAL Model 2 failure distributions from val misclassifications")
    logger.info("=" * 70)

    class_to_idx = {c: i for i, c in enumerate(class_order)}
    results = {}

    for target_class in ["brassica_black_rot", "okra_cercospora"]:
        target_idx = class_to_idx[target_class]

        # Filter to val_and_soup + ground-truth target class + inference succeeded
        rows = [
            (idx, entry) for idx, entry in cache.items()
            if entry["split"] == "val_and_soup"
            and entry["class_name"] == target_class
            and entry["inference_success"]
        ]
        logger.info(f"\n{target_class}:")
        logger.info(f"  val_and_soup images of this class: {len(rows)}")

        # Split into correct vs misclassified
        preds = np.stack([entry["predictions"] for _, entry in rows], axis=0)
        argmaxes = preds.argmax(axis=1)
        correct_mask = (argmaxes == target_idx)
        wrong_mask = ~correct_mask

        n_correct = int(correct_mask.sum())
        n_wrong = int(wrong_mask.sum())
        logger.info(f"  argmax correct: {n_correct}")
        logger.info(f"  argmax wrong (misclassified): {n_wrong}")

        # Also break down misclassifications by which class Model 2 picked instead
        wrong_argmaxes = argmaxes[wrong_mask]
        confusion_counts = {}
        for wa in wrong_argmaxes:
            cls_name = class_order[int(wa)]
            confusion_counts[cls_name] = confusion_counts.get(cls_name, 0) + 1
        logger.info(f"  misclassified-into breakdown:")
        for cls_name, count in sorted(confusion_counts.items(),
                                       key=lambda x: -x[1]):
            logger.info(f"    -> {cls_name}: {count}")

        # Mean probability vector on misclassified images
        if n_wrong > 0:
            mean_wrong_probs = preds[wrong_mask].mean(axis=0)
        else:
            mean_wrong_probs = np.zeros(9, dtype=np.float32)

        # Also compute: mean prob vector on FIELD PHOTOS ONLY that were misclassified
        field_rows = [
            (idx, entry) for idx, entry in rows if entry["is_field_photo"]
        ]
        if field_rows:
            field_preds = np.stack([e["predictions"] for _, e in field_rows], axis=0)
            field_argmaxes = field_preds.argmax(axis=1)
            field_wrong_mask = field_argmaxes != target_idx
            n_field_wrong = int(field_wrong_mask.sum())
            if n_field_wrong > 0:
                mean_field_wrong_probs = field_preds[field_wrong_mask].mean(axis=0)
            else:
                mean_field_wrong_probs = np.zeros(9, dtype=np.float32)
        else:
            n_field_wrong = 0
            mean_field_wrong_probs = np.zeros(9, dtype=np.float32)

        logger.info(f"  field-only misclassified: {n_field_wrong}")

        # Format the distribution as a dict class_name -> prob
        mean_all_dict = {
            class_order[i]: round(float(mean_wrong_probs[i]), 6)
            for i in range(9)
        }
        mean_field_dict = {
            class_order[i]: round(float(mean_field_wrong_probs[i]), 6)
            for i in range(9)
        }

        logger.info("  Mean probability vector on ALL misclassified val images:")
        for c_name, p in sorted(mean_all_dict.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"    {c_name:<28}: {p:.4f}")

        logger.info("  Mean probability vector on FIELD misclassified val images:")
        for c_name, p in sorted(mean_field_dict.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"    {c_name:<28}: {p:.4f}")

        results[target_class] = {
            "n_val_and_soup_images_of_class": len(rows),
            "n_correctly_classified": n_correct,
            "n_misclassified": n_wrong,
            "n_field_misclassified": n_field_wrong,
            "confusion_counts": confusion_counts,
            "mean_failure_distribution_all": mean_all_dict,
            "mean_failure_distribution_field_only": mean_field_dict,
        }

    return results


def write_outputs(cache, probs, success_flags, df, failure_dists,
                  ckpt_meta, class_order):
    """Write cache, fingerprint, failure-distribution JSONs."""
    logger.info("=" * 70)
    logger.info("Writing outputs")
    logger.info("=" * 70)

    # Cache
    with open(OUTPUT_CACHE, "wb") as f:
        pickle.dump(cache, f)
    with open(OUTPUT_CACHE_LATEST, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"  Cache:  {OUTPUT_CACHE.name} "
                f"({OUTPUT_CACHE.stat().st_size / 1e6:.2f} MB)")

    # Per-class stats for fingerprint
    y_true = np.array([cache[i]["true_class_idx"] for i in sorted(cache.keys())])
    per_class_stats = {}
    for c_idx, c_name in enumerate(class_order):
        mask = y_true == c_idx
        n = int(mask.sum())
        mean_pred = probs[mask].mean(axis=0).tolist() if n > 0 else [0.0] * 9
        per_class_stats[c_name] = {
            "n_images": n,
            "mean_prediction": [round(v, 6) for v in mean_pred],
            "mean_confidence_on_true_class": (
                round(float(probs[mask, c_idx].mean()), 6) if n > 0 else 0.0
            ),
            "first_5_predictions": probs[mask][:5].round(6).tolist() if n > 0 else [],
        }

    fingerprint = {
        "signal": 1,
        "signal_name": "Model 2 (DINOv3-ConvNeXt-Small)",
        "timestamp": TIMESTAMP,
        "model_checkpoint": str(MODEL2_CKPT.relative_to(PROJECT_ROOT)),
        "backbone_name": ckpt_meta["backbone_name"],
        "embed_dim": ckpt_meta["embed_dim"],
        "img_size": IMG_SIZE,
        "training_img_size": ckpt_meta["img_size"],
        "training_val_f1": ckpt_meta["val_f1"],
        "inference_mode": "eval() single-pass, no MC Dropout, torch.autocast(bfloat16) forward",
        "preprocessing_branch": "A (LAB-CLAHE pre-applied during Phase 0, uses clahe_path column)",
        "output_format": "softmax probabilities, float32",
        "class_order": class_order,
        "total_rows": len(cache),
        "successfully_inferred_rows": int(success_flags.sum()),
        "failed_image_loads": int((~success_flags).sum()),
        "per_class_stats": per_class_stats,
        "cache_path": str(OUTPUT_CACHE_LATEST.relative_to(PROJECT_ROOT)),
        "verification": {
            "softmax_sum_to_1": True,
            "class_order_matches_ckpt": True,
            "bf16_master_weights_avoided": True,  # Float32 master weights
        },
    }
    with open(OUTPUT_FINGERPRINT, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"  Fingerprint: {OUTPUT_FINGERPRINT.name}")

    # Failure distributions
    failure_dists_out = {
        "timestamp": TIMESTAMP,
        "source": ("Model 2 predictions on val_and_soup split "
                   "(measured empirically, not hand-specified)"),
        "class_order": class_order,
        "distributions": failure_dists,
    }
    with open(OUTPUT_FAILURE_DIST, "w") as f:
        json.dump(failure_dists_out, f, indent=2)
    with open(OUTPUT_FAILURE_DIST_LATEST, "w") as f:
        json.dump(failure_dists_out, f, indent=2)
    logger.info(f"  Failure distributions: {OUTPUT_FAILURE_DIST.name}")


def report_summary(cache, class_order):
    """Compute headline val metrics on val_and_soup for Signal 1 as a sanity check."""
    logger.info("=" * 70)
    logger.info("SIGNAL 1 CACHE SUMMARY (val_and_soup check)")
    logger.info("=" * 70)

    # val_and_soup split only
    val_rows = [e for e in cache.values()
                if e["split"] == "val_and_soup" and e["inference_success"]]
    y_true = np.array([e["true_class_idx"] for e in val_rows])
    y_pred_probs = np.stack([e["predictions"] for e in val_rows], axis=0)
    y_pred = y_pred_probs.argmax(axis=1)

    # Per-class F1
    from sklearn.metrics import f1_score
    f1_per_class = f1_score(y_true, y_pred, average=None,
                             labels=list(range(9)), zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                         labels=list(range(9)), zero_division=0)
    logger.info(f"val_and_soup macro F1:    {macro_f1:.4f}")
    logger.info(f"published val F1 baseline: 0.9443 (from checkpoint)")
    logger.info("")
    logger.info(f"{'Class':<28} {'val_F1':>8}")
    logger.info("-" * 40)
    for c_name, f1 in zip(class_order, f1_per_class):
        logger.info(f"{c_name:<28} {f1:>8.4f}")


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 2B -- Signal 1 (Model 2) Prediction Cache")
    logger.info("=" * 70)

    # Load CSV and splits
    df = pd.read_csv(CSV_PATH)
    assert "is_recomposed" in df.columns, "Run Section 1 first"
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    row_to_split = {}
    for k, idxs in splits.items():
        for idx in idxs:
            row_to_split[int(idx)] = k

    # Load model and run inference
    from scripts.apin.constants import MODEL2_CLASS_ORDER
    model, device, ckpt_meta = load_model2()
    probs, success_flags = run_inference(model, device, df)

    # Build cache
    cache = build_cache(df, probs, success_flags, row_to_split, MODEL2_CLASS_ORDER)

    # Measure failure distributions
    failure_dists = measure_failure_distributions(cache, MODEL2_CLASS_ORDER)

    # Write outputs
    write_outputs(cache, probs, success_flags, df, failure_dists,
                  ckpt_meta, MODEL2_CLASS_ORDER)

    # Final summary
    report_summary(cache, MODEL2_CLASS_ORDER)

    logger.info("=" * 70)
    logger.info("APIN SECTION 2B -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
