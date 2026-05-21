"""
Section 2A -- Signal 4 Prediction Cache (DINOv2 Nonlinear Head).

Uses the already-trained DINOv2 nonlinear head from last session:
  Model:  scripts/dinov2_probe/results/dinov2_nonlinear_head_20260416_204427.pt
  Scaler: scripts/dinov2_probe/results/dinov2_nonlinear_head_scaler_20260416_204427.pkl
  Config: scripts/dinov2_probe/results/dinov2_nonlinear_head_config_20260416_204427.json

Features are cached at:
  scripts/dinov2_probe/results/dinov2_features_cache.pkl  (9006 x 768, cls_mean)

This script:
  1. Loads features, scaler, MLP weights, and the authoritative MODEL2_CLASS_ORDER.
  2. Applies scaler.transform() + MLP forward pass + softmax.
  3. Saves signal4_predictions_cache.pkl keyed by CSV row index.
  4. Saves a fingerprint JSON with sanity checks (per-class mean prediction,
     total count, first 5 feature values per class).
  5. Verifies class ordering matches MODEL2_CLASS_ORDER exactly.

Output format (per row):
  {
    'predictions': np.array(9,) float32,   # softmax probabilities
    'class_name': str,                     # ground truth class
    'source_dataset': str,
    'is_field_photo': bool,
    'split': str,                          # 'train' / 'val_and_soup' / etc.
    'is_recomposed': bool,
    'true_class_idx': int,                 # index in MODEL2_CLASS_ORDER
  }
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section2a_signal4_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section2a")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.handlers[0].setFormatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)


# Paths (from last session artifacts)
FEATURES_CACHE_PATH = (
    PROJECT_ROOT / "scripts" / "dinov2_probe" / "results" /
    "dinov2_features_cache.pkl"
)
MLP_WEIGHTS_PATH = (
    PROJECT_ROOT / "scripts" / "dinov2_probe" / "results" /
    "dinov2_nonlinear_head_20260416_204427.pt"
)
SCALER_PATH = (
    PROJECT_ROOT / "scripts" / "dinov2_probe" / "results" /
    "dinov2_nonlinear_head_scaler_20260416_204427.pkl"
)
CONFIG_PATH = (
    PROJECT_ROOT / "scripts" / "dinov2_probe" / "results" /
    "dinov2_nonlinear_head_config_20260416_204427.json"
)
CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "model2_unified_source_map.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "split_indices.json"

OUTPUT_CACHE = CACHE_DIR / f"signal4_predictions_cache_{TIMESTAMP}.pkl"
OUTPUT_CACHE_LATEST = CACHE_DIR / "signal4_predictions_cache.pkl"
OUTPUT_FINGERPRINT = CACHE_DIR / f"signal4_predictions_fingerprint_{TIMESTAMP}.json"


class NonlinearHead(nn.Module):
    """Mirrors the architecture from
    scripts/dinov2_probe/train_nonlinear_head.py exactly.
    Topology: 768 -> 512 -> 256 -> 9 with LayerNorm + GELU + Dropout 0.3.
    """

    def __init__(self, in_dim: int = 768, hidden_dims=(512, 256),
                 num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.body(x))


def load_all_artifacts():
    """Load features, splits, CSV, scaler, MLP. Verify dimensions."""
    logger.info(f"Loading CSV: {CSV_PATH.name}")
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 9006
    assert "is_recomposed" in df.columns, (
        "Run Section 1 first — is_recomposed column missing"
    )

    logger.info(f"Loading splits: {SPLITS_PATH.name}")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    # Build row_idx -> split_name reverse lookup
    row_to_split = {}
    for split_name, idxs in splits.items():
        for idx in idxs:
            row_to_split[int(idx)] = split_name
    assert len(row_to_split) == 9006, (
        f"Split coverage: {len(row_to_split)} != 9006"
    )

    logger.info(f"Loading features: {FEATURES_CACHE_PATH.name}")
    with open(FEATURES_CACHE_PATH, "rb") as f:
        features_cache = pickle.load(f)
    logger.info(f"  features_cache entries: {len(features_cache)}")
    # Sanity check feature shape
    first = next(iter(features_cache.values()))
    assert first["feature"].shape == (768,), f"Expected 768d features, got {first['feature'].shape}"

    logger.info(f"Loading scaler: {SCALER_PATH.name}")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    logger.info(f"Loading MLP config: {CONFIG_PATH.name}")
    with open(CONFIG_PATH) as f:
        mlp_config = json.load(f)
    logger.info(f"  MLP topology: 768 -> {' -> '.join(map(str, mlp_config['hidden_dims']))} -> 9")
    logger.info(f"  Training val macro F1: {mlp_config.get('cv_mean_macro_f1', 'unknown')}")

    logger.info(f"Loading MLP weights: {MLP_WEIGHTS_PATH.name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")
    state_dict = torch.load(MLP_WEIGHTS_PATH, map_location=device, weights_only=True)
    mlp = NonlinearHead(
        in_dim=mlp_config["feature_dim"],
        hidden_dims=mlp_config["hidden_dims"],
        num_classes=mlp_config["num_classes"],
        dropout=mlp_config["dropout"],
    )
    mlp.load_state_dict(state_dict)
    mlp.to(device)
    mlp.eval()
    logger.info("  MLP loaded successfully and set to eval()")

    return df, row_to_split, features_cache, scaler, mlp, mlp_config, device


def verify_class_ordering(mlp_config: dict):
    """CRITICAL: assert canonical class ordering matches MLP training."""
    from scripts.apin.constants import MODEL2_CLASS_ORDER, NUM_CLASSES

    cfg_classes = mlp_config.get("class_names", [])
    if list(cfg_classes) != MODEL2_CLASS_ORDER:
        raise AssertionError(
            f"Class ordering mismatch!\n"
            f"  MLP config: {cfg_classes}\n"
            f"  APIN canonical: {MODEL2_CLASS_ORDER}\n"
            f"Would silently corrupt reliability matrix and MLP input — abort."
        )
    assert len(MODEL2_CLASS_ORDER) == NUM_CLASSES == 9
    logger.info("Class ordering verified against apin/constants.py")
    return MODEL2_CLASS_ORDER


def generate_cache(df, row_to_split, features_cache, scaler, mlp, device, class_order):
    """Build signal4_predictions_cache.pkl keyed by CSV row index."""
    logger.info("-" * 70)
    logger.info("Generating Signal 4 predictions for all 9006 rows")
    logger.info("-" * 70)

    class_to_idx = {c: i for i, c in enumerate(class_order)}

    # Align features with CSV row order — features_cache is keyed by image_path
    missing = 0
    feat_matrix = np.zeros((len(df), 768), dtype=np.float32)
    for i, row in df.iterrows():
        key = str(row["image_path"])
        if key not in features_cache:
            missing += 1
            continue
        feat_matrix[i] = features_cache[key]["feature"]

    logger.info(f"  Rows without features: {missing}/{len(df)}")
    if missing > 0:
        logger.warning(f"  {missing} rows have no cached features — will yield all-zero predictions")

    # Scale features (scaler was fit on train split in last session)
    logger.info("  Applying StandardScaler.transform()")
    feat_scaled = scaler.transform(feat_matrix).astype(np.float32)

    # Batched MLP forward pass
    logger.info("  Running MLP forward pass (softmax)")
    with torch.no_grad():
        x = torch.from_numpy(feat_scaled).to(device)
        # Process in batches to be memory-friendly even though 9006x768 is tiny
        batch_size = 512
        all_probs = []
        for start in range(0, len(x), batch_size):
            logits = mlp(x[start:start + batch_size])
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            all_probs.append(probs)
        probs = np.concatenate(all_probs, axis=0)

    assert probs.shape == (len(df), 9), f"Expected ({len(df)}, 9), got {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4), "Softmax does not sum to 1"

    # Build the cache
    cache = {}
    for i, row in df.iterrows():
        cls = row["class_name"]
        cache[int(i)] = {
            "predictions": probs[i].astype(np.float32),
            "class_name": cls,
            "source_dataset": str(row["source_dataset"]),
            "is_field_photo": bool(row["is_field_photo"]),
            "split": row_to_split[int(i)],
            "is_recomposed": bool(row["is_recomposed"]),
            "true_class_idx": class_to_idx[cls],
        }

    logger.info(f"  Built cache with {len(cache)} entries")
    return cache, probs, missing


def write_cache_and_fingerprint(cache, probs, missing, class_order, mlp_config):
    """Write cache + fingerprint JSON."""
    logger.info("-" * 70)
    logger.info("Writing outputs")
    logger.info("-" * 70)

    # Write cache with timestamp AND as 'latest'
    with open(OUTPUT_CACHE, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"  Cache:  {OUTPUT_CACHE.name} ({OUTPUT_CACHE.stat().st_size / 1e6:.2f} MB)")

    with open(OUTPUT_CACHE_LATEST, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"  Latest: {OUTPUT_CACHE_LATEST.name}")

    # Compute per-class statistics for sanity checks
    y_true = np.array([cache[i]["true_class_idx"] for i in sorted(cache.keys())])
    per_class_stats = {}
    for c_idx, c_name in enumerate(class_order):
        mask = y_true == c_idx
        n = int(mask.sum())
        mean_pred = probs[mask].mean(axis=0).tolist() if n > 0 else [0.0] * 9
        first5 = probs[mask][:5].tolist() if n > 0 else []
        per_class_stats[c_name] = {
            "n_images": n,
            "mean_prediction": [round(v, 6) for v in mean_pred],
            "first_5_predictions": [[round(v, 6) for v in row] for row in first5],
            "mean_confidence_on_true_class": round(float(probs[mask, c_idx].mean()), 6) if n > 0 else 0.0,
        }

    # Build fingerprint
    fingerprint = {
        "signal": 4,
        "signal_name": "DINOv2 nonlinear head",
        "timestamp": TIMESTAMP,
        "model_path": str(MLP_WEIGHTS_PATH.relative_to(PROJECT_ROOT)),
        "scaler_path": str(SCALER_PATH.relative_to(PROJECT_ROOT)),
        "features_cache_path": str(FEATURES_CACHE_PATH.relative_to(PROJECT_ROOT)),
        "preprocessing_branch": "A (LAB-CLAHE)",
        "feature_aggregation": "cls_mean (CLS + mean_patch concat, 768d)",
        "mlp_topology": [768, *mlp_config["hidden_dims"], 9],
        "training_val_macro_f1": mlp_config.get("cv_mean_macro_f1"),
        "output_format": "softmax probabilities, float32",
        "class_order": class_order,
        "total_rows": len(cache),
        "missing_features_rows": missing,
        "per_class_stats": per_class_stats,
        "cache_path": str(OUTPUT_CACHE_LATEST.relative_to(PROJECT_ROOT)),
        "verification": {
            "all_rows_have_9d_predictions": True,
            "softmax_sum_to_1_check": True,
        },
    }
    with open(OUTPUT_FINGERPRINT, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"  Fingerprint: {OUTPUT_FINGERPRINT.name}")

    return fingerprint


def report_summary(fingerprint, probs):
    """Print a summary table of cache contents by class and split."""
    logger.info("=" * 70)
    logger.info("SIGNAL 4 CACHE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total cached predictions: {fingerprint['total_rows']}")
    logger.info(f"Missing feature rows:     {fingerprint['missing_features_rows']}")
    logger.info(f"Training val macro F1:    {fingerprint['training_val_macro_f1']}")
    logger.info("")
    logger.info(f"{'Class':<28} {'N':>6} {'MeanConf(true)':>14}")
    logger.info("-" * 50)
    for cls, stats in fingerprint["per_class_stats"].items():
        logger.info(
            f"{cls:<28} {stats['n_images']:>6} "
            f"{stats['mean_confidence_on_true_class']:>14.4f}"
        )


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 2A -- Signal 4 (DINOv2 head) Prediction Cache")
    logger.info("=" * 70)

    df, row_to_split, features_cache, scaler, mlp, mlp_config, device = load_all_artifacts()
    class_order = verify_class_ordering(mlp_config)
    cache, probs, missing = generate_cache(
        df, row_to_split, features_cache, scaler, mlp, device, class_order
    )
    fingerprint = write_cache_and_fingerprint(
        cache, probs, missing, class_order, mlp_config
    )
    report_summary(fingerprint, probs)

    logger.info("=" * 70)
    logger.info("APIN SECTION 2A -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
