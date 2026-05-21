"""
DINOv2 Feature Cache — Extract and cache frozen backbone features for all images.

Loads the frozen DINOv2-Small-Registers backbone (same as router), runs all
Model 2 training images through it, and saves the resulting feature vectors.
This runs ONCE (~30-45 min on GPU). All subsequent scripts load from cache.

Research-informed choices:
  - Backbone: timm pretrained vit_small_patch14_reg4_dinov2.lvd142m
    Identical to router backbone (router only trained a head, backbone frozen).
  - Feature aggregation: CLS + mean_patch concat (768d) per DINOv2 paper.
    Register tokens at indices 1-4 EXCLUDED from mean patch.
  - Preprocessing: Resize(224) + ImageNet normalize. Images loaded from
    clahe_path (already LAB-CLAHE processed). Do NOT apply CLAHE again.
"""

import os
import sys
import json
import time
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dinov2_probe.config import (
    PROJECT_ROOT, MODEL2_CSV, SPLIT_INDICES,
    TIMM_MODEL_NAME, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, NUM_PREFIX_TOKENS,
    FEATURE_AGGREGATION, FEATURE_DIM_CLS, FEATURE_DIM_CLS_MEAN,
    CLASS_TO_IDX, CLASS_NAMES, NUM_CLASSES,
    FEATURES_CACHE_PATH, CACHE_FINGERPRINT_PATH, EXTRACTION_FAILURES_PATH,
    RESULTS_DIR, RANDOM_SEED,
)


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

class Model2ImageDataset(Dataset):
    """Dataset that loads images from Model 2 CSV with correct preprocessing."""

    def __init__(self, df: pd.DataFrame, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        row = self.df.iloc[idx]

        # Prefer CLAHE path (already LAB-CLAHE processed)
        img_path = row.get('clahe_path', row['image_path'])
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            img_path = row['image_path']

        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
        except Exception as e:
            # Return a black image on failure — flagged in metadata
            logger.warning(f"Failed to load {img_path}: {e}")
            tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        metadata = {
            'image_path': str(row['image_path']),
            'clahe_path': str(row.get('clahe_path', '')),
            'class_name': str(row['class_name']),
            'label': CLASS_TO_IDX.get(str(row['class_name']), -1),
            'source_dataset': str(row.get('source_dataset', '')),
            'is_field_photo': str(row.get('is_field_photo', 'False')).lower() == 'true',
            'crop': str(row.get('crop', '')),
        }

        return tensor, metadata


# ═══════════════════════════════════════════════════════════════════════
# BACKBONE LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_backbone(device: str = 'cuda') -> nn.Module:
    """
    Load frozen DINOv2-Small-Registers backbone from timm.

    This is the SAME backbone as the router. The router only trained a
    linear head; the backbone was frozen during router training.
    Using timm pretrained directly = using the router backbone.
    """
    import timm

    logger.info(f"Loading backbone: {TIMM_MODEL_NAME}")
    model = timm.create_model(
        TIMM_MODEL_NAME,
        pretrained=True,
        num_classes=0,       # Remove classification head, return features
        img_size=IMG_SIZE,   # CRITICAL: default 518 crashes with 224px input
    )
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Verify frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable == 0, f"Backbone has {trainable} trainable params — should be 0!"
    logger.info(f"Backbone loaded: {total:,} params, ALL frozen")
    logger.info(f"num_prefix_tokens: {model.num_prefix_tokens}")

    # Verify prefix token count matches config
    assert model.num_prefix_tokens == NUM_PREFIX_TOKENS, (
        f"Model has {model.num_prefix_tokens} prefix tokens but config says {NUM_PREFIX_TOKENS}")

    return model


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def get_feature_dim(aggregation: str) -> int:
    """Return expected feature dimension for given aggregation strategy."""
    if aggregation == 'cls':
        return FEATURE_DIM_CLS          # 384
    elif aggregation == 'mean_patch':
        return FEATURE_DIM_CLS          # 384
    elif aggregation == 'cls_mean':
        return FEATURE_DIM_CLS_MEAN     # 768
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    aggregation: str,
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Extract DINOv2 features for all images in the dataloader.

    Args:
        model: frozen DINOv2 backbone
        dataloader: Model2ImageDataset wrapped in DataLoader
        aggregation: 'cls', 'mean_patch', or 'cls_mean'
        device: 'cuda' or 'cpu'

    Returns:
        Dict with image_path as key, value is dict with feature + metadata
    """
    cache = {}
    failures = []
    total = len(dataloader.dataset)
    processed = 0
    t0 = time.time()

    expected_dim = get_feature_dim(aggregation)

    for batch_idx, (images, batch_meta) in enumerate(dataloader):
        images = images.to(device)
        batch_size = images.shape[0]

        # Extract features based on aggregation strategy
        if aggregation == 'cls':
            # num_classes=0 returns CLS token directly
            features = model(images)  # (B, 384)
        else:
            # Need full token output for mean_patch or cls_mean
            full_out = model.forward_features(images)  # (B, 261, 384)
            cls_token = full_out[:, 0, :]                # (B, 384) — CLS
            # Patch tokens: skip prefix tokens (CLS + registers)
            patch_tokens = full_out[:, NUM_PREFIX_TOKENS:, :]  # (B, 256, 384)
            mean_patch = patch_tokens.mean(dim=1)        # (B, 384)

            if aggregation == 'mean_patch':
                features = mean_patch
            elif aggregation == 'cls_mean':
                features = torch.cat([cls_token, mean_patch], dim=1)  # (B, 768)

        features = features.float().cpu().numpy()  # (B, D)

        # Store each sample
        for i in range(batch_size):
            img_path = batch_meta['image_path'][i]
            label = batch_meta['label'][i].item() if torch.is_tensor(batch_meta['label'][i]) else int(batch_meta['label'][i])

            feat = features[i]

            # Sanity check
            if np.isnan(feat).any() or np.isinf(feat).any():
                logger.warning(f"NaN/Inf in features for {img_path}")
                failures.append(img_path)
                continue

            if feat.shape[0] != expected_dim:
                logger.error(f"Wrong feature dim {feat.shape[0]} for {img_path}, expected {expected_dim}")
                failures.append(img_path)
                continue

            cache[img_path] = {
                'feature': feat,
                'label': label,
                'class_name': batch_meta['class_name'][i],
                'source_dataset': batch_meta['source_dataset'][i],
                'is_field_photo': bool(batch_meta['is_field_photo'][i]),
                'clahe_path': batch_meta['clahe_path'][i],
                'image_path': img_path,
                'crop': batch_meta['crop'][i],
            }

        processed += batch_size
        elapsed = time.time() - t0
        rate = processed / max(elapsed, 1)
        eta = (total - processed) / max(rate, 0.1)

        if (batch_idx + 1) % 20 == 0 or processed >= total:
            print(f"  [{processed}/{total}] {processed/total*100:.0f}% "
                  f"({rate:.0f} img/s, ETA {eta:.0f}s)", flush=True)

    return cache, failures


# ═══════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def save_cache(cache: Dict, path: Path) -> None:
    """Save feature cache to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / 1e6
    logger.info(f"Cache saved: {path} ({size_mb:.1f} MB, {len(cache)} images)")


def load_cache(path: Path) -> Optional[Dict]:
    """Load feature cache from pickle file."""
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"Cache loaded: {path} ({len(cache)} images)")
    return cache


def save_fingerprint(cache: Dict, aggregation: str, failures: list) -> None:
    """Save cache fingerprint for validation."""
    # Per-class counts
    class_counts = {}
    field_counts = {}
    for entry in cache.values():
        cls = entry['class_name']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if entry['is_field_photo']:
            field_counts[cls] = field_counts.get(cls, 0) + 1

    # Feature means per class (first 5 dims for quick sanity check)
    feature_means = {}
    for cls in CLASS_NAMES:
        feats = [e['feature'] for e in cache.values() if e['class_name'] == cls]
        if feats:
            mean = np.mean(feats, axis=0)[:5].tolist()
            feature_means[cls] = [round(v, 6) for v in mean]

    fingerprint = {
        'timm_model': TIMM_MODEL_NAME,
        'feature_aggregation': aggregation,
        'feature_dim': get_feature_dim(aggregation),
        'img_size': IMG_SIZE,
        'total_images': len(cache),
        'per_class_counts': class_counts,
        'per_class_field_counts': field_counts,
        'feature_mean_per_class': feature_means,
        'failed_images': failures,
        'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'preprocessing': f'Resize({IMG_SIZE}), ToTensor, Normalize(ImageNet)',
    }

    CACHE_FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FINGERPRINT_PATH, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"Fingerprint saved: {CACHE_FINGERPRINT_PATH}")


def validate_cache(cache: Dict, df: pd.DataFrame) -> None:
    """Validate the extracted feature cache against the CSV."""
    errors = []

    # 1. Total count
    expected = len(df)
    actual = len(cache)
    if actual < expected * 0.95:
        errors.append(f"Cache has {actual} images but CSV has {expected} ({actual/expected*100:.1f}%)")

    # 2. Per-class counts
    csv_counts = df['class_name'].value_counts().to_dict()
    cache_counts = {}
    for entry in cache.values():
        cls = entry['class_name']
        cache_counts[cls] = cache_counts.get(cls, 0) + 1

    for cls in CLASS_NAMES:
        expected_n = csv_counts.get(cls, 0)
        actual_n = cache_counts.get(cls, 0)
        if actual_n < expected_n * 0.90:
            errors.append(f"{cls}: cache has {actual_n} but CSV has {expected_n}")

    # 3. Feature dimension
    sample = next(iter(cache.values()))
    dim = sample['feature'].shape[0]
    expected_dim = get_feature_dim(FEATURE_AGGREGATION)
    if dim != expected_dim:
        errors.append(f"Feature dim {dim} != expected {expected_dim}")

    # 4. No NaN/Inf
    nan_count = sum(1 for e in cache.values() if np.isnan(e['feature']).any())
    inf_count = sum(1 for e in cache.values() if np.isinf(e['feature']).any())
    if nan_count > 0:
        errors.append(f"{nan_count} features have NaN values")
    if inf_count > 0:
        errors.append(f"{inf_count} features have Inf values")

    # 5. Not all identical
    feats = np.array([e['feature'] for e in list(cache.values())[:100]])
    if feats.std() < 1e-6:
        errors.append("Feature vectors are all nearly identical!")

    # 6. Per-class means differ
    class_means = {}
    for cls in CLASS_NAMES:
        cls_feats = [e['feature'] for e in cache.values() if e['class_name'] == cls]
        if cls_feats:
            class_means[cls] = np.mean(cls_feats, axis=0)
    if len(class_means) >= 2:
        means_list = list(class_means.values())
        max_cos_sim = 0
        for i in range(len(means_list)):
            for j in range(i + 1, len(means_list)):
                cos = np.dot(means_list[i], means_list[j]) / (
                    np.linalg.norm(means_list[i]) * np.linalg.norm(means_list[j]) + 1e-8)
                max_cos_sim = max(max_cos_sim, cos)
        if max_cos_sim > 0.999:
            errors.append(f"Per-class feature means are nearly identical (max cos sim {max_cos_sim:.6f})")

    if errors:
        for err in errors:
            logger.error(f"VALIDATION FAILURE: {err}")
        raise ValueError(f"Cache validation failed with {len(errors)} errors:\n" +
                         "\n".join(f"  - {e}" for e in errors))

    logger.info("Cache validation PASSED (counts, dims, NaN/Inf, diversity all OK)")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_feature_extraction(
    aggregation: str = None,
    force_reextract: bool = False,
) -> Dict:
    """
    Main entry point: extract and cache DINOv2 features for all Model 2 images.

    Args:
        aggregation: override config FEATURE_AGGREGATION
        force_reextract: ignore existing cache

    Returns:
        Feature cache dict
    """
    aggregation = aggregation or FEATURE_AGGREGATION

    # Check for existing valid cache
    if not force_reextract and FEATURES_CACHE_PATH.exists():
        cache = load_cache(FEATURES_CACHE_PATH)
        if cache and CACHE_FINGERPRINT_PATH.exists():
            with open(CACHE_FINGERPRINT_PATH) as f:
                fp = json.load(f)
            if (fp.get('timm_model') == TIMM_MODEL_NAME and
                fp.get('feature_aggregation') == aggregation and
                fp.get('total_images', 0) > 8000):
                logger.info("Valid cache found — skipping extraction")
                return cache
            else:
                logger.info("Cache fingerprint mismatch — re-extracting")
        else:
            logger.info("No fingerprint — re-extracting")

    # Load CSV
    logger.info(f"Loading CSV: {MODEL2_CSV}")
    df = pd.read_csv(MODEL2_CSV)
    logger.info(f"Total images: {len(df)}")

    # Print per-class summary
    print("\nPer-class image counts:", flush=True)
    for cls in CLASS_NAMES:
        n = len(df[df['class_name'] == cls])
        field = len(df[(df['class_name'] == cls) &
                       (df['is_field_photo'].astype(str).str.lower() == 'true')])
        print(f"  {cls:30s}: {n:5d} total, {field:4d} field", flush=True)

    # Setup transforms
    # CRITICAL: must match router training preprocessing exactly
    # Router used: Resize(224), Normalize(ImageNet mean/std)
    # Images from clahe_path are already LAB-CLAHE processed
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = Model2ImageDataset(df, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load backbone
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_backbone(device)

    # Extract features
    print(f"\nExtracting features ({aggregation}, {get_feature_dim(aggregation)}d)...",
          flush=True)
    t0 = time.time()
    cache, failures = extract_features(model, dataloader, aggregation, device)
    elapsed = time.time() - t0
    print(f"Extraction complete: {len(cache)} images in {elapsed:.0f}s "
          f"({len(cache)/elapsed:.0f} img/s)", flush=True)

    # Log failures
    if failures:
        logger.warning(f"{len(failures)} images failed feature extraction")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(EXTRACTION_FAILURES_PATH, 'w') as f:
            for fp in failures:
                f.write(fp + '\n')
        if len(failures) / len(df) > 0.05:
            raise RuntimeError(f"Too many failures: {len(failures)}/{len(df)} "
                              f"({len(failures)/len(df)*100:.1f}%)")

    # Validate
    validate_cache(cache, df)

    # Save
    save_cache(cache, FEATURES_CACHE_PATH)
    save_fingerprint(cache, aggregation, failures)

    # Print summary
    print(f"\nFeature cache summary:", flush=True)
    print(f"  Total images: {len(cache)}", flush=True)
    print(f"  Feature dim: {get_feature_dim(aggregation)}", flush=True)
    print(f"  Aggregation: {aggregation}", flush=True)
    print(f"  Failures: {len(failures)}", flush=True)
    print(f"  Cache size: {FEATURES_CACHE_PATH.stat().st_size / 1e6:.1f} MB", flush=True)

    return cache


if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run_feature_extraction()
