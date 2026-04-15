"""
train_utils.py — Shared Training Utilities for Specialist Pipeline

All training scripts (Router, Model 2, Model 3) import from here.
No technique is duplicated across scripts. Each section is self-contained.

Sections:
  A: Checkpointing (save/load/find_latest with RNG state)
  B: EMA (setup, reset for Stage 2)
  C: Model Soup (greedy selection)
  D: Conformal Prediction (APS for specialists, routing thresholds)
  E: Augmentation (AugMix + GridDistortion for curl classes)
  F: CutMix (Stage 2 only, thin classes)
  G: Capsicum Monitoring (subsource F1 tracking, adaptive intervention)
  H: Rollback (save/check/apply, resolution-aware for Model 2)
  I: Sampling Verification (Monte Carlo weight simulation)
  J: Data Loading (split loading with exclusion rules)
  K: Self-Distillation (soft labels with agreement mask)
  L: Evaluation (per-class F1, macro F1, subsource tracking)
  M: Compilation (no-op on Windows — Triton unavailable)
  N: TTA (Test-Time Augmentation — 5 views router, 8 views specialists)
  O: MC Dropout (5 passes, model.train() mode for epistemic uncertainty)
  P: DINO Attention Maps (Model 3 heatmaps from self-attention)
  Q: SupCon Loss (supervised contrastive, temperature=0.10)
  R: ASAM Wrapper (sharpness-aware minimisation for Model 2)
  S: Parameter Groups (no-decay + LLRD for ConvNeXt)
  T: Stage Transitions (freeze/unfreeze, Stage 1 -> Stage 2)
  U: FiLM Module (crop-conditioned LoRA output modulation)
  V: Mixed Loss (KL for soft targets + CE for hard targets)
  W: ENS Class Weights (effective number of samples weighting)
"""

import os
import sys
import glob
import json
import math
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(epoch: int, model: nn.Module, ema_model,
                    optimizer, scheduler, scaler,
                    best_f1: float, path: str,
                    extra: dict = None):
    """
    Save full training state for crash-safe resume.
    Includes RNG states for exact reproducibility.

    Args:
        epoch: current epoch number
        model: the training model (may be peft-wrapped)
        ema_model: EMA shadow model (or None)
        optimizer: optimizer state
        scheduler: LR scheduler state
        scaler: GradScaler state (for BF16)
        best_f1: best validation F1 seen so far
        path: checkpoint file path
        extra: any additional data to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_f1': best_f1,
        'rng_state_cpu': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
    # [FIX I-2] Save CUDA RNG state for exact reproducibility
    if torch.cuda.is_available():
        state['rng_state_gpu'] = torch.cuda.get_rng_state()
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    if ema_model is not None:
        # timm ModelEmaV2 stores state in .module
        if hasattr(ema_model, 'module'):
            state['ema_state_dict'] = ema_model.module.state_dict()
        else:
            state['ema_state_dict'] = ema_model.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, ema_model=None,
                    optimizer=None, scheduler=None, scaler=None,
                    device='cuda') -> Tuple[int, float]:
    """
    Restore full training state from checkpoint.
    Returns (epoch, best_f1).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if ema_model and 'ema_state_dict' in ckpt:
        if hasattr(ema_model, 'module'):
            ema_model.module.load_state_dict(ckpt['ema_state_dict'])
        else:
            ema_model.load_state_dict(ckpt['ema_state_dict'])
    # Restore RNG states [FIX I-2: includes CUDA RNG]
    # [FIX: ByteTensor cast] PyTorch requires RNG state as ByteTensor
    if 'rng_state_cpu' in ckpt:
        rng = ckpt['rng_state_cpu']
        torch.set_rng_state(rng.byte() if hasattr(rng, 'byte') else rng)
    elif 'rng_state' in ckpt:
        rng = ckpt['rng_state']
        torch.set_rng_state(rng.byte() if hasattr(rng, 'byte') else rng)
    if 'rng_state_gpu' in ckpt and torch.cuda.is_available():
        rng_gpu = ckpt['rng_state_gpu']
        torch.cuda.set_rng_state(rng_gpu.byte() if hasattr(rng_gpu, 'byte') else rng_gpu)
    if 'numpy_rng_state' in ckpt:
        np.random.set_state(ckpt['numpy_rng_state'])
    if 'python_rng_state' in ckpt:
        random.setstate(ckpt['python_rng_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_f1', 0.0)


def find_latest_checkpoint(ckpt_dir: str, prefix: str = '') -> Optional[str]:
    """Find most recent checkpoint file by epoch number (not alphabetical)."""
    import re
    pattern = os.path.join(ckpt_dir, f'{prefix}*.pt')
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    # [FIX I-3] Sort by epoch number extracted from filename, not alphabetically
    # epoch_9.pt must sort before epoch_10.pt
    def _epoch_key(path):
        m = re.search(r'epoch[_-]?(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else -1
    ckpts.sort(key=_epoch_key)
    return ckpts[-1]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════════════════

def setup_ema(model: nn.Module, decay: float = 0.9999, device='cuda'):
    """
    Create an EMA shadow model using timm's ModelEmaV2.
    Falls back to manual EMA if timm version incompatible.
    """
    try:
        from timm.utils import ModelEmaV2
        ema = ModelEmaV2(model, decay=decay, device=device)
        return ema
    except ImportError:
        # Manual fallback
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        return _ManualEMA(ema_model, decay)


class _ManualEMA:
    """Fallback EMA implementation if timm ModelEmaV2 unavailable."""
    def __init__(self, model, decay):
        self.module = model
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()


def reset_ema(ema_model, source_model: nn.Module, new_decay: float = None):
    """
    Reinitialise EMA from source model's CURRENT weights.

    Two use cases:
    1. After epoch 0 (EMA warmup): the random-init head has now trained for 1 epoch,
       so re-seeding EMA from these partially-trained weights gives a MUCH better
       starting point than the random init that setup_ema() captured.
       [FIX: Router EMA was broken because decay=0.9999 retained 56% random weights
       after only 5800 steps. Re-seeding after epoch 0 eliminates this.]

    2. Stage 1 → Stage 2 transition: reset EMA from Stage 1 best weights and
       optionally change decay rate (Stage 2 is shorter, needs faster EMA).

    Args:
        ema_model: the EMA model to reinitialise
        source_model: the training model whose current weights to copy
        new_decay: if provided, update the EMA decay rate
    """
    if hasattr(ema_model, 'module'):
        ema_model.module.load_state_dict(source_model.state_dict())
        if new_decay is not None:
            ema_model.decay = new_decay
    elif hasattr(ema_model, 'decay'):
        ema_model.module.load_state_dict(source_model.state_dict())
        if new_decay is not None:
            ema_model.decay = new_decay


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: GREEDY MODEL SOUP
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_soup(checkpoint_paths: List[str], model: nn.Module,
                val_loader, evaluate_fn, device='cuda',
                improvement_threshold: float = 1e-4) -> Tuple[dict, float, List[str]]:
    """
    Greedy model soup: average checkpoint weights, keep only improvements.

    Args:
        checkpoint_paths: list of .pt checkpoint file paths (sorted by val F1, best first)
        model: model architecture (used for loading state dicts)
        val_loader: validation DataLoader
        evaluate_fn: callable(model, val_loader, device) -> float (returns macro F1)
        device: CUDA device
        improvement_threshold: minimum F1 improvement to include a checkpoint

    Returns:
        (soup_state_dict, final_f1, selected_checkpoint_paths)
    """
    # Load best checkpoint as starting soup
    ckpt0 = torch.load(checkpoint_paths[0], map_location=device, weights_only=False)
    soup_state = ckpt0.get('ema_state_dict', ckpt0.get('model_state_dict'))
    model.load_state_dict(soup_state)
    model.to(device)
    model.eval()
    current_f1 = evaluate_fn(model, val_loader, device)
    selected = [checkpoint_paths[0]]
    print(f'  Soup start: {os.path.basename(checkpoint_paths[0])} F1={current_f1:.4f}')

    for path in checkpoint_paths[1:]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        candidate_state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict'))

        # Blend: average soup_state and candidate_state
        n = len(selected)
        blended = {}
        for key in soup_state:
            # [FIX I-9] Skip integer buffers (num_batches_tracked, etc.)
            if soup_state[key].dtype in (torch.int32, torch.int64, torch.uint8, torch.bool):
                blended[key] = candidate_state[key]  # take from candidate, don't average
            else:
                blended[key] = (soup_state[key].float() * n + candidate_state[key].float()) / (n + 1)
                blended[key] = blended[key].to(soup_state[key].dtype)

        model.load_state_dict(blended)
        model.eval()
        blended_f1 = evaluate_fn(model, val_loader, device)

        if blended_f1 > current_f1 + improvement_threshold:
            soup_state = blended
            current_f1 = blended_f1
            selected.append(path)
            print(f'  + Added {os.path.basename(path)} -> F1={blended_f1:.4f}')
        else:
            print(f'  - Skipped {os.path.basename(path)} (F1={blended_f1:.4f}, no improvement)')

    print(f'  Soup final: {len(selected)} checkpoints, F1={current_f1:.4f}')
    return soup_state, current_f1, selected


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D: CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_aps_thresholds(model: nn.Module, cal_loader,
                           device='cuda', alpha: float = 0.05) -> float:
    """
    Compute APS (Adaptive Prediction Sets) threshold for (1-alpha) coverage.
    Uses held-out calibration set.

    Returns q_hat: threshold for prediction set construction.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in cal_loader:
            images = batch[0].to(device)
            labels = batch[1]
            logits = model(images)
            # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    probs = np.vstack(all_probs)
    labels = np.array(all_labels)
    n = len(labels)

    # For each calibration sample, compute conformity score
    scores = np.zeros(n)
    for i in range(n):
        sorted_idx = np.argsort(probs[i])[::-1]
        sorted_probs = probs[i][sorted_idx]
        true_class = labels[i]
        rank = np.where(sorted_idx == true_class)[0][0]
        scores[i] = sorted_probs[:rank + 1].sum()

    # Quantile for coverage guarantee
    # [FIX C-3] NumPy 2.0 renamed 'interpolation' to 'method'
    _q_level = np.ceil((n + 1) * (1 - alpha)) / n
    _q_level = min(_q_level, 1.0)  # clamp to valid range
    try:
        q_hat = np.quantile(scores, _q_level, method='higher')
    except TypeError:
        q_hat = np.quantile(scores, _q_level, interpolation='higher')
    return float(q_hat)


def predict_with_aps(probs: np.ndarray, q_hat: float) -> List[int]:
    """
    Given softmax probabilities and APS threshold, return prediction set.
    Accumulates classes by descending probability until cumsum >= q_hat.
    """
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    k = int(np.searchsorted(cumulative, q_hat)) + 1
    return sorted_idx[:k].tolist()


def compute_routing_thresholds(model: nn.Module, cal_loader,
                               crop_names: List[str],
                               device='cuda', alpha: float = 0.05) -> Dict[str, float]:
    """
    Compute per-crop abstention thresholds for the router.
    Returns dict {crop_name: min_confidence_for_routing}.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in cal_loader:
            images = batch[0].to(device)
            labels = batch[1]
            logits = model(images)
            # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    probs = np.vstack(all_probs)
    labels = np.array(all_labels)

    thresholds = {}
    for crop_idx, crop_name in enumerate(crop_names):
        mask = labels == crop_idx
        if not mask.any():
            thresholds[crop_name] = 0.5  # default
            continue
        # Confidence of correct predictions for this crop
        crop_probs = probs[mask]
        correct_conf = crop_probs[np.arange(mask.sum()), crop_idx]
        # Threshold at alpha quantile (bottom alpha% of correct confidences)
        threshold = float(np.quantile(correct_conf, alpha))
        thresholds[crop_name] = threshold

    return thresholds


# ═══════════════════════════════════════════════════════════════════════════
# SECTION E: AUGMENTATION PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

def get_augmentation_pipeline(img_size: int = 224, curl_class: bool = False):
    """
    Returns albumentations pipeline.
    Standard: AugMix + geometric augmentations.
    Curl classes: additionally applies GridDistortion + ElasticTransform.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    base_transforms = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ]

    if curl_class:
        base_transforms.extend([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        ])

    base_transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return A.Compose(base_transforms)


def get_eval_transform(img_size: int = 224):
    """Deterministic evaluation/inference transform."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# SECTION F: CUTMIX
# ═══════════════════════════════════════════════════════════════════════════

def apply_cutmix(images: torch.Tensor, labels: torch.Tensor,
                 alpha: float = 1.0, thin_class_indices: List[int] = None,
                 probability: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply CutMix to a batch, optionally only for thin classes.

    Returns (mixed_images, labels_a, labels_b, lam) where lam is the
    area fraction of the original image that remains.
    """
    if random.random() > probability:
        return images, labels, labels, 1.0

    batch_size = images.size(0)

    # [FIX I-1] Only apply CutMix when thin classes are present in batch
    if thin_class_indices is not None and len(thin_class_indices) > 0:
        has_thin = any((labels == idx).any() for idx in thin_class_indices)
        if not has_thin:
            return images, labels, labels, 1.0  # no thin class in batch, skip
    lam = np.random.beta(alpha, alpha)

    # Generate random bounding box
    W, H = images.size(3), images.size(2)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Shuffle indices for mixing
    rand_index = torch.randperm(batch_size)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    # Adjust lambda for actual box area
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)

    return mixed, labels, labels[rand_index], lam


# ═══════════════════════════════════════════════════════════════════════════
# SECTION G: CAPSICUM MONITORING
# ═══════════════════════════════════════════════════════════════════════════

def track_subsource_f1(all_preds: np.ndarray, all_labels: np.ndarray,
                       all_sources: List[str], target_class_idx: int,
                       capsicum_source: str = 'multi_D') -> Tuple[float, float, float]:
    """
    Track F1 separately for Capsicum vs real chilli_healthy.

    Returns (f1_capsicum, f1_real, gap).
    """
    from sklearn.metrics import f1_score

    target_mask = all_labels == target_class_idx
    if not target_mask.any():
        return 0.0, 0.0, 0.0

    sources = np.array(all_sources)
    cap_mask = target_mask & (sources == capsicum_source)
    real_mask = target_mask & (sources != capsicum_source)

    pred_binary = (all_preds[:, target_class_idx] > 0.5).astype(int)
    label_binary = (all_labels == target_class_idx).astype(int)

    f1_cap = 0.0
    f1_real = 0.0

    if cap_mask.any():
        f1_cap = f1_score(label_binary[cap_mask], pred_binary[cap_mask],
                          zero_division=0)
    if real_mask.any():
        f1_real = f1_score(label_binary[real_mask], pred_binary[real_mask],
                           zero_division=0)

    return f1_cap, f1_real, f1_cap - f1_real


# ═══════════════════════════════════════════════════════════════════════════
# SECTION H: ROLLBACK
# ═══════════════════════════════════════════════════════════════════════════

def should_rollback(current_f1: float, rollback_f1: float,
                    threshold: float = 0.95) -> bool:
    """Returns True if current F1 has degraded beyond threshold."""
    return current_f1 < rollback_f1 * threshold


def apply_rollback(model: nn.Module, optimizer, ema_model,
                   rollback_path: str, device='cuda',
                   lr_reduction: float = 0.2):
    """
    Load rollback checkpoint and reduce LR by lr_reduction factor.
    """
    epoch, best_f1 = load_checkpoint(rollback_path, model, ema_model,
                                     optimizer, device=device)
    for pg in optimizer.param_groups:
        pg['lr'] *= lr_reduction
    print(f'ROLLBACK: loaded epoch {epoch}, LR reduced by {lr_reduction}x')
    return epoch


# ═══════════════════════════════════════════════════════════════════════════
# SECTION I: SAMPLING WEIGHT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def verify_sampling_weights(df, weights: np.ndarray, targets: dict,
                            crop_col: str = 'crop',
                            source_col: str = 'source_dataset',
                            field_col: str = 'is_field_photo',
                            epochs: int = 30,
                            samples_per_epoch: int = 29200):
    """
    Monte Carlo verification of sampling weights.
    Simulates multiple epochs and prints mean +/- std per bucket.
    Returns True if all buckets within 10% of targets.
    """
    print(f'Monte Carlo sampling verification ({epochs} epochs, {samples_per_epoch} samples/epoch)')
    all_ok = True

    for crop, target in targets.items():
        crop_mask = df[crop_col] == crop
        counts_total = []
        counts_field = []
        counts_scidb = []

        for ep in range(epochs):
            sampler = WeightedRandomSampler(
                weights, num_samples=samples_per_epoch, replacement=True,
                # [FIX I-5] Use deterministic hash, not Python's randomized hash()
                generator=torch.Generator().manual_seed(ep * 1000 + sum(ord(c) for c in crop) % 1000)
            )
            idx = list(sampler)
            sampled = df.iloc[idx]
            crop_sampled = sampled[sampled[crop_col] == crop]
            counts_total.append(len(crop_sampled))
            counts_field.append(
                crop_sampled[field_col].astype(str).str.lower().isin(['true']).sum()
            )
            counts_scidb.append(
                (crop_sampled[source_col] == 'scidb_data_merged').sum()
            )

        mean_t, std_t = np.mean(counts_total), np.std(counts_total)
        mean_f, std_f = np.mean(counts_field), np.std(counts_field)
        mean_s, std_s = np.mean(counts_scidb), np.std(counts_scidb)

        target_total = target.get('total', 0)
        pct_off = abs(mean_t - target_total) / max(target_total, 1) * 100

        status = 'OK' if pct_off < 10 else 'WARN'
        if pct_off >= 10:
            all_ok = False

        print(f'  {crop:<10}: total={mean_t:.0f}+/-{std_t:.0f} (target={target_total}, '
              f'off={pct_off:.1f}%) [{status}]')
        print(f'             field={mean_f:.0f}+/-{std_f:.0f}, scidb={mean_s:.0f}+/-{std_s:.0f}')

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# SECTION J: DATA LOADING
# [FIX C-8] This section was entirely missing from the original train_utils.py
# ═══════════════════════════════════════════════════════════════════════════

def load_split(csv_path: str, split_name: str = 'train',
               exclude_classes: List[str] = None,
               exclude_sources: List[str] = None,
               exclude_indices: List[int] = None) -> 'pd.DataFrame':
    """
    Load a specific split from a unified source map CSV.

    Args:
        csv_path: path to unified CSV (model2/model3/router)
        split_name: 'train', 'val', 'soup_selection', 'final_val', 'conformal'
                    If the CSV has no 'split' column, returns all rows.
        exclude_classes: list of class names to drop (e.g., quarantined classes)
        exclude_sources: list of source_dataset values to drop
        exclude_indices: list of row indices to exclude (e.g., conformal indices)

    Returns:
        DataFrame with all columns preserved, filtered by criteria.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Filter by split if column exists
    if 'split' in df.columns and split_name:
        df = df[df['split'] == split_name].copy()

    # Exclude quarantined/dropped classes
    if exclude_classes and 'class_name' in df.columns:
        df = df[~df['class_name'].isin(exclude_classes)]

    # Exclude specific sources
    if exclude_sources and 'source_dataset' in df.columns:
        df = df[~df['source_dataset'].isin(exclude_sources)]

    # Exclude specific row indices
    if exclude_indices:
        df = df[~df.index.isin(exclude_indices)]

    df = df.reset_index(drop=True)
    return df


def resolution_aware_rollback_check(current_f1: float, baseline_f1: float,
                                    current_resolution: int,
                                    baseline_resolution: int,
                                    threshold: float = 0.95) -> bool:
    """
    [FIX I-6] Resolution-aware rollback for Model 2 progressive resizing.

    Do NOT compare F1 across different resolutions — 128px F1 is always lower
    than 384px F1 due to information loss, and comparing them falsely triggers
    rollback at resolution transitions.

    Returns True if rollback should be triggered (same resolution AND F1 dropped).
    Returns False at resolution transitions (expected F1 change).
    """
    if current_resolution != baseline_resolution:
        return False  # resolution transition — F1 change expected, no rollback
    return current_f1 < baseline_f1 * threshold


def evaluate_with_subsource(model: nn.Module, val_loader, device: str,
                            class_names: List[str],
                            source_labels: List[str],
                            target_class: str,
                            subsource_name: str) -> Tuple[float, float, float]:
    """
    [FIX I-6] Evaluate model and return per-subsource F1 breakdown.
    Used for Capsicum monitoring: tracks F1 separately for Capsicum-source
    vs real-chilli-source within chilli_healthy.

    Returns (f1_subsource, f1_other, gap).
    """
    from sklearn.metrics import f1_score

    model.eval()
    all_preds = []
    all_labels = []

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32

    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            with torch.autocast(device_type, dtype=amp_dtype, enabled=(device_type == 'cuda')):
                logits = model(images)
            # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
            preds = logits.float().argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch[1].numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    sources = np.array(source_labels[:len(all_labels)])

    target_idx = class_names.index(target_class) if target_class in class_names else -1
    if target_idx < 0:
        return 0.0, 0.0, 0.0

    target_mask = all_labels == target_idx
    sub_mask = target_mask & (sources == subsource_name)
    other_mask = target_mask & (sources != subsource_name)

    f1_sub = f1_score(all_labels[sub_mask] == target_idx,
                      all_preds[sub_mask] == target_idx,
                      zero_division=0) if sub_mask.any() else 0.0
    f1_other = f1_score(all_labels[other_mask] == target_idx,
                        all_preds[other_mask] == target_idx,
                        zero_division=0) if other_mask.any() else 0.0

    return f1_sub, f1_other, f1_sub - f1_other


# ═══════════════════════════════════════════════════════════════════════════
# SECTION K: SELF-DISTILLATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_soft_labels(model: nn.Module, dataset, device='cuda',
                         agreement_threshold: float = 0.70,
                         temperature: float = 3.0,
                         exclude_indices: set = None,
                         num_classes: int = 10,
                         batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate filtered soft labels for self-distillation.

    [FIX C-5] Returns FULL-LENGTH arrays (len=len(dataset)), with one-hot
    hard labels at excluded indices. Caller can use arrays directly without
    index remapping.

    Returns:
        soft_labels: (N, num_classes) array — soft probs for agreed images,
                     one-hot for disagreed/low-confidence/excluded images
        use_soft: (N,) boolean array — True where soft label is used
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()
    N = len(dataset)

    # [FIX C-5] Build full-length output arrays; excluded indices get one-hot
    full_labels = np.zeros((N, num_classes), dtype=np.float32)
    full_use_soft = np.zeros(N, dtype=bool)

    # Determine which indices to process
    if exclude_indices:
        valid_idx = [i for i in range(N) if i not in exclude_indices]
    else:
        valid_idx = list(range(N))

    subset = Subset(dataset, valid_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_hard_labels = []

    # [FIX C-7] Device-aware autocast
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32

    for batch in loader:
        images = batch[0].to(device)
        with torch.autocast(device_type, dtype=amp_dtype, enabled=(device_type == 'cuda')):
            logits = model(images)
        # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
        probs = torch.softmax(logits.float() / temperature, dim=1).cpu().numpy()
        all_probs.append(probs)
        if len(batch) > 1:
            all_hard_labels.extend(batch[1].numpy())

    soft_probs = np.vstack(all_probs)
    hard_labels = np.array(all_hard_labels)

    # Agreement mask
    pred_classes = soft_probs.argmax(axis=1)
    agreement = pred_classes == hard_labels
    confidence = soft_probs.max(axis=1)
    confident = confidence > agreement_threshold
    use_soft_subset = agreement & confident

    # Per-class agreement check
    for cls in range(num_classes):
        cls_mask = hard_labels == cls
        if cls_mask.sum() == 0:
            continue
        cls_agreement = agreement[cls_mask].mean()
        if cls_agreement < 0.50:
            use_soft_subset[cls_mask] = False
            print(f'  Class {cls}: agreement={cls_agreement:.2f} < 0.50, '
                  f'using hard labels for all {cls_mask.sum()} images')

    # [FIX C-5] Map back to full-length arrays using valid_idx
    for pos, orig_idx in enumerate(valid_idx):
        if use_soft_subset[pos]:
            full_labels[orig_idx] = soft_probs[pos]
            full_use_soft[orig_idx] = True
        else:
            full_labels[orig_idx] = np.eye(num_classes)[hard_labels[pos]]

    # Fill excluded indices with one-hot hard labels from dataset
    if exclude_indices:
        for idx in exclude_indices:
            if idx < N:
                _, label = dataset[idx][0], dataset[idx][1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                full_labels[idx] = np.eye(num_classes)[label]

    overall_rate = full_use_soft.mean()
    print(f'  Soft labels: {full_use_soft.sum()} ({overall_rate:.1%}), '
          f'Hard labels: {(~full_use_soft).sum()} ({1-overall_rate:.1%})')

    return full_labels, full_use_soft


# ═══════════════════════════════════════════════════════════════════════════
# SECTION L: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device='cuda',
             class_names: List[str] = None,
             num_classes: int = 10) -> Dict[str, float]:
    """
    Full evaluation: per-class F1, macro F1, crop accuracy.
    Returns dict of metrics.
    """
    from sklearn.metrics import f1_score

    model.eval()
    all_probs = []
    all_labels = []

    # [FIX C-7] Device-aware autocast
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32

    for batch in val_loader:
        images = batch[0].to(device)
        labels = batch[1]
        with torch.autocast(device_type, dtype=amp_dtype, enabled=(device_type == 'cuda')):
            logits = model(images)
        # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())

    probs = np.vstack(all_probs)
    labels = np.array(all_labels)
    preds = probs.argmax(axis=1)

    # [FIX CRITICAL: Pessimistic Audit] labels= parameter ensures all classes are
    # represented in the output even if absent from predictions. Without it, per_class
    # array can be shorter than num_classes, misattributing F1 to wrong diseases.
    all_class_labels = list(range(num_classes))
    macro_f1 = f1_score(labels, preds, average='macro', labels=all_class_labels, zero_division=0)
    per_class = f1_score(labels, preds, average=None, labels=all_class_labels, zero_division=0)

    metrics = {'macro_f1': float(macro_f1)}
    if class_names:
        for i, name in enumerate(class_names):
            if i < len(per_class):
                metrics[f'f1_{name}'] = float(per_class[i])

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# SECTION M: COMPILATION (no-op on Windows)
# ═══════════════════════════════════════════════════════════════════════════

def compile_model_safe(model: nn.Module, mode: str = 'default'):
    """
    Attempt torch.compile. Returns compiled model or original on failure.
    On Windows without Triton, this is a no-op.
    """
    try:
        compiled = torch.compile(model, mode=mode)
        print(f'torch.compile: ENABLED (mode={mode})')
        return compiled
    except Exception as e:
        print(f'torch.compile: unavailable ({e}). Running without compilation.')
        return model


# ═══════════════════════════════════════════════════════════════════════════
# SECTION N: TTA (Test-Time Augmentation)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_tta(model: nn.Module, image: torch.Tensor,
                     device='cuda', num_views: int = 5) -> np.ndarray:
    """
    TTA prediction. Returns averaged softmax probabilities.

    5 views (router): original, h-flip, v-flip, brightness jitter, ±15deg rotation
    8 views (specialists): above 5 + 3 random crops
    """
    import torchvision.transforms.functional as TF

    model.eval()
    views = [image]

    # H-flip
    views.append(torch.flip(image, dims=[3]))
    # V-flip
    views.append(torch.flip(image, dims=[2]))
    # [FIX I-12] Brightness jitter on already-normalized tensors
    # Don't clamp — normalized values are in ~[-2.5, 2.5], clamping to [0,1] destroys them
    factor = 1.0 + random.uniform(-0.15, 0.15)
    views.append(image * factor)
    # Rotation ±15 degrees
    angle = random.choice([-15, 15])
    views.append(TF.rotate(image, angle))

    if num_views >= 8:
        # 3 additional random crops (center crop at 90%, 85%, 80% of original)
        for crop_frac in [0.90, 0.85, 0.80]:
            h, w = image.shape[-2:]
            ch, cw = int(h * crop_frac), int(w * crop_frac)
            top = (h - ch) // 2
            left = (w - cw) // 2
            cropped = image[:, :, top:top+ch, left:left+cw]
            cropped = F.interpolate(cropped, size=(h, w), mode='bilinear',
                                    align_corners=False)
            views.append(cropped)

    all_probs = []
    for v in views[:num_views]:
        logits = model(v.to(device))
        # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION O: MC DROPOUT
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_mc_dropout(model: nn.Module, image: torch.Tensor,
                            device='cuda', passes: int = 5) -> Tuple[np.ndarray, float]:
    """
    MC Dropout uncertainty estimation.
    Sets only Dropout layers to train mode (not BatchNorm/LayerNorm).
    Returns (mean_probs, uncertainty_std).
    """
    model.eval()
    # Enable dropout only
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    mc_probs = []
    for _ in range(passes):
        logits = model(image.to(device))
        # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        mc_probs.append(probs)

    # Restore full eval
    model.eval()

    mc_array = np.array(mc_probs)  # (passes, batch, classes)
    mean_probs = mc_array.mean(axis=0)
    std_probs = mc_array.std(axis=0)
    uncertainty = float(std_probs.mean())

    return mean_probs, uncertainty


# ═══════════════════════════════════════════════════════════════════════════
# SECTION P: DINO ATTENTION MAPS
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_dino_attention_map(model: nn.Module, image: torch.Tensor,
                               device='cuda', block_idx: int = -1,
                               img_size: int = 224, patch_size: int = 14):
    """
    Extract DINO self-attention map from DINOv2-ViT.

    Uses CLS token attention to patch tokens, averaged across all heads.
    Returns 2D attention map at (grid_size, grid_size) resolution.
    """
    model.eval()

    # Hook to capture attention weights
    attentions = []

    def attention_hook(module, input, output):
        # timm ViT attention: output is tuple (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attentions.append(output[1].detach().cpu())

    # Register hook on target block's attention
    # For timm DINOv2: model.blocks[block_idx].attn
    target_block = None
    if hasattr(model, 'blocks'):
        target_block = model.blocks[block_idx].attn
    elif hasattr(model, 'base_model'):
        # peft-wrapped model
        base = model.base_model.model if hasattr(model, 'base_model') else model
        if hasattr(base, 'blocks'):
            target_block = base.blocks[block_idx].attn

    if target_block is None:
        warnings.warn('Could not find attention block for DINO attention map')
        grid_size = img_size // patch_size
        return np.ones((grid_size, grid_size)) * 0.5

    hook = target_block.register_forward_hook(attention_hook)

    try:
        _ = model(image.to(device))
    finally:
        hook.remove()

    if not attentions:
        grid_size = img_size // patch_size
        return np.ones((grid_size, grid_size)) * 0.5

    # attention shape: (batch, num_heads, num_tokens, num_tokens)
    attn = attentions[0][0]  # first image in batch
    # CLS token (index 0) attention to all patch tokens
    num_heads = attn.shape[0]
    grid_size = img_size // patch_size  # 224/14 = 16

    # Average across all heads
    cls_attn = attn[:, 0, 1:].mean(dim=0)  # (num_patches,)
    cls_attn = cls_attn.numpy().reshape(grid_size, grid_size)

    # Normalise to [0, 1]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    return cls_attn


# ═══════════════════════════════════════════════════════════════════════════
# SECTION Q: SUPCON LOSS
# ═══════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    Pulls same-class embeddings together, pushes different-class apart.

    Used in Model 2 Stage 1 only (epochs 1-15, not at 384px).
    Requires class-balanced batches (>=2 images per class).
    """
    def __init__(self, temperature: float = 0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, embed_dim) L2-normalised embeddings
            labels: (batch_size,) class labels
        """
        device = features.device
        batch_size = features.shape[0]

        # L2 normalise
        features = F.normalize(features, dim=1)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask: same class = 1, different class = 0, self = 0
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Log-sum-exp of all negatives
        exp_logits = torch.exp(logits)
        # Exclude self
        self_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        exp_logits = exp_logits * self_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-prob over positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -mean_log_prob.mean()
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# SECTION R: ASAM WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class ASAMWrapper:
    """
    Adaptive Sharpness-Aware Minimization (ASAM) wrapper.
    Wraps any base optimizer. Two-pass per step:
      1. Perturb weights in steepest gradient direction (ascent step)
      2. Compute gradients at perturbed point (descent step)
    Finds flatter minima that generalise better.

    Used in Model 2 Stage 1 at 224px (rho=0.10) and 384px (rho=0.20).
    """
    def __init__(self, optimizer, model: nn.Module, rho: float = 0.10):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        """Perturb weights in the direction of steepest loss increase."""
        grads = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Adaptive scaling: normalise by weight magnitude
                scale = torch.abs(p.data).clamp(min=1e-12)
                e_w = p.grad * scale * scale
                grads.append(e_w.view(-1))

        if not grads:
            return

        grad_norm = torch.cat(grads).norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_scale = torch.abs(p.data).clamp(min=1e-12)
                e_w = p.grad * param_scale * param_scale * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def descent_step(self):
        """Restore weights and apply the optimizer step."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.state:
                    p.sub_(self.state[p]['e_w'])
        self.optimizer.step()
        self.state.clear()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION S: PARAMETER GROUPS
# ═══════════════════════════════════════════════════════════════════════════

def get_param_groups_no_decay(model: nn.Module,
                              weight_decay: float = 0.01) -> List[dict]:
    """
    Create parameter groups excluding bias and LayerNorm from weight decay.
    Standard practice for ViT and ConvNeXt training.
    """
    no_decay_keywords = {'bias', 'LayerNorm.weight', 'layernorm.weight',
                         'ln.weight', 'norm.weight'}
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


def get_llrd_param_groups(model: nn.Module, base_lr: float,
                          decay_rate: float = 0.90,
                          weight_decay: float = 0.01) -> List[dict]:
    """
    GENERIC Layer-wise learning rate decay for ConvNeXt.
    NOTE: For Model 2, prefer Model2ConvNeXt.get_llrd_param_groups() in models.py
    which handles the transformers DINOv3-ConvNeXt stage structure correctly.
    Head gets base_lr, each deeper block gets lr * decay_rate^depth.
    """
    # [FIX I-4] Include 'ln.weight' to match get_param_groups_no_decay
    no_decay = {'bias', 'LayerNorm.weight', 'layernorm.weight', 'norm.weight', 'ln.weight'}
    param_groups = []

    # Head: full base_lr
    head_params = {'decay': [], 'no_decay': []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name or 'classifier' in name:
            if any(nd in name for nd in no_decay):
                head_params['no_decay'].append(param)
            else:
                head_params['decay'].append(param)

    if head_params['decay']:
        param_groups.append({'params': head_params['decay'],
                            'lr': base_lr, 'weight_decay': weight_decay})
    if head_params['no_decay']:
        param_groups.append({'params': head_params['no_decay'],
                            'lr': base_lr, 'weight_decay': 0.0})

    # Backbone stages: decaying LR
    if hasattr(model, 'stages'):
        stages = list(model.stages)
    elif hasattr(model, 'features') and hasattr(model.features, '__len__'):
        stages = list(model.features)
    else:
        # Fallback: treat all non-head params as one group
        other_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'head' not in n and 'classifier' not in n]
        if other_params:
            param_groups.append({'params': other_params,
                                'lr': base_lr * decay_rate,
                                'weight_decay': weight_decay})
        return param_groups

    num_stages = len(stages)
    for i, stage in enumerate(reversed(stages)):
        lr = base_lr * (decay_rate ** (i + 1))
        stage_decay = []
        stage_no_decay = []
        for name, param in stage.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                stage_no_decay.append(param)
            else:
                stage_decay.append(param)
        if stage_decay:
            param_groups.append({'params': stage_decay,
                                'lr': lr, 'weight_decay': weight_decay})
        if stage_no_decay:
            param_groups.append({'params': stage_no_decay,
                                'lr': lr, 'weight_decay': 0.0})

    return param_groups


# ═══════════════════════════════════════════════════════════════════════════
# SECTION T: STAGE TRANSITIONS
# ═══════════════════════════════════════════════════════════════════════════

def freeze_backbone(model: nn.Module, freeze_mode: str = 'backbone_only',
                    head_keywords=('head', 'classifier', 'fc', 'film')):
    """
    Freeze model parameters with LoRA-aware modes.

    [FIX C-6] Three freeze modes for different training stages:
      'backbone_only': freeze base weights, keep LoRA + head + FiLM trainable
        Use for: Model 3 Stage 1 (LoRA adapts, head trains)
      'backbone_and_lora': freeze backbone AND LoRA, keep head + FiLM only
        Use for: Model 3 Stage 2 (only head refines on balanced data)
      'all_except_head': freeze everything except head keywords
        Use for: Model 2 Stage 2 (ConvNeXt frozen, head trains with CutMix)
    """
    for name, param in model.named_parameters():
        is_head = any(kw in name for kw in head_keywords)
        is_lora = 'lora_' in name

        if freeze_mode == 'backbone_only':
            param.requires_grad = is_head or is_lora
        elif freeze_mode == 'backbone_and_lora':
            param.requires_grad = is_head
        elif freeze_mode == 'all_except_head':
            param.requires_grad = is_head
        else:
            raise ValueError(f'Unknown freeze_mode: {freeze_mode}')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'freeze_mode={freeze_mode}: {trainable/1e3:.1f}K trainable / {total/1e6:.1f}M total')


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def switch_to_stage2(model: nn.Module, optimizer_cls=torch.optim.AdamW,
                     head_lr: float = 1e-4, weight_decay: float = 0.01,
                     freeze_mode: str = 'all_except_head',
                     head_keywords=('head', 'classifier', 'fc')):
    """
    Transition from Stage 1 to Stage 2:
    - Freeze backbone (mode depends on model type)
    - Reinitialise optimizer for trainable params only

    freeze_mode options:
      'all_except_head': Model 2 Stage 2 (ConvNeXt frozen, head trains)
      'backbone_and_lora': Model 3 Stage 2 (backbone+LoRA frozen, head+FiLM train)
      'backbone_only': keeps LoRA trainable (not typical for Stage 2)
    """
    # [FIX Round 2] Pass freeze_mode as keyword, not head_keywords as positional
    freeze_backbone(model, freeze_mode=freeze_mode, head_keywords=head_keywords)
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad]
    optimizer = optimizer_cls(head_params, lr=head_lr, weight_decay=weight_decay)
    return optimizer


# ═══════════════════════════════════════════════════════════════════════════
# SECTION U: FiLM MODULE
# ═══════════════════════════════════════════════════════════════════════════

class FiLMWrapper(nn.Module):
    """
    Feature-wise Linear Modulation on LoRA adapter outputs.
    Wraps a peft-modified layer: modulates its output based on crop identity.

    crop_embedding_dim=4 is sufficient for binary (tomato/chilli) signal.
    gamma initialised near 1 (near-identity), beta near 0.
    """
    def __init__(self, wrapped_layer: nn.Module, crop_embedding_dim: int = 4,
                 num_crops: int = 2, output_dim: int = 384):
        super().__init__()
        self.wrapped_layer = wrapped_layer
        self.crop_embed = nn.Embedding(num_crops, crop_embedding_dim)
        self.gamma_proj = nn.Linear(crop_embedding_dim, output_dim)
        self.beta_proj = nn.Linear(crop_embedding_dim, output_dim)

        # Initialise near-identity
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x, crop_ids=None):
        out = self.wrapped_layer(x)
        if crop_ids is not None:
            emb = self.crop_embed(crop_ids)  # (batch, crop_embed_dim)
            gamma = 1.0 + self.gamma_proj(emb)  # scale near 1
            beta = self.beta_proj(emb)           # shift near 0
            # Expand for token dimension if needed
            if out.dim() == 3 and gamma.dim() == 2:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            out = out * gamma + beta
        return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION V: MIXED LOSS (Soft + Hard targets)
# ═══════════════════════════════════════════════════════════════════════════

def soft_hard_mixed_loss(logits: torch.Tensor, soft_targets: torch.Tensor,
                         is_soft: torch.Tensor, temperature: float = 3.0):
    """
    Combined loss: KL divergence for soft-label images,
    CrossEntropy for hard-label images.

    Args:
        logits: (batch, num_classes) model output
        soft_targets: (batch, num_classes) soft probability targets
        is_soft: (batch,) boolean tensor — True = use KL, False = use CE
        temperature: softening temperature for KL divergence
    """
    loss = torch.zeros(logits.shape[0], device=logits.device)

    soft_mask = is_soft.bool()
    hard_mask = ~soft_mask

    if soft_mask.any():
        student_log_probs = F.log_softmax(logits[soft_mask] / temperature, dim=1)
        kl = F.kl_div(student_log_probs, soft_targets[soft_mask],
                      reduction='none').sum(dim=1)
        loss[soft_mask] = kl * (temperature ** 2)

    if hard_mask.any():
        hard_labels = soft_targets[hard_mask].argmax(dim=1)
        ce = F.cross_entropy(logits[hard_mask], hard_labels, reduction='none')
        loss[hard_mask] = ce

    return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION W: ENS CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

def compute_ens_class_weights(class_counts: List[int],
                              beta: float = 0.9999) -> torch.Tensor:
    """
    Effective Number of Samples (ENS) class weighting.
    Cui et al. 2019: weight = (1 - beta) / (1 - beta^n)

    More balanced than raw inverse frequency. Prevents extreme weights
    for very small classes while still upweighting them.
    """
    weights = []
    for n in class_counts:
        if n == 0:
            weights.append(0.0)
        else:
            effective_n = (1.0 - beta ** n) / (1.0 - beta)
            weights.append(1.0 / effective_n)

    # Normalise so weights sum to num_classes
    total = sum(weights)
    num_classes = len(class_counts)
    weights = [w * num_classes / total for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION X: GRADIENT ACCUMULATION (correct loss scaling)
# Bug confirmed: https://unsloth.ai/blog/gradient
#                https://huggingface.co/blog/gradient_accumulation
# ═══════════════════════════════════════════════════════════════════════════

def gradient_accumulation_step(model, micro_batches, loss_fn, optimizer,
                               grad_accum_steps: int, scaler=None,
                               max_grad_norm: float = 1.0):
    """
    Correct gradient accumulation with loss scaling.
    Handles both regular optimizers AND ASAM two-pass protocol.

    ASAM protocol (when optimizer is ASAMWrapper):
      Pass 1: accumulate gradients at current weights → ascent_step (perturb)
      Pass 2: accumulate gradients at perturbed weights → descent_step (restore + step)
    This ensures ASAM samples the loss landscape at the perturbed point.

    Regular optimizer: single pass with (loss / G).backward() scaling.

    Args:
        model: training model
        micro_batches: list of (images, labels, ...) tuples
        loss_fn: callable(model, batch) -> loss tensor
        optimizer: optimizer (or ASAMWrapper)
        grad_accum_steps: number of micro-batches to accumulate
        scaler: GradScaler for mixed precision (or None)
        max_grad_norm: gradient clipping threshold

    Returns:
        total_loss (float): accumulated loss value for logging
    """
    is_asam = hasattr(optimizer, 'ascent_step')
    inner_opt = optimizer.optimizer if is_asam else optimizer
    total_loss = 0.0

    def _backward_pass(micro_batches_list):
        """Run forward-backward on all micro-batches with loss scaling."""
        nonlocal total_loss
        for batch in micro_batches_list:
            loss = loss_fn(model, batch)
            scaled_loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss += loss.item()

    if is_asam:
        # ── ASAM TWO-PASS PROTOCOL ────────────────────────────────────
        # Pass 1: compute gradients at current weights
        optimizer.zero_grad()
        _backward_pass(micro_batches)

        # Unscale before ascent (ASAM uses gradient magnitudes for perturbation)
        if scaler is not None:
            scaler.unscale_(inner_opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )

        # Perturb weights in steepest gradient direction
        optimizer.ascent_step()

        # Pass 2: compute gradients at PERTURBED weights
        optimizer.zero_grad()
        _backward_pass(micro_batches)  # re-run forward-backward at perturbed point

        if scaler is not None:
            scaler.unscale_(inner_opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )

        # Restore weights and step
        optimizer.descent_step()
        if scaler is not None:
            scaler.update()

    else:
        # ── REGULAR OPTIMIZER (AdamW, etc.) ───────────────────────────
        optimizer.zero_grad()
        _backward_pass(micro_batches)

        if scaler is not None:
            scaler.unscale_(inner_opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )

        if scaler is not None:
            scaler.step(inner_opt)
            scaler.update()
        else:
            optimizer.step()

    return total_loss / max(len(micro_batches) * (2 if is_asam else 1), 1)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION Y: ADDITIONAL INFERENCE AND MONITORING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def predict_with_routing(probs: np.ndarray, thresholds: Dict[str, float],
                         crop_names: List[str]) -> Tuple[Optional[str], float, bool]:
    """
    Inference-time routing decision with abstention.

    Args:
        probs: (num_crops,) softmax probabilities from router
        thresholds: {crop_name: min_confidence} from conformal calibration
        crop_names: ordered list of crop names matching prob indices

    Returns:
        (predicted_crop, confidence, should_abstain)
        If abstaining: predicted_crop=None, should_abstain=True
    """
    top_idx = int(np.argmax(probs))
    top_crop = crop_names[top_idx]
    confidence = float(probs[top_idx])
    threshold = thresholds.get(top_crop, 0.5)

    if confidence < threshold:
        return None, confidence, True
    return top_crop, confidence, False


def adaptive_capsicum_intervention(current_gap: float,
                                   sampling_weights: np.ndarray,
                                   capsicum_indices: List[int],
                                   threshold: float = 0.20) -> Tuple[np.ndarray, bool]:
    """
    Halve sampling weights for Capsicum images if shortcut learning detected.

    Args:
        current_gap: F1(capsicum) - F1(real_chilli) from track_subsource_f1
        sampling_weights: current per-image sampling weight array
        capsicum_indices: row indices of Capsicum images in the dataset
        threshold: gap above which intervention triggers

    Returns:
        (updated_weights, intervention_triggered)
    """
    if current_gap > threshold:
        updated = sampling_weights.copy()
        for idx in capsicum_indices:
            if idx < len(updated):
                updated[idx] *= 0.5
        print(f'  CAPSICUM INTERVENTION: gap={current_gap:.3f} > {threshold}. '
              f'Halved weights for {len(capsicum_indices)} Capsicum images.')
        return updated, True
    return sampling_weights, False


def get_gradcam_map(model, image: torch.Tensor, target_layer_name: str,
                    device='cuda', target_class: int = None) -> np.ndarray:
    """
    Generate GradCAM++ heatmap for Model 2 (ConvNeXt-based).

    Uses pytorch_grad_cam library with the specified target layer.
    For DINOv3-ConvNeXt loaded via transformers, typical target is 'stages.3'.

    Returns 2D numpy array (H, W) normalised to [0, 1].
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        # Find the target layer module
        target_module = model
        for attr in target_layer_name.split('.'):
            target_module = getattr(target_module, attr)

        # Get the last conv layer in the target stage
        target_layers = []
        for name, module in target_module.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layers = [module]  # keep overwriting to get the LAST conv

        if not target_layers:
            warnings.warn(f'No Conv2d found in {target_layer_name}')
            return np.ones((12, 12)) * 0.5

        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            grayscale = cam(input_tensor=image.to(device), targets=targets)[0]

        return grayscale

    except Exception as e:
        warnings.warn(f'GradCAM++ failed: {e}')
        return np.ones((12, 12)) * 0.5


@torch.no_grad()
def monitor_class_prototypes(model: nn.Module, loader, device='cuda',
                              num_classes: int = 10,
                              class_names: List[str] = None) -> Dict[str, float]:
    """
    Compute per-class embedding centroids and inter/intra-class distances.
    Early warning for embedding collapse (e.g., okra_enation merging with okra_yvmv).

    Returns dict with:
        'intra_{class}': mean distance from class centroid (compactness)
        'inter_{class_a}_{class_b}': distance between centroids (separation)
        'collapse_warning': list of class pairs with distance < threshold
    """
    model.eval()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Collect embeddings per class
    class_embeddings = defaultdict(list)

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].numpy()

        # Get features before classification head
        # Works for both timm and transformers models
        if hasattr(model, 'backbone'):
            out = model.backbone(images)
            if hasattr(out, 'pooler_output'):
                # [FIX] .float() before .numpy() — NumPy doesn't support BFloat16
                features = out.pooler_output.float().cpu().numpy()
            else:
                features = out.float().cpu().numpy()
        elif hasattr(model, 'forward_features'):
            features = model.forward_features(images).float().cpu().numpy()
        else:
            # Fallback: use forward but grab intermediate
            features = model(images).float().cpu().numpy()

        for i, label in enumerate(labels):
            class_embeddings[int(label)].append(features[i])

    # Compute centroids
    centroids = {}
    for cls_idx, embs in class_embeddings.items():
        centroids[cls_idx] = np.mean(embs, axis=0)

    results = {}

    # Intra-class distances (compactness)
    for cls_idx, embs in class_embeddings.items():
        centroid = centroids[cls_idx]
        distances = [np.linalg.norm(e - centroid) for e in embs]
        cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
        results[f'intra_{cls_name}'] = float(np.mean(distances))

    # Inter-class distances (separation)
    collapse_warnings = []
    cls_indices = sorted(centroids.keys())
    for i, cls_a in enumerate(cls_indices):
        for cls_b in cls_indices[i+1:]:
            dist = float(np.linalg.norm(centroids[cls_a] - centroids[cls_b]))
            name_a = class_names[cls_a] if class_names and cls_a < len(class_names) else str(cls_a)
            name_b = class_names[cls_b] if class_names and cls_b < len(class_names) else str(cls_b)
            results[f'inter_{name_a}_{name_b}'] = dist

            # Warn if inter-class distance is less than sum of intra-class distances
            intra_a = results.get(f'intra_{name_a}', 0)
            intra_b = results.get(f'intra_{name_b}', 0)
            if dist < (intra_a + intra_b) * 0.5:
                collapse_warnings.append((name_a, name_b, dist))

    if collapse_warnings:
        print(f'  COLLAPSE WARNING: {len(collapse_warnings)} class pairs too close:')
        for a, b, d in collapse_warnings[:5]:
            print(f'    {a} <-> {b}: distance={d:.4f}')

    results['collapse_warnings'] = collapse_warnings
    return results


def generate_splits(df, split_config: dict, seed: int = 42,
                    recomposed_sources: List[str] = None) -> Dict[str, List[int]]:
    """
    Generate source-aware stratified splits for a unified CSV DataFrame.

    Uses composite key class_name + '_' + source_bucket for stratification,
    ensuring proportional field/lab representation in every split.
    Recomposed images are forced into training only.

    Args:
        df: unified source map DataFrame with columns:
            class_name, source_dataset, is_field_photo
        split_config: dict of {split_name: fraction}, e.g.
            {'train': 0.68, 'val': 0.10, 'soup': 0.07, 'final_val': 0.10, 'conformal': 0.05}
        seed: random seed for reproducibility
        recomposed_sources: list of source_dataset values that are recomposed
            (these go to training only, never val/conformal)

    Returns:
        dict of {split_name: list of integer row indices}
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit

    recomposed_sources = recomposed_sources or []

    # Step 1: Separate recomposed images (training only)
    recomp_mask = df['source_dataset'].isin(recomposed_sources)
    recomp_indices = df[recomp_mask].index.tolist()
    non_recomp_df = df[~recomp_mask].copy()

    # Step 2: Create source_bucket for composite stratification key
    def assign_source_bucket(row):
        src = str(row.get('source_dataset', '')).lower()
        is_field = str(row.get('is_field_photo', 'False')).lower() == 'true'
        if 'scidb' in src:
            return 'scidb'
        elif is_field:
            return 'field'
        elif 'recomposed' in src:
            return 'recomposed'
        else:
            return 'lab'

    non_recomp_df['source_bucket'] = non_recomp_df.apply(assign_source_bucket, axis=1)
    non_recomp_df['strat_key'] = non_recomp_df['class_name'] + '_' + non_recomp_df['source_bucket']

    # Step 3: Sequential splitting with composite key
    # Sort splits by name to ensure deterministic order
    remaining_indices = non_recomp_df.index.tolist()
    remaining_df = non_recomp_df.copy()
    splits = {}

    # Calculate absolute sizes
    total = len(remaining_df)
    split_names = list(split_config.keys())

    for i, split_name in enumerate(split_names):
        if i == len(split_names) - 1:
            # Last split gets everything remaining
            splits[split_name] = remaining_indices
            break

        frac = split_config[split_name]
        n_split = max(1, int(total * frac))

        if len(remaining_indices) <= n_split:
            splits[split_name] = remaining_indices
            remaining_indices = []
            break

        # Stratified split using composite key
        strat_values = remaining_df.loc[remaining_indices, 'strat_key']

        # Handle rare strat_keys that have only 1 sample (can't stratify)
        key_counts = strat_values.value_counts()
        rare_keys = key_counts[key_counts < 2].index.tolist()

        if rare_keys:
            # Move rare-key samples to this split directly
            rare_mask = strat_values.isin(rare_keys)
            rare_idx = [idx for idx, is_rare in zip(remaining_indices, rare_mask) if is_rare]
            normal_idx = [idx for idx, is_rare in zip(remaining_indices, rare_mask) if not is_rare]

            if len(normal_idx) == 0:
                splits[split_name] = remaining_indices[:n_split]
                remaining_indices = remaining_indices[n_split:]
                continue

            strat_values_normal = strat_values[~rare_mask]
        else:
            normal_idx = remaining_indices
            strat_values_normal = strat_values
            rare_idx = []

        try:
            test_size = min(n_split / len(normal_idx), 0.5)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + i)
            _, split_sub_idx = next(sss.split(normal_idx, strat_values_normal))
            split_indices = [normal_idx[j] for j in split_sub_idx]
        except ValueError:
            # Stratification failed (too few samples per key) — fall back to random
            rng = np.random.default_rng(seed + i)
            split_indices = rng.choice(normal_idx, size=n_split, replace=False).tolist()

        # Add rare-key samples to split if under quota
        if len(split_indices) < n_split and rare_idx:
            needed = n_split - len(split_indices)
            split_indices.extend(rare_idx[:needed])
            rare_idx = rare_idx[needed:]

        splits[split_name] = split_indices
        remaining_indices = [idx for idx in remaining_indices if idx not in set(split_indices)]
        remaining_df = non_recomp_df.loc[remaining_indices]

    # Step 4: Add recomposed images to training split
    if 'train' in splits:
        splits['train'] = splits['train'] + recomp_indices
    elif split_names:
        splits[split_names[0]] = splits.get(split_names[0], []) + recomp_indices

    # Step 5: Verify no overlap between splits (full pairwise check)
    split_names_list = list(splits.keys())
    for i_s, name_a in enumerate(split_names_list):
        for name_b in split_names_list[i_s+1:]:
            overlap = set(splits[name_a]) & set(splits[name_b])
            if overlap:
                warnings.warn(f'SPLIT OVERLAP: {name_a} and {name_b} share '
                             f'{len(overlap)} indices!')

    # Also check for intra-split duplicates (e.g., recomposed added twice)
    for name, indices in splits.items():
        if len(indices) != len(set(indices)):
            n_dup = len(indices) - len(set(indices))
            warnings.warn(f'INTRA-SPLIT DUPLICATES: {name} has {n_dup} duplicate indices!')
            splits[name] = list(set(indices))  # deduplicate

    # Step 6: Print summary
    for name, indices in splits.items():
        n = len(indices)
        pct = n / len(df) * 100
        has_recomp = len([i for i in indices if i in set(recomp_indices)])
        print(f'  {name:<15}: {n:>6} samples ({pct:.1f}%), recomposed: {has_recomp}')

    return splits


# ═══════════════════════════════════════════════════════════════════════════
# MODULE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('train_utils.py — Module Self-Test')
    print('=' * 60)

    # Test ENS weights
    counts = [2965, 1612, 602, 335, 288, 1080, 338, 723, 1063]
    ens = compute_ens_class_weights(counts)
    print(f'ENS weights (Model 2): {ens.tolist()}')
    print(f'  Min weight: {ens.min():.4f} (largest class)')
    print(f'  Max weight: {ens.max():.4f} (smallest class)')

    # Test SupCon loss
    supcon = SupConLoss(temperature=0.10)
    feats = torch.randn(16, 384)
    labels = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7])
    loss = supcon(feats, labels)
    print(f'SupCon loss: {loss.item():.4f}')

    # Test ASAM wrapper
    model = nn.Linear(10, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    asam = ASAMWrapper(opt, model, rho=0.10)
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    asam.ascent_step()
    asam.zero_grad()
    out2 = model(x)
    loss2 = F.cross_entropy(out2, y)
    loss2.backward()
    asam.descent_step()
    print(f'ASAM step: OK')

    # Test CutMix
    imgs = torch.randn(8, 3, 224, 224)
    lbls = torch.tensor([0,1,2,3,4,5,6,7])
    mixed, la, lb, lam = apply_cutmix(imgs, lbls, probability=1.0)
    print(f'CutMix: lam={lam:.3f}, shapes OK')

    print()
    print('All tests passed.')
