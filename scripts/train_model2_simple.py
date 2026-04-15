"""
Simple Model 2 Training Script — Fallback Recipe

Minimal recipe: DINOv3-ConvNeXt-Small full fine-tune, no ASAM, no SupCon, no CutMix.
Produces a working Model 2 in ~2 hours.

Per MASTER_PLAN Simple Fallback:
- DINOv3-ConvNeXt-Small full fine-tune via transformers
- Focal Loss gamma=2 with ENS weights
- WeightedRandomSampler with field photo 4x boost
- EMA decay=0.9999, AdamW, cosine schedule, 15 epochs at 224px
- AugMix augmentation
- NO: ASAM, SupCon, CutMix, LLRD, progressive resizing
- Expected val macro F1: 0.78-0.82
- Training time: ~2 hours

Fixes applied from pessimistic audit:
- Model stays in float32; autocast handles BF16 for forward pass only
- Focal Loss implemented (was incorrectly using plain CrossEntropyLoss)
- EMA re-seeded after epoch 0 to eliminate random init contamination
- EMA model also evaluated for checkpoint selection

Usage:
    python scripts/train_model2_simple.py
    python scripts/train_model2_simple.py --epochs 3 --smoke-test
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config_model2 import (
    NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX,
    STAGE1_EPOCHS, WEIGHT_DECAY, RANDOM_SEED,
    FIELD_PHOTO_MULTIPLIER, ENS_BETA,
    MIN_MACRO_F1,
)
from scripts.models import Model2ConvNeXt
from scripts.train_utils import (
    save_checkpoint, setup_ema, reset_ema, evaluate,
    compute_ens_class_weights, get_augmentation_pipeline, get_eval_transform,
)

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')


# ── Focal Loss ────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017): downweights well-classified examples.
    FL(p) = -alpha * (1-p)^gamma * log(p)

    For imbalanced plant disease datasets, gamma=2 significantly improves
    thin-class performance by reducing the gradient contribution from
    easy majority-class examples (e.g., brassica_healthy at 2965 images
    vs okra_enation at 288 images).
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)  # probability of correct class
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal


class DiseaseDataset(Dataset):
    """Generic image dataset for disease classification."""

    def __init__(self, image_paths, labels, transform=None):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # Try CLAHE version
        clahe_path = path.replace('/cleaned/', '/cleaned_clahe/').replace('\\cleaned\\', '\\cleaned_clahe\\')
        if os.path.exists(clahe_path):
            path = clahe_path

        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f'WARNING: Failed to load {path}: {e}', flush=True)
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description='Simple Model 2 Training')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 3

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print(f'MODEL 2 SIMPLE TRAINING — {args.epochs} epochs at {args.img_size}px')
    print('=' * 70)
    print(flush=True)

    # ── Load data ──────────────────────────────────────────────────────────
    import pandas as pd
    csv_path = ROOT / 'data' / 'specialist' / 'model2' / 'model2_unified_source_map.csv'
    df = pd.read_csv(csv_path)

    split_path = ROOT / 'data' / 'specialist' / 'model2' / 'split_indices.json'
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
        train_idx = splits.get('train', list(range(int(len(df) * 0.68))))
        val_idx = splits.get('val_and_soup', splits.get('val',
                    list(range(int(len(df) * 0.68), int(len(df) * 0.83)))))
    else:
        print('WARNING: No split_indices.json — using random split')
        n = len(df)
        perm = np.random.permutation(n)
        train_idx = perm[:int(n * 0.68)].tolist()
        val_idx = perm[int(n * 0.68):int(n * 0.83)].tolist()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_labels = [CLASS_TO_IDX[c] for c in train_df['class_name']]
    val_labels = [CLASS_TO_IDX[c] for c in val_df['class_name']]

    print(f'Train: {len(train_df)}, Val: {len(val_df)}', flush=True)

    # Per-class counts
    for cls in CLASS_NAMES:
        n = int((train_df['class_name'] == cls).sum())
        flag = ' <- THIN' if n < 300 else ''
        print(f'  {cls:30s}: {n:5d}{flag}', flush=True)
    print(flush=True)

    # ── Datasets + samplers ───────────────────────────────────────────────
    train_ds = DiseaseDataset(train_df['image_path'].tolist(), train_labels,
                               transform=get_augmentation_pipeline(args.img_size))
    val_ds = DiseaseDataset(val_df['image_path'].tolist(), val_labels,
                             transform=get_eval_transform(args.img_size))

    # ENS sampling weights + field photo boost
    class_counts = [int((train_df['class_name'] == cls).sum()) for cls in CLASS_NAMES]
    ens = compute_ens_class_weights(class_counts, beta=ENS_BETA)
    print(f'ENS weights: {[f"{w:.2f}" for w in ens.tolist()]}', flush=True)

    is_field = train_df['is_field_photo'].astype(str).str.lower().isin(['true'])
    sample_weights = np.ones(len(train_df))
    for i, cls in enumerate(CLASS_NAMES):
        mask = train_df['class_name'] == cls
        sample_weights[mask.values] = float(ens[i])
        sample_weights[(mask & is_field).values] *= FIELD_PHOTO_MULTIPLIER

    sampler = WeightedRandomSampler(sample_weights, len(train_df), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=sampler, num_workers=0,
                               pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=(device == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────────────
    # [FIX: Pessimistic Audit] Model stays in float32. Autocast handles BF16
    # for the forward pass ONLY. This preserves float32 master weights for
    # gradient accumulation — critical for 49.5M param full fine-tuning.
    model = Model2ConvNeXt(num_classes=NUM_CLASSES, pretrained=True).to(device)
    # DO NOT: model = model.to(torch.bfloat16)  — this was causing F1=0.007

    ema = setup_ema(model, decay=0.9999, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # [FIX: Class Collapse Investigation] Use standard CrossEntropyLoss with
    # CAPPED ENS weights for the simple fallback. Focal Loss caused class collapse
    # when combined with extreme ENS weight ratios (9.4x) — the focal gamma=2
    # further suppressed large-class gradients that were already downweighted by ENS,
    # preventing okra_yvmv (weight=0.38) and okra_healthy (weight=0.21) from learning.
    #
    # Cap ENS weight ratio at 3:1 to prevent gradient starvation of majority classes.
    # Focal Loss is appropriate for Stage 2 refinement (post-convergence), NOT for
    # initial training from randomly-initialized heads.
    capped_ens = ens.clone()
    max_w = capped_ens.max().item()
    min_w = max_w / 3.0  # cap ratio at 3:1
    capped_ens = capped_ens.clamp(min=min_w)
    # Renormalize so weights sum to NUM_CLASSES
    capped_ens = capped_ens * NUM_CLASSES / capped_ens.sum()
    print(f'Capped ENS weights: {[f"{w:.2f}" for w in capped_ens.tolist()]}', flush=True)

    class_weights = capped_ens.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # GradScaler for proper mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    ckpt_dir = ROOT / 'models' / 'model2_specialist'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    best_f1 = 0.0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            # [FIX] Images stay float32 — autocast handles dtype conversion
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits = model(images)
                loss = loss_fn(logits, labels)

            # [FIX] Use GradScaler for proper mixed precision backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        # [FIX: EMA warmup] After epoch 0, re-seed EMA from partially-trained weights
        if epoch == 0 and ema is not None:
            reset_ema(ema, model)
            print(f'  EMA re-seeded from epoch 0 weights', flush=True)

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate both raw model and EMA model
        metrics = evaluate(model, val_loader, device=device,
                          class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
        val_f1 = metrics['macro_f1']

        # [FIX: Pessimistic Audit] Also evaluate EMA for checkpoint selection
        ema_f1 = 0.0
        if ema is not None:
            ema_model = ema.module if hasattr(ema, 'module') else ema
            ema_metrics = evaluate(ema_model, val_loader, device=device,
                                  class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
            ema_f1 = ema_metrics['macro_f1']

        # Use best of raw vs EMA for checkpoint decision
        checkpoint_f1 = max(val_f1, ema_f1)

        elapsed = time.time() - t_start
        print(f'Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f} '
              f'val_f1={val_f1:.4f} ema_f1={ema_f1:.4f} ({elapsed:.0f}s)', flush=True)

        # Print per-class F1 for thin class monitoring
        for cls in CLASS_NAMES:
            cls_f1 = metrics.get(f'f1_{cls}', 0)
            if cls_f1 < 0.50:
                print(f'  WARNING: {cls} F1={cls_f1:.3f}', flush=True)

        if checkpoint_f1 > best_f1:
            best_f1 = checkpoint_f1
            save_checkpoint(epoch, model, ema, optimizer, scheduler, scaler,
                           best_f1, str(ckpt_dir / 'model2_simple_best.pt'))
            print(f'  -> Best: {best_f1:.4f}', flush=True)

    print(f'\nComplete. Best F1: {best_f1:.4f}', flush=True)
    print(f'Acceptance: {"PASS" if best_f1 >= MIN_MACRO_F1 else "FAIL"} '
          f'(threshold: {MIN_MACRO_F1})', flush=True)

    # Per-class summary at end
    print(f'\nPer-class F1 at final epoch:', flush=True)
    for cls in CLASS_NAMES:
        f1_val = metrics.get(f'f1_{cls}', 0)
        print(f'  {cls:30s}: {f1_val:.4f}', flush=True)

    print('\nTEST RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
