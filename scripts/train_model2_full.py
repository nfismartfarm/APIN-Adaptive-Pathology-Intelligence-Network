"""
Full Model 2 Training Script — Complete MASTER_PLAN Recipe

Stage 1 (25 epochs): Progressive resize 128->224->384px
  - DINOv3-ConvNeXt-Small full fine-tune with LLRD
  - ASAM optimizer (rho=0.10 at 224px, rho=0.20 at 384px)
  - SupCon auxiliary loss (epochs 1-15, lambda=0.10)
  - ENS class weights (capped 3:1) + field photo 4x boost
  - EMA decay=0.9999 with epoch-0 re-seed
  - Cosine LR schedule per resolution phase

Stage 2 (7 epochs): Head-only retraining at 384px
  - Backbone frozen, only head trains
  - CutMix for thin classes (okra_enation, okra_cercospora)
  - Focal Loss gamma=2
  - EMA decay=0.999 (faster for short stage)

Target: macro F1 >= 0.92 (must beat existing 10-class EfficientNetV2-S model)

Usage:
    python scripts/train_model2_full.py
    python scripts/train_model2_full.py --resume   # resume from checkpoint
"""
import os
import sys
import json
import argparse
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config_model2 import (
    NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS,
    STAGE1_EPOCHS, STAGE2_EPOCHS, WEIGHT_DECAY, RANDOM_SEED,
    FIELD_PHOTO_MULTIPLIER, ENS_BETA,
    BATCH_SIZES_STAGE1, BATCH_SIZE_STAGE2,
    GRAD_ACCUM_STEPS_STAGE1, GRAD_ACCUM_STEPS_STAGE2,
    STAGE1_BASE_LR, STAGE2_HEAD_LR, LLRD_DECAY, GRAD_CLIP_NORM,
    ASAM_RHO_BY_RESOLUTION, ASAM_WARMUP_EPOCHS_PER_RESOLUTION,
    SUPCON_TEMPERATURE, SUPCON_LAMBDA, SUPCON_MAX_EPOCH,
    CUTMIX_CLASSES_STAGE2, CUTMIX_PROBABILITY, CUTMIX_ALPHA,
    EMA_DECAY_STAGE1, EMA_DECAY_STAGE2,
    SOUP_CHECKPOINT_EPOCHS, MIN_MACRO_F1,
    EARLY_STOPPING_PATIENCE, LABEL_SMOOTHING,
    assert_config_consistency,
)
from scripts.models import Model2ConvNeXt
from scripts.train_utils import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    setup_ema, reset_ema, evaluate,
    compute_ens_class_weights, get_augmentation_pipeline, get_eval_transform,
    SupConLoss, ASAMWrapper, apply_cutmix,
)

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')


# ── Focal Loss (Stage 2 only) ────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Dataset ───────────────────────────────────────────────────────────
class DiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        clahe = path.replace('/cleaned/', '/cleaned_clahe/').replace('\\cleaned\\', '\\cleaned_clahe\\')
        if os.path.exists(clahe):
            path = clahe
        try:
            img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f'WARNING: Failed to load {path}: {e}', flush=True)
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def get_resolution_schedule():
    """Progressive resize: 128px (ep 0-5) -> 224px (ep 6-15) -> 384px (ep 16-24)"""
    schedule = {}
    for ep in range(STAGE1_EPOCHS):
        if ep < 6:
            schedule[ep] = 128
        elif ep < 16:
            schedule[ep] = 224
        else:
            schedule[ep] = 384
    return schedule


def build_loaders(train_df, val_df, train_labels, val_labels,
                  img_size, batch_size, sample_weights):
    """Rebuild DataLoaders for a new resolution."""
    train_ds = DiseaseDataset(
        train_df['image_path'].tolist(), train_labels,
        transform=get_augmentation_pipeline(img_size)
    )
    val_ds = DiseaseDataset(
        val_df['image_path'].tolist(), val_labels,
        transform=get_eval_transform(img_size)
    )
    sampler = WeightedRandomSampler(sample_weights, len(train_df), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               sampler=sampler, num_workers=0,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Full Model 2 Training')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    assert_config_consistency()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('MODEL 2 FULL TRAINING — Stage 1 (25 ep) + Stage 2 (7 ep)')
    print('Progressive resize: 128px -> 224px -> 384px')
    print('ASAM + SupCon + LLRD + CutMix + Focal Loss')
    print('=' * 70, flush=True)

    # ── Load data ──────────────────────────────────────────────────────────
    import pandas as pd
    csv_path = ROOT / 'data' / 'specialist' / 'model2' / 'model2_unified_source_map.csv'
    df = pd.read_csv(csv_path)

    split_path = ROOT / 'data' / 'specialist' / 'model2' / 'split_indices.json'
    with open(split_path) as f:
        splits = json.load(f)
    train_idx = splits['train']
    val_idx = splits.get('val_and_soup', splits.get('val'))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_labels = [CLASS_TO_IDX[c] for c in train_df['class_name']]
    val_labels = [CLASS_TO_IDX[c] for c in val_df['class_name']]

    print(f'Train: {len(train_df)}, Val: {len(val_df)}', flush=True)
    for cls in CLASS_NAMES:
        n = int((train_df['class_name'] == cls).sum())
        flag = ' <- THIN' if n < 300 else ''
        print(f'  {cls:30s}: {n:5d}{flag}', flush=True)

    # ── Sampling weights ──────────────────────────────────────────────────
    class_counts = [int((train_df['class_name'] == cls).sum()) for cls in CLASS_NAMES]
    ens = compute_ens_class_weights(class_counts, beta=ENS_BETA)
    # Cap ENS ratio at 3:1
    capped_ens = ens.clone()
    min_w = capped_ens.max().item() / 3.0
    capped_ens = capped_ens.clamp(min=min_w)
    capped_ens = capped_ens * NUM_CLASSES / capped_ens.sum()
    print(f'Capped ENS: {[f"{w:.2f}" for w in capped_ens.tolist()]}', flush=True)

    is_field = train_df['is_field_photo'].astype(str).str.lower().isin(['true'])
    sample_weights = np.ones(len(train_df))
    for i, cls in enumerate(CLASS_NAMES):
        mask = train_df['class_name'] == cls
        sample_weights[mask.values] = float(capped_ens[i])
        sample_weights[(mask & is_field).values] *= FIELD_PHOTO_MULTIPLIER

    # ── Model ─────────────────────────────────────────────────────────────
    model = Model2ConvNeXt(num_classes=NUM_CLASSES, pretrained=True).to(device)
    # Model stays float32 — autocast handles BF16 forward only
    ema = setup_ema(model, decay=EMA_DECAY_STAGE1, device=device)

    # ── LLRD optimizer ────────────────────────────────────────────────────
    # [FIX: LR Investigation] STAGE1_BASE_LR=1e-3 destroyed features at epoch 0.
    # Simple fallback at 1e-4 worked (F1=0.27 at ep0 vs 0.008 with 1e-3).
    # Fix: use 1e-4 as base LR with LLRD. The config's 1e-3 was intended for
    # the head-only scenario, not for full fine-tuning with LLRD.
    effective_base_lr = 1e-4  # was STAGE1_BASE_LR (1e-3) — 10x too high
    param_groups = model.get_llrd_param_groups(
        base_lr=effective_base_lr, decay_rate=LLRD_DECAY, weight_decay=WEIGHT_DECAY
    )
    base_optimizer = torch.optim.AdamW(param_groups)

    # GradScaler for proper mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # ── SupCon loss ───────────────────────────────────────────────────────
    supcon_loss_fn = SupConLoss(temperature=SUPCON_TEMPERATURE)

    # ── Checkpointing ────────────────────────────────────────────────────
    ckpt_dir = ROOT / 'models' / 'model2_specialist'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    res_schedule = get_resolution_schedule()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Progressive resize + ASAM + SupCon + LLRD (25 epochs)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print('STAGE 1: Full fine-tune with ASAM + SupCon + progressive resize')
    print(f'{"="*70}', flush=True)

    best_f1 = 0.0
    current_res = None
    train_loader = val_loader = None
    optimizer = base_optimizer
    asam = None
    scheduler = None
    t_start = time.time()

    for epoch in range(STAGE1_EPOCHS):
        target_res = res_schedule[epoch]
        batch_size = BATCH_SIZES_STAGE1.get(target_res, 16)

        # ── Resolution transition ─────────────────────────────────────────
        if target_res != current_res:
            print(f'\n  Resolution transition: {current_res}px -> {target_res}px '
                  f'(batch={batch_size})', flush=True)
            current_res = target_res
            train_loader, val_loader = build_loaders(
                train_df, val_df, train_labels, val_labels,
                target_res, batch_size, sample_weights
            )

            # Cosine scheduler with linear warmup for this resolution phase
            remaining = sum(1 for e in range(epoch, STAGE1_EPOCHS)
                           if res_schedule[e] == target_res)
            warmup_epochs = min(2, remaining // 2)  # 2 epoch warmup per resolution

            def lr_lambda(ep, warmup=warmup_epochs, total=remaining):
                if ep < warmup:
                    return (ep + 1) / warmup  # linear warmup from ~0 to 1
                # cosine decay after warmup
                progress = (ep - warmup) / max(total - warmup, 1)
                return max(0.5 * (1 + np.cos(np.pi * progress)), 1e-6 / effective_base_lr)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            # ASAM: enable if resolution >= 224px
            rho = ASAM_RHO_BY_RESOLUTION.get(target_res, 0)
            if rho > 0:
                asam = ASAMWrapper(optimizer, model, rho=rho)
                print(f'  ASAM enabled: rho={rho}', flush=True)
            else:
                asam = None
                print(f'  ASAM disabled (128px warmup)', flush=True)

        # ── ASAM warmup check ─────────────────────────────────────────────
        epochs_at_res = epoch - min(e for e in range(STAGE1_EPOCHS)
                                    if res_schedule[e] == target_res)
        asam_active = (asam is not None and
                       epochs_at_res >= ASAM_WARMUP_EPOCHS_PER_RESOLUTION)

        # ── SupCon active check ───────────────────────────────────────────
        supcon_active = (epoch < SUPCON_MAX_EPOCH and target_res < 384)

        # ── CE loss for this epoch ────────────────────────────────────────
        class_weights = capped_ens.to(device)
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights,
                                          label_smoothing=LABEL_SMOOTHING)

        # ── Training loop ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_supcon = 0.0
        n_batches = 0
        grad_accum = GRAD_ACCUM_STEPS_STAGE1
        active_opt = asam if asam_active else optimizer

        active_opt.zero_grad()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                if supcon_active:
                    logits, features = model(images, return_features=True)
                else:
                    logits = model(images)

                loss = ce_loss_fn(logits, labels)

                # SupCon auxiliary loss
                if supcon_active:
                    sc_loss = supcon_loss_fn(F.normalize(features, dim=1), labels)
                    if not torch.isnan(sc_loss):
                        loss = loss + SUPCON_LAMBDA * sc_loss
                        epoch_supcon += sc_loss.item()

            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            n_batches += 1

            if n_batches % grad_accum == 0:
                if asam_active:
                    # [FIX: Pessimistic Audit #3] ASAM two-pass with GradScaler:
                    # Pass 1: unscale gradients, perturb weights (ascent)
                    # Do NOT call scaler.update() here -- update() without step() causes
                    # the scaler to think inf/NaN was detected, shrinking the scale factor.
                    scaler.unscale_(active_opt.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    active_opt.ascent_step()
                    # NO scaler.update() here -- intentional

                    # Pass 2: fresh forward-backward at perturbed weights
                    active_opt.zero_grad()
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                        logits2 = model(images)
                        loss2 = ce_loss_fn(logits2, labels) / grad_accum
                    # Use a fresh scale cycle for pass 2
                    loss2.backward()  # No scaler.scale() needed -- ascent already unscaled
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    active_opt.descent_step()
                    scaler.update()  # Single update() for the entire two-pass cycle
                else:
                    scaler.unscale_(active_opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(active_opt)
                    scaler.update()

                active_opt.zero_grad()

                if ema is not None:
                    ema.update(model)

            epoch_loss += loss.item()

        # Flush partial accumulation
        if n_batches % grad_accum != 0:
            if asam_active:
                scaler.unscale_(active_opt.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                # For flush, just do descent (skip second forward pass)
                active_opt.descent_step()
                scaler.update()
            else:
                scaler.unscale_(active_opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(active_opt)
                scaler.update()
            active_opt.zero_grad()
            if ema is not None:
                ema.update(model)

        # EMA re-seed after epoch 0
        if epoch == 0 and ema is not None:
            reset_ema(ema, model)
            print(f'  EMA re-seeded from epoch 0 weights', flush=True)

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, device=device,
                          class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
        val_f1 = metrics['macro_f1']

        ema_f1 = 0.0
        if ema is not None:
            ema_model = ema.module if hasattr(ema, 'module') else ema
            ema_metrics = evaluate(ema_model, val_loader, device=device,
                                  class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
            ema_f1 = ema_metrics['macro_f1']

        checkpoint_f1 = max(val_f1, ema_f1)
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start
        lr = optimizer.param_groups[0]['lr']

        features_str = []
        if asam_active: features_str.append('ASAM')
        if supcon_active: features_str.append('SupCon')
        feat_tag = f' [{"+".join(features_str)}]' if features_str else ''

        print(f'Ep {epoch:2d}/{STAGE1_EPOCHS} @{target_res}px: '
              f'loss={avg_loss:.4f} F1={val_f1:.4f} ema={ema_f1:.4f} '
              f'lr={lr:.2e} ({elapsed:.0f}s){feat_tag}', flush=True)

        # Per-class warnings
        for cls in CLASS_NAMES:
            cls_f1 = metrics.get(f'f1_{cls}', 0)
            if cls_f1 < 0.50:
                print(f'  WARNING: {cls} F1={cls_f1:.3f}', flush=True)

        # Checkpoint
        if checkpoint_f1 > best_f1:
            best_f1 = checkpoint_f1
            save_checkpoint(epoch, model, ema, optimizer, scheduler, scaler,
                           best_f1, str(ckpt_dir / 'model2_stage1_best.pt'))
            print(f'  -> Best: {best_f1:.4f}', flush=True)

        # Soup checkpoints
        if epoch + 1 in [20, 22, 24, 25]:
            save_checkpoint(epoch, model, ema, optimizer, scheduler, scaler,
                           checkpoint_f1,
                           str(ckpt_dir / f'model2_soup_ep{epoch:02d}.pt'))

    stage1_f1 = best_f1
    print(f'\nStage 1 complete. Best F1: {stage1_f1:.4f}', flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Head-only retraining at 384px with CutMix + Focal Loss
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print('STAGE 2: Head-only + CutMix + Focal Loss (7 epochs @ 384px)')
    print(f'{"="*70}', flush=True)

    # Load best Stage 1 checkpoint
    best_ckpt = str(ckpt_dir / 'model2_stage1_best.pt')
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model, ema, device=device)
        print(f'Loaded Stage 1 best: {best_ckpt}', flush=True)

    # Freeze backbone, train only head
    model.freeze_backbone()

    # Reset EMA for Stage 2 (faster decay)
    reset_ema(ema, model, new_decay=EMA_DECAY_STAGE2)

    # New optimizer for head-only
    head_params = list(model.head.parameters())
    stage2_optimizer = torch.optim.AdamW(head_params, lr=STAGE2_HEAD_LR,
                                          weight_decay=WEIGHT_DECAY)
    stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage2_optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-6
    )

    # Focal Loss for Stage 2
    focal_loss_fn = FocalLoss(weight=capped_ens.to(device), gamma=2.0,
                               label_smoothing=LABEL_SMOOTHING)

    # CutMix setup
    thin_class_indices = [CLASS_TO_IDX[c] for c in CUTMIX_CLASSES_STAGE2
                          if c in CLASS_TO_IDX]

    # 384px loaders
    train_loader, val_loader = build_loaders(
        train_df, val_df, train_labels, val_labels,
        384, BATCH_SIZE_STAGE2, sample_weights
    )

    stage2_scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    for epoch in range(STAGE2_EPOCHS):
        global_epoch = STAGE1_EPOCHS + epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        grad_accum = GRAD_ACCUM_STEPS_STAGE2

        stage2_optimizer.zero_grad()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # CutMix for thin classes
            mixed_images, labels_a, labels_b, lam = apply_cutmix(
                images, labels, alpha=CUTMIX_ALPHA,
                thin_class_indices=thin_class_indices,
                probability=CUTMIX_PROBABILITY
            )

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits = model(mixed_images)
                if isinstance(lam, float) and lam == 1.0:
                    loss = focal_loss_fn(logits, labels_a)
                else:
                    loss = lam * focal_loss_fn(logits, labels_a) + \
                           (1 - lam) * focal_loss_fn(logits, labels_b)

            scaled_loss = loss / grad_accum
            stage2_scaler.scale(scaled_loss).backward()
            n_batches += 1

            if n_batches % grad_accum == 0:
                stage2_scaler.unscale_(stage2_optimizer)
                torch.nn.utils.clip_grad_norm_(head_params, GRAD_CLIP_NORM)
                stage2_scaler.step(stage2_optimizer)
                stage2_scaler.update()
                stage2_optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            epoch_loss += loss.item()

        # Flush
        if n_batches % grad_accum != 0:
            stage2_scaler.unscale_(stage2_optimizer)
            torch.nn.utils.clip_grad_norm_(head_params, GRAD_CLIP_NORM)
            stage2_scaler.step(stage2_optimizer)
            stage2_scaler.update()
            stage2_optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        stage2_scheduler.step()

        # Validation
        metrics = evaluate(model, val_loader, device=device,
                          class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
        val_f1 = metrics['macro_f1']

        ema_f1 = 0.0
        if ema is not None:
            ema_model = ema.module if hasattr(ema, 'module') else ema
            ema_metrics = evaluate(ema_model, val_loader, device=device,
                                  class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
            ema_f1 = ema_metrics['macro_f1']

        checkpoint_f1 = max(val_f1, ema_f1)
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start

        print(f'Stage2 Ep {epoch}/{STAGE2_EPOCHS}: loss={avg_loss:.4f} '
              f'F1={val_f1:.4f} ema={ema_f1:.4f} ({elapsed:.0f}s) [CutMix+Focal]',
              flush=True)

        for cls in CLASS_NAMES:
            cls_f1 = metrics.get(f'f1_{cls}', 0)
            if cls_f1 < 0.50:
                print(f'  WARNING: {cls} F1={cls_f1:.3f}', flush=True)

        if checkpoint_f1 > best_f1:
            best_f1 = checkpoint_f1
            save_checkpoint(global_epoch, model, ema, stage2_optimizer,
                           stage2_scheduler, stage2_scaler,
                           best_f1, str(ckpt_dir / 'model2_best.pt'))
            print(f'  -> Best: {best_f1:.4f}', flush=True)

        # Soup checkpoints for Stage 2
        if global_epoch in SOUP_CHECKPOINT_EPOCHS:
            save_checkpoint(global_epoch, model, ema, stage2_optimizer,
                           stage2_scheduler, stage2_scaler,
                           checkpoint_f1,
                           str(ckpt_dir / f'model2_soup_ep{global_epoch:02d}.pt'))

    # ══════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'TRAINING COMPLETE')
    print(f'  Stage 1 best F1: {stage1_f1:.4f}')
    print(f'  Overall best F1: {best_f1:.4f}')
    print(f'  Total time: {total_time/3600:.1f} hours')
    print(f'  Target: >= {MIN_MACRO_F1}')

    if best_f1 >= 0.92:
        print(f'  EXCELLENT: Surpasses old 10-class model (0.92)')
    elif best_f1 >= MIN_MACRO_F1:
        print(f'  PASS: Meets minimum threshold ({MIN_MACRO_F1})')
    else:
        print(f'  BELOW TARGET: {best_f1:.4f} < {MIN_MACRO_F1}')
        print(f'  Consider: more epochs, lower LR, or architecture changes')

    # Per-class final F1
    print(f'\nPer-class F1 (final epoch):')
    for cls in CLASS_NAMES:
        f1_val = metrics.get(f'f1_{cls}', 0)
        print(f'  {cls:30s}: {f1_val:.4f}', flush=True)

    print(f'\n{"="*70}', flush=True)


if __name__ == '__main__':
    main()
