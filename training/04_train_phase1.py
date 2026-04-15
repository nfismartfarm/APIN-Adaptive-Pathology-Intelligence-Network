# training/04_train_phase1.py
"""
Phase 2A: Train FPN + attention pooling + all heads with backbone frozen.

Unlike the original Phase 1 which used cached features, Phase 2A trains on
raw images because the FPN and attention pooling need to learn meaningful
feature aggregation from the Swin-Tiny backbone output. Cached features from
random FPN+attention are not useful.

Uses Focal Loss with gamma warmup, teacher-forced crop routing for MoE,
and EMA for stable validation.

Saves: models/checkpoints/phase2a_best.pt
Run AFTER 03_cache_features.py completes (for severity labels).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

os.environ['WANDB_MODE'] = 'disabled'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score

from app.config import (
    DEVICE, ROOT, SOURCE_MAP, CKPT_DIR, MODELS, TEACHER_MODEL,
    NUM_CLASSES, NUM_CROPS, CLASS_NAMES, RANDOM_SEED,
    PHASE2A_EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, THIN_CLASS_INDICES,
    FOCAL_GAMMA, LABEL_SMOOTHING,
    LOSS_WEIGHT_CROP, LOSS_WEIGHT_SEVERITY,
    CLASS_TO_IDX, CROP_FROM_IDX,
)
from app.model import PlantDiseaseModel
from training.losses import FocalBCELoss, EMAModel
from training.dataset import PlantDiseaseDataset, load_severity_labels, make_weighted_sampler
from training.transforms import get_train_transform, get_eval_transform


def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_phase2a(model, val_loader):
    """Compute val macro F1 using full forward pass on raw images."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in val_loader:
            images = images.to(DEVICE)
            c_log, d_log, s_log = model(images)
            all_probs.append(torch.sigmoid(d_log).cpu().numpy())
            all_labels.append(d_lab.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = (all_probs > 0.5).astype(int)
    macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    per_class = f1_score(all_labels, preds, average=None, zero_division=0)
    return macro_f1, per_class


def train_phase2a():
    set_seeds(RANDOM_SEED)

    # Verify teacher intact
    teacher_path = TEACHER_MODEL if os.path.isabs(TEACHER_MODEL) else os.path.join(ROOT, TEACHER_MODEL)
    assert os.path.exists(teacher_path), f'Teacher missing: {teacher_path}'
    teacher_mb = os.path.getsize(teacher_path) / 1e6
    assert teacher_mb > 80, f'Teacher corrupted: {teacher_mb:.1f}MB'
    print(f'Teacher intact: {teacher_mb:.1f}MB')

    # Build model — freeze backbone, train FPN + attention + heads
    model = PlantDiseaseModel().to(DEVICE)
    model.freeze_backbone()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Model: {trainable:,} trainable, {frozen:,} frozen (backbone)')

    # Optimizer: all trainable params (FPN + attention + all heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2,
    )

    # Mixed precision
    use_amp = (DEVICE.type == 'cuda')
    scaler = GradScaler(device='cuda' if use_amp else 'cpu', enabled=use_amp)

    focal_loss = FocalBCELoss(gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING)
    ema = EMAModel(model, decay=0.999)

    # Data loading
    df = pd.read_csv(SOURCE_MAP)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    train_records = train_df.to_dict('records')
    val_records = val_df.to_dict('records')
    for r in train_records + val_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx'] = CROP_FROM_IDX.get(r['class_idx'], 0)

    sev_labels = load_severity_labels()
    train_ds = PlantDiseaseDataset(train_records, get_train_transform(), sev_labels)
    val_ds = PlantDiseaseDataset(val_records, get_eval_transform(), sev_labels)
    sampler = make_weighted_sampler(train_records)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True, persistent_workers=False,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=False,
    )

    print(f'Train: {len(train_records)} images, Val: {len(val_records)} images')
    print(f'Batches/epoch: {len(train_loader)}, Batch size: {BATCH_SIZE}')

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_f1 = 0.0
    patience_counter = 0
    total_start = time.time()

    for epoch in range(PHASE2A_EPOCHS):
        model.train()
        epoch_loss = 0.0
        total_steps = len(train_loader)
        t0 = time.time()

        for step, (images, d_lab, c_lab, s_lab) in enumerate(train_loader):
            images = images.to(DEVICE)
            d_lab = d_lab.to(DEVICE)
            c_lab = c_lab.to(DEVICE)
            s_lab = s_lab.to(DEVICE)

            # Gamma warmup over first epoch
            if epoch == 0:
                focal_loss.set_gamma(FOCAL_GAMMA * (step / max(total_steps - 1, 1)))
            else:
                focal_loss.set_gamma(FOCAL_GAMMA)

            with autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                # Full forward pass (backbone frozen, FPN+heads trainable)
                # Teacher-force crop routing: use GT crop labels for MoE
                features = model.backbone(images)
                fpn_out = model.fpn(features)
                pooled = model.att_pool(fpn_out)

                # Crop classifier
                crop_logits, crop_emb = model.crop_classifier(pooled)

                # Teacher-forcing: use GT crop labels for MoE routing
                crop_probs_tf = F.one_hot(c_lab, num_classes=NUM_CROPS).float()

                # CLN + MoE with teacher-forced routing
                x_cln = model.cln(pooled, crop_probs_tf)
                disease_logits = model.disease_head(x_cln, crop_probs_tf)
                severity_logits = model.severity_head(pooled)

                # Losses
                loss_disease = focal_loss(disease_logits, d_lab)
                loss_crop = F.cross_entropy(crop_logits, c_lab.long())
                loss_severity = F.cross_entropy(severity_logits, s_lab.long())
                total_loss = loss_disease + LOSS_WEIGHT_CROP * loss_crop + LOSS_WEIGHT_SEVERITY * loss_severity

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            epoch_loss += total_loss.item()

        # Validation (without teacher-forcing — use predicted crop_probs)
        ema.apply(model)
        val_f1, per_class_f1 = validate_phase2a(model, val_loader)
        ema.restore(model)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / total_steps
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch:2d} | {elapsed/60:.1f}min | loss={avg_loss:.4f} '
              f'val_macro_F1={val_f1:.4f} lr={current_lr:.2e}')
        print('Per-class F1:')
        for i, (cls, f1v) in enumerate(zip(CLASS_NAMES, per_class_f1)):
            thin = ' *THIN*' if i in THIN_CLASS_INDICES else ''
            print(f'  {cls:<40} {f1v:.3f}{thin}')

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            ckpt_path = os.path.join(CKPT_DIR, 'phase2a_best.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> New best: {best_val_f1:.4f} — saved')
        else:
            patience_counter += 1
            print(f'  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}')
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

    total_time = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'Phase 2A complete.')
    print(f'Best val macro F1: {best_val_f1:.4f}')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'phase2a_best.pt: {os.path.exists(os.path.join(CKPT_DIR, "phase2a_best.pt"))}')

    assert os.path.exists(teacher_path)
    assert os.path.getsize(teacher_path) / 1e6 > 80
    print(f'Teacher preserved: {os.path.getsize(teacher_path)/1e6:.1f}MB INTACT')
    print(f'{"="*60}')


if __name__ == '__main__':
    train_phase2a()
