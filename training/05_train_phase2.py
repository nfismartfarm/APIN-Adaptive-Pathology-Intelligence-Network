# training/05_train_phase2.py
"""
Phase 2B: Full fine-tuning with all loss functions.

Loads phase2a_best.pt (head-trained checkpoint) as starting point.
Unfreezes entire backbone for end-to-end training.

Loss components: Focal BCE, CORAL, ArcFace, DeiT distillation, crop CE, severity CE
Data augmentation: CutMix, RandAugment, RandomErasing (via albumentations)
Model averaging: EMA (every step) + SWA (final 20% of epochs)
LR schedule: CosineAnnealingWarmRestarts with 5-epoch warmup

Saves final model to: models/swin_best_model.pt (SWA-averaged weights)
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
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, update_bn

from app.config import (
    DEVICE, ROOT, SOURCE_MAP, NUM_CLASSES, CLASS_NAMES, RANDOM_SEED,
    BEST_MODEL, TEACHER_MODEL, CKPT_DIR, MODELS,
    PHASE2B_PEAK_LR, PHASE2B_BACKBONE_LR, PHASE2B_FPN_LR,
    PHASE2B_BATCH, PHASE2B_EPOCHS, PHASE2B_WARMUP_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    CORAL_LAMBDA, ARCFACE_WEIGHT, ARCFACE_IN_FEATURES,
    CUTMIX_PROB, CUTMIX_ALPHA, RANDOM_ERASING_PROB,
    THIN_CLASS_INDICES, EMA_DECAY, SWA_START_FRACTION,
    DISTILLATION_ALPHA, DISTILLATION_TEMP,
    PLANTDOC_SOURCE_PATTERN, MIN_PLANTDOC_PER_BATCH,
    LOSS_WEIGHT_CROP, LOSS_WEIGHT_SEVERITY,
    COSINE_T0, COSINE_T_MULT, COSINE_ETA_MIN,
    FPN_OUT_CH, FOCAL_GAMMA, LABEL_SMOOTHING,
    CLASS_TO_IDX, CROP_FROM_IDX,
)
from app.model import PlantDiseaseModel
from training.losses import (
    FocalBCELoss, CORALLoss, ArcFaceLoss,
    DeiTDistillationLoss, EMAModel, cutmix_batch,
)
from training.teacher_model import load_teacher_model
from training.dataset import (
    PlantDiseaseDatasetPhase2B, DomainBalancedSampler,
    load_severity_labels,
)
from training.transforms import get_train_transform, get_eval_transform


def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_phase2b(model, val_loader, device):
    """Compute val macro F1 using full forward pass on raw images."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            d_lab = batch['disease_labels']
            c_log, d_log, s_log = model(images)
            all_probs.append(torch.sigmoid(d_log).cpu().numpy())
            all_labels.append(d_lab.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    thresholds = np.full(NUM_CLASSES, 0.5)
    preds = (all_probs > thresholds).astype(int)
    macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    per_class = f1_score(all_labels, preds, average=None, zero_division=0)
    return macro_f1, per_class


def train_phase2b():
    set_seeds(RANDOM_SEED)

    # ================================================================
    # STEP 1: Load model and unfreeze ALL backbone parameters
    # CRITICAL: Must happen BEFORE optimizer creation
    # ================================================================
    model = PlantDiseaseModel().to(DEVICE)

    # Load Phase 2A checkpoint
    phase2a_ckpt = os.path.join(CKPT_DIR, 'phase2a_best.pt')
    if os.path.exists(phase2a_ckpt):
        model.load_state_dict(torch.load(phase2a_ckpt, map_location=DEVICE, weights_only=False))
        print(f'Loaded Phase 2A checkpoint: {phase2a_ckpt}')
    else:
        print('WARNING: No Phase 2A checkpoint — starting from scratch')

    # CRITICAL: Unfreeze entire backbone
    model.backbone.unfreeze_all()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'After unfreeze_all(): {trainable:,} trainable, {frozen:,} frozen')
    assert frozen == 0, f'Expected 0 frozen params, got {frozen}'

    # ================================================================
    # STEP 2: Load and verify teacher model
    # ================================================================
    teacher_path = TEACHER_MODEL if os.path.isabs(TEACHER_MODEL) else os.path.join(ROOT, TEACHER_MODEL)
    assert os.path.exists(teacher_path), f'Teacher missing: {teacher_path}'
    teacher_mb = os.path.getsize(teacher_path) / 1e6
    assert teacher_mb > 80, f'Teacher corrupted: {teacher_mb:.1f}MB'
    teacher = load_teacher_model(teacher_path, DEVICE)
    teacher.eval()
    assert sum(p.numel() for p in teacher.parameters() if p.requires_grad) == 0
    print(f'Teacher loaded and frozen: {teacher_mb:.1f}MB')

    # ================================================================
    # STEP 3: ArcFace module (learnable params in optimizer)
    # ================================================================
    arcface = ArcFaceLoss(in_features=ARCFACE_IN_FEATURES, num_classes=NUM_CLASSES).to(DEVICE)

    # ================================================================
    # STEP 4: Optimizer with separate LR per component group
    # ArcFace weight explicitly included
    # ================================================================
    param_groups = [
        {
            'params': list(model.backbone.parameters()),
            'lr': PHASE2B_BACKBONE_LR,
            'name': 'backbone',
        },
        {
            'params': list(model.fpn.parameters()) + list(model.att_pool.parameters()),
            'lr': PHASE2B_FPN_LR,
            'name': 'fpn_pool',
        },
        {
            'params': (list(model.crop_classifier.parameters()) +
                       list(model.cln.parameters()) +
                       list(model.disease_head.parameters()) +
                       list(model.severity_head.parameters()) +
                       list(model.distillation_head.parameters()) +
                       list(arcface.parameters())),
            'lr': PHASE2B_PEAK_LR,
            'name': 'heads_and_arcface',
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # CosineAnnealingWarmRestarts — compatible with early stopping
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=COSINE_ETA_MIN,
    )

    # Mixed precision DISABLED for Phase 2B — AMP produces inf/nan gradients
    # in some parameters which contaminates clip_grad_norm_ and causes scaler
    # to skip optimizer steps. FP32 is safe at batch=20 (~3.6GB VRAM).
    use_amp = False
    scaler = GradScaler(device='cuda', enabled=False)

    # ================================================================
    # STEP 5: Data loading with DomainBalancedSampler
    # ================================================================
    df = pd.read_csv(SOURCE_MAP)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    # Add class_idx and crop_idx to records
    train_records = train_df.to_dict('records')
    val_records = val_df.to_dict('records')
    for r in train_records + val_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx'] = CROP_FROM_IDX.get(r['class_idx'], 0)

    sev_labels = load_severity_labels()
    train_ds = PlantDiseaseDatasetPhase2B(train_records, get_train_transform(), sev_labels)
    val_ds = PlantDiseaseDatasetPhase2B(val_records, get_eval_transform(), sev_labels)

    sampler = DomainBalancedSampler(
        train_records, PHASE2B_BATCH, MIN_PLANTDOC_PER_BATCH,
        source_pattern=PLANTDOC_SOURCE_PATTERN,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=PHASE2B_BATCH, sampler=sampler,
        num_workers=2, pin_memory=True, persistent_workers=False,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=PHASE2B_BATCH, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=False,
    )

    print(f'Train: {len(train_records)} images, Val: {len(val_records)} images')
    print(f'Batches/epoch: {len(train_loader)}, Batch size: {PHASE2B_BATCH}')

    # ================================================================
    # STEP 6: Loss functions and training utilities
    # ================================================================
    focal_loss = FocalBCELoss(gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING)
    coral_criterion = CORALLoss(lambda_coral=CORAL_LAMBDA)
    deit_criterion = DeiTDistillationLoss(alpha=DISTILLATION_ALPHA, temperature=DISTILLATION_TEMP)
    ema = EMAModel(model, decay=EMA_DECAY)

    # SWA setup
    swa_model = AveragedModel(model)
    swa_start_epoch = int(PHASE2B_EPOCHS * SWA_START_FRACTION)
    swa_active = False

    # RandomErasing
    from torchvision.transforms import v2 as T
    random_erasing = T.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.2))

    os.makedirs(CKPT_DIR, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0
    total_start = time.time()

    # Phase 2A epoch 14 per-class F1 baseline for regression detection
    phase2a_baseline = {
        'okra_yvmv': 0.893, 'okra_powdery_mildew': 0.519, 'okra_cercospora': 0.422,
        'okra_enation': 0.854, 'okra_healthy': 0.417, 'brassica_black_rot': 0.714,
        'brassica_downy_mildew': 0.252, 'brassica_alternaria': 0.513,
        'brassica_clubroot': 0.684, 'brassica_healthy': 0.426,
        'tomato_bacterial_spot': 0.0, 'tomato_early_blight': 0.0,
        'tomato_late_blight': 0.0, 'tomato_leaf_mold': 0.0,
        'tomato_septoria_leaf_spot': 0.0, 'tomato_target_spot': 0.415,
        'tomato_mosaic_virus': 0.510, 'tomato_yellow_leaf_curl_virus': 0.701,
        'tomato_healthy': 0.135, 'chilli_anthracnose': 0.574,
        'chilli_cercospora_leaf_spot': 0.545, 'chilli_leaf_curl': 0.724,
        'chilli_healthy': 0.435,
    }
    tomato_indices_set = set(range(10, 19))  # indices 10-18

    # ================================================================
    # STEP 7: Training loop
    # ================================================================
    for epoch in range(PHASE2B_EPOCHS):
        model.train()
        arcface.train()
        t0 = time.time()

        # LR warmup for first PHASE2B_WARMUP_EPOCHS
        if epoch < PHASE2B_WARMUP_EPOCHS:
            warmup_factor = (epoch + 1) / PHASE2B_WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                if pg['name'] == 'backbone':
                    pg['lr'] = PHASE2B_BACKBONE_LR * warmup_factor
                elif pg['name'] == 'fpn_pool':
                    pg['lr'] = PHASE2B_FPN_LR * warmup_factor
                else:
                    pg['lr'] = PHASE2B_PEAK_LR * warmup_factor

        total_steps = len(train_loader)
        running = {'total': 0.0, 'disease': 0.0, 'coral': 0.0,
                   'arcface': 0.0, 'kl': 0.0, 'crop': 0.0, 'severity': 0.0}

        for step, batch in enumerate(train_loader):
            images = batch['image'].to(DEVICE)
            disease_labels = batch['disease_labels'].to(DEVICE)
            crop_labels = batch['crop_label'].to(DEVICE)
            severity_labels = batch['severity_label'].to(DEVICE)
            is_plantdoc = batch['is_plantdoc'].to(DEVICE)
            source_mask = ~is_plantdoc
            target_mask = is_plantdoc

            # Gamma warmup
            if epoch == 0:
                focal_loss.set_gamma(FOCAL_GAMMA * (step / max(total_steps - 1, 1)))
            else:
                focal_loss.set_gamma(FOCAL_GAMMA)

            # Apply RandomErasing per image
            images = torch.stack([random_erasing(img) for img in images])

            # CutMix with probability CUTMIX_PROB
            cutmix_applied = False
            disease_labels_for_loss = disease_labels.float()
            if torch.rand(1).item() < CUTMIX_PROB:
                images, disease_labels_mixed, lam, rand_idx = cutmix_batch(
                    images, disease_labels, alpha=CUTMIX_ALPHA
                )
                disease_labels_for_loss = disease_labels_mixed.to(DEVICE)
                cutmix_applied = True

            # Full forward pass (single pass — all outputs + pooled features)
            with autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                (crop_logits, disease_logits, severity_logits,
                 distillation_logits, pooled) = model.forward_with_features(images)

                # Teacher forward (no grad, no AMP autocast for teacher)
                with torch.no_grad():
                    teacher_logits = teacher(images)

                # ---- DISEASE FOCAL LOSS ----
                loss_disease = focal_loss(disease_logits, disease_labels_for_loss)

                # ---- CORAL LOSS (gradients flow to backbone — NOT detached) ----
                source_pooled = pooled[source_mask]
                target_pooled = pooled[target_mask]
                loss_coral = coral_criterion(source_pooled, target_pooled)

                # ---- ARCFACE LOSS (use hard labels, not CutMix mixed) ----
                loss_arcface = ARCFACE_WEIGHT * arcface(pooled, disease_labels.float())

                # ---- DEIT DISTILLATION LOSS ----
                distill_scale = 0.5 if cutmix_applied else 1.0
                loss_deit, hard_l, kl_l = deit_criterion(
                    disease_logits, distillation_logits,
                    teacher_logits, disease_labels_for_loss,
                    source_mask, scale=distill_scale,
                )

                # ---- CROP AND SEVERITY LOSSES ----
                loss_crop = F.cross_entropy(crop_logits, crop_labels.long())
                loss_severity = F.cross_entropy(severity_logits, severity_labels.long())

                # ---- TOTAL LOSS ----
                total_loss = (loss_deit +
                              loss_coral +
                              loss_arcface +
                              LOSS_WEIGHT_CROP * loss_crop +
                              LOSS_WEIGHT_SEVERITY * loss_severity)

            # Backprop with mixed precision
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arcface.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            # SWA averaging
            if epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                if not swa_active:
                    swa_active = True
                    print(f'  SWA averaging started at epoch {epoch}')

            # Track losses
            running['total'] += total_loss.item()
            running['disease'] += hard_l
            running['coral'] += loss_coral.item()
            running['arcface'] += loss_arcface.item()
            running['kl'] += kl_l
            running['crop'] += loss_crop.item()
            running['severity'] += loss_severity.item()

        # Step LR scheduler after warmup
        if epoch >= PHASE2B_WARMUP_EPOCHS:
            scheduler.step()

        # ---- VALIDATION WITH EMA WEIGHTS ----
        ema.apply(model)
        val_f1, per_class_f1 = validate_phase2b(model, val_loader, DEVICE)
        ema.restore(model)

        elapsed = time.time() - t0
        n_steps = max(total_steps, 1)
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_heads = optimizer.param_groups[2]['lr']

        # Early stopping and checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            ckpt_path = os.path.join(CKPT_DIR, 'phase2b_best.pt')
            torch.save(model.state_dict(), ckpt_path)
            checkpoint_msg = f'  -> New best: {best_val_f1:.4f} -- checkpoint saved'
        else:
            patience_counter += 1
            checkpoint_msg = f'  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}'

        # Tomato watch
        tomato_above_010 = sum(1 for i in range(10, 19) if per_class_f1[i] > 0.10)

        # Regression detection
        regressions = []
        for cls_name, baseline_f1 in phase2a_baseline.items():
            if baseline_f1 > 0.05:
                idx = CLASS_NAMES.index(cls_name)
                current_f1 = per_class_f1[idx]
                if current_f1 < baseline_f1 - 0.05:
                    regressions.append(f'    WARNING: {cls_name} dropped {baseline_f1:.3f} -> {current_f1:.3f}')

        # VRAM usage
        vram_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == 'cuda' else 0.0

        # SWA status
        swa_status = f'active (started epoch {swa_start_epoch})' if swa_active else f'waiting -- starts epoch {swa_start_epoch}'

        # ---- FORMATTED EPOCH REPORT ----
        print(flush=True)
        print('=' * 59)
        print(f'EPOCH {epoch} | Time: {elapsed/60:.1f}min | LR backbone={lr_backbone:.2e} heads={lr_heads:.2e}')
        print('-' * 59)
        print('LOSSES:')
        print(f'  total={running["total"]/n_steps:.4f}  disease={running["disease"]/n_steps:.4f}  coral={running["coral"]/n_steps:.6f}')
        print(f'  arcface={running["arcface"]/n_steps:.4f}  kl_distill={running["kl"]/n_steps:.4f}')
        print(f'  crop={running["crop"]/n_steps:.4f}  severity={running["severity"]/n_steps:.4f}')
        print('-' * 59)
        print(f'VAL MACRO F1: {val_f1:.4f}  |  Best so far: {best_val_f1:.4f}  |  Patience: {patience_counter}/3')
        print('-' * 59)
        print('PER-CLASS F1 (all 23 classes):')
        print('  OKRA:')
        for i in range(0, 5):
            thin = '  *THIN*' if i in THIN_CLASS_INDICES else ''
            print(f'    {CLASS_NAMES[i]:<34} {per_class_f1[i]:.3f}{thin}')
        print('  BRASSICA:')
        for i in range(5, 10):
            thin = '  *THIN*' if i in THIN_CLASS_INDICES else ''
            print(f'    {CLASS_NAMES[i]:<34} {per_class_f1[i]:.3f}{thin}')
        print('  TOMATO:')
        for i in range(10, 19):
            thin = '  *THIN*' if i in THIN_CLASS_INDICES else ''
            print(f'    {CLASS_NAMES[i]:<34} {per_class_f1[i]:.3f}{thin}')
        print('  CHILLI:')
        for i in range(19, 23):
            thin = '  *THIN*' if i in THIN_CLASS_INDICES else ''
            print(f'    {CLASS_NAMES[i]:<34} {per_class_f1[i]:.3f}{thin}')
        print('-' * 59)
        print(f'TOMATO WATCH: {tomato_above_010} of 9 tomato classes above 0.10 (target: all 9 by epoch 5)')
        print(f'SWA status: {swa_status}')
        print(f'VRAM: {vram_gb:.1f}GB')
        if regressions:
            for r in regressions:
                print(r)
        print(checkpoint_msg)
        print('=' * 59, flush=True)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

    # ================================================================
    # STEP 8: SWA finalisation
    # ================================================================
    print(f'\nFinalising model...')
    if swa_active:
        print('Updating SWA batch norm statistics...')
        swa_model.train()
        with torch.no_grad():
            for batch in train_loader:
                images = batch['image'].to(DEVICE)
                swa_model(images)
        swa_model.eval()
        final_state = swa_model.module.state_dict()
        swa_epochs = max(0, epoch + 1 - swa_start_epoch)
        print(f'SWA: averaged {swa_epochs} epochs (started at epoch {swa_start_epoch})')
    else:
        print('SWA not reached — using best EMA checkpoint')
        best_ckpt = os.path.join(CKPT_DIR, 'phase2b_best.pt')
        if os.path.exists(best_ckpt):
            final_state = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        else:
            final_state = model.state_dict()

    # Save final student model
    student_path = BEST_MODEL if os.path.isabs(BEST_MODEL) else os.path.join(ROOT, BEST_MODEL)
    torch.save(final_state, student_path)
    student_mb = os.path.getsize(student_path) / 1e6
    print(f'Final student model: {student_path} ({student_mb:.1f}MB)')

    # Verify teacher STILL intact
    assert os.path.exists(teacher_path)
    assert os.path.getsize(teacher_path) / 1e6 > 80
    print(f'Teacher preserved: {os.path.getsize(teacher_path)/1e6:.1f}MB INTACT')

    total_time = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'Phase 2B complete.')
    print(f'Best val macro F1: {best_val_f1:.4f}')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'{"="*60}')


if __name__ == '__main__':
    train_phase2b()
