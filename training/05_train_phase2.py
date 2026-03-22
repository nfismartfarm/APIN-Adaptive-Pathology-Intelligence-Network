# training/05_train_phase2.py
"""
Phase 2: Full fine-tuning with top 1/3 of backbone unfrozen.
Loads phase1_best.pt as starting weights.
Uses LLRD optimizer and OneCycleLR scheduler.
Saves: models/best_model.pt (state_dict only — use for inference)
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [FIX GAP 65] Load environment variables at module level
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional for manual execution

# [FIX GAP 1, 42] Import ALL training helpers at MODULE LEVEL
from training.helpers import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    get_llrd_optimizer,
)

import torch
import torch.nn as nn
# [FIX GAP 69] Use torch.amp, not deprecated torch.cuda.amp
from torch.amp import autocast, GradScaler
import wandb

from app.config import (
    DEVICE, SOURCE_MAP, SEV_LABELS, CKPT_DIR, MODELS, BEST_MODEL,
    PHASE2_EPOCHS, PHASE2_BASE_LR, LLRD_DECAY, WEIGHT_DECAY,
    BATCH_SIZE, GRAD_ACCUM_STEPS, GRAD_CLIP_NORM,
    EARLY_STOP_PAT, EARLY_STOP_DELTA, KEEP_CKPTS, RANDOM_SEED,
    ONE_CYCLE_PCT, ONE_CYCLE_DIV, ONE_CYCLE_FDIV,   # [FIX GAP 35]
    WANDB_PROJECT, WANDB_CONFIG,
)
from app.model import PlantDiseaseModel, verify_backbone_shapes
from training.dataset import PlantDiseaseDataset, load_severity_labels, make_weighted_sampler
from training.transforms import get_train_transform, get_eval_transform
from training.loss import compute_loss
from training.metrics import compute_all_metrics, warn_on_thin_classes


def set_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase2(train_records, val_records):
    # [FIX GAP 6] Phase 2 must call set_seeds() — was missing in v5
    set_seeds(RANDOM_SEED)

    # [FIX GAP 66] WANDB_MODE=offline if no API key, so training is not blocked
    if not os.environ.get('WANDB_API_KEY'):
        os.environ.setdefault('WANDB_MODE', 'offline')
        print("WANDB_API_KEY not set. Running wandb in offline mode.")

    is_windows = sys.platform.startswith('win')
    n_workers  = 0 if is_windows else 2

    # ── Load model from Phase 1 best ──────────────────────────────────────
    phase1_best = os.path.join(CKPT_DIR, 'phase1_best.pt')
    if not os.path.exists(phase1_best):
        raise FileNotFoundError(
            f"Phase 1 best checkpoint not found at {phase1_best}. "
            f"Run training/04_train_phase1.py first."
        )

    model = PlantDiseaseModel().to(DEVICE)
    verify_backbone_shapes(model, device=DEVICE)

    # Load Phase 1 weights
    ckpt  = torch.load(phase1_best, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded Phase 1 weights from {phase1_best}")

    # Unfreeze top 1/3 of backbone blocks
    model.unfreeze_top_fraction(fraction=0.33)

    # ── DataLoaders ───────────────────────────────────────────────────────
    sev_labels  = load_severity_labels()
    train_ds    = PlantDiseaseDataset(train_records, get_train_transform(), sev_labels)
    val_ds      = PlantDiseaseDataset(val_records,   get_eval_transform(),  sev_labels)
    sampler     = make_weighted_sampler(train_records)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=False,
        prefetch_factor=2 if n_workers > 0 else None,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=n_workers,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=False,
        drop_last=False,
    )

    # ── [FIX GAP 34] pos_weight from binary label matrix ──────────────────
    import pandas as pd
    import numpy as np
    from app.config import CLASS_TO_IDX, NUM_CLASSES, MAX_POS_WEIGHT
    d_labels_all = torch.zeros(len(train_records), NUM_CLASSES)
    for i, r in enumerate(train_records):
        idx = r.get('class_idx', -1)
        if 0 <= idx < NUM_CLASSES:
            d_labels_all[i, idx] = 1.0
    n_total   = float(len(train_records))
    n_pos     = d_labels_all.sum(dim=0).clamp(min=1.0)
    n_neg     = n_total - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=MAX_POS_WEIGHT)

    # ── [FIX GAP 42] LLRD optimizer from helpers only ─────────────────────
    optimizer = get_llrd_optimizer(model, PHASE2_BASE_LR, LLRD_DECAY, WEIGHT_DECAY)

    # ── OneCycleLR — [FIX GAP 35] use config constants ────────────────────
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[pg['lr'] for pg in optimizer.param_groups],
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=ONE_CYCLE_PCT,      # from config, not hardcoded
        anneal_strategy='cos',
        div_factor=ONE_CYCLE_DIV,     # from config
        final_div_factor=ONE_CYCLE_FDIV,  # from config
    )

    # ── Mixed precision ────────────────────────────────────────────────────
    use_amp = (DEVICE.type == 'cuda')
    # [FIX GAP 69] GradScaler from torch.amp, device_type parameter required
    scaler  = GradScaler(device='cuda' if use_amp else 'cpu', enabled=use_amp)

    # ── torch.compile ─────────────────────────────────────────────────────
    # [FIX GAP 45] compiled flag is actually used for model unwrapping
    compiled = False
    if use_amp and not is_windows:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            compiled = True
            print("torch.compile enabled (25-35% speedup)")
        except Exception as e:
            print(f"torch.compile unavailable: {e}. Continuing without.")

    # ── [FIX GAP 71] Resume from latest phase2 checkpoint if it exists ────
    start_epoch = 0
    phase2_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'phase2_epoch*.pt')))
    if phase2_ckpts:
        latest = phase2_ckpts[-1]
        print(f"Resuming Phase 2 from checkpoint: {latest}")
        raw_model = getattr(model, '_orig_mod', model)
        resume_epoch, resume_metrics = load_checkpoint(
            raw_model, optimizer, scheduler, scaler, latest, DEVICE
        )
        start_epoch = resume_epoch + 1
        print(f"Resumed from epoch {resume_epoch}, "
              f"val_f1={resume_metrics.get('val/macro_f1', 0):.4f}")

    wandb.init(
        project=WANDB_PROJECT,
        name='phase2',
        config={**WANDB_CONFIG, 'phase': 2, 'amp': use_amp,
                'resume_epoch': start_epoch},
    )

    early_stop  = EarlyStopping(EARLY_STOP_PAT, EARLY_STOP_DELTA)
    best_val_f1 = 0.0

    for epoch in range(start_epoch, PHASE2_EPOCHS):
        model.train()
        epoch_loss    = 0.0
        accum_counter = 0
        optimizer.zero_grad()

        for batch_idx, (images, d_lab, c_lab, s_lab) in enumerate(train_loader):
            images = images.to(DEVICE)
            d_lab  = d_lab.to(DEVICE)
            c_lab  = c_lab.to(DEVICE)
            s_lab  = s_lab.to(DEVICE)

            # [FIX GAP 69] device_type parameter required in torch.amp.autocast
            with autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                c_log, d_log, s_log = model(images)
                total_loss, _ = compute_loss(
                    c_log, d_log, s_log, c_lab, d_lab, s_lab,
                    pos_weight.to(DEVICE)
                )
                scaled_loss = total_loss / GRAD_ACCUM_STEPS

            scaler.scale(scaled_loss).backward()
            accum_counter += 1

            if accum_counter % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                wandb.log({'train/grad_norm': grad_norm.item(),
                           'train/lr': scheduler.get_last_lr()[0]})

            epoch_loss += total_loss.item()

        # Flush incomplete accumulation window at epoch end
        if accum_counter % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        val_metrics = compute_all_metrics(model, val_loader, pos_weight,
                                          DEVICE, phase='phase2_full')
        val_f1 = val_metrics.get('val/macro_f1', 0.0)

        wandb.log({'epoch': epoch,
                   'train/loss': epoch_loss / max(len(train_loader), 1),
                   **val_metrics})
        print(f"Phase2 Epoch {epoch:2d}: "
              f"loss={epoch_loss / len(train_loader):.4f}  "
              f"val_macro_f1={val_f1:.4f}")

        warn_on_thin_classes(val_metrics, epoch)

        # [FIX GAP 45] Use compiled flag to unwrap model for saving
        raw_model = getattr(model, '_orig_mod', model) if compiled else model
        ckpt_path = os.path.join(
            CKPT_DIR, f"phase2_epoch{epoch:02d}_f1{val_f1:.3f}.pt"
        )
        save_checkpoint(raw_model, optimizer, scheduler, scaler,
                        epoch, val_metrics, ckpt_path)
        cleanup_old_checkpoints(CKPT_DIR, KEEP_CKPTS, phase='phase2')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {'model_state_dict': raw_model.state_dict(),
                 'val_metrics': val_metrics,
                 'epoch': epoch},
                BEST_MODEL
            )
            best_ckpt = os.path.join(CKPT_DIR, 'phase2_best.pt')
            save_checkpoint(raw_model, optimizer, scheduler, scaler,
                            epoch, val_metrics, best_ckpt)
            print(f"  → Best model saved: macro_f1={val_f1:.4f}")

        if early_stop(val_f1):
            print(f"Early stopping at epoch {epoch}")
            break

    wandb.finish()
    print(f"\nPhase 2 complete. Best macro F1: {best_val_f1:.4f}")
    if best_val_f1 < 0.50:
        print("WARNING: macro F1 < 0.50. Check data balance and training setup.")


if __name__ == '__main__':
    import pandas as pd
    from app.config import CLASS_TO_IDX, CROP_FROM_IDX, SOURCE_MAP

    df = pd.read_csv(SOURCE_MAP)
    train_records = df[df['split'] == 'train'].to_dict('records')
    val_records   = df[df['split'] == 'val'].to_dict('records')
    for r in train_records + val_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

    train_phase2(train_records, val_records)
