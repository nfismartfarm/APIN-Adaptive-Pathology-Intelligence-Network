# training/04_train_phase1.py
"""
Phase 1: Train classification heads on cached backbone features.
Backbone is frozen. Training runs from pre-computed feature tensors.
This takes 25-35 minutes instead of 3.5 hours.

Run AFTER 03_cache_features.py completes.
Saves: models/checkpoints/phase1_best.pt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [FIX GAP 1] Import helpers at MODULE LEVEL, not inside __main__
from training.helpers import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

from app.config import (
    DEVICE, TRAIN_CACHE, VAL_CACHE, CKPT_DIR, MODELS,
    PHASE1_EPOCHS, PHASE1_LR, BATCH_SIZE, WEIGHT_DECAY,
    LOSS_W_CROP, LOSS_W_DISEASE, LOSS_W_SEVERITY,
    MAX_POS_WEIGHT, EARLY_STOP_PAT, EARLY_STOP_DELTA, KEEP_CKPTS,
    NUM_CLASSES, NUM_CROPS, CROP_EMB_DIM, HEAD_HIDDEN_DIM, POOLED_DIM,
    DROPOUT_P, RANDOM_SEED, WANDB_PROJECT, WANDB_CONFIG,
)
from app.model import PlantDiseaseModel
from training.metrics import compute_multilabel_pos_weights
from training.loss import compute_loss


def set_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase1():
    set_seeds(RANDOM_SEED)

    # ── Load cached features ───────────────────────────────────────────────
    if not os.path.exists(TRAIN_CACHE):
        raise FileNotFoundError(
            f"Training cache not found at {TRAIN_CACHE}. "
            f"Run training/03_cache_features.py first."
        )
    if not os.path.exists(VAL_CACHE):
        raise FileNotFoundError(
            f"Validation cache not found at {VAL_CACHE}. "
            f"Run training/03_cache_features.py first."
        )

    print("Loading cached features...")
    train_cache = torch.load(TRAIN_CACHE, weights_only=False)
    val_cache   = torch.load(VAL_CACHE, weights_only=False)

    # Feature tensors
    train_pooled   = train_cache['pooled_features']    # [N, 256]
    train_crop_emb = train_cache['crop_embeddings']    # [N, 64]
    train_d_lab    = train_cache['disease_labels']     # [N, 10]
    train_c_lab    = train_cache['crop_labels']        # [N]
    train_s_lab    = train_cache['severity_labels']    # [N]

    val_pooled   = val_cache['pooled_features']
    val_crop_emb = val_cache['crop_embeddings']
    val_d_lab    = val_cache['disease_labels']
    val_c_lab    = val_cache['crop_labels']
    val_s_lab    = val_cache['severity_labels']

    print(f"Train features: {train_pooled.shape}, Val features: {val_pooled.shape}")

    # ── [FIX GAP 34] pos_weight: compute from train_d_lab directly ─────────
    # train_d_lab is [N, NUM_CLASSES] multi-hot binary matrix.
    # n_pos[j] = number of training images positive for class j
    # n_neg[j] = N - n_pos[j] = number of training images negative for class j
    # pos_weight[j] = n_neg[j] / n_pos[j]
    # This is correct multi-label formula. NOT sklearn compute_class_weight.
    n_total  = float(train_d_lab.shape[0])
    n_pos    = train_d_lab.float().sum(dim=0).clamp(min=1.0)   # [NUM_CLASSES]
    n_neg    = n_total - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=MAX_POS_WEIGHT)
    print(f"pos_weight range: {pos_weight.min():.2f} to {pos_weight.max():.2f}")

    # ── DataLoaders from cached features ───────────────────────────────────
    train_ds = TensorDataset(train_pooled, train_crop_emb,
                             train_d_lab, train_c_lab, train_s_lab)
    val_ds   = TensorDataset(val_pooled, val_crop_emb,
                             val_d_lab, val_c_lab, val_s_lab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Build model — heads only ────────────────────────────────────────────
    # In Phase 1 we only train heads. The model is instantiated but only
    # head parameters are passed to the optimizer.
    model = PlantDiseaseModel().to(DEVICE)
    model.freeze_backbone()

    head_params = (
        list(model.crop_classifier.parameters()) +
        list(model.disease_head.parameters()) +
        list(model.severity_head.parameters()) +
        list(model.fpn.parameters())
    )
    optimizer = torch.optim.Adam(head_params, lr=PHASE1_LR,
                                 weight_decay=WEIGHT_DECAY)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name='phase1',
        config={**WANDB_CONFIG, 'phase': 1, 'epochs': PHASE1_EPOCHS},
    )

    early_stop  = EarlyStopping(EARLY_STOP_PAT, EARLY_STOP_DELTA)
    best_val_f1 = 0.0

    for epoch in range(PHASE1_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for pooled, crop_emb, d_lab, c_lab, s_lab in train_loader:
            pooled   = pooled.to(DEVICE)
            crop_emb = crop_emb.to(DEVICE)
            d_lab    = d_lab.to(DEVICE)
            c_lab    = c_lab.to(DEVICE)
            s_lab    = s_lab.to(DEVICE)

            # Forward using cached features directly
            # We run only the heads, not the full forward pass
            crop_logits, crop_emb_out = model.crop_classifier(pooled)
            disease_logits = model.disease_head(pooled, crop_emb_out)
            severity_logits = model.severity_head(pooled)

            total_loss, loss_dict = compute_loss(
                crop_logits, disease_logits, severity_logits,
                c_lab, d_lab, s_lab, pos_weight.to(DEVICE)
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_preds_disease = []
        val_true_disease  = []
        val_preds_crop    = []
        val_true_crop     = []
        total_val_loss    = 0.0

        with torch.no_grad():
            for pooled, crop_emb, d_lab, c_lab, s_lab in val_loader:
                pooled   = pooled.to(DEVICE)
                crop_emb = crop_emb.to(DEVICE)
                d_lab_d  = d_lab.to(DEVICE)
                c_lab_d  = c_lab.to(DEVICE)
                s_lab_d  = s_lab.to(DEVICE)

                c_log, _ = model.crop_classifier(pooled)
                d_log    = model.disease_head(pooled, model.crop_classifier(pooled)[1])
                s_log    = model.severity_head(pooled)

                loss, _ = compute_loss(c_log, d_log, s_log,
                                       c_lab_d, d_lab_d, s_lab_d,
                                       pos_weight.to(DEVICE))
                total_val_loss += loss.item()

                val_preds_disease.append(torch.sigmoid(d_log).cpu())
                val_true_disease.append(d_lab)
                val_preds_crop.append(c_log.argmax(dim=1).cpu())
                val_true_crop.append(c_lab)

        # Compute macro F1
        from sklearn.metrics import f1_score
        import numpy as np
        d_preds = (torch.cat(val_preds_disease).numpy() > 0.5).astype(int)
        d_true  = torch.cat(val_true_disease).numpy()
        val_f1  = f1_score(d_true, d_preds, average='macro', zero_division=0)
        crop_acc = (torch.cat(val_preds_crop) == torch.cat(val_true_crop)).float().mean().item()

        train_loss_avg = epoch_loss / max(len(train_loader), 1)
        val_loss_avg   = total_val_loss / max(len(val_loader), 1)

        wandb.log({
            'epoch'       : epoch,
            'train/loss'  : train_loss_avg,
            'val/loss'    : val_loss_avg,
            'val/macro_f1': val_f1,
            'val/crop_acc': crop_acc,
        })
        print(f"Phase1 Epoch {epoch:2d}: "
              f"train_loss={train_loss_avg:.4f}  "
              f"val_loss={val_loss_avg:.4f}  "
              f"val_macro_f1={val_f1:.4f}  "
              f"crop_acc={crop_acc:.3f}")

        # Checkpoint
        ckpt_path = os.path.join(
            CKPT_DIR, f"phase1_epoch{epoch:02d}_f1{val_f1:.3f}.pt"
        )
        save_checkpoint(model, optimizer, None, None, epoch, {'val/macro_f1': val_f1}, ckpt_path)
        cleanup_old_checkpoints(CKPT_DIR, KEEP_CKPTS, phase='phase1')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path   = os.path.join(CKPT_DIR, 'phase1_best.pt')
            save_checkpoint(model, optimizer, None, None, epoch,
                           {'val/macro_f1': val_f1}, best_path)
            print(f"  → New best phase1 model: macro_f1={val_f1:.4f}")

        if early_stop(val_f1):
            print(f"Early stopping at epoch {epoch}")
            break

    wandb.finish()
    print(f"\nPhase 1 complete. Best macro F1: {best_val_f1:.4f}")
    if best_val_f1 < 0.30:
        print("WARNING: macro F1 < 0.30. Check data pipeline and labels.")


if __name__ == '__main__':
    train_phase1()
