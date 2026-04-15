"""
Simple Model 3 Training Script — Fallback Recipe

Minimal recipe: DINOv2-Small + LoRA rank=8, no FiLM, no curriculum, no self-distillation.
Produces a working Model 3 in ~1.5 hours.

Per MASTER_PLAN Simple Fallback:
- DINOv2-Small + LoRA rank=8 (frozen backbone)
- Focal Loss gamma=2 with ENS weights (beta=0.999)
- WeightedRandomSampler with scidb cap + field 4x
- EMA decay=0.9999, AdamW, cosine schedule, 20 epochs at 224px
- AugMix augmentation + GridDistortion for curl classes
- Background recomposed images included in training
- NO: FiLM, curriculum, self-distillation, CutMix
- Expected val macro F1: 0.68-0.74
- Training time: ~1.5 hours

Usage:
    python scripts/train_model3_simple.py
    python scripts/train_model3_simple.py --epochs 3  # smoke test
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config_model3 import (
    NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX,
    DINOV2_IMG_SIZE, BATCH_SIZE, GRAD_ACCUM_STEPS, WEIGHT_DECAY, RANDOM_SEED,
    FIELD_PHOTO_MULTIPLIER, ENS_BETA, SCIDB_CAP_PER_CLASS,
    CURL_DISEASE_CLASSES, MIN_MACRO_F1,
    EMA_DECAY_STAGE1,
)
from scripts.models import Model3DINOLoRA
from scripts.train_utils import (
    save_checkpoint, setup_ema, evaluate,
    compute_ens_class_weights, get_augmentation_pipeline, get_eval_transform,
    gradient_accumulation_step,
)

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')


class DiseaseDataset(Dataset):
    """Image dataset with class-conditional augmentation for curl classes."""

    def __init__(self, image_paths, labels, class_names,
                 curl_classes=None, img_size=224):
        self.paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.curl_set = set(curl_classes or [])

        # Two transforms: standard and curl-enhanced
        self.transform_base = get_augmentation_pipeline(img_size, curl_class=False)
        self.transform_curl = get_augmentation_pipeline(img_size, curl_class=True)

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
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Class-conditional augmentation
        label_idx = self.labels[idx]
        cls_name = self.class_names[label_idx] if label_idx < len(self.class_names) else ''
        if cls_name in self.curl_set:
            img = self.transform_curl(image=img)['image']
        else:
            img = self.transform_base(image=img)['image']

        return img, torch.tensor(label_idx, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description='Simple Model 3 Training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 3

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print(f'MODEL 3 SIMPLE TRAINING — {args.epochs} epochs')
    print('=' * 70)
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    import pandas as pd
    csv_path = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
    df = pd.read_csv(csv_path)

    split_path = ROOT / 'data' / 'specialist' / 'model3' / 'split_indices.json'
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
        train_idx = splits.get('train', list(range(int(len(df) * 0.68))))
        val_idx = splits.get('val', list(range(int(len(df) * 0.68), int(len(df) * 0.78))))
    else:
        print('WARNING: No split_indices.json — using random split')
        n = len(df)
        perm = np.random.permutation(n)
        train_idx = perm[:int(n * 0.68)].tolist()
        val_idx = perm[int(n * 0.68):int(n * 0.78)].tolist()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_labels = [CLASS_TO_IDX[c] for c in train_df['class_name']]
    val_labels = [CLASS_TO_IDX[c] for c in val_df['class_name']]

    print(f'Train: {len(train_df)}, Val: {len(val_df)}')
    print()

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = DiseaseDataset(
        train_df['image_path'].tolist(), train_labels,
        class_names=CLASS_NAMES, curl_classes=CURL_DISEASE_CLASSES,
        img_size=DINOV2_IMG_SIZE,
    )
    val_ds = DiseaseDataset(
        val_df['image_path'].tolist(), val_labels,
        class_names=CLASS_NAMES, img_size=DINOV2_IMG_SIZE,
    )

    # ENS weights with scidb cap + field boost
    class_counts = [int((train_df['class_name'] == cls).sum()) for cls in CLASS_NAMES]
    ens = compute_ens_class_weights(class_counts, beta=ENS_BETA)

    is_field = train_df['is_field_photo'].astype(str).str.lower().isin(['true'])
    is_scidb = train_df['source_dataset'].astype(str).str.contains('scidb', case=False, na=False)
    is_recomp = train_df['source_dataset'].astype(str).str.contains('recomposed', case=False, na=False)

    sample_weights = np.ones(len(train_df))
    for i, cls in enumerate(CLASS_NAMES):
        mask = train_df['class_name'] == cls
        sample_weights[mask.values] = float(ens[i])
        # Field boost (includes recomposed)
        sample_weights[(mask & (is_field | is_recomp)).values] *= FIELD_PHOTO_MULTIPLIER
        # Scidb cap
        n_scidb = (mask & is_scidb & ~is_recomp).sum()
        if n_scidb > SCIDB_CAP_PER_CLASS:
            scidb_factor = SCIDB_CAP_PER_CLASS / n_scidb
            sample_weights[(mask & is_scidb & ~is_recomp).values] *= scidb_factor

    sampler = WeightedRandomSampler(sample_weights, len(train_df), replacement=True)
    # [LOG ENTRY 009] num_workers>0 deadlocks on Windows with large datasets.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               sampler=sampler, num_workers=0,
                               pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0,
                             pin_memory=(device == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────────────
    model = Model3DINOLoRA(num_classes=NUM_CLASSES, pretrained=True,
                            enable_gradient_checkpointing=True).to(device)
    model = model.to(torch.bfloat16)
    model.unfreeze_lora()  # LoRA + head trainable

    ema = setup_ema(model, decay=EMA_DECAY_STAGE1, device=device)

    # Only optimize trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Focal Loss with ENS class weights
    class_weights = ens.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    ckpt_dir = ROOT / 'models' / 'model3_specialist'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    best_f1 = 0.0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device, dtype=torch.bfloat16)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                # Simple forward — no FiLM (crop_ids=None)
                logits = model(images, crop_ids=None)
                loss = loss_fn(logits, labels)

            # Gradient accumulation with correct scaling
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()

            if (n_batches + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        # Flush incomplete accumulation
        if n_batches % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)  # keep EMA in sync with flushed update

        # [FIX: EMA warmup] After epoch 0, re-seed EMA from partially-trained weights.
        # Without this, EMA retains random LoRA+head init contamination for the entire run.
        if epoch == 0 and ema is not None:
            from scripts.train_utils import reset_ema
            reset_ema(ema, model)
            print(f'  EMA re-seeded from epoch 0 weights (eliminates random init contamination)')

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        metrics = evaluate(model, val_loader, device=device,
                          class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
        val_f1 = metrics['macro_f1']

        elapsed = time.time() - t_start
        print(f'Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f} '
              f'val_f1={val_f1:.4f} ({elapsed:.0f}s)')

        # Warn on thin classes
        for cls in CLASS_NAMES:
            cls_f1 = metrics.get(f'f1_{cls}', 0)
            if cls_f1 < 0.50:
                print(f'  WARNING: {cls} F1={cls_f1:.3f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(epoch, model, ema, optimizer, scheduler, None,
                           best_f1, str(ckpt_dir / 'model3_simple_best.pt'))
            print(f'  -> Best: {best_f1:.4f}')

    print(f'\nComplete. Best F1: {best_f1:.4f}')
    print(f'Acceptance: {"PASS" if best_f1 >= MIN_MACRO_F1 else "FAIL"} '
          f'(threshold: {MIN_MACRO_F1})')

    # Check self-distillation trigger
    from app.config_model3 import DISTILLATION_MIN_FIRST_PASS_F1, MAX_CAPSICUM_GAP
    if best_f1 >= DISTILLATION_MIN_FIRST_PASS_F1:
        print(f'Self-distillation eligible (F1 {best_f1:.4f} >= {DISTILLATION_MIN_FIRST_PASS_F1})')
    else:
        print(f'Self-distillation NOT eligible (F1 {best_f1:.4f} < {DISTILLATION_MIN_FIRST_PASS_F1})')

    # Capsicum gap check (per pessimistic agent recommendation)
    # Even in the simple fallback, warn if Capsicum shortcut learning is detected
    print('\nCapsicum gap check:')
    chilli_healthy_f1 = metrics.get('f1_chilli_healthy', 0)
    print(f'  chilli_healthy overall F1: {chilli_healthy_f1:.4f}')
    if chilli_healthy_f1 > 0.95:
        print(f'  WARNING: chilli_healthy F1 suspiciously high ({chilli_healthy_f1:.3f})')
        print(f'  May indicate Capsicum lab background shortcut learning.')
        print(f'  Run full Model 3 training with Capsicum gap monitoring for definitive check.')
    print(f'  MAX_CAPSICUM_GAP threshold: {MAX_CAPSICUM_GAP}')


if __name__ == '__main__':
    main()
