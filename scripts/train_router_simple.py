"""
Simple Router Training Script — Fallback Recipe (with Feature Caching)

Minimal recipe guaranteed to produce a working router in ~15 minutes.
Uses frozen DINOv2-Small+Registers backbone with a linear classification head.

FEATURE CACHING: Since the backbone is frozen, we pre-compute all DINOv2
features ONCE (~10 min), then train the linear head on cached feature vectors
for all epochs (~5 seconds/epoch). This is 100x faster than running the full
forward pass every epoch.

Per MASTER_PLAN Section 4 (Router spec):
- DINOv2-Small+Registers frozen backbone, Linear(384→4) head
- AdamW, cosine LR with 5-epoch warmup, 20 epochs
- WeightedRandomSampler with ENS weights + field photo 5x boost
- EMA decay=0.9999
- BF16 autocast
- No ASAM, no SupCon, no CutMix, no MC Dropout

Usage:
    python scripts/train_router_simple.py                  # full 20 epochs
    python scripts/train_router_simple.py --smoke-test     # 3 epochs
    python scripts/train_router_simple.py --no-cache       # skip caching (slow)
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
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config_router import (
    BACKBONE_NAME, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM,
    NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX,
    NUM_EPOCHS, BATCH_SIZE, WEIGHT_DECAY, RANDOM_SEED,
    FIELD_PHOTO_WEIGHT_MULTIPLIER, EARLY_STOPPING_PATIENCE,
)
from scripts.models import RouterDINO
from scripts.train_utils import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    setup_ema, get_augmentation_pipeline, get_eval_transform,
    evaluate, compute_ens_class_weights,
)

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')


class RouterDataset(Dataset):
    """Simple image dataset for router training."""

    def __init__(self, image_paths, labels, transform=None, use_clahe=True):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # Try CLAHE version first
        if self.use_clahe:
            clahe_path = path.replace('/cleaned/', '/cleaned_clahe/').replace('\\cleaned\\', '\\cleaned_clahe\\')
            if os.path.exists(clahe_path):
                path = clahe_path

        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)


@torch.no_grad()
def cache_backbone_features(backbone, dataset, device, batch_size=64):
    """
    Pre-compute frozen backbone features for all images.
    Returns (features_tensor [N, 384], labels_tensor [N]).
    """
    backbone.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device == 'cuda'))

    all_features = []
    all_labels = []
    total = len(dataset)
    processed = 0

    for images, labels in loader:
        images = images.to(device, dtype=torch.bfloat16)
        features = backbone(images).float().cpu()  # [B, 384]
        all_features.append(features)
        all_labels.append(labels)
        processed += len(images)
        if processed % (batch_size * 10) == 0 or processed >= total:
            print(f'  Caching: {processed}/{total} ({processed/total*100:.0f}%)', flush=True)

    return torch.cat(all_features), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Simple Router Training')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run 3 epochs only as infrastructure test')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable feature caching (run full forward pass every epoch)')
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 3

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print(f'ROUTER SIMPLE TRAINING — {args.epochs} epochs')
    print(f'Feature caching: {"DISABLED" if args.no_cache else "ENABLED"}')
    print('=' * 70)
    print(flush=True)

    # ��─ Load data ──────────────────────────────────────────────────────────
    split_path = ROOT / 'data' / 'specialist' / 'router' / 'split_indices.json'
    csv_path = ROOT / 'data' / 'specialist' / 'router' / 'router_unified_source_map.csv'

    import pandas as pd
    df = pd.read_csv(csv_path)

    if 'class_name' not in df.columns:
        df['class_name'] = df['crop']

    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
        train_idx = splits.get('train', list(range(int(len(df) * 0.75))))
        val_idx = splits.get('val', list(range(int(len(df) * 0.75), len(df))))
    else:
        print('WARNING: No split_indices.json found — using random 75/25 split')
        n = len(df)
        perm = np.random.permutation(n)
        train_idx = perm[:int(n * 0.75)].tolist()
        val_idx = perm[int(n * 0.75):].tolist()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_labels = [CLASS_TO_IDX[c] for c in train_df['crop']]
    val_labels = [CLASS_TO_IDX[c] for c in val_df['crop']]

    print(f'Train: {len(train_df)} images', flush=True)
    print(f'Val: {len(val_df)} images', flush=True)
    print(f'Class distribution (train): {dict(train_df["crop"].value_counts())}', flush=True)
    print(flush=True)

    # ── Sampling weights ──────────────────────────────────────────────────
    class_counts = [int((train_df['crop'] == cls).sum()) for cls in CLASS_NAMES]
    ens = compute_ens_class_weights(class_counts)

    is_field = train_df['is_field_photo'].astype(str).str.lower().isin(['true'])
    sample_weights = np.ones(len(train_df))
    for i, cls in enumerate(CLASS_NAMES):
        mask = train_df['crop'] == cls
        sample_weights[mask.values] = float(ens[i])
        sample_weights[(mask & is_field).values] *= FIELD_PHOTO_WEIGHT_MULTIPLIER

    # ── Model ─────────────────────────────────────────────────────────────
    model = RouterDINO(num_classes=NUM_CLASSES, pretrained=True).to(device)
    model = model.to(torch.bfloat16)

    # ── Feature Caching (or direct training) ─────────────────────────────
    if not args.no_cache:
        # CACHED PATH: pre-compute backbone features once, then train head on tensors
        print('Phase A: Caching backbone features...', flush=True)
        t_cache_start = time.time()

        # Use eval transform for caching (no augmentation — frozen backbone, deterministic)
        eval_transform = get_eval_transform(img_size=DINOV2_IMG_SIZE)
        train_ds_for_cache = RouterDataset(
            train_df['image_path'].tolist(), train_labels,
            transform=eval_transform
        )
        val_ds_for_cache = RouterDataset(
            val_df['image_path'].tolist(), val_labels,
            transform=eval_transform
        )

        train_features, train_label_tensor = cache_backbone_features(
            model.backbone, train_ds_for_cache, device, batch_size=BATCH_SIZE
        )
        val_features, val_label_tensor = cache_backbone_features(
            model.backbone, val_ds_for_cache, device, batch_size=BATCH_SIZE
        )

        cache_time = time.time() - t_cache_start
        print(f'Caching complete: {cache_time:.0f}s', flush=True)
        print(f'  Train features: {train_features.shape}', flush=True)
        print(f'  Val features: {val_features.shape}', flush=True)
        print(flush=True)

        # Build TensorDataset loaders from cached features
        train_tensor_ds = TensorDataset(train_features, train_label_tensor)
        val_tensor_ds = TensorDataset(val_features, val_label_tensor)

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)

        train_loader = DataLoader(train_tensor_ds, batch_size=BATCH_SIZE,
                                   sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_tensor_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)

        # Training mode: head-only on cached features
        use_cached = True
    else:
        # UNCACHED PATH: full forward pass every epoch
        train_transform = get_augmentation_pipeline(img_size=DINOV2_IMG_SIZE)
        eval_transform = get_eval_transform(img_size=DINOV2_IMG_SIZE)

        train_dataset = RouterDataset(
            train_df['image_path'].tolist(), train_labels,
            transform=train_transform
        )
        val_dataset = RouterDataset(
            val_df['image_path'].tolist(), val_labels,
            transform=eval_transform
        )

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                   sampler=sampler, num_workers=0,
                                   pin_memory=(device == 'cuda'))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0,
                                 pin_memory=(device == 'cuda'))
        use_cached = False

    # ── Optimizer + scheduler ────────────────────────────────────────────
    ema = setup_ema(model, decay=0.9999, device=device)
    optimizer = torch.optim.AdamW(model.get_trainable_params(),
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)

    warmup_epochs = min(5, args.epochs // 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=1e-6
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_dir = ROOT / 'models' / 'router'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────��───────────────────────────────
    print(f'Phase B: Training head ({args.epochs} epochs)...', flush=True)
    best_f1 = 0.0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_cached:
            # CACHED: train head directly on feature tensors
            for features, labels in train_loader:
                # Cast to model dtype (BF16) — cached features are float32
                features = features.to(device=device, dtype=model.head.weight.dtype)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model.head(features)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if ema is not None:
                    ema.update(model)

                epoch_loss += loss.item()
                n_batches += 1

            # Cached validation
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device=device, dtype=model.head.weight.dtype)
                    logits = model.head(features)
                    preds = logits.float().argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(labels.numpy())

            from sklearn.metrics import f1_score
            val_preds = np.array(val_preds)
            val_true = np.array(val_true)
            val_f1 = float(f1_score(val_true, val_preds, average='macro', zero_division=0))
            per_class = f1_score(val_true, val_preds, average=None, zero_division=0)
            metrics = {'macro_f1': val_f1}
            for i, name in enumerate(CLASS_NAMES):
                if i < len(per_class):
                    metrics[f'f1_{name}'] = float(per_class[i])

        else:
            # UNCACHED: full forward pass
            for images, labels in train_loader:
                images = images.to(device, dtype=torch.bfloat16)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if ema is not None:
                    ema.update(model)

                epoch_loss += loss.item()
                n_batches += 1

            metrics = evaluate(model, val_loader, device=device,
                              class_names=CLASS_NAMES, num_classes=NUM_CLASSES)
            val_f1 = metrics['macro_f1']

        # Scheduler
        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t_start
        print(f'Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f} '
              f'val_f1={val_f1:.4f} lr={lr:.2e} ({elapsed:.0f}s)', flush=True)

        # Per-class F1
        for cls in CLASS_NAMES:
            cls_f1 = metrics.get(f'f1_{cls}', 0)
            if cls_f1 < 0.70:
                print(f'  WARNING: {cls} F1={cls_f1:.3f} < 0.70', flush=True)

        # Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                epoch, model, ema, optimizer, scheduler, None,
                best_f1, str(ckpt_dir / 'router_best.pt')
            )
            print(f'  -> New best: {best_f1:.4f}', flush=True)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})', flush=True)
                break

    # ── Save final model ──────────────────────────────────────────────────
    print(flush=True)
    print(f'Training complete. Best val F1: {best_f1:.4f}', flush=True)
    print(f'Total time: {time.time() - t_start:.0f}s', flush=True)
    if not args.no_cache:
        print(f'  (includes {cache_time:.0f}s feature caching)', flush=True)

    # Check acceptance criteria
    with open(ROOT / 'acceptance_criteria.json') as f:
        criteria = json.load(f)
    min_f1 = criteria['router']['min_macro_f1']
    if best_f1 >= min_f1:
        print(f'PASS: F1 {best_f1:.4f} >= {min_f1} threshold', flush=True)
    else:
        print(f'FAIL: F1 {best_f1:.4f} < {min_f1} threshold', flush=True)
        print(f'Action: {criteria["router"]["action_below_minimum"]}', flush=True)

    print('\nTEST RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
