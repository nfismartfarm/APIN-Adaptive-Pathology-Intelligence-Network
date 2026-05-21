"""
PSV MLP Decision Maker — 4-signal fusion with adversarial training.

Input: 36 values (9 Model2 + 9 EfficientNet + 9 PSV + 9 DINOv2)
Architecture: 36 -> 64 -> 32 -> 9
Output: softmax over 9 classes

Includes:
  - Stratified k-fold training
  - Adversarial augmentation for failure classes
  - Online learning with feedback buffer
  - Temperature scaling for calibrated confidence
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from scripts.psv.config import PSV_CFG
from scripts.psv.adversarial_augment import generate_adversarial_points, compute_sample_weights


class PSVDecisionMLP(nn.Module):
    """Small MLP for 4-signal fusion. Architecture: 36 -> 64 -> 32 -> 9."""

    def __init__(self, input_dim=36, hidden_dims=(64, 32),
                 output_dim=9, dropout=0.35):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              weights: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = None, lr: float = None,
              verbose: bool = True) -> Tuple[PSVDecisionMLP, Dict]:
    """
    Train the MLP decision maker.

    Args:
        X_train: [N, 36] feature vectors
        y_train: [N] integer class labels
        weights: [N] per-sample weights
        X_val: validation features
        y_val: validation labels
        epochs: override config epochs
        lr: override config LR

    Returns:
        (trained_model, metrics_dict)
    """
    cfg = PSV_CFG
    epochs = epochs or cfg.MLP_EPOCHS
    lr = lr or cfg.MLP_LR
    device = 'cpu'  # PSV MLP always on CPU

    model = PSVDecisionMLP(
        input_dim=cfg.MLP_INPUT_DIM,
        hidden_dims=cfg.MLP_HIDDEN_DIMS,
        output_dim=cfg.MLP_OUTPUT_DIM,
        dropout=cfg.MLP_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=cfg.MLP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # Prepare data
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    if weights is not None:
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(TensorDataset(X_t, y_t),
                           batch_size=cfg.MLP_BATCH_SIZE, sampler=sampler)
    else:
        loader = DataLoader(TensorDataset(X_t, y_t),
                           batch_size=cfg.MLP_BATCH_SIZE, shuffle=True)

    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val, dtype=torch.float32))
                val_preds = val_logits.argmax(dim=1).numpy()
            val_f1 = f1_score(y_val, val_preds, average='macro',
                             labels=list(range(cfg.NUM_CLASSES)), zero_division=0)
            per_class = f1_score(y_val, val_preds, average=None,
                               labels=list(range(cfg.NUM_CLASSES)), zero_division=0)

            # Track best by minimum of failure class F1s
            failure_f1s = []
            for fc in cfg.FAILURE_CLASSES:
                if fc in cfg.CLASS_NAMES:
                    idx = cfg.CLASS_NAMES.index(fc)
                    if idx < len(per_class):
                        failure_f1s.append(per_class[idx])
            min_failure = min(failure_f1s) if failure_f1s else val_f1
            score = min(min_failure, val_f1)

            if score > best_f1:
                best_f1 = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f'  Ep {epoch:3d}: loss={total_loss/n_batches:.4f} '
                      f'val_f1={val_f1:.4f} min_failure={min_failure:.4f}', flush=True)

            if patience_counter >= cfg.MLP_EARLY_STOP_PATIENCE:
                if verbose:
                    print(f'  Early stop at epoch {epoch}', flush=True)
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {'best_f1': best_f1, 'epochs_trained': epoch + 1}
    if X_val is not None and y_val is not None:
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32))
            val_preds = val_logits.argmax(dim=1).numpy()
        per_class = f1_score(y_val, val_preds, average=None,
                           labels=list(range(cfg.NUM_CLASSES)), zero_division=0)
        for i, cls in enumerate(cfg.CLASS_NAMES):
            metrics[f'f1_{cls}'] = float(per_class[i]) if i < len(per_class) else 0.0
        metrics['macro_f1'] = float(f1_score(y_val, val_preds, average='macro',
                                            labels=list(range(cfg.NUM_CLASSES)),
                                            zero_division=0))

    return model, metrics


def run_kfold_training(X: np.ndarray, y: np.ndarray, weights: np.ndarray = None,
                       k: int = None, verbose: bool = True) -> Tuple[PSVDecisionMLP, Dict]:
    """
    Stratified k-fold cross-validation training.

    Returns the best model and per-fold metrics.
    """
    cfg = PSV_CFG
    k = k or cfg.MLP_K_FOLDS

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    best_model = None
    best_overall = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f'\nFold {fold + 1}/{k}:', flush=True)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = weights[train_idx] if weights is not None else None

        # Add adversarial points to training
        adv_X, adv_y, adv_w = generate_adversarial_points(X_train, y_train)
        if len(adv_X) > 0:
            X_train = np.concatenate([X_train, adv_X])
            y_train = np.concatenate([y_train, adv_y])
            if w_train is not None:
                w_train = np.concatenate([w_train, adv_w])
            if verbose:
                print(f'  Added {len(adv_X)} adversarial points', flush=True)

        model, metrics = train_mlp(X_train, y_train, w_train, X_val, y_val,
                                   verbose=verbose)
        fold_metrics.append(metrics)

        score = metrics.get('best_f1', 0)
        if score > best_overall:
            best_overall = score
            best_model = model

    # Aggregate metrics
    agg = {}
    for key in fold_metrics[0]:
        if key.startswith('f1_') or key == 'macro_f1' or key == 'best_f1':
            vals = [fm[key] for fm in fold_metrics if key in fm]
            agg[key] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))

    if verbose:
        print(f'\n{"="*50}')
        print(f'K-Fold Results (k={k}):')
        print(f'  Macro F1: {agg.get("macro_f1", 0):.4f} +/- {agg.get("macro_f1_std", 0):.4f}')
        for cls in cfg.CLASS_NAMES:
            key = f'f1_{cls}'
            print(f'  {cls:30s}: {agg.get(key, 0):.4f} +/- {agg.get(f"{key}_std", 0):.4f}')

    return best_model, agg


def save_mlp(model: PSVDecisionMLP, path: str = None):
    """Save MLP checkpoint."""
    if path is None:
        path = os.path.join(PSV_CFG.ROOT, PSV_CFG.MLP_CHECKPOINT_PATH)
    torch.save({
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }, path)


def load_mlp(path: str = None) -> Optional[PSVDecisionMLP]:
    """Load MLP from checkpoint."""
    if path is None:
        path = os.path.join(PSV_CFG.ROOT, PSV_CFG.MLP_CHECKPOINT_PATH)
    if not os.path.exists(path):
        return None
    cfg = PSV_CFG
    model = PSVDecisionMLP(cfg.MLP_INPUT_DIM, cfg.MLP_HIDDEN_DIMS,
                           cfg.MLP_OUTPUT_DIM, cfg.MLP_DROPOUT)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# ONLINE LEARNING — Feedback Buffer
# ═══════════════════════════════════════════════════════════════════════

class FeedbackBuffer:
    """Persistent feedback storage for online MLP retraining."""

    def __init__(self, path: str = None):
        self.path = path or os.path.join(PSV_CFG.ROOT, PSV_CFG.FEEDBACK_BUFFER_PATH)
        self.entries = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
                self.entries = data.get('entries', [])

    def _save(self):
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else '.', exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({'entries': self.entries}, f, indent=2)

    def add(self, feature_vector: list, correct_label: str, source: str = 'unknown',
            psv_confidence: float = 1.0, quality_flags: list = None):
        """Add a feedback entry."""
        self.entries.append({
            'feature_vector': feature_vector,
            'correct_label': correct_label,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'psv_confidence': psv_confidence,
            'quality_flags': quality_flags or [],
        })
        self._save()

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert buffer to training arrays with recency weighting."""
        if not self.entries:
            return np.array([]).reshape(0, PSV_CFG.MLP_INPUT_DIM), np.array([]), np.array([])

        cfg = PSV_CFG
        class_to_idx = {n: i for i, n in enumerate(cfg.CLASS_NAMES)}
        X, y, w = [], [], []

        now = datetime.now()
        for entry in self.entries:
            if entry['correct_label'] not in class_to_idx:
                continue
            X.append(entry['feature_vector'])
            y.append(class_to_idx[entry['correct_label']])

            # Recency weighting
            try:
                ts = datetime.fromisoformat(entry['timestamp'])
                days_old = (now - ts).days
                if days_old <= 7:
                    weight = cfg.RECENCY_WEIGHT_7DAYS
                elif days_old <= 30:
                    weight = cfg.RECENCY_WEIGHT_30DAYS
                else:
                    weight = cfg.RECENCY_WEIGHT_OLDER
            except:
                weight = cfg.RECENCY_WEIGHT_OLDER
            w.append(weight)

        if not X:
            return np.array([]).reshape(0, PSV_CFG.MLP_INPUT_DIM), np.array([]), np.array([])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(w, dtype=np.float32)

    def should_retrain(self) -> bool:
        """Check if enough new feedback has accumulated for retraining."""
        cfg = PSV_CFG
        if len(self.entries) < cfg.MIN_FEEDBACK_FOR_RETRAIN:
            return False
        # Check if any failure class has enough entries
        class_counts = {}
        for entry in self.entries:
            cls = entry['correct_label']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        for fc in cfg.FAILURE_CLASSES:
            if class_counts.get(fc, 0) >= cfg.MIN_FAILURE_CLASS_FEEDBACK:
                return True
        return False

    def __len__(self):
        return len(self.entries)
