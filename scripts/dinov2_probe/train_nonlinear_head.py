"""
DINOv2 Nonlinear Head Training — Signal 4 of APIN ensemble.

Trains a nonlinear MLP on top of frozen DINOv2-Small-Registers cls_mean features
(already cached on disk at scripts/dinov2_probe/results/dinov2_features_cache.pkl).

Architecture choices (research-validated; see architecture_claude_decisions.md):
    Topology: 768 -> 512 -> 256 -> 9  (2-hidden MLP, moderate capacity)
    Normalisation: LayerNorm (NOT BatchNorm — ViT features have no batch stats)
    Activation: GELU
    Dropout: 0.3 (between each linear layer)
    Optimizer: AdamW (no ASAM — noise on a <1M param head)
    Weight decay: 1e-2
    LR: 1e-3 peak, cosine annealing
    Epochs: 50 max, early stop on "min(black_rot_F1, cercospora_F1, macro_F1)"

Preprocessing:
    StandardScaler on features (fit on train only, applied to val/final_val)
    — research confirmed beats L2-norm for DINOv2 linear-probe setup

Training:
    Field-photo sample weighting: 5x
    5-fold CV stratified by composite (class, source_dataset) key
    Primary metric: min(brassica_black_rot_F1, okra_cercospora_F1, macro_F1)

Outputs (scripts/dinov2_probe/results/):
    dinov2_nonlinear_head_{ts}.pt         — trained MLP weights
    dinov2_nonlinear_head_scaler_{ts}.pkl — fitted StandardScaler
    dinov2_nonlinear_head_config_{ts}.json — full training config
    dinov2_nonlinear_head_results_{ts}.json — full eval results
    dinov2_nonlinear_head_cm_{ts}.png — confusion matrix plot
    dinov2_nonlinear_head_curves_{ts}.png — training curves plot
"""

import os
import sys
import json
import pickle
import random
import logging
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Determine project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dinov2_probe.config import (
    CLASS_NAMES, NUM_CLASSES, FEATURE_AGGREGATION, FEATURE_DIM,
    FEATURES_CACHE_PATH, SPLIT_INDICES, MODEL2_CSV, RESULTS_DIR,
    MODEL2_VAL_F1, FAILURE_CLASSES, RANDOM_SEED,
)

# MODEL2_PER_CLASS_F1 derived from the MODEL2_VAL_F1 dict (drop the 'macro' key)
MODEL2_PER_CLASS_F1 = {k: v for k, v in MODEL2_VAL_F1.items() if k != 'macro'}

# FEATURE_DIM is a dict keyed by aggregation; resolve to the int we expect
FEATURE_DIM_INT = FEATURE_DIM[FEATURE_AGGREGATION] if isinstance(FEATURE_DIM, dict) else FEATURE_DIM

# =============================================================================
# CONFIGURATION
# =============================================================================
HIDDEN_DIMS       = [512, 256]
DROPOUT           = 0.3
LR                = 1e-3
WEIGHT_DECAY      = 1e-2
EPOCHS            = 50
BATCH_SIZE        = 256
FIELD_PHOTO_WEIGHT = 5.0
PATIENCE          = 8          # early stopping on min-class metric
N_FOLDS           = 5
SEED              = RANDOM_SEED
DEVICE_PREF       = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# LOGGING
# =============================================================================
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH  = RESULTS_DIR / f'dinov2_nonlinear_head_{TIMESTAMP}.log'
logger = logging.getLogger('dinov2_head')
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


# =============================================================================
# MODEL
# =============================================================================
class NonlinearHead(nn.Module):
    """2-hidden MLP with LayerNorm + GELU + Dropout.
    768 -> 512 -> 256 -> 9
    """
    def __init__(self, in_dim: int, hidden_dims, num_classes: int, dropout: float):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)
        # Kaiming init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.body(x))


# =============================================================================
# DATA LOADING
# =============================================================================
def load_cache_and_splits():
    """Load the DINOv2 feature cache and the 4-way split indices."""
    logger.info(f"Loading feature cache from {FEATURES_CACHE_PATH}")
    with open(FEATURES_CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)

    # Verify shape — REQUIRED STARTUP CHECK (per prompt Addition 1)
    first_key = next(iter(cache))
    first_feat = cache[first_key]['feature']
    assert first_feat.shape == (FEATURE_DIM_INT,), (
        f"Feature shape mismatch! Cache has {first_feat.shape}, "
        f"config expects ({FEATURE_DIM_INT},)"
    )
    logger.info(
        f"Cache verified: {len(cache)} entries, feature_dim={first_feat.shape[0]}, "
        f"aggregation='{FEATURE_AGGREGATION}'"
    )

    logger.info(f"Loading splits from {SPLIT_INDICES}")
    with open(SPLIT_INDICES) as f:
        splits = json.load(f)

    import pandas as pd
    df = pd.read_csv(MODEL2_CSV)
    logger.info(f"Master CSV loaded: {len(df)} rows")

    datasets = {}
    for out_name, split_key in [('train', 'train'),
                                 ('val',   'val_and_soup'),
                                 ('final_val', 'final_val')]:
        idxs = splits[split_key]
        sub_df = df.iloc[idxs]
        X, y, is_field, sources, paths = [], [], [], [], []
        missing = 0
        for _, row in sub_df.iterrows():
            key = str(row['image_path'])
            if key not in cache:
                missing += 1
                continue
            e = cache[key]
            X.append(e['feature'])
            y.append(e['label'])
            is_field.append(bool(e['is_field_photo']))
            sources.append(str(e['source_dataset']))
            paths.append(key)
        datasets[out_name] = {
            'X': np.array(X, dtype=np.float32),
            'y': np.array(y, dtype=np.int64),
            'is_field': np.array(is_field, dtype=bool),
            'sources': sources,
            'paths': paths,
        }
        logger.info(
            f"Split '{out_name}' ({split_key}): n={len(X)}, "
            f"field={int(sum(is_field))}, missing={missing}"
        )
    return datasets


# =============================================================================
# TRAIN ONE MODEL
# =============================================================================
def make_sample_weights(is_field: np.ndarray, field_weight: float) -> torch.Tensor:
    w = np.ones(len(is_field), dtype=np.float32)
    w[is_field] = field_weight
    return torch.from_numpy(w)


def compute_min_class_metric(y_true, y_pred, class_names, focus_classes):
    """min(brassica_black_rot_F1, okra_cercospora_F1, macro_F1)."""
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))),
                         zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', labels=list(range(len(class_names))),
                     zero_division=0)
    focus_f1 = [per_class[class_names.index(c)] for c in focus_classes]
    return min(min(focus_f1), macro), per_class, macro


def train_one_model(
    X_train, y_train, w_train,
    X_val, y_val,
    class_names, focus_classes,
    device, log_prefix='',
):
    """Train a single MLP on (X_train, y_train) with sample weights w_train,
    early stopping on min-class-F1 on (X_val, y_val).
    Returns (best_state_dict, best_metric, history).
    """
    model = NonlinearHead(
        in_dim=X_train.shape[1],
        hidden_dims=HIDDEN_DIMS,
        num_classes=len(class_names),
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Torch tensors
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    wt = w_train.to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best = {'metric': -1.0, 'state': None, 'epoch': -1}
    history = {'train_f1': [], 'val_f1': [], 'val_min_class': [],
               'val_black_rot': [], 'val_cercospora': []}
    no_improve = 0

    bs = BATCH_SIZE
    n = len(X_train)
    idx_all = np.arange(n)
    g = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(wt.cpu().numpy(), num_samples=n,
                                    replacement=True, generator=g)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        # sample weighted indices
        idxs = list(sampler)
        for i in range(0, n, bs):
            batch_idx = idxs[i:i + bs]
            xb = Xt[batch_idx]; yb = yt[batch_idx]
            logits = model(xb)
            loss = criterion(logits, yb).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_idx)
        scheduler.step()
        epoch_loss /= n

        # Eval
        model.eval()
        with torch.no_grad():
            tr_logits = model(Xt)
            tr_pred = tr_logits.argmax(dim=1).cpu().numpy()
            vl_logits = model(Xv)
            vl_pred = vl_logits.argmax(dim=1).cpu().numpy()

        tr_macro = f1_score(y_train, tr_pred, average='macro',
                            labels=list(range(len(class_names))), zero_division=0)
        vl_min_class, vl_per_class, vl_macro = compute_min_class_metric(
            y_val, vl_pred, class_names, focus_classes)

        br_idx = class_names.index('brassica_black_rot')
        ce_idx = class_names.index('okra_cercospora')
        history['train_f1'].append(tr_macro)
        history['val_f1'].append(vl_macro)
        history['val_min_class'].append(vl_min_class)
        history['val_black_rot'].append(float(vl_per_class[br_idx]))
        history['val_cercospora'].append(float(vl_per_class[ce_idx]))

        # Class-collapse safety: any class at 0 after epoch 3 is a red flag
        if epoch >= 3:
            zero_classes = [class_names[i] for i in range(len(class_names))
                            if vl_per_class[i] < 0.3]
            if zero_classes:
                logger.warning(
                    f"{log_prefix}Epoch {epoch}: classes below F1=0.3: {zero_classes}"
                )

        if vl_min_class > best['metric']:
            best = {
                'metric': float(vl_min_class),
                'state': deepcopy(model.state_dict()),
                'epoch': int(epoch),
                'val_macro': float(vl_macro),
                'val_black_rot': float(vl_per_class[br_idx]),
                'val_cercospora': float(vl_per_class[ce_idx]),
            }
            no_improve = 0
        else:
            no_improve += 1

        logger.info(
            f"{log_prefix}Ep {epoch:02d}: loss={epoch_loss:.4f}  "
            f"train_f1={tr_macro:.4f}  val_f1={vl_macro:.4f}  "
            f"val_min={vl_min_class:.4f}  black_rot={vl_per_class[br_idx]:.4f}  "
            f"cerc={vl_per_class[ce_idx]:.4f}"
        )

        if no_improve >= PATIENCE:
            logger.info(f"{log_prefix}Early stop at epoch {epoch} (best={best['epoch']})")
            break

    return best, history


# =============================================================================
# 5-FOLD CV
# =============================================================================
def composite_stratify_key(y: np.ndarray, sources):
    """class_idx + '_' + source — for joint stratification."""
    return np.array([f"{yy}_{ss}" for yy, ss in zip(y, sources)])


def run_cv(X, y, is_field, sources, class_names, focus_classes, device):
    """5-fold stratified (class, source) CV on training data.
    Returns mean min-class metric across folds and full per-fold history.
    """
    strat = composite_stratify_key(y, sources)
    # Drop ultra-rare composite keys that would cause a fold with zero samples
    vals, counts = np.unique(strat, return_counts=True)
    rare_keys = set(vals[counts < N_FOLDS])
    if rare_keys:
        # Replace rare composite key with the class alone (fallback stratification)
        strat = np.array([
            (f"{y[i]}" if strat[i] in rare_keys else strat[i])
            for i in range(len(strat))
        ])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, strat)):
        logger.info(f"\n=== CV fold {fold_i + 1}/{N_FOLDS} ===")
        # Fit scaler on fold train, apply to fold val
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx]).astype(np.float32)
        Xva = scaler.transform(X[va_idx]).astype(np.float32)
        wtr = make_sample_weights(is_field[tr_idx], FIELD_PHOTO_WEIGHT)
        best, _ = train_one_model(
            Xtr, y[tr_idx], wtr,
            Xva, y[va_idx],
            class_names, focus_classes,
            device, log_prefix=f"[fold {fold_i + 1}] "
        )
        logger.info(
            f"Fold {fold_i + 1} best: min_class_metric={best['metric']:.4f} "
            f"(macro={best['val_macro']:.4f})"
        )
        fold_metrics.append(best)
    mean_min = float(np.mean([f['metric'] for f in fold_metrics]))
    mean_macro = float(np.mean([f['val_macro'] for f in fold_metrics]))
    logger.info(
        f"\n=== CV summary === mean_min_class={mean_min:.4f}  "
        f"mean_macro={mean_macro:.4f}"
    )
    return fold_metrics, mean_min, mean_macro


# =============================================================================
# EVALUATION (final model on val)
# =============================================================================
def evaluate_full(model: nn.Module, X, y, is_field, sources, paths,
                  class_names, device):
    """Full evaluation on a split. Returns dict with per-class, field, and source."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    n = len(y)
    macro = f1_score(y, preds, average='macro', labels=list(range(len(class_names))),
                     zero_division=0)
    weighted = f1_score(y, preds, average='weighted', labels=list(range(len(class_names))),
                        zero_division=0)
    acc = accuracy_score(y, preds)
    per_class = f1_score(y, preds, average=None, labels=list(range(len(class_names))),
                         zero_division=0)

    per_class_dict = {c: float(per_class[i]) for i, c in enumerate(class_names)}

    # Field-only analysis for failure classes
    field_analysis = {}
    for c in FAILURE_CLASSES:
        cidx = class_names.index(c)
        mask = (y == cidx) & is_field
        if mask.sum() > 0:
            n_field = int(mask.sum())
            acc_field = float((preds[mask] == cidx).mean())
            field_analysis[c] = {'n_field': n_field, 'field_accuracy': acc_field}
        else:
            field_analysis[c] = {'n_field': 0, 'field_accuracy': None}

    # Per-source for failure classes
    source_analysis = {}
    for c in FAILURE_CLASSES:
        cidx = class_names.index(c)
        entries = []
        srcs = np.array(sources)
        for s in sorted(set(srcs[y == cidx])):
            mask = (y == cidx) & (srcs == s)
            if mask.sum() > 0:
                entries.append({
                    'source': s,
                    'n': int(mask.sum()),
                    'accuracy': float((preds[mask] == cidx).mean()),
                })
        entries.sort(key=lambda e: e['accuracy'])
        source_analysis[c] = entries

    cm = confusion_matrix(y, preds, labels=list(range(len(class_names))))

    return {
        'n': n,
        'macro_f1': float(macro),
        'weighted_f1': float(weighted),
        'accuracy': float(acc),
        'per_class_f1': per_class_dict,
        'field_analysis': field_analysis,
        'source_analysis': source_analysis,
        'confusion_matrix': cm.tolist(),
        'probs': probs,
        'preds': preds,
    }


# =============================================================================
# PLOTTING
# =============================================================================
def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i][j]), ha='center', va='center',
                    color='white' if cm[i][j] > cm.max() / 2 else 'black', fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('DINOv2 Nonlinear Head — Validation Confusion Matrix')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_training_curves(history, out_path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(history['train_f1'], label='train_macro_F1')
    axs[0].plot(history['val_f1'], label='val_macro_F1')
    axs[0].plot(history['val_min_class'], label='val_min(br,ce,macro)', linewidth=2)
    axs[0].set_ylabel('F1'); axs[0].legend(); axs[0].grid(True, alpha=0.3)
    axs[0].set_title('Training Curves — DINOv2 Nonlinear Head')

    axs[1].plot(history['val_black_rot'], label='val brassica_black_rot F1', color='C3')
    axs[1].plot(history['val_cercospora'], label='val okra_cercospora F1', color='C4')
    axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Per-class F1')
    axs[1].legend(); axs[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("=" * 78)
    logger.info("DINOv2 NONLINEAR HEAD TRAINING — Signal 4 of APIN ensemble")
    logger.info("=" * 78)
    logger.info(f"Timestamp      : {TIMESTAMP}")
    logger.info(f"Device         : {DEVICE_PREF}")
    logger.info(f"Architecture   : 768 -> {' -> '.join(map(str, HIDDEN_DIMS))} -> {NUM_CLASSES}")
    logger.info(f"Dropout        : {DROPOUT}")
    logger.info(f"Optimizer      : AdamW(lr={LR}, wd={WEIGHT_DECAY})")
    logger.info(f"Epochs         : {EPOCHS}")
    logger.info(f"Batch size     : {BATCH_SIZE}")
    logger.info(f"Field weight   : {FIELD_PHOTO_WEIGHT}x")
    logger.info(f"CV folds       : {N_FOLDS}")
    logger.info(f"Seed           : {SEED}")
    logger.info(f"Feature aggreg : {FEATURE_AGGREGATION}  (expected dim {FEATURE_DIM_INT})")

    set_seeds(SEED)

    # ------------------------------------------------------------
    # 1. LOAD CACHE + SPLITS
    # ------------------------------------------------------------
    datasets = load_cache_and_splits()

    # ------------------------------------------------------------
    # 2. 5-FOLD CV ON TRAINING DATA
    # ------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("5-FOLD CV (stratified by (class, source))")
    logger.info("=" * 78)
    fold_metrics, mean_min_class, mean_macro = run_cv(
        datasets['train']['X'],
        datasets['train']['y'],
        datasets['train']['is_field'],
        datasets['train']['sources'],
        CLASS_NAMES,
        FAILURE_CLASSES,
        DEVICE_PREF,
    )

    # ------------------------------------------------------------
    # 3. RETRAIN ON FULL TRAINING SET, EVALUATE ON val AND final_val
    # ------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("FINAL TRAINING — full train split, hold-out val + final_val")
    logger.info("=" * 78)

    # Fit scaler on full train
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(datasets['train']['X']).astype(np.float32)
    Xv  = scaler.transform(datasets['val']['X']).astype(np.float32)
    Xfv = scaler.transform(datasets['final_val']['X']).astype(np.float32)

    wtr = make_sample_weights(datasets['train']['is_field'], FIELD_PHOTO_WEIGHT)
    best, history = train_one_model(
        Xtr, datasets['train']['y'], wtr,
        Xv, datasets['val']['y'],
        CLASS_NAMES, FAILURE_CLASSES,
        DEVICE_PREF, log_prefix='[final] '
    )

    # Build the final model and load best weights
    model = NonlinearHead(
        in_dim=FEATURE_DIM_INT, hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES, dropout=DROPOUT,
    ).to(DEVICE_PREF)
    model.load_state_dict(best['state'])

    # ------------------------------------------------------------
    # 4. EVALUATE ON val AND final_val
    # ------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("EVALUATION")
    logger.info("=" * 78)
    val_results = evaluate_full(
        model, Xv,
        datasets['val']['y'], datasets['val']['is_field'],
        datasets['val']['sources'], datasets['val']['paths'],
        CLASS_NAMES, DEVICE_PREF,
    )
    final_val_results = evaluate_full(
        model, Xfv,
        datasets['final_val']['y'], datasets['final_val']['is_field'],
        datasets['final_val']['sources'], datasets['final_val']['paths'],
        CLASS_NAMES, DEVICE_PREF,
    )

    # Probe baseline numbers (from the logistic regression experiment)
    probe_per_class_baseline = {
        'okra_yvmv': 0.9628, 'okra_powdery_mildew': 0.8989,
        'okra_cercospora': 0.9216, 'okra_enation': 0.8395,
        'okra_healthy': 0.9664, 'brassica_black_rot': 0.9342,
        'brassica_downy_mildew': 0.8679, 'brassica_alternaria': 0.9038,
        'brassica_healthy': 0.9390,
    }
    probe_val_macro = 0.9149
    probe_black_rot_field = 0.917
    probe_cercospora_field = 1.000

    # ------------------------------------------------------------
    # 5. REPORTS
    # ------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("REPORT A — Overall val performance")
    logger.info("=" * 78)
    logger.info(f"Val macro F1    : {val_results['macro_f1']:.4f}")
    logger.info(f"Val weighted F1 : {val_results['weighted_f1']:.4f}")
    logger.info(f"Val accuracy    : {val_results['accuracy']:.4f}")
    logger.info(f"LogReg baseline : {probe_val_macro:.4f}")
    delta = val_results['macro_f1'] - probe_val_macro
    logger.info(f"Delta (nonlinear - logreg): {delta:+.4f}")
    if delta >= 0.02:
        interp = "nonlinear head IS better than logistic regression by a meaningful margin"
    elif delta >= 0.005:
        interp = "nonlinear head is marginally better than logistic regression"
    elif delta >= -0.005:
        interp = "nonlinear head is essentially tied with logistic regression (no meaningful gain)"
    else:
        interp = "nonlinear head IS WORSE than logistic regression (unexpected — inspect for overfitting)"
    logger.info(f"Interpretation  : {interp}")

    logger.info("\nREPORT B — Per-class F1 comparison")
    logger.info(f"{'class_name':<28} {'nonlin':>8} {'logreg':>8} {'model2':>8} {'delta':>8}")
    for cn in CLASS_NAMES:
        nh = val_results['per_class_f1'][cn]
        pr = probe_per_class_baseline.get(cn, float('nan'))
        m2 = MODEL2_PER_CLASS_F1.get(cn, float('nan'))
        dlt = nh - pr
        logger.info(f"{cn:<28} {nh:>8.4f} {pr:>8.4f} {m2:>8.4f} {dlt:>+8.4f}")

    logger.info("\nREPORT C — Field-photo-only evaluation (PRIMARY METRIC)")
    for c in FAILURE_CLASSES:
        fa = val_results['field_analysis'][c]
        logger.info(f"{c}:")
        logger.info(f"  n_field images: {fa['n_field']}")
        logger.info(f"  Nonlinear head accuracy: {fa['field_accuracy']}")
        if c == 'brassica_black_rot':
            logger.info(f"  LogReg accuracy: {probe_black_rot_field}")
            logger.info(f"  Model 2 real-world: 2-20% (known anecdotal; not measured)")
        if c == 'okra_cercospora':
            logger.info(f"  LogReg accuracy: {probe_cercospora_field}")

    logger.info("\nREPORT D — Per-source breakdown for failure classes")
    for c in FAILURE_CLASSES:
        logger.info(f"\n  {c}:")
        for entry in val_results['source_analysis'][c]:
            logger.info(
                f"    {entry['source']:<30} n={entry['n']:>4}  acc={entry['accuracy']:.4f}"
            )

    logger.info("\nREPORT F — Train/val gap")
    max_train_f1 = max(history['train_f1']) if history['train_f1'] else 0.0
    val_f1_at_best = history['val_f1'][best['epoch']] if history['val_f1'] else 0.0
    logger.info(f"  max train F1: {max_train_f1:.4f}")
    logger.info(f"  val F1 at best epoch ({best['epoch']}): {val_f1_at_best:.4f}")
    logger.info(f"  gap: {max_train_f1 - val_f1_at_best:+.4f}")

    # ------------------------------------------------------------
    # 6. SAVE ARTIFACTS
    # ------------------------------------------------------------
    model_path  = RESULTS_DIR / f'dinov2_nonlinear_head_{TIMESTAMP}.pt'
    scaler_path = RESULTS_DIR / f'dinov2_nonlinear_head_scaler_{TIMESTAMP}.pkl'
    config_path = RESULTS_DIR / f'dinov2_nonlinear_head_config_{TIMESTAMP}.json'
    results_path = RESULTS_DIR / f'dinov2_nonlinear_head_results_{TIMESTAMP}.json'
    cm_path     = RESULTS_DIR / f'dinov2_nonlinear_head_cm_{TIMESTAMP}.png'
    curves_path = RESULTS_DIR / f'dinov2_nonlinear_head_curves_{TIMESTAMP}.png'

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    config_dump = {
        'timestamp': TIMESTAMP,
        'architecture': 'DINOv2-Small-Registers + MLP head',
        'backbone_timm_name': 'vit_small_patch14_reg4_dinov2.lvd142m',
        'backbone_frozen': True,
        'feature_aggregation': FEATURE_AGGREGATION,
        'feature_dim': FEATURE_DIM_INT,
        'hidden_dims': HIDDEN_DIMS,
        'dropout': DROPOUT,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'optimizer': 'AdamW',
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'epochs_planned': EPOCHS,
        'epochs_actual_final_training': best['epoch'] + 1,
        'batch_size': BATCH_SIZE,
        'field_photo_weight': FIELD_PHOTO_WEIGHT,
        'scaler': 'StandardScaler',
        'preprocessing_used': 'LAB-CLAHE (L-channel only) applied in original cache extraction',
        'random_seed': SEED,
        'cv_n_folds': N_FOLDS,
        'cv_mean_min_class_metric': mean_min_class,
        'cv_mean_macro_f1': mean_macro,
        'model_path_rel': str(model_path.relative_to(PROJECT_ROOT)),
        'scaler_path_rel': str(scaler_path.relative_to(PROJECT_ROOT)),
    }
    with open(config_path, 'w') as f:
        json.dump(config_dump, f, indent=2)

    results_dump = {
        'timestamp': TIMESTAMP,
        'cv': {
            'n_folds': N_FOLDS,
            'mean_min_class_metric': mean_min_class,
            'mean_macro_f1': mean_macro,
            'per_fold': [
                {'fold': i, 'metric': f['metric'], 'macro': f['val_macro'],
                 'black_rot': f['val_black_rot'], 'cercospora': f['val_cercospora'],
                 'epoch': f['epoch']}
                for i, f in enumerate(fold_metrics)
            ],
        },
        'best_epoch': best['epoch'],
        'best_val_min_class_metric': best['metric'],
        'val': {
            'macro_f1': val_results['macro_f1'],
            'weighted_f1': val_results['weighted_f1'],
            'accuracy': val_results['accuracy'],
            'per_class_f1': val_results['per_class_f1'],
            'field_analysis': val_results['field_analysis'],
            'source_analysis': val_results['source_analysis'],
            'confusion_matrix': val_results['confusion_matrix'],
        },
        'final_val': {
            'macro_f1': final_val_results['macro_f1'],
            'weighted_f1': final_val_results['weighted_f1'],
            'accuracy': final_val_results['accuracy'],
            'per_class_f1': final_val_results['per_class_f1'],
            'field_analysis': final_val_results['field_analysis'],
            'source_analysis': final_val_results['source_analysis'],
            'confusion_matrix': final_val_results['confusion_matrix'],
        },
        'baselines': {
            'probe_val_macro_f1': probe_val_macro,
            'probe_black_rot_field_acc': probe_black_rot_field,
            'probe_cercospora_field_acc': probe_cercospora_field,
            'model2_val_macro_f1': MODEL2_VAL_F1.get('macro'),
            'model2_per_class_f1': MODEL2_PER_CLASS_F1,
        },
        'history': history,
    }
    with open(results_path, 'w') as f:
        json.dump(results_dump, f, indent=2)

    import numpy as _np
    plot_confusion_matrix(_np.array(val_results['confusion_matrix']),
                          CLASS_NAMES, cm_path)
    plot_training_curves(history, curves_path)

    logger.info("\n" + "=" * 78)
    logger.info("SAVED ARTIFACTS")
    logger.info("=" * 78)
    logger.info(f"  Model   : {model_path}")
    logger.info(f"  Scaler  : {scaler_path}")
    logger.info(f"  Config  : {config_path}")
    logger.info(f"  Results : {results_path}")
    logger.info(f"  CM plot : {cm_path}")
    logger.info(f"  Curves  : {curves_path}")

    # ------------------------------------------------------------
    # 7. FINAL ASSESSMENT STRING
    # ------------------------------------------------------------
    br_field = val_results['field_analysis']['brassica_black_rot']['field_accuracy']
    ce_field = val_results['field_analysis']['okra_cercospora']['field_accuracy']
    logger.info("\n" + "=" * 78)
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 78)
    ready = (val_results['macro_f1'] >= 0.90 and
             (br_field is None or br_field >= 0.85) and
             (ce_field is None or ce_field >= 0.90))
    logger.info(
        f"Signal 4 (DINOv2 nonlinear head) training complete. Results:"
    )
    logger.info(f"  Val macro F1: {val_results['macro_f1']:.4f}")
    logger.info(f"  Black_rot field accuracy: {br_field}")
    logger.info(f"  Cercospora field accuracy: {ce_field}")
    status = "READY" if ready else "CONDITIONAL"
    logger.info(f"  Assessment: Signal 4 {status} for production.")

    return {
        'val_macro_f1': val_results['macro_f1'],
        'black_rot_field_accuracy': br_field,
        'cercospora_field_accuracy': ce_field,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'ready': ready,
    }


if __name__ == '__main__':
    main()
