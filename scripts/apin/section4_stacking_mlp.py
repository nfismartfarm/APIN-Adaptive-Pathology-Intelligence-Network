"""
Section 4 -- APIN Stacking MLP with Two-Phase MoE gate.

Loads all 4 signal prediction caches. Builds 36-dim input (Signal 1 softmax +
Signal 2 raw sigmoid + Signal 3 PSV scores + Signal 4 softmax). Applies
reliability-matrix modulation in Layer 3. Adds Dirichlet adversarial
augmentation for the two failure classes (black_rot, cercospora). Trains
two-phase schedule: Phase A (15 epochs, gate frozen at uniform 0.25) then
Phase B (35 epochs, gate unfrozen with entropy + load-balancing + 0.05 floor).

Loss: CE + constrained per-class penalty (prompt Section 4E).
Sampling: field-photo 5x, source-diverse 2x (capped 8x).
Adversarial augmentation: uses the measured failure distributions from
scripts/apin/caches/signal1_measured_failure_distributions.json.

Output:
  scripts/apin/caches/apin_stacking_mlp_{ts}.pt
  scripts/apin/caches/apin_stacking_mlp_config_{ts}.json
  scripts/apin/caches/apin_stacking_mlp_history_{ts}.json

If the PSV cache is missing, falls back to 3-signal mode (27-dim input) with
a --fallback-3signal flag. Otherwise runs 4-signal.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section4_stacking_mlp_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section4")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

# ========================================================================
# CONFIG
# ========================================================================
SEED = 42
BATCH_SIZE = 256
PHASE_A_EPOCHS = 15
PHASE_B_EPOCHS = 35
GATE_WARMUP_EPOCHS = 5  # within Phase B
MLP_LR = 1e-3
GATE_LR_WARMUP = 1e-5
GATE_LR_FULL = 1e-4
MLP_LR_PHASE_B = 5e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.35

# Per-class F1 floors (constrained optimization penalty)
FLOORS = {
    "okra_yvmv": 0.92,
    "okra_powdery_mildew": 0.87,
    "okra_cercospora": 0.80,
    "okra_enation": 0.83,
    "okra_healthy": 0.93,
    "brassica_black_rot": 0.75,
    "brassica_downy_mildew": 0.85,
    "brassica_alternaria": 0.88,
    "brassica_healthy": 0.93,
}
PENALTY_WEIGHT_INIT = 0.5
PENALTY_WEIGHT_HIGH = 1.0

# Sampling weights
FIELD_WEIGHT = 5.0
SOURCE_DIVERSE_WEIGHT = 2.0  # for non-dominant sources
MAX_SAMPLE_WEIGHT = 8.0

# Adversarial augmentation
ADV_ALPHA = 5.0  # Dirichlet concentration
ADV_COPIES_PER_IMAGE = 5
ADV_MAX_FRACTION = 0.30  # cap adversarial at 30% of training data

# MoE gate losses
GATE_MIN_FLOOR = 0.05
GATE_ENTROPY_WEIGHT = 0.05
GATE_LOAD_BALANCE_WEIGHT = 0.01


# ========================================================================
# ARCHITECTURE
# ========================================================================
class MoEGate(nn.Module):
    """Input dim 36 (or 27 for 3-signal) -> 32 -> N_SIGNALS.
    Outputs softmax gating weights, clamped to min 0.05 and renormalized."""
    def __init__(self, input_dim: int, n_signals: int):
        super().__init__()
        self.n_signals = n_signals
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_signals),
        )

    def forward(self, x):
        logits = self.net(x)
        w = F.softmax(logits, dim=1)
        # Floor + renormalize
        w = torch.clamp(w, min=GATE_MIN_FLOOR)
        w = w / w.sum(dim=1, keepdim=True)
        return w  # (batch, n_signals)


class StackingMLP(nn.Module):
    """36 (or 27) -> 128 -> 64 -> 32 -> 9 with BatchNorm + GELU + Dropout.

    Returns raw logits (caller softmaxes). Also exposes the 32-dim
    penultimate features via `forward_with_features` so the contrastive
    loss (Gap 8 audit fix) can pull together real-vs-adversarial pairs of
    the same class in embedding space.
    """
    def __init__(self, input_dim: int, num_classes: int = 9, dropout: float = 0.35):
        super().__init__()
        # Split sequential at the 32-d feature so we can return both
        self.body = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),                                nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        feat = self.body(x)
        return self.classifier(feat)

    def forward_with_features(self, x):
        """Returns (logits, 32-d penultimate features). Used by the
        contrastive loss to compare real vs adversarial embeddings."""
        feat = self.body(x)
        return self.classifier(feat), feat


class APIN_Ensemble(nn.Module):
    """Combines gate + MLP. The gate modulates per-signal weights applied to
    9-dim slices of the input BEFORE the MLP sees them. Gate can be frozen."""
    def __init__(self, n_signals: int, num_classes: int = 9, dropout: float = 0.35):
        super().__init__()
        self.n_signals = n_signals
        self.num_classes = num_classes
        input_dim = n_signals * num_classes
        self.gate = MoEGate(input_dim, n_signals)
        self.mlp = StackingMLP(input_dim, num_classes, dropout)
        self._gate_frozen = True  # Phase A starts with gate frozen

    def freeze_gate(self):
        for p in self.gate.parameters():
            p.requires_grad = False
        self._gate_frozen = True

    def unfreeze_gate(self):
        for p in self.gate.parameters():
            p.requires_grad = True
        self._gate_frozen = False

    def forward(self, x, return_gate_weights: bool = False):
        """x: (batch, n_signals * num_classes)
        Slice into per-signal blocks, apply per-signal gate weight,
        recombine, then feed through the stacking MLP.
        """
        B = x.shape[0]
        blocks = x.view(B, self.n_signals, self.num_classes)  # (B, S, C)

        if self._gate_frozen:
            # Uniform 1/S weights
            w = torch.full(
                (B, self.n_signals), 1.0 / self.n_signals,
                device=x.device, dtype=x.dtype,
            )
        else:
            w = self.gate(x)  # (B, S)

        # Apply per-signal weight to each 9-dim block
        modulated = blocks * w.unsqueeze(-1)  # (B, S, C)
        modulated_flat = modulated.view(B, -1)  # (B, S*C)

        logits = self.mlp(modulated_flat)
        if return_gate_weights:
            return logits, w
        return logits

    def forward_with_features(self, x, return_gate_weights: bool = False):
        """Same as forward() but also returns 32-dim penultimate features.
        Used by the contrastive loss in section 4 training to pull together
        embeddings of real-vs-adversarial samples of the same class."""
        B = x.shape[0]
        blocks = x.view(B, self.n_signals, self.num_classes)
        if self._gate_frozen:
            w = torch.full(
                (B, self.n_signals), 1.0 / self.n_signals,
                device=x.device, dtype=x.dtype,
            )
        else:
            w = self.gate(x)
        modulated_flat = (blocks * w.unsqueeze(-1)).view(B, -1)
        logits, feat = self.mlp.forward_with_features(modulated_flat)
        if return_gate_weights:
            return logits, feat, w
        return logits, feat


# ========================================================================
# DATA
# ========================================================================
def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_signal_caches(use_psv: bool):
    """Load all available signal caches. Returns a dict.
    Signal ordering in the 36-dim vector: [S1_M2, S2_EN, S3_PSV, S4_DINOv2]
    or [S1_M2, S2_EN, S4_DINOv2] if use_psv=False.
    """
    caches = {}
    names = {1: "signal1_predictions_cache.pkl",
              2: "signal2_predictions_cache.pkl",
              4: "signal4_predictions_cache.pkl"}
    if use_psv:
        names[3] = "signal3_psv_predictions_cache.pkl"

    for sig_id, fn in sorted(names.items()):
        path = CACHE_DIR / fn
        if not path.exists():
            raise FileNotFoundError(f"Missing signal cache: {path}")
        with open(path, "rb") as f:
            caches[sig_id] = pickle.load(f)
        logger.info(f"  Signal {sig_id}: {len(caches[sig_id])} entries from {fn}")
    return caches


def build_inputs(caches, use_psv: bool):
    """Align all caches by row index and build (X, y, splits, is_field, sources, is_recomp).
    Input X is (N, n_signals * 9).
    """
    # Determine row-index intersection
    all_indices = set(caches[1].keys())
    for sig_id in caches:
        all_indices &= set(caches[sig_id].keys())
    indices = sorted(all_indices)
    logger.info(f"  Row-index intersection across all caches: {len(indices)}")

    sig_order = [1, 2, 3, 4] if use_psv else [1, 2, 4]

    X = np.zeros((len(indices), len(sig_order) * 9), dtype=np.float32)
    y = np.zeros(len(indices), dtype=np.int64)
    splits = []
    is_field = np.zeros(len(indices), dtype=bool)
    sources = []
    is_recomp = np.zeros(len(indices), dtype=bool)
    success_all = np.ones(len(indices), dtype=bool)

    for row_pos, row_idx in enumerate(indices):
        entry1 = caches[1][row_idx]
        y[row_pos] = entry1["true_class_idx"]
        splits.append(entry1["split"])
        is_field[row_pos] = entry1["is_field_photo"]
        sources.append(entry1["source_dataset"])
        is_recomp[row_pos] = entry1["is_recomposed"]
        for pos, sig_id in enumerate(sig_order):
            entry = caches[sig_id][row_idx]
            # Handle success flag heterogeneity across caches
            if not entry.get("extraction_success",
                              entry.get("inference_success", True)):
                success_all[row_pos] = False
            vec = entry["predictions"]
            X[row_pos, pos * 9: (pos + 1) * 9] = vec

    return X, y, np.array(splits), is_field, np.array(sources), is_recomp, success_all, indices


def load_reliability_matrix(use_psv: bool):
    """Load R matrix. For 3-signal, use reliability_matrix_3signal.json.
    For 4-signal, we compute it from the caches directly since there's no
    pre-built 4x9 version yet.
    """
    if not use_psv:
        path = CACHE_DIR / "reliability_matrix_3signal.json"
        with open(path) as f:
            m = json.load(f)
        # 3x9 matrix, rows [S1, S2, S4]
        R = np.array(m["matrix_3x9"], dtype=np.float32)
        return R
    # For 4-signal, compute inline from caches
    return None  # signals the caller to compute below


def compute_reliability_matrix_inline(caches, use_psv: bool, y, splits, is_field):
    """When use_psv=True, build a 4x9 reliability matrix from caches on the
    val_and_soup + field subset."""
    n_signals = 4 if use_psv else 3
    sig_order = [1, 2, 3, 4] if use_psv else [1, 2, 4]
    mask = (splits == "val_and_soup") & is_field
    R = np.zeros((n_signals, 9), dtype=np.float32)
    for sig_pos, sig_id in enumerate(sig_order):
        cache = caches[sig_id]
        for c_idx in range(9):
            cls_mask = mask & (y == c_idx)
            if cls_mask.sum() == 0: continue
            # Gather predictions for these rows
            correct = 0; total = 0
            # We need to iterate via the index list; but we only have
            # positional masks here. Re-derive.
            # Easier: iterate all rows in np, using y + splits already aligned.
            # For this we need the full predictions matrix — re-compute from caches.
            pass
    # Simpler direct implementation:
    # Iterate each signal's cache directly, filtered by our indices.
    return None  # placeholder, use external reliability compute


def apply_reliability_modulation(X, R, n_signals: int):
    """X: (N, S*9), R: (S, 9). Multiply each 9-block by R[s] elementwise."""
    X_mod = X.copy()
    for s in range(n_signals):
        X_mod[:, s * 9: (s + 1) * 9] *= R[s]
    return X_mod


def make_sample_weights(is_field, sources, class_order):
    """Field 5x + source-diverse 2x (for non-dominant sources), capped 8x."""
    dominant_sources = {"original_pool"}  # per our data analysis
    w = np.ones(len(is_field), dtype=np.float32)
    w[is_field] *= FIELD_WEIGHT
    src_arr = np.array(sources)
    non_dominant = ~np.isin(src_arr, list(dominant_sources))
    w[non_dominant] *= SOURCE_DIVERSE_WEIGHT
    w = np.clip(w, 0.0, MAX_SAMPLE_WEIGHT)
    return w


# ========================================================================
# ADVERSARIAL AUGMENTATION
# ========================================================================
def load_measured_failure_distributions():
    path = CACHE_DIR / "signal1_measured_failure_distributions.json"
    with open(path) as f:
        data = json.load(f)
    return data["distributions"]


def generate_adversarial_augmentation(X_train, y_train, is_recomp_train,
                                       class_order, n_signals: int,
                                       reliability_matrix: np.ndarray = None):
    """For each brassica_black_rot and okra_cercospora training image that is
    NOT recomposed, add ADV_COPIES_PER_IMAGE adversarial variants:
      - Signal 1 (Model 2) vector replaced with Dirichlet-sampled failure dist
      - Signals 2, 3, 4 kept as real
    The injected p_failure comes from the measured empirical distributions
    (Decision 21). Validate correct-class prob < 0.30 per sample.

    Returns: (X_aug, y_aug, w_aug, stats)
    """
    from scripts.apin.constants import MODEL2_CLASS_ORDER
    class_to_idx = {c: i for i, c in enumerate(MODEL2_CLASS_ORDER)}

    distributions = load_measured_failure_distributions()
    failure_patterns = {}
    for target_cls in ("brassica_black_rot", "okra_cercospora"):
        entry = distributions.get(target_cls, {})
        # Prefer field_only if non-zero; else fall back to all
        field_dist = entry.get("mean_failure_distribution_field_only", {})
        all_dist = entry.get("mean_failure_distribution_all", {})
        if sum(field_dist.values()) > 0.01:
            dist = field_dist
        else:
            dist = all_dist
        p = np.array([dist.get(c, 0.0) for c in MODEL2_CLASS_ORDER], dtype=np.float32)
        # Normalize (defensive)
        if p.sum() > 0:
            p = p / p.sum()
        else:
            # No measurable failure distribution — skip adversarial for this class
            logger.info(f"  {target_cls}: no measurable failure distribution — skipping adversarial")
            continue
        failure_patterns[target_cls] = p
        logger.info(f"  {target_cls} failure distribution (sum={p.sum():.4f}):")
        for c, prob in zip(MODEL2_CLASS_ORDER, p):
            if prob > 0.02:
                logger.info(f"    {c:<28} {prob:.4f}")

    rng = np.random.default_rng(SEED)
    adv_X, adv_y, adv_w, adv_pair = [], [], [], []
    stats = {"generated": 0, "resampled_invalid": 0}

    for target_cls, p_failure in failure_patterns.items():
        cls_idx = class_to_idx[target_cls]
        mask = (y_train == cls_idx) & (~is_recomp_train)
        idxs = np.where(mask)[0]
        logger.info(f"  {target_cls}: {len(idxs)} training images, "
                    f"generating {len(idxs) * ADV_COPIES_PER_IMAGE} adversarial copies")

        for i in idxs:
            for _ in range(ADV_COPIES_PER_IMAGE):
                # Sample from Dirichlet centered on p_failure
                alpha = ADV_ALPHA * p_failure + 1e-6  # avoid zero alpha
                attempts = 0
                while True:
                    p_sampled = rng.dirichlet(alpha)
                    if p_sampled[cls_idx] < 0.30:
                        break
                    attempts += 1
                    if attempts > 10:
                        # Resample hit limit — use the last sample anyway
                        stats["resampled_invalid"] += 1
                        break

                # Build adversarial input vector: copy original, replace
                # Signal 1 block. X_train is reliability-modulated, so the
                # injected Dirichlet sample must also be modulated by the
                # Signal 1 row of the reliability matrix to keep the per-
                # signal scale comparable with the rest of the row. Without
                # this scaling the MLP saw modulated S1 in real samples and
                # raw S1 in adversarial samples — a distributional mismatch
                # that overstates how much it should discount Signal 1.
                if reliability_matrix is not None:
                    p_inject = (p_sampled * reliability_matrix[0]).astype(np.float32)
                else:
                    p_inject = p_sampled.astype(np.float32)
                orig_row = X_train[i].copy()
                orig_row[0:9] = p_inject  # Signal 1 is the first block
                adv_X.append(orig_row.astype(np.float32))
                adv_y.append(cls_idx)
                adv_w.append(1.0)  # weight same as real images
                # Track which real-row index this adversarial sample was
                # spawned from so the contrastive loss can pair them
                # (Gap 8 audit fix).
                adv_pair.append(int(i))
                stats["generated"] += 1

    if not adv_X:
        return (np.zeros((0, n_signals * 9), dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
                stats)
    return (np.vstack(adv_X), np.array(adv_y), np.array(adv_w),
            np.array(adv_pair, dtype=np.int64), stats)


# ========================================================================
# LOSS
# ========================================================================
# Contrastive loss weight (Gap 8 audit fix). At 0.10 the contrastive term
# is one-tenth the cross-entropy magnitude — small enough to not dominate
# but large enough to actually shape the embedding space.
CONTRASTIVE_WEIGHT = 0.10
CONTRASTIVE_TEMPERATURE = 0.20


def contrastive_loss(features: torch.Tensor, labels: torch.Tensor,
                      partner_idx: torch.Tensor) -> torch.Tensor:
    """Supervised contrastive (SupCon-like) loss restricted to real-vs-
    adversarial pairs of the same class. The architecture spec calls for
    "pulling together embeddings of real black_rot images where all signals
    agree and adversarial black_rot images where only DINOv2+PSV agree" —
    this implementation generalises that to all classes that have
    adversarial copies.

    For each adversarial sample i with partner p (its source real sample):
        - positive: features[p] (same class, real)
        - negatives: features of OTHER samples in batch with DIFFERENT class
    Loss = -log(exp(sim(i, p)/T) / sum_j exp(sim(i, j)/T))

    If no in-batch real partners are present (e.g., the random batch happens
    to drop all real samples for the class), the loss is zero for that
    sample. Returns mean over all adversarial samples that found a partner.
    """
    if features.shape[0] < 2:
        return torch.zeros((), device=features.device)
    # L2-normalise so dot product = cosine similarity
    feats = F.normalize(features, dim=1)
    sim = (feats @ feats.t()) / CONTRASTIVE_TEMPERATURE  # (B, B)

    # Mask out self-similarity
    B = feats.shape[0]
    eye = torch.eye(B, device=feats.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    # For each adversarial sample, find its partner index within this batch
    # (partner_idx values are global indices into X_train_full; we need to
    # map them to within-batch positions). Build a lookup: for global idx g
    # in this batch, position is the row whose underlying global index is g.
    # The caller passes batch_global_idx separately; here we use partner_idx
    # already pre-mapped to within-batch indices (-1 = no partner in batch).
    losses = []
    for i in range(B):
        p = int(partner_idx[i].item())
        if p < 0 or p >= B:
            continue  # this sample is real or its partner not in batch
        # Negatives: indices with DIFFERENT class than this sample
        same_class = (labels == labels[i])
        valid_neg = (~same_class) & (~eye[i])
        if not valid_neg.any():
            continue
        # Numerator: sim with positive partner
        pos_score = sim[i, p]
        # Denominator: pos + all valid negatives
        neg_scores = sim[i, valid_neg]
        denom = torch.logsumexp(
            torch.cat([pos_score.unsqueeze(0), neg_scores]), dim=0
        )
        losses.append(denom - pos_score)
    if not losses:
        return torch.zeros((), device=features.device)
    return torch.stack(losses).mean()


def constrained_loss(logits, targets, per_class_f1, sample_weights,
                      penalty_weight: float, class_order):
    """CE + penalty for classes below their floor.
    per_class_f1 is the per-class F1 from the PREVIOUS evaluation (initially
    zeros in first step; computed live after each epoch)."""
    # Weighted CE
    ce = F.cross_entropy(logits, targets, reduction="none")
    ce_weighted = (ce * sample_weights).mean()

    # Per-class floor penalty — only if per_class_f1 has been computed
    penalty = 0.0
    if per_class_f1 is not None:
        for c_name, floor in FLOORS.items():
            c_idx = class_order.index(c_name)
            f1 = per_class_f1[c_idx]
            if f1 < floor:
                gap = floor - f1
                penalty = penalty + penalty_weight * (gap ** 2)
    return ce_weighted + penalty, ce_weighted.detach(), float(penalty) if isinstance(penalty, float) else float(penalty.item() if hasattr(penalty, 'item') else 0.0)


def gate_regularization_loss(gate_weights):
    """MoE gate aux losses: entropy (penalize too-spiky), load-balance."""
    # Entropy per sample; we want it moderate, not too low (spiky) nor too high (uniform)
    # Use entropy loss to prevent collapse — penalize LOW entropy
    eps = 1e-8
    entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=1).mean()
    # For 4 signals: max entropy = log(4) = 1.386
    # We want entropy to stay in [0.5, 1.2] range; penalize below 0.5
    entropy_loss = F.relu(0.5 - entropy)

    # Load balance: average per-signal weight across batch should be ~1/S
    batch_avg = gate_weights.mean(dim=0)  # (S,)
    S = gate_weights.shape[1]
    target = 1.0 / S
    load_loss = ((batch_avg - target) ** 2).sum()

    return GATE_ENTROPY_WEIGHT * entropy_loss + GATE_LOAD_BALANCE_WEIGHT * load_loss


# ========================================================================
# TRAIN
# ========================================================================
def train_one_model(X_train_t, y_train_t, w_train_t,
                     X_val_t, y_val_t,
                     model, device, class_order,
                     use_psv: bool, phase_a_epochs: int,
                     phase_b_epochs: int, logger,
                     real_partner_idx=None):
    """Two-phase training. Returns best state_dict + history.

    real_partner_idx: numpy array of shape (n_train,) — for each row, the
        global index of its real source row if adversarial, or -1 if real.
        When None, contrastive loss is disabled.
    """
    n_signals = 4 if use_psv else 3
    # Default: no contrastive partners (treat all rows as independent reals)
    if real_partner_idx is None:
        real_partner_idx = np.full(X_train_t.shape[0], -1, dtype=np.int64)

    # Optimizer with parameter groups: gate + MLP
    gate_params = list(model.gate.parameters())
    mlp_params = list(model.mlp.parameters())

    optimizer = torch.optim.AdamW(
        [{"params": mlp_params, "lr": MLP_LR},
         {"params": gate_params, "lr": 0.0}],  # gate starts at 0 (frozen)
        weight_decay=WEIGHT_DECAY,
    )

    n = X_train_t.shape[0]
    batch_size = BATCH_SIZE
    # Note: removed unused "gate_entropy_mean" key — it was declared but
    # never appended to. Downstream code that misaligned by index would
    # have produced subtly wrong plots.
    history = {"epoch": [], "phase": [], "train_loss": [], "train_macro_f1": [],
                "val_macro_f1": [], "val_black_rot": [], "val_cercospora": [],
                "val_min_metric": [], "val_per_class": [],
                "gate_weights_mean": []}

    best = {"val_min": -1, "state": None, "epoch": -1}
    per_class_f1 = None  # updated after each val eval
    total_epochs = phase_a_epochs + phase_b_epochs
    penalty_w = PENALTY_WEIGHT_INIT

    for epoch in range(total_epochs):
        # Switch phases
        if epoch < phase_a_epochs:
            phase = "A"
            model.freeze_gate()
            for pg in optimizer.param_groups:
                pg["lr"] = MLP_LR if pg["params"] is mlp_params else 0.0
        else:
            phase = "B"
            model.unfreeze_gate()
            # Gate LR warmup within Phase B
            b_epoch = epoch - phase_a_epochs
            if b_epoch < GATE_WARMUP_EPOCHS:
                g_lr = GATE_LR_WARMUP
            else:
                g_lr = GATE_LR_FULL
            for pg in optimizer.param_groups:
                if pg["params"] is gate_params:
                    pg["lr"] = g_lr
                else:
                    pg["lr"] = MLP_LR_PHASE_B

            # Bump penalty weight if floors still violated after 10 epochs of Phase A
            if epoch >= phase_a_epochs + 5 and per_class_f1 is not None:
                below_floor = any(
                    per_class_f1[class_order.index(c)] < floor - 0.01
                    for c, floor in FLOORS.items()
                )
                if below_floor:
                    penalty_w = PENALTY_WEIGHT_HIGH

        # Weighted sampling
        sampler_weights = w_train_t.cpu().numpy()
        sampler_probs = sampler_weights / sampler_weights.sum()
        rng = np.random.default_rng(SEED + epoch)

        # Train
        model.train()
        perm = rng.choice(n, size=n, replace=True, p=sampler_probs)
        total_loss = 0.0
        total_ce = 0.0
        total_pen = 0.0
        n_batches = 0
        all_train_preds = []
        all_train_targets = []
        gate_ws_all = []

        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            wb = torch.ones(len(batch_idx), device=device)  # uniform within batch

            # Map global partner indices into within-batch positions for the
            # contrastive loss. partner_idx[i] = -1 → real sample (no partner).
            # For adversarial samples whose real partner is also in this batch,
            # the contrastive loss pulls their embeddings together.
            global_partners = real_partner_idx[batch_idx]  # numpy
            in_batch_partner = np.full_like(global_partners, -1)
            # Build lookup: global_idx → within-batch position
            global_to_pos = {int(g): pos for pos, g in enumerate(batch_idx)}
            for pos, g_partner in enumerate(global_partners):
                if g_partner >= 0 and int(g_partner) in global_to_pos:
                    in_batch_partner[pos] = global_to_pos[int(g_partner)]
            partner_t = torch.from_numpy(in_batch_partner.astype(np.int64)).to(device)

            if phase == "B":
                logits, feat_pen, gate_w = model.forward_with_features(
                    xb, return_gate_weights=True
                )
            else:
                logits, feat_pen = model.forward_with_features(xb)
                gate_w = None

            loss, ce, pen = constrained_loss(
                logits, yb, per_class_f1, wb, penalty_w, class_order
            )
            if gate_w is not None:
                gate_loss = gate_regularization_loss(gate_w)
                loss = loss + gate_loss
                gate_ws_all.append(gate_w.detach().cpu().numpy().mean(axis=0))

            # Gap 8 audit: contrastive loss on adversarial-real pairs.
            # Fires only when the batch contains at least one adversarial
            # sample whose real partner is also in the batch.
            if (in_batch_partner >= 0).any():
                con = contrastive_loss(feat_pen, yb, partner_t)
                loss = loss + CONTRASTIVE_WEIGHT * con

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_pen += float(pen)
            n_batches += 1
            all_train_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_train_targets.append(yb.cpu().numpy())

        train_preds = np.concatenate(all_train_preds)
        train_targets = np.concatenate(all_train_targets)
        train_macro_f1 = f1_score(train_targets, train_preds, average="macro",
                                    labels=list(range(9)), zero_division=0)

        # Val eval
        model.eval()
        with torch.no_grad():
            val_logits, val_gate_w = model(X_val_t, return_gate_weights=True) \
                if (phase == "B" and not model._gate_frozen) \
                else (model(X_val_t), None)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_targets = y_val_t.cpu().numpy()

        val_macro_f1 = f1_score(val_targets, val_preds, average="macro",
                                  labels=list(range(9)), zero_division=0)
        val_per_class = f1_score(val_targets, val_preds, average=None,
                                   labels=list(range(9)), zero_division=0)
        per_class_f1 = val_per_class.tolist()
        val_black_rot = float(val_per_class[class_order.index("brassica_black_rot")])
        val_cerc = float(val_per_class[class_order.index("okra_cercospora")])
        val_min = min(val_black_rot, val_cerc, val_macro_f1)

        gate_mean = (np.mean(gate_ws_all, axis=0).tolist() if gate_ws_all else
                      [1.0 / (4 if use_psv else 3)] * (4 if use_psv else 3))

        history["epoch"].append(epoch)
        history["phase"].append(phase)
        history["train_loss"].append(total_loss / n_batches)
        history["train_macro_f1"].append(float(train_macro_f1))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["val_black_rot"].append(val_black_rot)
        history["val_cercospora"].append(val_cerc)
        history["val_min_metric"].append(float(val_min))
        history["val_per_class"].append([float(v) for v in val_per_class])
        history["gate_weights_mean"].append(gate_mean)

        logger.info(
            f"  Ep {epoch:02d} [{phase}] loss={total_loss/n_batches:.4f} "
            f"train_f1={train_macro_f1:.4f} val_f1={val_macro_f1:.4f} "
            f"min={val_min:.4f} br={val_black_rot:.4f} ce={val_cerc:.4f}  "
            f"gate={[round(x, 3) for x in gate_mean]}"
        )

        # Safety: abort training if any class F1 stays below 0.01 for 3
        # consecutive epochs. ONLY active during Phase B — Phase A is by
        # design a frozen-gate warmup where most classes start near zero
        # before the classifier head trains up. Only trigger after Phase B
        # has had at least 3 epochs to learn (epoch >= phase_a_epochs + 2).
        if epoch >= phase_a_epochs + 2:
            recent_per_class = history["val_per_class"][-3:]
            zero_run = any(
                all(epoch_per_class[c] < 0.01 for epoch_per_class in recent_per_class)
                for c in range(9)
            )
            if zero_run:
                logger.warning(
                    f"Training collapsed: at least one class has F1 < 0.01 "
                    f"for 3 consecutive epochs in Phase B. Aborting at epoch {epoch}."
                )
                break

        if val_min > best["val_min"]:
            best = {
                "val_min": float(val_min),
                "state": deepcopy(model.state_dict()),
                "epoch": int(epoch),
                "val_macro_f1": float(val_macro_f1),
                "val_black_rot": val_black_rot,
                "val_cercospora": val_cerc,
                "val_per_class": [float(v) for v in val_per_class],
                "gate_mean": gate_mean,
            }

    return best, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback-3signal", action="store_true",
                        help="Use only Signals 1, 2, 4 (skip PSV)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("APIN SECTION 4 -- Stacking MLP + Two-Phase MoE training")
    logger.info("=" * 70)

    set_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Probe PSV cache
    psv_cache_path = CACHE_DIR / "signal3_psv_predictions_cache.pkl"
    use_psv = psv_cache_path.exists() and not args.fallback_3signal
    n_signals = 4 if use_psv else 3
    logger.info(f"PSV cache present: {psv_cache_path.exists()}")
    logger.info(f"Training mode: {'4-signal' if use_psv else '3-signal fallback'}")

    from scripts.apin.constants import MODEL2_CLASS_ORDER, NUM_CLASSES

    # Load caches + build inputs
    caches = load_signal_caches(use_psv)
    X_all, y_all, splits_all, is_field_all, sources_all, is_recomp_all, success_all, indices = \
        build_inputs(caches, use_psv)
    logger.info(f"Input shape: {X_all.shape}")
    logger.info(f"  Successful across all signals: {int(success_all.sum())}/{len(success_all)}")

    # Reliability matrix modulation
    if use_psv:
        # Compute inline for 4-signal
        sig_order = [1, 2, 3, 4]
        R = np.zeros((4, 9), dtype=np.float32)
        mask_field_val = (splits_all == "val_and_soup") & is_field_all
        for s, sig_id in enumerate(sig_order):
            preds_block = X_all[:, s*9:(s+1)*9]
            argmax_block = preds_block.argmax(axis=1)
            for c in range(9):
                sel = mask_field_val & (y_all == c)
                if sel.sum() > 0:
                    R[s, c] = (argmax_block[sel] == c).mean()
    else:
        R = load_reliability_matrix(use_psv=False)
    logger.info(f"Reliability matrix shape: {R.shape}")

    X_mod = apply_reliability_modulation(X_all, R, n_signals)

    # Split
    train_mask = (splits_all == "train") & success_all
    val_mask = (splits_all == "val_and_soup") & success_all

    X_train = X_mod[train_mask]; y_train = y_all[train_mask]
    is_field_train = is_field_all[train_mask]; sources_train = sources_all[train_mask]
    is_recomp_train = is_recomp_all[train_mask]

    X_val = X_mod[val_mask]; y_val = y_all[val_mask]

    logger.info(f"Train set: {len(y_train)}  Val set: {len(y_val)}")

    # Sample weights
    sample_weights = make_sample_weights(is_field_train, sources_train, MODEL2_CLASS_ORDER)
    logger.info(f"  Sample weights: min={sample_weights.min():.2f}, "
                f"mean={sample_weights.mean():.2f}, max={sample_weights.max():.2f}")

    # Adversarial augmentation — pass reliability matrix so injected
    # Dirichlet samples are modulated to match the training-distribution
    # scale of the rest of the row.
    adv_X, adv_y, adv_w, adv_pair, adv_stats = generate_adversarial_augmentation(
        X_train, y_train, is_recomp_train, MODEL2_CLASS_ORDER, n_signals,
        reliability_matrix=R)
    logger.info(f"Adversarial: generated={adv_stats['generated']}, "
                f"invalid_resamples={adv_stats['resampled_invalid']}")

    # Cap adversarial at ADV_MAX_FRACTION of training data
    max_adv = int(len(X_train) * ADV_MAX_FRACTION / (1 - ADV_MAX_FRACTION))
    if len(adv_X) > max_adv:
        logger.info(f"  Capping adversarial from {len(adv_X)} to {max_adv}")
        keep = np.random.default_rng(SEED).choice(len(adv_X), max_adv, replace=False)
        adv_X = adv_X[keep]; adv_y = adv_y[keep]; adv_w = adv_w[keep]
        adv_pair = adv_pair[keep]

    # Combine real + adversarial. Build pair_index for the contrastive
    # loss: -1 for real samples, otherwise the row index in X_train_full
    # of the real sample this adversarial copy was spawned from.
    if len(adv_X) > 0:
        X_train_full = np.vstack([X_train, adv_X])
        y_train_full = np.concatenate([y_train, adv_y])
        w_train_full = np.concatenate([sample_weights, adv_w])
        n_real = len(X_train)
        # For each adversarial sample, its real partner is at index adv_pair[i]
        # in the combined array (real samples occupy [0, n_real)).
        real_partner_idx = np.concatenate([
            np.full(n_real, -1, dtype=np.int64),
            adv_pair.astype(np.int64),  # already in [0, n_real)
        ])
    else:
        X_train_full = X_train; y_train_full = y_train; w_train_full = sample_weights
        real_partner_idx = np.full(len(X_train), -1, dtype=np.int64)

    logger.info(f"Full training set (real + adv): {len(y_train_full)} "
                f"(adv fraction: {len(adv_X)/len(y_train_full):.3f})")

    # Torch tensors
    X_train_t = torch.from_numpy(X_train_full.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train_full.astype(np.int64)).to(device)
    w_train_t = torch.from_numpy(w_train_full.astype(np.float32)).to(device)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    # Build model
    model = APIN_Ensemble(n_signals=n_signals, num_classes=9, dropout=DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model total params: {total_params}")

    # Train
    logger.info(f"\nTraining: {PHASE_A_EPOCHS} epochs Phase A + {PHASE_B_EPOCHS} epochs Phase B")
    best, history = train_one_model(
        X_train_t, y_train_t, w_train_t,
        X_val_t, y_val_t,
        model, device, MODEL2_CLASS_ORDER,
        use_psv, PHASE_A_EPOCHS, PHASE_B_EPOCHS, logger,
        real_partner_idx=real_partner_idx,
    )

    logger.info(f"\nBest epoch {best['epoch']}: val_macro_f1={best['val_macro_f1']:.4f}, "
                f"val_min={best['val_min']:.4f}")
    logger.info(f"  Per-class best: {[round(v,3) for v in best['val_per_class']]}")
    logger.info(f"  Gate mean: {best['gate_mean']}")

    # Save
    out_model = CACHE_DIR / f"apin_stacking_mlp_{TIMESTAMP}.pt"
    out_latest = CACHE_DIR / "apin_stacking_mlp.pt"
    out_cfg = CACHE_DIR / f"apin_stacking_mlp_config_{TIMESTAMP}.json"
    out_hist = CACHE_DIR / f"apin_stacking_mlp_history_{TIMESTAMP}.json"

    torch.save({
        "model_state_dict": best["state"],
        "best_epoch": best["epoch"],
        "val_macro_f1": best["val_macro_f1"],
        "val_per_class": best["val_per_class"],
        "gate_mean": best["gate_mean"],
        "n_signals": n_signals,
        "use_psv": use_psv,
        "reliability_matrix": R.tolist(),
        "class_order": MODEL2_CLASS_ORDER,
    }, out_model)
    torch.save({
        "model_state_dict": best["state"],
        "best_epoch": best["epoch"],
        "val_macro_f1": best["val_macro_f1"],
        "val_per_class": best["val_per_class"],
        "gate_mean": best["gate_mean"],
        "n_signals": n_signals,
        "use_psv": use_psv,
        "reliability_matrix": R.tolist(),
        "class_order": MODEL2_CLASS_ORDER,
    }, out_latest)

    config = {
        "timestamp": TIMESTAMP,
        "n_signals": n_signals,
        "use_psv": use_psv,
        "dropout": DROPOUT,
        "phase_a_epochs": PHASE_A_EPOCHS,
        "phase_b_epochs": PHASE_B_EPOCHS,
        "mlp_lr_phase_a": MLP_LR,
        "mlp_lr_phase_b": MLP_LR_PHASE_B,
        "gate_lr_warmup": GATE_LR_WARMUP,
        "gate_lr_full": GATE_LR_FULL,
        "weight_decay": WEIGHT_DECAY,
        "floors": FLOORS,
        "field_weight": FIELD_WEIGHT,
        "source_diverse_weight": SOURCE_DIVERSE_WEIGHT,
        "adv_alpha": ADV_ALPHA,
        "adv_copies_per_image": ADV_COPIES_PER_IMAGE,
        "adv_max_fraction": ADV_MAX_FRACTION,
        "gate_min_floor": GATE_MIN_FLOOR,
        "gate_entropy_weight": GATE_ENTROPY_WEIGHT,
        "gate_load_balance_weight": GATE_LOAD_BALANCE_WEIGHT,
        "class_order": MODEL2_CLASS_ORDER,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "best_val_per_class": {
            c: round(best["val_per_class"][i], 6)
            for i, c in enumerate(MODEL2_CLASS_ORDER)
        },
        "gate_mean_at_best": best["gate_mean"],
        "reliability_matrix": R.tolist(),
        "model_path": str(out_latest.relative_to(PROJECT_ROOT)),
        "adversarial_stats": adv_stats,
        "n_train": int(len(y_train)),
        "n_adv": int(len(adv_X)),
        "n_val": int(len(y_val)),
    }
    with open(out_cfg, "w") as f:
        json.dump(config, f, indent=2)
    with open(out_hist, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nSaved model: {out_latest.name}")
    logger.info(f"Saved config: {out_cfg.name}")
    logger.info(f"Saved history: {out_hist.name}")
    logger.info("=" * 70)
    logger.info("APIN SECTION 4 -- COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
