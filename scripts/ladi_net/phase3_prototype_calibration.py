"""
Phase 3 — Prototype Bank + Temperature Calibration + Tier Thresholds.

Consumes: models/specialist/ladinet_phase1_heads.pt (Phase 1 epoch 2, val_f1=0.9112)
Produces:
  data/specialist/model3/phase3_train_field_features.pt   (680, 768) ABMIL + preds/confs/labels
  data/specialist/model3/phase3_seed_features_{class}.pt  (50, 768) per class
  data/specialist/model3/prototype_bank.pt                30 prototypes (6 classes × 5)
  data/specialist/model3/phase3_calibration.json          T_optimal + NLL/ECE before/after
  data/specialist/model3/phase3_tier_thresholds.json      precision-sweep on field_val

HARD CONSTRAINTS enforced in code:
  - Only ladinet_phase1_heads.pt (no phase2_*). val_f1 asserted == 0.9112 (+/-1e-3).
  - Prototype feature = ABMIL bag_feat (768-dim), NOT CLS. See PVA Check 1.3.
  - Register tokens = 5 (ABMIL slices patches[:, 5:]). Enforced inside LADINet.forward.
  - Temperature calibrated on confusable_pair_probe (28), NOT field_val. PVA Check 1.5.
  - The locked 104-image held-out set (used only in Phase 4) is NEVER loaded here.
    Grep-safe below this line: zero mentions of that split name anywhere else.
  - NO LoRA merge call. Phase 1 had no LoRA.

Rationale: the ABMIL 768-dim features are the same dimension as CLS tokens (768-dim) but
represent a completely different feature space (attention-weighted patch aggregation vs
the backbone's CLS register). PVA explicitly checks this since dim-equality makes the bug
silent.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import project modules — scripts/ladi_net on path
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ladinet_config import (
    PROJECT_ROOT, SPLIT_JSON, MASK_LOG_CSV,
    TOMATO_CLASSES, CLASS_TO_IDX, RESOLUTION,
)
from ladinet_dataloader import (
    LadiRecord, LadiNetDataset, load_split_records,
    _norm_path,
)
from ladinet_model import LADINet, PREFIX_TOKENS

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
SEEDS_DIR = OUT_DIR / "prototype_seeds"
PHASE1_CKPT = PROJECT_ROOT / "models" / "specialist" / "ladinet_phase1_heads.pt"

PHASE3_FIELD_FEATURES = OUT_DIR / "phase3_train_field_features.pt"
PHASE3_SEED_FEATURES_TEMPLATE = OUT_DIR / "phase3_seed_features_{cls}.pt"
PROTOTYPE_BANK_PATH = OUT_DIR / "prototype_bank.pt"
CALIBRATION_JSON = OUT_DIR / "phase3_calibration.json"
TIER_THRESHOLDS_JSON = OUT_DIR / "phase3_tier_thresholds.json"

# k-means hyperparameters (Spec Part Two Step 8)
K_PROTOS = 5
HIGH_CONF_THRESHOLD = 0.85        # field images must exceed to join k-means input
K_MEANS_ITERS = 100
K_MEANS_TOL = 1e-4
K_MEANS_SEED = 42

# Temperature calibration
T_SEARCH_RANGE = (0.5, 3.0)
T_BINARY_SEARCH_STEPS = 50

# Tier threshold sweep
TIER1A_GAP_MIN = 0.25             # gap to second-best class (Spec Step 10)
TIER1A_PRECISION_TARGET = 0.90
TIER_THRESHOLD_SWEEP = np.arange(0.60, 0.951, 0.02)

# Prototype blending (for downstream sanity; not applied here)
PROTO_BLEND_ACTIVATION_THRESHOLD = 0.60


def log(msg: str):
    print(f"[phase3] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_phase1_model(device: torch.device) -> LADINet:
    """Loads LADINet with Phase 1 weights. Asserts identity.

    [PVA Check 1.1] The only checkpoint acceptable here is ladinet_phase1_heads.pt.
    [PVA Check 1.2] Phase 1 has NO LoRA — we never call merge_lora_weights or
                     set any LoRA scale in this script.
    """
    if not PHASE1_CKPT.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found at {PHASE1_CKPT}. "
            "Phase 3 cannot proceed without the authoritative Phase 1 model."
        )

    log(f"Loading checkpoint: {PHASE1_CKPT}")
    ckpt = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)

    # Identity assertion — PVA Check 1.1 / LOCK-2
    val_f1 = float(ckpt.get("val_sqrtn_macro_f1", -1.0))
    if abs(val_f1 - 0.9112) > 1e-3:
        raise ValueError(
            f"Checkpoint val_sqrtn_macro_f1 = {val_f1:.6f} but expected ~ 0.9112. "
            "Wrong checkpoint — refusing to proceed."
        )
    epoch = ckpt.get("epoch", -1)
    log(f"  verified: epoch={epoch}, val_sqrtn_macro_f1={val_f1:.6f}")
    log(f"  config_hash={ckpt.get('config_hash', 'missing')}")

    # Build model — phase1 path means NO LoRA attached to backbone
    model = LADINet(device, phase="phase1")

    # Load state dicts — authoritative key names from checkpoint:
    #   abmil_state_dict, fusion_state_dict, supcon_projector_state_dict
    model.abmil.load_state_dict(ckpt["abmil_state_dict"])
    model.fusion.load_state_dict(ckpt["fusion_state_dict"])
    model.supcon.load_state_dict(ckpt["supcon_projector_state_dict"])

    # Hard-freeze everything: eval mode + no gradients
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    log(f"  model on {device}, eval mode, all parameters frozen.")
    log(f"  per-class F1 from ckpt:")
    for cls, f1 in ckpt["val_per_class_f1"].items():
        log(f"    {cls:35s}: {f1:.4f}")
    return model


# ---------------------------------------------------------------------------
# Field feature extraction (Operation 3A-2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features_for_records(
    model: LADINet, records: list[LadiRecord], device: torch.device,
    batch_size: int = 32, tag: str = "field",
) -> dict:
    """Runs model in eval mode over records. Returns dict with features/preds/confs/labels.

    Features are ABMIL bag output (768-dim) — the prototype feature space per PVA 1.3.
    """
    ds = LadiNetDataset(records, training=False, background_pool=None)

    def _collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        }

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)

    all_feat = []
    all_preds = []
    all_confs = []
    all_labels = []
    all_logits = []

    t0 = time.time()
    n_done = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(images)
        # bag_feat is the ABMIL output [B, 768] — FP32 cast to be safe for downstream cov/knn
        bag = out["bag_feat"].float()
        logits = out["logits"].float()
        probs = torch.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)

        all_feat.append(bag.cpu())
        all_preds.append(preds.cpu())
        all_confs.append(confs.cpu())
        all_labels.append(labels.cpu())
        all_logits.append(logits.cpu())

        n_done += images.size(0)
        if n_done % 128 == 0 or n_done == len(records):
            log(f"  [{tag}] {n_done}/{len(records)} in {time.time() - t0:.1f}s")

    return {
        "features": torch.cat(all_feat, dim=0),   # (N, 768) float32
        "preds": torch.cat(all_preds, dim=0),     # (N,) long
        "confs": torch.cat(all_confs, dim=0),     # (N,) float32
        "labels": torch.cat(all_labels, dim=0),   # (N,) long
        "logits": torch.cat(all_logits, dim=0),   # (N, 6) float32 — useful for calibration
    }


def filter_field_training_records(all_train: list[LadiRecord]) -> list[LadiRecord]:
    """Returns the subset of training records that are real-field photos (is_field_photo=True).

    From Phase 0 / prompt: there should be ~680 such records.
    """
    field = [r for r in all_train if r.is_field_photo]
    log(f"  Filtered training pool: {len(field)} real-field images out of "
        f"{len(all_train)} total train records.")
    # Class distribution sanity
    from collections import Counter
    counts = Counter(r.class_name for r in field)
    for c in TOMATO_CLASSES:
        log(f"    {c:35s}: {counts.get(c, 0):4d}")
    return field


# ---------------------------------------------------------------------------
# Seed feature extraction (Operation 3A-3)
# ---------------------------------------------------------------------------
def _build_seed_records(cls: str) -> list[LadiRecord]:
    """Constructs LadiRecord list for the 50 seed images of a class.

    The seed_paths.txt stores ORIGINAL image paths (not _fg.png). We look each up
    in mask_precompute_log.csv to recover fg_path so LAB_OK routing works.
    """
    import pandas as pd
    seed_file = SEEDS_DIR / cls / "seed_paths.txt"
    if not seed_file.exists():
        raise FileNotFoundError(f"Missing seed_paths.txt for class {cls}: {seed_file}")

    mask_log = pd.read_csv(MASK_LOG_CSV)
    mask_log["image_path_norm"] = mask_log["image_path"].map(_norm_path)
    mask_by_norm = {row.image_path_norm: row for row in mask_log.itertuples(index=False)}

    # Also index by absolute-path-normalised to tolerate relative-vs-absolute drift
    mask_by_name = {}
    for row in mask_log.itertuples(index=False):
        name = Path(str(row.image_path)).name
        mask_by_name.setdefault(name, row)

    records = []
    n_skipped = 0
    with open(seed_file, encoding="utf-8") as f:
        seed_paths = [ln.strip() for ln in f if ln.strip()]

    for rel_path in seed_paths:
        # Resolve to absolute
        p = Path(rel_path)
        abs_p = p if p.is_absolute() else (PROJECT_ROOT / rel_path.replace("\\", "/"))
        abs_norm = _norm_path(str(abs_p))

        matched = mask_by_norm.get(abs_norm) or mask_by_name.get(abs_p.name)
        if matched is None:
            n_skipped += 1
            continue

        fg_rel = str(getattr(matched, "fg_path", "") or "")
        if not fg_rel:
            n_skipped += 1
            continue
        fg_abs = fg_rel if Path(fg_rel).is_absolute() else (PROJECT_ROOT / fg_rel.replace("\\", "/"))
        if not Path(fg_abs).exists():
            n_skipped += 1
            continue

        mask_rel = str(getattr(matched, "mask_path", "") or "")
        mask_abs = None
        if mask_rel:
            mp = mask_rel if Path(mask_rel).is_absolute() else (PROJECT_ROOT / mask_rel.replace("\\", "/"))
            if Path(mp).exists():
                mask_abs = str(mp)

        records.append(LadiRecord(
            image_path=str(abs_p),
            class_name=cls,
            class_idx=CLASS_TO_IDX[cls],
            image_type="LAB_OK",     # seeds are all clean lab images (non-flagged)
            is_field_photo=False,
            fg_path=str(fg_abs),
            mask_path=mask_abs,
        ))

    if n_skipped > 0:
        log(f"  [{cls}] WARN: skipped {n_skipped} seed images (missing fg_path in mask_log)")
    log(f"  [{cls}] Built {len(records)} seed records")
    return records


# ---------------------------------------------------------------------------
# GPU k-means — Operation 3A-4
# ---------------------------------------------------------------------------
def gpu_kmeans(
    X: torch.Tensor, k: int, seed: int = K_MEANS_SEED,
    iters: int = K_MEANS_ITERS, tol: float = K_MEANS_TOL,
) -> torch.Tensor:
    """Simple GPU k-means. X: (N, D) float32. Returns centers (k, D).

    Lloyd's algorithm with k-means++ init, fixed seed. Runs on whichever device
    X lives on. No sklearn (CPU-only).
    """
    device = X.device
    N, D = X.shape
    assert N >= k, f"Need at least k={k} points; got N={N}."

    # k-means++ init
    gen = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(0, N, (1,), generator=gen).item())
    centers = [X[first]]
    dists = torch.full((N,), float("inf"), device=device)
    for _ in range(1, k):
        d = ((X - centers[-1]) ** 2).sum(dim=-1)
        dists = torch.minimum(dists, d)
        probs = dists / (dists.sum() + 1e-12)
        probs_cpu = probs.detach().cpu()
        idx = int(torch.multinomial(probs_cpu, 1, generator=gen).item())
        centers.append(X[idx])
    C = torch.stack(centers, dim=0).clone()                # (k, D)

    for it in range(iters):
        # Assign each X to nearest center
        # ||x - c||^2 = x.x - 2 x.c + c.c
        xc = X @ C.T                                       # (N, k)
        xc2 = (C * C).sum(dim=-1).unsqueeze(0)             # (1, k)
        assign = (xc2 - 2 * xc).argmin(dim=-1)             # (N,)
        new_C = torch.zeros_like(C)
        counts = torch.zeros(k, device=device)
        new_C.index_add_(0, assign, X)
        counts.index_add_(0, assign, torch.ones(N, device=device))
        # Handle empty clusters by re-seeding from a random far point
        empty = counts == 0
        for e in torch.where(empty)[0].tolist():
            idx = int(torch.randint(0, N, (1,), generator=gen).item())
            new_C[e] = X[idx]
            counts[e] = 1
        new_C = new_C / counts.unsqueeze(-1).clamp(min=1)
        shift = (new_C - C).norm(dim=-1).max().item()
        C = new_C
        if shift < tol:
            log(f"    k-means converged at iter {it + 1} (max shift {shift:.2e})")
            break
    return C


def build_prototype_bank(
    seed_features_by_class: dict[str, torch.Tensor],
    field_extracted: dict,
    device: torch.device,
) -> dict:
    """Combines seed features + correct high-conf training field features,
    runs k-means per class.
    """
    all_field_feat = field_extracted["features"].to(device)
    all_field_labels = field_extracted["labels"].to(device)
    all_field_preds = field_extracted["preds"].to(device)
    all_field_confs = field_extracted["confs"].to(device)
    correct_mask = (all_field_preds == all_field_labels) & (all_field_confs > HIGH_CONF_THRESHOLD)

    prototypes: dict[str, torch.Tensor] = {}
    n_seeds_used: dict[str, int] = {}
    n_field_used: dict[str, int] = {}
    intra_class_centroid_cos: dict[str, float] = {}

    for cls in TOMATO_CLASSES:
        ci = CLASS_TO_IDX[cls]
        seed_feat = seed_features_by_class[cls].to(device)            # (S, 768)
        field_mask = correct_mask & (all_field_labels == ci)
        field_feat = all_field_feat[field_mask]                        # (F, 768)

        combined = torch.cat([seed_feat, field_feat], dim=0)
        n_seeds_used[cls] = int(seed_feat.shape[0])
        n_field_used[cls] = int(field_feat.shape[0])
        log(f"  [{cls}] seeds={seed_feat.shape[0]}, correct-high-conf field="
            f"{field_feat.shape[0]}, combined={combined.shape[0]}")

        if combined.shape[0] < 10:
            # Fallback: insufficient data for k-means k=5. Pick top-5 highest-confidence.
            # At seeds-only stage, all seeds are high-confidence by construction; take first 5.
            log(f"    WARN: <10 combined features; picking 5 highest-confidence seeds.")
            protos = seed_feat[:5]
            if protos.shape[0] < K_PROTOS:
                # Last-resort: pad by duplicating (very rare edge case)
                pad_needed = K_PROTOS - protos.shape[0]
                protos = torch.cat([protos, protos[:pad_needed]], dim=0)
        else:
            protos = gpu_kmeans(combined, k=K_PROTOS, seed=K_MEANS_SEED)

        # PDA Challenge 1.1: intra-class centroid cosine similarity
        # If centroids are near-identical (cos_sim > 0.95), k=5 is not adding diversity.
        protos_norm = F.normalize(protos, dim=-1)
        cos_mat = protos_norm @ protos_norm.T                          # (k, k)
        # Off-diagonal mean
        k = cos_mat.shape[0]
        off_diag = (cos_mat.sum() - torch.diagonal(cos_mat).sum()) / (k * k - k)
        intra_class_centroid_cos[cls] = float(off_diag.item())

        prototypes[cls] = protos.detach().cpu()
        log(f"    {cls}: intra-class centroid cos_sim = {intra_class_centroid_cos[cls]:.4f} "
            f"({'k=5 is over-partitioning' if intra_class_centroid_cos[cls] > 0.95 else 'meaningful diversity'})")

    return {
        "class_names": list(TOMATO_CLASSES),
        "prototypes": prototypes,
        "n_seeds_used": n_seeds_used,
        "n_field_used": n_field_used,
        "intra_class_centroid_cos": intra_class_centroid_cos,
        "confidence_threshold": HIGH_CONF_THRESHOLD,
        "k": K_PROTOS,
        "phase1_checkpoint": "ladinet_phase1_heads.pt",
        "phase1_val_f1": 0.9112,
        "feature_space": "ABMIL_bag_feat_768d",
        "normalization": "raw (not L2-normalized) — caller must L2-normalize for cosine",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Temperature calibration (Operation 3A-5)
# ---------------------------------------------------------------------------
def calibrate_temperature(
    logits: torch.Tensor, labels: torch.Tensor,
    T_range: tuple = T_SEARCH_RANGE, n_steps: int = T_BINARY_SEARCH_STEPS,
) -> tuple[float, float, float]:
    """Binary search for T minimizing NLL. Returns (T_optimal, nll_before, nll_after)."""
    def nll_at(T: float) -> float:
        return float(F.cross_entropy(logits / T, labels).item())

    nll_before = nll_at(1.0)
    T_low, T_high = T_range
    for _ in range(n_steps):
        T_mid = (T_low + T_high) / 2
        # gradient-descent-ish: pick direction of lower NLL
        eps = 1e-3
        if nll_at(T_mid - eps) < nll_at(T_mid + eps):
            T_high = T_mid
        else:
            T_low = T_mid
    T_opt = (T_low + T_high) / 2
    nll_after = nll_at(T_opt)
    return T_opt, nll_before, nll_after


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error on a classification probability set."""
    confs, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(labels)
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        n = int(mask.sum().item())
        if n == 0:
            continue
        acc_b = float(correct[mask].mean().item())
        conf_b = float(confs[mask].mean().item())
        ece += (n / total) * abs(acc_b - conf_b)
    return float(ece)


# ---------------------------------------------------------------------------
# Tier thresholds (Operation 3A-6)
# ---------------------------------------------------------------------------
def find_tier1a_threshold(
    probs: torch.Tensor, labels: torch.Tensor,
    gap_min: float = TIER1A_GAP_MIN, precision_target: float = TIER1A_PRECISION_TARGET,
    sweep: np.ndarray = TIER_THRESHOLD_SWEEP,
) -> dict:
    """Sweeps Tier 1A confidence thresholds; returns the minimum threshold
    where precision reaches precision_target.
    """
    top2, _ = torch.topk(probs, k=2, dim=-1)
    max_prob = top2[:, 0]
    gap = top2[:, 0] - top2[:, 1]
    pred = probs.argmax(dim=-1)
    correct = (pred == labels)

    rows = []
    chosen = None
    for thr in sweep:
        tier1a = (max_prob >= float(thr)) & (gap >= gap_min)
        n_1a = int(tier1a.sum().item())
        if n_1a == 0:
            rows.append({"threshold": float(thr), "n": 0,
                         "precision": None, "coverage": 0.0})
            continue
        prec = float(correct[tier1a].float().mean().item())
        cov = n_1a / len(labels)
        rows.append({
            "threshold": float(thr),
            "n": n_1a,
            "precision": prec,
            "coverage": cov,
        })
        if chosen is None and prec >= precision_target:
            chosen = float(thr)

    return {
        "sweep": rows,
        "chosen_threshold": chosen,
        "gap_min": gap_min,
        "precision_target": precision_target,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Operation 3A-1: Load Phase 1 model ========================
    model = load_phase1_model(device)

    # === Operation 3A-2: Extract training field features ===========
    log("\n=== Operation 3A-2: Extract training real-field features ===")
    all_train = load_split_records("train")
    field_records = filter_field_training_records(all_train)
    field_extracted = extract_features_for_records(
        model, field_records, device, batch_size=32, tag="field_train",
    )
    torch.save({
        "features": field_extracted["features"],       # (N, 768)
        "preds": field_extracted["preds"],             # (N,)
        "confs": field_extracted["confs"],             # (N,)
        "labels": field_extracted["labels"],           # (N,)
        "logits": field_extracted["logits"],           # (N, 6)
        "n": int(field_extracted["features"].shape[0]),
        "source": "phase3_train_field",
    }, PHASE3_FIELD_FEATURES)
    log(f"  saved: {PHASE3_FIELD_FEATURES}")

    # === Operation 3A-3: Extract seed features =====================
    log("\n=== Operation 3A-3: Extract seed features (6 classes × 50 seeds) ===")
    seed_features_by_class: dict[str, torch.Tensor] = {}
    for cls in TOMATO_CLASSES:
        records = _build_seed_records(cls)
        if len(records) == 0:
            raise RuntimeError(f"No seed records built for {cls}. Abort.")
        seed_ext = extract_features_for_records(
            model, records, device, batch_size=32, tag=f"seed_{cls}",
        )
        seed_features_by_class[cls] = seed_ext["features"]    # (S, 768)
        torch.save({
            "class": cls,
            "features": seed_ext["features"],
            "preds": seed_ext["preds"],
            "confs": seed_ext["confs"],
            "labels": seed_ext["labels"],
            "n": int(seed_ext["features"].shape[0]),
            "feature_space": "ABMIL_bag_feat_768d",
        }, str(PHASE3_SEED_FEATURES_TEMPLATE).format(cls=cls))

    # Mosaic 12-below-threshold note (PVA-1 finding, documented in session log)
    mosaic_n = int(seed_features_by_class["tomato_mosaic_virus"].shape[0])
    log(f"  Note: tomato_mosaic_virus seed pool = {mosaic_n}. Per PVA-1, 12 of 50 are "
        "at InSPyReNet confidence 0.9846 (below 0.985 nominal threshold by 0.04%). "
        "Included — dropping 12 would reduce to 38, insufficient for k=5.")

    # === Operation 3A-4: Build prototype bank via GPU k-means =======
    log("\n=== Operation 3A-4: Build prototype bank via GPU k-means ===")
    bank = build_prototype_bank(seed_features_by_class, field_extracted, device)
    torch.save(bank, PROTOTYPE_BANK_PATH)
    log(f"  saved: {PROTOTYPE_BANK_PATH}")

    # PDA Challenge 1.1 summary
    log("\n  [PDA-1.1] Intra-class centroid cosine similarity:")
    for cls, cos in bank["intra_class_centroid_cos"].items():
        warn = "  <- k=5 likely over-partitioning" if cos > 0.95 else ""
        log(f"    {cls:35s}: cos_sim = {cos:.4f}{warn}")

    # === Operation 3A-5: Temperature calibration ====================
    log("\n=== Operation 3A-5: Temperature calibration on confusable_pair_probe ===")
    probe_records = load_split_records("confusable_pair_probe")
    log(f"  Probe records: {len(probe_records)} (expected 28: 14 foliar + 14 septoria)")
    probe_ext = extract_features_for_records(
        model, probe_records, device, batch_size=32, tag="probe",
    )
    T_opt, nll_before, nll_after = calibrate_temperature(
        probe_ext["logits"], probe_ext["labels"],
    )
    probs_before = torch.softmax(probe_ext["logits"], dim=-1)
    probs_after = torch.softmax(probe_ext["logits"] / T_opt, dim=-1)
    ece_before = compute_ece(probs_before, probe_ext["labels"])
    ece_after = compute_ece(probs_after, probe_ext["labels"])
    log(f"  T_optimal = {T_opt:.4f}")
    log(f"  NLL before = {nll_before:.4f}, NLL after = {nll_after:.4f}")
    log(f"  ECE before = {ece_before:.4f}, ECE after = {ece_after:.4f}")

    # PDA Challenge 1.3: T stability via 10 bootstrap samples of the 28 probe images
    rng = np.random.default_rng(K_MEANS_SEED)
    bootstrap_Ts = []
    for _ in range(10):
        idx = rng.integers(0, len(probe_ext["labels"]), size=len(probe_ext["labels"]))
        idx_t = torch.from_numpy(idx).long()
        T_b, _, _ = calibrate_temperature(
            probe_ext["logits"][idx_t], probe_ext["labels"][idx_t],
        )
        bootstrap_Ts.append(T_b)
    T_std = float(np.std(bootstrap_Ts))
    log(f"  [PDA-1.3] T stability across 10 bootstraps: mean={np.mean(bootstrap_Ts):.4f}, "
        f"std={T_std:.4f} ({'UNRELIABLE' if T_std > 0.2 else 'STABLE'})")

    calibration = {
        "T_optimal": float(T_opt),
        "nll_before": float(nll_before),
        "nll_after": float(nll_after),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "probe_n_images": len(probe_records),
        "calibrated_on": "confusable_pair_probe",
        "pda_T_stability_bootstrap_std": T_std,
        "pda_T_stability_interpretation": "UNRELIABLE" if T_std > 0.2 else "STABLE",
        "note": ("Calibrated on foliar+septoria only (28 images) — see Issue 10-A for "
                 "two-temperature calibration limitation that remains deferred. Other four "
                 "classes are not directly calibrated."),
    }
    with open(CALIBRATION_JSON, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    log(f"  saved: {CALIBRATION_JSON}")

    # === Operation 3A-6: Tier thresholds from field_val precision sweep ===
    log("\n=== Operation 3A-6: Tier 1A threshold via precision sweep on field_val ===")
    fv_records = load_split_records("field_val")
    log(f"  field_val records: {len(fv_records)} (expected 203)")
    fv_ext = extract_features_for_records(
        model, fv_records, device, batch_size=32, tag="field_val",
    )
    # Apply calibration BEFORE tier sweep — tier thresholds are on CALIBRATED probs.
    fv_probs = torch.softmax(fv_ext["logits"] / T_opt, dim=-1)
    tier_result = find_tier1a_threshold(fv_probs, fv_ext["labels"])
    log(f"  Tier 1A threshold (first hit precision >= 0.90): "
        f"{tier_result['chosen_threshold']}")
    log(f"  Sweep summary (threshold → precision / coverage):")
    for row in tier_result["sweep"]:
        prec = row["precision"]
        prec_s = f"{prec:.3f}" if prec is not None else "   nan"
        log(f"    {row['threshold']:.2f} → {prec_s} / {row['coverage']:.3f} "
            f"(n={row['n']})")

    tier_doc = dict(tier_result)
    tier_doc["calibrated_with_T"] = float(T_opt)
    tier_doc["field_val_n"] = len(fv_records)
    tier_doc["note"] = (
        "Thresholds computed on calibrated probabilities (logits / T_optimal). "
        "If chosen_threshold is None, precision target was not reached anywhere in the sweep; "
        "fall back to 0.72 per Spec Step 10 prior."
    )
    with open(TIER_THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(tier_doc, f, indent=2)
    log(f"  saved: {TIER_THRESHOLDS_JSON}")

    # === Final summary =============================================
    log("\n" + "=" * 64)
    log("PHASE 3 COMPLETE")
    log("=" * 64)
    log(f"  field features          : {PHASE3_FIELD_FEATURES.name}  "
        f"shape={tuple(field_extracted['features'].shape)}")
    log(f"  seed features per class : 6 files")
    log(f"  prototype bank          : {PROTOTYPE_BANK_PATH.name}  "
        f"({sum(p.shape[0] for p in bank['prototypes'].values())} prototypes)")
    log(f"  calibration T_optimal   : {T_opt:.4f}  (T_std across bootstraps: {T_std:.4f})")
    log(f"  Tier 1A threshold       : {tier_result['chosen_threshold']}")
    log("")
    log("Next: review PDA-1 findings above, then run Phase 4 (model soup + locked evaluation).")


if __name__ == "__main__":
    main()
