"""
Task A.1 — Build single-pass LoRA prototype bank + temperature calibration + tier thresholds.

Sibling of phase3_prototype_calibration.py but for the single-pass LoRA architecture.

KEY DIFFERENCES FROM PHASE 3 (the Phase 1 LADI-Net version):
  - Loads sp_lora_epoch13_f10.9113_PRESERVED.pt (SinglePassLoRA, NOT LADINet).
  - Feature space is CLS tokens from DINOv2-Reg-Base last block (NOT ABMIL bag_feat).
  - Output filenames use _sp_lora_ep13 suffix to avoid collision with Phase 1 artifacts.
  - feature_space metadata reads "CLS_token_768d" not "ABMIL_bag_feat_768d".

DEPLOYMENT NOTE (added post-Phase-4 reflection):
  The prototype bank produced here is a DIAGNOSTIC ASSET. It is NOT used to blend/modify
  predictions in the deployed inference path. PDA-1.2 in Phase 4 measured blending effect
  as NEUTRAL on the only empirical test available (2/6 correct before, 2/6 after).
  Task A.4 will run one ensemble variant with blending to empirically revisit this
  decision on field_val; if blending demonstrably helps on the larger sample, the
  deployment policy may change.

HARD CONSTRAINTS:
  - Only sp_lora_epoch13_f10.9113_PRESERVED.pt is accepted; val_sqrtn_f1 asserted ~0.9113.
  - Prototype feature = CLS token (out["cls"] from SinglePassLoRA).
  - Zero modifications to any file under scripts/apin/ or scripts/apin_v2/.
  - Zero touches on the locked 104-image held-out split.

Outputs:
  data/specialist/model3/prototype_bank_sp_lora_ep13.pt         (30 prototypes: 6 x 5)
  data/specialist/model3/phase3_calibration_sp_lora_ep13.json   (T_sp_lora + NLL/ECE)
  data/specialist/model3/phase3_tier_thresholds_sp_lora_ep13.json
  data/specialist/model3/phase3_train_field_features_sp_lora.pt
  data/specialist/model3/phase3_seed_features_sp_lora_{class}.pt  (6 files)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
# The single-pass LoRA model class.
from single_pass_lora_train import SinglePassLoRA


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
SEEDS_DIR = OUT_DIR / "prototype_seeds"
SP_LORA_CKPT = (PROJECT_ROOT / "models" / "specialist" / "sp_lora_checkpoints"
                / "sp_lora_epoch13_f10.9113_PRESERVED.pt")

SP_TRAIN_FIELD_FEATURES = OUT_DIR / "phase3_train_field_features_sp_lora.pt"
SP_SEED_FEATURES_TEMPLATE = OUT_DIR / "phase3_seed_features_sp_lora_{cls}.pt"
SP_PROTOTYPE_BANK_PATH = OUT_DIR / "prototype_bank_sp_lora_ep13.pt"
SP_CALIBRATION_JSON = OUT_DIR / "phase3_calibration_sp_lora_ep13.json"
SP_TIER_THRESHOLDS_JSON = OUT_DIR / "phase3_tier_thresholds_sp_lora_ep13.json"

# k-means hyperparameters (same as Phase 3 for consistency).
K_PROTOS = 5
HIGH_CONF_THRESHOLD = 0.85
K_MEANS_ITERS = 100
K_MEANS_TOL = 1e-4
K_MEANS_SEED = 42

# Temperature calibration.
T_SEARCH_RANGE = (0.5, 3.0)
T_BINARY_SEARCH_STEPS = 50
T_STABILITY_UNRELIABLE_SIGMA = 0.20  # > this -> use_calibration=false

# Tier threshold sweep.
TIER1A_GAP_MIN = 0.25
TIER1A_PRECISION_TARGET = 0.90
TIER_THRESHOLD_SWEEP = np.arange(0.60, 0.951, 0.02)


def log(msg: str):
    print(f"[A.1] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_sp_lora_model(device: torch.device) -> SinglePassLoRA:
    """Load single-pass LoRA epoch 13 preserved checkpoint. Asserts identity."""
    if not SP_LORA_CKPT.exists():
        raise FileNotFoundError(
            f"Single-pass LoRA preserved checkpoint not found at {SP_LORA_CKPT}. "
            f"Task A.1 cannot proceed without the epoch 13 weights."
        )
    log(f"Loading checkpoint: {SP_LORA_CKPT.name}")
    ckpt = torch.load(SP_LORA_CKPT, map_location=device, weights_only=False)

    val_f1 = float(ckpt.get("val_sqrtn_macro_f1", -1.0))
    if abs(val_f1 - 0.9113) > 1e-3:
        raise ValueError(
            f"Checkpoint val_sqrtn_macro_f1 = {val_f1:.6f} but expected ~ 0.9113. "
            f"Wrong checkpoint -- refusing to proceed."
        )
    log(f"  verified: epoch={ckpt.get('epoch')}, val_sqrtn_macro_f1={val_f1:.6f}")

    # Belt-and-suspenders: compare_server does `SinglePassLoRA(device).to(device)` and
    # `map_location=device` — both prevent LoRA adapter tensors being stranded on CPU.
    model = SinglePassLoRA(device).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log(f"  model on {device}, eval mode, all parameters frozen.")
    return model


# ---------------------------------------------------------------------------
# Feature extraction over records
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features_for_records(
    model: SinglePassLoRA, records: list[LadiRecord], device: torch.device,
    batch_size: int = 32, tag: str = "field",
) -> dict:
    """Runs single-pass LoRA in eval mode. Returns dict with CLS features + preds.

    Features are CLS tokens from the last DINOv2-Reg-Base transformer block
    (accessed via model.forward()["cls"]).
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

        # CLS tokens from last DINOv2-Reg-Base block. Cast to FP32 for downstream cov/knn.
        cls = out["cls"].float()
        logits = out["logits"].float()
        probs = torch.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)

        all_feat.append(cls.cpu())
        all_preds.append(preds.cpu())
        all_confs.append(confs.cpu())
        all_labels.append(labels.cpu())
        all_logits.append(logits.cpu())

        n_done += images.size(0)
        if n_done % 128 == 0 or n_done == len(records):
            log(f"  [{tag}] {n_done}/{len(records)} in {time.time() - t0:.1f}s")

    return {
        "features": torch.cat(all_feat, dim=0),
        "preds": torch.cat(all_preds, dim=0),
        "confs": torch.cat(all_confs, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "logits": torch.cat(all_logits, dim=0),
    }


def filter_field_training_records(all_train: list[LadiRecord]) -> list[LadiRecord]:
    field = [r for r in all_train if r.is_field_photo]
    log(f"  Filtered training pool: {len(field)} real-field images out of "
        f"{len(all_train)} total train records.")
    from collections import Counter
    counts = Counter(r.class_name for r in field)
    for c in TOMATO_CLASSES:
        log(f"    {c:35s}: {counts.get(c, 0):4d}")
    return field


# ---------------------------------------------------------------------------
# Seed feature extraction
# ---------------------------------------------------------------------------
def _build_seed_records(cls: str) -> list[LadiRecord]:
    """Construct LadiRecord list for the 50 seed images of a class."""
    import pandas as pd
    seed_file = SEEDS_DIR / cls / "seed_paths.txt"
    if not seed_file.exists():
        raise FileNotFoundError(f"Missing seed_paths.txt for class {cls}: {seed_file}")

    mask_log = pd.read_csv(MASK_LOG_CSV)
    mask_log["image_path_norm"] = mask_log["image_path"].map(_norm_path)
    mask_by_norm = {row.image_path_norm: row for row in mask_log.itertuples(index=False)}

    mask_by_name = {}
    for row in mask_log.itertuples(index=False):
        name = Path(str(row.image_path)).name
        mask_by_name.setdefault(name, row)

    records = []
    n_skipped = 0
    with open(seed_file, encoding="utf-8") as f:
        seed_paths = [ln.strip() for ln in f if ln.strip()]

    for rel_path in seed_paths:
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
            image_type="LAB_OK",
            is_field_photo=False,
            fg_path=str(fg_abs),
            mask_path=mask_abs,
        ))

    if n_skipped > 0:
        log(f"  [{cls}] WARN: skipped {n_skipped} seed images (missing fg_path in mask_log)")
    log(f"  [{cls}] Built {len(records)} seed records")
    return records


# ---------------------------------------------------------------------------
# GPU k-means (same as Phase 3)
# ---------------------------------------------------------------------------
def gpu_kmeans(
    X: torch.Tensor, k: int, seed: int = K_MEANS_SEED,
    iters: int = K_MEANS_ITERS, tol: float = K_MEANS_TOL,
) -> torch.Tensor:
    device = X.device
    N, D = X.shape
    assert N >= k, f"Need at least k={k} points; got N={N}."

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
    C = torch.stack(centers, dim=0).clone()

    for it in range(iters):
        xc = X @ C.T
        xc2 = (C * C).sum(dim=-1).unsqueeze(0)
        assign = (xc2 - 2 * xc).argmin(dim=-1)
        new_C = torch.zeros_like(C)
        counts = torch.zeros(k, device=device)
        new_C.index_add_(0, assign, X)
        counts.index_add_(0, assign, torch.ones(N, device=device))
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
        seed_feat = seed_features_by_class[cls].to(device)
        field_mask = correct_mask & (all_field_labels == ci)
        field_feat = all_field_feat[field_mask]

        combined = torch.cat([seed_feat, field_feat], dim=0)
        n_seeds_used[cls] = int(seed_feat.shape[0])
        n_field_used[cls] = int(field_feat.shape[0])
        log(f"  [{cls}] seeds={seed_feat.shape[0]}, correct-high-conf field="
            f"{field_feat.shape[0]}, combined={combined.shape[0]}")

        if combined.shape[0] < 10:
            log(f"    WARN: <10 combined features; picking 5 highest-confidence seeds.")
            protos = seed_feat[:5]
            if protos.shape[0] < K_PROTOS:
                pad_needed = K_PROTOS - protos.shape[0]
                protos = torch.cat([protos, protos[:pad_needed]], dim=0)
        else:
            protos = gpu_kmeans(combined, k=K_PROTOS, seed=K_MEANS_SEED)

        protos_norm = F.normalize(protos, dim=-1)
        cos_mat = protos_norm @ protos_norm.T
        k_here = cos_mat.shape[0]
        off_diag = (cos_mat.sum() - torch.diagonal(cos_mat).sum()) / (k_here * k_here - k_here)
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
        "source_checkpoint": SP_LORA_CKPT.name,
        "source_checkpoint_val_f1": 0.9113,
        "feature_space": "CLS_token_768d",
        "normalization": "raw (not L2-normalized) -- caller must L2-normalize for cosine",
        "deployment_role": ("DIAGNOSTIC ONLY. Not mechanically blended into production "
                            "inference output per post-Phase-4 decision. "
                            "Task A.4 includes one empirical blended-ensemble variant "
                            "to revisit this policy on field_val."),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------
def calibrate_temperature(
    logits: torch.Tensor, labels: torch.Tensor,
    T_range: tuple = T_SEARCH_RANGE, n_steps: int = T_BINARY_SEARCH_STEPS,
) -> tuple[float, float, float]:
    def nll_at(T: float) -> float:
        return float(F.cross_entropy(logits / T, labels).item())

    nll_before = nll_at(1.0)
    T_low, T_high = T_range
    for _ in range(n_steps):
        T_mid = (T_low + T_high) / 2
        eps = 1e-3
        if nll_at(T_mid - eps) < nll_at(T_mid + eps):
            T_high = T_mid
        else:
            T_low = T_mid
    T_opt = (T_low + T_high) / 2
    nll_after = nll_at(T_opt)
    return T_opt, nll_before, nll_after


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
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
# Tier thresholds
# ---------------------------------------------------------------------------
def find_tier1a_threshold(
    probs: torch.Tensor, labels: torch.Tensor,
    gap_min: float = TIER1A_GAP_MIN, precision_target: float = TIER1A_PRECISION_TARGET,
    sweep: np.ndarray = TIER_THRESHOLD_SWEEP,
) -> dict:
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
    log(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load single-pass LoRA epoch 13
    model = load_sp_lora_model(device)

    # === Extract training real-field features ===
    log("\n=== Extract training real-field CLS features ===")
    all_train = load_split_records("train")
    field_records = filter_field_training_records(all_train)
    field_extracted = extract_features_for_records(
        model, field_records, device, batch_size=32, tag="field_train",
    )
    # Sanity check: CLS tokens should have plausible FP32 L2 norm
    cls_norms = field_extracted["features"].float().norm(dim=-1)
    log(f"  CLS token L2 norms: min={cls_norms.min():.2f}, "
        f"mean={cls_norms.mean():.2f}, max={cls_norms.max():.2f}")

    torch.save({
        "features": field_extracted["features"],
        "preds": field_extracted["preds"],
        "confs": field_extracted["confs"],
        "labels": field_extracted["labels"],
        "logits": field_extracted["logits"],
        "n": int(field_extracted["features"].shape[0]),
        "feature_space": "CLS_token_768d",
        "source": "sp_lora_ep13_train_field",
    }, SP_TRAIN_FIELD_FEATURES)
    log(f"  saved: {SP_TRAIN_FIELD_FEATURES.name}")

    # === Extract seed features ===
    log("\n=== Extract seed CLS features (6 classes x 50 seeds) ===")
    seed_features_by_class: dict[str, torch.Tensor] = {}
    for cls in TOMATO_CLASSES:
        records = _build_seed_records(cls)
        if len(records) == 0:
            raise RuntimeError(f"No seed records built for {cls}. Abort.")
        seed_ext = extract_features_for_records(
            model, records, device, batch_size=32, tag=f"seed_{cls}",
        )
        seed_features_by_class[cls] = seed_ext["features"]
        torch.save({
            "class": cls,
            "features": seed_ext["features"],
            "preds": seed_ext["preds"],
            "confs": seed_ext["confs"],
            "labels": seed_ext["labels"],
            "n": int(seed_ext["features"].shape[0]),
            "feature_space": "CLS_token_768d",
        }, str(SP_SEED_FEATURES_TEMPLATE).format(cls=cls))

    # === Prototype bank via GPU k-means ===
    log("\n=== Build prototype bank via GPU k-means (CLS feature space) ===")
    bank = build_prototype_bank(seed_features_by_class, field_extracted, device)
    torch.save(bank, SP_PROTOTYPE_BANK_PATH)
    log(f"  saved: {SP_PROTOTYPE_BANK_PATH.name}")

    log("\n  [PDA-1.1 on single-pass] Intra-class centroid cosine similarity:")
    for cls, cos in bank["intra_class_centroid_cos"].items():
        warn = "  <- k=5 likely over-partitioning" if cos > 0.95 else ""
        log(f"    {cls:35s}: cos_sim = {cos:.4f}{warn}")

    # === Temperature calibration on confusable_pair_probe ===
    log("\n=== Temperature calibration on confusable_pair_probe (28 images) ===")
    probe_records = load_split_records("confusable_pair_probe")
    log(f"  Probe records: {len(probe_records)}")
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

    # Bootstrap T stability
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
    T_mean = float(np.mean(bootstrap_Ts))
    unreliable = T_std > T_STABILITY_UNRELIABLE_SIGMA
    log(f"  Bootstrap T stability: mean={T_mean:.4f}, std={T_std:.4f} "
        f"({'UNRELIABLE' if unreliable else 'STABLE'})")
    use_calibration = not unreliable

    calibration = {
        "T_optimal": float(T_opt),
        "nll_before": float(nll_before),
        "nll_after": float(nll_after),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "probe_n_images": len(probe_records),
        "calibrated_on": "confusable_pair_probe",
        "source_checkpoint": SP_LORA_CKPT.name,
        "pda_T_stability_bootstrap_mean": T_mean,
        "pda_T_stability_bootstrap_std": T_std,
        "pda_T_stability_interpretation": "UNRELIABLE" if unreliable else "STABLE",
        "use_calibration": use_calibration,
        "use_calibration_rationale": (
            f"Bootstrap sigma(T)={T_std:.4f} {'>' if unreliable else '<='} "
            f"{T_STABILITY_UNRELIABLE_SIGMA} threshold. "
            f"{'Defaulting to T=1.0 at inference.' if unreliable else 'Using measured T at inference.'}"
        ),
        "note": (
            "Calibrated on foliar+septoria probe (28 images, class split 20+8). "
            "Small sample drives bootstrap variance; treat T_optimal as a reference number, "
            "not a reliable estimate. See use_calibration flag for deployment decision."
        ),
    }
    with open(SP_CALIBRATION_JSON, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    log(f"  saved: {SP_CALIBRATION_JSON.name}")

    # === Tier thresholds on field_val ===
    log("\n=== Tier 1A threshold sweep on field_val (203 images) ===")
    fv_records = load_split_records("field_val")
    log(f"  field_val records: {len(fv_records)}")
    fv_ext = extract_features_for_records(
        model, fv_records, device, batch_size=32, tag="field_val",
    )
    T_apply = T_opt if use_calibration else 1.0
    fv_probs = torch.softmax(fv_ext["logits"] / T_apply, dim=-1)
    tier_result = find_tier1a_threshold(fv_probs, fv_ext["labels"])
    log(f"  Tier 1A threshold (first hit precision >= 0.90): {tier_result['chosen_threshold']}")
    log(f"  Sweep summary (threshold -> precision / coverage):")
    for row in tier_result["sweep"]:
        prec = row["precision"]
        prec_s = f"{prec:.3f}" if prec is not None else "   nan"
        log(f"    {row['threshold']:.2f} -> {prec_s} / {row['coverage']:.3f} (n={row['n']})")

    tier_doc = dict(tier_result)
    tier_doc["calibrated_with_T"] = float(T_apply)
    tier_doc["T_optimal_available"] = float(T_opt)
    tier_doc["use_calibration"] = use_calibration
    tier_doc["field_val_n"] = len(fv_records)
    tier_doc["source_checkpoint"] = SP_LORA_CKPT.name
    tier_doc["note"] = (
        "Thresholds computed on calibrated probabilities (logits / T_applied). "
        "If use_calibration=false, T_applied=1.0. If chosen_threshold is None, "
        "precision target was not reached; fall back to 0.72 per Spec Step 10."
    )
    with open(SP_TIER_THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(tier_doc, f, indent=2)
    log(f"  saved: {SP_TIER_THRESHOLDS_JSON.name}")

    # === Summary ===
    log("\n" + "=" * 64)
    log("TASK A.1 COMPLETE")
    log("=" * 64)
    log(f"  prototype bank     : {SP_PROTOTYPE_BANK_PATH.name}  "
        f"({sum(p.shape[0] for p in bank['prototypes'].values())} prototypes)")
    log(f"  calibration        : T_optimal={T_opt:.4f} sigma={T_std:.4f} "
        f"use_calibration={use_calibration}")
    log(f"  Tier 1A threshold  : {tier_result['chosen_threshold']}")
    log("")
    log("Next: Task A.2 (v3 tomato calibration + tier thresholds).")


if __name__ == "__main__":
    main()
