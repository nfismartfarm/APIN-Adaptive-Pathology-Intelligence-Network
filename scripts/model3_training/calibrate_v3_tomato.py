"""
Task A.2 — Temperature calibration + tier thresholds for v3's renormalized tomato subset.

v3 is a 10-class tomato+chilli specialist. For the tomato-only inference path, we extract
v3's first-6 logits (tomato subset, index-remapped to canonical TOMATO_CLASSES order),
renormalize via softmax, and fit a temperature T on the 28-image confusable_pair_probe.

KEY DETAILS:
  - v3 call signature: model(x, crop_mode=torch.tensor([2]), domain_labels=None).
  - crop_mode=2 ("uncertain"): the server doesn't know crop at inference time.
  - V3_INDEX_FOR_LORA_CLASS remapping: v3's [0..5] are NOT in the same order as LoRA's [0..5].
    v3 order: [foliar_spot, late_blight, septoria_leaf_spot, YLCV, mosaic, healthy, <chilli>*4]
    LoRA order (canonical): [foliar_spot, septoria_leaf_spot, late_blight, YLCV, mosaic, healthy]
    So v3->LoRA remap = [0, 2, 1, 3, 4, 5] (late_blight and septoria swapped).

DEPLOYMENT NOTE:
  v3's checkpoint is a "soup" of 5 ingredient epochs — averaging already provides some
  calibration smoothing. Per-sample 28-image probe T may be UNRELIABLE (small sample).
  If bootstrap sigma(T) > 0.20, use_calibration=false is written and inference falls back
  to T=1.0.

HARD CONSTRAINTS:
  - Zero modifications to any file under scripts/apin/ or scripts/apin_v2/.
  - Zero touches on the locked 104-image held-out split.
  - v3 checkpoint read-only.

Outputs:
  data/specialist/model3/phase3_calibration_v3_tomato.json
  data/specialist/model3/phase3_tier_thresholds_v3_tomato.json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ladi_net"))

from scripts.model3_training.architecture.model3_full import Model3
from scripts.model3_training.model3_config import (
    CLASS_NAMES as V3_CLASSES, NUM_CLASSES as V3_N,
    IMAGENET_MEAN as V3_MEAN, IMAGENET_STD as V3_STD,
    CHECKPOINT_DIR as V3_CKPT_DIR, PRODUCTION_V3_CHECKPOINT_NAME as V3_FNAME,
    LORA_RANK,
)
from scripts.model3_training.data.preprocessing import apply_lab_clahe as v3_apply_clahe

# LADI-Net imports (for split loader + class names)
from ladinet_config import PROJECT_ROOT as LADI_ROOT, TOMATO_CLASSES, CLASS_TO_IDX as LADI_CLASS_TO_IDX
from ladinet_dataloader import LadiRecord, load_split_records


# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
V3_CKPT = V3_CKPT_DIR / V3_FNAME
OUT_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
V3_CALIBRATION_JSON = OUT_DIR / "phase3_calibration_v3_tomato.json"
V3_TIER_THRESHOLDS_JSON = OUT_DIR / "phase3_tier_thresholds_v3_tomato.json"

# v3 -> LoRA canonical tomato index remap.
# v3's 10-class order has late_blight at idx 1, septoria at idx 2.
# LoRA's 6-class order has septoria at idx 1, late_blight at idx 2.
# Canonical TOMATO_CLASSES matches LoRA's order.
V3_INDEX_FOR_LORA_CLASS = [0, 2, 1, 3, 4, 5]

# Temperature calibration hyperparameters.
T_SEARCH_RANGE = (0.5, 3.0)
T_BINARY_SEARCH_STEPS = 50
T_STABILITY_UNRELIABLE_SIGMA = 0.20

# Tier threshold sweep.
TIER1A_GAP_MIN = 0.25
TIER1A_PRECISION_TARGET = 0.90
TIER_THRESHOLD_SWEEP = np.arange(0.60, 0.951, 0.02)

SEED = 42
V3_INPUT_RES = 224


def log(msg: str):
    print(f"[A.2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Load v3
# ---------------------------------------------------------------------------
def load_v3(device: torch.device) -> Model3:
    if not V3_CKPT.exists():
        raise FileNotFoundError(f"v3 checkpoint not found at {V3_CKPT}")
    log(f"Loading v3 checkpoint: {V3_CKPT.name}")
    ckpt = torch.load(V3_CKPT, map_location="cpu", weights_only=False)

    model = Model3(n_classes=V3_N, pretrained=False, lora_rank=LORA_RANK).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log(f"  run_name: {ckpt.get('run_name')}, "
        f"soup_field_f1: {ckpt.get('soup_selection_field_f1'):.4f}")
    return model


# ---------------------------------------------------------------------------
# Preprocessing: v3 uses 224px + LAB-CLAHE(L) + stretch-resize + RGB + ImageNet norm
# ---------------------------------------------------------------------------
def _preprocess_v3(img_bgr: np.ndarray) -> torch.Tensor:
    import cv2
    img = v3_apply_clahe(img_bgr)
    img = cv2.resize(img, (V3_INPUT_RES, V3_INPUT_RES), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(V3_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(V3_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Run v3 on records, extract 6-class tomato logits + labels
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_v3_tomato_logits(
    model: Model3, records: list[LadiRecord], device: torch.device, tag: str,
) -> dict:
    """Returns dict with 6-class tomato logits (index-remapped) + labels + preds.

    Applies v3 with crop_mode=[2] (uncertain).
    Then extracts the 6 tomato-class logits via V3_INDEX_FOR_LORA_CLASS.
    """
    import cv2
    # Read raw image bytes from paths; can't reuse LadiNetDataset because v3's
    # preprocessing is incompatible (224 stretch vs 392 letterbox).
    all_tomato_logits = []
    all_labels = []
    all_full_10class = []

    t0 = time.time()
    for i, rec in enumerate(records):
        img_bgr = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback: read via PIL if cv2 fails (e.g. some PNGs)
            from PIL import Image
            pil = Image.open(rec.image_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        x = _preprocess_v3(img_bgr).to(device)
        crop_mode = torch.tensor([2], dtype=torch.long, device=device)
        out = model(x, crop_mode, domain_labels=None)
        logits_full = out["logits"].float().squeeze(0).cpu()  # [10]

        # Extract 6 tomato logits via remap
        tomato_logits = logits_full[V3_INDEX_FOR_LORA_CLASS]  # [6]
        all_tomato_logits.append(tomato_logits)
        all_full_10class.append(logits_full)

        # Label in canonical TOMATO_CLASSES space (LoRA/LADI convention)
        canonical_label = LADI_CLASS_TO_IDX.get(rec.class_name, -1)
        all_labels.append(canonical_label)

        if (i + 1) % 32 == 0 or (i + 1) == len(records):
            log(f"  [{tag}] {i + 1}/{len(records)} in {time.time() - t0:.1f}s")

    return {
        "tomato_logits": torch.stack(all_tomato_logits, dim=0),   # [N, 6]
        "labels": torch.tensor(all_labels, dtype=torch.long),      # [N]
        "full_10class_logits": torch.stack(all_full_10class, dim=0),  # [N, 10]
    }


# ---------------------------------------------------------------------------
# Calibration helpers (same as A.1)
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
            rows.append({"threshold": float(thr), "n": 0, "precision": None, "coverage": 0.0})
            continue
        prec = float(correct[tier1a].float().mean().item())
        cov = n_1a / len(labels)
        rows.append({"threshold": float(thr), "n": n_1a, "precision": prec, "coverage": cov})
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

    # Sanity check the remap before any model loading:
    expected_map = {
        "tomato_foliar_spot": "tomato_foliar_spot",
        "tomato_septoria_leaf_spot": "tomato_septoria_leaf_spot",
        "tomato_late_blight": "tomato_late_blight",
        "tomato_yellow_leaf_curl_virus": "tomato_yellow_leaf_curl_virus",
        "tomato_mosaic_virus": "tomato_mosaic_virus",
        "tomato_healthy": "tomato_healthy",
    }
    for i, cls in enumerate(TOMATO_CLASSES):
        v3_idx = V3_INDEX_FOR_LORA_CLASS[i]
        v3_cls = V3_CLASSES[v3_idx]
        if v3_cls != expected_map[cls]:
            raise AssertionError(
                f"Remap error: LoRA[{i}]={cls} mapped to v3[{v3_idx}]={v3_cls} (expected {cls})"
            )
    log(f"  Remap verified: TOMATO_CLASSES -> v3 indices {V3_INDEX_FOR_LORA_CLASS}")

    model = load_v3(device)

    # === Calibration on confusable_pair_probe ===
    log("\n=== v3 tomato calibration on confusable_pair_probe (28 images) ===")
    probe_records = load_split_records("confusable_pair_probe")
    log(f"  Probe records: {len(probe_records)}")
    probe_ext = extract_v3_tomato_logits(model, probe_records, device, tag="probe")
    log(f"  Probe logits shape: {tuple(probe_ext['tomato_logits'].shape)}")

    # Sanity check renormalization
    probs_before = torch.softmax(probe_ext["tomato_logits"], dim=-1)
    sum_check = probs_before.sum(dim=-1)
    assert torch.allclose(sum_check, torch.ones_like(sum_check), atol=1e-5), \
        f"Renormalization fail: probs don't sum to 1 (got {sum_check[:3]})"
    log(f"  Renormalization verified: sum(probs) = 1.0 for all samples")

    T_opt, nll_before, nll_after = calibrate_temperature(
        probe_ext["tomato_logits"], probe_ext["labels"],
    )
    probs_after = torch.softmax(probe_ext["tomato_logits"] / T_opt, dim=-1)
    ece_before = compute_ece(probs_before, probe_ext["labels"])
    ece_after = compute_ece(probs_after, probe_ext["labels"])
    log(f"  T_optimal = {T_opt:.4f}")
    log(f"  NLL before = {nll_before:.4f}, NLL after = {nll_after:.4f}")
    log(f"  ECE before = {ece_before:.4f}, ECE after = {ece_after:.4f}")

    # Bootstrap T stability
    rng = np.random.default_rng(SEED)
    bootstrap_Ts = []
    for _ in range(10):
        idx = rng.integers(0, len(probe_ext["labels"]), size=len(probe_ext["labels"]))
        idx_t = torch.from_numpy(idx).long()
        T_b, _, _ = calibrate_temperature(
            probe_ext["tomato_logits"][idx_t], probe_ext["labels"][idx_t],
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
        "source_checkpoint": V3_CKPT.name,
        "source_architecture": "Model3 (DINOv2-Small-Registers + LoRA + FiLM + Linear(10))",
        "v3_to_lora_index_remap": V3_INDEX_FOR_LORA_CLASS,
        "pda_T_stability_bootstrap_mean": T_mean,
        "pda_T_stability_bootstrap_std": T_std,
        "pda_T_stability_interpretation": "UNRELIABLE" if unreliable else "STABLE",
        "use_calibration": use_calibration,
        "use_calibration_rationale": (
            f"Bootstrap sigma(T)={T_std:.4f} {'>' if unreliable else '<='} "
            f"{T_STABILITY_UNRELIABLE_SIGMA} threshold. "
            f"{'Defaulting to T=1.0 at inference.' if unreliable else 'Using measured T at inference.'} "
            f"Also note: v3 checkpoint is a 5-ingredient soup; averaging already smooths calibration."
        ),
        "note": (
            "v3 logits remapped from 10-class to 6-class via V3_INDEX_FOR_LORA_CLASS, "
            "then renormalized via softmax over 6 tomato classes. Probe is 28 images "
            "(20 foliar + 8 septoria). Calibrating over foliar/septoria only; other "
            "4 classes not represented in probe."
        ),
    }
    with open(V3_CALIBRATION_JSON, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    log(f"  saved: {V3_CALIBRATION_JSON.name}")

    # === Tier thresholds on field_val (tomato subset only) ===
    log("\n=== Tier 1A threshold sweep on field_val tomato subset ===")
    fv_records_all = load_split_records("field_val")
    fv_records = [r for r in fv_records_all if r.class_name.startswith("tomato_")]
    log(f"  field_val tomato subset: {len(fv_records)} / {len(fv_records_all)}")
    if len(fv_records) == 0:
        log("  WARN: no tomato records in field_val (all records may be other crops). "
            "Using empty sweep as placeholder.")
        tier_result = {"sweep": [], "chosen_threshold": None,
                       "gap_min": TIER1A_GAP_MIN, "precision_target": TIER1A_PRECISION_TARGET}
    else:
        fv_ext = extract_v3_tomato_logits(model, fv_records, device, tag="field_val_tomato")
        T_apply = T_opt if use_calibration else 1.0
        fv_probs = torch.softmax(fv_ext["tomato_logits"] / T_apply, dim=-1)
        tier_result = find_tier1a_threshold(fv_probs, fv_ext["labels"])
        log(f"  Tier 1A threshold (first hit precision >= 0.90): {tier_result['chosen_threshold']}")
        log(f"  Sweep summary:")
        for row in tier_result["sweep"]:
            prec = row["precision"]
            prec_s = f"{prec:.3f}" if prec is not None else "   nan"
            log(f"    {row['threshold']:.2f} -> {prec_s} / {row['coverage']:.3f} (n={row['n']})")

    tier_doc = dict(tier_result)
    tier_doc["calibrated_with_T"] = float(T_opt) if use_calibration else 1.0
    tier_doc["T_optimal_available"] = float(T_opt)
    tier_doc["use_calibration"] = use_calibration
    tier_doc["field_val_tomato_n"] = len(fv_records)
    tier_doc["source_checkpoint"] = V3_CKPT.name
    tier_doc["v3_to_lora_index_remap"] = V3_INDEX_FOR_LORA_CLASS
    tier_doc["note"] = (
        "v3 thresholds computed on tomato-only subset of field_val (filtered by class_name). "
        "If use_calibration=false, T_applied=1.0."
    )
    with open(V3_TIER_THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(tier_doc, f, indent=2)
    log(f"  saved: {V3_TIER_THRESHOLDS_JSON.name}")

    log("\n" + "=" * 64)
    log("TASK A.2 COMPLETE")
    log("=" * 64)
    log(f"  calibration        : T_optimal={T_opt:.4f} sigma={T_std:.4f} "
        f"use_calibration={use_calibration}")
    log(f"  Tier 1A threshold  : {tier_result['chosen_threshold']}")


if __name__ == "__main__":
    main()
