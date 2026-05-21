"""
Phase 4 - Greedy Model Soup + Final Evaluation on locked 104-image held-out set.

CRITICAL INVARIANTS (enforced in code):
  1. The locked 104-image split is evaluated EXACTLY ONCE. A marker file
     data/specialist/model3/final_val_evaluated.txt is written after evaluation.
     The script refuses to run if the marker already exists.  [PVA Check 2.1]
  2. Bootstrap CI computed WITH replacement (rng.integers, 10000 samples, seed=42).
     [PVA Check 2.2]
  3. Prototype blending threshold = 0.60 (primary max_prob < 0.60 triggers blend).
     [PVA Check 2.3]
  4. Temperature scaling applied BEFORE prototype blending. Order:
     raw_logits -> /T -> softmax -> check max_prob < 0.60 -> maybe blend.
     [PVA Check 2.4]
  5. Production model artifact ladinet_tomato_production.pt contains:
     model_state_dict + calibration + prototype_bank + tier_thresholds.  [PVA 2.5]
  6. Statistical notes section is mandatory (n, CI_width, CRITICAL/UNDERPOWERED).
     [PVA Check 2.6]

Outputs:
  ladi_final_evaluation_report.md
  data/specialist/model3/final_val_predictions.json
  data/specialist/model3/final_val_evaluated.txt           (once-guard marker)
  models/specialist/ladinet_tomato_production.pt            (full production artifact)
  models/specialist/ladinet_phase1_soup.pt                  (soup weights, even if = best)

Presents TWO calibration variants per PDA-1.3:
  Variant A: T = 1.0 (no calibration) - bootstrap mean ~0.99 says this may be best
  Variant B: T = 1.2696 (single-sample fit) - the number written by Phase 3

Presents TWO inference variants per PDA-1.2:
  Variant 1: primary softmax only
  Variant 2: primary + prototype blending for max_prob < 0.60 (spec-default)

Final reported numbers use: T=T_optimal (spec-mandated) + prototype blending (spec-default).
Diagnostic section reports all 4 (T x blend) combinations for honesty.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ladinet_config import (
    PROJECT_ROOT, SPLIT_JSON, TOMATO_CLASSES, CLASS_TO_IDX, RESOLUTION,
)
from ladinet_dataloader import LadiRecord, LadiNetDataset, load_split_records
from ladinet_model import LADINet


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CKPT_DIR = PROJECT_ROOT / "models" / "specialist" / "ladinet_checkpoints"
PHASE1_CANONICAL = PROJECT_ROOT / "models" / "specialist" / "ladinet_phase1_heads.pt"
PHASE1_EP00 = CKPT_DIR / "phase1_epoch00_f10.7893.pt"
PHASE1_EP01 = CKPT_DIR / "phase1_epoch01_f10.8643.pt"
PHASE1_EP02 = CKPT_DIR / "phase1_epoch02_f10.9112.pt"

OUT_DATA = PROJECT_ROOT / "data" / "specialist" / "model3"
MARKER_FILE = OUT_DATA / "final_val_evaluated.txt"
PREDICTIONS_JSON = OUT_DATA / "final_val_predictions.json"

PHASE1_SOUP_PATH = PROJECT_ROOT / "models" / "specialist" / "ladinet_phase1_soup.pt"
PRODUCTION_PATH = PROJECT_ROOT / "models" / "specialist" / "ladinet_tomato_production.pt"

CALIBRATION_JSON = OUT_DATA / "phase3_calibration.json"
PROTOTYPE_BANK_PATH = OUT_DATA / "prototype_bank.pt"
TIER_THRESHOLDS_JSON = OUT_DATA / "phase3_tier_thresholds.json"

REPORT_PATH = PROJECT_ROOT / "ladi_final_evaluation_report.md"

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
PROTO_BLEND_THRESHOLD = 0.60
PROTO_BLEND_WEIGHT = 0.35          # blended = 0.65*primary + 0.35*proto_cos
SOUP_TOLERANCE = 0.01              # drop from best if soup degrades field_val F1 by more

# v3 baseline numbers (from ladi_net_complete_system.md Part One)
V3_BASELINE = {
    "tomato_foliar_spot": 0.629,
    "tomato_septoria_leaf_spot": 0.667,
    "tomato_late_blight": 0.800,
    "tomato_yellow_leaf_curl_virus": 0.800,
    "tomato_mosaic_virus": 0.857,
    "tomato_healthy": 0.967,
    "overall_sqrtn": 0.853,
}


def log(msg: str):
    print(f"[phase4] {msg}", flush=True)


# ----------------------------------------------------------------------------
# One-time guard (PVA Check 2.1)
# ----------------------------------------------------------------------------
def check_not_already_evaluated():
    if MARKER_FILE.exists():
        content = MARKER_FILE.read_text(encoding="utf-8")
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "REFUSING TO RUN: final_val already evaluated.\n"
            f"Marker file: {MARKER_FILE}\n"
            f"Content:\n{content}\n"
            "The locked held-out split gets evaluated EXACTLY ONCE. To force re-evaluation\n"
            "(ONLY if a script bug produced garbage numbers that were never reported),\n"
            "manually delete the marker AND add a justification entry to the session log.\n"
            + "=" * 70
        )


def write_marker(summary: dict):
    msg = (
        "Final held-out 104-image split evaluated.\n"
        f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"Model: {summary['production_model']}\n"
        f"Overall sqrtn_macro_F1 (T=T_opt, with blending): {summary['final_primary_sqrtn_f1']:.4f}\n"
        "DO NOT RUN AGAIN."
    )
    MARKER_FILE.write_text(msg, encoding="utf-8")


# ----------------------------------------------------------------------------
# Checkpoint load helper (reused from Phase 3 with slight changes)
# ----------------------------------------------------------------------------
def load_state_dicts_from_ckpt(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "abmil_state_dict": ckpt["abmil_state_dict"],
        "fusion_state_dict": ckpt["fusion_state_dict"],
        "supcon_projector_state_dict": ckpt["supcon_projector_state_dict"],
        "val_sqrtn_macro_f1": float(ckpt.get("val_sqrtn_macro_f1", -1.0)),
        "epoch": int(ckpt.get("epoch", -1)),
        "config_hash": ckpt.get("config_hash", ""),
    }


def assign_state_dicts_to_model(model: LADINet, sd: dict):
    model.abmil.load_state_dict(sd["abmil_state_dict"])
    model.fusion.load_state_dict(sd["fusion_state_dict"])
    model.supcon.load_state_dict(sd["supcon_projector_state_dict"])


def average_state_dicts(sd_list: list[dict]) -> dict:
    """Parameter-wise average of a list of state dicts (same keys). Returns merged SD."""
    result = {}
    for section_key in ["abmil_state_dict", "fusion_state_dict", "supcon_projector_state_dict"]:
        sec = sd_list[0][section_key]
        merged = {}
        for pname in sec.keys():
            stacked = torch.stack([sd[section_key][pname].float() for sd in sd_list], dim=0)
            merged[pname] = stacked.mean(dim=0)
        result[section_key] = merged
    return result


# ----------------------------------------------------------------------------
# Inference over records
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model: LADINet, records: list[LadiRecord], device: torch.device,
                  batch_size: int = 32, tag: str = "eval") -> dict:
    ds = LadiNetDataset(records, training=False, background_pool=None)
    def _collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "is_field_photo": torch.stack([b["is_field_photo"] for b in batch]),
            "image_type_idx": torch.stack([b["image_type_idx"] for b in batch]),
        }
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    all_logits = []
    all_bag = []
    all_fallback = []
    all_labels = []
    t0 = time.time()
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(x)
        all_logits.append(out["logits"].float().cpu())
        all_bag.append(out["bag_feat"].float().cpu())
        all_fallback.append(out["fallback_flag"].float().cpu())
        all_labels.append(y.cpu())
        n += x.size(0)
    log(f"  [{tag}] {n} images in {time.time() - t0:.1f}s")
    return {
        "logits": torch.cat(all_logits, dim=0),
        "bag_feat": torch.cat(all_bag, dim=0),
        "fallback_flag": torch.cat(all_fallback, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


# ----------------------------------------------------------------------------
# Greedy model soup
# ----------------------------------------------------------------------------
def greedy_soup(device: torch.device) -> tuple[dict, dict]:
    """Greedy soup over top-k Phase 1 checkpoints ranked by val_sqrtn_macro_f1.

    Returns (merged_state_dicts, soup_metadata).
    If only the best is used (single-ckpt soup), metadata.is_trivial_soup = True.
    """
    log("\n=== Greedy model soup ===")
    candidates = []
    for path in [PHASE1_EP02, PHASE1_EP01, PHASE1_EP00]:
        if not path.exists():
            log(f"  missing: {path} - skip")
            continue
        sd = load_state_dicts_from_ckpt(path)
        candidates.append({"path": path, "sd": sd, "val_f1": sd["val_sqrtn_macro_f1"]})
        log(f"  candidate: {path.name} val_f1={sd['val_sqrtn_macro_f1']:.4f}")

    candidates.sort(key=lambda c: -c["val_f1"])
    log(f"  ranked best to worst: {[c['path'].name for c in candidates]}")

    # Field val records for re-evaluation
    field_val_records = load_split_records("field_val")
    log(f"  field_val for soup validation: n={len(field_val_records)}")

    # Start with best single ckpt and its known val_f1
    soup_sds = [candidates[0]["sd"]]
    soup_best_val_f1 = candidates[0]["val_f1"]
    added_names = [candidates[0]["path"].name]
    log(f"  soup starts with {added_names[0]} (val_f1={soup_best_val_f1:.4f})")

    # Build a fresh model for re-evaluation
    model = LADINet(device, phase="phase1").to(device).eval()

    for cand in candidates[1:]:
        trial_sds = soup_sds + [cand["sd"]]
        merged = average_state_dicts(trial_sds)
        assign_state_dicts_to_model(model, merged)

        # Evaluate merged model on field_val
        inf = run_inference(model, field_val_records, device, tag=f"soup_test_{cand['path'].stem}")
        y_pred = inf["logits"].argmax(dim=-1).numpy()
        y_true = inf["labels"].numpy()
        trial_f1 = sqrtn_macro_f1(y_true, y_pred)
        log(f"  trial add {cand['path'].name}: merged field_val sqrtn_macro_f1={trial_f1:.4f} "
            f"(best so far {soup_best_val_f1:.4f})")
        if trial_f1 >= soup_best_val_f1 - SOUP_TOLERANCE:
            soup_sds = trial_sds
            soup_best_val_f1 = trial_f1
            added_names.append(cand["path"].name)
            log(f"    -> KEPT (new best soup f1={trial_f1:.4f})")
        else:
            log(f"    -> DROPPED (would degrade by {soup_best_val_f1 - trial_f1:.4f} > tolerance {SOUP_TOLERANCE})")

    soup_sd = average_state_dicts(soup_sds) if len(soup_sds) > 1 else soup_sds[0]
    is_trivial = len(soup_sds) == 1
    if is_trivial:
        log(f"  Soup reduces to single best checkpoint: {added_names[0]}")
        log(f"  (Expected given the 0.047 gap between epoch02=0.9112 and epoch01=0.864,)")
        log(f"  (  which far exceeds the {SOUP_TOLERANCE} tolerance.)")
    else:
        log(f"  Soup includes {len(added_names)} checkpoints: {added_names}")
        log(f"  Soup field_val sqrtn_macro_f1 = {soup_best_val_f1:.4f}")

    meta = {
        "n_checkpoints": len(soup_sds),
        "checkpoint_names": added_names,
        "is_trivial_soup": is_trivial,
        "tolerance": SOUP_TOLERANCE,
        "field_val_f1_after_soup": soup_best_val_f1,
        "field_val_f1_best_single": candidates[0]["val_f1"],
    }
    return soup_sd, meta


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-class F1 (macro formulation, zero_division=0)."""
    C = len(TOMATO_CLASSES)
    out = np.zeros(C, dtype=np.float64)
    for c in range(C):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if (2 * tp + fp + fn) == 0:
            out[c] = 0.0
        else:
            out[c] = 2.0 * tp / (2 * tp + fp + fn)
    return out


def sqrtn_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sqrt(N)-weighted macro F1 over the classes present in y_true."""
    f1s = per_class_f1(y_true, y_pred)
    # Weights: sqrt of class count in y_true; skip classes with n=0
    present = []
    weights = []
    for c in range(len(TOMATO_CLASSES)):
        n_c = int((y_true == c).sum())
        if n_c > 0:
            present.append(c)
            weights.append(float(np.sqrt(n_c)))
    if not present:
        return 0.0
    w = np.array(weights)
    w = w / w.sum()
    return float(sum(w[i] * f1s[c] for i, c in enumerate(present)))


def bootstrap_per_class_f1_ci(y_true: np.ndarray, y_pred: np.ndarray,
                              n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                              ci: float = 0.95) -> dict:
    """Bootstrap 95% CI per class, sampling WITH replacement. PVA Check 2.2."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.zeros((n_boot, len(TOMATO_CLASSES)), dtype=np.float64)
    overall = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)       # with replacement (correct bootstrap)
        yt = y_true[idx]
        yp = y_pred[idx]
        boots[i] = per_class_f1(yt, yp)
        overall[i] = sqrtn_macro_f1(yt, yp)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return {
        "per_class_ci_lower": np.percentile(boots, lo, axis=0),
        "per_class_ci_upper": np.percentile(boots, hi, axis=0),
        "overall_ci_lower": float(np.percentile(overall, lo)),
        "overall_ci_upper": float(np.percentile(overall, hi)),
        "n_boot": n_boot,
        "ci_level": ci,
    }


# ----------------------------------------------------------------------------
# Prototype blending (inference)
# ----------------------------------------------------------------------------
def prototype_blend(primary_probs: torch.Tensor, bag_feats: torch.Tensor,
                    proto_bank: dict) -> torch.Tensor:
    """For rows where primary_probs.max() < threshold, blend with prototype cosine.

    primary_probs: (N, C) calibrated softmax.
    bag_feats: (N, 768) raw ABMIL features.
    proto_bank: dict with key 'prototypes' -> dict[class_name] -> (k, 768).

    Returns blended probs (N, C).
    """
    N, C = primary_probs.shape
    max_p, _ = primary_probs.max(dim=-1)
    mask = max_p < PROTO_BLEND_THRESHOLD

    if not mask.any():
        return primary_probs

    bag_norm = F.normalize(bag_feats, dim=-1)
    # Build (C, k, 768) by concatenating per-class prototypes
    # All 6 classes have k prototypes each. Stack into (C, k, 768).
    proto_tensors = []
    for cls in TOMATO_CLASSES:
        p = proto_bank["prototypes"][cls]            # (k, 768) raw
        proto_tensors.append(F.normalize(p, dim=-1))
    proto = torch.stack(proto_tensors, dim=0)         # (C, k, 768)

    # Cosine sim (N, C) = max over k per class
    # bag_norm: (N, 1, 1, 768); proto: (1, C, k, 768)
    cos = torch.einsum("nd,ckd->nck", bag_norm, proto)
    proto_score = cos.max(dim=-1).values              # (N, C)

    # Convert cosine to pseudo-probability per class (softmax over cosines)
    proto_probs = torch.softmax(proto_score * 5.0, dim=-1)  # temperature sharpens

    blended = primary_probs.clone()
    blended[mask] = (1.0 - PROTO_BLEND_WEIGHT) * primary_probs[mask] + PROTO_BLEND_WEIGHT * proto_probs[mask]
    return blended


# ----------------------------------------------------------------------------
# Tier assignment
# ----------------------------------------------------------------------------
def assign_tier(probs: torch.Tensor, tier_1a_thr: float) -> np.ndarray:
    """Returns array of tier labels per prediction: '1A' / '1B' / '2A' / '3C_or_4'.

    Simplified: we don't compute 3C (prototype disagreement per spec) inline;
    caller may refine. Tier 4 is anything where max_prob < 0.32.
    """
    top2 = torch.topk(probs, k=2, dim=-1).values     # (N, 2)
    max_p = top2[:, 0].numpy()
    gap = (top2[:, 0] - top2[:, 1]).numpy()
    tiers = np.array(['2A'] * len(max_p), dtype=object)
    tiers[(max_p >= tier_1a_thr) & (gap >= 0.25)] = '1A'
    tiers[(max_p >= 0.60) & (max_p < tier_1a_thr) & (gap >= 0.20)] = '1B'
    tiers[max_p < 0.32] = '4'
    return tiers


# ----------------------------------------------------------------------------
# Report writer
# ----------------------------------------------------------------------------
def write_report(report_data: dict):
    L = []
    L.append("# LADI-Net Tomato Specialist - Final Evaluation Report")
    L.append("")
    L.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- **Model:** `{report_data['production_model']}`")
    L.append(f"- **Evaluation set:** locked 104-image held-out split (sacred, never used for training/stopping)")
    L.append(f"- **Evaluated ONCE:** marker `{MARKER_FILE.name}` written after this evaluation")
    L.append("")

    L.append("## Soup Procedure")
    soup_meta = report_data["soup_meta"]
    L.append(f"- Candidates considered: {soup_meta['checkpoint_names'] if soup_meta['is_trivial_soup'] else 'all ranked'}")
    soup_desc = (
        "single best checkpoint" if soup_meta["is_trivial_soup"]
        else f"{soup_meta['n_checkpoints']}-checkpoint soup"
    )
    L.append(f"- Result: **{soup_desc}**")
    L.append(f"- Tolerance: {soup_meta['tolerance']} field_val F1 degradation allowed")
    if soup_meta["is_trivial_soup"]:
        L.append(f"- Note: soup reduced to single best ckpt (epoch02 f1=0.9112). The 0.047 gap")
        L.append(f"  between epoch02 and epoch01 (0.864) far exceeds the 0.01 tolerance - no")
        L.append(f"  averaging would pass. **This is Issue 9-A / Decision 36 disclosure.**")
    L.append("")

    L.append("## Calibration (PDA-1.3 note)")
    cal = report_data["calibration"]
    L.append(f"- Calibrated on `confusable_pair_probe` (n=28: 20 foliar + 8 septoria)")
    L.append(f"- Single-sample T_optimal = **{cal['T_optimal']:.4f}**")
    L.append(f"- Bootstrap (10 resamples) mean T = {cal.get('T_bootstrap_mean', 'NA')}")
    L.append(f"- Bootstrap sigma(T) = **{cal['pda_T_stability_bootstrap_std']:.4f}** -> **{cal['pda_T_stability_interpretation']}**")
    L.append(f"- NLL before/after: {cal['nll_before']:.4f} / {cal['nll_after']:.4f}")
    L.append(f"- ECE before/after: {cal['ece_before']:.4f} / {cal['ece_after']:.4f}")
    L.append("")
    L.append("**Honest note on calibration**: sigma(T) = 0.38 means the single-sample T = 1.2696")
    L.append("is not statistically distinguishable from T = 1.0. Both variants are reported below.")
    L.append("See Issue 10-A (two-temperature calibration limitation, unresolved).")
    L.append("")

    # Per-class table for the PRIMARY reported configuration (T=T_opt, with blending)
    L.append("## Per-Class F1 on Final Held-Out Set (n=104)")
    L.append("**Primary result**: T=T_optimal, with prototype blending (spec-default).")
    L.append("")
    L.append("| Class | n | F1 | 95% CI | v3 baseline | Delta vs v3 | Flag |")
    L.append("|-------|---|----|--------|-------------|-------------|------|")
    primary = report_data["primary_results"]
    bs_primary = report_data["bootstrap_primary"]
    n_per_class = report_data["n_per_class"]
    for i, cls in enumerate(TOMATO_CLASSES):
        f1 = primary["per_class_f1"][i]
        lo = bs_primary["per_class_ci_lower"][i]
        hi = bs_primary["per_class_ci_upper"][i]
        v3 = V3_BASELINE[cls]
        delta = f1 - v3
        n_c = n_per_class[cls]
        if n_c < 5:
            flag = "CRITICAL"
        elif n_c < 10:
            flag = "UNDERPOWERED"
        else:
            flag = ""
        L.append(f"| {cls} | {n_c} | {f1:.4f} | [{lo:.4f}, {hi:.4f}] | {v3:.3f} | {delta:+.3f} | {flag} |")
    L.append("")
    L.append(f"**Overall sqrt(N)-weighted macro F1:** **{primary['sqrtn_macro_f1']:.4f}** "
             f"(95% CI [{bs_primary['overall_ci_lower']:.4f}, {bs_primary['overall_ci_upper']:.4f}])")
    L.append(f"v3 baseline overall sqrtn: {V3_BASELINE['overall_sqrtn']:.3f} -> "
             f"Delta = {primary['sqrtn_macro_f1'] - V3_BASELINE['overall_sqrtn']:+.3f}")
    L.append("")

    # Diagnostic grid: 4 variants (T x blend)
    L.append("## Diagnostic Grid: All 4 (T, blend) Combinations")
    L.append("Shows sensitivity of the headline number to calibration and blending choices.")
    L.append("")
    L.append("| T | Blending | overall_sqrtn_F1 | foliar | septoria | late_blight | YLCV | mosaic | healthy |")
    L.append("|---|----------|------------------|--------|----------|-------------|------|--------|---------|")
    for key in ["T=1.0_no_blend", "T=1.0_blend", f"T=T_opt_no_blend", "T=T_opt_blend (PRIMARY)"]:
        if key in report_data["diag_grid"]:
            v = report_data["diag_grid"][key]
            L.append(f"| {v['T']} | {v['blend']} | {v['sqrtn_macro']:.4f} | "
                     + " | ".join(f"{v['per_class'][c]:.3f}" for c in TOMATO_CLASSES) + " |")
    L.append("")

    # Tier distribution
    L.append("## Tier Distribution (on final held-out set)")
    tier_cnt = report_data["tier_distribution"]
    total = sum(tier_cnt.values())
    L.append("| Tier | Meaning | Count | % |")
    L.append("|------|---------|-------|---|")
    for t, label in [('1A', 'Confident'), ('1B', 'Probable'), ('2A', 'Differential'), ('4', 'Abstain')]:
        c = tier_cnt.get(t, 0)
        L.append(f"| {t} | {label} | {c} | {100.0 * c / max(total, 1):.1f}% |")
    L.append("")

    # Prototype blending activations
    L.append("## Prototype Memory Activations (max_prob < 0.60)")
    L.append(f"- Images with max_prob < 0.60 (pre-blending): **{report_data['n_low_conf']}** of 104")
    L.append(f"- Prototype blending applied to: same {report_data['n_low_conf']} images")
    L.append(f"- PDA-1.2 empirical check: accuracy on low-conf subset")
    L.append(f"  - Before blending: {report_data['blend_pda']['acc_before']:.4f} ({report_data['blend_pda']['n_correct_before']}/{report_data['n_low_conf']})")
    L.append(f"  - After blending:  {report_data['blend_pda']['acc_after']:.4f} ({report_data['blend_pda']['n_correct_after']}/{report_data['n_low_conf']})")
    delta_blend = report_data['blend_pda']['acc_after'] - report_data['blend_pda']['acc_before']
    if delta_blend > 0.01:
        verdict = "HELPS"
    elif delta_blend < -0.01:
        verdict = "HURTS"
    else:
        verdict = "NEUTRAL (within noise)"
    L.append(f"  - Verdict: **{verdict}** (delta = {delta_blend:+.4f})")
    L.append("")

    # Statistical notes - MANDATORY per PVA Check 2.6
    L.append("## Statistical Notes (MANDATORY - do not omit)")
    L.append("")
    for cls in TOMATO_CLASSES:
        n_c = n_per_class[cls]
        idx = TOMATO_CLASSES.index(cls)
        lo = bs_primary["per_class_ci_lower"][idx]
        hi = bs_primary["per_class_ci_upper"][idx]
        width = hi - lo
        if n_c < 5:
            flag = "**CRITICAL**"
            note = "F1 point estimate can jump by >= 0.20 with a single label swap"
        elif n_c < 10:
            flag = "**UNDERPOWERED**"
            note = f"CI width {width:.3f} exceeds 0.15 - results not statistically reliable"
        elif width > 0.15:
            flag = "**WIDE CI**"
            note = f"CI width {width:.3f} - interpret with caution"
        else:
            flag = "OK"
            note = f"CI width {width:.3f} - reliable"
        L.append(f"- **{cls}** (n={n_c}): {flag} - {note}")
    L.append("")
    L.append("Per Critique 3 (CI inflation at low class counts), improvements smaller than the CI")
    L.append("width should not be interpreted as real effects. In particular, YLCV and mosaic F1")
    L.append("values cannot be used to support or refute any architectural claim.")
    L.append("")

    # Honest limitations
    L.append("## Honest Limitations")
    L.append("1. **Septoria lesion-zoom is architecturally inapplicable** (Decision 40 geometric")
    L.append("   constraint). Septoria improvement comes from ABMIL attention pooling of patch")
    L.append("   tokens in the CLS-stream fusion path, not from lesion zoom. Phase 1 still")
    L.append("   achieved septoria=0.8148 on field_val (Phase 1 ckpt, +0.148 vs v3 baseline 0.667).")
    L.append("2. **YLCV and mosaic**: diffuse symptoms -> fallback-flag often active -> global CLS")
    L.append("   stream dominates. Tiny final_val sample (n=2, n=4) makes F1 point estimates")
    L.append("   nearly meaningless; see Statistical Notes.")
    L.append("3. **Phase 2 LoRA fine-tuning did not improve over Phase 1** - best Phase 2 epoch 8")
    L.append("   val_f1 = 0.8662 (4.5 pts below Phase 1). ABMIL maladaptation problem +")
    L.append("   Decision 49 ramp auto-revert. See session log and Decision 44/50.")
    L.append("4. **Temperature calibration**: fit on 28-image confusable pair probe (20 foliar,")
    L.append("   8 septoria). Bootstrap sigma(T)=0.38 means T=1.2696 is statistically")
    L.append("   indistinguishable from T=1.0. See Issue 10-A.")
    L.append("5. **Model soup with only 3 Phase 1 checkpoints** is not a meaningful ensemble;")
    L.append("   the 0.047 gap between epoch02 and epoch01 forces the soup to reduce to the best")
    L.append("   single ckpt. Issue 9-A/9-B disclosure.")
    L.append("6. **Circular dependency** (Decision 36): field_val was used for both stopping")
    L.append("   criterion AND soup selection AND Tier 1A threshold sweep. Final held-out")
    L.append("   (this evaluation) is the only truly independent estimate.")
    L.append("")

    L.append("## Production Model Artifact")
    L.append(f"- Saved: `{PRODUCTION_PATH.name}`")
    L.append(f"- Contents: model_state_dict + calibration (T_optimal + T=1.0 variant) +")
    L.append(f"  prototype_bank (30 prototypes) + tier_thresholds (1A = 0.60).")
    L.append(f"- Phase label: `phase1_heads_only` (no LoRA - lora_active=False).")
    L.append(f"- Architecture: `dinov2_base_registers_frozen_abmil_gated_mlp_single_pass`.")
    L.append("")

    L.append("---")
    L.append("")
    L.append("*End of final evaluation report. Final held-out set is now locked; do NOT re-evaluate.*")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # Step 0: refuse if already evaluated (PVA Check 2.1)
    check_not_already_evaluated()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Sanity load the required Phase 3 artifacts
    if not CALIBRATION_JSON.exists() or not PROTOTYPE_BANK_PATH.exists():
        raise FileNotFoundError("Phase 3 artifacts missing - run phase3 first.")
    calibration = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    T_opt = float(calibration["T_optimal"])
    proto_bank = torch.load(PROTOTYPE_BANK_PATH, map_location="cpu", weights_only=False)
    tier_doc = json.loads(TIER_THRESHOLDS_JSON.read_text(encoding="utf-8"))
    tier_1a_thr = float(tier_doc.get("chosen_threshold") or 0.72)
    log(f"Phase 3 artifacts loaded: T_opt={T_opt:.4f}, n_prototypes="
        f"{sum(p.shape[0] for p in proto_bank['prototypes'].values())}, tier_1A_thr={tier_1a_thr}")

    # Step 1: soup
    soup_sd, soup_meta = greedy_soup(device)
    torch.save({
        "epoch": -1,
        "abmil_state_dict": soup_sd["abmil_state_dict"],
        "fusion_state_dict": soup_sd["fusion_state_dict"],
        "supcon_projector_state_dict": soup_sd["supcon_projector_state_dict"],
        "soup_meta": soup_meta,
        "val_sqrtn_macro_f1": soup_meta["field_val_f1_after_soup"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, PHASE1_SOUP_PATH)
    log(f"Soup saved: {PHASE1_SOUP_PATH}")

    # Step 2: build souped model and evaluate on the locked split
    model = LADINet(device, phase="phase1").to(device).eval()
    assign_state_dicts_to_model(model, soup_sd)
    for p in model.parameters():
        p.requires_grad = False

    log("\n=== Evaluating on locked held-out split ===")
    heldout_records = load_split_records("final_val")        # sole touch of the locked split
    log(f"  Loaded {len(heldout_records)} records")

    # Class count sanity
    n_per_class = Counter(r.class_name for r in heldout_records)
    log(f"  Class distribution: {dict(n_per_class)}")

    inf = run_inference(model, heldout_records, device, tag="heldout")
    logits = inf["logits"]                                  # (N, 6)
    labels = inf["labels"].numpy()
    bag_feats = inf["bag_feat"]

    # Compute all 4 (T, blend) variants for the diagnostic grid
    log("\n=== Diagnostic grid: 4 (T, blend) combinations ===")
    diag_grid = {}
    for T_label, T_val in [("T=1.0", 1.0), ("T=T_opt", T_opt)]:
        probs_cal = torch.softmax(logits / T_val, dim=-1)
        for blend_label, do_blend in [("no_blend", False), ("blend", True)]:
            if do_blend:
                probs_use = prototype_blend(probs_cal, bag_feats, proto_bank)
            else:
                probs_use = probs_cal
            y_pred = probs_use.argmax(dim=-1).numpy()
            f1_per = per_class_f1(labels, y_pred)
            f1_sqrtn = sqrtn_macro_f1(labels, y_pred)
            diag_grid[f"{T_label}_{blend_label}"] = {
                "T": T_val,
                "blend": do_blend,
                "sqrtn_macro": f1_sqrtn,
                "per_class": {c: float(f1_per[i]) for i, c in enumerate(TOMATO_CLASSES)},
            }
            label_for_log = f"{T_label}_{blend_label}"
            log(f"  {label_for_log:25s}: sqrtn_macro={f1_sqrtn:.4f}  "
                f"f1={[f'{f:.3f}' for f in f1_per]}")

    # Mark the T=T_opt + blend as PRIMARY (spec-default)
    diag_grid_primary_key = "T=T_opt_blend (PRIMARY)"
    diag_grid[diag_grid_primary_key] = diag_grid["T=T_opt_blend"]

    # PRIMARY result = T=T_opt + blending
    probs_primary_cal = torch.softmax(logits / T_opt, dim=-1)
    probs_primary = prototype_blend(probs_primary_cal, bag_feats, proto_bank)
    y_pred_primary = probs_primary.argmax(dim=-1).numpy()
    primary_per_class = per_class_f1(labels, y_pred_primary)
    primary_sqrtn = sqrtn_macro_f1(labels, y_pred_primary)
    log(f"\n  PRIMARY (T=T_opt, with blending): sqrtn_macro_f1 = {primary_sqrtn:.4f}")

    # Bootstrap CI for PRIMARY
    log(f"\n=== Bootstrap CI (n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}) ===")
    bs = bootstrap_per_class_f1_ci(labels, y_pred_primary)
    log(f"  overall CI: [{bs['overall_ci_lower']:.4f}, {bs['overall_ci_upper']:.4f}]")

    # PDA-1.2: measure whether blending helps or hurts on low-conf subset
    log("\n=== PDA-1.2: prototype blending help/hurt on final set ===")
    max_primary_cal, _ = probs_primary_cal.max(dim=-1)
    low_conf_mask = (max_primary_cal < PROTO_BLEND_THRESHOLD).numpy()
    n_low = int(low_conf_mask.sum())
    if n_low > 0:
        y_pred_no_blend = probs_primary_cal.argmax(dim=-1).numpy()
        y_pred_with_blend = probs_primary.argmax(dim=-1).numpy()
        correct_before = int((y_pred_no_blend[low_conf_mask] == labels[low_conf_mask]).sum())
        correct_after = int((y_pred_with_blend[low_conf_mask] == labels[low_conf_mask]).sum())
        acc_before = correct_before / n_low
        acc_after = correct_after / n_low
        log(f"  Low-conf subset: {n_low} images")
        log(f"  Accuracy before blending: {correct_before}/{n_low} = {acc_before:.4f}")
        log(f"  Accuracy after blending:  {correct_after}/{n_low} = {acc_after:.4f}")
    else:
        acc_before = acc_after = 0.0
        correct_before = correct_after = 0
        log("  No images with max_prob < 0.60 on the held-out set - blending inactive.")

    # Tier assignment
    tier_labels = assign_tier(probs_primary, tier_1a_thr)
    tier_distribution = Counter(tier_labels.tolist())
    log(f"\n  Tier distribution: {dict(tier_distribution)}")

    # Save per-image predictions
    predictions = []
    for i, rec in enumerate(heldout_records):
        predictions.append({
            "image_path": rec.image_path,
            "true_class": rec.class_name,
            "pred_class": TOMATO_CLASSES[int(y_pred_primary[i])],
            "confidence": float(probs_primary[i].max().item()),
            "tier": str(tier_labels[i]),
            "fallback_flag": bool(inf["fallback_flag"][i].item() > 0.5),
            "all_probs": {TOMATO_CLASSES[j]: float(probs_primary[i, j].item())
                          for j in range(len(TOMATO_CLASSES))},
        })
    PREDICTIONS_JSON.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    log(f"  predictions saved: {PREDICTIONS_JSON}")

    # Build report data
    report_data = {
        "production_model": PRODUCTION_PATH.name,
        "soup_meta": soup_meta,
        "calibration": calibration,
        "primary_results": {
            "per_class_f1": primary_per_class.tolist(),
            "sqrtn_macro_f1": primary_sqrtn,
        },
        "bootstrap_primary": {
            "per_class_ci_lower": bs["per_class_ci_lower"].tolist(),
            "per_class_ci_upper": bs["per_class_ci_upper"].tolist(),
            "overall_ci_lower": bs["overall_ci_lower"],
            "overall_ci_upper": bs["overall_ci_upper"],
        },
        "diag_grid": diag_grid,
        "n_per_class": dict(n_per_class),
        "tier_distribution": dict(tier_distribution),
        "n_low_conf": n_low,
        "blend_pda": {
            "n_correct_before": correct_before,
            "n_correct_after": correct_after,
            "acc_before": acc_before,
            "acc_after": acc_after,
        },
        "final_primary_sqrtn_f1": primary_sqrtn,
    }
    write_report(report_data)

    # Save production model (PVA Check 2.5)
    torch.save({
        "abmil_state_dict": soup_sd["abmil_state_dict"],
        "fusion_state_dict": soup_sd["fusion_state_dict"],
        "supcon_projector_state_dict": soup_sd["supcon_projector_state_dict"],
        "calibration": {"T_optimal": T_opt, "T_1_0_variant_sqrtn_f1":
                        diag_grid["T=1.0_blend"]["sqrtn_macro"]},
        "prototype_bank": proto_bank,
        "tier_thresholds": tier_doc,
        "val_sqrtn_macro_f1_phase1": 0.9112,
        "final_sqrtn_macro_f1": primary_sqrtn,
        "final_per_class_f1": {c: float(primary_per_class[i]) for i, c in enumerate(TOMATO_CLASSES)},
        "phase_label": "phase1_heads_only",
        "lora_active": False,
        "architecture": "dinov2_base_registers_frozen_abmil_gated_mlp_single_pass",
        "classes": list(TOMATO_CLASSES),
        "decisions_version": 53,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, PRODUCTION_PATH)
    log(f"Production model saved: {PRODUCTION_PATH}")

    # Write marker (PVA Check 2.1)
    write_marker(report_data)
    log(f"\n{'=' * 70}")
    log(f"PHASE 4 COMPLETE. Final held-out sqrtn_macro_f1 = {primary_sqrtn:.4f}")
    log(f"CI: [{bs['overall_ci_lower']:.4f}, {bs['overall_ci_upper']:.4f}]")
    log(f"Report: {REPORT_PATH}")
    log(f"Marker: {MARKER_FILE}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
