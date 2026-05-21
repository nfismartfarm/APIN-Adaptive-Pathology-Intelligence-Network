"""
Task A.4 — Three-way field_val comparison (v3, single-pass LoRA, ensemble variants).

Purpose: empirical basis for the deployment architecture decision. Measures whether
ensembling helps, whether weighting matters, and whether prototype blending helps,
using field_val (203 images, tomato subset).

SIX CONFIGURATIONS:
  1. v3 alone                                  — tomato subset of 10-class output, T_v3 applied if use_calibration
  2. LoRA alone                                — T_sp_lora applied if use_calibration
  3. Ensemble 50/50, no blending               — primary candidate
  4. Ensemble 30/70 (LoRA-favored), no blend
  5. Ensemble 70/30 (v3-favored), no blend
  6. Ensemble 50/50 WITH LoRA-internal blend   — empirical test of Option ζ revisit

For each configuration: per-class F1 + sqrtn macro F1 on the tomato subset of field_val.

IMPORTANT METHODOLOGICAL NOTE:
  field_val is NOT pristine — it was used during training for stopping criterion,
  soup selection (v3), and tier threshold sweeps. Results here inform the deployment
  DECISION, not the deployment NUMBER. The held-out number for single-pass alone
  (A.3) is the pristine deliverable.

HARD CONSTRAINTS:
  - Zero modifications to any file under scripts/apin/ or scripts/apin_v2/.
  - Zero touches on the locked 104-image held-out split.
  - Uses artifacts from A.1 (prototype bank, LoRA calibration) and A.2 (v3 calibration).
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

from ladinet_config import TOMATO_CLASSES, CLASS_TO_IDX
from ladinet_dataloader import LadiRecord, LadiNetDataset, load_split_records
from single_pass_lora_train import SinglePassLoRA

# v3 imports
from scripts.model3_training.architecture.model3_full import Model3
from scripts.model3_training.model3_config import (
    CLASS_NAMES as V3_CLASSES, NUM_CLASSES as V3_N,
    IMAGENET_MEAN as V3_MEAN, IMAGENET_STD as V3_STD,
    CHECKPOINT_DIR as V3_CKPT_DIR, PRODUCTION_V3_CHECKPOINT_NAME as V3_FNAME,
    LORA_RANK,
)
from scripts.model3_training.data.preprocessing import apply_lab_clahe as v3_apply_clahe


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V3_CKPT = V3_CKPT_DIR / V3_FNAME
SP_LORA_CKPT = (PROJECT_ROOT / "models" / "specialist" / "sp_lora_checkpoints"
                / "sp_lora_epoch13_f10.9113_PRESERVED.pt")
OUT_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
SP_BANK = OUT_DIR / "prototype_bank_sp_lora_ep13.pt"
SP_CAL = OUT_DIR / "phase3_calibration_sp_lora_ep13.json"
V3_CAL = OUT_DIR / "phase3_calibration_v3_tomato.json"
REPORT_JSON = OUT_DIR / "a4_field_val_three_way.json"
REPORT_MD = PROJECT_ROOT / "a4_field_val_three_way_report.md"

# v3 tomato remap
V3_INDEX_FOR_LORA_CLASS = [0, 2, 1, 3, 4, 5]

# Blending
PROTO_BLEND_THRESHOLD = 0.60
PROTO_BLEND_WEIGHT = 0.35

V3_INPUT_RES = 224


def log(msg: str):
    print(f"[A.4] {msg}", flush=True)


# ---------------------------------------------------------------------------
# v3 preprocessing (same as calibrate_v3_tomato)
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
# Metrics
# ---------------------------------------------------------------------------
def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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


def sqrtn_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = per_class_f1(y_true, y_pred)
    present, weights = [], []
    for c in range(len(TOMATO_CLASSES)):
        n_c = int((y_true == c).sum())
        if n_c > 0:
            present.append(c)
            weights.append(float(np.sqrt(n_c)))
    if not present:
        return 0.0
    w = np.array(weights) / sum(weights)
    return float(sum(w[i] * f1s[c] for i, c in enumerate(present)))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_v3_tomato(
    model: Model3, records: list[LadiRecord], device: torch.device,
    T_applied: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run v3 on records, return 6-class tomato probs + labels.

    Returns:
        probs: [N, 6] tensor (softmax over remapped tomato logits)
        labels: [N] numpy int array
    """
    import cv2
    all_logits = []
    all_labels = []
    t0 = time.time()
    for i, rec in enumerate(records):
        img_bgr = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            from PIL import Image
            pil = Image.open(rec.image_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        x = _preprocess_v3(img_bgr).to(device)
        crop_mode = torch.tensor([2], dtype=torch.long, device=device)
        out = model(x, crop_mode, domain_labels=None)
        logits_full = out["logits"].float().squeeze(0).cpu()  # [10]
        tomato_logits = logits_full[V3_INDEX_FOR_LORA_CLASS]
        all_logits.append(tomato_logits)
        all_labels.append(CLASS_TO_IDX.get(rec.class_name, -1))
        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            log(f"  [v3] {i + 1}/{len(records)} in {time.time() - t0:.1f}s")

    logits = torch.stack(all_logits, dim=0)
    probs = torch.softmax(logits / T_applied, dim=-1)
    return probs, np.array(all_labels)


@torch.no_grad()
def infer_sp_lora(
    model: SinglePassLoRA, records: list[LadiRecord], device: torch.device,
    T_applied: float,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Run single-pass LoRA. Returns (probs, cls_feats, labels)."""
    ds = LadiNetDataset(records, training=False, background_pool=None)
    def _collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        }
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=_collate)
    all_logits, all_cls, all_labels = [], [], []
    t0 = time.time()
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"]
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(x)
        all_logits.append(out["logits"].float().cpu())
        all_cls.append(out["cls"].float().cpu())
        all_labels.append(y)
        n += x.size(0)
        if n % 64 == 0 or n == len(records):
            log(f"  [sp_lora] {n}/{len(records)} in {time.time() - t0:.1f}s")
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits / T_applied, dim=-1)
    cls = torch.cat(all_cls, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    return probs, cls, labels


def prototype_blend(probs: torch.Tensor, cls_feats: torch.Tensor, proto_bank: dict):
    max_p, _ = probs.max(dim=-1)
    mask = max_p < PROTO_BLEND_THRESHOLD
    if not mask.any():
        return probs, 0
    cls_norm = F.normalize(cls_feats, dim=-1)
    proto_tensors = [F.normalize(proto_bank["prototypes"][cls], dim=-1) for cls in TOMATO_CLASSES]
    proto = torch.stack(proto_tensors, dim=0)
    cos = torch.einsum("nd,ckd->nck", cls_norm, proto)
    proto_score = cos.max(dim=-1).values
    proto_probs = torch.softmax(proto_score * 5.0, dim=-1)
    blended = probs.clone()
    blended[mask] = (1.0 - PROTO_BLEND_WEIGHT) * probs[mask] + PROTO_BLEND_WEIGHT * proto_probs[mask]
    return blended, int(mask.sum().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # === Load artifacts ===
    if not SP_BANK.exists() or not SP_CAL.exists() or not V3_CAL.exists():
        raise FileNotFoundError("A.1 and/or A.2 artifacts missing — run them first.")

    sp_cal = json.loads(SP_CAL.read_text(encoding="utf-8"))
    sp_T = float(sp_cal["T_optimal"]) if sp_cal.get("use_calibration", True) else 1.0
    log(f"  sp_lora T_applied = {sp_T:.4f} (use_calibration={sp_cal.get('use_calibration')})")

    v3_cal = json.loads(V3_CAL.read_text(encoding="utf-8"))
    v3_T = float(v3_cal["T_optimal"]) if v3_cal.get("use_calibration", True) else 1.0
    log(f"  v3 T_applied = {v3_T:.4f} (use_calibration={v3_cal.get('use_calibration')})")

    proto_bank = torch.load(SP_BANK, map_location="cpu", weights_only=False)
    assert proto_bank.get("feature_space") == "CLS_token_768d"
    log(f"  prototype bank loaded (CLS_token_768d)")

    # === Load v3 ===
    log("\n=== Loading v3 ===")
    model_v3 = Model3(n_classes=V3_N, pretrained=False, lora_rank=LORA_RANK).to(device)
    ckpt_v3 = torch.load(V3_CKPT, map_location=device, weights_only=False)
    model_v3.load_state_dict(ckpt_v3["model_state_dict"])
    model_v3.eval()
    for p in model_v3.parameters():
        p.requires_grad = False

    # === Load single-pass LoRA ===
    log("\n=== Loading single-pass LoRA ===")
    model_sp = SinglePassLoRA(device).to(device)
    ckpt_sp = torch.load(SP_LORA_CKPT, map_location=device, weights_only=False)
    model_sp.load_state_dict(ckpt_sp["model_state_dict"])
    model_sp.eval()
    for p in model_sp.parameters():
        p.requires_grad = False

    # === Load field_val tomato subset ===
    log("\n=== Loading field_val tomato subset ===")
    fv_all = load_split_records("field_val")
    fv = [r for r in fv_all if r.class_name.startswith("tomato_")]
    log(f"  field_val tomato subset: {len(fv)} / {len(fv_all)} total")
    from collections import Counter
    n_per_class = dict(Counter(r.class_name for r in fv))
    log(f"  class distribution: {n_per_class}")

    # === Inference ===
    log("\n=== Running v3 on tomato subset ===")
    v3_probs, labels_v3 = infer_v3_tomato(model_v3, fv, device, T_applied=v3_T)
    log("\n=== Running single-pass LoRA on tomato subset ===")
    sp_probs, sp_cls, labels_sp = infer_sp_lora(model_sp, fv, device, T_applied=sp_T)

    # Sanity: labels should match between the two paths (same records, same order)
    assert (labels_v3 == labels_sp).all(), "label mismatch between v3 and sp_lora inference — order drift"
    labels = labels_v3  # unified

    # === Build 6 configurations ===
    log("\n=== Building 6 configurations ===")
    configs = {}

    # Config 1: v3 alone
    y_pred_v3 = v3_probs.argmax(dim=-1).numpy()
    configs["v3_alone"] = {
        "label": "v3 alone (tomato subset, T_v3 applied)",
        "probs": v3_probs,
        "y_pred": y_pred_v3,
    }

    # Config 2: LoRA alone
    y_pred_sp = sp_probs.argmax(dim=-1).numpy()
    configs["sp_lora_alone"] = {
        "label": "single-pass LoRA alone",
        "probs": sp_probs,
        "y_pred": y_pred_sp,
    }

    # Config 3: Ensemble 50/50 no-blend
    ens_50 = 0.5 * v3_probs + 0.5 * sp_probs
    y_pred_50 = ens_50.argmax(dim=-1).numpy()
    configs["ensemble_50_50"] = {
        "label": "Ensemble 50/50 (no blending)",
        "probs": ens_50,
        "y_pred": y_pred_50,
    }

    # Config 4: Ensemble 30/70 (LoRA-favored)
    ens_30_70 = 0.30 * v3_probs + 0.70 * sp_probs
    y_pred_30_70 = ens_30_70.argmax(dim=-1).numpy()
    configs["ensemble_30_70_lora_favored"] = {
        "label": "Ensemble 30/70 (LoRA-favored, no blending)",
        "probs": ens_30_70,
        "y_pred": y_pred_30_70,
    }

    # Config 5: Ensemble 70/30 (v3-favored)
    ens_70_30 = 0.70 * v3_probs + 0.30 * sp_probs
    y_pred_70_30 = ens_70_30.argmax(dim=-1).numpy()
    configs["ensemble_70_30_v3_favored"] = {
        "label": "Ensemble 70/30 (v3-favored, no blending)",
        "probs": ens_70_30,
        "y_pred": y_pred_70_30,
    }

    # Config 6: Ensemble 50/50 with LoRA-internal blend (Option α on the LoRA side)
    sp_probs_blended, n_blended = prototype_blend(sp_probs, sp_cls, proto_bank)
    ens_50_blend = 0.5 * v3_probs + 0.5 * sp_probs_blended
    y_pred_50_blend = ens_50_blend.argmax(dim=-1).numpy()
    configs["ensemble_50_50_with_blend"] = {
        "label": "Ensemble 50/50 with LoRA-internal prototype blending",
        "probs": ens_50_blend,
        "y_pred": y_pred_50_blend,
        "n_blended": n_blended,
    }

    # === Compute metrics for each ===
    log("\n=== Results ===")
    rows = []
    for key, cfg in configs.items():
        y_pred = cfg["y_pred"]
        pc = per_class_f1(labels, y_pred)
        sqrtn = sqrtn_macro(labels, y_pred)
        acc = float((y_pred == labels).mean())
        n_correct = int((y_pred == labels).sum())
        row = {
            "config": key,
            "label": cfg["label"],
            "sqrtn_macro_f1": sqrtn,
            "accuracy": acc,
            "n_correct": n_correct,
            "per_class_f1": {cls: float(pc[i]) for i, cls in enumerate(TOMATO_CLASSES)},
        }
        if "n_blended" in cfg:
            row["n_blended"] = cfg["n_blended"]
        rows.append(row)
        log(f"  {key:35s}: sqrtn={sqrtn:.4f} acc={acc:.4f} ({n_correct}/{len(labels)})")

    # === Save JSON ===
    out_data = {
        "n_images": len(fv),
        "n_per_class": n_per_class,
        "T_v3_applied": v3_T,
        "T_sp_lora_applied": sp_T,
        "configurations": rows,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "methodological_note": (
            "field_val is NOT pristine — it was used as stopping criterion for single-pass "
            "LoRA training, for v3 soup selection, and for tier threshold sweeps. These "
            "numbers inform DEPLOYMENT ARCHITECTURE decision, not deployment performance."
        ),
    }
    REPORT_JSON.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    log(f"  saved: {REPORT_JSON.name}")

    # === Report markdown ===
    L = []
    L.append("# Task A.4 — Three-Way field_val Comparison Report")
    L.append("")
    L.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- **field_val tomato subset:** {len(fv)} images")
    L.append(f"- **T_v3 applied:** {v3_T:.4f}")
    L.append(f"- **T_sp_lora applied:** {sp_T:.4f}")
    L.append("")
    L.append("## field_val caveat")
    L.append("")
    L.append("field_val is NOT pristine. It was used for:")
    L.append("- single-pass LoRA stopping criterion (training phase)")
    L.append("- v3 soup selection (Round-B training)")
    L.append("- tier threshold sweeps in both A.1 and A.2")
    L.append("")
    L.append("These results inform the **deployment architecture decision**, not the deployment "
             "performance number. The pristine deliverable number is single-pass LoRA's held-out "
             "from A.3 (see `ladi_final_evaluation_report_sp_lora.md`).")
    L.append("")
    L.append("## Class distribution on tomato subset")
    L.append("")
    for cls, n in n_per_class.items():
        L.append(f"- {cls}: n={n}")
    L.append("")
    L.append("## Six configurations — headline comparison")
    L.append("")
    L.append("| Config | sqrtn macro F1 | Accuracy | foliar | septoria | late_blight | YLCV | mosaic | healthy |")
    L.append("|--------|----------------|----------|--------|----------|-------------|------|--------|---------|")
    for row in rows:
        pc = row["per_class_f1"]
        L.append(
            f"| {row['config']} | **{row['sqrtn_macro_f1']:.4f}** | "
            f"{row['accuracy']:.4f} | "
            f"{pc['tomato_foliar_spot']:.3f} | "
            f"{pc['tomato_septoria_leaf_spot']:.3f} | "
            f"{pc['tomato_late_blight']:.3f} | "
            f"{pc['tomato_yellow_leaf_curl_virus']:.3f} | "
            f"{pc['tomato_mosaic_virus']:.3f} | "
            f"{pc['tomato_healthy']:.3f} |"
        )
    L.append("")

    # Identify the best config and interpretation
    best = max(rows, key=lambda r: r["sqrtn_macro_f1"])
    worst = min(rows, key=lambda r: r["sqrtn_macro_f1"])
    ens_50 = next(r for r in rows if r["config"] == "ensemble_50_50")
    ens_50_blend = next(r for r in rows if r["config"] == "ensemble_50_50_with_blend")
    v3_alone = next(r for r in rows if r["config"] == "v3_alone")
    sp_alone = next(r for r in rows if r["config"] == "sp_lora_alone")
    better_of_solo = v3_alone if v3_alone["sqrtn_macro_f1"] > sp_alone["sqrtn_macro_f1"] else sp_alone

    L.append("## Interpretation")
    L.append("")
    L.append(f"- **Best config on field_val**: `{best['config']}` at {best['sqrtn_macro_f1']:.4f}")
    L.append(f"- **Worst config on field_val**: `{worst['config']}` at {worst['sqrtn_macro_f1']:.4f}")
    L.append(f"- **v3 alone**: {v3_alone['sqrtn_macro_f1']:.4f}")
    L.append(f"- **single-pass LoRA alone**: {sp_alone['sqrtn_macro_f1']:.4f}")
    L.append(f"- **Ensemble 50/50 (primary candidate)**: {ens_50['sqrtn_macro_f1']:.4f}")
    L.append(f"- **Ensemble 50/50 + LoRA-internal blend**: {ens_50_blend['sqrtn_macro_f1']:.4f} "
             f"(n_blended = {ens_50_blend.get('n_blended', '—')})")
    L.append("")

    delta_ens_vs_best_solo = ens_50["sqrtn_macro_f1"] - better_of_solo["sqrtn_macro_f1"]
    if delta_ens_vs_best_solo >= 0.005:
        ens_verdict = (
            f"Ensemble 50/50 beats the better individual model ({better_of_solo['config']}) "
            f"by +{delta_ens_vs_best_solo:.4f}. Ensembling is empirically justified."
        )
    elif delta_ens_vs_best_solo <= -0.005:
        ens_verdict = (
            f"Ensemble 50/50 is WORSE than the better individual model ({better_of_solo['config']}) "
            f"by {delta_ens_vs_best_solo:.4f}. Consider deploying the individual instead; "
            f"this warrants DEVELOPER ATTENTION."
        )
    else:
        ens_verdict = (
            f"Ensemble 50/50 is within ±0.005 of the better individual model "
            f"({better_of_solo['config']}, Δ={delta_ens_vs_best_solo:+.4f}). "
            f"Ensembling is neutral on field_val but likely provides error diversity."
        )
    L.append(f"**Ensembling verdict**: {ens_verdict}")
    L.append("")

    blend_delta = ens_50_blend["sqrtn_macro_f1"] - ens_50["sqrtn_macro_f1"]
    n_blended_val = ens_50_blend.get("n_blended", 0)
    if n_blended_val == 0:
        blend_verdict = "Blending activated on zero images (no max_prob < 0.60 cases on field_val)."
    elif blend_delta >= 0.005:
        blend_verdict = (
            f"Blending improved the 50/50 ensemble by +{blend_delta:.4f} "
            f"(blending fired on {n_blended_val} images). Revisit Option ζ: "
            f"prototype blending may deserve reinstatement in production."
        )
    elif blend_delta <= -0.005:
        blend_verdict = (
            f"Blending HURT the 50/50 ensemble by {blend_delta:.4f} "
            f"(blending fired on {n_blended_val} images). Option ζ validated: keep blending OFF."
        )
    else:
        blend_verdict = (
            f"Blending effect within ±0.005 (Δ={blend_delta:+.4f}, "
            f"fired on {n_blended_val} images). Option ζ stands — skip blending in production."
        )
    L.append(f"**Blending verdict**: {blend_verdict}")
    L.append("")

    # Weight sweep interpretation
    w50 = ens_50["sqrtn_macro_f1"]
    w30_70 = next(r for r in rows if r["config"] == "ensemble_30_70_lora_favored")["sqrtn_macro_f1"]
    w70_30 = next(r for r in rows if r["config"] == "ensemble_70_30_v3_favored")["sqrtn_macro_f1"]
    spread = max(w50, w30_70, w70_30) - min(w50, w30_70, w70_30)
    if spread < 0.005:
        weight_verdict = (
            f"Weight choice does not meaningfully affect ensemble performance "
            f"(spread={spread:.4f}). Use 50/50 for simplicity and defensibility."
        )
    else:
        best_w = max([("50/50", w50), ("30/70", w30_70), ("70/30", w70_30)], key=lambda x: x[1])
        weight_verdict = (
            f"Weight choice matters (spread={spread:.4f}). Best on field_val is {best_w[0]} "
            f"at {best_w[1]:.4f}. RECOMMENDATION: deploy 50/50 anyway to avoid field_val-based "
            f"weight tuning (which would further inflate field_val numbers). Document the "
            f"trade-off."
        )
    L.append(f"**Weight sweep verdict**: {weight_verdict}")
    L.append("")

    L.append("## Deployment recommendation")
    L.append("")
    if delta_ens_vs_best_solo <= -0.005:
        L.append(f"Deploy `{better_of_solo['config']}` as the tomato specialist (single model, "
                 f"not ensemble). The ensemble shows negative synergy on field_val.")
    elif delta_ens_vs_best_solo >= 0.005:
        L.append(f"Deploy the 50/50 ensemble. Weight sweep and blending comparison do not "
                 f"support more complex alternatives.")
    else:
        L.append(f"Deploy the 50/50 ensemble. Even though ensemble is within noise of best "
                 f"individual on field_val, ensembling provides error diversity and the "
                 f"deployment complexity cost is minimal.")
    L.append("")
    L.append("Production pipeline specifics:")
    L.append(f"- T_v3 = {v3_T:.4f} applied")
    L.append(f"- T_sp_lora = {sp_T:.4f} applied")
    L.append(f"- Prototype blending: {'disabled by default (Option ζ)' if blend_delta < 0.005 else 'empirically validated'}")
    L.append("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    log(f"  report saved: {REPORT_MD.name}")

    log("\n" + "=" * 64)
    log("TASK A.4 COMPLETE")
    log("=" * 64)
    log(f"  best config on field_val : {best['config']} @ {best['sqrtn_macro_f1']:.4f}")
    log(f"  v3 alone                 : {v3_alone['sqrtn_macro_f1']:.4f}")
    log(f"  sp_lora alone            : {sp_alone['sqrtn_macro_f1']:.4f}")
    log(f"  ensemble 50/50 no-blend  : {ens_50['sqrtn_macro_f1']:.4f}")
    log(f"  ensemble 50/50 with-blend: {ens_50_blend['sqrtn_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
