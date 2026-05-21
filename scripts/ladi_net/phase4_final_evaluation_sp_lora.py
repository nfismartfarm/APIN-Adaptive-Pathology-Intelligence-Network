"""
Task A.3 — One-time final held-out evaluation of single-pass LoRA (epoch 13 preserved).

AUTHORIZATION: This evaluation is the second touch of the locked 104-image final_val set
in this session. Decision 54 addendum (ladi_decisions.md, appended 2026-04-23 23:22) authorizes
re-evaluation under the revised gate of field_val sqrtn_macro_f1 >= 0.88. Single-pass LoRA
epoch 13 has field_val sqrtn_macro_f1 = 0.9113, well above the gate.

Before running, this script:
  1. Verifies Decision 54 addendum text is present in ladi_decisions.md (text match check).
  2. Verifies single-pass LoRA epoch 13 checkpoint val_sqrtn_macro_f1 >= 0.88.
  3. Reads the existing marker file content (for the record) and rewrites it with the
     justification text tying this re-evaluation to Decision 54 addendum.

The evaluation:
  - Loads sp_lora_epoch13_f10.9113_PRESERVED.pt.
  - Loads A.1's artifacts: prototype_bank_sp_lora_ep13.pt + phase3_calibration_sp_lora_ep13.json
    + phase3_tier_thresholds_sp_lora_ep13.json.
  - Runs inference on 104 held-out images.
  - Reports TWO variants for empirical transparency:
      (a) PRIMARY: T_applied softmax (T=T_opt if use_calibration else 1.0), NO prototype blending.
          This is the deployment-intent configuration under Option ζ.
      (b) DIAGNOSTIC: same T, WITH prototype blending when max_prob < 0.60.
          Kept to empirically test whether blending would help on held-out.
  - Bootstrap CI with replacement, n=10000, seed=42, per-class + overall.
  - CRITICAL / UNDERPOWERED flags for small-n classes (YLCV n=2, mosaic n=4).
  - Comparison to v3 baseline (0.853) and Phase 1 (0.7958) as reference points.

DEPLOYMENT ARTIFACT: produces `tomato_sp_lora_production.pt` (no "ladinet" prefix per
2026-04-24 correction) containing single-pass weights + calibration + prototype bank +
tier thresholds, for use by the tomato inference pipeline.

HARD CONSTRAINTS:
  - Zero modifications to any file under scripts/apin/ or scripts/apin_v2/.
  - EXACTLY ONE inference pass over final_val records (the marker protects future re-runs).
  - v3 checkpoint is NOT touched here (A.2 / A.4 handle v3).
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
    PROJECT_ROOT, TOMATO_CLASSES, CLASS_TO_IDX,
)
from ladinet_dataloader import LadiRecord, LadiNetDataset, load_split_records
from single_pass_lora_train import SinglePassLoRA


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SP_LORA_CKPT = (PROJECT_ROOT / "models" / "specialist" / "sp_lora_checkpoints"
                / "sp_lora_epoch13_f10.9113_PRESERVED.pt")
OUT_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
PROTO_BANK_PATH = OUT_DIR / "prototype_bank_sp_lora_ep13.pt"
CALIBRATION_PATH = OUT_DIR / "phase3_calibration_sp_lora_ep13.json"
TIER_PATH = OUT_DIR / "phase3_tier_thresholds_sp_lora_ep13.json"

MARKER_FILE = OUT_DIR / "final_val_evaluated.txt"
PREDICTIONS_JSON = OUT_DIR / "final_val_predictions_sp_lora.json"
REPORT_PATH = PROJECT_ROOT / "ladi_final_evaluation_report_sp_lora.md"
PRODUCTION_PATH = PROJECT_ROOT / "models" / "specialist" / "tomato_sp_lora_production.pt"

DECISIONS_PATH = PROJECT_ROOT / "ladi_decisions.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
PROTO_BLEND_THRESHOLD = 0.60
PROTO_BLEND_WEIGHT = 0.35   # blended = 0.65 * primary + 0.35 * proto
GATE_FIELD_VAL_MIN = 0.88   # Decision 54 addendum

V3_BASELINE_HELDOUT = 0.853
PHASE1_HELDOUT = 0.7958

# ---------------------------------------------------------------------------
# Logging + safety guards
# ---------------------------------------------------------------------------
def log(msg: str):
    print(f"[A.3] {msg}", flush=True)


def verify_decision54_addendum() -> None:
    """Check ladi_decisions.md contains the Decision 54 addendum text referencing 0.88 gate."""
    if not DECISIONS_PATH.exists():
        raise RuntimeError(f"{DECISIONS_PATH} missing — cannot verify Decision 54 addendum.")
    text = DECISIONS_PATH.read_text(encoding="utf-8")
    required_markers = [
        "Decision 54",  # section ref
        "0.88",          # gate value
        "field_val sqrtn_macro_f1",  # what the gate gates on
    ]
    for m in required_markers:
        if m not in text:
            raise RuntimeError(
                f"Decision 54 addendum verification failed: '{m}' not found in "
                f"{DECISIONS_PATH.name}. Refusing to touch held-out."
            )
    log("  Decision 54 addendum verified — gate 0.88 on field_val sqrtn_macro_f1 is logged.")


def verify_sp_lora_cleared_gate(ckpt) -> None:
    """Verify single-pass LoRA preserved checkpoint exceeds the 0.88 gate."""
    val_f1 = float(ckpt.get("val_sqrtn_macro_f1", -1.0))
    if val_f1 < GATE_FIELD_VAL_MIN:
        raise RuntimeError(
            f"Single-pass LoRA val_f1 = {val_f1:.4f} < gate {GATE_FIELD_VAL_MIN}. "
            f"Gate check failed — refusing to touch held-out."
        )
    log(f"  Gate cleared: single-pass val_f1={val_f1:.4f} >= {GATE_FIELD_VAL_MIN}")


def read_and_amend_marker() -> str:
    """Read the existing marker (for the record), then overwrite with the Decision 54
    justification. Returns the OLD content so we can log it.
    """
    old_content = ""
    if MARKER_FILE.exists():
        old_content = MARKER_FILE.read_text(encoding="utf-8")
        log(f"  Existing marker content:\n{old_content}")
    justification = (
        f"Final held-out set re-evaluated for single-pass LoRA epoch 13.\n"
        f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"Prior marker content (preserved here for audit):\n"
        f"--- BEGIN PRIOR MARKER ---\n{old_content}--- END PRIOR MARKER ---\n\n"
        f"Justification for re-evaluation:\n"
        f"Per Decision 54 addendum (ladi_decisions.md, appended 2026-04-23 23:22), "
        f"single-pass LoRA is authorized for held-out re-evaluation under revised gate "
        f"field_val sqrtn_macro_f1 >= 0.88. Single-pass LoRA epoch 13 has field_val "
        f"sqrtn_macro_f1 = 0.9113, clearing the gate by +0.0313.\n\n"
        f"This is the SECOND and FINAL touch of final_val in this session. The first "
        f"touch evaluated LADI-Net Phase 1 (result: sqrtn_macro_f1 = 0.7958). LADI-Net "
        f"Phase 1 is NOT in the deployed tomato path — single-pass LoRA is one of the "
        f"two deployed models (the other is v3).\n\n"
        f"DO NOT RUN AGAIN. Any third held-out touch requires a new pre-committed gate "
        f"decision, not just marker deletion."
    )
    MARKER_FILE.write_text(justification, encoding="utf-8")
    log(f"  Marker rewritten with Decision 54 justification.")
    return old_content


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, records, device, tag="heldout") -> dict:
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
    log(f"  [{tag}] {n} images in {time.time() - t0:.1f}s")
    return {
        "logits": torch.cat(all_logits, dim=0),
        "cls": torch.cat(all_cls, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


def prototype_blend(primary_probs: torch.Tensor, cls_feats: torch.Tensor,
                    proto_bank: dict) -> torch.Tensor:
    """Apply prototype blending to rows with max_prob < PROTO_BLEND_THRESHOLD."""
    N, C = primary_probs.shape
    max_p, _ = primary_probs.max(dim=-1)
    mask = max_p < PROTO_BLEND_THRESHOLD
    if not mask.any():
        return primary_probs, 0
    cls_norm = F.normalize(cls_feats, dim=-1)
    proto_tensors = [F.normalize(proto_bank["prototypes"][cls], dim=-1) for cls in TOMATO_CLASSES]
    proto = torch.stack(proto_tensors, dim=0)  # [C, k, 768]
    cos = torch.einsum("nd,ckd->nck", cls_norm, proto)
    proto_score = cos.max(dim=-1).values  # [N, C]
    proto_probs = torch.softmax(proto_score * 5.0, dim=-1)
    blended = primary_probs.clone()
    blended[mask] = (1.0 - PROTO_BLEND_WEIGHT) * primary_probs[mask] + PROTO_BLEND_WEIGHT * proto_probs[mask]
    return blended, int(mask.sum().item())


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


def bootstrap_ci(y_true, y_pred, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.zeros((n_boot, len(TOMATO_CLASSES)), dtype=np.float64)
    overall = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # WITH replacement
        yt, yp = y_true[idx], y_pred[idx]
        boots[i] = per_class_f1(yt, yp)
        overall[i] = sqrtn_macro(yt, yp)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return {
        "per_class_lo": np.percentile(boots, lo, axis=0),
        "per_class_hi": np.percentile(boots, hi, axis=0),
        "overall_lo": float(np.percentile(overall, lo)),
        "overall_hi": float(np.percentile(overall, hi)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def class_flag(n: int, ci_width: float) -> str:
    if n < 5:
        return "CRITICAL"
    if n < 10:
        return "UNDERPOWERED"
    if ci_width > 0.15:
        return "WIDE CI"
    return "OK"


def write_report(data: dict):
    L = []
    L.append("# Single-Pass LoRA — Final Held-Out Evaluation Report")
    L.append("")
    L.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- **Model:** single-pass LoRA epoch 13 (`sp_lora_epoch13_f10.9113_PRESERVED.pt`)")
    L.append(f"- **Gate:** field_val sqrtn_macro_f1 >= 0.88 (Decision 54 addendum)")
    L.append(f"- **Clearance:** {data['val_f1_preserved']:.4f} >= 0.88 ✓")
    L.append(f"- **Held-out set:** locked 104-image final_val, evaluated once this session")
    L.append(f"- **Prior held-out touch:** LADI-Net Phase 1 (result 0.7958) — separate artifact")
    L.append("")

    L.append("## Section 1 — Held-out evaluation under pre-committed gate (PRIMARY)")
    L.append("")
    L.append("Configuration: single-pass LoRA alone. T_applied={0:.4f}, prototype blending DISABLED "
             "(per Option ζ post-Phase-4 decision).".format(data["T_applied"]))
    L.append("")
    L.append("| Class | n | F1 | 95% CI | v3 baseline | Δ | Flag |")
    L.append("|-------|---|----|--------|-------------|---|------|")
    for i, cls in enumerate(TOMATO_CLASSES):
        n_c = data["n_per_class"][cls]
        f1 = data["primary"]["per_class_f1"][i]
        lo = data["primary"]["ci"]["per_class_lo"][i]
        hi = data["primary"]["ci"]["per_class_hi"][i]
        v3 = data["v3_per_class_baseline"].get(cls, None)
        delta = (f1 - v3) if v3 is not None else None
        flag = class_flag(n_c, hi - lo)
        v3_str = f"{v3:.3f}" if v3 is not None else "—"
        d_str = f"{delta:+.3f}" if delta is not None else "—"
        L.append(f"| {cls} | {n_c} | {f1:.4f} | [{lo:.4f}, {hi:.4f}] | {v3_str} | {d_str} | {flag} |")
    L.append("")
    L.append(f"**Overall sqrt(N)-weighted macro F1:** **{data['primary']['sqrtn']:.4f}** "
             f"(95% CI [{data['primary']['ci']['overall_lo']:.4f}, "
             f"{data['primary']['ci']['overall_hi']:.4f}])")
    L.append(f"")
    L.append(f"- v3 final_val baseline: {V3_BASELINE_HELDOUT:.3f} → Δ = "
             f"{data['primary']['sqrtn'] - V3_BASELINE_HELDOUT:+.3f}")
    L.append(f"- LADI-Net Phase 1 held-out: {PHASE1_HELDOUT:.4f} → Δ = "
             f"{data['primary']['sqrtn'] - PHASE1_HELDOUT:+.4f}")
    L.append(f"- Raw accuracy: {data['primary']['accuracy']:.4f} "
             f"({data['primary']['n_correct']}/{data['n_heldout']})")
    L.append("")

    L.append("## Section 2 — Diagnostic comparison: with vs without prototype blending")
    L.append("")
    L.append("Same model, same images, only difference is whether prototype blending fires for "
             "predictions with max_prob < 0.60.")
    L.append("")
    L.append("| Variant | overall sqrtn F1 | # images blended | same final predictions? |")
    L.append("|---------|------------------|-------------------|--------------------------|")
    L.append(f"| No blending (PRIMARY) | **{data['primary']['sqrtn']:.4f}** | 0 | — |")
    L.append(f"| With blending (DIAGNOSTIC) | {data['blended']['sqrtn']:.4f} | "
             f"{data['blended']['n_blended']} | {data['prediction_delta']} |")
    L.append("")
    L.append(f"Interpretation: {data['blend_interpretation']}")
    L.append("")

    L.append("## Section 3 — Statistical notes (mandatory)")
    L.append("")
    for cls in TOMATO_CLASSES:
        idx = TOMATO_CLASSES.index(cls)
        n_c = data["n_per_class"][cls]
        width = data["primary"]["ci"]["per_class_hi"][idx] - data["primary"]["ci"]["per_class_lo"][idx]
        flag = class_flag(n_c, width)
        if flag == "CRITICAL":
            note = "F1 point estimate can jump by >= 0.20 with a single label swap"
        elif flag == "UNDERPOWERED":
            note = f"CI width {width:.3f} exceeds 0.15 - results not statistically reliable"
        elif flag == "WIDE CI":
            note = f"CI width {width:.3f} - interpret with caution"
        else:
            note = f"CI width {width:.3f} - reliable"
        L.append(f"- **{cls}** (n={n_c}): **{flag}** — {note}")
    L.append("")
    L.append("Improvements smaller than the CI width should NOT be interpreted as real effects. "
             "In particular, YLCV and mosaic F1 values cannot on their own support or refute "
             "architectural claims.")
    L.append("")

    L.append("## Section 4 — Production architecture note")
    L.append("")
    L.append("The deployed tomato system is an ENSEMBLE of v3 + single-pass LoRA (see Task A.4 "
             "for the comparative field_val evidence supporting this choice). This report's "
             f"{data['primary']['sqrtn']:.4f} is the held-out number for single-pass LoRA ALONE, "
             "serving as a lower-bound on ensemble held-out performance under the assumption "
             "'ensembling rarely decreases performance relative to the best individual model.' "
             "Held-out measurement for the ensemble does not exist (would violate the one-gate "
             "policy).")
    L.append("")

    L.append("## Section 5 — Honest limitations")
    L.append("")
    L.append("1. **Small-n classes**: YLCV (n=2) and mosaic (n=4) produce unreliable F1 estimates.")
    L.append("2. **Temperature calibration**: fit on 28-image probe; bootstrap σ(T) may classify "
             "as UNRELIABLE with T=1.0 used at inference.")
    L.append("3. **Prototype blending disabled**: Phase 4 PDA-1.2 showed NEUTRAL effect; "
             "Option ζ disabled it by default. Section 2 verifies this on held-out.")
    L.append("4. **Ensemble unmeasured on held-out**: the production pipeline ensembles two "
             "models; only single-pass-alone has pristine held-out measurement.")
    L.append("")
    L.append("---")
    L.append("*End of A.3 held-out evaluation report.*")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log(f"  Report written: {REPORT_PATH.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # === Pre-flight: verify Decision 54 addendum ===
    log("\n=== Pre-flight: verifying authorization ===")
    verify_decision54_addendum()

    # === Load artifacts ===
    if not SP_LORA_CKPT.exists():
        raise FileNotFoundError(f"SP-LoRA ckpt not found: {SP_LORA_CKPT}")
    if not PROTO_BANK_PATH.exists() or not CALIBRATION_PATH.exists() or not TIER_PATH.exists():
        raise FileNotFoundError("A.1 artifacts missing — run Task A.1 first.")

    ckpt = torch.load(SP_LORA_CKPT, map_location=device, weights_only=False)
    verify_sp_lora_cleared_gate(ckpt)

    proto_bank = torch.load(PROTO_BANK_PATH, map_location="cpu", weights_only=False)
    assert proto_bank.get("feature_space") == "CLS_token_768d", \
        f"Proto bank feature_space mismatch: {proto_bank.get('feature_space')}"
    log(f"  Proto bank loaded: {sum(p.shape[0] for p in proto_bank['prototypes'].values())} "
        f"prototypes, feature_space={proto_bank['feature_space']}")

    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    T_opt = float(calibration["T_optimal"])
    use_cal = bool(calibration.get("use_calibration", True))
    T_applied = T_opt if use_cal else 1.0
    log(f"  T_optimal={T_opt:.4f}, use_calibration={use_cal}, T_applied={T_applied:.4f}")

    tier_doc = json.loads(TIER_PATH.read_text(encoding="utf-8"))
    tier_1a_thr = float(tier_doc.get("chosen_threshold") or 0.72)
    log(f"  Tier 1A threshold (from A.1): {tier_1a_thr}")

    # === Rewrite marker (now we're committed to evaluating) ===
    log("\n=== Amending final_val_evaluated.txt marker ===")
    old_marker = read_and_amend_marker()

    # === Build model ===
    model = SinglePassLoRA(device).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log(f"  single-pass LoRA loaded and frozen.")

    # === Load held-out records ===
    log("\n=== Loading locked 104-image held-out set ===")
    heldout = load_split_records("final_val")
    log(f"  n_heldout = {len(heldout)}")
    from collections import Counter
    n_per_class = dict(Counter(r.class_name for r in heldout))
    log(f"  class distribution: {n_per_class}")

    # === Single forward pass ===
    inf = run_inference(model, heldout, device, tag="heldout")
    logits = inf["logits"]
    cls_feats = inf["cls"]
    labels_np = inf["labels"].numpy()

    # Primary: T_applied, no blending
    probs_cal = torch.softmax(logits / T_applied, dim=-1)
    y_pred_primary = probs_cal.argmax(dim=-1).numpy()
    primary_per_class = per_class_f1(labels_np, y_pred_primary)
    primary_sqrtn = sqrtn_macro(labels_np, y_pred_primary)
    primary_acc = float((y_pred_primary == labels_np).mean())
    primary_n_correct = int((y_pred_primary == labels_np).sum())
    primary_ci = bootstrap_ci(labels_np, y_pred_primary)
    log(f"\n  PRIMARY (no blending): sqrtn={primary_sqrtn:.4f} "
        f"CI=[{primary_ci['overall_lo']:.4f}, {primary_ci['overall_hi']:.4f}] "
        f"acc={primary_acc:.4f}")

    # Diagnostic: with blending
    probs_blended, n_blended = prototype_blend(probs_cal, cls_feats, proto_bank)
    y_pred_blended = probs_blended.argmax(dim=-1).numpy()
    blended_sqrtn = sqrtn_macro(labels_np, y_pred_blended)
    prediction_delta_count = int((y_pred_primary != y_pred_blended).sum())
    if prediction_delta_count == 0:
        prediction_delta_str = "IDENTICAL — blending changed no final predictions"
    else:
        prediction_delta_str = f"{prediction_delta_count} predictions differ"
    log(f"  DIAGNOSTIC (with blending): sqrtn={blended_sqrtn:.4f}, "
        f"{n_blended} images had max_prob<0.60, {prediction_delta_count} pred changes")

    # Blending interpretation
    if prediction_delta_count == 0:
        blend_interp = ("Prototype blending changed zero final predictions on held-out. "
                        "Empirically confirms Option ζ: blending adds no deployment value.")
    elif blended_sqrtn > primary_sqrtn + 0.005:
        blend_interp = ("Blending improved held-out F1. Revisit Option ζ: blending may deserve "
                        "reinstatement in production.")
    elif blended_sqrtn < primary_sqrtn - 0.005:
        blend_interp = ("Blending decreased held-out F1. Option ζ validated: keep blending OFF.")
    else:
        blend_interp = ("Blending effect within ±0.005 — statistically indistinguishable from "
                        "no-blend. Option ζ stands.")

    # === Save per-image predictions ===
    predictions = []
    for i, rec in enumerate(heldout):
        predictions.append({
            "image_path": rec.image_path,
            "true_class": rec.class_name,
            "pred_class_primary": TOMATO_CLASSES[int(y_pred_primary[i])],
            "pred_class_blended": TOMATO_CLASSES[int(y_pred_blended[i])],
            "max_prob_primary": float(probs_cal[i].max().item()),
            "max_prob_blended": float(probs_blended[i].max().item()),
            "all_probs_primary": {TOMATO_CLASSES[j]: float(probs_cal[i, j].item())
                                  for j in range(len(TOMATO_CLASSES))},
        })
    PREDICTIONS_JSON.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    log(f"  per-image predictions saved: {PREDICTIONS_JSON.name}")

    # === Report ===
    v3_per_class_baseline = {
        "tomato_foliar_spot": 0.629,
        "tomato_septoria_leaf_spot": 0.667,
        "tomato_late_blight": 0.800,
        "tomato_yellow_leaf_curl_virus": 0.800,
        "tomato_mosaic_virus": 0.857,
        "tomato_healthy": 0.967,
    }
    report_data = {
        "val_f1_preserved": float(ckpt.get("val_sqrtn_macro_f1", 0.0)),
        "n_heldout": len(heldout),
        "n_per_class": n_per_class,
        "T_applied": T_applied,
        "primary": {
            "per_class_f1": primary_per_class.tolist(),
            "sqrtn": primary_sqrtn,
            "accuracy": primary_acc,
            "n_correct": primary_n_correct,
            "ci": {
                "per_class_lo": primary_ci["per_class_lo"].tolist(),
                "per_class_hi": primary_ci["per_class_hi"].tolist(),
                "overall_lo": primary_ci["overall_lo"],
                "overall_hi": primary_ci["overall_hi"],
            },
        },
        "blended": {
            "sqrtn": blended_sqrtn,
            "n_blended": n_blended,
        },
        "prediction_delta": prediction_delta_str,
        "blend_interpretation": blend_interp,
        "v3_per_class_baseline": v3_per_class_baseline,
    }
    write_report(report_data)

    # === Save production artifact ===
    production = {
        "sp_lora_state_dict": ckpt["model_state_dict"],
        "val_f1_preserved": float(ckpt.get("val_sqrtn_macro_f1", 0.0)),
        "final_val_sqrtn_f1_primary": float(primary_sqrtn),
        "final_val_sqrtn_f1_ci_lo": float(primary_ci["overall_lo"]),
        "final_val_sqrtn_f1_ci_hi": float(primary_ci["overall_hi"]),
        "final_val_accuracy": float(primary_acc),
        "prototype_bank": proto_bank,
        "calibration": calibration,
        "tier_thresholds": tier_doc,
        "deployment_blending_enabled": False,  # Option ζ default
        "source_checkpoint": SP_LORA_CKPT.name,
        "tomato_classes": list(TOMATO_CLASSES),
        "architecture": "DINOv2-Base-Registers + LoRA(blocks 4-11, rank=8, alpha=16) + Linear(768,6)",
        "input_resolution": 392,
        "preprocessing": "800px cap -> letterbox(392, pad=114) -> LAB-CLAHE(L) -> RGB -> ImageNet norm",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(production, PRODUCTION_PATH)
    log(f"  Production artifact saved: {PRODUCTION_PATH.name}")

    log("\n" + "=" * 64)
    log("TASK A.3 COMPLETE")
    log("=" * 64)
    log(f"  held-out sqrtn macro F1 (PRIMARY, no blending): {primary_sqrtn:.4f}")
    log(f"  95% bootstrap CI: [{primary_ci['overall_lo']:.4f}, {primary_ci['overall_hi']:.4f}]")
    log(f"  vs v3 baseline ({V3_BASELINE_HELDOUT:.3f}): "
        f"{primary_sqrtn - V3_BASELINE_HELDOUT:+.4f}")
    log(f"  vs Phase 1 ({PHASE1_HELDOUT:.4f}): {primary_sqrtn - PHASE1_HELDOUT:+.4f}")
    log(f"  blending verdict: {blend_interp}")


if __name__ == "__main__":
    main()
