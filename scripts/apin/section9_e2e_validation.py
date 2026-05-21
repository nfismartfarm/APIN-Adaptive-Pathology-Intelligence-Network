"""Section 9 — End-to-End APIN Validation on val_and_soup.

Loads the trained APIN system + the 3 (or 4) signal caches and computes
the full APIN ensemble macro F1 from cached signal predictions
(faster than re-running image inference for 1350 val images).

This is the offline validation: it bypasses Gate Zero / preprocessing
because we already have validated signal predictions from Section 2A/B/C/D.
The result tells us "what would the stacking MLP + tier logic produce on
the val set if all the signals are working correctly."

Outputs:
  scripts/apin/results/apin_e2e_val_metrics_{ts}.json
  scripts/apin/results/apin_e2e_val_summary_{ts}.txt
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"
RESULTS_DIR = APIN_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section9_e2e_val_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section9")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)


def load_caches_and_build_X(use_psv: bool, class_order):
    s1 = pickle.load(open(CACHE_DIR / "signal1_predictions_cache.pkl", "rb"))
    s2 = pickle.load(open(CACHE_DIR / "signal2_predictions_cache.pkl", "rb"))
    s4 = pickle.load(open(CACHE_DIR / "signal4_predictions_cache.pkl", "rb"))
    s3_path = CACHE_DIR / "signal3_psv_predictions_cache.pkl"
    s3 = pickle.load(open(s3_path, "rb")) if (use_psv and s3_path.exists()) else None
    n_sig = 4 if (use_psv and s3 is not None) else 3
    # Restrict to keys present in ALL signal caches; PSV cache may be smaller
    # if extraction failed for some images. Intersection guarantees we never
    # KeyError when zipping signals together.
    keys = sorted(set(s1.keys()) & set(s2.keys()) & set(s4.keys())
                   & (set(s3.keys()) if s3 is not None else set(s1.keys())))
    X = np.zeros((len(keys), n_sig * 9), dtype=np.float32)
    y = np.zeros(len(keys), dtype=np.int64)
    splits = []
    is_field = np.zeros(len(keys), dtype=bool)
    sources = []
    for i, k in enumerate(keys):
        vec = [s1[k]["predictions"], s2[k]["predictions"]]
        if n_sig == 4:
            vec.append(s3[k]["predictions"])
        vec.append(s4[k]["predictions"])
        X[i] = np.concatenate(vec).astype(np.float32)
        y[i] = s1[k]["true_class_idx"]
        splits.append(s1[k]["split"])
        is_field[i] = s1[k]["is_field_photo"]
        sources.append(s1[k]["source_dataset"])
    return X, y, np.array(splits), is_field, np.array(sources), n_sig


def load_apin_mlp(n_sig):
    """Load the trained APIN stacking MLP from cache."""
    from scripts.apin.section4_stacking_mlp import APIN_Ensemble
    ckpt_path = CACHE_DIR / "apin_stacking_mlp.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["n_signals"] != n_sig:
        logger.warning(f"  Trained MLP n_sig={ckpt['n_signals']} != requested {n_sig}; "
                        "using checkpoint's setting")
        n_sig = ckpt["n_signals"]
    model = APIN_Ensemble(n_signals=n_sig, num_classes=9)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, n_sig, ckpt


def evaluate_split(model, X, y, split_mask, label, class_order, device,
                    per_class_temps=None):
    """Evaluate the stacking MLP on a subset of pre-built X.

    [BUG FIX 2026-04-17]: Section 9 originally argmaxed raw logits without
    applying per-class temperature scaling, while production inference does
    apply it before final softmax. Per-class T values include T<1.0 for
    okra_cercospora and okra_enation, which sharpen those logits and CAN
    shift argmax. Honest production-equivalent F1 needs T applied here too.

    Args:
      X must already be reliability-matrix-modulated (matches training).
      per_class_temps: 9-dim numpy array of per-class temperatures or None.
    """
    Xs = X[split_mask]; ys = y[split_mask]
    if len(ys) == 0:
        logger.info(f"{label}: empty split")
        return None
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(Xs).to(device)
        if hasattr(model, "forward") and "return_gate_weights" in \
                model.forward.__code__.co_varnames:
            logits, gates = model(x_t, return_gate_weights=True)
        else:
            out = model(x_t)
            if isinstance(out, tuple): logits, gates = out
            else: logits, gates = out, None
        # Apply per-class temperature scaling to match production inference
        if per_class_temps is not None:
            t = torch.from_numpy(per_class_temps).to(device)
            logits = logits / t.unsqueeze(0).clamp(min=1e-3)
        preds = logits.argmax(dim=1).cpu().numpy()
    macro_f1 = f1_score(ys, preds, average="macro",
                          labels=list(range(9)), zero_division=0)
    weighted_f1 = f1_score(ys, preds, average="weighted",
                            labels=list(range(9)), zero_division=0)
    acc = accuracy_score(ys, preds)
    per_class = f1_score(ys, preds, average=None,
                          labels=list(range(9)), zero_division=0)
    cm = confusion_matrix(ys, preds, labels=list(range(9)))
    logger.info(f"\n{label} (n={len(ys)}):")
    logger.info(f"  macro F1:    {macro_f1:.4f}")
    logger.info(f"  weighted F1: {weighted_f1:.4f}")
    logger.info(f"  accuracy:    {acc:.4f}")
    logger.info(f"  Per-class F1:")
    for i, c in enumerate(class_order):
        logger.info(f"    {c:<28} {per_class[i]:.4f}")
    return {"label": label, "n": int(len(ys)),
             "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
             "accuracy": float(acc),
             "per_class_f1": {c: float(p) for c, p in zip(class_order, per_class)},
             "confusion_matrix": cm.tolist()}


def main():
    logger.info("=" * 70)
    logger.info("APIN SECTION 9 -- End-to-End Validation")
    logger.info("=" * 70)

    from scripts.apin.constants import MODEL2_CLASS_ORDER
    class_order = MODEL2_CLASS_ORDER

    # Load caches; check if PSV available
    use_psv = (CACHE_DIR / "signal3_psv_predictions_cache.pkl").exists()
    logger.info(f"Use PSV (Signal 3)? {use_psv}")

    X, y, splits, is_field, sources, n_sig = load_caches_and_build_X(use_psv, class_order)
    logger.info(f"Loaded {len(y)} samples, n_signals={n_sig}, X.shape={X.shape}")

    model, n_sig, ckpt = load_apin_mlp(n_sig)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info(f"Loaded APIN MLP: trained val F1 = {ckpt.get('val_macro_f1')}")

    # [BUG FIX 2026-04-17]: apply reliability matrix to X (training did this;
    # Section 9 was eval-ing un-modulated inputs against weights learned on
    # modulated inputs). The shipped checkpoint stores R; reuse it.
    R = np.asarray(ckpt["reliability_matrix"], dtype=np.float32)
    if R.shape != (n_sig, 9):
        logger.warning(f"Reliability matrix shape {R.shape} != ({n_sig}, 9); "
                        "skipping modulation (results may diverge from production)")
    else:
        from scripts.apin.section4_stacking_mlp import apply_reliability_modulation
        X = apply_reliability_modulation(X, R, n_sig)
        logger.info("Reliability matrix applied to X (matches training pipeline)")

    # [BUG FIX 2026-04-17]: load per-class temperature scaling from
    # apin_calibration.json so argmax matches production inference.
    per_class_temps = None
    cal_path = APIN_DIR / "caches" / "apin_calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        t_map = cal["temperature_scaling"]["per_class_temperatures"]
        per_class_temps = np.array([t_map[c] for c in class_order],
                                    dtype=np.float32)
        logger.info(f"Per-class temperature scaling loaded: "
                    f"min={per_class_temps.min():.3f}, max={per_class_temps.max():.3f}")
    else:
        logger.warning("No apin_calibration.json — temperature scaling skipped")

    # Evaluate per split
    results_by_split = {}
    for split_name in ["val_and_soup", "final_val", "conformal"]:
        mask = splits == split_name
        r = evaluate_split(model, X, y, mask, split_name, class_order, device,
                            per_class_temps=per_class_temps)
        if r is not None: results_by_split[split_name] = r

    # Field-photo subset on val_and_soup
    field_mask = (splits == "val_and_soup") & is_field
    r_field = evaluate_split(model, X, y, field_mask,
                              "val_and_soup_FIELD_only",
                              class_order, device,
                              per_class_temps=per_class_temps)
    if r_field is not None:
        results_by_split["val_and_soup_FIELD_only"] = r_field

    # Per-source breakdown for failure classes on val_and_soup
    logger.info("\n" + "=" * 70)
    logger.info("Per-source accuracy on val_and_soup for failure classes")
    logger.info("=" * 70)
    val_mask = splits == "val_and_soup"
    Xv = X[val_mask]; yv = y[val_mask]; src_v = sources[val_mask]
    with torch.no_grad():
        out = model(torch.from_numpy(Xv).to(device))
        if isinstance(out, tuple): logits = out[0]
        else: logits = out
        # Apply per-class temperature scaling for production-equivalent argmax
        if per_class_temps is not None:
            t = torch.from_numpy(per_class_temps).to(device)
            logits = logits / t.unsqueeze(0).clamp(min=1e-3)
        preds_v = logits.argmax(dim=1).cpu().numpy()
    for cls_name in ["brassica_black_rot", "okra_cercospora"]:
        cls_idx = class_order.index(cls_name)
        cls_mask = yv == cls_idx
        if cls_mask.sum() == 0: continue
        logger.info(f"\n{cls_name} (n={int(cls_mask.sum())} val):")
        for src in sorted(set(src_v[cls_mask])):
            sub_mask = cls_mask & (src_v == src)
            n = int(sub_mask.sum())
            if n < 1: continue
            correct = int(((preds_v == cls_idx) & sub_mask).sum())
            logger.info(f"  source={src:<25} n={n:>4}  acc={correct/n:.4f}")

    # Save report
    report = {
        "timestamp": TIMESTAMP,
        "n_signals": int(n_sig),
        "use_psv": bool(use_psv),
        "stacking_mlp_trained_val_macro_f1": ckpt.get("val_macro_f1"),
        "results_by_split": results_by_split,
        "model_card_summary": {
            "section_4_best_epoch": ckpt.get("best_epoch"),
            "stacking_mlp_n_signals": int(n_sig),
            "gate_mean": ckpt.get("gate_mean"),
            "reliability_matrix": ckpt.get("reliability_matrix"),
        },
    }
    out_path = RESULTS_DIR / f"apin_e2e_val_metrics_{TIMESTAMP}.json"
    out_latest = RESULTS_DIR / "apin_e2e_val_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out_latest, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path.name} (+ latest)")
    logger.info("=" * 70)
    logger.info("APIN SECTION 9 -- COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
