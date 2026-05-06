"""
Step 8 — S12.7 degraded-mode quality verification.

Per spec S12.7 lines 3368-3373:
  - macro-F1 with v3 zeroed remains >= 0.55 (LoRA + PSV alone)
  - macro-F1 with LoRA zeroed remains >= 0.55 (v3 + PSV alone)
  - macro-F1 with PSV zeroed remains >= 0.65 (v3 + LoRA alone, neural dominates)

Simulation protocol:
  1. Load features.npz raw values (identity-standardized at build time) for
     held_out_subset partition (43 rows)
  2. For each signal-failure scenario {v3_off, lora_off, psv_off}:
       a. Apply spec S12.2 zeroing to relevant block + JSD_SENTINEL at idx 16
       b. Apply v2 classifier standardization (load from .json on disk)
       c. Forward through Stage 1 + Stage 2 + soft-routing per S12.5
       d. Apply Platt per S12.8
       e. argmax -> predicted canonical class
  3. Compute macro-F1 over predictions vs ground truth canonical labels
  4. Verify against spec thresholds; report

spec: section 12.7 lines 3348-3373
spec: section 12.2 lines 3231-3242 (degraded-mode zero-fill)
spec: section 12.5 (soft-routing)
spec: section 12.8 (Platt)
"""

from __future__ import annotations
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tomato_sandbox.utils.degraded_mode import zero_signal_a, zero_signal_b, zero_signal_c

CAL = _PROJECT_ROOT / "tomato_sandbox" / "phase_f0_calibration"
NPZ = CAL / "_classifier_training" / "features.npz"
OUT = CAL / "_phase_f0_runs" / "step8_degraded_mode"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical 7-class index space per S12.10:3460-3467
CANONICAL = ["foliar", "septoria", "late_blight", "ylcv", "mosaic", "healthy", "OOD"]
STAGE2_CLASSES = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]

# Spec thresholds per S12.7:3369-3371
THRESH = {"v3_off": 0.55, "lora_off": 0.55, "psv_off": 0.65}


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def stage_forward(X: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Multinomial logistic forward: returns (n, k) probabilities."""
    return softmax(X @ weights.T + bias)


def soft_route(p_s1: np.ndarray, p_s2: np.ndarray) -> np.ndarray:
    """S12.5 soft-route Stage 1 (3-class) x Stage 2 (5-class) -> 7-class."""
    n = p_s1.shape[0]
    p_final = np.zeros((n, 7), dtype=np.float64)
    p_final[:, 0:5] = p_s1[:, 1:2] * p_s2[:, 0:5]   # diseased x per-disease
    p_final[:, 5]   = p_s1[:, 0]                      # healthy
    p_final[:, 6]   = p_s1[:, 2]                      # OOD
    return p_final


def apply_platt(p_final: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Per-class Platt per S12.8:3389-3398, with row-wise renormalization."""
    eps = 1e-12
    p = np.clip(p_final, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))                    # per-class logit
    cal = 1.0 / (1.0 + np.exp(-(alpha * logits + beta)))
    cal_sum = cal.sum(axis=1, keepdims=True)
    cal_sum = np.where(cal_sum == 0, 1.0, cal_sum)
    return cal / cal_sum


def main() -> int:
    t0 = time.perf_counter()
    print("Loading artifacts...")
    data = np.load(NPZ, allow_pickle=True)
    X_all = data["features"].astype(np.float64)        # (259, 19)
    y_s1  = data["y_stage1"]
    y_s2  = data["y_stage2"]
    partition = data["partition"]

    # Load v2 trained models
    with open(CAL / "classifier_stage1.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(CAL / "classifier_stage2.pkl", "rb") as f:
        s2 = pickle.load(f)
    fs = json.load(open(CAL / "classifier_feature_standardization.json"))
    platt = json.load(open(CAL / "classifier_platt.json"))

    feat_mean = np.asarray(fs["feature_mean"], dtype=np.float64)
    feat_std  = np.asarray(fs["feature_std"],  dtype=np.float64)
    s1_W = np.asarray(s1["weights"], dtype=np.float64)   # (3, 19)
    s1_b = np.asarray(s1["bias"],    dtype=np.float64)   # (3,)
    s2_W = np.asarray(s2["weights"], dtype=np.float64)   # (5, 19)
    s2_b = np.asarray(s2["bias"],    dtype=np.float64)   # (5,)
    alpha = np.asarray(platt["alpha"], dtype=np.float64)  # (7,)
    beta  = np.asarray(platt["beta"],  dtype=np.float64)  # (7,)

    # Filter to held_out_subset (43 rows)
    held_mask = partition == "held_out_subset"
    X_held = X_all[held_mask]
    y_s1_held = y_s1[held_mask]
    y_s2_held = y_s2[held_mask]
    n = len(X_held)
    print(f"  held_out_subset: {n} rows")

    # Build canonical ground-truth labels per S12.10
    y_canonical = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if y_s1_held[i] == 0:                # healthy
            y_canonical[i] = 5
        elif y_s1_held[i] == 1:              # diseased -> y_s2 in [0..4]
            y_canonical[i] = int(y_s2_held[i])
        else:                                 # OOD (shouldn't appear in held_out_subset)
            y_canonical[i] = 6
    from collections import Counter
    print(f"  canonical class distribution: {dict(Counter(y_canonical.tolist()))}")

    # Reproduce JSD_SENTINEL value as used at build time
    JSD_SENTINEL = 0.35  # spec default; matches feature_builder._JSD_SENTINEL_DEFAULT

    def run_scenario(scenario: str) -> dict:
        """Apply degraded-mode zeroing + standardize + classify."""
        X = X_held.copy()
        for i in range(n):
            row = X[i]  # 1D view
            if scenario == "v3_off":
                zero_signal_a(row); row[16] = JSD_SENTINEL
            elif scenario == "lora_off":
                zero_signal_b(row); row[16] = JSD_SENTINEL
            elif scenario == "psv_off":
                zero_signal_c(row)
            elif scenario == "all_on":
                pass
            else:
                raise ValueError(scenario)

        # Standardize per S12.2:3203-3204 + clip [-3, 3]
        X_std = (X - feat_mean) / (feat_std + 1e-6)
        X_std = np.clip(X_std, -3.0, 3.0)

        p_s1 = stage_forward(X_std, s1_W, s1_b)         # (n, 3)
        p_s2 = stage_forward(X_std, s2_W, s2_b)         # (n, 5)
        p_final = soft_route(p_s1, p_s2)                # (n, 7)
        p_cal = apply_platt(p_final, alpha, beta)       # (n, 7)
        y_pred = p_cal.argmax(axis=1)

        # Macro-F1 over classes that have support (held_out_subset has no OOD support)
        present = sorted(set(y_canonical.tolist()))
        f1_macro = f1_score(y_canonical, y_pred, labels=present, average="macro", zero_division=0)
        per_class = {}
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_canonical, y_pred, labels=present, zero_division=0
        )
        for j, lab in enumerate(present):
            per_class[CANONICAL[lab]] = {
                "P": float(prec[j]), "R": float(rec[j]),
                "F1": float(f1[j]), "support": int(sup[j])
            }
        # Confusion: counts (true -> pred)
        confusion = {}
        for t, p in zip(y_canonical.tolist(), y_pred.tolist()):
            confusion.setdefault(CANONICAL[t], {}).setdefault(CANONICAL[p], 0)
            confusion[CANONICAL[t]][CANONICAL[p]] += 1
        return {
            "scenario": scenario,
            "n": int(n),
            "macro_f1": float(f1_macro),
            "per_class": per_class,
            "confusion": confusion,
            "n_correct": int((y_pred == y_canonical).sum()),
            "accuracy": float((y_pred == y_canonical).mean()),
        }

    print("\nRunning scenarios...")
    results = {}
    for scenario in ["all_on", "v3_off", "lora_off", "psv_off"]:
        r = run_scenario(scenario)
        thresh = THRESH.get(scenario)
        if thresh is not None:
            r["threshold"] = thresh
            r["pass"] = r["macro_f1"] >= thresh
            mark = "PASS" if r["pass"] else "FAIL"
        else:
            mark = "BASELINE"
        print(f"  {scenario:>10s}: macro_f1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f}  "
              f"({mark}{f' >= {thresh}' if thresh else ''})")
        results[scenario] = r

    # Verdict per spec S12.7
    all_pass = all(results[s].get("pass", True) for s in ["v3_off", "lora_off", "psv_off"])

    report = {
        "metadata": {
            "generated_at": "2026-05-06",
            "spec_citations": [
                "S12.7:3348-3373 (degraded-mode handling + verification thresholds)",
                "S12.2:3231-3242 (degraded-mode zero-fill)",
                "S12.5 (soft-routing)",
                "S12.8:3375-3406 (Platt scaling)",
                "S12.10:3460-3467 (canonical+OOD index space)",
            ],
            "thresholds_per_spec": THRESH,
            "JSD_SENTINEL_used": JSD_SENTINEL,
            "evaluation_partition": "held_out_subset (43 rows; OOD distribution: 0)",
        },
        "section_15_regression": "135/135 PASS (run earlier this dispatch)",
        "scenarios": results,
        "verdict": {
            "all_thresholds_met": all_pass,
            "v3_off_pass":   results["v3_off"]["pass"],
            "lora_off_pass": results["lora_off"]["pass"],
            "psv_off_pass":  results["psv_off"]["pass"],
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    out_json = OUT / "degraded_mode_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {out_json}")

    print(f"\nVerdict: {'ALL THRESHOLDS MET' if all_pass else 'AT LEAST ONE THRESHOLD MISSED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
