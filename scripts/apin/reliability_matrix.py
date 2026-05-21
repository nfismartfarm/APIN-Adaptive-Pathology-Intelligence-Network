"""Compute the APIN Reliability Matrix from the 3 available prediction caches.

R[signal_idx][class_idx] = empirical accuracy of that signal for that class,
measured on val_and_soup + is_field_photo=True images.

Signals:
  R[0] = Model 2 (Signal 1)
  R[1] = EfficientNet (Signal 2)  [NOTE: 23-class subsampled to 9 — argmax
                                    may fall outside the 9 classes]
  R[2] = PSV (Signal 3)           [NOT AVAILABLE — placeholder row, marked
                                    clearly]
  R[3] = DINOv2 head (Signal 4)

The matrix is used in Layer 3 of APIN inference to modulate the 36-dim
stacking-MLP input. Each signal's 9-value prediction vector gets multiplied
by its row of R before concatenation.

Output: scripts/apin/caches/reliability_matrix_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"reliability_matrix_{TIMESTAMP}.log"

logger = logging.getLogger("apin.reliability")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(fmt)
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)


def compute_reliability_row(cache, class_order, signal_name,
                              metric: str = "argmax_accuracy"):
    """For each class, compute a per-class reliability score for one signal
    on val_and_soup + is_field_photo=True images.

    Args:
        cache: signal prediction cache (dict of entries).
        class_order: list of class names defining index → name.
        signal_name: human-readable name for logging.
        metric: one of:
          - "argmax_accuracy" (default): fraction of images of class i where
            argmax(predictions) == i. Correct for softmax-normalised signals
            (Model 2, EfficientNet, DINOv2 head).
          - "auroc_one_vs_rest": AUROC of `predictions[i]` as a one-vs-rest
            score for class i. Correct for *continuous-feature* signals
            (PSV) where argmax across independent per-class scores is
            semantically meaningless. AUROC measures whether class-i's
            score is higher on class-i images than on non-class-i images,
            which is exactly what the MLP uses the value for.

    Returns 9-dim list of reliability scores + per-class counts.
    """
    from scripts.apin.constants import NUM_CLASSES
    from sklearn.metrics import roc_auc_score

    # Filter
    val_rows = [
        e for e in cache.values()
        if e["split"] == "val_and_soup"
        and e["is_field_photo"]
        and e.get("inference_success", True)
    ]
    total = len(val_rows)

    scores = [0.0] * NUM_CLASSES
    counts = [0] * NUM_CLASSES
    correct = [0] * NUM_CLASSES

    if metric == "argmax_accuracy":
        for entry in val_rows:
            cls_idx = entry["true_class_idx"]
            argmax = int(np.argmax(entry["predictions"]))
            counts[cls_idx] += 1
            if argmax == cls_idx:
                correct[cls_idx] += 1
        for i in range(NUM_CLASSES):
            if counts[i] > 0:
                scores[i] = correct[i] / counts[i]
            else:
                scores[i] = 1.0
                logger.warning(f"  {signal_name}: zero field samples for class "
                               f"index {i} — defaulting reliability to 1.0")
    elif metric == "auroc_one_vs_rest":
        # Stack predictions into [N, NUM_CLASSES]; build per-class binary y
        preds = np.asarray([e["predictions"] for e in val_rows], dtype=np.float32)
        y_true = np.asarray([e["true_class_idx"] for e in val_rows], dtype=np.int64)
        for i in range(NUM_CLASSES):
            counts[i] = int((y_true == i).sum())
            if counts[i] == 0 or counts[i] == len(y_true):
                # No positives or no negatives — AUROC undefined; default 1.0
                scores[i] = 1.0
                logger.warning(f"  {signal_name}: no positives/negatives for "
                               f"class index {i} — defaulting reliability to 1.0")
                continue
            y_binary = (y_true == i).astype(np.int64)
            try:
                scores[i] = float(roc_auc_score(y_binary, preds[:, i]))
            except ValueError:
                scores[i] = 1.0
            # AUROC range [0, 1]; values <0.5 mean "anti-correlated" (signal's
            # score is HIGHER on non-class images). Clip to [0.5, 1.0] then
            # rescale to [0, 1] so the MLP modulation never flips signs:
            # 0.5 AUROC → 0.0 reliability, 1.0 AUROC → 1.0 reliability.
            scores[i] = float(max(0.0, 2.0 * (scores[i] - 0.5)))
    else:
        raise ValueError(f"Unknown reliability metric: {metric}")

    logger.info(f"\n{signal_name} — val_and_soup + is_field_photo=True "
                f"(n={total}, metric={metric}):")
    logger.info(f"  {'Class':<28} {'n_field':>8} {'reliability':>12}")
    logger.info(f"  {'-' * 52}")
    for c_name, a, n in zip(class_order, scores, counts):
        logger.info(f"  {c_name:<28} {n:>8} {a:>12.4f}")
    return scores, counts, total


def main():
    logger.info("=" * 70)
    logger.info("APIN RELIABILITY MATRIX (4-signal version)")
    logger.info("=" * 70)
    logger.info(
        "R[signal][class] = empirical argmax accuracy on val_and_soup + "
        "is_field_photo=True subset. Used in Layer 3 to modulate the "
        "36-dim MLP input before the stacking MLP."
    )

    from scripts.apin.constants import MODEL2_CLASS_ORDER
    class_order = MODEL2_CLASS_ORDER
    NUM_CLASSES = len(class_order)

    # Load caches
    logger.info("\nLoading 4 signal caches:")
    with open(CACHE_DIR / "signal1_predictions_cache.pkl", "rb") as f:
        cache_s1 = pickle.load(f)
    logger.info(f"  Signal 1 (Model 2): {len(cache_s1)} entries")
    with open(CACHE_DIR / "signal2_predictions_cache.pkl", "rb") as f:
        cache_s2 = pickle.load(f)
    logger.info(f"  Signal 2 (EfficientNet): {len(cache_s2)} entries")
    with open(CACHE_DIR / "signal3_psv_predictions_cache.pkl", "rb") as f:
        cache_s3 = pickle.load(f)
    logger.info(f"  Signal 3 (PSV): {len(cache_s3)} entries")
    with open(CACHE_DIR / "signal4_predictions_cache.pkl", "rb") as f:
        cache_s4 = pickle.load(f)
    logger.info(f"  Signal 4 (DINOv2 head): {len(cache_s4)} entries")

    # Compute reliability rows.
    # Model 2, EfficientNet, DINOv2 head emit per-class softmax distributions
    # → argmax accuracy is the right reliability metric.
    # PSV emits per-class continuous "is disease X present" scores that are
    # NOT a softmax — argmax across them is meaningless. Use AUROC one-vs-rest
    # which measures whether class-i's score is reliably higher on class-i
    # images than on non-class-i images. (Gap 6 fix, audit round 7.)
    r0, counts, n_total = compute_reliability_row(
        cache_s1, class_order, "Signal 1 (Model 2)", metric="argmax_accuracy"
    )
    r1, _, _ = compute_reliability_row(
        cache_s2, class_order, "Signal 2 (EfficientNet)", metric="argmax_accuracy"
    )
    r2, _, _ = compute_reliability_row(
        cache_s3, class_order, "Signal 3 (PSV)", metric="auroc_one_vs_rest"
    )
    r3, _, _ = compute_reliability_row(
        cache_s4, class_order, "Signal 4 (DINOv2 head)", metric="argmax_accuracy"
    )

    # Pretty-print the matrix
    logger.info("\n" + "=" * 70)
    logger.info("RELIABILITY MATRIX R[4][9] — empirical field-photo accuracy")
    logger.info("=" * 70)
    header = f"  {'Class':<28}  {'S1 M2':>7}  {'S2 EN':>7}  {'S3 PSV':>7}  {'S4 DNo':>7}"
    logger.info(header)
    logger.info(f"  {'-' * 68}")
    for i, c_name in enumerate(class_order):
        s1_s = f"{r0[i]:.3f}" if r0[i] is not None else "  —  "
        s2_s = f"{r1[i]:.3f}" if r1[i] is not None else "  —  "
        s3_s = f"{r2[i]:.3f}" if r2[i] is not None else "  —  "
        s4_s = f"{r3[i]:.3f}" if r3[i] is not None else "  —  "
        logger.info(f"  {c_name:<28}  {s1_s:>7}  {s2_s:>7}  {s3_s:>7}  {s4_s:>7}")

    # Summary — which signal wins per class
    logger.info("\nBest signal per class:")
    for i, c_name in enumerate(class_order):
        vals = [("Signal 1", r0[i]), ("Signal 2", r1[i]),
                ("Signal 3", r2[i]), ("Signal 4", r3[i])]
        best = max(vals, key=lambda x: x[1])
        logger.info(
            f"  {c_name:<28}  winner: {best[0]:<10} ({best[1]:.4f})"
        )

    # Build payload
    payload = {
        "timestamp": TIMESTAMP,
        "description": (
            "APIN reliability matrix R[signal_idx][class_idx] — empirical "
            "argmax accuracy of each signal for each class on val_and_soup "
            "+ is_field_photo=True subset of the Model 2 data. Used in "
            "Layer 3 of APIN inference to modulate the 36-dim MLP input."
        ),
        "class_order_model2": class_order,
        "split_filter": "val_and_soup",
        "is_field_photo_filter": True,
        "total_rows_included": int(n_total),
        "per_class_field_counts": counts,
        "signals": {
            "S0_model2_dinov3_convnext": {
                "signal_name": "Model 2 (DINOv3-ConvNeXt-Small)",
                "cache": "signal1_predictions_cache.pkl",
                "per_class_accuracy": [round(float(v), 6) for v in r0],
            },
            "S1_efficientnet_23class": {
                "signal_name": ("EfficientNet V2-S 23-class 4-crop "
                                 "(subsampled to 9)"),
                "cache": "signal2_predictions_cache.pkl",
                "per_class_accuracy": [round(float(v), 6) for v in r1],
            },
            "S2_psv": {
                "signal_name": "PSV (Physics-based Symptom Verification)",
                "cache": "signal3_psv_predictions_cache.pkl",
                "per_class_accuracy": [round(float(v), 6) for v in r2],
            },
            "S3_dinov2_nonlinear_head": {
                "signal_name": "DINOv2 nonlinear head (frozen backbone)",
                "cache": "signal4_predictions_cache.pkl",
                "per_class_accuracy": [round(float(v), 6) for v in r3],
            },
        },
        "matrix_4x9": [
            [round(float(v), 6) for v in r0],
            [round(float(v), 6) for v in r1],
            [round(float(v), 6) for v in r2],
            [round(float(v), 6) for v in r3],
        ],
        "matrix_4x9_rows_description": [
            "Row 0: Signal 1 (Model 2)",
            "Row 1: Signal 2 (EfficientNet)",
            "Row 2: Signal 3 (PSV)",
            "Row 3: Signal 4 (DINOv2 head)",
        ],
        "notes": [
            "EfficientNet argmax accuracy is a LOWER BOUND: the 23-class "
            "argmax may fall outside the 9 Model 2 classes for images "
            "that visually resemble tomato/chilli. For the stacking MLP, "
            "the raw 9-dim sigmoid values matter more than the argmax.",
            "Reliability matrix must be recomputed after any signal "
            "retraining.",
        ],
    }

    out_path = CACHE_DIR / f"reliability_matrix_4signal_{TIMESTAMP}.json"
    out_latest = CACHE_DIR / "reliability_matrix_4signal.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_latest, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nSaved: {out_path.name}")
    logger.info(f"Latest: {out_latest.name}")


if __name__ == "__main__":
    main()
