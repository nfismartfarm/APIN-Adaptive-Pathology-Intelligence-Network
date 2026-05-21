"""Compute the 4-signal APIN Reliability Matrix once the PSV cache exists.

R[signal_idx][class_idx] = empirical argmax accuracy on val_and_soup +
is_field_photo=True subset. Row order: [S1_Model2, S2_EfficientNet,
S3_PSV, S4_DINOv2-head].

Output: scripts/apin/caches/reliability_matrix_4signal.json
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
LOG_PATH = APIN_DIR / f"reliability_matrix_4signal_{TIMESTAMP}.log"

logger = logging.getLogger("apin.reliability4")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)


def compute_row(cache, class_order, signal_name):
    from scripts.apin.constants import NUM_CLASSES
    val_rows = [
        e for e in cache.values()
        if e["split"] == "val_and_soup"
        and e["is_field_photo"]
        and e.get("extraction_success", e.get("inference_success", True))
    ]
    accuracies = [0.0] * NUM_CLASSES
    counts = [0] * NUM_CLASSES
    correct = [0] * NUM_CLASSES
    for entry in val_rows:
        cls_idx = entry["true_class_idx"]
        argmax = int(np.argmax(entry["predictions"]))
        counts[cls_idx] += 1
        if argmax == cls_idx:
            correct[cls_idx] += 1
    for i in range(NUM_CLASSES):
        accuracies[i] = correct[i] / counts[i] if counts[i] > 0 else 0.0
    logger.info(f"\n{signal_name} — n_val_field={len(val_rows)}:")
    logger.info(f"  {'Class':<28} {'n':>6} {'acc':>8}")
    for c_name, a, n in zip(class_order, accuracies, counts):
        logger.info(f"  {c_name:<28} {n:>6} {a:>8.4f}")
    return accuracies, counts, len(val_rows)


def main():
    logger.info("=" * 70)
    logger.info("APIN RELIABILITY MATRIX — 4-SIGNAL")
    logger.info("=" * 70)

    from scripts.apin.constants import MODEL2_CLASS_ORDER

    psv_cache_path = CACHE_DIR / "signal3_psv_predictions_cache.pkl"
    if not psv_cache_path.exists():
        logger.error(f"Missing PSV cache at {psv_cache_path}")
        logger.error("  Run section2d_psv_features.py + section3c + section2d_phase2 first.")
        return 1

    files = [
        (0, "Signal 1 (Model 2)", "signal1_predictions_cache.pkl"),
        (1, "Signal 2 (EfficientNet)", "signal2_predictions_cache.pkl"),
        (2, "Signal 3 (PSV)", "signal3_psv_predictions_cache.pkl"),
        (3, "Signal 4 (DINOv2 head)", "signal4_predictions_cache.pkl"),
    ]
    rows = []
    all_counts = None
    n_total = 0
    for _, name, fn in files:
        with open(CACHE_DIR / fn, "rb") as f:
            cache = pickle.load(f)
        r, c, n = compute_row(cache, MODEL2_CLASS_ORDER, name)
        rows.append(r)
        if all_counts is None:
            all_counts = c
        n_total = n

    R = np.array(rows, dtype=np.float32)

    logger.info("\n" + "=" * 70)
    logger.info("RELIABILITY MATRIX R[4][9]")
    logger.info("=" * 70)
    header = f"  {'Class':<28}" + "".join(f"{s:>8}" for s in ["S1_M2", "S2_EN", "S3_PSV", "S4_DNo"])
    logger.info(header)
    for i, c_name in enumerate(MODEL2_CLASS_ORDER):
        row_s = f"  {c_name:<28}"
        for s in range(4):
            row_s += f"{R[s, i]:>8.4f}"
        logger.info(row_s)

    # Best per class
    logger.info("\nBest signal per class (including PSV):")
    signal_names = ["Signal 1 (M2)", "Signal 2 (EN)", "Signal 3 (PSV)", "Signal 4 (DNo)"]
    for i, c_name in enumerate(MODEL2_CLASS_ORDER):
        winner_idx = int(R[:, i].argmax())
        logger.info(f"  {c_name:<28}: {signal_names[winner_idx]:<18} ({R[winner_idx, i]:.4f})")

    payload = {
        "timestamp": TIMESTAMP,
        "description": (
            "APIN 4-signal reliability matrix. Rows: [S1_Model2, S2_EfficientNet, "
            "S3_PSV, S4_DINOv2-head]. Columns: MODEL2_CLASS_ORDER."
        ),
        "class_order_model2": MODEL2_CLASS_ORDER,
        "split_filter": "val_and_soup",
        "is_field_photo_filter": True,
        "total_rows_included": int(n_total),
        "per_class_field_counts": all_counts,
        "matrix_4x9": [[float(v) for v in row] for row in R],
        "matrix_row_description": [
            "Row 0: Signal 1 (Model 2)",
            "Row 1: Signal 2 (EfficientNet)",
            "Row 2: Signal 3 (PSV)",
            "Row 3: Signal 4 (DINOv2 head)",
        ],
    }
    out = CACHE_DIR / f"reliability_matrix_4signal_{TIMESTAMP}.json"
    latest = CACHE_DIR / "reliability_matrix_4signal.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    with open(latest, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nSaved: {latest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
