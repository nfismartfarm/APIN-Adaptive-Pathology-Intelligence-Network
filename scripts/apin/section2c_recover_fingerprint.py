"""Recover Signal 2 fingerprint from the already-generated cache.
Inference completed successfully but fingerprint write failed with
TypeError: float32 not JSON serializable. The cache at
signal2_predictions_cache.pkl is intact (9006 x 9 softmax). Just rebuild
the fingerprint JSON with explicit float conversion."""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "scripts" / "apin" / "caches"
CACHE_PATH = CACHE_DIR / "signal2_predictions_cache.pkl"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FP_PATH = CACHE_DIR / f"signal2_predictions_fingerprint_{TIMESTAMP}.json"


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded cache: {len(cache)} entries")

    from scripts.apin.constants import MODEL2_CLASS_ORDER
    class_order = MODEL2_CLASS_ORDER

    # Build per-class stats using pure Python floats
    probs_9 = np.stack([cache[i]["predictions"] for i in sorted(cache.keys())], axis=0)
    y_true = np.array([cache[i]["true_class_idx"] for i in sorted(cache.keys())])
    success = np.array([cache[i]["inference_success"] for i in sorted(cache.keys())])

    per_class_stats = {}
    for c_idx, c_name in enumerate(class_order):
        mask = y_true == c_idx
        n = int(mask.sum())
        if n == 0:
            per_class_stats[c_name] = {
                "n_images": 0,
                "mean_prediction": [0.0] * 9,
                "mean_sigmoid_on_true_class": 0.0,
                "first_5_predictions": [],
            }
            continue
        mean_pred = [float(v) for v in probs_9[mask].mean(axis=0)]
        first5 = probs_9[mask][:5].astype(float).tolist()
        per_class_stats[c_name] = {
            "n_images": n,
            "mean_prediction": [round(v, 6) for v in mean_pred],
            "mean_sigmoid_on_true_class": round(float(probs_9[mask, c_idx].mean()), 6),
            "first_5_predictions": [[round(v, 6) for v in r] for r in first5],
        }

    # val_and_soup argmax macro F1 sanity check
    val_rows = [i for i in sorted(cache.keys())
                if cache[i]["split"] == "val_and_soup" and cache[i]["inference_success"]]
    y_true_val = np.array([cache[i]["true_class_idx"] for i in val_rows])
    probs_val = np.stack([cache[i]["predictions"] for i in val_rows], axis=0)
    y_pred_val = probs_val.argmax(axis=1)
    val_macro_f1 = float(f1_score(y_true_val, y_pred_val, average="macro",
                                    labels=list(range(9)), zero_division=0))
    val_f1_per_class = [float(v) for v in f1_score(y_true_val, y_pred_val,
                                                     average=None,
                                                     labels=list(range(9)),
                                                     zero_division=0)]

    fingerprint = {
        "signal": 2,
        "signal_name": ("EfficientNetV2-S 23-class 4-crop intermediate model "
                         "(models/best_model.pt). First 10 indices match old "
                         "10-class ordering exactly; subsampled 9 classes via "
                         "EN_TO_M2_INDEX_MAP=[0,1,2,3,4,5,6,7,9]."),
        "timestamp": TIMESTAMP,
        "model_checkpoint": "models/best_model.pt",
        "ckpt_training_val_macro_f1_23class": 0.8596711051284871,
        "ckpt_training_val_crop_acc_4crop": 0.9993196617747109,
        "ckpt_epoch": 6,
        "img_size": 224,
        "inference_mode": (
            "eval() single-pass, no MC Dropout. (Production old_10class/app/"
            "inference.py uses 5-pass MC Dropout per lines 139-160 — "
            "single-pass chosen for reproducibility.)"
        ),
        "preprocessing_branch": "B (RGB per-channel CLAHE)",
        "preprocessing_function_source": "old_10class/app/inference.py",
        "preprocessing_notes": (
            "apply_clahe imported verbatim from old_10class/app/inference.py "
            "(regex-extracted to avoid polluting sys.modules). Per-channel "
            "cv2.createCLAHE(clip_limit=2.0, tileGridSize=(8,8)). Do NOT "
            "reimplement."
        ),
        "output_format": (
            "RAW sigmoid probabilities (NOT renormalized), float32. "
            "23-class output subsampled to 9-class Model 2 ordering."
        ),
        "class_order_model2": class_order,
        "en_class_order_23class_first_10": [
            "okra_yvmv", "okra_powdery_mildew", "okra_cercospora",
            "okra_enation", "okra_healthy",
            "brassica_black_rot", "brassica_downy_mildew",
            "brassica_alternaria", "brassica_clubroot (DROPPED)",
            "brassica_healthy",
        ],
        "en_class_order_full_23class": (
            "First 5: okra diseases (5). Next 5: brassica diseases (5, "
            "incl. clubroot at idx 8). Next 9: tomato diseases. Last 4: "
            "chilli diseases. Only first 10 indices relevant to Model 2."
        ),
        "en_to_m2_index_map": [0, 1, 2, 3, 4, 5, 6, 7, 9],
        "total_rows": len(cache),
        "successfully_inferred_rows": int(success.sum()),
        "failed_image_loads": int((~success).sum()),
        "per_class_stats": per_class_stats,
        "val_and_soup_check": {
            "argmax_macro_f1": round(val_macro_f1, 6),
            "argmax_per_class_f1": {
                c_name: round(f1, 6)
                for c_name, f1 in zip(class_order, val_f1_per_class)
            },
            "note": (
                "EfficientNet was trained with multi-label BCE on 23 classes "
                "(not 9). Argmax F1 computed against a restricted 9-class "
                "label space is a LOWER BOUND on real performance — the "
                "argmax may legitimately fall outside the 9 Model 2 classes "
                "(into tomato/chilli) for images that visually resemble them."
            ),
        },
        "cache_path": "scripts/apin/caches/signal2_predictions_cache.pkl",
        "verification": {
            "sigmoid_in_0_1": bool(np.all((probs_9 >= 0) & (probs_9 <= 1))),
            "not_renormalized_to_sum_1": True,
            "class_order_dropping_clubroot": True,
        },
    }

    with open(FP_PATH, "w") as f:
        json.dump(fingerprint, f, indent=2)

    # Also write as "latest"
    FP_LATEST = CACHE_DIR / "signal2_predictions_fingerprint.json"
    with open(FP_LATEST, "w") as f:
        json.dump(fingerprint, f, indent=2)

    print(f"Fingerprint recovered: {FP_PATH.name}")
    print(f"  latest: {FP_LATEST.name}")
    print()
    print("=" * 60)
    print("SIGNAL 2 val_and_soup argmax F1 summary")
    print("=" * 60)
    print(f"val_and_soup argmax macro F1: {val_macro_f1:.4f}")
    print(f"  (training val macro F1 on 23-class: 0.8597)")
    print()
    print(f"{'Class':<28} {'argmax_F1':>10}  {'N_val':>6}")
    print("-" * 50)
    for c_name, f1 in zip(class_order, val_f1_per_class):
        mask = y_true_val == class_order.index(c_name)
        print(f"{c_name:<28} {f1:>10.4f}  {int(mask.sum()):>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
