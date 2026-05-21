"""Test-Time Augmentation evaluation for model3_production_v3.pt.

Spec reference: TTA EVALUATION PROMPT (Round B follow-up).

Pipeline per image:
  1. Load BGR uint8 via existing `_load_image(row)` — returns image already
     LAB-CLAHE'd (either from `clahe_path` pre-processed file or by applying
     `apply_lab_clahe` at load time on `image_path`). DO NOT re-apply CLAHE.
  2. For each of N views (deterministic transforms), run:
        bgr_view = transform(bgr)            # geometric or photometric on BGR uint8
        tensor   = bgr_to_normalized(bgr_view)   # [3, 224, 224] float32
        logits   = model(tensor[None], crop_mode_tensor)
        probs    = softmax(logits)
     Collect probs per view.
  3. averaged_probs = mean(probs across views)        # softmax-then-average
  4. prediction = argmax(averaged_probs)

Aggregation rule: average probabilities (NOT logits). Averaging logits then
softmax sharpens the distribution and defeats the ensemble effect.

Two crop-mode runs per split:
  - ceiling: tomato classes -> mode=0, chilli classes -> mode=1 (uses GT label)
  - conservative: every image gets mode=2 (uncertain) — measures deployment cost
                  when crop is unknown to the system.

Compares against the existing no-TTA v3 baseline in
`scripts/model3_training/logs/evaluate_model3_production_v3.json`.
Also runs a no-TTA conservative-mode baseline (single forward with crop_mode=2)
because the existing eval only logged ceiling-mode results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# ── Project imports ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.model3_training.architecture.model3_full import Model3
from scripts.model3_training.data.model3_dataset import (
    _load_image, load_and_split_csv,
)
from scripts.model3_training.model3_config import (
    CLASSES_10, NUM_CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CHECKPOINT_DIR, LOG_DIR,
)


# ═════════════════════════════════════════════════════════════════════════
# Tensor preparation — the ONE place BGR-uint8 becomes a model-ready tensor.
# Used by every view. No CLAHE here (already done in _load_image).
# ═════════════════════════════════════════════════════════════════════════
_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD  = np.asarray(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)


def bgr_to_normalized_tensor(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (any HxW) -> RGB float tensor [3, 224, 224] ImageNet-normalized.

    Matches `_to_imagenet_tensor` in data/model3_dataset.py — same resize-then-
    normalize order, same channel order (BGR->RGB), same constants.
    """
    img = cv2.resize(bgr, IMG_SIZE)                              # 224x224 BGR
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)                                 # [3, H, W]
    x = (chw - _MEAN) / _STD
    return torch.from_numpy(x).float()


# ═════════════════════════════════════════════════════════════════════════
# 8 deterministic TTA views.
# Input:  bgr_clahe uint8 [H, W, 3]  (already LAB-CLAHE'd by _load_image)
# Output: tensor [3, 224, 224] float32 ImageNet-normalized
# Each view is a NAMED function (not a lambda) to avoid closure-capture bugs.
# ═════════════════════════════════════════════════════════════════════════

def view_original(bgr: np.ndarray) -> torch.Tensor:
    return bgr_to_normalized_tensor(bgr)


def view_hflip(bgr: np.ndarray) -> torch.Tensor:
    return bgr_to_normalized_tensor(cv2.flip(bgr, 1))   # left-right flip


def view_vflip(bgr: np.ndarray) -> torch.Tensor:
    return bgr_to_normalized_tensor(cv2.flip(bgr, 0))   # top-bottom flip


def view_rot90(bgr: np.ndarray) -> torch.Tensor:
    # cv2.ROTATE_90_COUNTERCLOCKWISE = "rotate 90° CCW" = numpy rot90(k=1).
    return bgr_to_normalized_tensor(cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE))


def view_rot270(bgr: np.ndarray) -> torch.Tensor:
    return bgr_to_normalized_tensor(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE))


def view_crop90(bgr: np.ndarray) -> torch.Tensor:
    h, w = bgr.shape[:2]
    crop_h, crop_w = int(h * 0.9), int(w * 0.9)
    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2
    cropped = bgr[y1:y1 + crop_h, x1:x1 + crop_w]
    return bgr_to_normalized_tensor(cropped)


def view_bright_up(bgr: np.ndarray) -> torch.Tensor:
    # Multiply intensity in BGR space (linear pre-normalization).
    # NOT in-place — np.clip on a fresh float32 copy.
    boosted = np.clip(bgr.astype(np.float32) * 1.15, 0.0, 255.0).astype(np.uint8)
    return bgr_to_normalized_tensor(boosted)


def view_bright_down(bgr: np.ndarray) -> torch.Tensor:
    dimmed = np.clip(bgr.astype(np.float32) * 0.85, 0.0, 255.0).astype(np.uint8)
    return bgr_to_normalized_tensor(dimmed)


# Order matters: first 4 are the "core" 4-view ablation set.
ALL_VIEWS: List[Tuple[str, Callable[[np.ndarray], torch.Tensor]]] = [
    ('original',    view_original),
    ('hflip',       view_hflip),
    ('vflip',       view_vflip),
    ('rot90',       view_rot90),
    ('rot270',      view_rot270),
    ('crop90',      view_crop90),
    ('bright_up',   view_bright_up),
    ('bright_down', view_bright_down),
]


# ═════════════════════════════════════════════════════════════════════════
# Model loading — matches `model3_main.py --mode evaluate` pattern.
# ═════════════════════════════════════════════════════════════════════════
def load_v3_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load v3 production soup checkpoint into a fresh Model3 instance.
    Verifies non-LoRA backbone is float32 + frozen, MixStyle is in eval mode.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = Model3(n_classes=NUM_CLASSES, pretrained=False, use_lora=True).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        # strict=True will have raised; this is defensive.
        raise RuntimeError(f"State dict mismatch. missing={missing[:3]} unexpected={unexpected[:3]}")

    model.eval()

    # Sanity asserts — match dual_stream_eval / Model3.assert_backbone_dtype_and_freeze.
    model.assert_backbone_dtype_and_freeze()
    assert not model.mixstyle.training, "MixStyle still in training mode after model.eval()"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {checkpoint_path.name}  ({n_params:,} params)")
    return model


# ═════════════════════════════════════════════════════════════════════════
# TTA forward pass for ONE image.
# ═════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def tta_predict_single(model: nn.Module,
                       bgr_clahe: np.ndarray,
                       crop_mode: int,
                       views: List[Tuple[str, Callable]],
                       device: torch.device) -> np.ndarray:
    """Run TTA on a single image. Returns averaged probability vector [n_classes].

    bgr_clahe: BGR uint8 image, ALREADY LAB-CLAHE'd (no further CLAHE).
    crop_mode: int in {0, 1, 2}.
    """
    crop_tensor = torch.tensor([crop_mode], dtype=torch.long, device=device)
    probs_acc = None
    n_used = 0
    for name, fn in views:
        try:
            t = fn(bgr_clahe).unsqueeze(0).to(device, non_blocking=True)  # [1, 3, 224, 224]
            assert t.shape == (1, 3, 224, 224), f"view {name}: bad shape {tuple(t.shape)}"
            out = model(t, crop_mode=crop_tensor, domain_labels=None)
            p = torch.softmax(out['logits'], dim=-1).squeeze(0).cpu().numpy()  # [n_classes]
            probs_acc = p if probs_acc is None else probs_acc + p
            n_used += 1
        except Exception as e:
            # Log and continue — partial views are better than skipping the image.
            print(f"  TTA view '{name}' failed: {e}")
            continue
    if n_used == 0:
        raise RuntimeError("All TTA views failed for this image")
    averaged = probs_acc / n_used
    # Sanity: each softmax sums to 1, so the mean of vectors that each sum to 1
    # also sums to 1 (within fp tolerance).
    assert abs(float(averaged.sum()) - 1.0) < 1e-4, f"averaged probs sum={averaged.sum()}"
    return averaged


@torch.no_grad()
def single_pass_predict(model: nn.Module,
                        bgr_clahe: np.ndarray,
                        crop_mode: int,
                        device: torch.device) -> np.ndarray:
    """Single forward pass (no TTA) — used to compute conservative-mode baseline."""
    crop_tensor = torch.tensor([crop_mode], dtype=torch.long, device=device)
    t = bgr_to_normalized_tensor(bgr_clahe).unsqueeze(0).to(device, non_blocking=True)
    out = model(t, crop_mode=crop_tensor, domain_labels=None)
    return torch.softmax(out['logits'], dim=-1).squeeze(0).cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════
# Run a full split + crop-mode evaluation.
# ═════════════════════════════════════════════════════════════════════════
def run_evaluation(model: nn.Module,
                   items: List[dict],
                   split_name: str,
                   mode: str,                  # 'ceiling' or 'conservative'
                   class_names: List[str],
                   device: torch.device,
                   views: Optional[List[Tuple[str, Callable]]] = None) -> dict:
    """Iterate over `items`, run TTA (or single pass if views=None), compute metrics."""
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    n_total = len(items)
    use_tta = views is not None

    print(f"\nEvaluating: split={split_name} mode={mode} "
          f"{'TTA n_views=' + str(len(views)) if use_tta else 'no-TTA (single pass)'} "
          f"n_images={n_total}")

    preds: List[int] = []
    labels: List[int] = []
    is_field: List[bool] = []
    n_load_fail = 0
    t_start = time.time()

    for i, row in enumerate(items):
        if i and i % 500 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else 0
            print(f"  [{i}/{n_total}] elapsed={elapsed:6.1f}s rate={rate:5.1f} img/s eta={eta:6.1f}s")

        bgr = _load_image(row)
        if bgr is None:
            n_load_fail += 1
            continue

        cls = row['class_name']
        if mode == 'ceiling':
            crop_mode = 0 if 'tomato' in cls else 1
        elif mode == 'conservative':
            crop_mode = 2
        else:
            raise ValueError(f"unknown mode {mode!r}")

        if use_tta:
            probs = tta_predict_single(model, bgr, crop_mode, views, device)
        else:
            probs = single_pass_predict(model, bgr, crop_mode, device)

        preds.append(int(np.argmax(probs)))
        labels.append(name_to_idx[cls])
        is_field.append(bool(row.get('is_field_photo', False)))

    elapsed = time.time() - t_start
    if n_load_fail:
        print(f"  WARNING: {n_load_fail} image load failures (skipped)")

    preds_a   = np.asarray(preds)
    labels_a  = np.asarray(labels)
    field_a   = np.asarray(is_field, dtype=bool)
    n_ids = list(range(len(class_names)))

    def _f1(p, l):
        if len(p) == 0:
            return 0.0
        return float(f1_score(l, p, average='macro', zero_division=0, labels=n_ids))

    def _per_class(p, l):
        if len(p) == 0:
            return {c: 0.0 for c in class_names}
        s = f1_score(l, p, average=None, zero_division=0, labels=n_ids)
        return {class_names[i]: float(s[i]) for i in range(len(class_names))}

    overall_f1 = _f1(preds_a, labels_a)
    field_f1   = _f1(preds_a[field_a],  labels_a[field_a])
    lab_f1     = _f1(preds_a[~field_a], labels_a[~field_a])

    pc_overall = _per_class(preds_a, labels_a)
    pc_field   = _per_class(preds_a[field_a],  labels_a[field_a])
    pc_lab     = _per_class(preds_a[~field_a], labels_a[~field_a])

    field_counts = np.bincount(labels_a[field_a],  minlength=len(class_names))
    lab_counts   = np.bincount(labels_a[~field_a], minlength=len(class_names))
    field_class_counts = {class_names[i]: int(field_counts[i]) for i in range(len(class_names))}
    lab_class_counts   = {class_names[i]: int(lab_counts[i])   for i in range(len(class_names))}

    print(f"  overall_f1={overall_f1:.4f}  field_f1={field_f1:.4f}  lab_f1={lab_f1:.4f}  "
          f"({elapsed:.1f}s, {len(preds_a)/elapsed:.1f} img/s)")

    return {
        'split': split_name,
        'mode': mode,
        'tta': bool(use_tta),
        'n_views': len(views) if use_tta else 1,
        'view_names': [n for n, _ in views] if use_tta else ['single_pass'],
        'n_images': int(len(preds_a)),
        'n_field': int(field_a.sum()),
        'n_lab': int((~field_a).sum()),
        'n_load_fail': int(n_load_fail),
        'elapsed_seconds': float(elapsed),
        'overall_f1': overall_f1,
        'field_f1': field_f1,
        'lab_f1': lab_f1,
        'per_class_overall': pc_overall,
        'per_class_field': pc_field,
        'per_class_lab': pc_lab,
        'field_class_counts': field_class_counts,
        'lab_class_counts': lab_class_counts,
    }


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
def _atomic_write_json(path: Path, payload, max_retries: int = 5) -> None:
    """Write JSON atomically: write to .tmp then rename.
    Retries on Windows PermissionError (transient AV / handle-lock races).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    last_err = None
    for attempt in range(max_retries):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default=str(CHECKPOINT_DIR / 'model3_production_v3.pt'))
    parser.add_argument('--n_views', type=int, default=8,
                        help='Number of TTA views from ALL_VIEWS (1..8).')
    parser.add_argument('--output', type=str,
                        default=str(LOG_DIR / 'tta_results.json'))
    parser.add_argument('--include_no_tta_baseline', action='store_true', default=True,
                        help='Also run a single-pass conservative-mode baseline '
                             '(no logged baseline exists for that mode).')
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    if not (1 <= args.n_views <= len(ALL_VIEWS)):
        raise ValueError(f"--n_views must be in [1, {len(ALL_VIEWS)}], got {args.n_views}")
    views = ALL_VIEWS[:args.n_views]
    print(f"TTA views ({len(views)}): {[n for n, _ in views]}")

    model = load_v3_model(Path(args.checkpoint), device)

    splits = load_and_split_csv()

    # Sanity: classes match
    csv_classes = sorted({row['class_name'] for row in splits['final_val']})
    if 'tomato_target_spot' in csv_classes:
        raise RuntimeError("Quarantined class tomato_target_spot present in CSV")
    for c in csv_classes:
        if c not in CLASSES_10:
            raise RuntimeError(f"CSV class {c!r} not in CLASSES_10")

    results = {
        'meta': {
            'checkpoint': args.checkpoint,
            'n_views': len(views),
            'view_names': [n for n, _ in views],
            'device': str(device),
            'class_names': list(CLASSES_10),
        },
        'runs': {},
    }

    # 4 TTA runs: val/final_val × ceiling/conservative
    for split_name in ('val', 'final_val'):
        items = splits[split_name]
        for mode in ('ceiling', 'conservative'):
            key = f"{split_name}_{mode}_tta{len(views)}"
            results['runs'][key] = run_evaluation(
                model, items, split_name, mode, list(CLASSES_10), device, views=views
            )
            _atomic_write_json(Path(args.output), results)  # checkpoint after each run

    # Optional: no-TTA conservative baseline for both splits
    # (existing eval logged ceiling-mode no-TTA only).
    if args.include_no_tta_baseline:
        for split_name in ('val', 'final_val'):
            items = splits[split_name]
            key = f"{split_name}_conservative_no_tta"
            results['runs'][key] = run_evaluation(
                model, items, split_name, 'conservative',
                list(CLASSES_10), device, views=None,
            )
            _atomic_write_json(Path(args.output), results)

    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
