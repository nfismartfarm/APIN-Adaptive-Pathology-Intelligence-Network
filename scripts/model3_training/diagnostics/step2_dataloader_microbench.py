"""
Step 2 / Diagnostic — DataLoader num_workers microbenchmark on Windows.

Spec contradiction:
  - new spec says: num_workers=2 (verified stable)
  - PHASE0_LOG Entry 009 says: num_workers=2 was 28x slower than num_workers=0; use 0.

Decide empirically. Test num_workers ∈ {0, 2, 4, 6} for 200 batches each, end-to-end
with model.forward() so we measure what training will actually see (minus backward).

Writes:
  scripts/model3_training/logs/step2_dataloader_microbench.json
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.config_model3 import (
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, CROP_FROM_IDX,
    DINOV2_BACKBONE, DINOV2_IMG_SIZE, IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
)
from scripts.models import Model3DINOLoRA

CSV_PATH  = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
SPLIT_PATH = ROOT / 'data' / 'specialist' / 'model3' / 'split_indices.json'
OUT_PATH  = ROOT / 'scripts' / 'model3_training' / 'logs' / 'step2_dataloader_microbench.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
N_BATCHES = 200
WORKER_SETTINGS = [0, 2, 4, 6]


def apply_lab_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class MicroDataset(Dataset):
    """Minimal dataset that mirrors what training will see:
    - Read image from disk
    - LAB-CLAHE (when clahe_path missing)
    - Resize to 224
    - ImageNet normalize
    Returns: (img tensor, label, crop_id)
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clahe = row.get('clahe_path')
        img = None
        if isinstance(clahe, str) and clahe:
            full = ROOT / clahe
            if full.exists():
                img = cv2.imread(str(full), cv2.IMREAD_COLOR)
        if img is None:
            full = ROOT / row['image_path']
            img = cv2.imread(str(full), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
            else:
                img = apply_lab_clahe(img)
        img = cv2.resize(img, IMG_SIZE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb.transpose(2, 0, 1) - self.mean) / self.std
        x = torch.from_numpy(x).float()
        cls = row['class_name']
        label = CLASS_TO_IDX.get(cls, 0)
        crop = CROP_FROM_IDX.get(label, 0)
        return x, label, int(crop)


def benchmark(num_workers: int, model, ds, n_batches: int = N_BATCHES) -> dict:
    """Measure end-to-end img/sec for the given num_workers."""
    print(f"\n--- num_workers={num_workers} ---")
    print(f"  building DataLoader...", flush=True)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Warmup: 10 batches (DataLoader spin-up + first cuDNN tuning)
    print(f"  warmup (10 batches)...", flush=True)
    it = iter(dl)
    t_warm = time.time()
    for i in range(10):
        try:
            x, y, c = next(it)
        except StopIteration:
            it = iter(dl)
            x, y, c = next(it)
        x = x.to(DEVICE, non_blocking=True)
        c = c.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            _ = model(x, crop_ids=c)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  warmup done in {time.time()-t_warm:.1f}s")

    # Timed run
    print(f"  timing {n_batches} batches...", flush=True)
    t0 = time.time()
    n = 0
    for i in range(n_batches):
        try:
            x, y, c = next(it)
        except StopIteration:
            it = iter(dl)
            x, y, c = next(it)
        x = x.to(DEVICE, non_blocking=True)
        c = c.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            _ = model(x, crop_ids=c)
        n += x.size(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    img_s = n / elapsed
    sec_per_batch = elapsed / n_batches
    print(f"  imgs/sec: {img_s:.1f}   sec/batch: {sec_per_batch:.3f}   total: {elapsed:.1f}s")

    # Cleanup workers
    del dl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'num_workers': num_workers,
        'n_batches': n_batches,
        'images_processed': n,
        'elapsed_seconds': round(elapsed, 2),
        'images_per_second': round(img_s, 2),
        'seconds_per_batch': round(sec_per_batch, 4),
    }


def main():
    print("=" * 72)
    print("STEP 2 — DataLoader num_workers microbenchmark (Windows)")
    print("=" * 72)
    print(f"DEVICE: {DEVICE}")
    print(f"BATCH_SIZE: {BATCH_SIZE},  N_BATCHES: {N_BATCHES} per setting")

    # Load train rows so the benchmark hits realistic file paths
    df = pd.read_csv(CSV_PATH)
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    train_idx = splits['train']
    train_df = df.iloc[train_idx].reset_index(drop=True)
    print(f"Train rows available: {len(train_df)}")

    # Use a 6400-row sample so 200 batches @ 32 is fully covered without epoch wrap
    sample = train_df.sample(n=min(6400, len(train_df)), random_state=42).reset_index(drop=True)
    ds = MicroDataset(sample)
    print(f"Microbench dataset size: {len(ds)}")

    # Build model once
    print(f"\nBuilding Model3DINOLoRA (forward-only, eval mode)...")
    model = Model3DINOLoRA(num_classes=NUM_CLASSES, num_crops=2,
                           pretrained=False, enable_gradient_checkpointing=False)
    model = model.to(DEVICE).eval()

    results = []
    for nw in WORKER_SETTINGS:
        try:
            r = benchmark(nw, model, ds)
            results.append(r)
        except Exception as e:
            print(f"  FAILED at num_workers={nw}: {e}")
            results.append({'num_workers': nw, 'error': str(e)})

    # Pick the winner
    valid = [r for r in results if 'images_per_second' in r]
    winner = max(valid, key=lambda r: r['images_per_second']) if valid else None

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'num_workers':<14} {'imgs/sec':>10} {'sec/batch':>11} {'elapsed':>10}")
    print("-" * 50)
    for r in results:
        if 'error' in r:
            print(f"{r['num_workers']:<14} ERROR: {r['error'][:40]}")
        else:
            print(f"{r['num_workers']:<14} {r['images_per_second']:>10.2f} "
                  f"{r['seconds_per_batch']:>11.4f} {r['elapsed_seconds']:>10.2f}")
    if winner:
        print(f"\nWINNER: num_workers={winner['num_workers']}  ({winner['images_per_second']:.1f} img/s)")
        # vs alternatives
        baseline_zero = next((r for r in valid if r['num_workers'] == 0), None)
        if baseline_zero and winner['num_workers'] != 0:
            ratio = winner['images_per_second'] / baseline_zero['images_per_second']
            print(f"  vs num_workers=0: {ratio:.2f}x faster")

    out = {
        'device': str(DEVICE),
        'batch_size': BATCH_SIZE,
        'n_batches_per_setting': N_BATCHES,
        'results': results,
        'winner': winner,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
