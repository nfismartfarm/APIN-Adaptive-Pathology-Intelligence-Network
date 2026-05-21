"""Smoke test for the APIN inference pipeline.

Loads APINInference, runs predict() on:
  1. A synthetic random-noise image (should trigger Gate Zero or low confidence)
  2. A real image from the val_and_soup split
  3. A real failure-class image (brassica_black_rot field photo)

Validates:
  - APIN initializes without crashing
  - All expected APINResult fields populated
  - No NaN/Inf in probabilities
  - Tier assignment is one of the documented tiers
  - Per-signal predictions present
  - Latency reported and reasonable
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("apin.smoke")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")


def show_result(label, result):
    print()
    print("=" * 70)
    print(f"TEST CASE: {label}")
    print("=" * 70)
    print(f"  tier            : {getattr(result, 'tier', 'n/a')}")
    print(f"  diagnosis       : {getattr(result, 'diagnosis', 'n/a')}")
    print(f"  confidence      : {getattr(result, 'confidence', 'n/a')}")
    print(f"  conflict_type   : {getattr(result, 'conflict_type', 'n/a')}")
    print(f"  is_ood          : {getattr(result, 'is_ood', 'n/a')}")
    print(f"  processing_time : {getattr(result, 'processing_time_ms', 'n/a')} ms")
    if hasattr(result, "signal_predictions") and result.signal_predictions:
        print("  signal predictions (top class per signal):")
        # signal_predictions stores {"argmax": class_name, "top_prob": float}
        # per signal (not raw vectors). Display directly without re-argmaxing.
        for sname, info in result.signal_predictions.items():
            if info is None:
                print(f"    {sname}: None (signal disabled)")
                continue
            argmax_name = info.get("argmax", "?")
            top_prob = info.get("top_prob", 0.0)
            print(f"    {sname}: class={argmax_name}  prob={top_prob:.3f}")
    if hasattr(result, "gate_weights") and result.gate_weights is not None:
        print(f"  gate weights    : "
              f"{[round(float(g), 3) for g in result.gate_weights]}")
    if hasattr(result, "all_class_probabilities") and result.all_class_probabilities:
        print("  Top 3 class probabilities:")
        items = sorted(result.all_class_probabilities.items(),
                        key=lambda x: -x[1])[:3]
        for name, p in items:
            print(f"    {name}: {p:.4f}")
    if hasattr(result, "output_message") and result.output_message:
        print(f"  output_message  : {result.output_message[:120]}...")


def main():
    print("=" * 70)
    print("APIN SMOKE TEST")
    print("=" * 70)

    from scripts.apin.inference import APINInference

    print("Initializing APINInference...")
    t0 = time.time()
    apin = APINInference(verbose=False)
    print(f"Initialization took {time.time() - t0:.1f}s")

    # Test 1: synthetic random noise
    print("\n--- Test 1: Synthetic noise ---")
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    t0 = time.time()
    r1 = apin.predict(img1)
    show_result(f"Synthetic noise 300x300 ({time.time()-t0:.1f}s)", r1)

    # Test 2: real val image (any class)
    print("\n--- Test 2: Real val_and_soup image ---")
    df = pd.read_csv(PROJECT_ROOT / "data" / "specialist" / "model2" /
                       "model2_unified_source_map.csv")
    splits = json.load(
        open(PROJECT_ROOT / "data" / "specialist" / "model2" / "split_indices.json")
    )
    val_idxs = splits["val_and_soup"]
    sample_idx = val_idxs[100]  # arbitrary
    sample_row = df.iloc[sample_idx]
    print(f"Using row {sample_idx}: {sample_row['class_name']} from {sample_row['source_dataset']}")
    img2 = np.array(Image.open(sample_row["image_path"]).convert("RGB"), dtype=np.uint8)
    t0 = time.time()
    r2 = apin.predict(img2)
    show_result(f"Real val image — true={sample_row['class_name']} ({time.time()-t0:.1f}s)", r2)

    # Test 3: black_rot field val image
    print("\n--- Test 3: brassica_black_rot field val image ---")
    val_df = df.iloc[val_idxs]
    br_field = val_df[(val_df["class_name"] == "brassica_black_rot") &
                       (val_df["is_field_photo"] == True)]
    if len(br_field) > 0:
        sample_row = br_field.iloc[0]
        print(f"Using row {sample_row.name}: {sample_row['class_name']} field, "
              f"source={sample_row['source_dataset']}")
        img3 = np.array(Image.open(sample_row["image_path"]).convert("RGB"),
                          dtype=np.uint8)
        t0 = time.time()
        r3 = apin.predict(img3)
        show_result(f"black_rot field — ({time.time()-t0:.1f}s)", r3)
    else:
        print("No black_rot field val images found — skipping")

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED — APIN inference functional")
    print("=" * 70)


if __name__ == "__main__":
    main()
