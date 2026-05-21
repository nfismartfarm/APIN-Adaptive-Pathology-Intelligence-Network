"""
PSV Calibration — fits percentile normalization from training data.

Run ONCE against training images to compute per-feature statistics.
Saves calibration parameters to psv_calibration.json.
Subsequent PSV runs load saved calibration for consistent normalization.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List
from pathlib import Path

from scripts.psv.config import PSV_CFG


def compute_calibration(feature_dicts: List[Dict[str, float]],
                        labels: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration parameters from a collection of feature dictionaries.

    For each feature, computes:
      - p5: 5th percentile (low anchor)
      - p25: 25th percentile
      - p50: median
      - p75: 75th percentile
      - p95: 95th percentile (high anchor)
      - mean, std

    Args:
        feature_dicts: list of {feature_name: value} dicts (one per image)
        labels: optional list of class labels (for per-class stats)

    Returns:
        Dict of {feature_name: {p5, p25, p50, p75, p95, mean, std}}
    """
    if not feature_dicts:
        return {}

    # Collect all feature names
    all_names = set()
    for fd in feature_dicts:
        all_names.update(fd.keys())

    calibration = {}
    for name in sorted(all_names):
        values = [fd.get(name, 0.0) for fd in feature_dicts]
        values = [v for v in values if v is not None and np.isfinite(v)]
        if not values:
            continue
        arr = np.array(values)
        calibration[name] = {
            'p5': float(np.percentile(arr, 5)),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'count': len(values),
        }

    return calibration


def save_calibration(calibration: Dict, path: str = None):
    """Save calibration to JSON."""
    if path is None:
        path = os.path.join(PSV_CFG.ROOT, PSV_CFG.CALIBRATION_PATH)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f'Calibration saved: {path} ({len(calibration)} features)')


def load_calibration(path: str = None) -> Dict:
    """Load calibration from JSON. Returns empty dict if not found."""
    if path is None:
        path = os.path.join(PSV_CFG.ROOT, PSV_CFG.CALIBRATION_PATH)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run_calibration_pipeline(csv_path: str = None, max_images: int = 500,
                             save_path: str = None):
    """
    Full calibration pipeline: load training images, extract features, compute stats.

    Args:
        csv_path: path to unified source map CSV
        max_images: maximum images to process (for speed)
        save_path: where to save calibration JSON
    """
    import pandas as pd
    from PIL import Image
    from scripts.psv.feature_extractor import extract_all_features

    if csv_path is None:
        csv_path = os.path.join(PSV_CFG.ROOT, 'data', 'specialist', 'model2',
                                'model2_unified_source_map.csv')

    df = pd.read_csv(csv_path)

    # Sample balanced across classes
    sampled = []
    per_class = max(max_images // PSV_CFG.NUM_CLASSES, 10)
    for cls in PSV_CFG.CLASS_NAMES:
        cls_df = df[df['class_name'] == cls]
        n = min(len(cls_df), per_class)
        sampled.append(cls_df.sample(n=n, random_state=42))
    sampled_df = pd.concat(sampled).reset_index(drop=True)

    print(f'Calibration: processing {len(sampled_df)} images...', flush=True)

    feature_dicts = []
    labels = []
    t0 = time.time()

    for i, (_, row) in enumerate(sampled_df.iterrows()):
        path = row.get('clahe_path', row['image_path'])
        if not isinstance(path, str) or not os.path.exists(path):
            path = row['image_path']
        try:
            img = np.array(Image.open(path).convert('RGB'))
            result = extract_all_features(img)
            feature_dicts.append(result.features)
            labels.append(row['class_name'])
        except Exception as e:
            print(f'  Skip {os.path.basename(str(path))}: {e}')

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(sampled_df)} ({elapsed:.0f}s)', flush=True)

    calibration = compute_calibration(feature_dicts, labels)
    save_calibration(calibration, save_path)

    elapsed = time.time() - t0
    print(f'Calibration complete: {len(calibration)} features from {len(feature_dicts)} images '
          f'in {elapsed:.0f}s', flush=True)

    return calibration


if __name__ == '__main__':
    run_calibration_pipeline()
