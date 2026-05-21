"""
OOD Detection — Mahalanobis distance in DINOv2 feature space.

Detects when a new image is outside the training distribution entirely.
Uses class-conditional Mahalanobis with Ledoit-Wolf shrinkage for
regularized covariance estimation (handles n_samples < n_features).
"""

import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dinov2_probe.config import (
    CLASS_NAMES, NUM_CLASSES, OOD_DISTANCE_PERCENTILE,
    RESULTS_DIR, RANDOM_SEED,
)


class MahalanobisOODDetector:
    """Class-conditional Mahalanobis distance OOD detector."""

    def __init__(self):
        self.class_means = {}      # {cls_idx: mean_vector}
        self.class_cov_inv = {}    # {cls_idx: precision_matrix}
        self.threshold = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit per-class means and covariance matrices from training data."""
        for cls_idx in range(NUM_CLASSES):
            mask = y_train == cls_idx
            if mask.sum() < 2:
                logger.warning(f"Class {cls_idx} has <2 samples, using global stats")
                continue

            cls_features = X_train[mask]
            self.class_means[cls_idx] = cls_features.mean(axis=0)

            # Ledoit-Wolf shrinkage covariance (handles high-dim)
            try:
                lw = LedoitWolf()
                lw.fit(cls_features)
                self.class_cov_inv[cls_idx] = lw.precision_
            except Exception as e:
                logger.warning(f"LedoitWolf failed for class {cls_idx}: {e}. "
                             f"Using diagonal covariance.")
                var = cls_features.var(axis=0) + 1e-6
                self.class_cov_inv[cls_idx] = np.diag(1.0 / var)

        self.is_fitted = True
        logger.info(f"OOD detector fitted on {len(self.class_means)} classes")

    def compute_distance(self, x: np.ndarray) -> float:
        """Compute min Mahalanobis distance across all classes."""
        if not self.is_fitted:
            raise RuntimeError("OOD detector not fitted")

        min_dist = float('inf')
        for cls_idx in self.class_means:
            diff = x - self.class_means[cls_idx]
            dist = np.sqrt(diff @ self.class_cov_inv[cls_idx] @ diff)
            min_dist = min(min_dist, dist)

        return float(min_dist)

    def calibrate(self, X_val: np.ndarray, percentile: float = 95) -> float:
        """Set OOD threshold from val set distances."""
        distances = np.array([self.compute_distance(x) for x in X_val])
        self.threshold = float(np.percentile(distances, percentile))
        logger.info(f"OOD threshold: {self.threshold:.4f} "
                   f"(p{percentile} of {len(X_val)} val samples)")
        return self.threshold

    def is_ood(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check if a sample is OOD. Returns (is_ood, distance)."""
        dist = self.compute_distance(x)
        return dist > self.threshold, dist

    def save(self, path: Path) -> None:
        """Save fitted detector."""
        with open(path, 'wb') as f:
            pickle.dump({
                'class_means': self.class_means,
                'class_cov_inv': self.class_cov_inv,
                'threshold': self.threshold,
            }, f)

    def load(self, path: Path) -> None:
        """Load fitted detector."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.class_means = data['class_means']
        self.class_cov_inv = data['class_cov_inv']
        self.threshold = data['threshold']
        self.is_fitted = True


def run_ood_detection(data: Dict) -> Dict:
    """Run OOD detection pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}", flush=True)
    print("OOD DETECTION (Mahalanobis distance)", flush=True)
    print(f"{'='*60}", flush=True)

    detector = MahalanobisOODDetector()
    detector.fit(data['train']['X'], data['train']['y'])
    threshold = detector.calibrate(data['val']['X'], OOD_DISTANCE_PERCENTILE)

    # Test on val set
    val_distances = np.array([detector.compute_distance(x)
                              for x in data['val']['X']])
    n_ood = (val_distances > threshold).sum()
    print(f"Val set: {n_ood}/{len(val_distances)} flagged as OOD "
          f"({n_ood/len(val_distances)*100:.1f}%)", flush=True)

    # Test on synthetic OOD
    np.random.seed(RANDOM_SEED)
    dim = data['train']['X'].shape[1]
    synthetic_ood = {
        'gaussian_noise': np.random.randn(10, dim).astype(np.float32),
        'uniform_random': np.random.uniform(-2, 2, (10, dim)).astype(np.float32),
        'zeros': np.zeros((5, dim), dtype=np.float32),
    }

    print(f"\nSynthetic OOD detection:", flush=True)
    ood_results = {}
    for name, samples in synthetic_ood.items():
        dists = [detector.compute_distance(x) for x in samples]
        detected = sum(d > threshold for d in dists)
        rate = detected / len(samples)
        print(f"  {name}: {detected}/{len(samples)} detected as OOD "
              f"({rate*100:.0f}%), mean dist={np.mean(dists):.2f}", flush=True)
        ood_results[name] = {'rate': rate, 'mean_dist': float(np.mean(dists))}

    # Save detector
    detector_path = RESULTS_DIR / f'ood_detector_{ts}.pkl'
    detector.save(detector_path)
    print(f"OOD detector saved: {detector_path}", flush=True)

    return {
        'threshold': threshold,
        'val_ood_count': int(n_ood),
        'val_total': len(val_distances),
        'synthetic_results': ood_results,
        'detector_path': str(detector_path),
    }


if __name__ == '__main__':
    from scripts.dinov2_probe.feature_cache import load_cache
    from scripts.dinov2_probe.probe_train import prepare_data
    from scripts.dinov2_probe.config import FEATURES_CACHE_PATH
    cache = load_cache(FEATURES_CACHE_PATH)
    data = prepare_data(cache)
    run_ood_detection(data)
