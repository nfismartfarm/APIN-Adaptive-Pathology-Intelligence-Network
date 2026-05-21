"""
PSV Adversarial Augmentation — synthetic failure pattern generation.

Creates training points that mimic the EXACT failure patterns observed
on real-world photos:
  - Model 2 says alternaria 77%, black_rot 8% on a real black_rot photo
  - Model 2 says okra_healthy 82%, cercospora 5% on a real cercospora photo

The PSV features are from the REAL image (unchanged).
Only the Model 2 signal is replaced with the failure pattern.
This teaches the MLP: "when Model 2 is wrong but PSV says disease X, trust PSV."
"""

import numpy as np
from typing import Dict, List, Tuple
from scripts.psv.config import PSV_CFG


def generate_adversarial_points(
    original_features: np.ndarray,  # [N, 36] — all 4 signals concatenated
    labels: np.ndarray,             # [N] — class indices
    model2_dim: int = 9,            # first 9 values = Model 2 signal
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate adversarial training points for the MLP.

    For each image of a failure class, creates a synthetic point where:
    - Model 2 signal is replaced with the known failure pattern
    - EfficientNet, PSV, DINOv2 signals are unchanged (from real image)
    - Label is the CORRECT class

    Args:
        original_features: [N, 36] feature vectors for all training images
        labels: [N] integer class labels
        model2_dim: number of Model 2 output dimensions (first 9 of the 36)

    Returns:
        (adversarial_features, adversarial_labels, adversarial_weights)
        All numpy arrays ready to append to training data.
    """
    cfg = PSV_CFG
    class_to_idx = {name: i for i, name in enumerate(cfg.CLASS_NAMES)}

    adv_features = []
    adv_labels = []
    adv_weights = []

    for class_name, failure_pattern in cfg.ADVERSARIAL_PATTERNS.items():
        if class_name not in class_to_idx:
            continue
        class_idx = class_to_idx[class_name]
        failure_signal = np.array(failure_pattern, dtype=np.float32)

        # Find all images of this class
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]

        for idx in class_indices:
            # Copy original feature vector
            adv_vec = original_features[idx].copy()
            # Replace Model 2 signal (first model2_dim values) with failure pattern
            adv_vec[:model2_dim] = failure_signal
            # Keep EfficientNet, PSV, DINOv2 signals unchanged

            adv_features.append(adv_vec)
            adv_labels.append(class_idx)
            adv_weights.append(cfg.ADVERSARIAL_WEIGHT)

    if not adv_features:
        return (np.array([]).reshape(0, original_features.shape[1]),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32))

    return (np.array(adv_features, dtype=np.float32),
            np.array(adv_labels, dtype=np.int64),
            np.array(adv_weights, dtype=np.float32))


def compute_sample_weights(labels: np.ndarray, is_field: np.ndarray,
                           sources: np.ndarray = None) -> np.ndarray:
    """
    Compute per-sample training weights.

    Tier 1 (weight=5.0): field photos
    Tier 2 (weight=2.0): diverse lab photos
    Tier 3 (weight=1.5): recomposed synthetic
    Tier 4 (weight=1.0): standard lab
    """
    cfg = PSV_CFG
    weights = np.ones(len(labels), dtype=np.float32) * cfg.STANDARD_LAB_WEIGHT

    if is_field is not None:
        field_mask = is_field.astype(bool) if is_field.dtype != bool else is_field
        weights[field_mask] = cfg.FIELD_PHOTO_WEIGHT

    if sources is not None:
        for i, src in enumerate(sources):
            src_str = str(src).lower()
            if 'recomp' in src_str:
                weights[i] = cfg.RECOMPOSED_WEIGHT
            elif not is_field[i] and src_str not in ('', 'nan'):
                weights[i] = max(weights[i], cfg.DIVERSE_LAB_WEIGHT)

    return weights
