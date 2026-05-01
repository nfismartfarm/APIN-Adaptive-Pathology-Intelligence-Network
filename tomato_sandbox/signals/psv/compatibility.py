"""
PSV Stage 4 — Compatibility scoring.

Loads the 6×26 weight matrix from psv_weights.yaml (agronomist-readable).
Loads per-feature standardization parameters from psv_standardization.json.
Converts the 26 raw features into 6 botanical compatibility scores via:
  1. Standardize (subtract mean, divide std, clip ±3)
  2. WEIGHT_MATRIX @ standardized  → raw_scores [6]
  3. Softmax with temperature T_PSV → compatibility [6]

# spec: 10.6 lines 2642-2751
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from tomato_sandbox.signals.psv.features import FEATURE_NAMES, NUM_FEATURES
from tomato_sandbox.utils.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_SANDBOX_ROOT = Path(__file__).parent.parent.parent  # tomato_sandbox/
_WEIGHTS_PATH = _SANDBOX_ROOT / "config" / "psv_weights.yaml"
_STANDARDIZATION_PATH = (
    _SANDBOX_ROOT / "phase_f0_calibration" / "psv_standardization.json"
)

# ---------------------------------------------------------------------------
# Canonical class order — spec: Section 3.7 / 10.6 lines 2646-2652
# ---------------------------------------------------------------------------
_CLASS_ORDER = ["foliar", "septoria", "late_blight", "ylcv", "mosaic", "healthy"]
NUM_CLASSES = 6


def _load_yaml_weights() -> np.ndarray:
    """Load and validate 6×26 weight matrix from psv_weights.yaml.

    Validates that the feature names in the YAML exactly match FEATURE_NAMES.
    Constructs rows in canonical class order.

    # spec: 10.6.1 lines 2660-2664
    Returns:
        np.ndarray of shape [6, 26], dtype float32.
    Raises:
        RuntimeError if names mismatch or file missing.
    """
    try:
        import yaml  # PyYAML — optional; fall back to regex parser if absent
        loader = yaml.safe_load
    except ImportError:
        # Minimal YAML parser for our simple key: [list] format
        def loader(text: str):  # type: ignore[misc]
            result = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    nums = [float(x.strip()) for x in val[1:-1].split(",")]
                    result[key] = nums
            return result

    if not _WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"psv_weights.yaml not found at {_WEIGHTS_PATH}. "
            "Cannot initialize PSV compatibility scorer."
        )

    with open(_WEIGHTS_PATH, "r", encoding="utf-8") as fh:
        raw = loader(fh.read())

    # Filter to only entries that are lists (skip string/scalar metadata keys)
    weight_dict = {k: v for k, v in raw.items() if isinstance(v, list)}

    # Validate feature name coverage
    yaml_features = set(weight_dict.keys())
    expected_features = set(FEATURE_NAMES)
    if yaml_features != expected_features:
        missing = expected_features - yaml_features
        extra = yaml_features - expected_features
        raise RuntimeError(
            f"psv_weights.yaml feature name mismatch.\n"
            f"  Missing from YAML: {sorted(missing)}\n"
            f"  Extra in YAML:     {sorted(extra)}\n"
            f"Keeping WEIGHT_MATRIX aligned with feature catalog (spec 10.6.1)."
        )

    # Build [6, 26] matrix: rows = canonical class order, cols = FEATURE_NAMES order
    matrix = np.zeros((NUM_CLASSES, NUM_FEATURES), dtype=np.float32)
    for col_idx, feat_name in enumerate(FEATURE_NAMES):
        weights_for_feature = weight_dict[feat_name]  # list of 6 values
        if len(weights_for_feature) != NUM_CLASSES:
            raise RuntimeError(
                f"Feature '{feat_name}' in psv_weights.yaml has "
                f"{len(weights_for_feature)} values; expected {NUM_CLASSES}."
            )
        for row_idx in range(NUM_CLASSES):
            matrix[row_idx, col_idx] = float(weights_for_feature[row_idx])

    _log.debug("PSV weight matrix loaded", shape=list(matrix.shape), path=str(_WEIGHTS_PATH))
    return matrix


def _load_standardization_params() -> tuple[np.ndarray, np.ndarray, float]:
    """Load F0_FEATURE_MEAN, F0_FEATURE_STD, T_PSV from psv_standardization.json.

    Returns placeholders (zeros, ones, 1.0) if file is missing.

    # spec: 10.6.2 lines 2733-2736
    Returns:
        (mean [26], std [26], T_PSV float)
    """
    if not _STANDARDIZATION_PATH.exists():
        _log.warning(
            "psv_standardization.json not found — using placeholder standardization",
            path=str(_STANDARDIZATION_PATH),
        )
        mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        std = np.ones(NUM_FEATURES, dtype=np.float32)
        return mean, std, 1.0

    with open(_STANDARDIZATION_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    mean_list = data.get("F0_FEATURE_MEAN", [0.0] * NUM_FEATURES)
    std_list = data.get("F0_FEATURE_STD", [1.0] * NUM_FEATURES)
    t_psv = float(data.get("T_PSV", 1.0))

    if len(mean_list) != NUM_FEATURES or len(std_list) != NUM_FEATURES:
        raise RuntimeError(
            f"psv_standardization.json has {len(mean_list)} mean entries and "
            f"{len(std_list)} std entries; expected {NUM_FEATURES} each."
        )

    mean = np.array(mean_list, dtype=np.float32)
    std = np.array(std_list, dtype=np.float32)
    _log.debug("PSV standardization params loaded", T_PSV=t_psv, path=str(_STANDARDIZATION_PATH))
    return mean, std, t_psv


# ---------------------------------------------------------------------------
# Module-level load at import time
# spec: 10.6.1 lines 2660-2664 — "loader at startup"
# ---------------------------------------------------------------------------
try:
    WEIGHT_MATRIX: np.ndarray = _load_yaml_weights()
    F0_FEATURE_MEAN: np.ndarray
    F0_FEATURE_STD: np.ndarray
    T_PSV: float
    F0_FEATURE_MEAN, F0_FEATURE_STD, T_PSV = _load_standardization_params()
    _LOAD_ERROR: Optional[str] = None
except Exception as _e:
    # Do not crash at import; degrade gracefully
    _LOAD_ERROR = str(_e)
    WEIGHT_MATRIX = np.zeros((NUM_CLASSES, NUM_FEATURES), dtype=np.float32)
    F0_FEATURE_MEAN = np.zeros(NUM_FEATURES, dtype=np.float32)
    F0_FEATURE_STD = np.ones(NUM_FEATURES, dtype=np.float32)
    T_PSV = 1.0
    _log.error("PSV compatibility config load failed; using zero matrix", error=str(_e))


def standardize_features(raw_features: np.ndarray) -> np.ndarray:
    """Standardize the 26 raw features using F0 calibration params.

    Applies: standardized[i] = clip((raw[i] - mean[i]) / (std[i] + 1e-6), -3, 3)

    # spec: 10.6.2 lines 2715-2716

    Args:
        raw_features: float32 array [26].

    Returns:
        float32 array [26], values clipped to [-3, 3].
    """
    standardized = (raw_features - F0_FEATURE_MEAN) / (F0_FEATURE_STD + 1e-6)
    return np.clip(standardized, -3.0, 3.0).astype(np.float32)


def compute_compatibility_scores(standardized_features: np.ndarray) -> np.ndarray:
    """Convert standardized features to 6 compatibility scores via softmax.

    Algorithm (spec 10.6.2 lines 2708-2726):
      1. raw_scores = WEIGHT_MATRIX @ standardized  — shape [6]
      2. logits = raw_scores / T_PSV
      3. exp = exp(logits - logits.max())  — numerically stable
      4. compatibility = exp / exp.sum()   — shape [6]

    Args:
        standardized_features: float32 array [26], already clipped ±3.

    Returns:
        float32 array [6] in canonical order:
        [foliar, septoria, late_blight, ylcv, mosaic, healthy].
        Sums to 1.0.
    """
    # spec: 10.6.2 line 2719
    raw_scores = WEIGHT_MATRIX @ standardized_features  # [6]

    # spec: 10.6.2 lines 2723-2725 — softmax with temperature, numerically stable
    logits = raw_scores / max(float(T_PSV), 1e-6)
    exp = np.exp(logits - logits.max())
    compatibility = (exp / exp.sum()).astype(np.float32)

    return compatibility
