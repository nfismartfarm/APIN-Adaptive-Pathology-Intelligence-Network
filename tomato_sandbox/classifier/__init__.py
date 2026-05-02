"""
tomato_sandbox.classifier — Hierarchical stacking classifier sub-package.

Spec section: 12 (Hierarchical classifier), lines 3145–3505.

Public surface re-exported from sub-modules per DEC-033 module layout policy:
  - ``ClassifierResult``      (spec S12.10)
  - ``build_classifier_input`` (spec S12.2)
  - ``compute_classifier``    (spec S12.11)
  - ``JSD_SENTINEL``          (spec S12.2)
  - ``jensen_shannon_divergence`` (spec S12.2)
  - ``CLASSIFIER_FEATURE_MEAN``, ``CLASSIFIER_FEATURE_STD`` (spec S12.2)

Both import paths work per DEC-033:
  from tomato_sandbox.classifier import ClassifierResult
  from tomato_sandbox.classifier.hierarchical_classifier import ClassifierResult
"""

from tomato_sandbox.classifier.feature_builder import (
    JSD_SENTINEL,
    CLASSIFIER_FEATURE_MEAN,
    CLASSIFIER_FEATURE_STD,
    build_classifier_input,
    jensen_shannon_divergence,
    load_feature_standardization,
)
from tomato_sandbox.classifier.hierarchical_classifier import (
    ClassifierResult,
    NUM_FINAL_CLASSES,
    IDX_FOLIAR,
    IDX_SEPTORIA,
    IDX_LATE_BLIGHT,
    IDX_YLCV,
    IDX_MOSAIC,
    IDX_HEALTHY,
    IDX_OOD,
    compute_classifier,
    _stage1_forward,
    _stage2_forward,
    _soft_route,
    _apply_platt,
)

__all__ = [
    # Feature builder
    "JSD_SENTINEL",
    "CLASSIFIER_FEATURE_MEAN",
    "CLASSIFIER_FEATURE_STD",
    "build_classifier_input",
    "jensen_shannon_divergence",
    "load_feature_standardization",
    # Hierarchical classifier
    "ClassifierResult",
    "NUM_FINAL_CLASSES",
    "IDX_FOLIAR",
    "IDX_SEPTORIA",
    "IDX_LATE_BLIGHT",
    "IDX_YLCV",
    "IDX_MOSAIC",
    "IDX_HEALTHY",
    "IDX_OOD",
    "compute_classifier",
    "_stage1_forward",
    "_stage2_forward",
    "_soft_route",
    "_apply_platt",
]
