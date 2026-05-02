"""
Flat-path re-export shim for TTA controller.

Spec 11.7 line 3103 specifies: "tomato_sandbox/tta.py defines TTAReport,
should_trigger_tta, apply_tta".
Task card DEC-033 pattern: canonical module at tomato_sandbox/signals/tta.py;
this shim re-exports the public API so spec-cited imports resolve.

DEC-037: path discrepancy documented — spec says tomato_sandbox/tta.py,
task card says tomato_sandbox/signals/tta.py.  Resolution: canonical at
signals/tta.py; this shim satisfies the spec-cited flat path.
"""

from tomato_sandbox.signals.tta import (  # noqa: F401
    TTAReport,
    should_trigger_tta,
    build_augmentations,
    apply_augmentation,
    aggregate_views,
    jensen_shannon_divergence,
    apply_tta,
)

__all__ = [
    "TTAReport",
    "should_trigger_tta",
    "build_augmentations",
    "apply_augmentation",
    "aggregate_views",
    "jensen_shannon_divergence",
    "apply_tta",
]
