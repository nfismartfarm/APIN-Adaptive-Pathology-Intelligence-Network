"""
Task-card alias shim for tomato_sandbox/orchestrator/orchestrator.py.

Canonical implementation is at tomato_sandbox/orchestrator/pipeline.py
(spec Section 21.1 line 6608 names the canonical file "pipeline.py").

This file exists so that imports from the task-card path
  from tomato_sandbox.orchestrator.orchestrator import predict_single
also work, per DEC-033 (sub-package + alias shim pattern).

DEC-042: alias shim per DEC-033.
"""

from tomato_sandbox.orchestrator.pipeline import (  # noqa: F401
    PipelineContext,
    predict_single,
    predict_multi,
    _make_sentinel_classifier_result,
    _make_fallback_conformal,
    _apply_nan_guard,
    _signal_a_to_dict,
    _signal_b_to_dict,
    _signal_c_to_dict,
    _classifier_to_dict,
    _conformal_to_dict,
    _iqa_to_dict,
    _image_hash,
    _error_response,
    _build_pipeline_result,
)

__all__ = [
    "PipelineContext",
    "predict_single",
    "predict_multi",
]
