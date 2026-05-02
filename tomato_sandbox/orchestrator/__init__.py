"""
Re-export shim for tomato_sandbox.orchestrator package.

Canonical implementation is at tomato_sandbox/orchestrator/pipeline.py
(spec Section 21.1 line 6608).

This __init__.py makes `from tomato_sandbox.orchestrator import predict_single`
work at the package level.

DEC-042: shim placement per DEC-033 pattern.
"""

from tomato_sandbox.orchestrator.pipeline import (
    PipelineContext,
    predict_single,
    predict_multi,
)

__all__ = [
    "PipelineContext",
    "predict_single",
    "predict_multi",
]
