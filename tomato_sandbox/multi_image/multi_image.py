"""
Re-export shim for task-card path compatibility.

Spec canonical path: tomato_sandbox/multi_image/aggregator.py (S21 line 6539).
Task card path: tomato_sandbox/multi_image/multi_image.py.
DEC-044 Decision 1: both paths work; this file is the task-card shim.
"""
from tomato_sandbox.multi_image.aggregator import (
    PerImageInput,
    PerImageSummary,
    AggregatedResult,
    aggregate_multi_image,
)

__all__ = [
    "PerImageInput",
    "PerImageSummary",
    "AggregatedResult",
    "aggregate_multi_image",
]
