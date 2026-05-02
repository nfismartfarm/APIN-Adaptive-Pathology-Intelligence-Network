# This file marks the directory as a Python package.
# Re-exports the public surface of multi_image/aggregator.py
# spec: section 18 (Multi-image input), DEC-044

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
