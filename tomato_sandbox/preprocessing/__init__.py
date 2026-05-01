"""
tomato_sandbox.preprocessing — image preprocessing pipelines.

Re-exports the three public preprocessing functions and the color-constancy
helper so callers can use either:

    from tomato_sandbox.preprocessing import preprocess_for_v3
    from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3

Both forms resolve to the same implementation.

spec: section 7.6 line 1563 — "tomato_sandbox/preprocessing.py defines all
three preprocessing functions plus shades_of_gray."
(Sub-package layout adopted per DEC-031; public API is identical to the
flat-file spec.)
"""

from tomato_sandbox.preprocessing.preprocess import (
    preprocess_for_lora,
    preprocess_for_psv,
    preprocess_for_v3,
    shades_of_gray,
)

__all__ = [
    "preprocess_for_v3",
    "preprocess_for_lora",
    "preprocess_for_psv",
    "shades_of_gray",
]
