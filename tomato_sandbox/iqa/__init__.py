# This file marks the directory as a Python package.
# Re-exports the public IQA API so callers can use:
#   from tomato_sandbox.iqa import compute_iqa, IQAResult
# This matches the flat-file import path implied by spec Section 6.6 line 1374.
from tomato_sandbox.iqa.iqa import IQAResult, compute_iqa

__all__ = ["IQAResult", "compute_iqa"]
