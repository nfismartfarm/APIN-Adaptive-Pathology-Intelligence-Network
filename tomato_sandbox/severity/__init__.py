# This file marks the directory as a Python package.
# Re-exports the public surface of severity/grader.py
# spec: section 17 (Severity grading), DEC-044

from tomato_sandbox.severity.grader import SeverityResult, compute_severity

__all__ = ["SeverityResult", "compute_severity"]
