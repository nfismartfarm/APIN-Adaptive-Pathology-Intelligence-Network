"""
Re-export shim for task-card path compatibility.

Spec canonical path: tomato_sandbox/severity/grader.py (S21 line 6537).
Task card path: tomato_sandbox/severity/severity.py.
DEC-044 Decision 1: both paths work; this file is the task-card shim.
"""
from tomato_sandbox.severity.grader import SeverityResult, compute_severity

__all__ = ["SeverityResult", "compute_severity"]
