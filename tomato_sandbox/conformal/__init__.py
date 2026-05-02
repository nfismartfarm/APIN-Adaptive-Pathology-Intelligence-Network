"""
Conformal prediction sub-package.

Re-exports the public API from conformal.py so callers can use either:
  from tomato_sandbox.conformal import ConformalResult, compute_conformal_set
  from tomato_sandbox.conformal.conformal import ConformalResult, compute_conformal_set

# spec: section 13 — conformal prediction sets
# DEC-033: sub-package + re-export shim when spec and plan disagree on layout
# DEC-040: canonical at tomato_sandbox/conformal/conformal.py
"""

from __future__ import annotations

from tomato_sandbox.conformal.conformal import (  # noqa: F401
    ConformalResult,
    compute_conformal_set,
    compute_conformal_tau,
    load_tau,
)
