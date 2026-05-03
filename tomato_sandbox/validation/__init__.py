"""
tomato_sandbox.validation — F.0 calibration and validation sub-package.

Re-exports the public API of:
  - fit_calibration.py (Component B, DEC-052)
  - run_f0.py (Component A, DEC-053)

per DEC-033 (sub-package + __init__ re-export) and DEC-052 / DEC-053.

# spec: section 29 lines 8105-8243 — F.0 validation suite
# spec: section 13.5 lines 3583-3619 — τ derivation
# spec: section 12.8 lines 3375-3407 — Platt scaling
# spec: section 17.3 lines 5966-5982 — per-disease severity thresholds
# spec: section 8.4 lines 1691-1701 — chilli_leakage threshold
"""

from tomato_sandbox.validation.fit_calibration import (  # noqa: F401
    fit_conformal_tau,
    fit_platt_scaling,
    fit_severity_thresholds,
    fit_chilli_leakage_threshold,
    run_full_calibration,
    SEVERITY_DEFAULTS,
)

from tomato_sandbox.validation.run_f0 import (  # noqa: F401
    run_f0_validation,
)

__all__ = [
    # Component B (DEC-052)
    "fit_conformal_tau",
    "fit_platt_scaling",
    "fit_severity_thresholds",
    "fit_chilli_leakage_threshold",
    "run_full_calibration",
    "SEVERITY_DEFAULTS",
    # Component A (DEC-053)
    "run_f0_validation",
]
