"""
tomato_sandbox.signals.psv — PSV (Plant Symptom Visual) signal sub-package.

Public API re-exported from this __init__ per DEC-033 (sub-package + re-export
shim policy). Callers may use either:
    from tomato_sandbox.signals.psv import compute_signal_c, SignalCResult
    from tomato_sandbox.signals.psv.psv import compute_signal_c, SignalCResult

# spec: 10.10 lines 2867-2880 — "PSV is split into multiple files for clarity"
"""

from .psv import compute_signal_c, SignalCResult  # noqa: F401

__all__ = ["compute_signal_c", "SignalCResult"]
