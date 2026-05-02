"""
Re-export shim for tomato_sandbox.response package.

Per DEC-033 pattern (sub-package + re-export shim).
DEC-043: module placement per task card spec.

Public API surface:
    build_response  — spec section 16.1 lines 5643-5644
"""

from tomato_sandbox.response.response_builder import build_response

__all__ = ["build_response"]
