"""
Tier assignment package for the Tomato 3-Signal system.

Spec section: 14.8 lines 4026-4033
DEC-033: re-export shim so both import paths work:
  from tomato_sandbox.tier import assign_tier, TierAssignment
  from tomato_sandbox.tier.tier_assignment import assign_tier, TierAssignment
"""

from .tier_assignment import TierAssignment, assign_tier

__all__ = ["assign_tier", "TierAssignment"]
