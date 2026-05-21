"""Per-class collapse detection.

Spec reference: Part 7 adaptive action #4 + Round B refinement.

ORIGINAL criterion (v1, v2): if any class's field F1 stays below
COLLAPSE_F1_THRESHOLD for N consecutive epochs, trigger STOP.

ROUND B refinement: also require that the class has NEVER exceeded
`recovery_floor` field F1 (default 0.50) in this run. This distinguishes
"genuinely never learned" classes (true collapse) from "oscillating"
classes that recover (e.g. YLCV's 0.000↔1.000 binary swing in v2).

State persists across epochs. Call update(per_class_field) each epoch;
returns the offending class name if collapse is detected, else None.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional


class CollapseMonitor:
    def __init__(self,
                 num_classes: int,
                 class_names: List[str],
                 threshold: float = 0.05,
                 consecutive: int = 5,
                 recovery_floor: Optional[float] = 0.50):
        """
        Args:
            threshold: F1 below this counts as "low" for the streak.
            consecutive: epochs in a row required to trigger collapse.
            recovery_floor: if set, also require that the class has NEVER
                exceeded this value in this run. None disables the recovery
                check (legacy v1/v2 behavior).
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.threshold = threshold
        self.consecutive = consecutive
        self.recovery_floor = recovery_floor
        self._low_streak: dict = defaultdict(int)
        # Round B: track per-class peak field F1 across the whole run.
        # If a class has ever exceeded recovery_floor, we treat any future
        # below-threshold streak as oscillation rather than permanent collapse.
        self._peak_f1: dict = defaultdict(float)

    def update(self, per_class_field: dict) -> Optional[str]:
        """
        Args:
            per_class_field: {class_name: field_f1}
        Returns:
            Name of the class that hit the collapse threshold, or None.
        """
        for cls in self.class_names:
            f1 = per_class_field.get(cls, 0.0)
            # Track all-time peak per class.
            if f1 > self._peak_f1[cls]:
                self._peak_f1[cls] = f1

            if f1 < self.threshold:
                self._low_streak[cls] += 1
            else:
                self._low_streak[cls] = 0

            if self._low_streak[cls] >= self.consecutive:
                # Round B gate: only flag as collapse if the class has NEVER
                # demonstrated it can learn (peak <= recovery_floor).
                if (self.recovery_floor is None
                        or self._peak_f1[cls] <= self.recovery_floor):
                    return cls
                # Otherwise treat as oscillation; do NOT return halt.
        return None

    def state(self) -> dict:
        return {
            'low_streak': dict(self._low_streak),
            'peak_f1': dict(self._peak_f1),
        }
