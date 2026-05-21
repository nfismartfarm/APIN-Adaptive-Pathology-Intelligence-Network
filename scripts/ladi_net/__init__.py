"""LADI-Net: Lesion-Aware Domain-Invariant Network.

Phase 0 — data preparation and validation.
Phase 1 — ABMIL head initialization.
Phase 2 — full LoRA training.
Phase 3 — prototype bank construction and calibration.
Phase 4 — model soup and final evaluation.

This package contains only the LADI-Net-specific code; shared components
(CSVs, InSPyReNet wrapper, CLAHE) live in existing project directories and
are imported without modification.
"""
