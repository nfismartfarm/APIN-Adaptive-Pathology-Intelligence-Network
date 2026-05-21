"""Canonical constants for APIN Model 2 ensemble.
Every APIN script imports from this module. Do NOT redefine these elsewhere.
Written by scripts/apin/section1_data_prep.py at 20260416_234452.
"""

# The 9 classes in Model 2's training order. All 4 signal caches, the
# stacking MLP, the MoE gate, the reliability matrix, and the server
# output all use THIS ordering. Mismatches would silently corrupt the
# entire ensemble — never define this elsewhere.
MODEL2_CLASS_ORDER = ['okra_yvmv', 'okra_powdery_mildew', 'okra_cercospora', 'okra_enation', 'okra_healthy', 'brassica_black_rot', 'brassica_downy_mildew', 'brassica_alternaria', 'brassica_healthy']

NUM_CLASSES = 9

# EfficientNet's 10-class ordering with brassica_clubroot at index 8.
# Mapping from EN index to Model 2 index for Signal 2 cache generation.
EFFICIENTNET_CLASS_ORDER = [
    'okra_yvmv', 'okra_powdery_mildew', 'okra_cercospora', 'okra_enation',
    'okra_healthy', 'brassica_black_rot', 'brassica_downy_mildew',
    'brassica_alternaria', 'brassica_clubroot', 'brassica_healthy',
]
# EN indices to keep when reordering to MODEL2_CLASS_ORDER (drops index 8)
EN_TO_M2_INDEX_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 9]
# The dropped index from EN (brassica_clubroot is quarantined in Model 2)
EN_DROPPED_INDEX = 8
EN_DROPPED_CLASS = 'brassica_clubroot'

# The two "failure classes" where Model 2 catastrophically fails on field photos
# (2-20% confidence for the correct class, documented in
# architecture_claude_decisions.md Decision 11 and probe_results JSON).
FAILURE_CLASSES = ('brassica_black_rot', 'okra_cercospora')
FAILURE_CLASS_INDICES = (5, 2)
