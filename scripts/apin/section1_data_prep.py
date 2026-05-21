"""
Section 1 — Data Pipeline Preparation for Model 2 APIN.

What this does:
  1A. Adds `is_recomposed` column (all False) to Model 2's unified CSV.
      Model 2 has NO recomposed images (verified against PHASE0_LOG.md:308 —
      9,705 recomposed images belong to Model 3 only). This column is a
      defensive scaffold so Model 3 APIN can reuse the same cache/pipeline
      code with the flag in a different state.
  1B. Verifies split structure matches the prompt's expected counts.
  1C. Writes canonical MODEL2_CLASS_ORDER to scripts/apin/constants.py.

Outputs:
  - data/specialist/model2/model2_unified_source_map.csv (updated in place
    with backup to .bak)
  - scripts/apin/constants.py (canonical class-ordering import target)
  - scripts/apin/section1_fingerprint.json (audit trail)
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging to both file and terminal
APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
APIN_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section1_data_prep_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section1")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(fmt)
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)


CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "model2_unified_source_map.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "specialist" / "model2" / "split_indices.json"
CONSTANTS_PATH = APIN_DIR / "constants.py"
FINGERPRINT_PATH = APIN_DIR / f"section1_fingerprint_{TIMESTAMP}.json"


# Expected split counts per prompt
EXPECTED_SPLIT_COUNTS = {
    "train": 6126,
    "val_and_soup": 1350,
    "final_val": 1080,
    "conformal": 450,
}

# Canonical 9-class ordering for Model 2 APIN
MODEL2_CLASS_ORDER = [
    "okra_yvmv",
    "okra_powdery_mildew",
    "okra_cercospora",
    "okra_enation",
    "okra_healthy",
    "brassica_black_rot",
    "brassica_downy_mildew",
    "brassica_alternaria",
    "brassica_healthy",
]


def step_1a_add_is_recomposed_column() -> dict:
    """Add is_recomposed=False column to Model 2 CSV.
    Backs up existing CSV to .bak before writing.
    Verifies no recomposed images exist (they belong to Model 3 only).
    Returns audit dict with counts.
    """
    logger.info("=" * 70)
    logger.info("STEP 1A — Add is_recomposed column")
    logger.info("=" * 70)

    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded CSV: {CSV_PATH}")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")

    # Sanity checks before modification
    assert len(df) == 9006, f"Expected 9006 rows, got {len(df)}"
    assert "class_name" in df.columns
    assert "source_dataset" in df.columns
    assert "image_path" in df.columns

    # Verify no recomposed images slipped through — defensive check
    recomp_by_path = df["image_path"].astype(str).str.contains(
        "recomp|synth", case=False, na=False
    ).sum()
    recomp_by_source = df["source_dataset"].astype(str).str.contains(
        "recomp|synth|capsicum_recomp", case=False, na=False
    ).sum()
    logger.info(f"  recomposed paths detected: {recomp_by_path}")
    logger.info(f"  recomposed sources detected: {recomp_by_source}")
    assert recomp_by_path == 0, "Unexpected recomposed images by path in Model 2"
    assert recomp_by_source == 0, "Unexpected recomposed images by source in Model 2"

    if "is_recomposed" in df.columns:
        logger.info("  is_recomposed column already exists — skipping add")
        added = False
        existing_values = df["is_recomposed"].value_counts().to_dict()
        logger.info(f"  existing values: {existing_values}")
    else:
        # Backup original CSV
        backup_path = CSV_PATH.with_suffix(f".csv.bak_{TIMESTAMP}")
        shutil.copy2(CSV_PATH, backup_path)
        logger.info(f"  Backup created at: {backup_path.name}")

        # Add the column — all False for Model 2 (no recomposed images)
        df["is_recomposed"] = False
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"  Added is_recomposed=False to all {len(df)} rows")
        logger.info(f"  CSV re-saved to: {CSV_PATH.name}")
        added = True

    # Reload and verify
    df_check = pd.read_csv(CSV_PATH)
    assert "is_recomposed" in df_check.columns
    # pandas reads False as bool — verify no True values
    false_count = (df_check["is_recomposed"] == False).sum()
    true_count = (df_check["is_recomposed"] == True).sum()
    logger.info(f"  Verification: False={false_count}, True={true_count}")
    assert true_count == 0, f"Expected 0 True values, got {true_count}"
    assert false_count == len(df_check), "Count mismatch"

    return {
        "rows": int(len(df)),
        "is_recomposed_added_this_run": bool(added),
        "is_recomposed_false_count": int(false_count),
        "is_recomposed_true_count": int(true_count),
        "recomposed_by_path": int(recomp_by_path),
        "recomposed_by_source": int(recomp_by_source),
    }


def step_1b_verify_splits() -> dict:
    """Verify the 4-way split matches expected counts."""
    logger.info("=" * 70)
    logger.info("STEP 1B — Verify split structure")
    logger.info("=" * 70)

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    actual_counts = {k: len(v) for k, v in splits.items()}
    logger.info(f"  Split keys: {list(splits.keys())}")
    logger.info(f"  Actual counts: {actual_counts}")
    logger.info(f"  Expected counts: {EXPECTED_SPLIT_COUNTS}")

    # Verify
    mismatches = []
    for k, expected in EXPECTED_SPLIT_COUNTS.items():
        actual = actual_counts.get(k, 0)
        if actual != expected:
            mismatches.append(f"{k}: expected {expected}, got {actual}")

    # Verify all 4 expected keys present
    missing_keys = set(EXPECTED_SPLIT_COUNTS.keys()) - set(splits.keys())
    if missing_keys:
        mismatches.append(f"missing keys: {missing_keys}")

    # Verify non-overlap between splits (critical for the 4 caches)
    all_indices = []
    for k, v in splits.items():
        all_indices.extend(v)
    total_indices = len(all_indices)
    unique_indices = len(set(all_indices))
    logger.info(f"  Total split indices: {total_indices}")
    logger.info(f"  Unique split indices: {unique_indices}")
    if total_indices != unique_indices:
        mismatches.append(
            f"split overlap: {total_indices - unique_indices} duplicate indices"
        )

    # Total should equal CSV length (9006)
    if total_indices != 9006:
        mismatches.append(
            f"total split coverage: expected 9006, got {total_indices}"
        )

    assert not mismatches, f"Split verification failures: {mismatches}"
    logger.info("  All split checks PASS")

    return {
        "actual_counts": actual_counts,
        "expected_counts": EXPECTED_SPLIT_COUNTS,
        "total_indices": int(total_indices),
        "unique_indices": int(unique_indices),
        "mismatches": [],
    }


def step_1c_write_constants_module() -> dict:
    """Write scripts/apin/constants.py as the canonical import target.
    Every downstream APIN script imports MODEL2_CLASS_ORDER from here.
    """
    logger.info("=" * 70)
    logger.info("STEP 1C — Write canonical constants module")
    logger.info("=" * 70)

    # Verify the class order matches what's in app/config_model2.py
    from app.config_model2 import CLASS_NAMES as CONFIG_M2_CLASSES

    logger.info(f"  app/config_model2.py CLASS_NAMES: {list(CONFIG_M2_CLASSES)}")
    logger.info(f"  MODEL2_CLASS_ORDER (APIN canonical): {MODEL2_CLASS_ORDER}")

    if list(CONFIG_M2_CLASSES) != MODEL2_CLASS_ORDER:
        logger.error(
            "Class order mismatch between app/config_model2.py and "
            "MODEL2_CLASS_ORDER defined here. Aborting."
        )
        raise AssertionError(
            "Class order mismatch — would corrupt all downstream caches"
        )
    logger.info("  Class orderings match exactly — PASS")

    content = f'''"""Canonical constants for APIN Model 2 ensemble.
Every APIN script imports from this module. Do NOT redefine these elsewhere.
Written by scripts/apin/section1_data_prep.py at {TIMESTAMP}.
"""

# The 9 classes in Model 2's training order. All 4 signal caches, the
# stacking MLP, the MoE gate, the reliability matrix, and the server
# output all use THIS ordering. Mismatches would silently corrupt the
# entire ensemble — never define this elsewhere.
MODEL2_CLASS_ORDER = {MODEL2_CLASS_ORDER!r}

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
'''

    # Write atomically
    tmp_path = CONSTANTS_PATH.with_suffix(".py.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(CONSTANTS_PATH)
    logger.info(f"  Wrote: {CONSTANTS_PATH}")

    # Verify the write by re-importing
    import importlib

    spec = importlib.util.spec_from_file_location(
        "apin_constants_check", CONSTANTS_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.MODEL2_CLASS_ORDER == MODEL2_CLASS_ORDER
    assert mod.NUM_CLASSES == 9
    assert len(mod.EN_TO_M2_INDEX_MAP) == 9
    assert mod.EN_DROPPED_INDEX == 8
    logger.info("  Re-import verification: PASS")

    return {
        "constants_path": str(CONSTANTS_PATH.relative_to(PROJECT_ROOT)),
        "model2_class_order": MODEL2_CLASS_ORDER,
        "en_to_m2_index_map": [0, 1, 2, 3, 4, 5, 6, 7, 9],
        "num_classes": 9,
    }


def write_fingerprint(audit_1a: dict, audit_1b: dict, audit_1c: dict):
    """Write section1 fingerprint JSON."""
    fingerprint = {
        "section": 1,
        "timestamp": TIMESTAMP,
        "project_root": str(PROJECT_ROOT),
        "csv_path": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        "splits_path": str(SPLITS_PATH.relative_to(PROJECT_ROOT)),
        "constants_path": str(CONSTANTS_PATH.relative_to(PROJECT_ROOT)),
        "step_1a": audit_1a,
        "step_1b": audit_1b,
        "step_1c": audit_1c,
    }
    with open(FINGERPRINT_PATH, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"Fingerprint written: {FINGERPRINT_PATH}")


def main() -> int:
    logger.info("=" * 70)
    logger.info("APIN SECTION 1 -- Data Pipeline Preparation")
    logger.info("=" * 70)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Timestamp   : {TIMESTAMP}")

    audit_1a = step_1a_add_is_recomposed_column()
    audit_1b = step_1b_verify_splits()
    audit_1c = step_1c_write_constants_module()

    write_fingerprint(audit_1a, audit_1b, audit_1c)

    logger.info("=" * 70)
    logger.info("APIN SECTION 1 -- COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
