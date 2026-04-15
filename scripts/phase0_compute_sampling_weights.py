"""
Phase 0 Step 0.12: Compute Sampling Weights + Monte Carlo Verification

Per MASTER_PLAN Section 3.4:
- Router: target-based per-bucket weights (7,300 per crop, scidb cap 3,000)
- Model 2: ENS (beta=0.9999) + field 4x multiplier
- Model 3: ENS (beta=0.999) + field 4x + scidb cap (1,000/class/epoch)
- Recomposed images: full field weight, NOT counted in scidb cap

Monte Carlo verification: 30-epoch simulation, all buckets within 10% of targets.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_utils import compute_ens_class_weights, verify_sampling_weights

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')


def compute_router_weights():
    """Target-based weights for the crop router."""
    from app.config_router import UNDERSAMPLE_TOMATO_TO, FIELD_PHOTO_WEIGHT_MULTIPLIER

    csv_path = ROOT / 'data' / 'specialist' / 'router' / 'router_unified_source_map.csv'
    df = pd.read_csv(csv_path)

    TARGETS = {
        'okra':     {'total': 7300, 'field': 4500, 'lab': 2800, 'scidb': 0},
        'brassica': {'total': 7300, 'field': 6700, 'lab': 600,  'scidb': 0},
        'tomato':   {'total': 7300, 'field': 3000, 'lab': 1300, 'scidb': 3000},
        'chilli':   {'total': 7300, 'field': 7100, 'lab': 200,  'scidb': 0},
    }

    weights = np.ones(len(df))

    for crop, targets in TARGETS.items():
        crop_mask = df['crop'] == crop
        is_field = df['is_field_photo'].astype(str).str.lower().isin(['true'])
        is_scidb = df['source_dataset'].astype(str).str.contains('scidb', case=False, na=False)

        mask_field = crop_mask & is_field & ~is_scidb
        mask_scidb = crop_mask & is_scidb
        mask_lab = crop_mask & ~is_field & ~is_scidb

        n_field = mask_field.sum()
        n_scidb = mask_scidb.sum()
        n_lab = mask_lab.sum()

        if n_field > 0:
            weights[mask_field] = targets.get('field', 0) / n_field
        if n_scidb > 0:
            weights[mask_scidb] = targets.get('scidb', 0) / n_scidb
        if n_lab > 0:
            weights[mask_lab] = targets.get('lab', 0) / n_lab

    # Zero out any weights that target 0 (e.g., okra scidb)
    weights = np.maximum(weights, 0)

    return df, weights, TARGETS


def compute_model2_weights():
    """ENS + field multiplier weights for Model 2."""
    from app.config_model2 import (CLASS_NAMES, ENS_BETA, FIELD_PHOTO_MULTIPLIER)

    csv_path = ROOT / 'data' / 'specialist' / 'model2' / 'model2_unified_source_map.csv'
    df = pd.read_csv(csv_path)

    # Compute ENS per-class weights
    class_counts = [int((df['class_name'] == cls).sum()) for cls in CLASS_NAMES]
    ens_weights = compute_ens_class_weights(class_counts, beta=ENS_BETA)

    # Per-image weight = ENS weight for class * field multiplier
    weights = np.ones(len(df))
    is_field = df['is_field_photo'].astype(str).str.lower().isin(['true'])

    for i, cls in enumerate(CLASS_NAMES):
        cls_mask = df['class_name'] == cls
        weights[cls_mask] = float(ens_weights[i])
        # Field photo boost
        weights[cls_mask & is_field] *= FIELD_PHOTO_MULTIPLIER

    return df, weights


def compute_model3_weights():
    """ENS + field multiplier + scidb cap for Model 3."""
    from app.config_model3 import (
        CLASS_NAMES, ENS_BETA, FIELD_PHOTO_MULTIPLIER, SCIDB_CAP_PER_CLASS,
    )

    csv_path = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
    df = pd.read_csv(csv_path)

    class_counts = [int((df['class_name'] == cls).sum()) for cls in CLASS_NAMES]
    ens_weights = compute_ens_class_weights(class_counts, beta=ENS_BETA)

    weights = np.ones(len(df))
    is_field = df['is_field_photo'].astype(str).str.lower().isin(['true'])
    is_scidb = df['source_dataset'].astype(str).str.contains('scidb', case=False, na=False)
    is_recomp = df['source_dataset'].astype(str).str.contains('recomposed', case=False, na=False)

    for i, cls in enumerate(CLASS_NAMES):
        cls_mask = df['class_name'] == cls
        base_w = float(ens_weights[i])

        # Default ENS weight
        weights[cls_mask] = base_w

        # Field photo boost (includes recomposed)
        weights[cls_mask & is_field] *= FIELD_PHOTO_MULTIPLIER
        weights[cls_mask & is_recomp] *= FIELD_PHOTO_MULTIPLIER  # recomposed = field

        # Scidb cap: reduce weight so expected count per epoch <= SCIDB_CAP_PER_CLASS
        n_scidb = (cls_mask & is_scidb & ~is_recomp).sum()
        if n_scidb > SCIDB_CAP_PER_CLASS:
            scidb_factor = SCIDB_CAP_PER_CLASS / n_scidb
            weights[cls_mask & is_scidb & ~is_recomp] *= scidb_factor

    return df, weights


def main():
    print('=' * 70)
    print('PHASE 0 STEP 0.12: SAMPLING WEIGHT COMPUTATION + MONTE CARLO')
    print('=' * 70)
    print()

    # ── Router ────────────────────────────────────────────────────────────
    print('--- ROUTER WEIGHTS ---')
    r_df, r_weights, r_targets = compute_router_weights()
    print(f'  Weight range: [{r_weights.min():.4f}, {r_weights.max():.4f}]')
    print(f'  Non-zero weights: {(r_weights > 0).sum()} / {len(r_weights)}')

    # Save weights
    r_out = ROOT / 'data' / 'specialist' / 'router' / 'sampling_weights.json'
    with open(r_out, 'w') as f:
        json.dump(r_weights.tolist(), f)
    print(f'  Saved to: {r_out}')

    # Monte Carlo verification
    print('  Running Monte Carlo verification (30 epochs)...')
    r_ok = verify_sampling_weights(
        r_df, r_weights, r_targets,
        crop_col='crop', source_col='source_dataset',
        field_col='is_field_photo', epochs=30, samples_per_epoch=29200
    )
    print(f'  Monte Carlo: {"PASS" if r_ok else "FAIL — check targets"}')
    print()

    # ── Model 2 ───────────────────────────────────────────────────────────
    print('--- MODEL 2 WEIGHTS ---')
    m2_df, m2_weights = compute_model2_weights()
    print(f'  Weight range: [{m2_weights.min():.4f}, {m2_weights.max():.4f}]')
    m2_out = ROOT / 'data' / 'specialist' / 'model2' / 'sampling_weights.json'
    with open(m2_out, 'w') as f:
        json.dump(m2_weights.tolist(), f)
    print(f'  Saved to: {m2_out}')
    print()

    # ── Model 3 ───────────────────────────────────────────────────────────
    print('--- MODEL 3 WEIGHTS ---')
    m3_df, m3_weights = compute_model3_weights()
    print(f'  Weight range: [{m3_weights.min():.4f}, {m3_weights.max():.4f}]')
    m3_out = ROOT / 'data' / 'specialist' / 'model3' / 'sampling_weights.json'
    with open(m3_out, 'w') as f:
        json.dump(m3_weights.tolist(), f)
    print(f'  Saved to: {m3_out}')
    print()

    print('SAMPLING WEIGHT COMPUTATION COMPLETE')
    print('NOTE: Re-run after background recomposition adds new images')


if __name__ == '__main__':
    main()
