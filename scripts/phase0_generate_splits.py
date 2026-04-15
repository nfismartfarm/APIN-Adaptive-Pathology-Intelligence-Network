"""
Phase 0 Step 0.11: Generate Source-Aware Data Splits

Creates train/val/soup/final_val/conformal splits for each model using
source-aware composite stratification keys.

Per MASTER_PLAN Section 3.3:
- Composite key: class_name + '_' + source_bucket
- Recomposed images forced to training only
- Conformal indices excluded from all other splits (including self-distillation)
- Model 2: 4-way split (val+soup merged for thin class statistical power)
- Model 3: 5-way split
- Router: 3-way split

Per architecture_update.md:
- StratifiedGroupKFold for classes with 3+ distinct sources (optional enhancement)
- Source buckets: scidb_original, field_verified, lab_non_scidb, recomposed

Output: data/specialist/{model}/split_indices.json
"""
import os
import sys
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_utils import generate_splits

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')


def run_splits():
    print('=' * 70)
    print('PHASE 0 STEP 0.11: GENERATE SOURCE-AWARE DATA SPLITS')
    print('=' * 70)
    print()

    # ── MODEL 2: 4-way split ──────────────────────────────────────────────
    print('--- MODEL 2 (9 classes, 4-way split) ---')
    m2_csv = ROOT / 'data' / 'specialist' / 'model2' / 'model2_unified_source_map.csv'
    m2_df = pd.read_csv(m2_csv)
    print(f'  Total images: {len(m2_df)}')

    # IMPORTANT: train MUST be LAST — generate_splits gives the last split
    # everything remaining, so train gets the bulk of the data
    m2_split_config = {
        'conformal': 0.05,
        'final_val': 0.12,
        'val_and_soup': 0.15,
        'train': 0.68,  # last = gets remainder
    }

    m2_splits = generate_splits(
        m2_df,
        split_config=m2_split_config,
        seed=42,
        recomposed_sources=['scidb_recomposed', 'capsicum_recomposed'],
    )

    m2_out = ROOT / 'data' / 'specialist' / 'model2' / 'split_indices.json'
    # Convert to serializable format (int, not numpy)
    m2_save = {k: [int(i) for i in v] for k, v in m2_splits.items()}
    with open(m2_out, 'w') as f:
        json.dump(m2_save, f)
    print(f'  Saved to: {m2_out}')
    print()

    # ── MODEL 3: 5-way split ──────────────────────────────────────────────
    print('--- MODEL 3 (10 classes, 5-way split) ---')
    m3_csv = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
    m3_df = pd.read_csv(m3_csv)
    print(f'  Total images: {len(m3_df)}')

    m3_split_config = {
        'conformal': 0.05,
        'final_val': 0.10,
        'soup_selection': 0.07,
        'val': 0.10,
        'train': 0.68,  # last = gets remainder
    }

    m3_splits = generate_splits(
        m3_df,
        split_config=m3_split_config,
        seed=42,
        recomposed_sources=['scidb_recomposed', 'capsicum_recomposed'],
    )

    m3_out = ROOT / 'data' / 'specialist' / 'model3' / 'split_indices.json'
    m3_save = {k: [int(i) for i in v] for k, v in m3_splits.items()}
    with open(m3_out, 'w') as f:
        json.dump(m3_save, f)
    print(f'  Saved to: {m3_out}')
    print()

    # ── ROUTER: 3-way split ───────────────────────────────────────────────
    print('--- ROUTER (4 crops, 3-way split) ---')
    r_csv = ROOT / 'data' / 'specialist' / 'router' / 'router_unified_source_map.csv'
    r_df = pd.read_csv(r_csv)
    # Router CSV has 'crop' instead of 'class_name' — create class_name alias
    if 'class_name' not in r_df.columns:
        r_df['class_name'] = r_df['crop']
    print(f'  Total images: {len(r_df)}')

    r_split_config = {
        'conformal': 0.10,
        'val': 0.15,
        'train': 0.75,  # last = gets remainder
    }

    r_splits = generate_splits(
        r_df,
        split_config=r_split_config,
        seed=42,
        recomposed_sources=['scidb_recomposed', 'capsicum_recomposed'],
    )

    r_out = ROOT / 'data' / 'specialist' / 'router' / 'split_indices.json'
    r_save = {k: [int(i) for i in v] for k, v in r_splits.items()}
    with open(r_out, 'w') as f:
        json.dump(r_save, f)
    print(f'  Saved to: {r_out}')
    print()

    # ── VERIFICATION ──────────────────────────────────────────────────────
    print('=' * 70)
    print('SPLIT VERIFICATION')
    print('=' * 70)

    for name, splits, total in [('Model 2', m2_splits, len(m2_df)),
                                  ('Model 3', m3_splits, len(m3_df)),
                                  ('Router', r_splits, len(r_df))]:
        print(f'\n{name}:')
        total_in_splits = sum(len(v) for v in splits.values())
        print(f'  Total in splits: {total_in_splits} / {total} '
              f'({total_in_splits/total*100:.1f}%)')

        # Check no overlap
        all_idx = set()
        overlap_found = False
        for split_name, indices in splits.items():
            overlap = all_idx & set(indices)
            if overlap:
                print(f'  OVERLAP: {split_name} shares {len(overlap)} indices!')
                overlap_found = True
            all_idx.update(indices)

        if not overlap_found:
            print(f'  No overlap between splits: OK')

        # Check conformal is disjoint from training
        if 'conformal' in splits and 'train' in splits:
            conf_in_train = set(splits['conformal']) & set(splits['train'])
            if conf_in_train:
                print(f'  CRITICAL: {len(conf_in_train)} conformal indices in training!')
            else:
                print(f'  Conformal disjoint from training: OK')

    print()
    print('SPLIT GENERATION COMPLETE')
    print('NOTE: Re-run after background recomposition (Step 0.10) finishes')
    print('      to include recomposed images in the training splits.')


if __name__ == '__main__':
    run_splits()
