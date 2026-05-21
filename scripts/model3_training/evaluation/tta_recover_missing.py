"""Recover the 8-view TTA runs that were lost when the original 8-view run
crashed on a Windows PermissionError during atomic JSON write.

What the original 8-view run completed and saved:
  - val_ceiling_tta8           ✓
  - val_conservative_tta8      ✓

What it computed but failed to save:
  - final_val_ceiling_tta8     (stdout: field=0.8768, overall=0.9412)

What it never reached:
  - final_val_conservative_tta8
  - val_conservative_no_tta
  - final_val_conservative_no_tta

This script re-runs the 3 missing 8-view items and merges them into the
existing tta_results.json. The no-TTA baselines from the 4-view file
(which completed cleanly) are also valid; they're identical regardless
of view count since no-TTA is a single forward pass.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from scripts.model3_training.evaluation.tta_eval import (
    ALL_VIEWS, load_v3_model, run_evaluation, _atomic_write_json,
)
from scripts.model3_training.data.model3_dataset import load_and_split_csv
from scripts.model3_training.model3_config import (
    CLASSES_10, CHECKPOINT_DIR, LOG_DIR,
)


def main():
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_path = LOG_DIR / 'tta_results.json'
    payload = json.loads(out_path.read_text(encoding='utf-8'))
    print(f"Loaded existing payload: {list(payload['runs'].keys())}")

    views8 = ALL_VIEWS[:8]
    model = load_v3_model(CHECKPOINT_DIR / 'model3_production_v3.pt', device)
    splits = load_and_split_csv()

    todo = [
        ('final_val', 'ceiling',      views8, 'final_val_ceiling_tta8'),
        ('final_val', 'conservative', views8, 'final_val_conservative_tta8'),
        ('val',       'conservative', None,   'val_conservative_no_tta'),
        ('final_val', 'conservative', None,   'final_val_conservative_no_tta'),
    ]

    for split_name, mode, views, key in todo:
        if key in payload['runs']:
            print(f"  SKIP {key} (already in payload)")
            continue
        print(f"\n>>> Running {key}")
        result = run_evaluation(
            model, splits[split_name], split_name, mode,
            list(CLASSES_10), device, views=views,
        )
        payload['runs'][key] = result
        _atomic_write_json(out_path, payload)
        print(f"  Saved → {out_path}")

    print(f"\nFinal run keys: {sorted(payload['runs'].keys())}")


if __name__ == '__main__':
    main()
