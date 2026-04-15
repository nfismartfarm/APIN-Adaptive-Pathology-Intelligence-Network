# training/08_evaluate_tier2_plantdoc.py
"""
Tier-2 PlantDoc evaluation. Run ONCE after all training is final.
[FIX GAP 55] Temperature scaling applied (same as production inference).
[FIX GAP 54] --yes flag for non-interactive pipeline execution.
"""

import os
import sys
import argparse
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import f1_score

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, REPORTS, SOURCE_MAP,
    CLASS_NAMES, CLASS_TO_IDX, CROP_FROM_IDX,
    DISEASE_THRESH, TIER2_MIN_F1, PLANTDOC_CLASS_MAP, SEV_LABELS
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform


def run_tier2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()

    if not args.yes:
        try:
            confirm = input(
                "\nTIER-2 EVALUATION: Run ONCE only. No model changes after this.\n"
                "Type 'yes' to proceed: "
            ).strip().lower()
        except EOFError:
            confirm = 'yes'
        if confirm != 'yes':
            print("Aborted.")
            sys.exit(0)

    model = load_model_for_inference(BEST_MODEL, DEVICE)
    T_disease = 1.0
    if os.path.exists(TEMP_PATH):
        t = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease = float(t.get('T_disease', 1.0))

    df = pd.read_csv(SOURCE_MAP)
    # Phase 0 placed PlantDoc eval images with source_dataset='plantdoc_eval'
    # (not split='plantdoc'). Filter by source_dataset to find them.
    plantdoc_records = df[
        df['source_dataset'].str.contains('plantdoc_eval', case=False, na=False)
    ].to_dict('records')
    # Fallback: try split='plantdoc' for backward compatibility
    if not plantdoc_records:
        plantdoc_records = df[df['split'] == 'plantdoc'].to_dict('records')
    for r in plantdoc_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)
    plantdoc_records = [r for r in plantdoc_records if r['class_idx'] >= 0]

    if not plantdoc_records:
        print("No PlantDoc records in source_map.csv. Run 01_prepare_data.py after downloading PlantDoc.")
        return
    print(f"Found {len(plantdoc_records)} PlantDoc evaluation images")

    sev_labels = load_severity_labels()
    ds = PlantDiseaseDataset(plantdoc_records, get_eval_transform(), sev_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_d_probs = []
    all_d_true  = []

    model.eval()
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in dl:
            _, d_log, _ = model(images.to(DEVICE))
            d_probs = torch.sigmoid(d_log / T_disease).cpu()
            all_d_probs.append(d_probs)
            all_d_true.append(d_lab)

    d_probs  = torch.cat(all_d_probs).numpy()
    d_true   = torch.cat(all_d_true).numpy()
    d_binary = (d_probs > DISEASE_THRESH).astype(int)

    mappable_classes = list(set(PLANTDOC_CLASS_MAP.values()))
    mappable_idx     = [CLASS_NAMES.index(c) for c in mappable_classes if c in CLASS_NAMES]
    d_probs_m  = d_probs[:, mappable_idx]
    d_true_m   = d_true[:,  mappable_idx]
    d_binary_m = d_binary[:, mappable_idx]

    per_class_f1 = f1_score(d_true_m, d_binary_m, average=None, zero_division=0)
    macro_f1     = float(np.mean(per_class_f1))

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'tier2_plantdoc_{ts}.md')

    lines = [
        '# Tier-2 PlantDoc Evaluation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'PlantDoc images evaluated: {len(plantdoc_records)}',
        f'T_disease applied: {T_disease:.4f}',
        '',
        '## Results (mappable classes only)',
        f'Macro F1: {macro_f1:.4f}  (acceptance threshold: {TIER2_MIN_F1})',
        '',
        '| Class | F1 |',
        '|-------|-----|',
    ]
    eval_classes = [CLASS_NAMES[i] for i in mappable_idx]
    for cls, f1 in zip(eval_classes, per_class_f1):
        lines.append(f'| {cls} | {f1:.4f} |')

    lines += ['', '## Decision']
    if macro_f1 >= TIER2_MIN_F1:
        lines.append(f'✓ PASS — Model is deployment-ready at tier-2.')
    else:
        lines.append(f'✗ FAIL — Gap analysis required before deployment.')
        for cls, f1 in zip(eval_classes, per_class_f1):
            if f1 < 0.40:
                lines.append(f'  - {cls}: F1={f1:.3f} — needs more diverse training data')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Tier-2 report: {path}")
    print(f"Tier-2 macro F1: {macro_f1:.4f}  {'PASS' if macro_f1 >= TIER2_MIN_F1 else 'FAIL'}")


if __name__ == '__main__':
    run_tier2()
