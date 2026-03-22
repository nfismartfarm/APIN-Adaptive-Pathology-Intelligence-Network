# training/10_evaluate_local_test.py
"""
Evaluates the 15% locked local test split. Run ONCE after tier-2.
[FIX GAP 37] Missing from v5 — now specified and implemented.
"""

import os
import sys
import argparse
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, SOURCE_MAP, SEV_LABELS, REPORTS,
    CLASS_NAMES, NUM_CLASSES, DISEASE_THRESH, CLASS_TO_IDX, CROP_FROM_IDX
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform
from training.metrics import compute_ece


def run_local_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()

    if not args.yes:
        try:
            confirm = input(
                "\nLOCAL TEST EVALUATION: Run only after tier-2 is complete.\n"
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
    test_records = df[df['split'] == 'test'].to_dict('records')
    for r in test_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

    if not test_records:
        print("No test records in source_map.csv.")
        return

    sev_labels = load_severity_labels()
    ds = PlantDiseaseDataset(test_records, get_eval_transform(), sev_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_d_probs = []
    all_d_true  = []
    all_c_preds = []
    all_c_true  = []

    model.eval()
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in dl:
            c_log, d_log, s_log = model(images.to(DEVICE))
            d_probs = torch.sigmoid(d_log / T_disease).cpu()
            all_d_probs.append(d_probs)
            all_d_true.append(d_lab)
            all_c_preds.append(c_log.argmax(dim=1).cpu())
            all_c_true.append(c_lab)

    d_probs  = torch.cat(all_d_probs).numpy()
    d_true   = torch.cat(all_d_true).numpy()
    c_preds  = torch.cat(all_c_preds).numpy()
    c_true   = torch.cat(all_c_true).numpy()
    d_binary = (d_probs > DISEASE_THRESH).astype(int)

    macro_f1     = float(f1_score(d_true, d_binary, average='macro', zero_division=0))
    per_class_f1 = f1_score(d_true, d_binary, average=None, zero_division=0)
    crop_acc     = float(accuracy_score(c_true, c_preds))
    ece          = compute_ece(d_probs, d_true)

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'local_test_report_{ts}.md')

    lines = [
        '# Local Test Set Evaluation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'Test images: {len(test_records)}  |  T_disease: {T_disease:.4f}',
        '',
        '## Summary',
        f'- Macro F1: {macro_f1:.4f}',
        f'- Crop accuracy: {crop_acc:.4f}',
        f'- ECE: {ece:.4f}',
        '',
        '## Per-Class F1',
        '| Class | F1 |',
        '|-------|-----|',
    ]
    for cls, f1_val in zip(CLASS_NAMES, per_class_f1):
        lines.append(f'| {cls} | {f1_val:.4f} |')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Local test report: {path}")
    print(f"Test macro F1={macro_f1:.4f}  Crop acc={crop_acc:.4f}  ECE={ece:.4f}")


if __name__ == '__main__':
    run_local_test()
