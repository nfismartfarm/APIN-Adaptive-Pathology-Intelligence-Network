# training/07_evaluate_validation.py
"""
Tier-1 validation set evaluation. Writes comprehensive report to reports/.
[FIX GAP 39,58] Re-runs full inference on val set — reads no stored scalars.
[FIX GAP 54] Accepts --yes flag for pipeline compatibility.
"""

import os
import sys
import argparse
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, SOURCE_MAP, SEV_LABELS, REPORTS,
    CLASS_NAMES, NUM_CLASSES, DISEASE_THRESH, CLASS_TO_IDX, CROP_FROM_IDX
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform
from training.metrics import compute_ece


def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()

    model = load_model_for_inference(BEST_MODEL, DEVICE)

    T_disease = 1.0
    if os.path.exists(TEMP_PATH):
        t = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease = float(t.get('T_disease', 1.0))

    df  = pd.read_csv(SOURCE_MAP)
    val = df[df['split'] == 'val'].to_dict('records')
    for r in val:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

    sev_labels = load_severity_labels()
    ds  = PlantDiseaseDataset(val, get_eval_transform(), sev_labels)
    dl  = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

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
    cm           = confusion_matrix(d_true.argmax(axis=1), d_binary.argmax(axis=1))

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'validation_report_{ts}.md')

    lines = [
        '# Validation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'Val images: {len(val)}  |  T_disease: {T_disease:.4f}',
        '',
        '## Summary',
        f'- Macro F1 (disease): {macro_f1:.4f}',
        f'- Crop accuracy: {crop_acc:.4f}',
        f'- ECE (calibration error): {ece:.4f}',
        '',
        '## Per-Class F1',
        '| Class | F1 |',
        '|-------|-----|',
    ]
    for cls, f1 in zip(CLASS_NAMES, per_class_f1):
        flag = ' ← LOW' if f1 < 0.40 else ''
        lines.append(f'| {cls} | {f1:.4f}{flag} |')

    lines += [
        '',
        '## Confusion Matrix (argmax actual vs argmax predicted)',
        '```',
        str(cm),
        '```',
        '',
        '## Acceptance Status',
    ]
    if macro_f1 >= 0.50:
        lines.append(f'✓ PASS — macro F1 {macro_f1:.4f} >= 0.50')
    else:
        lines.append(f'✗ FAIL — macro F1 {macro_f1:.4f} < 0.50. Training needs improvement.')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Validation report: {path}")
    print(f"Macro F1={macro_f1:.4f}  Crop acc={crop_acc:.4f}  ECE={ece:.4f}")


if __name__ == '__main__':
    run_evaluation()
