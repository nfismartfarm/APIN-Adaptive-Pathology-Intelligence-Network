# training/09_evaluate_tier3_kerala.py

import os
import sys
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, SOURCE_MAP, SEV_LABELS, REPORTS,
    CLASS_NAMES, NUM_CLASSES, DISEASE_THRESH, CLASS_TO_IDX, CROP_FROM_IDX,
    TIER3_MIN_ACC, TIER3_MIN_IMGS, TIER3_MIN_CLS
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform


def run_tier3():
    model = load_model_for_inference(BEST_MODEL, DEVICE)
    T_disease = 1.0
    if os.path.exists(TEMP_PATH):
        t = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease = float(t.get('T_disease', 1.0))

    df = pd.read_csv(SOURCE_MAP)
    kerala_records = df[df['split'] == 'kerala'].to_dict('records')
    for r in kerala_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)
    kerala_records = [r for r in kerala_records if r['class_idx'] >= 0]

    if len(kerala_records) < TIER3_MIN_IMGS:
        print(f"Only {len(kerala_records)} Kerala images. Need {TIER3_MIN_IMGS}.")
        print("Use: python tools/add_kerala_image.py --path img.jpg --class class_name")
        return

    sev_labels = load_severity_labels()
    ds = PlantDiseaseDataset(kerala_records, get_eval_transform(), sev_labels)
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

    class_counts = Counter(r['class_name'] for r in kerala_records)
    results      = {}
    overall_pass = True

    for cls in CLASS_NAMES:
        cnt = class_counts.get(cls, 0)
        if cnt < TIER3_MIN_CLS:
            continue
        idx  = CLASS_NAMES.index(cls)
        mask = d_true[:, idx].astype(bool)
        if not mask.any():
            continue
        acc = float(accuracy_score(d_true[mask, idx], d_binary[mask, idx]))
        results[cls] = {'count': cnt, 'accuracy': acc}
        if acc < TIER3_MIN_ACC:
            overall_pass = False

    os.makedirs(REPORTS, exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS, f'tier3_kerala_{ts}.md')

    lines = [
        '# Tier-3 Kerala Field Evaluation Report',
        f'Generated: {datetime.datetime.now().isoformat()}',
        f'Kerala images: {len(kerala_records)}',
        '',
        '## Per-Class Results',
        '| Class | Count | Accuracy | Pass? |',
        '|-------|-------|----------|-------|',
    ]
    for cls, r in results.items():
        passed = '✓' if r['accuracy'] >= TIER3_MIN_ACC else '✗'
        lines.append(f"| {cls} | {r['count']} | {r['accuracy']:.3f} | {passed} |")

    lines += ['', '## Overall Result']
    if overall_pass:
        lines.append('✓ PASS — Project is DEPLOYMENT-VALIDATED for Kerala.')
    else:
        lines.append('✗ FAIL — Some classes below accuracy threshold. Collect more Kerala images.')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Tier-3 report: {path}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == '__main__':
    run_tier3()
