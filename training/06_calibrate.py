# training/06_calibrate.py
"""
Temperature scaling calibration on validation set.
Fits T_disease, T_crop, T_severity separately using LBFGS.
Saves: models/temperature.pt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

# wandb disabled on Windows (service port timeout)
os.environ['WANDB_MODE'] = 'disabled'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# wandb stub to avoid import hanging
class _WandbStub:
    def init(self, *a, **kw): pass
    def log(self, *a, **kw): pass
    def finish(self, *a, **kw): pass
wandb = _WandbStub()

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, TEMP_INIT, SOURCE_MAP, SEV_LABELS,
    CLASS_TO_IDX, CROP_FROM_IDX, MODELS, NUM_CLASSES,
    WANDB_PROJECT, WANDB_CONFIG
)
from app.model import load_model_for_inference
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform
from training.metrics import compute_ece


def calibrate():
    print("=" * 60)
    print("06_CALIBRATE — Temperature scaling")
    print("=" * 60)

    model = load_model_for_inference(BEST_MODEL, DEVICE)

    df = pd.read_csv(SOURCE_MAP)
    val_records = df[df['split'] == 'val'].to_dict('records')
    for r in val_records:
        r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
        r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

    sev_labels = load_severity_labels()
    ds = PlantDiseaseDataset(val_records, get_eval_transform(), sev_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # Collect all logits
    all_d_logits = []
    all_c_logits = []
    all_s_logits = []
    all_d_labels = []
    all_c_labels = []
    all_s_labels = []

    model.eval()
    with torch.no_grad():
        for images, d_lab, c_lab, s_lab in dl:
            c_log, d_log, s_log = model(images.to(DEVICE))
            all_d_logits.append(d_log.cpu())
            all_c_logits.append(c_log.cpu())
            all_s_logits.append(s_log.cpu())
            all_d_labels.append(d_lab)
            all_c_labels.append(c_lab)
            all_s_labels.append(s_lab)

    d_logits = torch.cat(all_d_logits)
    c_logits = torch.cat(all_c_logits)
    s_logits = torch.cat(all_s_logits)
    d_labels = torch.cat(all_d_labels)
    c_labels = torch.cat(all_c_labels)
    s_labels = torch.cat(all_s_labels)

    # ECE before calibration
    d_probs_before = torch.sigmoid(d_logits).numpy()
    ece_before = compute_ece(d_probs_before, d_labels.numpy())
    print(f"ECE before calibration: {ece_before:.4f}")

    # Fit temperatures using LBFGS in log-space to guarantee T > 0.
    # Parameterize as log_T, then T = exp(log_T). Since exp() > 0 always,
    # the optimizer cannot produce negative temperatures.
    # Clamp final T to [0.1, 10.0] to prevent extreme values.
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()
    log_init = float(np.log(TEMP_INIT))

    # Fit T_disease
    log_T_disease = nn.Parameter(torch.tensor(log_init))
    optimizer_d = torch.optim.LBFGS([log_T_disease], lr=0.01, max_iter=50)

    def closure_d():
        optimizer_d.zero_grad()
        T = torch.exp(log_T_disease)
        loss = bce(d_logits / T, d_labels.float())
        loss.backward()
        return loss

    optimizer_d.step(closure_d)

    # Fit T_crop
    log_T_crop = nn.Parameter(torch.tensor(log_init))
    optimizer_c = torch.optim.LBFGS([log_T_crop], lr=0.01, max_iter=50)

    def closure_c():
        optimizer_c.zero_grad()
        T = torch.exp(log_T_crop)
        loss = ce(c_logits / T, c_labels)
        loss.backward()
        return loss

    optimizer_c.step(closure_c)

    # Fit T_severity
    log_T_severity = nn.Parameter(torch.tensor(log_init))
    optimizer_s = torch.optim.LBFGS([log_T_severity], lr=0.01, max_iter=50)

    def closure_s():
        optimizer_s.zero_grad()
        T = torch.exp(log_T_severity)
        loss = ce(s_logits / T, s_labels)
        loss.backward()
        return loss

    optimizer_s.step(closure_s)

    # Convert from log-space and clamp to safe range [0.1, 10.0]
    T_disease_val  = torch.exp(log_T_disease.detach()).clamp(0.1, 10.0)
    T_crop_val     = torch.exp(log_T_crop.detach()).clamp(0.1, 10.0)
    T_severity_val = torch.exp(log_T_severity.detach()).clamp(0.1, 10.0)

    # ECE after calibration
    d_probs_after = torch.sigmoid(d_logits / T_disease_val).numpy()
    ece_after = compute_ece(d_probs_after, d_labels.numpy())

    T_d = float(T_disease_val)
    T_c = float(T_crop_val)
    T_s = float(T_severity_val)

    print(f"T_disease:  {T_d:.4f}")
    print(f"T_crop:     {T_c:.4f}")
    print(f"T_severity: {T_s:.4f}")
    print(f"ECE after:  {ece_after:.4f}")

    # Save
    os.makedirs(MODELS, exist_ok=True)
    torch.save({
        'T_disease'  : T_d,
        'T_crop'     : T_c,
        'T_severity' : T_s,
        'ece_before' : ece_before,
        'ece_after'  : ece_after,
        'num_classes': NUM_CLASSES,  # for stale calibration detection
    }, TEMP_PATH)
    print(f"Saved temperature values to {TEMP_PATH}")

    # Wandb logging
    wandb.init(
        project=WANDB_PROJECT,
        name='calibration',
        config={**WANDB_CONFIG, 'phase': 'calibration',
                'T_disease': T_d, 'T_crop': T_c, 'T_severity': T_s},
    )
    wandb.log({
        'calibration/T_disease' : T_d,
        'calibration/T_crop'    : T_c,
        'calibration/T_severity': T_s,
        'calibration/ece_before': ece_before,
        'calibration/ece_after' : ece_after,
    })
    wandb.finish()


if __name__ == '__main__':
    calibrate()
