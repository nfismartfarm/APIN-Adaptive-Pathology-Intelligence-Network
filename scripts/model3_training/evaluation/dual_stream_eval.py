"""Dual-stream (field vs lab) validation evaluation.

Spec reference: Part 7.

Runs ONE inference pass over the val loader. Uses ground-truth crop labels
(crop_mode_groundtruth) — NOT mode-2 uncertain — because we want the ceiling
estimate per spec Fix 5.

Returns a dict with every field the trainer and the final report need.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def dual_stream_evaluate(model: torch.nn.Module,
                         val_dl: torch.utils.data.DataLoader,
                         device: torch.device,
                         class_names: List[str]) -> dict:
    model.eval()
    preds, labels, is_field_flags = [], [], []
    for batch in val_dl:
        imgs = batch['image'].to(device, non_blocking=True)
        crop_mode = batch['crop_mode_groundtruth'].to(device, non_blocking=True)
        out = model(imgs, crop_mode=crop_mode, domain_labels=None)
        p = out['logits'].argmax(dim=1).cpu()
        preds.append(p)
        labels.append(batch['label'].cpu())
        is_field_flags.append(batch['is_field'].cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    is_field = torch.cat(is_field_flags).numpy().astype(bool)
    n_class_ids = list(range(len(class_names)))

    def _f1(p, l):
        if len(p) == 0:
            return 0.0
        return float(f1_score(l, p, average='macro', zero_division=0, labels=n_class_ids))

    def _per_class(p, l):
        if len(p) == 0:
            return {c: 0.0 for c in class_names}
        s = f1_score(l, p, average=None, zero_division=0, labels=n_class_ids)
        return {class_names[i]: float(s[i]) for i in range(len(class_names))}

    overall_acc = float(accuracy_score(labels, preds))
    overall_f1 = _f1(preds, labels)
    field_f1 = _f1(preds[is_field], labels[is_field])
    lab_f1 = _f1(preds[~is_field], labels[~is_field])

    pc_overall = _per_class(preds, labels)
    pc_field = _per_class(preds[is_field], labels[is_field])
    pc_lab = _per_class(preds[~is_field], labels[~is_field])

    # PDA Round 2 finding G5: gap is spurious for classes with zero samples
    # in either split (F1 collapses to 0 via zero_division=0, not NaN).
    # Count per-class occurrences in each split to mark gap as None in that case.
    class_ids = np.arange(len(class_names))
    field_counts = np.bincount(labels[is_field], minlength=len(class_names))
    lab_counts = np.bincount(labels[~is_field], minlength=len(class_names))
    gap = {}
    gap_warnings = []
    for i, c in enumerate(class_names):
        if field_counts[i] == 0 or lab_counts[i] == 0:
            gap[c] = None    # undefined — at least one split has no samples
            continue
        g = pc_lab[c] - pc_field[c]
        gap[c] = g
        if g > 0.20:
            gap_warnings.append(c)

    stop = 0.6 * field_f1 + 0.4 * overall_f1

    return {
        'overall_acc': overall_acc,
        'overall_f1': overall_f1,
        'field_f1': field_f1,
        'lab_f1': lab_f1,
        'stopping_metric': stop,
        'per_class_overall': pc_overall,
        'per_class_field': pc_field,
        'per_class_lab': pc_lab,
        'gap': gap,
        'gap_warnings': gap_warnings,
        'n_field_val': int(is_field.sum()),
        'n_lab_val': int((~is_field).sum()),
    }
