# training/metrics.py

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from app.config import NUM_CLASSES, CLASS_NAMES, DISEASE_THRESH


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error over n_bins equal-width probability bins."""
    probs  = np.array(probs).flatten()
    labels = np.array(labels).flatten()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += np.abs(acc - conf) * mask.sum() / n
    return float(ece)


def compute_multilabel_pos_weights(train_records):
    """Correct multi-label pos_weight computation. NOT sklearn compute_class_weight."""
    from app.config import MAX_POS_WEIGHT
    d_labels = torch.zeros(len(train_records), NUM_CLASSES)
    for i, r in enumerate(train_records):
        idx = r.get('class_idx', -1)
        if 0 <= idx < NUM_CLASSES:
            d_labels[i, idx] = 1.0
    n_total   = float(len(train_records))
    n_pos     = d_labels.sum(dim=0).clamp(min=1.0)
    n_neg     = n_total - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=MAX_POS_WEIGHT)
    return pos_weight


def compute_all_metrics(model, data_loader, pos_weight, device, phase='phase2_full'):
    """Runs full inference pass on data_loader and computes all metrics."""
    model.eval()
    from training.loss import compute_loss

    all_d_preds = []
    all_d_true  = []
    all_c_preds = []
    all_c_true  = []
    total_loss  = 0.0

    with torch.no_grad():
        for batch in data_loader:
            if phase == 'phase1_cached':
                pooled, crop_emb, d_lab, c_lab, s_lab = batch
                pooled   = pooled.to(device)
                c_log, crop_emb_out = model.crop_classifier(pooled)
                d_log    = model.disease_head(pooled, crop_emb_out)
                s_log    = model.severity_head(pooled)
            else:
                images, d_lab, c_lab, s_lab = batch
                c_log, d_log, s_log = model(images.to(device))

            d_lab = d_lab.to(device)
            c_lab = c_lab.to(device)
            s_lab = s_lab.to(device)

            loss, _ = compute_loss(c_log, d_log, s_log, c_lab, d_lab, s_lab,
                                   pos_weight.to(device))
            total_loss += loss.item()

            all_d_preds.append(torch.sigmoid(d_log).cpu().numpy())
            all_d_true.append(d_lab.cpu().numpy())
            all_c_preds.append(c_log.argmax(dim=1).cpu().numpy())
            all_c_true.append(c_lab.cpu().numpy())

    d_preds = np.concatenate(all_d_preds)
    d_true  = np.concatenate(all_d_true)
    c_preds = np.concatenate(all_c_preds)
    c_true  = np.concatenate(all_c_true)

    d_binary = (d_preds > DISEASE_THRESH).astype(int)

    per_class_f1 = f1_score(d_true, d_binary, average=None, zero_division=0)
    macro_f1     = float(np.mean(per_class_f1))
    crop_acc     = float(accuracy_score(c_true, c_preds))
    ece          = compute_ece(d_preds, d_true)

    metrics = {
        'val/macro_f1' : macro_f1,
        'val/crop_acc' : crop_acc,
        'val/ece'      : ece,
        'val/loss'     : total_loss / max(len(data_loader), 1),
    }
    for i, cls in enumerate(CLASS_NAMES):
        metrics[f'val/f1_{cls}'] = float(per_class_f1[i])

    return metrics


def warn_on_thin_classes(val_metrics, epoch, threshold=0.40):
    """Prints a warning for any class with per-class F1 < threshold after epoch 3."""
    if epoch < 3:
        return
    for cls in CLASS_NAMES:
        f1 = val_metrics.get(f'val/f1_{cls}', 1.0)
        if f1 < threshold:
            print(f"  Warning: Thin class {cls} F1={f1:.3f} < {threshold}")
