# training/loss.py

import torch
import torch.nn as nn
from app.config import (
    LOSS_W_CROP, LOSS_W_DISEASE, LOSS_W_SEVERITY,
    LABEL_SMOOTH, NUM_CLASSES, LABEL_SMOOTHING
)


def compute_loss(crop_logits, disease_logits, severity_logits,
                 crop_labels, disease_labels, severity_labels, pos_weight):
    """
    Combined loss for three heads.
    Returns: (total_loss, loss_dict)
    CRITICAL: Returns a TUPLE. Unpack before calling .backward().

    Label smoothing is applied to the disease head BCE targets only.
    For multi-label BCE: smoothed = target * (1 - eps) + 0.5 * eps
    This makes positive targets 0.95 and negative targets 0.05 (with eps=0.1),
    preventing overconfident predictions on thin classes.
    """
    crop_loss  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)(
        crop_logits, crop_labels
    )

    # Apply label smoothing to disease targets for multi-label BCE
    # Formula: smoothed = target * (1 - eps) + 0.5 * eps
    # 0.5 is correct for binary (not 1/K) because each output is independent
    disease_targets = disease_labels.float()
    disease_targets_smoothed = disease_targets * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

    pos_w = pos_weight.to(disease_logits.device)
    bce   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dis_loss = bce(disease_logits, disease_targets_smoothed)

    sev_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)(
        severity_logits, severity_labels
    )

    total = (LOSS_W_CROP * crop_loss
             + LOSS_W_DISEASE * dis_loss
             + LOSS_W_SEVERITY * sev_loss)

    loss_dict = {
        'loss/crop'    : crop_loss.item(),
        'loss/disease' : dis_loss.item(),
        'loss/severity': sev_loss.item(),
        'loss/total'   : total.item(),
    }
    return total, loss_dict
