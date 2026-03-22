# training/loss.py

import torch
import torch.nn as nn
from app.config import (
    LOSS_W_CROP, LOSS_W_DISEASE, LOSS_W_SEVERITY,
    LABEL_SMOOTH, NUM_CLASSES
)


def compute_loss(crop_logits, disease_logits, severity_logits,
                 crop_labels, disease_labels, severity_labels, pos_weight):
    """
    Combined loss for three heads.
    Returns: (total_loss, loss_dict)
    CRITICAL: Returns a TUPLE. Unpack before calling .backward().
    """
    crop_loss  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)(
        crop_logits, crop_labels
    )

    pos_w = pos_weight.to(disease_logits.device)
    bce   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dis_loss = bce(disease_logits, disease_labels.float())

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
