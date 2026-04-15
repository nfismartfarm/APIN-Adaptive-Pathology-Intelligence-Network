# training/losses.py
"""
All custom loss functions and training utilities for Phase 2.

FocalBCELoss — replaces standard BCE, down-weights easy negatives
CORALLoss    — domain adaptation aligning source/target covariance
ArcFaceLoss  — angular margin between confusable disease embeddings
DeiTDistillationLoss — teacher→student knowledge transfer
EMAModel     — exponential moving average of model weights
cutmix_batch — CutMix data augmentation utility
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from app.config import (
    FOCAL_GAMMA, LABEL_SMOOTHING, ARCFACE_SCALE, ARCFACE_MARGIN,
    ARCFACE_IN_FEATURES, CORAL_LAMBDA, NUM_CLASSES,
    DISTILLATION_TEMP, DISTILLATION_ALPHA, EMA_DECAY,
    RANDAUGMENT_N, RANDAUGMENT_M_DEFAULT, RANDAUGMENT_M_THIN,
)


class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross Entropy for multi-label disease classification.

    Adds modulating factor (1 - p_t)^gamma to each BCE term.
    When gamma=0: standard BCE. When gamma=2: easy example at p=0.9
    contributes (1-0.9)^2 = 0.01x the weight of a hard misclassified example.

    Gamma warmup: ramps from 0 to target gamma over the first epoch to
    prevent instability before the model has learned basic features.
    """
    def __init__(self, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.current_gamma = 0.0

    def set_gamma(self, gamma):
        """Called each step/epoch for gamma warmup."""
        self.current_gamma = min(gamma, self.gamma)

    def forward(self, logits, targets):
        """
        logits:  [N, C] raw (before sigmoid)
        targets: [N, C] binary or float labels in [0, 1]
        Returns: scalar loss
        """
        # Label smoothing: shift {0,1} to {eps/C, 1-eps+eps/C}
        eps = self.smoothing / NUM_CLASSES
        smooth = targets.float() * (1.0 - self.smoothing) + eps

        bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction='none')

        if self.current_gamma > 0:
            probs = torch.sigmoid(logits.detach())
            p_t = probs * targets.float() + (1 - probs) * (1 - targets.float())
            focal_weight = (1.0 - p_t).clamp(min=1e-6) ** self.current_gamma
            bce = focal_weight * bce

        return bce.mean()


class CORALLoss(nn.Module):
    """
    CORrelation ALignment domain adaptation loss.

    Minimises Frobenius norm difference between source and target domain
    covariance matrices in the 256-dim feature space.

    CRITICAL: features must NOT be detached. CORAL trains the backbone to
    produce domain-invariant features by backpropagating through it.
    """
    def __init__(self, lambda_coral=CORAL_LAMBDA):
        super().__init__()
        self.lambda_coral = lambda_coral

    def forward(self, source_features, target_features):
        """
        source_features: [N_s, D] — PlantVillage features WITH gradients
        target_features: [N_t, D] — PlantDoc features WITH gradients
        Returns: scalar CORAL loss (0.0 if either set has < 2 samples)
        """
        n_s = source_features.shape[0]
        n_t = target_features.shape[0]

        if n_s < 2 or n_t < 2:
            return torch.tensor(0.0, device=source_features.device,
                                requires_grad=False)

        d = source_features.shape[1]

        src = source_features - source_features.mean(dim=0, keepdim=True)
        tgt = target_features - target_features.mean(dim=0, keepdim=True)

        cov_s = (src.T @ src) / (n_s - 1)
        cov_t = (tgt.T @ tgt) / (n_t - 1)

        loss = torch.sum((cov_s - cov_t) ** 2) / (4.0 * d * d)
        return self.lambda_coral * loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace auxiliary loss enforcing angular margin between disease embeddings.

    self.weight [C, in_features] is a learnable parameter that MUST be included
    in the optimizer param_groups or it will never be updated.
    """
    def __init__(self, in_features=ARCFACE_IN_FEATURES, num_classes=NUM_CLASSES,
                 scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features, labels_onehot):
        """
        features:      [N, 256] pooled features (before CLN)
        labels_onehot: [N, C] binary disease labels (hard, not CutMix mixed)
        Returns: scalar ArcFace loss
        """
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        cosine = features_norm @ weight_norm.T   # [N, C]
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-6))

        phi = cosine * self.cos_m - sine * self.sin_m   # cos(theta + m)
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # Apply margin to positive classes, use cosine for negatives
        logits = torch.where(labels_onehot.bool(), phi, cosine)
        logits = self.scale * logits

        return F.binary_cross_entropy_with_logits(
            logits, labels_onehot.float(), reduction='mean'
        )


class DeiTDistillationLoss(nn.Module):
    """
    DeiT knowledge distillation: teacher (EfficientNetV2-S) → student (Swin-Tiny).

    Loss = alpha * focal_hard + (1-alpha) * T^2 * KL(student_soft || teacher_soft)

    Applied ONLY to source domain (non-PlantDoc) images via is_source_domain_mask.
    When CutMix is active, pass scale=0.5 to reduce distillation weight.
    """
    def __init__(self, alpha=DISTILLATION_ALPHA, temperature=DISTILLATION_TEMP):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.focal = FocalBCELoss(gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING)

    def forward(self, student_disease_logits, distillation_logits,
                teacher_disease_logits, hard_labels, is_source_domain_mask,
                scale=1.0):
        """
        student_disease_logits: [N, 23] from MoE disease head
        distillation_logits:    [N, 23] from model.distillation_head
        teacher_disease_logits: [N, 23] from frozen teacher
        hard_labels:            [N, 23] binary ground truth (or CutMix float)
        is_source_domain_mask:  [N] bool, True = source domain
        scale: 0.5 when CutMix active, 1.0 otherwise

        Returns: (total_loss, hard_loss_item, kl_loss_item)
        """
        self.focal.set_gamma(FOCAL_GAMMA)
        hard_loss = self.focal(student_disease_logits, hard_labels)

        kl_loss_value = 0.0
        if is_source_domain_mask.sum() > 0:
            src_student = distillation_logits[is_source_domain_mask]
            src_teacher = teacher_disease_logits[is_source_domain_mask]

            with torch.no_grad():
                teacher_soft = F.softmax(src_teacher / self.temperature, dim=-1)

            student_log_soft = F.log_softmax(
                src_student / self.temperature, dim=-1
            )

            kl = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
            kl = kl * (self.temperature ** 2)
            kl_loss_value = kl.item()
        else:
            kl = torch.tensor(0.0, device=student_disease_logits.device)

        total = self.alpha * hard_loss + (1 - self.alpha) * kl * scale
        return total, hard_loss.item(), kl_loss_value


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Updated after every optimizer.step(). decay=0.999 means the EMA
    represents approximately the last 1000 training steps.
    """
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def update(self, model):
        """Call after every optimizer.step()."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name].detach() +
                    (1.0 - self.decay) * param.data.detach()
                )

    def apply(self, model):
        """Apply EMA weights for validation. Call restore() after."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore training weights after EMA validation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def cutmix_batch(images, labels, alpha=1.0):
    """
    CutMix: replace a rectangular region in each image with the corresponding
    region from a paired image. Labels are blended proportionally.

    Returns: (mixed_images, mixed_labels_float, lam, rand_idx)
    """
    N, C_img, H, W = images.shape
    lam = float(np.random.beta(alpha, alpha))

    rand_idx = torch.randperm(N, device=images.device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

    # Recalculate actual lambda from box area
    lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)

    mixed_labels = lam * labels.float() + (1.0 - lam) * labels[rand_idx].float()

    return mixed, mixed_labels, lam, rand_idx


def get_randaugment_transform(is_thin_batch, n=RANDAUGMENT_N,
                               m_default=RANDAUGMENT_M_DEFAULT,
                               m_thin=RANDAUGMENT_M_THIN):
    """Returns RandAugment transform with appropriate magnitude for the batch."""
    from torchvision.transforms import v2 as T
    magnitude = m_thin if is_thin_batch else m_default
    return T.RandAugment(num_ops=n, magnitude=magnitude)
