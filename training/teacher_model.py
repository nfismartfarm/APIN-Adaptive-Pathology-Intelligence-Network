"""
Teacher model loading for Phase 2 DeiT knowledge distillation.

Reconstructs the OLD EfficientNetV2-S architecture to load models/best_model.pt.
This must match the exact module names/shapes from the previous PlantDiseaseModel.

The teacher is NEVER updated during training. All parameters frozen.
Used exclusively in Phase 2 to generate soft target disease predictions.
"""

import os
import torch
import torch.nn as nn
import timm


# OLD architecture constants (matching the teacher's training config)
_TEACHER_BACKBONE = 'tf_efficientnetv2_s.in21k_ft_in1k'
_TEACHER_FPN_IN = [64, 160, 256]
_TEACHER_FPN_OUT = 256
_TEACHER_POOLED = 256
_TEACHER_EMB = 64
_TEACHER_HIDDEN = 256
_TEACHER_DROPOUT = 0.3
_TEACHER_CLASSES = 23
_TEACHER_CROPS = 4


class _OldFPN(nn.Module):
    """Exact reconstruction of the old FPN class."""
    def __init__(self):
        super().__init__()
        self.lat_p3 = nn.Conv2d(_TEACHER_FPN_IN[0], _TEACHER_FPN_OUT, 1)
        self.lat_p4 = nn.Conv2d(_TEACHER_FPN_IN[1], _TEACHER_FPN_OUT, 1)
        self.lat_p5 = nn.Conv2d(_TEACHER_FPN_IN[2], _TEACHER_FPN_OUT, 1)
        self.out_p3 = nn.Conv2d(_TEACHER_FPN_OUT, _TEACHER_FPN_OUT, 3, padding=1)
        self._upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, p3_feat, p4_feat, p5_feat):
        p5 = self.lat_p5(p5_feat)
        p4 = self.lat_p4(p4_feat) + self._upsample(p5)
        p3 = self.lat_p3(p3_feat) + self._upsample(p4)
        return self.out_p3(p3)


class _OldCropClassifier(nn.Module):
    """Exact reconstruction of the old CropClassifier."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_TEACHER_POOLED, _TEACHER_HIDDEN)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(_TEACHER_DROPOUT)
        self.embed = nn.Linear(_TEACHER_HIDDEN, _TEACHER_EMB)
        self.fc2 = nn.Linear(_TEACHER_EMB, _TEACHER_CROPS)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        emb = self.embed(x)
        return self.fc2(emb), emb


class _OldDiseaseHead(nn.Module):
    """Exact reconstruction of the old FiLM-conditioned DiseaseHead."""
    def __init__(self):
        super().__init__()
        self.film_gamma = nn.Linear(_TEACHER_EMB, _TEACHER_POOLED)
        self.film_beta = nn.Linear(_TEACHER_EMB, _TEACHER_POOLED)
        self.fc1 = nn.Linear(_TEACHER_POOLED, _TEACHER_HIDDEN)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(_TEACHER_DROPOUT)
        self.fc2 = nn.Linear(_TEACHER_HIDDEN, _TEACHER_CLASSES)

    def forward(self, pooled, crop_emb):
        gamma = torch.sigmoid(self.film_gamma(crop_emb))
        beta = self.film_beta(crop_emb)
        x = pooled * gamma + beta
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class _OldSeverityHead(nn.Module):
    """Exact reconstruction of the old SeverityHead."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_TEACHER_POOLED, _TEACHER_HIDDEN)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(_TEACHER_DROPOUT)
        self.fc2 = nn.Linear(_TEACHER_HIDDEN, 3)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class TeacherModel(nn.Module):
    """
    Exact reconstruction of the OLD PlantDiseaseModel (EfficientNetV2-S).
    Module names must match the state_dict keys in models/best_model.pt.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            _TEACHER_BACKBONE, pretrained=False,
            features_only=True, out_indices=(2, 3, 4)
        )
        self.fpn = _OldFPN()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.crop_classifier = _OldCropClassifier()
        self.disease_head = _OldDiseaseHead()
        self.severity_head = _OldSeverityHead()

    def forward(self, x):
        """Returns disease_logits [N, 23] for soft target generation."""
        features = self.backbone(x)
        p3, p4, p5 = features[0], features[1], features[2]
        fused = self.fpn(p3, p4, p5)
        pooled = self.gap(fused).flatten(1)
        crop_logits, crop_emb = self.crop_classifier(pooled)
        disease_logits = self.disease_head(pooled, crop_emb)
        return disease_logits


def load_teacher_model(teacher_path, device):
    """
    Load the EfficientNetV2-S teacher from models/best_model.pt.
    All parameters frozen. Used exclusively for Phase 2 DeiT distillation.
    """
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f'Teacher model not found at {teacher_path}. '
            f'This file is required for Phase 2 DeiT distillation.'
        )

    teacher = TeacherModel()
    checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict',
                     checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    load_result = teacher.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        # Filter out expected missing keys (distillation head etc.)
        real_missing = [k for k in load_result.missing_keys
                       if 'distillation' not in k and 'att_pool' not in k and 'cln' not in k]
        if real_missing:
            print(f'Teacher WARNING: missing keys: {real_missing[:5]}...')
    if load_result.unexpected_keys:
        print(f'Teacher: {len(load_result.unexpected_keys)} unexpected keys (ignored)')

    # Freeze all parameters
    for param in teacher.parameters():
        param.requires_grad = False

    teacher.eval()
    teacher.to(device)
    print(f'Teacher model loaded from {teacher_path} ({os.path.getsize(teacher_path)/1e6:.1f}MB)')
    return teacher
