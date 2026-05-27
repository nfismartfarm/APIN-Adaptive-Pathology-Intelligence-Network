# app/model.py
"""
Phase 1 Architecture: Swin-Tiny + FPN + Attention Pooling + CLN + MoE + DeiT

Replaces: EfficientNetV2-S + FPN + GAP + FiLM + unified disease head

Key changes from previous architecture:
1. SwinTinyBackbone outputs NHWC — permuted to NCHW before FPN
2. AttentionPooling learns spatial importance (replaces GAP)
3. ConditionalLayerNorm per-channel per-crop (replaces FiLM scale+shift)
4. MixtureOfExpertsDiseaseHead with 4 crop-specific experts (replaces unified head)
5. DeiT distillation head for Phase 2 knowledge transfer from EfficientNetV2-S teacher

Forward returns: (crop_logits, disease_logits, severity_logits) — SAME tuple as before.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from app.config import (
    BACKBONE_NAME, FPN_IN_CH, FPN_OUT_CH, POOLED_DIM,
    CROP_EMB_DIM, HEAD_HIDDEN_DIM, DROPOUT_P,
    NUM_CLASSES, NUM_CROPS, CROP_TO_DISEASE_INDICES,
    MOE_HIDDEN_DIM, MOE_NUM_EXPERTS, MOE_DROPOUT,
    CLN_FEATURE_DIM,
)


class SwinTinyBackbone(nn.Module):
    """
    Swin-Tiny backbone with shifted window self-attention.

    CRITICAL: timm Swin-Tiny outputs NHWC format. This class permutes to NCHW.
    Patch embedding and first stage frozen for Phase 1 head training stability.
    Call unfreeze_all() at START of Phase 2 full fine-tuning.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        # Freeze patch embedding and first stage for training stability
        for name, param in self.backbone.named_parameters():
            if 'patch_embed' in name or 'layers.0' in name:
                param.requires_grad = False

    def forward(self, x):
        """Returns 3 NCHW feature maps: [N,192,28,28], [N,384,14,14], [N,768,7,7]"""
        features = self.backbone(x)  # list of NHWC tensors
        # CRITICAL: Swin outputs NHWC, permute to NCHW for Conv2d
        return [f.permute(0, 3, 1, 2).contiguous() for f in features]

    def unfreeze_all(self):
        """Unfreeze all parameters for Phase 2 full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print('Swin-Tiny backbone fully unfrozen for Phase 2 fine-tuning')


class FeaturePyramidNetwork(nn.Module):
    """
    FPN fusing Swin-Tiny multi-scale features.
    out_p3 is the Grad-CAM target — SAME NAME as previous architecture.
    """
    def __init__(self):
        super().__init__()
        self.lat_p3 = nn.Conv2d(FPN_IN_CH[0], FPN_OUT_CH, 1)
        self.lat_p4 = nn.Conv2d(FPN_IN_CH[1], FPN_OUT_CH, 1)
        self.lat_p5 = nn.Conv2d(FPN_IN_CH[2], FPN_OUT_CH, 1)
        # out_p3 is the Grad-CAM target — MUST keep this exact attribute name
        self.out_p3 = nn.Conv2d(FPN_OUT_CH, FPN_OUT_CH, 3, padding=1)
        self.bn_p3 = nn.BatchNorm2d(FPN_OUT_CH)
        self.bn_p4 = nn.BatchNorm2d(FPN_OUT_CH)
        self.bn_p5 = nn.BatchNorm2d(FPN_OUT_CH)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """features: list of 3 NCHW tensors. Returns [N, 256, 28, 28]."""
        p3_raw, p4_raw, p5_raw = features
        p5 = self.relu(self.bn_p5(self.lat_p5(p5_raw)))
        p4 = self.relu(self.bn_p4(self.lat_p4(p4_raw)))
        p3 = self.relu(self.bn_p3(self.lat_p3(p3_raw)))
        p4 = p4 + F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='nearest')
        p3 = self.out_p3(p3)
        return p3


class AttentionPooling(nn.Module):
    """
    Learned spatial attention pooling (replaces Global Average Pooling).
    Learns which of the HxW spatial locations matter for classification.
    Background locations get near-zero attention weights during training.
    """
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(FPN_OUT_CH, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """x: [N, C, H, W] → [N, C] attention-weighted pooled features"""
        N, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(N, H * W, C)  # [N, HW, C]
        att_weights = self.attention(x_flat)  # [N, HW, 1]
        att_weights = torch.softmax(att_weights, dim=1)
        pooled = (x_flat * att_weights).sum(dim=1)  # [N, C]
        return pooled


class CropClassifier(nn.Module):
    """4-class crop classifier. crop_emb [N,64] shape UNCHANGED for cache compat."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(FPN_OUT_CH, CROP_EMB_DIM),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(CROP_EMB_DIM, NUM_CROPS)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """x: [N,256] → crop_logits [N,4], crop_emb [N,64]"""
        crop_emb = self.embed(self.dropout(x))
        crop_logits = self.fc(crop_emb)
        return crop_logits, crop_emb


class ConditionalLayerNorm(nn.Module):
    """
    Per-channel per-crop normalisation (replaces FiLM).
    256 independent gamma+beta per crop = 256*4 parameters vs FiLM's 2*4.
    Uses soft crop_probs for differentiable parameter interpolation.
    """
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(CLN_FEATURE_DIM)
        self.crop_gamma = nn.Parameter(torch.ones(NUM_CROPS, CLN_FEATURE_DIM))
        self.crop_beta = nn.Parameter(torch.zeros(NUM_CROPS, CLN_FEATURE_DIM))

    def forward(self, x, crop_probs):
        """x: [N,256], crop_probs: [N,4] → [N,256] normalised features"""
        x_norm = self.layer_norm(x)
        gamma = torch.matmul(crop_probs, self.crop_gamma)  # [N, 256]
        beta = torch.matmul(crop_probs, self.crop_beta)    # [N, 256]
        return gamma * x_norm + beta


class MixtureOfExpertsDiseaseHead(nn.Module):
    """
    4 crop-specific expert MLPs gated by crop_probs.
    crop_mask enforces biological constraints (okra expert cannot predict tomato diseases).
    """
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FPN_OUT_CH, MOE_HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(p=MOE_DROPOUT),
                nn.Linear(MOE_HIDDEN_DIM, NUM_CLASSES),
            )
            for _ in range(MOE_NUM_EXPERTS)
        ])
        # Binary mask: mask[crop][class] = 1 if class belongs to crop
        mask = torch.zeros(NUM_CROPS, NUM_CLASSES)
        for crop_idx, class_indices in CROP_TO_DISEASE_INDICES.items():
            for class_idx in class_indices:
                mask[crop_idx][class_idx] = 1.0
        self.register_buffer('crop_mask', mask)

    def forward(self, x, crop_probs):
        """x: [N,256], crop_probs: [N,4] → disease_logits [N,23]"""
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # [N, 4, 23]
        masked = expert_outputs * self.crop_mask.unsqueeze(0)  # [N, 4, 23]
        disease_logits = (masked * crop_probs.unsqueeze(-1)).sum(dim=1)  # [N, 23]
        return disease_logits


class SeverityHead(nn.Module):
    """3-class severity classifier: mild=0, moderate=1, severe=2. UNCHANGED."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(POOLED_DIM, HEAD_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_P)
        self.fc2 = nn.Linear(HEAD_HIDDEN_DIM, 3)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class PlantDiseaseModel(nn.Module):
    """
    Complete model: Swin-Tiny + FPN + AttPool + CLN + MoE + DeiT.

    forward() returns (crop_logits, disease_logits, severity_logits) — same as before.
    forward_with_distillation() adds distillation_logits for Phase 2 DeiT training.
    extract_features() returns (pooled [N,256], crop_emb [N,64]) for cache.
    """
    def __init__(self):
        super().__init__()
        self.backbone = SwinTinyBackbone()
        self.fpn = FeaturePyramidNetwork()
        self.att_pool = AttentionPooling()
        self.crop_classifier = CropClassifier()
        self.cln = ConditionalLayerNorm()
        self.disease_head = MixtureOfExpertsDiseaseHead()
        self.severity_head = SeverityHead()
        # DeiT distillation head — used ONLY during Phase 2 training
        self.distillation_head = nn.Linear(FPN_OUT_CH, NUM_CLASSES)
        # Pre-compute dropout layer references for MC Dropout
        self._dropout_layers = [
            m for m in self.modules() if isinstance(m, nn.Dropout)
        ]

    def forward(self, x):
        """Standard forward. Returns same tuple as previous architecture."""
        features = self.backbone(x)
        fpn_out = self.fpn(features)
        pooled = self.att_pool(fpn_out)
        crop_logits, crop_emb = self.crop_classifier(pooled)
        crop_probs = torch.softmax(crop_logits, dim=-1)
        x_cln = self.cln(pooled, crop_probs)
        disease_logits = self.disease_head(x_cln, crop_probs)
        severity_logits = self.severity_head(pooled)
        return crop_logits, disease_logits, severity_logits

    def forward_with_distillation(self, x):
        """Extended forward for Phase 2 DeiT training. Never called in inference."""
        features = self.backbone(x)
        fpn_out = self.fpn(features)
        pooled = self.att_pool(fpn_out)
        crop_logits, crop_emb = self.crop_classifier(pooled)
        crop_probs = torch.softmax(crop_logits, dim=-1)
        x_cln = self.cln(pooled, crop_probs)
        disease_logits = self.disease_head(x_cln, crop_probs)
        severity_logits = self.severity_head(pooled)
        distillation_logits = self.distillation_head(pooled)
        return crop_logits, disease_logits, severity_logits, distillation_logits

    def forward_with_features(self, x):
        """Extended forward returning pooled features for CORAL + DeiT distillation.
        Returns: crop_logits, disease_logits, severity_logits, distillation_logits, pooled"""
        features = self.backbone(x)
        fpn_out = self.fpn(features)
        pooled = self.att_pool(fpn_out)
        crop_logits, crop_emb = self.crop_classifier(pooled)
        crop_probs = torch.softmax(crop_logits, dim=-1)
        x_cln = self.cln(pooled, crop_probs)
        disease_logits = self.disease_head(x_cln, crop_probs)
        severity_logits = self.severity_head(pooled)
        distillation_logits = self.distillation_head(pooled)
        return crop_logits, disease_logits, severity_logits, distillation_logits, pooled

    def extract_features(self, x):
        """Returns (pooled [N,256], crop_emb [N,64]) for Step 07 cache."""
        features = self.backbone(x)
        fpn_out = self.fpn(features)
        pooled = self.att_pool(fpn_out)
        _, crop_emb = self.crop_classifier(pooled)
        return pooled, crop_emb

    def enable_mc_dropout(self):
        """Set Dropout layers to train mode for MC Dropout inference."""
        for layer in self._dropout_layers:
            layer.train()

    def disable_mc_dropout(self):
        """Restore all layers to eval mode after MC Dropout."""
        self.eval()

    def freeze_backbone(self):
        """Freeze entire backbone for Phase 1 head training."""
        for p in self.backbone.parameters():
            p.requires_grad = False


def verify_backbone_shapes(model, device='cpu'):
    """Verify Swin-Tiny backbone outputs correct NCHW shapes."""
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        feats = model.backbone(dummy)
    expected = [(1, 192, 28, 28), (1, 384, 14, 14), (1, 768, 7, 7)]
    for i, (feat, exp) in enumerate(zip(feats, expected)):
        assert tuple(feat.shape) == exp, (
            f"Backbone stage {i+1} shape {tuple(feat.shape)} != expected {exp}. "
            f"Check NHWC→NCHW permute in SwinTinyBackbone.forward()"
        )
    print(f"Backbone shapes verified (NCHW): {[tuple(f.shape) for f in feats]}")


def load_model_for_inference(checkpoint_path, device):
    """
    Load Swin-Tiny student model for inference.
    If checkpoint doesn't exist, returns untrained model (expected before Phase 2).
    DO NOT use for EfficientNetV2-S teacher — use training/teacher_model.py instead.
    """
    model = PlantDiseaseModel()

    if checkpoint_path and os.path.exists(str(checkpoint_path)):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict',
                         checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
        print(f'Loaded Swin-Tiny checkpoint from {checkpoint_path}')
    else:
        print(f'No checkpoint at {checkpoint_path} — using untrained Swin-Tiny')

    model.eval()
    model.to(device)
    return model
