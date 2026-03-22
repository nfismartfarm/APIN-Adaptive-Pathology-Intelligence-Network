# app/model.py

import torch
import torch.nn as nn
import timm
from app.config import (
    BACKBONE_NAME, FPN_IN_CH, FPN_OUT_CH, POOLED_DIM,
    CROP_EMB_DIM, HEAD_HIDDEN_DIM, DROPOUT_P, NUM_CLASSES, NUM_CROPS
)


class FPN(nn.Module):
    """
    Feature Pyramid Network. Takes three backbone stages and produces
    a single fused feature map at P3 resolution (28x28 for 224px input).
    """
    def __init__(self):
        super().__init__()
        self.lat_p3 = nn.Conv2d(FPN_IN_CH[0], FPN_OUT_CH, 1)
        self.lat_p4 = nn.Conv2d(FPN_IN_CH[1], FPN_OUT_CH, 1)
        self.lat_p5 = nn.Conv2d(FPN_IN_CH[2], FPN_OUT_CH, 1)
        self.out_p3 = nn.Conv2d(FPN_OUT_CH, FPN_OUT_CH, 3, padding=1)
        self._upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, p3_feat, p4_feat, p5_feat):
        p5 = self.lat_p5(p5_feat)
        p4 = self.lat_p4(p4_feat) + self._upsample(p5)
        p3 = self.lat_p3(p3_feat) + self._upsample(p4)
        return self.out_p3(p3)


class CropClassifier(nn.Module):
    """Binary crop classifier (okra vs brassica)."""
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(POOLED_DIM, HEAD_HIDDEN_DIM)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(DROPOUT_P)
        self.embed = nn.Linear(HEAD_HIDDEN_DIM, CROP_EMB_DIM)
        self.fc2   = nn.Linear(CROP_EMB_DIM, NUM_CROPS)

    def forward(self, x):
        x   = self.drop(self.relu(self.fc1(x)))
        emb = self.embed(x)
        return self.fc2(emb), emb


class DiseaseHead(nn.Module):
    """Multi-label disease classifier. FiLM-conditioned on crop embedding."""
    def __init__(self):
        super().__init__()
        self.film_gamma = nn.Linear(CROP_EMB_DIM, POOLED_DIM)
        self.film_beta  = nn.Linear(CROP_EMB_DIM, POOLED_DIM)
        self.fc1  = nn.Linear(POOLED_DIM, HEAD_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_P)
        self.fc2  = nn.Linear(HEAD_HIDDEN_DIM, NUM_CLASSES)

    def forward(self, pooled, crop_emb):
        gamma = torch.sigmoid(self.film_gamma(crop_emb))
        beta  = self.film_beta(crop_emb)
        x     = pooled * gamma + beta
        x     = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class SeverityHead(nn.Module):
    """3-class severity classifier: mild=0, moderate=1, severe=2."""
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(POOLED_DIM, HEAD_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_P)
        self.fc2  = nn.Linear(HEAD_HIDDEN_DIM, 3)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class PlantDiseaseModel(nn.Module):
    """Full model: EfficientNetV2-S backbone + FPN + three heads."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=True,
            features_only=True, out_indices=(2, 3, 4)
        )
        self.fpn           = FPN()
        self.gap           = nn.AdaptiveAvgPool2d(1)
        self.crop_classifier = CropClassifier()
        self.disease_head    = DiseaseHead()
        self.severity_head   = SeverityHead()

    def forward(self, x):
        features       = self.backbone(x)
        p3, p4, p5     = features[0], features[1], features[2]
        fused          = self.fpn(p3, p4, p5)
        pooled         = self.gap(fused).flatten(1)
        crop_logits, crop_emb = self.crop_classifier(pooled)
        disease_logits = self.disease_head(pooled, crop_emb)
        severity_logits = self.severity_head(pooled)
        return crop_logits, disease_logits, severity_logits

    def extract_features(self, x):
        features   = self.backbone(x)
        fused      = self.fpn(features[0], features[1], features[2])
        pooled     = self.gap(fused).flatten(1)
        _, crop_emb = self.crop_classifier(pooled)
        return pooled, crop_emb

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_fraction(self, fraction=0.33):
        blocks = self._get_backbone_blocks()
        n_unfreeze = max(1, int(len(blocks) * fraction))
        for b in blocks:
            for p in b.parameters():
                p.requires_grad = False
        for b in blocks[-n_unfreeze:]:
            for p in b.parameters():
                p.requires_grad = True
        print(f"Unfroze top {n_unfreeze}/{len(blocks)} backbone blocks for Phase 2")

    def _get_backbone_blocks(self):
        try:
            blocks = list(self.backbone.model.blocks)
            if blocks:
                return blocks
        except AttributeError:
            pass
        try:
            blocks = list(self.backbone.blocks)
            if blocks:
                return blocks
        except AttributeError:
            pass
        raise RuntimeError(
            "_get_backbone_blocks() could not find backbone.model.blocks or "
            "backbone.blocks. Check timm version and BACKBONE_NAME. "
            "EfficientNetV2-S in timm >= 0.9 exposes blocks at backbone.model.blocks. "
            "Verify with: model.backbone.model.blocks"
        )

    def _get_stem_params(self):
        params = []
        for attr in ['conv_stem', 'bn1']:
            try:
                layer = getattr(self.backbone.model, attr)
                params.extend(list(layer.parameters()))
            except AttributeError:
                pass
        return params


def verify_backbone_shapes(model, device='cpu'):
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        feats = model.backbone(dummy)
    expected = [(1, 48, 28, 28), (1, 160, 14, 14), (1, 256, 7, 7)]
    for i, (feat, exp) in enumerate(zip(feats, expected)):
        assert tuple(feat.shape) == exp, (
            f"Backbone stage {i+2} shape {tuple(feat.shape)} != expected {exp}. "
            f"Update FPN_IN_CH in app/config.py to {[f.shape[1] for f in feats]}"
        )
    print(f"Backbone shapes verified: {[tuple(f.shape) for f in feats]}")


def load_model_for_inference(model_path, device):
    model = PlantDiseaseModel()
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model
