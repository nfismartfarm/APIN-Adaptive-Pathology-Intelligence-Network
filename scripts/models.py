"""
models.py — Model Wrapper Classes for the 3-Model Specialist Pipeline

Each class encapsulates:
  - Backbone loading (timm or transformers as appropriate)
  - Classification head attachment
  - Forward pass with correct signature
  - Feature/attention extraction for heatmaps
  - Stage transition logic (freeze/unfreeze)
  - Parameter group construction for optimizers

Training scripts import these classes — they never build models inline.

Classes:
  RouterDINO       — DINOv2-Small+Registers (timm), frozen, Linear(384→4)
  Model2ConvNeXt   — DINOv3-ConvNeXt-Small (transformers), full FT, Linear(768→9)
  Model3DINOLoRA   — DINOv2-Small (timm) + LoRA + FiLM, Linear(384→10)
"""

import warnings
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# ROUTER: DINOv2-Small with Registers — frozen backbone + linear head
# ═══════════════════════════════════════════════════════════════════════════

class RouterDINO(nn.Module):
    """
    Crop Router: identifies okra / brassica / tomato / chilli.

    Architecture:
      DINOv2-Small-with-Registers (22M params, FROZEN)
      + Linear(384 → 4)
      Only the 4-class linear head trains (~1.5K params).

    Why frozen: 89% of tomato router images are lab photos. Fine-tuning
    would teach the backbone 'gray background = tomato'. Frozen DINOv2
    features are domain-invariant from 142M image pretraining.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        import timm
        from app.config_router import BACKBONE_NAME, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM

        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,        # remove default head
            img_size=DINOV2_IMG_SIZE,  # CRITICAL: default 518 crashes
        )
        # Freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Linear(DINOV2_EMBED_DIM, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, num_classes) logits."""
        with torch.no_grad():
            features = self.backbone(x)  # (batch, 384)
        return self.head(features)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Returns only head parameters (for optimizer)."""
        return list(self.head.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: DINOv3-ConvNeXt-Small — full fine-tune + classification head
# ═══════════════════════════════════════════════════════════════════════════

class Model2ConvNeXt(nn.Module):
    """
    Okra + Brassica Specialist: 9-class disease classifier.

    Architecture:
      DINOv3-ConvNeXt-Small (49.5M params, loaded via transformers)
      + Linear(768 → 9)
      Full fine-tuning with LLRD in Stage 1, head-only in Stage 2.

    Why DINOv3: 1.7B image self-supervised pretraining (distilled from 7B ViT
    teacher) gives richer starting features than ImageNet-22k supervised.
    ConvNeXt architecture supports GradCAM++ natively.

    Why transformers not timm: DINOv3-ConvNeXt-Small's HuggingFace config
    uses 'DINOv3ConvNextModel' architecture which timm can't parse.
    """

    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        from app.config_model2 import (DINOV3_BACKBONE, FALLBACK_BACKBONE,
                                       BACKBONE_EMBED_DIM, BACKBONE_LIBRARY)

        # Load DINOv3-ConvNeXt-Small via transformers
        try:
            if BACKBONE_LIBRARY == 'transformers':
                self.backbone = AutoModel.from_pretrained(DINOV3_BACKBONE)
                self._backbone_type = 'transformers'
            else:
                raise ImportError("Fallback to timm requested")
        except Exception as e:
            warnings.warn(f'DINOv3 load failed ({e}), using timm fallback')
            import timm
            self.backbone = timm.create_model(FALLBACK_BACKBONE,
                                               pretrained=pretrained,
                                               num_classes=0)
            self._backbone_type = 'timm'

        self.head = nn.Linear(BACKBONE_EMBED_DIM, num_classes)
        self.num_classes = num_classes
        self.embed_dim = BACKBONE_EMBED_DIM

    def forward(self, x: torch.Tensor,
                return_features: bool = False) -> torch.Tensor:
        """
        Returns logits (batch, num_classes).
        If return_features=True, also returns pooled features for SupCon.
        """
        if self._backbone_type == 'transformers':
            out = self.backbone(x)
            pooled = out.pooler_output  # (batch, 768)
        else:
            pooled = self.backbone(x)   # timm: already pooled

        logits = self.head(pooled)

        if return_features:
            return logits, pooled
        return logits

    def get_hidden_states(self, x: torch.Tensor):
        """Get intermediate feature maps for GradCAM++."""
        if self._backbone_type == 'transformers':
            out = self.backbone(x, output_hidden_states=True)
            return out.hidden_states  # list of tensors per stage
        else:
            # timm features_only would need a different model creation
            warnings.warn('get_hidden_states not supported for timm fallback')
            return None

    def get_gradcam_target_layer(self):
        """Returns the target layer module for GradCAM++."""
        if self._backbone_type == 'transformers':
            # DINOv3-ConvNeXt: stages.3 is the last conv stage
            if hasattr(self.backbone, 'stages') and len(self.backbone.stages) > 3:
                stage = self.backbone.stages[3]
                # Find last Conv2d in this stage
                last_conv = None
                for module in stage.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                return last_conv
        return None

    def freeze_backbone(self):
        """Freeze all backbone params for Stage 2 head-only training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Model2 backbone frozen: {trainable} trainable params (head only)')

    def unfreeze_backbone(self):
        """Unfreeze backbone for Stage 1 full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_llrd_param_groups(self, base_lr: float, decay_rate: float = 0.90,
                              weight_decay: float = 0.01) -> List[dict]:
        """
        Layer-wise learning rate decay for ConvNeXt stages.
        Head gets base_lr; each deeper stage gets lr * decay^depth.
        Bias and LayerNorm excluded from weight decay.
        """
        no_decay = {'bias', 'LayerNorm.weight', 'layernorm.weight',
                    'norm.weight', 'ln.weight'}
        param_groups = []

        # Head: full base_lr
        head_decay = [p for n, p in self.head.named_parameters()
                     if not any(nd in n for nd in no_decay)]
        head_no_decay = [p for n, p in self.head.named_parameters()
                        if any(nd in n for nd in no_decay)]
        if head_decay:
            param_groups.append({'params': head_decay, 'lr': base_lr,
                                'weight_decay': weight_decay, 'name': 'head'})
        if head_no_decay:
            param_groups.append({'params': head_no_decay, 'lr': base_lr,
                                'weight_decay': 0.0, 'name': 'head_no_decay'})

        # Backbone stages: decaying LR
        if self._backbone_type == 'transformers' and hasattr(self.backbone, 'stages'):
            stages = list(self.backbone.stages)
            for i, stage in enumerate(reversed(stages)):
                lr = base_lr * (decay_rate ** (i + 1))
                stage_decay = [p for n, p in stage.named_parameters()
                             if p.requires_grad and not any(nd in n for nd in no_decay)]
                stage_no_decay = [p for n, p in stage.named_parameters()
                                if p.requires_grad and any(nd in n for nd in no_decay)]
                if stage_decay:
                    param_groups.append({'params': stage_decay, 'lr': lr,
                                       'weight_decay': weight_decay,
                                       'name': f'stage_{len(stages)-1-i}'})
                if stage_no_decay:
                    param_groups.append({'params': stage_no_decay, 'lr': lr,
                                       'weight_decay': 0.0,
                                       'name': f'stage_{len(stages)-1-i}_no_decay'})
        else:
            # Fallback: all backbone params at decayed LR
            bb_params = [p for p in self.backbone.parameters() if p.requires_grad]
            if bb_params:
                param_groups.append({'params': bb_params,
                                    'lr': base_lr * decay_rate,
                                    'weight_decay': weight_decay,
                                    'name': 'backbone_all'})

        return param_groups


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3: DINOv2-Small + LoRA + FiLM + classification head
# ═══════════════════════════════════════════════════════════════════════════

class Model3DINOLoRA(nn.Module):
    """
    Tomato + Chilli Specialist: 10-class disease classifier.

    Architecture:
      DINOv2-Small (22M params, FROZEN via timm)
      + LoRA rank=8 on QKV attention projections (151K trainable)
      + FiLM conditioning on crop identity (tomato=0, chilli=1)
      + Linear(384 → 10)

    Why LoRA on frozen DINOv2: 97-99% of tomato disease images are lab photos.
    Full fine-tuning would destroy DINOv2's domain-invariant features.
    LoRA trains <1% of parameters while preserving the pretraining.

    CRITICAL: img_size=224 MUST be passed to timm.create_model.
    Default is 518 which crashes with our 224px training images.
    """

    def __init__(self, num_classes: int = 10, num_crops: int = 2,
                 pretrained: bool = True, enable_gradient_checkpointing: bool = True):
        super().__init__()
        import timm
        from peft import LoraConfig, get_peft_model
        from app.config_model3 import (
            DINOV2_BACKBONE, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM,
            LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES, LORA_DROPOUT,
            FILM_CROP_EMBEDDING_DIM,
        )

        # Step 1: Create DINOv2 backbone
        backbone = timm.create_model(
            DINOV2_BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            img_size=DINOV2_IMG_SIZE,  # CRITICAL
        )

        # Step 2: Enable gradient checkpointing BEFORE LoRA (peft wraps the model)
        if enable_gradient_checkpointing:
            backbone.set_grad_checkpointing(True)

        # Step 3: Attach LoRA
        # [FIX CRITICAL 3] Don't use modules_to_save for timm num_classes=0
        # (head may not exist or may be Identity — peft gets confused)
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
        )
        self.backbone = get_peft_model(backbone, lora_config)

        # Step 4: FiLM conditioning (crop identity modulation)
        self.crop_embed = nn.Embedding(num_crops, FILM_CROP_EMBEDDING_DIM)
        self.film_gamma = nn.Linear(FILM_CROP_EMBEDDING_DIM, DINOV2_EMBED_DIM)
        self.film_beta = nn.Linear(FILM_CROP_EMBEDDING_DIM, DINOV2_EMBED_DIM)
        # Initialise near-identity (gamma~1, beta~0)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Step 5: Classification head
        self.head = nn.Linear(DINOV2_EMBED_DIM, num_classes)
        self.num_classes = num_classes
        self.embed_dim = DINOV2_EMBED_DIM

    def forward(self, x: torch.Tensor,
                crop_ids: torch.Tensor = None,
                return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) input images
            crop_ids: (batch,) integer tensor — 0=tomato, 1=chilli
                      If None, FiLM conditioning is skipped.
            return_features: if True, also return pooled features

        Returns logits (batch, num_classes).
        """
        out = self.backbone(x)
        # [FIX CRITICAL 3] Handle both tensor and PeftModelOutput return types
        if isinstance(out, torch.Tensor):
            features = out  # (batch, embed_dim)
        elif hasattr(out, 'logits'):
            features = out.logits
        elif hasattr(out, 'last_hidden_state'):
            features = out.last_hidden_state[:, 0]  # CLS token
        else:
            features = out  # hope for the best

        # Apply FiLM conditioning if crop_ids provided
        if crop_ids is not None:
            emb = self.crop_embed(crop_ids)           # (batch, crop_embed_dim)
            gamma = 1.0 + self.film_gamma(emb)        # scale near 1
            beta = self.film_beta(emb)                # shift near 0
            features = features * gamma + beta

        logits = self.head(features)

        if return_features:
            return logits, features
        return logits

    def extract_attention_map(self, x: torch.Tensor, block_idx: int = -1,
                               img_size: int = 224, patch_size: int = 14):
        """
        Extract DINO self-attention map for heatmap visualisation.
        CLS token attention to patch tokens, averaged across all heads.

        Returns 2D numpy array at (grid_size, grid_size) resolution.
        """
        # Delegate to train_utils implementation
        from scripts.train_utils import extract_dino_attention_map
        return extract_dino_attention_map(
            self.backbone, x, block_idx=block_idx,
            img_size=img_size, patch_size=patch_size
        )

    def freeze_for_stage2(self):
        """
        Stage 2: freeze backbone AND LoRA, train only head + FiLM.
        """
        for name, param in self.named_parameters():
            is_head = 'head' in name
            is_film = 'film' in name or 'crop_embed' in name
            param.requires_grad = is_head or is_film

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'Model3 Stage 2: {trainable/1e3:.1f}K trainable / {total/1e6:.1f}M total '
              f'(head + FiLM only)')

    def unfreeze_lora(self):
        """Unfreeze LoRA + head + FiLM for Stage 1 training."""
        for name, param in self.named_parameters():
            is_lora = 'lora_' in name
            is_head = 'head' in name
            is_film = 'film' in name or 'crop_embed' in name
            param.requires_grad = is_lora or is_head or is_film

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'Model3 Stage 1: {trainable/1e3:.1f}K trainable / {total/1e6:.1f}M total '
              f'(LoRA + head + FiLM)')


# ═══════════════════════════════════════════════════════════════════════════
# MODULE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print('models.py — Model Wrapper Self-Test')
    print('=' * 60)

    # Test 1: RouterDINO
    print('\n[Test 1] RouterDINO')
    try:
        router = RouterDINO(num_classes=4, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        logits = router(x)
        trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
        total = sum(p.numel() for p in router.parameters())
        print(f'  Output: {tuple(logits.shape)} (expected (2, 4))')
        print(f'  Trainable: {trainable} / {total} ({trainable/total*100:.2f}%)')
        assert logits.shape == (2, 4), f'Wrong output shape: {logits.shape}'
        print('  PASSED')
    except Exception as e:
        print(f'  FAILED: {e}')

    # Test 2: Model2ConvNeXt
    print('\n[Test 2] Model2ConvNeXt')
    try:
        model2 = Model2ConvNeXt(num_classes=9, pretrained=False)
        x = torch.randn(2, 3, 384, 384)
        logits = model2(x)
        logits_f, features = model2(x, return_features=True)
        print(f'  Output: {tuple(logits.shape)} (expected (2, 9))')
        print(f'  Features: {tuple(features.shape)} (expected (2, 768))')
        print(f'  Backbone type: {model2._backbone_type}')

        # Test LLRD param groups
        groups = model2.get_llrd_param_groups(base_lr=1e-4)
        print(f'  LLRD param groups: {len(groups)}')
        for g in groups[:3]:
            print(f'    {g.get("name", "?")}: lr={g["lr"]:.6f}, '
                  f'params={sum(p.numel() for p in g["params"])/1e6:.2f}M')

        # Test freeze
        model2.freeze_backbone()
        model2.unfreeze_backbone()
        print('  PASSED')
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback
        traceback.print_exc()

    # Test 3: Model3DINOLoRA
    print('\n[Test 3] Model3DINOLoRA')
    try:
        model3 = Model3DINOLoRA(num_classes=10, pretrained=False,
                                enable_gradient_checkpointing=False)
        x = torch.randn(2, 3, 224, 224)
        crop_ids = torch.tensor([0, 1])  # tomato, chilli
        logits = model3(x, crop_ids=crop_ids)
        logits_nc = model3(x)  # without crop_ids (FiLM skipped)
        logits_f, features = model3(x, crop_ids=crop_ids, return_features=True)
        print(f'  Output: {tuple(logits.shape)} (expected (2, 10))')
        print(f'  Features: {tuple(features.shape)} (expected (2, 384))')

        # Test stage transitions
        model3.unfreeze_lora()
        model3.freeze_for_stage2()
        print('  PASSED')
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback
        traceback.print_exc()

    print('\n' + '=' * 60)
    print('All model wrapper tests complete.')
