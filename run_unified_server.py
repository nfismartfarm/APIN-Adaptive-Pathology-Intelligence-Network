"""
Unified Pipeline Server -- Router -> Specialist on localhost:8005
Full inference pipeline: crop routing -> disease classification with crop gating + smart thresholds.

Flow:
  1. Upload leaf image
  2. Router (DINOv2-Small) identifies crop: okra / brassica / tomato / chilli
  3. If okra or brassica -> Model 2 (DINOv3-ConvNeXt) diagnoses disease
  4. Crop gating: irrelevant crop classes zeroed out before softmax
  5. Multi-threshold: if "healthy" but top disease >25% -> flag as possible early-stage
  6. User can override crop and re-predict

Usage: python run_unified_server.py
Open: http://localhost:8005
"""

import os
import sys
import io
import json
import time
import base64

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))

from app.config_router import (
    BACKBONE_NAME as ROUTER_BACKBONE, DINOV2_IMG_SIZE as ROUTER_IMG_SIZE,
    DINOV2_EMBED_DIM as ROUTER_EMBED_DIM,
    NUM_CLASSES as ROUTER_NUM_CLASSES, CLASS_NAMES as ROUTER_CLASSES,
)
from app.config_model2 import (
    NUM_CLASSES as M2_NUM_CLASSES, CLASS_NAMES as M2_CLASSES,
    CLASS_TO_IDX as M2_CLASS_TO_IDX, OKRA_INDICES, BRASSICA_INDICES,
)
from scripts.models import Model2ConvNeXt
from scripts.tier1_postprocessing import apply_tier1_fixes


# ── Disease info for display ──────────────────────────────────────────
DISEASE_INFO = {
    'okra_yvmv': {'emoji': '🟡', 'name': 'Yellow Vein Mosaic Virus', 'severity': 'High', 'color': '#e74c3c',
        'desc': 'Viral disease spread by whitefly. Yellow vein network on leaves. No cure -- remove infected plants immediately.',
        'action': 'Remove and destroy infected plants. Spray imidacloprid for whitefly control.'},
    'okra_powdery_mildew': {'emoji': '🤍', 'name': 'Powdery Mildew', 'severity': 'Medium', 'color': '#e67e22',
        'desc': 'White powdery coating on leaf surface. Thrives in warm dry conditions.',
        'action': 'Apply wettable sulphur 80 WP at 2.5g/L. Do NOT spray above 35C.'},
    'okra_cercospora': {'emoji': '🟤', 'name': 'Cercospora Leaf Spot', 'severity': 'Medium', 'color': '#e67e22',
        'desc': 'Brown spots with grey center and yellow halo. Spread by airborne spores.',
        'action': 'Spray mancozeb 75 WP at 2.5g/L every 7-10 days. Remove infected leaves.'},
    'okra_enation': {'emoji': '🔴', 'name': 'Enation Leaf Curl', 'severity': 'High', 'color': '#e74c3c',
        'desc': 'Severe leaf curling with bumpy enations on vein undersides. Viral, no cure.',
        'action': 'Remove infected plants. Control whitefly vector with imidacloprid.'},
    'okra_healthy': {'emoji': '💚', 'name': 'Healthy Okra', 'severity': 'None', 'color': '#27ae60',
        'desc': 'No disease detected. Leaf appears healthy.', 'action': 'Continue monitoring every 5-7 days.'},
    'brassica_black_rot': {'emoji': '⬛', 'name': 'Black Rot', 'severity': 'High', 'color': '#e74c3c',
        'desc': 'V-shaped lesions from leaf margins with darkened veins. Bacterial.',
        'action': 'Remove infected plants. Spray copper oxychloride. Do NOT work when plants are wet.'},
    'brassica_downy_mildew': {'emoji': '🔵', 'name': 'Downy Mildew', 'severity': 'Medium', 'color': '#e67e22',
        'desc': 'Yellow patches above, white sporulation below. Cool humid conditions.',
        'action': 'Apply metalaxyl + mancozeb (Ridomil Gold MZ) at 2.5g/L.'},
    'brassica_alternaria': {'emoji': '🎯', 'name': 'Alternaria Leaf Spot', 'severity': 'Medium', 'color': '#e67e22',
        'desc': 'Dark concentric ring spots (target pattern). Airborne fungal spores.',
        'action': 'Spray mancozeb 75 WP at 2.5g/L. Treat seeds with thiram before sowing.'},
    'brassica_healthy': {'emoji': '💚', 'name': 'Healthy Brassica', 'severity': 'None', 'color': '#27ae60',
        'desc': 'No disease detected. Leaf appears healthy.', 'action': 'Continue monitoring every 7-10 days.'},
}

CROP_EMOJI = {'okra': '🫛', 'brassica': '🥦', 'tomato': '🍅', 'chilli': '🌶️'}


def load_router(device):
    backbone = timm.create_model(ROUTER_BACKBONE, pretrained=True, num_classes=0, img_size=ROUTER_IMG_SIZE)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    head = nn.Linear(ROUTER_EMBED_DIM, ROUTER_NUM_CLASSES).to(device)
    ckpt_path = ROOT / 'models' / 'router' / 'router_best.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(ckpt['head_state_dict'])
        print(f'Router loaded: F1={ckpt.get("best_f1", "?"):.4f}')
    head.eval()
    return backbone, head


def load_model2(device):
    model = Model2ConvNeXt(num_classes=M2_NUM_CLASSES, pretrained=False).to(device)
    ckpt_path = ROOT / 'models' / 'model2_specialist' / 'model2_production.pt'
    if not ckpt_path.exists():
        ckpt_path = ROOT / 'models' / 'model2_specialist' / 'model2_best.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Model 2 loaded: F1={ckpt.get("val_f1", ckpt.get("best_f1", "?"))}')
    model.eval()
    return model


def generate_gradcam(model, tensor, original_np, target_class):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    target_layer = model.get_gradcam_target_layer()
    if not target_layer:
        return ''
    try:
        targets = [ClassifierOutputTarget(target_class)]
        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
            grayscale = cam(input_tensor=tensor, targets=targets)[0]
        orig = cv2.resize(original_np, (384, 384)).astype(np.float32) / 255.0
        overlay = show_cam_on_image(orig, grayscale, use_rgb=True)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f'GradCAM++ error: {e}', flush=True)
        return ''


def crop_gated_predict(model, tensor, crop_type, router_conf, device):
    """
    Run Model 2 with adaptive crop gating.

    [FIX: Verifier Agent] Adaptive soft masking instead of hard -inf:
    - High router confidence (>=0.80): hard mask (irrelevant classes -> -inf)
    - Medium confidence (0.50-0.80): soft mask (irrelevant classes downweighted by 0.1x)
    - Low confidence (<0.50): no mask (let all classes compete)

    This prevents catastrophic cascading errors when the router is wrong.
    With hard masking, a router error at 51% confidence permanently zeros
    the correct crop's diseases. Soft masking preserves them as fallback.
    """
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
        logits = model(tensor)
    logits = logits.float().detach()

    if crop_type == 'okra':
        irrelevant = BRASSICA_INDICES
    elif crop_type == 'brassica':
        irrelevant = OKRA_INDICES
    else:
        irrelevant = []

    # Adaptive gating based on router confidence
    if router_conf >= 0.80:
        # High confidence: hard mask
        logits[0, irrelevant] = float('-inf')
    elif router_conf >= 0.50:
        # Medium confidence: soft mask (downweight by 10x)
        logits[0, irrelevant] -= 2.3  # log(10) ~ 2.3, equivalent to 0.1x in probability
    # else: low confidence -- no masking, let all classes compete

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, logits


def smart_diagnosis(probs, crop_type):
    """
    Multi-threshold decision logic.
    Returns (primary_disease, confidence, alert_type, secondary_disease, secondary_conf)

    alert_type:
      'confident' - clear diagnosis
      'possible_early' - healthy but disease signal >25%
      'uncertain' - top confidence <40%
      'multi_disease' - two diseases both >20%
    """
    top_idx = int(np.argmax(probs))
    top_cls = M2_CLASSES[top_idx]
    top_conf = float(probs[top_idx])

    # Get sorted probabilities
    sorted_idx = np.argsort(probs)[::-1]
    second_idx = sorted_idx[1]
    second_cls = M2_CLASSES[second_idx]
    second_conf = float(probs[second_idx])

    # Healthy class indices
    healthy_cls = 'okra_healthy' if crop_type == 'okra' else 'brassica_healthy'

    # [FIX: Pessimistic Audit #4] Check uncertain FIRST, then possible_early.
    # Without this, a 32% healthy / 30% disease case triggers 'possible_early'
    # when it should trigger 'uncertain' (both are below 40%).
    if top_conf < 0.40:
        # Very uncertain -- check if it's also a possible early-stage case
        if top_cls == healthy_cls and second_conf > 0.20:
            return second_cls, second_conf, 'possible_early', top_cls, top_conf
        return top_cls, top_conf, 'uncertain', second_cls, second_conf
    elif top_cls == healthy_cls and second_conf > 0.25:
        # Healthy wins with decent confidence but disease signal present
        return second_cls, second_conf, 'possible_early', top_cls, top_conf
    elif top_conf < 0.60 and second_conf > 0.20:
        # Two candidates both plausible
        return top_cls, top_conf, 'multi_disease', second_cls, second_conf
    else:
        return top_cls, top_conf, 'confident', second_cls, second_conf


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    router_backbone, router_head = load_router(device)
    model2 = load_model2(device)
    app.state.router_backbone = router_backbone
    app.state.router_head = router_head
    app.state.model2 = model2
    app.state.device = device
    # [FIX: Verifier Agent] LAB-CLAHE (not RGB-CLAHE) to match training.
    # Training used LAB-CLAHE: CLAHE on L channel only in LAB colorspace.
    # RGB-CLAHE shifts hues (each channel adjusted independently), while
    # LAB-CLAHE preserves color signatures (only luminance/contrast changes).
    # Input is RGB from PIL -> convert RGB->LAB, CLAHE on L, convert LAB->RGB.
    def apply_lab_clahe(image, **kwargs):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Only L channel
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    app.state.router_transform = A.Compose([
        A.Lambda(image=apply_lab_clahe, p=1.0),  # CLAHE first
        A.Resize(ROUTER_IMG_SIZE, ROUTER_IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    app.state.model2_transform = A.Compose([
        A.Lambda(image=apply_lab_clahe, p=1.0),  # CLAHE first
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    print(f'Unified pipeline ready on {device}', flush=True)
    yield


app = FastAPI(title='Plant Disease Detection Pipeline', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


@app.get('/', response_class=HTMLResponse)
async def index():
    return (ROOT / 'templates' / 'unified_pipeline.html').read_text(encoding='utf-8')


@app.post('/predict')
async def predict(file: UploadFile = File(...), override_crop: Optional[str] = Form(None)):
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, 'File too large (max 10 MB)')
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
    except Exception:
        raise HTTPException(400, 'Could not open image')

    device = app.state.device
    t0 = time.time()

    # ── Step 1: Router ────────────────────────────────────────────────────
    router_tensor = app.state.router_transform(image=img_np)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        router_feats = app.state.router_backbone(router_tensor)
        router_logits = app.state.router_head(router_feats)
        router_probs = torch.softmax(router_logits.float(), dim=1).cpu().numpy()[0]

    router_crop_idx = int(np.argmax(router_probs))
    router_crop = ROUTER_CLASSES[router_crop_idx]
    router_conf = float(router_probs[router_crop_idx])

    # Allow user override
    active_crop = override_crop if override_crop in ('okra', 'brassica') else router_crop

    # ── Step 2: Model 2 (if okra or brassica) ─────────────────────────────
    if active_crop not in ('okra', 'brassica'):
        latency = int((time.time() - t0) * 1000)
        return JSONResponse({
            'router_crop': router_crop,
            'router_confidence': round(router_conf, 4),
            'router_all': {ROUTER_CLASSES[i]: round(float(router_probs[i]), 4) for i in range(ROUTER_NUM_CLASSES)},
            'active_crop': active_crop,
            'supported': False,
            'message': f'{active_crop} specialist (Model 3) not yet deployed. Coming soon.',
            'latency_ms': latency,
        })

    model2_tensor = app.state.model2_transform(image=img_np)['image'].unsqueeze(0).to(device)

    # Crop-gated prediction (adaptive: hard mask at high conf, soft at medium, none at low)
    probs, logits = crop_gated_predict(app.state.model2, model2_tensor, active_crop, router_conf, device)

    # Tier 1 post-processing: vein detection + confusion correction + margin fix
    # Safety validated: +0.0007 macro F1, black_rot +1.4%, no class degraded >0.6%
    probs = apply_tier1_fixes(probs, img_np, active_crop)

    # Smart diagnosis
    primary, primary_conf, alert_type, secondary, secondary_conf = smart_diagnosis(probs, active_crop)

    # GradCAM++
    primary_idx = M2_CLASSES.index(primary)
    heatmap_b64 = generate_gradcam(app.state.model2, model2_tensor, img_np, primary_idx)

    latency = int((time.time() - t0) * 1000)

    # Build response
    info = DISEASE_INFO.get(primary, {})
    sec_info = DISEASE_INFO.get(secondary, {})

    # Only include relevant crop probabilities
    relevant_indices = OKRA_INDICES if active_crop == 'okra' else BRASSICA_INDICES
    relevant_probs = {M2_CLASSES[i]: round(float(probs[i]), 4) for i in relevant_indices}

    return JSONResponse({
        # Router results
        'router_crop': router_crop,
        'router_confidence': round(router_conf, 4),
        'router_all': {ROUTER_CLASSES[i]: round(float(router_probs[i]), 4) for i in range(ROUTER_NUM_CLASSES)},
        'active_crop': active_crop,
        'crop_overridden': override_crop is not None,
        'supported': True,
        # Disease results
        'primary_disease': primary,
        'primary_confidence': round(primary_conf, 4),
        'primary_name': info.get('name', primary),
        'primary_emoji': info.get('emoji', '?'),
        'primary_desc': info.get('desc', ''),
        'primary_action': info.get('action', ''),
        'primary_severity': info.get('severity', 'Unknown'),
        'primary_color': info.get('color', '#888'),
        # Alert type
        'alert_type': alert_type,
        # Secondary
        'secondary_disease': secondary,
        'secondary_confidence': round(secondary_conf, 4),
        'secondary_name': sec_info.get('name', secondary),
        'secondary_emoji': sec_info.get('emoji', '?'),
        # All probabilities (crop-gated)
        'disease_probabilities': relevant_probs,
        # Heatmap
        'heatmap_b64': heatmap_b64,
        'latency_ms': latency,
    })


@app.get('/health')
async def health():
    return {'status': 'ok', 'pipeline': 'Router -> Model 2',
            'router_classes': ROUTER_CLASSES, 'model2_classes': M2_CLASSES}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("run_unified_server:app", host="0.0.0.0", port=8005, reload=False)
