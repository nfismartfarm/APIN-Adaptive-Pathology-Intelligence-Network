# app/inference.py
"""
Inference pipeline: preprocess -> MC Dropout -> temperature scaling -> Grad-CAM.

[FIX GAP 8]  apply_clahe defined inline — does NOT import from training.transforms.
[FIX GAP 7]  Grad-CAM target: model.fpn.out_p3 (was wrongly documented as output_p3).
[FIX GAP 36] threading.Lock protects MC Dropout state changes.
[FIX GAP 10] No merge_diagnoses() function — diagnosis merging is inline in run_inference().
"""

import os
import io
import base64
import json
import threading
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from app.config import (
    DEVICE, BEST_MODEL, TEMP_PATH, DIAG_JSON,
    NUM_CLASSES, CLASS_NAMES, OKRA_INDICES, BRASSICA_INDICES,
    HEALTHY_INDICES, CROP_NAMES, CROP_FROM_IDX,
    DISEASE_THRESH, OOD_CONF_THRESH, OOD_UNC_THRESH, MC_PASSES,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    PER_CLASS_THRESHOLDS, COPRED_GAP_THRESH, MAX_COPREDICTIONS,
    HEALTHY_SUPPRESS_MIN,
)


# ── [FIX GAP 8,11] apply_clahe defined INLINE — NOT imported from training ─
def apply_clahe(image: np.ndarray, clip_limit=2.0, tile_size=(8, 8)) -> np.ndarray:
    """
    CLAHE per RGB channel. Defined inline — do NOT import from training.transforms.
    Both this and the training version use only cv2 and numpy.
    Input/output: uint8 numpy array [H, W, 3].
    """
    clahe  = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = clahe.apply(image[:, :, c])
    return result


# ── Lock for MC Dropout state modification [FIX GAP 36] ────────────────────
_mc_dropout_lock = threading.Lock()


def preprocess_for_inference(image_np: np.ndarray) -> torch.Tensor:
    """
    Applies CLAHE, resizes to IMG_SIZE, normalises with ImageNet stats.
    Returns float32 Tensor [1, 3, H, W].
    image_np: uint8 numpy [H, W, 3].
    """
    img = apply_clahe(image_np)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = (img - mean) / std
    img  = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return img


class _DiseaseLogitsWrapper(nn.Module):
    """Wrapper that returns only disease_logits from PlantDiseaseModel.
    pytorch_grad_cam expects model(x) -> single tensor, but our model
    returns (crop_logits, disease_logits, severity_logits). This wrapper
    extracts disease_logits so GradCAM can compute gradients correctly."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Expose fpn so target_layer resolution works
        self.fpn = model.fpn
        self.backbone = model.backbone

    def forward(self, x):
        _, disease_logits, _ = self.model(x)
        return disease_logits


def generate_heatmap(model, image_tensor: torch.Tensor, original_np: np.ndarray) -> str:
    """
    Generates Grad-CAM heatmap overlay on the original image.
    Returns base64 PNG string.

    [FIX GAP 7] Target layer is model.fpn.out_p3 — do NOT use model.fpn.output_p3
    (that attribute does not exist and will raise AttributeError).

    Uses _DiseaseLogitsWrapper because pytorch_grad_cam expects model(x) to
    return a single tensor, but PlantDiseaseModel returns a 3-tuple.

    The model must be in eval mode with no Dropout active (not MC mode).
    Run all MC passes first, then call this function.
    """
    model.eval()
    wrapper = _DiseaseLogitsWrapper(model)
    wrapper.eval()

    # [FIX GAP 7] Correct attribute name: out_p3
    target_layer = wrapper.fpn.out_p3

    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=image_tensor.to(DEVICE))[0]

    # Overlay on original image resized to match
    orig_resized = cv2.resize(original_np, IMG_SIZE)
    orig_float   = orig_resized.astype(np.float32) / 255.0
    overlay      = show_cam_on_image(orig_float, grayscale, use_rgb=True)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_inference(model, image_np: np.ndarray) -> dict:
    """
    Full inference pipeline. Returns result dict with all fields.

    [FIX GAP 36] Uses _mc_dropout_lock to serialise MC Dropout state changes.
    Multiple concurrent requests cannot race on model.eval()/module.train().

    [FIX GAP 10] Diagnosis merging is inline — no merge_diagnoses() function.
    """
    image_tensor = preprocess_for_inference(image_np).to(DEVICE)

    # ── Load temperature scalars ────────────────────────────────────────────
    T_disease = T_crop = T_severity = 1.0
    if os.path.exists(TEMP_PATH):
        temp_data  = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        T_disease  = float(temp_data.get('T_disease',  1.0))
        T_crop     = float(temp_data.get('T_crop',     1.0))
        T_severity = float(temp_data.get('T_severity', 1.0))

    # ── MC Dropout passes ──────────────────────────────────────────────────
    # [FIX GAP 36] Lock ensures no race condition on module state
    with _mc_dropout_lock:
        # Set model to eval, then set only Dropout layers to train mode
        # BatchNorm stays in eval — prevents batch-of-1 statistics problem
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        mc_disease  = []
        mc_crop     = []
        mc_severity = []
        with torch.no_grad():
            for _ in range(MC_PASSES):
                c_log, d_log, s_log = model(image_tensor)
                mc_disease.append(torch.sigmoid(d_log / T_disease).cpu())
                mc_crop.append(torch.softmax(c_log / T_crop, dim=1).cpu())
                mc_severity.append(torch.softmax(s_log / T_severity, dim=1).cpu())

        # Restore full eval mode
        model.eval()

    mc_disease  = torch.stack(mc_disease, dim=0)   # [MC, 1, NUM_CLASSES]
    mc_crop     = torch.stack(mc_crop,    dim=0)   # [MC, 1, 2]
    mc_severity = torch.stack(mc_severity,dim=0)   # [MC, 1, 3]

    mean_dis = mc_disease.mean(dim=0).squeeze(0)   # [NUM_CLASSES]
    std_dis  = mc_disease.std(dim=0).squeeze(0)    # [NUM_CLASSES]
    mean_crop  = mc_crop.mean(dim=0).squeeze(0)    # [2]
    mean_sev   = mc_severity.mean(dim=0).squeeze(0) # [3]

    uncertainty = float(std_dis.mean())

    # ── Crop prediction ────────────────────────────────────────────────────
    crop_idx  = int(mean_crop.argmax())
    crop_conf = float(mean_crop.max())
    crop_name = CROP_NAMES[crop_idx]

    # ── OOD detection ──────────────────────────────────────────────────────
    ood_flagged = (crop_conf < OOD_CONF_THRESH or uncertainty > OOD_UNC_THRESH)

    # ── Disease predictions (5 post-processing improvements) ────────────────

    # FIX 1: Crop gate — zero out predictions for the other crop
    gate = torch.zeros(NUM_CLASSES)
    relevant = OKRA_INDICES if crop_idx == 0 else BRASSICA_INDICES
    for i in relevant:
        gate[i] = 1.0
    gate      = gate.to(mean_dis.device)
    gated_dis = mean_dis * gate
    gated_std = std_dis * gate  # also gate the uncertainty

    # FIX 2: Per-class thresholds + MC lower-bound thresholding
    # A disease is predicted only if (mean - std) > per-class threshold
    # This requires consistent confidence across all MC passes
    healthy_cls = 'okra_healthy' if crop_idx == 0 else 'brassica_healthy'
    candidates = []  # list of (class_name, confidence)
    for i in range(NUM_CLASSES):
        cls = CLASS_NAMES[i]
        if gated_dis[i].item() == 0:
            continue  # wrong crop, gated out
        thresh = PER_CLASS_THRESHOLDS.get(cls, DISEASE_THRESH)
        mean_val = gated_dis[i].item()
        std_val  = gated_std[i].item()
        # FIX 3: MC lower-bound thresholding — use (mean - 0.5*std) for softer check
        # Full (mean - std) was too aggressive for thin classes with high variance
        lower_bound = mean_val - 0.5 * std_val
        if lower_bound > thresh:
            candidates.append((cls, mean_val))

    # FIX 4: Bidirectional healthy suppression
    # If any disease has confidence > HEALTHY_SUPPRESS_MIN, suppress healthy
    disease_candidates = [(c, conf) for c, conf in candidates if c != healthy_cls]
    healthy_candidate  = [(c, conf) for c, conf in candidates if c == healthy_cls]
    max_disease_conf   = max((conf for _, conf in disease_candidates), default=0.0)

    if max_disease_conf > HEALTHY_SUPPRESS_MIN and healthy_candidate:
        # Strong disease signal — remove healthy from candidates
        candidates = disease_candidates
    elif healthy_candidate and disease_candidates:
        # Healthy fires AND diseases fire — keep only healthy (original suppression)
        candidates = healthy_candidate

    # FIX 5: Confidence gap suppression for co-predictions
    if len(candidates) > 1:
        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        filtered = [candidates[0]]
        for cls, conf in candidates[1:]:
            gap = filtered[0][1] - conf
            if gap <= COPRED_GAP_THRESH:
                filtered.append((cls, conf))
            # else: suppress — too far below the top prediction
        candidates = filtered

    # FIX 6: Max co-predictions cap (max 2 diseases per image)
    if len(candidates) > MAX_COPREDICTIONS:
        candidates = candidates[:MAX_COPREDICTIONS]

    # Final detected list
    detected = [cls for cls, _ in candidates]
    if not detected:
        detected = [healthy_cls]

    # Confidence = mean of detected class probabilities
    detected_idx = [CLASS_NAMES.index(c) for c in detected]
    confidence   = float(mean_dis[detected_idx].mean()) if detected_idx else 0.5

    # ── Severity ───────────────────────────────────────────────────────────
    sev_idx    = int(mean_sev.argmax())
    sev_labels = ['mild', 'moderate', 'severe']
    severity   = sev_labels[sev_idx]
    sev_std    = float(mean_sev.std())
    # [low, high] interval based on MC uncertainty
    sev_low    = max(0.0, float(mean_sev[sev_idx]) - sev_std)
    sev_high   = min(1.0, float(mean_sev[sev_idx]) + sev_std)

    # ── Grad-CAM ───────────────────────────────────────────────────────────
    try:
        heatmap_b64 = generate_heatmap(model, image_tensor, image_np)
    except Exception as e:
        import traceback
        print(f"Grad-CAM failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        heatmap_b64 = ''

    # ── Diagnosis lookup [FIX GAP 10] inline merging ──────────────────────
    with open(DIAG_JSON, 'r', encoding='utf-8') as f:
        diag_db = json.load(f)

    # Merge treatment/prevention across all detected diseases (inline, no function)
    treatment  = []
    prevention = []
    urgency    = 'Low'
    urgency_reason = ''
    urgency_priority = {'High': 3, 'Medium': 2, 'Low': 1}
    for cls in detected:
        if cls in diag_db:
            entry = diag_db[cls]
            treatment.extend(entry.get('treatment', []))
            prevention.extend(entry.get('prevention', []))
            entry_urgency = entry.get('urgency', 'Low')
            if urgency_priority.get(entry_urgency, 0) > urgency_priority.get(urgency, 0):
                urgency        = entry_urgency
                urgency_reason = entry.get('urgency_reason', '')

    # Deduplicate while preserving order
    seen = set()
    treatment  = [t for t in treatment  if not (t in seen or seen.add(t))]
    seen       = set()
    prevention = [p for p in prevention if not (p in seen or seen.add(p))]

    # Per-class probabilities for frontend display
    all_probs = {}
    for i, cls in enumerate(CLASS_NAMES):
        all_probs[cls] = round(float(mean_dis[i]), 4)

    return {
        'crop'            : crop_name,
        'crop_confidence' : round(crop_conf, 3),
        'diseases'        : detected,
        'confidence'      : round(confidence, 3),
        'uncertainty'     : round(uncertainty, 3),
        'severity'        : severity,
        'severity_interval': [round(sev_low, 3), round(sev_high, 3)],
        'treatment'       : treatment,
        'prevention'      : prevention,
        'urgency'         : urgency,
        'urgency_reason'  : urgency_reason,
        'heatmap_b64'     : heatmap_b64,
        'ood_flagged'     : ood_flagged,
        'all_probabilities': all_probs,
    }
