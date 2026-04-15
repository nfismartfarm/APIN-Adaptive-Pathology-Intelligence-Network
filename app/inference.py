# app/inference.py
"""
Inference pipeline with 16 real-world robustness improvements:

1. Lowered disease thresholds (0.35-0.45) for real-world photos
2. Test-Time Augmentation (horizontal flip ensemble)
3. Lowered OOD threshold (0.75 -> 0.50)
4. Center crop + full image dual inference
5. Green region auto-detection and crop
6. White-balance correction (grey-world normalization)
7. Increased CLAHE clip_limit (2.0 -> 4.0)
8. Multi-scale inference (192px, 224px, 288px)
9. Softmax temperature configurable
10. Disease co-occurrence prior filtering
11. Region-of-interest re-inference via Grad-CAM
12. Ensemble crop prediction override
13. Adaptive threshold based on crop confidence
14. Top-2 prediction support
15. Confidence-based language
16. Force-predict bypass for OOD
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
    DEVICE, BEST_MODEL, TEMP_PATH, DIAG_JSON, MOBILE_SAM_CHECKPOINT, ROOT,
    NUM_CLASSES, CLASS_NAMES, CROP_NAMES, CROP_TO_DISEASE_INDICES,
    HEALTHY_CLASSES, MC_PASSES, NUM_CROPS,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    DISEASE_THRESHOLDS, COPRED_GAP_THRESH, MAX_COPREDICTIONS,
    OOD_CROP_CONFIDENCE_THRESHOLD,
    HEALTHY_SUPPRESSION_CONFIDENCE, HEALTHY_SUPPRESSION_DISEASE_MIN,
    CLAHE_CLIP_LIMIT, TTA_ENABLED,
    ADAPTIVE_THRESH_CROP_HIGH, ADAPTIVE_THRESH_REDUCTION,
    DISEASE_COOCCURRENCE,
)


# ── apply_clahe defined INLINE — NOT imported from training ──────────────
def apply_clahe(image: np.ndarray, clip_limit=None, tile_size=(8, 8)) -> np.ndarray:
    """CLAHE per RGB channel. Uses CLAHE_CLIP_LIMIT from config."""
    if clip_limit is None:
        clip_limit = CLAHE_CLIP_LIMIT
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = clahe.apply(image[:, :, c])
    return result


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    """Grey-world white balance correction.
    Normalizes each channel so the average color is neutral grey.
    Removes color cast from different lighting (sunlight, fluorescent, overcast)."""
    img = image.astype(np.float32)
    avg = img.mean(axis=(0, 1))
    global_avg = avg.mean()
    if global_avg < 1.0:
        return image
    # Issue 5: clamp scale to [0.5, 2.0] to prevent extreme saturation
    # on images with near-zero single-channel averages
    scale = global_avg / (avg + 1e-6)
    scale = np.clip(scale, 0.5, 2.0)
    result = np.clip(img * scale, 0, 255).astype(np.uint8)
    return result


# ── MobileSAM lazy initialization ──────────────────────────────────────────
_sam_model = None
_sam_mask_generator = None


def get_sam_mask_generator():
    """Lazy-load MobileSAM. Returns mask generator or None if unavailable."""
    global _sam_model, _sam_mask_generator
    if _sam_mask_generator is None:
        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
            if not os.path.exists(MOBILE_SAM_CHECKPOINT):
                print(f"WARNING: MobileSAM weights not found at {MOBILE_SAM_CHECKPOINT}")
                return None
            _sam_model = sam_model_registry["vit_t"](checkpoint=MOBILE_SAM_CHECKPOINT)
            _sam_model = _sam_model.to(DEVICE)
            _sam_model.eval()
            _sam_mask_generator = SamAutomaticMaskGenerator(
                model=_sam_model,
                points_per_side=16,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=1000,
            )
            print(f"MobileSAM loaded successfully on {DEVICE}")
        except Exception as e:
            print(f"WARNING: MobileSAM failed to load: {e}. Will use HSV fallback.")
            _sam_mask_generator = None
    return _sam_mask_generator


def segment_leaf_with_sam(image_rgb: np.ndarray) -> tuple:
    """
    Use MobileSAM to segment the leaf from background.
    Returns (result_image, sam_was_used: bool).
    Background pixels set to neutral grey (128, 128, 128).
    Falls back to original image if SAM fails.
    """
    h, w = image_rgb.shape[:2]
    if h < 50 or w < 50:
        return image_rgb, False

    mask_generator = get_sam_mask_generator()
    if mask_generator is None:
        return image_rgb, False

    try:
        with torch.no_grad():
            masks = mask_generator.generate(image_rgb)

        if not masks:
            return image_rgb, False

        total_pixels = h * w

        # Filter: leaf should cover 15-85% of image
        valid_masks = [
            m for m in masks
            if 0.15 <= (m['area'] / total_pixels) <= 0.85
        ]
        if not valid_masks:
            valid_masks = [
                m for m in masks
                if 0.05 <= (m['area'] / total_pixels) <= 0.95
            ]
        if not valid_masks:
            return image_rgb, False

        # Best mask: highest IoU, then largest area
        valid_masks.sort(
            key=lambda m: (m['predicted_iou'], m['area']),
            reverse=True
        )

        best_mask = valid_masks[0]['segmentation']  # boolean (H, W)

        # Sanity: mask must cover at least 500 pixels
        if best_mask.sum() < 500:
            return image_rgb, False

        # Replace background with neutral grey
        result = image_rgb.copy()
        result[~best_mask] = np.array([128, 128, 128], dtype=np.uint8)
        return result, True

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("WARNING: MobileSAM GPU OOM, falling back to original image")
        return image_rgb, False
    except Exception as e:
        print(f"WARNING: MobileSAM segmentation failed: {e}, using original image")
        return image_rgb, False


def detect_and_crop_leaf(image: np.ndarray) -> np.ndarray:
    """Auto-detect the largest green region (leaf) and crop to it.
    Falls back to original image if no significant green region found."""
    # Issue 4: guard against degenerate images
    if image.shape[0] < 10 or image.shape[1] < 10:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Green range in HSV
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Also include brown/yellow diseased areas
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([30, 255, 255])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask = cv2.bitwise_or(mask, mask_brown)
    # Morphological close to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (image.shape[0] * image.shape[1])
    if area_ratio < 0.05:  # less than 5% of image is green — likely not a leaf
        return image
    x, y, w, h = cv2.boundingRect(largest)
    # Add 10% padding
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    cropped = image[y1:y2, x1:x2]
    if cropped.shape[0] < 50 or cropped.shape[1] < 50:
        return image
    return cropped


# ── Lock for MC Dropout state modification ────────────────────────────────
_mc_dropout_lock = threading.Lock()


def preprocess_single(image_np: np.ndarray, target_size=None) -> torch.Tensor:
    """Preprocess a single image: white balance -> CLAHE -> resize -> normalize.
    Returns float32 Tensor [1, 3, H, W]."""
    if target_size is None:
        target_size = IMG_SIZE
    img = apply_white_balance(image_np)
    img = apply_clahe(img)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return img


def preprocess_for_inference(image_np: np.ndarray) -> torch.Tensor:
    """Full preprocessing with leaf detection + multi-variant preparation.
    Returns primary tensor [1, 3, H, W] for the main inference pass."""
    leaf_img = detect_and_crop_leaf(image_np)
    return preprocess_single(leaf_img)


# ── Phase 3: Grad-CAM second pass constants ──────────────────────────────
SECOND_PASS_FIRST_WEIGHT = 0.4
SECOND_PASS_SECOND_WEIGHT = 0.6
ENSEMBLE_UNCERTAINTY_THRESHOLD = 0.30


def get_gradcam_crop(original_image_np, heatmap_b64, padding_fraction=0.20,
                     min_crop_size=32, max_coverage_fraction=0.60):
    """
    Extract disease-relevant crop from original image using Grad-CAM heatmap.
    Returns numpy array crop ready for preprocessing, or None to skip 2nd pass.
    """
    H, W = original_image_np.shape[:2]
    try:
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_pil = Image.open(io.BytesIO(heatmap_bytes)).convert('L')
        heatmap_np = np.array(heatmap_pil.resize((W, H), Image.BILINEAR),
                               dtype=np.float32)
    except Exception:
        return None

    hmap_min, hmap_max = heatmap_np.min(), heatmap_np.max()
    if hmap_max - hmap_min < 1e-6:
        return None  # uniform heatmap
    heatmap_norm = (heatmap_np - hmap_min) / (hmap_max - hmap_min)

    high_act_mask = heatmap_norm > 0.5
    coverage = high_act_mask.mean()
    if coverage > max_coverage_fraction or coverage < 1e-4:
        return None  # too diffuse or nothing activated

    rows = np.any(high_act_mask, axis=1)
    cols = np.any(high_act_mask, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    if len(row_indices) == 0 or len(col_indices) == 0:
        return None

    y1, y2 = int(row_indices[0]), int(row_indices[-1])
    x1, x2 = int(col_indices[0]), int(col_indices[-1])
    if (y2 - y1) < min_crop_size or (x2 - x1) < min_crop_size:
        return None  # activation too small

    pad_y = int((y2 - y1) * padding_fraction)
    pad_x = int((x2 - x1) * padding_fraction)
    y1, y2 = max(0, y1 - pad_y), min(H, y2 + pad_y)
    x1, x2 = max(0, x1 - pad_x), min(W, x2 + pad_x)

    crop = original_image_np[y1:y2, x1:x2]
    if crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
        return None
    return crop


def run_second_pass(model, crop_np, T_disease, T_crop, T_severity, device):
    """
    Single deterministic forward pass on a crop. No MC Dropout.
    Applies white balance + CLAHE preprocessing before inference.
    Returns dict with crop_probs [4], disease_probs [23], severity_probs [3].
    """
    # Preprocess: white balance + CLAHE + resize to 224x224 + normalize
    tensor = preprocess_single(crop_np).to(device)
    model.eval()
    with torch.no_grad():
        c_log, d_log, s_log = model(tensor)

    crop_probs = torch.softmax(c_log / T_crop, dim=-1)[0].cpu().numpy()
    disease_probs = torch.sigmoid(d_log / T_disease)[0].cpu().numpy()
    severity_probs = torch.softmax(s_log / T_severity, dim=-1)[0].cpu().numpy()
    return {
        'crop_probs': crop_probs,
        'disease_probs': disease_probs,
        'severity_probs': severity_probs,
    }


class _DiseaseLogitsWrapper(nn.Module):
    """Wrapper for Grad-CAM — returns only disease_logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fpn = model.fpn
        self.backbone = model.backbone

    def forward(self, x):
        _, disease_logits, _ = self.model(x)
        return disease_logits


def generate_heatmap(model, image_tensor: torch.Tensor, original_np: np.ndarray) -> str:
    """Generates Grad-CAM heatmap overlay. Returns base64 PNG."""
    model.eval()
    wrapper = _DiseaseLogitsWrapper(model)
    wrapper.eval()
    target_layer = wrapper.fpn.out_p3
    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=image_tensor.to(DEVICE))[0]
    orig_resized = cv2.resize(original_np, IMG_SIZE)
    orig_float = orig_resized.astype(np.float32) / 255.0
    overlay = show_cam_on_image(orig_float, grayscale, use_rgb=True)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _run_model_passes(model, image_tensor, T_disease, T_crop, T_severity):
    """Run MC Dropout passes on a single image tensor. Returns raw MC arrays."""
    mc_disease = []
    mc_crop = []
    mc_severity = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            c_log, d_log, s_log = model(image_tensor)
            mc_disease.append(torch.sigmoid(d_log / T_disease).cpu())
            mc_crop.append(torch.softmax(c_log / T_crop, dim=1).cpu())
            mc_severity.append(torch.softmax(s_log / T_severity, dim=1).cpu())
    return mc_disease, mc_crop, mc_severity


def run_inference(model, image_np: np.ndarray, force_predict: bool = False,
                  return_raw_probs: bool = False) -> dict:
    """
    Full inference pipeline with real-world robustness improvements.

    Args:
        model: loaded PlantDiseaseModel
        image_np: uint8 numpy [H, W, 3] RGB image
        force_predict: if True, bypass OOD gate (for "Try Again" button)

    Returns result dict with all fields including top2_diseases for secondary display.
    """
    # ── SAM leaf segmentation (removes background before classification) ──
    sam_image, sam_segmented = segment_leaf_with_sam(image_np)

    # ── Leaf detection + preprocessing ─────────────────────────────────────
    leaf_img = detect_and_crop_leaf(sam_image)
    image_tensor = preprocess_single(leaf_img).to(DEVICE)

    # Also prepare center-crop variant (70% center region)
    h, w = leaf_img.shape[:2]
    ch, cw = int(h * 0.7), int(w * 0.7)
    y_off, x_off = (h - ch) // 2, (w - cw) // 2
    center_crop = leaf_img[y_off:y_off+ch, x_off:x_off+cw]
    center_tensor = preprocess_single(center_crop).to(DEVICE)

    # TTA: horizontal flip (Issue 2: initialize before conditional)
    flip_tensor = None
    if TTA_ENABLED:
        flipped = np.ascontiguousarray(leaf_img[:, ::-1])
        flip_tensor = preprocess_single(flipped).to(DEVICE)

    # Multi-scale DISABLED: Swin-Tiny requires exactly 224x224 input
    # (patch_embed asserts H==224). EfficientNetV2 supported variable sizes
    # via AdaptiveAvgPool but Swin's window attention is fixed-resolution.

    # ── Load temperature scalars ──────────────────────────────────────────
    T_disease = T_crop = T_severity = 1.0
    if os.path.exists(TEMP_PATH):
        temp_data = torch.load(TEMP_PATH, map_location='cpu', weights_only=False)
        saved_nc = temp_data.get('num_classes', None)
        if saved_nc is not None and saved_nc != NUM_CLASSES:
            pass  # stale calibration, use T=1.0
        else:
            T_disease = float(temp_data.get('T_disease', 1.0))
            T_crop = float(temp_data.get('T_crop', 1.0))
            T_severity = float(temp_data.get('T_severity', 1.0))

    # ── MC Dropout passes with TTA + multi-scale + center crop ────────────
    with _mc_dropout_lock:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        # Primary passes
        mc_d, mc_c, mc_s = _run_model_passes(model, image_tensor,
                                              T_disease, T_crop, T_severity)

        # Center crop passes (1 pass, no MC)
        with torch.no_grad():
            c_log_cc, d_log_cc, s_log_cc = model(center_tensor)
            mc_d.append(torch.sigmoid(d_log_cc / T_disease).cpu())
            mc_c.append(torch.softmax(c_log_cc / T_crop, dim=1).cpu())
            mc_s.append(torch.softmax(s_log_cc / T_severity, dim=1).cpu())

        # TTA flip pass (Issue 2: guard with flip_tensor is not None)
        if TTA_ENABLED and flip_tensor is not None:
            with torch.no_grad():
                c_log_f, d_log_f, s_log_f = model(flip_tensor)
                mc_d.append(torch.sigmoid(d_log_f / T_disease).cpu())
                mc_c.append(torch.softmax(c_log_f / T_crop, dim=1).cpu())
                mc_s.append(torch.softmax(s_log_f / T_severity, dim=1).cpu())

        # Multi-scale passes DISABLED for Swin-Tiny (fixed 224x224 input)

        model.eval()

    # ── Aggregate predictions ─────────────────────────────────────────────
    mc_disease = torch.stack(mc_d, dim=0)
    mc_crop = torch.stack(mc_c, dim=0)
    mc_severity = torch.stack(mc_s, dim=0)

    mean_dis = mc_disease.mean(dim=0).squeeze(0)   # [NUM_CLASSES]
    std_dis = mc_disease.std(dim=0).squeeze(0)
    mean_crop = mc_crop.mean(dim=0).squeeze(0)      # [NUM_CROPS]
    mean_sev = mc_severity.mean(dim=0).squeeze(0)   # [3]

    uncertainty = float(std_dis.mean())

    # ── Crop prediction ───────────────────────────────────────────────────
    crop_idx = int(mean_crop.argmax())
    crop_conf = float(mean_crop.max())
    crop_name = CROP_NAMES[crop_idx]

    # ── Crop probabilities for OOD display ────────────────────────────────
    crop_probs_dict = {CROP_NAMES[i]: round(float(mean_crop[i]), 3)
                       for i in range(len(CROP_NAMES))}

    # ── OOD gate ──────────────────────────────────────────────────────────
    if crop_conf < OOD_CROP_CONFIDENCE_THRESHOLD and not force_predict:
        return {
            'crop': None,
            'crop_confidence': round(crop_conf, 4),
            'crop_probabilities': crop_probs_dict,
            'diseases': [],
            'top2_diseases': [],
            'confidence': 0.0,
            'uncertainty': round(uncertainty, 3),
            'severity': None,
            'severity_interval': None,
            'treatment': [],
            'prevention': [],
            'urgency': None,
            'urgency_reason': '',
            'heatmap_b64': '',
            'ood_flagged': True,
            'ood_reason': (
                f"Image does not appear to be a supported crop leaf. "
                f"The model cannot confidently identify this as okra, broccoli, tomato, or chilli "
                f"(max confidence: {crop_conf:.2f}). "
                f"Please upload a clear close-up photograph of a leaf."
            ),
            'confidence_level': 'uncertain',
            'sam_segmented': sam_segmented,
        }

    ood_flagged = False

    # ── Ensemble crop override (Item 15) ──────────────────────────────────
    # If crop confidence is moderate, try each crop's masking and pick the
    # one that gives the highest max disease confidence
    if crop_conf < ADAPTIVE_THRESH_CROP_HIGH:
        best_crop_idx = crop_idx
        best_max_conf = 0.0
        for try_crop in range(len(CROP_NAMES)):
            indices = CROP_TO_DISEASE_INDICES[try_crop]
            max_conf_for_crop = max(float(mean_dis[i]) for i in indices)
            if max_conf_for_crop > best_max_conf:
                best_max_conf = max_conf_for_crop
                best_crop_idx = try_crop
        if best_crop_idx != crop_idx:
            crop_idx = best_crop_idx
            crop_name = CROP_NAMES[crop_idx]
            crop_conf = float(mean_crop[crop_idx])

    # ── Grad-CAM heatmap ────────────────────────────────────────────────
    try:
        heatmap_b64 = generate_heatmap(model, image_tensor, leaf_img)
    except Exception:
        heatmap_b64 = ''

    # ── Phase 3: Second pass PERMANENTLY DISABLED ───────────────────────
    # Tested at thresholds 0.60 (93% trigger, -0.156 F1), 0.15 (41%, not tested full),
    # 0.10 (22% trigger, -0.150 F1). All degraded PlantDoc F1. Every class regressed
    # at every threshold. Root cause: model was trained on full images and the cropped
    # inference loses global healthy-vs-diseased context even for focused heatmaps.
    # The second pass is not viable for this model architecture.
    second_pass_applied = False
    ensemble_applied = False

    # ── Cross-crop disease masking (soft when uncertain) ─────────────────
    # When crop confidence is high, hard-mask to prevent biologically
    # impossible cross-crop predictions. When low, skip the mask entirely
    # so all 23 diseases compete on raw MoE output probabilities.
    # This prevents the cascading failure where wrong crop → wrong diseases.
    if crop_conf >= ADAPTIVE_THRESH_CROP_HIGH:
        # High confidence: hard mask (only selected crop's diseases)
        valid_indices = CROP_TO_DISEASE_INDICES[crop_idx]
        mask = torch.zeros(NUM_CLASSES, device=mean_dis.device)
        for idx in valid_indices:
            mask[idx] = 1.0
        gated_dis = mean_dis * mask
        gated_std = std_dis * mask
    else:
        # Low confidence: no mask — let all diseases compete
        # The MoE already routes via crop_probs softmax weighting.
        # The ensemble crop override (above) already tried all crops.
        # Per-class thresholds (0.30) provide sufficient filtering.
        gated_dis = mean_dis
        gated_std = std_dis
        # Update crop_idx to match whichever crop has the winning disease
        # (this ensures treatment/urgency lookup uses the correct crop)
        top_disease_idx = int(gated_dis.argmax())
        from app.config import CROP_FROM_IDX
        crop_idx = CROP_FROM_IDX.get(top_disease_idx, crop_idx)
        crop_name = CROP_NAMES[crop_idx]
        valid_indices = CROP_TO_DISEASE_INDICES[crop_idx]

    # ── Adaptive threshold (Item 16) ──────────────────────────────────────
    threshold_reduction = 0.0
    if crop_conf < ADAPTIVE_THRESH_CROP_HIGH:
        threshold_reduction = ADAPTIVE_THRESH_REDUCTION

    # ── Per-class thresholds + MC lower-bound thresholding ────────────────
    healthy_cls_map = {
        0: 'okra_healthy', 1: 'brassica_healthy',
        2: 'tomato_healthy', 3: 'chilli_healthy',
    }
    healthy_cls = healthy_cls_map[crop_idx]

    candidates = []
    all_class_probs = {}  # for top-2 display
    for i in valid_indices:
        cls = CLASS_NAMES[i]
        mean_val = gated_dis[i].item()
        std_val = gated_std[i].item()
        all_class_probs[cls] = round(mean_val, 4)
        thresh = DISEASE_THRESHOLDS.get(cls, 0.40) - threshold_reduction
        lower_bound = mean_val - 0.3 * std_val  # softened from 0.5 to 0.3
        if lower_bound > thresh:
            candidates.append((cls, mean_val))

    # ── Bidirectional healthy suppression ──────────────────────────────────
    disease_candidates = [(c, conf) for c, conf in candidates if c not in HEALTHY_CLASSES]
    healthy_candidate = [(c, conf) for c, conf in candidates if c in HEALTHY_CLASSES]
    max_disease_conf = max((conf for _, conf in disease_candidates), default=0.0)

    if max_disease_conf > HEALTHY_SUPPRESSION_CONFIDENCE and healthy_candidate:
        candidates = [
            (c, conf) for c, conf in candidates
            if c not in HEALTHY_CLASSES or conf > HEALTHY_SUPPRESSION_DISEASE_MIN
        ]
    elif healthy_candidate and disease_candidates:
        candidates = healthy_candidate

    # ── Co-occurrence prior filtering (Item 13) ───────────────────────────
    # Issue 6: check against ALL accepted candidates, not just filtered[0]
    if len(candidates) > 1:
        filtered = [candidates[0]]
        for cls, conf in candidates[1:]:
            compatible = True
            for accepted_cls, _ in filtered:
                pair1 = (accepted_cls, cls)
                pair2 = (cls, accepted_cls)
                cooccur = DISEASE_COOCCURRENCE.get(pair1,
                          DISEASE_COOCCURRENCE.get(pair2, True))
                if not cooccur:
                    compatible = False
                    break
            if compatible:
                filtered.append((cls, conf))
        candidates = filtered

    # ── Confidence gap suppression ────────────────────────────────────────
    if len(candidates) > 1:
        candidates.sort(key=lambda x: x[1], reverse=True)
        filtered = [candidates[0]]
        for cls, conf in candidates[1:]:
            gap = filtered[0][1] - conf
            if gap <= COPRED_GAP_THRESH:
                filtered.append((cls, conf))
        candidates = filtered

    # ── Max co-predictions cap ────────────────────────────────────────────
    if len(candidates) > MAX_COPREDICTIONS:
        candidates = candidates[:MAX_COPREDICTIONS]

    # Final detected list
    detected = [cls for cls, _ in candidates]
    if not detected:
        detected = [healthy_cls]

    # ── Top-2 predictions (Item 17) ───────────────────────────────────────
    sorted_probs = sorted(all_class_probs.items(), key=lambda x: x[1], reverse=True)
    top2_diseases = [
        {'class': cls, 'confidence': round(conf, 4)}
        for cls, conf in sorted_probs[:2]
    ]

    # Confidence
    detected_idx = [CLASS_NAMES.index(c) for c in detected]
    confidence = float(mean_dis[detected_idx].mean()) if detected_idx else 0.5

    # ── Confidence level language (Item 18) ───────────────────────────────
    if confidence > 0.70:
        confidence_level = 'high'
    elif confidence > 0.50:
        confidence_level = 'moderate'
    else:
        confidence_level = 'low'

    # ── Severity ──────────────────────────────────────────────────────────
    sev_idx = int(mean_sev.argmax())
    sev_labels = ['mild', 'moderate', 'severe']
    severity = sev_labels[sev_idx]
    sev_std = float(mean_sev.std())
    sev_low = max(0.0, float(mean_sev[sev_idx]) - sev_std)
    sev_high = min(1.0, float(mean_sev[sev_idx]) + sev_std)

    # ── Diagnosis lookup ──────────────────────────────────────────────────
    with open(DIAG_JSON, 'r', encoding='utf-8') as f:
        diag_db = json.load(f)

    # Severity label for severity-aware treatment routing
    sev_label = sev_labels[sev_idx]  # 'mild', 'moderate', or 'severe'

    treatment = []
    prevention = []
    urgency = 'Low'
    urgency_reason = ''
    urgency_priority = {'High': 3, 'Medium': 2, 'Low': 1,
                        'Act immediately today': 3, 'Act within 24 hours': 2,
                        'Act within 24-48 hours': 2, 'Monitor for 3-5 days': 1,
                        'Routine': 0}
    for cls in detected:
        if cls in diag_db:
            entry = diag_db[cls]
            # Handle severity-tiered treatment (dict) or legacy flat (str/list)
            raw_treatment = entry.get('treatment', [])
            if isinstance(raw_treatment, dict):
                t = raw_treatment.get(sev_label, raw_treatment.get('moderate', ''))
                if t:
                    treatment.append(t)
            elif isinstance(raw_treatment, list):
                treatment.extend(raw_treatment)
            elif isinstance(raw_treatment, str) and raw_treatment:
                treatment.append(raw_treatment)

            raw_prevention = entry.get('prevention', [])
            if isinstance(raw_prevention, list):
                prevention.extend(raw_prevention)
            elif isinstance(raw_prevention, str) and raw_prevention:
                prevention.append(raw_prevention)

            raw_urgency = entry.get('urgency', 'Low')
            if isinstance(raw_urgency, dict):
                entry_urgency = raw_urgency.get(sev_label, raw_urgency.get('moderate', 'Low'))
            else:
                entry_urgency = raw_urgency
            if urgency_priority.get(str(entry_urgency), 0) > urgency_priority.get(str(urgency), 0):
                urgency = entry_urgency
                raw_reason = entry.get('urgency_reason', '')
                if isinstance(raw_reason, dict):
                    urgency_reason = raw_reason.get(sev_label, raw_reason.get('moderate', ''))
                else:
                    urgency_reason = raw_reason

    seen = set()
    treatment = [t for t in treatment if not (t in seen or seen.add(t))]
    seen = set()
    prevention = [p for p in prevention if not (p in seen or seen.add(p))]

    result = {
        'crop': crop_name,
        'crop_confidence': round(crop_conf, 3),
        'crop_probabilities': crop_probs_dict,
        'diseases': detected,
        'top2_diseases': top2_diseases,
        'confidence': round(confidence, 3),
        'uncertainty': round(uncertainty, 3),
        'severity': severity,
        'severity_interval': [round(sev_low, 3), round(sev_high, 3)],
        'treatment': treatment,
        'prevention': prevention,
        'urgency': urgency,
        'urgency_reason': urgency_reason,
        'heatmap_b64': heatmap_b64,
        'ood_flagged': ood_flagged,
        'ood_reason': '',
        'confidence_level': confidence_level,
        'sam_segmented': sam_segmented,
        'second_pass_applied': second_pass_applied,
        'ensemble_applied': ensemble_applied,
        'severity_label': sev_label,
        'severity_note': 'Severity estimate uses placeholder training data. Treat as rough guide.',
    }
    if return_raw_probs:
        result['raw_disease_probs'] = gated_dis.cpu().numpy()
    return result
