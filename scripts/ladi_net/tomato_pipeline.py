"""
Task B.2 — TomatoPipeline module (NEW file; nothing imports it yet).

Self-contained ensemble inference for tomato leaf disease detection. Loads v3
(10-class, DINOv2-Small + LoRA + FiLM) and single-pass LoRA epoch 13 (6-class,
DINOv2-Reg-Base + LoRA on CLS), combines their predictions via 50/50 probability
averaging, and returns a response matching the APIN okra/brassica schema (with
None/empty for APIN-specific fields that tomato cannot natively produce) plus a
tomato-specific `tomato_details` extension.

CRITICAL DOCUMENTATION (PDA condition B1):
    The "50/50" ensemble label describes probability-space averaging of two
    DIFFERENTLY-SHARPENED distributions:
      - v3 at T=0.5 (logits divided by 0.5 = sharpened by 2x)
      - single-pass LoRA at T=1.0 (no sharpening — calibration was unreliable)

    In practice, this means v3 dominates the argmax on high-confidence cases
    (v3 output ~0.97+ overrides LoRA's ~0.80). On uncertain cases the two
    models contribute comparably. This asymmetry is intentional: the +0.024
    field_val lift over v3_alone (per Task A.4) was measured WITH this exact
    asymmetric sharpening setup. Do not change one without re-measuring.

    See Decision 56 (ladi_decisions.md) for full reasoning.

PDA CONDITION B3: T is applied ONCE per model. The tier_thresholds files
    (phase3_tier_thresholds_sp_lora_ep13.json, phase3_tier_thresholds_v3_tomato.json)
    were built from already-T-calibrated probabilities. Phase B MUST NOT
    double-apply T.

PDA CONDITION B4: YLCV class (n=2 on held-out, F1=0.0) adds a
    `calibration_warning` to tomato_details when predicted. Frontend should
    downgrade the displayed confidence tier for YLCV predictions.

PDA CONDITION B5: v3's 0.853 held-out baseline is second-hand (prior session
    documentation, not re-verified in this integration session). Any reference
    to "expected held-out ~0.85" must carry this qualifier. See
    `expected_heldout_note` in response.

ARCHITECTURE CONSTRAINTS (inherited from Decision 56):
    - v3 preprocessing: 224x224 stretch-resize + LAB-CLAHE(L) + ImageNet norm
    - LoRA preprocessing: 800px cap -> letterbox 392 pad=114 + LAB-CLAHE(L) + norm
    - v3 forward: model(x, crop_mode=torch.tensor([2]), domain_labels=None)
    - LoRA forward: model(x) -> dict with 'logits' + 'cls'
    - v3 output remap: V3_INDEX_FOR_LORA_CLASS = [0, 2, 1, 3, 4, 5]
      (late_blight and septoria swap positions)
    - Prototype blending: DISABLED by default (Option ζ empirically validated)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("tomato.pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# These sys.path insertions are defensive; Phase B.3 imports from app module scope.
import sys
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts" / "ladi_net")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ladinet_config import TOMATO_CLASSES  # noqa: E402
from single_pass_lora_train import SinglePassLoRA  # noqa: E402

from scripts.model3_training.architecture.model3_full import Model3  # noqa: E402
from scripts.model3_training.model3_config import (  # noqa: E402
    CLASS_NAMES as V3_CLASSES_10, NUM_CLASSES as V3_N,
    IMAGENET_MEAN as V3_MEAN, IMAGENET_STD as V3_STD,
    CHECKPOINT_DIR as V3_CKPT_DIR, PRODUCTION_V3_CHECKPOINT_NAME as V3_FNAME,
    LORA_RANK as V3_LORA_RANK,
)
from scripts.model3_training.data.preprocessing import apply_lab_clahe as v3_apply_clahe  # noqa: E402


# ---------------------------------------------------------------------------
# Artifact paths (all frozen by Phase A)
# ---------------------------------------------------------------------------
V3_CKPT = V3_CKPT_DIR / V3_FNAME
SP_LORA_CKPT = (PROJECT_ROOT / "models" / "specialist" / "sp_lora_checkpoints"
                / "sp_lora_epoch13_f10.9113_PRESERVED.pt")
PHASE_A_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
SP_BANK = PHASE_A_DIR / "prototype_bank_sp_lora_ep13.pt"
SP_CAL = PHASE_A_DIR / "phase3_calibration_sp_lora_ep13.json"
SP_TIERS = PHASE_A_DIR / "phase3_tier_thresholds_sp_lora_ep13.json"
V3_CAL = PHASE_A_DIR / "phase3_calibration_v3_tomato.json"
V3_TIERS = PHASE_A_DIR / "phase3_tier_thresholds_v3_tomato.json"
DIAGNOSIS_LOOKUP = PROJECT_ROOT / "diagnosis" / "diagnosis_lookup.json"

# ---------------------------------------------------------------------------
# Constants (Decision 56-locked)
# ---------------------------------------------------------------------------
V3_INPUT_RES = 224
LORA_INPUT_RES = 392
LORA_LETTERBOX_PAD = 114

# Ensemble weight (Decision 56 locked; not tunable at runtime)
W_V3 = 0.50
W_LORA = 0.50

# v3 10-class -> LoRA canonical 6-class tomato index remap
V3_INDEX_FOR_LORA_CLASS = [0, 2, 1, 3, 4, 5]

# Tier thresholds (deployed)
TIER_1A_MAX_PROB = 0.60    # from Phase A sweep (chosen_threshold)
TIER_1A_GAP_MIN = 0.25     # from spec Step 10
TIER_1B_GAP_MIN = 0.20     # tier 1B gap floor
TIER_2A_PROB_FLOOR = 0.40  # below this -> tier 4 (abstain)
TIER_ABSTAIN_PROB_FLOOR = 0.32  # below this -> tier 4 (abstain)

# PDA B4: YLCV underpowered on held-out (n=2, F1=0.0). Hedge confidence in UI.
YLCV_CLASS = "tomato_yellow_leaf_curl_virus"
YLCV_HEDGE_MESSAGE = ("Limited validation data for this class (only 2 held-out "
                       "samples). Treat as a preliminary indication; verify with "
                       "a second photograph or agricultural extension officer.")

# Prototype blending (Option ζ: DISABLED by default)
ENABLE_PROTOTYPE_BLENDING = False


# ---------------------------------------------------------------------------
# Visualization constants for GradCAM
# ---------------------------------------------------------------------------
GRADCAM_OUT_SIZE = 384
HEATMAP_ALPHA = 0.45


# ---------------------------------------------------------------------------
# Disease info (from diagnosis_lookup.json, flattened to match APIN M2 fields)
# ---------------------------------------------------------------------------
SEVERITY_TO_COLOR = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"}
# Phase C-Redesign Stage 1.1: emoji removed. The frontend renders icons from
# the SVG symbol library instead. Keep the dict with the SAME keys so any
# info.get("emoji", "") call still works, but return empty strings so nothing
# renders on client strings that interpolate this value. The class icon
# category is still carried in info["icon_category"] below for frontend use.
CLASS_EMOJI = {
    "tomato_foliar_spot": "",
    "tomato_septoria_leaf_spot": "",
    "tomato_late_blight": "",
    "tomato_yellow_leaf_curl_virus": "",
    "tomato_mosaic_virus": "",
    "tomato_healthy": "",
}
# Semantic category per class; frontend maps these to stroke icons.
CLASS_ICON_CATEGORY = {
    "tomato_foliar_spot": "fungal",
    "tomato_septoria_leaf_spot": "fungal",
    "tomato_late_blight": "critical",
    "tomato_yellow_leaf_curl_virus": "viral",
    "tomato_mosaic_virus": "viral",
    "tomato_healthy": "healthy",
}

# Stage 1.4 region labels used by _analyse_cam_region. Order matches the
# 3x3 scan order (top row first, then mid, then bot). Farmer-readable
# phrasing is handled on the frontend; the server emits the raw key.
_REGION_LABELS = (
    "top_left",    "top_centre",    "top_right",
    "mid_left",    "mid_centre",    "mid_right",
    "bot_left",    "bot_centre",    "bot_right",
)

# Phase E.1: per-class plain-English symptom descriptions for the
# constellation hover-card mini field guide. Two short sentences each,
# emoji-free, em-dash-free, written for a farmer with zero pathology
# background. Used by the constellation hover popover. Shipped under
# tomato_details.class_symptom_descriptions.
TOMATO_CLASS_SYMPTOMS = {
    "tomato_foliar_spot": (
        "Small to medium dark spots scattered across the leaf, often "
        "ringed by a yellow halo. Bacterial spot, early blight and "
        "target spot all produce visually similar marks."
    ),
    "tomato_septoria_leaf_spot": (
        "Many small grey-tan circular spots with a clear dark border. "
        "A tiny black dot at the centre is often visible on close look."
    ),
    "tomato_late_blight": (
        "Large, irregular dark blotches with watery edges. Affected "
        "leaves wilt and droop quickly, and white mould can appear "
        "underneath in damp weather."
    ),
    "tomato_yellow_leaf_curl_virus": (
        "Leaves curl strongly upward at the edges and turn yellow along "
        "the veins. Plants are visibly stunted and produce few fruits."
    ),
    "tomato_mosaic_virus": (
        "Mottled green and yellow patchwork across the leaf surface, "
        "with distorted leaf shape. Veins may also be irregular."
    ),
    "tomato_healthy": (
        "Uniform green colour with smooth margins and no visible spots, "
        "patches, curling, or blotches."
    ),
}

# Phase D SRV-2: training-image directories used for medoid resolution.
# Each tomato class maps to ONE or MORE source directories that contain
# representative training images in full color. The medoid resolver at
# TomatoPipeline init scans up to MEDOID_SCAN_LIMIT images per directory,
# computes the single-pass LoRA CLS token for each, and picks the image
# whose CLS token is closest (by L2 distance) to the mean of the class's
# prototype centroids. That image is the "medoid" representative.
#
# tomato_foliar_spot is the composite v3/LoRA class covering bacterial_spot,
# early_blight, target_spot; we scan all three folders to find the medoid.
_PLANTVILLAGE_ROOT = "data/raw/plantvillage_tomato/plantvillage dataset/color"
TOMATO_CLASS_TRAIN_DIRS = {
    "tomato_foliar_spot": [
        f"{_PLANTVILLAGE_ROOT}/Tomato___Bacterial_spot",
        f"{_PLANTVILLAGE_ROOT}/Tomato___Early_blight",
        f"{_PLANTVILLAGE_ROOT}/Tomato___Target_Spot",
    ],
    "tomato_septoria_leaf_spot": [
        f"{_PLANTVILLAGE_ROOT}/Tomato___Septoria_leaf_spot",
    ],
    "tomato_late_blight": [
        f"{_PLANTVILLAGE_ROOT}/Tomato___Late_blight",
    ],
    "tomato_yellow_leaf_curl_virus": [
        f"{_PLANTVILLAGE_ROOT}/Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ],
    "tomato_mosaic_virus": [
        f"{_PLANTVILLAGE_ROOT}/Tomato___Tomato_mosaic_virus",
    ],
    "tomato_healthy": [
        f"{_PLANTVILLAGE_ROOT}/Tomato___healthy",
    ],
}
MEDOID_SCAN_LIMIT = 30  # images sampled per class for medoid search
MEDOID_THUMB_PX = 128   # output thumbnail size
PIPELINE_THUMB_PX = 128  # per-station thumbnail size


def _build_tier_path(max_prob: float, gap: float, tier_1a_thr: float,
                      tier_1a_gap_min: float, tier_1b_gap_min: float,
                      tier_2a_floor: float, tier_abstain_floor: float,
                      agreement: bool, strength: str) -> list:
    """Walk the exact _assign_tier decision logic and return the branch taken
    as a list of step dicts. Mirrors the thresholds passed in so the path
    always matches the actual tier the assignment function returned.

    Stays structural and language agnostic; the UI prose renderer converts
    this to a human-readable tree. Plain function (not a method) so it can
    be unit-tested without TomatoPipeline state.

    Each step dict: {check, passed, detail}. The sequence corresponds to
    how _assign_tier evaluates top to bottom.
    """
    steps = []
    # Step A agreement check. Informational: current _assign_tier does not
    # consume agreement, but the UI needs to surface it as part of the story.
    steps.append({
        "check": "agreement",
        "passed": bool(agreement),
        "detail": ("v3 and sp_lora predict the same top class"
                   if agreement
                   else "v3 and sp_lora disagree on the top class"),
        "strength": strength,
    })
    # Step B 1A prob threshold + gap
    if max_prob >= tier_1a_thr and gap >= tier_1a_gap_min:
        steps.append({"check": "tier_1a_prob_and_gap",
                       "passed": True,
                       "detail": (f"max probability {max_prob:.3f} above 1A "
                                  f"threshold {tier_1a_thr:.2f} and gap "
                                  f"{gap:.3f} meets 1A gap minimum "
                                  f"{tier_1a_gap_min:.2f}")})
        return steps
    # Step C 1B prob threshold + smaller gap
    if max_prob >= tier_1a_thr and gap >= tier_1b_gap_min:
        steps.append({"check": "tier_1a_prob_and_gap",
                       "passed": False,
                       "detail": (f"max probability {max_prob:.3f} above 1A "
                                  f"threshold {tier_1a_thr:.2f} but gap "
                                  f"{gap:.3f} below 1A minimum "
                                  f"{tier_1a_gap_min:.2f}")})
        steps.append({"check": "tier_1b_gap",
                       "passed": True,
                       "detail": (f"gap {gap:.3f} meets 1B gap minimum "
                                  f"{tier_1b_gap_min:.2f}")})
        return steps
    # Step D 2A floor (reached when neither 1A nor 1B passed).
    # PDA Issue 5 fix: always emit an explanatory preceding step so the UI
    # tree does not silently jump from "agreement" straight to "2A floor".
    if max_prob >= tier_2a_floor:
        if max_prob >= tier_1a_thr:
            # max_prob was high enough for 1A/1B, but the gap was too small
            steps.append({"check": "tier_1b_gap",
                           "passed": False,
                           "detail": (f"gap {gap:.3f} below 1B gap minimum "
                                      f"{tier_1b_gap_min:.2f}")})
        else:
            # max_prob was below 1A threshold, so 1A/1B were never eligible
            steps.append({"check": "tier_1a_prob",
                           "passed": False,
                           "detail": (f"max probability {max_prob:.3f} below 1A "
                                      f"threshold {tier_1a_thr:.2f}")})
        steps.append({"check": "tier_2a_floor",
                       "passed": True,
                       "detail": (f"max probability {max_prob:.3f} at or above "
                                  f"differential floor {tier_2a_floor:.2f}")})
        return steps
    # Falling through to Step E means max_prob < tier_2a_floor, which also
    # implies max_prob < tier_1a_thr (since tier_1a_thr >= tier_2a_floor in
    # practice). Emit the 1A-prob fail first so the tree shows why the higher
    # tiers were skipped, then the 2A floor fail.
    steps.append({"check": "tier_1a_prob",
                   "passed": False,
                   "detail": (f"max probability {max_prob:.3f} below 1A "
                              f"threshold {tier_1a_thr:.2f}")})
    steps.append({"check": "tier_2a_floor",
                   "passed": False,
                   "detail": (f"max probability {max_prob:.3f} below "
                              f"differential floor {tier_2a_floor:.2f}")})
    # Step E 2B floor
    if max_prob >= tier_abstain_floor:
        steps.append({"check": "tier_2b_floor",
                       "passed": True,
                       "detail": (f"max probability {max_prob:.3f} at or above "
                                  f"2B floor {tier_abstain_floor:.2f}")})
        return steps
    # Step F below abstain floor
    steps.append({"check": "tier_2b_floor",
                   "passed": False,
                   "detail": (f"max probability {max_prob:.3f} below abstain "
                              f"floor {tier_abstain_floor:.2f}; tier set to 4A")})
    return steps


def _compose_foliar_spot_entry(lookup: dict) -> dict:
    """Composite entry for 'tomato_foliar_spot' which is NOT in diagnosis_lookup
    (it's the v3/LoRA consolidation of bacterial_spot + early_blight + target_spot).
    """
    components = ["tomato_bacterial_spot", "tomato_early_blight", "tomato_target_spot"]
    parts = [lookup[c] for c in components if c in lookup]
    if not parts:
        return {}
    treatments_mod = [p.get("treatment", {}).get("moderate", "") for p in parts]
    return {
        "full_name": "Foliar spot complex",
        "symptoms": ("Dark leaf spots with yellow halos. Visually consistent with "
                     "bacterial spot, early blight, or target spot; precise "
                     "identification requires lab confirmation."),
        "treatment_moderate": ("Combined broad-spectrum approach: "
                               + " + ".join(t.split(".")[0] for t in treatments_mod if t)),
        "urgency_moderate": "Medium",
        "note": "Composite entry: v3/LoRA consolidate 3 foliar-spot-like diseases.",
    }


def load_disease_info() -> dict:
    """Load 6 tomato entries from diagnosis_lookup.json, flattened to APIN-M2
    field shape.
    """
    if not DIAGNOSIS_LOOKUP.exists():
        logger.warning("diagnosis_lookup.json missing; DISEASE_INFO will be empty.")
        return {}
    with open(DIAGNOSIS_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)

    info = {}
    flat_mapping = {
        "tomato_septoria_leaf_spot": "tomato_septoria_leaf_spot",
        "tomato_late_blight": "tomato_late_blight",
        "tomato_yellow_leaf_curl_virus": "tomato_yellow_leaf_curl_virus",
        "tomato_mosaic_virus": "tomato_mosaic_virus",
        "tomato_healthy": "tomato_healthy",
    }
    for cls, lookup_key in flat_mapping.items():
        if lookup_key not in lookup:
            continue
        e = lookup[lookup_key]
        treatment = e.get("treatment", {})
        urgency = e.get("urgency", {})
        urgency_reason = e.get("urgency_reason", {})
        # Pick the 'moderate' severity as the default displayed recommendation
        action = (treatment.get("moderate") if isinstance(treatment, dict) else str(treatment))
        severity = (urgency.get("moderate") if isinstance(urgency, dict) else str(urgency))
        info[cls] = {
            "name": e.get("full_name", cls.replace("_", " ").title()),
            "desc": e.get("symptoms", "")[:200],
            "action": action or "",
            "severity": severity or "Medium",
            "color": SEVERITY_TO_COLOR.get(severity, "#888"),
            "emoji": CLASS_EMOJI.get(cls, "🍃"),
            "urgency_reason": (urgency_reason.get("moderate")
                               if isinstance(urgency_reason, dict) else ""),
        }

    # Composite entry for foliar_spot
    composite = _compose_foliar_spot_entry(lookup)
    if composite:
        info["tomato_foliar_spot"] = {
            "name": composite["full_name"],
            "desc": composite["symptoms"][:200],
            "action": composite["treatment_moderate"],
            "severity": composite["urgency_moderate"],
            "color": SEVERITY_TO_COLOR.get(composite["urgency_moderate"], "#888"),
            "emoji": CLASS_EMOJI.get("tomato_foliar_spot", "🍃"),
            "urgency_reason": composite["note"],
        }

    missing = [c for c in TOMATO_CLASSES if c not in info]
    if missing:
        logger.warning(f"DISEASE_INFO missing entries for: {missing}")
    return info


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------
def _png_base64(img_rgb: np.ndarray, max_side: int = 256) -> str:
    """Phase D SRV helper. Encode an RGB uint8 array as base64 PNG.

    Optionally downscales so the longer side is at most `max_side` pixels
    to keep response payload bounded. Returns empty string on failure.
    """
    try:
        arr = img_rgb
        if arr is None or not isinstance(arr, np.ndarray):
            return ""
        h, w = arr.shape[:2]
        longer = max(h, w)
        if longer > max_side:
            scale = max_side / float(longer)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        # cv2.imencode wants BGR; convert from RGB
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:
        logger.debug(f"_png_base64 failed: {e}")
        return ""


def _preview_v3_input(img_rgb: np.ndarray) -> np.ndarray:
    """Return the RGB uint8 version of the v3 preprocessing pipeline
    WITHOUT the ImageNet normalization. Matches the visual state the model
    actually processes, for display in the Ensemble Flow detail panel.
    """
    import cv2 as _cv2
    bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
    # LAB-CLAHE on L channel (Decision 56 preprocessing for v3)
    lab = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2LAB)
    L, a, b = _cv2.split(lab)
    clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = _cv2.merge([L2, a, b])
    bgr2 = _cv2.cvtColor(lab2, _cv2.COLOR_LAB2BGR)
    # Stretch resize to 224x224 (v3 spec)
    resized = _cv2.resize(bgr2, (224, 224), interpolation=_cv2.INTER_AREA)
    return _cv2.cvtColor(resized, _cv2.COLOR_BGR2RGB)


def _preview_v3_step_resized(img_rgb: np.ndarray) -> np.ndarray:
    """Phase E.1: stretch-resize to 224x224 WITHOUT CLAHE. Lets the UI
    show the resize step distinct from the CLAHE step."""
    import cv2 as _cv2
    bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
    resized = _cv2.resize(bgr, (224, 224), interpolation=_cv2.INTER_AREA)
    return _cv2.cvtColor(resized, _cv2.COLOR_BGR2RGB)


def _preview_router_input(img_rgb: np.ndarray) -> np.ndarray:
    """Phase E.1: router preprocessing (224x224 resize). Same shape as v3
    pre-CLAHE; the router uses a fixed 224 input on a frozen DINOv2-Small
    head. We reuse the v3 step_resized image since the resize is identical."""
    return _preview_v3_step_resized(img_rgb)


def _preview_sp_lora_step_capped(img_rgb: np.ndarray) -> np.ndarray:
    """Phase E.1: 800 px cap step from sp_lora preprocessing, no letterbox
    or CLAHE applied yet. Keeps original aspect ratio."""
    import cv2 as _cv2
    bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    longer = max(h, w)
    if longer > 800:
        s = 800.0 / longer
        bgr = _cv2.resize(bgr, (int(round(w * s)), int(round(h * s))),
                           interpolation=_cv2.INTER_AREA)
    return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)


def _preview_sp_lora_step_letterbox(img_rgb: np.ndarray) -> np.ndarray:
    """Phase E.1: 392x392 letterbox step from sp_lora preprocessing,
    BEFORE CLAHE. Shows the grey-padded canvas the model receives."""
    import cv2 as _cv2
    bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    longer = max(h, w)
    if longer > 800:
        s = 800.0 / longer
        bgr = _cv2.resize(bgr, (int(round(w * s)), int(round(h * s))),
                           interpolation=_cv2.INTER_AREA)
    target = 392
    pad_color = (114, 114, 114)
    h, w = bgr.shape[:2]
    s = target / float(max(h, w))
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = _cv2.resize(bgr, (nw, nh), interpolation=_cv2.INTER_AREA)
    canvas = np.full((target, target, 3), pad_color, dtype=np.uint8)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return _cv2.cvtColor(canvas, _cv2.COLOR_BGR2RGB)


def _preview_sp_lora_input(img_rgb: np.ndarray) -> np.ndarray:
    """Return the RGB uint8 version of the sp_lora preprocessing pipeline
    WITHOUT the ImageNet normalization. 800 px cap, 392 px letterbox, LAB-
    CLAHE on the L channel.
    """
    import cv2 as _cv2
    bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # 800 px cap
    longer = max(h, w)
    if longer > 800:
        s = 800.0 / longer
        bgr = _cv2.resize(bgr, (int(round(w * s)), int(round(h * s))),
                           interpolation=_cv2.INTER_AREA)
    # Letterbox to 392x392 with pad=114 (matches sp_lora train pipeline)
    target = 392
    pad_color = (114, 114, 114)
    h, w = bgr.shape[:2]
    s = target / float(max(h, w))
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = _cv2.resize(bgr, (nw, nh), interpolation=_cv2.INTER_AREA)
    canvas = np.full((target, target, 3), pad_color, dtype=np.uint8)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    # LAB-CLAHE on L channel
    lab = _cv2.cvtColor(canvas, _cv2.COLOR_BGR2LAB)
    L, a, b = _cv2.split(lab)
    clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = _cv2.merge([L2, a, b])
    bgr2 = _cv2.cvtColor(lab2, _cv2.COLOR_LAB2BGR)
    return _cv2.cvtColor(bgr2, _cv2.COLOR_BGR2RGB)


def _ensure_rgb_uint8(img) -> np.ndarray:
    """Take whatever APIN parse_image returns and coerce to valid RGB uint8 3-channel.
    Handles: grayscale, RGBA, RGB, edge cases.
    """
    if img is None:
        raise ValueError("img is None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img).__name__}")
    if img.ndim == 2:  # grayscale
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _preprocess_v3(img_rgb: np.ndarray) -> torch.Tensor:
    """v3 preprocessing: RGB in, 224x224 stretch, LAB-CLAHE(L), ImageNet norm."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = v3_apply_clahe(img_bgr)
    img_bgr = cv2.resize(img_bgr, (V3_INPUT_RES, V3_INPUT_RES), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(V3_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(V3_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def _letterbox_392(img_bgr: np.ndarray, pad_value: int = LORA_LETTERBOX_PAD) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if (h, w) == (LORA_INPUT_RES, LORA_INPUT_RES):
        return img_bgr
    scale = LORA_INPUT_RES / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    top = (LORA_INPUT_RES - new_h) // 2
    bottom = LORA_INPUT_RES - new_h - top
    left = (LORA_INPUT_RES - new_w) // 2
    right = LORA_INPUT_RES - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(pad_value,) * 3)


def _preprocess_sp_lora(img_rgb: np.ndarray) -> torch.Tensor:
    """LoRA preprocessing: RGB in, 800px cap, letterbox 392, LAB-CLAHE(L), ImageNet norm."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    if max(h, w) > 800:  # smartphone-native cap (Decision 50)
        scale = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    img_bgr = _letterbox_392(img_bgr)
    # LAB-CLAHE on L channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    from ladinet_config import IMAGENET_MEAN, IMAGENET_STD
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# GradCAM (lean; single-pass LoRA CLS target layer)
# ---------------------------------------------------------------------------
def _get_vit_last_block(backbone: nn.Module) -> nn.Module:
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        return backbone.blocks[-1]
    if (hasattr(backbone, "base_model")
            and hasattr(backbone.base_model, "model")
            and hasattr(backbone.base_model.model, "blocks")):
        return backbone.base_model.model.blocks[-1]
    raise AttributeError("Cannot locate ViT blocks on backbone")


def _gradcam_sp_lora_at_block(model, x: torch.Tensor, target_idx: int,
                              block_idx: int) -> tuple:
    """Grad-CAM at a specific ViT block index. Returns (cam_2d, max_value).
    block_idx is -1 for the last block, -3 for three-from-last, etc.
    Returns max_value so callers can detect a saturated (~all-zero) CAM and
    decide to retry with a different block.
    """
    blocks = None
    if hasattr(model.backbone, "blocks") and len(model.backbone.blocks) > 0:
        blocks = model.backbone.blocks
    elif (hasattr(model.backbone, "base_model")
          and hasattr(model.backbone.base_model, "model")
          and hasattr(model.backbone.base_model.model, "blocks")):
        blocks = model.backbone.base_model.model.blocks
    if blocks is None:
        raise AttributeError("Cannot locate ViT blocks on backbone")
    target_layer = blocks[block_idx]
    activations = {}
    gradients = {}
    def fwd_hook(_m, _i, out): activations["x"] = out
    def bwd_hook(_m, _gi, go): gradients["x"] = go[0]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    model.zero_grad(set_to_none=True)
    x_in = x.clone().detach().requires_grad_(True)
    try:
        with torch.enable_grad():
            out = model(x_in)
            logits = out["logits"]
            logits[0, target_idx].backward()
    finally:
        h1.remove(); h2.remove()
    act = activations["x"]
    grad = gradients["x"]
    n_prefix = 5  # 1 CLS + 4 registers
    act_p = act[:, n_prefix:, :]
    grad_p = grad[:, n_prefix:, :]
    weights = grad_p.mean(dim=1, keepdim=True)
    cam = (weights * act_p).sum(dim=-1).squeeze(0)
    cam = torch.relu(cam)
    m = cam.max()
    raw_max = float(m.item())
    if m > 0:
        cam = cam / m
    cam_2d = cam.reshape(28, 28).detach().cpu().float().numpy()
    return cam_2d, raw_max


def _gradcam_sp_lora(model, x: torch.Tensor, target_idx: int,
                     device: torch.device) -> tuple:
    """Multi-block Grad-CAM with saturation fallback.

    Phase E.5b: When the last block's Grad-CAM saturates (max ≤ epsilon,
    meaning gradients vanished on a highly-confident prediction), retry at
    earlier blocks where attention is still spatially varied. Return the
    first block whose CAM has actual signal, plus a label identifying which
    block was used so the UI can be honest about provenance.

    Returns (cam_2d, source_label) where source_label is e.g. "sp_lora_cls"
    (last block, the original behavior) or "sp_lora_block-3" (fallback).
    """
    SATURATION_EPS = 1e-6
    # Try last block first; if saturated, walk back through earlier blocks.
    candidate_blocks = [-1, -3, -6, -9]
    last_cam = None
    last_label = "sp_lora_cls"
    for bi in candidate_blocks:
        try:
            cam_2d, raw_max = _gradcam_sp_lora_at_block(model, x, target_idx, bi)
        except (IndexError, AttributeError):
            continue
        # Save the first one in case all saturate, so we never return None.
        if last_cam is None:
            last_cam = cam_2d
            last_label = "sp_lora_cls" if bi == -1 else f"sp_lora_block{bi}"
        # Accept this block if it has real signal AND spatial variance.
        if raw_max > SATURATION_EPS and float(cam_2d.std()) > 1e-3:
            label = "sp_lora_cls" if bi == -1 else f"sp_lora_block{bi}"
            return cam_2d, label
    return last_cam, last_label


def _overlay_heatmap(img_rgb: np.ndarray, cam_2d: np.ndarray) -> str:
    """Base64 PNG of the cam overlaid on the (letterboxed) image."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    scale = GRADCAM_OUT_SIZE / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (GRADCAM_OUT_SIZE - nh) // 2
    bot = GRADCAM_OUT_SIZE - nh - top
    left = (GRADCAM_OUT_SIZE - nw) // 2
    right = GRADCAM_OUT_SIZE - nw - left
    base = cv2.copyMakeBorder(resized, top, bot, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cam_u = np.clip(cam_2d * 255.0, 0, 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_u, (GRADCAM_OUT_SIZE, GRADCAM_OUT_SIZE),
                             interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 1.0 - HEATMAP_ALPHA, heat, HEATMAP_ALPHA, 0)
    ok, png = cv2.imencode(".png", overlay)
    if not ok:
        return ""
    return base64.b64encode(png.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# TomatoPipeline
# ---------------------------------------------------------------------------
class TomatoPipeline:
    """v3 + single-pass LoRA tomato ensemble.

    Thread-safe inference (models are frozen after init; forward calls are stateless).

    Usage:
        pipeline = TomatoPipeline(device=torch.device('cuda'))
        result_dict = pipeline.infer(rgb_uint8_numpy_array)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()  # serializes Grad-CAM (backward passes)

        t0 = time.time()
        logger.info(f"Loading TomatoPipeline on {self.device}")
        self._load_v3()
        self._load_sp_lora()
        self._load_calibrations()
        self._load_tier_thresholds()
        self._load_prototype_bank()
        self._load_disease_info()
        # Phase D SRV-2: resolve a medoid training image per tomato class,
        # once at startup. Needs sp_lora (for CLS extraction) and the
        # prototype bank (for the class centroid). Graceful fallback on
        # missing folders: self._class_ref_thumbnails becomes {} and the
        # response emits class_reference_thumbnails: null.
        self._compute_class_medoids()
        self._validate()
        logger.info(f"TomatoPipeline loaded in {time.time() - t0:.1f}s")

    # ----- loaders ---------------------------------------------------------
    def _load_v3(self):
        if not V3_CKPT.exists():
            raise FileNotFoundError(f"v3 checkpoint not found: {V3_CKPT}")
        self.v3 = Model3(n_classes=V3_N, pretrained=False, lora_rank=V3_LORA_RANK).to(self.device)
        ckpt = torch.load(V3_CKPT, map_location=self.device, weights_only=False)
        self.v3.load_state_dict(ckpt["model_state_dict"])
        self.v3.eval()
        for p in self.v3.parameters():
            p.requires_grad = False
        self.v3_meta = {
            "run_name": ckpt.get("run_name"),
            "soup_field_f1": ckpt.get("soup_selection_field_f1"),
        }
        logger.info(f"  v3 loaded ({V3_CKPT.name})")

    def _load_sp_lora(self):
        if not SP_LORA_CKPT.exists():
            raise FileNotFoundError(f"sp_lora checkpoint not found: {SP_LORA_CKPT}")
        self.sp_lora = SinglePassLoRA(self.device).to(self.device)
        ckpt = torch.load(SP_LORA_CKPT, map_location=self.device, weights_only=False)
        self.sp_lora.load_state_dict(ckpt["model_state_dict"])
        self.sp_lora.eval()
        for p in self.sp_lora.parameters():
            p.requires_grad = False
        self.sp_lora_meta = {
            "epoch": ckpt.get("epoch"),
            "val_sqrtn_macro_f1": ckpt.get("val_sqrtn_macro_f1"),
        }
        logger.info(f"  sp_lora loaded ({SP_LORA_CKPT.name})")

    def _load_calibrations(self):
        with open(SP_CAL, encoding="utf-8") as f:
            self.sp_cal = json.load(f)
        with open(V3_CAL, encoding="utf-8") as f:
            self.v3_cal = json.load(f)
        # PDA B3: apply T ONCE per model, at inference (tier thresholds already T-applied).
        self.T_sp = (float(self.sp_cal["T_optimal"])
                     if self.sp_cal.get("use_calibration", True) else 1.0)
        self.T_v3 = (float(self.v3_cal["T_optimal"])
                     if self.v3_cal.get("use_calibration", True) else 1.0)
        logger.info(f"  T_sp_lora={self.T_sp:.4f} (use_cal={self.sp_cal.get('use_calibration')}), "
                    f"T_v3={self.T_v3:.4f} (use_cal={self.v3_cal.get('use_calibration')})")

    def _load_tier_thresholds(self):
        with open(SP_TIERS, encoding="utf-8") as f:
            self.sp_tier_doc = json.load(f)
        with open(V3_TIERS, encoding="utf-8") as f:
            self.v3_tier_doc = json.load(f)
        # Use sp_lora's tier threshold for the ensemble (LoRA is the 6-class native model)
        self.tier_1a_threshold = float(self.sp_tier_doc.get("chosen_threshold") or TIER_1A_MAX_PROB)
        logger.info(f"  Tier 1A threshold = {self.tier_1a_threshold}")

    def _load_prototype_bank(self):
        # Loaded even though blending is disabled (used for diagnostic output,
        # nearest-prototype info in tomato_details, and the Stage 1.3 prototype
        # constellation visualization).
        if SP_BANK.exists():
            self.proto_bank = torch.load(SP_BANK, map_location="cpu", weights_only=False)
            assert self.proto_bank.get("feature_space") == "CLS_token_768d", \
                f"Expected CLS_token_768d prototype bank, got {self.proto_bank.get('feature_space')}"
            logger.info(f"  Prototype bank loaded (CLS_token_768d, "
                        f"{sum(p.shape[0] for p in self.proto_bank['prototypes'].values())} prototypes)")
            # Stage 1.3: fit a deterministic PCA projection over all 30 centroids
            # once at startup. At inference time the current image's CLS token
            # is projected through this same matrix, so coordinates are
            # commensurable across requests. PCA is diagnostic only (never used
            # by any decision logic). The 2D projection is stored in
            # self._proto_pca = {"mean": [768], "components": [2, 768],
            #                    "centroids_2d": [30, 2], "class_labels": [30],
            #                    "explained_var": [2]}.
            self._fit_proto_pca()
        else:
            self.proto_bank = None
            self._proto_pca = None
            logger.warning(f"  Prototype bank missing ({SP_BANK.name}); diagnostic features disabled")

    def _fit_proto_pca(self):
        """Fit a 2D PCA over all prototype centroids for the constellation viz.

        Reproducible across server restarts (PDA Issue 1 fix): torch.pca_lowrank
        uses randomized SVD internally whose component sign and column ordering
        depend on the global torch random state. Without a fixed seed the same
        prototype bank would produce mirrored 2D coordinates across restarts,
        making saved screenshots and cached UI state non-comparable. We seed
        locally with the same constant every time; this does not affect any
        other torch state the server uses because we only care about the
        output of this single call.
        """
        try:
            all_protos = []
            all_labels = []
            for cls in TOMATO_CLASSES:
                protos = self.proto_bank["prototypes"][cls]  # [k, 768]
                for row in protos:
                    all_protos.append(row)
                    all_labels.append(cls)
            X = torch.stack(all_protos, dim=0).float()  # [30, 768]
            mean = X.mean(dim=0, keepdim=True)           # [1, 768]
            Xc = X - mean
            # PDA Issue 1: deterministic SVD sign via seeded generator.
            # pca_lowrank's randomness comes from the random initial subspace
            # in its internal subspace iteration. Seeding torch globally
            # would affect other code; we save/restore BOTH CPU and CUDA
            # RNG state so this fit has zero side effects on any consumer
            # of torch randomness (PDA R2 Issue A fix). In practice the
            # prototype tensors are loaded with map_location='cpu' so the
            # PCA runs on CPU and CUDA RNG is never touched, but the
            # save/restore is defensive correctness for any future code that
            # might move these tensors to GPU.
            _cpu_rng = torch.random.get_rng_state()
            _cuda_rngs = None
            if torch.cuda.is_available():
                try:
                    _cuda_rngs = torch.cuda.get_rng_state_all()
                except Exception:
                    _cuda_rngs = None
            try:
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.manual_seed_all(0)
                    except Exception:
                        pass
                U, S, V = torch.pca_lowrank(Xc, q=2, center=False)
            finally:
                torch.random.set_rng_state(_cpu_rng)
                if _cuda_rngs is not None:
                    try:
                        torch.cuda.set_rng_state_all(_cuda_rngs)
                    except Exception:
                        pass
            # Force a deterministic sign per component: each column of V is
            # flipped so that its largest-absolute-value entry is positive.
            for j in range(V.shape[1]):
                col = V[:, j]
                idx = int(torch.argmax(torch.abs(col)).item())
                if float(col[idx].item()) < 0:
                    V[:, j] = -V[:, j]
            components = V.t()                            # [2, 768]
            centroids_2d = (Xc @ V)                       # [30, 2]
            total_var = (Xc * Xc).sum()
            explained_var = (S * S) / max(float(total_var), 1e-12)
            self._proto_pca = {
                "mean": mean.squeeze(0),                  # [768]
                "components": components,                 # [2, 768]
                "centroids_2d": centroids_2d,             # [30, 2]
                "class_labels": all_labels,               # list of 30 str
                "explained_var": [float(explained_var[0]), float(explained_var[1])],
            }
            logger.info(f"  Prototype PCA fitted (explained var: "
                        f"{explained_var[0]:.3f}, {explained_var[1]:.3f})")
        except Exception as e:
            logger.warning(f"  Prototype PCA failed ({e}); constellation viz will be empty")
            self._proto_pca = None

    def _project_cls_token_to_2d(self, cls_feats: torch.Tensor) -> tuple:
        """Project an unnormalized CLS token [1, 768] into the PCA 2D plane.

        Returns (x, y) as Python floats, or (None, None) if PCA unavailable.
        Uses raw (not L2-normalized) features because PCA was fit on raw
        centroids from the prototype bank (centroids are class-means, not
        cosine-normalized).
        """
        if self._proto_pca is None:
            return (None, None)
        try:
            x = cls_feats.squeeze(0).cpu().float()        # [768]
            xc = x - self._proto_pca["mean"]              # [768]
            # components is [2, 768], we want [2]
            coords = self._proto_pca["components"] @ xc   # [2]
            return (float(coords[0]), float(coords[1]))
        except Exception as e:
            logger.warning(f"  CLS PCA projection failed: {e}")
            return (None, None)

    def _compute_class_medoids(self):
        """Phase D SRV-2: find ONE representative training image per tomato
        class via CLS-token medoid search in the single-pass LoRA feature
        space.

        Algorithm per class:
          1) Compute the class centroid as the mean of its 5 prototype
             centroids (all already in CLS_token_768d space).
          2) Scan up to MEDOID_SCAN_LIMIT images from the class's training
             directories.
          3) For each image: run the single-pass LoRA forward pass, extract
             the CLS token, measure L2 distance to the class centroid.
          4) The medoid is the image with minimum distance.
          5) Resize the medoid to MEDOID_THUMB_PX x MEDOID_THUMB_PX and
             base64-encode as PNG. Store on self._class_ref_thumbnails.

        Graceful fallback: if the training directories do not exist, the
        prototype bank is absent, or an exception fires, the method sets
        self._class_ref_thumbnails = {} and returns. The downstream
        response emits class_reference_thumbnails = null.
        """
        self._class_ref_thumbnails = {}
        if self.proto_bank is None:
            logger.info("  [medoid] prototype bank missing, skipping medoid resolution")
            return
        import os as _os
        import random as _random
        for cls in TOMATO_CLASSES:
            try:
                protos = self.proto_bank["prototypes"].get(cls)
                if protos is None or protos.shape[0] == 0:
                    continue
                # PDA D.2 Finding 1 fix: the prototype bank stores raw CLS
                # tokens (confirmed by the 'normalization' field on the bank
                # which states 'raw (not L2-normalized) -- caller must
                # L2-normalize for cosine'). Raw L2 distance on 768-d
                # vectors is dominated by vector magnitude differences
                # (illumination, leaf size) rather than semantic similarity.
                # _nearest_prototypes handles this correctly via F.normalize
                # before cosine; the medoid search must do the same. We
                # compute cosine similarity between normalized vectors,
                # i.e. 1 - cos == minimizes the "semantic distance".
                centroid_raw = protos.float().mean(dim=0).to(self.device)  # [768]
                centroid = F.normalize(centroid_raw, dim=-1)
                dirs = TOMATO_CLASS_TRAIN_DIRS.get(cls, [])
                candidates = []
                for d in dirs:
                    full_dir = PROJECT_ROOT / d if not _os.path.isabs(d) else Path(d)
                    if not full_dir.exists() or not full_dir.is_dir():
                        continue
                    for fname in _os.listdir(str(full_dir)):
                        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            candidates.append(str(full_dir / fname))
                if not candidates:
                    logger.warning(f"  [medoid] no images found for {cls}")
                    continue
                # PDA D.2 Finding 2 fix: os.listdir order is filesystem-
                # dependent (inode order on ext4, creation order on NTFS,
                # not guaranteed stable across OS versions or directory
                # mutations). Sorting before seeded sampling guarantees
                # the same medoid is picked across reinstalls or reboots.
                candidates.sort()
                # Sample to keep startup cost bounded.
                # Phase D FINAL PDA Finding 1 fix: Python's builtin hash() on
                # strings is randomized per process (PYTHONHASHSEED) by default.
                # Using it as the seed produces a different sample subset on
                # every server restart, which silently changes which medoid is
                # chosen across runs even though candidates.sort() pins the
                # candidate order. md5 is deterministic across processes.
                _seed = int.from_bytes(
                    hashlib.md5(cls.encode("utf-8")).digest()[:4],
                    "big",
                ) & 0xFFFFFFFF
                _random.seed(_seed)
                sample = (candidates if len(candidates) <= MEDOID_SCAN_LIMIT
                           else _random.sample(candidates, MEDOID_SCAN_LIMIT))
                best_dist = float("inf")
                best_path = None
                for path in sample:
                    try:
                        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            continue
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        with torch.no_grad():
                            tens = _preprocess_sp_lora(img_rgb).to(self.device)
                            out = self.sp_lora(tens)
                            feat_raw = out["cls"].squeeze(0).float()  # [768]
                            feat = F.normalize(feat_raw, dim=-1)
                        # L2 distance on unit vectors; smaller = more similar
                        # in cosine sense. Equivalent to 2 - 2*cos(a, b).
                        d = float(torch.norm(feat - centroid).item())
                        if d < best_dist:
                            best_dist = d
                            best_path = path
                    except Exception as e:
                        logger.debug(f"  [medoid] skipped {path}: {e}")
                        continue
                if best_path is None:
                    continue
                # Resize and base64 encode the medoid image.
                img_bgr = cv2.imread(best_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_thumb = cv2.resize(img_rgb, (MEDOID_THUMB_PX, MEDOID_THUMB_PX),
                                         interpolation=cv2.INTER_AREA)
                b64 = _png_base64(img_thumb)
                self._class_ref_thumbnails[cls] = {
                    "b64": b64,
                    "source_file": _os.path.basename(best_path),
                    "distance_to_centroid": round(best_dist, 4),
                    "n_candidates_scored": len(sample),
                }
            except Exception as e:
                logger.warning(f"  [medoid] failed for {cls}: {e}")
                continue
        got = len(self._class_ref_thumbnails)
        logger.info(f"  [medoid] resolved {got}/{len(TOMATO_CLASSES)} class reference thumbnails")

    def _load_disease_info(self):
        self.disease_info = load_disease_info()
        logger.info(f"  DISEASE_INFO entries: {sorted(self.disease_info.keys())}")

    def _validate(self):
        # Check all essentials
        required = {"v3", "sp_lora", "T_sp", "T_v3", "tier_1a_threshold", "disease_info"}
        missing = [a for a in required if not hasattr(self, a)]
        if missing:
            raise RuntimeError(f"TomatoPipeline missing attributes: {missing}")
        # Verify remap sanity (defensive)
        from scripts.model3_training.model3_config import CLASS_NAMES as V3_ORDER
        for i, cls in enumerate(TOMATO_CLASSES):
            v3_idx = V3_INDEX_FOR_LORA_CLASS[i]
            if V3_ORDER[v3_idx] != cls:
                raise RuntimeError(
                    f"Index remap error: TOMATO_CLASSES[{i}]={cls} != V3[{v3_idx}]={V3_ORDER[v3_idx]}"
                )
        logger.info("  TomatoPipeline validated.")

    # ----- tier assignment -------------------------------------------------
    def _assign_tier(self, probs: torch.Tensor) -> str:
        """Map (max_prob, gap_to_second) -> tier string matching APIN 12-tier taxonomy.

        Returns one of: "1A", "1B", "2A", "2B", "4A", "4B", per APIN conventions.
        Not all 12 tiers are reachable by tomato path (e.g., 3C requires prototype
        disagreement which Option ζ disables; 5 requires APIN's 4-signal critical
        detection which tomato doesn't have).
        """
        top2 = torch.topk(probs, k=2, dim=-1).values
        max_p = float(top2[0, 0])
        gap = float(top2[0, 0] - top2[0, 1])
        if max_p >= self.tier_1a_threshold and gap >= TIER_1A_GAP_MIN:
            return "1A"
        if max_p >= self.tier_1a_threshold and gap >= TIER_1B_GAP_MIN:
            return "1B"
        if max_p >= TIER_2A_PROB_FLOOR:
            return "2A"
        if max_p >= TIER_ABSTAIN_PROB_FLOOR:
            return "2B"
        return "4A"

    # ----- main inference --------------------------------------------------
    @torch.no_grad()
    def _forward_v3(self, img_rgb: np.ndarray):
        """Returns (post-T 6-class probs, pre-T 6-class probs, 10-class raw logits).

        Stage 1.2 addition: pre_T_probs is the softmax at T=1.0 so the frontend
        can animate the "sharpening" morph from pre-T to post-T. The ensemble
        still uses post-T probs; pre-T is diagnostic only, never fed to
        downstream decision logic.
        """
        x = _preprocess_v3(img_rgb).to(self.device)
        crop_mode = torch.tensor([2], dtype=torch.long, device=self.device)
        out = self.v3(x, crop_mode, domain_labels=None)
        logits_full = out["logits"].float()  # [1, 10]
        tomato_logits = logits_full[:, V3_INDEX_FOR_LORA_CLASS]  # [1, 6]
        # PDA B3: apply T exactly once here
        tomato_probs = torch.softmax(tomato_logits / self.T_v3, dim=-1)
        pre_T_probs = torch.softmax(tomato_logits, dim=-1)
        return tomato_probs, pre_T_probs, logits_full

    @torch.no_grad()
    def _forward_sp_lora(self, img_rgb: np.ndarray):
        """Returns (post-T probs, pre-T probs, logits, cls_features).

        Stage 1.2 addition: pre_T_probs diagnostic; when T_sp == 1.0 (the
        current unreliable-calibration fallback) pre_T_probs == post-T probs,
        and the frontend renders that as "identity, no sharpening applied".
        """
        x = _preprocess_sp_lora(img_rgb).to(self.device)
        out = self.sp_lora(x)
        logits = out["logits"].float()  # [1, 6]
        cls_feats = out["cls"].float()   # [1, 768]
        # PDA B3: apply T exactly once (T=1.0 if use_calibration=False, so this is identity)
        probs = torch.softmax(logits / self.T_sp, dim=-1)
        pre_T_probs = torch.softmax(logits, dim=-1)
        return probs, pre_T_probs, logits, cls_feats

    def _compute_gradcam(self, img_rgb: np.ndarray, target_idx: int):
        """Compute Grad-CAM on single-pass LoRA.

        Stage 1.4 change: returns (base64_png, region_dict) rather than just
        the base64. region_dict has keys {primary, quadrant_mass} used by the
        Leaf Attention Badge widget in the UI.

        Phase E.5b: returns (b64, region, source_label) so the UI can show
        which block produced the heatmap. region.cam_saturated stays True
        only if EVERY fallback block also saturated, in which case the b64
        is suppressed (returned as "") so the UI shows a clean leaf with a
        clear "diffuse attention" message rather than a misleading uniform
        blue tint from JET-mapping a flat zero CAM.

        On failure returns ("", None, "sp_lora_cls").
        """
        try:
            with self._lock:  # serialize backward passes across threads
                x = _preprocess_sp_lora(img_rgb).to(self.device)
                cam_2d, source_label = _gradcam_sp_lora(
                    self.sp_lora, x, target_idx, self.device)
            region = self._analyse_cam_region(cam_2d)
            # Phase E.5b: suppress the overlay PNG when the CAM is genuinely
            # saturated even after the multi-block fallback. The JET colormap
            # of an all-zero CAM is just a uniform dark blue, which when
            # alpha-blended over the leaf produces a misleading "everything
            # is the heatmap" tint. Better to show no overlay and label why.
            if region and region.get("cam_saturated"):
                b64 = ""
            else:
                b64 = _overlay_heatmap(img_rgb, cam_2d)
            return b64, region, source_label
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}; returning empty heatmap")
            return "", None, "sp_lora_cls"

    def _analyse_cam_region(self, cam_2d) -> dict:
        """Map a 2D CAM (any size, numpy or torch) to a coarse region label.

        Splits the CAM into a 3x3 grid, computes normalized mass per cell,
        and labels the dominant cell. Purely descriptive; used by the UI to
        say, for example, 'Model attention concentrated on the lower-centre
        leaf region.'
        """
        try:
            import numpy as _np
            arr = cam_2d
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = _np.asarray(arr, dtype=_np.float32)
            # Normalize to [0, 1] so total mass is well-defined
            mn, mx = float(arr.min()), float(arr.max())
            if mx - mn < 1e-9:
                # PDA Issue 4 fix: distinguish a genuinely uniform CAM (rare)
                # from a saturated CAM (common on very-high-confidence
                # predictions where gradients vanish and every cell is ~zero).
                # Both cases yield the same 1/9 distribution, but the UI
                # should render them differently. cam_saturated tells the
                # Stage 3 Leaf Attention Badge to say "model very confident,
                # attention not localized" rather than implying the model
                # literally saw the whole leaf.
                return {"primary": "uniform",
                        "quadrant_mass": {k: 1.0/9 for k in _REGION_LABELS},
                        "dispersion": 1.0,
                        "cam_saturated": True}
            arr = (arr - mn) / (mx - mn)
            H, W = arr.shape[:2]
            # 3x3 grid indexing rows (top, mid, bot) x cols (left, centre, right)
            row_edges = [0, H//3, 2*H//3, H]
            col_edges = [0, W//3, 2*W//3, W]
            masses = {}
            total = 0.0
            for ri, rlabel in enumerate(("top", "mid", "bot")):
                for ci, clabel in enumerate(("left", "centre", "right")):
                    cell = arr[row_edges[ri]:row_edges[ri+1],
                                col_edges[ci]:col_edges[ci+1]]
                    m = float(cell.sum())
                    masses[f"{rlabel}_{clabel}"] = m
                    total += m
            if total < 1e-9:
                return {"primary": "uniform",
                        "quadrant_mass": {k: 1.0/9 for k in _REGION_LABELS},
                        "dispersion": 1.0,
                        "cam_saturated": True}
            # Normalize to fractions of total mass
            masses_n = {k: round(v/total, 4) for k, v in masses.items()}
            primary = max(masses_n, key=masses_n.get)
            # Simple dispersion measure: how far the top cell is from uniform (1/9)
            top_frac = masses_n[primary]
            dispersion = round(max(0.0, min(1.0, 1.0 - (top_frac - 1.0/9) / (1.0 - 1.0/9))), 4)
            return {
                "primary": primary,
                "quadrant_mass": masses_n,
                "dispersion": dispersion,
                "cam_saturated": False,
            }
        except Exception as e:
            logger.warning(f"CAM region analysis failed: {e}")
            return None

    def _nearest_prototypes(self, cls_feats: torch.Tensor, k: int = 3) -> list:
        """Top-k nearest prototypes by cosine similarity (diagnostic only).

        Returns list of {class, prototype_index, cosine_similarity} sorted by similarity desc.
        """
        if self.proto_bank is None:
            return []
        cls_norm = F.normalize(cls_feats.squeeze(0), dim=-1)  # [768]
        results = []
        for cls in TOMATO_CLASSES:
            protos = self.proto_bank["prototypes"][cls]  # [k_protos, 768]
            protos_norm = F.normalize(protos, dim=-1)
            sims = protos_norm.cpu() @ cls_norm.cpu()  # [k_protos]
            for i, sim in enumerate(sims.tolist()):
                results.append({
                    "class": cls,
                    "prototype_index": i,
                    "cosine_similarity": round(float(sim), 4),
                })
        results.sort(key=lambda r: -r["cosine_similarity"])
        return results[:k]

    def infer(self, img_rgb: np.ndarray,
              routing_info: Optional[dict] = None) -> dict:
        """Run the ensemble on an RGB uint8 numpy image. Returns response dict
        matching APIN okra/brassica top-level schema, with tomato-specific
        fields added under `tomato_details`.

        PDA B1 asymmetric sharpening is baked into this call: v3 uses T=0.5
        (sharpen), LoRA uses T=1.0 (no sharpen). The 50/50 probability average
        in consequence means v3 dominates argmax on high-confidence cases.

        Args:
            img_rgb: RGB uint8 numpy array from APIN's parse_image().
            routing_info: dict with 'router_crop', 'router_confidence' etc.
                          If None, defaults to {'router_crop':'tomato', ...}
                          as if router detected tomato with high confidence.
        """
        t0 = time.time()
        img_rgb = _ensure_rgb_uint8(img_rgb)

        # 1) Forward both models. pre-T variants (Stage 1.2) are diagnostic only;
        # they are not fed to the ensemble and do not affect any decision.
        v3_probs, v3_pre_T_probs, _ = self._forward_v3(img_rgb)
        lora_probs, lora_pre_T_probs, _, cls_feats = self._forward_sp_lora(img_rgb)

        # 2) Ensemble in probability space (50/50 with asymmetric T sharpening — see module docstring)
        ensemble_probs = W_V3 * v3_probs + W_LORA * lora_probs  # [1, 6]

        # 3) Argmax + all-class breakdown
        top_idx = int(ensemble_probs.argmax(dim=-1).item())
        top_class = TOMATO_CLASSES[top_idx]
        top_prob = float(ensemble_probs[0, top_idx].item())

        # Second-most-likely for differential
        top2_values, top2_indices = torch.topk(ensemble_probs, k=2, dim=-1)
        second_idx = int(top2_indices[0, 1].item())
        second_class = TOMATO_CLASSES[second_idx]
        second_prob = float(top2_values[0, 1].item())

        # Per-class probs dict (matches APIN structure; only 6 tomato classes, no chilli)
        all_class_probs = {
            cls: round(float(ensemble_probs[0, i].item()), 6)
            for i, cls in enumerate(TOMATO_CLASSES)
        }

        # 4) Tier assignment
        tier = self._assign_tier(ensemble_probs)

        # 5) Prototype blending (Option ζ: disabled by default, kept as debug diagnostic)
        prototype_blending_applied = False
        if ENABLE_PROTOTYPE_BLENDING:
            # Placeholder: would blend here if re-enabled in future
            prototype_blending_applied = True

        # 6) Disease info lookup
        info = self.disease_info.get(top_class, {})
        second_info = self.disease_info.get(second_class, {})

        # 7) Grad-CAM on single-pass LoRA (the 392px model with 28x28 patches).
        # Stage 1.4: _compute_gradcam now returns (base64, region_dict). The
        # region dict carries quadrant mass used by the Leaf Attention Badge.
        # Phase E.5b: tuple is now (b64, region, source_label). source_label
        # tells us which ViT block produced the CAM after multi-block fall-
        # back; surfaced to the UI as `gradcam_source_signal`.
        heatmap_b64, attention_region, gradcam_source = self._compute_gradcam(
            img_rgb, top_idx)

        # 8) Diagnostic: nearest prototypes
        nearest = self._nearest_prototypes(cls_feats, k=3)

        # 8b) Stage 1.3: project the current CLS token into the same 2D PCA
        # plane as the prototype centroids, for the constellation widget.
        proto_2d = None
        if self._proto_pca is not None:
            qx, qy = self._project_cls_token_to_2d(cls_feats)
            if qx is not None:
                centroids_xy = self._proto_pca["centroids_2d"].tolist()
                proto_2d = {
                    "centroids_xy": [[round(float(c[0]), 6),
                                       round(float(c[1]), 6)] for c in centroids_xy],
                    "class_labels": list(self._proto_pca["class_labels"]),
                    "query_xy": [round(qx, 6), round(qy, 6)],
                    "explained_variance": [round(float(v), 6)
                                             for v in self._proto_pca["explained_var"]],
                }

        # 9) Agreement analysis (for tomato_details)
        v3_argmax = TOMATO_CLASSES[int(v3_probs.argmax(dim=-1).item())]
        lora_argmax = TOMATO_CLASSES[int(lora_probs.argmax(dim=-1).item())]
        agreement = (v3_argmax == lora_argmax)
        v3_max = float(v3_probs.max().item())
        lora_max = float(lora_probs.max().item())
        agreement_strength = (
            "strong" if (agreement and v3_max > 0.80 and lora_max > 0.80)
            else "moderate" if agreement
            else "weak"
        )

        # 10) Calibration warnings for the response
        cal_warnings = []
        if self.sp_cal.get("pda_T_stability_interpretation") == "UNRELIABLE":
            cal_warnings.append("single-pass LoRA T calibration flagged UNRELIABLE; using T=1.0")
        # Note v3 T=0.5 boundary (PDA Issue 2)
        if abs(self.T_v3 - 0.5) < 1e-6:
            cal_warnings.append("v3 T=0.5 is at calibration search lower boundary; "
                                "true optimum may be lower (see Decision 56 addendum)")
        # PDA B4: YLCV hedging
        if top_class == YLCV_CLASS:
            cal_warnings.append(YLCV_HEDGE_MESSAGE)

        # 10b) Stage 1.5 tier decision path. Documents HOW the current tier
        # was reached by the existing _assign_tier logic. Does NOT change the
        # behavior; it only exposes the reasoning as structured data so the
        # UI Tier Decision Tree widget can render the branch taken.
        # Keys match the exact thresholds used in _assign_tier at this moment.
        tier_1a_thr = float(self.tier_1a_threshold)
        gap_to_second = round(top_prob - second_prob, 6)
        tier_decision = {
            "max_prob": round(top_prob, 6),
            "gap_to_second": gap_to_second,
            "agreement": agreement,
            "agreement_strength": agreement_strength,
            "thresholds": {
                "tier_1a_prob": tier_1a_thr,
                "tier_1a_gap_min": float(TIER_1A_GAP_MIN),
                "tier_1b_gap_min": float(TIER_1B_GAP_MIN),
                "tier_2a_prob_floor": float(TIER_2A_PROB_FLOOR),
                "tier_abstain_floor": float(TIER_ABSTAIN_PROB_FLOOR),
            },
            "path_taken": _build_tier_path(
                top_prob, gap_to_second, tier_1a_thr,
                float(TIER_1A_GAP_MIN), float(TIER_1B_GAP_MIN),
                float(TIER_2A_PROB_FLOOR), float(TIER_ABSTAIN_PROB_FLOOR),
                agreement, agreement_strength,
            ),
            "final_tier": tier,
        }

        # PDA Issue 5 runtime guard: verify that the path's terminal reasoning
        # is consistent with the tier _assign_tier actually returned. We compare
        # the last step's check key (tier_*_*) against the tier string. If the
        # two drift (e.g. because someone edits _assign_tier without updating
        # _build_tier_path), log a warning rather than crash. Silent drift
        # would mislead the UI; a loud log ensures it gets caught in CI.
        try:
            last_step = tier_decision["path_taken"][-1] if tier_decision["path_taken"] else {}
            terminal_map = {
                "tier_1a_prob_and_gap": "1A",
                "tier_1b_gap": "1B",
                "tier_2a_floor": "2A",
                "tier_2b_floor": "2B",
            }
            expected_tier = (
                terminal_map.get(last_step.get("check"))
                if last_step.get("passed") else None
            )
            if last_step.get("check") == "tier_2b_floor" and not last_step.get("passed"):
                expected_tier = "4A"
            if expected_tier is not None and expected_tier != tier:
                logger.warning(
                    f"tier_decision drift: path terminal implies {expected_tier}, "
                    f"_assign_tier returned {tier}. This indicates _build_tier_path "
                    f"has diverged from _assign_tier; both should be updated together."
                )
        except Exception as e:
            logger.warning(f"tier_decision drift check failed: {e}")

        # 10c) Stage 1.6a + PDA Issue 3 fix: honest confidence interval.
        # Earlier implementation used [min(v3_max, lora_max), max(v3_max, lora_max)]
        # where the maxes referred to each model's OWN top class. When the two
        # models disagree on the top class that combined two probabilities of
        # different classes into a single interval, which is misleading.
        #
        # Correct semantic: both endpoints must refer to the SAME class, namely
        # the ensemble's predicted top class. That is, "how sure is each model
        # that the ensemble's answer is right?" If v3 agrees strongly and
        # sp_lora has low prob for the same class, the interval widens, which
        # is the correct visual cue for user uncertainty.
        v3_prob_of_top = float(v3_probs[0, top_idx].item())
        lora_prob_of_top = float(lora_probs[0, top_idx].item())
        ci_low = round(min(v3_prob_of_top, lora_prob_of_top), 6)
        ci_high = round(max(v3_prob_of_top, lora_prob_of_top), 6)
        confidence_interval = [ci_low, ci_high]

        # 10d) Stage 1.6b input quality heuristic. Uses image dimensions plus
        # router confidence from routing_info. Deliberately coarse (3 levels);
        # the farmer-facing UI renders a single word plus an icon.
        #
        # PDA Issue 6 fix: pure min-dimension thresholding mislabels extreme
        # aspect ratios (e.g., a 100x1000 rotated leaf photo has 100k pixels
        # but min_dim=100 forces 'low'). We now combine min_dim with total
        # pixel count: a photo with >= 50k pixels and a usable short side
        # (>= 100 px) earns at least 'moderate' even if min_dim < 224.
        try:
            h, w = int(img_rgb.shape[0]), int(img_rgb.shape[1])
        except Exception:
            h = w = 0
        rconf = 0.0
        if routing_info and isinstance(routing_info, dict):
            try:
                rconf = float(routing_info.get("router_confidence") or 0.0)
            except Exception:
                rconf = 0.0
        min_dim = min(h, w) if (h > 0 and w > 0) else 0
        total_px = (h * w) if (h > 0 and w > 0) else 0
        if min_dim >= 224 and rconf >= 0.70:
            iq = "good"
        elif (min_dim < 100) or (total_px < 20_000) or (rconf < 0.40):
            # Tiny image or very low router confidence: unambiguous low.
            iq = "low"
        else:
            iq = "moderate"
        input_quality = {
            "level": iq,
            "min_dimension_px": min_dim,
            "total_pixels": total_px,
            "router_confidence": round(rconf, 4),
        }

        # 10e) Stage 1.7 icon category for the diagnosed class. Maps to a
        # stroke icon on the frontend (fungal, viral, critical, healthy).
        icon_category = CLASS_ICON_CATEGORY.get(top_class, "generic")

        # Phase D SRV-1: generate per-request pipeline thumbnails so the
        # Ensemble Flow Inspector can show what the model actually "saw" at
        # each station. Router input is the raw upload; v3 input is the
        # 224 px LAB-CLAHE stretch-resized view; sp_lora input is the 392 px
        # letterboxed LAB-CLAHE view; gradcam overlay reuses the heatmap
        # PNG that was already built above. All encoded as base64 PNG at
        # max 128 px longer side to keep response size bounded. Any failure
        # returns an empty string so the UI can fall back cleanly.
        # Phase E.1: pipeline_thumbnails now carries intermediate
        # preprocessing steps so the side panel can show the image
        # transformation strip per model. router_step_resized and
        # v3_step_resized are the same image (224 px stretch resize)
        # but emitted under both keys so each station's strip can be
        # built independently of any cross-station knowledge.
        try:
            v3_resized_rgb = _preview_v3_step_resized(img_rgb)
            sp_capped_rgb = _preview_sp_lora_step_capped(img_rgb)
            sp_letter_rgb = _preview_sp_lora_step_letterbox(img_rgb)
            pipeline_thumbnails = {
                # Existing fields kept for backward compatibility
                "router_input": _png_base64(img_rgb, max_side=PIPELINE_THUMB_PX),
                "v3_input": _png_base64(_preview_v3_input(img_rgb),
                                          max_side=PIPELINE_THUMB_PX),
                "sp_lora_input": _png_base64(_preview_sp_lora_input(img_rgb),
                                               max_side=PIPELINE_THUMB_PX),
                "gradcam_overlay": (heatmap_b64 if (heatmap_b64 and not
                    (attention_region and attention_region.get("cam_saturated")))
                    else ""),
                # Phase E.1 new intermediate-step thumbnails (96 px to keep
                # payload bounded; they are explanatory, not specimen-quality)
                "router_step_resized": _png_base64(
                    _preview_router_input(img_rgb), max_side=96),
                "v3_step_resized": _png_base64(v3_resized_rgb, max_side=96),
                "sp_lora_step_capped": _png_base64(sp_capped_rgb, max_side=96),
                "sp_lora_step_letterbox": _png_base64(sp_letter_rgb, max_side=96),
            }
        except Exception as e:
            logger.warning(f"pipeline_thumbnails generation failed: {e}")
            pipeline_thumbnails = {
                "router_input": "", "v3_input": "",
                "sp_lora_input": "", "gradcam_overlay": "",
                "router_step_resized": "", "v3_step_resized": "",
                "sp_lora_step_capped": "", "sp_lora_step_letterbox": "",
            }

        # 11) Output message (farmer-facing)
        output_message = self._compose_output_message(top_class, top_prob, tier, info)

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        # 12) Build response matching APIN top-level schema + tomato_details extension
        # Set APIN-specific fields to None/empty so frontend JS doesn't crash on missing keys.
        routing = routing_info or {
            "router_crop": "tomato", "router_confidence": 1.0,
            "router_handled": True, "low_router_confidence": False,
        }
        # Update router_handled since tomato is now handled
        if isinstance(routing, dict):
            routing = {**routing, "router_handled": True}

        return {
            # -- Core APIN-compatible fields --
            "routing": routing,
            "diagnosis": top_class,
            "confidence": round(top_prob, 6),
            "tier": tier,
            "all_class_probabilities": all_class_probs,
            "output_message": output_message,
            "treatment_recommendation": info.get("action", ""),
            "gradcam_b64_png": heatmap_b64 if heatmap_b64 else None,
            # Phase E.5b: source label is dynamic — may be sp_lora_cls (the
            # last-block CAM, original behavior) or sp_lora_block-3 / -6 / -9
            # if the multi-block fallback found earlier-block signal. None
            # when CAM was suppressed (saturated even after fallback).
            "gradcam_source_signal": gradcam_source if heatmap_b64 else None,
            "processing_time_ms": elapsed_ms,
            # -- APIN-specific fields — set to None/empty for tomato (frontend must handle) --
            "cold_start_tier_downgraded": False,
            "conflict_type": None,
            "conformal_prediction_set": [],
            "decision_trace": [{
                "step": "tomato_v3_lora_ensemble",
                "tier": tier,
                "diagnosis": top_class,
                "v3_prediction": v3_argmax,
                "lora_prediction": lora_argmax,
                "agreement": agreement,
            }],
            "differential_guidance": (
                f"Alternative diagnosis: {second_class} ({second_prob*100:.1f}%)"
                if second_prob > 0.10 else None
            ),
            "failed_signals": [],
            "gate_weights": None,
            "is_ood": (top_prob < TIER_ABSTAIN_PROB_FLOOR),
            "mahalanobis_distance": None,
            "monitoring_guidance": None,
            "per_class_mahalanobis": {},
            "per_signal_full_distributions": {},
            "pipeline_visualizations": {},
            "psv_feature_firing": [],
            "quality_flags": {},
            "retake_guidance": None,
            "signal_contributions": {},
            "signal_predictions": {
                "v3": {"diagnosis": v3_argmax, "confidence": round(v3_max, 6)},
                "sp_lora": {"diagnosis": lora_argmax, "confidence": round(lora_max, 6)},
            },
            "uncertainty_aleatoric": None,
            "uncertainty_epistemic": None,
            # -- Tomato-specific extension --
            "specialist": "tomato_v3_sp_lora_ensemble",
            "tomato_details": {
                "v3_prediction": {"class": v3_argmax, "confidence": round(v3_max, 6)},
                "lora_prediction": {"class": lora_argmax, "confidence": round(lora_max, 6)},
                "agreement": agreement,
                "agreement_strength": agreement_strength,
                "ensemble_type": "v3_lora_50_50_no_blend",
                "ensemble_weights": {"v3": W_V3, "sp_lora": W_LORA},
                "secondary": {"class": second_class, "confidence": round(second_prob, 6)},
                "prototype_matches": nearest,
                "prototype_blending_applied": prototype_blending_applied,
                # PDA Issue 13 fix: round to 6 dp for consistency with every
                # other float in tomato_details. Stored calibration constant
                # in the JSON file is 0.5000000000000011 due to LBFGS fit
                # float error; display should read a clean 0.5.
                "t_v3_applied": round(float(self.T_v3), 6),
                "t_sp_lora_applied": round(float(self.T_sp), 6),
                "calibration_warnings": cal_warnings,
                # Stage 1.2 pre-temperature probabilities (diagnostic only).
                # Frontend uses these to render the sharpening animation in
                # the Ensemble Metadata panel.
                "pre_temperature_probs": {
                    "v3": [round(float(p), 6) for p in v3_pre_T_probs.squeeze(0).tolist()],
                    "sp_lora": [round(float(p), 6) for p in lora_pre_T_probs.squeeze(0).tolist()],
                    "class_order": list(TOMATO_CLASSES),
                },
                # Stage 1.3 prototype 2D projection for the constellation widget.
                # None if the prototype bank or PCA was unavailable.
                "prototype_2d": proto_2d,
                # Stage 1.4 Grad-CAM region analysis for the attention badge.
                "attention_region": attention_region,
                # Stage 1.5 structured tier decision path for the tree widget.
                "tier_decision": tier_decision,
                # Stage 1.6a honest two-point confidence interval.
                "confidence_interval": confidence_interval,
                # Stage 1.6b input quality heuristic for the margin rail.
                "input_quality": input_quality,
                # Stage 1.7 class icon category for the frontend to choose
                # its stroke icon from the existing symbol library.
                "icon_category": icon_category,
                # PDA Issue 7 fix: clarify the role of every probability source
                # in the response so Stage 2/3 UI work has a single explicit
                # contract to follow. Prevents the UI from accidentally
                # rendering a diagnostic probability as the production one.
                "probability_sources": {
                    "authoritative": "all_class_probabilities",
                    "per_model_argmax": "signal_predictions",
                    "diagnostic_pre_temperature": "tomato_details.pre_temperature_probs",
                    "notes": (
                        "all_class_probabilities carries the production "
                        "50/50 ensemble. signal_predictions reports each model's "
                        "own argmax and confidence. pre_temperature_probs are "
                        "diagnostic softmax(logits at T=1) never fed into any "
                        "decision logic."
                    ),
                },
                "asymmetric_sharpening_note": (
                    "v3 at T=0.5 (sharpened 2x) vs sp_lora at T=1.0 (no sharpen). "
                    "'50/50' means probability-space averaging, not equal argmax "
                    "contribution. v3 dominates argmax on high-confidence cases. "
                    "See Decision 56."
                ),
                "expected_heldout_note": (
                    "Single-pass LoRA alone held-out = 0.7620 [CI 0.66-0.87]. "
                    "Ensemble held-out NOT measured (would violate LOCK-4). "
                    "v3 held-out baseline ~0.85 is second-hand (not re-verified "
                    "this session). Ensemble projected floor: single-pass held-out."
                ),
                # Phase D SRV-1: per-request pipeline thumbnails. Four base64
                # PNGs showing the image as it appears at each key station of
                # the ensemble flow. The Leaf Venation Flow Inspector uses
                # these when the user clicks a station to see what the model
                # actually "saw" at that step.
                "pipeline_thumbnails": pipeline_thumbnails,
                # Phase D SRV-2: class reference thumbnails resolved via
                # medoid search at server startup. Dict of 6 entries (or
                # fewer if any class directory was missing) with base64 PNG,
                # source filename, and distance-to-centroid. None when
                # medoid resolution failed at startup. Consumed by the
                # Prototype Proximity "Your leaf looks most like" strip.
                "class_reference_thumbnails": (
                    self._class_ref_thumbnails
                    if getattr(self, "_class_ref_thumbnails", None)
                    else None
                ),
                # Phase E.1: per-class plain-English symptom descriptions for
                # the constellation hover-card mini field guide. Static
                # constant from the module top, shipped on every tomato
                # response (small, ~700 bytes total).
                "class_symptom_descriptions": dict(TOMATO_CLASS_SYMPTOMS),
            },
        }

    def _compose_output_message(self, cls: str, prob: float, tier: str, info: dict) -> str:
        """Farmer-facing message string. Emoji-free and em-dash-free (Stage 1.1)."""
        name = info.get("name", cls.replace("_", " ").title())
        sev = info.get("severity", "Medium")
        if tier == "1A":
            hdr = "Confident diagnosis"
        elif tier == "1B":
            hdr = "Probable diagnosis"
        elif tier in ("2A", "2B", "2C", "2D"):
            hdr = "Differential diagnosis (two candidates possible)"
        elif tier in ("3A", "3B", "3C"):
            hdr = "Tentative diagnosis (confidence reduced by image or agreement checks)"
        elif tier in ("4A", "4B"):
            hdr = "Low confidence (consider retake or expert review)"
        elif tier == "5":
            hdr = "Critical urgency diagnosis"
        else:
            hdr = "Diagnosis produced"
        return f"{hdr}: {name} ({prob*100:.0f}%). Severity: {sev}."
