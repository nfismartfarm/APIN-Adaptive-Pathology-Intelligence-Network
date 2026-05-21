"""
APIN Unified Inference Pipeline — the authoritative production inference.

Layers 0-7 of APIN architecture. Builds on the 3-signal or 4-signal stacking
MLP trained in Section 4. All production inference flows through APINInference.

Key design:
  - Two preprocessing branches: A (LAB-CLAHE) for Model 2, DINOv2, PSV;
                                 B (RGB-CLAHE) for EfficientNet.
  - Single-pass eval() for all neural models (no MC Dropout).
  - Gate Zero pre-checks (quality, leaf presence, OOD-lite).
  - Signal caches are inference-time (not the training caches) — each
    inference produces fresh predictions.
  - Conformal prediction sets + 9 output tiers (1A, 1B, 1C, 2A, 2B, 2C,
    3A, 3B, 3C, 4A, 4B, 5).

Usage:
    from scripts.apin.inference import APINInference
    apin = APINInference()
    result = apin.predict(image_np_uint8_hw3)
    print(result.tier, result.diagnosis, result.confidence)
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "scripts" / "apin" / "caches"

logger = logging.getLogger("apin.inference")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)


# ========================================================================
# CONSTANTS
# ========================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
M2_IMG_SIZE = 384  # Model 2 trained at 384
EN_IMG_SIZE = 224  # EfficientNet trained at 224
DINO_IMG_SIZE = 224  # DINOv2 trained at 224

# Gate Zero thresholds
BLUR_SEVERE = 80
BLUR_MILD = 200
LAB_L_UNDEREXPOSED_SEVERE = 20
LAB_L_OVEREXPOSED_SEVERE = 92
LEAF_FRACTION_MIN = 0.10
MIN_IMG_DIM = 100

# Output tier thresholds — aligned with new_architecture_convo.md spec
# 1A: high confidence + ALL signals agree
# 1B: moderate-high confidence + 3-of-4 (or all-of-3) agree
# 1C: moderate confidence + clear gap to second class
# 2A: lower confidence, top still leads, second class material (>=0.20)
# 2B: two close contenders (small gap, both substantial)
# 2C: three or more candidates above 0.20 — true differential
# 3A: healthy prediction with adequate confidence (no disease evidence)
# 3B: image quality limits diagnosis (mild blur or exposure issue)
# 3C: atypical / signals strongly disagree (conflict C/D)
# 4A: OOD + low confidence
# 4B: OOD + moderate confidence
TIER_1A_CONFIDENCE = 0.55     # spec: top > 0.55 + entropy < 0.20
TIER_1A_ENTROPY_MAX = 0.20    # spec entropy ceiling
TIER_1B_CONFIDENCE = 0.45     # spec: 0.45-0.55 OR mild conflict B1
TIER_1B_ENTROPY_MAX = 0.40
TIER_1C_CONFIDENCE = 0.35     # spec: 0.35-0.55 + gap >= 0.15
TIER_1C_GAP_MIN = 0.15
TIER_2A_CONFIDENCE = 0.30
TIER_2A_SECOND_MIN = 0.20
TIER_2A_GAP_MAX = 0.20
TIER_2B_CONFIDENCE = 0.25
TIER_2B_SECOND_MIN = 0.25
TIER_2B_GAP_MAX = 0.10
TIER_2C_ABOVE = 0.20          # 3+ classes above this → 2C
TIER_3A_HEALTHY_MIN = 0.50    # healthy prediction needs at least this
TIER_4A_CONFIDENCE_MAX = 0.40 # OOD + below this → 4A, else 4B
TIER_DIFFERENTIAL_GAP = 0.10  # legacy

# Cold-start downgrade list — Decision 14 in architecture_claude_decisions.md
# These classes have either documented field-photo failure modes (M2 fails on
# brassica_black_rot/okra_cercospora) or low reliability matrix scores
# (okra_enation = 0.86 lowest in 4-signal matrix). Tier emitted is downgraded
# one level until cold-start phase completes.
COLD_START_DOWNGRADE_CLASSES = (
    "brassica_black_rot",
    "okra_cercospora",
    "okra_enation",
)

# Tier 5 — critical urgency override (Gap 4 audit fix).
# Triggers when ANY signal predicts a rapid-spread / high-impact disease at
# very high confidence regardless of consensus. Routing here bypasses the
# normal tier ladder. The classes listed are pathogens that:
#   - spread by airborne / vector / rain-splash within hours
#   - have no curative treatment once established at field scale
#   - cause >50% yield loss within one growing season if untreated
# A farmer must be told to act today, not wait for monitoring.
TIER_5_CRITICAL_DISEASES = (
    "brassica_black_rot",      # Xanthomonas — rain-splash + tool-borne, can wipe a field in days
    "brassica_downy_mildew",   # Hyaloperonospora — airborne sporulation, doubles every 24h cool-wet
    "okra_yvmv",               # Yellow Vein Mosaic Virus — whitefly vector, no cure, must rogue plants
    "okra_enation",            # Begomovirus complex — same vector pressure as YVMV
)
TIER_5_SIGNAL_CONFIDENCE = 0.85   # any single signal must clear this on the disease
TIER_5_MIN_AGREEING_SIGNALS = 2   # at least 2 of 4 signals must agree on the disease

# Disease stage estimator thresholds (Gap 3 audit fix).
# F08_disease_stage_index from PSV is in [0, 1]:
#   < EARLY  → "no symptoms / early stage" → Tier 3A (monitor)
#   < MID    → "mid stage" → Tier 3B catch-all (image-quality / uncertain)
#   ≥ MID    → "late stage" → escalate to Tier 5 if also TIER_5 disease
# Used only when PSV signal is present and disease prediction emerges.
DISEASE_STAGE_EARLY = 0.20
DISEASE_STAGE_MID = 0.55


# ========================================================================
# RESULT DATACLASSES
# ========================================================================
@dataclass
class GateZeroResult:
    hard_reject: bool = False
    retake_reason: Optional[str] = None
    quality_score: float = 1.0
    blur_score: float = 0.0
    lab_L_mean: float = 0.0
    leaf_fraction: float = 0.0
    is_duplicate: bool = False
    quality_flags: dict = field(default_factory=dict)


@dataclass
class APINResult:
    tier: str = ""                          # 1A / 1B / 1C / 2A / 2B / 2C / 3A / 3B / 3C / 4A / 4B / 5
    diagnosis: Optional[str] = None
    confidence: float = 0.0
    all_class_probabilities: dict = field(default_factory=dict)
    conformal_prediction_set: list = field(default_factory=list)
    conflict_type: str = ""                 # A / B1 / B2 / B3 / C1 / C2 / D
    signal_predictions: dict = field(default_factory=dict)  # per-signal argmax + top prob
    gate_weights: dict = field(default_factory=dict)
    uncertainty_aleatoric: float = 0.0
    uncertainty_epistemic: float = 0.0
    output_message: str = ""
    treatment_recommendation: Optional[str] = None
    monitoring_guidance: Optional[str] = None
    differential_guidance: Optional[str] = None
    retake_guidance: Optional[str] = None
    is_ood: bool = False
    mahalanobis_distance: float = 0.0
    processing_time_ms: float = 0.0
    quality_flags: dict = field(default_factory=dict)
    failed_signals: list = field(default_factory=list)
    cold_start_tier_downgraded: bool = False
    # Gap 2 audit: which signal's backbone produced the GradCAM heatmap.
    # Picked as the signal with the highest gate weight on the predicted
    # class (Addition 1 from the design spec). Empty if heatmap not generated.
    gradcam_b64_png: Optional[str] = None
    gradcam_source_signal: Optional[str] = None
    # Pipeline-inspector image transformations: per-stage base64 PNGs of
    # what the image actually LOOKED LIKE at each transforming stage of
    # the pipeline. Only stages that genuinely transform the image
    # produce entries here. Keys: "gate_zero_lap", "gate_zero_lab_l",
    # "gate_zero_leaf_mask", "preproc_lab_clahe", "preproc_rgb_clahe".
    pipeline_visualizations: dict = field(default_factory=dict)

    # ── Research/explainability additions (2026-04-18) ──────────────────
    # Decision trace — ordered list of conditions evaluated in _determine_tier
    # (each item {step, check, value, threshold, passed, verdict}). Emitted
    # so the UI can display "Why this tier?" without re-running logic.
    decision_trace: list = field(default_factory=list)
    # Full 9-dim probability vector from each successful signal (4×9 matrix
    # for 4-signal mode). Keyed by signal name.
    per_signal_full_distributions: dict = field(default_factory=dict)
    # Mahalanobis distance from the DINOv2 raw feature to EVERY class
    # prototype (not just the nearest). Keyed by class name.
    per_class_mahalanobis: dict = field(default_factory=dict)
    # Top PSV features firing for the predicted class, sorted by absolute
    # contribution. Each entry: {name, value, coef, contribution, sign}.
    psv_feature_firing: list = field(default_factory=list)
    # SHAP-style per-signal contribution toward the predicted class:
    # {signal_name: gate_weight × reliability × signal_prob_for_top_class}.
    signal_contributions: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ========================================================================
# PREPROCESSING BRANCHES
# ========================================================================
def apply_lab_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0,
                    tile_size: tuple = (8, 8)) -> np.ndarray:
    """Branch A: LAB-CLAHE (L channel only). Matches Model 2, DINOv2, PSV training."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_rgb_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0,
                     tile_size: tuple = (8, 8)) -> np.ndarray:
    """Branch B: RGB per-channel CLAHE. Matches EfficientNet (old_10class) training.
    This function replicates the exact behavior of apply_clahe() in
    old_10class/app/inference.py:36-46.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = np.zeros_like(img_rgb)
    for c in range(3):
        result[:, :, c] = clahe.apply(img_rgb[:, :, c])
    return result


def preprocess_branch_a(img_rgb: np.ndarray, size: int) -> torch.Tensor:
    """Branch A: LAB-CLAHE -> resize -> normalize."""
    img = apply_lab_clahe(img_rgb)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def preprocess_branch_b(img_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """Branch B: RGB-CLAHE -> resize -> normalize. For EfficientNet only."""
    img = apply_rgb_clahe(img_rgb)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


# ========================================================================
# GATE ZERO
# ========================================================================
def run_gate_zero(img_rgb: np.ndarray) -> GateZeroResult:
    """Fast (< 10ms) pre-checks: quality, leaf presence, basic OOD."""
    result = GateZeroResult()

    h, w = img_rgb.shape[:2]
    if h < MIN_IMG_DIM or w < MIN_IMG_DIM:
        result.hard_reject = True
        result.retake_reason = (
            f"Image too small ({w}x{h}). Please upload an image at least "
            f"{MIN_IMG_DIM}x{MIN_IMG_DIM} pixels."
        )
        return result

    # Blur detection (Laplacian variance on grayscale)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    result.blur_score = blur_score
    if blur_score < BLUR_SEVERE:
        result.hard_reject = True
        result.retake_reason = (
            f"Image too blurry (sharpness score {blur_score:.0f}). "
            f"Please retake with steadier hands or better lighting."
        )
        return result
    if blur_score < BLUR_MILD:
        result.quality_flags["mild_blur"] = True

    # Exposure (LAB L channel mean)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab_l_mean = float(lab[:, :, 0].mean()) / 255.0 * 100  # scale to 0-100
    result.lab_L_mean = lab_l_mean
    if lab_l_mean < LAB_L_UNDEREXPOSED_SEVERE:
        result.hard_reject = True
        result.retake_reason = (
            f"Image too dark (brightness {lab_l_mean:.0f}/100). "
            f"Please retake with better lighting."
        )
        return result
    if lab_l_mean > LAB_L_OVEREXPOSED_SEVERE:
        result.hard_reject = True
        result.retake_reason = (
            f"Image overexposed (brightness {lab_l_mean:.0f}/100). "
            f"Please retake with less direct sunlight."
        )
        return result

    # Leaf presence — HSV green region fraction
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # green range in HSV
    green_mask = cv2.inRange(hsv, (30, 30, 30), (95, 255, 255))
    leaf_fraction = float(green_mask.mean() / 255.0)
    result.leaf_fraction = leaf_fraction
    if leaf_fraction < LEAF_FRACTION_MIN:
        result.hard_reject = True
        result.retake_reason = (
            f"No clear leaf detected ({leaf_fraction*100:.1f}% green coverage). "
            f"Please upload a close-up of a single leaf."
        )
        return result

    # Compose quality score: product of per-check scores
    quality = 1.0
    if blur_score < BLUR_MILD:
        quality *= min(1.0, blur_score / BLUR_MILD)
    if lab_l_mean < 35:
        quality *= min(1.0, lab_l_mean / 35)
    if lab_l_mean > 85:
        quality *= min(1.0, (100 - lab_l_mean) / 15)
    result.quality_score = float(quality)

    return result


# ========================================================================
# MAIN INFERENCE CLASS
# ========================================================================
class APINInference:
    """Production APIN inference. Loads all models once at construction,
    runs predict() per image.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing APINInference on {self.device}")

        from scripts.apin.constants import (
            MODEL2_CLASS_ORDER, NUM_CLASSES, EN_TO_M2_INDEX_MAP,
        )
        self.class_order = list(MODEL2_CLASS_ORDER)
        self.num_classes = NUM_CLASSES
        self.en_index_map = EN_TO_M2_INDEX_MAP
        # Cold-start downgrade list lives in COLD_START_DOWNGRADE_CLASSES
        # (module-level) — broader than constants.FAILURE_CLASSES because it
        # also includes okra_enation (low reliability score 0.86). Do NOT
        # introduce another self.failure_classes attribute — that diverged
        # from COLD_START_DOWNGRADE_CLASSES in a prior version and caused
        # silent inconsistency.

        # Load stacking MLP
        self._load_stacking_mlp()

        # Load calibration
        self._load_calibration()

        # Load Mahalanobis OOD detector (DINOv2 feature space)
        self._ood_detector = None
        self._load_ood_detector()

        # Load the individual signal models lazily — only if predict() is called
        self._model2 = None
        self._efficientnet = None
        self._dinov2_backbone = None
        self._dinov2_head = None
        self._dinov2_scaler = None
        # Lock guards all _lazy_load_* methods. Without it, concurrent first
        # predict() calls could each pass the `is None` check (the GIL is
        # released during torch.load disk I/O) and race to load the same
        # model twice — wasted GPU memory at best, CUDA OOM at worst.
        self._load_lock = threading.Lock()

        # PSV
        from scripts.psv.feature_extractor import extract_all_features
        from scripts.psv.image_quality import assess_image_quality
        from scripts.psv.disease_scores import compute_disease_scores
        from scripts.psv.calibration import load_calibration as psv_load_cal
        self.psv_extract = extract_all_features
        self.psv_iqa = assess_image_quality
        self.psv_scores = compute_disease_scores
        psv_cal_path = CACHE_DIR / "psv_calibration.json"
        if psv_cal_path.exists():
            self.psv_calibration = psv_load_cal(str(psv_cal_path))
            logger.info(f"  PSV calibration loaded: {len(self.psv_calibration)} features")
        else:
            self.psv_calibration = None
            logger.info("  PSV calibration NOT loaded — Signal 3 disabled")

        # Check Section 3A supervised weights for the BR/ALT pair (Round 7
        # audit Issue B). PSV silently falls back to the hand-engineered
        # formula (separation -0.043) if the JSON is missing — and the
        # hand formula is significantly worse than the supervised path
        # (separation +0.5133). Surface this in the startup log so the
        # operator knows whether PSV is in degraded mode for this pair.
        from scripts.psv.disease_scores import _SUPERVISED_BR_ALT_PATH
        if _SUPERVISED_BR_ALT_PATH.exists():
            logger.info(f"  PSV BR/ALT supervised weights present "
                        f"(Section 3A artifact loaded, separation 0.51)")
        else:
            logger.warning(
                f"  PSV BR/ALT supervised weights MISSING at "
                f"{_SUPERVISED_BR_ALT_PATH.name} — PSV is using the "
                f"hand-engineered black_rot/alternaria formula with "
                f"separation -0.043 (vs +0.51 with supervised path). "
                f"Re-run scripts/apin/section3a_supervised_feature_importance.py "
                f"to restore."
            )

        # Diagnosis lookup (treatment + prevention)
        diag_path = PROJECT_ROOT / "diagnosis" / "diagnosis_lookup.json"
        if diag_path.exists():
            with open(diag_path) as f:
                self.diag_db = json.load(f)
        else:
            self.diag_db = {}

        logger.info("APIN inference ready.")

    # --------------------------------------------------------------------
    # Model loaders
    # --------------------------------------------------------------------
    def _load_stacking_mlp(self):
        from scripts.apin.section4_stacking_mlp import APIN_Ensemble
        ckpt_path = CACHE_DIR / "apin_stacking_mlp.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.n_signals = ckpt["n_signals"]
        self.use_psv = ckpt["use_psv"]
        self.reliability_matrix = np.array(ckpt["reliability_matrix"], dtype=np.float32)
        self.stacking_mlp = APIN_Ensemble(
            n_signals=self.n_signals, num_classes=self.num_classes
        ).to(self.device)
        self.stacking_mlp.load_state_dict(ckpt["model_state_dict"])
        self.stacking_mlp.eval()
        self.stacking_mlp_gate_mean = ckpt.get("gate_mean", None)
        logger.info(f"  Stacking MLP: {self.n_signals}-signal, "
                    f"val_macro_f1 at best={ckpt.get('val_macro_f1')}")

    def _load_calibration(self):
        cal_path = CACHE_DIR / "apin_calibration.json"
        if not cal_path.exists():
            logger.warning(f"  No calibration file at {cal_path} — using defaults")
            self.calibration = {}
            self.per_class_temps = np.ones(9, dtype=np.float32)
            self.conformal_thresholds = np.full(9, 0.5, dtype=np.float32)
            self.cold_start_active = True
            self.adaptive_multipliers = {}
            return
        with open(cal_path) as f:
            self.calibration = json.load(f)
        t_map = self.calibration["temperature_scaling"]["per_class_temperatures"]
        self.per_class_temps = np.array(
            [t_map[c] for c in self.class_order], dtype=np.float32
        )
        q_map = self.calibration["conformal_prediction"]["per_class_thresholds"]
        self.conformal_thresholds = np.array(
            [q_map[c] for c in self.class_order], dtype=np.float32
        )
        self.cold_start_active = self.calibration["conformal_prediction"][
            "cold_start_active"
        ]
        # Adaptive threshold multipliers (Section 5B): scale conformal +
        # tier confidence thresholds based on quality / agreement / OOD
        # distance bucket.
        self.adaptive_multipliers = self.calibration.get(
            "adaptive_threshold_multipliers", {}
        )
        logger.info(f"  Calibration loaded: cold_start={self.cold_start_active}, "
                    f"adaptive_multipliers={'yes' if self.adaptive_multipliers else 'no'}")

    def _load_ood_detector(self):
        """Load Mahalanobis OOD detector from DINOv2 feature space.
        Detector dict has: class_means {cls_idx: (768,)}, class_cov_inv
        {cls_idx: (768,768)}, threshold (float). Computed per-class on
        in-distribution features in scripts/dinov2_probe."""
        if hasattr(self, "_ood_detector") and self._ood_detector is not None:
            return
        ood_path_str = self.calibration.get("ood_detector_path")
        if not ood_path_str:
            self._ood_detector = None
            return
        # Path may be Windows-style; resolve relative to PROJECT_ROOT
        ood_path = PROJECT_ROOT / ood_path_str.replace("\\", "/")
        if not ood_path.exists():
            logger.warning(f"  OOD detector file missing: {ood_path}")
            self._ood_detector = None
            return
        with open(ood_path, "rb") as f:
            d = pickle.load(f)
        # Pre-cast to numpy float32 once for speed
        d["class_means"] = {int(k): np.asarray(v, dtype=np.float32)
                              for k, v in d["class_means"].items()}
        d["class_cov_inv"] = {int(k): np.asarray(v, dtype=np.float32)
                                for k, v in d["class_cov_inv"].items()}
        d["threshold"] = float(d["threshold"])
        self._ood_detector = d
        logger.info(f"  Mahalanobis OOD detector loaded "
                    f"(threshold={d['threshold']:.2f}, "
                    f"{len(d['class_means'])} class prototypes)")

    def _lazy_load_model2(self):
        # Double-checked locking: fast path reads without lock; slow path
        # acquires lock and rechecks. Prevents two threads from both loading
        # the model and exhausting GPU memory.
        if self._model2 is not None: return
        with self._load_lock:
            if self._model2 is not None: return
            from scripts.models import Model2ConvNeXt
            ckpt_path = PROJECT_ROOT / "models" / "model2_specialist" / "model2_production.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = Model2ConvNeXt(num_classes=9, pretrained=True)
            # Remap backbone keys
            remapped = {}
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("backbone.") and not k.startswith("backbone.model."):
                    remapped["backbone.model." + k[len("backbone."):]] = v
                else:
                    remapped[k] = v
            model.load_state_dict(remapped, strict=False)
            model = model.to(self.device).eval()
            self._model2 = model
            logger.info("  Model 2 loaded")

    def _lazy_load_efficientnet(self):
        if self._efficientnet is not None: return
        with self._load_lock:
            if self._efficientnet is not None: return
            self._do_lazy_load_efficientnet()

    def _do_lazy_load_efficientnet(self):
        import importlib.util, sys as _sys
        model_path = PROJECT_ROOT / "old_10class" / "app" / "model.py"
        config_path = PROJECT_ROOT / "old_10class" / "app" / "config.py"
        saved_app = _sys.modules.get("app")
        saved_app_config = _sys.modules.get("app.config")
        cfg_spec = importlib.util.spec_from_file_location("app.config", config_path)
        cfg_mod = importlib.util.module_from_spec(cfg_spec)
        app_mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("app", loader=None))
        app_mod.__path__ = [str(model_path.parent)]
        _sys.modules["app"] = app_mod
        _sys.modules["app.config"] = cfg_mod
        cfg_spec.loader.exec_module(cfg_mod)
        cfg_mod.NUM_CLASSES = 23; cfg_mod.NUM_CROPS = 4
        m_spec = importlib.util.spec_from_file_location("app.model", model_path)
        m_mod = importlib.util.module_from_spec(m_spec)
        _sys.modules["app.model"] = m_mod
        m_spec.loader.exec_module(m_mod)
        model = m_mod.PlantDiseaseModel()
        ckpt = torch.load(PROJECT_ROOT / "models" / "best_model.pt",
                           map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(self.device).eval()
        if saved_app is not None: _sys.modules["app"] = saved_app
        if saved_app_config is not None: _sys.modules["app.config"] = saved_app_config
        self._efficientnet = model
        logger.info("  EfficientNet loaded")

    def _lazy_load_dinov2(self):
        if self._dinov2_backbone is not None: return
        with self._load_lock:
            if self._dinov2_backbone is not None: return
            self._do_lazy_load_dinov2()

    def _do_lazy_load_dinov2(self):
        import timm
        backbone = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            pretrained=True, num_classes=0, img_size=224,
        ).to(self.device).eval()
        # Head + scaler from Section 4 artifacts
        from scripts.dinov2_probe.train_nonlinear_head import NonlinearHead
        head_path = (PROJECT_ROOT / "scripts" / "dinov2_probe" / "results" /
                      "dinov2_nonlinear_head_20260416_204427.pt")
        scaler_path = (PROJECT_ROOT / "scripts" / "dinov2_probe" / "results" /
                        "dinov2_nonlinear_head_scaler_20260416_204427.pkl")
        cfg_path = (PROJECT_ROOT / "scripts" / "dinov2_probe" / "results" /
                    "dinov2_nonlinear_head_config_20260416_204427.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        head = NonlinearHead(
            in_dim=cfg["feature_dim"], hidden_dims=cfg["hidden_dims"],
            num_classes=cfg["num_classes"], dropout=cfg["dropout"]
        )
        head.load_state_dict(torch.load(head_path, map_location=self.device,
                                            weights_only=True))
        head = head.to(self.device).eval()
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        self._dinov2_backbone = backbone
        self._dinov2_head = head
        self._dinov2_scaler = scaler
        logger.info("  DINOv2 head loaded")

    # --------------------------------------------------------------------
    # Per-signal inference
    # --------------------------------------------------------------------
    def _infer_model2(self, img_rgb: np.ndarray) -> np.ndarray:
        self._lazy_load_model2()
        tensor = preprocess_branch_a(img_rgb, M2_IMG_SIZE).to(self.device)
        with torch.no_grad():
            with torch.autocast(
                device_type=("cuda" if self.device == "cuda" else "cpu"),
                dtype=torch.bfloat16,
                enabled=(self.device == "cuda"),
            ):
                logits = self._model2(tensor)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        return probs.astype(np.float32)

    def _infer_efficientnet(self, img_rgb: np.ndarray) -> np.ndarray:
        self._lazy_load_efficientnet()
        tensor = preprocess_branch_b(img_rgb, EN_IMG_SIZE).to(self.device)
        with torch.no_grad():
            _crop, disease, _sev = self._efficientnet(tensor)
            probs_23 = torch.sigmoid(disease).cpu().numpy()[0]
        probs_9 = probs_23[self.en_index_map]
        return probs_9.astype(np.float32)

    def _infer_dinov2_head(self, img_rgb: np.ndarray) -> tuple:
        """Returns (probs_9, raw_feat_768).

        raw_feat_768 is the unscaled DINOv2 feature vector used by the
        downstream Mahalanobis OOD detector. Returning it explicitly
        (instead of stashing on self) avoids race conditions when the
        FastAPI server runs predict() concurrently in a thread pool —
        the singleton APINInference is shared across threads.
        """
        self._lazy_load_dinov2()
        tensor = preprocess_branch_a(img_rgb, DINO_IMG_SIZE).to(self.device)
        with torch.no_grad():
            # cls_mean aggregation: CLS + mean_patch concat = 768d
            full_out = self._dinov2_backbone.forward_features(tensor)  # (1, 261, 384)
            cls_token = full_out[:, 0, :]
            patch_tokens = full_out[:, 5:, :]  # skip 1 CLS + 4 register tokens
            mean_patch = patch_tokens.mean(dim=1)
            feat = torch.cat([cls_token, mean_patch], dim=1)
            feat_np = feat.cpu().numpy()
            raw_feat = feat_np[0].astype(np.float32)
            feat_scaled = self._dinov2_scaler.transform(feat_np).astype(np.float32)
            logits = self._dinov2_head(torch.from_numpy(feat_scaled).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs.astype(np.float32), raw_feat

    def _compute_pipeline_visualizations(self, img_rgb: np.ndarray) -> dict:
        """Compute per-stage image transformations for the Pipeline Inspector.

        Returns dict of base64-encoded PNGs (small ~300px wide) showing what
        the image actually became at each transforming pipeline stage:
          gate_zero_lap        — Laplacian edge response (sharpness signal)
          gate_zero_lab_l      — LAB-L luminance channel (exposure signal)
          gate_zero_leaf_mask  — HSV green-region mask (leaf coverage)
          preproc_lab_clahe    — Branch A: LAB-CLAHE @ 384 (Model 2/DINOv2/PSV input)
          preproc_rgb_clahe    — Branch B: RGB per-channel CLAHE @ 224 (EfficientNet input)

        Stages 1, 4, 5, 6, 7 don't transform the image so they aren't here —
        the frontend renders client-side composites for those (per-signal
        probabilities, gate weights, etc.).

        Total cost ~120ms; encoded PNGs are ~30-60KB each at 300px.
        """
        import io as _io
        import base64 as _b64
        from PIL import Image as _PIL
        out: dict = {}
        try:
            # Resize to a manageable display width while preserving aspect
            target_w = 320
            h0, w0 = img_rgb.shape[:2]
            tgt_h = max(1, int(h0 * (target_w / max(w0, 1))))
            small = cv2.resize(img_rgb, (target_w, tgt_h))

            def _encode(arr_uint8, mode="RGB"):
                buf = _io.BytesIO()
                _PIL.fromarray(arr_uint8, mode=mode).save(buf, format="PNG", optimize=True)
                return _b64.b64encode(buf.getvalue()).decode("ascii")

            # Stage 2.1 — Laplacian edge response (the actual sharpness check)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_norm = np.clip(np.abs(lap) * 4.0, 0, 255).astype(np.uint8)
            out["gate_zero_lap"] = _encode(lap_norm, mode="L")

            # Stage 2.2 — LAB L-channel (the actual exposure check)
            lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]  # 0-255
            out["gate_zero_lab_l"] = _encode(l_channel, mode="L")

            # Stage 2.3 — HSV green-region mask (the actual leaf-coverage check)
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            green_mask = cv2.inRange(hsv, (30, 30, 30), (95, 255, 255))
            # Overlay mask in green tint over greyscaled original for context
            grey_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            overlay = grey_rgb.copy()
            overlay[green_mask > 0] = (
                0.45 * grey_rgb[green_mask > 0]
                + 0.55 * np.array([47, 111, 62], dtype=np.float32)
            ).astype(np.uint8)
            out["gate_zero_leaf_mask"] = _encode(overlay, mode="RGB")

            # Stage 3.1 — Branch A: LAB-CLAHE @ M2_IMG_SIZE (visualised at 320 wide)
            lab_full = apply_lab_clahe(img_rgb)
            lab_resized = cv2.resize(lab_full, (target_w, tgt_h))
            out["preproc_lab_clahe"] = _encode(lab_resized, mode="RGB")

            # Stage 3.2 — Branch B: RGB-CLAHE
            rgb_full = apply_rgb_clahe(img_rgb)
            rgb_resized = cv2.resize(rgb_full, (target_w, tgt_h))
            out["preproc_rgb_clahe"] = _encode(rgb_resized, mode="RGB")

        except Exception as e:
            logger.warning(f"  Pipeline visualizations failed: {e}")
        return out

    def _generate_gradcam(self, img_rgb: np.ndarray,
                            gate_weights_array: np.ndarray,
                            predicted_class_idx: int) -> tuple:
        """Pick the signal with the highest gate weight on the predicted
        class and run GradCAM++ on its backbone (Gap 2 audit fix —
        Addition 1 from the design spec). Returns (b64_png, signal_name)
        or (None, None) on failure.

        Why gate-weighted selection: the heatmap a farmer sees should
        explain WHY the model decided what it did. If DINOv2's gate is
        0.75 for this prediction, Model 2's CAM is misleading. The signal
        that drove the decision is the one whose CAM should be shown.
        """
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            return None, None

        # Pick the signal with highest gate weight; tie-broken by signal
        # order (M2 > EN > PSV > DINOv2). PSV has no neural backbone for
        # CAM — fall back to next-highest neural signal in that case.
        n_sig = len(gate_weights_array)
        signal_order = (["model2", "efficientnet", "psv", "dinov2_head"]
                          if n_sig == 4
                          else ["model2", "efficientnet", "dinov2_head"])
        # Sort by gate weight descending
        ranked = sorted(zip(signal_order, gate_weights_array),
                          key=lambda x: -x[1])
        chosen = None
        for name, _w in ranked:
            if name == "psv":
                continue  # no neural backbone
            chosen = name
            break
        if chosen is None:
            return None, None

        try:
            import io as _io
            import base64 as _b64
            from PIL import Image as _PIL
            # Pick (model, target_layer, preprocess_size) per signal
            if chosen == "model2":
                self._lazy_load_model2()
                model = self._model2
                # Last conv stage of ConvNeXt for clean spatial attention.
                # Phase E.5c fix: the previous path was
                #   model.backbone.model.stages[3].layers[-1].depthwise_conv
                # which assumed a timm-style `.model` wrapper. The actual
                # backbone is HuggingFace's `DINOv3ConvNextModel`, which
                # exposes `.stages` directly with no `.model` indirection.
                # Verified structure:
                #   backbone.stages[3].layers[2] is the last DINOv3ConvNextLayer
                #   and has children {depthwise_conv, layer_norm,
                #                     pointwise_conv1, activation_fn,
                #                     pointwise_conv2, drop_path}.
                target = model.backbone.stages[3].layers[-1].depthwise_conv
                size = M2_IMG_SIZE
                tensor = preprocess_branch_a(img_rgb, size).to(self.device)
            elif chosen == "efficientnet":
                self._lazy_load_efficientnet()
                model = self._efficientnet
                # FPN's deepest fused output is the natural CAM target
                target = model.fpn.out_p3
                size = EN_IMG_SIZE
                tensor = preprocess_branch_b(img_rgb, size).to(self.device)
            elif chosen == "dinov2_head":
                self._lazy_load_dinov2()
                # GradCAM on a frozen DINOv2 ViT requires a transformer-aware
                # reshape. We use the last transformer block; pytorch_grad_cam
                # ships ViT support via a reshape_transform.
                model = self._dinov2_backbone
                target = model.blocks[-1].norm1
                size = DINO_IMG_SIZE
                tensor = preprocess_branch_a(img_rgb, size).to(self.device)
            else:
                return None, None

            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            # ViT reshape function (only for dinov2_head)
            reshape_transform = None
            if chosen == "dinov2_head":
                # DINOv2-Small with registers: 1 CLS + 4 register + N patches
                # Patch grid for 224 input at patch=14 = 16x16
                def _vit_reshape(t):
                    n_patches = 16 * 16
                    # t shape: (B, 1+4+N, dim)
                    patches = t[:, 5:5 + n_patches, :]
                    return patches.reshape(t.size(0), 16, 16, t.size(-1)).permute(0, 3, 1, 2)
                reshape_transform = _vit_reshape

            # Class targets: the MLP-final-prediction class index. For
            # signals with their own classification head (M2 has 9-class
            # softmax head; EfficientNet has multi-output disease head;
            # DINOv2 backbone alone has none) we pick the appropriate
            # output class. For DINOv2 backbone-only CAM we use the
            # average-pool-then-linear approach via wrapping into a small
            # adapter — but since we need the class-conditioned gradient
            # signal, fall back to a simpler approach: target the head's
            # logit for predicted_class_idx.
            #
            # For a clean cross-signal baseline, use Model 2 / EfficientNet
            # with their own heads; for DINOv2 we wrap the head.
            if chosen == "dinov2_head":
                # Build a tiny wrapper: backbone → CLS+mean → scaler → head.
                # Round-6 audit fix: the previous implementation used
                # `feat.detach().cpu().numpy()` then `torch.from_numpy(...)`
                # to apply the sklearn StandardScaler. That severs the
                # autograd graph — GradCAM's backward pass cannot reach
                # the backbone target layer, so the heatmap renders as a
                # uniform grey square. Apply the scaler as PyTorch tensor
                # arithmetic (mean/scale_ as buffers) to keep the graph
                # intact end-to-end.
                _scaler = self._dinov2_scaler
                _mean_t = torch.from_numpy(
                    np.asarray(_scaler.mean_, dtype=np.float32)
                ).to(self.device)
                _scale_t = torch.from_numpy(
                    np.asarray(_scaler.scale_, dtype=np.float32)
                ).to(self.device).clamp(min=1e-8)
                class _DinoCAMWrapper(nn.Module):
                    def __init__(self, backbone, head, mean_t, scale_t):
                        super().__init__()
                        self.backbone = backbone
                        self.head = head
                        self.register_buffer("mean_t", mean_t)
                        self.register_buffer("scale_t", scale_t)
                    def forward(self, x):
                        out = self.backbone.forward_features(x)
                        cls = out[:, 0, :]
                        patches = out[:, 5:, :]
                        mean_p = patches.mean(dim=1)
                        feat = torch.cat([cls, mean_p], dim=1)
                        # Tensor-arithmetic standardization keeps the
                        # autograd graph intact so GradCAM can backprop
                        # through to the backbone target layer.
                        scaled = (feat - self.mean_t) / self.scale_t
                        return self.head(scaled)
                model_for_cam = _DinoCAMWrapper(
                    self._dinov2_backbone, self._dinov2_head, _mean_t, _scale_t
                )
            elif chosen == "efficientnet":
                # EfficientNet returns (crop, disease, severity) — wrap to
                # return only disease logits at the right index slot.
                class _ENCAMWrapper(nn.Module):
                    def __init__(self, base, idx_map):
                        super().__init__()
                        self.base = base
                        self._idx_map = idx_map  # 9-dim mapping into 23-dim
                    def forward(self, x):
                        _crop, disease, _sev = self.base(x)
                        return disease[:, self._idx_map]
                model_for_cam = _ENCAMWrapper(model, self.en_index_map)
            else:
                model_for_cam = model

            with GradCAMPlusPlus(
                model=model_for_cam, target_layers=[target],
                reshape_transform=reshape_transform,
            ) as cam:
                targets = [ClassifierOutputTarget(predicted_class_idx)]
                grayscale = cam(input_tensor=tensor, targets=targets)[0]

            # Overlay onto resized RGB
            rgb_resized = cv2.resize(img_rgb, (size, size))
            rgb_norm = rgb_resized.astype(np.float32) / 255.0
            overlay = show_cam_on_image(rgb_norm, grayscale, use_rgb=True,
                                          image_weight=0.55)
            buf = _io.BytesIO()
            _PIL.fromarray(overlay).save(buf, format="PNG", optimize=True)
            return _b64.b64encode(buf.getvalue()).decode("ascii"), chosen
        except Exception as e:
            logger.warning(f"  GradCAM++ failed on signal '{chosen}': {e}")
            return None, chosen

    def _compute_ood_distance(self, predicted_class_idx: int,
                                dinov2_feat: Optional[np.ndarray]) -> tuple:
        """Mahalanobis distance from the supplied DINOv2 feature vector to
        the NEAREST class prototype, in per-class whitened space.

        Originally this used only the predicted-class prototype. For genuinely
        OOD images the MLP's argmax is essentially random — so distance to
        the wrong prototype was an unreliable OOD signal. Computing min over
        ALL class prototypes makes OOD = "feature is far from EVERY known
        class". This is the standard Mahalanobis-OOD formulation.

        Returns (distance, is_ood, distance_bucket).

        If the OOD detector or feature is unavailable, returns (0.0, False,
        'near') — fail-open so OOD never blocks legitimate predictions.
        """
        # Per-class distances are returned as the 4th tuple element so the
        # call is safe under concurrent requests (no instance-state stash).
        per_class: dict = {}
        if self._ood_detector is None or dinov2_feat is None:
            return 0.0, False, "near", per_class
        # Compute distance to every prototype; OOD if min is above threshold
        min_dist = float("inf")
        for cls_idx in self._ood_detector["class_means"].keys():
            mean = self._ood_detector["class_means"][cls_idx]
            cov_inv = self._ood_detector["class_cov_inv"][cls_idx]
            diff = dinov2_feat - mean
            m2 = float(diff @ cov_inv @ diff)
            # Numerical safety: catastrophic cancellation when cov_inv has
            # near-zero eigenvalues can produce small negative values
            m2 = max(m2, 0.0)
            d = float(np.sqrt(m2))
            # Record per-class distance
            cls_name = (self.class_order[int(cls_idx)]
                         if 0 <= int(cls_idx) < len(self.class_order)
                         else f"class_{cls_idx}")
            per_class[cls_name] = round(d, 4)
            if d < min_dist:
                min_dist = d
        if min_dist == float("inf"):
            return 0.0, False, "near", per_class
        dist = min_dist
        thresh = self._ood_detector["threshold"]
        is_ood = dist > thresh
        # Bucket by ratio of dist:threshold
        ratio = dist / max(thresh, 1e-6)
        if ratio < 0.6:
            bucket = "near"
        elif ratio < 1.0:
            bucket = "moderate"
        elif ratio < 1.5:
            bucket = "far"
        else:
            bucket = "extreme"
        return dist, is_ood, bucket, per_class

    def _compute_psv_feature_firing(self, raw_features: dict,
                                       predicted_class: str,
                                       top_k: int = 8) -> list:
        """For the predicted class, return top-k PSV features (positive and
        negative contributors) using Section 3A multi-class LR weights.

        Each entry: {name, value, coef, contribution, sign}
            contribution = coef × value (the LR dot-product term)

        Returns [] if Section 3A multi-class artifact is missing or the
        predicted class wasn't fit by the supervised path.
        """
        try:
            from scripts.psv.disease_scores import (
                _load_multiclass_weights, _calibrate_features,
            )
        except Exception:
            return []
        weights = _load_multiclass_weights()
        if weights is None or predicted_class not in weights:
            return []
        names, coefs = weights[predicted_class]
        # Calibrate first (matches how disease_scores.py scores)
        try:
            if self.psv_calibration is not None:
                calibrated = _calibrate_features(raw_features, self.psv_calibration)
            else:
                calibrated = raw_features
        except Exception:
            calibrated = raw_features
        entries = []
        for name, coef in zip(names, coefs):
            val = float(calibrated.get(name, 0.0))
            contrib = float(coef) * val
            entries.append({
                "name": name,
                "value": round(val, 4),
                "coef": round(float(coef), 4),
                "contribution": round(contrib, 4),
                "sign": "+" if contrib >= 0 else "-",
            })
        # Sort by |contribution| descending, take top_k
        entries.sort(key=lambda e: abs(e["contribution"]), reverse=True)
        return entries[:top_k]

    def _compute_signal_contributions(self, signal_preds_vecs: list,
                                         gate_weights: np.ndarray,
                                         predicted_class_idx: int) -> dict:
        """SHAP-style per-signal contribution toward the predicted class.

        Each signal s contributes: gate_weight[s] × reliability[s][c] ×
        signal_probability[s][c] (unnormalised; normalised in frontend).

        Returns dict keyed by signal name (e.g. "model2", "efficientnet",
        "psv", "dinov2_head") with {raw, normalised_pct} per signal.
        """
        sig_names_4 = ["model2", "efficientnet", "psv", "dinov2_head"]
        sig_names_3 = ["model2", "efficientnet", "dinov2_head"]
        n_sig = len(gate_weights)
        names = sig_names_4 if n_sig == 4 else sig_names_3
        raw_contribs = []
        cls_idx = int(predicted_class_idx)
        for s, vec in enumerate(signal_preds_vecs):
            if vec is None:
                raw_contribs.append(0.0)
                continue
            w = float(gate_weights[s])
            r = float(self.reliability_matrix[s][cls_idx])
            p = float(vec[cls_idx])
            raw_contribs.append(w * r * p)
        total = sum(raw_contribs) or 1e-9
        return {
            names[i]: {
                "raw": round(raw_contribs[i], 4),
                "normalised_pct": round(raw_contribs[i] / total * 100.0, 2),
            } for i in range(len(names)) if i < len(raw_contribs)
        }

    def _infer_psv(self, img_rgb: np.ndarray) -> tuple:
        """Returns (probs_9_or_None, disease_stage_or_None, raw_features_or_None).

        Round-1 audit fix: previously stashed disease_stage and raw_features
        on `self` as side effects, which was unsafe under FastAPI's thread-
        pool executor — two concurrent requests could overwrite each other.
        Now returns all three explicitly so the caller scopes them to the
        local stack frame.
        """
        if self.psv_calibration is None:
            return None, None, None
        try:
            result = self.psv_extract(img_rgb)
            scores_dict = self.psv_scores(result.features, self.psv_calibration)
            vec = np.array([float(scores_dict.get(c, 0.0)) for c in self.class_order],
                            dtype=np.float32)
            stage = float(result.features.get("F08_disease_stage_index", 0.0))
            raw = dict(result.features)
            return vec, stage, raw
        except Exception as e:
            logger.warning(f"  PSV failed: {e}")
            return None, None, None

    # --------------------------------------------------------------------
    # Conflict classification
    # --------------------------------------------------------------------
    def _classify_conflict(self, signal_preds: dict) -> str:
        """signal_preds maps signal_name -> argmax class index. Returns
        A / B1 / B2 / B3 / C1 / C2 / D."""
        preds = list(signal_preds.values())
        n = len(preds)
        unique = set(preds)
        if len(unique) == 1:
            return "A"
        if len(unique) == 2:
            # Majority analysis
            counts = {p: preds.count(p) for p in unique}
            if max(counts.values()) == n - 1:
                # 3-of-4 (or 2-of-3) agreement
                minority = [p for p, c in counts.items() if c == 1][0]
                # Find which signal disagreed
                disagreer = [name for name, p in signal_preds.items() if p == minority][0]
                return {"model2": "B1", "psv": "B2"}.get(disagreer, "B3")
            # 2/2 split (only possible if n=4)
            return "C2"
        if len(unique) >= 3:
            return "D"
        return "B3"

    # --------------------------------------------------------------------
    # Stacking MLP evaluation
    # --------------------------------------------------------------------
    def _run_stacking_mlp(self, signal_preds_list: list) -> tuple:
        """signal_preds_list: ordered list of 9-dim numpy arrays, one per
        signal in the order [S1, S2, (S3,) S4]. PSV slot may be None —
        use zeros in that case.

        Returns (final_probs (9,), gate_weights (n_signals,))
        """
        x = np.zeros(self.n_signals * 9, dtype=np.float32)
        for s, vec in enumerate(signal_preds_list):
            if vec is None:
                vec = np.zeros(9, dtype=np.float32)
            x[s * 9: (s + 1) * 9] = vec

        # Apply reliability matrix modulation
        for s in range(self.n_signals):
            x[s * 9: (s + 1) * 9] *= self.reliability_matrix[s]

        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        # BatchNorm expects batch>1. Use batch 2 (duplicate) then take first.
        x_t_dup = torch.cat([x_t, x_t], dim=0)
        # Critical correctness check: BN behaves differently in train vs eval.
        # Stacking MLP was set to eval() at load time. Assert it.
        assert not self.stacking_mlp.training, (
            "Stacking MLP must be in eval() mode for correct BN inference"
        )
        with torch.no_grad():
            logits, gate_w = self.stacking_mlp(x_t_dup, return_gate_weights=True)
        logits = logits[0:1]
        gate_w = gate_w[0].cpu().numpy()

        # CRITICAL FIX (audit finding #1): apply per-class temperature scaling
        # BEFORE softmax. Conformal thresholds in apin_calibration.json were
        # fitted on temperature-scaled probabilities; without applying T here
        # the thresholds are meaningless.
        if hasattr(self, "per_class_temps") and self.per_class_temps is not None:
            # logits shape (1, 9); per_class_temps shape (9,)
            t = torch.from_numpy(self.per_class_temps).to(self.device)
            logits = logits / t.unsqueeze(0).clamp(min=1e-3)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs.astype(np.float32), gate_w

    # --------------------------------------------------------------------
    # Adaptive threshold scaling
    # --------------------------------------------------------------------
    def _adaptive_multiplier(self, gate_z, conflict_type, distance_bucket):
        """Compose three multipliers from apin_calibration.json into a single
        scaling factor on confidence thresholds.

          quality:        higher when image quality drops (>1.0 means require
                           more confidence to emit a confident tier)
          agreement:      lower when all signals agree (Type A is 0.85), higher
                           on conflicts (Type D is 1.25)
          source_distance: higher as Mahalanobis distance grows beyond cal-set

        The product is multiplied into TIER_1*_CONFIDENCE thresholds before
        comparing top_p, so a low-quality + conflicting image needs higher raw
        probability to qualify for 1A/1B than a clean unanimous prediction.
        """
        if not self.adaptive_multipliers:
            return 1.0
        # quality bucket
        q_flags = gate_z.quality_flags or {}
        q = "high"
        if q_flags.get("mild_blur") and (
            gate_z.lab_L_mean < 35 or gate_z.lab_L_mean > 85
        ):
            q = "low_quality_combo"
        elif q_flags.get("mild_blur"):
            q = "mild_blur"
        elif gate_z.lab_L_mean < 35 or gate_z.lab_L_mean > 85:
            q = "mild_exposure_issue"
        q_mul = self.adaptive_multipliers.get("quality", {}).get(q, 1.0)
        # agreement bucket — conflict_type letter prefix maps directly
        a_mul = self.adaptive_multipliers.get("agreement", {}).get(
            conflict_type, 1.0
        )
        # source distance bucket
        s_mul = self.adaptive_multipliers.get("source_distance", {}).get(
            distance_bucket, 1.0
        )
        return float(q_mul * a_mul * s_mul)

    # --------------------------------------------------------------------
    # Output tier determination
    # --------------------------------------------------------------------
    def _determine_tier(self, probs, conflict_type, gate_z,
                         ood_flag, signal_preds_by_name,
                         entropy_norm: float, mult: float,
                         signal_preds_vecs: list = None,
                         disease_stage_index: float = None) -> tuple:
        """Returns (tier_name, diagnosis_class_name, confidence, trace).

        `trace` is a list of dicts describing every ladder condition that
        was evaluated — including the ones that didn't match — so the UI
        can render a "Why this tier?" view. Each entry has:
          {step, name, check, value, threshold, passed, tier_if_taken}
        """
        argmax = int(probs.argmax())
        top_p = float(probs[argmax])
        sorted_idx = np.argsort(-probs)
        second_p = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        gap = top_p - second_p
        diagnosis = self.class_order[argmax]
        trace: list = []

        def _log(name, check, value, threshold, passed, tier_if_taken):
            trace.append({
                "step": len(trace) + 1,
                "name": name, "check": check,
                "value": value, "threshold": threshold,
                "passed": bool(passed),
                "tier_if_taken": tier_if_taken,
            })

        # ── Tier 5: critical urgency override ────────────────────────────
        if signal_preds_vecs is not None:
            for crit_class in TIER_5_CRITICAL_DISEASES:
                if crit_class not in self.class_order:
                    continue
                cls_idx = self.class_order.index(crit_class)
                signal_high = [
                    v for v in signal_preds_vecs
                    if v is not None and float(v[cls_idx]) >= TIER_5_SIGNAL_CONFIDENCE
                ]
                n_signals_agree = sum(
                    1 for v in signal_preds_vecs
                    if v is not None and int(v.argmax()) == cls_idx
                )
                stage_bonus = (1 if (disease_stage_index is not None
                                      and disease_stage_index >= DISEASE_STAGE_MID)
                                else 0)
                required_agree = max(1, TIER_5_MIN_AGREEING_SIGNALS - stage_bonus)
                fired = bool(signal_high) and n_signals_agree >= required_agree
                if fired:
                    _log(f"Tier-5 critical ({crit_class})",
                          f"any signal ≥ {TIER_5_SIGNAL_CONFIDENCE} AND "
                          f"≥{required_agree} signals agree",
                          f"max_signal={max([float(v[cls_idx]) for v in signal_preds_vecs if v is not None] or [0]):.3f}, "
                          f"n_agree={n_signals_agree}",
                          f"≥{TIER_5_SIGNAL_CONFIDENCE} & ≥{required_agree}",
                          True, "5")
                    return "5", crit_class, max(top_p, float(signal_high[0][cls_idx])), trace
            _log("Tier-5 critical", "any critical-class signal ≥0.85 AND ≥2 agree",
                  "no critical class met threshold", "—", False, "5")

        # OOD override
        if ood_flag:
            if top_p < TIER_4A_CONFIDENCE_MAX:
                _log("OOD low-conf", "is_ood AND top_p < 0.40",
                      f"{top_p:.3f}", "< 0.40", True, "4A")
                return "4A", diagnosis, top_p, trace
            _log("OOD moderate-conf", "is_ood AND top_p ≥ 0.40",
                  f"{top_p:.3f}", "≥ 0.40", True, "4B")
            return "4B", diagnosis, top_p, trace
        _log("OOD check", "Mahalanobis in-distribution",
              "is_ood=False", "threshold not exceeded", False, "4A/4B")

        # Hard quality + weak confidence → 3B
        quality_borderline = (
            (gate_z.blur_score < BLUR_MILD)
            or (gate_z.lab_L_mean < 35)
            or (gate_z.lab_L_mean > 85)
        )
        if quality_borderline and top_p < 0.45:
            _log("Quality+weak-conf fallback", "borderline quality AND top_p<0.45",
                  f"borderline={quality_borderline}, top_p={top_p:.3f}",
                  "quality=False OR top_p≥0.45", True, "3B")
            return "3B", diagnosis, top_p, trace
        _log("Quality+weak-conf fallback", "borderline quality AND top_p<0.45",
              f"borderline={quality_borderline}, top_p={top_p:.3f}",
              "NOT triggered", False, "3B")

        # Degraded-mode floor
        n_signals_present = len(signal_preds_by_name)
        if n_signals_present <= 1:
            _log("Degraded-mode floor", "≤1 signals present",
                  f"{n_signals_present} signals", "> 1 required", True, "3B")
            return "3B", diagnosis, top_p, trace
        _log("Degraded-mode floor", "≤1 signals present",
              f"{n_signals_present} signals", "≥ 2 required", False, "3B")

        # 3A for confident healthy (no conflict)
        is_healthy = diagnosis in ("okra_healthy", "brassica_healthy")
        if is_healthy and top_p >= TIER_3A_HEALTHY_MIN and conflict_type not in ("C1", "C2", "D"):
            _log("3A confident healthy", "healthy AND top_p≥0.50 AND no C/D conflict",
                  f"diagnosis={diagnosis}, top_p={top_p:.3f}, conflict={conflict_type}",
                  "healthy & ≥0.50 & non-C/D", True, "3A")
            return "3A", diagnosis, top_p, trace

        # 3C for conflicted healthy
        if is_healthy and conflict_type in ("C1", "C2", "D"):
            _log("3C conflicted healthy", "healthy AND conflict in C/D",
                  f"diagnosis={diagnosis}, conflict={conflict_type}",
                  "C1/C2/D required", True, "3C")
            return "3C", diagnosis, top_p, trace

        # 3A early-disease (low stage + moderate confidence)
        if (disease_stage_index is not None and disease_stage_index < DISEASE_STAGE_EARLY
                and not is_healthy and top_p < 0.65):
            _log("3A early-disease", "low PSV stage AND disease AND top_p<0.65",
                  f"stage={disease_stage_index:.3f}, top_p={top_p:.3f}",
                  f"stage<{DISEASE_STAGE_EARLY} & top_p<0.65", True, "3A")
            return "3A", diagnosis, top_p, trace

        # Agreement counts
        n_agree = sum(1 for p in signal_preds_by_name.values() if p == argmax)
        total_signals = len(signal_preds_by_name)
        majority_cut = max(2, total_signals - 1)

        # Adaptive-scaled thresholds
        c1a = TIER_1A_CONFIDENCE * mult
        c1b = TIER_1B_CONFIDENCE * mult
        c1c = TIER_1C_CONFIDENCE * mult
        c2a = TIER_2A_CONFIDENCE * mult
        c2b = TIER_2B_CONFIDENCE * mult
        c2c_above = min(TIER_2C_ABOVE * mult, 0.5)

        # 1A
        cond_1a = (top_p >= c1a and entropy_norm <= TIER_1A_ENTROPY_MAX
                    and n_agree == total_signals)
        _log("1A confident", "top_p≥c1a AND entropy≤0.20 AND ALL signals agree",
              f"top_p={top_p:.3f}, entropy={entropy_norm:.3f}, agree={n_agree}/{total_signals}",
              f"≥{c1a:.3f} & ≤0.20 & {total_signals}/{total_signals}",
              cond_1a, "1A")
        if cond_1a:
            return "1A", diagnosis, top_p, trace

        # 1B
        cond_1b = (top_p >= c1b and entropy_norm <= TIER_1B_ENTROPY_MAX
                    and n_agree >= majority_cut)
        _log("1B model-confirmed", "top_p≥c1b AND entropy≤0.40 AND majority agree",
              f"top_p={top_p:.3f}, entropy={entropy_norm:.3f}, agree={n_agree}/{total_signals}",
              f"≥{c1b:.3f} & ≤0.40 & ≥{majority_cut}/{total_signals}",
              cond_1b, "1B")
        if cond_1b:
            return "1B", diagnosis, top_p, trace

        # 1C
        cond_1c = top_p >= c1c and gap >= TIER_1C_GAP_MIN
        _log("1C strong-leader", "top_p≥c1c AND gap≥0.15",
              f"top_p={top_p:.3f}, gap={gap:.3f}",
              f"≥{c1c:.3f} & ≥{TIER_1C_GAP_MIN}", cond_1c, "1C")
        if cond_1c:
            return "1C", diagnosis, top_p, trace

        # 2B (checked before 2A because 2B is stricter)
        cond_2b = (top_p >= c2b and second_p >= TIER_2B_SECOND_MIN
                    and gap < TIER_2B_GAP_MAX)
        _log("2B equal-pair", "top_p≥c2b AND second≥0.25 AND gap<0.10",
              f"top_p={top_p:.3f}, second={second_p:.3f}, gap={gap:.3f}",
              f"≥{c2b:.3f} & ≥0.25 & <0.10", cond_2b, "2B")
        if cond_2b:
            return "2B", diagnosis, top_p, trace

        # 2A
        cond_2a = (top_p >= c2a and second_p >= TIER_2A_SECOND_MIN
                    and gap < TIER_2A_GAP_MAX)
        _log("2A probable+second", "top_p≥c2a AND second≥0.20 AND gap<0.20",
              f"top_p={top_p:.3f}, second={second_p:.3f}, gap={gap:.3f}",
              f"≥{c2a:.3f} & ≥0.20 & <0.20", cond_2a, "2A")
        if cond_2a:
            return "2A", diagnosis, top_p, trace

        # 2C
        n_above = int((probs >= c2c_above).sum())
        cond_2c = n_above >= 3
        _log("2C three-candidates", f"≥3 classes above {c2c_above:.3f}",
              f"n_above={n_above}", "≥ 3", cond_2c, "2C")
        if cond_2c:
            return "2C", diagnosis, top_p, trace

        # 3C conflicted non-healthy
        cond_3c = conflict_type in ("C1", "C2", "D")
        _log("3C atypical", "conflict in C1/C2/D",
              f"conflict={conflict_type}", "C1/C2/D", cond_3c, "3C")
        if cond_3c:
            return "3C", diagnosis, top_p, trace

        # 3B catch-all
        _log("3B catch-all", "no confident tier matched — fall through",
              "all above failed", "N/A", True, "3B")
        return "3B", diagnosis, top_p, trace

    def _apply_cold_start_downgrade(self, tier: str, diagnosis: str) -> tuple:
        """Per Decision 14: during cold-start, downgrade tier by one level for
        classes with documented field-photo failures or low reliability scores.

        Failure classes (constants.FAILURE_CLASSES): brassica_black_rot,
        okra_cercospora — Model 2 catastrophically misses these in the field.
        Additional reliability-based downgrades: okra_enation (0.86 reliability,
        lowest in the 4-signal matrix).

        Returns (adjusted_tier, was_downgraded).
        """
        if not self.cold_start_active:
            return tier, False
        if diagnosis not in COLD_START_DOWNGRADE_CLASSES:
            return tier, False
        # Tier 5 (urgent override) is never downgraded — that's the whole
        # point of the override.
        if tier == "5":
            return tier, False
        downgrade_map = {
            "1A": "1B", "1B": "1C", "1C": "2A",
            "2A": "2B", "2B": "2C",
        }
        if tier in downgrade_map:
            return downgrade_map[tier], True
        return tier, False

    # --------------------------------------------------------------------
    # Output message composition
    # --------------------------------------------------------------------
    def _compose_output(self, result: APINResult):
        cls = result.diagnosis or ""
        # Friendly class name
        friendly = cls.replace("_", " ").title() if cls else ""
        tier = result.tier

        if tier == "5":
            # Critical-urgency override: a single signal alarmed at ≥0.85 on
            # a rapid-spread disease. Override the normal diagnosis message
            # with an URGENT advisory and force a treatment recommendation.
            result.output_message = (
                f"⚠ URGENT: features consistent with {friendly}. This "
                f"disease spreads rapidly under current crop conditions — "
                f"begin treatment within 24 hours and isolate affected "
                f"plants immediately. If uncertain, consult an agricultural "
                f"extension officer today."
            )
        elif tier == "1A":
            result.output_message = f"Diagnosis: {friendly} — confidence high."
        elif tier == "1B":
            result.output_message = (
                f"Diagnosis: {friendly} — confidence moderate. Pathology "
                f"verification inconclusive; monitor response to treatment."
            )
        elif tier == "1C":
            result.output_message = (
                f"Most likely: {friendly}. Confidence: moderate."
            )
        elif tier == "2A":
            result.output_message = f"Most likely: {friendly}, possible alternative."
        elif tier == "2B":
            result.output_message = f"Features consistent with multiple diseases."
        elif tier == "2C":
            result.output_message = (
                "Symptoms match several possible diseases. Please photograph "
                "again in 3-5 days when symptoms are more developed."
            )
        elif tier == "3A":
            result.output_message = (
                "No visible disease symptoms currently. Monitor in 7 days."
            )
        elif tier == "3B":
            result.output_message = (
                f"Best guess: {friendly} (image quality limited diagnosis). "
                f"Please retake with better lighting and focus."
            )
        elif tier == "3C":
            result.output_message = (
                f"Predicted: {friendly}, but presentation is atypical. "
                f"May be an unusual strain or early stage."
            )
        elif tier == "4A":
            result.output_message = (
                "This image does not match any disease in our training data. "
                "Best guess: " + friendly + ". Please contact your agricultural "
                "extension officer."
            )
        elif tier == "4B":
            result.output_message = (
                f"This may be {friendly} but presentation differs significantly "
                f"from known cases."
            )

        # Treatment / prevention from diagnosis_lookup.json.
        #
        # Schema (per all 9 entries): treatment is a dict
        #   {"mild": str, "moderate": str, "severe": str},
        # prevention is a plain str.
        #
        # The previous implementation called `"\n".join(entry.get("treatment", []))`
        # which iterated dict keys and produced "mild\nmoderate\nsevere"; and
        # `"\n".join(entry.get("prevention", ""))` which iterated the string
        # character-by-character. Both silently corrupted every farmer-facing
        # treatment/monitoring message. (Round 5 audit, CRITICAL.)
        #
        # APIN does not currently expose a measured severity index, so we
        # derive a rough severity bucket from the calibrated confidence:
        #   top_p >= 0.85 → severe (highly confident strong-presentation)
        #   top_p >= 0.55 → moderate (default for confident tiers 1A-1C)
        #   else          → mild
        # This is a reasonable approximation pending a future severity head.
        if cls in self.diag_db:
            entry = self.diag_db[cls]
            tx = entry.get("treatment", "")
            if isinstance(tx, dict):
                top_p = float(result.confidence or 0.0)
                if top_p >= 0.85:
                    severity_key = "severe"
                elif top_p >= 0.55:
                    severity_key = "moderate"
                else:
                    severity_key = "mild"
                tx_text = (tx.get(severity_key)
                            or tx.get("moderate")
                            or "\n".join(tx.values()))
            elif isinstance(tx, list):
                tx_text = "\n".join(tx)
            else:
                tx_text = str(tx) if tx else None

            prev = entry.get("prevention", "")
            if isinstance(prev, list):
                prev_text = "\n".join(prev)
            elif isinstance(prev, str):
                prev_text = prev or None
            else:
                prev_text = None

            # Treatment is shown for confident tiers AND tier 5 (urgent
            # advisory always carries the treatment text).
            result.treatment_recommendation = (
                tx_text if tier in ("1A", "1B", "1C", "5") else None
            )
            result.monitoring_guidance = (
                prev_text if tier.startswith("3") else None
            )

        if tier.startswith("2"):
            result.differential_guidance = (
                "Compare the leaf against the top candidate diseases' "
                "distinguishing features (margin shape, vein pattern, spot "
                "regularity)."
            )

        if tier == "3B":
            result.retake_guidance = (
                "For better diagnosis: fill the frame with one leaf, use "
                "diffuse natural light, avoid harsh shadows."
            )

        # Aleatoric vs epistemic differentiated guidance
        # (architecture Section 8.5/8.6: high aleatoric → retake; high
        # epistemic → contact agronomist; both high → both messages)
        ALEATORIC_HIGH = 0.55     # entropy already in [0,1]
        EPISTEMIC_HIGH = 0.20     # std of per-signal top probs
        if (result.uncertainty_aleatoric >= ALEATORIC_HIGH
                and tier not in ("1A", "3A")):
            extra_retake = (
                "High aleatoric uncertainty — the image itself is ambiguous "
                "(blur, glare, partial leaf, or atypical angle). Retake under "
                "softer light, fill the frame, and avoid specular highlights."
            )
            result.retake_guidance = (
                (result.retake_guidance + " " + extra_retake)
                if result.retake_guidance else extra_retake
            )
        if (result.uncertainty_epistemic >= EPISTEMIC_HIGH
                and tier not in ("1A",)):
            agronomist_note = (
                "Signals disagree substantially (high epistemic uncertainty). "
                "Recommend confirmation by a local agricultural extension "
                "officer before applying treatment."
            )
            if result.differential_guidance:
                result.differential_guidance += " " + agronomist_note
            else:
                result.differential_guidance = agronomist_note

    # --------------------------------------------------------------------
    # Main predict
    # --------------------------------------------------------------------
    def predict(self, img_rgb: np.ndarray) -> APINResult:
        t_start = time.time()
        result = APINResult()
        # Layer 0
        gate_z = run_gate_zero(img_rgb)
        result.quality_flags = dict(gate_z.quality_flags)

        if gate_z.hard_reject:
            result.tier = "4A"  # treat as OOD
            result.output_message = gate_z.retake_reason or "Image unsuitable."
            result.retake_guidance = gate_z.retake_reason
            result.is_ood = True
            result.processing_time_ms = (time.time() - t_start) * 1000
            return result

        # Layers 2-4: run neural signals
        signal_preds_vecs = []       # list of 9-dim arrays (None if failed)
        signal_preds_by_name = {}    # name -> argmax class idx (only successful)
        failed_signals = []

        # Signal 1 - Model 2
        try:
            s1 = self._infer_model2(img_rgb)
            signal_preds_vecs.append(s1)
            signal_preds_by_name["model2"] = int(s1.argmax())
        except Exception as e:
            logger.warning(f"Model 2 failed: {e}")
            failed_signals.append("model2")
            signal_preds_vecs.append(None)

        # Signal 2 - EfficientNet
        try:
            s2 = self._infer_efficientnet(img_rgb)
            signal_preds_vecs.append(s2)
            signal_preds_by_name["efficientnet"] = int(s2.argmax())
        except Exception as e:
            logger.warning(f"EfficientNet failed: {e}")
            failed_signals.append("efficientnet")
            signal_preds_vecs.append(None)

        # Signal 3 - PSV (only if 4-signal mode and calibration available).
        # _infer_psv returns (probs, disease_stage, raw_features) — all
        # three scoped to this call frame for thread safety.
        psv_disease_stage = None
        psv_raw_features = None
        if self.use_psv:
            s3, psv_disease_stage, psv_raw_features = self._infer_psv(img_rgb)
            signal_preds_vecs.append(s3)
            if s3 is not None:
                signal_preds_by_name["psv"] = int(s3.argmax())
            else:
                failed_signals.append("psv")

        # Signal 4 - DINOv2 head (also returns raw 768-dim feature for OOD)
        dinov2_raw_feat = None
        try:
            s4, dinov2_raw_feat = self._infer_dinov2_head(img_rgb)
            signal_preds_vecs.append(s4)
            signal_preds_by_name["dinov2_head"] = int(s4.argmax())
        except Exception as e:
            logger.warning(f"DINOv2 head failed: {e}")
            failed_signals.append("dinov2_head")
            signal_preds_vecs.append(None)

        result.failed_signals = failed_signals

        # Non-recoverable: Model 2 failed
        if "model2" in failed_signals:
            result.tier = "4A"
            result.output_message = "Primary model unavailable — service error."
            result.processing_time_ms = (time.time() - t_start) * 1000
            return result

        # Conflict detection
        conflict_type = self._classify_conflict(signal_preds_by_name)
        result.conflict_type = conflict_type

        # Layer 5: stacking MLP
        probs, gate_w = self._run_stacking_mlp(signal_preds_vecs)

        # Record per-signal info
        for name, argmax in signal_preds_by_name.items():
            vec = signal_preds_vecs[
                ["model2", "efficientnet", "psv", "dinov2_head"].index(name)
                if self.use_psv else ["model2", "efficientnet", "dinov2_head"].index(name)
            ]
            result.signal_predictions[name] = {
                "argmax": self.class_order[argmax],
                "top_prob": float(vec[argmax]),
            }
        gate_names = (["S1_M2", "S2_EN", "S3_PSV", "S4_DINOv2"] if self.use_psv
                      else ["S1_M2", "S2_EN", "S4_DINOv2"])
        result.gate_weights = {name: float(w) for name, w in zip(gate_names, gate_w)}

        # All class probabilities
        result.all_class_probabilities = {
            c: round(float(p), 6) for c, p in zip(self.class_order, probs)
        }

        # Layer 6: uncertainty decomposition
        # Aleatoric: entropy of MLP output
        eps = 1e-12
        entropy = float(-(probs * np.log(probs + eps)).sum())
        max_entropy = float(np.log(self.num_classes))
        result.uncertainty_aleatoric = round(entropy / max_entropy, 6)

        # Epistemic: variance of per-signal argmax probabilities
        per_signal_top_probs = [float(v.max()) for v in signal_preds_vecs if v is not None]
        if len(per_signal_top_probs) > 1:
            result.uncertainty_epistemic = round(float(np.std(per_signal_top_probs)), 6)
        else:
            result.uncertainty_epistemic = 0.0

        # Layer 7: OOD detection — Mahalanobis to NEAREST class prototype on
        # DINOv2 features. Returns per-class dict explicitly (not via
        # instance state) for thread safety under concurrent requests.
        argmax_for_ood = int(probs.argmax())
        ood_distance, ood_flag, distance_bucket, per_class_mahal = \
            self._compute_ood_distance(argmax_for_ood, dinov2_raw_feat)
        result.is_ood = bool(ood_flag)
        result.mahalanobis_distance = float(ood_distance)

        # Adaptive threshold multiplier (quality x agreement x source distance)
        adaptive_mul = self._adaptive_multiplier(
            gate_z, conflict_type, distance_bucket
        )

        # Disease stage index from PSV F08 (used for 3A early-stage routing
        # AND Tier 5 late-stage escalation). Local, from the _infer_psv call
        # above — no instance stash.
        disease_stage = psv_disease_stage

        tier, diagnosis, confidence, decision_trace = self._determine_tier(
            probs, conflict_type, gate_z, ood_flag, signal_preds_by_name,
            entropy_norm=result.uncertainty_aleatoric, mult=adaptive_mul,
            signal_preds_vecs=signal_preds_vecs,
            disease_stage_index=disease_stage,
        )
        result.decision_trace = decision_trace
        # Surface disease stage for downstream consumers
        if disease_stage is not None:
            result.quality_flags["disease_stage_index"] = round(float(disease_stage), 4)

        # Cold-start downgrade
        tier_adj, downgraded = self._apply_cold_start_downgrade(tier, diagnosis)
        result.tier = tier_adj
        result.cold_start_tier_downgraded = downgraded
        result.diagnosis = diagnosis
        result.confidence = float(confidence)
        # Round-2 audit fix (CRITICAL): when cold-start downgrades the tier
        # AFTER _determine_tier has finished, the original ladder entry
        # marked tier_if_taken=tier (e.g. "1A") no longer matches the final
        # result.tier (e.g. "1B"), so the UI's "Why this tier?" panel would
        # show NO row highlighted. Append a final downgrade entry so the
        # taken row is unambiguous.
        if downgraded:
            result.decision_trace.append({
                "step": len(result.decision_trace) + 1,
                "name": "Cold-start downgrade (Decision 14)",
                "check": (f"diagnosis '{diagnosis}' is in COLD_START_DOWNGRADE_CLASSES "
                           "AND cold_start_active=True"),
                "value": f"original tier {tier} → adjusted {tier_adj}",
                "threshold": "downgrade by one level",
                "passed": True,
                "tier_if_taken": tier_adj,
            })

        # Conformal prediction set: include class c if probs[c] >= 1 - q[c].
        # Adaptive multiplier is intentionally NOT applied here — conformal
        # sets carry their own coverage guarantee fitted on the calibration
        # split. Multiplying (1-q) by a multiplier inverts the intuition: at
        # high mult (extreme OOD) the cutoff (1-q)*1.5 grows for low-q classes
        # but shrinks the inclusion-set width for high-q classes, mixing
        # conservatism with un-conservatism in a single transform. The
        # adaptive multiplier is only applied to tier-1 confidence
        # thresholds in `_determine_tier`, where its semantics (require more
        # raw confidence) are unambiguous.
        include = probs >= (1.0 - self.conformal_thresholds)
        result.conformal_prediction_set = [
            self.class_order[i] for i in range(self.num_classes) if include[i]
        ]

        # Stash adaptive-multiplier breakdown into quality_flags for debugging
        result.quality_flags["adaptive_multiplier"] = round(adaptive_mul, 4)
        result.quality_flags["ood_distance_bucket"] = distance_bucket

        # ── Research/explainability data (2026-04-18) ────────────────────
        # Per-signal full 9-dim distributions (4×9 or 3×9 matrix)
        sig_layout = (["model2", "efficientnet", "psv", "dinov2_head"]
                       if self.use_psv
                       else ["model2", "efficientnet", "dinov2_head"])
        result.per_signal_full_distributions = {
            sig_layout[i]: [round(float(x), 6) for x in signal_preds_vecs[i]]
            for i in range(len(sig_layout))
            if signal_preds_vecs[i] is not None
        }
        # Per-class Mahalanobis distances (from _compute_ood_distance return)
        result.per_class_mahalanobis = dict(per_class_mahal)
        # SHAP-style signal contributions toward the predicted class
        result.signal_contributions = self._compute_signal_contributions(
            signal_preds_vecs, gate_w, int(probs.argmax())
        )
        # PSV feature firing — only if PSV ran AND raw features were captured
        if psv_raw_features:
            result.psv_feature_firing = self._compute_psv_feature_firing(
                psv_raw_features, diagnosis, top_k=8,
            )

        # Per-stage image transformations for the Pipeline Inspector
        try:
            result.pipeline_visualizations = self._compute_pipeline_visualizations(img_rgb)
        except Exception as e:
            logger.warning(f"Pipeline visualizations skipped: {e}")

        # GradCAM++ on the gate-weighted top signal (Gap 2 audit fix).
        # Skip for OOD/4A/4B (heatmap on an image the model doesn't recognize
        # is misleading) and for 3B (image-quality is too poor to interpret).
        if result.tier not in ("4A", "4B", "3B"):
            try:
                cam_b64, cam_signal = self._generate_gradcam(
                    img_rgb, gate_w, int(probs.argmax())
                )
                if cam_b64:
                    result.gradcam_b64_png = cam_b64
                    result.gradcam_source_signal = cam_signal
            except Exception as e:
                logger.warning(f"GradCAM++ generation skipped: {e}")

        # Compose message (uses tier + uncertainty for differentiated guidance)
        self._compose_output(result)

        result.processing_time_ms = round((time.time() - t_start) * 1000, 2)
        return result


# ========================================================================
# CLI self-test
# ========================================================================
if __name__ == "__main__":
    apin = APINInference(verbose=True)
    # Self-test with a random field-photo-like image
    rng = np.random.default_rng(42)
    img = rng.integers(50, 180, (400, 400, 3), dtype=np.uint8)
    # Add green tint to pass leaf check
    img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int32) + 60, 0, 255).astype(np.uint8)
    result = apin.predict(img)
    print(json.dumps(result.to_dict(), indent=2, default=str))
