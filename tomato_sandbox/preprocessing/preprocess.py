"""
Image preprocessing pipelines for the Tomato 3-Signal sandbox.

Three separate pipelines are implemented because each downstream signal
consumer (Signal A v3, Signal B LoRA, Signal C PSV) was trained or designed
expecting a specific input format. Mismatching is a recipe for silent accuracy
degradation — the model produces output but it is biased.

spec: section 7.1 lines 1394-1418
spec: section 7.2 lines 1419-1463  (Pipeline 1 — for v3 / Signal A)
spec: section 7.3 lines 1465-1506  (Pipeline 2 — for LoRA / Signal B)
spec: section 7.4 lines 1508-1551  (Pipeline 3 — for PSV / Signal C)
spec: section 7.5 lines 1553-1559  (Caching note — TTA caller's responsibility)
spec: section 7.6 line 1563        (File location)

Call pattern (spec section 7.1 lines 1407-1417):

    v3_input   = preprocess_for_v3(validated.pil_image)    # [3, 224, 224] tensor
    lora_input = preprocess_for_lora(validated.pil_image)  # [3, 392, 392] tensor
    psv_input  = preprocess_for_psv(validated.pil_image)   # [H, W, 3] uint8 numpy

TTA note: when TTA fires (Section 11) the orchestrator calls these functions
again with each augmented PIL image.  This module exposes the functions;
TTA orchestration is Section 11's responsibility (DEC-031).

No print() in this module.  All informational output uses get_logger.
"""

from __future__ import annotations

import cv2  # type: ignore[import]
import numpy as np

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    from PIL import Image as _PIL_Image  # type: ignore[import]
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

from tomato_sandbox.config import (
    CLAHE_CLIP_LIMIT,      # spec: section 7.2 line 1424
    CLAHE_TILE_GRID,       # spec: section 7.2 line 1425
    IMAGENET_MEAN,         # spec: section 7.2 lines 1426-1427
    IMAGENET_STD,          # spec: section 7.2 line 1427
    LORA_INPUT_SIZE,       # spec: section 7.2 line 1429
    LORA_PAD_VALUE,        # spec: section 7.2 lines 1430-1431
    V3_INPUT_SIZE,         # spec: section 7.2 line 1428
)
from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.nan_guards import guard_array

# ---------------------------------------------------------------------------
# Module logger — no print()
# ---------------------------------------------------------------------------
_logger = get_logger(__name__)

# Number of output classes for each neural signal (used in guard_array calls).
# Signal A (v3) and Signal B (LoRA) each produce 6-class probability vectors;
# the tensor shape itself ([3, H, W]) is validated, not the class count.
# We use guard_array on the float32 array (before tensor conversion) to catch
# edge-case non-finite pixel values produced by CLAHE or normalization on
# degenerate images (all-black, all-white, solid-colour).
# spec: section 26 (production hygiene) — non-finite values must not propagate.
_V3_ARRAY_LEN: int = V3_INPUT_SIZE * V3_INPUT_SIZE * 3    # 224*224*3 = 150528
_LORA_ARRAY_LEN: int = LORA_INPUT_SIZE * LORA_INPUT_SIZE * 3  # 392*392*3 = 460992


# ---------------------------------------------------------------------------
# Internal: LAB-CLAHE helper
# Used by both preprocess_for_v3 and preprocess_for_lora.
# spec: section 7.2 lines 1447-1450, section 7.3 lines 1491-1495
# "Why LAB-CLAHE and not RGB-CLAHE: LAB separates luminance (L) from color
#  (A, B). Applying CLAHE to L only enhances local contrast without distorting
#  color. v3 was trained with LAB-CLAHE; inference must match."
# ---------------------------------------------------------------------------

def _apply_lab_clahe(rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of an RGB image (in LAB space).

    Input:  uint8 [H, W, 3] RGB array.
    Output: uint8 [H, W, 3] RGB array with locally enhanced luminance contrast.

    spec: section 7.2 lines 1447-1450
    "lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
     clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
     lab[:, :, 0] = clahe.apply(lab[:, :, 0])
     rgb_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)"
    """
    # spec: section 7.2 line 1447 — COLOR_RGB2LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    # spec: section 7.2 lines 1448-1449
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,       # spec: section 7.2 line 1424
        tileGridSize=CLAHE_TILE_GRID,     # spec: section 7.2 line 1425
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    # spec: section 7.2 line 1450 — COLOR_LAB2RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Pipeline 1 — for v3 (Signal A)
# spec: section 7.2 lines 1437-1458
# ---------------------------------------------------------------------------

def preprocess_for_v3(pil_image: "_PIL_Image.Image") -> "torch.Tensor":
    """Return a [3, 224, 224] CPU tensor in the exact format v3 was trained with.

    Steps (spec section 7.2 lines 1442-1457):
      1. Resize with stretch (no aspect-ratio preservation) to V3_INPUT_SIZE × V3_INPUT_SIZE
         using PIL.Image.BILINEAR.  spec line 1443.
      2. LAB-CLAHE on L channel.  spec lines 1447-1450.
      3. Float32 conversion + ImageNet normalization.  spec lines 1453-1454.
      4. CHW tensor.  spec line 1457.

    Args:
        pil_image: PIL RGB image (any size).

    Returns:
        CPU float32 torch.Tensor of shape [3, 224, 224].

    spec: section 7.2 lines 1437-1458
    "Returns a [3, 224, 224] tensor on CPU, ImageNet-normalized,
     in the exact format v3 was trained with."
    """
    # Step 1 — stretch resize (no aspect-ratio preservation)
    # spec: section 7.2 lines 1442-1444
    # "Resize with stretch (no aspect-ratio preservation)"
    # PIL.Image.BILINEAR is the interpolation method specified verbatim.
    resized = pil_image.resize(
        (V3_INPUT_SIZE, V3_INPUT_SIZE),  # spec: section 7.2 line 1428, line 1443
        _PIL_Image.BILINEAR,              # spec: section 7.2 line 1443
    )
    rgb = np.array(resized, dtype=np.uint8)  # [224, 224, 3]

    # Step 2 — LAB-CLAHE on L channel
    # spec: section 7.2 lines 1447-1450
    rgb_clahe = _apply_lab_clahe(rgb)

    # Step 3 — to float32 and ImageNet normalize
    # spec: section 7.2 lines 1453-1454
    arr = rgb_clahe.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # spec: 7.2 lines 1426-1427

    # Finiteness guard — defensive per spec section 26 production hygiene.
    # Degenerate images (e.g. solid-black) can produce NaN after normalization
    # if std contains near-zero values from implementation edge cases.
    # guard_array returns zero-filled array if any value is non-finite.
    arr_flat = arr.flatten()
    arr_flat_guarded = guard_array(arr_flat, expected_len=_V3_ARRAY_LEN, default_value=0.0)
    if not np.array_equal(arr_flat, arr_flat_guarded):
        _logger.warning(
            "preprocess_for_v3 produced non-finite values; zeroing output",
            step="preprocess_for_v3",
            succeeded=False,
        )
        arr = arr_flat_guarded.reshape(V3_INPUT_SIZE, V3_INPUT_SIZE, 3)

    # Step 4 — CHW tensor
    # spec: section 7.2 lines 1456-1457 — "arr.transpose(2, 0, 1)"
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor  # [3, 224, 224] float32 CPU tensor


# ---------------------------------------------------------------------------
# Pipeline 2 — for LoRA (Signal B)
# spec: section 7.3 lines 1468-1501
# ---------------------------------------------------------------------------

def preprocess_for_lora(pil_image: "_PIL_Image.Image") -> "torch.Tensor":
    """Return a [3, 392, 392] CPU tensor with letterbox padding, ImageNet-normalized.

    Steps (spec section 7.3 lines 1472-1501):
      1. Letterbox resize: scale longest side to LORA_INPUT_SIZE, preserve AR,
         pad remainder with LORA_PAD_VALUE (114) using INTER_LINEAR.  spec lines
         1473-1489.
      2. LAB-CLAHE on L channel (same parameters as v3).  spec lines 1491-1495.
      3. Float32 conversion + ImageNet normalization.  spec lines 1497-1499.
      4. CHW tensor.  spec line 1501.

    Args:
        pil_image: PIL RGB image (any size, any aspect ratio).

    Returns:
        CPU float32 torch.Tensor of shape [3, 392, 392].

    spec: section 7.3 lines 1468-1501
    "Returns a [3, 392, 392] tensor, ImageNet-normalized, with letterbox padding."
    """
    # Step 1 — letterbox resize (preserves aspect ratio, pads to square)
    # spec: section 7.3 lines 1473-1489
    arr = np.array(pil_image, dtype=np.uint8)
    H, W = arr.shape[:2]
    # spec: section 7.3 line 1475 — "scale = LORA_INPUT_SIZE / max(H, W)"
    scale = LORA_INPUT_SIZE / max(H, W)
    # spec: section 7.3 line 1476
    new_h, new_w = int(H * scale), int(W * scale)
    # spec: section 7.3 line 1477 — cv2.INTER_LINEAR
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Symmetric padding — spec: section 7.3 lines 1480-1489
    pad_h = LORA_INPUT_SIZE - new_h
    pad_w = LORA_INPUT_SIZE - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # spec: section 7.3 lines 1486-1489
    # "cv2.copyMakeBorder with cv2.BORDER_CONSTANT value=(114, 114, 114)"
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,                                   # spec: 7.3 line 1487
        value=(LORA_PAD_VALUE, LORA_PAD_VALUE, LORA_PAD_VALUE),  # spec: 7.3 line 1488
    )

    # Step 2 — LAB-CLAHE on L channel (same as v3)
    # spec: section 7.3 lines 1491-1495
    rgb_clahe = _apply_lab_clahe(padded)

    # Step 3 — to float32 and ImageNet normalize
    # spec: section 7.3 lines 1497-1499
    arr_f = rgb_clahe.astype(np.float32) / 255.0
    arr_f = (arr_f - IMAGENET_MEAN) / IMAGENET_STD  # spec: 7.3 line 1499

    # Finiteness guard — defensive per spec section 26 production hygiene.
    arr_flat = arr_f.flatten()
    arr_flat_guarded = guard_array(arr_flat, expected_len=_LORA_ARRAY_LEN, default_value=0.0)
    if not np.array_equal(arr_flat, arr_flat_guarded):
        _logger.warning(
            "preprocess_for_lora produced non-finite values; zeroing output",
            step="preprocess_for_lora",
            succeeded=False,
        )
        arr_f = arr_flat_guarded.reshape(LORA_INPUT_SIZE, LORA_INPUT_SIZE, 3)

    # Step 4 — CHW tensor
    # spec: section 7.3 line 1501 — "torch.from_numpy(arr_f.transpose(2, 0, 1))"
    return torch.from_numpy(arr_f.transpose(2, 0, 1))  # [3, 392, 392] float32 CPU tensor


# ---------------------------------------------------------------------------
# Shades-of-Gray color constancy — used by preprocess_for_psv
# spec: section 7.4 lines 1532-1544
# ---------------------------------------------------------------------------

def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Shades-of-Gray color constancy (Finlayson & Trezzi 2004).

    p=1 is grey-world; p=infinity is max-RGB; p=6 was empirically best
    in the original paper and validated for our use in the PDA review.

    Args:
        img: uint8 [H, W, 3] RGB array.
        p:   Minkowski norm order.  Default 6 (spec: section 7.4 line 1536).

    Returns:
        uint8 [H, W, 3] RGB array with color constancy applied.

    spec: section 7.4 lines 1532-1544
    "Shades-of-Gray color constancy (Finlayson & Trezzi 2004).
     p=1 is grey-world; p=infinity is max-RGB; p=6 was empirically best..."
    """
    # spec: section 7.4 line 1538
    img_f = img.astype(np.float64)
    # spec: section 7.4 lines 1539-1540 — "Minkowski p-norm of each channel"
    illuminant = np.power(
        np.mean(img_f ** p, axis=(0, 1)),  # mean over H, W axes; shape (3,)
        1.0 / p,
    )
    # spec: section 7.4 lines 1541-1542
    # "Normalize: scale each channel so max-channel illuminant becomes 1"
    scale = illuminant.max() / illuminant
    img_corrected = img_f * scale
    # spec: section 7.4 line 1543
    return np.clip(img_corrected, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pipeline 3 — for PSV (Signal C)
# spec: section 7.4 lines 1511-1524
# ---------------------------------------------------------------------------

_PSV_MAX_SIDE: int = 1200  # spec: section 7.4 line 1519 — "capped at 1200 px"

def preprocess_for_psv(pil_image: "_PIL_Image.Image") -> np.ndarray:
    """Return an [H, W, 3] uint8 RGB array with color constancy applied.

    NO LAB-CLAHE, NO tensor conversion.  PSV operates on color-corrected RGB
    at native resolution, capped at 1200 px on the longest side to bound CPU cost.

    Steps (spec section 7.4 lines 1517-1523):
      1. Convert to uint8 RGB numpy array.
      2. If max(H, W) > 1200: scale down with cv2.INTER_AREA.
      3. Apply Shades-of-Gray (p=6) color constancy.

    Args:
        pil_image: PIL RGB image (any size).

    Returns:
        uint8 [H, W, 3] RGB numpy array, color-constancy corrected.
        H and W are at most 1200; aspect ratio is preserved.

    spec: section 7.4 lines 1511-1524
    "Returns an [H, W, 3] uint8 RGB array with color constancy applied.
     NO LAB-CLAHE, NO tensor conversion.  PSV operates on color-corrected RGB
     at native resolution, capped at 1200 px on the longest side..."

    spec: section 7.4 lines 1549 — "No CLAHE for PSV: CLAHE alters the color
    statistics PSV measures. Applying CLAHE before PSV would invalidate the F.0
    calibration of HSV thresholds."
    """
    # spec: section 7.4 line 1517
    rgb = np.array(pil_image, dtype=np.uint8)
    H, W = rgb.shape[:2]

    # spec: section 7.4 lines 1519-1522 — resize cap at 1200 px longest side
    if max(H, W) > _PSV_MAX_SIDE:
        scale = _PSV_MAX_SIDE / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        # spec: section 7.4 line 1522 — cv2.INTER_AREA
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # spec: section 7.4 line 1523 — "rgb_cc = shades_of_gray(rgb, p=6)"
    rgb_cc = shades_of_gray(rgb, p=6)
    return rgb_cc
