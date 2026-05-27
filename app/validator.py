# app/validator.py

import numpy as np
import io
from PIL import Image
from app.config import (
    MAX_FILE_MB, MIN_BLUR_VAR, MIN_PIXEL_MEAN, MAX_PIXEL_MEAN,
    MIN_IMG_DIM, MAX_CH_RATIO
)


def _check_magic_bytes(data: bytes) -> str:
    if data[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if len(data) >= 12 and data[8:12] == b'WEBP':
        return 'webp'
    return 'unknown'


def validate_image(data: bytes) -> dict:
    """
    Validates uploaded image. Returns dict with valid, reason, image keys.
    [FIX GAP 60] HEIC not accepted — pillow-heif not installed.

    Checks:
    1. File size <= MAX_FILE_MB
    2. Magic bytes: jpeg/png/webp only
    3. PIL can open it
    4. Minimum dimensions
    5. Blur check: Laplacian variance >= MIN_BLUR_VAR
    6. Pixel mean within [MIN_PIXEL_MEAN, MAX_PIXEL_MEAN]
    7. No single channel dominates > MAX_CH_RATIO
    """
    # 1. File size
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return {'valid': False, 'reason': f'File too large (max {MAX_FILE_MB} MB)', 'image': None}

    # 2. Magic bytes
    fmt = _check_magic_bytes(data)
    if fmt == 'unknown':
        return {'valid': False,
                'reason': 'Unsupported format. Upload JPEG, PNG, or WebP.',
                'image': None}

    # 3. PIL open
    try:
        pil_img = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return {'valid': False, 'reason': 'Could not open image file.', 'image': None}

    img_np = np.array(pil_img, dtype=np.uint8)

    # 4. Minimum dimensions (100x100 absolute minimum)
    h, w = img_np.shape[:2]
    if h < 100 or w < 100:
        return {'valid': False,
                'reason': 'Image resolution is too low. Please upload a photo of at least 100x100 pixels.',
                'image': None}
    if h < MIN_IMG_DIM or w < MIN_IMG_DIM:
        return {'valid': False,
                'reason': f'Image too small. Minimum {MIN_IMG_DIM}px in each dimension.',
                'image': None}

    # 4b. Extreme aspect ratio check
    aspect_ratio = w / h
    if aspect_ratio > 8.0 or aspect_ratio < 0.125:
        return {'valid': False,
                'reason': 'Image has an unusual shape. Please upload a standard portrait or landscape photograph of a leaf.',
                'image': None}

    # 4c. Blank or solid colour image check
    pixel_std = float(np.std(img_np))
    if pixel_std < 5.0:
        return {'valid': False,
                'reason': 'Image appears to be blank or a solid colour. Please upload a clear photograph of a leaf.',
                'image': None}

    # 5. Blur check
    import cv2
    gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < MIN_BLUR_VAR:
        return {'valid': False,
                'reason': f'Image too blurry (score {blur_var:.1f}). Take a clearer photo.',
                'image': None}

    # 6. Pixel mean
    mean_val = float(img_np.mean())
    if mean_val < MIN_PIXEL_MEAN:
        return {'valid': False, 'reason': 'Image is too dark.', 'image': None}
    if mean_val > MAX_PIXEL_MEAN:
        return {'valid': False, 'reason': 'Image is overexposed.', 'image': None}

    # 7. Single-channel dominance
    ch_means = img_np.mean(axis=(0, 1))
    total    = ch_means.sum()
    if total > 0 and ch_means.max() / total > MAX_CH_RATIO:
        return {'valid': False,
                'reason': 'Image does not appear to contain a plant leaf.',
                'image': None}

    return {'valid': True, 'reason': '', 'image': img_np}
