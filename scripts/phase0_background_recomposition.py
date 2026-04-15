"""
Phase 0 Step 0.10: Background Recomposition

Segments diseased tomato/Capsicum leaves from lab backgrounds using InSPyReNet
(transparent-background library, GPU-accelerated via PyTorch CUDA), pastes them
onto randomly selected field backgrounds from chilli/brassica datasets.

Per MASTER_PLAN Section 3.2:
- Cap at 2,000 recompositions per tomato disease class
- Also recompose Capsicum lab images in chilli_healthy (multi_D source)
- Quality filter: foreground mask covers 15-85% of image area
- Output: data/specialist/model3/recomposed/{class_name}/
- source_dataset: 'scidb_recomposed' or 'capsicum_recomposed'
- is_field_photo: True (synthetic field conditions)

Validated by: "Bridging the Lab-to-Field gap in plant disease diagnosis through
unsupervised domain adaptation enhanced by background recomposition" (ScienceDirect 2025)
"""
import os
import sys
import random
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# Use transparent-background (PyTorch-native, GPU-accelerated) instead of rembg (ONNX CPU)
# Speedup: 3.9 img/s GPU vs 0.33 img/s CPU = 11.8x
try:
    from transparent_background import Remover
    _remover = None  # lazy init
    _USE_TB = True
except ImportError:
    from rembg import remove
    _USE_TB = False
    print('WARNING: transparent-background not installed, falling back to rembg CPU')

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')
MODEL3_CLEANED = ROOT / 'data' / 'specialist' / 'model3' / 'cleaned'
MODEL2_CLEANED = ROOT / 'data' / 'specialist' / 'model2' / 'cleaned'
RECOMPOSED_DIR = ROOT / 'data' / 'specialist' / 'model3' / 'recomposed'
IMG_EXT = {'.jpg', '.jpeg', '.png'}

# Config from MASTER_PLAN
SCIDB_CAP_PER_CLASS = 2000
FOREGROUND_MIN_PCT = 0.15
FOREGROUND_MAX_PCT = 0.85
JPEG_QUALITY = 92

# Classes to recompose
TOMATO_CLASSES = [
    'tomato_foliar_spot', 'tomato_late_blight', 'tomato_septoria_leaf_spot',
    'tomato_yellow_leaf_curl_virus', 'tomato_mosaic_virus',
]
CAPSICUM_SOURCE = 'multi_D'  # Capsicum images in chilli_healthy

# Source identification: scidb images have these filename prefixes
SCIDB_PREFIXES = ['src2m3_scidb', 'src2_', 'original_pool']  # approximate


def collect_field_backgrounds(min_size=224):
    """Collect field background images from chilli and brassica cleaned directories."""
    backgrounds = []
    bg_dirs = [
        MODEL3_CLEANED / 'chilli_healthy',
        MODEL3_CLEANED / 'chilli_leaf_curl',
        MODEL2_CLEANED / 'brassica_healthy',
    ]
    for bg_dir in bg_dirs:
        if not bg_dir.exists():
            continue
        for f in bg_dir.iterdir():
            if f.suffix.lower() in IMG_EXT:
                backgrounds.append(f)
    random.shuffle(backgrounds)
    print(f'Field background pool: {len(backgrounds)} images')
    return backgrounds


def collect_scidb_images(class_name, cap=SCIDB_CAP_PER_CLASS):
    """Collect scidb-origin images from a tomato disease class, capped at cap."""
    cls_dir = MODEL3_CLEANED / class_name
    if not cls_dir.exists():
        return []

    # Identify scidb images by filename prefix
    scidb_images = []
    for f in cls_dir.iterdir():
        if f.suffix.lower() not in IMG_EXT:
            continue
        fname = f.name.lower()
        if any(fname.startswith(p.lower()) for p in SCIDB_PREFIXES):
            scidb_images.append(f)

    # If we can't identify by prefix, take images that are NOT obviously field
    # (field images have prefixes like src3_, src4_, taiwan_, inat_)
    if len(scidb_images) < 100:
        field_prefixes = ['src3_', 'src4_', 'taiwan_', 'inat_', 'srcJ_']
        all_images = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT]
        non_field = [f for f in all_images
                     if not any(f.name.lower().startswith(p) for p in field_prefixes)]
        scidb_images = non_field

    # Cap
    if len(scidb_images) > cap:
        random.seed(42)
        scidb_images = random.sample(scidb_images, cap)

    return scidb_images


def collect_capsicum_images():
    """Collect Capsicum (multi_D source) images from chilli_healthy."""
    cls_dir = MODEL3_CLEANED / 'chilli_healthy'
    if not cls_dir.exists():
        return []

    capsicum = [f for f in cls_dir.iterdir()
                if f.suffix.lower() in IMG_EXT and 'multi_D' in f.name]
    return capsicum


def _get_remover():
    """Lazy-init the GPU segmentation model."""
    global _remover
    if _USE_TB and _remover is None:
        _remover = Remover(mode='fast', device='cuda' if torch.cuda.is_available() else 'cpu')
    return _remover


def recompose_image(leaf_path, bg_path, output_path):
    """
    Segment leaf from lab background, paste onto field background.
    Uses transparent-background (GPU) or rembg (CPU fallback).
    Returns True if successful, False if quality check fails.
    """
    try:
        # Load and segment leaf
        leaf_img = Image.open(leaf_path).convert('RGB')
        if _USE_TB:
            remover = _get_remover()
            segmented = remover.process(leaf_img, type='rgba')  # returns RGBA PIL
        else:
            leaf_rgba = leaf_img.convert('RGBA')
            segmented = remove(leaf_rgba)
        seg_array = np.array(segmented)

        # Quality check: foreground coverage
        alpha = seg_array[:, :, 3]
        total_pixels = alpha.shape[0] * alpha.shape[1]
        fg_pixels = (alpha > 128).sum()
        fg_ratio = fg_pixels / total_pixels

        if fg_ratio < FOREGROUND_MIN_PCT or fg_ratio > FOREGROUND_MAX_PCT:
            return False  # segmentation quality too poor

        # Load field background
        bg_img = Image.open(bg_path).convert('RGB')

        # Resize background to match leaf dimensions
        leaf_w, leaf_h = leaf_img.size
        bg_img = bg_img.resize((leaf_w, leaf_h), Image.LANCZOS)

        # [FIX] Ensure segmented output matches background dimensions
        # transparent-background may internally resize; force match before composite
        if segmented.size != bg_img.size:
            segmented = segmented.resize(bg_img.size, Image.LANCZOS)

        # Composite: paste segmented leaf onto field background
        bg_rgba = bg_img.convert('RGBA')
        composite = Image.alpha_composite(bg_rgba, segmented)
        composite_rgb = composite.convert('RGB')

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite_rgb.save(output_path, 'JPEG', quality=JPEG_QUALITY)

        return True

    except Exception as e:
        return False


def main():
    print('=' * 70, flush=True)
    print('PHASE 0 STEP 0.10: BACKGROUND RECOMPOSITION', flush=True)
    print('Using: ' + ('transparent-background (GPU PyTorch)' if _USE_TB else 'rembg (CPU ONNX)'), flush=True)
    print('=' * 70, flush=True)
    print(flush=True)

    random.seed(42)

    # Collect field backgrounds
    backgrounds = collect_field_backgrounds()
    if not backgrounds:
        print('ERROR: No field background images found!')
        sys.exit(1)

    total_processed = 0
    total_success = 0
    total_quality_fail = 0
    total_error = 0
    t_start = time.time()

    # Process tomato scidb images
    for cls in TOMATO_CLASSES:
        scidb_images = collect_scidb_images(cls)
        if not scidb_images:
            print(f'  [{cls}] No scidb images found — skipping')
            continue

        out_dir = RECOMPOSED_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'  [{cls}] Processing {len(scidb_images)} scidb images...')
        cls_success = 0
        cls_fail = 0

        for i, leaf_path in enumerate(scidb_images):
            # Random background
            bg_path = backgrounds[random.randint(0, len(backgrounds) - 1)]

            # Output filename
            out_name = f'recomp_scidb_{leaf_path.stem}_{i:05d}.jpg'
            out_path = out_dir / out_name

            # Skip if already exists
            if out_path.exists():
                cls_success += 1
                continue

            if recompose_image(leaf_path, bg_path, out_path):
                cls_success += 1
                total_success += 1
            else:
                cls_fail += 1
                if out_path.exists():
                    out_path.unlink()  # remove incomplete file

            total_processed += 1

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                rate = total_processed / max(elapsed, 1)
                print(f'    {i+1}/{len(scidb_images)} '
                      f'({cls_success} ok, {cls_fail} quality-fail, '
                      f'{rate:.1f} img/s)')

        total_quality_fail += cls_fail
        print(f'  [{cls}] Done: {cls_success} success, {cls_fail} quality-fail')

    # Process Capsicum images
    capsicum_images = collect_capsicum_images()
    if capsicum_images:
        out_dir = RECOMPOSED_DIR / 'chilli_healthy'
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'  [capsicum_recomposed] Processing {len(capsicum_images)} Capsicum images...')
        cap_success = 0
        cap_fail = 0

        for i, leaf_path in enumerate(capsicum_images):
            bg_path = backgrounds[random.randint(0, len(backgrounds) - 1)]
            out_name = f'recomp_capsicum_{leaf_path.stem}_{i:05d}.jpg'
            out_path = out_dir / out_name

            if out_path.exists():
                cap_success += 1
                continue

            if recompose_image(leaf_path, bg_path, out_path):
                cap_success += 1
                total_success += 1
            else:
                cap_fail += 1
                if out_path.exists():
                    out_path.unlink()

            total_processed += 1

            if (i + 1) % 100 == 0:
                print(f'    {i+1}/{len(capsicum_images)} ({cap_success} ok, {cap_fail} fail)')

        total_quality_fail += cap_fail
        print(f'  [capsicum] Done: {cap_success} success, {cap_fail} quality-fail')

    # Summary
    elapsed = time.time() - t_start
    print()
    print('=' * 70)
    print('RECOMPOSITION COMPLETE')
    print(f'  Total processed: {total_processed}')
    print(f'  Successful: {total_success}')
    print(f'  Quality-filtered: {total_quality_fail}')
    print(f'  Time: {elapsed:.0f}s ({total_processed/max(elapsed,1):.1f} img/s)')
    print()

    # Count output
    print('Output directory contents:')
    if RECOMPOSED_DIR.exists():
        for d in sorted(RECOMPOSED_DIR.iterdir()):
            if d.is_dir():
                n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
                print(f'  {d.name}: {n} images')

    print()
    print('NEXT: Run rebuild_unified_csvs.py to include recomposed images in CSVs')


if __name__ == '__main__':
    main()
