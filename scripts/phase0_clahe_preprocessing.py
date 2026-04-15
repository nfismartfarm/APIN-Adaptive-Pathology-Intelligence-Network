"""
Phase 0 Step 0.9: LAB-CLAHE Offline Preprocessing

Applies CLAHE to the L (luminance) channel in LAB colorspace for all training
images across Model 2, Model 3, and Router cleaned directories.

Why LAB not RGB: Standard CLAHE on RGB channels independently causes hue shifts
(green leaves can become bluish/yellowish). LAB separates brightness (L) from
colour information (A=green-red, B=blue-yellow). CLAHE on L only adjusts contrast
and brightness while preserving the exact green-yellow colour signature.

Output: clahe/ subdirectory alongside each cleaned class folder.
  e.g., data/specialist/model3/cleaned/tomato_foliar_spot/ (originals)
        data/specialist/model3/cleaned_clahe/tomato_foliar_spot/ (CLAHE versions)

Does NOT modify original images. Creates parallel directory structure.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')
IMG_EXT = {'.jpg', '.jpeg', '.png'}

# Source and destination directories
SOURCES = [
    (ROOT / 'data' / 'specialist' / 'model2' / 'cleaned',
     ROOT / 'data' / 'specialist' / 'model2' / 'cleaned_clahe'),
    (ROOT / 'data' / 'specialist' / 'model3' / 'cleaned',
     ROOT / 'data' / 'specialist' / 'model3' / 'cleaned_clahe'),
    (ROOT / 'data' / 'specialist' / 'router' / 'cleaned',
     ROOT / 'data' / 'specialist' / 'router' / 'cleaned_clahe'),
]


def apply_lab_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to L channel only in LAB colorspace."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_directory(src_root, dst_root):
    """Process all images in src_root, save CLAHE versions to dst_root."""
    processed = 0
    skipped = 0
    failed = 0

    # Collect all image files
    all_files = []
    for cls_dir in sorted(src_root.iterdir()):
        if not cls_dir.is_dir() or 'QUARANTINED' in cls_dir.name:
            continue
        for img_file in cls_dir.iterdir():
            if img_file.suffix.lower() in IMG_EXT:
                all_files.append((cls_dir.name, img_file))

    print(f'  Source: {src_root}')
    print(f'  Destination: {dst_root}')
    print(f'  Images to process: {len(all_files)}')

    for cls_name, img_path in tqdm(all_files, desc=f'  CLAHE {src_root.name}',
                                    file=sys.stdout):
        dst_cls_dir = dst_root / cls_name
        dst_cls_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_cls_dir / img_path.name

        # Skip if already processed
        if dst_path.exists():
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            clahe_img = apply_lab_clahe(img)
            cv2.imwrite(str(dst_path), clahe_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            processed += 1
        except Exception as e:
            failed += 1

    return processed, skipped, failed


if __name__ == '__main__':
    print('=' * 70)
    print('PHASE 0 STEP 0.9: LAB-CLAHE OFFLINE PREPROCESSING')
    print('=' * 70)
    print()

    total_processed = 0
    total_skipped = 0
    total_failed = 0
    t_start = time.time()

    for src, dst in SOURCES:
        if not src.exists():
            print(f'  SKIP: {src} (not found)')
            continue
        p, s, f = process_directory(src, dst)
        total_processed += p
        total_skipped += s
        total_failed += f
        print(f'  Processed: {p}, Skipped (already done): {s}, Failed: {f}')
        print()

    elapsed = time.time() - t_start
    rate = total_processed / max(elapsed, 1)
    print(f'CLAHE COMPLETE')
    print(f'  Total processed: {total_processed}')
    print(f'  Total skipped: {total_skipped}')
    print(f'  Total failed: {total_failed}')
    print(f'  Time: {elapsed:.0f}s ({rate:.0f} img/s)')
    print(f'  Output directories created alongside cleaned/ folders')
