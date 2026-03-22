# tools/add_kerala_image.py
"""
[FIX GAP 25] Adds a verified Kerala field image to source_map.csv.
Usage: python tools/add_kerala_image.py --path img.jpg --class okra_yvmv
"""

import os
import sys
import csv
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ROOT, CLASS_TO_IDX, CROP_FROM_IDX, SOURCE_MAP


def add_kerala_image(src_path, class_name):
    if class_name not in CLASS_TO_IDX:
        print(f"Unknown class: {class_name}")
        print(f"Valid: {sorted(CLASS_TO_IDX.keys())}")
        sys.exit(1)

    if not os.path.exists(src_path):
        print(f"File not found: {src_path}")
        sys.exit(1)

    with open(src_path, 'rb') as f:
        data = f.read()
    from app.validator import validate_image
    result = validate_image(data)
    if not result['valid']:
        print(f"Validation failed: {result['reason']}")
        sys.exit(1)

    class_dir = os.path.join(ROOT, 'data', 'kerala', class_name)
    os.makedirs(class_dir, exist_ok=True)

    fname    = os.path.basename(src_path)
    dst_path = os.path.join(class_dir, fname)
    if os.path.exists(dst_path):
        import time
        base, ext = os.path.splitext(fname)
        fname     = f"{base}_{int(time.time())}{ext}"
        dst_path  = os.path.join(class_dir, fname)

    shutil.copy2(src_path, dst_path)
    rel_path  = os.path.relpath(dst_path, ROOT).replace('\\', '/')
    class_idx = CLASS_TO_IDX[class_name]
    crop_idx  = CROP_FROM_IDX[class_idx]

    write_header = not os.path.exists(SOURCE_MAP)
    with open(SOURCE_MAP, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['image_path', 'source_dataset', 'raw_label',
                      'class_name', 'class_idx', 'crop_idx', 'split']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'image_path'    : rel_path,
            'source_dataset': 'kerala',
            'raw_label'     : class_name,
            'class_name'    : class_name,
            'class_idx'     : class_idx,
            'crop_idx'      : crop_idx,
            'split'         : 'kerala',
        })

    print(f"Added: {rel_path}")
    print(f"Class: {class_name} (idx={class_idx})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',  required=True)
    parser.add_argument('--class', dest='class_name', required=True)
    args = parser.parse_args()
    add_kerala_image(args.path, args.class_name)
