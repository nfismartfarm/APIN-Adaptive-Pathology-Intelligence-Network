# training/01_prepare_data.py
# IMPLEMENT EXACTLY

import os
import sys
import csv
import json
import pathlib
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    ROOT, RAW, META, SOURCE_MAP, CLASS_COUNTS_PATH,
    CLASS_TO_IDX, CROP_FROM_IDX, VALID_EXT,
    KERALA_DIR, PLANTDOC_DIR, PLANTDOC_CLASS_MAP,
    LABEL_MAP, SOURCE_LABEL_OVERRIDES
)

# Import helpers from this file (defined below in same module)
# resolve_label and assert_all_labels_mapped are defined in this file.
# stratified_group_split is defined in this file.


def resolve_label(raw_label, source):
    key = (source, raw_label.lower().strip())
    if key in SOURCE_LABEL_OVERRIDES:
        return SOURCE_LABEL_OVERRIDES[key]
    normalised = raw_label.lower().strip()
    if normalised in LABEL_MAP:
        return LABEL_MAP[normalised]
    raise KeyError(f"No mapping for label='{raw_label}' from source='{source}'")


def assert_all_labels_mapped(records):
    unmapped = []
    for r in records:
        try:
            resolve_label(r['raw_label'], r['source_dataset'])
        except KeyError:
            unmapped.append(
                f"  source={r['source_dataset']!r}, "
                f"label={r['raw_label']!r}, path={r['image_path']!r}"
            )
    if unmapped:
        raise ValueError(
            f"Found {len(unmapped)} unmapped labels. "
            f"Add them to LABEL_MAP or SOURCE_LABEL_OVERRIDES in app/config.py:\n"
            + "\n".join(unmapped[:30])
        )
    print(f"Label assertion passed: all {len(records)} records are mapped.")


def _scan_source(source_id, source_dir):
    """
    Scans a single source directory tree.
    Returns list of record dicts for images with mappable labels.
    Skips images with unmappable labels and logs them.
    """
    records   = []
    skipped   = 0
    # Check for 'train' subdirectory (e.g. sabbir_okra)
    scan_root = source_dir
    train_sub = os.path.join(source_dir, 'train')
    if os.path.isdir(train_sub):
        scan_root = train_sub

    for dirpath, _, filenames in os.walk(scan_root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext not in VALID_EXT:
                continue
            full_path   = os.path.join(dirpath, fname)
            raw_label   = os.path.basename(dirpath)  # immediate parent = class folder
            # Compute relative path from ROOT
            try:
                rel_path = os.path.relpath(full_path, ROOT).replace('\\', '/')
            except ValueError:
                # Different drive on Windows — use absolute as fallback
                rel_path = full_path.replace('\\', '/')

            try:
                class_name = resolve_label(raw_label, source_id)
                class_idx  = CLASS_TO_IDX[class_name]
                crop_idx   = CROP_FROM_IDX[class_idx]
            except (KeyError, TypeError):
                skipped += 1
                continue

            records.append({
                'image_path'    : rel_path,
                'source_dataset': source_id,
                'raw_label'     : raw_label,
                'class_name'    : class_name,
                'class_idx'     : class_idx,
                'crop_idx'      : crop_idx,
                'split'         : '',  # filled in by stratified_group_split
            })

    if skipped > 0:
        print(f"  [{source_id}] Skipped {skipped} images with unmappable labels.")
    print(f"  [{source_id}] Loaded {len(records)} images.")
    return records


def _scan_misrak(source_dir):
    """
    misrak_veg: keep only subdirectories containing 'cabbage', 'broccoli',
    or 'brassica' (case-insensitive). Skip all others (tomato, etc.).
    """
    records = []
    for folder in os.listdir(source_dir):
        folder_lower = folder.lower()
        if not any(kw in folder_lower for kw in ('cabbage', 'broccoli', 'brassica')):
            continue
        class_dir = os.path.join(source_dir, folder)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1]
            if ext not in VALID_EXT:
                continue
            full_path = os.path.join(class_dir, fname)
            rel_path  = os.path.relpath(full_path, ROOT).replace('\\', '/')
            try:
                class_name = resolve_label(folder, 'misrak_veg')
                class_idx  = CLASS_TO_IDX[class_name]
                crop_idx   = CROP_FROM_IDX[class_idx]
            except (KeyError, TypeError):
                continue
            records.append({
                'image_path'    : rel_path,
                'source_dataset': 'misrak_veg',
                'raw_label'     : folder,
                'class_name'    : class_name,
                'class_idx'     : class_idx,
                'crop_idx'      : crop_idx,
                'split'         : '',
            })
    print(f"  [misrak_veg] Loaded {len(records)} brassica images.")
    return records


def _scan_plantdoc(plantdoc_dir):
    """
    PlantDoc: merge train/ and test/ subdirectories.
    Use PLANTDOC_CLASS_MAP for label resolution.
    All records get split='plantdoc' (never in training pool).
    """
    records = []
    for subset in ('train', 'test'):
        subset_dir = os.path.join(plantdoc_dir, subset)
        if not os.path.isdir(subset_dir):
            continue
        for folder in os.listdir(subset_dir):
            if folder not in PLANTDOC_CLASS_MAP:
                continue  # silently discard non-brassica classes
            class_name = PLANTDOC_CLASS_MAP[folder]
            class_idx  = CLASS_TO_IDX[class_name]
            crop_idx   = CROP_FROM_IDX[class_idx]
            class_dir  = os.path.join(subset_dir, folder)
            for fname in os.listdir(class_dir):
                ext = os.path.splitext(fname)[1]
                if ext not in VALID_EXT:
                    continue
                full_path = os.path.join(class_dir, fname)
                rel_path  = os.path.relpath(full_path, ROOT).replace('\\', '/')
                records.append({
                    'image_path'    : rel_path,
                    'source_dataset': 'plantdoc',
                    'raw_label'     : folder,
                    'class_name'    : class_name,
                    'class_idx'     : class_idx,
                    'crop_idx'      : crop_idx,
                    'split'         : 'plantdoc',
                })
    print(f"  [plantdoc] Loaded {len(records)} images.")
    return records


def _scan_kerala(kerala_dir):
    """
    Kerala tier-3 images. Each subdirectory is a class_name.
    All records get split='kerala'.
    """
    records = []
    if not os.path.isdir(kerala_dir):
        print("  [kerala] No images yet.")
        return records
    for class_name in os.listdir(kerala_dir):
        if class_name not in CLASS_TO_IDX:
            continue
        class_dir  = os.path.join(kerala_dir, class_name)
        class_idx  = CLASS_TO_IDX[class_name]
        crop_idx   = CROP_FROM_IDX[class_idx]
        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1]
            if ext not in VALID_EXT:
                continue
            full_path = os.path.join(class_dir, fname)
            rel_path  = os.path.relpath(full_path, ROOT).replace('\\', '/')
            records.append({
                'image_path'    : rel_path,
                'source_dataset': 'kerala',
                'raw_label'     : class_name,
                'class_name'    : class_name,
                'class_idx'     : class_idx,
                'crop_idx'      : crop_idx,
                'split'         : 'kerala',
            })
    print(f"  [kerala] Loaded {len(records)} images.")
    return records


def stratified_group_split(records, seed=42):
    """(full implementation in Section 6.3 above — copy verbatim)"""
    from sklearn.model_selection import StratifiedGroupKFold

    pool = [r for r in records
            if r.get('split') not in ('plantdoc', 'kerala', 'domain_adapt')]
    if not pool:
        raise ValueError("No records available for splitting. Check data/raw/.")

    X      = np.array([r['image_path']     for r in pool])
    labels = np.array([r['class_idx']      for r in pool])
    groups = np.array([r['source_dataset'] for r in pool])

    min_class = int(np.bincount(labels).min())
    if min_class < 2:
        raise ValueError(
            f"Minimum class count is {min_class}. Need >= 2. Download more data."
        )

    n_test = min(7, min_class)
    sgkf   = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(sgkf.split(X, labels, groups))

    X_tv, lab_tv, grp_tv = X[tv_idx], labels[tv_idx], groups[tv_idx]
    n_val = min(6, int(np.bincount(lab_tv).min()))
    if n_val < 2:
        raise ValueError("Not enough data for val split after test split.")
    sgkf2   = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_sub, val_sub = next(sgkf2.split(X_tv, lab_tv, grp_tv))

    train_idx = tv_idx[tr_sub]
    val_idx   = tv_idx[val_sub]

    train_r = [pool[i] for i in train_idx]
    val_r   = [pool[i] for i in val_idx]
    test_r  = [pool[i] for i in test_idx]

    for r in train_r: r['split'] = 'train'
    for r in val_r:   r['split'] = 'val'
    for r in test_r:  r['split'] = 'test'

    # Override: synthetic always in train
    for r in records:
        if r.get('source_dataset') == 'synthetic':
            r['split'] = 'train'
            if r not in train_r:
                train_r.append(r)

    train_src = {r['source_dataset'] for r in train_r}
    test_src  = {r['source_dataset'] for r in test_r}
    overlap   = train_src & test_src
    if overlap:
        print(f"WARNING: sources in both train and test: {overlap}")

    print(f"Split: {len(train_r)} train, {len(val_r)} val, {len(test_r)} test")
    return train_r, val_r, test_r


def write_source_map(all_records):
    """Write source_map.csv with all records."""
    os.makedirs(META, exist_ok=True)
    fieldnames = ['image_path', 'source_dataset', 'raw_label',
                  'class_name', 'class_idx', 'crop_idx', 'split']
    with open(SOURCE_MAP, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Written {len(all_records)} records to {SOURCE_MAP}")


def write_class_counts(all_records):
    """[FIX GAP 62] Write class_counts.csv: class_name, split, count."""
    from collections import Counter
    os.makedirs(META, exist_ok=True)
    counts = Counter((r['class_name'], r['split']) for r in all_records)
    with open(CLASS_COUNTS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'split', 'count'])
        for (cls, spl), cnt in sorted(counts.items()):
            writer.writerow([cls, spl, cnt])
    print(f"Written class counts to {CLASS_COUNTS_PATH}")


if __name__ == '__main__':
    print("=" * 60)
    print("01_PREPARE_DATA — scanning datasets, splitting, writing CSV")
    print("=" * 60)
    print("NOTE: Assumes datasets are already downloaded (Step 03 ran).")
    print("      Scanning data/raw/ for present datasets.\n")

    all_records = []

    # ── Priority-1 training datasets ──────────────────────────────────────
    TRAINING_SOURCES = [
        ('sabbir_okra',    os.path.join(RAW, 'sabbir_okra')),
        ('iubat_okra',     os.path.join(RAW, 'iubat_okra')),
        ('kareem_cabbage', os.path.join(RAW, 'kareem_cabbage')),
        ('faruk_okra',     os.path.join(RAW, 'faruk_okra')),
        ('ghose_cabbage',  os.path.join(RAW, 'ghose_cabbage')),
    ]
    for source_id, source_dir in TRAINING_SOURCES:
        if not os.path.isdir(source_dir):
            print(f"  [{source_id}] Directory not found — skipping. Run Step 03.")
            continue
        records = _scan_source(source_id, source_dir)
        all_records.extend(records)

    # misrak_veg has special filtering logic
    misrak_dir = os.path.join(RAW, 'misrak_veg')
    if os.path.isdir(misrak_dir):
        all_records.extend(_scan_misrak(misrak_dir))
    else:
        print("  [misrak_veg] Directory not found — skipping.")

    if not all_records:
        raise RuntimeError(
            "No training images found. Check that Step 03 (download) completed."
        )

    # ── Label assertion ────────────────────────────────────────────────────
    assert_all_labels_mapped(all_records)

    # ── Stratified split ───────────────────────────────────────────────────
    train_r, val_r, test_r = stratified_group_split(all_records, seed=42)

    # Collect all records including fixed-split sets
    split_map = {r['image_path']: r['split'] for r in train_r + val_r + test_r}
    for r in all_records:
        if r['image_path'] in split_map:
            r['split'] = split_map[r['image_path']]

    # ── PlantDoc (fixed split=plantdoc) ───────────────────────────────────
    if os.path.isdir(PLANTDOC_DIR):
        plantdoc_records = _scan_plantdoc(PLANTDOC_DIR)
        all_records.extend(plantdoc_records)

    # ── Kerala (fixed split=kerala) ───────────────────────────────────────
    kerala_records = _scan_kerala(KERALA_DIR)
    all_records.extend(kerala_records)

    # ── Write outputs ──────────────────────────────────────────────────────
    write_source_map(all_records)
    write_class_counts(all_records)

    # ── Print summary ──────────────────────────────────────────────────────
    from collections import Counter
    split_counts = Counter(r['split'] for r in all_records)
    class_counts = Counter(
        r['class_name'] for r in all_records if r['split'] == 'train'
    )
    print(f"\nSplit summary: {dict(split_counts)}")
    print("\nTraining class counts:")
    for cls, cnt in sorted(class_counts.items()):
        warn = " ← THIN" if cnt < 150 else ""
        print(f"  {cls:30s}: {cnt:5d}{warn}")
