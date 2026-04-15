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
    """
    Resolves a raw label string to a canonical class name.
    Check SOURCE_LABEL_OVERRIDES first (source-specific, exact folder match),
    then LABEL_MAP (global fuzzy mapping).
    Returns canonical class name or raises KeyError.
    """
    # Source-specific override: exact folder name match
    if source in SOURCE_LABEL_OVERRIDES:
        source_map = SOURCE_LABEL_OVERRIDES[source]
        if raw_label in source_map:
            return source_map[raw_label]
        # Folder not in this source's allowed list — skip it
        raise KeyError(f"Folder '{raw_label}' not in SOURCE_LABEL_OVERRIDES for source '{source}'")
    # Fallback: global LABEL_MAP (case-insensitive)
    normalised = raw_label.lower().strip()
    if normalised in LABEL_MAP:
        return LABEL_MAP[normalised]
    raise KeyError(f"No mapping for label='{raw_label}' from source='{source}'")


def assert_all_labels_mapped(records):
    unmapped = []
    # Sources that set class_name directly from manifests (no resolve_label needed)
    skip_sources = {'inaturalist_okra', 'inaturalist_chilli', 'inaturalist_tomato',
                    'inaturalist_brassica', 'plantdoc_train', 'plantdoc_eval'}
    for r in records:
        # Skip records with pre-resolved class names (from manifests)
        if r.get('source_dataset', '') in skip_sources:
            # Verify the class_name is valid
            if r.get('class_name', '') in CLASS_TO_IDX:
                continue
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
    # Scan the ENTIRE directory tree — os.walk handles nesting.
    # We use all splits (train, valid, test) from the source dataset.
    # Our own 70/15/15 split is applied later; the source's splits are ignored.
    scan_root = source_dir

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


def _scan_diya(diya_dir):
    """
    Scans asrafulme/dataset-diya which has structure:
    diya-dataset/{Crop}/{class_name}/images...
    or it may unzip directly to {Crop}/{class_name}/

    Splits into three source IDs: diya_broccoli, diya_cabbage, diya_cauliflower.
    Turnip and other non-mapped folders are silently skipped.
    """
    records = []

    # Find the root — could be diya_veg/diya-dataset/ or diya_veg/ directly
    scan_root = diya_dir
    diya_sub = os.path.join(diya_dir, 'diya-dataset')
    if os.path.isdir(diya_sub):
        scan_root = diya_sub

    crop_to_source = {
        'Broccoli': 'diya_broccoli',
        'broccoli': 'diya_broccoli',
        'Cabbage': 'diya_cabbage',
        'cabbage': 'diya_cabbage',
        'Cauliflower': 'diya_cauliflower',
        'cauliflower': 'diya_cauliflower',
    }

    for crop_folder in os.listdir(scan_root):
        crop_path = os.path.join(scan_root, crop_folder)
        if not os.path.isdir(crop_path):
            continue
        source_id = crop_to_source.get(crop_folder)
        if source_id is None:
            print(f"  [diya] Skipping non-target crop folder: {crop_folder}")
            continue

        for class_folder in os.listdir(crop_path):
            class_path = os.path.join(crop_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            # Check if this folder is in SOURCE_LABEL_OVERRIDES for this source
            try:
                class_name = resolve_label(class_folder, source_id)
                class_idx = CLASS_TO_IDX[class_name]
                crop_idx = CROP_FROM_IDX[class_idx]
            except (KeyError, TypeError):
                print(f"  [diya/{crop_folder}] Skipping unmapped folder: {class_folder}")
                continue

            for fname in os.listdir(class_path):
                ext = os.path.splitext(fname)[1]
                if ext not in VALID_EXT:
                    continue
                full_path = os.path.join(class_path, fname)
                try:
                    rel_path = os.path.relpath(full_path, ROOT).replace('\\', '/')
                except ValueError:
                    rel_path = full_path.replace('\\', '/')

                records.append({
                    'image_path': rel_path,
                    'source_dataset': source_id,
                    'raw_label': class_folder,
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'crop_idx': crop_idx,
                    'split': '',
                })

    print(f"  [diya_veg] Loaded {len(records)} images across broccoli/cabbage/cauliflower.")
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


def deduplicate_records(records, threshold=10):
    """
    Remove duplicate images using perceptual hash (pHash).

    Uses EXACT hash matching (set lookup, O(1) per image) for speed.
    Grouping by class_name so the same image in different classes is kept.
    Cross-source duplicates within the same class are removed.

    With 150K+ images, the old O(n^2) approach took hours.
    This O(n) approach takes ~5 minutes for 150K images.

    threshold parameter is kept for API compat but exact match is used.
    """
    try:
        import imagehash
        from PIL import Image as _PIL
    except ImportError:
        print("  imagehash not installed -- skipping deduplication.")
        return records

    from collections import defaultdict

    # Group by class_name — dedup within each class across all sources
    groups = defaultdict(list)
    for r in records:
        groups[r.get('class_name', '')].append(r)

    keep       = []
    total_dups = 0
    processed  = 0

    for cls, group_records in groups.items():
        seen = set()  # set of hash strings for O(1) lookup
        cls_dups = 0
        for r in group_records:
            full_path = os.path.join(ROOT, r['image_path'].replace('/', os.sep))
            try:
                img = _PIL.open(full_path).convert('RGB')
                h   = str(imagehash.phash(img))
                if h not in seen:
                    seen.add(h)
                    keep.append(r)
                else:
                    cls_dups += 1
            except Exception:
                keep.append(r)
            processed += 1
            if processed % 5000 == 0:
                print(f"    dedup progress: {processed}/{len(records)}...", flush=True)
        if cls_dups > 0:
            total_dups += cls_dups

    if total_dups > 0:
        print(f"  Deduplication removed {total_dups} near-duplicate images "
              f"({len(records)} -> {len(keep)}).")
    else:
        print(f"  Deduplication: no duplicates found.")
    return keep


def stratified_split(records, seed=42):
    """
    Splits records into train/val/test using StratifiedKFold (no source grouping).

    Why no source grouping: with only 4-5 sources, StratifiedGroupKFold forces
    entire single-source classes into one split, producing zero-image splits.
    pHash deduplication (run before this function) removes near-duplicates,
    mitigating the main data leakage risk from source-independent splitting.

    Target ratios: 70% train / 15% val / 15% test.
    Method:
      1. Hold out 15% as test (stratified by class_idx)
      2. Split remaining 85% into ~82% train / ~18% val (= 70/15 overall)

    Handles zero-count classes: classes with 0 images are excluded from
    stratification. The model keeps all 10 output neurons.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from collections import Counter

    pool = [r for r in records
            if r.get('split') not in ('plantdoc', 'kerala', 'domain_adapt')]
    if not pool:
        raise ValueError("No records available for splitting. Check data/raw/.")

    # Identify populated vs empty classes
    all_class_idx = [r['class_idx'] for r in pool]
    class_counts  = Counter(all_class_idx)
    populated_classes = {idx for idx, cnt in class_counts.items() if cnt > 0}
    empty_classes     = set()

    from app.config import NUM_CLASSES, CLASS_NAMES
    for idx in range(NUM_CLASSES):
        if class_counts.get(idx, 0) == 0:
            empty_classes.add(idx)

    if empty_classes:
        empty_names = [CLASS_NAMES[i] for i in sorted(empty_classes) if i < len(CLASS_NAMES)]
        print(f"  Zero-image classes (excluded from split): {empty_names}")

    # Filter to populated classes only
    split_pool = [r for r in pool if r['class_idx'] in populated_classes]
    if not split_pool:
        raise ValueError("No records with populated classes. Check data/raw/.")

    # Remap to contiguous indices for sklearn
    sorted_populated = sorted(populated_classes)
    orig_to_compact  = {orig: compact for compact, orig in enumerate(sorted_populated)}

    labels = np.array([orig_to_compact[r['class_idx']] for r in split_pool])

    min_class = int(np.bincount(labels).min())
    if min_class < 2:
        raise ValueError(
            f"Minimum populated class count is {min_class}. Need >= 2."
        )

    # Step 1: Hold out 15% as test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(labels, labels))

    # Step 2: Split remaining 85% into train (~82.4%) and val (~17.6%)
    # 0.15 / 0.85 = 0.1765 => ~15% of total becomes val
    trainval_labels = labels[trainval_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    train_sub, val_sub = next(sss_val.split(trainval_labels, trainval_labels))

    train_idx = trainval_idx[train_sub]
    val_idx   = trainval_idx[val_sub]

    train_r = [split_pool[i] for i in train_idx]
    val_r   = [split_pool[i] for i in val_idx]
    test_r  = [split_pool[i] for i in test_idx]

    for r in train_r: r['split'] = 'train'
    for r in val_r:   r['split'] = 'val'
    for r in test_r:  r['split'] = 'test'

    # Override: synthetic always in train
    for r in records:
        if r.get('source_dataset') == 'synthetic':
            r['split'] = 'train'
            if r not in train_r:
                train_r.append(r)

    # Report
    print(f"Split: {len(train_r)} train, {len(val_r)} val, {len(test_r)} test")
    print(f"  (populated classes: {len(populated_classes)}/10, "
          f"empty: {len(empty_classes)})")

    # Verify every populated class appears in every split
    for split_name, split_records in [('train', train_r), ('val', val_r), ('test', test_r)]:
        present = {r['class_idx'] for r in split_records}
        missing = populated_classes - present
        if missing:
            missing_names = [CLASS_NAMES[i] for i in sorted(missing)]
            print(f"  WARNING: {split_name} missing populated classes: {missing_names}")

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

    # ── Training datasets ─────────────────────────────────────────────────
    # gadde_okra: manojgadde/yellow-vein-mosaic-disease
    gadde_dir = os.path.join(RAW, 'gadde_okra')
    if os.path.isdir(gadde_dir):
        all_records.extend(_scan_source('gadde_okra', gadde_dir))
    else:
        print("  [gadde_okra] Directory not found — skipping. Run Step 03.")

    # cauliflower_noam: noamaanabdulazeem/cauliflower-dataset
    cauliflower_dir = os.path.join(RAW, 'cauliflower_noam')
    if os.path.isdir(cauliflower_dir):
        all_records.extend(_scan_source('cauliflower_noam', cauliflower_dir))
    else:
        print("  [cauliflower_noam] Directory not found — skipping.")

    # cabbage_balanced: mubbassir/balanced-cabbage-dataset-200-each
    cabbage_dir = os.path.join(RAW, 'cabbage_balanced')
    if os.path.isdir(cabbage_dir):
        all_records.extend(_scan_source('cabbage_balanced', cabbage_dir))
    else:
        print("  [cabbage_balanced] Directory not found — skipping.")

    # diya_veg: asrafulme/dataset-diya — splits into 3 sub-sources
    diya_dir = os.path.join(RAW, 'diya_veg')
    if os.path.isdir(diya_dir):
        all_records.extend(_scan_diya(diya_dir))
    else:
        print("  [diya_veg] Directory not found — skipping.")

    # ── New datasets (added after initial Step 03) ────────────────────────

    # yeesi: manhhoangvan/yeesidtaset — okra powdery mildew + healthy
    yeesi_dir = os.path.join(RAW, 'yeesi')
    if os.path.isdir(yeesi_dir):
        all_records.extend(_scan_source('yeesi', yeesi_dir))
    else:
        print("  [yeesi] Directory not found — skipping.")

    # mendeley_okra: Okra DiseaseNet — Training subfolder only
    mendeley_okra_dir = os.path.join(RAW, 'mendeley_okra')
    if os.path.isdir(mendeley_okra_dir):
        all_records.extend(_scan_source('mendeley_okra', mendeley_okra_dir))
    else:
        print("  [mendeley_okra] Directory not found — skipping.")

    # mendeley_caul_leaf: Cauliflower Leaf Diseases
    mendeley_caul_dir = os.path.join(RAW, 'mendeley_caul_leaf')
    if os.path.isdir(mendeley_caul_dir):
        all_records.extend(_scan_source('mendeley_caul_leaf', mendeley_caul_dir))
    else:
        print("  [mendeley_caul_leaf] Directory not found — skipping.")

    # mendeley_cabbage_dis: Cabbage Crop Diseases (originals only)
    mendeley_cab_dir = os.path.join(RAW, 'mendeley_cabbage_dis')
    if os.path.isdir(mendeley_cab_dir):
        all_records.extend(_scan_source('mendeley_cabbage_dis', mendeley_cab_dir))
    else:
        print("  [mendeley_cabbage_dis] Directory not found — skipping.")

    # caul_sharifashik: Cauliflower Image Dataset
    # Has top-level folder Disease_final_Dataset containing class subfolders
    sharifashik_dir = os.path.join(RAW, 'caul_sharifashik')
    if os.path.isdir(sharifashik_dir):
        # Check for Disease_final_Dataset subfolder
        inner = os.path.join(sharifashik_dir, 'Disease_final_Dataset')
        scan_dir = inner if os.path.isdir(inner) else sharifashik_dir
        all_records.extend(_scan_source('caul_sharifashik', scan_dir))
    else:
        print("  [caul_sharifashik] Directory not found — skipping.")

    # mendeley_vegnet: VegNet cauliflower (local zip extraction)
    vegnet_dir = os.path.join(RAW, 'mendeley_vegnet')
    if os.path.isdir(vegnet_dir):
        all_records.extend(_scan_source('mendeley_vegnet', vegnet_dir))
    else:
        print("  [mendeley_vegnet] Directory not found — skipping.")

    # ── Bangladesh Okra (Mendeley ck7vkp23c7) ─────────────────────────────
    bangladesh_dir = os.path.join(RAW, 'bangladesh_okra')
    if os.path.isdir(bangladesh_dir):
        all_records.extend(_scan_source('bangladesh_okra', bangladesh_dir))
    else:
        print("  [bangladesh_okra] Directory not found — skipping.")

    # ── Tomato datasets ───────────────────────────────────────────────────
    TOMATO_SOURCES = [
        ('plantvillage_tomato', os.path.join(RAW, 'plantvillage_tomato')),
        ('tomato_ashish',       os.path.join(RAW, 'tomato_ashish')),
        ('tomato_cookiefinder', os.path.join(RAW, 'tomato_cookiefinder')),
        ('tomato_luisolazo',    os.path.join(RAW, 'tomato_luisolazo')),
        ('tomato_hakim',        os.path.join(RAW, 'tomato_hakim')),
        ('tomato_kaustubh',     os.path.join(RAW, 'tomato_kaustubh')),
        ('tomato_mendeley',     os.path.join(RAW, 'tomato_mendeley')),
    ]
    for source_id, source_dir in TOMATO_SOURCES:
        if not os.path.isdir(source_dir):
            print(f"  [{source_id}] Directory not found — skipping.")
            continue
        all_records.extend(_scan_source(source_id, source_dir))

    # ── Chilli dataset ────────────────────────────────────────────────────
    CHILLI_SOURCES = [
        ('chilli_bangladesh',          os.path.join(RAW, 'chilli_bangladesh')),
        ('chilli_anthracnose_prudhvi', os.path.join(RAW, 'chilli_anthracnose_prudhvi')),
        ('chilli_cold_karnataka',      os.path.join(RAW, 'chilli_cold_karnataka')),
        ('chilli_bangladesh_2025',     os.path.join(RAW, 'chilli_bangladesh_2025')),
        ('chilli_annotated_smartphone',os.path.join(RAW, 'chilli_annotated_smartphone')),
    ]
    for source_id, source_dir in CHILLI_SOURCES:
        if not os.path.isdir(source_dir):
            print(f"  [{source_id}] Directory not found — skipping.")
            continue
        all_records.extend(_scan_source(source_id, source_dir))

    if not all_records:
        raise RuntimeError(
            "No training images found. Check that Step 03 (download) completed."
        )

    total_before = len(all_records)
    print(f"\nTotal images before Phase 0 modifications: {total_before}")

    # ── MOD 3: iNaturalist source scanning ────────────────────────────────
    # Load pseudo-labelled manifest and add iNaturalist images
    try:
        from app.config import INATURALIST_MANIFEST
        if os.path.exists(INATURALIST_MANIFEST):
            import pandas as _pd
            inat_df = _pd.read_csv(INATURALIST_MANIFEST)
            inat_count = 0
            for _, row in inat_df.iterrows():
                class_name = row.get('class_name', '')
                if class_name not in CLASS_TO_IDX:
                    continue
                class_idx = CLASS_TO_IDX[class_name]
                crop_idx = CROP_FROM_IDX[class_idx]
                all_records.append({
                    'image_path': row['image_path'],
                    'source_dataset': row.get('species', 'inaturalist'),
                    'raw_label': class_name,
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'crop_idx': crop_idx,
                    'split': '',
                })
                inat_count += 1
            print(f"  [iNaturalist] Added {inat_count} images from pseudo-labelled manifest.")
        else:
            print("  [iNaturalist] Manifest not found, skipping.")
    except ImportError:
        print("  [iNaturalist] Config constants not available, skipping.")

    # ── MOD 5: PlantDoc training images ───────────────────────────────────
    try:
        from app.config import PLANTDOC_TRAIN_MANIFEST
        if os.path.exists(PLANTDOC_TRAIN_MANIFEST):
            import pandas as _pd
            pd_train = _pd.read_csv(PLANTDOC_TRAIN_MANIFEST)
            pd_count = 0
            for _, row in pd_train.iterrows():
                class_name = row['class_name']
                if class_name not in CLASS_TO_IDX:
                    continue
                class_idx = CLASS_TO_IDX[class_name]
                crop_idx = CROP_FROM_IDX[class_idx]
                all_records.append({
                    'image_path': row['image_path'],
                    'source_dataset': 'plantdoc_train',
                    'raw_label': class_name,
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'crop_idx': crop_idx,
                    'split': '',
                })
                pd_count += 1
            print(f"  [PlantDoc] Added {pd_count} training images from manifest.")
    except ImportError:
        pass

    # ── Label assertion ────────────────────────────────────────────────────
    assert_all_labels_mapped(all_records)

    # ── MOD 2: Tomato capping ─────────────────────────────────────────────
    try:
        from app.config import TOMATO_CAP_ENABLED, TOMATO_CAP_CSV
        if TOMATO_CAP_ENABLED and os.path.exists(TOMATO_CAP_CSV):
            import pandas as _pd
            cap_df = _pd.read_csv(TOMATO_CAP_CSV)
            allowed_tomato_paths = set(cap_df['image_path'].tolist())
            tomato_classes = [c for c in CLASS_TO_IDX if c.startswith('tomato')]
            before_cap = len(all_records)
            # Keep non-tomato + allowed tomato + PlantDoc/iNaturalist tomato
            all_records = [
                r for r in all_records
                if (r['class_name'] not in tomato_classes) or
                   (r['image_path'] in allowed_tomato_paths) or
                   (r.get('source_dataset', '').startswith('inaturalist')) or
                   (r.get('source_dataset', '').startswith('plantdoc'))
            ]
            after_cap = len(all_records)
            print(f"  [Tomato Cap] {before_cap} -> {after_cap} (removed {before_cap - after_cap} tomato images)")
    except ImportError:
        pass

    # ── pHash deduplication ───────────────────────────────────────────────
    print("\nRunning pHash deduplication (threshold=10)...")
    all_records = deduplicate_records(all_records, threshold=10)

    # ── Stratified split (StratifiedShuffleSplit, no source grouping) ─────
    train_r, val_r, test_r = stratified_split(all_records, seed=42)

    # Collect all records including fixed-split sets
    split_map = {r['image_path']: r['split'] for r in train_r + val_r + test_r}
    for r in all_records:
        if r['image_path'] in split_map:
            r['split'] = split_map[r['image_path']]

    # ── MOD 1: Frozen test set enforcement ────────────────────────────────
    try:
        from app.config import FROZEN_TEST_SET_CSV
        if os.path.exists(FROZEN_TEST_SET_CSV):
            import pandas as _pd
            frozen = _pd.read_csv(FROZEN_TEST_SET_CSV)
            frozen_paths = set(frozen['image_path'].tolist())
            all_paths = {r['image_path'] for r in all_records}
            forced_count = 0
            missing_count = 0
            for r in all_records:
                if r['image_path'] in frozen_paths and r['split'] != 'test':
                    r['split'] = 'test'
                    forced_count += 1
            missing_count = len(frozen_paths - all_paths)
            print(f"  [Frozen Test] {forced_count} images forced to test split")
            print(f"  [Frozen Test] {missing_count} frozen images not in dataset (capped tomato)")
    except ImportError:
        pass

    # ── MOD 4: Training-only source enforcement ──────────────────────────
    try:
        from app.config import TRAINING_ONLY_SOURCES
        reassigned = 0
        for r in all_records:
            src = r.get('source_dataset', '')
            if src in TRAINING_ONLY_SOURCES and r['split'] != 'train':
                r['split'] = 'train'
                reassigned += 1
        if reassigned > 0:
            print(f"  [Training-Only] {reassigned} iNaturalist images forced to train split")
    except ImportError:
        pass

    # ── MOD 5b: PlantDoc eval enforcement ─────────────────────────────────
    try:
        from app.config import PLANTDOC_EVAL_MANIFEST
        if os.path.exists(PLANTDOC_EVAL_MANIFEST):
            import pandas as _pd
            pd_eval = _pd.read_csv(PLANTDOC_EVAL_MANIFEST)
            pd_eval_paths = set(pd_eval['image_path'].tolist())
            # Add PlantDoc eval images that aren't already in the dataset
            existing_paths = {r['image_path'] for r in all_records}
            for _, row in pd_eval.iterrows():
                if row['image_path'] not in existing_paths:
                    class_name = row['class_name']
                    if class_name in CLASS_TO_IDX:
                        all_records.append({
                            'image_path': row['image_path'],
                            'source_dataset': 'plantdoc_eval',
                            'raw_label': class_name,
                            'class_name': class_name,
                            'class_idx': CLASS_TO_IDX[class_name],
                            'crop_idx': CROP_FROM_IDX[CLASS_TO_IDX[class_name]],
                            'split': 'val',
                        })
            # Force existing PlantDoc eval images to val
            for r in all_records:
                if r['image_path'] in pd_eval_paths and r['split'] != 'val':
                    r['split'] = 'val'
            print(f"  [PlantDoc Eval] {len(pd_eval_paths)} images assigned to val split")
    except ImportError:
        pass

    # ── MOD 6: Flagged validation image removal ──────────────────────────
    try:
        from app.config import VAL_IMAGES_TO_REMOVE_TXT
        if os.path.exists(VAL_IMAGES_TO_REMOVE_TXT):
            with open(VAL_IMAGES_TO_REMOVE_TXT, 'r') as f:
                remove_paths = {line.strip() for line in f if line.strip()}
            if remove_paths:
                before_remove = len(all_records)
                all_records = [r for r in all_records if r['image_path'] not in remove_paths]
                after_remove = len(all_records)
                print(f"  [Val Audit] Removed {before_remove - after_remove} flagged images")
            else:
                print("  [Val Audit] No images flagged for removal (file empty)")
    except ImportError:
        pass

    # ── MOD 7: Comprehensive summary ──────────────────────────────────────
    from collections import Counter
    split_counts = Counter(r['split'] for r in all_records)
    print(f"\n{'='*60}")
    print(f"DATASET REBUILD SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(all_records)}")
    print(f"Train: {split_counts.get('train', 0)}")
    print(f"Val: {split_counts.get('val', 0)}")
    print(f"Test: {split_counts.get('test', 0)}")
    print(f"\nPer-crop training counts:")
    train_records = [r for r in all_records if r['split'] == 'train']
    for crop in ['okra', 'brassica', 'tomato', 'chilli']:
        count = sum(1 for r in train_records if r['class_name'].startswith(crop))
        print(f"  {crop}: {count}")
    crop_counts_train = {}
    for crop in ['okra', 'brassica', 'tomato', 'chilli']:
        crop_counts_train[crop] = sum(1 for r in train_records if r['class_name'].startswith(crop))
    if crop_counts_train:
        max_c = max(crop_counts_train.values())
        min_c = min(crop_counts_train.values())
        ratio = max_c / min_c if min_c > 0 else float('inf')
        print(f"  Crop imbalance ratio: {ratio:.2f}:1")

    # ── Write outputs ──────────────────────────────────────────────────────
    write_source_map(all_records)
    write_class_counts(all_records)

    # ── Print per-class summary ───────────────────────────────────────────
    class_counts = Counter(
        r['class_name'] for r in all_records if r['split'] == 'train'
    )
    print("\nTraining class counts:")
    for cls, cnt in sorted(class_counts.items()):
        warn = " <-- THIN" if cnt < 150 else ""
        print(f"  {cls:30s}: {cnt:5d}{warn}")
