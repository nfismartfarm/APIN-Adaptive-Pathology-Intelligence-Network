"""
Rebuild unified source-of-truth CSVs from the FILESYSTEM rather than from
Source A + Source B CSV joins. This is more reliable because it captures
all files that physically exist in cleaned/ directories, including those
from prior integrations that aren't tracked in either Source A or Source B.

For each file in cleaned/{class}/:
  - The class is determined by the parent directory name
  - The crop is derived from the class name prefix
  - source_dataset/source_code/is_field_photo are inferred from filename prefix
  - origin is set based on filename pattern
"""
import os
import sys
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(r'C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger')
MODEL2_CLEANED = ROOT / 'data' / 'specialist' / 'model2' / 'cleaned'
MODEL3_CLEANED = ROOT / 'data' / 'specialist' / 'model3' / 'cleaned'
ROUTER_CLEANED = ROOT / 'data' / 'specialist' / 'router' / 'cleaned'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

MODEL2_CLASSES = ['okra_yvmv', 'okra_powdery_mildew', 'okra_cercospora', 'okra_enation',
                  'okra_healthy', 'brassica_black_rot', 'brassica_downy_mildew',
                  'brassica_alternaria', 'brassica_healthy']
# tomato_target_spot REMOVED — quarantined (labelling suspect, 539 images)
MODEL3_CLASSES = ['tomato_foliar_spot', 'tomato_late_blight', 'tomato_septoria_leaf_spot',
                  'tomato_yellow_leaf_curl_virus', 'tomato_mosaic_virus', 'tomato_healthy',
                  'chilli_leaf_curl', 'chilli_healthy',
                  'chilli_cercospora_leaf_spot', 'chilli_anthracnose']


def class_to_crop(class_name):
    if class_name.startswith('okra_'):     return 'okra'
    if class_name.startswith('brassica_'): return 'brassica'
    if class_name.startswith('tomato_'):   return 'tomato'
    if class_name.startswith('chilli_'):   return 'chilli'
    return None


# Filename prefix → (source_dataset, source_code, is_field_photo, origin)
# These are the conventions established across earlier integrations.
PREFIX_LOOKUP = {
    # Original pool (from data/raw/ via consolidation script)
    'orig_': ('original_pool', 'orig', None, 'original_pool'),  # is_field set per dataset later
    # New brassica/chilli integration
    'mendeley_A': ('mendeley_A', 'mendeley_A', True, 'new_integration'),
    'balanced_B': ('balanced_B', 'balanced_B', False, 'new_integration'),
    'plantwild_C': ('plantwild_C', 'plantwild_C', True, 'new_integration'),
    'multi_D': ('multi_D', 'multi_D', True, 'new_integration'),
    # Earlier okra integrations
    'srcH_': ('okra_100', 'srcH', False, 'okra_integration'),
    'srcI_': ('leavesbank_okra', 'srcI', False, 'okra_integration'),
    'srcJ_': ('okra_disease_field', 'srcJ', True, 'okra_integration'),
    'srcK_': ('okra_roboflow', 'srcK', False, 'okra_integration'),
    # Older chilli/tomato integrations
    'srcG_': ('chilli_final_dataset', 'srcG', True, 'chilli_integration'),
    'srcF_': ('huggingface_waruni', 'srcF', True, 'chilli_integration'),
    'srcE_': ('multicrop_tamilnadu', 'srcE', True, 'chilli_integration'),
    'srcC_': ('model3_cleaned', 'srcC', True, 'chilli_integration'),
    'srcA_': ('source_map_swin', 'srcA', False, 'tomato_integration'),
    'src1_': ('mendeley_bangladesh_tomato', 'src1', False, 'tomato_integration'),
    'src2_': ('scidb_data_merged', 'src2', False, 'tomato_integration'),
    'src3_': ('tomato_village', 'src3', True, 'tomato_integration'),
    'src4_': ('bangladesh_field', 'src4', True, 'tomato_integration'),
    'src5_': ('tomato_leaf_multiclass', 'src5', False, 'tomato_integration'),
    'sourceA_': ('plantvillage_existing', 'sourceA', False, 'tomato_integration'),
    'sourceB_': ('figshare_leaf_curl', 'sourceB', True, 'tomato_integration'),
    'sourceC_': ('mendeley_bangladesh_chilli', 'sourceC', True, 'chilli_integration'),
    'sourceD_': ('multicrop_tamilnadu', 'sourceD', True, 'chilli_integration'),
    'sourceE_': ('kaggle_anthracnose', 'sourceE', True, 'chilli_integration'),
    'sourceF_': ('huggingface_waruni', 'sourceF', True, 'chilli_integration'),
    # Taiwan + iNaturalist tomato field integration
    'taiwan_': ('tomato_taiwan', 'taiwan', None, 'taiwan_integration'),  # is_field per-image
    'inat_tomato_': ('inaturalist_tomato', 'inat', True, 'inaturalist'),
    # Background recomposition (Phase 0 Step 0.10)
    'recomp_scidb_': ('scidb_recomposed', 'recomp_scidb', True, 'recomposition'),
    'recomp_capsicum_': ('capsicum_recomposed', 'recomp_cap', True, 'recomposition'),
}

# Field/lab determination for orig_ files based on the dataset name embedded in filename
ORIG_FIELD_KEYWORDS = ['plantdoc', 'inaturalist', 'bangladesh', 'gadde', 'yeesi',
                       'kerala', 'mendeley_caul', 'mendeley_cabbage', 'mendeley_okra',
                       'cauliflower_noam', 'caul_sharifashik', 'tomato_luisolazo',
                       'diya_', 'cabbage_balanced', 'chilli_cold_karnataka',
                       'chilli_anthracnose_prudhvi']
ORIG_LAB_KEYWORDS = ['plantvillage', 'tomato_ashish', 'tomato_kaustubh']


def infer_metadata(filename):
    """Given a filename, infer source_dataset, source_code, is_field_photo, origin."""
    for prefix, (sd, sc, field, origin) in PREFIX_LOOKUP.items():
        if filename.startswith(prefix):
            if prefix == 'orig_':
                # Determine field/lab from the rest of the filename
                rest = filename[len(prefix):]
                lower = rest.lower()
                if any(k in lower for k in ORIG_LAB_KEYWORDS):
                    return ('original_pool', 'orig', False, 'original_pool')
                if any(k in lower for k in ORIG_FIELD_KEYWORDS):
                    return ('original_pool', 'orig', True, 'original_pool')
                return ('original_pool', 'orig', False, 'original_pool')
            if prefix == 'taiwan_':
                # Taiwan field/lab decision per class:
                # health, Late_blight, Gray_spot -> field (confirmed field photos)
                # Bacterial_spot, Black_mold, powdery_mildew -> mixed, mark False
                lower = filename.lower()
                if any(k in lower for k in ['_health_', '_late_blight_', '_gray_spot_']):
                    return ('tomato_taiwan', 'taiwan', True, 'taiwan_integration')
                return ('tomato_taiwan', 'taiwan', False, 'taiwan_integration')
            return (sd, sc, field, origin)
    return ('unknown', 'unknown', False, 'unknown')


def scan_disease_directory(base_dir, class_list, model_name):
    """Scan a model's cleaned/ directory and return rows for the unified CSV."""
    rows = []
    for cls in class_list:
        cls_dir = base_dir / cls
        if not cls_dir.exists():
            continue
        crop = class_to_crop(cls)
        for f in cls_dir.iterdir():
            if f.suffix.lower() not in IMG_EXT:
                continue
            sd, sc, field, origin = infer_metadata(f.name)
            rows.append({
                'image_path': str(f),
                'class_name': cls,
                'crop': crop,
                'source_dataset': sd,
                'source_code': sc,
                'is_field_photo': field if field is not None else False,
                'origin': origin,
                'split': 'train',
            })
    return rows


def scan_router_directory(base_dir):
    """Scan router cleaned/ directory and return rows."""
    rows = []
    for crop_dir in base_dir.iterdir():
        if not crop_dir.is_dir():
            continue
        crop = crop_dir.name
        for f in crop_dir.iterdir():
            if f.suffix.lower() not in IMG_EXT:
                continue
            sd, sc, field, origin = infer_metadata(f.name)
            rows.append({
                'image_path': str(f),
                'crop': crop,
                'source_dataset': sd,
                'source_code': sc,
                'is_field_photo': field if field is not None else False,
                'origin': origin,
            })
    return rows


print('=' * 80)
print('REBUILDING UNIFIED CSVs FROM FILESYSTEM')
print('=' * 80)
print()

# Model 2
print('Scanning Model 2...')
m2_rows = scan_disease_directory(MODEL2_CLEANED, MODEL2_CLASSES, 'model2')
unified_m2 = pd.DataFrame(m2_rows)
m2_path = MODEL2_CLEANED.parent / 'model2_unified_source_map.csv'
unified_m2.to_csv(m2_path, index=False)
print(f'  Saved {len(unified_m2)} rows to {m2_path.name}')
print(f'  Class breakdown:')
for cls in MODEL2_CLASSES:
    n = (unified_m2['class_name'] == cls).sum()
    field_n = ((unified_m2['class_name'] == cls) &
               (unified_m2['is_field_photo'] == True)).sum()
    field_pct = field_n / n * 100 if n > 0 else 0
    n_src = unified_m2[unified_m2['class_name'] == cls]['source_dataset'].nunique()
    marker = ' <<THIN' if n < 300 else ''
    print(f'    {cls:<35} {n:>6}  {field_pct:>5.0f}% field  {n_src:>3} sources{marker}')
print()

# Model 3
print('Scanning Model 3 cleaned...')
m3_rows = scan_disease_directory(MODEL3_CLEANED, MODEL3_CLASSES, 'model3')

# Also scan recomposed directory (Phase 0 Step 0.10 output)
MODEL3_RECOMPOSED = MODEL3_CLEANED.parent / 'recomposed'
if MODEL3_RECOMPOSED.exists():
    print('Scanning Model 3 recomposed...')
    # Recomposed images use the same class names as cleaned
    recomp_classes = [d.name for d in MODEL3_RECOMPOSED.iterdir()
                      if d.is_dir() and d.name in MODEL3_CLASSES]
    recomp_rows = scan_disease_directory(MODEL3_RECOMPOSED, recomp_classes, 'model3')
    m3_rows.extend(recomp_rows)
    print(f'  Added {len(recomp_rows)} recomposed images')
else:
    print('  No recomposed directory found')

unified_m3 = pd.DataFrame(m3_rows)
m3_path = MODEL3_CLEANED.parent / 'model3_unified_source_map.csv'
unified_m3.to_csv(m3_path, index=False)
print(f'  Saved {len(unified_m3)} rows to {m3_path.name}')
print(f'  Class breakdown:')
for cls in MODEL3_CLASSES:
    n = (unified_m3['class_name'] == cls).sum()
    field_n = ((unified_m3['class_name'] == cls) &
               (unified_m3['is_field_photo'] == True)).sum()
    field_pct = field_n / n * 100 if n > 0 else 0
    n_src = unified_m3[unified_m3['class_name'] == cls]['source_dataset'].nunique()
    print(f'    {cls:<35} {n:>6}  {field_pct:>5.0f}% field  {n_src:>3} sources')
print()

# Router
print('Scanning Router...')
router_rows = scan_router_directory(ROUTER_CLEANED)
unified_router = pd.DataFrame(router_rows)
router_path = ROUTER_CLEANED.parent / 'router_unified_source_map.csv'
unified_router.to_csv(router_path, index=False)
print(f'  Saved {len(unified_router)} rows to {router_path.name}')
print(f'  Crop breakdown:')
crop_counts = {}
for crop in ['okra', 'brassica', 'tomato', 'chilli']:
    n = (unified_router['crop'] == crop).sum()
    field_n = ((unified_router['crop'] == crop) &
               (unified_router['is_field_photo'] == True)).sum()
    field_pct = field_n / n * 100 if n > 0 else 0
    crop_counts[crop] = n
    print(f'    {crop:<12} {n:>8}  {field_pct:>5.0f}% field')
print(f'  TOTAL: {len(unified_router)}')
if crop_counts and min(crop_counts.values()) > 0:
    print(f'  Imbalance ratio: {max(crop_counts.values())/min(crop_counts.values()):.1f}:1')
print()

# Verification: filesystem vs CSV reconciliation
print('=' * 80)
print('VERIFICATION: Filesystem vs CSV reconciliation')
print('=' * 80)
mismatches = 0
print('Model 2:')
for cls in MODEL2_CLASSES:
    cls_dir = MODEL2_CLEANED / cls
    fs = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT) if cls_dir.exists() else 0
    csv_n = (unified_m2['class_name'] == cls).sum()
    diff = abs(fs - csv_n)
    marker = ' MISMATCH' if diff > 0 else ' OK'
    if diff > 0:
        mismatches += 1
    print(f'  {cls:<35} disk={fs:>6} csv={csv_n:>6}{marker}')

print('Model 3:')
for cls in MODEL3_CLASSES:
    cls_dir = MODEL3_CLEANED / cls
    fs = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT) if cls_dir.exists() else 0
    csv_n = (unified_m3['class_name'] == cls).sum()
    diff = abs(fs - csv_n)
    marker = ' MISMATCH' if diff > 0 else ' OK'
    if diff > 0:
        mismatches += 1
    print(f'  {cls:<35} disk={fs:>6} csv={csv_n:>6}{marker}')

print('Router:')
for crop in ['okra', 'brassica', 'tomato', 'chilli']:
    crop_dir = ROUTER_CLEANED / crop
    fs = sum(1 for f in crop_dir.iterdir() if f.suffix.lower() in IMG_EXT) if crop_dir.exists() else 0
    csv_n = (unified_router['crop'] == crop).sum()
    diff = abs(fs - csv_n)
    marker = ' MISMATCH' if diff > 0 else ' OK'
    if diff > 0:
        mismatches += 1
    print(f'  {crop:<35} disk={fs:>6} csv={csv_n:>6}{marker}')

print()
print(f'Total mismatches: {mismatches}')
print()
print('Quarantine check:')
qd = MODEL2_CLEANED / 'brassica_clubroot_QUARANTINED'
if qd.exists():
    qn = sum(1 for f in qd.iterdir() if f.suffix.lower() in IMG_EXT)
    print(f'  brassica_clubroot_QUARANTINED: {qn} preserved')
else:
    print(f'  brassica_clubroot_QUARANTINED: NOT FOUND')
print(f'  brassica_clubroot in unified Model 2: {(unified_m2["class_name"] == "brassica_clubroot").sum()}')
print()

print('FILES UPDATED:')
print(f'  {m2_path}')
print(f'  {m3_path}')
print(f'  {router_path}')
print()

# Model integrity
print('MODEL INTEGRITY CHECK:')
for name, min_mb in [('best_model.pt', 84), ('swin_best_model.pt', 114)]:
    p = ROOT / 'models' / name
    sz = p.stat().st_size / 1e6 if p.exists() else 0
    status = 'INTACT' if sz >= min_mb else 'PROBLEM'
    print(f'  {name}: {sz:.1f}MB [{status}]')
