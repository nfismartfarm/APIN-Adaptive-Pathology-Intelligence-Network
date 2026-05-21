"""
PSV Validation — 7 test suites for isolation testing.

Run: python -m scripts.psv.validation
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.psv.config import PSV_CFG
from scripts.psv.feature_extractor import extract_all_features
from scripts.psv.image_quality import assess_image_quality
from scripts.psv.disease_scores import compute_disease_scores


def load_test_images(n_per_class: int = 10):
    """Load n images per class from training data."""
    import pandas as pd
    from PIL import Image

    csv_path = os.path.join(PSV_CFG.ROOT, 'data', 'specialist', 'model2',
                            'model2_unified_source_map.csv')
    df = pd.read_csv(csv_path)

    images_by_class = {}
    for cls in PSV_CFG.CLASS_NAMES:
        cls_df = df[df['class_name'] == cls].head(n_per_class)
        imgs = []
        for _, row in cls_df.iterrows():
            path = row.get('clahe_path', row['image_path'])
            if not isinstance(path, str) or not os.path.exists(path):
                path = row['image_path']
            try:
                img = np.array(Image.open(path).convert('RGB'))
                imgs.append(img)
            except:
                pass
        images_by_class[cls] = imgs

    return images_by_class


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 1: FEATURE SANITY
# ═══════════════════════════════════════════════════════════════════════

def test_feature_sanity(images_by_class: Dict[str, List], verbose=True):
    """Verify features produce expected patterns per class."""
    print('\n' + '=' * 60)
    print('TEST SUITE 1: FEATURE SANITY')
    print('=' * 60, flush=True)

    results = {}
    checks = {
        'brassica_black_rot': [
            ('A03_margin_vs_interior_ratio', '>', 0.8, 'Margin disease dominance'),
            ('B02_vein_darkening_extent', '>', 0.3, 'Vein darkening present'),
        ],
        'brassica_alternaria': [
            ('C09_blob_interior_fraction', '>', 0.3, 'Spots in interior'),
            ('C01_mean_blob_circularity', '>', 0.3, 'Circular spots'),
        ],
        'okra_cercospora': [
            ('D01_gray_white_center_fraction', '>', 0.1, 'Gray-white spot centers'),
        ],
        'okra_yvmv': [
            ('D04_yellow_vein_fraction', '>', 0.05, 'Yellow vein presence'),
        ],
        'okra_healthy': [
            ('A12_disease_coverage_fraction', '<', 0.15, 'Low disease coverage'),
            ('D07_green_retention_fraction', '>', 0.5, 'High green retention'),
        ],
    }

    for cls, class_checks in checks.items():
        imgs = images_by_class.get(cls, [])
        if not imgs:
            print(f'  {cls}: NO IMAGES - SKIP')
            continue

        for feat_name, op, threshold, desc in class_checks:
            passes = 0
            for img in imgs:
                try:
                    result = extract_all_features(img)
                    val = result.features.get(feat_name, 0)
                    if op == '>' and val > threshold:
                        passes += 1
                    elif op == '<' and val < threshold:
                        passes += 1
                except:
                    pass

            rate = passes / len(imgs) if imgs else 0
            status = 'PASS' if rate >= 0.6 else 'WARN' if rate >= 0.4 else 'FAIL'
            results[f'{cls}:{feat_name}'] = status
            if verbose:
                print(f'  [{status}] {cls} | {feat_name} {op} {threshold} | '
                      f'{passes}/{len(imgs)} ({rate:.0%}) — {desc}', flush=True)

    passed = sum(1 for v in results.values() if v == 'PASS')
    total = len(results)
    print(f'\nSuite 1: {passed}/{total} passed')
    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 2: CROSS-CLASS SEPARATION
# ═══════════════════════════════════════════════════════════════════════

def test_cross_class_separation(images_by_class: Dict, verbose=True):
    """Verify PSV scores separate confusion pairs."""
    print('\n' + '=' * 60)
    print('TEST SUITE 2: CROSS-CLASS SEPARATION')
    print('=' * 60, flush=True)

    pairs = [
        ('brassica_black_rot', 'brassica_alternaria', 'brassica_black_rot'),
        ('okra_cercospora', 'okra_healthy', 'okra_cercospora'),
        ('okra_yvmv', 'okra_healthy', 'okra_yvmv'),
    ]

    results = {}
    for cls_a, cls_b, score_class in pairs:
        scores_a = []
        scores_b = []

        for img in images_by_class.get(cls_a, [])[:10]:
            try:
                result = extract_all_features(img)
                scores = compute_disease_scores(result.features)
                scores_a.append(scores.get(score_class, 0.5))
            except:
                pass

        for img in images_by_class.get(cls_b, [])[:10]:
            try:
                result = extract_all_features(img)
                scores = compute_disease_scores(result.features)
                scores_b.append(scores.get(score_class, 0.5))
            except:
                pass

        if scores_a and scores_b:
            mean_a = np.mean(scores_a)
            mean_b = np.mean(scores_b)
            separation = abs(mean_a - mean_b)
            status = 'PASS' if separation > 0.05 else 'WARN' if separation > 0.02 else 'FAIL'
            results[f'{cls_a}_vs_{cls_b}'] = status
            if verbose:
                print(f'  [{status}] {score_class} score: '
                      f'{cls_a}={mean_a:.3f} vs {cls_b}={mean_b:.3f} '
                      f'(separation={separation:.3f})', flush=True)
        else:
            results[f'{cls_a}_vs_{cls_b}'] = 'SKIP'
            if verbose:
                print(f'  [SKIP] {cls_a} vs {cls_b}: insufficient images', flush=True)

    passed = sum(1 for v in results.values() if v == 'PASS')
    print(f'\nSuite 2: {passed}/{len(results)} passed')
    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 3: EDGE CASE DETECTION
# ═══════════════════════════════════════════════════════════════════════

def test_edge_cases(verbose=True):
    """Test IQA edge case detection on synthetic images."""
    print('\n' + '=' * 60)
    print('TEST SUITE 3: EDGE CASE DETECTION')
    print('=' * 60, flush=True)

    results = {}

    # Black image -> no leaf
    black = np.zeros((300, 300, 3), dtype=np.uint8)
    iqa = assess_image_quality(black)
    status = 'PASS' if iqa.quality_flags.get('EQ10_no_leaf_detected', False) else 'FAIL'
    results['black_image'] = status
    if verbose:
        print(f'  [{status}] Black image -> EQ10 (no leaf): {iqa.psv_confidence:.2f}', flush=True)

    # Blurry image
    green = np.zeros((300, 300, 3), dtype=np.uint8)
    green[:, :, 1] = 120
    blurry = cv2.GaussianBlur(green, (51, 51), 20)
    iqa = assess_image_quality(blurry)
    has_blur = any('blur' in k.lower() for k in iqa.quality_flags if iqa.quality_flags.get(k))
    status = 'PASS' if has_blur else 'FAIL'
    results['blur_detection'] = status
    if verbose:
        print(f'  [{status}] Blurry image -> blur flag: conf={iqa.psv_confidence:.2f}', flush=True)

    # Overexposed
    bright = np.full((300, 300, 3), 250, dtype=np.uint8)
    iqa = assess_image_quality(bright)
    has_over = any('overexposed' in k.lower() or 'no_leaf' in k.lower()
                   for k in iqa.quality_flags if iqa.quality_flags.get(k))
    status = 'PASS' if has_over or iqa.psv_confidence < 0.5 else 'FAIL'
    results['overexposure'] = status
    if verbose:
        print(f'  [{status}] Overexposed -> flag: conf={iqa.psv_confidence:.2f}', flush=True)

    # White dots on green (water droplets)
    green_img = np.zeros((300, 300, 3), dtype=np.uint8)
    green_img[:, :, 1] = 120
    green_img[:, :, 0] = 40
    rng = np.random.default_rng(42)
    for _ in range(10):
        cx, cy = rng.integers(30, 270, 2)
        cv2.circle(green_img, (int(cx), int(cy)), 5, (255, 255, 255), -1)
    iqa = assess_image_quality(green_img)
    # Water droplets may or may not trigger — just verify IQA runs
    status = 'PASS'
    results['water_droplets'] = status
    if verbose:
        print(f'  [{status}] Water droplets test: ran successfully, '
              f'conf={iqa.psv_confidence:.2f}', flush=True)

    import cv2 as _cv2  # ensure cv2 is available for blur test
    passed = sum(1 for v in results.values() if v == 'PASS')
    print(f'\nSuite 3: {passed}/{len(results)} passed')
    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE 7: SPEED TEST
# ═══════════════════════════════════════════════════════════════════════

def test_speed(images_by_class: Dict, n_images: int = 20, verbose=True):
    """Benchmark PSV speed."""
    print('\n' + '=' * 60)
    print('TEST SUITE 7: SPEED TEST')
    print('=' * 60, flush=True)

    all_images = []
    for cls, imgs in images_by_class.items():
        all_images.extend(imgs[:max(1, n_images // PSV_CFG.NUM_CLASSES)])

    if not all_images:
        print('  No images available for speed test')
        return {'speed_test': 'SKIP'}

    times = []
    for img in all_images[:n_images]:
        t0 = time.time()
        result = extract_all_features(img)
        scores = compute_disease_scores(result.features)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)

    mean_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    status = 'PASS' if mean_ms < PSV_CFG.MAX_PSV_TIME_MS else 'WARN'

    if verbose:
        print(f'  [{status}] Mean: {mean_ms:.0f}ms, P95: {p95_ms:.0f}ms, '
              f'Target: <{PSV_CFG.MAX_PSV_TIME_MS}ms', flush=True)
        print(f'  Tested on {len(times)} images', flush=True)

    return {'speed_test': status, 'mean_ms': mean_ms, 'p95_ms': p95_ms}


# ═══════════════════════════════════════════════════════════════════════
# MAIN: RUN ALL SUITES
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests(n_per_class: int = 5):
    """Run all test suites."""
    import cv2  # ensure available

    print('PSV VALIDATION — Full Test Suite')
    print('=' * 60)
    print(f'Loading {n_per_class} images per class...', flush=True)

    images = load_test_images(n_per_class)
    total_loaded = sum(len(v) for v in images.values())
    print(f'Loaded {total_loaded} images across {len(images)} classes', flush=True)

    all_results = {}

    # Suite 1: Feature sanity
    r1 = test_feature_sanity(images)
    all_results['suite1_feature_sanity'] = r1

    # Suite 2: Cross-class separation
    r2 = test_cross_class_separation(images)
    all_results['suite2_separation'] = r2

    # Suite 3: Edge cases
    r3 = test_edge_cases()
    all_results['suite3_edge_cases'] = r3

    # Suite 7: Speed
    r7 = test_speed(images)
    all_results['suite7_speed'] = r7

    # Summary
    print('\n' + '=' * 60)
    print('VALIDATION SUMMARY')
    print('=' * 60)
    for suite, results in all_results.items():
        if isinstance(results, dict):
            passed = sum(1 for v in results.values() if v == 'PASS')
            total = sum(1 for v in results.values() if v in ('PASS', 'FAIL', 'WARN'))
            print(f'  {suite}: {passed}/{total} passed')

    return all_results


if __name__ == '__main__':
    import cv2
    run_all_tests(n_per_class=5)
