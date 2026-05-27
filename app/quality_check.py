# app/quality_check.py
"""Pre-submission image quality checker. Runs before the model to catch poor images."""
import numpy as np


def check_image_quality(image_np):
    """
    Evaluate image quality along 4 dimensions.
    Returns dict with overall_score, passed, issues, feedback, dimensions.
    """
    H, W = image_np.shape[:2]
    issues = []
    dimension_scores = {}

    # Size
    min_dim = min(H, W)
    if min_dim < 100:
        size_score = 0.0
        issues.append('Image too small (minimum 100x100 pixels required)')
    elif min_dim < 200:
        size_score = 0.5
        issues.append('Image resolution is low — use a higher quality photo')
    else:
        size_score = 1.0
    dimension_scores['size'] = size_score

    # Blur (Laplacian variance)
    try:
        import cv2
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    except ImportError:
        gray = np.mean(image_np, axis=2)
        gy, gx = np.gradient(gray.astype(float))
        laplacian_var = np.var(np.sqrt(gx**2 + gy**2))

    if laplacian_var < 30:
        blur_score = 0.0
        issues.append('Image is too blurry — hold the camera steady and retake')
    elif laplacian_var < 80:
        blur_score = 0.5
        issues.append('Image is slightly blurry — a sharper photo will give better results')
    else:
        blur_score = min(1.0, laplacian_var / 200)
    dimension_scores['sharpness'] = blur_score

    # Exposure
    mean_brightness = float(np.mean(image_np))
    if mean_brightness < 40:
        exposure_score = 0.0
        issues.append('Image is too dark — move to better lighting or use flash')
    elif mean_brightness < 60:
        exposure_score = 0.5
        issues.append('Image is underexposed — better lighting will improve accuracy')
    elif mean_brightness > 220:
        exposure_score = 0.0
        issues.append('Image is overexposed — avoid direct sunlight on the leaf')
    elif mean_brightness > 200:
        exposure_score = 0.5
        issues.append('Image is slightly overexposed — find shade for the photo')
    else:
        exposure_score = 1.0
    dimension_scores['exposure'] = exposure_score

    # Leaf coverage (green + brown/yellow for diseased)
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    green_mask = (g.astype(int) > r.astype(int) + 10) & \
                 (g.astype(int) > b.astype(int) + 10) & (g > 30)
    brown_mask = (r.astype(int) > g.astype(int)) & (r > 60) & (g > 30) & (b < 100)
    yellow_mask = (r > 150) & (g > 150) & (b < 100)
    leaf_fraction = float(green_mask.mean()) + float(brown_mask.mean()) * 0.5 + \
                    float(yellow_mask.mean()) * 0.5

    if leaf_fraction < 0.15:
        leaf_score = 0.0
        issues.append('Very little leaf visible — ensure the leaf fills most of the frame')
    elif leaf_fraction < 0.30:
        leaf_score = 0.5
        issues.append('Leaf coverage is low — try to fill 60-70% of the frame with the leaf')
    else:
        leaf_score = min(1.0, leaf_fraction / 0.70)
    dimension_scores['leaf_coverage'] = leaf_fraction

    overall = (0.35 * blur_score + 0.30 * min(1.0, leaf_score) +
               0.20 * exposure_score + 0.15 * size_score)

    if overall >= 0.75 and not issues:
        feedback = 'Good photo quality. Proceeding with analysis.'
    elif overall >= 0.5:
        feedback = 'Acceptable photo. ' + (issues[0] if issues else '')
    else:
        feedback = 'Poor photo quality. ' + ' '.join(issues[:2])

    passed = overall >= 0.35 and size_score > 0 and blur_score > 0

    return {
        'overall_score': round(overall, 3),
        'passed': passed,
        'issues': issues,
        'feedback': feedback,
        'dimensions': {
            'sharpness': round(blur_score, 3),
            'exposure': round(exposure_score, 3),
            'leaf_coverage': round(leaf_fraction, 3),
            'size_ok': size_score > 0,
        }
    }
