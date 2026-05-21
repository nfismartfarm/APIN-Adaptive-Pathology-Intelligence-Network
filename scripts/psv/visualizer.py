"""
PSV Debug Visualizer — 8-panel diagnostic output for any input image.
"""

import os
import numpy as np
import cv2
from typing import Dict
from scripts.psv.config import PSV_CFG


def create_debug_visualization(image_rgb: np.ndarray, features: Dict[str, float],
                                scores: Dict[str, float], shared_maps,
                                quality_flags: Dict[str, bool] = None,
                                psv_confidence: float = 1.0,
                                save_path: str = None):
    """
    Create 8-panel debug visualization.

    Panels:
      1. Original with leaf mask overlay
      2. Disease mask
      3. Distance map (blue=edge, red=center)
      4. Frangi vein map
      5. Blob detection results
      6. Feature bar chart (grouped)
      7. Per-class PSV score bars
      8. IQA flags summary
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print('matplotlib not available for visualization')
        return

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('PSV Debug Visualization', fontsize=16, fontweight='bold')

    m = shared_maps

    # Panel 1: Original + leaf mask
    ax = axes[0, 0]
    ax.imshow(m.original_rgb)
    if m.leaf_mask.any():
        overlay = np.zeros((*m.leaf_mask.shape, 4))
        overlay[~m.leaf_mask] = [0, 0, 0, 0.5]  # darken non-leaf
        ax.imshow(overlay)
    ax.set_title(f'Leaf Mask (area={m.leaf_area}px)')
    ax.axis('off')

    # Panel 2: Disease mask
    ax = axes[0, 1]
    vis = np.zeros((*m.disease_mask.shape, 3), dtype=np.uint8)
    vis[m.leaf_mask & ~m.disease_mask] = [0, 150, 0]  # healthy = green
    vis[m.disease_mask] = [200, 50, 50]                 # disease = red
    vis[m.specular_mask] = [255, 255, 255]              # specular = white
    ax.imshow(vis)
    ax.set_title(f'Disease Mask ({m.disease_area}px)')
    ax.axis('off')

    # Panel 3: Distance map
    ax = axes[0, 2]
    ax.imshow(m.distance_map, cmap='coolwarm')
    ax.set_title('Distance Map (blue=edge, red=center)')
    ax.axis('off')

    # Panel 4: Frangi vein map
    ax = axes[0, 3]
    vein_vis = np.zeros((*m.frangi_map.shape, 3))
    vein_vis[:, :, 0] = np.clip(m.frangi_map / (m.frangi_map.max() + 1e-8), 0, 1)
    vein_vis[:, :, 1] = vein_vis[:, :, 0] * 0.5
    ax.imshow(vein_vis)
    ax.set_title('Frangi Vein Map')
    ax.axis('off')

    # Panel 5: Blob detection
    ax = axes[1, 0]
    ax.imshow(m.original_rgb)
    for r in m.blob_regions[:50]:
        cy, cx = r.centroid
        radius = np.sqrt(r.area / np.pi)
        circle = Circle((cx, cy), radius, fill=False, color='yellow', linewidth=1.5)
        ax.add_patch(circle)
    ax.set_title(f'Blobs Detected ({len(m.blob_regions)})')
    ax.axis('off')

    # Panel 6: Feature bar chart (top 20 by absolute value)
    ax = axes[1, 1]
    sorted_feats = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    names = [f[0].split('_', 1)[1][:15] for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]
    colors = []
    for f in sorted_feats:
        group = f[0][0]
        color_map = {'A': '#e74c3c', 'B': '#3498db', 'C': '#2ecc71',
                     'D': '#f39c12', 'E': '#9b59b6', 'F': '#1abc9c'}
        colors.append(color_map.get(group, '#95a5a6'))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title('Top 20 Features (by magnitude)')
    ax.invert_yaxis()

    # Panel 7: Per-class PSV scores
    ax = axes[1, 2]
    cls_names = list(scores.keys())
    cls_vals = list(scores.values())
    colors = ['#e74c3c' if v > 0.6 else '#f39c12' if v > 0.4 else '#95a5a6' for v in cls_vals]
    ax.barh(range(len(cls_names)), cls_vals, color=colors)
    ax.set_yticks(range(len(cls_names)))
    ax.set_yticklabels([n.replace('_', ' ') for n in cls_names], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_title('PSV Disease Scores')
    ax.invert_yaxis()

    # Panel 8: IQA flags
    ax = axes[1, 3]
    ax.axis('off')
    if quality_flags:
        text = f'PSV Confidence: {psv_confidence:.2f}\n\n'
        text += 'Quality Flags:\n'
        for flag, triggered in quality_flags.items():
            if triggered:
                text += f'  {flag}\n'
        if not any(quality_flags.values()):
            text += '  (none triggered)\n'
    else:
        text = f'PSV Confidence: {psv_confidence:.2f}\n\nNo IQA data available'
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Image Quality Assessment')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'Debug visualization saved: {save_path}')
    else:
        plt.close()
        return fig
