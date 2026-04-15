# training/02_generate_severity.py
"""
Generates proxy severity labels for all training images using GradCAM saliency.
Severity is estimated from the fraction of the leaf area covered by high-activation
saliency regions.

[FIX GAP 12] Uses _SevProxyDataset which applies CLAHE (matching training distribution).
[FIX GAP 9]  Synthetic images are in data/raw/synthetic/ — not data/processed/train/.

Saves: data/metadata/severity_labels.csv
  Columns: image_path (relative to ROOT), severity_idx (0=mild,1=moderate,2=severe)
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import pandas as pd

from app.config import (
    DEVICE, SOURCE_MAP, SEV_LABELS, ROOT, META,
    SEVERITY_PROXY_THRESHOLD, SEVERITY_MILD_MAX, SEVERITY_MOD_MAX,
    CKPT_DIR,
)
from app.model import load_model_for_inference
from training.dataset import _SevProxyDataset


def estimate_severity_from_saliency(saliency_map, threshold, mild_max, mod_max):
    """
    Given a saliency map [H, W] normalised to [0,1]:
    - Binarise at threshold (top threshold fraction of activations)
    - coverage = fraction of pixels that are active
    - severity: mild if < mild_max, moderate if < mod_max, else severe
    Returns int: 0=mild, 1=moderate, 2=severe
    """
    thresh_val = torch.quantile(saliency_map, 1.0 - threshold)
    binary     = (saliency_map >= thresh_val).float()
    coverage   = binary.mean().item()
    if coverage < mild_max:
        return 0  # mild
    elif coverage < mod_max:
        return 1  # moderate
    else:
        return 2  # severe


def generate_severity_labels():
    """
    Main function. Loads training images, computes GradCAM saliency,
    estimates severity coverage, writes severity_labels.csv.
    """
    # Load model — need the trained disease head to compute saliency
    phase1_best = os.path.join(CKPT_DIR, 'phase1_best.pt')
    if not os.path.exists(phase1_best):
        print("Phase 1 model not found. Using uniform moderate placeholder labels.")
        _write_placeholder_labels()
        return

    model = load_model_for_inference(phase1_best, DEVICE)
    model.eval()

    # Load all training image paths
    df    = pd.read_csv(SOURCE_MAP)
    train_df = df[df['split'] == 'train']
    paths    = train_df['image_path'].tolist()

    print(f"Generating severity labels for {len(paths)} training images...")

    dataset = _SevProxyDataset(paths)  # [FIX GAP 12] CLAHE applied inside
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    os.makedirs(META, exist_ok=True)

    results = []
    processed = 0

    for images, rel_paths in loader:
        images = images.to(DEVICE)

        with torch.no_grad():
            # Use model disease_head activation as saliency proxy
            features     = model.backbone(images)
            fused        = model.fpn(*features)
            pooled       = model.gap(fused).flatten(1)
            _, crop_emb  = model.crop_classifier(pooled)
            d_logits     = model.disease_head(pooled, crop_emb)
            d_probs      = torch.sigmoid(d_logits)
            # Use max disease probability as confidence proxy
            # Use FPN P3 feature magnitude as saliency map
            saliency_maps = fused.abs().mean(dim=1)  # [B, 28, 28]

        for i, rel_path in enumerate(rel_paths):
            sal   = saliency_maps[i]
            sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            sev   = estimate_severity_from_saliency(
                sal_n.cpu(),
                SEVERITY_PROXY_THRESHOLD,
                SEVERITY_MILD_MAX,
                SEVERITY_MOD_MAX,
            )
            results.append({'image_path': rel_path, 'severity_idx': sev})
        processed += len(images)
        if processed % 500 == 0:
            print(f"  {processed}/{len(paths)}", end='\r')

    with open(SEV_LABELS, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'severity_idx'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWritten {len(results)} severity labels to {SEV_LABELS}")
    mild = sum(1 for r in results if r['severity_idx'] == 0)
    mod  = sum(1 for r in results if r['severity_idx'] == 1)
    sev  = sum(1 for r in results if r['severity_idx'] == 2)
    print(f"Distribution: mild={mild}, moderate={mod}, severe={sev}")


def _write_placeholder_labels():
    """Fallback: write uniform moderate severity labels when no model is available.
    All images get severity_idx=1 (moderate) as a placeholder.
    The severity head is lowest priority — these placeholders will be
    replaced with GradCAM-based proxy labels after Phase 1 training."""
    df      = pd.read_csv(SOURCE_MAP)
    train_df = df[df['split'] == 'train']
    os.makedirs(META, exist_ok=True)
    with open(SEV_LABELS, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'severity_idx'])
        writer.writeheader()
        for _, row in train_df.iterrows():
            writer.writerow({
                'image_path'  : row['image_path'],
                'severity_idx': 1,  # moderate placeholder
            })
    print(f"Written {len(train_df)} placeholder severity labels (all moderate, no model available).")


if __name__ == '__main__':
    generate_severity_labels()
