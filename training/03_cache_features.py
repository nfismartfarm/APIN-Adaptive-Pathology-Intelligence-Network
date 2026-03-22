# training/03_cache_features.py
"""
Caches backbone + FPN features for Phase 1 training.
Uses get_eval_transform() — NOT get_train_transform().
Saves: cache/train_features.pt, cache/val_features.pt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from app.config import (
    DEVICE, TRAIN_CACHE, VAL_CACHE, CACHE, SOURCE_MAP, SEV_LABELS,
    CLASS_TO_IDX, CROP_FROM_IDX, NUM_CLASSES, RANDOM_SEED
)
from app.model import PlantDiseaseModel, verify_backbone_shapes
from training.dataset import PlantDiseaseDataset, load_severity_labels
from training.transforms import get_eval_transform


def cache_features():
    print("=" * 60)
    print("03_CACHE_FEATURES — caching backbone+FPN features")
    print("=" * 60)

    model = PlantDiseaseModel().to(DEVICE)
    verify_backbone_shapes(model, device=DEVICE)
    model.freeze_backbone()
    model.eval()

    df = pd.read_csv(SOURCE_MAP)
    sev_labels = load_severity_labels()

    for split_name, cache_path in [('train', TRAIN_CACHE), ('val', VAL_CACHE)]:
        split_df = df[df['split'] == split_name]
        records = split_df.to_dict('records')
        for r in records:
            r['class_idx'] = CLASS_TO_IDX.get(r.get('class_name', ''), -1)
            r['crop_idx']  = CROP_FROM_IDX.get(r['class_idx'], 0)

        if not records:
            print(f"  [{split_name}] No records found. Skipping.")
            continue

        ds = PlantDiseaseDataset(records, get_eval_transform(), sev_labels)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        all_pooled   = []
        all_crop_emb = []
        all_d_labels = []
        all_c_labels = []
        all_s_labels = []

        print(f"  [{split_name}] Caching {len(records)} images...")
        with torch.no_grad():
            for batch_idx, (images, d_lab, c_lab, s_lab) in enumerate(dl):
                images = images.to(DEVICE)
                pooled, crop_emb = model.extract_features(images)
                all_pooled.append(pooled.cpu())
                all_crop_emb.append(crop_emb.cpu())
                all_d_labels.append(d_lab)
                all_c_labels.append(c_lab)
                all_s_labels.append(s_lab)
                if (batch_idx + 1) % 50 == 0:
                    print(f"    batch {batch_idx + 1}/{len(dl)}", end='\r')

        os.makedirs(CACHE, exist_ok=True)
        cache_data = {
            'pooled_features' : torch.cat(all_pooled),
            'crop_embeddings' : torch.cat(all_crop_emb),
            'disease_labels'  : torch.cat(all_d_labels),
            'crop_labels'     : torch.cat(all_c_labels),
            'severity_labels' : torch.cat(all_s_labels),
        }
        torch.save(cache_data, cache_path)
        print(f"  [{split_name}] Saved {cache_data['pooled_features'].shape[0]} features to {cache_path}")

    print("\nFeature caching complete.")


if __name__ == '__main__':
    cache_features()
