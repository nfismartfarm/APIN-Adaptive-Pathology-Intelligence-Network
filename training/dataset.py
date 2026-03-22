# training/dataset.py

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from app.config import (
    ROOT, SEV_LABELS, CLASS_TO_IDX, CROP_FROM_IDX, NUM_CLASSES, VALID_EXT
)


def load_severity_labels():
    """Loads severity_labels.csv as dict {image_path: severity_idx}."""
    if not os.path.exists(SEV_LABELS):
        return {}
    sev_map = {}
    with open(SEV_LABELS, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sev_map[row['image_path']] = int(row['severity_idx'])
    return sev_map


class PlantDiseaseDataset(Dataset):
    """Loads images from source_map.csv records."""
    def __init__(self, records, transform, sev_labels=None):
        self.records    = records
        self.transform  = transform
        self.sev_labels = sev_labels or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r         = self.records[idx]
        rel_path  = r['image_path']
        full_path = os.path.join(ROOT, rel_path.replace('/', os.sep))

        try:
            img = Image.open(full_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        class_idx = r.get('class_idx', -1)
        d_label   = torch.zeros(NUM_CLASSES)
        if 0 <= class_idx < NUM_CLASSES:
            d_label[class_idx] = 1.0

        c_label = torch.tensor(r.get('crop_idx', 0), dtype=torch.long)
        s_label = torch.tensor(
            self.sev_labels.get(rel_path, 0), dtype=torch.long
        )

        return img, d_label, c_label, s_label


def make_weighted_sampler(records):
    """WeightedRandomSampler for clubroot oversampling."""
    from app.config import CLUBROOT_OVERSAMPLE
    weights = []
    for r in records:
        cls = r.get('class_name', '')
        w   = CLUBROOT_OVERSAMPLE if cls == 'brassica_clubroot' else 1.0
        weights.append(w)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(records),
        replacement=True,
    )


class _SevProxyDataset(Dataset):
    """Dataset for severity proxy label generation."""
    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from training.transforms import apply_clahe
        import torchvision.transforms as TF

        full_path = os.path.join(ROOT, self.paths[idx].replace('/', os.sep))
        try:
            img = Image.open(full_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            img = apply_clahe(img)
            img = Image.fromarray(img)
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        transform = TF.Compose([
            TF.Resize((224, 224)),
            TF.ToTensor(),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(img), self.paths[idx]
