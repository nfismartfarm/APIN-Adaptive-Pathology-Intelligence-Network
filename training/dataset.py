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
    """
    Two-tier weighted sampler:
    Tier 1: Ensures each crop contributes equally (25% each) to every batch.
    Tier 2: Within each crop, applies inverse-frequency weighting with 10x cap.

    This guarantees equal crop representation even with 13:1 tomato:chilli imbalance,
    while still balancing disease classes within each crop.
    """
    from app.config import (SAMPLER_MAX_WEIGHT_RATIO, SAMPLER_USE_TWO_TIER,
                             CROP_SAMPLING_WEIGHTS, CROP_LABEL_MAP, CLASS_NAMES)
    from collections import Counter

    if not SAMPLER_USE_TWO_TIER:
        # Fallback to simple inverse-frequency (legacy behavior)
        class_counts = Counter(r.get('class_name', '') for r in records)
        class_weights = {cls: 1.0 / max(count, 1) for cls, count in class_counts.items()}
        min_w = min(class_weights.values())
        max_allowed = min_w * SAMPLER_MAX_WEIGHT_RATIO
        for cls in class_weights:
            class_weights[cls] = min(class_weights[cls], max_allowed)
        weights = [class_weights.get(r.get('class_name', ''), 1.0) for r in records]
        return WeightedRandomSampler(weights=weights, num_samples=len(records), replacement=True)

    # ── Two-tier sampling ─────────────────────────────────────────────────

    # Map each class to its crop index
    crop_class_map = {}
    for class_name in CLASS_NAMES:
        for crop_name, crop_idx in CROP_LABEL_MAP.items():
            if class_name.startswith(crop_name):
                crop_class_map[class_name] = crop_idx
                break

    # Count images per crop
    total_train = len(records)
    crop_counts = {}
    for crop_idx in CROP_SAMPLING_WEIGHTS.keys():
        crop_classes = [cls for cls, idx in crop_class_map.items() if idx == crop_idx]
        crop_counts[crop_idx] = sum(1 for r in records if r.get('class_name', '') in crop_classes)

    # Tier 1: crop-level weight = target proportion / actual proportion
    tier1_weights = {}
    for crop_idx, target_prop in CROP_SAMPLING_WEIGHTS.items():
        actual_prop = crop_counts[crop_idx] / total_train if total_train > 0 else 0
        tier1_weights[crop_idx] = target_prop / actual_prop if actual_prop > 0 else 1.0

    # Tier 2: within-crop disease-level inverse frequency with cap
    class_counts = Counter(r.get('class_name', '') for r in records)
    tier2_weights = {}
    for crop_idx in CROP_SAMPLING_WEIGHTS.keys():
        crop_classes = [cls for cls, idx in crop_class_map.items() if idx == crop_idx]
        crop_class_counts = {cls: class_counts.get(cls, 0) for cls in crop_classes}
        if not crop_class_counts or all(v == 0 for v in crop_class_counts.values()):
            for cls in crop_classes:
                tier2_weights[cls] = 1.0
            continue
        inv_freq = {cls: 1.0 / max(cnt, 1) for cls, cnt in crop_class_counts.items()}
        min_w = min(inv_freq.values())
        max_allowed = min_w * SAMPLER_MAX_WEIGHT_RATIO
        for cls in inv_freq:
            inv_freq[cls] = min(inv_freq[cls], max_allowed)
        total_w = sum(inv_freq.values())
        for cls in crop_classes:
            tier2_weights[cls] = inv_freq.get(cls, 1.0) / total_w if total_w > 0 else 1.0

    # Final weight = tier1 * tier2
    final_weights = []
    for r in records:
        cls = r.get('class_name', '')
        crop_idx = crop_class_map.get(cls, 0)
        w = tier1_weights.get(crop_idx, 1.0) * tier2_weights.get(cls, 1.0)
        final_weights.append(w)

    return WeightedRandomSampler(
        weights=final_weights,
        num_samples=len(records),
        replacement=True,
    )


class PlantDiseaseDatasetPhase2B(Dataset):
    """
    Extended dataset for Phase 2B full fine-tuning.
    Returns dict with image, labels, AND domain/thin-class flags needed for
    CORAL loss, DeiT distillation, and RandAugment magnitude selection.
    """
    def __init__(self, records, transform, sev_labels=None):
        from app.config import PLANTDOC_SOURCE_PATTERN, THIN_CLASS_INDICES
        self.records = records
        self.transform = transform
        self.sev_labels = sev_labels or {}
        self.plantdoc_pattern = PLANTDOC_SOURCE_PATTERN
        self.thin_indices = set(THIN_CLASS_INDICES)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        rel_path = r['image_path']
        full_path = os.path.join(ROOT, rel_path.replace('/', os.sep))

        try:
            img = Image.open(full_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        class_idx = r.get('class_idx', -1)
        d_label = torch.zeros(NUM_CLASSES)
        if 0 <= class_idx < NUM_CLASSES:
            d_label[class_idx] = 1.0

        c_label = torch.tensor(r.get('crop_idx', 0), dtype=torch.long)
        s_label = torch.tensor(self.sev_labels.get(rel_path, 0), dtype=torch.long)

        source = str(r.get('source_dataset', ''))
        is_plantdoc = self.plantdoc_pattern in source.lower()
        is_thin = bool(class_idx in self.thin_indices)

        return {
            'image': img,
            'disease_labels': d_label,
            'crop_label': c_label,
            'severity_label': s_label,
            'is_plantdoc': torch.tensor(is_plantdoc, dtype=torch.bool),
            'is_thin': torch.tensor(is_thin, dtype=torch.bool),
        }


class DomainBalancedSampler(torch.utils.data.Sampler):
    """
    Sampler guaranteeing MIN_PLANTDOC_PER_BATCH PlantDoc images per batch.

    PlantDoc is 0.9% of training data. Without this sampler, ~83% of batches
    have zero PlantDoc images, making CORAL loss effectively zero.

    Strategy: reserve min_target slots per batch for PlantDoc, fill remaining
    slots using two-tier weighted sampling from source domain.
    """
    def __init__(self, records, batch_size, min_target_per_batch,
                 source_pattern='plantdoc'):
        from app.config import PLANTDOC_SOURCE_PATTERN
        self.batch_size = batch_size
        self.min_target = min(min_target_per_batch, batch_size - 1)

        # Separate indices by domain
        self.target_indices = []
        self.source_indices = []
        for i, r in enumerate(records):
            src = str(r.get('source_dataset', ''))
            if source_pattern in src.lower():
                self.target_indices.append(i)
            else:
                self.source_indices.append(i)

        self.n_source_per_batch = batch_size - self.min_target
        total_batches = max(1, len(records) // batch_size)
        self.total_samples = total_batches * batch_size

        print(f'  DomainBalancedSampler: {len(self.target_indices)} PlantDoc, '
              f'{len(self.source_indices)} source, '
              f'{self.min_target} PlantDoc/batch guaranteed')

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        total_batches = self.total_samples // self.batch_size
        indices = []
        for _ in range(total_batches):
            # Sample PlantDoc with replacement (only 131 images)
            target_batch = np.random.choice(
                self.target_indices, size=self.min_target, replace=True
            ).tolist()
            # Sample source without replacement
            source_batch = np.random.choice(
                self.source_indices,
                size=min(self.n_source_per_batch, len(self.source_indices)),
                replace=False,
            ).tolist()
            batch = target_batch + source_batch
            np.random.shuffle(batch)
            indices.extend(batch)
        return iter(indices)


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
