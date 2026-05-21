"""
LADI-Net DataLoader — 4-path routing (Decision 34) + ClassStratifiedBatchSampler (Decision 19).

Image types:
  LAB_OK      : is_field_photo=0, flagged=0, not recomposed → load _fg.png →
                stochastic tight-crop p=0.30 → recompose p=0.70 → aug
  LAB_FLAGGED : is_field_photo=0, flagged=1 → load _fg.png → deterministic tight-crop →
                no recompose → aug
  FIELD       : is_field_photo=1 → load original → letterbox-resize-to-392 → aug
  RECOMPOSED  : path under data/.../recomposed/ → load composite JPG → aug
                (already has field bg; no recomp, no tight-crop)

All paths apply LAB-CLAHE + standard augmentation (HFlip / Affine / ColorJitter /
RandomResizedCrop) after geometry is finalised.

Also provides:
- load_split_records(phase='phase1'|'phase2') → list of dicts with class/type/path/etc.
- ClassStratifiedBatchSampler: per-class slots [8,8,4,4,4,4] for Phase 1 (bs=32) or
  [4,4,2,2,2,2] for Phase 2 (bs=16). Field-images get 8× sampling weight within each
  class queue except YLCV/mosaic (capped at 4× per Decision 24).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from ladinet_config import (
    PROJECT_ROOT, MASK_LOG_CSV, SPLIT_JSON,
    TOMATO_CLASSES, CLASS_TO_IDX, RESOLUTION,
    IMAGENET_MEAN, IMAGENET_STD, LETTERBOX_PAD_VALUE,
    STOCHASTIC_TIGHT_CROP_PROB, TIGHT_CROP_PAD, RECOMPOSE_PROB_NON_FLAGGED,
    FIELD_SAMPLE_WEIGHT, FIELD_SAMPLE_WEIGHT_THIN, THIN_CLASS_THRESHOLD,
    PHASE1_CLASS_SLOTS, PHASE2_CLASS_SLOTS, PHASE1_BATCH_SIZE, PHASE2_BATCH_SIZE,
    AUG_HFLIP_P, AUG_AFFINE_ROTATE_DEG, AUG_AFFINE_P,
    AUG_COLOR_JITTER_BCS, AUG_COLOR_JITTER_P,
    AUG_RANDOM_RESIZED_CROP_SCALE, AUG_RANDOM_RESIZED_CROP_RATIO, AUG_RANDOM_RESIZED_CROP_P,
    SEED,
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN_ARR = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD_ARR = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)


# ===========================================================================
# Record enumeration
# ===========================================================================
@dataclass
class LadiRecord:
    image_path: str          # absolute path on disk
    class_name: str
    class_idx: int
    image_type: str          # LAB_OK / LAB_FLAGGED / FIELD / RECOMPOSED
    is_field_photo: bool     # True if FIELD
    fg_path: str | None      # for LAB_OK/LAB_FLAGGED only
    mask_path: str | None    # for LAB_OK only (used by recomposer)


def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")


def _classify_image(row: dict, is_recomposed_fn) -> str:
    """Decide image_type for a given source_map row."""
    path_norm = _norm_path(row.get("image_path", ""))
    if is_recomposed_fn(path_norm):
        return "RECOMPOSED"
    if bool(row.get("is_field_photo", 0)):
        return "FIELD"
    if bool(row.get("flagged", 0)):
        return "LAB_FLAGGED"
    return "LAB_OK"


def load_split_records(which: str = "train") -> list[LadiRecord]:
    """Load training records from split_indices.json + mask_precompute_log.csv.

    which: 'train' | 'field_val' | 'confusable_pair_probe' | 'final_val'
    """
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)

    paths = split.get(which, [])
    # Build lookup from the mask_precompute_log for is_field_photo and flagged
    mask_log = pd.read_csv(MASK_LOG_CSV)
    mask_log["image_path_norm"] = mask_log["image_path"].map(_norm_path)
    mask_lookup = {row.image_path_norm: row for row in mask_log.itertuples(index=False)}

    def is_recomposed(p_norm: str) -> bool:
        return (
            "/recomposed/" in p_norm
            or "/recomp_" in p_norm
            or "recomp_" in Path(p_norm).name.lower()
        )

    # Index mask_log by filename for O(1) lookup instead of O(N) suffix-match.
    # Filenames are unique per class; collisions across classes would have been
    # caught by mask precompute's own dedup.
    mask_by_filename = {}
    for row in mask_log.itertuples(index=False):
        name = Path(str(row.image_path)).name
        mask_by_filename[name] = row

    def _resolve_relative(rel_path: str | None) -> str | None:
        """Convert a PROJECT_ROOT-relative path to an absolute path."""
        if not rel_path:
            return None
        p = Path(rel_path)
        if p.is_absolute():
            return str(p) if p.exists() else None
        abs_p = PROJECT_ROOT / rel_path.replace("\\", "/")
        return str(abs_p) if abs_p.exists() else None

    records: list[LadiRecord] = []
    skipped = 0
    for abs_path in paths:
        p_norm = _norm_path(abs_path)
        filename = Path(p_norm).name

        # Find class from path
        cls = None
        for c in TOMATO_CLASSES:
            if f"/cleaned/{c}/" in p_norm or f"/{c}/" in p_norm:
                cls = c
                break
        if cls is None:
            skipped += 1
            continue

        # Decide type
        if is_recomposed(p_norm):
            image_type = "RECOMPOSED"
            fg_path = None
            mask_path = None
        else:
            matched = mask_by_filename.get(filename)
            if matched is not None:
                if bool(matched.is_field_photo):
                    image_type = "FIELD"
                    fg_path = None
                    mask_path = None
                elif bool(matched.flagged):
                    fg_path = _resolve_relative(str(getattr(matched, "fg_path", "") or ""))
                    mask_path = _resolve_relative(str(getattr(matched, "mask_path", "") or ""))
                    if not fg_path:
                        skipped += 1
                        continue
                    image_type = "LAB_FLAGGED"
                else:
                    fg_path = _resolve_relative(str(getattr(matched, "fg_path", "") or ""))
                    mask_path = _resolve_relative(str(getattr(matched, "mask_path", "") or ""))
                    if not fg_path:
                        skipped += 1
                        continue
                    image_type = "LAB_OK"
            else:
                # Not in mask log → treat as FIELD (real smartphone photo from the field split)
                image_type = "FIELD"
                fg_path = None
                mask_path = None

        records.append(LadiRecord(
            image_path=abs_path,
            class_name=cls,
            class_idx=CLASS_TO_IDX[cls],
            image_type=image_type,
            is_field_photo=(image_type == "FIELD"),
            fg_path=fg_path,
            mask_path=mask_path,
        ))

    if skipped > 0:
        print(f"[DataLoader] Skipped {skipped} rows from split['{which}'] "
              f"(unclassifiable or missing fg_path — includes 2 MemoryError cases)")

    return records


# ===========================================================================
# Image preprocessing helpers
# ===========================================================================
def _letterbox_392(img_bgr: np.ndarray, pad_value: int = LETTERBOX_PAD_VALUE) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if (h, w) == (RESOLUTION, RESOLUTION):
        return img_bgr
    scale = RESOLUTION / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    top = (RESOLUTION - new_h) // 2
    bottom = RESOLUTION - new_h - top
    left = (RESOLUTION - new_w) // 2
    right = RESOLUTION - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(pad_value,) * 3)


def _tight_crop_leaf(fg_bgr: np.ndarray, pad_frac: float = TIGHT_CROP_PAD) -> np.ndarray:
    """Crop bbox of non-black pixels, expand by pad_frac, clamp, resize to 392."""
    gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 5
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return fg_bgr                                  # no leaf detected; passthrough
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    H, W = fg_bgr.shape[:2]
    ph = int((y2 - y1) * pad_frac)
    pw = int((x2 - x1) * pad_frac)
    y1 = max(0, y1 - ph)
    y2 = min(H - 1, y2 + ph)
    x1 = max(0, x1 - pw)
    x2 = min(W - 1, x2 + pw)
    crop = fg_bgr[y1:y2 + 1, x1:x2 + 1]
    return cv2.resize(crop, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)


def _apply_lab_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _random_affine_rotate(img_bgr: np.ndarray, rng: random.Random,
                          max_deg: float) -> np.ndarray:
    angle = rng.uniform(-max_deg, max_deg)
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h),
                          borderMode=cv2.BORDER_REPLICATE)


def _color_jitter(img_bgr: np.ndarray, rng: random.Random, jitter: float) -> np.ndarray:
    b = 1.0 + rng.uniform(-jitter, jitter)
    c = 1.0 + rng.uniform(-jitter, jitter)
    s_scale = 1.0 + rng.uniform(-jitter, jitter)
    img = img_bgr.astype(np.float32)
    # brightness × contrast around mean
    mean = img.mean()
    img = (img - mean) * c + mean
    img *= b
    # saturation in HSV
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def _random_resized_crop(img_bgr: np.ndarray, rng: random.Random,
                         scale_range: tuple[float, float],
                         ratio_range: tuple[float, float]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    area = h * w
    for _ in range(10):
        target_area = area * rng.uniform(*scale_range)
        log_ratio = (np.log(ratio_range[0]), np.log(ratio_range[1]))
        aspect = float(np.exp(rng.uniform(*log_ratio)))
        new_w = int(round((target_area * aspect) ** 0.5))
        new_h = int(round((target_area / aspect) ** 0.5))
        if 0 < new_w <= w and 0 < new_h <= h:
            x = rng.randint(0, w - new_w)
            y = rng.randint(0, h - new_h)
            crop = img_bgr[y:y + new_h, x:x + new_w]
            return cv2.resize(crop, (RESOLUTION, RESOLUTION),
                              interpolation=cv2.INTER_AREA)
    return img_bgr  # give up — return input unchanged


def _standard_augment(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """HFlip + Affine rotate + ColorJitter + RandomResizedCrop — Decision 17 §17.5."""
    if rng.random() < AUG_HFLIP_P:
        img_bgr = cv2.flip(img_bgr, 1)
    if rng.random() < AUG_AFFINE_P:
        img_bgr = _random_affine_rotate(img_bgr, rng, AUG_AFFINE_ROTATE_DEG)
    if rng.random() < AUG_COLOR_JITTER_P:
        img_bgr = _color_jitter(img_bgr, rng, AUG_COLOR_JITTER_BCS)
    if rng.random() < AUG_RANDOM_RESIZED_CROP_P:
        img_bgr = _random_resized_crop(img_bgr, rng,
                                       AUG_RANDOM_RESIZED_CROP_SCALE,
                                       AUG_RANDOM_RESIZED_CROP_RATIO)
    return img_bgr


def _bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → RGB float32 normalised CHW tensor."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN_ARR) / IMAGENET_STD_ARR
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


# ===========================================================================
# Dataset (stateful — shares a recomposer instance)
# ===========================================================================
class LadiNetDataset(Dataset):
    """Routes each record through the right preprocessing path and returns tensors."""

    def __init__(self, records: list[LadiRecord], training: bool = True,
                 background_pool: list[np.ndarray] | None = None,
                 rng_seed: int = SEED):
        self.records = records
        self.training = training
        self.bg_pool = background_pool or []
        self.rng_seed = rng_seed

    def __len__(self):
        return len(self.records)

    def _load_image_for_record(self, rec: LadiRecord, rng: random.Random) -> np.ndarray:
        """Returns a 392×392 BGR uint8 image following the 4-path routing."""
        if rec.image_type in ("LAB_OK", "LAB_FLAGGED"):
            img = cv2.imread(rec.fg_path, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
            # Tight-crop logic
            if rec.image_type == "LAB_FLAGGED":
                img = _tight_crop_leaf(img)
            elif self.training and rng.random() < STOCHASTIC_TIGHT_CROP_PROB:
                img = _tight_crop_leaf(img)
            # Recomposition (LAB_OK only, training only)
            if (rec.image_type == "LAB_OK" and self.training
                    and rng.random() < RECOMPOSE_PROB_NON_FLAGGED
                    and self.bg_pool):
                img = self._recompose(img, rng)
            return img if img.shape[:2] == (RESOLUTION, RESOLUTION) else \
                cv2.resize(img, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        elif rec.image_type == "FIELD":
            img = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
            if img is None:
                img = np.full((RESOLUTION, RESOLUTION, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
            # [Decision 50 Fix 2] Cap FIELD image decode at 800px longest dim BEFORE any other
            # processing to prevent RAM OOM from smartphone-native 4000x3000 images (~34 MB uint8).
            # 800x600 ~= 1.4 MB uint8 (24x reduction). Subsequent letterbox-to-392 is unaffected.
            h, w = img.shape[:2]
            max_dim_cap = 800
            if max(h, w) > max_dim_cap:
                scale = max_dim_cap / max(h, w)
                img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
            img = _letterbox_392(img)
            return img
        elif rec.image_type == "RECOMPOSED":
            img = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
            return img if img.shape[:2] == (RESOLUTION, RESOLUTION) else _letterbox_392(img)
        else:
            raise ValueError(f"Unknown image_type {rec.image_type!r}")

    def _recompose(self, fg_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
        """Simple where(mask>0, fg, bg) composite using a random bg from pool."""
        bg = self.bg_pool[rng.randrange(len(self.bg_pool))]
        if bg.shape[:2] != fg_bgr.shape[:2]:
            bg = cv2.resize(bg, (fg_bgr.shape[1], fg_bgr.shape[0]),
                            interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > 5)[:, :, None]
        return np.where(mask, fg_bgr, bg).astype(np.uint8)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        # Per-item RNG is deterministic on (seed, idx) so the same epoch produces
        # the same augmentations for resume-on-epoch-boundary consistency.
        rng = random.Random((self.rng_seed * 1_000_003 + idx) & 0xFFFFFFFF)

        img = self._load_image_for_record(rec, rng)
        img = _apply_lab_clahe(img)
        if self.training:
            img = _standard_augment(img, rng)

        tensor = _bgr_to_tensor(img)
        return {
            "image": tensor,
            "label": torch.tensor(rec.class_idx, dtype=torch.long),
            "is_field_photo": torch.tensor(1 if rec.is_field_photo else 0, dtype=torch.long),
            "image_type_idx": torch.tensor(
                {"LAB_OK": 0, "LAB_FLAGGED": 1, "FIELD": 2, "RECOMPOSED": 3}[rec.image_type],
                dtype=torch.long,
            ),
        }


# ===========================================================================
# ClassStratifiedBatchSampler (Decision 19 / 24 / 31)
# ===========================================================================
class ClassStratifiedBatchSampler(Sampler[list[int]]):
    """Yields batches with deterministic per-class slot counts.

    Phase 1 slots = [8, 8, 4, 4, 4, 4] (bs=32)
    Phase 2 slots = [4, 4, 2, 2, 2, 2] (bs=16)

    Within each class queue, field images are oversampled at the class-conditional
    weight: 8× for normal classes, 4× for thin (field_train < 30) — Decision 24.
    """

    def __init__(self, records: list[LadiRecord], phase: str = "phase1",
                 seed: int = SEED):
        self.records = records
        self.seed = seed
        self.phase = phase
        self.slots = PHASE1_CLASS_SLOTS if phase == "phase1" else PHASE2_CLASS_SLOTS
        assert len(self.slots) == len(TOMATO_CLASSES), "slot count must match class count"
        self.batch_size = sum(self.slots)

        # Per-class index lists + field-indicator arrays
        self.per_class_idx: list[list[int]] = [[] for _ in TOMATO_CLASSES]
        self.per_class_isfield: list[list[bool]] = [[] for _ in TOMATO_CLASSES]
        for i, r in enumerate(records):
            ci = r.class_idx
            self.per_class_idx[ci].append(i)
            self.per_class_isfield[ci].append(r.is_field_photo)

        # Determine per-class sampling weight for field: 8× or 4×
        self.per_class_field_weight = []
        for ci in range(len(TOMATO_CLASSES)):
            n_field = sum(self.per_class_isfield[ci])
            w = FIELD_SAMPLE_WEIGHT_THIN if n_field < THIN_CLASS_THRESHOLD else FIELD_SAMPLE_WEIGHT
            self.per_class_field_weight.append(w)

        # Total batches per epoch = total_records // batch_size
        # With deterministic per-class draws and replacement within class, we can set
        # num_batches to len(records) // batch_size.
        self.num_batches = max(1, len(records) // self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + 1)  # reshuffle per epoch via seed bump externally
        for _ in range(self.num_batches):
            batch: list[int] = []
            for ci, slots in enumerate(self.slots):
                class_idx_list = self.per_class_idx[ci]
                if not class_idx_list:
                    continue
                isfield = np.array(self.per_class_isfield[ci], dtype=bool)
                weights = np.where(isfield, self.per_class_field_weight[ci], 1.0)
                probs = weights / weights.sum()
                # Draw `slots` indices with replacement (acceptable — pools are large)
                picked = rng.choice(len(class_idx_list), size=slots, replace=True, p=probs)
                batch.extend(class_idx_list[p] for p in picked)
            yield batch

    def set_epoch(self, epoch: int):
        self.seed = (SEED * 1_000_003 + epoch * 31) & 0xFFFFFFFF


# ===========================================================================
# Background pool loading (392px preload per Decision 12-A fix)
# ===========================================================================
def load_background_pool(max_images: int = 1000,   # [Decision 50 Fix 3] reduced from 2000 to 1000
                         target_res: int = RESOLUTION) -> list[np.ndarray]:
    """Preloads background images from the default dirs used by the recomposer."""
    # Reuse the recomposer's DEFAULT_BG_DIRS
    import sys as _sys
    sys_path_save = list(_sys.path)
    _sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ladi_net"))
    try:
        from background_recomposer import DEFAULT_BG_DIRS
    finally:
        _sys.path = sys_path_save

    paths: list[Path] = []
    for d in DEFAULT_BG_DIRS:
        dp = Path(d)
        if not dp.exists():
            continue
        for f in dp.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXTS:
                n = f.name.lower()
                if n.endswith("_mask.png") or n.endswith("_fg.png"):
                    continue
                paths.append(f)

    if len(paths) > max_images:
        rng = random.Random(SEED)
        paths = rng.sample(paths, max_images)

    loaded: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        m = max(h, w)
        if m != target_res:
            scale = target_res / m
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        # Pad to target_res x target_res if non-square
        h2, w2 = img.shape[:2]
        if (h2, w2) != (target_res, target_res):
            top = (target_res - h2) // 2
            bottom = target_res - h2 - top
            left = (target_res - w2) // 2
            right = target_res - w2 - left
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_REFLECT_101)
        loaded.append(img)
    return loaded
