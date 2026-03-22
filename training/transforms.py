# training/transforms.py

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from app.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """CLAHE per RGB channel. Defined here for training. Also defined inline in app/inference.py."""
    clahe  = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = clahe.apply(image[:, :, c])
    return result


def simulate_colour_temperature(image, **kwargs):
    """Simulates Kerala-specific lighting colour temperatures."""
    factor = np.random.uniform(0.75, 1.35)
    img    = image.astype(np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] * factor,         0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 / factor), 0, 255)
    return img.astype(np.uint8)


def get_train_transform():
    """Training augmentation. Kerala-specific. Applied ONLY to train split."""
    return A.Compose([
        A.Lambda(image=apply_clahe, p=1.0),
        A.Resize(256, 256),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101, p=0.6
        ),
        A.RandomCrop(IMG_SIZE[0], IMG_SIZE[1]),
        A.OneOf([
            A.Lambda(image=simulate_colour_temperature, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform():
    """Validation / test / inference transform."""
    return A.Compose([
        A.Lambda(image=apply_clahe, p=1.0),
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
