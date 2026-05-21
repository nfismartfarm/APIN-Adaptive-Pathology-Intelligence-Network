"""
Compute the CORAL target covariance from Phase-1-trained ABMIL features.

Runs at Phase 1 END — after the final Phase 1 epoch has finished and the
post-Phase-1 attention-quality gate has PASSED. Produces
`data/specialist/model3/coral_target_cov.pt` as a provenance-dict (Critique 10).
Phase 2 training script asserts `source == 'abmil_features_phase1'` at startup.

The current `coral_target_cov.pt` (CLS features from Phase 0D probe) is
renamed to `coral_target_cov.cls_PRE_PHASE_1_INVALID.pt` for audit.

Inputs:
  - `models/specialist/ladinet_phase1_heads.pt` — Phase 1 final checkpoint
    (ABMIL + gated MLP + SupCon projector weights; backbone frozen)
  - `data/specialist/model3/split_indices.json` — for the 680 train-real-field paths
  - DINOv2-Base-Registers from timm (frozen, loaded at 392px)

Output:
  - `data/specialist/model3/coral_target_cov.pt` with provenance dict:
    {
        'cov':           Tensor(768, 768) float32,
        'source':        'abmil_features_phase1',
        'n_samples':     680,
        'resolution':    392,
        'phase1_checkpoint_hash': <md5>,
        'generated_at':  ISO-8601 timestamp,
    }

Usage:
    python scripts/ladi_net/compute_coral_target_abmil.py \\
        --phase1_ckpt models/specialist/ladinet_phase1_heads.pt

Exit codes:
    0: success
    1: input file missing, checkpoint load failure, or shape mismatch
    2: output would overwrite a same-provenance file with different hash
       (prevents silent re-computation drift)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL3_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
SPLIT_PATH = MODEL3_DIR / "split_indices.json"
CORAL_OUT = MODEL3_DIR / "coral_target_cov.pt"
CORAL_INVALID_BACKUP = MODEL3_DIR / "coral_target_cov.cls_PRE_PHASE_1_INVALID.pt"
EMBED_DIM = 768
PATCH_SIZE = 14
RESOLUTION = 392
NUM_PATCHES = (RESOLUTION // PATCH_SIZE) ** 2  # 784
NUM_REGISTERS = 4
PREFIX_TOKENS = 1 + NUM_REGISTERS                # CLS + 4 registers


# ---------------------------------------------------------------------------
# ABMIL head (mirrors Decision 17 §17.2; Phase 1 checkpoint key: 'abmil')
# ---------------------------------------------------------------------------
class ABMIL(nn.Module):
    """Ilse et al. 2018 gated-attention MIL pool. Tanh activation.

    Input: patch tokens [B, N, D] with N=784, D=768.
    Output: bag feature [B, D] = softmax-weighted sum of patch tokens.
    """

    def __init__(self, in_dim: int = EMBED_DIM, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, N, 1] attention logits
        h = torch.tanh(self.fc1(patch_tokens))
        logits = self.fc2(h).squeeze(-1)             # [B, N]
        attn = F.softmax(logits, dim=1)               # [B, N]
        # Weighted sum -> [B, D]
        return (patch_tokens * attn.unsqueeze(-1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Letterbox preprocessing (matches Decision 30.1 for field images)
# ---------------------------------------------------------------------------
def _letterbox_392(img_bgr: np.ndarray, pad_value: int = 114) -> np.ndarray:
    """Aspect-preserving resize of BGR uint8 image to 392x392 with pad=114."""
    import cv2
    h, w = img_bgr.shape[:2]
    scale = RESOLUTION / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    # Pad to 392x392 symmetrically
    top = (RESOLUTION - new_h) // 2
    bottom = RESOLUTION - new_h - top
    left = (RESOLUTION - new_w) // 2
    right = RESOLUTION - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),
    )
    return padded


def _preprocess_to_tensor(img_path: Path, device: torch.device) -> torch.Tensor:
    """Load BGR image, letterbox-resize to 392, apply LAB-CLAHE, normalise, return CHW tensor."""
    import cv2
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not load {img_path}")
    padded = _letterbox_392(img_bgr, pad_value=114)
    # LAB-CLAHE on L channel only (matches training preprocessing)
    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Normalise with ImageNet stats (timm DINOv2 default)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase1_ckpt",
        type=Path,
        default=PROJECT_ROOT / "models" / "specialist" / "ladinet_phase1_heads.pt",
        help="Path to Phase 1 heads checkpoint (from Phase 1 training script)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=CORAL_OUT,
        help="Output path for the ABMIL-based CORAL target",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ABMIL-source CORAL file without hash check",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA required for feature extraction; run on the RTX 4060.")
        sys.exit(1)
    device = torch.device("cuda")

    # Validate inputs
    if not args.phase1_ckpt.exists():
        print(f"[FAIL] Phase 1 checkpoint not found: {args.phase1_ckpt}")
        sys.exit(1)
    if not SPLIT_PATH.exists():
        print(f"[FAIL] split_indices.json not found: {SPLIT_PATH}")
        sys.exit(1)

    # Hash the Phase 1 checkpoint for provenance
    ckpt_bytes = args.phase1_ckpt.read_bytes()
    ckpt_hash = hashlib.md5(ckpt_bytes).hexdigest()

    # Prevent silent re-computation drift: if an ABMIL-source output already
    # exists with a different ckpt hash, abort unless --force.
    if args.output.exists():
        try:
            existing = torch.load(args.output, map_location="cpu", weights_only=False)
            if isinstance(existing, dict) and existing.get("source") == "abmil_features_phase1":
                if existing.get("phase1_checkpoint_hash") == ckpt_hash:
                    print(f"[OK] Existing output matches current Phase 1 checkpoint hash. "
                          f"No re-computation needed.")
                    sys.exit(0)
                elif not args.force:
                    print(f"[FAIL] Output file {args.output} already exists with a different "
                          f"Phase 1 checkpoint hash. Re-run with --force to overwrite.")
                    print(f"  existing hash: {existing.get('phase1_checkpoint_hash')}")
                    print(f"  current hash : {ckpt_hash}")
                    sys.exit(2)
        except Exception:
            pass  # non-dict file; proceed

    # Backup the CLS-based CORAL (Phase 0D probe artifact)
    # Only back up if the current file is CLS-source (i.e. pre-Phase-1 artifact)
    if CORAL_OUT.exists() and not CORAL_INVALID_BACKUP.exists():
        try:
            existing = torch.load(CORAL_OUT, map_location="cpu", weights_only=False)
            if torch.is_tensor(existing) or (
                isinstance(existing, dict) and existing.get("source") != "abmil_features_phase1"
            ):
                CORAL_INVALID_BACKUP.write_bytes(CORAL_OUT.read_bytes())
                print(f"[INFO] Backed up CLS-based CORAL -> {CORAL_INVALID_BACKUP.name}")
        except Exception as e:
            print(f"[WARN] could not back up existing CORAL: {e}")

    # Load split and filter to train-real-field
    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)
    train_paths = split["train"]

    # Real-field training images are those that:
    #  (a) appear in split['train']
    #  (b) do NOT appear in mask_precompute_log.csv (which contains only LAB images)
    #  (c) are NOT recomposed composites (path contains '/recomp' or filename starts 'recomp_')
    # This mirrors the dataloader's FIELD-path detection logic.
    import pandas as pd
    mask_log = pd.read_csv(MODEL3_DIR / "mask_precompute_log.csv")
    lab_filenames = {Path(str(p).replace("\\", "/")).name
                     for p in mask_log["image_path"]}

    def _is_recomposed(p_norm: str) -> bool:
        return (
            "/recomposed/" in p_norm
            or "/recomp_" in p_norm
            or Path(p_norm).name.lower().startswith("recomp_")
        )

    real_field_paths: list[Path] = []
    n_lab = 0
    n_recomp = 0
    for p in train_paths:
        p_norm = str(p).replace("\\", "/")
        if _is_recomposed(p_norm):
            n_recomp += 1
        elif Path(p_norm).name in lab_filenames:
            n_lab += 1
        else:
            real_field_paths.append(Path(p))
    print(f"Split['train'] breakdown: lab={n_lab}  recomposed={n_recomp}  "
          f"real_field={len(real_field_paths)}")

    print(f"Found {len(real_field_paths)} real-field training images")
    if len(real_field_paths) < 100:
        print(f"[FAIL] Expected ~680 real-field training images; got {len(real_field_paths)}. "
              f"Check split_indices.json and mask_precompute_log.csv path matching.")
        sys.exit(1)

    # Load Phase 1 heads checkpoint
    print(f"Loading Phase 1 checkpoint: {args.phase1_ckpt}")
    phase1 = torch.load(args.phase1_ckpt, map_location=device, weights_only=False)
    if not isinstance(phase1, dict) or "abmil_state_dict" not in phase1:
        print(f"[FAIL] Phase 1 checkpoint missing 'abmil_state_dict' key. "
              f"Got keys: {list(phase1.keys()) if isinstance(phase1, dict) else type(phase1)}")
        sys.exit(1)

    abmil = ABMIL(in_dim=EMBED_DIM, hidden=256).to(device)
    abmil.load_state_dict(phase1["abmil_state_dict"])
    abmil.eval()
    for p in abmil.parameters():
        p.requires_grad = False

    # Load DINOv2-Base-Registers (frozen)
    import timm
    print("Loading DINOv2-Base-Registers...")
    backbone = timm.create_model(
        "vit_base_patch14_reg4_dinov2",
        pretrained=True, num_classes=0,
        img_size=224, dynamic_img_size=True,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Extract ABMIL features for each real-field training image
    print(f"Extracting ABMIL features at {RESOLUTION}px for {len(real_field_paths)} images...")
    feats = np.zeros((len(real_field_paths), EMBED_DIM), dtype=np.float32)
    t0 = datetime.datetime.now()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i, p in enumerate(real_field_paths):
            try:
                tensor = _preprocess_to_tensor(p, device)
                out = backbone.forward_features(tensor)   # [1, 789, 768]
                patch_tokens = out[:, PREFIX_TOKENS:, :]  # [1, 784, 768]
                abmil_feat = abmil(patch_tokens.float())  # [1, 768]
                feats[i] = abmil_feat.squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"  [WARN] skipping {p.name}: {e}")
                feats[i] = np.nan  # sentinel
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (datetime.datetime.now() - t0).total_seconds()
                eta_s = (len(real_field_paths) - i - 1) / max(rate, 0.1)
                print(f"  {i+1}/{len(real_field_paths)} features ({rate:.1f} img/s, eta {eta_s/60:.1f} min)")

    # Drop NaN rows
    valid_mask = ~np.any(np.isnan(feats), axis=1)
    n_valid = int(valid_mask.sum())
    feats_valid = feats[valid_mask]
    print(f"Valid features: {n_valid} / {len(real_field_paths)}")
    if n_valid < 100:
        print(f"[FAIL] Too few valid feature extractions: {n_valid}")
        sys.exit(1)

    # Compute covariance
    print("Computing covariance matrix...")
    cov = np.cov(feats_valid, rowvar=False).astype(np.float32)
    assert cov.shape == (EMBED_DIM, EMBED_DIM), f"unexpected cov shape {cov.shape}"
    fro = float(np.linalg.norm(cov))
    print(f"  cov shape: {cov.shape}")
    print(f"  cov Frobenius norm: {fro:.4f}")
    print(f"  cov mean: {cov.mean():.6f}  std: {cov.std():.6f}")

    # Save with provenance dict
    provenance = {
        "cov": torch.from_numpy(cov),
        "source": "abmil_features_phase1",
        "n_samples": n_valid,
        "resolution": RESOLUTION,
        "phase1_checkpoint_hash": ckpt_hash,
        "generated_at": datetime.datetime.now().isoformat(),
        "frobenius_norm": fro,
    }
    torch.save(provenance, args.output)
    print(f"[OK] Saved CORAL target (provenance dict) to {args.output}")
    print(f"  source='abmil_features_phase1'  n_samples={n_valid}  "
          f"resolution={RESOLUTION}  frobenius={fro:.4f}")


if __name__ == "__main__":
    main()
