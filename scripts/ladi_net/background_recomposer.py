"""LADI-Net Phase 0 Step 3 — Online background recomposer.

This module is a Dataset utility — not a training script. It will be
imported by the LADI-Net training DataLoader (Phase 2) to apply online
background recomposition on lab images at training time.

Usage (from training script):

    from scripts.ladi_net.background_recomposer import OnlineRecomposer

    recomposer = OnlineRecomposer(
        mask_log_csv="data/specialist/model3/mask_precompute_log.csv",
        background_pool_dir=None,  # auto-discover chilli+brassica field pool
        recompose_probability=0.70,
        seed=42,
    )

    # In Dataset.__getitem__:
    bgr = cv2.imread(str(image_path))
    bgr = recomposer.recompose(image_path=str(image_path), image_bgr=bgr, epoch=epoch)
    # ... continue with CLAHE, resize, normalize ...

Per LADI-Net spec (Part Three, Step 0A):
    - For every lab image (is_field_photo==False), with probability 0.70,
      replace the white/plain background with a randomly selected field
      background from the pool.
    - Uses the PRE-COMPUTED masks and foregrounds written by Step 1.
    - Never touches field images or recomposed images.
    - Different seed per epoch → each lab image sees different backgrounds
      across epochs (key to the ~9-23% accuracy improvement from literature).

Background pool:
    Per Decision 8 below, the pool is assembled from field photos of:
      - chilli_healthy (is_field_photo=True)
      - chilli_leaf_curl (is_field_photo=True)
      - brassica_healthy (from Model 2 cleaned dir; is_field_photo=True)
    The existing scripts/phase0_background_recomposition.py used these
    same three directories for the 9,705 static recompositions. We reuse
    them for consistency.

Test:
    python scripts/ladi_net/background_recomposer.py --test-sample 5
    → produces 5 recomposed test images in
      data/specialist/model3/verification_samples/step3/
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASK_LOG = PROJECT_ROOT / "data" / "specialist" / "model3" / "mask_precompute_log.csv"
DEFAULT_BG_DIRS = [
    PROJECT_ROOT / "data" / "specialist" / "model3" / "cleaned" / "chilli_healthy",
    PROJECT_ROOT / "data" / "specialist" / "model3" / "cleaned" / "chilli_leaf_curl",
    PROJECT_ROOT / "data" / "specialist" / "model2" / "cleaned" / "brassica_healthy",
]
VERIF_DIR = PROJECT_ROOT / "data" / "specialist" / "model3" / "verification_samples" / "step3"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}


# ------------------------------------------------------------------------
# OnlineRecomposer — the core class used by the training DataLoader
# ------------------------------------------------------------------------
class OnlineRecomposer:
    """Replace the background of lab images on-the-fly at training time.

    Parameters
    ----------
    mask_log_csv : Path | str
        Path to the `mask_precompute_log.csv` produced by Step 1. Used to
        identify which lab images have a valid (non-flagged) mask + fg.
    background_pool_dir : Path | list[Path] | None
        Directory (or list of directories) containing field background JPG/PNGs.
        If None, uses the DEFAULT_BG_DIRS (chilli+brassica field photos).
    recompose_probability : float, default 0.70
        Probability of applying recomposition when a valid mask exists.
    seed : int, default 42
        Base seed for reproducibility. Epoch-specific randomness comes from
        `(seed, epoch, image_path_hash)` — each lab image gets a different
        background every epoch, but the choice is reproducible.
    preload_backgrounds : bool, default True
        If True, load all backgrounds into memory as numpy arrays at
        construction time (preferred per spec — ~300 MB RAM for 2000×224²).
        If False, read from disk per recomposition (slow but memory-light).

    Notes
    -----
    - Images where the Step 1 mask is flagged are NOT recomposed — the
      original image passes through unchanged. This protects against
      compositing with a corrupt mask.
    - Field images are never touched — the caller is responsible for
      checking `is_field_photo` before invoking `recompose()`. (In
      practice the training dataloader will only call `recompose()` on
      lab samples; this class takes that on trust.)
    - Threading: thread-safe for reads; DO NOT share the internal RNG
      across threads. On Windows with num_workers=0 this is moot.
    """

    def __init__(
        self,
        mask_log_csv: Path | str = DEFAULT_MASK_LOG,
        background_pool_dir: Path | list[Path] | None = None,
        recompose_probability: float = 0.70,
        seed: int = 42,
        preload_backgrounds: bool = False,
        preload_max: int = 2000,                 # [Issue 12-A fix] was 1000
        preload_resize_max_dim: int = 392,       # [Issue 12-A fix] was 512 — match training resolution
        logger: Optional[logging.Logger] = None,
    ):
        self.recompose_probability = float(recompose_probability)
        self.seed = int(seed)
        self._log = logger or _noop_logger()

        # 1. Load the mask-precompute CSV to build a path → (mask_path, fg_path, flagged) map.
        mask_log_csv = Path(mask_log_csv)
        if not mask_log_csv.exists():
            raise FileNotFoundError(
                f"mask_precompute_log.csv not found at {mask_log_csv}. "
                f"Run scripts/ladi_net/phase0_mask_precompute.py first."
            )
        df = pd.read_csv(mask_log_csv)
        self._path_to_mask: dict[str, dict] = {}
        for row in df.itertuples(index=False):
            self._path_to_mask[str(row.image_path)] = {
                "mask": str(row.mask_path) if row.mask_path else "",
                "fg": str(row.fg_path) if row.fg_path else "",
                "flagged": bool(row.flagged),
                "confidence": float(row.inspyrenet_confidence),
            }
        self._log.info(f"Loaded mask log: {len(self._path_to_mask):,} entries "
                       f"({sum(1 for v in self._path_to_mask.values() if not v['flagged']):,} non-flagged)")

        # 2. Resolve and collect the background pool.
        if background_pool_dir is None:
            bg_dirs = DEFAULT_BG_DIRS
        elif isinstance(background_pool_dir, (str, Path)):
            bg_dirs = [Path(background_pool_dir)]
        else:
            bg_dirs = [Path(d) for d in background_pool_dir]

        bg_paths: list[Path] = []
        for d in bg_dirs:
            if not d.exists():
                self._log.warning(f"background dir missing: {d}")
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix in VALID_EXTS:
                    # Skip masks and fgs (not backgrounds)
                    if f.name.endswith("_mask.png") or f.name.endswith("_fg.png"):
                        continue
                    bg_paths.append(f)
        if len(bg_paths) < 100:
            raise ValueError(
                f"Background pool has only {len(bg_paths)} images — "
                f"need ≥100 for diversity. Searched: {[str(d) for d in bg_dirs]}"
            )
        self._log.info(f"Background pool: {len(bg_paths)} images from {len(bg_dirs)} dirs")

        # 3. Preload backgrounds into RAM if requested. Capped at preload_max
        #    (default 1000) and resized to max-dim preload_resize_max_dim (512)
        #    to keep memory usage bounded. 7000+ full-resolution images would
        #    easily exceed 30 GB RAM.
        self._bg_paths = bg_paths
        self._bg_cache: list[np.ndarray] = []
        self._preloaded = False
        if preload_backgrounds:
            rng_preload = random.Random(self.seed)
            # Sample a subset if the pool is larger than the preload cap.
            if len(bg_paths) > preload_max:
                sampled = rng_preload.sample(bg_paths, preload_max)
            else:
                sampled = list(bg_paths)
            for p in sampled:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                m = max(h, w)
                if m > preload_resize_max_dim:
                    scale = preload_resize_max_dim / m
                    img = cv2.resize(
                        img,
                        (int(round(w * scale)), int(round(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                self._bg_cache.append(img)
            if self._bg_cache:
                self._preloaded = True
                # Estimate RAM: sum of array sizes.
                total_bytes = sum(a.nbytes for a in self._bg_cache)
                self._log.info(
                    f"Preloaded {len(self._bg_cache)} backgrounds into RAM "
                    f"(max-dim={preload_resize_max_dim}, "
                    f"~{total_bytes / 1024**2:.0f} MB)"
                )
            else:
                self._log.warning("preload requested but no backgrounds loaded")

    # ------------------------------------------------------------------
    # Deterministic-per-image-per-epoch RNG — two lab images in the same
    # epoch get different backgrounds; the same image in two different
    # epochs also gets different backgrounds.
    # ------------------------------------------------------------------
    def _rng_for(self, image_path: str, epoch: int) -> random.Random:
        # random.Random only accepts a hashable scalar, not a tuple, as seed.
        # Combine the three parts deterministically into an int.
        key = (self.seed * 1_000_003 + epoch * 31 + hash(image_path)) & 0xFFFFFFFF
        return random.Random(key)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def recompose(
        self,
        image_path: str,
        image_bgr: np.ndarray,
        epoch: int = 0,
    ) -> np.ndarray:
        """Return either the recomposed BGR image or the original.

        Contract:
          - Returns a BGR uint8 ndarray of the SAME shape as `image_bgr`.
          - If mask entry is missing, flagged, or fails to load → returns
            `image_bgr` unchanged (caller's no-op path).
          - If `random() >= recompose_probability` → returns `image_bgr`
            unchanged (probabilistic no-op).
        """
        entry = self._path_to_mask.get(image_path)
        if entry is None or entry["flagged"] or not entry["mask"] or not entry["fg"]:
            return image_bgr

        rng = self._rng_for(image_path, epoch)
        if rng.random() >= self.recompose_probability:
            return image_bgr

        # Load precomputed mask + foreground. Resize to match image_bgr shape.
        mask_abs = PROJECT_ROOT / entry["mask"]
        fg_abs = PROJECT_ROOT / entry["fg"]
        mask = cv2.imread(str(mask_abs), cv2.IMREAD_GRAYSCALE)
        fg = cv2.imread(str(fg_abs), cv2.IMREAD_COLOR)
        if mask is None or fg is None:
            return image_bgr

        h, w = image_bgr.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if fg.shape[:2] != (h, w):
            fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)

        # Pick a background.
        if self._preloaded:
            bg_full = self._bg_cache[rng.randrange(len(self._bg_cache))]
        else:
            bg_full = cv2.imread(str(self._bg_paths[rng.randrange(len(self._bg_paths))]),
                                  cv2.IMREAD_COLOR)
            if bg_full is None:
                return image_bgr

        # Resize background to leaf image dims (INTER_AREA for downscale quality).
        if bg_full.shape[:2] != (h, w):
            bg = cv2.resize(bg_full, (w, h), interpolation=cv2.INTER_AREA)
        else:
            bg = bg_full

        # Composite — fg is already background-zeroed per Decision 5, so
        # where mask==0 we keep bg, where mask==255 we keep fg.
        # np.where with broadcasting is clean and fast.
        mask_3c = np.stack([mask, mask, mask], axis=-1)
        composite = np.where(mask_3c > 0, fg, bg).astype(np.uint8)
        return composite

    # ------------------------------------------------------------------
    # Utility — verification helpers used by test mode / PVA.
    # ------------------------------------------------------------------
    def test_on_samples(self, n: int = 5, out_dir: Path = VERIF_DIR) -> list[Path]:
        """Pick n random non-flagged lab images, recompose each, save the
        original+composite side-by-side to out_dir. Returns the list of
        saved images for PVA/PDA visual inspection."""
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates = [p for p, e in self._path_to_mask.items() if not e["flagged"]]
        if not candidates:
            self._log.warning("no non-flagged lab images in the mask log")
            return []
        rng = random.Random(self.seed)
        sample = rng.sample(candidates, min(n, len(candidates)))
        written: list[Path] = []
        for i, rel_path in enumerate(sample):
            abs_path = PROJECT_ROOT / rel_path
            bgr = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            # Force recompose=1.0 for the test by using a temp instance setting
            saved_p = self.recompose_probability
            self.recompose_probability = 1.0
            try:
                comp = self.recompose(rel_path, bgr, epoch=i)
            finally:
                self.recompose_probability = saved_p
            # side-by-side
            side = np.concatenate([bgr, comp], axis=1)
            out_p = out_dir / f"recompose_sample_{i:02d}_{abs_path.stem}.png"
            cv2.imwrite(str(out_p), side)
            written.append(out_p)
            self._log.info(f"  wrote {out_p.relative_to(PROJECT_ROOT).as_posix()}")
        return written


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
def _noop_logger() -> logging.Logger:
    lg = logging.getLogger("ladi.recomposer.default")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


# ------------------------------------------------------------------------
# Main — standalone test mode
# ------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mask-log", default=str(DEFAULT_MASK_LOG))
    parser.add_argument("--test-sample", type=int, default=5,
                        help="Recompose N lab images and save side-by-side PNGs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prob", type=float, default=0.70)
    args = parser.parse_args(argv)

    print("=" * 72)
    print("LADI-Net Phase 0 Step 3 — Online Background Recomposer")
    print("=" * 72)

    recomposer = OnlineRecomposer(
        mask_log_csv=args.mask_log,
        recompose_probability=args.prob,
        seed=args.seed,
        preload_backgrounds=True,
    )

    if args.test_sample > 0:
        print(f"\nGenerating {args.test_sample} side-by-side verification samples...")
        written = recomposer.test_on_samples(n=args.test_sample, out_dir=VERIF_DIR)
        print(f"\nWrote {len(written)} samples to {VERIF_DIR.relative_to(PROJECT_ROOT).as_posix()}")

    print("\n[OK] Step 3 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
