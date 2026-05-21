"""LADI-Net Phase 0 Step 1 — InSPyReNet Mask Pre-computation.

Input:
    All tomato-class lab images (is_field_photo == False) from
    data/specialist/model3/model3_unified_source_map.csv.

Processing per image:
    1. Open image via PIL (handles JPG/PNG/WebP etc.)
    2. Cap max dimension to 1024 (preserve aspect) for InSPyReNet input
    3. Run transparent_background.Remover → RGBA output with alpha channel
    4. Resize alpha back to original dimensions (nearest-neighbor)
    5. Compute coverage_fraction = (alpha > 128).sum() / total_pixels
    6. Compute confidence = mean(alpha[alpha > 128]) / 255.0
    7. Flag if:
         - confidence < 0.70
         - coverage < 0.15
         - coverage > 0.85
    8. Save binary mask (uint8, 0/255) as {stem}_mask.png alongside original
    9. Save foreground (3-channel BGR, background zeroed) as {stem}_fg.png
    10. Append row to data/specialist/model3/mask_precompute_log.csv

Outputs:
    - data/specialist/model3/mask_precompute_log.csv  (master log)
    - data/specialist/model3/mask_precompute_env.json (reproducibility snapshot)
    - data/specialist/model3/prototype_seeds/<class>/seed_paths.txt
      (top-50 highest-confidence non-flagged lab images per class)
    - {image_dir}/{stem}_mask.png, {image_dir}/{stem}_fg.png

Key rules:
    - Resumable: re-running skips images already in the CSV (any row, flagged
      or not).
    - Per-image error handling: one bad image does not crash the run.
    - num_workers=0 implicit (single-threaded; InSPyReNet owns the GPU).
    - Windows-safe path handling via pathlib.Path throughout.
    - SACRED FILES: never touched. We only WRITE to mask_precompute_log.csv,
      the _mask.png / _fg.png files (new), prototype_seeds/, and env.json.

Run:
    python scripts/ladi_net/phase0_mask_precompute.py

Expected runtime on RTX 4060 at batch=1: ~90-110 min for ~22,400 lab images
(3.9 img/s benchmark from scripts/phase0_background_recomposition.py docstring).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------------------
# Constants — keep here for easy audit
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model3" / "model3_unified_source_map.csv"
OUT_LOG_CSV = PROJECT_ROOT / "data" / "specialist" / "model3" / "mask_precompute_log.csv"
OUT_ENV_JSON = PROJECT_ROOT / "data" / "specialist" / "model3" / "mask_precompute_env.json"
PROTOTYPE_DIR = PROJECT_ROOT / "data" / "specialist" / "model3" / "prototype_seeds"
LOG_DIR = PROJECT_ROOT / "logs"

TOMATO_CLASSES = [
    "tomato_foliar_spot",
    "tomato_septoria_leaf_spot",
    "tomato_late_blight",
    "tomato_yellow_leaf_curl_virus",  # canonical per Decision 3
    "tomato_mosaic_virus",
    "tomato_healthy",
]

# Thresholds (Decision 2 confidence definition; Decision 5 mask format).
CONFIDENCE_THRESHOLD = 0.70
COVERAGE_MIN = 0.15
COVERAGE_MAX = 0.85
INSPYRENET_MAX_DIM = 1024  # resize-for-inference cap
FOREGROUND_ALPHA_THRESHOLD = 128  # matches phase0_background_recomposition.py convention

# Prototype seed selection.
SEEDS_PER_CLASS = 50


# ------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    """Timestamped log file + console output."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ladi_phase0_mask_precompute_{ts}.log"
    logger = logging.getLogger("ladi.phase0.mask")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


# ------------------------------------------------------------------------
# Per-row result type
# ------------------------------------------------------------------------
@dataclass
class PrecomputeResult:
    image_path: str            # relative to PROJECT_ROOT
    mask_path: str             # relative to PROJECT_ROOT
    fg_path: str               # relative to PROJECT_ROOT
    inspyrenet_confidence: float
    coverage_fraction: float
    flagged: bool
    flag_reason: str           # one of the enum strings below
    class_name: str
    source_dataset: str
    is_field_photo: bool


FLAG_REASONS = {
    "low_confidence",
    "low_coverage",
    "high_coverage",
    "processing_error",
    "corrupt_image",
    "multiple_flags",
    "",                         # not flagged
}


def _derive_flag(confidence: float, coverage: float) -> tuple[bool, str]:
    """Apply the 3 threshold checks and emit a single enum string."""
    reasons = []
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if coverage < COVERAGE_MIN:
        reasons.append("low_coverage")
    if coverage > COVERAGE_MAX:
        reasons.append("high_coverage")
    if not reasons:
        return False, ""
    if len(reasons) == 1:
        return True, reasons[0]
    return True, "multiple_flags:" + ",".join(reasons)


# ------------------------------------------------------------------------
# InSPyReNet wrapper (singleton — GPU model loaded once)
# ------------------------------------------------------------------------
_REMOVER = None


def _get_remover(logger: logging.Logger):
    """Lazy-initialise transparent_background.Remover on GPU."""
    global _REMOVER
    if _REMOVER is not None:
        return _REMOVER
    try:
        from transparent_background import Remover
    except ImportError as e:
        logger.error("=" * 72)
        logger.error("DEVELOPER ATTENTION REQUIRED")
        logger.error("transparent_background package missing — cannot proceed.")
        logger.error("Install: pip install transparent-background")
        logger.error("=" * 72)
        raise SystemExit(1) from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _REMOVER = Remover(mode="fast", device=device)
    logger.info(f"InSPyReNet (transparent_background) ready on {device} [mode=fast]")
    return _REMOVER


# ------------------------------------------------------------------------
# Per-image processor
# ------------------------------------------------------------------------
def _process_one(
    img_path: Path,
    class_name: str,
    source_dataset: str,
    is_field_photo: bool,
    logger: logging.Logger,
) -> PrecomputeResult:
    """Run InSPyReNet on one image, save mask + fg, return result row.

    Gracefully handles corrupt images and InSPyReNet errors. NEVER raises
    to the caller — failures are encoded as `flag_reason=processing_error`
    so a single bad image cannot crash the run.
    """
    rel_img = img_path.relative_to(PROJECT_ROOT).as_posix()
    mask_path = img_path.with_name(img_path.stem + "_mask.png")
    fg_path = img_path.with_name(img_path.stem + "_fg.png")
    rel_mask = mask_path.relative_to(PROJECT_ROOT).as_posix()
    rel_fg = fg_path.relative_to(PROJECT_ROOT).as_posix()

    # 1. Load image via PIL (handles all formats).
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.warning(f"corrupt_image {rel_img}: {type(e).__name__}: {e}")
        return PrecomputeResult(
            image_path=rel_img, mask_path="", fg_path="",
            inspyrenet_confidence=0.0, coverage_fraction=0.0,
            flagged=True, flag_reason="corrupt_image",
            class_name=class_name, source_dataset=source_dataset,
            is_field_photo=is_field_photo,
        )

    orig_w, orig_h = pil_img.size

    # 2. Resize for InSPyReNet if too large.
    max_dim = max(orig_w, orig_h)
    if max_dim > INSPYRENET_MAX_DIM:
        scale = INSPYRENET_MAX_DIM / max_dim
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pil_for_segment = pil_img.resize((new_w, new_h), Image.LANCZOS)
    else:
        pil_for_segment = pil_img

    # 3. InSPyReNet forward pass.
    try:
        remover = _get_remover(logger)
        rgba_pil = remover.process(pil_for_segment, type="rgba")  # RGBA PIL
        rgba = np.array(rgba_pil)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM on {rel_img}: {e}")
        torch.cuda.empty_cache()
        return PrecomputeResult(
            image_path=rel_img, mask_path="", fg_path="",
            inspyrenet_confidence=0.0, coverage_fraction=0.0,
            flagged=True, flag_reason="processing_error:cuda_oom",
            class_name=class_name, source_dataset=source_dataset,
            is_field_photo=is_field_photo,
        )
    except Exception as e:
        logger.warning(
            f"InSPyReNet error on {rel_img}: {type(e).__name__}: {e}"
        )
        return PrecomputeResult(
            image_path=rel_img, mask_path="", fg_path="",
            inspyrenet_confidence=0.0, coverage_fraction=0.0,
            flagged=True, flag_reason=f"processing_error:{type(e).__name__}",
            class_name=class_name, source_dataset=source_dataset,
            is_field_photo=is_field_photo,
        )

    # 4. Extract alpha and resize back to original dims with NEAREST neighbor
    #    (keeps the mask binary-like after re-scaling).
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        logger.warning(f"unexpected rgba shape {rgba.shape} for {rel_img}")
        return PrecomputeResult(
            image_path=rel_img, mask_path="", fg_path="",
            inspyrenet_confidence=0.0, coverage_fraction=0.0,
            flagged=True, flag_reason="processing_error:bad_alpha_shape",
            class_name=class_name, source_dataset=source_dataset,
            is_field_photo=is_field_photo,
        )

    alpha_small = rgba[:, :, 3]
    if (alpha_small.shape[1], alpha_small.shape[0]) != (orig_w, orig_h):
        alpha_full = cv2.resize(
            alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        alpha_full = alpha_small

    # 5. Coverage and confidence.
    total_px = orig_w * orig_h
    fg_mask_bool = alpha_full > FOREGROUND_ALPHA_THRESHOLD
    fg_count = int(fg_mask_bool.sum())
    coverage = fg_count / total_px
    if fg_count == 0:
        confidence = 0.0
    else:
        confidence = float(alpha_full[fg_mask_bool].mean() / 255.0)

    # 6. Binarize mask (uint8 0 or 255) + save.
    try:
        mask_binary = (fg_mask_bool.astype(np.uint8)) * 255
        cv2.imwrite(str(mask_path), mask_binary)

        # Foreground: 3-channel BGR, background zeroed. Load the ORIGINAL
        # image in BGR via cv2 (not the PIL-resized version); apply the mask.
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            # Unusual but possible for images PIL can open but cv2 can't
            # (certain WebP/ICC edge cases). Fall back to PIL→BGR.
            rgb_arr = np.array(pil_img)[:, :, ::-1]  # RGB → BGR
            bgr = rgb_arr.copy()
        if bgr.shape[:2] != (orig_h, orig_w):
            bgr = cv2.resize(bgr, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        fg_bgr = bgr.copy()
        fg_bgr[~fg_mask_bool] = 0
        cv2.imwrite(str(fg_path), fg_bgr)
    except Exception as e:
        logger.warning(
            f"file_write error on {rel_img}: {type(e).__name__}: {e}"
        )
        return PrecomputeResult(
            image_path=rel_img, mask_path="", fg_path="",
            inspyrenet_confidence=confidence, coverage_fraction=coverage,
            flagged=True, flag_reason=f"processing_error:{type(e).__name__}",
            class_name=class_name, source_dataset=source_dataset,
            is_field_photo=is_field_photo,
        )

    # 7. Threshold checks.
    flagged, flag_reason = _derive_flag(confidence, coverage)

    return PrecomputeResult(
        image_path=rel_img, mask_path=rel_mask, fg_path=rel_fg,
        inspyrenet_confidence=confidence, coverage_fraction=coverage,
        flagged=flagged, flag_reason=flag_reason,
        class_name=class_name, source_dataset=source_dataset,
        is_field_photo=is_field_photo,
    )


# ------------------------------------------------------------------------
# Resumable CSV writer
# ------------------------------------------------------------------------
CSV_HEADER = [
    "image_path", "mask_path", "fg_path",
    "inspyrenet_confidence", "coverage_fraction",
    "flagged", "flag_reason",
    "class_name", "source_dataset", "is_field_photo",
]


def _load_already_processed(csv_path: Path) -> set[str]:
    """Return the set of `image_path` values already in the CSV (resumability)."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        done = set(df["image_path"].astype(str).tolist())
        return done
    except Exception:
        # Corrupt CSV — best to start fresh, warn caller.
        return set()


def _append_csv_row(csv_path: Path, row: PrecomputeResult):
    """Append one result to the master log CSV; create file with header if needed."""
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow([
            row.image_path, row.mask_path, row.fg_path,
            f"{row.inspyrenet_confidence:.6f}", f"{row.coverage_fraction:.6f}",
            int(row.flagged), row.flag_reason,
            row.class_name, row.source_dataset, int(row.is_field_photo),
        ])


# ------------------------------------------------------------------------
# Prototype seed selection (Step 1 final artifact)
# ------------------------------------------------------------------------
def _select_prototype_seeds(csv_path: Path, out_dir: Path, logger: logging.Logger):
    """Per class, take top-50 non-flagged lab images by InSPyReNet confidence.

    If a class has fewer than 50 non-flagged lab images, take as many as
    available; if fewer than 20, emit a CRITIQUE flag (logged to chat, not
    to critique.md — the test is statistical, the critique is qualitative).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    stats_by_class: dict[str, dict] = {}

    for cls in TOMATO_CLASSES:
        sub = df[(df["class_name"] == cls)
                 & (df["is_field_photo"].astype(str).isin({"0", "False", "false"}))
                 & (df["flagged"] == 0)]
        ranked = sub.sort_values("inspyrenet_confidence", ascending=False)
        n_to_take = min(SEEDS_PER_CLASS, len(ranked))
        seeds = ranked.head(n_to_take)

        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        seed_txt = cls_dir / "seed_paths.txt"
        with open(seed_txt, "w", encoding="utf-8") as f:
            for p in seeds["image_path"].tolist():
                f.write(p + "\n")

        stats_by_class[cls] = {
            "seeds_saved": n_to_take,
            "available_non_flagged": len(sub),
            "min_confidence_in_seeds": (float(seeds["inspyrenet_confidence"].min())
                                        if n_to_take > 0 else 0.0),
            "seed_path": str(seed_txt.relative_to(PROJECT_ROOT).as_posix()),
        }
        logger.info(
            f"  prototype seeds [{cls:32s}]  saved={n_to_take:3d}  "
            f"available={len(sub):5d}  "
            f"min_conf={stats_by_class[cls]['min_confidence_in_seeds']:.3f}"
        )
        if n_to_take < 20:
            logger.warning(
                f"  WARNING: only {n_to_take} prototype seeds for {cls} — "
                f"prototype memory quality may be degraded."
            )
    return stats_by_class


# ------------------------------------------------------------------------
# Environment snapshot
# ------------------------------------------------------------------------
def _write_env_snapshot(logger: logging.Logger):
    """Record InSPyReNet version + package versions for reproducibility."""
    import sys as _sys
    env = {
        "timestamp": datetime.now().isoformat(),
        "python": _sys.version.split()[0],
        "platform": _sys.platform,
    }
    # Core packages
    for pkg in ["torch", "numpy", "PIL", "cv2", "pandas"]:
        try:
            if pkg == "PIL":
                import PIL as _p
                env["pillow"] = _p.__version__
            elif pkg == "cv2":
                import cv2 as _c
                env["opencv"] = _c.__version__
            elif pkg == "torch":
                import torch as _t
                env["torch"] = _t.__version__
                env["cuda_available"] = _t.cuda.is_available()
                if _t.cuda.is_available():
                    env["cuda_device"] = _t.cuda.get_device_name(0)
                    env["cuda_version"] = _t.version.cuda
            else:
                mod = __import__(pkg)
                env[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            env[pkg] = f"import_error: {e}"
    # transparent_background
    try:
        import transparent_background as _tb
        env["transparent_background"] = getattr(_tb, "__version__", "unknown")
    except Exception as e:
        env["transparent_background"] = f"import_error: {e}"
    env["inspyrenet_mode"] = "fast"
    env["inspyrenet_max_dim"] = INSPYRENET_MAX_DIM
    env["confidence_threshold"] = CONFIDENCE_THRESHOLD
    env["coverage_min"] = COVERAGE_MIN
    env["coverage_max"] = COVERAGE_MAX
    env["foreground_alpha_threshold"] = FOREGROUND_ALPHA_THRESHOLD
    OUT_ENV_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_ENV_JSON.write_text(json.dumps(env, indent=2), encoding="utf-8")
    logger.info(f"env snapshot → {OUT_ENV_JSON.relative_to(PROJECT_ROOT).as_posix()}")
    return env


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=str(CSV_PATH),
                        help="Model 3 unified source map CSV")
    parser.add_argument("--out-log", default=str(OUT_LOG_CSV),
                        help="Output mask_precompute_log.csv path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N images (smoke test)")
    parser.add_argument("--skip-seeds", action="store_true",
                        help="Do not run prototype seed selection at the end")
    args = parser.parse_args(argv)

    logger = _setup_logger()
    logger.info("=" * 72)
    logger.info("LADI-Net Phase 0 Step 1 — InSPyReNet Mask Pre-computation")
    logger.info("=" * 72)

    # 1. Env snapshot up front (reproducibility).
    _write_env_snapshot(logger)

    # 2. Load the Model 3 CSV and filter to tomato lab images.
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return 1
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV: {len(df):,} rows total")

    # is_field_photo may be bool or int/str depending on how CSV was written.
    # Normalize to bool.
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"1", "true", "t", "yes"}
    df["_is_field_bool"] = df["is_field_photo"].apply(_to_bool)
    work = df[(df["class_name"].isin(TOMATO_CLASSES)) & (~df["_is_field_bool"])].copy()
    logger.info(f"Tomato lab images to process: {len(work):,}")
    if args.limit:
        work = work.head(args.limit)
        logger.info(f"--limit {args.limit} applied; processing {len(work)} images")

    # 3. Resume: skip images already processed.
    out_csv = Path(args.out_log)
    already = _load_already_processed(out_csv)
    if already:
        logger.info(f"Resuming: {len(already):,} images already in log; "
                    f"skipping those.")

    # 4. Main loop.
    total = len(work)
    processed = 0
    flagged_count = 0
    start_time = time.time()
    per_class_counts: dict[str, int] = {c: 0 for c in TOMATO_CLASSES}

    for idx, row in enumerate(work.itertuples(index=False), start=1):
        img_path_str = str(row.image_path)
        if img_path_str in already:
            continue

        img_path = (PROJECT_ROOT / img_path_str).resolve()
        if not img_path.exists():
            logger.warning(f"file missing, skipping: {img_path_str}")
            # Log as corrupt_image so the row exists and resumability works.
            result = PrecomputeResult(
                image_path=img_path_str, mask_path="", fg_path="",
                inspyrenet_confidence=0.0, coverage_fraction=0.0,
                flagged=True, flag_reason="corrupt_image:file_missing",
                class_name=row.class_name,
                source_dataset=str(row.source_dataset),
                is_field_photo=False,
            )
        else:
            result = _process_one(
                img_path=img_path,
                class_name=row.class_name,
                source_dataset=str(row.source_dataset),
                is_field_photo=False,
                logger=logger,
            )

        _append_csv_row(out_csv, result)
        processed += 1
        per_class_counts[result.class_name] = per_class_counts.get(result.class_name, 0) + 1
        if result.flagged:
            flagged_count += 1

        if processed % 50 == 0 or processed == 1:
            elapsed = time.time() - start_time
            rate = processed / max(elapsed, 1e-6)
            eta_sec = (total - idx) / max(rate, 1e-6)
            logger.info(
                f"[{idx:5d}/{total:5d}] {row.class_name}/{img_path.name}  "
                f"conf={result.inspyrenet_confidence:.3f}  "
                f"coverage={result.coverage_fraction:.3f}  "
                f"flagged={result.flagged}  "
                f"rate={rate:.2f} img/s  "
                f"eta={eta_sec/60:.0f} min"
            )
            # Periodic VRAM hygiene.
            if processed % 500 == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    total_elapsed = time.time() - start_time
    logger.info("=" * 72)
    logger.info(f"Done. Processed {processed} images in {total_elapsed/60:.1f} min.")
    logger.info(f"Flagged (this run): {flagged_count} / {processed}")
    logger.info(f"Master log: {out_csv}")

    # 5. Summary report from the FULL CSV (includes resume rows).
    if out_csv.exists():
        full = pd.read_csv(out_csv)
        logger.info("")
        logger.info("-" * 72)
        logger.info("SUMMARY BY CLASS")
        logger.info("-" * 72)
        logger.info(f"{'class':<32}  {'total':>6}  {'flagged':>8}  {'flag%':>6}  "
                    f"{'mean_conf':>9}  {'mean_cov':>8}")
        for cls in TOMATO_CLASSES:
            sub = full[full["class_name"] == cls]
            if sub.empty:
                logger.info(f"{cls:<32}  {'—':>6}  {'—':>8}  {'—':>6}  "
                            f"{'—':>9}  {'—':>8}")
                continue
            flag_pct = sub["flagged"].mean() * 100
            mc = sub["inspyrenet_confidence"].mean()
            mv = sub["coverage_fraction"].mean()
            warn = "  [WARN >30%]" if flag_pct > 30 else ""
            logger.info(
                f"{cls:<32}  {len(sub):>6}  {int(sub['flagged'].sum()):>8}  "
                f"{flag_pct:>5.1f}%  {mc:>9.3f}  {mv:>8.3f}{warn}"
            )

    # 6. Prototype seed selection (unless --skip-seeds).
    if not args.skip_seeds and out_csv.exists():
        logger.info("")
        logger.info("-" * 72)
        logger.info("PROTOTYPE SEED SELECTION (top-50 highest-confidence non-flagged per class)")
        logger.info("-" * 72)
        _select_prototype_seeds(out_csv, PROTOTYPE_DIR, logger)

    logger.info("")
    logger.info("[OK] Step 1 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
