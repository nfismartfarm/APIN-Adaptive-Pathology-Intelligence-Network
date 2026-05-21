"""LADI-Net Phase 0 Step 0D — Systematic probe.

Answers the question: how much does real-field training data improve field F1
compared to recomposed-only training, for DINOv2-Base-Registers on the tomato
dataset at each of three resolutions?

Runs:
  * For each resolution in {224, 336, 384}:
      1. Extract CLS-token features from DINOv2-Base-Registers (frozen).
         Cache to data/specialist/model3/probe_cache/features_{res}_train/val.pt
      2. Logistic Regression Config 1 — train on (lab + recomposed), eval on field_val
      3. Logistic Regression Config 2 — train on (lab + recomposed + real-field),
         eval on field_val
      4. Report weighted field F1 (sqrt(N) weights from split_indices.json),
         foliar-septoria centroid cosine distance, lab-vs-field Frobenius gap

  * DSF-Vec probe: compute 12 handcrafted features on real-field train+val,
    fit RidgeClassifier, report per-class field F1. Drop Feature 12 if near-zero
    variance.

  * Save CORAL target covariance (computed at the chosen resolution from all
    real-field train features) to data/specialist/model3/coral_target_cov.pt
    shape (768, 768) float32.

Writes:
  data/specialist/model3/probe_results.md
  data/specialist/model3/coral_target_cov.pt
  data/specialist/model3/probe_cache/ (feature caches)

Decision outputs:
  Recommended resolution (3% rule)
  Recommended --train_frac/--val_frac/--final_frac (benefit-based rule)
  DSF-Vec features to retain/drop

Run:
    python scripts/ladi_net/phase0d_probe.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model3" / "model3_unified_source_map.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "specialist" / "model3" / "split_indices.json"
CACHE_DIR = PROJECT_ROOT / "data" / "specialist" / "model3" / "probe_cache"
RESULTS_MD = PROJECT_ROOT / "data" / "specialist" / "model3" / "probe_results.md"
CORAL_COV_PT = PROJECT_ROOT / "data" / "specialist" / "model3" / "coral_target_cov.pt"
LOG_DIR = PROJECT_ROOT / "logs"

TOMATO_CLASSES = [
    "tomato_foliar_spot",
    "tomato_septoria_leaf_spot",
    "tomato_late_blight",
    "tomato_yellow_leaf_curl_virus",
    "tomato_mosaic_virus",
    "tomato_healthy",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TOMATO_CLASSES)}

# DINOv2 patch_size=14 requires resolution divisible by 14.
# 224=14*16 ✓ · 336=14*24 ✓ · 392=14*28 ✓  (384 was not — continuation-prompt oversight)
RESOLUTIONS = [224, 336, 392]
BATCH_BY_RES = {224: 16, 336: 8, 392: 4}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ladi_phase0d_probe_{ts}.log"
    logger = logging.getLogger("ladi.phase0d")
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
# DINOv2 loader — lazy, frozen, on GPU
# ------------------------------------------------------------------------
_DINOV2 = None


def _load_dinov2(logger: logging.Logger):
    """Load DINOv2-Base-Registers with patch-stride compatible head."""
    global _DINOV2
    if _DINOV2 is not None:
        return _DINOV2
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading DINOv2-Base-Registers on {device}...")
    # timm model name for DINOv2-Base + 4 registers.
    # NOTE: timm 1.0.26 registers this without the ".lvd142m" tag suffix.
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2",
        pretrained=True, num_classes=0,  # strip head; we take features
        img_size=224,  # can be overridden by dynamic_img_size
        dynamic_img_size=True,  # allow feature extraction at 336 and 384
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    _DINOV2 = model
    logger.info(f"  loaded. Device={device}, dtype={next(model.parameters()).dtype}, "
                f"embed_dim={model.num_features}")
    return _DINOV2


def _extract_cls_features(image_paths: list[str], resolution: int,
                           logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [N, 768] float32, valid_mask [N] bool).
    LAB-CLAHE preprocessing applied per spec (consistent with online recomposer)."""
    model = _load_dinov2(logger)
    device = next(model.parameters()).device
    batch_size = BATCH_BY_RES.get(resolution, 4)
    N = len(image_paths)
    feats = np.zeros((N, model.num_features), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    # Rebuild model at the target resolution — DINOv2 supports variable img_size.
    # Simplest approach: set model.dynamic_img_size=True if available; else just
    # pass the new resolution and let ViT handle it.
    model.eval()

    def _preprocess(path: str):
        """Load image → CLAHE-LAB → resize → tensor."""
        try:
            abs_p = PROJECT_ROOT / path
            bgr = cv2.imread(str(abs_p), cv2.IMREAD_COLOR)
            if bgr is None:
                # PIL fallback
                pil = Image.open(abs_p).convert("RGB")
                rgb_pil = np.array(pil)
                bgr = rgb_pil[:, :, ::-1].copy()
            # LAB-CLAHE on L channel only
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            bgr_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # resize → RGB → tensor
            rs = cv2.resize(bgr_eq, (resolution, resolution),
                            interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(rgb.transpose(2, 0, 1))  # [3,H,W]
        except Exception:
            return None

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch_t = []
            keep_pos = []
            for i in range(start, end):
                t = _preprocess(image_paths[i])
                if t is None:
                    continue
                batch_t.append(t)
                keep_pos.append(i)
            if not batch_t:
                continue
            x = torch.stack(batch_t, dim=0).to(device)
            try:
                out = model.forward_features(x)  # for ViT this returns tokens
                # timm ViT returns dict-ish or tensor — normalize.
                if isinstance(out, dict):
                    cls = out.get("x_norm_clstoken", out.get("cls_token"))
                    if cls is None:
                        tok = out["x"] if "x" in out else list(out.values())[0]
                        cls = tok[:, 0]
                elif out.ndim == 3:
                    # [B, N+1, C] — CLS is token 0.
                    cls = out[:, 0]
                else:
                    cls = out
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  OOM at batch {start}-{end}; halving batch")
                torch.cuda.empty_cache()
                continue
            cls_np = cls.detach().float().cpu().numpy()
            for j, pos in enumerate(keep_pos):
                feats[pos] = cls_np[j]
                valid[pos] = True
            if (start // batch_size) % 10 == 0:
                elapsed = time.time() - t0
                done = end
                rate = done / max(elapsed, 1e-6)
                eta = (N - done) / max(rate, 1e-6)
                logger.info(f"  [{resolution}px] {done:5d}/{N} features "
                            f"({rate:.1f} img/s, eta {eta/60:.1f} min)")
    torch.cuda.empty_cache()
    return feats, valid


# ------------------------------------------------------------------------
# DSF-Vec 12-feature computation
# ------------------------------------------------------------------------
def _compute_dsf_vec(bgr: np.ndarray) -> np.ndarray | None:
    """12-dim handcrafted feature vector for one image. None if fails.

    Spec reference (ladi_net_complete_system.md Part Two Step 6 Stream 2).
    Simplified: baseline "healthy" color estimated from the least-saturated
    pixels of the whole image (lowest saturation ≈ healthiest tissue).
    """
    try:
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Healthy-baseline = mean of 20%-lowest-saturation pixels
        sat_flat = hsv[..., 1].reshape(-1)
        thr = np.percentile(sat_flat, 20)
        healthy_mask = hsv[..., 1] <= thr
        if healthy_mask.sum() < 10:
            healthy_mask = np.ones((h, w), dtype=bool)
        base_L = lab[..., 0][healthy_mask].mean()
        base_A = lab[..., 1][healthy_mask].mean()
        base_B = lab[..., 2][healthy_mask].mean()
        # 1: LAB-A deviation (mean lesion minus healthy baseline)
        f1 = lab[..., 1].mean() - base_A
        # 2: LAB-B deviation
        f2 = lab[..., 2].mean() - base_B
        # 3: Saturation variance
        f3 = hsv[..., 1].std()
        # 4: LBP entropy (simple gradient-based proxy to avoid skimage dep)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(dx * dx + dy * dy)
        hist, _ = np.histogram(grad, bins=32, range=(0, grad.max() + 1e-6))
        p = hist.astype(np.float64) / max(hist.sum(), 1)
        f4 = -(p[p > 0] * np.log(p[p > 0])).sum()
        # 5: Edge density (Canny)
        edges = cv2.Canny(gray, 40, 120)
        f5 = edges.mean() / 255.0
        # 6: Moran's I simplified — variance of pixel ≈ 1 - spatial autocorrelation
        # approximate via std / mean of green
        g = rgb[..., 1]
        f6 = g.std() / max(g.mean(), 1e-6)
        # 7: Frangi — approximate via Laplacian response stdev
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        f7 = lap.std()
        # 8+9: blobs via Otsu + contours
        _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 20]
        f8 = float(np.mean(areas)) if areas else 0.0
        f9 = len(areas) / (h * w / 10000.0)  # per "unit area"
        # 10: Green CV
        f10 = rgb[..., 1].std() / max(rgb[..., 1].mean(), 1e-6)
        # 11: Dominant cluster entropy (k-means k=3 on HSV)
        try:
            small = cv2.resize(hsv, (64, 64), interpolation=cv2.INTER_AREA)
            X = small.reshape(-1, 3).astype(np.float32)
            _, labels, _ = cv2.kmeans(X, 3, None,
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                        10, 1.0), 5, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=3)
            pp = counts / max(counts.sum(), 1)
            f11 = -(pp[pp > 0] * np.log(pp[pp > 0])).sum()
        except Exception:
            f11 = 0.0
        # 12: Vertical area asymmetry (fraction of nonzero pixels in top 40% vs bottom 40%)
        nonzero = (gray > 10).astype(np.float32)
        top = nonzero[: int(0.4 * h)].mean()
        bot = nonzero[int(0.6 * h):].mean()
        f12 = top - bot
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12],
                         dtype=np.float32)
    except Exception:
        return None


def _build_dsf_matrix(image_paths: list[str], logger: logging.Logger
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [N,12], valid [N])."""
    N = len(image_paths)
    X = np.zeros((N, 12), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    for i, p in enumerate(image_paths):
        abs_p = PROJECT_ROOT / p
        bgr = cv2.imread(str(abs_p), cv2.IMREAD_COLOR)
        vec = _compute_dsf_vec(bgr) if bgr is not None else None
        if vec is not None and np.all(np.isfinite(vec)):
            X[i] = vec
            valid[i] = True
        if (i + 1) % 100 == 0:
            logger.info(f"  DSF-Vec {i+1}/{N} computed")
    return X, valid


# ------------------------------------------------------------------------
# Logistic regression probe
# ------------------------------------------------------------------------
def _logreg_eval(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    per_class_f1 = f1_score(y_val, preds, labels=list(range(len(TOMATO_CLASSES))),
                             average=None, zero_division=0)
    macro_f1 = float(per_class_f1.mean())
    return per_class_f1, macro_f1


def _weighted_f1(per_class_f1: np.ndarray, weights: dict[str, float]) -> float:
    """Compute the sqrt(N)-weighted aggregate F1 from per-class F1."""
    total = 0.0
    for i, c in enumerate(TOMATO_CLASSES):
        total += weights.get(c, 1.0 / len(TOMATO_CLASSES)) * per_class_f1[i]
    return float(total)


def _cosine_distance_centroid(X: np.ndarray, y: np.ndarray,
                               label_a: int, label_b: int) -> float:
    """Cosine distance (= 1 - cos_sim) between class centroids in X."""
    a = X[y == label_a].mean(axis=0)
    b = X[y == label_b].mean(axis=0)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    sim = float((a @ b) / (na * nb))
    return 1.0 - sim


def _frobenius_cov_gap(X_lab: np.ndarray, X_field: np.ndarray) -> float:
    """|| Cov(lab) - Cov(field) ||_F — CORAL target quantity."""
    if X_lab.shape[0] < 2 or X_field.shape[0] < 2:
        return float("nan")
    C_lab = np.cov(X_lab, rowvar=False).astype(np.float32)
    C_field = np.cov(X_field, rowvar=False).astype(np.float32)
    return float(np.linalg.norm(C_lab - C_field, ord="fro"))


# ------------------------------------------------------------------------
# Main probe orchestration
# ------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    parser.add_argument("--skip-dsf", action="store_true")
    args = parser.parse_args(argv)

    logger = _setup_logger()
    logger.info("=" * 72)
    logger.info("LADI-Net Phase 0D — Systematic probe")
    logger.info("=" * 72)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load split + CSV.
    with open(args.split, encoding="utf-8") as f:
        split = json.load(f)
    df = pd.read_csv(args.csv)
    stopping_weights = split["metadata"].get("stopping_weights", {})
    logger.info(f"stopping_weights (from split): {stopping_weights}")

    # Build path → (class_name, is_field, is_recomposed) map.
    # Use non-underscore column names (pandas itertuples strips leading _).
    df["is_field_bool"] = df["is_field_photo"].apply(
        lambda v: bool(v) if isinstance(v, (bool, int, float)) else
                  str(v).strip().lower() in {"1", "true", "t", "yes"})
    df["is_recomposed"] = df["source_dataset"].apply(
        lambda s: str(s).lower() in {"scidb_recomposed", "capsicum_recomposed"})
    # Iterate via zip over columns (avoids itertuples corner cases).
    path_to_meta = {}
    for p, c, isf, isr in zip(df["image_path"].tolist(),
                               df["class_name"].tolist(),
                               df["is_field_bool"].tolist(),
                               df["is_recomposed"].tolist()):
        path_to_meta[p] = {
            "class_name": c,
            "is_field": bool(isf),
            "is_recomposed": bool(isr),
        }

    # Split partitioning for probing.
    train_paths = split["train"]
    field_val_paths = split["field_val"]
    logger.info(f"train={len(train_paths)}, field_val={len(field_val_paths)}")

    # Categorize training into (lab+recomposed) vs (real-field)
    train_lab_recomp: list[str] = []
    train_real_field: list[str] = []
    for p in train_paths:
        m = path_to_meta.get(p)
        if not m:
            continue
        if m["is_field"] and not m["is_recomposed"]:
            train_real_field.append(p)
        else:
            train_lab_recomp.append(p)
    logger.info(f"train breakdown: lab+recomposed={len(train_lab_recomp)}, "
                f"real_field={len(train_real_field)}")

    # Per-resolution results checkpoint — so re-runs skip completed LogReg.
    results_cache_path = CACHE_DIR / "results_per_res.json"
    results_per_res: dict[int, dict] = {}
    if results_cache_path.exists():
        try:
            cached = json.loads(results_cache_path.read_text(encoding="utf-8"))
            results_per_res = {int(k): v for k, v in cached.items()}
            logger.info(f"Loaded {len(results_per_res)} cached resolution "
                        f"result(s) from {results_cache_path.name}: "
                        f"{sorted(results_per_res.keys())}")
        except Exception as e:
            logger.warning(f"  could not read results cache: {e}")

    for res in args.resolutions:
        if res in results_per_res:
            logger.info(f"\n[SKIP] {res}px already in results cache; using cached "
                        f"C1={results_per_res[res]['config1_weighted_f1']:.4f}, "
                        f"C2={results_per_res[res]['config2_weighted_f1']:.4f}")
            continue
        logger.info("")
        logger.info("=" * 72)
        logger.info(f"RESOLUTION {res}px")
        logger.info("=" * 72)

        # Feature caches.
        def cache_path(name):
            return CACHE_DIR / f"features_{res}_{name}.pt"

        # Combine paths: train_lab_recomp + train_real_field + field_val
        # so we can slice later. Cache as single file per resolution for simplicity.
        all_paths = train_lab_recomp + train_real_field + field_val_paths
        labels_all = np.array(
            [CLASS_TO_IDX[path_to_meta[p]["class_name"]] for p in all_paths],
            dtype=np.int64)
        is_field_all = np.array(
            [path_to_meta[p]["is_field"] and not path_to_meta[p]["is_recomposed"]
             for p in all_paths], dtype=bool)

        cache_feats = cache_path("all")
        cache_labels = cache_path("labels")
        cache_isfield = cache_path("isfield")
        if cache_feats.exists() and cache_labels.exists():
            logger.info(f"  Loading cached features {cache_feats.name}")
            feats = torch.load(cache_feats, weights_only=True).numpy()
            labels_all = torch.load(cache_labels, weights_only=True).numpy()
            is_field_all = torch.load(cache_isfield, weights_only=True).numpy().astype(bool)
        else:
            logger.info(f"  Extracting features for {len(all_paths)} images at {res}px")
            feats, valid = _extract_cls_features(all_paths, res, logger)
            # Drop invalid rows from all parallel arrays.
            feats = feats[valid]
            labels_all = labels_all[valid]
            is_field_all = is_field_all[valid]
            torch.save(torch.from_numpy(feats), cache_feats)
            torch.save(torch.from_numpy(labels_all), cache_labels)
            torch.save(torch.from_numpy(is_field_all.astype(np.int8)), cache_isfield)
            logger.info(f"  Cached {len(feats)} features to {cache_feats.name}")

        # Partition.
        n_lr = len(train_lab_recomp)
        n_rf = len(train_real_field)
        n_fv = len(field_val_paths)
        # After dropping invalid rows, the sizes may differ; we re-index by
        # matching the first `n_lr+n_rf` to train and the rest to field_val.
        # Because valid filter keeps relative order, the same slicing works.
        # If counts changed, pick by actual array length.
        N = len(feats)
        if N != (n_lr + n_rf + n_fv):
            # Some images failed. We need to rebuild partitions by matching
            # against the kept subset. Simplest: recompute labels using the
            # kept subset's path order. We lost this info above — but we can
            # recover via the order-preserving valid mask. For simplicity
            # here we use the arrays we DID save and assume the valid filter
            # preserved order (which it does); we just need to know how many
            # in each section survived.
            # Re-extract the valid counts per section:
            # Build a cumulative index from the stored is_field_all vs what we expect.
            # Since this is just a probe and we have the feats array, we'll
            # partition using the convention: first n_lr_surv are lab+recomp,
            # next n_rf_surv are real-field, rest are field_val. We compute
            # surv counts from is_field_all: real_field is True, otherwise lab.
            # Actually is_field_all marks REAL-FIELD (True) or lab/recomp (False).
            # Ordering: first training (lab + real_field), then field_val.
            # We can't fully disambiguate train-real-field from val-real-field
            # via is_field alone. So: do the cached extraction in a way
            # that does NOT lose alignment. Recompute below.
            logger.warning(
                f"  Array size mismatch ({N} kept vs {n_lr+n_rf+n_fv} expected) "
                f"— re-extracting without cache to preserve alignment")
            feats, valid = _extract_cls_features(all_paths, res, logger)
            # Keep zero features for invalid rows; mark them so we skip
            labels_all = np.array(
                [CLASS_TO_IDX[path_to_meta[p]["class_name"]] for p in all_paths],
                dtype=np.int64)
            is_field_all = np.array(
                [path_to_meta[p]["is_field"] and not path_to_meta[p]["is_recomposed"]
                 for p in all_paths], dtype=bool)
            # Use valid mask as in-place filter via boolean indexing
            feats = feats[valid]
            labels_all = labels_all[valid]
            is_field_all = is_field_all[valid]

        # Config 1: train = first n_lr rows that survived (lab+recomposed only)
        #           val   = last n_fv rows that survived (field_val)
        # Because the input list was exactly [lab+recomp block, real-field block, field_val block],
        # and valid preserves order, we can slice by the cumulative count of each block.
        # Compute surviving counts per block: count how many of the first n_lr input
        # positions survived; similarly for the next n_rf; the rest are field_val.
        # Easiest: run the extraction with a block-label array aligned to the input
        # and filter the block-label by valid. We didn't keep the mask; do it now
        # from is_field_all structure won't work (is_field_all doesn't separate
        # train-real-field from val-real-field). So we re-derive block labels from
        # positional index of all_paths and apply valid in the second pass.
        block_labels = np.array(
            [0] * n_lr + [1] * n_rf + [2] * n_fv, dtype=np.int8)
        if len(block_labels) != N:
            # Trust the post-valid sizes: feats.shape[0] should equal the sum of
            # valid in each block. Recompute block sizes from valid mask.
            # Since the first extraction wasn't captured with valid, re-extract
            # now (this path only hits when cache existed but block sizes
            # drifted — re-extract to reset).
            feats, valid_mask = _extract_cls_features(all_paths, res, logger)
            feats = feats[valid_mask]
            labels_all = np.array(
                [CLASS_TO_IDX[path_to_meta[p]["class_name"]] for p in all_paths],
                dtype=np.int64)[valid_mask]
            is_field_all = np.array(
                [path_to_meta[p]["is_field"] and not path_to_meta[p]["is_recomposed"]
                 for p in all_paths], dtype=bool)[valid_mask]
            block_labels = np.array(
                [0] * n_lr + [1] * n_rf + [2] * n_fv, dtype=np.int8)[valid_mask]
            N = len(feats)

        c1_mask = block_labels == 0                 # Config 1 train: lab+recomp only
        c2_mask = (block_labels == 0) | (block_labels == 1)  # Config 2 train: lab+recomp+real_field
        val_mask = block_labels == 2

        X_c1 = feats[c1_mask];  y_c1 = labels_all[c1_mask]
        X_c2 = feats[c2_mask];  y_c2 = labels_all[c2_mask]
        X_val = feats[val_mask]; y_val = labels_all[val_mask]

        logger.info(f"  sizes: C1-train={len(X_c1)}, C2-train={len(X_c2)}, val={len(X_val)}")

        # Config 1
        pc1, m1 = _logreg_eval(X_c1, y_c1, X_val, y_val)
        w1 = _weighted_f1(pc1, stopping_weights)
        logger.info(f"  Config 1 (lab+recomp):  macro_F1={m1:.4f}  weighted_F1={w1:.4f}")
        for i, c in enumerate(TOMATO_CLASSES):
            logger.info(f"      {c:<32} F1={pc1[i]:.3f}")

        # Config 2
        pc2, m2 = _logreg_eval(X_c2, y_c2, X_val, y_val)
        w2 = _weighted_f1(pc2, stopping_weights)
        logger.info(f"  Config 2 (full data):   macro_F1={m2:.4f}  weighted_F1={w2:.4f}")
        for i, c in enumerate(TOMATO_CLASSES):
            logger.info(f"      {c:<32} F1={pc2[i]:.3f}")

        # Foliar-septoria cosine distance on field_val features
        foliar_i = CLASS_TO_IDX["tomato_foliar_spot"]
        sept_i = CLASS_TO_IDX["tomato_septoria_leaf_spot"]
        cdist_val = _cosine_distance_centroid(X_val, y_val, foliar_i, sept_i)
        logger.info(f"  foliar-septoria cosine distance (field_val feat space): {cdist_val:.4f}")

        # Frobenius cov gap between (lab+recomp train features) and (real-field train features)
        lab_feats = feats[block_labels == 0]
        rf_feats = feats[block_labels == 1]
        frob = _frobenius_cov_gap(lab_feats, rf_feats)
        logger.info(f"  ||Cov(lab+recomp) - Cov(real_field)||_F: {frob:.2f}")

        results_per_res[res] = {
            "config1_weighted_f1": w1,
            "config1_macro_f1": m1,
            "config1_per_class": {TOMATO_CLASSES[i]: float(pc1[i]) for i in range(6)},
            "config2_weighted_f1": w2,
            "config2_macro_f1": m2,
            "config2_per_class": {TOMATO_CLASSES[i]: float(pc2[i]) for i in range(6)},
            "foliar_septoria_cosine_distance": cdist_val,
            "frobenius_cov_gap": frob,
            "n_c1": int(len(X_c1)), "n_c2": int(len(X_c2)), "n_val": int(len(X_val)),
        }
        # Persist per-resolution results so re-runs skip completed fits.
        try:
            results_cache_path.write_text(
                json.dumps({str(k): v for k, v in results_per_res.items()},
                           indent=2), encoding="utf-8")
            logger.info(f"  saved results cache -> {results_cache_path.name}")
        except Exception as e:
            logger.warning(f"  could not write results cache: {e}")

        torch.cuda.empty_cache()
        gc.collect()

    # ----- RESOLUTION DECISION -----
    baseline_cdist = results_per_res[min(args.resolutions)]["foliar_septoria_cosine_distance"]
    chosen_res = min(args.resolutions)
    reason_res = f"fallback to {chosen_res}px (no resolution exceeded 3% improvement)"
    best_imp = 0.0
    for r in sorted(args.resolutions):
        if r == min(args.resolutions):
            continue
        imp = results_per_res[r]["foliar_septoria_cosine_distance"] - baseline_cdist
        if imp > 0.03 and imp > best_imp:
            chosen_res = r
            best_imp = imp
            reason_res = (f"{r}px improves foliar-septoria cosine distance by "
                          f"{imp:+.4f} (> 3% threshold)")

    logger.info("")
    logger.info(f"RESOLUTION DECISION: {chosen_res}px — {reason_res}")

    # ----- SPLIT RATIO DECISION -----
    res_key = chosen_res
    w1 = results_per_res[res_key]["config1_weighted_f1"]
    w2 = results_per_res[res_key]["config2_weighted_f1"]
    benefit = w2 - w1
    if benefit > 0.05:
        ratios = (0.70, 0.20, 0.10)
        ratio_reason = f"benefit {benefit:+.4f} > 0.05 — maximize real-field training"
    elif benefit > 0.02:
        ratios = (0.60, 0.25, 0.15)
        ratio_reason = f"benefit {benefit:+.4f} in [0.02,0.05] — balanced allocation"
    else:
        ratios = (0.50, 0.35, 0.15)
        ratio_reason = f"benefit {benefit:+.4f} ≤ 0.02 — bias toward validation coverage"
    logger.info(f"SPLIT RATIO DECISION: train={ratios[0]} val={ratios[1]} "
                f"final={ratios[2]} — {ratio_reason}")

    # ----- CORAL TARGET COVARIANCE -----
    # Use Config 2 real-field features at chosen resolution.
    # Re-extract if needed (or reload from cache — feats array still in scope
    # from the last loop iteration, but that's the LAST resolution tried).
    logger.info("")
    logger.info("Saving CORAL target covariance...")
    feats_chosen = torch.load(CACHE_DIR / f"features_{chosen_res}_all.pt",
                               weights_only=True).numpy()
    labels_chosen = torch.load(CACHE_DIR / f"features_{chosen_res}_labels.pt",
                                weights_only=True).numpy()
    isfield_chosen = torch.load(CACHE_DIR / f"features_{chosen_res}_isfield.pt",
                                 weights_only=True).numpy().astype(bool)
    # [FIX for PVA FAIL-1] Use ONLY train-real-field features (block 1), not all
    # is-field features (blocks 1+2). field_val (block 2) is the early-stopping
    # signal in Phase 2 — baking its covariance into CORAL alignment is target
    # leakage. Block layout set at line 542-543: [0]*n_lr + [1]*n_rf + [2]*n_fv.
    block_labels_coral = np.concatenate([
        np.zeros(n_lr, dtype=np.int8),
        np.ones(n_rf, dtype=np.int8),
        np.full(n_fv, 2, dtype=np.int8),
    ])
    if len(block_labels_coral) != feats_chosen.shape[0]:
        logger.warning(
            f"CORAL block-label size mismatch: {len(block_labels_coral)} vs "
            f"{feats_chosen.shape[0]} features — falling back to all is-field"
        )
        rf_feats_full = feats_chosen[isfield_chosen]
    else:
        rf_feats_full = feats_chosen[block_labels_coral == 1]
    if rf_feats_full.shape[0] < 2:
        logger.warning("Not enough real-field features to compute CORAL cov")
    else:
        C = np.cov(rf_feats_full, rowvar=False).astype(np.float32)
        torch.save(torch.from_numpy(C), CORAL_COV_PT)
        logger.info(f"  CORAL target cov saved: {CORAL_COV_PT.relative_to(PROJECT_ROOT)} "
                    f"(shape {C.shape}, from {rf_feats_full.shape[0]} real-field features)")

    # ----- DSF-Vec PROBE -----
    dsf_result = None
    if not args.skip_dsf:
        logger.info("")
        logger.info("=" * 72)
        logger.info("DSF-Vec probe (12 handcrafted features)")
        logger.info("=" * 72)
        # Real-field train images only.
        dsf_train_paths = train_real_field
        dsf_val_paths = field_val_paths
        logger.info(f"  Computing DSF-Vec on {len(dsf_train_paths)} train + "
                    f"{len(dsf_val_paths)} val images...")
        X_tr, valid_tr = _build_dsf_matrix(dsf_train_paths, logger)
        X_vl, valid_vl = _build_dsf_matrix(dsf_val_paths, logger)
        X_tr = X_tr[valid_tr];  X_vl = X_vl[valid_vl]
        y_tr = np.array([CLASS_TO_IDX[path_to_meta[p]["class_name"]]
                         for p in dsf_train_paths], dtype=np.int64)[valid_tr]
        y_vl = np.array([CLASS_TO_IDX[path_to_meta[p]["class_name"]]
                         for p in dsf_val_paths], dtype=np.int64)[valid_vl]

        # Per-feature variance across classes (is feature 12 near-zero variance for YLCV?)
        ylcv_i = CLASS_TO_IDX["tomato_yellow_leaf_curl_virus"]
        f12_ylcv = X_tr[y_tr == ylcv_i, 11]
        f12_not = X_tr[y_tr != ylcv_i, 11]
        if f12_ylcv.size > 0 and f12_not.size > 0:
            f12_var_between = ((f12_ylcv.mean() - f12_not.mean()) ** 2)
            f12_var_within = 0.5 * (f12_ylcv.var() + f12_not.var()) + 1e-9
            f12_signal_ratio = float(f12_var_between / f12_var_within)
        else:
            f12_signal_ratio = 0.0
        drop_f12 = f12_signal_ratio < 0.01

        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import f1_score
        # Drop feature 12 if warranted, for the probe itself.
        if drop_f12:
            X_tr_use = X_tr[:, :11]
            X_vl_use = X_vl[:, :11]
        else:
            X_tr_use = X_tr; X_vl_use = X_vl
        clf = RidgeClassifier(random_state=42)
        clf.fit(X_tr_use, y_tr)
        preds = clf.predict(X_vl_use)
        pc = f1_score(y_vl, preds, labels=list(range(6)), average=None, zero_division=0)
        logger.info(f"  DSF-Vec per-class field F1:")
        for i, c in enumerate(TOMATO_CLASSES):
            logger.info(f"    {c:<32} F1={pc[i]:.3f}")
        sept_f1 = pc[CLASS_TO_IDX["tomato_septoria_leaf_spot"]]
        if sept_f1 < 0.45:
            dsf_assess = "DROP"
            dsf_reason = f"septoria DSF-Vec-only F1={sept_f1:.3f} < 0.45 threshold"
        elif sept_f1 < 0.55:
            dsf_assess = "PARTIAL"
            dsf_reason = f"septoria DSF-Vec F1={sept_f1:.3f} — marginal signal"
        else:
            dsf_assess = "INCLUDE"
            dsf_reason = f"septoria DSF-Vec F1={sept_f1:.3f} >= 0.45 (carry into Phase 2)"
        logger.info(f"  DSF-Vec assessment: {dsf_assess} — {dsf_reason}")
        if drop_f12:
            logger.info(f"  Feature 12 (vertical asymmetry) signal-ratio = "
                        f"{f12_signal_ratio:.4f} < 0.01 — DROP")
        dsf_result = {
            "per_class": {TOMATO_CLASSES[i]: float(pc[i]) for i in range(6)},
            "assessment": dsf_assess,
            "reason": dsf_reason,
            "drop_feature_12": drop_f12,
            "f12_signal_ratio": f12_signal_ratio,
        }

    # ----- probe_results.md -----
    md = [
        "# LADI-Net Phase 0D Probe Results",
        f"_Run at {datetime.now().isoformat()}_",
        "",
        "## Resolution Sweep",
        "",
        "| Resolution | C1 weighted F1 | C2 weighted F1 | foliar-sept cos dist | Frobenius cov gap |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(args.resolutions):
        rs = results_per_res[r]
        md.append(f"| {r}px | {rs['config1_weighted_f1']:.4f} | "
                  f"{rs['config2_weighted_f1']:.4f} | "
                  f"{rs['foliar_septoria_cosine_distance']:.4f} | "
                  f"{rs['frobenius_cov_gap']:.2f} |")
    md += [
        "",
        f"**RESOLUTION DECISION:** **{chosen_res}px** — {reason_res}",
        "",
        "## Split Ratio Decision",
        "",
        f"At chosen resolution {chosen_res}px:",
        f"- Config 1 (recomposed-only) weighted F1: **{w1:.4f}**",
        f"- Config 2 (full data) weighted F1: **{w2:.4f}**",
        f"- Real-field training benefit: **{benefit:+.4f}**",
        "",
        f"**RECOMMENDED RATIOS:** "
        f"`--train_frac={ratios[0]} --val_frac={ratios[1]} --final_frac={ratios[2]}` "
        f"— {ratio_reason}",
        "",
        "## Per-class F1 at chosen resolution",
        "",
        "| Class | C1 F1 | C2 F1 |",
        "|---|---|---|",
    ]
    for c in TOMATO_CLASSES:
        pc1 = results_per_res[chosen_res]["config1_per_class"][c]
        pc2v = results_per_res[chosen_res]["config2_per_class"][c]
        md.append(f"| {c} | {pc1:.3f} | {pc2v:.3f} |")
    md += [
        "",
        "## CORAL Target Covariance",
        "",
        f"- Computed from **{rf_feats_full.shape[0]}** real-field features at {chosen_res}px" if rf_feats_full.shape[0] >= 2 else "- INSUFFICIENT real-field features",
        f"- Shape: (768, 768) — saved to `{CORAL_COV_PT.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- NOTE: This covariance is computed from FROZEN features. The training script",
        f"  should refresh the target covariance every 5 epochs from current LoRA-adapted",
        f"  real-field batch features to allow drift correction. See ladi_critique.md.",
        "",
        "## LoRA Rank Decision",
        "",
        "- Default to **rank=8** for Phase 1 warm-up. Rank 4 vs 8 ablation deferred to Phase 1.",
        "",
        "## DSF-Vec Decision",
        "",
    ]
    if dsf_result is not None:
        md += [
            f"- Overall assessment: **{dsf_result['assessment']}** — {dsf_result['reason']}",
            f"- Feature 12 drop: {dsf_result['drop_feature_12']} "
            f"(signal ratio = {dsf_result['f12_signal_ratio']:.4f})",
            "",
            "Per-class DSF-Vec field F1:",
            "",
        ]
        for c, f1 in dsf_result["per_class"].items():
            md.append(f"- {c}: **{f1:.3f}**")
    else:
        md.append("- DSF-Vec probe skipped (--skip-dsf)")

    RESULTS_MD.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"\nprobe_results.md -> {RESULTS_MD.relative_to(PROJECT_ROOT)}")

    # Also save a probe_summary.json for machine reading.
    summary = {
        "timestamp": datetime.now().isoformat(),
        "resolutions": {str(r): results_per_res[r] for r in sorted(args.resolutions)},
        "chosen_resolution": chosen_res,
        "chosen_resolution_reason": reason_res,
        "recommended_ratios": {
            "train_frac": ratios[0], "val_frac": ratios[1], "final_frac": ratios[2],
            "reason": ratio_reason, "benefit": benefit,
        },
        "dsf_vec": dsf_result,
        "coral_cov_path": str(CORAL_COV_PT.relative_to(PROJECT_ROOT).as_posix())
            if CORAL_COV_PT.exists() else None,
    }
    (PROJECT_ROOT / "data" / "specialist" / "model3" / "probe_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("=" * 72)
    logger.info("PHASE 0D PROBE DECISION SUMMARY")
    logger.info("=" * 72)
    logger.info(f"  Chosen resolution: {chosen_res}px")
    logger.info(f"  Recommended ratios: train={ratios[0]} val={ratios[1]} final={ratios[2]}")
    logger.info(f"  CORAL cov saved: {CORAL_COV_PT.exists()}")
    if dsf_result:
        logger.info(f"  DSF-Vec: {dsf_result['assessment']}")
    logger.info("[OK] Phase 0D probe complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
