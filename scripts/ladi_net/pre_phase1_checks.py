"""
LADI-Net Pre-Phase-1 diagnostic checks.

Runs 8 checks to surface any data-level, architecture-level, or configuration
bugs that would prevent Phase 1 training from starting correctly. Output is
written to `data/specialist/model3/pre_phase1_check_results.json` and printed
to stdout.

All checks prefer GPU where applicable. Designed for Windows/RTX 4060.

Usage:
    python scripts/ladi_net/pre_phase1_checks.py
"""

from __future__ import annotations

import datetime
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL3_DIR = PROJECT_ROOT / "data" / "specialist" / "model3"
MASK_LOG = MODEL3_DIR / "mask_precompute_log.csv"
SPLIT_PATH = MODEL3_DIR / "split_indices.json"
CORAL_PATH = MODEL3_DIR / "coral_target_cov.pt"
CORAL_BACKUP_224 = MODEL3_DIR / "coral_target_cov.at_224px_backup.pt"
CORAL_BACKUP_LEAKY = MODEL3_DIR / "coral_target_cov.leaky_backup.pt"
RESULTS_PATH = MODEL3_DIR / "pre_phase1_check_results.json"

TOMATO_CLASSES = [
    "tomato_foliar_spot",
    "tomato_septoria_leaf_spot",
    "tomato_late_blight",
    "tomato_yellow_leaf_curl_virus",
    "tomato_mosaic_virus",
    "tomato_healthy",
]

FLAG_REASON_COL = "flag_reason"


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Check 1 — Per-class recomposition eligibility
# ---------------------------------------------------------------------------
def check_1_recomposition_eligibility(df: pd.DataFrame) -> dict:
    _banner("CHECK 1 — Per-class recomposition eligibility")
    tdf = df[df["class_name"].isin(TOMATO_CLASSES)].copy()
    tdf = tdf[tdf["is_field_photo"] == 0]  # lab only

    per_class: dict[str, dict] = {}
    for cls in TOMATO_CLASSES:
        sub = tdf[tdf["class_name"] == cls]
        n_total = len(sub)
        n_flagged = int(sub["flagged"].sum())
        n_eligible = n_total - n_flagged
        eligibility = n_eligible / n_total if n_total > 0 else 0.0
        flag_breakdown = (
            sub[sub["flagged"] == 1][FLAG_REASON_COL]
            .fillna("unknown")
            .value_counts()
            .to_dict()
        )
        per_class[cls] = {
            "n_lab_total": n_total,
            "n_flagged": n_flagged,
            "n_eligible": n_eligible,
            "eligibility_pct": round(100 * eligibility, 2),
            "flag_breakdown": flag_breakdown,
        }
        print(f"  {cls:35s}: {n_eligible:5d}/{n_total:5d} "
              f"= {100*eligibility:5.2f}% eligible "
              f"| flags={flag_breakdown}")

    # Healthy vs avg-disease comparison
    healthy_pct = per_class["tomato_healthy"]["eligibility_pct"]
    disease_pcts = [per_class[c]["eligibility_pct"]
                    for c in TOMATO_CLASSES if c != "tomato_healthy"]
    avg_disease_pct = float(np.mean(disease_pcts))
    gap = avg_disease_pct - healthy_pct
    bias_detected = gap > 8.0

    # Check flag-reason type for healthy (matters for stop-condition triage)
    healthy_flags = per_class["tomato_healthy"]["flag_breakdown"]
    total_flags = sum(healthy_flags.values()) if healthy_flags else 0
    high_cov_frac = (healthy_flags.get("high_coverage", 0) / total_flags
                     if total_flags > 0 else 0.0)

    print(f"\n  Healthy eligibility : {healthy_pct:.2f}%")
    print(f"  Avg disease class   : {avg_disease_pct:.2f}%")
    print(f"  Gap (disease-healthy): {gap:+.2f} pp")
    if bias_detected:
        print(f"  [WARN] Healthy eligibility > 8 pp below disease avg — BACKGROUND BIAS RISK")
        print(f"  Healthy high_coverage fraction of flags: {high_cov_frac:.2%}")
        if high_cov_frac > 0.5:
            print(f"         (>50% of healthy flags are high_coverage — these may be "
                  f"valid recomposition candidates once mask coverage policy relaxed)")
    else:
        print(f"  [OK] Healthy eligibility gap within tolerance")

    # flag_reason dominance check
    all_flag_reasons = defaultdict(int)
    for d in per_class.values():
        for k, v in d["flag_breakdown"].items():
            all_flag_reasons[k] += v
    total_all = sum(all_flag_reasons.values())
    high_cov_global = all_flag_reasons.get("high_coverage", 0) / total_all if total_all else 0
    if high_cov_global > 0.30:
        print(f"  [WARN] high_coverage flags dominate ({high_cov_global:.2%} of all flags) — "
              f"re-check whether these images are truly unusable")

    return {
        "per_class": per_class,
        "healthy_eligibility_pct": healthy_pct,
        "avg_disease_eligibility_pct": avg_disease_pct,
        "gap_pp": round(gap, 2),
        "bias_detected": bias_detected,
        "healthy_high_coverage_frac_of_flags": round(high_cov_frac, 4),
        "global_high_coverage_frac_of_flags": round(high_cov_global, 4),
    }


# ---------------------------------------------------------------------------
# Check 2 — Batch composition simulation
# ---------------------------------------------------------------------------
def check_2_batch_simulation(split: dict, source_map_df: pd.DataFrame | None) -> dict:
    _banner("CHECK 2 — Batch composition simulation (100 batches of 16, field-weight 8x)")

    train_paths = split["train"]
    # Build a path → class_name lookup. The split indices store absolute paths.
    # We derive class from the path substring (robust to cwd).
    path_to_class: dict[str, str] = {}
    path_to_isfield: dict[str, bool] = {}
    for p in train_paths:
        # Class is in the directory name after "cleaned/"
        p_norm = str(p).replace("\\", "/")
        cls = None
        for c in TOMATO_CLASSES:
            if f"/cleaned/{c}/" in p_norm:
                cls = c
                break
        if cls is None:
            continue
        path_to_class[p] = cls
        # Lab vs real-field vs recomposed: recomposed paths contain "/recomposed/"
        # real-field by source_map_df lookup if present
        is_recomp = "recomposed" in p_norm.lower()
        path_to_isfield[p] = False  # default: treat as lab for sampling weight
        if is_recomp:
            path_to_isfield[p] = False
        else:
            # heuristic: check source map if available
            if source_map_df is not None and p in set(source_map_df["image_path"]):
                match = source_map_df[source_map_df["image_path"] == p]
                if len(match) > 0:
                    path_to_isfield[p] = bool(int(match.iloc[0].get("is_field_photo", 0)))

    # Build per-class index lists
    class_to_paths: dict[str, list[str]] = {c: [] for c in TOMATO_CLASSES}
    for p, c in path_to_class.items():
        class_to_paths[c].append(p)
    for c, lst in class_to_paths.items():
        print(f"  train pool [{c:35s}]: {len(lst):5d}")

    # Weighted sampler: field weight 8x, lab/recomp weight 1x
    all_paths = list(path_to_class.keys())
    rng = np.random.default_rng(42)
    weights = np.array([8.0 if path_to_isfield[p] else 1.0 for p in all_paths])
    probs = weights / weights.sum()
    path_idx_array = np.array(all_paths, dtype=object)

    N_BATCHES = 100
    BS = 16
    batches_with_both_confusable = 0
    zero_class_batch_count = 0
    per_class_counts_accumulator = {c: 0 for c in TOMATO_CLASSES}
    per_batch_class_counts = []
    for _ in range(N_BATCHES):
        picks = rng.choice(path_idx_array, size=BS, replace=False, p=probs)
        batch_classes = [path_to_class[p] for p in picks]
        cnt = Counter(batch_classes)
        per_batch_class_counts.append(dict(cnt))
        if cnt.get("tomato_foliar_spot", 0) >= 2 and cnt.get("tomato_septoria_leaf_spot", 0) >= 2:
            batches_with_both_confusable += 1
        if any(cnt.get(c, 0) == 0 for c in TOMATO_CLASSES):
            zero_class_batch_count += 1
        for c in TOMATO_CLASSES:
            per_class_counts_accumulator[c] += cnt.get(c, 0)

    per_class_avg = {c: per_class_counts_accumulator[c] / N_BATCHES for c in TOMATO_CLASSES}
    pct_both_confusable = batches_with_both_confusable / N_BATCHES
    pct_any_zero = zero_class_batch_count / N_BATCHES

    print(f"\n  Simulated {N_BATCHES} batches of {BS}:")
    print(f"  Fraction with >=2 foliar AND >=2 septoria : {pct_both_confusable*100:.1f}%")
    print(f"  Fraction with any class == 0              : {pct_any_zero*100:.1f}%")
    print(f"  Average per-class count per batch:")
    for c, avg in per_class_avg.items():
        print(f"     {c:35s}: {avg:5.2f}")

    if pct_both_confusable < 0.70:
        print(f"\n  [WARN] SUPCON CONFUSABLE PAIR COVERAGE INSUFFICIENT "
              f"({pct_both_confusable*100:.1f}% < 70%) — "
              f"class-stratified batch sampler required for Phase 2")
    else:
        print(f"\n  [OK] Confusable pair coverage adequate under weighted sampler")

    return {
        "n_batches_simulated": N_BATCHES,
        "batch_size": BS,
        "pct_with_both_confusable_ge2": round(pct_both_confusable * 100, 2),
        "pct_with_any_class_zero": round(pct_any_zero * 100, 2),
        "per_class_avg_per_batch": {k: round(v, 3) for k, v in per_class_avg.items()},
        "threshold_70_met": pct_both_confusable >= 0.70,
    }


# ---------------------------------------------------------------------------
# Check 3 — Background pool size and diversity
# ---------------------------------------------------------------------------
def check_3_background_pool() -> dict:
    _banner("CHECK 3 — Background pool size and reuse estimate")
    # Import the recomposer's default background dirs
    sys_path_before = list(sys.path)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ladi_net"))
    try:
        from background_recomposer import (
            DEFAULT_BG_DIRS, VALID_EXTS
        )
    finally:
        sys.path = sys_path_before

    from pathlib import Path as _P
    total = 0
    per_dir: dict[str, int] = {}
    for d in DEFAULT_BG_DIRS:
        dp = _P(d)
        if not dp.exists():
            per_dir[str(d)] = 0
            continue
        count = 0
        for f in dp.iterdir():
            if f.is_file() and f.suffix in VALID_EXTS:
                if f.name.endswith("_mask.png") or f.name.endswith("_fg.png"):
                    continue
                count += 1
        per_dir[str(d)] = count
        total += count

    # Verify preload settings in recomposer source
    recomposer_py = (PROJECT_ROOT / "scripts" / "ladi_net"
                     / "background_recomposer.py").read_text(encoding="utf-8")
    preload_max_392 = "preload_max: int = 2000" in recomposer_py
    preload_res_392 = "preload_resize_max_dim: int = 392" in recomposer_py

    # Estimate reuse
    lab_images = 22441
    p_recomp = 0.70
    n_epochs_hypothetical = 25
    reuse_per_bg_per_epoch = (lab_images * p_recomp) / max(1, total)
    reuse_over_training = reuse_per_bg_per_epoch * n_epochs_hypothetical

    print(f"  Background pool (across all dirs): {total} images")
    for d, c in per_dir.items():
        print(f"    {d}: {c}")
    print(f"  Preload resolution is 392px : {preload_res_392}")
    print(f"  Preload cap is 2000 images  : {preload_max_392}")
    print(f"\n  Estimated reuse per background per epoch "
          f"(22441 * 0.70 / {total} = {reuse_per_bg_per_epoch:.2f})")
    print(f"  Estimated reuse over {n_epochs_hypothetical} epochs: {reuse_over_training:.1f}")

    if reuse_per_bg_per_epoch > 50:
        print(f"  [WARN] BACKGROUND DIVERSITY CONCERN — backgrounds will repeat "
              f">50 times/epoch")
    else:
        print(f"  [OK] Background reuse within acceptable bounds")

    return {
        "total_pool_size": total,
        "per_dir": per_dir,
        "preload_resolution_392": preload_res_392,
        "preload_max_2000": preload_max_392,
        "reuse_per_bg_per_epoch": round(reuse_per_bg_per_epoch, 2),
        "reuse_over_25_epochs": round(reuse_over_training, 1),
        "diversity_concern": reuse_per_bg_per_epoch > 50,
    }


# ---------------------------------------------------------------------------
# Check 4 — CORAL target type verification
# ---------------------------------------------------------------------------
def check_4_coral_target() -> dict:
    _banner("CHECK 4 — CORAL target covariance verification")
    if not CORAL_PATH.exists():
        print(f"  [FAIL] CORAL not found at {CORAL_PATH}")
        return {"error": "CORAL file missing"}
    cov = torch.load(CORAL_PATH, weights_only=True)
    arr = cov.numpy() if torch.is_tensor(cov) else np.asarray(cov)

    frob = float(np.linalg.norm(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    max_val = float(np.max(arr))
    min_val = float(np.min(arr))

    print(f"  Path     : {CORAL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Shape    : {arr.shape}")
    print(f"  Dtype    : {arr.dtype}")
    print(f"  Mean     : {mean:.6f}")
    print(f"  Std      : {std:.6f}")
    print(f"  Min/Max  : {min_val:.4f} / {max_val:.4f}")
    print(f"  Frobenius: {frob:.4f}")

    print(f"\n  [CRITICAL WARNING] THIS FILE WAS COMPUTED FROM FROZEN CLS FEATURES.")
    print(f"  It is the WRONG FEATURE TYPE for CORAL loss in Phase 2.")
    print(f"  CORAL loss applies to ABMIL features (768-dim lesion bag feature).")
    print(f"  DO NOT USE this file directly in Phase 2 training.")
    print(f"  Recompute AFTER Phase 1 completes using ABMIL features from 680 "
          f"train-real-field images.")

    backup_224_exists = CORAL_BACKUP_224.exists()
    backup_leaky_exists = CORAL_BACKUP_LEAKY.exists()
    print(f"\n  Backup files present:")
    print(f"    coral_target_cov.at_224px_backup.pt  : {backup_224_exists}")
    print(f"    coral_target_cov.leaky_backup.pt     : {backup_leaky_exists}")

    return {
        "path": str(CORAL_PATH.relative_to(PROJECT_ROOT)),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": mean, "std": std, "frobenius": frob,
        "is_wrong_type": True,
        "wrong_type_reason": "CLS features, not ABMIL features — must recompute after Phase 1",
        "backup_224_exists": backup_224_exists,
        "backup_leaky_exists": backup_leaky_exists,
    }


# ---------------------------------------------------------------------------
# Check 5 — Split integrity verification
# ---------------------------------------------------------------------------
def check_5_split_integrity(split: dict) -> dict:
    _banner("CHECK 5 — Split integrity verification (13 checks)")
    meta = split["metadata"]
    train_set = set(split["train"])
    field_val_set = set(split["field_val"])
    probe_set = set(split["confusable_pair_probe"])
    final_val_set = set(split["final_val"])

    checks = {}
    checks["status_is_FINAL"] = meta.get("split_status") == "FINAL"

    # No overlap between buckets
    ov_tv = train_set & field_val_set
    ov_tp = train_set & probe_set
    ov_tf = train_set & final_val_set
    ov_vp = field_val_set & probe_set
    ov_vf = field_val_set & final_val_set
    ov_pf = probe_set & final_val_set
    checks["no_cross_split_overlap"] = all(
        len(o) == 0 for o in (ov_tv, ov_tp, ov_tf, ov_vp, ov_vf, ov_pf)
    )

    # Stopping weights sum to 1.0
    stopping_weights = meta.get("stopping_weights", {})
    sw_sum = sum(stopping_weights.values())
    checks["stopping_weights_sum_to_1"] = abs(sw_sum - 1.0) < 1e-4

    # Ratios sum to 1 — the split has 3 partition ratios {train, val, final}; the
    # `confusable_probe_frac_per_pair_class` is a DIFFERENT quantity (fraction of
    # the foliar/septoria real-field pool carved into the probe), NOT an additive
    # partition ratio. Sum only the 3 partition keys.
    ratios = meta.get("ratios", {})
    partition_keys = {"train_frac", "val_frac", "final_frac"}
    r_sum = sum(v for k, v in ratios.items() if k in partition_keys) if isinstance(ratios, dict) else 0.0
    checks["ratios_sum_to_1"] = abs(r_sum - 1.0) < 1e-4

    # Bucket sizes
    sizes = {
        "train": len(train_set),
        "field_val": len(field_val_set),
        "confusable_pair_probe": len(probe_set),
        "final_val": len(final_val_set),
    }

    # Per-class field_val counts & CI width
    per_class_ci = {}
    for cls in TOMATO_CLASSES:
        n_cls = sum(
            1 for p in split["field_val"]
            if f"/cleaned/{cls}/" in str(p).replace("\\", "/")
        )
        # Wilson-ish width = 2 * sqrt(0.25/N)
        ci = 2 * math.sqrt(0.25 / max(1, n_cls))
        severity = (
            "CRITICAL" if ci > 0.35 else
            "UNDERPOWERED" if ci > 0.25 else "OK"
        )
        per_class_ci[cls] = {"n": n_cls, "ci_width": round(ci, 4),
                             "severity": severity}

    print(f"  split_status     : {meta.get('split_status')}")
    print(f"  train / val / probe / final : "
          f"{sizes['train']} / {sizes['field_val']} / "
          f"{sizes['confusable_pair_probe']} / {sizes['final_val']}")
    print(f"  stopping_weights : {stopping_weights}  sum={sw_sum:.4f}")
    print(f"  ratios           : {ratios}  sum={r_sum:.4f}")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print(f"\n  Per-class field_val counts and CI widths:")
    for c, d in per_class_ci.items():
        print(f"    {c:35s} n={d['n']:4d}  CI={d['ci_width']:.3f}  [{d['severity']}]")

    all_pass = all(checks.values())
    return {
        "all_pass": all_pass,
        "checks": checks,
        "sizes": sizes,
        "stopping_weights": stopping_weights,
        "stopping_weights_sum": round(sw_sum, 6),
        "ratios": ratios,
        "per_class_ci": per_class_ci,
    }


# ---------------------------------------------------------------------------
# Check 6 — Register token count + 392px forward sanity
# ---------------------------------------------------------------------------
def check_6_register_tokens() -> dict:
    _banner("CHECK 6 — Register token count + 392px forward sanity")
    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2",
        pretrained=True, num_classes=0,
        img_size=224, dynamic_img_size=True,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    num_reg = getattr(model, "num_reg_tokens", None)
    num_prefix = 1 + (num_reg if num_reg is not None else 0)
    expected_patches_392 = (392 // 14) ** 2
    expected_seq_392 = num_prefix + expected_patches_392

    print(f"  model.num_reg_tokens : {num_reg}")
    print(f"  CLS + registers      : {num_prefix} prefix tokens")
    print(f"  Patches at 392px     : {expected_patches_392} (= 28x28)")
    print(f"  Total seq @ 392px    : {expected_seq_392}")
    print(f"  Correct attention indexing for spatial patches:")
    print(f"    attention[:, 0, {num_prefix}:]  # shape [batch, seq-prefix]")
    print(f"    reshape to [batch, 28, 28]")

    # Try a forward pass at 392px
    forward_ok = False
    out_shape = None
    with torch.no_grad():
        try:
            x = torch.randn(2, 3, 392, 392, device=device)
            out = model.forward_features(x)
            out_shape = tuple(out.shape)
            forward_ok = out_shape == (2, expected_seq_392, 768)
        except Exception as e:
            print(f"  [FAIL] Forward pass at 392px raised: {e}")
            out_shape = f"error: {e}"
    print(f"  Forward pass @ 392px : {'OK' if forward_ok else 'FAIL'}  out_shape={out_shape}")

    return {
        "num_reg_tokens": num_reg,
        "num_prefix_tokens": num_prefix,
        "expected_patches_392": expected_patches_392,
        "expected_seq_392": expected_seq_392,
        "correct_indexing_pattern": f"attention[:, 0, {num_prefix}:]",
        "forward_at_392_ok": forward_ok,
        "forward_out_shape": list(out_shape) if isinstance(out_shape, tuple) else str(out_shape),
    }


# ---------------------------------------------------------------------------
# Check 7 — Static recomposed many-to-one leaf check
# ---------------------------------------------------------------------------
def check_7_static_recomposed() -> dict:
    _banner("CHECK 7 — Static recomposed images many-to-one leaf duplication")
    # Scan the recomposed directory tree
    recomp_roots = [
        MODEL3_DIR / "recomposed",
        MODEL3_DIR / "cleaned",  # recomposed may be under cleaned/<class>/ with a pattern
    ]
    # Look in CSVs in the project for recomposed image records
    src_map = PROJECT_ROOT / "data" / "specialist" / "model3" / "model3_unified_source_map.csv"
    recomposed_rows = None
    if src_map.exists():
        df = pd.read_csv(src_map)
        if "source_dataset" in df.columns:
            recomposed_rows = df[df["source_dataset"].astype(str).str.contains(
                "recomp", case=False, na=False)]
        elif "source_bucket" in df.columns:
            recomposed_rows = df[df["source_bucket"].astype(str).str.contains(
                "recomp", case=False, na=False)]

    if recomposed_rows is None or len(recomposed_rows) == 0:
        print("  [INFO] No recomposed source map found — skipping per-source count analysis")
        # Fallback: count files in the recomposed dir
        count = 0
        for root in recomp_roots:
            if root.exists():
                for f in root.rglob("*.jpg"):
                    if "recomp" in f.name.lower():
                        count += 1
                for f in root.rglob("*.png"):
                    if "recomp" in f.name.lower():
                        count += 1
        print(f"  Disk count of 'recomp*' files: {count}")
        return {
            "source_map_found": False,
            "disk_count_recomp_files": count,
            "max_per_source": None,
            "duplication_risk": None,
        }

    # Parse the source_leaf_id. Typical naming: recomp_<sourcehash>_<bghash>.jpg
    # We'll group by a prefix before the last "_<8hex>.jpg" pattern.
    import re
    pat = re.compile(r"^(.*?)_[0-9a-f]{8}\.(jpg|png)$", re.IGNORECASE)
    source_counts: Counter = Counter()
    for p in recomposed_rows["image_path"]:
        name = Path(str(p)).name
        m = pat.match(name)
        key = m.group(1) if m else name
        source_counts[key] += 1

    if len(source_counts) == 0:
        return {
            "source_map_found": True,
            "max_per_source": 0,
            "duplication_risk": False,
        }
    max_per = max(source_counts.values())
    n_unique = len(source_counts)
    n_total = sum(source_counts.values())
    dup_risk = max_per > 3
    top5 = source_counts.most_common(5)

    print(f"  Total recomposed entries  : {n_total}")
    print(f"  Unique source-leaf IDs    : {n_unique}")
    print(f"  Max composites per source : {max_per}")
    print(f"  Top 5 (source_id, count)  : {top5}")
    if dup_risk:
        print(f"  [WARN] LEAF DUPLICATION RISK — some sources produce "
              f">3 composites, may cause memorization")
    else:
        print(f"  [OK] No excessive source duplication detected")

    return {
        "source_map_found": True,
        "total_recomposed_entries": int(n_total),
        "unique_source_ids": int(n_unique),
        "max_per_source": int(max_per),
        "duplication_risk": bool(dup_risk),
        "top5": [(k, int(v)) for k, v in top5],
    }


# ---------------------------------------------------------------------------
# Check 8 — DINOv2 block count + LoRA target block confirmation
# ---------------------------------------------------------------------------
def check_8_dinov2_blocks() -> dict:
    _banner("CHECK 8 — DINOv2-Base-Registers block count + LoRA target")
    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2",
        pretrained=True, num_classes=0,
        img_size=224, dynamic_img_size=True,
    ).to(device)
    blocks = list(model.blocks)
    n_blocks = len(blocks)
    top_8_indices = list(range(max(0, n_blocks - 8), n_blocks))
    frozen_indices = list(range(0, max(0, n_blocks - 8)))

    # Verify vram_test.py applies LoRA to the expected blocks
    vram_py = (PROJECT_ROOT / "scripts" / "ladi_net"
               / "vram_test.py").read_text(encoding="utf-8")
    expected_start = f"LORA_BLOCKS_FROM_TOP = 8"
    lora_const_found = expected_start in vram_py

    # Probe each block for qkv linear presence
    qkv_per_block = []
    for i, b in enumerate(blocks):
        has_qkv = hasattr(getattr(b, "attn", None), "qkv")
        qkv_per_block.append({"block": i, "has_qkv": bool(has_qkv)})

    print(f"  Total transformer blocks : {n_blocks}")
    print(f"  Target LoRA blocks (top 8) : {top_8_indices}")
    print(f"  Frozen-without-LoRA blocks : {frozen_indices}")
    print(f"  vram_test.py LORA_BLOCKS_FROM_TOP=8 : {lora_const_found}")
    print(f"  qkv presence per block   : "
          f"{'OK' if all(b['has_qkv'] for b in qkv_per_block) else 'FAIL'}")

    return {
        "total_blocks": n_blocks,
        "lora_target_blocks": top_8_indices,
        "frozen_blocks": frozen_indices,
        "vram_test_constant_correct": lora_const_found,
        "all_blocks_have_qkv": all(b["has_qkv"] for b in qkv_per_block),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device                 : {device}")
    if device.type == "cuda":
        print(f"GPU                    : {torch.cuda.get_device_name(device)}")

    # Load the mask log
    if not MASK_LOG.exists():
        print(f"[FAIL] mask log not found at {MASK_LOG}")
        sys.exit(1)
    df = pd.read_csv(MASK_LOG)

    # Load the split
    if not SPLIT_PATH.exists():
        print(f"[FAIL] split not found at {SPLIT_PATH}")
        sys.exit(1)
    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    # Optional: source map
    source_map_path = PROJECT_ROOT / "data" / "specialist" / "model3" / "model3_unified_source_map.csv"
    source_map_df = pd.read_csv(source_map_path) if source_map_path.exists() else None

    results = {"timestamp": datetime.datetime.now().isoformat(), "device": str(device)}
    try:
        results["recomposition_eligibility"] = check_1_recomposition_eligibility(df)
    except Exception as e:
        print(f"[CHECK 1 ERROR] {e}")
        results["recomposition_eligibility"] = {"error": str(e)}
    try:
        results["batch_simulation"] = check_2_batch_simulation(split, source_map_df)
    except Exception as e:
        print(f"[CHECK 2 ERROR] {e}")
        results["batch_simulation"] = {"error": str(e)}
    try:
        results["background_pool"] = check_3_background_pool()
    except Exception as e:
        print(f"[CHECK 3 ERROR] {e}")
        results["background_pool"] = {"error": str(e)}
    try:
        results["coral_target"] = check_4_coral_target()
    except Exception as e:
        print(f"[CHECK 4 ERROR] {e}")
        results["coral_target"] = {"error": str(e)}
    try:
        results["split_integrity"] = check_5_split_integrity(split)
    except Exception as e:
        print(f"[CHECK 5 ERROR] {e}")
        results["split_integrity"] = {"error": str(e)}
    try:
        results["register_tokens"] = check_6_register_tokens()
    except Exception as e:
        print(f"[CHECK 6 ERROR] {e}")
        results["register_tokens"] = {"error": str(e)}
    try:
        results["static_recomposed"] = check_7_static_recomposed()
    except Exception as e:
        print(f"[CHECK 7 ERROR] {e}")
        results["static_recomposed"] = {"error": str(e)}
    try:
        results["dinov2_blocks"] = check_8_dinov2_blocks()
    except Exception as e:
        print(f"[CHECK 8 ERROR] {e}")
        results["dinov2_blocks"] = {"error": str(e)}

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
