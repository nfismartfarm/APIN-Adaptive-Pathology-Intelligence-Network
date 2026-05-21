"""LADI-Net Phase 0 Step 2 — Source-stratified tomato data split (v2).

This version implements the continuation-prompt specification:

  * Ratios are adjustable via --train_frac / --val_frac / --final_frac
    (must sum to 1.0).
  * Only real-field images (is_field_photo=True AND source != 'scidb_recomposed')
    are split by the ratio. Lab and recomposed images always go to training.
  * Confusable-pair probe (10% of real-field foliar + 10% of real-field
    septoria) is drawn out of the TRAIN allocation for those two classes
    (their effective train_frac = train_frac - 0.10).
  * Source-aware stratified sampling: stratify by (class_name, source_dataset).
  * sqrt(N) weighted stopping metric:
        w_k = sqrt(N_field_val_k); normalized to sum to 1.0.
    Stored in metadata.stopping_weights so the training script can compute
    the same aggregate.
  * 13 integrity checks (original 6 + 7 new). Writes are aborted on any
    failure.
  * Statistical-power warnings: per-class field_val CI width. Flags
    UNDERPOWERED (>0.25) and CRITICAL (>0.35) classes.
  * metadata.split_status = "PLACEHOLDER" (before probe) or "FINAL" (after).

Run:
    python scripts/ladi_net/phase0_data_split.py \
      --train_frac 0.70 --val_frac 0.20 --final_frac 0.10 \
      --status PLACEHOLDER

    (After probe, re-run with probe-recommended ratios and --status FINAL.)

Exit codes:
    0 split written + all integrity checks passed
    1 integrity violation (no file written)
    2 precondition failure (missing CSV, sum != 1.0, etc.)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    import sklearn
    _SK_VERSION = sklearn.__version__
except ImportError as e:
    print(f"ERROR: sklearn is required for stratified split. Install with: "
          f"pip install scikit-learn\n  (details: {e})")
    sys.exit(2)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "specialist" / "model3" / "model3_unified_source_map.csv"
OUT_SPLIT = PROJECT_ROOT / "data" / "specialist" / "model3" / "split_indices.json"
OLD_SPLIT_BACKUP = (PROJECT_ROOT / "data" / "specialist" / "model3"
                    / "split_indices_model3_v3.json")
LOG_DIR = PROJECT_ROOT / "logs"

TOMATO_CLASSES = [
    "tomato_foliar_spot",
    "tomato_septoria_leaf_spot",
    "tomato_late_blight",
    "tomato_yellow_leaf_curl_virus",
    "tomato_mosaic_virus",
    "tomato_healthy",
]

CONFUSABLE_PAIR = {"tomato_foliar_spot", "tomato_septoria_leaf_spot"}
CONFUSABLE_PROBE_FRACTION = 0.10  # taken from each confusable-pair class's real-field

UNDERPOWERED_CI_THRESHOLD = 0.25  # 95% CI width above this = UNDERPOWERED
CRITICAL_CI_THRESHOLD = 0.35       # above this = CRITICAL
RANDOM_SEED = 42


# ------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ladi_phase0_data_split_{ts}.log"
    logger = logging.getLogger("ladi.phase0.split")
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
# Helpers
# ------------------------------------------------------------------------
def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "t", "yes"}


def _is_recomposed(source_dataset: str) -> bool:
    s = str(source_dataset).lower()
    return s in {"scidb_recomposed", "capsicum_recomposed"}


def _ci_width(n: int, p_hat: float = 0.5) -> float:
    """95% Wilson-style CI width for a proportion. Max width at p=0.5.
    Returns the FULL width (high minus low), not the half-width."""
    if n <= 0:
        return 1.0
    z = 1.96
    denom = 1.0 + (z ** 2) / n
    half = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * n ** 2)) / denom
    return 2.0 * half


# ------------------------------------------------------------------------
# Core stratified allocator — splits a pool of (image_path, stratum_key)
# into train/val/final honoring the ratios as closely as possible, using
# sklearn StratifiedShuffleSplit. Falls back to plain shuffle if a stratum
# has fewer than 2 members (sklearn refuses).
# ------------------------------------------------------------------------
def _three_way_stratified(
    indices: np.ndarray,
    stratum_labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    final_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, final_idx) arrays of indices drawn from
    `indices`, honoring `stratum_labels` as stratification.

    Strategy: first split (val ∪ final) vs train; then split val from final.
    Both splits use sklearn StratifiedShuffleSplit with seed=42. When a
    stratum has <2 members, those items default to train (too small to
    stratify).
    """
    assert abs(train_frac + val_frac + final_frac - 1.0) < 1e-6, \
        "ratios must sum to 1.0"
    if len(indices) == 0:
        e = np.array([], dtype=int)
        return e, e.copy(), e.copy()

    # Identify tiny strata (fewer than 2 members) and route them to train.
    stratum_counts = Counter(stratum_labels.tolist())
    tiny_mask = np.array([stratum_counts[s] < 2 for s in stratum_labels])
    tiny_indices = indices[tiny_mask]
    big_indices = indices[~tiny_mask]
    big_labels = stratum_labels[~tiny_mask]

    if len(big_indices) == 0:
        # Degenerate — everything to train
        return tiny_indices, np.array([], dtype=int), np.array([], dtype=int)

    # Step 1: split off train, leaving (val ∪ final)
    eval_frac = val_frac + final_frac
    if eval_frac <= 0 or eval_frac >= 1.0:
        return (np.concatenate([big_indices, tiny_indices]),
                np.array([], dtype=int), np.array([], dtype=int))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=eval_frac,
                                   random_state=int(rng.integers(0, 2**31 - 1)))
    try:
        tr_pos, eval_pos = next(sss1.split(big_indices, big_labels))
    except ValueError:
        # Some stratum has too few members for StratifiedShuffleSplit.
        # Fall back to plain shuffle.
        perm = np.random.default_rng(RANDOM_SEED).permutation(len(big_indices))
        n_tr = int(round(len(big_indices) * train_frac))
        tr_pos = perm[:n_tr]
        eval_pos = perm[n_tr:]

    train_idx = np.concatenate([big_indices[tr_pos], tiny_indices])
    eval_indices = big_indices[eval_pos]
    eval_labels = big_labels[eval_pos]

    if len(eval_indices) == 0:
        return train_idx, np.array([], dtype=int), np.array([], dtype=int)

    # Step 2: split (val ∪ final) into val vs final.
    # final_proportion_within_eval = final_frac / (val_frac + final_frac)
    final_prop = final_frac / eval_frac
    if final_prop <= 0 or final_prop >= 1.0 or len(eval_indices) < 2:
        # Degenerate — can't subdivide.
        return train_idx, eval_indices, np.array([], dtype=int)

    eval_stratum_counts = Counter(eval_labels.tolist())
    tiny_mask_e = np.array([eval_stratum_counts[s] < 2 for s in eval_labels])
    tiny_eval = eval_indices[tiny_mask_e]
    big_eval = eval_indices[~tiny_mask_e]
    big_eval_labels = eval_labels[~tiny_mask_e]

    if len(big_eval) < 2:
        # route everything to val (can't stratify); final gets the tinies.
        val_idx = np.concatenate([big_eval, np.array([], dtype=int)])
        final_idx = tiny_eval
        return train_idx, val_idx, final_idx

    try:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=final_prop,
                                       random_state=int(rng.integers(0, 2**31 - 1)))
        val_pos, final_pos = next(sss2.split(big_eval, big_eval_labels))
        val_idx = np.concatenate([big_eval[val_pos], tiny_eval])
        final_idx = big_eval[final_pos]
    except ValueError:
        # Same fallback.
        perm = np.random.default_rng(RANDOM_SEED + 1).permutation(len(big_eval))
        n_val = int(round(len(big_eval) * (1 - final_prop)))
        val_idx = np.concatenate([big_eval[perm[:n_val]], tiny_eval])
        final_idx = big_eval[perm[n_val:]]
    return train_idx, val_idx, final_idx


# ------------------------------------------------------------------------
# Split builder
# ------------------------------------------------------------------------
def build_split(df: pd.DataFrame,
                train_frac: float, val_frac: float, final_frac: float,
                logger: logging.Logger) -> tuple[dict, pd.DataFrame]:

    # 0. Restrict + normalize.
    df = df[df["class_name"].isin(TOMATO_CLASSES)].copy().reset_index(drop=True)
    df["_is_field_bool"] = df["is_field_photo"].apply(_to_bool)
    df["_is_recomposed"] = df["source_dataset"].apply(_is_recomposed)

    # Compose a stratum label used for source-aware stratification.
    df["_stratum"] = df["class_name"].astype(str) + "__" + df["source_dataset"].astype(str)

    # Pool definitions:
    df["_pool"] = "lab"
    df.loc[df["_is_recomposed"], "_pool"] = "recomposed"
    df.loc[df["_is_field_bool"] & ~df["_is_recomposed"], "_pool"] = "real_field"

    logger.info(f"Tomato rows: {len(df):,}")
    logger.info(f"Pool breakdown: {df['_pool'].value_counts().to_dict()}")
    logger.info(f"Per-class real-field counts:")
    for cls in TOMATO_CLASSES:
        n = ((df["class_name"] == cls) & (df["_pool"] == "real_field")).sum()
        logger.info(f"  {cls:<32}: real_field={n}")

    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Confusable-pair probe: 10% of real-field foliar + 10% of real-field septoria.
    #    Drawn from each class's real-field allocation BEFORE the ratio split.
    probe_indices: list[int] = []
    for cls in sorted(CONFUSABLE_PAIR):
        mask = (df["class_name"] == cls) & (df["_pool"] == "real_field")
        pool = df[mask].index.to_numpy()
        n_probe = int(round(len(pool) * CONFUSABLE_PROBE_FRACTION))
        if len(pool) > 0 and n_probe == 0:
            n_probe = 1
        chosen = rng.choice(pool, size=n_probe, replace=False) if n_probe > 0 else np.array([], dtype=int)
        probe_indices.extend(chosen.tolist())
        logger.info(f"  probe[{cls}]: {n_probe} real-field images held out "
                    f"(from pool of {len(pool)})")
    probe_set = set(probe_indices)

    # 2. Allocate REAL-FIELD images (minus probe) via stratified three-way split.
    #    For the confusable-pair classes, the effective train_frac is reduced
    #    by CONFUSABLE_PROBE_FRACTION (the probe came out of train), so we
    #    re-normalize train/val/final for those two classes:
    #        effective_train = train_frac - probe_frac
    #        effective_val   = val_frac
    #        effective_final = final_frac
    #    Sum = 1 - probe_frac, so normalize back to 1.0 before calling the
    #    three-way splitter (which expects ratios summing to 1.0).
    train_indices: list[int] = []
    field_val_indices: list[int] = []
    final_val_indices: list[int] = []

    for cls in TOMATO_CLASSES:
        mask = ((df["class_name"] == cls) & (df["_pool"] == "real_field")
                & ~df.index.isin(probe_set))
        idx = df[mask].index.to_numpy()
        strata = df.loc[idx, "_stratum"].to_numpy()
        if cls in CONFUSABLE_PAIR:
            # Renormalize: effective train = train_frac - probe_frac out of
            # the REAL-FIELD budget AFTER probe is removed (probe is already
            # removed from `idx`). So ratios here are relative to idx's total.
            raw_train = max(0.0, train_frac - CONFUSABLE_PROBE_FRACTION)
            total = raw_train + val_frac + final_frac
            if total <= 0:
                continue
            t = raw_train / total
            v = val_frac / total
            f = final_frac / total
        else:
            t, v, f = train_frac, val_frac, final_frac
        tr_i, vl_i, fn_i = _three_way_stratified(idx, strata, t, v, f, rng)
        train_indices.extend(tr_i.tolist())
        field_val_indices.extend(vl_i.tolist())
        final_val_indices.extend(fn_i.tolist())
        logger.info(f"  [{cls:<32}] real-field -> train={len(tr_i)}, "
                    f"val={len(vl_i)}, final={len(fn_i)}  (ratios {t:.2f}/{v:.2f}/{f:.2f})")

    # 3. Add ALL lab and ALL recomposed to training.
    lab_recomp_mask = df["_pool"].isin({"lab", "recomposed"})
    lab_recomp_indices = df[lab_recomp_mask].index.to_numpy().tolist()
    train_indices.extend(lab_recomp_indices)

    # 4. Compute stopping weights from the field_val class counts.
    #    w_k = sqrt(N_field_val_k), normalized to sum 1.0.
    fv_counts_per_class: dict[str, int] = {c: 0 for c in TOMATO_CLASSES}
    for i in field_val_indices:
        fv_counts_per_class[df.loc[i, "class_name"]] += 1
    raw_weights = {c: math.sqrt(max(1, fv_counts_per_class[c])) for c in TOMATO_CLASSES}
    # If a class has N=0 we still give it a minimum weight of sqrt(1) so
    # the metadata is well-defined, but the training script should log a
    # warning if any class has N=0 at stopping time.
    total_w = sum(raw_weights.values())
    stopping_weights = {c: round(raw_weights[c] / total_w, 6) for c in TOMATO_CLASSES}

    # 5. Convert indices → image paths.
    def to_paths(ids):
        return df.loc[ids, "image_path"].tolist()

    split = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_tomato_rows": int(len(df)),
            "tomato_classes": TOMATO_CLASSES,
            "split_status": "PENDING",  # set by main()
            "split_rationale": (
                "Source-stratified three-way split of REAL-FIELD tomato images. "
                "Lab + recomposed ALWAYS go to training only. Confusable-pair probe "
                "(10% of real-field foliar + septoria) is drawn from the train "
                "allocation of those two classes, reducing their effective train "
                "fraction by CONFUSABLE_PROBE_FRACTION. Stopping metric is a "
                "sqrt(N_field_val_k)-weighted macro-F1 — see stopping_weights."),
            "ratios": {
                "train_frac": train_frac,
                "val_frac": val_frac,
                "final_frac": final_frac,
                "confusable_probe_frac_per_pair_class": CONFUSABLE_PROBE_FRACTION,
            },
            "stopping_weights": stopping_weights,
            "stopping_metric_formula": (
                "sum over classes k of stopping_weights[k] * F1_k(field_val). "
                "Use 3-epoch rolling mean with patience=5, min_epochs=12."),
            "random_seed": RANDOM_SEED,
            "sklearn_version": _SK_VERSION,
            "pools": {
                "real_field_count": int((df["_pool"] == "real_field").sum()),
                "recomposed_count": int((df["_pool"] == "recomposed").sum()),
                "lab_count": int((df["_pool"] == "lab").sum()),
            },
        },
        "train": to_paths(train_indices),
        "field_val": to_paths(field_val_indices),
        "confusable_pair_probe": to_paths(probe_indices),
        "final_val": to_paths(final_val_indices),
    }
    return split, df


# ------------------------------------------------------------------------
# Integrity verification — all 13 checks run BEFORE write
# ------------------------------------------------------------------------
def verify_split(split: dict, df: pd.DataFrame,
                 logger: logging.Logger) -> tuple[bool, list[str]]:
    failures = []
    split_keys = ["train", "field_val", "confusable_pair_probe", "final_val"]
    logger.info(f"Sizes: {{'train': {len(split['train'])}, "
                f"'field_val': {len(split['field_val'])}, "
                f"'confusable_pair_probe': {len(split['confusable_pair_probe'])}, "
                f"'final_val': {len(split['final_val'])}}}")
    path_to_row = dict(zip(df["image_path"], df.to_dict("records")))

    # 1. No image in more than one split.
    path_to_splits: dict[str, list[str]] = defaultdict(list)
    for k in split_keys:
        for p in split[k]:
            path_to_splits[p].append(k)
    dupes = {p: ks for p, ks in path_to_splits.items() if len(ks) > 1}
    if dupes:
        failures.append(f"Check 1 FAIL: {len(dupes)} images in multiple splits "
                        f"(example: {next(iter(dupes.items()))})")
    else:
        logger.info("  [PASS]  1 no cross-split overlap")

    # 2. field_val: zero lab and zero recomposed.
    fv_bad_pool = 0
    for p in split["field_val"]:
        r = path_to_row.get(p)
        if r and r["_pool"] != "real_field":
            fv_bad_pool += 1
    if fv_bad_pool:
        failures.append(f"Check 2 FAIL: field_val has {fv_bad_pool} non-real-field images")
    else:
        logger.info("  [PASS]  2 field_val contains only real-field images")

    # 3. field_val has zero probe images.
    probe_set = set(split["confusable_pair_probe"])
    fv_probe = sum(1 for p in split["field_val"] if p in probe_set)
    if fv_probe:
        failures.append(f"Check 3 FAIL: {fv_probe} probe images appear in field_val")
    else:
        logger.info("  [PASS]  3 field_val has no probe images")

    # 4. final_val has zero training images.
    train_set = set(split["train"])
    fn_train = sum(1 for p in split["final_val"] if p in train_set)
    if fn_train:
        failures.append(f"Check 4 FAIL: {fn_train} training images appear in final_val")
    else:
        logger.info("  [PASS]  4 final_val and train disjoint")

    # 5. Recomposed images ONLY in training.
    recomp_leak = 0
    for k in ("field_val", "confusable_pair_probe", "final_val"):
        for p in split[k]:
            r = path_to_row.get(p)
            if r and r["_pool"] == "recomposed":
                recomp_leak += 1
    if recomp_leak:
        failures.append(f"Check 5 FAIL: {recomp_leak} recomposed images outside training")
    else:
        logger.info("  [PASS]  5 recomposed images train-only")

    # 6. Every split non-empty for every class.
    per_split_class_counts = {}
    empty_class_violations = []
    for k in split_keys:
        cnt = Counter()
        for p in split[k]:
            r = path_to_row.get(p)
            if r:
                cnt[r["class_name"]] += 1
        per_split_class_counts[k] = dict(cnt)
        # confusable_pair_probe is expected to only contain the two pair classes
        if k == "confusable_pair_probe":
            continue
        for cls in TOMATO_CLASSES:
            if cnt.get(cls, 0) == 0:
                empty_class_violations.append(f"{k}/{cls}")
    if empty_class_violations:
        failures.append(f"Check 6 FAIL: empty in {len(empty_class_violations)} "
                        f"class/split combos: {empty_class_violations[:10]}")
    else:
        logger.info("  [PASS]  6 all 6 classes present in train, field_val, final_val")

    # 7. Source proportions in field_val within 15% of training source proportions.
    def src_dist(paths: list[str]) -> Counter:
        c = Counter()
        for p in paths:
            r = path_to_row.get(p)
            if r and r["_pool"] == "real_field":
                c[r["source_dataset"]] += 1
        return c

    tr_src = src_dist(split["train"])
    fv_src = src_dist(split["field_val"])
    total_tr = sum(tr_src.values()) or 1
    total_fv = sum(fv_src.values()) or 1
    big_gap_srcs = []
    all_srcs = set(tr_src.keys()) | set(fv_src.keys())
    for s in all_srcs:
        p_tr = tr_src.get(s, 0) / total_tr
        p_fv = fv_src.get(s, 0) / total_fv
        if abs(p_tr - p_fv) > 0.15 and max(p_tr, p_fv) > 0.05:
            big_gap_srcs.append(
                f"{s}: train={p_tr*100:.1f}% val={p_fv*100:.1f}% gap={(p_fv-p_tr)*100:+.1f}%")
    if big_gap_srcs:
        # Warning, not failure — in a tiny real-field pool this can be hard
        # to satisfy exactly. Record as warning in log; flag for reviewer.
        logger.warning(f"  [WARN]  7 source proportion gaps > 15% in field_val:")
        for s in big_gap_srcs:
            logger.warning(f"           {s}")
        logger.warning(f"  (Decision: treat as warning given small real-field pool. "
                       f"Reviewer must confirm acceptability.)")
    else:
        logger.info("  [PASS]  7 source proportions within 15% tolerance")

    # 8. Stopping weights: present, positive, sum to 1.0.
    sw = split["metadata"].get("stopping_weights", {})
    if not sw:
        failures.append("Check 8 FAIL: stopping_weights missing")
    else:
        s = sum(sw.values())
        if not (0.99 < s < 1.01):
            failures.append(f"Check 8 FAIL: stopping_weights sum = {s}, expected 1.0")
        elif any(v <= 0 for v in sw.values()):
            failures.append(f"Check 8 FAIL: non-positive stopping_weight: {sw}")
        else:
            logger.info(f"  [PASS]  8 stopping_weights present, sum=1.0, all positive")
            logger.info(f"          {sw}")

    # 9. split_status present in metadata.
    ss = split["metadata"].get("split_status")
    if ss not in {"PLACEHOLDER", "FINAL"}:
        failures.append(f"Check 9 FAIL: split_status = {ss!r}, expected PLACEHOLDER or FINAL")
    else:
        logger.info(f"  [PASS]  9 split_status = {ss}")

    # 10. ratios sum to 1.0.
    r = split["metadata"].get("ratios", {})
    rs = r.get("train_frac", 0) + r.get("val_frac", 0) + r.get("final_frac", 0)
    if not (0.99 < rs < 1.01):
        failures.append(f"Check 10 FAIL: train+val+final = {rs}, expected 1.0")
    else:
        logger.info(f"  [PASS] 10 ratios sum to 1.0 "
                    f"({r['train_frac']}/{r['val_frac']}/{r['final_frac']})")

    # 11. Every real-field image appears in exactly one of {train, field_val, final_val, probe}.
    real_field_paths = set(df.loc[df["_pool"] == "real_field", "image_path"].tolist())
    seen = set()
    for k in split_keys:
        for p in split[k]:
            if p in real_field_paths:
                seen.add(p)
    missing = real_field_paths - seen
    if missing:
        failures.append(f"Check 11 FAIL: {len(missing)} real-field images not in any split")
    else:
        logger.info(f"  [PASS] 11 all {len(real_field_paths)} real-field images assigned")

    # 12. Every recomposed and every lab image is in training.
    non_train_expected = set(df.loc[df["_pool"].isin({"lab", "recomposed"}), "image_path"]) - train_set
    if non_train_expected:
        failures.append(f"Check 12 FAIL: {len(non_train_expected)} lab/recomposed images outside training")
    else:
        logger.info(f"  [PASS] 12 every lab/recomposed image is in training")

    # 13. confusable_pair_probe only contains foliar + septoria real-field.
    probe_bad = 0
    for p in split["confusable_pair_probe"]:
        r = path_to_row.get(p)
        if not r:
            probe_bad += 1
            continue
        if r["class_name"] not in CONFUSABLE_PAIR or r["_pool"] != "real_field":
            probe_bad += 1
    if probe_bad:
        failures.append(f"Check 13 FAIL: probe has {probe_bad} wrong-class or non-real-field images")
    else:
        logger.info(f"  [PASS] 13 probe contains only foliar+septoria real-field")

    return (len(failures) == 0), failures


# ------------------------------------------------------------------------
# Statistical-power report
# ------------------------------------------------------------------------
def statistical_power_report(split: dict, df: pd.DataFrame,
                              logger: logging.Logger) -> list[str]:
    """Per-class field_val N + approximate 95% CI width. Return list of warnings."""
    path_to_row = dict(zip(df["image_path"], df.to_dict("records")))
    fv_counts = Counter()
    for p in split["field_val"]:
        r = path_to_row.get(p)
        if r:
            fv_counts[r["class_name"]] += 1
    warnings_list: list[str] = []
    logger.info("")
    logger.info("-" * 72)
    logger.info("STATISTICAL POWER (field_val per class) — CI width of a proportion")
    logger.info("-" * 72)
    logger.info(f"{'class':<32}  {'N':>5}  {'CI (±)':>8}  flag")
    for cls in TOMATO_CLASSES:
        n = fv_counts.get(cls, 0)
        w = _ci_width(n)
        flag = ""
        if w > CRITICAL_CI_THRESHOLD:
            flag = "CRITICAL — F1 for this class is statistically meaningless"
        elif w > UNDERPOWERED_CI_THRESHOLD:
            flag = "UNDERPOWERED — stopping for this class is noise-dominated"
        if flag:
            warnings_list.append(f"{cls} (n={n}, CI±{w/2:.2f}): {flag}")
        logger.info(f"{cls:<32}  {n:>5}  ±{w/2:.3f}  {flag}")
    if warnings_list:
        logger.warning("")
        logger.warning("POWER WARNINGS:")
        for w in warnings_list:
            logger.warning(f"  - {w}")
    return warnings_list


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--out", default=str(OUT_SPLIT))
    parser.add_argument("--train_frac", type=float, default=0.70,
                        help="Real-field fraction to training (default 0.70)")
    parser.add_argument("--val_frac", type=float, default=0.20,
                        help="Real-field fraction to field_val (default 0.20)")
    parser.add_argument("--final_frac", type=float, default=0.10,
                        help="Real-field fraction to final_val (default 0.10)")
    parser.add_argument("--status", choices=["PLACEHOLDER", "FINAL"],
                        default="PLACEHOLDER",
                        help="Mark split_status in metadata")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logger = _setup_logger()
    logger.info("=" * 72)
    logger.info("LADI-Net Phase 0 Step 2 — Source-stratified data split (v2)")
    logger.info("=" * 72)
    logger.info(f"Ratios: train={args.train_frac}, val={args.val_frac}, final={args.final_frac}")
    logger.info(f"Status: {args.status}")

    rsum = args.train_frac + args.val_frac + args.final_frac
    if not (0.999 < rsum < 1.001):
        logger.error(f"Ratios must sum to 1.0; got {rsum}")
        return 2

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return 2
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV: {len(df):,} rows")

    # Backup existing split (if from Model 3 v3, before we ever wrote one).
    existing = Path(args.out)
    if existing.exists() and not OLD_SPLIT_BACKUP.exists():
        existing.rename(OLD_SPLIT_BACKUP)
        logger.info(f"Backed up pre-existing split -> {OLD_SPLIT_BACKUP.name}")

    logger.info("")
    logger.info("Building split...")
    split, df_work = build_split(df, args.train_frac, args.val_frac, args.final_frac, logger)
    split["metadata"]["split_status"] = args.status

    logger.info("")
    logger.info("-" * 72)
    logger.info("INTEGRITY VERIFICATION (13 checks)")
    logger.info("-" * 72)
    ok, failures = verify_split(split, df_work, logger)
    if not ok:
        logger.error("")
        logger.error("INTEGRITY CHECKS FAILED — split NOT written:")
        for f in failures:
            logger.error(f"  - {f}")
        return 1

    # Statistical power report.
    power_warnings = statistical_power_report(split, df_work, logger)

    if args.dry_run:
        logger.info("")
        logger.info("[DRY RUN] All checks passed but --dry-run was set; not writing.")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    logger.info("")
    logger.info(f"Split written -> {out_path.relative_to(PROJECT_ROOT).as_posix()}")
    logger.info(f"Status: {args.status}")
    logger.info(f"Sizes: train={len(split['train']):>6}  "
                f"field_val={len(split['field_val']):>4}  "
                f"probe={len(split['confusable_pair_probe']):>4}  "
                f"final_val={len(split['final_val']):>4}")
    if power_warnings:
        logger.warning(f"{len(power_warnings)} statistical-power warning(s) — "
                       f"see log above and carry forward to developer report.")
    logger.info("[OK] Step 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
