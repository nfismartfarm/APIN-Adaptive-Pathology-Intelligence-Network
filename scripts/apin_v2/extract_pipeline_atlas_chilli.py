"""Phase 4B extraction: chilli reach + sample gallery.

Counts chilli images in the router training pool (per split) and the staged
specialist data, then writes _qa_tmp/_pipeline_atlas_chilli.json.

Reads only from disk; no model inference. Safe to run multiple times.
"""
from __future__ import annotations
import json
import os
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_qa_tmp" / "_pipeline_atlas_chilli.json"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            n += 1
    return n


def sample_filenames(root: Path, k: int = 6) -> list[str]:
    """Pick up to k representative filenames (deterministic via sort)."""
    if not root.exists():
        return []
    all_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not all_files:
        return []
    if len(all_files) <= k:
        return [str(p.relative_to(ROOT)).replace("\\", "/") for p in all_files]
    # Evenly spaced selection across the sorted list
    step = len(all_files) // k
    picks = [all_files[i * step] for i in range(k)]
    return [str(p.relative_to(ROOT)).replace("\\", "/") for p in picks]


def main():
    router_root = ROOT / "data" / "specialist" / "router" / "cleaned" / "chilli"
    specialist_root = ROOT / "data" / "specialist" / "model3" / "cleaned"

    # Specialist-staged classes (counted per disease bucket, raw on disk)
    specialist_classes = {}
    if specialist_root.exists():
        for sub in sorted(specialist_root.iterdir()):
            if sub.is_dir() and sub.name.startswith("chilli_"):
                specialist_classes[sub.name] = count_images(sub)

    # The Pipeline Atlas Module 3 §5.8c table is about the ROUTER training pool:
    # the chilli images the router has seen (a single label "chilli", no disease
    # subclass; the disease specialist does not exist yet).
    n_router = count_images(router_root)

    # Split breakdown: the router's split metadata lives in source_map.csv
    # Try to read it, but degrade gracefully.
    rows = []
    sm_path = ROOT / "data" / "metadata" / "source_map.csv"
    split_counts = {"train": 0, "val": 0, "test": 0}
    if sm_path.exists():
        try:
            import csv
            with open(sm_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("class_name", "").lower() in ("chilli", "router_chilli") \
                       or "chilli" in r.get("class_name", "").lower():
                        sp = r.get("split", "").lower()
                        if sp in split_counts:
                            split_counts[sp] += 1
        except Exception:
            pass

    # If the source_map split rows came up empty (different schema), fall back
    # to the on-disk router/cleaned/chilli count under a single "pool" row.
    if sum(split_counts.values()) == 0 and n_router > 0:
        rows = [
            {"split": "router_pool", "n_images": n_router,
             "classes": "chilli (single label; not split for router training only)"},
        ]
    else:
        rows = [
            {"split": "router_train", "n_images": split_counts["train"],
             "classes": "chilli (single label)"},
            {"split": "router_val",   "n_images": split_counts["val"],
             "classes": "chilli"},
            {"split": "router_test",  "n_images": split_counts["test"],
             "classes": "chilli"},
        ]

    out = {
        "produced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "scripts/apin_v2/extract_pipeline_atlas_chilli.py",
        "reach": {
            "rows": rows,
            "router_pool_on_disk": n_router,
            "sum_of_split_rows": sum(r["n_images"] for r in rows),
        },
        "specialist_staged": {
            "note": (
                "These class folders are staged on disk for a future chilli "
                "specialist. They are NOT used by any currently-deployed model. "
                "Counted here for transparency about what data exists."
            ),
            "classes": specialist_classes,
        },
        "sample_gallery": sample_filenames(router_root, k=6),
    }
    # PDA round 4 R4-1: atomic write. Open a tempfile alongside the target,
    # write the full JSON, then os.replace() it. Eliminates the torn-read
    # window where the server could see a partial file mid-write. On Windows,
    # antivirus or shell-extension scanners can briefly hold a handle to the
    # .tmp file, so we retry a few times with a short sleep.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    if tmp.exists():
        try: tmp.unlink()
        except OSError: pass
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    import time as _t
    for _attempt in range(5):
        try:
            os.replace(tmp, OUT)
            break
        except PermissionError:
            _t.sleep(0.15)
    else:
        # Fallback: copy-then-unlink if replace stays blocked.
        import shutil as _sh
        _sh.copyfile(tmp, OUT)
        try: tmp.unlink()
        except OSError: pass
    print(f"Wrote {OUT}")
    print(f"  router chilli pool: {n_router}")
    print(f"  specialist staged classes: {len(specialist_classes)}")
    print(f"  sample gallery: {len(out['sample_gallery'])} files")


if __name__ == "__main__":
    main()
