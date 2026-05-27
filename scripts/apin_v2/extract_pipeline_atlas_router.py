"""Phase 4B extraction: router data table (sources, splits) + licenses.

Counts router training-pool images per crop and per source. Reads
data/metadata/source_map.csv for split labels. No model inference.

The §5.4 metrics charts (per-crop accuracy, confidence histogram, confusion
matrix) require actual router inference on the locked test split. That is
flagged as `metrics_pending: true` in the output JSON; the UI will show an
"extraction pending" empty-state for those charts. A separate script
`extract_router_metrics.py` (Phase D scope) will fill them in.
"""
from __future__ import annotations
import json
import os
import csv
import datetime
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_qa_tmp" / "_pipeline_atlas_router.json"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*")
               if p.is_file() and p.suffix.lower() in IMG_EXTS)


# Source metadata for each source_dataset name observed in source_map.csv.
# License + acquisition dates are documented in the project's data lineage
# notes; they live here so the Pipeline Atlas Data section can cite each
# source. (Edit this map if a new source is added to source_map.csv.)
SOURCE_META = {
    # Okra sources
    "gadde_okra":               {"license": "CC-BY-4.0", "acquired": "2025-09",
                                 "citation": "Gadde okra dataset (Kaggle, 2023)"},
    "bangladesh_okra":          {"license": "CC-BY-4.0", "acquired": "2025-09",
                                 "citation": "Bangladesh okra disease dataset, Mendeley, 2023"},
    "yeesi":                    {"license": "CC0",       "acquired": "2025-10",
                                 "citation": "Yeesi okra healthy/diseased, Kaggle, 2022"},
    "mendeley_okra":            {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "Mendeley okra 4-class, 2022"},
    "inaturalist_okra":         {"license": "CC-BY-NC",  "acquired": "2025-11",
                                 "citation": "iNaturalist okra observations (research grade)"},
    # Brassica sources
    "cauliflower_noam":         {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "Cauliflower Noam dataset, Kaggle, 2024"},
    "cabbage_balanced":         {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "Cabbage balanced disease, Mendeley, 2023"},
    "diya_broccoli":            {"license": "research use","acquired": "2025-11",
                                 "citation": "Diya field-collected veg dataset, 2025"},
    "diya_cabbage":             {"license": "research use","acquired": "2025-11",
                                 "citation": "Diya field-collected veg dataset, 2025"},
    "diya_cauliflower":         {"license": "research use","acquired": "2025-11",
                                 "citation": "Diya field-collected veg dataset, 2025"},
    "mendeley_caul_leaf":       {"license": "CC-BY-4.0", "acquired": "2025-09",
                                 "citation": "Cauliflower leaf disease, Mendeley, 2022"},
    "mendeley_cabbage_dis":     {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "Cabbage disease combined, Mendeley, 2023"},
    "caul_sharifashik":         {"license": "CC-BY-4.0", "acquired": "2025-11",
                                 "citation": "Sharifashik cauliflower final dataset, 2024"},
    "inaturalist_brassica":     {"license": "CC-BY-NC",  "acquired": "2025-11",
                                 "citation": "iNaturalist brassica observations (research grade)"},
    # Tomato sources
    "plantvillage_tomato":      {"license": "CC0",       "acquired": "2025-09",
                                 "citation": "PlantVillage tomato (Penn State, 2016)"},
    "tomato_ashish":            {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "Ashish tomato disease dataset, Kaggle, 2023"},
    "plantdoc_train":           {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "PlantDoc field tomato (Singh et al., 2020)"},
    "plantdoc_eval":            {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "PlantDoc eval split (Singh et al., 2020)"},
    "tomato_luisolazo":         {"license": "CC-BY-4.0", "acquired": "2025-10",
                                 "citation": "Luis Olazo tomato dataset, Kaggle, 2023"},
    "tomato_kaustubh":          {"license": "CC-BY-4.0", "acquired": "2025-11",
                                 "citation": "Kaustubh tomato dataset, Kaggle, 2022"},
    "inaturalist_tomato":       {"license": "CC-BY-NC",  "acquired": "2025-11",
                                 "citation": "iNaturalist tomato observations (research grade)"},
    # Chilli sources
    "chilli_bangladesh":        {"license": "CC-BY-4.0", "acquired": "2025-11",
                                 "citation": "Bangladesh chilli disease, Mendeley, 2024"},
    "chilli_anthracnose_prudhvi":{"license": "CC-BY-4.0","acquired": "2025-11",
                                  "citation": "Prudhvi anthracnose chilli, Kaggle, 2023"},
    "chilli_cold_karnataka":    {"license": "research use","acquired": "2026-01",
                                 "citation": "Karnataka field chilli (cold collection), 2025"},
    "chilli_bangladesh_2025":   {"license": "CC-BY-4.0", "acquired": "2026-02",
                                 "citation": "Bangladesh chilli 2025 (extended), Mendeley, 2025"},
    "inaturalist_chilli":       {"license": "CC-BY-NC",  "acquired": "2025-11",
                                 "citation": "iNaturalist chilli observations (research grade)"},
}


def main():
    router_clean = ROOT / "data" / "specialist" / "router" / "cleaned"
    crops = ["tomato", "okra", "brassica", "chilli"]

    # Read source_map.csv to split per-crop counts into train/val/test.
    # Schema (observed): image_path, source_dataset, raw_label, class_name,
    #                    class_idx, crop_idx, split
    sm_path = ROOT / "data" / "metadata" / "source_map.csv"
    crop_splits = {c: {"train": 0, "val": 0, "test": 0} for c in crops}
    source_per_crop = {c: Counter() for c in crops}
    earliest_acq = {c: "" for c in crops}

    if sm_path.exists():
        try:
            with open(sm_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cls = (r.get("class_name") or "").lower()
                    spl = (r.get("split") or "").lower()
                    # PDA round 6 R6-2: strip so hand-edited CSV rows with
                    # trailing whitespace match the SOURCE_META keys.
                    src = (r.get("source_dataset") or "").strip() or "unknown"
                    crop = None
                    for c in crops:
                        if cls.startswith(c) or cls == c:
                            crop = c
                            break
                    if crop is None:
                        # Heuristic: source name hints crop
                        if any(t in src.lower() for t in ("tomato", "plantdoc")):
                            crop = "tomato"
                        elif "chilli" in src.lower():
                            crop = "chilli"
                        elif any(t in src.lower() for t in ("okra",)):
                            crop = "okra"
                        elif any(t in src.lower() for t in ("cauliflower","cabbage","broccoli","caul","brassica")):
                            crop = "brassica"
                    if crop and spl in crop_splits[crop]:
                        crop_splits[crop][spl] += 1
                    if crop:
                        source_per_crop[crop][src] += 1
        except Exception as e:
            print(f"WARN: source_map.csv read failed: {e}")

    # Per-crop data row: total counts come from the router cleaned/ folder
    # (the authoritative pool the router actually saw at train time). Split
    # counts come from source_map.csv. If the totals don't match, we prefer
    # the on-disk total and note the discrepancy.
    rows = []
    for crop in crops:
        on_disk = count_images(router_clean / crop)
        splits = crop_splits[crop]
        sm_total = splits["train"] + splits["val"] + splits["test"]
        # Per-crop earliest acquisition date: the earliest source we know about
        crop_sources = list(source_per_crop[crop].keys())
        acquired_dates = [SOURCE_META.get(s, {}).get("acquired", "") for s in crop_sources]
        acquired_dates = [d for d in acquired_dates if d]
        acq = min(acquired_dates) if acquired_dates else "n/a"
        rows.append({
            "source": crop,
            "n_train": splits["train"],
            "n_val":   splits["val"],
            "n_test":  splits["test"],
            "n_pool_on_disk": on_disk,
            "n_sm_total":     sm_total,
            "acquired": acq,
        })

    # License table: deduplicate sources actually used by the router
    all_sources = set()
    for c in crops:
        all_sources.update(source_per_crop[c].keys())
    licenses = []
    for src in sorted(all_sources):
        meta = SOURCE_META.get(src, {"license": "unknown",
                                     "acquired": "unknown",
                                     "citation": "(no citation on file)"})
        licenses.append({
            "source":   src,
            "license":  meta["license"],
            "acquired": meta["acquired"],
            "citation": meta["citation"],
        })

    out = {
        "produced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "scripts/apin_v2/extract_pipeline_atlas_router.py",
        "data": {
            "rows": rows,
            "licenses": licenses,
        },
        "metrics": {
            "extraction_pending": True,
            "note": (
                "Router inference metrics (per-crop accuracy, confidence "
                "histogram, confusion matrix) require running the router on "
                "the locked test split. That is a separate extraction script "
                "(scripts/apin_v2/extract_router_metrics.py) deferred to Phase D."
            ),
        },
    }
    # PDA round 4 R4-1: atomic write so the server cannot read a torn file.
    # Windows note: antivirus / shell-extension scanners can briefly hold the
    # .tmp file open; retry os.replace a few times before giving up.
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
        import shutil as _sh
        _sh.copyfile(tmp, OUT)
        try: tmp.unlink()
        except OSError: pass
    print(f"Wrote {OUT}")
    print("Per-crop pool on disk vs source_map sum:")
    for row in rows:
        print(f"  {row['source']:<10}  on-disk={row['n_pool_on_disk']:>6}  "
              f"sm_total={row['n_sm_total']:>6}  "
              f"train/val/test={row['n_train']}/{row['n_val']}/{row['n_test']}")
    print(f"Licenses recorded: {len(licenses)}")


if __name__ == "__main__":
    main()
