"""Phase 4B extraction: tomato class metadata + per-class metrics + confusion
matrix + calibration.

Class metadata comes from report_figures/fig_7_4_tomato_classes_v2_provenance.json
(authoritative, manually-verified pathogen taxonomy + example files).

Per-class metrics + confusion matrix + calibration: looks for the most recent
tomato evaluation report under reports/. If none exist, writes the JSON with
`metrics_pending: true` so the UI shows an "extraction pending" state.

No model inference here. Safe to re-run.
"""
from __future__ import annotations
import json
import os
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_qa_tmp" / "_pipeline_atlas_tomato.json"
PROVENANCE = ROOT / "report_figures" / "fig_7_4_tomato_classes_v2_provenance.json"

# Symptom one-liners per class. These are agronomically verified and short.
# (The provenance JSON has `diagnostic_hint` for each entry; we use that and
# expand with a leading-symptom phrase.)
SYMPTOM_NOTES = {
    "tomato_late_blight":
        "Water-soaked spots on leaves and stems; white fuzzy mycelium on the underside "
        "of leaves during humid mornings.",
    "tomato_septoria_leaf_spot":
        "Many small circular leaf spots with grey-tan centres and dark borders; lower "
        "leaves are hit first and the disease climbs upward.",
    "tomato_foliar_spot":
        "Composite class. The deployed pipeline groups four diseases under this label: "
        "bacterial spot (Xanthomonas), early blight (Alternaria), leaf mold (Passalora), "
        "and target spot (Corynespora). The common thread is discrete brown-to-grey leaf "
        "spots, often with concentric rings, that the model could not reliably separate "
        "at the fine-grained level on the available training data.",
    "tomato_mosaic_virus":
        "Mottled light and dark green patches on leaves, leaf curl, distortion of new "
        "growth; sometimes fruit ripens unevenly.",
    "tomato_yellow_leaf_curl_virus":
        "Upward leaf curl, yellow leaf margins, severe stunting of young shoots; the "
        "main giveaway is the cup-shaped curl pattern.",
    "tomato_healthy":
        "Uniform green colour, intact leaf shape, no spots or discolouration.",
}


def main():
    # PDA round 2 F5: degrade gracefully rather than SystemExit when the
    # provenance file is missing. Write a JSON with an empty class list and
    # a clear note so the page can render an honest empty state.
    if not PROVENANCE.exists():
        print(f"WARN: provenance file missing at {PROVENANCE}; writing empty JSON.")
        # PDA round 4 R4-1 + round 5 Finding 2: atomic write with Windows retry.
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".json.tmp")
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "produced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "source": "scripts/apin_v2/extract_pipeline_atlas_tomato.py",
                "classes": [],
                "metrics": {"extraction_pending": True,
                            "note": f"Provenance file missing at {PROVENANCE}. Re-stage report_figures/ and re-run this extractor."},
                "deployed_pipeline": {},
            }, f, indent=2)
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
        return  # Don't fail CI; degraded output is the contract.
    with open(PROVENANCE, encoding="utf-8") as f:
        prov = json.load(f)

    # PVA round 2 N3: tomato_foliar_spot is a COMPOSITE class in the deployed
    # pipeline (scripts/ladi_net/tomato_pipeline.py covers bacterial_spot +
    # early_blight + leaf_mold + target_spot under this label). The
    # provenance JSON's per-class "pathogen_latin" + "pathogen_type" is built
    # for the figures and treats foliar_spot as a single bacterial label,
    # which is misleading on this page. Override here so the Pipeline Atlas
    # reflects the actual deployed taxonomy.
    DEPLOYED_TAXONOMY_OVERRIDE = {
        "tomato_foliar_spot": {
            "pathogen_type": "MIXED",
            "pathogen_latin": ("Composite: Xanthomonas spp. (bacterial spot), "
                               "Alternaria solani (early blight), Passalora fulva "
                               "(leaf mold), Corynespora cassiicola (target spot)"),
        },
    }

    classes = []
    for sel in prov.get("selections", []):
        cls = sel.get("class_name", "")
        override = DEPLOYED_TAXONOMY_OVERRIDE.get(cls, {})
        classes.append({
            "class_name":     cls,
            "pathogen_latin": override.get("pathogen_latin",
                                          sel.get("pathogen_latin", "")),
            "pathogen_type":  override.get("pathogen_type",
                                          sel.get("pathogen_type", "")),
            "diagnostic_hint": sel.get("diagnostic_hint", ""),
            "symptoms":       SYMPTOM_NOTES.get(cls, ""),
            "n_training_total": sel.get("n_training_total", 0),
            "example_file_lab":   sel.get("lab_file", ""),
            "example_file_field": sel.get("field_file", ""),
        })

    # Sort classes for stable display: healthy last; viral/oomycete/fungal/bacterial grouping
    type_order = {"OOMYCETE": 0, "FUNGAL": 1, "BACTERIAL": 2, "VIRAL": 3, "HEALTHY": 9}
    classes.sort(key=lambda c: (type_order.get(c["pathogen_type"], 5), c["class_name"]))

    # Try to load existing tomato metrics. Sources we'd inspect:
    # 1. reports/_phase4_results.json (g42 has per-class entries)
    # 2. reports/_tomato_raw.json (if it exists)
    # We extract whatever we can; we never invent numbers.
    metrics_payload = {"extraction_pending": True, "note": (
        "Per-class precision/recall/F1/ECE and the 6x6 confusion matrix "
        "require running the deployed tomato pipeline (V3 + SP-LoRA) on the "
        "locked tomato test split. That is a separate extraction step "
        "(scripts/apin_v2/extract_tomato_metrics.py, Phase D) which loads "
        "the pipeline weights and writes _qa_tmp/_pipeline_atlas_tomato_metrics.json."
    )}

    phase4 = ROOT / "reports" / "_phase4_results.json"
    if phase4.exists():
        try:
            with open(phase4, encoding="utf-8") as f:
                d = json.load(f)
            g42 = d.get("g42", [])
            # g42 entries are PlantDoc-domain checks, NOT the production
            # tomato pipeline's locked test metrics. Document this carefully
            # rather than misrepresent the numbers as the pipeline's score.
            if g42:
                metrics_payload["plantdoc_domain_check"] = {
                    "note": (
                        "These per-class numbers come from a PlantDoc cross-"
                        "domain check, NOT from the production pipeline's locked "
                        "test split. They illustrate domain-shift behavior but "
                        "are not the pipeline's headline metrics."
                    ),
                    "rows": [
                        {
                            "class":   row.get("class", ""),
                            "f1_ct":   row.get("f1_ct", None),
                            "f1_pd":   row.get("f1_pd", None),
                            "p_ct":    row.get("p_ct", None),
                            "p_pd":    row.get("p_pd", None),
                            "r_ct":    row.get("r_ct", None),
                            "r_pd":    row.get("r_pd", None),
                            "sup_ct":  row.get("sup_ct", None),
                            "sup_pd":  row.get("sup_pd", None),
                            "gap":     row.get("gap", None),
                            "failure": row.get("failure", ""),
                        }
                        for row in g42
                    ],
                }
        except Exception as e:
            print(f"WARN: _phase4_results.json read failed: {e}")

    out = {
        "produced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "scripts/apin_v2/extract_pipeline_atlas_tomato.py",
        "classes": classes,
        "metrics": metrics_payload,
        "deployed_pipeline": {
            "v3_backbone": "DINOv2-Small + LoRA + FiLM, 10-class trained, used as 6-class",
            "lora_branch": "DINOv2-Reg-Base + LoRA on CLS, 6-class single-pass, epoch 13",
            "ensemble": "50/50 probability averaging with asymmetric sharpening (v3 T=0.5, LoRA T=1.0)",
            # PVA round 4 N1: split per-branch since V3 and LoRA take different
            # input sizes. Old single `input_size` field was ambiguous.
            "v3_input_size":   [224, 224],
            "lora_input_size": [392, 392],
            "calibration": (
                "Tier thresholds (phase3_tier_thresholds_*.json) were built from "
                "already-T-calibrated probabilities; T is applied ONCE per model."
            ),
        },
    }
    # PDA round 5 Finding 1 (CRITICAL): atomic write with Windows retry, same
    # pattern as the router and chilli extractors. The main success path was
    # left writing directly to OUT in round 4; same race the fix was designed
    # to prevent. Now consistent across all three extractors.
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
    print(f"  classes: {len(classes)}")
    for c in classes:
        print(f"    {c['class_name']:<32}  {c['pathogen_type']:<10}  n_train={c['n_training_total']}")
    print(f"  metrics: extraction_pending={metrics_payload.get('extraction_pending')}")
    if "plantdoc_domain_check" in metrics_payload:
        print(f"  plantdoc_domain_check rows: {len(metrics_payload['plantdoc_domain_check']['rows'])}")


if __name__ == "__main__":
    main()
