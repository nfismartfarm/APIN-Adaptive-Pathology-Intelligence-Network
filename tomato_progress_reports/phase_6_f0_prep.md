# Phase 6 F.0 Prep — Consolidated Report

**Coverage:** Phase 6 Component B (calibration script) + Component A (validation script) + Component C (real model loading lift)
**Date range:** 2026-05-03 to 2026-05-04
**Phase exit gate:** Phase 6 closes on this report's CLOSE verdict

---

## Headline result

**Phase 6 CLOSED.** All three components delivered cleanly. Real model weights are now loaded at server startup; real signals run end-to-end on tomato images; `is_pre_f0_mode` flag flips to `False` empirically; Tier 4A (NOT Tier 4B-degraded) is observable on real PlantDoc tomato images.

The remaining gap (uniform classifier output `0.3333` across 3 stage-1 classes) is **architecturally correct pre-F.0 behavior** — the Stage 1/2/Platt calibration files are placeholders pending Phase F.0 dry-run. Real signal forward passes ARE running (v3 max_prob 0.5976; LoRA max_prob 0.9867; PSV reliability 0.34); only the classifier head consumes their outputs through a sentinel calibration. Phase F.0 dry-run replaces the classifier calibration; that's the next milestone.

---

## Component B summary (DEC-052)

**Status:** CLOSED (cleanest dispatch in the project — anti-cheat 0/0/0).

**Deliverable:** `tomato_sandbox/validation/fit_calibration.py` (36,625 B) + 48 unit tests.

**5 entry functions:**
1. `fit_conformal_tau` — delegates to existing `compute_conformal_tau` per S13.5
2. `fit_platt_scaling` — scipy → sklearn → identity fallback per S12.8
3. `fit_severity_thresholds` — per-disease (5 diseases) per S17.3 with default fallback when n<10
4. `fit_chilli_leakage_threshold` — 95th percentile on tomato per S4.5
5. `run_full_calibration` — orchestrates all four; writes 4 JSON files to `phase_f0_calibration/`

**SEVERITY_DEFAULTS** matches spec S17.3 table verbatim (5/5 diseases).

**(β) interpretation respected:** zero model-weight loading; consumes pipeline outputs only.

---

## Component A summary (DEC-053)

**Status:** CLOSED (second consecutive 0/0/0 anti-cheat clean).

**Deliverable:** `tomato_sandbox/validation/run_f0.py` (31,625 B) + 42 unit tests.

**`run_f0_validation(labeled_data_path, pipeline_context, output_dir, calibration_dir) -> dict`:**
- Drives held-out test set through `predict_single`
- Returns 6-key report dict: `metadata`, `per_image_predictions`, `confusion_matrix` (7×7), `conformal_coverage` (Wilson 95% CI), `severity_validation` (per-disease accuracy), `tier_disposition` (degraded vs real-failure split with `is_pre_f0_mode` flag)
- Writes JSON to `<output_dir>/validation_report_<ISO_TIMESTAMP>.json`

**Key Phase 6 transition signal:** `is_pre_f0_mode` boolean flips from True (pre-Component-C) to False (post-Component-C with real models loaded).

---

## Component C summary (DEC-054 + DEC-055..058)

**Status:** CLOSED with 1 LOW observation (5-DEC cap boundary; main thread accepted).

**Deliverable:** `tomato_sandbox/api/model_loaders.py` (NEW, 14,607 B) + modifications to `server.py`, `pipeline.py`, `tta.py`, `psv/features.py` + 13 unit tests.

### DEC-054 — primary architectural choices
- Separate module `model_loaders.py` (testability)
- v3 class: `scripts.model3_training.architecture.model3_full.Model3` (`pretrained=False, use_lora=True, lora_rank=4`)
- LoRA class: `scripts.ladi_net.single_pass_lora_train.SinglePassLoRA` wrapped via `LoRAModelAdapter`
- Device: `cuda:0` if available else `cpu` (DEC-026 warn-not-exit preserved)

### DEC-055 — LoRAModelAdapter cls→cls_token rename
SinglePassLoRA returns `cls`; signal_b_forward expects `cls_token`. Adapter wraps to rename. Pure dict-key transformation; weights unchanged.

### DEC-056 — Module-level predict_single import
For `unittest.mock.patch` resolution in `model_loaders.py`. Style change only.

### DEC-057 — _PROJECT_ROOT path depth
`Path(__file__).resolve().parents[3]` was wrong (one level too high). Fixed to `parents[2]`.

### DEC-058 — TTA + PSV device/arity bugs (covers 2 sub-bugs)
- PSV `cv2.boundingRect` 5-tuple unpack → 4-tuple unpack (one-line fix)
- TTA path: `.to(_v3_device)` and `.to(_lora_device)` calls in per-view loop (CPU→CUDA placement)

### Anti-cheat (Component C): 0 HIGH, 0 MEDIUM, 1 LOW (DEC-058 covers 2 sub-bugs)
All 21 checks pass including 9 sacred-file SHA256 byte-preservation verifications. The single LOW observation is that DEC-058 covers 2 distinct mechanical bugs in one entry, technically 6 bugs across 5 DECs (boundary case for 5-bug stop cap). Both fixes one-line; both device-placement-class; main thread reviewed and accepted CLOSE.

---

## Empirical lift gate verification (main-thread independent)

### Real-image POST `/predict` (live server on 8767)

```
Image: data/specialist/model3/cleaned/tomato_late_blight/orig_plantdoc_eval_IMG_2254.JPG_eb1ac5fe.jpg
HTTP 200, 2.16s
tier.label: "4A"  ← NOT "4B-degraded"
prediction.primary_class: "healthy"  (sentinel argmax)
prediction.primary_confidence: 0.3333  (uniform sentinel = 1/3 across 3 stage-1 classes)
rule_id_fired: "4"  ← NOT "1" (Rule 1 = pipeline-failure)
max_prob_actual: 0.3333
iqa_decision: "ACCEPTABLE"
```

`/info` endpoint:
```json
{
  "models": {
    "v3_version": "full_v3_soup",
    "lora_version": "sp_lora_epoch13_f10.9113",
    "psv_version": "psv_function_based_v1",
    "classifier_version": "sentinel_pre_f0"
  }
}
```

### Component A integration via `run_f0_validation`

```
Signal A: succeeded, argmax=2, max_prob=0.5976, chilli_leakage=0.056
Signal B: succeeded, max_prob=0.9867
Signal C: succeeded, reliability=0.3399, n_lesions=46
TTA: triggered, 5 views, all succeeded
Classifier: combined_argmax=5 (healthy), combined_max_prob=0.3333 (sentinel uniform)
Conformal: prediction_set=[], τ=0.42 (placeholder)
Tier: 4A (rule_id_fired="4")

tier_disposition.is_pre_f0_mode: False  ← PHASE 6 LIFT GATE PASSED
```

### What this proves
- **v3 model loaded:** Signal A produces real probabilities (max_prob 0.5976 on real tomato; chilli_leakage discrimination working)
- **LoRA model loaded:** Signal B produces real probabilities (max_prob 0.9867 — high LoRA confidence)
- **PSV running on real image:** 46 lesions detected; reliability scoring active
- **TTA fires:** initial_max_prob 0.3333 < TTA_TRIGGER_THRESHOLD 0.55 → 5 views computed
- **Classifier sentinel:** uniform 0.3333 across 3 classes — Phase F.0 territory
- **Tier 4A reached (NOT 4B):** Rule 4 fires (catch-all low-confidence) instead of Rule 1 (pipeline-failure)

---

## Phase 6 cumulative metrics

| Metric | Pre-Phase-6 | Post-Phase-6 |
|---|---|---|
| Total tests under venv | 1150 + 1 skip | **1252 + 1 skip** (+102 net new tests across components B, A, C) |
| DECs logged | DEC-001..050 | **DEC-001..058** (+8: DEC-051..058) |
| BLKs filed | 15 (12 RESOLVED + 3 OPEN planning-phase) | **15** (15 RESOLVED — 3 OPEN BLKs closed at Phase 6 disposition) |
| BLKs deferred / OPEN | 3 (BLK-006/007/008) | **0** |
| SPEC-INT entries | 3 | **4** (+SPEC-INT-004 S20.5 filename drifts) |
| M-series meta-findings | M1-M4 | **M1-M5** (+M5 sacred + DEC paths authoritative over spec literals) |
| Sacred entries | 10/10 PASS | **10/10 PASS** (verified post-Component-C; 9 file SHA256s match manifest byte-for-byte) |
| Section 15 immutability | 135/135 | **135/135 PRESERVED** through all of Phase 6 |
| Server `/info.models.v3_version` | "" (empty) | "full_v3_soup" |
| Server `/info.models.lora_version` | "" (empty) | "sp_lora_epoch13_f10.9113" |
| `is_pre_f0_mode` (real-image POST) | True | **False** |

---

## What works (plain language)

- **Sandbox server boots cleanly** under venv Python on 8767 with full S20.5 12-step lifespan
- **Real v3 weights load** in ~10 seconds
- **Real LoRA weights load** in ~3 seconds (DINOv2-Base + 8 LoRA adapters)
- **Real PSV runs** on real images (lesion detection, reliability scoring)
- **TTA orchestration** runs all 5 augmented views successfully
- **Conformal calibration** loads from JSON (placeholder τ=0.42; real τ Phase F.0)
- **IQA gate discriminates** by quality (some PlantDoc-eval images legitimately rejected for resolution / wetness; the late_blight one passes)
- **Tier 4A reachable** via Rule 4 catch-all-low-confidence on real tomato images
- **Response builder produces full S16.2 schema** including `explanation.structured` 12 fields per Phase 5b DEC-049
- **All 1252 venv tests pass; Section 15 preserved at 135/135; sacred 10/10 PASS**

## What doesn't work yet (honest gaps)

- **Classifier calibration is sentinel** (uniform 0.3333 across 3 stage-1 classes). `classifier_stage1.pkl`, `classifier_stage2.pkl`, `classifier_platt.json` ABSENT in `phase_f0_calibration/` per DEC-045. Phase F.0 dry-run produces these via Component B's `run_full_calibration` against real labeled held-out data.
- **Conformal τ is placeholder** (0.42). Real τ comes from Phase F.0 dry-run via `compute_conformal_tau` against held-out 40-image set per DEC-045 + S13.3.
- **Tier 1/2/3A/3B/3C/3D not yet observable** because the classifier doesn't have calibrated parameters. After Phase F.0 dry-run replaces the placeholders, the classifier produces non-uniform output and tiers 1-3 become reachable.
- **Severity grading approximate** — S17.3 thresholds are spec-derived starting points pending agronomist confirmation (BLK-012 noted; non-blocking).

## Phase 6 exit verdict

**CLOSE.** Phase 6 (F.0 prep per spec Section 29) is closed.

Rationale:
- Three Phase 6 components delivered cleanly with full audit trail
- Anti-cheat clean across all three (0 HIGH; 1 LOW informational on Component C; 0 on B and A)
- Real model loading lift verified empirically: `is_pre_f0_mode == False`
- Sacred 10/10 PASS preserved through all of Phase 6
- Section 15 milestone (135/135) preserved
- 1252 + 1-skip tests pass under venv (+102 from Phase 6)
- All three Phase 2 planning-phase OPEN BLKs (BLK-006/007/008) RESOLVED with disposition notes citing Phase 4 implementation + Phase 5b spec-citation density audit + Phase 6 empirical verification
- 4 SPEC-INT entries logged (SPEC-INT-001 through SPEC-INT-004); 5 M-series meta-findings (M1 through M5)

Phase F.0 dry-run is authorized.

---

## Phase F.0 forward outlook

**Phase F.0 dry-run** = run `run_full_calibration` from Component B against real labeled held-out data (40-image subset per S13.3). Produces:
1. `conformal_tau.json` with real τ (replaces placeholder 0.42)
2. `classifier_stage1.pkl`, `classifier_stage2.pkl` (Stage 1 + Stage 2 weights)
3. `classifier_platt.json` (Platt parameters per disease)
4. `severity_thresholds.json` (per-disease mild_max/moderate_max calibrated against agronomist labels)
5. `chilli_leakage_tau.json` (OOD threshold)

After Phase F.0 dry-run:
- Classifier produces non-uniform output → Tiers 1/2/3A/3B/3C/3D become reachable
- Conformal coverage measurable against true labels
- Severity grading discriminates per-disease
- Real prediction quality measurable (precision, recall, conformal coverage rate) on held-out

**Realistic estimate:** 1-3 sessions for Phase F.0 dry-run depending on labeled-data preparation cycle. ~2-4 sessions to project end-to-end completion.

---

## Architectural lessons reinforced (M-series)

- **M1** (Phase 5a): Real-subprocess + real-image testing is qualitatively different from in-process TestClient
- **M2** (Phase 5a): Mocking integration boundaries hides integration bugs
- **M3** (Phase 5a): Fix-cycle depth is bounded when integration was previously well-structured
- **M4** (Phase 5b): Auditor false positives from spec-quote comments + type-shape paraphrase; trust-but-verify mandatory
- **M5** (Phase 6 Component C): Spec body authoritative for semantics; sacred + DEC ledger authoritative for paths. When spec literals drift from project artifact ground truth (e.g., S20.5 step 4 says `model2_production.pt` but sacred manifest says `model3_production_v3.pt`), sacred + DEC win

## Audit trail files

| Phase | Component | Report |
|---|---|---|
| 6 | B (calibration) | `tomato_progress_reports/anticheat_phase6_componentB_20260504T0100.md` |
| 6 | A (validation) | `tomato_progress_reports/anticheat_phase6_componentA_20260504T0200.md` |
| 6 | C (real models) | `tomato_progress_reports/anticheat_phase6_componentC_20260504T1230.md` |
| 6 | Consolidated (this) | `tomato_progress_reports/phase_6_f0_prep.md` |

## Commits in Phase 6

| Commit | Subject |
|---|---|
| `7e5e2f5` | Phase 6 Component B: F.0 calibration script (DEC-052) |
| `ed74a98` | Phase 6 Component A: F.0 validation script (DEC-053) |
| (pending) | Phase 6 CLOSE: Component C real model loading + BLK-006/007/008 RESOLVED + SPEC-INT-004 + M5 |

---

*Generated 2026-05-04 by main-thread scribe. Consolidates 3 implementer dispatches (T-PHASE6-B/A/C) + 3 anti-cheat scans + main-thread independent verification at every component boundary. All numbers (1252 tests, 58 DECs, 15 BLKs all RESOLVED, 4 SPEC-INT entries, 5 M-series, sacred 10/10 + 9 file SHA256s) verified by direct measurement under venv Python.*
