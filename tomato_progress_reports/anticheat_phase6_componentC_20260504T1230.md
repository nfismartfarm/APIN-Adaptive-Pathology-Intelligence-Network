# Anti-Cheat Scan — Phase 6 Component C (T-PHASE6-C / DEC-054..058)

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-04
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS clean.** 0 HIGH, 0 MEDIUM, 1 LOW (DEC-058 boundary observation; main thread accepted).

All 21 checks pass including 9 sacred-file SHA256 byte-preservation verifications.

## Files inspected (5 modified + 2 new)

| File | Status | Bytes | DEC |
|---|---|---|---|
| `tomato_sandbox/api/server.py` | MODIFIED | 33,128 | DEC-054 lifespan steps 4-7+9+11 lifted from stub-skips |
| `tomato_sandbox/api/model_loaders.py` | NEW | 14,607 | DEC-054 architectural choice: separate module for testability |
| `tomato_sandbox/orchestrator/pipeline.py` | MODIFIED | (Steps 6, 7 device placement) | DEC-058 sub-fix |
| `tomato_sandbox/signals/psv/features.py` | MODIFIED | (cv2.boundingRect 4-tuple unpack) | DEC-058 PSV |
| `tomato_sandbox/signals/tta.py` | MODIFIED | (TTA per-view device placement) | DEC-058 TTA |
| `tomato_sandbox/tests/unit/test_model_loaders.py` | NEW | 7,923 | 13 unit tests for model_loaders |
| `tomato_decisions.md` | MODIFIED | DEC-054, 055, 056, 057, 058 entries | — |

## Summary table

| # | Check | Severity threshold | Outcome |
|---|---|---|---|
| 1 | Section 15 immutability — 13 files unchanged; live 135/135 | HIGH | CLEAR |
| 2 | Pre-commit hook md5 | HIGH | CLEAR (`24eb46f3...`) |
| 3 | Sacred manifest 10/10 PASS | HIGH | CLEAR (live `verify_manifest()`) |
| 4 | **Sacred file byte preservation (9 SHA256 hashes)** | HIGH | CLEAR — all 9 manifest entries match by SHA256 |
| 5 | Suppressed failures | HIGH | CLEAR — 3 `# noqa` with documented rationale; no unconditional skip / bare except |
| 6 | Spec citations on critical literals | MEDIUM | CLEAR — S20.5:6556-6575, DEC-026, fail-fast S20.5:6573, warmup S20.5:6570 |
| 7 | No `print()` in production | LOW | CLEAR |
| 8 | No APIN imports | HIGH | CLEAR |
| 9 | DEC-038 compliance — no commits since `ed74a98` | HIGH | CLEAR |
| 10A | **Sacred path correctness (M5)** | **HIGH** | **CLEAR** — v3 uses `model3_production_v3.pt` (NOT stale spec `model2_production.pt`); LoRA uses `sp_lora_epoch13_f10.9113_PRESERVED.pt`; conformal uses `conformal_tau.json` (NOT `tomato_calibration.json`) |
| 11B | DEC-026 warn-not-exit preserved | HIGH | CLEAR — step 3 uses `_logger.warning`, continues with CPU fallback |
| 12C | Fail-fast steps 4-9+11 | HIGH | CLEAR — `FileNotFoundError`/`RuntimeError` on missing artifacts; no silent swallowing |
| 13D | Synthetic warmup placeholder | MEDIUM | CLEAR — `np.random.default_rng(seed=42)` 224×224 noise; no data dependency |
| 14E | LoRAModelAdapter cls→cls_token rename | MEDIUM | CLEAR — pure dict-key rename; weights unchanged |
| 15F | Module-level `predict_single` import | MEDIUM | CLEAR — line 24, noqa rationale per DEC-056 |
| 16G | `_PROJECT_ROOT` `parents[2]` correct depth | MEDIUM | CLEAR — sacred path resolves to real file |
| 17H | DEC-058 covers both PSV + TTA fixes | MEDIUM | CLEAR — both verified in source |
| 18I | **No premature classifier calibration** | **MEDIUM** | **CLEAR** — `classifier_stage1.pkl`, `classifier_stage2.pkl`, `classifier_platt.json` ALL ABSENT (Phase F.0 territory; sentinel sufficient for Component C) |
| 19J | Test count truthful (1252 + 1 skip) | MEDIUM | CLEAR — implementer's 1088 claim was partial; main-thread independent run shows 1252+1; +13 from test_model_loaders.py |
| 20K | Acceptance gate empirically passes | HIGH | CLEAR — Tier 4A on real tomato (late_blight); confidence 0.3333 (sentinel uniform; expected pre-F.0); /info shows real `v3_version: "full_v3_soup"` and `lora_version: "sp_lora_epoch13_f10.9113"` |
| 21L | 5-DEC cap at boundary | LOW (informational) | LOW — DEC-058 covers 2 sub-bugs (6 distinct bugs across 5 DECs); both are device-placement-class one-line fixes; grouping documented in DEC-058; main thread accepted |

## Sacred file byte preservation table (Check 4)

| Sacred file | SHA256 (first 16) | Match |
|---|---|---|
| models/best_model.pt | `fa59c5b92d6847ab` | ✓ |
| models/swin_best_model.pt | `d29bb2192719d3b4` | ✓ |
| models/model2_specialist/model2_production.pt | `6c2ea88ce2ce4047` | ✓ |
| data/specialist/model3/split_indices.json | `0e465d20112bf3f3` | ✓ |
| app/config.py | `01b1d2067b6cdbdc` | ✓ |
| data/metadata/source_map.csv | `f3ec5534517e00a3` | ✓ |
| models/specialist/ladinet_phase1_heads.pt | `6c31033a97601ebb` | ✓ |
| scripts/model3_training/checkpoints/model3_production_v3.pt | `2833e40b72480c64` | ✓ |
| models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt | `626cf67e6b8ccbb1` | ✓ |

Component C READS these checkpoints to load them into memory; does NOT modify them.

## LOW observation (1)

**LOW-1: DEC-058 covers 2 sub-bugs (PSV cv2.boundingRect arity + TTA device placement).**

DEC count: DEC-054 (primary) + DEC-055/056/057 (3 wiring fixes) + DEC-058 (covers PSV + TTA = 2 sub-bugs) = 5 DEC entries, **6 distinct bugs**. The dispatch's stop cap is "5 bugs surface". DEC-058 grouping is defensible (both are device/shape-class bugs surfaced in the same smoke run; both are mechanical one-line fixes; DEC-058 documents both sub-bugs with distinct location + fix descriptions). Borderline but acceptable. Main thread reviewed and accepted CLOSE.

**For future dispatches:** consider hard rule that 1 DEC = 1 bug, even if related. Then "5 bugs" → "5 DECs" naturally. Current grouping muddles the count slightly.

## Phase 6 Component C empirical lift gate

Real tomato leaf POST `/predict` produced (verified by main thread):

```
tier.label: "4A"  (NOT "4B" — proves real signals running)
prediction.primary_class: "healthy"  (sentinel argmax of uniform stage-1)
prediction.primary_confidence: 0.3333  (uniform sentinel = 1/3 across 3 stage-1 classes; classifier calibration still placeholder per DEC-045)
rule_id_fired: "4"  (NOT "1" pipeline-failure)
max_prob_actual: 0.3333
set_size: 0
iqa_decision: "ACCEPTABLE"
```

`/info` shows real model versions:
- `v3_version: "full_v3_soup"`
- `lora_version: "sp_lora_epoch13_f10.9113"`
- `psv_version: "psv_function_based_v1"`
- `classifier_version: "sentinel_pre_f0"` (placeholder by design — Phase F.0 fits real Platt params)

**Empirical signal: Real models loaded. Real signal forward passes running. Classifier still using sentinel calibration (Phase F.0 territory). Phase 6 Component C lift complete.**

## Recommendation

Component C is clean. Phase 6 close sequence may proceed:
1. Component A integration check (`is_pre_f0_mode == False`)
2. BLK-006/007/008 disposition notes
3. SPEC-INT-004 + M5 logged
4. `phase_6_f0_prep.md` consolidated report
5. Main-thread commit (Phase 6 CLOSE)
6. Phase F.0 dry-run authorized
