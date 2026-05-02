# Anti-Cheat Scan — Phase 4 Batch 6 (T-IMPL-6a + T-IMPL-6b + T-IMPL-6c)

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-02
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS clean.** 0 HIGH, 0 MEDIUM, 1 LOW (defensive cleanup pattern; standard practice).

**Section 15 milestone preserved at 135/135.** Cumulative tests 1086 (951 unit + 135 integration).

## Summary table

| # | Check | Severity Cat. | Verdict |
|---|---|---|---|
| 1 | Section 15 LF-SHA256 vs DEC-032 baseline + 135/135 live pytest | HIGH | PASS — all 13 hashes match; live pytest 135/135 in 0.30s |
| 2 | `.git/hooks/pre-commit` md5 unchanged | HIGH | PASS — `24eb46f308751df3a125faca0680c9c7` |
| 3 | Suppressed failures | HIGH | PASS — zero unconditional skips. One `except Exception: pass` in `pipeline.py:653` GPU lock release `finally` block (LOW observation; defensive cleanup, not adversarial) |
| 4 | Spec citations on 8 spot-checked literals | MEDIUM | PASS — GPU lock cites S20.6/S21.3; TTA PSV exclusion cites S11.1:2925 + S11.9:3139-3140; Tier 4A routing cites S16.8:5813-5872 + BLK-010.3; severity thresholds cite S17.3; BLK-012 quotes S17.2:5955-5960 verbatim and verifies discrepancy in `FEATURE_NAMES` |
| 5 | No `print()` in production | MEDIUM | PASS — only one `print(` substring in `pipeline.py:17` docstring (not executable) |
| 6 | No APIN imports in any Batch 6 file | HIGH | PASS |
| 7 | DEC-038 compliance: no commits since `c757c5e` | HIGH | PASS — `git log c757c5e..HEAD` empty; all 3 implementers wrote files only |
| 8 | Honest test counts: 52 + 78 + 45 + 58 = 233; cumulative 951 unit | HIGH | PASS — pytest collect-only confirms; 1086 grand total live-verified |
| 9 | **NEW: No upstream module mutations** | HIGH | PASS — `git diff c757c5e..HEAD` for signals/, classifier/, conformal/, iqa/, tier/, preprocessing/, utils/, input_validation.py shows zero changes; only additions in this batch |
| 10 | **NEW: TTA PSV exclusion preserved** | HIGH | PASS — `test_psv_not_called_during_tta` (test_orchestrator.py:609) asserts PSV called exactly once via `counting_signal_c` wrapper; `apply_tta()` returns `(agg_signal_a, agg_signal_b, tta_report)` with no Signal C — structurally excluded from TTA |
| 11 | **NEW: GPU lock wraps A and B; excludes C** | HIGH | PASS — `pipeline.py:462-640` try block wraps Signal A + Signal B; `finally` at 641-654 releases; Signal C runs at line 661 **outside** the lock scope. `signals/psv/` has zero `gpu_lock` imports (Batch 3 contract preserved) |
| 12 | **NEW: BLK-010.3 Tier 4A routing correct** | HIGH | PASS — `_build_queue_block()` priority: 4B → never routed; tier5_alert → always routed; 3A/3B/3C/3D conditional; **4A → routed=False (user opt-in only)** unless Tier 5 co-fired (which check 2 catches first). Two tests cover both states: `test_tier4a_t5_false_not_routed` and `test_tier4a_t5_true_is_routed` |
| 13 | **NEW: BLK-012 honest** | HIGH | PASS — BLK-012 quotes spec S17.2:5955-5960 referencing `mean_lesion_intensity` and `lesion_size_distribution`; `FEATURE_NAMES` confirms neither exists (G3 has color fractions; G7 has IQA metrics). T-IMPL-6c used `mean_lesion_size` (G2 idx 3) and `lesion_size_std` (G2 idx 4) as proxies per Option A. Honest spec discrepancy disclosure |
| 14 | **NEW: DEC-044 covers BOTH S17 and S18** | MEDIUM | PASS — DEC-044 at line 976 has 7 decisions split: 1-5 address S17 (severity); 6-7 address S18 (multi-image). Non-monotonic file ordering (044 before 043 before 042) noted as parallel-append cosmetic artifact |
| 15 | **NEW: TierAssignment 3-field contract** | HIGH | PASS — `response_builder.py` accesses only `tier_label`, `tier5_alert`, `rule_id_fired`; `sub_rule_id_fired` in structured output is assigned from local `rule_id_fired` variable, not from a nonexistent attribute |

## LOW Observation (1)

**LOW-1:** `tomato_sandbox/orchestrator/pipeline.py` line 653 contains `except Exception: pass  # Best-effort release` in the `finally` block of GPU lock cleanup. This is the textbook defensive cleanup pattern for resource release where the outer exception handlers cover meaningful failures. The narrow scope (cleanup-only) and explicit comment mark it as intentional. Not a violation; not test-gaming.

## Process wins this batch

- **Section 15 milestone preserved.** The most important regression check passed — all 135 deterministic scenarios still pass after Batch 6 added orchestrator + response builder + severity + multi-image. This proves the orchestrator composes existing modules without mutating their behavior.
- **No upstream module mutations.** `git diff` for upstream Batch 1-5 modules is empty — Batch 6 is purely additive. Composability validated.
- **DEC-038 worked again, third batch in a row.** Three parallel implementers, full Bash access, zero implementer-driven commits. Pre-allocation rule honored: DEC-042/043/044 sequential with no collisions despite parallel writes.
- **BLK-012 surfaces real spec defect.** S17.2 references PSV features that don't exist in the implemented FEATURE_NAMES. Implementer used closest-spec-cited substitutions and documented honestly. Anti-cheat verified the discrepancy is real, not fabricated.
- **Three-parallel safe here** because each implementer's outputs are downstream consumers of upstream-stable contracts (orchestrator composes; response builder consumes dataclasses; severity/multi-image are spec-derived independents). No cross-implementer dependency = no two-wave needed.

## Carried-forward LOW concerns (re-noted, not new)

- 56 Pillow `getdata()` deprecation warnings (cosmetic; queued).
- DEC-022..044 pre-code-logging timing unverifiable (Critical Rule 9).
- Non-monotonic file ordering of DEC-042/043/044 in `tomato_decisions.md` (cosmetic; parallel-append artifact).

## Recommendation

Phase 4 Batch 6 is clean. Section 15 milestone preserved. Proceed to checkpoint 006 and main-thread commit per DEC-038. Batch 7 (server endpoint wiring) is the path to Q4 lift.
