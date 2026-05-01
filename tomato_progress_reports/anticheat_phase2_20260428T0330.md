# Anti-Cheat Inspection Report — Phase 2 (Planning)

**Inspector:** anti-cheat-inspector (read-only)
**Date:** 2026-04-28
**Saved by:** main-thread scribe per DEC-011

## Verdict: PASS with 3 CONCERNS

No deliberate gaming, fake completion, suppressed failures, or hidden defects detected. Fix-9/Fix-10 inversion errors are properly disclosed and corrected in `tomato_plan.md` with explicit `[CORRECTED 2026-04-28 by main-thread scribe]` annotations. BLK-004 three-location overstatement properly corrected and not re-introduced anywhere.

However, the inspector sampled 5 task-level spec citations and found **3 material spec-citation discrepancies in the plan** — these are not cheating but are comprehension errors that would propagate to Phase 4 if uncorrected.

## Section-by-section

| Check | Verdict |
|---|---|
| Section 15 test modifications | CLEAR (zero test files exist) |
| Implementation code in tomato_sandbox/ | CLEAR (zero .py/.yaml; only .gitkeep) |
| Suppressed failures | CLEAR (no pytest.skip/xfail/etc.) |
| Fake completion claims | CLEAR (log entries match disk state; no false "done" markers) |
| Fix-9/Fix-10 inversion disclosure | CLEAR (corrections labeled with author and rationale) |
| BLK-004 "three locations" claim | CLEAR (correction stable; not re-introduced) |
| Acknowledged vs hidden defects | CLEAR (B1/B2/B3 from auditor + 8 PDA defects all transparently logged) |
| Spec citation gaming (5 samples) | **3 of 5 had material discrepancies** — see concerns |

## CONCERN 1 (MEDIUM) — T-IMPL-3d TTA function diverges from Section 11 spec

**Plan (line 517):** `should_trigger_tta(signal_a, signal_b) -> bool` with `max_prob < 0.55 OR margin < 0.45`

**Spec summary (`section_11.md`):** `should_trigger_tta(combined_max_prob: float) -> int` returning **1, 2, or 5** (number of views), not bool. No margin parameter. 3-level decision based on `combined_max_prob` only:
- `>= 0.55` → 1 view (no TTA)
- `[0.45, 0.55)` → 2-view TTA
- `< 0.45` → 5-view TTA

**Risk:** Phase 4 implementer following plan would build wrong function signature, missing the 5-view path entirely.

## CONCERN 2 (MEDIUM/HIGH) — T-IMPL-3a "native ordering" annotation may contradict Section 8

**Plan annotation (Batch 3 + line 399):** "Signal A returns probs in NATIVE v3 ordering (NOT canonical). The remap [0,2,1,3,4,5] is applied ONLY at T-IMPL-4a."

**Spec summary (`section_08.md`):** `SignalAResult` field is named `tomato_probs_canonical` (already-canonical, post-remap). The `extract_v3_outputs` contract returns canonical-ordered probs. The remap is documented in Section 8 as part of the v3-to-canonical transform.

**Risk:** if Section 8 spec actually requires Signal A wrapper to deliver canonical probs (already remapped), then the plan's T-IMPL-4a "remap here only" annotation would DOUBLE-REMAP — exactly the failure mode the plan claims to prevent. The architectural invariant in the plan may be inverted relative to spec.

This contradicts dependency-graph critical edge 2 ("Index remap [0,2,1,3,4,5] applied during fusion (Section 12), NOT at signal output") which the plan was supposed to bake in. Either the dependency graph or the spec_summary is wrong; needs reconciliation against the spec body.

## CONCERN 3 (LOW for cheating; HIGH for implementation) — T-IMPL-5a chilli_leakage threshold conflated

**Plan (line 675, 707):** "R2: chilli_leakage guard (chilli_leakage >= 0.3 inclusive; BLK-004/BLK-005 boundary)"

**Spec summary (`section_14.md`):** Rule 3 fires at `chilli_leakage > 0.40` (strict greater than 0.40). Tier 2 eligibility uses `chilli_leakage < 0.30` (strict). Section 15 SB.7/SB.13 boundary traps confirm: 0.40 is Rule 3 boundary, 0.30 is Tier 2 boundary.

**Risk:** the plan conflated two separate thresholds. `>= 0.30` for Rule 3 would route many more cases to Tier 3C than spec intends. Plan also incorrectly cites BLK-004/BLK-005 as the source — neither blocker establishes a 0.30 threshold.

## Inspector verdict

PASS for cheating patterns. NOT-FAIL because no honesty issues. But these 3 spec-citation discrepancies are real and must be reconciled before Phase 4 begins. The Phase 3 precondition gate (T-PHASE-3-PRECONDITIONS) does not currently catch them. Suggest adding a "spec-citation reconciliation" step to the gate.
