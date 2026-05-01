# Phase 2 Exit Audit -- Second Fire
Audit timestamp: 2026-04-29T18:02
Auditor: phase-exit-auditor (run by main thread, read-only)
Previous audit: tomato_progress_reports/phase_2_exit_audit_20260427T213816.md -- verdict NOT READY

---

## Checklist

| Check | Location | Expected | Actual | Verdict |
|---|---|---|---|---|
| B1: summary table 9 columns | plan line 1345 | 9 cols | Task ID/Title/Owner subagent/Prerequisites/Spec sections/Files/AC/Complexity/Priority present | PASS |
| B1: expansion annotation | plan line 1343 | EXPANDED annotation | Present verbatim | PASS |
| B2: Fix-16 is last HIGH before MEDIUM section | plan item 13 + line 91 | Fix-16 at 13, MEDIUM heading follows | Confirmed | PASS |
| B2: Phase-3-critical preamble lists 7 fixes | plan lines 112-116 | Fix-9,10,11,12,16,19,20 | Present at line 115 | PASS |
| B3: Phase 3 Entry Preconditions 5 gates | plan lines 37-45 | 5 numbered gates | Gates 1-5 present; gates 4+5 have ADDITION 2026-04-28 markers | PASS |
| B3: spec_changelog gate 4 in T-PHASE-3-PRECONDITIONS | plan line 156 | Gate 4 verifies changelog | Present with required fields | PASS |
| B3: T-PHASE-3-PRECONDITIONS says 5 gates | plan line 152 | 5 gates | Header says 5 gates; steps 1-5 are verification gates; step 6 is doc output (cosmetic minor) | PASS |
| Defect-9.1: TTA signature corrected | plan lines 563-570 | single float -> int | Present; PATCHED annotation at line 594 | PASS |
| Defect-9.1: AC has 3-level tests + NaN guard | plan lines 597-600 | 4 tests | All present | PASS |
| Defect-9.2: T-IMPL-3a says internal remap | plan line 1356 | internal remap in title | Signal A wrapper (v3 10->6 with internal remap) | PASS |
| Defect-9.2: T-IMPL-4a NO REMAP + regression test | plan lines 619-635 | NO REMAP + remap-NOT-here test | Both present | PASS |
| Defect-9.2: dep graph critical edge 2 corrected | dep_graph line 72 | CORRECTED 2026-04-28; S12 no remap | Present verbatim | PASS |
| Defect-9.2: dep graph S8+S9 rows corrected | dep_graph lines 95-96 | CORRECTED on both | Both rows corrected | PASS |
| Defect-9.3: Rule 3 body says > 0.40 strict | plan line 748 | chilli_leakage > 0.40 strict + fix annotation | Present at line 748 | PASS |
| Defect-9.3/9.4: Full R1-R9 chain + sub-rules | plan lines 746-768 | Rules 1-9 with 7a/7b/7c 8a/8b/8c | Present | PASS |
| BLK-009 4 sub-defects PATCHED in blockers | blockers lines 220-234 | PATCHED on 9.1,9.2,9.3,9.4 | All PATCHED | PASS |
| Log records patch batch 2026-04-28 04:30 | log lines 139-159 | entry with file list | Present | PASS |
| RD-1 RESIDUAL: T-IMPL-5a AC boundary consistent with > 0.40 strict | plan line 796 | 0.41 fires; 0.40 does not | Line 796: 0.30 fires; 0.29 does not. WRONG. Contradicts Rule 3 body. | FAIL |
| RD-2 RESIDUAL: T-IMPL-5b SB.7 correct threshold + rule name | plan lines 818-819 828 | 0.41; rule R3 | Line 818: 0.30; line 828: rule R2. Both wrong. | FAIL |
| RD-3 RESIDUAL: T-EARLY-MP fix list ordered HIGH->MEDIUM->LOW | plan lines 77-88 | HIGH all before MEDIUM | Items 4-6 MEDIUM and 7-8 LOW appear before items 9-13 HIGH | FAIL |

---

## Detailed Findings

### RD-1 (HIGH) -- T-IMPL-5a acceptance criterion boundary test wrong

Location: tomato_plan.md line 796.
Written: chilli_leakage=0.30 -> guard fires; chilli_leakage=0.29 -> guard does not fire.
Required: Rule 3 body (line 748) says chilli_leakage > 0.40 strict.
Fix: chilli_leakage=0.41 -> guard fires; chilli_leakage=0.40 -> guard does NOT fire (strict).
Impact: Implementer following only the AC will code the wrong threshold.
The rule body was patched for Defect-9.3 but the acceptance criterion was not updated.

### RD-2 (HIGH) -- T-IMPL-5b smoke test SB.7 wrong boundary value and rule name

Location: tomato_plan.md lines 818-819 and 828.
Line 818: chilli_leakage=0.30 boundary trap (inclusive; guard fires)
Line 828: SB.7 (chilli_leakage=0.30) -> guard fires (rule R2 fires).
Fix: Use chilli_leakage=0.41; rule name R3 (Section 14 numbering).
Impact: Smoke test at 0.30 passes on a wrong implementation with old threshold, defeating the test.

### RD-3 (MEDIUM) -- T-EARLY-MP fix list ordering violates HIGH-before-MEDIUM

Location: tomato_plan.md lines 80-88 (items 4-8 in T-EARLY-MP fix list).
Current order: 1-3 HIGH, 4-6 MEDIUM, 7-8 LOW, 9-12 HIGH, 13 HIGH, 14-16 MEDIUM, 17 LOW
MEDIUM items 4-6 and LOW items 7-8 appear before HIGH items 9-13.
T-EARLY-MP AC line 122 requires HIGH-before-MEDIUM-before-LOW.
The B2 patch moved Fix-16 to item 13 but did not move Fix-4/5/6/7/8.
Fix: Renumber fix list so all HIGH items precede any MEDIUM, all MEDIUM precede any LOW.

---

## Verdict

NOT READY for Phase 3.

Blockers:
- RD-1 (HIGH): tomato_plan.md line 796 -- T-IMPL-5a AC tests 0.30 boundary; Rule 3 body says > 0.40 strict.
- RD-2 (HIGH): tomato_plan.md lines 818-819+828 -- T-IMPL-5b SB.7 uses 0.30 and rule R2; should be 0.41 and R3.
- RD-3 (MEDIUM): T-EARLY-MP fix list items 4-8 (MEDIUM/LOW) precede items 9-13 (HIGH); violates own AC.

To become READY: Fix RD-1, RD-2, RD-3 in tomato_plan.md. Re-fire gate.