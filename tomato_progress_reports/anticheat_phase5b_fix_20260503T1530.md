# Anti-Cheat Scan — Phase 5b Fix (T-AUDIT-5b-fix)

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-03
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS clean** on substantive checks. **0 HIGH, 1 MEDIUM (test count accounting), 2 LOW (cosmetic).**

All implementation-level checks (15 total) passed. The MEDIUM finding is in main-thread accounting (dispatch baseline number was 1121 unit+integration; actual all-tests grand total is 1150 = unit+integration+e2e). All 1150 tests pass; no test was suppressed or fabricated.

## Summary table

| Check | Result | Severity |
|---|---|---|
| 1. Section 15 LF-SHA256 baseline preserved (13 hashes) | PASS | — |
| 2. Pre-commit hook md5 unchanged (`24eb46f308751df3a125faca0680c9c7`) | PASS | — |
| 3. Suppressed failures (xfail / skipif / bare except) | PASS | — |
| 4. Spec citations on 6 spot-checked literals | PASS | LOW (range-cite vs per-line) |
| 5. No `print()` in production | PASS | — |
| 6. No APIN imports | PASS | — |
| 7. DEC-038 — no commits since `dd41794` | PASS | — |
| 8. **Test count accuracy** | **MEDIUM** | **MEDIUM** |
| 9A. signal_extra vs TierAssignment fattening | PASS | — |
| 9B. sub_rule_id_fired distinct ("default" vs actual) | PASS | — |
| 9C. SPEC-INT-003: same coverage_pct for all classes | PASS | — |
| 9D. Healthy/OOD exclusion from grade_per_class | PASS | — |
| 9E. multi_class_set only for Tier 3A/3B | PASS | — |
| 14. tier/ preserved + Section 15 135/135 | PASS | — |
| 15. No upstream module mutations beyond 5 expected files | PASS | — |
| Hardcoded test values | PASS | — |
| Mocked failures | PASS | — |

## MEDIUM finding (1)

### MEDIUM-1: Test count discrepancy in dispatch baseline (vs actual)

The dispatch prompt for T-AUDIT-5b-fix stated cumulative pre-fix baseline as "986 unit + 135 integration = 1121." The actual cumulative is **1150** (986 unit + 135 integration + 29 e2e at `tomato_sandbox/tests/e2e/test_endpoints.py` from T-IMPL-7 Batch 7). The dispatch number missed the e2e bucket.

**Resolution (main thread, 2026-05-03):** Verified independently via `venv/Scripts/python.exe -m pytest tomato_sandbox/tests/ --collect-only -q` → 1150 collected. Confirmed live full run: 1150 passed, 0 failed.

**Decomposition:**
- 986 unit tests (in `tests/unit/`)
- 135 Section 15 integration tests (in `tests/integration/`, immutable per Rule 6)
- 29 e2e tests (in `tests/e2e/test_endpoints.py`)
- **Total: 1150 PASS**

**No test was suppressed, removed, or fabricated.** The 25 new tests added by T-AUDIT-5b-fix (14 in test_response_builder.py + 11 in test_severity.py) all PASS and exercise the fixed behavior. The discrepancy is purely in main-thread accounting — going forward, all checkpoint reports and dispatch prompts must specify whether the cited count is "unit + integration" or "all tests including e2e".

## LOW findings (2)

### LOW-1: Section 15 SHA256 baseline not stored as verifiable record
DEC-032 entry describes the git-tracking policy, not a per-file hash registry. Computed hashes at audit time (Check 1 above) serve as the forward baseline. Git provenance (single commit `a926d3d` for all 13 files) provides immutability proof. **Suggested fix (deferred to T-EARLY-MP):** add a structured hash table to a dedicated baseline file, e.g. `tomato_sandbox/tests/integration/.section15_baseline.json`.

### LOW-2: Spec citation granularity for 4 new threshold fields
Implementation cites the region as `spec: section 16.4 lines 5759-5775` rather than per-field individual line numbers (`5759`, `5761`, `5763`, `5765`). The fields are correctly implemented and the region cite is accurate; documentation granularity gap only.

## Pass Detail Highlights

### Check 9A — signal_extra pattern (no TierAssignment fattening)
- `TierAssignment` dataclass at `tier_assignment.py:83-99` has exactly 3 fields (`tier_label`, `tier5_alert`, `rule_id_fired`); unchanged
- `build_response()` accepts `signal_extra: Optional[dict] = None` at line 511
- Pipeline.py extracts `signal_a.chilli_leakage` and `signal_c.psv_reliability`; passes as `signal_extra={"chilli_leakage_actual": ..., "psv_reliability_actual": ...}`
- 135 Section 15 tests that rely on `TierAssignment` 3-field shape are unaffected — verified by 135/135 still passing

### Check 9C — SPEC-INT-003 compliance (shared coverage_pct)
- `grader.py:445-477` reads `coverage_pct = feats[_IDX_DISEASE_COVERAGE_PCT]` ONCE before the per-class loop
- Loop at line 453 passes the SAME `coverage_pct` to `_grade_from_thresholds` and stores `"coverage_pct": round(coverage_pct, 2)` in every dict entry
- Test `test_grade_per_class_same_coverage_pct_for_all_classes` asserts `len({e["coverage_pct"] for e in result.grade_per_class}) == 1` — set has exactly 1 unique value

### Check 15 — No upstream module mutations
`git diff HEAD --name-only -- tomato_sandbox/` returns exactly the 5 expected files:
```
tomato_sandbox/orchestrator/pipeline.py
tomato_sandbox/response/response_builder.py
tomato_sandbox/severity/grader.py
tomato_sandbox/tests/unit/test_response_builder.py
tomato_sandbox/tests/unit/test_severity.py
```

## Cumulative metrics post-Phase-5b-fix

| Metric | Value |
|---|---|
| Unit tests | 986 PASS |
| Section 15 integration | 135/135 PASS |
| E2e tests | 29 PASS |
| **Grand total** | **1150 PASS** |
| Sacred | 10/10 PASS |
| DECs | DEC-001..050 |
| BLKs | 14/15 RESOLVED (BLK-014 + BLK-015 closure pending status update by main thread) |
| Pre-commit hook | armed (md5 unchanged) |
| DEC-038 compliance | verified (no commits since `dd41794`) |

## Recommendation

T-AUDIT-5b-fix is clean. Phase 5b can CLOSE. Main thread should:
1. Update BLK-014 and BLK-015 to RESOLVED in `tomato_blockers.md`
2. Log SPEC-INT-003 in `spec_changelog.md`
3. Log M4 family extension in project meta-findings (auditor paraphrase loose on F-07 type-shape)
4. Append `tomato_log.md` close-out entry (citing 1150 grand total, not 1121)
5. Single commit per DEC-038 (no Section 15 staged; pre-commit hook expected to pass cleanly)
6. Authorize Phase 5c (anti-cheat final sweep + Phase 5 consolidated report)
