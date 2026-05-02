# Anti-Cheat Scan — Phase 4 Batch 3 (T-IMPL-3a/3b/3c/3d)

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-02
**Saved by:** main-thread scribe per DEC-011

## Verdict: **CONDITIONAL PASS.** 0 HIGH, 1 MEDIUM (cosmetic docstring undercount), 4 LOW.

## Summary table

| # | Check | Severity | Verdict | Evidence headline |
|---|---|---|---|---|
| 1 | Section 15 LF-normalized SHA256 vs DEC-032 baseline | HIGH | PASS | All 13 hashes match exactly. No content drift since Phase 3 baseline (Windows autocrlf flips raw bytes; LF-normalized comparison is the truth source). |
| 2 | `.git/hooks/pre-commit` md5 | HIGH | PASS | `24eb46f308751df3a125faca0680c9c7` confirmed. |
| 3 | Suppression (xfail / skipif(True) / bare except) | HIGH | PASS | Zero unconditional skips. test_signal_b.py and test_tta.py have legit `skipif(not _TORCH_AVAILABLE / not _PIL_AVAILABLE)` env-dep guards. (Two LOW findings about coverage breadth — see below.) |
| 4 | Spec citations (8 spot checks) | MEDIUM | PASS | v3 remap `[0,2,1,3,4,5]` cites `# spec: section 8.3 lines 1672-1674`; PSV 26-feature count test asserts `len(FEATURE_NAMES) == 26`; PSV per-feature BLK-007 traceability comments verified on 5 sampled features; TTA `should_trigger_tta` cites S11; TTA "PSV not invoked" docstring cites S11.1:2925 and S11.9:3139-3140. (One LOW: v3 citation could be widened to BLK-009 full range 1664-1685.) |
| 5 | No `print()` in production | HIGH | PASS | Zero `print(` in v3_signal.py, lora_signal.py, psv/*.py, signals/tta.py, tta.py shim. |
| 6 | No APIN imports | HIGH | PASS | No `apin` references anywhere in Batch 3. |
| 7 | PSV CPU-only | HIGH | PASS | Zero `torch.cuda` and zero `gpu_lock` imports under `signals/psv/`. Section 10 constraint satisfied. |
| 8 | Honest test counts | MEDIUM | PASS (with M-1) | 15 + 18 + 56 + 34 = 123 Batch 3; cumulative 538. Pytest verified. **MEDIUM-1:** test_tta.py docstring claims "29 tests total" but 34 collected — undercount in docstring, not in reality. Cosmetic; tests themselves are real. |
| 9 | DEC-034/035/036/037 integrity | MEDIUM | PASS | Sequential, no collisions. Pre-allocation rule honored across both waves (3 parallel + 1 sequential). |
| 10 | signals/__init__.py minimality | MEDIUM | PASS | Bare module docstring + `__all__: list[str] = []`. No shadow imports, no circular-dep risk. |
| 11 | `tier/` absent; Section 15 still failing | HIGH | PASS | `ls tomato_sandbox/tier/` → does not exist. Live pytest fires 13 ModuleNotFoundError collection errors. |
| 12 | v3 remap inside `extract_v3_outputs` | HIGH | PASS | Remap applied at the function boundary (per BLK-009 Defect-9.2). `test_remap_correctness` sends v3-ordered → asserts canonical-ordered output. (One LOW: citation range narrower than BLK-009 spans.) |
| 13 | Signal B single-pass | HIGH | PASS | `test_single_pass_only` asserts `mock_model.call_count == 1` after `compute_signal_b`. (One LOW: skips with torch unavailable.) |
| 14 | TTA does NOT invoke PSV | HIGH | PASS | `test_psv_not_called` patches `compute_signal_c` and asserts `call_count == 0`. (One LOW: skip guard is PIL+torch but the test only needs mocks.) |

## Findings

### MEDIUM-1 — `test_tta.py` docstring claims "29 tests total"; 34 collected
- **Severity:** MEDIUM (cosmetic).
- **Direction:** undercount in docstring (more tests exist than claimed) — safer direction than overcount.
- **Fix:** update docstring to "34 tests total". Queue for follow-up commit.

### LOW-1 — v3 remap citation narrower than BLK-009 Defect-9.2 range
- v3_signal.py cites `# spec: section 8.3 lines 1672-1674` for the remap constant.
- BLK-009 Defect-9.2 derivation context spans 1664-1685.
- Suggested expansion: `# spec: section 8.3 lines 1664-1685 (BLK-009 Defect-9.2)`.
- Severity: LOW (citation is correct; broader context not cited).

### LOW-2 — Signal B single-pass test skips on no-torch CI
- `pytestmark = skipif(not _TORCH_AVAILABLE)` covers all 18 Signal B tests including `test_single_pass_only`.
- The test could potentially run with a pure-mock model not requiring torch.
- Severity: LOW (constraint is verified locally; gap is in no-torch CI environments).

### LOW-3 — TTA PSV-not-called test skips on no-PIL or no-torch
- The mock-based assertion `mock_psv.call_count == 0` doesn't require PIL or torch, but it inherits the class-level skip guard.
- Severity: LOW (test runs in dev env where Wave 1/2 ran).

### LOW-4 — Git provenance per-file granularity (auditor mislocation)
- Auditor reported "files landed in commit `28cc945`" — that's the old initial scaffold and did not contain Batch 3 files.
- Actual provenance: all Batch 3 files first tracked in commit `a926d3d` (2026-05-02 backfill commit per DEC-032). Subsequent Wave 1+Wave 2 implementer changes are uncommitted on disk at the time of the scan.
- Severity: LOW; auditor flagged a non-issue. For future, recommend per-batch commits before each anti-cheat run so provenance is traceable.

## Carried-forward LOW concerns (re-noted, not new)
- 56 Pillow `getdata()` deprecation warnings (cosmetic; queued).
- DEC-022..037 pre-code-logging timing unverifiable (Critical Rule 9; substantive content honest).
- 1 `# noqa: F401` in lora_signal.py for `GPULock` task-card-compliance import (DEC-035 documented).

## Recommendation

Phase 4 Batch 3 is clean. Proceed to checkpoint 003 and STOP per master prompt cadence.
