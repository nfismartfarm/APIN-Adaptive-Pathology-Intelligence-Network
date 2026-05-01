# Anti-Cheat Audit — Phase 3 Exit Gate

**Auditor:** anti-cheat-inspector
**Timestamp:** 2026-05-01
**Saved by:** main-thread scribe per DEC-011 (agent has Write tool per Fix-10 but interpreted its read-only operating constraint as forbidding report saves; returned text)

## Verdict: PASS with one LOW concern

No HIGH severity violations found. One LOW severity concern documented below.

## Summary table

| # | Check | Verdict | Severity |
|---|---|---|---|
| 1 | All 135 tests fail with ModuleNotFoundError | PASS | — |
| 2 | `tomato_sandbox/tier/` does not exist | PASS | — |
| 3 | No `pytest.skip`/`xfail`/`mark.skip`/bare-except | PASS with concern | LOW |
| 3a | `# noqa: E402` on import line in all 13 files | LOW concern | LOW |
| 4 | Spec citations match spec body (3 spot-checks: S1.1/S3B.4/S3D.5) | PASS | — |
| 5 | S1.1 priors = `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` per SPEC-INT-001 | PASS | — |
| 6 | 7 body-wins decisions (SPEC-INT-002) match spec body | PASS | — |
| 7 | Import contract is substantive | PASS | — |
| 8 | No hardcoded implementation knowledge in tests | PASS | — |
| 9 | DEC-019 manifest evolution honest | PASS | — |
| Sec15 mod | Section 15 test files not modified post-Phase-3 | PASS | — |

## Detailed findings

### Check 1 — PASS
`pytest tomato_sandbox/tests/integration/ -q --tb=no` → 13 collection errors, 0 tests collected, 0 skipped, 0 xfailed. All 13 errors with identical `ModuleNotFoundError: No module named 'tomato_sandbox.tier'`.

### Check 2 — PASS
`tomato_sandbox/tier/` does not exist. Phase 3 did not pre-create implementation.

### Check 3 — PASS with LOW concern
- Zero `pytest.skip`, `pytest.xfail`, `pytest.mark.skip`, `pytest.mark.skipif`, bare `except:`, empty except blocks.
- LOW concern: `# noqa: E402` on each import line in all 13 test files. E402 (module-level import not at top) does not actually fire under standard flake8 in this file structure (docstring → `import pytest` → comment → flagged import). The noqa comment is cosmetically suppressing a warning that wouldn't fire. Not functionally deceptive (does not bypass any test or change pytest collection); just noise. Suggested fix during T-EARLY-MP cleanup.

### Check 4 — PASS
Spot-checks of S1.1 (lines 4116-4124), S3B.4 (lines 4487-4493), S3D.5 (lines 4711-4720) against spec body all match exactly.

### Check 5 — PASS
Spec line 4117: `v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03`. Test uses identical values. Line 5558 typo not used.

### Check 6 — PASS
Both S3B.4 and S3D.5 spot-checked against spec body — assertions align with body content (Tier 4A/Rule 4 and Tier 3A/Rule 6 respectively), not with subsection heading.

### Check 7 — PASS
`.claude/import_contract.md` (8116 bytes) covers full assign_tier signature (keyword-only), TierAssignment dataclass (3 attrs), all 12 `rule_id_fired` values, full input dict schemas, T5 alert logic, threshold reference table (13 entries), scenario-to-file mapping (13 rows / 135 total).

### Check 8 — PASS
Test inputs come from spec scenario bodies; expected outputs come from scenario `→ Tier X` outcome lines. No pre-computed internal state asserted.

### Check 9 — PASS
`.claude/sacred_manifest.json` `rebaseline_history` array has 2 entries: pre-DEC-019 (a602722f, file_count 316, no exclusions) and post-DEC-019 (452d697b, file_count 145, `["*.log", "*.log.*"]` exclusions). Old hash preserved. Verification report at `tomato_progress_reports/sacred_post_dec019_20260501T0000.md` shows 10/10 PASS.

### Section 15 modification check — PASS
`tomato_sandbox/` is gitignored; files have never been in any commit. Filesystem birth timestamps 2026-04-30 16:46–17:10 align with `tomato_log.md` `[2026-04-30 17:15]` encoder dispatch entry. No modification timestamps after Phase 3 completion log entry.

## LOW concern detail

The `# noqa: E402` on the `assign_tier` import line in all 13 test files suppresses a flake8 warning that would not fire under standard rules in this file structure. The comment has no effect on test pass/fail outcomes, no effect on pytest collection, and no effect on the import error mode. Suggest removing during T-EARLY-MP cleanup or documenting the intent (encoder may have added it defensively against non-standard linters).

## Final verdict: PASS with concerns

One LOW concern. No HIGH or MEDIUM violations. Phase 3 deliverable is honest: tests genuinely fail, implementation absent, spec citations accurate, body-wins decisions match spec, manifest evolution audit-trailed.
