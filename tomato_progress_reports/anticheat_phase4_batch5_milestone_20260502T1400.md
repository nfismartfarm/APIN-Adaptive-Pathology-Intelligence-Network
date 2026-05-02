# Anti-Cheat Scan — Phase 4 Batch 5 (T-IMPL-5 Tier Assignment) MILESTONE

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-02
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS — milestone verified genuine.** 0 HIGH, 1 MEDIUM (documentation artifact), 1 LOW (documentation artifact).

The 135/135 Section 15 result is real. Evidence summary:
- All 13 Section 15 SHA256 hashes (LF-normalized) match DEC-032 baseline exactly.
- Live `pytest` confirms 135 PASSED, 0 SKIP, 0 XFAIL, 0 ERROR, 0 FAIL.
- Implementation at `tomato_sandbox/tier/tier_assignment.py` (25,546 bytes) honors the import contract.
- 3 BLK-011 sub-defects are real spec contradictions resolved with scenario-body authority per BLK-004 precedent.
- 88 unit tests + 135 integration tests = 223 new tests; cumulative 718 unit + 135 integration = 853 total.
- No suppression patterns, no trivial assertions, no fabricated bypasses.

## All 17 milestone-strength checks

| # | Check | Severity Cat. | Verdict |
|---|---|---|---|
| 1 | Section 15 LF-SHA256 vs DEC-032 baseline (13 files) | HIGH | PASS — all 13 match |
| 2 | Pre-commit hook md5 unchanged | HIGH | PASS — `24eb46f308751df3a125faca0680c9c7` |
| 3 | No test gaming (assert True / try-except / shallow assertions) | HIGH | PASS — all 135 have substantive `tier_label == "X"` style assertions; tier_label distribution: 4A=30, 3A=20, 2=20, 1=16, 3C=14, 4B=13, 3D=11, 3B=11; rule_id_fired covers all 12 contract values |
| 4 | No silent skips (live pytest count) | HIGH | PASS — 135 passed, 0 SKIP, 0 XFAIL, 0 ERROR |
| 5 | rule_id_fired enumeration (only 12 contract values) | MEDIUM | PASS — no fabricated `"2"` (Rule 2 unreachable by design, line 378-382); no `"9"`; Rule 9 correctly uses `"catch_all_low_confidence"` |
| 6 | 6 threshold inequalities match contract (strict vs inclusive) | MEDIUM | PASS — all 6 spot-checks correct, each with inline `# spec: section 14.X lines NNNN` citation |
| 7 | BLK-011 sub-defects honest | HIGH | PASS — all 3 sub-defects (Rule 4 before Rule 3, Rule 4 size=2 bypass, PSV in T5 in-set) verified with spec line cross-references and matching test scenarios |
| 8 | Honest test counts (88 / 135 / 718) | MEDIUM | PASS — all 4 counts verified by collect-only |
| 9 | DEC-041 cites real spec line ranges (5 spot-checks) | MEDIUM | PASS — all 5 cited ranges contain expected content (T5 conditions at 14.3:3784-3800, rule chain at 14.5:3818-3879, etc.) |
| 10 | DEC-038 compliance (no commits since `4af9fc5`) | HIGH | PASS — `git log 4af9fc5..HEAD` empty |
| 11 | Sacred files unchanged | HIGH | PASS — manifest 10 entries; no sacred files in T-IMPL-5 modification list |
| 12 | No `print()` in production | HIGH | PASS |
| 13 | No APIN imports in tier | HIGH | PASS — only `math, dataclasses, typing, tomato_sandbox.utils.logging` imported |
| 14 | No `gpu_lock` in tier | HIGH | PASS — pure CPU dispatch logic |
| 15 | No fabricated bypasses outside BLK-011 | HIGH | PASS — only one non-spec conditional (Rule 4 bypass `_genuine_two_class`) and it's BLK-011 sub-defect 11.2 + DEC-041 documented |
| 16 | Bug-fix unit tests cross-check BLK-011 | HIGH | PASS — all 4 named tests (`test_rule4_overrides_rule3_when_max_low`, etc.) assert behavior consistent with BLK-011 sub-defects |
| 17 | No back-derived test values | MEDIUM | PASS — assertion values clearly threshold-table-derived (max=0.91 above 0.85, max=0.80 between 0.65 and 0.85, etc.); no values that could only come from running the implementation |

## Findings (all documentation artifacts)

### MEDIUM-1 — BLK-011 sub-defect 11.2 prose contains obsolete intermediate hypothesis

- **Location:** `tomato_blockers.md` lines 311-312
- **Description:** BLK-011 sub-defect 11.2 describes the bypass condition as "size=2 AND margin > 0.0". This was the implementer's initial wrong hypothesis. The actual implementation (and DEC-041 Decision 2) uses `size=2 AND max >= 0.41`. The `margin > 0.0` formulation would have caused S4A.4 (max=0.40, size=2, margin=0.10) to fail because Rule 4 should fire there but the bypass would prevent it.
- **Impact:** Documentation artifact. Implementation is correct; tests pass; DEC-041 documents the correct condition. But a reader of BLK-011 alone would inherit the wrong understanding.
- **Fix:** annotate BLK-011 sub-defect 11.2 with a correction note pointing to DEC-041 Decision 2.
- **Severity:** MEDIUM (documentation correctness, no code impact).

### LOW-1 — `.claude/import_contract.md` rule priority not updated post-BLK-011

- **Location:** `.claude/import_contract.md` "Overall rule priority" line
- **Description:** Contract still says `Rule 1 > Rule 3 > Rule 4 > Rule 5 > Rule 6 > Rule 7 > Rule 8 > Rule 9`. Per BLK-011 sub-defect 11.1 + DEC-041 Decision 1, the implementation evaluates Rule 4 BEFORE Rule 3 (scenario-body authority over header). Contract was not updated.
- **Impact:** Documentation artifact. Section 15 tests are themselves authoritative for rule ordering; they pass.
- **Fix:** update import contract priority to `Rule 1 > Rule 4 > Rule 3 > Rule 5 > Rule 6 > Rule 7 > Rule 8 > Rule 9` with cross-reference to BLK-011 / DEC-041.
- **Severity:** LOW.

## Recommendation

Phase 4 Batch 5 is **clean and milestone-validated**. Both documentation findings should be addressed at the close-out commit so durable record matches code reality.

**The 135/135 result is genuine. Phase 4 implementation is functionally complete on the integration test gate.**
