# Anti-Cheat Scan — Phase 6 Component A (T-PHASE6-A / DEC-053)

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-04
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS clean. ALL 16 CHECKS CLEAR. 0 HIGH, 0 MEDIUM, 0 LOW.**

Second consecutive perfectly-clean dispatch in Phase 6 (matches Component B). The architectural separation per (β) interpretation + spec-pinned input contracts continues to produce predictable, low-risk implementations.

## Files inspected

- `tomato_sandbox/validation/run_f0.py` (31,625 B, 445 lines)
- `tomato_sandbox/validation/__init__.py` (1,160 B; UPDATED — was 941 B at Component B close)
- `tomato_sandbox/tests/unit/test_run_f0.py` (30,247 B, 42 tests)
- `tomato_decisions.md` (DEC-053 entry at line 1472)

## Summary table

| Check | Severity threshold | Outcome |
|---|---|---|
| 1. Section 15 immutability | HIGH | CLEAR — 13/13 files match; single commit `a926d3d`; 135/135 PASS live |
| 2. Pre-commit hook md5 | HIGH | CLEAR — `24eb46f308751df3a125faca0680c9c7` |
| 3. Suppressed failures | HIGH | CLEAR — zero unconditional skips/xfails/bare excepts; one justified `# noqa: F401` for mock-patchability |
| 4. Spec citations | MEDIUM | CLEAR — S29:8105-8243, S13.6:3621-3635, S17.3:5966-5982, S17.5:6015-6033, S16.2:5655-5712 all present |
| 5. No `print()` in production | LOW | CLEAR — zero |
| 6. No APIN imports | HIGH | CLEAR |
| 7. DEC-038 — no commits since `7e5e2f5` | HIGH | CLEAR |
| 8. Honest test count (42) | MEDIUM | CLEAR — 42 collected, 42 pass |
| 9A. No model checkpoint loading | HIGH | CLEAR — zero torch/checkpoint refs; (β) interpretation honored |
| 9B. labeled_data_path parameter | MEDIUM | CLEAR — first arg of `run_f0_validation`; zero hardcoded data paths |
| 9C. Tests use tmp_path | MEDIUM | CLEAR — zero writes to production calibration dir |
| 9D. predict_single import pattern | MEDIUM | CLEAR — module-level try/except ImportError; mock-patchable; failure mode auditable |
| 9E. **Tier 4B disposition distinguishes degraded vs real-failure** | **HIGH** | **CLEAR** — `_compute_tier_disposition` reads `rule_id_fired` from `explanation.structured`; 4 buckets tracked (`tier_4b_count_total`, `tier_4b_count_degraded`, `tier_4b_count_real_failure`, `is_pre_f0_mode`) |
| 9F. Conformal coverage formula | HIGH | CLEAR — `n_covered / n_total` exactly per S13.6; unknown-class indices explicitly skipped (not silently counted as failures); Wilson CI informational additive |
| 9G. Severity skip semantics | MEDIUM | CLEAR — `status: "skipped"`, `reason: "skipped_no_ground_truth"` explicit; tests assert exact strings |
| 9H. Calibration dir read-back pattern | MEDIUM | CLEAR — placeholder values surfaced in `metadata.calibration_artifacts`; missing-file → `"not_found"`; malformed JSON → `{"error": ...}` |

## Notable findings (positive)

### Spec citations — comprehensive
File-level docstring (lines 15-21) cites the FIVE primary contracts the script honors. Inline citations at every threshold/formula. No paraphrase drift detected.

### Tier 4B disposition logic
The 5-line spec contract from the dispatch ("distinguish Tier 4B from degraded vs real signal failure") is implemented as a 4-counter system in `_compute_tier_disposition`:
- `tier_4b_count_total` — total Tier 4B responses
- `tier_4b_count_degraded` — Tier 4B with `rule_id_fired in {"1", "pipeline_failure"}`
- `tier_4b_count_real_failure` — Tier 4B with any other rule_id
- `is_pre_f0_mode` — True when `tier_4b_total == n_total AND tier_4b_degraded == n_total`

This lets Phase F.0 dry-run (post-Component-C) immediately spot the transition: when `is_pre_f0_mode` flips False, real signals are working.

### Conformal formula honesty
`n_covered / n_total` matches spec S13.6 verbatim. Unknown-class indices (`y_true < 0`) are explicitly excluded from the denominator with a comment "unknown true class; skip". Empty prediction set contributes 0 to numerator (string lookup in empty list fails). Full prediction set with matching class contributes 1. No edge-case shortcuts.

### Calibration read-back
Placeholder values from current `conformal_tau.json` (τ=0.42) and `psv_standardization.json` (identity, T_PSV=1.0) are surfaced in `metadata.calibration_artifacts`. Tests 26-28 verify all 3 cases: placeholder file present, no files, malformed JSON.

### `# noqa: F401` justification
Single occurrence at line 50 — module-level `predict_single` import. Comment documents that the import is required at module level so `unittest.mock.patch("tomato_sandbox.validation.run_f0.predict_single", ...)` resolves correctly. This is standard, legitimate Python pattern; not failure suppression.

## Carry-forward (informational)

None. The auditor noted DEC-032 records the pre-commit hook md5 + sacred manifest state but does NOT record per-file Section 15 SHA256 hashes. Per Phase 5c LOW-1, this is a known gap deferred to T-EARLY-MP. The hashes captured in Phase 5c (and re-verified here) serve as the forward baseline.

## Recommendation

Component A is clean. Phase 6 Component C (real model loading lift, DEC-054) can dispatch on user approval.

State preserved:
- Sacred 10/10 PASS
- Section 15: 135/135
- 1239 + 1 skip cumulative under venv
- DEC-038 compliance (zero implementer commits since `7e5e2f5`)
- (β) interpretation respected
