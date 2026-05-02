# Anti-Cheat Scan — Phase 4 Batch 4 (T-IMPL-4a + T-IMPL-4b)

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-02
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS clean.** 0 HIGH, 0 MEDIUM, 1 LOW informational.

## Summary table

| # | Check | Severity | Verdict |
|---|---|---|---|
| 1 | Section 15 test files modified (LF-normalized SHA256 vs DEC-032 baseline) | HIGH | PASS — all 13 hashes match exactly |
| 2 | `.git/hooks/pre-commit` md5 unchanged | HIGH | PASS — `24eb46f308751df3a125faca0680c9c7` |
| 3 | DEC-038 compliance: no implementer commits since `84cbdb0` | HIGH | PASS — `git log 84cbdb0..HEAD` empty |
| 4 | Suppressed failures (xfail / skipif(True) / bare except) | HIGH | PASS — only one `# noqa: S301` on pickle.load with inline rationale (justified) |
| 5 | Spec citations on 6 spot-checked literals | MEDIUM | PASS — α=0.10 cites S13.2:3538; N=40 cites S13.2:3538 + S13.3:3557; ClassifierResult fields cite S12.10:3449-3457; 19-dim feature vector cites S12.2:3177-3189; NUM_FINAL_CLASSES=7 cites S12.1; conformal quantile formula cites S13.5:3594 verbatim |
| 6 | No `print()` in production | HIGH | PASS — zero `print(` in classifier or conformal |
| 7 | No APIN imports | HIGH | PASS |
| 8 | No `gpu_lock` import in classifier or conformal | HIGH | PASS — both modules CPU-only post-signals |
| 9 | Honest test counts: 48 + 44 = 92 new; cumulative 630 | MEDIUM | PASS — pytest --collect-only confirms |
| 10 | DEC-039 / DEC-040 sequential, no collisions | MEDIUM | PASS — pre-allocation rule honored |
| 11 | **NEW:** ClassifierResult 9-field spec compliance | HIGH | PASS — all 9 spec field names present (`p_final_calibrated`, `combined_argmax`, `combined_max_prob`, `combined_margin`, `p_final_uncalibrated`, `p_stage1`, `p_stage2`, `classifier_succeeded`, `failure_reason`); each field cites its specific S12.10 spec line. Notable: T-IMPL-4a's DEC-039 documents the discovery that spec defines 9 fields (not the 6 in user dispatch prompt, which was based on incomplete BLK-010.2). Spec wins per DEC-018. |
| 12 | **NEW:** Conformal consumes `p_final_calibrated` (spec-pinned, no aliasing) | HIGH | PASS — `compute_conformal_set(p_final_calibrated: np.ndarray, ...)` cites S12.10:3448-3449 |
| 13 | **NEW:** No paraphrase drift on α=0.10 / N=40 / 7-class | HIGH | PASS |
| 14 | **NEW:** `tier/` absent; Section 15 still failing on ModuleNotFoundError | HIGH | PASS — 13 collection errors confirmed |
| 15 | **NEW:** DEC-033 sub-package compliance — both import paths work | MEDIUM | PASS — both `from tomato_sandbox.classifier import ...` and `from tomato_sandbox.classifier.hierarchical_classifier import ...` resolve to same symbols; same for conformal; verified by `test_import_path_package` and `test_import_path_submodule` |
| 16 | Hardcoded test values | HIGH | PASS — assertion values mathematically derivable (e.g. `τ = 0.925` for N=40, α=0.10 via `ceil(41×0.9)/40 = 37/40`) |
| 17 | Fake completion claims | MEDIUM | PASS — DEC entries cite actual files; all listed files exist on disk |

## LOW informational note (1)

**LOW-1:** `tomato_sandbox/classifier/hierarchical_classifier.py` line 190 contains `# noqa: S301 — trusted internal calibration file` on a `pickle.load(f)` call for `classifier_stage1.pkl` and `classifier_stage2.pkl` from `phase_f0_calibration/`. S301 is a security-linting rule about pickle deserialization risk; the inline justification ("trusted internal calibration file") is accurate (these are first-party calibration artifacts loaded read-only). Not a failure-suppression pattern. **Severity: LOW (informational).**

## Process wins this batch

- **DEC-038 worked first try.** Zero implementer-driven commits despite both implementers running with full Bash access. The `.claude/agents/implementer.md` Rule 12 edit propagated immediately.
- **Pre-allocation rule scaled.** DEC-039 and DEC-040 allocated, no collision. Three batches in a row now (Batch 2/3/4) using pre-allocation cleanly.
- **Two-parallel safe here** because spec-pinned dataclass field name (`p_final_calibrated` per S12.10:3449) gives both implementers the same contract source — no need for two-wave (Batch 3 pattern was needed because TTA depended on actual function signatures, which weren't spec-pinned).
- **Implementer caught spec discovery:** T-IMPL-4a noticed user's dispatch prompt listed 6 ClassifierResult fields (from incomplete BLK-010.2), but spec body defines 9. Implementer extended to all 9 per DEC-018 (spec wins). Anti-cheat verified all 9 are present and individually cited.

## Carried-forward LOW concerns (re-noted, not new)

- 56 Pillow `getdata()` deprecation warnings (cosmetic; queued).
- DEC-022..040 pre-code-logging timing unverifiable (Critical Rule 9).
- One `# noqa: F401` in conformal `__init__.py` for re-export shim (DEC-033 pattern, intentional).

## Recommendation

Phase 4 Batch 4 is clean. Proceed to checkpoint 004 and main-thread commit per DEC-038.
