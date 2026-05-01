# Anti-Cheat Scan — Phase 4 Batch 2 (T-IMPL-2a/2b/2c)

**Inspector:** anti-cheat-inspector
**Date:** 2026-05-01
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS clean.** 0 HIGH, 0 MEDIUM, 3 LOW (all cosmetic / carry-forward / process).

## Summary table

| # | Check | Severity | Verdict | Evidence headline |
|---|---|---|---|---|
| 1 | Section 15 test modification | HIGH | PASS | All 13 files unchanged; `tomato_sandbox/tier/` absent; `ModuleNotFoundError: No module named 'tomato_sandbox.tier'` confirmed live. Byte sizes recorded for future baseline. |
| 2 | `.git/hooks/pre-commit` unchanged | HIGH | PASS | md5 `24eb46f308751df3a125faca0680c9c7` matches Phase 3 baseline. |
| 3 | Suppressed failures | HIGH | PASS (LOW note) | Zero `xfail` / unconditional skips. `pytestmark = skipif(not _TORCH_AVAILABLE / not _PIL_AVAILABLE)` in test_preprocess.py is a legit env-dep guard. One conditional `pytest.skip` in test_validate_input.py line 506 fires only on a JPEG codec edge case. No bare `except:`. |
| 4 | Spec citations (5 spot checks) | MEDIUM | PASS | All 5 literals (IMAGE_COUNT_MAX=5 → 5.2:936-937; _AGG_REJECT=0.40 → 6.4:1324; sharpness BAD_THRESHOLD=0.20 → 6.4:1305; LORA_PAD_VALUE=114 → 7.2:1430-1431; _PSV_MAX_SIDE=1200 → 7.4:1519) verified verbatim against spec body. |
| 5 | No `print()` in production | HIGH | PASS | Zero `print(` in input_validation.py, iqa/iqa.py, preprocess.py. The string "print" appears only in spec-quote comments (e.g. "never print() per S26.7"). |
| 6 | Port 8767 / no APIN imports | HIGH | PASS | No port 8766/8005 references; no `apin` / `section2d_psv` imports in any Batch 2 module. |
| 7 | Honest test counts | MEDIUM | PASS (LOW note) | `pytest --collect-only` reports 415; `grep -c "^def test_"` confirms 96 + 82 + 61 = 239 Batch 2 tests; 415 − 239 = 176 prior baseline. Math correct. test_preprocess.py 61 tests are env-dependent (torch+Pillow). |
| 8 | DEC numbering — no collisions | MEDIUM | PASS | DEC-029, DEC-030, DEC-031 sequential, no gaps in Batch-2 range, no renumbering required. **Pre-allocation rule honored** — each implementer used its assigned number. (DEC-016 is a pre-existing inline reference, outside Batch 2 scope.) |
| 9 | T-IMPL-2a path divergence documented | MEDIUM | PASS | DEC-029 Decision 1 explicitly cites: spec 5.7:1049 says `tomato_sandbox/input_validation.py`; task card said `tomato_sandbox/api/validate_input.py`. Resolution: canonical at the spec path; re-export shim at the task-card path; `TestReExportPath::test_api_path_validate_request_works` covers both. |
| 10 | T-IMPL-2c config.py additions spec-cited | MEDIUM | PASS | 8 constants added to config.py lines 170-205 under heading "spec: section 7.2 lines 1421-1432". All values match spec verbatim. Additive only — no modification of existing TomatoConfig fields. |
| 11 | `tier/` absent; Section 15 still fails | HIGH | PASS | `ls tomato_sandbox/tier/` → does not exist. Live pytest fires `ModuleNotFoundError` on import line. 13 collection errors confirmed. |

## LOW findings (3, cosmetic/process — none Phase-4-blocking)

### LOW-1: Environment-dependent skips in test_preprocess.py
`pytestmark = [skipif(not _TORCH_AVAILABLE), skipif(not _PIL_AVAILABLE)]` wraps all 61 tests. Legitimate optional-dep guard, but in a torch/Pillow-less environment all 61 would silently skip. Reviewers should confirm CI env has both. T-EARLY-MP queue candidate (low priority).

### LOW-2: Conditional pytest.skip in test_validate_input.py:506
Fires inside `if len(data) >= FILE_SIZE_MIN_BYTES:` branch — handles platform-dependent JPEG codec sizes. Not a cheat; environmental edge case.

### LOW-3: Git tracking inconsistency in Batch 2
`.gitignore` has `tomato*/` rule (lines 10-11). T-IMPL-2b's implementer subagent appears to have force-added IQA files (commit 69d8ce7 contains `tomato_sandbox/iqa/*` and `test_iqa.py`), while T-IMPL-2a (input_validation.py, validate_input.py, test_validate_input.py) and T-IMPL-2c (preprocessing/, test_preprocess.py, config.py additions) remain untracked. Creates uneven provenance for Batch 2 deliverables. **Recommend T-EARLY-MP queue:** decide whether tomato_sandbox/ is gitignored (current rule) or tracked (IQA precedent) and unify. Severity: LOW — not a cheating indicator; the on-disk content is spec-compliant.

## Carried-forward LOW concerns (re-noted, not new)

- DEC-022..025 / DEC-029..031 pre-code-logging timing unverifiable (Critical Rule 9; substantive content honest).
- Inline spec citations on assertion lines (vs docstrings/test names) cosmetic gap.
- 3 legit `# noqa` in `logging.py` and `# noqa: F401` in test_iqa.py (re-export verification imports).

## Recommendation

Phase 4 Batch 2 is clean. Proceed to checkpoint and STOP per master prompt cadence.
