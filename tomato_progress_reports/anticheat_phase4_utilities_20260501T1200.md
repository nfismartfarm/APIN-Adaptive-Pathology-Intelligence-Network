# Anti-Cheat Scan — Phase 4 Batch 0 (4 utility modules)

**Inspector:** anti-cheat-inspector
**Date:** 2026-05-01
**Saved by:** main-thread scribe per DEC-011 (agent returned findings as text)

## Verdict: PASS with 3 LOW concerns. No HIGH severity violations.

## Findings table

| # | Check | Severity target | Verdict | Detail |
|---|---|---|---|---|
| 1 | Section 15 test modification | HIGH | PASS (provenance caveat) | Byte sizes match Phase 3 baseline (`test_section15_tier1.py` = 14,431 B as documented in `phase_3_exit_audit_refire_20260501T1145.md`); `tomato_sandbox/tier/` still absent; no DEC entry approving modification; git log unavailable structurally (`.gitignore` has `tomato*/`). Structural limit, not a Phase 4 violation. |
| 2 | Hardcoded test values | HIGH | LOW concern | 3 constant-equality assertions (`TTA_TRIGGER_THRESHOLD == 0.55`, `TTA_ESCALATE_THRESHOLD == 0.45`, `VECTOR_DIM == 19`) verify spec-mandated values, not magic numbers. Spec-sourced (Section 11.2 + Section 12.2). LOW concern: missing inline `# from spec: <section> line <N>` annotation beside the literals — docstrings cite spec, but the assertion lines don't. |
| 3 | Suppressed failures | HIGH | PASS | Zero `pytest.skip`/`xfail`/`mark.skip`/`mark.skipif`; zero bare `except:`; zero empty except; zero `noqa` in test files. All except clauses typed: `ImportError`, `(ValueError, TypeError)`, `asyncio.TimeoutError`. |
| 4 | `noqa` in production | LOW | LOW concern (3 legitimate) | `logging.py` lines 66 + 167: `# noqa: PLC0415` (import-outside-toplevel; forced by `if _STRUCTLOG_AVAILABLE:` conditional-import pattern). Line 96: `# noqa: A003` (`format` shadows builtin; mandated by `logging.Formatter` API). All legitimate. |
| 5 | `print()` in production | HIGH | PASS | Zero executable `print(...)` calls. Two occurrences inside docstring/comment strings only. |
| 6 | Fabricated dataclass fields | HIGH | PASS (N/A) | No `@dataclass` in any utility module. Public APIs are functions + constants + 1 class (`GPULock`) + 1 exception (`GPULockTimeoutError`). No opportunity for field fabrication. |
| 7 | Public API vs spec (spot-check 2) | MEDIUM | PASS | `nan_guards.py` matches Section 11 (TTA thresholds 0.55/0.45; `tta_n_views` returns 1/2/5; `aggregate_views` returns `(zeros, 0)` on total failure). `gpu_lock.py` matches Section 20.6 (default 10s timeout; env var `TOMATO_GPU_LOCK_TIMEOUT_S`; asyncio.Lock + FIFO; SERVER_OVERLOAD on timeout). |
| 8 | DEC-022..025 honesty | MEDIUM | LOW concern | Substantive content honest. Each entry has spec section + verbatim quote + numbered "What we implemented" + rationale + impact + approval reference. Approval chain: "implicit per DEC-021 scope" — DEC-021 has explicit user quote naming all 4 modules. **LOW concern:** pre-code-logging timing (Critical Rule 9) cannot be verified from filesystem timestamps; no evidence of back-filing, but also no evidence ruling it out. |
| 9 | Test count reconciliation | MEDIUM | PASS | Actual count 22 + 18 + 34 + 29 = 103. Prompt's per-file counts (20/16/28/29) were underestimates; implementer produced more tests than minimum requested. Final count claim of 103 is correct. |

## LOW concerns (3)

1. **Inline spec citations on constant assertions** — `assert TTA_TRIGGER_THRESHOLD == 0.55` should carry `# from spec: 11.2 line ~2932` annotation alongside the literal. Currently spec citation lives in the docstring only. Cosmetic; not blocking. T-EARLY-MP cleanup candidate.
2. **`# noqa` legitimate suppressions in `logging.py`** — 3 instances; all justified by structural Python constraints (conditional imports + builtin name shadow on `format()`). Same accepted-LOW category as Phase 3's `# noqa: E402` on Section 15 imports.
3. **DEC-022..025 timing unverifiable** — implicit approval chain anchors to DEC-021's explicit user approval. Substantive honesty intact; only the pre-code-logging timing claim is unverifiable.

## Structural note

`.gitignore` has `tomato*/` rule. This permanently prevents `git log --follow` provenance checks for any file under `tomato_sandbox/`. It's a project-level infrastructure choice predating Phase 4. The pre-commit hook (verified functional during Phase 3) is the enforcement mechanism for Section 15 test immutability. Byte-size comparison + behavioral checks (module-still-absent) are the verification fallbacks.

## Recommendation

Accept Phase 4 Batch 0 utilities as clean. Add the 3 LOW concerns to T-EARLY-MP queue for follow-up. Continue with Batch 1 (T-IMPL-1a/1b/1c).
