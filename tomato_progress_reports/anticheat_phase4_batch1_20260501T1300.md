# Anti-Cheat Scan — Phase 4 Batch 1 (T-IMPL-1a/1b/1c)

**Inspector:** anti-cheat-inspector
**Date:** 2026-05-01
**Saved by:** main-thread scribe per DEC-011 (agent returned findings as text)

## Verdict: **PASS clean.** No HIGH or MEDIUM violations.

## Summary table

| # | Check | Severity target | Verdict | Evidence headline |
|---|---|---|---|---|
| 1 | Section 15 test modification | HIGH | PASS | `test_section15_tier1.py` byte size 14,431 = Phase 3 baseline. May 1 mtime explained by Phase 3 hook-verification restore (single newline appended → blocked → trimmed). No content change. `tomato_sandbox/tier/` still absent. |
| 2 | `.git/hooks/pre-commit` unchanged | HIGH | PASS | md5 `24eb46f308751df3a125faca0680c9c7` matches Phase 3 baseline byte-for-byte. Mode `-rwxr-xr-x`. T-IMPL-1c only created `.pre-commit-config.yaml` (separate framework config), did not touch the bash hook. |
| 3 | Suppressed failures | HIGH | PASS | Zero `pytest.skip`/`xfail`/`mark.skip`/`mark.skipif` in T-IMPL-1a/1b test files. Zero bare `except:` or empty except blocks. Server.py has typed `except ImportError` (optional torch) and `except Exception` (GPU init); both have non-empty bodies with logging. |
| 4 | Spec citations (3 spot-checks) | MEDIUM | PASS | `sacred_guard.py` cites manifest `directory_hash_algorithm_canonical` + DEC-019. `server.py` module docstring cites `# spec: section 20.3 lines 6452-6499`, `20.5 lines 6556-6575`, `20.6 lines 6577-6589`, `20.7 lines 6591-6603`. `pyproject.toml` ruff section: `# Per spec Section 26.6 lines 7747-7748`. All include section + line numbers. |
| 5 | No `print()` in production | HIGH | PASS | Zero `print(` calls in `sacred_guard.py`, `server.py`, `config.py`. The string "print" appears only in spec-quote comments referencing the no-print rule. |
| 6 | Port 8767 enforcement (BLK-002 / DEC-012) | HIGH | PASS | `config/default.yaml` line 19: `port: 8767` with BLK-002/DEC-012 inline comment. `server.py` module docstring: `Sandbox server port: 8767 (BLK-002 / DEC-012 / DEC-026)`. `8766` appears only as informational reference to APIN. |
| 7 | No APIN library import (BLK-003 / DEC-012) | HIGH | PASS | `server.py` imports: asyncio, contextlib, typing, fastapi, tomato_sandbox.config, tomato_sandbox.utils.gpu_lock, tomato_sandbox.utils.logging. Zero `scripts.apin`/`apin`/`section2d_psv` imports. `test_smoke_no_apin_import` test (line 477) asserts via `sys.modules` inspection. |
| 8 | Sacred-guard algorithm correctness | HIGH | PASS | `sacred_guard.py` 7-step canonical algorithm matches manifest `directory_hash_algorithm_canonical.pseudocode` verbatim: `fnmatch` on basename, `os.path.relpath` with forward-slash, `json.dumps(sort_keys=True, separators=(",", ":"))`, `sha256(...).hexdigest()`. DEC-019 `log_exclusions` extension implemented. **Independent verification:** main-thread Python implementation and in-sandbox `verify_manifest()` both return 10/10 PASS — two implementations of the canonical algorithm agree. |
| 9 | DEC-028 renumbering honesty | MEDIUM | PASS | `tomato_decisions.md` line 427 has full `[RENUMBERED 2026-05-01 from DEC-026 → DEC-028]` annotation explaining the parallel-dispatch race. Single DEC-026 confirmed (only one entry). DEC-027 unchanged. T-EARLY-MP defect queued (subagent-coordination on append-only logs). |
| 10 | Test count math (176 total) | MEDIUM | PASS | `test_sacred_guard.py` 30 + `test_server_skeleton.py` 43 + `test_degraded_mode.py` 29 + `test_gpu_lock.py` 18 + `test_logging.py` 22 + `test_nan_guards.py` 34 = **176**. Independently verified by main-thread `pytest tomato_sandbox/tests/unit/`: 176 passed in 2.31s. |
| 11 | `tier/` absent; Section 15 still fails | HIGH | PASS | `ls tomato_sandbox/tier/` → does not exist. `pytest tomato_sandbox/tests/integration/ --collect-only` → 13 ERROR (`ModuleNotFoundError: No module named 'tomato_sandbox.tier'`). Phase 4 has not prematurely implemented `tier_assignment.py`. |

## Carried-forward LOW concerns from Batch 0 (re-noted, not re-violations)

- LOW: Inline spec citations missing on constant-equality assertions in utility test files. Cosmetic; T-EARLY-MP queue.
- LOW: 3 legitimate `# noqa` suppressions in `logging.py` (structural Python constraints — conditional structlog import + `Formatter.format` builtin shadow).
- LOW: DEC-022..025 timing unverifiable (pre-code-logging Critical Rule 9 claim; substantive honesty intact).

## Carried-forward structural note

`.gitignore` has `tomato*/` rule — `git log --follow` permanently unavailable for `tomato_sandbox/` files. Pre-commit bash hook is the enforcement mechanism. Byte-size comparison + module-still-absent are the verification fallbacks. Pre-existing project-level decision.

## Recommendation

Phase 4 Batch 1 is clean. Continue to Step 9 (`phase_4_checkpoint_001.md`) and STOP per the user's first-session plan.
