# Phase 4 Checkpoint 001 — First-session work complete

**Date:** 2026-05-01
**Cadence trigger:** master prompt every-3-modules cadence (7 modules + tests done this session — well past).
**Session scope:** Phase 4 Batch 0 (4 utility modules, master-prompt-mandated FIRST per DEC-021) + Batch 1 (T-IMPL-1a/1b/1c per `tomato_plan.md`).
**Verdict:** **Session complete. STOP and await user direction on Batch 2.**

---

## 1. Modules implemented this session

### Batch 0 — 4 cross-cutting utility modules (per master prompt Section 4 Phase 4 + DEC-021)

| Module | Path | Bytes | Spec section | DEC |
|---|---|---|---|---|
| Structured logging | `tomato_sandbox/utils/logging.py` | 8,298 | 26.7 | DEC-022 |
| GPU lock | `tomato_sandbox/utils/gpu_lock.py` | 6,862 | 20.6 | DEC-023 |
| NaN guards | `tomato_sandbox/utils/nan_guards.py` | 8,640 | 11 (TTA + signals) | DEC-024 |
| Degraded mode | `tomato_sandbox/utils/degraded_mode.py` | 7,517 | 12.7 + 12.2 | DEC-025 |

Plus `tomato_sandbox/utils/__init__.py`.

### Batch 1 — 3 parallel-dispatched modules (per `tomato_plan.md`)

| Task | Path(s) | Bytes | Spec section | DEC |
|---|---|---|---|---|
| T-IMPL-1a | `tomato_sandbox/utils/sacred_guard.py` | 8,975 | 2 (sacred manifest) + DEC-019 | DEC-028 (renumbered from collision) |
| T-IMPL-1b | `tomato_sandbox/api/server.py` (14,467) + `tomato_sandbox/api/__init__.py` (168) + `tomato_sandbox/config.py` (5,933) + `tomato_sandbox/config/default.yaml` (1,419) | 21,987 total | 20 | DEC-026 |
| T-IMPL-1c | `pyproject.toml` (5,358 — appended to existing) + `.pre-commit-config.yaml` (4,570) | 9,928 | 26 (engineering hygiene) | DEC-027 |

## 2. Tests added

| Test file | Tests | Status |
|---|---|---|
| `test_logging.py` | 22 | PASS |
| `test_gpu_lock.py` | 18 | PASS |
| `test_nan_guards.py` | 34 | PASS |
| `test_degraded_mode.py` | 29 | PASS |
| `test_sacred_guard.py` | 30 | PASS |
| `test_server_skeleton.py` | 43 | PASS |

**Total: 176 unit tests passing** (verified by independent `pytest tomato_sandbox/tests/unit/` run: `176 passed in 2.31s`).

Section 15 integration tests: still 13 collection errors with `ModuleNotFoundError: No module named 'tomato_sandbox.tier'` — expected; Phase 4 has not yet implemented `tier_assignment.py`.

## 3. Audit verdicts

| Audit | Verdict | Notes |
|---|---|---|
| Sacred (main-thread independent canonical hash) | **10/10 PASS** | Run twice (post-Batch-0, post-Batch-1) |
| Sacred (in-sandbox `tomato_sandbox.utils.sacred_guard.verify_manifest`) | **10/10 PASS** | Cross-validates main-thread implementation |
| Anti-cheat (Batch 0 utilities) | **PASS with 3 LOW** | Cosmetic noqa, pre-code-logging timing unverifiable, missing inline spec citations on constant assertions |
| Anti-cheat (Batch 1 T-IMPL-1a/1b/1c) | **PASS clean** | No HIGH or MEDIUM violations. All 11 checks PASS. |

## 4. Decisions logged this session

| DEC | Title | Trigger |
|---|---|---|
| DEC-021 | Phase 4 Batch 1 ordering — master prompt and plan have different first-batch compositions; master prompt is authoritative | User-approved deviation before any code |
| DEC-022 | logging.py: structlog with stdlib fallback; stdout JSON; no print() in production | Batch 0 module |
| DEC-023 | gpu_lock.py: asyncio.Lock with timeout; SERVER_OVERLOAD on timeout | Batch 0 module |
| DEC-024 | nan_guards.py: guard functions for TTA + signal forward passes; finiteness checks | Batch 0 module |
| DEC-025 | degraded_mode.py: zero-fill helpers for failed signal blocks in 19-dim vector | Batch 0 module |
| DEC-026 | FastAPI server skeleton (T-IMPL-1b): port 8767, stub endpoints, lifespan startup, config hierarchy | Batch 1 |
| DEC-027 | Lint/test scaffold: ruff + mypy strict + black line-length 100; pre-commit framework config; rule set rationale | Batch 1 |
| DEC-028 | sacred_guard.py: project-root anchored paths; manifest loaded on each call; optional path override for testability **[renumbered from duplicate DEC-026 due to parallel-dispatch race]** | Batch 1 |

## 5. Open issues / surfaced findings

### Process finding — parallel implementer dispatch race on append-only ledger

T-IMPL-1a (sacred_guard) and T-IMPL-1b (server skeleton) ran in parallel via the Agent tool. Both subagents independently observed DEC-025 as the latest entry and each grabbed DEC-026 for their own decision. T-IMPL-1b's entry was written first to disk; T-IMPL-1a's was renumbered to DEC-028 by main-thread scribe. **Suggested T-EARLY-MP defect:** when dispatching multiple implementer subagents in parallel, document a serialization point on the decisions ledger — e.g., main thread reserves DEC numbers up-front and tells each subagent which number to use. Or: have implementers return their DEC entry *content* and let main thread allocate numbers and append. Severity: MEDIUM (caught and corrected; would silently produce duplicate IDs without verification).

### Side-fix during T-IMPL-1a — `logging.py` patched

T-IMPL-1a noticed that `tomato_sandbox/utils/logging.py` (Batch 0 module) had `structlog.stdlib.add_logger_name` in its processor chain, but that processor calls `logger.name` which doesn't exist on `PrintLogger` (structlog's stdout backend). The fix was to remove that processor from the chain. This is spec-compliant — Section 26.7 doesn't require a logger-name field; it requires `request_id`, `step`, `succeeded`, `duration_ms`. All 22 pre-existing `test_logging.py` tests still pass after the fix. Logged as a bug fix during T-IMPL-1a, not a separate DEC.

### Carried-forward LOW concerns (T-EARLY-MP queue, none Phase-4-blocking)

- **Inline spec citations on constant assertions** in test files — assertion lines lack `# from spec: <section>` annotation alongside the literals; docstrings cite spec but assertions don't.
- **3 legitimate `# noqa` suppressions in `logging.py`** — justified by structural Python constraints (conditional structlog import + `Formatter.format` builtin shadow).
- **DEC-022..025 pre-code-logging timing unverifiable** — Critical Rule 9 claim; substantive content honest but timestamps don't prove ordering.

## 6. Phase 4 progress vs `tomato_plan.md`

| Plan task | Status | Notes |
|---|---|---|
| Batch 0: 4 utility modules (per DEC-021) | ✓ DONE | Per master prompt Section 4 Phase 4; not in `tomato_plan.md`'s numbered batches; out-of-band per DEC-021 |
| T-IMPL-1a (sacred_guard) | ✓ DONE | 30 tests; verify_manifest 10/10 PASS |
| T-IMPL-1b (FastAPI skeleton) | ✓ DONE | 43 tests; uvicorn-launchable; port 8767; no APIN import |
| T-IMPL-1c (lint/test scaffolding) | ✓ DONE | pyproject.toml + .pre-commit-config.yaml; bash hook untouched |
| T-IMPL-2a, 2b, 2c (Batch 2 — input validation, IQA, preprocessing) | NOT STARTED | Awaiting user direction |
| T-IMPL-3 onwards | NOT STARTED | Future batches |

**Phase 4 cumulative effort: ~7-8 hours estimated** (4 utilities at 2h each + 3 Batch-1 tasks at 1-2h each, minus parallel speedups).

## 7. Servers running (background, untouched)

- `http://127.0.0.1:8766` — legacy APIN, PID 30160 (no field notes UI)
- `http://127.0.0.1:8768` — APIN v2, PID 19020 (field notes UI)
- New sandbox server NOT started this session (T-IMPL-1b created the skeleton; user can `uvicorn tomato_sandbox.api.server:app --port 8767` if desired, but spec says single uvicorn process and APIN-2 is not blocking 8767)

## 8. Cumulative metrics through Phase 4 first session

| Category | Count | Change this session |
|---|---|---|
| BLKs filed | 10 | +0 |
| Master-prompt defects | 53 | +0 (Defects 45..53 from Phase 3 close still queued) |
| DECs logged | 28 | +8 (DEC-021..028) |
| Phase exit gate fires | 12 | +0 (Phase 3 closed; Phase 4 has no exit gate yet) |
| Sacred drift events | 0 (post-DEC-019 baseline) | +0 |
| `.py` files in `tomato_sandbox/` | 18 | +14 this session (5 utils + 5 tests + sacred_guard + server + config + 1 init + test_sacred_guard + test_server) |
| Unit tests passing | 176 | +176 (was 0) |
| Section 15 tests passing | 0 | +0 (expected; needs T-IMPL-5a) |

## 9. Next steps (await user direction)

Phase 4 Batch 2 per `tomato_plan.md`:
- T-IMPL-2a — Input validation (Section 5)
- T-IMPL-2b — IQA (Section 6) **[D1-patched task card; spec-verbatim contract details available]**
- T-IMPL-2c — Preprocessing (Section 7)

These can run in parallel (no inter-dependencies). Each ~1-2h. Same dispatch pattern as Batch 1; same audit cadence (sacred after each, anti-cheat after the 3rd, checkpoint after 3 modules done = ~immediately).

Per master prompt: STOP after every checkpoint, wait for user approval before continuing.

**Awaiting your direction.**

---

*Generated 2026-05-01 by main-thread scribe; consolidates 4 implementer subagent dispatches (1 sequential + 3 parallel) and 2 anti-cheat scans + 2 sacred verifications. All claims independently verified by direct disk read + pytest run.*
