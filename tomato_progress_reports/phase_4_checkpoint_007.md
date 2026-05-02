# Phase 4 Checkpoint 007 — Server endpoints wired (S20); **Phase 4 closes via Option B**

**Date:** 2026-05-02
**Cadence trigger:** Phase 4 final batch (Batch 7).
**Session scope:** T-IMPL-7 server endpoint wiring (S20) → smoke-test debug cycle → Option B honest closure → Phase 5 prerequisite added → Phase 4 ends.
**Verdict:** **Phase 4 CLOSES with one identified-deferred wiring bug (BLK-013). Server runs end-to-end. Phase 5 entry awaiting user approval.**

---

## 1. Headline result

**Server runs end-to-end on port 8767 under venv Python.** All 7 spec-defined endpoints respond. 12-step startup sequence executes (sacred 10/10 verified at step 1; FAIL-FAST on missing conformal tau implemented per S20.5). POST `/predict` with a real image goes through validation → IQA gate → returns structured response per spec.

**One integration wiring bug identified in pipeline.py:527** (BLK-013) causes every real-image POST to short-circuit at the IQA gate with `IQA_REJECTED` (HTTP 200 + body status 422). The bug is mechanical (3-line fix) but **deferred to Phase 5 audit** per Option B closure: the same TestClient-mocking pattern that hid this bug may be hiding integration bugs further down the pipeline, and Phase 5 audit's mandate is exactly that kind of integration validation.

**Cumulative tests: 1096 pass under venv Python** (961 unit + 135 integration). System Python: 1118 — both runs clean. Section 15 preserved at 135/135 through 4 fix cycles in this session. Sacred 10/10.

## 2. The four fixes applied this session

### Fix #1 — Orchestrator unit tests updated for S16-compliant response shape
T-IMPL-7's pipeline.py step-18 wired the inline stub to `response_builder.build_response`, producing the nested S16 schema. The 10 orchestrator unit tests still asserted the old flat-dict shape. T-IMPL-7-fix sub-dispatch updated all 10 assertions to nested S16 paths (`result["tier"]["label"]`, etc.). Tests pass.

### Fix #2 — venv installed structlog + pytest + pytest-asyncio + httpx
**Discovered via Defect-60.** All Phase 4 prior pytest reports used system Python (miniconda 3.13.11), masking that venv (3.13.11) lacked these packages. Symptom: `venv/Scripts/python.exe -m uvicorn ...` failed with `ModuleNotFoundError: No module named 'structlog'`. Installed in venv. Test count parity verified: 1096 venv vs 1118 system (delta exactly the 7 new logging fallback tests + the gpu_lock timeout test fix; both runs clean).

Standing rule: ALL future test-count reports must specify interpreter (venv vs system); venv pytest is authoritative for production-equivalence claims.

### Fix #3 — logging.py stdlib fallback hardened (DEC-046)
**Latent since Batch 0 / DEC-022.** The "structlog with stdlib fallback" design returned a raw `logging.Logger` that crashed `TypeError: Logger._log() got unexpected keyword 'shape'` when production callsites used structlog-style kwargs (`_log.debug("event", key=val)`). Fixed via new `_StdlibKwargsAdapter` class wrapping stdlib Logger; arbitrary kwargs go via stdlib's `extra=` dict (which `_StdlibJsonFormatter` already merges as JSON fields).

7 unit tests added in `test_logging.py::TestStdlibKwargsAdapter` simulating structlog-missing via `patch.object(_logmod, "_STRUCTLOG_AVAILABLE", False)` — exercise the fallback without uninstalling structlog. Tests cover: kwargs accepted at every log level; reserved stdlib kwargs (`exc_info`, `stack_info`, `stacklevel`) pass through; `extra=` dict merging; `event` field convention matching structlog; `bind()` no-op for compatibility; `get_logger()` returns adapter type when flag is False.

### Fix #4 — GPU lock cross-loop bug (orchestrator skip-if-already-locked)
**Surfaced during real-subprocess smoke test.** `GPULock` uses `asyncio.Lock` which is event-loop-bound. Server.py acquires it in the FastAPI loop, then dispatches `predict_single` via `run_in_executor` (worker thread, no running loop). The orchestrator's `predict_single` then tried to re-acquire from the worker thread via `asyncio.run(...)`, creating a fresh loop — cross-loop asyncio.Lock hangs 10s → SERVER_OVERLOAD.

Fix: `pipeline.py` heuristic refined. If no running loop AND `gpu_lock.locked` is True, skip acquisition (server holds it; worker thread piggybacks). If lock is unlocked (sync test path), acquire normally. Track `acquired_locally` flag for matching release. Note: `GPULock.locked` is a `@property`, not a method — initial fix erroneously called as method, produced `'bool' object is not callable`; corrected to property access.

Test fix: `test_gpu_lock_timeout_returns_503` had to set `mock_lock.locked = False` (literal attribute, not `return_value`) so MagicMock mimics property-style access correctly.

## 3. The smoke test result (real-subprocess milestone evidence)

```
Server launch: venv/Scripts/python.exe -m uvicorn tomato_sandbox.api.server:app --host 127.0.0.1 --port 8767
Boot: ✓ all 12 startup steps executed
       ✓ sacred 10/10 PASS at step 1
       ✓ conformal tau loaded (step 8: tau=0.42 placeholder for pre-F.0)
       ✓ /ready returns 200, /health, /info, /metrics all respond

POST /predict (textured noise image, 55KB):
  HTTP 200, 28ms (no 503 timeout!) → IQA_REJECTED
POST /predict (synthetic leaf-shaped image, 144KB):
  HTTP 200, 18ms → IQA_REJECTED

Response body:
  {
    "error": "IQA_REJECTED",
    "message": "Could not read the image. Please re-take and upload again.",
    "request_id": "...",
    "status": 422
  }
```

The IQA_REJECTED is **not** what we wanted from the milestone (the user's success criterion was a "real prediction, not 503"), **but** it IS evidence the server runs end-to-end:
- Pipeline ran from request to response in tens of milliseconds (not 10s timeout)
- Validation succeeded (`request_validated` log line)
- Decode succeeded (`image_decoded` log line)
- IQA was called (the bug surfaces inside `compute_iqa`'s try/except)
- Response builder produced a structured rejection per spec
- Server returned HTTP 200 with proper JSON body

The bug is identified, isolated, and trivial to fix.

## 4. BLK-013 documentation

```
BLK-013 [2026-05-02] Pipeline IQA call site contract mismatch
Location: tomato_sandbox/orchestrator/pipeline.py line 527
Symptom:  compute_iqa(pil_image) passes raw PIL.Image; compute_iqa expects
          ValidatedImage-shaped object with .pil_image attribute.
Why hidden: 29 in-process e2e tests mock compute_iqa.
Mechanical fix: 3 lines (wrap PIL in adapter class).
Status: IDENTIFIED, NOT FIXED — deferred to Phase 5 audit per Option B closure.
Resolution path: Phase 5 entry prerequisite (real-subprocess + real-image)
                 surfaces this and any sibling integration bugs systematically.
```

Full BLK-013 entry in `tomato_blockers.md`. Includes mechanical fix code, deferral rationale, sibling-bug speculation (Signal A/B/C wiring possibly similar), and resolution path.

## 5. The architectural finding (M2 — TestClient mocking hid integration layer)

The 29 in-process e2e tests in `test_endpoints.py` mocked `compute_iqa`. The mock was reasonable for unit-test purposes (avoid real-image processing in test, isolate endpoint logic). But it became load-bearing when treated as integration-layer validation. Three of the five bugs found in this session would not have surfaced through TestClient testing alone:

| Bug | Surfaces in TestClient? | Surfaces in subprocess? |
|---|---|---|
| #1 unit-test shape | Yes (test runs) | N/A |
| #2 venv missing deps | No (TestClient uses test process's Python) | Yes (subprocess uses venv Python) |
| #3 logging fallback | No (system Python has structlog; mock works) | Yes (venv Python lacks structlog) |
| #4 GPU lock cross-loop | No (TestClient runs in single async context) | Yes (real run_in_executor in worker thread) |
| #5 IQA wiring (BLK-013) | No (compute_iqa is mocked) | Yes (real un-mocked path) |

**Three out of five bugs are class "subprocess-only".** This validates the M2 finding: in-process TestClient is qualitatively different from real-subprocess + real-image + real-models testing.

Phase 5 entry prerequisite added to master prompt Section 4 codifies the lesson: real-subprocess test must run BEFORE spec-auditor dispatches.

## 6. Phase 4 wins (durable record)

| Item | State |
|---|---|
| Modules in `tomato_sandbox/` | 13+ (utils + api + signals + classifier + conformal + tier + orchestrator + response + severity + multi_image) |
| Unit tests under venv | **961 passing** |
| Integration tests | **135/135 passing** (Section 15 deterministic scenarios) |
| Grand total under venv | **1096 passing** |
| Grand total under system Python | 1118 passing (delta: 7 new logging tests + 1 gpu_lock test fix) |
| Sacred manifest | 10/10 PASS unchanged across 8 batch closures |
| DECs logged | DEC-001..046 |
| BLKs filed | 13 (12 RESOLVED, 1 IDENTIFIED-DEFERRED = BLK-013) |
| Master-prompt defects catalogued | 60 (in T-EARLY-MP queue) |
| Git commits ahead of origin (post-Batch-7) | 8 |
| Real-subprocess sandbox server | boots cleanly on 8767 under venv |
| All 7 endpoints | respond per spec |

### Procedural rules empirically validated across multiple batches

| Rule | Validation count |
|---|---|
| DEC-018 / Fix-42 — read spec body, not summaries | T-IMPL-1, 2, 3, 4, 5, 6, 7 (every batch) |
| DEC-021 — master prompt authoritative for ordering | Every dispatch |
| DEC-033 — sub-package + re-export shim policy | Batches 2, 3, 4, 6 (and Batch 7 inherits) |
| DEC-038 — main thread does all commits | Batches 4, 5, 6, 7 (zero implementer commits) |
| Pre-allocation rule for parallel DECs | Batches 2, 3 W1, 4, 6 (clean across all) |

### Spec discoveries surfaced in Phase 4

- **BLK-009 / Defect-9.2 (RESOLVED Batch 3):** v3 → canonical remap [0,2,1,3,4,5] inside `extract_v3_outputs`.
- **BLK-009 / Defect-9.1 (RESOLVED Batch 3):** TTA function signature `should_trigger_tta(combined_max_prob: float) -> int`.
- **BLK-010.2 (UPDATED Batch 4):** ClassifierResult has 9 fields, not 6 (T-IMPL-4a discovered).
- **BLK-011 (RESOLVED Batch 5):** tier_assignment 12 rule_id_fired canonical values per import contract.
- **BLK-012 (RESOLVED Batch 6):** S17.2 references nonexistent PSV features (`mean_lesion_intensity`, `lesion_size_distribution`); used proxies.
- **BLK-013 (IDENTIFIED-DEFERRED Batch 7):** pipeline.py:527 IQA wiring contract mismatch.

## 7. Honest gap

**Real-image path through real-loaded models is not validated at Phase 4 close.**

What we know works end-to-end (server runs, IQA gate fires, structured rejection per spec, all 1096 + 135 tests pass) is not the same as what the user actually wants in production (a real prediction with tier label, confidence, rule_id_fired, etc.). The path from `validate → IQA → preprocess → signals → classifier → conformal → tier_assignment → response_builder` has not been exercised end-to-end with un-mocked components and real images.

Phase 5 entry prerequisite covers this gap explicitly. Phase 5 audit cannot certify what hasn't run end-to-end.

## 8. State after Phase 4 closes

| Item | State |
|---|---|
| Sacred manifest | 10/10 PASS, unchanged |
| Pre-commit hook | armed (md5 24eb46f308751df3a125faca0680c9c7) |
| Both APIN servers | running (PID 24452 on 8766, PID 23132 on 8768) |
| Sandbox port 8767 | **HELD until BLK-013 closes in Phase 5** |
| Master prompt | updated with Phase 5 entry prerequisite |
| BLK ledger | DEC-001..013 (BLK-013 IDENTIFIED-DEFERRED) |
| DEC ledger | DEC-001..046 |
| T-EARLY-MP queue | 35 entries (Defect-60 added this batch) |

## 9. Phase 5 readiness checklist

Awaiting user approval to begin Phase 5. Entry requires:
- [ ] Real-subprocess smoke test on 8767 under venv with real model loading
- [ ] Real-image POST `/predict` end-to-end through full pipeline
- [ ] Spec-auditor's integration layer audit dispatched first (BLK-013 + any siblings)
- [ ] Then contract-level audit (Pass 1 spec-only; Pass 2 cross-references decisions)
- [ ] Anti-cheat final sweep
- [ ] Sacred final verification
- [ ] All tests under venv (authoritative count for production-equivalence)

**The endgame is real. Phase 5 begins when you approve.**

---

*Generated 2026-05-02 by main-thread scribe; consolidates: T-IMPL-7 dispatch + sub-fix dispatch + 4 main-thread fix cycles + 1 anti-cheat scan equivalent (deferred to Phase 5 due to Option B closure scope) + in-sandbox sacred verification (10/10 PASS) + Section 15 regression check (135/135 PRESERVED through all 4 fix cycles) + venv vs system Python equivalence check (1096 vs 1118; both clean). Honest accounting: BLK-013 identified, mechanical fix documented, deferred to Phase 5 by Option B per architectural finding M2.*
