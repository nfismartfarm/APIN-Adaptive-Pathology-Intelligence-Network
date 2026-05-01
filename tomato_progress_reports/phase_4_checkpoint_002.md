# Phase 4 Checkpoint 002 — Batch 2 complete (input validation + IQA + preprocessing)

**Date:** 2026-05-01
**Cadence trigger:** master prompt every-3-modules cadence (Batch 2 = 3 modules since checkpoint_001).
**Session scope:** Phase 4 Batch 2 per `tomato_plan.md` — T-IMPL-2a (S5 input validation), T-IMPL-2b (S6 IQA, D1-patched), T-IMPL-2c (S7 preprocessing).
**Verdict:** **Session complete. STOP and await user direction on Batch 3.**

---

## 1. Procedural fixes applied before dispatch

**Pre-allocation rule (Defect-54 / Fix-55, HIGH).** Per the DEC-026/DEC-028 collision in Batch 1, the main thread now pre-allocates DEC numbers and passes them in dispatch prompts. Each Batch 2 implementer was told: "log your DEC entry as DEC-NNN; do not pick a different number; do not pick the next available number."

**Result:** Three parallel implementers wrote DEC-029, DEC-030, DEC-031 with zero collisions. Pre-allocation rule effective. Logged as Fix-55 in T-EARLY-MP queue (`tomato_plan.md` position 30).

## 2. Modules implemented this session

| Task | Path(s) | Bytes (impl + test) | Spec section | DEC |
|---|---|---|---|---|
| T-IMPL-2a | `tomato_sandbox/input_validation.py` (canonical, 23,700) + `tomato_sandbox/api/validate_input.py` (re-export shim, 958) + `test_validate_input.py` (39,140) | 63,798 total | 5 | DEC-029 |
| T-IMPL-2b | `tomato_sandbox/iqa/iqa.py` (21,059) + `tomato_sandbox/iqa/__init__.py` (341) + `test_iqa.py` (33,670) | 55,070 total | 6 (D1-patched) | DEC-030 |
| T-IMPL-2c | `tomato_sandbox/preprocessing/preprocess.py` (14,371) + `tomato_sandbox/preprocessing/__init__.py` (852) + `test_preprocess.py` (29,998) + `config.py` modifications (+8 constants) | 45,221 total + config delta | 7 | DEC-031 |

## 3. Tests added

| Test file | Tests | Status |
|---|---|---|
| `test_validate_input.py` | 96 | PASS |
| `test_iqa.py` | 82 | PASS |
| `test_preprocess.py` | 61 | PASS (env-dep on torch+Pillow) |

**Cumulative unit tests passing:** 415 (was 176; +239 from Batch 2). Verified by `pytest tomato_sandbox/tests/unit/` → `415 passed in 48.29s` with 56 Pillow-deprecation warnings (non-breaking, queued for cosmetic fix).

**Section 15 integration tests:** still 13 collection errors with `ModuleNotFoundError: No module named 'tomato_sandbox.tier'` — expected; tier_assignment.py not yet implemented (T-IMPL-3 territory).

## 4. Audit verdicts

| Audit | Verdict | Notes |
|---|---|---|
| Sacred (in-sandbox `tomato_sandbox.utils.sacred_guard.verify_manifest`) | **10/10 PASS** | Canonical algorithm per DEC-019; authoritative. |
| Sacred (main-thread Haiku sacred-guardian) | 9/9 file PASS; 1/1 directory algorithm-drift artifact | Subagent's ad-hoc reimplementation diverges from canonical algorithm. Subagent itself confirms file count (145 non-excluded) matches manifest. **No actual data drift.** Same pattern as checkpoint_001. |
| Anti-cheat (T-IMPL-2a/2b/2c) | **PASS clean. 0 HIGH, 0 MEDIUM, 3 LOW** | All LOWs are cosmetic/process (env-skip guards, codec-edge skip, git-tracking inconsistency). Pre-allocation rule honored. |

## 5. Decisions logged this session

| DEC | Title | Trigger |
|---|---|---|
| DEC-029 | T-IMPL-2a Input validation: canonical path divergence (`tomato_sandbox/input_validation.py` per spec 5.7:1049 vs task card's `api/validate_input.py`); Check B ordering; spec 5.5 line 1023 edge-case resolution; Pillow 14 deprecation note | Batch 2 (pre-allocated) |
| DEC-030 | T-IMPL-2b IQA: package layout vs flat-file spec (sub-package + re-export); ValidatedImage forward reference (parallel-task soft typing); nan_guards not used (bounded arithmetic); degraded_mode not used (IQA is precondition gate, not signal block) | Batch 2 (pre-allocated) |
| DEC-031 | T-IMPL-2c Preprocessing: sub-package layout vs flat-file spec; 8 preprocessing constants added to config.py per spec 7.2:1421-1432; guard_array imported for output finiteness on float pipelines (v3 + LoRA); no guard on PSV uint8 (always finite) | Batch 2 (pre-allocated) |

## 6. Open issues / surfaced findings

### Process success — pre-allocation rule worked
Three parallel implementers wrote three sequential DECs with zero collisions. Compare Batch 1's DEC-026 collision requiring main-thread renumbering. Procedural fix validated.

### Plan-vs-spec divergences (3, all minor, all documented)
- T-IMPL-2a: spec specifies `tomato_sandbox/input_validation.py`; plan/dispatch said `api/validate_input.py`. Resolved via shim. Per DEC-018, spec wins.
- T-IMPL-2b: spec specifies flat `iqa.py`; dispatch said `iqa/iqa.py`. Resolved via package + re-export.
- T-IMPL-2c: spec specifies flat `preprocessing.py`; dispatch said `preprocessing/preprocess.py`. Resolved via package + re-export.

All three: imports work via either path; tests cover the public surface. No spec content disregarded.

### LOW-3 from anti-cheat: git tracking inconsistency
T-IMPL-2b's implementer force-added IQA files past `.gitignore`'s `tomato*/` rule (commit `69d8ce7`); T-IMPL-2a and T-IMPL-2c remain untracked per gitignore. Creates uneven provenance for Batch 2. **Queued for T-EARLY-MP:** decide whether tomato_sandbox/ is tracked or gitignored, unify behavior. Not Phase-4-blocking.

### 56 Pillow deprecation warnings
`Image.Image.getdata` deprecated in Pillow 14 (2027-10-15). `_is_effectively_grayscale` in input_validation.py uses it on a 64×64 thumbnail. Non-breaking; queued for cosmetic fix when Pillow ≥ 14 is pinned.

## 7. Phase 4 progress vs `tomato_plan.md`

| Plan task | Status | Notes |
|---|---|---|
| Batch 0 (4 utilities) | ✓ DONE (checkpoint_001) | DEC-021/022/023/024/025 |
| Batch 1 (T-IMPL-1a/1b/1c) | ✓ DONE (checkpoint_001) | DEC-026/027/028 |
| Batch 2 (T-IMPL-2a/2b/2c) | ✓ DONE (this checkpoint) | DEC-029/030/031 |
| T-IMPL-3 onwards (tier assignment, signals, hierarchical, conformal, TTA, server orchestration) | NOT STARTED | Awaiting user direction |

**Phase 4 cumulative effort:** ~10-11 hours estimated (4 utils + Batch 1 ~7-8h, Batch 2 ~3h with parallel speedup).

## 8. Cumulative metrics through Phase 4 second session

| Category | Count | Change this session |
|---|---|---|
| BLKs filed | 10 | +0 |
| Master-prompt defects | 54 | +1 (Defect-54: DEC numbering race; resolved as Fix-55 procedural rule) |
| DECs logged | 31 | +3 (DEC-029..031, all pre-allocated, no collisions) |
| Phase exit gate fires | 12 | +0 |
| Sacred drift events | 0 (post-DEC-019 baseline) | +0 |
| `.py` files in `tomato_sandbox/` | 25 | +7 this session (input_validation + api/validate_input + iqa/iqa + iqa/__init__ + preprocessing/preprocess + preprocessing/__init__ + 3 test files; config.py modified not added) |
| Unit tests passing | 415 | +239 (was 176) |
| Section 15 tests passing | 0 | +0 (expected; needs T-IMPL-3+ then 5a) |

## 9. Next steps (await user direction)

`tomato_plan.md` Batch 3 candidates (consult plan for exact T-IMPL-3 sequencing):
- Tier assignment module (`tomato_sandbox/tier/tier_assignment.py`) — unblocks all 13 Section 15 integration test files
- Signal A wrapper (v3 model, Section 8)
- Signal B wrapper (LoRA epoch 13, Section 9)
- Signal C wrapper (PSV classical CV, Section 10)

Per master prompt: STOP after every checkpoint, wait for user approval before continuing.

**Awaiting your direction on Batch 3 composition and dispatch.**

---

*Generated 2026-05-01 by main-thread scribe; consolidates 3 parallel implementer subagent dispatches (with pre-allocated DEC numbers per Fix-55) + 1 sacred-guardian dispatch + in-sandbox sacred verification + 1 anti-cheat dispatch. All claims independently verified by direct disk read + pytest run + grep sampling.*
