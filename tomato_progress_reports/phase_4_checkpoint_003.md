# Phase 4 Checkpoint 003 — Batch 3 complete (Signals A/B/C + TTA orchestration)

**Date:** 2026-05-02
**Cadence trigger:** master prompt every-3-modules (Batch 3 = 4 modules since checkpoint_002).
**Session scope:** preparatory items (DEC-032 git-tracking + DEC-033 module-layout + tracking backfill commit) + Phase 4 Batch 3 two-wave dispatch — Wave 1 (T-IMPL-3a Signal A v3 / T-IMPL-3b Signal B LoRA / T-IMPL-3c Signal C PSV in parallel) + Wave 2 (T-IMPL-3d TTA, sequential after Wave 1 to read on-disk A/B contracts).
**Verdict:** **Session complete. STOP and await user direction on Batch 4.**

---

## 1. Procedural improvements applied this session

### Two-wave dispatch (replaces all-4-parallel)
TTA orchestrates Signals A and B and depends on their actual API. Three-parallel Wave 1 + sequential Wave 2 lets TTA's implementer read on-disk v3_signal.py and lora_signal.py rather than only spec paraphrases. Cost: one extra round-trip. Benefit: contract verification against real signatures, eliminating cross-task contract-mismatch risk.

### DEC-032 — git-tracking policy
`.gitignore` updated. `tomato_sandbox/` and `tomato_progress_reports/` now tracked normally per spec sections 26.6 (CI gates) and 28.5 (bringup). Phase-0's broad `tomato*/` rule (which on Windows case-insensitive matching also caught `tomato_sandbox/` via `Tomato*/`) was scaffolding from when no real sandbox code existed. Negation patterns `!tomato_sandbox/`, `!tomato_sandbox/**`, `!tomato_progress_reports/**` added; re-ignore for `scratch/`, `models/`, `__pycache__/`, `*.pyc` retained inside `tomato_sandbox/`.

Backfill commit `a926d3d`: 93 files staged (~80 source + 13 progress reports + .gitignore + 3 tomato_*.md updates). Two stale `.pyc` artifacts removed via `git rm --cached`. Resolves anti-cheat LOW-3 from checkpoint_002 (uneven Batch-2 provenance).

**One-time `--no-verify` bypass for the backfill commit, user-authorized.** Pre-commit hook literal logic blocks ALL `git diff --cached` matches on Section 15 tests, including first-time additions. Hook intent (DEC-008) was to block post-Phase-3 modifications, not initial-tracking transitions. Verification trail in `tomato_log.md` "DEC-032 addendum":
- Pre-commit SHA256 hashes recorded for all 13 Section 15 files
- Post-commit LF-normalized hashes match baseline exactly (Windows autocrlf flips raw bytes only)
- Hook re-armed verification: synthetic edit blocked correctly; reverted
- Filed as Fix-56 candidate for T-EARLY-MP queue (hook should use `--diff-filter=M`)

### DEC-033 — module-layout policy
Codified pattern: when spec describes flat module file but task-card describes sub-package, implementer creates sub-package + re-export shim. Both import paths must work. Empirical basis: 3 Batch 2 implementers independently arrived at this. Each Batch 3 implementer cited DEC-033 explicitly. Result: zero plan-vs-spec layout disagreement re-derivation cost in Batch 3.

## 2. Modules implemented this session

### Wave 1 (parallel, 3 implementers)
| Task | Path(s) | Bytes | Spec | DEC |
|---|---|---|---|---|
| T-IMPL-3a Signal A v3 | `tomato_sandbox/signals/__init__.py` (206) + `signals/v3_signal.py` (12,291) + `tests/unit/test_signal_a.py` (18,682) | 31,179 | S8 | DEC-034 |
| T-IMPL-3b Signal B LoRA | `signals/lora_signal.py` (21,470) + `tests/unit/test_signal_b.py` (21,947) | 43,417 | S9 | DEC-035 |
| T-IMPL-3c Signal C PSV | `signals/psv/__init__.py` (553) + `psv/psv.py` (10,462) + `psv/leaf_segmentation.py` (3,267) + `psv/disease_detection.py` (4,255) + `psv/features.py` (24,045) + `psv/compatibility.py` (8,359) + `psv/reliability.py` (5,404) + `tests/unit/test_psv.py` (23,292) + `config/psv_weights.yaml` + `phase_f0_calibration/psv_standardization.json` | 79,637+ | S10 | DEC-036 |

### Wave 2 (sequential, after Wave 1 reconciled)
| Task | Path(s) | Bytes | Spec | DEC |
|---|---|---|---|---|
| T-IMPL-3d TTA | `signals/tta.py` (canonical) + `tomato_sandbox/tta.py` (flat-path shim per spec 11.7:3103) + `tests/unit/test_tta.py` | ~ | S11 | DEC-037 |

## 3. Tests added

| Test file | Tests | Status |
|---|---|---|
| `test_signal_a.py` | 15 | PASS |
| `test_signal_b.py` | 18 | PASS (skipped without torch — env-dep guard) |
| `test_psv.py` | 56 | PASS |
| `test_tta.py` | 34 | PASS (4 classes skip without PIL+torch — env-dep guard) |

**Cumulative unit tests passing:** 538 (was 415; +123 from Batch 3). Verified by `pytest tomato_sandbox/tests/unit/` → `538 passed in 47.60s`.

**Section 15 integration tests:** still 13 collection errors with `ModuleNotFoundError: No module named 'tomato_sandbox.tier'` — expected; tier_assignment.py is T-IMPL-4 territory.

## 4. Critical contract pins verified

| Pin | Verdict | Evidence |
|---|---|---|
| v3 → canonical remap `[0,2,1,3,4,5]` INSIDE `extract_v3_outputs` (BLK-009 Defect-9.2) | ✓ | `_V3_TO_CANONICAL_REMAP` defined and applied inside the function. `test_remap_correctness` sends v3-ordered input → asserts canonical-ordered output. |
| LoRA index ordering matches canonical; no remap (S9.1) | ✓ | No remap in `lora_signal.py`. Inline citation at S9.1:1822. |
| Signal B single forward pass; no MC dropout (S9) | ✓ | `test_single_pass_only` asserts `mock_model.call_count == 1`. |
| PSV CPU-only; no `gpu_lock`; no `torch.cuda` (S10) | ✓ | Zero `gpu_lock` and `torch.cuda` references under `signals/psv/`. |
| 26 PSV features; BLK-007 traceability per feature (S10) | ✓ | `assert len(FEATURE_NAMES) == 26`; per-feature `# spec: 10.X.Y lines NNNN-NNNN` comments. |
| `should_trigger_tta(combined_max_prob: float) -> int` returning {1,2,5} (BLK-009 Defect-9.1) | ✓ | Signature matches; thresholds align with `nan_guards.py` constants. |
| TTA does NOT invoke PSV; Signal C used as single un-augmented value (S11) | ✓ | `test_psv_not_called` asserts `mock_psv.call_count == 0`. |

## 5. Audit verdicts

| Audit | Verdict | Notes |
|---|---|---|
| Sacred (in-sandbox `verify_manifest()`) | **10/10 PASS** | Canonical algorithm per DEC-019. Authoritative. |
| Anti-cheat (T-IMPL-3a/3b/3c/3d) | **CONDITIONAL PASS** — 0 HIGH, 1 MEDIUM, 4 LOW | MEDIUM: test_tta.py docstring claims 29 tests; 34 collected (undercount in docs, more tests than claimed — safe direction). LOWs: narrow remap citation range (1672-1674 vs BLK-009 1664-1685), Signal B single-pass test skips on no-torch CI, TTA PSV-not-called test class skip broader than needed, auditor mislocation of git provenance commit (a926d3d not 28cc945). |
| Section 15 immutability via pre-commit hook | **VERIFIED ARMED** | Hook re-fired correctly on synthetic edit during DEC-032 verification; SHA256 hashes recorded; LF-normalized comparison proves no content drift. |

## 6. Decisions logged this session

| DEC | Title | Trigger |
|---|---|---|
| DEC-032 | Git-tracking policy: `tomato_sandbox/` tracked normally; one-time `--no-verify` bypass for initial Section 15 tracking with hash verification trail | Pre-Batch-3 prep |
| DEC-033 | Module-layout policy: sub-package + re-export shim when spec and plan disagree | Pre-Batch-3 prep |
| DEC-034 | Signal A v3 wrapper: sub-package layout, mock backbone in tests, GPU lock as orchestrator concern (Sec 21), remap inside `extract_v3_outputs` | Wave 1 |
| DEC-035 | Signal B LoRA wrapper: sub-package layout, no remap, single-pass enforced, prototype blending wired (S9.5), mock model in tests | Wave 1 |
| DEC-036 | PSV (Signal C) classical CV: 7-file sub-package, 26 features per S10, BLK-007 traceability per feature, weight matrix YAML loader with regex fallback | Wave 1 |
| DEC-037 | TTA orchestration: canonical at `signals/tta.py`, flat-path shim at `tomato_sandbox/tta.py` (S11.7:3103), PSV excluded, Signal B single-pass preserved per view | Wave 2 |

## 7. Open issues / surfaced findings

### Process success — two-wave dispatch validated
TTA's implementer read v3_signal.py and lora_signal.py on disk before writing tests. Result: zero contract-mismatch issues between Wave 1 and Wave 2; TTA's mocks correctly call `compute_signal_a` and `compute_signal_b` with their actual signatures. Confirms the cost (one extra round-trip) is worth it for cross-task-dependency batches.

### Plan-vs-spec layout divergence — mostly clean
- T-IMPL-3a: sub-package only (DEC-034 noted spec 8.7 directly specified the sub-package; no shim needed).
- T-IMPL-3b: sub-package only (DEC-035 same).
- T-IMPL-3c: sub-package only (matches plan).
- T-IMPL-3d: spec 11.7:3103 specifies `tomato_sandbox/tta.py`; plan specifies `signals/tta.py`. Resolved per DEC-033: canonical at `signals/tta.py`, shim at `tomato_sandbox/tta.py`. Both paths verified working.

### MEDIUM-1 follow-up needed
test_tta.py docstring "29 tests" → "34 tests". One-line edit; queue for next session.

### LOW-1, LOW-2, LOW-3 → T-EARLY-MP queue
Cosmetic CI-coverage and citation-precision improvements; no blocking value.

### LOW-4 — auditor git mislocation
Anti-cheat scan referenced commit `28cc945` (old initial scaffold) for Batch 3 provenance; correct commit is `a926d3d` (this session's backfill). Auditor pattern hint: provide commit SHA in the dispatch prompt so the auditor doesn't have to discover it.

## 8. Cumulative metrics through Phase 4 third session

| Category | Count | Change this session |
|---|---|---|
| BLKs filed | 10 | +0 |
| Master-prompt defects | 56 | +2 (Defect-55 placeholder for Fix-56 hook-additions case; LOWs not promoted to defects) |
| DECs logged | 37 | +6 (DEC-032..037) |
| Phase exit gate fires | 12 | +0 |
| Sacred drift events | 0 (post-DEC-019 baseline) | +0 |
| `.py` files in `tomato_sandbox/` | 35+ | +10 this session (signals/__init__ + v3_signal + lora_signal + 6 PSV files + signals/tta + tta shim + 4 test files) |
| Unit tests passing | 538 | +123 (was 415) |
| Section 15 tests passing | 0 | +0 (expected; needs T-IMPL-4 then 5a) |
| Git commits | 1 new | `a926d3d` Pre-Batch-3 prep + tracking backfill (one-time `--no-verify` for initial Section 15 tracking, hash verified) |

## 9. Next steps (await user direction)

`tomato_plan.md` Batch 4 candidates — the highest-value next module is **tier assignment (`tomato_sandbox/tier/tier_assignment.py`)** because it would unblock all 13 Section 15 integration test files (currently 13 ModuleNotFoundError). Per the spec's tier rule chain (Rules 1-9, sub-rules 7a/7b/7c, 8a/8b/8c), this is also the heart of the system's decision logic.

Other Batch 4 candidates (in approximate spec order):
- Hierarchical classifier with soft routing (S12)
- Conformal prediction at 90% marginal coverage (S13)
- Server orchestration / 12-step pipeline (S20-21)

Per master prompt: STOP after every checkpoint, wait for user approval before continuing.

**Q4 reminder:** sandbox server launch on 8767 still held until enough downstream infrastructure is in place for real smoke tests (per your guidance, re-evaluate after Batch 4 — likely candidate when classifier + conformal land).

Two web servers stay running (legacy APIN 8766, APIN v2 8768). Pre-commit hook stays installed and armed. Sacred manifest stays at DEC-019 baseline.

**Awaiting your direction on Batch 4 composition and dispatch.**

---

*Generated 2026-05-02 by main-thread scribe; consolidates 4 implementer subagent dispatches (3 parallel Wave 1 + 1 sequential Wave 2 reading Wave 1 contracts) + 1 anti-cheat dispatch + in-sandbox sacred verification + git tracking backfill commit + Section 15 SHA256 hash verification trail. All claims independently verified by direct disk read + pytest run + grep sampling + LF-normalized hash comparison.*
