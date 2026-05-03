# Phase 5 Consolidated Audit Report

**Coverage:** Phase 5a (integration audit) + Phase 5b (spec contract audit) + Phase 5c (anti-cheat final sweep)
**Date range:** 2026-05-02 to 2026-05-03
**Phase exit gate:** Phase 5 closes on this report's CLOSE verdict

---

## Phase 5a Summary

- **Trigger:** Integration layer audit (T-AUDIT-5a) dispatched after Phase 4 close. Real-subprocess + real-image test under venv Python.
- **Bugs surfaced:** 1 (BLK-013 — orchestrator passed raw `PIL.Image` to `compute_iqa` which expects an object with `.pil_image` attribute per S6.6:1374; AttributeError caught internally and produced REJECT(0.0); every real-image POST short-circuited at IQA gate with HTTP 200 + body status 422).
- **Fix applied:** DEC-048 — `_PILAdapter` inner class injected at `pipeline.py:528`. 3-line mechanical change.
- **Verification:** Real-image POST to `/predict` returned a spec-compliant Tier 4B-degraded response (β interpretation per DEC-047: pre-F.0 sandbox does not load model weights; degraded-mode signals propagate through full pipeline → all-signals-failed sentinel → Rule 1 → Tier 4B; THIS is a valid S16.2 response, not an error).
- **BLK-013 → RESOLVED.**
- **Five-bug stopping cap:** not approached (1 bug found, cap is 5).
- **Stability across 5 real images:** 4 returned Tier 4B (expected pre-F.0 path); 1 returned legitimate IQA wetness rejection (proves IQA discriminates properly post-fix).
- **Sibling-bug speculation refuted:** `preprocess_for_v3/lora/psv` and `compute_signal_a/b/c` all take their declared input types correctly; only `compute_iqa` had the contract mismatch. The 11-row call-site verification matrix in `phase_5a_integration_audit.md` §3 is authoritative for orchestrator boundaries.

## Phase 5b Summary

- **Trigger:** Spec contract audit (T-AUDIT-5b) dispatched after Phase 5a close.
- **Modules audited:** 16 (all production modules in `tomato_sandbox/`).
- **Pass 1 findings: 8 total**
  - **F-01 — AUDITOR ERROR (M4 false positive)** caught by main-thread verification: auditor claimed `reason_human` had unfilled `{mime_type}` placeholder; actual code has English message + structured `received: "unknown"` field. Auditor confused spec-quote comment for code. **Reverted to JUSTIFIED on inspection.**
  - **F-02, F-03 — JUSTIFIED** (sub-package layouts per DEC-030, DEC-031, DEC-033)
  - **F-04 — JUSTIFIED-WITH-DEFERRAL** (TTA signature deviates from spec per DEC-037 D5; deferred until pipeline object formalized)
  - **F-05 — JUSTIFIED** (build_response signature omits `signal_a/b/c` per DEC-043 D1; data accessible through TierAssignment/ClassifierResult)
  - **F-06 — NEW DEFECT (BLK-014)**: `explanation.structured` block emitted only 4 of 12 fields per S16.4:5754-5778. Missing: `max_prob_threshold`, `margin_threshold`, `psv_reliability_threshold`, `psv_reliability_actual`, `chilli_leakage_threshold`, `chilli_leakage_actual`, `tier_sub_rule_checks` block (`iqa_degraded_check` + `underpowered_class_check`). `sub_rule_id_fired` was echoing `rule_id_fired` rather than spec's distinct `"default"` value.
  - **F-07 — COVERAGE GAP (BLK-015)**: `SeverityResult.grade_per_class` field declared but never populated for Tier 3A/3B per S17.5:6017.
  - **F-08 — CONFORMANCE-minor**: `multi_image/aggregator.py` omits `underpowered_classes` parameter from `assign_tier` call; spec allows omission.

- **Fixes applied (combined dispatch T-AUDIT-5b-fix):**
  - **DEC-049 (BLK-014):** `_get_structured_thresholds(rule_id_fired)` helper imports threshold constants from `tier_assignment.py`; `signal_extra` parameter pattern carries `chilli_leakage_actual` + `psv_reliability_actual` from orchestrator without fattening `TierAssignment` (preserves 3-field shape per DEC-041; 135 Section 15 tests unaffected); `tier_sub_rule_checks` block added with `iqa_degraded_check` + `underpowered_class_check`; `sub_rule_id_fired` distinct from `rule_id_fired` per spec example. 14 new unit tests.
  - **DEC-050 (BLK-015):** `compute_severity` extended with `multi_class_set: Optional[list] = None`. When ≥2 disease class indices present (healthy/OOD excluded per S17.6), iterates per-class with shared `coverage_pct` per SPEC-INT-003. Orchestrator passes `multi_class_set` only for Tier 3A/3B. 11 new unit tests.

- **SPEC-INT-003 logged**: S17.5 example shows different `coverage_pct` per class (11.2 vs 4.8); contradicts normative S17.2:5964 ("severity is a PSV-only computation"). Resolved as drafting noise; implementation passes shared `coverage_pct` to all classes (only threshold lookup varies).

- **M4 family extension logged**: F-07 auditor paraphrase typed `grade_per_class` as `Dict[str, SeverityGrade]`; actual spec example shape and implementation are list-of-dicts. Trust-but-verify caught the type-shape mismatch in main-thread verification. *(Note: Phase 5c report initially mis-attributed M4 to `signal_extra`; corrected here — M4 was about `grade_per_class`.)*

- **Verification:** 1150 tests pass under venv. Smoke test confirmed 8 new fields populated in live `/predict` response.

- **BLK-014 → RESOLVED. BLK-015 → RESOLVED.**

- **Anti-cheat (Phase 5b-fix dispatch):** PASS clean — 0 HIGH, 1 MEDIUM (test count accounting gap: dispatch baseline cited 1121 = unit+integration; actual 1150 = unit+integration+e2e; 29 e2e tests were missing from cumulative number since Batch 7), 2 LOW cosmetic. All tests pass; no test suppressed or fabricated.

## Phase 5c Summary

See `anticheat_phase5c_final_sweep.md` for granular details.

- **Verdict:** PASS clean. 0 HIGH, 2 MEDIUM, 3 LOW.
- **Section 15 LF-SHA256 vs DEC-032 baseline:** all 13 hashes match exactly (independent main-thread re-verification).
- **Pre-commit hook md5:** `24eb46f308751df3a125faca0680c9c7` unchanged.
- **Suppressed failures:** zero HIGH or MEDIUM. 3 LOW (config YAML fallback, GPU-lock cleanup, PSV degraded-mode shape extract — all with rationale).
- **DEC ledger:** 49 real headed entries (DEC-001..050 minus DEC-016 ghost). Two anomalies (DEC-016 body-only reference; parallel-dispatch ordering inversions in DEC-025/026 and DEC-042/043/044) — process artifacts, not cheats. MEDIUM WARN.
- **BLK ledger:** 15 real BLKs (1 template + 15). BLK-006/007/008 still OPEN (planning-phase non-blocking; implementations correct per spec read; recommend Phase 5 disposition note at T-EARLY-MP cycle). MEDIUM WARN.
- **Spec-citation density:** all 6 audited modules above 5-per-100-LOC threshold (range 5.9–14.9).
- **Test count accounting:** 1150 = 986+135+29 confirmed; no drift.
- **Server boot sanity:** all 4 endpoints HTTP 200.

## Phase 5 Cumulative Metrics

| Metric | Pre-Phase-5 | Post-Phase-5 |
|---|---|---|
| Total tests passing under venv | 1125 (961 unit + 135 int + 29 e2e) | **1150** (986 unit + 135 int + 29 e2e) |
| Net tests added | — | +25 (14 BLK-014 + 11 BLK-015) |
| HIGH cheat-pattern findings | 0 | 0 |
| BLKs OPEN | 1 (BLK-013 deferred) + 3 (BLK-006/007/008 planning-phase non-blocking) | 0 deferred + 3 planning-phase OPEN |
| BLKs RESOLVED in Phase 5 | 0 | 3 (BLK-013, BLK-014, BLK-015) |
| Section 15 immutability | Maintained | Maintained (all 13 LF-SHA256 hashes match DEC-032 baseline) |
| Sacred manifest | 10/10 PASS | 10/10 PASS unchanged |
| DECs logged | DEC-001..046 | **DEC-001..050** (+4: DEC-047/048/049/050) |
| BLKs filed | 13 | **15** (+BLK-014, BLK-015 added & RESOLVED) |
| SPEC-INT entries | 2 | **3** (+SPEC-INT-003) |
| Server endpoints functional | 7/7 (skeletons + lifecycle stubs) | 7/7 (full S16.2 response shape; new explanation.structured fields) |

## What Works (Plain Language)

- Sandbox server runs end-to-end on `127.0.0.1:8767` under venv Python; all 7 endpoints respond.
- 1150 tests pass: 986 unit + 135 Section 15 integration + 29 e2e.
- Section 15 milestone (135/135 from Batch 5) preserved intact through Phase 5a + 5b + 5c.
- Sacred manifest pre-commit hook armed and functional; md5 unchanged.
- Full pipeline computes without crashing: validate → IQA → preprocess → Signal A (v3 mock) → Signal B (LoRA mock) → Signal C (PSV) → TTA → Classifier → Conformal → Tier Assignment → Severity → Response Builder.
- Real-image POST to `/predict` returns structurally valid S16.2 JSON response (Tier 4B degraded as expected without loaded weights).
- `explanation.structured` block now includes all 12 fields per S16.4 (Phase 5b DEC-049).
- `SeverityResult.grade_per_class` populated for Tier 3A/3B multi-class prediction sets (Phase 5b DEC-050).
- Spec-citation density across 6 audited modules: 5.9–14.9 per 100 LOC, all above threshold.

## What Doesn't Work (Honest Gaps)

- **No real model weights loaded** (DEC-047 (β) deferral; F.0 territory). Pre-F.0 sandbox skip on startup steps 4-7 stays intact. v3 backbone, LoRA fine-tune, PSV calibration not loaded.
- **Tier 4B is the only observable response** in pre-F.0 because all signals fail without weights → all-signals-failed sentinel → Rule 1 → Tier 4B. Architecturally correct (degraded-mode path is exercised) but Tier 1-3 behavior not yet observable end-to-end.
- **F.0 calibration thresholds in S17.3** (severity grading: `mild_max`, `moderate_max`, etc.) are spec-derived approximations pending agronomist confirmation with ground-truth labeled images.
- **BLK-006/007/008** remain formally OPEN in the ledger (Platt calibration parameter list, PSV 26-feature list, prototype_blend coefficients in body vs summary). Phase 4 implementers applied option A (read spec body), but no Phase 5 disposition note was added. Recommend T-EARLY-MP cycle to add disposition.
- **DEC-016** referenced in DEC-012 body but never headlined. Decision content (sacred-guardian shell rewrite deferred) is documented inline.
- **conformal_tau.json** contains placeholder `tau=0.42`. Real τ will be computed during F.0 calibration on the held-out 40-image set per S13.3.

## Phase 5 Exit Verdict

**CLOSE: Phase 5 is closed. Phase 6 (F.0 prep per spec Section 29) is authorized.**

Rationale: zero HIGH cheat findings, zero unresolved blocking defects, 1150/1150 tests passing, Section 15 immutability maintained (13 LF-SHA256 hashes match DEC-032 baseline byte-for-byte), all three Phase 5 BLKs (BLK-013/014/015) resolved. The two MEDIUM anomalies (DEC ledger ordering artifacts from parallel-dispatch, BLK-006/007/008 disposition notes absent) are documentation hygiene items that do not block Phase 6 entry.

## Architectural Lessons Reinforced (M-Series Meta-Findings)

- **M1** (Phase 5a): Real-subprocess + real-image testing is qualitatively different from in-process TestClient. The BLK-013 PIL adapter bug was invisible to unit tests (which mock `compute_iqa`) but surfaced immediately when a real PIL.Image was passed through the orchestrator to the actual `compute_iqa` call site under venv Python.
- **M2** (Phase 5a): Mocking integration boundaries hides integration bugs. The unit-level mock of `compute_iqa` accepted any argument type; the real function required a `ValidatedImage`. This class of bug requires integration-layer testing with real (or near-real) object types flowing through the full call chain.
- **M3** (Phase 5a): Fix-cycle depth is bounded when integration was previously well-structured. The BLK-013 fix was a single-dispatch, single-fix resolution with no sibling bugs — because Phase 4 Batch 7 integration layer was well-structured and the only gap was the one call-site type mismatch.
- **M4** (Phase 5b): Auditor false positives arise from reading spec-quote comments instead of implementation body. F-01 paraphrased a spec-quote comment with `{mime_type}` placeholder as if it were the actual code; F-07 paraphrased `grade_per_class` type-shape as `Dict[str, SeverityGrade]` when the actual implementation is list-of-dicts. Trust-but-verify against spec body text (not comments) and against actual implementation code remains mandatory for contract-auditing tasks.

## Audit Trail Files

| Phase | Report | Location |
|---|---|---|
| 5a | Integration audit | `tomato_progress_reports/phase_5a_integration_audit.md` |
| 5b | Spec contract audit (Pass 1 + Pass 2) | `tomato_progress_reports/phase_5b_contract_audit.md` |
| 5b | Anti-cheat over fixes | `tomato_progress_reports/anticheat_phase5b_fix_20260503T1530.md` |
| 5c | Anti-cheat final sweep | `tomato_progress_reports/anticheat_phase5c_final_sweep.md` |
| 5 | Consolidated (this file) | `tomato_progress_reports/phase_5_audit.md` |

## Commits in Phase 5

| Commit | Subject |
|---|---|
| `dd41794` | Phase 5a: Integration layer audit CLOSED — BLK-013 RESOLVED via DEC-048 |
| `076c960` | Phase 5b: BLK-014 + BLK-015 RESOLVED via T-AUDIT-5b-fix; SPEC-INT-003 logged |
| (pending) | Phase 5c CLOSE / Phase 5 final |

## Forward Outlook (Phase 6 entry)

Phase 6 = F.0 prep per spec Section 29. Three components:
1. **F.0 validation script** (`tomato_sandbox/validation/run_f0.py`) — drives the held-out 40-image set through the full pipeline with real loaded weights; computes calibrated τ for conformal prediction; computes per-disease severity threshold validation against agronomist labels.
2. **Calibration script** (`tomato_sandbox/validation/fit_calibration.py`) — fits Platt parameters for classifier; fits per-disease severity thresholds; fits OOD chilli_leakage threshold.
3. **Real model loading** at startup steps 4-7 (lifts pre-F.0 deferral per DEC-047). Most architecturally uncertain piece. Same protocol pattern as Phase 5a: real-subprocess + real-image + real-models test; surface bugs; mechanical fixes via implementer sub-dispatch; CLOSE on non-error response with non-zero confidence values.

Realistic estimate: 2-4 sessions for Phase 6 (F.0 prep), 1-3 for Phase F.0 dry-run depending on labeled data availability.

---

*Generated 2026-05-03 by main-thread scribe. Consolidates 1 spec-auditor dispatch (T-AUDIT-5b, read-only) + 1 anti-cheat-inspector dispatch over fixes (T-AUDIT-5b-fix anti-cheat) + 1 anti-cheat-inspector dispatch (T-AUDIT-5c, read-only, this report's primary source) + 1 implementer dispatch (T-AUDIT-5a, BLK-013 fix) + 1 implementer dispatch (T-AUDIT-5b-fix, BLK-014/015 fixes) + main-thread independent verification of every substantive claim. All numbers (1150 tests, 49 DECs, 15 BLKs, 13 Section 15 file hashes, sacred 10/10) verified by direct measurement under venv Python.*
