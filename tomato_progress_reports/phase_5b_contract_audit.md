# Phase 5 Spec Contract Audit — Final Report (Pass 1 + Pass 2)

**Auditor:** spec-auditor (Sonnet 4.6, isolated context)
**Date:** 2026-05-03
**Saved by:** main-thread scribe (spec-auditor lacks Write tool; report content returned as text)

**Audit scope:** All 16 implementation modules in `tomato_sandbox/` (excluding test files).
**Spec:** `tomato_3_signal_system.md`.
**Decisions log:** `tomato_decisions.md` (DEC-001..048).
**Blockers:** `tomato_blockers.md` (BLK-001..013, all RESOLVED).
**Agent capability:** read-only (Read, Glob, Grep, Bash).

---

## SUMMARY TABLE

| # | Module | Pass 1 Finding | Severity | Pass 2 Verdict | Justification |
|---|--------|---------------|----------|----------------|---------------|
| F-01 | `input_validation.py` | ~~`reason_human` string for `unsupported_format` does not interpolate `{mime_type}`~~ — **AUDITOR ERROR (false positive); main-thread verification found auditor confused spec-quote comment with actual code** | MINOR | **AUDITOR ERROR — implementation deviates differently (more user-friendly message + structured `received` field); JUSTIFIED on inspection** | M4 meta-finding filed |
| F-02 | `iqa/iqa.py` | Package layout (`iqa/iqa.py`) vs spec flat path (`tomato_sandbox/iqa.py`) | DOC DRIFT | JUSTIFIED | DEC-030, DEC-033 |
| F-03 | `preprocessing/preprocess.py` | Package layout vs spec flat path (`tomato_sandbox/preprocessing.py`) | DOC DRIFT | JUSTIFIED | DEC-031, DEC-033 |
| F-04 | `signals/tta.py` | `apply_tta` signature differs from spec: `(pil_image, n_views, v3_model, lora_model, prototype_bank, initial_combined_max_prob)` vs spec `(pipeline, validated_image, n_views)` | MEDIUM | JUSTIFIED-WITH-DEFERRAL | DEC-037 Decision 5 |
| F-05 | `response/response_builder.py` | `build_response` omits `signal_a`, `signal_b`, `signal_c` parameters vs spec S16.1 line 5643 | MEDIUM | JUSTIFIED | DEC-043 Decision 1 |
| F-06 | `response/response_builder.py` | `explanation.structured` block is structurally incomplete vs spec S16.4 lines 5754-5778 | MEDIUM | NEW DEFECT | No DEC or BLK covers this |
| F-07 | `severity/grader.py` | Multi-class severity for Tier 3A/3B (spec S17.5 lines 6015-6032) not implemented; `grade_per_class` field exists but is never populated | MEDIUM | COVERAGE GAP | No DEC or BLK resolves this |
| F-08 | `multi_image/aggregator.py` | `assign_tier` call omits `underpowered_classes` parameter | MINOR | CONFORMANCE | Defaults to None, spec allows omission |

**All other modules: CONFORMANCE — no deviations found.**

| Tier | Count |
|---|---|
| HIGH | 0 |
| MEDIUM | 5 (F-01, F-04, F-05, F-06, F-07) |
| LOW (DOC DRIFT) | 2 (F-02, F-03) |
| INFO | 0 |
| MINOR | 1 (F-08) |
| **Total** | **8** |

| Pass 2 reconciliation (after main-thread verification) | Count |
|---|---|
| JUSTIFIED | 3 (F-02, F-03, F-05) |
| JUSTIFIED-WITH-DEFERRAL | 1 (F-04) |
| JUSTIFIED-AS-SPEC-DEFECT | 0 |
| AUDITOR ERROR (false positive corrected by main thread) | 1 (F-01) |
| **NEW DEFECT** | **1 (F-06)** |
| **COVERAGE GAP** | **1 (F-07)** |
| CONFORMANCE-minor | 1 (F-08) |

---

## DETAILED FINDINGS

### F-01 — `input_validation.py` — **AUDITOR ERROR (false positive) — corrected by main-thread verification**

**Auditor's original claim:** `reason_human` contains literal `{mime_type}` placeholder (unfilled f-string).

**[MAIN-THREAD VERIFICATION 2026-05-03]** False positive. The auditor confused a spec-quote comment (line 328: `# spec: 5.3 line 995 — "File type {mime_type} is not supported."` — the spec's parametric wording quoted in a code comment) with the actual code. The actual `reason_human` at lines 332-336 reads:

```python
reason_human=(
    "File type is not supported. Use JPEG or PNG. "
    "(iPhone HEIC photos can be shared as JPEG from the share menu.)"
),
```

There is no `{mime_type}` placeholder in the actual string. The implementation chose a different message structure (English message + structured `received: "unknown"` field at the rejection record) rather than the spec's parametric `"File type {mime_type} is not supported."` template.

**Re-verdict (main thread):** This is a **MINOR / DOC DRIFT — JUSTIFIED on inspection** at most. The implementation deviation is real but in spirit of the spec (more user-friendly: explicit JPEG/PNG guidance + HEIC tip). The structured rejection record carries the diagnostic info (`received: "unknown"`, `expected: "image/jpeg or image/png"`). No NEW DEFECT to file.

**Lesson for the meta-record:** spec-auditor's read-only nature can produce comment-vs-code confusion. Trust-but-verify caught it. This adds a project meta-finding: `M4 — Auditor false positives from spec-quote comments`. Future audit dispatches should explicitly cite both the comment line AND the actual code line in findings.

---

### F-02 — `iqa/iqa.py` — JUSTIFIED

**Spec section:** S2.6 / S4.1 file catalog which implies a flat `tomato_sandbox/iqa.py`.

**Finding (Pass 1):** The implementation places IQA code at `tomato_sandbox/iqa/iqa.py` (a sub-package with `__init__.py`) rather than at the flat path `tomato_sandbox/iqa.py` implied by the spec's file catalog.

**Pass 2:** DEC-030 explicitly records this deviation with user approval: *"IQA code lives at `tomato_sandbox/iqa/iqa.py`; the public import `from tomato_sandbox.iqa.iqa import compute_iqa` is the canonical import path. Spec's flat-file implication is overridden by sub-package layout approved in DEC-033."* DEC-033 establishes the general policy that sub-package layout is preferred when a module grows beyond a single file.

**Verdict: JUSTIFIED** (DEC-030, DEC-033).

---

### F-03 — `preprocessing/preprocess.py` — JUSTIFIED

**Spec section:** S4.1 file catalog implies flat `tomato_sandbox/preprocessing.py`.

**Finding (Pass 1):** The implementation places preprocessing code at `tomato_sandbox/preprocessing/preprocess.py` (a sub-package) rather than a flat module.

**Pass 2:** DEC-031 explicitly records this with user approval, following the same sub-package policy as DEC-033.

**Verdict: JUSTIFIED** (DEC-031, DEC-033).

---

### F-04 — `signals/tta.py` — JUSTIFIED-WITH-DEFERRAL

**Spec section:** S11 (TTA module), lines 2900-2960 approximately.

**Finding (Pass 1):** The `apply_tta` function signature in `tomato_sandbox/signals/tta.py` is:

```python
apply_tta(pil_image, n_views, v3_model, lora_model, prototype_bank, initial_combined_max_prob)
```

The spec S11 specifies the signature as `apply_tta(pipeline, validated_image, n_views)` — a single `pipeline` object encapsulating model references, and a `ValidatedImage` typed input.

The implementation passes models explicitly and takes a raw PIL image rather than a `ValidatedImage`. Signal C / PSV does not participate in TTA in the implementation (correct per spec), but the orchestration interface differs from the spec signature.

**Pass 2:** DEC-037 Decision 5 explicitly records this deviation: *"apply_tta signature differs from spec. Pipeline object does not yet exist as a concrete type; explicit model parameters used instead. Approved as deferred — will be refactored when the pipeline object is formalized."*

**Verdict: JUSTIFIED-WITH-DEFERRAL** (DEC-037 Decision 5). The deferral means this deviation is expected to remain until the `pipeline` object abstraction is implemented.

---

### F-05 — `response/response_builder.py` — JUSTIFIED

**Spec section:** S16.1, line 5643.

**Finding (Pass 1):** Spec S16.1 line 5643 specifies:

```
build_response(tier_assignment, classifier_result, conformal_result, iqa_result,
               signal_a, signal_b, signal_c, request_metadata) -> dict
```

The implementation `build_response` signature is:

```python
build_response(tier_assignment, classifier_result, conformal_result, iqa_result, request_metadata)
```

Parameters `signal_a`, `signal_b`, `signal_c` are absent.

**Pass 2:** DEC-043 Decision 1 explicitly records this deviation: *"build_response omits signal_a, signal_b, signal_c parameters. All information needed for the response is accessible through classifier_result and tier_assignment, which already carry the aggregated signal data. Including raw signal objects would require response_builder to understand signal internals — cross-layer coupling. Approved deviation."*

**Verdict: JUSTIFIED** (DEC-043 Decision 1).

---

### F-06 — `response/response_builder.py` — NEW DEFECT

**Spec section:** S16.4, lines 5754-5778.

**Finding (Pass 1):** Spec S16.4 lines 5754-5778 specifies the `explanation.structured` JSON block as containing:

Under `tier_main_conditions`:
- `max_prob_actual` (present)
- `max_prob_threshold` (MISSING in implementation)
- `margin_actual` (present)
- `margin_threshold` (MISSING)
- `psv_reliability_threshold` (MISSING)
- `psv_reliability_actual` (MISSING)
- `chilli_leakage_threshold` (MISSING)
- `chilli_leakage_actual` (MISSING)

Under `tier_sub_rule_checks` (entire sub-object MISSING):
- `iqa_degraded_check` (MISSING)
- `underpowered_class_check` (MISSING)

Additionally, `sub_rule_id_fired` in the implementation is set to the same value as `rule_id_fired` rather than a distinct sub-rule identifier as the spec intends.

The implementation's `explanation.structured` block thus exposes only 4 of the approximately 12 fields required by S16.4.

**Pass 2:** No DEC entry (DEC-001..048) addresses this. No BLK entry identifies the `explanation.structured` schema as a known gap. DEC-043 Decision 1 covers the `build_response` signature deviation but makes no mention of the `explanation.structured` schema completeness. This is not covered by any approved deviation.

**Verdict: NEW DEFECT.** The `explanation.structured` block is materially incomplete relative to the spec contract. Missing fields include threshold values that callers (dashboards, agronomist tooling) would use to display why a particular tier was assigned. The spec is explicit and detailed at S16.4.

---

### F-07 — `severity/grader.py` — COVERAGE GAP

**Spec section:** S17.5, lines 6015-6032.

**Finding (Pass 1):** Spec S17.5 lines 6015-6032 specifies multi-class severity grading for Tier 3A and 3B cases, where a prediction set contains multiple disease classes. The spec requires `grade_per_class: Dict[str, SeverityGrade]` to be populated with a severity grade for each class in the conformal prediction set.

The `SeverityResult` dataclass in `grader.py` does include the `grade_per_class` field. However, the `compute_severity` function body only handles a single `predicted_class` parameter. The function computes a single severity grade and returns it; `grade_per_class` is never populated (remains as its default value, an empty dict or None depending on dataclass initialization).

For Tier 3A/3B cases where the orchestrator passes a multi-class conformal set, the returned `SeverityResult` will have `grade_per_class` empty, meaning downstream response builder and any agronomist tooling gets no per-class breakdown for ambiguous diagnoses.

**Pass 2:** BLK-012 covers a related but distinct issue: the spec's group number citations for severity features are wrong (G3 for `mean_lesion_intensity`, G7/G8 for `lesion_size_distribution`). BLK-012's resolution (DEC-044 Decision 2) approves using `mean_lesion_size` as a proxy but does not address multi-class severity at all. No DEC (DEC-001..048) addresses the Tier 3A/3B multi-class severity implementation gap. No BLK identifies this coverage gap.

**Verdict: COVERAGE GAP.** The `grade_per_class` field exists as scaffolding but is never computed. This is not a deviation from spec (the field is present) — it is a specification contract that the implementation declares but does not execute. For single-class tiers (1, 2, 4A, 4B) this has no user-visible impact. For Tier 3A/3B the spec contract is unmet.

---

### F-08 — `multi_image/aggregator.py` — CONFORMANCE (minor note)

**Spec section:** S18.5, line 6193.

**Finding (Pass 1):** The `assign_tier` call in `aggregator.py`'s Step 7 does not pass `underpowered_classes`. In the `tier_assignment.py` implementation, `underpowered_classes` defaults to `None` when omitted. When `None`, the underpowered-class check (Rule 8a/8b/8c) evaluates against an empty set, meaning no class is treated as underpowered in multi-image aggregated predictions.

**Pass 2:** The spec S18.5 line 6193 says the aggregator calls `assign_tier` with the aggregated signal values and does not explicitly require that `underpowered_classes` be forwarded from per-image results. The `underpowered_classes` parameter is a system configuration (not per-image metadata); the aggregator correctly does not inject per-image underpowered class lists into the aggregate call.

This is a minor observation, not a defect. The behavior is consistent with "no class is underpowered at the aggregate level" which is a reasonable default.

**Verdict: CONFORMANCE.** No action needed.

---

## MODULES WITH FULL CONFORMANCE (no findings)

The following 16 modules were audited and found fully conformant with their claimed spec sections. No deviations, gaps, or excess behaviors beyond those listed in F-01..F-08 above:

1. **`utils/logging.py`** — S26.7 structlog with `_StdlibKwargsAdapter` fallback (DEC-022, DEC-046). `log_step()` enforces `request_id`, `step`, `succeeded`, `duration_ms`. Sensitive field redaction via `SENSITIVE_FIELDS` frozenset. CONFORMANCE.

2. **`utils/gpu_lock.py`** — `GPULock` with `acquire_with_timeout()`, `acquired()` context manager, timeout/retry logic. CONFORMANCE.

3. **`utils/nan_guards.py`** — TTA NaN-guard thresholds: trigger=0.55, escalate=0.45 match spec S11. Guard logic correct. CONFORMANCE.

4. **`utils/degraded_mode.py`** — `VECTOR_DIM=19`, signal slices A=[0:6], B=[6:12], C=[12:19] match spec S4.2. Degraded mode fallbacks correct. CONFORMANCE.

5. **`utils/sacred_guard.py`** — Directory hash uses compact JSON `separators=(",",":")` as required. SHA-256 verification chain correct. CONFORMANCE.

6. **`api/server.py`** — Port 8767, `/predict` multipart endpoint, `/health`, GPU warning (not exit) per DEC-026. CONFORMANCE with documented DEC-026 deviation (warning vs exit approved).

7. **`api/validate_input.py`** — Pure re-export shim per DEC-029 (dual-path input validation policy). CONFORMANCE.

8. **`signals/v3_signal.py`** — `_V3_TO_CANONICAL_REMAP = np.array([0, 2, 1, 3, 4, 5])` applied inside `extract_v3_outputs` per spec S8.3 and BLK-009 (BLK-009 confirmed v3 has septoria/late_blight swapped). `SignalAResult.tomato_probs_canonical` field carries already-remapped canonical probs. CONFORMANCE.

9. **`signals/lora_signal.py`** — Prototype blending: triggered when `lora_max_prob < 0.60`, `BLEND_WEIGHT=0.35`, `T_PROTO=0.3`. `SignalBResult` fields match spec S9. CONFORMANCE.

10. **`signals/psv/psv.py`** — 5-stage PSV pipeline, 26 features in `FEATURE_NAMES`, 13-field `SignalCResult` including `psv_argmax`, `psv_max_prob`, `coverage_pct`, `lesion_count`, `mean_lesion_size`, `lesion_size_std`, `chilli_leakage`, `reliability`, `psv_succeeded`, `failure_reason`, `processing_time_ms`. BLK-012 noted inline (G3/G7 group number inconsistency; proxy fields used). CONFORMANCE.

11. **`classifier/hierarchical_classifier.py`** — Stage 1 (3-class healthy/diseased/OOD), Stage 2 (5-class disease), soft routing, Platt calibration. All 9 `ClassifierResult` fields from spec S12.10 present verbatim. CONFORMANCE.

12. **`conformal/conformal.py`** — `CONFORMAL_ALPHA=0.10`, `CONFORMAL_N_CALIBRATION=40`, `τ` loaded from `conformal_tau.json`, 90% coverage guarantee. CONFORMANCE.

13. **`orchestrator/pipeline.py`** — All 22 pipeline steps present. `predict_multi` documented stub returning `aggregated: None` with per-image `predict_single` loop. `_build_pipeline_result` delegates correctly. `_make_fallback_conformal()` returns all-7-class set (conservative). BLK-013 (`_PILAdapter` fix) applied at IQA call site per DEC-048. CONFORMANCE.

14. **`severity/severity.py`** — Pure re-export shim per DEC-044 Decision 1. CONFORMANCE.

15. **`multi_image/multi_image.py`** — Pure re-export shim per DEC-044. CONFORMANCE.

16. **`multi_image/aggregator.py`** — All 7 aggregation steps per spec S18.4 implemented correctly. T5 OR preservation, pipeline failure handling, weighted class voting, conformal fraction (>= 0.50), worst-IQA selection, PSV min/max aggregation, final `assign_tier` call. CONFORMANCE (minor note on `underpowered_classes` omission at F-08).

---

## RECOMMENDED NEW BLK ENTRIES

**BLK-014 (suggested):** `explanation.structured` schema incomplete in `response_builder.py` (F-06). Spec S16.4 lines 5754-5778 requires approximately 12 fields in `tier_main_conditions` and `tier_sub_rule_checks`. Implementation exposes 4. The missing fields are: `max_prob_threshold`, `margin_threshold`, `psv_reliability_threshold`, `psv_reliability_actual`, `chilli_leakage_threshold`, `chilli_leakage_actual`, and the entire `tier_sub_rule_checks` sub-object (`iqa_degraded_check`, `underpowered_class_check`). The response builder has access to all this data (it receives `tier_assignment` which carries rule metadata). This is not a design question — it is straightforward field population.

**BLK-015 (suggested):** Multi-class severity grading for Tier 3A/3B not implemented (F-07). `compute_severity` in `grader.py` computes only a single-class grade. Spec S17.5 lines 6015-6032 requires `grade_per_class` to be populated for ambiguous multi-class tiers. The `grade_per_class` field exists in `SeverityResult` as scaffolding but is never written. Requires either: (A) `compute_severity` to accept a list of classes and iterate per-class grading, or (B) `compute_severity` to be called in a loop per class by the orchestrator. No spec change needed — the contract is clear at S17.5.

**No new BLK for F-01** — main-thread verification found this was an AUDITOR ERROR (auditor confused spec-quote comment with actual code; the implementation does not have an unfilled `{mime_type}` placeholder). No fix needed; no blocker needed. Filed as M4 meta-finding for the project record.

---

## STOP-RULE ASSESSMENT

The audit identified 3 NEW DEFECT / COVERAGE GAP findings (F-01, F-06, F-07). The 5-finding HIGH threshold for early stop was not reached. All findings are MEDIUM severity or below. Audit proceeded to completion across all 16 modules.

---

## AUDIT METADATA

- Modules audited: 16 of 16
- Pass 1 findings: 8 total (1 NEW DEFECT, 1 NEW DEFECT response_builder, 1 COVERAGE GAP, 3 JUSTIFIED, 1 JUSTIFIED-WITH-DEFERRAL, 1 CONFORMANCE-minor)
- Pass 2 verdicts assigned: all 8
- DEC entries consulted: DEC-022, DEC-026, DEC-029, DEC-030, DEC-031, DEC-033, DEC-037, DEC-043, DEC-044, DEC-046, DEC-048
- BLK entries consulted: BLK-009, BLK-012, BLK-013
- New BLK entries recommended: 2 (BLK-014, BLK-015)
- Agent performed no edits (read-only by toolset)
