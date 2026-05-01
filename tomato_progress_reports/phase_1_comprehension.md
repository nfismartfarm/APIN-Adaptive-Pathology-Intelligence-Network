# Phase 1 Comprehension Report

**Date:** 2026-04-27
**Author:** main thread (orchestrator)
**Status:** Phase 1 ready for exit gate

---

## Scope of Phase 1

Per master prompt section 4.2, Phase 1 is read-only comprehension. No tomato_sandbox/ code is written. Outputs are: section summaries, dependency graph, populated skills, and this report.

## Section coverage

32 of 32 sections summarized in `.claude/spec_summaries/`. Plus appendices documented as missing (BLK-005).

| Batch | Sections | Status |
|---|---|---|
| 1 | 1, 2, 3, 4 | Done (saved by main-thread scribe per DEC-011) |
| 2 | 5, 6, 7 | Done |
| 3 | 8, 9, 10, 11, 12, 13, 14, 15 | Done (Section 15 = 135 scenarios fully indexed) |
| 4 | 16, 17, 18, 19, 20, 21 | Done |
| 5 | 22, 23, 24, 25 | Done |
| 6 | 26, 27, 28, 29, 30, 31, 32, appendices | Done |

## Spot-check results (3 sections)

Per master prompt section 4.2 final activity. Sections selected to span subject matter:

| Section | Spec lines verified | Result |
|---|---|---|
| 17 (severity grading) | 5980-5986 | PASS — per-disease threshold table matches summary exactly (foliar/septoria/late_blight/YLCV/mosaic coverage and lesion counts) |
| 23 (queue API) | 7090-7096 | PASS with patch — summary listed 6 endpoints; spec has 7 (`/queue/stats` was missing). Patched inline. |
| 11 (TTA) | 2932-2939 | PASS — trigger 0.55, escalate 0.45, view counts 1/2/5 match exactly |

Cartographer fidelity is high. One minor completeness gap (Section 23 endpoint table) was correctable inline without re-running the subagent.

## Dependency graph

Built at `.claude/spec_dependency_graph.md`. 6-layer build order recommendation for Phase 4 (T-IMPL-1 through T-IMPL-10). Critical edges identified:

1. S15 → S14 (verification edge — Phase 3 encoder blocks on S14 implementation)
2. S12 → S8/S9/S10 (fusion; index remap [0,2,1,3,4,5] applied here)
3. S13 → S12 (calibration; τ depends on classifier output distribution)
4. S29 → S15 + S25 (F.0 integrates 135 scenarios as part of validation procedure)
5. S20 → S2 (Sandbox Directive — port 8767 wins over leftover 8766 prose; BLK-002)

## Skills populated (T-EARLY-A)

| Skill | Status | Source |
|---|---|---|
| `tomato-section15-format` | ACTIVE | Section 15 summary |
| `tomato-conformal` | ACTIVE | Section 13 summary |
| `tomato-gpu-lock` | ACTIVE | Section 20.6 summary |

All three replaced placeholder content with usable schemas, code patterns, and consumer references.

## Open blockers entering Phase 2

| ID | Severity for next phase | Recommendation |
|---|---|---|
| BLK-002 (port 8766/8767 contradiction in S1.3/S2.3) | Low — Sandbox Directive is authoritative; documentation noise only. T-EARLY-MP cleanup batch will write the spec_changelog entry. | Option A |
| BLK-003 (APIN library import vs sandbox-no-import) | Low — same as BLK-002. Sandbox Directive wins. | Option A |
| BLK-004 (Section 15 internal defects) | Defect-15.1 is **BLOCKING for Phase 3**; encoder cannot proceed until S1.1 v3 vector is canonicalized. Defect-15.2 is monitoring-relevant only. | Defect-15.1 → option A (line 4117 authoritative); Defect-15.2 → option A (encoder enumerates the missing 3 during Phase 3) |
| BLK-005 (Appendices A-F missing) | **BLOCKING for Phase 4 T-IMPL-5** (`tier_assignment.py` needs YAML schema). Body sections sufficient for everything else. | Option A — implementer derives YAML schema from Section 14 prose with traceability comments; spec author may later supply Appendix D as confirmation |

None of these block Phase 2 (planning). All four are flagged in `tomato_blockers.md` with recommended resolution paths and waiting on user decision.

## Spec health summary

- Body sections: complete and self-consistent for v1 implementation. Section 14 prose + Section 15 scenarios + Section 29 F.0 procedure form a solid spec→test→validate triangle.
- Documentation drift: ports/APIN-import contradictions in S1.3/S2.3 are minor and superseded by Sandbox Directive (Section 0).
- Section 15 internal arithmetic: 51+81=132≠135 in T5 distribution; Defect-15.2 deferred to Phase 3 enumeration.
- Appendices A-F: declared but absent. G18 fix mitigates Appendix A. Appendix D (tier_rules.yaml example) is the most consequential gap and will be filled by implementer with explicit traceability.

## What I now understand (operational synthesis)

The system at runtime executes this order, per request:

```
1. Image upload (Section 5) → ValidatedImage
2. IQA (Section 6) → ACCEPTABLE/HIGH/DEGRADED/REJECT
3. Preprocessing (Section 7) → pp_classifier (224×224) + pp_psv (LAB-CLAHE)
4. Acquire GPU lock (Section 20.6)
5.   Signal A (Section 8): v3 model → 10-class probs, slice [0..5] tomato
6.   Signal B (Section 9): LoRA → 6-class probs, apply remap [0,2,1,3,4,5]
7.   PSV (Section 10): 26 features → 6-class compatibility (CPU; can be outside lock)
8.   Hierarchical classifier (Section 12) → P_final + P_final_calibrated (7-class with OOD)
9.   Conformal (Section 13) → prediction_set, size, τ-applied
10.  TTA decision (Section 11): if combined_max_prob < 0.55, re-run with N=2 or N=5
11.  Severity (Section 17) — only if tier ∈ {1, 2, 3A}, PSV-only computation
12. Release GPU lock
13. Tier assignment (Section 14) — Rules 1-9 chain, sub-rules 7a/7b/7c, 8a/8b/8c
14. T5 alert evaluation (Section 14.3) — independent of base tier
15. Response build (Section 16) — UnifiedResponse envelope
16. Persist (Section 24) — Phase E SQLite log per step
17. Route to queue if tier ∈ {3A, 3B, 3C, 3D, 4A, 4B} (Section 23)
18. Emit metrics (Section 25)
```

Scenarios in Section 15 exercise step 13 (`assign_tier`) directly with synthesized inputs from steps 1-12. The 135 tests are the load-bearing contract.

## Phase 1 exit gate

Pending. Will run `/tomato-phase-exit 1` next.
