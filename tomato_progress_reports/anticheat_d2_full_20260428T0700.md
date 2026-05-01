# Anti-Cheat D2 Full-Coverage Spec-Citation Audit — 19 Tasks

**Date:** 2026-04-28
**Auditor:** anti-cheat-inspector (read-only)
**Saved by:** main-thread scribe per DEC-011
**Coverage:** all 19 unaudited tasks (FULL coverage, not sample) per user-approved D2

## Verdict: 12 DEFECTIVE + 1 AMBIGUOUS + 6 VERIFIED → ~68% defect rate

The user's D2 stopping rule was: *"If the rate is ≥40%, we have a methodological problem and we'll discuss next steps."* This rate hits that threshold. **STOP and discuss methodology before proceeding to inline patches or Round 4.**

## Summary table

| Task ID | Status | Severity | Headline finding |
|---|---|---|---|
| T-IMPL-1c | DEFECTIVE | LOW | Ruff rule set fabricated; not in spec |
| T-IMPL-2a | **DEFECTIVE** | **HIGH** | `ValidatedImage` has 5 fabricated fields, 2 spec fields missing (`mime_type`, `sha256_hash`); entry-point signature wrong; file path wrong |
| T-IMPL-2c | **DEFECTIVE** | **HIGH** | Function signatures take wrong type; return wrong shapes; CLAHE described as RGB-channel instead of LAB-L; missing `shades_of_gray`; wrong file path |
| T-IMPL-3b | **DEFECTIVE** | **HIGH** | `SignalBResult` 5 wrong + 6 missing fields; `PrototypeBank` wrong; signatures wrong; model path wrong |
| T-IMPL-3c | **DEFECTIVE** | **HIGH** | `SignalCResult` missing 9 spec fields; `compute_signal_c` missing IQA params; fallback uses 0.1 instead of spec's 0.05; 5-stage descriptions invented (LBP/FFT not in spec) |
| T-IMPL-4a | VERIFIED | — | Consistent (already patched in earlier round) |
| T-IMPL-5b | DEFECTIVE | LOW | SB.7 acceptance criterion ambiguous on which rule fires; substantively correct but needs clarification |
| T-IMPL-5c | DEFECTIVE | MEDIUM | Omission conditions incomplete; `treatment_templates.yaml` path wrong; `SeverityResult` structure diverges |
| T-IMPL-5d | **DEFECTIVE** | **HIGH** | 7-step "aggregation" describes per-image processing, not spec's aggregation algorithm; all-rejected behavior wrong |
| T-IMPL-6b | **DEFECTIVE** | **HIGH** | `PipelineContext` describes per-request log context, not spec's model holder; `predict_single` signature wrong; step 11 says "remap applied here" — directly contradicts BLK-009 patch in this same plan |
| T-IMPL-7b | VERIFIED | — | Consistent |
| T-IMPL-8a | DEFECTIVE | MEDIUM | `predictions` table has 9 of 30+ spec columns; single retention env var vs spec's 3 |
| T-IMPL-8b | VERIFIED | — | Consistent |
| T-IMPL-9b | VERIFIED | — | Consistent |
| T-IMPL-10a | DEFECTIVE | LOW | `ExecStart` uses relative path; spec uses absolute `/opt/tomato_sandbox/.venv/...` |
| T-IMPL-10b | AMBIGUOUS | LOW | `hard_ceiling: 0.95` is a re-labeling of spec's "85-95% band"; values match |
| T-EARLY-MP | VERIFIED | — | Internally self-consistent meta task |
| T-EARLY-VENV | DEFECTIVE | LOW | `pre-commit` mis-categorized into prod requirements; minor |
| T-PHASE-3-PRECONDITIONS | VERIFIED | — | All 5 gates correctly stated |

## Defect distribution

- **HIGH:** 6 (T-IMPL-2a, 2c, 3b, 3c, 5d, 6b)
- **MEDIUM:** 2 (T-IMPL-5c, 8a)
- **LOW:** 4 (T-IMPL-1c, 5b, 10a, T-EARLY-VENV)
- **AMBIGUOUS:** 1 (T-IMPL-10b)
- **VERIFIED:** 6 (T-IMPL-4a, 7b, 8b, 9b, T-EARLY-MP, T-PHASE-3-PRECONDITIONS)

## Pattern analysis

The 6 VERIFIED tasks have a structural commonality: they are either small wiring tasks (7b, 9b), tasks that were already patched in prior rounds (4a), task-summary tasks with no novel code-shape claims (T-EARLY-MP, T-PHASE-3-PRECONDITIONS), or tasks whose dataclass details were entirely in spec_summaries that got copied verbatim (8b QueueCase / QueueDisposition).

The 12 DEFECTIVE tasks share a pattern: each makes claims about dataclass field names, function signatures, or algorithm steps that look plausible but diverge from the spec body. The planner did not read the spec body when these were drafted; it relied on summaries (which paraphrase) or invented from intuition.

## Cumulative defect rate across 3 anti-cheat samples

| Sample | Tasks examined | Defective |
|---|---|---|
| Round 1 (BLK-009) | 5 | 3 |
| Round 3 (BLK-010) | 5 | 3 |
| D2 (this audit) | 19 | 12 (+1 AMBIGUOUS) |
| **Cumulative** | **29 of 30** | **18 (+ 1 AMBIGUOUS) = 19** |

**63-66% rate.** The remaining 1 task (T-IMPL-9a) was VERIFIED in Round 1.

## Per the user's D2 stopping criterion (≥40%): METHODOLOGY DISCUSSION REQUIRED

Inline-patching 12+ defective tasks would consume effort and not address the root cause: **the planner does not consistently read the spec body when writing task cards**. Even after Fix-34 was patched into the master prompt and `.claude/agents/planner.md` per D6, those patches affect FUTURE planner invocations, not the existing plan whose 30 task cards were already drafted.

Two methodological options for user consideration:

**Option α — Re-fire planner with Fix-34 in effect.** The patched planner agent definition now mandates verbatim spec-body quoting. A fresh planner invocation should produce a substantially better plan. Cost: 1 large subagent invocation; benefit: clean audit trail; risk: planner produces new fabrications in different places (Fix-34 is a guideline, not enforceable).

**Option β — Push verification to Phase 4 implementer.** Each T-IMPL task gets a "spec-body-read" prerequisite: the implementer reads the relevant spec section body BEFORE writing code, and writes a verification report citing the spec lines used. The plan's task cards become navigation/scaffolding, not authoritative contracts. Cost: more time per Phase 4 task; benefit: defects surface at code-write time when they're cheapest to fix; risk: plan task cards become misleading documentation.

**Option γ — Hybrid.** Keep current plan as scaffolding, but mark each DEFECTIVE task card with a banner "spec body must be read before implementation; this card is non-authoritative." Phase 4 implementer reads spec for any banner-marked task. Cost: low; benefit: low effort + clear gating.

**Option δ — Inline-patch the 6 HIGH defects only; defer MEDIUM/LOW; accept residual risk.** Cost: ~1-2h main-thread scribe work; benefit: highest-risk tasks fixed; risk: AMBIGUOUS and MEDIUM defects ship into Phase 4 unfixed.

Each option is consistent with one of the user's prior preferences. The choice depends on:
- Trust in the planner subagent post-Fix-34 (Option α)
- Cost tolerance for in-Phase-4 reads (Option β)
- Tolerance for unfixed MEDIUM defects (Option δ)
- Preference for scaffolding-vs-contract semantics on `tomato_plan.md` (Option γ)

**Until user picks an option, no further patches applied. No Round 4 fire.**
