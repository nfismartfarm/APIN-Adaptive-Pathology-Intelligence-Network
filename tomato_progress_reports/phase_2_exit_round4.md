# Phase 2 Exit Gate — Round 4 Consolidated Audit Summary

**Date:** 2026-04-28
**Audits consolidated:** 5 (phase-exit-auditor, PVA, PDA, anti-cheat, sacred-guardian)
**Source artifacts (all read directly from disk, not from agent memory):**

| Audit | File | Bytes | Verdict |
|---|---|---|---|
| phase-exit-auditor | `phase_2_exit_audit_round4_20260428T0700.md` | 1508 | READY |
| prompt-validator | `pva_phase2_round4_20260428T0700.md` | 5029 | READY-WITH-DEVIATIONS |
| prompt-defect-detector | `pda_phase2_round4_20260428T0700.md` | 4847 | 10 NEW master-prompt defects (none block Phase 3) |
| anti-cheat-inspector | `anticheat_phase2_round4_20260428T0700.md` | 2958 | PASS |
| sacred-guardian | `sacred_phase2_round4_20260428T0700.md` | 2426 | PASS 10/10 |

**Scribe:** main thread per DEC-011 (4 of 5 agents returned text without saving despite Write tool present; sacred-guardian wrote its own file).

---

## 1. Overall Verdict: **READY for Phase 3**

After 4 exit-gate rounds, Phase 2 deliverables converge:

- All 3 Round 1 plan blockers (B1/B2/B3) resolved
- All 3 Round 2 residuals (RD-1/RD-2/RD-3) resolved
- All 4 BLK-009 sub-defects resolved
- All 5 user-approved D1/D6 actions complete (BLK-010 inline patches; master prompt Section 27 fast-track Fix-9/10/16/27/28/34; SPEC-INT-001; DEC-013/DEC-015/DEC-014; document-level annotation per DEC-015)
- 10/10 sacred manifest entries verified (independent canonical hash; sacred-guardian agent agreed this round)
- D2 60-68% planner contract-paraphrase defect rate is now an explicit KNOWN PROPERTY managed via DEC-015 annotation, not a defect to fix

**D7 stopping rule outcome:** auditor reports 0 new substantive plan-content spec-citation defects in Round 4 (threshold was 4). D7 NOT triggered.

---

## 2. Summary Table

| # | Auditor | Verdict | Headline |
|---|---|---|---|
| 1 | phase-exit-auditor | **READY** | All 9 checks PASS; D7 verdict 0/4; all R1+R2+R3 patches still landed; document annotation present; Fix-9/10 master-prompt + agent files confirmed; SPEC-INT-001 present |
| 2 | PVA | READY-WITH-DEVIATIONS | 2 actionable (SD-1 DEC-014, SD-5 APPLIED markers — both addressed this round); 2 carry-forward LOW deferred |
| 3 | PDA | 10 NEW defects (Defect-35..44) | None block Phase 3; Defect-37 + Defect-42 HIGH for Phase 4 (Phase 4 implementer protocol absent from master prompt + Section 8.4 contradicts DEC-015); rest queued in T-EARLY-MP |
| 4 | anti-cheat | PASS | 6 originally-VERIFIED tasks unchanged; 3 D1-patched tasks intact with `# spec:` traceability; document annotation honest; SPEC-INT-001 line numbers verified by direct spec-body inspection (lines 4117 + 5558) |
| 5 | sacred-guardian | PASS 10/10 | Hash-verification table correct this round; persona reliability concern (BLK-010.5) persists for future rounds — main-thread independent hash remains the trusted source per DEC-016 deferral |

---

## 3. Per-Audit Findings

### Audit 1 — phase-exit-auditor (`phase_2_exit_audit_round4_20260428T0700.md`)

**Verdict: READY.** 9 of 9 checks PASS:

1. Plan annotation lines 10-42 — confirmed
2. Fix-9 + Fix-10 in master prompt lines 1480-1513 — confirmed
3. 5 agent files have correct tools lines (progress-reporter, phase-exit-auditor, prompt-validator, prompt-defect-detector all have Write; spec-cartographer already had Write per DEC-011) — confirmed
4. SPEC-INT-001 in `spec_changelog.md` lines 23-47 satisfies DEC-012(b) — confirmed
5. DEC-013 + DEC-015 in `tomato_decisions.md`; DEC-014 deferred (not absent) — confirmed
6. `tomato_log.md` entries through 07:00 — confirmed
7. 3 D1-patched cards (T-IMPL-2b/4b/6a) carry `# spec:` traceability comments — confirmed
8. Zero `.py`/`.yaml` in `tomato_sandbox/` — confirmed
9. Sacred 10/10 PASS — confirmed (cross-references the sacred_phase2_round4 report)

**D7 verdict: 0 new substantive plan-content defects in Round 4 (threshold: 4). D7 NOT triggered.**

### Audit 2 — PVA (`pva_phase2_round4_20260428T0700.md`)

**Verdict: READY-WITH-DEVIATIONS.**

Substantive instructions PASS (9 items): planner output saved, DEC-015 logged, DEC-013 logged, Fix-9/10 applied, sacred via independent hash, BLK-010 12 defective tasks NOT individually patched per DEC-015, no `.py`/`.yaml` in sandbox, no sacred files modified, SPEC-INT-001 satisfies DEC-012(b).

5 approved deviations (DEC-011, DEC-013, DEC-009, DEC-015, DEC-016 deferred).

**SD-1 (MEDIUM, was actionable):** DEC-014 not standalone entry. **NOW RESOLVED:** standalone DEC-014 written 2026-04-28 with verbatim user approval quote per template.

**SD-5 (NEW LOW, was actionable):** T-EARLY-MP APPLIED markers absent. **NOW RESOLVED:** APPLIED markers added inline to Fix-9, Fix-10, Fix-16; Fix-27/28/34 documented as out-of-band fast-track in new "Out-of-band fast-tracked fixes" subsection of T-EARLY-MP.

SD-2 (4h tasks) + SD-3 (Phase 1 batch grouping rationale) — carry-forward LOW, explicitly deferred (acceptable per Round 3 PVA decisions).

SD-4 (Round 4 true-parallel fire) — RESOLVED: this PVA + 4 other audits all returned content for Round 4; SD-5-new from Round 3 closes.

### Audit 3 — PDA (`pda_phase2_round4_20260428T0700.md`)

**Verdict: 10 NEW master-prompt defects (Defect-35..44); none block Phase 3.**

| ID | Severity | Class | Where it bites |
|---|---|---|---|
| Defect-35 | MEDIUM | Missing instruction | Section 8 has no body definitions for 8.9/8.10/8.11 (Amendment 2 agents); Fix-10 patched tools but not bodies |
| Defect-36 | MEDIUM | Contradiction | Section 8.1/8.8 original tool-line blocks vs Fix-9/10 patches (supersession is generic, not section-specific) |
| Defect-37 | **HIGH** | Missing instruction | Master prompt has no Phase 4 implementer protocol matching DEC-015 (plan = scaffolding, spec body = contract). Fresh-session re-create misses it. **Phase-4-blocking; must fix before implementer subagent dispatches.** |
| Defect-38 | MEDIUM | Ambiguity | Fix-28 "1-line patch boundary" breaks down at 12+ tasks (BLK-010 sat in this gap) |
| Defect-39 | MEDIUM | Missing instruction | sacred-guardian (Section 8.5) lacks anti-hallucination guardrails (BLK-010.5 cause) |
| Defect-40 | LOW | Stale reference | Section 8 header + Section 26 still say "8 agents" — actual is 11 |
| Defect-41 | LOW | Missing instruction | DEC-014 deviation (heading-card vs checkbox) needs reconciliation note |
| Defect-42 | **HIGH** | Contradiction | Section 8.4 implementer body says "read summaries"; DEC-015 requires reading spec body. Direct contradiction. **Phase-4-blocking.** |
| Defect-43 | LOW | Missing instruction | SPEC-INT-NNN format not in Section 11.4b |
| Defect-44 | MEDIUM | Unfollowable instruction | Fix-27 requires consolidating from artifact files; Fix-10 explicitly leaves anti-cheat read-only (no Write); contradiction |

All 10 → T-EARLY-MP queue. **Defect-37 + Defect-42 should be the next 2 fast-track items before Phase 4 implementer dispatches.** None block Phase 3 (section15-encoder doesn't depend on these).

### Audit 4 — anti-cheat-inspector (`anticheat_phase2_round4_20260428T0700.md`)

**Verdict: PASS.**

- Zero `.py`/`.yaml` in `tomato_sandbox/`: PASS
- 6 originally-VERIFIED tasks (T-IMPL-4a, T-IMPL-7b, T-IMPL-8b, T-IMPL-9b, T-EARLY-MP, T-PHASE-3-PRECONDITIONS): unchanged from D2 verdict, PASS
- 3 D1-patched tasks (T-IMPL-2b IQA, T-IMPL-4b ClassifierResult, T-IMPL-6a Tier 4A routing): patches intact with `# spec:` traceability — PASS
- Document annotation honesty: no false claims, "~68%" matches D2 audit, "29 of 30 tasks" matches cumulative coverage — PASS
- Fix-9/10 work matches disk reality: agent files match master prompt — PASS
- SPEC-INT-001 line 4117 verified via `sed -n '4114,4122p'`: `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` confirmed; sum=0.97 ✓ Convention 1 — PASS
- SPEC-INT-001 line 5558 verified via `sed -n '5554,5562p'`: `[0.92, 0.04, 0.01, 0.01, 0.01, 0.01]` confirmed; sum=1.00 ✗ Convention 1 — PASS
- Section 15 test mods: vacuously PASS (Phase 3 hasn't run)
- Suppressed failures: PASS (no test files)
- Fake completion claims: PASS (no DONE markers; Round 4 noted as "next" not "done")
- Hardcoded test values: PASS (synthetic test values are spec-sourced boundaries)

**Known residual (LOW, NOT a violation):** T-IMPL-6b step 11 still says "remap applied here per T-IMPL-4a" — directly contradicts BLK-009 patch in same plan. Documented in BLK-010, NOT concealed. Per DEC-015, the document-level annotation routes Phase 4 implementer to spec body for contracts; this contradiction is therefore non-load-bearing for code shape. Skipped per DEC-015 methodology.

**Skipped per user direction:** new spec-citation sampling. The 60-68% rate is now a known property managed via DEC-015 annotation.

### Audit 5 — sacred-guardian (`sacred_phase2_round4_20260428T0700.md`)

**Verdict: PASS 10/10.**

All 10 sacred manifest entries match byte-for-byte. Zero drift across all of Phase 2.

| # | Path | Type | Result |
|---|---|---|---|
| 1 | `scripts/apin/` | Directory (316 files) | OK |
| 2 | `models/best_model.pt` | File | OK |
| 3 | `models/swin_best_model.pt` | File | OK |
| 4 | `models/model2_specialist/model2_production.pt` | File | OK |
| 5 | `data/specialist/model3/split_indices.json` | File | OK |
| 6 | `app/config.py` | File | OK |
| 7 | `data/metadata/source_map.csv` | File | OK |
| 8 | `models/specialist/ladinet_phase1_heads.pt` | File | OK |
| 9 | `scripts/model3_training/checkpoints/model3_production_v3.pt` | File | OK |
| 10 | `models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt` | File | OK |

This Round 4 sacred-guardian invocation produced correct hash output (no persona drift, no fabricated phase names). Reliability concern from BLK-010.5 + post-D6 hash hallucination remains a known risk but did NOT manifest this round. **DEC-016 (deferred): main-thread independent canonical hash continues to be the trusted source; sacred-guardian agent is treated as a corroborator only.**

---

## 4. Open Blockers Going into Phase 3

| BLK | Severity | Status | Phase 3 impact |
|---|---|---|---|
| BLK-001 | — | RESOLVED (session restart) | none |
| BLK-002, 003, 004, 005 | — | RESOLVED via DEC-012 | none |
| BLK-006, 007, 008 | LOW | OPEN; deferred to Phase 4 implementer | none — implementer reads spec body per DEC-015 |
| BLK-009 (4 sub-defects) | — | All PATCHED in Round 2 | none |
| BLK-010 (5 sub-defects) | — | 10.1/10.2/10.3 PATCHED via D1; 10.4 retroactively scribed; 10.5 mitigated by DEC-016 deferral | none |

**No OPEN blockers prevent Phase 3 entry.**

---

## 5. Cumulative defect counts (Phase 0+1+2)

| Category | Total | Notes |
|---|---|---|
| BLKs filed | 10 | All RESOLVED, PATCHED, or mitigated |
| Master-prompt defects (PDA) | 44 | Phase 0: 1-8 (8); Phase 1: 9-18 (10); Phase 2 R1: 19-26 (8); Phase 2 R3: 27-33 (7); Phase 2 R4: 35-44 (10). One number gap (Defect-34 was the planner-spec-body rule, applied in D6). All queued in T-EARLY-MP except 6 fast-tracked (Fix-9, 10, 16, 27, 28, 34). |
| DECs logged | 15 | DEC-001..015. Plus DEC-014 (just written) = 15 total active. DEC-016 deferred. |
| Phase exit gate fires | 9 | Phase 0 (1), Phase 1 (1), Phase 2 (4 rounds + 1 retroactive scribe + 1 D2 audit) |
| Audit subagent invocations | ~26 | phase-exit-auditor ×5, PVA ×4, PDA ×4, anti-cheat ×4, sacred-guardian ×5 + main-thread independent hash ×3 |
| Sacred drift events | 0 | All 10 entries match across all main-thread independent verifications |
| `.py`/`.yaml` files in `tomato_sandbox/` | 0 | Read-only constraint honored |

---

## 6. What Phase 3 will do (preview)

Per master prompt section 4 + DEC-012 + Fix-16:

1. Dispatch `section15-encoder` subagent.
2. Encoder reads Section 15's 135 scenarios from spec body.
3. For each scenario, encoder produces a pytest test function in `tomato_sandbox/tests/integration/test_section15_*.py`.
4. Each test imports `assign_tier()` from `tomato_sandbox/tier/tier_assignment.py` — but T-IMPL-5a hasn't run yet, so all 135 tests should FAIL with ImportError or NotImplementedError. That's expected; Phase 3 produces failing tests; Phase 4 makes them pass.
5. Encoder applies Fix-16 conflict-resolution rule: S1.1 v3 priors = `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` per SPEC-INT-001 + line 4117 authoritative; line 5558 typo not used.
6. Encoder reports back: count of encoded tests (must = 135), import contract written to `.claude/import_contract.md`, failure output proving all 135 fail with expected modes.
7. `/tomato-phase-exit 3` runs; if READY, Phase 4 starts.

**Phase 3 entry preconditions (per `tomato_plan.md` lines 39-46):**
1. Phase 2 user-approved → pending this report's user approval.
2. T-IMPL-5a complete → **NOT met** (no implementation yet).
3. T-IMPL-5b complete → **NOT met**.
4. Master prompt Fix-16 applied → **MET** (in Section 27 + section15-encoder.md).
5. `spec_changelog.md` BLK-004 entry → **MET** (SPEC-INT-001).
6. BLK-009 sub-defects PATCHED → **MET**.

**Note:** preconditions 2-3 (T-IMPL-5a + 5b) are Phase 4 work, but the plan's gate logic requires them BEFORE section15-encoder dispatches. This is an inversion that will need user direction at Phase 3 entry: either run T-IMPL-5a+5b as the FIRST Phase 4 tasks before encoder fires, or relax the precondition. Not a Phase 2 problem; flagged for Phase 3 entry decision.

---

## 7. Recommended user actions before Phase 3 begins

**Required (none — Phase 2 closes cleanly):**
- (no further blocker resolutions needed)

**Strongly recommended (master-prompt fixes for Phase 4 readiness):**
- Apply Defect-37 (Phase 4 implementer protocol from DEC-015 → master prompt Section 4 Phase 4) before Phase 4 dispatches the implementer subagent.
- Apply Defect-42 (Section 8.4 implementer body contradiction with DEC-015) — change "read spec section summaries" to "read spec body for code-shape decisions; summaries for context only".
- Both are HIGH per PDA Round 4. Neither blocks Phase 3 (section15-encoder doesn't dispatch implementer).

**Optional (T-EARLY-MP queue, defer):**
- Defect-35, 36, 38, 39, 40, 41, 43, 44 (rest of PDA Round 4 findings).
- DEC-016 (sacred-guardian shell rewrite to eliminate persona drift).

---

## 8. Console Summary

```
PHASE        : 2 (Planning) — Round 4 exit gate
VERDICT      : READY for Phase 3
GATE FIRES   : 5/5 audit files on disk; consolidation reduced over real files (no memory)
SACRED       : 10/10 PASS, 0 drift
ANTI-CHEAT   : PASS (no fabrications; D1 patches intact; SPEC-INT-001 verified)
D7 STOPPING  : NOT TRIGGERED (0 new substantive plan defects vs threshold 4)
PHASE 3 GATE : 4/6 preconditions MET (Fix-16, SPEC-INT-001, BLK-009, user-approval-pending)
               2/6 NOT MET: T-IMPL-5a + 5b (Phase 4 work — see Section 6 of this report)

ROUND 4 OPEN ITEMS RESOLVED THIS TURN:
  PVA SD-1 (DEC-014 standalone)               WRITTEN this turn
  PVA SD-5 (T-EARLY-MP APPLIED markers)        ADDED this turn (Fix-9/10/16 inline + Fix-27/28/34 fast-track section)
  Round 4 consolidation report                 WRITTEN this turn

CUMULATIVE PROJECT METRICS:
  BLKs                                          10 (all RESOLVED/PATCHED/mitigated)
  Master-prompt defects                         44 (6 fast-tracked + 38 in T-EARLY-MP)
  DECs                                          15 (DEC-001..015 + DEC-014 + DEC-016 deferred)
  Sacred drift events                           0
  .py/.yaml in tomato_sandbox/                  0

NEXT STEP : STOP. Await user approval to enter Phase 3.
            User decision needed on Phase 3 entry preconditions 2-3 (T-IMPL-5a + 5b inversion).
```

---

*End of Phase 2 Round 4 exit gate consolidation. Generated 2026-04-30 from 5 real artifact files on disk. No memory consolidation. The user-approved D2 stopping rule (≥40% defect rate triggers methodology discussion) was honored on 2026-04-28 06:30 and resolved via DEC-015 (document-level annotation methodology) on 2026-04-28 07:00; D7 stopping rule (Round 4 ≥4 new substantive defects = STOP, escalate) NOT triggered (Round 4 = 0 new substantive defects).*
