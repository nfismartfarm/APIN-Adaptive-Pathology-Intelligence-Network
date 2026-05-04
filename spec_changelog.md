# Spec Changelog — `tomato_3_signal_system.md`

The spec is locked. This file records spec modifications, which require explicit user approval. Format per entry:

```
## SPEC-CHG-NNN [YYYY-MM-DD] <title>
- Spec section affected: ...
- Original text: <verbatim>
- New text: <verbatim>
- Reason: ...
- User approval (verbatim quote): ...
- Tasks affected: T-NNN, T-MMM (require re-audit)
```

---

## Status: zero spec changes as of 2026-04-27 17:30

The two path corrections in spec Section 2.6 (`model2_production.pt` actual location at `models/model2_specialist/`; `ladinet_phase1_heads.pt` actual location at `models/specialist/` not inside `ladinet_checkpoints/`) are NOT logged here because they are spec-text-vs-disk-reality discrepancies recorded in `.claude/sacred_manifest.json` under `spec_section_2_6_path_corrections`. They do not modify the spec text. If the user later wants the spec text itself amended to match disk reality, that will be logged here.

---

## SPEC-INT-001 [2026-04-28] BLK-004 Defect-15.1 — S1.1 v3 priors vector intra-spec conflict resolution

**Note: This is an INTERPRETATION entry, not a text-change entry.** The spec text at line 5558 is left unmodified pending future spec author review. This entry establishes the authoritative interpretation that downstream tooling (section15-encoder in Phase 3) must follow. Format follows the SPEC-INT-NNN variant per Defect-23 (queued in T-EARLY-MP; format below is consistent with what Defect-23 will codify).

- **Spec section affected:** Section 15.3 (Tier 1 scenarios), specifically scenario S1.1
- **Defect ID:** BLK-004 / Defect-15.1 (also referenced in master prompt Section 27 Fix-16 example)
- **Conflicting locations:**
  - **Line 4117** (S1.1 scenario body): `v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03` — sum = 0.97; consistent with Convention 1 (`tomato_probs` sum to `1 − chilli_leakage` = 0.97)
  - **Line 5558** (test code snippet within Section 15): `probs=[0.92, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03` — sum = 1.00; **violates** Convention 1
- **Authoritative location:** **Line 4117** (scenario body)
- **Non-authoritative location:** Line 5558 (treated as typo; test-code snippet is example/illustration, not the scenario contract)
- **Why line 4117 wins:**
  1. Convention 1 (Section 15.2 line ~4098) explicitly states `tomato_probs` sum to `1 − chilli_leakage`. Line 4117's vector `[0.89, ...]` satisfies this; line 5558's `[0.92, ...]` violates it.
  2. Section body (scenario block at line 4117) is the authoritative contract per master prompt Fix-16 (added 2026-04-28 in Section 27): *"When multiple conflicting values exist for a single scenario field at different spec locations, treat the scenario body text as authoritative."*
  3. Independent verification: main thread ran `sed -n '4115,4125p'` and `sed -n '5555,5565p'` on `tomato_3_signal_system.md` on 2026-04-27 to confirm both literal vectors. The narrative reference at line ~4098 ("S1.1's v3 vector sums to 0.97") agrees with line 4117, not line 5558.
- **BLK reference:** BLK-004 Defect-15.1 (RESOLVED via DEC-012 option A)
- **DEC reference:** DEC-012 (user-approved 2026-04-27 23:45)
- **User approval (verbatim quote from 2026-04-27 23:45 user message):** *"BLK-004 Defect-15.1 (S1.1 v3 vector conflict): line 4117 (scenario body) authoritative; line 5558 is a typo. spec_changelog.md entry to be written before Phase 3 begins. Encoder uses [0.89, 0.04, 0.01, 0.01, 0.01, 0.01]."*
- **Tasks affected:**
  - T-IMPL-3a (Signal A wrapper): BLK-004 cited in task card as the S1.1 priors reference vector.
  - T-IMPL-5a (tier_assignment.py): no direct effect (S1.1 is a scenario, not a rule definition).
  - **Phase 3 section15-encoder (downstream):** MUST encode S1.1 with `probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]`. Agent definition patched per Fix-16 in `.claude/agents/section15-encoder.md` and master prompt Section 27.
- **Author subagent:** main-thread scribe per DEC-011 (on 2026-04-28; verification via direct spec-body `sed` reads on 2026-04-27)
- **Spec line numbers cited:** 4098 (Convention 1 narrative), 4117 (S1.1 scenario body — authoritative), 5558 (test code snippet — typo)
- **Status:** RESOLVED via interpretation. Spec text unchanged. Future spec author may correct line 5558 to match line 4117; if so, that becomes a SPEC-CHG-NNN entry replacing this SPEC-INT-001.

---

## SPEC-INT-002 [2026-05-01] Section 15 subsection-vs-body conflicts (7 scenarios) — scenario body authoritative

**Note: This is an INTERPRETATION entry, not a text-change entry.** Spec text in Section 15 is left unmodified. This entry establishes the authoritative interpretation that downstream tooling (already applied by `section15-encoder` in Phase 3 on 2026-04-30) followed.

- **Spec sections affected:** Section 15.6 (Tier 3B subsection), Section 15.7 (Tier 3C subsection), Section 15.8 (Tier 3D subsection), Section 15.12 (Boundary subsection).
- **Pattern:** Section 15 organizes 135 scenarios into subsections by INTENDED outcome (e.g., 15.6 = "Tier 3B scenarios"). However, 7 specific scenario bodies inside these subsections describe inputs that, when run through the actual rule chain (Section 14), produce a tier OTHER than the subsection's heading. This is intentional per Section 15.1: *"grouped by intended outcome, not by input pattern"* — the subsections are content organization, not rule-chain classification.
- **Resolution rule applied:** scenario body wins over subsection heading. Same principle as SPEC-INT-001 (S1.1 case), now extended to Section 15 generally per master prompt Section 27 Fix-16.
- **The 7 scenarios:**

| # | Scenario | Subsection | Subsection-implied tier | Body-actual tier | Rule fired | Spec lines | Why body wins |
|---|---|---|---|---|---|---|---|
| 1 | **S3B.4** | 15.6 (Tier 3B) | 3B | **4A** | `4` | 4487-4493 | Classifier max=0.16 < 0.45 → Rule 4 fires before Rule 5 (3B). Subsection organizational; "included to show flat-distribution cases route to 4A, not 3B" |
| 2 | **S3C.8** | 15.7 (Tier 3C) | 3C | **2** | `8c` | (per body) | psv_reliability=0.40 exactly; Rule 3 strict `< 0.40` does NOT fire; Rule 8c default → Tier 2 |
| 3 | **S3C.9** | 15.7 (Tier 3C) | 3C | **4A** | `catch_all_low_confidence` | (per body) | chilli_leakage=0.40 exactly; Rule 3 strict `> 0.40` does NOT fire; falls to Rule 9 catch-all |
| 4 | **S3C.12** | 15.7 (Tier 3C) | 3C | **4A** | `catch_all_low_confidence` | (per body) | chilli_leakage=0.30 boundary trap (Tier 1 chilli `< 0.20` fails; Tier 2 chilli `< 0.30` strict fails; Rule 3 chilli `> 0.40` fails); Rule 9 catches |
| 5 | **S3D.5** | 15.8 (Tier 3D — DEGRADED IQA cap) | 3D | **3A** | `6` | (per body) | prediction_set_size==2; Rule 6 (set==2 → 3A) fires before sub-rule 7a (DEGRADED IQA → 3D cap). Set-size rules precede IQA cap |
| 6 | **S3D.7** | 15.8 (Tier 3D — DEGRADED IQA cap) | 3D | **3B** | `5` | (per body) | prediction_set_size==3; Rule 5 (set≥3 → 3B) fires before sub-rule 7a/8a (DEGRADED IQA → 3D cap). Same precedence reason as S3D.5 |
| 7 | **SB.11** | 15.12 (Boundary scenarios) | (no subsection-implied tier) | **4A** | `5` (empty-set sub-rule) | 5200-5210 | Classifier max=0.50 ≥ 0.45 (so Rule 4 doesn't fire), but conformal set is empty → Rule 5 empty-set sub-rule → Tier 4A. The Walk text says "max<0.45 → Rule 4" but the scenario body sets max=0.50 |

- **Defect ID:** BLK-004 Defect-15.3 class (subsection-vs-body conflicts; the broader pattern that BLK-004 Defect-15.1 was a special case of).
- **Authoritative source:** scenario body in each case (input fields + expected tier_label + expected rule_id_fired).
- **Non-authoritative:** subsection heading text (e.g., "All Tier 3B scenarios share..."). Subsection groupings are content organization per Section 15.1.
- **BLK reference:** BLK-004 Defect-15.3 (RESOLVED via DEC-012 option A; encoder protocol via Fix-16 in master prompt Section 27).
- **DEC reference:** DEC-019 user message specifying SPEC-INT-002 single-entry batch documentation; Phase 3 encoder dispatch under DEC-017 + DEC-018; encoder applied the rule autonomously per Fix-16.
- **Encoder enforcement (already done):** the 13 test files in `tomato_sandbox/tests/integration/test_section15_*.py` (created Phase 3, 2026-04-30) encode each of these 7 scenarios with assertions matching the body-actual tier/rule. Each test has an inline NOTE comment block documenting the subsection-vs-body conflict and citing this SPEC-INT-002 / Fix-16 / BLK-004 Defect-15.3 chain.
- **Verified by main-thread grep against test files (2026-05-01):** all 7 assertions match the table above.
- **Tasks affected:**
  - Phase 3 section15-encoder: APPLIED 2026-04-30. Tests encoded with body-wins decisions.
  - Phase 4 T-IMPL-5a (`tier_assignment.py`): no special handling needed. The rule chain (Rules 1-9, 7a/7b/7c, 8a/8b/8c) implemented per Section 14 will produce the body-actual tiers naturally — SPEC-INT-002 is documenting that the SCENARIOS are correct as encoded, not asking for rule-chain changes.
  - Phase 4 implementer: when reading Section 15 spec body for context, ignore the subsection heading taxonomy when it conflicts with scenario body content.
- **User approval (verbatim, 2026-05-01 message Q4):** *"Body-wins decisions: single SPEC-INT-002 entry covering all 7 cases. Same root cause, same resolution rule, batch documentation. Format: header explaining the pattern + table listing scenario, subsection, body tier, rule fired, spec lines. Reference Fix-16 / BLK-004 Defect-15.3. Each case has its inline comment in the test file already; SPEC-INT-002 is the audit-trail anchor."*
- **Author subagent:** main-thread scribe per DEC-011 (encoder applied the rule autonomously; this entry is the audit-trail anchor).
- **Status:** RESOLVED via interpretation. Spec text unchanged. Encoder output is the binding artifact; this entry is the audit-trail anchor for the 7-case batch. Future spec author may correct subsection headings or add explicit per-scenario "exception" markers; if so, that becomes a SPEC-CHG-NNN entry.


## SPEC-INT-003 [2026-05-03] S17.5 example coverage_pct drafting inconsistency

- **Spec section:** S17.5 (Severity for multi-class sets), lines 6015-6032.
- **Inconsistency:** Spec example at lines 6022-6025 shows different `coverage_pct` per class:
  ```json
  "grade_per_class": [
    { "class": "foliar", "grade": "moderate", "coverage_pct": 11.2 },
    { "class": "septoria", "grade": "mild", "coverage_pct": 4.8 }
  ]
  ```
  This contradicts the normative text at S17.2:5964 — *"severity is a PSV-only computation"* — which mandates a single PSV computation with shared inputs (singular `disease_coverage_pct`).
- **Resolution per main-thread spec read (Phase 5b):** **(α) interpretation confirmed:** the same `coverage_pct` value is reused across all classes in `grade_per_class`. Only the per-disease threshold lookup (S17.3 tabular thresholds) varies per class. The S17.5 example's different per-class values (11.2 vs 4.8) are **drafting noise**, not a contract — the author wrote two visually-distinct numbers for example readability without realizing it implied per-class PSV recomputation (which contradicts S17.2/S17.3/S17.4 normative text).
- **Implementation conformance:** T-AUDIT-5b-fix per DEC-050 implements the per-class loop with shared `coverage_pct`. Each entry in `grade_per_class` echoes the same PSV-computed value; only the `grade` differs per class via threshold lookup. Verified by anti-cheat Check 9C (`test_grade_per_class_same_coverage_pct_for_all_classes`).
- **Spec body update target (deferred to T-EARLY-MP):** the S17.5 example should be amended to show consistent `coverage_pct` across classes (e.g. both 11.2) with grade differing only by threshold table. This eliminates the documentation/example inconsistency.
- **Pattern:** this is the third SPEC-INT entry; the project's recurring-cause pattern is "spec examples drift from spec normative text." SPEC-INT-001 was a v3 priors vector intra-spec conflict; SPEC-INT-002 was Section 15 scenario-body-vs-subsection conflicts. SPEC-INT-003 follows the same pattern: examples were authored to be illustrative, not normative.
- **No implementation pause required.** No further action needed beyond eventual spec body cleanup.


## SPEC-INT-004 [2026-05-04] S20.5 startup-step filename drifts (steps 4 + 8) — sacred + DEC ledger authoritative for paths

- **Spec section:** S20.5 (Startup sequence), lines 6556-6575.
- **Drift 1 — step 4 (HIGH catastrophic risk if naive):** S20.5 line 6563 says *"Load v3 model weights from `model2_production.pt` to GPU"*. The literal filename `model2_production.pt` is the brassica Model 2 checkpoint (sacred — see manifest entry `models/model2_specialist/model2_production.pt`). Loading it into the tomato pipeline would catastrophically misclassify every tomato input as a brassica disease class. **Correct path** per `.claude/sacred_manifest.json`: `scripts/model3_training/checkpoints/model3_production_v3.pt` (also sacred; this IS the v3 tomato model). Component C / DEC-054 honored the sacred manifest; spec literal was overridden.
- **Drift 2 — step 8 (LOW):** S20.5 line 6567 says *"Load conformal calibration from `tomato_calibration.json`"*. Implementation actually consumes `tomato_sandbox/phase_f0_calibration/conformal_tau.json` per DEC-040 + DEC-045. The spec's filename predates Phase 4's per-artifact JSON layout (DEC-045 split calibration into 4 files). Component C honored the implementation path.
- **Resolution per main-thread spec read (Phase 6 Component C dispatch):** sacred manifest + DEC ledger are AUTHORITATIVE for paths; spec body is AUTHORITATIVE for semantics (load order, fail-fast, warmup, etc.). When spec literals drift from sacred/DEC paths, sacred + DEC win. Documented in DEC-054 Decision 2-4.
- **Implementation conformance (verified Phase 6 Component C anti-cheat Check 10A):** `model_loaders.py` uses `model3_production_v3.pt` (NOT `model2_production.pt`); inline comment notes the correction. Conformal load path uses `conformal_tau.json` per DEC-045.
- **Spec body update target (deferred to T-EARLY-MP):** S20.5 step 4 should be amended to read `model3_production_v3.pt`; step 8 should be amended to reference the per-artifact JSON layout. This eliminates the documentation/example inconsistency.
- **Pattern:** fourth SPEC-INT entry. SPEC-INT-001 was v3 priors vector intra-spec conflict; SPEC-INT-002 was Section 15 scenario-body-vs-subsection conflicts; SPEC-INT-003 was S17.5 example coverage_pct drafting noise. SPEC-INT-004 follows the same pattern: spec literals drifted from project artifact ground truth.
- **No implementation pause required.** No further action needed beyond eventual spec body cleanup.
