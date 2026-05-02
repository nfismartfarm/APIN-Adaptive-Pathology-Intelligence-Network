# tomato_plan.md — Tomato 3-Signal Sandbox Implementation Task List

**Spec:** `tomato_3_signal_system.md` (32 sections + declared appendices; 8756 lines)
**Dependency graph version:** `.claude/spec_dependency_graph.md` (produced 2026-04-27 Phase 1 exit)
**Plan produced:** 2026-04-27 by task-planning specialist (Phase 2 output)
**Implementation target:** `tomato_sandbox/` only — all other paths are SACRED

---

## How to use this plan

**[ADDED 2026-04-28 per DEC-015 — read this BEFORE relying on any task card content.]**

This document is **authoritative** for:

- Build order (sequence of T-IMPL tasks)
- Spec section pointers (which section governs each task)
- File targets in `tomato_sandbox/`
- Task dependencies
- Acceptance criteria pointers (Section 15 for `tier_assignment`, etc.)
- BLK resolutions referenced via DEC-012

This document is **NOT authoritative** for:

- Function signatures, dataclass field names, threshold values, rule numbers, dimension lists, IQA dimension names, or any verbatim spec contract detail.
- Phase 2 planner produced contract paraphrases at **~68% defect rate** against spec body (verified across 29 of 30 tasks via D2 anti-cheat audit on 2026-04-28; see `tomato_progress_reports/anticheat_d2_full_20260428T0700.md` and BLK-009/010). Contract paraphrases in this document are **not corrected** and should not be relied on.

## Phase 4 implementer protocol

For each task in this plan:

1. Read the spec body for the primary spec section listed in the task card.
2. Note any cross-referenced sections; read those too if they define inputs/outputs the task consumes.
3. Read DEC-012 for any BLK resolutions that apply to the task's spec sections.
4. Implement the file at the location listed in the task card.
5. Cite spec line numbers in code comments per master prompt rule (Fix-34 / Section 27).
6. Mark the task DONE with timestamp at the bottom of its card.

The task card body is **supplementary navigation**. The spec body is **the contract**.

**Note on the 3 patched task cards (T-IMPL-2b, T-IMPL-4b, T-IMPL-6a):** these were corrected via D1 inline patches on 2026-04-28 and carry `# spec: <section>.<sub> lines <N>` traceability. Their contract details are spec-verbatim; you may rely on them. All other task cards' contract details are paraphrases — use the protocol above.

---

## BLK Resolutions Baked Into This Plan (DEC-012, Option A for all)

| BLK | Decision | Impact on tasks |
|---|---|---|
| BLK-002 | Port 8767 is authoritative (Sandbox Directive wins over S1.3/S2.3 prose) | All server tasks use 8767 |
| BLK-003 | HTTP-client only — no `apin` library import anywhere in `tomato_sandbox/` | T-IMPL-2a validator, T-IMPL-7b routing |
| BLK-004 | S1.1 v3 priors vector = `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` (line 4117 in spec body, not line 5558 header) | T-IMPL-3a Signal A wrapper |
| BLK-005 | T-IMPL-5a derives `tier_rules.yaml` schema from Section 14 prose with traceability comments | T-IMPL-5a is a schema-decision task |

---

## Sacred Files — Implementer Must Never Touch

The following are outside `tomato_sandbox/` and must not be modified by any Phase 4 code:

- `models/model2_specialist/model2_production.pt`
- `scripts/model3_training/checkpoints/model3_production_v3.pt`
- `models/specialist/ladinet_phase1_heads.pt`
- `models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt`
- `app/config.py`, `app/model.py`
- `data/metadata/source_map.csv`
- `scripts/apin/` (entire directory)

---

## Phase 3 Entry Preconditions

Phase 3 (section15-encoder subagent) may not begin until ALL of the following are TRUE:

1. **Phase 2 user-approved.** (Round 4 exit gate verdict READY; user approved 2026-04-30.)
2. **Phase 3 produces FAILING tests by design** (CORRECTED 2026-04-30 per DEC-017): T-IMPL-5a and T-IMPL-5b are Phase 4 work, NOT Phase 3 preconditions. Phase 3 deliverable is **135 pytest tests in `tomato_sandbox/tests/integration/test_section15_*.py` that all fail with `ImportError` or `NotImplementedError`**, plus an import contract documenting the expected `assign_tier()` signature for Phase 4. This matches master prompt Section 8.3 section15-encoder rule: *"produce 135 tests that all FAIL with expected failure modes."* Phase 4 makes them pass. Earlier draft of preconditions 2-3 inverted this; corrected per DEC-017.
3. **Defect-16 resolved (master-prompt patch applied):** The section15-encoder subagent spec must contain the intra-spec conflict resolution rule. Specifically: scenario body text is authoritative; line 4117 v3 priors vector `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` is correct; line 5558 typo is non-authoritative. Tracked in T-EARLY-MP Fix-16. **[STATUS: APPLIED 2026-04-28 in master prompt Section 27 + section15-encoder.md.]**
4. **`spec_changelog.md` BLK-004 entry written** (per DEC-012 condition (b)). The entry must record the line 5558 typo resolution per the SPEC-INT-NNN format, citing line 4117 as authoritative. **[STATUS: APPLIED 2026-04-28 as SPEC-INT-001.]**
5. **BLK-009 sub-defects all resolved:** Defect-9.1 (TTA signature), Defect-9.2 (remap location annotation), Defect-9.3 (chilli_leakage threshold) all PATCHED in plan and dependency graph. **[STATUS: PATCHED 2026-04-28 (Round 2) and verified by Round 4 phase-exit-auditor.]**

These five conditions are the Phase 3 gate. No partial satisfaction is acceptable (HB-2: any scenario failure blocks v1 launch). All five are MET as of 2026-04-30.

---

## V2 Features — NOT Tasked (Section 30 exclusion list)

The following appear in the spec but are explicitly v2 or deferred; no task covers them:

- Chilli crop classification (S30: "chilli not implemented in v1; unified server returns 501")
- Multi-agronomist consensus for critical queue cases (S23 limitation 3)
- WCAG 2.1 AA compliance (S19 limitation 2)
- Localization / i18n (S19 limitation 3)
- Blue-green deployment (S28 limitation 3)
- DR plan / multi-region replication (S28 limitation 5)
- PSV Cython/C++ port (S30 medium)
- Per-class GradCAM++ for multi-class sets (S30 medium)
- SDK generation from OpenAPI (S27 limitation 2)
- Postgres migration for queue (S23 cross-ref S24.8)
- Random-sample spot-check coverage measurement (SB-4; Stage 3 precondition, not Phase 4 code)

---

## Special Tasks

### T-EARLY-MP — Master Prompt Defect Remediation
**Spec sections covered:** Meta (no implementation)
**Dependencies:** None (run before any implementation)
**Files to create or modify:** `tomato_master_prompt.md` (append correction block per DEC-007 append-only rule)
**Estimated effort:** 1h

Ordered fix list (25 items, GLOBALLY severity-sorted HIGH → MEDIUM → LOW per Auditor RD-3 / Defect-24's own rule). Each item cites its source defect (Fix-N IDs preserved for traceability across phases; sequential position numbers reflect global execution order).

### HIGH-severity fixes (apply first; Fix-16 is the last HIGH, all others are unordered within HIGH except Phase-3-blockers prioritized)

1. **Fix-1 (Defect-1, HIGH, Phase 0):** Section 2 sacred manifest — replace enumerated file paths with "see spec Section 2.6 (authoritative); paths verified against disk reality at manifest construction time." Remove stale `model2_production.pt` and `ladinet_checkpoints/` paths; add `scripts/model3_training/checkpoints/model3_production_v3.pt` and `sp_lora_epoch13_f10.9113_PRESERVED.pt`.
2. **Fix-2 (Defect-2, HIGH, Phase 0):** Section 8 subagent 8.5 directory-hash algorithm — replace verbal description with exact algorithm: `json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")` and `relative_path = relpath.replace(os.sep, "/")`. Reference the `directory_hash_algorithm_canonical` field in `.claude/sacred_manifest.json`.
3. **Fix-3 (Defect-3, HIGH, Phase 0):** Section 23.1 deny list — add `Edit(scripts/model3_training/**)` to deny list. (Phase 0 already patched `.claude/settings.local.json`; master prompt text still missing this entry.)
4. **Fix-9 (Defect-9, HIGH, Phase 1):** Master prompt section 8.1 (spec-cartographer agent definition) — change `tools: Read, Glob, Grep` to `tools: Read, Glob, Grep, Write`. Add scope-restriction note: *"Write tool usage: restricted to `.claude/spec_summaries/section_NN.md` files only."* This codifies the DEC-011 inline patch. **[APPLIED 2026-04-28 in master prompt Section 27 fast-track block; spec-cartographer.md already had Write per DEC-011 — no agent-file change needed.]**
5. **Fix-10 (Defect-10, HIGH, Phase 1):** Sweep all agent definitions whose body describes saving artifacts and add `Write` to their tools list. Specifically: master prompt sections 8.8 (progress-reporter), 8.9 (phase-exit-auditor), 8.10 (prompt-validator), 8.11 (prompt-defect-detector). For each, add scope-restriction note matching the file pattern they save to (`tomato_progress_reports/<artifact>_<timestamp>.md`). **[APPLIED 2026-04-28 in master prompt Section 27 fast-track block; 4 agent files patched: progress-reporter.md, phase-exit-auditor.md, prompt-validator.md, prompt-defect-detector.md — Write added to each tools line; verified by Round 4 PVA + anti-cheat.]**
6. **Fix-11 (Defect-11, HIGH, Phase 1):** Batch sizing — add instruction: "No single batch may cover more than 6 spec sections. Batches of 8+ sections produce summaries that exceed context limits and cause truncation. Split large batches before dispatching."
7. **Fix-12 (Defect-12, HIGH, Phase 1):** Skills authoring — add "skill authoring is a Phase 1 deliverable; skills at `.claude/skills/` must be substantively authored (not placeholder) before Phase 1 exit."
8. **Fix-19 (Defect-19, HIGH, Phase 2):** Master prompt section 8.2 (planner agent body) — mandate the checkbox output format from Section 4.3 task 5: *"Each task card begins with `- [ ] Task ID: T-NNN` and uses indented bullet fields. Do not use `###` heading cards."*
9. **Fix-20 (Defect-20, HIGH, Phase 2):** Master prompt sections 4.3 task 5 + 8.2 — add 9-column summary table requirement: *"Append a summary table at the end of `tomato_plan.md` with columns: Task ID | Title | Owner subagent | Prerequisites | Spec sections | Files to create | Acceptance criteria | Estimated complexity | Priority."*
10. **Fix-16 (Defect-16, HIGH — BLOCKS Phase 3):** Section15-encoder subagent spec — add intra-spec conflict resolution rule: *"When multiple conflicting values exist for a single scenario field at different spec locations, treat the scenario body text as authoritative; report the conflict in `tomato_blockers.md` with all locations and values quoted, and request confirmation before encoding that scenario."* Specifically: S1.1 v3 priors vector = `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` (line 4117 authoritative; line 5558 typo). **[Last HIGH item by convention — most directly Phase-3-blocking. Phase-3-critical preamble below lists which HIGH items must complete before Phase 3.]** **[APPLIED 2026-04-28 in master prompt Section 27 fast-track block; section15-encoder.md patched with intra-spec conflict rule including S1.1 example; spec_changelog.md SPEC-INT-001 entry written for the BLK-004/Defect-15.1 case per DEC-012 condition (b).]**

### MEDIUM-severity fixes (apply after all HIGH)

11. **Fix-4 (Defect-4, MEDIUM, Phase 0):** Section 4 phase-exit steps — add "run `/tomato-phase-exit <N>` before STOP" to all phase-exit instructions in sections 4, 9, and 10.
12. **Fix-5 (Defect-5, MEDIUM, Phase 0):** Section 26 acknowledgment requirement — add to phase-exit-auditor Phase 0 checklist: "acknowledgment text produced at session start; pasted as evidence in Phase 0 exit report."
13. **Fix-6 (Defect-6, MEDIUM, Phase 0):** Section 23.1 allow list — add clause acknowledging conda environments as valid virtual environments; reference `T-EARLY-VENV`.
14. **Fix-13 (Defect-13, MEDIUM, Phase 1):** Dependency-graph subagent — add instruction to verify the graph is buildable from summaries alone (no spec re-reads). **[ADDITION: also add "When summarizing a section that defines a function with input/output index spaces, quote the function signature verbatim and the dataclass field names verbatim, do not paraphrase index-space descriptions" — prevents BLK-009 / Defect-9.2 class of error.]**
15. **Fix-14 (Defect-14, MEDIUM, Phase 1):** Phase-exit auditor batch — add time-budget guidance: "Allow 2h for full Phase 1 audit; do not compress to under 45 minutes."
16. **Fix-15 (Defect-15, MEDIUM, Phase 1):** PVA gap-4 batch-grouping rationale — add to PVA template: "Batch grouping rationale must be stated; if omitted, audit flags it as a gap."
17. **Fix-21 (Defect-21, MEDIUM, Phase 2):** Master prompt section 4.3 Phase 3 entry — replace "Entry: Phase 2 complete and user approved" with explicit 5-condition gate checklist (Phase 2 approved, T-IMPL-5a complete, T-IMPL-5b complete, master-prompt Fix-16 applied, spec_changelog.md contains all required BLK entries).
18. **Fix-22 (Defect-22, MEDIUM, Phase 2):** Master prompt section 8.2 — change effort enumeration from "(1h, 2h, 4h)" to "(1h, 2h, 3h — maximum 3h; split larger tasks)."
19. **Fix-23 (Defect-23, MEDIUM, Phase 2):** Master prompt section 11.4b — add SPEC-INT-NNN interpretation-only entry variant for cases where spec text is NOT changed but an authoritative interpretation is recorded (BLK-004 case). Required fields: Conflicting locations / Authoritative location / Non-authoritative location / BLK reference / Defect ID / Author subagent.
20. **Fix-24 (Defect-24, MEDIUM, Phase 2):** Master prompt section 8.2 — add severity-ordering rule for fix lists: *"For T-EARLY-MP and similar fix-list tasks, order items HIGH first, then MEDIUM, then LOW. Within HIGH, items labeled 'BLOCKS Phase N' precede other HIGH items."*

### LOW-severity fixes (apply last)

21. **Fix-7 (Defect-7, LOW, Phase 0):** Section 3 spec line count — remove hardcoded "8683 lines" reference; replace with "spec is authoritative; current file is 8756 lines including G19/G20 additions."
22. **Fix-8 (Defect-8, LOW, Phase 0):** Section 2 sandbox directive — add exception clause for project-management files (see DEC-007 / Phase 0 step 11 list).
23. **Fix-17 (Defect-17 + Defect-18, LOW, Phase 1):** Minor spec-changelog hygiene — add instruction to spec-changelog: "Each entry must include (a) the defect ID, (b) the spec line number changed, and (c) the author subagent."
24. **Fix-25 (Defect-25, LOW, Phase 2):** Master prompt section 11.4 (DEC format) — add multi-BLK note: *"For multi-BLK DECs, repeat `Spec section:` per section and add `Resolves BLKs: BLK-NNN, BLK-MMM, ...` field."*
25. **Fix-26 (Defect-26, LOW, Phase 2):** Master prompt section 4.3 Phase 2 exit — extend to: *"Exit: (1) plan is comprehensive; (2) T-PHASE-3-PRECONDITIONS contains all DEC-N-required gate conditions; (3) all BLKs surfaced during planning are logged."*
26. **Fix-45 (Defect-45, LOW, Phase 3 — added 2026-05-01):** Test function naming inconsistency in `tomato_sandbox/tests/integration/test_section15_*.py` — 4 files (`tier1`, `tier2`, `tier3a`, `tier3b`) use `def test_scenario_S*_*():` per Section 15.2 Convention 14, while 9 files (`tier3c`, `tier3d`, `tier4a`, `tier4b`, `tier5`, `boundary`, `underpowered`, `disagreement`, `tta`) use the shorter `def test_S*_*():` form. Both are pytest-discoverable (any `test_*` function is picked up); total count is correct (135). Functionally identical for Phase 4 implementer — but stylistically inconsistent. **Action:** during T-EARLY-MP batch, rename all 9 short-form functions to the verbose `test_scenario_S*_*` form to match Convention 14. Single-pass `sed` or refactor across the 9 files. Severity: LOW. **No Phase 4 blocker** — pytest discovers both forms.

### Phase 3 Round-2 additions (Phase-3-close audit findings, added 2026-05-01)

27. **Fix-46 through Fix-52 (PDA Defect-45..51, Phase 3 exit gate, mixed severity):** 7 master-prompt defects surfaced during Phase 3 exit gate audit. Detailed in `tomato_progress_reports/pda_phase3_20260501T0030.md`:
    - **Defect-45 (HIGH):** Section 8.3 + Section 27 Fix-16 — "request confirmation" scope is ambiguous; encoder applied Fix-16 to 7 cases as a batch (SPEC-INT-002) without per-case confirmation. User pre-approved batch but text is unclear.
    - **Defect-46 (HIGH):** Section 11.4b — SPEC-INT-NNN format is undefined; SPEC-INT-001 + SPEC-INT-002 exist as informal templates. (Same as earlier Defect-23 — re-flagged.)
    - **Defect-47 (HIGH):** Section 11 / Section 23.2 — manifest evolution protocol absent. DEC-019 happened without master-prompt mechanism described.
    - **Defect-48 (MEDIUM):** Section 4 — no Phase 4 entry preconditions block (symmetric to T-PHASE-3-PRECONDITIONS).
    - **Defect-49 (MEDIUM):** Section 27 Fix-16 vs SPEC-INT-002 batch practice contradiction. Same fix as Defect-45.
    - **Defect-50 (LOW):** Section 8.3 encoder file list ("8 files" stale; encoder produced 13 correctly).
    - **Defect-51 (LOW):** Section 8.3 naming convention example-vs-normative ambiguity.
    
    **Action:** queue all 7 in T-EARLY-MP. Defects 45/46/49 (HIGH for Phase 3 re-runs; not Phase 4) recommended priority within batch.

28. **Fix-53 (Defect-52, LOW, Phase 3 anti-cheat finding — added 2026-05-01):** All 13 test files in `tomato_sandbox/tests/integration/test_section15_*.py` use `# noqa: E402` lint suppression on the `assign_tier` import line. E402 (module-level import not at top) does not actually fire under standard flake8 in this file structure (docstring → `import pytest` → comment → flagged import). The noqa comment is cosmetically suppressing a warning that wouldn't fire. **No functional effect** on pytest collection or test outcomes. **Action:** during T-EARLY-MP batch, strip the `# noqa: E402` annotations from all 13 files via single-pass `sed`. Severity: LOW. No Phase 4 blocker.

29. **Fix-54 (Defect-53, HIGH, meta-pattern, Phase 3 audit finding — added 2026-05-01):** **phase-exit-auditor checklist derivation defect.** The Phase 3 phase-exit-auditor returned READY based on a 13-check curated list that omitted master prompt Section 4 Phase 3 tasks 3, 4, 5, 6 (conftest/pyproject, phase_3_tests_initial.txt, unit test directory, pre-commit hook). PVA caught the gap by deriving its checks from the master prompt directly. This is a STRUCTURAL META-PATTERN: when one auditor produces the checklist that downstream audits verify against, gaps in the original list become invisible to all downstream audits. **Fix:** patch `phase-exit-auditor` agent definition (master prompt section 8.9) to instruct: *"For phase exit gates, derive the checklist from `tomato_master_prompt.md` Section 4 [phase] task list verbatim. Every numbered task in Section 4 for the relevant phase becomes a check. Do not curate or summarize; the master prompt's task list IS the checklist. Add additional checks (sacred manifest, log entries, etc.) AFTER the master prompt's task list, never instead of it."* Severity: HIGH (this defect causes silent Phase-incomplete states like the Phase 3 case). **Apply during T-EARLY-MP batch, prioritized within HIGH tier.** **No Phase 4 blocker** because the immediate gap (Phase 3 tasks 3-6 not done) was caught by PVA and being closed inline now; future phase exit gates need this fix.

30. **Fix-55 (Defect-54, HIGH, Phase 4 finding — added 2026-05-01):** **DEC numbering race condition under parallel implementer dispatch.** Phase 4 Batch 1 dispatched T-IMPL-1a, 1b, 1c in a single message with 3 Agent tool calls in parallel. Both T-IMPL-1a (sacred_guard) and T-IMPL-1b (server skeleton) independently observed DEC-025 as the latest entry in `tomato_decisions.md` and each grabbed DEC-026 for their own decision entry. T-IMPL-1b's entry was written first to disk; T-IMPL-1a's was renumbered to DEC-028 by main-thread scribe with full annotation. **Fix:** main thread MUST pre-allocate DEC numbers and pass them in dispatch prompts with explicit instruction: *"Log your architectural decisions as DEC-NNN; do not pick a different number; do not pick the next available number."* Same pattern applies to any append-only ledger that subagents write to (`tomato_log.md`, `tomato_blockers.md`). **Master prompt update target:** Section 11.4 (DEC entry format) should be updated with the pre-allocation rule for parallel dispatches. Also update master prompt Section 8.4 (implementer agent body) to require honoring assigned DEC numbers. Severity: HIGH because the failure mode (duplicate IDs) is silent unless audit catches it via positional collision count. **Already in effect as procedural rule from 2026-05-01 13:30** (per user direction in this turn); applied retroactively for Batch 2 dispatch. T-EARLY-MP batch will codify in master prompt text. **No Phase 4 blocker** — procedural rule prevents recurrence.

31. **Fix-56 (Defect-55, LOW, Phase 4 Batch 3 finding — added 2026-05-02):** **DEC-038 (commit discipline) needs codification in master prompt.** Implementer subagent edits to `.claude/agents/implementer.md` (Rule 12 inline correction) take effect for all future implementer dispatches in this project, but the master prompt itself does not yet reflect DEC-038's "main thread does all commits" rule. Master prompt Section 8.4 should explicitly state: implementer agents do not call `git add` or `git commit`; main thread handles all git operations after batch verification. Anti-cheat scans should treat any implementer-driven commit as a HIGH finding (deviation from DEC-038). **Master prompt update target:** Section 8.4 (implementer agent body) + Section 27 (Phase 4 corrections block). Severity: LOW (procedural; agent-level edit is sufficient for current execution). **No Phase 4 blocker.**

32. **Fix-57 (Defect-56, LOW, Phase 4 Batch 3 close-out finding — added 2026-05-02):** **`.claude/agents/implementer.md` Rule 9 wording vs actual practice.** Rule 9 reads "log every architectural choice in `tomato_decisions.md` BEFORE writing code that implements it. Wait for user approval." In actual Batch 1-3 practice, implementers logged DEC entries (DEC-022..037) and proceeded immediately; user approval happened at batch close (anti-cheat + checkpoint + main-thread commit) rather than per-DEC mid-batch. Neither pattern is broken — the audit cadence catches issues either way — but the rule wording diverges from observed practice. **Fix options:** (a) Soften Rule 9 to "Log architectural choice with rationale at the time of decision; main thread reviews at batch close." (Match practice.) (b) Tighten practice to match Rule 9's literal strictness: implementer stops after writing the DEC, waits for explicit user approval before any code. **Recommended: (a).** Practice is observed-safe; per-DEC user approval would 4-10× the dispatch round-trips per batch with no demonstrated quality benefit. **Master prompt update target:** Section 8.4 (implementer agent body wording) and `.claude/agents/implementer.md` Rule 9. Severity: LOW.

33. **Fix-58 (Defect-58, LOW, Phase 4 Batch 5 milestone close-out finding — added 2026-05-02):** **Plan `rule_fired` literal strings outdated.** `tomato_plan.md` Rule list still references the original paraphrased identifiers (e.g. `signal_failure_rule1`, `psv_unreliable_or_chilli_leakage`, etc.). The actual T-IMPL-5 implementation (DEC-041) correctly uses the import contract's 12 canonical values: `"1"`, `"3"`, `"4"`, `"5"`, `"6"`, `"7a"`, `"7b"`, `"7c"`, `"8a"`, `"8b"`, `"8c"`, `"catch_all_low_confidence"`. This is exactly the DEC-015 pattern — plan is scaffolding, contract is authority — so it is not a defect, but it would mislead a fresh-session reader who reads the plan first. **Fix:** either (a) update plan rule_fired references to the import contract's 12 values, or (b) add a DEC-015-style annotation noting the paraphrase is superseded by the import contract + DEC-041 + BLK-011. Recommended: (b) — preserves history, less invasive. **Master prompt update target:** none required; agent definition + import contract already authoritative. Severity: LOW. **No Phase 4 blocker.**

34. **Fix-59 (Defect-59, MEDIUM, Phase 4 Batch 7 close-out finding — added 2026-05-02):** **Latent Batch 0 logging fallback bug — RESOLVED via DEC-046.** Batch 0 logging.py "structlog with stdlib fallback" design (DEC-022) returned a raw stdlib `Logger` for the fallback path; ~20 production callsites use structlog-style kwargs (`_log.debug("event", key=val)`) which crashed with `TypeError: Logger._log() got unexpected keyword 'shape'` at module import time. Latent because all prior pytest runs used system Python (had structlog), but venv was missing structlog. Fixed in DEC-046 by introducing `_StdlibKwargsAdapter` class wrapping stdlib Logger with kwargs-translation. **Master prompt impact:** none required — fix is in implementation file, with 7 new unit tests covering the fallback path. No T-EARLY-MP follow-up needed beyond this entry as historical record. Severity: MEDIUM (was a real production blocker until fixed; now closed). **Status: RESOLVED in Batch 7.**

35. **Fix-60 (Defect-60, MEDIUM, Phase 4 Batch 7 close-out finding — added 2026-05-02):** **Process gap — Bash tool default Python is system, not venv.** All Phase 4 pytest reports through Batch 6 used system Python (`python -m pytest`). Venv (`venv/Scripts/python.exe`) was the supposed production-equivalent but was never invoked for testing. This was discovered in Batch 7 when `venv/Scripts/python.exe -m uvicorn` failed with structlog/pytest missing. Verified retroactively by running pytest under both interpreters: identical 1118-pass count (warning count differs by Pillow version: 71 system vs 15 venv; non-defect). Test outcomes are valid; venv equivalence is now confirmed. **Standing rule going forward:** ALL test-count reports in checkpoints must specify which interpreter ran them, AND venv pytest must run as part of every batch closure. **Master prompt update target:** Section 4 phase template (add interpreter requirement to test-running steps); Section 27 (Phase 4 corrections block — record this rule). Severity: MEDIUM (process / honest-accounting concern; not a code defect). **No Phase 4 blocker** — all batch reports are retroactively venv-equivalent.

**[GLOBAL REORDER 2026-04-28 per Auditor RD-3:** original list interleaved Phase 0 MEDIUM/LOW (Fix-4 through Fix-8) before Phase 1 HIGH (Fix-9 through Fix-12). Now strictly HIGH→MEDIUM→LOW globally. Fix-N IDs preserved for cross-reference; position numbers (1-25) reflect new execution order. Severity ordering verified: positions 1-10 are all HIGH, 11-20 are all MEDIUM, 21-25 are all LOW.]

### Out-of-band fast-tracked fixes (APPLIED 2026-04-28 per D6 — not in the numbered 1-25 list above)

These 3 fixes were added to `tomato_master_prompt.md` Section 27 (Phase 2 Corrections fast-track block) directly, without going through the numbered T-EARLY-MP slot. They affected how Round 4 itself runs, so they were applied out-of-band.

- **Fix-27 (Defect-27, HIGH):** Phase exit gate composition rule — fire all 5 audits in single parallel batch. **[APPLIED 2026-04-28 in master prompt Section 27 + phase-exit-auditor.md body addition; effective Round 4 onward.]**
- **Fix-28 (Defect-28, HIGH):** Plan-edit authority for inline patches — main thread may patch mechanical errors with DEC entry + traceability comment + gate re-fire. **[APPLIED 2026-04-28 in master prompt Section 27; codifies DEC-013 retrospectively and authorizes future inline patches.]**
- **Fix-34 (Defect-34, HIGH — NEW from Phase 2 60% defect rate):** Planner reads spec body verbatim — NOT spec summaries — when writing task cards that name code contracts. **[APPLIED 2026-04-28 in master prompt Section 27 + planner.md body addition; effective for any future planner invocation.]**

### Phase-3-critical execution order (read this first)

Execute these subset of fixes BEFORE Phase 3 begins (others can wait until later T-EARLY-MP slot):
- **Fix-9, Fix-10, Fix-11, Fix-12, Fix-16, Fix-19, Fix-20** — these are HIGH and either fix the planner protocol, the encoder protocol, or the missing Phase-3-blocker rule itself.
- All MEDIUM and LOW can run after Phase 3 if needed.

**Acceptance criteria:**
- All 25 fix descriptions are appended to `tomato_master_prompt.md` as a clearly delimited "Phase 2 Corrections" block.
- The append does not alter any text above the delimiter.
- Sacred guardian run after append shows 0 drift on all 10 manifest entries.
- Severity order verified: all HIGH items appear before any MEDIUM, all MEDIUM before any LOW (per Defect-24 / Fix-24).

---

### T-EARLY-VENV — Sandbox Virtual Environment Creation
**Spec sections covered:** S26 (Engineering Hygiene — Python 3.13, isolated venv)
**Dependencies:** None (precedes any implementation)
**Files to create or modify:** `tomato_sandbox/.venv/` (created by `python3.13 -m venv`); `tomato_sandbox/requirements.txt` (created)
**Estimated effort:** 1h

Actions:
1. Create `tomato_sandbox/.venv/` using Python 3.13.
2. Create `tomato_sandbox/requirements.txt` with pinned versions for: `fastapi`, `uvicorn[standard]`, `torch` (CUDA 13 wheel), `torchvision`, `timm`, `pytorch-grad-cam`, `Pillow`, `numpy`, `opencv-python-headless`, `pydantic`, `structlog`, `prometheus-client`, `pyyaml`, `aiofiles`, `scipy`, `scikit-image`, `pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`, `ruff`, `pre-commit`.
3. Install into `.venv`.
4. Verify `python --version` == 3.13.x inside `.venv`.
5. Add `.venv/` to `tomato_sandbox/.gitignore` (create the file if absent).

**Acceptance criteria:**
- `tomato_sandbox/.venv/bin/python --version` (or `.venv/Scripts/python.exe --version` on Windows) prints `Python 3.13.x`.
- `import torch; torch.cuda.is_available()` returns `True` from within `.venv`.
- `tomato_sandbox/requirements.txt` exists and is committed (venv directory is gitignored).

---

### T-PHASE-3-PRECONDITIONS — Phase 3 Gate Verification
**Spec sections covered:** S14, S15 (gate check only; no new implementation)
**Dependencies:** T-IMPL-5a, T-IMPL-5b, T-EARLY-MP (Fix-16, Fix-23 if applied)
**Files to create or modify:** `tomato_progress_reports/phase3_gate_check.md`; `spec_changelog.md` (BLK-004 entry written here per DEC-012 condition (b))
**Estimated effort:** 1h

Steps (5 gates, all must PASS):
1. Import `tomato_sandbox.tier.tier_assignment` and call `assign_tier()` with a synthetic input; verify it returns a `TierAssignment` dataclass (not a stub/NotImplementedError).
2. Verify `tomato_sandbox/tier/tier_rules.yaml` exists, parses without error, and contains `schema_version: 1`. Verify each rule has a `# spec: 14.N paragraph M` traceability comment.
3. Verify `tomato_master_prompt.md` contains Fix-16 (Defect-16 section15-encoder intra-spec conflict rule). `grep` for the literal string "scenario body text as authoritative" in the master prompt.
4. **[ADDITION 2026-04-28 per Auditor B3 / PVA SD-4 / PDA Defect-21]** Verify `spec_changelog.md` contains a BLK-004 / Defect-15.1 entry recording the line 5558 typo correction with line 4117 authoritative. Required fields per Fix-23 (or current SPEC-CHG-NNN format if Fix-23 unapplied): defect ID = Defect-15.1, conflicting locations = lines 4117 and 5558, authoritative = line 4117, BLK reference = BLK-004, user approval = DEC-012.
5. **[ADDITION 2026-04-28 per BLK-009 patches]** Verify BLK-009 sub-defects all marked PATCHED in `tomato_blockers.md`: Defect-9.1 (TTA signature), Defect-9.2 (remap location annotation in plan + dep graph), Defect-9.3 (chilli_leakage threshold).
6. Document gate status in `tomato_progress_reports/phase3_gate_check.md` with PASS/FAIL per gate.

**Acceptance criteria:**
- All 5 verification steps pass without error.
- Gate status file exists at `tomato_progress_reports/phase3_gate_check.md` with explicit PASS for all 5 gates.
- `spec_changelog.md` contains the BLK-004 entry.
- Phase 3 (section15-encoder dispatch) may only begin after this task is marked done.

---

## Implementation Tasks: T-IMPL Sequence

> **Build-order rule:** Each batch below is internally parallelizable. Batches must be executed in strict sequential order. No task within a batch may begin until all tasks in the preceding batch are done.

---

### BATCH 1 — T-IMPL-1: Foundation

#### T-IMPL-1a — Sacred Manifest Verifier Module
**Spec sections covered:** S2 (sacred manifest), S26 (engineering hygiene — spec compliance check)
**Dependencies:** T-EARLY-VENV
**Files to create:**
- `tomato_sandbox/utils/sacred_guard.py`
- `tomato_sandbox/utils/__init__.py`

**What to build:**
A Python module that loads `.claude/sacred_manifest.json` at import time and exposes `verify_manifest() -> dict` returning `{entry_name: "PASS"|"FAIL"|"MISSING"}` for all 10 manifest entries. Uses the canonical hash algorithm: `json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")` for directory entries; `sha256(file_bytes).hexdigest()` for file entries; relative paths use `/` separators. This module must be called by the startup sequence (T-IMPL-7a) but is also importable standalone.

**Acceptance criteria:**
- `from tomato_sandbox.utils.sacred_guard import verify_manifest` succeeds.
- Calling `verify_manifest()` with the actual project on disk returns all 10 entries as `"PASS"`.
- Deliberately corrupting one path returns `"FAIL"` for that entry only.
- Module has 100% unit-test coverage (S26 requirement for utility modules).

**Estimated effort:** 2h

---

#### T-IMPL-1b — FastAPI Skeleton Server
**Spec sections covered:** S20 (sandbox server architecture), S4 (startup sequence)
**Dependencies:** T-EARLY-VENV
**Files to create:**
- `tomato_sandbox/api/server.py`
- `tomato_sandbox/api/__init__.py`
- `tomato_sandbox/config.py`
- `tomato_sandbox/config/default.yaml`

**What to build:**
FastAPI app instance bound to `127.0.0.1:8767` (`--workers 1`). Implements only the structural skeleton:
- `app = FastAPI(title="Tomato Sandbox", version="1.0.0")`
- Lifespan context manager with 12-step startup stub (steps that load models return placeholder objects; steps log their execution).
- `GET /health` returning `{"status": "ok", "model_loaded": false}`
- `GET /ready` returning HTTP 503 until startup complete, then 200
- `GET /info` returning `{"model_version": "unloaded", "multi_image_max_n": 5}`
- Placeholder `POST /predict` returning HTTP 503 "Not ready"
- `asyncio.Lock` instance on `app.state.gpu_lock`
- `app.state.pipeline = None` placeholder
- `TOMATO_*` env vars loaded from `/etc/tomato_sandbox/env` if present, then environment, then `config/default.yaml` fallback.
- Port resolved from `TOMATO_PORT` env var (default 8767). Must not be 8766.

**Acceptance criteria:**
- `uvicorn tomato_sandbox.api.server:app --host 127.0.0.1 --port 8767` starts without error.
- `GET /health` returns 200 with JSON body.
- `GET /ready` returns 503 before startup complete.
- `POST /predict` returns 503 "Not ready".
- Default port constant is 8767 in `config/default.yaml`.

**Estimated effort:** 2h

---

#### T-IMPL-1c — Lint and Test Scaffolding
**Spec sections covered:** S26 (CI gates, test layers, mypy strict, ruff, pytest structure)
**Dependencies:** T-EARLY-VENV
**Files to create:**
- `tomato_sandbox/pyproject.toml` (ruff + mypy config)
- `tomato_sandbox/tests/__init__.py`
- `tomato_sandbox/tests/unit/__init__.py`
- `tomato_sandbox/tests/integration/__init__.py`
- `tomato_sandbox/tests/e2e/__init__.py`
- `tomato_sandbox/.pre-commit-config.yaml`
- `tomato_sandbox/Makefile` (targets: `lint`, `type-check`, `test-unit`, `test-integration`, `test-e2e`, `coverage`)
- `tomato_sandbox/tests/conftest.py`
- `tomato_sandbox/tests/unit/test_scaffold.py`

**What to build:**
- `pyproject.toml`: ruff rules `["E", "F", "W", "I", "N", "UP"]`; mypy `strict = true`, `python_version = "3.13"`; pytest `testpaths = ["tests"]`, `asyncio_mode = "auto"`.
- Pre-commit hooks: ruff, mypy, `openapi.yaml` regeneration check.
- Makefile targets wired to the venv's pytest/ruff/mypy.
- `conftest.py` with a minimal `test_app` fixture (TestClient wrapping the skeleton server).
- Stub test that asserts `1 == 1` (smoke that the test harness runs).

**Acceptance criteria:**
- `make lint` exits 0 on the skeleton codebase.
- `make type-check` exits 0 (or exits non-zero only for legitimate stub type issues annotated `# type: ignore[...]`).
- `make test-unit` runs and passes the scaffold smoke test.
- Pre-commit hook installs cleanly.

**Estimated effort:** 2h

---

### BATCH 2 — T-IMPL-2: Input and IQA

#### T-IMPL-2a — Image Input Validation (S5)
**Spec sections covered:** S5
**Dependencies:** T-IMPL-1b, T-IMPL-1c
**Files to create:**
- `tomato_sandbox/input/validator.py`
- `tomato_sandbox/input/__init__.py`
- `tomato_sandbox/tests/unit/test_validator.py`

**What to build:**
`ValidatedImage` dataclass per S5 contract:
```
image_id: str          # UUID
original_bytes: bytes
pil_image: PIL.Image
numpy_rgb: np.ndarray  # uint8 [H, W, 3]
width: int
height: int
file_size_bytes: int
format_detected: str   # "jpeg"|"png"|"webp"
rejection_reason: str | None
```
`validate_request(file_bytes: bytes, filename: str) -> ValidatedImage`
Implements all 6 checks (A-F) per S5:
- A: magic-byte format check (JPEG/PNG/WebP only)
- B: file size <= 10 MB (S16.9 cross-ref: HTTP 400 on >10 MB)
- C: PIL decode (corrupt file rejection)
- D: minimum dimension 224x224
- E: single-channel dominance (MAX_CH_RATIO)
- F: non-plant image guard per S5

Returns populated `ValidatedImage` with `rejection_reason=None` on success, or with `rejection_reason` set to one of the 11 reason codes from S5 on failure. No import from `scripts/apin/` or `app/` (BLK-003).

**Acceptance criteria:**
- All 6 checks covered by unit tests (at least 2 cases per check: pass and reject).
- 11 rejection reason code strings match S5 exactly.
- `rejection_reason` is `None` for a valid 300x300 JPEG.
- Non-None for a 100x100 PNG (too small).
- No import of anything from `scripts/apin/` (BLK-003).
- 100% branch coverage.

**Estimated effort:** 2h

---

#### T-IMPL-2b — Image Quality Assessment (S6)
**Spec sections covered:** S6
**Dependencies:** T-IMPL-2a
**Files to create:**
- `tomato_sandbox/input/iqa.py`
- `tomato_sandbox/tests/unit/test_iqa.py`

**What to build (CORRECTED 2026-04-28 per BLK-010.1; previous version had fabricated dimension/field names):**

`IQAResult` dataclass — verbatim from spec Section 6.5 (lines 1357-1366):
```python
@dataclass
class IQAResult:
    decision: str                     # REJECT / DEGRADED / ACCEPTABLE / HIGH
    aggregate_score: float            # in [0, 1]
    per_dimension: dict[str, float]   # 7 entries, dimension name -> score
    failing_dimensions: list[str]     # names where score < BAD_THRESHOLD
    retake_message: str | None        # if decision == REJECT, the user-facing message
    green_mask: np.ndarray | None     # rough HSV-green mask; passed to PSV as a hint
# spec: 6.5 lines 1357-1366
```

`compute_iqa(validated_image: ValidatedImage) -> IQAResult` (per Section 6.6 line 1374).

Implements 7 IQA dimensions per Section 6.2 (lines 1068-1280) — verbatim names:
1. **sharpness** — variance of Laplacian (Section 6.2.1, line 1074); BAD < 0.20, GOOD > 0.50
2. **exposure** — mean luminance with under/over-exposure penalties (Section 6.2.2, line 1097)
3. **leaf_presence** — HSV green mask coverage (Section 6.2.3, line 1127); produces `green_mask` field
4. **leaf_fill** — fraction of frame occupied by leaf (Section 6.2.4, line 1159); BAD < 0.30
5. **background_contamination** — non-leaf clutter score (Section 6.2.5, line 1194); BAD > 0.30
6. **resolution** — minimum dimension check (Section 6.2.6, line 1226)
7. **wetness** — specular highlight detector (Section 6.2.7, line 1252); BAD > 0.30

Aggregation (Section 6.3, line 1282) and 4-way decision (Section 6.4, line 1299):
- REJECT / DEGRADED / ACCEPTABLE / HIGH per spec thresholds (placeholder values; F.0 calibrates).

**Critical cross-contracts:**
- `green_mask` is consumed by Section 10 PSV (per Section 6.5 line 1368): used as sanity check + fallback by PSV's careful segmentation. IQA's green_mask is the rough HSV-based mask from `leaf_presence`. # spec: 6.5 line 1368
- `decision == "DEGRADED"` triggers Section 14 Rule 7a / 8a (Tier 3D cap), not "Tier 3C". (Section 14 Rule 3 = Tier 3C, but that's chilli_leakage / psv_reliability — not IQA. Earlier draft conflated.) # spec: 14, Rule 7a/8a
- `decision == "REJECT"` short-circuits the pipeline at the server before invoking the rest (Section 6.6 line 1374).

Failure mode from S3.10: if `compute_iqa()` throws, return `IQAResult(decision="DEGRADED", aggregate_score=0.0, per_dimension={}, failing_dimensions=[], retake_message=None, green_mask=None)` with the IQA failure logged in pipeline context.

**Acceptance criteria:**
- All 7 dimension function names match spec verbatim: `sharpness`, `exposure`, `leaf_presence`, `leaf_fill`, `background_contamination`, `resolution`, `wetness`. Test by listing `IQAResult.per_dimension.keys()` and asserting set equality.
- `IQAResult` dataclass has all 6 spec fields including `green_mask` and `retake_message`. Test by introspecting `dataclasses.fields(IQAResult)`.
- `green_mask` is a numpy array of dtype bool or uint8 with shape matching input image H×W when `decision != "REJECT"`. None when REJECT.
- Decision boundary tests: REJECT floor, DEGRADED threshold, ACCEPTABLE/HIGH boundary per Section 6.4.
- Exception safety test: mock a dimension function to throw; verify decision falls back to "DEGRADED" and `failing_dimensions` records the failed dimension name.
- 100% branch coverage on `compute_iqa()`.
- IQA→PSV contract test: feed an image with detectable leaf; verify `green_mask` is non-None and has expected coverage; pass into a PSV stub and verify it uses the mask as fallback when its own segmentation fails.

**Estimated effort:** 2h

---

#### T-IMPL-2c — Preprocessing (S7)
**Spec sections covered:** S7
**Dependencies:** T-IMPL-2a
**Files to create:**
- `tomato_sandbox/input/preprocessing.py`
- `tomato_sandbox/tests/unit/test_preprocessing.py`

**What to build:**
Three pure preprocessing functions per S7 contracts:

`preprocess_for_v3(image: ValidatedImage) -> torch.Tensor`
- CLAHE per RGB channel (clip_limit=2.0, tile_size=8x8)
- Resize to 224x224
- Normalize with ImageNet mean/std
- Returns float32 tensor [1, 3, 224, 224]

`preprocess_for_lora(image: ValidatedImage) -> torch.Tensor`
- CLAHE per RGB channel (same params)
- Resize with padding to 392x392 (pad_value=114)
- Normalize with ImageNet mean/std
- Returns float32 tensor [1, 3, 392, 392]

`preprocess_for_psv(image: ValidatedImage) -> np.ndarray`
- Resize (max dimension 1200px, preserve aspect ratio)
- Returns uint8 numpy array for OpenCV/skimage pipelines

Pinned constants (must not be local literals; imported from `tomato_sandbox/config.py`):
- `CLAHE_CLIP_LIMIT = 2.0`
- `V3_INPUT_SIZE = 224`
- `LORA_INPUT_SIZE = 392`
- `LORA_PAD_VALUE = 114`
- `PSV_MAX_DIM = 1200`

**Acceptance criteria:**
- Output shape assertions for all three functions.
- CLAHE is first step in v3 and LoRA pipelines (order verified by inspecting intermediate tensor values).
- PSV output preserves aspect ratio: an input of 600x900 with PSV_MAX_DIM=1200 produces 800x1200 output.
- Constants imported from config, not inline literals.
- Unit tests cover square, portrait, and landscape inputs.

**Estimated effort:** 2h

---

### BATCH 3 — T-IMPL-3: Signal Wrappers (parallel within batch)

> **CRITICAL ANNOTATION (CORRECTED 2026-04-28 per BLK-009 / Defect-9.2 spec-body verification at lines 1578-1792):**
> All three signals deliver canonical-ordered probabilities to S12. Specifically:
> - **Signal A (v3, Section 8):** internally produces 10-class native probs (line 1640 of spec), then `extract_v3_outputs` (Section 8.3, lines 1660-1685) applies the remap `[0, 2, 1, 3, 4, 5]` (named `LORA_INDEX_FOR_V3_CLASS`) and returns `SignalAResult.tomato_probs_canonical: np.ndarray [6]` already canonical-ordered (line 1719 of spec).
> - **Signal B (LoRA, Section 9):** outputs 6-class natively in canonical order. Spec Section 9.1 last line: *"This ordering matches canonical, so no remap is needed for LoRA → canonical."*
> - **Signal C (PSV, Section 10):** outputs 6-class compatibility scores in canonical order.
> - **The remap is INTERNAL to Signal A's `extract_v3_outputs`** (`tomato_sandbox/signals/v3_signal.py` per Section 8.7 line 1736). It is NOT visible to S12. T-IMPL-4a does NOT remap. Adding a remap inside T-IMPL-4a would double-remap Signal A's output and produce wrong results.
> - **Earlier wording in this annotation said the opposite** ("native ordering returned from signal; remap applied at T-IMPL-4a") — that was inverted relative to spec. PATCHED 2026-04-28.

#### T-IMPL-3a — Signal A Wrapper (v3 10-class, S8)
**Spec sections covered:** S8
**Dependencies:** T-IMPL-2c, T-IMPL-1a
**Files to create:**
- `tomato_sandbox/signals/signal_a.py`
- `tomato_sandbox/signals/__init__.py`
- `tomato_sandbox/tests/unit/test_signal_a.py`

**What to build (CORRECTED 2026-04-28 per BLK-009 / Defect-9.2 — match spec Section 8 verbatim):**

`SignalAResult` dataclass per Section 8.6 (lines 1717-1726 of spec):
```python
@dataclass
class SignalAResult:
    tomato_probs_canonical: np.ndarray       # [6], canonical ordering (foliar, septoria, late_blight, ylcv, mosaic, healthy)
    tomato_max_prob_canonical: float         # max of tomato_probs_canonical
    tomato_argmax_canonical: int             # index 0-5 of max in canonical
    chilli_leakage: float                    # in [0, 1], sum of v3 indices 6-9
    raw_probs_v3_order: np.ndarray | None    # [10] raw v3 native, kept for monitoring/debug
    forward_succeeded: bool                  # True unless exception or NaN occurred
    failure_reason: str | None               # "exception" | "numerical_instability" | None
```

Two functions per Section 8 (signatures verbatim from spec lines 1617-1640 and 1660-1685):
```python
def signal_a_forward(model, x: torch.Tensor) -> dict:
    # x: [B, 3, 224, 224] tensor on GPU; sets crop_mode=TOMATO_CROP_MODE_INDEX (=2)
    # Returns: {"logits": [B,10], "probs": [B,10] softmax, "ok": bool}
    # NaN/Inf guard returns {"logits": None, "probs": None, "ok": False}

def extract_v3_outputs(probs_10d: torch.Tensor) -> dict:
    # Applies remap [0, 2, 1, 3, 4, 5] INSIDE this function (spec line 1672-1678).
    # Returns: {"tomato_probs_canonical": [6] canonical, "chilli_leakage": float, "raw_probs_v3_order": [10]}
    # Per spec line 1687: "the 6 tomato probs do NOT sum to 1 after extraction — they sum to (1 - chilli_leakage)".
```

Plus `compute_signal_a(model, tensor: torch.Tensor) -> SignalAResult` per Section 8.7 lines 1741-1773 (scaffolding pasted in spec; copy verbatim).

- Model path: `scripts/model3_training/checkpoints/model3_production_v3.pt` (sacred; read-only via config).
- **Remap [0, 2, 1, 3, 4, 5] is applied INSIDE `extract_v3_outputs`** (spec lines 1672-1678) — not at T-IMPL-4a, not anywhere downstream.
- Re-normalization of tomato_probs_canonical is PROHIBITED (Section 8.3 line 1687: erases chilli_leakage signal that downstream relies on).
- BLK-004: v3 priors vector for S1.1 reference = `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` (line 4117 authoritative; not line 5558).
- On exception or NaN: return `SignalAResult` with zero-filled probs, `forward_succeeded=False`, `failure_reason="exception: ..."` or `"numerical_instability"` (spec lines 1742-1763).

**Acceptance criteria:**
- `tomato_probs_canonical.shape == (6,)` and is in **canonical** ordering (foliar=0, septoria=1, late_blight=2, ylcv=3, mosaic=4, healthy=5).
- Remap-direction unit test: feed v3 native probs `[0.1, 0.8, 0.0, 0.0, 0.05, 0.05, 0, 0, 0, 0]` (v3 native index 1=late_blight=0.8, index 2=septoria=0.0); after `extract_v3_outputs`, assert `tomato_probs_canonical[1] == 0.0` (canonical septoria) and `tomato_probs_canonical[2] == 0.8` (canonical late_blight). This is the inverse of the previous (wrong) test — the remap fires here, not at T-IMPL-4a.
- `abs(sum(tomato_probs_canonical) + chilli_leakage - 1.0) < 1e-5` (partition unity without re-norm).
- Re-normalization test: input where chilli_leakage=0.15; verify `sum(tomato_probs_canonical) ≈ 0.85`, not 1.0.
- Exception safety: mock model to throw `RuntimeError`; verify `forward_succeeded=False`, `failure_reason="exception: RuntimeError"`.
- NaN safety: mock model to return logits with NaN; verify `forward_succeeded=False`, `failure_reason="numerical_instability"`.
- Model path loaded from config, not hardcoded.

**Estimated effort:** 2h

---

#### T-IMPL-3b — Signal B Wrapper (LoRA 6-class, S9)
**Spec sections covered:** S9
**Dependencies:** T-IMPL-2c
**Files to create:**
- `tomato_sandbox/signals/signal_b.py`
- `tomato_sandbox/tests/unit/test_signal_b.py`

**What to build:**
`SignalBResult` dataclass per S9:
```
probs: np.ndarray            # shape [6], canonical ordering (LoRA is already canonical; no remap needed)
prototype_blend_applied: bool
bank_version: str
status: str                  # "ok"|"failed"|"degraded"
latency_ms: float
```
`PrototypeBank` dataclass per S9:
```
version: str
class_prototypes: dict       # {class_name: np.ndarray of shape [embedding_dim]}
```
`compute_signal_b(tensor: torch.Tensor, model: Any, bank: PrototypeBank) -> SignalBResult`
- Model path: `models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt` (sacred). Via config.
- LoRA output is already canonical ordering — no remap applied.
- `prototype_blend()`: blends model softmax with prototype cosine similarity per S9 formula.
- Startup bank-version mismatch = fatal: server startup fails with CRITICAL log.
- On exception: return `SignalBResult(status="failed", probs=np.zeros(6), ...)`.

`prototype_blend(softmax_probs: np.ndarray, embedding: np.ndarray, bank: PrototypeBank) -> np.ndarray`
- Returns blended probability vector shape [6].

**Acceptance criteria:**
- `probs.shape == (6,)` and `abs(sum(probs) - 1.0) < 1e-5`.
- Bank-version mismatch test: pass mismatched version string to startup loader; verify startup exception raised.
- Exception safety test.
- LoRA outputs canonical order (no remap applied or needed; test verifies by checking class-name-to-index mapping).
- Prototype blend result verified against manually computed expected value.

**Estimated effort:** 2h

---

#### T-IMPL-3c — Signal C PSV Pipeline (S10)
**Spec sections covered:** S10
**Dependencies:** T-IMPL-2c
**Files to create:**
- `tomato_sandbox/signals/signal_c.py`
- `tomato_sandbox/signals/psv_features.py`
- `tomato_sandbox/signals/psv_weights.yaml`
- `tomato_sandbox/tests/unit/test_signal_c.py`

**What to build:**
`SignalCResult` dataclass per S10:
```
features: np.ndarray          # shape [26], fixed index order per S10
compatibility: np.ndarray     # shape [6], softmax compatibility scores per canonical class order
reliability: float            # geometric mean of 3 reliability components
status: str                   # "ok"|"failed"|"degraded"
latency_ms: float
```
`compute_signal_c(image_array: np.ndarray) -> SignalCResult`
Orchestrates 5 stage functions from S10:
1. Color space conversion (LAB channels)
2. Texture analysis (LBP features)
3. Shape/morphology (blob analysis)
4. Lesion detection (connected component features)
5. Spectral features (FFT-based)

26 features produced in fixed index order per S10. PSV score canonical ordering per S3.7:
`[c_foliar, c_septoria, c_late_blight, c_ylcv, c_mosaic, c_healthy]`

`psv_weights.yaml`: 6x26 weight matrix (hand-designed; fixed, not learned). Loaded at startup.

Exception handling per S3.10: if entire PSV pipeline throws, set all reliability values to 0.1.

Note on BLK-007: Appendix C (full 26-feature catalog) is absent from spec. Implementer must derive the ordered feature list from Section 10 body prose and document it in `psv_features.py` docstring with paragraph citations. This is the BLK-007 analog for PSV features.

**Acceptance criteria:**
- `features.shape == (26,)`.
- `compatibility.shape == (6,)` and sums to 1.0.
- Feature index order is fixed (same image twice produces same feature at same index).
- Exception safety: mock stage 3 to throw; other stages still run; reliability reflects partial failure.
- Full PSV exception: `reliability=0.1` for all three components.
- `psv_weights.yaml` is parseable and has shape 6x26.
- `psv_features.py` docstring lists all 26 features with S10 paragraph citations.
- Latency logged; `latency_ms > 0` asserted in unit test.

**Estimated effort:** 4h

---

#### T-IMPL-3d — TTA (S11)
**Spec sections covered:** S11
**Dependencies:** T-IMPL-3a, T-IMPL-3b
**Files to create:**
- `tomato_sandbox/signals/tta.py`
- `tomato_sandbox/config/jsd_sentinel.json`
- `tomato_sandbox/tests/unit/test_tta.py`

**What to build (CORRECTED 2026-04-28 per BLK-009 / Defect-9.1 — match Section 11 spec verbatim):**

Trigger function (Section 11 spec — single float in, view-count int out):
```python
def should_trigger_tta(combined_max_prob: float) -> int:
    """Returns 1, 2, or 5 (number of views).
       combined_max_prob >= 0.55 → 1 (no TTA)
       0.45 <= combined_max_prob < 0.55 → 2 (2-view TTA)
       combined_max_prob < 0.45 → 5 (5-view TTA)
       NaN or non-finite → 1 (no TTA, per Section 11.2)
    """
```

NOTE: `combined_max_prob` is the classifier's combined output — NOT a per-signal max. The trigger is invoked AFTER the 1-view classifier pass, on its output, per Section 11.2 spec.

`TTAReport` dataclass per Section 11 (canonical-ordered fields throughout):
```
n_views_attempted: int                       # 1, 2, or 5
n_views_succeeded_v3: int
n_views_succeeded_lora: int
initial_combined_max_prob: float
final_combined_max_prob: float
aggregated_a_canonical: np.ndarray | None    # shape [6], canonical ordering (post-aggregation, post-Signal-A internal remap)
aggregated_b_canonical: np.ndarray | None    # shape [6], canonical (LoRA is canonical natively)
per_view_v3_argmax: list[int]                # -1 for failed views
per_view_v3_succeeded: list[bool]
status: str
```

`apply_tta(pipeline, validated_image, n_views) -> tuple[SignalAResult, SignalBResult, TTAReport]` per Section 11:
- Builds N augmented views (`build_augmentations(n_views) -> list`).
- Runs Signal A and Signal B forward pass once per view (PSV does NOT participate in TTA per Section 11.1).
- Aggregates softmax mean across views; failed views (`ok=False`) excluded.
- Returns aggregated `SignalAResult` and `SignalBResult` plus the report.

**Earlier draft of this task said `(signal_a, signal_b) -> bool` with `max_prob < 0.55 OR margin < 0.45`** — that diverged from spec on signature, return type, and trigger criterion. PATCHED 2026-04-28.

**Acceptance criteria:**
- 3-level trigger test: `should_trigger_tta(0.60)==1`, `should_trigger_tta(0.50)==2`, `should_trigger_tta(0.40)==5`.
- NaN guard: `should_trigger_tta(float("nan"))==1`.
- 5-view path exists and is reachable when combined_max_prob < 0.45.
- PSV not invoked during TTA (mock pipeline asserts `compute_signal_c` is never called inside `apply_tta`).
- Aggregated output is in canonical ordering (Section 11 — the aggregation operates on already-canonical Signal A output).
- Exception safety: model_a throws on view 2 of 5 → that view marked `succeeded=False`; aggregation excludes it; report records `n_views_succeeded_v3 = 4`.
- Conformal/JSD interaction follows Section 11 (separate from this task; JSD is in `tomato_sandbox/signals/jsd.py` if at all).

**Estimated effort:** 2h

---

### BATCH 4 — T-IMPL-4: Fusion and Calibration

#### T-IMPL-4a — Hierarchical Classifier Input Builder (S12)
**Spec sections covered:** S12 (build_classifier_input)
**Dependencies:** T-IMPL-3a, T-IMPL-3b, T-IMPL-3c, T-IMPL-3d
**Files to create:**
- `tomato_sandbox/classifier/feature_builder.py`
- `tomato_sandbox/classifier/__init__.py`
- `tomato_sandbox/tests/unit/test_feature_builder.py`

**What to build (CORRECTED 2026-04-28 per BLK-009 / Defect-9.2):**

`build_classifier_input(signal_a, signal_b, signal_c, tta_report=None) -> np.ndarray`
- Produces 19-dimensional feature vector per S12 fixed layout: 6 v3 canonical + 6 LoRA canonical + 4 PSV summary + 1 JSD + 1 PSV reliability + 1 chilli_leakage = 19 (per Section 8.4 spec line 1698).
- **NO REMAP IN THIS MODULE.** All inputs arrive already canonical:
  - `signal_a.tomato_probs_canonical` is canonical (remap was applied inside `extract_v3_outputs` at T-IMPL-3a, per Section 8.3 lines 1672-1685).
  - `signal_b.probs` is canonical natively (Section 9.1: *"This ordering matches canonical, so no remap is needed for LoRA → canonical."*).
  - `signal_c.compatibility` is canonical 6-class.
- Earlier draft of this task said "HERE is where the remap [0, 2, 1, 3, 4, 5] is applied to Signal A probs" — that was INVERTED relative to spec. PATCHED 2026-04-28.
- Degraded-mode zeroing rules per S12: if `signal_a.forward_succeeded == False`, zero the Signal A slice; analogous for B and C.
- `chilli_leakage` included in feature vector at its fixed index position.

**Acceptance criteria:**
- Output shape `== (19,)`.
- **Remap-NOT-here test (regression guard for the earlier inversion):** mock `signal_a.tomato_probs_canonical = [0.1, 0.0, 0.8, 0.0, 0.05, 0.05]` (canonical: late_blight=0.8 at index 2, septoria=0.0 at index 1). Build the feature vector. Assert the v3 slice of the output equals the input verbatim (no positions swapped). If T-IMPL-4a accidentally applies remap, this test fails because positions 1 and 2 would swap.
- Degraded zeroing test: `signal_a.forward_succeeded=False` → Signal A slice is all-zeros in output.
- Feature vector layout documented in module docstring with index table.

**Estimated effort:** 2h

---

#### T-IMPL-4b — Hierarchical Classifier (S12)
**Spec sections covered:** S12 (compute_classifier, Stage 1, Stage 2, soft routing, Platt calibration)
**Dependencies:** T-IMPL-4a
**Files to create:**
- `tomato_sandbox/classifier/classifier.py`
- `tomato_sandbox/tests/unit/test_classifier.py`
- `tomato_sandbox/tests/unit/test_classifier_degraded.py`

**What to build (CORRECTED 2026-04-28 per BLK-010.2; previous version had divergent field names):**

`ClassifierResult` dataclass — verbatim from spec Section 12.10 (lines 3447-3457):
```python
@dataclass
class ClassifierResult:
    p_final_calibrated: np.ndarray           # [7], post-Platt, sums to 1
    combined_argmax: int                     # 0-6 in canonical+OOD order
    combined_max_prob: float                 # max of p_final_calibrated
    combined_margin: float                   # max minus second-max
    p_final_uncalibrated: np.ndarray         # [7], pre-Platt, sums to 1 (for monitoring)
    p_stage1: np.ndarray                     # [3] healthy/diseased/OOD probs
    p_stage2: np.ndarray                     # [5] disease probs (only meaningful when stage1[diseased] is high)
    classifier_succeeded: bool               # False only if input was malformed
    failure_reason: str | None
# spec: 12.10 lines 3447-3457
```

7-class index space (Section 12.10 lines 3460-3467, "canonical with OOD"):
- 0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy, 6=OOD

`compute_classifier(sa: SignalAResult, sb: SignalBResult, sc: SignalCResult) -> ClassifierResult` (Section 12.1, line 3147 area). Note: input is the three signal results, not a flat feature vector. The 19-dim feature vector is built INSIDE `compute_classifier` per Section 12.2 (line 3169) using `build_classifier_input` from T-IMPL-4a.

- Stage 1: 3-way classifier (healthy / diseased / OOD) — Section 12.3 line 3249
- Stage 2: 5-way disease sub-classifier — Section 12.4 line 3279
- Soft routing combination — Section 12.5 line 3303: `P_final[disease_i] = P_stage1[diseased] × P_stage2[disease_i]`; healthy and OOD probabilities preserved from Stage 1
- Logistic default; MLP escalation rule — Section 12.6 line 3330
- Degraded-mode — Section 12.7 line 3348: zero-fill failed signal slices (per `forward_succeeded == False`); train Stage 1/2 with degraded augmentation
- Platt calibration — Section 12.8 line 3375: applied to `p_final_uncalibrated` to produce `p_final_calibrated`. Spec parameter count to verify against S12.8 body (BLK-006 deferred to implementer reading lines 3375-3407)
- Training procedure (out-of-fold) — Section 12.9 line 3408: not part of T-IMPL-4b; this task loads pre-trained pickles

**`combined_max_prob` is the field TTA reads** per Section 11.2 (cross-ref noted at spec line 3471).

**Acceptance criteria:**
- `dataclasses.fields(ClassifierResult)` lists exactly 9 fields with names from spec 12.10. Test asserts field-name set equality.
- `p_final_calibrated.shape == (7,)`, `p_final_uncalibrated.shape == (7,)`, both `abs(sum - 1.0) < 1e-5` when `classifier_succeeded == True`.
- Soft routing test: `p_stage1[diseased]=0.8`, `p_stage2=[0.5, 0.3, 0.1, 0.05, 0.05]` → `p_final_uncalibrated[0:5]` sums to 0.8 (the 5 disease classes); `p_final_uncalibrated[5]` = `p_stage1[healthy]`; `p_final_uncalibrated[6]` = `p_stage1[ood]`.
- `combined_argmax` is the argmax of `p_final_calibrated` (post-Platt, not pre).
- `combined_margin` = max minus second-max of `p_final_calibrated`.
- Platt-applied test: feed deliberately mis-calibrated logits; verify `p_final_calibrated` differs from `p_final_uncalibrated`.
- All-signals-failed test (Section 12.7 degraded mode): `sa.forward_succeeded=False`, `sb.forward_succeeded=False`, `sc.forward_succeeded=False` → `classifier_succeeded=False`, `failure_reason="all_signals_failed"`, all 7 probs zeroed except OOD which is 1.0 (or per Section 12.7 conv).
- Malformed input test: feed shape-(15,) feature vector → `classifier_succeeded=False`, `failure_reason="malformed_input: expected (19,) got (15,)"`.
- Platt params loaded from config path (not hardcoded); model loader reads `tomato_sandbox/calibration/classifier_platt.json` per Section 13.5 hash-validation contract.

**Estimated effort:** 2h

---

#### T-IMPL-4c — Conformal Prediction (S13)
**Spec sections covered:** S13
**Dependencies:** T-IMPL-4b
**Files to create:**
- `tomato_sandbox/conformal/conformal.py`
- `tomato_sandbox/conformal/__init__.py`
- `tomato_sandbox/tests/unit/test_conformal.py`

**What to build:**
`ConformalResult` dataclass per S13:
```
prediction_set: set[str]      # class names (e.g. {"late_blight", "healthy"})
tau: float                     # threshold loaded from file
prediction_set_size: int
coverage_estimate: float
```
`compute_conformal_tau(calibration_probs: np.ndarray, calibration_labels: np.ndarray, alpha: float = 0.10) -> float`
- Nonconformity score = `1 - P[true_class]` (S13 spec).
- `np.quantile(scores, 1 - alpha, method="higher")` (S13 spec: method="higher").
- Output stored to `conformal_tau.json` with `{"tau": ..., "model_version": ..., "calibration_date": ...}`.

`apply_conformal(classifier_result: ClassifierResult, tau: float) -> ConformalResult`
- Prediction set = all classes i where `1 - combined_probs[i] <= tau`.

Startup: load tau from `/etc/tomato_sandbox/calibration/tomato_calibration.json`. File missing/corrupt at startup = server fails to start (S3.10). File modified during runtime = lazy-detect; log warning; use tau from memory.

**Acceptance criteria:**
- `compute_conformal_tau` test: calibration set achieving exactly 90% coverage; verify tau selected achieves >= 90%.
- Prediction set test: class with `1 - prob <= tau` is included; class with `1 - prob > tau` is excluded.
- `method="higher"` verified (quantile is at least the 90th-percentile value, not below it).
- Startup file-missing test: mock file absence; verify startup exception raised.
- Runtime file-corruption test: mock corrupt file; verify warning logged, tau from memory used.

**Estimated effort:** 2h

---

### BATCH 5 — T-IMPL-5: Decision Logic

#### T-IMPL-5a — Tier Assignment Rules (S14) — SCHEMA-DECISION TASK (Phase 3 Gate)
**Spec sections covered:** S14
**Dependencies:** T-IMPL-4c
**Files to create:**
- `tomato_sandbox/tier/tier_assignment.py`
- `tomato_sandbox/tier/rules.py`
- `tomato_sandbox/tier/__init__.py`
- `tomato_sandbox/tier/tier_rules.yaml`

**What to build:**
This task begins with a schema decision (BLK-005): derive `tier_rules.yaml` from Section 14 prose with traceability comments referencing Section 14 paragraph numbers.

`TierAssignment` dataclass per S14:
```
tier_label: str               # one of {"1","2","3A","3B","3C","3D","4A","4B"}
tier5_alert: bool             # evaluated INDEPENDENTLY of tier_label
tier5_reason: str | None
routing: str                  # "route_to_queue"|"resolve"|"escalate"
rule_fired: str               # name of the rule that produced this tier
confidence_display: float
iqa_tier_ceiling_applied: bool
```

`assign_tier(classifier_result: ClassifierResult, conformal_result: ConformalResult, iqa_result: IQAResult, psv_result: SignalCResult) -> TierAssignment`

Implements the 9-rule priority chain from Section 14 in strict order (first firing rule wins). **CORRECTED 2026-04-28** to match Section 14 spec verbatim (per BLK-009 / Defect-9.3, plus newly-flagged Defect-9.4 rule-numbering divergence):

- **Rule 1** — Any signal failure (`forward_succeeded == False` for v3, LoRA, or PSV) → **Tier 4B**.
- **Rule 2** — IQA REJECT (handled at the input gate per Section 6.4; not in tier scenarios).
- **Rule 3** — `psv_reliability < 0.40 (strict)` OR `chilli_leakage > 0.40 (strict)` → **Tier 3C**. **[Defect-9.3 fix: threshold is `> 0.40 strict`, NOT `>= 0.30 inclusive`. Earlier draft conflated this with the Tier 2 eligibility boundary at `chilli_leakage < 0.30`.]**
- **Rule 4** — `combined_max_prob < 0.45 (strict)` → **Tier 4A**.
- **Rule 5** — `prediction_set_size >= 3` → **Tier 3B**; `prediction_set_size == 0` → Tier 4A (empty-set sub-rule).
- **Rule 6** — `prediction_set_size == 2` → **Tier 3A**.
- **Rule 7** — `prediction_set_size == 1` with high confidence; sub-rules:
  - **7a** IQA DEGRADED → Tier 3D (capped by IQA)
  - **7b** Underpowered class (YLCV/mosaic flagged underpowered) → Tier 3A (downgrade from Tier 1)
  - **7c** Default → Tier 1 (clean definitive prediction). Tier 1 specific thresholds: `combined_max_prob >= 0.85`, `combined_margin >= 0.30`, `psv_reliability >= 0.50`, `chilli_leakage < 0.20 (strict)`, `IQA in {ACCEPTABLE, HIGH}`.
  - 7a precedes 7b precedes 7c.
- **Rule 8** — `prediction_set_size == 1` with lower confidence; sub-rules 8a/8b/8c mirror 7a/7b/7c. Default 8c → Tier 2. Tier 2 specific thresholds: `combined_max_prob >= 0.65`, `combined_margin >= 0.20`, `psv_reliability >= 0.40`, `chilli_leakage < 0.30 (strict)`.
- **Rule 9** — Catch-all (anything else) → **Tier 4A**.

**`rule_fired` literal strings** (per Section 14 conventions, used in Section 15 scenarios):
- `signal_failure_rule1` (Rule 1)
- `psv_unreliable_or_chilli_leakage` (Rule 3)
- `low_confidence_rule4` (Rule 4)
- `prediction_set_large_rule5` / `prediction_set_empty_rule5` (Rule 5)
- `set_size_two_rule6` (Rule 6)
- `iqa_degraded_cap_7a` / `underpowered_downgrade_7b` / `definitive_single_class` (Rule 7c)
- `iqa_degraded_cap_8a` / `underpowered_downgrade_8b` / `confident_single_class` (Rule 8c)
- `catch_all_low_confidence` (Rule 9)

**Tier 5 alert** (evaluated INDEPENDENTLY of tier_label, per Section 14.3):
- Bullet 1: `argmax in {late_blight, ylcv, mosaic} AND max_prob >= 0.20` → T5 fires.
- Bullet 2: `late_blight in conformal_set AND P_late_blight >= 0.20` → T5 fires.
- Mosaic and YLCV do NOT have in-set T5 triggers (only argmax-trigger via bullet 1).

`tier_rules.yaml` schema example (schema_version: 1):
```yaml
schema_version: 1
# Source: tomato_3_signal_system.md Section 14, Rules R1-R9
# BLK-005: schema derived from Section 14 prose; Appendix D absent from spec file
rules:
  - id: R1
    # Section 14, paragraph [paragraph number]: all signals failed condition
    condition: "all_signals_failed"
    tier: "4B"
    routing: "resolve"
  # ... all 9 rules + sub-rules with paragraph citations
```

**CRITICAL:** `assign_tier()` must be implementation-complete (not stubbed) before Phase 3 begins.

**Acceptance criteria:**
- `tier_rules.yaml` exists, parses cleanly, has `schema_version: 1`, each rule has a comment citing S14 paragraph number.
- `assign_tier()` covers all 9 rules verified by synthetic inputs designed to fire each rule.
- IQA ceiling test: `iqa_result.decision="DEGRADED"` → `tier_label` not in {"1", "2"}.
- Tier 5 independence test: Tier 1 result with late_blight above threshold → `tier5_alert=True`, `tier_label="1"`.
- Chilli leakage Rule 3 strict-> boundary (PATCHED 2026-04-28 per Auditor RD-1): `chilli_leakage=0.41` → Rule 3 fires (Tier 3C); `chilli_leakage=0.40` → Rule 3 does NOT fire (strict `>`); `chilli_leakage=0.30` → Tier 2 boundary check (Rule 8 chilli_leakage `< 0.30` fails, so case is NOT eligible for Tier 2 default 8c — falls through to Rule 9). Earlier criterion at 0.30 was inverted; now matches Section 14 spec.
- 100% branch coverage required (S26).

**Estimated effort:** 4h

---

#### T-IMPL-5b — Decision Scenario Smoke Tests (S15 subset, 10 tests)
**Spec sections covered:** S15
**Dependencies:** T-IMPL-5a
**Files to create:**
- `tomato_sandbox/tests/integration/test_tier_smoke.py`

**What to build:**
10 manually written integration tests covering one scenario from each of the 10 scenario subsection groups in S15. These pre-Phase-3 smoke tests verify basic plumbing before the section15-encoder encodes all 135.

Scenarios (one per group; PATCHED 2026-04-28 per Auditor RD-2 — 0.30 boundary tests + "R2" rule name were stale from before Defect-9.3/9.4 patches):
- S1 group: S1.1 standard high-confidence healthy (Tier 1 expected via Rule 7c)
- S2 group: margin-boundary scenario (Tier 2 expected via Rule 8c)
- S3B group: S3B.4 (edge case that produces 4A from within the 3B section per S15 summary)
- S4 group: multi-image aggregate scenario
- S5 group: IQA DEGRADED ceiling — Rule 7a or 8a fires → Tier 3D (NOT Tier 3C as earlier draft said)
- SB group SB.7: chilli_leakage=0.40 boundary trap — Rule 3 strict `>` does NOT fire at exactly 0.40; expected outcome per Section 15.12 SB.7 is Tier 4A via Rule 9 catch-all
- SB group SB.13: chilli_leakage=0.20 boundary at Tier 1 chilli threshold — Rule 8 chilli check passes (since 0.20 < 0.30 strict but Tier 1 requires < 0.20 strict so Tier 1 path 7c fails); expected per S15.12 SB.13
- SUP group: underpowered class guard fires (sub-rule 7b or 8b → Tier 3A)
- SDIS group: disease-specific scenario
- STTA group: TTA-triggered scenario (post-TTA max < 0.65 → Rule 9 → Tier 4A per S15.15 STTA.3)

Each test: construct synthetic inputs, call `assign_tier()`, assert expected `tier_label`.

**Acceptance criteria:**
- All 10 smoke tests pass.
- SB.7 (chilli_leakage=0.40 exactly) → Rule 3 does NOT fire (strict `>`); falls through to Rule 9 → Tier 4A. **[PATCHED 2026-04-28: was wrongly stated as `chilli_leakage=0.30 → R2 fires`. Section 15.12 SB.7 confirms 0.40 boundary trap → Tier 4A.]**
- SB.13 (chilli_leakage=0.20 exactly) → Tier 1 chilli check `< 0.20 strict` does NOT pass, so Tier 1 not achievable; Tier 2 chilli `< 0.30 strict` passes; outcome depends on other Tier 2 conditions. Per Section 15.12 SB.13, expected outcome is Tier 2 admit (since `0.20 < 0.30`). **[PATCHED 2026-04-28: was wrongly stated as `0.29 → guard does not fire`.]**
- S3B.4 edge case correctly produces Tier 4A (not 3B).
- Rule numbering verified: tests cite "Rule 3", "Rule 7a", "Rule 8c", "Rule 9" — not "R2", "R3", etc. (older draft used informal numbering inconsistent with Section 14.)
- Tests are independent (no shared mutable state).

**Estimated effort:** 2h

---

#### T-IMPL-5c — Severity Grading (S17)
**Spec sections covered:** S17
**Dependencies:** T-IMPL-5a
**Files to create:**
- `tomato_sandbox/severity/grader.py`
- `tomato_sandbox/severity/__init__.py`
- `tomato_sandbox/severity/treatment_templates.yaml`
- `tomato_sandbox/tests/unit/test_grader.py`

**What to build:**
`SeverityResult` dataclass per S17:
```
grade: str | None              # "mild"|"moderate"|"severe"|null
coverage_pct: float | None
human_readable: str
recommended_action: str
grade_per_class: dict | None   # for multi-class cases (Tier 3A)
```
`compute_severity(tier_assignment: TierAssignment, signal_c: SignalCResult, classifier_result: ClassifierResult) -> SeverityResult`
- Severity computed only when tier_label in {"1", "2", "3A"} (S17 omission conditions).
- Uses PSV features G2/G3/G4/G7/G8 per S17 (specific PSV feature indices from S10).
- Per-disease thresholds from S17: foliar, septoria, late_blight, ylcv, mosaic.
- `treatment_templates.yaml`: one template entry per disease (5 entries).
- Tier 3A: populates `grade_per_class` with per-class severity.
- Omission conditions (grade=null): tier not in {1, 2, 3A}; OOD class; healthy prediction.

**Acceptance criteria:**
- Severity omitted for Tier 4A (grade=null, human_readable still populated).
- Per-class severity populated for Tier 3A input.
- Per-disease threshold test: synthetic PSV features at mild/moderate/severe boundary for late_blight.
- `treatment_templates.yaml` parseable with all 5 disease entries.

**Estimated effort:** 2h

---

#### T-IMPL-5d — Multi-Image Input Aggregation (S18)
**Spec sections covered:** S18
**Dependencies:** T-IMPL-5a, T-IMPL-5c
**Files to create:**
- `tomato_sandbox/orchestrator/multi_image.py`
- `tomato_sandbox/tests/unit/test_multi_image.py`
- `tomato_sandbox/tests/integration/test_multi_image_integration.py`

**What to build:**
7-step aggregation algorithm per S18:
1. Validate each image independently (T-IMPL-2a); REJECT per image, not batch.
2. Per-image IQA.
3. Per-image preprocessing (all three variants).
4. Per-image signal computation (A, B, C).
5. Aggregate signal probabilities (weighted mean per S18 formula).
6. Run classifier and conformal on aggregated signals.
7. Assign tier from aggregated result.

`POST /predict_multi` endpoint delegated from T-IMPL-7a to this module.
N range: 1-5 images; `TOMATO_MULTI_IMAGE_MAX_N` from config.
Per-image timeout: 5s (S18).
IQA REJECT for individual image: exclude that image, continue with remaining.
All images rejected: return Tier 4B.

**Acceptance criteria:**
- N=1 produces same result as single-image pipeline.
- N=3 where image 2 is IQA-REJECTED: aggregation uses images 1 and 3 only.
- All-images-rejected → Tier 4B.
- Per-image timeout test: mock image 3 to take 6s; verify it is dropped.
- Multi-image max N=5 enforced (N=6 → HTTP 400).

**Estimated effort:** 4h

---

### BATCH 6 — T-IMPL-6: Orchestrator and Response Builder

#### T-IMPL-6a — Response Builder (S16)
**Spec sections covered:** S16
**Dependencies:** T-IMPL-5a, T-IMPL-5c
**Files to create:**
- `tomato_sandbox/api/response_builder.py`
- `tomato_sandbox/tests/unit/test_response_builder.py`

**What to build:**
`build_response(tier, classifier, conformal, severity, iqa, context) -> dict`
- Produces the full JSON response envelope per S16 schema.
- All S16 fields present; null where spec says null.
- `tier5_alert` block populated when `tier.tier5_alert == True`.
- `agronomist_queue_block`: populated per tier routing rules S16.8.
- `agronomist_priority_hint`: derived from Tier 5 alert level and tier_label (S16.7).
- `confidence_display` rules per S16.
- GradCAM++ fields: `gradcam_url` null in v1 sandbox; `gradcam_alpha=0.5` default.
- Error response schema per S16.9.

**Acceptance criteria:**
- All S16 required fields present in output for Tier 1 case.
- Tier 5 alert block populated correctly for late_blight above threshold.
- Queue routing per Section 16 verbatim (CORRECTED 2026-04-28 per BLK-010.3):
  - **Tier 1, 2** → NOT routed unless Tier 5 also fires. # spec: 16 line 5856 (also see 7020 routing matrix)
  - **Tier 3A, 3B, 3C, 3D** → routed (subject to per-tier priority hint per Section 16.7).
  - **Tier 4A** → routed only if Tier 5 also fires; otherwise queued only on user opt-in. # spec: 16 line 5856 verbatim "Tier 4A -> routed if Tier 5 also fires; otherwise queued only on user opt-in"
  - **Tier 4B** → routed (system failure cases need agronomist review).
  - Earlier draft said "Tier 4A input → `route_to_queue=True`" unconditionally — that was incorrect; spec requires conditional routing on Tier-5-fire-or-opt-in.
- Null fields for Tier 4B (severity=null, gradcam=null).
- Output is JSON-serializable (no raw numpy arrays or datetime objects).

**Estimated effort:** 2h

---

#### T-IMPL-6b — Pipeline Orchestrator (S21)
**Spec sections covered:** S21
**Dependencies:** T-IMPL-6a, T-IMPL-5d, T-IMPL-2b
**Files to create:**
- `tomato_sandbox/orchestrator/orchestrator.py`
- `tomato_sandbox/orchestrator/nan_guards.py`
- `tomato_sandbox/orchestrator/degraded_mode.py`
- `tomato_sandbox/orchestrator/__init__.py`
- `tomato_sandbox/tests/unit/test_nan_guards.py`
- `tomato_sandbox/tests/unit/test_degraded_mode.py`
- `tomato_sandbox/tests/integration/test_orchestrator.py`

**What to build:**
`PipelineContext` dataclass per S21:
```
request_id: str
image_id: str
timestamp: datetime
signal_a_status: str
signal_b_status: str
signal_c_status: str
tta_triggered: bool
iqa_decision: str
tier_label: str
latency_total_ms: float
cache_hit: bool
```
`predict_single(validated_image: ValidatedImage, app_state: Any) -> dict`
22-step ordered pipeline per S21:
1. Check request cache (SHA256 key, LRU 1000 entries, TTL 3600s).
2. Acquire GPU lock (asyncio.Lock on app.state.gpu_lock).
3. IQA.
4. All-signals-failed short-circuit check (if IQA predicts total failure, return early).
5. Preprocessing (all three variants).
6. Signal A (under GPU lock).
7. Signal B (under GPU lock).
8. GPU lock released before PSV starts (PSV is CPU-only).
9. Signal C.
10. TTA check; apply TTA if triggered.
11. Build classifier input (remap applied here per T-IMPL-4a).
12. Run classifier.
13. Conformal.
14. Tier assignment.
15. Severity.
16. Build response.
17. Log to SQLite (if storage enabled).
18. Update Prometheus metrics.
19. Release GPU lock (in finally block; always released).
20. Store to request cache.
21. Return response.

`predict_multi`: delegates to `multi_image.py` (T-IMPL-5d).
`zero_failed_signals(context)`: degraded mode implementation.
All-signals-failed short-circuit: skip classifier/conformal/tier/severity; return Tier 4B directly.
`nan_guards.py`: NaN check after classifier; force Tier 4B on NaN (S21 pseudocode).
`degraded_mode.py`: `zero_failed_signals()` implementation; 100% coverage required (S26).

**Acceptance criteria:**
- Cache hit test: submit same image twice; second call returns cache hit.
- GPU lock test: mock two concurrent calls; second waits for first to release lock.
- All-signals-failed short-circuit: all three signals fail → Tier 4B, classifier not called.
- NaN guard module: 100% branch coverage (S26 requirement).
- Degraded mode module: 100% branch coverage (S26 requirement).
- GPU lock released in finally block even when exception occurs.

**Estimated effort:** 4h

---

### BATCH 7 — T-IMPL-7: Server and Routing

#### T-IMPL-7a — Server Endpoints Fully Wired (S20)
**Spec sections covered:** S20 (all 7 endpoints wired)
**Dependencies:** T-IMPL-6b, T-IMPL-1b
**Files to modify:**
- `tomato_sandbox/api/server.py` (extend skeleton from T-IMPL-1b)
**Files to create:**
- `tomato_sandbox/tests/e2e/test_endpoints.py`

**What to build:**
Wire all 7 endpoints per S20:
1. `POST /predict` → `predict_single()` via orchestrator. Validate input first. Return JSON.
2. `POST /predict_multi` → `predict_multi()`. Accept 1-N files.
3. `GET /health` → `{"status":"ok","model_loaded":true,"gpu_available":true}` (or false).
4. `GET /ready` → 200 when all 12 startup steps complete; 503 until ready.
5. `GET /info` → `{"model_version":"...","multi_image_max_n":5,"server_port":8767}`. `multi_image_max_n` from `TOMATO_MULTI_IMAGE_MAX_N`.
6. `GET /metrics` → Prometheus text format (T-IMPL-9a wires the metrics; endpoint stubbed here).
7. `GET /docs` → FastAPI Swagger UI.

12-step startup sequence wired per S4:
Step 1: verify sacred files (call T-IMPL-1a `verify_manifest()`; log CRITICAL and fail if any FAIL)
Steps 2-12: load config, models, PSV weights, conformal tau (fail-fast if missing), Platt params, severity thresholds, warm GPU.

GPU 503: `HTTP 503 {"error": {"code": "GPU_LOCK_TIMEOUT", "message": "Server busy; retry shortly."}}` per S3.10.
CUDA OOM: mark `needs_reload`, return Tier 4B with retry suggestion per S3.10.

**Acceptance criteria:**
- All 7 endpoints return correct status codes in e2e tests using HTTPX test client.
- Missing conformal tau at startup → server startup raises exception (does not silently continue).
- `POST /predict` returns Tier 4B when all signals fail (not HTTP 500).
- `GET /ready` returns 503 before startup, 200 after.
- Port is 8767 in all configuration (BLK-002); 8766 never referenced in `tomato_sandbox/`.
- No import from `scripts/apin/` (BLK-003).

**Estimated effort:** 4h

---

#### T-IMPL-7b — Unified Server Routing Client (S22)
**Spec sections covered:** S22
**Dependencies:** T-IMPL-7a
**Files to create:**
- `tomato_sandbox/routing/unified_client.py`
- `tomato_sandbox/routing/__init__.py`
- `tomato_sandbox/tests/unit/test_unified_client.py`

**What to build:**
HTTP client per S22 (BLK-003: HTTP-client only; no APIN library import).

`UnifiedRoutingClient`:
- Uses `httpx.AsyncClient` to call unified server at port 8005.
- Feature flag: `UNIFIED_TOMATO_ROUTE_ENABLED` env var. When False, returns bypass result without HTTP call.
- Request wrapping: add unified server envelope per S22.
- Response unwrapping: strip envelope, return `SandboxResponse`.
- Timeout: 10s; retry: 1 retry on network error only (not on 5xx).

**Acceptance criteria:**
- Flag disabled test: `UNIFIED_TOMATO_ROUTE_ENABLED=false` → no HTTP call made.
- Envelope wrap/unwrap test: synthetic response with envelope → correctly unwrapped.
- Timeout test: mock server takes 11s → `httpx.TimeoutException` raised and logged.
- No import from `scripts/apin/` (BLK-003).
- No import from `app/` (cross-layer violation).

**Estimated effort:** 2h

---

### BATCH 8 — T-IMPL-8: Persistence

#### T-IMPL-8a — SQLite Phase E Logger (S24)
**Spec sections covered:** S24
**Dependencies:** T-IMPL-6b
**Files to create:**
- `tomato_sandbox/storage/sqlite_logger.py`
- `tomato_sandbox/storage/__init__.py`
- `tomato_sandbox/storage/migrations/001_initial.sql`
- `tomato_sandbox/tests/unit/test_sqlite_logger.py`

**What to build:**
SQLite database at `/var/lib/tomato_sandbox/sandbox.db` (path from config; tests use temp path).
WAL mode enabled at startup.
4 tables per S24:
1. `predictions`: request_id, timestamp, tier_label, argmax_class, max_prob, latency_ms, image_hash, iqa_decision, cache_hit
2. `queue_cases`: all `QueueCase` columns per S23 dataclass
3. `per_signal_logs`: prediction_id, signal_name, status, latency_ms, top_prob
4. `metrics_snapshots`: timestamp, metric_name, value

Single-transaction atomicity for predictions write (S3.10).
Write failure handling: log warning, continue returning response. Never fail request due to SQLite write failure.
Retention: `TOMATO_DB_RETENTION_DAYS` env var; background pruning.

`log_prediction(context: PipelineContext, tier: TierAssignment, response: dict) -> None`
`log_queue_case(case: QueueCase) -> None`

**Acceptance criteria:**
- WAL mode verified: `PRAGMA journal_mode` returns `"wal"`.
- Transaction atomicity test: mock mid-transaction failure; partial write does not exist.
- Write failure test: mock SQLite write to throw; response still returned (warning logged).
- Retention pruning test: old timestamp rows removed by pruning job.
- Temp-path usage in all tests.

**Estimated effort:** 2h

---

#### T-IMPL-8b — Agronomist Queue Endpoints (S23)
**Spec sections covered:** S23
**Dependencies:** T-IMPL-8a, T-IMPL-6a
**Files to create:**
- `tomato_sandbox/queue/queue_models.py`
- `tomato_sandbox/queue/queue_service.py`
- `tomato_sandbox/queue/queue_router.py`
- `tomato_sandbox/queue/__init__.py`
- `tomato_sandbox/tests/unit/test_queue_service.py`

**What to build:**
`QueueCase` dataclass per S23.3 (all fields verbatim from spec).
`QueueDisposition` dataclass per S23.6 (all fields verbatim from spec).
7 API endpoints per S23.5:
1. `GET /queue/cases` — list pending, sortable by priority/age
2. `GET /queue/cases/{case_id}` — fetch single case
3. `POST /queue/cases/{case_id}/claim` — pending → in_review
4. `POST /queue/cases/{case_id}/resolve` — resolve with disposition
5. `POST /queue/cases/{case_id}/dismiss`
6. `POST /queue/cases/{case_id}/escalate`
7. `GET /queue/stats` — pending count, P50/P95 review time

Status FSM: `pending → in_review → resolved/dismissed/escalated`.
Stale claim reset: cases in `in_review` for > 24h reset to `pending` (daily background job).
Queue capacity alerts: > 200 pending for > 4h = MEDIUM alert; > 500 pending = HIGH alert.
`route_ambiguous_to_queue` config flag (default False per S23).
Queue router registered on FastAPI app in T-IMPL-7a.

**Acceptance criteria:**
- All 7 endpoints return correct shapes in unit tests.
- Status FSM: `pending → resolved` directly is blocked (must claim first).
- Stale claim reset test: case with `assigned_at` 25h ago reset to `pending` after job runs.
- Capacity alert test: 201 cases aged > 4h → MEDIUM alert logged.

**Estimated effort:** 4h

---

### BATCH 9 — T-IMPL-9: Monitoring and OpenAPI

#### T-IMPL-9a — Prometheus Metrics (S25)
**Spec sections covered:** S25
**Dependencies:** T-IMPL-6b, T-IMPL-7a
**Files to create:**
- `tomato_sandbox/monitoring/metrics.py`
- `tomato_sandbox/monitoring/thresholds.py`
- `tomato_sandbox/monitoring/__init__.py`
- `tomato_sandbox/tests/unit/test_metrics.py`

**What to build:**
20+ Prometheus metrics per S25 (5 categories):
1. Latency: `tomato_request_latency_seconds` (Histogram; buckets: 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
2. Tier distribution: `tomato_tier_total` (Counter; label: tier_label)
3. Signal health: `tomato_signal_status_total` (Counter; labels: signal_name, status)
4. Conformal coverage: `tomato_conformal_prediction_set_size` (Histogram)
5. Queue: `tomato_queue_pending_total` (Gauge), `tomato_queue_review_seconds` (Histogram)

SLO thresholds in `thresholds.py` (Python constants matching S25 values):
- P95 latency SLO target: 550ms; alert threshold: 1000ms CRITICAL
- Tier 4B rate alert: > 3% CRITICAL
- Conformal coverage drift: < 85% MEDIUM

`GET /metrics` endpoint wired in T-IMPL-7a to return prometheus_client text format.

**Acceptance criteria:**
- `GET /metrics` returns valid Prometheus text format.
- All 5 metric categories present.
- Counter increments verified in unit tests.
- `thresholds.py` constants match S25 values exactly.

**Estimated effort:** 2h

---

#### T-IMPL-9b — OpenAPI Documentation (S27)
**Spec sections covered:** S27
**Dependencies:** T-IMPL-7a, T-IMPL-8b
**Files to create:**
- `tomato_sandbox/api/openapi.yaml` (auto-generated and committed)
- `tomato_sandbox/api/schemas.py` (Pydantic models for OpenAPI)

**What to build:**
Pydantic schemas in `schemas.py` matching S27.4 schema list:
- `PredictRequest`, `PredictMultiRequest`
- `SandboxResponse` (unwrapped, S16.2), `UnifiedResponse` (envelope, S16.10)
- `ErrorResponse` (S16.9)
- `TierBlock`, `PredictionBlock`, `Tier5AlertBlock`, `SeverityBlock`, `ExplanationBlock`, `VisualizationBlock`, `AgronomistQueueBlock`
- `QueueCase`, `QueueDisposition`, `InfoResponse`

Generate `openapi.yaml` from FastAPI app using `app.openapi()` and commit it.
Update `.pre-commit-config.yaml` to add openapi.yaml regeneration check.
`openapi-spec-validator` CI check in `Makefile`.

**Acceptance criteria:**
- `openapi.yaml` is valid per `openapi-spec-validator` (run via `make lint` or `make validate-openapi`).
- All S27.4 schemas present in generated spec.
- Pre-commit hook rejects commit where `openapi.yaml` is out of sync.
- Swagger UI at `GET /docs`.
- `X-API-Version` header documented in spec (S27.6).

**Estimated effort:** 2h

---

### BATCH 10 — T-IMPL-10: Deployment Documentation

#### T-IMPL-10a — Deployment Runbooks (S28)
**Spec sections covered:** S28
**Dependencies:** T-IMPL-7a
**Files to create:**
- `tomato_sandbox/ops/systemd_unit.service`
- `tomato_sandbox/ops/bringup_procedure.md`
- `tomato_sandbox/ops/runbooks/latency_runbook.md`
- `tomato_sandbox/ops/runbooks/conformal_coverage_runbook.md`
- `tomato_sandbox/ops/runbooks/queue_capacity_runbook.md`
- `tomato_sandbox/ops/env.template`

**What to build:**
`systemd_unit.service`: exact unit file per S28 template:
- User=tomato-sandbox, WorkingDirectory=/opt/tomato_sandbox
- ExecStart=.venv/bin/uvicorn tomato_sandbox.api.server:app --host 127.0.0.1 --port 8767 --workers 1
- Restart=on-failure, RestartSec=10s

`bringup_procedure.md`: 13 numbered steps from S28.5.
`latency_runbook.md`: reduce `TOMATO_MULTI_IMAGE_MAX_N` to 3 under load per S28 latency runbook.
`conformal_coverage_runbook.md`: per S25.6.
`queue_capacity_runbook.md`: per S23.7 SLA targets.
SEV1/SEV2/SEV3 classification in `bringup_procedure.md`.
Safety valve documented: `UNIFIED_TOMATO_ROUTE_ENABLED=false` rollback.
`env.template`: all `TOMATO_*` env vars with comments.

Note: actual deployment is out of scope for Phase 4.

**Acceptance criteria:**
- `systemd_unit.service` is valid INI format with all required sections ([Unit], [Service], [Install]).
- `bringup_procedure.md` has exactly 13 numbered steps.
- All `TOMATO_*` env vars from S4, S22, S23, S24, S25, S28 are in `env.template`.
- All port numbers in runbooks use correct values: 8767 sandbox, 8005 unified, 8768 queue, 8766 APIN.

**Estimated effort:** 2h

---

#### T-IMPL-10b — Phase F.0 Procedure Docs (S29)
**Spec sections covered:** S29, S32 (quality bars from HB-1)
**Dependencies:** T-IMPL-5a
**Files to create:**
- `tomato_sandbox/ops/f0_procedure.md`
- `tomato_sandbox/ops/f0_quality_bars.yaml`

**What to build:**
`f0_procedure.md`: 6-step F.0 procedure per S29.3 (calibration run, fit params, evaluate test set, run 135 scenarios, produce report, go/no-go decision). Notes that actual F.0 run is out of scope for Phase 4 sandbox implementation; requires the F.0 dataset (~3000 images).

`f0_quality_bars.yaml`: all quality bars from S29.4 with hard floors and targets. Example structure:
```yaml
schema_version: 1
# Source: tomato_3_signal_system.md Section 29.4
quality_bars:
  overall_accuracy: {target: 0.80, hard_floor: 0.70}
  per_class_f1:
    foliar: {target: 0.80, hard_floor: 0.70}
    septoria: {target: 0.80, hard_floor: 0.70}
    late_blight: {target: 0.75, hard_floor: 0.65}
    ylcv: {target: 0.65, hard_floor: 0.55}
    mosaic: {target: 0.65, hard_floor: 0.55}
    healthy: {target: 0.80, hard_floor: 0.70}
  conformal_coverage: {target_min: 0.88, target_max: 0.92, hard_floor: 0.85, hard_ceiling: 0.95}
  tier_4b_rate: {target: 0.01, hard_floor: 0.03}
  tier5_alert_precision: {target: 0.70, hard_floor: 0.50}
  tier5_alert_recall: {target: 0.90, hard_floor: 0.80}
  calibration_ece: {target: 0.05, hard_floor: 0.10}
  section15_scenarios: {target: 1.00, hard_floor: 1.00}
note: "Hard floors are absolute. Any metric below hard floor blocks pilot go. See spec Section 29.4 and Section 32 HB-1."
```

**Acceptance criteria:**
- `f0_quality_bars.yaml` parses and all values match S29.4 verbatim.
- `f0_procedure.md` lists all 6 steps from S29.3 with S29 section references.
- Hard floor disclaimer present.
- Note that actual F.0 run requires the F.0 dataset and is not part of Phase 4.

**Estimated effort:** 1h

---

## Spec Section Coverage Matrix

Every spec section is claimed by at least one task.

| Spec Section | Covered By |
|---|---|
| S1 (Document scope) | T-EARLY-MP (BLK-002, BLK-003 baked per DEC-012) |
| S2 (Sacred manifest, class indices) | T-IMPL-1a, T-EARLY-MP Fix-1 |
| S3 (Architectural overview) | T-IMPL-1b (startup), T-IMPL-6b (failure table) |
| S4 (Component inventory, startup) | T-IMPL-1b (skeleton), T-IMPL-7a (12-step startup wired) |
| S5 (Input validation) | T-IMPL-2a |
| S6 (IQA) | T-IMPL-2b |
| S7 (Preprocessing) | T-IMPL-2c |
| S8 (Signal A v3) | T-IMPL-3a |
| S9 (Signal B LoRA) | T-IMPL-3b |
| S10 (Signal C PSV) | T-IMPL-3c |
| S11 (TTA) | T-IMPL-3d |
| S12 (Hierarchical classifier) | T-IMPL-4a, T-IMPL-4b |
| S13 (Conformal prediction) | T-IMPL-4c |
| S14 (Tier assignment) | T-IMPL-5a |
| S15 (135 scenarios) | T-IMPL-5b (10 smoke), T-PHASE-3-PRECONDITIONS (gate), Phase 3 (full 135) |
| S16 (Response builder) | T-IMPL-6a |
| S17 (Severity grading) | T-IMPL-5c |
| S18 (Multi-image) | T-IMPL-5d |
| S19 (Frontend integration) | Reference only — frontend lives outside sandbox per S19 limitation 5 |
| S20 (Sandbox server architecture) | T-IMPL-1b, T-IMPL-7a |
| S21 (Pipeline orchestrator) | T-IMPL-6b |
| S22 (Unified server routing) | T-IMPL-7b |
| S23 (Agronomist queue) | T-IMPL-8b |
| S24 (SQLite storage) | T-IMPL-8a |
| S25 (Monitoring) | T-IMPL-9a |
| S26 (Engineering hygiene) | T-IMPL-1c, coverage requirements in all task acceptance criteria |
| S27 (OpenAPI docs) | T-IMPL-9b |
| S28 (Deployment + operations) | T-IMPL-10a |
| S29 (Phase F.0 validation) | T-IMPL-10b |
| S30 (Consolidated limitations) | Acknowledged; v2 exclusion list at top of plan; no new code |
| S31 (Open questions) | Reference only; no Phase 4 implementation |
| S32 (Honest assessment) | T-IMPL-10b (quality bars per HB-1); T-PHASE-3-PRECONDITIONS (per HB-2) |
| Appendices A-F | Absent from spec file; T-IMPL-5a derives Appendix D; others are cross-reference only |

---

## Task Summary Table

**[EXPANDED 2026-04-28 from 5 columns to 9 per Auditor B1 / PVA SD-2 / PDA Defect-20.]** Owner subagent column reflects which agent the task is dispatched to in Phase 4. Priority reflects Phase-3-blocker status and severity.

| Task ID | Title | Owner subagent | Prerequisites | Spec sections | Files in `tomato_sandbox/` | Acceptance criteria (key) | Complexity | Priority |
|---|---|---|---|---|---|---|---|---|
| T-EARLY-MP | Master prompt defect remediation (25 fixes; HIGH-then-MEDIUM-then-LOW) | implementer (text-only) | None | meta | `tomato_master_prompt.md` (append-only block) + `.claude/agents/*.md` for Fix-9/10 sweep | All 25 fixes appended; sacred-guardian re-runs PASS; severity order verified | M | HIGH (Fix-16, Fix-19, Fix-20 block Phase 3) |
| T-EARLY-VENV | Sandbox venv creation (Python 3.13 + pinned deps) | implementer | None | S26 | `.venv/`, `requirements.txt`, `.gitignore` | `python --version == 3.13.x`; `torch.cuda.is_available() == True` | S | HIGH (blocks all T-IMPL) |
| T-PHASE-3-PRECONDITIONS | Phase 3 gate verification (5 gates) | phase-exit-auditor | T-IMPL-5a, T-IMPL-5b, T-EARLY-MP Fix-16+Fix-23 | S14, S15 | `tomato_progress_reports/phase3_gate_check.md` + `spec_changelog.md` BLK-004 entry | All 5 gates PASS; spec_changelog written | S | HIGH (gates Phase 3 entry) |
| T-IMPL-1a | Sacred manifest verifier module | implementer | T-EARLY-VENV | S2, S26 | `utils/sacred_guard.py`, `utils/__init__.py` | All 10 entries PASS; corrupting one returns FAIL; 100% test coverage | S | HIGH |
| T-IMPL-1b | FastAPI skeleton server | implementer | T-EARLY-VENV | S20, S4 | `api/server.py`, `api/__init__.py`, `config.py`, `config/default.yaml` | uvicorn starts on 8767; /health returns 200; /predict returns 503 | S | HIGH |
| T-IMPL-1c | Lint and test scaffolding | implementer | T-EARLY-VENV | S26 | `pyproject.toml`, `.pre-commit-config.yaml`, `tests/conftest.py` | `ruff check .` clean; `pytest tests/` returns no collection errors | S | MEDIUM |
| T-IMPL-2a | Input validation | implementer | T-IMPL-1b, T-IMPL-1c | S5 | `api/validate_input.py`, `tests/unit/test_validate_input.py` | Reject >10 MB; reject non-image; ValidatedImage dataclass returned | S | HIGH |
| T-IMPL-2b | IQA | implementer | T-IMPL-2a | S6 | `iqa/iqa.py`, `tests/unit/test_iqa.py` | accept/warn/reject decisions; degraded-mode emit logged | S | HIGH |
| T-IMPL-2c | Preprocessing (pp_classifier + pp_psv) | implementer | T-IMPL-2a | S7 | `preprocessing/preprocess.py`, `tests/unit/test_preprocess.py` | 224×224 normalized + LAB-CLAHE branches both produced | S | HIGH |
| T-IMPL-3a | Signal A wrapper (v3 10→6 with internal remap) | implementer | T-IMPL-2c, T-IMPL-1a | S8 | `signals/v3_signal.py`, `tests/unit/test_signal_a.py` | `tomato_probs_canonical.shape == (6,)`; remap-direction test; chilli_leakage formula | S | HIGH |
| T-IMPL-3b | Signal B wrapper (LoRA 6, native canonical) | implementer | T-IMPL-2c | S9 | `signals/lora_signal.py`, `tests/unit/test_signal_b.py` | 6-class probs, canonical ordering verified; cls_token returned | S | HIGH |
| T-IMPL-3c | Signal C PSV pipeline (26 features) | implementer | T-IMPL-2c | S10 | `signals/psv/*.py`, `tests/unit/test_psv.py` | 26 features computed; reliability score returned; BLK-007 traceability comments | M | HIGH |
| T-IMPL-3d | TTA (1/2/5-view, PSV excluded) | implementer | T-IMPL-3a, T-IMPL-3b | S11 | `signals/tta.py`, `tests/unit/test_tta.py` | `should_trigger_tta(0.60)==1`, `(0.50)==2`, `(0.40)==5`; PSV not invoked | S | HIGH |
| T-IMPL-4a | Classifier input builder (NO remap; canonical inputs) | implementer | T-IMPL-3a, T-IMPL-3b, T-IMPL-3c, T-IMPL-3d | S12 | `classifier/feature_builder.py`, `tests/unit/test_feature_builder.py` | shape (19,); remap-NOT-here regression test passes; degraded zeroing works | S | HIGH |
| T-IMPL-4b | Hierarchical classifier (Stage 1 + Stage 2 + soft routing + Platt) | implementer | T-IMPL-4a | S12 | `classifier/classifier.py`, tests | P_final 7-class; Platt-calibrated; soft-routing formula matches S12 | M | HIGH |
| T-IMPL-4c | Conformal prediction | implementer | T-IMPL-4b | S13 | `classifier/conformal.py`, tests | tau loaded from JSON; startup hash check refuses on mismatch; prediction_set construction | S | HIGH |
| T-IMPL-5a | Tier assignment + tier_rules.yaml (BLK-005 schema-decision; PHASE-3-GATE) | implementer | T-IMPL-4c | S14 | `tier/tier_assignment.py`, `tier/rules.py`, `tier/tier_rules.yaml`, tests | Rules 1-9 in priority order; chilli_leakage > 0.40 strict; YAML has S14 traceability comments | M | HIGH (Phase 3 gate) |
| T-IMPL-5b | Scenario smoke tests (10-of-135 subset) | implementer | T-IMPL-5a | S15 | `tests/integration/test_section15_smoke.py` | 10 hand-picked scenarios pass | S | HIGH |
| T-IMPL-5c | Severity grading | implementer | T-IMPL-5a | S17 | `severity/grader.py`, tests | Per-disease thresholds match S17.3 table; PSV-only computation | S | MEDIUM |
| T-IMPL-5d | Multi-image aggregation | implementer | T-IMPL-5a, T-IMPL-5c | S18 | `multi_image/aggregator.py`, tests | N up to 5; per-image conformal applied; aggregation per S18 | M | MEDIUM |
| T-IMPL-6a | Response builder | implementer | T-IMPL-5a, T-IMPL-5c | S16 | `responses/builder.py`, `responses/templates.yaml`, tests | UnifiedResponse envelope; tier-conditional blocks correct | S | HIGH |
| T-IMPL-6b | Pipeline orchestrator | implementer | T-IMPL-6a, T-IMPL-5d, T-IMPL-2b | S21 | `orchestrator/pipeline.py`, `orchestrator/degraded_mode.py`, tests | 18-step request flow; GPU lock honored; per-step Phase E logging | M | HIGH |
| T-IMPL-7a | Server endpoints fully wired | implementer | T-IMPL-6b, T-IMPL-1b | S20 | `api/server.py` (extend), endpoint tests | All 7 endpoints from S20 return correct contracts | M | HIGH |
| T-IMPL-7b | Unified server routing client | implementer | T-IMPL-7a | S22 | `clients/unified_routing.py`, tests | UNIFIED_TOMATO_ROUTE_ENABLED flag respected; HTTP-only (no APIN library import per BLK-003) | S | MEDIUM |
| T-IMPL-8a | SQLite Phase E logger | implementer | T-IMPL-6b | S24 | `storage/sqlite_logger.py`, tests | Per-step records persisted; schema migration script | S | MEDIUM |
| T-IMPL-8b | Agronomist queue endpoints (7 endpoints) | implementer | T-IMPL-8a, T-IMPL-6a | S23 | `queue/endpoints.py`, `queue/dataclasses.py`, tests | All 7 endpoints (incl. /queue/stats); QueueCase + QueueDisposition match S23 | M | MEDIUM |
| T-IMPL-9a | Prometheus metrics | implementer | T-IMPL-6b, T-IMPL-7a | S25 | `monitoring/metrics.py`, tests | Conformal coverage gauge; latency histograms; tier distribution counters | S | LOW |
| T-IMPL-9b | OpenAPI documentation | implementer | T-IMPL-7a, T-IMPL-8b | S27 | `api/openapi.yaml` (auto-generated) | Schema valid; CI check `openapi-spec-validator` passes | S | LOW |
| T-IMPL-10a | Deployment runbooks | implementer (docs only) | T-IMPL-7a | S28 | `docs/runbooks/*.md`, `deployment/systemd/tomato_sandbox.service` | systemd unit syntactically valid; runbooks cover bringup + 3 SEV severities | S | LOW |
| T-IMPL-10b | Phase F.0 procedure docs (no F.0 run) | implementer (docs only) | T-IMPL-5a | S29 | `docs/f0_procedure.md`, `config/f0_quality_bars.yaml` | quality_bars yaml matches S29.4 verbatim | S | LOW |

**Total tasks:** 30 (3 special + 27 T-IMPL sub-tasks)
**Total estimated effort:** 74h
**Owner subagent column note:** `implementer` is the Phase 4 default; T-PHASE-3-PRECONDITIONS is dispatched to `phase-exit-auditor` as a verification task.

---

## New Ambiguities Identified During Planning (BLK-006 through BLK-008)

These do not block Phase 4 but should be logged to `tomato_blockers.md`:

**BLK-006 (LOW):** S12 Platt calibration — spec summary says "14 parameters" but does not enumerate the parameter names or the formula. Section 12 body contains this detail. Recommended action: T-IMPL-4b implementer reads full S12 body before coding; no blocking.

**BLK-007 (LOW):** S10 PSV features — Appendix C (26-feature catalog) is absent from spec file per appendices.md. Section 10 body contains partial descriptions. Implementer must derive the full ordered 26-feature list from Section 10 body prose and document in `psv_features.py` docstring with paragraph citations. Same resolution pattern as BLK-005.

**BLK-008 (LOW):** S9 `prototype_blend()` blend formula coefficients — spec summary cites the signature but the exact blend weights are in the S9 body. Recommended action: T-IMPL-3b implementer reads full S9 body before coding; no blocking.

---

## Plan-Level Acceptance Criteria

This plan is considered complete and correct when:

1. All 32 spec sections are covered by at least one task (verified in coverage matrix above).
2. No v2 feature (per S30 exclusion list) is tasked.
3. All four BLK resolutions (DEC-012) are baked into specific task acceptance criteria: BLK-002 in T-IMPL-1b, BLK-003 in T-IMPL-2a and T-IMPL-7b, BLK-004 in T-IMPL-3a, BLK-005 in T-IMPL-5a.
4. Phase 3 gate is explicit and tied to T-IMPL-5a + T-IMPL-5b + T-EARLY-MP Fix-16 (HB-2 compliance).
5. Index remap annotation is present in the T-IMPL-3 batch header AND in T-IMPL-4a acceptance criteria.
6. T-EARLY-MP contains all 17 fix items (8 from Phase 0 PDA + 9 from Phase 1 exit Defect-9 through Defect-18 with Defect-17 and Defect-18 merged as Fix-17).
7. Every task's "Files to create or modify" lists paths under `tomato_sandbox/` only (except T-EARLY-MP which modifies `tomato_master_prompt.md` per DEC-007 append-only rule, and T-PHASE-3-PRECONDITIONS which writes to `tomato_progress_reports/`).
8. Sacred files list at top of plan matches the 10-entry manifest.
9. New ambiguities BLK-006 through BLK-008 are flagged for logging but do not block Phase 4.
10. S19 (frontend) is correctly marked as reference-only (frontend lives outside the Sandbox Directive scope per S19 limitation 5).
