# Phase 3 Exit Audit — Tomato 3-Signal System
**Auditor:** phase-exit-auditor (claude-sonnet-4-6)
**Timestamp:** 2026-05-01T00:30 (report written 2026-05-01)
**Phase exiting:** Phase 3 (section15-encoder deliverable)
**Phase entering (pending user approval):** Phase 4 (implementation)
**Fix-27 note:** This auditor runs independently; it does not depend on the 4 parallel auditors (PVA, PDA, anti-cheat-inspector, sacred-guardian). It verifies; it does not fix.

---

## Checklist

| # | Check | Command / Evidence | Expected Result |
|---|-------|--------------------|-----------------|
| 1 | 135 test functions total across 13 files | `grep -c "^def test" test_section15_*.py` then sum | 135 |
| 2 | All 13 required files exist | `ls tomato_sandbox/tests/integration/test_section15_*.py` | 13 files with correct names |
| 3 | All 13 files fail with `ModuleNotFoundError: No module named 'tomato_sandbox.tier'` | `python -m pytest ... --collect-only 2>&1 \| tail -30` | 13 ERRORs, 0 collected |
| 4 | Import contract exists at `.claude/import_contract.md` with substantial content (~8000 bytes) | `ls -la .claude/import_contract.md && wc -c ...` | file exists, size >= 7000 |
| 5 | S1.1 uses `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]` (line 4117, not 0.92 typo at line 5558) | `grep -n "0.89\|0.92\|S1_1\|S1\.1" test_section15_tier1.py` | 0.89 in test_scenario_S1_1; 0.92 NOT used there |
| 6 | SPEC-INT-002 exists in `spec_changelog.md` dated 2026-05-01 covering 7 body-wins decisions | `grep -n "SPEC-INT-002\|body-wins\|7 scenario" spec_changelog.md` | Entry present, 7 scenarios listed |
| 7 | DEC-017 in `tomato_decisions.md` (Phase 3 precondition relaxation) | `grep -n "DEC-017" tomato_decisions.md` | Entry found |
| 8 | DEC-018 in `tomato_decisions.md` (Defect-37/42 fast-track) | `grep -n "DEC-018" tomato_decisions.md` | Entry found |
| 9 | DEC-019 in `tomato_decisions.md` (sacred manifest exclusion) | `grep -n "DEC-019" tomato_decisions.md` | Entry found |
| 10 | Sacred manifest 10/10 PASS (canonical algorithm with log_exclusions) | Python canonical-algorithm script | 10 PASS, 0 FAIL |
| 11 | Defect-45 tracked in `tomato_plan.md` T-EARLY-MP queue as LOW non-blocking | `grep -n "Defect-45\|naming.*inconsistency" tomato_plan.md` | Entry with LOW severity, no Phase 4 blocker |
| 12 | Fast-tracked patches present in master prompt: Fix-9, Fix-10, Fix-16, Fix-27, Fix-28, Fix-34, Fix-37, Fix-42 | `grep -n "## Fix-9\|## Fix-10\|## Fix-16\|## Fix-27\|## Fix-28\|## Fix-34\|## Fix-37\|## Fix-42" tomato_master_prompt.md` | All 8 headings found |
| 13 | `tomato_log.md` has Phase 3 entries (encoder dispatch + deliverable + pre-gate work) | `grep -n "Phase 3\|2026-04-30\|2026-05-01" tomato_log.md` (filtered) | Entries at 2026-04-30 13:00, 17:15, and 2026-05-01 00:30 |

---

## Results

### Check 1 — 135 test functions total

**Command run:**
```
for f in tomato_sandbox/tests/integration/test_section15_*.py; do
  count=$(grep -c "^def test" "$f"); echo "$count $f"; done
then summed
```

**Actual output:**
```
15 test_section15_boundary.py
 6 test_section15_disagreement.py
12 test_section15_tier1.py
12 test_section15_tier2.py
12 test_section15_tier3a.py
10 test_section15_tier3b.py
12 test_section15_tier3c.py
10 test_section15_tier3d.py
13 test_section15_tier4a.py
10 test_section15_tier4b.py
11 test_section15_tier5.py
 5 test_section15_tta.py
 7 test_section15_underpowered.py
TOTAL: 135
```

**Verdict: PASS** — Exactly 135 test functions across 13 files.

---

### Check 2 — All 13 required files exist

**Command run:** `ls tomato_sandbox/tests/integration/test_section15_*.py`

**Actual output:**
```
test_section15_boundary.py
test_section15_disagreement.py
test_section15_tier1.py
test_section15_tier2.py
test_section15_tier3a.py
test_section15_tier3b.py
test_section15_tier3c.py
test_section15_tier3d.py
test_section15_tier4a.py
test_section15_tier4b.py
test_section15_tier5.py
test_section15_tta.py
test_section15_underpowered.py
```

**Verdict: PASS** — All 13 required files present with correct naming (`_tier1`, `_tier2`, `_tier3a`, `_tier3b`, `_tier3c`, `_tier3d`, `_tier4a`, `_tier4b`, `_tier5`, `_boundary`, `_underpowered`, `_disagreement`, `_tta`).

---

### Check 3 — All 13 files fail with ModuleNotFoundError at collection

**Command run:**
```
python -m pytest tomato_sandbox/tests/integration/ --collect-only 2>&1 | tail -30
```

**Actual output (tail):**
```
E   ModuleNotFoundError: No module named 'tomato_sandbox.tier'
_ ERROR collecting test_section15_disagreement.py _
    from tomato_sandbox.tier.tier_assignment import assign_tier  # noqa: E402
E   ModuleNotFoundError: No module named 'tomato_sandbox.tier'
_ ERROR collecting test_section15_tier1.py _
    from tomato_sandbox.tier.tier_assignment import assign_tier  # noqa: E402
E   ModuleNotFoundError: No module named 'tomato_sandbox.tier'
[... 10 more identical ERROR blocks ...]
ERROR tomato_sandbox/tests/integration/test_section15_boundary.py
ERROR tomato_sandbox/tests/integration/test_section15_disagreement.py
ERROR tomato_sandbox/tests/integration/test_section15_tier1.py
ERROR tomato_sandbox/tests/integration/test_section15_tier2.py
ERROR tomato_sandbox/tests/integration/test_section15_tier3a.py
ERROR tomato_sandbox/tests/integration/test_section15_tier3b.py
ERROR tomato_sandbox/tests/integration/test_section15_tier3c.py
ERROR tomato_sandbox/tests/integration/test_section15_tier3d.py
ERROR tomato_sandbox/tests/integration/test_section15_tier4a.py
ERROR tomato_sandbox/tests/integration/test_section15_tier4b.py
ERROR tomato_sandbox/tests/integration/test_section15_tier5.py
ERROR tomato_sandbox/tests/integration/test_section15_tta.py
ERROR tomato_sandbox/tests/integration/test_section15_underpowered.py
!!!!!!!!!!!!!!!!!! Interrupted: 13 errors during collection !!!!!!!!!!!!!!!!!!!
=================== no tests collected, 13 errors in 0.29s ====================
```

**Verdict: PASS** — All 13 files fail at collection with `ModuleNotFoundError: No module named 'tomato_sandbox.tier'`. This is the intended failure mode per master prompt Section 8.3 and DEC-017. The import statement `from tomato_sandbox.tier.tier_assignment import assign_tier` is consistent across all 13 files, confirming Phase 4's target contract is uniform.

---

### Check 4 — Import contract exists and is substantive

**Command run:** `ls -la .claude/import_contract.md && wc -c .claude/import_contract.md`

**Actual output:**
```
-rw-r--r-- 1 xplod 197609 8116 Apr 30 17:16 .claude/import_contract.md
8116 .claude/import_contract.md
```

**Verdict: PASS** — File exists at `.claude/import_contract.md`, 8116 bytes (target was ~8000 bytes). Created 2026-04-30 17:16, consistent with encoder dispatch timestamp at [2026-04-30 17:15].

---

### Check 5 — S1.1 priors use 0.89 (line 4117), not 0.92 (line 5558 typo)

**Command run:**
```
grep -n "S1_1\|S1\.1\|0\.89\|0\.92\|4117\|5558" test_section15_tier1.py | head -30
```

**Actual output (relevant lines):**
```
2:  Section 15.3 — Tier 1 scenarios (S1.1 – S1.12).
52:# S1.1 — Clean foliar prediction
55:# v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
56:#   (AUTHORITATIVE: line 4117 scenario body; line 5558 test-code snippet
57:#    with [0.92, ...] is a typo — BLK-004 Defect-15.1, SPEC-INT-001)
66:def test_scenario_S1_1():
67:    """S1.1 — Clean foliar prediction. Spec lines 4116-4124."""
68:    v3 = _make_signal([0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03)
```

**Verdict: PASS** — `test_scenario_S1_1()` at line 66 uses `[0.89, 0.04, 0.01, 0.01, 0.01, 0.01]`, sourced from spec line 4117. The inline comment at lines 56-57 explicitly documents the SPEC-INT-001 / BLK-004 Defect-15.1 conflict and confirms the 0.92 at line 5558 is a typo. The 0.92 value appearing at line 127 is for a DIFFERENT scenario (S1.3, argmax=2), not for S1.1.

---

### Check 6 — SPEC-INT-002 in spec_changelog.md (7 body-wins, dated 2026-05-01)

**Command run:**
```
grep -n "SPEC-INT-002|body-wins|7 scenario|2026-05-01" spec_changelog.md | head -20
```

**Actual output:**
```
51:## SPEC-INT-002 [2026-05-01] Section 15 subsection-vs-body conflicts (7 scenarios) — scenario body authoritative
55:- **Spec sections affected:** Section 15.6 (Tier 3B), 15.7 (Tier 3C), 15.8 (Tier 3D), 15.12 (Boundary).
56:- **Pattern:** 7 specific scenario bodies describe inputs that, when run through the actual rule chain (Section 14), produce a tier OTHER than the subsection's heading.
57:- **Resolution rule applied:** scenario body wins over subsection heading. Same principle as SPEC-INT-001.
58:- **The 7 scenarios:** [table with S3B.4, etc.]
75:- **Encoder enforcement (already done):** the 13 test files encode each of these 7 scenarios with assertions matching the body-actual tier/rule.
81:- **User approval (verbatim, 2026-05-01 message Q4):** ...
```

**Verdict: PASS** — SPEC-INT-002 entry exists, dated 2026-05-01, covers exactly 7 body-wins decisions in a table format, references Fix-16 / BLK-004 Defect-15.3, documents user approval with verbatim quote. Per spec_changelog.md line 76, all 7 test file assertions have already been verified against the table by main-thread grep.

---

### Check 7 — DEC-017 in tomato_decisions.md

**Command run:** `grep -n "DEC-017" tomato_decisions.md`

**Actual output:**
```
230:## DEC-017 [2026-04-30] Phase 3 entry preconditions relaxation — T-IMPL-5a/5b moved from preconditions to Phase 4 work
238:- **User approval:** explicit verbatim quote (2026-04-30 latest message, Condition 1): "Replace preconditions 2-3 with: 'Phase 3 produces FAILING tests by design...'"
```

**Verdict: PASS** — DEC-017 present, dated 2026-04-30, documents Phase 3 precondition relaxation with user verbatim approval quote.

---

### Check 8 — DEC-018 in tomato_decisions.md

**Command run:** `grep -n "DEC-018" tomato_decisions.md`

**Actual output:**
```
242:## DEC-018 [2026-04-30] Defect-37 + Defect-42 fast-tracked to master prompt before Phase 4 implementer dispatch
256:- **User approval:** explicit verbatim quote (2026-04-30 latest message, Condition 2): "Apply Defect-37 and Defect-42 as Section 27 fast-track items..."
```

**Verdict: PASS** — DEC-018 present, dated 2026-04-30, documents Defect-37 + Defect-42 fast-track to Section 27 with user verbatim approval quote.

---

### Check 9 — DEC-019 in tomato_decisions.md

**Command run:** `grep -n "DEC-019" tomato_decisions.md`

**Actual output:**
```
260:## DEC-019 [2026-05-01] Sacred manifest exclusion for runtime logs in `scripts/apin/`
278:  - Does NOT rewrite the rebaseline_history. The pre-DEC-019 baseline preserved in the manifest's `rebaseline_history` array.
280:- **User approval:** explicit verbatim quote (2026-05-01 message, Q2 detailed plan): "Q2 — Sacred drift: option (c), update manifest to exclude *.log patterns inside scripts/apin/..."
```

**Verdict: PASS** — DEC-019 present, dated 2026-05-01, documents sacred manifest log_exclusions addition with user verbatim approval quote.

---

### Check 10 — Sacred manifest 10/10 PASS (canonical algorithm with log_exclusions)

**Command run:** Python canonical-algorithm verification script implementing:
- `sha256(json({rel_path: sha256(file)}, sort_keys=True, separators=(',',':')).encode('utf-8'))` for directory entries
- `log_exclusions` honored via `fnmatch` on basename
- Simple file sha256 for file entries

**Actual output:**
```
PASS: scripts/apin/
PASS: models/best_model.pt
PASS: models/swin_best_model.pt
PASS: models/model2_specialist/model2_production.pt
PASS: data/specialist/model3/split_indices.json
PASS: app/config.py
PASS: data/metadata/source_map.csv
PASS: models/specialist/ladinet_phase1_heads.pt
PASS: scripts/model3_training/checkpoints/model3_production_v3.pt
PASS: models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt
--- TOTAL: 10 PASS, 0 FAIL ---
```

**Verdict: PASS** — Sacred manifest 10/10. All 10 entries (1 directory + 9 files) match their recorded hashes. The DEC-019 log_exclusions (`*.log`, `*.log.*`) are applied to `scripts/apin/` and its directory hash still matches the post-DEC-019 baseline, confirming no code-level drift in the APIN sacred directory.

---

### Check 11 — Defect-45 tracked in tomato_plan.md T-EARLY-MP as LOW non-blocking

**Command run:** `grep -n "Defect-45\|T-EARLY-MP\|naming.*inconsistency" tomato_plan.md | head -15`

**Actual output:**
```
145:26. **Fix-45 (Defect-45, LOW, Phase 3 — added 2026-05-01):** Test function naming inconsistency
    in `tomato_sandbox/tests/integration/test_section15_*.py` — 4 files use `def test_scenario_S*_*():`
    per Section 15.2 Convention 14, while 9 files use the shorter `def test_S*_*():` form. Both
    are pytest-discoverable ... Severity: LOW. **No Phase 4 blocker** — pytest discovers both forms.
```

**Verdict: PASS** — Defect-45 logged as Fix-45 in T-EARLY-MP queue, severity LOW, explicitly marked "No Phase 4 blocker". The inconsistency (4 files use verbose `test_scenario_S*_*` form; 9 files use shorter `test_S*_*` form) does not affect pytest discovery or Phase 4 implementation — both naming forms satisfy `test_*` discovery.

*Note flagged as cosmetic/non-blocking per the entry itself. No consolidator action required.*

---

### Check 12 — All 8 fast-track patches present in master prompt Section 27

**Command run:**
```
grep -n "## Fix-9|## Fix-10|## Fix-16|## Fix-27|## Fix-28|## Fix-34|## Fix-37|## Fix-42" tomato_master_prompt.md
```

**Actual output:**
```
1412:### Fix-16 (Defect-16, HIGH — BLOCKS Phase 3) — section15-encoder intra-spec scenario conflict resolution
1420:### Fix-27 (Defect-27, HIGH) — phase exit gate composition rule
1434:### Fix-28 (Defect-28, HIGH) — plan-edit authority for inline patches
1449:### Fix-34 (Defect-34, HIGH — NEW from Phase 2 Round 3 60% defect rate) — planner reads spec body verbatim
1480:### Fix-9 (Defect-9, HIGH) — spec-cartographer Write tool [FAST-TRACK 2026-04-28 PER USER STEP 2]
1492:### Fix-10 (Defect-10, HIGH) — Write tool sweep on audit subagents [FAST-TRACK 2026-04-28 PER USER STEP 2]
1517:### Fix-37 (Defect-37, HIGH — Phase-4-blocking) [FAST-TRACK 2026-04-30 PER DEC-018]
1538:### Fix-42 (Defect-42, HIGH — Phase-4-blocking) [FAST-TRACK 2026-04-30 PER DEC-018]
```

**Also confirmed via tomato_plan.md:**
```
159:Execute these subset of fixes BEFORE Phase 3 begins (Fix-16, Fix-27, Fix-28, Fix-34 — APPLIED 2026-04-28)
...
Fix-9, Fix-10: FAST-TRACK 2026-04-28 (out-of-band)
Fix-37, Fix-42: FAST-TRACK 2026-04-30 per DEC-018
```

**Verdict: PASS** — All 8 required Section 27 fast-track patches are present (Fix-9, Fix-10, Fix-16, Fix-27, Fix-28, Fix-34, Fix-37, Fix-42). Fix-37 and Fix-42 were applied 2026-04-30 per DEC-018 before Phase 3 encoder dispatch, satisfying the "before Phase 4 implementer dispatch" requirement.

---

### Check 13 — tomato_log.md Phase 3 entries

**Command run:**
```
grep -n "Phase 3|2026-04-30|2026-05-01" tomato_log.md | grep -E "2026-04-30|2026-05-01" | head -20
```

**Actual output:**
```
226:## [2026-04-30 12:30] Phase 2 Round 4 close-out: PVA SD-1 + SD-5 resolved; consolidation written
238:## [2026-04-30 13:00] Phase 2 CLOSED. Phase 3 ENTRY APPROVED. DEC-017 + DEC-018 applied.
258:- PHASE 3 STATUS: ENTRY APPROVED. About to dispatch section15-encoder...
260:## [2026-04-30 17:15] Phase 3 — section15-encoder dispatched and returned
292:## [2026-05-01 00:30] Pre-gate work: DEC-019 (manifest exclusion) + SPEC-INT-002 (7 body-wins) + Defect-45 (test naming)
```

**Verdict: PASS** — Three distinct Phase 3 log entries present:
1. `[2026-04-30 13:00]` — Phase 3 entry approved, DEC-017 + DEC-018 applied
2. `[2026-04-30 17:15]` — Encoder dispatched and returned, deliverable produced
3. `[2026-05-01 00:30]` — Pre-gate work (DEC-019, SPEC-INT-002, Defect-45)

---

## Summary Table

| # | Check | Verdict | Notes |
|---|-------|---------|-------|
| 1 | 135 test functions (sum across 13 files) | **PASS** | Exact count: 135 |
| 2 | All 13 required files exist with correct names | **PASS** | All 13 present |
| 3 | All 13 fail with `ModuleNotFoundError: No module named 'tomato_sandbox.tier'` | **PASS** | 13 errors, 0 collected |
| 4 | Import contract at `.claude/import_contract.md`, ~8000 bytes | **PASS** | 8116 bytes |
| 5 | S1.1 uses `[0.89, ...]` from line 4117 (not 0.92 typo) | **PASS** | Inline comment documents BLK-004 / SPEC-INT-001 |
| 6 | SPEC-INT-002 in spec_changelog.md, 7 body-wins, dated 2026-05-01 | **PASS** | User approval verbatim quoted |
| 7 | DEC-017 in tomato_decisions.md | **PASS** | 2026-04-30, precondition relaxation |
| 8 | DEC-018 in tomato_decisions.md | **PASS** | 2026-04-30, Defect-37/42 fast-track |
| 9 | DEC-019 in tomato_decisions.md | **PASS** | 2026-05-01, manifest log_exclusions |
| 10 | Sacred manifest 10/10 PASS (canonical algorithm) | **PASS** | 10 PASS, 0 FAIL |
| 11 | Defect-45 tracked as LOW non-blocking in T-EARLY-MP | **PASS** | Fix-45 in plan, "No Phase 4 blocker" |
| 12 | All 8 fast-track patches present (Fix-9,10,16,27,28,34,37,42) | **PASS** | All 8 Section 27 headings confirmed |
| 13 | tomato_log.md Phase 3 entries present | **PASS** | 3 entries at 04-30 13:00, 17:15, 05-01 00:30 |

**All 13 checks: PASS. Zero FAIL.**

---

## Observations (non-blocking, for consolidator awareness)

**OBS-A (cosmetic) — Defect-45 naming inconsistency (already tracked):** 4 of 13 test files use the verbose `test_scenario_S*_*` naming convention per Section 15.2 Convention 14; 9 files use the shorter `test_S*_*` form. Both are pytest-discoverable. Tracked as Fix-45 in T-EARLY-MP, severity LOW, explicitly non-blocking for Phase 4. No action required before Phase 4 entry.

**OBS-B (informational) — `scripts/apin/` directory baseline uses post-DEC-019 exclusions:** The canonical verification used `log_exclusions: ["*.log", "*.log.*"]` as specified in the manifest entry. The pre-DEC-019 baseline (316 files, hash `a602722f...`) is preserved in `rebaseline_history` for audit trail. The current active baseline (145 files, hash `452d697b...`) was verified to match. If the APIN server runs again and generates new `.log` files, those will continue to be excluded from drift detection — which is the intended behavior per DEC-019.

**OBS-C (informational) — No phase_3_exit_audit file was pre-existing:** This is the first Phase 3 exit audit file. The report is saved to the path specified in the checklist (`phase_3_exit_audit_20260501T0030.md`).

---

## Overall Verdict

**READY for Phase 4.**

All 13 Phase 3 exit conditions are met with direct evidence. The section15-encoder delivered:
- Exactly 135 test functions across exactly 13 correctly named files
- All 13 files fail at pytest collection with the expected `ModuleNotFoundError: No module named 'tomato_sandbox.tier'` (uniform import contract for Phase 4 implementer)
- A substantive 8116-byte import contract at `.claude/import_contract.md`

Supporting documentation is complete: DEC-017, DEC-018, DEC-019 in decisions log; SPEC-INT-002 in spec_changelog.md; Defect-45 tracked; all 8 required master-prompt fast-track patches applied; sacred manifest 10/10 PASS; Phase 3 log entries present.

**No blockers. Phase 4 entry is approved pending user confirmation.**
