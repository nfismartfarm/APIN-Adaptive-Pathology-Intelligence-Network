# Phase 0 — Setup Report

**Date:** 2026-04-27
**Phase:** 0 (setup; no implementation code)
**Status:** complete; awaiting user approval to enter Phase 1

---

## Phase 0 task checklist (master prompt section 4 Phase 0)

| # | Task | Status |
|---|---|---|
| 1 | Save master prompt verbatim to `tomato_master_prompt.md` | done |
| 2 | Catalogue existing repository state | done; reuses earlier `DISK_SPACE_REPORT.md` plus `git status` (uncommitted-outside-sandbox flagged below) |
| 3 | Verify environment (Python, PyTorch, CUDA) | done; matches spec exactly |
| 4 | Build sacred manifest at `.claude/sacred_manifest.json` | done; 10 entries, 0 problems, sourced from spec Section 2.6 with disk-corrected paths |
| 5 | Create `.claude/` directory structure | done |
| 6 | Write CLAUDE.md (per DEC-007: dual-write) | done; new file at `tomato_sandbox/claude_tomato_system.md`; appended activity-log section to existing root `CLAUDE.md` |
| 7 | Create 8 subagent files | done |
| 8 | Create 5 slash command files | done |
| 9 | Pre-create 3 skill files (per DEC-008: empty placeholders, populated in Phase 1) | done |
| 10 | Write `.claude/settings.local.json` (merged with existing 5 user-approved entries) | done |
| 11 | Initialize log files (`tomato_plan.md`, `tomato_log.md`, `tomato_decisions.md`, `tomato_blockers.md`, `spec_changelog.md`, `tomato_progress_reports/`) | done |
| 12 | Initialize `tomato_sandbox/scratch/.gitkeep` and `tomato_sandbox/models/.gitkeep` | done |
| 13 | Run pip installs (`pre-commit`, `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-mock`) | done |

---

## Environment

```
Python    3.13.11
PyTorch   2.11.0+cu130
CUDA      13.0
GPU       NVIDIA GeForce RTX 4060 Laptop GPU
Active    miniconda3 base (NOT a venv; see DEC-009)
```

All four match spec Section 28.2 hardware target. The "active environment is not a venv" detail is documented in DEC-009 with a queued spec-Phase-4 task to introduce a dedicated `tomato_sandbox/.venv/` once heavyweight deps appear.

---

## Sacred manifest (`.claude/sacred_manifest.json`)

10 entries hash-tracked. All re-hashed at Phase 0 close: **OVERALL PASS** (zero drift).

| Path | Type | Size | Source |
|---|---|---:|---|
| `scripts/apin/` | dir, 316 files | n/a | spec 2.6 |
| `models/best_model.pt` | file | 84.2 MB | spec 2.6 |
| `models/swin_best_model.pt` | file | 114.9 MB | spec 2.6 |
| `models/model2_specialist/model2_production.pt` | file | 198.0 MB | spec 2.6 (PATH CORRECTED from spec table's "model2_production.pt" at root) |
| `data/specialist/model3/split_indices.json` | file | 6.4 MB | spec 2.6 |
| `app/config.py` | file | 43 KB | spec 2.6 |
| `data/metadata/source_map.csv` | file | 3.3 MB | spec 2.6 |
| `models/specialist/ladinet_phase1_heads.pt` | file | 25.7 MB | spec 2.6 (PATH CORRECTED from spec table's `ladinet_checkpoints/` subpath) |
| `scripts/model3_training/checkpoints/model3_production_v3.pt` | file | 87.7 MB | spec 8.7 (added beyond Section 2.6 table per Sandbox Directive) |
| `models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt` | file | 349.8 MB | spec 9.1 (added beyond Section 2.6 table) |

The Sandbox Directive ("entire repository outside `tomato_sandbox/` is sacred") is the broader rule; the manifest table is the most-important hash-tracked subset.

---

## Decisions logged this phase

| ID | Title | Source |
|---|---|---|
| DEC-001 | Sacred manifest from spec Section 2.6 (not master-prompt section 2) | Q-NEW-1 |
| DEC-002 | v3 weights loaded read-only; not copied | Q-NEW-2, spec 8.7 |
| DEC-003 | LoRA copy is spec Phase A.3 task | Q-NEW-3, spec 9.7 |
| DEC-004 | PSV reimplemented; sacred reference not copied | Q-NEW-4, spec 10 |
| DEC-005 | LadiNet sacred at corrected path regardless of v1 use | Q-NEW-5, spec 2.6 |
| DEC-006 | Phase 0 reading scope confirmed; deeper reading deferred to Phase 1 | Q-NEW-6 |
| DEC-007 | Dual CLAUDE.md write (root append-only + sandbox tomato-specific) | session-start user instruction |
| DEC-008 | Skills creation deferred from Phase 0 to Phase 1 post-summaries | Q-A1 option (d) |
| DEC-009 | APIN venv shared for dev tools; sandbox venv deferred to Phase 4 | Q-A2 option (c) |

Full text of each in `tomato_decisions.md`.

---

## Existing-repo cataloguing observations (master prompt Phase 0 step 2)

**Files at root that are project artifacts (preserved unchanged):**
- `CLAUDE.md` (244 KB, okra+brassica project memory; activity log appended below the existing content per DEC-007)
- `README.md` (2.7 KB)
- `tomato_3_signal_system.md` (573 KB, the locked spec)
- `psv_calibration.json`, `psv_convo.md`, `decisions.md`, `architecture_claude_decisions.md`, and many other prior-session artifacts
- `DISK_SPACE_REPORT.md` (kept per user instruction; ignored as historical reference)

**No `pyproject.toml` exists at root.** The implementer in Phase 4 will create one inside `tomato_sandbox/` when needed; an at-root pyproject.toml is not present to extend.

**Existing `.gitignore` (962 bytes) preserved.** Phase 0 did not extend it. The implementer will add `tomato_sandbox/scratch/`, `tomato_progress_reports/`, etc. via append when relevant (no need to do this in Phase 0).

**`git status` shows uncommitted changes outside `tomato_sandbox/`** (which did not exist before Phase 0). Modified: `agents/download_*.py` (3 files), `agents/download_orchestrator.py`, `agents/kaggle_utils.py`, `architecture_claude_decisions.md`, `context.md`, `decisions.md`, `scripts/train_model3_simple.py`, `setup/install_cuda.py`, `setup/package_deployment.py`, `setup/test_server.py`, `tools/add_kerala_image.py`. Deleted: 5 download scripts (`download_faruk_okra.py`, `download_ghose_cabbage.py`, `download_iubat_okra.py`, `download_kareem_cabbage.py`, `download_misrak_vegetables.py`). Many untracked items including chilli/tomato dataset folders and several `*.md` files. **None of these are Phase 0 outputs; they are your prior workstream.** Master prompt Phase 0 step 2 asks me to surface this so you know the working tree is non-clean. Tomato sandbox setup did not touch any of these files.

---

## Tooling installed

```
pre-commit    4.6.0
pytest        9.0.2
pytest-cov    7.1.0
pytest-xdist  3.8.0
pytest-mock   3.15.1
```

Installed into `miniconda3` base (DEC-009). Verified each via `pip show`.

---

## Observations to flag

1. **APIN server on port 8766 is no longer responding.** The launch I did earlier this session served `/predict/full` requests successfully (log shows 200 OK responses around `21:12:30`); afterward the process went idle and is now unreachable. Most likely cause: the Bash background-task runner's lifecycle limit terminated the process. The pip installs in this Phase 0 cannot have caused this (they ran into the same conda environment, but installed-but-not-imported packages do not affect a running Python process's module state). Per your earlier statement that the server can be stopped if necessary, this is not a blocker. If you want APIN running again, re-launch with `python scripts/apin_v2/apin_server.py --port 8766`.

2. **Master prompt section 2 manifest is stale.** The master prompt's illustrative manifest disagrees with spec Section 2.6 on three paths (`model2_production.pt` location, `ladinet_phase1_heads.pt` location, presence of `data/specialist/model3/okra_brassica/`). Per your procedural note in the Q-A answers, a master-prompt update is queued for after Phase 0 closes. This update goes through master-prompt update flow (Section 19), not `spec_changelog.md`. Recorded in `tomato_plan.md` as `T-EARLY-MP`. **Action required:** decide whether to do this update before Phase 1 or defer to a later checkpoint.

3. **Skills are placeholders.** `.claude/skills/tomato-section15-format.md`, `.claude/skills/tomato-conformal.md`, `.claude/skills/tomato-gpu-lock.md` contain only "PLACEHOLDER. Populated in Phase 1" stubs. Per DEC-008, substantive content is authored after `spec-cartographer` produces summaries for spec Sections 13, 15, 20 in Phase 1. Recorded in `tomato_plan.md` as `T-EARLY-A`.

4. **Pre-commit hook NOT installed yet.** The Section 15 protection hook (master prompt Section 5 Rule A) is installed at the END of Phase 3 (after the 135 tests are encoded), per master prompt Phase 3 step 6. Phase 0 only ensures `pre-commit` the package is installed. The actual git pre-commit hook script is written in Phase 3.

---

## Files created or modified by Phase 0

```
NEW:
  tomato_master_prompt.md                                  (master prompt verbatim)
  CLAUDE.md                                                (APPENDED activity-log section; existing 244 KB content unchanged)
  tomato_decisions.md                                      (DEC-001 through DEC-009)
  tomato_log.md                                            (Phase 0 entry)
  tomato_plan.md                                           (placeholder + 4 forward tasks)
  tomato_blockers.md                                       (zero open blockers)
  spec_changelog.md                                        (zero spec changes)
  tomato_sandbox/claude_tomato_system.md                   (tomato project memory)
  tomato_sandbox/scratch/.gitkeep
  tomato_sandbox/models/.gitkeep
  tomato_progress_reports/.gitkeep
  tomato_progress_reports/phase_0_setup.md                 (this file)
  .claude/sacred_manifest.json                             (10 entries, all PASS)
  .claude/spec_summaries/.gitkeep
  .claude/agents/spec-cartographer.md
  .claude/agents/planner.md
  .claude/agents/section15-encoder.md
  .claude/agents/implementer.md
  .claude/agents/sacred-guardian.md
  .claude/agents/spec-auditor.md
  .claude/agents/anti-cheat-inspector.md
  .claude/agents/progress-reporter.md
  .claude/commands/tomato-status.md
  .claude/commands/tomato-audit.md
  .claude/commands/tomato-checkpoint.md
  .claude/commands/tomato-verify-sacred.md
  .claude/commands/tomato-section.md
  .claude/skills/tomato-section15-format.md                (placeholder per DEC-008)
  .claude/skills/tomato-conformal.md                       (placeholder per DEC-008)
  .claude/skills/tomato-gpu-lock.md                        (placeholder per DEC-008)

MODIFIED (extension only, not overwrite):
  .claude/settings.local.json                              (added master-prompt baseline; preserved existing 5 user-approved Bash entries)
  CLAUDE.md                                                (appended activity-log section after existing 244 KB)

NOT TOUCHED:
  Anything else outside tomato_sandbox/ (per Sandbox Directive)
```

---

## Phase 1 prerequisites

Before Phase 1 can begin, the user must explicitly approve. After approval, Phase 1 will:
1. `spec-cartographer` reads spec in 6 batches, produces summaries to `.claude/spec_summaries/section_NN.md`.
2. Spot-check 3 random summaries against the original spec text.
3. Build dependency graph at `.claude/spec_dependency_graph.md`.
4. Identify ambiguities; write to `tomato_blockers.md`.
5. **Skills authoring** (T-EARLY-A): populate the 3 placeholder skill files from the relevant batch summaries (per DEC-008).
6. Produce comprehension report at `tomato_progress_reports/phase_1_comprehension.md`.
7. STOP and report.

---

## STOP

Phase 0 is complete. Awaiting user approval to enter Phase 1. No further work without explicit "approve" / "proceed" / "continue with Phase 1" signal per master prompt section 17 approval signals.

If you want me to address `T-EARLY-MP` (master-prompt section 2 update) or relaunch APIN before Phase 1, say so. Otherwise on your "proceed" I will begin Phase 1 with `spec-cartographer` Batch 1 (Sections 1-4 foundations).
