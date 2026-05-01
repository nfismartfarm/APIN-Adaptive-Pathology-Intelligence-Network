# Tomato 3-Signal System — Project Memory (sandbox-scoped)

## Identity

This is the v1 implementation of the tomato disease detection sandbox per `tomato_3_signal_system.md` (8756-line spec, locked). All implementation lives in `tomato_sandbox/` only. Sacred files outside this directory must not be touched (read OK; write/edit/delete forbidden).

Hierarchical CLAUDE.md: this file is the tomato-specific project memory. Root `CLAUDE.md` carries the older okra+brassica context unchanged plus an append-only activity log of tomato sandbox events. See DEC-007 in `tomato_decisions.md`.

## Sacred files (NEVER modify)

See `.claude/sacred_manifest.json`. The `sacred-guardian` subagent verifies hashes after every change. Reads are explicitly allowed (loading model weights via `torch.load` is fine; the protection is against modification). Sacred manifest is sourced from spec Section 2.6 with paths corrected against disk reality. See DEC-001.

## Specification

- Source of truth: `tomato_3_signal_system.md` at project root (NOT inside the sandbox; the spec is shared reference material).
- Section summaries: `.claude/spec_summaries/section_NN.md` (populated in Phase 1 by `spec-cartographer`).
- Spec dependency graph: `.claude/spec_dependency_graph.md` (Phase 1).
- Use `spec-cartographer` subagent to query the spec; cite section numbers in every implementation decision.
- Spec changes (rare) tracked in `spec_changelog.md`.

## Three signals (canonical)

- **Signal A — v3 / Model 3** (10-class, tomato + chilli). Lives at `scripts/model3_training/checkpoints/model3_production_v3.pt` (sacred, OUTSIDE sandbox; loaded read-only; not copied). Spec Section 8.
- **Signal B — Single-pass LoRA, epoch 13**. Originally at `models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt`; copied/renamed to `tomato_sandbox/models/tomato_sp_lora_production.pt` in spec Phase A.3. Spec Section 9.
- **Signal C — PSV** (Plant Symptom Visual). Reimplemented in `tomato_sandbox/signals/psv/`. Reference in sacred `scripts/apin/section2d/3a/3c_psv_*.py` (read-only, not copied). Spec Section 10.

## Section 15 tests are immutable

135 deterministic scenarios encoded in `tomato_sandbox/tests/integration/test_section15_*.py` (Phase 3). Pre-commit hook installed at end of Phase 3 blocks modifications. If a test fails after implementation, the implementation is wrong (not the test). Import contract at `.claude/import_contract.md`.

## Workflow phases

0 setup -> 1 comprehension -> 2 planning -> 3 test infrastructure -> 4 implementation -> 5 audit -> 6 F.0 prep. Each phase ends with STOP and user approval. See master prompt section 4 for phase entry/exit criteria.

## Logs

- `tomato_plan.md` - task checklist (populated in Phase 2)
- `tomato_log.md` - chronological work log (append after meaningful units)
- `tomato_decisions.md` - architectural decisions and master-prompt deviations
- `tomato_blockers.md` - open questions blocking work
- `spec_changelog.md` - spec modifications (require user approval; rare)
- `tomato_progress_reports/` - timestamped phase reports

## Subagents

- `spec-cartographer` - reads spec, produces summaries (read-only)
- `planner` - builds task lists (read-only)
- `section15-encoder` - encodes Section 15 scenarios as tests
- `implementer` - writes implementation code in `tomato_sandbox/` only
- `sacred-guardian` - verifies sacred files unchanged (read-only)
- `spec-auditor` - independent code review against spec (isolated context, read-only, two passes)
- `anti-cheat-inspector` - scans for cheating patterns (read-only)
- `progress-reporter` - generates honest status (read-only)

## Slash commands

- `/tomato-status` - generate progress report
- `/tomato-audit` - run spec-auditor + anti-cheat-inspector
- `/tomato-checkpoint` - update logs and stop for user
- `/tomato-verify-sacred` - run sacred-guardian
- `/tomato-section <N>` - load summary for a specific spec section

## Skills (Phase 1 will populate)

- `.claude/skills/tomato-section15-format.md` - schema of Section 15 scenarios
- `.claude/skills/tomato-conformal.md` - conformal prediction patterns (spec Section 13)
- `.claude/skills/tomato-gpu-lock.md` - GPU lock pattern usage (spec Section 20.6)

Currently placeholders. Per DEC-008, substantive content is authored in Phase 1 after the relevant batch summaries exist.

## Constraints

- All new code in `tomato_sandbox/` only (Sandbox Directive at top of spec).
- Cross-cutting utilities first (logging, gpu_lock, nan_guards, degraded_mode) at `tomato_sandbox/utils/`.
- No v2 features (spec Section 30 lists v2 scope).
- Hardware: RTX 4060 8GB VRAM (spec Section 28.2). Single-host; APIN runs separately on port 8766.
- Sandbox server runs on port 8767.
- Cite spec sections; never invent behavior.
- When uncertain, write to `tomato_blockers.md` and stop.
- Scratch space at `tomato_sandbox/scratch/` for ad-hoc experiments (excluded from protocol; not shipped).

## Communication style

- Simple language; no em-dashes; bullets fine; concise; evidence-based; no marketing adjectives.
- Test claims require pasted output, not summary.
- Honest reporting: complete vs in-progress vs blocked vs not-started; no "going well" language.

## Class taxonomy

23 classes total (defined in sacred `app/config.py`):

- okra (5): yvmv, powdery_mildew, cercospora, enation, healthy
- brassica (5): black_rot, downy_mildew, alternaria, clubroot, healthy
- tomato (9): bacterial_spot, early_blight, late_blight, leaf_mold, septoria_leaf_spot, target_spot, mosaic_virus, yellow_leaf_curl_virus, healthy
- chilli (4): anthracnose, cercospora_leaf_spot, leaf_curl, healthy

This sandbox produces tomato predictions only. Chilli is "not deployed" in v1 (spec Section 1.3, Section 22.2). v3's chilli outputs collapse into `chilli_leakage` per spec Section 8.4. Okra/brassica go to APIN at 8766 unchanged.

## Canonical 7-class output for tomato

Spec Section 2.4 canonical index space: 0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy, 6=OOD. The 9 PlantVillage tomato classes collapse into 6 deployment classes plus an OOD bucket per the canonical-index map.

## When compacting context

Preserve: list of completed tasks, current module, sacred manifest hash, current open blockers, latest test results, current phase.

## Subsequent sessions

If `tomato_master_prompt.md` exists at project root (it does after Phase 0), this is not the first session. Read prompt section 15 (Resuming after interrupt) and skip Phase 0 setup.
