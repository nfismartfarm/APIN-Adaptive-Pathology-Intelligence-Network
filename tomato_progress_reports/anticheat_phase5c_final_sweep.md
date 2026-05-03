# Anti-Cheat Final Sweep — Phase 5c

**Inspector:** anti-cheat-inspector (Sonnet 4.6, isolated read-only context)
**Date:** 2026-05-03
**Scope:** Holistic cheat-pattern detection across `tomato_sandbox/` codebase
**Saved by:** main-thread scribe per DEC-011 (auditor lacks Write tool)

## Verdict: **PASS clean** (0 HIGH, 2 MEDIUM, 3 LOW — all documentation hygiene / process anomalies; zero engineering dishonesty)

## Summary table

| Check | Severity threshold | Findings | Verdict |
|---|---|---|---|
| 1. Section 15 LF-SHA256 vs DEC-032 baseline | HIGH | 0 | PASS |
| 2. Pre-commit hook md5 | HIGH | 0 | PASS |
| 3. Suppressed failures | HIGH | 0 HIGH, 0 MEDIUM, 3 LOW | PASS |
| 4. Hardcoded test values | HIGH | 0 suspect literals | PASS |
| 5. Fabricated dataclass fields | HIGH | 0 ungrounded fields | PASS |
| 6. Fake completion claims | HIGH | 0 fabricated DECs | PASS |
| 7. `print()` in production | HIGH | 0 print() calls | PASS |
| 8. DEC ledger consistency | MEDIUM | 2 anomalies (DEC-016 gap; parallel-dispatch ordering) | WARN |
| 9. BLK ledger consistency | MEDIUM | 1 anomaly (BLK-006/007/008 OPEN no Phase 5 disposition note) | WARN |
| 10. Spec-citation density | MEDIUM | 0 modules below threshold | PASS |
| 11. Test count accounting | MEDIUM | 0 drift (1150 = 986+135+29 confirmed) | PASS |
| 12. Server boot sanity | INFO | All 4 endpoints HTTP 200 | PASS |

## Check 1 — Section 15 LF-SHA256 vs DEC-032 baseline

All 13 integration test files hashed via `tr -d '\r' < FILE | sha256sum`. Every hash matches the DEC-032 baseline recorded at `tomato_log.md` lines 506-518:

| File | Hash (first 12) | DEC-032 baseline | Match |
|---|---|---|---|
| test_section15_boundary.py | `0cfdae923b18` | `0cfdae923b18` | ✓ |
| test_section15_disagreement.py | `78b8f8c83c9a` | `78b8f8c83c9a` | ✓ |
| test_section15_tier1.py | `7dd63be0e127` | `7dd63be0e127` | ✓ |
| test_section15_tier2.py | `fc5eada27ee0` | `fc5eada27ee0` | ✓ |
| test_section15_tier3a.py | `b15f71dc7a2c` | `b15f71dc7a2c` | ✓ |
| test_section15_tier3b.py | `bc413eaff721` | `bc413eaff721` | ✓ |
| test_section15_tier3c.py | `67ab89bd3421` | `67ab89bd3421` | ✓ |
| test_section15_tier3d.py | `814db57ca6b8` | `814db57ca6b8` | ✓ |
| test_section15_tier4a.py | `5193b0a7c711` | `5193b0a7c711` | ✓ |
| test_section15_tier4b.py | `6792aae32ec8` | `6792aae32ec8` | ✓ |
| test_section15_tier5.py | `dac139192370` | `dac139192370` | ✓ |
| test_section15_tta.py | `202ee630458d` | `202ee630458d` | ✓ |
| test_section15_underpowered.py | `36c0595cd872` | `36c0595cd872` | ✓ |

Git provenance: all 13 files have ONE commit (`a926d3d` — DEC-032 git-tracking backfill). No phase 5 commit touched any Section 15 file. **No violation.**

## Check 2 — Pre-commit hook md5

`md5sum .git/hooks/pre-commit` → `24eb46f308751df3a125faca0680c9c7`. Matches sacred baseline. **No violation.**

## Check 3 — Suppressed failures

Patterns swept across `tomato_sandbox/`: `pytest.skip`, `pytest.xfail`, `@pytest.mark.skip`, `@pytest.mark.skipif`, bare `except:`, `except Exception: pass`, `# noqa`, `# type: ignore`.

**Test files:**
- `test_tta.py` lines 247/333/371/426: `@pytest.mark.skipif(not _PIL_AVAILABLE/...)` — LOW (env-dependent guards for optional deps; condition is the rationale)
- `test_validate_input.py:506`: conditional `pytest.skip` for JPEG-codec environment edge case — LOW (legitimate env guard)

**Production files (each had main-thread re-read of context):**
- `config.py:47`: `except Exception: return {}` for YAML parse fallback — LOW (defensive default; behavior clear from context; minor: lacks inline rationale comment)
- `pipeline.py:625`: `except Exception: lora_tensor_batched = lora_tensor` — LOW (torch-unsqueeze graceful fallback; outer handler at :632 logs structurally)
- `pipeline.py:675`: `except Exception: pass  # Best-effort release` — LOW (GPU-lock cleanup with inline rationale)
- `psv/psv.py:248`: `except Exception: h, w = 1, 1` — LOW (degraded-mode shape extract; outer handler logs)

**Assessment:** zero unconditional skips, zero unjustified bare excepts in failure paths, all suppressions are cleanup/fallback paths with documented rationale or obvious context. **No HIGH or MEDIUM findings.**

## Check 4 — Hardcoded test values

Spot-checked 5 test files:

| Test file | Numeric/string literal | Backing |
|---|---|---|
| `test_conformal.py` | `CONFORMAL_ALPHA == 0.10` | spec 13.2:3538 |
| `test_conformal.py` | `CONFORMAL_N_CALIBRATION == 40` | spec 13.3:3557 |
| `test_conformal.py` | `tau == 0.6234` | fixture round-trip echo |
| `test_severity.py` | `mild_max == 5.0` | spec 17.3:5974 |
| `test_severity.py` | `disease_coverage_pct == 12.0` | fixture echo |

All literals either spec-cited, fixture round-trip, or boundary-trivial. **No violation.**

## Check 5 — Fabricated dataclass fields

Cross-checked 11 production dataclasses against spec citations:

- `TierAssignment` (3 fields per DEC-041 + import_contract.md, spec S14.7) — clean
- `SignalAResult/B/C` — spec S8.6, S9.6, S10.6 citations — clean
- `ClassifierResult` (9 fields per DEC-039, spec S12.10) — clean
- `ConformalResult` — spec S13 — clean
- `SeverityResult` (incl. `grade_per_class` per S17.5 + DEC-050) — clean
- `ValidatedImage`, `IQAResult`, `PipelineContext` — all spec-grounded — clean

**No fabricated fields. No violation.**

## Check 6 — Fake completion claims

DEC-001..050 spot-checked. Recent DECs (045-050) reference modules that exist on disk (verified via `ls`). `phase_f0_calibration/conformal_tau.json` referenced in DEC-045 exists with documented placeholder `tau=0.42`. `tomato_sandbox/models/` contains only `.gitkeep` — consistent with DEC-003 + DEC-047 (β interpretation: no real weights pre-F.0). **No fabricated references. No violation.**

## Check 7 — `print()` in production

Grep across `tomato_sandbox/` non-test files: zero executable `print()` calls. Only comment-line spec quotes (`# spec: 26.7 — never print()`). **No violation.**

## Check 8 — DEC ledger consistency (MEDIUM)

`grep -c "^## DEC-[0-9]" tomato_decisions.md` = **49 real headed entries** (DEC-001..050 minus DEC-016 = 49).

**Anomalies:**
- DEC-016 referenced inline in DEC-012 body but never headlined (decision content documented but no top-level entry)
- DEC-025/026 appear out of file order (DEC-026 at line 366, DEC-025 at line 386); DEC-028 has explicit `[RENUMBERED]` annotation explaining the parallel-subagent collision
- DEC-042/043/044 inverted (DEC-044 at line 976, DEC-043 at line 1055, DEC-042 at line 1134) — Phase 4 Batch 6 parallel-dispatch artifact (DEC-038 was introduced precisely to prevent this going forward)

**Assessment:** process anomalies, not cheat indicators. Renumbering is documented; ordering inversions are known parallel-dispatch side effects. **MEDIUM WARN — not a violation.**

## Check 9 — BLK ledger consistency (MEDIUM)

`grep -c "^## BLK-" tomato_blockers.md` = **16** (1 template + 15 real BLK-001..015).

| BLK | Status |
|---|---|
| BLK-001 | RESOLVED |
| BLK-002..005 | RESOLVED via DEC-012 |
| BLK-006/007/008 | **OPEN** (planning-phase non-blocking; option A applied in Phase 4 implementation; no Phase 5 disposition note added) |
| BLK-009 | PATCHED |
| BLK-010..011 | RESOLVED/PATCHED |
| BLK-012 | NON-BLOCKING (filed for agronomic review) |
| BLK-013 | RESOLVED via DEC-048 (Phase 5a) |
| BLK-014 | RESOLVED via DEC-049 (Phase 5b) |
| BLK-015 | RESOLVED via DEC-050 (Phase 5b) |

**Recommendation:** add a Phase 5 disposition note to BLK-006/007/008 confirming option A was applied during Phase 4 (T-IMPL-3b/3c/4b implementations followed spec body per Fix-42). **MEDIUM WARN — ledger maintenance, not a violation.**

## Check 10 — Spec-citation density

| Module | LOC | `# spec:` citations | Density per 100 LOC |
|---|---|---|---|
| `orchestrator/pipeline.py` | 1207 | 71 | 5.9 |
| `tier/tier_assignment.py` | 616 | 37 | 6.0 |
| `classifier/hierarchical_classifier.py` | 542 | 57 | 10.5 |
| `conformal/conformal.py` | 349 | 45 | 12.9 |
| `severity/grader.py` | 519 | 34 | 6.6 |
| `response/response_builder.py` | 756 | 113 | 14.9 |

All 6 above the 5-per-100-LOC threshold. Major branch decisions (Rule 1-8 in tier_assignment, severity thresholds, conformal formula) all have inline citations. **No violation.**

## Check 11 — Test count accounting

Live `pytest --collect-only -q`:
- Total: **1150** tests
- Unit: 986
- Integration: 135
- E2e: 29
- Sum: 986+135+29 = **1150** ✓

Phase 5b's 1150 grand-total reconciliation holds; no further accounting drift. **No violation.**

## Check 12 — Server boot sanity

Server already running on `127.0.0.1:8767` (PID from earlier session). All 4 endpoints HTTP 200:
- `/health` → 200 (`{"status":"ok","model_loaded":true,"gpu_available":true}`)
- `/info` → 200 (full service config JSON; `conformal_tau: 0.42` placeholder per DEC-045)
- `/ready` → 200 (`{"ready":true}`)
- `/metrics` → 200 (Prometheus text)

**Sanity PASS.**

## Carry-forward LOW concerns (informational)

1. `config.py:47` outer YAML-fallback `except Exception` lacks inline rationale (behavior clear from context). LOW.
2. DEC-016 ghost number — body-referenced only; decision documented in DEC-012 body. LOW.
3. BLK-006/007/008 OPEN status without Phase 5 disposition note. Implementations correct per spec read; recommend ledger-only update at T-EARLY-MP cycle. LOW.

## Recommendation

**CLOSE Phase 5c.** Zero HIGH findings. Two MEDIUM anomalies are documentation hygiene (DEC ordering artifacts from parallel-dispatch; BLK-006/007/008 disposition notes absent), not engineering dishonesty. Phase 6 (F.0 prep per spec Section 29) is authorized.
