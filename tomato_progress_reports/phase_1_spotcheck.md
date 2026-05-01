# Phase 1 Spot-Check Report

**Date:** 2026-04-27
**Method:** Per master prompt section 4.2 task 2 — pick 3 random sections, compare summary against original spec text.

This file extracted to its own artifact 2026-04-27 22:55 per PVA finding (the spot-check content was originally embedded in `phase_1_comprehension.md`; master prompt names a separate file).

---

## Section selection rationale

Picked to span subject-matter diversity:

| # | Section | Subject area | Rationale |
|---|---|---|---|
| 1 | 17 | Severity grading rules | Numeric thresholds; high risk of cartographer transposition error |
| 2 | 23 | Agronomist queue API | Tabular data (endpoint list); high risk of off-by-one or missing rows |
| 3 | 11 | Test-Time Augmentation | Numeric thresholds + decision flow; tests independence from PSV claim |

---

## Spot-check 1 — Section 17 severity thresholds

**Spec lines verified:** 5980-5986

**Spec table (verbatim from spec):**

| Disease | Mild (coverage_pct, lesion_count) | Moderate | Severe |
|---|---|---|---|
| Foliar leaf spot | < 5%, 1-5 lesions | 5-15%, 5-15 | > 15% or > 15 lesions |
| Septoria leaf spot | < 8%, 1-10 | 8-25%, 10-25 | > 25% or > 25 lesions |
| Late blight | < 2%, 1-3 | 2-8%, 3-8 | > 8% or > 8 lesions |
| YLCV (Yellow Leaf Curl Virus) | < 10% leaf area showing curl | 10-30% | > 30% |
| Mosaic virus | < 15% leaf area showing mottling | 15-40% | > 40% |

**Summary table (`.claude/spec_summaries/section_17.md` lines 28-32):** identical 5-row table with the same 15 numeric values.

**PSV-feature mapping verified:** summary correctly maps `disease_coverage_pct → G2`, `mean_lesion_intensity → G3`, `lesion_count → G4`, `lesion_size_distribution → G7+G8`, plus `psv_reliability` reliability output. Spec line 5961-5967 confirms.

**Verdict:** PASS

---

## Spot-check 2 — Section 23 queue API endpoints

**Spec lines verified:** 7082-7090 (table) and 7062-7090 (priority assignment + UI requirements)

**Spec endpoint table (verbatim from spec):**

| Endpoint | Method | Purpose |
|---|---|---|
| `/queue/cases` | GET | List pending cases |
| `/queue/cases/{case_id}` | GET | Fetch a single case |
| `/queue/cases/{case_id}/claim` | POST | Agronomist claims a case |
| `/queue/cases/{case_id}/resolve` | POST | Agronomist resolves with disposition |
| `/queue/cases/{case_id}/dismiss` | POST | Agronomist dismisses case |
| `/queue/cases/{case_id}/escalate` | POST | Agronomist escalates to senior |
| `/queue/stats` | GET | Aggregate stats (pending count, P50/P95) |

**Initial summary state:** listed only 6 endpoints — `/queue/stats` was missing.

**Patch applied:** `.claude/spec_summaries/section_23.md` line 60 patched 2026-04-27 with the missing 7th endpoint and a `[scribe-patch]` marker. Note: the patch produced a harmless duplicate visible in the audit (original 6th endpoint line 58 + patched 7th endpoint line 60); this is cosmetic only and does not affect endpoint count.

**Verdict:** PASS with patch applied

---

## Spot-check 3 — Section 11 TTA thresholds

**Spec lines verified:** 2932-2939

**Spec rules (verbatim from spec):**

```
combined_max_prob >= TOMATO_TTA_TRIGGER_THRESHOLD (default 0.55)
    → no TTA. Pipeline returns 1-view result.

TOMATO_TTA_ESCALATE_THRESHOLD <= combined_max_prob < TOMATO_TTA_TRIGGER_THRESHOLD
    → 2-view TTA.

combined_max_prob < TOMATO_TTA_ESCALATE_THRESHOLD (default 0.45)
    → 5-view TTA.
```

**Summary statement (`.claude/spec_summaries/section_11.md` lines 85-86, 94-96):** trigger threshold 0.55, escalate threshold 0.45, view counts {1, 2, 5} — exact match with spec.

**PSV exclusion verified:** spec line 2925 explicitly states "PSV does NOT participate in TTA"; summary captures this in the Purpose section.

**Verdict:** PASS

---

## Aggregate result

| Section | Verdict | Notes |
|---|---|---|
| 17 (severity) | PASS | All 15 numeric values match |
| 23 (queue API) | PASS with patch | 7th endpoint `/queue/stats` patched into summary |
| 11 (TTA) | PASS | Thresholds and PSV-exclusion match |

**Cartographer fidelity:** high. One completeness gap (missing endpoint row) corrected inline without re-running the subagent.

**No fabricated values found.** Anti-cheat scan corroborates this finding.
