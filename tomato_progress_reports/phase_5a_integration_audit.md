# Phase 5a — Integration Layer Audit & Wiring Fix

**Date:** 2026-05-02 / 2026-05-03 (across the dispatch and main-thread verification)
**Dispatch:** T-AUDIT-5a, single `implementer` subagent (per DEC-047 Ambiguity 1 resolution)
**Scope:** Phase 5 entry prerequisite — real-subprocess + real-image + un-mocked compute paths through the sandbox pipeline
**Verdict:** **CLOSE.** Single bug surfaced (BLK-013, pre-identified). Fix applied. All gates green. Phase 5b is unblocked.

---

## 1. Headline result

**Phase 5a CLOSED with one bug fixed (BLK-013, pre-identified at Batch 7 close).** No sibling integration bugs surfaced. The orchestrator's PIL-handling at `preprocess_for_v3/lora/psv` call sites was already correct (those functions take raw PIL by design, unlike `compute_iqa`).

POST `/predict` with a real chilli anthracnose leaf image returns spec-compliant S16.2 response with `tier.label = "4B"`, `rule_id_fired = "1"` (signal failure → 4B, expected pre-F.0 path since model weights aren't loaded), and no top-level `error` field. 521ms response time. **CLOSE criterion satisfied.**

## 2. The bug — BLK-013 (RESOLVED)

**Location:** `tomato_sandbox/orchestrator/pipeline.py:527`

**Symptom:** Every real-image POST short-circuited at IQA gate with HTTP 200 + body `{"error": "IQA_REJECTED", "status": 422}`.

**Root cause:** Orchestrator passed raw `PIL.Image` to `compute_iqa(...)`. The `compute_iqa` function expects an object with a `.pil_image` attribute (per its docstring at `tomato_sandbox/iqa/iqa.py:357`). The `AttributeError` was caught internally by IQA's try/except and produced `IQAResult(decision="REJECT", aggregate_score=0.0, ...)` — masking the wiring bug as a quality rejection.

**Fix:** Inner `_PILAdapter` class at the call site wraps the raw PIL.Image with `.pil_image` attribute before passing to `compute_iqa`. 3-line mechanical change at `pipeline.py:528-537`.

```python
class _PILAdapter:
    """Minimal ValidatedImage-shaped wrapper for compute_iqa.
    spec: section 6.6 line 1374 — compute_iqa expects an object with .pil_image attr.
    BLK-013 / DEC-048 — orchestrator was passing raw PIL.Image; now wraps before call.
    """
    def __init__(self, pil): self.pil_image = pil
iqa_result = compute_iqa(_PILAdapter(pil_image))
```

**Logged:** DEC-048 in `tomato_decisions.md`. BLK-013 status updated to `RESOLVED 2026-05-03 by T-AUDIT-5a (DEC-048)` in `tomato_blockers.md`.

## 3. Sibling integration bugs surfaced

**Zero.** BLK-013 was the sole integration bug.

### Why no siblings (BLK-013 speculation refuted by evidence)

The BLK-013 entry speculated that `preprocess_for_v3(pil_image)`, `preprocess_for_lora(pil_image)`, and `compute_signal_c(rgb_cc, ...)` might have similar PIL-vs-ValidatedImage contract mismatches. The implementer verified each on disk:

| Call site | Expects | What orchestrator passes | Verdict |
|---|---|---|---|
| `compute_iqa(validated_image)` | `.pil_image` attr | raw PIL → **AttributeError** (BLK-013) | BUG (now fixed) |
| `preprocess_for_v3(pil_image)` | raw `PIL.Image` | raw PIL | OK |
| `preprocess_for_lora(pil_image)` | raw `PIL.Image` | raw PIL | OK |
| `preprocess_for_psv(pil_image)` | raw `PIL.Image` | raw PIL | OK |
| `compute_signal_a(model, tensor)` | tensor from preprocess | tensor from preprocess | OK |
| `compute_signal_b(model, tensor)` | tensor from preprocess | tensor from preprocess | OK |
| `compute_signal_c(rgb_cc, mask, score)` | numpy + mask + scalar | matches | OK |
| `compute_classifier(sa, sb, sc, ...)` | typed dataclasses | matches | OK |
| `compute_conformal_set(p_final_calibrated, ...)` | numpy from `ClassifierResult.p_final_calibrated` | matches (DEC-040 spec-pinned name) | OK |
| `assign_tier(...)` | 7 keyword dicts | matches (DEC-041 import contract) | OK |
| `build_response(...)` | `TierAssignment + ClassifierResult + IQAResult + ConformalResult` | matches (DEC-043) | OK |

The IQA call site was an outlier because `compute_iqa` was intentionally written to consume a `ValidatedImage` (T-IMPL-2a output) — that's the spec contract per S6.6:1374. The orchestrator predates that decision and didn't adapt.

## 4. Smoke test evidence (5 real images)

All 5 returned spec-compliant responses; 4 reached `tier.label="4B"` (expected pre-F.0 degraded path); 1 was legitimately rejected by IQA `wetness` dimension.

| # | Image | Source | Disease | HTTP | Tier | IQA | Verdict |
|---|---|---|---|---|---|---|---|
| 1 (impl) | sourceA_IMG-20231104-WA0108.jpg (347KB) | data/specialist/model3/cleaned/chilli_anthracnose/ | anthracnose | 200 | 4B | HIGH | PASS — CLOSE |
| 2 (impl) | sourceA_IMG-20231104-WA0114.jpg (369KB) | data/specialist/model3/cleaned/chilli_anthracnose/ | anthracnose | 200 | 4B | HIGH | PASS |
| 3 (impl) | orig_chilli_bangladesh_2025_Healthy Leaf00001 (254KB) | data/specialist/model3/cleaned/chilli_healthy/ | healthy | 200 | 4B | HIGH | PASS |
| 4 (impl) | orig_chilli_bangladesh_2025_Healthy Leaf00004 (311KB) | data/specialist/model3/cleaned/chilli_healthy/ | healthy | 200 | 4B | HIGH | PASS |
| 5 (impl) | orig_chilli_bangladesh_2025_Cercospora... (249KB) | data/specialist/model3/cleaned/chilli_cercospora/ | cercospora | 200 | n/a | wet | LEGITIMATE IQA_REJECT |
| 6 (main verify) | sourceA_IMG-20231104-WA0108.jpg (re-test) | data/specialist/model3/cleaned/chilli_anthracnose/ | anthracnose | 200, 521ms | 4B | HIGH | PASS — CLOSE confirmed |

The cercospora image hitting IQA wetness is meaningful evidence that **IQA is now functional** — it correctly distinguishes images. Pre-fix, every image returned IQA_REJECTED regardless of quality; post-fix, only images with actual quality problems are rejected.

### Sample S16.2 response (image 1, main-thread re-test)

```json
{
  "request_id": "0844a861-c6db-48be-9c7d-96d2d73c98f0",
  "image_hash": "14d658226346744317726fc0793130f9b37f27a009e652a39a649f9529772955",
  "timestamp_iso": "2026-05-02T20:04:26.161891Z",
  "tier": {
    "label": "4B",
    "human_readable": "Pipeline issue — please retake or contact support",
    "alert_level": "error"
  },
  "prediction": {
    "primary_class": "foliar",
    "primary_confidence": 0.0,
    "prediction_set": ["foliar", "septoria", "late_blight", "ylcv", "mosaic", "healthy", "OOD"],
    "prediction_set_human": ["Foliar leaf spot", "Septoria leaf spot", "Late blight",
                             "Yellow Leaf Curl Virus (YLCV)", "Mosaic virus", "Healthy",
                             "Out-of-distribution (OOD)"]
  },
  "tier5_alert": { "fired": false, "reason": null, ... },
  "severity": { "grade": null, "human_readable": "Severity could not be reliably graded.", ... },
  "explanation": {
    "user_strings": [...],
    "structured": {
      "rule_id_fired": "1",
      "sub_rule_id_fired": "1",
      "tier_main_conditions": { "max_prob_actual": 0.0, "margin_actual": 0.0, "iqa_decision": "HIGH", "set_size": 7 },
      "tier5_evaluation": { ... }
    }
  }
}
```

`rule_id_fired="1"` is Rule 1 firing on `signal_failure → Tier 4B`, which is the **expected** pre-F.0 path: signals fail because models aren't loaded → degraded mode → all-signals-failed sentinel → tier_assignment Rule 1 → Tier 4B. This is the (β) interpretation per DEC-047 working as designed.

## 5. Stability verification (main-thread independent)

| Gate | Result |
|---|---|
| Sacred (in-sandbox `verify_manifest()`) | **10/10 PASS** |
| Section 15 integration regression | **135/135 PASS** in 0.30s |
| Unit tests (venv Python) | **961 PASS** |
| Grand total under venv | **1096 PASS** (961 unit + 135 integration) |
| DEC-038 compliance (no implementer commits since `fac25ef`) | **VERIFIED** — `git log fac25ef..HEAD` empty |
| Pre-allocation rule | DEC-048 used as pre-allocated; no collision |

## 6. Files touched by T-AUDIT-5a

| File | Change | Rationale |
|---|---|---|
| `tomato_sandbox/orchestrator/pipeline.py` | +10 lines (lines 526-537) — `_PILAdapter` inner class + use at IQA call site | BLK-013 fix per DEC-048 |
| `tomato_decisions.md` | +DEC-048 entry | Decision logging |
| `tomato_blockers.md` | BLK-013 status: IDENTIFIED → RESOLVED | BLK-013 closure |

**Sacred files: zero touched.** No edits to `scripts/apin/`, `models/`, `data/specialist/`, `data/metadata/source_map.csv`, `app/config.py`, or `.git/hooks/pre-commit`.

**Section 15 tests: zero touched.** All 13 files unchanged from DEC-032 baseline.

## 7. Architectural lessons (Phase 5a confirms M2 once more)

The single bug (BLK-013) had been **identified at Batch 7 close** but **deferred to Phase 5a per Option B**. Phase 5a's value:

1. **Confirmed M2 was correctly applied:** un-mocked compute paths surfaced exactly the bug that the mocked TestClient e2e tests had hidden. No additional surprises beyond the one already identified.
2. **Refuted the BLK-013 sibling-bug speculation:** the orchestrator's other call sites (preprocess functions, signal compute, classifier, conformal, tier_assignment, response_builder) were all wired correctly. Spec-pinned dataclass field names (DEC-040 / DEC-041 / DEC-043) prevented downstream contract drift.
3. **Validated the (β) interpretation per DEC-047:** Tier 4B-from-degraded-mode IS a spec-compliant S16.2 response. The pipeline produces the full schema; the prediction is degraded but the response shape is real. Phase F.0 territory begins where real-weight predictions begin.

## 8. Phase 5b readiness

**Phase 5b prerequisites are now satisfied:**

- Real-subprocess sandbox server boots cleanly on 8767 under venv Python ✓
- Real-image POST `/predict` end-to-end returns spec-compliant S16.2 response ✓
- No top-level `error` for valid leaf images; legitimate IQA rejection works for quality-failed images ✓
- Tier 4B-degraded acceptable per DEC-047 ✓
- Integration layer wiring audit complete; no integration bugs remain ✓

Phase 5b can dispatch the spec-auditor's two-pass contract audit (Pass 1 spec-only; Pass 2 cross-references decisions) immediately. Anti-cheat final sweep + sacred final verification follow.

## 9. Q4 sandbox lift status

**Q4 was held until BLK-013 closes in Phase 5.** BLK-013 is now RESOLVED. The sandbox server on 8767 is genuinely operational under venv Python — boots cleanly, all 7 endpoints respond, real-image POST produces spec-compliant responses.

**Recommendation:** Q4 lift can proceed once user reviews this checkpoint. The sandbox server is no longer "held"; it's a working pre-F.0 sandbox awaiting model weights for full predictions.

## 10. Cumulative state after Phase 5a close

| Metric | Pre-Phase-5a | Post-Phase-5a | Δ |
|---|---|---|---|
| Unit tests under venv | 961 | **961** | 0 (BLK-013 fix is a wiring change inside try-block; doesn't add tests) |
| Section 15 | 135/135 | **135/135 PRESERVED** | 0 |
| Grand total under venv | 1096 | **1096** | 0 |
| DECs logged | DEC-001..047 | **DEC-001..048** | +1 (DEC-048) |
| BLKs filed | 13 (1 deferred) | **13** (0 deferred — BLK-013 RESOLVED) | 0 file count, +1 RESOLVED |
| Master-prompt defects | 60 | 60 | 0 |
| Sacred entries | 10/10 PASS | **10/10 PASS** | 0 |
| Real-image smoke test | failing (BLK-013) | **passing** | RESOLVED |

## 11. Recommendations for main thread

1. **Commit** the 3 files (`pipeline.py`, `tomato_decisions.md`, `tomato_blockers.md`) plus this `phase_5a_integration_audit.md` and `tomato_log.md` (if appended) as the Phase 5a close commit per DEC-038. Pre-commit hook will pass cleanly (no Section 15 staged).
2. **BLK-013 status:** RESOLVED — no further deferral. BLK ledger now has 0 deferred blockers.
3. **Q4 sandbox lift:** can proceed.
4. **Phase 5b prerequisite check:** all satisfied. spec-auditor dispatch authorized when user approves.
5. **Phase F.0 entry condition:** Phase 5b complete + user approval. Real-weight loading happens in Phase F.0 / Phase 6.

## 12. Process discipline observed in Phase 5a

| Rule | Status |
|---|---|
| DEC-038 (no implementer git ops) | **HONORED** — `git log fac25ef..HEAD` empty before main-thread commit |
| Pre-allocation (DEC-048 specifically reserved) | **HONORED** — implementer used exactly DEC-048; no collision |
| Rule 6 (Section 15 immutable) | **HONORED** — all 13 files unchanged |
| Sacred manifest (10/10 PASS) | **HONORED** — no sacred file touched |
| Fix-42 (read spec body) | **HONORED** — implementer cited `iqa.py:357` docstring, not paraphrase |
| Defect-60 (venv Python authoritative) | **HONORED** — all test runs under venv |
| Stopping rules (5-bug cap, non-mechanical fix) | **NOT TRIGGERED** — single mechanical fix; CLOSE achieved |

## 13. Concluding statement

**Phase 5a closes with no integration bugs remaining in the un-mocked compute path.** The architectural finding M2 from Batch 7 produced exactly one identifiable bug (BLK-013), and Phase 5a's audit-style dispatch caught it cleanly. The five-bug cap was not approached. The integration layer is now verified end-to-end under the (β) interpretation, with Tier 4B from degraded-mode firing as the expected pre-F.0 path per DEC-047.

Phase 5b can proceed immediately on user approval. Spec-auditor's contract audit dispatches against a verified-wiring foundation.

---

*Generated 2026-05-02/2026-05-03 by main-thread scribe; consolidates 1 implementer subagent dispatch (T-AUDIT-5a, DEC-048) + main-thread independent verification (sacred 10/10 PASS, Section 15 135/135, 1096 venv tests, real-image smoke test confirming CLOSE) + DEC-038 compliance check (no implementer commits). All claims independently verified by direct disk read + grep + pytest run + curl smoke test + JSON shape inspection.*
