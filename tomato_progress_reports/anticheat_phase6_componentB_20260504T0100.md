# Anti-Cheat Scan — Phase 6 Component B (T-PHASE6-B / DEC-052)

**Inspector:** anti-cheat-inspector (Sonnet 4.6)
**Date:** 2026-05-04
**Saved by:** main-thread scribe per DEC-011

## Verdict: **PASS clean. ALL 15 CHECKS CLEAR. 0 HIGH, 0 MEDIUM, 0 LOW.**

This is the cleanest dispatch in the project. No findings of any severity.

## Files inspected

- `tomato_sandbox/validation/__init__.py` (941 B)
- `tomato_sandbox/validation/fit_calibration.py` (36,625 B)
- `tomato_sandbox/tests/unit/test_fit_calibration.py` (36,607 B)

## Summary table

| Check | Severity threshold | Outcome |
|---|---|---|
| 1. Section 15 immutability (LF-SHA256) | HIGH | CLEAR — 13/13 hashes match DEC-032 baseline; live 135/135 PASS |
| 2. Pre-commit hook md5 `24eb46f3...` | HIGH | CLEAR — unchanged |
| 3. Suppressed failures | HIGH | CLEAR — 1 conditional skip (PipelineContext.make_degraded helper absent in pre-F.0); rationale documented inline; consistent with DEC-052 Decision 6; no unconditional skips, xfails, or bare excepts |
| 4. Spec citations | MEDIUM | CLEAR — all 4 required citations present (S13.5 conformal, S12.8 Platt, S17.3 severity defaults, S4.5+S8.4 chilli_leakage) |
| 5. No `print()` in production | LOW | CLEAR — zero |
| 6. No APIN imports | HIGH | CLEAR — only csv, json, math, datetime, pathlib, typing, numpy, tomato_sandbox.utils.logging, tomato_sandbox.conformal.conformal |
| 7. DEC-038 — no commits since `ffaddb2` | HIGH | CLEAR — `git log ffaddb2..HEAD` empty |
| 8. Honest test count | MEDIUM | CLEAR — 48 collected (47 pass + 1 conditional skip) |
| 9. **SEVERITY_DEFAULTS values vs S17.3 table** | **HIGH** | **CLEAR** — all 5 disease pairs match spec exactly: foliar(5,15), septoria(8,25), late_blight(2,8), ylcv(10,30), mosaic(15,40) |
| 10. **Conformal τ formula verbatim** | **HIGH** | **CLEAR** — `q = np.ceil((N+1)*(1.0-alpha))/N` with `method="higher"` quantile per S13.5; test asserts q=0.925 for N=40, α=0.10 (= 37/40 verified) |
| 11. labeled_data_path is parameter | MEDIUM | CLEAR — `run_full_calibration(labeled_data_path: Path, ...)`; zero hardcoded data paths; only `_DEFAULT_OUTPUT_DIR` is hardcoded (output, overridable) |
| 12. Tests use tmp_path isolation | MEDIUM | CLEAR — zero writes to `tomato_sandbox/phase_f0_calibration/` from tests; all use `tmp_path` fixture |
| 13. Platt fallback chain honest | HIGH | CLEAR — scipy L-BFGS-B → sklearn LogisticRegression → identity (1.0, 0.0); identity fallback logs WARNING with explicit message; degenerate-label check also logs WARNING |
| 14. Severity defaults fallback documented | MEDIUM | CLEAR — `_MIN_SEVERITY_SAMPLES: int = 10` named constant (not magic literal); output dict includes `default_used: True/False`; logged at INFO/WARNING with disease + counts |
| 15. **No premature lifting of pre-F.0 model-loading deferral** | **HIGH** | **CLEAR** — zero `torch`, `torch.load`, `torch.nn` imports; sacred checkpoint paths absent; orchestrator import deferred to function body (avoids circular at test time); DEC-052 Decision 8 explicit: "consumes pipeline outputs, does NOT load checkpoints" |

## Highlights

### Conformal τ formula verbatim
The formula in `tomato_sandbox/conformal/conformal.py` (which `fit_calibration.fit_conformal_tau` delegates to per DEC-052 Decision 2) is:
```python
q = np.ceil((N + 1) * (1.0 - alpha)) / N
q = min(q, 1.0)
tau = float(np.quantile(nonconformity_scores, q, method="higher"))
```
Spec S13.5 verbatim. Not the off-by-one variant `ceil(N*(1-alpha))/N`. Not the unclamped variant. `method="higher"` cited explicitly to S13.5:3596-3600.

Test `test_n40_alpha010_quantile_level` asserts the analytically-derivable value `expected_tau = 37/40 = 0.925`. Passes.

### Platt fallback honesty
Three-tier fallback (scipy → sklearn → identity) with the right ordering. Identity fallback only fires when both scipy and sklearn are unavailable AND logs `WARNING`-level message stating "scipy and sklearn both unavailable; returning identity (a=1, b=0)". Not a silent fake-completion path. Test `test_identity_calibration` confirms the optimizer actually runs (alpha tolerance allows 0.1-5.0 range, proving non-trivial parameter fitting).

### Pre-F.0 deferral respected
`fit_calibration.py` imports zero `torch`-related symbols. Sacred checkpoint paths (`model3_production_v3.pt`, `sp_lora_epoch13_f10.9113_PRESERVED.pt`) appear nowhere in the file. The script consumes pipeline outputs (numpy arrays, dataclasses) — Component C territory (real model loading) is properly untouched.

### Conditional skip is honest
Single test self-skips when `PipelineContext.make_degraded()` is unavailable. This helper does not yet exist in `tomato_sandbox/orchestrator/pipeline.py` (verified by main-thread grep). The skip is gated on `try/except` for the helper's presence, not unconditional. DEC-052 Decision 6 documents the dependency. When Component C lifts the deferral and adds the helper (or when degraded-mode contexts can be constructed), this test will run automatically.

## Recommendation

Component B is clean. Phase 6 Component A (validation script, DEC-053) can dispatch on user approval. Sacred verify, Section 15 regression, and 1197-test pass count all preserved.
