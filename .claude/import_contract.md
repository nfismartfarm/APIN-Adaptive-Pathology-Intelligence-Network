# Import Contract (set in Phase 3, honored in Phase 4)

This file is the authoritative reference for all import paths used by the
135 integration tests in `tomato_sandbox/tests/integration/test_section15_*.py`.
Phase 4 implementers MUST create modules at exactly these paths. Changing
paths requires updating this file AND all 135 test files simultaneously.

---

## Primary import

```python
from tomato_sandbox.tier.tier_assignment import assign_tier
```

Every test file imports `assign_tier` at module level (line 1 of imports).
This causes `ImportError` or `ModuleNotFoundError` until Phase 4 creates
`tomato_sandbox/tier/tier_assignment.py`.

---

## assign_tier signature

```python
def assign_tier(
    *,
    v3_signal: dict,
    lora_signal: dict,
    psv_signal: dict,
    classifier: dict,
    conformal: dict,
    iqa: dict,
    underpowered_classes: set[int] | None = None,
) -> TierAssignment:
    ...
```

All parameters are keyword-only (enforced by `*`).
`underpowered_classes` defaults to `None` (equivalent to empty set).

---

## Return type: TierAssignment

Phase 4 must provide a return value with these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `tier_label` | `str` | One of: `"1"`, `"2"`, `"3A"`, `"3B"`, `"3C"`, `"3D"`, `"4A"`, `"4B"` |
| `tier5_alert` | `bool` | True when T5 alert fires independently of base tier |
| `rule_id_fired` | `str` | Shorthand identifier of the rule that set the tier (see table below) |

### TierAssignment can be:
- A dataclass: `@dataclass class TierAssignment: tier_label: str; tier5_alert: bool; rule_id_fired: str`
- A namedtuple
- Any object with `.tier_label`, `.tier5_alert`, `.rule_id_fired` attributes

---

## rule_id_fired shorthand semantics

| Value | Meaning |
|-------|---------|
| `"1"` | Rule 1: at least one signal's `forward_succeeded` is False → Tier 4B |
| `"3"` | Rule 3: `psv_reliability < 0.40` OR `chilli_leakage > 0.40` → Tier 3C |
| `"4"` | Rule 4: `combined_max_prob < 0.45` → Tier 4A |
| `"5"` | Rule 5: `prediction_set_size >= 3` → Tier 3B, OR `size == 0` (empty set) → Tier 4A |
| `"6"` | Rule 6: `prediction_set_size == 2` → Tier 3A |
| `"7a"` | Rule 7, sub-rule a: definitive YLCV criteria met AND IQA == DEGRADED → Tier 3D |
| `"7b"` | Rule 7, sub-rule b: definitive criteria met AND argmax class is underpowered → Tier 3A |
| `"7c"` | Rule 7, sub-rule c: definitive criteria met, no downgrade → Tier 1 |
| `"8a"` | Rule 8, sub-rule a: confident criteria met AND IQA == DEGRADED → Tier 3D |
| `"8b"` | Rule 8, sub-rule b: confident criteria met AND argmax class is underpowered → Tier 3A |
| `"8c"` | Rule 8, sub-rule c: confident criteria met, no downgrade → Tier 2 |
| `"catch_all_low_confidence"` | Rule 9: no prior rule matched → Tier 4A |

Sub-rule precedence within Rule 7: 7a > 7b > 7c.
Sub-rule precedence within Rule 8: 8a > 8b > 8c.
Overall rule priority: **Rule 1 > Rule 4 > Rule 3 > Rule 5 > Rule 6 > Rule 7 > Rule 8 > Rule 9.**

**[CORRECTED 2026-05-02 per BLK-011 sub-defect 11.1 + DEC-041 Decision 1 + Batch 5 anti-cheat LOW-1]:** the priority above shows Rule 4 BEFORE Rule 3. Earlier text (and spec Section 14.5 header) listed `Rule 1 > Rule 3 > Rule 4 > ...`, but the spec scenario walk for SB.10 (spec lines 5208-5217) treats Rule 4 as evaluated before Rule 3 (with psv_reliability=0.30, max=0.143 → Tier 4A from Rule 4, not Tier 3C from Rule 3). Per BLK-004 precedent (scenario body authoritative over header text), implementation evaluates Rule 4 first. Section 15 tests 135/135 PASS confirms.

---

## Input dict schemas

### v3_signal / lora_signal

```python
{
    "probs": list[float],          # length 6: [foliar, septoria, late_blight, ylcv, mosaic, healthy]
    "chilli_leak": float,          # v3 only: chilli leakage probability
    "forward_succeeded": bool,     # False if the model forward pass failed
}
```

Note: `chilli_leak` on `lora_signal` is always `0.0` in spec scenarios; the field
may be required or optional. Tests always pass it on v3; lora tests omit it or use 0.0.

### psv_signal

```python
{
    "argmax": int,                 # class index of PSV argmax
    "max": float,                  # probability of argmax class
    "margin": float,               # max - second_max
    "reliability": float,          # PSV reliability score (0.0 – 1.0)
    "forward_succeeded": bool,     # False if PSV forward pass failed
}
```

### classifier

```python
{
    "argmax": int,                 # class index of combined classifier argmax
    "max": float,                  # combined_max_prob
    "margin": float,               # combined_margin
}
```

### conformal

```python
{
    "set": set[int],               # prediction set (class indices)
    "size": int,                   # len(set)
    "tau": float | None,           # conformal threshold tau (informational)
}
```

### iqa

```python
{
    "decision": str,               # "ACCEPTABLE", "DEGRADED", or "HIGH"
    # Additional fields (e.g. aggregate_score) may be present and must be ignored
}
```

---

## underpowered_classes parameter convention

- Type: `set[int] | None`
- Default: `None` (treated as empty set; no underpowered guard fires)
- Values: integer class indices. `{3}` = YLCV underpowered. `{4}` = mosaic underpowered.
- The underpowered guard only activates within sub-rules 7b and 8b; it has no effect
  when Rules 1–6 or Rule 9 fire first.

---

## T5 alert logic

T5 alert is evaluated independently of the base tier. It fires when:

1. `classifier["argmax"] in {2, 3, 4}` (late_blight=2, ylcv=3, mosaic=4) AND
   `classifier["max"] >= 0.20` (inclusive)

OR

2. `2 in conformal["set"]` (late_blight in prediction set) AND
   `v3_probs[2] >= 0.20` OR `lora_probs[2] >= 0.20` OR `classifier["max"] >= 0.20`
   — the exact in-set probability used is the classifier's probability for late_blight.
   Per spec: late_blight in set AND late_blight_prob >= 0.20 where late_blight_prob
   is the classifier's calibrated probability for class 2.

Note: mosaic (4) and YLCV (3) have argmax-only T5 triggers. Only late_blight (2)
has an in-set T5 trigger. See spec Section 14.3.

T5 threshold `>= 0.20` is INCLUSIVE.

---

## Class index reference

| Index | Class |
|-------|-------|
| 0 | foliar |
| 1 | septoria |
| 2 | late_blight |
| 3 | ylcv |
| 4 | mosaic |
| 5 | healthy |
| 6 | OOD |

---

## Threshold reference (all taken verbatim from spec)

| Threshold | Value | Comparison | Rule |
|-----------|-------|------------|------|
| Rule 3 psv_reliability | 0.40 | strict `<` | Rule 3 fires if reliability < 0.40 |
| Rule 3 chilli_leakage | 0.40 | strict `>` | Rule 3 fires if chilli > 0.40 |
| Rule 4 max | 0.45 | strict `<` | Rule 4 fires if max < 0.45 |
| Rule 7 max | 0.85 | inclusive `>=` | Rule 7 requires max >= 0.85 |
| Rule 7 margin | 0.30 | inclusive `>=` | Rule 7 requires margin >= 0.30 |
| Rule 7 psv_reliability | 0.50 | inclusive `>=` | Rule 7 requires reliability >= 0.50 |
| Rule 7 chilli_leakage | 0.20 | strict `<` | Rule 7 requires chilli < 0.20 |
| Rule 8 max | 0.65 | inclusive `>=` | Rule 8 requires max >= 0.65 |
| Rule 8 margin | 0.20 | inclusive `>=` | Rule 8 requires margin >= 0.20 |
| Rule 8 psv_reliability | 0.40 | inclusive `>=` | Rule 8 requires reliability >= 0.40 |
| Rule 8 chilli_leakage | 0.30 | strict `<` | Rule 8 requires chilli < 0.30 |
| T5 max | 0.20 | inclusive `>=` | T5 fires if argmax max >= 0.20 |
| T5 in-set prob | 0.20 | inclusive `>=` | T5 in-set rule fires if late_blight prob >= 0.20 |

---

## File → scenario mapping

| Test file | Scenarios | Count |
|-----------|-----------|-------|
| `test_section15_tier1.py` | S1.1 – S1.12 | 12 |
| `test_section15_tier2.py` | S2.1 – S2.12 | 12 |
| `test_section15_tier3a.py` | S3A.1 – S3A.12 | 12 |
| `test_section15_tier3b.py` | S3B.1 – S3B.10 | 10 |
| `test_section15_tier3c.py` | S3C.1 – S3C.12 | 12 |
| `test_section15_tier3d.py` | S3D.1 – S3D.10 | 10 |
| `test_section15_tier4a.py` | S4A.1 – S4A.12, S4A.5b | 13 |
| `test_section15_tier4b.py` | S4B.1 – S4B.10 | 10 |
| `test_section15_tier5.py` | S5.1 – S5.11 | 11 |
| `test_section15_boundary.py` | SB.1 – SB.15 | 15 |
| `test_section15_underpowered.py` | SUP.1 – SUP.7 | 7 |
| `test_section15_disagreement.py` | SDIS.1 – SDIS.6 | 6 |
| `test_section15_tta.py` | STTA.1 – STTA.5 | 5 |
| **Total** | | **135** |

---

## BLK-004 Defect-15.3 note

Multiple scenarios appear under subsection headers implying one tier but the
scenario body specifies a different tier. The scenario body is ALWAYS authoritative.
Affected scenarios: S3C.8, S3C.9, S3C.12, S3D.5, S3D.7, S3B.4.

Reported in `tomato_blockers.md` as BLK-004 Defect-15.3.
