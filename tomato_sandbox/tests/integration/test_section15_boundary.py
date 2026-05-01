"""
Section 15.12 — Boundary and edge cases (SB.1 – SB.15).
Spec source: tomato_3_signal_system.md lines 5115–5269.

Every threshold used by the rule chain is exercised at its exact boundary value.
Verbatim inputs and expected outputs taken from each scenario body.

BLK-004 Defect-15.3 note: scenario body is authoritative over subsection header.

Import contract: from tomato_sandbox.tier.tier_assignment import assign_tier
These tests FAIL with ImportError until Phase 4 implements the module.
"""

import pytest

# Phase 4 will provide this module. Until then, every test fails with ImportError.
from tomato_sandbox.tier.tier_assignment import assign_tier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(probs, chilli_leak=0.0, succeeded=True):
    return {
        "probs": probs,
        "chilli_leak": chilli_leak,
        "forward_succeeded": succeeded,
    }


def _make_signal_failed():
    return {
        "probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "chilli_leak": 0.0,
        "forward_succeeded": False,
    }


def _make_psv(argmax, max_val, margin, reliability, succeeded=True):
    return {
        "argmax": argmax,
        "max": max_val,
        "margin": margin,
        "reliability": reliability,
        "forward_succeeded": succeeded,
    }


def _make_classifier(argmax, max_val, margin):
    return {"argmax": argmax, "max": max_val, "margin": margin}


def _make_conformal(pred_set, size, tau=None):
    return {"set": pred_set, "size": size, "tau": tau}


# ---------------------------------------------------------------------------
# SB.1 — combined_max_prob = 0.85 exactly (Tier 1; boundary inclusive)
# Spec lines 5119–5127
#
# v3: probs=[0.82, 0.05, 0.04, 0.03, 0.02, 0.01], chilli_leak=0.03
# LoRA: probs=[0.83, 0.06, 0.04, 0.03, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.85, margin=0.55
# Conformal (tau=0.45): set={0}, size=1
# -> Tier 1, T5 alert: False (rule 7c; max 0.85 >= 0.85 boundary inclusive)
# ---------------------------------------------------------------------------

def test_SB_1():
    """SB.1 — max=0.85 exactly satisfies Rule 7's >= 0.85 (inclusive boundary) → Tier 1."""
    v3 = _make_signal([0.82, 0.05, 0.04, 0.03, 0.02, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.83, 0.06, 0.04, 0.03, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.85, margin=0.55)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# SB.2 — combined_max_prob = 0.84999999 (Tier 2; just below Rule 7's threshold)
# Spec lines 5129–5138
#
# v3: probs=[0.82, 0.05, 0.04, 0.03, 0.02, 0.01], chilli_leak=0.03
# LoRA: probs=[0.83, 0.06, 0.04, 0.03, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: argmax=0 (foliar), max=0.84999999, margin=0.55
# Conformal (tau=0.45): set={0}, size=1
# -> Tier 2, T5 alert: False (argmax=0 foliar; late_blight prob 0.04 < 0.20)
# ---------------------------------------------------------------------------

def test_SB_2():
    """SB.2 — max=0.84999999 just below Rule 7's 0.85 threshold; Rule 8 fires → Tier 2."""
    v3 = _make_signal([0.82, 0.05, 0.04, 0.03, 0.02, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.83, 0.06, 0.04, 0.03, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.84999999, margin=0.55)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# SB.3 — combined_margin = 0.30 exactly (Tier 1; boundary inclusive)
# Spec lines 5140–5148
#
# v3: probs=[0.90, 0.05, 0.01, 0.01, 0.00, 0.01], chilli_leak=0.02
# LoRA: probs=[0.90, 0.06, 0.02, 0.01, 0.00, 0.01]
# PSV: argmax=0, max=0.71, margin=0.30, reliability=0.74
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.85, margin=0.30
# Conformal (tau=0.45): set={0}, size=1
# -> Tier 1, T5 alert: False (rule 7c; margin 0.30 >= 0.30 boundary inclusive)
# ---------------------------------------------------------------------------

def test_SB_3():
    """SB.3 — margin=0.30 exactly satisfies Rule 7's >= 0.30 (inclusive boundary) → Tier 1."""
    v3 = _make_signal([0.90, 0.05, 0.01, 0.01, 0.00, 0.01], chilli_leak=0.02)
    lora = _make_signal([0.90, 0.06, 0.02, 0.01, 0.00, 0.01])
    psv = _make_psv(argmax=0, max_val=0.71, margin=0.30, reliability=0.74)
    classifier = _make_classifier(argmax=0, max_val=0.85, margin=0.30)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# SB.4 — combined_margin = 0.29 (just below Rule 7; falls to Rule 8 → Tier 2)
# Spec lines 5150–5158
#
# v3: probs=[0.90, 0.05, 0.01, 0.01, 0.00, 0.01], chilli_leak=0.02
# LoRA: probs=[0.90, 0.06, 0.02, 0.01, 0.00, 0.01]
# PSV: argmax=0, max=0.71, margin=0.30, reliability=0.74
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.85, margin=0.29
# Conformal (tau=0.45): set={0}, size=1
# -> Tier 2, T5 alert: False
# ---------------------------------------------------------------------------

def test_SB_4():
    """SB.4 — margin=0.29 fails Rule 7's >= 0.30; Rule 8 fires (margin >= 0.20) → Tier 2."""
    v3 = _make_signal([0.90, 0.05, 0.01, 0.01, 0.00, 0.01], chilli_leak=0.02)
    lora = _make_signal([0.90, 0.06, 0.02, 0.01, 0.00, 0.01])
    psv = _make_psv(argmax=0, max_val=0.71, margin=0.30, reliability=0.74)
    classifier = _make_classifier(argmax=0, max_val=0.85, margin=0.29)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# SB.5 — psv_reliability = 0.40 exactly (NOT 3C; Rule 3 strict `< 0.40`)
# Spec lines 5160–5168
#
# v3: probs=[0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05
# LoRA: probs=[0.89, 0.04, 0.03, 0.02, 0.01, 0.01]
# PSV: argmax=0, max=0.45, margin=0.08, reliability=0.40 exactly
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.91, margin=0.86
# Conformal (tau=0.40): set={0}, size=1
# -> Tier 2, T5 alert: False
# Walk: Rule 3 `psv_reliability < 0.40` fails at 0.40 (strict). Rule 8 `>= 0.40` passes.
# ---------------------------------------------------------------------------

def test_SB_5():
    """SB.5 — psv_reliability=0.40 exactly; Rule 3's strict < 0.40 fails → Rule 8 fires → Tier 2."""
    v3 = _make_signal([0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05)
    lora = _make_signal([0.89, 0.04, 0.03, 0.02, 0.01, 0.01])
    psv = _make_psv(argmax=0, max_val=0.45, margin=0.08, reliability=0.40)
    classifier = _make_classifier(argmax=0, max_val=0.91, margin=0.86)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# SB.6 — psv_reliability = 0.39 (Rule 3 fires → Tier 3C; just below threshold)
# Spec lines 5170–5178
#
# v3: probs=[0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05
# LoRA: probs=[0.89, 0.04, 0.03, 0.02, 0.01, 0.01]
# PSV: argmax=0, max=0.40, margin=0.05, reliability=0.39
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.91, margin=0.86
# Conformal (tau=0.40): set={0}, size=1
# -> Tier 3C, T5 alert: False (rule 3 fires)
# ---------------------------------------------------------------------------

def test_SB_6():
    """SB.6 — psv_reliability=0.39 satisfies Rule 3's strict < 0.40 → Tier 3C."""
    v3 = _make_signal([0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05)
    lora = _make_signal([0.89, 0.04, 0.03, 0.02, 0.01, 0.01])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.05, reliability=0.39)
    classifier = _make_classifier(argmax=0, max_val=0.91, margin=0.86)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# SB.7 — chilli_leakage = 0.40 exactly (NOT 3C; Rule 3 strict `> 0.40`; falls to Rule 9)
# Spec lines 5180–5188
#
# v3: probs=[0.55, 0.04, 0.01, 0.00, 0.00, 0.00], chilli_leak=0.40 exactly
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.74
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.82, margin=0.71
# Conformal (tau=0.45): set={0}, size=1
# -> Tier 4A, T5 alert: False
# Walk: Rule 3 `chilli_leakage > 0.40` fails at 0.40 (strict). Rule 7 `< 0.20` fails.
#       Rule 8 `< 0.30` fails. Rule 9 catch-all -> Tier 4A.
# ---------------------------------------------------------------------------

def test_SB_7():
    """SB.7 — chilli=0.40 exactly; Rule 3's strict > 0.40 fails; Rules 7/8 chilli caps also fail → Rule 9 → Tier 4A."""
    v3 = _make_signal([0.55, 0.04, 0.01, 0.00, 0.00, 0.00], chilli_leak=0.40)
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.74)
    classifier = _make_classifier(argmax=0, max_val=0.82, margin=0.71)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "catch_all_low_confidence"


# ---------------------------------------------------------------------------
# SB.8 — chilli_leakage = 0.41 (Rule 3 fires → Tier 3C)
# Spec lines 5190–5198
#
# v3: probs=[0.50, 0.04, 0.02, 0.02, 0.00, 0.01], chilli_leak=0.41
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.78, margin=0.65
# Conformal (tau=0.50): set={0}, size=1
# -> Tier 3C, T5 alert: False (rule 3 fires)
# ---------------------------------------------------------------------------

def test_SB_8():
    """SB.8 — chilli=0.41 satisfies Rule 3's strict > 0.40 → Tier 3C."""
    v3 = _make_signal([0.50, 0.04, 0.02, 0.02, 0.00, 0.01], chilli_leak=0.41)
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.78, margin=0.65)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.50)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# SB.9 — combined_max_prob = NaN (orchestrator routes to Tier 4B; all signals marked failed)
# Spec lines 5200–5205
#
# Per spec Section 21 NaN-handling contract: NaN in classifier → orchestrator marks
# all signals failed → assign_tier receives all forward_succeeded=False → Rule 1.
# -> Tier 4B, T5 alert: False (cannot be evaluated reliably with NaN; defaults to False)
# ---------------------------------------------------------------------------

def test_SB_9():
    """SB.9 — NaN in classifier output; orchestrator marks all signals failed → Rule 1 → Tier 4B, T5 False."""
    v3 = _make_signal_failed()
    lora = _make_signal_failed()
    psv = _make_psv(argmax=0, max_val=0.0, margin=0.0, reliability=0.0, succeeded=False)
    classifier = _make_classifier(argmax=0, max_val=float("nan"), margin=float("nan"))
    conformal = _make_conformal(pred_set=set(), size=0)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# SB.10 — All classifier outputs equal (1/7 each, tied argmax)
# Spec lines 5208–5217
#
# v3: probs=[0.143, 0.143, 0.143, 0.143, 0.143, 0.143], chilli_leak=0.142
# LoRA: probs=[0.167, 0.167, 0.167, 0.167, 0.166, 0.166]
# PSV: argmax=0, max=0.143, margin=0.00, reliability=0.30
# IQA: ACCEPTABLE
# Classifier: argmax=0 (numpy first-index when tied), max=0.143, margin=0.000
# Conformal (tau=0.86): set={0,1,2,3,4,5}, size=6
# -> Tier 4A (Rule 4: max 0.143 < 0.45), T5 alert: False
# ---------------------------------------------------------------------------

def test_SB_10():
    """SB.10 — Uniform 1/7 distribution; Rule 4 fires (max 0.143 < 0.45) → Tier 4A, T5 False."""
    v3 = _make_signal([0.143, 0.143, 0.143, 0.143, 0.143, 0.143], chilli_leak=0.142)
    lora = _make_signal([0.167, 0.167, 0.167, 0.167, 0.166, 0.166])
    psv = _make_psv(argmax=0, max_val=0.143, margin=0.00, reliability=0.30)
    classifier = _make_classifier(argmax=0, max_val=0.143, margin=0.000)
    conformal = _make_conformal(pred_set={0, 1, 2, 3, 4, 5}, size=6, tau=0.86)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# SB.11 — prediction_set_size = 0 (empty set; max=0.50 >= 0.45 so Rule 4 fails first)
# Spec lines 5219–5228
#
# v3: probs=[0.50, 0.10, 0.10, 0.10, 0.10, 0.10], chilli_leak=0.00
# LoRA: probs=[0.50, 0.12, 0.10, 0.10, 0.10, 0.08]
# PSV: argmax=0, max=0.45, margin=0.10, reliability=0.60
# IQA: ACCEPTABLE
# Classifier: argmax=0 (foliar), max=0.50, margin=0.40
# Conformal (tau=0.40): threshold 0.60; set={}, size=0
# -> Tier 4A (rule 5 empty-set sub-rule), T5 alert: False
# Note: spec Walk says "max<0.45 -> Rule 4 fires -> Tier 4A" but max=0.50 >= 0.45 so
#       Rule 4 fails; empty set -> Rule 5 sub-rule; rule_id_fired = "5"
# ---------------------------------------------------------------------------

def test_SB_11():
    """SB.11 — Empty prediction set (size=0) and max=0.50; Rule 5 empty-set sub-rule → Tier 4A."""
    v3 = _make_signal([0.50, 0.10, 0.10, 0.10, 0.10, 0.10], chilli_leak=0.00)
    lora = _make_signal([0.50, 0.12, 0.10, 0.10, 0.10, 0.08])
    psv = _make_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.60)
    classifier = _make_classifier(argmax=0, max_val=0.50, margin=0.40)
    conformal = _make_conformal(pred_set=set(), size=0, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# SB.12 — prediction_set_size = 7 (all classes; Rule 4 fires first)
# Spec lines 5231–5236
#
# Classifier: argmax=0 (foliar), max=0.18, margin=0.02
# Rule 4 fires first (max 0.18 < 0.45) -> Tier 4A, before Rule 5.
# -> Tier 4A (Rule 4), T5 alert: False (argmax=0 foliar; late_blight prob 0.15 < 0.20)
# Note: spec does not fully specify v3/LoRA/PSV inputs, so we use minimal valid values
#       that don't trigger Rules 1 or 3 (signals succeeded, psv_reliability >= 0.40,
#       chilli <= 0.40).
# ---------------------------------------------------------------------------

def test_SB_12():
    """SB.12 — All 7 classes in prediction set; Rule 4 fires first (max 0.18 < 0.45) → Tier 4A."""
    v3 = _make_signal([0.18, 0.16, 0.15, 0.14, 0.14, 0.13], chilli_leak=0.10)
    lora = _make_signal([0.18, 0.16, 0.15, 0.14, 0.14, 0.13])
    psv = _make_psv(argmax=0, max_val=0.18, margin=0.02, reliability=0.45)
    classifier = _make_classifier(argmax=0, max_val=0.18, margin=0.02)
    conformal = _make_conformal(pred_set={0, 1, 2, 3, 4, 5, 6}, size=7, tau=0.99)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# SB.13 — chilli_leakage = 0.20 exactly (Tier 1's strict `< 0.20` cap fails; falls to Tier 2)
# Spec lines 5238–5247
#
# v3: probs=[0.74, 0.04, 0.01, 0.00, 0.00, 0.01], chilli_leak=0.20 exactly
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.71, margin=0.42, reliability=0.74
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.86, margin=0.79
# Conformal (tau=0.40): set={0}, size=1
# -> Tier 2, T5 alert: False
# Walk: Rule 7 chilli < 0.20 fails at 0.20 (strict). Rule 8 chilli < 0.30 passes at 0.20.
# ---------------------------------------------------------------------------

def test_SB_13():
    """SB.13 — chilli=0.20 exactly; Rule 7's strict < 0.20 fails; Rule 8's < 0.30 passes → Tier 2."""
    v3 = _make_signal([0.74, 0.04, 0.01, 0.00, 0.00, 0.01], chilli_leak=0.20)
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.71, margin=0.42, reliability=0.74)
    classifier = _make_classifier(argmax=0, max_val=0.86, margin=0.79)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# SB.14 — combined_margin = 0 exactly (top two classes tied; max=0.40 < 0.45 → Rule 4)
# Spec lines 5249–5258
#
# v3: probs=[0.40, 0.40, 0.05, 0.05, 0.05, 0.05], chilli_leak=0.00
# LoRA: probs=[0.40, 0.40, 0.05, 0.05, 0.05, 0.05]
# PSV: argmax=0, max=0.50, margin=0.00, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: argmax=0 (numpy first-index when foliar and septoria tied at 0.40), max=0.40, margin=0.00
# Conformal (tau=0.65): threshold 0.35; set={0, 1}, size=2
# -> Tier 4A (Rule 4: max 0.40 < 0.45), T5 alert: False
# ---------------------------------------------------------------------------

def test_SB_14():
    """SB.14 — margin=0 (tie); max=0.40 < 0.45; Rule 4 fires before Rule 6 → Tier 4A."""
    v3 = _make_signal([0.40, 0.40, 0.05, 0.05, 0.05, 0.05], chilli_leak=0.00)
    lora = _make_signal([0.40, 0.40, 0.05, 0.05, 0.05, 0.05])
    psv = _make_psv(argmax=0, max_val=0.50, margin=0.00, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.40, margin=0.00)
    conformal = _make_conformal(pred_set={0, 1}, size=2, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# SB.15 — combined_margin = 0 with high max (Rule 6 fires for tied top classes)
# Spec lines 5260–5268
#
# v3: probs=[0.50, 0.50, 0.00, 0.00, 0.00, 0.00], chilli_leak=0.00
# LoRA: probs=[0.50, 0.50, 0.00, 0.00, 0.00, 0.00]
# PSV: argmax=0, max=0.55, margin=0.00, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.50, margin=0.00
# Conformal (tau=0.55): threshold 0.45; set={0, 1}, size=2
# -> Tier 3A (Rule 6: set_size==2), T5 alert: False
# ---------------------------------------------------------------------------

def test_SB_15():
    """SB.15 — margin=0, max=0.50 >= 0.45; Rule 4 fails; set_size==2 → Rule 6 → Tier 3A."""
    v3 = _make_signal([0.50, 0.50, 0.00, 0.00, 0.00, 0.00], chilli_leak=0.00)
    lora = _make_signal([0.50, 0.50, 0.00, 0.00, 0.00, 0.00])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.00, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.50, margin=0.00)
    conformal = _make_conformal(pred_set={0, 1}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"
