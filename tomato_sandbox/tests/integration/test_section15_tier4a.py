"""
Section 15.9 — Tier 4A scenarios (S4A.1 – S4A.12, plus S4A.5b).
Spec source: tomato_3_signal_system.md lines 4769–4900.

Most Tier 4A scenarios share: combined_max_prob < 0.45 → Rule 4 fires.
Special cases:
  S4A.5b: empty conformal set + max >= 0.45 → Rule 5 empty-set sub-rule.
  S4A.7:  max=0.45 exactly → Rule 4 strict `< 0.45` fails; falls to Rule 9.
  S4A.8:  max=0.50, margin=0.15 < 0.20 → Rule 8 margin fails; falls to Rule 9.
  (S4A.7 and S4A.8 both have rule_id_fired == "catch_all_low_confidence")

T5 True: S4A.6 (late_blight argmax AND in set), S4A.11 (late_blight in set).

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
# S4A.1 — Highly uncertain across all classes
# Spec lines 4773–4780
#
# v3: probs=[0.18, 0.17, 0.15, 0.13, 0.12, 0.05], chilli_leak=0.20
# LoRA: probs=[0.20, 0.18, 0.15, 0.15, 0.14, 0.18]
# PSV: argmax=0, max=0.30, margin=0.05, reliability=0.55
# Classifier: argmax=0, max=0.21, margin=0.03
# Conformal (τ=0.85, threshold 0.15): set={0, 1, 2, 3, 4}, size=5
# → Tier 4A, T5 alert: False (rule 4; max 0.21 < 0.45)
# ---------------------------------------------------------------------------

def test_S4A_1():
    """S4A.1 — Highly uncertain; max=0.21 < 0.45 → Rule 4 → Tier 4A."""
    v3 = _make_signal([0.18, 0.17, 0.15, 0.13, 0.12, 0.05], chilli_leak=0.20)
    lora = _make_signal([0.20, 0.18, 0.15, 0.15, 0.14, 0.18])
    psv = _make_psv(argmax=0, max_val=0.30, margin=0.05, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.21, margin=0.03)
    conformal = _make_conformal(pred_set={0, 1, 2, 3, 4}, size=5, tau=0.85)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.2 — Max prob just under threshold
# Spec lines 4782–4789
#
# v3: probs=[0.45, 0.20, 0.10, 0.10, 0.07, 0.06], chilli_leak=0.02
# LoRA: probs=[0.42, 0.22, 0.12, 0.10, 0.08, 0.06]
# PSV: argmax=0, max=0.40, margin=0.05, reliability=0.55
# Classifier: argmax=0, max=0.44, margin=0.21
# Conformal (τ=0.65): set={0}, size=1
# → Tier 4A, T5 alert: False (rule 4; 0.44 < 0.45)
# ---------------------------------------------------------------------------

def test_S4A_2():
    """S4A.2 — max=0.44 (just below 0.45) → Rule 4 → Tier 4A."""
    v3 = _make_signal([0.45, 0.20, 0.10, 0.10, 0.07, 0.06], chilli_leak=0.02)
    lora = _make_signal([0.42, 0.22, 0.12, 0.10, 0.08, 0.06])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.05, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.44, margin=0.21)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.3 — Single-class set with very low confidence (chilli=0.40 at boundary)
# Spec lines 4791–4798
#
# v3: probs=[0.30, 0.10, 0.05, 0.05, 0.05, 0.05], chilli_leak=0.40
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.40, margin=0.10, reliability=0.55
# Classifier: argmax=0, max=0.30, margin=0.18
# Conformal (τ=0.74): set={0}, size=1
# → Tier 4A, T5 alert: False
# (chilli_leak=0.40 exactly fails Rule 3 `> 0.40`; max=0.30 < 0.45 → Rule 4)
# ---------------------------------------------------------------------------

def test_S4A_3():
    """S4A.3 — chilli=0.40 exactly (Rule 3 fails); max=0.30 < 0.45 → Rule 4 → Tier 4A."""
    v3 = _make_signal([0.30, 0.10, 0.05, 0.05, 0.05, 0.05], chilli_leak=0.40)
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.10, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.30, margin=0.18)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.74)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.4 — Max 0.40 with multi-class set (Rule 4 has higher priority than Rule 6)
# Spec lines 4800–4808
#
# v3: probs=[0.40, 0.30, 0.10, 0.05, 0.05, 0.05], chilli_leak=0.05
# LoRA: probs=[0.42, 0.31, 0.10, 0.05, 0.06, 0.06]
# PSV: argmax=0, max=0.50, margin=0.18, reliability=0.65
# Classifier: argmax=0, max=0.40, margin=0.10
# Conformal (τ=0.65): set={0, 1}, size=2
# → Tier 4A, T5 alert: False (rule 4 fires before rule 6)
# ---------------------------------------------------------------------------

def test_S4A_4():
    """S4A.4 — max=0.40 < 0.45; Rule 4 fires before Rule 6 despite set_size==2 → Tier 4A."""
    v3 = _make_signal([0.40, 0.30, 0.10, 0.05, 0.05, 0.05], chilli_leak=0.05)
    lora = _make_signal([0.42, 0.31, 0.10, 0.05, 0.06, 0.06])
    psv = _make_psv(argmax=0, max_val=0.50, margin=0.18, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.40, margin=0.10)
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
# S4A.5 — Empty set with low max (Rule 4 catches first)
# Spec lines 4810–4818
#
# v3: probs=[0.38, 0.19, 0.10, 0.10, 0.10, 0.08], chilli_leak=0.05
# LoRA: probs=[0.40, 0.22, 0.12, 0.10, 0.10, 0.06]
# PSV: argmax=0, max=0.40, margin=0.10, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: P_final=[0.42, 0.20, 0.10, 0.10, 0.10, 0.05, 0.03], max=0.42, margin=0.22
# Conformal (τ=0.55): threshold 0.45; set={}, size=0
# → Tier 4A, T5 alert: False (rule 4 fires first; max 0.42 < 0.45)
# ---------------------------------------------------------------------------

def test_S4A_5():
    """S4A.5 — Empty set; but max=0.42 < 0.45 so Rule 4 fires first → Tier 4A."""
    v3 = _make_signal([0.38, 0.19, 0.10, 0.10, 0.10, 0.08], chilli_leak=0.05)
    lora = _make_signal([0.40, 0.22, 0.12, 0.10, 0.10, 0.06])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.10, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.42, margin=0.22)
    conformal = _make_conformal(pred_set=set(), size=0, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.5b — Empty set with adequate max (Rule 5 empty-set sub-rule fires)
# Spec lines 4820–4828
#
# v3: probs=[0.51, 0.10, 0.08, 0.08, 0.08, 0.10], chilli_leak=0.05
# LoRA: probs=[0.50, 0.12, 0.10, 0.08, 0.10, 0.10]
# PSV: argmax=0, max=0.45, margin=0.10, reliability=0.60
# IQA: ACCEPTABLE
# Classifier: P_final=[0.50, 0.10, 0.10, 0.10, 0.10, 0.08, 0.02], max=0.50, margin=0.40
# Conformal (τ=0.40): threshold 0.60; set={}, size=0 (no class above 0.60)
# → Tier 4A, T5 alert: False (rule 5 empty-set sub-rule; max 0.50 >= 0.45)
# ---------------------------------------------------------------------------

def test_S4A_5b():
    """S4A.5b — max=0.50 >= 0.45 so Rule 4 fails; empty set → Rule 5 empty-set sub-rule → Tier 4A."""
    v3 = _make_signal([0.51, 0.10, 0.08, 0.08, 0.08, 0.10], chilli_leak=0.05)
    lora = _make_signal([0.50, 0.12, 0.10, 0.08, 0.10, 0.10])
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
# S4A.6 — Tier 4A with late_blight in set (T5 fires)
# Spec lines 4830–4837
#
# v3: probs=[0.20, 0.20, 0.21, 0.10, 0.07, 0.02], chilli_leak=0.20
# LoRA: probs=[0.25, 0.22, 0.25, 0.10, 0.08, 0.10]
# PSV: argmax=2, max=0.40, margin=0.05, reliability=0.55
# Classifier: argmax=2, max=0.21, margin=0.01
# Conformal (τ=0.83, threshold 0.17): set={0(0.20), 1(0.20), 2(0.21)}, size=3
# → Tier 4A, T5 alert: True (rule 4; argmax late_blight fires T5 first bullet)
# ---------------------------------------------------------------------------

def test_S4A_6():
    """S4A.6 — max=0.21 < 0.45 → Rule 4 → Tier 4A; late_blight argmax fires T5."""
    v3 = _make_signal([0.20, 0.20, 0.21, 0.10, 0.07, 0.02], chilli_leak=0.20)
    lora = _make_signal([0.25, 0.22, 0.25, 0.10, 0.08, 0.10])
    psv = _make_psv(argmax=2, max_val=0.40, margin=0.05, reliability=0.55)
    classifier = _make_classifier(argmax=2, max_val=0.21, margin=0.01)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.83)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.7 — Max prob exactly 0.45 (boundary; routes to 4A via Rule 9 catch-all)
# Spec lines 4839–4848
#
# v3: probs=[0.45, 0.20, 0.10, 0.10, 0.10, 0.03], chilli_leak=0.02
# LoRA: probs=[0.46, 0.22, 0.12, 0.10, 0.06, 0.04]
# PSV: argmax=0, max=0.45, margin=0.10, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.45 exactly, margin=0.20
# Conformal (τ=0.65): set={0}, size=1
# → Tier 4A (Rule 9 catch-all), T5 alert: False
# (Rule 4 `< 0.45` ✗ at 0.45 exactly; Rules 5/6 ✗; Rule 7 `>= 0.85` ✗; Rule 8 `>= 0.65` ✗; Rule 9 fires)
# ---------------------------------------------------------------------------

def test_S4A_7():
    """S4A.7 — max=0.45 exactly; Rule 4 strict < 0.45 fails; falls to Rule 9 → Tier 4A."""
    v3 = _make_signal([0.45, 0.20, 0.10, 0.10, 0.10, 0.03], chilli_leak=0.02)
    lora = _make_signal([0.46, 0.22, 0.12, 0.10, 0.06, 0.04])
    psv = _make_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.45, margin=0.20)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "catch_all_low_confidence"


# ---------------------------------------------------------------------------
# S4A.8 — Catch-all Rule 9 fires for max=0.50 with margin=0.15
# Spec lines 4850–4859
#
# v3: probs=[0.46, 0.19, 0.10, 0.10, 0.05, 0.05], chilli_leak=0.05
# LoRA: probs=[0.52, 0.22, 0.10, 0.08, 0.04, 0.04]
# PSV: argmax=0, max=0.55, margin=0.20, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.50, margin=0.15
# Conformal (τ=0.55): set={0}, size=1
# → Tier 4A (Rule 9 catch-all), T5 alert: False
# (max=0.50 >= 0.45 → Rule 4 ✗; Rule 8 margin=0.15 < 0.20 ✗; falls to Rule 9)
# ---------------------------------------------------------------------------

def test_S4A_8():
    """S4A.8 — max=0.50, margin=0.15 < 0.20; Rule 8 margin fails; falls to Rule 9 → Tier 4A."""
    v3 = _make_signal([0.46, 0.19, 0.10, 0.10, 0.05, 0.05], chilli_leak=0.05)
    lora = _make_signal([0.52, 0.22, 0.10, 0.08, 0.04, 0.04])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.20, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.50, margin=0.15)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "catch_all_low_confidence"


# ---------------------------------------------------------------------------
# S4A.9 — Tier 4A with healthy argmax
# Spec lines 4861–4868
#
# v3: probs=[0.10, 0.10, 0.05, 0.05, 0.05, 0.40], chilli_leak=0.25
# LoRA: probs=[0.15, 0.15, 0.10, 0.10, 0.10, 0.40]
# PSV: argmax=5, max=0.40, margin=0.05, reliability=0.62
# Classifier: argmax=5, max=0.42, margin=0.10
# Conformal (τ=0.66): set={5}, size=1
# → Tier 4A, T5 alert: False (rule 4; healthy argmax)
# ---------------------------------------------------------------------------

def test_S4A_9():
    """S4A.9 — max=0.42 < 0.45 + healthy argmax → Rule 4 → Tier 4A, T5 False."""
    v3 = _make_signal([0.10, 0.10, 0.05, 0.05, 0.05, 0.40], chilli_leak=0.25)
    lora = _make_signal([0.15, 0.15, 0.10, 0.10, 0.10, 0.40])
    psv = _make_psv(argmax=5, max_val=0.40, margin=0.05, reliability=0.62)
    classifier = _make_classifier(argmax=5, max_val=0.42, margin=0.10)
    conformal = _make_conformal(pred_set={5}, size=1, tau=0.66)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.10 — Tier 4A with OOD argmax
# Spec lines 4870–4879
#
# v3: probs=[0.13, 0.13, 0.06, 0.06, 0.06, 0.26], chilli_leak=0.30
# LoRA: probs=[0.15, 0.15, 0.10, 0.10, 0.10, 0.40]
# PSV: argmax=6 (OOD), max=0.30, margin=0.05, reliability=0.50
# IQA: ACCEPTABLE
# Classifier: P_final=[0.10, 0.08, 0.05, 0.05, 0.05, 0.27, 0.40]; argmax=6, max=0.40, margin=0.13
# Conformal (τ=0.65): set={6}, size=1
# → Tier 4A (Rule 4: max 0.40 < 0.45), T5 alert: False
# ---------------------------------------------------------------------------

def test_S4A_10():
    """S4A.10 — OOD argmax; max=0.40 < 0.45 → Rule 4 → Tier 4A, T5 False."""
    v3 = _make_signal([0.13, 0.13, 0.06, 0.06, 0.06, 0.26], chilli_leak=0.30)
    lora = _make_signal([0.15, 0.15, 0.10, 0.10, 0.10, 0.40])
    psv = _make_psv(argmax=6, max_val=0.30, margin=0.05, reliability=0.50)
    classifier = _make_classifier(argmax=6, max_val=0.40, margin=0.13)
    conformal = _make_conformal(pred_set={6}, size=1, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.11 — Tier 4A with late_blight in set (T5 fires via in-set rule)
# Spec lines 4881–4889
#
# v3: probs=[0.25, 0.20, 0.20, 0.10, 0.05, 0.10], chilli_leak=0.10
# LoRA: probs=[0.27, 0.22, 0.18, 0.10, 0.06, 0.17]
# PSV: argmax=0, max=0.40, margin=0.10, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: P_final=[0.30, 0.22, 0.20, 0.08, 0.05, 0.10, 0.05]; argmax=0, max=0.30, margin=0.08
# Conformal (τ=0.83): threshold 0.17; set={0, 1, 2}, size=3
# → Tier 4A (Rule 4: max 0.30 < 0.45), T5 alert: True (late_blight in set with prob 0.20 >= 0.20)
# ---------------------------------------------------------------------------

def test_S4A_11():
    """S4A.11 — max=0.30 < 0.45 → Rule 4 → Tier 4A; late_blight in set with 0.20 fires T5."""
    v3 = _make_signal([0.25, 0.20, 0.20, 0.10, 0.05, 0.10], chilli_leak=0.10)
    lora = _make_signal([0.27, 0.22, 0.18, 0.10, 0.06, 0.17])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.10, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.30, margin=0.08)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.83)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S4A.12 — Tier 4A despite IQA HIGH (input is genuinely confusing)
# Spec lines 4891–4900
#
# v3: probs=[0.20, 0.18, 0.15, 0.15, 0.12, 0.10], chilli_leak=0.10
# LoRA: probs=[0.22, 0.20, 0.16, 0.14, 0.12, 0.16]
# PSV: argmax=0, max=0.30, margin=0.05, reliability=0.78
# IQA: HIGH
# Classifier: argmax=0, max=0.21, margin=0.03
# Conformal (τ=0.83): set={0, 1, 2, 3, 4}, size=5
# → Tier 4A, T5 alert: False (HIGH IQA does NOT lift Tier 4A)
# ---------------------------------------------------------------------------

def test_S4A_12():
    """S4A.12 — HIGH IQA; max=0.21 < 0.45 → Rule 4 → Tier 4A (IQA HIGH doesn't lift)."""
    v3 = _make_signal([0.20, 0.18, 0.15, 0.15, 0.12, 0.10], chilli_leak=0.10)
    lora = _make_signal([0.22, 0.20, 0.16, 0.14, 0.12, 0.16])
    psv = _make_psv(argmax=0, max_val=0.30, margin=0.05, reliability=0.78)
    classifier = _make_classifier(argmax=0, max_val=0.21, margin=0.03)
    conformal = _make_conformal(pred_set={0, 1, 2, 3, 4}, size=5, tau=0.83)
    iqa = {"decision": "HIGH"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"
