"""
Section 15.3 — Tier 1 scenarios (S1.1 – S1.12).
Spec source: tomato_3_signal_system.md lines 4112–4225.

All Tier 1 scenarios share:
  prediction_set_size==1, combined_max_prob >= 0.85, combined_margin >= 0.30,
  IQA ACCEPTABLE or HIGH, all signals succeeded, psv_reliability >= 0.50,
  chilli_leakage < 0.20. Sub-rule 7c (default) fires.

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
    """Minimal dict representing a signal result."""
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
# S1.1 — Clean foliar prediction
# Spec lines 4116–4124
#
# v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
#   (AUTHORITATIVE: line 4117 scenario body; line 5558 test-code snippet
#    with [0.92, ...] is a typo — BLK-004 Defect-15.1, SPEC-INT-001)
# LoRA: probs=[0.88, 0.05, 0.02, 0.02, 0.02, 0.01]
# PSV: argmax=0 (foliar), max=0.71, margin=0.45, reliability=0.78
# IQA: ACCEPTABLE
# Classifier: argmax=0 (foliar), max=0.91, margin=0.86
# Conformal (τ=0.40): set={0}, size=1
# → Tier 1, T5 alert: False (rule 7c)
# ---------------------------------------------------------------------------

def test_scenario_S1_1():
    """S1.1 — Clean foliar prediction. Spec lines 4116-4124."""
    v3 = _make_signal([0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.88, 0.05, 0.02, 0.02, 0.02, 0.01])
    psv = _make_psv(argmax=0, max_val=0.71, margin=0.45, reliability=0.78)
    classifier = _make_classifier(argmax=0, max_val=0.91, margin=0.86)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.2 — Clean septoria prediction
# Spec lines 4126–4133
# ---------------------------------------------------------------------------

def test_scenario_S1_2():
    """S1.2 — Clean septoria prediction. Spec lines 4126-4133."""
    v3 = _make_signal([0.04, 0.90, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.02)
    lora = _make_signal([0.05, 0.86, 0.03, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=1, max_val=0.74, margin=0.48, reliability=0.81)
    classifier = _make_classifier(argmax=1, max_val=0.89, margin=0.83)
    conformal = _make_conformal(pred_set={1}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.3 — Clean late_blight prediction (Tier 5 alert fires)
# Spec lines 4135–4143
# ---------------------------------------------------------------------------

def test_scenario_S1_3():
    """S1.3 — Clean late_blight prediction. Tier 5 alert fires. Spec lines 4135-4143."""
    v3 = _make_signal([0.02, 0.02, 0.89, 0.01, 0.01, 0.01], chilli_leak=0.04)
    lora = _make_signal([0.03, 0.03, 0.87, 0.02, 0.02, 0.03])
    psv = _make_psv(argmax=2, max_val=0.78, margin=0.55, reliability=0.74)
    classifier = _make_classifier(argmax=2, max_val=0.92, margin=0.87)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.4 — Clean YLCV prediction (Tier 5 alert fires)
# Spec lines 4144–4151
# ---------------------------------------------------------------------------

def test_scenario_S1_4():
    """S1.4 — Clean YLCV prediction. Tier 5 alert fires. Spec lines 4144-4151."""
    v3 = _make_signal([0.02, 0.02, 0.02, 0.84, 0.02, 0.02], chilli_leak=0.06)
    lora = _make_signal([0.03, 0.02, 0.02, 0.85, 0.04, 0.04])
    psv = _make_psv(argmax=3, max_val=0.81, margin=0.62, reliability=0.85)
    classifier = _make_classifier(argmax=3, max_val=0.87, margin=0.78)
    conformal = _make_conformal(pred_set={3}, size=1, tau=0.42)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.5 — Clean mosaic prediction (Tier 5 alert fires)
# Spec lines 4153–4160
# ---------------------------------------------------------------------------

def test_scenario_S1_5():
    """S1.5 — Clean mosaic prediction. Tier 5 alert fires. Spec lines 4153-4160."""
    v3 = _make_signal([0.04, 0.03, 0.02, 0.02, 0.86, 0.01], chilli_leak=0.02)
    lora = _make_signal([0.05, 0.03, 0.02, 0.02, 0.84, 0.04])
    psv = _make_psv(argmax=4, max_val=0.69, margin=0.42, reliability=0.71)
    classifier = _make_classifier(argmax=4, max_val=0.88, margin=0.81)
    conformal = _make_conformal(pred_set={4}, size=1, tau=0.43)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.6 — Clean healthy prediction
# Spec lines 4162–4169
# ---------------------------------------------------------------------------

def test_scenario_S1_6():
    """S1.6 — Clean healthy prediction. Spec lines 4162-4169."""
    v3 = _make_signal([0.01, 0.02, 0.01, 0.02, 0.01, 0.91], chilli_leak=0.02)
    lora = _make_signal([0.02, 0.03, 0.02, 0.02, 0.02, 0.89])
    psv = _make_psv(argmax=5, max_val=0.79, margin=0.58, reliability=0.83)
    classifier = _make_classifier(argmax=5, max_val=0.93, margin=0.88)
    conformal = _make_conformal(pred_set={5}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.7 — Foliar with HIGH IQA
# Spec lines 4171–4179
# ---------------------------------------------------------------------------

def test_scenario_S1_7():
    """S1.7 — Foliar with HIGH IQA. Spec lines 4171-4179."""
    v3 = _make_signal([0.94, 0.02, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.00)
    lora = _make_signal([0.91, 0.03, 0.02, 0.01, 0.01, 0.02])
    psv = _make_psv(argmax=0, max_val=0.82, margin=0.65, reliability=0.92)
    classifier = _make_classifier(argmax=0, max_val=0.96, margin=0.93)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.35)
    iqa = {"decision": "HIGH"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.8 — Late_blight at exact threshold values (boundary inclusive)
# Spec lines 4181–4188
# PSV reliability=0.50 exactly, Classifier max=0.85 exactly, margin=0.30 exactly
# ---------------------------------------------------------------------------

def test_scenario_S1_8():
    """S1.8 — Late_blight at exact threshold values. Tier 5 fires. Spec lines 4181-4188."""
    v3 = _make_signal([0.02, 0.04, 0.84, 0.02, 0.02, 0.01], chilli_leak=0.05)
    lora = _make_signal([0.04, 0.05, 0.80, 0.03, 0.04, 0.04])
    psv = _make_psv(argmax=2, max_val=0.65, margin=0.30, reliability=0.50)
    classifier = _make_classifier(argmax=2, max_val=0.85, margin=0.30)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.50)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.9 — Foliar with PSV reliability at lower bound (0.50)
# Spec lines 4190–4197
# ---------------------------------------------------------------------------

def test_scenario_S1_9():
    """S1.9 — Foliar with PSV reliability at lower bound. Spec lines 4190-4197."""
    v3 = _make_signal([0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05)
    lora = _make_signal([0.88, 0.05, 0.02, 0.02, 0.02, 0.01])
    psv = _make_psv(argmax=0, max_val=0.62, margin=0.32, reliability=0.50)
    classifier = _make_classifier(argmax=0, max_val=0.91, margin=0.86)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.42)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.10 — Septoria with chilli_leakage at upper bound for Tier 1 (0.18 < 0.20)
# Spec lines 4199–4206
# ---------------------------------------------------------------------------

def test_scenario_S1_10():
    """S1.10 — Septoria with chilli_leakage=0.18. Spec lines 4199-4206."""
    v3 = _make_signal([0.05, 0.74, 0.01, 0.01, 0.01, 0.00], chilli_leak=0.18)
    lora = _make_signal([0.06, 0.85, 0.02, 0.02, 0.03, 0.02])
    psv = _make_psv(argmax=1, max_val=0.71, margin=0.42, reliability=0.74)
    classifier = _make_classifier(argmax=1, max_val=0.86, margin=0.79)
    conformal = _make_conformal(pred_set={1}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.11 — Healthy at exact margin threshold (margin=0.30 exactly)
# Spec lines 4208–4215
# ---------------------------------------------------------------------------

def test_scenario_S1_11():
    """S1.11 — Healthy at exact margin threshold. Spec lines 4208-4215."""
    v3 = _make_signal([0.05, 0.05, 0.02, 0.05, 0.02, 0.78], chilli_leak=0.03)
    lora = _make_signal([0.06, 0.05, 0.04, 0.04, 0.03, 0.78])
    psv = _make_psv(argmax=5, max_val=0.65, margin=0.32, reliability=0.61)
    classifier = _make_classifier(argmax=5, max_val=0.85, margin=0.30)
    conformal = _make_conformal(pred_set={5}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# S1.12 — Foliar with high margin from confident agreement
# Spec lines 4217–4224
# ---------------------------------------------------------------------------

def test_scenario_S1_12():
    """S1.12 — Foliar with high margin from confident agreement. Spec lines 4217-4224."""
    v3 = _make_signal([0.96, 0.01, 0.00, 0.00, 0.00, 0.00], chilli_leak=0.03)
    lora = _make_signal([0.95, 0.02, 0.01, 0.01, 0.00, 0.01])
    psv = _make_psv(argmax=0, max_val=0.85, margin=0.72, reliability=0.94)
    classifier = _make_classifier(argmax=0, max_val=0.97, margin=0.94)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.32)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"
