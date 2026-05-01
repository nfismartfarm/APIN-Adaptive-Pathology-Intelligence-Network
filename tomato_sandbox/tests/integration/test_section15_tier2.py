"""
Section 15.4 — Tier 2 scenarios (S2.1 – S2.12).
Spec source: tomato_3_signal_system.md lines 4226–4337.

All Tier 2 scenarios share: prediction_set_size==1, 0.65 <= combined_max_prob < 0.85,
combined_margin >= 0.20, IQA acceptable/high/degraded (handled by sub-rule 8a),
all signals succeeded, psv_reliability >= 0.40, chilli_leakage < 0.30.
Sub-rule 8c (default) fires.

Import contract: from tomato_sandbox.tier.tier_assignment import assign_tier
These tests FAIL with ImportError until Phase 4 implements the module.
"""

import pytest

# Phase 4 will provide this module. Until then, every test fails with ImportError.
from tomato_sandbox.tier.tier_assignment import assign_tier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (duplicated from tier1 for file-level independence)
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
# S2.1 — Foliar at 0.70 confidence
# Spec lines 4230–4237
#
# v3: probs=[0.71, 0.10, 0.05, 0.03, 0.05, 0.04], chilli_leak=0.02
# LoRA: probs=[0.68, 0.12, 0.07, 0.04, 0.05, 0.04]
# PSV: argmax=0, max=0.58, margin=0.28, reliability=0.65
# Classifier: argmax=0, max=0.71, margin=0.45
# Conformal (τ=0.62): set={0}, size=1
# → Tier 2, T5 alert: False (rule 8c)
# ---------------------------------------------------------------------------

def test_scenario_S2_1():
    """S2.1 — Foliar at 0.70 confidence. Spec lines 4230-4237."""
    v3 = _make_signal([0.71, 0.10, 0.05, 0.03, 0.05, 0.04], chilli_leak=0.02)
    lora = _make_signal([0.68, 0.12, 0.07, 0.04, 0.05, 0.04])
    psv = _make_psv(argmax=0, max_val=0.58, margin=0.28, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.71, margin=0.45)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.62)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.2 — Septoria at exact lower bound (0.65)
# Spec lines 4239–4246
#
# v3: probs=[0.10, 0.65, 0.07, 0.05, 0.06, 0.05], chilli_leak=0.02
# LoRA: probs=[0.12, 0.62, 0.08, 0.05, 0.07, 0.06]
# PSV: argmax=1, max=0.55, margin=0.25, reliability=0.58
# Classifier: argmax=1, max=0.65 (exactly), margin=0.42
# Conformal (τ=0.65): set={1}, size=1
# → Tier 2, T5 alert: False (rule 8c; max 0.65 >= 0.65)
# ---------------------------------------------------------------------------

def test_scenario_S2_2():
    """S2.2 — Septoria at exact lower bound (0.65). Spec lines 4239-4246."""
    v3 = _make_signal([0.10, 0.65, 0.07, 0.05, 0.06, 0.05], chilli_leak=0.02)
    lora = _make_signal([0.12, 0.62, 0.08, 0.05, 0.07, 0.06])
    psv = _make_psv(argmax=1, max_val=0.55, margin=0.25, reliability=0.58)
    classifier = _make_classifier(argmax=1, max_val=0.65, margin=0.42)
    conformal = _make_conformal(pred_set={1}, size=1, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.3 — Late_blight at 0.75 (Tier 2 + Tier 5)
# Spec lines 4248–4255
#
# v3: probs=[0.05, 0.05, 0.74, 0.03, 0.05, 0.04], chilli_leak=0.04
# LoRA: probs=[0.06, 0.06, 0.71, 0.04, 0.06, 0.07]
# PSV: argmax=2, max=0.62, margin=0.32, reliability=0.66
# Classifier: argmax=2, max=0.75, margin=0.55
# Conformal (τ=0.55): set={2}, size=1
# → Tier 2, T5 alert: True (rule 8c; T5 fires for late_blight argmax)
# ---------------------------------------------------------------------------

def test_scenario_S2_3():
    """S2.3 — Late_blight at 0.75 (Tier 2 + T5). Spec lines 4248-4255."""
    v3 = _make_signal([0.05, 0.05, 0.74, 0.03, 0.05, 0.04], chilli_leak=0.04)
    lora = _make_signal([0.06, 0.06, 0.71, 0.04, 0.06, 0.07])
    psv = _make_psv(argmax=2, max_val=0.62, margin=0.32, reliability=0.66)
    classifier = _make_classifier(argmax=2, max_val=0.75, margin=0.55)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.4 — YLCV at 0.70 (Tier 2 + Tier 5)
# Spec lines 4257–4264
#
# v3: probs=[0.04, 0.04, 0.04, 0.69, 0.06, 0.05], chilli_leak=0.08
# LoRA: probs=[0.05, 0.05, 0.05, 0.66, 0.10, 0.09]
# PSV: argmax=3, max=0.59, margin=0.30, reliability=0.62
# Classifier: argmax=3, max=0.70, margin=0.46
# Conformal (τ=0.60): set={3}, size=1
# → Tier 2, T5 alert: True (rule 8c; YLCV argmax)
# ---------------------------------------------------------------------------

def test_scenario_S2_4():
    """S2.4 — YLCV at 0.70 (Tier 2 + T5). Spec lines 4257-4264."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.69, 0.06, 0.05], chilli_leak=0.08)
    lora = _make_signal([0.05, 0.05, 0.05, 0.66, 0.10, 0.09])
    psv = _make_psv(argmax=3, max_val=0.59, margin=0.30, reliability=0.62)
    classifier = _make_classifier(argmax=3, max_val=0.70, margin=0.46)
    conformal = _make_conformal(pred_set={3}, size=1, tau=0.60)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.5 — Mosaic at 0.78 (Tier 2 + Tier 5)
# Spec lines 4266–4273
#
# v3: probs=[0.06, 0.05, 0.04, 0.04, 0.76, 0.04], chilli_leak=0.01
# LoRA: probs=[0.07, 0.06, 0.04, 0.05, 0.74, 0.04]
# PSV: argmax=4, max=0.61, margin=0.31, reliability=0.69
# Classifier: argmax=4, max=0.78, margin=0.62
# Conformal (τ=0.55): set={4}, size=1
# → Tier 2, T5 alert: True (rule 8c; mosaic argmax)
# ---------------------------------------------------------------------------

def test_scenario_S2_5():
    """S2.5 — Mosaic at 0.78 (Tier 2 + T5). Spec lines 4266-4273."""
    v3 = _make_signal([0.06, 0.05, 0.04, 0.04, 0.76, 0.04], chilli_leak=0.01)
    lora = _make_signal([0.07, 0.06, 0.04, 0.05, 0.74, 0.04])
    psv = _make_psv(argmax=4, max_val=0.61, margin=0.31, reliability=0.69)
    classifier = _make_classifier(argmax=4, max_val=0.78, margin=0.62)
    conformal = _make_conformal(pred_set={4}, size=1, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.6 — Healthy at 0.72
# Spec lines 4275–4282
#
# v3: probs=[0.06, 0.06, 0.04, 0.06, 0.04, 0.71], chilli_leak=0.03
# LoRA: probs=[0.07, 0.07, 0.05, 0.06, 0.05, 0.70]
# PSV: argmax=5, max=0.55, margin=0.24, reliability=0.61
# Classifier: argmax=5, max=0.72, margin=0.50
# Conformal (τ=0.62): set={5}, size=1
# → Tier 2, T5 alert: False (rule 8c)
# ---------------------------------------------------------------------------

def test_scenario_S2_6():
    """S2.6 — Healthy at 0.72. Spec lines 4275-4282."""
    v3 = _make_signal([0.06, 0.06, 0.04, 0.06, 0.04, 0.71], chilli_leak=0.03)
    lora = _make_signal([0.07, 0.07, 0.05, 0.06, 0.05, 0.70])
    psv = _make_psv(argmax=5, max_val=0.55, margin=0.24, reliability=0.61)
    classifier = _make_classifier(argmax=5, max_val=0.72, margin=0.50)
    conformal = _make_conformal(pred_set={5}, size=1, tau=0.62)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.7 — Foliar with PSV reliability at Tier 2 lower bound (0.40 exactly)
# Spec lines 4284–4291
#
# v3: probs=[0.74, 0.10, 0.04, 0.04, 0.04, 0.02], chilli_leak=0.02
# LoRA: probs=[0.71, 0.11, 0.05, 0.04, 0.05, 0.04]
# PSV: argmax=0, max=0.51, margin=0.22, reliability=0.40 (exactly at Tier 2 lower bound)
# Classifier: argmax=0, max=0.74, margin=0.55
# Conformal (τ=0.58): set={0}, size=1
# → Tier 2, T5 alert: False (PSV 0.40 — Rule 3 strict `< 0.40` fails; Rule 8 `>= 0.40` passes)
# ---------------------------------------------------------------------------

def test_scenario_S2_7():
    """S2.7 — Foliar with PSV reliability at 0.40 exactly. Spec lines 4284-4291."""
    v3 = _make_signal([0.74, 0.10, 0.04, 0.04, 0.04, 0.02], chilli_leak=0.02)
    lora = _make_signal([0.71, 0.11, 0.05, 0.04, 0.05, 0.04])
    psv = _make_psv(argmax=0, max_val=0.51, margin=0.22, reliability=0.40)
    classifier = _make_classifier(argmax=0, max_val=0.74, margin=0.55)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.58)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.8 — Septoria with chilli_leakage at Tier 2 upper bound (0.28)
# Spec lines 4293–4300
#
# v3: probs=[0.08, 0.52, 0.03, 0.03, 0.03, 0.03], chilli_leak=0.28
# LoRA: probs=[0.12, 0.74, 0.04, 0.04, 0.03, 0.03]
# PSV: argmax=1, max=0.59, margin=0.30, reliability=0.66
# Classifier: argmax=1, max=0.78, margin=0.60
# Conformal (τ=0.55): set={1}, size=1
# → Tier 2, T5 alert: False (chilli 0.28 < 0.30 passes Rule 8; 0.28 not > 0.40 so Rule 3 fails)
# ---------------------------------------------------------------------------

def test_scenario_S2_8():
    """S2.8 — Septoria with chilli_leakage=0.28. Spec lines 4293-4300."""
    v3 = _make_signal([0.08, 0.52, 0.03, 0.03, 0.03, 0.03], chilli_leak=0.28)
    lora = _make_signal([0.12, 0.74, 0.04, 0.04, 0.03, 0.03])
    psv = _make_psv(argmax=1, max_val=0.59, margin=0.30, reliability=0.66)
    classifier = _make_classifier(argmax=1, max_val=0.78, margin=0.60)
    conformal = _make_conformal(pred_set={1}, size=1, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.9 — Foliar at exact margin threshold for Tier 2 (margin=0.20 exactly)
# Spec lines 4302–4309
#
# v3: probs=[0.66, 0.19, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: probs=[0.66, 0.22, 0.04, 0.03, 0.03, 0.02]
# PSV: argmax=0, max=0.50, margin=0.20, reliability=0.55
# Classifier: argmax=0, max=0.68, margin=0.20 (exactly)
# Conformal (τ=0.62): set={0}, size=1
# → Tier 2, T5 alert: False (rule 8c; margin 0.20 >= 0.20)
# ---------------------------------------------------------------------------

def test_scenario_S2_9():
    """S2.9 — Foliar at exact margin threshold (0.20). Spec lines 4302-4309."""
    v3 = _make_signal([0.66, 0.19, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal([0.66, 0.22, 0.04, 0.03, 0.03, 0.02])
    psv = _make_psv(argmax=0, max_val=0.50, margin=0.20, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.68, margin=0.20)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.62)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.10 — Late_blight at exact max threshold (Tier 2 + Tier 5)
# Spec lines 4311–4318
#
# v3: probs=[0.10, 0.10, 0.65, 0.05, 0.05, 0.05], chilli_leak=0.00
# LoRA: probs=[0.12, 0.10, 0.62, 0.06, 0.06, 0.04]
# PSV: argmax=2, max=0.51, margin=0.20, reliability=0.49
# Classifier: argmax=2, max=0.65 (exactly), margin=0.40
# Conformal (τ=0.65): set={2}, size=1
# → Tier 2, T5 alert: True (rule 8c; late_blight argmax)
# Note: PSV reliability 0.49 — Rule 3 strict < 0.40 fails; Rule 8 >= 0.40 passes
# ---------------------------------------------------------------------------

def test_scenario_S2_10():
    """S2.10 — Late_blight at exact max threshold (0.65). Spec lines 4311-4318."""
    v3 = _make_signal([0.10, 0.10, 0.65, 0.05, 0.05, 0.05], chilli_leak=0.00)
    lora = _make_signal([0.12, 0.10, 0.62, 0.06, 0.06, 0.04])
    psv = _make_psv(argmax=2, max_val=0.51, margin=0.20, reliability=0.49)
    classifier = _make_classifier(argmax=2, max_val=0.65, margin=0.40)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.11 — Healthy just under Tier 1 cutoff
# Spec lines 4320–4327
#
# v3: probs=[0.04, 0.04, 0.02, 0.04, 0.02, 0.83], chilli_leak=0.01
# LoRA: probs=[0.05, 0.05, 0.03, 0.04, 0.03, 0.80]
# PSV: argmax=5, max=0.72, margin=0.50, reliability=0.78
# Classifier: argmax=5, max=0.84, margin=0.78 (max < 0.85 → not Tier 1)
# Conformal (τ=0.40): set={5}, size=1
# → Tier 2, T5 alert: False (rule 8c; max 0.84 < 0.85 fails Rule 7; 0.84 >= 0.65 passes Rule 8)
# ---------------------------------------------------------------------------

def test_scenario_S2_11():
    """S2.11 — Healthy just under Tier 1 cutoff (max=0.84). Spec lines 4320-4327."""
    v3 = _make_signal([0.04, 0.04, 0.02, 0.04, 0.02, 0.83], chilli_leak=0.01)
    lora = _make_signal([0.05, 0.05, 0.03, 0.04, 0.03, 0.80])
    psv = _make_psv(argmax=5, max_val=0.72, margin=0.50, reliability=0.78)
    classifier = _make_classifier(argmax=5, max_val=0.84, margin=0.78)
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

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# S2.12 — Foliar with high max but tight margin
# Spec lines 4329–4336
#
# v3: probs=[0.80, 0.10, 0.02, 0.01, 0.01, 0.01], chilli_leak=0.05
# LoRA: probs=[0.83, 0.12, 0.02, 0.01, 0.01, 0.01]
# PSV: argmax=0, max=0.55, margin=0.25, reliability=0.65
# Classifier: argmax=0, max=0.87, margin=0.25 (margin < 0.30 → not Tier 1)
# Conformal (τ=0.40): set={0}, size=1
# → Tier 2, T5 alert: False (rule 8c; max 0.87 >= 0.85 but margin 0.25 < 0.30 fails Rule 7;
#   margin 0.25 >= 0.20 passes Rule 8)
# ---------------------------------------------------------------------------

def test_scenario_S2_12():
    """S2.12 — Foliar with high max but tight margin (0.25). Spec lines 4329-4336."""
    v3 = _make_signal([0.80, 0.10, 0.02, 0.01, 0.01, 0.01], chilli_leak=0.05)
    lora = _make_signal([0.83, 0.12, 0.02, 0.01, 0.01, 0.01])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.25, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.87, margin=0.25)
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

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"
