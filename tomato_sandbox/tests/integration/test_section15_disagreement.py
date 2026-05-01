"""
Section 15.14 — Cross-signal disagreement scenarios (SDIS.1 – SDIS.6).
Spec source: tomato_3_signal_system.md lines 5352–5421.

These scenarios explore behavior when v3, LoRA, and PSV disagree.
High JSD between v3 and LoRA is a classifier feature that reduces confidence.

Class indices: 0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy, 6=OOD

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
# SDIS.1 — v3 says foliar, LoRA says septoria, PSV agrees with v3 (high JSD)
# Spec lines 5356–5366
#
# v3: probs=[0.78, 0.12, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.00
# LoRA: probs=[0.10, 0.78, 0.05, 0.03, 0.02, 0.02]
# PSV: argmax=0 (foliar), max=0.65, margin=0.30, reliability=0.71
# JSD(v3, LoRA) ≈ 0.45 (relatively high)
# Classifier: argmax=0, max=0.50, margin=0.25
# Conformal (τ=0.55): set={0, 1}, size=2
# → Tier 3A, T5 alert: False (rule 6)
# ---------------------------------------------------------------------------

def test_SDIS_1():
    """SDIS.1 — v3 foliar vs LoRA septoria; PSV agrees with v3; set_size==2 → Rule 6 → Tier 3A; T5 False."""
    v3 = _make_signal([0.78, 0.12, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.00)
    lora = _make_signal([0.10, 0.78, 0.05, 0.03, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.50, margin=0.25)
    conformal = _make_conformal(pred_set={0, 1}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# SDIS.2 — All three disagree (v3 foliar, LoRA septoria, PSV late_blight)
# Spec lines 5368–5378
#
# v3: probs=[0.50, 0.30, 0.10, 0.04, 0.04, 0.02], chilli_leak=0.00
# LoRA: probs=[0.20, 0.55, 0.15, 0.04, 0.04, 0.02]
# PSV: argmax=2 (late_blight), max=0.45, margin=0.10, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: argmax=0 (foliar by numpy first-index when tied with septoria), max=0.30, margin=0.00
# Conformal (τ=0.80): set={0, 1, 2}, size=3
# → Tier 4A (Rule 4: max 0.30 < 0.45), T5 alert: True (late_blight in set with prob 0.25 >= 0.20)
# ---------------------------------------------------------------------------

def test_SDIS_2():
    """SDIS.2 — Three-way disagreement; classifier max=0.30; Rule 4 → Tier 4A; late_blight in set at 0.25 fires T5."""
    v3 = _make_signal([0.50, 0.30, 0.10, 0.04, 0.04, 0.02], chilli_leak=0.00)
    lora = _make_signal([0.20, 0.55, 0.15, 0.04, 0.04, 0.02])
    psv = _make_psv(argmax=2, max_val=0.45, margin=0.10, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.30, margin=0.00)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.80)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# SDIS.3 — v3 confident foliar, LoRA confident healthy (extreme disagreement)
# Spec lines 5380–5391
#
# v3: probs=[0.85, 0.04, 0.02, 0.02, 0.02, 0.05], chilli_leak=0.00
# LoRA: probs=[0.05, 0.04, 0.02, 0.02, 0.02, 0.85]
# PSV: argmax=5 (healthy), max=0.50, margin=0.10, reliability=0.55
# IQA: ACCEPTABLE
# JSD(v3, LoRA) ≈ 0.55 (very high)
# Classifier: argmax=5 (healthy), max=0.45, margin=0.05
# Conformal (τ=0.60): threshold 0.40; set={0(0.40), 5(0.45)}, size=2
# → Tier 3A (Rule 6: set_size==2), T5 alert: False
# Walk: max 0.45 >= 0.45 → Rule 4 fails. set_size==2 → Rule 6 → Tier 3A.
# ---------------------------------------------------------------------------

def test_SDIS_3():
    """SDIS.3 — v3 foliar vs LoRA healthy extreme disagreement; set_size==2 → Rule 6 → Tier 3A; T5 False."""
    v3 = _make_signal([0.85, 0.04, 0.02, 0.02, 0.02, 0.05], chilli_leak=0.00)
    lora = _make_signal([0.05, 0.04, 0.02, 0.02, 0.02, 0.85])
    psv = _make_psv(argmax=5, max_val=0.50, margin=0.10, reliability=0.55)
    classifier = _make_classifier(argmax=5, max_val=0.45, margin=0.05)
    conformal = _make_conformal(pred_set={0, 5}, size=2, tau=0.60)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# SDIS.4 — v3 says late_blight, LoRA + PSV say foliar (LoRA + PSV outvote v3)
# Spec lines 5393–5402
#
# v3: probs=[0.20, 0.05, 0.65, 0.04, 0.04, 0.02], chilli_leak=0.00
# LoRA: probs=[0.78, 0.10, 0.05, 0.03, 0.02, 0.02]
# PSV: argmax=0, max=0.62, margin=0.30, reliability=0.74
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.62, margin=0.37
# Conformal (τ=0.78): set={0(0.62), 2(0.25)}, size=2
# → Tier 3A (Rule 6: set_size==2), T5 alert: True (late_blight in set with prob 0.25 >= 0.20)
# ---------------------------------------------------------------------------

def test_SDIS_4():
    """SDIS.4 — v3 late_blight vs LoRA+PSV foliar; set admits late_blight at 0.25; set_size==2 → Tier 3A; T5 True."""
    v3 = _make_signal([0.20, 0.05, 0.65, 0.04, 0.04, 0.02], chilli_leak=0.00)
    lora = _make_signal([0.78, 0.10, 0.05, 0.03, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.62, margin=0.30, reliability=0.74)
    classifier = _make_classifier(argmax=0, max_val=0.62, margin=0.37)
    conformal = _make_conformal(pred_set={0, 2}, size=2, tau=0.78)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# SDIS.5 — PSV strongly disagrees with v3 + LoRA (which agree)
# Spec lines 5404–5412
#
# v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: probs=[0.83, 0.06, 0.05, 0.02, 0.02, 0.02]
# PSV: argmax=4 (mosaic), max=0.55, margin=0.20, reliability=0.55
# Classifier: argmax=0, max=0.78, margin=0.62
# Conformal (τ=0.45): set={0}, size=1
# → Tier 2, T5 alert: False (rule 8c; PSV's mosaic call doesn't make it into set or argmax)
# ---------------------------------------------------------------------------

def test_SDIS_5():
    """SDIS.5 — PSV disagrees (mosaic) vs v3+LoRA (foliar); classifier weights v3+LoRA higher → Tier 2; T5 False."""
    v3 = _make_signal([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal([0.83, 0.06, 0.05, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=4, max_val=0.55, margin=0.20, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.78, margin=0.62)
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
# SDIS.6 — All three agree but on a class with low PSV reliability
# Spec lines 5414–5421
#
# v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: probs=[0.83, 0.06, 0.05, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.55, margin=0.18, reliability=0.35
# Classifier: argmax=0, max=0.84, margin=0.78
# Conformal (τ=0.40): set={0}, size=1
# → Tier 3C (Rule 3 fires due to PSV reliability), T5 alert: False
# ---------------------------------------------------------------------------

def test_SDIS_6():
    """SDIS.6 — All agree on foliar but PSV reliability=0.35 < 0.40; Rule 3 fires → Tier 3C; T5 False."""
    v3 = _make_signal([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal([0.83, 0.06, 0.05, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.18, reliability=0.35)
    classifier = _make_classifier(argmax=0, max_val=0.84, margin=0.78)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "3"
