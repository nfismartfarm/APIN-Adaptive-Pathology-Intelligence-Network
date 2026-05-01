"""
Section 15.13 — Underpowered class downgrade scenarios (SUP.1 – SUP.7).
Spec source: tomato_3_signal_system.md lines 5270–5350.

These scenarios demonstrate the per-class minimum-recall guard (Section 14.4).
When a class is underpowered (recall < 0.50 from F.0), sub-rules 7b/8b downgrade
from Tier 1/2 to Tier 3A. Sub-rules 7a/8a (DEGRADED IQA) take precedence over 7b/8b.

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
# SUP.1 — Definitive YLCV with underpowered guard → 3A downgrade + T5
# Spec lines 5274–5283
#
# v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
# LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
# PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
# IQA: ACCEPTABLE
# Classifier: argmax=3 (YLCV), max=0.88, margin=0.82
# Conformal (tau=0.40): set={3}, size=1
# Underpowered: YLCV recall < 0.50 (flagged underpowered)
# -> Tier 3A (downgrade via sub-rule 7b), T5 alert: True (YLCV argmax, max 0.88 >= 0.20)
# ---------------------------------------------------------------------------

def test_SUP_1():
    """SUP.1 — Definitive YLCV; underpowered guard sub-rule 7b fires → Tier 3A; T5 True."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00)
    lora = _make_signal([0.05, 0.05, 0.04, 0.81, 0.02, 0.03])
    psv = _make_psv(argmax=3, max_val=0.74, margin=0.50, reliability=0.78)
    classifier = _make_classifier(argmax=3, max_val=0.88, margin=0.82)
    conformal = _make_conformal(pred_set={3}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
        underpowered_classes={3},
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7b"


# ---------------------------------------------------------------------------
# SUP.2 — Confident mosaic with underpowered guard → 3A downgrade + T5
# Spec lines 5285–5294
#
# v3: probs=[0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04
# LoRA: probs=[0.05, 0.05, 0.05, 0.06, 0.71, 0.08]
# PSV: argmax=4, max=0.62, margin=0.32, reliability=0.69
# IQA: ACCEPTABLE
# Classifier: argmax=4 (mosaic), max=0.72, margin=0.55
# Conformal (tau=0.55): set={4}, size=1
# Underpowered: mosaic recall < 0.50 (flagged underpowered)
# -> Tier 3A (downgrade via sub-rule 8b), T5 alert: True (mosaic argmax, max 0.72 >= 0.20)
# ---------------------------------------------------------------------------

def test_SUP_2():
    """SUP.2 — Confident mosaic; underpowered guard sub-rule 8b fires → Tier 3A; T5 True."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04)
    lora = _make_signal([0.05, 0.05, 0.05, 0.06, 0.71, 0.08])
    psv = _make_psv(argmax=4, max_val=0.62, margin=0.32, reliability=0.69)
    classifier = _make_classifier(argmax=4, max_val=0.72, margin=0.55)
    conformal = _make_conformal(pred_set={4}, size=1, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
        underpowered_classes={4},
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "8b"


# ---------------------------------------------------------------------------
# SUP.3 — YLCV with IQA DEGRADED — sub-rule 7a wins over 7b → Tier 3D not 3A
# Spec lines 5296–5306
#
# v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
# LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
# PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
# IQA: DEGRADED
# Classifier: argmax=3 (YLCV), max=0.88, margin=0.82
# Conformal (tau=0.40): set={3}, size=1
# Underpowered: YLCV recall < 0.50 (flagged underpowered)
# -> Tier 3D (sub-rule 7a wins over 7b), T5 alert: True (YLCV argmax, max 0.88 >= 0.20)
# ---------------------------------------------------------------------------

def test_SUP_3():
    """SUP.3 — YLCV + DEGRADED IQA + underpowered; 7a takes precedence over 7b → Tier 3D; T5 True."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00)
    lora = _make_signal([0.05, 0.05, 0.04, 0.81, 0.02, 0.03])
    psv = _make_psv(argmax=3, max_val=0.74, margin=0.50, reliability=0.78)
    classifier = _make_classifier(argmax=3, max_val=0.88, margin=0.82)
    conformal = _make_conformal(pred_set={3}, size=1, tau=0.40)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
        underpowered_classes={3},
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# SUP.4 — Mosaic with PSV unreliable — Rule 3 fires before Rule 7 → Tier 3C, T5 fires
# Spec lines 5308–5316
#
# v3: probs=[0.05, 0.04, 0.02, 0.02, 0.85, 0.02], chilli_leak=0.00
# LoRA: probs=[0.06, 0.05, 0.04, 0.02, 0.81, 0.02]
# PSV: argmax=4, max=0.30, margin=0.05, reliability=0.30 (PSV unreliable)
# IQA: ACCEPTABLE
# Classifier: argmax=4 (mosaic), max=0.88, margin=0.83
# Conformal (tau=0.40): set={4}, size=1
# -> Tier 3C (Rule 3 fires due to PSV unreliable), T5 alert: True (mosaic argmax, max 0.88 >= 0.20)
# ---------------------------------------------------------------------------

def test_SUP_4():
    """SUP.4 — Mosaic + PSV reliability=0.30; Rule 3 fires first (before underpowered check) → Tier 3C; T5 True."""
    v3 = _make_signal([0.05, 0.04, 0.02, 0.02, 0.85, 0.02], chilli_leak=0.00)
    lora = _make_signal([0.06, 0.05, 0.04, 0.02, 0.81, 0.02])
    psv = _make_psv(argmax=4, max_val=0.30, margin=0.05, reliability=0.30)
    classifier = _make_classifier(argmax=4, max_val=0.88, margin=0.83)
    conformal = _make_conformal(pred_set={4}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# SUP.5 — Healthy not underpowered, no downgrade
# Spec lines 5318–5327
#
# v3: probs=[0.01, 0.02, 0.01, 0.02, 0.01, 0.91], chilli_leak=0.02
# LoRA: probs=[0.02, 0.03, 0.02, 0.02, 0.02, 0.89]
# PSV: argmax=5 (healthy), max=0.79, margin=0.58, reliability=0.83
# IQA: ACCEPTABLE
# Classifier: argmax=5 (healthy), max=0.93, margin=0.88
# Conformal (tau=0.40): set={5}, size=1
# Underpowered: healthy is NOT flagged underpowered
# -> Tier 1 (no downgrade), T5 alert: False
# ---------------------------------------------------------------------------

def test_SUP_5():
    """SUP.5 — Healthy class; not in underpowered set; sub-rule 7c default fires → Tier 1; T5 False."""
    v3 = _make_signal([0.01, 0.02, 0.01, 0.02, 0.01, 0.91], chilli_leak=0.02)
    lora = _make_signal([0.02, 0.03, 0.02, 0.02, 0.02, 0.89])
    psv = _make_psv(argmax=5, max_val=0.79, margin=0.58, reliability=0.83)
    classifier = _make_classifier(argmax=5, max_val=0.93, margin=0.88)
    conformal = _make_conformal(pred_set={5}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# SUP.6 — YLCV at low confidence (Tier 4A) — underpowered guard doesn't apply
# Spec lines 5329–5338
#
# v3: probs=[0.20, 0.20, 0.05, 0.30, 0.10, 0.05], chilli_leak=0.10
# LoRA: probs=[0.22, 0.22, 0.06, 0.30, 0.10, 0.10]
# PSV: argmax=3, max=0.40, margin=0.05, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: argmax=3 (YLCV), max=0.31, margin=0.02
# Conformal (tau=0.83): threshold 0.17; set={0, 1, 3}, size=3
# Underpowered: YLCV recall < 0.50 (flagged underpowered)
# -> Tier 4A (Rule 4: max 0.31 < 0.45 fires before sub-rules 7b/8b), T5 alert: True (YLCV argmax, max 0.31 >= 0.20)
# ---------------------------------------------------------------------------

def test_SUP_6():
    """SUP.6 — YLCV underpowered but max=0.31 < 0.45; Rule 4 fires before underpowered check → Tier 4A; T5 True."""
    v3 = _make_signal([0.20, 0.20, 0.05, 0.30, 0.10, 0.05], chilli_leak=0.10)
    lora = _make_signal([0.22, 0.22, 0.06, 0.30, 0.10, 0.10])
    psv = _make_psv(argmax=3, max_val=0.40, margin=0.05, reliability=0.55)
    classifier = _make_classifier(argmax=3, max_val=0.31, margin=0.02)
    conformal = _make_conformal(pred_set={0, 1, 3}, size=3, tau=0.83)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
        underpowered_classes={3},
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# SUP.7 — Confident mosaic with IQA DEGRADED + underpowered: sub-rule 8a wins over 8b → Tier 3D
# Spec lines 5340–5350
#
# v3: probs=[0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04
# LoRA: probs=[0.05, 0.05, 0.05, 0.06, 0.71, 0.08]
# PSV: argmax=4, max=0.62, margin=0.32, reliability=0.69
# IQA: DEGRADED
# Classifier: argmax=4 (mosaic), max=0.72, margin=0.55
# Conformal (tau=0.55): set={4}, size=1
# Underpowered: mosaic recall < 0.50 (flagged underpowered)
# -> Tier 3D (sub-rule 8a wins over 8b), T5 alert: True (mosaic argmax, max 0.72 >= 0.20)
# ---------------------------------------------------------------------------

def test_SUP_7():
    """SUP.7 — Mosaic + DEGRADED IQA + underpowered; 8a takes precedence over 8b → Tier 3D; T5 True."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04)
    lora = _make_signal([0.05, 0.05, 0.05, 0.06, 0.71, 0.08])
    psv = _make_psv(argmax=4, max_val=0.62, margin=0.32, reliability=0.69)
    classifier = _make_classifier(argmax=4, max_val=0.72, margin=0.55)
    conformal = _make_conformal(pred_set={4}, size=1, tau=0.55)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
        underpowered_classes={4},
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "8a"
