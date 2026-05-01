"""
Section 15.5 — Tier 3A scenarios (S3A.1 – S3A.12).
Spec source: tomato_3_signal_system.md lines 4338–4450.

All Tier 3A scenarios share: prediction_set_size == 2, Rule 6 fires.
Exceptions: S3A.11 (rule 7b — underpowered downgrade from would-be Tier 1)
            S3A.12 (rule 8b — underpowered downgrade from would-be Tier 2)

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
# S3A.1 — Foliar vs septoria (small-lesion confusion)
# Spec lines 4342–4349
#
# v3: probs=[0.44, 0.39, 0.05, 0.03, 0.04, 0.03], chilli_leak=0.02
# LoRA: probs=[0.42, 0.38, 0.06, 0.04, 0.05, 0.05]
# PSV: argmax=0, max=0.51, margin=0.18, reliability=0.71
# Classifier: argmax=0, max=0.46, margin=0.04
# Conformal (τ=0.55): set={0, 1}, size=2
# → Tier 3A, T5 alert: False (rule 6)
# ---------------------------------------------------------------------------

def test_scenario_S3A_1():
    """S3A.1 — Foliar vs septoria. Spec lines 4342-4349."""
    v3 = _make_signal([0.44, 0.39, 0.05, 0.03, 0.04, 0.03], chilli_leak=0.02)
    lora = _make_signal([0.42, 0.38, 0.06, 0.04, 0.05, 0.05])
    psv = _make_psv(argmax=0, max_val=0.51, margin=0.18, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.04)
    conformal = _make_conformal(pred_set={0, 1}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.2 — Late_blight vs foliar (T5 fires via in-set rule)
# Spec lines 4351–4358
#
# v3: probs=[0.40, 0.05, 0.46, 0.02, 0.03, 0.04], chilli_leak=0.00
# LoRA: probs=[0.38, 0.06, 0.43, 0.04, 0.05, 0.04]
# PSV: argmax=2, max=0.49, margin=0.10, reliability=0.62
# Classifier: argmax=2, max=0.45, margin=0.06
# Conformal (τ=0.55): set={0, 2}, size=2
# → Tier 3A, T5 alert: True (rule 6; late_blight in set with prob 0.45 >= 0.20)
# ---------------------------------------------------------------------------

def test_scenario_S3A_2():
    """S3A.2 — Late_blight vs foliar (T5 via in-set rule). Spec lines 4351-4358."""
    v3 = _make_signal([0.40, 0.05, 0.46, 0.02, 0.03, 0.04], chilli_leak=0.00)
    lora = _make_signal([0.38, 0.06, 0.43, 0.04, 0.05, 0.04])
    psv = _make_psv(argmax=2, max_val=0.49, margin=0.10, reliability=0.62)
    classifier = _make_classifier(argmax=2, max_val=0.45, margin=0.06)
    conformal = _make_conformal(pred_set={0, 2}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.3 — YLCV vs healthy (T5 fires via argmax)
# Spec lines 4360–4367
#
# v3: probs=[0.04, 0.04, 0.04, 0.42, 0.04, 0.40], chilli_leak=0.02
# LoRA: probs=[0.05, 0.05, 0.05, 0.40, 0.05, 0.40]
# PSV: argmax=3, max=0.46, margin=0.08, reliability=0.69
# Classifier: argmax=3, max=0.42, margin=0.02
# Conformal (τ=0.55): set={3, 5}, size=2
# → Tier 3A, T5 alert: True (rule 6; YLCV argmax)
# ---------------------------------------------------------------------------

def test_scenario_S3A_3():
    """S3A.3 — YLCV vs healthy (T5 via argmax). Spec lines 4360-4367."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.42, 0.04, 0.40], chilli_leak=0.02)
    lora = _make_signal([0.05, 0.05, 0.05, 0.40, 0.05, 0.40])
    psv = _make_psv(argmax=3, max_val=0.46, margin=0.08, reliability=0.69)
    classifier = _make_classifier(argmax=3, max_val=0.42, margin=0.02)
    conformal = _make_conformal(pred_set={3, 5}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.4 — Mosaic vs foliar (T5 fires via argmax)
# Spec lines 4369–4376
#
# v3: probs=[0.41, 0.04, 0.03, 0.03, 0.46, 0.03], chilli_leak=0.00
# LoRA: probs=[0.39, 0.05, 0.04, 0.04, 0.43, 0.05]
# PSV: argmax=4, max=0.48, margin=0.12, reliability=0.66
# Classifier: argmax=4, max=0.45, margin=0.04
# Conformal (τ=0.55): set={0, 4}, size=2
# → Tier 3A, T5 alert: True (rule 6; mosaic argmax)
# ---------------------------------------------------------------------------

def test_scenario_S3A_4():
    """S3A.4 — Mosaic vs foliar (T5 via argmax). Spec lines 4369-4376."""
    v3 = _make_signal([0.41, 0.04, 0.03, 0.03, 0.46, 0.03], chilli_leak=0.00)
    lora = _make_signal([0.39, 0.05, 0.04, 0.04, 0.43, 0.05])
    psv = _make_psv(argmax=4, max_val=0.48, margin=0.12, reliability=0.66)
    classifier = _make_classifier(argmax=4, max_val=0.45, margin=0.04)
    conformal = _make_conformal(pred_set={0, 4}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.5 — Healthy vs OOD (uncertain whether image is even a tomato)
# Spec lines 4378–4385
#
# v3: probs=[0.10, 0.05, 0.05, 0.05, 0.05, 0.55], chilli_leak=0.15
# LoRA: probs=[0.10, 0.05, 0.05, 0.05, 0.05, 0.70]
# PSV: argmax=5, max=0.40, margin=0.05, reliability=0.45
# Classifier: argmax=5, max=0.45, margin=0.05; OOD probability 0.40
# Conformal (τ=0.60): set={5, 6}, size=2
# → Tier 3A, T5 alert: False (rule 6; PSV reliability 0.45 >= 0.40 so Rule 3 fails)
# ---------------------------------------------------------------------------

def test_scenario_S3A_5():
    """S3A.5 — Healthy vs OOD. Spec lines 4378-4385."""
    v3 = _make_signal([0.10, 0.05, 0.05, 0.05, 0.05, 0.55], chilli_leak=0.15)
    lora = _make_signal([0.10, 0.05, 0.05, 0.05, 0.05, 0.70])
    psv = _make_psv(argmax=5, max_val=0.40, margin=0.05, reliability=0.45)
    classifier = _make_classifier(argmax=5, max_val=0.45, margin=0.05)
    conformal = _make_conformal(pred_set={5, 6}, size=2, tau=0.60)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.6 — Late_blight vs mosaic (T5 fires via argmax + in-set; mosaic in-set irrelevant)
# Spec lines 4387–4394
#
# v3: probs=[0.04, 0.04, 0.46, 0.04, 0.40, 0.02], chilli_leak=0.00
# LoRA: probs=[0.05, 0.05, 0.42, 0.05, 0.39, 0.04]
# PSV: argmax=2, max=0.55, margin=0.18, reliability=0.71
# Classifier: argmax=2, max=0.44, margin=0.04
# Conformal (τ=0.55): set={2, 4}, size=2
# → Tier 3A, T5 alert: True (rule 6; late_blight argmax + in-set)
# ---------------------------------------------------------------------------

def test_scenario_S3A_6():
    """S3A.6 — Late_blight vs mosaic (T5 via argmax). Spec lines 4387-4394."""
    v3 = _make_signal([0.04, 0.04, 0.46, 0.04, 0.40, 0.02], chilli_leak=0.00)
    lora = _make_signal([0.05, 0.05, 0.42, 0.05, 0.39, 0.04])
    psv = _make_psv(argmax=2, max_val=0.55, margin=0.18, reliability=0.71)
    classifier = _make_classifier(argmax=2, max_val=0.44, margin=0.04)
    conformal = _make_conformal(pred_set={2, 4}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.7 — Septoria vs late_blight (T5 fires via in-set rule)
# Spec lines 4396–4403
#
# v3: probs=[0.05, 0.46, 0.40, 0.03, 0.03, 0.03], chilli_leak=0.00
# LoRA: probs=[0.06, 0.43, 0.38, 0.04, 0.04, 0.05]
# PSV: argmax=1, max=0.50, margin=0.15, reliability=0.74
# Classifier: argmax=1, max=0.45, margin=0.05
# Conformal (τ=0.55): set={1, 2}, size=2
# → Tier 3A, T5 alert: True (rule 6; late_blight in set with prob >= 0.20)
# ---------------------------------------------------------------------------

def test_scenario_S3A_7():
    """S3A.7 — Septoria vs late_blight (T5 via in-set). Spec lines 4396-4403."""
    v3 = _make_signal([0.05, 0.46, 0.40, 0.03, 0.03, 0.03], chilli_leak=0.00)
    lora = _make_signal([0.06, 0.43, 0.38, 0.04, 0.04, 0.05])
    psv = _make_psv(argmax=1, max_val=0.50, margin=0.15, reliability=0.74)
    classifier = _make_classifier(argmax=1, max_val=0.45, margin=0.05)
    conformal = _make_conformal(pred_set={1, 2}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.8 — YLCV vs mosaic (T5 fires via YLCV argmax only; mosaic in-set is irrelevant)
# Spec lines 4405–4412
#
# v3: probs=[0.04, 0.04, 0.03, 0.45, 0.40, 0.04], chilli_leak=0.00
# LoRA: probs=[0.05, 0.05, 0.04, 0.43, 0.39, 0.04]
# PSV: argmax=3, max=0.51, margin=0.16, reliability=0.69
# Classifier: argmax=3, max=0.44, margin=0.04
# Conformal (τ=0.55): set={3, 4}, size=2
# → Tier 3A, T5 alert: True (rule 6; YLCV argmax fires T5; mosaic in-set does NOT)
# ---------------------------------------------------------------------------

def test_scenario_S3A_8():
    """S3A.8 — YLCV vs mosaic (T5 via YLCV argmax only). Spec lines 4405-4412."""
    v3 = _make_signal([0.04, 0.04, 0.03, 0.45, 0.40, 0.04], chilli_leak=0.00)
    lora = _make_signal([0.05, 0.05, 0.04, 0.43, 0.39, 0.04])
    psv = _make_psv(argmax=3, max_val=0.51, margin=0.16, reliability=0.69)
    classifier = _make_classifier(argmax=3, max_val=0.44, margin=0.04)
    conformal = _make_conformal(pred_set={3, 4}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.9 — Foliar vs healthy at boundary
# Spec lines 4414–4421
#
# v3: probs=[0.42, 0.04, 0.02, 0.02, 0.02, 0.43], chilli_leak=0.05
# LoRA: probs=[0.40, 0.05, 0.04, 0.04, 0.04, 0.43]
# PSV: argmax=5, max=0.49, margin=0.12, reliability=0.65
# Classifier: argmax=5, max=0.43, margin=0.02
# Conformal (τ=0.55): set={0, 5}, size=2
# → Tier 3A, T5 alert: False (rule 6)
# ---------------------------------------------------------------------------

def test_scenario_S3A_9():
    """S3A.9 — Foliar vs healthy at boundary. Spec lines 4414-4421."""
    v3 = _make_signal([0.42, 0.04, 0.02, 0.02, 0.02, 0.43], chilli_leak=0.05)
    lora = _make_signal([0.40, 0.05, 0.04, 0.04, 0.04, 0.43])
    psv = _make_psv(argmax=5, max_val=0.49, margin=0.12, reliability=0.65)
    classifier = _make_classifier(argmax=5, max_val=0.43, margin=0.02)
    conformal = _make_conformal(pred_set={0, 5}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.10 — Confident enough for Tier 1 conditions but set size 2 (Rule 6 wins)
# Spec lines 4423–4430
#
# v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: probs=[0.50, 0.40, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.78, margin=0.55, reliability=0.85
# Classifier: argmax=0, max=0.86, margin=0.45 (would-be Tier 1)
# Conformal (τ=0.65): set={0, 1}, size=2
# → Tier 3A, T5 alert: False (rule 6 fires before rule 7)
# ---------------------------------------------------------------------------

def test_scenario_S3A_10():
    """S3A.10 — Would-be Tier 1 but set size 2 (Rule 6 wins). Spec lines 4423-4430."""
    v3 = _make_signal([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal([0.50, 0.40, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.78, margin=0.55, reliability=0.85)
    classifier = _make_classifier(argmax=0, max_val=0.86, margin=0.45)
    conformal = _make_conformal(pred_set={0, 1}, size=2, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3A.11 — Definitive single-class but argmax is underpowered class (YLCV)
# Spec lines 4432–4440
#
# v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
# LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
# PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
# Classifier: argmax=3 (YLCV), max=0.88, margin=0.82
# Conformal (τ=0.40): set={3}, size=1 (would-be Tier 1)
# Underpowered: YLCV flagged underpowered (F.0 recall < 0.50)
# → Tier 3A (downgrade via sub-rule 7b), T5 alert: True (YLCV argmax)
# ---------------------------------------------------------------------------

def test_scenario_S3A_11():
    """S3A.11 — Definitive YLCV but underpowered (sub-rule 7b). Spec lines 4432-4440."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00)
    lora = _make_signal([0.05, 0.05, 0.04, 0.81, 0.02, 0.03])
    psv = _make_psv(argmax=3, max_val=0.74, margin=0.50, reliability=0.78)
    classifier = _make_classifier(argmax=3, max_val=0.88, margin=0.82)
    conformal = _make_conformal(pred_set={3}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
        underpowered_classes={3},  # YLCV index=3 flagged underpowered
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7b"


# ---------------------------------------------------------------------------
# S3A.12 — Confident single-class but argmax is underpowered (mosaic)
# Spec lines 4442–4450
#
# v3: probs=[0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04
# LoRA: probs=[0.05, 0.05, 0.05, 0.06, 0.71, 0.08]
# PSV: argmax=4, max=0.62, margin=0.32, reliability=0.69
# Classifier: argmax=4 (mosaic), max=0.72, margin=0.55
# Conformal (τ=0.55): set={4}, size=1
# Underpowered: mosaic flagged underpowered
# → Tier 3A (downgrade via sub-rule 8b), T5 alert: True (mosaic argmax)
# ---------------------------------------------------------------------------

def test_scenario_S3A_12():
    """S3A.12 — Confident mosaic but underpowered (sub-rule 8b). Spec lines 4442-4450."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04)
    lora = _make_signal([0.05, 0.05, 0.05, 0.06, 0.71, 0.08])
    psv = _make_psv(argmax=4, max_val=0.62, margin=0.32, reliability=0.69)
    classifier = _make_classifier(argmax=4, max_val=0.72, margin=0.55)
    conformal = _make_conformal(pred_set={4}, size=1, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
        underpowered_classes={4},  # mosaic index=4 flagged underpowered
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "8b"
