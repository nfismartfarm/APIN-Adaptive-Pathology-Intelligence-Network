"""
Section 15.7 — Tier 3C scenarios (S3C.1 – S3C.12).
Spec source: tomato_3_signal_system.md lines 4551–4665.

All Tier 3C scenarios share: Rule 3 fires due to psv_reliability < 0.40 OR
chilli_leakage > 0.40.  T5 is evaluated independently even when Rule 3 fires.

EXCEPTIONS (BLK-004 Defect-15.3 — scenario body is authoritative):
  S3C.8: subsection header implies Tier 3C but outcome is Tier 2
         (psv reliability exactly 0.40, Rule 3 strict `< 0.40` fails).
  S3C.9: outcome is Tier 4A Rule 9 catch-all
         (chilli_leakage exactly 0.40, Rule 3 strict `> 0.40` fails;
          Rules 7 and 8 chilli caps also exclude it).
  S3C.12: outcome is Tier 4A Rule 9 catch-all, T5 True
          (chilli_leakage exactly 0.30, excluded from Rule 8 `< 0.30`;
           late_blight argmax fires T5 independently).

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
# S3C.1 — PSV reliability just under threshold
# Spec lines 4555–4562
#
# v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: probs=[0.82, 0.06, 0.05, 0.03, 0.02, 0.02]
# PSV: argmax=0, max=0.45, margin=0.10, reliability=0.39
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.85, margin=0.78
# Conformal (τ=0.40): set={0}, size=1
# → Tier 3C, T5 alert: False (rule 3; PSV unreliable trumps Tier 1 conditions)
# ---------------------------------------------------------------------------

def test_S3C_1():
    """S3C.1 — PSV reliability=0.39 (just below 0.40) fires Rule 3 → Tier 3C."""
    v3 = _make_signal([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal([0.82, 0.06, 0.05, 0.03, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.39)
    classifier = _make_classifier(argmax=0, max_val=0.85, margin=0.78)
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
# S3C.2 — Severe PSV unreliability
# Spec lines 4564–4571
#
# v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
# LoRA: probs=[0.88, 0.05, 0.03, 0.02, 0.01, 0.01]
# PSV: argmax=0, max=0.40, margin=0.05, reliability=0.10
# Classifier: argmax=0, max=0.91, margin=0.86
# Conformal (τ=0.40): set={0}, size=1
# → Tier 3C, T5 alert: False (rule 3)
# ---------------------------------------------------------------------------

def test_S3C_2():
    """S3C.2 — PSV reliability=0.10 (very low) fires Rule 3 → Tier 3C."""
    v3 = _make_signal([0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.88, 0.05, 0.03, 0.02, 0.01, 0.01])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.05, reliability=0.10)
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
# S3C.3 — Chilli leakage just over threshold
# Spec lines 4573–4580
#
# v3: probs=[0.50, 0.04, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.41
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
# Classifier: argmax=0, max=0.78, margin=0.65
# Conformal (τ=0.50): set={0}, size=1
# → Tier 3C, T5 alert: False (rule 3; chilli leakage triggers)
# ---------------------------------------------------------------------------

def test_S3C_3():
    """S3C.3 — chilli_leak=0.41 (just over 0.40) fires Rule 3 → Tier 3C."""
    v3 = _make_signal([0.50, 0.04, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.41)
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
# S3C.4 — Very high chilli leakage (probably actually a chilli)
# Spec lines 4582–4589
#
# v3: probs=[0.10, 0.02, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.80
# LoRA: probs=[0.55, 0.10, 0.10, 0.10, 0.10, 0.05]
# PSV: argmax=0, max=0.40, margin=0.10, reliability=0.55
# Classifier: argmax=0, max=0.50, margin=0.30; OOD prob 0.30
# Conformal (τ=0.65): set={0, 6}, size=2
# → Tier 3C, T5 alert: False (rule 3; chilli_leak > 0.40 trumps set size 2)
# ---------------------------------------------------------------------------

def test_S3C_4():
    """S3C.4 — chilli_leak=0.80 fires Rule 3 before Rule 6 → Tier 3C."""
    v3 = _make_signal([0.10, 0.02, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.80)
    lora = _make_signal([0.55, 0.10, 0.10, 0.10, 0.10, 0.05])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.10, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.50, margin=0.30)
    conformal = _make_conformal(pred_set={0, 6}, size=2, tau=0.65)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# S3C.5 — Both PSV unreliable AND chilli leakage
# Spec lines 4591–4599
#
# v3: probs=[0.50, 0.04, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.45
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.30, margin=0.05, reliability=0.30
# Classifier: argmax=0, max=0.78, margin=0.65
# Conformal (τ=0.50): set={0}, size=1
# → Tier 3C, T5 alert: False (rule 3; both subconditions fire)
# ---------------------------------------------------------------------------

def test_S3C_5():
    """S3C.5 — both PSV reliability=0.30 AND chilli_leak=0.45 fire Rule 3 → Tier 3C."""
    v3 = _make_signal([0.50, 0.04, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.45)
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.30, margin=0.05, reliability=0.30)
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
# S3C.6 — PSV unreliable but otherwise would-be Tier 1
# Spec lines 4601–4608
#
# v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
# LoRA: probs=[0.89, 0.04, 0.03, 0.01, 0.01, 0.02]
# PSV: argmax=0, max=0.40, margin=0.05, reliability=0.35
# Classifier: argmax=0, max=0.93, margin=0.89
# Conformal (τ=0.35): set={0}, size=1
# → Tier 3C, T5 alert: False (rule 3)
# ---------------------------------------------------------------------------

def test_S3C_6():
    """S3C.6 — PSV reliability=0.35 fires Rule 3, trumping would-be Tier 1 → Tier 3C."""
    v3 = _make_signal([0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.89, 0.04, 0.03, 0.01, 0.01, 0.02])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.05, reliability=0.35)
    classifier = _make_classifier(argmax=0, max_val=0.93, margin=0.89)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.35)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# S3C.7 — PSV unreliable but late_blight detected (T5 still fires)
# Spec lines 4610–4617
#
# v3: probs=[0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04
# LoRA: probs=[0.06, 0.06, 0.81, 0.03, 0.02, 0.02]
# PSV: argmax=2, max=0.30, margin=0.05, reliability=0.32
# Classifier: argmax=2, max=0.88, margin=0.83
# Conformal (τ=0.40): set={2}, size=1
# → Tier 3C, T5 alert: True (rule 3 sets tier; T5 fires for late_blight argmax)
# ---------------------------------------------------------------------------

def test_S3C_7():
    """S3C.7 — PSV reliability=0.32 fires Rule 3; late_blight argmax fires T5 → Tier 3C, T5 True."""
    v3 = _make_signal([0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04)
    lora = _make_signal([0.06, 0.06, 0.81, 0.03, 0.02, 0.02])
    psv = _make_psv(argmax=2, max_val=0.30, margin=0.05, reliability=0.32)
    classifier = _make_classifier(argmax=2, max_val=0.88, margin=0.83)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# S3C.8 — PSV reliability at exactly 0.40 (NOT 3C — boundary exclusive)
# Spec lines 4619–4626
# BLK-004 Defect-15.3: subsection header implies Tier 3C but outcome is Tier 2.
# Rule 3 condition is `< 0.40`; at 0.40 exactly, Rule 3 does NOT fire.
# Rule 8 condition `>= 0.40` ✓ → Tier 2.
#
# v3: probs=[0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05
# LoRA: probs=[0.89, 0.04, 0.03, 0.02, 0.01, 0.01]
# PSV: argmax=0, max=0.45, margin=0.08, reliability=0.40 exactly
# Classifier: argmax=0, max=0.91, margin=0.86
# Conformal (τ=0.40): set={0}, size=1
# → Tier 2, T5 alert: False
# ---------------------------------------------------------------------------

def test_S3C_8():
    """S3C.8 — PSV reliability=0.40 exactly; Rule 3 strict < 0.40 fails → Tier 2, T5 False."""
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
# S3C.9 — Chilli leakage at exactly 0.40 (NOT 3C — boundary exclusive)
# Spec lines 4628–4636
# BLK-004 Defect-15.3: outcome is Tier 4A (Rule 9 catch-all).
# Rule 3 condition `> 0.40` ✗ at 0.40 exactly.
# Rule 8 condition `chilli < 0.30` ✗ at 0.40 exactly.
# Rule 7 condition `chilli < 0.20` ✗ at 0.40 exactly.
# Falls to Rule 9 catch-all → Tier 4A.
#
# v3: probs=[0.55, 0.04, 0.01, 0.00, 0.00, 0.00], chilli_leak=0.40 exactly
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.74
# Classifier: argmax=0, max=0.82, margin=0.71
# Conformal (τ=0.45): set={0}, size=1
# → Tier 4A, T5 alert: False
# ---------------------------------------------------------------------------

def test_S3C_9():
    """S3C.9 — chilli_leak=0.40 exactly; Rule 3 strict > 0.40 fails; Rules 7/8 caps also fail → Tier 4A."""
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
# S3C.10 — PSV unreliable due to mask disagreement
# Spec lines 4638–4645
#
# v3: probs=[0.81, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.04
# LoRA: probs=[0.83, 0.06, 0.05, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.50, margin=0.18, reliability=0.32
# Classifier: argmax=0, max=0.84, margin=0.78
# Conformal (τ=0.42): set={0}, size=1
# → Tier 3C, T5 alert: False (rule 3)
# ---------------------------------------------------------------------------

def test_S3C_10():
    """S3C.10 — PSV reliability=0.32 (mask disagreement) fires Rule 3 → Tier 3C."""
    v3 = _make_signal([0.81, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.04)
    lora = _make_signal([0.83, 0.06, 0.05, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.50, margin=0.18, reliability=0.32)
    classifier = _make_classifier(argmax=0, max_val=0.84, margin=0.78)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.42)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# S3C.11 — PSV unreliable due to coverage > 90%, late_blight detected (T5 fires)
# Spec lines 4647–4654
#
# v3: probs=[0.05, 0.05, 0.80, 0.02, 0.02, 0.01], chilli_leak=0.05
# LoRA: probs=[0.06, 0.05, 0.83, 0.02, 0.02, 0.02]
# PSV: argmax=2, max=0.45, margin=0.10, reliability=0.20
# Classifier: argmax=2, max=0.86, margin=0.80
# Conformal (τ=0.40): set={2}, size=1
# → Tier 3C, T5 alert: True (rule 3 sets tier; T5 fires for late_blight argmax)
# ---------------------------------------------------------------------------

def test_S3C_11():
    """S3C.11 — PSV reliability=0.20 fires Rule 3; late_blight argmax fires T5 → Tier 3C, T5 True."""
    v3 = _make_signal([0.05, 0.05, 0.80, 0.02, 0.02, 0.01], chilli_leak=0.05)
    lora = _make_signal([0.06, 0.05, 0.83, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=2, max_val=0.45, margin=0.10, reliability=0.20)
    classifier = _make_classifier(argmax=2, max_val=0.86, margin=0.80)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3C"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# S3C.12 — Chilli leakage at exact Tier 2 boundary with late_blight argmax
# Spec lines 4656–4664
# BLK-004 Defect-15.3: outcome is Tier 4A (Rule 9 catch-all), T5 True.
# chilli_leak=0.30 exactly. Rule 3 `> 0.40` ✗. Rule 8 `< 0.30` ✗ (0.30 not
# strictly less than 0.30). Rule 7 `< 0.20` ✗. Falls to Rule 9 → Tier 4A.
# T5 still fires for late_blight argmax (max=0.78 ≥ 0.20).
#
# v3: probs=[0.05, 0.05, 0.55, 0.02, 0.02, 0.01], chilli_leak=0.30
# LoRA: probs=[0.06, 0.05, 0.83, 0.02, 0.02, 0.02]
# PSV: argmax=2, max=0.55, margin=0.20, reliability=0.62
# IQA: ACCEPTABLE
# Classifier: argmax=2, max=0.78, margin=0.65
# Conformal (τ=0.50): set={2}, size=1
# → Tier 4A (Rule 9 catch-all), T5 alert: True
# ---------------------------------------------------------------------------

def test_S3C_12():
    """S3C.12 — chilli_leak=0.30 exactly; Rule 8 < 0.30 fails; Rule 9 fires; T5 True for late_blight."""
    v3 = _make_signal([0.05, 0.05, 0.55, 0.02, 0.02, 0.01], chilli_leak=0.30)
    lora = _make_signal([0.06, 0.05, 0.83, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=2, max_val=0.55, margin=0.20, reliability=0.62)
    classifier = _make_classifier(argmax=2, max_val=0.78, margin=0.65)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.50)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "catch_all_low_confidence"
