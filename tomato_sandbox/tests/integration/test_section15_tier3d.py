"""
Section 15.8 — Tier 3D scenarios (S3D.1 – S3D.10).
Spec source: tomato_3_signal_system.md lines 4667–4767.

Tier 3D occurs when IQA is DEGRADED and the classifier would otherwise route to
Tier 1 (Rule 7 main conditions met → sub-rule 7a fires) or Tier 2 (Rule 8 main
conditions met → sub-rule 8a fires).

EXCEPTIONS (BLK-004 Defect-15.3 — scenario body is authoritative):
  S3D.5: subsection header implies Tier 3D; outcome is Tier 3A (Rule 6 fires
         first because set_size==2; 3D cap never applies to Tier 3 tiers).
  S3D.7: subsection header implies Tier 3D; outcome is Tier 3B (Rule 5 fires
         first because set_size==3; IQA DEGRADED has no effect on 3B).

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
# S3D.1 — Would-be Tier 1 → 3D
# Spec lines 4671–4679
#
# v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
# LoRA: probs=[0.88, 0.05, 0.02, 0.02, 0.02, 0.01]
# PSV: argmax=0, max=0.71, margin=0.45, reliability=0.78
# IQA: DEGRADED
# Classifier: argmax=0, max=0.91, margin=0.86
# Conformal (τ=0.40): set={0}, size=1
# → Tier 3D, T5 alert: False (rule 7a; would have been Tier 1 except for IQA)
# ---------------------------------------------------------------------------

def test_S3D_1():
    """S3D.1 — DEGRADED IQA on would-be Tier 1 → sub-rule 7a fires → Tier 3D."""
    v3 = _make_signal([0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.88, 0.05, 0.02, 0.02, 0.02, 0.01])
    psv = _make_psv(argmax=0, max_val=0.71, margin=0.45, reliability=0.78)
    classifier = _make_classifier(argmax=0, max_val=0.91, margin=0.86)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# S3D.2 — Would-be Tier 2 → 3D
# Spec lines 4681–4689
#
# v3: probs=[0.71, 0.10, 0.05, 0.03, 0.05, 0.04], chilli_leak=0.02
# LoRA: probs=[0.68, 0.12, 0.07, 0.04, 0.05, 0.04]
# PSV: argmax=0, max=0.58, margin=0.28, reliability=0.65
# IQA: DEGRADED
# Classifier: argmax=0, max=0.71, margin=0.45
# Conformal (τ=0.62): set={0}, size=1
# → Tier 3D, T5 alert: False (rule 8a)
# ---------------------------------------------------------------------------

def test_S3D_2():
    """S3D.2 — DEGRADED IQA on would-be Tier 2 → sub-rule 8a fires → Tier 3D."""
    v3 = _make_signal([0.71, 0.10, 0.05, 0.03, 0.05, 0.04], chilli_leak=0.02)
    lora = _make_signal([0.68, 0.12, 0.07, 0.04, 0.05, 0.04])
    psv = _make_psv(argmax=0, max_val=0.58, margin=0.28, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.71, margin=0.45)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.62)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8a"


# ---------------------------------------------------------------------------
# S3D.3 — Tier 3D for late_blight (T5 fires)
# Spec lines 4691–4699
#
# v3: probs=[0.04, 0.04, 0.82, 0.02, 0.02, 0.02], chilli_leak=0.04
# LoRA: probs=[0.05, 0.05, 0.83, 0.02, 0.02, 0.03]
# PSV: argmax=2, max=0.65, margin=0.32, reliability=0.71
# IQA: DEGRADED
# Classifier: argmax=2, max=0.89, margin=0.83
# Conformal (τ=0.42): set={2}, size=1
# → Tier 3D, T5 alert: True (rule 7a; T5 fires independently for late_blight argmax)
# ---------------------------------------------------------------------------

def test_S3D_3():
    """S3D.3 — DEGRADED IQA + late_blight argmax → Tier 3D, T5 True."""
    v3 = _make_signal([0.04, 0.04, 0.82, 0.02, 0.02, 0.02], chilli_leak=0.04)
    lora = _make_signal([0.05, 0.05, 0.83, 0.02, 0.02, 0.03])
    psv = _make_psv(argmax=2, max_val=0.65, margin=0.32, reliability=0.71)
    classifier = _make_classifier(argmax=2, max_val=0.89, margin=0.83)
    conformal = _make_conformal(pred_set={2}, size=1, tau=0.42)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# S3D.4 — Tier 3D for YLCV (T5 fires)
# Spec lines 4701–4709
#
# v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
# LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
# PSV: argmax=3, max=0.69, margin=0.42, reliability=0.78
# IQA: DEGRADED
# Classifier: argmax=3, max=0.88, margin=0.82
# Conformal (τ=0.40): set={3}, size=1
# → Tier 3D, T5 alert: True (rule 7a; YLCV argmax → T5)
# ---------------------------------------------------------------------------

def test_S3D_4():
    """S3D.4 — DEGRADED IQA + YLCV argmax → Tier 3D, T5 True."""
    v3 = _make_signal([0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00)
    lora = _make_signal([0.05, 0.05, 0.04, 0.81, 0.02, 0.03])
    psv = _make_psv(argmax=3, max_val=0.69, margin=0.42, reliability=0.78)
    classifier = _make_classifier(argmax=3, max_val=0.88, margin=0.82)
    conformal = _make_conformal(pred_set={3}, size=1, tau=0.40)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# S3D.5 — Already Tier 3A, IQA DEGRADED — stays 3A (Rule 6 fires before 7/8)
# Spec lines 4711–4720
# BLK-004 Defect-15.3: subsection is "Tier 3D" but outcome is Tier 3A.
# Rule 6 fires on set_size==2 before Rule 7/8; 3D cap (sub-rule 7a/8a) never applies.
#
# v3: probs=[0.44, 0.39, 0.05, 0.03, 0.04, 0.03], chilli_leak=0.02
# LoRA: probs=[0.42, 0.38, 0.06, 0.04, 0.05, 0.05]
# PSV: argmax=0, max=0.51, margin=0.18, reliability=0.71
# IQA: DEGRADED
# Classifier: argmax=0, max=0.46, margin=0.04
# Conformal (τ=0.55): set={0, 1}, size=2
# → Tier 3A, T5 alert: False (rule 6)
# ---------------------------------------------------------------------------

def test_S3D_5():
    """S3D.5 — set_size==2 fires Rule 6 → Tier 3A (DEGRADED IQA has no effect on 3A)."""
    v3 = _make_signal([0.44, 0.39, 0.05, 0.03, 0.04, 0.03], chilli_leak=0.02)
    lora = _make_signal([0.42, 0.38, 0.06, 0.04, 0.05, 0.05])
    psv = _make_psv(argmax=0, max_val=0.51, margin=0.18, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.04)
    conformal = _make_conformal(pred_set={0, 1}, size=2, tau=0.55)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S3D.6 — Would-be Tier 1 with PSV reliability at boundary (0.50)
# Spec lines 4722–4730
#
# v3: probs=[0.90, 0.03, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
# LoRA: probs=[0.90, 0.04, 0.02, 0.02, 0.01, 0.01]
# PSV: argmax=0, max=0.62, margin=0.32, reliability=0.50 (Tier 1 lower bound)
# IQA: DEGRADED
# Classifier: argmax=0, max=0.92, margin=0.87
# Conformal (τ=0.40): set={0}, size=1
# → Tier 3D, T5 alert: False (rule 7a; would have been Tier 1 but DEGRADED caps)
# ---------------------------------------------------------------------------

def test_S3D_6():
    """S3D.6 — DEGRADED IQA + PSV reliability=0.50 (Tier 1 lower bound) → Tier 3D."""
    v3 = _make_signal([0.90, 0.03, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.90, 0.04, 0.02, 0.02, 0.01, 0.01])
    psv = _make_psv(argmax=0, max_val=0.62, margin=0.32, reliability=0.50)
    classifier = _make_classifier(argmax=0, max_val=0.92, margin=0.87)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# S3D.7 — DEGRADED IQA + multi-class set (Rule 5 fires first → still 3B)
# Spec lines 4732–4740
# BLK-004 Defect-15.3: subsection is "Tier 3D" but outcome is Tier 3B.
# Rule 5 fires on set_size>=3 before Rule 7/8; T5 fires for late_blight in set.
#
# v3: probs=[0.45, 0.30, 0.20, 0.02, 0.02, 0.01], chilli_leak=0.00
# LoRA: probs=[0.42, 0.32, 0.18, 0.03, 0.03, 0.02]
# PSV: argmax=0, max=0.55, margin=0.18, reliability=0.71
# IQA: DEGRADED
# Classifier: argmax=0, max=0.46, margin=0.14
# Conformal (τ=0.55): set={0, 1, 2}, size=3
# → Tier 3B, T5 alert: True (rule 5; late_blight in set with 0.20 ≥ 0.20)
# ---------------------------------------------------------------------------

def test_S3D_7():
    """S3D.7 — set_size==3 fires Rule 5 → Tier 3B; DEGRADED IQA has no effect; T5 True."""
    v3 = _make_signal([0.45, 0.30, 0.20, 0.02, 0.02, 0.01], chilli_leak=0.00)
    lora = _make_signal([0.42, 0.32, 0.18, 0.03, 0.03, 0.02])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.18, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.14)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.55)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3D.8 — Would-be Tier 1 healthy → 3D
# Spec lines 4742–4750
#
# v3: probs=[0.02, 0.02, 0.01, 0.02, 0.01, 0.90], chilli_leak=0.02
# LoRA: probs=[0.03, 0.03, 0.02, 0.02, 0.02, 0.88]
# PSV: argmax=5, max=0.79, margin=0.55, reliability=0.83
# IQA: DEGRADED
# Classifier: argmax=5, max=0.92, margin=0.88
# Conformal (τ=0.40): set={5}, size=1
# → Tier 3D, T5 alert: False (rule 7a; healthy argmax doesn't trigger T5)
# ---------------------------------------------------------------------------

def test_S3D_8():
    """S3D.8 — DEGRADED IQA + healthy argmax → Tier 3D, T5 False."""
    v3 = _make_signal([0.02, 0.02, 0.01, 0.02, 0.01, 0.90], chilli_leak=0.02)
    lora = _make_signal([0.03, 0.03, 0.02, 0.02, 0.02, 0.88])
    psv = _make_psv(argmax=5, max_val=0.79, margin=0.55, reliability=0.83)
    classifier = _make_classifier(argmax=5, max_val=0.92, margin=0.88)
    conformal = _make_conformal(pred_set={5}, size=1, tau=0.40)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# S3D.9 — IQA score 0.55 → DEGRADED threshold (same inputs as S3D.1)
# Spec lines 4752–4756
# IQA aggregate_score=0.55 falls in DEGRADED=[0.40, 0.65).
# v3, LoRA, PSV, Classifier: as in S3D.1
# → Tier 3D, T5 alert: False (rule 7a)
# ---------------------------------------------------------------------------

def test_S3D_9():
    """S3D.9 — IQA score=0.55 → DEGRADED; same inputs as S3D.1 → Tier 3D."""
    v3 = _make_signal([0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03)
    lora = _make_signal([0.88, 0.05, 0.02, 0.02, 0.02, 0.01])
    psv = _make_psv(argmax=0, max_val=0.71, margin=0.45, reliability=0.78)
    classifier = _make_classifier(argmax=0, max_val=0.91, margin=0.86)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    # IQA score 0.55 maps to DEGRADED per Section 6.4 thresholds
    iqa = {"decision": "DEGRADED", "aggregate_score": 0.55}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# S3D.10 — DEGRADED IQA + underpowered class (sub-rule 7a wins over 7b → Tier 3D not 3A)
# Spec lines 4758–4767
#
# v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
# LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
# PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
# IQA: DEGRADED
# Classifier: argmax=3 (YLCV, underpowered), max=0.88, margin=0.82
# Conformal (τ=0.40): set={3}, size=1
# underpowered_classes={3} (YLCV recall < 0.50)
# → Tier 3D, T5 alert: True (rule 7a wins over 7b; YLCV argmax → T5)
# ---------------------------------------------------------------------------

def test_S3D_10():
    """S3D.10 — DEGRADED IQA + YLCV underpowered; sub-rule 7a wins → Tier 3D, T5 True."""
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
