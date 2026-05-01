"""
Section 15.6 — Tier 3B scenarios (S3B.1 – S3B.10).
Spec source: tomato_3_signal_system.md lines 4452–4549.

All Tier 3B scenarios share: prediction_set_size >= 3 AND combined_max_prob >= 0.45.
Rule 5 fires.

EXCEPTION: S3B.4 — degenerate flat distribution routes to Tier 4A (Rule 4).
  Subsection header says "Tier 3B" but scenario body outcome is Tier 4A.
  Encoder uses scenario body as authoritative per Convention 1 / BLK-004 Defect-15.3.

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
# S3B.1 — Three small-lesion classes, late_blight in set fires T5
# Spec lines 4456–4464
#
# v3: probs=[0.45, 0.30, 0.20, 0.02, 0.02, 0.01], chilli_leak=0.00
# LoRA: probs=[0.42, 0.32, 0.18, 0.03, 0.03, 0.02]
# PSV: argmax=0, max=0.55, margin=0.18, reliability=0.71
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.46, margin=0.16
# Conformal (τ=0.55): set={0, 1, 2}, size=3
# → Tier 3B, T5 alert: True (rule 5; late_blight in set with prob 0.20 >= 0.20)
# ---------------------------------------------------------------------------

def test_scenario_S3B_1():
    """S3B.1 — Three classes, late_blight in set fires T5. Spec lines 4456-4464."""
    v3 = _make_signal([0.45, 0.30, 0.20, 0.02, 0.02, 0.01], chilli_leak=0.00)
    lora = _make_signal([0.42, 0.32, 0.18, 0.03, 0.03, 0.02])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.18, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.16)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.2 — Four-class spread including healthy
# Spec lines 4466–4474
#
# v3: probs=[0.33, 0.29, 0.06, 0.04, 0.04, 0.21], chilli_leak=0.03
# LoRA: probs=[0.30, 0.30, 0.07, 0.05, 0.05, 0.23]
# PSV: argmax=0, max=0.41, margin=0.05, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: P_final_calibrated=[0.34, 0.30, 0.06, 0.04, 0.04, 0.22, 0.00],
#   argmax=0, max=0.46, margin=0.16
# Conformal (τ=0.78): threshold 0.22; set={0, 1, 5}, size=3
# → Tier 3B, T5 alert: False (rule 5; no dangerous class in set or argmax)
# ---------------------------------------------------------------------------

def test_scenario_S3B_2():
    """S3B.2 — Four-class spread including healthy (no T5). Spec lines 4466-4474."""
    v3 = _make_signal([0.33, 0.29, 0.06, 0.04, 0.04, 0.21], chilli_leak=0.03)
    lora = _make_signal([0.30, 0.30, 0.07, 0.05, 0.05, 0.23])
    psv = _make_psv(argmax=0, max_val=0.41, margin=0.05, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.16)
    conformal = _make_conformal(pred_set={0, 1, 5}, size=3, tau=0.78)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.3 — Three-class set with late_blight admitted (genuine T5 case)
# Spec lines 4476–4485
#
# v3: probs=[0.30, 0.22, 0.20, 0.10, 0.10, 0.05], chilli_leak=0.03
# LoRA: probs=[0.28, 0.22, 0.20, 0.12, 0.12, 0.06]
# PSV: argmax=0, max=0.46, margin=0.10, reliability=0.58
# IQA: ACCEPTABLE
# Classifier: P_final_calibrated=[0.46, 0.22, 0.20, 0.05, 0.04, 0.02, 0.01],
#   argmax=0, max=0.46, margin=0.24
# Conformal (τ=0.83): threshold 0.17; set={0, 1, 2}, size=3
# → Tier 3B, T5 alert: True (rule 5; late_blight in set with prob 0.20 >= 0.20)
# ---------------------------------------------------------------------------

def test_scenario_S3B_3():
    """S3B.3 — Three-class set with late_blight in set (T5). Spec lines 4476-4485."""
    v3 = _make_signal([0.30, 0.22, 0.20, 0.10, 0.10, 0.05], chilli_leak=0.03)
    lora = _make_signal([0.28, 0.22, 0.20, 0.12, 0.12, 0.06])
    psv = _make_psv(argmax=0, max_val=0.46, margin=0.10, reliability=0.58)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.24)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.83)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.4 — Extreme uncertainty (degenerate case routes to 4A, not 3B)
# Spec lines 4487–4493
#
# Classifier: P_final_calibrated=[0.16, 0.15, 0.15, 0.14, 0.14, 0.13, 0.13],
#   argmax=0, max=0.16
# → Tier 4A (Rule 4: max 0.16 < 0.45), T5 alert: False
#
# NOTE: S3B.4 subsection header says "Tier 3B" but scenario body outcome is Tier 4A.
# Scenario body is authoritative per spec Section 15.2 Convention 1 / BLK-004 Defect-15.3.
# ---------------------------------------------------------------------------

def test_scenario_S3B_4():
    """S3B.4 — Degenerate flat distribution routes to Tier 4A (not 3B). Spec lines 4487-4493."""
    # Spec only gives classifier output; using minimal valid signals that won't trigger Rule 1/3
    v3 = _make_signal([0.17, 0.16, 0.16, 0.15, 0.15, 0.14], chilli_leak=0.07)
    lora = _make_signal([0.17, 0.16, 0.16, 0.15, 0.15, 0.21])
    psv = _make_psv(argmax=0, max_val=0.20, margin=0.01, reliability=0.55)
    # Verbatim classifier from spec
    classifier = _make_classifier(argmax=0, max_val=0.16, margin=0.01)
    conformal = _make_conformal(pred_set={0, 1, 2, 3, 4, 5}, size=6, tau=0.85)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S3B.5 — Three classes including late_blight argmax (T5 fires via both bullets)
# Spec lines 4495–4503
#
# v3: probs=[0.20, 0.05, 0.45, 0.05, 0.05, 0.05], chilli_leak=0.15
# LoRA: probs=[0.22, 0.06, 0.43, 0.05, 0.06, 0.18]
# PSV: argmax=2, max=0.55, margin=0.20, reliability=0.66
# IQA: ACCEPTABLE
# Classifier: P_final_calibrated=[0.20, 0.05, 0.45, 0.05, 0.05, 0.15, 0.05],
#   argmax=2, max=0.45, margin=0.25
# Conformal (τ=0.85): threshold 0.15; set={0, 2, 5}, size=3
# → Tier 3B, T5 alert: True (rule 5; late_blight argmax + late_blight in set both fire)
# Walk: max=0.45 — Rule 4 `< 0.45` strict fails; Rule 5 fires
# ---------------------------------------------------------------------------

def test_scenario_S3B_5():
    """S3B.5 — Three classes, late_blight argmax (T5 fires). Spec lines 4495-4503."""
    v3 = _make_signal([0.20, 0.05, 0.45, 0.05, 0.05, 0.05], chilli_leak=0.15)
    lora = _make_signal([0.22, 0.06, 0.43, 0.05, 0.06, 0.18])
    psv = _make_psv(argmax=2, max_val=0.55, margin=0.20, reliability=0.66)
    classifier = _make_classifier(argmax=2, max_val=0.45, margin=0.25)
    conformal = _make_conformal(pred_set={0, 2, 5}, size=3, tau=0.85)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.6 — Three classes with mosaic argmax + YLCV in set (only mosaic argmax fires T5)
# Spec lines 4505–4513
#
# v3: probs=[0.20, 0.05, 0.05, 0.20, 0.45, 0.05], chilli_leak=0.00
# LoRA: probs=[0.22, 0.06, 0.05, 0.20, 0.42, 0.05]
# PSV: argmax=4, max=0.51, margin=0.15, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: P_final=[0.20, 0.05, 0.05, 0.20, 0.45, 0.04, 0.01],
#   argmax=4, max=0.45, margin=0.25
# Conformal (τ=0.85): threshold 0.15; set={0, 3, 4}, size=3
# → Tier 3B, T5 alert: True (rule 5; mosaic argmax fires T5; YLCV in set does NOT)
# ---------------------------------------------------------------------------

def test_scenario_S3B_6():
    """S3B.6 — Mosaic argmax, YLCV in set (T5 via mosaic argmax only). Spec lines 4505-4513."""
    v3 = _make_signal([0.20, 0.05, 0.05, 0.20, 0.45, 0.05], chilli_leak=0.00)
    lora = _make_signal([0.22, 0.06, 0.05, 0.20, 0.42, 0.05])
    psv = _make_psv(argmax=4, max_val=0.51, margin=0.15, reliability=0.65)
    classifier = _make_classifier(argmax=4, max_val=0.45, margin=0.25)
    conformal = _make_conformal(pred_set={0, 3, 4}, size=3, tau=0.85)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.7 — Three small-lesion classes with foliar argmax, late_blight in set fires T5
# Spec lines 4515–4523
#
# v3: probs=[0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01
# LoRA: probs=[0.46, 0.21, 0.18, 0.04, 0.06, 0.05]
# PSV: argmax=0, max=0.55, margin=0.20, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: P_final=[0.46, 0.20, 0.20, 0.04, 0.05, 0.04, 0.01],
#   argmax=0, max=0.46, margin=0.26
# Conformal (τ=0.81): threshold 0.19; set={0, 1, 2}, size=3
# → Tier 3B, T5 alert: True (rule 5; late_blight in set with prob 0.20 >= 0.20)
# ---------------------------------------------------------------------------

def test_scenario_S3B_7():
    """S3B.7 — Foliar argmax, late_blight in set fires T5. Spec lines 4515-4523."""
    v3 = _make_signal([0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01)
    lora = _make_signal([0.46, 0.21, 0.18, 0.04, 0.06, 0.05])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.20, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.26)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.81)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.8 — Three classes with healthy argmax, no T5
# Spec lines 4525–4533
#
# v3: probs=[0.20, 0.18, 0.04, 0.05, 0.04, 0.45], chilli_leak=0.04
# LoRA: probs=[0.22, 0.20, 0.05, 0.05, 0.04, 0.44]
# PSV: argmax=5, max=0.51, margin=0.18, reliability=0.69
# IQA: ACCEPTABLE
# Classifier: P_final=[0.20, 0.18, 0.04, 0.05, 0.04, 0.45, 0.04],
#   argmax=5, max=0.45, margin=0.25
# Conformal (τ=0.83): threshold 0.17; set={0, 1, 5}, size=3
# → Tier 3B, T5 alert: False (rule 5; no dangerous class in set or argmax)
# ---------------------------------------------------------------------------

def test_scenario_S3B_8():
    """S3B.8 — Healthy argmax, three classes, no T5. Spec lines 4525-4533."""
    v3 = _make_signal([0.20, 0.18, 0.04, 0.05, 0.04, 0.45], chilli_leak=0.04)
    lora = _make_signal([0.22, 0.20, 0.05, 0.05, 0.04, 0.44])
    psv = _make_psv(argmax=5, max_val=0.51, margin=0.18, reliability=0.69)
    classifier = _make_classifier(argmax=5, max_val=0.45, margin=0.25)
    conformal = _make_conformal(pred_set={0, 1, 5}, size=3, tau=0.83)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.9 — Three classes with septoria argmax + IQA DEGRADED (3B sticks; 3D doesn't apply)
# Spec lines 4535–4543
#
# v3: probs=[0.20, 0.45, 0.20, 0.04, 0.04, 0.04], chilli_leak=0.03
# LoRA: probs=[0.22, 0.46, 0.18, 0.04, 0.05, 0.05]
# PSV: argmax=1, max=0.50, margin=0.20, reliability=0.55
# IQA: DEGRADED
# Classifier: P_final=[0.20, 0.46, 0.20, 0.04, 0.04, 0.04, 0.02],
#   argmax=1, max=0.46, margin=0.26
# Conformal (τ=0.83): threshold 0.17; set={0, 1, 2}, size=3
# → Tier 3B, T5 alert: True (rule 5 fires before rule 7/8; late_blight in set with 0.20 >= 0.20)
# Note: DEGRADED IQA cap (3D) only applies when Rule 7/8 fires; Rule 5 fires first
# ---------------------------------------------------------------------------

def test_scenario_S3B_9():
    """S3B.9 — Septoria argmax + IQA DEGRADED (3B sticks; T5 fires). Spec lines 4535-4543."""
    v3 = _make_signal([0.20, 0.45, 0.20, 0.04, 0.04, 0.04], chilli_leak=0.03)
    lora = _make_signal([0.22, 0.46, 0.18, 0.04, 0.05, 0.05])
    psv = _make_psv(argmax=1, max_val=0.50, margin=0.20, reliability=0.55)
    classifier = _make_classifier(argmax=1, max_val=0.46, margin=0.26)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.83)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S3B.10 — Borderline: τ admits exactly 3 classes
# Spec lines 4545–4549
#
# Classifier: P_final=[0.50, 0.30, 0.13, 0.03, 0.02, 0.01, 0.01],
#   argmax=0, max=0.50, margin=0.20
# Conformal (τ=0.87): threshold 0.13; set={0, 1, 2}, size=3
# → Tier 3B, T5 alert: False (rule 5; late_blight prob 0.13 < 0.20 fails T5)
# ---------------------------------------------------------------------------

def test_scenario_S3B_10():
    """S3B.10 — Borderline: τ admits exactly 3 classes. Spec lines 4545-4549."""
    # Spec provides only classifier/conformal; use minimal valid signals
    v3 = _make_signal([0.50, 0.30, 0.10, 0.03, 0.02, 0.02], chilli_leak=0.03)
    lora = _make_signal([0.51, 0.29, 0.11, 0.03, 0.03, 0.03])
    psv = _make_psv(argmax=0, max_val=0.60, margin=0.20, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.50, margin=0.20)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.87)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3,
        lora_signal=lora,
        psv_signal=psv,
        classifier=classifier,
        conformal=conformal,
        iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "5"
