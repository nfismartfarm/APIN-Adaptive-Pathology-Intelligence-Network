"""
Section 15.11 — Tier 5 alert combinations beyond prior coverage (S5.1 – S5.11).
Spec source: tomato_3_signal_system.md lines 4997–5113.

These scenarios primarily test that T5 is computed independently of the base tier.
T5 fires when:
  (argmax ∈ {late_blight=2, mosaic=4, ylcv=3} AND max >= 0.20)
  OR (late_blight in set AND late_blight_prob >= 0.20).
Mosaic and YLCV have argmax-only T5 triggers (no in-set trigger).

T5 True:  S5.1, S5.2, S5.3, S5.4, S5.5, S5.6, S5.8, S5.10, S5.11
T5 False: S5.7 (mosaic max=0.19 < 0.20), S5.9 (late_blight in set at 0.19 < 0.20)

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
# S5.1 — Tier 3B with late_blight in set (T5 fires)
# Spec lines 5001–5009
#
# v3: probs=[0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01
# LoRA: probs=[0.46, 0.21, 0.18, 0.04, 0.06, 0.05]
# PSV: argmax=0, max=0.55, margin=0.20, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: P_final=[0.46, 0.20, 0.20, 0.04, 0.05, 0.04, 0.01], argmax=0, max=0.46, margin=0.26
# Conformal (tau=0.81): set={0, 1, 2}, size=3
# → Tier 3B, T5 alert: True (rule 5; late_blight in set with prob 0.20 >= 0.20)
# ---------------------------------------------------------------------------

def test_S5_1():
    """S5.1 — Tier 3B (set_size>=3); late_blight in set at 0.20 fires T5."""
    v3 = _make_signal([0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01)
    lora = _make_signal([0.46, 0.21, 0.18, 0.04, 0.06, 0.05])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.20, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.46, margin=0.26)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.81)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "5"


# ---------------------------------------------------------------------------
# S5.2 — Tier 4A with late_blight at exactly 0.20 in set (T5 fires)
# Spec lines 5011–5019
#
# v3: probs=[0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02
# LoRA: probs=[0.32, 0.20, 0.20, 0.10, 0.10, 0.08]
# PSV: argmax=0, max=0.42, margin=0.12, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: P_final=[0.30, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05], argmax=0, max=0.30, margin=0.10
# Conformal (tau=0.83): set={0, 1, 2}, size=3
# → Tier 4A (Rule 4: max 0.30 < 0.45), T5 alert: True (late_blight in set at 0.20 >= 0.20)
# ---------------------------------------------------------------------------

def test_S5_2():
    """S5.2 — Tier 4A; late_blight in set at 0.20 exactly (boundary inclusive) fires T5."""
    v3 = _make_signal([0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02)
    lora = _make_signal([0.32, 0.20, 0.20, 0.10, 0.10, 0.08])
    psv = _make_psv(argmax=0, max_val=0.42, margin=0.12, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.30, margin=0.10)
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
# S5.3 — Tier 3C with late_blight argmax (T5 fires despite Rule 3)
# Spec lines 5021–5029
#
# v3: probs=[0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04
# LoRA: probs=[0.06, 0.06, 0.81, 0.02, 0.02, 0.03]
# PSV: argmax=2, max=0.45, margin=0.10, reliability=0.20
# IQA: ACCEPTABLE
# Classifier: argmax=2, max=0.86, margin=0.80
# Conformal (tau=0.40): set={2}, size=1
# → Tier 3C, T5 alert: True (rule 3 sets tier; T5 fires for late_blight argmax)
# ---------------------------------------------------------------------------

def test_S5_3():
    """S5.3 — Tier 3C (PSV reliability=0.20); late_blight argmax fires T5 independently."""
    v3 = _make_signal([0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04)
    lora = _make_signal([0.06, 0.06, 0.81, 0.02, 0.02, 0.03])
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
# S5.4 — Tier 4B with late_blight argmax (T5 fires despite Rule 1)
# Spec lines 5031–5039
#
# v3: failed (CUDA OOM); succeeded=False
# LoRA: probs=[0.05, 0.05, 0.81, 0.03, 0.03, 0.03]
# PSV: argmax=2, max=0.62, margin=0.30, reliability=0.71
# IQA: ACCEPTABLE
# Classifier (v3 zeroed): argmax=2, max=0.71, margin=0.55
# Conformal: set={2}, size=1
# → Tier 4B, T5 alert: True (rule 1 sets tier; T5 fires for late_blight argmax)
# ---------------------------------------------------------------------------

def test_S5_4():
    """S5.4 — Tier 4B (v3 failed); late_blight argmax fires T5 even with degraded pipeline."""
    v3 = _make_signal_failed()
    lora = _make_signal([0.05, 0.05, 0.81, 0.03, 0.03, 0.03])
    psv = _make_psv(argmax=2, max_val=0.62, margin=0.30, reliability=0.71)
    classifier = _make_classifier(argmax=2, max_val=0.71, margin=0.55)
    conformal = _make_conformal(pred_set={2}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# S5.5 — Tier 3A with late_blight in set (T5 fires)
# Spec lines 5041–5049
#
# v3: probs=[0.05, 0.45, 0.40, 0.03, 0.03, 0.03], chilli_leak=0.01
# LoRA: probs=[0.06, 0.43, 0.38, 0.04, 0.04, 0.05]
# PSV: argmax=1, max=0.50, margin=0.15, reliability=0.74
# IQA: ACCEPTABLE
# Classifier: argmax=1, max=0.45, margin=0.05
# Conformal (tau=0.55): set={1, 2}, size=2
# → Tier 3A, T5 alert: True (rule 6; late_blight in set with prob 0.40 >= 0.20)
# ---------------------------------------------------------------------------

def test_S5_5():
    """S5.5 — Tier 3A (set_size==2); late_blight in set at 0.40 fires T5."""
    v3 = _make_signal([0.05, 0.45, 0.40, 0.03, 0.03, 0.03], chilli_leak=0.01)
    lora = _make_signal([0.06, 0.43, 0.38, 0.04, 0.04, 0.05])
    psv = _make_psv(argmax=1, max_val=0.50, margin=0.15, reliability=0.74)
    classifier = _make_classifier(argmax=1, max_val=0.45, margin=0.05)
    conformal = _make_conformal(pred_set={1, 2}, size=2, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# S5.6 — Tier 3D with mosaic argmax (T5 fires)
# Spec lines 5051–5059
#
# v3: probs=[0.04, 0.03, 0.02, 0.02, 0.85, 0.02], chilli_leak=0.02
# LoRA: probs=[0.05, 0.03, 0.02, 0.02, 0.84, 0.04]
# PSV: argmax=4, max=0.69, margin=0.42, reliability=0.71
# IQA: DEGRADED
# Classifier: argmax=4 (mosaic), max=0.88, margin=0.81
# Conformal (tau=0.43): set={4}, size=1
# → Tier 3D, T5 alert: True (rule 7a; mosaic argmax with max 0.88 >= 0.20)
# ---------------------------------------------------------------------------

def test_S5_6():
    """S5.6 — Tier 3D (DEGRADED IQA); mosaic argmax fires T5 independently."""
    v3 = _make_signal([0.04, 0.03, 0.02, 0.02, 0.85, 0.02], chilli_leak=0.02)
    lora = _make_signal([0.05, 0.03, 0.02, 0.02, 0.84, 0.04])
    psv = _make_psv(argmax=4, max_val=0.69, margin=0.42, reliability=0.71)
    classifier = _make_classifier(argmax=4, max_val=0.88, margin=0.81)
    conformal = _make_conformal(pred_set={4}, size=1, tau=0.43)
    iqa = {"decision": "DEGRADED"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "3D"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "7a"


# ---------------------------------------------------------------------------
# S5.7 — Mosaic at exactly 0.19 (T5 boundary; does NOT fire)
# Spec lines 5061–5070
#
# v3: probs=[0.20, 0.18, 0.06, 0.06, 0.20, 0.20], chilli_leak=0.10
# LoRA: probs=[0.18, 0.17, 0.05, 0.05, 0.20, 0.35]
# PSV: argmax=4 (mosaic), max=0.30, margin=0.05, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: P_final=[0.18, 0.17, 0.05, 0.05, 0.19, 0.17, 0.19], argmax=4 (mosaic), max=0.19, margin=0.00
# Conformal (tau=0.85): set={0, 1, 4, 5, 6}, size=5
# → Tier 4A (Rule 4: max 0.19 < 0.45), T5 alert: False (mosaic max 0.19 < 0.20)
# ---------------------------------------------------------------------------

def test_S5_7():
    """S5.7 — Mosaic argmax but max=0.19 < 0.20; T5 boundary strict ≥ 0.20 → T5 False."""
    v3 = _make_signal([0.20, 0.18, 0.06, 0.06, 0.20, 0.20], chilli_leak=0.10)
    lora = _make_signal([0.18, 0.17, 0.05, 0.05, 0.20, 0.35])
    psv = _make_psv(argmax=4, max_val=0.30, margin=0.05, reliability=0.55)
    classifier = _make_classifier(argmax=4, max_val=0.19, margin=0.00)
    conformal = _make_conformal(pred_set={0, 1, 4, 5, 6}, size=5, tau=0.85)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S5.8 — Late_blight at exactly 0.20 in set (in-set T5 boundary fires)
# Spec lines 5072–5080
#
# v3: probs=[0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02
# LoRA: probs=[0.32, 0.22, 0.20, 0.10, 0.10, 0.06]
# PSV: argmax=0, max=0.45, margin=0.10, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: P_final=[0.30, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05], argmax=0, max=0.30, margin=0.10
# Conformal (tau=0.83): set={0, 1, 2}, size=3
# → Tier 4A (Rule 4: max 0.30 < 0.45), T5 alert: True (late_blight at 0.20 satisfies >= 0.20)
# ---------------------------------------------------------------------------

def test_S5_8():
    """S5.8 — Late_blight prob=0.20 in set (boundary inclusive >= 0.20) fires T5; Tier 4A."""
    v3 = _make_signal([0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02)
    lora = _make_signal([0.32, 0.22, 0.20, 0.10, 0.10, 0.06])
    psv = _make_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.30, margin=0.10)
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
# S5.9 — Late_blight at 0.19 in set (T5 in-set bullet does NOT fire)
# Spec lines 5082–5091 (paired with S5.8)
#
# v3: probs=[0.30, 0.20, 0.18, 0.11, 0.10, 0.09], chilli_leak=0.02
# LoRA: probs=[0.32, 0.22, 0.19, 0.10, 0.10, 0.07]
# PSV: argmax=0, max=0.45, margin=0.10, reliability=0.65
# IQA: ACCEPTABLE
# Classifier: P_final=[0.30, 0.21, 0.19, 0.11, 0.09, 0.05, 0.05], argmax=0, max=0.30, margin=0.09
# Conformal (tau=0.83): set={0, 1, 2}, size=3
# → Tier 4A, T5 alert: False (late_blight in set at 0.19 < 0.20 strict)
# ---------------------------------------------------------------------------

def test_S5_9():
    """S5.9 — Late_blight prob=0.19 in set; strict >= 0.20 fails → T5 False; Tier 4A."""
    v3 = _make_signal([0.30, 0.20, 0.18, 0.11, 0.10, 0.09], chilli_leak=0.02)
    lora = _make_signal([0.32, 0.22, 0.19, 0.10, 0.10, 0.07])
    psv = _make_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.30, margin=0.09)
    conformal = _make_conformal(pred_set={0, 1, 2}, size=3, tau=0.83)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S5.10 — Three dangerous classes in set with foliar argmax (T5 fires via late_blight in-set only)
# Spec lines 5093–5102
#
# v3: probs=[0.30, 0.05, 0.20, 0.18, 0.18, 0.04], chilli_leak=0.05
# LoRA: probs=[0.32, 0.06, 0.20, 0.18, 0.18, 0.06]
# PSV: argmax=0, max=0.45, margin=0.10, reliability=0.55
# IQA: ACCEPTABLE
# Classifier: P_final=[0.30, 0.05, 0.20, 0.18, 0.18, 0.05, 0.04], argmax=0 (foliar), max=0.30, margin=0.10
# Conformal (tau=0.83): set={0, 2, 3, 4}, size=4 (all three dangerous classes admitted)
# → Tier 4A (Rule 4: max 0.30 < 0.45), T5 alert: True (late_blight in set at 0.20 >= 0.20)
# Note: mosaic and YLCV in set do NOT trigger T5 (argmax-only triggers for those two)
# ---------------------------------------------------------------------------

def test_S5_10():
    """S5.10 — Three dangerous classes in set; only late_blight in-set fires T5; mosaic/YLCV in set does not."""
    v3 = _make_signal([0.30, 0.05, 0.20, 0.18, 0.18, 0.04], chilli_leak=0.05)
    lora = _make_signal([0.32, 0.06, 0.20, 0.18, 0.18, 0.06])
    psv = _make_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.30, margin=0.10)
    conformal = _make_conformal(pred_set={0, 2, 3, 4}, size=4, tau=0.83)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# S5.11 — Tier 4B with late_blight in set but not argmax (T5 fires via in-set rule)
# Spec lines 5104–5113
#
# v3: failed; succeeded=False
# LoRA: probs=[0.50, 0.05, 0.25, 0.05, 0.05, 0.10]
# PSV: argmax=0, max=0.55, margin=0.20, reliability=0.65
# IQA: ACCEPTABLE
# Classifier (v3 zeroed): P_final=[0.45, 0.05, 0.25, 0.05, 0.05, 0.10, 0.05], argmax=0, max=0.45, margin=0.20
# Conformal (tau=0.80): set={0, 2}, size=2 (foliar at 0.45 >= 0.20; late_blight at 0.25 >= 0.20)
# → Tier 4B (Rule 1: v3 failed), T5 alert: True (late_blight in set with prob 0.25 >= 0.20)
# ---------------------------------------------------------------------------

def test_S5_11():
    """S5.11 — Tier 4B (v3 failed); late_blight in set at 0.25 fires T5 despite degraded pipeline."""
    v3 = _make_signal_failed()
    lora = _make_signal([0.50, 0.05, 0.25, 0.05, 0.05, 0.10])
    psv = _make_psv(argmax=0, max_val=0.55, margin=0.20, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.45, margin=0.20)
    conformal = _make_conformal(pred_set={0, 2}, size=2, tau=0.80)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is True
    assert result.rule_id_fired == "1"
