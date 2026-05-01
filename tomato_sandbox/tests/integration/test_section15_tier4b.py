"""
Section 15.10 — Tier 4B scenarios (S4B.1 – S4B.10).
Spec source: tomato_3_signal_system.md lines 4902–4995.

All Tier 4B scenarios share: at least one signal has forward_succeeded == False.
Rule 1 fires immediately → Tier 4B.  T5 is evaluated independently.

T5 True: S4B.5 (late_blight argmax), S4B.6 (late_blight argmax), S4B.9 (late_blight argmax).

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
    """Returns a signal dict representing a failed forward pass."""
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
# S4B.1 — v3 failed (CUDA OOM)
# Spec lines 4906–4914
#
# v3: failed (CUDA OOM); probs all 0.0; succeeded=False
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.78, margin=0.65
# Conformal: set={0}, size=1
# → Tier 4B, T5 alert: False (rule 1)
# ---------------------------------------------------------------------------

def test_S4B_1():
    """S4B.1 — v3 CUDA OOM (succeeded=False) → Rule 1 → Tier 4B."""
    v3 = _make_signal_failed()
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.78, margin=0.65)
    conformal = _make_conformal(pred_set={0}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# S4B.2 — LoRA failed (NaN in forward)
# Spec lines 4916–4924
#
# v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: failed (NaN propagation); probs all 0.0; succeeded=False
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.74, margin=0.55
# Conformal: set={0}, size=1
# → Tier 4B, T5 alert: False (rule 1)
# ---------------------------------------------------------------------------

def test_S4B_2():
    """S4B.2 — LoRA NaN (succeeded=False) → Rule 1 → Tier 4B."""
    v3 = _make_signal([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal_failed()
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.74, margin=0.55)
    conformal = _make_conformal(pred_set={0}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# S4B.3 — PSV failed (segmentation crash)
# Spec lines 4926–4934
#
# v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: probs=[0.83, 0.06, 0.04, 0.02, 0.02, 0.03]
# PSV: failed (exception in disease detection); succeeded=False; reliability=0.05
# IQA: ACCEPTABLE
# Classifier: argmax=0, max=0.80, margin=0.65
# Conformal: set={0}, size=1
# → Tier 4B, T5 alert: False (rule 1)
# ---------------------------------------------------------------------------

def test_S4B_3():
    """S4B.3 — PSV segmentation crash (succeeded=False) → Rule 1 → Tier 4B."""
    v3 = _make_signal([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal([0.83, 0.06, 0.04, 0.02, 0.02, 0.03])
    psv = _make_psv(argmax=0, max_val=1/6, margin=0.0, reliability=0.05, succeeded=False)
    classifier = _make_classifier(argmax=0, max_val=0.80, margin=0.65)
    conformal = _make_conformal(pred_set={0}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# S4B.4 — v3 + LoRA both failed (PSV alone drives output)
# Spec lines 4936–4944
#
# v3: failed; probs all 0.0; succeeded=False
# LoRA: failed; probs all 0.0; succeeded=False
# PSV: argmax=0 (foliar), max=0.65, margin=0.30, reliability=0.55, succeeded=True
# IQA: ACCEPTABLE
# Classifier (PSV alone via degraded-mode): argmax=0, max=0.50, margin=0.25
# Conformal: set={0}, size=1
# → Tier 4B, T5 alert: False (rule 1; PSV drove foliar argmax, not dangerous)
# ---------------------------------------------------------------------------

def test_S4B_4():
    """S4B.4 — v3 + LoRA both failed; PSV alone drives classifier → Rule 1 → Tier 4B."""
    v3 = _make_signal_failed()
    lora = _make_signal_failed()
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.50, margin=0.25)
    conformal = _make_conformal(pred_set={0}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# S4B.5 — v3 failed but late_blight detected by LoRA + PSV (T5 fires)
# Spec lines 4946–4955
#
# v3: failed; succeeded=False
# LoRA: probs=[0.05, 0.05, 0.81, 0.03, 0.03, 0.03]
# PSV: argmax=2, max=0.62, margin=0.30, reliability=0.71
# IQA: ACCEPTABLE
# Classifier (v3 zeroed): argmax=2, max=0.71, margin=0.55
# Conformal: set={2}, size=1
# → Tier 4B, T5 alert: True (rule 1 sets tier; T5 fires for late_blight argmax)
# ---------------------------------------------------------------------------

def test_S4B_5():
    """S4B.5 — v3 failed; LoRA+PSV detect late_blight → Rule 1 → Tier 4B; T5 True."""
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
# S4B.6 — LoRA failed but T5 fires for late_blight
# Spec lines 4957–4963
#
# v3: probs=[0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04
# LoRA: failed; succeeded=False
# PSV: argmax=2, max=0.62, margin=0.30, reliability=0.71
# Classifier (LoRA zeroed): argmax=2, max=0.74, margin=0.60
# → Tier 4B, T5 alert: True (rule 1; late_blight argmax)
# ---------------------------------------------------------------------------

def test_S4B_6():
    """S4B.6 — LoRA failed; late_blight drives classifier → Rule 1 → Tier 4B; T5 True."""
    v3 = _make_signal([0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04)
    lora = _make_signal_failed()
    psv = _make_psv(argmax=2, max_val=0.62, margin=0.30, reliability=0.71)
    classifier = _make_classifier(argmax=2, max_val=0.74, margin=0.60)
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
# S4B.7 — All 5 TTA views failed for v3 (effectively v3 failure)
# Spec lines 4965–4970
#
# v3: all 5 TTA views succeeded=False; aggregated: zero probs; v3.forward_succeeded=False
# LoRA: 5/5 views succeeded; probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
# PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
# → Tier 4B, T5 alert: False (rule 1; v3 failed across all views)
# ---------------------------------------------------------------------------

def test_S4B_7():
    """S4B.7 — All TTA views of v3 failed → v3.forward_succeeded=False → Rule 1 → Tier 4B."""
    v3 = _make_signal_failed()  # aggregated from all-failed TTA views
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.71)
    classifier = _make_classifier(argmax=0, max_val=0.78, margin=0.65)
    conformal = _make_conformal(pred_set={0}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# S4B.8 — PSV mid-feature-computation failure (GLCM crash)
# Spec lines 4972–4977
#
# v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
# LoRA: probs=[0.83, 0.06, 0.04, 0.02, 0.02, 0.03]
# PSV: GLCM computation exception; succeeded=False; reliability=0.05
# → Tier 4B, T5 alert: False (rule 1)
# ---------------------------------------------------------------------------

def test_S4B_8():
    """S4B.8 — PSV GLCM crash (succeeded=False) → Rule 1 → Tier 4B."""
    v3 = _make_signal([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05)
    lora = _make_signal([0.83, 0.06, 0.04, 0.02, 0.02, 0.03])
    psv = _make_psv(argmax=0, max_val=0.0, margin=0.0, reliability=0.05, succeeded=False)
    classifier = _make_classifier(argmax=0, max_val=0.80, margin=0.65)
    conformal = _make_conformal(pred_set={0}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# S4B.9 — Tier 4B with degraded classifier pointing at dangerous disease (T5 fires)
# Spec lines 4979–4985
#
# v3: probs=[0.05, 0.05, 0.40, 0.04, 0.05, 0.04], chilli_leak=0.37
# LoRA: failed; succeeded=False
# PSV: argmax=2, max=0.45, margin=0.10, reliability=0.55
# Classifier (LoRA zeroed): argmax=2 (late_blight), max=0.55, margin=0.20
# → Tier 4B, T5 alert: True (rule 1; T5 fires for late_blight)
# ---------------------------------------------------------------------------

def test_S4B_9():
    """S4B.9 — LoRA failed; degraded classifier shows late_blight → Rule 1 → Tier 4B; T5 True."""
    v3 = _make_signal([0.05, 0.05, 0.40, 0.04, 0.05, 0.04], chilli_leak=0.37)
    lora = _make_signal_failed()
    psv = _make_psv(argmax=2, max_val=0.45, margin=0.10, reliability=0.55)
    classifier = _make_classifier(argmax=2, max_val=0.55, margin=0.20)
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
# S4B.10 — Multiple failure modes (PSV failed + v3 numerical issue)
# Spec lines 4987–4995
#
# v3: succeeded=False (RuntimeWarning during softmax); probs all 0.0
# LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02], succeeded=True
# PSV: succeeded=False; reliability=0.05
# IQA: ACCEPTABLE
# Classifier (v3 + PSV zeroed; LoRA-driven): argmax=0 (foliar), max=0.62, margin=0.45
# Conformal: set={0}, size=1
# → Tier 4B, T5 alert: False (rule 1; two signals failed)
# ---------------------------------------------------------------------------

def test_S4B_10():
    """S4B.10 — v3 + PSV both failed; LoRA drives classifier → Rule 1 → Tier 4B."""
    v3 = _make_signal_failed()
    lora = _make_signal([0.85, 0.05, 0.04, 0.02, 0.02, 0.02])
    psv = _make_psv(argmax=0, max_val=0.0, margin=0.0, reliability=0.05, succeeded=False)
    classifier = _make_classifier(argmax=0, max_val=0.62, margin=0.45)
    conformal = _make_conformal(pred_set={0}, size=1)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4B"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "1"
