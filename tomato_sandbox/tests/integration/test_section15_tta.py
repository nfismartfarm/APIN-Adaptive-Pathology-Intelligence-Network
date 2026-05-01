"""
Section 15.15 — TTA-specific scenarios (STTA.1 – STTA.5).
Spec source: tomato_3_signal_system.md lines 5423–5463.

TTA (Test-Time Augmentation) changes the classifier's input by aggregating
multiple augmented views of v3 and LoRA outputs. assign_tier receives the
post-TTA classifier output and assigns tier normally. These scenarios show that
the tier assignment logic is agnostic to whether TTA fired; it only sees the
final (post-TTA) classifier output.

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
# STTA.1 — Initial 0.50 → TTA fires (2-view) → final 0.72 (Tier 2; foliar argmax)
# Spec lines 5427–5434
#
# 1-view: argmax=0, max=0.50, margin=0.30 (triggers 2-view TTA)
# Post-TTA classifier: argmax=0 (foliar), max=0.72, margin=0.50
# Conformal (τ=0.50): set={0}, size=1
# → Tier 2, T5 alert: False (rule 8c; argmax=0 foliar; late_blight prob ≈ 0.05 < 0.20)
# Note: assign_tier receives the post-TTA classifier output directly.
# PSV and signal inputs are illustrative of a typical Tier 2 scenario.
# ---------------------------------------------------------------------------

def test_STTA_1():
    """STTA.1 — Post-2-view TTA classifier max=0.72; Rule 8 main IF met → Tier 2; T5 False."""
    v3 = _make_signal([0.70, 0.10, 0.05, 0.05, 0.05, 0.05], chilli_leak=0.00)
    lora = _make_signal([0.72, 0.10, 0.05, 0.05, 0.05, 0.03])
    psv = _make_psv(argmax=0, max_val=0.60, margin=0.30, reliability=0.65)
    classifier = _make_classifier(argmax=0, max_val=0.72, margin=0.50)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.50)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"


# ---------------------------------------------------------------------------
# STTA.2 — Initial 0.92 → no TTA (max >= 0.55; above TTA trigger threshold)
# Spec lines 5436–5439
#
# 1-view classifier: max=0.92 >= 0.55 → no TTA fires
# → Tier 1 (all Rule 7 conditions met)
# ---------------------------------------------------------------------------

def test_STTA_2():
    """STTA.2 — max=0.92 (no TTA fires); all Rule 7 conditions met → Tier 1; T5 False."""
    v3 = _make_signal([0.90, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.02)
    lora = _make_signal([0.91, 0.03, 0.02, 0.01, 0.01, 0.02])
    psv = _make_psv(argmax=0, max_val=0.75, margin=0.50, reliability=0.80)
    classifier = _make_classifier(argmax=0, max_val=0.92, margin=0.85)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.40)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "1"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# STTA.3 — TTA fires (2-view) but post-TTA max=0.52 still below Rule 8 threshold → Tier 4A
# Spec lines 5441–5449
#
# 1-view: max=0.50 (triggers 2-view TTA)
# Post-2-view classifier: argmax=0, max=0.52, margin=0.32 (small improvement)
# 5-view does NOT fire (post-2-view max 0.52 >= 0.45)
# Conformal: set={0}, size=1
# → Tier 4A (Rule 9 catch-all; max 0.52 fails Rule 4's < 0.45, fails Rules 7/8 max thresholds)
# Walk: max 0.52 >= 0.45 (Rule 4 fails), max 0.52 < 0.65 (Rule 8 fails), max 0.52 < 0.85 (Rule 7 fails).
#       set_size==1 (Rules 5/6 don't fire). Falls to Rule 9 → Tier 4A.
# ---------------------------------------------------------------------------

def test_STTA_3():
    """STTA.3 — Post-2-view TTA max=0.52; Rule 4 fails (>= 0.45), Rule 8 fails (< 0.65); catch-all → Tier 4A."""
    v3 = _make_signal([0.50, 0.20, 0.10, 0.08, 0.06, 0.06], chilli_leak=0.00)
    lora = _make_signal([0.52, 0.20, 0.10, 0.08, 0.06, 0.04])
    psv = _make_psv(argmax=0, max_val=0.48, margin=0.15, reliability=0.55)
    classifier = _make_classifier(argmax=0, max_val=0.52, margin=0.32)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.55)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "catch_all_low_confidence"


# ---------------------------------------------------------------------------
# STTA.4 — TTA fires (5-view) → low-confidence multi-class → Tier 4A
# Spec lines 5451–5456
#
# 1-view: argmax=0, max=0.40, margin=0.05 (max < 0.45 → 5-view TTA fires)
# 5-view aggregated: post-TTA argmax=0, max=0.42, margin=0.05 (still low)
# Conformal (τ=0.85): threshold 0.15; set has 4 classes
# → Tier 4A (Rule 4: max 0.42 < 0.45)
# ---------------------------------------------------------------------------

def test_STTA_4():
    """STTA.4 — Post-5-view TTA max=0.42 still < 0.45; Rule 4 → Tier 4A; T5 False."""
    v3 = _make_signal([0.40, 0.20, 0.15, 0.10, 0.10, 0.05], chilli_leak=0.00)
    lora = _make_signal([0.42, 0.20, 0.15, 0.10, 0.08, 0.05])
    psv = _make_psv(argmax=0, max_val=0.40, margin=0.05, reliability=0.50)
    classifier = _make_classifier(argmax=0, max_val=0.42, margin=0.05)
    conformal = _make_conformal(pred_set={0, 1, 2, 3}, size=4, tau=0.85)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "4A"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "4"


# ---------------------------------------------------------------------------
# STTA.5 — 5-view TTA escalation resolves uncertainty → Tier 2
# Spec lines 5458–5463
#
# 1-view: max=0.40, margin=0.10 (max < 0.45 → 5-view TTA fires)
# 5-view aggregated: post-TTA argmax=0, max=0.78, margin=0.55 (substantial improvement)
# Conformal: set={0}, size=1
# → Tier 2, T5 alert: False (argmax foliar; rule 8c fires after TTA)
# Note: without TTA, 1-view alone would have routed to Tier 4A.
# ---------------------------------------------------------------------------

def test_STTA_5():
    """STTA.5 — Post-5-view TTA max=0.78 (1-view max=0.40 alone → Tier 4A); Rule 8 → Tier 2; T5 False."""
    v3 = _make_signal([0.75, 0.10, 0.05, 0.04, 0.03, 0.03], chilli_leak=0.00)
    lora = _make_signal([0.76, 0.10, 0.05, 0.04, 0.03, 0.02])
    psv = _make_psv(argmax=0, max_val=0.65, margin=0.40, reliability=0.70)
    classifier = _make_classifier(argmax=0, max_val=0.78, margin=0.55)
    conformal = _make_conformal(pred_set={0}, size=1, tau=0.45)
    iqa = {"decision": "ACCEPTABLE"}

    result = assign_tier(
        v3_signal=v3, lora_signal=lora, psv_signal=psv,
        classifier=classifier, conformal=conformal, iqa=iqa,
    )

    assert result.tier_label == "2"
    assert result.tier5_alert is False
    assert result.rule_id_fired == "8c"
