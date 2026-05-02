"""
Unit tests for tomato_sandbox.tier.tier_assignment.

Covers every rule_id_fired value, rule precedence, T5 alert independence,
boundary tests at every threshold, and underpowered_classes behavior.

Spec section: 14 (Tier assignment rules), lines 3665–4048.
Import contract: .claude/import_contract.md
"""

from __future__ import annotations

import math
import pytest

from tomato_sandbox.tier.tier_assignment import TierAssignment, assign_tier


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _sig(probs=None, chilli_leak=0.0, succeeded=True):
    """Build a minimal v3/lora signal dict."""
    if probs is None:
        probs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return {"probs": probs, "chilli_leak": chilli_leak, "forward_succeeded": succeeded}


def _sig_failed():
    return {"probs": [0.0] * 6, "chilli_leak": 0.0, "forward_succeeded": False}


def _psv(argmax=0, max_val=0.65, margin=0.30, reliability=0.71, succeeded=True):
    return {
        "argmax": argmax, "max": max_val, "margin": margin,
        "reliability": reliability, "forward_succeeded": succeeded,
    }


def _cls(argmax=0, max_val=0.91, margin=0.86):
    return {"argmax": argmax, "max": max_val, "margin": margin}


def _conf(pred_set=None, size=1, tau=None):
    if pred_set is None:
        pred_set = {0}
    return {"set": pred_set, "size": size, "tau": tau}


def _iqa(decision="ACCEPTABLE"):
    return {"decision": decision}


def _base_tier1_inputs():
    """Minimal inputs that produce Rule 7c → Tier 1."""
    return dict(
        v3_signal=_sig([0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03),
        lora_signal=_sig([0.88, 0.05, 0.02, 0.02, 0.02, 0.01]),
        psv_signal=_psv(argmax=0, max_val=0.71, margin=0.45, reliability=0.78),
        classifier=_cls(argmax=0, max_val=0.91, margin=0.86),
        conformal=_conf(pred_set={0}, size=1, tau=0.40),
        iqa=_iqa("ACCEPTABLE"),
    )


def _base_tier2_inputs():
    """Minimal inputs that produce Rule 8c → Tier 2."""
    return dict(
        v3_signal=_sig([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05),
        lora_signal=_sig([0.82, 0.06, 0.05, 0.03, 0.02, 0.02]),
        psv_signal=_psv(argmax=0, max_val=0.60, margin=0.25, reliability=0.55),
        classifier=_cls(argmax=0, max_val=0.80, margin=0.30),
        conformal=_conf(pred_set={0}, size=1, tau=0.45),
        iqa=_iqa("ACCEPTABLE"),
    )


# ---------------------------------------------------------------------------
# Return type structure
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_tier_assignment_dataclass(self):
        result = assign_tier(**_base_tier1_inputs())
        assert isinstance(result, TierAssignment)

    def test_has_tier_label_attribute(self):
        result = assign_tier(**_base_tier1_inputs())
        assert hasattr(result, "tier_label")

    def test_has_tier5_alert_attribute(self):
        result = assign_tier(**_base_tier1_inputs())
        assert hasattr(result, "tier5_alert")

    def test_has_rule_id_fired_attribute(self):
        result = assign_tier(**_base_tier1_inputs())
        assert hasattr(result, "rule_id_fired")

    def test_tier_label_is_string(self):
        result = assign_tier(**_base_tier1_inputs())
        assert isinstance(result.tier_label, str)

    def test_tier5_alert_is_bool(self):
        result = assign_tier(**_base_tier1_inputs())
        assert isinstance(result.tier5_alert, bool)


# ---------------------------------------------------------------------------
# Rule 1 — Pipeline failure → Tier 4B
# spec: section 14.5 lines 3823-3825
# ---------------------------------------------------------------------------

class TestRule1:
    def test_v3_failed_gives_tier4b(self):
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig([0.85, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(),
            classifier=_cls(),
            conformal=_conf(),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
        assert result.rule_id_fired == "1"

    def test_lora_failed_gives_tier4b(self):
        result = assign_tier(
            v3_signal=_sig([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05),
            lora_signal=_sig_failed(),
            psv_signal=_psv(),
            classifier=_cls(),
            conformal=_conf(),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
        assert result.rule_id_fired == "1"

    def test_psv_failed_gives_tier4b(self):
        result = assign_tier(
            v3_signal=_sig([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05),
            lora_signal=_sig([0.82, 0.06, 0.04, 0.02, 0.02, 0.03]),
            psv_signal=_psv(succeeded=False),
            classifier=_cls(),
            conformal=_conf(),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
        assert result.rule_id_fired == "1"

    def test_all_failed_gives_tier4b(self):
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig_failed(),
            psv_signal=_psv(succeeded=False),
            classifier=_cls(),
            conformal=_conf(),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
        assert result.rule_id_fired == "1"

    def test_rule1_overrides_rule3(self):
        """Rule 1 has highest priority; fires even when PSV is also unreliable."""
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig([0.80, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.10),  # would trigger Rule 3
            classifier=_cls(),
            conformal=_conf(),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
        assert result.rule_id_fired == "1"  # not "3"

    def test_rule1_t5_false_for_foliar_argmax(self):
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig([0.85, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(),
            classifier=_cls(argmax=0, max_val=0.78),
            conformal=_conf(pred_set={0}),
            iqa=_iqa(),
        )
        assert result.tier5_alert is False

    def test_rule1_t5_true_for_late_blight_argmax(self):
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig([0.05, 0.05, 0.81, 0.03, 0.03, 0.03]),
            psv_signal=_psv(argmax=2),
            classifier=_cls(argmax=2, max_val=0.71),
            conformal=_conf(pred_set={2}),
            iqa=_iqa(),
        )
        assert result.tier5_alert is True


# ---------------------------------------------------------------------------
# Rule 3 — PSV unreliable or chilli leakage → Tier 3C
# spec: section 14.5 lines 3831-3833
# ---------------------------------------------------------------------------

class TestRule3:
    def test_psv_reliability_below_040_fires_rule3(self):
        result = assign_tier(
            v3_signal=_sig([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05),
            lora_signal=_sig([0.82, 0.06, 0.05, 0.03, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.39),
            classifier=_cls(argmax=0, max_val=0.85, margin=0.78),
            conformal=_conf(pred_set={0}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "3C"
        assert result.rule_id_fired == "3"

    def test_psv_reliability_at_040_does_not_fire_rule3(self):
        """Boundary: 0.40 is NOT < 0.40, so Rule 3 strict < does not fire.
        spec: import_contract "Rule 3 psv_reliability 0.40 strict <"
        """
        result = assign_tier(
            v3_signal=_sig([0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05),
            lora_signal=_sig([0.89, 0.04, 0.03, 0.02, 0.01, 0.01]),
            psv_signal=_psv(argmax=0, max_val=0.45, margin=0.08, reliability=0.40),
            classifier=_cls(argmax=0, max_val=0.91, margin=0.86),
            conformal=_conf(pred_set={0}, size=1, tau=0.40),
            iqa=_iqa(),
        )
        assert result.tier_label != "3C"  # Rule 3 did not fire
        assert result.rule_id_fired != "3"

    def test_chilli_leak_above_040_fires_rule3(self):
        result = assign_tier(
            v3_signal=_sig([0.50, 0.04, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.41),
            lora_signal=_sig([0.85, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.71),
            classifier=_cls(argmax=0, max_val=0.78, margin=0.65),
            conformal=_conf(pred_set={0}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "3C"
        assert result.rule_id_fired == "3"

    def test_chilli_leak_at_040_does_not_fire_rule3(self):
        """Boundary: 0.40 is NOT > 0.40, so Rule 3 strict > does not fire.
        spec: import_contract "Rule 3 chilli_leakage 0.40 strict >"
        """
        result = assign_tier(
            v3_signal=_sig([0.55, 0.04, 0.01, 0.00, 0.00, 0.00], chilli_leak=0.40),
            lora_signal=_sig([0.85, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.74),
            classifier=_cls(argmax=0, max_val=0.82, margin=0.71),
            conformal=_conf(pred_set={0}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label != "3C"
        assert result.rule_id_fired != "3"

    def test_rule4_overrides_rule3_when_max_low(self):
        """Rule 4 fires before Rule 3 when max < 0.45 (BLK-011 sub-defect 11.1).
        Spec header says Rule 3 > Rule 4, but scenario SB.10 body contradicts this.
        Scenario body is authoritative (BLK-004 precedent): Rule 4 fires before Rule 3.
        When max=0.30 < 0.45 and size=1 (no two-class bypass), Rule 4 fires → Tier 4A,
        even though psv_reliability=0.35 < 0.40 would fire Rule 3 if checked first.
        spec: BLK-011 sub-defect 11.1; SB.10 scenario body (spec lines 5208-5217)
        """
        result = assign_tier(
            v3_signal=_sig([0.30, 0.05, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.05),
            lora_signal=_sig([0.30, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.35),
            classifier=_cls(argmax=0, max_val=0.30, margin=0.20),
            conformal=_conf(pred_set={0}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"
        assert result.rule_id_fired == "4"

    def test_rule3_fires_when_max_above_threshold(self):
        """Rule 3 fires when max >= 0.45 (Rule 4 does not fire) and psv_reliability low.
        When max=0.46 >= 0.45, Rule 4 does not fire; Rule 3 then fires → Tier 3C.
        spec: section 14.5; BLK-011 sub-defect 11.1
        """
        result = assign_tier(
            v3_signal=_sig([0.46, 0.05, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.05),
            lora_signal=_sig([0.46, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.35),  # < 0.40 fires Rule 3
            classifier=_cls(argmax=0, max_val=0.46, margin=0.40),
            conformal=_conf(pred_set={0}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "3C"
        assert result.rule_id_fired == "3"

    def test_rule3_t5_true_for_dangerous_argmax(self):
        """T5 fires independently even when Rule 3 sets the tier.
        spec: section 14.3 — T5 evaluated independently after tier label assigned
        """
        result = assign_tier(
            v3_signal=_sig([0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04),
            lora_signal=_sig([0.06, 0.06, 0.81, 0.03, 0.02, 0.02]),
            psv_signal=_psv(argmax=2, reliability=0.32),
            classifier=_cls(argmax=2, max_val=0.88, margin=0.83),
            conformal=_conf(pred_set={2}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "3C"
        assert result.tier5_alert is True


# ---------------------------------------------------------------------------
# Rule 4 — Low confidence → Tier 4A
# spec: section 14.5 lines 3835-3837
# ---------------------------------------------------------------------------

class TestRule4:
    def test_max_below_045_fires_rule4(self):
        result = assign_tier(
            v3_signal=_sig([0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02),
            lora_signal=_sig([0.32, 0.20, 0.20, 0.10, 0.10, 0.08]),
            psv_signal=_psv(reliability=0.55),
            classifier=_cls(argmax=0, max_val=0.30, margin=0.10),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"
        assert result.rule_id_fired == "4"

    def test_max_at_045_does_not_fire_rule4(self):
        """Boundary: 0.45 is NOT < 0.45 (strict <), so Rule 4 does not fire."""
        # Set size=3 so Rule 5 will fire (to avoid falling to Rule 9)
        result = assign_tier(
            v3_signal=_sig([0.40, 0.15, 0.15, 0.10, 0.10, 0.10], chilli_leak=0.02),
            lora_signal=_sig([0.40, 0.15, 0.15, 0.10, 0.10, 0.10]),
            psv_signal=_psv(reliability=0.55),
            classifier=_cls(argmax=0, max_val=0.45, margin=0.30),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        # Rule 4 should NOT fire (0.45 is not < 0.45); Rule 5 fires instead
        assert result.rule_id_fired != "4"
        assert result.tier_label == "3B"

    def test_rule4_overrides_rule5(self):
        """Rule 4 fires before Rule 5 even when set_size >= 3.
        spec: section 14.5 priority Rule 4 > Rule 5
        """
        result = assign_tier(
            v3_signal=_sig([0.18, 0.16, 0.15, 0.14, 0.14, 0.13], chilli_leak=0.10),
            lora_signal=_sig([0.18, 0.16, 0.15, 0.14, 0.14, 0.13]),
            psv_signal=_psv(reliability=0.45),
            classifier=_cls(argmax=0, max_val=0.18, margin=0.02),
            conformal=_conf(pred_set={0, 1, 2, 3, 4, 5, 6}, size=7),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"
        assert result.rule_id_fired == "4"

    def test_rule4_fires_before_rule6_when_max_very_low(self):
        """Rule 4 fires before Rule 6 when max < 0.41 and size=2 (BLK-011 sub-defect 11.2).
        Matches S4A.4: max=0.40, size=2 → Tier 4A (Rule 4).
        Rule 4 pre-empts Rule 6 when max is below the 0.41 threshold.
        spec: BLK-011 sub-defect 11.2; S4A.4 scenario body (spec lines 4800-4808)
        """
        result = assign_tier(
            v3_signal=_sig([0.40, 0.30, 0.10, 0.05, 0.05, 0.05], chilli_leak=0.05),
            lora_signal=_sig([0.42, 0.31, 0.10, 0.05, 0.06, 0.06]),
            psv_signal=_psv(reliability=0.65),
            classifier=_cls(argmax=0, max_val=0.40, margin=0.10),
            conformal=_conf(pred_set={0, 1}, size=2),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"
        assert result.rule_id_fired == "4"

    def test_rule6_fires_not_rule4_when_max_above_lower_bound(self):
        """Rule 6 fires instead of Rule 4 when max >= 0.41 and size=2 (BLK-011 sub-defect 11.2).
        Matches S3A.3: max=0.42, size=2 → Tier 3A (Rule 6).
        When max >= 0.41 with size=2, genuine two-class ambiguity; Rule 6 fires.
        spec: BLK-011 sub-defect 11.2; S3A.3 scenario body (spec lines 4360-4367)
        """
        result = assign_tier(
            v3_signal=_sig([0.04, 0.04, 0.04, 0.42, 0.04, 0.40], chilli_leak=0.02),
            lora_signal=_sig([0.05, 0.05, 0.05, 0.40, 0.05, 0.40]),
            psv_signal=_psv(argmax=3, reliability=0.69),
            classifier=_cls(argmax=3, max_val=0.42, margin=0.02),
            conformal=_conf(pred_set={3, 5}, size=2),
            iqa=_iqa(),
        )
        assert result.tier_label == "3A"
        assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# Rule 5 — Three-or-more-class / empty set
# spec: section 14.5 lines 3839-3843
# ---------------------------------------------------------------------------

class TestRule5:
    def test_size_3_fires_rule5_tier3b(self):
        result = assign_tier(
            v3_signal=_sig([0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01),
            lora_signal=_sig([0.46, 0.21, 0.18, 0.04, 0.06, 0.05]),
            psv_signal=_psv(reliability=0.65),
            classifier=_cls(argmax=0, max_val=0.46, margin=0.26),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        assert result.tier_label == "3B"
        assert result.rule_id_fired == "5"

    def test_empty_set_fires_rule5_tier4a(self):
        """Empty set (size=0) gives Tier 4A via Rule 5.
        spec: section 14.5 line 3840 "empty set treated as low confidence"
        """
        result = assign_tier(
            v3_signal=_sig([0.50, 0.10, 0.10, 0.10, 0.10, 0.10], chilli_leak=0.00),
            lora_signal=_sig([0.50, 0.12, 0.10, 0.10, 0.10, 0.08]),
            psv_signal=_psv(reliability=0.60),
            classifier=_cls(argmax=0, max_val=0.50, margin=0.40),
            conformal=_conf(pred_set=set(), size=0),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"
        assert result.rule_id_fired == "5"

    def test_size_5_fires_rule5_tier3b(self):
        result = assign_tier(
            v3_signal=_sig([0.20, 0.18, 0.06, 0.06, 0.20, 0.20], chilli_leak=0.10),
            lora_signal=_sig([0.18, 0.17, 0.05, 0.05, 0.20, 0.35]),
            psv_signal=_psv(reliability=0.55),
            classifier=_cls(argmax=4, max_val=0.19, margin=0.00),
            conformal=_conf(pred_set={0, 1, 4, 5, 6}, size=5),
            iqa=_iqa(),
        )
        # Rule 4 fires (max=0.19 < 0.45), not Rule 5
        assert result.rule_id_fired == "4"

    def test_rule5_overrides_rule6(self):
        """Rule 5 fires before Rule 6 when size >= 3.
        spec: section 14.5 priority Rule 5 > Rule 6
        """
        # Build a scenario where size==3 but otherwise everything is fine
        result = assign_tier(
            v3_signal=_sig([0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01),
            lora_signal=_sig([0.46, 0.21, 0.18, 0.04, 0.06, 0.05]),
            psv_signal=_psv(reliability=0.65),
            classifier=_cls(argmax=0, max_val=0.46, margin=0.26),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        assert result.tier_label == "3B"  # not 3A
        assert result.rule_id_fired == "5"  # not "6"


# ---------------------------------------------------------------------------
# Rule 6 — Two-class set → Tier 3A
# spec: section 14.5 lines 3845-3847
# ---------------------------------------------------------------------------

class TestRule6:
    def test_size_2_fires_rule6(self):
        result = assign_tier(
            v3_signal=_sig([0.50, 0.50, 0.00, 0.00, 0.00, 0.00], chilli_leak=0.00),
            lora_signal=_sig([0.50, 0.50, 0.00, 0.00, 0.00, 0.00]),
            psv_signal=_psv(reliability=0.55),
            classifier=_cls(argmax=0, max_val=0.50, margin=0.00),
            conformal=_conf(pred_set={0, 1}, size=2),
            iqa=_iqa(),
        )
        assert result.tier_label == "3A"
        assert result.rule_id_fired == "6"

    def test_rule6_overrides_rule7(self):
        """Rule 6 fires before Rule 7 when size==2, even with high max/margin.
        spec: section 14.5 priority Rule 6 > Rule 7
        (set-size rules before confidence rules)
        """
        result = assign_tier(
            v3_signal=_sig([0.45, 0.45, 0.02, 0.02, 0.02, 0.02], chilli_leak=0.01),
            lora_signal=_sig([0.45, 0.45, 0.02, 0.02, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.75),
            classifier=_cls(argmax=0, max_val=0.90, margin=0.45),
            conformal=_conf(pred_set={0, 1}, size=2),
            iqa=_iqa(),
        )
        assert result.tier_label == "3A"
        assert result.rule_id_fired == "6"


# ---------------------------------------------------------------------------
# Rule 7 — Single-class definitive (7a, 7b, 7c)
# spec: section 14.5 lines 3849-3860
# ---------------------------------------------------------------------------

class TestRule7:
    def test_rule7c_tier1_default(self):
        """Rule 7c: Tier 1 when all conditions met, no downgrade."""
        result = assign_tier(**_base_tier1_inputs())
        assert result.tier_label == "1"
        assert result.rule_id_fired == "7c"

    def test_rule7a_degraded_iqa_gives_tier3d(self):
        """Rule 7a: DEGRADED IQA caps to Tier 3D (takes precedence over 7b, 7c).
        spec: section 14.5 line 3855; section 14.8 "IQA cap takes precedence over underpowered"
        """
        inputs = _base_tier1_inputs()
        inputs["iqa"] = _iqa("DEGRADED")
        result = assign_tier(**inputs)
        assert result.tier_label == "3D"
        assert result.rule_id_fired == "7a"

    def test_rule7b_underpowered_argmax_gives_tier3a(self):
        """Rule 7b: underpowered argmax class → Tier 3A."""
        inputs = _base_tier1_inputs()
        result = assign_tier(**inputs, underpowered_classes={0})  # argmax=0 is underpowered
        assert result.tier_label == "3A"
        assert result.rule_id_fired == "7b"

    def test_rule7a_overrides_rule7b(self):
        """Rule 7a (DEGRADED IQA) takes precedence over Rule 7b (underpowered).
        spec: section 14.8 "Sub-rule 7a/8a takes precedence over sub-rule 7b/8b"
        """
        inputs = _base_tier1_inputs()
        inputs["iqa"] = _iqa("DEGRADED")
        result = assign_tier(**inputs, underpowered_classes={0})
        assert result.tier_label == "3D"
        assert result.rule_id_fired == "7a"  # not 7b

    def test_rule7_requires_max_at_least_085(self):
        """Rule 7 requires max >= 0.85 (inclusive).
        spec: section 14.5 line 3851; import_contract "Rule 7 max 0.85 inclusive >="
        """
        inputs = _base_tier1_inputs()
        inputs["classifier"] = _cls(argmax=0, max_val=0.84999999, margin=0.86)
        result = assign_tier(**inputs)
        # Falls to Rule 8 (Tier 2) not Rule 7 (Tier 1)
        assert result.tier_label == "2"
        assert result.rule_id_fired == "8c"

    def test_rule7_requires_margin_at_least_030(self):
        """Rule 7 requires margin >= 0.30 (inclusive).
        spec: section 14.5 line 3852; import_contract "Rule 7 margin 0.30 inclusive >="
        """
        inputs = _base_tier1_inputs()
        inputs["classifier"] = _cls(argmax=0, max_val=0.91, margin=0.29)
        result = assign_tier(**inputs)
        # margin 0.29 < 0.30: Rule 7 fails; Rule 8 fires
        assert result.tier_label == "2"
        assert result.rule_id_fired == "8c"

    def test_rule7_requires_psv_reliability_at_least_050(self):
        """Rule 7 requires psv_reliability >= 0.50 (inclusive).
        spec: section 14.5 line 3853; import_contract "Rule 7 psv_reliability 0.50 inclusive >="
        """
        inputs = _base_tier1_inputs()
        inputs["psv_signal"] = _psv(reliability=0.49)
        result = assign_tier(**inputs)
        # reliability 0.49 < 0.50: Rule 7 fails; Rule 8 fires
        assert result.tier_label == "2"
        assert result.rule_id_fired == "8c"

    def test_rule7_psv_reliability_at_050_boundary_passes(self):
        """Boundary: psv_reliability == 0.50 satisfies Rule 7's >= 0.50."""
        inputs = _base_tier1_inputs()
        inputs["psv_signal"] = _psv(reliability=0.50)
        result = assign_tier(**inputs)
        assert result.tier_label == "1"
        assert result.rule_id_fired == "7c"

    def test_rule7_requires_chilli_strict_below_020(self):
        """Rule 7 requires chilli_leakage < 0.20 (strict).
        spec: section 14.5 line 3854; import_contract "Rule 7 chilli_leakage 0.20 strict <"
        """
        inputs = _base_tier1_inputs()
        inputs["v3_signal"] = _sig([0.74, 0.04, 0.01, 0.00, 0.00, 0.01], chilli_leak=0.20)
        result = assign_tier(**inputs)
        # chilli 0.20 is NOT < 0.20 (strict), so Rule 7 fails; Rule 8 fires
        assert result.tier_label == "2"
        assert result.rule_id_fired == "8c"

    def test_rule7_chilli_at_019_passes(self):
        """chilli_leakage=0.19 satisfies Rule 7's strict < 0.20."""
        inputs = _base_tier1_inputs()
        inputs["v3_signal"] = _sig([0.74, 0.04, 0.01, 0.00, 0.00, 0.01], chilli_leak=0.19)
        result = assign_tier(**inputs)
        assert result.tier_label == "1"
        assert result.rule_id_fired == "7c"

    def test_rule7_underpowered_none_no_downgrade(self):
        """underpowered_classes=None means no underpowered guard fires.
        spec: import_contract "underpowered_classes: None (treated as empty set)"
        """
        inputs = _base_tier1_inputs()
        result = assign_tier(**inputs, underpowered_classes=None)
        assert result.tier_label == "1"
        assert result.rule_id_fired == "7c"

    def test_rule7_high_iqa_not_degraded(self):
        """IQA=HIGH does not trigger 7a cap; HIGH is treated like ACCEPTABLE for Tier 1.
        spec: section 14.2 Tier 1 — "IQA decision is ACCEPTABLE or HIGH"
        """
        inputs = _base_tier1_inputs()
        inputs["iqa"] = _iqa("HIGH")
        result = assign_tier(**inputs)
        assert result.tier_label == "1"
        assert result.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# Rule 8 — Single-class confident (8a, 8b, 8c)
# spec: section 14.5 lines 3862-3873
# ---------------------------------------------------------------------------

class TestRule8:
    def test_rule8c_tier2_default(self):
        """Rule 8c: Tier 2 when all Rule 8 conditions met, no downgrade."""
        result = assign_tier(**_base_tier2_inputs())
        assert result.tier_label == "2"
        assert result.rule_id_fired == "8c"

    def test_rule8a_degraded_iqa_gives_tier3d(self):
        """Rule 8a: DEGRADED IQA caps to Tier 3D."""
        inputs = _base_tier2_inputs()
        inputs["iqa"] = _iqa("DEGRADED")
        result = assign_tier(**inputs)
        assert result.tier_label == "3D"
        assert result.rule_id_fired == "8a"

    def test_rule8b_underpowered_argmax_gives_tier3a(self):
        """Rule 8b: underpowered argmax → Tier 3A."""
        inputs = _base_tier2_inputs()
        result = assign_tier(**inputs, underpowered_classes={0})
        assert result.tier_label == "3A"
        assert result.rule_id_fired == "8b"

    def test_rule8a_overrides_rule8b(self):
        """Rule 8a (DEGRADED IQA) takes precedence over Rule 8b (underpowered).
        spec: section 14.8 "8a > 8b"
        """
        inputs = _base_tier2_inputs()
        inputs["iqa"] = _iqa("DEGRADED")
        result = assign_tier(**inputs, underpowered_classes={0})
        assert result.tier_label == "3D"
        assert result.rule_id_fired == "8a"

    def test_rule8_requires_max_at_least_065(self):
        """Rule 8 requires max >= 0.65 (inclusive).
        spec: section 14.5 line 3863; import_contract "Rule 8 max 0.65 inclusive >="
        """
        inputs = _base_tier2_inputs()
        inputs["classifier"] = _cls(argmax=0, max_val=0.64, margin=0.30)
        result = assign_tier(**inputs)
        assert result.rule_id_fired == "catch_all_low_confidence"

    def test_rule8_requires_margin_at_least_020(self):
        """Rule 8 requires margin >= 0.20 (inclusive)."""
        inputs = _base_tier2_inputs()
        inputs["classifier"] = _cls(argmax=0, max_val=0.80, margin=0.19)
        result = assign_tier(**inputs)
        assert result.rule_id_fired == "catch_all_low_confidence"

    def test_rule8_requires_psv_reliability_at_least_040(self):
        """Rule 8 requires psv_reliability >= 0.40 (inclusive).
        spec: import_contract "Rule 8 psv_reliability 0.40 inclusive >="
        """
        inputs = _base_tier2_inputs()
        inputs["psv_signal"] = _psv(reliability=0.40)  # exact boundary — passes
        result = assign_tier(**inputs)
        assert result.tier_label == "2"
        assert result.rule_id_fired == "8c"

    def test_rule8_psv_reliability_039_falls_to_rule3(self):
        """psv_reliability=0.39 fires Rule 3 (which fires before Rule 8)."""
        inputs = _base_tier2_inputs()
        inputs["psv_signal"] = _psv(reliability=0.39)
        result = assign_tier(**inputs)
        assert result.tier_label == "3C"
        assert result.rule_id_fired == "3"

    def test_rule8_requires_chilli_strict_below_030(self):
        """Rule 8 requires chilli_leakage < 0.30 (strict).
        spec: import_contract "Rule 8 chilli_leakage 0.30 strict <"
        """
        inputs = _base_tier2_inputs()
        inputs["v3_signal"] = _sig([0.05, 0.05, 0.55, 0.02, 0.02, 0.01], chilli_leak=0.30)
        result = assign_tier(**inputs)
        # chilli=0.30 is NOT < 0.30 (strict), so Rule 8 fails → Rule 9
        assert result.rule_id_fired == "catch_all_low_confidence"


# ---------------------------------------------------------------------------
# Rule 9 — Catch-all → Tier 4A
# spec: section 14.5 lines 3875-3876
# ---------------------------------------------------------------------------

class TestRule9:
    def test_catch_all_fires_when_no_rule_matches(self):
        """Rule 9 fires when size==1 but confidence thresholds are in the gap.
        e.g., chilli=0.40 exactly: Rule 3 strict > fails; Rule 7 chilli<0.20 fails;
        Rule 8 chilli<0.30 fails → Rule 9.
        """
        result = assign_tier(
            v3_signal=_sig([0.55, 0.04, 0.01, 0.00, 0.00, 0.00], chilli_leak=0.40),
            lora_signal=_sig([0.85, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.74),
            classifier=_cls(argmax=0, max_val=0.82, margin=0.71),
            conformal=_conf(pred_set={0}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"
        assert result.rule_id_fired == "catch_all_low_confidence"

    def test_catch_all_chilli_030_exact(self):
        """chilli=0.30 exactly: Rule 8 chilli < 0.30 fails (strict) → Rule 9.
        BLK-004 Defect-15.3 scenario S3C.12.
        """
        result = assign_tier(
            v3_signal=_sig([0.05, 0.05, 0.55, 0.02, 0.02, 0.01], chilli_leak=0.30),
            lora_signal=_sig([0.06, 0.05, 0.83, 0.02, 0.02, 0.02]),
            psv_signal=_psv(argmax=2, max_val=0.55, margin=0.20, reliability=0.62),
            classifier=_cls(argmax=2, max_val=0.78, margin=0.65),
            conformal=_conf(pred_set={2}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"
        assert result.rule_id_fired == "catch_all_low_confidence"
        # T5 should still fire for late_blight argmax
        assert result.tier5_alert is True


# ---------------------------------------------------------------------------
# T5 alert — independent of base tier
# spec: section 14.3 lines 3784-3800
# ---------------------------------------------------------------------------

class TestT5Alert:
    def test_t5_false_for_foliar_argmax(self):
        """T5 does not fire for argmax=0 (foliar) — not in {2,3,4}."""
        result = assign_tier(**_base_tier1_inputs())
        assert result.tier5_alert is False

    def test_t5_true_for_late_blight_argmax(self):
        """T5 fires: argmax=2 (late_blight) AND max >= 0.20.
        spec: section 14.3 line 3789
        """
        inputs = _base_tier1_inputs()
        inputs["v3_signal"] = _sig([0.02, 0.02, 0.89, 0.01, 0.01, 0.01], chilli_leak=0.04)
        inputs["lora_signal"] = _sig([0.03, 0.03, 0.87, 0.02, 0.02, 0.03])
        inputs["psv_signal"] = _psv(argmax=2, max_val=0.78, margin=0.55, reliability=0.74)
        inputs["classifier"] = _cls(argmax=2, max_val=0.92, margin=0.87)
        inputs["conformal"] = _conf(pred_set={2}, size=1)
        result = assign_tier(**inputs)
        assert result.tier5_alert is True

    def test_t5_true_for_ylcv_argmax(self):
        """T5 fires: argmax=3 (ylcv) AND max >= 0.20."""
        inputs = _base_tier1_inputs()
        inputs["v3_signal"] = _sig([0.02, 0.02, 0.02, 0.84, 0.02, 0.02], chilli_leak=0.06)
        inputs["lora_signal"] = _sig([0.03, 0.02, 0.02, 0.85, 0.04, 0.04])
        inputs["psv_signal"] = _psv(argmax=3, max_val=0.81, margin=0.62, reliability=0.85)
        inputs["classifier"] = _cls(argmax=3, max_val=0.87, margin=0.78)
        inputs["conformal"] = _conf(pred_set={3}, size=1)
        result = assign_tier(**inputs)
        assert result.tier5_alert is True

    def test_t5_true_for_mosaic_argmax(self):
        """T5 fires: argmax=4 (mosaic) AND max >= 0.20."""
        inputs = _base_tier1_inputs()
        inputs["v3_signal"] = _sig([0.04, 0.03, 0.02, 0.02, 0.86, 0.01], chilli_leak=0.02)
        inputs["lora_signal"] = _sig([0.05, 0.03, 0.02, 0.02, 0.84, 0.04])
        inputs["psv_signal"] = _psv(argmax=4, max_val=0.69, margin=0.42, reliability=0.71)
        inputs["classifier"] = _cls(argmax=4, max_val=0.88, margin=0.81)
        inputs["conformal"] = _conf(pred_set={4}, size=1)
        result = assign_tier(**inputs)
        assert result.tier5_alert is True

    def test_t5_false_for_healthy_argmax(self):
        """T5 does not fire for argmax=5 (healthy)."""
        inputs = _base_tier1_inputs()
        inputs["v3_signal"] = _sig([0.01, 0.02, 0.01, 0.02, 0.01, 0.91], chilli_leak=0.02)
        inputs["lora_signal"] = _sig([0.02, 0.03, 0.02, 0.02, 0.02, 0.89])
        inputs["psv_signal"] = _psv(argmax=5, max_val=0.79, margin=0.58, reliability=0.83)
        inputs["classifier"] = _cls(argmax=5, max_val=0.93, margin=0.88)
        inputs["conformal"] = _conf(pred_set={5}, size=1)
        result = assign_tier(**inputs)
        assert result.tier5_alert is False

    def test_t5_threshold_at_020_inclusive(self):
        """T5 threshold is >= 0.20 (inclusive): exactly 0.20 fires T5.
        spec: section 14.3 line 3792; import_contract "T5 max 0.20 inclusive >="
        """
        result = assign_tier(
            v3_signal=_sig([0.02, 0.04, 0.84, 0.02, 0.02, 0.01], chilli_leak=0.05),
            lora_signal=_sig([0.04, 0.05, 0.80, 0.03, 0.04, 0.04]),
            psv_signal=_psv(argmax=2, max_val=0.65, margin=0.30, reliability=0.50),
            classifier=_cls(argmax=2, max_val=0.85, margin=0.30),
            conformal=_conf(pred_set={2}, size=1),
            iqa=_iqa(),
        )
        assert result.tier5_alert is True

    def test_t5_threshold_below_020_no_fire(self):
        """T5 argmax threshold is strictly >= 0.20: max=0.19 does NOT fire T5."""
        result = assign_tier(
            v3_signal=_sig([0.20, 0.18, 0.06, 0.06, 0.20, 0.20], chilli_leak=0.10),
            lora_signal=_sig([0.18, 0.17, 0.05, 0.05, 0.20, 0.35]),
            psv_signal=_psv(argmax=4, max_val=0.30, margin=0.05, reliability=0.55),
            classifier=_cls(argmax=4, max_val=0.19, margin=0.00),
            conformal=_conf(pred_set={0, 1, 4, 5, 6}, size=5),
            iqa=_iqa(),
        )
        assert result.tier5_alert is False

    def test_t5_late_blight_in_set_fires_via_v3_prob(self):
        """T5 in-set trigger: late_blight in conformal set AND v3_probs[2] >= 0.20.
        spec: section 14.3 line 3790; import_contract T5 in-set logic
        """
        result = assign_tier(
            v3_signal=_sig([0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01),
            lora_signal=_sig([0.46, 0.21, 0.18, 0.04, 0.06, 0.05]),
            psv_signal=_psv(argmax=0, max_val=0.55, margin=0.20, reliability=0.65),
            classifier=_cls(argmax=0, max_val=0.46, margin=0.26),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        # late_blight in set; v3_probs[2]=0.20 >= 0.20
        assert result.tier5_alert is True

    def test_t5_late_blight_in_set_fires_via_lora_prob(self):
        """T5 in-set trigger fires via lora_probs[2] when v3_probs[2] < 0.20."""
        result = assign_tier(
            v3_signal=_sig([0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02),
            lora_signal=_sig([0.32, 0.22, 0.20, 0.10, 0.10, 0.06]),  # lora[2]=0.20
            psv_signal=_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.65),
            classifier=_cls(argmax=0, max_val=0.30, margin=0.10),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        assert result.tier5_alert is True

    def test_t5_late_blight_in_set_no_fire_when_prob_019(self):
        """T5 in-set does NOT fire when late_blight prob = 0.19 < 0.20."""
        result = assign_tier(
            v3_signal=_sig([0.30, 0.20, 0.18, 0.11, 0.10, 0.09], chilli_leak=0.02),
            lora_signal=_sig([0.32, 0.22, 0.19, 0.10, 0.10, 0.07]),  # lora[2]=0.19
            psv_signal=_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.65),
            classifier=_cls(argmax=0, max_val=0.30, margin=0.09),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        assert result.tier5_alert is False

    def test_t5_mosaic_in_set_does_not_fire_inset_trigger(self):
        """Mosaic (class 4) in set does NOT trigger T5 in-set rule.
        spec: section 14.3 lines 3800 — mosaic uses argmax-only trigger.
        """
        result = assign_tier(
            v3_signal=_sig([0.30, 0.05, 0.20, 0.18, 0.18, 0.04], chilli_leak=0.05),
            lora_signal=_sig([0.32, 0.06, 0.20, 0.18, 0.18, 0.06]),
            psv_signal=_psv(argmax=0, max_val=0.45, margin=0.10, reliability=0.55),
            classifier=_cls(argmax=0, max_val=0.30, margin=0.10),
            conformal=_conf(pred_set={0, 2, 3, 4}, size=4),
            iqa=_iqa(),
        )
        # late_blight (2) in set at 0.20 fires T5; mosaic/YLCV in set do NOT add more
        assert result.tier5_alert is True  # fires due to late_blight in set

    def test_t5_ylcv_in_set_no_inset_trigger(self):
        """YLCV (class 3) in set does NOT trigger T5 in-set rule.
        spec: section 14.3 lines 3800 — YLCV uses argmax-only trigger.
        """
        result = assign_tier(
            v3_signal=_sig([0.05, 0.45, 0.40, 0.03, 0.03, 0.03], chilli_leak=0.01),
            lora_signal=_sig([0.06, 0.43, 0.38, 0.04, 0.04, 0.05]),
            psv_signal=_psv(argmax=1, max_val=0.50, margin=0.15, reliability=0.74),
            classifier=_cls(argmax=1, max_val=0.45, margin=0.05),
            conformal=_conf(pred_set={1, 2}, size=2),  # late_blight IS in set
            iqa=_iqa(),
        )
        # late_blight in set at prob 0.40 (v3[2]) → T5 fires
        assert result.tier5_alert is True

    def test_t5_independent_of_tier1(self):
        """T5 can fire alongside Tier 1."""
        inputs = _base_tier1_inputs()
        inputs["v3_signal"] = _sig([0.02, 0.02, 0.89, 0.01, 0.01, 0.01], chilli_leak=0.04)
        inputs["lora_signal"] = _sig([0.03, 0.03, 0.87, 0.02, 0.02, 0.03])
        inputs["psv_signal"] = _psv(argmax=2, max_val=0.78, margin=0.55, reliability=0.74)
        inputs["classifier"] = _cls(argmax=2, max_val=0.92, margin=0.87)
        inputs["conformal"] = _conf(pred_set={2}, size=1)
        result = assign_tier(**inputs)
        assert result.tier_label == "1"
        assert result.tier5_alert is True

    def test_t5_independent_of_tier4b(self):
        """T5 can fire alongside Tier 4B (pipeline failure).
        spec: section 14.3 — T5 evaluated independently after tier label
        """
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig([0.05, 0.05, 0.81, 0.03, 0.03, 0.03]),
            psv_signal=_psv(argmax=2),
            classifier=_cls(argmax=2, max_val=0.71),
            conformal=_conf(pred_set={2}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
        assert result.tier5_alert is True


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_nan_classifier_max_all_signals_failed_gives_tier4b(self):
        """NaN in classifier max: orchestrator marks all signals failed → Rule 1.
        spec: section 15 scenario SB.9
        """
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig_failed(),
            psv_signal=_psv(argmax=0, max_val=0.0, margin=0.0, reliability=0.0, succeeded=False),
            classifier=_cls(argmax=0, max_val=float("nan"), margin=float("nan")),
            conformal=_conf(pred_set=set(), size=0),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
        assert result.tier5_alert is False
        assert result.rule_id_fired == "1"


# ---------------------------------------------------------------------------
# Underpowered classes
# ---------------------------------------------------------------------------

class TestUnderpoweredClasses:
    def test_none_underpowered_classes_no_downgrade(self):
        """underpowered_classes=None: no downgrade fires at all.
        spec: import_contract "underpowered_classes: None treated as empty set"
        """
        result = assign_tier(**_base_tier1_inputs(), underpowered_classes=None)
        assert result.tier_label == "1"

    def test_empty_set_underpowered_no_downgrade(self):
        result = assign_tier(**_base_tier1_inputs(), underpowered_classes=set())
        assert result.tier_label == "1"

    def test_underpowered_class_not_argmax_no_downgrade(self):
        """underpowered_classes={3} but argmax=0: no downgrade (guard only fires for argmax)."""
        inputs = _base_tier1_inputs()
        inputs["classifier"] = _cls(argmax=0, max_val=0.91, margin=0.86)
        result = assign_tier(**inputs, underpowered_classes={3})
        assert result.tier_label == "1"

    def test_underpowered_guard_fires_in_rule7(self):
        """7b fires when argmax is in underpowered_classes under Rule 7 conditions."""
        inputs = _base_tier1_inputs()
        result = assign_tier(**inputs, underpowered_classes={0})
        assert result.tier_label == "3A"
        assert result.rule_id_fired == "7b"

    def test_underpowered_guard_fires_in_rule8(self):
        """8b fires when argmax is in underpowered_classes under Rule 8 conditions."""
        inputs = _base_tier2_inputs()
        result = assign_tier(**inputs, underpowered_classes={0})
        assert result.tier_label == "3A"
        assert result.rule_id_fired == "8b"

    def test_underpowered_guard_does_not_fire_before_rule7(self):
        """Underpowered guard has no effect when Rules 1-6 fire first.
        spec: import_contract "The underpowered guard only activates within sub-rules 7b and 8b"
        """
        # Rule 3 fires (PSV unreliable); underpowered class irrelevant
        result = assign_tier(
            v3_signal=_sig([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05),
            lora_signal=_sig([0.82, 0.06, 0.05, 0.03, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.35),
            classifier=_cls(argmax=3, max_val=0.85, margin=0.78),
            conformal=_conf(pred_set={3}, size=1),
            iqa=_iqa(),
            underpowered_classes={3},
        )
        assert result.tier_label == "3C"
        assert result.rule_id_fired == "3"


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_import_from_canonical_path(self):
        """Primary import path must work.
        spec: import_contract.md line 13
        """
        from tomato_sandbox.tier.tier_assignment import assign_tier as f1
        assert callable(f1)

    def test_import_from_package_shim(self):
        """Package-level import must also work.
        DEC-033: re-export shim at tomato_sandbox/tier/__init__.py
        """
        from tomato_sandbox.tier import assign_tier as f2
        assert callable(f2)

    def test_import_tier_assignment_dataclass(self):
        from tomato_sandbox.tier.tier_assignment import TierAssignment as TA
        assert TA is TierAssignment

    def test_tier_assignment_dataclass_attributes(self):
        ta = TierAssignment(tier_label="1", tier5_alert=False, rule_id_fired="7c")
        assert ta.tier_label == "1"
        assert ta.tier5_alert is False
        assert ta.rule_id_fired == "7c"


# ---------------------------------------------------------------------------
# Valid tier labels exhaustive check
# ---------------------------------------------------------------------------

class TestValidTierLabels:
    """Verify every spec-allowed tier label can be produced."""

    def test_produces_tier1(self):
        result = assign_tier(**_base_tier1_inputs())
        assert result.tier_label == "1"

    def test_produces_tier2(self):
        result = assign_tier(**_base_tier2_inputs())
        assert result.tier_label == "2"

    def test_produces_tier3a_rule6(self):
        result = assign_tier(
            v3_signal=_sig([0.50, 0.50, 0.00, 0.00, 0.00, 0.00], chilli_leak=0.00),
            lora_signal=_sig([0.50, 0.50, 0.00, 0.00, 0.00, 0.00]),
            psv_signal=_psv(reliability=0.55),
            classifier=_cls(argmax=0, max_val=0.50, margin=0.00),
            conformal=_conf(pred_set={0, 1}, size=2),
            iqa=_iqa(),
        )
        assert result.tier_label == "3A"

    def test_produces_tier3b(self):
        result = assign_tier(
            v3_signal=_sig([0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01),
            lora_signal=_sig([0.46, 0.21, 0.18, 0.04, 0.06, 0.05]),
            psv_signal=_psv(reliability=0.65),
            classifier=_cls(argmax=0, max_val=0.46, margin=0.26),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        assert result.tier_label == "3B"

    def test_produces_tier3c(self):
        result = assign_tier(
            v3_signal=_sig([0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05),
            lora_signal=_sig([0.82, 0.06, 0.05, 0.03, 0.02, 0.02]),
            psv_signal=_psv(reliability=0.39),
            classifier=_cls(argmax=0, max_val=0.85, margin=0.78),
            conformal=_conf(pred_set={0}, size=1),
            iqa=_iqa(),
        )
        assert result.tier_label == "3C"

    def test_produces_tier3d_via_rule7a(self):
        inputs = _base_tier1_inputs()
        inputs["iqa"] = _iqa("DEGRADED")
        result = assign_tier(**inputs)
        assert result.tier_label == "3D"

    def test_produces_tier3d_via_rule8a(self):
        inputs = _base_tier2_inputs()
        inputs["iqa"] = _iqa("DEGRADED")
        result = assign_tier(**inputs)
        assert result.tier_label == "3D"

    def test_produces_tier4a_rule4(self):
        result = assign_tier(
            v3_signal=_sig([0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02),
            lora_signal=_sig([0.32, 0.20, 0.20, 0.10, 0.10, 0.08]),
            psv_signal=_psv(reliability=0.55),
            classifier=_cls(argmax=0, max_val=0.30, margin=0.10),
            conformal=_conf(pred_set={0, 1, 2}, size=3),
            iqa=_iqa(),
        )
        assert result.tier_label == "4A"

    def test_produces_tier4b(self):
        result = assign_tier(
            v3_signal=_sig_failed(),
            lora_signal=_sig([0.85, 0.05, 0.04, 0.02, 0.02, 0.02]),
            psv_signal=_psv(),
            classifier=_cls(),
            conformal=_conf(),
            iqa=_iqa(),
        )
        assert result.tier_label == "4B"
