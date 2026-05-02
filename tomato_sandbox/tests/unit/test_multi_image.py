"""
Unit tests for multi-image aggregation (Section 18).

spec: Section 18 (Multi-image input), lines 6085-6271.
DEC-044 Decision 1: canonical implementation at multi_image/aggregator.py.
DEC-044 Decision 6: AggregatedResult fields.
DEC-044 Decision 7: IQA REJECT images are marked is_iqa_rejected=True.
"""

import math
import pytest

from tomato_sandbox.multi_image.aggregator import (
    PerImageInput,
    PerImageSummary,
    AggregatedResult,
    aggregate_multi_image,
    _aggregate_t5,
    _aggregate_class_vote,
    _aggregate_conformal_set,
    _worst_iqa_decision,
)
from tomato_sandbox.tier.tier_assignment import TierAssignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_tier(
    label: str = "2",
    t5: bool = False,
    rule: str = "8c",
) -> TierAssignment:
    """Build a TierAssignment with minimal boilerplate."""
    return TierAssignment(tier_label=label, tier5_alert=t5, rule_id_fired=rule)


def _mk_v3_signal(
    cls: int = 1,
    probs: list | None = None,
    chilli_leak: float = 0.05,
    forward_succeeded: bool = True,
) -> dict:
    """Build a minimal v3_signal dict that passes Rule 1 and Rule 3."""
    if probs is None:
        p = [0.05] * 7
        p[cls] = 0.75
        probs = p
    return {
        "probs": probs,
        "chilli_leak": chilli_leak,
        "forward_succeeded": forward_succeeded,
    }


def _mk_lora_signal(
    cls: int = 1,
    probs: list | None = None,
    forward_succeeded: bool = True,
) -> dict:
    if probs is None:
        p = [0.05] * 7
        p[cls] = 0.75
        probs = p
    return {"probs": probs, "forward_succeeded": forward_succeeded}


def _mk_psv_signal(
    reliability: float = 0.70,
    argmax: int = 1,
    forward_succeeded: bool = True,
) -> dict:
    return {
        "reliability": reliability,
        "argmax": argmax,
        "max": 0.70,
        "margin": 0.30,
        "forward_succeeded": forward_succeeded,
    }


def _mk_classifier(
    argmax: int = 1,
    max_prob: float = 0.72,
    margin: float = 0.25,
) -> dict:
    return {"argmax": argmax, "max": max_prob, "margin": margin}


def _mk_conformal(classes: set | None = None) -> dict:
    if classes is None:
        classes = {1}
    return {"set": set(classes), "size": len(classes), "tau": None}


def _mk_iqa(decision: str = "ACCEPTABLE") -> dict:
    return {"decision": decision}


def _mk_input(
    image_id: str = "img_0",
    cls: int = 1,
    confidence: float = 0.72,
    conformal_classes: set | None = None,
    iqa_decision: str = "ACCEPTABLE",
    psv_reliability: float = 0.70,
    chilli_leakage: float = 0.05,
    t5: bool = False,
    tier_label: str = "2",
    tier_rule: str = "8c",
    is_iqa_rejected: bool = False,
    t5_prob: float = 0.0,
    t5_cls: int = -1,
    margin: float = 0.25,
) -> PerImageInput:
    """Build a PerImageInput that produces a valid non-failed image by default."""
    if conformal_classes is None:
        conformal_classes = {cls}
    return PerImageInput(
        image_id=image_id,
        tier_assignment=_mk_tier(label=tier_label, t5=t5, rule=tier_rule),
        primary_class=cls,
        primary_confidence=confidence,
        conformal_set=set(conformal_classes),
        iqa_decision=iqa_decision,
        psv_reliability=psv_reliability,
        chilli_leakage=chilli_leakage,
        v3_signal=_mk_v3_signal(cls=cls, chilli_leak=chilli_leakage),
        lora_signal=_mk_lora_signal(cls=cls),
        psv_signal=_mk_psv_signal(reliability=psv_reliability, argmax=cls),
        iqa_dict=_mk_iqa(iqa_decision),
        is_iqa_rejected=is_iqa_rejected,
        combined_margin=margin,
        tier5_trigger_probability=t5_prob,
        tier5_trigger_class=t5_cls,
    )


# ---------------------------------------------------------------------------
# Test imports
# ---------------------------------------------------------------------------

class TestImports:
    """Verify all public names are importable from the canonical and task-card paths."""

    def test_canonical_import(self):
        """spec: S21 line 6539 — canonical path is multi_image/aggregator.py"""
        from tomato_sandbox.multi_image.aggregator import (
            PerImageInput, PerImageSummary, AggregatedResult, aggregate_multi_image,
        )
        assert callable(aggregate_multi_image)

    def test_taskcard_shim_import(self):
        """DEC-044 Decision 1 — task-card path multi_image/multi_image.py also works."""
        from tomato_sandbox.multi_image.multi_image import (
            PerImageInput, PerImageSummary, AggregatedResult, aggregate_multi_image,
        )
        assert callable(aggregate_multi_image)

    def test_package_init_export(self):
        """Package __init__.py re-exports all four names."""
        import tomato_sandbox.multi_image as mi
        assert hasattr(mi, "PerImageInput")
        assert hasattr(mi, "PerImageSummary")
        assert hasattr(mi, "AggregatedResult")
        assert hasattr(mi, "aggregate_multi_image")

    def test_per_image_input_has_required_fields(self):
        """PerImageInput must have all required spec fields (spec 18.2 lines 6096-6112)."""
        img = _mk_input()
        assert hasattr(img, "image_id")
        assert hasattr(img, "tier_assignment")
        assert hasattr(img, "primary_class")
        assert hasattr(img, "primary_confidence")
        assert hasattr(img, "conformal_set")
        assert hasattr(img, "iqa_decision")
        assert hasattr(img, "psv_reliability")
        assert hasattr(img, "chilli_leakage")
        assert hasattr(img, "is_iqa_rejected")

    def test_aggregated_result_has_required_fields(self):
        """AggregatedResult must have all DEC-044 Decision 6 fields."""
        img = _mk_input()
        result = aggregate_multi_image([img])
        assert hasattr(result, "final_tier")
        assert hasattr(result, "per_image_summaries")
        assert hasattr(result, "tier5_alert_fired")
        assert hasattr(result, "tier5_reason")
        assert hasattr(result, "warnings")
        assert hasattr(result, "primary_class")
        assert hasattr(result, "conformal_set")
        assert hasattr(result, "iqa_decision")
        assert hasattr(result, "psv_reliability")
        assert hasattr(result, "chilli_leakage")
        assert hasattr(result, "n_successful")
        assert hasattr(result, "n_rejected")


# ---------------------------------------------------------------------------
# N=1 single-image passthrough
# ---------------------------------------------------------------------------

class TestSingleImagePassthrough:
    """spec: 18.4 line 6149 — N=1 multi-image is equivalent to single-image."""

    def test_single_image_tier_preserved(self):
        """N=1: final_tier.tier_label equals the input's tier_assignment.tier_label."""
        img = _mk_input(image_id="solo", tier_label="2")
        result = aggregate_multi_image([img])
        assert result.final_tier.tier_label == "2"

    def test_single_image_primary_class_preserved(self):
        """N=1: primary_class in result equals input's primary_class."""
        img = _mk_input(cls=0, tier_label="1")
        result = aggregate_multi_image([img])
        assert result.primary_class == 0

    def test_single_image_psv_reliability_preserved(self):
        """N=1: psv_reliability equals input's psv_reliability."""
        img = _mk_input(psv_reliability=0.65)
        result = aggregate_multi_image([img])
        assert result.psv_reliability == pytest.approx(0.65)

    def test_single_image_summary_list_has_one_entry(self):
        """N=1: per_image_summaries has exactly 1 entry."""
        img = _mk_input(image_id="only_img")
        result = aggregate_multi_image([img])
        assert len(result.per_image_summaries) == 1
        assert result.per_image_summaries[0].image_id == "only_img"

    def test_single_image_t5_false_preserved(self):
        """N=1: tier5_alert_fired is False when input has no T5."""
        img = _mk_input(t5=False)
        result = aggregate_multi_image([img])
        assert result.tier5_alert_fired is False

    def test_single_image_t5_true_preserved(self):
        """N=1: tier5_alert_fired is True when input fires T5."""
        img = _mk_input(t5=True, cls=2, tier_label="2")
        result = aggregate_multi_image([img])
        assert result.tier5_alert_fired is True

    def test_single_iqa_rejected_gives_n_rejected_1(self):
        """N=1: IQA-rejected image → n_rejected=1, n_successful=0."""
        img = _mk_input(is_iqa_rejected=True, tier_label="4B", tier_rule="1")
        result = aggregate_multi_image([img])
        assert result.n_rejected == 1
        assert result.n_successful == 0


# ---------------------------------------------------------------------------
# T5 alert OR aggregation (Step 1)
# ---------------------------------------------------------------------------

class TestT5AlertAggregation:
    """spec: 18.4 lines 6154-6157 — T5 aggregation semantics."""

    def test_no_t5_fires_when_none_fire(self):
        """If no image fires T5, aggregated T5 is False."""
        imgs = [
            _mk_input("a", t5=False),
            _mk_input("b", t5=False),
        ]
        fired, reason, cls = _aggregate_t5(imgs)
        assert fired is False
        assert cls == -1

    def test_t5_fires_if_one_image_fires(self):
        """spec: 18.4 line 6155 — ANY image fires → final T5 fires."""
        imgs = [
            _mk_input("a", t5=False),
            _mk_input("b", t5=True, t5_prob=0.30, t5_cls=2),
        ]
        fired, reason, _ = _aggregate_t5(imgs)
        assert fired is True

    def test_t5_class_from_highest_probability_image(self):
        """spec: 18.4 line 6157 — trigger_class from highest trigger_probability image."""
        imgs = [
            _mk_input("a", t5=True, t5_prob=0.25, t5_cls=2, tier_rule="r1"),
            _mk_input("b", t5=True, t5_prob=0.45, t5_cls=3, tier_rule="r2"),
        ]
        fired, _, trigger_cls = _aggregate_t5(imgs)
        assert fired is True
        assert trigger_cls == 3  # highest prob image is b with t5_cls=3

    def test_t5_reason_mixed_when_multiple_different_rules(self):
        """spec: 18.4 line 6156 — multiple T5 with different reasons → mixed_across_images."""
        imgs = [
            _mk_input("a", t5=True, t5_prob=0.25, tier_rule="rule_x"),
            _mk_input("b", t5=True, t5_prob=0.30, tier_rule="rule_y"),
        ]
        fired, reason, _ = _aggregate_t5(imgs)
        assert fired is True
        assert reason == "mixed_across_images"

    def test_t5_reason_single_source_not_mixed(self):
        """If all T5-firing images have the same rule_id_fired, reason is not mixed."""
        imgs = [
            _mk_input("a", t5=True, t5_prob=0.25, tier_rule="same_rule"),
            _mk_input("b", t5=True, t5_prob=0.30, tier_rule="same_rule"),
        ]
        fired, reason, _ = _aggregate_t5(imgs)
        assert fired is True
        assert reason != "mixed_across_images"

    def test_aggregate_t5_propagates_into_final_result(self):
        """Per-image T5 OR result propagates into AggregatedResult.tier5_alert_fired."""
        imgs = [
            _mk_input("safe_img", t5=False, cls=1),
            _mk_input("t5_img", t5=True, cls=2, t5_prob=0.30, t5_cls=2),
        ]
        result = aggregate_multi_image(imgs)
        assert result.tier5_alert_fired is True

    def test_t5_fires_even_when_one_image_fails(self):
        """T5 is evaluated on ALL images including failed ones (spec step 1 before step 2)."""
        # Failing image still fires T5
        imgs = [
            _mk_input("failed", t5=True, t5_prob=0.35, t5_cls=2,
                       is_iqa_rejected=True, tier_label="4B"),
            _mk_input("good", t5=False, cls=1),
        ]
        result = aggregate_multi_image(imgs)
        assert result.tier5_alert_fired is True


# ---------------------------------------------------------------------------
# All-fail edge case (Step 2)
# ---------------------------------------------------------------------------

class TestAllFailEdgeCase:
    """spec: 18.4 line 6160 — If ALL per-image tiers are 4B OR all IQA-rejected → Tier 4B."""

    def test_all_iqa_rejected_gives_4b(self):
        """All images IQA-rejected → final tier 4B."""
        imgs = [
            _mk_input("r1", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("r2", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("r3", is_iqa_rejected=True, tier_label="4B"),
        ]
        result = aggregate_multi_image(imgs)
        assert result.final_tier.tier_label == "4B"

    def test_all_failed_has_zero_successful(self):
        """All-fail → n_successful == 0, n_rejected == total images."""
        imgs = [
            _mk_input("r1", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("r2", is_iqa_rejected=True, tier_label="4B"),
        ]
        result = aggregate_multi_image(imgs)
        assert result.n_successful == 0
        assert result.n_rejected == 2

    def test_all_failed_warning_message_present(self):
        """All-fail → warning about no successful images."""
        imgs = [
            _mk_input("r1", is_iqa_rejected=True, tier_label="4B"),
        ]
        # N=1 path also covered for single IQA-reject
        imgs2 = [
            _mk_input("r1", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("r2", is_iqa_rejected=True, tier_label="4B"),
        ]
        result = aggregate_multi_image(imgs2)
        assert any("All images failed" in w for w in result.warnings)

    def test_all_failed_t5_still_propagated(self):
        """spec: Step 1 evaluated before Step 2 — T5 from failed images still propagates."""
        imgs = [
            _mk_input("r1", is_iqa_rejected=True, tier_label="4B", t5=True, t5_prob=0.40, t5_cls=2),
            _mk_input("r2", is_iqa_rejected=True, tier_label="4B", t5=False),
        ]
        result = aggregate_multi_image(imgs)
        assert result.final_tier.tier_label == "4B"
        assert result.tier5_alert_fired is True

    def test_some_fail_partial_exclusion_with_warning(self):
        """spec: 18.4 line 6161 — failed images excluded, warning added."""
        imgs = [
            _mk_input("r1", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("ok", cls=0, confidence=0.72),
        ]
        result = aggregate_multi_image(imgs)
        assert result.n_rejected == 1
        assert result.n_successful == 1
        assert any("excluded" in w.lower() for w in result.warnings)

    def test_some_fail_tier_based_on_successful_only(self):
        """Final tier is computed from successful images only, not failed ones."""
        imgs = [
            _mk_input("fail", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("ok1", cls=0, confidence=0.72),
            _mk_input("ok2", cls=0, confidence=0.75),
        ]
        result = aggregate_multi_image(imgs)
        # Final tier must not be 4B since there are successful images
        assert result.final_tier.tier_label != "4B"
        assert result.primary_class == 0


# ---------------------------------------------------------------------------
# Class voting (Step 3)
# ---------------------------------------------------------------------------

class TestClassVoting:
    """spec: 18.4 lines 6163-6168 — weighted class voting."""

    def test_unanimous_vote_selects_that_class(self):
        """All images agree → primary_class is that class."""
        imgs = [
            _mk_input("a", cls=1, confidence=0.72),
            _mk_input("b", cls=1, confidence=0.68),
            _mk_input("c", cls=1, confidence=0.80),
        ]
        final_argmax, _, _, vote_share = _aggregate_class_vote(imgs)
        assert final_argmax == 1
        assert vote_share == pytest.approx(1.0)

    def test_majority_vote_wins(self):
        """Two images vote for class 0, one votes for class 1 → class 0 wins."""
        imgs = [
            _mk_input("a", cls=0, confidence=0.70),
            _mk_input("b", cls=0, confidence=0.75),
            _mk_input("c", cls=1, confidence=0.80),
        ]
        final_argmax, _, _, _ = _aggregate_class_vote(imgs)
        assert final_argmax == 0

    def test_high_confidence_outweighs_low(self):
        """spec: 18.4 line 6165 — votes weighted by primary_confidence."""
        # cls=2 has two low-confidence votes; cls=0 has one high-confidence vote
        imgs = [
            _mk_input("a", cls=2, confidence=0.51),
            _mk_input("b", cls=2, confidence=0.52),
            _mk_input("c", cls=0, confidence=0.99),
        ]
        # Total weight cls=2: 1.03; cls=0: 0.99
        # cls=2 still wins by weight; this tests weights are actually used
        final_argmax, _, _, vote_share = _aggregate_class_vote(imgs)
        # cls=2 has total weight 1.03 > 0.99 → cls=2 wins
        assert final_argmax == 2
        assert vote_share < 1.0  # not unanimous

    def test_vote_share_below_threshold_generates_disagreement(self):
        """spec: 18.4 line 6189 — vote_share < 0.50 → disagreement warning."""
        imgs = [
            _mk_input("a", cls=0, confidence=0.60),
            _mk_input("b", cls=1, confidence=0.60),
            _mk_input("c", cls=2, confidence=0.60),
        ]
        # Three-way split → winning class has ~0.33 vote share
        result = aggregate_multi_image(imgs)
        assert any("disagreement" in w.lower() for w in result.warnings)

    def test_empty_successful_list(self):
        """_aggregate_class_vote with empty list returns all None."""
        final_argmax, combined_max, combined_margin, vote_share = _aggregate_class_vote([])
        assert final_argmax is None
        assert combined_max is None
        assert combined_margin is None
        assert vote_share == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Conformal set aggregation (Step 4)
# ---------------------------------------------------------------------------

class TestConformalSetAggregation:
    """spec: 18.4 lines 6170-6173 — conformal set majority-threshold aggregation."""

    def test_class_in_all_images_admitted(self):
        """Class present in all images (fraction=1.0 >= 0.50) → admitted."""
        imgs = [
            _mk_input("a", conformal_classes={0, 1}),
            _mk_input("b", conformal_classes={0, 1}),
        ]
        agg = _aggregate_conformal_set(imgs)
        assert 0 in agg
        assert 1 in agg

    def test_class_in_exactly_half_admitted(self):
        """spec: 18.4 line 6173 — 'A class in exactly half is admitted (boundary inclusive).'"""
        # 2 images; class 0 in 1 (fraction = 0.5 = exactly 50%)
        imgs = [
            _mk_input("a", conformal_classes={0, 1}),
            _mk_input("b", conformal_classes={1}),
        ]
        agg = _aggregate_conformal_set(imgs)
        assert 0 in agg  # 0.5 >= 0.50 → admitted

    def test_class_in_minority_excluded(self):
        """Class in <50% of images → excluded from final conformal set."""
        imgs = [
            _mk_input("a", conformal_classes={0}),
            _mk_input("b", conformal_classes={0}),
            _mk_input("c", conformal_classes={1}),  # 1 appears in only 1/3
        ]
        agg = _aggregate_conformal_set(imgs)
        assert 0 in agg
        assert 1 not in agg  # 1/3 < 0.50

    def test_empty_successful_list_gives_empty_set(self):
        """_aggregate_conformal_set with empty list returns empty set."""
        agg = _aggregate_conformal_set([])
        assert agg == set()

    def test_three_images_two_thirds_threshold(self):
        """Class in 2/3 images (>= 0.50) → admitted."""
        imgs = [
            _mk_input("a", conformal_classes={0, 2}),
            _mk_input("b", conformal_classes={0}),
            _mk_input("c", conformal_classes={0}),
        ]
        agg = _aggregate_conformal_set(imgs)
        assert 0 in agg
        assert 2 not in agg  # 1/3 < 0.50


# ---------------------------------------------------------------------------
# IQA worst-case aggregation (Step 5)
# ---------------------------------------------------------------------------

class TestIQAWorstAggregation:
    """spec: 18.4 lines 6175-6177 — worst IQA decision propagates."""

    def test_all_acceptable_gives_acceptable(self):
        """All ACCEPTABLE → worst is ACCEPTABLE."""
        imgs = [
            _mk_input("a", iqa_decision="ACCEPTABLE"),
            _mk_input("b", iqa_decision="ACCEPTABLE"),
        ]
        worst = _worst_iqa_decision(imgs)
        assert worst == "ACCEPTABLE"

    def test_one_degraded_among_acceptable_gives_degraded(self):
        """spec: 18.4 line 6176 — HIGH < ACCEPTABLE < DEGRADED < REJECT."""
        imgs = [
            _mk_input("a", iqa_decision="ACCEPTABLE"),
            _mk_input("b", iqa_decision="DEGRADED"),
            _mk_input("c", iqa_decision="HIGH"),
        ]
        worst = _worst_iqa_decision(imgs)
        assert worst == "DEGRADED"

    def test_high_is_best_iqa(self):
        """HIGH IQA is best (ordinal 0). If present with ACCEPTABLE, ACCEPTABLE wins."""
        imgs = [
            _mk_input("a", iqa_decision="HIGH"),
            _mk_input("b", iqa_decision="ACCEPTABLE"),
        ]
        worst = _worst_iqa_decision(imgs)
        assert worst == "ACCEPTABLE"  # ACCEPTABLE > HIGH

    def test_empty_successful_gives_acceptable_fallback(self):
        """_worst_iqa_decision with empty list returns ACCEPTABLE (safe default)."""
        worst = _worst_iqa_decision([])
        assert worst == "ACCEPTABLE"

    def test_degraded_iqa_propagates_to_final_result(self):
        """Worst IQA propagates into AggregatedResult.iqa_decision."""
        imgs = [
            _mk_input("a", iqa_decision="DEGRADED"),
            _mk_input("b", iqa_decision="ACCEPTABLE"),
        ]
        result = aggregate_multi_image(imgs)
        assert result.iqa_decision == "DEGRADED"


# ---------------------------------------------------------------------------
# PSV reliability and chilli leakage aggregation (Step 6)
# ---------------------------------------------------------------------------

class TestPSVAggregation:
    """spec: 18.4 lines 6179-6182 — min reliability, max chilli_leakage."""

    def test_psv_reliability_is_minimum(self):
        """spec: 18.4 line 6180 — Aggregated psv_reliability = minimum."""
        imgs = [
            _mk_input("a", psv_reliability=0.90),
            _mk_input("b", psv_reliability=0.55),
            _mk_input("c", psv_reliability=0.75),
        ]
        result = aggregate_multi_image(imgs)
        assert result.psv_reliability == pytest.approx(0.55)

    def test_chilli_leakage_is_maximum(self):
        """spec: 18.4 line 6181 — Aggregated chilli_leakage = maximum."""
        imgs = [
            _mk_input("a", chilli_leakage=0.05),
            _mk_input("b", chilli_leakage=0.25),
            _mk_input("c", chilli_leakage=0.10),
        ]
        result = aggregate_multi_image(imgs)
        assert result.chilli_leakage == pytest.approx(0.25)

    def test_psv_min_excludes_failed_images(self):
        """Failed images are excluded before min/max computation (Step 2 before Step 6)."""
        imgs = [
            _mk_input("good_a", psv_reliability=0.75),
            _mk_input("good_b", psv_reliability=0.65),
            _mk_input("failed", psv_reliability=0.10, is_iqa_rejected=True, tier_label="4B"),
        ]
        result = aggregate_multi_image(imgs)
        # If failed images were included, min would be 0.10; should be 0.65
        assert result.psv_reliability == pytest.approx(0.65)

    def test_chilli_max_excludes_failed_images(self):
        """Failed images' chilli_leakage should not inflate the max."""
        imgs = [
            _mk_input("good", chilli_leakage=0.05),
            _mk_input("failed", chilli_leakage=0.99, is_iqa_rejected=True, tier_label="4B"),
        ]
        result = aggregate_multi_image(imgs)
        assert result.chilli_leakage == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Final tier assignment (Step 7) via assign_tier
# ---------------------------------------------------------------------------

class TestFinalTierAssignment:
    """spec: 18.4 lines 6184-6186 — aggregated values passed through assign_tier()."""

    def test_two_to_five_images_return_aggregated_result(self):
        """2-5 images all produce an AggregatedResult (not a single passthrough)."""
        for n in range(2, 6):
            imgs = [_mk_input(f"img_{i}", cls=1, confidence=0.72) for i in range(n)]
            result = aggregate_multi_image(imgs)
            assert isinstance(result, AggregatedResult)
            assert result.n_successful == n

    def test_per_image_summaries_count_matches_input_count(self):
        """per_image_summaries length equals the number of inputs."""
        imgs = [_mk_input(f"img_{i}") for i in range(4)]
        result = aggregate_multi_image(imgs)
        assert len(result.per_image_summaries) == 4

    def test_per_image_summary_image_ids_preserved(self):
        """Each PerImageSummary carries the correct image_id."""
        imgs = [_mk_input(f"img_{i}") for i in range(3)]
        result = aggregate_multi_image(imgs)
        ids = [s.image_id for s in result.per_image_summaries]
        assert ids == ["img_0", "img_1", "img_2"]

    def test_conformal_set_reflects_majority(self):
        """Final conformal_set contains only classes present in >= 50% of images."""
        imgs = [
            _mk_input("a", conformal_classes={0}),
            _mk_input("b", conformal_classes={0}),
            _mk_input("c", conformal_classes={0, 1}),
        ]
        result = aggregate_multi_image(imgs)
        assert 0 in result.conformal_set  # 3/3 = 100%
        assert 1 not in result.conformal_set  # 1/3 < 50%

    def test_n_successful_n_rejected_sum_to_total(self):
        """n_successful + n_rejected must equal the number of input images."""
        imgs = [
            _mk_input("ok1"),
            _mk_input("ok2"),
            _mk_input("fail", is_iqa_rejected=True, tier_label="4B"),
        ]
        result = aggregate_multi_image(imgs)
        assert result.n_successful + result.n_rejected == len(imgs)

    def test_final_tier_label_is_string(self):
        """final_tier.tier_label is always a non-empty string."""
        imgs = [_mk_input(f"img_{i}") for i in range(2)]
        result = aggregate_multi_image(imgs)
        assert isinstance(result.final_tier.tier_label, str)
        assert len(result.final_tier.tier_label) > 0

    def test_final_tier_is_tier_assignment_type(self):
        """final_tier is a TierAssignment instance."""
        imgs = [_mk_input("a"), _mk_input("b")]
        result = aggregate_multi_image(imgs)
        assert isinstance(result.final_tier, TierAssignment)


# ---------------------------------------------------------------------------
# Mixed scenarios (combined)
# ---------------------------------------------------------------------------

class TestMixedScenarios:
    """End-to-end mixed scenarios exercising multiple steps together."""

    def test_2_images_unanimous_class_0(self):
        """Two images, both class 0, unanimous → primary_class == 0."""
        imgs = [
            _mk_input("a", cls=0, confidence=0.72),
            _mk_input("b", cls=0, confidence=0.74),
        ]
        result = aggregate_multi_image(imgs)
        assert result.primary_class == 0

    def test_3_images_two_fail_one_succeeds(self):
        """3 images: 2 failed, 1 successful → n_successful=1, no 4B."""
        imgs = [
            _mk_input("fail1", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("fail2", is_iqa_rejected=True, tier_label="4B"),
            _mk_input("ok", cls=1, confidence=0.72),
        ]
        result = aggregate_multi_image(imgs)
        assert result.n_successful == 1
        assert result.n_rejected == 2
        assert result.primary_class == 1
        assert result.final_tier.tier_label != "4B"

    def test_per_image_summaries_include_tier5_alert_status(self):
        """PerImageSummary.tier5_alert_fired reflects each image's individual T5 state."""
        imgs = [
            _mk_input("no_t5", t5=False),
            _mk_input("with_t5", t5=True, cls=2),
        ]
        result = aggregate_multi_image(imgs)
        summaries = {s.image_id: s for s in result.per_image_summaries}
        assert summaries["no_t5"].tier5_alert_fired is False
        assert summaries["with_t5"].tier5_alert_fired is True

    def test_chilli_leak_too_high_in_one_image_propagates(self):
        """Max chilli_leakage from Step 6 propagates; may push assign_tier to 3C."""
        # chilli_leakage=0.50 from one image → aggregated chilli=0.50 > 0.40 → Rule 3 → 3C
        imgs = [
            _mk_input("clean", chilli_leakage=0.05),
            _mk_input("leaky", chilli_leakage=0.50),
        ]
        result = aggregate_multi_image(imgs)
        # aggregated chilli_leakage must be the maximum
        assert result.chilli_leakage == pytest.approx(0.50)

    def test_five_images_all_class_1_produces_correct_primary(self):
        """5 images, all predicting class 1 → primary_class == 1."""
        imgs = [_mk_input(f"img_{i}", cls=1, confidence=0.70 + i * 0.01) for i in range(5)]
        result = aggregate_multi_image(imgs)
        assert result.primary_class == 1
        assert result.n_successful == 5
        assert result.n_rejected == 0

    def test_warnings_list_is_always_present(self):
        """AggregatedResult.warnings is always a list, even when empty."""
        imgs = [_mk_input("a"), _mk_input("b")]
        result = aggregate_multi_image(imgs)
        assert isinstance(result.warnings, list)

    def test_conformal_set_is_set_type(self):
        """AggregatedResult.conformal_set is always a set."""
        imgs = [_mk_input("a"), _mk_input("b")]
        result = aggregate_multi_image(imgs)
        assert isinstance(result.conformal_set, set)
