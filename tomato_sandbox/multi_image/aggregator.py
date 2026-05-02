"""
Multi-image aggregation for the Tomato 3-Signal system.

Spec section: 18 (Multi-image input), lines 6085-6271.
DEC-044: canonical implementation at multi_image/aggregator.py per spec file layout (S21 line 6539).
Task-card path multi_image/multi_image.py is a re-export shim.

Aggregation strategy (spec 18.4 lines 6144-6186):
  Pre-step: N=1 passthrough; 4B/IQA-REJECT exclusion.
  Step 1: T5 alert — ANY image fires → final T5 fires (logical OR).
  Step 2: Pipeline failure — ALL failed → Tier 4B. Some failed → exclude, warn.
  Step 3: Class voting — weighted by primary_confidence.
  Step 4: Conformal set — union, keep classes in >= 50% of successful images.
  Step 5: Aggregated IQA — WORST decision across successful images.
  Step 6: Aggregated PSV — min(reliability), max(chilli_leakage).
  Step 7: Final tier — run assign_tier() on aggregated values.
  Disagreement detection: vote share < 0.50 → warning (spec 18.4 line 6189).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.tier.tier_assignment import TierAssignment, assign_tier

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# IQA decision ordering (worst = highest ordinal) for Step 5
# spec: 18.4 line 6176 — "ordering is HIGH < ACCEPTABLE < DEGRADED < REJECT"
# ---------------------------------------------------------------------------
_IQA_ORDER: dict[str, int] = {
    "HIGH": 0,
    "ACCEPTABLE": 1,
    "DEGRADED": 2,
    "REJECT": 3,   # REJECT images are excluded from successful pool
}

# Threshold for strong disagreement warning
# spec: 18.4 line 6189 — "top-voted class has weighted-vote share < 0.50"
_DISAGREE_VOTE_SHARE_THRESHOLD: float = 0.50

# Conformal set inclusion threshold
# spec: 18.4 line 6173 — "fraction >= 0.50 (majority of successful images agreed)"
_CONFORMAL_FRACTION_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Per-image input dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerImageInput:
    """Per-image data supplied to aggregate_multi_image.

    Contains the TierAssignment plus the raw classifier/signal values needed
    for Steps 3-6 of the aggregation (spec 18.4).

    Fields:
        image_id: Unique string identifier (spec 18.2 line 6100).
        tier_assignment: TierAssignment from assign_tier() for this image.
        primary_class: classifier["argmax"] for this image. -1 if unavailable.
        primary_confidence: classifier["max"] for this image.
        conformal_set: conformal["set"] for this image (set[int]).
        iqa_decision: iqa["decision"] string.
        psv_reliability: psv_signal["reliability"] for this image.
        chilli_leakage: v3_signal["chilli_leak"] for this image.
        v3_signal: Full v3_signal dict (for assign_tier aggregated call).
        lora_signal: Full lora_signal dict (for assign_tier aggregated call).
        psv_signal: Full psv_signal dict (for assign_tier aggregated call).
        iqa_dict: Full iqa dict (for assign_tier aggregated call).
        is_iqa_rejected: True if image was rejected at IQA gate (before assign_tier).
        combined_margin: classifier["margin"] for this image.
        tier5_trigger_probability: Probability that drove T5 (best effort; 0.0 if unavailable).
        tier5_trigger_class: Class index that triggered T5 (-1 if not applicable).
    """
    image_id: str
    tier_assignment: TierAssignment
    primary_class: int
    primary_confidence: float
    conformal_set: "set[int]"
    iqa_decision: str
    psv_reliability: float
    chilli_leakage: float
    v3_signal: dict
    lora_signal: dict
    psv_signal: dict
    iqa_dict: dict
    is_iqa_rejected: bool = False
    combined_margin: float = 0.0
    tier5_trigger_probability: float = 0.0
    tier5_trigger_class: int = -1


# ---------------------------------------------------------------------------
# Per-image summary for response (spec 18.6 JSON lines 6206-6224)
# ---------------------------------------------------------------------------

@dataclass
class PerImageSummary:
    """Compact summary of one image's result, returned in the API response.

    spec: 18.6 lines 6207-6224 JSON block.
    """
    image_id: str
    tier: str             # tier_label string
    primary_class: int    # class index
    primary_confidence: float
    tier5_alert_fired: bool


# ---------------------------------------------------------------------------
# Aggregated result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AggregatedResult:
    """Result of aggregate_multi_image.

    DEC-044 Decision 6: fields cover final tier + per-image summaries +
    aggregated signal values needed by response builder.

    spec: 18.4 (all 7 steps), 18.5 (final tier), 18.6 (per-image breakdown).
    """
    final_tier: TierAssignment             # spec: 18.5 line 6193
    per_image_summaries: list[PerImageSummary]  # spec: 18.6 lines 6206-6224
    tier5_alert_fired: bool                # spec: 18.4 Step 1 line 6155
    tier5_reason: str                      # spec: 18.4 Step 1 line 6156
    tier5_trigger_class: int               # spec: 18.4 Step 1 line 6157
    warnings: list[str]                    # spec: 18.4 line 6161, 6189
    primary_class: Optional[int]           # final argmax (Step 3)
    combined_max_prob: Optional[float]     # weighted mean confidence for winning class
    combined_margin: Optional[float]       # weighted mean margin for winning class
    conformal_set: "set[int]"             # aggregated conformal set (Step 4)
    iqa_decision: str                      # worst IQA decision (Step 5)
    psv_reliability: float                 # min reliability (Step 6)
    chilli_leakage: float                  # max chilli_leakage (Step 6)
    n_successful: int                      # number of images that contributed to aggregation
    n_rejected: int                        # number of IQA-rejected or 4B images


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_failed_image(img: PerImageInput) -> bool:
    """True if image is IQA-rejected or Tier 4B (pipeline failure).

    spec: 18.4 lines 6148-6152 — Pre-step exclusion logic.
    """
    return img.is_iqa_rejected or img.tier_assignment.tier_label == "4B"


def _iqa_order(decision: str) -> int:
    """Ordinal for IQA decision comparison (higher = worse).

    spec: 18.4 line 6176 — "HIGH < ACCEPTABLE < DEGRADED < REJECT"
    """
    return _IQA_ORDER.get(decision, 1)  # unknown → ACCEPTABLE


def _aggregate_t5(
    images: list[PerImageInput],
) -> tuple[bool, str, int]:
    """Step 1: T5 alert aggregation.

    spec: 18.4 lines 6154-6157
    "Final tier5_alert.fired = True if ANY per-image tier has tier5_alert.fired == True."
    "Final tier5_alert.reason = reason from per-image alert with highest trigger_probability."
    "Final tier5_alert.trigger_class = trigger class from highest-probability T5 firing."

    Returns (fired, reason, trigger_class).
    """
    fired = False
    best_prob = -1.0
    reason = ""
    trigger_class = -1

    t5_fires = [img for img in images if img.tier_assignment.tier5_alert]
    if not t5_fires:
        return False, "", -1

    fired = True
    # Collect T5 firings; pick reason from highest trigger_probability
    for img in t5_fires:
        if img.tier5_trigger_probability > best_prob:
            best_prob = img.tier5_trigger_probability
            reason = img.tier_assignment.rule_id_fired  # best available source for reason
            trigger_class = img.tier5_trigger_class

    # If multiple images fire T5 with different reasons, say mixed_across_images
    # spec: 18.4 line 6156
    t5_rules = {img.tier_assignment.rule_id_fired for img in t5_fires}
    if len(t5_fires) > 1 and len(t5_rules) > 1:
        reason = "mixed_across_images"

    return fired, reason, trigger_class


def _aggregate_class_vote(
    successful: list[PerImageInput],
) -> tuple[Optional[int], Optional[float], Optional[float], float]:
    """Step 3: Weighted class voting.

    spec: 18.4 lines 6163-6168
    "Each image's vote is weighted by its primary_confidence."
    "Pick the class with the highest total weighted vote as the final argmax."
    "combined_max_prob = weighted mean of per-image max probabilities for winning class."
    "combined_margin = weighted mean of per-image margins for winning class."

    Returns (final_argmax, combined_max_prob, combined_margin, vote_share_of_winner).
    """
    if not successful:
        return None, None, None, 0.0

    # Accumulate weighted votes per class
    vote_weights: dict[int, float] = {}
    for img in successful:
        cls = img.primary_class
        weight = img.primary_confidence if math.isfinite(img.primary_confidence) else 0.0
        vote_weights[cls] = vote_weights.get(cls, 0.0) + weight

    total_weight = sum(vote_weights.values())
    if total_weight <= 0.0:
        return None, None, None, 0.0

    final_argmax = max(vote_weights, key=lambda c: vote_weights[c])
    vote_share = vote_weights[final_argmax] / total_weight

    # Weighted mean of per-image max_prob and margin for the winning class only
    # spec: 18.4 line 6167-6168 — "images that voted for other classes are excluded"
    winners = [img for img in successful if img.primary_class == final_argmax]
    if not winners:
        return final_argmax, None, None, vote_share

    total_winner_weight = sum(
        img.primary_confidence for img in winners
        if math.isfinite(img.primary_confidence)
    )
    if total_winner_weight <= 0.0:
        # Fallback to simple mean
        combined_max = sum(img.primary_confidence for img in winners) / len(winners)
        combined_margin = sum(img.combined_margin for img in winners) / len(winners)
    else:
        combined_max = sum(
            img.primary_confidence * img.primary_confidence
            for img in winners
            if math.isfinite(img.primary_confidence)
        ) / total_winner_weight
        combined_margin = sum(
            img.combined_margin * img.primary_confidence
            for img in winners
            if math.isfinite(img.primary_confidence)
        ) / total_winner_weight

    return final_argmax, combined_max, combined_margin, vote_share


def _aggregate_conformal_set(
    successful: list[PerImageInput],
) -> "set[int]":
    """Step 4: Conformal set aggregation.

    spec: 18.4 lines 6170-6173
    "Compute union of per-image prediction sets."
    "For each class in the union, compute fraction of images that included it."
    "Final prediction set = classes with fraction >= 0.50."
    "A class in exactly half is admitted (boundary inclusive)."
    """
    if not successful:
        return set()

    n = len(successful)
    # Count how many images include each class
    class_counts: dict[int, int] = {}
    for img in successful:
        for cls in img.conformal_set:
            class_counts[cls] = class_counts.get(cls, 0) + 1

    # spec: 18.4 line 6173 — "fraction >= 0.50 (inclusive)"
    return {
        cls
        for cls, count in class_counts.items()
        if count / n >= _CONFORMAL_FRACTION_THRESHOLD
    }


def _worst_iqa_decision(successful: list[PerImageInput]) -> str:
    """Step 5: Aggregated IQA decision — worst across successful images.

    spec: 18.4 lines 6175-6177
    "Aggregated IQA decision = WORST per-image IQA decision across successful images."
    "Ordering: HIGH < ACCEPTABLE < DEGRADED < REJECT (REJECT excluded from successful)."
    """
    if not successful:
        return "ACCEPTABLE"
    return max(successful, key=lambda img: _iqa_order(img.iqa_decision)).iqa_decision


def _build_aggregated_signals(
    final_argmax: int,
    combined_max: float,
    combined_margin: float,
    conformal_set: "set[int]",
    iqa_decision: str,
    psv_reliability: float,
    chilli_leakage: float,
    successful: list[PerImageInput],
) -> tuple[dict, dict, dict, dict, dict]:
    """Build aggregated signal dicts for the assign_tier() call (Step 7).

    spec: 18.5 line 6193 — "aggregated values are passed through assign_tier()
    as if from a single image."

    The aggregated v3_signal and lora_signal are built from the first successful
    image's signal (since we don't have access to raw probs from all images at this
    layer; the aggregated max_prob encodes the consensus confidence). forward_succeeded
    is True if we have at least one successful image.
    """
    # Use the representative image from the winning class for signal structures
    # (forward pass succeeded = True since they are in the successful pool)
    representative = None
    for img in successful:
        if img.primary_class == final_argmax:
            representative = img
            break
    if representative is None and successful:
        representative = successful[0]

    if representative is None:
        # All failed — degenerate case; return failed signals
        return (
            {"probs": [0.0] * 7, "chilli_leak": chilli_leakage, "forward_succeeded": False},
            {"probs": [0.0] * 7, "forward_succeeded": False},
            {"argmax": -1, "max": 0.0, "margin": 0.0, "reliability": psv_reliability, "forward_succeeded": False},
            {"argmax": final_argmax, "max": combined_max, "margin": combined_margin},
            {"decision": iqa_decision},
        )

    # Aggregate v3_signal: use representative's probs but override chilli_leakage
    # (max chilli_leakage from Step 6 is more conservative)
    # spec: 18.4 lines 6179-6182 — "max across successful images' chilli leakage"
    agg_v3 = dict(representative.v3_signal)
    agg_v3["chilli_leak"] = chilli_leakage
    agg_v3["forward_succeeded"] = True

    # lora_signal: use representative's (forward pass already succeeded)
    agg_lora = dict(representative.lora_signal)
    agg_lora["forward_succeeded"] = True

    # PSV: override reliability with min (Step 6)
    # spec: 18.4 line 6180 — "Aggregated psv_reliability = minimum across successful images"
    agg_psv = dict(representative.psv_signal)
    agg_psv["reliability"] = psv_reliability
    agg_psv["forward_succeeded"] = True

    # Classifier: aggregated argmax + max + margin (Steps 3)
    agg_classifier = {
        "argmax": final_argmax,
        "max": combined_max,
        "margin": combined_margin,
    }

    # Conformal: aggregated set
    agg_conformal = {
        "set": conformal_set,
        "size": len(conformal_set),
        "tau": None,
    }

    # IQA: worst decision (Step 5)
    agg_iqa = {"decision": iqa_decision}

    return agg_v3, agg_lora, agg_psv, agg_classifier, agg_conformal


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def aggregate_multi_image(
    per_image_inputs: list[PerImageInput],
) -> AggregatedResult:
    """Aggregate per-image TierAssignments into a single final tier.

    Implements the 7-step aggregation strategy from spec 18.4 (lines 6144-6186).

    Args:
        per_image_inputs: List of 1-5 PerImageInput objects (one per uploaded image).
                          Length 1 is valid (degenerates to single-image passthrough).

    Returns:
        AggregatedResult with final_tier, per_image_summaries, warnings, and
        all aggregated signal values.

    spec: 18.4 lines 6144-6186
    DEC-044 Decisions 6, 7.
    """
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Pre-step: N=1 passthrough
    # spec: 18.4 line 6149 — "N=1 multi-image is equivalent to single-image"
    # ------------------------------------------------------------------
    if len(per_image_inputs) == 1:
        img = per_image_inputs[0]
        per_image_summaries = [
            PerImageSummary(
                image_id=img.image_id,
                tier=img.tier_assignment.tier_label,
                primary_class=img.primary_class,
                primary_confidence=img.primary_confidence,
                tier5_alert_fired=img.tier_assignment.tier5_alert,
            )
        ]
        _log.debug("multi_image_single_passthrough", image_id=img.image_id)
        t5_fired, t5_reason, t5_class = _aggregate_t5(per_image_inputs)
        return AggregatedResult(
            final_tier=img.tier_assignment,
            per_image_summaries=per_image_summaries,
            tier5_alert_fired=t5_fired,
            tier5_reason=t5_reason,
            tier5_trigger_class=t5_class,
            warnings=warnings,
            primary_class=img.primary_class,
            combined_max_prob=img.primary_confidence,
            combined_margin=img.combined_margin,
            conformal_set=set(img.conformal_set),
            iqa_decision=img.iqa_decision,
            psv_reliability=img.psv_reliability,
            chilli_leakage=img.chilli_leakage,
            n_successful=0 if _is_failed_image(img) else 1,
            n_rejected=1 if _is_failed_image(img) else 0,
        )

    # ------------------------------------------------------------------
    # Per-image summary (always built regardless of failure state)
    # spec: 18.6 lines 6206-6224
    # ------------------------------------------------------------------
    per_image_summaries = [
        PerImageSummary(
            image_id=img.image_id,
            tier=img.tier_assignment.tier_label,
            primary_class=img.primary_class,
            primary_confidence=img.primary_confidence,
            tier5_alert_fired=img.tier_assignment.tier5_alert,
        )
        for img in per_image_inputs
    ]

    # ------------------------------------------------------------------
    # Step 1: T5 alert aggregation (ALL images, including failed)
    # spec: 18.4 lines 6154-6157 — "ANY image fires → final T5 fires"
    # ------------------------------------------------------------------
    t5_fired, t5_reason, t5_trigger_class = _aggregate_t5(per_image_inputs)
    _log.debug("multi_image_t5_aggregation", fired=t5_fired, reason=t5_reason)

    # ------------------------------------------------------------------
    # Step 2: Pipeline failure / IQA REJECT aggregation
    # spec: 18.4 lines 6159-6161
    # ------------------------------------------------------------------
    failed = [img for img in per_image_inputs if _is_failed_image(img)]
    successful = [img for img in per_image_inputs if not _is_failed_image(img)]
    n_rejected = len(failed)
    n_successful = len(successful)

    if n_successful == 0:
        # All images failed — Tier 4B
        # spec: 18.4 line 6160 — "If ALL per-image tiers are 4B OR all IQA rejected → Tier 4B"
        _log.warning("multi_image_all_failed", n_images=len(per_image_inputs))
        fallback_tier = TierAssignment(tier_label="4B", tier5_alert=t5_fired, rule_id_fired="1")
        return AggregatedResult(
            final_tier=fallback_tier,
            per_image_summaries=per_image_summaries,
            tier5_alert_fired=t5_fired,
            tier5_reason=t5_reason,
            tier5_trigger_class=t5_trigger_class,
            warnings=["All images failed IQA or pipeline — no successful images to aggregate."],
            primary_class=None,
            combined_max_prob=None,
            combined_margin=None,
            conformal_set=set(),
            iqa_decision="ACCEPTABLE",
            psv_reliability=0.0,
            chilli_leakage=0.0,
            n_successful=0,
            n_rejected=n_rejected,
        )

    if n_rejected > 0:
        # spec: 18.4 line 6161 — "failed images excluded; warning added"
        warnings.append(
            f"{n_rejected} image(s) failed IQA or pipeline and were excluded from aggregation."
        )
        _log.debug("multi_image_partial_failure", n_rejected=n_rejected, n_successful=n_successful)

    # ------------------------------------------------------------------
    # Step 3: Class voting
    # spec: 18.4 lines 6163-6168
    # ------------------------------------------------------------------
    final_argmax, combined_max_prob, combined_margin, vote_share = _aggregate_class_vote(successful)

    if final_argmax is None:
        final_argmax = 5   # healthy fallback (defensive; should not reach here)
        combined_max_prob = 0.0
        combined_margin = 0.0

    # Clamp None to safe defaults
    if combined_max_prob is None:
        combined_max_prob = 0.0
    if combined_margin is None:
        combined_margin = 0.0

    # Disagreement detection (spec: 18.4 line 6189)
    # "If top-voted class has vote share < 0.50 → flag disagreement-among-images"
    if vote_share < _DISAGREE_VOTE_SHARE_THRESHOLD:
        warnings.append(
            f"Strong disagreement among images: winning class vote share {vote_share:.2f} < "
            f"{_DISAGREE_VOTE_SHARE_THRESHOLD}. Result may be unreliable."
        )
        _log.debug("multi_image_strong_disagreement", vote_share=vote_share)

    # ------------------------------------------------------------------
    # Step 4: Conformal set aggregation
    # spec: 18.4 lines 6170-6173
    # ------------------------------------------------------------------
    conformal_set = _aggregate_conformal_set(successful)

    # ------------------------------------------------------------------
    # Step 5: Aggregated IQA decision
    # spec: 18.4 lines 6175-6177
    # ------------------------------------------------------------------
    iqa_decision = _worst_iqa_decision(successful)

    # ------------------------------------------------------------------
    # Step 6: Aggregated PSV reliability and chilli leakage
    # spec: 18.4 lines 6179-6182
    # "psv_reliability = minimum across successful images"
    # "chilli_leakage = maximum across successful images"
    # ------------------------------------------------------------------
    psv_reliability = min(
        img.psv_reliability for img in successful
        if math.isfinite(img.psv_reliability)
    ) if successful else 0.0
    chilli_leakage = max(
        img.chilli_leakage for img in successful
        if math.isfinite(img.chilli_leakage)
    ) if successful else 0.0

    # ------------------------------------------------------------------
    # Step 7: Final tier assignment using aggregated values
    # spec: 18.4 lines 6184-6186 + 18.5 line 6193
    # "Run the standard tier rule chain using the aggregated values."
    # "The aggregated values are passed through assign_tier() as if from a single image."
    # ------------------------------------------------------------------
    agg_v3, agg_lora, agg_psv, agg_classifier, agg_conformal = _build_aggregated_signals(
        final_argmax=final_argmax,
        combined_max=combined_max_prob,
        combined_margin=combined_margin,
        conformal_set=conformal_set,
        iqa_decision=iqa_decision,
        psv_reliability=psv_reliability,
        chilli_leakage=chilli_leakage,
        successful=successful,
    )

    final_tier = assign_tier(
        v3_signal=agg_v3,
        lora_signal=agg_lora,
        psv_signal=agg_psv,
        classifier=agg_classifier,
        conformal=agg_conformal,
        iqa={"decision": iqa_decision},
    )

    # Preserve T5 aggregation (Step 1) — OR with assign_tier's own T5 check
    # spec: 18.4 line 6155 — "fired if ANY per-image T5 fires"
    # assign_tier computes T5 freshly from aggregated inputs; we OR with per-image T5 fires.
    if t5_fired and not final_tier.tier5_alert:
        final_tier = TierAssignment(
            tier_label=final_tier.tier_label,
            tier5_alert=True,
            rule_id_fired=final_tier.rule_id_fired,
        )

    _log.debug(
        "multi_image_aggregation_complete",
        final_tier=final_tier.tier_label,
        tier5_alert=final_tier.tier5_alert,
        primary_class=final_argmax,
        combined_max_prob=combined_max_prob,
        n_successful=n_successful,
        n_rejected=n_rejected,
        vote_share=vote_share,
    )

    return AggregatedResult(
        final_tier=final_tier,
        per_image_summaries=per_image_summaries,
        tier5_alert_fired=final_tier.tier5_alert,
        tier5_reason=t5_reason,
        tier5_trigger_class=t5_trigger_class,
        warnings=warnings,
        primary_class=final_argmax,
        combined_max_prob=combined_max_prob,
        combined_margin=combined_margin,
        conformal_set=conformal_set,
        iqa_decision=iqa_decision,
        psv_reliability=psv_reliability,
        chilli_leakage=chilli_leakage,
        n_successful=n_successful,
        n_rejected=n_rejected,
    )


__all__ = [
    "PerImageInput",
    "PerImageSummary",
    "AggregatedResult",
    "aggregate_multi_image",
]
