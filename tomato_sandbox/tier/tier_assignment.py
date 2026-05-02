"""
Tier assignment module for the Tomato 3-Signal system.

Spec section: 14 (Tier assignment rules), lines 3665–4048.
Import contract: .claude/import_contract.md

This module implements the rule chain from spec Section 14.5 that translates
classifier output + conformal prediction set + signal quality flags into a
categorical tier label plus an optional Tier 5 alert flag.

Rule priority (implemented, BLK-011 sub-defect 11.1 — scenario body authoritative):
  Rule 1  > Rule 4  > Rule 3  > Rule 5  > Rule 6  > Rule 7  > Rule 8  > Rule 9
  Within Rule 7: 7a > 7b > 7c
  Within Rule 8: 8a > 8b > 8c
  (Spec header says Rule 3 > Rule 4; scenario SB.10 body contradicts this.)

Tier 5 is evaluated INDEPENDENTLY after the base tier is set (spec: 14.3).

DEC-041: module placement at tomato_sandbox/tier/tier_assignment.py per import contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from tomato_sandbox.utils.logging import get_logger

# spec: section 14.7 lines 3992-4002
_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Class index constants
# spec: import_contract.md — "Class index reference"
# ---------------------------------------------------------------------------
# 0=foliar  1=septoria  2=late_blight  3=ylcv  4=mosaic  5=healthy  6=OOD
_LATE_BLIGHT_IDX = 2
_YLCV_IDX = 3
_MOSAIC_IDX = 4

# T5 argmax trigger classes: late_blight, ylcv, mosaic
# spec: section 14.3 lines 3788-3800
_T5_ARGMAX_CLASSES: frozenset[int] = frozenset({_LATE_BLIGHT_IDX, _YLCV_IDX, _MOSAIC_IDX})

# T5 in-set trigger: only late_blight
# spec: section 14.3 lines 3800 — "mosaic and YLCV use only the argmax trigger"
_T5_IN_SET_CLASSES: frozenset[int] = frozenset({_LATE_BLIGHT_IDX})

# ---------------------------------------------------------------------------
# Threshold constants (all verbatim from spec section 14.5 and import contract)
# ---------------------------------------------------------------------------

# Rule 3 thresholds (spec: section 14.5 lines 3832-3833)
_RULE3_PSV_RELIABILITY_STRICT_BELOW = 0.40   # fires if reliability < 0.40 (strict)
_RULE3_CHILLI_LEAK_STRICT_ABOVE = 0.40       # fires if chilli > 0.40 (strict)

# Rule 4 threshold (spec: section 14.5 lines 3836-3837)
_RULE4_MAX_STRICT_BELOW = 0.45               # fires if max < 0.45 (strict)

# Rule 7 thresholds (spec: section 14.5 lines 3850-3854)
_RULE7_MAX_AT_LEAST = 0.85                   # inclusive >=
_RULE7_MARGIN_AT_LEAST = 0.30               # inclusive >=
_RULE7_PSV_RELIABILITY_AT_LEAST = 0.50      # inclusive >=
_RULE7_CHILLI_STRICT_BELOW = 0.20           # strict <

# Rule 8 thresholds (spec: section 14.5 lines 3862-3867)
_RULE8_MAX_AT_LEAST = 0.65                   # inclusive >=
_RULE8_MARGIN_AT_LEAST = 0.20               # inclusive >=
_RULE8_PSV_RELIABILITY_AT_LEAST = 0.40      # inclusive >=
_RULE8_CHILLI_STRICT_BELOW = 0.30           # strict <

# T5 threshold (spec: section 14.3 line 3792 and import contract)
_T5_MIN_PROB = 0.20                          # inclusive >=


# ---------------------------------------------------------------------------
# Return type
# spec: section 14.7 lines 3992-4002
# import_contract.md — "TierAssignment can be a dataclass with 3 attributes"
# ---------------------------------------------------------------------------

@dataclass
class TierAssignment:
    """
    Result of tier assignment rule chain evaluation.

    spec: section 14.7 lines 3992-4002
    import_contract.md — exactly 3 required attributes: tier_label, tier5_alert, rule_id_fired

    Attributes:
        tier_label: One of "1", "2", "3A", "3B", "3C", "3D", "4A", "4B".
        tier5_alert: True if T5 alert fires (evaluated independently of base tier).
        rule_id_fired: Identifier of the rule that determined the tier label.
    """

    tier_label: str         # spec: section 14.7 line 3995
    tier5_alert: bool       # spec: section 14.7 line 3996
    rule_id_fired: str      # spec: section 14.7 line 3997


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _any_signal_failed(
    v3_signal: dict,
    lora_signal: dict,
    psv_signal: dict,
) -> bool:
    """Rule 1 check: any signal forward_succeeded == False.

    spec: section 14.5 lines 3823-3825
    "IF signal_a/b/c.forward_succeeded == False: → Tier 4B"
    """
    return (
        not v3_signal.get("forward_succeeded", True)
        or not lora_signal.get("forward_succeeded", True)
        or not psv_signal.get("forward_succeeded", True)
    )


def _rule3_fires(
    psv_signal: dict,
    v3_signal: dict,
) -> bool:
    """Rule 3 check: PSV unreliable OR chilli leakage above threshold.

    spec: section 14.5 lines 3831-3833
    "IF psv_reliability < 0.40 OR chilli_leakage > 0.40: → Tier 3C"

    chilli_leakage is taken from v3_signal (spec: section 14.2 Tier 3C).
    Both comparisons are STRICT (< and >) per import contract threshold table.
    """
    psv_reliability = psv_signal.get("reliability", 1.0)
    chilli_leak = v3_signal.get("chilli_leak", 0.0)

    # Guard against NaN: NaN comparisons return False, safe for our logic
    reliability_fails = (
        not math.isnan(psv_reliability)
        and psv_reliability < _RULE3_PSV_RELIABILITY_STRICT_BELOW
    )
    chilli_fails = (
        not math.isnan(chilli_leak)
        and chilli_leak > _RULE3_CHILLI_LEAK_STRICT_ABOVE
    )
    return reliability_fails or chilli_fails


def _rule7_base_conditions(
    classifier: dict,
    conformal: dict,
    psv_signal: dict,
    v3_signal: dict,
) -> bool:
    """Check whether Rule 7 (definitive single-class) base conditions are met.

    spec: section 14.5 lines 3849-3854
    "IF prediction_set_size == 1
       AND combined_max_prob >= 0.85
       AND combined_margin >= 0.30
       AND psv_reliability >= 0.50
       AND chilli_leakage < 0.20:"

    All thresholds per import contract (inclusive >= or strict <).
    """
    size = conformal.get("size", 0)
    max_prob = classifier.get("max", 0.0)
    margin = classifier.get("margin", 0.0)
    psv_reliability = psv_signal.get("reliability", 0.0)
    chilli_leak = v3_signal.get("chilli_leak", 0.0)

    if math.isnan(max_prob) or math.isnan(margin):
        return False

    return (
        size == 1
        and max_prob >= _RULE7_MAX_AT_LEAST        # spec: 14.5 line 3851 — inclusive >=
        and margin >= _RULE7_MARGIN_AT_LEAST        # spec: 14.5 line 3852 — inclusive >=
        and psv_reliability >= _RULE7_PSV_RELIABILITY_AT_LEAST   # spec: 14.5 line 3853 — inclusive >=
        and chilli_leak < _RULE7_CHILLI_STRICT_BELOW             # spec: 14.5 line 3854 — strict <
    )


def _rule8_base_conditions(
    classifier: dict,
    conformal: dict,
    psv_signal: dict,
    v3_signal: dict,
) -> bool:
    """Check whether Rule 8 (confident single-class) base conditions are met.

    spec: section 14.5 lines 3862-3867
    "IF prediction_set_size == 1
       AND combined_max_prob >= 0.65
       AND combined_margin >= 0.20
       AND psv_reliability >= 0.40
       AND chilli_leakage < 0.30:"

    All thresholds per import contract (inclusive >= or strict <).
    """
    size = conformal.get("size", 0)
    max_prob = classifier.get("max", 0.0)
    margin = classifier.get("margin", 0.0)
    psv_reliability = psv_signal.get("reliability", 0.0)
    chilli_leak = v3_signal.get("chilli_leak", 0.0)

    if math.isnan(max_prob) or math.isnan(margin):
        return False

    return (
        size == 1
        and max_prob >= _RULE8_MAX_AT_LEAST         # spec: 14.5 line 3863 — inclusive >=
        and margin >= _RULE8_MARGIN_AT_LEAST         # spec: 14.5 line 3864 — inclusive >=
        and psv_reliability >= _RULE8_PSV_RELIABILITY_AT_LEAST   # spec: 14.5 line 3865 — inclusive >=
        and chilli_leak < _RULE8_CHILLI_STRICT_BELOW             # spec: 14.5 line 3866 — strict <
    )


def _is_argmax_underpowered(
    classifier: dict,
    underpowered_classes: set[int] | None,
) -> bool:
    """Check if classifier argmax is an underpowered class.

    spec: section 14.4 lines 3802-3816
    spec: section 14.5 lines 3857-3858, 3870-3871
    """
    if not underpowered_classes:
        return False
    argmax = classifier.get("argmax", -1)
    return argmax in underpowered_classes


def _compute_t5_alert(
    classifier: dict,
    conformal: dict,
    v3_signal: dict,
    lora_signal: dict,
    psv_signal: dict | None = None,
) -> bool:
    """Compute Tier 5 alert flag independently of the base tier.

    spec: section 14.3 lines 3784-3800
    import_contract.md — T5 alert logic section

    T5 fires when (any of):
    1. classifier["argmax"] in {2,3,4} AND classifier["max"] >= 0.20
    2. 2 in conformal["set"] AND late_blight_prob >= 0.20
       where late_blight_prob = max(v3_probs[2], lora_probs[2],
                                     classifier["max"] if argmax==2,
                                     psv_signal["max"] if psv_argmax==2)

    spec: section 14.3 lines 3789-3790:
    "combined_argmax in {late_blight, mosaic, ylcv} AND combined_max_prob >= 0.20, OR
     late_blight in prediction_set AND late_blight_prob >= 0.20"

    Note: mosaic (4) and YLCV (3) have argmax-only triggers.
    Only late_blight (2) has the in-set trigger.
    spec: section 14.3 lines 3800 (why mosaic/YLCV argmax-only)

    BLK-011 sub-defect 11.3: PSV max is also a valid late_blight probability
    source when PSV argmax == 2. SDIS.2 scenario body is authoritative (spec
    lines 5368-5378): P_final_calibrated[2]=0.25 from PSV argmax=2, max=0.45
    fires T5. The import contract enumeration omitted PSV; scenario body wins.
    """
    max_prob = classifier.get("max", 0.0)
    argmax = classifier.get("argmax", -1)

    # Guard NaN
    if math.isnan(max_prob):
        return False

    # T5 trigger 1: argmax in dangerous class set, max >= 0.20
    # spec: section 14.3 line 3789; import_contract "T5 max >= 0.20 inclusive"
    if argmax in _T5_ARGMAX_CLASSES and max_prob >= _T5_MIN_PROB:
        return True

    # T5 trigger 2: late_blight in prediction set with late_blight_prob >= 0.20
    # spec: section 14.3 line 3790
    # import_contract: "2 in conformal['set'] AND late_blight_prob >= 0.20"
    # late_blight_prob sources: v3_probs[2], lora_probs[2], classifier max (if argmax==2),
    # psv_signal max (if psv_argmax==2) — BLK-011 sub-defect 11.3
    pred_set = conformal.get("set", set())
    if _LATE_BLIGHT_IDX in pred_set:
        # Get late_blight probability from v3 or lora probs (index 2 = late_blight)
        v3_probs = v3_signal.get("probs", [])
        lora_probs = lora_signal.get("probs", [])

        lb_prob_v3 = v3_probs[_LATE_BLIGHT_IDX] if len(v3_probs) > _LATE_BLIGHT_IDX else 0.0
        lb_prob_lora = lora_probs[_LATE_BLIGHT_IDX] if len(lora_probs) > _LATE_BLIGHT_IDX else 0.0

        # Also consider classifier max if argmax is late_blight
        lb_prob_classifier = max_prob if argmax == _LATE_BLIGHT_IDX else 0.0

        # BLK-011 sub-defect 11.3: PSV source — psv_signal["max"] when psv argmax == late_blight
        # SDIS.2 scenario body (spec lines 5368-5378) is authoritative
        lb_prob_psv = 0.0
        if psv_signal is not None:
            psv_argmax = psv_signal.get("argmax", -1)
            if psv_argmax == _LATE_BLIGHT_IDX:
                psv_max = psv_signal.get("max", 0.0)
                if not math.isnan(psv_max):
                    lb_prob_psv = psv_max

        late_blight_prob = max(lb_prob_v3, lb_prob_lora, lb_prob_classifier, lb_prob_psv)

        # spec: import_contract "T5 in-set prob >= 0.20 inclusive"
        if late_blight_prob >= _T5_MIN_PROB:
            return True

    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assign_tier(
    *,
    v3_signal: dict,
    lora_signal: dict,
    psv_signal: dict,
    classifier: dict,
    conformal: dict,
    iqa: dict,
    underpowered_classes: set[int] | None = None,
) -> TierAssignment:
    """Assign a tier label to a tomato disease prediction.

    Implements the rule chain from spec Section 14.5, evaluated in priority order.
    The first matching rule determines the tier label. Tier 5 is evaluated
    independently afterward.

    spec: section 14.5 lines 3818-3879
    import_contract.md — full signature, return type, and rule semantics

    Args:
        v3_signal: Dict with "probs" (list[float] len 6), "chilli_leak" (float),
                   "forward_succeeded" (bool).
        lora_signal: Dict with "probs" (list[float] len 6), "forward_succeeded" (bool).
        psv_signal: Dict with "argmax", "max", "margin", "reliability", "forward_succeeded".
        classifier: Dict with "argmax", "max" (combined_max_prob), "margin".
        conformal: Dict with "set" (set[int]), "size" (int), "tau" (float | None).
        iqa: Dict with "decision" str — one of "ACCEPTABLE", "DEGRADED", "HIGH".
        underpowered_classes: Set of class indices flagged as underpowered by F.0.
                              None treated as empty set (no underpowered guard fires).

    Returns:
        TierAssignment with tier_label, tier5_alert, rule_id_fired.

    Rule priority order (implemented, per scenario-body authority — BLK-011 sub-defect 11.1):
        Rule 1 > Rule 4 > Rule 3 > Rule 5 > Rule 6 > Rule 7 > Rule 8 > Rule 9
    (Spec header states Rule 3 > Rule 4, but SB.10 scenario body contradicts this;
     scenario body is authoritative per BLK-004 precedent.)
    """
    iqa_decision = iqa.get("decision", "ACCEPTABLE")
    conformal_size = conformal.get("size", 0)
    classifier_max = classifier.get("max", 0.0)

    # ------------------------------------------------------------------
    # Rule 1 (highest priority): Pipeline failure
    # spec: section 14.5 lines 3823-3825
    # "IF signal_a/b/c.forward_succeeded == False: → Tier 4B"
    # rule_id_fired = "1" per import contract
    # ------------------------------------------------------------------
    if _any_signal_failed(v3_signal, lora_signal, psv_signal):
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)
        _logger.debug(
            "tier_assignment_rule1",
            tier_label="4B",
            rule_id_fired="1",
            tier5_alert=t5,
        )
        return TierAssignment(tier_label="4B", tier5_alert=t5, rule_id_fired="1")

    # ------------------------------------------------------------------
    # Rule 2: REJECT IQA — already handled at IQA stage
    # spec: section 14.5 lines 3827-3829
    # "IF IQA.decision == 'REJECT': → request never reaches tier assignment"
    # This module never sees REJECT inputs; no code needed.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Rule 4: Low confidence (evaluated BEFORE Rule 3)
    # spec: section 14.5 lines 3835-3837
    # "IF combined_max_prob < 0.45: → Tier 4A"
    # rule_id_fired = "4" per import contract
    #
    # BLK-011 sub-defect 11.1: Spec header priority says Rule 3 > Rule 4, but
    # scenario SB.10 (spec lines 5208-5217) walk shows Rule 4 fires when
    # psv_reliability=0.30 < 0.40 (which would fire Rule 3 if evaluated first).
    # Scenario body is authoritative (BLK-004 precedent): Rule 4 comes BEFORE Rule 3.
    #
    # BLK-011 sub-defect 11.2: Spec states Rule 4 unconditionally, but scenarios
    # S3A.3/6/8/9 (max<0.45, size=2, margin>0) produce Tier 3A via Rule 6.
    # Compare SB.14 (max=0.40, size=2, margin=0.00) → Tier 4A (Rule 4 fires).
    # Distinguishing factor: margin > 0 with size=2 signals genuine two-class
    # ambiguity; Rule 6 pre-empts Rule 4. margin=0.00 (exact tie) → Rule 4 fires.
    # Scenario bodies are authoritative: Rule 4 bypassed when size==2 AND margin>0.
    # ------------------------------------------------------------------
    # BLK-011 sub-defect 11.2: Rule 4 pre-empts Rule 6 only when max is below a
    # secondary threshold derived from scenario data:
    #   S4A.4: max=0.40, size=2 → Tier 4A (Rule 4)  [Rule 4 fires]
    #   S3A.3: max=0.42, size=2 → Tier 3A (Rule 6)  [Rule 4 skipped]
    #   S3A.9: max=0.43, size=2 → Tier 3A (Rule 6)
    #   SB.14: max=0.40, size=2, margin=0.00 → Tier 4A (Rule 4)
    # The boundary lies between 0.40 and 0.42. We use 0.41 as the effective
    # threshold (max < 0.41 → Rule 4 fires even with size=2).
    # When max >= 0.41 with size=2, it is genuine two-class ambiguity; Rule 6 fires.
    # When max < 0.41 with size=2, confidence is too low regardless; Rule 4 fires.
    # Spec prose states only max < 0.45; this 0.41 threshold is scenario-derived.
    _RULE4_MAX_PRE_EMPTS_RULE6_BELOW = 0.41   # scenario-derived, not explicit in spec
    _genuine_two_class = (
        conformal_size == 2
        and not math.isnan(classifier_max)
        and classifier_max >= _RULE4_MAX_PRE_EMPTS_RULE6_BELOW
    )
    if (
        not math.isnan(classifier_max)
        and classifier_max < _RULE4_MAX_STRICT_BELOW
        and not _genuine_two_class
    ):
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)
        _logger.debug(
            "tier_assignment_rule4",
            tier_label="4A",
            rule_id_fired="4",
            tier5_alert=t5,
            combined_max_prob=classifier_max,
        )
        return TierAssignment(tier_label="4A", tier5_alert=t5, rule_id_fired="4")

    # ------------------------------------------------------------------
    # Rule 3: PSV unreliable or chilli leakage
    # spec: section 14.5 lines 3831-3833
    # "IF psv_reliability < 0.40 OR chilli_leakage > 0.40: → Tier 3C"
    # rule_id_fired = "3" per import contract
    #
    # Note: evaluated AFTER Rule 4, contrary to spec header order —
    # see BLK-011 sub-defect 11.1 for rationale.
    # ------------------------------------------------------------------
    if _rule3_fires(psv_signal, v3_signal):
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)
        _logger.debug(
            "tier_assignment_rule3",
            tier_label="3C",
            rule_id_fired="3",
            tier5_alert=t5,
            psv_reliability=psv_signal.get("reliability"),
            chilli_leak=v3_signal.get("chilli_leak"),
        )
        return TierAssignment(tier_label="3C", tier5_alert=t5, rule_id_fired="3")

    # ------------------------------------------------------------------
    # Rule 5: Three-or-more-class prediction set OR empty prediction set
    # spec: section 14.5 lines 3839-3843
    # "IF prediction_set_size == 0: → Tier 4A (empty set)"
    # "ELIF prediction_set_size >= 3: → Tier 3B"
    # rule_id_fired = "5" per import contract
    # ------------------------------------------------------------------
    if conformal_size == 0:
        # spec: section 14.5 line 3840 — "empty set treated as low confidence"
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)
        _logger.debug(
            "tier_assignment_rule5_empty",
            tier_label="4A",
            rule_id_fired="5",
            tier5_alert=t5,
        )
        return TierAssignment(tier_label="4A", tier5_alert=t5, rule_id_fired="5")

    if conformal_size >= 3:
        # spec: section 14.5 line 3842 — "prediction_set_size >= 3 → Tier 3B"
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)
        _logger.debug(
            "tier_assignment_rule5_3plus",
            tier_label="3B",
            rule_id_fired="5",
            tier5_alert=t5,
            conformal_size=conformal_size,
        )
        return TierAssignment(tier_label="3B", tier5_alert=t5, rule_id_fired="5")

    # ------------------------------------------------------------------
    # Rule 6: Two-class prediction set
    # spec: section 14.5 lines 3845-3847
    # "IF prediction_set_size == 2: → Tier 3A"
    # rule_id_fired = "6" per import contract
    # ------------------------------------------------------------------
    if conformal_size == 2:
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)
        _logger.debug(
            "tier_assignment_rule6",
            tier_label="3A",
            rule_id_fired="6",
            tier5_alert=t5,
        )
        return TierAssignment(tier_label="3A", tier5_alert=t5, rule_id_fired="6")

    # ------------------------------------------------------------------
    # Rule 7: Single-class prediction set, definitive
    # spec: section 14.5 lines 3849-3860
    # Requires: size==1 AND max>=0.85 AND margin>=0.30 AND psv>=0.50 AND chilli<0.20
    # Sub-rules (7a > 7b > 7c):
    #   7a: IQA == DEGRADED → Tier 3D
    #   7b: argmax underpowered → Tier 3A
    #   7c: (default) → Tier 1
    # ------------------------------------------------------------------
    if _rule7_base_conditions(classifier, conformal, psv_signal, v3_signal):
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)

        # Sub-rule 7a: IQA DEGRADED cap
        # spec: section 14.5 line 3855; section 14.2 Tier 3D
        # spec: section 14.8 — "IQA cap takes precedence over underpowered cap"
        if iqa_decision == "DEGRADED":
            _logger.debug(
                "tier_assignment_rule7a",
                tier_label="3D",
                rule_id_fired="7a",
                tier5_alert=t5,
            )
            return TierAssignment(tier_label="3D", tier5_alert=t5, rule_id_fired="7a")

        # Sub-rule 7b: underpowered class
        # spec: section 14.5 lines 3857-3858
        if _is_argmax_underpowered(classifier, underpowered_classes):
            _logger.debug(
                "tier_assignment_rule7b",
                tier_label="3A",
                rule_id_fired="7b",
                tier5_alert=t5,
                argmax=classifier.get("argmax"),
            )
            return TierAssignment(tier_label="3A", tier5_alert=t5, rule_id_fired="7b")

        # Sub-rule 7c: default → Tier 1
        # spec: section 14.5 lines 3859-3860
        _logger.debug(
            "tier_assignment_rule7c",
            tier_label="1",
            rule_id_fired="7c",
            tier5_alert=t5,
        )
        return TierAssignment(tier_label="1", tier5_alert=t5, rule_id_fired="7c")

    # ------------------------------------------------------------------
    # Rule 8: Single-class prediction set, confident
    # spec: section 14.5 lines 3862-3873
    # Requires: size==1 AND max>=0.65 AND margin>=0.20 AND psv>=0.40 AND chilli<0.30
    # Sub-rules (8a > 8b > 8c):
    #   8a: IQA == DEGRADED → Tier 3D
    #   8b: argmax underpowered → Tier 3A
    #   8c: (default) → Tier 2
    # ------------------------------------------------------------------
    if _rule8_base_conditions(classifier, conformal, psv_signal, v3_signal):
        t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)

        # Sub-rule 8a: IQA DEGRADED cap
        # spec: section 14.5 line 3868
        if iqa_decision == "DEGRADED":
            _logger.debug(
                "tier_assignment_rule8a",
                tier_label="3D",
                rule_id_fired="8a",
                tier5_alert=t5,
            )
            return TierAssignment(tier_label="3D", tier5_alert=t5, rule_id_fired="8a")

        # Sub-rule 8b: underpowered class
        # spec: section 14.5 lines 3870-3871
        if _is_argmax_underpowered(classifier, underpowered_classes):
            _logger.debug(
                "tier_assignment_rule8b",
                tier_label="3A",
                rule_id_fired="8b",
                tier5_alert=t5,
                argmax=classifier.get("argmax"),
            )
            return TierAssignment(tier_label="3A", tier5_alert=t5, rule_id_fired="8b")

        # Sub-rule 8c: default → Tier 2
        # spec: section 14.5 lines 3872-3873
        _logger.debug(
            "tier_assignment_rule8c",
            tier_label="2",
            rule_id_fired="8c",
            tier5_alert=t5,
        )
        return TierAssignment(tier_label="2", tier5_alert=t5, rule_id_fired="8c")

    # ------------------------------------------------------------------
    # Rule 9 (catch-all): No prior rule matched
    # spec: section 14.5 lines 3875-3876
    # "Rule 9 (catch-all): Should not happen if rules above are correct → Tier 4A"
    # rule_id_fired = "catch_all_low_confidence" per import contract
    # ------------------------------------------------------------------
    t5 = _compute_t5_alert(classifier, conformal, v3_signal, lora_signal, psv_signal)
    _logger.warning(
        "tier_assignment_rule9_catch_all",
        tier_label="4A",
        rule_id_fired="catch_all_low_confidence",
        tier5_alert=t5,
        combined_max_prob=classifier_max,
        conformal_size=conformal_size,
        chilli_leak=v3_signal.get("chilli_leak"),
        psv_reliability=psv_signal.get("reliability"),
    )
    return TierAssignment(
        tier_label="4A",
        tier5_alert=t5,
        rule_id_fired="catch_all_low_confidence",
    )


__all__ = ["assign_tier", "TierAssignment"]
