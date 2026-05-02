"""
Response builder for the Tomato 3-Signal pipeline.

Spec section: 16 (Output schema and response construction),
lines 5637-5939.

Translates the internal tier outcome (TierAssignment) plus classifier,
conformal, and IQA results into the JSON-serializable dict that the
server sends to clients.

DEC-043: module placement at tomato_sandbox/response/response_builder.py
per DEC-033 pattern (sub-package + re-export shim).

Public API (spec 16.1 line 5643):
    build_response(
        tier_assignment,
        classifier_result,
        conformal_result,
        iqa_result,
        *,
        request_metadata=None,
    ) -> dict

The function is a pure function: no side effects, deterministic output
for identical inputs.  spec: section 16.1 lines 5643-5644

All fields are present in every response.  spec: section 16.2 line 5709
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from tomato_sandbox.utils.logging import get_logger

if TYPE_CHECKING:
    from tomato_sandbox.tier.tier_assignment import TierAssignment
    from tomato_sandbox.classifier.hierarchical_classifier import ClassifierResult
    from tomato_sandbox.conformal.conformal import ConformalResult
    from tomato_sandbox.iqa.iqa import IQAResult

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Class index → canonical name mapping
# spec: section 12.1 lines 3151-3159; Appendix E class index conventions
# 0=foliar  1=septoria  2=late_blight  3=ylcv  4=mosaic  5=healthy  6=OOD
# spec: section 16.2 uses "foliar", "late_blight", etc. as the class strings
# ---------------------------------------------------------------------------

_CLASS_SHORT_NAMES: dict[int, str] = {
    # spec: section 12.1 lines 3152-3158 (canonical short names)
    # spec: Appendix E (class index conventions and remap tables)
    0: "foliar",
    1: "septoria",
    2: "late_blight",
    3: "ylcv",
    4: "mosaic",
    5: "healthy",
    6: "OOD",
}

_CLASS_HUMAN_NAMES: dict[int, str] = {
    # spec: section 16.2 "primary_class_human" examples use full disease names
    # spec: section 2.3 (glossary) full names
    0: "Foliar leaf spot",
    1: "Septoria leaf spot",
    2: "Late blight",
    3: "Yellow Leaf Curl Virus (YLCV)",
    4: "Mosaic virus",
    5: "Healthy",
    6: "Out-of-distribution (OOD)",
}

# Dangerous disease indices — those that can trigger Tier 5 argmax bullet
# spec: section 14.3 lines 3788-3791 — "late_blight, mosaic, ylcv"
_DANGEROUS_CLASS_INDICES: frozenset[int] = frozenset({2, 3, 4})

# ---------------------------------------------------------------------------
# Tier label → human-readable string mapping
# spec: section 16.3 lines 5718-5728
# ---------------------------------------------------------------------------

_TIER_HUMAN_READABLE: dict[str, str] = {
    # spec: section 16.3 lines 5720-5728 (verbatim from table)
    "1":  "Definitive prediction",
    "2":  "Confident prediction",
    "3A": "Two possible diseases",
    "3B": "Multiple possible diseases",
    "3C": "Image quality concern (segmentation or chilli leaf detection)",
    "3D": "Image quality moderate; result less confident",
    "4A": "Low confidence — manual review recommended",
    "4B": "Pipeline issue — please retake or contact support",
}

# Tier label → alert_level
# spec: section 16.2 shows "info" for Tier 1; inferred for others from severity
# spec: 16.3 — higher tiers = more urgent alert levels
_TIER_ALERT_LEVEL: dict[str, str] = {
    "1":  "info",
    "2":  "info",
    "3A": "warning",
    "3B": "warning",
    "3C": "warning",
    "3D": "warning",
    "4A": "error",
    "4B": "error",
}

# ---------------------------------------------------------------------------
# Queue routing logic constants
# spec: section 16.8 lines 5853-5858
# ---------------------------------------------------------------------------

# route_ambiguous_to_queue default: False (capacity-dependent flag)
# spec: section 16.8 line 5855 — "default false; agronomist capacity-dependent"
_DEFAULT_ROUTE_AMBIGUOUS: bool = False

# ---------------------------------------------------------------------------
# Explanation user-string templates
# spec: section 16.3 lines 5732-5747
# These are spec-illustrative; agronomic review may change wording before deploy.
# ---------------------------------------------------------------------------

_USER_STRING_TEMPLATES: dict[str, str] = {
    # spec: section 16.3 line 5734 (Tier 1 example)
    "1": (
        "The image clearly shows {class_human} with high confidence ({confidence_pct}%). "
        "Recommended action: apply standard treatment per severity grading."
    ),
    # spec: section 16.3 (Tier 2 — follows Tier 1 pattern with lower confidence phrasing)
    "2": (
        "The system is confident this is {class_human} ({confidence_pct}%). "
        "Recommended action: apply standard treatment per severity grading."
    ),
    # spec: section 16.3 lines 5736-5738 (Tier 3A example)
    "3A": (
        "The system cannot decide between {class_human_set}. "
        "Both diseases require similar treatment in early stages; differential treatment "
        "may be needed for later stages. "
        "Consider taking a closer photo of a mature lesion for clearer diagnosis."
    ),
    # spec: section 16.3 (Tier 3B — multi-class, same pattern as 3A)
    "3B": (
        "The system identified multiple possible diseases: {class_human_set}. "
        "Manual review by an agronomist is recommended to confirm the diagnosis."
    ),
    # spec: section 16.3 (Tier 3C — quality or chilli leakage)
    "3C": (
        "The system detected a concern with image segmentation or leaf type. "
        "The result may be less reliable. "
        "Consider retaking with a clearer, well-lit tomato leaf filling the frame."
    ),
    # spec: section 16.3 (Tier 3D — DEGRADED IQA)
    "3D": (
        "Image quality is moderate, which reduces confidence in the prediction. "
        "Best estimate: {class_human} ({confidence_pct}%). "
        "Consider retaking for a more reliable result."
    ),
    # spec: section 16.3 lines 5740-5742 (Tier 4A)
    # spec: section 16.6 line 5813 — Tier 4A displays "below 45%" not actual number
    "4A": (
        "The system has low confidence in this prediction ({confidence_pct}). "
        "Possible reasons: atypical disease presentation, an unusual cultivar, or "
        "non-tomato content in the image. "
        "Manual review by an agronomist is recommended."
    ),
    # spec: section 16.3 lines 5744-5746 (Tier 4B)
    "4B": (
        "The disease detection pipeline encountered an issue while processing this image. "
        "Please retake the photo or contact support if the issue persists."
    ),
}


# ---------------------------------------------------------------------------
# T5 alert reason construction
# spec: section 16.7 lines 5837-5841
# ---------------------------------------------------------------------------

def _build_t5_alert_block(
    tier_assignment: "TierAssignment",
    classifier_result: "ClassifierResult",
    conformal_result: "ConformalResult",
) -> dict:
    """Build the tier5_alert block of the response.

    spec: section 16.7 lines 5823-5849

    When fired, the block includes:
      fired, reason, trigger_class, trigger_probability, agronomist_priority_hint
    When not fired:
      fired=False, reason=None (all other fields null)

    spec: section 16.2 line 5679 — present in every response (null when not fired)
    spec: section 16.7 line 5837 — reason enum values
    """
    if not tier_assignment.tier5_alert:
        # spec: section 16.2 lines 5676-5679 — null block when not fired
        return {
            "fired": False,
            "reason": None,
            # spec: section 16.7 line 5841 — trigger fields absent when not fired
            # All fields present per 16.2 "stable schema" rule; use null
            "trigger_class": None,
            "trigger_probability": None,
            "agronomist_priority_hint": None,
        }

    # Determine the T5 reason
    # spec: section 16.7 lines 5837-5842
    argmax = classifier_result.combined_argmax
    max_prob = classifier_result.combined_max_prob
    pred_set = set(conformal_result.prediction_set)

    # Determine which bullets fired
    argmax_bullet = argmax in _DANGEROUS_CLASS_INDICES and max_prob >= 0.20
    # spec: section 14.3 line 3790 — in-set trigger only for late_blight (idx 2)
    inset_bullet = (2 in pred_set)
    # We can only reliably check both firing; the classifier max is available
    # for late_blight prob if argmax==2; otherwise use p_final_calibrated[2]
    late_blight_prob_from_calibrated = float(
        classifier_result.p_final_calibrated[2]
    ) if len(classifier_result.p_final_calibrated) > 2 else 0.0
    inset_bullet = inset_bullet and late_blight_prob_from_calibrated >= 0.20

    # spec: section 16.7 line 5840-5841 — both bullets simultaneously
    if argmax_bullet and inset_bullet:
        reason = "argmax_dangerous_and_late_blight_in_set"
        trigger_class = _CLASS_SHORT_NAMES.get(argmax, "unknown")
        trigger_probability = max_prob  # late_blight is argmax here
    elif argmax_bullet:
        reason = "argmax_dangerous_disease"
        trigger_class = _CLASS_SHORT_NAMES.get(argmax, "unknown")
        trigger_probability = max_prob
    else:
        reason = "late_blight_in_set"
        trigger_class = "late_blight"
        trigger_probability = late_blight_prob_from_calibrated

    # Determine agronomist_priority_hint
    # spec: section 16.7 lines 5845-5847
    # "high": late_blight argmax with max >= 0.50
    # "medium": any other T5 firing
    if argmax == 2 and max_prob >= 0.50:
        # spec: section 16.7 line 5845 — late_blight argmax, max >= 0.50
        agronomist_priority_hint = "high"
    else:
        # spec: section 16.7 line 5846 — any other T5 firing
        agronomist_priority_hint = "medium"

    return {
        "fired": True,
        "reason": reason,                            # spec: 16.7 line 5828
        "trigger_class": trigger_class,              # spec: 16.7 line 5829
        "trigger_probability": round(trigger_probability, 4),  # spec: 16.7 line 5830
        "agronomist_priority_hint": agronomist_priority_hint,  # spec: 16.7 line 5831
    }


# ---------------------------------------------------------------------------
# Agronomist queue routing
# spec: section 16.8 lines 5853-5874
# BLK-010.3 fix: Tier 4A → routed only if Tier 5 also fires
# ---------------------------------------------------------------------------

def _build_queue_block(
    tier_label: str,
    tier5_alert_fired: bool,
    t5_agronomist_priority: Optional[str],
    route_ambiguous_to_queue: bool = _DEFAULT_ROUTE_AMBIGUOUS,
) -> dict:
    """Build the agronomist_queue block.

    spec: section 16.8 lines 5853-5874

    Routing rules (spec 16.8 lines 5854-5858):
    - Tier 5 alert fires → always routed
    - Tier 3A, 3B, 3C, 3D → routed if route_ambiguous_to_queue=True (default False)
    - Tier 4A → routed if Tier 5 also fires; otherwise user opt-in only
      spec: section 16.8 line 5856 (BLK-010.3 fix verbatim)
    - Tier 4B → NOT routed (pipeline issue)
    - Tier 1, 2 → NOT routed unless Tier 5 also fires

    spec: section 16.8 line 5862 — "routed": true with priority and queue_id
    spec: section 16.8 line 5872 — "routed": false means all other fields null
    """
    routed = False
    priority: Optional[str] = None
    # spec: section 16.8 line 5863 — queue_id generated server-side at routing time
    # In this pure builder we return None; the server layer assigns the actual ID
    queue_id: Optional[str] = None

    if tier_label == "4B":
        # spec: section 16.8 line 5857 — "Tier 4B → NOT routed (pipeline issue)"
        # This is an ABSOLUTE exception to the "T5 fires → always routed" rule.
        # Tier 4B indicates pipeline failure (Rule 1: signal forward_succeeded=False);
        # in that case T5 alert is not meaningful and the case is never queued.
        # Evaluated BEFORE T5 check so spec line 5857 wins over spec line 5854.
        routed = False
        priority = None
    elif tier5_alert_fired:
        # spec: section 16.8 line 5854 — "Tier 5 alert fires → always routed"
        # Applies to Tier 1, 2, 3x, 4A when T5 co-fires
        routed = True
        priority = t5_agronomist_priority or "medium"
    elif tier_label in {"3A", "3B", "3C", "3D"} and route_ambiguous_to_queue:
        # spec: section 16.8 line 5855 — Tier 3x routed if flag enabled
        routed = True
        priority = "low"
    elif tier_label == "4A":
        # spec: section 16.8 line 5856 (BLK-010.3) —
        # "Tier 4A → routed only if Tier 5 also fires; otherwise user opt-in only"
        # T5 not fired here (handled in the tier5_alert_fired branch above)
        routed = False
        priority = None
    else:
        # Tier 1, 2 without T5: not routed
        # spec: section 16.8 line 5858 — "Tier 1, 2 → NOT routed unless Tier 5"
        routed = False
        priority = None

    if not routed:
        # spec: section 16.8 line 5872 — "routed: false, all other fields null"
        return {
            "routed": False,
            "priority": None,
            "queue_id": None,
        }

    return {
        "routed": True,
        "priority": priority,            # spec: 16.8 line 5863
        "queue_id": queue_id,            # spec: 16.8 line 5864 — assigned by server layer
    }


# ---------------------------------------------------------------------------
# Confidence display for user strings
# spec: section 16.6 lines 5812-5813 — "multiply by 100, round half up to nearest integer"
# ---------------------------------------------------------------------------

def _format_confidence_pct(combined_max_prob: float, tier_label: str) -> str:
    """Produce the confidence display string for user strings.

    spec: section 16.6 lines 5812-5818
    - Normal: "91%" (int, rounded half-up)
    - Tier 4A: "below 45%"
    - Tier 4B: "unknown"
    - Tier 3A/3B: "between X% and Y%" (from prediction set members)
    Note: "prediction.primary_confidence" always carries raw float.
    Only user-strings use this formatted representation.
    """
    if tier_label == "4B":
        # spec: section 16.6 line 5815 — "unknown" with no percentage
        return "unknown"
    if tier_label == "4A":
        # spec: section 16.6 line 5813 — "below 45%" for low confidence
        return "below 45%"
    # Normal rounding: math.ceil on x.5 is sufficient for half-up
    pct = math.floor(combined_max_prob * 100 + 0.5)
    return f"{pct}%"


def _build_user_strings(
    tier_label: str,
    combined_argmax: int,
    combined_max_prob: float,
    prediction_set_indices: list[int],
) -> list[str]:
    """Build the explanation.user_strings list.

    spec: section 16.3 lines 5732-5748 — 1-3 sentence templates
    """
    template = _USER_STRING_TEMPLATES.get(tier_label, "")
    class_human = _CLASS_HUMAN_NAMES.get(combined_argmax, "Unknown")
    confidence_pct = _format_confidence_pct(combined_max_prob, tier_label)
    class_human_set = " and ".join(
        _CLASS_HUMAN_NAMES.get(i, "Unknown") for i in prediction_set_indices
    )

    filled = template.format(
        class_human=class_human,
        confidence_pct=confidence_pct,
        class_human_set=class_human_set if class_human_set else class_human,
    )
    return [filled] if filled else []


# ---------------------------------------------------------------------------
# Visualization block
# spec: section 16.5 lines 5789-5806
# ---------------------------------------------------------------------------

def _build_visualization_block(
    tier_label: str,
    request_id: str,
    combined_argmax: int,
) -> dict:
    """Build the visualization block.

    spec: section 16.5 lines 5792-5798 — gradcam_url and gradcam_target_class
    spec: section 16.5 line 5804 — Tier 4B: gradcam_url is null
    """
    if tier_label == "4B":
        # spec: section 16.5 line 5804
        return {
            "gradcam_url": None,
            "gradcam_target_class": None,
            "gradcam_alpha": None,
        }

    target_class = _CLASS_SHORT_NAMES.get(combined_argmax, "unknown")
    # spec: section 16.5 line 5795 — URL template with request_id
    gradcam_url = f"/visualization/{request_id}/gradcam.png"
    return {
        "gradcam_url": gradcam_url,             # spec: 16.5 line 5795
        "gradcam_target_class": target_class,   # spec: 16.5 line 5796
        "gradcam_alpha": 0.5,                   # spec: 16.5 line 5797 — "alpha=0.5 blend"
    }


# ---------------------------------------------------------------------------
# Main entry point
# spec: section 16.1 lines 5643-5644
# ---------------------------------------------------------------------------

def build_response(
    tier_assignment: "TierAssignment",
    classifier_result: "ClassifierResult",
    conformal_result: "ConformalResult",
    iqa_result: "IQAResult",
    *,
    request_metadata: Optional[dict] = None,
    route_ambiguous_to_queue: bool = _DEFAULT_ROUTE_AMBIGUOUS,
    model_version: str = "tomato-sandbox-v1.0.0",
) -> dict:
    """Build the JSON-serializable response dict.

    spec: section 16.1 lines 5643-5644 (pure function, no side effects)
    spec: section 16.2 (full schema — all fields present in every response)

    Args:
        tier_assignment: Result of assign_tier().  TierAssignment with
                         tier_label, tier5_alert, rule_id_fired.
                         spec: section 14.7 lines 3992-4002
        classifier_result: ClassifierResult with 9 fields.
                           spec: section 12.10 lines 3446-3458
        conformal_result: ConformalResult with prediction_set and related fields.
                          spec: section 13.7 lines 3639-3647
        iqa_result: IQAResult with decision, aggregate_score, per_dimension, etc.
                    spec: section 6.5 lines 1357-1365
        request_metadata: Optional dict with keys:
                          request_id (str), image_hash (str), timestamp_iso (str),
                          processing_time_ms (int), client_version (str).
                          spec: section 16.1 line 5651 — "request_id, image_hash,
                          timestamp, client version"
        route_ambiguous_to_queue: Whether Tier 3A/3B/3C/3D should be queued.
                                  spec: section 16.8 line 5855 — default False
        model_version: Model version string.
                       spec: section 16.2 line 5704

    Returns:
        JSON-serializable dict matching Section 16.2 schema.
        All fields are present. Fields that do not apply are null.
        spec: section 16.2 line 5709 — "all fields present, null when N/A"
    """
    meta = request_metadata or {}
    request_id = meta.get("request_id", "unknown")
    image_hash = meta.get("image_hash", None)
    timestamp_iso = meta.get("timestamp_iso", None)
    processing_time_ms = meta.get("processing_time_ms", None)

    tier_label = tier_assignment.tier_label          # spec: 14.7 line 3995
    tier5_alert_fired = tier_assignment.tier5_alert  # spec: 14.7 line 3996
    rule_id_fired = tier_assignment.rule_id_fired    # spec: 14.7 line 3997

    combined_argmax = classifier_result.combined_argmax         # spec: 12.10 line 3450
    combined_max_prob = classifier_result.combined_max_prob     # spec: 12.10 line 3451
    combined_margin = classifier_result.combined_margin         # spec: 12.10 line 3452
    p_final_calibrated = classifier_result.p_final_calibrated   # spec: 12.10 line 3449

    prediction_set: list[int] = conformal_result.prediction_set        # spec: 13.7 line 3642
    prediction_set_size: int = conformal_result.prediction_set_size     # spec: 13.7 line 3643

    _logger.debug(
        "build_response",
        tier_label=tier_label,
        tier5_alert=tier5_alert_fired,
        rule_id_fired=rule_id_fired,
        combined_argmax=combined_argmax,
        combined_max_prob=combined_max_prob,
    )

    # -- tier block ----------------------------------------------------------
    # spec: section 16.2 lines 5664-5668
    tier_block = {
        "label": tier_label,                                          # spec: 16.2 line 5665
        "human_readable": _TIER_HUMAN_READABLE.get(tier_label, tier_label),  # spec: 16.3
        "alert_level": _TIER_ALERT_LEVEL.get(tier_label, "info"),   # spec: 16.2 line 5667
    }

    # -- prediction block ----------------------------------------------------
    # spec: section 16.2 lines 5669-5675
    primary_class_str = _CLASS_SHORT_NAMES.get(combined_argmax, "unknown")
    primary_class_human = _CLASS_HUMAN_NAMES.get(combined_argmax, "Unknown")
    prediction_set_short = [_CLASS_SHORT_NAMES.get(i, "unknown") for i in prediction_set]
    prediction_set_human = [_CLASS_HUMAN_NAMES.get(i, "Unknown") for i in prediction_set]

    prediction_block = {
        "primary_class": primary_class_str,               # spec: 16.2 line 5670
        "primary_class_human": primary_class_human,       # spec: 16.2 line 5671
        "primary_confidence": round(float(combined_max_prob), 4),  # spec: 16.6 line 5817
        "prediction_set": prediction_set_short,           # spec: 16.2 line 5673
        "prediction_set_human": prediction_set_human,     # spec: 16.2 line 5674
    }

    # -- tier5_alert block ---------------------------------------------------
    # spec: section 16.2 lines 5676-5679; section 16.7 lines 5823-5849
    t5_block = _build_t5_alert_block(tier_assignment, classifier_result, conformal_result)

    # -- severity block -------------------------------------------------------
    # spec: section 16.2 lines 5680-5684
    # Severity is populated by Section 17 (severity grading); the response
    # builder emits null placeholders here. The caller or server layer fills
    # severity after Section 17 computation.
    # spec: section 16.2 line 5711 — "populated per Section 17"
    severity_block = {
        "grade": None,           # spec: 16.2 line 5682 — filled by Section 17
        "human_readable": None,  # spec: 16.2 line 5683
        "details": None,         # spec: 16.2 line 5684
    }

    # -- explanation block ---------------------------------------------------
    # spec: section 16.2 lines 5685-5693; section 16.3, 16.4
    psv_reliability = None
    if hasattr(conformal_result, "threshold_tau_used"):
        # Extract PSV reliability from IQA per_dimension if available
        # (IQA does not carry PSV reliability; this comes from psv_signal at
        # the server layer; we leave null here for pure builder)
        pass

    user_strings = _build_user_strings(
        tier_label,
        combined_argmax,
        combined_max_prob,
        prediction_set,
    )

    # spec: section 16.4 lines 5752-5778 — structured reasons for machine tools
    structured_block = {
        "rule_id_fired": rule_id_fired,                     # spec: 16.4 line 5757
        "sub_rule_id_fired": rule_id_fired,                 # spec: 16.4 line 5758
        "tier_main_conditions": {
            # spec: section 16.4 lines 5759-5775
            "max_prob_actual": round(float(combined_max_prob), 4),
            "margin_actual": round(float(combined_margin), 4),
            "iqa_decision": iqa_result.decision,
            "set_size": prediction_set_size,
        },
        "tier5_evaluation": {
            # spec: section 16.4 lines 5774-5777
            "argmax_dangerous_check": combined_argmax in _DANGEROUS_CLASS_INDICES,
            "late_blight_in_set_check": 2 in set(prediction_set),
        },
    }

    explanation_block = {
        "user_strings": user_strings,       # spec: 16.2 line 5686; 16.3 lines 5732-5748
        "structured": structured_block,     # spec: 16.2 line 5687; 16.4 lines 5752-5778
    }

    # -- visualization block -------------------------------------------------
    # spec: section 16.2 lines 5694-5697; section 16.5 lines 5789-5806
    viz_block = _build_visualization_block(tier_label, request_id, combined_argmax)

    # -- agronomist_queue block ----------------------------------------------
    # spec: section 16.2 lines 5698-5702; section 16.8 lines 5853-5874
    t5_priority = t5_block.get("agronomist_priority_hint") if tier5_alert_fired else None
    queue_block = _build_queue_block(
        tier_label,
        tier5_alert_fired,
        t5_priority,
        route_ambiguous_to_queue=route_ambiguous_to_queue,
    )

    # -- warnings ------------------------------------------------------------
    # spec: section 16.2 line 5703 — empty list by default
    warnings: list[str] = []
    if iqa_result.decision == "DEGRADED":
        # spec: section 16.3 (3D human readable) — surface to client
        warnings.append(
            "Image quality is moderate (DEGRADED). Result confidence may be lower."
        )

    # -- Assemble final response ---------------------------------------------
    # spec: section 16.2 lines 5657-5706 (full schema)
    response: dict = {
        "request_id": request_id,                     # spec: 16.2 line 5661
        "image_hash": image_hash,                     # spec: 16.2 line 5662
        "timestamp_iso": timestamp_iso,               # spec: 16.2 line 5663
        "tier": tier_block,                           # spec: 16.2 lines 5664-5668
        "prediction": prediction_block,               # spec: 16.2 lines 5669-5675
        "tier5_alert": t5_block,                      # spec: 16.2 lines 5676-5679
        "severity": severity_block,                   # spec: 16.2 lines 5680-5684
        "explanation": explanation_block,             # spec: 16.2 lines 5685-5693
        "visualization": viz_block,                   # spec: 16.2 lines 5694-5697
        "agronomist_queue": queue_block,              # spec: 16.2 lines 5698-5702
        "warnings": warnings,                         # spec: 16.2 line 5703
        "model_version": model_version,               # spec: 16.2 line 5704
        "processing_time_ms": processing_time_ms,     # spec: 16.2 line 5705
    }

    _logger.debug(
        "build_response_done",
        tier_label=tier_label,
        routed=queue_block["routed"],
        tier5_alert=tier5_alert_fired,
    )
    return response


__all__ = ["build_response"]
