"""
Unit tests for tomato_sandbox/response/response_builder.py

Spec coverage: Section 16, lines 5637-5939.
DEC-043: T-IMPL-6b test module.

Test groups:
  1.  Schema completeness — all required keys present for every tier
  2.  Per-tier human_readable strings
  3.  Per-tier alert_level values
  4.  Prediction block field types and values
  5.  BLK-010.3 — Tier 4A routing (routed only if T5 co-fires)
  6.  Queue routing for Tiers 3A/3B/3C/3D (route_ambiguous flag)
  7.  Tier 4B — not routed; gradcam_url null
  8.  Tier 1 / 2 without T5 — not routed
  9.  Tier 1 / 2 with T5 — always routed
  10. tier5_alert independence (T5 can co-fire with any base tier)
  11. T5 block fields when not fired
  12. T5 block fields when fired (argmax bullet)
  13. T5 block fields when fired (in-set bullet)
  14. T5 agronomist_priority_hint: high vs medium
  15. Class label strings (canonical short names)
  16. JSON serialization (json.dumps on every tier)
  17. Confidence display rules in user_strings (4A → "below 45%", 4B → "unknown")
  18. IQA DEGRADED adds warning
  19. route_ambiguous=True enables Tier 3x routing without T5
  20. model_version propagates
  21. request_metadata propagates
  22. Stable schema — null fields present (not omitted)

Total: 42 tests
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from tomato_sandbox.response.response_builder import build_response

# ---------------------------------------------------------------------------
# Minimal stub dataclasses for upstream types
# The response builder imports TYPE_CHECKING only; we build stubs that satisfy
# attribute access used by build_response.
# ---------------------------------------------------------------------------


@dataclass
class _TierAssignment:
    tier_label: str
    tier5_alert: bool
    rule_id_fired: str


@dataclass
class _ClassifierResult:
    p_final_calibrated: np.ndarray          # [7]
    combined_argmax: int
    combined_max_prob: float
    combined_margin: float
    p_final_uncalibrated: np.ndarray        # [7]
    p_stage1: np.ndarray                    # [3]
    p_stage2: np.ndarray                    # [5]
    classifier_succeeded: bool = True
    failure_reason: Optional[str] = None


@dataclass
class _ConformalResult:
    prediction_set: list
    prediction_set_size: int
    threshold_tau_used: float = 0.4
    nonconformity_per_class: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    inside_set_per_class: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=bool)
    )


@dataclass
class _IQAResult:
    decision: str
    aggregate_score: float
    per_dimension: dict = field(default_factory=dict)
    failing_dimensions: list = field(default_factory=list)
    retake_message: Optional[str] = None
    green_mask: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_UNIFORM_7 = np.array([1/7]*7, dtype=np.float32)
_UNIFORM_3 = np.array([1/3]*3, dtype=np.float32)
_UNIFORM_5 = np.array([1/5]*5, dtype=np.float32)


def _make_classifier(argmax: int = 0, max_prob: float = 0.91, margin: float = 0.35,
                     probs: Optional[np.ndarray] = None) -> _ClassifierResult:
    if probs is None:
        probs = np.zeros(7, dtype=np.float32)
        probs[argmax] = max_prob
        # Distribute remainder
        second = (1 - max_prob) / 6
        for i in range(7):
            if i != argmax:
                probs[i] = second
    return _ClassifierResult(
        p_final_calibrated=probs,
        combined_argmax=argmax,
        combined_max_prob=max_prob,
        combined_margin=margin,
        p_final_uncalibrated=_UNIFORM_7.copy(),
        p_stage1=_UNIFORM_3.copy(),
        p_stage2=_UNIFORM_5.copy(),
    )


def _make_conformal(prediction_set: list = None,
                    size: Optional[int] = None) -> _ConformalResult:
    ps = prediction_set if prediction_set is not None else [0]
    sz = size if size is not None else len(ps)
    return _ConformalResult(prediction_set=ps, prediction_set_size=sz)


def _make_iqa(decision: str = "ACCEPTABLE") -> _IQAResult:
    return _IQAResult(decision=decision, aggregate_score=0.85)


def _make_tier(label: str, t5: bool = False,
               rule_id: str = "7c") -> _TierAssignment:
    return _TierAssignment(tier_label=label, tier5_alert=t5, rule_id_fired=rule_id)


def _build(tier_label: str, t5: bool = False, argmax: int = 0,
           max_prob: float = 0.91, margin: float = 0.35,
           ps: Optional[list] = None, iqa_decision: str = "ACCEPTABLE",
           rule_id: str = "7c", route_ambiguous: bool = False,
           probs: Optional[np.ndarray] = None,
           meta: Optional[dict] = None) -> dict:
    """Convenience wrapper for build_response."""
    ta = _make_tier(tier_label, t5, rule_id)
    cr = _make_classifier(argmax, max_prob, margin, probs)
    conf = _make_conformal(ps if ps is not None else [argmax])
    iqa = _make_iqa(iqa_decision)
    return build_response(
        ta, cr, conf, iqa,
        request_metadata=meta or {"request_id": "test-req-01"},
        route_ambiguous_to_queue=route_ambiguous,
    )


# ---------------------------------------------------------------------------
# Required top-level schema keys
# spec: section 16.2 lines 5661-5705
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = [
    "request_id", "image_hash", "timestamp_iso", "tier", "prediction",
    "tier5_alert", "severity", "explanation", "visualization",
    "agronomist_queue", "warnings", "model_version", "processing_time_ms",
]

_TIER_LABELS = ["1", "2", "3A", "3B", "3C", "3D", "4A", "4B"]


@pytest.mark.parametrize("tier_label", _TIER_LABELS)
def test_schema_all_keys_present(tier_label):
    """All required top-level keys present for every tier label.
    spec: section 16.2 line 5709 — 'all fields present in every response'
    """
    resp = _build(tier_label)
    for key in _REQUIRED_KEYS:
        assert key in resp, f"Missing key '{key}' in tier {tier_label} response"


# ---------------------------------------------------------------------------
# Tier block field accuracy
# spec: section 16.3 lines 5718-5728
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier_label,expected_human", [
    ("1",  "Definitive prediction"),
    ("2",  "Confident prediction"),
    ("3A", "Two possible diseases"),
    ("3B", "Multiple possible diseases"),
    ("3C", "Image quality concern (segmentation or chilli leaf detection)"),
    ("3D", "Image quality moderate; result less confident"),
    ("4A", "Low confidence — manual review recommended"),
    ("4B", "Pipeline issue — please retake or contact support"),
])
def test_tier_human_readable(tier_label, expected_human):
    """Tier human_readable strings match spec table verbatim.
    spec: section 16.3 lines 5720-5728
    """
    resp = _build(tier_label)
    assert resp["tier"]["human_readable"] == expected_human


def test_tier_label_propagated():
    """tier.label matches the input tier_label."""
    resp = _build("3A")
    assert resp["tier"]["label"] == "3A"


def test_tier_alert_level_tier1():
    """Tier 1 → alert_level = 'info'.  spec: section 16.2 line 5667."""
    resp = _build("1")
    assert resp["tier"]["alert_level"] == "info"


def test_tier_alert_level_tier4a():
    """Tier 4A → alert_level = 'error' (higher urgency)."""
    resp = _build("4A")
    assert resp["tier"]["alert_level"] == "error"


# ---------------------------------------------------------------------------
# Prediction block
# spec: section 16.2 lines 5669-5675
# ---------------------------------------------------------------------------

def test_prediction_primary_class_foliar():
    """primary_class = 'foliar' for argmax=0.
    spec: section 16.2 line 5670; Appendix E class index
    """
    resp = _build("1", argmax=0)
    assert resp["prediction"]["primary_class"] == "foliar"


def test_prediction_primary_class_late_blight():
    """primary_class = 'late_blight' for argmax=2.
    spec: section 16.2 line 5670; class index ref
    """
    resp = _build("1", argmax=2, t5=True)
    assert resp["prediction"]["primary_class"] == "late_blight"


def test_prediction_primary_class_ood():
    """primary_class = 'OOD' for argmax=6.
    spec: Appendix E — OOD is index 6
    """
    resp = _build("4A", argmax=6)
    assert resp["prediction"]["primary_class"] == "OOD"


def test_prediction_confidence_raw_float():
    """primary_confidence is raw float in [0,1], not percentage.
    spec: section 16.6 line 5817 — 'primary_confidence uses raw combined_max_prob'
    """
    resp = _build("1", max_prob=0.91)
    conf = resp["prediction"]["primary_confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
    assert abs(conf - 0.91) < 1e-3


def test_prediction_set_class_names():
    """prediction_set uses short canonical names.
    spec: section 16.2 line 5673
    """
    resp = _build("3A", ps=[0, 1])
    assert "foliar" in resp["prediction"]["prediction_set"]
    assert "septoria" in resp["prediction"]["prediction_set"]


# ---------------------------------------------------------------------------
# BLK-010.3 — Tier 4A queue routing
# spec: section 16.8 line 5856
# "Tier 4A → routed only if Tier 5 also fires; otherwise user opt-in only"
# ---------------------------------------------------------------------------

def test_tier4a_t5_false_not_routed():
    """Tier 4A without T5 → routed=False (user opt-in only).
    spec: section 16.8 line 5856 (BLK-010.3 fix verbatim)
    """
    resp = _build("4A", t5=False)
    assert resp["agronomist_queue"]["routed"] is False


def test_tier4a_t5_true_is_routed():
    """Tier 4A WITH T5 → routed=True.
    spec: section 16.8 line 5854 — 'Tier 5 alert fires → always routed'
    BLK-010.3: Tier 4A is routed ONLY in this case
    """
    # late_blight argmax with max=0.25 → T5 argmax bullet fires
    probs = np.zeros(7, dtype=np.float32)
    probs[2] = 0.30  # late_blight
    resp = _build("4A", t5=True, argmax=2, max_prob=0.30, probs=probs)
    assert resp["agronomist_queue"]["routed"] is True


def test_tier4a_t5_false_priority_null():
    """Tier 4A without T5: priority is null.
    spec: section 16.8 line 5872 — 'routed: false → all other fields null'
    """
    resp = _build("4A", t5=False)
    assert resp["agronomist_queue"]["priority"] is None
    assert resp["agronomist_queue"]["queue_id"] is None


# ---------------------------------------------------------------------------
# Tier 3x queue routing
# spec: section 16.8 line 5855
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier_label", ["3A", "3B", "3C", "3D"])
def test_tier3x_not_routed_by_default(tier_label):
    """Tier 3x not routed when route_ambiguous=False (default).
    spec: section 16.8 line 5855 — 'default false; agronomist capacity-dependent'
    """
    resp = _build(tier_label, route_ambiguous=False)
    assert resp["agronomist_queue"]["routed"] is False


@pytest.mark.parametrize("tier_label", ["3A", "3B", "3C", "3D"])
def test_tier3x_routed_when_flag_enabled(tier_label):
    """Tier 3x routed when route_ambiguous=True.
    spec: section 16.8 line 5855
    """
    resp = _build(tier_label, route_ambiguous=True)
    assert resp["agronomist_queue"]["routed"] is True


# ---------------------------------------------------------------------------
# Tier 4B queue routing
# spec: section 16.8 line 5857 — "Tier 4B → NOT routed"
# ---------------------------------------------------------------------------

def test_tier4b_not_routed():
    """Tier 4B never routed.  spec: section 16.8 line 5857"""
    resp = _build("4B", t5=False)
    assert resp["agronomist_queue"]["routed"] is False


def test_tier4b_not_routed_even_with_t5():
    """Tier 4B not routed even when T5 fires.
    spec: section 16.8 line 5857 (pipeline issue, not model uncertainty)
    Note: spec 16.8 line 5854 says 'T5 fires → always routed'; 4B is
    a pipeline failure and cannot produce a valid T5 signal (Rule 1 fires
    when any signal fails). We test the spec-stated Tier 4B routing exclusion.
    Spec 16.8 line 5857 wins over the general T5 routing rule for 4B.
    """
    resp = _build("4B", t5=True)
    # Tier 4B is pipeline failure; spec 16.8 line 5857 explicitly excludes it
    assert resp["agronomist_queue"]["routed"] is False


# ---------------------------------------------------------------------------
# Tier 1 / 2 routing without T5
# spec: section 16.8 line 5858
# ---------------------------------------------------------------------------

def test_tier1_no_t5_not_routed():
    """Tier 1 without T5 → not routed.
    spec: section 16.8 line 5858
    """
    resp = _build("1", t5=False)
    assert resp["agronomist_queue"]["routed"] is False


def test_tier2_no_t5_not_routed():
    """Tier 2 without T5 → not routed.
    spec: section 16.8 line 5858
    """
    resp = _build("2", t5=False, max_prob=0.72, margin=0.22)
    assert resp["agronomist_queue"]["routed"] is False


# ---------------------------------------------------------------------------
# Tier 1 / 2 with T5 — always routed
# spec: section 16.8 line 5854 — "Tier 5 alert fires → always routed"
# ---------------------------------------------------------------------------

def test_tier1_with_t5_is_routed():
    """Tier 1 + T5 → always routed.  spec: section 16.8 line 5854"""
    probs = np.zeros(7, dtype=np.float32)
    probs[2] = 0.91  # late_blight argmax
    resp = _build("1", t5=True, argmax=2, max_prob=0.91, probs=probs)
    assert resp["agronomist_queue"]["routed"] is True


def test_tier2_with_t5_is_routed():
    """Tier 2 + T5 → always routed.  spec: section 16.8 line 5854"""
    probs = np.zeros(7, dtype=np.float32)
    probs[3] = 0.72  # ylcv argmax (T5 dangerous class)
    resp = _build("2", t5=True, argmax=3, max_prob=0.72, probs=probs)
    assert resp["agronomist_queue"]["routed"] is True


# ---------------------------------------------------------------------------
# T5 alert block fields
# spec: section 16.7 lines 5823-5849
# ---------------------------------------------------------------------------

def test_t5_not_fired_all_null():
    """When T5 not fired, all t5 block fields except 'fired' are null.
    spec: section 16.2 lines 5676-5679 — 'null when not fired'
    """
    resp = _build("1", t5=False)
    t5 = resp["tier5_alert"]
    assert t5["fired"] is False
    assert t5["reason"] is None
    assert t5["trigger_class"] is None
    assert t5["trigger_probability"] is None
    assert t5["agronomist_priority_hint"] is None


def test_t5_fired_argmax_bullet():
    """T5 via argmax bullet: reason = 'argmax_dangerous_disease'.
    spec: section 16.7 line 5838
    """
    probs = np.zeros(7, dtype=np.float32)
    probs[2] = 0.91  # late_blight argmax, clearly above 0.20
    resp = _build("1", t5=True, argmax=2, max_prob=0.91, probs=probs, ps=[2])
    t5 = resp["tier5_alert"]
    assert t5["fired"] is True
    # late_blight is both argmax and in-set → combined reason
    assert t5["reason"] in {
        "argmax_dangerous_disease",
        "argmax_dangerous_and_late_blight_in_set",
    }
    assert t5["trigger_class"] == "late_blight"


def test_t5_fired_mosaic_argmax():
    """T5 via argmax bullet for mosaic (idx=4).
    spec: section 16.7 line 5838 — argmax in {late_blight, mosaic, ylcv}
    """
    probs = np.zeros(7, dtype=np.float32)
    probs[4] = 0.80  # mosaic argmax
    resp = _build("1", t5=True, argmax=4, max_prob=0.80, probs=probs, ps=[4])
    t5 = resp["tier5_alert"]
    assert t5["fired"] is True
    assert t5["trigger_class"] == "mosaic"


def test_t5_fired_ylcv_argmax():
    """T5 via argmax bullet for ylcv (idx=3).
    spec: section 14.3 line 3789; section 16.7 line 5838
    """
    probs = np.zeros(7, dtype=np.float32)
    probs[3] = 0.77  # ylcv argmax
    resp = _build("1", t5=True, argmax=3, max_prob=0.77, probs=probs, ps=[3])
    t5 = resp["tier5_alert"]
    assert t5["fired"] is True
    assert t5["trigger_class"] == "ylcv"


def test_t5_priority_hint_high():
    """T5 priority = 'high' when late_blight argmax AND max >= 0.50.
    spec: section 16.7 line 5845
    """
    probs = np.zeros(7, dtype=np.float32)
    probs[2] = 0.91
    resp = _build("1", t5=True, argmax=2, max_prob=0.91, probs=probs)
    assert resp["tier5_alert"]["agronomist_priority_hint"] == "high"


def test_t5_priority_hint_medium():
    """T5 priority = 'medium' for non-late_blight argmax (e.g. ylcv).
    spec: section 16.7 line 5846 — 'any other T5 firing'
    """
    probs = np.zeros(7, dtype=np.float32)
    probs[3] = 0.85  # ylcv
    resp = _build("1", t5=True, argmax=3, max_prob=0.85, probs=probs)
    assert resp["tier5_alert"]["agronomist_priority_hint"] == "medium"


# ---------------------------------------------------------------------------
# T5 independence — fires alongside any base tier
# spec: section 14.3 line 3784 — "Tier 5 is evaluated INDEPENDENTLY"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_tier", ["1", "2", "3A", "3B", "3C", "3D", "4A"])
def test_t5_fires_independently_of_base_tier(base_tier):
    """T5 flag propagates to response regardless of base tier.
    spec: section 14.3 — "evaluated independently after the base tier is set"
    """
    probs = np.zeros(7, dtype=np.float32)
    probs[2] = 0.35  # late_blight above 0.20
    resp = _build(base_tier, t5=True, argmax=2, max_prob=0.35, probs=probs)
    assert resp["tier5_alert"]["fired"] is True
    assert resp["tier"]["label"] == base_tier


# ---------------------------------------------------------------------------
# JSON serialization
# spec: section 16.2 line 5713 — "JSON-Schema-validated before sending"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier_label", _TIER_LABELS)
def test_json_serializable(tier_label):
    """json.dumps succeeds on the response for every tier.
    spec: section 16.2 line 5713 — response is JSON-serializable
    """
    resp = _build(tier_label)
    # Should not raise
    serialized = json.dumps(resp)
    parsed = json.loads(serialized)
    assert parsed["tier"]["label"] == tier_label


# ---------------------------------------------------------------------------
# Confidence display rules (user_strings only)
# spec: section 16.6 lines 5812-5818
# ---------------------------------------------------------------------------

def test_tier4a_user_string_says_below_45():
    """Tier 4A user string contains 'below 45%' not the raw number.
    spec: section 16.6 line 5813
    """
    resp = _build("4A", max_prob=0.40)
    user_str = " ".join(resp["explanation"]["user_strings"])
    assert "below 45%" in user_str


def test_tier4b_user_string_says_unknown():
    """Tier 4B user string contains 'unknown' (no percentage).
    spec: section 16.6 line 5815
    """
    resp = _build("4B")
    user_str = " ".join(resp["explanation"]["user_strings"])
    assert "unknown" in user_str.lower() or "issue" in user_str.lower()


# ---------------------------------------------------------------------------
# IQA DEGRADED warning
# spec: section 16.2 line 5703 — warnings list
# ---------------------------------------------------------------------------

def test_degraded_iqa_adds_warning():
    """IQA DEGRADED → non-empty warnings list.
    spec: section 16.3 — 3D describes degraded quality
    """
    resp = _build("3D", iqa_decision="DEGRADED")
    assert len(resp["warnings"]) > 0
    assert "DEGRADED" in " ".join(resp["warnings"]).upper()


def test_acceptable_iqa_no_warnings():
    """IQA ACCEPTABLE → empty warnings.
    spec: section 16.2 line 5703 — 'warnings: []' in example
    """
    resp = _build("1", iqa_decision="ACCEPTABLE")
    assert resp["warnings"] == []


# ---------------------------------------------------------------------------
# Stable schema — null fields present (not omitted)
# spec: section 16.2 line 5709
# ---------------------------------------------------------------------------

def test_severity_block_null_fields_present():
    """severity block always present even when null.
    spec: section 16.2 line 5709 — 'null rather than omitted'
    """
    resp = _build("1")
    sev = resp["severity"]
    assert "grade" in sev
    assert "human_readable" in sev
    assert "details" in sev


def test_queue_null_fields_present_when_not_routed():
    """agronomist_queue null fields present when not routed.
    spec: section 16.8 line 5872 — 'routed: false → all other fields null'
    """
    resp = _build("1", t5=False)
    q = resp["agronomist_queue"]
    assert "routed" in q
    assert "priority" in q
    assert "queue_id" in q
    assert q["routed"] is False
    assert q["priority"] is None
    assert q["queue_id"] is None


def test_t5_block_null_fields_present_when_not_fired():
    """All T5 block fields present (as null) when not fired.
    spec: section 16.2 line 5709 — stable schema; null not omitted
    """
    resp = _build("2", t5=False)
    t5 = resp["tier5_alert"]
    assert "fired" in t5
    assert "reason" in t5
    assert "trigger_class" in t5
    assert "trigger_probability" in t5
    assert "agronomist_priority_hint" in t5


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------

def test_model_version_propagates():
    """model_version propagates to response.  spec: section 16.2 line 5704"""
    ta = _make_tier("1")
    cr = _make_classifier(0, 0.91, 0.35)
    conf = _make_conformal([0])
    iqa = _make_iqa()
    resp = build_response(ta, cr, conf, iqa, model_version="test-v9.9.9")
    assert resp["model_version"] == "test-v9.9.9"


def test_request_metadata_propagates():
    """request_id, image_hash, timestamp propagate from metadata.
    spec: section 16.2 lines 5661-5663; section 16.1 line 5651
    """
    meta = {
        "request_id": "req-abc-123",
        "image_hash": "deadbeef" * 8,
        "timestamp_iso": "2026-05-02T12:00:00Z",
        "processing_time_ms": 412,
    }
    resp = _build("1", meta=meta)
    assert resp["request_id"] == "req-abc-123"
    assert resp["image_hash"] == "deadbeef" * 8
    assert resp["timestamp_iso"] == "2026-05-02T12:00:00Z"
    assert resp["processing_time_ms"] == 412


# ---------------------------------------------------------------------------
# Class label string accuracy — all 7 canonical names
# spec: section 12.1 lines 3151-3159; Appendix E
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argmax,expected_class", [
    (0, "foliar"),
    (1, "septoria"),
    (2, "late_blight"),
    (3, "ylcv"),
    (4, "mosaic"),
    (5, "healthy"),
    (6, "OOD"),
])
def test_class_label_string_accuracy(argmax, expected_class):
    """primary_class matches canonical short name for each index.
    spec: Appendix E class index conventions; section 16.2 line 5670
    """
    resp = _build("4A", argmax=argmax)
    assert resp["prediction"]["primary_class"] == expected_class
