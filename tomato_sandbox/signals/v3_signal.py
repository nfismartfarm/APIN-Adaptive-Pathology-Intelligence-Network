"""
Signal A — v3 model (10-class tomato+chilli specialist) wrapper.

Spec section: 8 (Signal A — v3 Model), lines 1578-1789.

Public entry point:
    compute_signal_a(model, tensor) -> SignalAResult

Internal helpers (also importable for testing):
    signal_a_forward(model, x) -> dict
    extract_v3_outputs(probs_10d) -> dict

The v3 weights live at (sacred file, loaded read-only at startup):
    scripts/model3_training/checkpoints/model3_production_v3.pt
    spec: section 8.7 lines 1776-1777

No print() anywhere in this module. All informational output uses get_logger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from tomato_sandbox.config import TOMATO_CROP_MODE_INDEX  # spec: section 7.2 line 1431 / section 8.2 line 1643
from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.nan_guards import guard_array

# Re-export zero_signal_a so downstream (build_classifier_input) can import it
# from the signals package.  The function itself lives in degraded_mode.
# spec: section 12.2 lines 3232-3234 — "raw[0:6] = 0.0; raw[18] = 0.0"
from tomato_sandbox.utils.degraded_mode import zero_signal_a as zero_signal_a  # noqa: F401

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# v3 class ordering at output indices 0-5 (tomato) and 6-9 (chilli).
# spec: section 8.1 line 1610
# "0=foliar, 1=late_blight, 2=septoria, 3=ylcv, 4=mosaic, 5=healthy,
#  6=chilli_leaf_curl, 7=chilli_healthy, 8=chilli_cercospora, 9=chilli_anthracnose"
_V3_NUM_OUTPUTS: int = 10  # spec: section 8.1 lines 1607-1610
_V3_TOMATO_SLICE_END: int = 6  # first 6 are tomato
_V3_CHILLI_SLICE_START: int = 6  # last 4 are chilli

# Canonical ordering (Section 2.4):
#   [foliar=0, septoria=1, late_blight=2, ylcv=3, mosaic=4, healthy=5]
# v3 ordering (Section 8.1):
#   [foliar=0, late_blight=1, septoria=2, ylcv=3, mosaic=4, healthy=5]
# Remap v3_idx → canonical_idx:
#   v3[0]=foliar → canonical[0]
#   v3[1]=late_blight → canonical[2]
#   v3[2]=septoria → canonical[1]
#   v3[3]=ylcv → canonical[3]   (unchanged)
#   v3[4]=mosaic → canonical[4] (unchanged)
#   v3[5]=healthy → canonical[5] (unchanged)
# spec: section 8.3 lines 1672-1678
_V3_TO_CANONICAL_REMAP: np.ndarray = np.array(
    [0, 2, 1, 3, 4, 5], dtype=np.intp
)  # spec: section 8.3 lines 1672-1674 — "LORA_INDEX_FOR_V3_CLASS = [0, 2, 1, 3, 4, 5]"

_NUM_TOMATO_CLASSES: int = 6  # spec: section 8.3 line 1664

# Threshold for high chilli leakage (informational; used by tier assignment).
# spec: section 8.4 line 1695
# "TOMATO_CHILLI_LEAKAGE_THRESHOLD (default 0.40)"
CHILLI_LEAKAGE_THRESHOLD: float = 0.40  # spec: section 8.4 line 1695


# ---------------------------------------------------------------------------
# SignalAResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SignalAResult:
    """Result of Signal A (v3 model) forward pass.

    All fields are numpy arrays or Python scalars.
    No torch tensors leak out of this module.

    spec: section 8.6 lines 1716-1726
    """

    tomato_probs_canonical: np.ndarray
    """[6] float32, canonical ordering: [foliar, septoria, late_blight, ylcv, mosaic, healthy].
    spec: section 8.6 line 1719
    """

    tomato_max_prob_canonical: float
    """max of tomato_probs_canonical.
    spec: section 8.6 line 1720
    """

    tomato_argmax_canonical: int
    """index 0-5 of max in canonical ordering.
    spec: section 8.6 line 1721
    """

    chilli_leakage: float
    """Sum of v3 probs at indices 6, 7, 8, 9. Range [0, 1].
    NOT renormalized — kept as-is to preserve the leakage signal.
    spec: section 8.6 line 1722
    spec: section 8.3 lines 1687-1689 — "do NOT renormalize"
    """

    raw_probs_v3_order: np.ndarray | None
    """[10] float32, original v3 output before remap (diagnostics/monitoring only).
    None when forward_succeeded is False.
    spec: section 8.6 line 1723
    """

    forward_succeeded: bool
    """True unless an exception or NaN occurred.
    spec: section 8.6 line 1724
    """

    failure_reason: str | None
    """"exception:<type>" | "numerical_instability" | None.
    spec: section 8.6 line 1725
    """


# ---------------------------------------------------------------------------
# signal_a_forward — low-level forward pass
# ---------------------------------------------------------------------------


def signal_a_forward(model: object, x: "torch.Tensor") -> dict:
    """Run v3 model forward pass.

    Args:
        model: The v3 model callable. Must accept
            (x, crop_mode, domain_labels) and return dict with "logits" key.
        x: [B, 3, 224, 224] tensor on the correct device.

    Returns:
        dict with keys:
          "logits": [B, 10] raw logits Tensor, or None
          "probs":  [B, 10] softmax probabilities Tensor, or None
          "ok":     bool

    spec: section 8.2 lines 1616-1641
    """
    model.eval()  # spec: section 8.2 line 1625
    with torch.no_grad():
        # crop_mode = TOMATO_CROP_MODE_INDEX (=2) for tomato-conditioned features.
        # spec: section 8.2 lines 1627-1633
        # "Wrong values produce silently degraded predictions."
        crop_mode = torch.full(
            (x.shape[0],),
            TOMATO_CROP_MODE_INDEX,  # spec: section 7.2 line 1431
            dtype=torch.long,
            device=x.device,
        )
        # domain_labels=None: MixStyle no-op at inference.
        # spec: section 8.2 line 1634 — "at inference it is a no-op when None"
        out = model(x, crop_mode=crop_mode, domain_labels=None)
        logits = out["logits"]  # [B, 10] — spec: section 8.2 line 1635

        # NaN/Inf guard: rare but possible from numerical instability.
        # spec: section 8.2 lines 1637-1638
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            _logger.warning(
                "signal_a_forward: NaN/Inf in logits",
                step="signal_a_forward",
                succeeded=False,
            )
            return {"logits": None, "probs": None, "ok": False}

        probs = torch.softmax(logits, dim=1)  # spec: section 8.2 line 1639
    return {"logits": logits, "probs": probs, "ok": True}  # spec: section 8.2 line 1640


# ---------------------------------------------------------------------------
# extract_v3_outputs — remap + chilli leakage extraction
# ---------------------------------------------------------------------------


def extract_v3_outputs(probs_10d: "torch.Tensor") -> dict:
    """Extract tomato probabilities and chilli leakage from v3 10-class output.

    The 6 tomato probs are remapped from v3 ordering to canonical ordering.
    They do NOT sum to 1 (they sum to 1 - chilli_leakage). Do NOT renormalize.

    Args:
        probs_10d: [10] vector of v3 softmax probabilities (torch.Tensor).

    Returns:
        dict with:
          "tomato_probs_canonical": [6] float32 ndarray in canonical ordering
          "chilli_leakage":         float, sum of probs[6..9]
          "raw_probs_v3_order":     [10] float32 ndarray (diagnostics only)

    spec: section 8.3 lines 1659-1685
    """
    # Convert to numpy; torch→numpy boundary.
    # spec: section 8.6 line 1728 — "torch→numpy conversion happens inside extract_v3_outputs"
    p: np.ndarray = probs_10d.cpu().numpy()  # [10], float32

    # Guard: ensure finiteness before slicing.
    # spec: section 8.2 lines 1637-1638 (caller already checked, but be defensive)
    p = guard_array(p, expected_len=_V3_NUM_OUTPUTS, default_value=0.0)

    # v3 ordering for tomato slice: [foliar, late_blight, septoria, ylcv, mosaic, healthy]
    # spec: section 8.3 line 1669
    tomato_v3: np.ndarray = p[0:_V3_TOMATO_SLICE_END]  # [6]

    # Chilli leakage: sum of indices 6, 7, 8, 9.
    # spec: section 8.3 line 1670
    chilli_leakage: float = float(
        p[_V3_CHILLI_SLICE_START]
        + p[_V3_CHILLI_SLICE_START + 1]
        + p[_V3_CHILLI_SLICE_START + 2]
        + p[_V3_CHILLI_SLICE_START + 3]
    )

    # Remap v3 → canonical ordering.
    # spec: section 8.3 lines 1672-1678
    # remap[v3_idx] gives the canonical position for that v3 class.
    tomato_canonical = np.zeros(_NUM_TOMATO_CLASSES, dtype=np.float32)
    for v3_idx in range(_NUM_TOMATO_CLASSES):
        canonical_idx: int = int(_V3_TO_CANONICAL_REMAP[v3_idx])
        tomato_canonical[canonical_idx] = tomato_v3[v3_idx]

    return {
        "tomato_probs_canonical": tomato_canonical,   # spec: section 8.3 line 1681
        "chilli_leakage": chilli_leakage,              # spec: section 8.3 line 1682
        "raw_probs_v3_order": p,                       # spec: section 8.3 line 1683
    }


# ---------------------------------------------------------------------------
# compute_signal_a — public entry point
# ---------------------------------------------------------------------------


def compute_signal_a(model: object, tensor: "torch.Tensor") -> SignalAResult:
    """Run Signal A (v3) end-to-end and return a structured result.

    Args:
        model: The loaded v3 model. Caller (orchestrator) loads at startup
            from the sacred path and holds in app.state.
            spec: section 8.7 lines 1776-1777
        tensor: [3, 224, 224] CPU or GPU tensor, already preprocessed by
            preprocess_for_v3().
            spec: section 8.7 lines 1741-1742 — "add batch dim"

    Returns:
        SignalAResult with all fields populated.

    On exception:
        Returns SignalAResult(forward_succeeded=False, failure_reason="exception:<Type>")
        with zero-filled prob arrays.

    On NaN logits:
        Returns SignalAResult(forward_succeeded=False, failure_reason="numerical_instability")
        with zero-filled prob arrays.

    spec: section 8.7 lines 1738-1773
    """
    _zeros = np.zeros(_NUM_TOMATO_CLASSES, dtype=np.float32)

    # Add batch dimension: [3,224,224] → [1,3,224,224].
    # spec: section 8.7 line 1742 — "tensor.unsqueeze(0)"
    try:
        fwd = signal_a_forward(model, tensor.unsqueeze(0))
    except Exception as exc:
        _logger.error(
            "signal_a_forward raised exception",
            step="signal_a",
            succeeded=False,
            exc_info=exc,
        )
        return SignalAResult(
            tomato_probs_canonical=_zeros.copy(),
            tomato_max_prob_canonical=0.0,
            tomato_argmax_canonical=0,
            chilli_leakage=0.0,
            raw_probs_v3_order=None,
            forward_succeeded=False,
            failure_reason=f"exception: {type(exc).__name__}",
        )  # spec: section 8.7 lines 1744-1753

    if not fwd["ok"]:
        # NaN/Inf in logits.
        # spec: section 8.7 lines 1754-1763
        return SignalAResult(
            tomato_probs_canonical=_zeros.copy(),
            tomato_max_prob_canonical=0.0,
            tomato_argmax_canonical=0,
            chilli_leakage=0.0,
            raw_probs_v3_order=None,
            forward_succeeded=False,
            failure_reason="numerical_instability",
        )

    # Remove batch dim from probs: [1, 10] → [10].
    # spec: section 8.7 line 1764 — "fwd['probs'][0]"
    extracted = extract_v3_outputs(fwd["probs"][0])

    canonical: np.ndarray = extracted["tomato_probs_canonical"]

    result = SignalAResult(
        tomato_probs_canonical=canonical,
        tomato_max_prob_canonical=float(canonical.max()),
        tomato_argmax_canonical=int(canonical.argmax()),
        chilli_leakage=extracted["chilli_leakage"],
        raw_probs_v3_order=extracted["raw_probs_v3_order"],
        forward_succeeded=True,
        failure_reason=None,
    )  # spec: section 8.7 lines 1764-1773

    _logger.debug(
        "signal_a complete",
        step="signal_a",
        succeeded=True,
        argmax=result.tomato_argmax_canonical,
        max_prob=round(result.tomato_max_prob_canonical, 4),
        chilli_leakage=round(result.chilli_leakage, 4),
    )
    return result
