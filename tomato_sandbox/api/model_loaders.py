"""
Real-model loading helpers for the Tomato 3-Signal sandbox startup sequence.

Implements the loader functions for steps 4, 5, 9, and 11 of the S20.5
startup sequence. Extracted into a separate module so:
  - server.py stays focused on FastAPI wiring.
  - Each loader is independently unit-testable without importing the FastAPI app.

# spec: section 20.5 steps 4-9, 11 (lines 6560-6572)
# DEC-054: separate module rationale, architecture choices.
# DEC-055: LoRAModelAdapter wraps SinglePassLoRA to rename "cls" → "cls_token".
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from tomato_sandbox.utils.logging import get_logger

# Imported at module level so tests can patch "tomato_sandbox.api.model_loaders.predict_single"
# DEC-056: module-level import makes mock.patch work in unit tests.
from tomato_sandbox.orchestrator.pipeline import predict_single as predict_single  # noqa: F401

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project root (two levels above this file: tomato_sandbox/api/ → project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sacred paths (must match sacred_manifest.json)
# spec: section 20.5 step 4 corrected path per task card M5 meta-finding
_V3_CHECKPOINT_PATH = (
    _PROJECT_ROOT
    / "scripts"
    / "model3_training"
    / "checkpoints"
    / "model3_production_v3.pt"
)

# spec: section 20.5 step 5 corrected path per task card M5 meta-finding
_LORA_CHECKPOINT_PATH = (
    _PROJECT_ROOT
    / "models"
    / "specialist"
    / "sp_lora_checkpoints"
    / "sp_lora_epoch13_f10.9113_PRESERVED.pt"
)

# IQA reference (optional; absent → use module defaults per DEC-054 Decision 6)
_IQA_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]  # tomato_sandbox/
    / "phase_f0_calibration"
    / "iqa_reference.json"
)


# ---------------------------------------------------------------------------
# LoRAModelAdapter — DEC-055
# spec: section 9.2 lines 1842-1843 "uniform forward dict contract"
# ---------------------------------------------------------------------------


class LoRAModelAdapter:
    """Thin adapter that wraps SinglePassLoRA to rename forward key.

    SinglePassLoRA.forward() returns {"logits": ..., "cls": ..., "proj": ...}.
    signal_b_forward() expects {"logits": ..., "cls_token": ..., ...}.

    This adapter renames "cls" → "cls_token" so the model conforms to the
    spec-authoritative signal_b_forward interface without modifying either
    the model source or the signal module.

    # DEC-055: key rename adapter; no weight changes.
    # spec: section 9.2 line 1843 — forward dict must include key "cls_token"
    """

    def __init__(self, inner_model: Any) -> None:
        self._model = inner_model

    def __call__(self, x: Any) -> dict:
        """Call inner model and rename 'cls' → 'cls_token' in output dict."""
        out = self._model(x)
        # DEC-055: rename key to satisfy spec 9.2 line 1843 contract
        if "cls" in out and "cls_token" not in out:
            out["cls_token"] = out.pop("cls")
        return out

    def eval(self) -> "LoRAModelAdapter":
        """Delegate eval() to inner model (called by signal_b_forward)."""
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> "LoRAModelAdapter":
        """Delegate train() to inner model."""
        self._model.train(mode)
        return self

    def to(self, device: Any) -> "LoRAModelAdapter":
        """Delegate device transfer to inner model."""
        self._model.to(device)
        return self

    def parameters(self):
        """Delegate parameters() to inner model."""
        return self._model.parameters()

    @property
    def metadata(self) -> dict:
        """Return metadata from the inner model if available."""
        return getattr(self._model, "metadata", {})


# ---------------------------------------------------------------------------
# Step 4: Load v3 model
# spec: section 20.5 step 4 "Load v3 model weights from model2_production.pt to GPU"
# (corrected path per M5 meta-finding: model3_production_v3.pt)
# DEC-054 Decision 2: Model3(n_classes=10, pretrained=False, use_lora=True, lora_rank=4)
# ---------------------------------------------------------------------------


def load_v3_model(
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda:0",
) -> Any:
    """Load the v3 model weights and return the model in eval mode.

    Args:
        checkpoint_path: Path to checkpoint. Defaults to sacred path.
        device: PyTorch device string.

    Returns:
        Model3 instance in eval mode on ``device``.

    Raises:
        FileNotFoundError: if checkpoint file does not exist (fail-fast per S20.5).
        RuntimeError: if load_state_dict fails with unexpected keys.

    # spec: section 20.5 step 4 lines 6562
    # DEC-054 Decision 2: Model3 class, pretrained=False to skip DINOv2 download.
    """
    import sys as _sys
    import torch as _torch

    path = checkpoint_path or _V3_CHECKPOINT_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"v3 checkpoint not found at {path}. "
            f"Sacred file missing — cannot start server. "
            f"spec: section 20.5 step 4 / DEC-054."
        )

    _log.info(
        "startup_step_4_loading",
        step=4,
        path=str(path),
        device=device,
    )

    # Add project root to sys.path so scripts.model3_training.architecture is importable
    project_root = str(_PROJECT_ROOT)
    if project_root not in _sys.path:
        _sys.path.insert(0, project_root)

    from scripts.model3_training.architecture.model3_full import Model3  # type: ignore[import]

    # pretrained=False: checkpoint provides all weights; skip DINOv2 hub download
    # DEC-054 Decision 2
    model = Model3(n_classes=10, pretrained=False, use_lora=True, lora_rank=4)

    ckpt = _torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(
            f"v3 load_state_dict: missing keys: {missing[:5]}. "
            f"Checkpoint and architecture mismatch."
        )
    if unexpected:
        raise RuntimeError(
            f"v3 load_state_dict: unexpected keys: {unexpected[:5]}. "
            f"Checkpoint and architecture mismatch."
        )

    model.to(device)
    model.eval()

    run_name = ckpt.get("run_name", "unknown")
    overall_f1 = ckpt.get("soup_selection_overall_f1", 0.0)

    _log.info(
        "startup_step_4_loaded",
        step=4,
        run_name=run_name,
        overall_f1=overall_f1,
        device=device,
        # spec: section 20.5 step 4
        spec_ref="section 20.5 step 4 lines 6562",
    )

    return model, {"run_name": run_name, "overall_f1": overall_f1}


# ---------------------------------------------------------------------------
# Step 5: Load LoRA model
# spec: section 20.5 step 5 "Load LoRA model weights to GPU"
# DEC-054 Decision 3: SinglePassLoRA + LoRAModelAdapter wrapper
# DEC-055: LoRAModelAdapter renames "cls" → "cls_token"
# ---------------------------------------------------------------------------


def load_lora_model(
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda:0",
) -> Any:
    """Load the LoRA epoch-13 model and return it wrapped in LoRAModelAdapter.

    Args:
        checkpoint_path: Path to checkpoint. Defaults to sacred path.
        device: PyTorch device string.

    Returns:
        LoRAModelAdapter wrapping SinglePassLoRA, in eval mode on ``device``.

    Raises:
        FileNotFoundError: if checkpoint file does not exist (fail-fast per S20.5).
        RuntimeError: if load_state_dict fails.

    # spec: section 20.5 step 5 lines 6563
    # DEC-054 Decision 3: SinglePassLoRA wrapped in LoRAModelAdapter.
    # DEC-055: adapter renames "cls" → "cls_token" in forward output.
    """
    import sys as _sys
    import torch as _torch

    path = checkpoint_path or _LORA_CHECKPOINT_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found at {path}. "
            f"Sacred file missing — cannot start server. "
            f"spec: section 20.5 step 5 / DEC-054."
        )

    _log.info(
        "startup_step_5_loading",
        step=5,
        path=str(path),
        device=device,
    )

    # Add project root and ladi_net to sys.path for SinglePassLoRA import
    project_root = str(_PROJECT_ROOT)
    ladi_net_dir = str(_PROJECT_ROOT / "scripts" / "ladi_net")
    for p in [project_root, ladi_net_dir]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    import torch as _torch2  # noqa: F401 — already imported; keep for clarity

    # Construct model on device — SinglePassLoRA requires device at construction
    # spec: section 9.1 lines 1793-1810 — DINOv2-Base + LoRA on blocks 4-11
    from single_pass_lora_train import SinglePassLoRA  # type: ignore[import]

    _torch_device = _torch.device(device)
    model = SinglePassLoRA(device=_torch_device, n_classes=6)

    ckpt = _torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Load weights: strict=True requires exact key match
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(
            f"LoRA load_state_dict: missing keys: {missing[:5]}."
        )
    if unexpected:
        raise RuntimeError(
            f"LoRA load_state_dict: unexpected keys: {unexpected[:5]}."
        )

    model.to(_torch_device)
    model.eval()

    epoch = ckpt.get("epoch", 13)
    field_val_f1 = ckpt.get("val_sqrtn_macro_f1", 0.0)

    # Wrap in adapter (DEC-055)
    wrapped = LoRAModelAdapter(model)

    _log.info(
        "startup_step_5_loaded",
        step=5,
        epoch=epoch,
        field_val_f1=field_val_f1,
        device=device,
        # spec: section 20.5 step 5
        spec_ref="section 20.5 step 5 lines 6563",
    )

    return wrapped, {"epoch": epoch, "field_val_f1": field_val_f1}


# ---------------------------------------------------------------------------
# Step 9: Load IQA reference distributions
# spec: section 20.5 step 9 "Load IQA reference distributions"
# DEC-054 Decision 6: absent → use module defaults; log INFO
# ---------------------------------------------------------------------------


def load_iqa_reference(
    reference_path: Optional[Path] = None,
) -> Optional[dict]:
    """Load IQA reference distributions from JSON if available.

    Returns:
        dict with reference data if file exists; None if absent
        (module will use built-in defaults).

    # spec: section 20.5 step 9 lines 6567
    # DEC-054 Decision 6: absent → use module defaults, log INFO.
    """
    import json as _json

    path = reference_path or _IQA_REFERENCE_PATH

    if not path.exists():
        _log.info(
            "startup_step_9_iqa_reference_absent",
            step=9,
            path=str(path),
            description="IQA reference absent — using module defaults",
            spec_ref="section 20.5 step 9 lines 6567",
        )
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        _log.info(
            "startup_step_9_iqa_reference_loaded",
            step=9,
            path=str(path),
            spec_ref="section 20.5 step 9 lines 6567",
        )
        return data
    except Exception as exc:
        _log.warning(
            "startup_step_9_iqa_reference_load_failed",
            step=9,
            path=str(path),
            error=str(exc),
            description="IQA reference load failed — using module defaults",
        )
        return None


# ---------------------------------------------------------------------------
# Step 11: Warmup inference
# spec: section 20.5 step 11 "Run a single warmup inference on a placeholder image"
# DEC-054 Decision 7: deterministic synthetic image; fail-fast on exception.
# ---------------------------------------------------------------------------


def run_warmup_inference(pipeline: Any, device: str) -> float:
    """Run one warmup inference pass to JIT-compile CUDA kernels.

    Creates a deterministic synthetic 224×224 image (constant fill 0.5,
    ImageNet-normalized) and calls predict_single once. Logs elapsed time.

    Args:
        pipeline: PipelineContext with loaded models.
        device: PyTorch device string (for informational logging).

    Returns:
        Elapsed time in seconds.

    Raises:
        Exception: any exception from predict_single propagates up
        (fail-fast per spec 20.5 line 6573).

    # spec: section 20.5 step 11 lines 6569-6571
    # DEC-054 Decision 7: synthetic image; fail-fast if exception.
    """
    import io as _io

    import numpy as _np
    from PIL import Image as _PIL_Image

    _log.info(
        "startup_step_11_warmup_start",
        step=11,
        device=device,
        description="Running warmup inference to JIT-compile CUDA kernels",
        spec_ref="section 20.5 step 11 lines 6569-6571",
    )

    # Create deterministic synthetic image (constant grey, ImageNet-normalized)
    # 224×224 is the v3 input size (spec 8.1); LoRA uses 392×392 but the
    # preprocessing pipeline handles resize internally.
    # DEC-054 Decision 7: uint8 constant fill at mean of ImageNet mean ≈ 127
    rng = _np.random.default_rng(seed=42)
    arr = rng.integers(80, 180, (224, 224, 3), dtype=_np.uint8)
    pil_img = _PIL_Image.fromarray(arr, mode="RGB")

    # Encode as JPEG bytes (same path as real /predict handler)
    buf = _io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    warmup_request_id = "warmup-startup-00000000"

    t0 = time.monotonic()
    # Use module-level predict_single (patchable in tests via DEC-056)
    # Fail-fast: any exception propagates (spec 20.5 line 6573)
    predict_single(image_bytes, warmup_request_id, pipeline)  # noqa: F821 — defined at module level
    elapsed = time.monotonic() - t0

    _log.info(
        "startup_step_11_warmup_done",
        step=11,
        elapsed_s=round(elapsed, 3),
        description="Warmup inference complete",
        spec_ref="section 20.5 step 11 lines 6569-6571",
    )

    return elapsed
