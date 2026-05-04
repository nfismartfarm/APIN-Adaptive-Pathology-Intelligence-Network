"""
Unit tests for tomato_sandbox.api.model_loaders.

Tests cover:
  - LoRAModelAdapter key renaming (DEC-055)
  - load_iqa_reference absent/present paths
  - run_warmup_inference (mocked predict_single)
  - load_v3_model fail-fast on missing file
  - load_lora_model fail-fast on missing file

# DEC-054: model_loaders.py separate module rationale.
# DEC-055: LoRAModelAdapter renames "cls" → "cls_token".
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tomato_sandbox.api.model_loaders import (
    LoRAModelAdapter,
    load_iqa_reference,
    run_warmup_inference,
)


# ---------------------------------------------------------------------------
# LoRAModelAdapter tests
# ---------------------------------------------------------------------------


class TestLoRAModelAdapter:
    """Tests for the DEC-055 adapter that renames 'cls' → 'cls_token'."""

    def test_renames_cls_to_cls_token(self):
        """Adapter renames 'cls' → 'cls_token' in forward output.

        # DEC-055: adapter bridges SinglePassLoRA output ("cls") to
        # signal_b_forward contract ("cls_token").
        """
        inner = MagicMock()
        inner.return_value = {
            "logits": MagicMock(),
            "cls": MagicMock(),
            "proj": MagicMock(),
        }
        adapter = LoRAModelAdapter(inner)
        out = adapter(MagicMock())
        assert "cls_token" in out, "Expected 'cls_token' key in adapter output"
        assert "cls" not in out, "Key 'cls' should be removed by adapter"

    def test_preserves_logits_and_proj(self):
        """Adapter preserves other keys from inner model output."""
        logits_mock = MagicMock()
        proj_mock = MagicMock()
        inner = MagicMock()
        inner.return_value = {
            "logits": logits_mock,
            "cls": MagicMock(),
            "proj": proj_mock,
        }
        adapter = LoRAModelAdapter(inner)
        out = adapter(MagicMock())
        assert out["logits"] is logits_mock
        assert out["proj"] is proj_mock

    def test_does_not_double_rename_if_cls_token_already_present(self):
        """If 'cls_token' already present and 'cls' absent, output unchanged."""
        cls_token_mock = MagicMock()
        inner = MagicMock()
        inner.return_value = {
            "logits": MagicMock(),
            "cls_token": cls_token_mock,
        }
        adapter = LoRAModelAdapter(inner)
        out = adapter(MagicMock())
        assert out["cls_token"] is cls_token_mock

    def test_eval_delegates_to_inner(self):
        """eval() delegates to inner model and returns self."""
        inner = MagicMock()
        adapter = LoRAModelAdapter(inner)
        result = adapter.eval()
        inner.eval.assert_called_once()
        assert result is adapter

    def test_to_delegates_to_inner(self):
        """to(device) delegates to inner model."""
        inner = MagicMock()
        adapter = LoRAModelAdapter(inner)
        adapter.to("cpu")
        inner.to.assert_called_once_with("cpu")


# ---------------------------------------------------------------------------
# load_iqa_reference tests
# ---------------------------------------------------------------------------


class TestLoadIqaReference:
    """Tests for IQA reference loader."""

    def test_returns_none_when_file_absent(self, tmp_path):
        """Returns None when reference JSON is absent.

        # DEC-054 Decision 6: absent → use module defaults.
        """
        nonexistent = tmp_path / "no_such_file.json"
        result = load_iqa_reference(reference_path=nonexistent)
        assert result is None

    def test_returns_dict_when_file_present(self, tmp_path):
        """Returns dict when reference JSON is present."""
        ref_data = {"mu": [0.5, 0.6], "sigma": [0.1, 0.2]}
        ref_path = tmp_path / "iqa_reference.json"
        ref_path.write_text(json.dumps(ref_data), encoding="utf-8")
        result = load_iqa_reference(reference_path=ref_path)
        assert result == ref_data

    def test_returns_none_on_malformed_json(self, tmp_path):
        """Returns None (with warning) when JSON is malformed."""
        ref_path = tmp_path / "iqa_reference.json"
        ref_path.write_text("{not valid json", encoding="utf-8")
        result = load_iqa_reference(reference_path=ref_path)
        assert result is None


# ---------------------------------------------------------------------------
# run_warmup_inference tests
# ---------------------------------------------------------------------------


class TestRunWarmupInference:
    """Tests for warmup inference runner."""

    def test_returns_float_elapsed_time(self):
        """run_warmup_inference returns a non-negative float elapsed time."""
        mock_pipeline = MagicMock()

        with patch(
            "tomato_sandbox.api.model_loaders.predict_single",
            return_value={"tier": {"label": "2"}},
        ) as mock_predict:
            elapsed = run_warmup_inference(mock_pipeline, device="cpu")

        assert isinstance(elapsed, float)
        assert elapsed >= 0.0
        mock_predict.assert_called_once()

    def test_uses_warmup_request_id(self):
        """Warmup call uses the deterministic warmup request_id."""
        mock_pipeline = MagicMock()

        with patch(
            "tomato_sandbox.api.model_loaders.predict_single",
            return_value={"tier": {"label": "2"}},
        ) as mock_predict:
            run_warmup_inference(mock_pipeline, device="cpu")

        call_args = mock_predict.call_args
        assert call_args[0][1] == "warmup-startup-00000000"

    def test_propagates_exception_on_failure(self):
        """Exceptions from predict_single propagate (fail-fast per spec 20.5 line 6573).

        # spec: section 20.5 line 6573 — "process exits with non-zero code"
        """
        mock_pipeline = MagicMock()

        with patch(
            "tomato_sandbox.api.model_loaders.predict_single",
            side_effect=RuntimeError("synthetic failure"),
        ):
            with pytest.raises(RuntimeError, match="synthetic failure"):
                run_warmup_inference(mock_pipeline, device="cpu")


# ---------------------------------------------------------------------------
# load_v3_model fail-fast tests (no real weights needed)
# ---------------------------------------------------------------------------


class TestLoadV3ModelFailFast:
    """Tests that load_v3_model fails fast on missing checkpoint."""

    def test_raises_file_not_found_when_checkpoint_absent(self, tmp_path):
        """load_v3_model raises FileNotFoundError when checkpoint is absent.

        # spec: section 20.5 line 6573 — fail-fast on any step failure.
        """
        from tomato_sandbox.api.model_loaders import load_v3_model

        nonexistent = tmp_path / "no_checkpoint.pt"
        with pytest.raises(FileNotFoundError, match="v3 checkpoint not found"):
            load_v3_model(checkpoint_path=nonexistent, device="cpu")


# ---------------------------------------------------------------------------
# load_lora_model fail-fast tests (no real weights needed)
# ---------------------------------------------------------------------------


class TestLoadLoraModelFailFast:
    """Tests that load_lora_model fails fast on missing checkpoint."""

    def test_raises_file_not_found_when_checkpoint_absent(self, tmp_path):
        """load_lora_model raises FileNotFoundError when checkpoint is absent.

        # spec: section 20.5 line 6573 — fail-fast on any step failure.
        """
        from tomato_sandbox.api.model_loaders import load_lora_model

        nonexistent = tmp_path / "no_lora_checkpoint.pt"
        with pytest.raises(FileNotFoundError, match="LoRA checkpoint not found"):
            load_lora_model(checkpoint_path=nonexistent, device="cpu")
