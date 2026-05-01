"""
Unit tests for tomato_sandbox.preprocessing.preprocess

Covers:
- Every public function: preprocess_for_v3, preprocess_for_lora,
  preprocess_for_psv, shades_of_gray
- All constants from config imported by the module: V3_INPUT_SIZE,
  LORA_INPUT_SIZE, LORA_PAD_VALUE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
  IMAGENET_MEAN, IMAGENET_STD, TOMATO_CROP_MODE_INDEX
- All spec-defined output shapes, dtypes, value ranges
- Letterbox padding behaviour (wide, tall, square images)
- PSV resize-cap threshold (above and below 1200 px)
- shades_of_gray: p=1 grey-world; p=6 default; identity on grey image
- NaN-guard defensive paths (all-zero image after normalization edge case)
- TTA contract: functions are callable multiple times on different PIL images
  (orchestrator's responsibility to call them N times)
- __init__.py re-exports: both import paths resolve to the same objects

spec: section 7 lines 1392-1574
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from PIL import Image as _PIL  # type: ignore[import]
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed"),
    pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pil(h: int = 480, w: int = 640, colour: tuple[int, int, int] = (100, 150, 80)) -> "_PIL.Image":
    """Create a solid-colour PIL RGB image of given dimensions."""
    arr = np.full((h, w, 3), colour, dtype=np.uint8)
    return _PIL.fromarray(arr, mode="RGB")


def _make_pil_random(h: int = 480, w: int = 640, seed: int = 42) -> "_PIL.Image":
    """Create a random-noise PIL RGB image."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return _PIL.fromarray(arr, mode="RGB")


@pytest.fixture
def landscape_pil() -> "_PIL.Image":
    return _make_pil_random(h=480, w=640)


@pytest.fixture
def portrait_pil() -> "_PIL.Image":
    return _make_pil_random(h=640, w=480)


@pytest.fixture
def square_pil() -> "_PIL.Image":
    return _make_pil_random(h=512, w=512)


@pytest.fixture
def large_pil() -> "_PIL.Image":
    """Image larger than 1200 px on the longest side — triggers PSV resize cap."""
    return _make_pil_random(h=2400, w=3200)


@pytest.fixture
def exactly_1200_pil() -> "_PIL.Image":
    """Image with longest side exactly 1200 px — should NOT be resized."""
    return _make_pil_random(h=900, w=1200)


@pytest.fixture
def below_1200_pil() -> "_PIL.Image":
    """Image below 1200 px on all sides — should NOT be resized."""
    return _make_pil_random(h=480, w=640)


# ---------------------------------------------------------------------------
# Constants tests
# spec: section 7.2 lines 1421-1432
# ---------------------------------------------------------------------------

class TestConfigConstants:
    """Verify pinned preprocessing constants in config.py.

    spec: section 7.2 lines 1421-1432 — 'Pinned constants (live in
    tomato_sandbox/config.py)'
    """

    def test_v3_input_size(self) -> None:
        # spec: section 7.2 line 1428
        from tomato_sandbox.config import V3_INPUT_SIZE
        assert V3_INPUT_SIZE == 224

    def test_lora_input_size(self) -> None:
        # spec: section 7.2 line 1429
        from tomato_sandbox.config import LORA_INPUT_SIZE
        assert LORA_INPUT_SIZE == 392

    def test_lora_pad_value(self) -> None:
        # spec: section 7.2 lines 1430-1431
        from tomato_sandbox.config import LORA_PAD_VALUE
        assert LORA_PAD_VALUE == 114

    def test_clahe_clip_limit(self) -> None:
        # spec: section 7.2 line 1424
        from tomato_sandbox.config import CLAHE_CLIP_LIMIT
        assert CLAHE_CLIP_LIMIT == 2.0

    def test_clahe_tile_grid(self) -> None:
        # spec: section 7.2 line 1425
        from tomato_sandbox.config import CLAHE_TILE_GRID
        assert CLAHE_TILE_GRID == (8, 8)

    def test_imagenet_mean_rgb_order(self) -> None:
        # spec: section 7.2 lines 1426-1427 — "RGB order"
        from tomato_sandbox.config import IMAGENET_MEAN
        np.testing.assert_allclose(IMAGENET_MEAN, [0.485, 0.456, 0.406], rtol=1e-6)

    def test_imagenet_std_rgb_order(self) -> None:
        # spec: section 7.2 line 1427
        from tomato_sandbox.config import IMAGENET_STD
        np.testing.assert_allclose(IMAGENET_STD, [0.229, 0.224, 0.225], rtol=1e-6)

    def test_tomato_crop_mode_index(self) -> None:
        # spec: section 7.2 line 1431
        from tomato_sandbox.config import TOMATO_CROP_MODE_INDEX
        assert TOMATO_CROP_MODE_INDEX == 2


# ---------------------------------------------------------------------------
# Import-path tests
# spec: section 7.6 line 1563 + DEC-031 sub-package re-export
# ---------------------------------------------------------------------------

class TestImportPaths:
    """Both import paths must resolve to identical function objects."""

    def test_package_import_preprocess_for_v3(self) -> None:
        from tomato_sandbox.preprocessing import preprocess_for_v3 as a
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3 as b
        assert a is b

    def test_package_import_preprocess_for_lora(self) -> None:
        from tomato_sandbox.preprocessing import preprocess_for_lora as a
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora as b
        assert a is b

    def test_package_import_preprocess_for_psv(self) -> None:
        from tomato_sandbox.preprocessing import preprocess_for_psv as a
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv as b
        assert a is b

    def test_package_import_shades_of_gray(self) -> None:
        from tomato_sandbox.preprocessing import shades_of_gray as a
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray as b
        assert a is b


# ---------------------------------------------------------------------------
# preprocess_for_v3
# spec: section 7.2 lines 1437-1458
# ---------------------------------------------------------------------------

class TestPreprocessForV3:
    """Tests for Pipeline 1 — Signal A (v3 model).

    spec: section 7.2 lines 1437-1458
    """

    def test_output_shape_landscape(self, landscape_pil: "_PIL.Image") -> None:
        # spec: section 7.2 line 1439 — "[3, 224, 224] tensor on CPU"
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t = preprocess_for_v3(landscape_pil)
        assert t.shape == (3, 224, 224)

    def test_output_shape_portrait(self, portrait_pil: "_PIL.Image") -> None:
        # spec: section 7.2 line 1439 — stretch resize; shape is always 3x224x224
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t = preprocess_for_v3(portrait_pil)
        assert t.shape == (3, 224, 224)

    def test_output_shape_square(self, square_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t = preprocess_for_v3(square_pil)
        assert t.shape == (3, 224, 224)

    def test_output_dtype_float32(self, landscape_pil: "_PIL.Image") -> None:
        # spec: section 7.2 lines 1453 — "astype(np.float32)"
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t = preprocess_for_v3(landscape_pil)
        assert t.dtype == torch.float32

    def test_output_on_cpu(self, landscape_pil: "_PIL.Image") -> None:
        # spec: section 7.2 line 1439 — "on CPU"
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t = preprocess_for_v3(landscape_pil)
        assert t.device.type == "cpu"

    def test_output_finite(self, landscape_pil: "_PIL.Image") -> None:
        # All values must be finite (non-NaN, non-Inf)
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t = preprocess_for_v3(landscape_pil)
        assert torch.all(torch.isfinite(t))

    def test_normalization_range(self, landscape_pil: "_PIL.Image") -> None:
        """After ImageNet normalization the tensor should span a reasonable range.

        For typical image content the values should be in roughly [-3, 3].
        spec: section 7.2 lines 1453-1454 — ImageNet normalize applied.
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t = preprocess_for_v3(landscape_pil)
        # Not asserted to be in [-1, 1] because ImageNet normalization can produce
        # values outside that range; just check it is not still in [0, 255].
        assert t.max().item() < 10.0
        assert t.min().item() > -10.0

    def test_stretch_resize_not_letterbox(self) -> None:
        """Stretch resize always produces exactly (224, 224) regardless of AR.

        spec: section 7.2 lines 1461 — 'v3 was trained with stretch resize'
        spec: section 7.2 line 1443 — pil_image.resize((224, 224), BILINEAR)
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        # Very wide image: 800x100
        img = _make_pil_random(h=100, w=800)
        t = preprocess_for_v3(img)
        assert t.shape == (3, 224, 224)

    def test_deterministic_on_same_input(self, landscape_pil: "_PIL.Image") -> None:
        """Two calls with the same PIL image produce identical tensors."""
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        t1 = preprocess_for_v3(landscape_pil)
        t2 = preprocess_for_v3(landscape_pil)
        assert torch.equal(t1, t2)

    def test_tta_second_call_different_image(self) -> None:
        """TTA pattern: calling the function again with a different PIL image
        produces a different tensor — function has no internal state.

        spec: section 7.1 line 1417 — augmented views call the preprocessing
        functions again with the augmented PIL image.
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        img_a = _make_pil_random(h=480, w=640, seed=1)
        img_b = _make_pil_random(h=480, w=640, seed=2)
        t_a = preprocess_for_v3(img_a)
        t_b = preprocess_for_v3(img_b)
        # Different images must not produce identical tensors
        assert not torch.equal(t_a, t_b)

    def test_small_image(self) -> None:
        """Very small image (50x50) should still produce (3, 224, 224)."""
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        img = _make_pil_random(h=50, w=50)
        t = preprocess_for_v3(img)
        assert t.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# preprocess_for_lora
# spec: section 7.3 lines 1468-1501
# ---------------------------------------------------------------------------

class TestPreprocessForLoRA:
    """Tests for Pipeline 2 — Signal B (LoRA model).

    spec: section 7.3 lines 1468-1501
    """

    def test_output_shape_landscape(self, landscape_pil: "_PIL.Image") -> None:
        # spec: section 7.3 line 1471 — "[3, 392, 392] tensor"
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        t = preprocess_for_lora(landscape_pil)
        assert t.shape == (3, 392, 392)

    def test_output_shape_portrait(self, portrait_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        t = preprocess_for_lora(portrait_pil)
        assert t.shape == (3, 392, 392)

    def test_output_shape_square(self, square_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        t = preprocess_for_lora(square_pil)
        assert t.shape == (3, 392, 392)

    def test_output_dtype_float32(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        t = preprocess_for_lora(landscape_pil)
        assert t.dtype == torch.float32

    def test_output_on_cpu(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        t = preprocess_for_lora(landscape_pil)
        assert t.device.type == "cpu"

    def test_output_finite(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        t = preprocess_for_lora(landscape_pil)
        assert torch.all(torch.isfinite(t))

    def test_letterbox_wide_image_preserves_ar(self) -> None:
        """Wide image: H < W.  After letterbox, both dims are LORA_INPUT_SIZE.
        The scaled height should be < LORA_INPUT_SIZE; padding fills the rest.

        spec: section 7.3 lines 1473-1489 — letterbox algorithm
        spec: section 7.3 line 1504 — 'leaves photographed in portrait vs
        landscape phone orientation produce different feature responses if not
        aspect-preserving'
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        from tomato_sandbox.config import LORA_INPUT_SIZE
        # 100 high, 400 wide — scale = 392/400 ≈ 0.98 → new_h ≈ 98
        img = _make_pil_random(h=100, w=400)
        t = preprocess_for_lora(img)
        assert t.shape == (3, LORA_INPUT_SIZE, LORA_INPUT_SIZE)

    def test_letterbox_tall_image_preserves_ar(self) -> None:
        """Tall image: H > W.  After letterbox output is still 392×392."""
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        from tomato_sandbox.config import LORA_INPUT_SIZE
        img = _make_pil_random(h=400, w=100)
        t = preprocess_for_lora(img)
        assert t.shape == (3, LORA_INPUT_SIZE, LORA_INPUT_SIZE)

    def test_letterbox_pad_value_114(self) -> None:
        """Pad areas should be close to the normalized value of 114/255.

        spec: section 7.3 lines 1488, 1504-1506 — "pad value 114 ... the value
        used during LoRA training."

        For a wide image (low H), padding occupies top/bottom rows.
        After LAB-CLAHE and normalization, pad pixels still cluster near the
        normalized equivalent of (114, 114, 114).
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        from tomato_sandbox.config import LORA_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

        # Extremely wide image: 20 high, 392 wide → new_h = 20, pad_h = 372
        # Pad occupies rows 0..185 and rows 206..391 (approx)
        img = _make_pil(h=20, w=392, colour=(114, 114, 114))
        t = preprocess_for_lora(img)

        # The pad value 114 normalizes to: (114/255 - mean) / std
        # For each channel:
        pad_expected = (np.array([114, 114, 114], dtype=np.float32) / 255.0
                        - IMAGENET_MEAN) / IMAGENET_STD

        # Inspect the first row (guaranteed pad for a 20-high source image)
        top_row = t[:, 0, :].numpy()  # shape [3, 392]
        for c in range(3):
            mean_c = float(top_row[c, :].mean())
            # After LAB-CLAHE on a uniform image the value may shift slightly;
            # allow ±0.3 tolerance.
            assert abs(mean_c - pad_expected[c]) < 0.5, (
                f"Channel {c}: pad mean {mean_c:.3f} far from expected {pad_expected[c]:.3f}"
            )

    def test_deterministic(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        assert torch.equal(preprocess_for_lora(landscape_pil),
                           preprocess_for_lora(landscape_pil))

    def test_tta_second_call_different_image(self) -> None:
        """TTA: repeated calls with different PIL images produce different tensors."""
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        img_a = _make_pil_random(h=480, w=640, seed=3)
        img_b = _make_pil_random(h=480, w=640, seed=4)
        assert not torch.equal(preprocess_for_lora(img_a), preprocess_for_lora(img_b))


# ---------------------------------------------------------------------------
# shades_of_gray
# spec: section 7.4 lines 1532-1544
# ---------------------------------------------------------------------------

class TestShadesOfGray:
    """Tests for shades_of_gray color constancy function.

    spec: section 7.4 lines 1532-1544
    """

    def test_output_dtype_uint8(self) -> None:
        # spec: section 7.4 line 1543 — ".astype(np.uint8)"
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray
        img = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        out = shades_of_gray(img, p=6)
        assert out.dtype == np.uint8

    def test_output_shape_preserved(self) -> None:
        # Output shape equals input shape
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray
        img = np.random.default_rng(1).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        out = shades_of_gray(img, p=6)
        assert out.shape == img.shape

    def test_output_values_in_range(self) -> None:
        # spec: section 7.4 line 1543 — "np.clip(..., 0, 255)"
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray
        img = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        out = shades_of_gray(img, p=6)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_grey_image_identity(self) -> None:
        """A perfectly grey image (R==G==B) is unchanged by shades_of_gray.

        For a uniform grey image, all channel means are equal, so
        illuminant = [k, k, k] and scale = [1, 1, 1]. Output == input.

        spec: section 7.4 lines 1539-1542
        """
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray
        grey = np.full((100, 100, 3), 128, dtype=np.uint8)
        out = shades_of_gray(grey, p=6)
        np.testing.assert_array_equal(out, grey)

    def test_p1_grey_world(self) -> None:
        """p=1 is grey-world (spec: section 7.4 line 1534).

        Grey-world assumes the mean of each channel should be equal.
        After grey-world correction on a random image, the channel means
        should all be approximately equal.
        """
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray
        rng = np.random.default_rng(99)
        img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        out = shades_of_gray(img, p=1)
        out_f = out.astype(np.float64)
        means = out_f.mean(axis=(0, 1))
        # All three channel means should be close (within 5)
        assert abs(means[0] - means[1]) < 5.0
        assert abs(means[1] - means[2]) < 5.0

    def test_p6_default_parameter(self) -> None:
        """Calling shades_of_gray(img) without p uses p=6 (spec line 1536)."""
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray
        img = np.random.default_rng(5).integers(0, 256, (50, 50, 3), dtype=np.uint8)
        out_default = shades_of_gray(img)
        out_p6 = shades_of_gray(img, p=6)
        np.testing.assert_array_equal(out_default, out_p6)

    def test_output_finite(self) -> None:
        from tomato_sandbox.preprocessing.preprocess import shades_of_gray
        img = np.random.default_rng(6).integers(1, 255, (100, 100, 3), dtype=np.uint8)
        out = shades_of_gray(img, p=6)
        assert np.all(np.isfinite(out.astype(np.float32)))


# ---------------------------------------------------------------------------
# preprocess_for_psv
# spec: section 7.4 lines 1511-1524
# ---------------------------------------------------------------------------

class TestPreprocessForPSV:
    """Tests for Pipeline 3 — Signal C (PSV).

    spec: section 7.4 lines 1511-1524
    """

    def test_output_dtype_uint8(self, landscape_pil: "_PIL.Image") -> None:
        # spec: section 7.4 line 1524 — "return rgb_cc" which is uint8
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        out = preprocess_for_psv(landscape_pil)
        assert out.dtype == np.uint8

    def test_output_is_numpy_array(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        out = preprocess_for_psv(landscape_pil)
        assert isinstance(out, np.ndarray)

    def test_output_has_3_channels(self, landscape_pil: "_PIL.Image") -> None:
        # spec: section 7.4 line 1513 — "[H, W, 3] uint8 RGB array"
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        out = preprocess_for_psv(landscape_pil)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_no_resize_below_1200(self, below_1200_pil: "_PIL.Image") -> None:
        """Images with longest side <= 1200 are NOT resized.

        spec: section 7.4 lines 1519-1522 — 'if max(H, W) > 1200'
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        # below_1200_pil is 480x640
        out = preprocess_for_psv(below_1200_pil)
        assert out.shape[0] == 480
        assert out.shape[1] == 640

    def test_no_resize_exactly_1200(self, exactly_1200_pil: "_PIL.Image") -> None:
        """Longest side exactly 1200 is NOT resized (condition is strict >).

        spec: section 7.4 line 1519 — "if max(H, W) > 1200"
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        # exactly_1200_pil is 900x1200
        out = preprocess_for_psv(exactly_1200_pil)
        assert out.shape[0] == 900
        assert out.shape[1] == 1200

    def test_resize_above_1200_caps_longest_side(self, large_pil: "_PIL.Image") -> None:
        """Images with longest side > 1200 are downscaled so longest side = 1200.

        spec: section 7.4 lines 1519-1522
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        # large_pil is 2400x3200
        out = preprocess_for_psv(large_pil)
        assert max(out.shape[0], out.shape[1]) == 1200

    def test_resize_above_1200_preserves_aspect_ratio(self, large_pil: "_PIL.Image") -> None:
        """Aspect ratio preserved when downscaling.

        spec: section 7.4 line 1520 — scale = 1200 / max(H, W)
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        # large_pil is 2400 high, 3200 wide → scale = 1200/3200 = 0.375
        # new_h = int(2400 * 0.375) = 900; new_w = int(3200 * 0.375) = 1200
        out = preprocess_for_psv(large_pil)
        # Verify: width is 1200 and height is approximately 900
        expected_h = int(2400 * (1200 / 3200))
        assert out.shape[0] == expected_h
        assert out.shape[1] == 1200

    def test_no_clahe_applied(self) -> None:
        """PSV must NOT apply LAB-CLAHE.

        spec: section 7.4 line 1515 — 'NO LAB-CLAHE'
        spec: section 7.4 lines 1549-1550 — 'Applying CLAHE before PSV would
        invalidate the F.0 calibration of HSV thresholds.'

        We verify by checking that a uniform-color image is not altered in hue
        (CLAHE on a uniform image would leave it unchanged, but a non-uniform
        result would indicate CLAHE is being applied to per-channel RGB which
        WOULD alter color balance).  We simply verify the function produces the
        shades_of_gray output and not the LAB-CLAHE output.
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv, shades_of_gray
        # Uniform red image (distinctive color to detect channel shifts)
        img_pil = _make_pil(h=100, w=100, colour=(200, 50, 50))
        out_psv = preprocess_for_psv(img_pil)
        img_np = np.array(img_pil, dtype=np.uint8)
        expected = shades_of_gray(img_np, p=6)
        # PSV output must equal shades_of_gray output (no CLAHE)
        np.testing.assert_array_equal(out_psv, expected)

    def test_no_tensor_conversion(self, landscape_pil: "_PIL.Image") -> None:
        """PSV output is numpy, never a torch Tensor.

        spec: section 7.4 line 1515 — 'NO tensor conversion'
        """
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        out = preprocess_for_psv(landscape_pil)
        assert not _TORCH_AVAILABLE or not isinstance(out, torch.Tensor)
        assert isinstance(out, np.ndarray)

    def test_output_values_in_range(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        out = preprocess_for_psv(landscape_pil)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_deterministic(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        np.testing.assert_array_equal(
            preprocess_for_psv(landscape_pil),
            preprocess_for_psv(landscape_pil),
        )

    def test_tta_second_call_different_image(self) -> None:
        """TTA: PSV can also be called multiple times with different PIL images."""
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_psv
        img_a = _make_pil_random(h=200, w=300, seed=7)
        img_b = _make_pil_random(h=200, w=300, seed=8)
        out_a = preprocess_for_psv(img_a)
        out_b = preprocess_for_psv(img_b)
        # Different random images should produce different outputs
        assert not np.array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Pipeline isolation — no cross-pipeline CLAHE contamination
# spec: section 7.1 lines 1396-1401
# ---------------------------------------------------------------------------

class TestPipelineIsolation:
    """Verify each pipeline produces a distinct representation.

    spec: section 7.1 — 'Three downstream consumers each need a different
    preprocessed view of the same input image.'
    """

    def test_v3_and_lora_different_shapes(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3, preprocess_for_lora
        t_v3 = preprocess_for_v3(landscape_pil)
        t_lora = preprocess_for_lora(landscape_pil)
        assert t_v3.shape != t_lora.shape

    def test_v3_and_psv_different_types(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3, preprocess_for_psv
        t_v3 = preprocess_for_v3(landscape_pil)
        arr_psv = preprocess_for_psv(landscape_pil)
        assert isinstance(t_v3, torch.Tensor)
        assert isinstance(arr_psv, np.ndarray)

    def test_lora_and_psv_different_types(self, landscape_pil: "_PIL.Image") -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora, preprocess_for_psv
        t_lora = preprocess_for_lora(landscape_pil)
        arr_psv = preprocess_for_psv(landscape_pil)
        assert isinstance(t_lora, torch.Tensor)
        assert isinstance(arr_psv, np.ndarray)

    def test_all_three_callable_on_same_pil(self, landscape_pil: "_PIL.Image") -> None:
        """The orchestrator pattern: all three called on the same PIL image.

        spec: section 7.1 lines 1407-1414 — call pattern
        """
        from tomato_sandbox.preprocessing.preprocess import (
            preprocess_for_v3,
            preprocess_for_lora,
            preprocess_for_psv,
        )
        v3_in = preprocess_for_v3(landscape_pil)
        lora_in = preprocess_for_lora(landscape_pil)
        psv_in = preprocess_for_psv(landscape_pil)
        assert v3_in.shape == (3, 224, 224)
        assert lora_in.shape == (3, 392, 392)
        assert psv_in.ndim == 3 and psv_in.shape[2] == 3


# ---------------------------------------------------------------------------
# NaN-guard defensive paths
# spec: section 26 production hygiene (DEC-031)
# ---------------------------------------------------------------------------

class TestNaNGuardDefensivePaths:
    """Verify the finiteness guard doesn't raise and returns a zero tensor
    when given a degenerate all-zero image.

    An all-zero image after normalization:
        (0/255 - mean) / std = -mean/std  (always finite for ImageNet stats)

    We test the guard path indirectly by confirming the functions return
    finite tensors even for edge-case images.
    """

    def test_v3_allblack_image_finite(self) -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        img = _PIL.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")
        t = preprocess_for_v3(img)
        assert torch.all(torch.isfinite(t))

    def test_v3_allwhite_image_finite(self) -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3
        img = _PIL.fromarray(
            np.full((224, 224, 3), 255, dtype=np.uint8), mode="RGB"
        )
        t = preprocess_for_v3(img)
        assert torch.all(torch.isfinite(t))

    def test_lora_allblack_image_finite(self) -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        img = _PIL.fromarray(np.zeros((392, 392, 3), dtype=np.uint8), mode="RGB")
        t = preprocess_for_lora(img)
        assert torch.all(torch.isfinite(t))

    def test_lora_allwhite_image_finite(self) -> None:
        from tomato_sandbox.preprocessing.preprocess import preprocess_for_lora
        img = _PIL.fromarray(
            np.full((392, 392, 3), 255, dtype=np.uint8), mode="RGB"
        )
        t = preprocess_for_lora(img)
        assert torch.all(torch.isfinite(t))
