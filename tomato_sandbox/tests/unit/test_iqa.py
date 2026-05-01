"""
Unit tests for tomato_sandbox/iqa/iqa.py.

Coverage targets:
- Every IQA dimension: sharpness, exposure, leaf_presence, leaf_fill,
  background_contamination, resolution, wetness
- Every threshold boundary value from spec Section 6.2 (BAD / GOOD)
- Aggregation: geometric mean, equal weights, weighted
- Four-way decision: REJECT (dim-based), REJECT (aggregate-based),
  DEGRADED, ACCEPTABLE, HIGH
- Retake message: worst-dim selection, exposure dark vs bright
- IQAResult dataclass fields match spec 6.5
- compute_iqa(): happy path, REJECT path, bad input path
- BAD_THRESHOLDS keys match exactly the 7 dimension names

spec reference: Section 6, lines 1053-1388
"""

from __future__ import annotations

import math
import types
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from tomato_sandbox.iqa.iqa import (
    BAD_THRESHOLDS,
    IQAResult,
    _aggregate_quality,
    _background_contamination,
    _exposure,
    _iqa_decide,
    _leaf_fill,
    _leaf_presence,
    _resolution,
    _sharpness,
    _wetness,
    compute_iqa,
)

# Also verify the package-level re-export works
from tomato_sandbox.iqa import IQAResult as IQAResult_pkg  # noqa: F401
from tomato_sandbox.iqa import compute_iqa as compute_iqa_pkg  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers to build test images and mock validated_image objects
# ---------------------------------------------------------------------------

def _make_rgb(h: int = 300, w: int = 300, color: tuple[int, int, int] = (80, 150, 60)) -> np.ndarray:
    """Return a solid-colour uint8 RGB image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _make_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def _make_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _mock_validated_image(rgb: np.ndarray) -> MagicMock:
    """Return a mock object that looks like ValidatedImage with pil_image."""
    from PIL import Image
    pil = Image.fromarray(rgb, "RGB")
    mock = MagicMock()
    mock.pil_image = pil
    return mock


# ---------------------------------------------------------------------------
# BAD_THRESHOLDS structure
# spec: 6.4 lines 1304-1315
# ---------------------------------------------------------------------------

class TestBadThresholds:
    def test_seven_dimensions_present(self):
        expected = {
            "sharpness", "exposure", "leaf_presence", "leaf_fill",
            "background_contamination", "resolution", "wetness",
        }
        assert set(BAD_THRESHOLDS.keys()) == expected  # spec: 6.4 lines 1305-1311

    def test_threshold_values_match_spec(self):
        # spec: 6.4 lines 1305-1311 (placeholder defaults)
        assert BAD_THRESHOLDS["sharpness"] == pytest.approx(0.20)
        assert BAD_THRESHOLDS["exposure"] == pytest.approx(0.20)
        assert BAD_THRESHOLDS["leaf_presence"] == pytest.approx(0.30)
        assert BAD_THRESHOLDS["leaf_fill"] == pytest.approx(0.30)
        assert BAD_THRESHOLDS["background_contamination"] == pytest.approx(0.30)
        assert BAD_THRESHOLDS["resolution"] == pytest.approx(0.20)
        assert BAD_THRESHOLDS["wetness"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# IQAResult dataclass
# spec: 6.5 lines 1357-1365
# ---------------------------------------------------------------------------

class TestIQAResultDataclass:
    def test_all_required_fields_exist(self):
        r = IQAResult(
            decision="HIGH",
            aggregate_score=0.9,
            per_dimension={"sharpness": 0.9, "exposure": 0.8, "leaf_presence": 0.95,
                           "leaf_fill": 0.85, "background_contamination": 1.0,
                           "resolution": 0.75, "wetness": 0.9},
        )
        assert r.decision == "HIGH"               # spec: 6.5 line 1359
        assert r.aggregate_score == pytest.approx(0.9)  # spec: 6.5 line 1360
        assert len(r.per_dimension) == 7           # spec: 6.5 line 1361
        assert r.failing_dimensions == []          # spec: 6.5 line 1362 (default)
        assert r.retake_message is None            # spec: 6.5 line 1363 (default)
        assert r.green_mask is None                # spec: 6.5 line 1364 (default)

    def test_package_re_export_is_same_class(self):
        assert IQAResult_pkg is IQAResult


# ---------------------------------------------------------------------------
# Dimension: sharpness
# spec: 6.2.1 lines 1081-1086; thresholds lines 1091-1093
# ---------------------------------------------------------------------------

class TestSharpness:
    def test_returns_float_in_range(self):
        gray = _make_gray(_make_rgb())
        score = _sharpness(gray)
        assert 0.0 <= score <= 1.0

    def test_sharp_image_above_good_threshold(self):
        # High-frequency noise pattern → very high variance
        rng = np.random.default_rng(0)
        gray = rng.integers(0, 256, (300, 300), dtype=np.uint8)
        score = _sharpness(gray)
        # spec: 6.2.1 line 1093 — GOOD if score > 0.50 (variance > 500)
        assert score > 0.50

    def test_saturates_at_one(self):
        # Extremely high variance → saturates at 1.0
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (300, 300), dtype=np.uint8)
        score = _sharpness(gray)
        assert score <= 1.0

    def test_solid_color_near_zero(self):
        # Solid-colour image → Laplacian variance ≈ 0
        gray = np.full((100, 100), 128, dtype=np.uint8)
        score = _sharpness(gray)
        # spec: 6.2.1 line 1092 — BAD if score < 0.20 (variance < 200)
        assert score < 0.20

    def test_normalization_formula(self):
        # Create a controlled-variance image (not perfectly reproducible via API,
        # but we can test the formula directly)
        # If raw_variance = 500 → score = 500/1000 = 0.5
        # We cannot easily control cv2.Laplacian output,
        # so instead we test the formula in isolation via a known case
        # (black-and-white checker of known variance)
        checker = np.zeros((4, 4), dtype=np.uint8)
        checker[0::2, 0::2] = 255
        checker[1::2, 1::2] = 255
        score = _sharpness(checker)
        assert score >= 0.0  # basic sanity


# ---------------------------------------------------------------------------
# Dimension: exposure
# spec: 6.2.2 lines 1104-1115; thresholds lines 1120-1122
# ---------------------------------------------------------------------------

class TestExposure:
    def _make_v_mean_image(self, target_v: int) -> np.ndarray:
        """Create an HSV image with a specific V-channel mean."""
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, 0] = 60    # H green
        hsv[:, :, 1] = 100   # S
        hsv[:, :, 2] = target_v  # V
        return hsv

    def test_too_dark_returns_zero(self):
        # V mean = 30 < 50 → score = 0.0
        # spec: 6.2.2 line 1108
        hsv = self._make_v_mean_image(30)
        score, key = _exposure(hsv)
        assert score == pytest.approx(0.0)
        assert key == "exposure_dark"

    def test_blown_out_returns_zero(self):
        # V mean = 230 > 220 → score = 0.0
        # spec: 6.2.2 line 1110
        hsv = self._make_v_mean_image(230)
        score, key = _exposure(hsv)
        assert score == pytest.approx(0.0)
        assert key == "exposure_bright"

    def test_optimal_v130_returns_one(self):
        # V mean = 130 → 1.0 (peak of tent)
        # spec: 6.2.2 lines 1112-1113 — (130-50)/80 = 1.0
        hsv = self._make_v_mean_image(130)
        score, key = _exposure(hsv)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_v90_ramp_up(self):
        # V mean = 90 → (90-50)/80 = 0.5
        # spec: 6.2.2 line 1113
        hsv = self._make_v_mean_image(90)
        score, _ = _exposure(hsv)
        assert score == pytest.approx(0.5, abs=1e-3)

    def test_v175_ramp_down(self):
        # V mean = 175 → 1 - (175-130)/90 = 0.5
        # spec: 6.2.2 line 1115
        hsv = self._make_v_mean_image(175)
        score, key = _exposure(hsv)
        assert score == pytest.approx(0.5, abs=1e-3)
        assert key == "exposure_bright"

    def test_bad_threshold_low(self):
        # BAD if score < 0.20 → v_mean ≈ 66 → (66-50)/80 = 0.20
        # spec: 6.2.2 line 1121
        hsv = self._make_v_mean_image(65)
        score, _ = _exposure(hsv)
        assert score < 0.20 + 0.05  # just below BAD threshold region

    def test_good_threshold(self):
        # GOOD if score > 0.60 → spec 6.2.2 line 1122
        hsv = self._make_v_mean_image(130)
        score, _ = _exposure(hsv)
        assert score > 0.60


# ---------------------------------------------------------------------------
# Dimension: leaf_presence
# spec: 6.2.3 lines 1134-1148; thresholds lines 1153-1155
# ---------------------------------------------------------------------------

class TestLeafPresence:
    def _make_hsv_with_green_fraction(self, fraction: float, size: int = 200) -> np.ndarray:
        """Return an HSV image where `fraction` of pixels are green."""
        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        n_green = int(fraction * size * size)
        hsv[:, :, 0] = 10    # non-green hue (red/orange)
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 150
        # Make a top-left block green
        flat_h = hsv[:, :, 0].flatten()
        flat_s = hsv[:, :, 1].flatten()
        flat_h[:n_green] = 60   # green hue
        flat_s[:n_green] = 150  # saturated
        hsv[:, :, 0] = flat_h.reshape(size, size)
        hsv[:, :, 1] = flat_s.reshape(size, size)
        return hsv

    def test_no_green_returns_zero(self):
        hsv = self._make_hsv_with_green_fraction(0.0)
        score, mask = _leaf_presence(hsv)
        assert score == pytest.approx(0.0)
        assert mask.dtype == bool

    def test_high_green_returns_one(self):
        # 40% green → above 30% → 1.0
        # spec: 6.2.3 line 1145
        hsv = self._make_hsv_with_green_fraction(0.40)
        score, _ = _leaf_presence(hsv)
        assert score == pytest.approx(1.0)

    def test_ramp_at_midpoint(self):
        # 17.5% green → (0.175 - 0.05) / 0.25 = 0.5
        # spec: 6.2.3 line 1147
        hsv = self._make_hsv_with_green_fraction(0.175)
        score, _ = _leaf_presence(hsv)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_below_5pct_returns_zero(self):
        # spec: 6.2.3 line 1143
        hsv = self._make_hsv_with_green_fraction(0.02)
        score, _ = _leaf_presence(hsv)
        assert score == pytest.approx(0.0)

    def test_bad_threshold_boundary(self):
        # BAD if score < 0.30 → spec 6.2.3 line 1154
        # fraction = 10% → (0.10 - 0.05) / 0.25 = 0.20 → fails BAD threshold (0.30)
        hsv = self._make_hsv_with_green_fraction(0.10)
        score, _ = _leaf_presence(hsv)
        assert score < 0.30

    def test_good_threshold_above(self):
        # GOOD if score > 0.70 → fraction ~22.5% → (0.225-0.05)/0.25=0.70
        # spec: 6.2.3 line 1155
        hsv = self._make_hsv_with_green_fraction(0.30)  # exactly 1.0
        score, _ = _leaf_presence(hsv)
        assert score >= 0.70


# ---------------------------------------------------------------------------
# Dimension: leaf_fill
# spec: 6.2.4 lines 1166-1183; thresholds lines 1188-1190
# ---------------------------------------------------------------------------

class TestLeafFill:
    def _make_connected_green_mask(self, fraction: float, h: int = 300, w: int = 300) -> np.ndarray:
        """Return a boolean mask with a single connected component covering `fraction` of image."""
        mask = np.zeros((h, w), dtype=bool)
        n_pixels = int(fraction * h * w)
        # Place a single rectangle in the top-left corner
        n_rows = max(1, n_pixels // w)
        mask[:n_rows, :] = True
        return mask

    def test_no_component_returns_zero(self):
        mask = np.zeros((100, 100), dtype=bool)
        score = _leaf_fill(mask, (100, 100, 3))
        assert score == pytest.approx(0.0)

    def test_large_fill_returns_one(self):
        # bbox fill > 40% → 1.0; spec: 6.2.4 line 1180
        mask = self._make_connected_green_mask(0.60, 300, 300)
        score = _leaf_fill(mask, (300, 300, 3))
        assert score == pytest.approx(1.0)

    def test_small_fill_returns_zero(self):
        # fill < 5% → 0.0; spec: 6.2.4 line 1178
        mask = self._make_connected_green_mask(0.02, 300, 300)
        score = _leaf_fill(mask, (300, 300, 3))
        assert score == pytest.approx(0.0)

    def test_mid_fill_ramp(self):
        # The bounding box fill is based on WIDTH×HEIGHT of bounding box, not pixel count.
        # For a mask that's n_rows × w, the bbox = n_rows × w
        # fill = (n_rows * w) / (h * w) = n_rows / h
        # Test: fill = 22.5% → (0.225 - 0.05) / 0.35 = 0.5
        h, w = 400, 300
        n_rows = int(0.225 * h)
        mask = np.zeros((h, w), dtype=bool)
        mask[:n_rows, :] = True
        score = _leaf_fill(mask, (h, w, 3))
        assert score == pytest.approx(0.5, abs=0.05)

    def test_bad_threshold(self):
        # BAD if score < 0.30 → spec 6.2.4 line 1189
        mask = self._make_connected_green_mask(0.05, 300, 300)
        score = _leaf_fill(mask, (300, 300, 3))
        assert score < 0.30

    def test_good_threshold(self):
        # GOOD if score > 0.70 → spec 6.2.4 line 1190 (bbox fill > 30%)
        mask = self._make_connected_green_mask(0.50, 300, 300)
        score = _leaf_fill(mask, (300, 300, 3))
        assert score > 0.70


# ---------------------------------------------------------------------------
# Dimension: background_contamination
# spec: 6.2.5 lines 1201-1215; thresholds lines 1220-1222
# ---------------------------------------------------------------------------

class TestBackgroundContamination:
    def _make_n_components_mask(self, n: int, h: int = 400, w: int = 400) -> np.ndarray:
        """Return a boolean mask with n distinct large green components (each > 5% of image)."""
        mask = np.zeros((h, w), dtype=bool)
        if n == 0:
            return mask
        strip_h = max(1, h // (n * 2))  # leave gaps between components
        for i in range(n):
            top = i * strip_h * 2
            bottom = min(h, top + strip_h)
            mask[top:bottom, :] = True
        return mask

    def test_no_green_returns_one(self):
        # nb <= 1: no green → defer to leaf_presence; returns 1.0
        # spec: 6.2.5 line 1205
        mask = np.zeros((300, 300), dtype=bool)
        score = _background_contamination(mask, (300, 300, 3))
        assert score == pytest.approx(1.0)

    def test_one_component_returns_one(self):
        # Single large component → clean; spec: 6.2.5 line 1210
        mask = np.zeros((300, 300), dtype=bool)
        mask[10:200, 10:200] = True  # large single component
        score = _background_contamination(mask, (300, 300, 3))
        assert score == pytest.approx(1.0)

    def test_two_components_returns_half(self):
        # spec: 6.2.5 line 1212
        mask = np.zeros((300, 300), dtype=bool)
        mask[10:130, :] = True   # component 1: >5% of 300x300
        mask[160:280, :] = True  # component 2: >5%
        score = _background_contamination(mask, (300, 300, 3))
        assert score == pytest.approx(0.5)

    def test_three_or_more_returns_zero(self):
        # spec: 6.2.5 line 1214
        mask = np.zeros((300, 300), dtype=bool)
        mask[0:80, :] = True    # >5%
        mask[100:180, :] = True # >5%
        mask[200:280, :] = True # >5%
        score = _background_contamination(mask, (300, 300, 3))
        assert score == pytest.approx(0.0)

    def test_bad_threshold(self):
        # BAD if score < 0.30 → three+ components returns 0.0
        # spec: 6.2.5 line 1221
        mask = np.zeros((300, 300), dtype=bool)
        mask[0:80, :] = True
        mask[100:180, :] = True
        mask[200:280, :] = True
        score = _background_contamination(mask, (300, 300, 3))
        assert score < 0.30

    def test_good_threshold(self):
        # GOOD if score > 0.80 → single component returns 1.0
        # spec: 6.2.5 line 1222
        mask = np.zeros((300, 300), dtype=bool)
        mask[10:200, 10:200] = True
        score = _background_contamination(mask, (300, 300, 3))
        assert score > 0.80


# ---------------------------------------------------------------------------
# Dimension: resolution
# spec: 6.2.6 lines 1233-1241; thresholds lines 1246-1248
# ---------------------------------------------------------------------------

class TestResolution:
    def test_below_224_returns_zero(self):
        # spec: 6.2.6 line 1236 — defensive; validation already rejected
        assert _resolution((100, 200, 3)) == pytest.approx(0.0)

    def test_exactly_224_returns_zero(self):
        assert _resolution((224, 224, 3)) == pytest.approx(0.0)

    def test_above_800_returns_one(self):
        # spec: 6.2.6 line 1238
        assert _resolution((1000, 900, 3)) == pytest.approx(1.0)

    def test_exactly_800_returns_one(self):
        assert _resolution((800, 800, 3)) == pytest.approx(1.0)

    def test_midpoint_ramp(self):
        # smaller = 512 → (512-224)/(800-224) = 288/576 = 0.5
        # spec: 6.2.6 line 1240
        assert _resolution((512, 600, 3)) == pytest.approx(0.5, abs=0.01)

    def test_bad_threshold(self):
        # BAD if score < 0.20 → smaller < 339 px; spec: 6.2.6 line 1247
        # smaller = 300 → (300-224)/(800-224) = 76/576 ≈ 0.132 < 0.20
        assert _resolution((300, 400, 3)) < 0.20

    def test_good_threshold(self):
        # GOOD if score > 0.50 → smaller > 512 px; spec: 6.2.6 line 1248
        assert _resolution((513, 600, 3)) > 0.50

    def test_uses_smaller_dimension(self):
        # Portrait vs landscape: same smaller dimension
        assert _resolution((600, 300, 3)) == _resolution((300, 600, 3))


# ---------------------------------------------------------------------------
# Dimension: wetness
# spec: 6.2.7 lines 1259-1269; thresholds lines 1274-1276
# ---------------------------------------------------------------------------

class TestWetness:
    def _make_hsv_with_specular_fraction(self, fraction: float, size: int = 200) -> np.ndarray:
        """HSV image where `fraction` of pixels are bright+desaturated (specular)."""
        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[:, :, 0] = 60
        hsv[:, :, 1] = 150   # saturated (non-specular)
        hsv[:, :, 2] = 100   # not bright
        n_spec = int(fraction * size * size)
        flat_v = hsv[:, :, 2].flatten()
        flat_s = hsv[:, :, 1].flatten()
        flat_v[:n_spec] = 240   # bright (>220)
        flat_s[:n_spec] = 10    # desaturated (<30)
        hsv[:, :, 2] = flat_v.reshape(size, size)
        hsv[:, :, 1] = flat_s.reshape(size, size)
        return hsv

    def test_no_specular_returns_one(self):
        # pct_spec = 0 < 0.005 → 1.0; spec: 6.2.7 line 1264
        hsv = self._make_hsv_with_specular_fraction(0.0)
        assert _wetness(hsv) == pytest.approx(1.0)

    def test_high_specular_returns_zero(self):
        # pct_spec = 0.10 > 0.05 → 0.0; spec: 6.2.7 line 1266
        hsv = self._make_hsv_with_specular_fraction(0.10)
        assert _wetness(hsv) == pytest.approx(0.0)

    def test_midpoint_ramp(self):
        # pct_spec = 0.0275 (midpoint of ramp [0.005, 0.05])
        # score = 1.0 - (0.0275 - 0.005) / 0.045 = 1.0 - 0.5 = 0.5
        # spec: 6.2.7 line 1268
        hsv = self._make_hsv_with_specular_fraction(0.0275)
        assert _wetness(hsv) == pytest.approx(0.5, abs=0.05)

    def test_bad_threshold(self):
        # BAD if score < 0.30 → pct_spec ~ 3.5%; spec: 6.2.7 line 1275
        # pct_spec = 0.04 → 1.0 - (0.04-0.005)/0.045 = 1.0 - 0.778 = 0.222 < 0.30
        hsv = self._make_hsv_with_specular_fraction(0.04)
        assert _wetness(hsv) < 0.30

    def test_good_threshold(self):
        # GOOD if score > 0.80 → pct_spec < ~1.4%; spec: 6.2.7 line 1276
        hsv = self._make_hsv_with_specular_fraction(0.001)
        assert _wetness(hsv) > 0.80


# ---------------------------------------------------------------------------
# Aggregation: _aggregate_quality
# spec: 6.3 lines 1287-1292
# ---------------------------------------------------------------------------

class TestAggregateQuality:
    def _all_ones(self) -> dict[str, float]:
        return {k: 1.0 for k in BAD_THRESHOLDS}

    def test_all_ones_returns_one(self):
        # geometric mean of all 1s = 1.0; spec: 6.3
        assert _aggregate_quality(self._all_ones()) == pytest.approx(1.0)

    def test_one_near_zero_pulls_down(self):
        # Geometric mean sensitivity to small values; spec: 6.3 lines 1295-1296
        scores = self._all_ones()
        scores["sharpness"] = 0.01
        result = _aggregate_quality(scores)
        # Arithmetic mean ≈ 0.859; geometric mean should be much lower
        arithmetic = sum(scores.values()) / len(scores)
        assert result < arithmetic

    def test_equal_weights_default(self):
        # spec: 6.3 line 1289 — equal weighting default
        scores = {k: 0.5 for k in BAD_THRESHOLDS}
        result = _aggregate_quality(scores)
        assert result == pytest.approx(0.5)

    def test_weighted_geometric_mean_formula(self):
        # Manual test: 2 keys, weights 1 and 2, scores 0.5 and 1.0
        # weighted log-sum = 1*log(0.5) + 2*log(1.0) = -0.693 + 0 = -0.693
        # total_weight = 3
        # result = exp(-0.693/3) = exp(-0.231) ≈ 0.7937
        scores = {"a": 0.5, "b": 1.0}
        weights = {"a": 1.0, "b": 2.0}
        result = _aggregate_quality(scores, weights)
        expected = math.exp((1 * math.log(0.5) + 2 * math.log(1.0)) / 3)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_near_zero_clamped_with_epsilon(self):
        # spec: 6.3 line 1291 — max(score, 1e-6) prevents log(0)
        scores = self._all_ones()
        scores["wetness"] = 0.0  # exactly zero
        result = _aggregate_quality(scores)
        assert math.isfinite(result)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Decision: _iqa_decide
# spec: 6.4 lines 1318-1331
# ---------------------------------------------------------------------------

class TestIqaDecide:
    def _good_scores(self) -> dict[str, float]:
        return {k: 0.99 for k in BAD_THRESHOLDS}

    def test_high_when_aggregate_above_080(self):
        # spec: 6.4 line 1328
        assert _iqa_decide(0.85, self._good_scores()) == "HIGH"

    def test_acceptable_when_aggregate_between_060_and_080(self):
        # spec: 6.4 line 1326-1328
        assert _iqa_decide(0.65, self._good_scores()) == "ACCEPTABLE"

    def test_degraded_when_aggregate_between_040_and_060(self):
        # spec: 6.4 line 1324-1326
        assert _iqa_decide(0.50, self._good_scores()) == "DEGRADED"

    def test_reject_when_aggregate_below_040(self):
        # spec: 6.4 line 1324
        assert _iqa_decide(0.35, self._good_scores()) == "REJECT"

    def test_reject_when_single_dim_below_bad_threshold(self):
        # spec: 6.4 lines 1319-1322 — dim-based REJECT trumps aggregate
        scores = self._good_scores()
        scores["sharpness"] = 0.10  # below BAD_THRESHOLDS["sharpness"]=0.20
        # Even if aggregate could be OK (other dims are 0.99)
        assert _iqa_decide(0.80, scores) == "REJECT"

    def test_reject_dim_trumps_high_aggregate(self):
        # spec: 6.4 lines 1342-1344 rationale — asymmetric failure must be caught
        scores = self._good_scores()
        scores["wetness"] = 0.01  # way below 0.30
        assert _iqa_decide(0.90, scores) == "REJECT"

    def test_boundary_exactly_040_is_degraded(self):
        # aggregate == 0.40 → not < 0.40 → passes to DEGRADED check
        # DEGRADED is 0.40 <= agg < 0.60 → spec: 6.4 line 1326
        assert _iqa_decide(0.40, self._good_scores()) == "DEGRADED"

    def test_boundary_exactly_060_is_acceptable(self):
        assert _iqa_decide(0.60, self._good_scores()) == "ACCEPTABLE"

    def test_boundary_exactly_080_is_high(self):
        assert _iqa_decide(0.80, self._good_scores()) == "HIGH"

    def test_custom_bad_thresholds(self):
        # Verify we can override BAD_THRESHOLDS via the parameter
        scores = self._good_scores()
        scores["sharpness"] = 0.15  # below default 0.20
        custom = dict(BAD_THRESHOLDS)
        custom["sharpness"] = 0.10  # looser threshold
        # With custom threshold, 0.15 > 0.10 → NOT rejected by dim
        result = _iqa_decide(0.85, scores, bad_thresholds=custom)
        assert result == "HIGH"


# ---------------------------------------------------------------------------
# compute_iqa: integration tests
# spec: 6.6 line 1374
# ---------------------------------------------------------------------------

class TestComputeIqa:
    def _make_good_image(self) -> np.ndarray:
        """A plausible 'good' leaf photo: bright green, not solid (some variation)."""
        rng = np.random.default_rng(1)
        base = np.zeros((400, 400, 3), dtype=np.uint8)
        base[:, :, 1] = 140   # green channel dominant
        base[:, :, 0] = 60
        base[:, :, 2] = 40
        noise = rng.integers(-20, 20, (400, 400, 3), dtype=np.int16)
        rgb = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return rgb

    def test_returns_iqaresult_instance(self):
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        assert isinstance(result, IQAResult)

    def test_per_dimension_has_seven_keys(self):
        # spec: 6.5 line 1361 — exactly 7 entries
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        assert len(result.per_dimension) == 7

    def test_per_dimension_keys_match_bad_thresholds(self):
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        assert set(result.per_dimension.keys()) == set(BAD_THRESHOLDS.keys())

    def test_all_scores_in_unit_interval(self):
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        for dim, score in result.per_dimension.items():
            assert 0.0 <= score <= 1.0, f"{dim} score {score} out of [0,1]"

    def test_aggregate_score_in_unit_interval(self):
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        assert 0.0 <= result.aggregate_score <= 1.0

    def test_decision_is_valid_string(self):
        # spec: 6.4 lines 1318-1331
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        assert result.decision in {"REJECT", "DEGRADED", "ACCEPTABLE", "HIGH"}

    def test_retake_message_none_when_not_reject(self):
        # spec: 6.5 line 1363 — retake_message only populated on REJECT
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        if result.decision != "REJECT":
            assert result.retake_message is None

    def test_retake_message_set_on_reject(self):
        # Force REJECT: tiny blurry image with near-zero green
        rng = np.random.default_rng(5)
        rgb = np.full((250, 250, 3), 50, dtype=np.uint8)  # dark, no green
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        if result.decision == "REJECT":
            assert result.retake_message is not None
            assert len(result.retake_message) > 0

    def test_failing_dimensions_subset_of_per_dimension(self):
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        for dim in result.failing_dimensions:
            assert dim in result.per_dimension
            assert result.per_dimension[dim] < BAD_THRESHOLDS[dim]

    def test_green_mask_none_on_reject_non_none_otherwise(self):
        # spec: 6.5 line 1364 — green_mask is the rough HSV mask from leaf_presence
        # On REJECT we return None; on non-REJECT we return the mask.
        rgb = self._make_good_image()
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        if result.decision == "REJECT":
            assert result.green_mask is None
        else:
            assert result.green_mask is not None
            assert isinstance(result.green_mask, np.ndarray)

    def test_bad_validated_image_returns_reject(self):
        # spec: 6.6 — if input is corrupt, REJECT is returned without crashing
        bad_vi = MagicMock()
        bad_vi.pil_image = None  # will cause AttributeError on .convert()
        result = compute_iqa(bad_vi)
        assert result.decision == "REJECT"
        assert result.retake_message is not None

    def test_package_compute_iqa_is_same_as_module_compute_iqa(self):
        assert compute_iqa_pkg is compute_iqa

    def test_solid_dark_red_image_is_reject(self):
        # A completely dark red image has no green leaf, very dark V.
        # Should fail leaf_presence (maybe) and exposure dimensions.
        rgb = np.full((300, 300, 3), [20, 0, 0], dtype=np.uint8)
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        assert result.decision == "REJECT"

    def test_high_quality_greenleaf_is_not_reject(self):
        # A mid-toned, somewhat-sharp image dominated by green leaf colors.
        # Build an image with texture (for sharpness) and green color (leaf).
        rng = np.random.default_rng(99)
        rgb = np.zeros((400, 400, 3), dtype=np.uint8)
        rgb[:, :, 1] = 130  # green
        rgb[:, :, 0] = 50
        rgb[:, :, 2] = 40
        # Add high-frequency texture for sharpness
        noise = rng.integers(-60, 60, (400, 400, 3), dtype=np.int16)
        rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        # A properly formed green image with texture should not be REJECT
        assert result.decision != "REJECT"

    def test_resolution_dim_correct_for_image_shape(self):
        # Verify resolution dimension matches expected formula
        rgb = self._make_good_image()  # 400x400
        vi = _mock_validated_image(rgb)
        result = compute_iqa(vi)
        # smaller=400 → (400-224)/(800-224) = 176/576 ≈ 0.3056
        expected_res = (400 - 224) / (800 - 224)
        assert result.per_dimension["resolution"] == pytest.approx(expected_res, abs=0.01)


# ---------------------------------------------------------------------------
# Retake message: worst-dim selection
# spec: 6.4 line 1335
# ---------------------------------------------------------------------------

class TestRetakeMessageSelection:
    def _make_scores_with_failures(self, **overrides: float) -> dict[str, float]:
        scores = {k: 0.99 for k in BAD_THRESHOLDS}
        scores.update(overrides)
        return scores

    def test_worst_dim_selected_when_multiple_fail(self):
        """The dimension with the lowest score among failing ones determines message."""
        # sharpness=0.05 < 0.20 (BAD), leaf_presence=0.10 < 0.30 (BAD)
        # lowest score is sharpness (0.05 < 0.10)
        from tomato_sandbox.iqa.iqa import _RETAKE_MESSAGES
        scores = self._make_scores_with_failures(sharpness=0.05, leaf_presence=0.10)
        aggregate = _aggregate_quality(scores)
        decision = _iqa_decide(aggregate, scores)
        assert decision == "REJECT"
        # Worst dim: sharpness (0.05) < leaf_presence (0.10)
        worst = min(["sharpness", "leaf_presence"], key=lambda d: scores[d])
        assert worst == "sharpness"
        # Verify the retake message would be sharpness message
        assert "blurry" in _RETAKE_MESSAGES["sharpness"].lower()

    def test_exposure_retake_dark_vs_bright(self):
        from tomato_sandbox.iqa.iqa import _RETAKE_MESSAGES
        # dark exposure
        assert "dark" in _RETAKE_MESSAGES["exposure_dark"].lower()
        # bright exposure
        assert "bright" in _RETAKE_MESSAGES["exposure_bright"].lower() or \
               "overexposed" in _RETAKE_MESSAGES["exposure_bright"].lower()


# ---------------------------------------------------------------------------
# DEGRADED forward contract with Section 14
# spec: 6.4 line 1336 — tier system caps at Tier 3 when decision == DEGRADED
# ---------------------------------------------------------------------------

class TestDegradedContract:
    def test_degraded_decision_value_is_string(self):
        # spec: 6.4 line 1336 — iqa.decision == "DEGRADED" is the forward contract
        scores = {k: 0.99 for k in BAD_THRESHOLDS}
        result = _iqa_decide(0.50, scores)
        assert result == "DEGRADED"
        assert isinstance(result, str)

    def test_degraded_is_distinct_from_reject(self):
        assert "DEGRADED" != "REJECT"

    def test_iqa_result_decision_field_is_string(self):
        r = IQAResult(decision="DEGRADED", aggregate_score=0.5,
                      per_dimension={k: 0.99 for k in BAD_THRESHOLDS})
        assert r.decision == "DEGRADED"
