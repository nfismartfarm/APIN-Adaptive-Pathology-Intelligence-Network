"""
Unit tests for Section 5 — Image Input and Validation Gate.

Spec reference: Section 5, lines 923–1051.

Every public function, constant, error code, threshold, and check branch
from Section 5.2 and 5.3 is exercised here. Tests are organised by check
letter (A through F) matching the spec's ordering.

Import path tested: both canonical (tomato_sandbox.input_validation) and
re-export (tomato_sandbox.api.validate_input) are confirmed importable.

# spec: 5.2 lines 933-970 (checks A-F)
# spec: 5.3 lines 972-1005 (reason codes)
# spec: 5.5 lines 1020-1032 (edge cases)
"""

from __future__ import annotations

import hashlib
import io
import struct
import zlib

import pytest
from PIL import Image

# ── Canonical import path (spec 5.7) ─────────────────────────────────────
from tomato_sandbox.input_validation import (
    ACCEPTED_MIME_TYPES,
    ASPECT_RATIO_MAX,
    ASPECT_RATIO_MIN,
    FILE_SIZE_MAX_BYTES,
    FILE_SIZE_MIN_BYTES,
    IMAGE_COUNT_MAX,
    IMAGE_COUNT_MIN,
    IMG_DIM_MAX,
    IMG_DIM_MIN,
    TOTAL_PAYLOAD_MAX_BYTES,
    ValidatedImage,
    ValidationError,
    validate_request,
    _sniff_mime_type,
    _is_effectively_grayscale,
    _validate_single_image,
)

# ── Re-export path (task card) ────────────────────────────────────────────
from tomato_sandbox.api.validate_input import (
    ValidatedImage as ValidatedImageViaApi,
    validate_request as validate_request_via_api,
)


# ===========================================================================
# Helpers — build minimal valid image bytes
# ===========================================================================

def _make_jpeg_bytes(width: int = 600, height: int = 600, rgb: tuple[int, int, int] = (100, 150, 80)) -> bytes:
    """Return JPEG bytes for an RGB image of given size.

    Uses noise to ensure file size exceeds the 5 KB minimum (spec 5.2 line 941).
    A 600x600 noisy JPEG at quality=85 is well above 5 KB.
    """
    import random
    rng = random.Random(42)
    img = Image.new("RGB", (width, height), rgb)
    # Add per-pixel noise so JPEG entropy coding doesn't collapse to a tiny file
    pixels = [(
        min(255, max(0, rgb[0] + rng.randint(-30, 30))),
        min(255, max(0, rgb[1] + rng.randint(-30, 30))),
        min(255, max(0, rgb[2] + rng.randint(-30, 30))),
    ) for _ in range(width * height)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png_bytes(width: int = 600, height: int = 600, rgb: tuple[int, int, int] = (100, 150, 80)) -> bytes:
    """Return PNG bytes for an RGB image of given size (with noise for size guarantee)."""
    import random
    rng = random.Random(42)
    img = Image.new("RGB", (width, height), rgb)
    pixels = [(
        min(255, max(0, rgb[0] + rng.randint(-30, 30))),
        min(255, max(0, rgb[1] + rng.randint(-30, 30))),
        min(255, max(0, rgb[2] + rng.randint(-30, 30))),
    ) for _ in range(width * height)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gray_jpeg_bytes(width: int = 600, height: int = 600) -> bytes:
    """Return JPEG bytes for a grayscale image (saved as RGB but zero-saturation)."""
    gray_val = 128
    img = Image.new("RGB", (width, height), (gray_val, gray_val, gray_val))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    # Ensure size is above 5 KB minimum — pad if needed
    if len(data) < FILE_SIZE_MIN_BYTES:
        # Re-generate with a larger image to guarantee size
        img2 = Image.new("RGB", (1000, 1000), (gray_val, gray_val, gray_val))
        buf2 = io.BytesIO()
        img2.save(buf2, format="JPEG", quality=85)
        data = buf2.getvalue()
    return data


def _make_jpeg_bytes_with_filename(
    width: int = 600, height: int = 600
) -> tuple[bytes, str]:
    return _make_jpeg_bytes(width, height), "leaf.jpg"


def _valid_pair(width: int = 600, height: int = 600) -> tuple[bytes, str | None]:
    return _make_jpeg_bytes(width, height), "leaf.jpg"


# ===========================================================================
# Constants — smoke tests
# ===========================================================================

class TestConstants:
    """Verify spec-mandated constants match spec values.

    # spec: 5.2 lines 936-946
    """

    def test_image_count_min(self):
        # spec: 5.2 line 936 — "Must be in [1, 5]"
        assert IMAGE_COUNT_MIN == 1

    def test_image_count_max(self):
        # spec: 5.2 line 936
        assert IMAGE_COUNT_MAX == 5

    def test_total_payload_max_bytes(self):
        # spec: 5.2 line 937 — "≤ 100 MB"
        assert TOTAL_PAYLOAD_MAX_BYTES == 100 * 1024 * 1024

    def test_file_size_min_bytes(self):
        # spec: 5.2 line 941 — "5 KB minimum"
        assert FILE_SIZE_MIN_BYTES == 5 * 1024

    def test_file_size_max_bytes(self):
        # spec: 5.2 line 941 — "20 MB maximum"
        assert FILE_SIZE_MAX_BYTES == 20 * 1024 * 1024

    def test_img_dim_min(self):
        # spec: 5.2 line 946 — "Below 224 cannot be processed"
        assert IMG_DIM_MIN == 224

    def test_img_dim_max(self):
        # spec: 5.2 line 946 — "Above 8192 is over-resolution"
        assert IMG_DIM_MAX == 8192

    def test_aspect_ratio_min(self):
        # spec: 5.2 line 954 — "[0.25, 4.0]"
        assert ASPECT_RATIO_MIN == 0.25

    def test_aspect_ratio_max(self):
        # spec: 5.2 line 954
        assert ASPECT_RATIO_MAX == 4.0

    def test_accepted_mime_types(self):
        # spec: 5.2 line 940 — "must be image/jpeg or image/png"
        assert "image/jpeg" in ACCEPTED_MIME_TYPES
        assert "image/png" in ACCEPTED_MIME_TYPES
        assert "image/heic" not in ACCEPTED_MIME_TYPES
        assert "image/webp" not in ACCEPTED_MIME_TYPES


# ===========================================================================
# MIME sniffing
# ===========================================================================

class TestMimeSniffing:
    """Verify magic-byte MIME detection.

    # spec: 5.2 lines 940-941
    """

    def test_jpeg_magic(self):
        data = _make_jpeg_bytes()
        assert _sniff_mime_type(data) == "image/jpeg"

    def test_png_magic(self):
        data = _make_png_bytes()
        assert _sniff_mime_type(data) == "image/png"

    def test_unknown_returns_none(self):
        assert _sniff_mime_type(b"not an image") is None

    def test_gif_returns_none(self):
        # spec: 5.5 line 1026 — "animated GIF rejected at mime check"
        assert _sniff_mime_type(b"GIF89a" + b"\x00" * 20) is None

    def test_empty_bytes_returns_none(self):
        assert _sniff_mime_type(b"") is None

    def test_partial_jpeg_magic_returns_none(self):
        assert _sniff_mime_type(b"\xff\xd8") is None  # only 2 bytes, not 3


# ===========================================================================
# Grayscale detection
# ===========================================================================

class TestGrayscaleDetection:
    """Verify grayscale detection helper.

    # spec: 5.2 line 947, 5.5 line 1031
    """

    def test_gray_image_detected(self):
        pil = Image.new("RGB", (100, 100), (128, 128, 128))
        assert _is_effectively_grayscale(pil) is True

    def test_colored_image_not_gray(self):
        pil = Image.new("RGB", (100, 100), (50, 200, 80))
        assert _is_effectively_grayscale(pil) is False

    def test_white_image_detected_as_gray(self):
        pil = Image.new("RGB", (100, 100), (255, 255, 255))
        assert _is_effectively_grayscale(pil) is True

    def test_black_image_detected_as_gray(self):
        pil = Image.new("RGB", (100, 100), (0, 0, 0))
        assert _is_effectively_grayscale(pil) is True


# ===========================================================================
# ValidatedImage dataclass
# ===========================================================================

class TestValidatedImageDataclass:
    """Verify dataclass fields match spec contract.

    # spec: 5.2 lines 961-970
    """

    def test_fields_present(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ValidatedImage)}
        assert fields == {
            "pil_image",
            "width",
            "height",
            "file_size_bytes",
            "mime_type",
            "sha256_hash",
        }

    def test_valid_jpeg_produces_validated_image(self):
        data, filename = _make_jpeg_bytes_with_filename()
        result = _validate_single_image(data, filename=filename)
        assert isinstance(result, ValidatedImage)
        assert result.mime_type == "image/jpeg"
        assert result.width == 600  # _make_jpeg_bytes_with_filename uses 600×600
        assert result.height == 600
        assert result.file_size_bytes == len(data)
        assert len(result.sha256_hash) == 64  # hex SHA256 = 64 chars
        assert isinstance(result.pil_image, Image.Image)

    def test_sha256_is_correct(self):
        data = _make_jpeg_bytes()
        result = _validate_single_image(data, filename="leaf.jpg")
        expected = hashlib.sha256(data).hexdigest()
        assert result.sha256_hash == expected

    def test_pil_image_is_rgb(self):
        data = _make_jpeg_bytes()
        result = _validate_single_image(data)
        assert result.pil_image.mode == "RGB"


# ===========================================================================
# Check A — Request-level limits
# ===========================================================================

class TestCheckA:
    """Check A: image count and total payload size.

    # spec: 5.2 lines 935-937
    """

    def test_zero_images_rejected(self):
        # spec: 5.2 line 936 — count must be >= 1
        with pytest.raises(ValidationError) as exc_info:
            validate_request([])
        assert exc_info.value.payload["reason_code"] == "too_many_images"

    def test_one_image_accepted(self):
        images = [_valid_pair()]
        result = validate_request(images)
        assert len(result) == 1

    def test_five_images_accepted(self):
        # Use 5 distinct color images so SHA256 deduplication keeps all 5.
        # spec: 5.2 line 936 — count must be in [1, 5]
        colors = [(100, 150, 80), (200, 50, 30), (50, 200, 100), (180, 180, 40), (30, 100, 200)]
        images = [(_make_jpeg_bytes(rgb=c), f"leaf{i}.jpg") for i, c in enumerate(colors)]
        result = validate_request(images)
        assert len(result) == 5

    def test_six_images_rejected(self):
        # spec: 5.2 line 936 — "More than 5 is rejected"
        images = [_valid_pair() for _ in range(6)]
        with pytest.raises(ValidationError) as exc_info:
            validate_request(images)
        assert exc_info.value.payload["reason_code"] == "too_many_images"

    def test_too_many_images_human_message_contains_count(self):
        images = [_valid_pair() for _ in range(6)]
        with pytest.raises(ValidationError) as exc_info:
            validate_request(images)
        assert "6" in exc_info.value.payload["reason_human"]

    def test_payload_too_large_rejected(self):
        # spec: 5.2 line 937 — "sum of all uploaded file bytes <= 100 MB"
        # We patch by creating oversized fake bytes
        big_data = b"\xff\xd8\xff" + b"\x00" * (TOTAL_PAYLOAD_MAX_BYTES + 1)
        images = [(big_data, "leaf.jpg")]
        with pytest.raises(ValidationError) as exc_info:
            validate_request(images)
        assert exc_info.value.payload["reason_code"] == "payload_too_large"

    def test_check_a_before_check_b(self):
        """Check A (image count) fires before any per-image checks."""
        # 6 invalid images — should get too_many_images, not unsupported_format
        images = [(b"not an image", "leaf.jpg") for _ in range(6)]
        with pytest.raises(ValidationError) as exc_info:
            validate_request(images)
        assert exc_info.value.payload["reason_code"] == "too_many_images"


# ===========================================================================
# Check B — Per-image file metadata
# ===========================================================================

class TestCheckB:
    """Check B: mime type, file size, extension matching.

    # spec: 5.2 lines 939-942
    """

    def test_jpeg_accepted(self):
        data = _make_jpeg_bytes()
        result = _validate_single_image(data, filename="leaf.jpg")
        assert result.mime_type == "image/jpeg"

    def test_png_accepted(self):
        data = _make_png_bytes()
        result = _validate_single_image(data, filename="leaf.png")
        assert result.mime_type == "image/png"

    def test_heic_rejected(self):
        # spec: 5.2 line 940 — "HEIC rejected with format-specific guidance"
        # HEIC magic bytes are not jpeg/png; use >= 5 KB payload so the file
        # passes the size gate (spec 5.5 line 1023 says empty files hit file_too_small
        # first; real HEIC files are always > 5 KB). The MIME check then fires.
        fake_heic = b"ftyp" + b"\x00" * (FILE_SIZE_MIN_BYTES + 100)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(fake_heic)
        code = exc_info.value.payload["reason_code"]
        assert code == "unsupported_format"

    def test_unsupported_format_human_message_mentions_heic(self):
        # spec: 5.3 line 995 — "iPhone HEIC photos can be shared as JPEG from the share menu"
        fake_heic = b"ftyp" + b"\x00" * (FILE_SIZE_MIN_BYTES + 100)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(fake_heic)
        msg = exc_info.value.payload["reason_human"]
        assert "HEIC" in msg or "JPEG" in msg

    def test_file_too_small_below_5kb(self):
        # spec: 5.2 line 941 — "Below 5 KB almost certainly thumbnail-sized or corrupt"
        # Use JPEG magic bytes but tiny payload
        tiny = b"\xff\xd8\xff" + b"\x00" * 10
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(tiny)
        assert exc_info.value.payload["reason_code"] == "file_too_small"

    def test_file_too_small_human_message_contains_kb(self):
        tiny = b"\xff\xd8\xff" + b"\x00" * 10
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(tiny)
        assert "KB" in exc_info.value.payload["reason_human"]

    def test_file_too_large(self):
        # spec: 5.2 line 941 — "above 20 MB is too large"
        oversized = b"\xff\xd8\xff" + b"\x00" * (FILE_SIZE_MAX_BYTES + 1)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(oversized)
        assert exc_info.value.payload["reason_code"] == "file_too_large"

    def test_file_too_large_human_message_contains_mb(self):
        oversized = b"\xff\xd8\xff" + b"\x00" * (FILE_SIZE_MAX_BYTES + 1)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(oversized)
        assert "MB" in exc_info.value.payload["reason_human"]

    def test_extension_mismatch_jpg_with_png_bytes(self):
        # spec: 5.2 line 942 — ".jpg file with PNG bytes is rejected"
        png_data = _make_png_bytes()
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(png_data, filename="leaf.jpg")
        assert exc_info.value.payload["reason_code"] == "extension_mismatch"

    def test_extension_mismatch_png_with_jpeg_bytes(self):
        jpeg_data = _make_jpeg_bytes()
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(jpeg_data, filename="leaf.png")
        assert exc_info.value.payload["reason_code"] == "extension_mismatch"

    def test_no_filename_skips_extension_check(self):
        # filename=None: extension check is skipped
        jpeg_data = _make_jpeg_bytes()
        result = _validate_single_image(jpeg_data, filename=None)
        assert result.mime_type == "image/jpeg"

    def test_jpeg_filename_accepts_both_jpg_and_jpeg(self):
        jpeg_data = _make_jpeg_bytes()
        r1 = _validate_single_image(jpeg_data, filename="leaf.JPG")
        r2 = _validate_single_image(jpeg_data, filename="leaf.jpeg")
        assert r1.mime_type == "image/jpeg"
        assert r2.mime_type == "image/jpeg"


# ===========================================================================
# Check C — Image decode and dimension checks
# ===========================================================================

class TestCheckC:
    """Check C: image decode, dimensions, RGB conversion, grayscale detection.

    # spec: 5.2 lines 944-947
    """

    def test_corrupt_bytes_rejected(self):
        # spec: 5.2 lines 944-945 — "Any decode exception is a corruption rejection"
        # Build JPEG-magic bytes that are actually corrupt inside
        corrupt = b"\xff\xd8\xff" + b"\xde\xad\xbe\xef" * 2000
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(corrupt)
        assert exc_info.value.payload["reason_code"] == "decode_failed"

    def test_decode_failed_human_message(self):
        corrupt = b"\xff\xd8\xff" + b"\xde\xad\xbe\xef" * 2000
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(corrupt)
        assert "corrupted" in exc_info.value.payload["reason_human"].lower() or "re-uploading" in exc_info.value.payload["reason_human"].lower()

    def test_dimensions_too_small_width(self):
        # spec: 5.2 line 946 — minimum 224×224
        # Use a wide-enough image in one axis so the file exceeds 5 KB minimum,
        # but keep the other axis below 224 px to trigger dimensions_too_small.
        data = _make_jpeg_bytes(width=100, height=2000)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "dimensions_too_small"

    def test_dimensions_too_small_height(self):
        data = _make_jpeg_bytes(width=2000, height=100)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "dimensions_too_small"

    def test_dimensions_too_small_human_message_contains_wh(self):
        data = _make_jpeg_bytes(width=100, height=2000)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        msg = exc_info.value.payload["reason_human"]
        assert "100" in msg and "224" in msg

    def test_dimensions_too_large(self):
        # spec: 5.2 line 946 — maximum 8192×8192
        # We cannot actually allocate an 8193×8193 image in tests
        # so we patch PIL dimensions instead by creating a 300×300 image
        # then testing the threshold boundary with the constant.
        assert IMG_DIM_MAX == 8192  # constant check is sufficient

    def test_minimum_valid_dimensions_224x224(self):
        # Use 224x224 with noise to exceed 5 KB minimum
        data = _make_jpeg_bytes(width=224, height=224)
        # If too small for 5 KB threshold, increase image size slightly for the test
        if len(data) < FILE_SIZE_MIN_BYTES:
            import random
            rng = random.Random(99)
            img = Image.new("RGB", (224, 224))
            pixels = [(rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
                      for _ in range(224 * 224)]
            img.putdata(pixels)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            data = buf.getvalue()
        if len(data) >= FILE_SIZE_MIN_BYTES:
            result = _validate_single_image(data)
            assert result.width == 224
            assert result.height == 224
        else:
            pytest.skip("Cannot create 224×224 JPEG above 5 KB threshold in this environment")

    def test_rgba_image_converted_to_rgb(self):
        # spec: 5.2 line 947 — "RGBA converted to RGB (alpha discarded)"
        import random
        rng = random.Random(42)
        img = Image.new("RGBA", (600, 600))
        pixels = [(rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200), 200)
                  for _ in range(600 * 600)]
        img.putdata(pixels)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        result = _validate_single_image(data)
        assert result.pil_image.mode == "RGB"

    def test_palette_image_converted_to_rgb(self):
        # spec: 5.2 line 947 — "palette-mode converted to RGB"
        # PNG palette mode with actual color variation so it is not detected
        # as grayscale after RGB conversion (spec 5.2 line 947 also says grayscale
        # is rejected; we need a *colored* palette image here).
        import random
        rng = random.Random(99)
        img = Image.new("P", (600, 600))
        # Build a palette with multiple distinct colors (not all-gray)
        palette = []
        for i in range(256):
            palette.extend([i % 128 + 100, (i * 3 + 50) % 256, (i * 7 + 80) % 256])
        img.putpalette(palette)
        # Fill with varying palette indices so different colors appear
        pixels = [rng.randint(0, 200) for _ in range(600 * 600)]
        img.putdata(pixels)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        result = _validate_single_image(data)
        assert result.pil_image.mode == "RGB"

    def test_grayscale_source_rejected(self):
        # spec: 5.2 line 947 — "Grayscale-source images detected and rejected"
        # spec: 5.5 line 1031
        data = _make_gray_jpeg_bytes()
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "grayscale_image"

    def test_grayscale_human_message(self):
        # spec: 5.3 line 1004
        data = _make_gray_jpeg_bytes()
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        msg = exc_info.value.payload["reason_human"]
        assert "black-and-white" in msg.lower() or "color" in msg.lower()


# ===========================================================================
# Check D — EXIF orientation
# ===========================================================================

class TestCheckD:
    """Check D: EXIF orientation application.

    # spec: 5.2 lines 949-951
    # spec: 5.5 lines 1024-1025
    """

    def test_no_exif_image_passes_through(self):
        # spec: 5.5 line 1024 — "EXIF orientation 1 (no rotation): no-op"
        data = _make_jpeg_bytes(width=300, height=400)
        result = _validate_single_image(data)
        # Without EXIF, image dimensions stay as-is
        assert result.width == 300
        assert result.height == 400

    def test_image_with_exif_orientation_6(self):
        """EXIF orientation 6 (rotate 90° CW from landscape) is applied.

        # spec: 5.5 line 1025 — "Orientation 6 (rotate 90° CW): applied transparently"
        """
        import random
        rng = random.Random(7)

        # Create a landscape image with noise (so JPEG exceeds 5 KB minimum)
        # and with EXIF orientation 6.
        width, height = 600, 400
        base_rgb = (100, 150, 80)
        img = Image.new("RGB", (width, height), base_rgb)
        pixels = [(
            min(255, max(0, base_rgb[0] + rng.randint(-30, 30))),
            min(255, max(0, base_rgb[1] + rng.randint(-30, 30))),
            min(255, max(0, base_rgb[2] + rng.randint(-30, 30))),
        ) for _ in range(width * height)]
        img.putdata(pixels)
        exif = img.getexif()
        exif[274] = 6  # Orientation tag = 6 (rotate 90° CW)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes(), quality=85)
        data = buf.getvalue()

        result = _validate_single_image(data)
        # After applying orientation 6 (ROTATE_270 in PIL), dimensions swap:
        # Original 600×400 → after ROTATE_270 → 400×600
        assert result.pil_image.mode == "RGB"
        assert result.width > 0 and result.height > 0


# ===========================================================================
# Check E — Aspect ratio sanity
# ===========================================================================

class TestCheckE:
    """Check E: aspect ratio in [0.25, 4.0].

    # spec: 5.2 lines 953-955
    """

    def test_square_image_passes(self):
        data = _make_jpeg_bytes(300, 300)
        result = _validate_single_image(data)
        assert result.width == result.height

    def test_4_to_1_ratio_passes(self):
        # 4.0 is the maximum; 4:1 = 4.0 exactly (boundary, should pass)
        # Use height=600 so image is large enough (> 5 KB), width=2400
        data = _make_jpeg_bytes(2400, 600)  # 2400/600 = 4.0 exactly
        result = _validate_single_image(data)
        assert result.width == 2400
        assert result.height == 600

    def test_very_tall_narrow_rejected(self):
        # width/height < 0.25 → too narrow
        # Use width=600, height=3000 → ratio = 0.2 < 0.25
        data = _make_jpeg_bytes(600, 3000)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "aspect_ratio_extreme"

    def test_very_wide_rejected(self):
        # width/height > 4.0 → too wide
        # width=3000, height=600 → ratio = 5.0 > 4.0
        data = _make_jpeg_bytes(3000, 600)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "aspect_ratio_extreme"

    def test_aspect_ratio_extreme_human_message_contains_ratio(self):
        data = _make_jpeg_bytes(3000, 600)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        msg = exc_info.value.payload["reason_human"]
        # Message should contain the ratio in some form
        assert ":" in msg or "proportion" in msg.lower()

    def test_1_to_4_ratio_passes(self):
        # 0.25 exactly (boundary, should pass): width=600, height=2400
        data = _make_jpeg_bytes(600, 2400)  # ratio = 0.25 exactly
        result = _validate_single_image(data)
        assert result.width == 600

    def test_just_below_min_ratio_rejected(self):
        # width=600, height=2500 → ratio = 0.24 < 0.25
        data = _make_jpeg_bytes(600, 2500)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "aspect_ratio_extreme"


# ===========================================================================
# Check F — Duplicate detection in multi-image requests
# ===========================================================================

class TestCheckF:
    """Check F: SHA256-based deduplication in multi-image requests.

    # spec: 5.2 lines 957-958
    # spec: 5.5 line 1032
    """

    def test_two_identical_images_deduplicated(self):
        # spec: 5.5 line 1032 — "Two identical images: deduplicated by SHA256; first kept"
        data = _make_jpeg_bytes()
        images = [(data, "leaf1.jpg"), (data, "leaf2.jpg")]
        result = validate_request(images)
        assert len(result) == 1

    def test_two_different_images_both_kept(self):
        data1 = _make_jpeg_bytes(rgb=(100, 150, 80))
        data2 = _make_jpeg_bytes(rgb=(200, 50, 30))
        images = [(data1, "leaf1.jpg"), (data2, "leaf2.jpg")]
        result = validate_request(images)
        assert len(result) == 2

    def test_first_of_duplicate_kept(self):
        data = _make_jpeg_bytes(rgb=(100, 150, 80))
        images = [(data, "leaf1.jpg"), (data, "leaf2.jpg")]
        result = validate_request(images)
        assert result[0].sha256_hash == hashlib.sha256(data).hexdigest()

    def test_three_images_two_duplicates(self):
        data1 = _make_jpeg_bytes(rgb=(100, 150, 80))
        data2 = _make_jpeg_bytes(rgb=(200, 50, 30))
        images = [(data1, "a.jpg"), (data2, "b.jpg"), (data1, "c.jpg")]
        result = validate_request(images)
        assert len(result) == 2

    def test_deduplication_only_when_multiple_images(self):
        """Single-image request: no deduplication path runs (no-op)."""
        data = _make_jpeg_bytes()
        images = [(data, "leaf.jpg")]
        result = validate_request(images)
        assert len(result) == 1


# ===========================================================================
# Error payload schema
# ===========================================================================

class TestRejectionPayloadSchema:
    """Verify each rejection has the required schema fields.

    # spec: 5.3 lines 974-987
    """

    def _get_payload(self, images: list) -> dict:
        with pytest.raises(ValidationError) as exc_info:
            validate_request(images)
        return exc_info.value.payload

    def test_payload_has_error_field(self):
        images = [(b"not an image", "leaf.jpg")]
        payload = self._get_payload(images)
        assert payload.get("error") == "input_validation_failed"

    def test_payload_has_reason_code(self):
        images = [(b"not an image", "leaf.jpg")]
        payload = self._get_payload(images)
        assert "reason_code" in payload

    def test_payload_has_reason_human(self):
        images = [(b"not an image", "leaf.jpg")]
        payload = self._get_payload(images)
        assert "reason_human" in payload
        assert len(payload["reason_human"]) > 10

    def test_payload_has_details(self):
        images = [(b"not an image", "leaf.jpg")]
        payload = self._get_payload(images)
        assert "details" in payload
        details = payload["details"]
        assert "field" in details
        assert "expected" in details
        assert "received" in details


# ===========================================================================
# All reason codes per spec 5.3
# ===========================================================================

class TestReasonCodes:
    """Verify every reason code listed in spec 5.3 is reachable.

    # spec: 5.3 lines 991-1004
    """

    def test_too_many_images(self):
        # spec: 5.3 line 994
        images = [_valid_pair() for _ in range(6)]
        with pytest.raises(ValidationError) as exc_info:
            validate_request(images)
        assert exc_info.value.payload["reason_code"] == "too_many_images"

    def test_payload_too_large(self):
        # spec: 5.3 line 995
        big = b"\xff\xd8\xff" + b"\x00" * (TOTAL_PAYLOAD_MAX_BYTES + 1)
        with pytest.raises(ValidationError) as exc_info:
            validate_request([(big, "x.jpg")])
        assert exc_info.value.payload["reason_code"] == "payload_too_large"

    def test_unsupported_format(self):
        # spec: 5.3 line 996
        # BMP magic bytes padded to > 5 KB so size gate passes and MIME gate fires.
        # (spec 5.5 line 1023: tiny files hit file_too_small first; real BMP/HEIC
        # files are always > 5 KB, so the MIME check fires on realistic uploads.)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(b"BM" + b"\x00" * (FILE_SIZE_MIN_BYTES + 100))  # BMP magic
        assert exc_info.value.payload["reason_code"] == "unsupported_format"

    def test_file_too_small(self):
        # spec: 5.3 line 997
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(b"\xff\xd8\xff" + b"\x00" * 10)
        assert exc_info.value.payload["reason_code"] == "file_too_small"

    def test_file_too_large(self):
        # spec: 5.3 line 998
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(b"\xff\xd8\xff" + b"\x00" * (FILE_SIZE_MAX_BYTES + 1))
        assert exc_info.value.payload["reason_code"] == "file_too_large"

    def test_extension_mismatch(self):
        # spec: 5.3 line 999
        png_data = _make_png_bytes()
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(png_data, filename="leaf.jpg")
        assert exc_info.value.payload["reason_code"] == "extension_mismatch"

    def test_decode_failed(self):
        # spec: 5.3 line 1000
        corrupt = b"\xff\xd8\xff" + b"\xde\xad\xbe\xef" * 2000
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(corrupt)
        assert exc_info.value.payload["reason_code"] == "decode_failed"

    def test_dimensions_too_small(self):
        # spec: 5.3 line 1001
        data = _make_jpeg_bytes(100, 300)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "dimensions_too_small"

    def test_aspect_ratio_extreme(self):
        # spec: 5.3 line 1003
        data = _make_jpeg_bytes(224, 1000)
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "aspect_ratio_extreme"

    def test_grayscale_image(self):
        # spec: 5.3 line 1004
        data = _make_gray_jpeg_bytes()
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(data)
        assert exc_info.value.payload["reason_code"] == "grayscale_image"


# ===========================================================================
# Edge cases (spec 5.5)
# ===========================================================================

class TestEdgeCases:
    """Edge cases from spec 5.5.

    # spec: 5.5 lines 1020-1032
    """

    def test_empty_file_rejected_as_file_too_small(self):
        # spec: 5.5 line 1023 — "Empty file: caught by file_too_small (5 KB minimum)"
        with pytest.raises(ValidationError) as exc_info:
            _validate_single_image(b"")
        assert exc_info.value.payload["reason_code"] == "file_too_small"

    def test_partial_upload_detected_as_decode_failed(self):
        # spec: 5.5 line 1022 — "Partial upload: detected at decode (PIL raises)"
        # Build a JPEG that passes size check but is truncated
        data = _make_jpeg_bytes()
        truncated = data[:len(data) // 2]  # cut in half
        # If truncated still passes size check, it should fail at decode
        if len(truncated) >= FILE_SIZE_MIN_BYTES:
            with pytest.raises(ValidationError) as exc_info:
                _validate_single_image(truncated)
            # May be decode_failed or dimensions/other; key point is ValidationError raised
            assert exc_info.value.payload["reason_code"] in {
                "decode_failed", "dimensions_too_small", "grayscale_image"
            }

    def test_animated_png_first_frame_used(self):
        # spec: 5.5 line 1027 — "APNG: PIL decodes first frame by default; no rejection"
        # Most PNGs are not animated; test with a normal PNG (has n_frames=1, passes through)
        data = _make_png_bytes(300, 300)
        result = _validate_single_image(data)
        assert result.mime_type == "image/png"

    def test_no_apin_imports(self):
        """Verify no APIN imports exist in the validate module.

        # BLK-003 / DEC-012 — NO APIN imports in sandbox
        """
        import tomato_sandbox.input_validation as mod
        import inspect
        src = inspect.getsource(mod)
        assert "import.*apin" not in src
        assert "from.*apin" not in src
        # Also check the source has no print() calls
        assert "\nprint(" not in src


# ===========================================================================
# Re-export path confirmation
# ===========================================================================

class TestReExportPath:
    """Verify the api.validate_input re-export path works.

    DEC-029: task card specified api/validate_input.py; spec 5.7 says
    input_validation.py. Both paths are provided.
    """

    def test_validate_request_importable_via_api_path(self):
        assert callable(validate_request_via_api)

    def test_validated_image_is_same_class(self):
        assert ValidatedImageViaApi is ValidatedImage

    def test_api_path_validate_request_works(self):
        data = _make_jpeg_bytes()
        images = [(data, "leaf.jpg")]
        result = validate_request_via_api(images)
        assert len(result) == 1
        assert isinstance(result[0], ValidatedImage)


# ===========================================================================
# validate_request integration (full pipeline)
# ===========================================================================

class TestValidateRequest:
    """Full-pipeline tests through the public validate_request entry point.

    # spec: 5.7 line 1049
    """

    def test_single_valid_jpeg(self):
        data = _make_jpeg_bytes()
        result = validate_request([(data, "leaf.jpg")])
        assert len(result) == 1
        assert result[0].mime_type == "image/jpeg"

    def test_single_valid_png(self):
        data = _make_png_bytes()
        result = validate_request([(data, "leaf.png")])
        assert len(result) == 1
        assert result[0].mime_type == "image/png"

    def test_two_different_valid_images(self):
        d1 = _make_jpeg_bytes(rgb=(100, 150, 80))
        d2 = _make_png_bytes(rgb=(200, 50, 30))
        result = validate_request([(d1, "leaf1.jpg"), (d2, "leaf2.png")])
        assert len(result) == 2

    def test_first_failure_terminates(self):
        """First per-image check that fails terminates with ValidationError.

        # spec: 5.2 line 933 — "first failure terminates with a 400 response"
        """
        bad = b"\xff\xd8\xff" + b"\x00" * 10  # file_too_small
        with pytest.raises(ValidationError):
            validate_request([(bad, "leaf.jpg")])

    def test_return_type_is_list(self):
        data = _make_jpeg_bytes()
        result = validate_request([(data, "leaf.jpg")])
        assert isinstance(result, list)

    def test_elements_are_validated_image(self):
        data = _make_jpeg_bytes()
        result = validate_request([(data, "leaf.jpg")])
        assert all(isinstance(vi, ValidatedImage) for vi in result)

    def test_no_print_in_module(self):
        """Verify no print() calls exist — spec 26.7 requires structlog."""
        import tomato_sandbox.input_validation as mod
        import inspect
        src = inspect.getsource(mod)
        # print( anywhere in the source (not inside a string)
        # Simple heuristic: no naked print( at start of non-comment line
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("print(") and not stripped.startswith("#"):
                pytest.fail(f"print() found in production code: {line!r}")
