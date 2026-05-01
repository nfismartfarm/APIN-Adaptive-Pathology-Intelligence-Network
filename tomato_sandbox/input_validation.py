"""
Input validation gate for the Tomato 3-Signal sandbox pipeline.

Spec section: 5 (Image Input and Validation Gate), lines 923–1051.

This module is the first piece of code that touches an uploaded image.
It rejects inputs that cannot possibly be processed before any expensive
GPU, PSV, or IQA work runs. It is fast, cheap, and defensive.

Contract (spec 5.7, line 1049):
  - ``ValidatedImage`` dataclass is the output structure.
  - ``validate_request(images_bytes)`` is the entry-point function.
    The server calls this before invoking the tomato pipeline.

This module has NO GPU dependency and NO ML model dependency.
All validation is pure Python / Pillow / hashlib.

# spec: 5.7 lines 1047-1050
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image, ExifTags, UnidentifiedImageError

from tomato_sandbox.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# spec: 26.7 — "Use structlog for structured logging; never print()"
# ---------------------------------------------------------------------------

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — sourced verbatim from spec Section 5.2
# ---------------------------------------------------------------------------

# Check A — Request-level limits
# spec: 5.2 lines 936-937
IMAGE_COUNT_MIN: int = 1
IMAGE_COUNT_MAX: int = 5
TOTAL_PAYLOAD_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MB

# Check B — Per-image file metadata
# spec: 5.2 lines 940-942
ACCEPTED_MIME_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png"})
FILE_SIZE_MIN_BYTES: int = 5 * 1024          # 5 KB
FILE_SIZE_MAX_BYTES: int = 20 * 1024 * 1024  # 20 MB

# MIME-type to accepted extension mapping (for extension_mismatch check)
# spec: 5.2 line 942 — "extension must agree with sniffed mime type"
_MIME_TO_EXTENSIONS: dict[str, frozenset[str]] = {
    "image/jpeg": frozenset({".jpg", ".jpeg"}),
    "image/png":  frozenset({".png"}),
}

# Check C — Image decode
# spec: 5.2 lines 946-947
IMG_DIM_MIN: int = 224   # spec: "Below 224 cannot be processed at v3's expected input size"
IMG_DIM_MAX: int = 8192  # spec: "Above 8192 is over-resolution"

# Check E — Aspect ratio sanity
# spec: 5.2 lines 954-955
ASPECT_RATIO_MIN: float = 0.25  # width/height
ASPECT_RATIO_MAX: float = 4.0   # width/height

# Magic-byte signatures for MIME sniffing
# spec: 5.2 line 940-941 — accepted: jpeg, png; others rejected
_JPEG_MAGIC: bytes = b"\xff\xd8\xff"
_PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# ValidatedImage dataclass
# spec: 5.2 lines 961-970
# ---------------------------------------------------------------------------

@dataclass
class ValidatedImage:
    """Output of the validation gate for a single image.

    All fields are set by ``validate_request``; downstream pipeline trusts
    these values and does not re-validate.

    # spec: 5.2 lines 961-970
    """

    pil_image: Image.Image   # RGB, EXIF-applied, ready for downstream
    width: int
    height: int
    file_size_bytes: int
    mime_type: str
    sha256_hash: str          # hex string; used by the response cache (Sec 26.1)


# ---------------------------------------------------------------------------
# Rejection response schema
# spec: 5.3 lines 972-1005
# ---------------------------------------------------------------------------

def _rejection(
    reason_code: str,
    reason_human: str,
    field: str,
    expected: Any,
    received: Any,
) -> dict[str, Any]:
    """Build a standardised 400 rejection dict.

    Callers raise HTTPException(status_code=400, detail=<returned dict>)
    or return it as a JSONResponse. The server layer decides the HTTP machinery;
    this function only builds the payload.

    # spec: 5.3 lines 974-987
    """
    return {
        "error": "input_validation_failed",
        "reason_code": reason_code,
        "reason_human": reason_human,
        "details": {
            "field": field,
            "expected": str(expected),
            "received": str(received),
        },
    }


class ValidationError(ValueError):
    """Raised by ``validate_request`` on the first failed check.

    Carries the rejection payload dict so the server can return it as
    a 400 response body without building it a second time.

    # spec: 5.3 lines 974-987 — each rejection terminates with a 400 response
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(str(payload))


# ---------------------------------------------------------------------------
# MIME sniffing
# spec: 5.2 lines 940-941
# ---------------------------------------------------------------------------

def _sniff_mime_type(data: bytes) -> str | None:
    """Return MIME type inferred from magic bytes, or None if unknown.

    Only checks jpeg and png because those are the only two accepted types.

    # spec: 5.2 lines 940-942
    """
    if data[:3] == _JPEG_MAGIC:
        return "image/jpeg"
    if data[:8] == _PNG_MAGIC:
        return "image/png"
    return None


# ---------------------------------------------------------------------------
# Grayscale detection
# spec: 5.2 line 947, 5.5 line 1031
# ---------------------------------------------------------------------------

def _is_effectively_grayscale(pil_img: Image.Image) -> bool:
    """Return True if the RGB image has zero saturation (all channels equal).

    Uses a fast histogram-based check: collapse to HSV and look at S channel
    mode/max. If the HSV saturation channel max is 0 across the whole image,
    all pixels are R==G==B → effectively grayscale.

    Spec requirement: "Grayscale-source images are detected (zero saturation
    in all pixels of the converted RGB) and rejected with a specific message
    rather than silently treated as colorless leaves."

    # spec: 5.2 line 947
    # spec: 5.5 line 1031
    """
    import colorsys

    # PIL thumbnail for speed (we don't need full resolution for this check)
    small = pil_img.copy()
    small.thumbnail((64, 64))
    rgb_data = list(small.getdata())  # list of (R, G, B) tuples

    for r, g, b in rgb_data:
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        _h, s, _v = colorsys.rgb_to_hsv(rf, gf, bf)
        if s > 0.01:  # tolerance for JPEG compression artefacts in near-gray images
            return False
    return True


# ---------------------------------------------------------------------------
# EXIF orientation application
# spec: 5.2 lines 949-951, 5.5 lines 1024-1025
# ---------------------------------------------------------------------------

def _apply_exif_orientation(pil_img: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag and strip it from the result.

    Many smartphone cameras store the image in landscape and use EXIF tag 274
    (Orientation) to indicate rotation. Without applying it, the system sees
    images sideways.

    # spec: 5.2 lines 949-951 — "Apply EXIF orientation tag if present.
    #   After applying orientation, the image is in its visually-correct
    #   orientation. EXIF tag is then stripped."
    # spec: 5.5 lines 1024-1025 — "EXIF orientation 1 (no rotation): no-op.
    #   Orientation 6 (rotate 90° CW): applied transparently."
    """
    try:
        exif_data = pil_img._getexif()  # type: ignore[attr-defined]
        if exif_data is None:
            return pil_img
    except AttributeError:
        # PNG and other formats may not have _getexif
        return pil_img

    # Find the orientation tag key
    orientation_key: int | None = None
    for tag, name in ExifTags.TAGS.items():
        if name == "Orientation":
            orientation_key = tag
            break

    if orientation_key is None or orientation_key not in exif_data:
        return pil_img

    orientation = exif_data[orientation_key]

    # Apply rotation/flip per EXIF orientation value
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Transpose
    _ORIENTATION_TO_METHOD: dict[int, Image.Transpose] = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,   # 90° CW phone → rotate 270° in PIL
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }

    method = _ORIENTATION_TO_METHOD.get(orientation)
    if method is not None:
        pil_img = pil_img.transpose(method)

    # Strip EXIF so downstream doesn't see stale orientation data
    # (orientation is now baked in)
    data = list(pil_img.getdata())
    clean = Image.new(pil_img.mode, pil_img.size)
    clean.putdata(data)
    return clean


# ---------------------------------------------------------------------------
# Single-image validation
# ---------------------------------------------------------------------------

def _validate_single_image(
    data: bytes,
    filename: str | None = None,
) -> ValidatedImage:
    """Validate one image's raw bytes and return a ``ValidatedImage``.

    Runs checks B through E in order. Raises ``ValidationError`` on the
    first failure (spec 5.2: "The first failure terminates with a 400 response").

    Args:
        data: Raw file bytes.
        filename: Original filename from the upload (used for extension check).
                  May be None if not available (extension check is skipped).

    Returns:
        ``ValidatedImage`` with pil_image in RGB, EXIF applied.

    Raises:
        ``ValidationError``: on the first check that fails.

    # spec: 5.2 lines 933-970
    """
    t0 = time.perf_counter()

    file_size = len(data)

    # ── Check B1 — file size (lower bound first) ─────────────────────────
    # spec: 5.5 line 1023 — "Empty file: zero-byte upload. Caught by file_too_small
    #   (5 KB minimum)."  This edge-case note is the authoritative behavioral
    #   contract for zero/tiny bytes: size must be gated before MIME sniffing so
    #   that empty files yield file_too_small, not unsupported_format.
    # spec: 5.2 line 941 — "[5 KB, 20 MB]"
    if file_size < FILE_SIZE_MIN_BYTES:
        raise ValidationError(
            _rejection(
                reason_code="file_too_small",
                # spec: 5.3 line 997
                reason_human=f"Image file is unusually small ({file_size // 1024} KB). Re-take the photo at higher quality.",
                field="file_size_bytes",
                expected=f">= {FILE_SIZE_MIN_BYTES} bytes",
                received=f"{file_size} bytes",
            )
        )
    if file_size > FILE_SIZE_MAX_BYTES:
        raise ValidationError(
            _rejection(
                reason_code="file_too_large",
                # spec: 5.3 line 998
                reason_human=f"Image file is too large ({file_size // (1024 * 1024)} MB). Limit per image: 20 MB.",
                field="file_size_bytes",
                expected=f"<= {FILE_SIZE_MAX_BYTES} bytes",
                received=f"{file_size} bytes",
            )
        )

    # ── Check B2 — MIME type (magic bytes) ────────────────────────────────
    # spec: 5.2 line 940 — "must be image/jpeg or image/png"
    # Note: size is gated first (see B1 above) so that empty/tiny files yield
    # file_too_small rather than unsupported_format (spec 5.5 line 1023).
    sniffed_mime = _sniff_mime_type(data)
    if sniffed_mime is None:
        # spec: 5.3 line 995 — "File type {mime_type} is not supported."
        raise ValidationError(
            _rejection(
                reason_code="unsupported_format",
                reason_human=(
                    "File type is not supported. Use JPEG or PNG. "
                    "(iPhone HEIC photos can be shared as JPEG from the share menu.)"
                ),
                field="mime_type",
                expected="image/jpeg or image/png",
                received="unknown",
            )
        )

    # ── Check B3 — extension matches mime ─────────────────────────────────
    # spec: 5.2 line 942 — "extension must agree with sniffed mime type"
    if filename is not None:
        import os
        _root, ext = os.path.splitext(filename.lower())
        accepted_exts = _MIME_TO_EXTENSIONS.get(sniffed_mime, frozenset())
        if ext and ext not in accepted_exts:
            raise ValidationError(
                _rejection(
                    reason_code="extension_mismatch",
                    # spec: 5.3 line 999
                    reason_human="File extension and content do not match. Re-save the file using its original format.",
                    field="extension_matches_mime",
                    expected=f"extension in {sorted(accepted_exts)} for {sniffed_mime}",
                    received=f"extension={ext!r}, sniffed={sniffed_mime}",
                )
            )

    # ── Check C1 — image decode ───────────────────────────────────────────
    # spec: 5.2 lines 944-945 — "Decode the bytes using PIL/Pillow. Any decode
    #   exception is a corruption rejection."
    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img.load()  # force full decode (catches partial uploads)
    except (UnidentifiedImageError, Exception) as exc:
        raise ValidationError(
            _rejection(
                reason_code="decode_failed",
                # spec: 5.3 line 1000
                reason_human="Image file could not be opened. It may be corrupted or partially uploaded. Try re-uploading.",
                field="decode",
                expected="valid image file",
                received=f"decode error: {type(exc).__name__}",
            )
        ) from exc

    # Log APNG detection (animated PNG — first frame only)
    # spec: 5.5 line 1027 — "passes mime check; PIL decodes first frame by default;
    #   remaining frames silently discarded with a debug log entry"
    if hasattr(pil_img, "n_frames") and pil_img.n_frames > 1:
        _logger.debug(
            "apng_first_frame_only",
            mime_type=sniffed_mime,
            n_frames=pil_img.n_frames,
            note="Only first frame used; remaining frames discarded (spec 5.5)",
        )

    # ── Check C2 — Convert to RGB ─────────────────────────────────────────
    # spec: 5.2 line 947 — "If not RGB, converted to RGB in-memory"
    # Also handles: RGBA (alpha discarded), palette-mode (P), grayscale (L),
    # CMYK, HDR / 16-bit (spec 5.5 line 1030: PIL converts to 8-bit)
    pil_img = pil_img.convert("RGB")

    # ── Check C3 — Grayscale source detection ─────────────────────────────
    # spec: 5.2 line 947 — "Grayscale-source images are detected (zero saturation
    #   in all pixels of the converted RGB) and rejected"
    # spec: 5.5 line 1031
    if _is_effectively_grayscale(pil_img):
        raise ValidationError(
            _rejection(
                reason_code="grayscale_image",
                # spec: 5.3 line 1004
                reason_human="Image appears to be black-and-white. Color is needed for disease detection. Re-take in color mode.",
                field="saturation",
                expected="color image (HSV saturation > 0)",
                received="zero-saturation image (grayscale source converted to RGB)",
            )
        )

    # ── Check C4 — Dimensions ────────────────────────────────────────────
    # spec: 5.2 line 946 — "width and height both in [224, 8192]"
    width, height = pil_img.size
    if width < IMG_DIM_MIN or height < IMG_DIM_MIN:
        raise ValidationError(
            _rejection(
                reason_code="dimensions_too_small",
                # spec: 5.3 line 1001
                reason_human=f"Image is too small ({width}×{height}). Minimum: 224×224 pixels.",
                field="dimensions",
                expected=f">= {IMG_DIM_MIN}x{IMG_DIM_MIN}",
                received=f"{width}x{height}",
            )
        )
    if width > IMG_DIM_MAX or height > IMG_DIM_MAX:
        raise ValidationError(
            _rejection(
                reason_code="dimensions_too_large",
                # spec: 5.3 line 1002
                reason_human=(
                    f"Image is too large ({width}×{height}). Maximum: 8192×8192 pixels. "
                    "Most phone cameras do not exceed this; check if you have a special-mode photo."
                ),
                field="dimensions",
                expected=f"<= {IMG_DIM_MAX}x{IMG_DIM_MAX}",
                received=f"{width}x{height}",
            )
        )

    # ── Check D — EXIF orientation ────────────────────────────────────────
    # spec: 5.2 lines 949-951
    pil_img = _apply_exif_orientation(pil_img)
    # Re-read size after orientation correction (rotate 90° swaps w/h)
    width, height = pil_img.size

    # ── Check E — Aspect ratio ────────────────────────────────────────────
    # spec: 5.2 lines 953-955 — ratio in [0.25, 4.0]
    aspect = width / height
    if aspect < ASPECT_RATIO_MIN or aspect > ASPECT_RATIO_MAX:
        raise ValidationError(
            _rejection(
                reason_code="aspect_ratio_extreme",
                # spec: 5.3 line 1003
                reason_human=(
                    f"Image proportions ({aspect:.2f}:1) are unusual for a leaf photo. "
                    "Re-frame with the leaf taking up most of the photo."
                ),
                field="aspect_ratio",
                expected=f"in [{ASPECT_RATIO_MIN}, {ASPECT_RATIO_MAX}]",
                received=f"{aspect:.4f}",
            )
        )

    # ── SHA256 hash (used by response cache, spec 26.1) ───────────────────
    sha256_hash = hashlib.sha256(data).hexdigest()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _logger.debug(
        "single_image_validated",
        mime_type=sniffed_mime,
        width=width,
        height=height,
        file_size_bytes=file_size,
        sha256=sha256_hash[:12] + "...",  # partial hash; full hash not logged
        elapsed_ms=round(elapsed_ms, 2),
    )

    return ValidatedImage(
        pil_image=pil_img,
        width=width,
        height=height,
        file_size_bytes=file_size,
        mime_type=sniffed_mime,
        sha256_hash=sha256_hash,
    )


# ---------------------------------------------------------------------------
# Request-level entry point
# spec: 5.7 line 1049 — validate_request is the entry point
# ---------------------------------------------------------------------------

def validate_request(
    images_bytes: list[tuple[bytes, str | None]],
) -> list[ValidatedImage]:
    """Validate all images in a single HTTP request.

    Runs checks A through F in the order mandated by spec Section 5.2.
    Check A runs first (request-level limits). Checks B-E run per image.
    Check F (duplicate deduplication) runs after all images pass B-E.

    Args:
        images_bytes: List of (raw_bytes, filename_or_None) pairs.
                      Each pair corresponds to one uploaded file.
                      ``filename`` is used for the extension-mismatch check;
                      pass None if not available.

    Returns:
        List of ``ValidatedImage`` objects, one per unique image that passed
        all checks. Duplicate images (same SHA256) are silently dropped after
        the first occurrence (spec 5.2 Check F).

    Raises:
        ``ValidationError``: on the first request-level or per-image check
        that fails. The caller (server) catches this and returns HTTP 400
        with ``error.payload`` as the response body.

    # spec: 5.2 lines 933-970 (full check sequence)
    # spec: 5.7 line 1049 — "validate_request(request) -> List[ValidatedImage]"
    """
    t0_request = time.perf_counter()

    # ── Check A1 — image count ─────────────────────────────────────────────
    # spec: 5.2 line 936 — "Must be in [1, 5]"
    n = len(images_bytes)
    if n < IMAGE_COUNT_MIN or n > IMAGE_COUNT_MAX:
        raise ValidationError(
            _rejection(
                reason_code="too_many_images",
                # spec: 5.3 line 994
                reason_human=f"You uploaded {n} images. Up to 5 are allowed in a single request.",
                field="image_count",
                expected=f"in [{IMAGE_COUNT_MIN}, {IMAGE_COUNT_MAX}]",
                received=str(n),
            )
        )

    # ── Check A2 — total payload size ─────────────────────────────────────
    # spec: 5.2 line 937 — "<= 100 MB"
    total_bytes = sum(len(b) for b, _fn in images_bytes)
    if total_bytes > TOTAL_PAYLOAD_MAX_BYTES:
        raise ValidationError(
            _rejection(
                reason_code="payload_too_large",
                # spec: 5.3 line 995
                reason_human="Total upload size is too large. Limit: 100 MB across all images.",
                field="total_payload_size",
                expected=f"<= {TOTAL_PAYLOAD_MAX_BYTES} bytes",
                received=f"{total_bytes} bytes",
            )
        )

    # ── Checks B-E — per image ─────────────────────────────────────────────
    # spec: 5.2 lines 939-970 — first failure terminates with 400
    validated: list[ValidatedImage] = []
    for raw_bytes, filename in images_bytes:
        vi = _validate_single_image(raw_bytes, filename=filename)
        validated.append(vi)

    # ── Check F — duplicate detection (multi-image only) ──────────────────
    # spec: 5.2 lines 957-958 — "compute SHA256 of each; if two or more share a
    #   hash, only the first is kept; others silently dropped"
    if len(validated) > 1:
        seen_hashes: set[str] = set()
        deduped: list[ValidatedImage] = []
        for vi in validated:
            if vi.sha256_hash not in seen_hashes:
                seen_hashes.add(vi.sha256_hash)
                deduped.append(vi)
            else:
                _logger.debug(
                    "duplicate_image_dropped",
                    sha256=vi.sha256_hash[:12] + "...",
                    note="spec 5.2 Check F — only first occurrence kept",
                )
        validated = deduped

    elapsed_ms = (time.perf_counter() - t0_request) * 1000.0
    _logger.debug(
        "request_validated",
        images_accepted=len(validated),
        images_submitted=n,
        total_payload_bytes=total_bytes,
        elapsed_ms=round(elapsed_ms, 2),
    )

    return validated
