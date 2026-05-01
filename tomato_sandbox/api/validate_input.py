"""
Re-export shim: ``tomato_sandbox.api.validate_input``.

The canonical module implementing Section 5 (Image Input and Validation Gate)
lives at ``tomato_sandbox.input_validation`` per spec Section 5.7 (line 1049):

    "``tomato_sandbox/input_validation.py`` defines the ``ValidatedImage``
    dataclass and the ``validate_request(request) -> List[ValidatedImage]``
    entry point."

This file re-exports all public symbols so callers can also use the
``tomato_sandbox.api.validate_input`` path.

DEC-029 documents this dual-path decision and the spec-vs-task-card divergence.

# spec: 5.7 line 1049
"""

from tomato_sandbox.input_validation import (  # noqa: F401
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
)
