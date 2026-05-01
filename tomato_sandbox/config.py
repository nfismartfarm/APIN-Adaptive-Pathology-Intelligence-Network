"""
Application configuration for the tomato 3-signal sandbox.

Reads TOMATO_* env vars + config/default.yaml (+ optional config/local.yaml).
Priority (highest first):
  1. Env vars (``TOMATO_`` prefix)
  2. ``tomato_sandbox/config/local.yaml`` (gitignored; local overrides)
  3. ``tomato_sandbox/config/default.yaml`` (committed defaults)
  4. Hardcoded fallbacks defined in this module

# spec: section 20.7 (configuration sources) lines 6591-6603
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config directory locations
# ---------------------------------------------------------------------------

_THIS_DIR: Path = Path(__file__).parent
_CONFIG_DIR: Path = _THIS_DIR / "config"
_DEFAULT_YAML: Path = _CONFIG_DIR / "default.yaml"
_LOCAL_YAML: Path = _CONFIG_DIR / "local.yaml"  # gitignored; may not exist


# ---------------------------------------------------------------------------
# YAML loader (optional dep — falls back gracefully)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return as nested dict. Returns {} on missing file."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import]
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except ImportError:
        # PyYAML not installed; fall back to hardcoded defaults silently.
        return {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# TomatoConfig dataclass
# spec: section 20.7 lines 6594-6598
# ---------------------------------------------------------------------------


@dataclass
class TomatoConfig:
    """Flat configuration surface for the sandbox.

    All values have hardcoded fallbacks so the server can start even if
    the YAML files are missing. Env vars override everything.

    # spec: section 20.7 — "Env vars at process startup" is highest precedence
    """

    # --- Server ---
    host: str = "127.0.0.1"
    # Port 8767 is the sandbox; 8766 is APIN; 8005 is unified server.
    # spec: section 20.3, BLK-002 / DEC-012
    port: int = 8767
    workers: int = 1  # spec: section 20.2 single-process per RTX 4060 VRAM constraint

    # --- GPU lock ---
    # spec: section 20.6 "configurable timeout (TOMATO_GPU_LOCK_TIMEOUT_S, default 10)"
    gpu_lock_timeout_s: float = 10.0

    # --- Pipeline ---
    # spec: section 20.3 info endpoint config block lines 6485-6489
    multi_image_max_n: int = 5
    tta_trigger_threshold: float = 0.55

    # --- Service identity (informational; served via /info) ---
    service_version: str = "tomato-sandbox-v1.0.0"
    build_hash: str = "stub"


def _deep_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Navigate nested dict with dot-path keys; return default if missing."""
    node: Any = data
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, None)
        if node is None:
            return default
    return node


def load_config() -> TomatoConfig:
    """Build a ``TomatoConfig`` by merging all config sources in priority order.

    Priority (highest first):
      1. ``TOMATO_*`` environment variables
      2. ``config/local.yaml`` (gitignored; local overrides)
      3. ``config/default.yaml`` (committed defaults)
      4. Hardcoded dataclass fallbacks

    # spec: section 20.7 lines 6591-6603
    """
    # Layer 3: default.yaml
    defaults = _load_yaml(_DEFAULT_YAML)
    # Layer 2: local.yaml (may be absent; silently ignored)
    local = _load_yaml(_LOCAL_YAML)

    def _merged(section: str, key: str, fallback: Any) -> Any:
        """Read from local → default → fallback (before env override)."""
        v = _deep_get(local, section, key)
        if v is not None:
            return v
        v = _deep_get(defaults, section, key)
        if v is not None:
            return v
        return fallback

    # Merge YAML layers (still below env vars)
    host = _merged("server", "host", "127.0.0.1")
    port = int(_merged("server", "port", 8767))
    workers = int(_merged("server", "workers", 1))
    gpu_lock_timeout_s = float(_merged("gpu", "lock_timeout_s", 10.0))
    multi_image_max_n = int(_merged("pipeline", "multi_image_max_n", 5))
    tta_trigger_threshold = float(_merged("pipeline", "tta_trigger_threshold", 0.55))

    # Layer 1: env vars override everything (TOMATO_ prefix)
    # spec: section 20.7 "Env vars at process startup" (highest precedence)
    host = os.environ.get("TOMATO_HOST", host)
    port = int(os.environ.get("TOMATO_PORT", str(port)))
    gpu_lock_timeout_s = float(
        os.environ.get("TOMATO_GPU_LOCK_TIMEOUT_S", str(gpu_lock_timeout_s))
    )
    multi_image_max_n = int(
        os.environ.get("TOMATO_MULTI_IMAGE_MAX_N", str(multi_image_max_n))
    )
    tta_trigger_threshold = float(
        os.environ.get("TOMATO_TTA_TRIGGER_THRESHOLD", str(tta_trigger_threshold))
    )
    build_hash = os.environ.get("TOMATO_BUILD_HASH", "stub")
    service_version = os.environ.get(
        "TOMATO_SERVICE_VERSION", "tomato-sandbox-v1.0.0"
    )

    return TomatoConfig(
        host=host,
        port=port,
        workers=workers,
        gpu_lock_timeout_s=gpu_lock_timeout_s,
        multi_image_max_n=multi_image_max_n,
        tta_trigger_threshold=tta_trigger_threshold,
        service_version=service_version,
        build_hash=build_hash,
    )


# Module-level singleton — loaded once at import time.
# Tests that need a custom config can call load_config() directly with env vars set.
CONFIG: TomatoConfig = load_config()


# ---------------------------------------------------------------------------
# Preprocessing pipeline constants
# spec: section 7.2 lines 1421-1432 — "Pinned constants (live in
#   tomato_sandbox/config.py; asserted at startup against checkpoint metadata
#   where available, per Section 4.4 training-inference parity)"
#
# These are PINNED INFERENCE-TIME CONSTANTS — they must match the values used
# at training time.  A mismatch produces silent accuracy degradation.
# They are NOT in TomatoConfig (no YAML override; pinned by training).
# ---------------------------------------------------------------------------

import numpy as _np  # noqa: E402 — import after dataclass block for readability

# spec: section 7.2 line 1424
CLAHE_CLIP_LIMIT: float = 2.0

# spec: section 7.2 line 1425
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)

# spec: section 7.2 lines 1426-1427 — "RGB order"
IMAGENET_MEAN: "_np.ndarray" = _np.array([0.485, 0.456, 0.406], dtype=_np.float32)

# spec: section 7.2 line 1427
IMAGENET_STD: "_np.ndarray" = _np.array([0.229, 0.224, 0.225], dtype=_np.float32)

# spec: section 7.2 line 1428
V3_INPUT_SIZE: int = 224

# spec: section 7.2 line 1429
LORA_INPUT_SIZE: int = 392

# spec: section 7.2 lines 1430-1431 — "used by preprocess_for_lora; matches LoRA's
# training pad value"
LORA_PAD_VALUE: int = 114

# spec: section 7.2 line 1431 — "passed to v3's HardFiLM at inference (Section 8)"
TOMATO_CROP_MODE_INDEX: int = 2
