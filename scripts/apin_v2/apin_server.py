"""APIN v2 — copy of the APIN website with a Field Notes drawer layered on top.

Architecture:
    This server REUSES the original `scripts.apin.section8_apin_server.make_app()`
    factory so every endpoint (/predict, /predict/apin, /predict/full, /feedback,
    /feedback/*, /warmup/status, /apin/info, /health) behaves identically to the
    original server on :8765. The ONLY differences are:

      1. The `/` route is overridden to serve `scripts/apin_v2/ui_template.html`
         (the copy that adds the Field Notes drawer) instead of the original
         `scripts/apin/ui_template.html`.
      2. Default port is 8766 (sibling of 8765), so both servers can run
         concurrently without port conflicts.

Why a wrapper, not a fork:
    The ensemble brain (inference.py, reliability matrix, PSV features,
    calibration) lives in scripts.apin.* and is shared with the original.
    Duplicating inference.py (1798 lines) would bit-rot the moment anyone
    fixes a bug in one copy. The wrapper pattern keeps the "website shell"
    independent while the scientific pipeline stays canonical.

Run:
    python scripts/apin_v2/apin_server.py --port 8766
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the original factory verbatim — every endpoint, every lock, every
# lazy singleton, every background task is shared.
from scripts.apin.section8_apin_server import make_app, get_apin  # noqa: E402

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apin_v2.server")


# ── Heatmap fix (monkeypatch over the read-only canonical pipeline) ────────
# scripts/apin/inference.py builds the Grad-CAM target from
# `model2.backbone.stages`, but the HuggingFace DINOv3ConvNextModel keeps its
# conv stages one level deeper, under `.model` (verified backbone children:
# model, layer_norm, pool). The mismatch made GradCAM++ raise AttributeError
# on every Model-2-driven prediction, so confident okra / brassica diagnoses
# returned no heatmap. scripts/apin/ is the shared canonical pipeline and is
# not edited from here; instead, right after Model 2 loads, we alias
# `backbone.stages` -> `backbone.model.stages` so the existing path resolves.
from scripts.apin.inference import APINInference as _APINInference  # noqa: E402

_orig_lazy_load_model2 = _APINInference._lazy_load_model2


def _lazy_load_model2_with_gradcam_fix(self):
    _orig_lazy_load_model2(self)
    try:
        bb = self._model2.backbone
        if (not hasattr(bb, "stages")
                and getattr(bb, "model", None) is not None
                and hasattr(bb.model, "stages")):
            # object.__setattr__ bypasses nn.Module.__setattr__, so the
            # ModuleList is aliased without being re-registered as a
            # duplicate child module.
            object.__setattr__(bb, "stages", bb.model.stages)
            logger.info("Grad-CAM fix applied: model2 backbone.stages "
                        "aliased to backbone.model.stages")
    except Exception as e:
        logger.warning(f"Grad-CAM backbone alias skipped: {e}")


_APINInference._lazy_load_model2 = _lazy_load_model2_with_gradcam_fix


# ── Grad-CAM target fix (monkeypatch over the read-only pipeline) ─────────
# scripts/apin/inference.py:_generate_gradcam targets, for the Model 2
# signal, `stages[3].layers[-1].depthwise_conv` — the depthwise conv INSIDE
# the last ConvNeXt block, before the pointwise (channel-mixing) convs.
# Those activations are not class-discriminative: the resulting CAM is a
# rainbow wash on confident predictions and a border artifact otherwise.
# The correct CAM target is the whole last block's output (after the
# pointwise convs + residual). Verified by rendering both: the block
# output localises tightly on the leaf lesions. Non-model2 signals are
# left to the canonical implementation.
import numpy as _np  # noqa: E402

_orig_generate_gradcam = _APINInference._generate_gradcam


def _pick_cam_signal(gate_weights_array):
    """The gate-weighted top signal that has a neural backbone (PSV has
    none). Mirrors the canonical _generate_gradcam selection."""
    n = len(gate_weights_array)
    order = (["model2", "efficientnet", "psv", "dinov2_head"] if n == 4
             else ["model2", "efficientnet", "dinov2_head"])
    for name, _w in sorted(zip(order, gate_weights_array), key=lambda x: -x[1]):
        if name != "psv":
            return name
    return None


def _generate_gradcam_fixed(self, img_rgb, gate_weights_array,
                            predicted_class_idx):
    chosen = _pick_cam_signal(gate_weights_array)
    if chosen != "model2":
        # efficientnet / dinov2 branches are unchanged — defer.
        return _orig_generate_gradcam(self, img_rgb, gate_weights_array,
                                      predicted_class_idx)
    try:
        import io as _io
        import base64 as _b64
        import cv2 as _cv2
        from PIL import Image as _PIL
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from scripts.apin.inference import preprocess_branch_a, M2_IMG_SIZE

        self._lazy_load_model2()
        model = self._model2
        bb = model.backbone
        stages = getattr(getattr(bb, "model", bb), "stages", None)
        if stages is None:
            stages = bb.stages
        # Target the last stage module itself. Its forward output equals
        # its last block's output (verified: identical CAM), and targeting
        # the stage avoids navigating version-specific block-list
        # attributes — transformers renamed the ConvNeXt stage internals
        # (`.layers` exists on some versions, not on the Space's 5.5.x).
        target = stages[-1]

        tensor = preprocess_branch_a(img_rgb, M2_IMG_SIZE).to(self.device)
        with GradCAMPlusPlus(model=model, target_layers=[target]) as cam:
            grayscale = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(predicted_class_idx)])[0]

        rgb_resized = _cv2.resize(img_rgb, (M2_IMG_SIZE, M2_IMG_SIZE))
        overlay = show_cam_on_image(
            rgb_resized.astype(_np.float32) / 255.0,
            grayscale, use_rgb=True, image_weight=0.55)
        buf = _io.BytesIO()
        _PIL.fromarray(overlay).save(buf, format="PNG", optimize=True)
        return _b64.b64encode(buf.getvalue()).decode("ascii"), "model2"
    except Exception as e:
        logger.warning(f"Grad-CAM (fixed model2 target) failed: {e}")
        return None, "model2"


_APINInference._generate_gradcam = _generate_gradcam_fixed


# ── OOD heatmap restoration (monkeypatch over the read-only pipeline) ──────
# scripts/apin/inference.py skips Grad-CAM for OOD tiers 4A/4B (line ~1778),
# so a best-guess card shows "no heatmap available". Farmers still want to
# see where the model looked. We regenerate the heatmap after predict()
# returns, reusing the pipeline's own _generate_gradcam (same GradCAM++,
# same target layer) — but only attach a genuinely FOCUSED map, never a
# diffuse full-image rainbow wash. Tier 3B (poor image quality) stays
# skipped: a heatmap on an unreadable photo is meaningless.
_orig_apin_predict = _APINInference.predict

# Focus gate: fraction of the frame that reads as "hot" in the JET overlay.
_OOD_CAM_HOT_MIN = 0.015   # below this → no spatial signal at all
_OOD_CAM_HOT_MAX = 0.62    # above this → diffuse wash, not a focused blob


def _ood_cam_focus(overlay_b64):
    """Return (is_focused, hot_fraction) for a Grad-CAM++ overlay PNG.
    In a JET overlay a hot pixel is strongly red-over-blue; a focused blob
    covers a modest, concentrated fraction of the frame, a rainbow wash
    covers most of it."""
    try:
        import io as _io
        import base64 as _b64
        from PIL import Image as _PIL
        arr = _np.asarray(
            _PIL.open(_io.BytesIO(_b64.b64decode(overlay_b64))).convert("RGB"),
            dtype=_np.int16)
        heat = arr[:, :, 0] - arr[:, :, 2]          # red minus blue
        hot_fraction = float((heat > 60).mean())
        focused = _OOD_CAM_HOT_MIN <= hot_fraction <= _OOD_CAM_HOT_MAX
        return focused, hot_fraction
    except Exception as e:
        logger.warning(f"OOD heatmap focus check failed: {e}")
        return False, 0.0


def _predict_with_ood_heatmap(self, img_rgb):
    result = _orig_apin_predict(self, img_rgb)
    try:
        if (result.gradcam_b64_png is None
                and getattr(result, "tier", None) in ("4A", "4B")
                and result.diagnosis in self.class_order
                and result.gate_weights):
            # result.gate_weights is insertion-ordered S1_M2, S2_EN,
            # (S3_PSV), S4_DINOv2 — positionally aligned with the signal
            # order _generate_gradcam expects (model2, efficientnet, psv,
            # dinov2_head).
            gate_arr = _np.array(list(result.gate_weights.values()),
                                 dtype=float)
            class_idx = self.class_order.index(result.diagnosis)
            cam_b64, cam_sig = self._generate_gradcam(
                img_rgb, gate_arr, class_idx)
            if cam_b64:
                focused, frac = _ood_cam_focus(cam_b64)
                if focused:
                    result.gradcam_b64_png = cam_b64
                    result.gradcam_source_signal = cam_sig
                    logger.info(
                        f"OOD heatmap restored: tier={result.tier} "
                        f"signal={cam_sig} hot_fraction={frac:.3f}")
                else:
                    logger.info(
                        f"OOD heatmap withheld (not focused): tier="
                        f"{result.tier} hot_fraction={frac:.3f}")
    except Exception as e:
        logger.warning(f"OOD heatmap regeneration skipped: {e}")
    return result


_APINInference.predict = _predict_with_ood_heatmap

APIN_V2_DIR = PROJECT_ROOT / "scripts" / "apin_v2"

# ── SEO / public-site constants ────────────────────────────────────────────
# The canonical production origin. Used for canonical links, Open Graph
# urls, robots.txt and sitemap.xml. The app is also reachable embedded in
# an iframe under huggingface.co/spaces/...; canonical points search
# engines at this direct origin.
_PUBLIC_BASE_URL = "https://dxv-404-apin.hf.space"

# Fresh-but-indexable: the public pages must always reflect the newest
# deploy (no stale browser copy), but unlike the dashboard headers these
# carry NO X-Robots-Tag, so search engines may still index them.
_PUBLIC_PAGE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma":        "no-cache",
}

# Hand-drawn leaf favicon (SVG — crisp at every size, dark-mode friendly).
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<path d="M13 51 C 7 32 19 9 53 11 C 56 44 35 58 13 51 Z" '
    'fill="#3f8a4d" stroke="#1b3a26" stroke-width="3.4" '
    'stroke-linejoin="round"/>'
    '<path d="M15 50 C 27 41 39 27 50 14" fill="none" '
    'stroke="#1b3a26" stroke-width="3" stroke-linecap="round"/>'
    '<path d="M26 42 C 25 36 23 32 18 30" fill="none" '
    'stroke="#1b3a26" stroke-width="2.4" stroke-linecap="round"/>'
    '<path d="M37 30 C 36 24 35 21 31 18" fill="none" '
    'stroke="#1b3a26" stroke-width="2.4" stroke-linecap="round"/>'
    '</svg>'
)


# ── Public agronomy content for the share view ─────────────────────────────
# diagnosis/diagnosis_lookup.json holds per-class full name, cause, symptoms,
# treatment (by severity) and prevention. It is static, public reference
# content, so the public /share view can safely surface it. Loaded once.
_DIAG_LOOKUP_CACHE = None


def _share_diagnosis_info(predicted_class):
    """Return the public diagnosis-lookup entry for a class, or None."""
    global _DIAG_LOOKUP_CACHE
    if _DIAG_LOOKUP_CACHE is None:
        try:
            with open(PROJECT_ROOT / "diagnosis" / "diagnosis_lookup.json",
                      encoding="utf-8") as f:
                _DIAG_LOOKUP_CACHE = json.load(f)
        except Exception as e:
            logger.warning(f"diagnosis_lookup.json not loaded: {e}")
            _DIAG_LOOKUP_CACHE = {}
    return _DIAG_LOOKUP_CACHE.get(predicted_class or "")


def _diag_lookup_all() -> dict:
    """The full diagnosis lookup dict, lazily loaded and cached."""
    _share_diagnosis_info("")          # primes _DIAG_LOOKUP_CACHE
    return _DIAG_LOOKUP_CACHE or {}


# ════════════════════════════════════════════════════════════════════
# Stage 2 · telemetry capture helpers
#
# These are pure functions consumed by _maybe_record() in the
# /predict/full override and by other inference paths that want to
# tag rows with EXIF, geo, and client metadata.
#
# All helpers swallow their own exceptions and return None/{} on
# failure so a malformed image / missing Pillow / weird header value
# can never break the actual inference response.
# ════════════════════════════════════════════════════════════════════
import hashlib as _hashlib
import os as _os
from typing import Optional as _Opt


# REV-R2-I10 (§22.3 row 6): IP_HASH_SALT is now MANDATORY at import time.
# If absent, the module fails to load with a clear RuntimeError instead of
# silently using a hardcoded fallback. This protects the privacy contract:
# an attacker with source-code access cannot re-hash known IPs to find them
# in the audit log because the salt is environment-private.
try:
    _IP_HASH_SALT = _os.environ["IP_HASH_SALT"]
except KeyError:
    raise RuntimeError(
        "IP_HASH_SALT environment variable is required. Set it to a long, "
        "random, deployment-private string (e.g. `python -c \"import "
        "secrets; print(secrets.token_urlsafe(32))\"`). See §11.4 of the "
        "API Console spec or .env.example."
    )
if not _IP_HASH_SALT or len(_IP_HASH_SALT) < 16:
    raise RuntimeError(
        "IP_HASH_SALT must be at least 16 characters long (got "
        f"{len(_IP_HASH_SALT)} chars). A short salt is trivially brute-forced."
    )


def _hash_client_ip(ip: _Opt[str], *, salt: _Opt[str] = None) -> _Opt[str]:
    """Privacy-preserving IP fingerprint. The salt is read from the mandatory
    `IP_HASH_SALT` environment variable (validated at module-import time).

    Returns the first 16 hex chars of sha256(salt || ip) — enough entropy
    to detect abuse / repeat visitors without storing the raw address.

    The `salt` parameter is an optional per-call override (used by tests
    that want to compare hashes against a fixed value). When omitted,
    the module-level `_IP_HASH_SALT` constant is used — that value comes
    from the `IP_HASH_SALT` env var at import time and has no fallback.
    There is NO hardcoded default; absence of the env var raises
    RuntimeError at import (REV-R2-I10 + PDA-P0-R1-F08 + R2-F01).
    """
    if not ip or not isinstance(ip, str) or ip.strip() in ("", "unknown"):
        return None
    try:
        # REV-R2-I10 (§22.3 row 6): the salt MUST come from the environment.
        # The previous hardcoded fallback `"apin-v2-ip-salt-2026"` weakened
        # the privacy contract — anyone reading the source could re-hash
        # known IPs and look them up in the audit log. The salt is now
        # mandatory; absence is a deploy-time bug, not a runtime fallback.
        s = salt or _IP_HASH_SALT
        h = _hashlib.sha256((s + ":" + ip.strip()).encode("utf-8")).hexdigest()
        return h[:16]
    except Exception:
        return None


def _client_ip_rightmost(request) -> str | None:
    """Rightmost-untrusted client-IP extraction (REV-R5-I08 / REV-R2-I03 / §11.3).

    SECURITY: an attacker can prepend arbitrary values to X-Forwarded-For. The
    LEFTMOST entry is therefore attacker-controlled and MUST NOT be trusted.
    Reverse proxies APPEND to the right; only the rightmost N entries written by
    OUR trusted hops can be relied on. The chosen IP is the one just *before*
    those trusted hops — i.e. the closest non-proxy entry.

    Algorithm (per spec §11.3 + RFC 7239 guidance):
        - Read APIN_TRUSTED_PROXY_HOPS env var (default 1 for HF Space edge).
        - Split XFF on "," and strip each entry.
        - Take entries[-(hops + 1)] when index is in range; otherwise the
          leftmost entry of the trimmed list (safer fallback than [0]).
        - If XFF is empty/missing, fall back to request.client.host.

    Example with `X-Forwarded-For: evil, attacker_proxy, 10.0.0.1` and hops=1:
        idx = -2 -> "attacker_proxy"   (NOT "evil")

    Returns the chosen IP string or None.
    """
    try:
        hops = int(_os.environ.get("APIN_TRUSTED_PROXY_HOPS", "1"))
    except Exception:
        hops = 1
    if hops < 0:
        hops = 0
    try:
        if request is None or not hasattr(request, "headers"):
            return None
        xff = request.headers.get("x-forwarded-for")
        if xff:
            entries = [e.strip() for e in xff.split(",") if e.strip()]
            if entries:
                idx = -(hops + 1)
                if -len(entries) <= idx < 0:
                    return entries[idx]
                # REV-R6-I04: when len(entries) <= hops, the ENTIRE chain is
                # within the configured trusted-hop window, meaning there is
                # no pre-proxy entry to trust. Falling back to entries[0]
                # would hand the attacker control in misconfigured deployments
                # (operator set hops too high). Fall through to request.client
                # — the direct TCP peer — which is the only trustworthy value
                # in this state.
        # XFF absent OR XFF entirely within trusted-hop window:
        # direct socket peer is the authoritative source.
        try:
            if request.client is not None:
                return request.client.host
        except Exception:
            pass
        return None
    except Exception:
        return None


def _client_geo_from_headers(headers) -> dict:
    """Best-effort {country, region, city} extraction from edge-injected
    request headers. Recognised:
       - Cloudflare       : CF-IPCountry, CF-Region, CF-IPCity
       - Vercel           : X-Vercel-IP-Country, X-Vercel-IP-Country-Region,
                            X-Vercel-IP-City
       - Fly.io           : Fly-Client-Country, Fly-Client-Region
       - generic          : X-Country, X-Region, X-City
    Returns an empty dict when no edge is in front of the server (localhost).

    Stage 2.5 audit fixes:
      - F-1/F-3: only accept exactly 2 uppercase ASCII alpha chars as country;
                 alpha-3 codes (USA) and 1-char garbage (U) are dropped, not
                 silently truncated/passed
      - F-4   : strip whitespace from region/city and only emit when non-empty
    """
    out = {}
    try:
        def _h(name):
            try: return headers.get(name)
            except Exception: return None
        country = (_h("CF-IPCountry") or _h("X-Vercel-IP-Country")
                   or _h("Fly-Client-Country") or _h("X-Country"))
        region  = (_h("X-Vercel-IP-Country-Region") or _h("CF-Region")
                   or _h("Fly-Client-Region") or _h("X-Region"))
        city    = (_h("X-Vercel-IP-City") or _h("CF-IPCity") or _h("X-City"))
        # [F-1, F-3] Country MUST be exactly 2 alpha chars and not a known
        # "unknown" sentinel. Anything else is dropped — better to have no
        # country signal than a wrong one polluting the analytics.
        if country and isinstance(country, str):
            c = country.strip().upper()
            if len(c) == 2 and c.isalpha() and c not in ("XX", "T1"):
                out["client_country"] = c
        # [F-4] Region / city — strip and require non-empty content.
        if region and isinstance(region, str):
            r = region.strip()
            if r:
                out["client_region"] = r[:64]
        if city and isinstance(city, str):
            ct = city.strip()
            if ct:
                out["client_city"] = ct[:64]
    except Exception:
        pass
    return out


def _extract_image_metadata(image_bytes: bytes) -> dict:
    """Best-effort image metadata via Pillow.

    Returns up to: image_width, image_height, image_mimetype, image_n_bytes,
                   exif_camera_model, exif_capture_timestamp,
                   exif_gps_lat, exif_gps_lon.

    The function is deliberately permissive: any EXIF parsing failure just
    skips that field. We never raise. This is the inference path —
    photographing latency budget is sacred and we don't trade ~5ms of
    metadata extraction for a chance to crash the prediction response.
    """
    out = {}
    # [PDA-2 F-7] Defensive isinstance: any non-bytes-like input (numpy
    # ndarray, str, dict, etc.) must not crash. An ndarray makes
    # `not image_bytes` raise "ambiguous truth value"; str / dict cause
    # nonsense image_n_bytes values. Refusing them outright is safer.
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        return out
    if len(image_bytes) == 0:
        return out
    try:
        out["image_n_bytes"] = len(image_bytes)
        from PIL import Image as _PIL, ExifTags as _ExifTags
        import io as _io
        img = _PIL.open(_io.BytesIO(image_bytes))
        out["image_width"]    = int(img.width)
        out["image_height"]   = int(img.height)
        # Pillow format names: JPEG, PNG, WEBP, ... → mime-ish string
        fmt = (img.format or "").lower()
        if fmt:
            out["image_mimetype"] = "image/" + fmt if fmt != "jpeg" else "image/jpeg"
        # EXIF — JPEG only in practice, but Pillow returns {} for others.
        exif = None
        try:
            exif = img._getexif()  # type: ignore[attr-defined]
        except Exception:
            exif = None
        if exif:
            # Build name->value map using ExifTags.TAGS
            tags = {_ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            model = tags.get("Model") or tags.get("CameraModelName")
            if model:
                try:
                    out["exif_camera_model"] = str(model).strip()[:120]
                except Exception:
                    pass
            ts = tags.get("DateTimeOriginal") or tags.get("DateTime")
            if ts:
                try:
                    s = str(ts).strip()
                    # EXIF format is "YYYY:MM:DD HH:MM:SS" — normalise to ISO-ish
                    if len(s) >= 19 and s[4] == ":" and s[7] == ":":
                        s = s[:4] + "-" + s[5:7] + "-" + s[8:10] + "T" + s[11:19]
                    out["exif_capture_timestamp"] = s[:32]
                except Exception:
                    pass
            # GPS — nested IFD under tag 'GPSInfo'
            gps_raw = tags.get("GPSInfo")
            if isinstance(gps_raw, dict) and gps_raw:
                gps = {}
                for k, v in gps_raw.items():
                    name = _ExifTags.GPSTAGS.get(k, k) if hasattr(_ExifTags, "GPSTAGS") else k
                    gps[name] = v

                def _dms_to_deg(dms, ref):
                    try:
                        d, m, s = dms
                        # Pillow IFDRational or tuple-of-ints/tuples
                        def _f(x):
                            try: return float(x)
                            except Exception:
                                try: return float(x.numerator) / float(x.denominator)
                                except Exception: return 0.0
                        deg = _f(d) + _f(m) / 60.0 + _f(s) / 3600.0
                        if str(ref).upper() in ("S", "W"):
                            deg = -deg
                        return round(deg, 6)
                    except Exception:
                        return None

                lat = _dms_to_deg(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
                lon = _dms_to_deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
                # [F-2] Reject (0.0, 0.0) outright. A phone with a broken GPS
                # chip reports DMS=(0,0,0) for both axes; the result is the
                # "Null Island" coordinate in the Atlantic Ocean which would
                # otherwise pollute the geo analytics. The probability of a
                # legitimate inference being shot inside the 110-metre square
                # centred on (0,0) is vanishingly small for okra/brassica
                # farmers anywhere in the world.
                _null_pair = (lat is not None and abs(lat) < 1e-6 and
                              lon is not None and abs(lon) < 1e-6)
                if (not _null_pair
                        and lat is not None and -90.0 <= lat <= 90.0):
                    out["exif_gps_lat"] = lat
                if (not _null_pair
                        and lon is not None and -180.0 <= lon <= 180.0):
                    out["exif_gps_lon"] = lon
    except Exception:
        # Pillow failure or any nested IFD weirdness — keep whatever
        # we already extracted (image_n_bytes is always present).
        pass
    return out


# Deployment fingerprint — lazily computed once per process.
_DEPLOYMENT_VERSION_CACHE: _Opt[str] = None


def _deployment_version() -> _Opt[str]:
    """Short identifier for the running build. First non-None of:
       - env DEPLOYMENT_VERSION
       - env GIT_COMMIT
       - git rev-parse --short HEAD (cached for the process lifetime)
    """
    global _DEPLOYMENT_VERSION_CACHE
    if _DEPLOYMENT_VERSION_CACHE is not None:
        return _DEPLOYMENT_VERSION_CACHE or None
    try:
        v = _os.environ.get("DEPLOYMENT_VERSION") or _os.environ.get("GIT_COMMIT")
        if not v:
            try:
                import subprocess as _sp
                r = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, timeout=2,
                            cwd=str(PROJECT_ROOT))
                if r.returncode == 0:
                    v = r.stdout.strip()
            except Exception:
                v = None
        _DEPLOYMENT_VERSION_CACHE = (v or "")[:32]
        return _DEPLOYMENT_VERSION_CACHE or None
    except Exception:
        _DEPLOYMENT_VERSION_CACHE = ""
        return None


def _build_inference_extras(*, request, image_bytes: bytes,
                            result: _Opt[dict], total_ms: _Opt[int] = None,
                            endpoint: str = "/predict/full",
                            api_version: str = "v2",
                            request_id: _Opt[str] = None,
                            cold_start: bool = False) -> dict:
    """Assemble the `extras` dict for record_prediction / record_guest_prediction.

    All keys are optional. None values are skipped by the DB layer.
    """
    import json as _json
    extras: dict = {}

    # Network / geo
    try:
        if request is not None:
            # REV-R5-I08: use rightmost-untrusted extraction (§11.3). The
            # previous `xff.split(",")[0]` form trusted the attacker-controlled
            # leftmost entry.
            client_ip = _client_ip_rightmost(request)
            extras["client_ip_hash"] = _hash_client_ip(client_ip)
            try:
                ua = request.headers.get("user-agent", "")
                if ua:
                    # Coarse family bucket — full UA is PII / fingerprintable.
                    ual = ua.lower()
                    if   "android" in ual: extras["user_agent_family"] = "android"
                    elif "iphone"  in ual or "ipad" in ual: extras["user_agent_family"] = "ios"
                    elif "windows" in ual: extras["user_agent_family"] = "windows"
                    elif "mac os"  in ual or "macintosh" in ual: extras["user_agent_family"] = "macos"
                    elif "linux"   in ual: extras["user_agent_family"] = "linux"
                    else: extras["user_agent_family"] = "other"
            except Exception:
                pass
            try:
                extras.update(_client_geo_from_headers(request.headers))
            except Exception:
                pass
    except Exception:
        pass

    # Image metadata + EXIF
    try:
        extras.update(_extract_image_metadata(image_bytes))
    except Exception:
        pass

    # Inference shape (signal_predictions, gate path, conformal set, ood)
    try:
        if isinstance(result, dict):
            # signal predictions — APIN ensemble exposes these on the
            # `signals` / `ensemble` keys depending on tier. Be permissive.
            sig = (result.get("signals") or result.get("signal_predictions")
                   or result.get("ensemble"))
            if sig is not None:
                try:
                    extras["signal_predictions"] = _json.dumps(sig, default=str)[:8000]
                except Exception:
                    pass
            gate = result.get("gate_decision_path") or result.get("gate")
            if gate is not None:
                try:
                    extras["gate_decision_path"] = (
                        gate if isinstance(gate, str)
                        else _json.dumps(gate, default=str)
                    )[:512]
                except Exception:
                    pass
            cset = (result.get("conformal_set") or result.get("prediction_set")
                    or result.get("conformal"))
            if cset is not None:
                try:
                    extras["conformal_set"] = _json.dumps(cset, default=str)[:2000]
                    if isinstance(cset, list):
                        extras["conformal_set_size"] = len(cset)
                    elif isinstance(cset, dict) and isinstance(cset.get("set"), list):
                        extras["conformal_set_size"] = len(cset["set"])
                except Exception:
                    pass
            ood = result.get("ood") or result.get("ood_flag")
            if ood is not None:
                try:
                    if isinstance(ood, bool):
                        extras["ood_flag"] = 1 if ood else 0
                    elif isinstance(ood, dict) and "is_ood" in ood:
                        extras["ood_flag"] = 1 if bool(ood["is_ood"]) else 0
                except Exception:
                    pass
            top3 = result.get("top3") or result.get("predicted_top3") or result.get("topk")
            if top3 is not None:
                try:
                    extras["predicted_top3"] = _json.dumps(top3, default=str)[:1000]
                except Exception:
                    pass
            if result.get("heatmap_b64") or result.get("gradcam_b64"):
                extras["grad_cam_generated"] = 1
            if "calibration_warning" in result:
                try:
                    cw = result["calibration_warning"]
                    extras["calibration_warning"] = (
                        cw if isinstance(cw, str) else _json.dumps(cw, default=str)
                    )[:512]
                except Exception:
                    pass
    except Exception:
        pass

    # Latency / build
    if total_ms is not None:
        try: extras["total_ms"] = int(total_ms)
        except Exception: pass
    try:
        dv = _deployment_version()
        if dv:
            extras["deployment_version"] = dv
    except Exception:
        pass
    try:
        import torch as _torch
        extras["gpu_used"] = 1 if _torch.cuda.is_available() else 0
    except Exception:
        pass
    if cold_start:
        extras["cold_start"] = 1

    extras["endpoint"]    = endpoint
    extras["api_version"] = api_version
    if request_id:
        extras["request_id"] = str(request_id)[:64]

    return extras


# ════════════════════════════════════════════════════════════════════
# Day-4 account-chip overlay — injected into ui_template.html at load time.
# This is a self-contained <style>+<script> block that adds a floating
# account chip in the top-right of the dashboard. When authenticated
# it shows the user's name + a popover with sign-out; when not, it shows
# a "Sign in" link to /landing.
# Pure additive — the dashboard HTML file itself is never modified.
# ════════════════════════════════════════════════════════════════════
_ACCOUNT_CHIP_SNIPPET = """
<style id="apin-account-chip-styles">
.apin-chip-root{
  position:fixed;
  top:14px;right:18px;
  z-index:9999;
  font-family:'Inter',system-ui,sans-serif;
  font-size:13px;
  opacity:0;
  transition:opacity 280ms cubic-bezier(0.16,1,0.3,1);
}
.apin-chip-root.ready{opacity:1}
.apin-chip-signin{
  display:inline-flex;align-items:center;gap:6px;
  padding:7px 14px;
  background:rgba(244,239,230,0.86);
  backdrop-filter:blur(6px);
  -webkit-backdrop-filter:blur(6px);
  border:1px solid #c7bca9;
  border-radius:18px;
  color:#1a1612;
  text-decoration:none;
  font-family:'Fraunces',Georgia,serif;
  font-style:italic;
  font-size:13px;
  letter-spacing:0.02em;
  transition:all 220ms cubic-bezier(0.16,1,0.3,1);
  box-shadow:0 1px 3px rgba(26,22,18,0.06);
}
.apin-chip-signin:hover{
  background:#1a1612; color:#fbf9f3;
  border-color:#1a1612;
}
.apin-chip-user{
  display:inline-flex;align-items:center;gap:10px;
  padding:5px 14px 5px 5px;
  background:rgba(251,249,243,0.92);
  backdrop-filter:blur(6px);
  -webkit-backdrop-filter:blur(6px);
  border:1px solid #c7bca9;
  border-radius:24px;
  cursor:pointer;
  user-select:none;
  transition:all 220ms cubic-bezier(0.16,1,0.3,1);
  box-shadow:0 1px 3px rgba(26,22,18,0.06);
}
.apin-chip-user:hover{
  background:#fbf9f3;
  box-shadow:0 4px 12px rgba(26,22,18,0.10);
  border-color:#7a6f60;
}
.apin-chip-avatar{
  width:28px;height:28px;border-radius:50%;
  background:#d9e6d3;
  display:flex;align-items:center;justify-content:center;
  font-family:'Fraunces',serif;
  font-weight:600;
  font-size:12px;
  color:#1e4d29;
  letter-spacing:0;
  text-transform:uppercase;
  flex-shrink:0;
}
.apin-chip-name{
  font-family:'Fraunces',serif;
  font-size:13.5px;
  color:#1a1612;
  font-weight:500;
  letter-spacing:-0.005em;
  max-width:160px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.apin-chip-caret{
  width:8px;height:8px;
  border-right:1.5px solid #7a6f60;
  border-bottom:1.5px solid #7a6f60;
  transform:rotate(45deg) translateY(-2px);
  margin-right:2px;
  transition:transform 200ms cubic-bezier(0.16,1,0.3,1);
}
.apin-chip-user.open .apin-chip-caret{
  transform:rotate(-135deg) translateY(2px);
}

.apin-popover{
  position:fixed;
  top:60px;right:18px;
  z-index:9998;
  min-width:300px;max-width:340px;
  background:#fbf9f3;
  border:1px solid #c7bca9;
  border-radius:8px;
  padding:0;
  box-shadow:0 8px 28px rgba(26,22,18,0.18), 0 2px 6px rgba(26,22,18,0.08);
  opacity:0;
  transform:translateY(-8px);
  pointer-events:none;
  transition:opacity 240ms cubic-bezier(0.16,1,0.3,1),
             transform 240ms cubic-bezier(0.16,1,0.3,1);
  font-family:'Inter',system-ui,sans-serif;
}
.apin-popover.show{
  opacity:1; transform:translateY(0); pointer-events:auto;
}
.apin-popover-header{
  padding:18px 20px 14px;
  border-bottom:1px solid #ded3bf;
}
.apin-popover-name{
  font-family:'Fraunces',serif;
  font-weight:500;
  font-size:18px;
  color:#1a1612;
  letter-spacing:-0.01em;
  font-variation-settings:'SOFT' 30,'WONK' 0;
  margin:0;
}
.apin-popover-handle{
  font-family:'Fraunces',serif;
  font-style:italic;
  font-size:12.5px;
  color:#7a6f60;
  margin:2px 0 0;
}
.apin-popover-meta{
  padding:12px 20px;
  border-bottom:1px solid #ded3bf;
  display:flex;justify-content:space-between;align-items:baseline;
  font-size:12px;
}
.apin-popover-meta .label{
  font-family:'Inter',sans-serif;
  color:#7a6f60;
  letter-spacing:0.06em;
  text-transform:uppercase;
  font-size:10px;
  font-weight:500;
}
.apin-popover-meta .value{
  font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:12.5px;
  color:#1a1612;
  font-weight:600;
}
.apin-popover-meta .value.muted{color:#7a6f60;font-weight:400}
.apin-popover-actions{
  padding:12px 14px;
  display:flex;flex-direction:column;gap:4px;
}
.apin-popover-link{
  display:flex;align-items:center;
  padding:9px 12px;
  border-radius:5px;
  font-family:'Inter',sans-serif;
  font-size:13px;
  color:#1a1612;
  text-decoration:none;
  cursor:pointer;
  transition:background 180ms;
  border:none;background:transparent;
  text-align:left;width:100%;
}
.apin-popover-link:hover{background:#e9e2d1}
.apin-popover-link.danger{color:#b01820}
.apin-popover-link.danger:hover{background:#fdedea}
.apin-popover-link .icon{
  display:inline-block;width:16px;text-align:center;margin-right:10px;
  font-size:13px;
}
</style>

<div class="apin-chip-root" id="apin-chip-root"></div>

<script>
(function(){
  "use strict";
  function hasSessionMarker(){
    return document.cookie.split(';').some(c => c.trim().startsWith('apin_v2_signed_in='));
  }
  function initials(name){
    if(!name) return '?';
    const parts = name.trim().split(/\\s+/);
    if(parts.length === 1) return parts[0].slice(0,2).toUpperCase();
    return (parts[0][0] + parts[parts.length-1][0]).toUpperCase();
  }
  function formatDate(iso){
    if(!iso) return '—';
    try {
      const d = new Date(iso);
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      return d.getDate() + ' ' + months[d.getMonth()] + ' ' + d.getFullYear();
    } catch(e){ return iso; }
  }

  const root = document.getElementById('apin-chip-root');
  if(!root) return;

  function renderSignedOut(){
    root.innerHTML = '<a class="apin-chip-signin" href="/landing">sign in →</a>';
    root.classList.add('ready');
  }
  function renderSignedIn(user){
    const dn = user.display_name || user.username || 'collector';
    root.innerHTML = ''
      + '<div class="apin-chip-user" id="apin-chip-user" role="button" tabindex="0" aria-expanded="false">'
      +   '<div class="apin-chip-avatar">' + initials(dn) + '</div>'
      +   '<span class="apin-chip-name">' + dn + '</span>'
      +   '<span class="apin-chip-caret"></span>'
      + '</div>';

    const pop = document.createElement('div');
    pop.className = 'apin-popover';
    pop.id = 'apin-popover';
    pop.innerHTML = ''
      + '<div class="apin-popover-header">'
      +   '<div class="apin-popover-name">' + dn + '</div>'
      +   '<div class="apin-popover-handle">@' + (user.username || '') + ' · ' + (user.email || '') + '</div>'
      + '</div>'
      + '<div class="apin-popover-meta">'
      +   '<span class="label">Member since</span>'
      +   '<span class="value muted">' + formatDate(user.created_at) + '</span>'
      + '</div>'
      + '<div class="apin-popover-meta">'
      +   '<span class="label">Specimens recorded</span>'
      +   '<span class="value">' + (user.predictions_count != null ? user.predictions_count : '—') + '</span>'
      + '</div>'
      + '<div class="apin-popover-actions">'
      +   '<a class="apin-popover-link" href="/dashboard"><span class="icon">◫</span>Open dashboard</a>'
      +   '<button class="apin-popover-link" id="apin-view-history"><span class="icon">▤</span>View history</button>'
      +   '<button class="apin-popover-link danger" id="apin-logout"><span class="icon">↩</span>Sign out</button>'
      + '</div>';
    document.body.appendChild(pop);

    const chip = document.getElementById('apin-chip-user');
    let open = false;
    function setOpen(o){
      open = o;
      chip.classList.toggle('open', o);
      pop.classList.toggle('show', o);
      chip.setAttribute('aria-expanded', String(o));
    }
    chip.addEventListener('click', () => setOpen(!open));
    chip.addEventListener('keydown', (e) => {
      if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); setOpen(!open); }
      if(e.key === 'Escape') setOpen(false);
    });
    document.addEventListener('click', (e) => {
      if(!open) return;
      if(chip.contains(e.target) || pop.contains(e.target)) return;
      setOpen(false);
    });
    document.addEventListener('keydown', (e) => {
      if(e.key === 'Escape') setOpen(false);
    });

    document.getElementById('apin-view-history').addEventListener('click', async () => {
      // Day-5 will build a real history view. For now: console-dump the list.
      try {
        const r = await fetch('/auth/history?limit=20', { credentials: 'same-origin' });
        const j = await r.json();
        console.log('[apin v2] History (' + j.total + ' total):', j.results);
        alert('Recent predictions: ' + j.total + '. (Full UI in next milestone — see console for raw data.)');
      } catch(e){ alert('Could not load history.'); }
    });

    document.getElementById('apin-logout').addEventListener('click', async () => {
      try {
        await fetch('/auth/logout', { method:'POST', credentials:'same-origin' });
      } catch(e){}
      // Clear non-HttpOnly marker locally too (cookie may already be cleared by server)
      document.cookie = 'apin_v2_signed_in=; Max-Age=0; path=/; samesite=lax';
      // Visual cue then refresh
      pop.classList.remove('show');
      chip.classList.remove('open');
      setTimeout(() => { window.location.reload(); }, 220);
    });

    root.classList.add('ready');
  }

  // Decide whether to even try fetching /auth/me
  if(!hasSessionMarker()){
    renderSignedOut();
    return;
  }
  fetch('/auth/me', { credentials: 'same-origin' })
    .then(r => r.ok ? r.json() : null)
    .then(user => {
      if(user && user.id){
        renderSignedIn(user);
      } else {
        // Stale marker — clear it so we don't keep hitting /me
        document.cookie = 'apin_v2_signed_in=; Max-Age=0; path=/; samesite=lax';
        renderSignedOut();
      }
    })
    .catch(() => renderSignedOut());
})();
</script>
"""


def _load_v2_html() -> str:
    """Load the v2 UI template + inject account-chip overlay.

    The injected snippet is appended just before </body>. The dashboard
    HTML file itself is never modified on disk — the overlay is purely
    server-side composition.
    """
    html_path = APIN_V2_DIR / "ui_template.html"
    try:
        html = html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (f"<h1>apin_v2/ui_template.html missing</h1>"
                f"<p>Expected at {html_path}</p>")

    # NOTE: the legacy Day-4 account-chip overlay (_ACCOUNT_CHIP_SNIPPET) is
    # no longer injected.  ui_template.html now ships its own account area
    # (#acct-area) + auth modal which understand the full auth model
    # (user / guest / anonymous).  Injecting the old chip too produced a
    # duplicate "sign in" affordance in the masthead.
    return html


def _load_landing_html() -> str:
    """Load the cinematic landing/login page (apin_v2 only)."""
    html_path = APIN_V2_DIR / "landing.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (f"<h1>apin_v2/landing.html missing</h1>"
                f"<p>Expected at {html_path}</p>")


def _add_landing_route(app):
    """Register GET /landing — the cinematic okra-leaf intro + login/signup page.

    This is purely additive (does NOT remove or shadow the existing '/' route
    which serves the dashboard). Anonymous access to '/' still works; once
    the Day-3 auth backend is wired, '/' can optionally enforce a redirect
    to '/landing' for unauthenticated users.
    """
    html = _load_landing_html()

    @app.get("/landing", response_class=HTMLResponse)
    async def v2_landing():
        return HTMLResponse(html, headers=_PUBLIC_PAGE_HEADERS)

    logger.info(f"v2 '/landing' route registered — serving "
                f"{APIN_V2_DIR / 'landing.html'} ({len(html):,} bytes)")


def _add_seo_routes(app):
    """Favicon, logo, robots.txt and sitemap.xml. Public and
    unauthenticated, served from this process so the Space needs no
    separate static-file mount."""
    from fastapi.responses import Response as _Resp, PlainTextResponse

    _ASSET_HEADERS = {"Cache-Control": "public, max-age=86400"}

    @app.get("/favicon.svg")
    async def v2_favicon_svg():
        return _Resp(content=_FAVICON_SVG, media_type="image/svg+xml",
                     headers=_ASSET_HEADERS)

    @app.get("/favicon.ico")
    async def v2_favicon_ico():
        # Browsers auto-request /favicon.ico on every page; modern
        # browsers render an SVG returned with the svg media type.
        return _Resp(content=_FAVICON_SVG, media_type="image/svg+xml",
                     headers=_ASSET_HEADERS)

    @app.get("/logo.png")
    async def v2_logo_png():
        logo = PROJECT_ROOT / "logo.png"
        if not logo.exists():
            return _Resp(status_code=404, content=b"")
        return _Resp(content=logo.read_bytes(), media_type="image/png",
                     headers=_ASSET_HEADERS)

    # Phase 8.H.D · serve the service worker from root so its scope "/"
    # encompasses /account/api/* without needing the
    # Service-Worker-Allowed header. Cache busted by ETag.
    @app.get("/apin_sw.js")
    async def v2_apin_sw():
        from fastapi.responses import Response as _Resp
        import hashlib as _hashlib
        sw_path = Path(__file__).resolve().parent / "apin_sw.js"
        try:
            body = sw_path.read_bytes()
        except FileNotFoundError:
            return _Resp(status_code=404, content=b"sw missing")
        etag = '"' + _hashlib.sha256(body).hexdigest()[:12] + '"'
        return _Resp(
            content=body,
            media_type="application/javascript",
            headers={
                "ETag": etag,
                "Cache-Control": "no-cache",
                # Allow the SW to claim the entire site as its scope.
                "Service-Worker-Allowed": "/",
            },
        )

    @app.get("/robots.txt")
    async def v2_robots():
        body = (
            "User-agent: *\n"
            "Disallow: /dashboard\n"
            "Disallow: /auth/\n"
            "Disallow: /share/\n"
            "Disallow: /predict\n"
            "Allow: /\n"
            f"Sitemap: {_PUBLIC_BASE_URL}/sitemap.xml\n"
        )
        return PlainTextResponse(body)

    @app.get("/sitemap.xml")
    async def v2_sitemap():
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f'  <url><loc>{_PUBLIC_BASE_URL}/</loc>'
            '<changefreq>weekly</changefreq><priority>1.0</priority></url>\n'
            f'  <url><loc>{_PUBLIC_BASE_URL}/landing</loc>'
            '<changefreq>monthly</changefreq><priority>0.6</priority></url>\n'
            '</urlset>\n'
        )
        return _Resp(content=body, media_type="application/xml")

    logger.info("v2 SEO routes registered — /favicon.svg /favicon.ico "
                "/logo.png /robots.txt /sitemap.xml")


def _add_auth_routes(app):
    """Mount the real DB-backed auth router (Day 3).

    Endpoints exposed:
      POST /auth/signup
      POST /auth/login
      POST /auth/logout
      GET  /auth/me
      GET  /auth/check          (replaces Day-1 stub)
      GET  /auth/next-accession (replaces Day-1 stub)
    """
    from scripts.apin_v2 import auth_routes
    auth_routes.attach(app)


# ════════════════════════════════════════════════════════════════════
# Day 4 — Dashboard routes
#
#   GET /dashboard          → HTML page (auth-gated, redirects to /landing)
#   GET /dashboard/data     → JSON payload for the widget set
#
# The HTML lives in apin_v2/dashboard.html and is loaded once at startup
# (same pattern as ui_template / landing). The JSON endpoint hits the
# `predictions` table via auth_db.get_dashboard_data().
# ════════════════════════════════════════════════════════════════════

def _load_dashboard_html() -> str:
    """Read scripts/apin_v2/dashboard.html into memory once at boot."""
    html_path = APIN_V2_DIR / "dashboard.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (f"<h1>apin_v2/dashboard.html missing</h1>"
                f"<p>Expected at {html_path}</p>")


def _load_history_html() -> str:
    """Read scripts/apin_v2/history.html into memory once at boot."""
    html_path = APIN_V2_DIR / "history.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (f"<h1>apin_v2/history.html missing</h1>"
                f"<p>Expected at {html_path}</p>")


def _load_html(filename: str) -> str:
    """Generic loader for Phase 3 HTML pages (reports.html, loupe.html,
    gallery.html, settings.html, share.html). Returns a friendly fallback
    page if the file is missing — useful during dev when a page is half-
    built."""
    html_path = APIN_V2_DIR / filename
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (f"<!doctype html><meta charset=utf-8>"
                f"<title>{filename} missing</title>"
                f"<style>body{{font:14px/1.5 system-ui;padding:40px;background:#fbf9f3}}</style>"
                f"<h1>{filename} not found</h1>"
                f"<p>Expected at <code>{html_path}</code></p>")


def _add_dashboard_routes(app):
    """Register all /dashboard/* routes.

    HTML pages (always served regardless of auth state — page decides what
    to show based on the data endpoint's response):
        GET  /dashboard
        GET  /dashboard/history

    JSON data endpoints (return dashtester's data via DEV-MODE FALLBACK
    when no session cookie is present):
        GET  /dashboard/data
        GET  /dashboard/history/data?page=&page_size=&crop=&disease=&tier=
                                    &from=&to=&search=&sort=

    Export action endpoints (honour all the same filters):
        GET  /dashboard/history/export.csv
        GET  /dashboard/history/export.json
    """
    from scripts.apin_v2 import auth_db, daily_brief
    from scripts.apin_v2.auth_routes import COOKIE_NAME as _AUTH_COOKIE

    dashboard_html = _load_dashboard_html()
    history_html   = _load_history_html()

    # ── AUTHENTICATION ──────────────────────────────────────────────────
    # Every /dashboard/* route is scoped to the signed-in user via their
    # session cookie.  There is NO anonymous fallback: an unauthenticated
    # request to a /dashboard route receives a clean 401 and the frontend
    # redirects to the inference page / shows the login modal.
    #
    # (A prior development build had a "dev-mode fallback" that resolved
    # anonymous requests to a fixed account so the dashboard UI could be
    # built without logging in.  That bypass has been removed — it exposed
    # one account's private data to every anonymous visitor.)

    _NO_CACHE_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma":        "no-cache",
        "Expires":       "0",
        # Dashboard, share and private API responses must never be indexed
        # (they carry user data). X-Robots-Tag is the HTTP-header form of
        # <meta name="robots" content="noindex"> and covers every page and
        # response that uses these headers in one place.
        "X-Robots-Tag":  "noindex, nofollow",
    }

    def _resolve_user(request: Request):
        """Resolve the signed-in user from the session cookie.

        Returns (user_dict, False) for a valid session, or (None, False)
        when there is no valid session.  The second tuple slot is retained
        as always-False for call-site compatibility (it used to flag the
        dev fallback, which no longer exists).
        """
        token = request.cookies.get(_AUTH_COOKIE)
        user = auth_db.get_session_user(token) if token else None
        if user is not None:
            return user, False
        return None, False

    def _auth_required_response():
        """401 returned by every /dashboard JSON route when the caller is
        not authenticated.  The frontend treats 401 as 'redirect to login'."""
        return JSONResponse(
            status_code=401,
            content={"detail": "authentication required",
                     "auth_required": True},
        )

    # ────────────────────────────────────────────────────────────────────
    #  HTML pages
    # ────────────────────────────────────────────────────────────────────

    # Server-side login gate for every dashboard HTML page.  An
    # unauthenticated visitor (anonymous OR guest — guests have no
    # dashboard) is redirected to the inference page before any dashboard
    # markup is sent, so there is never a flash of a logged-out dashboard.
    def _guard_dashboard_page(request: Request, html: str):
        user, _ = _resolve_user(request)
        if user is None:
            return RedirectResponse(url="/", status_code=303)
        return HTMLResponse(html, headers=_NO_CACHE_HEADERS)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def v2_dashboard(request: Request):
        return _guard_dashboard_page(request, dashboard_html)

    @app.get("/dashboard/history", response_class=HTMLResponse)
    async def v2_dashboard_history(request: Request):
        return _guard_dashboard_page(request, history_html)

    # ────────────────────────────────────────────────────────────────────
    #  /dashboard/data — main dashboard payload (extended)
    # ────────────────────────────────────────────────────────────────────

    @app.get("/dashboard/data")
    async def v2_dashboard_data(request: Request):
        user, is_dev = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        if is_dev:
            logger.info("[/dashboard/data] DEV-MODE fallback → user=%s id=%s",
                        user.get("username"), user.get("id"))

        # Base payload from the existing aggregator
        payload = auth_db.get_dashboard_data(user["id"])

        # ── Phase 1 additions ──────────────────────────────────────────
        # Confidence histogram (10 bins over [0,1]) for Container C mode 2
        payload["confidence_histogram"] = auth_db.confidence_histogram(
            user["id"], bins=10,
        )
        # Daily Brief narrative — template-grammar generator (LLM-swappable).
        # Pass the already-computed dashboard payload so the brief doesn't
        # re-run the 6 aggregation queries we already paid for.
        try:
            payload["daily_brief"] = daily_brief.build_brief_for_user(
                user["id"], auth_db, dashboard_data=payload,
            )
        except Exception as e:
            # The brief is a "nice to have" — failure here must not break
            # the rest of the dashboard. Log and ship an empty stub instead.
            logger.warning("[/dashboard/data] daily_brief failed: %s", e)
            payload["daily_brief"] = {"text": "", "salt": "", "tokens": {}}

        if is_dev:
            payload["_dev_fallback"] = {"username": user.get("username")}
        return JSONResponse(payload)

    # ────────────────────────────────────────────────────────────────────
    #  /dashboard/history/data — paginated, filtered prediction list
    # ────────────────────────────────────────────────────────────────────

    def _parse_history_query(request: Request) -> dict:
        """Read filter / pagination params off the query string with safe
        defaults. Values that don't pass the whitelist are silently dropped
        (the helpers in auth_db do this too — belt and braces)."""
        q = request.query_params
        try:
            page      = max(1, int(q.get("page", "1")))
        except ValueError:
            page = 1
        try:
            page_size = max(1, min(100, int(q.get("page_size", "25"))))
        except ValueError:
            page_size = 25
        sort = q.get("sort", "newest")
        if sort not in ("newest", "oldest", "highest", "lowest"):
            sort = "newest"

        # Truncate user-supplied free-text to defuse abusive queries.
        # All filter strings get a generous 60-char cap — no legitimate
        # value is longer than ~40 chars (brassica_alternaria_leaf_spot is 30).
        search = (q.get("q") or q.get("search") or "").strip()[:60] or None
        return {
            "crop":      ((q.get("crop")    or "").strip()[:60] or None),
            "disease":   ((q.get("disease") or "").strip()[:60] or None),
            "tier":      ((q.get("tier")    or "").strip()[:30] or None),
            "date_from": ((q.get("from")    or "").strip()[:10] or None),
            "date_to":   ((q.get("to")      or "").strip()[:10] or None),
            "search":    search,
            "sort":      sort,
            "page":      page,
            "page_size": page_size,
        }

    @app.get("/dashboard/history/data")
    async def v2_history_data(request: Request):
        user, is_dev = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        if is_dev:
            logger.info("[/dashboard/history/data] DEV-MODE fallback → %s",
                        user.get("username"))

        params = _parse_history_query(request)
        rows = auth_db.list_predictions(user["id"], **params)
        total_filtered = auth_db.count_predictions(
            user["id"],
            crop=params["crop"], disease=params["disease"], tier=params["tier"],
            date_from=params["date_from"], date_to=params["date_to"],
            search=params["search"],
        )
        total_unfiltered = auth_db.count_predictions(user["id"])

        # Per-disease + per-crop catalogs for the filter dropdowns — so the
        # UI knows what values are actually available rather than guessing.
        diseases = [
            d["class"] for d in auth_db.aggregate_by_disease(user["id"], limit=200)
        ]
        # Use the single-query helper instead of get_dashboard_data() which
        # ran 6 aggregation queries just to extract crop names. (Code-review
        # finding #4 — High severity, hot path.)
        crops = auth_db.list_user_crops(user["id"])

        # Pagination math
        page_size = params["page_size"]
        n_pages = max(1, (total_filtered + page_size - 1) // page_size)

        payload = {
            "results":         rows,
            "total":           total_filtered,
            "total_unfiltered": total_unfiltered,
            "page":            params["page"],
            "page_size":       page_size,
            "n_pages":         n_pages,
            "sort":            params["sort"],
            "filters_applied": {
                "crop":      params["crop"],
                "disease":   params["disease"],
                "tier":      params["tier"],
                "date_from": params["date_from"],
                "date_to":   params["date_to"],
                "search":    params["search"],
            },
            "available_filters": {
                "crops":    crops,
                "diseases": diseases,
                "tiers":    ["FIELD_GRADE", "LAB_GRADE", "UNCERTAIN", "OOD"],
                "sorts":    ["newest", "oldest", "highest", "lowest"],
            },
        }
        if is_dev:
            payload["_dev_fallback"] = {"username": user.get("username")}
        return JSONResponse(payload)

    # ────────────────────────────────────────────────────────────────────
    #  /dashboard/history/export.csv  +  .json
    #  Honour all the same filters; cap to 5,000 rows to prevent runaway.
    # ────────────────────────────────────────────────────────────────────

    EXPORT_CAP = 5000

    def _fetch_all_filtered(user_id: int, request: Request) -> list[dict]:
        params = _parse_history_query(request)
        params["page"]      = 1
        params["page_size"] = EXPORT_CAP
        return auth_db.list_predictions(user_id, **params)

    @app.get("/dashboard/history/export.csv")
    async def v2_history_export_csv(request: Request):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        rows = _fetch_all_filtered(user["id"], request)
        import csv, io
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([
            "id", "crop", "predicted_class", "confidence", "tier",
            "image_sha256", "created_at",
        ])
        for r in rows:
            w.writerow([
                r.get("id", ""), r.get("crop", "") or "",
                r.get("predicted_class", "") or "",
                r.get("confidence", ""),
                r.get("tier", "") or "",
                r.get("image_sha256", "") or "",
                r.get("created_at", "") or "",
            ])
        from fastapi.responses import Response
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={
                **_NO_CACHE_HEADERS,
                "Content-Disposition":
                    f"attachment; filename=\"apin_field_history_{len(rows)}.csv\"",
            },
        )

    @app.get("/dashboard/history/export.json")
    async def v2_history_export_json(request: Request):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        rows = _fetch_all_filtered(user["id"], request)
        from fastapi.responses import Response
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        payload = {
            "exported_at": _dt.now(_tz.utc).isoformat(),
            "n_rows":      len(rows),
            "results":     rows,
        }
        return Response(
            content=_json.dumps(payload, indent=2),
            media_type="application/json",
            headers={
                **_NO_CACHE_HEADERS,
                "Content-Disposition":
                    f"attachment; filename=\"apin_field_history_{len(rows)}.json\"",
            },
        )

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 2 ROUTES
    # ════════════════════════════════════════════════════════════════════

    # ── GET /dashboard/prediction/{id}  →  single prediction + signal votes
    #    Used by the Signal Vote Card modal (Phase 2 Overlay #3).
    @app.get("/dashboard/prediction/{pred_id}")
    async def v2_prediction_detail(request: Request, pred_id: int):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        pred = auth_db.get_prediction_with_signals(pred_id, user_id=user["id"])
        if pred is None:
            return JSONResponse(status_code=404,
                                content={"detail": "prediction not found"})
        # Drop the raw response_json blob from the wire payload — we already
        # parsed it into parsed_signals; the blob can be huge.
        pred = {k: v for k, v in pred.items() if k != "response_json"}
        return JSONResponse(pred)

    # ── GET /dashboard/disease/{class_name}/predictions  →  drill-down list
    #    Used by the Disease Drill-down modal (Phase 2 Overlay #2). Returns
    #    up to 200 predictions of the requested disease, newest first.
    @app.get("/dashboard/disease/{class_name}/predictions")
    async def v2_disease_predictions(request: Request, class_name: str):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        # Bound the class_name length for the same reason as filter inputs.
        cls = (class_name or "").strip()[:60]
        # max_page_size=200 lifts the default 100 cap so a user with up
        # to 200 occurrences of one disease gets the full mini-timeline.
        rows = auth_db.list_predictions(
            user["id"], disease=cls, sort="newest",
            page=1, page_size=200, max_page_size=200,
        )
        total = auth_db.count_predictions(user["id"], disease=cls)
        tax = auth_db.DISEASE_TAXONOMY.get(cls, {})
        return JSONResponse({
            "class":      cls,
            "taxonomy":   tax,
            "total":      total,
            "results":    rows,
        })

    # ── GET /dashboard/taxonomy  →  user-restricted disease family tree
    #    Used by Container D mode 2 (Disease Family Tree dendrogram).
    @app.get("/dashboard/taxonomy")
    async def v2_taxonomy(request: Request):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        return JSONResponse({"tree": auth_db.build_taxonomy_tree(user["id"])})

    # ── GET /dashboard/notes  →  list this user's margin notes
    #    Optional filters: ?date=YYYY-MM-DD or ?prediction_id=N
    @app.get("/dashboard/notes")
    async def v2_notes_list(request: Request):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        q = request.query_params
        date = (q.get("date") or "").strip()[:10] or None
        pid_raw = q.get("prediction_id")
        try:
            pid = int(pid_raw) if pid_raw else None
        except ValueError:
            pid = None
        notes = auth_db.list_margin_notes(
            user["id"],
            attached_date=date,
            attached_prediction_id=pid,
        )
        return JSONResponse({"results": notes, "total": len(notes)})

    # ── POST /dashboard/notes  →  create a new margin note
    #    Body JSON: {text, attached_date?, attached_prediction_id?, mood?}
    @app.post("/dashboard/notes")
    async def v2_notes_create(request: Request):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"detail": "invalid JSON body"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400,
                                content={"detail": "body must be object"})
        try:
            note = auth_db.create_margin_note(
                user["id"],
                text=body.get("text", ""),
                attached_date=(body.get("attached_date") or None),
                attached_prediction_id=body.get("attached_prediction_id"),
                mood=body.get("mood"),
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        return JSONResponse(note, status_code=201)

    # ── PATCH /dashboard/notes/{id}  →  edit text and/or mood
    @app.patch("/dashboard/notes/{note_id}")
    async def v2_notes_update(request: Request, note_id: int):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"detail": "invalid JSON body"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400,
                                content={"detail": "body must be object"})
        try:
            updated = auth_db.update_margin_note(
                note_id, user_id=user["id"],
                text=body.get("text"),
                mood=body.get("mood"),
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        if updated is None:
            return JSONResponse(status_code=404,
                                content={"detail": "note not found"})
        return JSONResponse(updated)

    # ── DELETE /dashboard/notes/{id}
    @app.delete("/dashboard/notes/{note_id}")
    async def v2_notes_delete(request: Request, note_id: int):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        ok = auth_db.delete_margin_note(note_id, user_id=user["id"])
        if not ok:
            return JSONResponse(status_code=404,
                                content={"detail": "note not found"})
        return JSONResponse({"deleted": note_id})

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 3 ROUTES — Treatment Log + Share Tokens + PDF + new pages
    # ════════════════════════════════════════════════════════════════════
    from scripts.apin_v2 import report_pdf as _report_pdf

    # Pre-load Phase 3 HTML once at boot
    reports_html  = _load_html("reports.html")
    loupe_html    = _load_html("loupe.html")
    gallery_html  = _load_html("gallery.html")
    settings_html = _load_html("settings.html")
    share_html    = _load_html("share.html")

    # ── /dashboard/reports HTML page (hub for PDF + Share + Treatment Export)
    # All four standalone dashboard pages use the same server-side login
    # gate as /dashboard itself.
    @app.get("/dashboard/reports", response_class=HTMLResponse)
    async def v2_reports_page(request: Request):
        return _guard_dashboard_page(request, reports_html)

    @app.get("/dashboard/loupe", response_class=HTMLResponse)
    async def v2_loupe_page(request: Request):
        return _guard_dashboard_page(request, loupe_html)

    @app.get("/dashboard/gallery", response_class=HTMLResponse)
    async def v2_gallery_page(request: Request):
        return _guard_dashboard_page(request, gallery_html)

    @app.get("/dashboard/settings", response_class=HTMLResponse)
    async def v2_settings_page(request: Request):
        return _guard_dashboard_page(request, settings_html)

    # ── Treatment Log CRUD ─────────────────────────────────────────────

    @app.get("/dashboard/treatments")
    async def v2_treatments_list(request: Request):
        user, is_dev = _resolve_user(request)
        if user is None: return _auth_required_response()
        q = request.query_params
        rows = auth_db.list_treatments(
            user["id"],
            crop=(q.get("crop") or "").strip()[:30] or None,
            disease=(q.get("disease") or "").strip()[:60] or None,
            date_from=(q.get("from") or "").strip()[:10] or None,
            date_to=(q.get("to") or "").strip()[:10] or None,
        )
        payload = {"results": rows, "total": len(rows)}
        if is_dev: payload["_dev_fallback"] = {"username": user["username"]}
        return JSONResponse(payload)

    @app.post("/dashboard/treatments")
    async def v2_treatments_create(request: Request):
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        try: body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "invalid JSON"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"detail": "body must be object"})
        # Code-review finding #3: target_prediction_id is a foreign key but
        # SQLite cannot enforce cross-table column constraints (i.e. that
        # the referenced prediction belongs to the SAME user). Verify
        # ownership here BEFORE the insert. Without this check, an
        # authenticated user could link their treatment to ANY user's
        # prediction id, turning a successful insert into an enumeration
        # oracle.
        tpid = body.get("target_prediction_id")
        if tpid is not None:
            if not isinstance(tpid, int):
                return JSONResponse(status_code=400,
                                    content={"detail": "target_prediction_id must be integer"})
            owned = auth_db.get_prediction(tpid, user_id=user["id"])
            if owned is None:
                return JSONResponse(status_code=404,
                                    content={"detail": "target prediction not found"})
        try:
            row = auth_db.create_treatment(
                user["id"],
                treatment=body.get("treatment", ""),
                applied_date=body.get("applied_date", ""),
                crop=body.get("crop"),
                disease=body.get("disease"),
                plot=body.get("plot"),
                notes=body.get("notes"),
                target_prediction_id=tpid,
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        return JSONResponse(row, status_code=201)

    @app.patch("/dashboard/treatments/{tid}")
    async def v2_treatments_update(request: Request, tid: int):
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        try: body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "invalid JSON"})
        try:
            row = auth_db.update_treatment(
                tid, user_id=user["id"],
                treatment=body.get("treatment"),
                crop=body.get("crop"),
                disease=body.get("disease"),
                plot=body.get("plot"),
                notes=body.get("notes"),
                applied_date=body.get("applied_date"),
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        if row is None:
            return JSONResponse(status_code=404, content={"detail": "treatment not found"})
        return JSONResponse(row)

    @app.delete("/dashboard/treatments/{tid}")
    async def v2_treatments_delete(request: Request, tid: int):
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        ok = auth_db.delete_treatment(tid, user_id=user["id"])
        if not ok:
            return JSONResponse(status_code=404, content={"detail": "treatment not found"})
        return JSONResponse({"deleted": tid})

    # ── Treatment Log CSV export ───────────────────────────────────────
    @app.get("/dashboard/treatments/export.csv")
    async def v2_treatments_export_csv(request: Request):
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        q = request.query_params
        rows = auth_db.list_treatments(
            user["id"],
            crop=(q.get("crop") or "").strip()[:30] or None,
            disease=(q.get("disease") or "").strip()[:60] or None,
            date_from=(q.get("from") or "").strip()[:10] or None,
            date_to=(q.get("to") or "").strip()[:10] or None,
            limit=2000,
        )
        import csv, io
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["id", "applied_date", "treatment", "crop", "disease",
                    "plot", "notes", "target_prediction_id", "created_at"])
        for r in rows:
            w.writerow([
                r.get("id", ""), r.get("applied_date", "") or "",
                r.get("treatment", "") or "", r.get("crop", "") or "",
                r.get("disease", "") or "", r.get("plot", "") or "",
                (r.get("notes") or "").replace("\n", " "),
                r.get("target_prediction_id", "") or "",
                r.get("created_at", "") or "",
            ])
        from fastapi.responses import Response
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={
                **_NO_CACHE_HEADERS,
                "Content-Disposition":
                    f"attachment; filename=\"apin_treatment_log_{len(rows)}.csv\"",
            },
        )

    # ── Share-token CRUD ───────────────────────────────────────────────

    @app.get("/dashboard/shares")
    async def v2_shares_list(request: Request):
        user, is_dev = _resolve_user(request)
        if user is None: return _auth_required_response()
        rows = auth_db.list_share_tokens(user["id"])
        payload = {"results": rows, "total": len(rows)}
        if is_dev: payload["_dev_fallback"] = {"username": user["username"]}
        return JSONResponse(payload)

    @app.post("/dashboard/shares")
    async def v2_shares_create(request: Request):
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        try: body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "invalid JSON"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"detail": "body must be object"})
        pid = body.get("prediction_id")
        if not isinstance(pid, int):
            return JSONResponse(status_code=400,
                                content={"detail": "prediction_id must be integer"})
        # Verify the prediction is owned by this user before exposing it
        pred = auth_db.get_prediction(pid, user_id=user["id"])
        if pred is None:
            return JSONResponse(status_code=404,
                                content={"detail": "prediction not found or not yours"})
        out = auth_db.create_share_token(
            user["id"], pid,
            label=body.get("label"),
            expires_at=body.get("expires_at"),
        )
        return JSONResponse(out, status_code=201)

    @app.delete("/dashboard/shares/{sid}")
    async def v2_shares_revoke(request: Request, sid: int):
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        ok = auth_db.revoke_share_token(sid, user_id=user["id"])
        if not ok:
            return JSONResponse(status_code=404,
                                content={"detail": "share not found or already revoked"})
        return JSONResponse({"revoked": sid})

    # ── PUBLIC share viewer (no auth required) ─────────────────────────
    @app.get("/share/{token}")
    async def v2_share_view(token: str):
        # Serve the share viewer HTML as-is. The page reads the token from
        # window.location.pathname itself, so we DO NOT inject the token
        # via string-replace anymore — doing so was a reflected-XSS vector
        # because a crafted token like '";alert(1);//' would break out of
        # the JS string literal in the template.  See code-review fix #1.
        return HTMLResponse(share_html, headers=_NO_CACHE_HEADERS)

    @app.get("/share/{token}/data")
    async def v2_share_data(token: str, request: Request):
        # Defensive — bound token length so a malicious caller can't pass GB
        token = (token or "")[:128]
        # Count a view at most once per browser per share: a refresh by the
        # same viewer must not inflate the owner's view count. A per-token
        # cookie marks "already counted"; it expires after 12 h so a genuine
        # return visit later still registers as a new view.
        import hashlib
        seen_cookie = "apin_sv_" + hashlib.sha256(
            token.encode("utf-8")).hexdigest()[:10]
        already_seen = request.cookies.get(seen_cookie) is not None
        pred = auth_db.resolve_share_token(token, count_view=not already_seen)
        if pred is None:
            return JSONResponse(status_code=404,
                                content={"detail": "share link not found, "
                                                   "revoked, or expired"})
        # Enrich with public agronomy content (full name, cause, symptoms,
        # treatment, prevention) so the public view is informative.
        info = _share_diagnosis_info(pred.get("predicted_class"))
        if info:
            pred["diagnosis_info"] = info
        resp = JSONResponse(pred)
        if not already_seen:
            resp.set_cookie(seen_cookie, "1", max_age=43200,
                            httponly=True, samesite="lax")
        return resp

    # ── PUBLIC share image / heatmap (binary, owner-stripped) ──────────
    # These two routes serve the exact bytes for a shared specimen.  They
    # do NOT consult the request's session at all — only the share token
    # is the auth principal — which is why they live alongside the
    # /share/{token}/data route and not under /dashboard/.
    #
    # Important security properties:
    #   - Token alphabet is enforced by FastAPI's path parser (any byte
    #     that breaks the URL grammar 404s before this handler runs) AND
    #     by the [:128] length cap below.
    #   - Revoked / expired tokens 404 via _resolve_share_pid().
    #   - Tokens for predictions whose owner deleted the row 404 because
    #     the row no longer exists.
    #   - Pre-Phase-3.5 prediction rows (where image_bytes IS NULL) 404
    #     — the frontend renders an honest "image not captured" tile.
    @app.get("/share/{token}/image")
    async def v2_share_image(token: str):
        from fastapi.responses import Response
        token = (token or "")[:128]
        data = auth_db.resolve_share_image(token)
        if data is None:
            return JSONResponse(status_code=404,
                                content={"detail": "image not available"})
        # We don't know the original mime type for sure (the upload could
        # have been JPEG or PNG).  Sniff the first bytes: PNG starts with
        # 0x89 P N G, JPEG with 0xFF 0xD8, WebP via RIFF.....WEBP.
        head = bytes(data[:12])
        if head[:3] == b"\xff\xd8\xff":
            mt = "image/jpeg"
        elif head[:8] == b"\x89PNG\r\n\x1a\n":
            mt = "image/png"
        elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            mt = "image/webp"
        else:
            mt = "application/octet-stream"
        return Response(content=data, media_type=mt,
                        headers={**_NO_CACHE_HEADERS,
                                 "Cache-Control": "private, max-age=60"})

    @app.get("/share/{token}/heatmap")
    async def v2_share_heatmap(token: str):
        from fastapi.responses import Response
        token = (token or "")[:128]
        data = auth_db.resolve_share_heatmap(token)
        if data is None:
            return JSONResponse(status_code=404,
                                content={"detail": "heatmap not available"})
        # Grad-CAM is always produced as PNG by the inference pipeline.
        return Response(content=data, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=60"})

    # ── OWNER-GATED image / heatmap (binary) ───────────────────────────
    # Same shape as the share routes but auth principal is the session
    # cookie.  IDOR shield: the SQL in get_prediction_image() requires
    # `id = ? AND user_id = ?` so a logged-in attacker cannot probe
    # another user's prediction ids — they get 404 indistinguishable
    # from "no such row".
    @app.get("/dashboard/predictions/{pid}/image")
    async def v2_pred_image(request: Request, pid: int):
        from fastapi.responses import Response
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        data = auth_db.get_prediction_image(int(pid), user_id=user["id"])
        if data is None:
            return JSONResponse(status_code=404,
                                content={"detail": "image not available"})
        head = bytes(data[:12])
        if head[:3] == b"\xff\xd8\xff":
            mt = "image/jpeg"
        elif head[:8] == b"\x89PNG\r\n\x1a\n":
            mt = "image/png"
        elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            mt = "image/webp"
        else:
            mt = "application/octet-stream"
        return Response(content=data, media_type=mt,
                        headers={"Cache-Control": "private, max-age=300"})

    @app.get("/dashboard/predictions/{pid}/heatmap")
    async def v2_pred_heatmap(request: Request, pid: int):
        from fastapi.responses import Response
        user, _ = _resolve_user(request)
        if user is None: return _auth_required_response()
        data = auth_db.get_prediction_heatmap(int(pid), user_id=user["id"])
        if data is None:
            return JSONResponse(status_code=404,
                                content={"detail": "heatmap not available"})
        return Response(content=data, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=300"})

    # ── Weekly reports: list / generate / view / delete / restore ──────
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    def _week_range(monday):
        """For a Monday date return (start_date_iso, end_date_iso,
        start_dt_iso, end_dt_iso). The dt bounds are [Mon 00:00, next
        Mon 00:00), used for the created_at range query."""
        sunday = monday + _td(days=6)
        nxt = monday + _td(days=7)
        return (monday.isoformat(), sunday.isoformat(),
                monday.isoformat() + "T00:00:00+00:00",
                nxt.isoformat() + "T00:00:00+00:00")

    @app.get("/dashboard/reports/list")
    async def v2_reports_list(request: Request):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        counts = auth_db.weekly_prediction_counts(user["id"])
        existing = {r["week_start"]: r
                    for r in auth_db.list_reports(user["id"])}
        today = _dt.now(_tz.utc).date()
        this_mon = today - _td(days=today.weekday())
        weeks = []
        for i in range(12):
            mon = this_mon - _td(days=7 * i)
            ws, we, _, _ = _week_range(mon)
            rep = existing.get(ws)
            weeks.append({
                "week_start": ws, "week_end": we, "is_current": (i == 0),
                "specimens": counts.get(ws, 0),
                "report": ({"id": rep["id"],
                            "generated_at": rep.get("generated_at"),
                            "summary": rep.get("summary")}
                           if rep else None),
            })
        return JSONResponse({"weeks": weeks})

    @app.post("/dashboard/reports/generate")
    async def v2_reports_generate(request: Request):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        q = request.query_params
        today = _dt.now(_tz.utc).date()
        try:
            mon = _dt.fromisoformat(q.get("week_start", "")).date()
            mon = mon - _td(days=mon.weekday())
        except (ValueError, TypeError):
            mon = today - _td(days=today.weekday())
        ws, we, start_dt, end_dt = _week_range(mon)
        _, _, p_start, p_end = _week_range(mon - _td(days=7))
        uid = user["id"]
        preds = auth_db.predictions_in_range(uid, start_dt, end_dt)
        prev = auth_db.predictions_in_range(uid, p_start, p_end)
        treatments = auth_db.list_treatments(uid, date_from=ws, date_to=we)

        def _fi(pid):
            try:
                return auth_db.get_prediction_image(int(pid), user_id=uid)
            except Exception:
                return None

        def _fh(pid):
            try:
                return auth_db.get_prediction_heatmap(int(pid), user_id=uid)
            except Exception:
                return None

        try:
            pdf_bytes, summary = _report_pdf.generate_weekly_pdf(
                user=user, predictions=preds, prev_predictions=prev,
                week_start=ws, week_end=we,
                fetch_image=_fi, fetch_heatmap=_fh,
                treatments=treatments, diagnosis_lookup=_diag_lookup_all())
        except Exception as e:
            logger.exception(f"weekly report render failed: {e}")
            return JSONResponse(status_code=500,
                                content={"detail": "report render failed"})
        try:
            rid = auth_db.save_report(uid, week_start=ws, week_end=we,
                                      pdf_bytes=pdf_bytes, summary=summary)
        except Exception as e:
            logger.exception(f"weekly report save failed: {e}")
            return JSONResponse(status_code=500,
                                content={"detail": "report save failed"})
        return JSONResponse({"id": rid, "week_start": ws, "week_end": we,
                             "summary": summary})

    @app.get("/dashboard/reports/{rid}/pdf")
    async def v2_report_pdf(request: Request, rid: int):
        from fastapi.responses import Response
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        data = auth_db.get_report_pdf(int(rid), user_id=user["id"])
        if data is None:
            return JSONResponse(status_code=404,
                                content={"detail": "report not found"})
        disp = ("attachment" if request.query_params.get("download")
                else "inline")
        return Response(
            content=data, media_type="application/pdf",
            headers={**_NO_CACHE_HEADERS,
                     "Content-Disposition":
                         f'{disp}; filename="apin_weekly_report.pdf"'})

    @app.post("/dashboard/reports/{rid}/delete")
    async def v2_report_delete(request: Request, rid: int):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        ok = auth_db.soft_delete_report(int(rid), user_id=user["id"])
        if not ok:
            return JSONResponse(status_code=404,
                                content={"detail": "report not found"})
        return JSONResponse({"ok": True})

    @app.post("/dashboard/reports/{rid}/restore")
    async def v2_report_restore(request: Request, rid: int):
        user, _ = _resolve_user(request)
        if user is None:
            return _auth_required_response()
        ok = auth_db.restore_report(int(rid), user_id=user["id"])
        if not ok:
            return JSONResponse(status_code=409,
                                content={"detail": "could not restore "
                                                   "(it may no longer exist)"})
        return JSONResponse({"ok": True})

    logger.info(
        "v2 dashboard routes registered — Phase 1 + 2 + 3 routes online. "
        "Phase 3: /dashboard/reports + /loupe + /gallery + /settings + "
        "/dashboard/treatments + /dashboard/treatments/export.csv + "
        "/dashboard/shares + /share/{token} + /dashboard/reports/*",
    )


def _override_index_route(app):
    """Replace the '/' GET route on the existing FastAPI app so it serves
    the v2 HTML instead of the original template.

    FastAPI stores routes on app.router.routes. We REMOVE the original
    GET '/' route entirely, then register a fresh one via app.get() so
    FastAPI rebuilds the ASGI-wrapped callable correctly. Mutating
    route.app directly breaks the ASGI contract (route.app expects
    (scope, receive, send), not the plain endpoint).
    """
    from starlette.routing import Route
    html = _load_v2_html()

    # Remove the original '/' GET route in-place.
    original_routes = app.router.routes
    removed = 0
    for i in range(len(original_routes) - 1, -1, -1):
        r = original_routes[i]
        if isinstance(r, Route) and r.path == "/" and "GET" in r.methods:
            original_routes.pop(i)
            removed += 1

    # Register a fresh '/' route via the normal FastAPI path.
    # FastAPI builds the ASGI wrapper correctly this way.
    @app.get("/", response_class=HTMLResponse)
    async def v2_index():
        return HTMLResponse(html, headers=_PUBLIC_PAGE_HEADERS)

    logger.info(f"v2 '/' route replaced (removed {removed} original) — "
                f"serving {APIN_V2_DIR / 'ui_template.html'} "
                f"({len(html):,} bytes)")


# ---------------------------------------------------------------------------
# Phase B Task B.3: /predict/full route override — ADDITIVE, no scripts/apin edits.
#
# Pattern identical to `_override_index_route` above. We remove the original
# POST /predict/full route registered by `make_app()` in scripts.apin.section8
# and register a wrapper route that:
#   - runs the router (via _classify_crop imported from apin)
#   - for okra/brassica (or low-conf): delegates to the ORIGINAL apin engine
#     (BYTE-IDENTICAL behavior — we re-implement the original function inline
#      by calling get_apin().predict() directly, matching section8 lines 296-314)
#   - for tomato: calls TomatoPipeline.infer(img) [integrated tomato path]
#   - for chilli: returns the same ROUTER_REJECTED message as before
#
# CRITICAL INVARIANT: for okra/brassica and chilli inputs, the response produced
# by this override must be semantically equivalent to what the original
# predict_full produced (response_run1 from tests/regression_baseline_pre_integration.json).
# Phase B.7 will verify this field-by-field (excluding processing_time_ms which
# is known non-deterministic per B.1 findings).
# ---------------------------------------------------------------------------
def _override_predict_full_route(app, tomato_pipeline):
    """Replace POST /predict/full on the app with an augmented wrapper.

    tomato_pipeline: instance of scripts.ladi_net.tomato_pipeline.TomatoPipeline,
                     or None (if load failed — tomato branch returns graceful error).
    """
    import io
    from dataclasses import asdict
    from PIL import Image
    import numpy as np
    from starlette.routing import Route
    # UploadFile, File, HTTPException, JSONResponse are imported at module top
    # so FastAPI/Pydantic can resolve type annotations on the decorator.

    # Import the apin module's helpers WITHOUT modifying the module.
    from scripts.apin.section8_apin_server import get_apin  # noqa: E402

    # _classify_crop lives in a closure inside make_app(); we cannot import it
    # directly. Instead we import the raw Router pieces and build our own
    # classify function matching section8's logic (same threshold, same tensor).
    try:
        from app.config_router import (
            BACKBONE_NAME as ROUTER_BACKBONE, DINOV2_IMG_SIZE as ROUTER_IMG_SIZE,
            CLASS_NAMES as ROUTER_CLASSES,
        )
        import timm, torch, cv2
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        _ROUTER_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Router imports failed: {e}; /predict/full override will error for non-tomato crops")
        _ROUTER_AVAILABLE = False

    # Identify & remove the original POST /predict/full route.
    original_routes = app.router.routes
    removed_predict_full = 0
    for i in range(len(original_routes) - 1, -1, -1):
        r = original_routes[i]
        if isinstance(r, Route) and r.path == "/predict/full" and "POST" in r.methods:
            original_routes.pop(i)
            removed_predict_full += 1

    # Constants mirroring section8's EXACTLY. Verified against section8_apin_server.py
    # lines 193-195 on 2026-04-24. Any drift here causes okra/brassica regressions.
    APIN_HANDLED_CROPS = {"okra", "brassica"}
    CROP_CONF_MIN = 0.40  # section8 line 194; NOT 0.60

    # Router state + init, copied from section8 lines 196-249 to ensure identical
    # routing behavior. Uses the DEDICATED router checkpoint at models/router/router_best.pt,
    # NOT models/best_model.pt.
    _router_state = {"backbone": None, "head": None, "transform": None,
                     "device": None, "class_names": None, "loaded": False}
    import threading
    _router_init_lock = threading.Lock()

    def _ensure_router():
        if _router_state["loaded"]:
            return _router_state
        with _router_init_lock:
            if _router_state["loaded"]:
                return _router_state
            try:
                from app.config_router import (
                    BACKBONE_NAME, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM,
                    NUM_CLASSES, CLASS_NAMES,
                )
                from torch import nn
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # CRITICAL: pretrained=True matches section8 line 216. NOT False.
                backbone = timm.create_model(
                    BACKBONE_NAME, pretrained=True,
                    num_classes=0, img_size=DINOV2_IMG_SIZE,
                ).eval().to(device)
                for p in backbone.parameters():
                    p.requires_grad = False
                head = nn.Linear(DINOV2_EMBED_DIM, NUM_CLASSES).to(device)
                # CRITICAL: section8 loads models/router/router_best.pt (line 222), NOT
                # models/best_model.pt. Using the wrong checkpoint causes Router to
                # produce completely different crop confidences.
                ckpt_path = PROJECT_ROOT / "models" / "router" / "router_best.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    if "head_state_dict" in ckpt:
                        head.load_state_dict(ckpt["head_state_dict"])
                    elif "model_state_dict" in ckpt:
                        sd = ckpt["model_state_dict"]
                        head_sd = {k.replace("head.", ""): v for k, v in sd.items()
                                     if k.startswith("head.")}
                        if head_sd:
                            head.load_state_dict(head_sd)
                    head.eval()
                else:
                    logger.warning(f"Router checkpoint missing: {ckpt_path}")
                transform = A.Compose([
                    A.Resize(DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
                _router_state.update(backbone=backbone, head=head,
                                       transform=transform, device=device,
                                       class_names=CLASS_NAMES, loaded=True)
                logger.info("Router loaded (in-process, apin_v2 override)")
            except Exception as e:
                logger.warning(f"Router load failed; /predict/full override will skip routing: {e}")
                _router_state["loaded"] = True   # mark loaded-with-failure
        return _router_state

    def _classify_crop(img_np):
        """Mirror of section8's _classify_crop (lines 252-269)."""
        st = _ensure_router()
        if st.get("backbone") is None:
            return None, 0.0
        try:
            tens = st["transform"](image=img_np)["image"].unsqueeze(0).to(st["device"])
            with torch.no_grad():
                feat = st["backbone"](tens)
                logits = st["head"](feat)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            argmax = int(probs.argmax())
            return st["class_names"][argmax], float(probs[argmax])
        except Exception as e:
            logger.warning(f"Router inference failed: {e}")
            return None, 0.0

    def parse_image(file_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _convert_np(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        if isinstance(o, dict): return {k: _convert_np(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert_np(v) for v in o]
        return o

    @app.post("/predict/full")
    async def v2_predict_full(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ):
        """V2 override: preserves okra/brassica/chilli behavior byte-for-byte,
        routes tomato to TomatoPipeline for ensemble inference.

        Day-4 addition: when the request carries a valid session cookie, the
        prediction is written to the `predictions` table after the response
        is returned (via FastAPI BackgroundTasks). Anonymous requests are NOT
        recorded (privacy-preserving default).
        """
        try:
            data = await file.read()
            img = parse_image(data)

            # ── Authentication gate ───────────────────────────────────────
            # /predict/full requires EITHER a signed-in user OR a guest
            # session with free checks remaining.  An anonymous or
            # quota-exhausted caller receives a 401 that the frontend
            # turns into the login / sign-up modal.  This is the
            # authoritative server-side gate — the frontend also gates,
            # but the quota is only trustworthy because it's enforced here.
            from scripts.apin_v2 import auth_db as _auth_db
            from scripts.apin_v2.auth_routes import (
                COOKIE_NAME as _COOKIE, GUEST_COOKIE as _GUEST_COOKIE)

            current_user_id = None
            guest_token     = None
            guest_session_id = None   # numeric row id, populated below
            auth_mode       = "anonymous"   # user | guest | guest_exhausted | anonymous
            if request is not None:
                try:
                    _tok = request.cookies.get(_COOKIE)
                    if _tok:
                        _u = _auth_db.get_session_user(_tok)
                        if _u:
                            current_user_id = _u["id"]
                            auth_mode = "user"
                except Exception:
                    pass  # never fail the request because of an auth lookup
                if current_user_id is None:
                    try:
                        guest_token = request.cookies.get(_GUEST_COOKIE)
                        if guest_token:
                            _g = _auth_db.get_guest_session(guest_token)
                            if _g is not None:
                                guest_session_id = int(_g["id"])
                                auth_mode = ("guest_exhausted"
                                             if _g["exhausted"] else "guest")
                    except Exception:
                        pass

            # Stage 2: capture wall-clock start so we can record total_ms.
            import time as _stage2_time
            _t_start = _stage2_time.perf_counter()

            if auth_mode == "anonymous":
                return JSONResponse(status_code=401, content={
                    "detail": "Sign in or continue as a guest to run a diagnosis.",
                    "auth_required": True, "reason": "no_auth"})
            if auth_mode == "guest_exhausted":
                return JSONResponse(status_code=401, content={
                    "detail": ("You've used all your free guest checks. "
                               "Create a free account to keep diagnosing."),
                    "auth_required": True, "reason": "guest_exhausted"})

            def _maybe_record(result_dict):
                """Post-inference finalisation:
                  • signed-in users → schedule a background DB write of the
                    prediction (with image bytes + Stage 2 telemetry extras)
                    under their account
                  • guests          → consume one free check, write the
                    prediction to guest_predictions (Stage 2 — was discarded
                    before), and stamp the remaining quota onto the response
                    as `_guest`
                Server-side errors (the TOMATO_* error tiers) are NOT
                counted against a guest's quota and are NOT written to
                guest_predictions — only real results are.
                """
                if not isinstance(result_dict, dict):
                    return result_dict
                _tier = str(result_dict.get("tier", "")).upper()
                _is_error = _tier in ("TOMATO_UNAVAILABLE",
                                      "TOMATO_INFERENCE_ERROR")

                # Stage 2: build the shared extras envelope once.
                # This wraps the EXIF / IP-hash / geo / signal-predictions /
                # latency / build metadata that both record paths will store.
                try:
                    _total_ms = int(
                        (_stage2_time.perf_counter() - _t_start) * 1000.0
                    )
                except Exception:
                    _total_ms = None
                try:
                    _extras = _build_inference_extras(
                        request=request,
                        image_bytes=data,
                        result=result_dict,
                        total_ms=_total_ms,
                        endpoint="/predict/full",
                        api_version="v2",
                        request_id=request.headers.get("x-request-id")
                            if request is not None else None,
                    )
                except Exception:
                    _extras = None
                # Capture the per-row error fields when the response is a tier-error.
                if _is_error and isinstance(_extras, dict):
                    err_class = result_dict.get("error")
                    if err_class:
                        _extras["error_class"] = str(err_class)[:64]
                        _extras["error_message"] = str(
                            result_dict.get("message", "")
                        )[:512]
                    _extras["status_code"] = 200  # we still return 200 with tier=error

                if current_user_id is not None and background_tasks is not None:
                    background_tasks.add_task(
                        _auth_db.record_prediction,
                        current_user_id,
                        result_dict,
                        image_bytes=data,
                        extras=_extras,
                    )
                elif auth_mode == "guest" and guest_token and not _is_error:
                    # Atomically consume one free check first — this is the
                    # quota gate, and a guest_predictions row must not exist
                    # for a request the quota didn't actually allow.
                    _gc = _auth_db.consume_guest_inference(guest_token)
                    if _gc is not None and not _gc.get("denied"):
                        result_dict["_guest"] = {
                            "remaining": _gc.get("remaining", 0),
                            "limit": _auth_db.GUEST_INFERENCE_LIMIT,
                        }
                        # Stage 2: persist the prediction to guest_predictions.
                        # Backgrounded so the response is not blocked on disk I/O.
                        if (guest_session_id is not None
                                and background_tasks is not None):
                            background_tasks.add_task(
                                _auth_db.record_guest_prediction,
                                guest_session_id,
                                result_dict,
                                image_bytes=data,
                                extras=_extras,
                            )
                    elif _gc is not None and _gc.get("denied"):
                        # Race: another concurrent request consumed the last
                        # free check between our auth gate and this point.
                        # The response was generated but we cannot count it.
                        result_dict["_guest"] = {
                            "remaining": 0,
                            "limit": _auth_db.GUEST_INFERENCE_LIMIT,
                            "denied_at_record": True,
                        }
                return result_dict

            # Route
            crop, conf = _classify_crop(img)
            routing = {
                "router_crop": crop,
                "router_confidence": round(conf, 4),
                "router_handled": crop in APIN_HANDLED_CROPS,
                "low_router_confidence": (conf < CROP_CONF_MIN),
            }

            # Tomato integrated path
            if crop == "tomato" and conf >= CROP_CONF_MIN:
                # PDA R1 B-6 fix: tomato error/unavailable paths must be distinguishable
                # from a genuine ROUTER_REJECTED (non-handled crop). Use tier=
                # "TOMATO_UNAVAILABLE" and include specialist field so the client can
                # show the right message, and the regression test can separate the two.
                if tomato_pipeline is None:
                    return _maybe_record({
                        "routing": routing,
                        "specialist": "tomato_v3_sp_lora_ensemble",
                        "diagnosis": None,
                        "confidence": 0.0,
                        "tier": "TOMATO_UNAVAILABLE",
                        "message": ("Detected crop: tomato (confidence {:.2f}). "
                                    "Tomato specialist failed to load at startup. "
                                    "Please check server logs.").format(conf),
                        "error": "pipeline_not_loaded",
                    })
                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: tomato_pipeline.infer(img, routing_info=routing)
                    )
                    # Result from TomatoPipeline already has `routing` embedded; ensure JSON-safe
                    safe = _convert_np(result)
                    _maybe_record(safe)
                    return JSONResponse(safe)
                except Exception as e:
                    logger.exception("TomatoPipeline inference failed")
                    return _maybe_record({
                        "routing": routing,
                        "specialist": "tomato_v3_sp_lora_ensemble",
                        "diagnosis": None,
                        "confidence": 0.0,
                        "tier": "TOMATO_INFERENCE_ERROR",
                        "message": f"Tomato inference error: {type(e).__name__}. Please retry.",
                        "error": type(e).__name__,
                    })

            # Chilli / other non-APIN crops: byte-identical to section8's original behavior.
            if crop is not None and crop not in APIN_HANDLED_CROPS and conf >= CROP_CONF_MIN:
                return _maybe_record({
                    "routing": routing,
                    "diagnosis": None,
                    "message": (
                        f"Detected crop: {crop} (confidence {conf:.2f}). "
                        f"This is outside Model 2 APIN's scope (okra + brassica). "
                        f"A {crop} specialist is planned but not yet deployed. "
                        f"Please retake with an okra or brassica leaf, or "
                        f"contact your agricultural extension officer."
                    ),
                    "tier": "ROUTER_REJECTED",
                })

            # Okra / brassica / low-confidence: delegate to APIN (byte-identical to original).
            apin_engine = get_apin()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: apin_engine.predict(img))
            try:
                out = asdict(result)
            except Exception:
                out = {k: v for k, v in vars(result).items() if not k.startswith("_")}
            out = _convert_np(out)
            out["routing"] = routing
            return _maybe_record(out)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("v2_predict_full failed")
            raise HTTPException(500, str(e))

    # Expose v2 router state to the app so /warmup/status override can read it.
    app.state.v2_router_state = _router_state
    app.state.v2_tomato_pipeline = tomato_pipeline

    logger.info(f"v2 '/predict/full' route replaced (removed {removed_predict_full} original) — "
                f"tomato_pipeline {'loaded' if tomato_pipeline is not None else 'UNAVAILABLE'}")


def _override_warmup_status_route(app):
    """Override /warmup/status so router_loaded reflects the v2 override's
    _router_state (not section8's closure state which is never populated
    when v2 is the active wrapper).

    PDA Round 1 Issue B-1 fix: without this override, /warmup/status on port
    8766 reports router_loaded=false forever, causing the frontend cold-start
    loader to hang.

    This is purely additive — removes the original route and registers a new
    one that wraps the original's body plus patches router_loaded from
    app.state.v2_router_state.
    """
    from starlette.routing import Route
    from datetime import datetime

    # Remove original /warmup/status
    removed = 0
    for i in range(len(app.router.routes) - 1, -1, -1):
        r = app.router.routes[i]
        if isinstance(r, Route) and r.path == "/warmup/status" and "GET" in r.methods:
            app.router.routes.pop(i)
            removed += 1

    # Need access to section8's _apin singleton to report other loaded states
    from scripts.apin.section8_apin_server import get_apin as _s8_get_apin

    @app.get("/warmup/status")
    async def v2_warmup_status():
        import torch
        cuda_avail = torch.cuda.is_available()
        entry = {
            "apin_constructed": False,
            "router_loaded": False,
            "model2_loaded": False,
            "efficientnet_loaded": False,
            "psv_ready": False,
            "dinov2_loaded": False,
            "all_ready": False,
            "gpu_vram_gb": (round(torch.cuda.memory_allocated() / 1e9, 2)
                            if cuda_avail else 0.0),
            "device": "cuda" if cuda_avail else "cpu",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            # Tomato-specific extension (apin_v2 only):
            "tomato_pipeline_loaded": (getattr(app.state, "v2_tomato_pipeline", None) is not None),
        }

        # Read v2 router state (the one actually used by /predict/full override)
        v2_rs = getattr(app.state, "v2_router_state", {})
        entry["router_loaded"] = bool(v2_rs.get("loaded") and
                                       v2_rs.get("backbone") is not None)

        # Fetch section8's APIN singleton to report other flags (APIN is lazy-loaded)
        try:
            a = _s8_get_apin()
            entry["apin_constructed"] = (a is not None)
            if a is not None:
                entry["model2_loaded"]       = (getattr(a, "_model2", None) is not None)
                entry["efficientnet_loaded"] = (getattr(a, "_efficientnet", None) is not None)
                entry["dinov2_loaded"]       = (getattr(a, "_dinov2_backbone", None) is not None)
                entry["psv_ready"]           = (getattr(a, "psv_calibration", None) is not None)
                ood = getattr(a, "_ood_detector", None)
                entry["ood_threshold"] = (float(ood["threshold"]) if ood is not None else None)
        except Exception as e:
            logger.warning(f"warmup_status: APIN introspection failed: {e}")

        entry["apin_ready"] = (
            entry["model2_loaded"] and entry["efficientnet_loaded"]
            and entry["dinov2_loaded"] and entry["psv_ready"]
        )
        entry["all_ready"] = entry["apin_ready"] and entry["router_loaded"]
        return entry

    logger.info(f"v2 '/warmup/status' route replaced (removed {removed} original) — "
                f"router_loaded now reflects v2 override state")


def _load_tomato_pipeline_safe():
    """Load TomatoPipeline with exception handling; return None on failure.

    Honors `APIN_SKIP_TOMATO=1` for QA / observability-test runs that don't
    need real predictions (usage telemetry, dashboard tests, etc.). When
    skipped, the /predict/full handler will return graceful errors, which
    is fine — the UsageRecordingMiddleware still records every request.
    """
    import os as _os_skip
    if _os_skip.environ.get("APIN_SKIP_TOMATO") == "1":
        logger.warning("TomatoPipeline load skipped (APIN_SKIP_TOMATO=1) — "
                        "predictions will error but auth + telemetry work.")
        return None
    try:
        import sys
        _ladi_path = str(PROJECT_ROOT / "scripts" / "ladi_net")
        if _ladi_path not in sys.path:
            sys.path.insert(0, _ladi_path)
        from scripts.ladi_net.tomato_pipeline import TomatoPipeline
        logger.info("Loading TomatoPipeline...")
        tp = TomatoPipeline()
        logger.info("TomatoPipeline loaded successfully")
        return tp
    except Exception as e:
        logger.exception(f"TomatoPipeline failed to load: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
# Status & health monitoring
#
#   GET /status        → public visual status page (status.html)
#   GET /status/data   → JSON the status page polls every 30 s
#   GET /health        → content-negotiated: a browser gets health.html,
#                        a machine (curl / UptimeRobot) gets JSON
#
# A background task records one heartbeat a minute into the database
# (auth_db.record_heartbeat). The pages read that history so the 90-day
# uptime bars survive a process restart. Every component probe is wrapped
# so a single failing check degrades ONE row rather than erroring the page.
#
# Honest limitation: no in-process page can report that the whole Space
# has crashed - that is exactly what the external UptimeRobot monitor covers.
# ════════════════════════════════════════════════════════════════════

from datetime import datetime as _dtmod, timezone as _tzmod, timedelta as _td_


def _utcnow():
    """UTC-aware current time. Tiny wrapper so the status code reads cleanly."""
    return _dtmod.now(_tzmod.utc)


# Wall-clock time this worker process started - drives the "uptime" KPI.
_PROCESS_START = _utcnow()

# Guard so the heartbeat loop is started exactly once (see _ensure_heartbeat).
_heartbeat_started = False

# Fixed display order + human labels for the six tracked components.
_STATUS_COMPONENTS = [
    ("web",     "Web application"),
    ("db",      "Database"),
    ("router",  "Crop router"),
    ("okra",    "Okra & Brassica model"),
    ("tomato",  "Tomato model"),
    ("signals", "Diagnostic signals"),
]


def _collect_status(app) -> dict:
    """Run every component probe in isolation and return a status snapshot.

    Each probe is wrapped in try/except: a failing component reports as
    'down' or 'degraded' instead of raising, so the snapshot is ALWAYS a
    valid dict no matter what state the app is in.

    Returns: {overall, components{key: {...}}, checked_at,
              process_uptime_s, check_latency_ms}
    """
    import time as _t
    t0 = _t.perf_counter()
    components: dict = {}

    # 1 - Web application. If this code is running, the process is serving.
    components["web"] = {
        "status": "up", "label": "Web application",
        "detail": "Serving requests normally",
    }

    # 2 - Database (Turso in production / SQLite locally).
    try:
        from scripts.apin_v2 import auth_db
        probe = auth_db.status_db_probe()
        if probe.get("ok"):
            components["db"] = {
                "status": "up", "label": "Database",
                "detail": (f"{probe.get('backend', '?')} responding in "
                           f"{probe.get('latency_ms', '?')} ms"),
                "backend": probe.get("backend"),
                "latency_ms": probe.get("latency_ms"),
            }
        else:
            components["db"] = {
                "status": "down", "label": "Database",
                "detail": f"Unreachable - {probe.get('error', 'no response')}"[:200],
                "backend": probe.get("backend"),
            }
    except Exception as e:
        components["db"] = {
            "status": "down", "label": "Database",
            "detail": f"Probe error - {e}"[:200],
        }

    # 3 - Crop router. Lazy-loaded; "ready, not yet warm" is healthy.
    try:
        rs = getattr(app.state, "v2_router_state", {}) or {}
        loaded = bool(rs.get("loaded") and rs.get("backbone") is not None)
        components["router"] = {
            "status": "up", "label": "Crop router",
            "detail": ("Loaded and routing crops"
                       if loaded else "Ready - warms on first diagnosis"),
            "loaded": loaded,
        }
    except Exception as e:
        components["router"] = {
            "status": "down", "label": "Crop router",
            "detail": f"Error - {e}"[:200],
        }

    # 4 - Okra & Brassica model. Surfaces real DINOv3 vs the timm fallback,
    #     which is the production-misdiagnosis signal we want visible.
    try:
        from scripts.apin.section8_apin_server import get_apin as _ga
        a = _ga()
        m2 = getattr(a, "_model2", None) if a is not None else None
        if m2 is None:
            components["okra"] = {
                "status": "up", "label": "Okra & Brassica model",
                "detail": "Ready - backbone loads on first diagnosis",
                "backbone": "pending",
            }
        else:
            bt = getattr(m2, "_backbone_type", None)
            if bt == "transformers":
                components["okra"] = {
                    "status": "up", "label": "Okra & Brassica model",
                    "detail": "Running the real DINOv3-ConvNeXt backbone",
                    "backbone": "dinov3",
                }
            else:
                components["okra"] = {
                    "status": "degraded", "label": "Okra & Brassica model",
                    "detail": ("On the generic timm fallback backbone - "
                               "accuracy reduced. Set the HF_TOKEN secret."),
                    "backbone": "fallback",
                }
    except Exception as e:
        components["okra"] = {
            "status": "down", "label": "Okra & Brassica model",
            "detail": f"Error - {e}"[:200], "backbone": "error",
        }

    # 5 - Tomato model. Optional pipeline: absent => degraded, not down,
    #     because okra/brassica/chilli diagnosis is unaffected by it.
    try:
        tp = getattr(app.state, "v2_tomato_pipeline", None)
        if tp is not None:
            components["tomato"] = {
                "status": "up", "label": "Tomato model",
                "detail": "Pipeline loaded and ready",
            }
        else:
            components["tomato"] = {
                "status": "degraded", "label": "Tomato model",
                "detail": "Pipeline unavailable - tomato diagnosis offline",
            }
    except Exception as e:
        components["tomato"] = {
            "status": "down", "label": "Tomato model",
            "detail": f"Error - {e}"[:200],
        }

    # 6 - Diagnostic signals. The APIN ensemble (Model 2 / EfficientNet /
    #     DINOv2 / PSV). Reachable orchestrator => healthy; signals warm
    #     lazily so a low warm-count before first use is normal.
    try:
        from scripts.apin.section8_apin_server import get_apin as _ga
        a = _ga()
        if a is None:
            components["signals"] = {
                "status": "up", "label": "Diagnostic signals",
                "detail": "Ready - signals warm on first diagnosis",
                "warm": 0, "total": 4,
            }
        else:
            warm = sum([
                getattr(a, "_model2", None) is not None,
                getattr(a, "_efficientnet", None) is not None,
                getattr(a, "_dinov2_backbone", None) is not None,
                getattr(a, "psv_calibration", None) is not None,
            ])
            components["signals"] = {
                "status": "up", "label": "Diagnostic signals",
                "detail": (f"{warm}/4 signals warm and ready" if warm
                           else "Ready - signals warm on first diagnosis"),
                "warm": int(warm), "total": 4,
            }
    except Exception as e:
        components["signals"] = {
            "status": "down", "label": "Diagnostic signals",
            "detail": f"Error - {e}"[:200],
        }

    # Overall = the worst single component.
    rank = {"down": 0, "degraded": 1, "up": 2}
    worst = min((rank.get(c["status"], 1) for c in components.values()),
                default=2)
    overall = {0: "down", 1: "degraded", 2: "operational"}[worst]

    now = _utcnow()
    return {
        "overall": overall,
        "components": components,
        "checked_at": now.isoformat(),
        "process_uptime_s": round((now - _PROCESS_START).total_seconds()),
        "check_latency_ms": round((_t.perf_counter() - t0) * 1000.0),
    }


def _parse_iso(s):
    """Parse an ISO-8601 timestamp string to a datetime, or None."""
    if not s:
        return None
    try:
        return _dtmod.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


# Two heartbeats are a "gap" if more than this many seconds apart. The loop
# ticks every 60s, so >90s means at least one beat was missed.
_GAP_THRESHOLD_S = 90


def _detect_heartbeat_gaps(recent, threshold_s=_GAP_THRESHOLD_S):
    """Find stretches of monitor silence in a newest-first heartbeat list.

    A gap longer than threshold_s means at least one 60s heartbeat was
    missed: the process crashed, was redeployed, or the Space went to sleep.
    The monitor cannot record its own downtime as a 'down' status (the
    heartbeat task dies with the process), so these gaps are the only way an
    in-app outage becomes visible at all. Returns a newest-first list of
    {start, end, seconds}.
    """
    rows = list(reversed(recent or []))   # oldest -> newest
    gaps = []
    for a, b in zip(rows, rows[1:]):
        ta = _parse_iso(a.get("recorded_at"))
        tb = _parse_iso(b.get("recorded_at"))
        if ta is None or tb is None:
            continue
        dt = (tb - ta).total_seconds()
        if dt > threshold_s:
            gaps.append({"start": a.get("recorded_at"),
                         "end":   b.get("recorded_at"),
                         "seconds": round(dt)})
    gaps.reverse()   # newest first
    return gaps


def _build_status_payload(app) -> dict:
    """Assemble the full JSON the /status page renders.

    Combines the live snapshot with the persisted 90-day history. Wrapped so
    that even a total DB outage yields a valid payload (empty history, live
    snapshot only) rather than a 500.
    """
    snap = _collect_status(app)
    days, recent, total_diag = [], [], 0
    try:
        from scripts.apin_v2 import auth_db
        days       = auth_db.get_status_days(90)
        # 48h of raw heartbeats - powers the pulse strip, the median-latency
        # KPI, and gap detection (the raw table is pruned to ~48h anyway).
        recent     = auth_db.get_recent_heartbeats(hours=48, limit=4000)
        total_diag = auth_db.count_all_predictions()
    except Exception as e:
        logger.warning(f"status payload: history unavailable - {e}")

    today = _utcnow().date()
    day_keys = [(today - _td_(days=i)).isoformat() for i in range(89, -1, -1)]
    day_map = {d.get("day"): d for d in days}
    comp_keys = [k for k, _ in _STATUS_COMPONENTS]

    # Per-component 90-day uptime bars + per-component uptime %.
    bars, uptime_pct = {}, {}
    for ck in comp_keys:
        cells, up_n, tot_n = [], 0, 0
        for dk in day_keys:
            row = day_map.get(dk)
            cd = ((row.get("components") if row else None) or {}).get(ck)
            if not cd:
                cells.append({"day": dk, "status": "nodata"})
                continue
            u = int(cd.get("up", 0))
            g = int(cd.get("deg", 0))
            d_ = int(cd.get("down", 0))
            t = u + g + d_
            up_n += u
            tot_n += t
            if t == 0:
                st = "nodata"
            elif d_ > 0:
                st = "down"
            elif g > 0:
                st = "degraded"
            else:
                st = "up"
            cells.append({"day": dk, "status": st, "samples": t})
        bars[ck] = cells
        uptime_pct[ck] = round(100.0 * up_n / tot_n, 3) if tot_n else None

    # Overall window uptime - fraction of ticks where everything was up.
    # Summed over EXACTLY the 90 displayed days (day_keys), not the slightly
    # wider set get_status_days() can return, so the KPI matches the bars.
    ov_op  = sum(int(day_map[dk].get("op_count", 0))
                 for dk in day_keys if dk in day_map)
    ov_tot = sum(int(day_map[dk].get("samples", 0))
                 for dk in day_keys if dk in day_map)
    overall_uptime = round(100.0 * ov_op / ov_tot, 3) if ov_tot else None

    # Median heartbeat latency from the raw recent rows.
    lat = sorted(int(h["response_ms"]) for h in recent
                 if h.get("response_ms") is not None)
    median_ms = (lat[len(lat) // 2] if lat else snap["check_latency_ms"])

    # Downtime log - every non-operational day, newest first. Bounded to
    # day_keys (the 90 displayed days) so it never lists an incident for a
    # day the uptime bars do not also show.
    _day_key_set = set(day_keys)
    downtime = [
        {"day": d.get("day"), "status": d.get("overall")}
        for d in reversed(days)
        if d.get("overall") and d.get("overall") != "operational"
        and d.get("day") in _day_key_set
    ][:14]

    # Monitor interruptions - gaps in the heartbeat stream over the last
    # 48h. Each one is a stretch where APIN was crashed, redeploying, or
    # the Space was asleep, so nothing was alive to record a heartbeat.
    interruptions = _detect_heartbeat_gaps(recent)[:14]

    # Recent pulse strip - last ~90 heartbeats oldest-first, with grey gap
    # markers inserted wherever the monitor fell silent.
    last90 = list(reversed(recent[:90]))   # oldest -> newest
    pulse = []
    prev_t = None
    for h in last90:
        t = _parse_iso(h.get("recorded_at"))
        if prev_t is not None and t is not None:
            dt = (t - prev_t).total_seconds()
            if dt > _GAP_THRESHOLD_S:
                pulse.append({"gap": True, "seconds": round(dt)})
        pulse.append({"t": h.get("recorded_at"),
                      "overall": h.get("overall")})
        if t is not None:
            prev_t = t

    try:
        import torch
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    except Exception:
        device = "unknown"

    components_list = []
    for ck, _label in _STATUS_COMPONENTS:
        c = dict(snap["components"].get(ck, {}))
        c["key"] = ck
        c["uptime_pct"] = uptime_pct.get(ck)
        c["bars"] = bars.get(ck, [])
        components_list.append(c)

    return {
        "overall": snap["overall"],
        "checked_at": snap["checked_at"],
        "process_uptime_s": snap["process_uptime_s"],
        "check_latency_ms": snap["check_latency_ms"],
        "components": components_list,
        "kpis": {
            "uptime_90d": overall_uptime,
            "total_diagnoses": total_diag,
            "median_latency_ms": median_ms,
            "device": device,
        },
        "downtime": downtime,
        "interruptions": interruptions,
        "pulse": pulse,
        "history_days": len(days),
        "poll_interval_s": 30,
    }


def _heartbeat_tick(app):
    """One monitoring sample: probe every component, persist the heartbeat.
    Hoisted to module scope (not redefined per loop iteration) and only ever
    run inside asyncio.to_thread, since record_heartbeat does blocking DB I/O."""
    from scripts.apin_v2 import auth_db
    snap = _collect_status(app)
    simple = {k: v.get("status", "up")
              for k, v in snap["components"].items()}
    auth_db.record_heartbeat(snap["overall"], simple, snap["check_latency_ms"])


async def _status_heartbeat_loop(app):
    """Record one heartbeat a minute, forever. Each tick is isolated so a
    transient DB failure logs and the loop keeps running."""
    await asyncio.sleep(20)   # let the process settle before the first sample
    while True:
        try:
            await asyncio.to_thread(_heartbeat_tick, app)
        except Exception as e:
            logger.warning(f"heartbeat tick failed: {e}")
        await asyncio.sleep(60)


def _ensure_heartbeat(app):
    """Idempotently start the heartbeat loop.

    Called both from the startup event AND lazily from the first /status/data
    request. The lazy path means the task still starts even if a future
    upstream change switches make_app() to a lifespan handler (which would
    silently disable on_event startup hooks). asyncio.create_task needs a
    running loop, so a too-early call is caught and simply retried later.
    """
    global _heartbeat_started
    if _heartbeat_started:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return   # no loop yet — a later call will succeed
    _heartbeat_started = True
    asyncio.create_task(_status_heartbeat_loop(app))
    logger.info("status heartbeat task started (60 s interval)")

    # Phase 8 Wave B: webhook delivery worker. Idempotent — only one start
    # per process. Skipped silently if APIN_SECRET_KEY is unset (the worker
    # needs to decrypt webhook secrets to sign payloads). The worker's stats
    # become available via `app.state.webhook_worker.stats()`.
    # (FX-P8-RUNTIME: this section runs inside _ensure_heartbeat where the
    # module-top `import os` is aliased as `_os` to avoid shadowing a local
    # var. Use _os here for the env lookup.)
    import os as _os_local
    if not getattr(app.state, "_webhook_worker_started", False):
        if _os_local.environ.get("APIN_SECRET_KEY"):
            try:
                from scripts.apin_v2.webhook_worker import WebhookWorker
                app.state.webhook_worker = WebhookWorker.start_in_background()
                app.state._webhook_worker_started = True
            except Exception as e:
                logger.warning("Webhook worker failed to start: %s", e)
        else:
            logger.warning(
                "Webhook delivery worker NOT started — APIN_SECRET_KEY env "
                "var is unset. Set it (see WI-P8-DELIVERY-WORKER docs) to "
                "enable webhook deliveries.")

    # Phase 9.A — usage telemetry flusher. Drains the in-memory request
    # buffer into api_key_request_log / api_key_usage_minute / api_keys
    # every ~2 s, runs an exact p50/p95/p99 rollup every 60 s. Has no
    # environment-variable dependency — always start.
    if not getattr(app.state, "_usage_flusher_started", False):
        try:
            from scripts.apin_v2.usage_recorder import UsageFlusher
            app.state.usage_flusher = UsageFlusher.start_in_background()
            app.state._usage_flusher_started = True
        except Exception as e:
            logger.warning("Usage flusher failed to start: %s", e)


def _add_status_routes(app):
    """Register /status + /status/data and arm the heartbeat task."""
    status_html = _load_html("status.html")

    @app.get("/status", response_class=HTMLResponse)
    async def v2_status_page():
        return HTMLResponse(status_html, headers=_PUBLIC_PAGE_HEADERS)

    @app.get("/status/data")
    async def v2_status_data():
        _ensure_heartbeat(app)   # lazy-start fallback if startup hook missed
        # Build off the event loop - component probes touch the DB.
        try:
            payload = await asyncio.to_thread(_build_status_payload, app)
        except Exception as e:
            logger.warning(f"/status/data failed: {e}")
            payload = {
                "overall": "degraded", "components": [],
                "checked_at": _utcnow().isoformat(),
                "kpis": {}, "downtime": [], "interruptions": [], "pulse": [],
                "error": "status snapshot unavailable",
                "poll_interval_s": 30,
            }
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.on_event("startup")
    async def _start_heartbeat():
        _ensure_heartbeat(app)

    logger.info(f"v2 status routes registered - /status + /status/data "
                f"(status.html {len(status_html):,} bytes)")


def _override_health_route(app):
    """Replace section8's JSON-only /health with a content-negotiated route.

    A browser (Accept: text/html) gets the visual health.html page; a machine
    - curl, the UptimeRobot monitor - gets JSON. The JSON keeps `status: ok`
    whenever the route runs, so the external monitor's keyword check is
    unaffected; component nuance rides in the `overall` + `components` fields.
    `?format=json` / `?format=html` force either branch.
    """
    from starlette.routing import Route
    removed = 0
    for i in range(len(app.router.routes) - 1, -1, -1):
        r = app.router.routes[i]
        if isinstance(r, Route) and r.path == "/health" and "GET" in r.methods:
            app.router.routes.pop(i)
            removed += 1

    health_html = _load_html("health.html")

    @app.get("/health")
    async def v2_health(request: Request):
        accept = (request.headers.get("accept") or "").lower()
        fmt = (request.query_params.get("format") or "").lower()
        want_html = (fmt != "json") and (fmt == "html" or "text/html" in accept)
        if want_html:
            return HTMLResponse(health_html, headers=_PUBLIC_PAGE_HEADERS)

        # JSON branch - superset of the original /health shape.
        try:
            import torch
            cuda = torch.cuda.is_available()
            vram = round((torch.cuda.memory_allocated() / 1e9) if cuda else 0, 2)
        except Exception:
            cuda, vram = False, 0.0
        snap = await asyncio.to_thread(_collect_status, app)
        comp = {k: {"status": v.get("status"), "label": v.get("label"),
                    "detail": v.get("detail")}
                for k, v in snap["components"].items()}
        okra = snap["components"].get("okra", {})
        return JSONResponse({
            "status": "ok",                       # external-monitor keyword
            "overall": snap["overall"],
            "device": "cuda" if cuda else "cpu",
            "gpu_vram_gb": vram,
            "version": "apin-1.0",
            "apin_loaded": okra.get("status") in ("up", "degraded"),
            "components": comp,
            "process_uptime_s": snap["process_uptime_s"],
            "check_latency_ms": snap["check_latency_ms"],
            "timestamp": snap["checked_at"],
        }, headers={"Cache-Control": "no-store"})

    logger.info(f"v2 '/health' route replaced (removed {removed} original) - "
                f"content-negotiated HTML/JSON (health.html "
                f"{len(health_html):,} bytes)")


# ════════════════════════════════════════════════════════════════════
# API v1 — contract-conforming machine endpoints (API_CONTRACT.md)
#
# Phase 1, batch 1: read-only reference endpoints. Purely additive new
# routes — they touch nothing existing. Every response uses the §2
# envelope via the @api_endpoint decorator; errors use ApiError (§3).
#   GET /version       service + model + runtime identity
#   GET /diseases      the full okra/brassica disease taxonomy (agronomy)
#   GET /diseases/{class}   one disease, full detail
#   GET /model/card    the published, honest model card
# ════════════════════════════════════════════════════════════════════

def _git_commit_short() -> str:
    """Best-effort short git commit hash; 'unknown' if unavailable."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(PROJECT_ROOT), capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _load_diagnosis_db() -> dict:
    """Load diagnosis/diagnosis_lookup.json (the agronomy reference). {} on
    failure — endpoints degrade honestly rather than crash."""
    try:
        import json as _json
        p = PROJECT_ROOT / "diagnosis" / "diagnosis_lookup.json"
        with open(p, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception as e:
        logger.warning(f"diagnosis_lookup.json unavailable: {e}")
        return {}


def _disease_record(cls: str, diag: dict) -> dict:
    """One rich disease record from a diagnosis-lookup entry."""
    e = diag.get(cls, {})
    crop = "okra" if cls.startswith("okra") else "brassica"
    return {
        "class": cls,
        "crop": crop,
        "label": e.get("full_name") or cls.replace("_", " ").title(),
        "is_healthy": cls.endswith("_healthy"),
        "pathogen_and_cause": e.get("cause"),
        "symptoms": e.get("symptoms"),
        "treatment": e.get("treatment"),          # {mild,moderate,severe}
        "prevention": e.get("prevention"),
        "urgency": e.get("urgency"),              # {mild,moderate,severe}
        "urgency_reason": e.get("urgency_reason"),
        "has_agronomy_detail": bool(e),
    }


def _add_reference_routes(app):
    """Register the Phase-1 read-only reference endpoints."""
    from scripts.apin_v2.api_envelope import api_endpoint, ApiError, API_VERSION
    from scripts.apin.constants import MODEL2_CLASS_ORDER
    import platform

    git_commit = _git_commit_short()
    diag_db = _load_diagnosis_db()
    classes = list(MODEL2_CLASS_ORDER)

    # ── GET /version ───────────────────────────────────────────────────
    @app.get("/api/version")
    @api_endpoint("/api/version")
    async def v1_version():
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "unknown"
        return {
            "service": "APIN - Adaptive Pathological Intelligence Network",
            "api_version": API_VERSION,
            "git_commit": git_commit,
            "model": {
                "name": "APIN okra/brassica 3-signal ensemble",
                "crop_scope": ["okra", "brassica"],
                "classes": classes,
                "n_classes": len(classes),
                "n_signals": 4,
                "backbone": "DINOv3-ConvNeXt-Small",
            },
            "runtime": {
                "device": device,
                "python": platform.python_version(),
            },
            "contract": "v1.0 (stable)",
        }

    # ── GET /diseases ──────────────────────────────────────────────────
    @app.get("/api/diseases")
    @api_endpoint("/api/diseases")
    async def v1_diseases():
        recs = [_disease_record(c, diag_db) for c in classes]
        meta = {}
        if not diag_db:
            meta["warnings"] = [{
                "code": "agronomy_detail_unavailable",
                "message": ("diagnosis_lookup.json could not be loaded; "
                            "disease records are present but agronomy "
                            "detail is null."),
            }]
        return ({
            "crop_scope": ["okra", "brassica"],
            "count": len(recs),
            "diseases": recs,
            "severity_levels": ["mild", "moderate", "severe"],
            "note": ("Treatment and urgency are keyed by severity. APIN "
                     "diagnoses these 9 okra/brassica classes; tomato is a "
                     "separate pipeline and chilli is not supported."),
        }, meta)

    # ── GET /diseases/{disease_class} ──────────────────────────────────
    @app.get("/api/diseases/{disease_class}")
    @api_endpoint("/api/diseases/{disease_class}")
    async def v1_disease_detail(disease_class: str):
        if disease_class not in classes:
            raise ApiError(
                "not_found",
                f"No APIN disease class named '{disease_class}'.",
                hint="Call GET /api/diseases for the list of valid classes.",
                field="disease_class")
        return _disease_record(disease_class, diag_db)

    # ── GET /model/card ────────────────────────────────────────────────
    @app.get("/api/model/card")
    @api_endpoint("/api/model/card")
    async def v1_model_card():
        return {
            "name": "APIN - Adaptive Pathological Intelligence Network",
            "model_version": "model2-convnext (okra/brassica specialist)",
            "git_commit": git_commit,
            "task": ("Multi-class leaf-disease classification for okra "
                     "and brassica from a single leaf photograph."),
            "classes": classes,
            "n_classes": len(classes),
            "architecture": {
                "type": ("3-signal stacking ensemble with a learned "
                         "MLP gate"),
                "signals": [
                    {"name": "model2", "backbone": "DINOv3-ConvNeXt-Small",
                     "role": "primary disease classifier"},
                    {"name": "efficientnet", "backbone": "EfficientNet",
                     "role": "secondary classifier signal"},
                    {"name": "dinov2",
                     "backbone": "DINOv2 ViT (frozen) + linear head",
                     "role": "representation signal"},
                    {"name": "psv",
                     "backbone": "engineered Plant-Signal-Vector features",
                     "role": "feature-based signal"},
                ],
                "fusion": "stacking MLP over per-signal class distributions",
                "calibration": ("per-class temperature scaling + "
                                "split-conformal prediction sets"),
                "ood_detection": ("Mahalanobis-distance detector over "
                                  "per-class prototypes"),
                "confidence_tiers": ("tiered output 1A..5 with abstention "
                                     "on low-confidence / out-of-distribution "
                                     "inputs"),
                "explainability": "Grad-CAM++ heatmap on the top signal",
            },
            "input": {
                "format": "single RGB leaf photograph",
                "preprocessing": ("Gate-Zero quality check -> LAB-CLAHE "
                                  "-> resize to 224x224"),
                "min_resolution_px": 100,
            },
            "evaluation": {
                "test_set": ("1,379 held-out okra/brassica images, "
                             "content-hash verified clean of every "
                             "training pool"),
                "details_endpoint": "/api/benchmarks",
            },
            "limitations": [
                ("Diagnoses okra and brassica leaves only. Tomato is "
                 "handled by a separate pipeline; chilli is not supported."),
                ("Single-leaf close-up images only - not whole plants, "
                 "stems, roots, pods, or wide field shots."),
                ("Trained predominantly on cultivated-dataset imagery; "
                 "accuracy under unusual field conditions can be lower."),
                ("An independent benchmark found a competently fine-tuned "
                 "single model matches this ensemble on raw accuracy; the "
                 "ensemble's measured advantage is in calibration and "
                 "uncertainty quantification, not peak accuracy."),
            ],
            "intended_use": ("Decision support for farmers and "
                             "agronomists."),
            "not_a_substitute_for": ("professional agronomic or "
                                     "plant-pathology diagnosis"),
            "contract": "v1.0 (stable)",
        }

    logger.info("APIN /api/ reference routes registered - /api/version "
                "/api/diseases /api/diseases/{class} /api/model/card")


# ════════════════════════════════════════════════════════════════════════
# API v1 action routes — Phase 1 batch 2
# ════════════════════════════════════════════════════════════════════════
# Adds the contract-conforming machine endpoints that DO work (predict,
# warmup, benchmark, key management), as opposed to the read-only
# reference endpoints in _add_reference_routes.
#
# Every route here is purely additive: nothing existing is removed or
# patched. The existing browser-facing /predict/full route is untouched
# and continues to serve the inference website exactly as before. The
# new /predict/quick and /predict/batch routes are at fresh paths and
# carry their own auth scheme (API-key Bearer tokens), so they cannot
# collide with the cookie-session auth used by the website.

def _load_latest_benchmark() -> tuple[Optional[dict], Optional[str]]:
    """Find the newest apin_vs_baselines_final_*.json report on disk and
    return (parsed_dict, filename). Returns (None, None) if no report
    has been generated yet. Never raises; on parse error returns (None,
    filename) so the endpoint can explain what went wrong without leaking
    a Python traceback to the caller."""
    import glob
    pattern = str(PROJECT_ROOT / "reports" / "apin_vs_baselines_final_*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        return None, None
    latest = files[-1]
    fname  = Path(latest).name
    try:
        with open(latest, encoding="utf-8") as f:
            return json.load(f), fname
    except Exception as e:
        logger.warning("Failed to parse %s: %s", latest, e)
        return None, fname


def _benchmark_summary(report: dict) -> dict:
    """Compress the (large) finalized comparison into a headline summary
    the API can return without the per-class breakdown drowning the
    payload. The full per-class detail is also returned in the parent
    `comparison` block so analysts can drill in."""
    apin = report.get("apin_ensemble", {}) or {}
    v2   = apin.get("v2_gate_diagnosed_subset", {}) or {}
    bases = report.get("baselines_on_apin_v2_subset_1195", {}) or {}

    best_base_name, best_base_f1 = None, -1.0
    for name, m in bases.items():
        f1 = float(m.get("macro_f1", 0.0))
        if f1 > best_base_f1:
            best_base_name, best_base_f1 = name, f1

    return {
        "primary_metric": "macro_f1",
        "test_set_size":  int(report.get("test_images", 0)),
        "apin_ensemble": {
            "n_diagnosed":     int(v2.get("n", 0)),
            "accuracy":        round(float(v2.get("accuracy", 0.0)), 4),
            "macro_f1":        round(float(v2.get("macro_f1", 0.0)), 4),
            "ece":             round(float(v2.get("ece", 0.0)), 4),
            "gate":            "v2 (resolution-invariant blur)",
            "strict_accuracy_full_1379": round(float(apin.get(
                "strict_accuracy_all_1379", {}).get("v2_gate", 0.0)), 4),
        },
        "best_external_baseline": {
            "model":    best_base_name,
            "macro_f1": round(best_base_f1, 4) if best_base_name else None,
            "ece":      round(float(bases.get(best_base_name, {}).get(
                            "ece", 0.0)), 4) if best_base_name else None,
        } if best_base_name else None,
        "honest_finding": (
            "On raw macro-F1 a competently fine-tuned single model "
            "(EfficientNet-B0) matches the APIN ensemble. APIN's "
            "measured advantage on this test set is calibration: ECE "
            "is roughly half that of the best baseline, which means "
            "the confidence numbers APIN returns are far closer to "
            "true accuracy than a single softmax model's are."
        ),
    }


def _add_v1_action_routes(app):
    """Register the Phase-1 batch-2 action endpoints:
        GET    /api/benchmarks       (public)
        POST   /api/warmup           (public, idempotent)
        POST   /api/predict/quick    (API-key auth)
        POST   /api/predict/batch    (API-key auth)
        POST   /api/keys             (session-cookie auth)
        GET    /api/keys             (session-cookie auth)
        DELETE /api/keys/{key_id}    (session-cookie auth)
    """
    import io
    from dataclasses import asdict
    from PIL import Image as _PIL_Image
    import numpy as np
    from fastapi import Header
    from typing import List, Optional as _Opt
    from scripts.apin_v2.api_envelope import (
        api_endpoint, ApiError, API_VERSION, paginated)
    from scripts.apin_v2 import auth_db as _auth_db
    from scripts.apin_v2.auth_routes import COOKIE_NAME as _SESSION_COOKIE

    # ── helper: parse a multipart UploadFile into an RGB numpy array ──
    _BATCH_MAX             = 16          # max files per /predict/batch
    _PER_FILE_MAX_BYTES    = 12_000_000  # per-image cap (~12 MB)
    # PDA F2: aggregate-size cap so 16 × 12 MB cannot OOM the 512 MB
    # Space dyno. Empirically the dyno tolerates ~32 MB of in-flight
    # batch upload while serving inference; tune downward if observed.
    _BATCH_TOTAL_MAX_BYTES = 32_000_000  # ~32 MB across all files

    def _read_image(data: bytes,
                    *, max_bytes: int = _PER_FILE_MAX_BYTES) -> np.ndarray:
        """Decode bytes to RGB uint8 ndarray. Raises ApiError on bad input.

        PDA F8: we deliberately do NOT surface the raw decoder exception
        string in `details`. Pillow's exceptions can contain filesystem
        tmpfile paths and internal struct offsets that would violate §8's
        "no internal detail" rule. The error class name is benign and
        sufficient to diagnose; the request_id ties it to server logs.
        """
        if not data:
            raise ApiError("invalid_parameter",
                           "Image file is empty.",
                           hint="Upload a non-empty JPEG or PNG image.",
                           field="file")
        if len(data) > max_bytes:
            raise ApiError(
                "payload_too_large",
                f"Image exceeds the {max_bytes // 1_000_000} MB per-file "
                f"limit (got {len(data) // 1_000_000} MB).",
                hint="Re-encode the image at lower quality, or downscale "
                     "the longest side to ~2048 px.",
                field="file",
                details={"per_file_limit_bytes": max_bytes,
                         "received_bytes": len(data)})
        try:
            img = _PIL_Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            # PDA §8: decoder error class logged, not returned.
            logger.exception("_read_image: image decode failed")
            raise ApiError(
                "unsupported_media_type",
                "Could not decode the uploaded file as an image.",
                hint="Send a JPEG, PNG, or WebP image.")
        return np.array(img, dtype=np.uint8)

    def _convert_np(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        if isinstance(o, dict): return {k: _convert_np(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert_np(v) for v in o]
        return o

    def _lean_predict(img_rgb: np.ndarray) -> dict:
        """Run the full APIN ensemble and project the 29-field APINResult
        down to the lean machine-API shape. We DON'T re-implement
        inference — every byte of math is the same as /predict/full. We
        just drop the heavy explainability fields (Grad-CAM, decision
        trace, pipeline visualisations) that machine consumers rarely
        want and that bloat the response by ~150 KB."""
        from scripts.apin.section8_apin_server import get_apin
        apin_engine = get_apin()
        result = apin_engine.predict(img_rgb)
        try:
            out = asdict(result)
        except Exception:
            out = {k: v for k, v in vars(result).items()
                   if not k.startswith("_")}
        out = _convert_np(out)

        # Top-3 from all_class_probabilities, sorted descending.
        probs = out.get("all_class_probabilities") or {}
        top3 = [{"class": k, "probability": round(float(v), 4)}
                for k, v in sorted(probs.items(),
                                    key=lambda kv: -float(kv[1]))[:3]]

        return {
            "diagnosis":        out.get("diagnosis"),
            "confidence":       round(float(out.get("confidence", 0.0)), 4),
            "tier":             out.get("tier"),
            "top3":             top3,
            "out_of_distribution": bool(out.get("is_ood", False)),
            "mahalanobis_distance": round(float(
                out.get("mahalanobis_distance", 0.0)), 4),
            "uncertainty": {
                "aleatoric": round(float(
                    out.get("uncertainty_aleatoric", 0.0)), 4),
                "epistemic": round(float(
                    out.get("uncertainty_epistemic", 0.0)), 4),
            },
            "conformal_prediction_set": list(
                out.get("conformal_prediction_set") or []),
            "quality_flags":  out.get("quality_flags") or {},
            "failed_signals": list(out.get("failed_signals") or []),
            "processing_time_ms": round(float(
                out.get("processing_time_ms", 0.0)), 1),
            "mode": "quick",
            "explainability_available_at": "/api/predict/full",
        }

    # ── helper: require an API key ────────────────────────────────────
    def _require_api_key(authorization: _Opt[str],
                        x_api_key: _Opt[str]) -> dict:
        """Resolve a key from either `Authorization: Bearer <token>` or
        `X-API-Key: <token>`. Raises ApiError on missing/invalid.
        Returns the auth record from auth_db.find_api_key."""
        raw = None
        if authorization and authorization.lower().startswith("bearer "):
            raw = authorization[7:].strip()
        elif x_api_key:
            raw = x_api_key.strip()
        if not raw:
            raise ApiError(
                "auth_required",
                "This endpoint requires an API key.",
                hint=("Send Authorization: Bearer <token> or "
                      "X-API-Key: <token>. Mint a key at POST /api/keys "
                      "after signing in to the dashboard."))
        auth = _auth_db.find_api_key(raw)
        if auth is None:
            raise ApiError(
                "auth_invalid",
                "The provided API key is not valid or has been revoked.",
                hint="List your active keys with GET /api/keys; mint a new "
                     "one with POST /api/keys.")
        return auth

    # ── helper: require a logged-in user (cookie session) ─────────────
    def _require_session_user(request: Request) -> dict:
        try:
            tok = request.cookies.get(_SESSION_COOKIE)
        except Exception:
            tok = None
        if not tok:
            raise ApiError(
                "auth_required",
                "Sign in to manage API keys.",
                hint="POST /auth/login or open the inference site and "
                     "sign in, then retry this request from the same "
                     "browser session.")
        u = _auth_db.get_session_user(tok)
        if not u:
            raise ApiError(
                "auth_expired",
                "Your session has expired or been revoked.",
                hint="Sign in again to refresh your session cookie.")
        return u

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ GET /benchmarks                                              ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.get("/api/benchmarks")
    @api_endpoint("/api/benchmarks")
    async def v1_benchmarks():
        report, fname = _load_latest_benchmark()
        if report is None and fname is None:
            return ({
                "available": False,
                "message": ("No benchmark report has been generated yet. "
                            "Run _qa_tmp/rescore_baselines.py to produce "
                            "reports/apin_vs_baselines_final_*.json, then "
                            "retry."),
            }, {"warnings": [{
                "code": "benchmark_report_missing",
                "message": "No apin_vs_baselines_final_*.json found.",
            }]})
        if report is None:
            raise ApiError(
                "internal_error",
                f"Latest benchmark report '{fname}' is unreadable.",
                hint="Server logs hold the parser error; the file may "
                     "have been truncated by an interrupted writer. "
                     "Regenerate it with _qa_tmp/rescore_baselines.py.",
                details={"report_file": fname})

        # PDA F7: if the JSON is structurally valid but a field has
        # an unexpected type (e.g. macro_f1 is null instead of a float),
        # _benchmark_summary's float() casts would raise. Catch here so
        # the caller gets a clean internal_error pointing at the offending
        # file, not an opaque 500 with no clue what to fix.
        try:
            summary = _benchmark_summary(report)
        except Exception as e:
            logger.exception(
                "Benchmark summary failed for %s", fname)
            raise ApiError(
                "internal_error",
                f"Benchmark report '{fname}' parsed but is semantically "
                "malformed (an expected numeric field is missing or "
                "has the wrong type).",
                hint="Regenerate the report with "
                     "_qa_tmp/rescore_baselines.py; if the failure "
                     "persists open a support ticket with this "
                     "request_id.",
                # PDA §8: report_file is safe (it IS the user-visible
                # filename); exception class is NOT — logger captured
                # the class server-side already.
                details={"report_file": fname})

        return {
            "available":     True,
            "report_file":   fname,
            "generated_at":  report.get("generated_at"),
            "test_set": {
                "n":                int(report.get("test_images", 0)),
                "integrity":        "content-hash verified clean of every "
                                    "training pool",
                "locked":           True,
                "locked_rationale": ("Test split is frozen so every "
                                     "subsequent model change is measured "
                                     "on identical images, eliminating "
                                     "rubric drift."),
            },
            "summary":       summary,
            "comparison": {
                "apin_ensemble":              report.get("apin_ensemble") or {},
                "baselines_on_apin_v2_subset":
                    report.get("baselines_on_apin_v2_subset_1195") or {},
                "baselines_on_apin_v1_subset":
                    report.get("baselines_on_apin_v1_subset_719") or {},
                "baselines_full_1379":        report.get("baselines_full_1379")
                                                or {},
            },
            "methodology": {
                "metrics":        ["accuracy", "macro_f1", "ece (15-bin)"],
                "apples_to_apples": ("APIN and every baseline are scored "
                                     "on the SAME images. APIN's "
                                     "diagnosed-subset numbers compare "
                                     "only on the images APIN's quality "
                                     "gate accepted; the strict_accuracy "
                                     "number compares on all 1,379 with "
                                     "abstentions counted as errors."),
                "gates_compared": ["v1 (Laplacian-variance blur)",
                                   "v2 (resolution-invariant blur)"],
            },
            "interpretation": (
                "Read the summary's `honest_finding` before quoting any "
                "single number from this report. Cherry-picking either "
                "v1-gate macro-F1 OR baseline-on-full-set accuracy in "
                "isolation will misrepresent what the comparison says."
            ),
            "contract": "v1.0 (stable)",
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /warmup — idempotent trigger                            ║
    # ╚══════════════════════════════════════════════════════════════╝
    # PDA F5: lazy loaders are CPU+GPU heavy and were running on the
    # event loop, blocking everything else during a cold start. They
    # now run in the default executor, and a process-wide cooldown
    # prevents a tight POST /warmup loop from thrashing the loaders.
    import time as _time
    _warmup_state = {"last_run_ts": 0.0,
                     "cooldown_s":  10.0,
                     "lock":        asyncio.Lock()}

    def _do_warmup_blocking() -> list[dict]:
        """Runs in an executor thread; never touches the event loop."""
        from scripts.apin.section8_apin_server import get_apin
        out: list[dict] = []
        a = get_apin()
        for label, fn in (
            ("model2",       getattr(a, "_lazy_load_model2",       None)),
            ("efficientnet", getattr(a, "_lazy_load_efficientnet", None)),
            ("dinov2",       getattr(a, "_lazy_load_dinov2",       None)),
        ):
            if fn is None:
                out.append({"signal": label, "status": "missing_loader"})
                continue
            try:
                fn()
                out.append({"signal": label, "status": "ready"})
            except Exception as e:
                # Internal log only; nothing sensitive leaks to the API
                # response (we set status="load_failed" without an
                # exception name on the wire).
                logger.warning("warmup: %s lazy-load failed: %r", label, e)
                out.append({"signal": label, "status": "load_failed"})
        return out

    @app.post("/api/warmup")
    @api_endpoint("/api/warmup", success_status=202)
    async def v1_warmup():
        """Trigger lazy-loading of every APIN signal. Idempotent and
        executor-backed: the heavy load runs in a worker thread so the
        event loop continues to serve `/health`, `/status`, etc. while
        models warm up.

        A short cooldown rate-limits concurrent callers: if the previous
        call was less than _warmup_state['cooldown_s'] seconds ago and
        another call is already in flight, the new caller is fast-pathed
        with `coalesced: True` instead of triggering a second load.
        Authoritative readiness still comes from GET /warmup/status."""
        loop = asyncio.get_running_loop()
        # Coalesce: if a warmup is already running, just wait for it to
        # finish (the lock serialises) rather than queueing more loads.
        async with _warmup_state["lock"]:
            now = _time.monotonic()
            since_last = now - _warmup_state["last_run_ts"]
            if since_last < _warmup_state["cooldown_s"]:
                return {
                    "accepted": True,
                    "coalesced": True,
                    "signals":   [],
                    "cooldown_seconds": _warmup_state["cooldown_s"],
                    "seconds_since_last_run": round(since_last, 2),
                    "check_status_at": "GET /warmup/status",
                    "note": ("Cooldown is active: a warmup ran "
                             "recently. Calling /warmup/status will "
                             "give you the authoritative ready/not-ready "
                             "state."),
                }
            try:
                actions = await loop.run_in_executor(
                    None, _do_warmup_blocking)
            except Exception:
                # PDA §8: exception class logged, not returned.
                logger.exception("warmup: APIN engine construction failed")
                raise ApiError(
                    "service_unavailable",
                    "APIN engine could not be constructed.",
                    hint="Check server logs; the inference pipeline may be "
                         "missing model weights on disk.")
            _warmup_state["last_run_ts"] = _time.monotonic()

        return {
            "accepted":  True,
            "coalesced": False,
            "signals":   actions,
            "check_status_at": "GET /warmup/status",
            "note": ("Warmup is idempotent. /warmup/status is the "
                     "authoritative readiness probe; this endpoint "
                     "only kicks off any not-yet-warm signal."),
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /predict/quick — single image, lean payload             ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.post("/api/predict/quick")
    @api_endpoint("/api/predict/quick")
    async def v1_predict_quick(
        file: UploadFile = File(...),
        # NB: use `str | None`, NOT `Optional[str]` and NOT the local
        # `_Opt[str]` alias. This module sits under
        # `from __future__ import annotations`, so pydantic re-evaluates
        # the annotation against module globals — where any local typing
        # alias is invisible (PydanticUserError: not fully defined). The
        # builtin union form resolves cleanly.
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        auth = _require_api_key(authorization, x_api_key)
        try:
            data = await file.read()
        except Exception as e:
            raise ApiError("invalid_parameter",
                           "Could not read the uploaded file.",
                           details={"error": str(e)[:160]})
        img = _read_image(data)
        loop = asyncio.get_running_loop()
        # APIN inference is CPU+GPU bound; offload from the event loop.
        result = await loop.run_in_executor(None, lambda: _lean_predict(img))
        result["authenticated_as"] = {
            "key_id":   auth["key_id"],
            "key_name": auth["name"],
        }
        return result

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /predict/batch — up to 16 images per call               ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.post("/api/predict/batch")
    @api_endpoint("/api/predict/batch")
    async def v1_predict_batch(
        # `list[UploadFile]`, NOT `List[UploadFile]` — same reason as the
        # `str | None` headers above: `from __future__ import annotations`
        # stringifies the annotation and pydantic evaluates it against the
        # module globals, where `typing.List` is not imported. The PEP-585
        # builtin generic resolves cleanly without any import.
        files: list[UploadFile] = File(...),
        # NB: use `str | None`, NOT `Optional[str]` and NOT the local
        # `_Opt[str]` alias. This module sits under
        # `from __future__ import annotations`, so pydantic re-evaluates
        # the annotation against module globals — where any local typing
        # alias is invisible (PydanticUserError: not fully defined). The
        # builtin union form resolves cleanly.
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        auth = _require_api_key(authorization, x_api_key)
        if not files:
            raise ApiError("missing_parameter",
                           "Send at least one image in `files`.",
                           field="files")
        if len(files) > _BATCH_MAX:
            raise ApiError(
                "payload_too_large",
                f"Batch is limited to {_BATCH_MAX} images per request; "
                f"got {len(files)}.",
                hint=f"Split your job into batches of <= {_BATCH_MAX} "
                     "and call /api/predict/batch repeatedly.",
                field="files",
                details={"limit": _BATCH_MAX, "received": len(files)})

        loop = asyncio.get_running_loop()
        items = []
        n_ok = n_err = 0
        # PDA F2: track cumulative bytes across the batch; reject when the
        # next file would push us past _BATCH_TOTAL_MAX_BYTES. We reject
        # rather than truncate so the caller knows their batch did not run
        # in full — silent truncation is worse than a clean failure.
        total_bytes = 0
        for i, f in enumerate(files):
            entry = {"index": i, "filename": f.filename}
            try:
                data = await f.read()
                total_bytes += len(data)
                if total_bytes > _BATCH_TOTAL_MAX_BYTES:
                    raise ApiError(
                        "payload_too_large",
                        ("Total batch upload exceeds the "
                         f"{_BATCH_TOTAL_MAX_BYTES // 1_000_000} MB "
                         "aggregate limit."),
                        hint=("Split the batch into smaller chunks (try "
                              f"<= {_BATCH_TOTAL_MAX_BYTES // 1_000_000} "
                              "MB total per request)."),
                        field="files",
                        details={"aggregate_limit_bytes":
                                    _BATCH_TOTAL_MAX_BYTES,
                                 "received_so_far_bytes": total_bytes,
                                 "stopped_at_index": i})
                img  = _read_image(data)
                r    = await loop.run_in_executor(
                    None, lambda im=img: _lean_predict(im))
                entry.update({"ok": True, "result": r})
                n_ok += 1
            except ApiError as e:
                entry.update({"ok": False,
                              "error": {"code": e.code, "message": e.message,
                                        "hint": e.hint}})
                n_err += 1
                # If we hit the aggregate cap, stop processing further
                # files — they would all fail for the same reason.
                if e.code == "payload_too_large":
                    items.append(entry)
                    break
            except Exception as e:
                # PDA F4: do NOT surface the python exception class to the
                # caller (§8 forbids internal detail). Log it under the
                # request_id; the caller gets a stable error contract.
                logger.exception(
                    "predict/batch inference failed on index %d", i)
                entry.update({"ok": False,
                              "error": {"code": "inference_failed",
                                        "message": ("Inference failed for "
                                                    "this image."),
                                        "hint": "Retry; if it persists, "
                                                "open a support ticket "
                                                "with the request_id."}})
                n_err += 1

        # Single-page (no paging) since the batch is hard-capped at 16.
        return paginated(
            items, page=1, page_size=len(items), total=len(items),
            ok_count=n_ok, error_count=n_err,
            authenticated_as={"key_id": auth["key_id"],
                              "key_name": auth["name"]},
        )

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /keys — mint a new API key                              ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.post("/api/keys")
    @api_endpoint("/api/keys", success_status=201)
    async def v1_keys_create(request: Request):
        u = _require_session_user(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        name = (body or {}).get("name")
        if not isinstance(name, str) or not name.strip():
            raise ApiError(
                "missing_parameter",
                "Provide a human-readable name for the key.",
                hint='Send JSON like {"name": "drone-fleet-prod"}.',
                field="name")
        try:
            k = _auth_db.create_api_key(u["id"], name.strip())
        except ValueError as e:
            raise ApiError("invalid_parameter", str(e), field="name")

        return ({
            "id":           k["id"],
            "name":         k["name"],
            "token":        k["token"],
            "token_prefix": k["token_prefix"],
            "created_at":   k["created_at"],
            "owner": {
                "user_id":      u["id"],
                "display_name": u.get("display_name"),
            },
            "usage_examples": {
                "curl_bearer": (
                    f"curl -H 'Authorization: Bearer {k['token']}' "
                    "-F 'file=@leaf.jpg' "
                    "https://dxv-404-apin.hf.space/api/predict/quick"
                ),
                "curl_x_api_key": (
                    f"curl -H 'X-API-Key: {k['token']}' "
                    "-F 'file=@leaf.jpg' "
                    "https://dxv-404-apin.hf.space/api/predict/quick"
                ),
            },
            "warning": ("Store this `token` now. It is shown ONCE and "
                        "is not recoverable. The dashboard shows only "
                        "the prefix from now on."),
        }, {"warnings": [{
            "code": "token_one_time_display",
            "message": "Raw token is shown ONCE on creation.",
        }]})

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ GET /keys — list the caller's active API keys                ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.get("/api/keys")
    @api_endpoint("/api/keys")
    async def v1_keys_list(request: Request):
        u = _require_session_user(request)
        rows = _auth_db.list_api_keys(u["id"], include_revoked=False)
        # PDA F6: list endpoints must use the §6 paginated shape so SDKs
        # that generalise across our list endpoints don't need a special
        # case for /keys. `owner` and `note` ride along via paginated()'s
        # **extra and appear in `data` next to `items` and `pagination`.
        return paginated(
            rows,
            page=1,
            page_size=len(rows) if rows else 1,
            total=len(rows),
            owner={"user_id":      u["id"],
                   "display_name": u.get("display_name")},
            note=("Raw tokens are only ever shown on POST /api/keys. If "
                  "you've lost a token, revoke this key and mint a "
                  "new one."),
        )

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ DELETE /keys/{key_id} — revoke a key                         ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.delete("/api/keys/{key_id}")
    @api_endpoint("/api/keys/{key_id}")
    async def v1_keys_revoke(key_id: int, request: Request):
        u = _require_session_user(request)
        ok = _auth_db.revoke_api_key(u["id"], int(key_id))
        if not ok:
            raise ApiError(
                "not_found",
                f"No active key with id {key_id} found on your account.",
                hint="GET /api/keys to see your active keys.",
                field="key_id")
        from datetime import datetime as _now_dt, timezone as _now_tz
        return {
            "id":         int(key_id),
            "revoked":    True,
            "revoked_at": _now_dt.now(_now_tz.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z"),
            "note": ("Existing requests using this key will continue to "
                     "complete; new requests authenticate as failed "
                     "immediately."),
        }

    logger.info("APIN /api/ action routes registered - /api/benchmarks "
                "/api/warmup /api/predict/quick /api/predict/batch "
                "/api/keys (3)")


def _add_v1_mirror_routes(app):
    """Register the Phase-1 batch-3 mirror endpoints under /api/...

    Every endpoint here is the §2/§3 envelope-wrapped twin of an
    EXISTING legacy endpoint. The legacy endpoints are NOT touched, so
    the inference website (which reads unwrapped fields directly off
    /predict/full and /apin/info) keeps working byte-for-byte.

    Machine consumers should prefer the /api/... paths — they get a
    stable shape, a request_id they can quote in support tickets, and
    error envelopes they can branch on programmatically.

        GET    /api/info       (envelope-wrap of /apin/info)
        GET    /api/feedback/stats  (envelope-wrap of /feedback/stats)
        GET    /api/retrain/history (envelope-wrap of /feedback/retrain/history)
        POST   /api/feedback        (envelope-wrap of /feedback; Bearer optional)
        POST   /api/predict/full    (rich-payload inference, Bearer required)
    """
    import io
    from dataclasses import asdict
    from PIL import Image as _PIL_Image
    import numpy as np
    from fastapi import Body, Header
    from scripts.apin_v2.api_envelope import api_endpoint, ApiError
    from scripts.apin_v2 import auth_db as _auth_db

    # ── inlined helpers (closure-local in section8, not importable) ──
    # The legacy /feedback, /feedback/stats, and /feedback/retrain/history
    # use validators + path constants that live inside make_app() in
    # scripts/apin/section8_apin_server.py — closure scope, not module
    # scope. We can NOT touch section8 (write-locked). So we re-derive
    # the small handful of constants/functions we need from authoritative
    # sources:
    #   _VALID_CLASSES         ← scripts.apin.constants.MODEL2_CLASS_ORDER
    #   RETRAIN_LOG, _MIN_S    ← APIN_DIR / "retrain_history.jsonl" + 6h
    #   _validate_feedback     ← byte-identical copy of section8 line 410
    #   _recency_weight        ← byte-identical copy of section8 line 437
    # If section8 ever changes its validation logic, this copy will drift
    # — the test suite asserts the same input → same outcome on both
    # endpoints to catch the drift loudly.
    from scripts.apin.constants import MODEL2_CLASS_ORDER as _CANON_CLASSES
    from scripts.apin.section8_apin_server import (
        FEEDBACK_LOG, APIN_DIR)
    _VALID_CLASSES_V1 = set(_CANON_CLASSES)
    _RETRAIN_LOG_V1   = APIN_DIR / "retrain_history.jsonl"
    _RETRAIN_MIN_S_V1 = 6 * 3600  # 6 hours; matches section8 line 489
    _FEEDBACK_RECENCY_BUCKETS_V1 = [
        (7,     5.0),   # 0-7 days  : 5x
        (30,    3.0),   # 8-30 days : 3x
        (90,    1.5),   # 31-90 days: 1.5x
        (10**9, 1.0),   # older     : 1x
    ]

    def _validate_feedback_v1(payload: dict) -> tuple:
        """Byte-identical to section8's _validate_feedback (line 410)."""
        if not isinstance(payload, dict):
            return False, "payload must be a JSON object"
        if "is_correct" not in payload:
            return False, "missing required field: is_correct (bool)"
        if not isinstance(payload["is_correct"], bool):
            return False, "is_correct must be boolean"
        if not payload["is_correct"]:
            cc = payload.get("correct_class")
            if not cc:
                return False, ("correct_class required when "
                               "is_correct=false")
            if cc not in _VALID_CLASSES_V1:
                return False, (
                    f"correct_class '{cc}' not in canonical class set "
                    f"(must be one of: {sorted(_VALID_CLASSES_V1)})")
            predicted = payload.get("predicted_class")
            if predicted == cc:
                return False, (
                    "predicted_class matches correct_class but "
                    "is_correct=false — ambiguous feedback, rejected")
        return True, ""

    def _recency_weight_v1(timestamp_iso: str) -> float:
        """Byte-identical to section8's _recency_weight (line 437)."""
        from datetime import datetime as _rw_dt, timezone as _rw_tz
        try:
            t = _rw_dt.fromisoformat(
                timestamp_iso.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=_rw_tz.utc)
            now = _rw_dt.now(_rw_tz.utc)
            days = max(0.0, (now - t).total_seconds() / 86400.0)
        except Exception:
            return 1.0
        for cutoff, w in _FEEDBACK_RECENCY_BUCKETS_V1:
            if days <= cutoff:
                return w
        return 1.0

    # Reuse the same Bearer-auth gate as the batch-2 endpoints. We
    # duplicate the closure here rather than reach into _add_v1_action_routes
    # so the two registrars stay decoupled.
    def _require_api_key(authorization: str | None,
                          x_api_key:     str | None) -> dict:
        raw = None
        if authorization and authorization.lower().startswith("bearer "):
            raw = authorization[7:].strip()
        elif x_api_key:
            raw = x_api_key.strip()
        if not raw:
            raise ApiError(
                "auth_required",
                "This endpoint requires an API key.",
                hint=("Send Authorization: Bearer <token> or "
                      "X-API-Key: <token>. Mint a key at POST /api/keys."))
        auth = _auth_db.find_api_key(raw)
        if auth is None:
            raise ApiError(
                "auth_invalid",
                "The provided API key is not valid or has been revoked.",
                hint="List your active keys with GET /api/keys; mint a new "
                     "one with POST /api/keys.")
        return auth

    def _convert_np(o):
        if isinstance(o, np.ndarray):              return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)):     return int(o)
        if isinstance(o, dict):
            return {k: _convert_np(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_convert_np(v) for v in o]
        return o

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ GET /api/info                                            ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.get("/api/info")
    @api_endpoint("/api/info")
    async def v1_apin_info():
        try:
            from scripts.apin.section8_apin_server import get_apin
            a = get_apin()
            return {
                "n_signals":              int(a.n_signals),
                "use_psv":                bool(a.use_psv),
                "class_order":            list(a.class_order),
                # NB: stacking_mlp_gate_mean is a LIST of per-class
                # means (one entry per class), not a single scalar. We
                # mirror the legacy /apin/info shape exactly so machine
                # consumers that already parse the legacy can lift to
                # v1 without changing their type assumptions.
                "stacking_mlp_gate_mean": _convert_np(
                    a.stacking_mlp_gate_mean),
                # PDA: consistent _convert_np for every numpy attribute.
                # Direct .tolist() works today but raises AttributeError
                # if the attribute ever arrives as a plain Python list
                # (which is exactly the bug class that the v1.3 cold-start
                # test caught on stacking_mlp_gate_mean).
                "reliability_matrix":     _convert_np(a.reliability_matrix),
                "cold_start_active":      bool(a.cold_start_active),
                "per_class_temperatures": _convert_np(a.per_class_temps),
                "conformal_thresholds":   _convert_np(a.conformal_thresholds),
                "what_these_mean": {
                    "n_signals":              "How many independent signals fuse into the diagnosis (4 in steady state).",
                    "stacking_mlp_gate_mean": "Per-class average gate weights across the validation set; closer to 1.0 = the gate is confident in fusing all signals for that class.",
                    "reliability_matrix":     "Per-signal per-class reliability scores used as priors by the stacking gate.",
                    "cold_start_active":      "True until the validation buffer fills; while True, tiers downgrade conservatively.",
                    "per_class_temperatures": "Temperature-scaling values applied to each class's logits to calibrate confidence.",
                    "conformal_thresholds":   "Per-class quantiles used to build the conformal prediction set at the contracted miscoverage rate.",
                },
                "legacy_endpoint":  "/apin/info",
                "contract":         "v1.0 (stable)",
            }
        except Exception:
            # PDA §8: do NOT surface exception class name to caller.
            # logger.exception() already captures it server-side under
            # the same request_id (api_envelope decorator binds it).
            logger.exception("/api/info: model introspection failed")
            raise ApiError(
                "service_unavailable",
                "Model introspection is unavailable; APIN engine may "
                "still be cold.",
                hint="Call POST /api/warmup, then retry GET /api/info "
                     "once /warmup/status reports all_ready=true.")

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ GET /api/feedback/stats                                       ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.get("/api/feedback/stats")
    @api_endpoint("/api/feedback/stats")
    async def v1_feedback_stats():
        # Read the same FEEDBACK_LOG with the same recency-weighting
        # rules section8 uses; the helpers are inlined above because
        # they live in section8's closure scope, not its module scope.
        if not FEEDBACK_LOG.exists():
            return {
                "total":             0,
                "correct":           0,
                "wrong":             0,
                "by_class":          {},
                "weighted_by_class": {},
                "ready_for_retrain": False,
                "retrain_gate_reason":
                    "no feedback recorded yet",
                "retrain_gate_thresholds": {
                    "min_wrong_total":         10,
                    "min_classes_with_3_each": 3,
                },
                "log_path":         str(FEEDBACK_LOG),
                "legacy_endpoint":  "/feedback/stats",
            }
        total = correct = wrong = 0
        by_class: dict = {}
        weighted_by_class: dict = {}
        with open(FEEDBACK_LOG) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                total += 1
                if r.get("is_correct"):
                    correct += 1
                    continue
                wrong += 1
                cc = r.get("correct_class", "unknown")
                by_class[cc] = by_class.get(cc, 0) + 1
                w = _recency_weight_v1(r.get("timestamp", ""))
                weighted_by_class[cc] = round(
                    weighted_by_class.get(cc, 0.0) + w, 4)
        classes_with_min = sum(1 for v in by_class.values() if v >= 3)
        ready = (wrong >= 10) and (classes_with_min >= 3)
        if ready:
            reason = "thresholds met"
        else:
            reason = (f"need >= 10 wrong (got {wrong}) AND >= 3 classes "
                      f"with >= 3 corrections (got {classes_with_min})")
        return {
            "total":              total,
            "correct":            correct,
            "wrong":              wrong,
            "accuracy":           round(correct / total, 4) if total else None,
            "by_class":           by_class,
            "weighted_by_class":  weighted_by_class,
            "ready_for_retrain":  ready,
            "retrain_gate_reason": reason,
            "retrain_gate_thresholds": {
                "min_wrong_total":         10,
                "min_classes_with_3_each": 3,
            },
            "legacy_endpoint":   "/feedback/stats",
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ GET /api/retrain/history                                      ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.get("/api/retrain/history")
    @api_endpoint("/api/retrain/history")
    async def v1_retrain_history():
        if not _RETRAIN_LOG_V1.exists():
            return {
                "events":             [],
                "n_events_returned":  0,
                "n_events_total":     0,
                "last_complete":      None,
                "rate_limit_seconds": _RETRAIN_MIN_S_V1,
                "legacy_endpoint":    "/feedback/retrain/history",
            }
        events = []
        last_complete = None
        with open(_RETRAIN_LOG_V1, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    events.append(e)
                    if e.get("status") == "complete":
                        last_complete = e.get("timestamp")
                except Exception:
                    continue
        return {
            "events":             events[-20:],
            "n_events_returned":  min(20, len(events)),
            "n_events_total":     len(events),
            "last_complete":      last_complete,
            "rate_limit_seconds": _RETRAIN_MIN_S_V1,
            "legacy_endpoint":    "/feedback/retrain/history",
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /api/feedback                                            ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.post("/api/feedback", status_code=201)
    @api_endpoint("/api/feedback", success_status=201)
    async def v1_feedback(
        payload: dict = Body(...),
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        """Submit a correction or confirmation against a past prediction.

        Bearer auth is OPTIONAL on this endpoint: anonymous feedback is
        accepted (mirrors the legacy /feedback behaviour so the public
        farmer feedback button keeps working), but feedback with a valid
        Bearer token is tagged with the user_id and shows up in their
        retraining-influence dashboard.
        """
        from datetime import datetime as _fb_dt
        if not isinstance(payload, dict):
            raise ApiError(
                "validation_failed",
                "Feedback body must be a JSON object.",
                hint='Send a JSON body like {"is_correct": false, '
                     '"predicted_class": "okra_yvmv", '
                     '"correct_class": "okra_healthy"}.')
        ok, reason = _validate_feedback_v1(payload)
        if not ok:
            raise ApiError(
                "validation_failed",
                f"Feedback rejected: {reason}",
                hint="See API_CONTRACT.md or GET /api/diseases for the list "
                     "of valid class names.",
                details={"reason": reason})

        # Optional auth: tag the record but never demand it.
        user_id = None
        if authorization or x_api_key:
            try:
                auth = _require_api_key(authorization, x_api_key)
                user_id = auth["user_id"]
            except ApiError:
                # An explicitly-bad key is a 401 — we don't silently
                # drop it to anonymous.
                raise

        record = {
            "timestamp":   _fb_dt.utcnow().isoformat() + "Z",
            "via":         "/api/feedback",
            "user_id":     user_id,
            **payload,
        }
        # Atomic append (fits in one write call; OS-level atomic at
        # the filesystem block size).
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {
            "status":     "recorded",
            "validated":  True,
            "tagged_to":  ({"user_id": user_id} if user_id else None),
            "stats_endpoint": "/api/feedback/stats",
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /api/predict/full                                        ║
    # ╚══════════════════════════════════════════════════════════════╝
    # The machine-grade rich-payload inference endpoint. Bearer auth
    # (machine consumers, not browser cookies). Routes only okra/brassica
    # diagnoses through APIN; for tomato or chilli leaves it returns a
    # clean ROUTER_REJECTED tier — the cookie-auth /predict/full still
    # routes those into TomatoPipeline for the website.
    # The contract chose this scope intentionally: machine consumers
    # most commonly want the okra/brassica decision logic. A separate
    # /api/predict/tomato can be added later under the perception
    # (drone) phase.

    def _v1_full_predict(img_rgb: np.ndarray) -> dict:
        from scripts.apin.section8_apin_server import get_apin
        apin = get_apin()
        result = apin.predict(img_rgb)
        try:
            out = asdict(result)
        except Exception:
            out = {k: v for k, v in vars(result).items()
                   if not k.startswith("_")}
        out = _convert_np(out)
        # Build the top-3 the same way /predict/quick does so machine
        # consumers don't have to re-sort the probability dict.
        probs = out.get("all_class_probabilities") or {}
        top3 = [{"class": k, "probability": round(float(v), 4)}
                for k, v in sorted(probs.items(),
                                    key=lambda kv: -float(kv[1]))[:3]]
        out["top3"]                 = top3
        out["mode"]                 = "full"
        out["api_scope"]             = "okra,brassica"
        out["api_tomato_note"] = (
            "Tomato leaves are NOT routed through this endpoint. Use "
            "the cookie-authenticated /predict/full from the website "
            "for tomato; /api/predict/tomato is on the roadmap.")
        return out

    @app.post("/api/predict/full")
    @api_endpoint("/api/predict/full")
    async def v1_predict_full(
        file: UploadFile = File(...),
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        auth = _require_api_key(authorization, x_api_key)
        try:
            data = await file.read()
        except Exception:
            # PDA §8: exception class logged server-side, not returned.
            logger.exception("/api/predict/full: upload read failed")
            raise ApiError(
                "invalid_parameter",
                "Could not read the uploaded file.",
                hint="Retry the upload; if it persists, the multipart "
                     "stream may be truncated.")
        if not data:
            raise ApiError(
                "invalid_parameter",
                "Image file is empty.",
                hint="Upload a non-empty JPEG, PNG, or WebP image.",
                field="file")
        if len(data) > 12_000_000:
            raise ApiError(
                "payload_too_large",
                f"Image exceeds the 12 MB per-file limit "
                f"(got {len(data) // 1_000_000} MB).",
                hint="Downscale the longest side to ~2048 px and retry.",
                field="file")
        try:
            pil = _PIL_Image.open(io.BytesIO(data)).convert("RGB")
            img = np.array(pil, dtype=np.uint8)
        except Exception:
            # PDA §8: decoder error class logged server-side only.
            logger.exception("/api/predict/full: image decode failed")
            raise ApiError(
                "unsupported_media_type",
                "Could not decode the uploaded file as an image.",
                hint="Send a JPEG, PNG, or WebP image.")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _v1_full_predict(img))
        result["authenticated_as"] = {
            "key_id":   auth["key_id"],
            "key_name": auth["name"],
        }
        return result

    logger.info("APIN /api/ mirror routes registered - /api/info "
                "/api/feedback /api/feedback/stats /api/retrain/history "
                "/api/predict/full")


def _add_docs_route(app):
    """Register GET /docs serving the API monograph (Phase 3).

    The page is a single self-contained HTML file at
    scripts/apin_v2/docs.html. It calls the live /api/* endpoints
    same-origin via its playground. Public, no auth required.

    FastAPI auto-registers /docs as Swagger UI. We REMOVE that
    default route first and then register ours, so the human-facing
    monograph wins. The raw OpenAPI 3.1 schema remains at
    /openapi.json (linked from the monograph header).
    """
    from starlette.routing import Route
    removed = 0
    for i in range(len(app.router.routes) - 1, -1, -1):
        r = app.router.routes[i]
        if isinstance(r, Route) and r.path == "/docs":
            app.router.routes.pop(i)
            removed += 1

    # PDA: match other HTML pages' no-cache policy so dev iterations
    # are visible immediately and so users always get the latest
    # contract documentation.
    _DOCS_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }

    @app.get("/docs", response_class=HTMLResponse)
    async def docs_page():
        return HTMLResponse(_load_html("docs.html"), headers=_DOCS_HEADERS)
    logger.info("APIN /docs page registered (Phase 3 API monograph; "
                "removed %d default Swagger route(s))", removed)


# ════════════════════════════════════════════════════════════════════════
# /api/ perception routes — Phase 2 batch 1
# ════════════════════════════════════════════════════════════════════════
# Drone-grade endpoints for fleet integrations. Every scan is a GPS-tagged
# inference; responses are RFC 7946 GeoJSON so any GIS tool (QGIS, Mapbox,
# Google Earth, drone mission planners) can render them directly. Scans
# persist to the user's account; the same user can list, retrieve, and
# soft-delete past scans for right-to-erasure compliance.
#
# Scope of batch 1: single-frame /api/scan, batch /api/scan/batch, list
# GET /api/scans, retrieve GET /api/scans/{uid}, soft-delete DELETE.
# A future batch will add flight sessions, hotspot clustering, mosaic
# tiling, and webhook callbacks. The DB schema already carries a
# nullable flight_id column so the flights table can be added without
# a data migration.

# Geo validation — drone telemetry can be sloppy. We accept the IETF
# IS-19115/WGS84 ranges and reject NaN/Inf which JSON allows but isn't
# a valid coordinate.
def _validate_geo(payload: dict) -> dict:
    """Returns the cleaned geo dict, or raises ApiError on bad input.

    Required: latitude (float, [-90, 90]), longitude (float, [-180, 180]).
    Optional: altitude_m (float), heading_deg (float, [0, 360)),
              accuracy_m (float, >= 0).
    Trailing whitespace and stray None values are tolerated.
    """
    import math
    from scripts.apin_v2.api_envelope import ApiError

    def _f(name, val, *, required, lo=None, hi=None,
           hi_exclusive=False, non_negative=False):
        if val is None or val == "":
            if required:
                raise ApiError("missing_parameter",
                               f"Field '{name}' is required.",
                               field=name)
            return None
        try:
            x = float(val)
        except (TypeError, ValueError):
            raise ApiError("invalid_parameter",
                           f"Field '{name}' must be a number.",
                           field=name)
        if math.isnan(x) or math.isinf(x):
            raise ApiError("invalid_parameter",
                           f"Field '{name}' must be a finite number.",
                           field=name)
        if non_negative and x < 0:
            raise ApiError("invalid_parameter",
                           f"Field '{name}' must be >= 0.",
                           field=name)
        if lo is not None and x < lo:
            raise ApiError(
                "invalid_parameter",
                f"Field '{name}' = {x} is below the minimum "
                f"({lo}).",
                field=name)
        if hi is not None:
            if hi_exclusive:
                if x >= hi:
                    raise ApiError(
                        "invalid_parameter",
                        f"Field '{name}' = {x} must be strictly less "
                        f"than {hi}.",
                        field=name)
            elif x > hi:
                raise ApiError(
                    "invalid_parameter",
                    f"Field '{name}' = {x} is above the maximum "
                    f"({hi}).",
                    field=name)
        return x

    return {
        "latitude":    _f("latitude",    payload.get("latitude"),
                          required=True, lo=-90.0, hi=90.0),
        "longitude":   _f("longitude",   payload.get("longitude"),
                          required=True, lo=-180.0, hi=180.0),
        "altitude_m":  _f("altitude_m",  payload.get("altitude_m"),
                          required=False),
        # PDA Phase-2: previously this silently wrapped 0..360 (e.g. -10
        # -> 350). For an industry-grade drone API that is data
        # corruption: a drone SDK reporting heading=-90 should fail
        # loud, not silently store 270. Enforce [0, 360) strictly.
        "heading_deg": _f("heading_deg", payload.get("heading_deg"),
                          required=False, lo=0.0, hi=360.0,
                          hi_exclusive=True),
        "accuracy_m":  _f("accuracy_m",  payload.get("accuracy_m"),
                          required=False, non_negative=True),
    }


def _parse_iso_captured_at(s: object) -> str:
    """Validate an ISO-8601 timestamp. Returns the canonical form
    (UTC, millisecond precision, `Z` suffix). Raises ApiError on bad
    input. We accept both naive and offset-aware strings; naive ones
    are interpreted as UTC (consistent with the contract §8 rule that
    all server timestamps are UTC)."""
    from datetime import datetime, timezone
    from scripts.apin_v2.api_envelope import ApiError
    if s is None or s == "":
        # captured_at is optional — drones occasionally drop GPS time;
        # we substitute the server's now() if missing. We document
        # this in the response under `geo_meta.captured_at_source`.
        return ""
    if not isinstance(s, str):
        raise ApiError(
            "invalid_parameter",
            "Field 'captured_at' must be an ISO-8601 timestamp string.",
            field="captured_at")
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        raise ApiError(
            "invalid_parameter",
            "Field 'captured_at' is not a valid ISO-8601 timestamp.",
            hint='Send a string like "2026-05-23T12:34:56Z" or '
                 '"2026-05-23T12:34:56+05:30".',
            field="captured_at")
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (t.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"))


def _scan_to_geojson_feature(scan: dict,
                             diagnosis_db: Optional[dict] = None) -> dict:
    """Convert a scan dict (from auth_db._scan_row_to_dict) into an
    RFC 7946 GeoJSON Feature. Coordinates are [longitude, latitude].

    Properties carry the diagnosis + treatment urgency + a one-line
    recommended action so a fleet operator's map UI can colour pins by
    urgency without a second API call. The full APINResult lives at
    GET /api/scans/{uid}?include=result for callers who want it.
    """
    geo = scan.get("geo") or {}
    lat = geo.get("latitude"); lon = geo.get("longitude")
    diag_class = scan.get("diagnosis")
    diag_info = (diagnosis_db or {}).get(diag_class) or {}
    severity = scan.get("severity") or "unknown"
    # Pick a one-line recommendation matching the severity bucket.
    treatment = diag_info.get("treatment") or {}
    recommended = None
    if isinstance(treatment, dict):
        for sev_key in (severity, "moderate", "mild", "severe"):
            recs = treatment.get(sev_key)
            if recs:
                if isinstance(recs, list) and recs:
                    recommended = str(recs[0])
                elif isinstance(recs, str):
                    recommended = recs
                break
    urgency_map = diag_info.get("urgency") or {}
    urgency = (urgency_map.get(severity) or
               urgency_map.get("moderate") or
               diag_info.get("default_urgency"))

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat],     # GeoJSON: [lon, lat]
        },
        "properties": {
            "scan_uid":          scan.get("scan_uid"),
            "flight_id":         scan.get("flight_id"),
            "diagnosis":         diag_class,
            "confidence":        scan.get("confidence"),
            "tier":              scan.get("tier"),
            "severity":          severity,
            "is_ood":            scan.get("is_ood"),
            "captured_at":       scan.get("captured_at"),
            "altitude_m":        geo.get("altitude_m"),
            "heading_deg":       geo.get("heading_deg"),
            "accuracy_m":        geo.get("accuracy_m"),
            "urgency":           urgency,
            "recommended_action": recommended,
            "image_sha256":      (scan.get("image") or {}).get("sha256"),
        },
    }


# ── Shared Console top-nav (Phase 8.F) ──────────────────────────────────
# Pre-Phase-8.F, each Console page had only `dashboard · API Console ·
# PAGE` crumbs at the top — useful for "where am I" but useless for
# cross-page navigation. This shared <nav> block ships in every Console
# page via the `<!-- @CONSOLE_NAV@ -->` placeholder, substituted at
# startup (one cost, never per-request). The active link is computed
# client-side from window.location.pathname so the same bytes work for
# every page. Styling lives inline (no extra /static/*.css fetch).
_CONSOLE_NAV_BYTES = (
    b'<nav class="apin-console-nav" aria-label="API Console">\n'
    b'  <a class="acn-logo" href="/account/api/dashboard">'
    b'<span class="acn-logo-mark">A</span><span class="acn-logo-word">APIN console</span></a>\n'
    b'  <a href="/account/api/dashboard" data-acn="/account/api/dashboard">Dashboard</a>\n'
    b'  <a href="/account/api/keys" data-acn="/account/api/keys">Keys</a>\n'
    b'  <a href="/account/api/webhooks" data-acn="/account/api/webhooks">Webhooks</a>\n'
    # Phase 8.H · Alerts nav link gets an unread-count badge child. The
    # badge is .hidden by default and revealed by apin_nav_badge.js when
    # the count is > 0. Uses the shared odometer for the digit animation.
    b'  <a href="/account/api/alerts" data-acn="/account/api/alerts" class="acn-with-badge">'
    b'Alerts'
    b'<span class="apin-nav-badge" id="apin-nav-badge" hidden aria-label="unread alerts">'
    b'<span class="apin-nav-badge-count apin-odometer" id="apin-nav-badge-count"></span>'
    b'</span>'
    b'</a>\n'
    # Phase 9.E · Usage observability — account-wide charts + drill drawer.
    b'  <a href="/account/api/usage" data-acn="/account/api/usage">Usage</a>\n'
    b'  <a href="/account/api/sandbox" data-acn="/account/api/sandbox">Sandbox</a>\n'
    b'  <a href="/account/api/settings" data-acn="/account/api/settings">Settings</a>\n'
    b'  <a href="/account/api/quickstart" data-acn="/account/api/quickstart">Quickstart</a>\n'
    b'  <span class="acn-spacer"></span>\n'
    # Phase 8.G · account chip + dropdown holder. Renders client-side
    # once /api/auth/me lands. JS lives in /static/console_account_chip.js.
    b'  <div class="apin-account-chip-wrap" id="apin-account-chip-holder">\n'
    b'    <button type="button" class="apin-chip" id="apin-chip-button"\n'
    b'            aria-haspopup="true" aria-expanded="false" aria-label="Account menu">\n'
    b'      <span class="apin-chip-avatar" id="apin-chip-avatar"></span>\n'
    b'      <span class="apin-chip-name" id="apin-chip-name">&hellip;</span>\n'
    b'      <svg class="apin-chip-caret" aria-hidden="true"><use href="#i-chevron-right"/></svg>\n'
    b'    </button>\n'
    b'    <div class="apin-chip-dropdown" id="apin-chip-dropdown" role="menu" hidden></div>\n'
    b'  </div>\n'
    b'</nav>\n'
    b'<style>\n'
    b'.apin-console-nav{position:sticky;top:0;z-index:100;display:flex;'
    b'align-items:center;gap:2px;padding:10px 22px;background:rgba(251,249,243,0.96);'
    b'border-bottom:1px solid #c7bca9;backdrop-filter:blur(6px);'
    b'font-family:\'Inter\',system-ui,sans-serif;font-size:13px;flex-wrap:wrap}\n'
    b'.apin-console-nav a{color:#5a5246;padding:6px 12px;border-radius:6px;'
    b'text-decoration:none;font-weight:500;transition:background .12s,color .12s}\n'
    b'.apin-console-nav a:hover{color:#1a1612;background:#e9e2d1}\n'
    b'.apin-console-nav a.acn-active{background:#1a1612;color:#fbf9f3}\n'
    b'.apin-console-nav .acn-logo{display:inline-flex;align-items:center;gap:8px;'
    b'margin-right:14px;padding:6px 10px 6px 6px;font-family:\'Fraunces\',serif;'
    b'font-weight:500;font-size:14px;color:#1a1612}\n'
    b'.apin-console-nav .acn-logo:hover{background:transparent}\n'
    b'.apin-console-nav .acn-logo-mark{display:inline-flex;align-items:center;'
    b'justify-content:center;width:24px;height:24px;border-radius:50%;'
    b'background:#2f6f3e;color:#fbf9f3;font-family:\'Inter\',sans-serif;'
    b'font-weight:600;font-size:12px}\n'
    b'.apin-console-nav .acn-spacer{flex:1}\n'
    b'.apin-console-nav .acn-docs,.apin-console-nav .acn-out{'
    b'font-family:\'JetBrains Mono\',monospace;font-size:11.5px;color:#5a5246}\n'
    b'.apin-console-nav .acn-docs:hover,.apin-console-nav .acn-out:hover{color:#1a1612}\n'
    b'@media (max-width:780px){.apin-console-nav{font-size:12px;padding:8px 14px}'
    b'.apin-console-nav .acn-logo-word{display:none}}\n'
    # ── Account chip + dropdown (Phase 8.G) ─────────────────────────
    b'.apin-account-chip-wrap{position:relative;display:inline-flex;align-items:center}\n'
    b'.apin-chip{display:inline-flex;align-items:center;gap:8px;padding:5px 10px 5px 5px;'
    b'border:1px solid transparent;border-radius:999px;background:transparent;'
    b'color:#1a1612;font:inherit;cursor:pointer;'
    b'transition:background .12s,border-color .12s}\n'
    b'.apin-chip:hover{background:#fbf9f3;border-color:#c7bca9}\n'
    b'.apin-chip[aria-expanded="true"]{background:#fbf9f3;border-color:#a59885}\n'
    b'.apin-chip-avatar{width:28px;height:28px;border-radius:50%;'
    b'background:#e9e2d1;display:inline-flex;align-items:center;justify-content:center;'
    b'overflow:hidden;flex-shrink:0}\n'
    b'.apin-chip-avatar svg{width:100%;height:100%;display:block}\n'
    b'.apin-chip-initial{font-family:\'Fraunces\',serif;font-weight:500;'
    b'font-size:14px;color:#2f6f3e}\n'
    b'.apin-chip-name{font-family:\'Inter\',system-ui,sans-serif;font-weight:500;'
    b'font-size:13px;max-width:120px;overflow:hidden;text-overflow:ellipsis;'
    b'white-space:nowrap}\n'
    b'.apin-chip-caret{width:12px;height:12px;color:#5a5246;'
    b'transform:rotate(90deg);transition:transform .15s}\n'
    b'.apin-chip[aria-expanded="true"] .apin-chip-caret{transform:rotate(270deg)}\n'
    b'.apin-chip-signed-out{display:inline-flex;align-items:center;gap:6px;'
    b'padding:6px 12px;border:1px solid #c7bca9;border-radius:999px;'
    b'color:#5a5246;font-size:13px;text-decoration:none;background:transparent}\n'
    b'.apin-chip-signed-out:hover{background:#fbf9f3;color:#1a1612}\n'
    # Dropdown panel
    b'.apin-chip-dropdown{position:absolute;top:calc(100% + 8px);right:0;'
    b'width:300px;background:#fbf9f3;border:1px solid #c7bca9;border-radius:12px;'
    b'box-shadow:0 12px 32px rgba(0,0,0,0.15);z-index:120;overflow:hidden;'
    b'font-family:\'Inter\',system-ui,sans-serif}\n'
    b'.apin-chip-dropdown[hidden]{display:none}\n'
    # Drifting-leaves identity strip
    b'.acd-strip{height:84px;background:linear-gradient(135deg,#e9e2d1,#d9d0bd);'
    b'border-bottom:1px solid #c7bca9}\n'
    b'.acd-identity{padding:10px 16px 12px;border-bottom:1px solid #e9e2d1;'
    b'background:#fbf9f3}\n'
    b'.acd-name{font-family:\'Fraunces\',serif;font-weight:500;font-size:16px;'
    b'color:#1a1612;letter-spacing:-0.005em;margin-bottom:2px}\n'
    b'.acd-email{font-size:12px;color:#5a5246;font-family:\'JetBrains Mono\',monospace;'
    b'overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n'
    b'.acd-menu{list-style:none;padding:6px 0;margin:0}\n'
    b'.acd-sep{height:1px;background:#e9e2d1;margin:6px 0}\n'
    b'.acd-item{display:flex;align-items:center;gap:10px;padding:8px 16px;'
    b'width:100%;color:#1a1612;text-decoration:none;font-size:13px;'
    b'background:transparent;border:0;font-family:inherit;cursor:pointer}\n'
    b'.acd-item:hover{background:#e9e2d1;text-decoration:none}\n'
    b'.acd-item-here{background:#f4efe6}\n'
    b'.acd-item-danger{color:#b01820}\n'
    b'.acd-item-danger:hover{background:#f6cfd2;color:#b01820}\n'
    b'.acd-icon{width:18px;height:18px;color:#5a5246;flex-shrink:0;'
    b'display:inline-flex;align-items:center;justify-content:center}\n'
    b'.acd-item-here .acd-icon,.acd-item-danger .acd-icon{color:inherit}\n'
    b'.acd-icon svg{width:16px;height:16px}\n'
    b'.acd-label{flex:1}\n'
    b'.acd-pill{font-size:10px;font-family:\'JetBrains Mono\',monospace;'
    b'background:#2f6f3e;color:#fbf9f3;padding:2px 6px;border-radius:8px;'
    b'text-transform:uppercase;letter-spacing:0.05em}\n'
    # Session block in dropdown
    b'.acd-session{padding:10px 16px;background:#f4efe6;'
    b'border-top:1px solid #e9e2d1;border-bottom:1px solid #e9e2d1}\n'
    b'.acd-session-label{font-size:11px;color:#5a5246;text-transform:uppercase;'
    b'letter-spacing:0.05em;margin-bottom:2px;font-family:\'JetBrains Mono\',monospace}\n'
    b'.acd-session-time{font-family:\'Fraunces\',serif;font-weight:500;font-size:18px;'
    b'color:#1a1612;margin-bottom:8px}\n'
    b'.acd-extend-btn{display:inline-flex;align-items:center;gap:6px;'
    b'padding:5px 10px;background:transparent;border:1px solid #c7bca9;'
    b'border-radius:6px;color:#1a1612;font-family:inherit;font-size:12px;'
    b'cursor:pointer}\n'
    b'.acd-extend-btn:hover{background:#fbf9f3;border-color:#5a5246}\n'
    b'.acd-extend-btn svg{width:12px;height:12px}\n'
    # Session modal (warning + ended)
    b'.apin-modal-backdrop{position:fixed;inset:0;background:rgba(26,22,18,0.55);'
    b'display:flex;align-items:center;justify-content:center;z-index:1000;'
    b'animation:apin-fade-in .15s ease-out}\n'
    b'@keyframes apin-fade-in{from{opacity:0}to{opacity:1}}\n'
    b'.apin-modal-card{background:#fbf9f3;border-radius:14px;padding:28px 32px;'
    b'max-width:420px;width:90%;text-align:center;box-shadow:0 12px 40px rgba(0,0,0,0.25);'
    b'font-family:\'Inter\',system-ui,sans-serif}\n'
    b'.apin-modal-icon{display:inline-flex;align-items:center;justify-content:center;'
    b'width:56px;height:56px;border-radius:14px;background:#fae7c8;color:#b87d1e;'
    b'margin-bottom:14px}\n'
    b'.apin-modal-icon svg{width:28px;height:28px}\n'
    b'.apin-modal-card h2{font-family:\'Fraunces\',serif;font-weight:500;font-size:22px;'
    b'color:#1a1612;margin:0 0 10px;letter-spacing:-0.005em}\n'
    b'.apin-modal-card p{color:#5a5246;font-size:14px;line-height:1.5;'
    b'margin:0 0 20px}\n'
    b'.apin-modal-actions{display:flex;gap:10px;justify-content:center}\n'
    b'.apin-btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;'
    b'padding:8px 18px;border-radius:8px;font-family:inherit;font-size:13px;'
    b'font-weight:500;cursor:pointer;text-decoration:none;border:1px solid transparent;'
    b'transition:background .15s,border-color .15s}\n'
    b'.apin-btn-primary{background:#1a1612;color:#fbf9f3;border-color:#1a1612}\n'
    b'.apin-btn-primary:hover{background:#2a2520;color:#fbf9f3}\n'
    b'.apin-btn-secondary{background:transparent;color:#1a1612;border-color:#c7bca9}\n'
    b'.apin-btn-secondary:hover{background:#fbf9f3;border-color:#5a5246}\n'
    b'@media (max-width:780px){.apin-chip-name{max-width:80px;font-size:12px}\n'
    b'.apin-chip-dropdown{width:280px;right:-8px}}\n'
    # ── Toast system (Phase 8.H) — paper card stack, bottom-right ───
    b'.apin-toast-container{position:fixed;bottom:24px;right:24px;z-index:1100;'
    b'display:flex;flex-direction:column-reverse;gap:10px;'
    b'max-width:380px;pointer-events:none}\n'
    b'.apin-toast-container > *{pointer-events:auto}\n'
    b'.apin-toast{display:grid;grid-template-columns:auto 1fr auto;gap:10px;'
    b'background:#fbf9f3;border:1px solid #c7bca9;border-radius:12px;'
    b'padding:14px 14px 14px 12px;box-shadow:0 12px 28px rgba(0,0,0,0.18);'
    b'font-family:\'Inter\',system-ui,sans-serif;'
    b'transform:translateX(0);opacity:1;'
    b'transition:transform .35s cubic-bezier(0.22,1,0.36,1),'
    b'opacity .25s ease,box-shadow .25s ease}\n'
    b'.apin-toast.apin-toast-entering{transform:translateX(calc(100% + 32px));opacity:0}\n'
    b'.apin-toast.apin-toast-leaving{transform:translateX(calc(100% + 32px));opacity:0;'
    b'box-shadow:none}\n'
    b'.apin-toast.apin-toast-collapsing{transform:translateX(0) scale(0.9);opacity:0}\n'
    b'.apin-toast-icon{width:32px;height:32px;border-radius:9px;'
    b'background:#e9e2d1;color:#5a5246;'
    b'display:inline-flex;align-items:center;justify-content:center;flex-shrink:0}\n'
    b'.apin-toast-icon svg{width:18px;height:18px}\n'
    b'.apin-toast-info .apin-toast-icon{background:#d4e4ef;color:#2d6a96}\n'
    b'.apin-toast-warn .apin-toast-icon{background:#fae7c8;color:#b87d1e}\n'
    b'.apin-toast-critical .apin-toast-icon{background:#f6cfd2;color:#b01820}\n'
    b'.apin-toast-critical{border-color:#e8b0b4}\n'
    b'.apin-toast-warn{border-color:#d6c39a}\n'
    b'.apin-toast-body{min-width:0}\n'
    b'.apin-toast-title{font-family:\'Fraunces\',serif;font-weight:500;'
    b'font-size:15px;letter-spacing:-0.005em;color:#1a1612;line-height:1.25;'
    b'margin-bottom:3px}\n'
    b'.apin-toast-text{font-size:13px;color:#5a5246;line-height:1.45;'
    b'word-break:break-word}\n'
    b'.apin-toast-actions{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}\n'
    b'.apin-toast-action{display:inline-flex;align-items:center;gap:4px;'
    b'font-family:\'Inter\',system-ui,sans-serif;font-weight:500;font-size:12px;'
    b'text-decoration:none;padding:5px 10px;border-radius:6px;'
    b'border:1px solid transparent;cursor:pointer;background:transparent;'
    b'transition:background .12s,border-color .12s,color .12s}\n'
    b'.apin-toast-action svg{width:12px;height:12px}\n'
    b'.apin-toast-action-info{color:#2d6a96;border-color:#bcd2e1}\n'
    b'.apin-toast-action-info:hover{background:#d4e4ef;text-decoration:none}\n'
    b'.apin-toast-action-warn{color:#b87d1e;border-color:#d6c39a}\n'
    b'.apin-toast-action-warn:hover{background:#fae7c8;text-decoration:none}\n'
    b'.apin-toast-action-critical{color:#b01820;border-color:#e8b0b4}\n'
    b'.apin-toast-action-critical:hover{background:#f6cfd2;text-decoration:none}\n'
    b'.apin-toast-action-button{font:inherit;font-weight:500;font-size:12px;'
    b'padding:5px 10px;line-height:1.2}\n'
    b'.apin-toast-close{background:transparent;border:0;color:#5a5246;'
    b'cursor:pointer;padding:2px;border-radius:4px;align-self:flex-start;'
    b'flex-shrink:0;width:22px;height:22px;display:inline-flex;'
    b'align-items:center;justify-content:center;'
    b'transition:background .12s,color .12s}\n'
    b'.apin-toast-close svg{width:14px;height:14px}\n'
    b'.apin-toast-close:hover{background:#e9e2d1;color:#1a1612}\n'
    # ── Pill (collapsed-stack) ────────────────────────────────────
    b'.apin-toast-pill{display:inline-flex;align-items:center;gap:10px;'
    b'background:#1a1612;color:#fbf9f3;border:1px solid #1a1612;'
    b'border-radius:999px;padding:8px 8px 8px 14px;'
    b'box-shadow:0 12px 28px rgba(0,0,0,0.25);'
    b'cursor:pointer;font:inherit;'
    b'font-family:\'Inter\',system-ui,sans-serif;font-size:13px;font-weight:500;'
    b'transform:translateX(0);opacity:1;'
    b'transition:transform .35s cubic-bezier(0.22,1,0.36,1),'
    b'opacity .25s ease,box-shadow .25s ease}\n'
    b'.apin-toast-pill-entering{transform:translateX(calc(100% + 32px));opacity:0}\n'
    b'.apin-toast-pill-leaving{transform:translateX(calc(100% + 32px));opacity:0}\n'
    b'.apin-toast-pill-bump{animation:apin-pill-bump .42s cubic-bezier(0.22,1,0.36,1)}\n'
    b'@keyframes apin-pill-bump{0%{transform:scale(1)}40%{transform:scale(1.07)}'
    b'100%{transform:scale(1)}}\n'
    b'.apin-toast-pill-icon{width:24px;height:24px;border-radius:50%;'
    b'background:#fbf9f3;color:#1a1612;'
    b'display:inline-flex;align-items:center;justify-content:center}\n'
    b'.apin-toast-pill-icon svg{width:14px;height:14px}\n'
    b'.apin-toast-pill-count{font-family:\'Fraunces\',serif;font-weight:500;'
    b'font-size:16px;display:inline-flex;align-items:center;'
    b'min-width:1ch;color:#fbf9f3}\n'
    b'.apin-toast-pill-label{color:#fbf9f3;opacity:0.85}\n'
    b'.apin-toast-pill-close{display:inline-flex;align-items:center;'
    b'justify-content:center;width:22px;height:22px;border-radius:50%;'
    b'background:rgba(255,255,255,0.08);color:#fbf9f3;cursor:pointer;'
    b'margin-left:4px;transition:background .12s}\n'
    b'.apin-toast-pill-close:hover{background:rgba(255,255,255,0.18)}\n'
    b'.apin-toast-pill-close svg{width:12px;height:12px}\n'
    # ── Nav badge (Phase 8.H.C — unread count on the Alerts link) ─
    b'.apin-console-nav a.acn-with-badge{display:inline-flex;align-items:center;'
    b'gap:6px}\n'
    b'.apin-nav-badge{display:inline-flex;align-items:center;justify-content:center;'
    b'min-width:18px;height:18px;padding:0 5px;border-radius:9px;'
    b'background:#b01820;color:#fbf9f3;font-family:\'JetBrains Mono\',monospace;'
    b'font-size:10.5px;font-weight:600;line-height:1;letter-spacing:0;'
    b'transform-origin:center;'
    b'transition:transform .2s ease,background .15s ease}\n'
    b'.apin-nav-badge[hidden]{display:none !important}\n'
    b'.apin-console-nav a.acn-active .apin-nav-badge{background:#fbf9f3;color:#1a1612}\n'
    # Pulse on increment — half-second swell so the user notices a new alert
    # even if their cursor isn\'t near the nav.
    b'.apin-nav-badge-pulse{animation:apin-nav-badge-bump .5s cubic-bezier(0.22,1,0.36,1)}\n'
    b'@keyframes apin-nav-badge-bump{'
    b'0%{transform:scale(1)}30%{transform:scale(1.35)}'
    b'70%{transform:scale(0.92)}100%{transform:scale(1)}}\n'
    # Make the odometer inside the badge stay compact + cream-coloured.
    b'.apin-nav-badge .apin-odometer{font-family:\'JetBrains Mono\',monospace;'
    b'font-size:10.5px;font-weight:600;color:inherit}\n'
    b'.apin-nav-badge .apin-odometer-digit{width:.55em}\n'
    # ── Odometer styles (shared with the badge in 6.C) ────────────
    b'.apin-odometer{display:inline-flex;align-items:center;line-height:1;'
    b'font-feature-settings:"tnum" 1,"lnum" 1}\n'
    b'.apin-odometer-digit{display:inline-block;width:.6em;height:1em;'
    b'overflow:hidden;position:relative;vertical-align:baseline}\n'
    b'.apin-odometer-digit-col{display:inline-block;width:100%;'
    b'will-change:transform}\n'
    b'.apin-odometer-digit-col > span{display:block;height:1em;line-height:1;'
    b'text-align:center}\n'
    b'.apin-odometer-static{display:inline-block;line-height:1}\n'
    b'.apin-odometer-digit.is-spinning .apin-odometer-digit-col{'
    b'filter:blur(0.5px)}\n'
    # ── Responsive — make the toast stack hug the bottom on phones ─
    b'@media (max-width:560px){.apin-toast-container{left:12px;right:12px;'
    b'bottom:12px;max-width:none}}\n'
    b'</style>\n'
    # Active-link highlighting JS lives in /static/console_nav.js — strict
    # `script-src 'self'` blocks the previous inline <script> block.
    b'<script src="/static/console_nav.js"></script>\n'
    # Phase 8.G · chip + session machinery. pressed_leaf.js first so the
    # avatar generator is on window before the chip JS runs.
    b'<script src="/static/pressed_leaf.js"></script>\n'
    b'<script src="/static/console_account_chip.js"></script>\n'
    # Phase 8.H · global toast system. odometer.js first so the pill count
    # animation can use it. apin_toast.js initialises on DOMContentLoaded.
    b'<script src="/static/odometer.js"></script>\n'
    b'<script src="/static/apin_toast.js"></script>\n'
    # Phase 8.H.C · nav badge for unread alerts.
    b'<script src="/static/apin_nav_badge.js"></script>\n'
    # Phase 9.N.1 · shared animation library. Loads before any chart code
    # so APIN.fx.* is on window for every page. (Icons come from the
    # already-injected console_icons.svg — same hand-drawn set.)
    b'<script src="/static/apin_fx.js"></script>\n'
)
_CONSOLE_NAV_PLACEHOLDER = b"<!-- @CONSOLE_NAV@ -->"


def _inject_console_nav(body_bytes: bytes) -> bytes:
    """Substitute the nav placeholder in a cached page body. Returns the
    original bytes unchanged if the placeholder is absent (defensive: if
    a page hasn't been migrated yet, it still serves)."""
    if _CONSOLE_NAV_PLACEHOLDER in body_bytes:
        return body_bytes.replace(_CONSOLE_NAV_PLACEHOLDER, _CONSOLE_NAV_BYTES)
    return body_bytes


# ── Shared icon sprite injection (Phase 8.G) ────────────────────────────
# Console pages don't include ui_template.html so <use href="#i-*"/>
# wouldn't resolve. We inline-inject the 62-symbol sprite into each Console
# page via the <!-- @ICON_SPRITE@ --> placeholder. Inlining (vs an external
# /static/icons.svg) is necessary because external sprites have a long-
# standing browser bug where `currentColor` doesn't inherit across documents
# — and our icons are drawn with stroke=currentColor specifically so they
# can pick up paper-ink theming.
#
# Source: scripts/apin_v2/console_icons.svg (regenerated by
# _build_icon_sprite.py from ui_template.html's symbol block).
_CONSOLE_ICON_SPRITE_PATH = Path(__file__).resolve().parent / "console_icons.svg"
try:
    _CONSOLE_ICON_SPRITE_BYTES = _CONSOLE_ICON_SPRITE_PATH.read_bytes()
except FileNotFoundError:
    # Defensive: if the sprite hasn't been built yet, fall back to empty.
    # Pages still serve; icons just don't render.
    _CONSOLE_ICON_SPRITE_BYTES = b""
    logger.warning(
        "Console icon sprite missing at %s. Run "
        "`python scripts/apin_v2/_build_icon_sprite.py` to regenerate.",
        _CONSOLE_ICON_SPRITE_PATH,
    )
_CONSOLE_ICON_SPRITE_PLACEHOLDER = b"<!-- @ICON_SPRITE@ -->"


def _inject_icon_sprite(body_bytes: bytes) -> bytes:
    """Substitute the icon-sprite placeholder. Same defensive shape as
    `_inject_console_nav`: returns body unchanged if placeholder is absent."""
    if _CONSOLE_ICON_SPRITE_PLACEHOLDER in body_bytes:
        return body_bytes.replace(
            _CONSOLE_ICON_SPRITE_PLACEHOLDER, _CONSOLE_ICON_SPRITE_BYTES)
    return body_bytes


def _inject_console_chrome(body_bytes: bytes) -> bytes:
    """Convenience wrapper · run BOTH nav + sprite injectors on a page.

    Adopted in Phase 8.G — the 7 Console pages now get both bits of chrome.
    Cheap call: each helper is a no-op if its placeholder is absent."""
    body_bytes = _inject_console_nav(body_bytes)
    body_bytes = _inject_icon_sprite(body_bytes)
    return body_bytes


def _add_account_console_routes(app):
    """Mount the API Console — Stage 7 / Phase 3.2 (R1-fixed).

    SPEC-CONFORMANCE QUICK REFERENCE (read this first):

      Implemented (5 of 7 slots): GZip(1) · TokenRedaction(3) ·
      TokenFormat(4) · Session(6) · Sudo(7).
      NOT implemented: AuditRecentMiddleware(5).
      Factory-mounted (slot drift): CORSMiddleware(2) ends up INNERMOST
      because the factory mounts it before this helper runs.
      See DEC-P32-MW-1 and DEC-P32-FIX-* in
      `_qa_tmp/api_console_spec/decisions.md` for the deviation log.

    What this helper wires into the apin_v2 app:

    1. The Phase 2.3 + 3.2 security middleware stack, registered in
       REVERSE slot order per spec §9.1. Starlette `add_middleware()`
       prepends to `app.user_middleware`, so LAST-added = OUTERMOST at
       runtime:

         add_middleware(SudoMiddleware)              # slot 7  innermost
         add_middleware(SessionMiddleware)           # slot 6
         add_middleware(TokenFormatMiddleware)       # slot 4
         add_middleware(TokenRedactionMiddleware)    # slot 3
         add_middleware(GZipMiddleware)              # slot 1  outermost

       Final runtime order outermost→innermost:
         GZip > TokenRedaction > TokenFormat > Session > Sudo > CORS > handler

       Slot dependencies:
         - SudoMiddleware reads `scope.state.session.session_id` (set
           by SessionMiddleware on slot 6) to bind sudo to the
           originating session. Without SessionMiddleware, Sudo
           fail-CLOSES on `session_id is None` (Phase 2 R1 F02 fix).
           PDA-P3.2-R1-F01 caught the missing-Session case in R1 and
           this commit installs SessionMiddleware to close it.
         - TokenRedactionMiddleware must be INNER of GZip — otherwise
           it scans gzipped bytes and misses plaintext leaks. PDA-P3.2-
           R1-F02 caught the inverted ordering in R1 (when GZip was
           mounted in main() before this helper ran).

       Slot gaps:
         - AuditRecentMiddleware (slot 5) — not implemented. Audit
           logging is per-handler via `audit_log()`. Tracked in spec
           §22.3 IMPL-PHASE.
         - CORSMiddleware (slot 2) — mounted by the factory before
           this helper, so ends up INNERMOST. Harmless for the Phase 3
           same-origin Console scope; cross-origin developer API
           calls would lose CORS headers on 401/403 responses. Phase 4
           work item: clear and re-stack, or move CORS-add into this
           helper.

    2. The FULL keys router (`routes_keys.router`) — 2 GET + 4 mutation
       routes. SD-2's Phase 3.1 read-only filter is RETIRED: now that
       SessionMiddleware + SudoMiddleware are mounted, mutations are
       reachable AND properly sudo-gated per spec §7.1.

    3. `GET /account/api/keys` HTML page serving `keys.html` with the
       FX-A CSP, FX-B 4-directive Cache-Control, and FX-H startup cache.

    4. `GET /account/api` → 307 → `GET /account/api/dashboard` → 307 →
       `GET /account/api/keys`. Dashboard remains a STUB per DEC-P31-SD-1.

    5. `GET /static/keys.js` — externalised page JS (FX-A) with ETag
       revalidation, same hand-rolled pattern as `/static/telemetry.js`.

    Full audit trail: `_qa_tmp/api_console_spec/decisions.md`
    (DEC-P31-SD-1, SD-2, FX-A through FX-I, DEC-P32-MW-1,
    DEC-P32-FIX-F01 through F06).
    """
    from scripts.apin_v2.account import routes_keys as _rk
    from scripts.apin_v2.account.middlewares import (
        TokenFormatMiddleware,
        TokenRedactionMiddleware,
        SessionMiddleware,
        SudoMiddleware,
    )
    from fastapi.middleware.gzip import GZipMiddleware

    # ── Middleware stack (spec §9.1 REVERSE registration order) ─────
    # Starlette prepends, so register innermost FIRST, outermost LAST.
    # SessionMiddleware MUST be registered before SudoMiddleware so the
    # session_id is populated before Sudo's binding check. GZip MUST be
    # registered LAST so it runs OUTERMOST and never strips redaction's
    # ability to see plaintext.
    #
    # Verify ordering via _qa_tmp/test_stage7_phase3_2_middleware_order.py
    # (spec §22.2 — created in Phase 3.2 R1 fix per VER-P3.2-R1 C5).
    # Phase 5.2 (CORS-SLOT): re-mount CORS at spec slot 2.
    #
    # The factory `make_app()` mounts CORSMiddleware first, so by the
    # time this helper runs, `app.user_middleware == [CORS]`. The
    # straightforward `add_middleware` chain below would leave CORS
    # INNERMOST (Starlette prepends, factory was first → ends up at
    # the tail). DEC-P32-MW-1 documented the drift as harmless for
    # same-origin Console traffic; it's a real problem for third-party
    # cross-origin developer code (401/403 from inner middlewares
    # arrives without CORS headers → browser shows "blocked by CORS"
    # instead of the actual error code).
    #
    # Fix: extract the factory's CORS (preserving its config), then
    # filter it out of user_middleware. Re-add it at the right slot
    # below — between TokenRedaction (slot 3, inner) and GZip (slot 1,
    # outer). Final stack: GZip > CORS > TokenRedaction > TokenFormat
    # > Session > Sudo > handler (canonical spec §9.1 order).
    from fastapi.middleware.cors import CORSMiddleware as _CORS
    _factory_cors_kwargs = None
    # FX-P5-3 (PDA-P5-R1 F04): scan ALL CORS middlewares (don't break
    # after first). If multiple are mounted, the FIRST is preserved as
    # the canonical config — subsequent ones get logged as a warning so
    # an operator can investigate accidental double-mounts. The filter
    # below still removes ALL of them, which is correct: we re-add ONE
    # at slot 2.
    _cors_found = 0
    for _mw in list(app.user_middleware):
        if getattr(_mw, "cls", None) is _CORS:
            _cors_found += 1
            if _factory_cors_kwargs is None:
                _factory_cors_kwargs = dict(getattr(_mw, "kwargs", {}) or {})
    if _cors_found > 1:
        logger.warning(
            "API Console mount: factory pre-mounted %d CORSMiddleware "
            "instances. Preserving config of the FIRST; subsequent "
            "configs are discarded. Verify the factory isn't mounting "
            "CORS twice — see PDA-P5-R1 F04 / DEC-P5-R1-FIXES.",
            _cors_found,
        )
    if _factory_cors_kwargs is None:
        # Factory's CORS was not mounted (defensive — shouldn't happen
        # in production, but in tests that build a bare FastAPI app, we
        # still want the right ordering). Use the same config the factory
        # would have used (allow_origins=["*"], allow_credentials=True,
        # allow_methods=["*"], allow_headers=["*"]).
        _factory_cors_kwargs = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    # Filter the factory's CORS out so we don't end up with two CORS
    # middlewares (the original innermost + our new slot-2 one).
    app.user_middleware = [
        _mw for _mw in app.user_middleware
        if getattr(_mw, "cls", None) is not _CORS
    ]

    # Now register in REVERSE spec-slot order. Starlette prepends, so the
    # LAST-added ends up OUTERMOST at runtime.
    #
    # Phase 9.F · UsageRecordingMiddleware sits INNERMOST (registered
    # FIRST). That position lets it observe:
    #   - the FINAL response status the client actually receives
    #     (post-redaction, post-CORS, post-token-format rewrites)
    #   - the real wire bytes_out (sum of body chunks)
    # and ensures any auth-rejected request from an API key still gets
    # recorded with its true 401/403 status code rather than swallowed.
    from scripts.apin_v2.account.middlewares import UsageRecordingMiddleware
    app.add_middleware(UsageRecordingMiddleware)             # slot 8  innermost  (Phase 9.F)
    app.add_middleware(SudoMiddleware)                       # slot 7
    app.add_middleware(SessionMiddleware)                    # slot 6
    app.add_middleware(TokenFormatMiddleware)                # slot 4
    app.add_middleware(TokenRedactionMiddleware)             # slot 3
    app.add_middleware(_CORS, **_factory_cors_kwargs)        # slot 2  (re-positioned from innermost — Phase 5.2)
    app.add_middleware(GZipMiddleware, minimum_size=1024)    # slot 1  outermost  (registered LAST)
    logger.info(
        "API Console middleware stack (Phase 5.2 R1-fixed + 9.F): "
        "GZip (1) > CORS (2) > TokenRedaction (3) > TokenFormat (4) > "
        "Session (6) > Sudo (7) > UsageRecording (8, innermost) > handler"
    )

    # ── Routers (Phase 3.2 keys + Phase 3.3 sudo) ──────────────────
    app.include_router(_rk.router)
    _route_count = sum(
        1 for r in _rk.router.routes
        if hasattr(r, "methods") and r.methods
    )
    logger.info(
        "API Console keys router mounted (Phase 3.2): all %d routes from "
        "routes_keys.router (2 GET + 4 mutation). Mutations gated by "
        "Sudo+Session middleware per spec §7.1.",
        _route_count,
    )

    # Phase 5.3 (FX-I): self-hosted font serving route.
    #
    # The Console pages currently link to fonts.googleapis.com (which
    # serves a CSS file that references fonts.gstatic.com for the woff2
    # binaries). To self-host: run `python tools/fetch_fonts.py` to
    # populate `scripts/apin_v2/fonts/*.woff2`, then the route below
    # serves them at `/fonts/<filename>`.
    #
    # When the fonts directory is empty (no fetch script run yet), this
    # route returns 404 and the page continues working via the Google
    # Fonts CDN fallback. When populated, the @font-face local rules
    # (Phase 5+ work item — keys.html addition) take precedence and the
    # CDN is bypassed. At that point the CSP `style-src`/`font-src` can
    # be tightened to drop the fonts.googleapis.com / fonts.gstatic.com
    # entries.
    from pathlib import Path as _PFonts
    import re as _re_fonts
    _FONTS_DIR = _PFonts(__file__).resolve().parent / "fonts"
    # Whitelist allowed filenames to prevent path traversal. Only woff2.
    _FONTS_FILENAME_RE = _re_fonts.compile(r"^[A-Za-z0-9._-]+\.woff2$")

    @app.get("/fonts/{filename}")
    async def v2_serve_font(filename: str):
        from fastapi.responses import Response, FileResponse
        if not _FONTS_FILENAME_RE.match(filename):
            return Response(status_code=400, content=b"invalid filename")
        path = _FONTS_DIR / filename
        if not path.is_file():
            # Not yet self-hosted — let the Google Fonts CDN fallback
            # handle it. 404 is fine because the @font-face rules will
            # also reference the CDN as a secondary src.
            return Response(status_code=404, content=b"font not present "
                            b"locally; run tools/fetch_fonts.py")
        # woff2 is the only format we serve; immutable content (binary
        # data, never edited). Long cache + immutable.
        return FileResponse(
            str(path),
            media_type="font/woff2",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
            },
        )

    # Phase 3.3 sudo endpoints (POST/GET /api/account/sudo + POST /revoke).
    # These are EXEMPT from SudoMiddleware (spec §9.2 / middlewares.py
    # `_SUDO_EXEMPT_PATHS` includes `/api/account/sudo` prefix) — the
    # chicken-and-egg avoidance: you mint sudo via these endpoints.
    from scripts.apin_v2.account import routes_sudo as _rs
    app.include_router(_rs.router)

    # Phase 6.B.1: account settings GET + PATCH (PATCH gated by sudo).
    from scripts.apin_v2.account import routes_settings as _rset
    app.include_router(_rset.router)

    # Phase 7 Wave 1: Webhook + Alert routers.
    # Webhooks: 8 routes (CRUD + rotate-secret + test-ping + deliveries).
    # Alerts:   6 routes (list + unread-count + get + read + dismiss + restore).
    # Webhook mutations gated by SudoMiddleware (slot 7); alert mutations do
    # NOT require sudo per spec §9.2 (read-state housekeeping only). All
    # mutating verbs additionally enforce CSRF at the route layer.
    from scripts.apin_v2.account import routes_webhooks as _rwh
    from scripts.apin_v2.account import routes_alerts as _ral
    app.include_router(_rwh.router)
    app.include_router(_ral.router)
    logger.info(
        "API Console webhook + alert routers mounted (Phase 7 Wave 1): "
        "%d webhook routes + %d alert routes. Webhook secret encryption "
        "requires APIN_SECRET_KEY env var; webhook endpoints return 503 "
        "service_unavailable when unset.",
        len(_rwh.router.routes), len(_ral.router.routes),
    )

    # Phase 9.B — usage observability router.
    # Five GET endpoints under /api/account/usage/* (summary, timeseries,
    # top, requests, minute-detail). Session-cookie auth only — Bearer/
    # X-API-Key rejected by TokenFormatMiddleware (slot 4).
    from scripts.apin_v2.account import routes_usage as _ru
    app.include_router(_ru.router)
    logger.info(
        "API Console usage router mounted (Phase 9.B): %d usage routes.",
        len(_ru.router.routes),
    )

    # Phase 6.B.1 + B.2: Console settings + quickstart HTML pages.
    # Settings page needs CSRF substitution (same pattern as keys.html);
    # quickstart is read-only static content, no CSRF token needed.
    _console_settings_body_str = _load_html("console_settings.html")
    _console_settings_body_bytes = (
        _console_settings_body_str.encode("utf-8")
        if isinstance(_console_settings_body_str, str)
        else _console_settings_body_str
    )
    _console_settings_body_bytes = _inject_console_chrome(_console_settings_body_bytes)
    _console_quickstart_body_str = _load_html("console_quickstart.html")
    _console_quickstart_body_bytes = (
        _console_quickstart_body_str.encode("utf-8")
        if isinstance(_console_quickstart_body_str, str)
        else _console_quickstart_body_str
    )
    _console_quickstart_body_bytes = _inject_console_chrome(_console_quickstart_body_bytes)

    # FX-P7-F05: hoist the CSRF placeholder constant ABOVE the first route
    # handler that closes over it. The previous definition was 100 lines
    # below at the bottom of this function, which worked only thanks to
    # Python's lazy free-variable lookup but was fragile to refactors.
    _CSRF_PLACEHOLDER_BYTES = b"__APIN_CSRF_TOKEN__"

    @app.get("/account/api/settings", response_class=HTMLResponse)
    async def account_settings_page(request: Request):
        from scripts.apin_v2 import auth_db as _adb
        csrf_value = b""
        raw_session = request.cookies.get("apin_v2_session")
        if raw_session:
            try:
                row = _adb.lookup_session_by_token(raw_session)
                if row and row.get("csrf_token"):
                    csrf_value = row["csrf_token"].encode("ascii")
            except Exception:
                pass
        body = _console_settings_body_bytes.replace(
            _CSRF_PLACEHOLDER_BYTES, csrf_value)
        return HTMLResponse(body, headers=_ACCOUNT_PAGE_HEADERS)

    @app.get("/account/api/quickstart", response_class=HTMLResponse)
    async def account_quickstart_page():
        # No CSRF substitution — quickstart page makes no API calls.
        return HTMLResponse(_console_quickstart_body_bytes,
                            headers=_ACCOUNT_PAGE_HEADERS)

    # ── Phase 7 Waves 4 + 5: Sandbox + Alerts + Webhooks pages ─────────────
    # All three pages get CSRF substitution (they each fire mutating fetches).
    _console_sandbox_body_str = _load_html("console_sandbox.html")
    _console_sandbox_body_bytes = (
        _console_sandbox_body_str.encode("utf-8")
        if isinstance(_console_sandbox_body_str, str)
        else _console_sandbox_body_str
    )
    _console_sandbox_body_bytes = _inject_console_chrome(_console_sandbox_body_bytes)
    _console_alerts_body_str = _load_html("console_alerts.html")
    _console_alerts_body_bytes = (
        _console_alerts_body_str.encode("utf-8")
        if isinstance(_console_alerts_body_str, str)
        else _console_alerts_body_str
    )
    _console_alerts_body_bytes = _inject_console_chrome(_console_alerts_body_bytes)
    _console_webhooks_body_str = _load_html("console_webhooks.html")
    _console_webhooks_body_bytes = (
        _console_webhooks_body_str.encode("utf-8")
        if isinstance(_console_webhooks_body_str, str)
        else _console_webhooks_body_str
    )
    _console_webhooks_body_bytes = _inject_console_chrome(_console_webhooks_body_bytes)

    def _serve_csrf_page(request: Request, body_bytes: bytes) -> HTMLResponse:
        """Shared helper: substitute the CSRF placeholder with the caller's
        live csrf_token from their session row, then serve with the standard
        account-page security headers."""
        from scripts.apin_v2 import auth_db as _adb
        csrf_value = b""
        raw_session = request.cookies.get("apin_v2_session")
        if raw_session:
            try:
                row = _adb.lookup_session_by_token(raw_session)
                if row and row.get("csrf_token"):
                    csrf_value = row["csrf_token"].encode("ascii")
            except Exception:
                pass
        return HTMLResponse(
            body_bytes.replace(_CSRF_PLACEHOLDER_BYTES, csrf_value),
            headers=_ACCOUNT_PAGE_HEADERS,
        )

    @app.get("/account/api/sandbox", response_class=HTMLResponse)
    async def account_sandbox_page(request: Request):
        return _serve_csrf_page(request, _console_sandbox_body_bytes)

    @app.get("/account/api/alerts", response_class=HTMLResponse)
    async def account_alerts_page(request: Request):
        return _serve_csrf_page(request, _console_alerts_body_bytes)

    @app.get("/account/api/webhooks", response_class=HTMLResponse)
    async def account_webhooks_page(request: Request):
        return _serve_csrf_page(request, _console_webhooks_body_bytes)

    # Phase 8 Wave D: key detail full PAGE. Replaces the modal-only flow.
    _console_key_detail_str = _load_html("console_key_detail.html")
    _console_key_detail_bytes = (
        _console_key_detail_str.encode("utf-8")
        if isinstance(_console_key_detail_str, str)
        else _console_key_detail_str
    )
    _console_key_detail_bytes = _inject_console_chrome(_console_key_detail_bytes)
    _PID_PLACEHOLDER = b"__KEY_PUBLIC_ID__"

    @app.get("/account/api/keys/{public_id}", response_class=HTMLResponse)
    async def account_key_detail_page(public_id: str, request: Request):
        # Validate the public_id format (defence-in-depth — also serves
        # to keep the placeholder substitution honest).
        if not (1 <= len(public_id) <= 80) or not all(
                c.isalnum() or c in '_-' for c in public_id):
            return HTMLResponse(b"invalid public_id", status_code=400,
                                 headers=_ACCOUNT_PAGE_HEADERS)
        from scripts.apin_v2 import auth_db as _adb
        csrf_value = b""
        raw_session = request.cookies.get("apin_v2_session")
        if raw_session:
            try:
                row = _adb.lookup_session_by_token(raw_session)
                if row and row.get("csrf_token"):
                    csrf_value = row["csrf_token"].encode("ascii")
            except Exception:
                pass
        body = _console_key_detail_bytes.replace(
            _CSRF_PLACEHOLDER_BYTES, csrf_value
        ).replace(_PID_PLACEHOLDER, public_id.encode("ascii"))
        return HTMLResponse(body, headers=_ACCOUNT_PAGE_HEADERS)

    # Phase 9.E · Usage observability page.
    # Account-wide charts (KPI strip + time-series + status donut + latency
    # histogram + top-N panels + recent requests + CSV export + drill drawer).
    # Backed by /api/account/usage/* (Phase 9.B).
    _console_usage_body_bytes = _load_html("console_usage.html")
    if isinstance(_console_usage_body_bytes, str):
        _console_usage_body_bytes = _console_usage_body_bytes.encode("utf-8")
    _console_usage_body_bytes = _inject_console_chrome(_console_usage_body_bytes)

    @app.get("/account/api/usage", response_class=HTMLResponse)
    async def account_usage_page(request: Request):
        return _serve_csrf_page(request, _console_usage_body_bytes)

    logger.info(
        "API Console pages mounted (Phase 7 Waves 4+5 + Phase 8 Wave D + "
        "Phase 9.E): /account/api/sandbox, /alerts, /webhooks, /usage, "
        "/keys/{public_id}"
    )

    _sudo_count = sum(
        1 for r in _rs.router.routes
        if hasattr(r, "methods") and r.methods
    )
    logger.info(
        "API Console sudo router mounted (Phase 3.3): all %d routes from "
        "routes_sudo.router (POST /sudo, POST /sudo/revoke, GET /sudo).",
        _sudo_count,
    )

    # HTML page for the keys list
    #
    # Security headers per spec §18.1 with two pragmatic deviations
    # documented in DEC-P31-FX-A:
    #   - script-src 'self' (stricter than spec's nonce variant; we have
    #     zero inline scripts after the keys.js extraction)
    #   - style-src 'self' 'unsafe-inline' fonts.googleapis.com (allows
    #     the existing inline <style> block + the Google Fonts CSS link;
    #     extracting CSS was out-of-scope for Phase 3.1)
    #   - font-src 'self' fonts.gstatic.com (allows Google Fonts woff2;
    #     self-hosting deferred to Phase 4 per DEC-P31-FX-I)
    #
    # Phase 6.C.3 update: @font-face local rules added to keys.html,
    # console_settings.html, console_quickstart.html (referencing
    # /fonts/<name>.woff2). When `python tools/fetch_fonts.py` has been
    # run and all 3 families self-host successfully, the
    # `https://fonts.googleapis.com` entry from style-src and
    # `https://fonts.gstatic.com` from font-src can be dropped. Until
    # Fraunces is reliably self-hosted (needs fetch_fonts.py v2 for
    # variable-font support), the allowlist must stay.
    # FX-P8-A1 (closes QA-1 properly): all 5 Phase 7 Console pages now
    # source their JS via <script src="/static/console_<name>.js">, matching
    # the keys.html pattern. The previous Phase 7 'unsafe-inline' band-aid
    # is REMOVED here — strict `script-src 'self'` restored. style-src
    # keeps 'unsafe-inline' because the pages still ship inline <style>
    # blocks (separate item — out of Phase 8 scope, deferred to Phase 9+
    # along with self-host Fraunces).
    _ACCOUNT_PAGE_CSP = (
        "default-src 'none'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    _ACCOUNT_PAGE_HEADERS = {
        # FX-B: max-age=0 added — spec R3-I17 4-directive canonical form.
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "X-Content-Type-Options": "nosniff",
        # Console pages should NEVER be embedded in iframes (clickjacking
        # defence — the create wizard reveals one-time plaintext tokens).
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        # FX-A: CSP defined above. frame-ancestors 'none' duplicates the
        # X-Frame-Options DENY guarantee for modern browsers.
        "Content-Security-Policy": _ACCOUNT_PAGE_CSP,
    }

    # FX-H: cache keys.html bytes at startup. Same pattern the inference
    # site uses for reports.html / loupe.html / etc. Reads disk once per
    # process boot rather than per request.
    _keys_html_body_str = _load_html("keys.html")
    _keys_html_body_bytes = (
        _keys_html_body_str.encode("utf-8")
        if isinstance(_keys_html_body_str, str)
        else _keys_html_body_str
    )
    _keys_html_body_bytes = _inject_console_chrome(_keys_html_body_bytes)
    # FX-P7-F05: _CSRF_PLACEHOLDER_BYTES is now defined at the top of this
    # function (above the first route handler that needs it). The previous
    # definition here was redundant and the previous definition order made
    # the earlier route handlers' closure-binding fragile.

    @app.get("/account/api/keys", response_class=HTMLResponse)
    async def account_keys_page(request: Request):
        # WI-P4-CSRF: substitute the per-session CSRF token into the
        # cached HTML body. The placeholder __APIN_CSRF_TOKEN__ appears
        # exactly once (in the <meta name="csrf-token"> tag at the top
        # of keys.html). On miss (no session, expired session, DB error),
        # we emit an empty token — the JS will then send empty
        # X-Console-Csrf, the server-side `_require_csrf` will reject
        # with invalid_or_missing_token, and the wizard surfaces the
        # session-expired modal flow.
        from scripts.apin_v2 import auth_db as _adb
        csrf_value = b""
        raw_session = request.cookies.get("apin_v2_session")
        if raw_session:
            try:
                row = _adb.lookup_session_by_token(raw_session)
                if row and row.get("csrf_token"):
                    csrf_value = row["csrf_token"].encode("ascii")
            except Exception:
                # Defensive: DB failure falls through to empty token.
                pass
        body = _keys_html_body_bytes.replace(
            _CSRF_PLACEHOLDER_BYTES, csrf_value)
        return HTMLResponse(body, headers=_ACCOUNT_PAGE_HEADERS)

    # FX-A: serve the extracted JS. Same hand-rolled pattern used for
    # /static/telemetry.js (see line ~5984 of this file): read+cache on
    # first hit (with mtime-based invalidation in dev), ETag header for
    # cheap revalidation, application/javascript media type. No mounted
    # StaticFiles directory — only the whitelisted filenames are reachable.
    #
    # FX-P8-A1: factored out for reuse by the 5 Phase 7 console_*.js files.
    # Filename is validated against a strict whitelist to prevent path
    # traversal. Per-file cache is keyed by name.
    from pathlib import Path as _Path
    _STATIC_JS_DIR = _Path(__file__).resolve().parent
    _STATIC_JS_ALLOWLIST = {
        "keys.js",
        "console_settings.js",
        "console_quickstart.js",
        "console_sandbox.js",
        "console_alerts.js",
        "console_webhooks.js",
        "console_key_detail.js",   # Phase 8 Wave D
        "console_nav.js",          # Phase 8 Wave F — shared top-nav activator
        "console_dashboard.js",    # Phase 8 Wave F — dashboard widgets
        "pressed_leaf.js",         # Phase 8 Wave G — shared avatar generator
        "console_account_chip.js", # Phase 8 Wave G — account chip + dropdown
        "odometer.js",             # Phase 8 Wave H — shared digit-odometer
        "apin_toast.js",           # Phase 8 Wave H — global toast system
        "apin_nav_badge.js",       # Phase 8 Wave H — nav unread-count badge
        "console_alert_prefs.js",  # Phase 8 Wave H.D — prefs UI + push prompt
        "apin_sw.js",              # Phase 8 Wave H.D — service worker for push
        "apin_charts.js",          # Phase 9.D — paper-ink chart library
        "console_usage.js",        # Phase 9.E — account-wide Usage page
        "apin_fx.js",              # Phase 9.N.1 — shared animation library
        "apin_lightbox.js",        # Phase 9.N.3 — FLIP-grow lightbox shell
        "apin_live_stream.js",     # Phase 9.N.7 — SSE live-stream client
        "apin_live_pulse.js",      # Phase 9.N.7 — live req/sec chart + commentary
        "apin_syntax.js",          # Phase 9.N.7 — homegrown code highlighter
        "apin_scrubber.js",        # Phase 9.N.11 — time scrubber slider
    }
    _STATIC_JS_CACHE: dict[str, dict] = {}

    @app.get("/static/{filename}")
    async def v2_static_js(filename: str, request: Request):
        from fastapi.responses import Response
        import hashlib
        if filename not in _STATIC_JS_ALLOWLIST:
            return Response(status_code=404,
                            content=f"{filename}: not on allowlist".encode())
        path = _STATIC_JS_DIR / filename
        cache = _STATIC_JS_CACHE.setdefault(
            filename, {"mtime": 0.0, "body": None, "etag": None})
        try:
            mt = path.stat().st_mtime
            if (cache["mtime"] != mt) or (cache["body"] is None):
                cache["body"]  = path.read_bytes()
                cache["mtime"] = mt
                cache["etag"]  = (
                    '"' + hashlib.sha256(cache["body"]).hexdigest()[:12] + '"'
                )
            inm = request.headers.get("if-none-match")
            if inm and inm == cache["etag"]:
                return Response(status_code=304, headers={
                    "ETag": cache["etag"],
                    "Cache-Control": "public, max-age=0, must-revalidate",
                })
            return Response(
                content=cache["body"],
                media_type="application/javascript",
                headers={
                    "Cache-Control": "public, max-age=0, must-revalidate",
                    "ETag": cache["etag"],
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except FileNotFoundError:
            return Response(status_code=404, content=f"{filename} not found".encode())

    # ── SD-1: /account/api -> /account/api/dashboard (spec §15.2) ──
    # Spec §15.2 row 1 mandates that /account/api lands on
    # /account/api/dashboard. The dashboard page itself is a Phase 3.4
    # deliverable; until then we provide a stub that 307s onward to the
    # keys list (natural landing for users with existing keys).
    # Phase 8.F · Dashboard is now a real page. The stub redirect that
    # previously dumped the user at /keys has been replaced with a Console
    # landing page that pulls /api/auth/me + /api/account/keys + the alerts
    # unread count + the audit recent feed.
    @app.get("/account/api")
    async def account_root():
        return RedirectResponse(url="/account/api/dashboard", status_code=307)

    # Phase 8.F · Dashboard HTML page — cached body bytes with CSRF + nav
    # injection (matches the pattern used by the other 7 Console pages).
    _console_dashboard_body_str = _load_html("console_dashboard.html")
    _console_dashboard_body_bytes = (
        _console_dashboard_body_str.encode("utf-8")
        if isinstance(_console_dashboard_body_str, str)
        else _console_dashboard_body_str
    )
    _console_dashboard_body_bytes = _inject_console_chrome(
        _console_dashboard_body_bytes)

    @app.get("/account/api/dashboard", response_class=HTMLResponse)
    async def account_dashboard_page(request: Request):
        return _serve_csrf_page(request, _console_dashboard_body_bytes)

    # Phase 8.F · /api/account/audit/recent — across all keys for the
    # current user. Reuses auth_db.list_audit_log() without a key_id filter.
    # Returns a §3-shaped envelope manually (don't double-wrap via
    # @api_endpoint here — this is a Console-internal helper, not a public
    # surface).
    @app.get("/api/account/audit/recent")
    async def account_audit_recent(request: Request, limit: int = 10):
        from scripts.apin_v2 import auth_db as _adb
        # Authentication: cookie session, same pattern as /api/auth/me.
        raw_session = request.cookies.get("apin_v2_session")
        if not raw_session:
            return JSONResponse({"ok": False,
                                 "error": {"code": "unauthenticated",
                                           "message": "login required"}},
                                status_code=401)
        sess = _adb.lookup_session_by_token(raw_session)
        if not sess or not sess.get("user_id"):
            return JSONResponse({"ok": False,
                                 "error": {"code": "unauthenticated",
                                           "message": "session expired"}},
                                status_code=401)
        try:
            n = max(1, min(int(limit), 50))
        except Exception:
            n = 10
        try:
            events = _adb.list_audit_log(
                user_id=int(sess["user_id"]), key_id=None, limit=n)
        except Exception as _exc:
            logger.warning("audit_recent failed for user=%s: %s",
                           sess.get("user_id"), _exc)
            events = []
        # Normalise field names for the dashboard JS: `at` instead of
        # `timestamp`, `action` left as-is, `public_id` mirrors `key_id`.
        out = []
        for e in events:
            out.append({
                "action": e.get("action"),
                "at": e.get("timestamp"),
                "key_id": e.get("key_id"),
                "public_id": e.get("key_id"),
                "key_name_at_time": e.get("key_name_at_time"),
                "actor_ip": e.get("actor_ip"),
                "detail": (e.get("details")
                           if isinstance(e.get("details"), str)
                           else (e.get("details") or {}).get("note")
                                if isinstance(e.get("details"), dict)
                                else None),
            })
        return {"ok": True, "data": {"events": out, "count": len(out)}}


def _add_perception_routes(app):
    """Register the Phase-2 batch-1 drone perception endpoints."""
    import io
    from dataclasses import asdict
    from PIL import Image as _PIL_Image
    import numpy as np
    import hashlib
    from datetime import datetime, timezone
    from fastapi import Header, Form, Query
    from scripts.apin_v2.api_envelope import (
        api_endpoint, ApiError, paginated)
    from scripts.apin_v2 import auth_db as _auth_db

    diag_db = _load_diagnosis_db()

    # ── Bearer auth gate (same shape used elsewhere) ──────────────────
    def _require_api_key(authorization: str | None,
                          x_api_key:     str | None) -> dict:
        raw = None
        if authorization and authorization.lower().startswith("bearer "):
            raw = authorization[7:].strip()
        elif x_api_key:
            raw = x_api_key.strip()
        if not raw:
            raise ApiError(
                "auth_required",
                "This endpoint requires an API key.",
                hint=("Send Authorization: Bearer <token> or "
                      "X-API-Key: <token>. Mint a key at POST /api/keys."))
        auth = _auth_db.find_api_key(raw)
        if auth is None:
            raise ApiError(
                "auth_invalid",
                "The provided API key is not valid or has been revoked.",
                hint="List your active keys with GET /api/keys; mint a "
                     "new one with POST /api/keys.")
        return auth

    # ── image decode + inference (shared with /api/predict/full) ──────
    def _decode_image_with_sha(data: bytes,
                                *, max_bytes: int = 12_000_000
                                ) -> tuple[np.ndarray, str, int]:
        if not data:
            raise ApiError("invalid_parameter",
                           "Image file is empty.",
                           hint="Upload a non-empty JPEG, PNG, or WebP.",
                           field="file")
        if len(data) > max_bytes:
            raise ApiError(
                "payload_too_large",
                f"Image exceeds the {max_bytes // 1_000_000} MB "
                f"per-file limit (got {len(data) // 1_000_000} MB).",
                hint="Downscale the longest side to ~2048 px and retry.",
                field="file")
        try:
            pil = _PIL_Image.open(io.BytesIO(data)).convert("RGB")
            img = np.array(pil, dtype=np.uint8)
        except Exception:
            logger.exception("scan: image decode failed")
            raise ApiError(
                "unsupported_media_type",
                "Could not decode the uploaded file as an image.",
                hint="Send a JPEG, PNG, or WebP image.")
        sha = hashlib.sha256(data).hexdigest()
        return img, sha, len(data)

    def _convert_np(o):
        if isinstance(o, np.ndarray):              return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)):     return int(o)
        if isinstance(o, dict):
            return {k: _convert_np(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_convert_np(v) for v in o]
        return o

    def _run_apin(img_rgb: np.ndarray) -> dict:
        from scripts.apin.section8_apin_server import get_apin
        result = get_apin().predict(img_rgb)
        try:
            return _convert_np(asdict(result))
        except Exception:
            return _convert_np(
                {k: v for k, v in vars(result).items()
                 if not k.startswith("_")})

    # ── helpers for the /api/scan input parsing ───────────────────────
    # Form-field path (drones with simple HTTP clients) — accept geo as
    # individual form fields. Multipart-with-JSON sidecar (fancier
    # clients) — accept a `geo` JSON field. Either works.
    def _geo_from_form(latitude, longitude, altitude_m, heading_deg,
                       accuracy_m, captured_at, flight_id,
                       persist_image, geo_json_blob) -> tuple[dict, str,
                                                              str, bool]:
        # If geo JSON is provided, prefer it (richer + future-proof for
        # extra fields like camera intrinsics later).
        if geo_json_blob:
            try:
                geo_in = json.loads(geo_json_blob)
            except Exception:
                raise ApiError(
                    "invalid_parameter",
                    "Field 'geo' is not valid JSON.",
                    field="geo")
            if not isinstance(geo_in, dict):
                raise ApiError(
                    "invalid_parameter",
                    "Field 'geo' must be a JSON object.",
                    field="geo")
            cap = geo_in.get("captured_at")
            fid = geo_in.get("flight_id")
            persist = bool(geo_in.get("persist_image"))
        else:
            geo_in = {
                "latitude":    latitude,
                "longitude":   longitude,
                "altitude_m":  altitude_m,
                "heading_deg": heading_deg,
                "accuracy_m":  accuracy_m,
            }
            cap = captured_at
            fid = flight_id
            persist = bool(persist_image)
        geo = _validate_geo(geo_in)
        cap_iso = _parse_iso_captured_at(cap)
        return geo, cap_iso, (fid or None), persist

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /api/scan                                               ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.post("/api/scan", status_code=201)
    @api_endpoint("/api/scan", success_status=201)
    async def v1_scan(
        file: UploadFile = File(...),
        latitude:      str | None = Form(default=None),
        longitude:     str | None = Form(default=None),
        altitude_m:    str | None = Form(default=None),
        heading_deg:   str | None = Form(default=None),
        accuracy_m:    str | None = Form(default=None),
        captured_at:   str | None = Form(default=None),
        flight_id:     str | None = Form(default=None),
        persist_image: str | None = Form(default=None),
        geo:           str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        """Run inference on one drone frame and persist the result.

        Returns the scan record + a single RFC 7946 GeoJSON Feature
        ready to drop into a Mapbox / QGIS / Leaflet layer.
        """
        auth = _require_api_key(authorization, x_api_key)
        geo_parsed, cap_iso, flight_id_in, persist = _geo_from_form(
            latitude, longitude, altitude_m, heading_deg, accuracy_m,
            captured_at, flight_id, persist_image, geo)
        try:
            data = await file.read()
        except Exception:
            logger.exception("/api/scan: upload read failed")
            raise ApiError(
                "invalid_parameter",
                "Could not read the uploaded file.",
                hint="Retry; if it persists the multipart stream may "
                     "be truncated.")
        img, image_sha, n_bytes = _decode_image_with_sha(data)
        loop = asyncio.get_running_loop()
        import time as _t
        t0 = _t.perf_counter()
        result = await loop.run_in_executor(None,
                                             lambda: _run_apin(img))
        elapsed_ms = int((_t.perf_counter() - t0) * 1000)
        # Substitute now() if captured_at was omitted, and remember
        # where the value came from so callers can audit.
        cap_source = "client"
        if not cap_iso:
            cap_iso = (datetime.now(timezone.utc)
                       .isoformat(timespec="milliseconds")
                       .replace("+00:00", "Z"))
            cap_source = "server"
        scan = _auth_db.create_scan(
            user_id=auth["user_id"],
            api_key_id=auth["key_id"],
            flight_id=flight_id_in,
            geo=geo_parsed,
            captured_at=cap_iso,
            image_sha256=image_sha,
            image_bytes=(data if persist else None),
            result=result,
            processing_ms=elapsed_ms)
        feature = _scan_to_geojson_feature(scan, diag_db)
        return {
            "scan":     scan,
            "geojson":  feature,
            "captured_at_source": cap_source,
            "authenticated_as": {
                "user_id":   auth["user_id"],
                "key_id":    auth["key_id"],
                "key_name":  auth["name"],
            },
            "links": {
                "self":      f"/api/scans/{scan['scan_uid']}",
                "with_full_result":
                    f"/api/scans/{scan['scan_uid']}?include=result",
            },
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ POST /api/scan/batch                                         ║
    # ╚══════════════════════════════════════════════════════════════╝
    _BATCH_MAX_SCANS = 16
    _BATCH_MAX_TOTAL_BYTES = 32_000_000

    @app.post("/api/scan/batch", status_code=201)
    @api_endpoint("/api/scan/batch", success_status=201)
    async def v1_scan_batch(
        files: list[UploadFile] = File(...),
        # The geo manifest is a JSON array, one entry per file, in the
        # SAME ORDER. We chose an out-of-band JSON manifest over
        # per-field arrays because multipart form arrays are clumsy and
        # most drone ground-station SDKs already emit JSON.
        manifest:      str | None = Form(default=None),
        flight_id:     str | None = Form(default=None),
        persist_image: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        auth = _require_api_key(authorization, x_api_key)
        if not files:
            raise ApiError("missing_parameter",
                           "Send at least one image in `files`.",
                           field="files")
        if len(files) > _BATCH_MAX_SCANS:
            raise ApiError(
                "payload_too_large",
                f"Batch is limited to {_BATCH_MAX_SCANS} frames per "
                f"request; got {len(files)}.",
                hint=f"Split into batches of <= {_BATCH_MAX_SCANS} and "
                     "call /api/scan/batch repeatedly.",
                field="files",
                details={"limit": _BATCH_MAX_SCANS,
                         "received": len(files)})
        if not manifest:
            raise ApiError(
                "missing_parameter",
                "Field 'manifest' is required for /api/scan/batch.",
                hint='Send a JSON array, one entry per file in the '
                     'SAME order, each like {"latitude":12.3,'
                     '"longitude":76.5,"altitude_m":50,'
                     '"captured_at":"2026-05-23T12:34:56Z"}.',
                field="manifest")
        try:
            man = json.loads(manifest)
        except Exception:
            raise ApiError("invalid_parameter",
                           "Field 'manifest' is not valid JSON.",
                           field="manifest")
        if not isinstance(man, list):
            raise ApiError(
                "invalid_parameter",
                "Field 'manifest' must be a JSON array.",
                field="manifest")
        if len(man) != len(files):
            raise ApiError(
                "invalid_parameter",
                f"Manifest length ({len(man)}) does not match files "
                f"count ({len(files)}).",
                hint="Send one manifest entry per file in the same "
                     "order.",
                field="manifest",
                details={"files":    len(files),
                         "manifest": len(man)})

        loop = asyncio.get_running_loop()
        import time as _t
        items: list[dict] = []
        features: list[dict] = []
        total_bytes = 0
        n_ok = n_err = 0
        persist = bool(persist_image)
        diag_counter: dict = {}
        for i, (f, m) in enumerate(zip(files, man)):
            entry = {"index": i, "filename": f.filename}
            try:
                if not isinstance(m, dict):
                    raise ApiError(
                        "invalid_parameter",
                        f"Manifest entry {i} must be a JSON object.",
                        field=f"manifest[{i}]")
                data = await f.read()
                total_bytes += len(data)
                if total_bytes > _BATCH_MAX_TOTAL_BYTES:
                    raise ApiError(
                        "payload_too_large",
                        ("Total batch upload exceeds the "
                         f"{_BATCH_MAX_TOTAL_BYTES // 1_000_000} MB "
                         "aggregate limit."),
                        hint=f"Send a smaller batch (<= "
                             f"{_BATCH_MAX_TOTAL_BYTES // 1_000_000} "
                             "MB total).",
                        field="files",
                        details={
                            "aggregate_limit_bytes":
                                _BATCH_MAX_TOTAL_BYTES,
                            "received_so_far_bytes": total_bytes,
                            "stopped_at_index": i})
                img, image_sha, _ = _decode_image_with_sha(data)
                geo = _validate_geo(m)
                cap_iso = _parse_iso_captured_at(m.get("captured_at"))
                if not cap_iso:
                    cap_iso = (datetime.now(timezone.utc)
                               .isoformat(timespec="milliseconds")
                               .replace("+00:00", "Z"))
                t0 = _t.perf_counter()
                result = await loop.run_in_executor(
                    None, lambda im=img: _run_apin(im))
                elapsed_ms = int((_t.perf_counter() - t0) * 1000)
                scan = _auth_db.create_scan(
                    user_id=auth["user_id"],
                    api_key_id=auth["key_id"],
                    flight_id=(m.get("flight_id") or flight_id),
                    geo=geo,
                    captured_at=cap_iso,
                    image_sha256=image_sha,
                    image_bytes=(data if persist else None),
                    result=result,
                    processing_ms=elapsed_ms)
                feature = _scan_to_geojson_feature(scan, diag_db)
                features.append(feature)
                entry.update({"ok": True,
                              "scan_uid": scan["scan_uid"],
                              "diagnosis": scan["diagnosis"],
                              "confidence": scan["confidence"]})
                if scan["diagnosis"]:
                    diag_counter[scan["diagnosis"]] = (
                        diag_counter.get(scan["diagnosis"], 0) + 1)
                n_ok += 1
            except ApiError as e:
                entry.update({"ok": False,
                              "error": {"code": e.code,
                                        "message": e.message,
                                        "hint": e.hint}})
                n_err += 1
                if e.code == "payload_too_large":
                    items.append(entry)
                    break
            except Exception:
                logger.exception(
                    "/api/scan/batch: frame %d failed", i)
                entry.update({"ok": False,
                              "error": {"code": "inference_failed",
                                        "message": ("Inference failed "
                                                    "for this frame."),
                                        "hint": ("Retry; if persistent, "
                                                 "quote the request_id "
                                                 "in support.")}})
                n_err += 1
            items.append(entry)

        feature_collection = {
            "type": "FeatureCollection",
            "features": features,
        }
        return paginated(
            items, page=1, page_size=len(items), total=len(items),
            feature_collection=feature_collection,
            ok_count=n_ok,
            error_count=n_err,
            diagnosis_counts=diag_counter,
            authenticated_as={"user_id":  auth["user_id"],
                              "key_id":   auth["key_id"],
                              "key_name": auth["name"]})

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ GET /api/scans                                               ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.get("/api/scans")
    @api_endpoint("/api/scans")
    async def v1_scans_list(
        page: int             = Query(default=1, ge=1),
        page_size: int        = Query(default=50, ge=1, le=200),
        flight_id: str | None = Query(default=None),
        diagnosis: str | None = Query(default=None),
        since:     str | None = Query(default=None,
                                        description="ISO-8601 lower bound on captured_at"),
        until:     str | None = Query(default=None,
                                        description="ISO-8601 upper bound on captured_at"),
        as_geojson: int       = Query(default=0,
                                        description="If 1, include a FeatureCollection"),
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        auth = _require_api_key(authorization, x_api_key)
        rows, total = _auth_db.list_scans(
            auth["user_id"],
            page=page, page_size=page_size,
            flight_id=flight_id, diagnosis=diagnosis,
            since_iso=since, until_iso=until)
        extra: dict = {
            "filters": {
                "flight_id": flight_id,
                "diagnosis": diagnosis,
                "since":     since,
                "until":     until,
            },
            "owner": {"user_id": auth["user_id"]},
        }
        if as_geojson:
            extra["feature_collection"] = {
                "type": "FeatureCollection",
                "features": [_scan_to_geojson_feature(r, diag_db)
                             for r in rows],
            }
        return paginated(rows, page=page, page_size=page_size,
                          total=total, **extra)

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ GET /api/scans/{scan_uid}                                    ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.get("/api/scans/{scan_uid}")
    @api_endpoint("/api/scans/{scan_uid}")
    async def v1_scan_get(
        scan_uid: str,
        include:  str | None = Query(default=None,
                                       description="Comma-separated extras: 'result', 'image'"),
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        auth = _require_api_key(authorization, x_api_key)
        extras = {p.strip() for p in (include or "").split(",") if p.strip()}
        scan = _auth_db.get_scan(
            auth["user_id"], scan_uid,
            include_image=("image"  in extras),
            include_result=("result" in extras))
        if scan is None:
            raise ApiError(
                "not_found",
                f"No scan with uid {scan_uid!r} on your account.",
                hint="List your scans with GET /api/scans.",
                field="scan_uid")
        return {
            "scan":    scan,
            "geojson": _scan_to_geojson_feature(scan, diag_db),
        }

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ DELETE /api/scans/{scan_uid}                                 ║
    # ╚══════════════════════════════════════════════════════════════╝
    @app.delete("/api/scans/{scan_uid}")
    @api_endpoint("/api/scans/{scan_uid}")
    async def v1_scan_delete(
        scan_uid: str,
        authorization: str | None = Header(default=None),
        x_api_key:     str | None = Header(default=None,
                                            alias="X-API-Key"),
    ):
        auth = _require_api_key(authorization, x_api_key)
        ok = _auth_db.delete_scan(auth["user_id"], scan_uid)
        if not ok:
            raise ApiError(
                "not_found",
                f"No active scan with uid {scan_uid!r} on your account.",
                hint="GET /api/scans to see your active scans.",
                field="scan_uid")
        return {
            "scan_uid":   scan_uid,
            "deleted":    True,
            "deleted_at": (datetime.now(timezone.utc)
                           .isoformat(timespec="milliseconds")
                           .replace("+00:00", "Z")),
            "note": ("Scan was soft-deleted (deleted_at timestamp set). "
                     "It no longer appears in list/get queries; admin "
                     "tooling can restore it from the DB if needed."),
        }

    logger.info("APIN /api/ perception routes registered - /api/scan "
                "/api/scan/batch /api/scans /api/scans/{uid} GET+DELETE")


# ══════════════════════════════════════════════════════════════════════
# Pipeline Atlas (Phase 4) · the /pipeline page + GET /api/recent
# ══════════════════════════════════════════════════════════════════════
# The Pipeline Atlas is APIN's architecture page. It explains the
# 4-module pipeline (Router · Tomato · Chilli · APIN okra+brassica)
# with interactive visualisations, animated charts, and a live request
# log driven by the new /api/recent endpoint.
#
# The middleware below records EVERY request the server handles into a
# bounded in-process deque, skipping static assets. The /api/recent
# endpoint reads from that deque. Auth is intentionally absent: the log
# only carries path templates, status codes, and latency, never request
# bodies or auth headers, so it is safe to expose.

from collections import deque as _deque
import time as _pa_time

_RECENT_REQUESTS_BUFFER = _deque(maxlen=200)
_RECENT_SKIP_PREFIXES = (
    "/favicon", "/logo", "/static/", "/robots.txt", "/sitemap.xml",
    "/api/recent",   # don't pollute the log with self-polls
)
_RECENT_SKIP_SUFFIXES = (
    ".css", ".js", ".png", ".svg", ".ico", ".jpg", ".jpeg",
    ".webp", ".woff", ".woff2", ".ttf", ".map",
)


def _add_pipeline_route(app):
    """Register the Pipeline Atlas page and its supporting /api/recent
    endpoint, plus the request-capture middleware.

    Purely additive. The middleware is read-only: it observes every
    request, never modifies request or response, and any internal error
    is swallowed so it cannot break inference.
    """
    # PDA round 1: BaseHTTPMiddleware import removed (dead - we use the
    # @app.middleware("http") decorator, not the BaseHTTPMiddleware class).
    from datetime import datetime as _pa_dt, timezone as _pa_tz
    from fastapi import Query, HTTPException
    import re as _pa_re
    import os as _pa_os
    import json as _pa_json

    # Sanitiser for unmatched paths (404s). When FastAPI does not match a
    # route template we still receive a rendered URL which can carry
    # user-supplied IDs (scn_..., apin_..., long numeric ids). We
    # collapse those to {id} so the public log never echoes raw values.
    _PATH_SANITISE_PATTERNS = [
        (_pa_re.compile(r"/(scn|apin|req)_[0-9a-f]{8,}"), r"/\1_{id}"),
        # PDA round 5: was `/\d{4,}` which missed short ids like
        # /api/keys/42. Lower bound to 1+ so any all-digit segment
        # gets collapsed.
        (_pa_re.compile(r"/\d+"), "/{id}"),
    ]

    def _sanitise_raw_path(p):
        out = p
        for pat, repl in _PATH_SANITISE_PATTERNS:
            out = pat.sub(repl, out)
        return out

    # ── middleware ────────────────────────────────────────────────────
    @app.middleware("http")
    async def _capture_recent(request, call_next):
        path = request.url.path
        method = request.method
        # Skip static assets + our own polling endpoint
        if any(path.startswith(p) for p in _RECENT_SKIP_PREFIXES):
            return await call_next(request)
        if any(path.endswith(s) for s in _RECENT_SKIP_SUFFIXES):
            return await call_next(request)

        t0 = _pa_time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            try:
                elapsed_ms = int(
                    (_pa_time.perf_counter() - t0) * 1000)
                # Prefer the route template if FastAPI matched one;
                # falls back to a sanitised raw path on 404 / static so
                # the log never echoes user-supplied IDs verbatim
                # (PDA round 1 finding).
                route = request.scope.get("route", None)
                path_template = (getattr(route, "path", None)
                                 or _sanitise_raw_path(path))
                _RECENT_REQUESTS_BUFFER.appendleft({
                    "captured_at": (_pa_dt.now(_pa_tz.utc)
                                    .isoformat(timespec="milliseconds")
                                    .replace("+00:00", "Z")),
                    "method": method,
                    "path": path_template,
                    "status": status,
                    "latency_ms": elapsed_ms,
                })
            except Exception:
                # Never let the middleware crash a request
                pass

    # ── GET /api/recent ───────────────────────────────────────────────
    from scripts.apin_v2.api_envelope import api_endpoint, paginated

    @app.get("/api/recent")
    @api_endpoint("/api/recent")
    async def api_recent(
        limit: int = Query(default=50, ge=1, le=200,
                            description="Number of recent requests to return (1-200, default 50)."),
    ):
        """Return the N most recent HTTP requests this process has
        handled. Skips static assets. Carries path templates only;
        never request bodies, auth tokens, or user identifiers."""
        # FastAPI's Query(ge=1, le=200) already validates limit, so we
        # trust it directly. The earlier defensive clamp was dead code.
        n = limit
        rows = list(_RECENT_REQUESTS_BUFFER)[:n]
        return paginated(rows, page=1, page_size=n, total=len(rows),
                          buffer_capacity=_RECENT_REQUESTS_BUFFER.maxlen)

    # ── GET /pipeline (the page) ──────────────────────────────────────
    _PIPELINE_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }

    @app.get("/pipeline", response_class=HTMLResponse)
    async def pipeline_page():
        return HTMLResponse(_load_html("pipeline.html"),
                            headers=_PIPELINE_HEADERS)

    # Phase 4B: serve the precomputed module data JSON files. These are written
    # by scripts/apin_v2/extract_pipeline_atlas_{router,tomato,chilli}.py. The
    # files live under _qa_tmp/ (outside the public assets dir) so the route
    # reads them from disk and returns them with no-store. Path traversal is
    # impossible because we whitelist exact filenames.
    _ATLAS_DATA_FILES = {
        "router":  "_pipeline_atlas_router.json",
        "tomato":  "_pipeline_atlas_tomato.json",
        "chilli":  "_pipeline_atlas_chilli.json",
        # Phase 4C: animated forward-pass diagram data (4 models × 3 images
        # × N stages, with hand-drafted narrations). Appended last per plan
        # v4 §A15 so the not_found error hint enumerates slugs in insertion
        # order. ~2-4 MB payload; served with the same no-store header as
        # all other slugs.
        "forward_pass": "_pipeline_atlas_forward_pass.json",
        # Phase 4D: calibration deep dive + scattered fills + lineage
        # data computed from real cached results by extract_phase_d.py.
        "phase_d":      "_pipeline_atlas_phase_d.json",
    }
    _ATLAS_ROOT = _pa_os.path.dirname(_pa_os.path.dirname(
        _pa_os.path.dirname(_pa_os.path.abspath(__file__))
    ))
    _ATLAS_QA_DIR = _pa_os.path.join(_ATLAS_ROOT, "_qa_tmp")

    # Use the project's ApiError convention so the envelope shape stays
    # consistent with every other /api/* endpoint.
    from scripts.apin_v2.api_envelope import ApiError as _PA_ApiError

    @app.get("/api/pipeline_data/{slug}")
    @api_endpoint("/api/pipeline_data/{slug}")
    async def pipeline_data(slug: str):
        fname = _ATLAS_DATA_FILES.get(slug)
        if fname is None:
            raise _PA_ApiError(
                "not_found",
                f"Unknown atlas data slug: {slug!r}",
                hint="Valid slugs: " + ", ".join(_ATLAS_DATA_FILES.keys()),
                field="slug",
            )
        path = _pa_os.path.join(_ATLAS_QA_DIR, fname)
        if not _pa_os.path.exists(path):
            raise _PA_ApiError(
                "not_found",
                f"Atlas data not extracted yet for {slug}.",
                hint=f"Run python scripts/apin_v2/extract_pipeline_atlas_{slug}.py",
                field="slug",
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = _pa_json.load(f)
        except Exception as e:
            raise _PA_ApiError(
                "internal_error",
                "Atlas data file is unreadable.",
                hint=str(e)[:200],
            )
        return payload

    logger.info("Pipeline Atlas registered · /pipeline page + "
                "GET /api/recent endpoint + GET /api/pipeline_data/{slug} + "
                "recent-requests middleware")


def _add_v1_validation_handler(app):
    """Wrap FastAPI's RequestValidationError into the §3 error envelope.

    PVA drift fix: without this, calling POST /predict/quick with no
    file (or /predict/batch with no files) returned FastAPI's default
    raw {"detail":[{"type":"missing",…}]} body — a totally different
    shape from every other §3 error response. Machine clients that
    branch on `body.error.code` would crash with KeyError on a missing
    field.

    Scope: ONLY pydantic body/query/header validation failures hit
    this handler. The existing browser-facing endpoints (/predict/full,
    /auth/*, /feedback, ...) parse their bodies manually and either
    succeed or raise HTTPException — neither path goes through this
    handler, so this is a strictly additive change.
    """
    from fastapi.exceptions import RequestValidationError
    from scripts.apin_v2.api_envelope import (
        new_request_id, API_VERSION, ERROR_STATUS)
    from datetime import datetime as _vh_dt, timezone as _vh_tz

    _DOCS = "https://dxv-404-apin.hf.space/docs#errors"

    @app.exception_handler(RequestValidationError)
    async def _validation_failed(request: Request,
                                  exc: RequestValidationError):
        rid = new_request_id()
        # Surface the first failing field; the full list is in details
        # so the client can show every offense.
        errs    = exc.errors() or []
        first   = errs[0] if errs else {}
        loc     = first.get("loc", []) or []
        # loc looks like ("body","file") or ("query","page")
        field   = ".".join(str(x) for x in loc[1:]) if len(loc) > 1 else None
        msg     = first.get("msg", "Validation failed.")
        # Trim the per-error dicts to the safe subset — never echo
        # `ctx`, which can contain raw exception text from validators.
        safe_errs = [
            {"loc":  e.get("loc"),
             "msg":  e.get("msg"),
             "type": e.get("type")}
            for e in errs
        ]
        # FX-P8-A2: ERROR_STATUS has no "validation_failed" key — the dict
        # only holds canonical spec codes + legacy codes. Pre-fix this
        # crashed with a KeyError and the user saw a generic 500. Map
        # FastAPI's RequestValidationError to the canonical `invalid_parameter`
        # envelope (400) so callers get a clean response with the offending
        # field surfaced in details.
        try:
            status = ERROR_STATUS["invalid_parameter"]
        except KeyError:
            status = 400
        code_for_envelope = "invalid_parameter"
        now_iso = (_vh_dt.now(_vh_tz.utc).isoformat(timespec="milliseconds")
                   .replace("+00:00", "Z"))
        env = {
            "api_version":         API_VERSION,
            "request_id":          rid,
            "endpoint":            request.url.path,
            "ok":                  False,
            "processed_at":        now_iso,
            "processing_time_ms":  0,
            "status_code":         status,
            "error": {
                "code":     code_for_envelope,
                "message":  ("Request did not match the expected shape: "
                              + str(msg)),
                "docs_url": _DOCS,
                "hint":     ("Check the field listed in `error.field` "
                              "and the full list in `error.details.errors`."),
                "field":    field,
                "details":  {"errors": safe_errs},
            },
        }
        return JSONResponse(env, status_code=status,
                            headers={"X-Request-Id": rid,
                                     "X-API-Version": API_VERSION,
                                     "Cache-Control": "no-store"})

    logger.info("APIN /api/ validation handler registered - "
                "RequestValidationError -> §3 envelope")


# ════════════════════════════════════════════════════════════════════
# Stage 2 · KPI summary + telemetry ingest routes
#
# /api/stats/summary  — composes the 4 KPI tiles (inferences served,
#                       top disease this week, test quality, live activity)
#                       from auth_db helpers; cache-friendly read-only.
#
# /api/telemetry/ingest — accepts a batch payload (page_views, clicks,
#                         impressions, events, api_calls, errors, goals,
#                         experiments_exposures, plus an optional session
#                         object) and forwards it to ingest_telemetry_batch.
#                         Never blocks the user — returns 200 with per-table
#                         insert counts even on malformed payloads.
# ════════════════════════════════════════════════════════════════════
def _add_stats_and_telemetry_routes(app):
    """Register the KPI summary endpoint, the telemetry ingest beacon,
    and (Stage 6.2) the Server-Sent Events live-stream endpoint."""
    from fastapi.responses import JSONResponse, StreamingResponse
    from scripts.apin_v2 import auth_db as _auth_db
    import asyncio as _asyncio
    import json as _json_mod

    # ════════════════════════════════════════════════════════════════════
    # Stage 6.2 · Real-time Live Now feed via Server-Sent Events
    # ════════════════════════════════════════════════════════════════════
    # Why SSE replaced the 2 s poll: Chrome (and every Chromium-based
    # browser) clamps setInterval in hidden tabs to >=1 Hz, and after 5
    # minutes hidden invokes "intensive throttling" — 1 fire / minute
    # at most. So when /pipeline was the observer tab and the user
    # switched away, the count visibly lagged by 2-5 minutes on return.
    #
    # SSE is server-driven push. The browser does NOT throttle event
    # delivery the way it throttles timers; events emitted to a hidden
    # tab queue locally and fire as soon as the tab returns to visible.
    # Combined with the visibility-resume manual refetch on the client
    # (defensive belt-and-suspenders for the rare case where the SSE
    # connection died while hidden) this gives the tile sub-second
    # update latency in every realistic browser state.
    #
    # Architecture:
    #   _LiveNowBus         — process-local pub-sub. Subscribers are
    #                         bounded asyncio.Queue objects (max 8) so a
    #                         slow consumer cannot exhaust memory.
    #   _build_live_now_snapshot()
    #                       — builds the same shape the polling endpoint
    #                         returns under .live_now. Single source of
    #                         truth — both endpoints call it.
    #   _publish_live_now() — convenience wrapper: snapshot + fan out.
    #   _live_now_sweeper() — 2 s periodic task that re-publishes
    #                         whenever the signature changes for reasons
    #                         OTHER than a telemetry beacon (a heartbeat
    #                         aging past the window, the GPU memory
    #                         crossing a threshold, etc.). Without this
    #                         the count would freeze when sessions go
    #                         silent (the explicit pagehide beacon
    #                         covers graceful close but not browser
    #                         crashes / OS kills / mobile suspend).
    #
    # Production notes:
    #   · X-Accel-Buffering: no  — disables Nginx response buffering on
    #     the HF Space upstream path. Without it the edge layer holds
    #     individual events and ships them in 5+ second batches.
    #   · 15 s SSE comment-ping  — keeps the connection alive against
    #     proxies that drop idle streams. HF Space's edge times out
    #     around 60 s of silence; 15 s is comfortably under that.
    #   · Bounded queue + drop-oldest backpressure — one bad client
    #     cannot block the publisher loop or balloon memory.
    #   · subscribe()/unsubscribe() are async + lock-protected so the
    #     subscriber set is safe under concurrent connects + disconnects.
    #   · Sweeper is idempotently started · the first SSE connection
    #     triggers it via _ensure_sweeper(), so it does not even spin
    #     up on a server nobody is observing.
    # ════════════════════════════════════════════════════════════════════

    class _LiveNowBus:
        """In-memory pub-sub for live_now snapshots. One instance per app."""
        def __init__(self):
            self._subscribers: set = set()
            self._lock = _asyncio.Lock()

        async def subscribe(self) -> _asyncio.Queue:
            q: _asyncio.Queue = _asyncio.Queue(maxsize=8)
            async with self._lock:
                self._subscribers.add(q)
            return q

        async def unsubscribe(self, q: _asyncio.Queue) -> None:
            async with self._lock:
                self._subscribers.discard(q)

        async def publish(self, snapshot: dict) -> None:
            async with self._lock:
                subs = list(self._subscribers)
            for q in subs:
                try:
                    q.put_nowait(snapshot)
                except _asyncio.QueueFull:
                    # Slow consumer · drop oldest, append new. Keeps the
                    # subscriber alive but ensures they get the LATEST
                    # state next read instead of an arbitrarily-old one.
                    try: q.get_nowait()
                    except Exception: pass
                    try: q.put_nowait(snapshot)
                    except Exception: pass

        def subscriber_count(self) -> int:
            return len(self._subscribers)

    if not hasattr(app.state, "_live_now_bus"):
        app.state._live_now_bus = _LiveNowBus()
        app.state._live_now_sweeper_started = False
    _bus: _LiveNowBus = app.state._live_now_bus

    def _build_live_now_snapshot() -> dict:
        """Compose the live_now object the polling endpoint also returns."""
        try:
            ln = _auth_db.live_sessions_by_route(window_s=30)
        except Exception:
            ln = {"active_count": 0, "by_route": [], "as_of": None}
        compute = {"device": "cpu", "vram_gb": None, "vram_allocated_gb": None}
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                compute["device"] = "cuda"
                try:
                    compute["vram_allocated_gb"] = round(
                        _torch.cuda.memory_allocated() / (1024 ** 3), 2)
                except Exception:
                    pass
                try:
                    props = _torch.cuda.get_device_properties(0)
                    compute["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
                    compute["device_name"] = props.name
                except Exception:
                    pass
        except Exception:
            pass
        ln["compute"] = compute
        return ln

    async def _publish_live_now() -> None:
        try:
            snap = _build_live_now_snapshot()
            await _bus.publish(snap)
        except Exception:
            logger.exception("live_now publish failed")

    async def _live_now_sweeper() -> None:
        """Periodic re-publisher · catches state changes the ingest hook
        cannot see (sessions aging out of the 30 s heartbeat window with
        no new telemetry arriving). Signature is (active_count, sorted
        route counts, vram_allocated_gb) — any change re-publishes."""
        last_sig = None
        while True:
            try:
                snap = _build_live_now_snapshot()
                sig = (
                    snap.get("active_count"),
                    tuple(sorted(
                        ((r.get("route"), r.get("count"))
                         for r in (snap.get("by_route") or [])),
                    )),
                    (snap.get("compute") or {}).get("vram_allocated_gb"),
                )
                if sig != last_sig:
                    await _bus.publish(snap)
                    last_sig = sig
            except Exception:
                logger.exception("live_now sweeper iteration failed")
            await _asyncio.sleep(2.0)

    def _ensure_live_now_sweeper() -> None:
        if app.state._live_now_sweeper_started:
            return
        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            return
        app.state._live_now_sweeper_started = True
        _asyncio.create_task(_live_now_sweeper())
        logger.info("live_now SSE sweeper started (2 s signature-change interval)")

    @app.on_event("startup")
    async def _live_now_startup_hook():
        _ensure_live_now_sweeper()

    @app.get("/api/stats/summary")
    async def v2_stats_summary(window_days: int = 7):
        """Aggregate snapshot for the Pipeline Atlas KPI tiles.

        Returns:
          inferences_served : {total, user, guest, scan}
          top_disease       : {class, count, share, window_days} or null
          live_activity     : {last_60s_count, last_5min_count,
                               last_hour_count, median_latency_ms,
                               error_rate_pct, as_of}
          test_quality      : (placeholder; populated by Phase 4D extractor)

        The endpoint is intentionally cheap — it issues four COUNT(*) /
        GROUP BY queries against indexed columns. Safe to poll every 5-10s.
        """
        try:
            totals = _auth_db.count_total_inferences()
        except Exception:
            totals = {"total": 0, "user": 0, "guest": 0, "scan": 0}
        try:
            wnd = int(window_days)
            if wnd < 1 or wnd > 365:
                wnd = 7
            top = _auth_db.top_disease_in_window(days=wnd)
        except Exception:
            top = None
        try:
            live = _auth_db.live_activity_summary()
        except Exception:
            live = None

        # Stage 6.2 · delegate to the shared snapshot builder so the
        # polling endpoint and the SSE stream return BIT-IDENTICAL shapes.
        # window_s defaults to 30 s now (was 90 s in Stage 6) — the SSE
        # push is the primary liveness signal, so the SQL window only
        # needs to cover the "client died silently" fallback case, and
        # 30 s gives much faster cleanup of crashed/force-quit tabs.
        live_now = _build_live_now_snapshot()

        # test_quality is sourced from the Phase 4D extractor's frozen JSON.
        # The Pipeline Atlas already inlines PHASE_D_DATA; the API mirror is
        # provided for clients that want to drive the same tile without
        # re-parsing the HTML.
        test_quality = None
        try:
            from pathlib import Path as _P
            import json as _json
            _td = (PROJECT_ROOT / "scripts" / "apin_v2"
                   / "phase_d_data.json")
            if _td.exists():
                _d = _json.loads(_td.read_text(encoding="utf-8"))
                # Surface a compact pair: ECE + macro-F1 (or whatever the
                # extractor labels as the headline quality numbers).
                test_quality = {
                    "macro_f1": _d.get("macro_f1") or _d.get("test_macro_f1"),
                    "ece":      _d.get("ece")      or _d.get("test_ece"),
                    "n_test":   _d.get("n_test")   or _d.get("test_n"),
                    "as_of":    _d.get("generated_at")
                                or _d.get("as_of"),
                }
        except Exception:
            test_quality = None

        return JSONResponse({
            "inferences_served": totals,
            "top_disease":       top,
            "live_activity":     live,
            "test_quality":      test_quality,
            "live_now":          live_now,   # Stage 6 · tile 4
        }, headers={"Cache-Control": "no-store"})

    @app.get("/api/stats/live-stream")
    async def v2_stats_live_stream(request: Request):
        """Server-Sent Events feed for the Live Now tile.

        Emits a `snapshot` event with the same shape as
        /api/stats/summary's `live_now` field:
          { active_count, by_route[], compute{}, as_of }

        Events fire:
          · immediately on connect (initial state — client doesn't wait)
          · whenever a telemetry beacon causes a session change (instant)
          · whenever the 2 s sweeper detects a state change for reasons
            other than a beacon (stale eviction · GPU memory change ·
            etc.)

        SSE comment-ping every 15 s prevents proxy idle-timeout. The
        EventSource API on the client side auto-reconnects on drop.

        Production deployment note: this works on HF Spaces because we
        emit X-Accel-Buffering: no, which disables Nginx response
        buffering on the upstream path. Without that header, individual
        events are batched into 5+ second flushes.
        """
        _ensure_live_now_sweeper()
        q = await _bus.subscribe()

        async def gen():
            try:
                # 1) Initial snapshot — sent on connect so the client UI
                #    populates immediately without waiting for the next
                #    change event.
                try:
                    initial = _build_live_now_snapshot()
                    yield f"event: snapshot\ndata: {_json_mod.dumps(initial)}\n\n"
                except Exception:
                    pass

                # 2) Stream changes. timeout=15 wakes the loop so we can
                #    (a) check is_disconnected() and (b) emit a comment
                #    keep-alive.
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        snap = await _asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"event: snapshot\ndata: {_json_mod.dumps(snap)}\n\n"
                    except _asyncio.TimeoutError:
                        # SSE protocol · lines starting with `:` are
                        # comments. Maintains the connection through
                        # idle-timeout-aggressive proxies.
                        yield ": keep-alive\n\n"
            finally:
                await _bus.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-store",
                "X-Accel-Buffering": "no",     # HF Space / Nginx · disable buffering
                "Connection":        "keep-alive",
            },
        )

    @app.post("/api/telemetry/ingest")
    async def v2_telemetry_ingest(request: Request):
        """Accept a batch of client-side telemetry events.

        Expected payload shape (every key optional, missing → empty):
        {
          "session": {
            "id": "<browser_session_id>",
            "session_start_at": "<iso>",
            "device_type": "...",
            "ip_country": "...",
            ...
          },
          "page_views":            [{...}, ...],
          "clicks":                [{...}, ...],
          "impressions":           [{...}, ...],
          "events":                [{...}, ...],
          "api_calls":             [{...}, ...],
          "inference_telemetry":   [{...}, ...],
          "errors":                [{...}, ...],
          "goals":                 [{...}, ...],
          "experiments_exposures": [{...}, ...]
        }

        Returns per-table insert counts. Malformed rows in a batch are
        silently dropped — the rest are still inserted, mirroring how
        real-world telemetry pipelines treat partial failures.
        """
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return JSONResponse(
                {"ok": False, "reason": "expected JSON object",
                 "counts": None},
                status_code=200,
                headers={"Cache-Control": "no-store"},
            )

        # [F-5] Cap per-table batch size. A malicious client could otherwise
        # POST millions of events in one request. The cap is per-table so a
        # well-behaved client sending 200 page_views + 200 clicks still works.
        _BATCH_CAP = 500
        _truncated = False
        for _k in ("page_views", "clicks", "impressions", "events",
                   "api_calls", "inference_telemetry", "errors",
                   "goals", "experiments_exposures"):
            _items = body.get(_k)
            if isinstance(_items, list) and len(_items) > _BATCH_CAP:
                body[_k] = _items[:_BATCH_CAP]
                _truncated = True

        # Best-effort server-side enrichment of the session object: stamp
        # IP country / region / city from edge headers if the client
        # didn't already provide them, plus a salted IP hash. This way
        # the client never has to know the visitor's geo or IP.
        try:
            sess = body.get("session")
            if isinstance(sess, dict):
                if "client_ip_hash" not in sess or not sess.get("client_ip_hash"):
                    try:
                        # REV-R5-I08: rightmost-untrusted via helper, not
                        # leftmost XFF entry.
                        ip = _client_ip_rightmost(request)
                        ih = _hash_client_ip(ip)
                        if ih:
                            sess["client_ip_hash"] = ih
                    except Exception:
                        pass
                geo = _client_geo_from_headers(request.headers)
                for k in ("client_country", "client_region", "client_city"):
                    if k in geo and not sess.get(k.replace("client_", "ip_")):
                        # browser_sessions column is `ip_country` etc.
                        sess[k.replace("client_", "ip_")] = geo[k]
        except Exception:
            pass

        try:
            counts = _auth_db.ingest_telemetry_batch(body)
        except Exception:
            logger.exception("ingest_telemetry_batch raised — returning zero counts")
            counts = {
                "session_upserted": 0,
                "page_views": 0, "clicks": 0, "impressions": 0,
                "events": 0, "api_calls": 0, "inference_telemetry": 0,
                "errors": 0, "goals": 0, "experiments_exposures": 0,
            }

        # Stage 6.2 · push fresh snapshot to all SSE subscribers the
        # instant a session row changed. This is the primary realtime
        # signal: a tab opening, heartbeating, changing route, or
        # ending propagates to every /pipeline observer in <1 hop
        # round-trip. Sweeper handles state changes that DIDN'T arrive
        # via telemetry (silent client deaths) on a 2 s cadence.
        # Skipped when no session content arrived (counts.session_upserted = 0),
        # which avoids pushing duplicate snapshots when a batch only
        # contained click/api_call rows.
        if counts.get("session_upserted"):
            try:
                await _publish_live_now()
            except Exception:
                logger.warning("SSE publish after ingest failed", exc_info=True)

        return JSONResponse(
            {"ok": True, "counts": counts, "truncated": _truncated},
            headers={"Cache-Control": "no-store"},
        )

    # Stage 3 · serve the client-side telemetry library at /static/telemetry.js
    #
    # We use a hand-rolled GET route (rather than mounting a StaticFiles
    # directory) for three reasons:
    #   1. Only one file in scripts/apin_v2/ should be reachable from the
    #      web. Mounting the whole directory would expose every adjacent
    #      .py module.
    #   2. We want strong caching headers so the browser does not re-fetch
    #      the same ~14 KB blob on every navigation.
    #   3. The Content-Type must be application/javascript for the browser
    #      to execute it; FastAPI's FileResponse picks text/javascript on
    #      some platforms which works but is technically the older mimetype.
    _TLM_FILE = Path(__file__).resolve().parent / "telemetry.js"
    _TLM_CACHE = {"mtime": 0.0, "body": None, "etag": None}

    @app.get("/static/telemetry.js")
    async def v2_telemetry_js(request: Request):
        from fastapi.responses import Response
        try:
            mt = _TLM_FILE.stat().st_mtime
            if (_TLM_CACHE["mtime"] != mt) or (_TLM_CACHE["body"] is None):
                _TLM_CACHE["body"]  = _TLM_FILE.read_bytes()
                _TLM_CACHE["mtime"] = mt
                # ETag = first 12 hex chars of sha256(content). Cheap
                # invariant: any byte change → new ETag → forced re-download.
                import hashlib
                _TLM_CACHE["etag"] = (
                    '"' + hashlib.sha256(_TLM_CACHE["body"]).hexdigest()[:12] + '"'
                )
            # Stage 5 [post-mortem] · the 1-hour cache header that lived
            # here meant the browser served a STALE telemetry.js after we
            # shipped a bug fix; the in-browser walkthrough only worked
            # because we manually busted the cache via fetch+eval. Switch
            # to revalidate-every-time + ETag so deploys propagate
            # immediately while still saving bandwidth on unchanged files.
            inm = request.headers.get("if-none-match")
            if inm and inm == _TLM_CACHE["etag"]:
                return Response(status_code=304, headers={
                    "ETag": _TLM_CACHE["etag"],
                    "Cache-Control": "public, max-age=0, must-revalidate",
                })
            return Response(
                content=_TLM_CACHE["body"],
                media_type="application/javascript",
                headers={
                    "Cache-Control": "public, max-age=0, must-revalidate",
                    "ETag": _TLM_CACHE["etag"],
                    "X-APIN-Telemetry-Version": "v1",
                },
            )
        except FileNotFoundError:
            return Response(status_code=404, content=b"telemetry.js not found")

    logger.info("Stage 2 + 6.2 routes registered — /api/stats/summary + "
                "/api/stats/live-stream (SSE) + /api/telemetry/ingest + "
                "/static/telemetry.js")

    # The auto-inject HTML middleware lives in
    # _add_telemetry_inject_middleware() (called from main BEFORE gzip)
    # — see comment block there for the rationale.


# ════════════════════════════════════════════════════════════════════
# Stage 6 follow-up · auto-inject the telemetry library into every HTML
# page so visits to /docs, /, /landing, /dashboard, etc. all count
# toward the "Live now" KPI tile.
#
# Why this middleware lives at module-scope (and is registered in
# main() BEFORE GZipMiddleware): if gzip processes the response first,
# the response body is compressed bytes by the time we see it · UTF-8
# decode fails · we skip injection. Registering this middleware EARLIER
# (so it's the INNERMOST on the response phase, seen first going out)
# means we always get the raw, uncompressed HTML, modify it, then gzip
# compresses the modified version downstream of us.
#
# Why a middleware rather than touching every HTML file:
#   1. 13 HTML pages. Edit-once vs edit-everywhere is a much smaller
#      maintenance surface.
#   2. The cache-bust query (`?v=<etag>`) lives in ONE place. When
#      telemetry.js changes byte-for-byte its ETag flips, the script
#      URL changes, every browser refetches automatically — no manual
#      version bumping required.
#   3. Idempotent: if a page already has /static/telemetry.js in its
#      HTML (pipeline.html does), we just rewrite the URL to add the
#      version query · no duplicate <script> tags.
# ════════════════════════════════════════════════════════════════════
# ⚠ DEAD CODE — DO NOT REGISTER FROM main() WITHOUT REVIEW ⚠
#
# This middleware is the abandoned auto-inject design (see the design
# post-mortem comment block ~125 lines below). It is intentionally
# defined but never wired. PDA-P3.1-R2 nearly flagged this as a P0 on
# the false assumption that it was active — saved by the fact that
# grep confirms zero `_add_telemetry_inject_middleware(app)` call sites
# in main().
#
# If you register this in the future, you WILL break the API Console
# Content-Security-Policy. The Console page at /account/api/keys uses
# `script-src 'self'` (no inline scripts, no nonces) — see DEC-P31-FX-A
# in _qa_tmp/api_console_spec/decisions.md. Auto-injecting an inline
# <script> tag into that response would be silently CSP-rejected by
# the browser, leaving the page non-functional without any server-side
# error. The fix would be to either (a) inject as `<script src="..."`
# (external reference, not inline) or (b) keep the per-page <script>
# tag pattern that replaced this middleware.
#
# Keeping the implementation here for archaeology only.
def _add_telemetry_inject_middleware(app):
    from re import sub as _re_sub
    from fastapi.responses import Response as _Resp
    from pathlib import Path as _Path
    _TLM = _Path(__file__).resolve().parent / "telemetry.js"

    def _etag_for_telemetry():
        try:
            import hashlib
            data = _TLM.read_bytes()
            return hashlib.sha256(data).hexdigest()[:12]
        except Exception:
            return "v1"

    @app.middleware("http")
    async def _inject_telemetry_into_html(request: Request, call_next):
        response = await call_next(request)
        if request.method != "GET":
            return response
        ct = response.headers.get("content-type", "")
        if "text/html" not in ct.lower():
            return response
        if request.url.path.startswith("/static/"):
            return response
        try:
            body_chunks = []
            async for chunk in response.body_iterator:
                body_chunks.append(chunk)
        except Exception:
            return response
        body_bytes = b"".join(body_chunks)

        # If gzip ran before us (added BEFORE in main · this happens
        # when FastAPI's middleware order surprises us · see code-review
        # comment for the rationale), decompress so we can edit. We'll
        # leave the body uncompressed and clear Content-Encoding; gzip
        # won't run again on the way out because middleware only sees a
        # response once per request. The downside is slightly larger
        # bytes on the wire for HTML; this is acceptable since HTML is
        # only ~600 KB and the auto-inject visibility is more important.
        encoding = response.headers.get("content-encoding", "").lower()
        if "gzip" in encoding:
            try:
                import gzip as _gz
                body_bytes = _gz.decompress(body_bytes)
            except Exception:
                return _Resp(content=body_bytes, status_code=response.status_code,
                             headers=dict(response.headers), media_type=ct)
        elif "deflate" in encoding:
            try:
                import zlib as _zl
                body_bytes = _zl.decompress(body_bytes)
            except Exception:
                return _Resp(content=body_bytes, status_code=response.status_code,
                             headers=dict(response.headers), media_type=ct)
        elif "br" in encoding:
            # Brotli requires the brotli package; skip if not available.
            try:
                import brotli as _br
                body_bytes = _br.decompress(body_bytes)
            except Exception:
                return _Resp(content=body_bytes, status_code=response.status_code,
                             headers=dict(response.headers), media_type=ct)

        try:
            text = body_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return _Resp(content=body_bytes, status_code=response.status_code,
                         headers=dict(response.headers), media_type=ct)
        idx = text.lower().rfind("</body>")
        if idx < 0:
            return _Resp(content=body_bytes, status_code=response.status_code,
                         headers=dict(response.headers), media_type=ct)
        ver = _etag_for_telemetry()
        if "/static/telemetry.js" in text:
            # Idempotent rewrite · upgrade any pre-existing reference
            # with the current cache-bust version.
            text = _re_sub(
                r'(/static/telemetry\.js)(\?[^"\'\s>]*)?',
                lambda m: m.group(1) + '?v=' + ver,
                text,
            )
            new_text = text
        else:
            route = request.url.path or "/"
            route_js = route.replace("\\", "\\\\").replace('"', '\\"')
            snippet = (
                '\n<!-- apin telemetry · auto-injected by server middleware -->'
                '\n<script src="/static/telemetry.js?v=' + ver + '" defer></script>'
                '\n<script>document.addEventListener("DOMContentLoaded",function(){'
                'try{if(window.APIN_TLM)window.APIN_TLM.init({page_route:"' + route_js + '"})}'
                'catch(e){}});</script>\n'
            )
            idx = text.lower().rfind("</body>")
            new_text = text[:idx] + snippet + text[idx:]
        new_bytes = new_text.encode("utf-8")
        # Drop Content-Length (we mutated size) AND Content-Encoding (we
        # decompressed earlier, so the body is now uncompressed even if
        # the original was gzipped).
        _drop = {"content-length", "content-encoding"}
        headers = {k: v for k, v in response.headers.items()
                   if k.lower() not in _drop}
        return _Resp(content=new_bytes, status_code=response.status_code,
                     headers=headers, media_type=ct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766,
                         help="Default 8766 so it can run alongside the "
                              "original on 8765")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preload", action="store_true",
                         help="Eager-load APIN at startup (same as original)")
    args = parser.parse_args()

    import uvicorn
    app = make_app()

    # Stage 6 follow-up · Telemetry instrumentation
    # ────────────────────────────────────────────────────────────────
    # The earlier approach (a response-body-rewriting middleware that
    # auto-injected the <script> tag) fought with FastAPI's gzip
    # middleware in unfortunate ways · gzip compressed responses
    # BEFORE the inject middleware could see them, leaving the inject
    # middleware looking at compressed bytes. After repeated attempts
    # to negotiate that ordering, the simpler-and-more-reliable design
    # is: each HTML page directly includes <script src="/static/telemetry.js"
    # defer> in its <head>. telemetry.js's auto-init now falls back to
    # location.pathname when no data-tlm-route attribute is present,
    # so a single one-line script tag per page is enough.

    # PVA round 6 D-11: gzip compression on responses >= 1 KB.
    # MOVED to `_add_account_console_routes()` LAST in Phase 3.2 R1 fix
    # bundle (PDA-P3.2-R1-F02). Original placement here ran GZip BEFORE
    # the Phase 2.3 security middlewares, putting TokenRedactionMiddleware
    # outside GZip in the stack — TokenRedaction would then scan already-
    # compressed bytes and fail to detect plaintext token leaks.
    #
    # By registering GZip inside `_add_account_console_routes()` AFTER the
    # 4 security middlewares, GZip ends up OUTERMOST at runtime (Starlette
    # `add_middleware()` prepends — last added = outermost). The new order:
    #   GZip (outer) > TokenRedaction > TokenFormat > Session > Sudo (inner)
    # ↳ redaction sees plaintext; GZip compresses the (already-redacted)
    # final response.
    #
    # See DEC-P32-MW-1 + DEC-P32-FIX-F02 in
    # _qa_tmp/api_console_spec/decisions.md.

    _override_index_route(app)
    _add_landing_route(app)  # cinematic landing + login/signup (additive)
    _add_seo_routes(app)     # favicon, logo, robots.txt, sitemap.xml
    _add_auth_routes(app)    # /auth/* - real SQLite+argon2id, Day 3
    _add_dashboard_routes(app)  # /dashboard + /dashboard/data, Day 4

    # Phase B.3: load tomato pipeline (v3 + single-pass LoRA ensemble) and override /predict/full.
    # Additive - no edits to scripts/apin/. Tomato pipeline is optional: if it fails to
    # load, the override still runs and tomato requests return a graceful error. Okra /
    # brassica / chilli paths are unaffected regardless of tomato pipeline status.
    tomato_pipeline = _load_tomato_pipeline_safe()
    _override_predict_full_route(app, tomato_pipeline)
    # PDA R1 B-1 fix: /warmup/status must reflect the v2 override's router state,
    # not section8's closure state (which is never populated in this process).
    _override_warmup_status_route(app)

    # Status & health monitoring - public /status page, content-negotiated
    # /health, and a once-a-minute heartbeat task. Additive + override.
    _add_status_routes(app)
    _override_health_route(app)

    # API v1 — contract-conforming machine endpoints (API_CONTRACT.md).
    # Phase 1, batch 1: read-only reference endpoints. Purely additive.
    _add_reference_routes(app)
    # Phase 1, batch 2: action endpoints (predict/quick, predict/batch,
    # warmup, benchmarks, keys). Purely additive; existing /predict/full
    # and the inference website are untouched.
    _add_v1_action_routes(app)
    # Phase 1, batch 3: envelope-mirror endpoints under /api/... of
    # legacy /predict/full, /apin/info, /feedback, /feedback/stats,
    # and /feedback/retrain/history. Legacy paths remain untouched so
    # the inference website is unaffected.
    _add_v1_mirror_routes(app)
    # Phase 3: /docs page (the human-facing API monograph).
    _add_docs_route(app)
    # Phase 2, batch 1: drone perception endpoints (/api/scan,
    # /api/scan/batch, /api/scans, /api/scans/{uid}). Persists scans
    # to the scans table; emits RFC 7946 GeoJSON. Additive.
    _add_perception_routes(app)
    # Phase 1, batch 2 (PVA drift fix): wrap FastAPI's
    # RequestValidationError so the new v1 endpoints emit §3 errors when
    # the body is missing/wrong, instead of raw {"detail":[…]}.
    _add_v1_validation_handler(app)
    # Phase 4: Pipeline Atlas page (/pipeline) + GET /api/recent + the
    # request-capture middleware that feeds the live request log.
    _add_pipeline_route(app)
    # Stage 2: KPI summary (/api/stats/summary) + telemetry ingest
    # (/api/telemetry/ingest). The Pipeline Atlas KPI tiles will read
    # from /api/stats/summary; the client-side telemetry library
    # (Stage 3) beacons batches into /api/telemetry/ingest.
    _add_stats_and_telemetry_routes(app)
    # Stage 7 / Phase 3.1: API Console (keys list + the 6 CRUD endpoints
    # from scripts.apin_v2.account.routes_keys). Middleware stack wiring
    # is intentionally deferred to Phase 3.4 — see the helper's docstring
    # for the security model rationale.
    _add_account_console_routes(app)

    if args.preload:
        logger.info("Warming APIN (Model 2 + EfficientNet + DINOv2 + PSV)...")
        a = get_apin()
        import time
        t0 = time.time()
        try: a._lazy_load_model2();       logger.info(f"  Model 2 warm ({time.time()-t0:.1f}s)")
        except Exception as e: logger.warning(f"  Model 2 warmup failed: {e}")
        t0 = time.time()
        try: a._lazy_load_efficientnet(); logger.info(f"  EfficientNet warm ({time.time()-t0:.1f}s)")
        except Exception as e: logger.warning(f"  EfficientNet warmup failed: {e}")
        t0 = time.time()
        try: a._lazy_load_dinov2();       logger.info(f"  DINOv2 warm ({time.time()-t0:.1f}s)")
        except Exception as e: logger.warning(f"  DINOv2 warmup failed: {e}")
        logger.info("APIN fully warm")

    logger.info(f"Starting APIN v2 server on {args.host}:{args.port} "
                 f"(original still on 8765 if it's running)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
