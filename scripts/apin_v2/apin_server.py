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
                                auth_mode = ("guest_exhausted"
                                             if _g["exhausted"] else "guest")
                    except Exception:
                        pass

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
                    prediction (with image bytes) under their account
                  • guests          → consume one free check and stamp the
                    remaining quota onto the response as `_guest`
                Server-side errors (the TOMATO_* error tiers) are NOT
                counted against a guest's quota — only real results are.
                """
                if not isinstance(result_dict, dict):
                    return result_dict
                _tier = str(result_dict.get("tier", "")).upper()
                _is_error = _tier in ("TOMATO_UNAVAILABLE",
                                      "TOMATO_INFERENCE_ERROR")
                if current_user_id is not None and background_tasks is not None:
                    background_tasks.add_task(
                        _auth_db.record_prediction,
                        current_user_id,
                        result_dict,
                        image_bytes=data,
                    )
                elif auth_mode == "guest" and guest_token and not _is_error:
                    _gc = _auth_db.consume_guest_inference(guest_token)
                    if _gc is not None:
                        result_dict["_guest"] = {
                            "remaining": _gc.get("remaining", 0),
                            "limit": _auth_db.GUEST_INFERENCE_LIMIT,
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
    """Load TomatoPipeline with exception handling; return None on failure."""
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
