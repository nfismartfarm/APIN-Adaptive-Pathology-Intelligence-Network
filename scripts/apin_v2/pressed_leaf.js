// Phase 8.G · Shared pressed-leaf avatar generator.
//
// Lifted from dashboard.html's `avatarSVG(crop, seed)` so both the inference
// site and the Console use the same per-user identity SVG. The user's
// `pressed_leaf_seed` (0-5, picked at signup) drives the fill colour; the
// `crop` argument picks the leaf shape (okra / brassica / tomato / chilli
// / default ovate). For the Console chip the user doesn't have a dominant
// crop, so callers pass null and get the default ovate leaf.
//
// Exposes one global function:  window.APIN_pressedLeaf(crop, seed) -> string
// (returns an `<svg>...</svg>` markup string suitable for innerHTML).
//
// Style: hand-drawn paper aesthetic — green palette, ink stroke,
// stroke-linejoin=round so corners look pen-drawn not sharp.

(function () {
  "use strict";

  // Six green hues — same palette dashboard.html uses. Don't change these
  // without coordinating with the inference dashboard; the user expects
  // their leaf to look the same across both surfaces.
  var PALETTE = ["#2f6f3e", "#4d8a52", "#5a9568", "#3d7044", "#669973", "#1e4d29"];
  var STROKE  = "#1e4d29";

  function pressedLeaf(crop, seed) {
    var fill = PALETTE[((seed | 0) % PALETTE.length + PALETTE.length) % PALETTE.length];

    // crop-specific path — same paths as scripts/apin_v2/dashboard.html
    if (crop === "okra") {
      return '<svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        + '<g stroke="' + STROKE + '" stroke-width="1" fill="' + fill + '" stroke-linejoin="round">'
        + '<path d="M30 6 L 38 24 L 56 22 L 42 36 L 50 54 L 30 44 L 10 54 L 18 36 L 4 22 L 22 24 Z"/>'
        + '<path d="M30 6 L 30 50" stroke="' + STROKE + '" stroke-width="0.7" fill="none"/>'
        + '</g></svg>';
    }
    if (crop === "brassica") {
      return '<svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        + '<g stroke="' + STROKE + '" stroke-width="1" fill="' + fill + '" stroke-linejoin="round">'
        + '<path d="M30 6 C 16 8, 6 22, 8 38 C 10 52, 22 56, 30 54 C 38 56, 50 52, 52 38 C 54 22, 44 8, 30 6 Z"/>'
        + '<path d="M30 6 L 30 54 M 30 20 L 16 26 M 30 20 L 44 26 M 30 36 L 14 42 M 30 36 L 46 42" '
        + 'stroke="' + STROKE + '" stroke-width="0.6" fill="none"/>'
        + '</g></svg>';
    }
    if (crop === "tomato") {
      return '<svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        + '<g stroke="' + STROKE + '" stroke-width="1" fill="' + fill + '" stroke-linejoin="round">'
        + '<path d="M30 4 L 24 14 L 14 12 L 18 22 L 8 24 L 18 32 L 12 42 L 24 40 L 22 52 L 30 46 L 38 52 L 36 40 L 48 42 L 42 32 L 52 24 L 42 22 L 46 12 L 36 14 Z"/>'
        + '<path d="M30 4 L 30 50" stroke="' + STROKE + '" stroke-width="0.7" fill="none"/>'
        + '</g></svg>';
    }
    if (crop === "chilli") {
      return '<svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        + '<g stroke="' + STROKE + '" stroke-width="1" fill="' + fill + '" stroke-linejoin="round">'
        + '<path d="M30 6 C 18 12, 12 28, 14 42 C 16 52, 26 56, 30 54 C 34 56, 44 52, 46 42 C 48 28, 42 12, 30 6 Z"/>'
        + '<path d="M30 6 L 30 54" stroke="' + STROKE + '" stroke-width="0.7" fill="none"/>'
        + '</g></svg>';
    }
    // Default — soft ovate. Used on the Console chip where the user
    // doesn't have a dominant crop.
    return '<svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
      + '<g stroke="' + STROKE + '" stroke-width="1" fill="' + fill + '" stroke-linejoin="round">'
      + '<path d="M30 8 C 18 14, 14 30, 18 44 C 22 54, 32 54, 36 44 C 42 30, 42 14, 30 8 Z"/>'
      + '<path d="M30 8 L 30 50" stroke="' + STROKE + '" stroke-width="0.7" fill="none"/>'
      + '</g></svg>';
  }

  // ── Drifting-leaves identity strip ─────────────────────────────────
  // Used in the chip dropdown header. Renders a horizontal strip with the
  // primary leaf (pressed_leaf_seed) prominent in the center, and 4 smaller
  // leaves drifting at varied positions / sizes / opacities around it.
  // Deterministic per seed so it's stable across visits.
  function driftingLeavesStrip(seed) {
    var s = (seed | 0);
    // Pseudo-random sequence seeded by the user's seed — Mulberry32.
    var state = (s * 0x9E3779B1) >>> 0;
    function rng() {
      state = (state + 0x6D2B79F5) >>> 0;
      var t = state;
      t = (t ^ (t >>> 15)) * (t | 1) >>> 0;
      t = (t + ((t ^ (t >>> 7)) * (t | 61))) >>> 0;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    var W = 260, H = 84;
    var parts = [];
    // Background drifting leaves — 5 small at varied positions
    for (var i = 0; i < 5; i++) {
      var x = 10 + rng() * (W - 20);
      var y = 8 + rng() * (H - 30);
      var size = 16 + Math.floor(rng() * 14);
      var rot = Math.floor(rng() * 360);
      var opacity = 0.18 + rng() * 0.22;
      var driftSeed = (s + i + 1) % 6;
      var fill = PALETTE[driftSeed];
      parts.push(
        '<g transform="translate(' + x.toFixed(1) + ',' + y.toFixed(1) + ') '
        + 'rotate(' + rot + ') scale(' + (size / 60).toFixed(3) + ')" '
        + 'opacity="' + opacity.toFixed(2) + '">'
        + '<path d="M30 8 C 18 14, 14 30, 18 44 C 22 54, 32 54, 36 44 C 42 30, 42 14, 30 8 Z" '
        + 'fill="' + fill + '" stroke="' + STROKE + '" stroke-width="0.6" stroke-linejoin="round"/>'
        + '</g>'
      );
    }
    // Primary leaf — center-left, crisp and large
    var primaryFill = PALETTE[((s | 0) % PALETTE.length + PALETTE.length) % PALETTE.length];
    parts.push(
      '<g transform="translate(20,42) scale(0.95)">'
      + '<path d="M30 8 C 18 14, 14 30, 18 44 C 22 54, 32 54, 36 44 C 42 30, 42 14, 30 8 Z" '
      + 'fill="' + primaryFill + '" stroke="' + STROKE + '" stroke-width="1" stroke-linejoin="round"/>'
      + '<path d="M30 8 L 30 50" stroke="' + STROKE + '" stroke-width="0.6" fill="none"/>'
      + '</g>'
    );
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" '
      + 'aria-hidden="true" preserveAspectRatio="xMidYMid slice" '
      + 'style="display:block;width:100%;height:100%">'
      + parts.join('')
      + '</svg>';
  }

  // Export to window so HTML pages can call without bundler.
  window.APIN_pressedLeaf = pressedLeaf;
  window.APIN_driftingLeavesStrip = driftingLeavesStrip;
})();
