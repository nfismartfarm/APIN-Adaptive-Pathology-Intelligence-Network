// 9.N.T27c · Endpoint Genome — a hand-drawn identity sigil per endpoint.
//
// Not a chart. Each endpoint gets a unique, STABLE "ink seal" generated
// deterministically from its path (seed = hash(path)), then modulated by its
// live metrics. Every stroke is explainable:
//
//   core polygon vertices  ← branch complexity (methods + downstream calls)
//   core radius            ← traffic volume
//   shape regularity       ← errors (healthy = symmetric, failing = fractured)
//   radiating arms         ← branches (one per vertex); broken/dashed ← retries
//   halo ring wobble       ← latency (calm = smooth, slow = wavy)
//   stipple density        ← payload weight
//   stroke jitter          ← burstiness
//   coupling filament      ← errors-track-traffic correlation
//
// Drawn in our paper-ink hand-sketch language (seeded wobble + quadratic
// smoothing = organic ink, no external rough-draw lib). Idle motion is
// CSS-only; live reactions (ripple / flicker / stretch) are class toggles.
//
// Exposes:
//   APIN.genome.svg(metrics, opts)   -> { svg, legend, tone }
//   APIN.genome.mount(el, metrics, opts) -> handle { ripple, flicker,
//                                            stretch, setMetrics, destroy }
(function () {
  "use strict";
  if (!window.APIN) window.APIN = {};

  // ── deterministic PRNG (mulberry32) seeded from the path ──
  function _hash(str) {
    let h = 2166136261 >>> 0;
    str = String(str || "?");
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function _rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // tone (from weather) → stroke colour. Mostly ink; accent used sparingly.
  const TONE = {
    calm: "var(--c-accent,#52b788)", ok: "var(--ink,#2b2620)",
    busy: "var(--green-deep,#2d6a4f)", warn: "var(--ochre-deep,#b6822a)",
    crit: "var(--crimson-deep,#b3402f)", mute: "var(--ink-mute,#9a9186)",
  };

  // build a closed "hand-drawn" blob path through `verts` points around (cx,cy)
  // with per-vertex radius irregularity + a small wobble, quadratic-smoothed.
  function _blob(rng, cx, cy, r, verts, irregular, wobble, rot) {
    const pts = [];
    for (let i = 0; i < verts; i++) {
      const ang = rot + (i / verts) * Math.PI * 2;
      // irregularity pushes radius per-vertex; errors raise this → fractured
      const rr = r * (1 - irregular * (0.12 + 0.55 * rng()))
        + (rng() - 0.5) * wobble * r * 0.5;
      pts.push([cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr]);
    }
    // quadratic smoothing through midpoints → organic closed stroke
    let d = "";
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i], nx = pts[(i + 1) % pts.length];
      const mx = (p[0] + nx[0]) / 2, my = (p[1] + nx[1]) / 2;
      if (i === 0) d += `M ${mx.toFixed(2)} ${my.toFixed(2)} `;
      d += `Q ${p[0].toFixed(2)} ${p[1].toFixed(2)} ${((nx[0] + ((pts[(i + 2) % pts.length])[0])) / 2).toFixed(2)} ${(((nx[1]) + (pts[(i + 2) % pts.length][1])) / 2).toFixed(2)} `;
    }
    return d + "Z";
  }

  // a wavy ring (halo) — wobble amplitude encodes latency
  function _ring(rng, cx, cy, r, wobble, segs, rot) {
    const pts = [];
    for (let i = 0; i <= segs; i++) {
      const ang = rot + (i / segs) * Math.PI * 2;
      const rr = r + Math.sin(ang * 3 + rng() * 6.28) * wobble * r * 0.16
        + (rng() - 0.5) * r * 0.02;
      pts.push([cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr]);
    }
    let d = `M ${pts[0][0].toFixed(2)} ${pts[0][1].toFixed(2)} `;
    for (let i = 1; i < pts.length; i++) d += `L ${pts[i][0].toFixed(2)} ${pts[i][1].toFixed(2)} `;
    return d;
  }

  function _esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  function build(metrics, opts) {
    metrics = metrics || {};
    opts = opts || {};
    const tone = opts.tone || "ok";
    const col = TONE[tone] || TONE.ok;
    const seed = _hash(metrics.seed || opts.seed || "?");
    const rng = _rng(seed);
    const rot0 = rng() * Math.PI * 2;              // unique base rotation per path

    const traffic = +metrics.traffic || 0;        // 0..1
    const errBreak = +metrics.err_break || 0;      // 0..1
    const latOsc = +metrics.latency_osc || 0;      // 0..1
    const branches = Math.max(3, Math.min(8, +metrics.branches || 4));
    const payload = +metrics.payload_density || 0; // 0..1
    const retryFrag = +metrics.retry_frag || 0;    // 0..1
    const burst = +metrics.burst || 0;             // 0..1
    const coupling = +metrics.coupling || 0;       // 0..1

    const C = 50;                                  // centre of 100×100 viewBox
    const coreR = 12 + traffic * 16;               // traffic → core size
    const haloR = 34 + latOsc * 6;
    const sw = (opts.strokeBase || 1.6);

    // ── halo ring (latency wobble) ──
    const haloD = _ring(rng, C, C, haloR, 0.15 + latOsc, 64, rot0);

    // ── radiating arms (one per branch); broken if retries high ──
    let arms = "";
    const armRng = _rng(seed ^ 0x9e3779b9);
    for (let i = 0; i < branches; i++) {
      const sym = (i / branches) * Math.PI * 2;
      const ang = rot0 + sym + (armRng() - 0.5) * errBreak * 1.3;  // errors skew angles
      const x0 = C + Math.cos(ang) * coreR, y0 = C + Math.sin(ang) * coreR;
      const len = haloR - coreR - 2 + (armRng() - 0.5) * burst * 6;
      const x1 = C + Math.cos(ang) * (coreR + len), y1 = C + Math.sin(ang) * (coreR + len);
      // a slight hand-drawn bow on the arm
      const mx = (x0 + x1) / 2 + (armRng() - 0.5) * (2 + burst * 4);
      const my = (y0 + y1) / 2 + (armRng() - 0.5) * (2 + burst * 4);
      const broken = retryFrag > 0.15 && (i % 2 === 0);
      arms += `<path class="gn-arm${broken ? " gn-arm-broken" : ""}" d="M ${x0.toFixed(2)} ${y0.toFixed(2)} Q ${mx.toFixed(2)} ${my.toFixed(2)} ${x1.toFixed(2)} ${y1.toFixed(2)}" `
        + `fill="none" stroke="${col}" stroke-width="${(sw * 0.7).toFixed(2)}" stroke-linecap="round"${broken ? ` stroke-dasharray="${(2 + retryFrag * 3).toFixed(1)} ${(2 + retryFrag * 4).toFixed(1)}"` : ""}/>`;
      // a tiny node at the arm tip
      arms += `<circle class="gn-node" cx="${x1.toFixed(2)}" cy="${y1.toFixed(2)}" r="${(0.9 + traffic * 0.8).toFixed(2)}" fill="${col}" opacity="0.7"/>`;
    }

    // ── core blob (vertices = branches, irregularity = errors) ──
    const coreD = _blob(rng, C, C, coreR, branches, errBreak, 0.08 + burst * 0.25, rot0);
    const coreInnerD = _blob(_rng(seed ^ 0x51ed270b), C, C, coreR * 0.6, branches, errBreak * 0.7, 0.06 + burst * 0.2, rot0 + 0.3);

    // ── stipple inside the core (payload density) ──
    let stipple = "";
    const nDots = Math.round(payload * 14);
    const sRng = _rng(seed ^ 0x27d4eb2f);
    for (let i = 0; i < nDots; i++) {
      const a = sRng() * Math.PI * 2, rr = Math.sqrt(sRng()) * coreR * 0.7;
      stipple += `<circle cx="${(C + Math.cos(a) * rr).toFixed(2)}" cy="${(C + Math.sin(a) * rr).toFixed(2)}" r="${(0.5 + sRng() * 0.7).toFixed(2)}" fill="${col}" opacity="0.5"/>`;
    }

    // ── coupling filament: errors track traffic → thread core to halo ──
    let filament = "";
    if (coupling > 0.4) {
      const a = rot0 + 1.1;
      filament = `<path class="gn-filament" d="M ${(C + Math.cos(a) * coreR).toFixed(2)} ${(C + Math.sin(a) * coreR).toFixed(2)} `
        + `Q ${(C + Math.cos(a + 0.4) * haloR * 0.7).toFixed(2)} ${(C + Math.sin(a + 0.4) * haloR * 0.7).toFixed(2)} `
        + `${(C + Math.cos(a) * haloR).toFixed(2)} ${(C + Math.sin(a) * haloR).toFixed(2)}" fill="none" `
        + `stroke="${col}" stroke-width="0.6" stroke-dasharray="1 2" opacity="${(0.3 + coupling * 0.4).toFixed(2)}"/>`;
    }

    const svg =
      `<svg class="gn-svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" aria-hidden="true">`
      + `<g class="gn-all">`
      + `<g class="gn-rot">`
      + `<path class="gn-halo" d="${haloD}" fill="none" stroke="${col}" stroke-width="${(sw * 0.5).toFixed(2)}" opacity="0.4" stroke-linejoin="round"/>`
      + filament
      + `</g>`
      + arms
      + `<path class="gn-core" d="${coreD}" fill="${col}" fill-opacity="0.08" stroke="${col}" stroke-width="${sw}" stroke-linejoin="round"/>`
      + `<path class="gn-core-in" d="${coreInnerD}" fill="none" stroke="${col}" stroke-width="${(sw * 0.6).toFixed(2)}" opacity="0.45" stroke-linejoin="round"/>`
      + stipple
      + `<circle class="gn-ripple" cx="${C}" cy="${C}" r="${coreR}" fill="none" stroke="${col}" stroke-width="1.4" opacity="0"/>`
      + `</g></svg>`;

    const legend = [
      { k: "Core size", v: "traffic volume" },
      { k: branches + " sides / arms", v: "branch complexity (methods + downstream)" },
      { k: "Shape regularity", v: errBreak > 0.25 ? "fractured — elevated errors" : "balanced — low errors" },
      { k: "Halo ring", v: latOsc > 0.4 ? "wavy — high latency" : "smooth — fast" },
      { k: "Inner stipple", v: (payload > 0.5 ? "dense — heavy payloads" : "sparse — light payloads") },
    ];
    if (retryFrag > 0.15) legend.push({ k: "Broken arms", v: "retries detected" });
    if (coupling > 0.4) legend.push({ k: "Filament", v: "errors rise with traffic (load-driven)" });

    return { svg, legend, tone };
  }

  function mount(el, metrics, opts) {
    if (!el) return null;
    let cur = build(metrics, opts);                 // tracks the LATEST render
    el.innerHTML = cur.svg;
    el.classList.add("gn-host");
    let _t = null;
    // all reactions re-query the live DOM so they survive setMetrics() rebuilds
    function _flash(cls, ms) {
      const s = el.querySelector(".gn-svg"); if (!s) return;
      s.classList.add(cls);
      clearTimeout(_t);
      _t = setTimeout(() => { const s2 = el.querySelector(".gn-svg"); if (s2) s2.classList.remove(cls); }, ms);
    }
    return {
      tone: () => cur.tone,
      // live request arrives → ink ripple from the core
      ripple() {
        const ripEl = el.querySelector(".gn-ripple"); if (!ripEl) return;
        try { ripEl.animate([{ r: 12, opacity: 0.55 }, { r: 46, opacity: 0 }], { duration: 620, easing: "cubic-bezier(.22,1,.36,1)" }); } catch (_) {}
      },
      // error → one arm flickers ember
      flicker() {
        const a = el.querySelectorAll(".gn-arm"); if (!a.length) return;
        const pick = a[Math.floor(Math.random() * a.length)];
        try { pick.animate([{ stroke: "#c74f33", opacity: 1 }, {}], { duration: 420, easing: "ease-out" }); } catch (_) {}
      },
      // latency spike → halo briefly stretches
      stretch() { _flash("gn-stretch", 520); },
      setMetrics(m, o) { cur = build(m || metrics, o || opts); el.innerHTML = cur.svg; return cur; },
      legend: () => cur.legend,
      destroy() { clearTimeout(_t); el.innerHTML = ""; el.classList.remove("gn-host"); },
    };
  }

  window.APIN.genome = { svg: build, mount, _hash };
})();
