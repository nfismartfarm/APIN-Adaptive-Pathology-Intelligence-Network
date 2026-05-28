// apin_fx.js — Phase 9.N.1 · shared animation library
// Single Web-Animations-API-based engine that every chart and panel uses.
// All durations come from a frozen DURATIONS object — no magic numbers.
// All easing comes from a single cubic-bezier — the "settling inked nib" curve.
//
// Public surface lives at window.APIN.fx.* — usage:
//   APIN.fx.enter(el)              // opacity 0→1 + translateY 8→0
//   APIN.fx.exit(el)               // opacity 1→0 + translateY 0→-8
//   APIN.fx.ripple(el, x, y)       // radial expand from click point
//   APIN.fx.pulse(el)              // 1px ring outline glow flash
//   APIN.fx.lift(el)               // hover lift 2px + soft shadow
//   APIN.fx.flip(els, before, after) — FLIP technique for row re-ordering
//   APIN.fx.drawPath(svgPath, dur) — stroke-dasharray draw-in
//   APIN.fx.arcMorph(slice, from, to) — d-attribute interpolation for donut arcs
//   APIN.fx.bar(rect, fromY, toY)  — y-height interpolation for histogram bars
//   APIN.fx.cardGrow(el, fromRect, toRect) — FLIP card into lightbox
//   APIN.fx.chipSlide(el)          // filter chip slide-in from left
//   APIN.fx.wetInk(el)             // live-stream row blur-fade entry
//   APIN.fx.fadeReplace(el, mutate) // cross-fade textContent without flash
//
// Returns Promise<Animation> for every call so callers can await completion.

(function () {
  "use strict";

  // ── Easing & durations ────────────────────────────────────────────────────
  // Single shared easing: "settling inked nib". Don't reach for anything else.
  const EASE = "cubic-bezier(0.22, 1, 0.36, 1)";

  // Duration ladder. Animations that don't fit one of these are a bug.
  const DUR = Object.freeze({
    hover:    150,
    active:   240,
    cardGrow: 380,
    kpiRoll:  420,
    arc:      500,
    flip:     600,
    drawPath: 700,
    chipSlide: 200,
    wetInk:    220,
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  const _supported = typeof Element !== "undefined"
                       && typeof Element.prototype.animate === "function";

  function _ani(el, keyframes, opts) {
    if (!_supported || !el || !el.animate) {
      return { finished: Promise.resolve(), cancel: () => {} };
    }
    const a = el.animate(keyframes, opts);
    return a;
  }

  // ── Entry / exit ──────────────────────────────────────────────────────────
  function enter(el, opts) {
    opts = opts || {};
    const fromY = (opts.from || 8);
    const dur = opts.duration || DUR.cardGrow;
    return _ani(el, [
      { opacity: 0, transform: `translateY(${fromY}px)` },
      { opacity: 1, transform: "translateY(0)" }
    ], { duration: dur, easing: EASE, fill: "both" });
  }

  function exit(el, opts) {
    opts = opts || {};
    const toY = (opts.to != null ? opts.to : -8);
    const dur = opts.duration || DUR.active;
    return _ani(el, [
      { opacity: 1, transform: "translateY(0)" },
      { opacity: 0, transform: `translateY(${toY}px)` }
    ], { duration: dur, easing: EASE, fill: "both" });
  }

  // ── Ripple ────────────────────────────────────────────────────────────────
  // Spawns a circular ripple at (x, y) inside `el`, scales 0→1, fades out.
  function ripple(el, x, y) {
    if (!el) return Promise.resolve();
    const rect = el.getBoundingClientRect();
    const cx = (x != null) ? x - rect.left : rect.width / 2;
    const cy = (y != null) ? y - rect.top  : rect.height / 2;
    const size = Math.max(rect.width, rect.height) * 1.6;
    const r = document.createElement("span");
    r.className = "apin-fx-ripple";
    Object.assign(r.style, {
      position: "absolute",
      left: (cx - size / 2) + "px",
      top:  (cy - size / 2) + "px",
      width: size + "px",
      height: size + "px",
      borderRadius: "50%",
      background: "currentColor",
      opacity: "0.16",
      transform: "scale(0)",
      pointerEvents: "none",
      zIndex: "1",
    });
    // Ensure host can clip
    const cs = getComputedStyle(el);
    if (cs.position === "static") el.style.position = "relative";
    if (cs.overflow === "visible") el.style.overflow = "hidden";
    el.appendChild(r);
    const a = _ani(r, [
      { transform: "scale(0)", opacity: 0.20 },
      { transform: "scale(1)", opacity: 0 }
    ], { duration: DUR.active, easing: EASE, fill: "forwards" });
    const cleanup = () => { try { r.remove(); } catch (e) {} };
    if (a && a.finished) a.finished.then(cleanup, cleanup);
    else setTimeout(cleanup, DUR.active + 40);
    return a;
  }

  // ── Pulse (1px ring outline glow) ─────────────────────────────────────────
  function pulse(el) {
    if (!el) return Promise.resolve();
    return _ani(el, [
      { boxShadow: "0 0 0 0px rgba(45, 106, 79, 0.4)" },
      { boxShadow: "0 0 0 6px rgba(45, 106, 79, 0)" }
    ], { duration: 300, easing: EASE, fill: "none" });
  }

  // ── Lift (hover) ──────────────────────────────────────────────────────────
  function lift(el, on) {
    if (!el) return;
    if (on) {
      el.style.transition = `transform ${DUR.hover}ms ${EASE}, box-shadow ${DUR.hover}ms ${EASE}`;
      el.style.transform = "translateY(-2px)";
      el.style.boxShadow = "0 6px 14px rgba(20, 16, 12, 0.08)";
    } else {
      el.style.transform = "";
      el.style.boxShadow = "";
    }
  }

  // ── FLIP for row re-order ─────────────────────────────────────────────────
  // Usage:
  //   const before = APIN.fx.flipMeasure(els);
  //   // …mutate DOM (re-order children)…
  //   APIN.fx.flipPlay(els, before);
  function flipMeasure(els) {
    const m = new Map();
    [...els].forEach(el => {
      const r = el.getBoundingClientRect();
      m.set(el, { left: r.left, top: r.top });
    });
    return m;
  }
  function flipPlay(els, before) {
    if (!before) return [];
    const animations = [];
    [...els].forEach(el => {
      const prev = before.get(el);
      if (!prev) return;
      const now = el.getBoundingClientRect();
      const dx = prev.left - now.left;
      const dy = prev.top  - now.top;
      if (dx === 0 && dy === 0) return;
      const a = _ani(el, [
        { transform: `translate(${dx}px, ${dy}px)` },
        { transform: "translate(0, 0)" }
      ], { duration: DUR.flip, easing: EASE, fill: "both" });
      animations.push(a);
    });
    return animations;
  }

  // ── Draw-path (stroke-dasharray entry for SVG <path>) ─────────────────────
  function drawPath(pathEl, opts) {
    if (!pathEl || !pathEl.getTotalLength) return Promise.resolve();
    const len = pathEl.getTotalLength();
    const dur = (opts && opts.duration) || DUR.drawPath;
    pathEl.style.strokeDasharray = String(len);
    pathEl.style.strokeDashoffset = String(len);
    const a = _ani(pathEl, [
      { strokeDashoffset: len },
      { strokeDashoffset: 0 }
    ], { duration: dur, easing: EASE, fill: "forwards" });
    return a;
  }

  // ── Arc morph (interpolate `d` between two SVG arc paths) ────────────────
  // For donut slice resize. We don't try to fully interpolate the SVG-path
  // grammar; instead we re-draw the slice at frame N using a callback the
  // caller supplies. The caller knows how to compute arcPath(startAngle, endAngle).
  function arcMorph(slice, builder, from, to) {
    if (!slice || !builder) return Promise.resolve();
    const dur = DUR.arc;
    const t0 = performance.now();
    return new Promise(res => {
      const tick = () => {
        const t = Math.min(1, (performance.now() - t0) / dur);
        // settling-inked-nib easing approximation
        const e = 1 - Math.pow(1 - t, 3);
        const cur = {
          startAngle: from.startAngle + (to.startAngle - from.startAngle) * e,
          endAngle:   from.endAngle   + (to.endAngle   - from.endAngle)   * e,
        };
        slice.setAttribute("d", builder(cur.startAngle, cur.endAngle));
        if (t < 1) requestAnimationFrame(tick);
        else res();
      };
      requestAnimationFrame(tick);
    });
  }

  // ── Bar y-height interpolation ────────────────────────────────────────────
  function bar(rectEl, fromY, fromH, toY, toH, opts) {
    if (!rectEl) return Promise.resolve();
    const dur = (opts && opts.duration) || 450;
    const delay = (opts && opts.delay) || 0;
    rectEl.setAttribute("y", String(fromY));
    rectEl.setAttribute("height", String(fromH));
    return _ani(rectEl, [
      { y: fromY, height: fromH, attributeName: "y" },
      { y: toY,   height: toH,   attributeName: "y" }
    ], { duration: dur, delay, easing: EASE, fill: "forwards" });
  }

  // ── Card grow into lightbox ───────────────────────────────────────────────
  // The lightbox shell measures source-card rect and target rect, then calls
  // cardGrow to transform the source's clone to the target without reflowing
  // the grid. The clone is a fixed-position floating element.
  function cardGrow(cloneEl, fromRect, toRect) {
    if (!cloneEl || !fromRect || !toRect) return Promise.resolve();
    const dx = toRect.left - fromRect.left;
    const dy = toRect.top  - fromRect.top;
    const sx = toRect.width  / fromRect.width;
    const sy = toRect.height / fromRect.height;
    return _ani(cloneEl, [
      { transform: "translate(0,0) scale(1,1)" },
      { transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` }
    ], { duration: DUR.cardGrow, easing: EASE, fill: "forwards" });
  }

  // ── Chip slide-in ─────────────────────────────────────────────────────────
  function chipSlide(el) {
    if (!el) return Promise.resolve();
    return _ani(el, [
      { transform: "translateX(-20px)", opacity: 0 },
      { transform: "translateX(0)",     opacity: 1 }
    ], { duration: DUR.chipSlide, easing: EASE, fill: "both" });
  }

  // ── Wet ink (live stream row entry — blur + translate + opacity) ──────────
  function wetInk(el) {
    if (!el) return Promise.resolve();
    return _ani(el, [
      { transform: "translateY(-16px)", opacity: 0, filter: "blur(0.5px)" },
      { transform: "translateY(0)",     opacity: 1, filter: "blur(0px)"   }
    ], { duration: DUR.wetInk, easing: EASE, fill: "both" });
  }

  // ── Fade-replace (avoid textContent flash; cross-fade) ────────────────────
  // mutate() is called at midpoint, when the element is at opacity 0.
  function fadeReplace(el, mutate) {
    if (!el) { try { mutate(); } catch (e) {} return Promise.resolve(); }
    const half = 100;
    return new Promise(res => {
      const a1 = _ani(el, [
        { opacity: 1 }, { opacity: 0 }
      ], { duration: half, easing: EASE, fill: "forwards" });
      const done1 = a1 && a1.finished ? a1.finished : Promise.resolve();
      done1.then(() => {
        try { mutate(); } catch (e) {}
        const a2 = _ani(el, [
          { opacity: 0 }, { opacity: 1 }
        ], { duration: half, easing: EASE, fill: "forwards" });
        const done2 = a2 && a2.finished ? a2.finished : Promise.resolve();
        done2.then(res, res);
      }, res);
    });
  }

  // ── Cancel-all on element ─────────────────────────────────────────────────
  function cancel(el) {
    if (!el || !el.getAnimations) return;
    try { el.getAnimations().forEach(a => a.cancel()); } catch (e) {}
  }

  // ── Time helper (UTC storage → device-local display) ──────────────────────
  // The DB stores timestamps in UTC as 'YYYY-MM-DD HH:MM:SS.ssssss' with NO
  // zone marker. Parsing that naively makes the browser read it as LOCAL time,
  // skewing every "x ago" by the viewer's UTC offset. APIN.time anchors the
  // string to UTC (appends Z) and then renders in the viewer's own timezone,
  // so the console reads correctly on whatever device you're on — no hardcoded
  // IST. Use this everywhere a server timestamp is shown.
  function _utcDate(s) {
    if (s == null) return null;
    if (s instanceof Date) return s;
    let str = String(s).trim().replace(" ", "T");
    // Append Z only if the string carries no explicit zone (Z or ±HH:MM).
    if (!/[zZ]$|[+-]\d\d:?\d\d$/.test(str)) str += "Z";
    const d = new Date(str);
    return isNaN(d.getTime()) ? null : d;
  }
  const _TZ = (() => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch (_) { return ""; } })();
  const _tzOffsetMin = () => -new Date().getTimezoneOffset(); // e.g. IST → +330
  function _ago(s) {
    const d = _utcDate(s); if (!d) return "—";
    const sec = Math.max(0, Math.round((Date.now() - d.getTime()) / 1000));
    if (sec < 60) return sec + "s ago";
    if (sec < 3600) return Math.floor(sec / 60) + "m ago";
    if (sec < 86400) return Math.floor(sec / 3600) + "h ago";
    return Math.floor(sec / 86400) + "d ago";
  }
  function _local(s, opts) {
    const d = _utcDate(s); if (!d) return "—";
    try { return d.toLocaleString([], opts || { dateStyle: "medium", timeStyle: "short" }); }
    catch (_) { return d.toLocaleString(); }
  }
  function _localTime(s) {
    const d = _utcDate(s); if (!d) return "—";
    try { return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
    catch (_) { return d.toLocaleTimeString(); }
  }

  // ── Public surface ────────────────────────────────────────────────────────
  window.APIN = window.APIN || {};
  window.APIN.fx = {
    DUR, EASE,
    enter, exit,
    ripple, pulse, lift,
    flipMeasure, flipPlay,
    drawPath, arcMorph, bar,
    cardGrow, chipSlide, wetInk,
    fadeReplace, cancel,
  };
  // Timezone-correct formatting, shared across every console page.
  window.APIN.time = {
    utcDate: _utcDate, ago: _ago, local: _local, localTime: _localTime,
    zone: _TZ, offsetMin: _tzOffsetMin,
  };
})();
