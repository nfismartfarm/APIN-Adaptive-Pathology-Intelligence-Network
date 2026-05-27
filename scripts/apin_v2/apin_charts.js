/* Phase 9.D — APIN paper-ink chart library.
 *
 * One file, five renderers, one shared interaction layer.
 *
 *   APIN.charts.timeseries(host, payload, opts)
 *   APIN.charts.donut(host, items, opts)
 *   APIN.charts.histogram(host, buckets, opts)
 *   APIN.charts.topBar(host, items, opts)
 *   APIN.charts.sparkline(host, items, opts)
 *
 * Each renderer is pure: it takes a host element + data and writes SVG.
 * It returns a controller object with mutate methods (setDomain, refresh,
 * etc.) so the same chart can be redrawn from data updates without
 * rebuilding the wrapper DOM. Renderers also expose hooks the shared
 * interaction layer uses to attach hover / zoom / context-menu:
 *
 *   ctrl.getDomain()         — returns the current X domain [lo, hi]
 *   ctrl.setDomain(lo, hi)   — request a new domain (drag-to-zoom)
 *   ctrl.coordsAtX(px)       — given a host-relative x in px, return
 *                              { bucketIndex, dataPoint, label }
 *   ctrl.applyLogY(bool)     — toggle log/linear Y axis
 *   ctrl.exportSVG()         — string of the current chart SVG
 *
 * Aesthetic
 * ---------
 * Paper-ink palette only. Series colors come from semantic CSS variables
 * (--c-ok / --c-warn / --c-amber / --c-danger / --c-info / --c-ink /
 *  --c-muted / --c-series-{0..4}) which are declared on `:root` in the
 * page CSS. Charts never bake hex colors in.
 *
 * No dependencies. SVG everywhere. The font for axis labels matches
 * 'JetBrains Mono' as a CSS rule on `.chart-axis`.
 */
(function () {
  'use strict';
  if (!window.APIN) window.APIN = {};
  if (!APIN.charts) APIN.charts = {};

  // ─── Shared helpers ──────────────────────────────────────────────────
  function el(tag, attrs, kids) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) for (const k in attrs) {
      if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    }
    if (kids) for (const k of kids) e.appendChild(k);
    return e;
  }
  function fmtNum(n) {
    if (n == null || isNaN(n)) return '—';
    n = Number(n);
    if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + 'k';
    if (n < 10 && n !== Math.round(n)) return n.toFixed(1);
    return String(Math.round(n));
  }
  function tFmt(iso) {
    if (!iso) return '';
    const d = new Date(iso.replace(' ', 'T') + (/Z$/.test(iso) ? '' : 'Z'));
    if (isNaN(d)) return iso;
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  function dFmt(iso) {
    if (!iso) return '';
    const d = new Date(iso.replace(' ', 'T') + (/Z$/.test(iso) ? '' : 'Z'));
    if (isNaN(d)) return iso;
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) +
           ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function colorVarFor(token) {
    // Map a chart-component color_token onto a CSS var.
    const map = {
      ok: '--c-ok', warn: '--c-warn', amber: '--c-amber',
      danger: '--c-danger', info: '--c-info', ink: '--c-ink',
      muted: '--c-muted',
      'series-0': '--c-series-0', 'series-1': '--c-series-1',
      'series-2': '--c-series-2', 'series-3': '--c-series-3',
      'series-4': '--c-series-4',
    };
    return 'var(' + (map[token] || '--c-ink') + ')';
  }

  // ─── 9.N.4 · Paper-ink aesthetic helpers ─────────────────────────────
  // niceScale: returns "nice round" tick values for a Y axis.
  // Adapted from the classic Heckbert algorithm. Input: rawMax (max data
  // value), numTicks (desired tick count). Returns { max, ticks: [...] }.
  // Example: niceScale(67, 4) → { max: 80, ticks: [0,20,40,60,80] }
  function niceScale(rawMax, numTicks) {
    numTicks = numTicks || 4;
    if (!Number.isFinite(rawMax) || rawMax <= 0) {
      return { max: 1, ticks: [0, 0.25, 0.5, 0.75, 1] };
    }
    const range = _niceNum(rawMax, false);
    const step  = _niceNum(range / numTicks, true);
    const max   = Math.ceil(rawMax / step) * step;
    const ticks = [];
    for (let v = 0; v <= max + 1e-9; v += step) ticks.push(Number(v.toFixed(6)));
    return { max, ticks };
  }
  function _niceNum(x, round) {
    const exp = Math.floor(Math.log10(x));
    const f = x / Math.pow(10, exp);
    let nf;
    if (round) {
      if (f < 1.5) nf = 1; else if (f < 3) nf = 2;
      else if (f < 7) nf = 5; else nf = 10;
    } else {
      if (f <= 1) nf = 1; else if (f <= 2) nf = 2;
      else if (f <= 5) nf = 5; else nf = 10;
    }
    return nf * Math.pow(10, exp);
  }

  // injectPaperInkDefs: inserts (idempotently) the shared <defs> patterns
  // + filters every chart needs. ID-prefixed with 'pi-' to avoid collisions.
  //   pi-hatch-{color}-{density}  — 45° hatch fills for area gradients
  //   pi-stipple-{tone}           — dot patterns for histogram bars
  //   pi-wobble                   — turbulence + displacement filter for hand-drawn lines
  // Returns the <defs> markup string so caller appends inside the SVG.
  function paperInkDefs(seriesColors) {
    const seen = new Set();
    const colors = (seriesColors || []).filter(c => {
      if (seen.has(c)) return false;
      seen.add(c); return true;
    });
    let defs = '<defs>';
    // 45° hatch pattern per series color, density 12px stride
    colors.forEach((c, i) => {
      const id = 'pi-hatch-' + i;
      defs += '<pattern id="' + id + '" patternUnits="userSpaceOnUse" '
        + 'width="6" height="6" patternTransform="rotate(45)">'
        + '<line x1="0" y1="0" x2="0" y2="6" stroke="' + c
        + '" stroke-width="1.2" stroke-linecap="round" opacity="0.32"/>'
        + '</pattern>';
    });
    // Stipple patterns at 3 tonal densities. Used by histogram/heatmap.
    const stipples = [
      { id: 'pi-stipple-1', dots: 1, op: 0.22 }, // light
      { id: 'pi-stipple-2', dots: 3, op: 0.34 }, // medium
      { id: 'pi-stipple-3', dots: 5, op: 0.50 }, // heavy
    ];
    stipples.forEach(s => {
      let dots = '';
      // Place s.dots in a 8x8 cell pseudo-randomly but deterministically
      const seedDots = [
        [1,1],[5,2],[2,5],[6,6],[3,7],[7,4],[4,3],[6,1]
      ];
      for (let k = 0; k < s.dots; k++) {
        const [dx, dy] = seedDots[k % seedDots.length];
        dots += '<circle cx="' + dx + '" cy="' + dy + '" r="0.8" fill="currentColor" opacity="'
              + s.op + '"/>';
      }
      defs += '<pattern id="' + s.id + '" patternUnits="userSpaceOnUse" '
        + 'width="8" height="8">' + dots + '</pattern>';
    });
    // Wobble filter — subtle hand-drawn imperfection on strokes.
    // baseFrequency low → long-wavelength wave; scale small → tiny displacement.
    defs += '<filter id="pi-wobble" x="-2%" y="-2%" width="104%" height="104%">'
      + '<feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves="2" seed="3" result="t"/>'
      + '<feDisplacementMap in="SourceGraphic" in2="t" scale="0.45" xChannelSelector="R" yChannelSelector="G"/>'
      + '</filter>';
    defs += '</defs>';
    return defs;
  }

  // marginaliaLabel: returns SVG fragment for a small serif annotation
  // pointing at a data point (used on time-series peaks / min / anomalies).
  // anchorX, anchorY = the data point in chart coords.
  // text  = the label string.
  // side  = 'right' | 'left' | 'above' | 'below' — leader-line direction.
  function marginaliaLabel(anchorX, anchorY, text, side) {
    side = side || 'right';
    const dx = side === 'right' ? 28 : side === 'left' ? -28 : 0;
    const dy = side === 'above' ? -22 : side === 'below' ? 22 : -8;
    const lx = anchorX + dx;
    const ly = anchorY + dy;
    // Small curved leader (quadratic) from (anchor) to (label)
    const cx = (anchorX + lx) / 2;
    const cy = anchorY - 4;
    const anchor = side === 'right' ? 'start' : side === 'left' ? 'end' : 'middle';
    return ''
      + '<g class="chart-marginalia" opacity="0">'
      + '<circle cx="' + anchorX + '" cy="' + anchorY + '" r="2.2" '
      +   'fill="var(--ink, #1a1612)" stroke="var(--paper, #fbf9f3)" stroke-width="1"/>'
      + '<path d="M' + anchorX + ',' + anchorY
      +   ' Q' + cx + ',' + cy + ' ' + lx + ',' + ly + '" '
      +   'fill="none" stroke="var(--ink-soft, #6b6453)" stroke-width="0.8" '
      +   'stroke-linecap="round"/>'
      + '<text x="' + lx + '" y="' + ly + '" text-anchor="' + anchor + '" '
      +   'font-family="Fraunces, serif" font-style="italic" font-size="11" '
      +   'fill="var(--ink, #1a1612)">' + escHtml(text) + '</text>'
      + '</g>';
  }

  // ─── Tooltip singleton ───────────────────────────────────────────────
  let _tip = null;
  function tipEl() {
    if (_tip) return _tip;
    _tip = document.createElement('div');
    _tip.className = 'apin-chart-tip';
    _tip.style.cssText =
      'position:fixed;pointer-events:none;background:var(--paper,#fbf9f3);' +
      'border:1px solid var(--ink-soft,#5a5246);padding:8px 11px;' +
      'border-radius:6px;font-family:"JetBrains Mono",monospace;' +
      'font-size:11.5px;color:var(--ink,#1a1612);' +
      'box-shadow:0 4px 14px rgba(0,0,0,0.10);z-index:10000;' +
      'transition:opacity 120ms;opacity:0;line-height:1.55;max-width:280px';
    document.body.appendChild(_tip);
    return _tip;
  }
  function showTip(x, y, html) {
    const t = tipEl();
    t.innerHTML = html;
    t.style.opacity = '1';
    // Position with a small offset, clamped to viewport.
    const w = t.offsetWidth, h = t.offsetHeight;
    let px = x + 12, py = y - h - 8;
    if (px + w > window.innerWidth - 8) px = x - w - 12;
    if (py < 8) py = y + 16;
    t.style.left = px + 'px';
    t.style.top = py + 'px';
  }
  function hideTip() {
    if (_tip) _tip.style.opacity = '0';
  }
  // Hide on scroll/visibility loss so a stale tip never lingers.
  window.addEventListener('scroll', hideTip, { passive: true, capture: true });
  document.addEventListener('visibilitychange', hideTip);

  // ─── Context menu singleton ──────────────────────────────────────────
  let _menu = null;
  function menuEl() {
    if (_menu) return _menu;
    _menu = document.createElement('div');
    _menu.className = 'apin-chart-menu';
    _menu.style.cssText =
      'position:fixed;background:var(--paper,#fbf9f3);' +
      'border:1px solid var(--paper-edge,#c7bca9);' +
      'border-radius:8px;padding:4px 0;font-size:12.5px;' +
      'font-family:Inter,system-ui,sans-serif;' +
      'color:var(--ink,#1a1612);box-shadow:0 8px 24px rgba(0,0,0,0.10);' +
      'z-index:10001;display:none;min-width:200px';
    document.body.appendChild(_menu);
    return _menu;
  }
  function openMenu(x, y, items) {
    const m = menuEl();
    m.innerHTML = items.map((it, i) => {
      if (it.divider) return '<div style="height:1px;background:var(--paper-edge);margin:4px 0"></div>';
      return '<div class="mi" data-i="' + i + '" style="padding:6px 14px;' +
        'cursor:pointer;display:flex;align-items:center;gap:8px;' +
        (it.danger ? 'color:var(--c-danger,#b01820);' : '') +
        '">' + escHtml(it.label) + '</div>';
    }).join('');
    m.style.display = 'block';
    // Clamp inside viewport
    const w = m.offsetWidth, h = m.offsetHeight;
    let px = x, py = y;
    if (px + w > window.innerWidth - 8) px = window.innerWidth - w - 8;
    if (py + h > window.innerHeight - 8) py = window.innerHeight - h - 8;
    m.style.left = px + 'px';
    m.style.top = py + 'px';
    m.querySelectorAll('.mi').forEach(mi => {
      mi.addEventListener('mouseenter', () => {
        mi.style.background = 'var(--paper-deep,#e9e2d1)';
      });
      mi.addEventListener('mouseleave', () => { mi.style.background = ''; });
      mi.addEventListener('click', () => {
        const i = Number(mi.dataset.i);
        const it = items[i];
        m.style.display = 'none';
        if (it && it.onClick) it.onClick();
      });
    });
  }
  function closeMenu() { if (_menu) _menu.style.display = 'none'; }
  document.addEventListener('click', e => {
    if (_menu && !_menu.contains(e.target)) closeMenu();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });

  // Expose context-menu API for outer code (filters/drawer use it).
  APIN.charts._openMenu = openMenu;
  APIN.charts._closeMenu = closeMenu;
  APIN.charts._showTip = showTip;
  APIN.charts._hideTip = hideTip;

  // ─── 1. timeseries (line / stacked area) ─────────────────────────────
  /**
   * data: { buckets: [{t, values: {key: number, ...}}, ...],
   *         series_meta: [{key, label, color_token}, ...],
   *         granularity_seconds: 300 }
   * opts: { mode: 'line'|'stacked'|'area',
   *         showLegend: true,
   *         showGrid: true,
   *         logY: false,
   *         height: 280,
   *         onClickBucket: (bucket, seriesKey) => void,
   *         onContextBucket: (bucket, mouseEvent) => void,
   *         emptyMessage: '' }
   */
  APIN.charts.timeseries = function timeseries(host, data, opts) {
    opts = opts || {};
    const mode = opts.mode || 'line';
    const buckets = (data && data.buckets) || [];
    const meta = (data && data.series_meta) || [];
    if (buckets.length === 0) {
      host.innerHTML = '<div class="chart-empty">' + escHtml(opts.emptyMessage || 'no data in this window') + '</div>';
      return { type: 'timeseries', empty: true };
    }

    // Adaptive width: minimum 60px per bucket so dense time-series stays
    // legible. If the natural chart width exceeds the host's clientWidth,
    // the host scrolls horizontally (CSS `.chart-host{overflow-x:auto}`).
    // This is the 9.K-A scrollability fix the dashboard needed for ranges
    // like 7d/30d where 24 buckets squeezed into 820px were unreadable.
    const MIN_BUCKET_PX = 60;
    const minW = Math.max(720, (buckets.length || 1) * MIN_BUCKET_PX);
    const W = Math.max(host.clientWidth || 720, minW);
    const H = opts.height || 280;
    const padL = 50, padR = 14, padT = 12, padB = 30;
    const iw = W - padL - padR, ih = H - padT - padB;

    // Compute totals per series + per-bucket totals.
    const seriesKeys = meta.map(m => m.key);
    const data2D = buckets.map(b => {
      const o = {};
      let total = 0;
      for (const k of seriesKeys) {
        const v = Number((b.values || {})[k] || 0);
        o[k] = v; total += v;
      }
      return { t: b.t, values: o, total };
    });

    // Y-domain — 9.N.4 nice-numbered ticks (0/20/40/60/80, not 0/17/34/50/67)
    let yMaxRaw;
    if (mode === 'stacked') {
      yMaxRaw = Math.max(1, ...data2D.map(d => d.total));
    } else {
      yMaxRaw = 1;
      for (const d of data2D) {
        for (const k of seriesKeys) yMaxRaw = Math.max(yMaxRaw, d.values[k] || 0);
      }
    }
    const _nice = niceScale(yMaxRaw, 4);
    const yMax = _nice.max;
    const yTicks = _nice.ticks;
    const yScale = opts.logY ? (v => v <= 0 ? 0 : Math.log10(v + 1) / Math.log10(yMax + 1)) :
                                (v => v / yMax);

    const x = i => padL + (data2D.length <= 1 ? iw / 2 : (i / (data2D.length - 1)) * iw);
    const y = v => padT + ih - yScale(v) * ih;

    // Explicit width/height (not just viewBox) so the SVG physically occupies
    // its intrinsic width — required for horizontal scrollability via the
    // .chart-host wrapper. preserveAspectRatio="xMinYMid meet" keeps the
    // y-axis labels left-anchored when the host width matches W.
    let svg = `<svg class="chart-svg chart-timeseries" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMinYMid meet">`;

    // 9.N.4 · inject paper-ink defs (hatch patterns per series + wobble filter)
    // The series colors are resolved at paint-time via getComputedStyle so the
    // hatch picks up CSS-variable values. We pass the raw var() strings as
    // pattern strokes — browser resolves them against the chart's CSSOM.
    const seriesResolved = meta.map(m => colorVarFor(m.color_token));
    svg += paperInkDefs(seriesResolved);

    // 9.N.4 · Grid lines (nice-numbered ticks)
    if (opts.showGrid !== false) {
      yTicks.forEach(tv => {
        const ty = padT + ih - yScale(tv) * ih;
        svg += `<line class="chart-grid" x1="${padL}" y1="${ty}" x2="${padL + iw}" y2="${ty}" stroke="var(--paper-edge,#c7bca9)" stroke-width="0.6" stroke-dasharray="2 3" opacity="0.55"/>`;
        svg += `<text class="chart-axis" x="${padL - 8}" y="${ty + 4}" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="10.5" fill="var(--ink-soft,#6b6453)" style="font-variant-numeric:tabular-nums">${fmtNum(tv)}</text>`;
      });
      // Baseline axis stroke (slight wobble, hand-drawn)
      const baseY = padT + ih;
      svg += `<line x1="${padL}" y1="${baseY}" x2="${padL + iw}" y2="${baseY}" stroke="var(--ink-soft,#6b6453)" stroke-width="1" filter="url(#pi-wobble)"/>`;
    }

    // 9.N.4 · Render series — handdrawn paper-ink style
    // Strategy:
    //   single-series area mode → hatched area-fill (url(#pi-hatch-0)) + 1.6px stroke through wobble filter
    //   multi-line mode          → 1.6px strokes with halo, each through wobble filter
    //   stacked mode             → hatched fills per series (different patterns) + thin ink contours
    if (mode === 'stacked') {
      const stackTops = data2D.map(() => 0);
      meta.forEach((m, si) => {
        const color = colorVarFor(m.color_token);
        const cur = data2D.map((d, i) => {
          const yLo = padT + ih - yScale(stackTops[i]) * ih;
          stackTops[i] += d.values[m.key] || 0;
          const yHi = padT + ih - yScale(stackTops[i]) * ih;
          return { x: x(i), yLo, yHi };
        });
        if (cur.length === 0) return;
        let path = `M${cur[0].x},${cur[0].yHi}`;
        for (let i = 1; i < cur.length; i++) path += ` L${cur[i].x},${cur[i].yHi}`;
        let back = ` L${cur[cur.length - 1].x},${cur[cur.length - 1].yLo}`;
        for (let i = cur.length - 2; i >= 0; i--) back += ` L${cur[i].x},${cur[i].yLo}`;
        // Hatched fill (per-series pattern)
        svg += `<path d="${path}${back} Z" fill="url(#pi-hatch-${si})" stroke="${color}" stroke-width="1" stroke-linejoin="round" filter="url(#pi-wobble)"/>`;
      });
    } else if (mode === 'area' || meta.length === 1) {
      const m = meta[0] || { key: seriesKeys[0], color_token: 'ink' };
      const color = colorVarFor(m.color_token);
      let pts = data2D.map((d, i) => ({ x: x(i), y: y(d.values[m.key] || 0) }));
      let path = pts.length ? `M${pts[0].x},${pts[0].y}` : '';
      for (let i = 1; i < pts.length; i++) path += ` L${pts[i].x},${pts[i].y}`;
      let area = path + ` L${pts.length ? pts[pts.length - 1].x : padL},${padT + ih} L${pts.length ? pts[0].x : padL},${padT + ih} Z`;
      // Cross-hatched area fill (handdrawn, not gradient)
      svg += `<path d="${area}" fill="url(#pi-hatch-0)" stroke="none"/>`;
      // Outer halo stroke (1.4px @ 18% opacity) — lifts the line off the paper
      svg += `<path d="${path}" class="chart-line-halo" fill="none" stroke="${color}" stroke-width="3.4" stroke-linejoin="round" stroke-linecap="round" opacity="0.18" filter="url(#pi-wobble)"/>`;
      // Inner ink stroke (1.6px crisp)
      svg += `<path d="${path}" class="chart-line" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" filter="url(#pi-wobble)"/>`;
    } else {
      // Multi-line. Distinct linestyle per series: solid / dashed / dotted —
      // color-blind friendly even without color discrimination.
      // Class tags: .chart-line-halo (outer 3.4px halo at 16% opacity),
      //             .chart-line (1.6px crisp). The primary series (si=0) gets
      //             .chart-line-primary so the entry draw-path animation can
      //             target just one line (otherwise three lines drawing in
      //             parallel is visually busy).
      meta.forEach((m, si) => {
        const color = colorVarFor(m.color_token);
        const dash  = si === 0 ? null : si === 1 ? '6 4' : si === 2 ? '2 3' : '8 3 2 3';
        let pts = data2D.map((d, i) => ({ x: x(i), y: y(d.values[m.key] || 0) }));
        let path = pts.length ? `M${pts[0].x},${pts[0].y}` : '';
        for (let i = 1; i < pts.length; i++) path += ` L${pts[i].x},${pts[i].y}`;
        // Halo (under)
        svg += `<path class="chart-line-halo" d="${path}" fill="none" stroke="${color}" stroke-width="3.4" stroke-linejoin="round" stroke-linecap="round" opacity="0.16" filter="url(#pi-wobble)"/>`;
        // Crisp ink line
        const cls = si === 0 ? 'chart-line chart-line-primary' : 'chart-line';
        svg += `<path class="${cls}" d="${path}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"${dash ? ' stroke-dasharray="' + dash + '"' : ''} filter="url(#pi-wobble)"/>`;
      });
    }

    // 9.N.10 · Compare overlay — if data.prev_buckets is supplied, draw the
    // primary series of the previous period as a dashed ink-soft stroke.
    // Aligned to the SAME x-positions so visual comparison is direct.
    const prevBuckets = (data && data.prev_buckets) || null;
    if (prevBuckets && prevBuckets.length > 0 && meta.length > 0) {
      const primaryKey = meta[0].key;
      // Pad/truncate prev to same length as data2D for alignment
      const prev2D = prevBuckets.slice(-data2D.length);
      while (prev2D.length < data2D.length) prev2D.unshift({ t: '', values: {} });
      const pts = prev2D.map((b, i) => ({
        x: x(i),
        y: y(Number((b.values || {})[primaryKey] || 0)),
      }));
      let pathPrev = pts.length ? `M${pts[0].x},${pts[0].y}` : '';
      for (let i = 1; i < pts.length; i++) pathPrev += ` L${pts[i].x},${pts[i].y}`;
      svg += `<path class="chart-line-prev" d="${pathPrev}" fill="none" stroke="var(--ink-soft, #6b6453)" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="4 4" opacity="0.65" filter="url(#pi-wobble)"/>`;
    }

    // 9.N.4 · Marginalia annotation — automatic, on the global max point
    // of the primary series. Renders once after the chart paints. Adds the
    // "← peak N at HH:MM" callout in italic Fraunces.
    if (data2D.length >= 3 && opts.showMarginalia !== false) {
      // Find the index of the global max in the primary series (first meta key)
      const primaryKey = meta[0] && meta[0].key;
      if (primaryKey) {
        let maxI = 0, maxV = -Infinity;
        data2D.forEach((d, i) => {
          const v = d.values[primaryKey] || 0;
          if (v > maxV) { maxV = v; maxI = i; }
        });
        if (maxV > 0) {
          const mx = x(maxI), my = y(maxV);
          // Side: if peak is in left half → label points right, else left
          const side = (maxI < data2D.length / 2) ? 'right' : 'left';
          const peakTime = data2D[maxI].t ? tFmt(data2D[maxI].t) : '';
          svg += marginaliaLabel(mx, my, '← peak ' + fmtNum(maxV) +
                                  (peakTime ? ' · ' + peakTime : ''), side);
        }
      }
    }

    // Crosshair (hidden until hover) + hot bucket marker
    svg += `<line class="chart-cursor" id="${cssId(host, 'cursor')}" x1="0" y1="${padT}" x2="0" y2="${padT + ih}" stroke="var(--ink-soft,#5a5246)" stroke-width="1" stroke-dasharray="3 2" opacity="0"/>`;
    svg += `<circle class="chart-marker" id="${cssId(host, 'marker')}" r="0" fill="var(--ink,#1a1612)" stroke="var(--paper,#fbf9f3)" stroke-width="2"/>`;

    // X-axis labels: first, mid, last
    if (data2D.length >= 1) {
      const span = data2D[data2D.length - 1].t && data2D[0].t ?
        (new Date(data2D[data2D.length - 1].t.replace(' ', 'T') + 'Z') -
         new Date(data2D[0].t.replace(' ', 'T') + 'Z')) : 0;
      const fmt = span > 24 * 3600 * 1000 ? dFmt : tFmt;
      svg += `<text class="chart-axis" x="${x(0)}" y="${H - 8}" text-anchor="start">${escHtml(fmt(data2D[0].t))}</text>`;
      if (data2D.length > 2) {
        const mi = Math.floor(data2D.length / 2);
        svg += `<text class="chart-axis" x="${x(mi)}" y="${H - 8}" text-anchor="middle">${escHtml(fmt(data2D[mi].t))}</text>`;
      }
      svg += `<text class="chart-axis" x="${x(data2D.length - 1)}" y="${H - 8}" text-anchor="end">${escHtml(fmt(data2D[data2D.length - 1].t))}</text>`;
    }

    // Drag-to-zoom rectangle
    svg += `<rect class="chart-zoom-rect" id="${cssId(host, 'zoom')}" x="0" y="${padT}" width="0" height="${ih}" fill="var(--ink,#1a1612)" fill-opacity="0.08" stroke="var(--ink,#1a1612)" stroke-dasharray="3 2" pointer-events="none" opacity="0"/>`;

    // Capture rect (transparent — handles mouse events)
    svg += `<rect class="chart-capture" id="${cssId(host, 'capture')}" x="${padL}" y="${padT}" width="${iw}" height="${ih}" fill="transparent" style="cursor:crosshair"/>`;

    svg += '</svg>';

    // Legend (if multiple series)
    let legend = '';
    if (opts.showLegend !== false && meta.length > 1) {
      legend = '<div class="chart-legend">' + meta.map(m =>
        '<span class="chart-legend-item">' +
          '<span class="chart-swatch" style="background:' + colorVarFor(m.color_token) + '"></span>' +
          escHtml(m.label || m.key) +
        '</span>').join('') + '</div>';
    }

    host.innerHTML = svg + legend;

    // 9.N.4 · Animate-in: PRIMARY line draws via stroke-dasharray over 700ms.
    // Other lines (halos, secondaries) are already painted. Then marginalia
    // fades in 320ms later via WAAPI (with fill:'forwards' so it sticks at 1).
    const _primary = host.querySelector('.chart-line-primary') || host.querySelector('.chart-line');
    if (_primary && window.APIN && APIN.fx) {
      try { APIN.fx.drawPath(_primary, { duration: 700 }); } catch (e) {}
    }
    const _marg = host.querySelector('.chart-marginalia');
    if (_marg) {
      // SVG <g> opacity via WAAPI + fill:forwards is unreliable across browsers.
      // Use the SVG-native attribute as the steady-state, with a WAAPI animation
      // for the transition. Snapshot the attribute on finish() as a safety net.
      try {
        const a = _marg.animate(
          [{ opacity: 0 }, { opacity: 1 }],
          { duration: 320, delay: 700, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' }
        );
        if (a && a.finished) {
          a.finished.then(() => { _marg.setAttribute('opacity', '1'); }, () => {});
        }
        // Safety net: ensure visible after 1.2s regardless of animation state
        setTimeout(() => { _marg.setAttribute('opacity', '1'); }, 1200);
      } catch (e) {
        _marg.setAttribute('opacity', '1');
      }
    }

    // Attach interactions
    const svgEl = host.querySelector('svg.chart-timeseries');
    const cursor = host.querySelector('#' + cssId(host, 'cursor'));
    const marker = host.querySelector('#' + cssId(host, 'marker'));
    const zoomRect = host.querySelector('#' + cssId(host, 'zoom'));
    const capture = host.querySelector('#' + cssId(host, 'capture'));

    function coordsAt(clientX) {
      const r = svgEl.getBoundingClientRect();
      const px = (clientX - r.left) / r.width * W;
      // Snap to nearest bucket
      let best = 0, bestDx = Infinity;
      for (let i = 0; i < data2D.length; i++) {
        const xi = x(i);
        const dx = Math.abs(xi - px);
        if (dx < bestDx) { bestDx = dx; best = i; }
      }
      return { i: best, x: x(best), bucket: data2D[best] };
    }

    capture.addEventListener('mousemove', e => {
      const c = coordsAt(e.clientX);
      cursor.setAttribute('x1', c.x);
      cursor.setAttribute('x2', c.x);
      cursor.setAttribute('opacity', '0.7');
      // Marker: in line modes, pin to the line; in stacked, pin to the top
      const totalY = mode === 'stacked' ? y(c.bucket.total) :
        (meta.length === 1 ? y(c.bucket.values[meta[0].key] || 0) : y(c.bucket.values[meta[0].key] || 0));
      marker.setAttribute('cx', c.x);
      marker.setAttribute('cy', totalY);
      marker.setAttribute('r', '4');
      // Tooltip content
      let rows = '';
      for (const m of meta) {
        const v = c.bucket.values[m.key] || 0;
        rows += '<div style="display:flex;justify-content:space-between;gap:14px;align-items:center">' +
          '<span style="display:inline-flex;align-items:center;gap:6px">' +
            '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:' + colorVarFor(m.color_token) + '"></span>' +
            escHtml(m.label || m.key) +
          '</span>' +
          '<b style="font-variant-numeric:tabular-nums">' + fmtNum(v) + '</b>' +
        '</div>';
      }
      if (mode === 'stacked') {
        rows += '<div style="display:flex;justify-content:space-between;gap:14px;margin-top:4px;border-top:1px solid var(--paper-edge,#c7bca9);padding-top:3px"><span>total</span><b>' + fmtNum(c.bucket.total) + '</b></div>';
      }
      const tipHtml =
        '<div style="font-size:11px;color:var(--ink-soft);margin-bottom:4px">' +
          escHtml(c.bucket.t) + ' UTC</div>' + rows;
      showTip(e.clientX, e.clientY, tipHtml);
    });
    capture.addEventListener('mouseleave', () => {
      cursor.setAttribute('opacity', '0');
      marker.setAttribute('r', '0');
      hideTip();
    });

    // Click — drill into the bucket
    capture.addEventListener('click', e => {
      if (zoomingActive) return;
      const c = coordsAt(e.clientX);
      if (opts.onClickBucket) opts.onClickBucket(c.bucket, c.i);
    });

    // Right-click — context menu
    capture.addEventListener('contextmenu', e => {
      e.preventDefault();
      const c = coordsAt(e.clientX);
      const items = [
        { label: 'Drill into this minute', onClick: () => opts.onClickBucket && opts.onClickBucket(c.bucket, c.i) },
        { divider: true },
        { label: 'Copy timestamp', onClick: () => navigator.clipboard.writeText(c.bucket.t) },
      ];
      if (opts.onContextBucket) opts.onContextBucket(c.bucket, e, items);
      openMenu(e.clientX, e.clientY, items);
    });

    // Drag-to-zoom
    let zoomingActive = false, zoomStartX = null;
    capture.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      zoomingActive = false;
      zoomStartX = e.clientX;
      zoomRect.setAttribute('opacity', '0');
    });
    window.addEventListener('mousemove', e => {
      if (zoomStartX == null) return;
      const dx = e.clientX - zoomStartX;
      if (Math.abs(dx) > 6) zoomingActive = true;
      if (!zoomingActive) return;
      const r = svgEl.getBoundingClientRect();
      const x0 = (Math.min(zoomStartX, e.clientX) - r.left) / r.width * W;
      const x1 = (Math.max(zoomStartX, e.clientX) - r.left) / r.width * W;
      zoomRect.setAttribute('x', Math.max(padL, x0));
      zoomRect.setAttribute('width', Math.max(0, Math.min(padL + iw, x1) - Math.max(padL, x0)));
      zoomRect.setAttribute('opacity', '0.6');
    });
    window.addEventListener('mouseup', e => {
      if (zoomStartX == null) return;
      const wasZoom = zoomingActive;
      const startX = zoomStartX;
      zoomStartX = null;
      zoomingActive = false;
      zoomRect.setAttribute('opacity', '0');
      if (!wasZoom) return;
      const r = svgEl.getBoundingClientRect();
      const xa = (Math.min(startX, e.clientX) - r.left) / r.width * W;
      const xb = (Math.max(startX, e.clientX) - r.left) / r.width * W;
      // Convert to bucket index range
      function idxAt(px) {
        const u = (px - padL) / iw;
        return Math.max(0, Math.min(data2D.length - 1,
          Math.round(u * (data2D.length - 1))));
      }
      const lo = idxAt(xa);
      const hi = idxAt(xb);
      if (hi - lo < 1) return;
      if (opts.onZoom) opts.onZoom(data2D[lo], data2D[hi]);
    });

    return {
      type: 'timeseries',
      empty: false,
      data: data2D,
      exportSVG: () => svgEl ? svgEl.outerHTML : '',
    };
  };

  function cssId(host, key) {
    if (!host._chartUid) host._chartUid = 'ch' + Math.random().toString(36).slice(2, 8);
    return host._chartUid + '-' + key;
  }

  // ─── 2. donut ────────────────────────────────────────────────────────
  /**
   * items: [{ label, value, color_token }, ...]
   * opts: { size: 220, innerRatio: 0.6, totalLabel: 'Requests',
   *         onClickSlice: (item) => void }
   */
  APIN.charts.donut = function donut(host, items, opts) {
    opts = opts || {};
    const size = opts.size || 220;
    const innerRatio = opts.innerRatio || 0.6;
    const total = items.reduce((a, it) => a + (Number(it.value) || 0), 0);
    if (total === 0 || items.length === 0) {
      host.innerHTML = '<div class="chart-empty">no data</div>';
      return { type: 'donut', empty: true };
    }
    const cx = size / 2, cy = size / 2, r = size / 2 - 8, rInner = r * innerRatio;
    let angle = -Math.PI / 2;   // start at top
    const slices = items.map(it => {
      const frac = (Number(it.value) || 0) / total;
      const a1 = angle, a2 = angle + frac * Math.PI * 2;
      angle = a2;
      const large = (a2 - a1) > Math.PI ? 1 : 0;
      // outer arc
      const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
      const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2);
      // inner arc
      const xi1 = cx + rInner * Math.cos(a2), yi1 = cy + rInner * Math.sin(a2);
      const xi2 = cx + rInner * Math.cos(a1), yi2 = cy + rInner * Math.sin(a1);
      const d = `M${x1},${y1} A${r},${r} 0 ${large} 1 ${x2},${y2} ` +
                 `L${xi1},${yi1} A${rInner},${rInner} 0 ${large} 0 ${xi2},${yi2} Z`;
      return Object.assign({}, it, { d, frac, a1, a2 });
    });

    let svg = `<svg class="chart-svg chart-donut" viewBox="0 0 ${size} ${size + 6}" preserveAspectRatio="xMidYMid meet">`;
    // 9.N.4 · Each slice = flat ink fill + 1px ink-deep contour, separated
    // by a paper-color 2px stroke gap. NO 3D, NO shadow, NO gradient.
    // Entry animation = arc draw (via stroke-dasharray on outer-edge arc),
    // sliced opacity 0→1, staggered 70ms per slice. We render the slices
    // with full d but `opacity:0` initially; .donut-slice-arc is a separate
    // overlay path that draws the outer arc as a stroke for the entry FX.
    slices.forEach((s, i) => {
      const fill = colorVarFor(s.color_token || 'ink');
      svg += `<path class="donut-slice" data-i="${i}" d="${s.d}" fill="${fill}" stroke="var(--paper,#fbf9f3)" stroke-width="2" stroke-linejoin="round" style="opacity:0;transform:scale(0.97);transform-origin:${cx}px ${cy}px;transition:opacity .50s cubic-bezier(.22,1,.36,1) ${i * 70}ms, transform .50s cubic-bezier(.22,1,.36,1) ${i * 70}ms"/>`;
      // 1px ink-deep contour overlay (drawn ON TOP for crispness)
      svg += `<path class="donut-contour" d="${s.d}" fill="none" stroke="var(--ink, #1a1612)" stroke-width="1" stroke-opacity="0.42" pointer-events="none" style="opacity:0;transition:opacity .42s cubic-bezier(.22,1,.36,1) ${(i * 70) + 280}ms"/>`;
    });
    // 9.N.4 · Stamped serif center label — Fraunces total + italic "requests" caption.
    // Held inside a <g> so we can fade it in after slices land.
    svg += '<g class="donut-stamp" style="opacity:0;transition:opacity .32s cubic-bezier(.22,1,.36,1) ' + (slices.length * 70 + 320) + 'ms">';
    svg += `<text x="${cx}" y="${cy + 4}" text-anchor="middle" font-family="Fraunces,serif" font-size="36" fill="var(--ink,#1a1612)" font-weight="500" letter-spacing="-0.01em">${fmtNum(total)}</text>`;
    svg += `<text x="${cx}" y="${cy + 24}" text-anchor="middle" font-family="Fraunces,serif" font-style="italic" font-size="11" fill="var(--ink-soft,#6b6453)" letter-spacing="0.02em">${escHtml(opts.totalLabel || 'requests')}</text>`;
    svg += '</g>';
    svg += '</svg>';

    // Legend
    let legend = '<div class="chart-legend chart-legend-vertical">';
    for (const it of items) {
      const pct = total ? ((Number(it.value) || 0) / total * 100).toFixed(1) : '0';
      legend += '<div class="chart-legend-item">' +
        '<span class="chart-swatch" style="background:' + colorVarFor(it.color_token || 'ink') + '"></span>' +
        '<span style="flex:1">' + escHtml(it.label) + '</span>' +
        '<b style="font-family:JetBrains Mono,monospace;font-variant-numeric:tabular-nums">' + fmtNum(it.value) + '</b>' +
        '<span style="color:var(--ink-soft);font-family:JetBrains Mono,monospace">' + pct + '%</span>' +
      '</div>';
    }
    legend += '</div>';

    host.innerHTML =
      '<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center">' +
        '<div style="flex:0 0 auto">' + svg + '</div>' +
        '<div style="flex:1;min-width:160px">' + legend + '</div>' +
      '</div>';

    // 9.N.4 · Trigger entry: slice opacity→1 + contour reveal + center stamp.
    requestAnimationFrame(() => {
      host.querySelectorAll('.donut-slice').forEach(p => {
        p.style.opacity = '1';
        p.style.transform = 'scale(1)';
      });
      host.querySelectorAll('.donut-contour').forEach(p => { p.style.opacity = '1'; });
      const stamp = host.querySelector('.donut-stamp');
      if (stamp) stamp.style.opacity = '1';
    });

    // 9.N.4 · Hover = radial pop 6px + brightness lift + soft inkshadow.
    // The transform applies via getCTM-relative offset so we don't have to
    // recompute the arc path.
    host.querySelectorAll('.donut-slice').forEach((p, i) => {
      const it = slices[i];
      // Compute radial offset vector for hover-pop
      const mid = (it.a1 + it.a2) / 2;
      const popX = Math.cos(mid) * 6, popY = Math.sin(mid) * 6;
      p.style.transformBox = 'fill-box';
      p.addEventListener('mouseenter', e => {
        p.style.transition = 'transform .18s cubic-bezier(.22,1,.36,1), filter .18s';
        p.style.transformOrigin = cx + 'px ' + cy + 'px';
        p.style.transform = `translate(${popX.toFixed(2)}px, ${popY.toFixed(2)}px)`;
        p.style.filter = 'drop-shadow(0 2px 4px rgba(20,16,12,0.18))';
        const pct = ((it.value / total) * 100).toFixed(1);
        showTip(e.clientX, e.clientY,
          '<b style="font-family:Fraunces,serif;font-weight:500">' + escHtml(it.label) + '</b><br>' +
          '<span style="font-family:JetBrains Mono,monospace">' + fmtNum(it.value) + ' · ' + pct + '%</span>');
      });
      p.addEventListener('mousemove', e => {
        if (_tip) showTip(e.clientX, e.clientY, _tip.innerHTML);
      });
      p.addEventListener('mouseleave', () => {
        p.style.transform = 'translate(0,0)';
        p.style.filter = '';
        hideTip();
      });
      p.style.cursor = 'pointer';
      p.addEventListener('click', () => {
        if (window.APIN && APIN.fx) {
          const r = p.getBoundingClientRect();
          APIN.fx.ripple(p.parentElement || host, r.left + r.width/2, r.top + r.height/2);
        }
        if (opts.onClickSlice) opts.onClickSlice(it);
      });
    });
    return { type: 'donut', empty: false };
  };

  // ─── 3. histogram (latency buckets etc.) ─────────────────────────────
  /**
   * buckets: [{ label, value, color_token?, hint? }, ...]
   * opts: { height: 200, onClickBar: (bucket) => void, yLabel: '' }
   */
  APIN.charts.histogram = function histogram(host, buckets, opts) {
    opts = opts || {};
    if (!buckets || buckets.length === 0) {
      host.innerHTML = '<div class="chart-empty">no data</div>';
      return { type: 'histogram', empty: true };
    }
    const W = host.clientWidth || 600, H = opts.height || 220;
    // 9.N.4 · larger padT/padB for percentile thread-line labels and rotated x labels
    const padL = 44, padR = 12, padT = 22, padB = 56;
    const iw = W - padL - padR, ih = H - padT - padB;
    const rawMax = Math.max(1, ...buckets.map(b => Number(b.value) || 0));
    const _nice = niceScale(rawMax, 4);
    const yMax = _nice.max;
    const yTicks = _nice.ticks;
    const barW = iw / buckets.length * 0.74;

    let svg = `<svg class="chart-svg chart-histogram" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
    // 9.N.4 · paper-ink defs
    const histColors = buckets.map(b => colorVarFor(b.color_token || 'ink'));
    svg += paperInkDefs(histColors);

    // Grid (nice ticks)
    yTicks.forEach(tv => {
      const ty = padT + ih - (tv / yMax) * ih;
      svg += `<line x1="${padL}" y1="${ty}" x2="${padL + iw}" y2="${ty}" stroke="var(--paper-edge,#c7bca9)" stroke-width="0.6" stroke-dasharray="2 3" opacity="0.55"/>`;
      svg += `<text x="${padL - 6}" y="${ty + 4}" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="10.5" fill="var(--ink-soft,#6b6453)" style="font-variant-numeric:tabular-nums">${fmtNum(tv)}</text>`;
    });
    // Baseline
    svg += `<line x1="${padL}" y1="${padT + ih}" x2="${padL + iw}" y2="${padT + ih}" stroke="var(--ink-soft,#6b6453)" stroke-width="1" filter="url(#pi-wobble)"/>`;

    // 9.N.5 fix · Bars — visible stipple-fill + bold ink top edge.
    // Earlier the stipple at 3-dots / 8×8 was too sparse to read; bumped
    // to the denser pi-stipple-3 + a higher base wash + thicker top edge
    // (2px) that extends 3px past the bar on each side as an editorial mark.
    buckets.forEach((b, i) => {
      const v = Number(b.value) || 0;
      const xpos = padL + (i + 0.5) * (iw / buckets.length) - barW / 2;
      const yEnd = padT + ih - (v / yMax) * ih;
      const h = padT + ih - yEnd;
      const color = colorVarFor(b.color_token || 'ink');
      svg += `<g class="hist-bar-group" data-i="${i}" style="color:${color};cursor:pointer">`;
      // Solid color wash — more opaque base for visual weight
      svg += `<rect class="hist-bar-wash" x="${xpos}" y="${padT + ih}" width="${barW}" height="0" fill="${color}" fill-opacity="0.34"/>`;
      // Stipple overlay (denser) — adds texture without losing the color
      svg += `<rect class="hist-bar-stipple" x="${xpos}" y="${padT + ih}" width="${barW}" height="0" fill="url(#pi-stipple-3)"/>`;
      // Top edge ink line — 2px thick, extends 3px past bar each side
      svg += `<line class="hist-bar-top" x1="${xpos - 3}" y1="${padT + ih}" x2="${xpos + barW + 3}" y2="${padT + ih}" stroke="${color}" stroke-width="2" stroke-linecap="round"/>`;
      // Value label on top of bar (when bar is tall enough)
      svg += `<text class="hist-bar-val" x="${xpos + barW/2}" y="${padT + ih}" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="10.5" fill="var(--ink,#1a1612)" style="font-variant-numeric:tabular-nums;opacity:0">${fmtNum(v)}</text>`;
      svg += `</g>`;
      // X label — rotated -28° + italic Fraunces
      const lx = xpos + barW / 2;
      const ly = padT + ih + 12;
      svg += `<text class="hist-x-label" x="${lx}" y="${ly}" text-anchor="end" transform="rotate(-28 ${lx} ${ly})" font-family="Fraunces,serif" font-style="italic" font-size="10.5" fill="var(--ink-soft,#6b6453)" letter-spacing="0.02em">${escHtml(b.label)}</text>`;
    });

    // 9.N.4 · Percentile threads (dashed vertical lines + serif callouts)
    // opts.percentiles = [{ label: 'p50', value: 176, color_token: 'amber' }, ...]
    // value is interpolated against the histogram bucket positions if the
    // caller passes opts.bucketBoundaries (array of cumulative thresholds in ms).
    if (opts.percentiles && opts.bucketBoundaries) {
      // bucketBoundaries: e.g. [50, 100, 200, 500, 1000, 2000, 5000, Infinity]
      function xFromMs(ms) {
        // Find the bucket the ms falls into, then interpolate within that bucket's width
        const bounds = opts.bucketBoundaries;
        let lo = 0;
        for (let i = 0; i < bounds.length; i++) {
          const hi = bounds[i];
          if (ms <= hi || !Number.isFinite(hi)) {
            const t = Number.isFinite(hi) && hi !== lo ? (ms - lo) / (hi - lo) : 0.5;
            const cellW = iw / buckets.length;
            return padL + (i + Math.max(0, Math.min(1, t))) * cellW;
          }
          lo = hi;
        }
        return padL + iw;
      }
      opts.percentiles.forEach(p => {
        const px = xFromMs(p.value);
        const pc = colorVarFor(p.color_token || 'amber');
        svg += `<line x1="${px}" y1="${padT - 8}" x2="${px}" y2="${padT + ih}" stroke="${pc}" stroke-width="1" stroke-dasharray="3 3" opacity="0.78"/>`;
        svg += `<text x="${px}" y="${padT - 10}" text-anchor="middle" font-family="Fraunces,serif" font-style="italic" font-size="10.5" fill="${pc}">${escHtml(p.label)} · ${fmtNum(p.value)}ms</text>`;
      });
    }

    // 9.N.5 fix · smooth fit line — a density curve interpolating bucket tops
    // using a Catmull-Rom spline. Helps the eye see the shape of the
    // distribution at a glance. Reactive to data: re-computes per render.
    if (buckets.length >= 3) {
      const pts = buckets.map((b, i) => {
        const v = Number(b.value) || 0;
        const xc = padL + (i + 0.5) * (iw / buckets.length);
        const yc = padT + ih - (v / yMax) * ih;
        return { x: xc, y: yc };
      });
      // Catmull-Rom → Cubic Bezier conversion
      let d = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[Math.max(0, i - 1)];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = pts[Math.min(pts.length - 1, i + 2)];
        const c1x = p1.x + (p2.x - p0.x) / 6;
        const c1y = p1.y + (p2.y - p0.y) / 6;
        const c2x = p2.x - (p3.x - p1.x) / 6;
        const c2y = p2.y - (p3.y - p1.y) / 6;
        d += ` C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
      }
      svg += `<path class="hist-fit-line" d="${d}" fill="none" stroke="var(--ink,#1a1612)" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="0" opacity="0.78" filter="url(#pi-wobble)"/>`;
    }

    svg += '</svg>';
    host.innerHTML = svg;

    // 9.N.5 fix · animate the fit line drawing in over 700ms after bars settle
    const fit = host.querySelector('.hist-fit-line');
    if (fit && window.APIN && APIN.fx) {
      try {
        const len = fit.getTotalLength();
        fit.style.strokeDasharray = String(len);
        fit.style.strokeDashoffset = String(len);
        fit.animate(
          [{ strokeDashoffset: len }, { strokeDashoffset: 0 }],
          { duration: 700, delay: 60 * buckets.length + 200, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'forwards' }
        );
      } catch (e) {}
    }

    // 9.N.4 · Bar entry animation — wash + stipple grow from baseline to value
    // 9.N.5 fix · value label also rises with the bar + fades in
    const bars = host.querySelectorAll('.hist-bar-group');
    bars.forEach((g, i) => {
      const v = Number(buckets[i].value) || 0;
      const targetH = (v / yMax) * ih;
      const targetY = padT + ih - targetH;
      const wash    = g.querySelector('.hist-bar-wash');
      const stipple = g.querySelector('.hist-bar-stipple');
      const topLine = g.querySelector('.hist-bar-top');
      const valText = g.querySelector('.hist-bar-val');
      const delay   = 60 * i;
      const easing = 'cubic-bezier(0.22, 1, 0.36, 1)';
      if (wash) wash.animate(
        [{ y: padT + ih, height: 0 }, { y: targetY, height: targetH }],
        { duration: 450, delay, easing, fill: 'forwards' }
      );
      if (stipple) stipple.animate(
        [{ y: padT + ih, height: 0 }, { y: targetY, height: targetH }],
        { duration: 450, delay, easing, fill: 'forwards' }
      );
      if (topLine) topLine.animate(
        [{ y1: padT + ih, y2: padT + ih }, { y1: targetY, y2: targetY }],
        { duration: 450, delay, easing, fill: 'forwards' }
      );
      if (valText && v > 0) {
        valText.animate(
          [{ y: padT + ih, opacity: 0 }, { y: targetY - 4, opacity: 1 }],
          { duration: 450, delay: delay + 200, easing, fill: 'forwards' }
        );
      }
    });

    // Hover + click handlers
    bars.forEach((g, i) => {
      const it = buckets[i];
      const topLine = g.querySelector('.hist-bar-top');
      g.addEventListener('mouseenter', e => {
        if (topLine) topLine.style.strokeWidth = '2.4';
        const wash = g.querySelector('.hist-bar-wash');
        if (wash) wash.setAttribute('fill-opacity', '0.32');
        showTip(e.clientX, e.clientY,
          '<b style="font-family:Fraunces,serif">' + escHtml(it.label) + '</b><br>' +
          '<span style="font-family:JetBrains Mono,monospace">' + fmtNum(it.value) + '</span>' +
          (it.hint ? '<br><span style="color:var(--ink-soft);font-style:italic">' + escHtml(it.hint) + '</span>' : ''));
      });
      g.addEventListener('mousemove', e => { if (_tip) showTip(e.clientX, e.clientY, _tip.innerHTML); });
      g.addEventListener('mouseleave', () => {
        if (topLine) topLine.style.strokeWidth = '1.2';
        const wash = g.querySelector('.hist-bar-wash');
        if (wash) wash.setAttribute('fill-opacity', '0.18');
        hideTip();
      });
      g.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(host, e.clientX, e.clientY);
        if (opts.onClickBar) opts.onClickBar(it);
      });
    });
    return { type: 'histogram', empty: false };
  };

  // ─── 4. top-N horizontal bars ────────────────────────────────────────
  /**
   * items: [{ label, count, pct, extra?: {…} }, ...]
   * opts: { height: auto, onClickItem: (item) => void, color_token: 'ink' }
   */
  APIN.charts.topBar = function topBar(host, items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) {
      host.innerHTML = '<div class="chart-empty">no data</div>';
      return { type: 'topBar', empty: true };
    }
    const max = Math.max(1, ...items.map(i => Number(i.count) || 0));
    // 9.N.4 · Grid 1fr / 100px / 60px (was 60/50). Tabular nums. Arrow slot.
    const rows = items.map((it, idx) => {
      const w = ((Number(it.count) || 0) / max) * 100;
      const pct = it.pct != null ? Number(it.pct).toFixed(1) : '';
      const color = colorVarFor(opts.color_token || (idx < 5 ? 'series-' + idx : 'muted'));
      // 9.N.5 fix · No cream wrapper. Just a colored bar on paper, with the
      // label inside the bar (or to the right of it for short bars). The bar
      // itself IS the visualization — no background container.
      return '<div class="topbar-row" data-i="' + idx + '" style="display:grid;'
        + 'grid-template-columns:1fr 100px 60px 18px;gap:12px;align-items:center;'
        + 'padding:7px 8px;cursor:pointer;border-radius:0;'
        + 'border-bottom:1px solid var(--paper-edge,#c7bca9);'
        + 'transition:background .15s cubic-bezier(0.22, 1, 0.36, 1)">' +
        // Label cell: paper background, colored bar inside grows from 0% width.
        // Label sits ABOVE the bar in z-order, ink-deep text for crisp reading.
        '<div style="position:relative;font-family:\'JetBrains Mono\',monospace;font-size:12.5px;'
          + 'color:var(--ink);overflow:hidden;padding:6px 10px;min-width:0;'
          + 'border:1px solid var(--paper-edge,#c7bca9)">' +
          '<div class="topbar-fill" style="position:absolute;left:0;top:0;height:100%;'
            + 'width:0%;background:' + color + ';opacity:0.78;'
            + 'transition:width .4s cubic-bezier(0.22, 1, 0.36, 1) ' + (60*idx) + 'ms"></div>' +
          '<span style="position:relative;display:block;overflow:hidden;text-overflow:ellipsis;'
            + 'white-space:nowrap;font-weight:500">' + escHtml(it.label) + '</span>' +
        '</div>' +
        // Count cell — tabular-nums + slashed-zero
        '<b class="topbar-count" style="font-family:\'JetBrains Mono\',monospace;'
          + 'font-variant-numeric:tabular-nums slashed-zero;text-align:right;'
          + 'font-weight:600;font-size:13px;color:var(--ink)">' +
          fmtNum(it.count) + '</b>' +
        // Percent cell
        '<span class="topbar-pct" style="color:var(--ink-soft);'
          + 'font-family:\'JetBrains Mono\',monospace;'
          + 'font-variant-numeric:tabular-nums;text-align:right;font-size:11.5px">' +
          (pct ? pct + '%' : '') + '</span>' +
        // Arrow slot — slides in on hover
        '<span class="topbar-arrow" style="font-family:\'JetBrains Mono\',monospace;'
          + 'color:var(--ink-soft);text-align:right;font-size:13px;opacity:0;'
          + 'transform:translateX(-6px);transition:opacity .15s cubic-bezier(0.22, 1, 0.36, 1),'
          + 'transform .15s cubic-bezier(0.22, 1, 0.36, 1)">→</span>' +
      '</div>';
    }).join('');
    host.innerHTML = '<div class="chart-topbar">' + rows + '</div>';
    // 9.N.4 · Trigger bar fill animation on next frame
    requestAnimationFrame(() => {
      host.querySelectorAll('.topbar-row').forEach((row, i) => {
        const it = items[i];
        const fill = row.querySelector('.topbar-fill');
        const w = ((Number(it.count) || 0) / max) * 100;
        if (fill) fill.style.width = w + '%';
      });
    });
    host.querySelectorAll('.topbar-row').forEach((row, i) => {
      const it = items[i];
      const arrow = row.querySelector('.topbar-arrow');
      row.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(row, e.clientX, e.clientY);
        if (opts.onClickItem) opts.onClickItem(it);
      });
      row.addEventListener('mouseenter', e => {
        // 9.N.4 · Row hover: paper-deep bg fade + slide-in arrow
        row.style.background = 'var(--paper-deep, #e9e2d1)';
        if (arrow) {
          arrow.style.opacity = '1';
          arrow.style.transform = 'translateX(0)';
          arrow.style.color = 'var(--ink, #1a1612)';
        }
        const extras = (it.extra && Object.keys(it.extra).length)
          ? ('<br>' + Object.entries(it.extra).map(([k, v]) =>
              '<span style="color:var(--ink-soft);font-style:italic">' + escHtml(k) + ':</span> ' + escHtml(String(v))).join(' · '))
          : '';
        showTip(e.clientX, e.clientY,
          '<b style="font-family:Fraunces,serif">' + escHtml(it.label) + '</b><br>' +
          '<span style="font-family:\'JetBrains Mono\',monospace">' + fmtNum(it.count) +
          (it.pct != null ? ' · ' + Number(it.pct).toFixed(1) + '%' : '') + '</span>' +
          extras);
      });
      row.addEventListener('mousemove', e => {
        if (_tip) showTip(e.clientX, e.clientY, _tip.innerHTML);
      });
      row.addEventListener('mouseleave', () => {
        row.style.background = '';
        if (arrow) {
          arrow.style.opacity = '0';
          arrow.style.transform = 'translateX(-6px)';
        }
        hideTip();
      });
    });
    return { type: 'topBar', empty: false };
  };

  // ─── 5. sparkline (compact line) ─────────────────────────────────────
  APIN.charts.sparkline = function sparkline(host, items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) {
      host.innerHTML = '<div class="chart-empty" style="height:40px;display:flex;align-items:center;justify-content:center">·</div>';
      return { type: 'sparkline', empty: true };
    }
    const W = host.clientWidth || 200, H = opts.height || 50, pad = 2;
    const valueKey = opts.valueKey || 'requests';
    const max = Math.max(1, ...items.map(x => Number(x[valueKey]) || 0));
    const step = (W - 2 * pad) / Math.max(1, items.length - 1);
    let path = '';
    items.forEach((x, i) => {
      const px = pad + i * step;
      const py = H - pad - ((Number(x[valueKey]) || 0) / max) * (H - 2 * pad);
      path += (i === 0 ? 'M' : ' L') + px.toFixed(1) + ',' + py.toFixed(1);
    });
    const color = colorVarFor(opts.color_token || 'ink');
    host.innerHTML =
      `<svg class="chart-sparkline" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:${H}px;display:block">` +
        `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>` +
      `</svg>`;
    return { type: 'sparkline', empty: false };
  };

  // ════════════════════════════════════════════════════════════════════════
  // ─── 9.N.5 · Six new chart renderers in paper-ink language ──────────────
  // ════════════════════════════════════════════════════════════════════════

  // ─── 6. activityHeatmap — day-of-week × hour-of-day grid ─────────────────
  /**
   * data: { cells: [{ day_idx: 0-6, hour: 0-23, count: N }, ...] }
   *       day_idx 0 = Monday (ISO).
   * opts: { onClickCell: ({day_idx, hour, count}) => void }
   */
  APIN.charts.activityHeatmap = function activityHeatmap(host, data, opts) {
    opts = opts || {};
    const cells = (data && data.cells) || [];
    if (cells.length === 0) {
      host.innerHTML = '<div class="chart-empty" style="padding:24px;font-style:italic;color:var(--ink-soft,#6b6453)">no activity in this window</div>';
      return { type: 'activityHeatmap', empty: true };
    }
    // 9.N.5.f · Multi-mode support. Use data.rows/cols + row_labels/col_labels
    // from the multi-mode endpoint when present; fall back to the old 7×24
    // shape for backward compat.
    const gridRows = (data && Number(data.rows)) || 7;
    const gridCols = (data && Number(data.cols)) || 24;
    const dayLabels = (data && data.row_labels) || ['MON','TUE','WED','THU','FRI','SAT','SUN'];
    const colLabelsRaw = (data && data.col_labels) || [...Array(gridCols)].map((_, i) => String(i).padStart(2, '0'));
    const W = host.clientWidth || 720;
    const padL = 60, padR = 12, padT = 26, padB = 8;
    const cellW = (W - padL - padR) / gridCols;
    const cellH = gridRows >= 5 ? 22 : (gridRows >= 3 ? 30 : 44);
    const H = padT + padB + gridRows * cellH + 18;
    // Build grid populated from cells
    const grid = Array.from({length: gridRows}, () => Array.from({length: gridCols}, () => 0));
    const cellMeta = Array.from({length: gridRows}, () => Array.from({length: gridCols}, () => null));
    let maxV = 0;
    cells.forEach(c => {
      // Accept both new {row, col} and legacy {day_idx, hour} shapes
      const d = c.row != null ? c.row : c.day_idx;
      const h = c.col != null ? c.col : c.hour;
      const v = Number(c.count) || 0;
      if (d >= 0 && d < gridRows && h >= 0 && h < gridCols) {
        grid[d][h] = v;
        cellMeta[d][h] = c;
        if (v > maxV) maxV = v;
      }
    });
    // Density tiers: 0 / 1-25% / 25-50% / 50-100% (relative to max)
    function tier(v) {
      if (v === 0) return 0;
      const p = v / Math.max(1, maxV);
      if (p < 0.1) return 1;
      if (p < 0.33) return 2;
      if (p < 0.66) return 3;
      return 4;
    }
    let svg = `<svg class="chart-svg chart-heatmap" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="${W}" height="${H}">`;
    svg += paperInkDefs([colorVarFor('ink')]);
    // Column labels (top) — show every Nth label to avoid crowding
    const colLabelStride = gridCols >= 24 ? 3 : gridCols >= 12 ? 1 : 1;
    for (let h = 0; h < gridCols; h++) {
      if (h % colLabelStride === 0 || h === gridCols - 1) {
        const hx = padL + h * cellW + cellW / 2;
        svg += `<text x="${hx}" y="${padT - 8}" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9.5" fill="var(--ink-soft,#6b6453)">${escHtml(colLabelsRaw[h] || '')}</text>`;
      }
    }
    // Row labels (left)
    for (let d = 0; d < gridRows; d++) {
      const dy = padT + d * cellH + cellH / 2 + 4;
      svg += `<text x="${padL - 12}" y="${dy}" text-anchor="end" font-family="Fraunces,serif" font-style="italic" font-size="10.5" letter-spacing="0.08em" fill="var(--ink-soft,#6b6453)">${escHtml((dayLabels[d] || '').toUpperCase())}</text>`;
    }
    // Cells
    for (let d = 0; d < gridRows; d++) {
      for (let h = 0; h < gridCols; h++) {
        const v = grid[d][h];
        const t = tier(v);
        const x = padL + h * cellW;
        const y = padT + d * cellH;
        const innerPad = 1.5;
        const meta = cellMeta[d][h] || {};
        const cellLabel = meta.label || (colLabelsRaw[h] || '') || '';
        if (t === 0) {
          // Empty cell — small centered dot (still clickable so users can drill into "no traffic" cells)
          svg += `<circle cx="${x + cellW/2}" cy="${y + cellH/2}" r="0.9" fill="var(--paper-edge,#c7bca9)"/>`;
          svg += `<rect class="hm-cell" data-row="${d}" data-col="${h}" data-count="0" data-label="${escHtml(cellLabel)}" `
              + `x="${x + innerPad}" y="${y + innerPad}" width="${cellW - 2*innerPad}" height="${cellH - 2*innerPad}" `
              + `fill="transparent" style="cursor:pointer"/>`;
        } else {
          const ops = { 1: 0.22, 2: 0.44, 3: 0.66, 4: 0.88 };
          svg += `<rect class="hm-cell" data-row="${d}" data-col="${h}" data-count="${v}" data-label="${escHtml(cellLabel)}" `
              + `x="${x + innerPad}" y="${y + innerPad}" width="${cellW - 2*innerPad}" height="${cellH - 2*innerPad}" `
              + `fill="var(--ink,#1a1612)" fill-opacity="${ops[t]}" stroke="var(--ink,#1a1612)" stroke-opacity="0.12" stroke-width="0.5" style="cursor:pointer"/>`;
        }
      }
    }
    // Legend (bottom-right)
    const legY = padT + gridRows * cellH + 16;
    let lx = padL;
    svg += `<text x="${lx}" y="${legY}" font-family="Fraunces,serif" font-style="italic" font-size="10" fill="var(--ink-soft,#6b6453)">less</text>`;
    lx += 30;
    [0.22, 0.44, 0.66, 0.88].forEach(op => {
      svg += `<rect x="${lx}" y="${legY - 9}" width="11" height="11" fill="var(--ink,#1a1612)" fill-opacity="${op}"/>`;
      lx += 14;
    });
    svg += `<text x="${lx + 2}" y="${legY}" font-family="Fraunces,serif" font-style="italic" font-size="10" fill="var(--ink-soft,#6b6453)">more</text>`;
    // Max-value caption on the right
    if (maxV > 0) {
      svg += `<text x="${W - padR}" y="${legY}" text-anchor="end" font-family="Fraunces,serif" font-style="italic" font-size="10" fill="var(--ink-soft,#6b6453)">peak ${fmtNum(maxV)} requests</text>`;
    }
    svg += '</svg>';
    host.innerHTML = svg;
    // Hover + click
    host.querySelectorAll('.hm-cell').forEach(c => {
      c.addEventListener('mouseenter', e => {
        if (c.getAttribute('fill') !== 'transparent') {
          c.setAttribute('stroke', 'var(--ink, #1a1612)');
          c.setAttribute('stroke-opacity', '0.7');
          c.setAttribute('stroke-width', '1.4');
        }
        const d = Number(c.getAttribute('data-row'));
        const h = Number(c.getAttribute('data-col'));
        const v = Number(c.getAttribute('data-count'));
        const lbl = c.getAttribute('data-label') || '';
        const titleA = dayLabels[d] || ('row ' + d);
        const titleB = lbl || colLabelsRaw[h] || ('col ' + h);
        showTip(e.clientX, e.clientY,
          '<b style="font-family:Fraunces,serif">' + escHtml(titleA) + ' · ' + escHtml(titleB) + '</b><br>'
          + '<span style="font-family:\'JetBrains Mono\',monospace">' + fmtNum(v) + ' requests</span>');
      });
      c.addEventListener('mousemove', e => { if (_tip) showTip(e.clientX, e.clientY, _tip.innerHTML); });
      c.addEventListener('mouseleave', () => {
        if (c.getAttribute('fill') !== 'transparent') {
          c.setAttribute('stroke', 'var(--ink, #1a1612)');
          c.setAttribute('stroke-opacity', '0.12');
          c.setAttribute('stroke-width', '0.5');
        }
        hideTip();
      });
      c.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(host, e.clientX, e.clientY);
        if (opts.onClickCell) opts.onClickCell({
          row: Number(c.getAttribute('data-row')),
          col: Number(c.getAttribute('data-col')),
          count: Number(c.getAttribute('data-count')),
          label: c.getAttribute('data-label') || '',
          // Legacy aliases for week mode
          day_idx: Number(c.getAttribute('data-row')),
          hour: Number(c.getAttribute('data-col')),
        });
      });
    });
    return { type: 'activityHeatmap', empty: false };
  };

  // ─── 7. sparkGrid — per-endpoint small-multiples sparklines ──────────────
  /**
   * items: [{ label, count, pct, sparkline: [v, v, ...] }, ...]
   * opts: { onClickItem: (item) => void }
   */
  APIN.charts.sparkGrid = function sparkGrid(host, items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) {
      host.innerHTML = '<div class="chart-empty" style="padding:24px;color:var(--ink-soft,#6b6453);font-style:italic">no endpoint activity</div>';
      return { type: 'sparkGrid', empty: true };
    }
    // 9.N.5 fix · Each row's sparkline self-scales (max within its OWN row)
    // so endpoints with tiny absolute counts still show their shape. Add
    // a hatch-fill area beneath the line for visual weight + an end-accent dot.
    const rows = items.map((it, idx) => {
      const spark = (it.sparkline || []).map(v => Number(v) || 0);
      const max = Math.max(1, ...spark);
      const sparkW = 140, sparkH = 26, pad = 2;
      let path = '', area = '';
      if (spark.length >= 2) {
        const step = (sparkW - 2 * pad) / (spark.length - 1);
        const pts = spark.map((v, i) => [
          pad + i * step,
          sparkH - pad - (v / max) * (sparkH - 2 * pad)
        ]);
        path = 'M' + pts[0][0].toFixed(1) + ',' + pts[0][1].toFixed(1);
        for (let i = 1; i < pts.length; i++) path += ' L' + pts[i][0].toFixed(1) + ',' + pts[i][1].toFixed(1);
        // Area path (line → bottom-right → bottom-left → close)
        const lastX = pts[pts.length - 1][0];
        const firstX = pts[0][0];
        const baseY = sparkH - pad;
        area = path + ' L' + lastX.toFixed(1) + ',' + baseY.toFixed(1)
                    + ' L' + firstX.toFixed(1) + ',' + baseY.toFixed(1) + ' Z';
      }
      const endDot = spark.length >= 2 ? {
        x: pad + (sparkW - 2*pad),
        y: sparkH - pad - (spark[spark.length-1] / max) * (sparkH - 2 * pad)
      } : null;
      const pct = it.pct != null ? Number(it.pct).toFixed(1) : '';
      // Health tint based on error rate (matches treemap color logic)
      const er = Number(it.error_rate);
      const tint = Number.isFinite(er) ? (er > 0.05 ? 'var(--c-danger,#b01820)' : er > 0 ? 'var(--c-amber,#d49620)' : 'var(--c-ok,#2f6f3e)') : 'var(--ink,#1a1612)';
      // Unique pattern id per row to avoid SVG <defs> collisions
      const patId = 'sg-h-' + idx + '-' + Math.floor(Math.random() * 9999);
      return '<div class="sg-row" data-i="' + idx + '" style="display:grid;'
        + 'grid-template-columns:1fr 160px 80px 60px 18px;gap:14px;align-items:center;'
        + 'padding:10px 8px;cursor:pointer;'
        + 'border-bottom:1px solid var(--paper-edge,#c7bca9);'
        + 'transition:background .15s cubic-bezier(0.22, 1, 0.36, 1)">'
        + '<span style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;'
        +   'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink);font-weight:500">' + escHtml(it.label) + '</span>'
        + '<svg class="sg-spark" viewBox="0 0 ' + sparkW + ' ' + sparkH + '" '
        +   'width="' + sparkW + '" height="' + sparkH + '" preserveAspectRatio="none" style="color:' + tint + '">'
        + '<defs><pattern id="' + patId + '" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(45)">'
        +   '<line x1="0" y1="0" x2="0" y2="4" stroke="currentColor" stroke-width="0.6" opacity="0.4"/>'
        + '</pattern></defs>'
        + (path
            ? '<path d="' + area + '" fill="url(#' + patId + ')" stroke="none"/>'
              + '<path id="sg-path-' + idx + '" d="' + path + '" fill="none" stroke="' + tint + '" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
              // 9.N.5.f · pulsing tracer that travels along the line.
              // Duration scales INVERSELY with endpoint count — busier
              // endpoints pulse faster. SMIL animateMotion is GPU-cheap.
              + '<circle r="2.2" fill="' + tint + '" opacity="0.85">'
              +   '<animateMotion dur="' + Math.max(1.2, Math.min(6, 6 - Math.log10(Math.max(1, it.count)))).toFixed(1) + 's" repeatCount="indefinite">'
              +     '<mpath href="#sg-path-' + idx + '"/>'
              +   '</animateMotion>'
              + '</circle>'
              + (endDot ? '<circle cx="' + endDot.x.toFixed(1) + '" cy="' + endDot.y.toFixed(1) + '" r="3" fill="var(--paper,#fbf9f3)"/>'
                        + '<circle cx="' + endDot.x.toFixed(1) + '" cy="' + endDot.y.toFixed(1) + '" r="2" fill="' + tint + '"/>' : '')
            : '<text x="' + (sparkW/2) + '" y="' + (sparkH/2 + 3) + '" text-anchor="middle" '
              + 'font-family="\'JetBrains Mono\',monospace" font-size="9" font-style="italic" '
              + 'fill="var(--ink-soft,#6b6453)" opacity="0.6">·</text>')
        + '</svg>'
        + '<b style="font-family:\'JetBrains Mono\',monospace;font-variant-numeric:tabular-nums slashed-zero;'
        +   'text-align:right;font-size:13px;color:var(--ink)">' + fmtNum(it.count) + '</b>'
        + '<span style="color:var(--ink-soft);font-family:\'JetBrains Mono\',monospace;'
        +   'text-align:right;font-size:11.5px">' + (pct ? pct + '%' : '') + '</span>'
        + '<span class="sg-arrow" style="font-family:\'JetBrains Mono\',monospace;color:var(--ink-soft);'
        +   'text-align:right;font-size:13px;opacity:0;transform:translateX(-6px);'
        +   'transition:opacity .15s cubic-bezier(0.22, 1, 0.36, 1), transform .15s cubic-bezier(0.22, 1, 0.36, 1)">→</span>'
        + '</div>';
    }).join('');
    host.innerHTML = '<div class="chart-sparkgrid">' + rows + '</div>';
    host.querySelectorAll('.sg-row').forEach((row, i) => {
      const it = items[i];
      const arrow = row.querySelector('.sg-arrow');
      row.addEventListener('mouseenter', () => {
        row.style.background = 'var(--paper-deep, #e9e2d1)';
        if (arrow) { arrow.style.opacity = '1'; arrow.style.transform = 'translateX(0)'; arrow.style.color = 'var(--ink, #1a1612)'; }
      });
      row.addEventListener('mouseleave', () => {
        row.style.background = '';
        if (arrow) { arrow.style.opacity = '0'; arrow.style.transform = 'translateX(-6px)'; }
      });
      row.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(row, e.clientX, e.clientY);
        if (opts.onClickItem) opts.onClickItem(it);
      });
    });
    return { type: 'sparkGrid', empty: false };
  };

  // ─── 8. treemap — squarified treemap with paper-ink tonal fills ─────────
  /**
   * items: [{ label, value, color_token? }, ...]
   * opts:  { height: 260, onClickTile: (item) => void }
   */
  APIN.charts.treemap = function treemap(host, items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) {
      host.innerHTML = '<div class="chart-empty" style="padding:24px;color:var(--ink-soft,#6b6453);font-style:italic">no data to map</div>';
      return { type: 'treemap', empty: true };
    }
    const W = host.clientWidth || 600, H = opts.height || 260;
    const sorted = [...items].sort((a, b) => (Number(b.value) || 0) - (Number(a.value) || 0));
    const total = sorted.reduce((a, it) => a + (Number(it.value) || 0), 0);
    if (total === 0) {
      host.innerHTML = '<div class="chart-empty" style="padding:24px;color:var(--ink-soft,#6b6453);font-style:italic">no volume in this window</div>';
      return { type: 'treemap', empty: true };
    }
    // 9.N.5 fix · slice-and-dice layout. Matches the ASCII mockup: the
    // dominant tile takes the LEFT half (proportional width), then the
    // right column is sliced HORIZONTALLY for the next group of items,
    // then those columns get sliced again as needed. Simpler than
    // squarified and visually closer to a print-shop layout.
    const tiles = [];
    function layoutSliceDice(area, list, depth) {
      if (list.length === 0) return;
      if (list.length === 1) {
        tiles.push({ x: area.x, y: area.y, w: area.w, h: area.h, it: list[0] });
        return;
      }
      const totVal = list.reduce((a, it) => a + (Number(it.value) || 0), 0);
      const dominant = list[0];
      const split = (Number(dominant.value) || 0) / totVal;
      // alternate split direction by depth: even depth → horizontal split,
      // odd depth → vertical split.
      const horizontal = (depth % 2) === 0;
      if (horizontal) {
        const w1 = area.w * split;
        tiles.push({ x: area.x, y: area.y, w: w1, h: area.h, it: dominant });
        layoutSliceDice(
          { x: area.x + w1, y: area.y, w: area.w - w1, h: area.h },
          list.slice(1), depth + 1
        );
      } else {
        const h1 = area.h * split;
        tiles.push({ x: area.x, y: area.y, w: area.w, h: h1, it: dominant });
        layoutSliceDice(
          { x: area.x, y: area.y + h1, w: area.w, h: area.h - h1 },
          list.slice(1), depth + 1
        );
      }
    }
    layoutSliceDice({ x: 0, y: 0, w: W, h: H }, sorted, 0);

    // 9.N.5 fix · color-code tiles by health (error rate when available)
    function tileColor(it) {
      const er = Number(it.error_rate);
      if (Number.isFinite(er)) {
        if (er > 0.05) return { stroke: 'var(--c-danger,#b01820)', fillOp: 0.32 };
        if (er > 0)    return { stroke: 'var(--c-amber,#d49620)',  fillOp: 0.32 };
        return                   { stroke: 'var(--c-ok,#2f6f3e)',     fillOp: 0.28 };
      }
      // Fallback when caller doesn't supply error_rate: use semantic series
      return { stroke: 'var(--ink,#1a1612)', fillOp: 0.24 };
    }

    // 9.N.5 fix · text MUST be clipped to its tile bounds — earlier render
    // had labels bleeding into neighbouring tiles. Each tile gets a unique
    // <clipPath> and its text group is bound by that clip. Labels are only
    // RENDERED if there's safely room for them (stricter thresholds).
    let svg = `<svg class="chart-svg chart-treemap" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="${W}" height="${H}">`;
    let clipDefs = '<defs>';
    tiles.forEach((tl, i) => {
      clipDefs += `<clipPath id="tm-clip-${i}"><rect x="${tl.x + 4}" y="${tl.y + 4}" width="${Math.max(0, tl.w - 8)}" height="${Math.max(0, tl.h - 8)}"/></clipPath>`;
    });
    clipDefs += '</defs>';
    svg += clipDefs;
    svg += paperInkDefs([colorVarFor('ink')]);
    tiles.forEach((tl, i) => {
      const tc = tileColor(tl.it);
      svg += `<g class="tm-tile" data-i="${i}" data-label="${escHtml(tl.it.label)}" data-value="${tl.it.value}" style="cursor:pointer">`;
      // Colored wash + bold stroke (gives the "color-coded" feel)
      svg += `<rect x="${tl.x + 1}" y="${tl.y + 1}" width="${Math.max(0, tl.w - 2)}" height="${Math.max(0, tl.h - 2)}" `
        + `fill="${tc.stroke}" fill-opacity="${tc.fillOp}" stroke="${tc.stroke}" stroke-width="1.4"/>`;
      // Inner ink contour for crispness
      svg += `<rect x="${tl.x + 3}" y="${tl.y + 3}" width="${Math.max(0, tl.w - 6)}" height="${Math.max(0, tl.h - 6)}" `
        + `fill="none" stroke="${tc.stroke}" stroke-width="0.5" stroke-opacity="0.55" stroke-dasharray="2 4"/>`;
      // Labels are wrapped in a <g> bound by the tile's clipPath so they CANNOT
      // bleed past tile bounds. Stricter rendering thresholds — only show full
      // label when tile is comfortably large, only count for medium tiles,
      // nothing for tiny tiles.
      const fullLabelMin = { w: 90, h: 50 };
      const mediumMin    = { w: 60, h: 30 };
      const tinyMin      = { w: 30, h: 16 };
      svg += `<g clip-path="url(#tm-clip-${i})">`;
      if (tl.w >= fullLabelMin.w && tl.h >= fullLabelMin.h) {
        svg += `<text x="${tl.x + 10}" y="${tl.y + 22}" font-family="'JetBrains Mono',monospace" font-size="12.5" fill="var(--ink,#1a1612)" font-weight="500">${escHtml(tl.it.label)}</text>`;
        const pctTxt = total > 0 ? ' · ' + ((tl.it.value / total) * 100).toFixed(1) + '%' : '';
        svg += `<text x="${tl.x + 10}" y="${tl.y + 38}" font-family="Fraunces,serif" font-style="italic" font-size="11" fill="var(--ink-soft,#6b6453)" style="font-variant-numeric:tabular-nums">${fmtNum(tl.it.value)}${pctTxt}</text>`;
        if (Number.isFinite(Number(tl.it.error_rate)) && tl.h > 70) {
          const er = Number(tl.it.error_rate) * 100;
          const erTxt = er > 0 ? er.toFixed(1) + '% err' : 'all 2xx';
          svg += `<text x="${tl.x + 10}" y="${tl.y + 54}" font-family="Fraunces,serif" font-style="italic" font-size="10" fill="${tc.stroke}">${erTxt}</text>`;
        }
      } else if (tl.w >= mediumMin.w && tl.h >= mediumMin.h) {
        // Medium tile: label only (truncated) + count below
        const labelTruncMax = Math.floor(tl.w / 7);
        const labelTxt = (tl.it.label || '').length > labelTruncMax
          ? (tl.it.label || '').slice(0, labelTruncMax - 1) + '…' : (tl.it.label || '');
        svg += `<text x="${tl.x + 6}" y="${tl.y + 16}" font-family="'JetBrains Mono',monospace" font-size="10.5" fill="var(--ink,#1a1612)" font-weight="500">${escHtml(labelTxt)}</text>`;
        svg += `<text x="${tl.x + 6}" y="${tl.y + 28}" font-family="'JetBrains Mono',monospace" font-size="9.5" fill="var(--ink-soft,#6b6453)" style="font-variant-numeric:tabular-nums">${fmtNum(tl.it.value)}</text>`;
      } else if (tl.w >= tinyMin.w && tl.h >= tinyMin.h) {
        // Tiny tile: just the count, no label
        svg += `<text x="${tl.x + 4}" y="${tl.y + 12}" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--ink,#1a1612)" style="font-variant-numeric:tabular-nums">${fmtNum(tl.it.value)}</text>`;
      }
      // Tiles smaller than tinyMin → no label at all (tooltip on hover instead)
      svg += `</g>`;
      svg += `</g>`;
    });
    svg += '</svg>';
    host.innerHTML = svg;
    host.querySelectorAll('.tm-tile').forEach((g) => {
      const label = g.getAttribute('data-label');
      const value = Number(g.getAttribute('data-value'));
      g.addEventListener('mouseenter', e => {
        const rect = g.querySelector('rect');
        if (rect) {
          rect.style.transition = 'transform .14s cubic-bezier(0.22, 1, 0.36, 1)';
          g.style.filter = 'drop-shadow(1px 1px 0 var(--ink-soft, #6b6453))';
        }
        showTip(e.clientX, e.clientY,
          '<b style="font-family:Fraunces,serif">' + escHtml(label) + '</b><br>'
          + '<span style="font-family:\'JetBrains Mono\',monospace">' + fmtNum(value) + ' requests · '
          + ((value/total)*100).toFixed(1) + '%</span>');
      });
      g.addEventListener('mousemove', e => { if (_tip) showTip(e.clientX, e.clientY, _tip.innerHTML); });
      g.addEventListener('mouseleave', () => { g.style.filter = ''; hideTip(); });
      g.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(host, e.clientX, e.clientY);
        if (opts.onClickTile) opts.onClickTile({ label, value });
      });
    });
    return { type: 'treemap', empty: false };
  };

  // ─── 9. boxplot — per-row horizontal box+whisker on shared x-axis ───────
  /**
   * items: [{ label, p10, p25, p50, p75, p90, max, outliers: [v, v, ...] }, ...]
   * opts:  { height: auto, xUnit: 'ms', onClickRow: (item) => void }
   */
  APIN.charts.boxplot = function boxplot(host, items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) {
      host.innerHTML = '<div class="chart-empty" style="padding:24px;color:var(--ink-soft,#6b6453);font-style:italic">no rows to plot</div>';
      return { type: 'boxplot', empty: true };
    }
    // 9.N.5 fix · more vertical breathing room per row, taller boxes
    const W = host.clientWidth || 720;
    const rowH = 42;
    const padL = 180, padR = 32, padT = 26, padB = 36;
    const H = padT + padB + items.length * rowH;
    // x-domain
    let xMax = 1;
    items.forEach(it => {
      ['p10','p25','p50','p75','p90','max'].forEach(k => {
        if (Number(it[k]) > xMax) xMax = Number(it[k]);
      });
      (it.outliers || []).forEach(v => { if (Number(v) > xMax) xMax = Number(v); });
    });
    const _nice = niceScale(xMax, 5);
    xMax = _nice.max;
    const xTicks = _nice.ticks;
    const iw = W - padL - padR;
    const x = v => padL + (Math.max(0, Math.min(xMax, v)) / xMax) * iw;
    let svg = `<svg class="chart-svg chart-boxplot" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="${W}" height="${H}">`;
    svg += paperInkDefs([colorVarFor('ink')]);
    // x-axis ticks
    xTicks.forEach(tv => {
      const tx = x(tv);
      svg += `<line x1="${tx}" y1="${padT - 4}" x2="${tx}" y2="${H - padB + 4}" stroke="var(--paper-edge,#c7bca9)" stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>`;
      svg += `<text x="${tx}" y="${H - padB + 18}" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="10.5" fill="var(--ink-soft,#6b6453)" style="font-variant-numeric:tabular-nums">${fmtNum(tv)}${opts.xUnit ? '' : ''}</text>`;
    });
    if (opts.xUnit) {
      svg += `<text x="${padL + iw}" y="${H - padB + 18}" text-anchor="end" font-family="Fraunces,serif" font-style="italic" font-size="10" fill="var(--ink-soft,#6b6453)">${escHtml(opts.xUnit)}</text>`;
    }
    // Rows
    items.forEach((it, i) => {
      const rowY = padT + i * rowH;
      const cy = rowY + rowH / 2;
      // Label
      svg += `<text x="${padL - 12}" y="${cy + 4}" text-anchor="end" font-family="'JetBrains Mono',monospace" font-size="11.5" fill="var(--ink,#1a1612)">${escHtml(it.label)}</text>`;
      svg += `<g class="bp-row" data-i="${i}" style="cursor:pointer">`;
      // Whisker (p10 → p90) — bolder ink line
      const p10 = Number(it.p10) || 0, p25 = Number(it.p25) || 0;
      const p50 = Number(it.p50) || 0, p75 = Number(it.p75) || 0, p90 = Number(it.p90) || 0;
      svg += `<line x1="${x(p10)}" y1="${cy}" x2="${x(p90)}" y2="${cy}" stroke="var(--ink,#1a1612)" stroke-width="1.4" opacity="0.85"/>`;
      // 9.N.5 fix · Whisker caps as serif ticks (taller, 1.6px stroke)
      svg += `<line x1="${x(p10)}" y1="${cy - 7}" x2="${x(p10)}" y2="${cy + 7}" stroke="var(--ink,#1a1612)" stroke-width="1.6" stroke-linecap="round"/>`;
      svg += `<line x1="${x(p90)}" y1="${cy - 7}" x2="${x(p90)}" y2="${cy + 7}" stroke="var(--ink,#1a1612)" stroke-width="1.6" stroke-linecap="round"/>`;
      // Box (p25 → p75) — taller (18px), more opaque fill
      const bx = x(p25), bxR = x(p75);
      const bw = Math.max(2, bxR - bx);
      svg += `<rect x="${bx}" y="${cy - 11}" width="${bw}" height="22" fill="var(--ink,#1a1612)" fill-opacity="0.18" stroke="var(--ink,#1a1612)" stroke-width="1.4"/>`;
      // Median line (p50) — extends past box like a printer's stamp
      svg += `<line x1="${x(p50)}" y1="${cy - 13}" x2="${x(p50)}" y2="${cy + 13}" stroke="var(--ink,#1a1612)" stroke-width="2.6"/>`;
      // p50 numeric callout under the median
      svg += `<text x="${x(p50)}" y="${cy + 26}" text-anchor="middle" font-family="Fraunces,serif" font-style="italic" font-size="10" fill="var(--ink-soft,#6b6453)" style="font-variant-numeric:tabular-nums">${fmtNum(p50)}</text>`;
      // Outliers — open circles
      (it.outliers || []).forEach(v => {
        const ox = x(Number(v) || 0);
        svg += `<circle cx="${ox}" cy="${cy}" r="3" fill="var(--paper,#fbf9f3)" stroke="var(--ink,#1a1612)" stroke-width="1.2"/>`;
      });
      // Row click target (transparent rect spanning the chart band)
      svg += `<rect x="${padL}" y="${rowY}" width="${iw}" height="${rowH}" fill="transparent" data-row-click/>`;
      svg += `</g>`;
    });
    svg += '</svg>';
    host.innerHTML = svg;
    host.querySelectorAll('.bp-row').forEach((g, i) => {
      const it = items[i];
      g.addEventListener('mouseenter', e => {
        const box = g.querySelector('rect:not([data-row-click])');
        if (box) box.setAttribute('fill-opacity', '0.22');
        showTip(e.clientX, e.clientY,
          '<b style="font-family:Fraunces,serif">' + escHtml(it.label) + '</b><br>'
          + '<span style="font-family:\'JetBrains Mono\',monospace">'
          + 'p50 ' + fmtNum(it.p50) + ' · p95 ' + fmtNum(it.p90)
          + (it.max ? ' · max ' + fmtNum(it.max) : '') + '</span>');
      });
      g.addEventListener('mousemove', e => { if (_tip) showTip(e.clientX, e.clientY, _tip.innerHTML); });
      g.addEventListener('mouseleave', () => {
        const box = g.querySelector('rect:not([data-row-click])');
        if (box) box.setAttribute('fill-opacity', '0.12');
        hideTip();
      });
      g.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(host, e.clientX, e.clientY);
        if (opts.onClickRow) opts.onClickRow(it);
      });
    });
    return { type: 'boxplot', empty: false };
  };

  // ─── 10. bullet — KPI bullet chart with target + tonal bands ─────────────
  /**
   * data: { value: 1537, target: 1500, ranges: [{ hi: 800, tone: 'great' }, { hi: 1500, tone: 'target' }, { hi: 2000, tone: 'poor' }],
   *         unit: 'ms', label: '' }
   * opts: {}
   */
  APIN.charts.bullet = function bullet(host, data, opts) {
    opts = opts || {};
    if (!data || data.value == null) {
      host.innerHTML = '<div class="chart-empty" style="padding:24px;color:var(--ink-soft,#6b6453);font-style:italic">no value</div>';
      return { type: 'bullet', empty: true };
    }
    const value = Number(data.value) || 0;
    const target = data.target != null ? Number(data.target) : null;
    const ranges = data.ranges || [];
    const xMax = Math.max(value, target || 0, ...(ranges.map(r => Number(r.hi) || 0))) || 1;
    // 9.N.5 fix · taller bands, more breathing room
    const W = host.clientWidth || 460;
    const H = 116;
    const padL = 18, padR = 18, padT = 34, padB = 30;
    const iw = W - padL - padR;
    const ih = H - padT - padB;
    const x = v => padL + Math.min(1, Math.max(0, (Number(v) || 0) / xMax)) * iw;
    // Tonal-band stipple densities: great=light, target=medium, poor=heavy
    const stippleFor = tone => tone === 'great' ? 'url(#pi-stipple-1)' : tone === 'target' ? 'url(#pi-stipple-2)' : 'url(#pi-stipple-3)';
    const labelFor = tone => tone === 'great' ? 'great' : tone === 'target' ? 'target' : 'poor';
    let svg = `<svg class="chart-svg chart-bullet" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="${W}" height="${H}">`;
    svg += paperInkDefs([colorVarFor('ink')]);
    // Bands
    let lastX = padL;
    ranges.forEach(r => {
      const rx = x(r.hi);
      const colorByTone = r.tone === 'great' ? 'var(--c-ok,#2f6f3e)' : r.tone === 'target' ? 'var(--c-amber,#d49620)' : 'var(--c-danger,#b01820)';
      svg += `<rect x="${lastX}" y="${padT}" width="${rx - lastX}" height="${ih}" fill="${colorByTone}" fill-opacity="0.08" stroke="var(--paper-edge,#c7bca9)" stroke-width="0.5"/>`;
      svg += `<rect x="${lastX}" y="${padT}" width="${rx - lastX}" height="${ih}" style="color:${colorByTone}" fill="${stippleFor(r.tone)}"/>`;
      // Band label
      svg += `<text x="${(lastX + rx) / 2}" y="${padT - 8}" text-anchor="middle" font-family="Fraunces,serif" font-style="italic" font-size="10.5" fill="var(--ink-soft,#6b6453)">${labelFor(r.tone)}</text>`;
      lastX = rx;
    });
    // Target marker (vertical line with serif "T" callout)
    if (target != null) {
      const tx = x(target);
      svg += `<line x1="${tx}" y1="${padT - 4}" x2="${tx}" y2="${padT + ih + 4}" stroke="var(--ink,#1a1612)" stroke-width="1" stroke-dasharray="3 3" opacity="0.8"/>`;
      svg += `<text x="${tx}" y="${padT + ih + 18}" text-anchor="middle" font-family="Fraunces,serif" font-style="italic" font-size="10" fill="var(--ink-soft,#6b6453)">target ${fmtNum(target)}${data.unit || ''}</text>`;
    }
    // 9.N.5 fix · Stronger value marker:
    //  - the filled "track" bar runs from 0 to value (6px tall, ink-deep, 78% opacity)
    //  - the triangle marker is larger (8px wide), points DOWN at the value
    //  - vertical stroke through the band marks the exact x-position
    const vx = x(value);
    // Filled track bar
    svg += `<rect class="bul-bar" x="${padL}" y="${padT + ih/2 - 4}" width="${vx - padL}" height="8" fill="var(--ink,#1a1612)" fill-opacity="0.82" stroke="var(--ink,#1a1612)" stroke-width="0.5"/>`;
    // Vertical pin through the full band height
    svg += `<line x1="${vx}" y1="${padT}" x2="${vx}" y2="${padT + ih}" stroke="var(--ink,#1a1612)" stroke-width="2.4"/>`;
    // Triangle pointing DOWN at the value
    svg += `<polygon points="${vx-7},${padT-6} ${vx+7},${padT-6} ${vx},${padT+3}" fill="var(--ink,#1a1612)" stroke="var(--paper,#fbf9f3)" stroke-width="0.8"/>`;
    // Value caption — Fraunces serif with mono numeric
    const pctOfTarget = target ? Math.round((value / target) * 100) : null;
    const pctTone = pctOfTarget == null ? 'var(--ink,#1a1612)'
                  : pctOfTarget > 100 ? 'var(--c-danger,#b01820)'
                  : pctOfTarget > 75  ? 'var(--c-amber,#d49620)'
                                       : 'var(--c-ok,#2f6f3e)';
    svg += `<text x="${vx}" y="${padT + ih + 20}" text-anchor="middle" font-family="Fraunces,serif" font-style="italic" font-size="13" fill="var(--ink,#1a1612)" style="font-variant-numeric:tabular-nums">${fmtNum(value)}${data.unit || ''}</text>`;
    if (pctOfTarget != null) {
      svg += `<text x="${vx}" y="${padT + ih + 34}" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="10" fill="${pctTone}">${pctOfTarget}% of target</text>`;
    }
    svg += '</svg>';
    host.innerHTML = svg;
    // Entry animation: bar grows from width 0
    const bar = host.querySelector('.bul-bar');
    if (bar) {
      bar.animate([
        { width: 0 }, { width: (vx - padL) }
      ], { duration: 600, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'forwards' });
    }
    return { type: 'bullet', empty: false };
  };

  // ─── 11. quadrant — endpoint health quadrant chart ───────────────────────
  /**
   * items: [{ label, x_val, y_val, size, color_token }, ...]
   *   x_val: typically volume (count)
   *   y_val: typically error_rate (0..1)
   *   size:  typically p95 (px will be sqrt(size) * factor)
   * opts: { xLabel: 'volume', yLabel: 'error %', xMid: 'median', yMid: 0.05,
   *         onClickPoint: (item) => void }
   */
  APIN.charts.quadrant = function quadrant(host, items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) {
      host.innerHTML = '<div class="chart-empty" style="padding:24px;color:var(--ink-soft,#6b6453);font-style:italic">no endpoints to plot</div>';
      return { type: 'quadrant', empty: true };
    }
    const W = host.clientWidth || 720, H = 400;
    const padL = 70, padR = 30, padT = 42, padB = 52;
    const iw = W - padL - padR, ih = H - padT - padB;
    // 9.N.5 fix · LOG-scale on volume axis (sparse + dominant data overlaps
    // catastrophically on linear). Y axis: linear error-rate (0..yMax).
    const xVals = items.map(it => Math.max(1, Number(it.x_val) || 1));
    const xMaxRaw = Math.max(2, ...xVals);
    const logBase = Math.log10(xMaxRaw + 1);
    const xLog = v => Math.log10(Math.max(1, Number(v) || 1) + 1) / logBase;
    const yMaxRaw = Math.max(0.001, ...items.map(it => Number(it.y_val) || 0));
    // Snap yMax up to a nice value above the max data point
    const yMax = yMaxRaw < 0.01 ? 0.02
               : yMaxRaw < 0.05 ? 0.10
               : yMaxRaw < 0.20 ? 0.30
               : yMaxRaw < 0.50 ? 0.60 : 1.00;
    const sizeMax = Math.max(1, ...items.map(it => Number(it.size) || 0));
    const x = v => padL + xLog(v) * iw;
    const y = v => padT + ih - (Math.min(yMax, Number(v) || 0) / yMax) * ih;
    // Mid lines for quadrant separation
    const xMid = opts.xMid === 'median' ? (() => {
      const sorted = [...items].map(it => Number(it.x_val) || 0).sort((a,b) => a - b);
      return sorted[Math.floor(sorted.length / 2)] || 10;
    })() : (Number(opts.xMid) || Math.pow(10, logBase * 0.5));
    const yMid = opts.yMid != null ? Number(opts.yMid) : yMax / 4;
    let svg = `<svg class="chart-svg chart-quadrant" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="${W}" height="${H}">`;
    svg += paperInkDefs([colorVarFor('ink')]);
    // Axes (1px hand-drawn, wobble filter)
    svg += `<line x1="${padL}" y1="${padT + ih}" x2="${padL + iw}" y2="${padT + ih}" stroke="var(--ink-soft,#6b6453)" stroke-width="1" filter="url(#pi-wobble)"/>`;
    svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + ih}" stroke="var(--ink-soft,#6b6453)" stroke-width="1" filter="url(#pi-wobble)"/>`;
    // Quadrant midlines (dashed)
    svg += `<line x1="${x(xMid)}" y1="${padT}" x2="${x(xMid)}" y2="${padT + ih}" stroke="var(--ink-soft,#6b6453)" stroke-width="0.5" stroke-dasharray="3 3" opacity="0.6"/>`;
    svg += `<line x1="${padL}" y1="${y(yMid)}" x2="${padL + iw}" y2="${y(yMid)}" stroke="var(--ink-soft,#6b6453)" stroke-width="0.5" stroke-dasharray="3 3" opacity="0.6"/>`;
    // Quadrant captions
    svg += `<text x="${padL + iw - 8}" y="${padT + 14}" text-anchor="end" font-family="Fraunces,serif" font-style="italic" font-size="10" letter-spacing="0.08em" fill="var(--ink-soft,#6b6453)" text-transform="uppercase">needs attention</text>`;
    svg += `<text x="${padL + 8}" y="${padT + 14}" font-family="Fraunces,serif" font-style="italic" font-size="10" letter-spacing="0.08em" fill="var(--ink-soft,#6b6453)">noisy but cheap</text>`;
    svg += `<text x="${padL + iw - 8}" y="${padT + ih - 8}" text-anchor="end" font-family="Fraunces,serif" font-style="italic" font-size="10" letter-spacing="0.08em" fill="var(--ink-soft,#6b6453)">healthy heavy-hitters</text>`;
    svg += `<text x="${padL + 8}" y="${padT + ih - 8}" font-family="Fraunces,serif" font-style="italic" font-size="10" letter-spacing="0.08em" fill="var(--ink-soft,#6b6453)">background</text>`;
    // 9.N.5 fix · Axis tick labels (log x, linear y as %)
    // X ticks at 1, 10, 100, 1000, ...
    const xDecades = [];
    for (let dec = 0; dec <= Math.ceil(logBase); dec++) {
      const v = Math.pow(10, dec);
      if (v <= xMaxRaw * 1.2) xDecades.push(v);
    }
    xDecades.forEach(tv => {
      const tx = x(tv);
      svg += `<line x1="${tx}" y1="${padT + ih}" x2="${tx}" y2="${padT + ih + 4}" stroke="var(--ink-soft,#6b6453)" stroke-width="0.6"/>`;
      svg += `<text x="${tx}" y="${padT + ih + 16}" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="10" fill="var(--ink-soft,#6b6453)" style="font-variant-numeric:tabular-nums">${fmtNum(tv)}</text>`;
    });
    // Y ticks at 0, yMid, yMax (as percentages if yPct)
    [0, yMid, yMax].forEach(tv => {
      const ty = y(tv);
      svg += `<line x1="${padL - 4}" y1="${ty}" x2="${padL}" y2="${ty}" stroke="var(--ink-soft,#6b6453)" stroke-width="0.6"/>`;
      const lbl = opts.yPct ? (tv * 100).toFixed(tv < 0.01 ? 1 : 0) + '%' : tv.toFixed(2);
      svg += `<text x="${padL - 8}" y="${ty + 3}" text-anchor="end" font-family="'JetBrains Mono',monospace" font-size="10" fill="var(--ink-soft,#6b6453)">${lbl}</text>`;
    });
    // Axis titles
    svg += `<text x="${padL + iw / 2}" y="${H - 8}" text-anchor="middle" font-family="Fraunces,serif" font-style="italic" font-size="11" fill="var(--ink-soft,#6b6453)">${escHtml(opts.xLabel || 'volume')} →  (log scale)</text>`;
    svg += `<text x="${padL - 54}" y="${padT + ih / 2}" text-anchor="middle" transform="rotate(-90 ${padL - 54} ${padT + ih / 2})" font-family="Fraunces,serif" font-style="italic" font-size="11" fill="var(--ink-soft,#6b6453)">${escHtml(opts.yLabel || 'error rate')} →</text>`;
    // 9.N.5 fix · Stronger label dodging.
    //  - prioritize: label the points with highest x_val (volume) first;
    //    they're the most important and get the best slot.
    //  - if NO candidate position is collision-free, drop the label entirely.
    //    A jammed label is worse than no label.
    //  - tooltip on hover still shows the label, so users can recover.
    const placedLabels = [];   // [{ lx, ly, w, h }]
    const placedBubbles = [];  // [{ cx, cy, r }] to avoid label-over-bubble overlap
    const labelSize = { fz: 10.5, charPx: 6.2, h: 14 };
    function rectsOverlap(a, b) {
      return !(a.lx + a.w < b.lx || b.lx + b.w < a.lx
            || a.ly + a.h < b.ly || b.ly + b.h < a.ly);
    }
    function rectCollidesBubble(rect, bub) {
      const closestX = Math.max(bub.cx - bub.r, Math.min(rect.lx + rect.w / 2, bub.cx + bub.r));
      const closestY = Math.max(bub.cy - bub.r, Math.min(rect.ly + rect.h / 2, bub.cy + bub.r));
      const dx = closestX - bub.cx, dy = closestY - bub.cy;
      return (dx * dx + dy * dy) < (bub.r * bub.r * 0.8);
    }
    function fits(box) {
      // Check inside chart bounds
      if (box.lx < padL + 2 || box.lx + box.w > padL + iw - 2) return false;
      if (box.ly < padT + 2 || box.ly + box.h > padT + ih - 2) return false;
      for (const p of placedLabels) if (rectsOverlap(box, p)) return false;
      for (const b of placedBubbles) if (rectCollidesBubble(box, b)) return false;
      return true;
    }

    // Sort by importance: high volume + high error rate first
    const ordered = items.map((it, i) => ({ it, i,
      importance: (Number(it.x_val) || 0) * (1 + (Number(it.y_val) || 0) * 10)
    })).sort((a, b) => b.importance - a.importance);

    // First pass: place ALL bubbles (without labels) so they're known
    items.forEach((it, i) => {
      const cx = x(Number(it.x_val) || 0);
      const cy = y(Number(it.y_val) || 0);
      const sizeVal = Number(it.size) || 0;
      const r = Math.max(5, Math.min(26, 5 + Math.sqrt(sizeVal / sizeMax) * 18));
      placedBubbles.push({ cx, cy, r });
    });

    // Second pass: render bubbles in original order; render labels in
    // importance order, dropping ones that can't fit.
    const labelToRender = {};
    ordered.forEach(({ it, i }) => {
      const cx = x(Number(it.x_val) || 0);
      const cy = y(Number(it.y_val) || 0);
      const r = placedBubbles[i].r;
      const labelText = (it.label || '').length > 20
        ? (it.label || '').slice(0, 18) + '…' : (it.label || '');
      const lw = labelText.length * labelSize.charPx;
      const candidates = [
        { lx: cx + r + 6, ly: cy + 3, anchor: 'start' },
        { lx: cx - r - 6, ly: cy + 3, anchor: 'end'   },
        { lx: cx,         ly: cy - r - 6, anchor: 'middle' },
        { lx: cx,         ly: cy + r + 14, anchor: 'middle' },
      ];
      let chosen = null;
      for (const c of candidates) {
        const boxX = c.anchor === 'middle' ? c.lx - lw/2 : (c.anchor === 'end' ? c.lx - lw : c.lx);
        const box = { lx: boxX, ly: c.ly - labelSize.h + 2, w: lw, h: labelSize.h };
        if (fits(box)) {
          chosen = { ...c, box };
          break;
        }
      }
      if (chosen) {
        placedLabels.push(chosen.box);
        labelToRender[i] = { chosen, text: labelText };
      }
      // else: no label rendered for this point — tooltip on hover
    });

    // Now emit SVG for all points (bubble always, label if assigned)
    items.forEach((it, i) => {
      const cx = x(Number(it.x_val) || 0);
      const cy = y(Number(it.y_val) || 0);
      const r = placedBubbles[i].r;
      const color = colorVarFor(it.color_token || 'ink');
      svg += `<g class="qd-point" data-i="${i}" style="cursor:pointer">`;
      svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="1.6" opacity="0.85"/>`;
      svg += `<circle cx="${cx}" cy="${cy}" r="${Math.max(1.6, r * 0.22)}" fill="${color}"/>`;
      const lbl = labelToRender[i];
      if (lbl) {
        const ch = lbl.chosen;
        if (ch.anchor === 'start' || ch.anchor === 'end') {
          const lx0 = ch.anchor === 'start' ? cx + r : cx - r;
          const lx1 = ch.lx + (ch.anchor === 'start' ? -3 : 3);
          svg += `<line x1="${lx0}" y1="${cy}" x2="${lx1}" y2="${ch.ly - 4}" stroke="var(--ink-soft,#6b6453)" stroke-width="0.5" opacity="0.45"/>`;
        }
        svg += `<text x="${ch.lx}" y="${ch.ly}" text-anchor="${ch.anchor}" font-family="'JetBrains Mono',monospace" font-size="10.5" fill="var(--ink,#1a1612)">${escHtml(lbl.text)}</text>`;
      }
      svg += `</g>`;
    });
    svg += '</svg>';
    host.innerHTML = svg;
    host.querySelectorAll('.qd-point').forEach((g, i) => {
      const it = items[i];
      g.addEventListener('mouseenter', e => {
        const outer = g.querySelector('circle');
        if (outer) { outer.setAttribute('stroke-width', '2.4'); outer.setAttribute('opacity', '1'); }
        showTip(e.clientX, e.clientY,
          '<b style="font-family:Fraunces,serif">' + escHtml(it.label) + '</b><br>'
          + '<span style="font-family:\'JetBrains Mono\',monospace">'
          + (opts.xLabel || 'x') + ': ' + fmtNum(it.x_val) + ' · '
          + (opts.yLabel || 'y') + ': ' + (Number(it.y_val) * (opts.yPct ? 100 : 1)).toFixed(opts.yPct ? 1 : 2) + (opts.yPct ? '%' : '')
          + (it.size ? '<br>size: ' + fmtNum(it.size) : '') + '</span>');
      });
      g.addEventListener('mousemove', e => { if (_tip) showTip(e.clientX, e.clientY, _tip.innerHTML); });
      g.addEventListener('mouseleave', () => {
        const outer = g.querySelector('circle');
        if (outer) { outer.setAttribute('stroke-width', '1.6'); outer.setAttribute('opacity', '0.78'); }
        hideTip();
      });
      g.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(host, e.clientX, e.clientY);
        if (opts.onClickPoint) opts.onClickPoint(it);
      });
    });
    return { type: 'quadrant', empty: false };
  };

})();
