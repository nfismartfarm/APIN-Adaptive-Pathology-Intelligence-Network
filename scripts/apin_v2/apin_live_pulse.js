// apin_live_pulse.js — Phase 9.N.7 · live req/sec line chart (TradingView-style)
// + commentary engine.
//
// Two exports:
//   APIN.livePulse.attach(hostEl, opts) → controller
//   APIN.commentary.attach(hostEl, opts) → ticker (or list) renderer
//
// The pulse widget owns:
//   1. An in-memory data accumulator (1-second buckets, ring of 5 min)
//   2. A line chart renderer with smooth scroll-left animation via rAF
//   3. Toggle controls for window length (60s / 5m) + Y-axis metric
//   4. Hover tooltip on the line (snap to nearest second)
//   5. Click-to-inspect (opens request drawer for the click second)
//   6. Three-state pulse dot color (sage / amber / crimson)
//
// The commentary engine watches the data stream and emits humanized
// observations via a phrasing pool (~6 variants per trigger), with
// no-repeat-within-60s. Subscribers render the latest as a ticker or
// as a full list (expanded view).

(function () {
  "use strict";
  if (!window.APIN) window.APIN = {};

  // ─── Per-second bucket accumulator + filter-aware ───────────────────
  function _now() { return Math.floor(Date.now() / 1000); }
  function _pct(arr, p) {
    if (!arr || !arr.length) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    return sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1))))];
  }
  function _statusBucket(code) {
    if (code >= 200 && code < 300) return '2xx';
    if (code >= 300 && code < 400) return '3xx';
    if (code === 429) return '429';
    if (code >= 400 && code < 500) return '4xx';
    if (code >= 500) return '5xx';
    return 'other';
  }

  class LiveAccumulator {
    constructor() {
      // 9.N.7.f · memory caps. Previously ringSec=360 + events=600. Under a
      // sustained 5 req/s session that's still bounded but the per-bucket
      // .lat arrays grew unbounded. New caps are aggressive: 6-min ring
      // for the 5m view, hard 1000 raw-event cap, 200 latencies/bucket max.
      this.ringSec = 360;
      this.maxEvents = 1000;
      this.maxLatPerBucket = 200;
      this.buckets = new Map();
      this.events = [];
      this.subscribers = new Set();
    }
    add(ev) {
      const ts = _now();
      let b = this.buckets.get(ts);
      if (!b) {
        b = { count: 0, errors: 0, lat: [], bytes: 0, byStatus: {} };
        this.buckets.set(ts, b);
      }
      b.count++;
      if ((ev.status_code || 0) >= 400) b.errors++;
      if (ev.latency_ms != null && b.lat.length < this.maxLatPerBucket) {
        b.lat.push(Number(ev.latency_ms));
      }
      if (ev.bytes_out) b.bytes += Number(ev.bytes_out) || 0;
      const sb = _statusBucket(ev.status_code);
      b.byStatus[sb] = (b.byStatus[sb] || 0) + 1;
      // Store the raw event for hover lookups — hard cap
      this.events.push({ t: ts, ev });
      if (this.events.length > this.maxEvents) {
        this.events.splice(0, this.events.length - this.maxEvents);
      }
      // Only prune occasionally (every 50 adds) so a burst of 25 doesn't
      // walk the entire bucket map 25 times.
      this._adds = (this._adds || 0) + 1;
      if ((this._adds & 0x1F) === 0) this._prune();
      // Subscriber callbacks are sync — keep them tiny.
      this.subscribers.forEach(fn => { try { fn(ev, ts); } catch (e) {} });
    }
    _prune() {
      const cutoff = _now() - this.ringSec;
      for (const ts of this.buckets.keys()) if (ts < cutoff) this.buckets.delete(ts);
      while (this.events.length && this.events[0].t < cutoff) this.events.shift();
    }
    /** Returns per-second values: [{ t, v }, ...] across the requested window. */
    series(metric, windowSec) {
      const now = _now();
      const out = [];
      for (let t = now - windowSec; t <= now; t++) {
        const b = this.buckets.get(t);
        let v = 0;
        if (b) {
          switch (metric) {
            case 'rate':    v = b.count; break;
            case 'errors':  v = b.count > 0 ? (b.errors / b.count) * 100 : 0; break;
            case 'p95':     v = _pct(b.lat, 0.95); break;
            case 'p50':     v = _pct(b.lat, 0.50); break;
            case 'bytes':   v = b.bytes; break;
          }
        }
        out.push({ t, v });
      }
      return out;
    }
    /** Per-status breakdown series for multi-line mode. */
    seriesByStatus(windowSec) {
      const now = _now();
      const keys = ['2xx', '4xx', '5xx', '429'];
      const result = {};
      keys.forEach(k => result[k] = []);
      for (let t = now - windowSec; t <= now; t++) {
        const b = this.buckets.get(t);
        keys.forEach(k => result[k].push({ t, v: (b && b.byStatus[k]) || 0 }));
      }
      return result;
    }
    eventsAt(ts, windowSize = 1) {
      const lo = ts - windowSize, hi = ts + windowSize;
      return this.events.filter(e => e.t >= lo && e.t <= hi).map(e => e.ev);
    }
    lastEventAge() {
      if (this.events.length === 0) return null;
      return _now() - this.events[this.events.length - 1].t;
    }
    subscribe(fn) { this.subscribers.add(fn); return () => this.subscribers.delete(fn); }
    rollingStats(windowSec = 30) {
      const now = _now();
      let count = 0, errors = 0;
      const lats = [];
      for (let t = now - windowSec; t <= now; t++) {
        const b = this.buckets.get(t);
        if (!b) continue;
        count += b.count;
        errors += b.errors;
        b.lat.forEach(l => lats.push(l));
      }
      return {
        rate: count / windowSec,
        errorRate: count > 0 ? (errors / count) * 100 : 0,
        p50: _pct(lats, 0.5),
        p95: _pct(lats, 0.95),
        totalCount: count,
        totalErrors: errors,
      };
    }
  }

  // Singleton — shared between widget instances + commentary engine
  const _accum = new LiveAccumulator();

  // Expose ingest so the SSE client can publish into it
  window.APIN = window.APIN || {};
  window.APIN.livePulseData = {
    feed: ev => _accum.add(ev),
    accumulator: _accum,
  };

  // ═══════════════════════════════════════════════════════════════════
  // ─── Commentary engine — humanised phrasing pool + triggers ────────
  // ═══════════════════════════════════════════════════════════════════

  const PHRASES = {
    burst: [
      'Burst at {time} — {n} calls in {dur}s',
      'Sudden activity around {time}',
      'Things picked up at {time}',
      '{n} requests landed together at {time}',
      'A wave of {n} just hit',
      'Brief spike — {n} in {dur}s, then quiet again',
      'Burst of {n} requests at {time}',
      'Surge at {time} — {n} in {dur} seconds',
    ],
    slowdown: [
      '{endpoint} slowing — p95 hit {lat}',
      'Latency creeping up on {endpoint}',
      '{endpoint} taking its time — p95 over {lat} now',
      'Slowdown on {endpoint}',
      '{endpoint} sluggish — p95 climbed to {lat}',
      'Watch {endpoint} — p95 just touched {lat}',
      '{endpoint} getting heavy — {lat}',
    ],
    first_5xx: [
      'First server error in this window — {endpoint}',
      'Heads up: a 5xx from {endpoint} at {time}',
      'Something broke briefly on {endpoint}',
      '{endpoint} hiccuped — a 5xx at {time}',
      'Server side error on {endpoint}',
      '5xx just landed from {endpoint}',
    ],
    error_drift: [
      'Error rate ticked up — {old}% → {new}%',
      'Errors edging higher ({new}% now)',
      'More failures lately — now at {new}%',
      'Error rate climbed a bit ({old}→{new}%)',
      'Failure ratio up to {new}%',
      'Things looking flakier — {new}% errors',
    ],
    error_recovery: [
      'Errors dropped back — now at {new}%',
      'Failure rate cooled off',
      'Errors easing — {old}% → {new}%',
      'Things stabilising — {new}% errors',
      'Error rate back down',
    ],
    recovery: [
      'All quiet now',
      'Calmer waters',
      'Steady as she goes',
      'Things settled down',
      'Quiet again',
      'Back to a steady hum',
    ],
    healthy_streak: [
      'Streak of clean responses',
      'No errors in the last {dur}',
      'Healthy run going',
      'Clean window — {n} requests, zero errors',
      '{n} requests, all 2xx',
      'No drama for {dur} now',
    ],
    silence: [
      'No traffic for a while',
      '{dur} of silence',
      'Nothing coming through',
      'Quiet — no requests in {dur}',
    ],
    new_endpoint: [
      'New endpoint live: {endpoint}',
      'First requests to {endpoint}',
      '{endpoint} just came online',
      'Traffic on {endpoint} for the first time today',
    ],
  };

  // Round-robin "no repeat within 60s" pool selection
  const _recentPhrases = [];
  function _pick(category, vars) {
    const pool = PHRASES[category] || [];
    if (pool.length === 0) return '';
    const now = Date.now();
    // Drop expired entries
    while (_recentPhrases.length && now - _recentPhrases[0].at > 60_000) _recentPhrases.shift();
    const recentlyUsed = new Set(_recentPhrases.filter(p => p.category === category).map(p => p.idx));
    const candidates = pool.map((_, i) => i).filter(i => !recentlyUsed.has(i));
    const finalPool = candidates.length > 0 ? candidates : pool.map((_, i) => i);
    const idx = finalPool[Math.floor(Math.random() * finalPool.length)];
    _recentPhrases.push({ category, idx, at: now });
    let txt = pool[idx];
    for (const k in (vars || {})) txt = txt.replace(new RegExp('\\{' + k + '\\}', 'g'), vars[k]);
    return txt;
  }

  class CommentaryEngine {
    constructor(accum) {
      this.accum = accum;
      this.entries = [];   // { time, text, tone }
      this.subscribers = new Set();
      this.endpointsSeen = new Set();
      this.lastErrorRate = null;
      this.lastBurstAt = 0;
      this.lastSlowdown = {};   // endpoint -> last p95-alert ts
      this.tickInterval = setInterval(() => this._tick(), 5000);
      this._lastSilenceAt = 0;
      this._lastHealthyStreakAt = 0;
      accum.subscribe(ev => this._onEvent(ev));
    }
    push(text, tone) {
      tone = tone || 'info';
      const entry = { time: _now(), text, tone };
      this.entries.unshift(entry);
      if (this.entries.length > 50) this.entries.length = 50;
      this.subscribers.forEach(fn => { try { fn(entry); } catch (e) {} });
    }
    subscribe(fn) { this.subscribers.add(fn); return () => this.subscribers.delete(fn); }
    _fmtTime(ts) {
      const d = new Date(ts * 1000);
      return d.toTimeString().slice(0, 8);   // HH:MM:SS
    }
    _onEvent(ev) {
      const now = _now();
      // New endpoint detection
      if (ev.path && !this.endpointsSeen.has(ev.path)) {
        this.endpointsSeen.add(ev.path);
        // Don't fire on the very first event (boot noise)
        if (this.endpointsSeen.size > 1) {
          this.push(_pick('new_endpoint', { endpoint: ev.path }), 'info');
        }
      }
      // First 5xx detection (per minute)
      if (ev.status_code >= 500) {
        const hasFiveInWindow = this.entries.some(e =>
          (now - e.time) < 60 && e.tone === 'danger');
        if (!hasFiveInWindow) {
          this.push(_pick('first_5xx', {
            endpoint: ev.path, time: this._fmtTime(now)
          }), 'danger');
        }
      }
      // Slowdown detection — p95 alert per endpoint, max once per 30s
      if (ev.latency_ms && ev.latency_ms > 1500 && ev.path) {
        const last = this.lastSlowdown[ev.path] || 0;
        if (now - last > 30) {
          this.lastSlowdown[ev.path] = now;
          this.push(_pick('slowdown', {
            endpoint: ev.path, lat: (ev.latency_ms / 1000).toFixed(1) + 's'
          }), 'warn');
        }
      }
    }
    _tick() {
      // Runs every 5s — looks for slower-moving patterns
      const stats = this.accum.rollingStats(10);
      const stats30 = this.accum.rollingStats(30);
      const lastAge = this.accum.lastEventAge();

      // Burst detection — > 5 events in 10s window
      if (stats.totalCount > 5) {
        const since = _now() - this.lastBurstAt;
        if (since > 30) {   // don't spam
          this.lastBurstAt = _now();
          this.push(_pick('burst', {
            time: this._fmtTime(_now()),
            n: stats.totalCount, dur: 10
          }), 'info');
        }
      }

      // Error-rate drift
      if (stats30.totalCount >= 5) {
        const er = stats30.errorRate;
        if (this.lastErrorRate != null) {
          if (er - this.lastErrorRate > 5) {
            this.push(_pick('error_drift', {
              old: this.lastErrorRate.toFixed(0),
              new: er.toFixed(0)
            }), 'warn');
          } else if (this.lastErrorRate - er > 5) {
            this.push(_pick('error_recovery', {
              old: this.lastErrorRate.toFixed(0),
              new: er.toFixed(0)
            }), 'great');
          }
        }
        this.lastErrorRate = er;
      }

      // Healthy streak — 30s of clean traffic, fire once per 60s
      if (stats30.totalCount >= 5 && stats30.totalErrors === 0
            && _now() - this._lastHealthyStreakAt > 60) {
        this._lastHealthyStreakAt = _now();
        this.push(_pick('healthy_streak', {
          dur: '30s', n: stats30.totalCount
        }), 'great');
      }

      // Silence — no events in 90s, fire once per 180s
      if (lastAge != null && lastAge > 90 && _now() - this._lastSilenceAt > 180) {
        this._lastSilenceAt = _now();
        const mins = Math.floor(lastAge / 60);
        const txt = mins >= 1 ? mins + ' min' : lastAge + 's';
        this.push(_pick('silence', { dur: txt }), 'info');
      }
    }
  }

  const _commentary = new CommentaryEngine(_accum);
  window.APIN.commentary = {
    engine: _commentary,
    subscribe: fn => _commentary.subscribe(fn),
    entries: () => _commentary.entries.slice(),
    attachTicker(host, opts) {
      // Single-line ticker (for compact widgets). Rotates through entries,
      // newest first, fades after 30s.
      opts = opts || {};
      if (!host) return;
      host.innerHTML = '<div class="lc-ticker" data-empty="true"><span class="lc-bullet">—</span><span class="lc-text">no commentary yet</span></div>';
      const render = (entry) => {
        const wrap = host.querySelector('.lc-ticker');
        if (!wrap || !entry) return;
        const tone = entry.tone || 'info';
        wrap.setAttribute('data-tone', tone);
        wrap.removeAttribute('data-empty');
        wrap.style.opacity = '0';
        setTimeout(() => {
          wrap.innerHTML = '<span class="lc-bullet">·</span><span class="lc-text">' +
            (entry.text || '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])) + '</span>';
          wrap.style.opacity = '1';
        }, 200);
      };
      // Show last entry on attach
      const last = _commentary.entries[0];
      if (last) render(last);
      return _commentary.subscribe(render);
    },
    attachList(host, opts) {
      // Full list (for expanded view).
      opts = opts || {};
      if (!host) return;
      const render = () => {
        const entries = _commentary.entries.slice(0, opts.max || 20);
        if (entries.length === 0) {
          host.innerHTML = '<div class="lc-empty">no commentary yet — make a request to see narrative observations</div>';
          return;
        }
        host.innerHTML = entries.map(e => {
          const ago = _now() - e.time;
          const agoTxt = ago < 60 ? ago + 's ago' : Math.floor(ago / 60) + 'm ago';
          const tone = e.tone || 'info';
          return '<div class="lc-row" data-tone="' + tone + '">'
            + '<span class="lc-bullet">·</span>'
            + '<span class="lc-text">' + (e.text || '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])) + '</span>'
            + '<span class="lc-time">' + agoTxt + '</span>'
            + '</div>';
        }).join('');
      };
      render();
      return _commentary.subscribe(render);
    },
  };

  // ═══════════════════════════════════════════════════════════════════
  // ─── Live pulse widget (line chart) ────────────────────────────────
  // ═══════════════════════════════════════════════════════════════════

  const METRIC_DEFS = {
    rate:   { label: 'req/sec',     yFormat: v => v.toFixed(1),                color: 'var(--ink, #1a1612)' },
    errors: { label: 'error rate',  yFormat: v => v.toFixed(1) + '%',          color: 'var(--c-amber, #d49620)' },
    p95:    { label: 'p95 latency', yFormat: v => v < 1000 ? Math.round(v) + 'ms' : (v/1000).toFixed(1) + 's', color: 'var(--c-info, #2d6a96)' },
    p50:    { label: 'p50 latency', yFormat: v => v < 1000 ? Math.round(v) + 'ms' : (v/1000).toFixed(1) + 's', color: 'var(--c-info, #2d6a96)' },
    bytes:  { label: 'bytes/sec',   yFormat: v => v >= 1024 ? (v/1024).toFixed(1) + ' KB/s' : Math.round(v) + ' B/s', color: 'var(--ink-soft, #6b6453)' },
  };

  function attach(hostEl, opts) {
    opts = opts || {};
    if (!hostEl) return { destroy() {} };
    const state = {
      windowSec: opts.windowSec || 60,
      metric: opts.metric || 'rate',
      stacked: false,   // when true, show by-status breakdown
      compact: !!opts.compact,
      paused: false,
      onClickPoint: opts.onClickPoint,
    };

    // Build shell — chrome only (bar + chart container + footer).
    // The SVG INSIDES are built imperatively below so we can keep stable
    // node refs and mutate them in-place each tick (no innerHTML thrash).
    hostEl.innerHTML =
      '<div class="lp-shell">'
      + '<div class="lp-bar">'
      +   '<div class="lp-bar-left">'
      +     '<span class="lp-pulse-dot" data-state="idle" title="live state"></span>'
      +     '<span class="lp-metric-label">' + (METRIC_DEFS[state.metric].label) + '</span>'
      +   '</div>'
      +   '<div class="lp-bar-spacer"></div>'
      +   '<div class="lp-controls">'
      +     '<div class="lp-toggle" data-toggle="metric">'
      +       Object.entries(METRIC_DEFS).map(([k, def]) =>
              '<button data-val="' + k + '"' + (k === state.metric ? ' aria-pressed="true"' : '') + ' title="' + def.label + '">' + def.label + '</button>'
            ).join('')
      +     '</div>'
      +     '<div class="lp-toggle" data-toggle="window">'
      +       '<button data-val="60"  ' + (state.windowSec === 60 ? 'aria-pressed="true"' : '') + '>60s</button>'
      +       '<button data-val="300" ' + (state.windowSec === 300 ? 'aria-pressed="true"' : '') + '>5m</button>'
      +     '</div>'
      +   '</div>'
      + '</div>'
      + '<div class="lp-chart" tabindex="0">'
      +   '<canvas class="lp-canvas"></canvas>'
      +   '<svg viewBox="0 0 100 100" preserveAspectRatio="none" class="lp-svg"></svg>'
      +   '<div class="lp-tooltip" hidden></div>'
      +   '<div class="lp-cursor" hidden></div>'
      + '</div>'
      + '<div class="lp-footer">'
      +   '<span class="lp-stat lp-stat-current">·</span>'
      +   '<span class="lp-stat lp-stat-peak">peak ·</span>'
      +   (state.compact
            ? '<span class="lp-ticker-host"></span>'
            : '<span class="lp-stat lp-stat-window">' + (state.windowSec === 60 ? 'last 60s' : 'last 5m') + '</span>')
      + '</div>'
      + '</div>';

    const svgEl     = hostEl.querySelector('.lp-svg');
    const tipEl     = hostEl.querySelector('.lp-tooltip');
    const cursorEl  = hostEl.querySelector('.lp-cursor');
    const pulseDot  = hostEl.querySelector('.lp-pulse-dot');
    const curStat   = hostEl.querySelector('.lp-stat-current');
    const peakStat  = hostEl.querySelector('.lp-stat-peak');
    const winStat   = hostEl.querySelector('.lp-stat-window');
    const metricLbl = hostEl.querySelector('.lp-metric-label');
    const chartHost = hostEl.querySelector('.lp-chart');
    let tickerUnsub = null;
    const tickerHost = hostEl.querySelector('.lp-ticker-host');
    if (state.compact && tickerHost && window.APIN && APIN.commentary) {
      tickerUnsub = APIN.commentary.attachTicker(tickerHost);
    }

    // ── Build SVG skeleton ONCE — we mutate attributes on each tick ─────
    // Why: SVG.innerHTML = ... at any rate forces full re-parse + reflow.
    // The previous design rebuilt ~3KB innerHTML at 4Hz → renderer froze
    // when a JS eval also forced layout. With named refs + setAttribute
    // we now do O(1) DOM work per frame, so we can run at 24fps safely.
    const SVGNS = 'http://www.w3.org/2000/svg';
    const _mkSvg = (tag, attrs) => {
      const el = document.createElementNS(SVGNS, tag);
      if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
      return el;
    };
    // defs (hatch pattern for area fill)
    const defs = _mkSvg('defs');
    const pat = _mkSvg('pattern', { id:'lp-hatch', patternUnits:'userSpaceOnUse', width:5, height:5, patternTransform:'rotate(45)' });
    pat.appendChild(_mkSvg('line', { x1:0, y1:0, x2:0, y2:5, stroke:'currentColor', 'stroke-width':0.7, opacity:0.30 }));
    defs.appendChild(pat);
    svgEl.appendChild(defs);

    // tick layer — 3 horizontal gridlines + 3 y-axis labels
    const tickGroup = _mkSvg('g', { class:'lp-ticks' });
    const tickLines = [0,1,2].map(() => _mkSvg('line', { stroke:'var(--paper-edge, #c7bca9)', 'stroke-width':0.4, 'stroke-dasharray':'1.5 2', opacity:0.5 }));
    const tickTexts = [0,1,2].map(() => {
      const t = _mkSvg('text', { 'text-anchor':'end', 'font-family':'JetBrains Mono,monospace', 'font-size':8, fill:'var(--ink-soft, #6b6453)', style:'font-variant-numeric:tabular-nums' });
      t.textContent = '';
      return t;
    });
    tickLines.forEach(l => tickGroup.appendChild(l));
    tickTexts.forEach(t => tickGroup.appendChild(t));
    svgEl.appendChild(tickGroup);

    // SVG keeps the STATIC chrome only (low-frequency updates):
    //   - baseline (1 line, repositioned on resize only)
    //   - pulse dot (2 small circles, repositioned each tick — cheap)
    //   - capture rect (mouse hit-test target, repositioned on resize only)
    // The DYNAMIC layer (line + area + halo) goes on the canvas below.
    const baselineEl = _mkSvg('line', { stroke:'var(--ink-soft, #6b6453)', 'stroke-width':0.7, opacity:0.6 });
    svgEl.appendChild(baselineEl);
    const dotOuterEl = _mkSvg('circle', { r:3.4, class:'lp-svg-pulse', fill:'var(--paper, #fbf9f3)' });
    svgEl.appendChild(dotOuterEl);
    const dotInnerEl = _mkSvg('circle', { r:2, class:'lp-svg-pulse-inner' });
    svgEl.appendChild(dotInnerEl);
    const captureEl = _mkSvg('rect', { class:'lp-capture', fill:'transparent' });
    svgEl.appendChild(captureEl);

    // ── Canvas setup ────────────────────────────────────────────────────
    // Why canvas for the dynamic data layer? Industry-standard live-chart
    // libraries (TradingView Lightweight Charts, Grafana, Datadog, Chart.js
    // streaming) all use canvas because:
    //   - clearRect + bezierCurveTo are direct pixel ops; no parse, no
    //     layout, no GC pressure from setAttribute('d', longString).
    //   - One bitmap allocation (W*H*4 bytes ≈ 830KB for our chart) regardless
    //     of how many data points or paths we draw.
    //   - Browser raster pipeline is heavily optimized for canvas; SVG path
    //     re-stroking on every change is the slow path.
    //   - No DOM nodes per data point → no style/layout invalidation cascade.
    //
    // HiDPI handling: cap devicePixelRatio at 2 for performance. On a 4K
    // retina (DPR=2), we draw at 2x; on DPR=3 phones, we'd downscale to 2.
    // This gives crisp visuals without paying 9x pixel cost on DPR=3.
    const canvasEl = hostEl.querySelector('.lp-canvas');
    // 9.N.7.f · Plain 2d context — NO desynchronized:true.
    // desynchronized puts the canvas on its own GPU compositor surface
    // (intended for low-latency video). On many drivers it renders opaque
    // BLACK until every pixel is explicitly painted, even with alpha:true.
    // That caused the chart to look like a black box. Plain context
    // composites through the normal alpha pipeline → transparent canvas
    // properly shows the .lp-chart paper background behind it.
    const ctx = canvasEl.getContext('2d', { alpha: true });
    let _dpr = Math.min(window.devicePixelRatio || 1, 2);

    // Cached layout — only recomputed on ResizeObserver tick
    let _W = 320, _H = 140, _padL = 32, _padR = 10, _padT = 6, _padB = 14;
    function _recomputeLayout() {
      const rect = chartHost.getBoundingClientRect();
      _W = Math.max(40, Math.round(rect.width)) || 320;
      _H = Math.max(40, Math.round(rect.height)) || 140;
      svgEl.setAttribute('viewBox', '0 0 ' + _W + ' ' + _H);
      captureEl.setAttribute('x', _padL); captureEl.setAttribute('y', _padT);
      captureEl.setAttribute('width', _W - _padL - _padR);
      captureEl.setAttribute('height', _H - _padT - _padB);
      // Resize canvas — note we resize backing-store at DPR but CSS-size at
      // viewport units, then scale ctx. This is the standard HiDPI dance.
      _dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvasEl.width  = Math.round(_W * _dpr);
      canvasEl.height = Math.round(_H * _dpr);
      canvasEl.style.width  = _W + 'px';
      canvasEl.style.height = _H + 'px';
      ctx.setTransform(_dpr, 0, 0, _dpr, 0, 0);   // identity * dpr scale
    }
    _recomputeLayout();
    const _ro = (typeof ResizeObserver !== 'undefined') ? new ResizeObserver(_recomputeLayout) : null;
    if (_ro) _ro.observe(chartHost);

    // Tick layer needs to update y-axis labels when scale changes — handled
    // separately from canvas redraw since it updates rarely (Y-axis EMA
    // changes label text slowly, not every frame).
    let _lastTickMax = null;
    let _lastTickMetric = null;

    // Y-axis EMA — grow instantly on new highs, decay slowly so a passing
    // spike doesn't make all earlier history look like ant hills.
    let _displayMax = null;
    // Floor: first 30 seconds of widget life use a minimum nice-max of 5
    // so a single 1-req second doesn't fill the entire chart vertically.
    const _bootAt = Date.now();
    function _niceMax(m) {
      if (m <= 1) return 1;
      const pow = Math.pow(10, Math.floor(Math.log10(m)));
      const f = m / pow;
      const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
      return nf * pow;
    }

    // Toggle wiring
    hostEl.querySelectorAll('.lp-toggle [data-val]').forEach(btn => {
      btn.addEventListener('click', () => {
        const toggle = btn.closest('.lp-toggle')?.getAttribute('data-toggle');
        const val = btn.getAttribute('data-val');
        if (toggle === 'metric') {
          state.metric = val;
          metricLbl.textContent = METRIC_DEFS[val].label;
        }
        if (toggle === 'window') {
          state.windowSec = Number(val);
          if (winStat) winStat.textContent = state.windowSec === 60 ? 'last 60s' : 'last 5m';
        }
        btn.closest('.lp-toggle').querySelectorAll('button').forEach(b => b.removeAttribute('aria-pressed'));
        btn.setAttribute('aria-pressed', 'true');
      });
    });

    // ── Render loop — attribute mutations only (no innerHTML rebuild) ───
    // Runs at ~24fps. Because each tick only sets a handful of attributes
    // on existing nodes, the cost is O(points) work + 0 layout thrash.
    // Continuous x-position recomputation gives smooth video-like scroll
    // even when no events arrive (the time-axis slides; the line trails).
    // Helper to resolve CSS variables to actual colors for canvas.
    // Canvas ctx.fillStyle/strokeStyle don't accept "var(--c-ok)" — must be
    // a literal "#2f6f3e" or "rgb(...)". We parse the var() syntax to find
    // the token; if the document hasn't defined it, use the fallback color
    // embedded in the var() declaration itself. Cached after first lookup.
    const _colorCache = {};
    function _resolveColor(cssVar, fallback) {
      if (_colorCache[cssVar]) return _colorCache[cssVar];
      let resolved = cssVar;
      if (typeof cssVar === 'string' && cssVar.indexOf('var(') === 0) {
        // Parse: var(--token-name, fallback-color)
        const m = cssVar.match(/^var\(\s*(--[^,)]+)\s*(?:,\s*([^)]+))?\s*\)$/);
        if (m) {
          const token = m[1].trim();
          const cssFallback = (m[2] || '').trim();
          const v = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
          resolved = v || cssFallback || fallback;
        } else {
          resolved = fallback;
        }
      }
      _colorCache[cssVar] = resolved;
      return resolved;
    }

    let _hasPainted = false;
    function _draw() {
      if (!hostEl.isConnected) return;
      // Skip subsequent draws when tab is hidden — but always do at LEAST
      // one paint so the chart shows a flatline if the user comes back.
      if (_hasPainted && typeof document !== 'undefined' && document.hidden) return;
      _hasPainted = true;
      const iw = _W - _padL - _padR;
      const ih = _H - _padT - _padB;
      const baseY = _padT + ih;

      const data = _accum.series(state.metric, state.windowSec);
      // ── Y-axis EMA — sticky scale so a passing spike doesn't dwarf
      //    surrounding history when it disappears.
      const rawMax = data.reduce((m, d) => d.v > m ? d.v : m, 0);
      // Boot floor: minimum 5 for first 30s so a 1-req spike doesn't
      // visually fill the chart on cold start.
      const isBoot = (Date.now() - _bootAt) < 30000;
      const target = _niceMax(Math.max(rawMax, isBoot ? 5 : 1));
      if (_displayMax == null) {
        _displayMax = target;
      } else if (target > _displayMax) {
        _displayMax = target;   // grow instantly so spike is visible
      } else {
        // Decay slowly — at 10fps, 0.012 ≈ 12%/s shrink. Earlier peaks
        // remain visible for ~10s after they pass.
        _displayMax = _displayMax + (target - _displayMax) * 0.012;
        if (_displayMax < target) _displayMax = target;
      }
      const useMax = _displayMax;

      // ── Sub-second smooth scroll: position by wall-clock float
      const nowFloat = Date.now() / 1000;
      const xFor = ts => _padL + (1 - (nowFloat - ts) / state.windowSec) * iw;
      const yFor = v => _padT + ih - (Math.min(v, useMax) / useMax) * ih;

      // Compute points — single loop, no string allocation
      const pts = [];
      for (let i = 0; i < data.length; i++) {
        const d = data[i];
        const x = xFor(d.t);
        if (x < _padL - 2) continue;
        pts.push({ x: x, y: yFor(d.v), v: d.v, t: d.t });
      }
      const lastBucketV = data.length ? data[data.length - 1].v : 0;
      const tipX = _padL + iw;
      const tipY = yFor(lastBucketV);
      // Synthetic right-edge point so the line always extends to "now",
      // giving a visible flatline when no events arrive.
      pts.push({ x: tipX, y: tipY, v: lastBucketV, t: nowFloat });

      // ── Canvas pass: clear + draw area + halo + line ─────────────────
      // clearRect is O(pixels) — bounded, no parse, no GC.
      ctx.clearRect(0, 0, _W, _H);

      if (pts.length > 1) {
        // Build the smoothed path once (Catmull-Rom → cubic Bezier).
        // We use Path2D so the area fill, halo, and line strokes all
        // reuse the same path without redoing the bezier math.
        const path = new Path2D();
        path.moveTo(pts[0].x, pts[0].y);
        for (let i = 0; i < pts.length - 1; i++) {
          const p0 = pts[Math.max(0, i - 1)];
          const p1 = pts[i];
          const p2 = pts[i + 1];
          const p3 = pts[Math.min(pts.length - 1, i + 2)];
          const c1x = p1.x + (p2.x - p0.x) / 6;
          const c1y = p1.y + (p2.y - p0.y) / 6;
          const c2x = p2.x - (p3.x - p1.x) / 6;
          const c2y = p2.y - (p3.y - p1.y) / 6;
          path.bezierCurveTo(c1x, c1y, c2x, c2y, p2.x, p2.y);
        }

        // ── Area fill (subtle wash below the line) ─────────────────────
        // Close path to baseline for a fill region. We use globalAlpha
        // rather than hex8 ("#rrggbbaa") so the fill color can be a CSS
        // named color or rgb(...) just as easily.
        const area = new Path2D(path);
        area.lineTo(pts[pts.length - 1].x, baseY);
        area.lineTo(pts[0].x, baseY);
        area.closePath();
        ctx.fillStyle = _resolveColor('var(--ink-soft, #6b6453)', '#6b6453');
        ctx.globalAlpha = 0.13;
        ctx.fill(area);
        ctx.globalAlpha = 1.0;

        const lineColor = _resolveColor(METRIC_DEFS[state.metric].color, '#1a1612');

        // ── Halo (wider, translucent stroke beneath main line) ─────────
        ctx.strokeStyle = lineColor;
        ctx.globalAlpha = 0.18;
        ctx.lineWidth = 2.6;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.stroke(path);

        // ── Main line ──────────────────────────────────────────────────
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = 1.4;
        ctx.stroke(path);
      }

      // ── SVG chrome updates (low-frequency: only when changed) ────────
      // Y-axis labels: only re-set text + position when scale changes.
      const needTickUpdate = (Math.abs((_lastTickMax || -1) - useMax) > 1e-6)
                          || _lastTickMetric !== state.metric;
      if (needTickUpdate) {
        const yTickValues = [0, useMax / 2, useMax];
        for (let i = 0; i < 3; i++) {
          const tv = yTickValues[i];
          const ty = yFor(tv);
          tickLines[i].setAttribute('x1', _padL);
          tickLines[i].setAttribute('x2', _padL + iw);
          tickLines[i].setAttribute('y1', ty);
          tickLines[i].setAttribute('y2', ty);
          tickTexts[i].setAttribute('x', _padL - 5);
          tickTexts[i].setAttribute('y', ty + 3);
          tickTexts[i].textContent = METRIC_DEFS[state.metric].yFormat(tv);
        }
        // Baseline only repositioned when layout (W/H) or metric changes
        baselineEl.setAttribute('x1', _padL);
        baselineEl.setAttribute('x2', _padL + iw);
        baselineEl.setAttribute('y1', baseY);
        baselineEl.setAttribute('y2', baseY);
        _lastTickMax = useMax;
        _lastTickMetric = state.metric;
      }
      // Pulse dot — these are cheap, do every frame
      dotOuterEl.setAttribute('cx', tipX);
      dotOuterEl.setAttribute('cy', tipY);
      dotInnerEl.setAttribute('cx', tipX);
      dotInnerEl.setAttribute('cy', tipY);

      // Pulse state (sage / amber / crimson)
      const ageSec = _accum.lastEventAge();
      const isDisc = (window.APIN && APIN.liveStreamConn && APIN.liveStreamConn.state) === 'disconnected';
      let pulseState = 'active';
      if (isDisc) pulseState = 'disc';
      else if (ageSec == null || ageSec > 30) pulseState = 'idle';
      if (pulseDot.getAttribute('data-state') !== pulseState) {
        pulseDot.setAttribute('data-state', pulseState);
      }

      // Footer stats (text content updates — cheap)
      const peakV = data.reduce((m, d) => d.v > m ? d.v : m, 0);
      curStat.textContent = 'current ' + METRIC_DEFS[state.metric].yFormat(lastBucketV);
      peakStat.textContent = 'peak ' + METRIC_DEFS[state.metric].yFormat(peakV);
    }
    // 9.N.7.f · requestAnimationFrame loop with internal throttling.
    // Why rAF over setInterval: when the main thread is busy (SSE bursts,
    // GC, layout reflows from elsewhere), rAF naturally backs off — the
    // browser skips firing the callback. setInterval keeps queueing ticks
    // that all fire when the thread frees up, causing a stampede.
    //
    // Now that we render via canvas (no DOM thrash per frame), we can run
    // closer to native refresh: target ~30fps. The canvas pipeline is:
    //   - 1 clearRect (W*H pixels)
    //   - 1 Path2D bezier construction (O(60-300) ops)
    //   - 3 ctx.stroke/fill calls
    //   - 0 setAttribute, 0 SVG parse, 0 GC pressure
    // Compared to SVG-only (~30 setAttribute calls + 3 path parses per frame),
    // this is ~10x cheaper. We can also coalesce: if rAF fires faster than
    // 30fps target, we still respect the throttle.
    let _lastPaintTs = 0;
    let _dirty = true;
    let _rafHandle = 0;
    const PAINT_INTERVAL_MS = 33;   // ~30fps target. rAF backs off naturally
                                     // if the main thread can't keep up.
    function _loop(ts) {
      if (!hostEl.isConnected) return;   // widget removed
      // When tab hidden, skip work — but only after first paint
      if (_hasPainted && typeof document !== 'undefined' && document.hidden) {
        _rafHandle = requestAnimationFrame(_loop);
        return;
      }
      if (_dirty || (ts - _lastPaintTs) >= PAINT_INTERVAL_MS) {
        _dirty = false;
        _lastPaintTs = ts;
        try { _draw(); } catch (e) {
          // Defensive: a broken draw must not kill the loop.
          if (window.console) console.warn('lp draw error', e);
        }
      }
      _rafHandle = requestAnimationFrame(_loop);
    }
    // Mark dirty whenever any event arrives, so paints happen ASAP on
    // burst rather than waiting up to 100ms.
    const _unsubDirty = _accum.subscribe(() => { _dirty = true; });
    _rafHandle = requestAnimationFrame(_loop);
    _draw();   // initial paint (so flatline shows even if user never focuses tab)

    // ── Hover tooltip + click ───────────────────────────────────────────
    // Resolve event-cluster at cursor x. Returns: { tStart, tEnd, events,
    // statusBreakdown, latP50, latP95 }. Uses a ±1s window because at 60s
    // visible width on a ~860px chart, 1 second ≈ 14px — humans can't
    // pixel-snap to a single second, so we cluster nearby buckets.
    function _eventsNearCursor(e) {
      const rect = svgEl.getBoundingClientRect();
      if (rect.width <= 0) return null;
      const W = rect.width;
      const iw = W - _padL - _padR;
      const xPx = (e.clientX - rect.left) * (W / rect.width);
      const now = Date.now() / 1000;
      const t = now - (1 - (xPx - _padL) / iw) * state.windowSec;
      const tInt = Math.round(t);
      // Cluster ±1 second to make hover tolerant
      const clusterRadius = state.windowSec <= 60 ? 1 : 2;
      const events = _accum.eventsAt(tInt, clusterRadius);
      const status = { '2xx':0, '3xx':0, '4xx':0, '5xx':0, '429':0, 'other':0 };
      let latencies = [];
      let bytes = 0;
      const paths = new Set();
      for (const ev of events) {
        const sb = _statusBucket(ev.status_code);
        status[sb]++;
        if (ev.latency_ms != null) latencies.push(Number(ev.latency_ms));
        if (ev.bytes_out) bytes += Number(ev.bytes_out) || 0;
        if (ev.path) paths.add(ev.path);
      }
      return {
        tInt, events, status, latencies, bytes, paths,
        xPx, cursorY: e.clientY - rect.top,
      };
    }

    function _showTooltip(e) {
      const info = _eventsNearCursor(e);
      if (!info) return;
      const { tInt, events, status, latencies, paths } = info;
      const rect = svgEl.getBoundingClientRect();
      const timeStr = new Date(tInt * 1000).toTimeString().slice(0, 8);

      // Build status-split rows (only show non-zero counts)
      const statusRows = [];
      const statusColors = { '2xx':'var(--c-ok, #2f6f3e)', '3xx':'var(--ink-soft, #6b6453)', '4xx':'var(--c-amber, #d49620)', '5xx':'var(--c-danger, #b13d2e)', '429':'var(--c-amber, #d49620)', 'other':'var(--ink-soft, #6b6453)' };
      for (const k of ['2xx','3xx','4xx','5xx','429']) {
        if (status[k] > 0) {
          statusRows.push(
            '<div style="display:flex;justify-content:space-between;gap:10px;font-size:10.5px">' +
              '<span style="color:' + statusColors[k] + '">' + k + '</span>' +
              '<span style="color:var(--ink);font-variant-numeric:tabular-nums">' + status[k] + '</span>' +
            '</div>'
          );
        }
      }

      let html = '<div style="font-family:JetBrains Mono,monospace;font-size:11px;line-height:1.4">';
      html += '<div style="font-weight:600;color:var(--ink);margin-bottom:4px">' + timeStr + '</div>';
      if (events.length === 0) {
        html += '<div style="color:var(--ink-soft);font-style:italic">no requests this second</div>';
      } else {
        html += '<div style="margin-bottom:4px;color:var(--ink)">' + events.length + ' request' + (events.length === 1 ? '' : 's') + '</div>';
        if (statusRows.length > 0) {
          html += '<div style="border-top:1px dashed var(--paper-edge);padding-top:4px;margin-top:4px">' + statusRows.join('') + '</div>';
        }
        if (latencies.length > 0) {
          const p50 = Math.round(_pct(latencies, 0.5));
          const p95 = Math.round(_pct(latencies, 0.95));
          html += '<div style="border-top:1px dashed var(--paper-edge);padding-top:4px;margin-top:4px;color:var(--ink-soft);font-size:10.5px">' +
                  'p50 ' + p50 + 'ms · p95 ' + p95 + 'ms</div>';
        }
        if (paths.size > 0 && paths.size <= 3) {
          const pathList = Array.from(paths).map(p => '<div style="font-size:10px;color:var(--ink-soft);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px">· ' + p.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])) + '</div>').join('');
          html += '<div style="margin-top:4px">' + pathList + '</div>';
        } else if (paths.size > 3) {
          html += '<div style="margin-top:4px;font-size:10px;color:var(--ink-soft)">' + paths.size + ' endpoints</div>';
        }
        html += '<div style="margin-top:5px;font-size:9.5px;color:var(--ink-soft);font-style:italic;border-top:1px dashed var(--paper-edge);padding-top:4px">click to inspect</div>';
      }
      html += '</div>';
      tipEl.innerHTML = html;

      // Position tooltip — flip to left side if it would overflow chart right
      const cursorX = e.clientX - rect.left;
      const cursorY = e.clientY - rect.top;
      tipEl.style.left = (cursorX + 12) + 'px';
      tipEl.style.top  = Math.max(4, cursorY - 60) + 'px';
      tipEl.hidden = false;
      // After layout, check overflow and flip if needed
      requestAnimationFrame(() => {
        if (tipEl.hidden) return;
        const tr = tipEl.getBoundingClientRect();
        const cr = chartHost.getBoundingClientRect();
        if (tr.right > cr.right - 4) {
          tipEl.style.left = Math.max(4, cursorX - tr.width - 12) + 'px';
        }
      });
      cursorEl.style.left = cursorX + 'px';
      cursorEl.hidden = false;
    }

    chartHost.addEventListener('mousemove', _showTooltip);
    chartHost.addEventListener('mouseleave', () => {
      tipEl.hidden = true;
      cursorEl.hidden = true;
    });
    chartHost.addEventListener('click', e => {
      const info = _eventsNearCursor(e);
      if (!info || info.events.length === 0) return;
      // Pass FULL array of events at this point — caller may open a list
      // modal (preferred for >1 event) or just inspect the first.
      if (opts.onClickPoint) {
        // Backward-compat: callers expecting a single event get the first;
        // callers wanting multi will detect Array.isArray(arg) or arg.length.
        opts.onClickPoint(info.events.length === 1 ? info.events[0] : info.events, {
          allEvents: info.events,
          tInt: info.tInt,
          statusBreakdown: info.status,
        });
      }
    });

    // Redraw on visibility change so a freshly-revealed tab gets a paint
    // immediately rather than waiting for the next 42ms tick.
    const _visHandler = () => { if (!document.hidden) _draw(); };
    try { document.addEventListener('visibilitychange', _visHandler); } catch (e) {}

    return {
      destroy() {
        if (_rafHandle) cancelAnimationFrame(_rafHandle);
        if (_ro) { try { _ro.disconnect(); } catch (e) {} }
        try { document.removeEventListener('visibilitychange', _visHandler); } catch (e) {}
        try { _unsubDirty(); } catch (e) {}
        if (tickerUnsub) tickerUnsub();
      },
      /** Force a paint — useful for QA when document.hidden=true */
      redraw() { _draw(); },
      setMetric(m) { state.metric = m; metricLbl.textContent = METRIC_DEFS[m].label; },
      setWindow(s) { state.windowSec = s; if (winStat) winStat.textContent = s === 60 ? 'last 60s' : 'last 5m'; },
      state,
    };
  }

  window.APIN.livePulse = { attach };
})();
