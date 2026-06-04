/* ADM-T · Stats Deck — a premium 3D observability deck (the product, not a
 * container). 18 intelligence artifacts ride a CSS-3D circular ring you drag /
 * scroll to rotate infinitely; hover lifts + tilts + animates an artifact;
 * click unfolds its room; pin flies a card to the Pinned Canvas. One deck,
 * mounted in BOTH Traffic panes, fed that page's real data (admin_deck).
 *
 * Each card owns a metaphor + motion language. Real data where computable; an
 * honest "calibrating" face where the analytic needs more signal. No fabrication.
 *
 * window.ADM_DECK.create(host, dataset) -> { dispose }.
 */
(function () {
  'use strict';
  var A = window.ADM || {};
  var esc = A.esc || function (s) { return String(s == null ? '' : s); };
  var fmtInt = A.fmtInt || function (n) { return String(n); };

  function fetchDeck(dataset) {
    var url = '/api/account/admin/traffic?widget=deck_' + dataset + '&window=all';
    if (A.adminFetch) return A.adminFetch(url);
    var csrf = (document.querySelector('meta[name=csrf-token]') || {}).content || '';
    return fetch(url, { credentials: 'include', headers: { 'X-Console-Csrf': csrf } })
      .then(function (r) { return r.json(); }).then(function (j) { return j.data || j; }).catch(function () { return null; });
  }

  // ── tiny shared artifact helpers ─────────────────────────────────────────────
  function spark(series, w, h, color, fill) {
    series = (series || []).filter(function (s) { return s != null; });
    if (!series.length) return '';
    var max = series.reduce(function (a, b) { return Math.max(a, (b.c != null ? b.c : (b.err_rate != null ? b.err_rate : b)) || 0); }, 0) || 1;
    var n = series.length, dx = w / (n - 1 || 1);
    var pts = series.map(function (s, i) { var v = (s.c != null ? s.c : (s.err_rate != null ? s.err_rate : s)) || 0; return [i * dx, h - 3 - (h - 6) * (v / max)]; });
    var path = pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ');
    var area = fill ? '<path d="M0 ' + h + ' ' + path.replace('M', 'L') + ' L' + w + ' ' + h + ' Z" fill="' + color + '" opacity="0.16"/>' : '';
    return '<svg viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" class="dk-spark">' + area + '<path d="' + path + '" fill="none" stroke="' + color + '" stroke-width="1.6"/></svg>';
  }
  function calib(reason) { return '<div class="dk-calib"><span class="dk-calib-dot"></span>calibrating<small>' + esc(reason || 'collecting signal') + '</small></div>'; }
  function tail(p) { p = (p == null ? '' : String(p)); return p.replace(/^\/api\//, '').replace(/^\//, ''); }
  // [F3] attribute-safe tooltip payload: bold value + dim label; only quotes escaped
  function dtip(val, lab) { return ('<b>' + esc(val) + '</b><span>' + esc(lab) + '</span>').replace(/"/g, '&quot;'); }

  // ── the 18 artifacts (spec order). Each: title, metaphor, art(), sum(). ──────
  var SEV = { c: '#4ade80', w: '#e0b341', b: '#e0584a' };
  var ART = [
    { key: 'change_detection', title: 'Change Detection', metaphor: 'Traffic Ripples',
      art: function (d) {
        if (!d || !d.series) return calib('no series');
        var b = d.biggest;
        var ep = tail((d.changes && d.changes[0] && d.changes[0].t) || '');
        var tip = b ? dtip((b.pct > 0 ? '+' : '') + b.pct + '%', ep || 'biggest move') : dtip('·', 'no change');
        return '<div class="art-ripple" data-tip="' + tip + '">' + [1, 2, 3].map(function (i) { return '<span class="rp" style="--i:' + i + '"></span>'; }).join('')
          + '<div class="rp-core' + (b && b.pct > 0 ? ' up' : ' down') + '">' + (b ? (b.pct > 0 ? '+' : '') + b.pct + '%' : '·') + '</div></div>'
          + spark(d.series, 240, 30, '#7da7ff', true);
      },
      sum: function (d) { return d && d.biggest ? '<b>' + tail((d.changes[0] || {}).t || '') + '</b> biggest move · ' + (d.changes || []).length + ' detected' : 'no significant changes'; } },

    { key: 'correlation', title: 'Correlation', metaphor: 'Magnetic Field',
      art: function (d) {
        if (!d || d.state) return calib(d && d.reason);
        var r = (d.pairs && d.pairs[0] && d.pairs[0].r) || 0;
        return '<div class="art-magnet" data-tip="' + dtip('r = ' + r, 'volume × latency') + '"><span class="mg-pole a">vol</span>'
          + '<svg viewBox="0 0 120 60" class="mg-field">' + [0, 1, 2, 3].map(function (i) { var o = 8 + i * 12; return '<path d="M14 30 C40 ' + (30 - o) + ' 80 ' + (30 - o) + ' 106 30" fill="none" stroke="#7da7ff" stroke-width="1" opacity="' + (0.7 - i * 0.13) + '"/><path d="M14 30 C40 ' + (30 + o) + ' 80 ' + (30 + o) + ' 106 30" fill="none" stroke="#7da7ff" stroke-width="1" opacity="' + (0.7 - i * 0.13) + '"/>'; }).join('') + '</svg>'
          + '<span class="mg-pole b">p95</span></div><div class="mg-r">r = ' + r + '</div>';
      },
      sum: function (d) { return d && !d.state ? 'volume × latency · ' + (Math.abs((d.pairs[0] || {}).r || 0) > 0.5 ? 'strong' : 'weak') + ' link' : 'collecting paired signal'; } },

    { key: 'regressions', title: 'Regressions', metaphor: 'Fractured Glass',
      art: function (d) {
        var it = (d && d.items) || []; if (!it.length) return '<div class="art-glass intact">no regressions</div>';
        return '<div class="art-glass"><svg viewBox="0 0 120 70">' + it.slice(0, 5).map(function (e, i) { var x = 20 + i * 20, sev = e.error_rate >= 50 ? SEV.b : SEV.w; return '<g class="dk-crack" data-tip="' + dtip(e.error_rate + '% err', tail(e.path)) + '" stroke="' + sev + '" stroke-width="1.1" opacity="' + (1 - i * 0.13) + '"><line x1="' + x + '" y1="35" x2="' + (x - 12) + '" y2="8"/><line x1="' + x + '" y1="35" x2="' + (x + 12) + '" y2="10"/><line x1="' + x + '" y1="35" x2="' + (x - 10) + '" y2="62"/><line x1="' + x + '" y1="35" x2="' + (x + 11) + '" y2="60"/></g>'; }).join('') + '</svg></div>'; },
      sum: function (d) { var it = (d && d.items) || []; return it.length ? '<b>' + tail(it[0].path) + '</b> ' + it[0].error_rate + '% err · ' + it.length + ' cracks' : 'all clear'; } },

    { key: 'improvements', title: 'Improvements', metaphor: 'Kintsugi',
      art: function (d) {
        var it = (d && d.items) || [];
        return '<div class="art-kintsugi" data-tip="' + dtip(it.length + ' repaired', 'recovered endpoints') + '"><svg viewBox="0 0 120 70">' + [0, 1, 2].map(function (i) { return '<path d="M8 ' + (20 + i * 18) + ' Q40 ' + (12 + i * 18) + ' 60 ' + (22 + i * 18) + ' T112 ' + (18 + i * 18) + '" fill="none" stroke="url(#gold)" stroke-width="2"/>'; }).join('') + '<defs><linearGradient id="gold"><stop offset="0" stop-color="#e0b341"/><stop offset="1" stop-color="#f0d98a"/></linearGradient></defs></svg></div>'; },
      sum: function (d) { var it = (d && d.items) || []; return it.length ? '<b>' + it.length + '</b> healthy endpoints repaired' : 'no recoveries yet'; } },

    { key: 'segments', title: 'Traffic Segments', metaphor: 'Passport Stamps',
      art: function (d) {
        var s = (d && d.segments) || []; if (!s.length) return calib('no segments');
        var col = { production: '#4ade80', testing: '#e0b341', automation: '#e0584a' };
        var tot = s.reduce(function (a, b) { return a + (b.count || 0); }, 0) || 1;
        return '<div class="art-stamps">' + s.slice(0, 3).map(function (x, i) { return '<span class="stamp" data-tip="' + dtip(fmtInt(x.count) + ' reqs', x.label + ' · ' + Math.round(100 * x.count / tot) + '%') + '" style="--r:' + ((i - 1) * 11) + 'deg;border-color:' + (col[x.label] || '#888') + ';color:' + (col[x.label] || '#888') + '">' + esc(x.label.toUpperCase()) + '<b>' + fmtInt(x.count) + '</b></span>'; }).join('') + '</div>'; },
      sum: function (d) { var s = (d && d.segments) || []; return s.length ? s.map(function (x) { return x.label + ' ' + fmtInt(x.count); }).join(' · ') : 'no traffic'; } },

    { key: 'first_seen', title: 'First Seen', metaphor: 'Museum Tags',
      art: function (d) {
        var it = (d && d.items) || [];
        return '<div class="art-tags">' + it.slice(0, 3).map(function (e, i) { return '<span class="mtag" data-tip="' + dtip('first seen', tail(e.path)) + '" style="--r:' + ((i % 2) ? 3 : -2) + 'deg"><i>NEW</i>' + esc(tail(e.path)) + '</span>'; }).join('') + '</div>'; },
      sum: function (d) { var it = (d && d.items) || []; return it.length ? 'newest: <b>' + tail(it[0].path) + '</b>' : 'no endpoints'; } },

    { key: 'last_seen', title: 'Last Seen', metaphor: 'Fading Polaroids',
      art: function (d) {
        var it = (d && d.items) || [];
        return '<div class="art-polaroids">' + it.slice(0, 3).map(function (e, i) { return '<span class="poly" data-tip="' + dtip('last seen', tail(e.path)) + '" style="--o:' + (0.9 - i * 0.28) + ';--r:' + ((i - 1) * 6) + 'deg"><i></i><small>' + esc(tail(e.path)) + '</small></span>'; }).join('') + '</div>'; },
      sum: function (d) { var it = (d && d.items) || []; return it.length ? 'stalest: <b>' + tail(it[0].path) + '</b>' : 'no endpoints'; } },

    { key: 'volatility', title: 'Volatility Index', metaphor: 'Seismograph',
      art: function (d) {
        if (!d || d.state) return calib(d && d.reason);
        return '<div class="art-seismo" data-tip="' + dtip('cv ' + d.index, (d.level || '') + ' volatility') + '"><svg viewBox="0 0 240 40" class="seismo-trace">' + (function () {
          var s = d.series || [], n = s.length, max = s.reduce(function (a, b) { return Math.max(a, b.c || 0); }, 1), dx = 240 / (n - 1 || 1);
          var p = s.map(function (x, i) { var jit = (x.c ? (Math.sin(i * 2.3) * (x.c / max) * 16) : 0); return (i ? 'L' : 'M') + (i * dx).toFixed(1) + ' ' + (20 - jit).toFixed(1); }).join(' ');
          return '<path d="' + p + '" fill="none" stroke="#e0584a" stroke-width="1.3"/>';
        })() + '</svg><span class="seismo-idx">' + d.index + '</span></div>'; },
      sum: function (d) { return d && !d.state ? '<b>' + (d.level || '') + '</b> volatility · cv ' + d.index : 'collecting'; } },

    { key: 'dependency_risk', title: 'Dependency Risk', metaphor: 'Jenga Tower',
      art: function (d) {
        var it = (d && d.items) || [];
        return '<div class="art-jenga">' + it.slice(0, 5).map(function (e, i) { var crit = e.risk >= 0.7; return '<span class="jblock dk-removable' + (crit ? ' crit' : '') + '" data-tip="' + dtip('risk ' + e.risk, tail(e.path)) + '" style="--w:' + (40 + e.risk * 50) + '%">' + (crit ? esc(tail(e.path)) : '') + '</span>'; }).join('') + '</div>'; },
      sum: function (d) { var it = (d && d.items) || []; return it.length ? '<b>' + tail(it[0].path) + '</b> load-bearing · risk ' + it[0].risk : 'stable'; } },

    { key: 'error_fingerprints', title: 'Error Fingerprints', metaphor: 'Forensic Board',
      art: function (d) {
        var it = (d && d.items) || []; if (!it.length) return '<div class="art-forensic clean">no errors</div>';
        return '<div class="art-forensic"><svg viewBox="0 0 120 70">' + it.slice(0, 6).map(function (e, i) { var x = 18 + (i % 3) * 40, y = 20 + ((i / 3) | 0) * 30; return '<line x1="' + x + '" y1="' + y + '" x2="60" y2="35" stroke="#e0584a" stroke-width="0.6" opacity="0.4"/><circle class="dk-pin" data-tip="' + dtip(e.status + ' ×' + e.count, tail(e.path)) + '" cx="' + x + '" cy="' + y + '" r="3.4" fill="#e0584a"/>'; }).join('') + '<circle cx="60" cy="35" r="2" fill="#fff"/></svg></div>'; },
      sum: function (d) { var it = (d && d.items) || []; return it.length ? '<b>' + tail(it[0].path) + '</b> ' + it[0].status + ' ×' + it[0].count + ' · ' + it.length + ' signatures' : 'clean'; } },

    { key: 'forecast', title: 'Forecast', metaphor: 'Weather Orb',
      art: function (d) {
        if (!d || d.state) return calib(d && d.reason);
        var ic = d.trend === 'rising' ? '▲' : (d.trend === 'falling' ? '▼' : '▬');
        return '<div class="art-orb" data-tip="' + dtip((d.trend || 'flat'), '~' + fmtInt(d.projection) + '/bucket projected') + '"><div class="orb"><span class="orb-cloud"></span><span class="orb-cloud b"></span><b>' + ic + '</b></div><span class="orb-proj">~' + fmtInt(d.projection) + '/bucket</span></div>'; },
      sum: function (d) { return d && !d.state ? '<b>' + (d.trend || '') + '</b> · proj ~' + fmtInt(d.projection) : 'collecting'; } },

    { key: 'baseline', title: 'Baseline Comparison', metaphor: 'Ghost Overlay',
      art: function (d) {
        if (!d || d.state) return calib(d && d.reason);
        return '<div class="art-ghost" data-tip="' + dtip(((d.delta_pct || 0) >= 0 ? '+' : '') + d.delta_pct + '%', 'now ' + fmtInt(d.now) + ' vs base ' + fmtInt(d.baseline)) + '">' + spark(d.series, 240, 40, '#4ade80', true) + '<div class="ghost-line" style="--y:' + (40 - 40 * Math.min(1, (d.baseline || 0) / ((d.series || []).reduce(function (a, b) { return Math.max(a, b.c || 0); }, 1)))) + 'px"></div></div>'
          + '<div class="ghost-d ' + ((d.delta_pct || 0) >= 0 ? 'up' : 'down') + '">' + ((d.delta_pct || 0) >= 0 ? '+' : '') + d.delta_pct + '% vs baseline</div>'; },
      sum: function (d) { return d && !d.state ? 'now <b>' + fmtInt(d.now) + '</b> vs base ' + fmtInt(d.baseline) : 'collecting'; } },

    { key: 'journeys', title: 'User Journeys', metaphor: 'Metro Map',
      art: function (d) {
        var rt = (d && d.routes) || []; if (!rt.length) return calib('no sequences');
        return '<div class="art-metro"><svg viewBox="0 0 120 70">' + rt.slice(0, 4).map(function (e, i) { var y = 14 + i * 15; return '<g class="dk-route" data-tip="' + dtip(tail(e.from) + ' → ' + tail(e.to), '×' + e.weight + ' journeys') + '"><line x1="14" y1="' + y + '" x2="106" y2="' + y + '" stroke="#7da7ff" stroke-width="2" opacity="' + (0.9 - i * 0.15) + '"/><circle cx="14" cy="' + y + '" r="3" fill="#fff"/><circle cx="106" cy="' + y + '" r="3" fill="#7da7ff"/></g>'; }).join('') + '</svg></div>'; },
      sum: function (d) { var rt = (d && d.routes) || []; return rt.length ? '<b>' + tail(rt[0].from) + '→' + tail(rt[0].to) + '</b> ×' + rt[0].weight : 'no journeys'; } },

    { key: 'recovery', title: 'Recovery Timeline', metaphor: 'Heartbeat Monitor',
      art: function (d) {
        var s = (d && d.series) || []; if (!s.length) return calib('no error series');
        var peak = s.reduce(function (a, b) { return Math.max(a, b.err_rate || 0); }, 0);
        return '<div class="art-heartbeat" data-tip="' + dtip('peak ' + peak + '%', 'error rate over time') + '">' + spark(s.map(function (x) { return { c: x.err_rate }; }), 240, 40, '#e0584a', false) + '</div>'; },
      sum: function (d) { var s = (d && d.series) || []; var peak = s.reduce(function (a, b) { return Math.max(a, b.err_rate || 0); }, 0); return s.length ? 'peak error <b>' + peak + '%</b>' : 'no incidents'; } },

    { key: 'latency_attribution', title: 'Latency Attribution', metaphor: 'Mechanical Watch',
      art: function (d) {
        var p = (d && d.percentiles) || {};
        return '<div class="art-watch">' + ['p50', 'p90', 'p99'].map(function (k, i) { return '<span class="gear" data-tip="' + dtip((p[k] || 0) + ' ms', k + ' latency') + '" style="--s:' + (10 + i * 5) + 'px;--sp:' + (3 + i) + 's"><b>' + (p[k] || 0) + '</b></span>'; }).join('') + '</div>'; },
      sum: function (d) { var p = (d && d.percentiles) || {}; return 'p50 ' + (p.p50 || 0) + ' · p95 ' + (p.p95 || 0) + ' · p99 ' + (p.p99 || 0) + 'ms'; } },

    { key: 'alert_simulator', title: 'Alert Simulator', metaphor: 'Control Panel',
      art: function (d) {
        var m = (d && d.metrics) || [];
        return '<div class="art-panel">' + m.slice(0, 3).map(function (x) { var on = x.suggested && x.value >= x.suggested; return '<span class="switch ' + (on ? 'on' : 'off') + '" data-tip="' + dtip((on ? 'WOULD FIRE' : 'quiet'), x.label + ' · ' + (x.value != null ? x.value : '?') + (x.suggested ? ' / ' + x.suggested : '')) + '"><i></i>' + esc(x.label) + '</span>'; }).join('') + '</div>'; },
      sum: function (d) { var m = (d && d.metrics) || []; var firing = m.filter(function (x) { return x.suggested && x.value >= x.suggested; }).length; return firing ? '<b>' + firing + '</b> would fire' : 'all quiet'; } },

    { key: 'metric_confidence', title: 'Metric Confidence', metaphor: 'Scientific Instrument',
      art: function (d) {
        var m = (d && d.metrics) || []; var c = (m[0] || {}).confidence || 0;
        var ang = -50 + c * 100;
        return '<div class="art-instrument" data-tip="' + dtip(Math.round(c * 100) + '% confident', 'n=' + fmtInt((m[0] || {}).n || 0)) + '"><svg viewBox="0 0 120 70"><path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="#2c3a34" stroke-width="3"/><line x1="60" y1="60" x2="' + (60 + 42 * Math.cos((ang - 90) * Math.PI / 180)).toFixed(1) + '" y2="' + (60 + 42 * Math.sin((ang - 90) * Math.PI / 180)).toFixed(1) + '" stroke="#4ade80" stroke-width="2"/><circle cx="60" cy="60" r="3" fill="#4ade80"/></svg><span class="inst-v">' + Math.round(c * 100) + '%</span></div>'; },
      sum: function (d) { var m = (d && d.metrics) || []; return m.length ? '<b>' + Math.round((m[0].confidence || 0) * 100) + '%</b> confident · n=' + fmtInt(m[0].n) : 'collecting'; } },

    { key: 'narrative', title: 'Narrative Feed', metaphor: 'Living Newspaper',
      art: function (d) {
        var h = (d && d.headlines) || []; if (!h.length) return calib('no headlines');
        return '<div class="art-news">' + h.slice(0, 3).map(function (x) { return '<div class="news-line ' + (x.kind || 'info') + '" data-tip="' + dtip((x.kind || 'info').toUpperCase(), x.text) + '">' + esc(x.text) + '</div>'; }).join('') + '</div>'; },
      sum: function (d) { var h = (d && d.headlines) || []; return h.length ? '<b>' + h.length + '</b> live headlines' : 'quiet'; } },
  ];

  // ── bespoke EXPANDED ROOMS ────────────────────────────────────────────────────
  // One scene per artifact, built to its metaphor with the card's REAL data. Built &
  // locked one at a time. ROOMS[key] → html string; roomBody() dispatches here, falling
  // back to the generic detail list for rooms not yet built. Hover tooltips inside the
  // lightbox use [data-tip] (wired in openCard, since the ring delegation can't reach the modal).
  var ROOMS = {};

  // R01 — Change Detection · THE LAKE. The traffic series is the lakebed; every
  // significant change (≥±25%) drops a stone whose shockwave ripples across the whole
  // surface. Surge = green stone, drop = crimson. Hover a shockwave → time + Δ + from→to.
  ROOMS.change_detection = function (c) {
    var d = c.data || {};
    var series = d.series || [], changes = d.changes || [];
    if (!series.length) return '<div class="room-lake">' + calib('no traffic series yet — the lake is dry') + '</div>';
    var n = series.length, max = series.reduce(function (a, b) { return Math.max(a, b.c || 0); }, 1);
    var idxOf = {}; series.forEach(function (s, i) { idxOf[s.t] = i; });
    var X = function (i) { return 60 + (n > 1 ? i / (n - 1) : 0.5) * 880; };
    // lakebed: inverted area of the series, sitting under the water surface (y=120)
    var pts = series.map(function (s, i) { return [X(i), 150 + 190 * ((s.c || 0) / max)]; });
    var bed = 'M60 150 ' + pts.map(function (p) { return 'L' + p[0].toFixed(0) + ' ' + p[1].toFixed(0); }).join(' ') + ' L940 150 Z';
    // calm surface: layered horizontal waves drifting left
    var waves = [0, 1, 2, 3].map(function (i) {
      var y = 118 + i * 5;
      return '<path class="lk-wave" style="--d:' + (i * 0.6).toFixed(1) + 's" d="M-40 ' + y + ' q 60 -7 120 0 t 120 0 t 120 0 t 120 0 t 120 0 t 120 0 t 120 0 t 120 0 t 120 0" fill="none" stroke="#7da7ff" stroke-width="1" opacity="' + (0.5 - i * 0.1).toFixed(2) + '"/>';
    }).join('');
    // shockwaves: concentric expanding rings at each change's time-x
    var shocks = changes.map(function (ch, k) {
      var i = idxOf[ch.t] != null ? idxOf[ch.t] : (n - 1), x = X(i).toFixed(0);
      var up = ch.pct > 0, mag = Math.min(1, Math.abs(ch.pct) / 150);
      var rMax = (34 + mag * 130).toFixed(0), col = up ? '#4ade80' : '#e0584a';
      var rings = [0, 1, 2].map(function (r) {
        return '<circle class="lk-ring" cx="' + x + '" cy="120" r="6" fill="none" stroke="' + col + '" stroke-width="2" style="--rmax:' + rMax + 'px;--dly:' + (k * 0.2 + r * 0.55).toFixed(2) + 's"/>';
      }).join('');
      var splash = (up ? -1 : 1) * (18 + mag * 40);
      return '<g class="lk-shock" data-tip="' + dtip((up ? '+' : '') + ch.pct + '%', tail(ch.t) + ' · ' + fmtInt(ch.from) + '→' + fmtInt(ch.to)) + '">'
        + '<line x1="' + x + '" y1="120" x2="' + x + '" y2="' + (120 + splash).toFixed(0) + '" stroke="' + col + '" stroke-width="1.5" opacity="0.55"/>'
        + rings + '<circle class="lk-stone" cx="' + x + '" cy="120" r="4.5" fill="' + col + '"/></g>';
    }).join('');
    var b = d.biggest;
    var head = '<div class="lk-head"><div class="lk-title"><h4>The Lake</h4><span>every change ≥ ±25% drops a stone — the whole surface ripples</span></div>'
      + (b ? '<div class="lk-biggest ' + (b.pct > 0 ? 'up' : 'down') + '"><b>' + (b.pct > 0 ? '+' : '') + b.pct + '%</b><span>' + esc(tail(b.t)) + ' · ' + fmtInt(b.from) + '→' + fmtInt(b.to) + '</span></div>'
            : '<div class="lk-biggest calm"><b>still</b><span>no stone dropped</span></div>') + '</div>';
    var svg = '<svg viewBox="0 0 1000 360" class="lk-svg" preserveAspectRatio="none">'
      + '<defs><linearGradient id="lkwater" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#15273f"/><stop offset="1" stop-color="#091321"/></linearGradient></defs>'
      + '<rect x="0" y="120" width="1000" height="240" fill="url(#lkwater)"/>'
      + '<path d="' + bed + '" fill="#7da7ff" opacity="0.13"/>'
      + '<line x1="0" y1="120" x2="1000" y2="120" stroke="#7da7ff" stroke-width="0.6" opacity="0.3"/>'
      + waves + shocks + '</svg>';
    var list = changes.length
      ? '<div class="lk-list">' + changes.map(function (ch) {
          var up = ch.pct > 0;
          return '<div class="lk-row ' + (up ? 'up' : 'down') + '" data-tip="' + dtip((up ? '+' : '') + ch.pct + '%', esc(tail(ch.t))) + '"><span class="lk-dot"></span><b>' + (up ? '+' : '') + ch.pct + '%</b><small>' + esc(tail(ch.t)) + '</small><i>' + fmtInt(ch.from) + ' → ' + fmtInt(ch.to) + '</i></div>';
        }).join('') + '</div>'
      : '<div class="lk-still">Still water — no change crossed the ±25% threshold this window.</div>';
    return '<div class="room-lake">' + head + svg + list + '</div>';
  };

  // ── deck instance ────────────────────────────────────────────────────────────
  function create(host, dataset) {
    // ── COVERFLOW model: cards fan left/right, near-flat → reliable hit-testing.
    // A steep 3D cylinder rotated side-cards edge-on and broke pointer events
    // (elementFromPoint returned null). Coverflow keeps every visible card
    // hoverable / clickable / grip-draggable, and matches the reference fan.
    var N = ART.length;
    var SPACING = 188;      // px between adjacent card centres
    var DEPTH = 120;        // z pushback per step
    var TILT = 26, MAXTILT = 52;   // rotateY per step (deg), capped
    var IDLE = 0.0017;      // gentle auto-advance (slots/frame) — keeps looping
    var pos = 0, vel = 0;   // continuous slot position
    var pressing = false, dragging = false, startX = 0, lastX = 0;
    var downCardEl = null, downPinI = -1, overDeck = false;
    var raf = null, disposed = false, paused = false;
    var tipEl = null, lastTipEl = null;     // [F3] per-artifact hover tooltip
    var cards = [], order = [], pinned = [], rearrange = null;
    var ringEl;

    host.innerHTML = '<div class="deck-wrap">'
      + '<div class="deck-head"><span class="deck-rule"></span><h3>Traffic Intelligence Deck</h3>'
      + '<span class="deck-sub">' + (dataset === 'website' ? 'website telemetry' : 'API telemetry') + ' · 18 artifacts</span><span class="deck-rule"></span></div>'
      + '<div class="deck-pinned" id="dk-pinned"><div class="dk-pin-empty">Pin cards here to compare — hovering one syncs the others to the same moment</div></div>'
      + '<div class="deck-viewport" id="dk-vp"><div class="deck-ring" id="dk-ring"></div></div>'
      + '<div class="deck-nav"><button class="dk-arrow" data-d="-1" aria-label="previous">◄</button>'
      + '<span class="dk-counter" id="dk-counter">Card 01 of 18</span>'
      + '<button class="dk-arrow" data-d="1" aria-label="next">►</button>'
      + '<span class="dk-hint">drag · scroll · grip ⠿ to reorder · click a tile</span></div></div>';
    ringEl = host.querySelector('#dk-ring');

    // ── [F1] instant-cache + live refresh ──────────────────────────────────────
    // The deck_* aggregation is heavy (~4s cold). Rather than make the user stare
    // at a blank ring on every reload-after-inactivity, we paint instantly from the
    // last good bundle in sessionStorage, then silently refetch + swap content in
    // place. A 45s background poll (skipped on hidden tabs / mid-interaction) plus
    // the SSE react() keep the deck live WITHOUT any manual refresh.
    var CACHE_KEY = 'admDeck:' + dataset, liveTimer = null, built = false;
    var openI = -1, roomArtEl = null;
    function readCache() { try { var s = sessionStorage.getItem(CACHE_KEY); return s ? JSON.parse(s) : null; } catch (e) { return null; } }
    function writeCache(b) { try { if (b && b.cards) sessionStorage.setItem(CACHE_KEY, JSON.stringify(b)); } catch (e) {} }

    function buildFrom(bundle) {
      var data = (bundle && bundle.cards) || {};
      ART.forEach(function (a, i) {
        var el = document.createElement('div'); el.className = 'deck-card'; el.setAttribute('data-i', i);
        var cd = data[a.key] || {};
        el.innerHTML = cardHTML(a, cd, i);
        ringEl.appendChild(el);
        order.push(i);
        cards.push({ a: a, el: el, data: cd, inner: el.querySelector('.deck-card-inner'), slot: i });
      });
      reslot();
      wire();
      loop();
      loadLayout();          // [F2] apply this admin's saved order + pins
      built = true;
    }

    // in-place content swap — preserves order, position, pins, open room, listeners
    function refreshCards(bundle) {
      if (!built) return;
      var data = (bundle && bundle.cards) || {};
      cards.forEach(function (c) {
        var cd = data[c.a.key]; if (cd == null) return;
        c.data = cd;
        var art = c.el.querySelector('.dkc-art'); if (art) art.innerHTML = c.a.art(cd);
        var sum = c.el.querySelector('.dkc-sum'); if (sum) sum.innerHTML = c.a.sum(cd);
      });
      if (pinned.length) renderPinned();
      if (openI >= 0 && roomArtEl) { try { roomArtEl.innerHTML = cards[openI].a.art(cards[openI].data); } catch (e) {} }
    }

    function loadFresh(cb) {
      fetchDeck(dataset).then(function (bundle) {
        if (disposed || !bundle || !bundle.cards) return;
        writeCache(bundle);
        cb(bundle);
      }).catch(function () {});
    }

    var _cachedBundle = readCache();
    if (_cachedBundle && _cachedBundle.cards) {
      buildFrom(_cachedBundle);                          // instant paint from last good bundle
      loadFresh(function (b) { refreshCards(b); });       // then quietly catch up to live
    } else {
      loadFresh(function (b) { if (!built) buildFrom(b); }); // cold first run
    }

    function startLive() {
      if (liveTimer) clearInterval(liveTimer);
      liveTimer = setInterval(function () {
        if (disposed || !built) return;
        if (document.hidden) return;                       // never poll a hidden tab
        var vp = host.querySelector('#dk-vp'); if (vp && vp.offsetParent === null) return; // pane hidden
        if (pressing || dragging || rearrange) return;     // don't yank an active interaction
        loadFresh(function (b) { refreshCards(b); });
      }, 45000);
    }
    startLive();

    function reslot() { order.forEach(function (ci, s) { cards[ci].slot = s; }); }

    // [F3] floating hover tooltip (one element, reused across all artifacts)
    function showTip(html, x, y) {
      if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'dk-arttip'; document.body.appendChild(tipEl); }
      tipEl.innerHTML = html;
      tipEl.style.display = 'block';
      var w = tipEl.offsetWidth, h = tipEl.offsetHeight;
      var L = Math.min(window.innerWidth - w - 8, Math.max(8, x + 14));
      var T = Math.max(8, y - h - 12);
      tipEl.style.left = L + 'px'; tipEl.style.top = T + 'px';
    }
    function hideTip() {
      if (tipEl) tipEl.style.display = 'none';
      ringEl.querySelectorAll('.dk-tip-on').forEach(function (e) { e.classList.remove('dk-tip-on'); });
    }

    // ── [F2] persisted layout (order + pins), per admin account ──────────────────
    function _csrf() { return (document.querySelector('meta[name=csrf-token]') || {}).content || ''; }
    function loadLayout() {
      var url = '/api/account/admin/traffic/deck-layout?dataset=' + dataset;
      var p = A.adminFetch ? A.adminFetch(url)
        : fetch(url, { credentials: 'include', headers: { 'X-Console-Csrf': _csrf() } })
            .then(function (r) { return r.json(); }).then(function (j) { return j.data || j; });
      p.then(function (lay) {
        if (disposed || !lay) return;
        if (Array.isArray(lay.order) && lay.order.length === N) {
          var ok = lay.order.slice().sort(function (a, b) { return a - b; }).every(function (v, i) { return v === i; });
          if (ok) { order = lay.order.slice(); reslot(); pos = Math.round(pos); }
        }
        if (Array.isArray(lay.pins) && lay.pins.length) {
          pinned = lay.pins.filter(function (i) { return i >= 0 && i < N; }).slice(0, 4);
          renderPinned();
        }
      }).catch(function () {});
    }
    var _saveT = null;
    function saveLayout() {
      if (_saveT) clearTimeout(_saveT);
      _saveT = setTimeout(function () {
        try {
          fetch('/api/account/admin/traffic/deck-layout', {
            method: 'POST', credentials: 'include',
            headers: { 'Content-Type': 'application/json', 'X-Console-Csrf': _csrf() },
            body: JSON.stringify({ dataset: dataset, order: order, pins: pinned })
          }).catch(function () {});
        } catch (e) {}
      }, 350);
    }
    function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }

    function cardHTML(a, cd, i) {
      return '<div class="deck-card-inner" tabindex="0">'
        + '<div class="dkc-h"><span class="dkc-grip" data-i="' + i + '" title="Drag to reorder">⠿</span>'
        + '<span class="dkc-n">' + String(i + 1).padStart(2, '0') + '</span><span class="dkc-t">' + esc(a.title) + '</span>'
        + '<button class="dkc-pin" data-i="' + i + '" title="Pin to compare">⊕</button></div>'
        + '<div class="dkc-meta">' + esc(a.metaphor) + '</div>'
        + '<div class="dkc-art" data-tip="' + dtip(a.title, a.metaphor) + '">' + a.art(cd) + '</div>'
        + '<div class="dkc-sum">' + a.sum(cd) + '</div></div>';
    }

    function frontSlot() { var s = Math.round(pos) % N; return ((s % N) + N) % N; }
    function loop() {
      if (disposed) return;
      raf = requestAnimationFrame(loop);
      var vp = host.querySelector('#dk-vp');
      if (vp && vp.offsetParent === null) return;       // pane hidden — idle
      if (!dragging && !rearrange && !paused) {
        if (Math.abs(vel) >= 0.0006) { pos += vel; vel *= 0.90; if (Math.abs(vel) < 0.0006) { vel = 0; pos = Math.round(pos); } }
        else if (!overDeck && !pressing) { pos += IDLE; } // drifts only when the pointer is away — holds still to interact
      }
      var fs = frontSlot();
      cards.forEach(function (c) {
        var off = c.slot - pos; off = off - N * Math.round(off / N);   // nearest wrapped copy
        var ao = Math.abs(off);
        if (ao > 4.6) { c.el.style.visibility = 'hidden'; c.el.style.pointerEvents = 'none'; return; }
        c.el.style.visibility = 'visible';
        var x = (off < 0 ? -1 : 1) * Math.pow(ao, 0.92) * SPACING;
        var z = -ao * DEPTH;
        var ry = clamp(-off * TILT, -MAXTILT, MAXTILT);
        var sc = Math.max(0.74, 1 - ao * 0.07);
        var op = ao <= 0.5 ? 1 : Math.max(0.22, 1 - ao * 0.17);
        c.el.style.transform = 'translate(-50%,-50%) translateX(' + x.toFixed(1) + 'px) translateZ(' + z.toFixed(1) + 'px) rotateY(' + ry.toFixed(1) + 'deg) scale(' + sc.toFixed(3) + ')';
        c.el.style.opacity = op.toFixed(2);
        c.el.style.zIndex = String(1000 - Math.round(ao * 10));
        c.el.style.pointerEvents = ao <= 3.4 ? 'auto' : 'none';
        c.el.classList.toggle('is-front', Math.round(off) === 0);
        c.inner.style.filter = ao < 0.5 ? 'none' : ('brightness(' + (1 - ao * 0.06).toFixed(2) + ')');
      });
      var ctr = host.querySelector('#dk-counter');
      if (ctr) ctr.textContent = 'Card ' + String(order[fs] + 1).padStart(2, '0') + ' of 18';
    }

    function snap(dir) { vel = 0; pos = Math.round(pos) + dir; }
    function wire() {
      var vp = host.querySelector('#dk-vp');
      // pointer over the deck freezes the idle drift so targets hold still
      vp.addEventListener('pointerenter', function () { overDeck = true; });
      vp.addEventListener('pointerleave', function () { overDeck = false; });
      // press: remember WHAT was pressed (the card/pin under the finger). The tap
      // is resolved on pointerup from this captured target — NOT the native click,
      // which the browser cancels whenever the deck moves between down and up.
      vp.addEventListener('pointerdown', function (e) {
        if (e.target.closest && e.target.closest('.dkc-grip')) return;   // grip → ring handles rearrange
        pressing = true; dragging = false; startX = e.clientX; lastX = e.clientX; vel = 0;
        downPinI = -1; downCardEl = null;
        var pin = e.target.closest && e.target.closest('.dkc-pin');
        if (pin) { downPinI = +pin.getAttribute('data-i'); return; }
        downCardEl = e.target.closest && e.target.closest('.deck-card');
      });
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
      vp.addEventListener('wheel', function (e) { e.preventDefault(); vel += e.deltaY * 0.0009; }, { passive: false });
      host.querySelectorAll('.dk-arrow').forEach(function (b) { b.addEventListener('click', function () { snap(+b.getAttribute('data-d')); }); });
      // hover amplifies the artifact
      ringEl.addEventListener('pointerover', function (e) { var ci = e.target.closest && e.target.closest('.deck-card-inner'); if (ci) ci.classList.add('hot'); });
      ringEl.addEventListener('pointerout', function (e) { var ci = e.target.closest && e.target.closest('.deck-card-inner'); if (ci && !ci.contains(e.relatedTarget)) ci.classList.remove('hot'); });
      // [F3] per-artifact hover micro-tooltip — any element carrying data-tip
      ringEl.addEventListener('mousemove', function (e) {
        var t = e.target.closest && e.target.closest('[data-tip]');
        if (t) { showTip(t.getAttribute('data-tip'), e.clientX, e.clientY); t.classList.add('dk-tip-on'); }
        else hideTip();
      });
      ringEl.addEventListener('mouseleave', function () { hideTip(); });
      // grip → start rearrange
      ringEl.addEventListener('pointerdown', function (e) {
        var grip = e.target.closest && e.target.closest('.dkc-grip');
        if (grip) { e.preventDefault(); e.stopPropagation(); startRearrange(+grip.getAttribute('data-i'), e); }
      });
    }
    function onMove(e) {
      if (!pressing) return;
      if (!dragging) {
        if (Math.abs(e.clientX - startX) <= 5) return;   // below threshold → still a tap
        dragging = true; host.querySelector('#dk-vp').classList.add('grabbing');
      }
      var dx = e.clientX - lastX; pos -= dx * 0.011; vel = -dx * 0.011; lastX = e.clientX;
    }
    function onUp() {
      if (!pressing) return;
      pressing = false;
      if (dragging) { dragging = false; host.querySelector('#dk-vp').classList.remove('grabbing'); return; }  // was a spin
      if (rearrange) return;
      // TAP — resolve from the captured down-target (robust against deck motion)
      if (downPinI >= 0) { pinCard(downPinI); return; }
      if (downCardEl) {
        var ci = +downCardEl.getAttribute('data-i');
        if (downCardEl.classList.contains('is-front')) openCard(ci);
        else { vel = 0; pos = cards[ci].slot; }          // a side tile rides to the front
      }
    }

    // ── drag-to-rearrange (grip handle → floating ghost → slot swap) ───────────────
    function startRearrange(ci, ev) {
      var src = cards[ci];
      var ghost = src.el.querySelector('.deck-card-inner').cloneNode(true);
      ghost.className = 'deck-card-inner dk-ghost';
      document.body.appendChild(ghost);
      src.el.classList.add('dk-dragging');
      rearrange = { ci: ci, ghost: ghost, overCi: null };
      moveGhost(ev);
      window.addEventListener('pointermove', dragRearrange);
      window.addEventListener('pointerup', dropRearrange);
    }
    function moveGhost(ev) { var g = rearrange.ghost; g.style.left = ev.clientX + 'px'; g.style.top = ev.clientY + 'px'; }
    function dragRearrange(ev) {
      if (!rearrange) return;
      moveGhost(ev);
      rearrange.ghost.style.pointerEvents = 'none';
      var under = document.elementFromPoint(ev.clientX, ev.clientY);
      var card = under && under.closest && under.closest('.deck-card');
      cards.forEach(function (c) { c.el.classList.remove('dk-droptarget'); });
      if (card) {
        var ti = +card.getAttribute('data-i');
        if (ti !== rearrange.ci) { card.classList.add('dk-droptarget'); rearrange.overCi = ti; }
        else rearrange.overCi = null;
      } else rearrange.overCi = null;
    }
    function dropRearrange() {
      window.removeEventListener('pointermove', dragRearrange);
      window.removeEventListener('pointerup', dropRearrange);
      if (!rearrange) return;
      var r = rearrange;
      if (r.overCi != null && r.overCi !== r.ci) {
        var sa = cards[r.ci].slot, sb = cards[r.overCi].slot;     // swap the two cards' slots
        order[sa] = r.overCi; order[sb] = r.ci;
        reslot();
        pos = Math.round(pos);
        saveLayout();                                              // [F2] persist new order
      }
      cards.forEach(function (c) { c.el.classList.remove('dk-droptarget'); });
      cards[r.ci].el.classList.remove('dk-dragging');
      if (r.ghost && r.ghost.parentNode) r.ghost.parentNode.removeChild(r.ghost);
      rearrange = null;
    }

    // ── live-data reaction: only the affected cards animate, everything else calm ──
    function reactCard(key, cls) {
      var c = null;
      for (var k = 0; k < cards.length; k++) { if (cards[k].a.key === key) { c = cards[k]; break; } }
      if (!c || !c.inner) return;
      c.inner.classList.remove(cls, 'dk-rx');
      void c.inner.offsetWidth;                 // restart the one-shot animation
      c.inner.classList.add(cls, 'dk-rx');
      setTimeout(function () { c.inner.classList.remove(cls, 'dk-rx'); }, 1200);
    }
    function react(ev) {
      if (!ev || ev.type !== 'request' || !cards.length) return;
      if ((ev.status || 0) >= 400) {            // a new error ripples through three cards only
        reactCard('error_fingerprints', 'dk-rx-crack');
        reactCard('regressions', 'dk-rx-fracture');
        reactCard('narrative', 'dk-rx-headline');
      } else {                                  // a healthy hit nudges change-detection alone
        reactCard('change_detection', 'dk-rx-ripple');
      }
    }

    // ── Pinned Canvas ────────────────────────────────────────────────────────────
    function pinCard(i) {
      if (pinned.indexOf(i) >= 0) return;
      if (pinned.length >= 4) { pinned.shift(); }
      pinned.push(i);
      renderPinned();
      saveLayout();                                                // [F2] persist pins
    }
    function renderPinned() {
      var host2 = host.querySelector('#dk-pinned');
      if (!pinned.length) { host2.innerHTML = '<div class="dk-pin-empty">Pin cards here to compare — hovering one syncs the others to the same moment</div>'; return; }
      host2.innerHTML = pinned.map(function (i) {
        var c = cards[i]; return '<div class="dk-pin-card" data-i="' + i + '"><div class="dkp-h"><span>' + esc(c.a.title) + '</span><button class="dkp-x" data-i="' + i + '">×</button></div>'
          + '<div class="dkp-art">' + c.a.art(c.data) + '</div></div>';
      }).join('');
      host2.querySelectorAll('.dkp-x').forEach(function (b) { b.addEventListener('click', function () { pinned = pinned.filter(function (x) { return x !== +b.getAttribute('data-i'); }); renderPinned(); saveLayout(); }); });
      // synced hover: hovering a pinned spark broadcasts the x-fraction to siblings
      host2.querySelectorAll('.dk-pin-card').forEach(function (pc) {
        pc.addEventListener('mousemove', function (e) {
          var sp = pc.querySelector('.dk-spark'); if (!sp) return;
          var r = sp.getBoundingClientRect(); var f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
          host2.querySelectorAll('.dk-pin-card').forEach(function (o) { var s = o.querySelector('.dk-spark'); if (!s) return; var cur = s.querySelector('.dkp-cursor') || (function () { var c = document.createElementNS('http://www.w3.org/2000/svg', 'line'); c.setAttribute('class', 'dkp-cursor'); c.setAttribute('stroke', '#fff'); c.setAttribute('stroke-width', '0.8'); c.setAttribute('opacity', '0.6'); s.appendChild(c); return c; })(); var x = (f * (s.viewBox.baseVal.width || 240)).toFixed(1); cur.setAttribute('x1', x); cur.setAttribute('x2', x); cur.setAttribute('y1', 0); cur.setAttribute('y2', s.viewBox.baseVal.height || 40); });
        });
        pc.addEventListener('mouseleave', function () { host2.querySelectorAll('.dkp-cursor').forEach(function (c) { c.remove(); }); });
      });
    }

    // ── expanded room ─────────────────────────────────────────────────────────────
    function openCard(i) {
      if (document.querySelector('.dk-lb')) return;     // a room is already open — never stack lightboxes
      var c = cards[i];
      var artHTML = '', bodyHTML = '';
      try { artHTML = c.a.art(c.data); } catch (e) { artHTML = ''; }
      try { bodyHTML = roomBody(c); } catch (e) { bodyHTML = '<div class="dk-room-note">' + esc(c.a.metaphor) + '</div>'; }
      var back = document.createElement('div'); back.className = 'dk-lb-back';
      var lb = document.createElement('div'); lb.className = 'dk-lb';
      lb.innerHTML = '<div class="dk-lb-h"><span class="dk-lb-n">' + String(i + 1).padStart(2, '0') + ' / 18</span><b>' + esc(c.a.title) + '</b><span class="dk-lb-meta">' + esc(c.a.metaphor) + '</span><button class="dk-lb-x">×</button></div>'
        + '<div class="dk-lb-art">' + artHTML + '</div>'
        + '<div class="dk-lb-body">' + bodyHTML + '</div>';
      document.body.appendChild(back); document.body.appendChild(lb);
      paused = true;                                    // deck pauses while a card is unfolded
      openI = i; roomArtEl = lb.querySelector('.dk-lb-art');   // [F1] live-refresh target while open
      requestAnimationFrame(function () { back.classList.add('on'); lb.classList.add('on'); });
      // room hover tooltips — the ring's mousemove delegation can't reach the modal,
      // so wire the floating [data-tip] tip on the lightbox itself.
      lb.addEventListener('mousemove', function (e) {
        var t = e.target.closest && e.target.closest('[data-tip]');
        if (t) showTip(t.getAttribute('data-tip'), e.clientX, e.clientY); else hideTip();
      });
      lb.addEventListener('mouseleave', hideTip);
      function close() { paused = false; openI = -1; roomArtEl = null; hideTip(); back.classList.remove('on'); lb.classList.remove('on'); setTimeout(function () { back.remove(); lb.remove(); }, 240); }
      back.addEventListener('click', close); lb.querySelector('.dk-lb-x').addEventListener('click', close);
      document.addEventListener('keydown', function ek(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', ek); } });
    }
    // a first-pass "room": the card's real data laid out as a detail list. Deepens
    // into bespoke laboratories (forensic room, watchmaker table…) in later waves.
    function roomBody(c) {
      // bespoke room if built; otherwise the generic detail list (fallback)
      if (ROOMS[c.a.key]) { try { return ROOMS[c.a.key](c); } catch (e) {} }
      var d = c.data || {};
      var rows = [];
      (d.items || d.changes || d.segments || d.routes || d.headlines || d.metrics || []).slice(0, 12).forEach(function (x) {
        var label = x.path || x.label || x.text || (x.from ? tail(x.from) + ' → ' + tail(x.to) : (x.t || ''));
        var val = x.count != null ? fmtInt(x.count) : (x.n != null ? fmtInt(x.n) : (x.error_rate != null ? x.error_rate + '%' : (x.weight != null ? '×' + x.weight : (x.pct != null ? x.pct + '%' : (x.confidence != null ? Math.round(x.confidence * 100) + '%' : '')))));
        rows.push('<div class="dk-room-row"><span>' + esc(label) + '</span><b>' + esc(val) + '</b></div>');
      });
      if (!rows.length && d.state) rows.push('<div class="dk-room-row dim"><span>' + esc(d.reason || 'calibrating') + '</span><b>—</b></div>');
      return '<div class="dk-room-note">' + esc(c.a.metaphor) + ' — full ' + ({ change_detection: 'lake', regressions: 'incident mirror', error_fingerprints: 'investigation room', latency_attribution: 'watchmaker table', alert_simulator: 'mission control' }[c.a.key] || 'observatory') + ' lands in an upcoming wave.</div>'
        + '<div class="dk-room-list">' + rows.join('') + '</div>';
    }

    return {
      react: react,
      dispose: function () {
        disposed = true; if (raf) cancelAnimationFrame(raf); if (liveTimer) clearInterval(liveTimer);
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        window.removeEventListener('pointermove', dragRearrange);
        window.removeEventListener('pointerup', dropRearrange);
        if (rearrange && rearrange.ghost && rearrange.ghost.parentNode) rearrange.ghost.parentNode.removeChild(rearrange.ghost);
        if (tipEl && tipEl.parentNode) tipEl.parentNode.removeChild(tipEl);
      }
    };
  }

  window.ADM_DECK = { create: create, count: ART.length };
})();
