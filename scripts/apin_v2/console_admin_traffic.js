/* ADM-T · Traffic section (API + Website) — renders the live aggregation data
 * layer (/api/account/admin/traffic) into the admin glass aesthetic.
 * Loaded separately from console_admin.js; uses the window.ADM bridge for the
 * shared helpers (slot-machine numbers, esc, fmtInt, relTime, gated fetch,
 * drill drawers). Self-disposing on section change.
 */
(function () {
  'use strict';
  var A = window.ADM || {};
  var esc = A.esc || function (s) { return String(s == null ? '' : s); };
  var fmtInt = A.fmtInt || function (n) { return String(n); };
  var fmtBytes = A.fmtBytes || function (n) { return n + ' B'; };
  var relTime = A.relTime || function (s) { return s; };
  var animateNum = A.animateNum || function (el, n) { if (el) el.textContent = fmtInt(n); };
  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var st = { sub: 'api', window: 'all', wired: false, mounted: false,
             renderedKey: {}, live: null, lastData: {}, terrain: null,
             galaxy: null, sse: null, sseStats: null, decks: {} };

  function disposeDecks() { Object.keys(st.decks).forEach(function (k) { try { st.decks[k].dispose(); } catch (e) {} }); st.decks = {}; }
  function mountDeck(pane, hostId, dataset) {
    var el = pane.querySelector('#' + hostId); if (!el) return;
    if (st.decks[dataset]) { try { st.decks[dataset].dispose(); } catch (e) {} }
    if (window.ADM_DECK) st.decks[dataset] = window.ADM_DECK.create(el, dataset);
    else el.innerHTML = '<div class="trf-empty">stats deck module unavailable</div>';
  }
  var cache = {};

  // ── shared floating tooltip (every chart is hoverable) ─────────────────────
  var _tip = null;
  function tip(html, x, y) {
    if (!_tip) {
      _tip = document.createElement('div'); _tip.className = 'trf-tip';
      document.body.appendChild(_tip);
    }
    _tip.innerHTML = html; _tip.style.display = 'block';
    var r = _tip.getBoundingClientRect();
    _tip.style.left = Math.min(window.innerWidth - r.width - 8, Math.max(8, x - r.width / 2)) + 'px';
    _tip.style.top = Math.max(8, y - r.height - 12) + 'px';
  }
  function hideTip() { if (_tip) _tip.style.display = 'none'; }

  function fetchWidget(widget, extra) {
    var url = '/api/account/admin/traffic?widget=' + widget + '&window=' + st.window + (extra || '');
    if (A.adminFetch) return A.adminFetch(url);
    var csrf = (document.querySelector('meta[name=csrf-token]') || {}).content || '';
    return fetch(url, { credentials: 'include', headers: { 'X-Console-Csrf': csrf } })
      .then(function (r) { return r.json(); }).then(function (j) { return j.data || j; })
      .catch(function () { return null; });
  }

  // ── small visual primitives ───────────────────────────────────────────────
  function num(v, cls) {
    return '<span class="trf-num ' + (cls || '') + '" data-num="0">0</span>';
  }
  function rollAll(root) {
    [].forEach.call(root.querySelectorAll('.trf-num[data-target]'), function (el) {
      animateNum(el, +el.getAttribute('data-target') || 0);
    });
  }
  function setRoll(root, id, value) {
    var el = root.querySelector('#' + id);
    if (el) { el.setAttribute('data-target', value); }
  }

  function kpi(id, label, opts) {
    opts = opts || {};
    return '<div class="trf-kpi' + (opts.drill ? ' clickable' : '') + (opts.tone ? ' ' + opts.tone : '') + '"'
      + (opts.drill ? ' data-drill="' + opts.drill + '" role="button" tabindex="0"' : '')
      + (opts.title ? ' title="' + esc(opts.title) + '"' : '') + '>'
      + '<span class="l">' + esc(label) + '</span>'
      + '<span class="v">' + (opts.raw != null ? opts.raw : '<span class="trf-num" id="' + id + '" data-num="0">0</span>' + (opts.suffix || '')) + '</span>'
      + (opts.sub ? '<span class="s">' + esc(opts.sub) + '</span>' : '')
      + (opts.drill ? '<span class="go">↗</span>' : '') + '</div>';
  }

  // area sparkline (series of {t,c})
  function sparkline(series, w, h) {
    series = series || []; w = w || 760; h = h || 130;
    if (!series.length) return '<div class="trf-empty">no data in window</div>';
    var max = series.reduce(function (a, b) { return Math.max(a, b.c || 0); }, 0) || 1;
    var n = series.length, dx = w / (n - 1 || 1);
    var pts = series.map(function (s, i) {
      var x = i * dx, y = h - 6 - (h - 14) * (s.c / max);
      return [x, y];
    });
    var path = pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ');
    var area = 'M0 ' + h + ' ' + path.replace('M', 'L') + ' L' + w + ' ' + h + ' Z';
    var dots = pts.map(function (p, i) {
      return '<circle class="sp-dot" cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3" data-i="' + i + '"></circle>';
    }).join('');
    return '<svg class="trf-spark" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">'
      + '<defs><linearGradient id="spgrad" x1="0" y1="0" x2="0" y2="1">'
      + '<stop offset="0" stop-color="var(--accent)" stop-opacity=".34"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>'
      + '<path class="sp-area" d="' + area + '" fill="url(#spgrad)"/>'
      + '<path class="sp-line" d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2"/>'
      + dots + '</svg>';
  }

  // donut (segments of {label,value})
  var SEG_COL = ['var(--accent)', 'var(--ochre)', 'var(--teal)', 'var(--crimson)', 'var(--violet)', 'var(--magenta)', '#7da7ff', '#b6e35a'];
  function donut(segs, centerLabel, centerVal) {
    segs = (segs || []).filter(function (s) { return (s.value || 0) > 0; });
    var total = segs.reduce(function (a, b) { return a + (b.value || 0); }, 0) || 1;
    var R = 52, C = 60, circ = 2 * Math.PI * R, off = 0;
    var arcs = segs.map(function (s, i) {
      var frac = (s.value || 0) / total, len = frac * circ;
      var seg = '<circle class="dn-seg" r="' + R + '" cx="' + C + '" cy="' + C + '" fill="none" '
        + 'stroke="' + SEG_COL[i % SEG_COL.length] + '" stroke-width="13" '
        + 'stroke-dasharray="' + len.toFixed(2) + ' ' + (circ - len).toFixed(2) + '" '
        + 'stroke-dashoffset="' + (-off).toFixed(2) + '" data-label="' + esc(s.label) + '" data-value="' + s.value + '" '
        + 'transform="rotate(-90 ' + C + ' ' + C + ')"></circle>';
      off += len; return seg;
    }).join('');
    var legend = segs.map(function (s, i) {
      return '<span class="dn-leg"><i style="background:' + SEG_COL[i % SEG_COL.length] + '"></i>'
        + esc(s.label) + ' <b>' + fmtInt(s.value) + '</b></span>';
    }).join('');
    return '<div class="trf-donut-wrap"><svg class="trf-donut" viewBox="0 0 120 120">' + arcs
      + '<text class="dn-cv" x="60" y="56" text-anchor="middle">' + esc(centerVal || total) + '</text>'
      + '<text class="dn-cl" x="60" y="74" text-anchor="middle">' + esc(centerLabel || 'total') + '</text>'
      + '</svg><div class="dn-legend">' + legend + '</div></div>';
  }

  // method × status hive (hexes)
  function hive(data) {
    var methods = (data && data.methods) || [], classes = (data && data.classes) || ['2xx', '3xx', '4xx', '5xx'];
    var cells = {}; ((data && data.cells) || []).forEach(function (c) { cells[c.method + '|' + c.cls] = c.value; });
    var max = 1; for (var k in cells) max = Math.max(max, cells[k]);
    var clsCol = { '2xx': 'var(--accent)', '3xx': 'var(--teal)', '4xx': 'var(--ochre)', '5xx': 'var(--crimson)' };
    var rows = methods.map(function (m) {
      var hexes = classes.map(function (cl) {
        var v = cells[m + '|' + cl] || 0, sc = v ? (0.4 + 0.6 * (v / max)) : 0;
        return '<div class="hv-cell' + (v ? '' : ' z') + '" data-m="' + esc(m) + '" data-cls="' + cl + '" data-v="' + v
          + '" title="' + esc(m) + ' ' + cl + ' · ' + fmtInt(v) + '" style="--c:' + clsCol[cl] + ';--s:' + sc.toFixed(2) + '">'
          + (v ? fmtInt(v) : '·') + '</div>';
      }).join('');
      return '<div class="hv-row"><span class="hv-m">' + esc(m) + '</span>' + hexes + '</div>';
    }).join('');
    var head = '<div class="hv-row hv-head"><span class="hv-m"></span>'
      + classes.map(function (c) { return '<span class="hv-h">' + c + '</span>'; }).join('') + '</div>';
    return '<div class="trf-hive">' + head + rows + '</div>';
  }

  // latency ridges (p50/p90/p99 bars per day)
  function ridges(rid) {
    rid = (rid || []).filter(function (r) { return r.n; });
    if (!rid.length) return '<div class="trf-empty">no latency samples in window</div>';
    var max = rid.reduce(function (a, b) { return Math.max(a, b.p99 || 0); }, 0) || 1;
    return '<div class="trf-ridges">' + rid.map(function (r) {
      function bar(p, cls, lab) {
        var w = Math.max(1, 100 * (p || 0) / max);
        return '<span class="rg-bar ' + cls + '" style="width:' + w.toFixed(1) + '%" title="' + lab + ' ' + (p || 0) + 'ms"></span>';
      }
      return '<div class="rg-row" data-t="' + esc(r.t) + '"><span class="rg-t">' + esc((r.t || '').slice(5)) + '</span>'
        + '<span class="rg-track">' + bar(r.p50, 'p50', 'p50') + bar(r.p90, 'p90', 'p90') + bar(r.p99, 'p99', 'p99') + '</span>'
        + '<span class="rg-v">' + fmtInt(r.p99 || 0) + 'ms</span></div>';
    }).join('') + '<div class="rg-key"><span class="p50">p50</span><span class="p90">p90</span><span class="p99">p99</span></div></div>';
  }

  // generic bar list ({label,value} + optional click)
  function barList(items, opts) {
    opts = opts || {}; items = items || [];
    if (!items.length) return '<div class="trf-empty">' + (opts.empty || 'no data') + '</div>';
    var max = items.reduce(function (a, b) { return Math.max(a, b.value || b.requests || 0); }, 0) || 1;
    return '<div class="trf-barlist">' + items.map(function (it) {
      var v = it.value != null ? it.value : (it.requests || 0);
      var sub = opts.sub ? opts.sub(it) : '';
      return '<div class="bl-row' + (opts.drill ? ' clickable' : '') + '"' + (opts.drill ? ' data-id="' + esc(opts.id ? opts.id(it) : '') + '" role="button" tabindex="0"' : '') + '>'
        + '<span class="bl-l">' + esc(opts.label ? opts.label(it) : it.label) + (sub ? '<small>' + esc(sub) + '</small>' : '') + '</span>'
        + '<span class="bl-bar"><i style="width:' + Math.round(100 * v / max) + '%' + (opts.barColor ? ';background:' + opts.barColor(it) : '') + '"></i></span>'
        + '<span class="bl-v">' + (opts.fmt ? opts.fmt(v, it) : fmtInt(v)) + '</span></div>';
    }).join('') + '</div>';
  }

  function card(span, label, body, expandKey, cardId) {
    return '<div class="trf-card col-' + span + '"' + (cardId ? ' data-card="' + cardId + '"' : '') + '>'
      + '<div class="trf-card-h"><span class="k-label">' + esc(label) + '</span>'
      + (expandKey ? '<button class="trf-x" data-exp="' + expandKey + '" aria-label="expand">⤢</button>' : '') + '</div>'
      + '<div class="trf-card-b">' + body + '</div></div>';
  }
  // progressive-render helpers: shell body + per-card fill (no blank wait)
  var LOADBODY = '<div class="trf-cardload"><span class="trf-spin"></span></div>';
  function cBody(pane, id) { var c = pane.querySelector('.trf-card[data-card="' + id + '"] .trf-card-b'); return c; }
  function cLabel(pane, id, txt) { var h = pane.querySelector('.trf-card[data-card="' + id + '"] .k-label'); if (h) h.textContent = txt; }

  // ── API tab ────────────────────────────────────────────────────────────────
  function renderApi(pane) {
    // PROGRESSIVE: paint the hero + ribbon + 6 card shells instantly (staggered
    // in), then fill each card the moment its own fetch resolves — no blank wait.
    var hero = '<div class="trf-hero" data-card="c-terrain">'
      + '<div class="trf-hero-h"><span class="k-label">traffic terrain · 7 days × 24 hours</span>'
      + '<span class="trf-hero-hint">drag to orbit · hover a peak</span>'
      + '<button class="trf-x" data-exp="api_terrain" aria-label="expand">⤢</button></div>'
      + '<div class="trf-hero-stage"><canvas class="trf-terrain-cv"></canvas>'
      + '<div class="trf-terrain-legend"><span class="cool">quiet</span><span class="bar"></span><span class="hot">peak</span>'
      + '<span class="sep"></span><i class="lg-jade"></i>clean<i class="lg-ember"></i>errors</div></div></div>';
    var ribbon = '<div class="trf-ribbon">'
      + kpi('k-req', 'requests', { drill: 'requests', title: 'All API requests in window' })
      + kpi('k-err', 'error rate', { raw: '<b class="trf-static" id="k-err-v">—</b>%', drill: 'errors' })
      + kpi('k-p95', 'p95 latency', { raw: '<span class="trf-num" id="k-p95" data-num="0">0</span>ms' })
      + kpi('k-thru', 'throughput', { raw: '<b class="trf-static" id="k-thru-v">—</b><small>/min</small>' })
      + kpi('k-bytes', 'bytes out', { raw: '<b class="trf-static" id="k-bytes-v">—</b>' })
      + '</div>';
    var grid = '<div class="trf-grid">'
      + card(8, 'requests over time', LOADBODY, 'api_series', 'c-series')
      + card(4, 'status mix', LOADBODY, 'api_status', 'c-status')
      + card(5, 'method × status', LOADBODY, 'api_methods', 'c-methods')
      + card(7, 'latency percentiles · per day', LOADBODY, 'api_latency', 'c-latency')
      + card(7, 'endpoints', LOADBODY, 'api_endpoints', 'c-endpoints')
      + card(5, 'top keys & origins', LOADBODY, 'api_top', 'c-top')
      + card(12, 'client origins · network truth', LOADBODY, '', 'c-origins')
      + card(12, 'endpoint galaxy · call sequences', LOADBODY, 'api_galaxy', 'c-galaxy')
      + '</div>';
    pane.innerHTML = hero + ribbon + grid + '<div class="trf-stats-sec" id="trf-stats-api"></div>';
    staggerIn(pane);
    wireApi(pane, {});
    mountTerrain(pane);
    mountGalaxy(pane);
    mountDeck(pane, 'trf-stats-api', 'api');

    fetchWidget('api_overview').then(function (ov) {
      ov = ov || {}; var lt = ov.latency || {};
      setRoll(pane, 'k-req', ov.requests || 0); setRoll(pane, 'k-p95', lt.p95 || 0); rollAll(pane);
      var ev = pane.querySelector('#k-err-v');
      if (ev) {
        ev.textContent = (ov.error_rate || 0);
        var kp = ev.closest('.trf-kpi'); if (kp) { kp.classList.toggle('bad', (ov.error_rate || 0) >= 20); kp.classList.toggle('warn', (ov.error_rate || 0) >= 5 && (ov.error_rate || 0) < 20); }
      }
      var tv = pane.querySelector('#k-thru-v'); if (tv) tv.textContent = (ov.throughput_per_min || 0);
      var bv = pane.querySelector('#k-bytes-v'); if (bv) bv.textContent = fmtBytes(ov.bytes_out || 0);
      var sc = cBody(pane, 'c-series');
      if (sc) { sc.innerHTML = sparkline(ov.series); var sp = sc.querySelector('.trf-spark'); if (sp) sp.__series = ov.series || []; cLabel(pane, 'c-series', ov.series_label || 'requests over time'); }
      var su = cBody(pane, 'c-status'); if (su) su.innerHTML = donut(ov.status_mix, 'requests', fmtInt(ov.requests || 0));
      wireHovers(pane);
    });
    fetchWidget('api_methods').then(function (ms) { var b = cBody(pane, 'c-methods'); if (b) { b.innerHTML = hive(ms || {}); wireHovers(pane); } });
    fetchWidget('api_latency').then(function (lat) { var b = cBody(pane, 'c-latency'); if (b) { b.innerHTML = ridges((lat || {}).ridges); wireHovers(pane); } });
    fetchWidget('api_endpoints').then(function (ep) { ep = ep || {}; var b = cBody(pane, 'c-endpoints'); if (b) { b.innerHTML = endpointTable(ep.endpoints); cLabel(pane, 'c-endpoints', 'endpoints · ' + ((ep.endpoints || []).length) + ' tracked'); wireHovers(pane); } });
    fetchWidget('api_top').then(function (top) { var b = cBody(pane, 'c-top'); if (b) { b.innerHTML = topBlock(top || {}); wireHovers(pane); } });
    fetchWidget('origins').then(function (o) { var b = cBody(pane, 'c-origins'); if (b) b.innerHTML = originsBlock(o || {}); });
  }

  // Honest client-origins panel: states plainly that all API traffic is from
  // the local/private network (not geolocatable), and points to Geography for
  // the real scan-origin globe. No fabricated map pins.
  function originsBlock(o) {
    var total = o.total || 0, priv = o.private || 0, pct = o.private_pct || 0;
    var buckets = (o.buckets || []).filter(function (b) { return b.count > 0; });
    var max = buckets.reduce(function (a, b) { return Math.max(a, b.count || 0); }, 0) || 1;
    var bars = buckets.map(function (b) {
      var col = b.label === 'public internet' ? 'var(--accent)' : 'var(--ochre)';
      return '<div class="bl-row"><span class="bl-l">' + esc(b.label) + '</span>'
        + '<span class="bl-bar"><i style="width:' + Math.round(100 * b.count / max) + '%;background:' + col + '"></i></span>'
        + '<span class="bl-v">' + fmtInt(b.count) + '</span></div>';
    }).join('');
    var headline = pct >= 99.9
      ? '<b>100% of API traffic originates from the local / private network</b> — these IPs (localhost + LAN) are not geolocatable.'
      : '<b>' + pct + '% of API traffic is from the local / private network.</b> Only public-internet IPs can be geolocated.';
    return '<div class="trf-origins"><div class="or-head">' + headline + '</div>'
      + '<div class="trf-barlist">' + bars + '</div>'
      + '<div class="or-note">Client coordinates are deliberately <b>not</b> fabricated. The real usage footprint — genuine field-scan GPS — is mapped on the <a data-goto="geo">Geography globe →</a></div></div>';
  }

  function endpointTable(eps) {
    eps = eps || [];
    if (!eps.length) return '<div class="trf-empty">no endpoints</div>';
    return '<div class="trf-eptable">' + eps.slice(0, 10).map(function (e) {
      var hc = e.health >= 70 ? 'ok' : (e.health >= 45 ? 'warn' : 'bad');
      var ec = (e.error_rate || 0) >= 20 ? 'bad' : ((e.error_rate || 0) >= 5 ? 'warn' : '');
      return '<div class="ep-row" data-path="' + esc(e.path) + '" title="p50 ' + (e.p50 || 0) + ' · p95 ' + (e.p95 || 0) + ' · p99 ' + (e.p99 || 0) + 'ms">'
        + '<span class="ep-h ' + hc + '" title="health ' + e.health + '"></span>'
        + '<span class="ep-p mono">' + esc(e.path) + '</span>'
        + '<span class="ep-r mono">' + fmtInt(e.requests) + '</span>'
        + '<span class="ep-e mono ' + ec + '">' + (e.error_rate || 0) + '%</span>'
        + '<span class="ep-l mono">' + fmtInt(e.p95 || 0) + 'ms</span></div>';
    }).join('') + '</div>';
  }

  function topBlock(top) {
    top = top || {};
    return '<div class="trf-subh">top keys</div>'
      + barList(top.keys, {
        drill: true, id: function (k) { return k.public_id; }, label: function (k) { return k.name; },
        sub: function (k) { return k.error_rate + '% err'; }, fmt: function (v) { return fmtInt(v); }
      })
      + '<div class="trf-subh" style="margin-top:14px">top origin IPs</div>'
      + barList(top.ips, { label: function (i) { return i.ip; }, fmt: function (v) { return fmtInt(v); } });
  }

  function wireApi(pane, ctx) {
    pane.addEventListener('click', function (e) {
      var k = e.target.closest && e.target.closest('.trf-kpi[data-drill]');
      if (k && A.openDrillList) {
        var d = k.getAttribute('data-drill');
        if (d === 'requests') A.openDrillList('requests', {}, 'all requests');
        else if (d === 'errors') A.openDrillList('requests', { status: 'error' }, 'errors');
        return;
      }
      var key = e.target.closest && e.target.closest('.bl-row.clickable[data-id]');
      if (key && A.openDetail) { A.openDetail('key', key.getAttribute('data-id')); return; }
      var go = e.target.closest && e.target.closest('[data-goto]');
      if (go) { var nav = document.querySelector('.adm-nav[data-sec=' + go.getAttribute('data-goto') + ']'); if (nav) nav.click(); return; }
    });
  }

  // ── Traffic Terrain (WebGL 3D hero) ─────────────────────────────────────────
  function disposeTerrain() { if (st.terrain) { try { st.terrain.dispose(); } catch (e) {} st.terrain = null; } }

  function mountTerrain(pane) {
    disposeTerrain();
    var cv = pane.querySelector('.trf-terrain-cv'); if (!cv) return;
    // size the canvas backing store to its CSS box
    function sizeCv() { var r = cv.getBoundingClientRect(); cv.width = Math.max(320, r.width | 0); cv.height = Math.max(200, r.height | 0); }
    fetchWidget('api_terrain').then(function (d) {
      d = d || {}; st.lastData.terrain = d;
      sizeCv();
      var has3d = window.ADM_TERRAIN && window.ADM_TERRAIN.ok;
      if (has3d) {
        st.terrain = window.ADM_TERRAIN.create(cv, d, {
          W: cv.width, H: cv.height,
          onHover: function (c, x, y) {
            tip('<b>' + fmtInt(c.n) + ' req' + (c.e ? ' · ' + c.e + ' err' : '') + '</b>'
              + '<span>' + esc((c.day || '').slice(5)) + ' · ' + String(c.hour).padStart(2, '0') + ':00 · ' + (c.lat || 0) + 'ms</span>', x, y);
          },
          onLeave: hideTip,
        });
      }
      if (!st.terrain) { cv.style.display = 'none'; paintTerrain2D(pane, d); }  // WebGL unavailable → 2D fallback
    });
  }

  // 2D fallback: flat day×hour heatmap on the same canvas-host
  function paintTerrain2D(pane, d) {
    var stage = pane.querySelector('.trf-hero-stage, .trf-terrain-stage-lg') || pane; if (!stage) return;
    var days = d.days || [], grid = d.grid || [];
    var maxN = 1; grid.forEach(function (r) { (r || []).forEach(function (c) { maxN = Math.max(maxN, (c || {}).n || 0); }); });
    var rows = days.map(function (day, di) {
      var cells = (grid[di] || []).map(function (c, h) {
        c = c || {}; var n = c.n || 0, err = n ? (c.e || 0) / n : 0;
        var col = err >= 0.2 ? '240,97,107' : (err >= 0.05 ? '240,181,65' : '74,222,128');
        var a = n ? (0.18 + 0.82 * (n / maxN)) : 0.04;
        return '<i title="' + esc(day.slice(5)) + ' ' + String(h).padStart(2, '0') + ':00 · ' + fmtInt(n) + ' req" style="background:rgba(' + col + ',' + a.toFixed(2) + ')"></i>';
      }).join('');
      return '<div class="t2-row"><span class="t2-d">' + esc(day.slice(5)) + '</span>' + cells + '</div>';
    }).join('');
    var host = document.createElement('div'); host.className = 'trf-terrain2d';
    host.innerHTML = rows + '<div class="t2-foot">3D view needs WebGL — showing flat heatmap</div>';
    stage.appendChild(host);
  }

  // ── Endpoint Galaxy (reused SVG module) ─────────────────────────────────────
  function disposeGalaxy() { if (st.galaxy) { try { st.galaxy.dispose(); } catch (e) {} st.galaxy = null; } }

  // Fuse traffic_endpoints (nodes) + traffic_sequences (edges) into the galaxy
  // module's contract: endpoints {path,n,err_rate}, edges {from,to,count}, hub.
  function buildGalaxy(epData, seqData) {
    var eps = (epData && epData.endpoints) || [];
    var edges = ((seqData && seqData.edges) || []).map(function (e) {
      return { from: e.from, to: e.to, count: e.weight || 0 };
    });
    var nodes = eps.map(function (e) {
      var n = e.requests || 0, errPct = e.error_rate || 0;
      return { path: e.path, n: n, err_rate: errPct / 100, err_pct: errPct, p95: e.p95 || 0, health: e.health };
    });
    // hub = endpoint with the most edge connections (fallback: busiest)
    var deg = {};
    edges.forEach(function (e) { deg[e.from] = (deg[e.from] || 0) + e.count; deg[e.to] = (deg[e.to] || 0) + e.count; });
    var hub = null, best = -1;
    nodes.forEach(function (nd) { var d = deg[nd.path] || 0; if (d > best) { best = d; hub = nd.path; } });
    if (!hub && nodes.length) hub = nodes[0].path;
    return { nodes: nodes, edges: edges, hub: hub };
  }

  function galaxyStrip(g) {
    var bits = [];
    if (g.hub) { var d = g.edges.filter(function (e) { return e.from === g.hub || e.to === g.hub; }).length; bits.push('<span class="cst-ins"><b>' + esc(g.hub) + '</b> hub · ' + d + ' pathways</span>'); }
    var top = g.edges.slice().sort(function (a, b) { return b.count - a.count; })[0];
    if (top) bits.push('<span class="cst-ins">hot route <b>' + esc(top.from) + ' → ' + esc(top.to) + '</b> · ' + fmtInt(top.count) + '×</span>');
    bits.push('<span class="cst-ins">' + g.nodes.length + ' endpoints · ' + g.edges.length + ' sequences</span>');
    return bits.join('');
  }

  function mountGalaxy(pane) {
    disposeGalaxy();
    var body = cBody(pane, 'c-galaxy'); if (!body) return;
    if (!(window.APIN && window.APIN.galaxy)) { body.innerHTML = '<div class="trf-empty">galaxy module unavailable</div>'; return; }
    Promise.all([fetchWidget('api_endpoints'), fetchWidget('api_sequences')]).then(function (res) {
      var g = buildGalaxy(res[0], res[1]); st.lastData.galaxy = g;
      if (!g.nodes.length || !g.edges.length) { body.innerHTML = '<div class="trf-empty">the constellation is dark — call sequences will light it up</div>'; return; }
      body.innerHTML = '<div class="trf-gx-wrap"><div class="trf-galaxy"><div class="cst-stage" id="trf-gx-stage"></div>'
        + '<div class="cst-strip">' + galaxyStrip(g) + '</div></div>'
        + '<div class="trf-gx-stats" id="trf-gx-stats">' + galaxyStats(g) + '</div></div>';
      var stage = body.querySelector('#trf-gx-stage');
      st.galaxy = window.APIN.galaxy.create(stage, {
        mode: 'compact', endpoints: g.nodes, edges: g.edges, hub: g.hub, focus: g.hub,
        onHoverNode: function (path, x, y) {
          var e = g.nodes.find(function (z) { return z.path === path; }); if (!e) return;
          tip('<b>' + esc(e.path) + '</b><span>' + fmtInt(e.n) + ' req · ' + (e.err_pct || 0) + '% err · p95 ' + fmtInt(e.p95) + 'ms</span>', x, y);
        },
        onHoverEdge: function (rec, x, y) { tip('<b>' + esc(rec.from) + ' → ' + esc(rec.to) + '</b><span>' + fmtInt(rec.count) + '× in sequence</span>', x, y); },
        onLeave: hideTip,
        onClickNode: function (path) { if (A.openDrillList) A.openDrillList('requests', { path: path }, path); },
      });
      wireGalaxyStats(body, g);
    });
  }

  // Stats sidebar next to the galaxy: live totals + busiest + most-error
  // endpoints. Hovering a row focuses/pulses that star; clicking drills it.
  function galaxyStats(g) {
    var nodes = g.nodes || [];
    var totReq = nodes.reduce(function (a, n) { return a + (n.n || 0); }, 0);
    var byReq = nodes.slice().sort(function (a, b) { return (b.n || 0) - (a.n || 0); }).slice(0, 6);
    var byErr = nodes.filter(function (n) { return (n.err_pct || 0) > 0; }).sort(function (a, b) { return (b.err_pct || 0) - (a.err_pct || 0); }).slice(0, 5);
    var maxR = byReq.reduce(function (a, n) { return Math.max(a, n.n || 0); }, 1);
    function row(n, val, sub, bad) {
      return '<div class="gxs-row" data-path="' + esc(n.path) + '" role="button" tabindex="0">'
        + '<span class="gxs-p mono">' + esc(n.path.replace(/^\/api\//, '')) + '</span>'
        + '<span class="gxs-bar"><i style="width:' + Math.round(100 * (n.n || 0) / maxR) + '%' + (bad ? ';background:var(--crimson)' : '') + '"></i></span>'
        + '<span class="gxs-v mono' + (bad ? ' bad' : '') + '">' + val + '</span></div>';
    }
    return '<div class="gxs-tot">'
      + '<div class="gxs-stat"><b>' + fmtInt(nodes.length) + '</b><span>endpoints</span></div>'
      + '<div class="gxs-stat"><b>' + fmtInt((g.edges || []).length) + '</b><span>sequences</span></div>'
      + '<div class="gxs-stat"><b>' + fmtInt(totReq) + '</b><span>requests</span></div></div>'
      + '<div class="gxs-h">busiest endpoints</div>'
      + '<div class="gxs-list">' + byReq.map(function (n) { return row(n, fmtInt(n.n || 0)); }).join('') + '</div>'
      + (byErr.length ? '<div class="gxs-h">highest error rate</div><div class="gxs-list">'
        + byErr.map(function (n) { return row(n, (n.err_pct || 0) + '%', null, true); }).join('') + '</div>' : '');
  }
  function wireGalaxyStats(body, g) {
    var stats = body.querySelector('#trf-gx-stats'); if (!stats) return;
    stats.addEventListener('mousemove', function (e) {
      var r = e.target.closest && e.target.closest('.gxs-row[data-path]'); if (!r) return;
      var p = r.getAttribute('data-path'); var nd = (g.nodes || []).find(function (z) { return z.path === p; });
      if (nd) tip('<b>' + esc(nd.path) + '</b><span>' + fmtInt(nd.n) + ' req · ' + (nd.err_pct || 0) + '% err · p95 ' + fmtInt(nd.p95) + 'ms</span>', e.clientX, e.clientY);
      if (st.galaxy && st.galaxy.setFocus) { try { st.galaxy.setFocus(p); } catch (er) {} }
    });
    stats.addEventListener('mouseleave', hideTip);
    stats.addEventListener('click', function (e) {
      var r = e.target.closest && e.target.closest('.gxs-row[data-path]'); if (!r) return;
      if (A.openDrillList) A.openDrillList('requests', { path: r.getAttribute('data-path') }, r.getAttribute('data-path'));
    });
  }

  // ── Website tab ─────────────────────────────────────────────────────────────
  function renderWeb(pane) {
    var ribbon = '<div class="trf-ribbon">'
      + kpi('w-vis', 'visits', {})
      + kpi('w-sess', 'sessions', {})
      + kpi('w-time', 'avg active', { raw: '<b class="trf-static" id="w-time-v">—</b><small>s</small>' })
      + kpi('w-bounce', 'bounce', { raw: '<b class="trf-static" id="w-bounce-v">—</b>%' })
      + kpi('w-scroll', 'avg scroll', { raw: '<span class="trf-num" id="w-scroll" data-num="0">0</span>%' })
      + kpi('w-ret', 'new / return', { raw: '<b class="trf-static" id="w-ret-v">—</b>' })
      + '</div>';
    var grid = '<div class="trf-grid">'
      + card(7, 'click heatmap · real coordinates', LOADBODY, 'web_heatmap', 'c-heatmap')
      + card(5, 'devices', LOADBODY, 'web_devices', 'c-devices')
      + card(7, 'top pages', LOADBODY, 'web_pages', 'c-pages')
      + card(5, 'acquisition', LOADBODY, 'web_acq', 'c-acq')
      + card(12, 'most-clicked elements', LOADBODY, 'web_elements', 'c-elements')
      + '</div>';
    pane.innerHTML = ribbon + grid + '<div class="trf-stats-sec" id="trf-stats-web"></div>';
    staggerIn(pane);
    wireWeb(pane);
    mountDeck(pane, 'trf-stats-web', 'website');

    fetchWidget('web_overview').then(function (ov) {
      ov = ov || {};
      setRoll(pane, 'w-vis', ov.visits || 0); setRoll(pane, 'w-sess', ov.sessions || 0); setRoll(pane, 'w-scroll', ov.avg_scroll || 0); rollAll(pane);
      var t = pane.querySelector('#w-time-v'); if (t) t.textContent = (ov.avg_active_s || 0);
      var b = pane.querySelector('#w-bounce-v'); if (b) b.textContent = (ov.bounce_rate || 0);
      var r = pane.querySelector('#w-ret-v'); if (r) r.textContent = (ov.new || 0) + ' / ' + (ov.returning || 0);
    });
    fetchWidget('web_heatmap').then(function (hm) {
      hm = hm || {}; st.lastData.heatmap = hm;
      var b = cBody(pane, 'c-heatmap');
      if (b) { b.innerHTML = heatmapBlock(hm); cLabel(pane, 'c-heatmap', 'click heatmap · ' + (hm.total || 0) + ' clicks · real coordinates'); paintHeatmap(pane, hm); }
      var el = cBody(pane, 'c-elements');
      if (el) el.innerHTML = barList((hm.elements || []), { label: function (e) { return e.label; }, sub: function (e) { return e.dead ? e.dead + ' dead' : ''; }, fmt: function (v) { return fmtInt(v); } });
      wireWeb(pane); wireHovers(pane);
    });
    fetchWidget('web_devices').then(function (dv) {
      dv = dv || {}; var b = cBody(pane, 'c-devices');
      if (b) { b.innerHTML = donut(dv.types, 'sessions', fmtInt((dv.types || []).reduce(function (a, x) { return a + x.value; }, 0))) + deviceMini(dv); wireHovers(pane); }
    });
    fetchWidget('web_pages').then(function (pg) { var b = cBody(pane, 'c-pages'); if (b) { b.innerHTML = pageTable((pg || {}).pages); wireHovers(pane); } });
    fetchWidget('web_acquisition').then(function (ac) { var b = cBody(pane, 'c-acq'); if (b) b.innerHTML = acqBlock(ac || {}); });
  }

  // entrance: stagger cards + ribbon tiles up/in
  function staggerIn(pane) {
    if (REDUCED) return;
    var els = [].slice.call(pane.querySelectorAll('.trf-kpi, .trf-card'));
    els.forEach(function (el, i) {
      el.style.animation = 'none';
      el.style.opacity = '0';
      el.style.transform = 'translateY(10px)';
      setTimeout(function () {
        el.style.transition = 'opacity .45s ease, transform .45s cubic-bezier(.2,.7,.2,1)';
        el.style.opacity = ''; el.style.transform = '';
      }, 30 + i * 35);
    });
  }

  function heatmapBlock(hm) {
    var routes = (hm.routes || []).slice(0, 6);
    var chips = routes.map(function (r, i) {
      return '<button class="hm-chip' + (i === 0 ? ' on' : '') + '" data-route="' + esc(r.route) + '">' + esc(r.route) + '<small>' + r.visits + '</small></button>';
    }).join('');
    return '<div class="hm-chips">' + chips + '</div>'
      + '<div class="hm-stage"><canvas class="hm-canvas" width="700" height="380"></canvas>'
      + '<div class="hm-legend"><span class="cool">cool</span><span class="bar"></span><span class="hot">hot</span>'
      + '<span class="hm-meta">' + (hm.dead || 0) + ' dead · ' + (hm.rage || 0) + ' rage</span></div></div>';
  }

  function paintHeatmap(pane, hm) {
    var cv = pane.querySelector('.hm-canvas'); if (!cv) return;
    var pts = (hm.points || []).filter(function (p) { return p.x != null && p.y != null; });
    var ctx = cv.getContext('2d'); var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    // backdrop wireframe
    ctx.strokeStyle = 'rgba(255,255,255,.05)'; ctx.lineWidth = 1;
    for (var gx = 0; gx <= W; gx += 70) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke(); }
    for (var gy = 0; gy <= H; gy += 50) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke(); }
    if (!pts.length) { ctx.fillStyle = 'rgba(255,255,255,.3)'; ctx.font = '13px monospace'; ctx.fillText('no clicks on this route', 20, 28); return; }
    var maxY = pts.reduce(function (a, b) { return Math.max(a, b.y); }, 1);
    var maxVw = pts.reduce(function (a, b) { return Math.max(a, b.vw || 1); }, 1);
    // additive bloom
    ctx.globalCompositeOperation = 'lighter';
    pts.forEach(function (p) {
      var fx = Math.max(0, Math.min(1, p.x / (p.vw || maxVw)));
      var fy = Math.max(0, Math.min(1, p.y / maxY));
      var cx = fx * (W - 20) + 10, cy = fy * (H - 20) + 10;
      var col = p.rage ? '240,97,107' : (p.dead ? '240,181,65' : '74,222,128');
      var g = ctx.createRadialGradient(cx, cy, 0, cx, cy, 26);
      g.addColorStop(0, 'rgba(' + col + ',.5)'); g.addColorStop(1, 'rgba(' + col + ',0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, 26, 0, 6.2832); ctx.fill();
    });
    ctx.globalCompositeOperation = 'source-over';
    // dead/rage markers on top
    pts.forEach(function (p) {
      if (!p.dead && !p.rage) return;
      var fx = Math.max(0, Math.min(1, p.x / (p.vw || maxVw))), fy = Math.max(0, Math.min(1, p.y / maxY));
      var cx = fx * (W - 20) + 10, cy = fy * (H - 20) + 10;
      ctx.strokeStyle = p.rage ? 'rgba(240,97,107,.9)' : 'rgba(240,181,65,.9)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 6.2832); ctx.stroke();
    });
  }

  function deviceMini(dv) {
    function mini(arr, title) {
      arr = (arr || []).filter(function (x) { return x.label && x.label !== 'unknown'; });
      if (!arr.length) return '';
      return '<div class="dm-line"><span class="dm-t">' + title + '</span>'
        + arr.slice(0, 4).map(function (x) { return '<span class="dm-pill">' + esc(x.label) + ' <b>' + x.value + '</b></span>'; }).join('') + '</div>';
    }
    return '<div class="trf-devmini">' + mini(dv.browsers, 'browser') + mini(dv.os, 'os') + mini(dv.locales, 'locale') + '</div>';
  }

  function pageTable(pages) {
    pages = pages || [];
    if (!pages.length) return '<div class="trf-empty">no page views</div>';
    var max = pages.reduce(function (a, b) { return Math.max(a, b.visits || 0); }, 0) || 1;
    return '<div class="trf-pagetable">' + pages.slice(0, 10).map(function (p) {
      return '<div class="pg-row" title="' + esc(p.title || '') + '">'
        + '<span class="pg-rt mono">' + esc(p.route) + '</span>'
        + '<span class="pg-bar"><i style="width:' + Math.round(100 * (p.visits || 0) / max) + '%"></i></span>'
        + '<span class="pg-v mono">' + fmtInt(p.visits) + '</span>'
        + '<span class="pg-x mono" title="avg active">' + (p.avg_active_s || 0) + 's</span></div>';
    }).join('') + '</div>';
  }

  function acqBlock(ac) {
    ac = ac || {};
    var items = (ac.referrers || []).slice();
    if (ac.direct) items.unshift({ label: '(direct)', value: ac.direct });
    return barList(items, { label: function (i) { return i.label; }, fmt: function (v) { return fmtInt(v); }, empty: 'no referrers recorded' })
      + ((ac.utm_sources || []).length ? '<div class="trf-subh" style="margin-top:12px">utm sources</div>'
        + barList(ac.utm_sources, { label: function (i) { return i.label; } }) : '');
  }

  function wireWeb(pane) {
    var chips = pane.querySelector('.hm-chips');
    if (chips) chips.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('.hm-chip[data-route]'); if (!b) return;
      [].forEach.call(chips.querySelectorAll('.hm-chip'), function (x) { x.classList.toggle('on', x === b); });
      var route = b.getAttribute('data-route');
      fetchWidget('web_heatmap', '&route=' + encodeURIComponent(route)).then(function (hm) { paintHeatmap(pane, hm || {}); });
    });
  }

  // ── tab + window control ────────────────────────────────────────────────────
  function moveInk() {
    var nav = document.getElementById('trf-subnav'), ink = document.getElementById('trf-tab-ink');
    var on = nav && nav.querySelector('.trf-tab.on');
    if (nav && ink && on) { ink.style.width = on.offsetWidth + 'px'; ink.style.transform = 'translateX(' + on.offsetLeft + 'px)'; }
  }
  function renderActive() {
    var apiP = document.getElementById('trf-api'), webP = document.getElementById('trf-web');
    if (!apiP || !webP) return;
    apiP.hidden = st.sub !== 'api'; webP.hidden = st.sub !== 'web';
    moveInk();
    // PERF: a pane already rendered for the current window just toggles
    // visibility — switching API↔Website is then INSTANT, no refetch.
    var wantKey = st.sub + '|' + st.window;
    if (st.renderedKey[st.sub] === wantKey) return;
    st.renderedKey[st.sub] = wantKey;
    if (st.sub === 'api') renderApi(apiP); else renderWeb(webP);
  }
  function wire() {
    if (st.wired) return; st.wired = true;
    var nav = document.getElementById('trf-subnav');
    if (nav) nav.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('.trf-tab[data-sub]'); if (!b) return;
      [].forEach.call(nav.querySelectorAll('.trf-tab'), function (x) {
        var on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      st.sub = b.getAttribute('data-sub'); renderActive();
    });
    var win = document.getElementById('trf-window');
    if (win) win.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('button[data-w]'); if (!b) return;
      [].forEach.call(win.querySelectorAll('button'), function (x) {
        var on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      st.window = b.getAttribute('data-w'); st.renderedKey = {}; renderActive();
    });
    window.addEventListener('resize', moveInk);
    // expand (⤢) → theatre modal (delegated across both panes)
    document.getElementById('trf-api').addEventListener('click', onExpandClick);
    document.getElementById('trf-web').addEventListener('click', onExpandClick);
  }

  function onExpandClick(e) {
    var x = e.target.closest && e.target.closest('.trf-x[data-exp]');
    if (x) { e.stopPropagation(); openTheatre(x.getAttribute('data-exp')); }
  }

  // ── hover tooltips for every chart (called after each pane render) ─────────
  function wireHovers(pane) {
    function on(sel, fn) {
      [].forEach.call(pane.querySelectorAll(sel), function (el) {
        el.addEventListener('mousemove', function (e) { fn(el, e); });
        el.addEventListener('mouseleave', hideTip);
      });
    }
    // sparkline dots
    var spark = pane.querySelector('.trf-spark');
    if (spark && spark.__series) {
      on('.sp-dot', function (el, e) {
        var s = spark.__series[+el.getAttribute('data-i')] || {};
        tip('<b>' + fmtInt(s.c || 0) + '</b><span>' + esc((s.t || '').replace('T', ' ')) + '</span>', e.clientX, el.getBoundingClientRect().top);
      });
    }
    // donut segments
    on('.dn-seg', function (el, e) {
      tip('<b>' + fmtInt(el.getAttribute('data-value')) + '</b><span>' + esc(el.getAttribute('data-label')) + '</span>', e.clientX, e.clientY);
    });
    // hive cells
    on('.hv-cell:not(.z)', function (el, e) {
      tip('<b>' + fmtInt(el.getAttribute('data-v')) + '</b><span>' + esc(el.getAttribute('data-m')) + ' · ' + esc(el.getAttribute('data-cls')) + '</span>', e.clientX, e.clientY);
    });
    // ridge rows
    on('.rg-row', function (el, e) {
      tip('<span>' + esc(el.getAttribute('data-t')) + '</span>', e.clientX, e.clientY);
    });
    // endpoint rows — title already set; add a richer tip
    on('.ep-row', function (el, e) {
      tip('<b>' + esc(el.getAttribute('data-path')) + '</b><span>' + el.getAttribute('title') + '</span>', e.clientX, e.clientY);
    });
  }

  // ── functional expand theatre (lightbox) ──────────────────────────────────
  var EXP_TITLES = {
    api_terrain: 'Traffic Terrain · 7d × 24h',
    api_galaxy: 'Endpoint Galaxy · call sequences',
    api_series: 'Requests over time', api_status: 'Status mix', api_methods: 'Method × Status',
    api_latency: 'Latency percentiles', api_endpoints: 'Endpoints', api_top: 'Top keys & origins',
    web_heatmap: 'Click heatmap', web_devices: 'Devices', web_pages: 'Top pages',
    web_acq: 'Acquisition', web_elements: 'Most-clicked elements',
  };
  function openTheatre(key) {
    // Terrain gets a fresh, larger WebGL instance in the theatre (cloning a live
    // canvas is dead — it must be re-created against st.lastData.terrain).
    if (key === 'api_terrain') return openTerrainTheatre();
    if (key === 'api_galaxy') return openGalaxyTheatre();
    // clone the source card's rendered body into a large modal (so the expand
    // is never dead) — full unique 5-section spines are the next pass.
    var srcBtn = document.querySelector('.trf-x[data-exp="' + key + '"]');
    var srcCard = srcBtn && srcBtn.closest('.trf-card');
    var bodyHTML = srcCard ? srcCard.querySelector('.trf-card-b').innerHTML : '<div class="trf-empty">no data</div>';
    var back = document.createElement('div'); back.className = 'trf-lb-back';
    var lb = document.createElement('div'); lb.className = 'trf-lb';
    lb.innerHTML = '<div class="trf-lb-h"><b>' + esc(EXP_TITLES[key] || key) + '</b>'
      + '<button class="trf-lb-x" aria-label="close">×</button></div>'
      + '<div class="trf-lb-b">' + bodyHTML + '</div>';
    document.body.appendChild(back); document.body.appendChild(lb);
    requestAnimationFrame(function () { back.classList.add('on'); lb.classList.add('on'); });
    function close() { back.classList.remove('on'); lb.classList.remove('on'); setTimeout(function () { back.remove(); lb.remove(); }, 260); }
    back.addEventListener('click', close);
    lb.querySelector('.trf-lb-x').addEventListener('click', close);
    document.addEventListener('keydown', function esc2(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc2); } });
    wireHovers(lb);
    // re-paint heatmap canvas if this is the heatmap card
    if (key === 'web_heatmap' && st.lastData.heatmap) paintHeatmap(lb, st.lastData.heatmap);
  }

  function openTerrainTheatre() {
    var d = st.lastData.terrain || {};
    var back = document.createElement('div'); back.className = 'trf-lb-back';
    var lb = document.createElement('div'); lb.className = 'trf-lb trf-lb-terrain';
    lb.innerHTML = '<div class="trf-lb-h"><b>' + esc(EXP_TITLES.api_terrain) + '</b>'
      + '<button class="trf-lb-x" aria-label="close">×</button></div>'
      + '<div class="trf-lb-b"><div class="trf-terrain-stage-lg"><canvas class="trf-terrain-cv-lg"></canvas>'
      + '<div class="trf-terrain-legend"><span class="cool">quiet</span><span class="bar"></span><span class="hot">peak</span>'
      + '<span class="sep"></span><i class="lg-jade"></i>clean<i class="lg-ember"></i>errors</div></div>'
      + '<p class="trf-lb-note">Each ridge is one hour; depth is the last 7 days. Height ∝ requests, colour ∝ error rate. Drag to orbit, hover a peak.</p></div>';
    document.body.appendChild(back); document.body.appendChild(lb);
    requestAnimationFrame(function () { back.classList.add('on'); lb.classList.add('on'); });
    var inst = null;
    setTimeout(function () {
      var cv = lb.querySelector('.trf-terrain-cv-lg'); if (!cv) return;
      var r = cv.getBoundingClientRect(); cv.width = Math.max(480, r.width | 0); cv.height = Math.max(320, r.height | 0);
      if (window.ADM_TERRAIN && window.ADM_TERRAIN.ok) {
        inst = window.ADM_TERRAIN.create(cv, d, {
          W: cv.width, H: cv.height,
          onHover: function (c, x, y) { tip('<b>' + fmtInt(c.n) + ' req' + (c.e ? ' · ' + c.e + ' err' : '') + '</b><span>' + esc((c.day || '').slice(5)) + ' · ' + String(c.hour).padStart(2, '0') + ':00 · ' + (c.lat || 0) + 'ms</span>', x, y); },
          onLeave: hideTip,
        });
      }
      if (!inst) { cv.style.display = 'none'; paintTerrain2D(lb, d); }
    }, 60);
    function close() {
      back.classList.remove('on'); lb.classList.remove('on');
      if (inst) { try { inst.dispose(); } catch (e) {} }
      setTimeout(function () { back.remove(); lb.remove(); }, 260);
    }
    back.addEventListener('click', close);
    lb.querySelector('.trf-lb-x').addEventListener('click', close);
    document.addEventListener('keydown', function ek(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', ek); } });
  }

  function openGalaxyTheatre() {
    var g = st.lastData.galaxy || { nodes: [], edges: [], hub: null };
    var back = document.createElement('div'); back.className = 'trf-lb-back';
    var lb = document.createElement('div'); lb.className = 'trf-lb trf-lb-terrain';
    lb.innerHTML = '<div class="trf-lb-h"><b>' + esc(EXP_TITLES.api_galaxy) + '</b>'
      + '<button class="trf-lb-x" aria-label="close">×</button></div>'
      + '<div class="trf-lb-b"><div class="trf-gxx-stage" id="trf-gxx-stage"></div>'
      + '<p class="trf-lb-note">Each star is an endpoint (size ∝ requests, colour ∝ error rate); lines are observed call sequences within a session. The hub is the most-traversed endpoint. Drag to pan, scroll to zoom, hover a star or link, click a star to drill its requests.</p></div>';
    document.body.appendChild(back); document.body.appendChild(lb);
    requestAnimationFrame(function () { back.classList.add('on'); lb.classList.add('on'); });
    var inst = null;
    setTimeout(function () {
      var stage = lb.querySelector('#trf-gxx-stage'); if (!stage) return;
      if (window.APIN && window.APIN.galaxy && g.nodes.length) {
        inst = window.APIN.galaxy.create(stage, {
          mode: 'full', endpoints: g.nodes, edges: g.edges, hub: g.hub, focus: g.hub,
          onHoverNode: function (path, x, y) { var e = g.nodes.find(function (z) { return z.path === path; }); if (!e) return; tip('<b>' + esc(e.path) + '</b><span>' + fmtInt(e.n) + ' req · ' + (e.err_pct || 0) + '% err · p95 ' + fmtInt(e.p95) + 'ms</span>', x, y); },
          onHoverEdge: function (rec, x, y) { tip('<b>' + esc(rec.from) + ' → ' + esc(rec.to) + '</b><span>' + fmtInt(rec.count) + '× in sequence</span>', x, y); },
          onLeave: hideTip,
          onClickNode: function (path) { if (A.openDrillList) A.openDrillList('requests', { path: path }, path); },
        });
      } else { stage.innerHTML = '<div class="trf-empty">no call sequences in window</div>'; }
    }, 60);
    function close() {
      back.classList.remove('on'); lb.classList.remove('on');
      if (inst) { try { inst.dispose(); } catch (e) {} }
      setTimeout(function () { back.remove(); lb.remove(); }, 260);
    }
    back.addEventListener('click', close);
    lb.querySelector('.trf-lb-x').addEventListener('click', close);
    document.addEventListener('keydown', function ek(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', ek); } });
  }

  // ── live auto-update (poll the active overview; roll the KPI numbers) ──────
  function startLive() {
    stopLive();
    st.live = setInterval(function () {
      if (!st.mounted || document.hidden) return;
      var pane = document.getElementById(st.sub === 'api' ? 'trf-api' : 'trf-web');
      if (!pane || pane.hidden) return;
      if (st.sub === 'api') {
        fetchWidget('api_overview').then(function (ov) {
          if (!ov) return;
          setRoll(pane, 'k-req', ov.requests || 0); setRoll(pane, 'k-p95', (ov.latency || {}).p95 || 0);
          rollAll(pane);
          var dot = document.querySelector('#trf-live-dot'); if (dot) { dot.classList.add('pulse'); setTimeout(function () { dot.classList.remove('pulse'); }, 600); }
        });
      } else {
        fetchWidget('web_overview').then(function (ov) {
          if (!ov) return;
          setRoll(pane, 'w-vis', ov.visits || 0); setRoll(pane, 'w-sess', ov.sessions || 0); rollAll(pane);
        });
      }
    }, 9000);
  }
  function stopLive() { if (st.live) { clearInterval(st.live); st.live = null; } }

  // ── SSE: org-wide live request feed → instant KPI + terrain + galaxy react ──
  function pulseLiveDot() {
    var dot = document.getElementById('trf-live-dot');
    if (dot) { dot.classList.add('pulse'); setTimeout(function () { dot.classList.remove('pulse'); }, 600); }
  }
  function liveBumpRequests() {
    var pane = document.getElementById('trf-api'); if (!pane || pane.hidden) return;
    var el = pane.querySelector('#k-req'); if (!el) return;
    var v = (+el.getAttribute('data-num') || 0) + 1;
    el.setAttribute('data-num', v); el.textContent = fmtInt(v);
    el.classList.add('trf-tick'); setTimeout(function () { el.classList.remove('trf-tick'); }, 320);
  }
  // one live request arrived → react across the visible widgets. Named + exposed
  // (window.ADM_TRAFFIC.onLiveRequest) so the live pipeline is testable without
  // forging a keyed request.
  function onLiveRequest(ev) {
    if (!ev || ev.type !== 'request') return;
    var isErr = (ev.status || 0) >= 400;
    pulseLiveDot();
    // Stats Deck live reaction — only the affected cards animate (errors → 3 cards,
    // healthy hits → change-detection). Runs on whichever pane's deck is mounted.
    var deck = st.decks[st.sub === 'api' ? 'api' : 'website'];
    if (deck && deck.react) { try { deck.react(ev); } catch (e) {} }
    if (st.sub !== 'api') return;   // Website tab has its own KPIs
    liveBumpRequests();
    if (st.terrain && st.terrain.bump) {
      var hr, ts = ev.ts || '';
      if (ts.length >= 13) hr = parseInt(ts.slice(11, 13), 10);
      if (isNaN(hr)) hr = new Date().getUTCHours();
      st.terrain.bump(hr, isErr);
    }
    if (st.galaxy && st.galaxy.pulse && ev.path) { try { st.galaxy.pulse(ev.path); } catch (e) {} }
  }
  function startSSE() {
    stopSSE();
    if (typeof EventSource === 'undefined') return;
    var es;
    try { es = new EventSource('/api/account/admin/traffic/stream'); }
    catch (e) { return; }
    st.sse = es;
    es.onmessage = function (m) { var ev; try { ev = JSON.parse(m.data); } catch (e) { return; } onLiveRequest(ev); };
    es.onerror = function () { /* EventSource auto-reconnects; poll covers gaps */ };
  }
  function stopSSE() { if (st.sse) { try { st.sse.close(); } catch (e) {} st.sse = null; } }

  window.ADM_TRAFFIC = {
    mount: function () { wire(); st.mounted = true; renderActive(); startLive(); startSSE(); },
    dispose: function () { st.mounted = false; stopLive(); stopSSE(); hideTip(); disposeTerrain(); disposeGalaxy(); disposeDecks(); },
    // diagnostics: feed a synthetic request through the exact live-reaction path
    onLiveRequest: onLiveRequest,
    _st: st,
  };
})();
