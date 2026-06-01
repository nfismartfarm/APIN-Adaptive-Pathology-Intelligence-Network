/* 9.N.T34 · Confidence Ladder — "Soil Horizon / Core Sample".
 *
 * Confidence as a soil cross-section: Certain (green topsoil) down to OOD
 * (terracotta bedrock). Band height = share of detections; a "trust spine"
 * gradient on the left carries the confidence axis with an average marker;
 * texture roughens downward (less certain = rockier). Everything is hoverable
 * and the whole thing reacts to live `scan` SSE events (no refresh).
 *
 * Registers window.APIN.analyticsConfidence = { mount, setData, onScan,
 * setRange, renderExpanded, alive, destroy }.
 */
(function () {
  'use strict';
  const C = {};
  // palette — matches the Pathology-Journal confidence semantics
  const BAND_COLOR = {
    certain: '#3f8f5e', likely: '#6cbf94', tentative: '#c98a2b',
    uncertain: '#8a8472', ood: '#b3402f',
  };
  const BAND_ORDER = ['certain', 'likely', 'tentative', 'uncertain', 'ood'];
  const BAND_LABEL = {
    certain: 'Certain', likely: 'Likely', tentative: 'Tentative',
    uncertain: 'Uncertain', ood: 'OOD',
  };
  const BAND_RANGE = {
    certain: '≥85%', likely: '65–85%', tentative: '45–65%',
    uncertain: '<45%', ood: 'flagged',
  };
  const BAND_ACTION = {
    certain: 'ship', likely: 'usually ok', tentative: 'verify',
    uncertain: 'check', ood: 'review',
  };
  const HEALTHY = { okra_healthy: 1, brassica_healthy: 1, tomato_healthy: 1 };

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
  function titleCase(s) { return String(s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }
  function api(p) { try { return window.APIN.analyticsApi(p); } catch (e) { return Promise.reject(e); } }
  function bandOf(conf, ood) { if (ood) return 'ood'; const c = conf || 0; return c >= 0.85 ? 'certain' : c >= 0.65 ? 'likely' : c >= 0.45 ? 'tentative' : 'uncertain'; }

  let M = null;  // single live instance

  function open() { /* compat */ }

  function mount(el, opts) {
    injectCSS();
    if (M) { try { M.destroy(); } catch (e) { } M = null; }
    M = make(el, opts || {});
    return M;
  }

  function make(el, opts) {
    let _data = null, _mode = 'bands', destroyed = false;
    const onPick = opts.onPick || function () { };

    el.innerHTML = shell();
    const host = el.querySelector('.anc');
    host.querySelector('#anc-seg').addEventListener('click', e => {
      const b = e.target.closest('[data-m]'); if (!b) return;
      _mode = b.getAttribute('data-m');
      host.querySelectorAll('#anc-seg [data-m]').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
      render();
    });

    function shell() {
      return '<div class="anc">' +
        '<div class="anc-top"><span class="anc-stat" id="anc-stat">·</span>' +
        '<span class="anc-seg" id="anc-seg"><button data-m="bands" aria-pressed="true">bands</button><button data-m="tier">model tier</button></span></div>' +
        '<div class="anc-body"><div class="anc-spine" id="anc-spine"></div>' +
        '<div class="anc-strata" id="anc-strata"><div class="an-ph">loading&hellip;</div></div></div>' +
        '</div>';
    }

    function setData(d) { _data = d; render(); }

    function render() {
      if (!_data) return;
      const stat = host.querySelector('#anc-stat');
      if (stat) stat.innerHTML = '<b>' + (_data.total || 0) + '</b> detections · avg <b>' + (_data.avg_confidence != null ? Math.round(_data.avg_confidence * 100) + '%' : '·') + '</b>' + (_data.needs_review ? ' · <span class="anc-review">' + _data.needs_review + ' need review</span>' : '');
      _mode === 'tier' ? renderTier() : renderBands();
      renderSpine();
    }

    function renderSpine() {
      const sp = host.querySelector('#anc-spine'); if (!sp) return;
      const avg = _data.avg_confidence;
      const y = avg != null ? (1 - Math.max(0, Math.min(1, avg))) * 100 : null;
      sp.innerHTML = '<i class="anc-spine-grad"></i>' +
        [0.85, 0.65, 0.45].map(t => '<i class="anc-spine-tick" style="top:' + ((1 - t) * 100) + '%"></i>').join('') +
        (y != null ? '<i class="anc-spine-avg" style="top:' + y + '%" title="average confidence ' + Math.round(avg * 100) + '%"><b>' + Math.round(avg * 100) + '%</b></i>' : '');
      // hoverable spine → tooltip with the band thresholds
      sp.onmousemove = (ev) => {
        const r = sp.getBoundingClientRect(); const f = (ev.clientY - r.top) / r.height;
        const conf = Math.max(0, Math.min(1, 1 - f));
        showTip(ev.clientX, ev.clientY, '<b>' + Math.round(conf * 100) + '%</b> confidence<br><i>' + BAND_LABEL[bandOf(conf, 0)] + '</i> band');
      };
      sp.onmouseleave = hideTip;
    }

    function renderBands() {
      const wrap = host.querySelector('#anc-strata'); if (!wrap) return;
      const bands = _data.bands || [];
      const total = Math.max(1, _data.total || 1);
      wrap.innerHTML = bands.map((b, i) => {
        const grow = Math.max(0.06, b.count / total);
        const dim = ''; const tex = ' anc-tex' + i;
        return '<div class="anc-band' + tex + (b.count ? '' : ' anc-band-empty') + '" data-band="' + b.key + '" style="flex:' + grow.toFixed(4) + ';--bc:' + BAND_COLOR[b.key] + '">' +
          '<div class="anc-band-l"><span class="anc-band-name">' + esc(BAND_LABEL[b.key]) + '</span> <span class="anc-band-rng">' + esc(BAND_RANGE[b.key]) + '</span></div>' +
          '<div class="anc-band-mid">' + (b.count ? '<b class="anc-band-n" data-n="' + b.key + '">' + b.count + '</b> <span class="anc-band-dis">' + (b.top_disease ? esc(titleCase(b.top_disease)) : '·') + '</span>' : '<span class="anc-band-zero">none</span>') + '</div>' +
          '<div class="anc-band-act">' + esc(BAND_ACTION[b.key]) + '</div></div>';
      }).join('');
      wireBands();
    }

    function renderTier() {
      const wrap = host.querySelector('#anc-strata'); if (!wrap) return;
      const tiers = (_data.tiers || []).slice();
      const total = Math.max(1, _data.total || 1);
      const TCOL = { '3': '#6cbf94', '4A': '#52b788', '4B': '#2d6a4f' };
      if (!tiers.length) { wrap.innerHTML = '<div class="an-ph an-ph-empty">no tier data</div>'; return; }
      wrap.innerHTML = tiers.map(t => {
        const grow = Math.max(0.1, t.count / total);
        const col = TCOL[t.tier] || '#8a8472';
        const conf = t.avg_confidence != null ? Math.round(t.avg_confidence * 100) + '%' : '·';
        return '<div class="anc-band anc-tier" data-tier="' + esc(t.tier) + '" style="flex:' + grow.toFixed(4) + ';--bc:' + col + '">' +
          '<div class="anc-band-l"><span class="anc-band-name">Tier ' + esc(t.tier) + '</span></div>' +
          '<div class="anc-band-mid"><b class="anc-band-n">' + t.count + '</b> <span class="anc-band-dis">' + (t.top_disease ? esc(titleCase(t.top_disease)) : '·') + '</span></div>' +
          '<div class="anc-band-act">avg ' + conf + '</div></div>';
      }).join('');
      wrap.querySelectorAll('.anc-tier').forEach(node => {
        const tier = node.getAttribute('data-tier');
        node.addEventListener('mouseenter', () => { node.classList.add('on'); const t = (_data.tiers || []).find(x => String(x.tier) === tier); if (t) tierTip(node, t); });
        node.addEventListener('mousemove', moveTip);
        node.addEventListener('mouseleave', () => { node.classList.remove('on'); hideTip(); });
      });
    }

    function tierTip(node, t) {
      const bands = t.bands || {};
      const rows = BAND_ORDER.filter(k => bands[k]).map(k => '<i style="color:' + BAND_COLOR[k] + '">●</i> ' + BAND_LABEL[k] + ' ' + bands[k]).join('<br>');
      const r = node.getBoundingClientRect();
      showTip(r.right, r.top + 8, '<b>Tier ' + esc(t.tier) + '</b> · ' + t.count + ' scans · avg ' + (t.avg_confidence != null ? Math.round(t.avg_confidence * 100) + '%' : '·') + '<br>' + rows);
    }

    function wireBands() {
      host.querySelectorAll('.anc-band[data-band]').forEach(node => {
        const key = node.getAttribute('data-band');
        const band = (_data.bands || []).find(b => b.key === key);
        node.addEventListener('mouseenter', () => {
          host.querySelectorAll('.anc-band').forEach(n => n.classList.toggle('dim', n !== node));
          node.classList.add('on'); if (band && band.count) bandTip(node, band);
        });
        node.addEventListener('mousemove', moveTip);
        node.addEventListener('mouseleave', () => {
          host.querySelectorAll('.anc-band').forEach(n => n.classList.remove('dim', 'on'));
          hideTip();
        });
        node.addEventListener('click', () => { if (band && band.count) openGallery(band); });
      });
    }

    function bandTip(node, b) {
      const sev = b.severity_mix || {};
      const sevs = ['mild', 'moderate', 'severe'].filter(s => sev[s]).map(s => s + ' ' + sev[s]).join(' · ');
      const dis = Object.entries((b.specimens || []).reduce((m, s) => { m[s.diagnosis] = (m[s.diagnosis] || 0) + 1; return m; }, {})).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([d, n]) => titleCase(d) + ' ' + n).join('<br>');
      const r = node.getBoundingClientRect();
      showTip(r.right, r.top + 8,
        '<b>' + BAND_LABEL[b.key] + '</b> · ' + b.count + ' (' + b.pct + '%) · <i>' + BAND_ACTION[b.key] + '</i>' +
        (b.avg_confidence != null ? '<br>avg conf ' + Math.round(b.avg_confidence * 100) + '%' : '') +
        (sevs ? '<br>' + sevs : '') + (dis ? '<br>' + dis : '') +
        '<br><span class="anc-tip-cta">click → leaf gallery</span>');
    }

    // ── live ──────────────────────────────────────────────────────────────
    function onScan(e) {
      if (destroyed || !_data || !e) return;
      const conf = e.confidence, ood = e.is_ood ? 1 : 0;
      const bk = bandOf(conf, ood);
      _data.total = (_data.total || 0) + 1;
      // recompute running avg
      if (e.confidence != null) {
        const n = _data.total; const prev = _data.avg_confidence || 0;
        _data.avg_confidence = ((prev * (n - 1)) + e.confidence) / n;
      }
      const band = (_data.bands || []).find(b => b.key === bk);
      if (band) { band.count++; if (e.diagnosis) band.top_disease = band.top_disease || e.diagnosis; }
      if (bk === 'uncertain' || bk === 'ood') _data.needs_review = (_data.needs_review || 0) + 1;
      // tier
      if (e.tier) { let t = (_data.tiers || []).find(x => String(x.tier) === String(e.tier)); if (!t && _data.tiers) { t = { tier: e.tier, count: 0, bands: {}, avg_confidence: e.confidence }; _data.tiers.push(t); } if (t) { t.count++; t.bands[bk] = (t.bands[bk] || 0) + 1; } }
      render();
      // pulse the band that grew
      const node = host.querySelector('.anc-band[data-band="' + bk + '"]') || host.querySelector('.anc-band[data-tier="' + (e.tier || '') + '"]');
      if (node) { node.classList.add('anc-hit'); setTimeout(() => node.classList.remove('anc-hit'), 900); }
    }

    // ── specimen gallery ────────────────────────────────────────────────────
    function openGallery(band) {
      try { if (window.APIN.analyticsFocus) window.APIN.analyticsFocus({ confidence: band.key }); } catch (e) { }
      if (!(window.APIN && window.APIN.lightbox)) return;
      window.APIN.lightbox.open({
        title: BAND_LABEL[band.key] + ' · ' + band.count + ' specimens', hashKey: 'anc-gal',
        build: (panel) => {
          const wrap = document.createElement('div'); wrap.className = 'anc-gal'; panel.appendChild(wrap);
          const specs = band.specimens || [];
          wrap.innerHTML = '<div class="anc-gal-head">' + esc(BAND_LABEL[band.key]) + ' band · ' + band.count + ' detections · ' + esc(BAND_ACTION[band.key]) + '</div>' +
            '<div class="anc-gal-grid">' + specs.map(s => galCard(s)).join('') + '</div>' +
            (band.count > specs.length ? '<div class="anc-gal-more">showing ' + specs.length + ' of ' + band.count + '</div>' : '');
          wrap.querySelectorAll('[data-uid]').forEach(c => c.addEventListener('click', () => {
            const uid = c.getAttribute('data-uid');
            if (window.APIN.analyticsObservatory && window.APIN.analyticsObservatory.openInspector) { window.APIN.analyticsObservatory.openInspector({ scan_uid: uid }); return; }
            api('scan/' + encodeURIComponent(uid)).then(d => alert(d && d.found ? titleCase(d.diagnosis) + ' · ' + Math.round((d.confidence || 0) * 100) + '%' : 'unavailable')).catch(() => { });
          }));
        }
      });
    }
    function galCard(s) {
      const cls = HEALTHY[s.diagnosis] ? 'ok' : (s.severity === 'severe' ? 'bad' : 'warn');
      const img = s.has_image ? '<img loading="lazy" src="/api/account/analytics/scan/' + encodeURIComponent(s.scan_uid) + '/image" alt="">' : '<span class="anc-gal-noimg"></span>';
      return '<button class="anc-gal-card ' + cls + '" data-uid="' + esc(s.scan_uid) + '">' + img +
        '<span class="anc-gal-cap">' + esc(titleCase(s.diagnosis)) + '<small>' + Math.round((s.confidence || 0) * 100) + '%' + (s.is_ood ? ' · OOD' : '') + '</small></span></button>';
    }

    function destroy() { destroyed = true; hideTip(); }
    M = { setData, onScan, destroy, alive: () => !destroyed && el.isConnected, getData: () => _data };
    return M;
  }

  // ── expanded · "Trust Core Sample" ────────────────────────────────────────
  function renderExpanded(el, data) {
    injectCSS();
    const d = data || (M && M.getData && M.getData()) || {};
    el.innerHTML =
      '<div class="ance">' +
      '<div class="ance-grid">' +
      '<div class="ance-col ance-strata-col"><h4>Trust strata</h4><div id="ance-strata"></div></div>' +
      '<div class="ance-col"><h4>Calibration</h4><div id="ance-calib"></div>' +
      '<h4>Confidence over time</h4><div id="ance-trend"></div></div>' +
      '<div class="ance-col"><h4>By disease</h4><div id="ance-disease"></div>' +
      '<h4>Needs review <span class="ance-rev">' + (d.needs_review || 0) + '</span></h4><div id="ance-review"></div></div>' +
      '</div></div>';
    // reuse compact strata as a tall static block
    const sc = el.querySelector('#ance-strata');
    const total = Math.max(1, d.total || 1);
    sc.innerHTML = '<div class="anc" style="height:360px"><div class="anc-body"><div class="anc-spine"></div><div class="anc-strata">' +
      (d.bands || []).map((b, i) => '<div class="anc-band anc-tex' + i + '" style="flex:' + Math.max(0.06, b.count / total) + ';--bc:' + BAND_COLOR[b.key] + '"><div class="anc-band-l"><span class="anc-band-name">' + BAND_LABEL[b.key] + '</span></div><div class="anc-band-mid"><b>' + b.count + '</b> ' + (b.top_disease ? esc(titleCase(b.top_disease)) : '') + '</div><div class="anc-band-act">' + BAND_ACTION[b.key] + '</div></div>').join('') +
      '</div></div></div>';
    renderCalib(el.querySelector('#ance-calib'), d.calibration);
    renderTrend(el.querySelector('#ance-trend'), d.over_time);
    renderDisease(el.querySelector('#ance-disease'), d.by_disease);
    renderReview(el.querySelector('#ance-review'), d.bands);
  }

  function renderCalib(host, cal) {
    if (!host) return; cal = cal || { mode: 'histogram', bins: [] };
    if (cal.mode === 'reliability') {
      const W = 280, H = 180, pad = 24;
      const pts = (cal.bins || []).filter(b => b.n).map(b => {
        const x = pad + ((b.lo + b.hi) / 2) * (W - 2 * pad);
        const y = H - pad - (b.accuracy || 0) * (H - 2 * pad);
        return { x, y, b };
      });
      const path = pts.map((p, i) => (i ? 'L' : 'M') + p.x.toFixed(1) + ' ' + p.y.toFixed(1)).join(' ');
      host.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="ance-svg"><line x1="' + pad + '" y1="' + (H - pad) + '" x2="' + (W - pad) + '" y2="' + pad + '" class="ance-diag"/>' +
        '<path d="' + path + '" class="ance-rel"/>' +
        pts.map(p => '<circle cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="3.5" class="ance-dot" data-tip="' + Math.round(p.b.confidence * 100) + '% stated → ' + Math.round(p.b.accuracy * 100) + '% right (n=' + p.b.n + ')"/>').join('') +
        '</svg><div class="ance-cap">stated vs actual · ECE ' + (cal.ece != null ? cal.ece.toFixed(3) : '·') + ' · n=' + cal.n + '</div>';
      wireDots(host);
    } else {
      // histogram fallback (always available)
      const bins = cal.bins || []; const max = Math.max(1, ...bins.map(b => b.count));
      host.innerHTML = '<div class="ance-hist">' + bins.map(b => {
        const h = Math.round(b.count / max * 100);
        const oodh = b.count ? Math.round(b.ood / b.count * h) : 0;
        return '<div class="ance-hbar" title="' + Math.round(b.lo * 100) + '–' + Math.round(b.hi * 100) + '%: ' + b.count + ' (' + b.ood + ' OOD)" style="height:' + h + '%"><i class="ance-hood" style="height:' + oodh + '%"></i></div>';
      }).join('') + '</div><div class="ance-cap">confidence distribution · ' + (cal.n || 0) + ' scans · red = OOD · (reliability curve unlocks with farmer feedback)</div>';
    }
  }

  function renderTrend(host, ot) {
    if (!host) return; ot = ot || {};
    const avg = (ot.avg || []).map(v => v == null ? null : v);
    const vals = avg.filter(v => v != null);
    if (!vals.length) { host.innerHTML = '<div class="an-ph an-ph-empty">no trend</div>'; return; }
    const W = 280, H = 90, n = avg.length;
    const pts = avg.map((v, i) => v == null ? null : { x: (i / (n - 1)) * W, y: H - (v) * H }).filter(Boolean);
    const path = pts.map((p, i) => (i ? 'L' : 'M') + p.x.toFixed(1) + ' ' + p.y.toFixed(1)).join(' ');
    host.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="ance-svg ance-trend"><path d="' + path + '" class="ance-rel"/>' +
      pts.map((p, i) => '<circle cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="2.5" class="ance-dot" data-tip="avg ' + Math.round((avg[i] || 0) * 100) + '%"/>').join('') +
      '</svg><div class="ance-cap">average confidence trend</div>';
    wireDots(host);
  }

  function renderDisease(host, list) {
    if (!host) return; list = (list || []).slice(0, 8);
    const max = Math.max(1, ...list.map(d => d.count));
    host.innerHTML = list.length ? list.map(d => '<button class="ance-drow" data-tip="' + esc(titleCase(d.disease)) + ' · avg ' + Math.round((d.avg_confidence || 0) * 100) + '% · OOD ' + Math.round((d.ood_rate || 0) * 100) + '%"><span class="ance-dnm">' + esc(titleCase(d.disease)) + '</span><span class="ance-dbar"><i style="width:' + Math.round(d.count / max * 100) + '%"></i></span><span class="ance-dconf">' + Math.round((d.avg_confidence || 0) * 100) + '%</span></button>').join('') : '<div class="an-ph an-ph-empty">no data</div>';
    wireDots(host);
  }

  function renderReview(host, bands) {
    if (!host) return;
    const specs = []; (bands || []).forEach(b => { if (b.key === 'uncertain' || b.key === 'ood') (b.specimens || []).forEach(s => specs.push(s)); });
    host.innerHTML = specs.length ? '<div class="anc-gal-grid ance-review-grid">' + specs.slice(0, 12).map(s => {
      const img = s.has_image ? '<img loading="lazy" src="/api/account/analytics/scan/' + encodeURIComponent(s.scan_uid) + '/image" alt="">' : '<span class="anc-gal-noimg"></span>';
      return '<button class="anc-gal-card bad" data-uid="' + esc(s.scan_uid) + '">' + img + '<span class="anc-gal-cap">' + esc(titleCase(s.diagnosis)) + '<small>' + Math.round((s.confidence || 0) * 100) + '%</small></span></button>';
    }).join('') + '</div>' : '<div class="an-ph an-ph-empty">nothing flagged — all confident</div>';
    host.querySelectorAll('[data-uid]').forEach(c => c.addEventListener('click', () => {
      const uid = c.getAttribute('data-uid');
      if (window.APIN.analyticsObservatory && window.APIN.analyticsObservatory.openInspector) window.APIN.analyticsObservatory.openInspector({ scan_uid: uid });
    }));
  }

  function wireDots(host) {
    host.querySelectorAll('[data-tip]').forEach(n => {
      n.addEventListener('mouseenter', e => showTip(e.clientX, e.clientY, n.getAttribute('data-tip')));
      n.addEventListener('mousemove', moveTip);
      n.addEventListener('mouseleave', hideTip);
    });
  }

  // ── tooltip ───────────────────────────────────────────────────────────────
  let tip = null;
  function tipEl() { if (!tip) { tip = document.createElement('div'); tip.className = 'anc-tip'; tip.style.display = 'none'; document.body.appendChild(tip); } return tip; }
  function showTip(x, y, html) { const t = tipEl(); t.innerHTML = html; t.style.display = 'block'; t.style.left = Math.min(x + 14, window.innerWidth - 230) + 'px'; t.style.top = (y + 14) + 'px'; }
  function moveTip(e) { if (tip && tip.style.display === 'block') { tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 230) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; } }
  function hideTip() { if (tip) tip.style.display = 'none'; }

  function injectCSS() {
    if (document.getElementById('anc-css')) return;
    const s = document.createElement('style'); s.id = 'anc-css';
    s.textContent = `
.anc{display:flex;flex-direction:column;gap:8px;height:100%;min-height:300px}
.anc-top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.anc-stat{font:12px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}.anc-stat b{color:var(--ink,#1a1612);font-weight:700}
.anc-review{color:#b3402f;font-weight:700}
.anc-seg{display:inline-flex;border:1px solid var(--paper-edge,#d8cdb2);border-radius:7px;overflow:hidden}
.anc-seg button{font:600 10px 'JetBrains Mono',monospace;border:0;background:var(--paper,#efe7d4);color:var(--ink-soft,#5b5446);padding:4px 9px;cursor:pointer}
.anc-seg button[aria-pressed=true]{background:var(--ink,#1a1612);color:#efe7d4}
.anc-body{flex:1;display:flex;gap:9px;min-height:240px}
.anc-spine{position:relative;width:14px;flex:none;border-radius:7px;overflow:visible}
.anc-spine-grad{position:absolute;inset:0;border-radius:7px;background:linear-gradient(to bottom,#3f8f5e,#6cbf94,#c98a2b,#8a8472,#b3402f)}
.anc-spine-tick{position:absolute;left:-2px;right:-2px;height:1px;background:rgba(26,22,18,.25)}
.anc-spine-avg{position:absolute;left:-3px;right:-3px;height:0;border-top:2px dashed var(--ink,#1a1612);transition:top .5s ease}
.anc-spine-avg b{position:absolute;left:18px;top:-8px;font:700 9px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);background:var(--paper,#efe7d4);padding:0 3px;border-radius:3px;white-space:nowrap}
.anc-strata{flex:1;display:flex;flex-direction:column;gap:2px;min-width:0}
.anc-band{position:relative;display:flex;align-items:center;gap:8px;min-height:26px;padding:4px 10px;border-radius:7px;background:color-mix(in srgb,var(--bc) 18%,var(--paper,#efe7d4));border-left:4px solid var(--bc);cursor:pointer;overflow:hidden;transition:transform .18s ease,opacity .18s ease,box-shadow .18s ease}
.anc-band:hover,.anc-band.on{transform:translateX(7px);box-shadow:-4px 3px 12px rgba(20,16,12,.16)}
.anc-band.dim{opacity:.4}
.anc-band.anc-hit{animation:ancHit .9s ease}
@keyframes ancHit{0%{box-shadow:0 0 0 0 var(--bc)}40%{box-shadow:0 0 0 4px color-mix(in srgb,var(--bc) 50%,transparent)}100%{box-shadow:none}}
.anc-band-empty{opacity:.45;cursor:default}.anc-band-empty:hover{transform:none;box-shadow:none}
.anc-band-l{flex:none;min-width:96px}
.anc-band-name{font:700 11px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.anc-band-rng{font:9px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.anc-band-mid{flex:1;min-width:0;font:11px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.anc-band-n{color:var(--ink,#1a1612);font-weight:700}.anc-band-zero{color:var(--ink-mute,#9a917d)}
.anc-band-act{flex:none;font:9.5px 'JetBrains Mono',monospace;color:var(--bc);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
/* texture roughens downward */
.anc-tex2{background-image:radial-gradient(color-mix(in srgb,var(--bc) 26%,transparent) .5px,transparent .5px);background-size:7px 7px}
.anc-tex3{background-image:radial-gradient(color-mix(in srgb,var(--bc) 34%,transparent) .7px,transparent .7px);background-size:6px 6px}
.anc-tex4{background-image:radial-gradient(color-mix(in srgb,var(--bc) 42%,transparent) 1px,transparent 1px);background-size:5px 5px}
.anc-tip{position:fixed;z-index:99999;pointer-events:none;max-width:230px;background:var(--ink,#1a1612);color:#efe7d4;font:11px 'JetBrains Mono',monospace;line-height:1.5;padding:7px 9px;border-radius:7px;box-shadow:0 6px 20px rgba(0,0,0,.3)}
.anc-tip b{color:#8fe0b4}.anc-tip i{color:#d9b08c;font-style:normal}.anc-tip-cta{color:#c9a85f}
/* gallery */
.anc-gal{padding:6px 2px}.anc-gal-head{font:600 13px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin-bottom:10px}
.anc-gal-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px}
.anc-gal-card{border:1px solid var(--paper-edge,#d8cdb2);border-radius:10px;overflow:hidden;background:var(--paper,#efe7d4);cursor:pointer;padding:0;text-align:left;transition:transform .15s ease,box-shadow .15s ease}
.anc-gal-card:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(20,16,12,.2)}
.anc-gal-card img{width:100%;height:96px;object-fit:cover;display:block}
.anc-gal-noimg{display:block;height:96px;background:var(--paper-deep,#e7dcc4)}
.anc-gal-cap{display:block;padding:6px 8px;font:600 10.5px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}.anc-gal-cap small{display:block;font-weight:400;color:var(--ink-mute,#9a917d)}
.anc-gal-card.bad{border-color:#d9a99a}.anc-gal-card.warn{border-color:#e0c98c}.anc-gal-card.ok{border-color:#a9d8bd}
.anc-gal-more{margin-top:10px;font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
/* expanded */
.ance-grid{display:grid;grid-template-columns:0.9fr 1.1fr 1.1fr;gap:16px}
.ance-col h4{font:600 12px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin:0 0 8px}.ance-col h4:not(:first-child){margin-top:16px}
.ance-rev{color:#b3402f}
.ance-svg{width:100%;height:auto;background:var(--paper-deep,#e7dcc4);border-radius:8px}
.ance-diag{stroke:var(--ink-mute,#9a917d);stroke-width:1;stroke-dasharray:3 3}
.ance-rel{fill:none;stroke:var(--accent-deep,#2d6a4f);stroke-width:2;stroke-linejoin:round}
.ance-dot{fill:var(--paper,#efe7d4);stroke:var(--accent-deep,#2d6a4f);stroke-width:1.6;cursor:pointer}
.ance-cap{font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:5px}
.ance-hist{display:flex;align-items:flex-end;gap:3px;height:150px;background:var(--paper-deep,#e7dcc4);border-radius:8px;padding:8px}
.ance-hbar{flex:1;background:var(--accent,#52b788);border-radius:2px 2px 0 0;position:relative;min-height:1px}
.ance-hood{position:absolute;left:0;right:0;bottom:0;background:#b3402f;border-radius:0 0 2px 2px}
.ance-drow{display:grid;grid-template-columns:1fr 80px 36px;align-items:center;gap:8px;width:100%;border:0;background:none;padding:4px;border-radius:6px;cursor:pointer;font:11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.ance-drow:hover{background:var(--paper,#efe7d4)}.ance-dnm{text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--ink-soft,#5b5446)}
.ance-dbar{height:7px;background:var(--paper-deep,#e7dcc4);border-radius:4px;overflow:hidden}.ance-dbar i{display:block;height:100%;background:var(--accent,#52b788);border-radius:4px}
.ance-dconf{text-align:right;font-weight:700;color:var(--ink-soft,#5b5446)}
.ance-review-grid{grid-template-columns:repeat(auto-fill,minmax(90px,1fr))}.ance-review-grid img{height:72px}
@media (max-width:900px){.ance-grid{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){.anc-band,.anc-spine-avg,.anc-gal-card{transition:none}.anc-band.anc-hit{animation:none}}
`;
    document.head.appendChild(s);
  }

  C.open = open; C.mount = mount; C.renderExpanded = renderExpanded;
  C.onScan = function (e) { if (M && M.onScan) M.onScan(e); };
  C.setData = function (d) { if (M && M.setData) M.setData(d); };
  C.alive = function () { return !!(M && M.alive && M.alive()); };
  window.APIN = window.APIN || {};
  window.APIN.analyticsConfidence = C;
})();
