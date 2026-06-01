/* 9.N.T34 · Client Distribution — "The Visitors' Log".
 *
 * A naturalist's log of who called the model. A live telegraph tape records
 * arrivals (each live request = a colored tick scrolling across); runtime lanes
 * fill with ink-wash bars; a roster lists the real callers (lib/version, p95,
 * region, last-seen). Everything is hoverable and reacts to live `request` SSE
 * events with no refresh.
 *
 * Registers window.APIN.analyticsClients = { mount, setData, onRequest,
 * renderExpanded, alive, destroy }.
 */
(function () {
  'use strict';
  const C = {};
  const FAM_COLOR = {
    'Python': '#3f7cc9', 'Node.js': '#52b788', 'Browser': '#c98a2b',
    'curl': '#7a8a6a', 'APIN-SDK': '#b6822a', 'Go': '#5fb0c4',
    'Java/Android': '#b3402f', 'Postman': '#e07b39', 'Wget': '#9a917d', 'Other': '#9a917d',
  };
  function famColor(f) { return FAM_COLOR[f] || '#9a917d'; }

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
  function fmt(n) { n = +n || 0; return Math.abs(n) >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n)); }
  function ms(v) { if (v == null) return '·'; return v >= 1000 ? (v / 1000).toFixed(1) + 's' : Math.round(v) + 'ms'; }
  function api(p) { try { return window.APIN.analyticsApi(p); } catch (e) { return Promise.reject(e); } }
  function ago(iso) { try { if (window.APIN.time && window.APIN.time.ago) return window.APIN.time.ago(iso); } catch (e) { } if (!iso) return ''; const d = Date.now() - new Date(String(iso).replace(' ', 'T')).getTime(); if (isNaN(d)) return ''; if (d < 6e4) return Math.max(1, d / 1e3 | 0) + 's'; if (d < 36e5) return (d / 6e4 | 0) + 'm'; if (d < 864e5) return (d / 36e5 | 0) + 'h'; return (d / 864e5 | 0) + 'd'; }

  function parseUa(ua) {
    const s = (ua || '').trim(); const low = s.toLowerCase();
    let fam = 'Other', lib = null, ver = null;
    const grab = (name) => { const m = low.match(new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '[/ ]v?([0-9][0-9a-z._-]*)')); return m ? m[1] : null; };
    if (low.includes('apin-sdk')) { fam = 'APIN-SDK'; lib = 'apin-sdk'; ver = grab('apin-sdk'); }
    else if (low.includes('python-requests')) { fam = 'Python'; lib = 'requests'; ver = grab('python-requests'); }
    else if (low.includes('python-urllib') || low.includes('urllib')) { fam = 'Python'; lib = 'urllib'; ver = grab('python-urllib') || grab('urllib'); }
    else if (low.includes('aiohttp')) { fam = 'Python'; lib = 'aiohttp'; ver = grab('aiohttp'); }
    else if (low.includes('httpx')) { fam = 'Python'; lib = 'httpx'; ver = grab('httpx'); }
    else if (low.includes('undici')) { fam = 'Node.js'; lib = 'undici'; ver = grab('undici'); }
    else if (low.includes('node-fetch')) { fam = 'Node.js'; lib = 'node-fetch'; ver = grab('node-fetch'); }
    else if (low.includes('axios')) { fam = 'Node.js'; lib = 'axios'; ver = grab('axios'); }
    else if (low.startsWith('got/')) { fam = 'Node.js'; lib = 'got'; ver = grab('got'); }
    else if (low.includes('okhttp')) { fam = 'Java/Android'; lib = 'okhttp'; ver = grab('okhttp'); }
    else if (low.includes('go-http-client')) { fam = 'Go'; lib = 'net/http'; ver = grab('go-http-client'); }
    else if (low.includes('postmanruntime')) { fam = 'Postman'; lib = 'postman'; ver = grab('postmanruntime'); }
    else if (low.includes('curl')) { fam = 'curl'; lib = 'curl'; ver = grab('curl'); }
    else if (low.includes('wget')) { fam = 'Wget'; lib = 'wget'; ver = grab('wget'); }
    else if (low.includes('mozilla') || low.includes('applewebkit')) {
      fam = 'Browser';
      if (low.includes('edg/')) { lib = 'Edge'; ver = grab('edg'); }
      else if (low.includes('firefox')) { lib = 'Firefox'; ver = grab('firefox'); }
      else if (low.includes('chrome') && !low.includes('chromium')) { lib = 'Chrome'; ver = grab('chrome'); }
      else if (low.includes('safari')) { lib = 'Safari'; ver = grab('version') || grab('safari'); }
      else { lib = 'Browser'; }
    }
    const label = (lib && ver) ? (lib + '/' + ver) : (lib || fam);
    return { family: fam, label };
  }

  let M = null;

  function mount(el, opts) {
    injectCSS();
    if (M) { try { M.destroy(); } catch (e) { } M = null; }
    M = make(el, opts || {});
    return M;
  }

  function make(el, opts) {
    let _data = null, destroyed = false, _hoverFam = null;
    el.innerHTML = shell();
    const host = el.querySelector('.acl');

    function shell() {
      return '<div class="acl">' +
        '<div class="acl-top"><span class="acl-stat" id="acl-stat">·</span></div>' +
        '<div class="acl-tape" id="acl-tape" aria-hidden="true"><i class="acl-tape-line"></i></div>' +
        '<div class="acl-bars" id="acl-bars"><div class="an-ph">loading&hellip;</div></div>' +
        '<div class="acl-roster" id="acl-roster"></div>' +
        '</div>';
    }

    function setData(d) { _data = d; render(); }

    function render() {
      if (!_data) return;
      const stat = host.querySelector('#acl-stat');
      if (stat) stat.innerHTML = '<b>' + fmt(_data.total || 0) + '</b> calls · <b>' + (_data.families || []).length + '</b> runtimes · ' + (_data.new_clients || 0) + ' new';
      renderBars(); renderRoster();
    }

    function renderBars() {
      const wrap = host.querySelector('#acl-bars'); if (!wrap) return;
      const fams = _data.families || []; const total = Math.max(1, _data.total || 1);
      wrap.innerHTML = fams.map(f => {
        const pct = Math.round(f.count / total * 100);
        const errPct = Math.round((f.error_rate || 0) * 100);
        const warn = errPct >= 10 ? ' acl-warn' : '';
        return '<button class="acl-bar' + warn + '" data-fam="' + esc(f.family) + '" style="--c:' + famColor(f.family) + '">' +
          '<span class="acl-bar-nm">' + esc(f.family) + '</span>' +
          '<span class="acl-bar-track"><i class="acl-bar-fill" style="width:' + pct + '%"></i><i class="acl-bar-err" style="width:' + Math.min(100, errPct) + '%"></i></span>' +
          '<span class="acl-bar-pct">' + pct + '%</span>' +
          '<span class="acl-bar-n">' + fmt(f.count) + '</span>' +
          '<span class="acl-bar-e' + (errPct >= 10 ? ' hot' : '') + '">err ' + errPct + '%</span>' +
          '</button>';
      }).join('') || '<div class="an-ph an-ph-empty">no clients yet</div>';
      wrap.querySelectorAll('.acl-bar[data-fam]').forEach(node => {
        const fam = node.getAttribute('data-fam');
        node.addEventListener('mouseenter', () => {
          _hoverFam = fam; wrap.querySelectorAll('.acl-bar').forEach(n => n.classList.toggle('dim', n !== node));
          famTip(node, fam);
          try { if (window.APIN.analyticsGeo && window.APIN.analyticsGeo.focusRegion) { /* geo highlight by region of this family */ const f = fams.find(x => x.family === fam); if (f && f.regions && f.regions[0]) window.APIN.analyticsGeo.focusRegion(f.regions[0].label); } } catch (e) { }
        });
        node.addEventListener('mousemove', moveTip);
        node.addEventListener('mouseleave', () => { _hoverFam = null; wrap.querySelectorAll('.acl-bar').forEach(n => n.classList.remove('dim')); hideTip(); });
        node.addEventListener('click', () => openAtlas(fam));
      });
    }

    function famTip(node, fam) {
      const f = (_data.families || []).find(x => x.family === fam); if (!f) return;
      const vers = (f.versions || []).slice(0, 5).map(v => v.label + ' ' + v.count).join('<br>');
      const eps = (f.top_endpoints || []).slice(0, 3).map(e => esc(e.path) + ' ' + e.count).join('<br>');
      const r = node.getBoundingClientRect();
      showTip(r.right, r.top + 6,
        '<b>' + esc(fam) + '</b> · ' + fmt(f.count) + ' calls<br>' +
        'p95 ' + ms(f.p95_latency) + ' · avg ' + ms(f.avg_latency) + ' · err ' + Math.round((f.error_rate || 0) * 100) + '%<br>' +
        '<i>versions</i><br>' + (vers || '·') + (eps ? '<br><i>top paths</i><br>' + eps : '') +
        '<br><span class="acl-tip-cta">click → Client Atlas</span>');
    }

    function renderRoster() {
      const wrap = host.querySelector('#acl-roster'); if (!wrap) return;
      const callers = (_data.callers || []).slice(0, 5);
      const max = Math.max(1, ...callers.map(c => c.count));
      wrap.innerHTML = '<div class="acl-roster-h">top callers<span>p95 · last · where</span></div>' +
        callers.map(c => '<button class="acl-row" data-label="' + esc(c.label) + '">' +
          '<span class="acl-row-dot" style="background:' + famColor(c.family) + '"></span>' +
          '<span class="acl-row-nm">' + esc(c.label) + '<i class="acl-row-bar" style="width:' + Math.round(c.count / max * 60) + 'px;background:' + famColor(c.family) + '"></i></span>' +
          '<span class="acl-row-n">' + fmt(c.count) + '</span>' +
          '<span class="acl-row-meta">' + ms(c.p95_latency) + ' · ' + (c.last_seen ? ago(c.last_seen) : '·') + ' · ' + (c.region || '·') + '</span>' +
          '</button>').join('');
      wrap.querySelectorAll('.acl-row[data-label]').forEach(node => {
        const label = node.getAttribute('data-label');
        const c = callers.find(x => x.label === label);
        node.addEventListener('mouseenter', () => { node.classList.add('on'); if (c) rowTip(node, c); });
        node.addEventListener('mousemove', moveTip);
        node.addEventListener('mouseleave', () => { node.classList.remove('on'); hideTip(); });
        node.addEventListener('click', () => openAtlas(c ? c.family : null));
      });
    }
    function rowTip(node, c) {
      const r = node.getBoundingClientRect();
      showTip(r.left, r.bottom + 4, '<b>' + esc(c.label) + '</b> · ' + esc(c.family) + '<br>' + fmt(c.count) + ' calls · p95 ' + ms(c.p95_latency) + ' · err ' + Math.round((c.error_rate || 0) * 100) + '%<br>last seen ' + (c.last_seen ? ago(c.last_seen) + ' ago' : '·') + ' · ' + (c.region || 'unknown'));
    }

    // ── live telegraph tape ─────────────────────────────────────────────────
    function tapeTick(fam) {
      const tape = host.querySelector('#acl-tape'); if (!tape) return;
      const t = document.createElement('i'); t.className = 'acl-tick'; t.style.setProperty('--c', famColor(fam));
      tape.appendChild(t);
      t.addEventListener('animationend', () => t.remove());
      while (tape.querySelectorAll('.acl-tick').length > 40) { const f = tape.querySelector('.acl-tick'); if (f) f.remove(); else break; }
    }

    function onRequest(e) {
      if (destroyed || !_data || !e) return;
      const p = parseUa(e.ua || '');
      const fam = p.family;
      _data.total = (_data.total || 0) + 1;
      const st = +e.status_code || 0;
      _data.status_mix = _data.status_mix || {}; _data.status_mix[st] = (_data.status_mix[st] || 0) + 1;
      let f = (_data.families || []).find(x => x.family === fam);
      if (!f) { f = { family: fam, count: 0, errors: 0, error_rate: 0, p95_latency: e.latency_ms, avg_latency: e.latency_ms, versions: [], top_endpoints: [], regions: [], last_seen: e.timestamp }; (_data.families = _data.families || []).push(f); }
      f.count++; if (st >= 400) f.errors = (f.errors || 0) + 1; f.error_rate = (f.errors || 0) / f.count; f.last_seen = e.timestamp || f.last_seen;
      // version bump
      const v = (f.versions || []).find(x => x.label === p.label); if (v) v.count++; else (f.versions = f.versions || []).push({ label: p.label, count: 1 });
      // caller bump
      let c = (_data.callers || []).find(x => x.label === p.label);
      if (!c) { c = { label: p.label, family: fam, count: 0, error_rate: 0, p95_latency: e.latency_ms, last_seen: e.timestamp, region: null }; (_data.callers = _data.callers || []).push(c); }
      c.count++; c.last_seen = e.timestamp || c.last_seen;
      (_data.families || []).sort((a, b) => b.count - a.count);
      (_data.callers || []).sort((a, b) => b.count - a.count);
      render(); tapeTick(fam);
      const node = host.querySelector('.acl-bar[data-fam="' + cssEsc(fam) + '"]');
      if (node) { node.classList.add('acl-hit'); setTimeout(() => node.classList.remove('acl-hit'), 700); }
      // pulse the shared live indicator handled by console_analytics
    }
    function cssEsc(s) { return String(s).replace(/"/g, '\\"'); }

    function destroy() { destroyed = true; hideTip(); }
    M = { setData, onRequest, destroy, alive: () => !destroyed && el.isConnected, getData: () => _data };
    return M;
  }

  // ── expanded · "Client Atlas" ─────────────────────────────────────────────
  function openAtlas(fam) {
    if (!(window.APIN && window.APIN.lightbox)) return;
    const expBtn = document.querySelector('[data-an-expand="clients"]');
    if (expBtn) { expBtn.click(); return; }   // route through console_analytics expand()
  }

  function renderExpanded(el, data) {
    injectCSS();
    const d = data || (M && M.getData && M.getData()) || {};
    el.innerHTML = '<div class="acle">' +
      '<div class="acle-grid">' +
      '<div class="acle-col"><h4>Runtime → version → endpoint</h4><div id="acle-drill"></div>' +
      '<h4>New vs returning</h4><div id="acle-nvr"></div><h4>Origins</h4><div id="acle-origins"></div></div>' +
      '<div class="acle-col acle-col-wide"><h4>Caller roster</h4><div id="acle-roster"></div></div>' +
      '<div class="acle-col"><h4>Latency distribution</h4><div id="acle-lat"></div>' +
      '<h4>Status codes</h4><div id="acle-status"></div><h4>Requests over time</h4><div id="acle-time"></div></div>' +
      '</div></div>';
    renderDrill(el.querySelector('#acle-drill'), d.families || []);
    renderNvr(el.querySelector('#acle-nvr'), d);
    renderOrigins(el.querySelector('#acle-origins'), d.origins || []);
    renderRosterFull(el.querySelector('#acle-roster'), d.callers || []);
    renderLatHist(el.querySelector('#acle-lat'), d.latency_hist || []);
    renderStatus(el.querySelector('#acle-status'), d.status_mix || {});
    renderTime(el.querySelector('#acle-time'), d.families || [], d.buckets || 24);
  }

  function renderDrill(host, fams) {
    if (!host) return;
    const total = Math.max(1, fams.reduce((a, f) => a + f.count, 0));
    host.innerHTML = fams.map(f => {
      const pct = Math.round(f.count / total * 100);
      const vers = (f.versions || []).slice(0, 6).map(v => '<span class="acle-chip" data-tip="' + esc(v.label) + ' · ' + v.count + ' calls">' + esc(v.label) + ' <b>' + v.count + '</b></span>').join('');
      const eps = (f.top_endpoints || []).slice(0, 4).map(e => '<span class="acle-ep" data-tip="' + esc(e.path) + ' · ' + e.count + '">' + esc(e.path) + '</span>').join('');
      return '<div class="acle-fam" style="--c:' + famColor(f.family) + '"><div class="acle-fam-h" data-tip="' + esc(f.family) + ' · ' + fmt(f.count) + ' · p95 ' + ms(f.p95_latency) + ' · err ' + Math.round((f.error_rate || 0) * 100) + '%"><b>' + esc(f.family) + '</b><i>' + pct + '%</i></div>' +
        '<div class="acle-bar2"><i style="width:' + pct + '%"></i></div>' +
        '<div class="acle-chips">' + vers + '</div><div class="acle-eps">' + eps + '</div></div>';
    }).join('');
    wireTips(host);
  }
  function renderNvr(host, d) {
    if (!host) return; const nw = d.new_clients || 0, rt = d.returning_clients || 0, tot = Math.max(1, nw + rt);
    host.innerHTML = '<div class="acle-nvr"><div class="acle-nvr-bar"><i class="nw" style="width:' + (nw / tot * 100) + '%" data-tip="new clients ' + nw + '"></i><i class="rt" style="width:' + (rt / tot * 100) + '%" data-tip="returning ' + rt + '"></i></div>' +
      '<div class="acle-nvr-lab"><span>◐ ' + nw + ' new</span><span>● ' + rt + ' returning</span></div></div>';
    wireTips(host);
  }
  function renderOrigins(host, origins) {
    if (!host) return; const max = Math.max(1, ...origins.map(o => o.count));
    host.innerHTML = origins.length ? origins.map(o => '<button class="acle-org" data-region="' + esc(o.label) + '" data-tip="' + esc(o.label) + ' · ' + fmt(o.count) + ' calls"><span>' + esc(o.label) + '</span><i style="width:' + Math.round(o.count / max * 100) + '%"></i><b>' + fmt(o.count) + '</b></button>').join('') : '<div class="an-ph an-ph-empty">·</div>';
    host.querySelectorAll('[data-region]').forEach(n => n.addEventListener('click', () => { try { window.APIN.analyticsFocus && window.APIN.analyticsFocus({ region: n.getAttribute('data-region') }); } catch (e) { } }));
    wireTips(host);
  }
  function renderRosterFull(host, callers) {
    if (!host) return; const max = Math.max(1, ...callers.map(c => c.count));
    host.innerHTML = '<table class="acle-tbl"><thead><tr><th>client</th><th>calls</th><th>err</th><th>p95</th><th>last</th><th>where</th></tr></thead><tbody>' +
      callers.map(c => '<tr data-tip="' + esc(c.label) + ' · ' + esc(c.family) + '"><td><span class="acle-rdot" style="background:' + famColor(c.family) + '"></span>' + esc(c.label) + '<i class="acle-rbar" style="width:' + Math.round(c.count / max * 50) + 'px;background:' + famColor(c.family) + '"></i></td>' +
        '<td>' + fmt(c.count) + '</td><td' + ((c.error_rate || 0) >= 0.1 ? ' class="hot"' : '') + '>' + Math.round((c.error_rate || 0) * 100) + '%</td><td>' + ms(c.p95_latency) + '</td><td>' + (c.last_seen ? ago(c.last_seen) : '·') + '</td><td>' + (c.region || '·') + '</td></tr>').join('') +
      '</tbody></table>';
    wireTips(host);
  }
  function renderLatHist(host, hist) {
    if (!host) return; const max = Math.max(1, ...hist.map(h => h.count));
    const lab = (lo, hi) => hi >= 1e9 ? (lo / 1000) + 's+' : (hi < 1000 ? hi + 'ms' : (hi / 1000) + 's');
    host.innerHTML = '<div class="acle-hist">' + hist.map(h => '<div class="acle-hbar" data-tip="' + lab(h.lo, h.hi) + ': ' + fmt(h.count) + ' calls" style="height:' + Math.round(h.count / max * 100) + '%"></div>').join('') + '</div><div class="acle-hist-x"><span>0</span><span>1s</span><span>10s+</span></div>';
    wireTips(host);
  }
  function renderStatus(host, mix) {
    if (!host) return; const entries = Object.entries(mix).sort((a, b) => b[1] - a[1]); const tot = Math.max(1, entries.reduce((a, e) => a + e[1], 0));
    const col = (s) => { s = +s; return s < 300 ? '#52b788' : s < 400 ? '#6cbf94' : s < 500 ? '#c98a2b' : '#b3402f'; };
    host.innerHTML = '<div class="acle-status">' + entries.map(([s, n]) => '<div class="acle-srow" data-tip="' + s + ' · ' + fmt(n) + ' (' + Math.round(n / tot * 100) + '%)"><span class="acle-scode" style="color:' + col(s) + '">' + s + '</span><span class="acle-sbar"><i style="width:' + Math.round(n / tot * 100) + '%;background:' + col(s) + '"></i></span><b>' + fmt(n) + '</b></div>').join('') + '</div>';
    wireTips(host);
  }
  function renderTime(host, fams, nb) {
    if (!host) return;
    const top = fams.slice(0, 5);
    host.innerHTML = top.map(f => {
      const s = f.series || []; const max = Math.max(1, ...s);
      const W = 220, H = 30, n = s.length || 1;
      const path = s.map((v, i) => (i ? 'L' : 'M') + ((i / (n - 1)) * W).toFixed(1) + ' ' + (H - v / max * H).toFixed(1)).join(' ');
      return '<div class="acle-trow" data-tip="' + esc(f.family) + ' · ' + fmt(f.count) + ' calls"><span class="acle-tnm" style="color:' + famColor(f.family) + '">' + esc(f.family) + '</span>' +
        '<svg viewBox="0 0 ' + W + ' ' + H + '" class="acle-spark"><path d="' + path + '" style="stroke:' + famColor(f.family) + '"/></svg></div>';
    }).join('');
    wireTips(host);
  }

  function wireTips(host) {
    host.querySelectorAll('[data-tip]').forEach(n => {
      n.addEventListener('mouseenter', e => showTip(e.clientX, e.clientY, n.getAttribute('data-tip')));
      n.addEventListener('mousemove', moveTip);
      n.addEventListener('mouseleave', hideTip);
    });
  }

  // ── tooltip ───────────────────────────────────────────────────────────────
  let tip = null;
  function tipEl() { if (!tip) { tip = document.createElement('div'); tip.className = 'acl-tip'; tip.style.display = 'none'; document.body.appendChild(tip); } return tip; }
  function showTip(x, y, html) { const t = tipEl(); t.innerHTML = html; t.style.display = 'block'; t.style.left = Math.min(x + 14, window.innerWidth - 250) + 'px'; t.style.top = (y + 14) + 'px'; }
  function moveTip(e) { if (tip && tip.style.display === 'block') { tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 250) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; } }
  function hideTip() { if (tip) tip.style.display = 'none'; }

  function injectCSS() {
    if (document.getElementById('acl-css')) return;
    const s = document.createElement('style'); s.id = 'acl-css';
    s.textContent = `
.acl{display:flex;flex-direction:column;gap:8px;height:100%;min-height:300px}
.acl-top{display:flex;align-items:center;justify-content:space-between}
.acl-stat{font:12px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}.acl-stat b{color:var(--ink,#1a1612);font-weight:700}
.acl-tape{position:relative;height:18px;border-radius:6px;background:var(--paper-deep,#e7dcc4);overflow:hidden;flex:none}
.acl-tape-line{position:absolute;left:0;right:0;top:50%;height:1px;background:var(--paper-edge,#cfc2a3)}
.acl-tick{position:absolute;right:0;top:3px;width:2px;height:12px;border-radius:1px;background:var(--c,#52b788);animation:aclTape 3.2s linear forwards}
@keyframes aclTape{from{right:0;opacity:1}to{right:100%;opacity:0}}
.acl-bars{display:flex;flex-direction:column;gap:5px}
.acl-bar{display:grid;grid-template-columns:64px 1fr 34px 44px 58px;align-items:center;gap:8px;width:100%;border:0;background:none;padding:3px 4px;border-radius:7px;cursor:pointer;transition:background .12s,opacity .15s}
.acl-bar:hover,.acl-bar.on{background:var(--paper-deep,#e7dcc4)}.acl-bar.dim{opacity:.38}
.acl-bar.acl-hit .acl-bar-fill{box-shadow:0 0 0 2px color-mix(in srgb,var(--c) 50%,transparent)}
.acl-bar-nm{text-align:left;font:600 11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.acl-bar-track{position:relative;height:11px;background:var(--paper-deep,#e7dcc4);border-radius:6px;overflow:hidden}
.acl-bar-fill{position:absolute;left:0;top:0;bottom:0;background:var(--c,#52b788);border-radius:6px;transition:width .5s cubic-bezier(.2,.9,.2,1.2)}
.acl-bar-err{position:absolute;right:0;top:0;bottom:0;background:repeating-linear-gradient(45deg,#b3402f,#b3402f 2px,transparent 2px,transparent 4px);opacity:.55;border-radius:0 6px 6px 0}
.acl-bar-pct{text-align:right;font:600 10px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}
.acl-bar-n{text-align:right;font:700 11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.acl-bar-e{text-align:right;font:9px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}.acl-bar-e.hot{color:#b3402f;font-weight:700}
.acl-roster{margin-top:2px;border-top:1px solid var(--paper-edge,#d8cdb2);padding-top:6px}
.acl-roster-h{display:flex;justify-content:space-between;font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.acl-row{display:grid;grid-template-columns:10px 1fr 40px 1.2fr;align-items:center;gap:7px;width:100%;border:0;background:none;padding:3px 4px;border-radius:6px;cursor:pointer;text-align:left;transition:background .12s}
.acl-row:hover,.acl-row.on{background:var(--paper-deep,#e7dcc4)}
.acl-row-dot{width:8px;height:8px;border-radius:50%}
.acl-row-nm{position:relative;font:11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-bottom:3px}
.acl-row-bar{position:absolute;left:0;bottom:0;height:2px;border-radius:1px;opacity:.6}
.acl-row-n{text-align:right;font:700 11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.acl-row-meta{text-align:right;font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.acl-tip,.acl-tip *{box-sizing:border-box}
.acl-tip{position:fixed;z-index:99999;pointer-events:none;max-width:250px;background:var(--ink,#1a1612);color:#efe7d4;font:11px 'JetBrains Mono',monospace;line-height:1.5;padding:7px 9px;border-radius:7px;box-shadow:0 6px 20px rgba(0,0,0,.3)}
.acl-tip b{color:#8fe0b4}.acl-tip i{color:#d9b08c;font-style:normal}.acl-tip-cta{color:#c9a85f}
/* expanded */
.acle-grid{display:grid;grid-template-columns:1fr 1.1fr 1fr;gap:16px}
.acle-col h4{font:600 12px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin:0 0 8px}.acle-col h4:not(:first-child){margin-top:16px}
.acle-fam{margin-bottom:10px}.acle-fam-h{display:flex;justify-content:space-between;font:11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}.acle-fam-h b{font-weight:700}.acle-fam-h i{color:var(--ink-soft,#5b5446)}
.acle-bar2{height:7px;background:var(--paper-deep,#e7dcc4);border-radius:4px;overflow:hidden;margin:3px 0}.acle-bar2 i{display:block;height:100%;background:var(--c,#52b788);border-radius:4px}
.acle-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:3px}
.acle-chip{font:9.5px 'JetBrains Mono',monospace;background:var(--paper-deep,#e7dcc4);border-radius:5px;padding:2px 6px;color:var(--ink-soft,#5b5446);cursor:default}.acle-chip b{color:var(--ink,#1a1612)}
.acle-eps{display:flex;flex-wrap:wrap;gap:4px;margin-top:3px}.acle-ep{font:9px 'JetBrains Mono',monospace;color:var(--accent-deep,#2d6a4f);cursor:default}
.acle-nvr-bar{display:flex;height:14px;border-radius:7px;overflow:hidden;background:var(--paper-deep,#e7dcc4)}.acle-nvr-bar .nw{background:var(--accent,#52b788)}.acle-nvr-bar .rt{background:var(--ochre,#b6822a)}
.acle-nvr-lab{display:flex;justify-content:space-between;font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:4px}
.acle-org{display:grid;grid-template-columns:1fr 60px 36px;align-items:center;gap:7px;width:100%;border:0;background:none;padding:3px;border-radius:6px;cursor:pointer;font:10.5px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.acle-org:hover{background:var(--paper-deep,#e7dcc4)}.acle-org i{height:6px;background:var(--accent,#52b788);border-radius:3px}.acle-org b{text-align:right;color:var(--ink-soft,#5b5446)}
.acle-tbl{width:100%;border-collapse:collapse;font:10.5px 'JetBrains Mono',monospace}
.acle-tbl th{text-align:right;font-weight:600;color:var(--ink-mute,#9a917d);padding:4px 6px;border-bottom:1px solid var(--paper-edge,#d8cdb2)}.acle-tbl th:first-child{text-align:left}
.acle-tbl td{text-align:right;padding:4px 6px;border-bottom:1px solid var(--paper-edge,#e3d9c0);color:var(--ink,#1a1612)}.acle-tbl td:first-child{text-align:left;position:relative}
.acle-tbl tr:hover td{background:var(--paper-deep,#e7dcc4)}.acle-tbl td.hot{color:#b3402f;font-weight:700}
.acle-rdot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}.acle-rbar{position:absolute;left:18px;bottom:1px;height:2px;border-radius:1px;opacity:.5}
.acle-hist{display:flex;align-items:flex-end;gap:3px;height:120px;background:var(--paper-deep,#e7dcc4);border-radius:8px;padding:8px}
.acle-hbar{flex:1;background:var(--accent,#52b788);border-radius:2px 2px 0 0;min-height:1px;transition:height .3s}
.acle-hist-x{display:flex;justify-content:space-between;font:9px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:3px}
.acle-srow{display:grid;grid-template-columns:36px 1fr 40px;align-items:center;gap:8px;padding:2px 0}
.acle-scode{font:700 11px 'JetBrains Mono',monospace}.acle-sbar{height:8px;background:var(--paper-deep,#e7dcc4);border-radius:4px;overflow:hidden}.acle-sbar i{display:block;height:100%;border-radius:4px}.acle-srow b{text-align:right;font:11px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}
.acle-trow{display:flex;align-items:center;gap:8px;margin-bottom:4px}.acle-tnm{flex:none;width:64px;font:9.5px 'JetBrains Mono',monospace}
.acle-spark{flex:1;height:30px}.acle-spark path{fill:none;stroke-width:1.6}
.acle-col-wide{overflow:auto}
@media (max-width:900px){.acle-grid{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){.acl-tick{animation-duration:.6s}.acl-bar-fill{transition:none}}
`;
    document.head.appendChild(s);
  }

  C.mount = mount; C.renderExpanded = renderExpanded;
  C.onRequest = function (e) { if (M && M.onRequest) M.onRequest(e); };
  C.setData = function (d) { if (M && M.setData) M.setData(d); };
  C.alive = function () { return !!(M && M.alive && M.alive()); };
  window.APIN = window.APIN || {};
  window.APIN.analyticsClients = C;
})();
