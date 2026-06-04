// console_admin.js — APIN Admin Console shell controller (Phase A)
//
// Responsibilities (Phase A):
//   • dismiss the boot/arrival overlay (with a hard failsafe so a user is
//     NEVER trapped behind it)
//   • play the arrival handshake animation (skippable, reduced-motion aware)
//   • fetch identity from /api/account/admin/whoami (defense-in-depth: the
//     server page route already gated admins; this confirms + fills name/avatar
//     and redirects belt-and-suspenders if the server ever served us in error)
//   • section switching across the nav rail (no page reloads)
//   • a live UTC clock
//
// CSP: served from /static (script-src 'self'); no inline handlers, no eval.

(function () {
  'use strict';

  var REDUCED = false;
  try { REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}

  var boot = document.getElementById('adm-boot');

  function dismissBoot() {
    if (!boot) return;
    if (boot.classList.contains('gone')) return;
    boot.classList.add('gone');
    setTimeout(function () { if (boot && boot.parentNode) boot.parentNode.removeChild(boot); }, 650);
  }

  // FAILSAFE #1 (JS-level): whatever happens below, the overlay is gone within
  // 3s. (FAILSAFE #2 is the pure-CSS auto-hide in the page's critical <style>.)
  var failsafe = setTimeout(dismissBoot, 3000);

  // Escapes &, <, >, ", and ' — the single-quote matters because future admin
  // sections will render server data, and 'unsafe-inline' in the console
  // style-src means an unescaped apostrophe in an attribute context is a sink.
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' })[c];
    });
  }

  // ── identity (whoami) ───────────────────────────────────────────────────
  function applyIdentity(d) {
    if (!d) return;
    var name = d.display_name || d.username || 'administrator';
    var who = document.getElementById('adm-who');
    if (who) who.innerHTML = esc(name) + '<small>ADMIN</small>';
    var av = document.getElementById('adm-av');
    if (av) {
      // Same pressed-leaf identity the rest of the app uses (not a gradient blob).
      var seed = (typeof d.pressed_leaf_seed === 'number') ? d.pressed_leaf_seed : 0;
      if (window.APIN_pressedLeaf) av.innerHTML = window.APIN_pressedLeaf(null, seed);
      else av.textContent = (String(name).trim()[0] || 'A').toUpperCase();
    }
    var bn = document.getElementById('adm-boot-name');
    if (bn) bn.textContent = name;
    var hello = document.getElementById('adm-hello');
    if (hello) hello.textContent = name;
    try { window.__ADMIN_SELF_ID__ = d.user_id; } catch (e) {}
  }

  function loadIdentity() {
    return fetch('/api/account/admin/whoami', {
      credentials: 'same-origin', headers: { 'Accept': 'application/json' },
    }).then(function (r) { return r.json(); }).then(function (j) {
      var d = (j && j.data) ? j.data : {};
      // Belt-and-suspenders: if the API says we are NOT admin, leave — even
      // though the page route should never have served us. The server remains
      // the real boundary; this just keeps the client honest.
      if (d && d.is_admin === false) { window.location.replace('/account/api/dashboard'); return null; }
      return d;
    }).catch(function () { return null; });   // network hiccup → stay (server already gated)
  }

  // ── cinematic arrival (R2) ──────────────────────────────────────────────
  // The CSS sequence is driven entirely by the .run class (kicker → ADMIN
  // CONSOLE reveal → welcome → progress bar, ≈2.0s). We add .run, hold briefly,
  // then dissolve so the bento stagger assembles behind it.
  function playArrival() {
    if (REDUCED) { clearTimeout(failsafe); dismissBoot(); return; }
    if (boot) boot.classList.add('run');
    setTimeout(function () { clearTimeout(failsafe); dismissBoot(); }, 2150);
  }

  // skip the arrival on any click or Esc
  if (boot) boot.addEventListener('click', function () { clearTimeout(failsafe); dismissBoot(); });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && boot && !boot.classList.contains('gone')) {
      clearTimeout(failsafe); dismissBoot();
    }
  });

  // ── section switching ───────────────────────────────────────────────────
  var titleEl = document.getElementById('adm-title');
  var navs = Array.prototype.slice.call(document.querySelectorAll('.adm-nav[data-sec]'));
  var secs = Array.prototype.slice.call(document.querySelectorAll('.adm-section[data-sec]'));

  function show(sec) {
    if (!document.querySelector('.adm-section[data-sec="' + sec + '"]')) return;
    navs.forEach(function (n) {
      n.setAttribute('aria-current', n.getAttribute('data-sec') === sec ? 'true' : 'false');
    });
    secs.forEach(function (s) {
      s.classList.toggle('active', s.getAttribute('data-sec') === sec);
    });
    var active = document.querySelector('.adm-nav[data-sec="' + sec + '"]');
    if (active && titleEl) titleEl.textContent = active.textContent.trim();
    try { history.replaceState(null, '', '#' + sec); } catch (e) {}
    var main = document.getElementById('adm-main'); if (main) main.scrollTop = 0;
    // Lazy section loaders — deferred so module-scope state (uState) is ready.
    setTimeout(function () {
      if (sec === 'users' && typeof loadUsers === 'function' && !uState.loaded) loadUsers(true);
      else if (sec === 'pulse' && typeof loadPulse === 'function') { loadPulse(); loadSignups(plWindow); loadFeed(); }
      else if (sec === 'database' && typeof loadDbTables === 'function' && !dbState.loaded) loadDbTables();
      // section-scoped widget modules: dispose the inactive ones now, then mount
      // the active one. On a COLD/HARD reload the widget module scripts can execute
      // AFTER this runs (script load-order race) → window.ADM_TRAFFIC is undefined →
      // the section silently never mounts and shows BLANK ("slow reload" symptom).
      // ensureWidgetMount() retries until the module arrives (or the user leaves).
      if (window.ADM_TRAFFIC && sec !== 'traffic') window.ADM_TRAFFIC.dispose();
      if (window.ADM_GEO && sec !== 'geo') window.ADM_GEO.dispose();
      ensureWidgetMount(sec);
    }, 0);
  }

  // Mount the active section's widget module as soon as it exists. Robust to the
  // script load-order race on cold reloads; aborts if the user navigates away.
  function ensureWidgetMount(sec) {
    var want = sec === 'traffic' ? 'ADM_TRAFFIC' : (sec === 'geo' ? 'ADM_GEO' : null);
    if (!want) return;
    var tries = 0;
    (function attempt() {
      if ((location.hash || '').replace('#', '') !== sec) return;   // navigated away — abort
      var mod = window[want];
      if (mod && typeof mod.mount === 'function') { try { mod.mount(); } catch (e) {} return; }
      if (tries++ < 60) setTimeout(attempt, 50);                    // up to ~3s for the script to load
    })();
  }

  navs.forEach(function (n) {
    n.addEventListener('click', function () { show(n.getAttribute('data-sec')); });
    n.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(n.getAttribute('data-sec')); }
    });
  });

  var initial = (location.hash || '').replace('#', '');
  if (initial) show(initial);

  // ── live clock — DEVICE-LOCAL time (not UTC) ────────────────────────────
  var clk = document.getElementById('adm-clock');
  function pad2(n) { return (n < 10 ? '0' : '') + n; }
  function localTz(d) {
    try {
      var s = d.toLocaleTimeString('en-US', { timeZoneName: 'short' });
      var m = s.match(/[A-Z]{2,5}$/); if (m) return m[0];
    } catch (e) {}
    var off = -d.getTimezoneOffset(), sign = off >= 0 ? '+' : '-', a = Math.abs(off);
    return 'GMT' + sign + Math.floor(a / 60) + (a % 60 ? ':' + pad2(a % 60) : '');
  }
  function tick() {
    if (!clk) return;
    try {
      var d = new Date();
      clk.textContent = pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds()) + ' ' + localTz(d);
    } catch (e) {}
  }
  tick(); setInterval(tick, 1000);

  // ── command palette (stub for Phase A) ──────────────────────────────────
  function cmd() { /* full ⌘K palette arrives in the polish pass */ }
  var cmdBtn = document.getElementById('adm-cmd');
  if (cmdBtn) cmdBtn.addEventListener('click', cmd);
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) { e.preventDefault(); cmd(); }
  });

  // ── admin data layer ────────────────────────────────────────────────────
  function admCsrf() {
    var m = document.querySelector('meta[name="csrf-token"]');
    return m ? (m.getAttribute('content') || '') : '';
  }
  // Admin GET reads are CSRF-gated server-side; always send the header.
  function adminFetch(path) {
    return fetch(path, {
      credentials: 'same-origin',
      headers: { 'Accept': 'application/json', 'X-Console-Csrf': admCsrf() },
    }).then(function (r) { return r.json(); })
      .then(function (j) { return (j && j.ok) ? j.data : null; })
      .catch(function () { return null; });
  }
  function fmtInt(n) { try { return Number(n || 0).toLocaleString('en-US'); } catch (e) { return String(n); } }

  function animateNum(el, to) {
    if (!el) return;
    to = Number(to || 0);
    if (REDUCED) { el.textContent = fmtInt(to); el.setAttribute('data-num', String(to)); return; }
    var from = Number(el.getAttribute('data-num') || 0), t0 = null, dur = 900;
    function step(ts) {
      if (t0 === null) t0 = ts;
      var p = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmtInt(Math.round(from + (to - from) * e));
      if (p < 1) requestAnimationFrame(step); else el.setAttribute('data-num', String(to));
    }
    requestAnimationFrame(step);
  }

  function setGauge(errRate) {
    var ring = document.getElementById('pl-gauge'), val = document.getElementById('pl-error');
    if (!ring) return;
    var er = Number(errRate || 0);
    var color = er < 1 ? 'var(--accent)' : (er < 5 ? 'var(--ochre)' : 'var(--crimson)');
    ring.style.setProperty('--g', String(Math.max(0, 100 - er)));   // success arc
    ring.style.setProperty('--g-color', color);
    if (val) val.textContent = (Math.round(er * 10) / 10) + '%';
  }

  // ── Overview (F2) — every metric is window-aware + each tile drills in ──
  var plData = null;
  var plWindow = 'all';
  var PL_WIN_LABEL = { '24h': 'last 24h', '7d': 'last 7 days', '30d': 'last 30 days', 'all': 'lifetime' };
  function setText(id, t) { var e = document.getElementById(id); if (e) e.textContent = t; }

  function sparkLabel(t) {
    if (!t) return '';
    if (t.length >= 13) return t.slice(11, 13) + ':00 · ' + t.slice(5, 10);  // hourly bucket
    try { return new Date(t + 'T00:00:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric' }); }
    catch (e) { return t; }
  }

  // signups area chart — zero-filled, window-aware, hoverable
  function renderSignups(series) {
    var svg = document.getElementById('pl-signups'); if (!svg) return;
    var W = 600, H = 80, pad = 8;
    var pts = (series || []).map(function (d) { return Number(d.c || 0); });
    if (!pts.length) { svg.innerHTML = ''; return; }
    var max = Math.max(1, Math.max.apply(null, pts)), n = pts.length;
    var xs = function (i) { return pad + (W - 2 * pad) * (n === 1 ? 0.5 : i / (n - 1)); };
    var ys = function (v) { return H - pad - (H - 2 * pad) * (v / max); };
    var line = pts.map(function (v, i) { return (i ? 'L' : 'M') + xs(i).toFixed(1) + ' ' + ys(v).toFixed(1); }).join(' ');
    var area = 'M' + xs(0).toFixed(1) + ' ' + (H - pad) + ' ' + line.slice(1) + ' L' + xs(n - 1).toFixed(1) + ' ' + (H - pad) + ' Z';
    var bw = (W - 2 * pad) / Math.max(1, n), cols = '';
    for (var i = 0; i < n; i++) cols += '<rect class="hv" x="' + (xs(i) - bw / 2).toFixed(1) + '" y="0" width="' + bw.toFixed(1) + '" height="' + H + '" fill="transparent" data-i="' + i + '"/>';
    svg.innerHTML = '<path class="ar" d="' + area + '"/><path class="ln" d="' + line + '"/>'
      + '<circle class="hd" id="pl-spark-dot" r="3.5"/>' + cols;
    svg._series = series; svg._xs = xs; svg._ys = ys; svg._pts = pts;
  }
  var signupsCache = {};
  function loadSignups(win) {
    function paint(series) {
      renderSignups(series);
      setText('pl-signups-win', PL_WIN_LABEL[win] || win);
      var tot = series.reduce(function (a, b) { return a + (b.c || 0); }, 0);
      setText('pl-signups-sub', fmtInt(tot) + ' signups · ' + (PL_WIN_LABEL[win] || win));
    }
    if (signupsCache[win]) { paint(signupsCache[win]); return; }
    adminFetch('/api/account/admin/metric/users?window=' + win).then(function (d) {
      if (!d) return; signupsCache[win] = d.series || []; paint(signupsCache[win]);
    });
  }
  (function initSignupsHover() {
    var svg = document.getElementById('pl-signups'), tip = document.getElementById('pl-spark-tip');
    if (!svg || !tip) return;
    svg.addEventListener('mouseover', function (e) {
      var hv = e.target.closest ? e.target.closest('.hv') : null; if (!hv) return;
      var i = parseInt(hv.getAttribute('data-i'), 10), s = svg._series; if (!s || !s[i]) return;
      var dot = document.getElementById('pl-spark-dot'), cx = svg._xs(i), cy = svg._ys(svg._pts[i]);
      if (dot) { dot.setAttribute('cx', cx); dot.setAttribute('cy', cy); dot.style.opacity = '1'; }
      var r = svg.getBoundingClientRect();
      tip.innerHTML = '<b>' + fmtInt(s[i].c) + '</b> signup' + (s[i].c === 1 ? '' : 's') + ' · ' + esc(sparkLabel(s[i].t));
      tip.style.left = (cx / 600 * r.width) + 'px'; tip.style.top = (cy / 80 * r.height) + 'px'; tip.hidden = false;
    });
    svg.addEventListener('mouseout', function (e) {
      if (e.target.closest && e.target.closest('.hv')) {
        tip.hidden = true; var dot = document.getElementById('pl-spark-dot'); if (dot) dot.style.opacity = '0';
      }
    });
  })();

  function renderPulse() {
    var d = plData; if (!d) return;
    var w = (d.windows && d.windows[plWindow]) ||
      { requests: 0, errors: 0, error_rate: 0, active_users: 0, keys_used: 0, new_users: 0, inferences: 0 };
    var lbl = PL_WIN_LABEL[plWindow] || plWindow;
    // requests (window)
    animateNum(document.getElementById('pl-requests'), w.requests);
    setText('pl-req-win', lbl);
    setText('pl-requests-sub', plWindow === 'all' ? 'org-wide · all keys · all time' : ('org-wide · ' + lbl));
    // error rate (window)
    setGauge(w.error_rate);
    setText('pl-error-win', lbl); setText('pl-error-errs', fmtInt(w.errors)); setText('pl-error-reqs', fmtInt(w.requests));
    setText('pl-error-sub', w.requests ? (fmtInt(w.errors) + ' of ' + fmtInt(w.requests) + ' requests · ' + lbl) : ('no requests in ' + lbl));
    // total users (roster) + windowed delta
    animateNum(document.getElementById('pl-users'), d.total_users);
    var us = document.getElementById('pl-users-sub');
    if (us) us.innerHTML = '<span class="up">▲</span> ' + fmtInt(w.new_users) + ' new · ' + lbl + ' · ' + fmtInt(d.admins) + ' admin';
    // active keys = ENABLED (status='active', current state); the windowed
    // "in use" count (keys that actually sent traffic) lives in the sub so the
    // headline no longer conflates "enabled" with "used".
    animateNum(document.getElementById('pl-keys'), d.active_keys);
    setText('pl-keys-win', 'enabled');
    setText('pl-keys-sub', fmtInt(w.keys_used) + ' in use · of ' + fmtInt(d.total_keys) + ' minted · ' + lbl);
    // inferences (window)
    animateNum(document.getElementById('pl-inferences'), w.inferences);
    setText('pl-inf-win', lbl); setText('pl-inf-sub', 'predictions · ' + lbl);
    // active users (window)
    animateNum(document.getElementById('pl-active'), w.active_users);
    setText('pl-active-win', lbl); setText('pl-active-sub', 'unique callers · ' + lbl);
  }

  function loadPulse() {
    adminFetch('/api/account/admin/overview').then(function (d) {
      if (!d) return; plData = d; renderPulse();
    });
  }

  (function initPulseWindow() {
    var seg = document.getElementById('pl-window');
    if (!seg) return;
    seg.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('button[data-w]') : null;
      if (!b) return;
      plWindow = b.getAttribute('data-w');
      Array.prototype.forEach.call(seg.querySelectorAll('button'), function (x) {
        var on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      renderPulse();
      loadSignups(plWindow);
    });
  })();

  // ── Metric detail lightbox (clickable tiles) ────────────────────────────
  var METRIC_TITLE = { requests: 'Requests', error_rate: 'Error rate', active_users: 'Active users',
    users: 'Users & signups', keys: 'API keys', inferences: 'Inferences' };
  var METRIC_LIST = { requests: 'requests', error_rate: 'requests', keys: 'keys',
    inferences: 'predictions', users: 'users', active_users: 'users' };
  var lbMetric = null, lbWindowSel = 'all';
  function onBreakdownClick(row) {
    var label = row.getAttribute('data-label'), id = row.getAttribute('data-id'), m = lbMetric;
    if (m === 'keys' && id) { openDetail('key', id); return; }
    if ((m === 'users' || m === 'active_users') && id) { openDetail('user', id); return; }
    var f = { window: lbWindowSel };
    if (m === 'requests') f.endpoint = label;
    else if (m === 'error_rate') f.status = label;
    else if (m === 'inferences') f.diagnosis = label;
    openDrillList(METRIC_LIST[m] || 'requests', f, label);
  }
  function openMetric(metric) {
    var lb = document.getElementById('pl-lb'), bk = document.getElementById('pl-lb-back');
    if (!lb) return;
    lb.hidden = false; bk.hidden = false;
    requestAnimationFrame(function () { lb.classList.add('open'); bk.classList.add('open'); });
    lb.innerHTML = '<div class="pl-lb-body" style="text-align:center;color:var(--text-faint);padding:70px">loading…</div>';
    adminFetch('/api/account/admin/metric/' + encodeURIComponent(metric) + '?window=' + plWindow).then(function (d) {
      if (!d) { lb.innerHTML = '<div class="pl-lb-body">could not load metric.</div>'; return; }
      renderMetricLB(d);
    });
  }
  function closeMetric() {
    var lb = document.getElementById('pl-lb'), bk = document.getElementById('pl-lb-back');
    if (lb) { lb.classList.remove('open'); setTimeout(function () { lb.hidden = true; }, 240); }
    if (bk) { bk.classList.remove('open'); setTimeout(function () { bk.hidden = true; }, 240); }
  }
  function renderMetricLB(d) {
    var lb = document.getElementById('pl-lb');
    var title = METRIC_TITLE[d.metric] || d.metric;
    var lbl = PL_WIN_LABEL[d.window] || d.window;
    var headline = (d.metric === 'error_rate') ? ((Math.round((d.headline || 0) * 10) / 10) + '') : fmtInt(d.headline || 0);
    var bd = d.breakdown || [];
    var maxv = bd.reduce(function (a, b) { return Math.max(a, b.value || 0); }, 0) || 1;
    var bdHTML = bd.length ? bd.map(function (r) {
      return '<div class="r" data-label="' + esc(r.label || '') + '"' + (r.id != null ? ' data-id="' + esc(r.id) + '"' : '')
        + ' role="button" tabindex="0"><span class="lab">' + esc(r.label || '—') + '</span><span class="v">' + fmtInt(r.value)
        + '</span><span class="b"><i style="width:' + Math.round(100 * (r.value || 0) / maxv) + '%"></i></span></div>';
    }).join('') : '<div class="pl-lb-empty">no data in this window</div>';
    lb.innerHTML = '<div class="pl-lb-head"><div class="ttl"><small>' + esc(lbl) + ' · org-wide</small><b>' + esc(title) + '</b></div>'
      + '<button class="x" id="pl-lb-x" aria-label="Close">×</button></div>'
      + '<div class="pl-lb-body">'
      + '<div class="pl-lb-hero"><span class="n">' + headline + '</span><span class="u">' + esc(d.unit || '') + '</span></div>'
      + '<div class="pl-lb-hint">' + esc(d.hint || ('over ' + lbl)) + '</div>'
      + '<div class="pl-lb-seglabel">' + esc(d.series_label || 'over time') + '</div>'
      + '<div class="pl-lb-chartwrap"><svg class="pl-lb-chart" id="pl-lb-chart" viewBox="0 0 800 190" preserveAspectRatio="none"></svg>'
      + '<div class="pl-lb-tip" id="pl-lb-tip" hidden></div></div>'
      + '<div class="pl-lb-bd"><div class="pl-lb-seglabel">' + esc(d.breakdown_label || 'breakdown') + '</div>' + bdHTML + '</div>'
      + '</div>';
    document.getElementById('pl-lb-x').addEventListener('click', closeMetric);
    lbMetric = d.metric; lbWindowSel = d.window;
    renderLBChart(d.series || []);
    // breakdown rows → drill
    var bdEl = lb.querySelector('.pl-lb-bd');
    if (bdEl) {
      bdEl.addEventListener('click', function (e) {
        var row = e.target.closest ? e.target.closest('.r[data-label]') : null; if (row) onBreakdownClick(row);
      });
      bdEl.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { var row = e.target.closest ? e.target.closest('.r[data-label]') : null; if (row) { e.preventDefault(); onBreakdownClick(row); } }
      });
    }
  }
  function renderLBChart(series) {
    var svg = document.getElementById('pl-lb-chart'); if (!svg) return;
    var W = 800, H = 190, pad = 10, bottom = 16;
    var pts = series.map(function (d) { return Number(d.c || 0); });
    if (!pts.length) { svg.innerHTML = '<text x="400" y="95" fill="#5b5b64" font-size="12" text-anchor="middle" font-family="monospace">no activity in this window</text>'; return; }
    var max = Math.max(1, Math.max.apply(null, pts)), n = pts.length, bw = (W - 2 * pad) / n, bars = '';
    for (var i = 0; i < n; i++) {
      var h = (H - pad - bottom) * (pts[i] / max), x = pad + i * bw, y = H - bottom - h;
      bars += '<rect class="bar" x="' + (x + 1).toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + Math.max(0.5, bw - 2).toFixed(1)
        + '" height="' + Math.max(0, h).toFixed(1) + '" data-i="' + i + '" rx="1.5"/>';
    }
    bars += '<line class="gl" x1="' + pad + '" y1="' + (H - bottom) + '" x2="' + (W - pad) + '" y2="' + (H - bottom) + '"/>';
    svg.innerHTML = bars;
    var tip = document.getElementById('pl-lb-tip');
    function clearBars() { Array.prototype.forEach.call(svg.querySelectorAll('.bar.on'), function (b) { b.classList.remove('on'); }); }
    svg.addEventListener('mousemove', function (e) {
      var r = svg.getBoundingClientRect(), sx = (e.clientX - r.left) / r.width * W, i = Math.floor((sx - pad) / bw);
      if (i < 0 || i >= n) { tip.hidden = true; clearBars(); return; }
      clearBars();
      var bar = svg.querySelector('.bar[data-i="' + i + '"]'); if (bar) bar.classList.add('on');
      var s = series[i];
      tip.innerHTML = '<b>' + fmtInt(s.c) + '</b> · ' + esc(sparkLabel(s.t));
      var px = (pad + i * bw + bw / 2) / W * r.width, py = (H - bottom - (H - pad - bottom) * (s.c / max)) / H * r.height;
      tip.style.left = px + 'px'; tip.style.top = py + 'px'; tip.hidden = false;
    });
    svg.addEventListener('mouseleave', function () { tip.hidden = true; clearBars(); });
    svg.style.cursor = 'pointer';
    svg.addEventListener('click', function (e) {
      var r = svg.getBoundingClientRect(), sx = (e.clientX - r.left) / r.width * W, i = Math.floor((sx - pad) / bw);
      if (i < 0 || i >= n) return;
      var s = series[i]; if (!s) return;
      var kind = METRIC_LIST[lbMetric] || 'requests';
      var f = { window: lbWindowSel, bucket: s.t };
      if (lbMetric === 'error_rate') f.status = 'error';
      openDrillList(kind, f, sparkLabel(s.t));
    });
  }
  (function initMetricTiles() {
    var sec = document.querySelector('.adm-section[data-sec="pulse"]');
    if (sec) {
      sec.addEventListener('click', function (e) {
        var t = e.target.closest ? e.target.closest('.pl-tile[data-metric]') : null;
        if (t) openMetric(t.getAttribute('data-metric'));
      });
      sec.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          var t = e.target.closest ? e.target.closest('.pl-tile[data-metric]') : null;
          if (t) { e.preventDefault(); openMetric(t.getAttribute('data-metric')); }
        }
      });
    }
    var bk = document.getElementById('pl-lb-back'); if (bk) bk.addEventListener('click', closeMetric);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { var lb = document.getElementById('pl-lb'); if (lb && !lb.hidden) closeMetric(); }
    });
  })();

  // ── Live activity feed (R5) ─────────────────────────────────────────────
  var evCat = 'all';
  var evSeen = {};
  var EV_ICONS = {
    identity:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="3.4"/><path d="M5 20a7 7 0 0 1 14 0"/></svg>',
    keys:      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="15" r="4"/><path d="M11 12l8-8 2 2-2 2 2 2-3 3-2-2-2 2"/></svg>',
    inference: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v5M12 21a6 6 0 0 0 6-6c0-4-6-9-6-9s-6 5-6 9a6 6 0 0 0 6 6Z"/></svg>',
    anomaly:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v5M12 18h.01"/></svg>',
  };
  var EV_CATLABEL = { identity: 'identity & access', keys: 'api key', inference: 'inference', anomaly: 'anomaly & ops' };

  function fmtFullTs(iso) {
    if (!iso) return ''; var t = Date.parse(iso); if (isNaN(t)) return String(iso);
    try { return new Date(t).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'medium' }); }
    catch (e) { return String(iso); }
  }
  function evRowHTML(e) {
    var ic = EV_ICONS[e.category] || EV_ICONS.identity;
    var actor = e.actor ? '<span class="ev-actor">' + esc(e.actor) + '</span>' : '';
    return '<div class="pl-ev" data-cat="' + esc(e.category) + '" data-sev="' + esc(e.severity) + '"'
      + ' data-id="' + esc(e.id) + '" data-title="' + esc(e.title) + '" data-detail="' + esc(e.detail || '') + '"'
      + ' data-actor="' + esc(e.actor || '') + '" data-ip="' + esc(e.ip || '') + '" data-ts="' + esc(e.ts || '') + '">'
      + '<span class="ev-ic">' + ic + '</span>'
      + '<span class="ev-main"><span class="ev-title">' + esc(e.title) + actor + '</span>'
      + (e.detail ? '<span class="ev-detail">' + esc(e.detail) + '</span>' : '') + '</span>'
      + '<span class="ev-when">' + esc(relTime(e.ts)) + '</span></div>';
  }
  function renderFeed(events, stagger) {
    var feed = document.getElementById('pl-feed'); if (!feed) return;
    if (!events.length) { feed.innerHTML = '<div class="pl-feed-empty">no events in this lane yet</div>'; return; }
    feed.innerHTML = events.map(evRowHTML).join('');
    Array.prototype.forEach.call(feed.querySelectorAll('.pl-ev'), function (row, i) {
      evSeen[row.getAttribute('data-id')] = true;
      if (stagger && !REDUCED) { row.style.animationDelay = (i * 0.028) + 's'; row.classList.add('pop'); }
    });
  }
  function loadFeed() {
    var qs = 'limit=40' + (evCat !== 'all' ? '&category=' + encodeURIComponent(evCat) : '');
    evSeen = {};
    adminFetch('/api/account/admin/events?' + qs).then(function (d) {
      if (d) renderFeed(d.events || [], true);
    });
  }
  function pollFeed() {
    var feed = document.getElementById('pl-feed'); if (!feed) return;
    if (feed.querySelector('.pl-feed-empty') || !feed.children.length) { loadFeed(); return; }
    var qs = 'limit=40' + (evCat !== 'all' ? '&category=' + encodeURIComponent(evCat) : '');
    adminFetch('/api/account/admin/events?' + qs).then(function (d) {
      if (!d) return;
      var fresh = (d.events || []).filter(function (e) { return !evSeen[e.id]; });
      if (!fresh.length) return;
      fresh.reverse().forEach(function (e) {         // oldest-new first → newest ends on top
        evSeen[e.id] = true;
        feed.insertAdjacentHTML('afterbegin', evRowHTML(e));
        if (!REDUCED && feed.firstChild && feed.firstChild.classList) feed.firstChild.classList.add('pop');
      });
      while (feed.children.length > 60) feed.removeChild(feed.lastChild);
    });
  }
  function showEvPop(row) {
    var pop = document.getElementById('pl-pop'); if (!pop) return;
    var cat = row.getAttribute('data-cat');
    var color = { identity: 'var(--teal)', keys: 'var(--violet)', inference: 'var(--accent)', anomaly: 'var(--crimson)' }[cat] || 'var(--text-dim)';
    function rw(k, v) { return v ? '<div class="pp-row"><span class="k">' + k + '</span><span class="v">' + esc(v) + '</span></div>' : ''; }
    pop.innerHTML = '<div class="pp-cat" style="color:' + color + '">' + esc(EV_CATLABEL[cat] || cat) + ' · ' + esc(row.getAttribute('data-sev')) + '</div>'
      + '<div class="pp-title">' + esc(row.getAttribute('data-title')) + '</div>'
      + rw('detail', row.getAttribute('data-detail'))
      + rw('actor', row.getAttribute('data-actor'))
      + rw('ip', row.getAttribute('data-ip'))
      + rw('when', fmtFullTs(row.getAttribute('data-ts')));
    pop.hidden = false;
    var r = row.getBoundingClientRect(), pw = 280, ph = pop.offsetHeight || 150;
    var left = Math.min(window.innerWidth - pw - 12, Math.max(12, r.left));
    var top = r.bottom + 8;
    if (top + ph > window.innerHeight - 12) top = r.top - ph - 8;
    pop.style.left = left + 'px'; pop.style.top = Math.max(12, top) + 'px';
  }
  (function initFeed() {
    var filters = document.getElementById('pl-feed-filters');
    if (filters) filters.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('button[data-cat]') : null; if (!b) return;
      evCat = b.getAttribute('data-cat');
      Array.prototype.forEach.call(filters.querySelectorAll('button'), function (x) {
        var on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      loadFeed();
    });
    var feed = document.getElementById('pl-feed'), pop = document.getElementById('pl-pop');
    if (feed && pop) {
      feed.addEventListener('mouseover', function (e) {
        var row = e.target.closest ? e.target.closest('.pl-ev') : null; if (row) showEvPop(row);
      });
      feed.addEventListener('mouseout', function (e) {
        var row = e.target.closest ? e.target.closest('.pl-ev') : null;
        if (row && (!e.relatedTarget || !row.contains(e.relatedTarget))) pop.hidden = true;
      });
      feed.addEventListener('scroll', function () { pop.hidden = true; });
    }
  })();

  // Live refresh — only while Pulse is the active section AND the tab is visible.
  setInterval(function () {
    if (document.hidden) return;
    if (!document.querySelector('.adm-section[data-sec="pulse"].active')) return;
    loadPulse(); pollFeed();
  }, 10000);

  // ── Users directory ─────────────────────────────────────────────────────
  var uState = { search: '', sort: 'created', order: 'desc', offset: 0, limit: 25, total: 0, loaded: false, loading: false };

  function relTime(iso) {
    if (!iso) return 'never';
    var t = Date.parse(iso); if (isNaN(t)) return '—';
    var s = Math.floor((Date.now() - t) / 1000);
    if (s < 60) return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    var d = Math.floor(s / 86400); if (d < 30) return d + 'd ago';
    try { return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }); } catch (e) { return iso; }
  }
  function fmtDate(iso) {
    if (!iso) return '—'; var t = Date.parse(iso); if (isNaN(t)) return '—';
    try { return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }); } catch (e) { return iso; }
  }

  // 14-day request sparkline → tiny inline bar SVG
  function uSpark(arr) {
    arr = arr || [];
    var n = arr.length;
    if (!n) return '<span class="u-dim">—</span>';
    var tot = 0, max = 1;
    for (var i = 0; i < n; i++) { tot += (+arr[i] || 0); if (arr[i] > max) max = arr[i]; }
    if (tot === 0) return '<span class="u-spk-empty" title="no requests in 14 days">no traffic</span>';
    var W = 78, H = 22, gap = 1.4, bw = (W - (n - 1) * gap) / n, bars = '';
    for (var j = 0; j < n; j++) {
      var v = +arr[j] || 0;
      var bh = v > 0 ? Math.max(2, (v / max) * (H - 2)) : 1;
      var x = j * (bw + gap), y = H - bh;
      bars += '<rect class="' + (v > 0 ? 'on' : 'z') + (j === n - 1 ? ' last' : '') + '" x="' + x.toFixed(2)
        + '" y="' + y.toFixed(2) + '" width="' + bw.toFixed(2) + '" height="' + bh.toFixed(2) + '" rx="0.7"></rect>';
    }
    return '<svg class="u-spk" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" role="img" '
      + 'aria-label="' + tot + ' requests over 14 days"><title>' + tot + ' requests · last 14 days</title>' + bars + '</svg>';
  }

  function userRowHTML(u) {
    var initial = ((u.display_name || u.username || '?').trim()[0] || '?').toUpperCase();
    var roleCls = (u.role === 'admin') ? 'admin' : 'collector';
    var roleTxt = u.is_admin ? (u.admin_via === 'allowlist' ? 'admin*' : 'admin') : 'collector';
    var liveTitle = u.active_now ? 'active now · live session' : ('last seen ' + relTime(u.last_seen_at));
    return '<div class="adm-urow" data-uid="' + u.id + '" role="button" tabindex="0">'
      + '<div class="u-id"><span class="u-av' + (u.active_now ? ' live' : '') + '">' + esc(initial) + '</span><span class="u-nm"><b>'
      + esc(u.display_name || u.username || '—') + '</b><span>' + esc(u.email || '') + '</span></span></div>'
      + '<div><span class="role-badge dot ' + roleCls + '">' + esc(roleTxt) + '</span></div>'
      + '<div class="u-stat">' + fmtInt(u.key_count) + '</div>'
      + '<div class="u-stat">' + fmtInt(u.request_count) + '</div>'
      + '<div class="u-stat u-infer' + (u.inference_count ? '' : ' z') + '">' + fmtInt(u.inference_count) + '</div>'
      + '<div class="u-stat u-guest' + (u.guest_count ? '' : ' z') + '">' + fmtInt(u.guest_count) + '</div>'
      + '<div class="u-spark">' + uSpark(u.spark) + '</div>'
      + '<div class="u-act"><span class="u-live-dot' + (u.active_now ? ' on' : '') + '" title="' + esc(liveTitle) + '"></span></div>'
      + '<div class="chev"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg></div>'
      + '</div>';
  }

  function renderUSummary(s) {
    var el = document.getElementById('u-summary');
    if (!el) return;
    if (!s || typeof s.total_users === 'undefined') { el.innerHTML = ''; el.hidden = true; return; }
    el.hidden = false;
    function cell(label, val, live) {
      return '<div class="u-sum' + (live ? ' live' : '') + '">'
        + (live ? '<span class="u-sum-dot' + (val > 0 ? ' on' : '') + '"></span>' : '')
        + '<span class="u-sum-v" data-num="0">0</span>'
        + '<span class="u-sum-l">' + label + '</span></div>';
    }
    el.innerHTML = cell('total users', s.total_users)
      + cell('admins', s.admins)
      + cell('active now', s.active_now, true)
      + cell('guests converted', s.guests_converted);
    var vals = el.querySelectorAll('.u-sum-v'),
        nums = [s.total_users, s.admins, s.active_now, s.guests_converted];
    for (var i = 0; i < vals.length; i++) animateNum(vals[i], nums[i] || 0);
  }

  function loadUsers(reset) {
    if (uState.loading) return;
    if (reset) uState.offset = 0;
    uState.loading = true;
    var qs = 'search=' + encodeURIComponent(uState.search) + '&sort=' + uState.sort
      + '&order=' + uState.order + '&limit=' + uState.limit + '&offset=' + uState.offset;
    adminFetch('/api/account/admin/users?' + qs).then(function (d) {
      uState.loading = false;
      var rows = document.getElementById('u-rows'); if (!rows) return;
      if (!d) { if (reset) rows.innerHTML = '<div class="adm-urow"><div class="u-dim">could not load users</div></div>'; return; }
      var pg = d.pagination || {}; uState.total = pg.total || 0; uState.loaded = true;
      if (d.summary) renderUSummary(d.summary);
      if (reset) rows.innerHTML = '';
      (d.items || []).forEach(function (u) { rows.insertAdjacentHTML('beforeend', userRowHTML(u)); });
      var cnt = document.getElementById('u-count'); if (cnt) cnt.textContent = uState.total + ' user' + (uState.total === 1 ? '' : 's');
      var shown = rows.querySelectorAll('.adm-urow[data-uid]').length;
      var pager = document.getElementById('u-pager');
      if (pager) {
        if (shown < uState.total) {
          pager.innerHTML = '<button class="adm-btn" id="u-more">load more · ' + (uState.total - shown) + ' left</button>';
          document.getElementById('u-more').addEventListener('click', function () { uState.offset += uState.limit; loadUsers(false); });
        } else pager.innerHTML = '';
      }
    });
  }

  function closeDrawer() {
    var dw = document.getElementById('u-drawer'), bk = document.getElementById('u-drawer-back');
    if (dw) { dw.classList.remove('open'); dw.setAttribute('aria-hidden', 'true'); }
    if (bk) bk.classList.remove('open');
  }
  function openUserDrawer(id) {
    var dw = document.getElementById('u-drawer'), bk = document.getElementById('u-drawer-back');
    if (!dw) return;
    dw.innerHTML = '<div class="dbody" style="padding-top:64px;text-align:center;color:var(--text-faint)">loading…</div>';
    dw.classList.add('open'); dw.setAttribute('aria-hidden', 'false');
    if (bk) bk.classList.add('open');
    adminFetch('/api/account/admin/users/' + id).then(function (u) {
      if (!u) { dw.innerHTML = '<div class="dbody" style="padding-top:64px;text-align:center;color:var(--text-faint)">could not load user</div>'; return; }
      renderDrawer(u);
    });
  }
  function renderDrawer(u) {
    var dw = document.getElementById('u-drawer'); if (!dw) return;
    var initial = ((u.display_name || u.username || '?').trim()[0] || '?').toUpperCase();
    var self = Number(window.__ADMIN_SELF_ID__) === Number(u.id);
    var roleTxt = u.is_admin ? (u.admin_via === 'allowlist' ? 'admin · via config allowlist' : 'admin · via role') : 'collector';
    var actions = '<button class="adm-btn primary" id="dw-profile">Open full profile →</button>';
    actions += (u.role === 'admin')
      ? '<button class="adm-btn danger" id="dw-demote">Revoke admin' + (self ? ' (yourself)' : '') + '</button>'
      : '<button class="adm-btn" id="dw-promote">Promote to admin</button>';
    actions += '<button class="adm-btn" id="dw-logout">Force sign-out' + (u.active_sessions ? ' · ' + u.active_sessions + ' token' + (u.active_sessions === 1 ? '' : 's') : '') + '</button>';
    var note = (u.admin_via === 'allowlist')
      ? 'Admin via the APIN_ADMIN_EMAILS config allowlist — changing the role here will not remove admin access until the email is removed from the server config.' : '';
    dw.innerHTML =
      '<div class="dh"><span class="av">' + esc(initial) + '</span><span class="nm"><b>'
      + esc(u.display_name || u.username || '—') + '</b><span>' + esc(u.email || '') + '</span></span>'
      + '<button class="x" id="dw-x" aria-label="Close">×</button></div>'
      + '<div class="dbody">'
      + '<div class="adm-dstats">'
      + '<div class="adm-dstat"><div class="l">API keys</div><div class="v">' + fmtInt(u.key_count) + '</div></div>'
      + '<div class="adm-dstat"><div class="l">Lifetime requests</div><div class="v">' + fmtInt(u.request_count) + '</div></div>'
      + '<div class="adm-dstat"><div class="l">Valid tokens</div><div class="v">' + fmtInt(u.active_sessions) + '</div></div>'
      + '<div class="adm-dstat"><div class="l">User id</div><div class="v">' + esc(String(u.id)) + '</div></div>'
      + '</div>'
      + '<div class="adm-drow"><span class="k">Role</span><span class="vv">' + esc(roleTxt) + '</span></div>'
      + '<div class="adm-drow"><span class="k">Username</span><span class="vv">' + esc(u.username || '—') + '</span></div>'
      + '<div class="adm-drow"><span class="k">Mobile</span><span class="vv">' + esc(u.mobile_masked || '—') + '</span></div>'
      + '<div class="adm-drow"><span class="k">Created</span><span class="vv">' + esc(fmtDate(u.created_at)) + '</span></div>'
      + '<div class="adm-drow"><span class="k">Last seen</span><span class="vv">' + esc(relTime(u.last_seen_at)) + '</span></div>'
      + '<div class="adm-dactions">' + actions + '</div>'
      + (note ? '<div class="adm-dnote">' + esc(note) + '</div>' : '')
      + '</div>';
    document.getElementById('dw-x').addEventListener('click', closeDrawer);
    var prof = document.getElementById('dw-profile');
    if (prof) prof.addEventListener('click', function () { closeDrawer(); openDetail('user', u.id); });
    var pr = document.getElementById('dw-promote'); if (pr) pr.addEventListener('click', function () { confirmRole(u, 'admin'); });
    var dm = document.getElementById('dw-demote'); if (dm) dm.addEventListener('click', function () { confirmRole(u, 'collector'); });
    var lo = document.getElementById('dw-logout'); if (lo) lo.addEventListener('click', function () { confirmLogout(u); });
  }

  function setRole(u, newRole) {
    return window.APIN.sudoFetch('/api/account/admin/users/' + u.id + '/role', 'POST', { role: newRole }).then(function (res) {
      var body = (res && res.body) ? res.body : res;
      if (body && body.ok) return;
      throw new Error((body && body.error && body.error.message) || 'Update failed.');
    });
  }
  function forceLogout(u) {
    return window.APIN.sudoFetch('/api/account/admin/users/' + u.id + '/logout-all', 'POST', {}).then(function (res) {
      var body = (res && res.body) ? res.body : res;
      if (body && body.ok) return;
      throw new Error((body && body.error && body.error.message) || 'Failed.');
    });
  }
  function confirmRole(u, newRole) {
    var M = window.APIN && window.APIN.modal;
    var promote = newRole === 'admin';
    var name = u.display_name || u.username || u.email || ('user ' + u.id);
    var self = Number(window.__ADMIN_SELF_ID__) === Number(u.id);
    var msg = promote
      ? 'They will gain full access to the admin console — including the live database mirror.'
      : ('They will lose all admin access.' + (self ? ' <strong>This is your own account.</strong>' : ''));
    function done() { openUserDrawer(u.id); loadUsers(true); }
    if (M && M.confirm) {
      M.confirm({
        danger: !promote,
        title: (promote ? 'Promote ' : 'Revoke admin from ') + name + (promote ? ' to admin?' : '?'),
        message: msg, confirmLabel: promote ? 'Promote' : 'Revoke', busyLabel: 'Working…',
        onConfirm: function () { return setRole(u, newRole).then(done); },
      });
    } else if (window.confirm((promote ? 'Promote ' : 'Revoke admin from ') + name + '?')) {
      setRole(u, newRole).then(done).catch(function (e) { window.alert(e.message); });
    }
  }
  function confirmLogout(u) {
    var M = window.APIN && window.APIN.modal;
    var name = u.display_name || u.username || u.email || ('user ' + u.id);
    function done() { openUserDrawer(u.id); }
    if (M && M.confirm) {
      M.confirm({
        title: 'Force sign-out ' + name + '?',
        message: 'All of their active sessions are revoked immediately — they will need to sign in again.',
        confirmLabel: 'Force sign-out', busyLabel: 'Revoking…',
        onConfirm: function () { return forceLogout(u).then(done); },
      });
    } else if (window.confirm('Force sign-out ' + name + '?')) {
      forceLogout(u).then(done).catch(function (e) { window.alert(e.message); });
    }
  }

  (function initUsers() {
    var rows = document.getElementById('u-rows');
    if (rows) rows.addEventListener('click', function (e) {
      var row = e.target.closest ? e.target.closest('.adm-urow[data-uid]') : null;
      if (row) openUserDrawer(row.getAttribute('data-uid'));
    });
    var srch = document.getElementById('u-search'), st = null;
    if (srch) srch.addEventListener('input', function () {
      clearTimeout(st); st = setTimeout(function () { uState.search = srch.value.trim(); loadUsers(true); }, 250);
    });
    var sort = document.getElementById('u-sort');
    if (sort) sort.addEventListener('change', function () {
      var p = sort.value.split('|'); uState.sort = p[0]; uState.order = p[1] || 'desc'; loadUsers(true);
    });
    var bk = document.getElementById('u-drawer-back'); if (bk) bk.addEventListener('click', closeDrawer);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { var dw = document.getElementById('u-drawer'); if (dw && dw.classList.contains('open')) closeDrawer(); }
    });
  })();

  // ══ DRILL: list container · detail drawer · dossier (P2 infra) ═══════════
  function dlOpenPanel(el, back) {
    if (!el) return;
    el.hidden = false; if (back) back.hidden = false;
    // Stamp open order so Esc can peel the most-recently-opened panel first
    // (nested cross-drills can stack dl-dw + dl-dossier at the same z-index).
    try { el.dataset.oat = String(Math.round(performance.now())); } catch (e) {}
    requestAnimationFrame(function () { el.classList.add('open'); if (back) back.classList.add('open'); });
  }
  function dlClosePanel(el, back) {
    if (el) { el.classList.remove('open'); setTimeout(function () { el.hidden = true; }, 340); }
    if (back) { back.classList.remove('open'); setTimeout(function () { back.hidden = true; }, 340); }
  }
  function closeDrillList() { dlClosePanel(document.getElementById('dl-list'), document.getElementById('dl-list-back')); }
  function closeDrillDrawer() { dlClosePanel(document.getElementById('dl-dw'), document.getElementById('dl-dw-back')); }
  function closeDossier() { dlClosePanel(document.getElementById('dl-dossier'), document.getElementById('dl-dossier-back')); }

  var ES_ANIM = {
    radar: '<svg class="es-radar" viewBox="0 0 76 76"><circle cx="38" cy="38" r="32"/><circle cx="38" cy="38" r="20"/><circle cx="38" cy="38" r="9"/><line class="sweep" x1="38" y1="38" x2="38" y2="6" stroke="var(--accent)" stroke-width="1.5"/><circle class="blip" cx="54" cy="28" r="3"/></svg>',
    cursor: '<svg class="es-cursor" viewBox="0 0 76 76"><path class="grid" d="M10 24H66M10 38H66M10 52H66" stroke-width="1"/><rect class="cur" x="14" y="34" width="8" height="9"/></svg>',
    seed:   '<svg class="es-seed" viewBox="0 0 76 76"><path class="sprout" d="M38 44 Q30 32 24 30 M38 44 Q46 30 52 28" fill="none" stroke-width="1.5"/><ellipse class="seed" cx="38" cy="50" rx="7" ry="9"/></svg>',
    flat:   '<svg class="es-flat" viewBox="0 0 76 40"><path class="ln" d="M6 20H70"/><circle class="pulse" cx="0" cy="20" r="3"/></svg>',
  };
  function dlEmpty(anim, tag, sub) {
    return '<div class="dl-empty"><div class="glyph">' + (ES_ANIM[anim] || ES_ANIM.flat) + '</div>'
      + '<div class="tag">' + esc(tag) + '</div>' + (sub ? '<div class="sub">' + esc(sub) + '</div>' : '') + '</div>';
  }

  // ── list container ──
  var LIST_TITLE = { requests: 'Requests', keys: 'API keys', scans: 'Inferences', predictions: 'Inferences', users: 'Users' };
  var DETAIL_OF = { requests: 'request', keys: 'key', scans: 'scan', predictions: 'prediction', users: 'user' };
  var ROW_COLS = { requests: '46px 1fr 44px 58px auto 16px', keys: '1fr auto 60px 16px',
    scans: '1fr auto auto 16px', predictions: '1fr auto auto 40px 16px', users: '1fr auto auto 16px' };
  function statusClass(s) { s = +s || 0; return s >= 500 ? 'err' : (s >= 400 ? 'warn' : 'ok'); }
  function fmtBytes(n) { n = +n || 0; return n < 1024 ? n + 'b' : (n < 1048576 ? (n / 1024).toFixed(1) + 'kb' : (n / 1048576).toFixed(1) + 'mb'); }

  var dlListState = { kind: null, filters: {}, offset: 0, total: 0 };
  var dlSeq = 0;   // monotonic guard — drop out-of-order list responses

  function drillRowHTML(kind, it) {
    if (kind === 'requests') {
      return '<div class="dl-row" data-id="' + esc(it.id) + '" style="grid-template-columns:' + ROW_COLS.requests + '">'
        + '<span class="dl-method">' + esc(it.method || '') + '</span>'
        + '<span class="path">' + esc(it.path || '') + '</span>'
        + '<span class="st ' + statusClass(it.status) + '">' + esc(it.status) + '</span>'
        + '<span class="mono">' + (it.latency_ms != null ? it.latency_ms + 'ms' : '—') + '</span>'
        + '<span class="mono">' + esc(relTime(it.ts)) + '</span>'
        + '<span class="chev">›</span></div>';
    }
    if (kind === 'keys') {
      return '<div class="dl-row" data-id="' + esc(it.public_id) + '" style="grid-template-columns:' + ROW_COLS.keys + '">'
        + '<span class="path">' + esc(it.name) + '<span class="mono" style="color:var(--text-faint)"> · ' + esc(it.owner || '') + '</span></span>'
        + '<span class="mono">' + fmtInt(it.requests) + ' req</span>'
        + '<span class="st ' + (it.error_rate >= 5 ? 'warn' : 'ok') + '">' + (it.error_rate || 0) + '%</span>'
        + '<span class="chev">›</span></div>';
    }
    if (kind === 'scans') {
      return '<div class="dl-row" data-id="' + esc(it.scan_uid) + '" style="grid-template-columns:' + ROW_COLS.scans + '">'
        + '<span class="path">' + esc((it.diagnosis || '—').replace(/_/g, ' ')) + '</span>'
        + '<span class="mono">' + (it.confidence != null ? Math.round(it.confidence * 100) + '%' : '—') + '</span>'
        + '<span class="mono" style="color:var(--text-dim)">' + esc(it.severity || '') + '</span>'
        + '<span class="chev">›</span></div>';
    }
    if (kind === 'predictions') {
      return '<div class="dl-row" data-id="' + esc(it.id) + '" style="grid-template-columns:' + ROW_COLS.predictions + '">'
        + '<span class="path">' + esc((it.diagnosis || '—').replace(/_/g, ' '))
        + (it.guest ? '<span class="mono" style="color:var(--text-faint)"> · guest</span>'
            : (it.owner ? '<span class="mono" style="color:var(--text-faint)"> · ' + esc(it.owner.display_name || ('user ' + it.owner.id)) + '</span>' : '')) + '</span>'
        + '<span class="mono">' + (it.confidence != null ? Math.round(it.confidence * 100) + '%' : '—') + '</span>'
        + '<span class="mono" style="color:var(--text-dim)">' + (it.tier ? 'T' + esc(it.tier) : '') + '</span>'
        + '<span class="dl-hm' + (it.has_heatmap ? ' on' : '') + '" title="' + (it.has_heatmap ? 'attention map captured' : 'no heatmap') + '">◉</span>'
        + '<span class="chev">›</span></div>';
    }
    // users
    return '<div class="dl-row" data-id="' + esc(it.id) + '" style="grid-template-columns:' + ROW_COLS.users + '">'
      + '<span class="path">' + esc(it.display_name || it.username || ('user ' + it.id))
      + '<span class="mono" style="color:var(--text-faint)"> · ' + esc(it.email || '') + '</span></span>'
      + '<span class="mono">' + fmtInt(it.keys) + ' keys</span>'
      + '<span class="mono">' + fmtInt(it.requests) + ' req</span>'
      + '<span class="chev">›</span></div>';
  }

  function openDrillList(kind, filters, sub) {
    var el = document.getElementById('dl-list'), back = document.getElementById('dl-list-back');
    if (!el) return;
    dlListState = { kind: kind, filters: filters || {}, offset: 0, total: 0 };
    el.innerHTML = '<div class="dl-head"><div class="ttl"><div class="sub">' + esc(sub || 'filtered')
      + '</div><div class="main">' + esc(LIST_TITLE[kind] || kind) + '</div></div>'
      + '<button class="x" id="dl-list-x" aria-label="Close">×</button></div>'
      + '<div class="dl-body" id="dl-list-body">' + dlEmpty('flat', 'loading…') + '</div>';
    dlOpenPanel(el, back);
    document.getElementById('dl-list-x').addEventListener('click', closeDrillList);
    fetchDrillList(true);
  }
  function fetchDrillList(reset) {
    var st = dlListState;
    var mySeq = ++dlSeq;                         // capture this request's order token
    var f = Object.assign({ limit: 50, offset: st.offset }, st.filters);
    var qs = Object.keys(f).filter(function (k) { return f[k] != null && f[k] !== ''; })
      .map(function (k) { return k + '=' + encodeURIComponent(f[k]); }).join('&');
    adminFetch('/api/account/admin/list/' + st.kind + (qs ? '?' + qs : '')).then(function (d) {
      if (mySeq !== dlSeq) return;               // a newer request superseded us — drop stale response
      var body = document.getElementById('dl-list-body'); if (!body) return;
      if (!d) { body.innerHTML = dlEmpty('flat', 'could not load'); return; }
      st.total = d.total || 0;
      renderDrillList(d, reset);
    });
  }
  function renderDrillList(d, reset) {
    var kind = dlListState.kind, body = document.getElementById('dl-list-body');
    var items = d.items || [];
    if (reset) {
      var head = '';
      if (kind === 'requests' && d.stats) {
        var s = d.stats;
        head += '<div class="dl-strip">'
          + '<div class="s"><b>' + fmtInt(s.total) + '</b><span>requests</span></div>'
          + '<div class="s"><b>' + (s.error_rate || 0) + '%</b><span>error rate</span></div>'
          + '<div class="s"><b>' + (s.p50 != null ? s.p50 + 'ms' : '—') + '</b><span>p50</span></div>'
          + '<div class="s"><b>' + (s.p95 != null ? s.p95 + 'ms' : '—') + '</b><span>p95</span></div></div>'
          + '<div class="dl-chips" id="dl-chips">'
          + ['all', '2xx', '4xx', '5xx'].map(function (c) {
            var cur = dlListState.filters.status || 'all';
            return '<button data-st="' + c + '" class="' + (cur === c ? 'on' : '') + '">' + c + '</button>';
          }).join('') + '</div>';
      }
      if (!items.length) {
        var tag = { requests: 'quiet on the wire', keys: 'no keys turned yet',
          scans: "this field hasn't been read", predictions: 'no leaves read here yet',
          users: 'no one signed the register' }[kind] || 'nothing here';
        var anim = { requests: 'flat', keys: 'cursor', scans: 'seed', predictions: 'seed', users: 'radar' }[kind] || 'flat';
        body.innerHTML = head + dlEmpty(anim, tag, 'try a wider window');
        if (kind === 'requests') wireListChips();
        return;
      }
      head += '<div class="dl-rows" id="dl-rows"></div>';
      head += '<div class="dl-more" id="dl-more-wrap"></div>';
      body.innerHTML = head;
      if (kind === 'requests') wireListChips();
      var rows = document.getElementById('dl-rows');
      rows.addEventListener('click', function (e) {
        var row = e.target.closest ? e.target.closest('.dl-row[data-id]') : null;
        if (row) openDetail(DETAIL_OF[kind], row.getAttribute('data-id'));
      });
    }
    var rowsEl = document.getElementById('dl-rows');
    items.forEach(function (it) { rowsEl.insertAdjacentHTML('beforeend', drillRowHTML(kind, it)); });
    dlListState.offset += items.length;
    var mw = document.getElementById('dl-more-wrap');
    if (mw) {
      if (dlListState.offset < dlListState.total) {
        mw.innerHTML = '<button class="adm-btn" id="dl-more">load ' + (dlListState.total - dlListState.offset) + ' more</button>';
        document.getElementById('dl-more').addEventListener('click', function () { fetchDrillList(false); });
      } else mw.innerHTML = '';
    }
  }
  function wireListChips() {
    var ch = document.getElementById('dl-chips'); if (!ch) return;
    ch.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('button[data-st]') : null; if (!b) return;
      var st = b.getAttribute('data-st');
      dlListState.filters.status = (st === 'all') ? undefined : st;
      dlListState.offset = 0;
      Array.prototype.forEach.call(ch.querySelectorAll('button'), function (x) { x.classList.toggle('on', x === b); });
      fetchDrillList(true);
    });
  }

  // ── detail dispatch (renderers filled per phase) ──
  function openDetail(kind, id) {
    if (kind === 'request') {
      // P3 · reuse the EXACT console request drawer (dark-reskinned). Same
      // module, same sections, same content — only the theme differs. The
      // detail URL was pointed at the admin endpoint in initDrillGlobal().
      if (window.APIN && APIN.requestDrawer && APIN.requestDrawer.open) {
        APIN.requestDrawer.open(id);
        return;
      }
      // Defensive fallback if the shared module failed to load.
    }
    if (kind === 'request' || kind === 'key') {
      var el = document.getElementById('dl-dw'), back = document.getElementById('dl-dw-back');
      el.innerHTML = '<div class="dl-body">' + dlEmpty('flat', 'loading…') + '</div>';
      dlOpenPanel(el, back);
      adminFetch('/api/account/admin/detail/' + kind + '/' + encodeURIComponent(id)).then(function (d) {
        if (!d) { el.innerHTML = '<div class="dl-head"><div class="ttl"><div class="main">not found</div></div><button class="x" id="dl-dw-x">×</button></div>'; bindClose('dl-dw-x', closeDrillDrawer); return; }
        if (kind === 'request') renderRequestDrawer(d); else renderKeyDrawer(d);
      });
    } else {
      var ds = document.getElementById('dl-dossier'), dback = document.getElementById('dl-dossier-back');
      ds.innerHTML = '<div class="dl-spine"></div><div class="dl-pane"><div class="dl-body">' + dlEmpty('flat', 'loading…') + '</div></div>';
      dlOpenPanel(ds, dback);
      adminFetch('/api/account/admin/detail/' + kind + '/' + encodeURIComponent(id)).then(function (d) {
        if (!d || (kind === 'scan' && d.found === false)) { ds.innerHTML = '<div class="dl-spine"></div><div class="dl-pane"><div class="dl-head"><div class="ttl"><div class="main">not found</div></div><button class="x" id="dl-ds-x">×</button></div></div>'; bindClose('dl-ds-x', closeDossier); return; }
        if (kind === 'scan') renderScanDossier(d);
        else if (kind === 'prediction') renderPredictionDossier(d);
        else renderUserDossier(d);
      });
    }
  }
  function bindClose(id, fn) { var b = document.getElementById(id); if (b) b.addEventListener('click', fn); }

  // Spine-nav for the wide dossiers (scan · user): click a node to glide to its
  // section; a scroll-spy keeps the active node lit. Generic — drives any pane
  // whose sections carry id === the node's data-sec.
  function wireDossierSpine(root, bodyId) {
    var body = document.getElementById(bodyId);
    var spine = root.querySelector('.dl-spine');
    if (!body || !spine) return;
    var nodes = Array.prototype.slice.call(spine.querySelectorAll('.dl-snode[data-sec]'));
    function sec(nd) { return document.getElementById(nd.getAttribute('data-sec')); }
    function topOf(t) { return t.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop; }
    nodes.forEach(function (nd) {
      function go() { var t = sec(nd); if (t) body.scrollTo({ top: Math.max(0, topOf(t) - 8), behavior: 'smooth' }); }
      nd.addEventListener('click', go);
      nd.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
    var spyRaf = null;
    body.addEventListener('scroll', function () {
      if (spyRaf) return;
      spyRaf = requestAnimationFrame(function () {
        spyRaf = null;
        var st = body.scrollTop, best = 0, bestD = Infinity;
        // At the bottom the final section can't reach the top — force-light the
        // last node so the spine never strands you a section behind.
        if (st + body.clientHeight >= body.scrollHeight - 4) {
          best = nodes.length - 1;
        } else {
          nodes.forEach(function (nd, i) { var s = sec(nd); if (!s) return; var dd = Math.abs(topOf(s) - 8 - st); if (dd < bestD) { bestD = dd; best = i; } });
        }
        nodes.forEach(function (nd, i) { nd.classList.toggle('on', i === best); });
      });
    }, { passive: true });
  }

  // Phase-filled renderers (basic shells until their phase enriches them) ──
  function renderRequestDrawer(d) { _basicDrawer('request', d, (d.row && (d.row.method + ' ' + d.row.path)) || 'request'); }

  // ── P4 · rich KEY drawer ────────────────────────────────────────────────
  var KD_BAND = { '2xx': 'var(--accent)', '3xx': 'var(--teal)', '4xx': 'var(--ochre)', '5xx': 'var(--crimson)', '1xx': 'var(--violet)' };
  var KD_LIFE = {  // audit action → glyph + label
    created: ['✦', 'minted'], rotated: ['↻', 'rotated'], disabled: ['⏻', 'disabled'],
    enabled: ['●', 're-enabled'], deleted: ['✕', 'deleted'], updated: ['✎', 'updated'],
    scope_changed: ['◈', 'scopes changed'], renamed: ['✎', 'renamed']
  };
  function kdSpark(series) {
    var pts = (series || []).map(function (p) { return Number(p.c || p.value || 0); });
    if (!pts.length || pts.every(function (v) { return v === 0; }))
      return '<div class="kd-spark-empty">no traffic in the last 30 days</div>';
    var W = 320, H = 54, max = Math.max.apply(null, pts), n = pts.length, bw = W / n;
    var bars = pts.map(function (v, i) {
      var h = max ? (H - 6) * (v / max) : 0;
      return '<rect x="' + (i * bw + 0.6).toFixed(1) + '" y="' + (H - h).toFixed(1) + '" width="' + Math.max(0.8, bw - 1.2).toFixed(1)
        + '" height="' + h.toFixed(1) + '" rx="1" fill="var(--accent)" opacity="' + (0.45 + 0.55 * (max ? v / max : 0)).toFixed(2) + '"/>';
    }).join('');
    return '<svg class="kd-spark" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' + bars + '</svg>';
  }
  function kdStatusMix(mix) {
    var total = (mix || []).reduce(function (a, b) { return a + (b.value || 0); }, 0);
    if (!total) return '';
    var seg = mix.map(function (m) {
      var pct = 100 * (m.value || 0) / total;
      return '<span class="kd-seg" style="width:' + pct.toFixed(2) + '%;background:' + (KD_BAND[m.label] || 'var(--text-faint)')
        + '" title="' + esc(m.label) + ': ' + fmtInt(m.value) + ' (' + pct.toFixed(1) + '%)"></span>';
    }).join('');
    var leg = mix.map(function (m) {
      return '<span class="kd-leg"><i style="background:' + (KD_BAND[m.label] || 'var(--text-faint)') + '"></i>'
        + esc(m.label) + ' <b>' + fmtInt(m.value) + '</b></span>';
    }).join('');
    return '<div class="kd-mixbar">' + seg + '</div><div class="kd-legs">' + leg + '</div>';
  }
  function renderKeyDrawer(d) {
    var el = document.getElementById('dl-dw');
    if (!d) { _basicDrawer('key', {}, 'key'); return; }
    var s = d.stats || {};
    var st = (d.status || 'active').toLowerCase();
    var stCls = st === 'active' ? 'ok' : (st === 'disabled' ? 'err' : 'warn');
    var maxEp = (d.top_endpoints || []).reduce(function (a, b) { return Math.max(a, b.value || 0); }, 0) || 1;

    var kpis = [
      ['requests', fmtInt(s.requests || 0)],
      ['error rate', (s.error_rate || 0) + '%'],
      ['p50', s.p50 != null ? s.p50 + 'ms' : '—'],
      ['p95', s.p95 != null ? s.p95 + 'ms' : '—'],
      ['org share', (s.org_share || 0) + '%'],
      ['errors', fmtInt(s.errors || 0)]
    ].map(function (k) {
      return '<div class="kd-kpi"><span class="v">' + k[1] + '</span><span class="l">' + k[0] + '</span></div>';
    }).join('');

    var scopes = (d.scopes && d.scopes.length)
      ? d.scopes.map(function (sc) { return '<span class="kd-scope">' + esc(sc) + '</span>'; }).join('')
      : '<span class="kd-scope muted">no explicit scopes</span>';

    var topEps = (d.top_endpoints && d.top_endpoints.length)
      ? d.top_endpoints.map(function (e) {
          return '<div class="kd-ep"><span class="p">' + esc(e.label) + '</span>'
            + '<span class="bar"><i style="width:' + Math.round(100 * (e.value || 0) / maxEp) + '%"></i></span>'
            + '<span class="v">' + fmtInt(e.value) + '</span></div>';
        }).join('')
      : '<div class="kd-empty-mini">no endpoint traffic yet</div>';

    var life = (d.lifecycle && d.lifecycle.length)
      ? d.lifecycle.map(function (l) {
          var g = KD_LIFE[l.action] || ['•', (l.action || 'event')];
          return '<div class="kd-life"><span class="g">' + g[0] + '</span><span class="a">' + esc(g[1])
            + '</span><span class="t">' + esc(relTime(l.ts)) + '</span></div>';
        }).join('')
      : '<div class="kd-empty-mini">no lifecycle events recorded</div>';

    var recent = (d.recent && d.recent.length)
      ? d.recent.map(function (r) {
          return '<div class="kd-req" data-rid="' + esc(r.id) + '" role="button" tabindex="0">'
            + '<span class="m">' + esc(r.method || '') + '</span>'
            + '<span class="p">' + esc(r.path || '') + '</span>'
            + '<span class="st ' + statusClass(r.status) + '">' + esc(r.status) + '</span>'
            + '<span class="t">' + esc(relTime(r.ts)) + '</span><span class="chev">›</span></div>';
        }).join('')
      : '<div class="kd-empty-mini">no requests on record</div>';

    var owner = d.owner || {};
    el.innerHTML =
      '<div class="dl-head kd-head"><div class="ttl"><div class="sub">api key' + (d.environment ? ' · ' + esc(d.environment) : '') + '</div>'
        + '<div class="main">' + esc(d.name || d.public_id) + '</div></div>'
        + '<button class="x" id="dl-dw-x" aria-label="Close">×</button></div>'
      + '<div class="dl-body kd-body">'
      + '<div class="kd-meta">'
        + '<span class="kd-pill ' + stCls + '">' + esc(st) + '</span>'
        + (d.group ? '<span class="kd-pill grp" data-grp="1">⬡ ' + esc(d.group) + '</span>' : '')
        + '<span class="kd-owner" data-uid="' + esc(owner.id || '') + '" role="button" tabindex="0">'
          + '<i>◷</i>' + esc(owner.display_name || ('user ' + (owner.id || '?'))) + '</span>'
        + '<span class="kd-pubid" title="public id">' + esc(d.public_id) + '</span>'
      + '</div>'
      + '<div class="kd-usage">'
        + '<span class="u-item ' + (st === 'active' ? 'on' : 'off') + '"><i></i>' + (st === 'active' ? 'enabled' : esc(st)) + '</span>'
        + '<span class="u-item ' + (d.in_use ? 'use' : 'muted') + '"><i></i>' + (d.in_use ? fmtInt(s.requests || 0) + ' requests served' : 'minted but never used') + '</span>'
        + '<span class="u-item last">last used ' + (d.last_used ? esc(relTime(d.last_used)) : 'never') + '</span>'
      + '</div>'
      + '<div class="kd-kpis">' + kpis + '</div>'
      + '<div class="kd-sec"><h5>30-day activity</h5>' + kdSpark(d.series) + '</div>'
      + (kdStatusMix(d.status_mix) ? '<div class="kd-sec"><h5>status mix</h5>' + kdStatusMix(d.status_mix) + '</div>' : '')
      + '<div class="kd-sec"><h5>busiest endpoints</h5>' + topEps + '</div>'
      + '<div class="kd-sec"><h5>scopes</h5><div class="kd-scopes">' + scopes + '</div></div>'
      + '<div class="kd-sec"><h5>lifecycle</h5><div class="kd-lifeline">' + life + '</div></div>'
      + '<div class="kd-sec"><h5>recent requests</h5><div class="kd-reqs">' + recent + '</div></div>'
      + '</div>';
    bindClose('dl-dw-x', closeDrillDrawer);

    // cross-drill: recent request → request drawer; owner → user dossier
    var reqs = el.querySelector('.kd-reqs');
    if (reqs) reqs.addEventListener('click', function (e) {
      var row = e.target.closest ? e.target.closest('.kd-req[data-rid]') : null;
      if (row) openDetail('request', row.getAttribute('data-rid'));
    });
    var ownEl = el.querySelector('.kd-owner[data-uid]');
    if (ownEl && owner.id) ownEl.addEventListener('click', function () { openDetail('user', owner.id); });
  }
  // ── P5 · rich SCAN dossier (wide · spine-nav · region map) ──────────────
  var SD_ICON = {
    overview: '<path d="M3 13a9 9 0 0 1 18 0M12 3v4M12 13a6 6 0 0 0 6-6c0-4-6 0-6 0s-6-4-6 0a6 6 0 0 0 6 6Z"/>',
    confidence: '<path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/>',
    location: '<circle cx="12" cy="10" r="3"/><path d="M12 21c5-5 7-8 7-11a7 7 0 1 0-14 0c0 3 2 6 7 11Z"/>',
    payload: '<path d="M8 6 3 12l5 6M16 6l5 6-5 6"/>',
    provenance: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'
  };
  var SD_SEVCOL = { mild: 'var(--accent)', moderate: 'var(--ochre)', severe: 'var(--crimson)' };
  function sdNode(sec, label, on) {
    return '<div class="dl-snode' + (on ? ' on' : '') + '" data-sec="sd-' + sec + '" role="button" tabindex="0">'
      + '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
      + SD_ICON[sec] + '</svg><span class="tip">' + label + '</span></div>';
  }
  function sdProbBars(probs, predicted) {
    if (!probs || typeof probs !== 'object') return '<div class="kd-empty-mini">no class probabilities in payload</div>';
    var arr = Object.keys(probs).map(function (k) { return { k: k, v: Number(probs[k]) || 0 }; })
      .sort(function (a, b) { return b.v - a.v; }).slice(0, 10);
    if (!arr.length) return '<div class="kd-empty-mini">no class probabilities in payload</div>';
    var max = arr[0].v || 1;
    return arr.map(function (p) {
      var hot = (p.k === predicted);
      return '<div class="sd-prob' + (hot ? ' hot' : '') + '"><span class="lab">' + esc(p.k.replace(/_/g, ' ')) + '</span>'
        + '<span class="bar"><i style="width:' + Math.round(100 * p.v / max) + '%"></i></span>'
        + '<span class="v">' + (p.v * 100).toFixed(1) + '%</span></div>';
    }).join('');
  }
  function renderScanDossier(d) {
    var el = document.getElementById('dl-dossier');
    if (!d || d.found === false) {
      el.innerHTML = '<div class="dl-spine"></div><div class="dl-pane"><div class="dl-head"><div class="ttl"><div class="main">scan not found</div></div><button class="x" id="dl-ds-x">×</button></div>'
        + '<div class="dl-body">' + dlEmpty('seed', "this scan has drifted off the record", 'it may have been purged') + '</div></div>';
      bindClose('dl-ds-x', closeDossier); return;
    }
    var conf = d.confidence != null ? Math.round(d.confidence * 100) : null;
    var sev = (d.severity || '').toLowerCase();
    var pl = d.payload || {};
    var owner = d.owner || {};
    var hasImg = d.has_image;
    var imgUrl = '/api/account/admin/scan-image/' + encodeURIComponent(d.scan_uid);

    var spine = sdNode('overview', 'Overview', true) + sdNode('confidence', 'Confidence')
      + sdNode('location', 'Location') + sdNode('payload', 'Payload') + sdNode('provenance', 'Provenance');

    var overview =
      '<div class="dl-dsec" id="sd-overview" data-sec="sd-overview"><div class="dl-dsec-h"><span class="pip"></span>Overview</div>'
      + '<div class="sd-hero">'
        + '<div class="sd-img">' + (hasImg
            ? '<img src="' + imgUrl + '" alt="leaf scan" loading="lazy" onerror="this.parentElement.classList.add(\'noimg\')">'
            : '') + '<span class="sd-noimg-tag">no image stored</span></div>'
        + '<div class="sd-hero-info">'
          + '<div class="sd-dx">' + esc((d.diagnosis || '—').replace(/_/g, ' ')) + '</div>'
          + '<div class="sd-chips">'
            + '<span class="sd-chip crop">' + esc(d.crop || '—') + '</span>'
            + (sev ? '<span class="sd-chip" style="color:' + (SD_SEVCOL[sev] || 'var(--text-dim)') + ';border-color:' + (SD_SEVCOL[sev] || 'var(--line)') + '">' + esc(sev) + '</span>' : '')
            + (d.tier ? '<span class="sd-chip">tier ' + esc(d.tier) + '</span>' : '')
            + (d.is_ood ? '<span class="sd-chip ood">out-of-distribution</span>' : '')
          + '</div>'
          + (conf != null ? '<div class="sd-conf"><div class="sd-conf-ring" style="--p:' + conf + '"><span>' + conf + '<small>%</small></span></div>'
              + '<div class="sd-conf-lbl">model confidence</div></div>' : '')
        + '</div>'
      + '</div></div>';

    var confidence =
      '<div class="dl-dsec" id="sd-confidence" data-sec="sd-confidence"><div class="dl-dsec-h"><span class="pip"></span>Class confidence</div>'
      + '<div class="sd-probs">' + sdProbBars(pl.all_class_probabilities, d.diagnosis) + '</div></div>';

    var location =
      '<div class="dl-dsec" id="sd-location" data-sec="sd-location"><div class="dl-dsec-h"><span class="pip"></span>Where it was taken</div>'
      + '<div class="sd-geo">'
        + '<div class="sd-map" id="sd-map"><div class="sd-map-load">charting the district…</div></div>'
        + '<div class="sd-geo-meta">'
          + '<div class="sd-geo-row"><span>district</span><b>' + esc(d.geo_district || '—') + '</b></div>'
          + '<div class="sd-geo-row"><span>state</span><b>' + esc(d.geo_state || '—') + '</b></div>'
          + '<div class="sd-geo-row"><span>country</span><b>' + esc(d.geo_cc || '—') + '</b></div>'
          + (d.latitude != null ? '<div class="sd-geo-row"><span>gps</span><b class="mono">' + Number(d.latitude).toFixed(4) + ', ' + Number(d.longitude).toFixed(4) + '</b></div>' : '')
        + '</div>'
      + '</div></div>';

    var payloadSec =
      '<div class="dl-dsec" id="sd-payload" data-sec="sd-payload"><div class="dl-dsec-h"><span class="pip"></span>Inference payload</div>'
      + '<pre class="sd-json">' + esc(JSON.stringify(pl, null, 2)) + '</pre></div>';

    var provenance =
      '<div class="dl-dsec" id="sd-provenance" data-sec="sd-provenance"><div class="dl-dsec-h"><span class="pip"></span>Provenance</div>'
      + '<div class="sd-prov">'
        + '<div class="sd-prov-row"><span>scan id</span><b class="mono">' + esc(d.scan_uid) + '</b></div>'
        + '<div class="sd-prov-row"><span>captured</span><b>' + esc(relTime(d.captured_at)) + '</b></div>'
        + '<div class="sd-prov-row"><span>processed</span><b>' + esc(relTime(d.processed_at)) + '</b></div>'
        + (d.processing_ms != null ? '<div class="sd-prov-row"><span>compute</span><b>' + fmtInt(d.processing_ms) + ' ms</b></div>' : '')
        + '<div class="sd-prov-row"><span>uploaded by</span><b class="sd-owner" data-uid="' + esc(owner.id || '') + '" role="button" tabindex="0">'
          + esc(owner.display_name || owner.username || ('user ' + (owner.id || '?'))) + ' ›</b></div>'
      + '</div></div>';

    el.innerHTML = '<div class="dl-spine">' + spine + '</div>'
      + '<div class="dl-pane">'
        + '<div class="dl-head"><div class="ttl"><div class="sub">inference · ' + esc(d.crop || 'scan') + '</div>'
          + '<div class="main">' + esc((d.diagnosis || 'scan').replace(/_/g, ' ')) + '</div></div>'
          + '<button class="x" id="dl-ds-x" aria-label="Close">×</button></div>'
        + '<div class="dl-body" id="sd-body">' + overview + confidence + location + payloadSec + provenance + '</div>'
      + '</div>';
    bindClose('dl-ds-x', closeDossier);
    wireDossierSpine(el, 'sd-body');

    // async: region map
    adminFetch('/api/account/admin/scan-region/' + encodeURIComponent(d.scan_uid)).then(function (rg) {
      var box = document.getElementById('sd-map'); if (!box) return;
      box.innerHTML = _sdRegion(rg);
    });

    // cross-drill: owner → user dossier
    var ow = el.querySelector('.sd-owner[data-uid]');
    if (ow && owner.id) ow.addEventListener('click', function () { openDetail('user', owner.id); });
  }
  // region SVG builder — neighbours muted, target district highlighted, GPS pin
  function _sdRegion(rg) {
    if (!rg || rg.found === false) {
      return '<div class="sd-map-fallback"><div class="es-seed-wrap">' + ES_ANIM.seed + '</div>'
        + '<div class="t">no district boundary for ' + esc((rg && (rg.state || rg.district)) || 'this location') + '</div>'
        + ((rg && rg.lat != null) ? '<div class="s mono">' + Number(rg.lat).toFixed(3) + ', ' + Number(rg.lon).toFixed(3) + '</div>' : '') + '</div>';
    }
    var neigh = (rg.neighbour_paths || []).map(function (p) { return '<path class="nb" d="' + p + '"/>'; }).join('');
    var mk = rg.marker ? '<circle class="mk" cx="' + rg.marker.x + '" cy="' + rg.marker.y + '" r="5"/>'
      + '<circle class="mk-pulse" cx="' + rg.marker.x + '" cy="' + rg.marker.y + '" r="5"/>' : '';
    return '<svg class="sd-map-svg" viewBox="' + esc(rg.viewbox) + '" preserveAspectRatio="xMidYMid meet">'
      + neigh + '<path class="tgt" d="' + rg.target_path + '"/>' + mk + '</svg>'
      + '<div class="sd-map-cap"><b>' + esc(rg.district) + '</b> · ' + esc(rg.state) + '</div>';
  }

  // ══ ADM-X Phase D · PREDICTION dossier (real website inference + heatmap) ══
  var PD_ICON = {
    diagnosis: '<path d="M12 3v4M12 13a6 6 0 0 0 6-6c0-4-6-9-6-9s-6 5-6 9a6 6 0 0 0 6 6Z"/>',
    classes: '<path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/>',
    signals: '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M6 8v3a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3V8M12 14v2"/>',
    calibration: '<path d="M3 13a9 9 0 0 1 18 0"/><path d="M12 13l4-3"/>',
    timing: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    guidance: '<path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V18h6v-1.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2Z"/>',
    capture: '<path d="M3 7h3l2-2h8l2 2h3v12H3z"/><circle cx="12" cy="13" r="3.5"/>',
    raw: '<path d="M8 6 3 12l5 6M16 6l5 6-5 6"/>'
  };
  var PD_SEVCOL = { mild: 'var(--accent)', moderate: 'var(--ochre)', severe: 'var(--crimson)' };
  function pdNode(sec, label, on) {
    return '<div class="dl-snode' + (on ? ' on' : '') + '" data-sec="pd-' + sec + '" role="button" tabindex="0">'
      + '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
      + PD_ICON[sec] + '</svg><span class="tip">' + label + '</span></div>';
  }
  function pdProbBars(probs, predicted) {
    if (!probs || typeof probs !== 'object') return '<div class="kd-empty-mini">no class probabilities recorded</div>';
    var arr = Object.keys(probs).map(function (k) { return { k: k, v: Number(probs[k]) || 0 }; })
      .sort(function (a, b) { return b.v - a.v; });
    if (!arr.length) return '<div class="kd-empty-mini">no class probabilities recorded</div>';
    var max = arr[0].v || 1;
    return arr.map(function (p) {
      var hot = (p.k === predicted);
      return '<div class="sd-prob' + (hot ? ' hot' : '') + '"><span class="lab">' + esc(p.k.replace(/_/g, ' ')) + '</span>'
        + '<span class="bar"><i style="width:' + Math.round(100 * p.v / max) + '%"></i></span>'
        + '<span class="v">' + (p.v * 100).toFixed(1) + '%</span></div>';
    }).join('');
  }
  function pdSignals(sig) {
    if (!sig || typeof sig !== 'object') return '<div class="kd-empty-mini">no per-model signals recorded</div>';
    var keys = Object.keys(sig);
    if (!keys.length) return '<div class="kd-empty-mini">no per-model signals recorded</div>';
    return keys.map(function (k) {
      var s = sig[k] || {};
      var arg = s.argmax || s.prediction || s.diagnosis || '—';
      var pr = s.top_prob != null ? s.top_prob : (s.confidence != null ? s.confidence : null);
      return '<div class="pd-sig"><span class="m">' + esc(k) + '</span>'
        + '<span class="a">' + esc(String(arg).replace(/_/g, ' ')) + '</span>'
        + '<span class="bar"><i style="width:' + (pr != null ? Math.round(pr * 100) : 0) + '%"></i></span>'
        + '<span class="p">' + (pr != null ? (pr * 100).toFixed(1) + '%' : '—') + '</span></div>';
    }).join('');
  }
  function pdGate(gate) {
    if (!gate) return '';
    var entries = [];
    if (Array.isArray(gate)) entries = gate.map(function (g, i) { return [String(i), g]; });
    else if (typeof gate === 'object') entries = Object.keys(gate).map(function (k) { return [k, gate[k]]; });
    entries = entries.filter(function (e) { return typeof e[1] === 'number'; });
    if (!entries.length) return '';
    var tot = entries.reduce(function (a, b) { return a + b[1]; }, 0) || 1;
    return '<div class="pd-gate">' + entries.map(function (e) {
      return '<div class="pd-gate-row"><span class="m">' + esc(e[0]) + '</span>'
        + '<span class="bar"><i style="width:' + Math.round(100 * e[1] / tot) + '%"></i></span>'
        + '<span class="p">' + (e[1] <= 1 ? (e[1] * 100).toFixed(0) + '%' : fmtInt(e[1])) + '</span></div>';
    }).join('') + '</div>';
  }
  function pdTimingWF(t) {
    var stages = [['validation', t.validation], ['router', t.router], ['specialist', t.specialist], ['calibration', t.calibration]];
    var present = stages.filter(function (s) { return s[1] != null; });
    var total = t.total != null ? t.total : present.reduce(function (a, b) { return a + (b[1] || 0); }, 0);
    if (!present.length && t.total == null) return '<div class="kd-empty-mini">stage timings not recorded</div>';
    var maxv = Math.max(total || 1, present.reduce(function (a, b) { return Math.max(a, b[1] || 0); }, 0)) || 1;
    var rows = present.map(function (s) {
      return '<div class="pd-wf-row"><span class="l">' + s[0] + '</span>'
        + '<span class="track"><i style="width:' + Math.round(100 * (s[1] || 0) / maxv) + '%"></i></span>'
        + '<span class="v">' + fmtInt(s[1]) + ' ms</span></div>';
    }).join('');
    rows += '<div class="pd-wf-row total"><span class="l">total</span>'
      + '<span class="track"><i style="width:100%"></i></span>'
      + '<span class="v">' + (total != null ? fmtInt(total) + ' ms' : '—') + '</span></div>';
    return rows;
  }
  function pdGuidanceText(d) {
    var r = d.result || {};
    var blocks = [
      ['diagnosis call', r.output_message],
      ['treatment', r.treatment_recommendation],
      ['monitoring', r.monitoring_guidance],
      ['differential', r.differential_guidance]
    ].filter(function (b) { return b[1]; });
    if (!blocks.length) return '<div class="kd-empty-mini">no guidance text in this record</div>';
    return blocks.map(function (b) {
      return '<div class="pd-guide"><div class="h">' + esc(b[0]) + '</div><p>' + esc(String(b[1])) + '</p></div>';
    }).join('');
  }
  function renderPredictionDossier(d) {
    var el = document.getElementById('dl-dossier');
    if (!d) { _basicDossier('prediction', {}, 'inference'); return; }
    var conf = d.confidence != null ? Math.round(d.confidence * 100) : null;
    var r = d.result || {};
    var probs = d.all_class_probabilities || r.all_class_probabilities;
    var conformal = d.conformal_set || r.conformal_prediction_set;
    var ua = (d.device || {});
    var reduced = (typeof REDUCED !== 'undefined') ? REDUCED : false;

    var spine = pdNode('diagnosis', 'Diagnosis', true) + pdNode('classes', 'Classes')
      + pdNode('signals', 'Signals & gate') + pdNode('calibration', 'Calibration')
      + pdNode('timing', 'Pipeline') + pdNode('guidance', 'Guidance')
      + pdNode('capture', 'Capture') + pdNode('raw', 'Raw');

    var imgUrl = '/api/account/admin/prediction-image/' + encodeURIComponent(d.id);
    var heatUrl = '/api/account/admin/prediction-heatmap/' + encodeURIComponent(d.id);
    var uaArr = []; if (r.uncertainty_aleatoric != null) uaArr.push(['aleatoric', r.uncertainty_aleatoric]);
    if (r.uncertainty_epistemic != null) uaArr.push(['epistemic', r.uncertainty_epistemic]);

    var diagnosis =
      '<div class="dl-dsec" id="pd-diagnosis" data-sec="pd-diagnosis"><div class="dl-dsec-h"><span class="pip"></span>Diagnosis</div>'
      + '<div class="sd-hero pd-hero">'
        + '<div class="pd-img' + (d.has_heatmap ? '' : ' nohm') + (reduced ? ' reduced' : '') + '" id="pd-img" role="button" tabindex="0" aria-pressed="false" aria-label="leaf scan — tap to toggle attention map">'
          + '<img class="raw" src="' + imgUrl + '" alt="leaf scan" loading="lazy" onerror="this.closest(\'.pd-img\').classList.add(\'broken\')">'
          + (d.has_heatmap ? '<img class="heat" src="' + heatUrl + '" alt="Grad-CAM attention map" loading="lazy">' : '')
          + '<span class="pd-sweep" aria-hidden="true"></span>'
          + '<span class="pd-img-hint">' + (d.has_heatmap ? 'tap for attention map' : 'no attention map') + '</span>'
          + '<span class="pd-legend" aria-hidden="true"><i></i>cool → hot</span>'
          + '<span class="pd-noheat-toast">no attention map was captured for this inference</span>'
        + '</div>'
        + '<div class="sd-hero-info">'
          + '<div class="sd-dx">' + esc((d.diagnosis || '—').replace(/_/g, ' ')) + '</div>'
          + '<div class="sd-chips">'
            + '<span class="sd-chip crop">' + esc(d.crop || '—') + '</span>'
            + (d.severity ? '<span class="sd-chip" style="color:' + (PD_SEVCOL[String(d.severity).toLowerCase()] || 'var(--text-dim)') + '">' + esc(d.severity) + '</span>' : '')
            + (d.tier ? '<span class="sd-chip">tier ' + esc(d.tier) + '</span>' : '')
            + (Number(d.is_ood) ? '<span class="sd-chip ood">out-of-distribution</span>' : '')
            + (Number(d.calibration_warning) ? '<span class="sd-chip ood">calibration warning</span>' : '')
            + (Number(d.confidence_outlier) ? '<span class="sd-chip ood">confidence outlier</span>' : '')
            + (d.guest ? '<span class="sd-chip">guest</span>' : '')
          + '</div>'
          + (conf != null ? '<div class="sd-conf"><div class="sd-conf-ring" style="--p:' + conf + '"><span>' + conf + '<small>%</small></span></div>'
              + '<div class="sd-conf-lbl">model confidence' + (conformal && conformal.length ? '<br><span class="mono" style="font-size:10px;color:var(--text-faint)">conformal set: ' + esc(conformal.length) + ' class' + (conformal.length === 1 ? '' : 'es') + '</span>' : '') + '</div></div>' : '')
          + (uaArr.length ? '<div class="pd-unc">' + uaArr.map(function (u) {
              return '<span class="pd-unc-i"><b>' + (Number(u[1]) * 100).toFixed(1) + '%</b>' + esc(u[0]) + ' uncertainty</span>'; }).join('') + '</div>' : '')
        + '</div>'
      + '</div></div>';

    var classes =
      '<div class="dl-dsec" id="pd-classes" data-sec="pd-classes"><div class="dl-dsec-h"><span class="pip"></span>Class confidence</div>'
      + '<div class="sd-probs">' + pdProbBars(probs, d.diagnosis) + '</div>'
      + (conformal && conformal.length ? '<div class="pd-conformal"><span class="h">conformal prediction set</span>'
          + conformal.map(function (cz) { return '<span class="pd-cz">' + esc(String(cz).replace(/_/g, ' ')) + '</span>'; }).join('') + '</div>' : '')
      + '</div>';

    var signals =
      '<div class="dl-dsec" id="pd-signals" data-sec="pd-signals"><div class="dl-dsec-h"><span class="pip"></span>Per-model signals &amp; gate</div>'
      + '<div class="pd-sigs">' + pdSignals(d.signal_predictions) + '</div>'
      + (pdGate(d.gate_decision_path) ? '<h5 class="pd-subh">gate weights</h5>' + pdGate(d.gate_decision_path) : '')
      + (r.conflict_type ? '<div class="pd-conflict">conflict type: <b>' + esc(r.conflict_type) + '</b></div>' : '')
      + '</div>';

    var calibration =
      '<div class="dl-dsec" id="pd-calibration" data-sec="pd-calibration"><div class="dl-dsec-h"><span class="pip"></span>Calibration &amp; uncertainty</div>'
      + '<div class="pd-kv">'
        + '<div><span>conformal set size</span><b>' + esc(d.conformal_set_size != null ? d.conformal_set_size : (conformal && conformal.length) || '—') + '</b></div>'
        + '<div><span>aleatoric</span><b>' + (r.uncertainty_aleatoric != null ? (r.uncertainty_aleatoric * 100).toFixed(1) + '%' : '—') + '</b></div>'
        + '<div><span>epistemic</span><b>' + (r.uncertainty_epistemic != null ? (r.uncertainty_epistemic * 100).toFixed(1) + '%' : '—') + '</b></div>'
        + '<div><span>calibration warning</span><b class="' + (Number(d.calibration_warning) ? 'warn' : '') + '">' + (Number(d.calibration_warning) ? 'yes' : 'no') + '</b></div>'
        + '<div><span>confidence outlier</span><b class="' + (Number(d.confidence_outlier) ? 'warn' : '') + '">' + (Number(d.confidence_outlier) ? 'yes' : 'no') + '</b></div>'
        + '<div><span>OOD</span><b class="' + (Number(d.is_ood) ? 'warn' : '') + '">' + (Number(d.is_ood) ? 'flagged' : 'in-distribution') + '</b></div>'
      + '</div></div>';

    var timing =
      '<div class="dl-dsec" id="pd-timing" data-sec="pd-timing"><div class="dl-dsec-h"><span class="pip"></span>Pipeline timing</div>'
      + '<div class="pd-wf">' + pdTimingWF(d.timings || {}) + '</div></div>';

    var guidance =
      '<div class="dl-dsec" id="pd-guidance" data-sec="pd-guidance"><div class="dl-dsec-h"><span class="pip"></span>Guidance</div>'
      + pdGuidanceText(d) + '</div>';

    function kv(label, val) { return '<div class="sd-prov-row"><span>' + label + '</span><b>' + esc(val == null || val === '' ? '—' : val) + '</b></div>'; }
    var capture =
      '<div class="dl-dsec" id="pd-capture" data-sec="pd-capture"><div class="dl-dsec-h"><span class="pip"></span>Capture &amp; device</div>'
      + '<div class="pd-cap-grid">'
        + '<div class="pd-cap-col"><h5>client</h5>'
          + kv('agent', ua.ua) + kv('country', ua.country) + kv('region', ua.region) + kv('city', ua.city)
          + ((ua.lat != null) ? kv('gps', Number(ua.lat).toFixed(4) + ', ' + Number(ua.lon).toFixed(4)) : '') + '</div>'
        + '<div class="pd-cap-col"><h5>image</h5>'
          + kv('camera', ua.camera) + kv('dimensions', (ua.width && ua.height) ? (ua.width + '×' + ua.height) : null)
          + kv('format', ua.mimetype) + kv('phash', ua.phash) + '</div>'
        + '<div class="pd-cap-col"><h5>model &amp; request</h5>'
          + kv('deployment', (d.model || {}).deployment_version) + kv('gpu', Number((d.model || {}).gpu_used) ? 'yes' : 'no')
          + kv('cold start', Number((d.model || {}).cold_start) ? 'yes' : 'no')
          + kv('endpoint', (d.request || {}).endpoint) + kv('captured', relTime(d.created_at)) + '</div>'
      + '</div>'
      + (d.owner ? '<div class="pd-owner-row">uploaded by <b class="sd-owner" data-uid="' + esc(d.owner.id || '') + '" role="button" tabindex="0">'
          + esc(d.owner.display_name || d.owner.username || ('user ' + (d.owner.id || '?'))) + ' ›</b></div>'
          : (d.guest ? '<div class="pd-owner-row">uploaded by <b>an anonymous guest</b></div>' : ''))
      + '</div>';

    var raw =
      '<div class="dl-dsec" id="pd-raw" data-sec="pd-raw"><div class="dl-dsec-h"><span class="pip"></span>Raw response</div>'
      + '<pre class="sd-json">' + esc(JSON.stringify(r, null, 2)) + '</pre></div>';

    el.innerHTML = '<div class="dl-spine">' + spine + '</div>'
      + '<div class="dl-pane">'
        + '<div class="dl-head"><div class="ttl"><div class="sub">website inference · ' + esc(d.crop || 'leaf') + (d.guest ? ' · guest' : '') + '</div>'
          + '<div class="main">' + esc((d.diagnosis || 'inference').replace(/_/g, ' ')) + '</div></div>'
          + '<button class="x" id="dl-ds-x" aria-label="Close">×</button></div>'
        + '<div class="dl-body" id="pd-body">' + diagnosis + classes + signals + calibration + timing + guidance + capture + raw + '</div>'
      + '</div>';
    bindClose('dl-ds-x', closeDossier);
    wireDossierSpine(el, 'pd-body');
    wirePredHeatmap(el, d.has_heatmap);
    var ow = el.querySelector('.sd-owner[data-uid]');
    if (ow && d.owner && d.owner.id) ow.addEventListener('click', function () { openDetail('user', d.owner.id); });
  }
  // Smooth click→sweep→heatmap toggle on the inference image.
  function wirePredHeatmap(el, hasHeat) {
    var box = el.querySelector('#pd-img'); if (!box) return;
    var on = false, busy = false;
    function toggle() {
      if (!hasHeat) { box.classList.add('noheat-flash'); setTimeout(function () { box.classList.remove('noheat-flash'); }, 1600); return; }
      if (busy) return; busy = true;
      on = !on;
      box.setAttribute('aria-pressed', on ? 'true' : 'false');
      box.classList.add('sweeping');
      box.classList.toggle('on', on);
      setTimeout(function () { box.classList.remove('sweeping'); busy = false; }, 680);
    }
    box.addEventListener('click', toggle);
    box.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  }

  // ── P6 · rich USER dossier (wide · spine-nav · engagement empty-states) ──
  var UD_ICON = {
    identity: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    vitals: '<path d="M3 12h4l2 6 4-14 2 8h6"/>',
    timeline: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    reliability: '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/><path d="M12 9v3"/><circle cx="12" cy="15.5" r=".6" fill="currentColor"/>',
    engagement: '<path d="M3 3v18h18"/><path d="M7 14l3-4 3 3 5-7"/>',
    api: '<path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/>',
    keys: '<circle cx="8" cy="15" r="4"/><path d="m10.85 12.15 7.65-7.65"/><path d="m18 5 2 2"/><path d="m15 8 2 2"/>',
    inferences: '<path d="M12 3v4M12 21a6 6 0 0 0 6-6c0-4-6-9-6-9s-6 5-6 9a6 6 0 0 0 6 6Z"/>',
    sessions: '<rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
    footprint: '<circle cx="12" cy="10" r="3"/><path d="M12 21c5-5 7-8 7-11a7 7 0 1 0-14 0c0 3 2 6 7 11Z"/>',
    account: '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/>'
  };
  var UD_DISEASE_COL = ['var(--accent)', 'var(--teal)', 'var(--ochre)', 'var(--violet)', 'var(--magenta)', 'var(--crimson)', '#7da7ff', '#b6e35a'];
  function udNode(sec, label, on) {
    return '<div class="dl-snode' + (on ? ' on' : '') + '" data-sec="ud-' + sec + '" role="button" tabindex="0">'
      + '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
      + UD_ICON[sec] + '</svg><span class="tip">' + label + '</span></div>';
  }
  function udAccountAge(iso) {
    if (!iso) return '—';
    var ms = Date.now() - new Date(String(iso).replace(' ', 'T')).getTime();
    var dys = Math.floor(ms / 86400000);
    if (isNaN(dys)) return '—';
    return dys < 1 ? 'today' : dys + ' day' + (dys === 1 ? '' : 's');
  }

  // ── ADM-U · User-360 dossier helpers ──────────────────────────────────────
  var UD_HEALTH = {
    healthy: { cls: 'ok',   label: 'Healthy' },
    at_risk: { cls: 'warn', label: 'At risk' },
    dormant: { cls: 'mute', label: 'Dormant' },
    new:     { cls: 'teal', label: 'New' }
  };
  var UD_PERSONA = {
    api_first: { cls: 'violet', label: 'API-first', ic: 'M4 19V9M10 19V5M16 19v-7M22 19H2' },
    web:       { cls: 'teal',   label: 'Web user',  ic: 'M2 12h20M12 2a15 15 0 0 1 0 20 15 15 0 0 1 0-20' },
    hybrid:    { cls: 'ok',     label: 'Hybrid',    ic: 'M12 3v18M3 12h18' },
    new:       { cls: 'mute',   label: 'New',       ic: 'M12 5v14M5 12h14' },
    observer:  { cls: 'mute',   label: 'Observer',  ic: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z' }
  };
  function udHealthChip(h) {
    if (!h) return '';
    var m = UD_HEALTH[h.state] || UD_HEALTH.healthy;
    return '<span class="ud-chip ' + m.cls + '" title="' + esc(h.reason || '') + '">'
      + '<span class="dot"></span>' + m.label + (h.score != null ? ' · ' + h.score : '') + '</span>';
  }
  function udPersonaChip(p) {
    if (!p) return '';
    var m = UD_PERSONA[p.type] || UD_PERSONA.observer;
    return '<span class="ud-chip ' + m.cls + '" title="' + esc(p.desc || '') + '">'
      + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="' + m.ic + '"/></svg>'
      + esc(m.label) + '</span>';
  }
  // unified event-timeline icon/colour per lane
  var UD_EV = {
    identity:  { cls: 'ok',     ic: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM5 21a7 7 0 0 1 14 0' },
    anomaly:   { cls: 'warn',   ic: 'M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0ZM12 9v4M12 17h.01' },
    keys:      { cls: 'violet', ic: 'M8 15a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm2.8-2.8 7.7-7.7M18 5l2 2' },
    inference: { cls: 'teal',   ic: 'M12 3v4M12 21a6 6 0 0 0 6-6c0-4-6-9-6-9s-6 5-6 9a6 6 0 0 0 6 6Z' }
  };
  // clickable KPI tile for the vitals band. drill = data-drill for H3 wiring.
  function udKpi(value, label, opts) {
    opts = opts || {};
    var cls = 'ud-kt' + (opts.drill ? ' clickable' : '') + (opts.tone ? ' ' + opts.tone : '');
    return '<div class="' + cls + '"' + (opts.drill ? ' data-drill="' + opts.drill + '" role="button" tabindex="0"' : '')
      + (opts.title ? ' title="' + esc(opts.title) + '"' : '') + '>'
      + '<span class="v">' + value + '</span><span class="l">' + esc(label) + '</span>'
      + (opts.drill ? '<span class="go" aria-hidden="true">↗</span>' : '') + '</div>';
  }
  function udFactor(label, val) {
    var v = Math.max(0, Math.min(100, val || 0));
    var tone = v >= 70 ? 'ok' : (v >= 40 ? 'warn' : 'err');
    return '<div class="ud-fac"><span class="fl">' + esc(label) + '</span>'
      + '<span class="fb"><i class="' + tone + '" style="width:' + v + '%"></i></span>'
      + '<span class="fv">' + v + '</span></div>';
  }
  function renderUserDossier(d) {
    var el = document.getElementById('dl-dossier');
    if (!d) { _basicDossier('user', {}, 'user'); return; }
    var eng = d.engagement || {}, api = d.api || {}, inf = d.inferences || {}, loc = d.location || {};
    var rel = d.reliability || {}, sx = d.sessions_detail || {}, evs = d.events || [], keys = d.keys || [];
    var persona = d.persona || {}, health = d.health || {};
    var engEmpty = !(eng.total_active_ms || eng.page_views || eng.sessions || eng.clicks);
    var apiFirst = (persona.type === 'api_first');
    var hasApi = (api.requests || 0) > 0;
    var hasKeys = (d.key_count || 0) > 0;
    var hasDistricts = (loc.districts || []).length > 0;

    // ── spine (only nodes for sections we actually render) ──
    var spine = udNode('identity', 'Identity', true) + udNode('vitals', 'Vitals')
      + udNode('timeline', 'Timeline')
      + (hasApi ? udNode('reliability', 'Reliability') : '')
      + (hasKeys ? udNode('keys', 'API keys') : '')
      + udNode('inferences', 'Inferences') + udNode('engagement', 'Engagement')
      + udNode('sessions', 'Sessions')
      + (hasDistricts ? udNode('footprint', 'Footprint') : '')
      + udNode('account', 'Account');

    // ── 1 · Identity (persona + health chips, copyable contacts) ──
    var roleBadge = d.is_admin
      ? '<span class="ud-role admin">admin' + (d.admin_via ? ' · ' + esc(d.admin_via) : '') + '</span>'
      : '<span class="ud-role">' + esc(d.role || 'collector') + '</span>';
    var seed = (typeof d.pressed_leaf_seed === 'number') ? d.pressed_leaf_seed : (d.id % 6);
    function copyCell(label, val, copy) {
      var v = esc(val || '—');
      return '<div><span>' + label + '</span><b class="mono' + (copy && val ? ' ud-copy' : '') + '"'
        + (copy && val ? ' data-copy="' + esc(val) + '" role="button" tabindex="0" title="copy"' : '') + '>' + v
        + (copy && val ? '<svg class="cpi" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>' : '') + '</b></div>';
    }
    var identity =
      '<div class="dl-dsec" id="ud-identity" data-sec="ud-identity"><div class="dl-dsec-h"><span class="pip"></span>Identity</div>'
      + '<div class="ud-id">'
        + '<div class="ud-avatar s' + seed + '">' + esc((d.display_name || d.username || '?').slice(0, 1).toUpperCase()) + '</div>'
        + '<div class="ud-id-main"><div class="ud-name">' + esc(d.display_name || d.username || ('user ' + d.id)) + ' ' + roleBadge + '</div>'
          + '<div class="ud-chips">' + udPersonaChip(persona) + udHealthChip(health) + '</div>'
          + (health.reason ? '<div class="ud-health-reason">' + esc(health.reason) + '</div>' : '')
          + '<div class="ud-id-grid">'
            + copyCell('username', '@' + (d.username || '—'), false)
            + copyCell('email', d.email, true)
            + copyCell('mobile', d.mobile_masked, false)
            + '<div><span>joined</span><b>' + esc(relTime(d.created_at)) + ' · ' + udAccountAge(d.created_at) + ' ago</b></div>'
            + '<div><span>last seen</span><b>' + esc(relTime(d.last_seen_at)) + '</b></div>'
            + '<div><span>user id</span><b class="mono">#' + esc(d.id) + '</b></div>'
          + '</div></div>'
      + '</div></div>';

    // ── 2 · Vitals (clickable KPI band + health factor bars) ──
    var liveStr = (sx.live || 0) + ' live · ' + (sx.valid_tokens || 0) + ' token' + ((sx.valid_tokens || 0) === 1 ? '' : 's');
    var errTone = (api.error_rate || 0) >= 20 ? 'bad' : ((api.error_rate || 0) >= 5 ? 'warn' : '');
    var vitals =
      '<div class="dl-dsec" id="ud-vitals" data-sec="ud-vitals"><div class="dl-dsec-h"><span class="pip"></span>Vitals</div>'
      + '<div class="ud-kband">'
        + udKpi(fmtInt(api.requests || 0), 'requests', { drill: hasApi ? 'requests' : '', title: 'API requests across all keys' })
        + udKpi(fmtInt(inf.count || 0), 'inferences', { drill: (inf.count ? 'inferences' : ''), title: 'Website inferences (predictions)' })
        + udKpi(fmtInt(d.key_count || 0), 'API keys', { drill: hasKeys ? 'keys' : '', title: 'Minted API keys' })
        + udKpi((api.error_rate || 0) + '%', 'error rate', { drill: hasApi ? 'errors' : '', tone: errTone, title: 'Share of API calls returning 4xx/5xx' })
        + udKpi(fmtInt(sx.live || 0), 'live now', { title: liveStr, tone: (sx.live ? 'good' : '') })
      + '</div>'
      + '<div class="ud-health-wrap"><div class="ud-health-h">health factors <b>' + (health.score != null ? health.score + '/100' : '—') + '</b></div>'
        + udFactor('recency', (health.factors || {}).recency)
        + udFactor('reliability', (health.factors || {}).reliability)
        + udFactor('volume', (health.factors || {}).volume)
      + '</div></div>';

    // ── 3 · Timeline (30d sparkline + unified event feed) ──
    var timeline =
      '<div class="dl-dsec" id="ud-timeline" data-sec="ud-timeline"><div class="dl-dsec-h"><span class="pip"></span>Activity timeline</div>'
      + (hasApi ? '<div class="kd-sec" style="padding:0 0 6px"><h5>30-day request activity</h5>' + kdSpark(api.series) + '</div>' : '')
      + (evs.length
          ? '<div class="ud-events">' + evs.map(function (e) {
              var m = UD_EV[e.kind] || UD_EV.identity;
              var ref = e.ref ? ' data-ref-kind="' + esc(e.ref.kind) + '" data-ref-id="' + esc(e.ref.id) + '" role="button" tabindex="0"' : '';
              return '<div class="ud-ev ' + m.cls + (e.ref ? ' clickable' : '') + '"' + ref + '>'
                + '<span class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="' + m.ic + '"/></svg></span>'
                + '<span class="tt">' + esc(e.title) + '</span>'
                + '<span class="tm">' + esc(relTime(e.ts)) + '</span>'
                + (e.ref ? '<span class="chev">›</span>' : '') + '</div>';
            }).join('') + '</div>'
          : '<div class="kd-empty-mini">no recorded events yet</div>')
      + '</div>';

    // ── 4 · Reliability (status breakdown that explains the error rate) ──
    var reliability = '';
    if (hasApi) {
      var sb = rel.status_breakdown || [];
      var sbTotal = sb.reduce(function (a, b) { return a + (b.n || 0); }, 0) || 1;
      reliability =
        '<div class="dl-dsec" id="ud-reliability" data-sec="ud-reliability"><div class="dl-dsec-h"><span class="pip"></span>Reliability</div>'
        + '<div class="ud-statbar">' + sb.map(function (s) {
            return '<span class="seg ' + s.cls + '" style="width:' + (100 * (s.n || 0) / sbTotal).toFixed(2) + '%" title="' + s.code + ' × ' + s.n + '"></span>';
          }).join('') + '</div>'
        + '<div class="ud-statleg">' + sb.map(function (s) {
            return '<span class="ud-sl ' + s.cls + '"><i></i>' + s.code + ' <b>' + fmtInt(s.n) + '</b></span>';
          }).join('') + '</div>'
        + ((rel.top_errors || []).length
            ? '<h5 class="ud-subh">top error endpoints</h5><div class="ud-errs">' + rel.top_errors.map(function (e) {
                return '<div class="ud-errrow"><span class="code ' + (e.code >= 500 ? 'err' : 'warn') + '">' + e.code + '</span>'
                  + '<span class="p mono">' + esc(e.label || '—') + '</span><span class="n">' + fmtInt(e.n) + '</span></div>';
              }).join('') + '</div>'
            : '')
        + '</div>';
    }

    // ── 5 · API keys sub-panel ──
    var keysSec = '';
    if (hasKeys) {
      keysSec =
        '<div class="dl-dsec" id="ud-keys" data-sec="ud-keys"><div class="dl-dsec-h"><span class="pip"></span>API keys <span class="dl-count">' + fmtInt(keys.length) + '</span></div>'
        + '<div class="ud-keys">' + keys.map(function (k) {
            var st = (k.status || 'active').toLowerCase();
            var stc = st === 'active' ? 'ok' : (st === 'disabled' || st === 'revoked' ? 'mute' : 'warn');
            var et = (k.error_rate || 0) >= 20 ? 'bad' : ((k.error_rate || 0) >= 5 ? 'warn' : '');
            return '<div class="ud-keyrow" data-key="' + esc(k.public_id) + '" role="button" tabindex="0">'
              + '<span class="st ' + stc + '" title="' + esc(st) + '"></span>'
              + '<span class="nm">' + esc(k.name || k.public_id) + '<small>' + fmtInt(k.scope_count) + ' scope' + (k.scope_count === 1 ? '' : 's') + (k.group_id ? ' · grouped' : '') + '</small></span>'
              + '<span class="rq mono">' + fmtInt(k.requests) + '</span>'
              + '<span class="er mono ' + et + '">' + (k.error_rate || 0) + '%</span>'
              + '<span class="lu">' + esc(k.last_used ? relTime(k.last_used) : 'never') + '</span>'
              + '<span class="chev">›</span></div>';
          }).join('') + '</div></div>';
    }

    // ── 6 · Inferences (disease mix + recent, cross-drillable) ──
    var dmTotal = (inf.disease_mix || []).reduce(function (a, b) { return a + (b.value || 0); }, 0) || 1;
    var inferences =
      '<div class="dl-dsec" id="ud-inferences" data-sec="ud-inferences"><div class="dl-dsec-h"><span class="pip"></span>Inferences <span class="dl-count">' + fmtInt(inf.count || 0) + '</span></div>'
      + (inf.count
          ? ('<div class="ud-dmix">' + (inf.disease_mix || []).map(function (m, i) {
                return '<span class="seg" style="width:' + (100 * (m.value || 0) / dmTotal).toFixed(2) + '%;background:' + UD_DISEASE_COL[i % UD_DISEASE_COL.length] + '" title="' + esc(m.label) + ': ' + m.value + '"></span>';
              }).join('') + '</div>'
              + '<div class="ud-dlegs">' + (inf.disease_mix || []).slice(0, 8).map(function (m, i) {
                return '<span class="ud-dleg"><i style="background:' + UD_DISEASE_COL[i % UD_DISEASE_COL.length] + '"></i>' + esc((m.label || '').replace(/_/g, ' ')) + ' <b>' + m.value + '</b></span>';
              }).join('') + '</div>'
              + '<h5 class="ud-subh">recent inferences</h5>'
              + '<div class="ud-scans">' + (inf.recent || []).map(function (r) {
                return '<div class="ud-scan" data-scan="' + esc(r.scan_uid) + '" role="button" tabindex="0">'
                  + '<span class="dx">' + esc((r.diagnosis || '—').replace(/_/g, ' ')) + '</span>'
                  + '<span class="cf">' + (r.confidence != null ? Math.round(r.confidence * 100) + '%' : '—') + '</span>'
                  + '<span class="sv">' + esc(r.severity || '') + '</span>'
                  + '<span class="tm">' + esc(relTime(r.ts)) + '</span><span class="chev">›</span></div>';
              }).join('') + '</div>')
          : '<div class="ud-eng-empty"><div class="glyph">' + ES_ANIM.seed + '</div><div class="tag">no fields read yet</div>'
            + '<div class="sub">this account has not submitted a single leaf for diagnosis. The disease mix and scan history will grow here with the first upload.</div></div>')
      + '</div>';

    // ── 7 · Engagement (persona-aware empty state) ──
    var engagement =
      '<div class="dl-dsec" id="ud-engagement" data-sec="ud-engagement"><div class="dl-dsec-h"><span class="pip"></span>On-page engagement</div>'
      + (engEmpty
          ? '<div class="ud-eng-empty"><div class="glyph">' + ES_ANIM.radar + '</div>'
            + '<div class="tag">' + (apiFirst ? 'integration account — telemetry isn’t expected' : 'no on-page telemetry yet') + '</div>'
            + '<div class="sub">' + (apiFirst
                ? 'This is an <b>API-first</b> account: it drives traffic through API keys, not the browser, so on-page metrics (time-on-page, scroll depth, click heatmap) don’t apply here. Its activity lives in <b>Vitals</b>, <b>Reliability</b> and <b>API keys</b> above.'
                : 'No attributed browsing sessions have settled against this account yet. Time-on-page, scroll depth, rage-clicks and the activity heatmap will appear here once it browses the site signed in.') + '</div></div>'
          : ('<div class="kd-kpis">'
              + '<div class="kd-kpi"><span class="v">' + Math.round((eng.total_active_ms || 0) / 60000) + 'm</span><span class="l">active time</span></div>'
              + '<div class="kd-kpi"><span class="v">' + fmtInt(eng.page_views || 0) + '</span><span class="l">page views</span></div>'
              + '<div class="kd-kpi"><span class="v">' + fmtInt(eng.sessions || 0) + '</span><span class="l">sessions</span></div>'
              + '<div class="kd-kpi"><span class="v">' + fmtInt(eng.clicks || 0) + '</span><span class="l">clicks</span></div>'
              + '<div class="kd-kpi"><span class="v">' + (eng.avg_scroll != null ? Math.round(eng.avg_scroll) + '%' : '—') + '</span><span class="l">avg scroll</span></div>'
              + '<div class="kd-kpi"><span class="v">' + fmtInt((eng.rage_clicks || 0)) + '</span><span class="l">rage clicks</span></div></div>'
              + udEngPages(eng.pages)))
      + '</div>';

    // ── 8 · Sessions & security (honest accounting) ──
    var sessions =
      '<div class="dl-dsec" id="ud-sessions" data-sec="ud-sessions"><div class="dl-dsec-h"><span class="pip"></span>Sessions &amp; security</div>'
      + '<div class="ud-kband">'
        + udKpi(fmtInt(sx.live || 0), 'live tabs', { tone: (sx.live ? 'good' : ''), title: 'Browser tabs heartbeating within 45s' })
        + udKpi(fmtInt(sx.idle_open || 0), 'idle open', { title: 'Open browser sessions gone quiet' })
        + udKpi(fmtInt(sx.valid_tokens || 0), 'valid tokens', { title: 'Non-expired auth tokens — what the old UI called “active”. Inflated by every login/restart.' })
      + '</div>'
      + ((sx.devices || []).length
          ? '<h5 class="ud-subh">recent clients</h5><div class="ud-devs">' + sx.devices.map(function (v) {
              return '<div class="ud-dev"><span class="dot ' + (v.open ? 'on' : '') + '"></span>'
                + '<span class="nm">' + esc(v.browser || '—') + (v.os && v.os !== '—' ? '<small>' + esc(v.os) + (v.loc ? ' · ' + esc(v.loc) : '') + '</small>' : (v.loc ? '<small>' + esc(v.loc) + '</small>' : '')) + '</span>'
                + '<span class="tm">' + esc(v.last ? relTime(v.last) : '—') + '</span></div>';
            }).join('') + '</div>'
          : '<div class="kd-empty-mini">no client sessions on record</div>')
      + '</div>';

    // ── 9 · Footprint (districts) ──
    var maxD = (loc.districts || []).reduce(function (a, b) { return Math.max(a, b.count || 0); }, 0) || 1;
    var footprint = hasDistricts
      ? '<div class="dl-dsec" id="ud-footprint" data-sec="ud-footprint"><div class="dl-dsec-h"><span class="pip"></span>Geographic footprint</div>'
        + '<div class="ud-districts">' + loc.districts.map(function (g) {
            return '<div class="ud-dist"><span class="nm">' + esc(g.district || '—') + '<small>' + esc(g.state || '') + '</small></span>'
              + '<span class="bar"><i style="width:' + Math.round(100 * (g.count || 0) / maxD) + '%"></i></span>'
              + '<span class="ct">' + fmtInt(g.count) + '</span></div>';
          }).join('') + '</div></div>'
      : '';

    // ── 10 · Account control (mutations) ──
    var account =
      '<div class="dl-dsec" id="ud-account" data-sec="ud-account"><div class="dl-dsec-h"><span class="pip"></span>Account control</div>'
      + '<div class="ud-acct-meta">'
        + '<div class="sd-prov-row"><span>role</span><b>' + esc(d.role || 'collector') + (d.is_admin ? ' · administrator' : '') + '</b></div>'
        + '<div class="sd-prov-row"><span>user id</span><b class="mono">#' + esc(d.id) + '</b></div>'
        + '<div class="sd-prov-row"><span>sessions</span><b>' + fmtInt(sx.live || 0) + ' live · ' + fmtInt(sx.valid_tokens || 0) + ' valid token' + ((sx.valid_tokens || 0) === 1 ? '' : 's') + '</b></div>'
      + '</div>'
      + '<div class="ud-actions">'
        + '<button class="ud-act" data-act="role" data-role="' + (d.is_admin ? 'collector' : 'admin') + '">'
          + (d.is_admin ? 'Revoke admin' : 'Promote to admin') + '</button>'
        + '<button class="ud-act danger" data-act="logout">Force log out · ' + fmtInt(sx.valid_tokens || 0) + ' token' + ((sx.valid_tokens || 0) === 1 ? '' : 's') + '</button>'
      + '</div>'
      + '<div class="ud-act-msg" id="ud-act-msg" hidden></div>'
      + '</div>';

    el.innerHTML = '<div class="dl-spine">' + spine + '</div>'
      + '<div class="dl-pane">'
        + '<div class="dl-head"><div class="ttl"><div class="sub">user · ' + esc(persona.label || d.role || 'collector') + '</div>'
          + '<div class="main">' + esc(d.display_name || d.username || ('user ' + d.id)) + '</div></div>'
          + '<button class="x" id="dl-ds-x" aria-label="Close">×</button></div>'
        + '<div class="dl-body" id="ud-body">' + identity + vitals + timeline + reliability + keysSec
            + inferences + engagement + sessions + footprint + account + '</div>'
      + '</div>';
    bindClose('dl-ds-x', closeDossier);
    wireDossierSpine(el, 'ud-body');
    udWireDossier(el, d);
  }

  // per-page engagement rollup (web users) — compact bars
  function udEngPages(pages) {
    pages = pages || [];
    if (!pages.length) return '';
    var max = pages.reduce(function (a, b) { return Math.max(a, b.active_ms || 0); }, 0) || 1;
    return '<h5 class="ud-subh">top pages</h5><div class="ud-pages">' + pages.slice(0, 8).map(function (p) {
      return '<div class="ud-page"><span class="rt mono">' + esc(p.route || '/') + '</span>'
        + '<span class="bar"><i style="width:' + Math.round(100 * (p.active_ms || 0) / max) + '%"></i></span>'
        + '<span class="vt">' + Math.round((p.active_ms || 0) / 60000) + 'm</span></div>';
    }).join('') + '</div>';
  }

  // wire all the dossier's interactive surfaces (cross-links + copy + mutations).
  // KPI-band drill targets are resolved in H3; here we attach the handlers.
  function udWireDossier(el, d) {
    // recent inference → prediction dossier
    var scans = el.querySelector('.ud-scans');
    if (scans) scans.addEventListener('click', function (e) {
      var row = e.target.closest ? e.target.closest('.ud-scan[data-scan]') : null;
      if (row) openDetail('prediction', row.getAttribute('data-scan'));
    });
    // timeline event with a ref → its drill (inference → prediction dossier)
    var evbox = el.querySelector('.ud-events');
    if (evbox) evbox.addEventListener('click', function (e) {
      var row = e.target.closest ? e.target.closest('.ud-ev[data-ref-id]') : null;
      if (row) openDetail(row.getAttribute('data-ref-kind'), row.getAttribute('data-ref-id'));
    });
    // key row → key drawer
    var krows = el.querySelector('.ud-keys');
    if (krows) krows.addEventListener('click', function (e) {
      var row = e.target.closest ? e.target.closest('.ud-keyrow[data-key]') : null;
      if (row) openDetail('key', row.getAttribute('data-key'));
    });
    // KPI band tiles → list drills (scoped to this user)
    var vit = el.querySelector('#ud-vitals .ud-kband');
    if (vit) vit.addEventListener('click', function (e) {
      var t = e.target.closest ? e.target.closest('.ud-kt[data-drill]') : null;
      if (t) udKpiDrill(d, t.getAttribute('data-drill'));
    });
    // copy email
    el.addEventListener('click', function (e) {
      var c = e.target.closest ? e.target.closest('.ud-copy[data-copy]') : null;
      if (!c) return;
      udCopy(c.getAttribute('data-copy'), c);
    });
    // account mutations
    var acts = el.querySelector('.ud-actions');
    if (acts) acts.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('.ud-act[data-act]') : null; if (!b) return;
      udMutate(d.id, b.getAttribute('data-act'), b.getAttribute('data-role'));
    });
  }

  function udCopy(val, node) {
    function done() {
      if (!node) return;
      node.classList.add('copied');
      setTimeout(function () { node.classList.remove('copied'); }, 1100);
    }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(val).then(done, done);
      } else {
        var ta = document.createElement('textarea'); ta.value = val; document.body.appendChild(ta);
        ta.select(); try { document.execCommand('copy'); } catch (e) {} document.body.removeChild(ta); done();
      }
    } catch (e) { /* clipboard blocked — silent */ }
  }

  // KPI-band → user-scoped drill list (wired fully in H3; safe no-op stubs here)
  function udKpiDrill(d, kind) {
    if (kind === 'requests' || kind === 'errors') {
      var f = { user_id: d.id };
      if (kind === 'errors') f.status = 'error';
      openDrillList('requests', f, (d.display_name || d.username || ('user ' + d.id)) + ' · ' + (kind === 'errors' ? 'errors' : 'requests'));
    } else if (kind === 'inferences') {
      openDrillList('predictions', { user_id: d.id }, (d.display_name || d.username || ('user ' + d.id)) + ' · inferences');
    } else if (kind === 'keys') {
      openDrillList('keys', { user_id: d.id }, (d.display_name || d.username || ('user ' + d.id)) + ' · keys');
    }
  }
  function udMutate(uid, act, role) {
    var msg = document.getElementById('ud-act-msg');
    var isDanger = (act !== 'role') || (role !== 'admin');   // revoke + logout are destructive
    var title = act === 'role' ? (role === 'admin' ? 'Promote to administrator' : 'Revoke admin access') : 'Force log out';
    var label = act === 'role'
      ? (role === 'admin' ? 'Grant this account full administrator privileges across the org?' : 'Remove this account’s administrator privileges?')
      : 'Sign this user out of every active session immediately?';
    // returns a promise; the modal shows a busy state and closes on resolve, or
    // surfaces the thrown error inline. Refreshes the dossier on success.
    function doFetch() {
      var path = act === 'role' ? '/api/account/admin/users/' + uid + '/role' : '/api/account/admin/users/' + uid + '/logout-all';
      var body = act === 'role' ? JSON.stringify({ role: role }) : '{}';
      return fetch(path, { method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json', 'X-Console-Csrf': (document.querySelector('meta[name=csrf-token]') || {}).content || '' },
        body: body })
        .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) {
          if (!r.ok) throw new Error((j.error && j.error.message) || 'Action failed.'); return j; }); })
        .then(function (j) {
          if (msg) { msg.hidden = false; msg.className = 'ud-act-msg ok';
            msg.textContent = act === 'role' ? 'Role updated.'
              : (j.sessions_revoked != null ? j.sessions_revoked + ' session(s) revoked.' : 'Sessions revoked.'); }
          setTimeout(function () { openDetail('user', uid); }, 1000);   // refresh dossier
        });
    }
    if (window.APIN && APIN.modal && APIN.modal.confirm) {
      APIN.modal.confirm({ title: title, message: label, danger: isDanger,
        confirmLabel: act === 'role' ? (role === 'admin' ? 'Promote' : 'Revoke') : 'Log out all',
        onConfirm: function () { return doFetch(); } });
    } else if (window.confirm(label)) {
      doFetch().catch(function (e) { if (msg) { msg.hidden = false; msg.className = 'ud-act-msg err'; msg.textContent = e.message; } });
    }
  }
  function _basicDrawer(kind, d, title) {
    var el = document.getElementById('dl-dw');
    el.innerHTML = '<div class="dl-head"><div class="ttl"><div class="sub">' + esc(kind) + '</div><div class="main">' + esc(title) + '</div></div><button class="x" id="dl-dw-x">×</button></div>'
      + '<div class="dl-body">' + dlEmpty('flat', 'detailed view loads here', 'phase build in progress') + '</div>';
    bindClose('dl-dw-x', closeDrillDrawer);
  }
  function _basicDossier(kind, d, title) {
    var el = document.getElementById('dl-dossier');
    el.innerHTML = '<div class="dl-spine"></div><div class="dl-pane"><div class="dl-head"><div class="ttl"><div class="sub">' + esc(kind) + '</div><div class="main">' + esc(title) + '</div></div><button class="x" id="dl-ds-x">×</button></div>'
      + '<div class="dl-body">' + dlEmpty('flat', 'detailed dossier loads here', 'phase build in progress') + '</div></div>';
    bindClose('dl-ds-x', closeDossier);
  }

  (function initDrillGlobal() {
    // P3 · point the reused request drawer at the admin's gated detail endpoint.
    // Same module, identical content — only the data source + dark theme differ.
    if (window.APIN && APIN.requestDrawer && APIN.requestDrawer.setDetailUrl) {
      APIN.requestDrawer.setDetailUrl(function (rid) {
        return '/api/account/admin/detail/request/' + encodeURIComponent(rid);
      });
    }
    var pairs = [['dl-list-back', closeDrillList], ['dl-dw-back', closeDrillDrawer], ['dl-dossier-back', closeDossier]];
    pairs.forEach(function (p) { var el = document.getElementById(p[0]); if (el) el.addEventListener('click', p[1]); });
    // Capture phase (3rd arg true) so this runs BEFORE the reused request
    // drawer's own bubble-phase Esc handler. When #reqd is open it clears its
    // own .show class synchronously; if we ran after that we'd no longer see it
    // open and would wrongly also collapse the list behind it. Running first
    // lets us bail while #reqd.show is still present — Esc peels one layer.
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape') return;
      var rq = document.getElementById('reqd');
      if (rq && rq.classList.contains('show')) return;  // request drawer owns this Esc
      // Close the MOST-RECENTLY-OPENED dl panel (the one visually on top),
      // not a fixed priority — nested cross-drills stack these at equal z.
      var cands = [['dl-dossier', closeDossier], ['dl-dw', closeDrillDrawer], ['dl-list', closeDrillList]]
        .map(function (c) { var el = document.getElementById(c[0]); return el && !el.hidden ? { fn: c[1], oat: +(el.dataset.oat || 0) } : null; })
        .filter(Boolean);
      if (!cands.length) return;
      cands.sort(function (a, b) { return b.oat - a.oat; });
      cands[0].fn();
    }, true);
  })();

  // ── Database mirror (Phase D · browse) ──────────────────────────────────
  var dbState = { table: null, search: '', sort: null, order: 'asc', offset: 0, limit: 50, cols: [], rows: [], loaded: false };

  function dbIsNum(ty) {
    ty = (ty || '').toUpperCase();
    return ty.indexOf('INT') >= 0 || ty.indexOf('REAL') >= 0 || ty.indexOf('NUM') >= 0
      || ty.indexOf('FLOA') >= 0 || ty.indexOf('DOUB') >= 0;
  }

  function loadDbTables() {
    var list = document.getElementById('db-tablelist');
    adminFetch('/api/account/admin/db/tables').then(function (d) {
      if (!list) return;
      if (!d || !d.tables) { list.innerHTML = '<div class="db-empty sm">could not load schema</div>'; return; }
      var meta = document.getElementById('db-rail-meta');
      if (meta) meta.textContent = d.tables.length + ' · ' + fmtInt(d.total_rows) + ' rows';
      list.innerHTML = d.tables.map(function (t) {
        return '<div class="db-titem" data-tb="' + esc(t.name) + '" role="button" tabindex="0">'
          + '<span class="nm">' + esc(t.name) + '</span>'
          + (t.has_secrets ? '<span class="lk" title="contains masked columns">🔒</span>' : '')
          + '<span class="ct">' + fmtInt(t.rows) + '</span></div>';
      }).join('');
      dbState.loaded = true;
    });
  }
  function dbHighlightRail() {
    var list = document.getElementById('db-tablelist'); if (!list) return;
    Array.prototype.forEach.call(list.querySelectorAll('.db-titem'), function (it) {
      it.classList.toggle('on', it.getAttribute('data-tb') === dbState.table);
    });
  }
  function selectTable(name) {
    dbState.table = name; dbState.search = ''; dbState.sort = null; dbState.order = 'asc'; dbState.offset = 0;
    var s = document.getElementById('db-search'); if (s) { s.value = ''; s.disabled = false; }
    dbHighlightRail(); loadDbRows();
  }
  function loadDbRows() {
    if (!dbState.table) return;
    var qs = 'name=' + encodeURIComponent(dbState.table) + '&limit=' + dbState.limit + '&offset=' + dbState.offset
      + (dbState.search ? '&search=' + encodeURIComponent(dbState.search) : '')
      + (dbState.sort ? '&sort=' + encodeURIComponent(dbState.sort) + '&order=' + dbState.order : '');
    var wrap = document.getElementById('db-grid-wrap');
    if (wrap) wrap.innerHTML = '<div class="db-empty">loading…</div>';
    adminFetch('/api/account/admin/db/table?' + qs).then(function (d) {
      if (!d) { if (wrap) wrap.innerHTML = '<div class="db-empty">could not load table</div>'; return; }
      dbState.cols = d.columns || []; dbState.rows = d.rows || [];
      dbState.sort = d.sort; dbState.order = d.order;
      var nm = document.getElementById('db-tname'); if (nm) nm.textContent = d.name;
      var rw = document.getElementById('db-trows');
      if (rw) rw.textContent = fmtInt(d.pagination.total) + ' rows · ' + d.columns.length + ' columns' + (d.search ? ' · filtered' : '');
      var sc = document.getElementById('db-schema');
      if (sc) sc.innerHTML = d.columns.map(function (c) {
        return '<span class="db-col' + (c.pk ? ' pk' : '') + (c.secret ? ' secret' : '') + '">'
          + '<span class="cn">' + esc(c.name) + '</span><span class="cty">' + esc(c.type) + '</span></span>';
      }).join('');
      renderDbGrid(d); renderDbPager(d);
    });
  }
  function renderDbGrid(d) {
    var wrap = document.getElementById('db-grid-wrap'); if (!wrap) return;
    if (!d.rows.length) {
      wrap.innerHTML = '<div class="db-empty">' + (d.search ? 'no rows match “' + esc(d.search) + '”' : 'this table is empty') + '</div>';
      return;
    }
    var head = '<tr>' + d.columns.map(function (c) {
      var arrow = (d.sort === c.name) ? '<span class="ar">' + (d.order === 'desc' ? '▼' : '▲') + '</span>' : '';
      return '<th data-col="' + esc(c.name) + '">' + esc(c.name) + arrow + '</th>';
    }).join('') + '</tr>';
    var body = d.rows.map(function (row, ri) {
      return '<tr data-ri="' + ri + '">' + row.map(function (val, ci) {
        var c = d.columns[ci];
        var cls = (val === null) ? 'null' : (c.secret ? 'secret' : (dbIsNum(c.type) ? 'num' : ''));
        var disp = (val === null) ? 'NULL' : esc(val);
        return '<td class="' + cls + '" title="' + (val === null ? '' : esc(val)) + '">' + disp + '</td>';
      }).join('') + '</tr>';
    }).join('');
    wrap.innerHTML = '<table class="db-grid"><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
  }
  function renderDbPager(d) {
    var pg = document.getElementById('db-pager'); if (!pg) return;
    var p = d.pagination, from = p.total ? p.offset + 1 : 0, to = p.offset + p.returned;
    pg.innerHTML = '<span class="info">' + fmtInt(from) + '–' + fmtInt(to) + ' of ' + fmtInt(p.total) + '</span>'
      + '<span class="nav">'
      + '<button class="adm-btn" id="db-prev"' + (p.offset <= 0 ? ' disabled' : '') + '>‹ prev</button>'
      + '<button class="adm-btn" id="db-next"' + (to >= p.total ? ' disabled' : '') + '>next ›</button></span>';
    var pv = document.getElementById('db-prev');
    if (pv) pv.addEventListener('click', function () { dbState.offset = Math.max(0, dbState.offset - dbState.limit); loadDbRows(); });
    var nx = document.getElementById('db-next');
    if (nx) nx.addEventListener('click', function () { dbState.offset += dbState.limit; loadDbRows(); });
  }
  function openDbRow(ri) {
    var dw = document.getElementById('db-drawer'), bk = document.getElementById('db-drawer-back');
    if (!dw) return;
    var row = dbState.rows[ri]; if (!row) return;
    var kv = dbState.cols.map(function (c, ci) {
      var val = row[ci];
      var vcls = (val === null) ? 'null' : (c.secret ? 'secret' : '');
      var disp = (val === null) ? 'NULL' : esc(val);
      return '<div class="row"><div class="k">' + esc(c.name)
        + (c.pk ? ' <span class="ty">★pk</span>' : '') + ' <span class="ty">' + esc(c.type) + '</span>'
        + (c.secret ? ' <span class="ty">🔒</span>' : '') + '</div>'
        + '<div class="v ' + vcls + '">' + disp + '</div></div>';
    }).join('');
    dw.innerHTML = '<div class="dh"><span class="nm"><b>' + esc(dbState.table) + '</b><span>row inspector</span></span>'
      + '<button class="x" id="db-x" aria-label="Close">×</button></div>'
      + '<div class="dbody"><div class="db-kv">' + kv + '</div></div>';
    dw.classList.add('open'); dw.setAttribute('aria-hidden', 'false');
    if (bk) bk.classList.add('open');
    document.getElementById('db-x').addEventListener('click', closeDbDrawer);
  }
  function closeDbDrawer() {
    var dw = document.getElementById('db-drawer'), bk = document.getElementById('db-drawer-back');
    if (dw) { dw.classList.remove('open'); dw.setAttribute('aria-hidden', 'true'); }
    if (bk) bk.classList.remove('open');
  }
  (function initDb() {
    var list = document.getElementById('db-tablelist');
    if (list) list.addEventListener('click', function (e) {
      var it = e.target.closest ? e.target.closest('.db-titem[data-tb]') : null;
      if (it) selectTable(it.getAttribute('data-tb'));
    });
    var wrap = document.getElementById('db-grid-wrap');
    if (wrap) wrap.addEventListener('click', function (e) {
      var th = e.target.closest ? e.target.closest('th[data-col]') : null;
      if (th) {
        var col = th.getAttribute('data-col');
        if (dbState.sort === col) dbState.order = (dbState.order === 'asc') ? 'desc' : 'asc';
        else { dbState.sort = col; dbState.order = 'asc'; }
        dbState.offset = 0; loadDbRows(); return;
      }
      var tr = e.target.closest ? e.target.closest('tr[data-ri]') : null;
      if (tr) openDbRow(parseInt(tr.getAttribute('data-ri'), 10));
    });
    var search = document.getElementById('db-search'), st = null;
    if (search) search.addEventListener('input', function () {
      clearTimeout(st); st = setTimeout(function () { dbState.search = search.value.trim(); dbState.offset = 0; loadDbRows(); }, 280);
    });
    var bk = document.getElementById('db-drawer-back'); if (bk) bk.addEventListener('click', closeDbDrawer);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { var dw = document.getElementById('db-drawer'); if (dw && dw.classList.contains('open')) closeDbDrawer(); }
    });
  })();

  // ── boot sequence ───────────────────────────────────────────────────────
  // The full handshake plays ONLY when we arrived via the login transition
  // (the one-time `apin_admin_arrival` flag). A manual refresh or a directly
  // typed URL gets a quick fade instead of re-running the handshake every
  // time — matching the design ("a directly-typed URL just fades in"). The
  // flag is consumed on read. Reduced-motion always quick-fades.
  // Identity is fetched in parallel and patched in when it resolves.
  var cameFromLogin = false;
  try {
    cameFromLogin = sessionStorage.getItem('apin_admin_arrival') === '1';
    sessionStorage.removeItem('apin_admin_arrival');
  } catch (e) {}
  if (!REDUCED && cameFromLogin) {
    playArrival();
  } else {
    clearTimeout(failsafe);
    dismissBoot();
  }
  loadIdentity().then(applyIdentity);
  loadPulse();
  loadSignups(plWindow);
  loadFeed();

  // ── ADM-T · bridge for the separately-loaded Traffic module ──────────────
  // console_admin_traffic.js renders the Traffic section but lives outside this
  // IIFE, so expose the small set of shared helpers it needs (slot-machine
  // numbers, escaping, time, the gated fetch, and the existing drill drawers)
  // rather than duplicating them.
  window.ADM = window.ADM || {};
  ADM.animateNum = animateNum; ADM.esc = esc; ADM.fmtInt = fmtInt; ADM.relTime = relTime;
  ADM.adminFetch = adminFetch; ADM.openDetail = openDetail; ADM.openDrillList = openDrillList;
  ADM.fmtBytes = function (n) {
    n = Number(n || 0); if (n < 1024) return n + ' B';
    var u = ['KB', 'MB', 'GB', 'TB'], i = -1;
    do { n /= 1024; i++; } while (n >= 1024 && i < u.length - 1);
    return (n < 10 ? n.toFixed(1) : Math.round(n)) + ' ' + u[i];
  };
})();
