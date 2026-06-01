/* 9.N.T31 · Per-key "Analytics" tab controller (inference observatory).
 *
 * Mounted lazily by console_key_detail.js ONLY when the Analytics tab is first
 * opened — so globe.gl, the boundary GeoJSON and the 3D field never touch the
 * initial site load. APIN.analytics.mount(rootEl, {publicId}) builds the layout,
 * fetches /api/account/analytics/*, renders every widget, wires the live SSE
 * scan feed, and lazy-loads the two WebGL modules (geo globe + inference field).
 *
 * Widgets here (SVG, dependency-free): quota leaf-fill, request stream,
 * severity spectrum, disease bloom, response payloads, recent scans. The two
 * 3D widgets live in console_analytics_geo.js / console_analytics_field.js and
 * register themselves on APIN.analyticsGeo / APIN.analyticsField.
 */
(function () {
  'use strict';
  const A = {};
  let _root = null, _pid = null, _range = '7d', _env = '';
  let _booted = false, _es = null, _liveTimer = null, _geoTier = 'state', _geoSource = 'requests';
  const _focus = { region: null, disease: null, crop: null };
  const _data = {};            // last payload per widget
  const _geoLib = { loaded: false, loading: null };
  const _fieldLib = { loaded: false, loading: null };

  // ── helpers ───────────────────────────────────────────────────────────
  const $ = (sel) => _root ? _root.querySelector(sel) : null;
  const $$ = (sel) => _root ? Array.from(_root.querySelectorAll(sel)) : [];
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function fmtNum(n) {
    if (n == null || isNaN(n)) return '·';
    n = Number(n);
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return String(Math.round(n));
  }
  function titleCase(s) {
    return String(s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
  function sevTone(sev) {
    return sev === 'severe' ? 'var(--c-danger,#b3402f)'
      : sev === 'moderate' ? 'var(--c-amber,#c98a2b)'
        : 'var(--c-accent,#52b788)';
  }
  async function api(path) {
    const sep = path.indexOf('?') >= 0 ? '&' : '?';
    const qs = `range=${encodeURIComponent(_range)}` +
      (_pid ? `&key_id=${encodeURIComponent(_pid)}` : '') +
      (_env ? `&env=${encodeURIComponent(_env)}` : '');
    const r = await fetch('/api/account/analytics/' + path + sep + qs,
      { credentials: 'include' });
    if (r.status === 401) { location.href = '/dashboard'; throw new Error('401'); }
    const b = await r.json().catch(() => ({}));
    return (b && b.ok && b.data !== undefined) ? b.data : (b && b.data) || b;
  }
  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.async = true; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }
  function setHostState(host, state, label) {
    if (!host) return;
    if (state === 'loading') host.innerHTML = '<div class="an-ph">loading&hellip;</div>';
    else if (state === 'empty') host.innerHTML =
      '<div class="an-ph an-ph-empty"><span class="an-leaf"></span>' +
      (esc(label || 'no scans in this window')) + '</div>';
  }

  // ── layout ────────────────────────────────────────────────────────────
  function layoutHTML() {
    return `
    <div class="an-wrap">
      <div class="an-controls card">
        <div class="an-range" role="tablist" aria-label="Time range">
          ${['24h', '7d', '30d'].map(r =>
      `<button data-an-range="${r}"${r === _range ? ' aria-pressed="true"' : ''}>${r}</button>`).join('')}
        </div>
        <div class="an-chips" id="an-chips"></div>
        <div class="an-spacer"></div>
        <span class="an-live" id="an-live"><span class="an-live-dot"></span>live</span>
      </div>

      <!-- Quota -->
      <div class="card an-quota" id="an-quota"><div class="an-ph">loading&hellip;</div></div>

      <!-- Geo + legend -->
      <div class="an-row an-row-geo">
        <div class="card an-card-geo">
          <div class="an-card-head"><h2>Geographic origin</h2>
            <div class="an-geo-ctl">
              <div class="an-seg" id="an-geo-source">
                <button data-an-source="requests" aria-pressed="true">requests</button>
                <button data-an-source="scans">scans</button>
              </div>
              <div class="an-seg" id="an-geo-layer">
                ${['origins', 'crop', 'disease', 'density'].map((l, i) =>
        `<button data-an-layer="${l}"${i === 0 ? ' aria-pressed="true"' : ''}>${l}</button>`).join('')}
              </div>
              <button class="an-expand" data-an-expand="geo" title="Expand" aria-label="Expand to full console"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button>
            </div>
          </div>
          <div class="an-card-body" id="an-geo-host"><div class="an-ph">loading map&hellip;</div></div>
          <div class="an-geo-credit">boundaries: Natural Earth (PD) · geoBoundaries (CC&nbsp;BY)</div>
        </div>
        <div class="card an-card-legend">
          <div class="an-card-head"><h2>Top regions</h2></div>
          <div class="an-card-body" id="an-legend-host"><div class="an-ph">·</div></div>
        </div>
      </div>

      <!-- Inference field -->
      <div class="card an-card-field">
        <div class="an-card-head"><h2 id="an-field-title">Inference field</h2>
          <div class="an-geo-ctl">
            <div class="an-seg" id="an-field-mode">
              <button data-an-fmode="crop" aria-pressed="true">crops</button>
              <button data-an-fmode="mosaic">mosaic</button>
            </div>
            <button class="an-expand" data-an-expand="field" title="Expand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button>
          </div>
        </div>
        <div class="an-card-body an-field-body">
          <div id="an-field-host"><div class="an-ph">loading field&hellip;</div></div>
          <div class="an-field-insight" id="an-field-insight"></div>
        </div>
      </div>

      <!-- Client distribution + Confidence ladder (replaces Request stream) -->
      <div class="an-row an-row-2">
        <div class="card an-card-clients">
          <div class="an-card-head"><h2>Client distribution</h2><span class="an-aux">who's calling · ${esc(_range)}</span>
            <button class="an-expand" data-an-expand="clients" title="Expand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button></div>
          <div class="an-card-body" id="an-clients-host"><div class="an-ph">loading&hellip;</div></div>
        </div>
        <div class="card an-card-confidence">
          <div class="an-card-head"><h2>Confidence ladder</h2><span class="an-aux">trust profile · ${esc(_range)}</span>
            <button class="an-expand" data-an-expand="confidence" title="Expand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button></div>
          <div class="an-card-body" id="an-confidence-host"><div class="an-ph">loading&hellip;</div></div>
        </div>
      </div>

      <!-- Severity + bloom -->
      <div class="an-row an-row-2">
        <div class="card">
          <div class="an-card-head"><h2>Severity spectrum</h2>
            <button class="an-expand" data-an-expand="severity" title="Expand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button></div>
          <div class="an-card-body" id="an-severity-host"><div class="an-ph">loading&hellip;</div></div>
        </div>
        <div class="card">
          <div class="an-card-head"><h2>Disease bloom</h2>
            <button class="an-expand" data-an-expand="bloom" title="Expand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button></div>
          <div class="an-card-body" id="an-bloom-host"><div class="an-ph">loading&hellip;</div></div>
        </div>
      </div>

      <!-- Payloads -->
      <div class="card">
        <div class="an-card-head"><h2>Response payloads</h2><span class="an-aux" id="an-payload-aux">·</span>
          <button class="an-expand" data-an-expand="payloads" title="Expand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button></div>
        <div class="an-card-body" id="an-payloads-host"><div class="an-ph">loading&hellip;</div></div>
      </div>

      <!-- Recent scans -->
      <div class="card">
        <div class="an-card-head"><h2>Recent scans</h2></div>
        <div class="an-card-body" id="an-recent-host"><div class="an-ph">loading&hellip;</div></div>
      </div>
    </div>`;
  }

  // ── widget: quota (leaf fill) ───────────────────────────────────────────
  function renderQuota(d) {
    const host = $('#an-quota'); if (!host) return;
    const q = d || {};
    const u = 'q' + Math.random().toString(36).slice(2, 7);
    if (q.quota_per_day == null) {
      // no-quota placeholder — outline leaf
      host.innerHTML = `
        <div class="an-quota-grid an-quota-empty">
          <svg viewBox="0 0 120 120" class="an-leaf-svg" aria-hidden="true">
            <path d="M60 14 C 26 30 22 86 60 108 C 98 86 94 30 60 14 Z" fill="none"
              stroke="var(--c-accent,#52b788)" stroke-width="1.6" stroke-dasharray="4 5"/>
            <path d="M60 104 L60 24" stroke="var(--c-accent,#52b788)" stroke-width="1" stroke-dasharray="3 5" opacity=".6"/>
          </svg>
          <div class="an-quota-meta">
            <h3>No quota set · unlimited</h3>
            <p>requests on this key are served without a daily cap.</p>
            <div class="an-quota-cta">
              <button class="an-btn primary" data-an-setquota>Set quota</button>
              <a class="an-btn" href="/docs" data-an-href>Read about limits</a>
            </div>
          </div>
        </div>`;
    } else {
      const pct = Math.max(0, Math.min(100, q.pct || 0));
      const fillY = 108 - (pct / 100) * 84;     // leaf body roughly y:24..108
      const tone = pct >= 90 ? 'var(--c-danger,#b3402f)' : pct >= 75 ? 'var(--c-amber,#c98a2b)' : 'var(--c-accent,#52b788)';
      const exhaust = q.projected_exhaust_iso && window.APIN && APIN.time
        ? APIN.time.localTime(q.projected_exhaust_iso) : '—';
      const resetH = q.resets_in_seconds != null ? Math.floor(q.resets_in_seconds / 3600) : '—';
      const resetM = q.resets_in_seconds != null ? Math.floor((q.resets_in_seconds % 3600) / 60) : '—';
      const burn = q.burn || [];
      const bmax = Math.max(1, ...burn);
      const spark = burn.map((v, i) =>
        `<rect x="${i * 7}" y="${24 - (v / bmax) * 22}" width="5" height="${(v / bmax) * 22 + 1}" rx="1"
          fill="var(--ink-mute,#9a907a)" opacity="${0.4 + 0.5 * (v / bmax)}"/>`).join('');
      host.innerHTML = `
        <div class="an-quota-grid">
          <div class="an-leaf-wrap">
            <svg viewBox="0 0 120 120" class="an-leaf-svg" aria-hidden="true">
              <defs><clipPath id="${u}-clip"><path d="M60 14 C 26 30 22 86 60 108 C 98 86 94 30 60 14 Z"/></clipPath></defs>
              <path d="M60 14 C 26 30 22 86 60 108 C 98 86 94 30 60 14 Z" fill="var(--paper-deep,#efe7d4)" stroke="${tone}" stroke-width="1.6"/>
              <g clip-path="url(#${u}-clip)">
                <rect class="an-leaf-fill" x="0" y="${fillY}" width="120" height="120" fill="${tone}" opacity="0.5"/>
                <path class="an-leaf-wave" d="M0 ${fillY} Q 30 ${fillY - 4} 60 ${fillY} T 120 ${fillY} V120 H0 Z" fill="${tone}" opacity="0.32"/>
              </g>
              <path d="M60 104 L60 24" stroke="${tone}" stroke-width="0.8" opacity=".55"/>
            </svg>
          </div>
          <div class="an-quota-meta">
            <div class="an-quota-big"><b>${fmtNum(q.used_today)}</b> / ${fmtNum(q.quota_per_day)} <span>today</span></div>
            <div class="an-quota-pct" style="color:${tone}">${pct.toFixed(1)}% used · ${fmtNum(q.remaining)} left</div>
            <div class="an-quota-rows">
              <span>rate <b>${fmtNum(q.rate_now)}</b>/${q.rate_limit_per_min != null ? fmtNum(q.rate_limit_per_min) : '∞'} per min</span>
              <span>resets in <b>${resetH}h ${resetM}m</b></span>
              <span>est. exhaust <b>${esc(exhaust)}</b></span>
            </div>
            <svg viewBox="0 0 168 26" class="an-quota-spark" aria-hidden="true">${spark}</svg>
            <button class="an-btn" data-an-setquota>Edit quota</button>
          </div>
        </div>`;
    }
    host.querySelectorAll('[data-an-setquota]').forEach(b =>
      b.addEventListener('click', () => openSetQuota(q)));
    host.querySelectorAll('[data-an-href]').forEach(b =>
      b.addEventListener('click', e => { e.preventDefault(); location.href = b.getAttribute('href'); }));
  }

  // 9.O.4 · real in-page quota editor (was a blank lightbox + fake save)
  function openSetQuota(q) {
    const MD = window.APIN && window.APIN.modal;
    if (!MD) return;
    MD.open({
      icon: 'i-gauge', title: 'Set quota & rate limit',
      body: (el) => {
        el.innerHTML =
          '<div class="apm-field"><label>Daily quota (req/day)</label><input class="apm-input" id="sq-day" type="number" min="0" placeholder="blank = unlimited" value="' + (q && q.quota_per_day != null ? q.quota_per_day : '') + '"></div>' +
          '<div class="apm-field"><label>Per-minute rate (req/min)</label><input class="apm-input" id="sq-rate" type="number" min="0" placeholder="blank = unlimited" value="' + (q && q.rate_limit_per_min != null ? q.rate_limit_per_min : '') + '"><div class="hint">Leave blank for no limit. Applies immediately to this key.</div></div>';
      },
      actions: [
        { label: 'Cancel', kind: 'ghost' },
        { label: 'Save', kind: 'primary', busyLabel: 'Saving…', onClick: async (ctx) => {
          const day = ctx.query('#sq-day').value.trim(), rate = ctx.query('#sq-rate').value.trim();
          const ka = window.APIN && window.APIN.keyActions;
          if (!ka || !_pid) throw new Error('Cannot save — open this key to edit its quota.');
          const payload = { quota_per_day: day === '' ? null : parseInt(day, 10), rate_limit_per_min: rate === '' ? null : parseInt(rate, 10) };
          const { body } = await ka.write('/api/account/keys/' + encodeURIComponent(_pid), 'PATCH', payload);
          if (!body.ok) throw new Error(ka.writeErr(body));
          api('quota').then(d => { _data.quota = d; renderQuota(d); }).catch(() => { });
        } },
      ],
    });
  }

  // ── widget: streamgraph (shared by request stream + disease bloom) ──────
  function streamGraph(host, series, opts) {
    opts = opts || {};
    if (!host) return;
    if (!series || !series.length) { setHostState(host, 'empty', opts.empty); return; }
    const W = 600, H = opts.height || 150, pad = 6;
    const n = series[0].values.length;
    const palette = ['var(--c-accent,#52b788)', 'var(--ochre-deep,#b6822a)', 'var(--green-deep,#2d6a4f)',
      'var(--ink-mute,#9a907a)', 'var(--c-amber,#c98a2b)', 'var(--c-danger,#b3402f)'];
    // stacked baseline-centered (streamgraph)
    const stackTop = new Array(n).fill(0), stackBot = new Array(n).fill(0);
    const totals = new Array(n).fill(0);
    series.forEach(s => s.values.forEach((v, i) => totals[i] += v));
    const maxTot = Math.max(1, ...totals);
    let baseline = totals.map(t => (H / 2) - (t / maxTot) * (H / 2 - pad) / 1);
    const x = i => pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
    const yScale = (H - 2 * pad) / maxTot;
    let acc = new Array(n).fill(0);
    let paths = '';
    series.forEach((s, si) => {
      const top = [], bot = [];
      for (let i = 0; i < n; i++) {
        const b0 = (H / 2) - (totals[i] * yScale) / 2 + acc[i] * yScale;
        const b1 = b0 + s.values[i] * yScale;
        top.push([x(i), b0]); bot.push([x(i), b1]);
        acc[i] += s.values[i];
      }
      const dTop = top.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
      const dBot = bot.slice().reverse().map(p => 'L' + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
      paths += `<path class="an-stream-band" data-an-key="${esc(s.disease || s.path || s.key)}" d="${dTop} ${dBot} Z"
        fill="${palette[si % palette.length]}" opacity="0.72"><title>${esc(s.disease || s.path)} · ${fmtNum(s.total)}</title></path>`;
    });
    const legend = series.map((s, si) =>
      `<span class="an-sl"><i style="background:${palette[si % palette.length]}"></i>${esc((s.disease || s.path || '').replace(/^\//, ''))} <b>${fmtNum(s.total)}</b></span>`).join('');
    host.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="an-stream-svg">${paths}</svg>
       <div class="an-stream-legend">${legend}</div>`;
    host.querySelectorAll('.an-stream-band').forEach(p => {
      p.addEventListener('mouseenter', () => p.style.opacity = '1');
      p.addEventListener('mouseleave', () => p.style.opacity = '0.72');
      p.addEventListener('click', () => {
        const k = p.getAttribute('data-an-key');
        if (opts.onPick) opts.onPick(k);
      });
    });
  }

  function renderStreams(d) { streamGraph($('#an-streams-host'), (d || {}).series, { height: 150, empty: 'no requests in this window' }); }
  function renderBloom(d) {
    streamGraph($('#an-bloom-host'), (d || {}).series, {
      height: 150, empty: 'no diagnoses yet',
      onPick: k => setFocus({ disease: k })
    });
  }

  // ── widget: severity spectrum ───────────────────────────────────────────
  function renderSeverity(d) {
    const host = $('#an-severity-host'); if (!host) return;
    const scans = (d && d.scans) || [];
    if (!scans.length) { setHostState(host, 'empty', 'no scans in this window'); return; }
    const W = 600, H = 120;
    const sevPos = { mild: 0.15, moderate: 0.5, severe: 0.85, unknown: 0.5 };
    const dots = scans.map((s, i) => {
      const base = sevPos[s.severity] != null ? sevPos[s.severity] : 0.5;
      const jx = base + (Math.random() - 0.5) * 0.12;
      const jy = 0.25 + Math.random() * 0.5;
      const r = 3 + (s.confidence || 0.5) * 3;
      return `<circle class="an-sev-dot" data-uid="${esc(s.scan_uid)}" cx="${(jx * W).toFixed(1)}" cy="${(jy * H).toFixed(1)}" r="${r.toFixed(1)}"
        fill="${sevTone(s.severity)}" opacity="0.8" style="--d:${(i % 12) * 0.3}s"><title>${esc(titleCase(s.diagnosis))} · ${s.severity} · conf ${(s.confidence || 0).toFixed(2)}</title></circle>`;
    }).join('');
    const dist = (d && d.distribution) || {};
    const counts = ['mild', 'moderate', 'severe'].map(k =>
      `<span class="an-sev-c" style="color:${sevTone(k)}">${k} <b>${fmtNum(dist[k] || 0)}</b></span>`).join('');
    host.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="an-sev-svg">
         <defs><linearGradient id="an-sev-grad" x1="0" x2="1">
           <stop offset="0" stop-color="var(--c-accent,#52b788)" stop-opacity=".18"/>
           <stop offset="0.5" stop-color="var(--c-amber,#c98a2b)" stop-opacity=".18"/>
           <stop offset="1" stop-color="var(--c-danger,#b3402f)" stop-opacity=".22"/>
         </linearGradient></defs>
         <rect x="0" y="0" width="${W}" height="${H}" fill="url(#an-sev-grad)"/>
         ${dots}
       </svg>
       <div class="an-sev-counts">${counts}</div>`;
    host.querySelectorAll('.an-sev-dot').forEach(c =>
      c.addEventListener('click', () => {
        const uid = c.getAttribute('data-uid');
        if (uid && window.APIN && APIN.requestDrawer) { /* drawer keyed by request id, not scan — toast for now */ }
        if (window.APIN && APIN.toast) APIN.toast({ msg: 'scan ' + uid });
      }));
  }

  // ── widget: payloads ────────────────────────────────────────────────────
  let _payIdx = 0;
  function renderPayloads(d) {
    const host = $('#an-payloads-host'); if (!host) return;
    const st = (d && d.stats) || {}, samples = (d && d.samples) || [];
    const aux = $('#an-payload-aux'); if (aux) aux.textContent = fmtNum(st.responses) + ' responses';
    const statRow =
      `<div class="an-pay-stats">
        <span>avg confidence <b>${st.avg_confidence != null ? st.avg_confidence.toFixed(2) : '·'}</b></span>
        <span>OOD-flagged <b>${st.ood_pct != null ? st.ood_pct + '%' : '·'}</b></span>
        <span>responses <b>${fmtNum(st.responses)}</b></span>
      </div>`;
    if (!samples.length) { host.innerHTML = statRow + '<div class="an-ph an-ph-empty">no response previews captured</div>'; return; }
    _payIdx = Math.min(_payIdx, samples.length - 1);
    const renderSample = () => {
      const s = samples[_payIdx] || {};
      let pretty = s.preview || '';
      try { pretty = JSON.stringify(JSON.parse(s.preview), null, 2); } catch (e) { }
      const codeEl = host.querySelector('.an-pay-code');
      if (codeEl) codeEl.textContent = pretty.length > 1800 ? pretty.slice(0, 1800) + '\n  …(truncated)' : pretty;
      const meta = host.querySelector('.an-pay-meta');
      if (meta) meta.innerHTML = `${esc(s.path)} · <b>${s.status}</b> · ${esc(s.ctype || '')} · ${_payIdx + 1}/${samples.length}`;
    };
    host.innerHTML = statRow +
      `<div class="an-pay-box">
        <div class="an-pay-bar"><span class="an-pay-meta"></span>
          <span class="an-pay-nav"><button data-an-pay="-1">‹</button><button data-an-pay="1">›</button></span></div>
        <pre class="an-pay-code"></pre>
      </div>`;
    renderSample();
    host.querySelectorAll('[data-an-pay]').forEach(b =>
      b.addEventListener('click', () => {
        _payIdx = (_payIdx + parseInt(b.getAttribute('data-an-pay'), 10) + samples.length) % samples.length;
        renderSample();
      }));
  }

  // ── widget: recent scans (from severity sample) ─────────────────────────
  function renderRecent(d) {
    const host = $('#an-recent-host'); if (!host) return;
    const scans = ((d || {}).scans || []).slice(0, 12);
    if (!scans.length) { setHostState(host, 'empty', 'no scans yet'); return; }
    host.innerHTML = '<div class="an-recent">' + scans.map(s =>
      `<div class="an-recent-row" data-uid="${esc(s.scan_uid)}">
        <span class="an-rec-sev" style="background:${sevTone(s.severity)}"></span>
        <span class="an-rec-diag">${esc(titleCase(s.diagnosis))}</span>
        <span class="an-rec-sev-t">${esc(s.severity)}</span>
        <span class="an-rec-conf">${(s.confidence || 0).toFixed(2)}</span>
        ${s.is_ood ? '<span class="an-rec-ood">OOD</span>' : ''}
      </div>`).join('') + '</div>';
  }

  // ── legend (geo) ────────────────────────────────────────────────────────
  function renderLegend(geo) {
    const host = $('#an-legend-host'); if (!host) return;
    const regions = (geo && geo.regions) || [];
    if (!regions.length) {
      setHostState(host, 'empty',
        _geoSource === 'requests' ? 'no geolocated requests' : 'no geolocated scans');
      return;
    }
    const max = Math.max(1, ...regions.map(r => r.count));
    host.innerHTML = '<div class="an-leg">' + regions.slice(0, 12).map(r => {
      const name = r.district || r.state || r.cc || '?';
      const sub = [r.state, r.cc].filter(Boolean).join(' · ');
      return `<button class="an-leg-row" data-an-region="${esc(name)}">
        <span class="an-leg-name">${esc(name)}<small>${esc(sub)}</small></span>
        <span class="an-leg-bar"><i style="width:${(r.count / max * 100).toFixed(0)}%"></i></span>
        <span class="an-leg-n">${fmtNum(r.count)}</span>
      </button>`;
    }).join('') +
      (geo.unmapped ? `<div class="an-leg-unmapped">${fmtNum(geo.unmapped)} unmapped (no GPS)</div>` : '') +
      '</div>';
    host.querySelectorAll('.an-leg-row').forEach(b =>
      b.addEventListener('click', () => setFocus({ region: b.getAttribute('data-an-region') })));
  }

  // ── focus bus (cross-widget linking) ────────────────────────────────────
  function setFocus(f) {
    Object.assign(_focus, f);
    renderChips();
    if (APIN.analyticsGeo && _focus.region) APIN.analyticsGeo.focusRegion(_focus.region);
    if (APIN.analyticsField && _focus.disease) APIN.analyticsField.focusDisease(_focus.disease);
  }
  function renderChips() {
    const host = $('#an-chips'); if (!host) return;
    const chips = [];
    if (_focus.region) chips.push(['region', _focus.region]);
    if (_focus.disease) chips.push(['disease', titleCase(_focus.disease)]);
    if (_focus.crop) chips.push(['crop', titleCase(_focus.crop)]);
    host.innerHTML = chips.map(([k, v]) =>
      `<span class="an-chip" data-an-clear="${k}">${esc(k)}: ${esc(v)} ×</span>`).join('');
    host.querySelectorAll('[data-an-clear]').forEach(c =>
      c.addEventListener('click', () => { _focus[c.getAttribute('data-an-clear')] = null; renderChips(); }));
  }

  // ── expanded states ─────────────────────────────────────────────────────
  function expand(which) {
    if (!(window.APIN && APIN.lightbox)) return;
    const titles = {
      geo: 'Geographic origin', field: 'Inference field', streams: 'Request stream',
      severity: 'Severity spectrum', bloom: 'Disease bloom', payloads: 'Response payloads',
      clients: 'Client Atlas', confidence: 'Trust Core Sample'
    };
    const card = $('[data-an-expand="' + which + '"]');
    // apin_lightbox.js populates its body via build(panel) — NOT an html option.
    APIN.lightbox.open({
      title: titles[which] || which,
      hashKey: 'an-' + which,
      sourceCard: card ? card.closest('.card') : null,
      build: (panel) => {
        const el = document.createElement('div');
        el.id = 'an-exp-' + which; el.className = 'an-exp';
        panel.appendChild(el);
        if (which === 'geo' && APIN.analyticsGeo && APIN.analyticsGeo.renderExpanded) APIN.analyticsGeo.renderExpanded(el, _data.geoAll, { source: _geoSource });
        else if (which === 'field' && APIN.analyticsField && APIN.analyticsField.renderExpanded) APIN.analyticsField.renderExpanded(el, _data.field);
        else if (which === 'streams') streamGraph(el, (_data.streams || {}).series, { height: 320 });
        else if (which === 'clients' && APIN.analyticsClients && APIN.analyticsClients.renderExpanded) APIN.analyticsClients.renderExpanded(el, _data.clients);
        else if (which === 'confidence' && APIN.analyticsConfidence && APIN.analyticsConfidence.renderExpanded) APIN.analyticsConfidence.renderExpanded(el, _data.confidence);
        else if (which === 'bloom') streamGraph(el, (_data.bloom || {}).series, { height: 320 });
        else if (which === 'severity') renderSeverityInto(el, _data.severity);
        else if (which === 'payloads') el.innerHTML = '<pre class="an-pay-code">' + esc(JSON.stringify((_data.payloads || {}).samples || [], null, 2).slice(0, 4000)) + '</pre>';
      }
    });
  }
  function renderSeverityInto(el, d) {
    const scans = (d && d.scans) || [];
    el.innerHTML = '<div style="font:13px JetBrains Mono,monospace;color:var(--ink-soft)">' +
      scans.map(s => `${titleCase(s.diagnosis)} — ${s.severity} (${(s.confidence || 0).toFixed(2)})`).join('<br>') + '</div>';
  }

  // ── controls ────────────────────────────────────────────────────────────
  function wire() {
    $$('[data-an-range]').forEach(b => b.addEventListener('click', () => {
      _range = b.getAttribute('data-an-range');
      $$('[data-an-range]').forEach(x => x.removeAttribute('aria-pressed'));
      b.setAttribute('aria-pressed', 'true');
      refreshAll();
      if (APIN.analyticsGeo) APIN.analyticsGeo.setRange(_range);
      if (APIN.analyticsField) APIN.analyticsField.setRange(_range);
    }));
    $$('[data-an-expand]').forEach(b =>
      b.addEventListener('click', () => expand(b.getAttribute('data-an-expand'))));
    $$('[data-an-layer]').forEach(b => b.addEventListener('click', () => {
      $$('[data-an-layer]').forEach(x => x.removeAttribute('aria-pressed'));
      b.setAttribute('aria-pressed', 'true');
      if (APIN.analyticsGeo) APIN.analyticsGeo.setLayer(b.getAttribute('data-an-layer'));
    }));
    $$('[data-an-source]').forEach(b => b.addEventListener('click', () => {
      _geoSource = b.getAttribute('data-an-source');
      $$('[data-an-source]').forEach(x => x.removeAttribute('aria-pressed'));
      b.setAttribute('aria-pressed', 'true');
      if (APIN.analyticsGeo) APIN.analyticsGeo.setSource(_geoSource);
      fetchAllGeo();   // zoom-LOD owns tier; just re-fetch all 3 for the new source
    }));
    $$('[data-an-fmode]').forEach(b => b.addEventListener('click', () => {
      $$('[data-an-fmode]').forEach(x => x.removeAttribute('aria-pressed'));
      b.setAttribute('aria-pressed', 'true');
      if (APIN.analyticsField) APIN.analyticsField.setMode(b.getAttribute('data-an-fmode'));
    }));
  }

  // ── data ────────────────────────────────────────────────────────────────
  // Zoom-LOD: fetch all three tier-aggregations ONCE (small JSON; the geo
  // lookups are cached after the first call). Zoom then switches which tier
  // renders, with no refetch.
  async function fetchAllGeo() {
    const src = '&source=' + _geoSource;
    const [country, state, district] = await Promise.all([
      api('geo?tier=country' + src), api('geo?tier=state' + src), api('geo?tier=district' + src),
    ]);
    _data.geoAll = { country, state, district };
    _data.geo = state;
    renderLegend(state);
    if (APIN.analyticsGeo) APIN.analyticsGeo.setData(_data.geoAll);
  }

  async function refreshAll() {
    const jobs = [
      ['quota', api('quota').then(d => { _data.quota = d; renderQuota(d); })],
      ['clients', api('clients').then(d => { _data.clients = d; if (APIN.analyticsClients) APIN.analyticsClients.setData(d); })],
      ['confidence', api('confidence').then(d => { _data.confidence = d; if (APIN.analyticsConfidence) APIN.analyticsConfidence.setData(d); })],
      ['severity', api('severity').then(d => { _data.severity = d; renderSeverity(d); renderRecent(d); })],
      ['bloom', api('bloom').then(d => { _data.bloom = d; renderBloom(d); })],
      ['payloads', api('payloads').then(d => { _data.payloads = d; renderPayloads(d); })],
      ['geo', fetchAllGeo()],
      ['field', api('field').then(d => { _data.field = d; if (APIN.analyticsField) APIN.analyticsField.setData(d); })],
    ];
    await Promise.allSettled(jobs.map(j => j[1]));
  }

  // ── live SSE ──────────────────────────────────────────────────────────
  function connectLive() {
    try {
      _es = new EventSource('/api/account/usage/stream', { withCredentials: true });
      _es.onmessage = (ev) => {
        let e; try { e = JSON.parse(ev.data); } catch (_) { return; }
        if (!e) return;
        // pulse the live indicator on any event
        const live = $('#an-live'); if (live) { live.classList.add('an-live-hit'); setTimeout(() => live.classList.remove('an-live-hit'), 700); }
        if (e.type === 'scan') {
          if (APIN.analyticsGeo) APIN.analyticsGeo.onScan(e);
          if (APIN.analyticsField) APIN.analyticsField.onScan(e);
          if (APIN.analyticsConfidence) APIN.analyticsConfidence.onScan(e);  // 9.N.T34 · ladder reacts live
        } else if (e.type === 'request') {
          if (APIN.analyticsClients) APIN.analyticsClients.onRequest(e);     // 9.N.T34 · visitors' log reacts live
        }
      };
    } catch (_) { }
    // Periodic LIGHT refresh — SVG widgets only. The two WebGL scenes update
    // via the SSE scan feed, NOT a full rebuild (rebuilding them on a timer
    // caused jank + memory growth). Range toggles still do a full refreshAll.
    _liveTimer = setInterval(() => {
      if (document.visibilityState !== 'visible') return;
      api('quota').then(d => { _data.quota = d; renderQuota(d); }).catch(() => { });
      api('severity').then(d => { _data.severity = d; renderSeverity(d); renderRecent(d); }).catch(() => { });
      // reconcile the two live widgets against the source of truth (the SSE
      // deltas keep them current between ticks; this corrects any drift).
      api('clients').then(d => { _data.clients = d; if (APIN.analyticsClients) APIN.analyticsClients.setData(d); }).catch(() => { });
      api('confidence').then(d => { _data.confidence = d; if (APIN.analyticsConfidence) APIN.analyticsConfidence.setData(d); }).catch(() => { });
    }, 45000);
  }

  // ── lazy-load the heavy 3D modules ──────────────────────────────────────
  function loadGeoModule() {
    if (_geoLib.loaded) { initGeo(); return; }
    if (_geoLib.loading) return;
    // 9.N.T31h · MapLibre GL JS + OpenFreeMap vector tiles (real tile LOD /
    // viewport loading + globe→flat projection). Lazy CSS link + JS.
    if (!document.getElementById('an-maplibre-css')) {
      const lk = document.createElement('link'); lk.id = 'an-maplibre-css';
      lk.rel = 'stylesheet'; lk.href = '/static/maplibre-gl.css?v=5';
      document.head.appendChild(lk);
    }
    _geoLib.loading = loadScript('/static/maplibre-gl.js?v=5')
      .then(() => loadScript('/static/console_analytics_geo.js?v=9t31u'))
      .then(() => { _geoLib.loaded = true; initGeo(); })
      .catch(() => { const h = $('#an-geo-host'); if (h) h.innerHTML = '<div class="an-ph an-ph-empty">map unavailable</div>'; });
  }
  function initGeo() {
    if (APIN.analyticsGeo) {
      APIN.analyticsGeo.mount($('#an-geo-host'), {
        publicId: _pid,
        onPickRegion: (name) => setFocus({ region: name }),
        onLevel: (tier, geo) => {
          // zoom-LOD still refreshes the Top-regions legend to match the tier,
          // but the on-map zoom-level pill was removed per design feedback.
          if (geo) renderLegend(geo);
        }
      });
      APIN.analyticsGeo.setSource(_geoSource);
      if (_data.geoAll) APIN.analyticsGeo.setData(_data.geoAll);
    }
  }
  function loadFieldModule() {
    if (_fieldLib.loaded) { initField(); return; }
    if (_fieldLib.loading) return;
    // three.min.js may already be loaded by the terrain; load if absent
    const needThree = (typeof window.THREE === 'undefined');
    const chain = needThree ? loadScript('/static/three.min.js?v=149') : Promise.resolve();
    _fieldLib.loading = chain
      .then(() => loadScript('/static/console_analytics_field.js?v=9t32a'))
      .then(() => { _fieldLib.loaded = true; initField(); })
      .catch(() => { const h = $('#an-field-host'); if (h) h.innerHTML = '<div class="an-ph an-ph-empty">field unavailable</div>'; });
  }
  function initField() {
    if (APIN.analyticsField) {
      APIN.analyticsField.mount($('#an-field-host'), {
        insightHost: $('#an-field-insight'),
        onPickDisease: (d) => setFocus({ disease: d }),
        onPickCrop: (c) => setFocus({ crop: c })
      });
      if (_data.field) APIN.analyticsField.setData(_data.field);
    }
  }

  // 9.N.T34 · lightweight SVG/DOM widgets — Visitors' Log + Soil-Horizon ladder
  function loadNewWidgets() {
    const V = '?v=9t34a';
    const jobs = [
      APIN.analyticsClients ? Promise.resolve() : loadScript('/static/console_analytics_clients.js' + V),
      APIN.analyticsConfidence ? Promise.resolve() : loadScript('/static/console_analytics_confidence.js' + V),
    ];
    Promise.all(jobs).then(() => {
      if (APIN.analyticsClients && $('#an-clients-host')) {
        APIN.analyticsClients.mount($('#an-clients-host'), { onPick: (f) => setFocus(f) });
        if (_data.clients) APIN.analyticsClients.setData(_data.clients);
      }
      if (APIN.analyticsConfidence && $('#an-confidence-host')) {
        APIN.analyticsConfidence.mount($('#an-confidence-host'), { onPick: (f) => setFocus(f) });
        if (_data.confidence) APIN.analyticsConfidence.setData(_data.confidence);
      }
    }).catch(() => { });
  }

  // ── mount / teardown ────────────────────────────────────────────────────
  async function mount(rootEl, opts) {
    _root = rootEl; opts = opts || {};
    _pid = opts.publicId || null;
    if (_booted && _root.querySelector('.an-wrap')) { refreshAll(); return; }
    _booted = true;
    rootEl.innerHTML = layoutHTML();
    wire();
    renderChips();
    await refreshAll();
    connectLive();
    loadNewWidgets();
    // defer heavy 3D until the layout has painted (so containers have size)
    requestAnimationFrame(() => { loadGeoModule(); loadFieldModule(); });
  }
  function deactivate() {
    if (_es) { try { _es.close(); } catch (_) { } _es = null; }
    if (_liveTimer) { clearInterval(_liveTimer); _liveTimer = null; }
    if (APIN.analyticsGeo && APIN.analyticsGeo.pause) APIN.analyticsGeo.pause();
    if (APIN.analyticsField && APIN.analyticsField.pause) APIN.analyticsField.pause();
  }
  function activate() {
    if (APIN.analyticsGeo && APIN.analyticsGeo.resume) APIN.analyticsGeo.resume();
    if (APIN.analyticsField && APIN.analyticsField.resume) APIN.analyticsField.resume();
  }

  A.mount = mount; A.deactivate = deactivate; A.activate = activate;
  A.expose = { setFocus };
  window.APIN = window.APIN || {};
  window.APIN.analytics = A;
  // Expose the scoped fetch helper so lazy modules (geo expanded console)
  // can hit /geo/region, /geo/replay, compare — range/key/env auto-injected.
  window.APIN.analyticsApi = api;
  window.APIN.analyticsFocus = (f) => setFocus(f);
  // current time-range + source so lazy modules (Compare window) can label
  // the "this window vs previous window" date spans.
  window.APIN.analyticsCtx = () => ({ range: _range, source: _geoSource, env: _env });
})();
