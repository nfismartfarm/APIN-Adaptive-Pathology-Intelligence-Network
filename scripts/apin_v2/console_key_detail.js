/* Phase 9.C — per-key detail page client.
 *
 * Routing model
 * =============
 * Tabs are hash-anchored. Each tab's content is rendered on demand by a
 * loader function:
 *
 *   #overview   → loadOverview()  KPI strip + sparkline + top endpoints
 *   #usage      → loadUsageTab()  Rich charts (filled by Phase 9.E; this
 *                                  file leaves the host empty for now).
 *   #requests   → loadRequests()  Paginated, filterable raw log
 *   #audit      → loadAudit()     Hash-chained audit timeline
 *   #settings   → renderSettings() Read-only DL of every key field
 *
 * The masthead always shows live key metadata so identifying the key
 * is one render away regardless of which tab is active. A 15-second
 * polling refresh updates whichever tab is currently visible (cheap on
 * SQLite, lets users see new requests appear without reload). Tab
 * switches are instant — no re-fetch unless cached data is older than
 * the visibility threshold.
 *
 * Action buttons (rotate/edit/disable/enable/delete) currently round-
 * trip through the keys-list page (which already hosts the modals).
 * A full inline action set would duplicate 600+ LOC from keys.js —
 * out of 9.C scope but filed as WI-9.F-INLINE-ACTIONS.
 */
(function () {
  'use strict';

  const csrfMeta = document.querySelector('meta[name="csrf-token"]');
  let csrf = csrfMeta ? csrfMeta.content : '';
  const pidMeta = document.querySelector('meta[name="key-public-id"]');
  const PID = pidMeta ? pidMeta.content : '';
  const TABS = ['overview', 'traffic', 'endpoints', 'usage', 'requests', 'audit', 'settings'];
  const DEFAULT_TAB = 'overview';

  // ─── helpers ──────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function fmtAgo(iso) {
    if (!iso) return 'never';
    const ms = Date.now() - new Date(iso).getTime();
    if (isNaN(ms)) return iso;
    if (ms < 0) return 'in the future';
    if (ms < 60_000) return Math.max(1, Math.round(ms / 1000)) + 's ago';
    if (ms < 3600_000) return Math.round(ms / 60_000) + 'm ago';
    if (ms < 86_400_000) return Math.round(ms / 3600_000) + 'h ago';
    if (ms < 7 * 86_400_000) return Math.round(ms / 86_400_000) + 'd ago';
    return new Date(iso).toLocaleDateString();
  }
  function fmtNum(n) {
    if (n == null || isNaN(n)) return '·';
    n = Number(n);
    if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + 'k';
    return String(Math.round(n));
  }
  function fmtBytes(n) {
    if (n == null) return '·';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' kB';
    return (n / 1024 / 1024).toFixed(2) + ' MB';
  }
  function statusBucket(s) {
    s = Number(s) || 0;
    if (s >= 500) return 'stat-5xx';
    if (s >= 400) return 'stat-4xx';
    if (s >= 300) return 'stat-3xx';
    if (s >= 200) return 'stat-2xx';
    return '';
  }
  function methClass(m) {
    if (!m) return '';
    const u = m.toUpperCase();
    if (u === 'GET' || u === 'POST' || u === 'PUT' || u === 'PATCH' || u === 'DELETE') {
      return 'meth-' + u;
    }
    return '';
  }

  // ─── API ──────────────────────────────────────────────────────────────
  async function api(url) {
    const r = await fetch(url, { credentials: 'include' });
    if (r.status === 401) {
      // Session lost — bounce to dashboard so they can re-auth.
      window.location.href = '/dashboard';
      throw new Error('unauthenticated');
    }
    const body = await r.json().catch(() => ({}));
    return { status: r.status, body };
  }

  // ─── state ────────────────────────────────────────────────────────────
  let keyData = null;          // last successful /keys/{pid} response
  let activeTab = DEFAULT_TAB;
  let pollTimer = null;
  let reqCursor = null;        // pagination cursor for Requests tab

  // ─── Masthead (always rendered, regardless of tab) ───────────────────
  function renderMasthead(k) {
    keyData = k;
    $('crumb-key-id').textContent = k.name || '(unnamed)';
    const nameEl = $('key-name');
    nameEl.textContent = k.name || '(unnamed)';
    nameEl.title = 'Click to rename';
    nameEl.style.cursor = 'text';
    if (!nameEl._editWired) { nameEl._editWired = true; nameEl.addEventListener('click', () => startNameEdit(nameEl)); }
    document.title = (k.name || k.public_id) + ' · APIN Console';

    const tokenStub =
      'apin_' + escHtml(k.environment || '?') +
      '_******' + escHtml(k.last_four || '????');
    $('key-id-line').innerHTML =
      tokenStub + ' <code>' + escHtml(k.public_id) + '</code>' +
      ' <button class="copy-pid" type="button" data-copy-pid>' +
      'copy id</button>';
    const btn = $('key-id-line').querySelector('[data-copy-pid]');
    if (btn) {
      btn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(k.public_id);
          btn.classList.add('copied');
          btn.textContent = 'copied';
          setTimeout(() => {
            btn.classList.remove('copied');
            btn.textContent = 'copy id';
          }, 1400);
        } catch {}
      });
    }

    const statusCls = 'status-' + (k.status || 'disabled');
    const meta = $('key-meta-row');
    const parts = [
      `<span class="status-pill ${statusCls}">${escHtml(k.status)}</span>`,
      `<span class="env-pill">${escHtml(k.environment)}</span>`,
    ];
    if (k.group_name) {
      parts.push(`<span class="group-pill" title="group: ${escHtml(k.group_name)}${k.group_role === 'special' ? ' (special)' : ''}">${escHtml(k.group_name)}${k.group_role === 'special' ? ' · special' : ''}</span>`);
    }
    parts.push(
      `<span class="pill">created <b>${fmtAgo(k.created_at)}</b></span>`,
      `<span class="pill">last used <b>${k.last_used_at ? fmtAgo(k.last_used_at) : 'never'}</b></span>`
    );
    if (k.expires_at) {
      parts.push(`<span class="pill">expires <b>${fmtAgo(k.expires_at)}</b></span>`);
    } else {
      parts.push('<span class="pill">no expiry</span>');
    }
    if (k.rate_limit_per_min) {
      parts.push(`<span class="pill">rate <b>${k.rate_limit_per_min}/min</b></span>`);
    }
    if (k.quota_per_day) {
      parts.push(`<span class="pill">quota <b>${k.quota_per_day}/day</b></span>`);
    }
    meta.innerHTML = parts.join('');

    $('key-scopes').innerHTML = (k.scopes || []).map(s =>
      '<span class="scope-chip">' + escHtml(s) + '</span>').join('');

    // Disable/enable button visibility based on status
    $('btn-disable').hidden = (k.status !== 'active');
    $('btn-enable').hidden = (k.status !== 'disabled');
  }

  // 9.O.4 · masthead name click-to-edit (Stripe-style inline rename)
  function startNameEdit(h1) {
    if (h1._editing) return; h1._editing = true;
    const cur = (keyData && keyData.name) || '';
    const input = document.createElement('input'); input.value = cur; input.className = 'key-name-edit';
    input.setAttribute('maxlength', '120'); input.setAttribute('aria-label', 'Key name');
    input.style.cssText = 'font:inherit;border:1px solid var(--paper-edge,#d8cdb2);border-radius:8px;padding:2px 8px;width:min(440px,80%);background:#fff;color:var(--ink,#1a1612)';
    h1.textContent = ''; h1.appendChild(input); input.focus(); input.select();
    let done = false;
    async function finish(save) {
      if (done) return; done = true; h1._editing = false;
      const val = input.value.trim();
      if (save && val && val !== cur) {
        try {
          const { body } = await apiWrite('/api/account/keys/' + encodeURIComponent(PID), 'PATCH', { name: val });
          if (!body.ok) throw new Error(writeErr(body));
        } catch (e) {
          if (window.APIN && window.APIN.modal) { const _m = (e && e.message) || 'Could not rename.'; window.APIN.modal.confirm({ icon: 'i-warning', title: 'Rename failed', message: window.APIN.modal._esc(_m), confirmLabel: 'OK' }); }
        }
      }
      await refreshMasthead();
      if (activeTab === 'settings') renderSettings();
    }
    input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); finish(true); } else if (e.key === 'Escape') { e.preventDefault(); finish(false); } });
    input.addEventListener('blur', () => finish(true));
  }

  async function refreshMasthead() {
    const { status, body } = await api('/api/account/keys/' + encodeURIComponent(PID));
    if (status === 404) {
      document.querySelector('.page').innerHTML =
        '<div class="masthead"><div><h1>Key not found</h1>' +
        '<p>This key was deleted, or it never existed. ' +
        '<a href="/account/api/keys">Back to keys</a>.</p></div></div>';
      return false;
    }
    if (!body || !body.ok) {
      $('key-name').textContent = 'Failed to load';
      $('key-id-line').textContent = (body && body.error && body.error.message) || '';
      return false;
    }
    renderMasthead(body.data);
    return true;
  }

  // ─── Tab routing ─────────────────────────────────────────────────────
  function readHashTab() {
    const h = (location.hash || '').replace(/^#/, '').toLowerCase();
    return TABS.includes(h) ? h : DEFAULT_TAB;
  }
  function setActiveTab(name, opts) {
    if (!TABS.includes(name)) name = DEFAULT_TAB;
    activeTab = name;
    document.querySelectorAll('[data-tab]').forEach(a => {
      const on = a.dataset.tab === name;
      if (on) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    document.querySelectorAll('.tabpane').forEach(p => {
      const on = p.id === 'pane-' + name;
      p.setAttribute('aria-hidden', on ? 'false' : 'true');
    });
    if (opts && opts.silent) return;
    if (location.hash !== '#' + name) {
      history.replaceState(null, '', '#' + name);
    }
    loadActiveTab();
  }
  function loadActiveTab() {
    // Stop the Traffic / Endpoints / Analytics live loops when leaving them.
    if (activeTab !== 'traffic' && window.APIN && APIN.keyTraffic) APIN.keyTraffic.deactivate();
    if (activeTab !== 'endpoints' && window.APIN && APIN.keyEndpoints) APIN.keyEndpoints.deactivate();
    if (activeTab !== 'usage' && window.APIN && APIN.analytics) APIN.analytics.deactivate();
    if (activeTab === 'overview')  loadOverview();
    else if (activeTab === 'traffic') { if (window.APIN && APIN.keyTraffic) APIN.keyTraffic.activate(PID); }
    else if (activeTab === 'endpoints') { if (window.APIN && APIN.keyEndpoints) APIN.keyEndpoints.activate(PID); }
    else if (activeTab === 'usage')    loadAnalyticsTab();   // 9.N.T31 · Analytics
    else if (activeTab === 'requests') loadRequests({ reset: true });
    else if (activeTab === 'audit')    loadAudit();
    else if (activeTab === 'settings') renderSettings();
  }

  // 9.N.T31 · Analytics tab — lazy-load the controller (and, transitively,
  // globe.gl + boundary data + 3D field) ONLY when the tab is first opened,
  // so the rest of the site's load time is untouched.
  let _analyticsLoading = null;
  function loadAnalyticsTab() {
    const mountEl = document.getElementById('an-mount');
    if (!mountEl) return;
    if (window.APIN && APIN.analytics) {
      APIN.analytics.activate();
      APIN.analytics.mount(mountEl, { publicId: PID });
      return;
    }
    if (_analyticsLoading) return;
    mountEl.innerHTML = '<div class="placeholder">loading analytics&hellip;</div>';
    _analyticsLoading = new Promise((res) => {
      const s = document.createElement('script');
      s.src = '/static/console_analytics.js?v=9o1'; s.async = true;
      s.onload = res; s.onerror = res;
      document.head.appendChild(s);
    }).then(() => {
      if (window.APIN && APIN.analytics) APIN.analytics.mount(mountEl, { publicId: PID });
      else mountEl.innerHTML = '<div class="placeholder">analytics unavailable</div>';
    });
  }
  window.addEventListener('hashchange', () => {
    setActiveTab(readHashTab());
  });
  document.querySelectorAll('[data-tab-link]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      setActiveTab(a.dataset.tabLink);
    });
  });

  // ─── Overview ─────────────────────────────────────────────────────────
  function setKpiTile(tileId, fields) {
    const tile = $(tileId);
    if (!tile) return;
    const numEl = tile.querySelector('[data-num]');
    const dEl = tile.querySelector('[data-delta]');
    const pEl = tile.querySelector('[data-prev]');
    if (fields.value != null) {
      const unit = numEl.querySelector('.unit');
      numEl.textContent = fields.value;
      if (unit) numEl.appendChild(unit);
    }
    const d = fields.delta;
    if (d == null) {
      dEl.className = 'delta flat';
      dEl.innerHTML = '<svg class="icon" aria-hidden="true" style="width:11px;height:11px"><use href="#i-minus"/></svg> —';
    } else {
      const dirUp = d > 0;
      const goodIfDown = fields.goodIfDown === true;
      const cls = 'delta ' + (dirUp ? 'up' : (d < 0 ? 'down' : 'flat')) +
        (goodIfDown ? ' good' : '');
      dEl.className = cls;
      const arrow = dirUp ? 'i-arrow-up-right' :
                    (d < 0 ? 'i-arrow-down-right' : 'i-minus');
      dEl.innerHTML = '<svg class="icon" aria-hidden="true" style="width:11px;height:11px"><use href="#' +
        arrow + '"/></svg> ' + Math.abs(d).toFixed(1) + '%';
    }
    if (pEl) {
      pEl.textContent = fields.prevLabel || '';
    }
  }

  async function loadOverview() {
    // 9.N.9 · The bento Overview is now owned by console_key_overview.js.
    // If the new bento shell is present, this legacy renderer stands down
    // (its old element IDs — kt-requests, spark-wrap, etc. — no longer
    // exist in the DOM, so running it would throw).
    if (document.getElementById('ov-bento')) {
      // 9.N.9(a) · expose the request drawer so the ribbon can open it.
      // 9.N.T · expose filterRequests so Traffic-tab drills land on a time slice.
      window.APIN = window.APIN || {};
      window.APIN.keyDetail = Object.assign(window.APIN.keyDetail || {}, {
        openRequest: function (rid) {
          // 9.N.T9 · Prefer the shared full drawer (payload+image, burst,
          // health, export) — identical to the Usage page. Fall back to the
          // local lightweight drawer only if the shared module didn't load.
          if (window.APIN && APIN.requestDrawer && APIN.requestDrawer.open) {
            return APIN.requestDrawer.open(rid);
          }
          return openRequestDetail(rid);
        },
        filterEndpoint: function (path) {
          // Cross-link from analytics (geo dossier / compare) → Requests tab
          // filtered to a single endpoint path. Mirrors filterRequests.
          setActiveTab('requests');
          var input = document.getElementById('req-q');
          if (input) { input.value = path || ''; }
          try { loadRequests({ reset: true }); } catch (e) { }
        },
        filterRequests: function (since, until) {
          reqTimeFilter = (since && until) ? { since: since, until: until } : null;
          setActiveTab('requests');
          var pane = document.getElementById('pane-requests');
          var banner = document.getElementById('req-time-banner');
          if (reqTimeFilter && pane) {
            if (!banner) {
              banner = document.createElement('div');
              banner.id = 'req-time-banner';
              banner.style.cssText = "display:flex;align-items:center;gap:10px;background:rgba(120,110,90,.08);border:1px solid var(--paper-edge);border-radius:8px;padding:8px 12px;margin-bottom:12px;font:12px 'JetBrains Mono',monospace;color:var(--ink-soft)";
              pane.insertBefore(banner, pane.firstChild);
            }
            var fmt = (window.APIN && APIN.time) ? APIN.time.local : function (s) { return s; };
            banner.innerHTML = 'filtered: ' + fmt(since) + ' → ' + fmt(until) +
              ' <button id="req-clear-filter" style="margin-left:auto;background:none;border:1px solid var(--paper-edge);border-radius:6px;padding:3px 9px;cursor:pointer;font:inherit;color:var(--ink)">clear</button>';
            var cb = document.getElementById('req-clear-filter');
            if (cb) cb.addEventListener('click', function () { reqTimeFilter = null; banner.remove(); loadRequests({ reset: true }); });
          } else if (banner) { banner.remove(); }
        },
      });
      if (window.APIN && window.APIN.keyOverview)
        window.APIN.keyOverview.activate(PID);
      return;
    }
    // Sparkline + KPI tiles built from /keys/{pid}/usage?minutes=1440 (24h).
    const { body } = await api('/api/account/keys/' +
      encodeURIComponent(PID) + '/usage?minutes=1440');
    if (!body || !body.ok) return;
    const items = (body.data && body.data.items) || [];

    // KPI tiles. Compute current 24h vs prior 24h would require a 48h
    // fetch; for the overview we use absolute current totals and leave
    // delta_pct = null (no comparison row shown). The Usage tab does
    // the full comparison via /api/account/usage/summary.
    let req = 0, err = 0, rl = 0, p95 = 0;
    items.forEach(x => {
      req += x.requests || 0;
      err += x.errors || 0;
      rl += x.rate_limited || 0;
      if (x.latency_p95_ms && x.latency_p95_ms > p95) p95 = x.latency_p95_ms;
    });
    setKpiTile('kt-requests', { value: fmtNum(req), delta: null, prevLabel: 'last 24h' });
    setKpiTile('kt-errors',   { value: fmtNum(err), delta: null, prevLabel: 'last 24h', goodIfDown: true });
    setKpiTile('kt-p95',      { value: p95 ? Math.round(p95) : '·', delta: null, prevLabel: 'last 24h', goodIfDown: true });
    setKpiTile('kt-rate-limited', { value: fmtNum(rl), delta: null, prevLabel: 'last 24h', goodIfDown: true });

    // Sparkline. We bucket the 24h data into ~96 buckets (15-min) to
    // keep the SVG readable; if there are fewer items we just use them.
    drawSparkline(items);

    // Top endpoints from the recent requests log. We don't have a
    // per-endpoint aggregate in /keys/{pid}/usage, so we read the last
    // 200 requests and group them client-side. Cheap + responsive.
    const { body: rb } = await api('/api/account/keys/' +
      encodeURIComponent(PID) + '/requests?limit=200');
    const rows = (rb && rb.data && rb.data.items) || [];
    const agg = new Map();
    for (const it of rows) {
      const key = it.path || '?';
      const a = agg.get(key) || { path: key, requests: 0, errors: 0 };
      a.requests++;
      if ((Number(it.status_code) || 0) >= 500) a.errors++;
      agg.set(key, a);
    }
    const aggSorted = [...agg.values()].sort((a, b) => b.requests - a.requests).slice(0, 8);
    const epTbody = $('ov-ep-tbody');
    if (aggSorted.length === 0) {
      epTbody.innerHTML = '<tr><td colspan="3">' + emptyHtml({
        title: 'no traffic yet',
        sub: 'make a request to this key to see endpoint breakdowns.',
      }) + '</td></tr>';
    } else {
      epTbody.innerHTML = aggSorted.map(a =>
        '<tr>' +
        '<td>' + escHtml(a.path) + '</td>' +
        '<td style="text-align:right">' + a.requests + '</td>' +
        '<td style="text-align:right">' +
          (a.errors ? '<span class="err">' + a.errors + '</span>' : '0') +
        '</td>' +
        '</tr>'
      ).join('');
    }

    // Latest 8 requests for the activity card.
    const recent = rows.slice(0, 8);
    const recentTbody = $('ov-recent-tbody');
    if (recent.length === 0) {
      recentTbody.innerHTML = '<tr><td colspan="5">' + emptyHtml({
        title: 'no requests yet',
        sub: 'the requests tab will fill in once your key sees traffic.',
      }) + '</td></tr>';
    } else {
      recentTbody.innerHTML = recent.map(it =>
        '<tr>' +
        '<td>' + escHtml(fmtAgo(it.timestamp)) + '</td>' +
        '<td><span class="meth ' + methClass(it.method) + '">' + escHtml(it.method || '') + '</span></td>' +
        '<td>' + escHtml(it.path || '') + '</td>' +
        '<td><span class="stat ' + statusBucket(it.status_code) + '">' + (it.status_code || '') + '</span></td>' +
        '<td style="text-align:right">' + (it.latency_ms != null ? it.latency_ms + ' ms' : '·') + '</td>' +
        '</tr>'
      ).join('');
    }
  }

  function emptyHtml(opts) {
    return '<div class="empty"><svg class="illus" aria-hidden="true"><use href="#i-sprout"/></svg>' +
      '<div class="empty-title">' + escHtml(opts.title || 'nothing here yet') + '</div>' +
      '<div class="empty-sub">' + escHtml(opts.sub || '') + '</div>' +
      '</div>';
  }

  // ─── 9.I · Request-detail drawer (per-key page) ──────────────────────
  function openReqd() {
    const d = $('reqd');
    const m = $('reqd-mask');
    if (d) { d.classList.add('show'); d.setAttribute('aria-hidden', 'false'); }
    if (m) m.classList.add('show');
  }
  function closeReqd() {
    const d = $('reqd');
    const m = $('reqd-mask');
    if (d) { d.classList.remove('show'); d.setAttribute('aria-hidden', 'true'); }
    if (m) m.classList.remove('show');
  }
  async function openRequestDetail(rid) {
    // 9.N.T9 · Single source of truth: delegate to the shared full drawer
    // (payload+image, stage waterfall, burst neighbours, endpoint health,
    // export) — identical to the Usage page. Every caller on this page
    // (Requests tab, Traffic container, Overview ribbon) lands here, so this
    // one guard upgrades them all. Falls back to the legacy inline drawer
    // below only if the shared module failed to load.
    if (window.APIN && APIN.requestDrawer && APIN.requestDrawer.open) {
      return APIN.requestDrawer.open(rid);
    }
    openReqd();
    $('reqd-title').textContent = 'Request #' + rid;
    $('reqd-sub').textContent = 'loading…';
    $('reqd-body').innerHTML = '<div class="placeholder">loading&hellip;</div>';
    const { body } = await api('/api/account/usage/request/' + encodeURIComponent(rid));
    if (!body || !body.ok) {
      $('reqd-body').innerHTML = '<div class="placeholder">failed to load</div>';
      return;
    }
    const r = body.data.row || {};
    const inf = body.data.inferred || {};
    const curlSnip = body.data.curl || '';
    const pySnip = body.data.as_python || '';
    const nodeSnip = body.data.as_node || '';

    $('reqd-title').innerHTML = '<span class="reqd-tp">' + escHtml((r.method || '') + ' ' + (r.path || '')) + '</span>'
      + '<span class="reqd-rno">Request #' + escHtml(String(r.id != null ? r.id : rid)) + '</span>';
    $('reqd-sub').textContent = ((window.APIN && APIN.time && APIN.time.localFull) ? APIN.time.localFull(r.timestamp) : (r.timestamp || ''))
      + ' · key: ' + (r.key_name || r.key_public_id || '?');

    const sb = r.status_code || 0;
    const sbCls = statusBucket(sb);
    const statusBar =
      '<div class="reqd-status-bar">' +
        '<span class="meth ' + methClass(r.method) + '">' + escHtml(r.method || '') + '</span>' +
        '<span class="stat ' + sbCls + '">' + sb + '</span>' +
        '<span class="path">' + escHtml(r.path || '') + '</span>' +
      '</div>';

    const latMs = r.latency_ms || 0;
    const latPct = Math.min(100, Math.round((latMs / 3000) * 100));
    const latBar =
      '<div class="reqd-section">' +
        '<h4><svg class="icon"><use href="#i-gauge"/></svg> Latency</h4>' +
        '<div class="reqd-latency-bar">' +
          '<div class="fill" style="width:' + latPct + '%"></div>' +
          '<div class="lbl">' + latMs + ' ms</div>' +
        '</div>' +
      '</div>';

    const sourceCard =
      '<div class="reqd-section"><h4><svg class="icon"><use href="#i-eye"/></svg> Where it came from</h4>' +
      '<div class="reqd-source">' +
        '<div class="source-icon"><svg class="icon"><use href="#' + escHtml(inf.icon || 'i-flask') + '"/></svg></div>' +
        '<div class="source-info">' +
          '<div class="source-name">' + escHtml(inf.label || 'Unknown') + '</div>' +
          '<div class="source-sub">' +
            (inf.via_page ? 'via <code>' + escHtml(inf.via_page) + '</code> · ' : '') +
            (inf.language ? 'language: <b>' + escHtml(inf.language) + '</b> · ' : '') +
            'User-Agent: ' + escHtml((r.ua || '').slice(0, 100)) +
          '</div>' +
        '</div>' +
      '</div></div>';

    const reqDetails =
      '<div class="reqd-section"><h4><svg class="icon"><use href="#i-list-ordered"/></svg> Request</h4>' +
      '<dl class="reqd-row"><dt>Method</dt><dd><code>' + escHtml(r.method || '') + '</code></dd></dl>' +
      '<dl class="reqd-row"><dt>Path</dt><dd><code>' + escHtml(r.path || '') + '</code></dd></dl>' +
      '<dl class="reqd-row"><dt>IP</dt><dd><code>' + escHtml(r.ip || '·') + '</code></dd></dl>' +
      '<dl class="reqd-row"><dt>Bytes in</dt><dd>' + (r.bytes_in != null ? fmtBytes(r.bytes_in) : '·') + '</dd></dl>' +
      '<dl class="reqd-row"><dt>Via</dt><dd>' + escHtml(r.via || '·') + '</dd></dl>' +
      '</div>';

    const respDetails =
      '<div class="reqd-section"><h4><svg class="icon"><use href="#i-arrow-down-right"/></svg> Response</h4>' +
      '<dl class="reqd-row"><dt>Status</dt><dd><span class="stat ' + sbCls + '">' + sb + '</span></dd></dl>' +
      (r.error_code ?
        '<dl class="reqd-row"><dt>Error</dt><dd><span class="err">' + escHtml(r.error_code) + '</span></dd></dl>' : '') +
      '<dl class="reqd-row"><dt>Latency</dt><dd>' + latMs + ' ms</dd></dl>' +
      '<dl class="reqd-row"><dt>Bytes out</dt><dd>' + (r.bytes_out != null ? fmtBytes(r.bytes_out) : '·') + '</dd></dl>' +
      '</div>';

    const codeCard =
      '<div class="reqd-section"><h4><svg class="icon"><use href="#i-flask"/></svg> Reproduce</h4>' +
      '<div class="reqd-code-tabs"><button data-snip="curl" aria-pressed="true">curl</button>' +
        '<button data-snip="python" aria-pressed="false">python</button>' +
        '<button data-snip="node" aria-pressed="false">node</button>' +
      '</div>' +
      '<div class="reqd-code" id="reqd-code-body">' +
        '<button class="copy-btn" id="reqd-copy">copy</button>' +
        '<code>' + escHtml(curlSnip) + '</code>' +
      '</div></div>';

    function fmtBytes(n) {
      if (n == null) return '·';
      if (n < 1024) return n + ' B';
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' kB';
      return (n / 1024 / 1024).toFixed(2) + ' MB';
    }

    $('reqd-body').innerHTML =
      statusBar + latBar + sourceCard + reqDetails + respDetails + codeCard;

    const snips = { curl: curlSnip, python: pySnip, node: nodeSnip };
    $('reqd-body').querySelectorAll('.reqd-code-tabs button').forEach(btn => {
      btn.addEventListener('click', () => {
        $('reqd-body').querySelectorAll('.reqd-code-tabs button')
          .forEach(b => b.setAttribute('aria-pressed', 'false'));
        btn.setAttribute('aria-pressed', 'true');
        const codeEl = $('reqd-code-body').querySelector('code');
        if (codeEl) codeEl.textContent = snips[btn.dataset.snip] || '';
      });
    });
    const copyBtn = $('reqd-copy');
    if (copyBtn) {
      copyBtn.addEventListener('click', async () => {
        const codeEl = $('reqd-code-body').querySelector('code');
        if (!codeEl) return;
        try {
          await navigator.clipboard.writeText(codeEl.textContent || '');
          copyBtn.classList.add('copied');
          copyBtn.textContent = 'copied!';
          setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.textContent = 'copy';
          }, 1500);
        } catch (e) {}
      });
    }
  }
  // Wire close + Escape
  document.addEventListener('DOMContentLoaded', () => {
    if ($('reqd-close')) $('reqd-close').addEventListener('click', closeReqd);
    if ($('reqd-mask')) $('reqd-mask').addEventListener('click', closeReqd);
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeReqd(); });

  // Also wire close handlers in case DOMContentLoaded already fired
  setTimeout(() => {
    if ($('reqd-close') && !$('reqd-close')._wired) {
      $('reqd-close')._wired = true;
      $('reqd-close').addEventListener('click', closeReqd);
    }
    if ($('reqd-mask') && !$('reqd-mask')._wired) {
      $('reqd-mask')._wired = true;
      $('reqd-mask').addEventListener('click', closeReqd);
    }
  }, 100);

  // ─── Sparkline (24h) ─────────────────────────────────────────────────
  function drawSparkline(items) {
    const host = $('spark-wrap');
    if (!items || items.length === 0) {
      host.innerHTML = emptyHtml({
        title: 'no traffic in the last 24 hours',
        sub: 'requests on this key will show up here within a minute.',
      });
      return;
    }
    // Pack into max ~96 buckets so the SVG stays readable. Aggregate
    // by stride if too many points.
    let pts = items;
    if (items.length > 96) {
      const stride = Math.ceil(items.length / 96);
      pts = [];
      for (let i = 0; i < items.length; i += stride) {
        const slice = items.slice(i, i + stride);
        const tx = slice[Math.floor(slice.length / 2)].minute_ts;
        const req = slice.reduce((a, x) => a + (x.requests || 0), 0);
        pts.push({ minute_ts: tx, requests: req });
      }
    }
    const w = 600, h = 140, padL = 36, padR = 8, padT = 10, padB = 22;
    const innerW = w - padL - padR;
    const innerH = h - padT - padB;
    const max = Math.max(1, ...pts.map(x => x.requests || 0));
    const step = pts.length > 1 ? innerW / (pts.length - 1) : 0;
    let path = '', area = `M${padL},${padT + innerH} `;
    const coords = pts.map((x, i) => {
      const px = padL + i * step;
      const py = padT + innerH - ((x.requests || 0) / max) * innerH;
      return { x: px, y: py, raw: x };
    });
    coords.forEach((c, i) => {
      path += (i === 0 ? 'M' : ' L') + c.x.toFixed(1) + ',' + c.y.toFixed(1);
      area += 'L' + c.x.toFixed(1) + ',' + c.y.toFixed(1) + ' ';
    });
    area += `L${padL + innerW},${padT + innerH} Z`;
    // Y-axis labels: max, 0
    const yMaxLbl = `<text class="spark-axis" x="${padL - 6}" y="${padT + 4}" text-anchor="end">${fmtNum(max)}</text>`;
    const yMidLbl = `<text class="spark-axis" x="${padL - 6}" y="${padT + innerH / 2 + 4}" text-anchor="end">${fmtNum(max / 2)}</text>`;
    const yZeroLbl = `<text class="spark-axis" x="${padL - 6}" y="${padT + innerH + 4}" text-anchor="end">0</text>`;
    // X-axis: first + middle + last bucket times
    const tFmt = iso => {
      const d = new Date(iso.replace(' ', 'T') + 'Z');
      return isNaN(d) ? '' : d.toISOString().slice(11, 16);
    };
    const xFirst = `<text class="spark-axis" x="${padL}" y="${h - 6}" text-anchor="start">${tFmt(pts[0].minute_ts)}</text>`;
    const xLast = `<text class="spark-axis" x="${padL + innerW}" y="${h - 6}" text-anchor="end">${tFmt(pts[pts.length - 1].minute_ts)}</text>`;

    host.innerHTML =
      `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">` +
      `<line class="spark-grid" x1="${padL}" y1="${padT}" x2="${padL + innerW}" y2="${padT}"/>` +
      `<line class="spark-grid" x1="${padL}" y1="${padT + innerH / 2}" x2="${padL + innerW}" y2="${padT + innerH / 2}"/>` +
      `<line class="spark-baseline" x1="${padL}" y1="${padT + innerH}" x2="${padL + innerW}" y2="${padT + innerH}"/>` +
      yMaxLbl + yMidLbl + yZeroLbl +
      `<path d="${area}" class="spark-area"/>` +
      `<path d="${path}" class="spark-line"/>` +
      coords.map(c => `<circle class="spark-marker" cx="${c.x}" cy="${c.y}" r="0" data-t="${escHtml(c.raw.minute_ts)}" data-v="${c.raw.requests || 0}"/>`).join('') +
      `<line class="spark-cursor" id="spark-cursor" x1="0" y1="${padT}" x2="0" y2="${padT + innerH}"/>` +
      xFirst + xLast +
      `</svg>` +
      `<div class="spark-tip" id="spark-tip"></div>`;

    // Hover interactivity. We don't bind on each marker (would be N=96
    // handlers) — instead one mousemove on the host that snaps to the
    // nearest x.
    const svg = host.querySelector('svg');
    const cursor = host.querySelector('#spark-cursor');
    const tip = host.querySelector('#spark-tip');
    host.addEventListener('mousemove', e => {
      const rect = svg.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width * w;
      // Snap to closest coord
      let nearest = coords[0], bestDx = Infinity;
      for (const c of coords) {
        const dx = Math.abs(c.x - x);
        if (dx < bestDx) { bestDx = dx; nearest = c; }
      }
      cursor.setAttribute('x1', nearest.x);
      cursor.setAttribute('x2', nearest.x);
      // Highlight the marker
      svg.querySelectorAll('.spark-marker').forEach(m => m.setAttribute('r', '0'));
      svg.querySelectorAll('.spark-marker').forEach(m => {
        if (Number(m.getAttribute('cx')) === nearest.x) {
          m.setAttribute('r', '4');
        }
      });
      // Tip
      tip.classList.add('show');
      tip.innerHTML =
        '<div style="color:var(--ink-soft)">' + escHtml(nearest.raw.minute_ts) + 'Z</div>' +
        '<div style="font-weight:600;color:var(--ink)">' + fmtNum(nearest.raw.requests || 0) + ' req</div>';
      // Position the tooltip (clamped within the host)
      const hostRect = host.getBoundingClientRect();
      const cx = (nearest.x / w) * hostRect.width;
      tip.style.left = Math.max(0, Math.min(hostRect.width - tip.offsetWidth - 4, cx + 8)) + 'px';
      tip.style.top = (padT - 4) + 'px';
    });
    host.addEventListener('mouseleave', () => {
      tip.classList.remove('show');
      cursor.setAttribute('x1', 0);
      cursor.setAttribute('x2', 0);
      svg.querySelectorAll('.spark-marker').forEach(m => m.setAttribute('r', '0'));
    });
  }

  // ─── 9.I · Per-key Usage tab (full chart embed scoped to this key) ────
  const pkState = { range: '24h', mode: 'total', logY: false };

  function pkBuildQS(extras) {
    const qs = new URLSearchParams({ range: pkState.range, key_id: PID });
    if (extras) for (const k in extras) qs.set(k, extras[k]);
    return qs;
  }

  async function pkLoadSummary() {
    const { body } = await api('/api/account/usage/summary?' + pkBuildQS());
    if (!body || !body.ok) return;
    const k = body.data.kpis;
    const setEl = (sel, val) => {
      const el = document.querySelector(sel);
      if (!el) return;
      if (el.classList && el.classList.contains('apin-odometer') &&
          window.APIN && APIN.odometer) {
        APIN.odometer.roll(el, val);
      } else {
        el.textContent = String(val);
      }
    };
    setEl('[data-pk-k="requests"] [data-num]', fmtNum(k.requests.current));
    setEl('[data-pk-k="errors"] [data-num]', fmtNum(k.errors.current));
    setEl('[data-pk-k="p95"] [data-num]',
      k.latency_p95_ms.current != null ? Math.round(k.latency_p95_ms.current) : '·');
    const bo = document.querySelector('[data-pk-k="bytes_out"] [data-num]');
    if (bo) bo.textContent = fmtBytes(k.bytes_out.current);

    // Sub-line breakdowns
    document.querySelectorAll('[data-pk-k="errors"] [data-breakdown]').forEach(el =>
      el.innerHTML = '5xx <b>' + (k.errors_5xx ? k.errors_5xx.current : 0) +
        '</b> · 4xx <b>' + (k.errors_4xx ? k.errors_4xx.current : 0) + '</b>');
    document.querySelectorAll('[data-pk-k="p95"] [data-breakdown]').forEach(el =>
      el.innerHTML = 'p50 <b>' + (k.latency_p50_ms.current != null ? Math.round(k.latency_p50_ms.current) : '·') +
        'ms</b> · p99 <b>' + (k.latency_p99_ms.current != null ? Math.round(k.latency_p99_ms.current) : '·') + 'ms</b>');

    // Plain delta sub-lines
    const fmtDelta = (d) => {
      if (d == null) return 'no prev data';
      if (Math.abs(d) < 0.05) return 'no change';
      return (d > 0 ? '+' : '') + d.toFixed(1) + '% vs prev';
    };
    document.querySelectorAll('[data-pk-k="requests"] [data-delta]').forEach(el =>
      el.textContent = fmtDelta(k.requests.delta_pct));
    document.querySelectorAll('[data-pk-k="bytes_out"] [data-delta]').forEach(el =>
      el.textContent = fmtDelta(k.bytes_out.delta_pct));
  }

  async function pkLoadTimeseries() {
    const qs = pkBuildQS({ mode: pkState.mode });
    const host = $('pk-ts-host');
    const titleMap = {
      total: 'Requests', by_status: 'Requests by status',
      by_endpoint: 'Requests by endpoint', latency: 'Latency (ms)',
      errors: 'Errors', bytes: 'Bytes in / out',
    };
    if ($('pk-ts-title'))
      $('pk-ts-title').textContent =
        (titleMap[pkState.mode] || 'Requests') + ' · ' + pkState.range;
    const { body } = await api('/api/account/usage/timeseries?' + qs);
    if (!body || !body.ok) {
      host.innerHTML = '<div class="placeholder">failed to load</div>';
      return;
    }
    const chartMode = (pkState.mode === 'by_status' || pkState.mode === 'by_endpoint')
      ? 'stacked' : 'line';
    APIN.charts.timeseries(host, body.data, {
      mode: chartMode, logY: pkState.logY, height: 300, showLegend: true,
      onClickBucket: b => pkOpenMinuteDrawer(b.t),
    });
  }

  async function pkLoadTop(dim, hostId, auxId, opts) {
    opts = opts || {};
    const qs = pkBuildQS({ dim, limit: 8 });
    const { body } = await api('/api/account/usage/top?' + qs);
    if (!body || !body.ok) return;
    const items = body.data.items || [];
    if (auxId && $(auxId)) $(auxId).textContent =
      items.length + ' / ' + fmtNum(body.data.total_for_pct);
    APIN.charts.topBar($(hostId), items, {
      color_token: opts.color_token || 'series-1',
      onClickItem: it => {
        if (dim === 'endpoints' && it.path) {
          pkOpenEndpointDrawer(it.path);
        }
      },
    });
  }

  async function pkLoadDonut() {
    const granForRange = {
      '15m': '1m', '1h': '5m', '6h': '15m', '24h': '1h', '7d': '6h', '30d': '1d',
    };
    const qs = pkBuildQS({
      mode: 'by_status', granularity: granForRange[pkState.range] || '1h',
    });
    const { body } = await api('/api/account/usage/timeseries?' + qs);
    if (!body || !body.ok) return;
    const totals = { '2xx': 0, '3xx': 0, '4xx': 0, '429': 0, '5xx': 0 };
    for (const b of (body.data.buckets || []))
      for (const k of Object.keys(totals)) totals[k] += (b.values && b.values[k]) || 0;
    const items = [
      { label: '2xx', value: totals['2xx'], color_token: 'ok' },
      { label: '3xx', value: totals['3xx'], color_token: 'info' },
      { label: '4xx', value: totals['4xx'], color_token: 'warn' },
      { label: '429', value: totals['429'], color_token: 'amber' },
      { label: '5xx', value: totals['5xx'], color_token: 'danger' },
    ].filter(x => x.value > 0);
    const total = items.reduce((a, x) => a + x.value, 0);
    if ($('pk-donut-aux')) $('pk-donut-aux').textContent = fmtNum(total) + ' requests';
    APIN.charts.donut($('pk-donut-host'), items, {
      size: 180, innerRatio: 0.62, totalLabel: 'reqs',
    });
  }

  async function pkLoadHist() {
    const qs = pkBuildQS({ limit: 200 });
    const { body } = await api('/api/account/usage/requests?' + qs);
    if (!body || !body.ok) {
      $('pk-hist-host').innerHTML = '<div class="placeholder">failed</div>';
      return;
    }
    const lats = (body.data.items || []).map(x => Number(x.latency_ms)).filter(x => x > 0);
    if (lats.length === 0) {
      $('pk-hist-host').innerHTML = '<div class="placeholder">no latency data</div>';
      return;
    }
    const edges = [0, 50, 100, 200, 500, 1000, 2000, 5000, Infinity];
    const labels = ['<50', '50-100', '100-200', '200-500', '500-1k', '1-2k', '2-5k', '5k+'];
    const buckets = labels.map((l, i) => ({
      label: l,
      value: lats.filter(v => v >= edges[i] && v < edges[i + 1]).length,
      hint: edges[i] + '-' + (edges[i + 1] === Infinity ? '∞' : edges[i + 1]) + ' ms',
      color_token: i >= 6 ? 'danger' : (i >= 4 ? 'warn' : (i >= 2 ? 'info' : 'ok')),
    }));
    APIN.charts.histogram($('pk-hist-host'), buckets, { height: 200 });
    if ($('pk-hist-aux')) $('pk-hist-aux').textContent = lats.length + ' samples';
  }

  function pkOpenMinuteDrawer(minute_ts) {
    // Reuses the account-wide drawer pattern by going to the Usage page
    // with the minute pre-applied. Lightweight handoff.
    window.open('/account/api/usage#range=' + pkState.range + '&key_id=' + encodeURIComponent(PID), '_blank');
  }
  function pkOpenEndpointDrawer(path) {
    window.open('/account/api/usage#range=' + pkState.range + '&key_id=' + encodeURIComponent(PID) + '&endpoint=' + encodeURIComponent(path), '_blank');
  }

  let _pkUsageBootstrapped = false;
  async function loadUsageTab() {
    if (!_pkUsageBootstrapped) {
      _pkUsageBootstrapped = true;
      // Bind range strip
      document.querySelectorAll('#pk-range-strip button').forEach(b => {
        b.addEventListener('click', () => {
          document.querySelectorAll('#pk-range-strip button').forEach(x =>
            x.removeAttribute('aria-pressed'));
          b.setAttribute('aria-pressed', 'true');
          pkState.range = b.dataset.range;
          pkRefreshAll();
        });
      });
      // Bind mode switcher
      document.querySelectorAll('#pk-ts-mode button').forEach(b => {
        b.addEventListener('click', () => {
          document.querySelectorAll('#pk-ts-mode button').forEach(x =>
            x.removeAttribute('aria-pressed'));
          b.setAttribute('aria-pressed', 'true');
          pkState.mode = b.dataset.mode;
          pkLoadTimeseries();
        });
      });
    }
    pkRefreshAll();
  }

  async function pkRefreshAll() {
    await Promise.allSettled([
      pkLoadSummary(),
      pkLoadTimeseries(),
      pkLoadTop('endpoints', 'pk-top-ep-host', 'pk-top-ep-aux', { color_token: 'series-1' }),
      pkLoadDonut(),
      pkLoadHist(),
    ]);
  }

  // ─── Requests tab (paginated raw log) ─────────────────────────────────
  let reqLastQuery = null;
  let reqTimeFilter = null;   // {since, until} set by Traffic-tab drills
  function reqQueryString({ append } = {}) {
    const p = new URLSearchParams();
    p.set('limit', '50');
    if (reqTimeFilter) { p.set('since', reqTimeFilter.since); p.set('until', reqTimeFilter.until); }
    const q = $('req-q') ? $('req-q').value.trim() : '';
    const m = $('req-method') ? $('req-method').value : '';
    const s = $('req-status') ? $('req-status').value : '';
    if (q) p.set('endpoint', q);
    // Note: per-key /requests endpoint accepts these filter params after
    // the Phase 9.A `list_key_requests` extension. The account-wide path
    // uses the same params under /api/account/usage/requests.
    if (m) p.set('method', m);
    if (s) p.set('status', s);
    if (append && reqCursor) p.set('cursor', String(reqCursor));
    return p.toString();
  }

  async function loadRequests(opts) {
    opts = opts || {};
    const tbody = $('req-tbody');
    const pager = $('req-pager');
    if (opts.reset) {
      reqCursor = null;
      tbody.innerHTML = '<tr><td colspan="9" class="placeholder">loading&hellip;</td></tr>';
    }
    const qs = reqQueryString({ append: !opts.reset && reqCursor != null });
    reqLastQuery = qs;
    const url = '/api/account/keys/' + encodeURIComponent(PID) + '/requests?' + qs;
    const { body } = await api(url);
    // Guard: if the user filter changed while waiting, drop this response.
    if (qs !== reqLastQuery) return;
    if (!body || !body.ok) {
      tbody.innerHTML = '<tr><td colspan="9" class="placeholder">failed to load — check console</td></tr>';
      return;
    }
    const items = (body.data && body.data.items) || [];
    if (opts.reset) {
      if (items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9">' + emptyHtml({
          title: 'no requests match these filters',
          sub: 'try a wider status range or clear the path filter.',
        }) + '</td></tr>';
      } else {
        tbody.innerHTML = renderReqRows(items);
      }
    } else {
      // Append mode: strip placeholder if it's still there.
      tbody.querySelectorAll('.placeholder').forEach(p =>
        p.parentElement.parentElement.removeChild(p.parentElement));
      tbody.insertAdjacentHTML('beforeend', renderReqRows(items));
    }
    // Pagination cursor — last row's id, since the list is DESC.
    if (items.length > 0) {
      reqCursor = items[items.length - 1].id;
      $('tab-count-requests').hidden = false;
      $('tab-count-requests').textContent = tbody.querySelectorAll('tr').length;
    }
    if (items.length === 50) {
      pager.hidden = false;
    } else {
      pager.hidden = true;
    }
    $('req-aux').textContent = tbody.querySelectorAll('tr').length + ' rows';
  }
  function renderReqRows(items) {
    return items.map(it =>
      '<tr class="row-clickable" data-rid="' + escHtml(it.id) + '" style="cursor:pointer">' +
      '<td title="' + escHtml(it.timestamp) + '">' + escHtml(fmtAgo(it.timestamp)) + '</td>' +
      '<td><span class="meth ' + methClass(it.method) + '">' + escHtml(it.method || '') + '</span></td>' +
      '<td>' + escHtml(it.path || '') + '</td>' +
      '<td><span class="stat ' + statusBucket(it.status_code) + '">' + (it.status_code || '') + '</span></td>' +
      '<td>' + (it.error_code ? '<span class="err">' + escHtml(it.error_code) + '</span>' : '') + '</td>' +
      '<td style="text-align:right">' + (it.latency_ms != null ? it.latency_ms + ' ms' : '·') + '</td>' +
      '<td style="text-align:right">' + (it.bytes_in != null ? fmtBytes(it.bytes_in) : '·') + '</td>' +
      '<td style="text-align:right">' + (it.bytes_out != null ? fmtBytes(it.bytes_out) : '·') + '</td>' +
      '<td>' + escHtml(it.ip || '·') + '</td>' +
      '</tr>'
    ).join('');
  }
  // Delegated click handler for request rows (works for both initial and
  // appended rows since events bubble up to tbody).
  if ($('req-tbody')) {
    $('req-tbody').addEventListener('click', (ev) => {
      const tr = ev.target.closest('tr.row-clickable');
      if (!tr) return;
      const rid = tr.getAttribute('data-rid');
      if (rid) openRequestDetail(rid);
    });
  }
  // Filter event wiring (debounced search)
  let reqDebounce = null;
  document.addEventListener('DOMContentLoaded', () => {
    if ($('req-q')) {
      $('req-q').addEventListener('input', () => {
        clearTimeout(reqDebounce);
        reqDebounce = setTimeout(() => loadRequests({ reset: true }), 220);
      });
    }
    if ($('req-method')) {
      $('req-method').addEventListener('change', () => loadRequests({ reset: true }));
    }
    if ($('req-status')) {
      $('req-status').addEventListener('change', () => loadRequests({ reset: true }));
    }
    if ($('req-refresh')) {
      $('req-refresh').addEventListener('click', () => loadRequests({ reset: true }));
    }
    if ($('req-more')) {
      $('req-more').addEventListener('click', () => loadRequests({ reset: false }));
    }
  });

  // ─── Audit tab ───────────────────────────────────────────────────────
  async function loadAudit() {
    const host = $('audit-list');
    if (!host) return;
    const { body } = await api('/api/account/keys/' +
      encodeURIComponent(PID) + '/audit?limit=100');
    if (!body || !body.ok) {
      host.innerHTML = '<div class="placeholder">failed to load audit log</div>';
      return;
    }
    const items = (body.data && body.data.items) || [];
    if (items.length === 0) {
      host.innerHTML = emptyHtml({
        title: 'no audit events yet',
        sub: 'lifecycle changes (rotate / disable / edit) will be hash-chained here.',
      });
      return;
    }
    $('tab-count-audit').hidden = false;
    $('tab-count-audit').textContent = items.length;
    host.innerHTML = items.map(it => {
      const actionCls = 'act-' + (it.action || 'other').replace(/[^a-z_]/g, '');
      let details = '';
      if (it.after || it.before || it.details) {
        const blob = {};
        if (it.before) blob.before = it.before;
        if (it.after) blob.after = it.after;
        if (it.details) blob.details = it.details;
        details = '<div class="audit-details">' + escHtml(JSON.stringify(blob, null, 2)) + '</div>';
      }
      return '<div class="audit-row">' +
        '<span class="audit-pill ' + actionCls + '">' + escHtml(it.action || '?') + '</span>' +
        '<div>' +
          '<div class="audit-meta">' +
            'prev <span class="hash">' + escHtml(it.prev_hash || '·') + '</span> ' +
            '&rarr; row <span class="hash">' + escHtml(it.row_hash || '·') + '</span>' +
            (it.actor_user_id ? ' &middot; actor #' + escHtml(it.actor_user_id) : '') +
          '</div>' +
          details +
        '</div>' +
        '<span class="audit-time" title="' + escHtml(it.timestamp || '') + '">' +
          fmtAgo(it.timestamp) + '</span>' +
      '</div>';
    }).join('');
  }

  // ─── 9.O.4 · Settings tab — control surface with inline section editing ──
  const SCOPE_GROUPS = [
    { key: 'predictions', label: 'Predictions', scopes: ['predict:read', 'predict:write'] },
    { key: 'reports', label: 'Reports', scopes: ['reports:read', 'reports:write'] },
    { key: 'models', label: 'Models', scopes: ['models:read'] },
    { key: 'feedback', label: 'Feedback', scopes: ['feedback:write'] },
    { key: 'account', label: 'Usage & account', scopes: ['usage:read', 'account:read'] },
  ];
  const SCOPE_DESC = {
    'predict:read': 'read inference results', 'predict:write': 'run new predictions',
    'reports:read': 'read reports', 'reports:write': 'create & export reports',
    'models:read': 'read model cards', 'feedback:write': 'submit corrections',
    'usage:read': 'read analytics & usage', 'account:read': 'read key & account info',
  };
  const KNOWN_SCOPES = SCOPE_GROUPS.reduce((a, g) => a.concat(g.scopes), []);
  let _scopeView = 'grouped';   // 'grouped' | 'all'

  let _setSection = 'identity';

  function settingsCSS() {
    if (document.getElementById('oset-css')) return;
    const s = document.createElement('style'); s.id = 'oset-css';
    s.textContent = `
#settings-root{max-width:1080px}
.oset-wrap{display:grid;grid-template-columns:236px 1fr;background:var(--paper,#fbf9f3);border:1px solid var(--paper-edge,#c7bca9);border-radius:16px;overflow:hidden;box-shadow:0 10px 30px rgba(20,16,12,.05)}
.oset-rail{display:flex;flex-direction:column;gap:2px;padding:14px 12px;background:var(--paper-soft,#f4efe6);border-right:1px solid var(--paper-edge,#e3d9c0)}
.oset-nav{display:flex;align-items:center;gap:11px;width:100%;text-align:left;border:0;background:transparent;border-radius:10px;padding:9px 11px;cursor:pointer;color:var(--ink-soft,#5a5246);transition:background .15s ease,color .15s ease}
.oset-nav:hover{background:rgba(0,0,0,.045)}
.oset-nav.on{background:var(--paper,#fbf9f3);box-shadow:0 1px 5px rgba(20,16,12,.09);color:var(--ink,#1a1612)}
.oset-nav .ic{width:17px;height:17px;flex:none;color:var(--ink-mute,#8b8273)}.oset-nav.on .ic{color:var(--green-deep,#1f5b32)}
.oset-nav .ic svg{width:17px;height:17px;fill:none;stroke:currentColor;stroke-width:1.7}
.oset-nav .lab{display:flex;flex-direction:column;line-height:1.25;min-width:0}
.oset-nav .lab b{font:600 12.5px 'JetBrains Mono',monospace}
.oset-nav .lab span{font:10px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.oset-nav.danger.on .ic,.oset-nav.danger:hover .ic{color:#b3402f}
.oset-sep{height:1px;background:var(--paper-edge,#e3d9c0);margin:8px 6px}
.oset-pane{padding:22px 26px;min-height:360px}
.oset-h{display:flex;align-items:center;gap:10px;margin-bottom:18px}
.oset-h .ic{width:18px;height:18px;color:var(--green-deep,#1f5b32)}.oset-h .ic svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.7}
.oset-h h3{flex:1;margin:0;font:600 17px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.orow{display:grid;grid-template-columns:150px 1fr;gap:16px;padding:11px 0;align-items:center}
.orow+.orow{border-top:1px solid var(--paper-edge,#efe6cf)}
.orow-lbl{font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273);text-transform:uppercase;letter-spacing:.05em}
.orow-val{font:13px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);min-width:0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.orow-val code{background:var(--paper-deep,#e9e2d1);border-radius:6px;padding:2px 8px;font-size:12px}
.oset-foot{display:flex;justify-content:flex-end;gap:9px;margin-top:20px;padding-top:16px;border-top:1px solid var(--paper-edge,#e3d9c0)}
.oset-err{display:none;margin:14px 0 0;background:#f6e3dd;border:1px solid #d9a99a;color:#92301f;border-radius:9px;padding:9px 12px;font:11.5px 'JetBrains Mono',monospace}
.oset-err.show{display:block}
.ofield{margin-bottom:15px}.ofield label{display:block;font:600 10.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5246);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.ohint{font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273);margin-top:5px;line-height:1.5}
.opreset{display:inline-flex;gap:6px;margin-left:8px}
.opreset button{border:1.2px solid var(--paper-edge,#c7bca9);background:var(--paper,#fbf9f3);border-radius:7px;padding:4px 10px;font:10.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5246);cursor:pointer;transition:background .15s,color .15s,border-color .15s}
.opreset button:hover{background:#fff;border-color:var(--ink-mute,#8b8273)}
.opreset button.on{background:var(--ink,#1a1612);color:var(--paper,#fbf9f3);border-color:var(--ink,#1a1612)}
.oscope-toggle{display:inline-flex;border:1.2px solid var(--paper-edge,#c7bca9);border-radius:8px;overflow:hidden;margin-left:auto}
.oscope-toggle button{border:0;background:var(--paper,#fbf9f3);color:var(--ink-soft,#5a5246);font:600 10.5px 'JetBrains Mono',monospace;padding:5px 11px;cursor:pointer;transition:background .15s}
.oscope-toggle button[aria-pressed=true]{background:var(--ink,#1a1612);color:var(--paper,#fbf9f3)}
.oscl{display:flex;flex-direction:column;gap:15px}
.oscl-cat{display:flex;align-items:baseline;gap:9px;margin-bottom:7px}
.oscl-cat b{font:600 12.5px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.oscl-cat span{font:10px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273)}
.oscl-row{display:flex;align-items:center;gap:10px;padding:4px 0 4px 4px;font:12px 'JetBrains Mono',monospace}
.oscl-row .tick{width:15px;height:15px;flex:none;color:var(--green-deep,#1f5b32)}.oscl-row .tick svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:2.4}
.oscl-row code{color:var(--ink,#1a1612);background:var(--paper-deep,#e9e2d1);border-radius:5px;padding:1px 7px}
.oscl-row .d{color:var(--ink-mute,#8b8273)}
.oscl-none{font:italic 12px Fraunces,Georgia,serif;color:var(--ink-mute,#8b8273)}
.ogroup-badge{display:inline-flex;align-items:center;gap:9px;background:var(--paper-soft,#f4efe6);border:1px solid var(--paper-edge,#e3d9c0);border-radius:10px;padding:7px 12px;margin-bottom:14px;font:12px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5246)}
.ogroup-badge .ic{width:15px;height:15px;color:var(--green-deep,#1f5b32)}.ogroup-badge .ic svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:1.8}
.ogroup-badge b{color:var(--ink,#1a1612)}
.orole{font:600 9.5px 'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.04em;border-radius:20px;padding:2px 8px}
.orole-locked{background:var(--paper-deep,#e9e2d1);color:var(--ink-mute,#8b8273)}
.orole-special{background:var(--ochre-soft,#fbe3c2);color:var(--ochre-deep,#7a4e0a)}
.ogrp{border:1.2px solid var(--paper-edge,#e3d9c0);border-radius:11px;padding:11px 13px;margin-bottom:10px;background:var(--paper-soft,#f4efe6)}
.ogrp-h{display:flex;align-items:center;gap:10px;font:600 12.5px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.ogrp-h label{display:flex;align-items:center;gap:9px;cursor:pointer}
.ogrp-h .sub{color:var(--ink-mute,#8b8273);font-weight:400;font-size:10.5px;margin-left:auto}
.oscope-row{display:flex;align-items:center;gap:10px;padding:5px 0 5px 28px;font:11.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5246)}
.oscope-row label{display:flex;align-items:center;gap:9px;cursor:pointer}
.oscope-row code{color:var(--ink,#1a1612)}
.oqbar-wrap{display:flex;flex-direction:column;gap:6px;width:100%;max-width:340px}
.oqbar-top{display:flex;justify-content:space-between;font:11px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5446)}
.oqbar{height:8px;background:var(--paper-deep,#e9e2d1);border-radius:5px;overflow:hidden}
.oqbar-fill{height:100%;border-radius:5px;background:linear-gradient(90deg,var(--green,#2f6f3e),var(--ochre,#b87d1e));transition:width .6s cubic-bezier(.2,.8,.2,1)}
.oqbar-fill.hot{background:linear-gradient(90deg,var(--ochre,#b87d1e),#b3402f)}
.opost{display:flex;align-items:flex-start;gap:9px;padding:9px 0;font:11.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5446);border-top:1px solid var(--paper-edge,#efe6cf)}
.opost:first-child{border-top:0}.opost .dot{width:8px;height:8px;border-radius:50%;margin-top:4px;flex:none}
.opost.warn .dot{background:var(--ochre,#b87d1e)}.opost.ok .dot{background:var(--green,#2f6f3e)}
.opost a{color:var(--green-deep,#1f5b32);cursor:pointer;text-decoration:underline;text-underline-offset:2px}
.opost a:hover{color:var(--ink,#1a1612)}
.odanger-row{display:flex;align-items:center;gap:14px;padding:13px 0;border-top:1px solid var(--paper-edge,#efe6cf)}
.odanger-row:first-child{border-top:0}.odanger-row .t{flex:1}.odanger-row .t b{display:block;font:600 12.5px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}.odanger-row .t span{font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273)}
.oin-mini{display:inline-flex;gap:8px;align-items:center;flex-wrap:wrap}
.oqedit{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.oqedit .apin-input{max-width:140px}.oqedit .apin-select{max-width:140px}
@media(max-width:760px){.oset-wrap{grid-template-columns:1fr}.oset-rail{flex-direction:row;overflow-x:auto;border-right:0;border-bottom:1px solid var(--paper-edge,#e3d9c0)}.oset-nav .lab span{display:none}.orow{grid-template-columns:1fr;gap:5px}}
`;
    document.head.appendChild(s);
  }

  function keyUrl() { return '/api/account/keys/' + encodeURIComponent(PID); }

  // a generic editable section: read view + (optional) edit view + save(PATCH)
  function osection(opts) {
    const wrap = document.createElement('div');
    wrap.className = 'oset' + (opts.danger ? ' danger' : '');
    let editing = false;
    function paint() {
      wrap.innerHTML = '';
      const head = document.createElement('div'); head.className = 'oset-h';
      head.innerHTML = (opts.icon ? '<span class="ic"><svg viewBox="0 0 24 24"><use href="#' + opts.icon + '"/></svg></span>' : '') +
        '<h3>' + escHtml(opts.title) + '</h3>';
      if (opts.editable && !editing) {
        const eb = document.createElement('button'); eb.className = 'apin-btn ghost oset-edit'; eb.textContent = 'Edit';
        eb.addEventListener('click', () => { editing = true; paint(); });
        head.appendChild(eb);
      }
      wrap.appendChild(head);
      const body = document.createElement('div'); body.className = 'oset-body'; wrap.appendChild(body);
      const err = document.createElement('div'); err.className = 'oset-err';
      if (editing && opts.edit) {
        opts.edit(body);
        wrap.appendChild(err);
        const foot = document.createElement('div'); foot.className = 'oset-foot';
        const cancel = document.createElement('button'); cancel.className = 'apin-btn ghost'; cancel.textContent = 'Cancel';
        cancel.addEventListener('click', () => { editing = false; paint(); });
        const save = document.createElement('button'); save.className = 'apin-btn primary'; save.textContent = 'Save';
        save.addEventListener('click', async () => {
          err.classList.remove('show');
          try {
            const payload = opts.collect(body);
            if (payload && Object.keys(payload).length) {
              save.disabled = true; save.textContent = 'Saving…';
              const { body: rb } = await apiWrite(keyUrl(), 'PATCH', payload);
              if (!rb.ok) throw new Error(writeErr(rb));
            }
            editing = false;
            await refreshMasthead();
            renderSettings();
          } catch (e) {
            save.disabled = false; save.textContent = 'Save';
            err.textContent = (e && e.message) || 'Save failed.'; err.classList.add('show');
          }
        });
        foot.appendChild(cancel); foot.appendChild(save); wrap.appendChild(foot);
      } else {
        opts.read(body);
      }
    }
    paint();
    return wrap;
  }

  function orow(lbl, valHtml, empty) {
    const isEmpty = (valHtml == null || valHtml === '');
    return '<div class="orow"><div class="orow-lbl">' + escHtml(lbl) + '</div>' +
      '<div class="orow-val' + (isEmpty ? ' empty' : '') + '">' +
      (isEmpty ? '<span class="apin-ghost">' + escHtml(empty || 'not set') + '</span>' : valHtml) +
      '</div></div>';
  }

  function renderSettings() {
    settingsCSS();
    const root = $('settings-root'); if (!root || !keyData) return;
    const k = keyData;
    root.innerHTML = '';

    function secIdentity() {
      return osection({
        icon: 'i-key', title: 'Identity & credentials', editable: true,
        read: (b) => {
          b.innerHTML =
            orow('Name', escHtml(k.name || '(unnamed)')) +
            orow('Public ID', '<code>' + escHtml(k.public_id) + '</code><button class="apin-iconbtn" data-cp="' + escHtml(k.public_id) + '" title="Copy public ID" aria-label="Copy public ID"><svg viewBox="0 0 24 24"><use href="#i-clipboard"/></svg></button>') +
            orow('Token', '<code>apin_' + escHtml(k.environment || '?') + '_******' + escHtml(k.last_four || '????') + '</code>') +
            orow('Environment', '<span class="env-pill">' + escHtml(k.environment || '·') + '</span>') +
            orow('Created', k.created_at ? new Date(k.created_at).toLocaleString() : '', 'unknown') +
            orow('Note', k.note ? escHtml(k.note) : '', 'no note');
          wireCopy(b);
        },
        edit: (b) => {
          b.innerHTML =
            '<div class="ofield"><label>Name</label><input class="apin-input" id="ed-name" value="' + escHtml(k.name || '') + '" maxlength="120" style="max-width:360px"></div>' +
            '<div class="ofield"><label>Note</label><textarea class="apin-input" id="ed-note" style="min-height:80px;resize:vertical" placeholder="optional — what this key is for">' + escHtml(k.note || '') + '</textarea></div>';
        },
        collect: (b) => {
          const out = {};
          const nm = b.querySelector('#ed-name').value.trim();
          const nt = b.querySelector('#ed-note').value;
          if (nm && nm !== (k.name || '')) out.name = nm;
          if (nt !== (k.note || '')) out.note = nt || null;
          return out;
        },
      });
    }

    function secAccess() {
      const inGroup = k.group_id != null;
      const role = k.group_role;          // 'locked' | 'special' | null
      const locked = inGroup && role === 'locked';
      const ceiling = (role === 'special') ? (k.scope_ceiling || null) : null;
      const badge = inGroup
        ? '<div class="ogroup-badge"><span class="ic"><svg viewBox="0 0 24 24"><use href="#i-shield-alert"/></svg></span>' +
          '<span>Group <b>' + escHtml(k.group_name || '?') + '</b></span>' +
          '<span class="orole orole-' + (role || 'locked') + '">' + escHtml(role || 'locked') + '</span></div>'
        : '';
      return osection({
        icon: 'i-shield-alert', title: 'Access & scopes',
        editable: !locked,
        read: (b) => {
          const sc = k.scopes || [];
          const tick = '<span class="tick"><svg viewBox="0 0 24 24"><use href="#i-check"/></svg></span>';
          let html = badge;
          if (locked) {
            html += '<div class="ohint" style="margin-bottom:13px">Scopes are inherited from group <b>' + escHtml(k.group_name || '') + '</b>. Edit the group (on the Keys page), or make this a special key, to change them.</div>';
          } else if (role === 'special') {
            html += '<div class="ohint" style="margin-bottom:13px">Special key — scopes are editable up to the group ceiling' +
              (ceiling ? ' (' + ceiling.length + ' allowed)' : '') + '.</div>';
          }
          if (!sc.length) { b.innerHTML = html + '<div class="oscl-none">No scopes — this key can do nothing.</div>'; return; }
          html += '<div class="oscl">';
          SCOPE_GROUPS.forEach(g => {
            const granted = g.scopes.filter(s => sc.indexOf(s) >= 0);
            if (!granted.length) return;
            html += '<div class="oscl-grp"><div class="oscl-cat"><b>' + escHtml(g.label) + '</b><span>' + granted.length + '/' + g.scopes.length + ' granted</span></div>' +
              granted.map(s => '<div class="oscl-row">' + tick + '<code>' + escHtml(s) + '</code><span class="d">' + escHtml(SCOPE_DESC[s] || '') + '</span></div>').join('') + '</div>';
          });
          const custom = sc.filter(s => KNOWN_SCOPES.indexOf(s) < 0);
          if (custom.length) {
            html += '<div class="oscl-grp"><div class="oscl-cat"><b>Custom</b></div>' +
              custom.map(s => '<div class="oscl-row">' + tick + '<code>' + escHtml(s) + '</code></div>').join('') + '</div>';
          }
          html += '</div>';
          b.innerHTML = html;
        },
        edit: (b) => { buildScopeEditor(b, k.scopes || [], ceiling); },
        collect: (b) => {
          const chosen = Array.prototype.map.call(b.querySelectorAll('input[data-scope]:checked'), i => i.getAttribute('data-scope'));
          const cur = (k.scopes || []).slice().sort().join(',');
          if (chosen.slice().sort().join(',') !== cur) return { scopes: chosen };
          return {};
        },
      });
    }

    function secLimits() {
      const PL = { hour: 'hour', day: 'day', week: 'week', month: 'month' };
      return osection({
        icon: 'i-gauge', title: 'Limits & quota', editable: true,
        read: (b) => {
          const per = k.quota_period || 'day';
          let quotaVal = '';
          if (k.quota_per_day) {
            const used = k.quota_used_period || 0, lim = k.quota_per_day;
            const pct = lim > 0 ? Math.min(100, Math.round((used / lim) * 100)) : 0;
            quotaVal = '<div class="oqbar-wrap"><div class="oqbar-top"><span><b>' + fmtNum(lim) + '</b> req / ' + PL[per] + '</span><span>' + fmtNum(used) + ' used · ' + pct + '%</span></div>' +
              '<div class="oqbar"><div class="oqbar-fill' + (pct >= 80 ? ' hot' : '') + '" style="width:' + pct + '%"></div></div></div>';
          }
          b.innerHTML =
            orow('Rate limit', k.rate_limit_per_min ? '<b>' + k.rate_limit_per_min + '</b> req/min' : '', 'unlimited') +
            orow('Quota', quotaVal, 'unlimited') +
            orow('Expires', k.expires_at ? new Date(k.expires_at).toLocaleString() + ' · ' + fmtAgo(k.expires_at) : '', 'no expiry');
        },
        edit: (b) => {
          const per = k.quota_period || 'day';
          const opt = (v) => '<option value="' + v + '"' + (per === v ? ' selected' : '') + '>per ' + v + '</option>';
          b.innerHTML =
            '<div class="ofield"><label>Rate limit (requests / minute)</label><input class="apin-input" id="ed-rate" type="number" min="0" placeholder="unlimited" value="' + (k.rate_limit_per_min || '') + '" style="max-width:200px"></div>' +
            '<div class="ofield"><label>Quota</label><div class="oqedit"><input class="apin-input" id="ed-quota" type="number" min="0" placeholder="unlimited" value="' + (k.quota_per_day || '') + '"><span style="color:var(--ink-mute,#8b8273);font:11px monospace">requests</span><select class="apin-select" id="ed-period">' + opt('hour') + opt('day') + opt('week') + opt('month') + '</select></div><div class="ohint">Leave blank for unlimited. Enforcement ships with the metering engine — this records the policy now.</div></div>' +
            '<div class="ofield"><label>Expiry</label><div class="oin-mini"><input class="apin-input" id="ed-exp" type="date" style="max-width:180px" value="' + (k.expires_at ? new Date(k.expires_at).toISOString().slice(0, 10) : '') + '"><span class="opreset" id="exp-presets"><button data-d="0">never</button><button data-d="30">30d</button><button data-d="60">60d</button><button data-d="90">90d</button></span></div></div>';
          b.querySelectorAll('#exp-presets button').forEach(btn => btn.addEventListener('click', () => {
            const d = parseInt(btn.getAttribute('data-d'), 10);
            const inp = b.querySelector('#ed-exp');
            if (!d) inp.value = '';
            else { const dt = new Date(Date.now() + d * 864e5); inp.value = dt.toISOString().slice(0, 10); }
          }));
        },
        collect: (b) => {
          const out = {};
          const rate = b.querySelector('#ed-rate').value.trim();
          const quota = b.querySelector('#ed-quota').value.trim();
          const period = b.querySelector('#ed-period').value;
          const exp = b.querySelector('#ed-exp').value;
          const rv = rate === '' ? null : parseInt(rate, 10);
          const qv = quota === '' ? null : parseInt(quota, 10);
          if (rv !== (k.rate_limit_per_min || null)) out.rate_limit_per_min = rv;
          if (qv !== (k.quota_per_day || null)) out.quota_per_day = qv;
          if (qv != null && period !== (k.quota_period || 'day')) out.quota_period = period;
          const curExp = k.expires_at ? new Date(k.expires_at).toISOString().slice(0, 10) : '';
          if (exp !== curExp) out.expires_at = exp ? new Date(exp + 'T23:59:59Z').toISOString() : null;
          return out;
        },
      });
    }

    function secNetwork() {
      return osection({
        icon: 'i-lock', title: 'Network allowlists', editable: true,
        read: (b) => {
          const ip = k.ip_allowlist || [], og = k.origin_allowlist || [];
          b.innerHTML =
            orow('IP allowlist', ip.length ? ip.map(x => '<code>' + escHtml(x) + '</code>').join(' ') : '', 'all IPs allowed') +
            orow('Origin allowlist', og.length ? og.map(x => '<code>' + escHtml(x) + '</code>').join(' ') : '', 'all origins allowed');
        },
        edit: (b) => {
          b.innerHTML =
            '<div class="ofield"><label>IP allowlist (one CIDR/IP per line — blank = all)</label><textarea class="apin-input" id="ed-ip" style="min-height:80px;resize:vertical" placeholder="e.g. 203.0.113.0/24">' + escHtml((k.ip_allowlist || []).join('\n')) + '</textarea><button class="apin-btn ghost" id="add-my-ip" type="button" style="margin-top:8px">+ add my current IP</button></div>' +
            '<div class="ofield"><label>Origin allowlist (one origin per line — blank = all)</label><textarea class="apin-input" id="ed-origin" style="min-height:80px;resize:vertical" placeholder="e.g. https://app.example.com">' + escHtml((k.origin_allowlist || []).join('\n')) + '</textarea></div>';
          const addBtn = b.querySelector('#add-my-ip');
          addBtn.addEventListener('click', async () => {
            addBtn.textContent = 'detecting…';
            try {
              const r = await fetch('/api/account/keys/_meta/my-ip', { credentials: 'include', headers: { 'Accept': 'application/json' } });
              const j = await r.json();
              const ip = j && j.data && j.data.ip;
              if (ip) { const ta = b.querySelector('#ed-ip'); const lines = ta.value.split('\n').map(s => s.trim()).filter(Boolean); if (!lines.includes(ip)) lines.push(ip); ta.value = lines.join('\n'); addBtn.textContent = '✓ added ' + ip; }
              else addBtn.textContent = 'could not detect';
            } catch (e) { addBtn.textContent = 'could not detect'; }
          });
        },
        collect: (b) => {
          const out = {};
          const ip = b.querySelector('#ed-ip').value.split('\n').map(s => s.trim()).filter(Boolean);
          const og = b.querySelector('#ed-origin').value.split('\n').map(s => s.trim()).filter(Boolean);
          if (ip.join(',') !== (k.ip_allowlist || []).join(',')) out.ip_allowlist = ip;
          if (og.join(',') !== (k.origin_allowlist || []).join(',')) out.origin_allowlist = og;
          return out;
        },
      });
    }

    function secSecurity() {
      return osection({
        icon: 'i-activity', title: 'Security & usage', editable: false,
        read: (b) => {
          const ua = k.last_used_ua ? (k.last_used_ua.length > 64 ? k.last_used_ua.slice(0, 64) + '…' : k.last_used_ua) : '';
          b.innerHTML =
            orow('Status', '<span class="status-pill status-' + (k.status || 'disabled') + '">' + escHtml(k.status || '·') + '</span>') +
            orow('Total requests', '<b>' + fmtNum(k.request_count || 0) + '</b>') +
            orow('5xx errors', '<b>' + fmtNum(k.error_count || 0) + '</b>' + (k.error_4xx_plus ? ' <span style="color:var(--ink-mute)">· ' + fmtNum(k.error_4xx_plus) + ' total 4xx+</span>' : '')) +
            orow('Last used', k.last_used_at ? fmtAgo(k.last_used_at) : '', 'never used') +
            orow('Last IP', k.last_used_ip ? '<code>' + escHtml(k.last_used_ip) + '</code>' : '', '—') +
            orow('Last client', ua ? escHtml(ua) : '', '—');
          const post = document.createElement('div'); post.style.marginTop = '16px';
          post.innerHTML = posture(k).map(p =>
            '<div class="opost ' + (p.ok ? 'ok' : 'warn') + '"><span class="dot"></span><span>' + escHtml(p.text) +
            (p.ok ? '' : ' <a data-alert="' + escHtml(p.alert || '') + '">' + (p.alert ? 'create alert' : '') + '</a>') + '</span></div>').join('');
          b.appendChild(post);
          post.querySelectorAll('[data-alert]').forEach(a => a.addEventListener('click', () => {
            window.location.href = '/account/api/alerts?focus=' + encodeURIComponent(a.getAttribute('data-alert')) + '&pid=' + encodeURIComponent(PID);
          }));
        },
      });
    }

    function secDanger() {
      const dz = document.createElement('div'); dz.className = 'oset danger';
      const enabling = k.status === 'disabled';
      dz.innerHTML = '<div class="oset-h"><span class="ic" style="color:#b3402f"><svg viewBox="0 0 24 24"><use href="#i-warning"/></svg></span><h3>Danger zone</h3></div>' +
        '<div class="oset-body">' +
        '<div class="odanger-row"><div class="t"><b>' + (enabling ? 'Enable key' : 'Disable key') + '</b><span>' + (enabling ? 'start accepting requests again' : 'stop accepting requests (reversible)') + '</span></div><button class="apin-btn" data-dz="disable">' + (enabling ? 'Enable' : 'Disable') + '</button></div>' +
        '<div class="odanger-row"><div class="t"><b>Rotate token</b><span>issue a new secret, with a grace window</span></div><button class="apin-btn" data-dz="rotate">Rotate</button></div>' +
        '<div class="odanger-row"><div class="t"><b>Delete key</b><span>permanent — removes the key and its analytics</span></div><button class="apin-btn danger" data-dz="delete">Delete</button></div>' +
        '</div>';
      dz.querySelector('[data-dz="disable"]').addEventListener('click', () => window.APIN.keyActions.disable());
      dz.querySelector('[data-dz="rotate"]').addEventListener('click', () => window.APIN.keyActions.rotate());
      dz.querySelector('[data-dz="delete"]').addEventListener('click', () => window.APIN.keyActions.del());
      return dz;
    }

    const SECTIONS = [
      { id: 'identity', icon: 'i-key', label: 'Identity', sub: 'name · token · note', build: secIdentity },
      { id: 'access', icon: 'i-shield-alert', label: 'Access', sub: 'scopes & permissions', build: secAccess },
      { id: 'limits', icon: 'i-gauge', label: 'Limits', sub: 'rate · quota · expiry', build: secLimits },
      { id: 'network', icon: 'i-lock', label: 'Network', sub: 'IP & origin allow', build: secNetwork },
      { id: 'security', icon: 'i-activity', label: 'Security', sub: 'live usage', build: secSecurity },
      { id: '_sep' },
      { id: 'danger', icon: 'i-warning', label: 'Danger zone', sub: 'rotate · delete', danger: true, build: secDanger },
    ];

    const wrap = document.createElement('div'); wrap.className = 'oset-wrap';
    const rail = document.createElement('nav'); rail.className = 'oset-rail';
    const pane = document.createElement('div'); pane.className = 'oset-pane';

    function setActive(id) {
      const valid = SECTIONS.find(s => s.id === id && s.build);
      _setSection = valid ? id : 'identity';
      Array.prototype.forEach.call(rail.querySelectorAll('.oset-nav'), n => n.classList.toggle('on', n.dataset.sid === _setSection));
      pane.innerHTML = '';
      const sec = SECTIONS.find(s => s.id === _setSection);
      pane.appendChild(sec.build());
    }

    SECTIONS.forEach(s => {
      if (s.id === '_sep') { const d = document.createElement('div'); d.className = 'oset-sep'; rail.appendChild(d); return; }
      const nav = document.createElement('button');
      nav.className = 'oset-nav' + (s.danger ? ' danger' : '');
      nav.dataset.sid = s.id;
      nav.innerHTML = '<span class="ic"><svg viewBox="0 0 24 24"><use href="#' + s.icon + '"/></svg></span>' +
        '<span class="lab"><b>' + escHtml(s.label) + '</b><span>' + escHtml(s.sub) + '</span></span>';
      nav.addEventListener('click', () => setActive(s.id));
      rail.appendChild(nav);
    });

    wrap.appendChild(rail); wrap.appendChild(pane);
    root.appendChild(wrap);
    setActive(_setSection || 'identity');
  }

  function wireCopy(b) {
    b.querySelectorAll('[data-cp]').forEach(btn => btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(btn.getAttribute('data-cp'));
        if (btn.classList.contains('apin-iconbtn')) {
          const orig = btn.innerHTML;
          btn.innerHTML = '<svg viewBox="0 0 24 24"><use href="#i-check"/></svg>';
          btn.classList.add('ok');
          setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('ok'); }, 1200);
        } else {
          const t = btn.textContent; btn.textContent = 'copied'; setTimeout(() => btn.textContent = t, 1200);
        }
      } catch (e) {}
    }));
  }

  function buildScopeEditor(b, current, ceiling) {
    const has = (s) => current.indexOf(s) >= 0;
    const allowed = (s) => !ceiling || ceiling.indexOf(s) >= 0;
    const dis = (s) => allowed(s) ? '' : ' disabled';
    const note = ceiling ? '<div class="ohint" style="margin-bottom:11px">Greyed scopes are above this special key’s ceiling — promote its ceiling from the group to allow them.</div>' : '';
    const tog = '<div style="display:flex;align-items:center;margin-bottom:12px"><span class="oscope-toggle" id="scope-view">' +
      '<button data-v="grouped" aria-pressed="' + (_scopeView === 'grouped') + '">Grouped</button>' +
      '<button data-v="all" aria-pressed="' + (_scopeView === 'all') + '">All permissions</button></span></div>';
    function render() {
      let html = note + tog;
      if (_scopeView === 'grouped') {
        html += SCOPE_GROUPS.map(g => '<div class="ogrp"><div class="ogrp-h"><label><input type="checkbox" class="apin-check" data-grp="' + escHtml(g.key) + '"> ' + escHtml(g.label) + '</label> <span class="sub">' + escHtml(g.scopes.join(' · ')) + '</span></div>' +
          g.scopes.map(s => '<div class="oscope-row"><label><input type="checkbox" class="apin-check" data-scope="' + escHtml(s) + '"' + (has(s) ? ' checked' : '') + dis(s) + '> <code>' + escHtml(s) + '</code> <span style="color:var(--ink-mute,#8b8273)">— ' + escHtml(SCOPE_DESC[s] || '') + '</span></label></div>').join('') + '</div>').join('');
      } else {
        const all = KNOWN_SCOPES.concat(current.filter(s => KNOWN_SCOPES.indexOf(s) < 0));
        html += all.map(s => '<div class="oscope-row"><label><input type="checkbox" class="apin-check" data-scope="' + escHtml(s) + '"' + (has(s) ? ' checked' : '') + dis(s) + '> <code>' + escHtml(s) + '</code> <span style="color:var(--ink-mute,#8b8273)">— ' + escHtml(SCOPE_DESC[s] || 'custom scope') + '</span></label></div>').join('');
      }
      b.innerHTML = html;
      b.querySelectorAll('#scope-view button').forEach(btn => btn.addEventListener('click', () => { _scopeView = btn.getAttribute('data-v'); current = Array.prototype.map.call(b.querySelectorAll('input[data-scope]:checked'), i => i.getAttribute('data-scope')); render(); }));
      SCOPE_GROUPS.forEach(g => {
        const parent = b.querySelector('input[data-grp="' + g.key + '"]'); if (!parent) return;
        const kids = g.scopes.map(s => b.querySelector('input[data-scope="' + s + '"]')).filter(Boolean).filter(c => !c.disabled);
        if (!kids.length) { parent.disabled = true; return; }
        const sync = () => { const on = kids.filter(c => c.checked).length; parent.checked = on === kids.length; parent.indeterminate = on > 0 && on < kids.length; };
        sync();
        parent.addEventListener('change', () => { kids.forEach(c => c.checked = parent.checked); });
        kids.forEach(c => c.addEventListener('change', sync));
      });
    }
    render();
  }

  function posture(k) {
    const out = [];
    if (!k.expires_at && k.environment === 'live') out.push({ ok: false, text: 'No expiry on a LIVE key — it never auto-rotates.', alert: 'key.no_expiry' });
    if (!(k.ip_allowlist && k.ip_allowlist.length)) out.push({ ok: false, text: 'No IP allowlist — any IP can use this key.', alert: 'key.no_ip_allowlist' });
    const writeScopes = (k.scopes || []).filter(s => /:write$/.test(s));
    if (writeScopes.length) out.push({ ok: false, text: 'Has write scopes (' + writeScopes.join(', ') + ') — keep this key secret.', alert: 'key.broad_scopes' });
    if (!out.length) out.push({ ok: true, text: 'No hygiene issues found.' });
    return out;
  }

  // ─── 9.O.3 · Lifecycle action modals (in-page, no navigation) ─────────
  // Sudo step-up: sensitive key mutations return 403 sudo_required. Prompt
  // for the account password, POST /api/account/sudo, capture the rotated
  // csrf token, then let the caller retry. Resolves true on success.
  function requireSudo() {
    const MD = window.APIN && window.APIN.modal;
    if (!MD) return Promise.resolve(false);
    return new Promise((resolve) => {
      let settled = false;
      const finish = (v) => { if (!settled) { settled = true; resolve(v); } };
      MD.open({
        icon: 'i-shield-alert', title: 'Confirm it’s you',
        subtitle: 'A quick password check protects key changes',
        body: (el) => {
          el.innerHTML =
            '<div class="apm-field"><label>Account password</label>' +
            '<input class="apm-input" id="sudo-pw" type="password" autocomplete="current-password" placeholder="••••••••"></div>' +
            '<div class="hint">You stay signed in. This just re-confirms it’s really you.</div>';
          const inp = el.querySelector('#sudo-pw');
          if (inp) setTimeout(() => inp.focus(), 30);
        },
        onClose: () => finish(false),
        actions: [
          { label: 'Cancel', kind: 'ghost' },
          { label: 'Confirm', kind: 'primary', busyLabel: 'Verifying…', closeOnClick: false,
            onClick: async (ctx) => {
              const pwEl = ctx.query('#sudo-pw');
              const pw = (pwEl && pwEl.value) || '';
              if (!pw) { ctx.setError('Password required.'); return false; }
              const r = await fetch('/api/account/sudo', { method: 'POST', credentials: 'include',
                headers: { 'Accept': 'application/json', 'Content-Type': 'application/json', 'X-Console-Csrf': csrf },
                body: JSON.stringify({ password: pw }) });
              const b = await r.json().catch(() => ({}));
              if (r.status === 200 && b.ok) {
                const nc = b.data && b.data.csrf_token;
                if (nc) { csrf = nc; if (csrfMeta) csrfMeta.content = nc; }
                settled = true; resolve(true); ctx.close(); return true;
              }
              ctx.setError((b.error && b.error.message) || 'That password did not match.');
              return false;
            } },
        ],
      });
    });
  }

  async function apiWrite(url, method, payload, _retried) {
    const opts = { method: method, credentials: 'include',
      headers: { 'X-Console-Csrf': csrf } };
    if (payload !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(payload);
    }
    const r = await fetch(url, opts);
    if (r.status === 401) { window.location.href = '/dashboard'; throw new Error('Session expired — please sign in again.'); }
    const body = await r.json().catch(() => ({}));
    if (r.status === 403 && body && body.error && body.error.code === 'sudo_required' && !_retried) {
      const ok = await requireSudo();
      if (ok) return apiWrite(url, method, payload, true);
    }
    return { status: r.status, body: body };
  }
  function writeErr(body) { return (body && body.error && body.error.message) || 'Request failed. Please try again.'; }
  async function afterAction() { try { await refreshMasthead(); loadActiveTab(); } catch (e) {} }
  const KEY_URL = '/api/account/keys/' + encodeURIComponent(PID);

  // expose so the Settings tab + masthead can call the same flows
  window.APIN = window.APIN || {};
  window.APIN.keyActions = { rotate: actRotate, disable: actDisable, del: actDelete, write: apiWrite, writeErr: writeErr, after: afterAction };

  function actRotate() {
    const MD = window.APIN && window.APIN.modal; if (!MD) return;
    MD.open({
      icon: 'i-refresh', title: 'Rotate token', subtitle: keyData ? keyData.name : '',
      body: (el) => {
        el.innerHTML =
          '<div class="apm-warn">Rotating issues a NEW secret. The old token keeps working during the grace window, then stops.</div>' +
          '<div class="apm-field"><label>Grace period</label>' +
          '<select class="apm-sel" id="rot-grace">' +
          '<option value="0">No grace — old token stops immediately</option>' +
          '<option value="3600">1 hour</option>' +
          '<option value="86400">24 hours</option>' +
          '<option value="172800" selected>48 hours (default)</option>' +
          '<option value="604800">7 days</option>' +
          '</select><div class="hint">During grace, both the old and new tokens are valid.</div></div>';
      },
      actions: [
        { label: 'Cancel', kind: 'ghost' },
        { label: 'Rotate', kind: 'primary', busyLabel: 'Rotating…', closeOnClick: false,
          onClick: async (ctx) => {
            const grace = parseInt(ctx.query('#rot-grace').value, 10) || 0;
            const { body } = await apiWrite(KEY_URL + '/rotate', 'POST', { grace_seconds: grace });
            if (!body.ok) throw new Error(writeErr(body));
            const nk = body.data || {};
            const secret = nk.plaintext_token || nk.token || nk.secret || '';
            const k = keyData || {};
            // Same one-time-view used when minting — marked as a rotation.
            MD.oneTimeToken(ctx, {
              name: k.name || '', env: k.environment || '',
              public_id: nk.public_id || k.public_id || '',
              scopes: k.scopes || [], secret: secret,
              expires: k.expires_at ? new Date(k.expires_at).toLocaleDateString() : 'no expiry',
              rotating: true, onDone: afterAction,
            });
            return false;   // keep the modal open to show the one-time secret
          } },
      ],
    });
  }

  function actDisable() {
    const MD = window.APIN && window.APIN.modal; if (!MD) return;
    const enabling = keyData && keyData.status === 'disabled';
    MD.confirm({
      icon: enabling ? 'i-play' : 'i-pause',
      title: enabling ? 'Enable this key?' : 'Disable this key?',
      message: enabling
        ? 'The key will start accepting requests again.'
        : 'Requests with this key will be rejected with <strong>401</strong> until you re-enable it. No data is deleted.',
      confirmLabel: enabling ? 'Enable' : 'Disable',
      busyLabel: enabling ? 'Enabling…' : 'Disabling…',
      onConfirm: async () => {
        const { body } = await apiWrite(KEY_URL + (enabling ? '/enable' : '/disable'), 'POST', {});
        if (!body.ok) throw new Error(writeErr(body));
        afterAction();
      },
    });
  }

  function actDelete() {
    const MD = window.APIN && window.APIN.modal; if (!MD) return;
    const name = (keyData && keyData.name) || PID;
    // delete needs status 'disabled'/'expired'. Anything else (active OR
    // rotating, legacy_pending…) is disabled first in the same step.
    const st = keyData && keyData.status;
    const needsDisable = !!st && st !== 'disabled' && st !== 'expired';
    MD.confirm({
      icon: 'i-trash', danger: true, title: 'Delete "' + name + '"',
      message: 'This permanently deletes the key and its analytics. <strong>This cannot be undone.</strong>' +
        (needsDisable ? ' The key will be disabled, then deleted.' : ''),
      requireText: name, confirmLabel: 'Delete key', busyLabel: 'Deleting…',
      onConfirm: async () => {
        // 9.S.4 · capture the toast cursor BEFORE the mutation, so a
        // refresh-poll that fires mid-flight can't advance it past the
        // soon-to-be-created key.deleted alert.
        let pendSince = null;
        try {
          if (window.APIN && window.APIN.toast && window.APIN.toast.cursor) {
            pendSince = window.APIN.toast.cursor();
          }
        } catch (_) {}
        if (needsDisable) {
          const dis = await apiWrite(KEY_URL + '/disable', 'POST', {});
          if (!dis.body.ok) throw new Error(writeErr(dis.body));
        }
        const { body } = await apiWrite(KEY_URL, 'DELETE', undefined);
        if (!body.ok) throw new Error(writeErr(body));
        // The delete emits a key.deleted alert, but we immediately navigate
        // away — hand off the pre-mutation cursor so the list page's toast
        // bootstrap delivers it instead of swallowing it.
        try {
          if (window.APIN && window.APIN.toast && window.APIN.toast.markPending) {
            window.APIN.toast.markPending(pendSince);
          }
        } catch (_) {}
        window.location.href = '/account/api/keys';
      },
    });
  }

  document.querySelectorAll('[data-act]').forEach(b => {
    b.addEventListener('click', () => {
      const act = b.dataset.act;
      if (act === 'rotate') actRotate();
      else if (act === 'disable' || act === 'enable') actDisable();
      else if (act === 'delete') actDelete();
    });
  });

  // ─── Polling refresh ─────────────────────────────────────────────────
  function startPoll() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      if (document.visibilityState !== 'visible') return;
      await refreshMasthead();
      // Reload whichever tab is active so the data feels live.
      if (activeTab === 'overview') loadOverview();
      else if (activeTab === 'requests') {
        // Only refresh first page silently. Don't disrupt scrolled position.
        if (!reqCursor) loadRequests({ reset: true });
      }
      else if (activeTab === 'audit') loadAudit();
    }, 15_000);
    if ($('sse-dot')) $('sse-dot').classList.remove('off');
  }
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      refreshMasthead().then(loadActiveTab);
      startPoll();
    } else {
      if (pollTimer) clearInterval(pollTimer);
      if ($('sse-dot')) $('sse-dot').classList.add('off');
    }
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', ev => {
    const t = ev.target;
    if (t && typeof t.matches === 'function' && t.matches('input, select, textarea')) return;
    if (t && t.isContentEditable) return;
    // Don't let tab shortcuts fire while a modal is open (sudo, rotate, etc.)
    if (document.querySelector('[class*="apm-root"], [class*="apm-backdrop"]')) return;
    const map = { '1': 'overview', '2': 'usage', '3': 'requests', '4': 'audit', '5': 'settings' };
    if (map[ev.key]) {
      ev.preventDefault();
      setActiveTab(map[ev.key]);
    } else if (ev.key.toLowerCase() === 'r' && activeTab === 'requests') {
      ev.preventDefault();
      loadRequests({ reset: true });
    }
  });

  // ─── Cross-module surface (exposed UNCONDITIONALLY) ───────────────────
  // 9.N.T · The Traffic tab's request container calls APIN.keyDetail.openRequest
  // and .filterRequests. These must exist even if the user lands directly on
  // #traffic (i.e. loadOverview() never ran). Define them at module scope.
  window.APIN = window.APIN || {};
  window.APIN.keyDetail = Object.assign(window.APIN.keyDetail || {}, {
    openRequest: openRequestDetail,
    filterRequests: function (since, until) {
      reqTimeFilter = (since && until) ? { since: since, until: until } : null;
      setActiveTab('requests');
      var pane = document.getElementById('pane-requests');
      var banner = document.getElementById('req-time-banner');
      if (reqTimeFilter && pane) {
        if (!banner) {
          banner = document.createElement('div');
          banner.id = 'req-time-banner';
          banner.style.cssText = "display:flex;align-items:center;gap:10px;background:rgba(120,110,90,.08);border:1px solid var(--paper-edge);border-radius:8px;padding:8px 12px;margin-bottom:12px;font:12px 'JetBrains Mono',monospace;color:var(--ink-soft)";
          pane.insertBefore(banner, pane.firstChild);
        }
        var fmt = (window.APIN && APIN.time) ? APIN.time.local : function (s) { return s; };
        banner.innerHTML = 'filtered: ' + fmt(since) + ' &rarr; ' + fmt(until) +
          ' <button id="req-clear-filter" style="margin-left:auto;background:none;border:1px solid var(--paper-edge);border-radius:6px;padding:3px 9px;cursor:pointer;font:inherit;color:var(--ink)">clear</button>';
        var cb = document.getElementById('req-clear-filter');
        if (cb) cb.addEventListener('click', function () { reqTimeFilter = null; banner.remove(); loadRequests({ reset: true }); });
      } else if (banner) { banner.remove(); }
    },
  });

  // ─── Boot ────────────────────────────────────────────────────────────
  (async function boot() {
    // Bind tab links FIRST so click works even before /keys fetch returns.
    document.querySelectorAll('a.tab').forEach(a => {
      a.addEventListener('click', e => {
        e.preventDefault();
        setActiveTab(a.dataset.tab);
      });
    });
    setActiveTab(readHashTab(), { silent: true });
    const ok = await refreshMasthead();
    if (!ok) return;
    loadActiveTab();
    startPoll();
  })();
})();
