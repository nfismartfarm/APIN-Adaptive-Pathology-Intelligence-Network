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
  const csrf = csrfMeta ? csrfMeta.content : '';
  const pidMeta = document.querySelector('meta[name="key-public-id"]');
  const PID = pidMeta ? pidMeta.content : '';
  const TABS = ['overview', 'traffic', 'usage', 'requests', 'audit', 'settings'];
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
    $('key-name').textContent = k.name || '(unnamed)';
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
      `<span class="pill">created <b>${fmtAgo(k.created_at)}</b></span>`,
      `<span class="pill">last used <b>${k.last_used_at ? fmtAgo(k.last_used_at) : 'never'}</b></span>`,
    ];
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
    // Stop the Traffic live loop (SSE + clock rAF) whenever we leave the tab.
    if (activeTab !== 'traffic' && window.APIN && APIN.keyTraffic) APIN.keyTraffic.deactivate();
    if (activeTab === 'overview')  loadOverview();
    else if (activeTab === 'traffic') { if (window.APIN && APIN.keyTraffic) APIN.keyTraffic.activate(PID); }
    else if (activeTab === 'usage')    loadUsageTab();
    else if (activeTab === 'requests') loadRequests({ reset: true });
    else if (activeTab === 'audit')    loadAudit();
    else if (activeTab === 'settings') renderSettings();
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

    $('reqd-title').textContent = (r.method || '') + ' ' + (r.path || '');
    $('reqd-sub').textContent = (r.timestamp || '') + ' UTC · key: ' + (r.key_name || r.key_public_id || '?');

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

  // ─── 9.J · Settings tab (industry-grade redesign) ────────────────────
  function renderSettings() {
    if (!keyData) return;
    const k = keyData;
    const setText = (id, v, empty) => {
      const el = $(id);
      if (!el) return;
      if (v == null || v === '' || (Array.isArray(v) && v.length === 0)) {
        el.classList.add('empty');
        el.textContent = empty || 'not set';
      } else {
        el.classList.remove('empty');
        if (Array.isArray(v)) el.innerHTML = v.map(s => '<code>' + escHtml(s) + '</code>').join(' ');
        else el.textContent = v;
      }
    };
    setText('set-pid', k.public_id);
    setText('set-token', 'apin_' + (k.environment || '?') +
      '_******' + (k.last_four || '????'));
    // Environment pill — color it like the masthead pills
    const envEl = $('set-env');
    if (envEl) {
      envEl.textContent = k.environment || '·';
      envEl.className = 'env-pill';
    }
    // Status pill with status-color class
    const statusEl = $('set-status');
    if (statusEl) {
      statusEl.textContent = k.status || '·';
      statusEl.className = 'status-pill status-' + (k.status || 'disabled');
    }
    // Scopes — render as chips
    const scopesEl = $('set-scopes-chips');
    if (scopesEl) {
      const scopes = k.scopes || [];
      if (scopes.length === 0) {
        scopesEl.innerHTML = '<span style="font-family:Fraunces,serif;font-style:italic;color:var(--ink-mute)">no scopes</span>';
      } else {
        scopesEl.innerHTML = scopes.map(s =>
          '<span class="scope-chip">' + escHtml(s) + '</span>').join('');
      }
    }
    setText('set-created', k.created_at ? new Date(k.created_at).toLocaleString() : null, 'unknown');
    setText('set-last-used', k.last_used_at ? fmtAgo(k.last_used_at) + ' (' + k.last_used_at + ')' : null, 'never used');
    setText('set-expires', k.expires_at ? new Date(k.expires_at).toLocaleString() : null, 'no expiry');
    setText('set-rate-limit', k.rate_limit_per_min
      ? k.rate_limit_per_min + ' req/min' : null, 'unlimited');
    setText('set-quota', k.quota_per_day
      ? k.quota_per_day + ' req/day' : null, 'unlimited');
    setText('set-ip-allow', k.ip_allowlist || [], 'all IPs allowed');
    setText('set-origin-allow', k.origin_allowlist || [], 'all origins allowed');
    setText('set-req-count', (k.request_count != null) ? fmtNum(k.request_count) : '0');
    setText('set-err-count', (k.error_count != null) ? fmtNum(k.error_count) : '0');
    setText('set-last-ip', k.last_used_ip, 'never used');
    setText('set-last-ua', k.last_used_ua ? (k.last_used_ua.length > 80 ? k.last_used_ua.slice(0, 80) + '…' : k.last_used_ua) : null, 'never used');

    // Wire setcard-edit buttons to the masthead edit flow
    document.querySelectorAll('.setcard-edit').forEach(btn => {
      if (btn._wired) return;
      btn._wired = true;
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        // Reuse the masthead's edit button click
        window.location.href = '/account/api/keys#act=edit&pid=' + encodeURIComponent(PID);
      });
    });
  }

  // ─── Action buttons ──────────────────────────────────────────────────
  // For Phase 9.C the action buttons round-trip to the keys list so the
  // existing modal flows light up.
  document.querySelectorAll('[data-act]').forEach(b => {
    b.addEventListener('click', () => {
      const act = b.dataset.act;
      window.location.href = '/account/api/keys#act=' + act + '&pid=' + encodeURIComponent(PID);
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
    if (ev.target.matches('input, select, textarea')) return;
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
