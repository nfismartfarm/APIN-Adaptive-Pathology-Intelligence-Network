/* Phase 9.E + 9.H/I/J — account-wide Usage page client.
 *
 * State model
 * -----------
 * One `filterState` object is the source of truth:
 *   { range:'24h', key_id:'', env:'', status:'', endpoint:'',
 *     mode:'total', logY:false }
 * Any UI control mutates filterState and calls `refreshAll()`. URL hash
 * mirrors filterState so the back button works and links are shareable.
 *
 * 9.H · D1-D8: data correctness — limits, ranges, error_codes,
 *   keys-dropdown population, delta-vs-prev rendering.
 * 9.I: every chart and every row is clickable. The marquee feature is the
 *   request-detail drawer (request-by-request HTTP+UA+source breakdown
 *   with redacted curl/python/node reconstructions).
 * 9.J: odometer-style proportional digit animation on every numeric KPI
 *   (uses /static/odometer.js). Skeleton-style fade on filter changes.
 */
(function () {
  'use strict';

  // ─── state ────────────────────────────────────────────────────────────
  const filterState = readHashFilters() || {
    range: '24h', key_id: '', env: '', status: '', endpoint: '',
    mode: 'total', logY: false, compare: false,
  };
  let availableKeys = [];
  let recentCursor = null;
  // 9.N.T29 · empty-state (seed→sprout) support
  let _lastLifetime = null;        // {ever,total,last_ts,last_id,last_method,last_path,last_status}
  let _accountLifetime = null;     // unfiltered (account-wide) lifetime, lazily fetched
  let _lastWinRequests = 0;        // windowed request count (for cross-metric hint)
  let _emptyHero = null;           // active hero handle (for teardown)
  let _heroActive = false;         // true while the empty hero is shown (live-react)
  const RANGE_MS = { '15m': 9e5, '1h': 36e5, '6h': 216e5, '24h': 864e5, '7d': 6048e5, '30d': 25920e5 };
  const RANGE_LADDER = ['15m', '1h', '6h', '24h', '7d', '30d'].map(id => ({ id: id, label: id, ms: RANGE_MS[id] }));

  // ─── helpers ──────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
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
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(2) + ' MB';
    return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
  }
  function fmtAgo(iso) {
    if (!iso) return '·';
    const ms = Date.now() - new Date(iso.replace(' ', 'T') + 'Z').getTime();
    if (isNaN(ms)) return iso;
    if (ms < 60_000) return Math.max(1, Math.round(ms / 1000)) + 's ago';
    if (ms < 3600_000) return Math.round(ms / 60_000) + 'm ago';
    if (ms < 86_400_000) return Math.round(ms / 3600_000) + 'h ago';
    return Math.round(ms / 86_400_000) + 'd ago';
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
    if (['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].includes(u)) return 'meth-' + u;
    return '';
  }

  async function api(url) {
    const r = await fetch(url, { credentials: 'include' });
    if (r.status === 401) { window.location.href = '/dashboard'; throw new Error('401'); }
    const body = await r.json().catch(() => ({}));
    return { status: r.status, body };
  }

  // ─── URL <-> filterState sync ────────────────────────────────────────
  function readHashFilters() {
    const h = (location.hash || '').replace(/^#/, '');
    if (!h) return null;
    const p = new URLSearchParams(h);
    const out = { range: '24h', mode: 'total', logY: false };
    for (const k of ['range', 'key_id', 'env', 'status', 'endpoint', 'mode']) {
      const v = p.get(k);
      if (v) out[k] = v;
    }
    out.logY = p.get('logY') === '1';
    return out;
  }
  function writeHashFilters() {
    const p = new URLSearchParams();
    Object.keys(filterState).forEach(k => {
      const v = filterState[k];
      if (k === 'logY') { if (v) p.set('logY', '1'); }
      else if (v) p.set(k, v);
    });
    const h = '#' + p.toString();
    if (h !== location.hash) history.replaceState(null, '', h);
  }

  // ─── 9.J · KPI tile + odometer rendering ─────────────────────────────
  function _odomElFor(k) {
    const tile = document.querySelector('.kpi[data-k="' + k + '"]');
    if (!tile) return null;
    return tile.querySelector('[data-num]');
  }

  function setKpi(k, value, delta, prevText, opts) {
    const el = document.querySelector('.kpi[data-k="' + k + '"]');
    if (!el) return;
    opts = opts || {};
    const numEl = el.querySelector('[data-num]');
    const dEl = el.querySelector('[data-delta]');
    const pEl = el.querySelector('[data-prev]');
    // 9.J · odometer animation — proportional speed, only for tiles whose
    // primary number is an .apin-odometer element. Bytes-out is text-only
    // (it has units like "10.3 MB" that the odometer can't roll digit-by-
    // digit cleanly), so it falls back to plain text.
    if (numEl && numEl.classList && numEl.classList.contains('apin-odometer')) {
      if (window.APIN && APIN.odometer && typeof APIN.odometer.roll === 'function') {
        APIN.odometer.roll(numEl, value);
      } else {
        numEl.textContent = String(value);
      }
    } else if (numEl) {
      numEl.textContent = String(value);
    }
    // Render the delta pill + "vs prev <range>" caption as a coupled pair.
    // When delta is null (no prior data), the pill shows "no prev data" and
    // the caption is HIDDEN — showing both at once was the original 9.J bug
    // that surfaced as "-- vs prev under every KPI tile."
    if (dEl) {
      if (delta == null || isNaN(delta)) {
        dEl.className = 'delta none';
        dEl.innerHTML = 'no prev data';
        if (pEl) { pEl.textContent = ''; pEl.style.display = 'none'; }
      } else if (Math.abs(delta) < 0.05) {
        dEl.className = 'delta flat';
        dEl.innerHTML = '<svg class="icon" aria-hidden="true" style="width:11px;height:11px"><use href="#i-minus"/></svg> no change';
        if (pEl) { pEl.textContent = prevText || ''; pEl.style.display = ''; }
      } else {
        const dirUp = delta > 0;
        const goodIfDown = opts.goodIfDown === true;
        const cls = 'delta ' + (dirUp ? 'up' : 'down') + (goodIfDown ? ' good' : '');
        dEl.className = cls;
        const arrow = dirUp ? 'i-arrow-up-right' : 'i-arrow-down-right';
        dEl.innerHTML = '<svg class="icon" aria-hidden="true" style="width:11px;height:11px"><use href="#' +
          arrow + '"/></svg> ' + Math.abs(delta).toFixed(1) + '%';
        if (pEl) { pEl.textContent = prevText || ''; pEl.style.display = ''; }
      }
    } else if (pEl) {
      pEl.textContent = prevText || '';
    }
  }

  // ─── 9.N.5 · Six new charts — loaders ────────────────────────────────
  // Each loader fetches its data and renders into its #*-host card.
  // All scoped to filterState.range / key_id / env where applicable.

  // 9.N.5.f · Activity calendar with 4 zoom modes.
  let _heatmapMode = 'week';
  async function loadHeatmap(forcedMode) {
    const host = document.getElementById('heatmap-host');
    if (!host) return;
    const mode = forcedMode || _heatmapMode || 'week';
    _heatmapMode = mode;
    const qs = new URLSearchParams({ mode });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env)    qs.set('env',    filterState.env);
    const { body } = await api('/api/account/usage/heatmap-calendar?' + qs);
    if (!body || !body.ok) return;
    // Update aux caption
    const aux = document.getElementById('heatmap-aux');
    if (aux) {
      const msg = { week: 'last 7 days · 24h × 7 grid', month: 'last 30 days · day grid', year: 'last 12 months', years: 'last 5 years' }[mode];
      aux.textContent = msg + ' · click a cell to drill';
    }
    APIN.charts.activityHeatmap(host, body.data, {
      onClickCell: cell => {
        if (window.APIN && APIN.toast) {
          const lbl = cell.label || (cell.row + '·' + cell.col);
          APIN.toast({ msg: lbl + ' · ' + cell.count + ' requests' });
        }
      }
    });
  }
  function _bindHeatmapModeSwitcher() {
    document.querySelectorAll('#heatmap-mode button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#heatmap-mode button').forEach(b => b.removeAttribute('aria-pressed'));
        btn.setAttribute('aria-pressed', 'true');
        loadHeatmap(btn.dataset.calMode);
      });
    });
  }

  // 9.N.10 · Compare-mode global toggle. Clicking flips filterState.compare
  // and re-runs the chart loaders. URL hash records `compare=1` so it sticks.
  function _bindCompareToggle() {
    const btn = document.getElementById('compare-toggle');
    if (!btn) return;
    const sync = () => {
      btn.classList.toggle('is-on', !!filterState.compare);
      btn.setAttribute('aria-pressed', filterState.compare ? 'true' : 'false');
      btn.title = filterState.compare ? 'Hide previous-period overlay' : 'Show previous-period overlay';
    };
    sync();
    btn.addEventListener('click', () => {
      filterState.compare = !filterState.compare;
      sync();
      applyFilters();
    });
  }

  // Per-endpoint detail feeds 4 of 6 charts. Fetch ONCE per refresh and fan out.
  let _perEndpointCache = null;
  async function loadPerEndpointDetail() {
    const qs = new URLSearchParams({
      range: filterState.range,
      limit: 12,
      spark_buckets: 20,
    });
    if (filterState.key_id)   qs.set('key_id',   filterState.key_id);
    if (filterState.env)      qs.set('env',      filterState.env);
    if (filterState.status)   qs.set('status',   filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    try {
      const { body } = await api('/api/account/usage/per-endpoint-detail?' + qs);
      if (!body || !body.ok) { _perEndpointCache = []; return; }
      _perEndpointCache = body.data.items || [];
    } catch (e) {
      _perEndpointCache = [];
    }
    // Fan out into 4 charts
    _renderTreemap();
    _renderSparkGrid();
    _renderBoxplot();
    _renderQuadrant();
  }

  function _renderTreemap() {
    const host = document.getElementById('treemap-host');
    if (!host) return;
    const items = (_perEndpointCache || []).map(it => ({
      label: it.label, value: it.count, error_rate: it.error_rate,
    }));
    APIN.charts.treemap(host, items, {
      height: 280,
      onClickTile: t => {
        filterState.endpoint = t.label;
        applyStateToUI();
        applyFilters();
      }
    });
  }

  function _renderSparkGrid() {
    const host = document.getElementById('sparkgrid-host');
    if (!host) return;
    const items = (_perEndpointCache || []).map(it => ({
      label: it.label, count: it.count, pct: it.pct,
      sparkline: it.sparkline || [], error_rate: it.error_rate,
    }));
    APIN.charts.sparkGrid(host, items, {
      onClickItem: t => {
        filterState.endpoint = t.label;
        applyStateToUI();
        applyFilters();
      }
    });
  }

  function _renderBoxplot() {
    const host = document.getElementById('boxplot-host');
    if (!host) return;
    const items = (_perEndpointCache || []).slice(0, 8).map(it => ({
      label: it.label,
      p10: it.p10, p25: it.p25, p50: it.p50, p75: it.p75, p90: it.p90,
      max: it.max, outliers: it.outliers || [],
    }));
    APIN.charts.boxplot(host, items, {
      xUnit: 'ms',
      onClickRow: t => {
        filterState.endpoint = t.label;
        applyStateToUI();
        applyFilters();
      }
    });
  }

  function _renderQuadrant() {
    const host = document.getElementById('quadrant-host');
    if (!host) return;
    const items = (_perEndpointCache || []).map(it => ({
      label: it.label,
      x_val: it.count,
      y_val: it.error_rate,
      size: it.p95,
      color_token: it.error_rate > 0.05 ? 'danger' : it.error_rate > 0 ? 'amber' : 'ok',
    }));
    APIN.charts.quadrant(host, items, {
      xLabel: 'volume', yLabel: 'error rate',
      yPct: true,
      xMid: 'median', yMid: 0.05,
      onClickPoint: t => {
        filterState.endpoint = t.label;
        applyStateToUI();
        applyFilters();
      }
    });
  }

  // 9.N.5.f · "What's noteworthy" — auto-derived insights from current data.
  // Picks up to 4 observations: highest-volume endpoint, error hot-spot,
  // slowest endpoint by p95, dormant key, anomalous status pattern.
  // Each insight is a short Fraunces+mono row with an icon, severity tone,
  // and a click-to-drill chip.
  // 9.N.6.f · Insights with richer derived narratives + EVERY row clickable.
  // Each insight has an explicit `action` object: filter-endpoint / filter-status /
  // expand-detail. The expand-detail action opens a per-insight modal explaining
  // the math behind the insight + giving deeper drill options.
  let _insightsList = [];
  function loadInsights() {
    const host = document.getElementById('insights-host');
    if (!host) return;
    if (!_perEndpointCache || _perEndpointCache.length === 0) {
      host.innerHTML = '<div class="placeholder" style="padding:24px;font-style:italic;color:var(--ink-soft,#6b6453)">no traffic in this window — nothing to surface</div>';
      _insightsList = [];
      return;
    }
    const insights = [];
    // 1. Busiest endpoint — always present
    const top = _perEndpointCache[0];
    if (top) {
      insights.push({
        tone: 'info', icon: 'i-chart-line',
        title: top.label + ' is your busiest endpoint',
        body: fmtNum(top.count) + ' requests · ' + Number(top.pct).toFixed(1) + '% of total · p50 ' + fmtNum(top.p50) + 'ms',
        narrative: 'This endpoint carries ' + Number(top.pct).toFixed(1) + '% of all traffic in the current window. ' +
                   'If it goes down, ' + Number(top.pct).toFixed(0) + '% of your users feel it.',
        action: { kind: 'filter-endpoint', value: top.label, label: 'filter to this endpoint' },
        item: top,
      });
    }
    // 2. Highest error rate endpoint
    const errs = [..._perEndpointCache].filter(it => it.error_rate > 0).sort((a, b) => b.error_rate - a.error_rate);
    if (errs.length > 0) {
      const e = errs[0];
      const tone = e.error_rate > 0.10 ? 'danger' : e.error_rate > 0.02 ? 'warn' : 'info';
      insights.push({
        tone: tone, icon: 'i-alert',
        title: e.label + ' has ' + (e.error_rate * 100).toFixed(1) + '% errors',
        body: fmtNum(e.errors) + ' of ' + fmtNum(e.count) + ' requests failed',
        narrative: e.error_rate > 0.05
          ? 'Error rate above 5% usually means a real bug — open the request drawer on a failed row to see exact status codes and bodies.'
          : 'Low background error rate — could be malformed client requests. Check the request drawer for any failed sample.',
        action: { kind: 'filter-endpoint', value: e.label, label: 'filter to this endpoint' },
        item: e,
      });
    }
    // 3. Slowest endpoint
    const slow = [..._perEndpointCache].sort((a, b) => (b.p95 || 0) - (a.p95 || 0))[0];
    if (slow && slow.p95 > 500) {
      const tone = slow.p95 > 2000 ? 'danger' : slow.p95 > 1000 ? 'warn' : 'info';
      insights.push({
        tone: tone, icon: 'i-gauge',
        title: slow.label + ' p95 latency is ' + fmtNum(slow.p95) + 'ms',
        body: 'p50 ' + fmtNum(slow.p50) + 'ms · p99 ' + fmtNum(slow.p99) + 'ms · max ' + fmtNum(slow.max) + 'ms',
        narrative: slow.p95 > 2000
          ? 'p95 above 2 seconds is over the 1.5s SLO budget — investigate the slowest 10 requests for the outlier shape.'
          : 'Within target band but trending up — keep an eye on p99/max.',
        action: { kind: 'filter-endpoint', value: slow.label, label: 'filter to this endpoint' },
        item: slow,
      });
    }
    // 4. Dormant keys
    const keysWithTraffic = (document.querySelector('.kpi[data-k="active_keys"] [data-num]')?.textContent || '').replace(/[^0-9]/g, '');
    const totalKeys = (document.querySelector('[data-breakdown-total]')?.textContent || '').replace(/[^0-9]/g, '');
    const kw = Number(keysWithTraffic) || 0, kt = Number(totalKeys) || 0;
    if (kt > kw && kt > 1) {
      const dormant = kt - kw;
      insights.push({
        tone: 'info', icon: 'i-key',
        title: dormant + ' of your ' + kt + ' keys are dormant',
        body: 'Only ' + kw + ' key' + (kw !== 1 ? 's' : '') + ' had traffic in this window',
        narrative: 'Unused keys are an attack surface. Consider rotating or revoking keys that have been inactive for over 30 days.',
        action: { kind: 'expand-detail', label: 'see dormant keys' },
      });
    }
    // 5. Traffic concentration insight (Pareto)
    if (_perEndpointCache.length >= 3) {
      const totalAll = _perEndpointCache.reduce((a, it) => a + (it.count || 0), 0);
      let cum = 0, n = 0;
      for (const it of _perEndpointCache) {
        cum += it.count || 0; n++;
        if (cum / totalAll >= 0.8) break;
      }
      if (n <= Math.max(2, _perEndpointCache.length * 0.3)) {
        insights.push({
          tone: 'info', icon: 'i-chart-bar',
          title: n + ' endpoints carry 80% of your traffic',
          body: 'Pareto distribution — focus your optimization on these',
          narrative: 'Out of ' + _perEndpointCache.length + ' endpoints with traffic, ' + n + ' of them ('
                   + ((n / _perEndpointCache.length) * 100).toFixed(0) + '%) account for 80%+ of all requests.',
          action: { kind: 'expand-detail', label: 'see the top ' + n },
        });
      }
    }
    _insightsList = insights;

    if (insights.length === 0) {
      host.innerHTML = '<div class="placeholder" style="padding:24px;font-style:italic;color:var(--ink-soft,#6b6453)">nothing unusual — operating cleanly</div>';
      return;
    }
    const toneColor = (t) => t === 'danger' ? 'var(--c-danger,#b01820)' : t === 'warn' ? 'var(--c-amber,#d49620)' : t === 'great' ? 'var(--c-ok,#2f6f3e)' : 'var(--ink-soft,#6b6453)';
    const rows = insights.map((ins, idx) => {
      const c = toneColor(ins.tone);
      return '<div class="ins-row" data-i="' + idx + '" style="display:grid;'
        + 'grid-template-columns:28px 1fr auto;gap:14px;align-items:center;'
        + 'padding:14px 18px;border-bottom:1px solid var(--paper-edge,#c7bca9);'
        + 'cursor:pointer;'   // 9.N.6.f · EVERY row clickable
        + 'transition:background .15s">'
        + '<svg viewBox="0 0 24 24" style="width:18px;height:18px;color:' + c + ';pointer-events:none" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><use href="#' + ins.icon + '"/></svg>'
        + '<div style="pointer-events:none">'   // pointer-events:none so click bubbles to row
        +   '<div style="font-family:Fraunces,serif;font-weight:500;font-size:14px;color:var(--ink,#1a1612)">' + escHtml(ins.title) + '</div>'
        +   '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11.5px;color:var(--ink-soft,#6b6453);margin-top:3px">' + escHtml(ins.body) + '</div>'
        + '</div>'
        + '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:' + c + ';white-space:nowrap;pointer-events:none">' + escHtml(ins.action.label) + ' →</div>'
        + '</div>';
    }).join('');
    host.innerHTML = '<div class="chart-insights">' + rows + '</div>';
    // Wire click handlers — ALL rows clickable
    host.querySelectorAll('.ins-row').forEach((row, i) => {
      const ins = insights[i];
      row.addEventListener('mouseenter', () => { row.style.background = 'var(--paper-deep, #e9e2d1)'; });
      row.addEventListener('mouseleave', () => { row.style.background = ''; });
      row.addEventListener('click', e => {
        if (window.APIN && APIN.fx) APIN.fx.ripple(row, e.clientX, e.clientY);
        if (ins.action.kind === 'filter-endpoint') {
          filterState.endpoint = ins.action.value;
          applyStateToUI(); applyFilters();
        } else if (ins.action.kind === 'filter-status') {
          filterState.status = ins.action.value;
          applyStateToUI(); applyFilters();
        } else if (ins.action.kind === 'expand-detail') {
          // Open a per-insight modal with full narrative + drill options
          if (window.APIN && APIN.lightbox) {
            APIN.lightbox.open({
              sourceCard: row,
              title: ins.title,
              subtitle: 'insight',
              hashKey: 'insight-' + i,
              build: (panel) => _lbInsightDetail(panel, ins),
            });
          }
        }
      });
    });
  }
  function _lbInsightDetail(panel, ins) {
    _appendSection(panel, 'Why this surfaced',
      '<p style="font-family:Fraunces,serif;font-size:14px;line-height:1.7;color:var(--ink);margin:0">'
      + escHtml(ins.narrative || ins.body) + '</p>');
    if (ins.item) {
      _appendSection(panel, 'Endpoint detail',
        _table([{label: 'Metric'}, {label: 'Value', numeric: true}], [
          ['Endpoint', ins.item.label],
          ['Volume', fmtNum(ins.item.count)],
          ['Errors', fmtNum(ins.item.errors)],
          ['Error rate', (ins.item.error_rate * 100).toFixed(2) + '%'],
          ['p50 latency', fmtNum(ins.item.p50) + 'ms'],
          ['p95 latency', fmtNum(ins.item.p95) + 'ms'],
          ['p99 latency', fmtNum(ins.item.p99) + 'ms'],
          ['Max latency', fmtNum(ins.item.max) + 'ms'],
          ['Bytes in',  fmtBytes(ins.item.bytes_in)],
          ['Bytes out', fmtBytes(ins.item.bytes_out)],
        ]));
    }
    const acts = [];
    if (ins.item && ins.item.label) acts.push({ kind: 'filter-endpoint', value: ins.item.label, icon: 'i-filter', label: 'filter dashboard to ' + ins.item.label });
    acts.push({ kind: 'export-csv', icon: 'i-download', label: 'export window CSV' });
    _appendSection(panel, 'Actions', _actionRow(acts));
    _wireActions(panel);
  }

  // ─── 9.N.4 · KPI sparklines (always last 24h, regardless of filterState.range)
  // Tile sparklines are STABLE baselines: they always show the last-24h shape so
  // the user can see "is today different from yesterday" while exploring any range.
  // Drawn into .kpi-spark[data-spark="KEY"] containers using inline SVG.
  // Caches the 24h timeseries on first call and reuses it for subsequent renders
  // (e.g. when filters change, we don't refetch the sparkline data).
  let _sparkCache24h = null;
  let _sparkLastFetchTs = 0;

  function renderSparkline(host, points, opts) {
    opts = opts || {};
    const W = host.clientWidth || 200, H = host.clientHeight || 32;
    if (!points || points.length < 2) {
      host.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
        '<text x="' + (W/2) + '" y="' + (H/2 + 3) + '" class="ks-empty">no prior 24h data</text>' +
        '</svg>';
      return;
    }
    // 9.N.5 fix · richer aesthetic — hatch-fill area + line + end-accent + axis hint.
    // Each sparkline gets a small italic Fraunces caption with the last-value
    // and the min-to-max range so the eye learns the shape AND the scale.
    const max = Math.max(1, ...points.map(p => p.v));
    const min = Math.min(...points.map(p => p.v));
    const padX = 4, padY = 5;
    const iw = W - padX * 2, ih = H - padY * 2;
    const x = i => padX + (i / Math.max(1, points.length - 1)) * iw;
    const y = v => padY + ih - (v / max) * ih;
    // Line path
    let d = 'M' + x(0).toFixed(1) + ',' + y(points[0].v).toFixed(1);
    for (let i = 1; i < points.length; i++) d += ' L' + x(i).toFixed(1) + ',' + y(points[i].v).toFixed(1);
    // Area path (closes to baseline for hatch fill)
    const baseY = padY + ih;
    const area = d + ' L' + x(points.length - 1).toFixed(1) + ',' + baseY.toFixed(1)
                   + ' L' + x(0).toFixed(1) + ',' + baseY.toFixed(1) + ' Z';
    const lastV = points[points.length - 1].v;
    const firstV = points[0].v;
    // Hatch pattern (unique per host to avoid id collisions in DOM)
    const patId = 'ks-h-' + (host.getAttribute('data-spark') || 'x') + '-' + Math.floor(Math.random()*1000);
    const svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
      '<defs>' +
        '<pattern id="' + patId + '" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(45)">' +
          '<line x1="0" y1="0" x2="0" y2="4" stroke="currentColor" stroke-width="0.6" opacity="0.32"/>' +
        '</pattern>' +
      '</defs>' +
      '<line class="ks-base" x1="0" y1="' + (H - padY).toFixed(1) + '" x2="' + W + '" y2="' + (H - padY).toFixed(1) + '"/>' +
      '<path d="' + area + '" fill="url(#' + patId + ')" stroke="none" style="color:var(--ink-soft,#6b6453)"/>' +
      '<path class="ks-line" d="' + d + '"/>' +
      '<circle class="ks-dot" cx="' + x(0).toFixed(1) + '" cy="' + y(firstV).toFixed(1) + '" r="1.4" opacity="0.6"/>' +
      // Accent end-dot: larger, opaque, with paper-color ring
      '<circle cx="' + x(points.length - 1).toFixed(1) + '" cy="' + y(lastV).toFixed(1) + '" r="3.4" fill="var(--paper,#fbf9f3)"/>' +
      '<circle cx="' + x(points.length - 1).toFixed(1) + '" cy="' + y(lastV).toFixed(1) + '" r="2.4" fill="var(--ink,#1a1612)"/>' +
      '</svg>';
    host.innerHTML = svg;
  }

  async function loadSparklines() {
    // Throttle: max once per 30 seconds (data doesn't change that fast)
    const now = Date.now();
    if (_sparkCache24h && (now - _sparkLastFetchTs < 30_000)) {
      _renderSparklinesFromCache();
      return;
    }
    try {
      const url = '/api/account/usage/timeseries?range=24h&mode=total';
      const { body } = await api(url);
      if (!body || !body.ok) return;
      _sparkCache24h = body.data;
      _sparkLastFetchTs = now;
      _renderSparklinesFromCache();
      // ALSO derive a latency sparkline by fetching mode=latency
      try {
        const u2 = '/api/account/usage/timeseries?range=24h&mode=latency';
        const r2 = await api(u2);
        if (r2.body && r2.body.ok) {
          _renderLatencySparkline(r2.body.data);
        }
      } catch (e) {}
      // bytes sparkline
      try {
        const u3 = '/api/account/usage/timeseries?range=24h&mode=bytes';
        const r3 = await api(u3);
        if (r3.body && r3.body.ok) {
          _renderBytesSparkline(r3.body.data);
        }
      } catch (e) {}
    } catch (e) {
      // Silent — sparklines are decorative
    }
  }

  function _renderSparklinesFromCache() {
    if (!_sparkCache24h) return;
    const buckets = _sparkCache24h.buckets || [];
    const meta = _sparkCache24h.series_meta || [];
    // Find series keys for 'total' / 'errors' / 'rate_limited' / 'active_keys'
    // The /timeseries?mode=total endpoint returns series: total + 5xx + 429
    const keyFor = label => {
      const m = meta.find(x => (x.key || '').toLowerCase() === label
        || (x.label || '').toLowerCase().includes(label));
      return m ? m.key : null;
    };
    const totalKey = keyFor('total') || (meta[0] && meta[0].key);
    const errKey   = keyFor('5xx') || keyFor('errors');
    const rlKey    = keyFor('429') || keyFor('rate');

    // requests sparkline = total series
    const reqEl = document.querySelector('.kpi-spark[data-spark="requests"]');
    if (reqEl && totalKey) {
      renderSparkline(reqEl, buckets.map(b => ({ t: b.t, v: Number((b.values||{})[totalKey] || 0) })));
    }
    // errors sparkline = 5xx + 4xx (combined). Fetch by_status if needed.
    // For now, fall back to errors series if present.
    const errEl = document.querySelector('.kpi-spark[data-spark="errors"]');
    if (errEl) {
      const errSeries = errKey
        ? buckets.map(b => ({ t: b.t, v: Number((b.values||{})[errKey] || 0) }))
        : [];
      renderSparkline(errEl, errSeries);
    }
    const rlEl = document.querySelector('.kpi-spark[data-spark="rate_limited"]');
    if (rlEl) {
      const rlSeries = rlKey
        ? buckets.map(b => ({ t: b.t, v: Number((b.values||{})[rlKey] || 0) }))
        : [];
      renderSparkline(rlEl, rlSeries);
    }
    // active_keys sparkline = total volume as proxy (no per-bucket key-count series)
    const keysEl = document.querySelector('.kpi-spark[data-spark="active_keys"]');
    if (keysEl && totalKey) {
      renderSparkline(keysEl, buckets.map(b => ({ t: b.t, v: Number((b.values||{})[totalKey] || 0) })));
    }
  }
  function _renderLatencySparkline(data) {
    const el = document.querySelector('.kpi-spark[data-spark="latency_p95_ms"]');
    if (!el || !data || !data.buckets) return;
    const meta = data.series_meta || [];
    const p95 = meta.find(m => (m.key || '').includes('p95') || (m.label || '').toLowerCase().includes('p95'));
    const k = p95 ? p95.key : (meta[0] && meta[0].key);
    if (!k) return;
    renderSparkline(el, data.buckets.map(b => ({ t: b.t, v: Number((b.values||{})[k] || 0) })));
  }
  function _renderBytesSparkline(data) {
    const el = document.querySelector('.kpi-spark[data-spark="bytes_out"]');
    if (!el || !data || !data.buckets) return;
    const meta = data.series_meta || [];
    const out = meta.find(m => (m.key || '').includes('bytes_out')) || meta[0];
    const k = out && out.key;
    if (!k) return;
    renderSparkline(el, data.buckets.map(b => ({ t: b.t, v: Number((b.values||{})[k] || 0) })));
  }

  // ─── Load summary ────────────────────────────────────────────────────
  async function loadSummary() {
    const qs = new URLSearchParams({ range: filterState.range });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env) qs.set('env', filterState.env);
    if (filterState.status) qs.set('status', filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    const { body } = await api('/api/account/usage/summary?' + qs);
    if (!body || !body.ok) return;
    if (body.data.lifetime) _lastLifetime = body.data.lifetime;   // 9.N.T29
    const k = body.data.kpis;
    _lastWinRequests = (k.requests && k.requests.current) || 0;   // 9.N.T29 cross-metric hint
    const prevLbl = 'vs prev ' + filterState.range;
    setKpi('requests', fmtNum(k.requests.current), k.requests.delta_pct, prevLbl);
    setKpi('errors', fmtNum(k.errors.current), k.errors.delta_pct, prevLbl, { goodIfDown: true });
    setKpi('rate_limited', fmtNum(k.rate_limited.current), k.rate_limited.delta_pct, prevLbl, { goodIfDown: true });
    setKpi('latency_p95_ms',
      (k.latency_p95_ms.current != null ? Math.round(k.latency_p95_ms.current) : '·'),
      k.latency_p95_ms.delta_pct, prevLbl, { goodIfDown: true });
    setKpi('active_keys', String(k.active_keys.current || 0), k.active_keys.delta_pct, prevLbl);
    // Bytes-out is text-only (with units); manual set
    const byEl = document.querySelector('.kpi[data-k="bytes_out"] [data-bytes-out]');
    if (byEl) byEl.textContent = fmtBytes(k.bytes_out.current);
    const byTile = document.querySelector('.kpi[data-k="bytes_out"]');
    if (byTile) {
      const dEl2 = byTile.querySelector('[data-delta]');
      const pEl2 = byTile.querySelector('[data-prev]');
      if (dEl2) {
        if (k.bytes_out.delta_pct == null) {
          dEl2.className = 'delta none'; dEl2.innerHTML = 'no prev data';
          if (pEl2) { pEl2.textContent = ''; pEl2.style.display = 'none'; }
        } else {
          const up = k.bytes_out.delta_pct > 0;
          dEl2.className = 'delta ' + (up ? 'up' : 'down');
          dEl2.innerHTML = '<svg class="icon" aria-hidden="true" style="width:11px;height:11px"><use href="#' +
            (up ? 'i-arrow-up-right' : 'i-arrow-down-right') + '"/></svg> ' +
            Math.abs(k.bytes_out.delta_pct).toFixed(1) + '%';
          if (pEl2) { pEl2.textContent = prevLbl; pEl2.style.display = ''; }
        }
      } else if (pEl2) {
        pEl2.textContent = prevLbl;
      }
    }

    // Sub-line breakdowns
    const b5 = document.querySelector('[data-breakdown-5xx]');
    const b4 = document.querySelector('[data-breakdown-4xx]');
    if (b5) b5.textContent = fmtNum(k.errors_5xx ? k.errors_5xx.current : 0);
    if (b4) b4.textContent = fmtNum(k.errors_4xx ? k.errors_4xx.current : 0);
    const p50 = document.querySelector('[data-breakdown-p50]');
    const p99 = document.querySelector('[data-breakdown-p99]');
    if (p50) p50.textContent = k.latency_p50_ms.current != null ? Math.round(k.latency_p50_ms.current) + 'ms' : '·';
    if (p99) p99.textContent = k.latency_p99_ms.current != null ? Math.round(k.latency_p99_ms.current) + 'ms' : '·';
    const tk = document.querySelector('[data-breakdown-total]');
    if (tk) tk.textContent = String((k.total_keys && k.total_keys.current) || 0);
  }

  // ─── Time-series ─────────────────────────────────────────────────────
  async function loadTimeseries() {
    const qs = new URLSearchParams({ range: filterState.range, mode: filterState.mode });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env) qs.set('env', filterState.env);
    if (filterState.status) qs.set('status', filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    // 9.N.10 · Compare-mode forward
    if (filterState.compare) qs.set('compare', 'prev_period');
    const host = $('ts-host');
    const { body } = await api('/api/account/usage/timeseries?' + qs);
    if (!body || !body.ok) {
      host.innerHTML = '<div class="placeholder">failed to load</div>';
      return;
    }
    const payload = body.data;
    const titleMap = {
      total: 'Requests', by_status: 'Requests by status',
      by_endpoint: 'Requests by endpoint', latency: 'Latency (ms)',
      errors: 'Errors (4xx + 5xx + 429)', bytes: 'Bytes in / out',
    };
    $('ts-title').textContent = (titleMap[filterState.mode] || 'Requests') + ' · ' + filterState.range;

    // 9.N.T29 · empty window → seed→sprout hero (replaces the chart)
    const _buckets = (payload && payload.buckets) || [];
    const _isEmpty = _buckets.length === 0 || _buckets.every(b => {
      const v = (b && b.values) || {}; return Object.keys(v).every(kk => !v[kk]);
    });
    const _scrub = $('ts-scrubber');
    if (_isEmpty && window.APIN && APIN.usageEmpty) {
      if (_scrub) _scrub.style.display = 'none';
      await _renderUsageEmpty(host);
      return;
    }
    if (_scrub) _scrub.style.display = '';
    _heroActive = false;
    if (_emptyHero) { try { _emptyHero.destroy(); } catch (_) {} _emptyHero = null; }

    const chartMode = (filterState.mode === 'by_status' || filterState.mode === 'by_endpoint')
      ? 'stacked' : 'line';
    APIN.charts.timeseries(host, payload, {
      mode: chartMode,
      logY: filterState.logY,
      height: 320,
      showLegend: true,
      onClickBucket: bucket => openMinuteDrawer(bucket.t),
      onContextBucket: (bucket, ev, items) => {
        items.push({
          label: 'Drill into this minute',
          onClick: () => openMinuteDrawer(bucket.t),
        });
      },
      onZoom: (lo, hi) => {
        // 9.I — drag-to-zoom drills to the *start* of the selected range
        // for now; full range-zoom on the time-series itself would change
        // filterState.range to a custom mode (filed for 9.K).
        openMinuteDrawer(lo.t);
      },
    });
  }

  // ─── 9.N.T29 · empty-state hero helpers ───────────────────────────────
  function _keyName(kid) {
    const k = (availableKeys || []).find(x => (x.public_id || x.id) === kid);
    return k ? (k.name || kid) : kid;
  }
  function _activeFilterChips() {
    const c = [];
    if (filterState.key_id) c.push('key = ' + _keyName(filterState.key_id));
    if (filterState.env) c.push('env = ' + filterState.env);
    if (filterState.status) c.push('status = ' + filterState.status);
    if (filterState.endpoint) c.push('endpoint contains "' + filterState.endpoint + '"');
    return c;
  }
  async function _fetchAccountLifetime() {
    if (_accountLifetime) return _accountLifetime;
    try {
      const { body } = await api('/api/account/usage/summary?range=' + filterState.range);
      _accountLifetime = (body && body.ok) ? (body.data.lifetime || null) : null;
    } catch (_) {}
    return _accountLifetime;
  }
  function _usageTestPing(btn) {
    // reuse a bearer token the user already pasted in the Sandbox this session
    let token = null;
    try {
      for (let i = 0; i < sessionStorage.length; i++) {
        const k = sessionStorage.key(i);
        if (k && k.indexOf('sandbox_bearer_for_') === 0) { token = sessionStorage.getItem(k); if (token) break; }
      }
    } catch (_) {}
    if (!token) {                       // none cached → Sandbox is where auth lives
      if (btn) btn.textContent = 'opening Sandbox…';
      setTimeout(() => { location.href = '/account/api/sandbox'; }, 500);
      return;
    }
    if (!btn) return;
    const orig = btn.textContent; btn.disabled = true; btn.textContent = 'pinging…';
    fetch('/api/version', { headers: { 'Authorization': 'Bearer ' + token, 'Accept': 'application/json' } })
      .then(r => { btn.textContent = 'sent · ' + r.status + ' — watch it appear'; })
      .catch(() => { btn.textContent = 'ping failed'; })
      .finally(() => { setTimeout(() => { btn.disabled = false; btn.textContent = orig; }, 2600); });
  }
  async function _renderUsageEmpty(host) {
    if (_emptyHero) { try { _emptyHero.destroy(); } catch (_) {} _emptyHero = null; }
    let lt = _lastLifetime;
    if (!lt) {   // race: summary not back yet — fetch lifetime directly (only on empty)
      try {
        const qs = new URLSearchParams({ range: filterState.range });
        if (filterState.key_id) qs.set('key_id', filterState.key_id);
        if (filterState.env) qs.set('env', filterState.env);
        const { body } = await api('/api/account/usage/summary?' + qs);
        lt = (body && body.ok) ? (body.data.lifetime || null) : null; _lastLifetime = lt;
      } catch (_) {}
    }
    const contentFilters = !!(filterState.endpoint || filterState.status);
    const scopeFilters = !!(filterState.key_id || filterState.env);
    let state;
    if (contentFilters) state = 'filtered';
    else if (!lt || !lt.ever) state = 'new';
    else state = 'dormant';
    const rangeLabelMap = { '15m': 'the last 15 minutes', '1h': 'the last hour', '6h': 'the last 6 hours', '24h': 'the last 24 hours', '7d': 'the last 7 days', '30d': 'the last 30 days' };
    const metricLabelMap = { total: 'request', by_status: 'status', by_endpoint: 'endpoint', latency: 'latency', errors: 'error', bytes: 'byte' };
    const ctx = {
      state: state, range: filterState.range, rangeLabel: rangeLabelMap[filterState.range] || 'this window',
      lifetime: lt, ranges: RANGE_LADDER, now: Date.now(),
      filtersActive: contentFilters || scopeFilters,
      filterChips: _activeFilterChips(),
      sampleCurl: 'curl -H "Authorization: Bearer apin_live_…" ' + location.origin + '/api/version',
      docsUrl: '/docs', quickstartUrl: '/account/api/quickstart', sandboxUrl: '/account/api/sandbox',
      onSetRange: (id) => { filterState.range = id; recentCursor = null; if (typeof applyStateToUI === 'function') applyStateToUI(); applyFilters(); },
      onClearFilters: () => { filterState.key_id = ''; filterState.env = ''; filterState.status = ''; filterState.endpoint = ''; recentCursor = null; if (typeof applyStateToUI === 'function') applyStateToUI(); applyFilters(); },
      onSetMetric: (m) => { filterState.mode = (m === 'requests' ? 'total' : m); if (typeof applyStateToUI === 'function') applyStateToUI(); loadTimeseries(); },
      onOpenRequest: (rid) => { if (window.APIN && APIN.requestDrawer) APIN.requestDrawer.open(rid); else if (typeof openRequestDetail === 'function') openRequestDetail(rid); },
      onTestPing: (btn) => _usageTestPing(btn),
    };
    if (state === 'dormant' && filterState.mode !== 'total' && _lastWinRequests > 0) {
      ctx.metricEmptyButOthers = { metric: metricLabelMap[filterState.mode] || 'these', other: 'requests', otherCount: _lastWinRequests };
    }
    // key-scope hint: filtered to an unused key while the account has traffic
    if (state === 'new' && filterState.key_id) {
      const acct = await _fetchAccountLifetime();
      if (acct && acct.ever) { ctx.state = 'dormant'; ctx.keyScope = { name: _keyName(filterState.key_id), accountTotal: acct.total }; ctx.lifetime = { ever: false }; }
    }
    _heroActive = true;
    _emptyHero = APIN.usageEmpty.hero(host, ctx);
  }

  // ─── Top-N panels ────────────────────────────────────────────────────
  async function loadTop(dim, hostId, auxId, opts) {
    opts = opts || {};
    // 9.N.6 fix · pass ALL filter state — top panels were ignoring
    // status/endpoint, so a "drill into 4xx" left them showing all traffic.
    const qs = new URLSearchParams({ range: filterState.range, dim, limit: 10 });
    if (filterState.key_id)   qs.set('key_id',   filterState.key_id);
    if (filterState.env)      qs.set('env',      filterState.env);
    if (filterState.status)   qs.set('status',   filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    const { body } = await api('/api/account/usage/top?' + qs);
    if (!body || !body.ok) return;
    const items = body.data.items || [];
    if (auxId) {
      const el = $(auxId);
      if (el) el.textContent = items.length + ' / ' + fmtNum(body.data.total_for_pct);
    }
    if (!items.length && window.APIN && APIN.usageEmpty) {   // 9.N.T29 ghost teaser
      APIN.usageEmpty.ghost($(hostId), opts.ghost || 'rows'); return;
    }
    APIN.charts.topBar($(hostId), items, {
      color_token: opts.color_token || 'series-0',
      // 9.I · I2 — clicking any row cascades a filter
      onClickItem: it => {
        if (dim === 'endpoints') {
          filterState.endpoint = it.path || it.label || '';
          if ($('filter-endpoint')) $('filter-endpoint').value = filterState.endpoint;
        } else if (dim === 'keys') {
          filterState.key_id = it.public_id || '';
          if ($('filter-key')) $('filter-key').value = filterState.key_id;
        } else if (dim === 'statuses') {
          // Map raw status code to status class shorthand
          const s = String(it.status || it.label || '');
          let mapped = s;
          if (/^2/.test(s)) mapped = '2xx';
          else if (/^3/.test(s)) mapped = '3xx';
          else if (s === '429') mapped = '429';
          else if (/^4/.test(s)) mapped = '4xx';
          else if (/^5/.test(s)) mapped = '5xx';
          filterState.status = mapped;
          if ($('filter-status')) $('filter-status').value = mapped;
        } else if (dim === 'error_codes') {
          // No direct filter for error_code — drill via the request table
          // by filtering on the associated status bucket.
          const ss = (it.extra && it.extra.sample_status) || null;
          if (ss) {
            const v = ss === 429 ? '429' :
                      (ss >= 500 ? '5xx' :
                      (ss >= 400 ? '4xx' : ''));
            if (v) {
              filterState.status = v;
              if ($('filter-status')) $('filter-status').value = v;
            }
          }
        } else if (dim === 'ips') {
          // No IP filter param — open a drawer scoped to this IP
          openIpDrawer(it.label);
          return;
        }
        applyFilters();
      },
    });
  }

  async function loadDonut() {
    // 9.H · D1 — use the same gran→bucket math but read the SUMMARY for
    // exact totals (single source of truth, no drift). The donut still
    // shows the *breakdown*, computed from a coarse-grain timeseries pull.
    const granForRange = {
      '15m': '1m', '1h': '5m', '6h': '15m', '24h': '1h',
      '7d': '6h', '30d': '1d',
    };
    const qs = new URLSearchParams({
      range: filterState.range, mode: 'by_status',
      granularity: granForRange[filterState.range] || '1h',
    });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env) qs.set('env', filterState.env);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    const { body } = await api('/api/account/usage/timeseries?' + qs);
    if (!body || !body.ok) return;
    const totals = { '2xx': 0, '3xx': 0, '4xx': 0, '429': 0, '5xx': 0 };
    for (const b of (body.data.buckets || [])) {
      for (const k of Object.keys(totals)) totals[k] += (b.values && b.values[k]) || 0;
    }
    const items = [
      { label: '2xx', value: totals['2xx'], color_token: 'ok' },
      { label: '3xx', value: totals['3xx'], color_token: 'info' },
      { label: '4xx', value: totals['4xx'], color_token: 'warn' },
      { label: '429', value: totals['429'], color_token: 'amber' },
      { label: '5xx', value: totals['5xx'], color_token: 'danger' },
    ].filter(x => x.value > 0);
    const total = items.reduce((a, x) => a + x.value, 0);
    if ($('donut-aux')) $('donut-aux').textContent = fmtNum(total) + ' requests';
    if (!items.length && window.APIN && APIN.usageEmpty) {   // 9.N.T29 ghost teaser
      APIN.usageEmpty.ghost($('donut-host'), 'donut'); return;
    }
    APIN.charts.donut($('donut-host'), items, {
      size: 200, innerRatio: 0.62, totalLabel: 'reqs',
      onClickSlice: it => {
        filterState.status = it.label;
        if ($('filter-status')) $('filter-status').value = it.label;
        applyFilters();
      },
    });
  }

  async function loadLatencyHistogram() {
    // 9.H · D6 fix.
    const qs = new URLSearchParams({ range: filterState.range, limit: 200 });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env) qs.set('env', filterState.env);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    if (filterState.status) qs.set('status', filterState.status);
    const { body } = await api('/api/account/usage/requests?' + qs);
    if (!body || !body.ok) {
      $('hist-host').innerHTML = '<div class="placeholder">failed to load</div>';
      return;
    }
    const lats = (body.data.items || []).map(x => Number(x.latency_ms)).filter(x => x > 0);
    if (lats.length === 0) {
      if (window.APIN && APIN.usageEmpty) APIN.usageEmpty.ghost($('hist-host'), 'bars');   // 9.N.T29
      else $('hist-host').innerHTML = '<div class="placeholder">no latency data in this window</div>';
      return;
    }
    const edges = [0, 50, 100, 200, 500, 1000, 2000, 5000, Infinity];
    const labels = ['<50', '50-100', '100-200', '200-500', '500-1k', '1-2k', '2-5k', '5k+'];
    const buckets = labels.map((l, i) => ({
      label: l,
      value: lats.filter(v => v >= edges[i] && v < edges[i + 1]).length,
      hint: edges[i] + '-' + (edges[i + 1] === Infinity ? '∞' : edges[i + 1]) + ' ms',
      color_token: i >= 6 ? 'danger' : (i >= 4 ? 'warn' : (i >= 2 ? 'info' : 'ok')),
      _min: edges[i],
      _max: edges[i + 1] === Infinity ? null : edges[i + 1],
    }));
    APIN.charts.histogram($('hist-host'), buckets, {
      height: 220,
      // 9.I · I3 — bar click → drill drawer
      onClickBar: b => openLatencyDrawer(b._min, b._max, b.label),
    });
    $('hist-aux').textContent = lats.length + ' samples';
  }

  // ─── Recent requests table ───────────────────────────────────────────
  // 9.N.8e · Recent requests now keeps a CLIENT-SIDE DATA MODEL so we can:
  //   · sort by any column (click header) without re-fetching
  //   · prepend live SSE events without losing the manual page
  //   · re-render efficiently on sort change
  // `_recentItems` is the source of truth; the DOM is derived from it.
  const _recentSort = { key: 'timestamp', dir: 'desc' };
  let _recentItems = [];
  let _recentLiveOn = true;

  function _renderRecentRow(it, fresh) {
    return '<tr class="row-clickable' + (fresh ? ' recent-row-fresh' : '') + '" data-rid="' + escHtml(it.id) + '">' +
      '<td title="' + escHtml(it.timestamp || '') + '">' + fmtAgo(it.timestamp) + '</td>' +
      '<td>' + escHtml(it.key_name || it.key_public_id || '·') +
        (it.env ? ' <span style="color:var(--ink-mute);font-size:10.5px">' + escHtml(it.env) + '</span>' : '') +
      '</td>' +
      '<td><span class="meth ' + methClass(it.method) + '">' + escHtml(it.method || '') + '</span></td>' +
      '<td>' + escHtml(it.path || '') + '</td>' +
      '<td><span class="stat ' + statusBucket(it.status_code) + '">' + (it.status_code || '') + '</span></td>' +
      '<td>' + (it.error_code ? '<span class="err">' + escHtml(it.error_code) + '</span>' : '') + '</td>' +
      '<td style="text-align:right">' + (it.latency_ms != null ? it.latency_ms + ' ms' : '·') + '</td>' +
      '<td style="text-align:right">' + (it.bytes_out != null ? fmtBytes(it.bytes_out) : '·') + '</td>' +
      '<td>' + escHtml(it.ip || '·') + '</td>' +
    '</tr>';
  }

  function _sortItemsInPlace(items, key, dir) {
    const mul = dir === 'asc' ? 1 : -1;
    items.sort((a, b) => {
      const av = a[key], bv = b[key];
      // Nulls always last regardless of direction (cleaner UX)
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * mul;
      return String(av).localeCompare(String(bv)) * mul;
    });
  }

  function _renderRecentBody() {
    const tbody = $('recent-tbody');
    if (!tbody) return;
    if (_recentItems.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="placeholder">no requests in this window</td></tr>';
      return;
    }
    tbody.innerHTML = _recentItems.map(it => _renderRecentRow(it, false)).join('');
    _wireRecentRowClicks(tbody);
    const auxEl = $('recent-aux');
    if (auxEl) auxEl.textContent = _recentItems.length + ' rows';
    _updateSortHeaderUI();
  }

  function _wireRecentRowClicks(scope) {
    scope.querySelectorAll('tr.row-clickable').forEach(tr => {
      tr.style.cursor = 'pointer';
      // Use a single delegated-style listener-per-row but only once
      if (tr._wired) return;
      tr._wired = true;
      tr.addEventListener('click', () => {
        const rid = tr.getAttribute('data-rid');
        if (rid) openRequestDetail(rid);
      });
    });
  }

  function _updateSortHeaderUI() {
    document.querySelectorAll('#recent-tbl thead th[data-sort-key]').forEach(th => {
      if (th.getAttribute('data-sort-key') === _recentSort.key) {
        th.setAttribute('data-sort-active', _recentSort.dir);
      } else {
        th.removeAttribute('data-sort-active');
      }
    });
  }

  async function loadRecent(opts) {
    opts = opts || {};
    const qs = new URLSearchParams({ range: filterState.range, limit: 50 });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env) qs.set('env', filterState.env);
    if (filterState.status) qs.set('status', filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    if (opts.append && recentCursor) qs.set('cursor', String(recentCursor));
    const { body } = await api('/api/account/usage/requests?' + qs);
    if (!body || !body.ok) return;
    const items = body.data.items || [];
    if (opts.append) {
      // Append while preserving sort
      _recentItems.push(...items);
    } else {
      _recentItems = items.slice();
    }
    _sortItemsInPlace(_recentItems, _recentSort.key, _recentSort.dir);
    _renderRecentBody();
    recentCursor = items.length > 0 ? items[items.length - 1].id : null;
    $('recent-pager').hidden = items.length < 50;
  }

  // ── Live SSE integration ──────────────────────────────────────────
  // Subscribe to the same live-stream feed the Live Tail uses; when an
  // event arrives, build a "row-like" object and prepend it to the data
  // model + DOM with a wet-ink animation. Throttle re-renders via rAF so
  // bursts don't thrash layout.
  let _pendingLiveAdds = [];
  let _liveFlushScheduled = false;
  function _scheduleLiveFlush() {
    if (_liveFlushScheduled) return;
    _liveFlushScheduled = true;
    // 9.N.8e · setTimeout instead of requestAnimationFrame.
    // rAF is SUSPENDED when the tab is hidden (visibilitychange "hidden"
    // state) — so live updates would queue up indefinitely while the user
    // is on another tab, then dump in a stampede when they return. With
    // setTimeout we keep updating regardless of visibility, and the 60ms
    // delay still coalesces bursts (≥16 events/s would still feel smooth).
    setTimeout(_flushLiveAdds, 60);
  }
  function _flushLiveAdds() {
    _liveFlushScheduled = false;
    if (_pendingLiveAdds.length === 0) return;
    const tbody = $('recent-tbody');
    if (!tbody) { _pendingLiveAdds.length = 0; return; }
    // Build all new rows in a fragment for ONE reflow.
    const frag = document.createDocumentFragment();
    // Process oldest-first so newest ends up at very top after prepend.
    const adds = _pendingLiveAdds.slice();
    _pendingLiveAdds.length = 0;
    for (const it of adds) {
      // Avoid dup if the live event raced with a fetch
      if (_recentItems.some(x => String(x.id) === String(it.id))) continue;
      _recentItems.unshift(it);
      const tmp = document.createElement('tbody');
      tmp.innerHTML = _renderRecentRow(it, true);
      const tr = tmp.firstElementChild;
      if (tr) frag.appendChild(tr);
    }
    // Only prepend when sort is descending by timestamp — otherwise we
    // need to re-sort the whole table (the live event may belong in a
    // different position by sort key).
    const livePrepend = (_recentSort.key === 'timestamp' && _recentSort.dir === 'desc');
    if (livePrepend && frag.children.length > 0) {
      tbody.insertBefore(frag, tbody.firstChild);
      // Cap visible rows to avoid unbounded growth
      while (tbody.children.length > 200) {
        tbody.lastElementChild.remove();
        _recentItems.pop();
      }
      _wireRecentRowClicks(tbody);
    } else {
      _sortItemsInPlace(_recentItems, _recentSort.key, _recentSort.dir);
      _renderRecentBody();
    }
    const auxEl = $('recent-aux');
    if (auxEl) auxEl.textContent = _recentItems.length + ' rows';
  }

  // 9.N.8g · Cache of recent live SSE events keyed by the synthetic row id
  // we assign at receive time. openRequestDetail consults this when the
  // server returns 404 (row hasn't been flushed from the buffer to the DB
  // yet) so the drawer can render the basic envelope from the live data
  // instead of "failed to load request detail".
  const _liveEventCache = new Map();   // id -> { event, capturedAt }
  const _LIVE_CACHE_LIMIT = 200;
  function _cacheLiveEvent(id, ev) {
    _liveEventCache.set(id, { event: ev, capturedAt: Date.now() });
    if (_liveEventCache.size > _LIVE_CACHE_LIMIT) {
      // FIFO eviction — oldest first
      const oldest = _liveEventCache.keys().next().value;
      _liveEventCache.delete(oldest);
    }
  }

  // 9.N.8g · key_id -> key_name lookup. Populated from availableKeys
  // (loaded once at page boot via /api/account/keys). Used to resolve the
  // key column on SSE events that lack key_name (older Space deploys).
  function _resolveKeyName(ev) {
    if (ev.key_name) return ev.key_name;
    const ks = window.availableKeys || [];
    const kid = ev.key_id || ev.key_public_id;
    if (!kid) return null;
    const match = ks.find(k => k.public_id === kid);
    return match ? (match.name || null) : null;
  }

  // 9.N.T29 · when a live request arrives while the empty hero is shown,
  // reload once (debounced) so the sprout dissolves into the real charts.
  let _heroReactT = null;
  function _scheduleHeroReact() {
    if (_heroReactT) return;
    _heroReactT = setTimeout(() => {
      _heroReactT = null;
      if (_heroActive) { _heroActive = false; if (typeof applyFilters === 'function') applyFilters(); }
    }, 900);
  }
  function _subscribeRecentToLiveStream() {
    // The live-pulse accumulator + live-stream client both publish to the
    // shared bus. We hook directly into the accumulator's subscribe API
    // so we get every event regardless of which widget renders it.
    const accum = window.APIN && APIN.livePulseData && APIN.livePulseData.accumulator;
    if (!accum || !accum.subscribe) return null;
    return accum.subscribe((ev, ts) => {
      if (_heroActive) _scheduleHeroReact();   // 9.N.T29 · first request dissolves the empty hero
      if (!_recentLiveOn) return;
      // 9.N.8g · Generate synthetic id once (was inside object literal,
      // which made it impossible to use the same id as the cache key).
      const id = ev.id || ('live-' + ts + '-' + Math.random().toString(36).slice(2, 7));
      const item = {
        id:           id,
        timestamp:    ev.timestamp || new Date().toISOString().replace('T', ' '),
        method:       ev.method || 'GET',
        path:         ev.path || '/',
        status_code:  ev.status_code || 0,
        error_code:   ev.error_code || null,
        latency_ms:   ev.latency_ms != null ? Number(ev.latency_ms) : null,
        bytes_out:    ev.bytes_out != null ? Number(ev.bytes_out) : null,
        ip:           ev.ip || null,
        key_name:     _resolveKeyName(ev),         // 9.N.8g · use availableKeys map
        key_public_id:ev.key_id || ev.key_public_id || null,
        env:          ev.env || null,
      };
      _cacheLiveEvent(id, item);                    // 9.N.8g · cache for drawer fallback
      _pendingLiveAdds.push(item);
      _scheduleLiveFlush();
    });
  }
  // Expose the cache + resolver so openRequestDetail can read them.
  window.APIN = window.APIN || {};
  window.APIN._liveEventCache = _liveEventCache;

  function _wireRecentControls() {
    // Sortable column headers
    document.querySelectorAll('#recent-tbl thead th[data-sort-key]').forEach(th => {
      th.addEventListener('click', () => {
        const key = th.getAttribute('data-sort-key');
        if (_recentSort.key === key) {
          _recentSort.dir = (_recentSort.dir === 'asc') ? 'desc' : 'asc';
        } else {
          _recentSort.key = key;
          // Default direction depends on column — timestamps + numbers
          // start descending (newest/biggest first), strings ascending.
          _recentSort.dir = (key === 'timestamp' || key === 'latency_ms' ||
                               key === 'bytes_out' || key === 'status_code')
            ? 'desc' : 'asc';
        }
        _sortItemsInPlace(_recentItems, _recentSort.key, _recentSort.dir);
        _renderRecentBody();
      });
    });
    // Refresh button
    const refresh = document.getElementById('recent-refresh');
    if (refresh) {
      refresh.addEventListener('click', () => {
        refresh.classList.add('is-spinning');
        setTimeout(() => refresh.classList.remove('is-spinning'), 750);
        recentCursor = null;
        loadRecent({ append: false });
      });
    }
    // Live toggle
    const liveBtn = document.getElementById('recent-live-toggle');
    if (liveBtn) {
      liveBtn.addEventListener('click', () => {
        _recentLiveOn = !_recentLiveOn;
        liveBtn.setAttribute('data-on', _recentLiveOn ? 'true' : 'false');
        liveBtn.querySelector('.recent-live-label').textContent =
          _recentLiveOn ? 'live' : 'paused';
      });
    }
  }

  // ─── Drawer plumbing ─────────────────────────────────────────────────
  function openDrawer() {
    $('drawer').classList.add('show');
    $('drawer-mask').classList.add('show');
    $('drawer').setAttribute('aria-hidden', 'false');
  }
  function closeDrawer() {
    $('drawer').classList.remove('show');
    $('drawer-mask').classList.remove('show');
    $('drawer').setAttribute('aria-hidden', 'true');
  }
  function openReqd() {
    $('reqd').classList.add('show');
    $('reqd-mask').classList.add('show');
    $('reqd').setAttribute('aria-hidden', 'false');
  }
  function closeReqd() {
    $('reqd').classList.remove('show');
    $('reqd-mask').classList.remove('show');
    $('reqd').setAttribute('aria-hidden', 'true');
  }

  // ─── Minute drill drawer (richer than before) ─────────────────────────
  async function openMinuteDrawer(minuteTs) {
    openDrawer();
    $('drawer-title').textContent = 'Minute · ' + minuteTs;
    $('drawer-sub').textContent = 'UTC · click any row to see the request detail';
    $('drawer-body').innerHTML = '<div class="placeholder">loading&hellip;</div>';
    const qs = new URLSearchParams({ minute_ts: minuteTs });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    if (filterState.status) qs.set('status', filterState.status);
    const { body } = await api('/api/account/usage/minute-detail?' + qs);
    if (!body || !body.ok) {
      $('drawer-body').innerHTML = '<div class="placeholder">failed to load minute detail</div>';
      return;
    }
    const agg = body.data.aggregate || {};
    const rows = body.data.requests || [];

    // Mini-KPI strip
    const kpiHtml =
      '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px">' +
        kpiBlock('Requests', fmtNum(agg.requests)) +
        kpiBlock('Errors',   fmtNum(agg.errors), agg.errors ? 'danger' : null) +
        kpiBlock('p50 ms',   (agg.p50 != null ? Math.round(agg.p50) : '·')) +
        kpiBlock('p99 ms',   (agg.p99 != null ? Math.round(agg.p99) : '·')) +
      '</div>';

    // Top-3 endpoints in this minute (computed client-side from rows)
    const epCounts = {};
    for (const r of rows) {
      const p = r.path || '?';
      epCounts[p] = (epCounts[p] || 0) + 1;
    }
    const top3 = Object.entries(epCounts).sort((a,b) => b[1] - a[1]).slice(0, 3);
    let topHtml = '';
    if (top3.length) {
      topHtml = '<div class="reqd-section"><h4>Top endpoints in this minute</h4>' +
        top3.map(([p, n]) =>
          '<div style="display:flex;justify-content:space-between;padding:3px 0;font-family:\'JetBrains Mono\',monospace;font-size:12px">' +
            '<span>' + escHtml(p) + '</span>' +
            '<b>' + n + '</b>' +
          '</div>'
        ).join('') + '</div>';
    }

    // Mini-donut by status
    const statusCounts = {};
    for (const r of rows) {
      const c = Number(r.status_code) || 0;
      const k = c >= 500 ? '5xx' :
                c === 429 ? '429' :
                c >= 400 ? '4xx' :
                c >= 300 ? '3xx' :
                c >= 200 ? '2xx' : '?';
      statusCounts[k] = (statusCounts[k] || 0) + 1;
    }
    let donutHtml = '';
    if (Object.keys(statusCounts).length > 0) {
      donutHtml = '<div class="reqd-section"><h4>Status mix</h4>' +
        '<div style="display:flex;flex-wrap:wrap;gap:8px">' +
          Object.entries(statusCounts).map(([k, v]) =>
            '<span class="stat ' + (k === '5xx' ? 'stat-5xx' :
                                     k === '4xx' || k === '429' ? 'stat-4xx' :
                                     k === '3xx' ? 'stat-3xx' :
                                     k === '2xx' ? 'stat-2xx' : '') + '">' +
              k + ' · ' + v +
            '</span>'
          ).join('') +
        '</div></div>';
    }

    const reqHtml = rows.length === 0
      ? '<div class="placeholder">no individual requests captured in this minute</div>'
      : '<div class="reqd-section"><h4>Requests (' + rows.length + ')</h4>' +
        '<table class="tbl"><thead><tr><th>time</th><th>method</th><th>path</th><th>status</th><th>ms</th></tr></thead>' +
        '<tbody>' + rows.map(it =>
          '<tr class="row-clickable" data-rid="' + escHtml(it.id) + '" style="cursor:pointer">' +
          '<td title="' + escHtml(it.timestamp) + '">' + escHtml((it.timestamp || '').slice(11, 19)) + '</td>' +
          '<td><span class="meth ' + methClass(it.method) + '">' + escHtml(it.method || '') + '</span></td>' +
          '<td>' + escHtml(it.path || '') + '</td>' +
          '<td><span class="stat ' + statusBucket(it.status_code) + '">' + (it.status_code || '') + '</span></td>' +
          '<td style="text-align:right">' + (it.latency_ms || '·') + '</td>' +
          '</tr>'
        ).join('') + '</tbody></table></div>';

    $('drawer-body').innerHTML = kpiHtml + donutHtml + topHtml + reqHtml;

    // Bind row clicks → request detail
    $('drawer-body').querySelectorAll('tr.row-clickable').forEach(tr => {
      tr.addEventListener('click', () => {
        const rid = tr.getAttribute('data-rid');
        if (rid) openRequestDetail(rid);
      });
    });
  }

  // ─── Latency-bucket drill drawer ──────────────────────────────────────
  async function openLatencyDrawer(minMs, maxMs, label) {
    openDrawer();
    $('drawer-title').textContent = 'Latency · ' + label + ' ms';
    $('drawer-sub').textContent =
      (maxMs == null ? '≥ ' + minMs + ' ms' : minMs + '–' + maxMs + ' ms') +
      ' · click a row to see the request detail';
    $('drawer-body').innerHTML = '<div class="placeholder">loading&hellip;</div>';
    const qs = new URLSearchParams({
      range: filterState.range, min_ms: String(minMs), limit: 100,
    });
    if (maxMs != null) qs.set('max_ms', String(maxMs));
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env) qs.set('env', filterState.env);
    const { body } = await api('/api/account/usage/latency-drill?' + qs);
    if (!body || !body.ok) {
      $('drawer-body').innerHTML = '<div class="placeholder">failed to load</div>';
      return;
    }
    const rows = body.data.requests || [];
    if (rows.length === 0) {
      $('drawer-body').innerHTML = '<div class="placeholder">no requests in this latency band</div>';
      return;
    }
    $('drawer-body').innerHTML =
      '<div class="reqd-section"><h4>' + rows.length + ' requests · sorted by latency (highest first)</h4>' +
      '<table class="tbl"><thead><tr><th>time</th><th>method</th><th>path</th><th>status</th><th style="text-align:right">latency</th></tr></thead>' +
      '<tbody>' + rows.map(it =>
        '<tr class="row-clickable" data-rid="' + escHtml(it.id) + '" style="cursor:pointer">' +
        '<td>' + escHtml((it.timestamp || '').slice(11, 19)) + '</td>' +
        '<td><span class="meth ' + methClass(it.method) + '">' + escHtml(it.method || '') + '</span></td>' +
        '<td>' + escHtml(it.path || '') + '</td>' +
        '<td><span class="stat ' + statusBucket(it.status_code) + '">' + (it.status_code || '') + '</span></td>' +
        '<td style="text-align:right"><b>' + (it.latency_ms || '·') + ' ms</b></td>' +
        '</tr>'
      ).join('') + '</tbody></table></div>';
    $('drawer-body').querySelectorAll('tr.row-clickable').forEach(tr => {
      tr.addEventListener('click', () => {
        const rid = tr.getAttribute('data-rid');
        if (rid) openRequestDetail(rid);
      });
    });
  }

  // ─── IP drill drawer (used when top-IPs row clicked) ──────────────────
  async function openIpDrawer(ip) {
    openDrawer();
    $('drawer-title').textContent = 'IP · ' + ip;
    $('drawer-sub').textContent = 'recent requests from this IP';
    // Use the standard /requests endpoint without an IP filter (we don't
    // have one server-side yet); we filter client-side after fetching.
    $('drawer-body').innerHTML = '<div class="placeholder">loading&hellip;</div>';
    const qs = new URLSearchParams({ range: filterState.range, limit: 200 });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    const { body } = await api('/api/account/usage/requests?' + qs);
    if (!body || !body.ok) return;
    const rows = (body.data.items || []).filter(r => (r.ip || '') === ip);
    if (rows.length === 0) {
      $('drawer-body').innerHTML = '<div class="placeholder">no requests from this IP in current window</div>';
      return;
    }
    $('drawer-body').innerHTML =
      '<div class="reqd-section"><h4>' + rows.length + ' requests from ' + escHtml(ip) + '</h4>' +
      '<table class="tbl"><thead><tr><th>time</th><th>method</th><th>path</th><th>status</th><th>ms</th></tr></thead>' +
      '<tbody>' + rows.map(it =>
        '<tr class="row-clickable" data-rid="' + escHtml(it.id) + '" style="cursor:pointer">' +
        '<td>' + escHtml(fmtAgo(it.timestamp)) + '</td>' +
        '<td><span class="meth ' + methClass(it.method) + '">' + escHtml(it.method || '') + '</span></td>' +
        '<td>' + escHtml(it.path || '') + '</td>' +
        '<td><span class="stat ' + statusBucket(it.status_code) + '">' + (it.status_code || '') + '</span></td>' +
        '<td style="text-align:right">' + (it.latency_ms || '·') + '</td>' +
        '</tr>'
      ).join('') + '</tbody></table></div>';
    $('drawer-body').querySelectorAll('tr.row-clickable').forEach(tr => {
      tr.addEventListener('click', () => {
        const rid = tr.getAttribute('data-rid');
        if (rid) openRequestDetail(rid);
      });
    });
  }

  // ─── 9.I · Request detail drawer ──────────────────────────────────────
  async function openRequestDetail(rid) {
    openReqd();
    $('reqd-title').textContent = 'Request #' + rid;
    $('reqd-sub').textContent = 'loading…';
    $('reqd-body').innerHTML = '<div class="placeholder">loading&hellip;</div>';

    // 9.N.8g · Synthetic live-IDs ("live-<ts>-<rand>") aren't in the DB
    // yet — the row is still in the in-memory buffer awaiting flush.
    // For these, render directly from the cached live event so the user
    // sees the basic envelope instead of "failed to load request detail".
    const ridStr = String(rid);
    const isSyntheticLive = ridStr.indexOf('live-') === 0;
    if (isSyntheticLive) {
      const cached = window.APIN && APIN._liveEventCache && APIN._liveEventCache.get(rid);
      if (cached && cached.event) {
        const e = cached.event;
        $('reqd-title').innerHTML = '<span class="reqd-tp">' + escHtml((e.method || '') + ' ' + (e.path || '')) + '</span>'
          + '<span class="reqd-rno">Request · live</span>';
        $('reqd-sub').textContent = ((window.APIN && APIN.time && APIN.time.localFull) ? APIN.time.localFull(e.timestamp) : (e.timestamp || ''))
          + ' · key: ' + (e.key_name || e.key_public_id || '?') + ' · (still buffering — full detail in ~5s)';
        $('reqd-body').innerHTML =
          '<div class="reqd-status-bar">' +
            '<span class="meth ' + methClass(e.method) + '">' + escHtml(e.method || '') + '</span>' +
            '<span class="stat ' + statusBucket(e.status_code) + '">' + (e.status_code || '') + '</span>' +
            '<span class="path">' + escHtml(e.path || '') + '</span>' +
          '</div>' +
          '<div class="reqd-card" style="margin-top:14px"><h3>Live event (server flush pending)</h3>' +
            '<div style="font-family:JetBrains Mono,monospace;font-size:12.5px;line-height:1.8;color:var(--ink)">' +
              '<div>latency: <b>' + (e.latency_ms != null ? e.latency_ms + ' ms' : '·') + '</b></div>' +
              '<div>bytes out: <b>' + (e.bytes_out != null ? fmtBytes(e.bytes_out) : '·') + '</b></div>' +
              '<div>error: <b>' + (e.error_code || 'none') + '</b></div>' +
              '<div>client ip: <b>' + escHtml(e.ip || '·') + '</b></div>' +
            '</div>' +
            '<p style="font-family:Fraunces,serif;font-style:italic;font-size:12.5px;color:var(--ink-soft);margin-top:14px;line-height:1.6">' +
              'This row was just received via the live stream and hasn\'t been written to the database yet. ' +
              'Headers, payload, stage timings, burst context, and the export snippets all appear once the buffer flushes ' +
              '(every few seconds). Refresh the page or click again shortly to see the full detail.' +
            '</p>' +
          '</div>';
        return;
      }
      // No cache hit — fall through to server lookup (may still 404)
    }

    const { body } = await api('/api/account/usage/request/' + encodeURIComponent(rid));
    if (!body || !body.ok) {
      $('reqd-body').innerHTML = '<div class="placeholder">failed to load request detail</div>';
      $('reqd-sub').textContent = '';
      return;
    }
    const r = body.data.row || {};
    const inf = body.data.inferred || {};
    const baselines = body.data.baselines || {};
    const burst = body.data.burst || {};
    const health = body.data.health || {};
    const payload = body.data.payload || {};
    const stageTimings = body.data.stage_timings || null;
    const curlSnip = body.data.curl || '';
    const pySnip = body.data.as_python || '';
    const nodeSnip = body.data.as_node || '';
    const jsSnip   = body.data.as_js || '';

    $('reqd-title').innerHTML = '<span class="reqd-tp">' + escHtml((r.method || '') + ' ' + (r.path || '')) + '</span>'
      + '<span class="reqd-rno">Request #' + escHtml(String(r.id != null ? r.id : rid)) + '</span>';
    $('reqd-sub').textContent = ((window.APIN && APIN.time && APIN.time.localFull) ? APIN.time.localFull(r.timestamp) : (r.timestamp || ''))
      + ' · key: ' + (r.key_name || r.key_public_id || '?');

    // Build status-bar header
    const sb = r.status_code || 0;
    const sbCls = statusBucket(sb);
    const statusBar =
      '<div class="reqd-status-bar">' +
        '<span class="meth ' + methClass(r.method) + '">' + escHtml(r.method || '') + '</span>' +
        '<span class="stat ' + sbCls + '">' + sb + '</span>' +
        '<span class="path">' + escHtml(r.path || '') + '</span>' +
      '</div>';

    // ── Latency bar with REAL endpoint baselines ──────────────────────
    // Was previously hardcoded "p50 baseline: ~200 ms / p99 baseline: ~2400 ms"
    // for EVERY request regardless of endpoint. Now computed server-side
    // over the last 200 requests to this exact path, so the comparison is
    // meaningful: a 4ms request on /api/scans showing "p50 of this endpoint
    // is 380ms" tells the user this request was unusually fast for what it is.
    const latMs = r.latency_ms || 0;
    const latPct = Math.min(100, Math.round((latMs / 3000) * 100));
    // Comparative tag (faster than typical / slower / typical)
    let comparisonTag = '';
    if (baselines.p50_ms != null && baselines.sample_size >= 5) {
      const ratio = latMs / Math.max(1, baselines.p50_ms);
      if (ratio < 0.5)      comparisonTag = '<span class="reqd-lat-tag good">faster than typical</span>';
      else if (ratio > 2.0) comparisonTag = '<span class="reqd-lat-tag warn">slower than typical</span>';
      else                  comparisonTag = '<span class="reqd-lat-tag mute">typical for this endpoint</span>';
    }
    const baselineRow = (baselines.p50_ms != null && baselines.sample_size >= 5)
      ? ('<div class="reqd-lat-baseline">' +
           '<span>p50 of this endpoint: <b>' + baselines.p50_ms + ' ms</b></span>' +
           '<span>p95 of this endpoint: <b>' + baselines.p95_ms + ' ms</b></span>' +
           '<span class="reqd-lat-sample">over ' + baselines.sample_size + ' recent requests</span>' +
         '</div>')
      : ('<div class="reqd-lat-baseline reqd-lat-mute">' +
           '<span>not enough history yet to compute endpoint baseline</span>' +
         '</div>');
    const latBar =
      '<div class="reqd-section">' +
        '<h4><svg class="icon"><use href="#i-gauge"/></svg> Latency</h4>' +
        '<div class="reqd-latency-bar">' +
          '<div class="fill" style="width:' + latPct + '%"></div>' +
          '<div class="lbl">' + latMs + ' ms</div>' +
        '</div>' +
        (comparisonTag ? '<div class="reqd-lat-tag-row">' + comparisonTag + '</div>' : '') +
        baselineRow +
      '</div>';

    // Inferred source card
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

    // Request / Response details
    //
    // Two bugs fixed here:
    //  (1) `apin_<REDACTED>` was being parsed by the browser as `<REDACTED>`
    //      being an unknown HTML tag, so the user only saw "apin_". Use
    //      typographic bullets (•••••) which render literally and convey
    //      "this is intentionally hidden".
    //  (2) The "Via" row was always showing literal "bearer" because the
    //      middleware tags every API-key auth as via="bearer". That's
    //      redundant with the "Auth: Bearer" row above. Only show "Via" if
    //      it's something interesting (sandbox, docs try-it-out, etc.).
    const viaRaw = String(r.via || '').toLowerCase();
    const showVia = viaRaw && viaRaw !== 'bearer' && viaRaw !== '·';
    const reqDetails =
      '<div class="reqd-section"><h4><svg class="icon"><use href="#i-list-ordered"/></svg> Request</h4>' +
      '<dl class="reqd-row"><dt>Method</dt><dd><code>' + escHtml(r.method || '') + '</code></dd></dl>' +
      '<dl class="reqd-row"><dt>Path</dt><dd><code>' + escHtml(r.path || '') + '</code></dd></dl>' +
      '<dl class="reqd-row"><dt>IP</dt><dd><code>' + escHtml(r.ip || '·') + '</code></dd></dl>' +
      '<dl class="reqd-row"><dt>Auth</dt><dd>Bearer · <code>apin_••••••••••</code> <span class="reqd-mute">(redacted for safety)</span></dd></dl>' +
      // 9.N.7.f · Bytes-in display. For body-less methods (GET/HEAD/DELETE/OPTIONS),
      // null means "no body expected" — show "0 B" with a quiet note rather
      // than a "·" that reads as missing data. For POST/PUT/PATCH with null,
      // the middleware didn't sniff the body — say so explicitly.
      '<dl class="reqd-row"><dt>Bytes in</dt><dd>' + (function(){
        const m = (r.method || '').toUpperCase();
        const bodyless = (m === 'GET' || m === 'HEAD' || m === 'DELETE' || m === 'OPTIONS');
        if (r.bytes_in != null && r.bytes_in > 0) return fmtBytes(r.bytes_in);
        if (r.bytes_in === 0) return '0 B <span class="reqd-mute">(empty body)</span>';
        if (bodyless) return '0 B <span class="reqd-mute">(none — ' + m + ' has no body)</span>';
        return '<span class="reqd-mute">not recorded</span>';
      })() + '</dd></dl>' +
      (showVia
        ? '<dl class="reqd-row"><dt>Via</dt><dd>' + escHtml(r.via) + '</dd></dl>'
        : '') +
      '</div>';

    const respDetails =
      '<div class="reqd-section"><h4><svg class="icon"><use href="#i-arrow-down-right"/></svg> Response</h4>' +
      '<dl class="reqd-row"><dt>Status</dt><dd><span class="stat ' + sbCls + '">' + sb + '</span></dd></dl>' +
      (r.error_code ?
        '<dl class="reqd-row"><dt>Error code</dt><dd><span class="err">' + escHtml(r.error_code) + '</span></dd></dl>' : '') +
      '<dl class="reqd-row"><dt>Latency</dt><dd>' + latMs + ' ms</dd></dl>' +
      '<dl class="reqd-row"><dt>Bytes out</dt><dd>' + (r.bytes_out != null ? fmtBytes(r.bytes_out) : '·') + '</dd></dl>' +
      '</div>';

    // Key card
    const keyCard =
      '<div class="reqd-section"><h4><svg class="icon"><use href="#i-key"/></svg> Authenticated key</h4>' +
      '<dl class="reqd-row"><dt>Name</dt><dd><b>' + escHtml(r.key_name || '(unnamed)') + '</b></dd></dl>' +
      '<dl class="reqd-row"><dt>Public ID</dt><dd><code>' + escHtml(r.key_public_id || '') + '</code></dd></dl>' +
      '<dl class="reqd-row"><dt>Environment</dt><dd><code>' + escHtml(r.env || 'live') + '</code></dd></dl>' +
      '<a href="/account/api/keys/' + encodeURIComponent(r.key_public_id || '') + '" target="_blank" rel="noopener" ' +
      'style="display:inline-block;margin-top:8px;font-size:12px;color:var(--green);font-family:\'JetBrains Mono\',monospace">' +
      '→ open this key in new tab</a>' +
      '</div>';

    // ── 9.N.8 · Timeline (server-side stage waterfall) ────────────────────
    const timelineCard = _renderTimelineCard(stageTimings, latMs);

    // ── 9.N.8 · Payload section (request/response headers + bodies) ───────
    const payloadCard = _renderPayloadCard(payload, rid);

    // ── 9.N.8 · Burst context ─────────────────────────────────────────────
    const burstCard = _renderBurstCard(burst, rid, r.timestamp);

    // ── 9.N.8 · Endpoint health mini-sparklines ───────────────────────────
    const healthCard = _renderHealthCard(health, r.path);

    // ── 9.N.8 · Export section (renamed from Reproduce) ──────────────────
    const exportCard = _renderExportCard(curlSnip, pySnip, jsSnip, nodeSnip, rid, r.path);

    // Assemble in the order locked in by request_detailed.md
    $('reqd-body').innerHTML =
      statusBar +
      latBar +
      sourceCard +
      timelineCard +
      payloadCard +
      reqDetails +
      respDetails +
      burstCard +
      healthCard +
      keyCard +
      exportCard;

    // ── Wire Payload section interactions ────────────────────────────────
    _wirePayloadSection($('reqd-body'), payload);

    // ── Wire Burst context — neighbour clicks open their own drawer ──────
    _wireBurstSection($('reqd-body'));

    // ── Wire Health sparklines — hover tooltips ──────────────────────────
    _wireHealthSection($('reqd-body'), health);

    // ── Wire Export section: tabs + copy + file exports + share ──────────
    _wireExportSection($('reqd-body'), { curl: curlSnip, python: pySnip,
                                            js: jsSnip, node: nodeSnip }, rid);

    // ── Section reveal animation (staggered fade-up) ─────────────────────
    if (window.APIN && APIN.fx) {
      const sections = $('reqd-body').querySelectorAll('.reqd-section');
      sections.forEach((sec, i) => {
        sec.style.opacity = '0';
        sec.style.transform = 'translateY(8px)';
        setTimeout(() => {
          sec.style.transition = 'opacity 180ms cubic-bezier(0.22,1,0.36,1), transform 180ms cubic-bezier(0.22,1,0.36,1)';
          sec.style.opacity = '1';
          sec.style.transform = 'translateY(0)';
        }, i * 30);
      });
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 9.N.8 · Renderers + wirers for the new drawer sections
  // ═══════════════════════════════════════════════════════════════════════

  function _renderTimelineCard(stages, totalLatMs) {
    // If stage timings aren't recorded (old rows before middleware update),
    // show a "not recorded" fallback so the section still appears in the
    // layout but explains its empty state.
    if (!stages || typeof stages !== 'object') {
      return '<div class="reqd-section"><h4><svg class="icon"><use href="#i-clock"/></svg> Timeline</h4>' +
        '<div class="reqd-mute">stage breakdown not recorded for this request</div>' +
      '</div>';
    }
    const order = ['auth', 'validate', 'handler', 'send'];
    const labels = { auth: 'auth', validate: 'validate', handler: 'handler', send: 'send' };
    let total = 0;
    for (const k of order) total += Number(stages[k] || 0);
    if (total <= 0) total = Math.max(1, Number(totalLatMs) || 1);
    // Identify the hot stage (largest)
    let hot = order[0]; let hotV = -1;
    for (const k of order) {
      const v = Number(stages[k] || 0);
      if (v > hotV) { hotV = v; hot = k; }
    }
    const rows = order.map(k => {
      const v = Number(stages[k] || 0);
      const pct = total > 0 ? (v / total) * 100 : 0;
      const isHot = (k === hot && v > 0);
      return '<div class="reqd-tl-row' + (isHot ? ' is-hot' : '') + '">' +
        '<div class="reqd-tl-label">' + labels[k] + '</div>' +
        '<div class="reqd-tl-bar"><div class="reqd-tl-fill" style="width:' + pct.toFixed(1) + '%"></div></div>' +
        '<div class="reqd-tl-val"><b>' + v + ' ms</b> · ' + pct.toFixed(0) + '%' +
          (isHot ? ' <span class="reqd-tl-hot">◀ hot</span>' : '') +
        '</div>' +
      '</div>';
    }).join('');
    return '<div class="reqd-section"><h4>' +
      '<svg class="icon"><use href="#i-clock"/></svg> Timeline' +
      '<span class="reqd-tl-total">total: ' + total + ' ms</span>' +
    '</h4>' + rows + '</div>';
  }

  function _renderPayloadCard(p, rid) {
    // p has headers_in, headers_out, body_in_preview, body_out_preview,
    // body_in_ctype, body_out_ctype, body_in_truncated, body_out_truncated
    const haveAny = p && (p.headers_in || p.headers_out || p.body_in_preview || p.body_out_preview);
    if (!haveAny) {
      return '<div class="reqd-section"><h4><svg class="icon"><use href="#i-package"/></svg> Payload</h4>' +
        '<div class="reqd-mute">payload not recorded for this request</div>' +
      '</div>';
    }
    // Mode toggle: full / body / headers / summary
    const modeBar = '<div class="reqd-payload-modes">' +
      '<button data-mode="full"    aria-pressed="true">full</button>' +
      '<button data-mode="body"    aria-pressed="false">body</button>' +
      '<button data-mode="headers" aria-pressed="false">headers</button>' +
      '<button data-mode="summary" aria-pressed="false">summary</button>' +
    '</div>';

    return '<div class="reqd-section reqd-payload-section">' +
      '<h4><svg class="icon"><use href="#i-package"/></svg> Payload' + modeBar + '</h4>' +
      _renderPayloadBlock('Request headers', 'headers-in', _renderHeadersList(p.headers_in)) +
      _renderPayloadBlock('Request body', 'body-in',
        _renderBodyBlock(p.body_in_preview, p.body_in_ctype, p.body_in_truncated, rid, 'in')) +
      _renderPayloadBlock('Response headers', 'headers-out', _renderHeadersList(p.headers_out)) +
      _renderPayloadBlock('Response body', 'body-out',
        _renderBodyBlock(p.body_out_preview, p.body_out_ctype, p.body_out_truncated, rid, 'out')) +
    '</div>';
  }

  function _renderPayloadBlock(label, dataKey, innerHtml) {
    return '<div class="reqd-pl-block" data-pl="' + dataKey + '">' +
      '<button class="reqd-pl-toggle" type="button" aria-expanded="true">' +
        '<span class="reqd-pl-chev">▾</span>' +
        '<span class="reqd-pl-label">' + label + '</span>' +
      '</button>' +
      '<div class="reqd-pl-content">' + innerHtml + '</div>' +
    '</div>';
  }

  /**
   * 9.N.8e · Build the summary-mode view: at-a-glance overview of the
   * payload — content types, header counts, multipart parts list, and
   * response JSON highlights (top-level keys with summarised values).
   * The summary is the answer to "what was this request about?" without
   * needing to open the full body.
   */
  function _buildPayloadSummary(p) {
    const hi = p.headers_in || {};
    const ho = p.headers_out || {};
    const ctIn = p.body_in_ctype || hi['content-type'] || '—';
    const ctOut = p.body_out_ctype || ho['content-type'] || '—';

    let html = '<div class="reqd-pl-summary-title">Summary</div>';

    // Top-level metadata grid
    html += '<dl class="reqd-headers-list reqd-pl-summary-grid">' +
      '<dt>request type</dt><dd>' + escHtml(ctIn) + '</dd>' +
      '<dt>response type</dt><dd>' + escHtml(ctOut) + '</dd>' +
      '<dt>request headers</dt><dd>' + Object.keys(hi).length + ' header' + (Object.keys(hi).length === 1 ? '' : 's') + '</dd>' +
      '<dt>response headers</dt><dd>' + Object.keys(ho).length + ' header' + (Object.keys(ho).length === 1 ? '' : 's') + '</dd>' +
    '</dl>';

    // Request multipart parts summary
    if (p.body_in_preview) {
      try {
        const obj = JSON.parse(p.body_in_preview);
        if (obj && obj.kind === 'multipart' && obj.parts && obj.parts.length > 0) {
          html += '<div class="reqd-pl-summary-subhead">Request parts (' + obj.parts.length + ')</div>';
          html += '<ul class="reqd-pl-summary-parts">';
          for (const part of obj.parts) {
            const ic = (part.ctype || '').indexOf('image/') === 0 ? 'i-image' : 'i-package';
            html += '<li>' +
              '<svg class="reqd-mp-icon"><use href="#' + ic + '"/></svg>' +
              '<span class="reqd-pl-summary-pname">' + escHtml(part.name || '(unnamed)') + '</span>' +
              (part.filename ? '<span class="reqd-pl-summary-pfile">' + escHtml(part.filename) + '</span>' : '') +
              '<span class="reqd-pl-summary-ptype">' + escHtml(part.ctype || '') + '</span>' +
              '<span class="reqd-pl-summary-psize">' + _fmtBytesShort(part.size_bytes) + '</span>' +
            '</li>';
          }
          html += '</ul>';
        }
      } catch (_) {}
    }

    // Response JSON highlights — top-level key/value extraction
    if (p.body_out_preview && (ctOut.indexOf('application/json') === 0 || ctOut.indexOf('+json') >= 0)) {
      try {
        const obj = JSON.parse(p.body_out_preview);
        if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
          html += '<div class="reqd-pl-summary-subhead">Response highlights</div>';
          html += '<dl class="reqd-pl-summary-resp">';
          const keys = Object.keys(obj).slice(0, 15);
          for (const k of keys) {
            const v = obj[k];
            html += '<dt>' + escHtml(k) + '</dt><dd>' + _fmtSummaryValue(v) + '</dd>';
          }
          if (Object.keys(obj).length > 15) {
            html += '<dt>…</dt><dd class="reqd-mute">' +
              (Object.keys(obj).length - 15) + ' more keys (open response body for full view)</dd>';
          }
          html += '</dl>';
        }
      } catch (_) {}
    }

    return html;
  }

  /** Format a JSON value into a compact, readable summary string. */
  function _fmtSummaryValue(v) {
    if (v === null) return '<span class="reqd-pl-summary-null">null</span>';
    if (v === undefined) return '<span class="reqd-pl-summary-null">undefined</span>';
    if (typeof v === 'boolean') return '<span class="reqd-pl-summary-bool">' + v + '</span>';
    if (typeof v === 'number') return '<span class="reqd-pl-summary-num">' + v + '</span>';
    if (typeof v === 'string') {
      if (v.length > 80) {
        return '<span class="reqd-pl-summary-str">"' + escHtml(v.slice(0, 64)) + '…"</span> ' +
          '<span class="reqd-mute">(' + v.length + ' chars)</span>';
      }
      return '<span class="reqd-pl-summary-str">"' + escHtml(v) + '"</span>';
    }
    if (Array.isArray(v)) {
      if (v.length === 0) return '<span class="reqd-pl-summary-arr">[]</span>';
      // Show first 3 if short, else just a summary
      const sample = v.slice(0, 3).map(x => {
        if (typeof x === 'string') return '"' + escHtml(x.slice(0, 24)) + (x.length > 24 ? '…' : '') + '"';
        if (typeof x === 'number') return String(x);
        if (x === null || x === undefined) return 'null';
        return typeof x;
      }).join(', ');
      return '<span class="reqd-pl-summary-arr">[' + sample + (v.length > 3 ? ', …' : '') + ']</span> ' +
        '<span class="reqd-mute">(' + v.length + ' items)</span>';
    }
    if (typeof v === 'object') {
      const keys = Object.keys(v);
      return '<span class="reqd-pl-summary-obj">{' + keys.slice(0, 3).map(escHtml).join(', ') +
        (keys.length > 3 ? ', …' : '') + '}</span> ' +
        '<span class="reqd-mute">(' + keys.length + ' keys)</span>';
    }
    return escHtml(String(v));
  }

  function _renderHeadersList(headers) {
    if (!headers || typeof headers !== 'object') {
      return '<div class="reqd-mute">no headers recorded</div>';
    }
    const keys = Object.keys(headers).sort();
    if (keys.length === 0) {
      return '<div class="reqd-mute">no headers recorded</div>';
    }
    return '<dl class="reqd-headers-list">' + keys.map(k =>
      '<dt>' + escHtml(k) + '</dt>' +
      '<dd>' + escHtml(String(headers[k])) + '</dd>'
    ).join('') + '</dl>';
  }

  function _renderBodyBlock(preview, ctype, truncated, rid, direction) {
    const ctypeLower = String(ctype || '').toLowerCase();
    if (!preview) {
      return '<div class="reqd-mute">no body</div>';
    }
    const isJson = ctypeLower.indexOf('application/json') === 0 ||
                   ctypeLower.indexOf('text/json') === 0 ||
                   ctypeLower.indexOf('+json') >= 0;
    const isHtml = ctypeLower.indexOf('text/html') === 0;
    const isMultipart = ctypeLower.indexOf('multipart/') === 0;

    // 9.N.8e · Detect structured payload JSON (multipart parts or
    // single-image binary). Both come from the middleware as a JSON
    // string with a `kind` field. Render with thumbnails + hover preview.
    let structured = null;
    if (isMultipart || /^\[binary/.test(preview)) {
      try {
        const obj = JSON.parse(preview);
        if (obj && (obj.kind === 'multipart' || obj.kind === 'image')) {
          structured = obj;
        }
      } catch (_) {}
    }
    // Also try when ctype is image/* directly
    if (!structured && ctypeLower.indexOf('image/') === 0) {
      try {
        const obj = JSON.parse(preview);
        if (obj && obj.kind === 'image') structured = obj;
      } catch (_) {}
    }

    // Header strip: content type + size + truncation badge
    const sizeStr = (structured && structured.size_bytes)
      ? ' · ' + _fmtBytesShort(structured.size_bytes)
      : '';
    let sizeNote = '';
    if (truncated || (structured && structured.preview_truncated)) {
      sizeNote = '<span class="reqd-pl-trunc">truncated for preview</span>';
    }
    const stripHtml = '<div class="reqd-pl-strip">' +
      '<span class="reqd-pl-ctype">' + escHtml(ctype || 'unknown') + sizeStr + '</span>' +
      sizeNote +
    '</div>';

    if (structured && structured.kind === 'multipart') {
      return stripHtml + _renderMultipartParts(structured);
    }
    if (structured && structured.kind === 'image') {
      return stripHtml + _renderImageBody(structured);
    }
    if (isJson) {
      return stripHtml +
        '<div class="reqd-pl-body">' +
          _renderLineNumberedCode(_jsonPrettyOrRaw(preview), 'json') +
        '</div>';
    }
    if (isHtml) {
      return stripHtml +
        '<div class="reqd-pl-body">' +
          _renderLineNumberedCode(preview, 'html') +
        '</div>';
    }
    // Default: plain text — also line-numbered so the layout is consistent
    return stripHtml +
      '<div class="reqd-pl-body">' +
        _renderLineNumberedCode(preview, 'text') +
      '</div>';
  }

  function _jsonPrettyOrRaw(s) {
    try {
      const o = JSON.parse(s);
      return JSON.stringify(o, null, 2);
    } catch (e) {
      return s;
    }
  }

  function _fmtBytesShort(n) {
    if (n == null) return '';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / (1024 * 1024)).toFixed(2) + ' MB';
  }

  /**
   * Render multipart parts as a structured list — each part shows
   * name, filename, content-type, size. Image parts get a hoverable
   * filename that pops a thumbnail anchored near the cursor, and a
   * click that opens the full-resolution image in the existing
   * APIN.lightbox if available.
   */
  function _renderMultipartParts(obj) {
    const parts = obj.parts || [];
    if (parts.length === 0) {
      return '<div class="reqd-pl-body"><div class="reqd-mute">no parts found in multipart body</div></div>';
    }
    const rows = parts.map((p, i) => {
      const isImage = (p.ctype || '').indexOf('image/') === 0;
      const hasPreview = !!p.preview_b64;
      const fnameAttrs = (isImage && hasPreview)
        ? ' data-img-preview="' + escHtml(p.ctype + ',' + p.preview_b64).replace(/"/g, '&quot;') + '"' +
          ' data-img-filename="' + escHtml(p.filename || '') + '"' +
          ' data-img-size="' + escHtml(_fmtBytesShort(p.size_bytes)) + '"' +
          ' class="reqd-mp-fname reqd-mp-fname-img"'
        : ' class="reqd-mp-fname"';
      const icon = isImage ? '#i-image' : '#i-package';
      return '<div class="reqd-mp-row">' +
        '<svg class="reqd-mp-icon"><use href="' + icon + '"/></svg>' +
        '<div class="reqd-mp-info">' +
          '<div class="reqd-mp-line1">' +
            '<span class="reqd-mp-name">' + escHtml(p.name || '(unnamed)') + '</span>' +
            (p.filename
              ? '<span' + fnameAttrs + '>' + escHtml(p.filename) + '</span>'
              : '') +
          '</div>' +
          '<div class="reqd-mp-line2">' +
            '<span class="reqd-mp-ctype">' + escHtml(p.ctype || 'unknown') + '</span>' +
            '<span class="reqd-mp-size">' + escHtml(_fmtBytesShort(p.size_bytes)) + '</span>' +
            (isImage && hasPreview
              ? '<span class="reqd-mp-hint">hover filename for preview</span>'
              : '') +
          '</div>' +
        '</div>' +
      '</div>';
    }).join('');
    return '<div class="reqd-pl-body reqd-pl-mp"><div class="reqd-mp-list">' +
      rows + '</div></div>';
  }

  /** Render a single-image body (raw image bytes, not multipart). */
  function _renderImageBody(obj) {
    if (!obj.preview_b64) {
      return '<div class="reqd-pl-body reqd-pl-binary">' +
        '<svg class="reqd-pl-bin-icon"><use href="#i-image"/></svg>' +
        '<span>image · ' + escHtml(obj.ctype || '') + ' · ' +
        _fmtBytesShort(obj.size_bytes) + '</span></div>';
    }
    const dataUri = 'data:' + obj.ctype + ';base64,' + obj.preview_b64;
    return '<div class="reqd-pl-body reqd-pl-image">' +
      '<img src="' + dataUri + '" alt="response image preview" class="reqd-img-inline"/>' +
      '<div class="reqd-mp-line2">' +
        '<span class="reqd-mp-ctype">' + escHtml(obj.ctype || '') + '</span>' +
        '<span class="reqd-mp-size">' + _fmtBytesShort(obj.size_bytes) + '</span>' +
        (obj.preview_truncated ? '<span class="reqd-mp-hint">preview truncated</span>' : '') +
      '</div>' +
    '</div>';
  }

  /**
   * Render a multi-line string as a line-numbered code block. Each line
   * is wrapped in <span class="reqd-code-line"> so the CSS counter can
   * emit gutter numbers. The whole thing is wrapped in a syntax-highlight
   * envelope; the highlighter wraps tokens with <span class="syn-*">
   * inside the per-line containers. We apply the highlighter to the
   * concatenated text once, then split by line.
   */
  function _renderLineNumberedCode(rawText, lang) {
    if (!rawText) {
      return '<div class="reqd-code"><div class="reqd-code-scroll"><pre><span class="reqd-code-line"></span></pre></div></div>';
    }
    // Highlight the entire text first (token spans don't cross lines in
    // our highlighter; we can safely split on \n afterwards).
    let highlighted = '';
    if (window.APIN && APIN.syntax && APIN.syntax.highlight) {
      try {
        highlighted = APIN.syntax.highlight(rawText, lang);
      } catch (_) {
        highlighted = escHtml(rawText);
      }
    } else {
      highlighted = escHtml(rawText);
    }
    // Split by line — but the highlighter may have wrapped tokens in
    // <span>...</span>. We need to split on the raw line breaks but keep
    // any open tag continued. Easiest: split on actual "\n" character;
    // each line goes into a .reqd-code-line.
    const lines = highlighted.split('\n');
    const out = lines.map(l => '<span class="reqd-code-line">' + (l || ' ') + '</span>').join('');
    return '<div class="reqd-code"><div class="reqd-code-scroll"><pre>' + out + '</pre></div></div>';
  }

  function _wirePayloadSection(root, payload) {
    // Mode toggle: filter visible blocks
    const sec = root.querySelector('.reqd-payload-section');
    if (!sec) return;
    const setMode = (mode) => {
      sec.querySelectorAll('.reqd-payload-modes button').forEach(b =>
        b.setAttribute('aria-pressed', b.dataset.mode === mode ? 'true' : 'false'));
      sec.querySelectorAll('.reqd-pl-block').forEach(blk => {
        const key = blk.getAttribute('data-pl') || '';
        const isHeaders = key.indexOf('headers-') === 0;
        const isBody    = key.indexOf('body-')    === 0;
        let visible = true;
        if (mode === 'body')    visible = isBody;
        else if (mode === 'headers') visible = isHeaders;
        else if (mode === 'summary') visible = false;   // summary = strips only
        blk.style.display = visible ? '' : 'none';
      });
      // 9.N.8e · Summary mode — enriched. Beyond just request/response
      // types + header counts, also include:
      //  · multipart parts summary (filenames, types, sizes)
      //  · response JSON highlights (top-level key→value pairs, with
      //    huge values like base64 collapsed to "[N chars]")
      // The goal: the user can see "what is this request actually about"
      // at a glance, without expanding into headers/body blocks.
      if (mode === 'summary') {
        let sumBox = sec.querySelector('.reqd-pl-summary');
        if (sumBox) sumBox.remove();   // rebuild each time so it's current
        sumBox = document.createElement('div');
        sumBox.className = 'reqd-pl-summary';
        sumBox.innerHTML = _buildPayloadSummary(payload);
        sec.appendChild(sumBox);
      } else {
        const sumBox = sec.querySelector('.reqd-pl-summary');
        if (sumBox) sumBox.style.display = 'none';
      }
    };
    sec.querySelectorAll('.reqd-payload-modes button').forEach(b => {
      b.addEventListener('click', () => setMode(b.dataset.mode));
    });
    // Wire collapsible block headers
    sec.querySelectorAll('.reqd-pl-toggle').forEach(t => {
      t.addEventListener('click', () => {
        const blk = t.closest('.reqd-pl-block');
        if (!blk) return;
        const content = blk.querySelector('.reqd-pl-content');
        const chev = blk.querySelector('.reqd-pl-chev');
        const expanded = t.getAttribute('aria-expanded') === 'true';
        if (expanded) {
          content.style.display = 'none';
          chev.textContent = '▸';
          t.setAttribute('aria-expanded', 'false');
        } else {
          content.style.display = '';
          chev.textContent = '▾';
          t.setAttribute('aria-expanded', 'true');
        }
      });
    });
    // 9.N.8e · Wire image-preview hover popover for multipart filenames.
    // The filename span carries data-img-preview="ctype,b64" — on hover
    // we pop a thumbnail anchored near the cursor.
    let _hoverImg = null;
    function _showImgPreview(el, e) {
      if (_hoverImg) { _hoverImg.remove(); _hoverImg = null; }
      const data = el.getAttribute('data-img-preview') || '';
      const i = data.indexOf(',');
      if (i < 0) return;
      const ctype = data.slice(0, i);
      const b64 = data.slice(i + 1);
      const fname = el.getAttribute('data-img-filename') || '';
      const size = el.getAttribute('data-img-size') || '';
      _hoverImg = document.createElement('div');
      _hoverImg.className = 'reqd-img-popover';
      _hoverImg.innerHTML =
        '<img src="data:' + ctype + ';base64,' + b64 + '" alt="' + escHtml(fname) + '"/>' +
        '<div class="reqd-img-popover-meta">' +
          '<div class="reqd-img-popover-fname">' + escHtml(fname) + '</div>' +
          '<div class="reqd-img-popover-sub">' + escHtml(ctype) + ' · ' + escHtml(size) + '</div>' +
        '</div>';
      document.body.appendChild(_hoverImg);
      // Position: anchored to cursor, prefer right side, flip if overflow
      const x = e.clientX + 14;
      const y = e.clientY - 8;
      _hoverImg.style.left = x + 'px';
      _hoverImg.style.top  = y + 'px';
      // 9.N.8e · Use setTimeout(0) instead of requestAnimationFrame so
      // positioning still adjusts when the tab is hidden (rAF callbacks
      // are paused in background tabs — that left the popover at its
      // off-screen initial position). 0ms is enough to let the browser
      // compute the inserted element's bounding rect.
      setTimeout(() => {
        if (!_hoverImg) return;
        const r = _hoverImg.getBoundingClientRect();
        // Clamp horizontally — flip to cursor-left if overflowing right
        let newLeft = parseFloat(_hoverImg.style.left);
        if (r.right > window.innerWidth - 8) {
          newLeft = Math.max(8, e.clientX - r.width - 14);
        }
        if (newLeft < 8) newLeft = 8;
        // Final right-edge clamp
        if (newLeft + r.width > window.innerWidth - 8) {
          newLeft = Math.max(8, window.innerWidth - r.width - 8);
        }
        _hoverImg.style.left = newLeft + 'px';
        // Clamp vertically — same logic, with hard top + bottom clamps so
        // the popover is always fully visible regardless of cursor pos.
        let newTop = parseFloat(_hoverImg.style.top);
        if (r.bottom > window.innerHeight - 8) {
          newTop = e.clientY - r.height - 8;
        }
        if (newTop < 8) newTop = 8;
        if (newTop + r.height > window.innerHeight - 8) {
          newTop = Math.max(8, window.innerHeight - r.height - 8);
        }
        _hoverImg.style.top = newTop + 'px';
        _hoverImg.style.opacity = '1';
      }, 0);
    }
    function _hideImgPreview() {
      if (_hoverImg) {
        _hoverImg.style.opacity = '0';
        const toRemove = _hoverImg;
        _hoverImg = null;
        setTimeout(() => { try { toRemove.remove(); } catch (_) {} }, 140);
      }
    }
    sec.querySelectorAll('[data-img-preview]').forEach(el => {
      el.addEventListener('mouseenter', e => _showImgPreview(el, e));
      el.addEventListener('mousemove',  e => {
        if (_hoverImg) {
          _hoverImg.style.left = (e.clientX + 14) + 'px';
          _hoverImg.style.top  = (e.clientY - 8) + 'px';
        }
      });
      el.addEventListener('mouseleave', _hideImgPreview);
      // Click → open full preview in lightbox.
      // Note: APIN.lightbox.open uses a `build:(panel)=>...` callback to
      // populate body, NOT an `html` string. Previous code passed `html`
      // which the lightbox silently ignored — modal opened with empty body.
      el.addEventListener('click', () => {
        const data = el.getAttribute('data-img-preview') || '';
        const i = data.indexOf(',');
        if (i < 0) return;
        const ctype = data.slice(0, i);
        const b64 = data.slice(i + 1);
        const dataUri = 'data:' + ctype + ';base64,' + b64;
        const fname = el.getAttribute('data-img-filename') || 'image preview';
        const sizeStr = el.getAttribute('data-img-size') || '';
        if (window.APIN && APIN.lightbox && APIN.lightbox.open) {
          APIN.lightbox.open({
            title: fname,
            subtitle: ctype + ' · ' + sizeStr,
            build: (panel) => {
              panel.innerHTML =
                '<div style="text-align:center;padding:24px;background:var(--paper)">' +
                  '<img src="' + dataUri + '" alt="' + escHtml(fname) + '" ' +
                       'style="max-width:100%;max-height:78vh;display:inline-block;' +
                       'border:1px solid var(--paper-edge);border-radius:4px"/>' +
                '</div>';
            },
          });
        } else {
          const w = window.open();
          if (w) w.document.write('<img src="' + dataUri + '" alt="' + escHtml(fname) + '"/>');
        }
      });
    });
    // Default mode
    setMode('full');
  }

  function _renderBurstCard(burst, currentRid, currentTs) {
    if (!burst || !burst.neighbours || burst.neighbours.length === 0) {
      return '<div class="reqd-section"><h4><svg class="icon"><use href="#i-cluster"/></svg> Burst context</h4>' +
        '<div class="reqd-mute">no neighbouring requests found in the ±1s window</div>' +
      '</div>';
    }
    const n = burst.cluster_size || burst.neighbours.length;
    const isBurst = !!burst.is_burst;
    // Cluster marker — N small bars; "this" one highlighted
    const totalForMarker = Math.min(burst.neighbours.length, 20);
    const markerBars = burst.neighbours.slice(0, totalForMarker).map(nb =>
      '<span class="reqd-bc-bar' + (nb.is_current ? ' is-current' : '') + '"' +
      ' title="' + escHtml((nb.timestamp || '').slice(11, 19) + ' · ' + (nb.method || '') + ' ' + (nb.path || '')) + '"></span>'
    ).join('');
    const summary = isBurst
      ? ('1 of ' + n + ' requests in a burst around ' + (currentTs || '').slice(11, 19))
      : ('this request stands alone in its ±1s window');
    const rows = burst.neighbours.map(nb => {
      const ts = (nb.timestamp || '').slice(11, 23);
      const isCur = nb.is_current ? 'is-current' : '';
      const arrow = nb.is_current ? '▸' : ' ';
      const methCls = methClass(nb.method);
      const statusCls = statusBucket(nb.status_code || 0);
      return '<div class="reqd-bc-row ' + isCur + '" data-rid="' + (nb.id || '') + '">' +
        '<span class="reqd-bc-arrow">' + arrow + '</span>' +
        '<span class="reqd-bc-ts">' + escHtml(ts) + '</span>' +
        '<span class="meth ' + methCls + '">' + escHtml(nb.method || '') + '</span>' +
        '<span class="reqd-bc-path">' + escHtml(nb.path || '') + '</span>' +
        '<span class="stat ' + statusCls + '">' + (nb.status_code || 0) + '</span>' +
        '<span class="reqd-bc-lat">' + (nb.latency_ms != null ? nb.latency_ms + 'ms' : '·') + '</span>' +
      '</div>';
    }).join('');
    return '<div class="reqd-section"><h4>' +
      '<svg class="icon"><use href="#i-cluster"/></svg> Burst context' +
    '</h4>' +
      '<div class="reqd-bc-summary">' + escHtml(summary) + '</div>' +
      (markerBars ? '<div class="reqd-bc-marker">' + markerBars + '</div>' : '') +
      '<div class="reqd-bc-neighbours-label">neighbours (±1 s):</div>' +
      '<div class="reqd-bc-list">' + rows + '</div>' +
    '</div>';
  }

  function _wireBurstSection(root) {
    root.querySelectorAll('.reqd-bc-row[data-rid]').forEach(row => {
      const rid = row.getAttribute('data-rid');
      if (!rid) return;
      row.style.cursor = 'pointer';
      row.addEventListener('click', () => {
        // Re-open the drawer with the clicked neighbour's id
        openRequestDetail(rid);
      });
    });
  }

  function _renderHealthCard(health, path) {
    const buckets = (health && health.buckets) || [];
    if (buckets.length === 0) {
      return '<div class="reqd-section"><h4><svg class="icon"><use href="#i-activity"/></svg> Endpoint health' +
        '<span class="reqd-mute"> · ' + escHtml(path || '') + ' · last 1h</span>' +
      '</h4>' +
        '<div class="reqd-mute">no recent activity on this endpoint</div>' +
      '</div>';
    }
    const totals = health.totals || {};
    // 9.N.8e · SVG sparkline — pixel-accurate column-fit. Unicode bars
    // overflowed because each char took a fixed width regardless of
    // available space. With SVG width=100% the bars scale to whatever
    // the host column gives us.
    const _spark = (vals, fillColor) => {
      const N = vals.length;
      const W = 100, H = 22;   // viewBox units; CSS scales to host width
      const maxV = Math.max(1, ...vals);
      const gap = 0.5;
      const barW = (W - gap * (N - 1)) / N;
      // Lay out bars left→right; oldest at left, newest at right.
      const bars = vals.map((v, i) => {
        const h = Math.max(1.2, (v / maxV) * (H - 2));
        const x = (barW + gap) * i;
        const y = H - h;
        return '<g class="reqd-hc-cell" data-i="' + i + '" data-v="' + v + '">' +
          '<rect x="' + x.toFixed(2) + '" y="' + y.toFixed(2) +
          '" width="' + barW.toFixed(2) + '" height="' + h.toFixed(2) +
          '" fill="' + (fillColor || 'currentColor') + '" rx="0.6"/>' +
          // invisible larger hit-target for hover (full column height)
          '<rect x="' + x.toFixed(2) + '" y="0" width="' + barW.toFixed(2) +
          '" height="' + H + '" fill="transparent"/>' +
        '</g>';
      }).join('');
      return '<span class="reqd-hc-spark"><svg viewBox="0 0 ' + W + ' ' + H +
        '" preserveAspectRatio="none">' + bars + '</svg></span>';
    };
    const counts = buckets.map(b => Number(b.count || 0));
    const errors = buckets.map(b => Number(b.errors || 0));
    const p50s   = buckets.map(b => Number(b.p50 || 0));
    const p95s   = buckets.map(b => Number(b.p95 || 0));
    const reqTone = errors.reduce((a,b)=>a+b,0) === 0 ? 'clean' : 'mixed';
    const summary = totals.requests + ' req · ' + totals.errors + ' err · p50 ' +
                    totals.p50 + 'ms · p95 ' + totals.p95 + 'ms';
    // Color per metric (matches the design system tokens)
    const INK_SOFT = 'var(--ink-soft, #6b6453)';
    const C_AMBER  = 'var(--c-amber, #d49620)';
    const C_DANGER = 'var(--c-danger, #b13d2e)';
    const C_INFO   = 'var(--c-info, #2d6a96)';
    return '<div class="reqd-section"><h4>' +
      '<svg class="icon"><use href="#i-activity"/></svg> Endpoint health' +
      '<span class="reqd-mute"> · ' + escHtml(path || '') + ' · last 1h</span>' +
    '</h4>' +
      '<div class="reqd-hc-row" data-metric="requests">' +
        '<span class="reqd-hc-label">requests</span>' +
        _spark(counts, INK_SOFT) +
        '<span class="reqd-hc-totals">' + totals.requests + '</span>' +
      '</div>' +
      '<div class="reqd-hc-row" data-metric="errors">' +
        '<span class="reqd-hc-label">errors</span>' +
        _spark(errors, C_DANGER) +
        '<span class="reqd-hc-totals">' + totals.errors + (reqTone==='clean'?' · clean':'') + '</span>' +
      '</div>' +
      '<div class="reqd-hc-row" data-metric="p50">' +
        '<span class="reqd-hc-label">p50</span>' +
        _spark(p50s, INK_SOFT) +
        '<span class="reqd-hc-totals">' + totals.p50 + ' ms</span>' +
      '</div>' +
      '<div class="reqd-hc-row" data-metric="p95">' +
        '<span class="reqd-hc-label">p95</span>' +
        _spark(p95s, C_AMBER) +
        '<span class="reqd-hc-totals">' + totals.p95 + ' ms</span>' +
      '</div>' +
      '<div class="reqd-hc-summary reqd-mute">' + escHtml(summary) + '</div>' +
      '<div class="reqd-hc-tip" hidden></div>' +
    '</div>';
  }

  function _wireHealthSection(root, health) {
    const sec = root.querySelector('.reqd-hc-tip')?.closest('.reqd-section');
    if (!sec) return;
    const buckets = (health && health.buckets) || [];
    const total = (health && health.total_seconds) || 3600;
    const bucketSec = (health && health.bucket_seconds) || (total / Math.max(1, buckets.length));
    const tip = sec.querySelector('.reqd-hc-tip');
    sec.querySelectorAll('.reqd-hc-cell').forEach(cell => {
      cell.addEventListener('mouseenter', e => {
        const i = parseInt(cell.getAttribute('data-i'), 10);
        const b = buckets[i] || {};
        // Time bucket: i*bucketSec ago to (i+1)*bucketSec ago, relative to now
        const endAgoSec = (buckets.length - i - 1) * bucketSec;
        const dt = new Date(Date.now() - endAgoSec * 1000);
        const hhmm = String(dt.getHours()).padStart(2,'0') + ':' + String(dt.getMinutes()).padStart(2,'0');
        tip.innerHTML =
          '<div class="reqd-hc-tip-time">' + hhmm + '</div>' +
          '<div>' + (b.count || 0) + ' req · ' + (b.errors || 0) + ' err</div>' +
          '<div class="reqd-hc-tip-lat">p50 ' + (b.p50 || 0) + ' ms · p95 ' + (b.p95 || 0) + ' ms</div>';
        const rect = cell.getBoundingClientRect();
        const sRect = sec.getBoundingClientRect();
        tip.style.left = (rect.left - sRect.left + rect.width / 2) + 'px';
        tip.style.top  = (rect.top  - sRect.top  - 8) + 'px';
        tip.hidden = false;
        tip.style.opacity = '0';
        requestAnimationFrame(() => { tip.style.opacity = '1'; });
      });
      cell.addEventListener('mouseleave', () => { tip.hidden = true; });
    });
  }

  function _renderExportCard(curlSnip, pySnip, jsSnip, nodeSnip, rid, path) {
    // 9.N.8e · Code block uses the same line-numbered component as the
    // payload bodies, for visual consistency + readability.
    return '<div class="reqd-section"><h4><svg class="icon"><use href="#i-bolt"/></svg> Export</h4>' +
      '<div class="reqd-export-row-label">as code:</div>' +
      '<div class="reqd-code-tabs" role="tablist">' +
        '<button data-snip="curl"   aria-pressed="true">curl</button>' +
        '<button data-snip="python" aria-pressed="false">python</button>' +
        '<button data-snip="js"     aria-pressed="false">js</button>' +
        '<button data-snip="node"   aria-pressed="false">node</button>' +
      '</div>' +
      '<div class="reqd-export-codewrap" id="reqd-code-body">' +
        '<button class="copy-btn" id="reqd-copy">copy</button>' +
        _renderLineNumberedCode(curlSnip, 'bash') +
      '</div>' +
      '<div class="reqd-export-row-label">as file:</div>' +
      '<div class="reqd-export-chips">' +
        '<button class="reqd-chip" data-export="postman">Postman .json</button>' +
        '<button class="reqd-chip" data-export="insomnia">Insomnia .yaml</button>' +
        '<button class="reqd-chip" data-export="har">HAR</button>' +
      '</div>' +
      '<div class="reqd-export-row-label">open elsewhere:</div>' +
      '<div class="reqd-export-chips">' +
        '<button class="reqd-chip" data-open="sandbox">open in Sandbox</button>' +
        '<button class="reqd-chip" data-open="permalink">copy permalink</button>' +
        '<button class="reqd-chip" data-open="newtab">open in new tab</button>' +
      '</div>' +
    '</div>';
  }

  function _wireExportSection(root, snips, rid) {
    const langMap = { curl: 'bash', python: 'python', js: 'javascript', node: 'javascript' };
    const codeWrap = root.querySelector('#reqd-code-body');
    // Cache the current raw text so the copy button can read it without
    // having to reconstruct from the line-numbered DOM.
    let _currentRawText = snips.curl || '';

    function _swapTab(which) {
      const lang = langMap[which] || 'bash';
      const raw = snips[which] || '';
      _currentRawText = raw;
      // Replace the line-numbered code in-place, keeping the copy button.
      const oldCode = codeWrap.querySelector('.reqd-code');
      if (oldCode) oldCode.remove();
      const tmp = document.createElement('div');
      tmp.innerHTML = _renderLineNumberedCode(raw, lang);
      const newCode = tmp.firstElementChild;
      // Insert AFTER the copy button so copy stays floating top-right
      codeWrap.appendChild(newCode);
      // Subtle CSS-driven fade-in (not WAAPI — see earlier closing notes
      // on the .finished-Promise race condition in fadeReplace).
      newCode.style.opacity = '0.3';
      void newCode.offsetWidth;
      newCode.style.transition = 'opacity 140ms cubic-bezier(0.22,1,0.36,1)';
      newCode.style.opacity = '1';
    }

    root.querySelectorAll('.reqd-code-tabs button').forEach(btn => {
      btn.addEventListener('click', () => {
        root.querySelectorAll('.reqd-code-tabs button')
          .forEach(b => b.setAttribute('aria-pressed', 'false'));
        btn.setAttribute('aria-pressed', 'true');
        _swapTab(btn.dataset.snip);
      });
    });
    // Copy reads from the cached raw text (the DOM is now line-by-line
    // wrapped — textContent would include all the gutter prefixes).
    const copyBtn = root.querySelector('#reqd-copy');
    if (copyBtn) {
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(_currentRawText);
          copyBtn.classList.add('copied');
          copyBtn.textContent = 'copied';
          setTimeout(() => { copyBtn.classList.remove('copied'); copyBtn.textContent = 'copy'; }, 1500);
        } catch (e) {}
      });
    }
    // File-export chips
    root.querySelectorAll('[data-export]').forEach(chip => {
      chip.addEventListener('click', () => _downloadExport(chip.dataset.export, snips, rid));
    });
    // Share chips
    root.querySelectorAll('[data-open]').forEach(chip => {
      chip.addEventListener('click', () => _handleOpenAction(chip.dataset.open, rid));
    });
  }

  function _downloadExport(kind, snips, rid) {
    let content = '', fname = 'request_' + rid, mime = 'text/plain';
    const url = 'http://localhost:8888';
    if (kind === 'postman') {
      const collection = {
        info: { name: 'APIN request ' + rid, schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json' },
        item: [{
          name: 'request_' + rid,
          request: {
            method: 'GET',
            header: [{ key: 'Authorization', value: 'Bearer apin_<your_token>' }],
            url: { raw: url + '/api/version' },
          },
        }],
      };
      content = JSON.stringify(collection, null, 2);
      fname += '.postman_collection.json';
      mime = 'application/json';
    } else if (kind === 'insomnia') {
      content = '_type: export\n__export_format: 4\nresources:\n  - _type: request\n    name: request_' + rid +
                '\n    method: GET\n    url: ' + url + '/api/version\n    headers:\n      - name: Authorization\n        value: Bearer apin_<your_token>\n';
      fname += '.insomnia.yaml';
      mime = 'text/yaml';
    } else if (kind === 'har') {
      const har = {
        log: { version: '1.2', creator: { name: 'APIN', version: '1.0' },
          entries: [{
            startedDateTime: new Date().toISOString(),
            request: { method: 'GET', url: url + '/api/version', httpVersion: 'HTTP/1.1',
              cookies: [], headers: [{ name: 'Authorization', value: 'Bearer apin_<your_token>' }],
              queryString: [], headersSize: -1, bodySize: 0 },
            response: { status: 200, statusText: 'OK', httpVersion: 'HTTP/1.1',
              cookies: [], headers: [], content: { size: 0, mimeType: 'application/json' },
              redirectURL: '', headersSize: -1, bodySize: 0 },
            cache: {}, timings: { send: 0, wait: 0, receive: 0 },
          }],
        },
      };
      content = JSON.stringify(har, null, 2);
      fname += '.har';
      mime = 'application/json';
    } else return;
    const blob = new Blob([content], { type: mime });
    const aurl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = aurl; a.download = fname;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(aurl), 1000);
  }

  function _handleOpenAction(kind, rid) {
    if (kind === 'sandbox') {
      window.open('/account/api/sandbox', '_blank', 'noopener');
    } else if (kind === 'permalink') {
      const url = location.origin + location.pathname + '#req-' + rid;
      navigator.clipboard.writeText(url).then(() => {
        // Subtle confirmation via toast if available, else nothing
        try { window.APIN && APIN.toast && APIN.toast.show && APIN.toast.show('link copied'); } catch (e) {}
      }).catch(() => {});
    } else if (kind === 'newtab') {
      window.open(location.pathname + '#req-' + rid, '_blank', 'noopener');
    }
  }

  function kpiBlock(label, value, tone) {
    const c = tone === 'danger' ? 'var(--c-danger)' : 'var(--ink)';
    return '<div style="background:var(--paper-deep);padding:10px 12px;border-radius:6px;">' +
      '<div style="font-size:10.5px;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;font-weight:600">' + escHtml(label) + '</div>' +
      '<div style="font-family:\'Fraunces\',serif;font-size:20px;color:' + c + ';font-variant-numeric:tabular-nums">' + value + '</div>' +
    '</div>';
  }
  $('drawer-close').addEventListener('click', closeDrawer);
  $('drawer-mask').addEventListener('click', closeDrawer);
  $('reqd-close').addEventListener('click', closeReqd);
  $('reqd-mask').addEventListener('click', closeReqd);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeDrawer(); closeReqd(); }
  });

  // ─── Filter event wiring ─────────────────────────────────────────────
  function applyFilters() {
    writeHashFilters();
    refreshAll();
  }
  document.querySelectorAll('#range-strip button').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#range-strip button').forEach(x => x.removeAttribute('aria-pressed'));
      b.setAttribute('aria-pressed', 'true');
      filterState.range = b.dataset.range;
      recentCursor = null;
      applyFilters();
    });
  });
  document.querySelectorAll('#ts-mode button').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#ts-mode button').forEach(x => x.removeAttribute('aria-pressed'));
      b.setAttribute('aria-pressed', 'true');
      filterState.mode = b.dataset.mode;
      writeHashFilters();
      loadTimeseries();
    });
  });
  $('logy-toggle').addEventListener('change', e => {
    filterState.logY = e.target.checked;
    writeHashFilters();
    loadTimeseries();
  });
  $('filter-key').addEventListener('change', e => { filterState.key_id = e.target.value; recentCursor = null; applyFilters(); });
  $('filter-env').addEventListener('change', e => { filterState.env = e.target.value; recentCursor = null; applyFilters(); });
  $('filter-status').addEventListener('change', e => { filterState.status = e.target.value; recentCursor = null; applyFilters(); });
  let endpointTimer;
  $('filter-endpoint').addEventListener('input', e => {
    clearTimeout(endpointTimer);
    endpointTimer = setTimeout(() => { filterState.endpoint = e.target.value.trim(); recentCursor = null; applyFilters(); }, 280);
  });
  $('btn-refresh').addEventListener('click', () => { recentCursor = null; refreshAll(); });
  $('recent-more').addEventListener('click', () => loadRecent({ append: true }));

  // 9.I — KPI tile click handlers (cascade filter or switch chart mode)
  document.querySelectorAll('.kpi[data-kpi-action]').forEach(tile => {
    tile.addEventListener('click', () => {
      const action = tile.dataset.kpiAction;
      if (action === 'reset') {
        filterState.status = '';
        filterState.endpoint = '';
        if ($('filter-status')) $('filter-status').value = '';
        if ($('filter-endpoint')) $('filter-endpoint').value = '';
        applyFilters();
      } else if (action === 'filter-errors') {
        filterState.status = '4xx';
        if ($('filter-status')) $('filter-status').value = '4xx';
        applyFilters();
      } else if (action === 'filter-429') {
        filterState.status = '429';
        if ($('filter-status')) $('filter-status').value = '429';
        applyFilters();
      } else if (action === 'mode-latency') {
        filterState.mode = 'latency';
        document.querySelectorAll('#ts-mode button').forEach(x => x.removeAttribute('aria-pressed'));
        const tgt = document.querySelector('#ts-mode button[data-mode="latency"]');
        if (tgt) tgt.setAttribute('aria-pressed', 'true');
        loadTimeseries();
      } else if (action === 'mode-by-endpoint') {
        filterState.mode = 'by_endpoint';
        document.querySelectorAll('#ts-mode button').forEach(x => x.removeAttribute('aria-pressed'));
        const tgt = document.querySelector('#ts-mode button[data-mode="by_endpoint"]');
        if (tgt) tgt.setAttribute('aria-pressed', 'true');
        loadTimeseries();
      } else if (action === 'mode-bytes') {
        filterState.mode = 'bytes';
        document.querySelectorAll('#ts-mode button').forEach(x => x.removeAttribute('aria-pressed'));
        const tgt = document.querySelector('#ts-mode button[data-mode="bytes"]');
        if (tgt) tgt.setAttribute('aria-pressed', 'true');
        loadTimeseries();
      }
    });
  });

  // ─── CSV export ──────────────────────────────────────────────────────
  $('btn-export').addEventListener('click', async () => {
    const qs = new URLSearchParams({ range: filterState.range, format: 'csv' });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env) qs.set('env', filterState.env);
    if (filterState.status) qs.set('status', filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    window.location.href = '/api/account/usage/requests?' + qs.toString();
  });

  // ─── Populate key dropdown ────────────────────────────────────────────
  async function loadKeys() {
    const { body } = await api('/api/account/keys?limit=100');
    if (!body || !body.ok) {
      console.warn('loadKeys: body not ok', body && body.error);
      return;
    }
    availableKeys = (body.data.items || body.data.keys || []);
    const sel = $('filter-key');
    sel.innerHTML = '<option value="">all keys (' + availableKeys.length + ')</option>' +
      availableKeys.map(k =>
        '<option value="' + escHtml(k.public_id) + '">' +
        escHtml(k.name || k.public_id) + ' · ' + escHtml(k.environment || 'live') +
        '</option>'
      ).join('');
    if (filterState.key_id) sel.value = filterState.key_id;
  }

  // ─── Refresh all panels in parallel ──────────────────────────────────
  // 9.N.2 · NO global skeleton fade. Each loader updates its own chart
  // host smoothly via the chart library's own enter/morph animations.
  // The page never goes "washed out" — old data stays visible until new
  // data lands. Filter changes call refreshAll() but the user sees
  // individual charts updating, not a full-page reload flash.
  async function refreshAll() {
    renderChipStrip();
    await Promise.allSettled([
      loadSummary(),
      loadTimeseries(),
      loadTop('endpoints', 'top-ep-host', 'top-ep-aux', { color_token: 'series-1' }),
      loadDonut(),
      loadLatencyHistogram(),
      loadTop('keys', 'top-keys-host', null, { color_token: 'series-0' }),
      loadTop('ips', 'top-ips-host', null, { color_token: 'series-2' }),
      loadTop('error_codes', 'top-errors-host', null, { color_token: 'series-3' }),
      loadRecent({ append: false }),
      loadSparklines(),
      // 9.N.5 · 6 new charts
      loadHeatmap(),
      loadPerEndpointDetail(),  // fans into treemap + spark grid + boxplot + quadrant
    ]);
    // 9.N.5.f · Insights run AFTER the per-endpoint detail has resolved
    // so the "What's noteworthy" card sees the same cache the other charts do
    loadInsights();
  }

  // 9.N.2 · Quiet refresh on tab return — KPIs only.
  // 9.N.5 fix · DO NOT re-render charts on tab return. Charts only animate
  // on (a) initial page load, (b) explicit user action (filter, refresh,
  // mode change). Tab return must be silent — the odometer's no-change-no-
  // reroll behavior handles the KPI side gracefully. Charts keep last-good
  // data on screen until the user explicitly hits refresh or changes a filter.
  async function quietRefresh() {
    await loadSummary();
  }

  // 9.N.2 · Filter chip strip. Each active filter renders as a dismissible
  // chip below the toolbar. Chips slide in via APIN.fx.chipSlide. "Clear all"
  // removes everything and resets to defaults.
  function renderChipStrip() {
    const strip = document.getElementById('chip-strip');
    if (!strip) return;
    const chips = [];
    if (filterState.key_id) {
      const sel = document.getElementById('filter-key');
      const opt = sel && [...sel.options].find(o => o.value === filterState.key_id);
      const label = opt ? opt.textContent : filterState.key_id;
      chips.push({ kind: 'key',      key: 'key_id',   icon: 'i-key',       label: 'key: ' + label });
    }
    if (filterState.env) {
      chips.push({ kind: 'env',      key: 'env',      icon: 'i-leaf',      label: 'env: ' + filterState.env });
    }
    if (filterState.status) {
      chips.push({ kind: 'status',   key: 'status',   icon: 'i-pie-chart', label: 'status: ' + filterState.status });
    }
    if (filterState.endpoint) {
      chips.push({ kind: 'endpoint', key: 'endpoint', icon: 'i-route',     label: 'endpoint: ' + filterState.endpoint });
    }
    const out = [];
    chips.forEach(c => {
      out.push(
        '<span class="chip" data-chip-key="' + c.key + '">' +
          '<svg class="chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><use href="#' + c.icon + '"/></svg>' +
          '<span class="chip-label">' + escHtml(c.label) + '</span>' +
          '<button class="chip-x" aria-label="remove filter" data-chip-remove="' + c.key + '">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M5 5 L19 19 M19 5 L5 19"/></svg>' +
          '</button>' +
        '</span>'
      );
    });
    if (chips.length > 0) {
      out.push('<button class="chip-clear" id="chip-clear-all">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><use href="#i-x"/></svg>' +
        ' clear all</button>');
    }
    const prevCount = strip.children.length;
    strip.innerHTML = out.join('');
    strip.style.display = chips.length ? 'flex' : 'none';
    // Animate-in new chips (cheap heuristic — animate all when count grows)
    if (chips.length > prevCount && window.APIN && APIN.fx) {
      [...strip.querySelectorAll('.chip')].forEach((el, i) => {
        APIN.fx.chipSlide(el);
      });
    }
    // Wire dismiss buttons
    strip.querySelectorAll('[data-chip-remove]').forEach(btn => {
      btn.addEventListener('click', () => {
        const k = btn.getAttribute('data-chip-remove');
        if (k === 'key_id')   filterState.key_id   = null;
        if (k === 'env')      filterState.env      = null;
        if (k === 'status')   filterState.status   = null;
        if (k === 'endpoint') filterState.endpoint = null;
        applyStateToUI();
        applyFilters();
      });
    });
    const clearBtn = document.getElementById('chip-clear-all');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        filterState.key_id = null;
        filterState.env = null;
        filterState.status = null;
        filterState.endpoint = null;
        applyStateToUI();
        applyFilters();
      });
    }
  }

  // ─── Apply initial filterState to UI controls ────────────────────────
  function applyStateToUI() {
    document.querySelectorAll('#range-strip button').forEach(b => {
      if (b.dataset.range === filterState.range) b.setAttribute('aria-pressed', 'true');
      else b.removeAttribute('aria-pressed');
    });
    document.querySelectorAll('#ts-mode button').forEach(b => {
      if (b.dataset.mode === filterState.mode) b.setAttribute('aria-pressed', 'true');
      else b.removeAttribute('aria-pressed');
    });
    // 9.N.2 · always sync, even when filterState slot is null — otherwise
    // chip × / clear-all doesn't reset the underlying control and a re-apply
    // would re-set the cleared filter.
    $('filter-env').value      = filterState.env      || '';
    $('filter-status').value   = filterState.status   || '';
    $('filter-endpoint').value = filterState.endpoint || '';
    const keySel = $('filter-key');
    if (keySel) keySel.value   = filterState.key_id   || '';
    $('logy-toggle').checked = !!filterState.logY;
  }

  // ─── Auto-refresh ────────────────────────────────────────────────────
  let pollTimer = null;
  function startPoll() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      if (document.visibilityState !== 'visible') return;
      loadSummary();
      loadTimeseries();
      loadDonut();
      if (recentCursor === null) loadRecent({ append: false });
    }, 15_000);
  }
  // 9.N.2 · On tab return, do a QUIET refresh (no global fade, no chart
  // rebuild). Charts pick up new data via their own morph animations.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      quietRefresh();
      startPoll();
    } else {
      if (pollTimer) clearInterval(pollTimer);
    }
  });

  // ─── Keyboard shortcuts ──────────────────────────────────────────────
  document.addEventListener('keydown', ev => {
    if (ev.target.matches('input, select, textarea')) return;
    const rangeMap = { '1': '15m', '2': '1h', '3': '6h', '4': '24h', '5': '7d', '6': '30d' };
    if (rangeMap[ev.key]) {
      filterState.range = rangeMap[ev.key];
      applyStateToUI();
      applyFilters();
    } else if (ev.key.toLowerCase() === 'r') {
      recentCursor = null;
      refreshAll();
    } else if (ev.key.toLowerCase() === 'l') {
      filterState.logY = !filterState.logY;
      applyStateToUI();
      loadTimeseries();
    } else if (ev.key === 'Escape' && !window.APIN?.lightbox?.isOpen?.()) {
      // 9.N.2 · ESC clears all filters (when lightbox isn't open)
      if (filterState.key_id || filterState.env || filterState.status || filterState.endpoint) {
        filterState.key_id = null;
        filterState.env = null;
        filterState.status = null;
        filterState.endpoint = null;
        applyStateToUI();
        applyFilters();
      }
    }
  });

  // ═══════════════════════════════════════════════════════════════════════
  // ─── 9.N.6 · Lightbox expand — universal dispatcher + sub-view builders
  // ═══════════════════════════════════════════════════════════════════════
  // Each chart card gets an expand button (⤢) in its header. Click opens a
  // centered lightbox with 4 sub-sections:
  //   1. Snapshot — high-level KPIs about the data shown
  //   2. Breakdown — drilled-down detail (table or sub-chart)
  //   3. Evidence — sample requests / rows
  //   4. Actions — drill into / export CSV / share link
  //
  // The lightbox itself is built by apin_lightbox.js. Here we just provide
  // the per-chart builders + wire up the expand button per card.

  const EXPAND_ICON_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true">'
    + '<path d="M4 10 L4 4 L10 4 M14 4 L20 4 L20 10 M20 14 L20 20 L14 20 M10 20 L4 20 L4 14"/>'
    + '</svg>';

  // Map: host-id → { key, title, subtitle, build }
  const CHART_LIGHTBOX = {
    'ts-host':         { key: 'timeseries',    title: 'Requests over time',     build: _lbTimeSeries    },
    'donut-host':      { key: 'status_mix',    title: 'Status mix',              build: _lbDonut         },
    'hist-host':       { key: 'latency_dist',  title: 'Latency distribution',    build: _lbHistogram     },
    'top-ep-host':     { key: 'top_endpoints', title: 'Top endpoints',           build: _lbTopEndpoints  },
    'top-keys-host':   { key: 'top_keys',      title: 'Top keys',                build: _lbTopKeys       },
    'top-ips-host':    { key: 'top_ips',       title: 'Top IPs',                 build: _lbTopIPs        },
    'top-errors-host': { key: 'top_errors',    title: 'Top error codes',         build: _lbTopErrors     },
    'heatmap-host':    { key: 'heatmap',       title: 'Activity calendar',       build: _lbHeatmap       },
    'treemap-host':    { key: 'treemap',       title: 'Endpoint treemap',        build: _lbTreemap       },
    'sparkgrid-host':  { key: 'spark_grid',    title: 'Per-endpoint shapes',     build: _lbSparkGrid     },
    'quadrant-host':   { key: 'quadrant',      title: 'Endpoint health quadrant',build: _lbQuadrant      },
    'boxplot-host':    { key: 'boxplot',       title: 'Latency per endpoint',    build: _lbBoxplot       },
    'insights-host':   { key: 'insights',      title: "What's noteworthy",       build: _lbInsights      },
    'livestream-host': { key: 'live_tail',     title: "Live tail",               build: _lbLiveStream    },
    'livepulse-host':  { key: 'live_pulse',    title: "Live pulse",              build: _lbLivePulse     },
  };

  function _injectExpandButtons() {
    Object.entries(CHART_LIGHTBOX).forEach(([hostId, spec]) => {
      const host = document.getElementById(hostId);
      if (!host) return;
      const card = host.closest('.card');
      if (!card) return;
      const head = card.querySelector('.card-head');
      if (!head || head.querySelector('.card-expand')) return;
      const btn = document.createElement('button');
      btn.className = 'card-expand';
      btn.setAttribute('aria-label', 'Expand');
      btn.setAttribute('data-chart-key', spec.key);
      btn.innerHTML = EXPAND_ICON_SVG;
      btn.addEventListener('click', e => {
        e.preventDefault();
        if (window.APIN && APIN.lightbox) {
          APIN.lightbox.open({
            sourceCard: card,
            title: spec.title,
            subtitle: 'range: ' + filterState.range + (filterState.key_id ? ' · key: ' + filterState.key_id : '') + (filterState.status ? ' · status: ' + filterState.status : ''),
            hashKey: spec.key,
            build: (panel) => spec.build(panel),
          });
        }
      });
      head.appendChild(btn);
    });
    // 9.N.6.f · KPI tile expand buttons (small ⤢ in upper-right corner).
    // Clicking the corner opens the tile's lightbox; clicking the tile
    // body keeps its existing preset action (filter/mode).
    document.querySelectorAll('.kpi[data-k]').forEach(tile => {
      if (tile.querySelector('.kpi-expand')) return;
      const k = tile.getAttribute('data-k');
      const btn = document.createElement('button');
      btn.className = 'kpi-expand';
      btn.setAttribute('aria-label', 'Expand KPI');
      btn.innerHTML = EXPAND_ICON_SVG;
      btn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();   // don't trigger the tile's preset-action click
        if (window.APIN && APIN.lightbox) {
          APIN.lightbox.open({
            sourceCard: tile,
            title: _kpiTitle(k),
            subtitle: 'KPI · range: ' + filterState.range,
            hashKey: 'kpi-' + k,
            build: (panel) => _lbKpiTile(panel, k),
          });
        }
      });
      tile.appendChild(btn);
    });
  }
  function _kpiTitle(k) {
    return { requests: 'Requests', errors: 'Errors',
             rate_limited: 'Rate-limited (429)', latency_p95_ms: 'p95 latency',
             active_keys: 'Keys with traffic', bytes_out: 'Bytes out' }[k] || k;
  }

  // 9.N.6.f · KPI tile lightbox — matches the discussed mockup:
  //   big number → 24h sparkline → period breakdown → breakdown by dim → comparison → action
  async function _lbKpiTile(panel, kpiKey) {
    // Fetch the summary for the headline numbers
    const qs = new URLSearchParams({ range: filterState.range });
    if (filterState.key_id)   qs.set('key_id',   filterState.key_id);
    if (filterState.env)      qs.set('env',      filterState.env);
    if (filterState.status)   qs.set('status',   filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    const { body } = await api('/api/account/usage/summary?' + qs);
    const k = (body && body.ok && body.data.kpis && body.data.kpis[kpiKey]) || {};
    const cur = k.current != null ? k.current : 0;
    // Big number header
    const head =
      '<div style="font-family:Fraunces,serif;font-weight:500;font-size:48px;line-height:1.1;color:var(--ink)">'
      + (kpiKey === 'bytes_out' ? fmtBytes(cur) : fmtNum(cur))
      + (kpiKey === 'latency_p95_ms' ? '<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;color:var(--ink-soft);margin-left:6px">ms</span>' : '')
      + '</div>';
    // 24h sparkline
    _appendSection(panel, _kpiTitle(kpiKey),
      head + '<div id="lb-kpi-spark" style="height:80px;margin-top:8px"></div>'
      + '<div style="display:flex;justify-content:space-between;font-family:\'JetBrains Mono\',monospace;font-size:10.5px;color:var(--ink-soft);margin-top:4px;letter-spacing:0.08em"><span>24h ago</span><span>now</span></div>'
    );
    setTimeout(async () => {
      try {
        const url = '/api/account/usage/timeseries?range=24h&mode=' + (kpiKey === 'latency_p95_ms' ? 'latency' : kpiKey === 'bytes_out' ? 'bytes' : 'total');
        const tres = await api(url);
        const buckets = (tres.body && tres.body.ok && tres.body.data.buckets) || [];
        const meta = (tres.body && tres.body.ok && tres.body.data.series_meta) || [];
        const series = meta[0] && meta[0].key;
        if (!series) return;
        const host = document.getElementById('lb-kpi-spark');
        if (!host) return;
        renderSparkline(host, buckets.map(b => ({ t: b.t, v: Number((b.values || {})[series] || 0) })));
      } catch (e) {}
    }, 350);

    // Period breakdown — peak/trough/active/avg-when-active.
    // Pull primary series timeseries for the current range
    let breakdownHtml = '<div class="lb-empty">computing…</div>';
    try {
      const qs2 = new URLSearchParams({ range: filterState.range, mode: kpiKey === 'latency_p95_ms' ? 'latency' : 'total' });
      if (filterState.key_id) qs2.set('key_id', filterState.key_id);
      const tr = await api('/api/account/usage/timeseries?' + qs2);
      const bb = (tr.body && tr.body.ok && tr.body.data.buckets) || [];
      const md = (tr.body && tr.body.ok && tr.body.data.series_meta) || [];
      const k0 = md[0] && md[0].key;
      if (k0 && bb.length) {
        const vals = bb.map(b => Number((b.values || {})[k0] || 0));
        const peak = Math.max(...vals), trough = Math.min(...vals.filter(v => v > 0).concat([0]));
        const peakI = vals.indexOf(peak);
        const active = vals.filter(v => v > 0).length;
        const sum = vals.reduce((a, v) => a + v, 0);
        const avgActive = active ? Math.round(sum / active) : 0;
        breakdownHtml = '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;line-height:1.9;color:var(--ink)">'
          + fmtNum(sum) + ' total<br>'
          + '├─ peak  ' + fmtNum(peak) + '/bucket @ ' + ((bb[peakI]?.t || '').slice(11, 16)) + '<br>'
          + '├─ trough ' + fmtNum(trough) + '/bucket' + (active < vals.length ? ' (' + (vals.length - active) + ' idle buckets)' : '') + '<br>'
          + '├─ active ' + active + ' of ' + vals.length + ' = ' + ((active / vals.length) * 100).toFixed(0) + '% duty cycle<br>'
          + '└─ avg when active: ' + fmtNum(avgActive) + '/bucket'
          + '</div>';
      }
    } catch (e) {}
    _appendSection(panel, 'Period breakdown', breakdownHtml);

    // Breakdown by — key / env / method
    try {
      const dims = [['keys', 'key'], ['envs', 'env'], ['methods', 'method']];
      const cards = [];
      for (const [d, label] of dims) {
        if (d === 'envs') continue; // envs dim doesn't exist; just keep key/method
        const qd = new URLSearchParams({ range: filterState.range, dim: d, limit: 3 });
        if (filterState.key_id) qd.set('key_id', filterState.key_id);
        if (filterState.status) qd.set('status', filterState.status);
        if (filterState.endpoint) qd.set('endpoint', filterState.endpoint);
        const dr = await api('/api/account/usage/top?' + qd);
        const di = (dr.body && dr.body.ok && dr.body.data.items) || [];
        const dTot = di.reduce((a, it) => a + (Number(it.count) || 0), 0);
        const maxV = Math.max(1, ...di.map(it => Number(it.count) || 0));
        cards.push(
          '<div style="margin-bottom:10px">'
          + '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11.5px;color:var(--ink-soft);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.08em">' + label + '</div>'
          + (di.length ? di.map(it => {
              const w = ((Number(it.count) || 0) / maxV) * 100;
              const pct = dTot ? ((Number(it.count) || 0) / dTot * 100) : 0;
              return '<div style="display:grid;grid-template-columns:80px 1fr 70px 70px;gap:10px;font-family:\'JetBrains Mono\',monospace;font-size:11.5px;padding:3px 0">'
                + '<div style="text-align:right;font-variant-numeric:tabular-nums">' + fmtNum(it.count) + '</div>'
                + '<div style="height:11px;background:var(--paper-deep);position:relative;border:1px solid var(--paper-edge);top:3px"><div style="position:absolute;left:0;top:0;height:100%;width:' + w + '%;background:var(--ink);opacity:0.62"></div></div>'
                + '<div>' + escHtml(it.label) + '</div>'
                + '<div style="text-align:right;color:var(--ink-soft)">(' + pct.toFixed(0) + '%)</div>'
                + '</div>';
            }).join('') : '<div class="lb-empty">no dim data</div>')
          + '</div>'
        );
      }
      _appendSection(panel, 'Breakdown by', cards.join(''));
    } catch (e) {
      _appendSection(panel, 'Breakdown by', '<div class="lb-empty">unable to break down</div>');
    }

    // Comparison
    const deltaPct = k.delta_pct;
    const compareHtml = (deltaPct == null)
      ? '<div class="lb-empty">no previous-period data yet — first day of activity</div>'
      : '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink)">'
        + 'vs prev ' + filterState.range + ': ' + (deltaPct > 0 ? '+' : '') + Number(deltaPct).toFixed(1) + '%'
        + '</div>';
    _appendSection(panel, 'Comparison', compareHtml);

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'view all ' + fmtNum(cur) + ' requests' },
      { kind: 'share-link', icon: 'i-link', label: 'share link' },
    ]));
    _wireActions(panel);
  }

  // ─── Shared lightbox helpers ─────────────────────────────────────────
  function _kpiTile(label, value, sublabel) {
    return '<div class="lb-kpi">'
      +   '<div class="lb-kpi-lbl">' + escHtml(label) + '</div>'
      +   '<div class="lb-kpi-val">' + escHtml(value) + '</div>'
      +   (sublabel ? '<div class="lb-kpi-sub">' + escHtml(sublabel) + '</div>' : '')
      + '</div>';
  }
  function _kpiGrid(tiles) {
    return '<div class="lb-kpi-grid">' + tiles.join('') + '</div>';
  }
  function _table(headers, rows) {
    return '<table class="lb-table"><thead><tr>'
      + headers.map(h => '<th' + (h.numeric ? ' class="num"' : '') + '>' + escHtml(h.label || h) + '</th>').join('')
      + '</tr></thead><tbody>'
      + (rows.length ? rows.map(r => '<tr>' + r.map((c, i) => '<td' + (headers[i] && headers[i].numeric ? ' class="num"' : '') + '>' + (c == null ? '—' : escHtml(c)) + '</td>').join('') + '</tr>').join('')
                     : '<tr><td colspan="' + headers.length + '"><div class="lb-empty">no rows</div></td></tr>')
      + '</tbody></table>';
  }
  function _emptyState(msg) {
    return '<div class="lb-empty">' + escHtml(msg) + '</div>';
  }
  function _actionRow(actions) {
    return '<div class="lb-actions">'
      + actions.map(a => '<button class="lb-action" data-act="' + a.kind + '" data-val="' + escHtml(a.value || '') + '">'
          + (a.icon ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><use href="#' + a.icon + '"/></svg>' : '')
          + escHtml(a.label)
          + '</button>').join('')
      + '</div>';
  }
  function _wireActions(panel) {
    panel.querySelectorAll('.lb-action').forEach(btn => {
      btn.addEventListener('click', () => {
        const kind = btn.getAttribute('data-act');
        const val = btn.getAttribute('data-val');
        if (kind === 'filter-endpoint') {
          filterState.endpoint = val;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        } else if (kind === 'filter-status') {
          filterState.status = val;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        } else if (kind === 'filter-key') {
          filterState.key_id = val;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        } else if (kind === 'export-csv') {
          const url = '/api/account/usage/requests?range=' + filterState.range
            + (filterState.key_id ? '&key_id=' + encodeURIComponent(filterState.key_id) : '')
            + (filterState.status ? '&status=' + encodeURIComponent(filterState.status) : '')
            + (filterState.endpoint ? '&endpoint=' + encodeURIComponent(filterState.endpoint) : '')
            + '&format=csv';
          window.open(url, '_blank');
        } else if (kind === 'share-link') {
          const fullUrl = location.origin + location.pathname + location.hash;
          if (navigator.clipboard) {
            navigator.clipboard.writeText(fullUrl).then(() => {
              btn.textContent = 'copied ✓';
              setTimeout(() => { btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-link"/></svg> share link'; }, 1500);
            });
          }
        }
      });
    });
  }
  function _appendSection(panel, title, contentHtml) {
    const sec = APIN.lightbox.section(title);
    sec.insertAdjacentHTML('beforeend', contentHtml);
    panel.appendChild(sec);
    return sec;
  }

  // ─── Per-chart builders (9.N.6 redesign — matches discussed_expanded_states/) ──

  async function _lbTimeSeries(panel) {
    const qs = new URLSearchParams({ range: filterState.range, mode: filterState.mode || 'total' });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env)    qs.set('env', filterState.env);
    if (filterState.status) qs.set('status', filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    if (filterState.compare) qs.set('compare', 'prev_period');
    const { body } = await api('/api/account/usage/timeseries?' + qs);
    const buckets = (body && body.ok && body.data.buckets) || [];
    const meta    = (body && body.ok && body.data.series_meta) || [];
    const primaryKey = meta[0] && meta[0].key;
    const counts = buckets.map(b => Number((b.values || {})[primaryKey] || 0));
    let total = 0, peak = 0, peakAt = '', peakIdx = -1;
    counts.forEach((v, i) => { total += v; if (v > peak) { peak = v; peakAt = buckets[i].t; peakIdx = i; } });

    // Header line: "357 requests · peak 67/min @ 12:13 · p50 176ms · p99 2.4s"
    const lats = (_perEndpointCache || []).flatMap(e => Array(Math.min(20, e.count || 0)).fill(e.p50));
    const lats2 = (_perEndpointCache || []).map(e => e.p50).filter(Boolean).sort((a,b) => a-b);
    const p50 = lats2.length ? lats2[Math.floor(lats2.length / 2)] : 0;
    const p99cache = (_perEndpointCache || []).map(e => e.p99).filter(Boolean);
    const p99 = p99cache.length ? Math.max(...p99cache) : 0;

    // Section 1: Headline summary + bigger chart re-render
    const headline = _appendSection(panel, 'Headline',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   fmtNum(total) + ' requests · peak ' + fmtNum(peak) + '/bucket'
      +   (peakAt ? ' @ ' + peakAt : '')
      +   ' · p50 ' + fmtNum(p50) + 'ms · p99 ' + fmtNum(p99) + 'ms'
      + '</div>'
      + '<div id="lb-ts-chart" style="height:280px"></div>');
    // Re-render the chart larger
    setTimeout(() => {
      const chartHost = document.getElementById('lb-ts-chart');
      if (chartHost && body.data) {
        APIN.charts.timeseries(chartHost, body.data, {
          mode: (filterState.mode === 'by_status' || filterState.mode === 'by_endpoint') ? 'stacked' : 'line',
          logY: filterState.logY,
          height: 280,
          showLegend: true,
          showMarginalia: true,
        });
      }
    }, 350);

    // Section 2: Insights (data, not chart docs)
    const insightBullets = [];
    // 1. Concentration ratio
    if (counts.length > 3 && total > 0) {
      const sorted = [...counts].map((v, i) => ({ v, i })).sort((a,b) => b.v - a.v);
      let cumPct = 0, n = 0;
      for (const x of sorted) {
        cumPct += x.v / total;
        n++;
        if (cumPct >= 0.78) break;
      }
      if (n > 0 && n < counts.length * 0.5) {
        const firstI = sorted.slice(0, n).map(x => x.i).sort((a,b) => a-b);
        const lo = buckets[firstI[0]]?.t || '', hi = buckets[firstI[n-1]]?.t || '';
        insightBullets.push('78% of traffic packed into ' + (lo.slice(11, 16)) + '–' + (hi.slice(11, 16)) + ' (' + n + ' buckets)');
      }
    }
    // 2. Error spike — find biggest bucket of errors
    const errKey = meta.find(m => /^(5xx|4xx|errors)/.test(m.key))?.key;
    if (errKey) {
      const errCounts = buckets.map(b => Number((b.values || {})[errKey] || 0));
      const errMax = Math.max(...errCounts, 0);
      if (errMax >= 3) {
        const idx = errCounts.indexOf(errMax);
        const t = buckets[idx]?.t || '';
        insightBullets.push('Error spike ' + (t.slice(11, 16)) + ': ' + errMax + ' × ' + errKey + ' in one bucket');
      }
    }
    // 3. Top endpoint concentration
    const topEp = (_perEndpointCache || [])[0];
    if (topEp && topEp.pct >= 25) {
      insightBullets.push(topEp.label + ' = ' + Number(topEp.pct).toFixed(1) + '% of all traffic in window');
    }
    // 4. Gap analysis
    const activeBuckets = counts.filter(v => v > 0).length;
    const dutyPct = ((activeBuckets / Math.max(1, counts.length)) * 100).toFixed(0);
    if (dutyPct < 80) insightBullets.push('Active in ' + dutyPct + '% of buckets — ' + (100 - dutyPct) + '% idle window');

    const insightsHtml = insightBullets.length
      ? '<ul style="margin:0;padding-left:18px;font-family:Fraunces,serif;font-size:13.5px;color:var(--ink);line-height:1.85">'
        + insightBullets.map(t => '<li>' + escHtml(t) + '</li>').join('')
        + '</ul>'
      : '<div class="lb-empty">no notable patterns in this window</div>';
    _appendSection(panel, 'Insights (data, not chart docs)', insightsHtml);

    // Section 3: Top minutes by volume (with mini bars)
    const ranked = buckets.map((b, i) => ({ t: b.t, count: counts[i] }))
      .filter(x => x.count > 0)
      .sort((a, b) => b.count - a.count).slice(0, 10);
    const maxRanked = Math.max(1, ...ranked.map(r => r.count));
    const minutesHtml = ranked.length
      ? '<div style="display:grid;grid-template-columns:80px 1fr 70px;gap:12px;align-items:center;font-family:\'JetBrains Mono\',monospace;font-size:12px">'
        + ranked.map(r => {
            const w = (r.count / maxRanked) * 100;
            return '<div style="color:var(--ink-soft)">' + escHtml((r.t || '').slice(11, 16)) + '</div>'
              + '<div style="height:14px;background:var(--paper-deep);position:relative">'
              +   '<div style="position:absolute;left:0;top:0;height:100%;width:' + w + '%;background:var(--ink);opacity:0.78"></div>'
              + '</div>'
              + '<div style="text-align:right;font-variant-numeric:tabular-nums">' + fmtNum(r.count) + ' req</div>';
          }).join('')
        + '</div>'
      : '<div class="lb-empty">no active buckets</div>';
    _appendSection(panel, 'Top buckets by volume', minutesHtml);

    // Section 4: Compare
    const prev = body.data.prev_buckets || null;
    let compareHtml;
    if (prev && prev.length > 0) {
      const prevTotal = prev.reduce((a, b) => a + Number((b.values || {})[primaryKey] || 0), 0);
      const delta = total - prevTotal;
      const pct = prevTotal > 0 ? ((delta / prevTotal) * 100).toFixed(1) : '—';
      compareHtml = '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink)">'
        + 'this period: <b>' + fmtNum(total) + '</b> · prev period: <b>' + fmtNum(prevTotal) + '</b>'
        + ' · Δ <b>' + (delta > 0 ? '+' : '') + fmtNum(delta) + (pct !== '—' ? ' (' + pct + '%)' : '') + '</b>'
        + '</div>';
    } else if (filterState.compare) {
      compareHtml = '<div class="lb-empty">no previous-period data available</div>';
    } else {
      compareHtml = '<div class="lb-empty">enable <i>compare</i> in the toolbar to see prev-period delta</div>';
    }
    _appendSection(panel, 'Compare', compareHtml);

    // Section 5: Actions
    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export window CSV' },
      { kind: 'share-link', icon: 'i-link', label: 'share link' },
    ]));
    _wireActions(panel);
  }

  async function _lbDonut(panel) {
    const qs = new URLSearchParams({ range: filterState.range });
    if (filterState.key_id)   qs.set('key_id',   filterState.key_id);
    if (filterState.env)      qs.set('env',      filterState.env);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    if (filterState.status)   qs.set('status',   filterState.status);
    const { body } = await api('/api/account/usage/summary?' + qs);
    const k = (body && body.ok && body.data.kpis) || {};
    const total = (k.requests && k.requests.current) || 0;
    const errs  = (k.errors   && k.errors.current)   || 0;
    const rl    = (k.rate_limited && k.rate_limited.current) || 0;
    const success = Math.max(0, total - errs - rl);

    // Section 1: Re-rendered donut + legend (matches mockup top)
    _appendSection(panel, 'Status mix · ' + fmtNum(total) + ' requests',
      '<div id="lb-donut-host" style="min-height:260px"></div>'
    );
    setTimeout(() => {
      const dHost = document.getElementById('lb-donut-host');
      if (dHost) {
        APIN.charts.donut(dHost, [
          { label: '2xx', value: success, color_token: 'ok' },
          { label: '4xx', value: errs,    color_token: 'amber' },
          { label: '5xx', value: 0,       color_token: 'danger' },
          { label: '429', value: rl,      color_token: 'warn' },
        ], { totalLabel: 'requests', size: 240 });
      }
    }, 350);

    // Section 2: 4xx breakdown (per-error-code bars from /top)
    const qs2 = new URLSearchParams({ range: filterState.range, dim: 'error_codes', limit: 20 });
    if (filterState.key_id)   qs2.set('key_id',   filterState.key_id);
    if (filterState.env)      qs2.set('env',      filterState.env);
    if (filterState.status)   qs2.set('status',   filterState.status);
    if (filterState.endpoint) qs2.set('endpoint', filterState.endpoint);
    const tres = await api('/api/account/usage/top?' + qs2);
    const items = (tres.body && tres.body.ok && tres.body.data.items) || [];
    const maxItem = Math.max(1, ...items.map(i => Number(i.count) || 0));
    const breakdownHtml = items.length
      ? '<div style="display:grid;grid-template-columns:1fr 120px 70px;gap:12px;align-items:center;font-family:\'JetBrains Mono\',monospace;font-size:12px">'
        + items.map(it => {
            const w = ((Number(it.count) || 0) / maxItem) * 100;
            return '<div>' + escHtml(it.label) + '</div>'
              + '<div style="height:14px;background:var(--paper-deep);position:relative;border:1px solid var(--paper-edge)">'
              +   '<div style="position:absolute;left:0;top:0;height:100%;width:' + w + '%;background:var(--c-amber);opacity:0.78"></div>'
              + '</div>'
              + '<div style="text-align:right;font-variant-numeric:tabular-nums">' + fmtNum(it.count) + ' (' + (Number(it.pct)||0).toFixed(1) + '%)</div>';
          }).join('')
        + '</div>'
      : '<div class="lb-empty">no error codes in this window</div>';
    _appendSection(panel, '4xx / 5xx breakdown', breakdownHtml);

    // Section 3: Endpoints producing errors
    const errEps = (_perEndpointCache || []).filter(it => it.error_rate > 0)
                    .sort((a,b) => b.errors - a.errors).slice(0, 8);
    const maxErr = Math.max(1, ...errEps.map(e => e.errors));
    const epHtml = errEps.length
      ? '<div style="display:grid;grid-template-columns:1fr 130px 70px;gap:12px;align-items:center;font-family:\'JetBrains Mono\',monospace;font-size:12px">'
        + errEps.map(e => {
            const w = (e.errors / maxErr) * 100;
            return '<div>' + escHtml(e.label) + '</div>'
              + '<div style="height:14px;background:var(--paper-deep);position:relative;border:1px solid var(--paper-edge)">'
              +   '<div style="position:absolute;left:0;top:0;height:100%;width:' + w + '%;background:var(--c-danger);opacity:0.72"></div>'
              + '</div>'
              + '<div style="text-align:right;font-variant-numeric:tabular-nums">' + fmtNum(e.errors) + ' × err</div>';
          }).join('')
        + '</div>'
      : '<div class="lb-empty">no endpoints producing errors</div>';
    _appendSection(panel, 'Endpoints producing errors', epHtml);

    // Section 4: When errors hit — mini-timeline from errors timeseries
    const qs3 = new URLSearchParams({ range: filterState.range, mode: 'errors' });
    if (filterState.key_id) qs3.set('key_id', filterState.key_id);
    if (filterState.env) qs3.set('env', filterState.env);
    if (filterState.endpoint) qs3.set('endpoint', filterState.endpoint);
    const tlRes = await api('/api/account/usage/timeseries?' + qs3);
    const tlBuckets = (tlRes.body && tlRes.body.ok && tlRes.body.data.buckets) || [];
    const errSum = tlBuckets.map(b => Number((b.values || {})['4xx'] || 0) + Number((b.values || {})['5xx'] || 0) + Number((b.values || {})['429'] || 0));
    const maxErrTl = Math.max(1, ...errSum);
    const tlHtml = tlBuckets.length
      ? '<div style="display:flex;align-items:flex-end;gap:6px;padding:4px 0;font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--ink-soft);overflow-x:auto">'
        + tlBuckets.map((b, i) => {
            const v = errSum[i];
            const filled = v > 0;
            const sz = filled ? Math.max(8, 8 + (v / maxErrTl) * 12) : 6;
            return '<div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex-shrink:0">'
              + '<div style="width:' + sz + 'px;height:' + sz + 'px;border-radius:50%;background:' + (filled ? 'var(--c-danger,#b01820)' : 'transparent') + ';border:1px solid ' + (filled ? 'var(--c-danger,#b01820)' : 'var(--paper-edge,#c7bca9)') + '" title="' + escHtml(b.t) + ' · ' + v + ' errors"></div>'
              + '<div style="font-size:9px">' + escHtml((b.t || '').slice(11, 16)) + '</div>'
              + '</div>';
          }).join('')
        + '</div>'
      : '<div class="lb-empty">no buckets to plot</div>';
    _appendSection(panel, 'When errors hit (mini-timeline)', tlHtml);

    // Section 5: Actions
    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'filter-status', value: '4xx', icon: 'i-filter', label: 'drill into all 4xx' },
      { kind: 'filter-status', value: '5xx', icon: 'i-filter', label: 'drill into all 5xx' },
      { kind: 'export-csv', icon: 'i-download', label: 'export filtered CSV' },
    ]));
    _wireActions(panel);
  }

  async function _lbHistogram(panel) {
    const qs = new URLSearchParams({ range: filterState.range, limit: 200 });
    if (filterState.key_id)   qs.set('key_id',   filterState.key_id);
    if (filterState.env)      qs.set('env',      filterState.env);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    if (filterState.status)   qs.set('status',   filterState.status);
    const { body } = await api('/api/account/usage/requests?' + qs);
    const items = (body && body.ok && body.data.items) || [];
    const lats = items.map(i => Number(i.latency_ms) || 0).filter(x => x > 0).sort((a,b) => a - b);
    const pct = p => lats.length ? lats[Math.floor(p * (lats.length - 1))] : 0;
    const p50 = pct(0.5), p90 = pct(0.9), p95 = pct(0.95), p99 = pct(0.99);
    const max = lats[lats.length - 1] || 0;

    // Section 1: Headline + re-rendered histogram larger
    _appendSection(panel, 'Latency distribution · ' + lats.length + ' samples',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   'p50 ' + fmtNum(p50) + 'ms · p90 ' + fmtNum(p90) + 'ms · p95 ' + fmtNum(p95) + 'ms · p99 ' + fmtNum(p99) + 'ms · max ' + fmtNum(max) + 'ms'
      + '</div>'
      + '<div id="lb-hist-host" style="min-height:240px"></div>');
    // Build histogram buckets from raw latencies
    setTimeout(() => {
      const hHost = document.getElementById('lb-hist-host');
      if (!hHost) return;
      const bounds = [50, 100, 200, 500, 1000, 2000, 5000, Infinity];
      const labels = ['<50','50-100','100-200','200-500','500-1k','1-2k','2-5k','5k+'];
      const tokens = ['ok','ok','info','amber','amber','warn','danger','danger'];
      const buckets = bounds.map((_, i) => ({ label: labels[i], value: 0, color_token: tokens[i] }));
      lats.forEach(v => {
        for (let i = 0; i < bounds.length; i++) {
          if (v < bounds[i]) { buckets[i].value++; break; }
        }
      });
      APIN.charts.histogram(hHost, buckets, {
        height: 240,
        percentiles: [
          { label: 'p50', value: p50, color_token: 'amber' },
          { label: 'p95', value: p95, color_token: 'danger' },
        ],
        bucketBoundaries: bounds,
      });
    }, 350);

    // Section 2: Per-endpoint latency (box plot list)
    const epLats = (_perEndpointCache || []).slice(0, 8);
    const maxP95 = Math.max(1, ...epLats.map(e => e.p95 || 0));
    const epHtml = epLats.length
      ? '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11.5px;display:grid;grid-template-columns:160px 1fr 80px;gap:14px;align-items:center">'
        + epLats.map(e => {
            const xP10 = ((e.p10 || 0) / maxP95) * 100;
            const xP25 = ((e.p25 || 0) / maxP95) * 100;
            const xP50 = ((e.p50 || 0) / maxP95) * 100;
            const xP75 = ((e.p75 || 0) / maxP95) * 100;
            const xP95 = ((e.p95 || 0) / maxP95) * 100;
            return '<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escHtml(e.label) + '</div>'
              + '<div style="position:relative;height:18px;background:var(--paper-deep);border:1px solid var(--paper-edge)">'
              // whisker
              +   '<div style="position:absolute;left:' + xP10 + '%;width:' + (xP95 - xP10) + '%;top:50%;height:1.5px;background:var(--ink);transform:translateY(-50%)"></div>'
              // box p25-p75
              +   '<div style="position:absolute;left:' + xP25 + '%;width:' + Math.max(2, xP75 - xP25) + '%;top:3px;height:12px;background:var(--ink);opacity:0.22;border:1px solid var(--ink)"></div>'
              // median p50
              +   '<div style="position:absolute;left:' + xP50 + '%;width:2.4px;top:1px;height:16px;background:var(--ink)"></div>'
              + '</div>'
              + '<div style="text-align:right;color:var(--ink-soft);font-style:italic">p50 ' + fmtNum(e.p50) + 'ms</div>';
          }).join('')
        + '</div>'
      : '<div class="lb-empty">no per-endpoint latency data</div>';
    _appendSection(panel, 'Per-endpoint latency (box plot)', epHtml);

    // Section 3: Slowest 10 requests
    const slowest = [...items].sort((a, b) => (Number(b.latency_ms) || 0) - (Number(a.latency_ms) || 0)).slice(0, 10);
    _appendSection(panel, 'Slowest 10 requests',
      _table([{label: '#'}, {label: 'Method'}, {label: 'Path'}, {label: 'Latency', numeric: true}, {label: 'Key'}, {label: 'UA'}],
        slowest.map(s => ['#' + s.id, s.method, s.path, fmtNum(s.latency_ms) + 'ms', s.key_name || s.key_public_id || '?', (s.ua || '').split('/')[0] || '?']))
    );

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export histogram CSV' },
    ]));
    _wireActions(panel);
  }

  function _lbTopEndpoints(panel) {
    const items = _perEndpointCache || [];
    const total = items.reduce((a, it) => a + (Number(it.count) || 0), 0);
    const maxC = Math.max(1, ...items.map(it => Number(it.count) || 0));
    // Section 1: per-endpoint cards — matches ASCII mockup. Top 4 get the
    // "full card" treatment; remaining endpoints get a compact row.
    const topN = items.slice(0, 4);
    const restN = items.slice(4);
    function bar(w, color) {
      return '<div style="height:14px;background:var(--paper-deep);border:1px solid var(--paper-edge);position:relative;display:inline-block;width:200px;vertical-align:middle">'
        + '<div style="position:absolute;left:0;top:0;height:100%;width:' + w + '%;background:' + color + ';opacity:0.78"></div>'
        + '</div>';
    }
    function shapeSvg(spark) {
      if (!spark || spark.length < 2) return '<span style="color:var(--ink-soft);font-style:italic">flat</span>';
      const mx = Math.max(1, ...spark);
      const w = 200, h = 18;
      const step = (w - 4) / (spark.length - 1);
      let d = 'M2,' + (h - 2 - (spark[0] / mx) * (h - 4));
      for (let i = 1; i < spark.length; i++) d += ' L' + (2 + i * step) + ',' + (h - 2 - (spark[i] / mx) * (h - 4));
      return '<svg viewBox="0 0 ' + w + ' ' + h + '" width="' + w + '" height="' + h + '" preserveAspectRatio="none" style="vertical-align:middle">'
        + '<path d="' + d + '" fill="none" stroke="var(--ink-soft, #6b6453)" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round" opacity="0.78"/>'
        + '</svg>';
    }
    const cardsHtml = topN.length
      ? topN.map((e, idx) => {
          const pctW = ((Number(e.count) || 0) / maxC) * 100;
          const color = e.error_rate > 0.05 ? 'var(--c-danger)' : e.error_rate > 0 ? 'var(--c-amber)' : 'var(--c-ok)';
          return '<div style="border:1px solid var(--paper-edge);padding:14px 16px;margin-bottom:12px;background:var(--paper);'
            + 'cursor:pointer" class="lb-ep-card" data-endpoint="' + escHtml(e.label) + '">'
            + '<div style="display:flex;align-items:center;gap:14px;margin-bottom:8px">'
            +   '<div style="flex:1;font-family:\'JetBrains Mono\',monospace;font-size:13.5px;font-weight:500;color:var(--ink)">' + escHtml(e.label) + '</div>'
            +   bar(pctW, color)
            +   '<div style="font-family:\'JetBrains Mono\',monospace;font-variant-numeric:tabular-nums;font-size:13px;color:var(--ink);min-width:80px;text-align:right">' + fmtNum(e.count) + ' (' + Number(e.pct).toFixed(1) + '%)</div>'
            + '</div>'
            + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-family:\'JetBrains Mono\',monospace;font-size:11.5px;color:var(--ink-soft)">'
            +   '<div>├─ p50 ' + fmtNum(e.p50) + 'ms · p99 ' + fmtNum(e.p99) + 'ms · ' + (e.errors === 0 ? '0 errors' : fmtNum(e.errors) + ' errors') + '</div>'
            +   '<div>├─ ' + fmtBytes(e.bytes_in) + ' in · ' + fmtBytes(e.bytes_out) + ' out</div>'
            +   '<div style="grid-column:1/-1;display:flex;align-items:center;gap:8px">└─ traffic shape ' + shapeSvg(e.sparkline) + '</div>'
            + '</div>'
            + '</div>';
        }).join('')
      : '<div class="lb-empty">no endpoint activity</div>';
    _appendSection(panel, 'Top ' + Math.min(4, topN.length) + ' endpoints', cardsHtml);

    // Section 2: remaining endpoints compact list
    if (restN.length > 0) {
      const compact = '<div style="display:grid;grid-template-columns:1fr 100px 60px;gap:12px;font-family:\'JetBrains Mono\',monospace;font-size:12.5px">'
        + restN.map(e => '<div class="lb-ep-row" data-endpoint="' + escHtml(e.label) + '" style="display:contents;cursor:pointer">'
            + '<div style="padding:7px 8px;border-bottom:1px solid var(--paper-edge)">' + escHtml(e.label) + '</div>'
            + '<div style="padding:7px 8px;border-bottom:1px solid var(--paper-edge);text-align:right;font-variant-numeric:tabular-nums">' + fmtNum(e.count) + '</div>'
            + '<div style="padding:7px 8px;border-bottom:1px solid var(--paper-edge);text-align:right;color:var(--ink-soft)">' + Number(e.pct).toFixed(1) + '%</div>'
            + '</div>').join('')
        + '</div>';
      _appendSection(panel, 'More endpoints (' + restN.length + ')', compact);
    }

    // Section 3: Sort affordance (text-only — sort already implicit by volume)
    _appendSection(panel, 'Sort',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11.5px;color:var(--ink-soft);font-style:italic">'
      + 'currently sorted by [volume▾]   ·   future toggles: [errors] [p95] [bytes-out] [name]'
      + '</div>'
    );

    _appendSection(panel, 'Actions', _actionRow([
      ...(items[0] ? [{ kind: 'filter-endpoint', value: items[0].label, icon: 'i-filter', label: 'filter to ' + items[0].label }] : []),
      { kind: 'export-csv', icon: 'i-download', label: 'export endpoints CSV' },
    ]));

    // Wire card clicks
    panel.querySelectorAll('.lb-ep-card, .lb-ep-row').forEach(c => {
      c.addEventListener('click', () => {
        const ep = c.getAttribute('data-endpoint');
        if (ep) {
          filterState.endpoint = ep;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        }
      });
    });
    _wireActions(panel);
  }

  // ─── 9.N.6.f3 · Rich expanded states for top-N panels ──────────────────
  // Each top-N gets: headline + re-rendered topBar chart + narrative
  // observations + per-row deep-drill table + actions. The topBar
  // re-render is interactive (hover tooltips show same data as the source).

  async function _lbTopKeys(panel) {
    const qs = new URLSearchParams({ range: filterState.range, dim: 'keys', limit: 30 });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env)    qs.set('env',    filterState.env);
    if (filterState.status) qs.set('status', filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    const { body } = await api('/api/account/usage/top?' + qs);
    const items = (body && body.ok && body.data.items) || [];
    const total = items.reduce((a, it) => a + (Number(it.count) || 0), 0);
    const totalKeys = availableKeys.length;
    const dormant = Math.max(0, totalKeys - items.length);

    // Section 1: Headline + re-rendered topBar with full list
    _appendSection(panel, 'Top keys · ' + fmtNum(total) + ' requests',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   items.length + ' of ' + totalKeys + ' keys active · '
      +   dormant + ' dormant'
      +   (items[0] ? ' · top key "' + escHtml(items[0].label) + '" carries ' + Number(items[0].pct).toFixed(1) + '%' : '')
      + '</div>'
      + '<div id="lb-tkeys-host" style="padding:4px 0"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-tkeys-host');
      if (h) APIN.charts.topBar(h, items, {
        color_token: 'series-0',
        onClickItem: it => {
          filterState.key_id = it.public_id || '';
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        }
      });
    }, 350);

    // Section 2: Insights
    const bullets = [];
    if (items[0] && items[0].pct >= 80) bullets.push('Single-key dependency: ' + items[0].label + ' carries ' + Number(items[0].pct).toFixed(1) + '% — consider sharding traffic');
    if (dormant >= 3) bullets.push(dormant + ' dormant keys — review or rotate keys with no recent traffic');
    const errKey = items.find(it => it.extra && it.extra.errors > 0);
    if (errKey) bullets.push(errKey.label + ' produced ' + fmtNum(errKey.extra.errors) + ' 5xx errors in this window');
    if (items.length === 1) bullets.push('Only 1 key sending traffic — no diversity to compare against');
    _appendSection(panel, 'Insights', bullets.length
      ? '<ul style="margin:0;padding-left:18px;font-family:Fraunces,serif;font-size:13.5px;color:var(--ink);line-height:1.85">'
        + bullets.map(b => '<li>' + escHtml(b) + '</li>').join('') + '</ul>'
      : '<div class="lb-empty">no notable key-level patterns</div>');

    // Section 3: Per-key deep drill
    _appendSection(panel, 'Per-key breakdown',
      _table(
        [{label: 'Key'}, {label: 'Env'}, {label: 'Count', numeric: true},
         {label: '%', numeric: true}, {label: '5xx errs', numeric: true},
         {label: 'Avg lat', numeric: true}],
        items.map(it => [it.label, it.env || 'live', fmtNum(it.count),
                         Number(it.pct).toFixed(1) + '%',
                         fmtNum((it.extra && it.extra.errors) || 0),
                         fmtNum((it.extra && it.extra.avg_latency_ms) || 0) + 'ms']))
    );

    _appendSection(panel, 'Actions', _actionRow([
      ...(items[0] ? [{ kind: 'filter-key', value: items[0].public_id || items[0].label, icon: 'i-filter', label: 'filter to ' + items[0].label }] : []),
      { kind: 'export-csv', icon: 'i-download', label: 'export key-scoped CSV' },
    ]));
    _wireActions(panel);
  }

  async function _lbTopIPs(panel) {
    const qs = new URLSearchParams({ range: filterState.range, dim: 'ips', limit: 30 });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env)    qs.set('env',    filterState.env);
    if (filterState.status) qs.set('status', filterState.status);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    const { body } = await api('/api/account/usage/top?' + qs);
    const items = (body && body.ok && body.data.items) || [];
    const total = items.reduce((a, it) => a + (Number(it.count) || 0), 0);
    const unique = items.length;
    const heavyHitter = items[0] && items[0].pct >= 70;

    _appendSection(panel, 'Top IPs · ' + fmtNum(total) + ' requests',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   unique + ' unique IP' + (unique !== 1 ? 's' : '') + ' · '
      +   (heavyHitter ? 'top IP carries ' + Number(items[0].pct).toFixed(1) + '% — single-host dominance' : 'distributed across ' + unique + ' hosts')
      + '</div>'
      + '<div id="lb-tips-host" style="padding:4px 0"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-tips-host');
      if (h) APIN.charts.topBar(h, items, { color_token: 'series-2' });
    }, 350);

    // Insights — suspicious patterns
    const bullets = [];
    if (heavyHitter) bullets.push('Single IP dominance — could indicate batch jobs from one machine, or a misbehaving client');
    if (unique === 1) bullets.push('Only one IP — likely localhost or a single deployment');
    if (unique >= 20) bullets.push(unique + ' distinct IPs — broad public usage');
    const localhost = items.find(it => it.label && (it.label === '127.0.0.1' || it.label === '::1' || it.label.startsWith('192.168.')));
    if (localhost) bullets.push(localhost.label + ' is a local-network address — likely test traffic, not production');
    _appendSection(panel, 'IP pattern insights', bullets.length
      ? '<ul style="margin:0;padding-left:18px;font-family:Fraunces,serif;font-size:13.5px;color:var(--ink);line-height:1.85">'
        + bullets.map(b => '<li>' + escHtml(b) + '</li>').join('') + '</ul>'
      : '<div class="lb-empty">no notable IP patterns</div>');

    _appendSection(panel, 'Full IP list',
      _table([{label: 'IP address'}, {label: 'Count', numeric: true}, {label: '%', numeric: true}],
        items.map(it => [it.label, fmtNum(it.count), Number(it.pct).toFixed(1) + '%']))
    );

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export with IP column' },
      { kind: 'share-link', icon: 'i-link', label: 'share link' },
    ]));
    _wireActions(panel);
  }

  async function _lbTopErrors(panel) {
    const qs = new URLSearchParams({ range: filterState.range, dim: 'error_codes', limit: 30 });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env)    qs.set('env',    filterState.env);
    if (filterState.endpoint) qs.set('endpoint', filterState.endpoint);
    const { body } = await api('/api/account/usage/top?' + qs);
    const items = (body && body.ok && body.data.items) || [];
    const total = items.reduce((a, it) => a + (Number(it.count) || 0), 0);

    _appendSection(panel, 'Error codes · ' + fmtNum(total) + ' failures',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   items.length + ' distinct error codes'
      +   (items[0] ? ' · most common: ' + escHtml(items[0].label) + ' (' + Number(items[0].pct).toFixed(1) + '%)' : '')
      + '</div>'
      + '<div id="lb-terr-host" style="padding:4px 0"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-terr-host');
      if (h) APIN.charts.topBar(h, items, { color_token: 'series-3' });
    }, 350);

    // Per-error narrative: list each top error with sample endpoints from cache
    const cache = _perEndpointCache || [];
    const errBuckets = items.slice(0, 5).map(it => {
      // Sample 2-3 endpoints that have errors
      const sampleEndpts = cache.filter(e => e.errors > 0).slice(0, 3).map(e => e.label);
      return {
        code: it.label,
        count: it.count,
        pct: it.pct,
        sampleEndpts,
      };
    });
    const narrativeHtml = errBuckets.length
      ? errBuckets.map(e =>
          '<div style="border-bottom:1px solid var(--paper-edge);padding:10px 0">'
          + '<div style="display:flex;justify-content:space-between;align-items:center;font-family:\'JetBrains Mono\',monospace;font-size:12.5px">'
          +   '<span style="color:var(--c-amber)"><b>' + escHtml(e.code) + '</b></span>'
          +   '<span style="color:var(--ink-soft);font-variant-numeric:tabular-nums">' + fmtNum(e.count) + ' (' + Number(e.pct).toFixed(1) + '%)</span>'
          + '</div>'
          + '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11.5px;color:var(--ink-soft);margin-top:4px">'
          +   (e.sampleEndpts.length ? 'seen on: ' + e.sampleEndpts.map(escHtml).join(', ') : 'no endpoint detail available')
          + '</div>'
          + '</div>').join('')
      : '<div class="lb-empty">no error codes recorded</div>';
    _appendSection(panel, 'Per-code drill', narrativeHtml);

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'filter-status', value: '4xx', icon: 'i-filter', label: 'filter to all 4xx' },
      { kind: 'filter-status', value: '5xx', icon: 'i-filter', label: 'filter to all 5xx' },
      { kind: 'export-csv', icon: 'i-download', label: 'export error log' },
    ]));
    _wireActions(panel);
  }

  async function _lbHeatmap(panel) {
    const qs = new URLSearchParams({ mode: _heatmapMode || 'week' });
    if (filterState.key_id) qs.set('key_id', filterState.key_id);
    if (filterState.env)    qs.set('env',    filterState.env);
    const { body } = await api('/api/account/usage/heatmap-calendar?' + qs);
    const d = (body && body.ok && body.data) || { cells: [], rows: 0, cols: 0 };
    const cells = d.cells || [];
    const filled = cells.filter(c => c.count > 0);
    const peak = filled.reduce((m, c) => Math.max(m, c.count), 0);
    const peakCell = filled.find(c => c.count === peak);
    const totalCells = (d.rows || 0) * (d.cols || 0);
    const totalReqs = filled.reduce((a, c) => a + c.count, 0);
    const dutyPct = totalCells > 0 ? ((filled.length / totalCells) * 100).toFixed(0) : '0';
    const modeNames = { week: 'last 7 days, 24h × 7 grid', month: 'last 30 days, daily', year: 'last 12 months', years: 'last 5 years' };

    // Section 1: Headline + re-rendered heatmap interactive
    _appendSection(panel, 'Activity calendar · ' + (modeNames[d.mode] || d.mode || '—'),
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   fmtNum(totalReqs) + ' total requests · '
      +   filled.length + ' of ' + totalCells + ' cells active (' + dutyPct + '%)'
      +   (peakCell ? ' · peak ' + fmtNum(peak) + ' in one cell' : '')
      + '</div>'
      + '<div id="lb-heatmap-host"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-heatmap-host');
      if (h) APIN.charts.activityHeatmap(h, d, {});
    }, 350);

    // Section 2: When you're busy (insights derived from cells)
    const bullets = [];
    if (d.mode === 'week' && filled.length > 0) {
      const hourCounts = Array(24).fill(0);
      const dayCounts = Array(7).fill(0);
      filled.forEach(c => { hourCounts[c.col] += c.count; dayCounts[c.row] += c.count; });
      const busiestHour = hourCounts.indexOf(Math.max(...hourCounts));
      const busiestDay = dayCounts.indexOf(Math.max(...dayCounts));
      const dayNames = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
      bullets.push('Busiest hour of day: ' + String(busiestHour).padStart(2,'0') + ':00 (' + fmtNum(hourCounts[busiestHour]) + ' requests over the week)');
      bullets.push('Busiest day of week: ' + dayNames[busiestDay] + ' (' + fmtNum(dayCounts[busiestDay]) + ' requests)');
      const deadHours = hourCounts.map((c, h) => ({ c, h })).filter(x => x.c === 0).length;
      if (deadHours >= 6) bullets.push(deadHours + ' hours of the day have zero traffic — could indicate timezone-locked usage or scheduled clients');
    }
    if (d.mode === 'month' && filled.length > 0) {
      const weekCounts = Array(d.rows || 5).fill(0);
      filled.forEach(c => { weekCounts[c.row] += c.count; });
      const peakWeek = weekCounts.indexOf(Math.max(...weekCounts));
      bullets.push('Busiest week (' + (peakWeek === 0 ? 'current' : peakWeek + ' weeks ago') + '): ' + fmtNum(weekCounts[peakWeek]) + ' requests');
    }
    if (dutyPct < 30) bullets.push('Only ' + dutyPct + '% of slots are active — sparse traffic pattern, consider a different view mode');
    _appendSection(panel, 'When you are busy', bullets.length
      ? '<ul style="margin:0;padding-left:18px;font-family:Fraunces,serif;font-size:13.5px;color:var(--ink);line-height:1.85">'
        + bullets.map(b => '<li>' + escHtml(b) + '</li>').join('') + '</ul>'
      : '<div class="lb-empty">not enough data to derive patterns yet</div>');

    // Section 3: Top 10 busiest cells
    const top = [...filled].sort((a, b) => b.count - a.count).slice(0, 10);
    _appendSection(panel, 'Top 10 busiest cells',
      _table([{label: 'When'}, {label: 'Label'}, {label: 'Count', numeric: true}, {label: 'Share', numeric: true}],
        top.map(c => {
          const rowLbl = d.row_labels?.[c.row] || 'row ' + c.row;
          const colLbl = c.label || d.col_labels?.[c.col] || 'col ' + c.col;
          const share = totalReqs ? ((c.count / totalReqs) * 100).toFixed(1) + '%' : '—';
          return [rowLbl + ' · ' + colLbl, c.label || '', fmtNum(c.count), share];
        }))
    );

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export traffic by hour' },
      { kind: 'share-link', icon: 'i-link', label: 'share this view' },
    ]));
    _wireActions(panel);
  }

  function _lbTreemap(panel) {
    const items = _perEndpointCache || [];
    const total = items.reduce((a, it) => a + (Number(it.count) || 0), 0);
    const errCount = items.reduce((a, it) => a + (Number(it.errors) || 0), 0);
    const healthy = items.filter(it => it.error_rate === 0).length;
    const withErrs = items.filter(it => it.error_rate > 0).length;

    _appendSection(panel, 'Endpoint treemap · ' + fmtNum(total) + ' requests',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   items.length + ' endpoints · '
      +   healthy + ' healthy · '
      +   (withErrs ? withErrs + ' with errors (' + fmtNum(errCount) + ' total)' : 'all clean')
      + '</div>'
      + '<div id="lb-treemap-host" style="height:360px"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-treemap-host');
      if (h) APIN.charts.treemap(h, items.map(it => ({
        label: it.label, value: it.count, error_rate: it.error_rate
      })), {
        height: 360,
        onClickTile: t => {
          filterState.endpoint = t.label;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        }
      });
    }, 350);

    // Insights
    const bullets = [];
    if (items[0] && items[0].pct >= 40) bullets.push(items[0].label + ' dominates with ' + Number(items[0].pct).toFixed(1) + '% — heavy-tile concentration');
    if (withErrs > 0) {
      const worst = [...items].filter(it => it.error_rate > 0).sort((a,b) => b.error_rate - a.error_rate)[0];
      bullets.push('Highest error rate: ' + worst.label + ' at ' + (worst.error_rate * 100).toFixed(1) + '%');
    }
    if (items.length >= 6) {
      const top3pct = items.slice(0, 3).reduce((a, it) => a + (Number(it.pct) || 0), 0);
      if (top3pct >= 75) bullets.push('Top 3 endpoints carry ' + top3pct.toFixed(0) + '% — clear Pareto distribution');
    }
    _appendSection(panel, 'Composition insights', bullets.length
      ? '<ul style="margin:0;padding-left:18px;font-family:Fraunces,serif;font-size:13.5px;color:var(--ink);line-height:1.85">'
        + bullets.map(b => '<li>' + escHtml(b) + '</li>').join('') + '</ul>'
      : '<div class="lb-empty">no notable composition patterns</div>');

    _appendSection(panel, 'All tiles ranked',
      _table([{label: 'Endpoint'}, {label: 'Count', numeric: true},
              {label: '%', numeric: true}, {label: 'Err rate', numeric: true},
              {label: 'p95', numeric: true}],
        items.map(e => [e.label, fmtNum(e.count), Number(e.pct).toFixed(1) + '%',
                        (e.error_rate * 100).toFixed(1) + '%', fmtNum(e.p95) + 'ms']))
    );

    _appendSection(panel, 'Actions', _actionRow([
      ...(items[0] ? [{ kind: 'filter-endpoint', value: items[0].label, icon: 'i-filter', label: 'filter to biggest tile' }] : []),
      { kind: 'export-csv', icon: 'i-download', label: 'export CSV' },
    ]));
    _wireActions(panel);
  }

  function _lbSparkGrid(panel) {
    const items = _perEndpointCache || [];
    const total = items.reduce((a, it) => a + (Number(it.count) || 0), 0);
    // Classify shape: rising / falling / spiky / steady (very rough heuristic)
    function shapeOf(s) {
      if (!s || s.length < 4) return 'flat';
      const first = s.slice(0, Math.floor(s.length / 2)).reduce((a, v) => a + v, 0);
      const last  = s.slice(Math.floor(s.length / 2)).reduce((a, v) => a + v, 0);
      const max = Math.max(...s), avg = s.reduce((a, v) => a + v, 0) / s.length;
      if (max > avg * 4) return 'spiky';
      if (last > first * 1.5) return 'rising';
      if (first > last * 1.5) return 'falling';
      return 'steady';
    }
    const shapes = items.map(it => ({ ...it, shape: shapeOf(it.sparkline) }));
    const rising = shapes.filter(s => s.shape === 'rising').length;
    const spiky  = shapes.filter(s => s.shape === 'spiky').length;
    const falling = shapes.filter(s => s.shape === 'falling').length;

    _appendSection(panel, 'Per-endpoint shapes · ' + fmtNum(total) + ' requests',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   items.length + ' endpoints · '
      +   rising + ' rising · ' + falling + ' falling · ' + spiky + ' spiky'
      + '</div>'
      + '<div id="lb-sparkgrid-host"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-sparkgrid-host');
      if (h) APIN.charts.sparkGrid(h, items.map(it => ({
        label: it.label, count: it.count, pct: it.pct,
        sparkline: it.sparkline || [], error_rate: it.error_rate
      })), {
        onClickItem: t => {
          filterState.endpoint = t.label;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        }
      });
    }, 350);

    // Shape pattern analysis
    const bullets = [];
    if (spiky > 0) {
      const spikyEps = shapes.filter(s => s.shape === 'spiky').slice(0, 3);
      bullets.push('Spiky shapes (' + spiky + '): ' + spikyEps.map(e => e.label).join(', ') + ' — bursty traffic, investigate cron-like patterns');
    }
    if (rising > 0) {
      const r = shapes.filter(s => s.shape === 'rising').slice(0, 2);
      bullets.push('Rising trend (' + rising + '): ' + r.map(e => e.label).join(', ') + ' — usage growing in this window');
    }
    if (falling > 0) {
      const f = shapes.filter(s => s.shape === 'falling').slice(0, 2);
      bullets.push('Falling trend (' + falling + '): ' + f.map(e => e.label).join(', ') + ' — could be deprecation or backlog clearing');
    }
    const flatHighVol = shapes.filter(s => s.shape === 'steady' && s.count >= 10).length;
    if (flatHighVol >= 3) bullets.push(flatHighVol + ' endpoints have steady high-volume shape — healthy production traffic');
    _appendSection(panel, 'Shape pattern analysis', bullets.length
      ? '<ul style="margin:0;padding-left:18px;font-family:Fraunces,serif;font-size:13.5px;color:var(--ink);line-height:1.85">'
        + bullets.map(b => '<li>' + escHtml(b) + '</li>').join('') + '</ul>'
      : '<div class="lb-empty">no notable shape patterns</div>');

    _appendSection(panel, 'Per-endpoint detail',
      _table([{label: 'Endpoint'}, {label: 'Shape'}, {label: 'Count', numeric: true},
              {label: 'p50', numeric: true}, {label: 'p95', numeric: true},
              {label: 'Bytes out', numeric: true}],
        shapes.map(e => [e.label, e.shape, fmtNum(e.count),
                         fmtNum(e.p50) + 'ms', fmtNum(e.p95) + 'ms', fmtBytes(e.bytes_out)]))
    );

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export per-endpoint stats' },
    ]));
    _wireActions(panel);
  }

  function _lbQuadrant(panel) {
    const items = _perEndpointCache || [];
    // Classify each endpoint into quadrants
    const sortedByVol = [...items].sort((a, b) => (a.count || 0) - (b.count || 0));
    const medianVol = sortedByVol.length ? sortedByVol[Math.floor(sortedByVol.length / 2)].count : 0;
    const ERR_THRESHOLD = 0.05; // 5%
    const quads = {
      attention:   items.filter(it => it.count >  medianVol && it.error_rate >  ERR_THRESHOLD),
      noisy:       items.filter(it => it.count <= medianVol && it.error_rate >  ERR_THRESHOLD),
      healthy:     items.filter(it => it.count >  medianVol && it.error_rate <= ERR_THRESHOLD),
      background:  items.filter(it => it.count <= medianVol && it.error_rate <= ERR_THRESHOLD),
    };

    _appendSection(panel, 'Endpoint health quadrant',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   items.length + ' endpoints plotted · '
      +   '<span style="color:var(--c-danger)">' + quads.attention.length + ' need attention</span> · '
      +   '<span style="color:var(--c-amber)">' + quads.noisy.length + ' noisy</span> · '
      +   '<span style="color:var(--c-ok)">' + quads.healthy.length + ' healthy</span> · '
      +   quads.background.length + ' background'
      + '</div>'
      + '<div id="lb-quad-host" style="height:400px"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-quad-host');
      if (h) APIN.charts.quadrant(h, items.map(it => ({
        label: it.label, x_val: it.count, y_val: it.error_rate, size: it.p95,
        color_token: it.error_rate > 0.05 ? 'danger' : it.error_rate > 0 ? 'amber' : 'ok',
      })), {
        xLabel: 'volume', yLabel: 'error rate', yPct: true,
        xMid: 'median', yMid: 0.05,
        onClickPoint: t => {
          filterState.endpoint = t.label;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        }
      });
    }, 350);

    // Per-quadrant playbook
    function quadrantCard(title, color, list, advice) {
      return '<div style="border:1px solid var(--paper-edge);padding:12px 14px;margin-bottom:10px">'
        + '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:' + color + ';margin-bottom:6px">' + escHtml(title) + ' (' + list.length + ')</div>'
        + (list.length
            ? '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11.5px;color:var(--ink);line-height:1.6">'
              + list.slice(0, 5).map(e => escHtml(e.label) + ' · ' + fmtNum(e.count) + ' reqs · ' + (e.error_rate * 100).toFixed(1) + '% err').join('<br>')
              + (list.length > 5 ? '<br><span style="color:var(--ink-soft);font-style:italic">+ ' + (list.length - 5) + ' more</span>' : '')
              + '</div>'
            : '<div class="lb-empty" style="padding:6px;text-align:left">none</div>')
        + '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11.5px;color:var(--ink-soft);margin-top:8px">' + escHtml(advice) + '</div>'
        + '</div>';
    }
    _appendSection(panel, 'Quadrant playbook',
      quadrantCard('NEEDS ATTENTION (high vol + high err)', 'var(--c-danger)', quads.attention, 'These are your biggest risks. Investigate root cause and roll a fix soon.')
      + quadrantCard('NOISY BUT CHEAP (low vol + high err)', 'var(--c-amber)', quads.noisy, 'Low-traffic endpoints that often fail. Could be deprecated routes or bot probing.')
      + quadrantCard('HEALTHY HEAVY-HITTERS (high vol + low err)', 'var(--c-ok)', quads.healthy, 'Your bread and butter. Monitor p95 latency to keep them healthy.')
      + quadrantCard('BACKGROUND (low vol + low err)', 'var(--ink-soft)', quads.background, 'Quiet, healthy traffic. No action needed.')
    );

    _appendSection(panel, 'Actions', _actionRow([
      ...(quads.attention[0] ? [{ kind: 'filter-endpoint', value: quads.attention[0].label, icon: 'i-filter', label: 'drill into worst offender' }] : []),
      { kind: 'export-csv', icon: 'i-download', label: 'export quadrant CSV' },
    ]));
    _wireActions(panel);
  }

  function _lbBoxplot(panel) {
    const items = _perEndpointCache || [];
    const slowestP95 = [...items].sort((a, b) => (b.p95 || 0) - (a.p95 || 0))[0];
    const totalOutliers = items.reduce((a, it) => a + (it.outliers || []).length, 0);
    const wideSpread = items.filter(it => (it.p95 - it.p10) > 1000).length;

    _appendSection(panel, 'Latency per endpoint',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   items.length + ' endpoints · '
      +   (slowestP95 ? 'highest p95: ' + escHtml(slowestP95.label) + ' (' + fmtNum(slowestP95.p95) + 'ms)' : '—')
      +   ' · ' + totalOutliers + ' outliers across all endpoints'
      + '</div>'
      + '<div id="lb-box-host"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-box-host');
      if (h) APIN.charts.boxplot(h, items.slice(0, 12).map(e => ({
        label: e.label, p10: e.p10, p25: e.p25, p50: e.p50,
        p75: e.p75, p90: e.p90, max: e.max, outliers: e.outliers || []
      })), {
        xUnit: 'ms',
        onClickRow: t => {
          filterState.endpoint = t.label;
          applyStateToUI(); applyFilters();
          APIN.lightbox.close();
        }
      });
    }, 350);

    // Latency anomaly detection insights
    const bullets = [];
    if (slowestP95 && slowestP95.p95 > 2000) {
      bullets.push(slowestP95.label + ' p95 ' + fmtNum(slowestP95.p95) + 'ms exceeds the 2s SLO — investigate the slowest 10 requests');
    }
    if (wideSpread > 0) {
      const widest = [...items].sort((a,b) => (b.p95 - b.p10) - (a.p95 - a.p10))[0];
      bullets.push('Widest spread: ' + widest.label + ' (p10 ' + fmtNum(widest.p10) + 'ms → p95 ' + fmtNum(widest.p95) + 'ms) — inconsistent latency, possibly cold-start');
    }
    if (totalOutliers >= 5) {
      bullets.push(totalOutliers + ' outliers detected (requests > 1.5× p95). These usually indicate retries, GC pauses, or upstream dependency stalls.');
    }
    const consistent = items.filter(it => (it.p95 - it.p50) < it.p50 * 0.5).length;
    if (consistent >= 3) {
      bullets.push(consistent + ' endpoints have tight latency distribution (p95 within 50% of p50) — predictable performance');
    }
    _appendSection(panel, 'Latency anomaly detection', bullets.length
      ? '<ul style="margin:0;padding-left:18px;font-family:Fraunces,serif;font-size:13.5px;color:var(--ink);line-height:1.85">'
        + bullets.map(b => '<li>' + escHtml(b) + '</li>').join('') + '</ul>'
      : '<div class="lb-empty">no anomalies in this window</div>');

    _appendSection(panel, 'Full percentile table',
      _table([{label: 'Endpoint'}, {label: 'p10', numeric: true},
              {label: 'p50', numeric: true}, {label: 'p95', numeric: true},
              {label: 'p99', numeric: true}, {label: 'max', numeric: true},
              {label: 'outliers', numeric: true}],
        items.map(e => [e.label, fmtNum(e.p10) + 'ms', fmtNum(e.p50) + 'ms',
                        fmtNum(e.p95) + 'ms', fmtNum(e.p99) + 'ms',
                        fmtNum(e.max) + 'ms',
                        String((e.outliers || []).length)]))
    );

    _appendSection(panel, 'Actions', _actionRow([
      ...(slowestP95 ? [{ kind: 'filter-endpoint', value: slowestP95.label, icon: 'i-filter', label: 'filter to slowest endpoint' }] : []),
      { kind: 'export-csv', icon: 'i-download', label: 'export latency CSV' },
    ]));
    _wireActions(panel);
  }

  // ─── 9.N.7 · Live stream + live pulse attach + expanded lightboxes ───
  let _liveStreamCtrl = null;
  let _liveStreamExpandedCtrl = null;
  let _livePulseCtrl = null;
  let _livePulseExpandedCtrl = null;

  // ─── 9.N.7.f · Time scrubber on the requests chart ──────────────────
  // Controls a "playhead" that clips the time-series line via SVG path
  // stroke-dasharray. Drag the handle → instantly clip. Click play →
  // animate playhead from current position to right edge over a duration
  // scaled by speed multiplier. Click "live" → jump to right edge.
  function _bindTimeScrubber() {
    const scrubber = document.getElementById('ts-scrubber');
    if (!scrubber) return;
    const handle = scrubber.querySelector('.ts-scrub-handle');
    const fill   = scrubber.querySelector('.ts-scrub-fill');
    const rail   = scrubber.querySelector('.ts-scrub-rail');
    const timeLabel = scrubber.querySelector('.ts-scrub-time');
    const playBtn = scrubber.querySelector('[data-act="play"]');
    const startBtn = scrubber.querySelector('[data-act="start"]');
    const endBtn  = scrubber.querySelector('[data-act="end"]');
    const speedSel = scrubber.querySelector('.ts-scrub-speed');

    // State: position is 0..1 (0 = start of window, 1 = now / live)
    let position = 1;
    let playing = false;
    let rafId = null;

    // Cache the path-length lookups — getTotalLength() forces SVG geometry
    // computation which can be slow. Cache after each chart re-render.
    let _cachedLines = [];   // [{el, len}, ...]

    function _refreshLineCache() {
      _cachedLines = [];
      // Note: we deliberately EXCLUDE .chart-line-prev (comparison overlay
      // for previous period — that's a parallel timeline, not "future"
      // events, so shouldn't be hidden when scrubbing).
      const selectors = '#ts-host .chart-line, #ts-host .chart-line-primary, #ts-host .chart-line-halo';
      document.querySelectorAll(selectors).forEach(line => {
        try {
          const len = line.getTotalLength ? line.getTotalLength() : 0;
          if (len > 0) _cachedLines.push({ el: line, len: len });
        } catch (e) {}
      });
    }

    function _applyDash(p, useTransition) {
      // 9.N.7.f · CRITICAL: cancel any in-flight WAAPI animations first.
      //
      // apin_charts.js calls APIN.fx.drawPath(_primary, { duration: 700 })
      // after every chart render. drawPath uses WAAPI with `fill: "forwards"`
      // which OVERRIDES inline styles indefinitely — even after the
      // animation finishes, the final keyframe (offset=0, line fully drawn)
      // stays "locked in" on top of any style.strokeDashoffset we set.
      //
      // The cascade priority is: WAAPI forwards-fill > !important > inline.
      // Cancelling the animation releases the lock and lets our inline
      // style take effect normally.
      //
      // This is THE reason the user saw "the line still showing fully"
      // when scrubbing — the math was right, but a WAAPI animation was
      // silently overriding our style.
      const transitionStr = useTransition ? 'stroke-dashoffset 120ms linear' : 'none';
      for (let i = 0; i < _cachedLines.length; i++) {
        const { el, len } = _cachedLines[i];
        const offset = len * (1 - p);
        // Kill any WAAPI animations holding this path's dash state hostage.
        if (el.getAnimations) {
          try {
            el.getAnimations().forEach(a => { try { a.cancel(); } catch (_) {} });
          } catch (_) {}
        }
        el.style.transition = transitionStr;
        el.style.strokeDasharray = String(len);
        el.style.strokeDashoffset = String(offset);
      }
    }

    function setPosition(p, opts) {
      position = Math.max(0, Math.min(1, p));
      const pct = position * 100;
      fill.style.width = pct + '%';
      handle.style.right = (100 - pct) + '%';
      // Update time label
      if (position >= 0.998) {
        timeLabel.textContent = 'live';
      } else {
        const rangeSec = _resolveRangeSecs(filterState.range);
        const offsetFromEndSec = (1 - position) * rangeSec;
        const t = new Date(Date.now() - offsetFromEndSec * 1000);
        const h = String(t.getHours()).padStart(2, '0');
        const m = String(t.getMinutes()).padStart(2, '0');
        timeLabel.textContent = h + ':' + m;
      }
      _applyDash(position, !!(opts && opts.smooth));
    }

    function _resolveRangeSecs(r) {
      return { '15m':900, '1h':3600, '6h':21600, '24h':86400, '7d':604800, '30d':2592000 }[r] || 86400;
    }

    function stopPlay() {
      if (!playing) return;
      playing = false;
      playBtn.setAttribute('data-playing', 'false');
      const ico = playBtn.querySelector('use');
      if (ico) ico.setAttribute('href', '#i-play');
      cancelAnimationFrame(rafId);
    }

    function startPlay() {
      if (position >= 0.998) position = 0;
      playing = true;
      playBtn.setAttribute('data-playing', 'true');
      const ico = playBtn.querySelector('use');
      if (ico) ico.setAttribute('href', '#i-pause');
      const speed = Number(speedSel.value || 1);
      // Play duration tuned for "video-like" feel: 6s at 1x to traverse
      // the whole window. Each rAF tick computes exact position from
      // wall-clock so playback is steady even if frames drop.
      const durMs = 6000 / speed;
      const startedAt = performance.now();
      const startPos = position;
      function tick(now) {
        if (!playing) return;
        const elapsed = now - startedAt;
        const newPos = Math.min(1, startPos + elapsed / durMs);
        // No CSS transition during play — rAF already gives 60fps. CSS
        // transitions chained per-frame cause "stop-motion" effect because
        // each new transition restarts from current animation state.
        setPosition(newPos, { smooth: false });
        if (newPos >= 1) {
          stopPlay();
        } else {
          rafId = requestAnimationFrame(tick);
        }
      }
      rafId = requestAnimationFrame(tick);
    }

    playBtn.addEventListener('click', () => {
      if (playing) stopPlay();
      else startPlay();
    });
    startBtn.addEventListener('click', () => {
      stopPlay();
      setPosition(0);
    });
    endBtn.addEventListener('click', () => {
      stopPlay();
      setPosition(1);
    });
    speedSel.addEventListener('change', () => {
      if (playing) { stopPlay(); startPlay(); }
    });

    // Drag handling — pointer events (works for mouse + touch + pen)
    let dragging = false;
    function clientXToPos(clientX) {
      const r = rail.getBoundingClientRect();
      return Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    }
    function onPointerDown(e) {
      stopPlay();
      dragging = true;
      if (e.target === handle) e.preventDefault();
      setPosition(clientXToPos(e.clientX), { smooth: false });
      // Capture pointer so drag continues outside the rail
      if (rail.setPointerCapture && e.pointerId != null) {
        try { rail.setPointerCapture(e.pointerId); } catch (_) {}
      }
    }
    function onPointerMove(e) {
      if (!dragging) return;
      // No transition during drag — must feel instant under cursor
      setPosition(clientXToPos(e.clientX), { smooth: false });
    }
    function onPointerUp() { dragging = false; }
    rail.addEventListener('pointerdown', onPointerDown);
    handle.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);

    // Keyboard arrow control when handle is focused
    handle.addEventListener('keydown', e => {
      if (e.key === 'ArrowLeft')  { stopPlay(); setPosition(position - 0.02); e.preventDefault(); }
      if (e.key === 'ArrowRight') { stopPlay(); setPosition(position + 0.02); e.preventDefault(); }
      if (e.key === ' ')          { e.preventDefault(); if (playing) stopPlay(); else startPlay(); }
    });

    // ── Chart re-render coordination ──────────────────────────────────────
    // When the chart re-renders (filter change, auto-refresh, range change),
    // the SVG paths are RECREATED. Our cached refs are stale + the wet-ink
    // draw-in animation re-runs over ~700ms and overrides any dash we set.
    //
    // Strategy: debounce. On any ts-host mutation, wait 800ms (long enough
    // for wet-ink to finish), then refresh the line cache and re-apply the
    // current scrubber position. This way the visual is preserved across
    // re-renders.
    let _reapplyTimer = null;
    function _scheduleReapply() {
      if (_reapplyTimer) clearTimeout(_reapplyTimer);
      _reapplyTimer = setTimeout(() => {
        _reapplyTimer = null;
        _refreshLineCache();
        if (!playing) {
          // If user has scrubbed off "live", restore their position.
          // If at live, leave the chart fully painted (no dash needed).
          if (position < 0.998) _applyDash(position, false);
        }
      }, 800);
    }
    const tsHost = document.getElementById('ts-host');
    if (tsHost) {
      const obs = new MutationObserver(_scheduleReapply);
      obs.observe(tsHost, { childList: true, subtree: true });
      // Initial cache (chart already rendered when boot calls us)
      setTimeout(_refreshLineCache, 100);
    }
  }

  function _attachLivePulse() {
    const host = document.getElementById('livepulse-host');
    if (!host || !window.APIN || !APIN.livePulse) return;
    if (_livePulseCtrl) return;
    _livePulseCtrl = APIN.livePulse.attach(host, {
      compact: true,
      windowSec: 60,
      metric: 'rate',
      onClickPoint: (arg, meta) => {
        // 9.N.7.f · The pulse may pass a single event OR an array of events
        // (when multiple requests landed in the same second). For single,
        // open the detail drawer directly. For multi, show a small chooser
        // popover listing them, each clickable to open detail.
        const events = (meta && meta.allEvents) ? meta.allEvents
                     : Array.isArray(arg) ? arg
                     : [arg];
        if (events.length === 1) {
          _openOneEventDetail(events[0]);
        } else {
          _showLivePulseChooser(events, meta && meta.tInt);
        }
      },
    });
  }

  function _openOneEventDetail(ev) {
    if (ev && ev.id != null && typeof openRequestDetail === 'function') {
      openRequestDetail(ev.id);
    } else if (ev) {
      api('/api/account/usage/requests?range=1h&limit=20').then(r => {
        const items = (r.body && r.body.ok && r.body.data && r.body.data.items) || [];
        const match = items.find(it => it.path === ev.path && it.method === ev.method && it.status_code === ev.status_code);
        if (match && typeof openRequestDetail === 'function') openRequestDetail(match.id);
      }).catch(()=>{});
    }
  }

  // Lightweight popover showing all events at a given second. Each row is
  // clickable → opens the request-detail drawer.
  function _showLivePulseChooser(events, tInt) {
    // Remove any previous chooser
    document.querySelectorAll('.lp-chooser').forEach(el => el.remove());
    const host = document.getElementById('livepulse-host');
    if (!host) return;
    const timeStr = tInt ? new Date(tInt * 1000).toTimeString().slice(0, 8) : 'now';
    const div = document.createElement('div');
    div.className = 'lp-chooser';
    div.innerHTML =
      '<div class="lp-chooser-head">' +
        '<strong>' + events.length + ' requests at ' + timeStr + '</strong>' +
        '<button class="lp-chooser-close" aria-label="close">×</button>' +
      '</div>' +
      '<div class="lp-chooser-list">' +
        events.map((e, i) => {
          const status = (e.status_code || 0);
          const color = status >= 500 ? 'var(--c-danger,#b13d2e)'
                      : status >= 400 ? 'var(--c-amber,#d49620)'
                      : 'var(--c-ok,#2f6f3e)';
          const path = (e.path || '/').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
          const lat = e.latency_ms != null ? Math.round(e.latency_ms) + 'ms' : '';
          return '<div class="lp-chooser-row" data-idx="' + i + '" role="button" tabindex="0">' +
                   '<span class="lp-chooser-status" style="color:' + color + '">' + status + '</span>' +
                   '<span class="lp-chooser-method">' + (e.method || 'GET') + '</span>' +
                   '<span class="lp-chooser-path">' + path + '</span>' +
                   '<span class="lp-chooser-lat">' + lat + '</span>' +
                 '</div>';
        }).join('') +
      '</div>';
    host.appendChild(div);
    // Wire close
    div.querySelector('.lp-chooser-close').addEventListener('click', () => div.remove());
    // Wire rows
    div.querySelectorAll('.lp-chooser-row').forEach(row => {
      const open = () => {
        const idx = parseInt(row.getAttribute('data-idx'), 10);
        const ev = events[idx];
        div.remove();
        _openOneEventDetail(ev);
      };
      row.addEventListener('click', open);
      row.addEventListener('keydown', e => { if (e.key === 'Enter') open(); });
    });
    // Dismiss on outside click (deferred so the originating click doesn't immediately close)
    setTimeout(() => {
      const onDoc = (e) => {
        if (!div.contains(e.target)) { div.remove(); document.removeEventListener('mousedown', onDoc); }
      };
      document.addEventListener('mousedown', onDoc);
    }, 50);
  }

  function _attachLiveStream() {
    const host = document.getElementById('livestream-host');
    if (!host || !window.APIN || !APIN.liveStream) return;
    if (_liveStreamCtrl) return;
    _liveStreamCtrl = APIN.liveStream.attach(host, {
      maxRows: 25,
      autoPauseHover: true,
      onClickRow: ev => {
        if (ev && ev.id != null && typeof openRequestDetail === 'function') {
          openRequestDetail(ev.id);
        } else if (ev) {
          // No id (event is the slim live frame, not a row id). Find the
          // most-recent matching row in the persisted log + open it.
          api('/api/account/usage/requests?range=1h&limit=20').then(r => {
            const items = (r.body && r.body.ok && r.body.data && r.body.data.items) || [];
            const match = items.find(it => it.path === ev.path && it.method === ev.method
                                          && it.status_code === ev.status_code);
            if (match && typeof openRequestDetail === 'function') {
              openRequestDetail(match.id);
            }
          }).catch(() => {});
        }
      },
    });
  }

  function _lbLiveStream(panel) {
    // The expanded live stream is a richer dashboard:
    //   1. Connection status + throughput
    //   2. Live filters (method, status, endpoint-contains)
    //   3. Rewind controls (last hour / between dates / between times)
    //   4. Bigger live feed (50 rows)
    //   5. Actions
    const stats = _liveStreamCtrl?.stats?.() || { eventCount: 0, uptimeMs: 0, lastEventAt: null, buffered: 0 };
    const uptimeMin = Math.round((stats.uptimeMs || 0) / 60000);
    const rate = (stats.uptimeMs > 0 ? (stats.eventCount / (stats.uptimeMs / 60000)).toFixed(1) : '0.0');
    const lastAgo = stats.lastEventAt ? Math.round((Date.now() - stats.lastEventAt) / 1000) + 's' : 'never';

    _appendSection(panel, 'Connection',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;line-height:1.85">'
      +   '<div><span class="ls-conn-dot" data-conn="connected" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--c-ok);margin-right:8px"></span> connected · /api/account/usage/stream</div>'
      +   '<div>stream uptime: <b>' + uptimeMin + ' min</b> · ' + stats.eventCount + ' events received</div>'
      +   '<div>last event <b>' + lastAgo + ' ago</b> · average rate <b>' + rate + ' req/min</b></div>'
      +   (stats.buffered > 0 ? '<div style="color:var(--c-amber)">' + stats.buffered + ' buffered (paused)</div>' : '')
      + '</div>');

    _appendSection(panel, 'Live filters',
      '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;font-family:\'JetBrains Mono\',monospace;font-size:12px">'
      +   '<label style="display:flex;flex-direction:column;gap:4px">'
      +     '<span style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:var(--ink-soft);letter-spacing:0.08em;text-transform:uppercase">show</span>'
      +     '<select class="lb-lf-mode" style="padding:6px 8px;border:1px solid var(--paper-edge);background:var(--paper);font-family:inherit;font-size:12px;color:var(--ink)">'
      +       '<option value="all">all rows</option><option value="errors">errors only</option>'
      +       '<option value="4xx">4xx only</option><option value="5xx">5xx only</option>'
      +     '</select>'
      +   '</label>'
      +   '<label style="display:flex;flex-direction:column;gap:4px">'
      +     '<span style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:var(--ink-soft);letter-spacing:0.08em;text-transform:uppercase">method</span>'
      +     '<select class="lb-lf-method" style="padding:6px 8px;border:1px solid var(--paper-edge);background:var(--paper);font-family:inherit;font-size:12px;color:var(--ink)">'
      +       '<option value="">any</option><option>GET</option><option>POST</option><option>PUT</option><option>DELETE</option><option>PATCH</option>'
      +     '</select>'
      +   '</label>'
      +   '<label style="display:flex;flex-direction:column;gap:4px;grid-column:span 2">'
      +     '<span style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:var(--ink-soft);letter-spacing:0.08em;text-transform:uppercase">endpoint contains</span>'
      +     '<input class="lb-lf-ep" type="search" placeholder="e.g. /predict" style="padding:6px 8px;border:1px solid var(--paper-edge);background:var(--paper);font-family:JetBrains Mono,monospace;font-size:12px;color:var(--ink)">'
      +   '</label>'
      + '</div>'
      + '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:var(--ink-soft);margin-top:8px">Filters apply only to the expanded feed below — the small card stays unfiltered.</div>');

    _appendSection(panel, 'Rewind',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:var(--ink)">'
      +   '<div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">'
      +     '<button class="lb-action" data-rewind="lasthour"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><use href="#i-clock"/></svg>last hour</button>'
      +     '<button class="lb-action" data-rewind="last24h"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><use href="#i-clock"/></svg>last 24h</button>'
      +     '<button class="lb-action" data-rewind="live">resume live</button>'
      +   '</div>'
      +   '<div style="display:grid;grid-template-columns:1fr 1fr auto;gap:10px;align-items:end">'
      +     '<label style="display:flex;flex-direction:column;gap:4px">'
      +       '<span style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:var(--ink-soft);letter-spacing:0.08em;text-transform:uppercase">from</span>'
      +       '<input class="lb-rw-from" type="datetime-local" style="padding:6px 8px;border:1px solid var(--paper-edge);background:var(--paper);font-family:inherit;font-size:12px;color:var(--ink)">'
      +     '</label>'
      +     '<label style="display:flex;flex-direction:column;gap:4px">'
      +       '<span style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:var(--ink-soft);letter-spacing:0.08em;text-transform:uppercase">to</span>'
      +       '<input class="lb-rw-to" type="datetime-local" style="padding:6px 8px;border:1px solid var(--paper-edge);background:var(--paper);font-family:inherit;font-size:12px;color:var(--ink)">'
      +     '</label>'
      +     '<button class="lb-action lb-rw-apply">apply range</button>'
      +   '</div>'
      + '</div>');

    _appendSection(panel, 'Live stream · last 50 rows',
      '<div id="lb-livestream-host" style="border:1px solid var(--paper-edge)"></div>');

    // 9.N.7.f · Commentary section inside live tail expanded
    _appendSection(panel, 'Live commentary',
      '<div class="lc-list" id="lb-tail-commentary"></div>');
    setTimeout(() => {
      const cmtHost = document.getElementById('lb-tail-commentary');
      if (cmtHost && window.APIN && APIN.commentary) {
        APIN.commentary.attachList(cmtHost, { max: 15 });
      }
    }, 250);

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export current rows' },
      { kind: 'share-link', icon: 'i-link', label: 'share this view' },
    ]));

    // Attach a second live-stream instance for the expanded view
    setTimeout(() => {
      const h = document.getElementById('lb-livestream-host');
      if (!h || !window.APIN || !APIN.liveStream) return;
      if (_liveStreamExpandedCtrl) try { _liveStreamExpandedCtrl.close(); } catch (e) {}
      _liveStreamExpandedCtrl = APIN.liveStream.attach(h, {
        maxRows: 50,
        autoPauseHover: false,   // expanded view stays live — user opted in
        onClickRow: ev => {
          if (ev && ev.id != null && typeof openRequestDetail === 'function') openRequestDetail(ev.id);
        },
      });
    }, 200);

    // Wire rewind buttons
    panel.querySelectorAll('[data-rewind]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const mode = btn.getAttribute('data-rewind');
        if (mode === 'live') {
          // Re-attach live stream
          const h = document.getElementById('lb-livestream-host');
          if (_liveStreamExpandedCtrl) try { _liveStreamExpandedCtrl.close(); } catch (e) {}
          _liveStreamExpandedCtrl = APIN.liveStream.attach(h, { maxRows: 50, autoPauseHover: false });
        } else {
          // Load static range into the expanded feed
          const range = mode === 'lasthour' ? '1h' : '24h';
          const { body } = await api('/api/account/usage/requests?range=' + range + '&limit=50');
          const items = (body && body.ok && body.data && body.data.items) || [];
          const h = document.getElementById('lb-livestream-host');
          if (_liveStreamExpandedCtrl) try { _liveStreamExpandedCtrl.close(); } catch (e) {}
          if (h) {
            h.innerHTML = '<div class="ls-shell"><div class="ls-status-bar"><span class="ls-conn-dot" data-conn="disconnected"></span><span class="ls-conn-label">rewound · ' + range + '</span><span class="ls-conn-count">' + items.length + ' rows</span></div><div class="ls-rows" id="lb-rewind-rows"></div></div>';
            const rowsHost = document.getElementById('lb-rewind-rows');
            items.forEach(it => {
              const div = document.createElement('div');
              div.innerHTML = '<div class="ls-row" data-rid="' + it.id + '">'
                + '<span class="ls-t">' + ((it.timestamp || '').slice(11, 19)) + '</span>'
                + '<span class="ls-m ls-m-' + (it.method || '').toLowerCase() + '">' + (it.method || '') + '</span>'
                + '<span class="ls-p">' + (it.path || '') + '</span>'
                + '<span class="ls-s">' + (it.status_code || '') + '</span>'
                + '<span class="ls-l">' + (it.latency_ms != null ? it.latency_ms + 'ms' : '·') + '</span>'
                + '<span class="ls-src"></span>'
                + '</div>';
              const r = div.firstElementChild;
              r.addEventListener('click', () => { if (typeof openRequestDetail === 'function') openRequestDetail(it.id); });
              rowsHost.appendChild(r);
            });
          }
        }
      });
    });

    _wireActions(panel);
  }

  // ─── 9.N.7.f · Live pulse expanded — 6 sections ──────────────────────
  function _lbLivePulse(panel) {
    const conn = (window.APIN && APIN.liveStreamConn) || { state: 'connected' };
    const accum = window.APIN && APIN.livePulseData && APIN.livePulseData.accumulator;
    const stats30 = accum ? accum.rollingStats(30) : { rate:0, errorRate:0, p50:0, p95:0, totalCount:0 };
    const stats300 = accum ? accum.rollingStats(300) : stats30;

    // Section 1: Connection diagnostics
    _appendSection(panel, 'Connection',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;line-height:1.85">'
      +   '<div><span class="lp-pulse-dot" data-state="active" style="display:inline-block;margin-right:8px;width:8px;height:8px;border-radius:50%;background:var(--c-ok)"></span> ' + (conn.state === 'connected' ? 'connected' : conn.state) + ' · /api/account/usage/stream</div>'
      +   '<div>last 30s: <b>' + stats30.totalCount + '</b> requests · <b>' + stats30.totalErrors + '</b> errors</div>'
      +   '<div>last 5 min: <b>' + stats300.totalCount + '</b> requests · rate <b>' + stats300.rate.toFixed(1) + '/sec</b></div>'
      + '</div>');

    // Section 2: Bigger pulse chart (re-attaches an expanded live-pulse)
    _appendSection(panel, 'Pulse · last 5 minutes',
      '<div id="lb-pulse-host" style="height:320px;border:1px solid var(--paper-edge);background:var(--paper)"></div>');
    setTimeout(() => {
      const h = document.getElementById('lb-pulse-host');
      if (h && window.APIN && APIN.livePulse) {
        if (_livePulseExpandedCtrl) try { _livePulseExpandedCtrl.destroy(); } catch (e) {}
        _livePulseExpandedCtrl = APIN.livePulse.attach(h, {
          compact: false,
          windowSec: 300,
          metric: 'rate',
          onClickPoint: ev => { if (ev && typeof openRequestDetail === 'function') openRequestDetail(ev.id); },
        });
      }
    }, 200);

    // Section 3: Live stats — 9.N.8g · KPI tiles now refresh every 1s from
    // the shared accumulator. Previously they were snapshotted at lightbox
    // open time and stayed frozen, which made "Error rate 0.0%" look wrong
    // when the underlying 30s window had actually shifted.
    _appendSection(panel, 'Live stats (last 30s)',
      '<div id="lb-pulse-kpis"></div>');
    function _renderPulseKpis() {
      const host = document.getElementById('lb-pulse-kpis');
      if (!host) return false;            // host gone — stop ticking
      const cur = accum ? accum.rollingStats(30)
                        : { rate:0, errorRate:0, p50:0, p95:0, totalCount:0, totalErrors:0 };
      host.innerHTML = _kpiGrid([
        _kpiTile('Rate',       cur.rate.toFixed(1) + '/sec',  'rolling 30s avg'),
        _kpiTile('p50',        fmtNum(cur.p50) + 'ms',         cur.totalCount + ' samples'),
        _kpiTile('p95',        fmtNum(cur.p95) + 'ms',         null),
        _kpiTile('Error rate', cur.errorRate.toFixed(1) + '%', (cur.totalErrors||0) + ' / ' + (cur.totalCount||0)),
      ]);
      return true;
    }
    _renderPulseKpis();
    const _kpiTimer = setInterval(() => {
      if (!_renderPulseKpis()) clearInterval(_kpiTimer);   // unmount cleanup
    }, 1000);

    // Section 4: Active endpoints (last 5min from accumulator events)
    if (accum) {
      const recentEvents = accum.events.slice(-200);
      const byEp = {};
      recentEvents.forEach(({ ev }) => {
        const p = ev.path || '?';
        byEp[p] = (byEp[p] || 0) + 1;
      });
      const ranked = Object.entries(byEp).sort((a, b) => b[1] - a[1]).slice(0, 6);
      const maxC = Math.max(1, ...ranked.map(r => r[1]));
      const epHtml = ranked.length
        ? '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;display:grid;grid-template-columns:1fr 120px 60px;gap:10px;align-items:center">'
          + ranked.map(([ep, c]) => {
              const w = (c / maxC) * 100;
              return '<div>' + escHtml(ep) + '</div>'
                + '<div style="height:12px;background:var(--paper-deep);border:1px solid var(--paper-edge);position:relative"><div style="position:absolute;left:0;top:0;height:100%;width:' + w + '%;background:var(--ink);opacity:0.65"></div></div>'
                + '<div style="text-align:right;font-variant-numeric:tabular-nums">' + c + '</div>';
            }).join('')
          + '</div>'
        : '<div class="lb-empty">no endpoint activity yet</div>';
      _appendSection(panel, 'Active endpoints (last few minutes)', epHtml);
    }

    // Section 5: Commentary list — humanized observations
    _appendSection(panel, 'Live commentary',
      '<div class="lc-list" id="lb-pulse-commentary"></div>');
    setTimeout(() => {
      const cmtHost = document.getElementById('lb-pulse-commentary');
      if (cmtHost && window.APIN && APIN.commentary) {
        APIN.commentary.attachList(cmtHost, { max: 20 });
      }
    }, 250);

    // Section 6: Actions
    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export window CSV' },
      { kind: 'share-link', icon: 'i-link', label: 'share this view' },
    ]));
    _wireActions(panel);
  }

  function _lbInsights(panel) {
    const insights = _insightsList || [];
    const toneColor = (t) => t === 'danger' ? 'var(--c-danger,#b01820)' : t === 'warn' ? 'var(--c-amber,#d49620)' : t === 'great' ? 'var(--c-ok,#2f6f3e)' : 'var(--ink-soft,#6b6453)';
    const byTone = { danger: 0, warn: 0, info: 0, great: 0 };
    insights.forEach(i => { byTone[i.tone || 'info'] = (byTone[i.tone || 'info'] || 0) + 1; });

    // Section 1: Overview
    _appendSection(panel, "What's noteworthy · " + insights.length + ' insights',
      '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12.5px;color:var(--ink);'
      + 'background:var(--paper-deep);border:1px solid var(--paper-edge);padding:12px 16px;margin-bottom:12px">'
      +   (byTone.danger ? '<span style="color:var(--c-danger)">' + byTone.danger + ' critical</span> · ' : '')
      +   (byTone.warn ? '<span style="color:var(--c-amber)">' + byTone.warn + ' warnings</span> · ' : '')
      +   (byTone.info ? byTone.info + ' informational' : '')
      +   (insights.length === 0 ? 'no insights surfaced — operating cleanly' : '')
      + '</div>'
      + '<p style="font-family:Fraunces,serif;font-style:italic;font-size:13px;color:var(--ink-soft);line-height:1.6;margin:0">'
      + 'These observations are derived automatically from the current data window. Each one combines per-endpoint stats with anomaly thresholds (error rate > 5% = critical, p95 > 2s = critical, traffic concentration > 80% in top-N = notable, etc.).'
      + '</p>');

    // Section 2: Each insight rendered in full with narrative
    if (insights.length > 0) {
      const fullHtml = insights.map((ins, idx) => {
        const c = toneColor(ins.tone);
        return '<div class="lb-ins-card" data-i="' + idx + '" style="border:1px solid var(--paper-edge);'
          + 'padding:14px 16px;margin-bottom:10px;background:var(--paper);cursor:pointer">'
          + '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
          +   '<svg viewBox="0 0 24 24" style="width:18px;height:18px;color:' + c + '" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#' + ins.icon + '"/></svg>'
          +   '<div style="font-family:Fraunces,serif;font-weight:500;font-size:14.5px;color:var(--ink);flex:1">' + escHtml(ins.title) + '</div>'
          +   '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11px;color:' + c + '">' + (ins.tone || 'info').toUpperCase() + '</div>'
          + '</div>'
          + '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11.5px;color:var(--ink-soft);margin-bottom:8px">' + escHtml(ins.body) + '</div>'
          + (ins.narrative ? '<div style="font-family:Fraunces,serif;font-size:13px;color:var(--ink);line-height:1.7;font-style:italic;border-left:2px solid ' + c + ';padding-left:12px;margin-top:8px">' + escHtml(ins.narrative) + '</div>' : '')
          + '<div style="font-family:Fraunces,serif;font-style:italic;font-size:11.5px;color:' + c + ';margin-top:10px">' + escHtml(ins.action.label) + ' →</div>'
          + '</div>';
      }).join('');
      _appendSection(panel, 'All insights', fullHtml);

      // Wire clicks on each insight card
      panel.querySelectorAll('.lb-ins-card').forEach((card, i) => {
        card.addEventListener('mouseenter', () => { card.style.background = 'var(--paper-deep, #e9e2d1)'; });
        card.addEventListener('mouseleave', () => { card.style.background = 'var(--paper)'; });
        card.addEventListener('click', e => {
          const ins = insights[i];
          if (!ins) return;
          if (window.APIN && APIN.fx) APIN.fx.ripple(card, e.clientX, e.clientY);
          if (ins.action.kind === 'filter-endpoint') {
            filterState.endpoint = ins.action.value;
            applyStateToUI(); applyFilters();
            APIN.lightbox.close();
          } else if (ins.action.kind === 'filter-status') {
            filterState.status = ins.action.value;
            applyStateToUI(); applyFilters();
            APIN.lightbox.close();
          }
        });
      });
    }

    _appendSection(panel, 'Actions', _actionRow([
      { kind: 'export-csv', icon: 'i-download', label: 'export window CSV' },
      { kind: 'share-link', icon: 'i-link', label: 'share this view' },
    ]));
    _wireActions(panel);
  }

  // ─── Boot ────────────────────────────────────────────────────────────
  (async function boot() {
    applyStateToUI();
    _bindHeatmapModeSwitcher();
    _bindCompareToggle();
    await loadKeys();
    await refreshAll();
    _injectExpandButtons();
    _attachLiveStream();   // 9.N.7 · SSE connection on the live-tail card
    _attachLivePulse();    // 9.N.7.f · Live pulse chart (shares SSE feed)
    _bindTimeScrubber();   // 9.N.7.f · Video-player on requests time-series
    // 9.N.8e · Recent requests: sortable headers + refresh + live updates.
    // The live feed reuses the SAME SSE accumulator the pulse + tail already
    // subscribe to, so there's no extra network connection — we just hook
    // a third listener into the existing event bus.
    _wireRecentControls();
    _subscribeRecentToLiveStream();
    startPoll();
  })();
})();
