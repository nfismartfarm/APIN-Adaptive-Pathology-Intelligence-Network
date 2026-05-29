// apin_request_drawer.js — Phase 9.N.T9 · SHARED request-detail drawer.
//
// The full drawer (status bar · latency-vs-endpoint-baseline · source ·
// stage-timing waterfall · payload card with image + headers + bodies ·
// request/response · burst neighbours · endpoint-health sparklines · key card ·
// export curl/python/js/node + share) extracted VERBATIM from the Usage page so
// the per-key page + Traffic-tab request container open the IDENTICAL drawer.
//
// Public surface:  APIN.requestDrawer.open(rid)  ·  .close()
// DOM contract (present on both pages): #reqd #reqd-mask #reqd-title #reqd-sub
//   #reqd-body #reqd-close. Styles: apin_request_drawer.css. Highlighter:
//   apin_syntax.js. Image lightbox: apin_lightbox.js.
(function () {
  'use strict';

  // ── utilities (verbatim from console_usage.js) ──────────────────────────
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

  // ════════════════════════════════════════════════════════════════════════
  // Drawer body + section renderers/wirers (verbatim from console_usage.js)
  // ════════════════════════════════════════════════════════════════════════
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
    let stageSum = 0;
    for (const k of order) stageSum += Number(stages[k] || 0);
    // Sub-millisecond requests (e.g. a cached GET /api/version) legitimately
    // have a 0 ms breakdown across every stage. Four empty "0 ms · 0%" bars
    // read as broken, so show an explicit note instead.
    if (stageSum <= 0) {
      const lat = Math.max(0, Number(totalLatMs) || 0);
      return '<div class="reqd-section"><h4><svg class="icon"><use href="#i-clock"/></svg> Timeline' +
        '<span class="reqd-tl-total">total: ' + lat + ' ms</span></h4>' +
        '<div class="reqd-mute">' + (lat <= 1
          ? 'completed in under 1 ms — no measurable per-stage breakdown.'
          : 'per-stage breakdown not recorded for this request.') +
        '</div></div>';
    }
    let total = stageSum;
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

  // ── close / escape wiring (idempotent) ──────────────────────────────────
  function _wireClose() {
    const c = $('reqd-close'), m = $('reqd-mask');
    if (c && !c._rdWired) { c._rdWired = true; c.addEventListener('click', closeReqd); }
    if (m && !m._rdWired) { m._rdWired = true; m.addEventListener('click', closeReqd); }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _wireClose);
  } else { _wireClose(); }
  setTimeout(_wireClose, 200);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeReqd(); });

  // ── public surface ──────────────────────────────────────────────────────
  window.APIN = window.APIN || {};
  window.APIN.requestDrawer = { open: openRequestDetail, close: closeReqd, openReqd: openReqd };
})();
