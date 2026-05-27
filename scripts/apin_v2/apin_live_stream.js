// apin_live_stream.js — Phase 9.N.7 · SSE live-stream client
// EventSource connection to /api/account/usage/stream + DOM renderer for a
// rolling feed of recent requests. Each new event slides in at the top with
// the "wet-ink" entry animation (translateY + blur fade). Click any row to
// open the request detail drawer (shared with the Recent requests table).
//
// Public surface:
//   APIN.liveStream.attach(hostEl, opts)
//     → returns a controller { pause(), resume(), clear(), isPaused() }
//
// opts:
//   maxRows        : 50               (cap; oldest fades when exceeded)
//   autoPauseHover : true             (pause stream while hovered)
//   onClickRow     : (event) => void  (called with the slim event object)
//
// Notes:
//   · The slim event objects come from the bus; if the user wants full
//     headers/body they open the request drawer (different endpoint).
//   · We auto-reconnect with exponential backoff on transport failures.
//   · Heartbeat comments (": heartbeat\n\n") are silently consumed by
//     EventSource; we do not need to handle them client-side.

(function () {
  "use strict";

  if (!window.APIN) window.APIN = {};

  // Map status code → glyph class (filled/half/outline open circle)
  function _statusGlyph(code) {
    const c = Number(code) || 0;
    if (c >= 200 && c < 300) return { cls: 'ls-status-ok',   filled: true };   // filled green
    if (c >= 300 && c < 400) return { cls: 'ls-status-info', filled: true };   // filled blue
    if (c === 429)           return { cls: 'ls-status-warn', filled: 'half' }; // half ochre
    if (c >= 400 && c < 500) return { cls: 'ls-status-warn', filled: 'half' }; // half ochre
    if (c >= 500)            return { cls: 'ls-status-err',  filled: false };  // outline crimson
    return { cls: 'ls-status-mute', filled: false };
  }

  function _glyphSvg(g) {
    const fill = g.filled === true ? 'currentColor' : 'none';
    return '<svg viewBox="0 0 16 16" class="ls-glyph ' + g.cls + '" aria-hidden="true">'
      + '<circle cx="8" cy="8" r="5" fill="' + fill + '" stroke="currentColor" stroke-width="1.4"/>'
      + (g.filled === 'half' ? '<path d="M8 3 A5 5 0 0 1 8 13 Z" fill="currentColor"/>' : '')
      + '</svg>';
  }

  // UA → source-icon-glyph (use existing console_icons.svg symbols)
  function _sourceIcon(ua) {
    const u = (ua || '').toLowerCase();
    if (u.includes('python')) return 'i-flask';
    if (u.includes('curl'))   return 'i-clipboard';
    if (u.includes('node') || u.includes('javascript')) return 'i-route';
    if (u.includes('mozilla') || u.includes('chrome') || u.includes('safari') || u.includes('firefox')) return 'i-eye';
    return 'i-flask';
  }

  function _fmtTime(iso) {
    if (!iso) return '';
    // "2026-05-26 19:00:43.583529" → "19:00:43"
    const m = String(iso).match(/(\d{2}:\d{2}:\d{2})/);
    return m ? m[1] : iso;
  }

  function _escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // The renderer for a single row
  function _rowHtml(ev) {
    const g = _statusGlyph(ev.status_code);
    const src = _sourceIcon(ev.ua || '');
    const cls = 'ls-row' + (ev.status_code >= 500 ? ' ls-row-err' : ev.status_code >= 400 ? ' ls-row-warn' : '');
    return '<div class="' + cls + '" data-rid="' + _escHtml(ev.id || '') + '" '
      + 'data-event=\'' + _escHtml(JSON.stringify(ev)) + '\'>'
      +   '<span class="ls-t">' + _escHtml(_fmtTime(ev.timestamp)) + '</span>'
      +   '<span class="ls-m ls-m-' + _escHtml((ev.method || 'GET').toLowerCase()) + '">' + _escHtml(ev.method || 'GET') + '</span>'
      +   '<span class="ls-p">' + _escHtml(ev.path || '/') + '</span>'
      +   '<span class="ls-s">' + _glyphSvg(g) + _escHtml(String(ev.status_code || '·')) + '</span>'
      +   '<span class="ls-l">' + _escHtml(String(ev.latency_ms != null ? ev.latency_ms + 'ms' : '·')) + '</span>'
      +   '<span class="ls-src" title="' + _escHtml(ev.ua || '') + '">'
      +     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><use href="#' + src + '"/></svg>'
      +   '</span>'
      + '</div>';
  }

  // Main attach function
  function attach(hostEl, opts) {
    opts = opts || {};
    const maxRows = opts.maxRows || 50;
    const autoPauseOnHover = opts.autoPauseHover !== false;

    if (!hostEl) {
      console.warn('APIN.liveStream.attach: no host element');
      return { pause() {}, resume() {}, clear() {}, isPaused: () => false };
    }

    // Build the host structure — compact layout, plant empty state,
    // commentary ticker line
    hostEl.innerHTML =
      '<div class="ls-shell">' +
        '<div class="ls-status-bar">' +
          '<span class="ls-conn-dot" data-conn="connecting"></span>' +
          '<span class="ls-conn-label">connecting…</span>' +
          '<span class="ls-conn-count" data-count>0 events</span>' +
          '<div class="ls-spacer"></div>' +
          '<button class="ls-pause-btn" type="button" data-paused="false" title="Pause stream">' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><use href="#i-pause"/></svg>' +
            '<span class="ls-pause-text">pause</span>' +
          '</button>' +
        '</div>' +
        '<div class="ls-hover-note" hidden>paused while you read · move cursor away to resume</div>' +
        '<div class="ls-empty" hidden>' +
          // Plant-themed leaf SVG with central vein traveler dot
          '<svg viewBox="0 0 120 140" class="ls-empty-leaf" aria-hidden="true">' +
            '<defs>' +
              '<filter id="ls-wobble" x="-2%" y="-2%" width="104%" height="104%">' +
                '<feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="5" result="t"/>' +
                '<feDisplacementMap in="SourceGraphic" in2="t" scale="0.5" xChannelSelector="R" yChannelSelector="G"/>' +
              '</filter>' +
            '</defs>' +
            // Stem
            '<line x1="60" y1="135" x2="60" y2="110" stroke="var(--ink-soft, #6b6453)" stroke-width="1.5" stroke-linecap="round" filter="url(#ls-wobble)"/>' +
            // Leaf outline (almond shape)
            '<path d="M60 110 C 30 100, 18 60, 60 18 C 102 60, 90 100, 60 110 Z" fill="none" stroke="var(--ink, #1a1612)" stroke-width="1.4" stroke-linejoin="round" filter="url(#ls-wobble)"/>' +
            // Central vein (path the dot will follow via animateMotion)
            '<path id="ls-vein-path" d="M60 108 L60 22" fill="none" stroke="var(--ink-soft, #6b6453)" stroke-width="0.7" filter="url(#ls-wobble)"/>' +
            // Side veins
            '<path d="M60 90 L40 75 M60 90 L80 75 M60 70 L36 55 M60 70 L84 55 M60 50 L42 40 M60 50 L78 40" fill="none" stroke="var(--ink-soft, #6b6453)" stroke-width="0.6" opacity="0.7" filter="url(#ls-wobble)"/>' +
            // Traveler dot (animateMotion along the central vein)
            '<circle r="2.4" fill="var(--c-ok, #2f6f3e)" opacity="0.85">' +
              '<animateMotion dur="3.5s" repeatCount="indefinite" rotate="auto">' +
                '<mpath href="#ls-vein-path"/>' +
              '</animateMotion>' +
              '<animate attributeName="opacity" values="0;0.9;0.9;0" dur="3.5s" repeatCount="indefinite"/>' +
            '</circle>' +
          '</svg>' +
          '<div class="ls-empty-caption">no requests in sight</div>' +
          '<div class="ls-empty-sub">send a request to see it here</div>' +
        '</div>' +
        '<div class="ls-rows" role="log" aria-live="polite"></div>' +
        '<div class="ls-ticker-host"></div>' +
        '<div class="ls-tail-divider">— live —</div>' +
      '</div>';

    // Show empty state when no rows
    const emptyEl = hostEl.querySelector('.ls-empty');
    const updateEmpty = () => {
      const hasRows = hostEl.querySelectorAll('.ls-rows .ls-row').length > 0;
      if (emptyEl) emptyEl.hidden = hasRows;
    };
    updateEmpty();

    // Commentary ticker below the rows (1-line rotating)
    const tickerHost = hostEl.querySelector('.ls-ticker-host');
    if (tickerHost && window.APIN && APIN.commentary) {
      try { APIN.commentary.attachTicker(tickerHost); } catch (e) {}
    }

    const rowsEl     = hostEl.querySelector('.ls-rows');
    const dotEl      = hostEl.querySelector('.ls-conn-dot');
    const labelEl    = hostEl.querySelector('.ls-conn-label');
    const countEl    = hostEl.querySelector('.ls-conn-count');
    const pauseBtn   = hostEl.querySelector('.ls-pause-btn');
    const hoverNote  = hostEl.querySelector('.ls-hover-note');

    let eventCount = 0;
    let manualPause = false;
    let hoverPause = false;
    const buffered = [];   // hold events while paused so we don't lose them
    let es = null;
    let reconnectTimer = null;
    let reconnectDelay = 1000;
    const startedAt = Date.now();
    let lastEventAt = null;

    function isPaused() { return manualPause || hoverPause; }

    function setConn(state, label) {
      dotEl.setAttribute('data-conn', state);
      labelEl.textContent = label;
      // 9.N.7.f · Expose connection state globally so the pulse widget's
      // pulse-dot color can react to disconnections too.
      try {
        window.APIN = window.APIN || {};
        window.APIN.liveStreamConn = { state, label, ts: Date.now() };
      } catch (e) {}
    }

    // 9.N.7.f · Batched DOM inserts via rAF coalescing.
    //
    // Previously each appendEvent did insertBefore() synchronously. A 25-
    // event burst meant 25 layout reflows + 25 wetInk animations all in
    // <1s, which compounded with the live-pulse redraw and SSE parser to
    // saturate the main thread.
    //
    // New strategy: appendEvent queues the event + schedules ONE rAF
    // callback that flushes all queued events via DocumentFragment in a
    // single insertBefore. Layout cost goes from O(N) reflows to O(1)
    // reflow per animation frame — the same trick virtualized lists use.
    //
    // The accumulator feed still happens synchronously per event so the
    // live-pulse chart sees them immediately; only the DOM insertion is
    // batched (the canvas can show data before the DOM row shows up).
    const _pendingRows = [];
    let _flushScheduled = false;

    function _scheduleFlush() {
      if (_flushScheduled) return;
      _flushScheduled = true;
      requestAnimationFrame(_flushPendingRows);
    }

    function _flushPendingRows() {
      _flushScheduled = false;
      if (_pendingRows.length === 0) return;

      // Build all rows in a DocumentFragment (off-tree, no layout cost).
      const frag = document.createDocumentFragment();
      const created = [];
      // We pop in reverse so when prepended in order, newest ends up on top.
      // Push order in _pendingRows is chronological (oldest queued first);
      // we want the newest at top, so iterate forward and stack at the
      // fragment start.
      for (let i = _pendingRows.length - 1; i >= 0; i--) {
        const ev = _pendingRows[i];
        const tmp = document.createElement('div');
        tmp.innerHTML = _rowHtml(ev);
        const row = tmp.firstElementChild;
        if (!row) continue;
        // Click → openRequestDetail
        row.addEventListener('click', (e) => {
          if (window.APIN && APIN.fx) APIN.fx.ripple(row, e.clientX, e.clientY);
          if (opts.onClickRow) opts.onClickRow(ev);
        });
        frag.appendChild(row);
        created.push(row);
      }
      _pendingRows.length = 0;

      // ONE insertBefore for the whole batch = 1 layout reflow.
      rowsEl.insertBefore(frag, rowsEl.firstChild);

      // Wet-ink animations on each new row (CSS animations don't trigger
      // reflow individually — they run on the compositor).
      if (window.APIN && APIN.fx && APIN.fx.wetInk) {
        for (const row of created) APIN.fx.wetInk(row);
      }

      // Cap row count — fade out excess oldest rows.
      // Doing this here (after batch insert) avoids removing rows one-by-one
      // during a burst, which would also be N reflows.
      const excess = rowsEl.children.length - maxRows;
      if (excess > 0) {
        for (let i = 0; i < excess; i++) {
          const last = rowsEl.lastElementChild;
          if (!last) break;
          last.style.transition = 'opacity 180ms cubic-bezier(0.22, 1, 0.36, 1)';
          last.style.opacity = '0';
          // Defer removal so the fade is visible; removeChild is one
          // layout, but staggered ~200ms apart it's invisible.
          setTimeout(((node) => () => { try { node.remove(); } catch (e) {} })(last), 200);
        }
      }
    }

    function appendEvent(ev) {
      eventCount++;
      countEl.textContent = eventCount + ' event' + (eventCount === 1 ? '' : 's');
      lastEventAt = Date.now();
      // Feed accumulator synchronously so the chart shows it on next paint.
      try {
        if (window.APIN && APIN.livePulseData) APIN.livePulseData.feed(ev);
      } catch (e) {}
      // Hide empty state immediately on first event.
      if (eventCount === 1) {
        const emptyEl = hostEl.querySelector('.ls-empty');
        if (emptyEl) emptyEl.hidden = true;
      }
      // Queue the row insert — flushed in next rAF.
      _pendingRows.push(ev);
      // Backpressure: if the queue grows beyond what we'd display anyway,
      // drop oldest queued events (they would just be evicted on render).
      if (_pendingRows.length > maxRows * 2) {
        _pendingRows.splice(0, _pendingRows.length - maxRows);
      }
      _scheduleFlush();
    }

    function handleEvent(ev) {
      if (!ev || ev.type === 'ready') return;
      if (isPaused()) {
        // Buffer up to 200 events while paused — drop oldest if overflow
        buffered.push(ev);
        if (buffered.length > 200) buffered.shift();
        return;
      }
      appendEvent(ev);
    }

    function drainBuffered() {
      while (buffered.length > 0 && !isPaused()) {
        const ev = buffered.shift();
        appendEvent(ev);
      }
    }

    function connect() {
      try {
        if (es) try { es.close(); } catch (e) {}
        setConn('connecting', 'connecting…');
        es = new EventSource('/api/account/usage/stream');
        es.addEventListener('open', () => {
          setConn('connected', 'connected');
          reconnectDelay = 1000;
        });
        es.addEventListener('message', (msgEv) => {
          try {
            const ev = JSON.parse(msgEv.data);
            handleEvent(ev);
          } catch (e) {}
        });
        es.addEventListener('error', () => {
          setConn('reconnecting', 'reconnecting…');
          try { es.close(); } catch (e) {}
          es = null;
          clearTimeout(reconnectTimer);
          reconnectTimer = setTimeout(connect, Math.min(reconnectDelay, 30000));
          reconnectDelay = Math.min(reconnectDelay * 2, 30000);
        });
      } catch (e) {
        setConn('disconnected', 'disconnected');
      }
    }

    // Pause button
    pauseBtn.addEventListener('click', () => {
      manualPause = !manualPause;
      pauseBtn.setAttribute('data-paused', String(manualPause));
      const tx = pauseBtn.querySelector('.ls-pause-text');
      const icon = pauseBtn.querySelector('use');
      if (tx) tx.textContent = manualPause ? 'resume' : 'pause';
      if (icon) icon.setAttribute('href', manualPause ? '#i-play' : '#i-pause');
      pauseBtn.title = manualPause ? 'Resume stream' : 'Pause stream';
      if (!manualPause) drainBuffered();
    });

    // Auto-pause on hover
    if (autoPauseOnHover) {
      hostEl.addEventListener('mouseenter', () => {
        hoverPause = true;
        if (!manualPause) hoverNote.hidden = false;
      });
      hostEl.addEventListener('mouseleave', () => {
        hoverPause = false;
        hoverNote.hidden = true;
        if (!manualPause) drainBuffered();
      });
    }

    connect();

    return {
      pause() { manualPause = true; pauseBtn.click(); pauseBtn.click(); },
      resume() { manualPause = false; drainBuffered(); },
      clear() { rowsEl.innerHTML = ''; eventCount = 0; countEl.textContent = '0 events'; },
      isPaused,
      stats() {
        return {
          eventCount,
          uptimeMs: Date.now() - startedAt,
          lastEventAt,
          buffered: buffered.length,
        };
      },
      close() {
        try { es && es.close(); } catch (e) {}
        clearTimeout(reconnectTimer);
      },
    };
  }

  window.APIN.liveStream = { attach };
})();
