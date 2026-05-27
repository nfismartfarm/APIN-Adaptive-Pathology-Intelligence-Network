/* APIN telemetry library · v1 (Stage 3)
 * ──────────────────────────────────────────────────────────────────────
 * Drop-in instrumentation that beacons batches into /api/telemetry/ingest.
 *
 * Goals
 *   1. Industry-grade event capture: page views, clicks, impressions,
 *      hovers, scroll depth, web-vitals, errors, API call timing,
 *      session lifecycle. Same shape as Stage 1's _PV_COLS / _EVT_COLS /
 *      _CLICK_COLS so server-side ingest accepts every field as-is.
 *   2. Privacy-respecting: no raw IP / no raw UA stored client-side
 *      (server bucket those), no fingerprinting, no third-party cookies.
 *   3. Never breaks the page. Every observer is wrapped in try/catch.
 *      Every flush has a sendBeacon fallback for unload.
 *   4. Works on a 3-year-old Android (feature-detect every API).
 *
 * Usage
 *   <script src="/static/telemetry.js" defer></script>
 *   <script>
 *     window.addEventListener('DOMContentLoaded', () => {
 *       APIN_TLM.init({
 *         page_route: '/pipeline',
 *         user_id:    null,        // server-set after login
 *         debug:      false,
 *       });
 *     });
 *   </script>
 *
 * Or auto-init via <body data-tlm-route="/pipeline"> attribute.
 * ────────────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  // ── Singleton guard ────────────────────────────────────────────────
  if (typeof window === 'undefined') return;
  if (window.APIN_TLM && window.APIN_TLM.__loaded) return;

  // ── Internal state ─────────────────────────────────────────────────
  var state = {
    initialized:        false,
    endpoint:           '/api/telemetry/ingest',
    flush_interval_ms:  10000,
    batch_max:          80,
    debug:              false,
    user_id:            null,
    guest_session_id:   null,
    browser_session_id: null,
    page_view_id:       null,
    page_route:         null,
    page_url:           null,
    page_title:         null,
    page_entered_at:    null,    // ISO string
    page_entered_at_ms: null,    // [PDA-1 F-1] performance.now() at pv start
    session_started_at_ms: null, // performance.now() at first init()
    last_active_at:     null,
    active_ms:          0,
    idle_ms:            0,
    hidden_ms:          0,
    last_visibility_change_at: null,
    visibility_state:   'visible',
    max_scroll_pct:     0,
    scroll_milestones:  [],
    click_count_pv:     0,
    api_call_count_pv:  0,
    error_count_pv:     0,
    queue: {
      page_views: [], clicks: [], impressions: [], events: [],
      api_calls: [], errors: [], goals: [], experiments_exposures: [],
    },
    flush_timer:        null,
    _flush_lock:        false,
    _observed_impr:     null,   // IntersectionObserver instance
    _vitals_observers:  [],     // PerformanceObserver instances
    _orig_fetch:        null,
    _onerror_orig:      null,
    _hover_idle_timer:  null,
  };

  // ── Tiny utilities ─────────────────────────────────────────────────
  function log() {
    if (state.debug && typeof console !== 'undefined' && console.log) {
      try { console.log.apply(console, ['[apin-tlm]'].concat([].slice.call(arguments))); } catch (e) {}
    }
  }
  function safe(fn) { return function () { try { return fn.apply(this, arguments); } catch (e) { log('safe() caught', e); } }; }
  function nowIso() { return new Date().toISOString(); }
  function nowMs()  { return (window.performance && performance.now) ? performance.now() : Date.now(); }

  // Stable per-tab session id (lives across navigation in the same tab).
  function makeId() {
    try {
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
      // Fallback: 22-char base36
      var s = '';
      for (var i = 0; i < 22; i++) s += Math.floor(Math.random() * 36).toString(36);
      return s;
    } catch (e) { return String(Date.now()) + String(Math.random()).slice(2, 8); }
  }

  function getOrMakeBrowserSessionId() {
    try {
      var k = '_apin_tlm_sid';
      var existing = sessionStorage.getItem(k);
      if (existing) return existing;
      var sid = makeId();
      sessionStorage.setItem(k, sid);
      return sid;
    } catch (e) {
      // Private mode / disabled storage — fall back to in-memory.
      return makeId();
    }
  }

  // Best-effort device hints (small, no raw UA exposed).
  function deviceProbe() {
    var hints = {};
    try {
      var w = window.screen.width || 0, h = window.screen.height || 0;
      hints.screen_width = w; hints.screen_height = h;
      hints.viewport_width  = window.innerWidth  || 0;
      hints.viewport_height = window.innerHeight || 0;
      hints.pixel_ratio = window.devicePixelRatio || 1;
      hints.timezone = (Intl && Intl.DateTimeFormat && Intl.DateTimeFormat().resolvedOptions().timeZone) || null;
      hints.locale = navigator.language || null;
      if (navigator.connection && navigator.connection.effectiveType) {
        hints.network_effective_type = navigator.connection.effectiveType;
      }
      if (navigator.hardwareConcurrency) hints.cpu_cores = navigator.hardwareConcurrency;
      if (navigator.deviceMemory)        hints.memory_gb = navigator.deviceMemory;
      // Coarse device_type bucket
      var ua = (navigator.userAgent || '').toLowerCase();
      if (/mobi|android|iphone|ipad|ipod/.test(ua)) hints.device_type = 'mobile';
      else if (/tablet/.test(ua))                   hints.device_type = 'tablet';
      else                                          hints.device_type = 'desktop';
      // PWA flag
      try {
        if (window.matchMedia('(display-mode: standalone)').matches
            || window.navigator.standalone === true) {
          hints.is_pwa_installed = 1;
        }
      } catch (e) {}
    } catch (e) {}
    return hints;
  }

  function utmFromUrl() {
    var out = {};
    try {
      var p = new URLSearchParams(location.search);
      ['utm_source','utm_medium','utm_campaign','utm_term','utm_content','gclid','fbclid']
        .forEach(function (k) { var v = p.get(k); if (v) out[k] = v.slice(0, 200); });
    } catch (e) {}
    return out;
  }

  function referrerInfo() {
    try {
      var ref = document.referrer || '';
      if (!ref) return {};
      var u = new URL(ref);
      return { referrer_url: ref.slice(0, 500),
               referrer_host: u.host,
               referrer_path: u.pathname };
    } catch (e) { return {}; }
  }

  // ── Click target metadata ──────────────────────────────────────────
  function describeTarget(el) {
    if (!el || !el.tagName) return {};
    var out = {
      target_tag:     (el.tagName || '').toLowerCase().slice(0, 32),
      target_id:      (el.id || '').slice(0, 120),
      target_classes: (el.className || '').toString().slice(0, 200),
    };
    try {
      var text = (el.innerText || el.value || '').trim();
      if (text) out.target_text = text.slice(0, 160);
    } catch (e) {}
    // Data attributes — opt-in via data-* tags the page declares
    try {
      var data = {};
      if (el.dataset) {
        for (var k in el.dataset) {
          if (Object.prototype.hasOwnProperty.call(el.dataset, k)) {
            data[k] = String(el.dataset[k]).slice(0, 200);
          }
        }
      }
      if (Object.keys(data).length) {
        out.target_data_attrs = JSON.stringify(data).slice(0, 600);
      }
    } catch (e) {}
    return out;
  }

  // ── Queue + flush ──────────────────────────────────────────────────
  function enqueue(table, item) {
    if (!state.initialized) return;
    var q = state.queue[table];
    if (!q) return;
    // Stamp common owner fields
    if (!item.browser_session_id) item.browser_session_id = state.browser_session_id;
    if (state.user_id && !item.user_id) item.user_id = state.user_id;
    if (state.guest_session_id && !item.guest_session_id) item.guest_session_id = state.guest_session_id;
    if (state.page_view_id && (table === 'clicks' || table === 'impressions'
                                || table === 'events' || table === 'api_calls'
                                || table === 'errors')) {
      if (!item.page_view_id) item.page_view_id = state.page_view_id;
    }
    q.push(item);
    if (totalQueued() >= state.batch_max) flush();
  }

  function totalQueued() {
    var n = 0;
    for (var k in state.queue) if (state.queue.hasOwnProperty(k)) n += state.queue[k].length;
    return n;
  }

  function buildBatch(opts) {
    opts = opts || {};
    // Always include the session object so the server can upsert.
    // [Stage 6] · stamp last_heartbeat_at on EVERY flush so the server's
    // "live now" tile can tell which sessions are currently alive. The
    // upsert helper already accepts this field; we just have to send it.
    // Also include the current page_route so the tile's per-route counts
    // are accurate even when the user never explicitly emitted a page_view.
    // [Stage 6.1] · on pagehide-beacon, stamp session_end_at so the
    // server can drop us from the active count INSTANTLY instead of
    // waiting for the 30 s heartbeat window to expire. The server's
    // live_sessions query treats `session_end_at IS NOT NULL AND
    // session_end_at >= last_heartbeat_at` as "this session has gone."
    var sess = {
      id: state.browser_session_id,
      session_start_at: state.page_entered_at,
      last_heartbeat_at: nowIso(),
    };
    if (opts.ending) sess.session_end_at = nowIso();
    if (state.page_route)       sess.current_route = state.page_route;
    if (state.user_id)          sess.user_id = state.user_id;
    if (state.guest_session_id) sess.guest_session_id = state.guest_session_id;
    var dp = deviceProbe(), utm = utmFromUrl(), ref = referrerInfo();
    for (var k in dp)  if (dp.hasOwnProperty(k))  sess[k]  = dp[k];
    for (var k2 in utm) if (utm.hasOwnProperty(k2)) sess[k2] = utm[k2];
    for (var k3 in ref) if (ref.hasOwnProperty(k3)) sess[k3] = ref[k3];

    var batch = { session: sess };
    for (var tbl in state.queue) if (state.queue.hasOwnProperty(tbl)) {
      if (state.queue[tbl].length) batch[tbl] = state.queue[tbl];
    }
    return batch;
  }

  function clearQueue() {
    for (var k in state.queue) if (state.queue.hasOwnProperty(k)) state.queue[k] = [];
  }

  function flush(opts) {
    opts = opts || {};
    if (!state.initialized) return;
    if (state._flush_lock && !opts.force) return;
    if (totalQueued() === 0 && !opts.force && !opts.ending) return;
    state._flush_lock = true;

    // [Stage 6.1] · forward the `ending` hint to buildBatch so the
    // session object carries a session_end_at on the final pagehide
    // beacon. The server uses that to drop us from the live count
    // instantly instead of waiting for the heartbeat window to lapse.
    var batch = buildBatch({ ending: !!opts.ending });
    var body = '';
    try { body = JSON.stringify(batch); } catch (e) { state._flush_lock = false; return; }
    clearQueue();

    // sendBeacon on unload (more reliable than fetch during pagehide).
    // [PDA-1 F-5] · browsers cap sendBeacon at 64 KB; oversize payloads
    // are dropped silently. Detect first and either chunk-flush or fall
    // through to fetch+keepalive (which has no fixed cap and tends to
    // survive page unload on modern browsers).
    var BEACON_MAX_BYTES = 60000;  // 60 KB · 4 KB safety margin under the 64 KB cap
    if (opts.beacon && navigator.sendBeacon && body.length <= BEACON_MAX_BYTES) {
      try {
        var blob = new Blob([body], { type: 'application/json' });
        var ok = navigator.sendBeacon(state.endpoint, blob);
        log('beacon flush', body.length, 'bytes · ok=' + ok);
        state._flush_lock = false;
        if (ok) return;
        // sendBeacon returned false (queue full / quota exhausted) — fall through to fetch.
      } catch (e) {
        log('beacon failed, falling back to fetch', e);
      }
    } else if (opts.beacon && body.length > BEACON_MAX_BYTES) {
      log('beacon payload too large (' + body.length + ' B); falling back to keepalive fetch');
    }

    try {
      // keepalive lets the request survive a page navigation without being
      // counted as the user's next-page navigation
      fetch(state.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body,
        keepalive: true,
        credentials: 'same-origin',
      }).then(function (r) {
        log('flush', r.status, body.length, 'bytes');
      }).catch(function (e) {
        log('flush failed', e);
      }).then(function () {
        state._flush_lock = false;
      });
    } catch (e) {
      log('flush threw', e);
      state._flush_lock = false;
    }
  }

  // ── Page view bookkeeping ──────────────────────────────────────────
  function startPageView(opts) {
    opts = opts || {};
    state.page_view_id    = makeId();
    state.page_route      = opts.page_route || location.pathname || '/';
    state.page_url        = opts.page_url   || location.href;
    state.page_title      = opts.page_title || document.title || '';
    state.page_entered_at    = nowIso();
    state.page_entered_at_ms = nowMs();  // [PDA-1 F-1] for ms_since_page_view
    state.last_active_at  = nowMs();
    state.active_ms       = 0;
    state.idle_ms         = 0;
    state.hidden_ms       = 0;
    state.last_visibility_change_at = nowMs();
    state.visibility_state = document.visibilityState || 'visible';
    state.max_scroll_pct  = 0;
    state.scroll_milestones = [];
    state.click_count_pv  = 0;
    state.api_call_count_pv = 0;
    state.error_count_pv  = 0;

    enqueue('page_views', {
      id:                 state.page_view_id,
      page_url:           state.page_url,
      page_route:         state.page_route,
      page_title:         state.page_title,
      entered_at:         state.page_entered_at,
      navigation_type:    (performance && performance.getEntriesByType
                            ? (performance.getEntriesByType('navigation')[0] || {}).type
                            : null) || 'navigate',
    });
  }

  function endPageView() {
    if (!state.page_view_id) return;
    accrueActivity();   // make sure the active/idle/hidden tally is current
    // Append a "left_at + accrued metrics" page_views ROW. The server
    // schema only INSERTs new rows, so we send an updated row with the
    // same id — _bulk_insert will skip it on PK conflict. Until we add
    // an UPDATE path, treat this as a sentinel event instead.
    var endedPvId = state.page_view_id;
    enqueue('events', {
      event_type:        'page_view_end',
      event_name:        'pv_end',
      occurred_at:       nowIso(),
      properties: JSON.stringify({
        page_view_id:        endedPvId,
        left_at:             nowIso(),
        active_duration_ms:  Math.round(state.active_ms),
        idle_duration_ms:    Math.round(state.idle_ms),
        hidden_duration_ms:  Math.round(state.hidden_ms),
        max_scroll_depth_pct: state.max_scroll_pct,
        scroll_milestones_reached: JSON.stringify(state.scroll_milestones),
        click_count:         state.click_count_pv,
        api_call_count:      state.api_call_count_pv,
        error_count:         state.error_count_pv,
      }),
    });
    // [PDA-1 F-3] · clear the pv id so subsequent events (after bfcache
    // resume, for example) don't attribute to a dead page_view_id. If
    // the page is resumed from bfcache, the next pageshow will trigger
    // startPageView() through the auto-init.
    state.page_view_id = null;
  }

  // Active / idle / hidden accounting · debounced timer.
  var IDLE_MS = 30000;  // 30 s of no input = idle
  function accrueActivity() {
    var t = nowMs();
    var delta = t - state.last_visibility_change_at;
    if (delta < 0) delta = 0;
    if (state.visibility_state === 'hidden') {
      state.hidden_ms += delta;
    } else {
      // visible — split active vs idle by last_active_at
      var idle_age = t - state.last_active_at;
      if (idle_age >= IDLE_MS) {
        state.idle_ms += delta;
      } else {
        state.active_ms += delta;
      }
    }
    state.last_visibility_change_at = t;
  }

  // ── Web Vitals (FCP, LCP, CLS, INP, TTFB) ──────────────────────────
  function initVitals() {
    if (!window.PerformanceObserver) return;
    var vitals = {};

    function emit(name, val) {
      vitals[name] = val;
      // Emit as a single event per metric (most metrics fire once or
      // a small number of times; CLS keeps accumulating).
      enqueue('events', {
        event_type:  'web_vital',
        event_name:  name,
        occurred_at: nowIso(),
        properties: JSON.stringify({ value: val, page_view_id: state.page_view_id }),
      });
    }

    // FCP — first-contentful-paint
    try {
      var fcpObs = new PerformanceObserver(function (list) {
        list.getEntries().forEach(function (e) {
          if (e.name === 'first-contentful-paint') emit('fcp_ms', Math.round(e.startTime));
        });
      });
      fcpObs.observe({ type: 'paint', buffered: true });
      state._vitals_observers.push(fcpObs);
    } catch (e) {}

    // LCP — largest-contentful-paint (cumulative; last entry wins on unload)
    try {
      var lcpEntries = [];
      var lcpObs = new PerformanceObserver(function (list) {
        list.getEntries().forEach(function (e) { lcpEntries.push(e); });
      });
      lcpObs.observe({ type: 'largest-contentful-paint', buffered: true });
      state._vitals_observers.push(lcpObs);
      var commitLcp = function () {
        if (lcpEntries.length) {
          var last = lcpEntries[lcpEntries.length - 1];
          emit('lcp_ms', Math.round(last.renderTime || last.loadTime || last.startTime));
        }
      };
      document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'hidden') commitLcp();
      }, { capture: true });
    } catch (e) {}

    // CLS — layout-shift (cumulative)
    try {
      var clsValue = 0;
      var clsObs = new PerformanceObserver(function (list) {
        list.getEntries().forEach(function (entry) {
          if (!entry.hadRecentInput) clsValue += entry.value;
        });
      });
      clsObs.observe({ type: 'layout-shift', buffered: true });
      state._vitals_observers.push(clsObs);
      document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'hidden') emit('cls', clsValue);
      }, { capture: true });
    } catch (e) {}

    // INP — interaction-to-next-paint (estimate via event timing)
    try {
      var maxInp = 0;
      var inpObs = new PerformanceObserver(function (list) {
        list.getEntries().forEach(function (e) {
          var dur = e.duration | 0;
          if (dur > maxInp) maxInp = dur;
        });
      });
      inpObs.observe({ type: 'event', durationThreshold: 16, buffered: true });
      state._vitals_observers.push(inpObs);
      document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'hidden' && maxInp > 0) emit('inp_ms', maxInp);
      }, { capture: true });
    } catch (e) {}

    // TTFB · from navigation timing
    try {
      var nav = performance.getEntriesByType('navigation')[0];
      if (nav && nav.responseStart) emit('ttfb_ms', Math.round(nav.responseStart));
    } catch (e) {}
  }

  // ── Click + hover + scroll tracking ───────────────────────────────
  function initClickTracking() {
    document.addEventListener('click', safe(function (ev) {
      var el = ev.target;
      // Walk up looking for a [data-track] ancestor — clicks on icons
      // inside buttons should attribute to the button.
      var pickEl = el;
      try {
        var cur = el;
        for (var i = 0; cur && i < 6; i++) {
          if (cur.hasAttribute && cur.hasAttribute('data-track')) { pickEl = cur; break; }
          if (cur.tagName && /^(button|a|input|select|textarea)$/i.test(cur.tagName)) {
            pickEl = cur; break;
          }
          cur = cur.parentNode;
        }
      } catch (e) {}

      var meta = describeTarget(pickEl);
      var clk = {
        target_tag:               meta.target_tag,
        target_id:                meta.target_id || null,
        target_classes:           meta.target_classes || null,
        target_text:              meta.target_text || null,
        target_data_attrs:        meta.target_data_attrs || null,
        click_x_viewport:         ev.clientX | 0,
        click_y_viewport:         ev.clientY | 0,
        click_x_page:             ((window.scrollX || 0) + ev.clientX) | 0,
        click_y_page:             ((window.scrollY || 0) + ev.clientY) | 0,
        viewport_width_at_click:  window.innerWidth  | 0,
        viewport_height_at_click: window.innerHeight | 0,
        viewport_y_pct: (function () {
          var docH = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight) || 1;
          return Math.round((window.scrollY + ev.clientY) * 100 / docH);
        })(),
        modifier_keys: (function () {
          var k = [];
          if (ev.shiftKey) k.push('shift'); if (ev.ctrlKey) k.push('ctrl');
          if (ev.altKey)   k.push('alt');   if (ev.metaKey) k.push('meta');
          return k.length ? k.join(',') : null;
        })(),
        click_type: ev.button === 0 ? 'left' : ev.button === 1 ? 'middle' : ev.button === 2 ? 'right' : 'other',
        // [PDA-1 F-1] · these were subtracting a duration from a timestamp,
        // producing nonsense. Now: ms-since the current page_view started,
        // and ms-since the original APIN_TLM.init() call (which is the
        // session start for telemetry purposes; not the browser tab open).
        ms_since_page_view:     Math.max(0, Math.round(nowMs() - (state.page_entered_at_ms || nowMs()))),
        ms_since_session_start: Math.max(0, Math.round(nowMs() - (state.session_started_at_ms || nowMs()))),
        occurred_at:            nowIso(),
      };
      state.click_count_pv += 1;
      enqueue('clicks', clk);
      bumpActivity();
    }), { capture: false, passive: true });
  }

  function initImpressionTracking() {
    if (!window.IntersectionObserver) return;
    var seen = new WeakSet();
    state._observed_impr = new IntersectionObserver(safe(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        if (seen.has(entry.target)) return;
        seen.add(entry.target);
        var meta = describeTarget(entry.target);
        enqueue('impressions', {
          target_id:      meta.target_id || null,
          target_classes: meta.target_classes || null,
          target_text:    meta.target_text || null,
          intersection_ratio_at_first_visible: Math.round(entry.intersectionRatio * 100) / 100,
          // [PDA-1 F-1] same fix as clicks: ms since this page_view started.
          ms_since_page_view: Math.max(0, Math.round(nowMs() - (state.page_entered_at_ms || nowMs()))),
          occurred_at: nowIso(),
        });
        // Stop observing once we've recorded it.
        try { state._observed_impr.unobserve(entry.target); } catch (e) {}
      });
    }), { threshold: 0.5 });

    // Auto-attach to elements that opt in NOW.
    try {
      document.querySelectorAll('[data-track-impression]').forEach(function (el) {
        state._observed_impr.observe(el);
      });
    } catch (e) {}

    // [PDA-1 F-6] · Pipeline Atlas (and the dashboard) render widgets
    // lazily; some [data-track-impression] elements appear MINUTES after
    // init. A MutationObserver on the body catches them as they get
    // inserted. We only watch for added nodes, not attributes — keeping
    // the observer cheap. The observer is disconnected on pagehide
    // along with the rest of the listeners (timer cleanup, etc.).
    if (window.MutationObserver) {
      try {
        var mo = new MutationObserver(safe(function (mutations) {
          mutations.forEach(function (m) {
            (m.addedNodes || []).forEach(function (node) {
              if (node.nodeType !== 1) return;   // ELEMENT_NODE
              try {
                if (node.matches && node.matches('[data-track-impression]')) {
                  state._observed_impr.observe(node);
                }
                if (node.querySelectorAll) {
                  node.querySelectorAll('[data-track-impression]').forEach(function (el) {
                    state._observed_impr.observe(el);
                  });
                }
              } catch (e) {}
            });
          });
        }));
        mo.observe(document.body, { childList: true, subtree: true });
        state._impr_mo = mo;
      } catch (e) {}
    }
  }

  function initScrollTracking() {
    var milestones = [25, 50, 75, 90, 100];
    var hit = {};
    var lastEmit = 0;
    window.addEventListener('scroll', safe(function () {
      var docH = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
      var winH = window.innerHeight;
      if (docH <= winH) return;
      var pct = Math.min(100, Math.round((window.scrollY + winH) * 100 / docH));
      if (pct > state.max_scroll_pct) state.max_scroll_pct = pct;
      milestones.forEach(function (m) {
        if (pct >= m && !hit[m]) {
          hit[m] = true;
          state.scroll_milestones.push(m);
          var t = nowMs();
          if (t - lastEmit > 500) {
            enqueue('events', {
              event_type:  'scroll',
              event_name:  'scroll_milestone_' + m,
              occurred_at: nowIso(),
              properties:  JSON.stringify({ pct: m }),
            });
            lastEmit = t;
          }
        }
      });
      bumpActivity();
    }), { passive: true });
  }

  function initActivityHooks() {
    function bump() { state.last_active_at = nowMs(); }
    ['mousemove','keydown','touchstart','wheel','pointermove'].forEach(function (k) {
      window.addEventListener(k, safe(bump), { capture: true, passive: true });
    });
  }
  function bumpActivity() { state.last_active_at = nowMs(); }

  function initVisibilityHooks() {
    document.addEventListener('visibilitychange', safe(function () {
      accrueActivity();
      state.visibility_state = document.visibilityState || 'visible';
      if (state.visibility_state === 'hidden') {
        flush({ beacon: true });
      } else if (state.visibility_state === 'visible') {
        // [Stage 6.2] · Producer-side visibility-resume force-flush.
        //
        // Chrome aggressively throttles setInterval in hidden tabs (1 Hz
        // floor, dropping to 1/min after 5 min, sometimes 1 per 5 min in
        // intensive-throttling mode). Our 10 s flush_timer therefore
        // skips heartbeats while we're backgrounded, and our
        // last_heartbeat_at ages past the server's 30 s live-window. The
        // observer (/pipeline) thinks this tab is dead.
        //
        // The instant the user looks at this tab again we want to be
        // visible in their Live Now count immediately — fire a flush
        // synchronously here so the next SSE push to /pipeline includes
        // us. Without `ending:true` so the server only updates the
        // heartbeat, not the session-end timestamp.
        try { flush({ force: true }); } catch (_) {}
      }
    }), false);
    window.addEventListener('pagehide', safe(function (ev) {
      accrueActivity();
      endPageView();
      // [Stage 6.1] · pass ending:true so the beacon body carries an
      // explicit session_end_at — the server uses that to drop us from
      // the Live Now count instantly, not on heartbeat-window expiry.
      // For bfcache (ev.persisted=true) we DON'T set ending — the page
      // might be resumed, and the pageshow handler will rearm the timer.
      flush({ beacon: true, force: true, ending: !ev || !ev.persisted });
      // [PDA-1 F-4] · only clear the flush timer when the page is being
      // unloaded for real. If `ev.persisted` is true, the page is going
      // into bfcache and might come back — pageshow will rearm. If it's
      // false, clear now so we don't keep firing in the background.
      if (!ev || !ev.persisted) {
        if (state.flush_timer) {
          clearInterval(state.flush_timer);
          state.flush_timer = null;
        }
      }
    }), { capture: true });
    // [PDA-1 F-4] · pageshow handles bfcache resume — restart the timer
    // and emit a fresh page_view if needed.
    window.addEventListener('pageshow', safe(function (ev) {
      if (ev && ev.persisted) {
        if (!state.page_view_id) startPageView({ page_route: state.page_route });
        if (!state.flush_timer) {
          state.flush_timer = setInterval(safe(flush), state.flush_interval_ms);
        }
      }
    }));
    window.addEventListener('beforeunload', safe(function () {
      accrueActivity();
      endPageView();
      // [Stage 6.1] · same explicit-end signal as pagehide
      flush({ beacon: true, force: true, ending: true });
    }));
    // [Stage 6.1] · visibilitychange to "hidden" isn't necessarily an
    // unload (could be tab-switch, screen-lock), so we DON'T set ending
    // there. The existing visibilitychange flush already runs without
    // the ending flag; if the user returns the periodic flush picks up
    // again and they reappear in Live Now.
  }

  // ── Fetch wrapper · captures API call timing for /api/* ───────────
  function initFetchWrapper() {
    if (!window.fetch || state._orig_fetch) return;
    state._orig_fetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      var url    = typeof input === 'string' ? input : (input && input.url) || '';
      var method = (init && init.method) || (input && input.method) || 'GET';
      var start  = nowMs();
      // Only track our own /api/ + /predict/ paths to avoid third-party noise.
      // [PDA-1 F-2] · explicitly skip our own ingest endpoint — we don't
      // want to record an api_call row for the flush itself (wasted bytes,
      // risk of feedback loop if the queue grows by one per flush).
      var trackable = (typeof url === 'string')
        && (url.indexOf(state.endpoint) < 0)
        && (url.indexOf('/api/telemetry/ingest') < 0)
        && (
          url.indexOf('/api/') >= 0 || url.indexOf('/predict/') >= 0
          || url.indexOf('/auth/') >= 0);
      var body_size = 0;
      try {
        if (init && init.body) {
          body_size = (typeof init.body === 'string') ? init.body.length
                    : (init.body.size || 0);
        }
      } catch (e) {}
      var p;
      try { p = state._orig_fetch(input, init); }
      catch (e) {
        if (trackable) emitApiCall(url, method, start, null, 0, e);
        throw e;
      }
      return p.then(function (resp) {
        if (trackable) emitApiCall(url, method, start, resp.status,
                                   body_size, null, resp);
        return resp;
      }).catch(function (e) {
        if (trackable) emitApiCall(url, method, start, null, body_size, e);
        throw e;
      });
    };
  }

  function emitApiCall(url, method, start, status, body_size, err, resp) {
    state.api_call_count_pv += 1;
    var lat = Math.round(nowMs() - start);
    var ep  = '';
    try { ep = new URL(url, location.origin).pathname.slice(0, 200); }
    catch (e) { ep = String(url).slice(0, 200); }
    var row = {
      endpoint:                ep,
      method:                  (method || 'GET').toUpperCase().slice(0, 16),
      request_body_size_bytes: body_size | 0,
      status_code:             status,
      client_latency_ms:       lat,
      retry_count:             0,
      occurred_at:             nowIso(),
    };
    if (err) {
      row.error_type = String((err && err.name) || 'NetworkError').slice(0, 64);
    }
    if (resp) {
      try {
        var cl = resp.headers.get('content-length');
        if (cl) row.response_body_size_bytes = parseInt(cl, 10) || 0;
      } catch (e) {}
    }
    enqueue('api_calls', row);
  }

  // ── Errors ────────────────────────────────────────────────────────
  function initErrorHooks() {
    state._onerror_orig = window.onerror;
    window.onerror = function (msg, src, lineno, colno, err) {
      try {
        state.error_count_pv += 1;
        enqueue('errors', {
          error_type:    'js_runtime',
          error_message: String(msg).slice(0, 500),
          error_stack:   err && err.stack ? String(err.stack).slice(0, 2000) : null,
          source_file:   String(src).slice(0, 200),
          line_number:   lineno | 0,
          column_number: colno | 0,
          occurred_at:   nowIso(),
        });
      } catch (e) {}
      if (typeof state._onerror_orig === 'function') {
        try { return state._onerror_orig.apply(this, arguments); } catch (e) {}
      }
      return false;
    };
    window.addEventListener('unhandledrejection', safe(function (ev) {
      state.error_count_pv += 1;
      enqueue('errors', {
        error_type:    'unhandled_promise',
        error_message: (ev.reason && (ev.reason.message || ev.reason.toString()) || '').slice(0, 500),
        error_stack:   (ev.reason && ev.reason.stack) ? String(ev.reason.stack).slice(0, 2000) : null,
        occurred_at:   nowIso(),
      });
    }));
  }

  // ── Public surface ─────────────────────────────────────────────────
  var api = {
    __loaded: true,
    init: function (opts) {
      if (state.initialized) return api;
      opts = opts || {};
      state.endpoint           = opts.endpoint           || state.endpoint;
      state.flush_interval_ms  = opts.flush_interval_ms  || state.flush_interval_ms;
      state.batch_max          = opts.batch_max          || state.batch_max;
      state.debug              = !!opts.debug;
      state.user_id            = opts.user_id            || null;
      state.guest_session_id   = opts.guest_session_id   || null;
      state.browser_session_id = getOrMakeBrowserSessionId();
      // [PDA-1 F-1] · stamp the session-relative origin so ms_since_*
      // timing fields can be computed without subtracting accumulators
      // from timestamps.
      state.session_started_at_ms = nowMs();

      log('init', { sid: state.browser_session_id, endpoint: state.endpoint });

      // [Stage 5 fix] Flip `initialized` BEFORE startPageView so the very
      // first page_view enqueue is actually accepted. Previously this
      // assignment lived at the bottom of init(), meaning startPageView()
      // -> enqueue('page_views', ...) hit the `if (!state.initialized)
      // return;` guard and the row was silently dropped. Every downstream
      // click / event / api_call carried a `page_view_id` referencing
      // that missing parent → FK violations cascaded for the entire
      // session.
      state.initialized = true;

      startPageView({
        page_route: opts.page_route,
        page_url:   opts.page_url,
        page_title: opts.page_title,
      });

      initActivityHooks();
      initVisibilityHooks();
      initScrollTracking();
      initClickTracking();
      initImpressionTracking();
      initVitals();
      initFetchWrapper();
      initErrorHooks();

      // Periodic flush
      state.flush_timer = setInterval(safe(flush), state.flush_interval_ms);

      // Stage 6.1 · land-to-visibility latency · fire the first flush
      // ASAP so the server sees the new session within ~300 ms of the
      // page becoming interactive. Previously this was 1500 ms; combined
      // with a 6 s KPI poll that meant up to ~7-8 s worst case before
      // the visitor showed up in the Live Now tile.
      setTimeout(safe(flush), 300);
      return api;
    },

    event: function (name, props) {
      if (!state.initialized) return;
      enqueue('events', {
        event_type:  'custom',
        event_name:  String(name).slice(0, 80),
        occurred_at: nowIso(),
        properties:  props ? JSON.stringify(props).slice(0, 2000) : null,
      });
    },

    goal: function (name, props) {
      if (!state.initialized) return;
      enqueue('goals', {
        goal_name:    String(name).slice(0, 80),
        achieved_at:  nowIso(),
        properties:   props ? JSON.stringify(props).slice(0, 2000) : null,
      });
    },

    exposure: function (experiment_name, variant) {
      if (!state.initialized) return;
      enqueue('experiments_exposures', {
        experiment_name: String(experiment_name).slice(0, 80),
        variant:         String(variant).slice(0, 80),
        exposed_at:      nowIso(),
      });
    },

    setUser: function (info) {
      info = info || {};
      if (info.user_id != null) state.user_id = info.user_id;
      if (info.guest_session_id != null) state.guest_session_id = info.guest_session_id;
      if (info.user_pseudoid != null) state.user_pseudoid = info.user_pseudoid;
    },

    observeImpression: function (el) {
      if (state._observed_impr && el) {
        try { state._observed_impr.observe(el); } catch (e) {}
      }
    },

    pageView: function (opts) {
      // For SPA navigations: end the current page view, start a new one.
      endPageView();
      flush();
      startPageView(opts || {});
    },

    flush: function () { flush({ force: true }); },

    _state: state,   // exposed for debug / tests; not part of public API
  };

  // ── Auto-init · runs on every page that loads telemetry.js ──
  // The library identifies the page in one of two ways:
  //   1. Explicit · <body data-tlm-route="/whatever"> (preferred — the
  //      page declares its own canonical route, which can differ from
  //      the URL path e.g. for parameterised routes).
  //   2. Implicit · falls back to location.pathname so EVERY page that
  //      simply includes <script src="/static/telemetry.js" defer> in
  //      its <head> gets instrumented without any other markup change.
  // The Stage-6 design moved away from a server-side middleware after
  // the gzip-middleware ordering proved brittle; instead each page now
  // includes the script directly. The implicit fallback makes that
  // a one-line edit per page.
  function maybeAutoInit() {
    try {
      var b = document.body;
      // Anti-opt-out: a page can suppress telemetry by setting
      // data-tlm-disable on <body>. Useful for static error pages.
      if (b && b.hasAttribute && b.hasAttribute('data-tlm-disable')) return;
      var route = null;
      if (b && b.getAttribute) route = b.getAttribute('data-tlm-route');
      if (!route) {
        // Fall back to URL pathname · strip trailing slash for stability.
        try {
          route = (location && location.pathname) || '/';
          if (route.length > 1 && route.endsWith('/')) {
            route = route.slice(0, -1);
          }
        } catch (e) { route = '/'; }
      }
      api.init({
        page_route: route,
        debug:      !!(b && b.hasAttribute && b.hasAttribute('data-tlm-debug')),
      });
    } catch (e) {}
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', maybeAutoInit);
  } else {
    maybeAutoInit();
  }

  window.APIN_TLM = api;
})();
