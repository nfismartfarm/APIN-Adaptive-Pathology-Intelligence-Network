// 9.N.8h · Splash overlay dismiss logic.
//
// The splash HTML+CSS is injected inline by _inject_splash() (server-side)
// so it paints immediately when the page bytes reach the browser. This
// file handles the dismiss logic — and now ALSO waits for the page's
// own data fetches to complete before fading, so the user never sees
// the underlying page's "Loading your console..." / empty-KPI states.
//
// Lives in an external file (not inline) because account pages enforce a
// strict CSP: `script-src 'self'`. Inline <script> blocks are blocked.
// External files served from /static/ ARE allowed because `self` covers
// same-origin URLs.
//
// Dismiss is gated on ALL of:
//   1. /health responsive (< 1.5s round-trip)
//   2. window.load event has fired (all <img>, CSS, JS done)
//   3. No in-flight fetch/XHR for >= 500ms (page data is settled)
// Plus a hard 18s failsafe so the splash can never trap the user.
//
// To catch fetches issued by other page scripts, we monkey-patch
// window.fetch and XMLHttpRequest BEFORE any of them get a chance to
// run. This requires the script tag to be a synchronous <script src=...>
// (no defer/async) at the top of <body> — which it is, courtesy of
// _inject_splash() placing the splash block right after <body>.

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────
  var startedAt    = Date.now();
  var MIN_SHOWN_MS = 350;
  var MAX_SHOWN_MS = 18000;     // hard failsafe — never trap the user
  var GRACE_MS     = 500;       // inflight must be zero this long before dismiss

  var state = {
    healthOk:           false,
    windowLoaded:       false,
    inflight:           0,
    inflightZeroSince:  null,
    dismissed:          false,
  };

  var current = 5;
  var probes  = 0;

  // ── Fetch / XHR interception (PHASE 1 — install BEFORE any other code)
  // We patch as the very first thing this script does, so any subsequent
  // call by page-level JS (e.g. console_usage.js) is counted.
  function _shouldIgnoreUrl(url) {
    if (!url) return false;
    var u = String(url);
    // SSE streams never close on their own — would block forever.
    // /health is the splash's own probe — would loop forever.
    return /\/stream(\?|$|\/)/.test(u)
        || /\/sse(\?|$|\/)/.test(u)
        || /\/health$/.test(u)
        || /\/apin_sw\.js$/.test(u);
  }

  var origFetch = window.fetch;
  if (origFetch) {
    window.fetch = function (input, init) {
      var url = (typeof input === 'string') ? input
              : (input && input.url) ? input.url : '';
      var ignore = _shouldIgnoreUrl(url);
      if (!ignore) {
        state.inflight++;
        state.inflightZeroSince = null;
      }
      var p = origFetch.apply(this, arguments);
      // Use .then/.catch instead of .finally for older browsers
      var done = function () {
        if (!ignore) {
          state.inflight = Math.max(0, state.inflight - 1);
          if (state.inflight === 0) state.inflightZeroSince = Date.now();
          _maybeDismiss();
        }
      };
      p.then(done, done);
      return p;
    };
  }

  var origOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url) {
    this._splashUrl = url;
    return origOpen.apply(this, arguments);
  };
  var origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send = function () {
    var url = this._splashUrl;
    var ignore = _shouldIgnoreUrl(url);
    if (!ignore) {
      state.inflight++;
      state.inflightZeroSince = null;
      this.addEventListener('loadend', function () {
        state.inflight = Math.max(0, state.inflight - 1);
        if (state.inflight === 0) state.inflightZeroSince = Date.now();
        _maybeDismiss();
      });
    }
    return origSend.apply(this, arguments);
  };

  // ── DOM hooks — defer until the splash element exists ──────────────
  // The script runs inline-synchronously right after <body>, so the
  // splash <div> SHOULD be in the DOM already (we're injected after it).
  // But guard for any edge case.
  var splash, bar, sub;
  function _bindDOM() {
    splash = document.getElementById('apin-splash');
    bar    = document.getElementById('apin-splash-bar');
    sub    = document.getElementById('apin-splash-sub');
  }
  _bindDOM();
  if (!splash) {
    // splash element not in DOM yet — bind on DOMContentLoaded
    document.addEventListener('DOMContentLoaded', _bindDOM);
  }

  function _setProgress(p, label) {
    current = Math.min(99, Math.max(current, p));
    if (bar) bar.style.width = current + '%';
    if (label && sub) sub.textContent = label;
  }

  function _dismiss() {
    if (state.dismissed) return;
    state.dismissed = true;
    var elapsed = Date.now() - startedAt;
    var wait = Math.max(0, MIN_SHOWN_MS - elapsed);
    setTimeout(function () {
      _setProgress(100, 'ready');
      try { sessionStorage.setItem('apin_warm', '1'); } catch (_) {}
      setTimeout(function () {
        if (splash) {
          splash.classList.add('is-hidden');
          setTimeout(function () {
            if (splash && splash.parentNode) splash.parentNode.removeChild(splash);
          }, 350);
        }
      }, 150);
    }, wait);
  }

  function _maybeDismiss() {
    if (state.dismissed) return;
    if (!state.healthOk)       return;
    if (!state.windowLoaded)   return;
    if (state.inflight > 0)    return;
    if (state.inflightZeroSince === null) {
      state.inflightZeroSince = Date.now();
    }
    var quietFor = Date.now() - state.inflightZeroSince;
    if (quietFor < GRACE_MS) {
      // Wait the grace period — new requests might fire
      setTimeout(_maybeDismiss, GRACE_MS - quietFor + 50);
      return;
    }
    _dismiss();
  }

  // ── Status text + progress ticker — keeps moving so it doesn't look stuck
  var tick = setInterval(function () {
    if (state.dismissed) { clearInterval(tick); return; }
    // Drift the progress bar so the user has continuous visual feedback
    // even while we're waiting for the page's slow XHR to settle.
    if (current < 90) _setProgress(current + 1);
  }, 350);

  // ── Health probe ───────────────────────────────────────────────────
  function _probeHealth() {
    probes++;
    var t0 = Date.now();
    // Use the un-patched origFetch so we don't count the probe itself
    var f = origFetch || window.fetch;
    f.call(window, '/health', { cache: 'no-store', credentials: 'omit' })
      .then(function (r) {
        var dt = Date.now() - t0;
        if (r.ok && dt < 1500) {
          state.healthOk = true;
          _setProgress(45, 'loading data…');
          _maybeDismiss();
        } else if (r.ok) {
          _setProgress(Math.min(40, 15 + probes * 5),
                       'starting up (' + probes + '×)');
          if (probes < 25) setTimeout(_probeHealth, 800);
          else { state.healthOk = true; _maybeDismiss(); }
        } else {
          if (probes < 15) setTimeout(_probeHealth, 1200);
          else { state.healthOk = true; _maybeDismiss(); }
        }
      })
      .catch(function () {
        _setProgress(Math.min(30, 15 + probes * 4),
                     'waking the container…');
        if (probes < 25) setTimeout(_probeHealth, 1500);
        else { state.healthOk = true; _maybeDismiss(); }
      });
  }

  // ── Skip the probe on internal navigation — flag set after first warmup
  try {
    if (sessionStorage.getItem('apin_warm') === '1') {
      // Already warm — but still wait for page data to settle
      state.healthOk = true;
      _setProgress(50, 'loading data…');
      _maybeDismiss();
    } else {
      _probeHealth();
    }
  } catch (_) {
    _probeHealth();
  }

  // ── window.load — fires when all <img>, CSS, JS subresources are done
  if (document.readyState === 'complete') {
    state.windowLoaded = true;
    _setProgress(Math.max(current, 70), 'finalizing…');
    // window.load already passed; give XHR a small grace and dismiss
    setTimeout(_maybeDismiss, GRACE_MS);
  } else {
    window.addEventListener('load', function () {
      state.windowLoaded = true;
      _setProgress(Math.max(current, 70), 'finalizing…');
      setTimeout(_maybeDismiss, GRACE_MS);
    });
  }

  // ── Hard failsafe — splash hides no matter what after MAX_SHOWN_MS ───
  setTimeout(function () {
    if (!state.dismissed) {
      state.healthOk     = true;
      state.windowLoaded = true;
      _dismiss();
    }
  }, MAX_SHOWN_MS);

  // ── Keep-alive: prevent mid-session hibernation on free-tier HF Space.
  setInterval(function () {
    var f = origFetch || window.fetch;
    f.call(window, '/health', { cache: 'no-store', credentials: 'omit' })
      .catch(function () {});
  }, 240000);

  // ── Service worker — caches /static/*.js so subsequent visits hit disk.
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('/apin_sw.js').catch(function () {});
    });
  }
})();
