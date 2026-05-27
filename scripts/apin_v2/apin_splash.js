// 9.N.8h · Splash overlay dismiss logic.
//
// The splash HTML+CSS is injected inline by _inject_splash() (server-side)
// so it paints immediately when the page bytes reach the browser. THIS
// file handles the dismiss logic — health probe, progress bar advance,
// fade-out, session flag, keep-alive ping, service-worker registration.
//
// Lives in an external file (not inline) because account pages enforce a
// strict CSP: `script-src 'self'`. Inline <script> blocks are blocked.
// External files served from /static/ ARE allowed because `self` covers
// same-origin URLs.
//
// Loaded by every HTML page that flows through _load_html, _load_v2_html,
// or _load_landing_html in apin_server.py — those three helpers all call
// _inject_splash which references this file.

(function () {
  'use strict';

  var splash = document.getElementById('apin-splash');
  if (!splash) return;
  var bar = document.getElementById('apin-splash-bar');
  var sub = document.getElementById('apin-splash-sub');

  var startedAt = Date.now();
  var MIN_SHOWN_MS = 350;
  var MAX_SHOWN_MS = 18000;   // hard failsafe — never trap the user
  var probes = 0;
  var current = 5;

  function _setProgress(p, label) {
    current = Math.min(99, Math.max(current, p));
    if (bar) bar.style.width = current + '%';
    if (label && sub) sub.textContent = label;
  }

  function _hideSplash() {
    var elapsed = Date.now() - startedAt;
    var wait = Math.max(0, MIN_SHOWN_MS - elapsed);
    setTimeout(function () {
      _setProgress(100, 'ready');
      try { sessionStorage.setItem('apin_warm', '1'); } catch (_) {}
      setTimeout(function () {
        splash.classList.add('is-hidden');
        setTimeout(function () {
          if (splash.parentNode) splash.parentNode.removeChild(splash);
        }, 350);
      }, 150);
    }, wait);
  }

  // ── Skip the probe on internal navigation — flag set after first warmup
  try {
    if (sessionStorage.getItem('apin_warm') === '1') {
      _setProgress(70, 'restoring session…');
      setTimeout(_hideSplash, 250);
    } else {
      _startProbe();
    }
  } catch (_) {
    _startProbe();
  }

  function _startProbe() {
    setTimeout(_hideSplash, MAX_SHOWN_MS);   // hard failsafe
    _probe();
  }

  function _probe() {
    probes++;
    var t0 = Date.now();
    fetch('/health', { cache: 'no-store', credentials: 'omit' })
      .then(function (r) {
        var dt = Date.now() - t0;
        if (r.ok && dt < 1500) {
          _setProgress(85, 'loading assets…');
          _hideSplash();
        } else if (r.ok) {
          _setProgress(Math.min(75, 15 + probes * 6),
                       'starting up (' + probes + '×)');
          if (probes < 25) setTimeout(_probe, 800);
          else _hideSplash();
        } else {
          if (probes < 15) setTimeout(_probe, 1200);
          else _hideSplash();
        }
      })
      .catch(function () {
        _setProgress(Math.min(60, 15 + probes * 5),
                     'waking the container…');
        if (probes < 25) setTimeout(_probe, 1500);
        else _hideSplash();
      });
  }

  // ── Keep-alive: prevent mid-session hibernation on free-tier HF Space.
  //    Container goes idle after ~30 min; ping every 4 min keeps it warm
  //    while the user has the tab open.
  setInterval(function () {
    fetch('/health', { cache: 'no-store', credentials: 'omit' })
      .catch(function () {});
  }, 240000);

  // ── Service worker — caches /static/*.js so subsequent visits hit disk.
  //    Best-effort: registration failure is non-fatal (HTTPS required, some
  //    browsers block SW in iframes, etc.).
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('/apin_sw.js').catch(function () {});
    });
  }
})();
