// Phase 8.H.D · Service worker — push notifications + static asset caching
//
// Two responsibilities:
//   1. Receive Web Push events and surface them as system notifications
//      (registered by /static/console_alert_prefs.js when user opts in).
//   2. 9.N.8h · Cache /static/*.js and other shared assets so subsequent
//      page loads bypass the network. Without this, every navigation
//      re-downloads ~3 MB of JS from the HF Space (slow on free tier).
//
// Registered globally by ui_template.html's splash bootstrap. It is
// non-fatal if registration fails (HTTPS required, some browsers block
// SW in iframes, etc.) — the site still works, just without the cache.

const CACHE_NAME = 'apin-static-v2';
// Assets cached on install (the common subset every page uses).
// Per-page extras (e.g. console_*.js) are cached opportunistically on
// first fetch via the runtime cache-first strategy below.
const PRECACHE_URLS = [
  '/static/apin_fx.js',
  '/static/apin_charts.js',
  '/static/apin_lightbox.js',
  '/static/apin_syntax.js',
  '/static/apin_toast.js',
  '/static/odometer.js',
  '/static/pressed_leaf.js',
  '/favicon.svg',
];

self.addEventListener("install", function (event) {
  // Activate immediately — no older SW to leave caches behind.
  self.skipWaiting();
  // Best-effort pre-cache — individual failures don't abort install
  event.waitUntil(
    caches.open(CACHE_NAME).then(function(cache){
      return Promise.allSettled(
        PRECACHE_URLS.map(function(u){
          return fetch(u, { credentials: 'omit' }).then(function(r){
            if (r && r.ok) return cache.put(u, r.clone());
          }).catch(function(){});
        })
      );
    })
  );
});

self.addEventListener("activate", function (event) {
  event.waitUntil(Promise.all([
    self.clients.claim(),
    // Purge old cache versions
    caches.keys().then(function(names){
      return Promise.all(names.map(function(n){
        if (n !== CACHE_NAME) return caches.delete(n);
      }));
    }),
  ]));
});

// ── Fetch handler — cache-first for static assets, network-first otherwise.
//
// Static (/static/*.js, /static/*.svg, /favicon.svg) is treated as
// effectively immutable for a session — serve from cache when present,
// fetch + store on miss. Stale-while-revalidate ensures the next visit
// picks up any updates.
//
// API calls (/api/*) and HTML pages skip the SW entirely — they need
// fresh data every time and the SW would only slow them down.
self.addEventListener("fetch", function (event) {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  // Match: /static/anything.js, /static/anything.svg, /favicon.svg, /logo.png
  const isStatic = (
    url.pathname.startsWith('/static/') ||
    url.pathname === '/favicon.svg' ||
    url.pathname === '/favicon.ico' ||
    url.pathname === '/logo.png' ||
    url.pathname === '/apin_sw.js'
  );
  if (!isStatic) return;   // let the page fall through to network normally

  event.respondWith(
    caches.open(CACHE_NAME).then(function(cache){
      return cache.match(req).then(function(cached){
        // Background refresh — serve stale, update silently
        const fetchAndCache = fetch(req, { credentials: 'omit' }).then(function(resp){
          if (resp && resp.ok) cache.put(req, resp.clone());
          return resp;
        }).catch(function(){ return cached; });
        return cached || fetchAndCache;
      });
    })
  );
});

self.addEventListener("push", function (event) {
  let payload = { title: "APIN alert", body: "" };
  try {
    if (event.data) payload = event.data.json();
  } catch (_) { /* fall through to defaults */ }
  const opts = {
    body: payload.body || "",
    icon: payload.icon || "/favicon.svg",
    badge: payload.badge || "/favicon.svg",
    tag: payload.tag || ("apin-" + Date.now()),  // dedup identical events
    data: payload.data || {},
    requireInteraction: payload.severity === "critical",
  };
  event.waitUntil(self.registration.showNotification(
    payload.title || "APIN alert", opts));
});

self.addEventListener("notificationclick", function (event) {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url)
    || "/account/api/alerts";
  event.waitUntil(
    self.clients.matchAll({ type: "window" }).then(function (clientsList) {
      // Focus an existing tab if one is open; otherwise open a new one.
      for (const c of clientsList) {
        if (c.url.indexOf("/account/api") >= 0 && "focus" in c) {
          c.navigate(url);
          return c.focus();
        }
      }
      if (self.clients.openWindow) return self.clients.openWindow(url);
    })
  );
});
