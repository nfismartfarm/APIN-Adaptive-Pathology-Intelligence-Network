// Phase 8.H.D · Service worker stub for browser push.
//
// Registered by /static/console_alert_prefs.js when the user clicks
// "Enable" on the alert prefs section. Its job is to receive `push`
// events fired by the server (via Web Push API + VAPID) and surface
// them as system notifications.
//
// The send-side (server fanout via pywebpush + VAPID keys) is being
// rolled out separately; this SW is the receiver. When the server
// starts sending push, no additional client changes are needed.

self.addEventListener("install", function (event) {
  // Activate immediately — no SW already exists to leave caches behind.
  self.skipWaiting();
});

self.addEventListener("activate", function (event) {
  event.waitUntil(self.clients.claim());
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
