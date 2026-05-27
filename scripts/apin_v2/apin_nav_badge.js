// Phase 8.H.C · unread-count badge on the "Alerts" nav link.
//
// Sits inside the nav anchor as:
//   <a href="/account/api/alerts">
//     Alerts
//     <span class="apin-nav-badge" id="apin-nav-badge" hidden>
//       <span class="apin-nav-badge-count apin-odometer"></span>
//     </span>
//   </a>
//
// The count is the user's UNREAD alerts (read + dismissed excluded).
// Updates from three signals:
//   1. Initial fetch on page load.
//   2. Polling every 30s via the same /unread-count endpoint.
//      (The toast layer also polls at 20s with since=cursor; we use a
//       slightly different cadence so the two don't always hit together.)
//   3. BroadcastChannel('apin-alerts') — when another tab sees new alerts
//      OR marks one read, our badge re-fetches.
//
// The odometer animation comes from /static/odometer.js (Stage 6.1b
// pattern, lifted from pipeline.html). Speed scales with the delta —
// going 0 → 1 slides cleanly, 0 → 50 cascades fast.

(function () {
  "use strict";

  const POLL_MS = 30 * 1000;

  let badgeEl, countEl;
  let pollTimer = null;
  let bc = null;
  let lastValue = -1;

  function unwrap(p) {
    if (p && typeof p === "object") {
      if (p.ok === true && "data" in p) return p.data;
      if (!("ok" in p)) return p;
    }
    return null;
  }

  function render(n) {
    if (!badgeEl || !countEl) return;
    n = Math.max(0, n | 0);
    if (n <= 0) {
      badgeEl.hidden = true;
    } else {
      badgeEl.hidden = false;
      if (window.APIN && window.APIN.odometer) {
        window.APIN.odometer.roll(countEl, n > 999 ? "999+" : n);
      } else {
        countEl.textContent = n > 999 ? "999+" : String(n);
      }
    }
    // Gentle pulse animation when the count INCREASES (not on decrement).
    if (lastValue >= 0 && n > lastValue) {
      badgeEl.classList.remove("apin-nav-badge-pulse");
      void badgeEl.offsetWidth;
      badgeEl.classList.add("apin-nav-badge-pulse");
    }
    lastValue = n;
  }

  async function fetchCount() {
    try {
      const r = await fetch("/api/account/alerts/unread-count",
                            { credentials: "same-origin" });
      if (!r.ok) return;
      const j = await r.json();
      const d = unwrap(j) || {};
      render(d.unread || 0);
    } catch (_) { /* offline — keep last value */ }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(function () {
      if (!document.hidden) fetchCount();
    }, POLL_MS);
  }
  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function setupBroadcast() {
    if (typeof BroadcastChannel === "undefined") return;
    try {
      bc = new BroadcastChannel("apin-alerts");
      bc.addEventListener("message", function (ev) {
        const msg = ev.data || {};
        // Any signal that COULD change unread count: re-fetch.
        if (msg.type === "cursor_advance" || msg.type === "read") {
          fetchCount();
        }
      });
    } catch (_) { /* unavailable */ }
  }

  function init() {
    badgeEl = document.getElementById("apin-nav-badge");
    countEl = document.getElementById("apin-nav-badge-count");
    if (!badgeEl || !countEl) return;  // not on a Console page
    fetchCount();
    startPolling();
    setupBroadcast();
    // Same-tab refresh: the toast layer fires `apin:alerts:changed` after
    // a successful mark-read. BroadcastChannel skips its own sender, so
    // we need this CustomEvent path for in-tab listeners.
    window.addEventListener("apin:alerts:changed", function () {
      fetchCount();
    });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        stopPolling();
      } else {
        startPolling();
        fetchCount();   // immediate refresh on focus
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
