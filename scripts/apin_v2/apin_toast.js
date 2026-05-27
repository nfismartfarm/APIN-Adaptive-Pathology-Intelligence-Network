// Phase 8.H.B · Global toast system for the Console.
//
// Behaviour:
//   - Polls /api/account/alerts/unread-count?since=<cursor> every 20s.
//   - Bootstraps `cursor` from latest_id on first load so we don't re-toast
//     the whole alert history when the user opens the Console.
//   - New unread alerts slide in from the bottom-right (paper-card style).
//   - Up to MAX_VISIBLE (3) toasts can be stacked at once.
//   - A 4th incoming alert collapses the stack into a single PILL that
//     reads "N new alerts" with an odometer-animated count. Click → /alerts.
//   - Auto-dismiss: info 6s, warn 10s, critical never.
//   - Closing a toast (× or inline-action click) marks the alert read.
//   - Auto-dismiss leaves the alert UNREAD (so the badge stays incremented
//     until the user goes to /alerts and engages explicitly).
//   - BroadcastChannel('apin-alerts') syncs across tabs:
//       * { type:"cursor_advance", latest_id }  — non-active tabs update
//         their internal cursor (so they don't re-toast on focus)
//       * { type:"read", id }                    — all tabs hide a toast
//         that another tab has dismissed
//   - Active tab only: window-visibility + document.hasFocus(). Background
//     tabs DON'T toast (the badge still increments via the chip's poll).
//
// Inline action verbs supported (driven by alert.details.action.kind):
//   view_key            → /account/api/keys/{public_id}
//   view_keys_list      → /account/api/keys
//   view_webhook        → /account/api/webhooks  (anchored to id)
//   view_webhooks_list  → /account/api/webhooks
//   view_delivery       → /account/api/webhooks#delivery-{delivery_id}
//   view_settings       → /account/api/settings
//   re_enable_webhook   → /account/api/webhooks#enable-{id}
//   view_request        → /account/api/keys/{public_id}#req-{request_id}
//   adjust_quota        → /account/api/settings#quotas
//   extend_session      → POST /auth/extend  (no nav — toast updates inline)
//   approve_block_ip    → opens a small confirmation prompt (Phase 8.H.C)
//
// Module exports via window.APIN.toast:
//   .showLocal(alert)   — same-page injection (used by chip session warning)
//   .markRead(id)       — fetch /alerts/{id}/read + tell broadcast peers
//   .cursor()           — current cursor value (for chip badge sync)

(function () {
  "use strict";

  // ── Tunables ─────────────────────────────────────────────────────────
  const POLL_INTERVAL_MS = 20 * 1000;
  const MAX_VISIBLE = 3;
  const AUTO_DISMISS = { info: 6000, warn: 10000, critical: 0 };
  const SLIDE_OUT_MS = 280;
  const PILL_COLLAPSE_THRESHOLD = 4;  // 4th + collapses

  // ── Internal state ───────────────────────────────────────────────────
  const state = {
    cursor: 0,
    pollTimer: null,
    bc: null,
    container: null,
    visible: [],    // [{id, alert, node, dismissTimer}]
    pillNode: null,
    pillCount: 0,
    pillIds: [],    // alert ids currently in the pill
  };

  // ── Tiny helpers ─────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function isConsolePage() {
    return location.pathname.indexOf("/account/api") === 0;
  }
  function getCsrf() {
    const m = document.querySelector('meta[name="csrf-token"]');
    return m ? (m.getAttribute("content") || "") : "";
  }
  function escapeHTML(s) {
    if (s == null) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function unwrap(payload) {
    if (payload && typeof payload === "object") {
      if (payload.ok === true && "data" in payload) return payload.data;
      if (!("ok" in payload)) return payload;
    }
    return null;
  }

  // ── Container init ───────────────────────────────────────────────────
  function ensureContainer() {
    if (state.container) return state.container;
    let c = $("apin-toast-container");
    if (!c) {
      c = document.createElement("div");
      c.id = "apin-toast-container";
      c.className = "apin-toast-container";
      c.setAttribute("role", "region");
      c.setAttribute("aria-label", "Notifications");
      c.setAttribute("aria-live", "polite");
      document.body.appendChild(c);
    }
    state.container = c;
    return c;
  }

  // ── Action button rendering ──────────────────────────────────────────
  // Returns { html, handler } where handler runs on click.
  function buildAction(alert) {
    const details = (alert.details && typeof alert.details === "object")
      ? alert.details : {};
    const act = details.action || {};
    const kind = act.kind || "";
    const sev = alert.severity || "info";

    function link(label, href) {
      return {
        html: '<a class="apin-toast-action apin-toast-action-' + sev + '" '
            + 'href="' + escapeHTML(href) + '">'
            + escapeHTML(label)
            + ' <svg aria-hidden="true"><use href="#i-chevron-right"/></svg>'
            + '</a>',
        handler: null,  // navigation handled by browser
      };
    }
    function button(label, fn, opts) {
      const cls = "apin-toast-action apin-toast-action-button apin-toast-action-"
        + ((opts && opts.tone) || sev);
      return {
        html: '<button class="' + cls + '" data-toast-action="1">'
            + escapeHTML(label)
            + '</button>',
        handler: fn,
      };
    }

    if (kind === "view_key" && act.public_id) {
      return link("View key", "/account/api/keys/" + encodeURIComponent(act.public_id));
    }
    if (kind === "view_keys_list") {
      return link("View keys", "/account/api/keys");
    }
    if (kind === "view_webhook" && act.id) {
      return link("View webhook",
        "/account/api/webhooks#wh-" + encodeURIComponent(act.id));
    }
    if (kind === "view_webhooks_list") {
      return link("View webhooks", "/account/api/webhooks");
    }
    if (kind === "view_delivery" && act.webhook_id) {
      return link("View delivery",
        "/account/api/webhooks#delivery-" + encodeURIComponent(act.delivery_id || act.webhook_id));
    }
    if (kind === "re_enable_webhook" && act.id) {
      return link("Re-enable",
        "/account/api/webhooks#enable-" + encodeURIComponent(act.id));
    }
    if (kind === "view_request" && act.public_id) {
      return link("View request",
        "/account/api/keys/" + encodeURIComponent(act.public_id)
        + "#req-" + encodeURIComponent(act.request_id || ""));
    }
    if (kind === "view_settings") {
      return link("Open settings", "/account/api/settings");
    }
    if (kind === "adjust_quota") {
      return link("Adjust quota", "/account/api/settings#quotas");
    }
    if (kind === "extend_session") {
      return button("Extend session", function (toastId) {
        fetch("/auth/extend", { method: "POST", credentials: "same-origin" })
          .catch(function () {});
        dismiss(toastId, { reason: "engaged" });
      });
    }
    // No action ⇒ no button
    return null;
  }

  // ── Single-toast rendering ───────────────────────────────────────────
  function renderToastNode(alert) {
    const sev = alert.severity || "info";
    const iconRef = {
      info: "i-bulb",
      warn: "i-warning",
      critical: "i-alert",
    }[sev] || "i-bulb";

    const li = document.createElement("div");
    li.className = "apin-toast apin-toast-" + sev + " apin-toast-entering";
    li.setAttribute("role", sev === "critical" ? "alert" : "status");
    li.setAttribute("data-alert-id", String(alert.id));

    const act = buildAction(alert);
    let actionHTML = "";
    if (act) actionHTML =
      '<div class="apin-toast-actions">' + act.html + '</div>';

    li.innerHTML =
      '<span class="apin-toast-icon"><svg><use href="#' + iconRef + '"/></svg></span>'
      + '<div class="apin-toast-body">'
      + '<div class="apin-toast-title">' + escapeHTML(alert.title || "") + '</div>'
      + '<div class="apin-toast-text">' + escapeHTML(alert.body || "") + '</div>'
      + actionHTML
      + '</div>'
      + '<button class="apin-toast-close" type="button" aria-label="Dismiss">'
      + '<svg aria-hidden="true"><use href="#i-x"/></svg></button>';

    // Wire close
    li.querySelector(".apin-toast-close")
      .addEventListener("click", function () {
        dismiss(alert.id, { reason: "engaged" });
      });

    // Wire button-style action (only kind that needs a JS handler today)
    if (act && act.handler) {
      const btn = li.querySelector("[data-toast-action]");
      if (btn) btn.addEventListener("click", function () {
        act.handler(alert.id);
      });
    }
    // Link-style actions: clicking the link counts as "engaged"
    const linkAction = li.querySelector("a.apin-toast-action");
    if (linkAction) {
      linkAction.addEventListener("click", function () {
        // Don't preventDefault — let navigation happen — but mark read first.
        markRead(alert.id);
      });
    }

    // Frame-skip → remove entering class so transition kicks in
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        li.classList.remove("apin-toast-entering");
      });
    });
    return li;
  }

  // ── Pill (collapsed-stack) rendering ────────────────────────────────
  function ensurePill() {
    if (state.pillNode) return state.pillNode;
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "apin-toast-pill apin-toast-pill-entering";
    pill.setAttribute("aria-label", "View new alerts");
    pill.innerHTML =
      '<span class="apin-toast-pill-icon"><svg><use href="#i-alert"/></svg></span>'
      + '<span class="apin-toast-pill-count" id="apin-toast-pill-count">0</span>'
      + '<span class="apin-toast-pill-label">new alerts</span>'
      + '<span class="apin-toast-pill-close" role="button" tabindex="0" '
      + 'aria-label="Dismiss stack" data-pill-close="1">'
      + '<svg aria-hidden="true"><use href="#i-x"/></svg></span>';
    // Click body → go to alerts page
    pill.addEventListener("click", function (ev) {
      if (ev.target.closest("[data-pill-close]")) {
        ev.stopPropagation();
        dismissPill({ reason: "user_close" });
        return;
      }
      // Engaged → mark all pilled alerts read on click-through
      const ids = state.pillIds.slice();
      state.pillIds = [];
      ids.forEach(function (id) { markRead(id); });
      location.href = "/account/api/alerts";
    });
    ensureContainer().appendChild(pill);
    state.pillNode = pill;
    state.pillCount = 0;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        pill.classList.remove("apin-toast-pill-entering");
      });
    });
    return pill;
  }

  function bumpPill(alert) {
    const pill = ensurePill();
    state.pillCount += 1;
    state.pillIds.push(alert.id);
    const countEl = $("apin-toast-pill-count");
    if (countEl && window.APIN && window.APIN.odometer) {
      window.APIN.odometer.roll(countEl, state.pillCount);
    } else if (countEl) {
      countEl.textContent = String(state.pillCount);
    }
    // tiny "bump" pulse so the count clearly changed
    pill.classList.remove("apin-toast-pill-bump");
    void pill.offsetWidth;
    pill.classList.add("apin-toast-pill-bump");
  }

  function collapseVisibleIntoPill() {
    // Move every currently-visible toast into the pill (preserve unread state).
    // The user gets one consolidated indicator instead of a wall of cards.
    state.visible.forEach(function (entry) {
      state.pillIds.push(entry.id);
      if (entry.dismissTimer) clearTimeout(entry.dismissTimer);
      // Slide each out, then remove DOM.
      entry.node.classList.add("apin-toast-collapsing");
      setTimeout(function () { entry.node.remove(); }, SLIDE_OUT_MS);
    });
    const moved = state.visible.length;
    state.visible = [];
    // Update pill count with the moved toasts (we'll bump for the NEW alert
    // separately in the caller).
    if (moved > 0) {
      const pill = ensurePill();
      state.pillCount += moved;
      const countEl = $("apin-toast-pill-count");
      if (countEl && window.APIN && window.APIN.odometer) {
        window.APIN.odometer.roll(countEl, state.pillCount);
      } else if (countEl) {
        countEl.textContent = String(state.pillCount);
      }
      pill.classList.remove("apin-toast-pill-bump");
      void pill.offsetWidth;
      pill.classList.add("apin-toast-pill-bump");
    }
  }

  function dismissPill(opts) {
    const pill = state.pillNode;
    if (!pill) return;
    pill.classList.add("apin-toast-pill-leaving");
    setTimeout(function () {
      if (pill.parentNode) pill.parentNode.removeChild(pill);
      state.pillNode = null;
      state.pillCount = 0;
      state.pillIds = [];
    }, SLIDE_OUT_MS);
    // User-close on the pill is NOT engagement — alerts stay unread,
    // still in the badge, still on the alerts page.
  }

  // ── Show / dismiss single toasts ─────────────────────────────────────
  function showToast(alert) {
    if (!alert || !alert.id) return;
    // Already shown?
    if (state.visible.some(function (v) { return v.id === alert.id; })) return;
    // Skip if read or dismissed already
    if (alert.read_at || alert.dismissed_at) return;
    // At capacity → collapse into pill
    if (state.visible.length >= MAX_VISIBLE) {
      collapseVisibleIntoPill();
      bumpPill(alert);
      return;
    }
    // If a pill is already showing, new alerts join the pill (not the stack)
    if (state.pillNode) {
      bumpPill(alert);
      return;
    }
    const node = renderToastNode(alert);
    ensureContainer().appendChild(node);
    const entry = { id: alert.id, alert: alert, node: node, dismissTimer: null };
    state.visible.push(entry);

    const ttl = AUTO_DISMISS[alert.severity || "info"] || 0;
    if (ttl > 0) {
      entry.dismissTimer = setTimeout(function () {
        dismiss(alert.id, { reason: "timeout" });
      }, ttl);
    }
  }

  function dismiss(id, opts) {
    opts = opts || {};
    const idx = state.visible.findIndex(function (v) { return v.id === id; });
    if (idx < 0) return;
    const entry = state.visible[idx];
    state.visible.splice(idx, 1);
    if (entry.dismissTimer) clearTimeout(entry.dismissTimer);
    entry.node.classList.add("apin-toast-leaving");
    setTimeout(function () {
      if (entry.node.parentNode) entry.node.parentNode.removeChild(entry.node);
    }, SLIDE_OUT_MS);
    if (opts.reason === "engaged") {
      markRead(id);
    }
    // "timeout" reason leaves the alert UNREAD — user must engage to clear.
  }

  // ── /api/account/alerts/{id}/read ────────────────────────────────────
  function markRead(id) {
    fetch("/api/account/alerts/" + encodeURIComponent(id) + "/read", {
      method: "PATCH",
      credentials: "same-origin",
      headers: { "X-Console-Csrf": getCsrf() },
    })
      .then(function (r) {
        if (!r || !r.ok) return;
        // Same-tab listeners — BroadcastChannel does NOT echo to its own
        // sender, so we must dispatch a CustomEvent for in-tab refreshers
        // (nav badge, dashboard tile) to react.
        try {
          window.dispatchEvent(new CustomEvent("apin:alerts:changed",
            { detail: { type: "read", id: id } }));
        } catch (_) { /* defensive */ }
      })
      .catch(function () {});
    if (state.bc) {
      // Cross-tab — other tabs hear this via BroadcastChannel.
      state.bc.postMessage({ type: "read", id: id });
    }
  }

  // ── Polling ──────────────────────────────────────────────────────────
  async function bootCursor() {
    try {
      const r = await fetch("/api/account/alerts/unread-count?since=0", {
        credentials: "same-origin"
      });
      if (!r.ok) return;
      const j = await r.json();
      const d = unwrap(j) || {};
      // Bootstrap: assume the user has already "seen" history. New alerts
      // are only those with id > current latest_id at boot.
      state.cursor = d.latest_id || 0;
    } catch (_) { /* offline — leave cursor=0 */ }
  }

  async function poll() {
    if (document.hidden) return;
    try {
      const r = await fetch(
        "/api/account/alerts/unread-count?since=" + state.cursor,
        { credentials: "same-origin" }
      );
      if (!r.ok) return;
      const j = await r.json();
      const d = unwrap(j) || {};
      const latest = d.latest_id || 0;
      const newAlerts = (d.new_alerts || []).slice();
      if (latest > state.cursor) {
        state.cursor = latest;
        if (state.bc) {
          state.bc.postMessage({ type: "cursor_advance", latest_id: latest });
        }
        // Active tab only shows toasts. Background tabs just update cursor.
        if (document.hasFocus()) {
          newAlerts
            .filter(function (a) { return !a.read_at && !a.dismissed_at; })
            .forEach(showToast);
        }
      }
    } catch (_) { /* swallow */ }
  }

  function startPolling() {
    if (state.pollTimer) return;
    state.pollTimer = setInterval(poll, POLL_INTERVAL_MS);
  }
  function stopPolling() {
    if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  }

  // ── Cross-tab broadcast ──────────────────────────────────────────────
  function setupBroadcastChannel() {
    if (typeof BroadcastChannel === "undefined") return;
    try {
      state.bc = new BroadcastChannel("apin-alerts");
      state.bc.addEventListener("message", function (ev) {
        const msg = ev.data || {};
        if (msg.type === "cursor_advance" && msg.latest_id) {
          // Another tab observed alerts; advance our cursor so we don't
          // re-poll the same ids.
          if (msg.latest_id > state.cursor) state.cursor = msg.latest_id;
        } else if (msg.type === "read" && msg.id) {
          // Another tab dismissed this alert; hide our copy if visible.
          dismiss(msg.id, { reason: "remote" });
        }
      });
    } catch (_) { /* unavailable */ }
  }

  // ── Bridge for the chip session warning ──────────────────────────────
  // The chip's idle warning fires *client-side* and isn't backed by a DB
  // row. Expose a tiny entry so the chip can render a "session-expiring"
  // toast through the same renderer (matches the unified delivery model
  // the user asked for — modal stays for 5-min, toast handles 30-min).
  function showLocal(alert) {
    if (!alert || !alert.id) return;
    alert.id = alert.id || ("local-" + Date.now());
    showToast(alert);
  }

  // ── Init ─────────────────────────────────────────────────────────────
  function init() {
    if (!isConsolePage()) return;
    ensureContainer();
    bootCursor().then(function () {
      startPolling();
      // Fire one poll immediately on boot (in case alerts arrived between
      // bootCursor and the first interval tick).
      poll();
    });
    setupBroadcastChannel();
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) stopPolling(); else startPolling();
    });
  }

  // Expose for the chip (session.expiring_soon)
  window.APIN = window.APIN || {};
  window.APIN.toast = {
    showLocal: showLocal,
    markRead: markRead,
    cursor: function () { return state.cursor; },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
