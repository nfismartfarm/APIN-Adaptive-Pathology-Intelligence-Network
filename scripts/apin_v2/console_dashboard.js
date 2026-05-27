// Phase 8.F · Console Dashboard widgets
// Pulls live data from existing endpoints — no new server work for stats:
//   GET /auth/me                       → username, predictions_count,
//                                            created_at, last_seen_at
//   GET /api/account/keys                  → keys list (count, active count)
//   GET /api/account/alerts/unread-count   → unread alert count
//   GET /api/account/alerts                → recent alerts (for tile sub)
//   GET /api/account/webhooks              → webhooks list
//   GET /api/account/audit/recent          → recent audit events (best-effort)
//   GET /health                            → API + DB pip
//
// Render strategy:
//   1. Fire all requests in parallel.
//   2. Update each section independently as its data lands (progressive).
//   3. On error per-section: leave placeholder text, do not block other tiles.
//   4. Re-poll alerts every 30s while the page is visible (cheap).

(function () {
  "use strict";

  // ───────────────────────────────────────────────────────────────
  //  Tiny helpers
  // ───────────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }

  function setText(id, value) {
    var el = $(id);
    if (el) el.textContent = value;
  }

  function setHTML(id, html) {
    var el = $(id);
    if (el) el.innerHTML = html;
  }

  function fmtRelativeTime(iso) {
    if (!iso) return "";
    try {
      var t = new Date(iso).getTime();
      if (isNaN(t)) return "";
      var diff = Math.floor((Date.now() - t) / 1000);
      if (diff < 0) diff = 0;
      if (diff < 60) return diff + "s ago";
      if (diff < 3600) return Math.floor(diff / 60) + "m ago";
      if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
      if (diff < 604800) return Math.floor(diff / 86400) + "d ago";
      // Beyond a week: drop to date
      var d = new Date(t);
      return d.toISOString().slice(0, 10);
    } catch (e) {
      return "";
    }
  }

  function fmtAbsDate(iso) {
    if (!iso) return "";
    try {
      var d = new Date(iso);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleDateString(undefined, {
        year: "numeric", month: "short", day: "numeric"
      });
    } catch (e) {
      return "";
    }
  }

  function escapeHTML(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Most APIN endpoints return the §3 9-key envelope; extract `data` for ok=true.
  function unwrap(payload) {
    if (payload && typeof payload === "object") {
      if (payload.ok === true && "data" in payload) return payload.data;
      // Some legacy endpoints return data straight (e.g. /auth/me).
      if (!("ok" in payload)) return payload;
    }
    return null;
  }

  function fetchJSON(url) {
    return fetch(url, { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) {
          // 401 on /me means logged out — the nav redirect handles UX.
          if (r.status === 401) return null;
          throw new Error(url + " " + r.status);
        }
        return r.json();
      })
      .catch(function (err) {
        console.warn("[dashboard] fetch failed", url, err);
        return null;
      });
  }

  // ───────────────────────────────────────────────────────────────
  //  Section: hero (user profile)
  // ───────────────────────────────────────────────────────────────
  function renderHero(me) {
    if (!me) {
      setText("dash-username", "guest");
      setText("dash-greeting",
        "You appear to be signed out. " +
        "Please log in to use the API Console.");
      setText("dash-pred-count", "");
      setText("dash-member-since", "");
      setText("dash-last-seen", "");
      return;
    }

    var name = me.display_name || me.username || "developer";
    setText("dash-username", name);

    // Time-of-day greeting (client-local) — Fraunces italic, conversational
    var hour = new Date().getHours();
    var tod  = "evening";
    if (hour < 5) tod = "late night";
    else if (hour < 12) tod = "morning";
    else if (hour < 17) tod = "afternoon";
    var pred = (typeof me.predictions_count === "number") ?
               me.predictions_count : null;
    var msg = "Good " + tod + ". " +
              (pred !== null && pred > 0
                ? "You have run " + pred.toLocaleString() +
                  (pred === 1 ? " prediction" : " predictions") +
                  " through APIN."
                : "Ready to build with APIN?");
    setText("dash-greeting", msg);

    setText("dash-pred-count",
      (pred !== null ? pred.toLocaleString() + " predictions" : "predictions: —"));
    setText("dash-member-since",
      me.created_at ? "joined " + fmtAbsDate(me.created_at) : "");
    setText("dash-last-seen",
      me.last_seen_at ? "last seen " + fmtRelativeTime(me.last_seen_at) : "");
  }

  // ───────────────────────────────────────────────────────────────
  //  Section: stat tiles — odometer-animated counters
  // ───────────────────────────────────────────────────────────────
  // Phase 8.H · numerical values flow through the shared odometer
  // (window.APIN.odometer.roll). Each tile's `.value` span is treated as
  // an odometer container; the apin-odometer class is auto-applied on
  // first roll. setNumber() handles both numeric values and placeholder
  // strings ("—") cleanly: numeric → odometer, non-numeric → plain text.
  function setNumber(id, value) {
    var el = $(id);
    if (!el) return;
    if (typeof value === "number" && Number.isFinite(value)) {
      if (!el.classList.contains("apin-odometer")) {
        el.classList.add("apin-odometer");
        el.textContent = "";       // clear placeholder so seed builds clean
        el._odo = null;
      }
      if (window.APIN && window.APIN.odometer) {
        window.APIN.odometer.roll(el, value);
      } else {
        el.textContent = String(value);
      }
    } else {
      // Non-numeric (placeholder, "—", error state) — reset to plain.
      el.classList.remove("apin-odometer");
      el._odo = null;
      el.textContent = String(value == null ? "" : value);
    }
  }

  function renderKeysTile(payload) {
    var data = unwrap(payload);
    if (!data) {
      setNumber("stat-keys", "—");
      setText("stat-keys-sub", "could not load");
      return;
    }
    // payload shape: { keys: [...], count: N } or array
    var keys = Array.isArray(data) ? data : (data.keys || data.items || []);
    var total = keys.length;
    var active = 0;
    var live = 0;
    keys.forEach(function (k) {
      if (k.status === "active" || (!k.status && !k.disabled_at)) active++;
      if (k.environment === "live") live++;
    });
    setNumber("stat-keys", total);
    var subText = total === 0
      ? "no keys yet — create one to start"
      : active + " active · " + live + " live · " + (total - live) + " test";
    setText("stat-keys-sub", subText);
  }

  function renderAlertsTile(payload) {
    var data = unwrap(payload);
    if (!data) {
      setNumber("stat-alerts", "—");
      setText("stat-alerts-sub", "could not load");
      return;
    }
    // Phase 8.H fix · the real shape is {unread: N}, not unread_count.
    // Earlier code read the wrong key and the tile always showed 0.
    var n = (typeof data.unread === "number") ? data.unread
          : (typeof data.unread_count === "number") ? data.unread_count
          : (typeof data.count === "number") ? data.count : 0;
    setNumber("stat-alerts", n);
    var tile = $("tile-alerts");
    if (tile) {
      // Border colour reflects unread state — clear it when count drops
      // back to 0 (the old code never cleared it).
      tile.style.borderColor = n > 0 ? "var(--accent-crimson)" : "";
    }
    setText("stat-alerts-sub", n > 0 ? "needs attention" : "all caught up");
  }

  function renderWebhooksTile(payload) {
    var data = unwrap(payload);
    if (!data) {
      setNumber("stat-webhooks", "—");
      setText("stat-webhooks-sub", "could not load");
      return;
    }
    var hooks = Array.isArray(data) ? data : (data.webhooks || data.items || []);
    var total = hooks.length;
    var enabled = 0;
    hooks.forEach(function (w) {
      if (w.enabled !== false && !w.disabled_at) enabled++;
    });
    setNumber("stat-webhooks", total);
    setText("stat-webhooks-sub", total === 0
      ? "none configured"
      : enabled + " active");
  }

  // ───────────────────────────────────────────────────────────────
  //  Section: activity feed
  // ───────────────────────────────────────────────────────────────
  function activityIconClass(action) {
    if (!action) return "";
    action = String(action).toLowerCase();
    if (action.indexOf("create") >= 0) return "create";
    if (action.indexOf("rotate") >= 0) return "rotate";
    if (action.indexOf("disable") >= 0 || action.indexOf("delete") >= 0)
      return "disable";
    if (action.indexOf("update") >= 0 || action.indexOf("edit") >= 0)
      return "update";
    return "";
  }

  function activityIconRef(action) {
    // Hand-drawn icon ID for a given audit action (Phase 8.G — was ASCII).
    // Falls back to i-help-circle for unrecognized actions.
    if (!action) return "i-help-circle";
    action = String(action).toLowerCase();
    if (action.indexOf("create") >= 0) return "i-plus";
    if (action.indexOf("rotate") >= 0) return "i-refresh";
    if (action.indexOf("disable") >= 0) return "i-pause";
    if (action.indexOf("delete") >= 0) return "i-trash";
    if (action.indexOf("update") >= 0 || action.indexOf("edit") >= 0) return "i-pencil";
    if (action.indexOf("enable") >= 0) return "i-check";
    return "i-help-circle";
  }

  function humanizeAction(action) {
    if (!action) return "Unknown event";
    return String(action)
      .replace(/^api_key_/, "key ")
      .replace(/^webhook_/, "webhook ")
      .replace(/^settings_/, "settings ")
      .replace(/_/g, " ");
  }

  function renderActivity(payload) {
    var loadEl = $("activity-loading");
    if (loadEl) loadEl.remove();
    var feed = $("activity-feed");
    if (!feed) return;
    var data = unwrap(payload);
    var events = [];
    if (data) {
      events = data.events || data.audit || data.items || data || [];
      if (!Array.isArray(events)) events = [];
    }
    if (events.length === 0) {
      feed.innerHTML = '<li class="activity-empty">' +
        "No activity yet. Create an API key, configure a webhook, or " +
        "tweak settings to see events appear here." +
        "</li>";
      return;
    }
    var html = events.slice(0, 10).map(function (ev) {
      var action = ev.action || ev.event || "";
      var pid = ev.key_id || ev.public_id || ev.target_id || "";
      var when = ev.at || ev.timestamp || ev.created_at || ev.row_at || "";
      var note = ev.note || ev.detail || "";
      var iconCls = activityIconClass(action);
      var iconRef = activityIconRef(action);
      var titleHTML = '<span>' + escapeHTML(humanizeAction(action)) + '</span>';
      if (pid) {
        titleHTML += ' <span style="color:var(--ink-soft);' +
          'font-family:\'JetBrains Mono\',monospace;font-size:12px">' +
          escapeHTML(String(pid).slice(0, 14)) +
          (String(pid).length > 14 ? '…' : '') + '</span>';
      }
      var metaHTML = note ? escapeHTML(note) : "";
      // Build the icon SVG via DOM so attribute escaping is correct.
      // (innerHTML below pastes pre-escaped strings; the SVG uses a fixed
      // template literal interpolating only the icon id from a closed list.)
      return '<li class="activity-item">' +
        '<span class="activity-icon ' + iconCls + '" aria-hidden="true">' +
        '<svg><use href="#' + iconRef + '"/></svg>' +
        '</span>' +
        '<div class="activity-body">' +
        '<div class="title">' + titleHTML + '</div>' +
        (metaHTML ? '<div class="meta">' + metaHTML + '</div>' : '') +
        '</div>' +
        '<span class="activity-time" title="' +
        escapeHTML(when) + '">' +
        escapeHTML(fmtRelativeTime(when)) + '</span>' +
        '</li>';
    }).join("");
    feed.innerHTML = html;

    // Phase 8.H fix · removed the deep-link override.
    // The HTML now hardcodes href="/account/api/keys" — a destination
    // that always exists. Previously this code rewrote the href to point
    // at the most recent event's key (e.g. .../keys/k_abc123#audit), but
    // when that key had been deleted (a key.deleted audit event still
    // references the dead public_id) the link was a 404 trap. The keys
    // list page is the right consolidated destination because users can
    // open any key from there to see its hash-chained audit panel.
  }

  // ───────────────────────────────────────────────────────────────
  //  Section: system status row
  // ───────────────────────────────────────────────────────────────
  function setPip(id, label, level) {
    var el = $(id);
    if (!el) return;
    el.textContent = label;
    el.className = "pip" + (level && level !== "ok" ? " " + level : "");
  }

  function renderHealth(payload) {
    var data = (payload && payload.data) ? payload.data : payload;
    if (!data) {
      setPip("sys-api", "API: unknown", "unknown");
      setPip("sys-worker", "Worker: unknown", "unknown");
      setPip("sys-db", "Database: unknown", "unknown");
      setText("sys-version", "v2");
      return;
    }
    // /health typically returns { status, api_version, db, worker, ... }
    var dbState = data.db || data.database || (data.checks && data.checks.db);
    var workerState = data.worker || data.webhook_worker ||
                      (data.checks && data.checks.worker);
    var apiState = data.status || (data.checks && data.checks.api) || "ok";

    function pipLevel(state) {
      if (!state) return "unknown";
      state = String(state).toLowerCase();
      if (state === "ok" || state === "healthy" || state === "running" ||
          state === "ready" || state === "up") return "ok";
      if (state === "warn" || state === "degraded") return "warn";
      return "err";
    }

    setPip("sys-api", "API: " + (apiState || "ok"), pipLevel(apiState));
    setPip("sys-worker", "Webhook worker: " +
      (workerState || "idle"), pipLevel(workerState));
    setPip("sys-db", "Database: " + (dbState || "ok"), pipLevel(dbState));
    setText("sys-version", "v" + (data.api_version || "2") +
      (data.build ? " · " + data.build : ""));
  }

  // ───────────────────────────────────────────────────────────────
  //  Boot
  // ───────────────────────────────────────────────────────────────
  function loadDashboard() {
    fetchJSON("/auth/me").then(renderHero);
    fetchJSON("/api/account/keys").then(renderKeysTile);
    fetchJSON("/api/account/alerts/unread-count").then(renderAlertsTile);
    fetchJSON("/api/account/webhooks").then(renderWebhooksTile);
    fetchJSON("/api/account/audit/recent").then(renderActivity);
    fetchJSON("/health").then(renderHealth);
  }

  // Refresh the alerts tile every 30s while the page is visible.
  var alertsTimer = null;
  function refreshAlertsTile() {
    if (document.hidden) return;
    fetchJSON("/api/account/alerts/unread-count").then(renderAlertsTile);
  }
  function startAlertsPolling() {
    if (alertsTimer) return;
    alertsTimer = setInterval(refreshAlertsTile, 30000);
  }
  function stopAlertsPolling() {
    if (alertsTimer) { clearInterval(alertsTimer); alertsTimer = null; }
  }
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      stopAlertsPolling();
    } else {
      startAlertsPolling();
      refreshAlertsTile();   // immediate refresh on focus
    }
  });
  // Phase 8.H · same-tab event from the toast layer when an alert is
  // marked read. BroadcastChannel doesn't echo to its own sender, so
  // without this listener the tile lagged the badge.
  window.addEventListener("apin:alerts:changed", refreshAlertsTile);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      loadDashboard();
      startAlertsPolling();
    });
  } else {
    loadDashboard();
    startAlertsPolling();
  }
})();
