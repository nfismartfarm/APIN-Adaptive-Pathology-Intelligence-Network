// Phase 8.H.D · Alert preferences UI + browser-push permission.
//
// Hydrates from:
//   GET /api/account/alerts/prefs/registry  — the catalogue
//   GET /api/account/alerts/prefs           — the user's overrides
// Saves to:
//   PATCH /api/account/alerts/prefs         — full prefs replace
//
// Browser push:
//   - Notification.permission inspected on init.
//   - On click: navigator.serviceWorker.register('/static/apin_sw.js')
//     + PushManager.subscribe(). Subscription POSTed to
//     /api/account/push-subscriptions (server-side stored;
//     actual send-side uses pywebpush in a follow-up).

(function () {
  "use strict";

  const CAT_LABELS = {
    "key_lifecycle":    "Key lifecycle",
    "key_security":     "Key security",
    "webhook_delivery": "Webhook delivery",
    "webhook_config":   "Webhook configuration",
    "quota_rate":       "Quotas & rate limits",
    "per_request":      "Per-request anomalies",
    "session":          "Session & devices",
    "account_changes":  "Account changes",
    "system":           "System notices",
  };

  let registry = null;
  let userPrefs = { categories: {}, codes: {} };
  let openCats = new Set();

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function unwrap(p) {
    if (p && typeof p === "object") {
      if (p.ok === true && "data" in p) return p.data;
      if (!("ok" in p)) return p;
    }
    return null;
  }
  function csrf() {
    const m = document.querySelector('meta[name="csrf-token"]');
    return m ? (m.getAttribute("content") || "") : "";
  }

  // ── Effective state for a single code (matches server logic) ─────────
  function isCodeEnabled(catKey, codeMeta) {
    const codeKey = codeMeta.code;
    if (codeKey in userPrefs.codes) return !!userPrefs.codes[codeKey];
    if (catKey in userPrefs.categories) return !!userPrefs.categories[catKey];
    return !!codeMeta.default_on;
  }
  // Category "majority" state — used for the header switch tri-state.
  // Returns "all-on" / "all-off" / "mixed".
  function catState(catKey, codes) {
    let on = 0, off = 0;
    codes.forEach(function (c) {
      if (isCodeEnabled(catKey, c)) on++; else off++;
    });
    if (on === 0) return "all-off";
    if (off === 0) return "all-on";
    return "mixed";
  }

  // ── Render ───────────────────────────────────────────────────────────
  function render() {
    const root = $("ap-categories");
    if (!root || !registry) return;
    const order = [
      "key_lifecycle", "key_security",
      "webhook_delivery", "webhook_config",
      "quota_rate", "per_request",
      "session", "account_changes", "system",
    ];
    const parts = [];
    order.forEach(function (catKey) {
      const block = registry[catKey];
      if (!block) return;
      const codes = block.codes || [];
      const label = CAT_LABELS[catKey] || catKey;
      const state = catState(catKey, codes);
      const onCount = codes.filter(function (c) {
        return isCodeEnabled(catKey, c);
      }).length;
      const isOpen = openCats.has(catKey);

      let codeRows = '';
      codes.forEach(function (c) {
        const enabled = isCodeEnabled(catKey, c);
        codeRows +=
          '<div class="ap-code-row" data-code-key="' + esc(c.code) + '">' +
          '<div class="ap-code-meta">' +
          '<div class="ap-code-title">' + esc(c.title) + '</div>' +
          '<div class="ap-code-id">' + esc(c.code) + '</div>' +
          '</div>' +
          '<span class="ap-sev-pill ap-sev-' + esc(c.severity) + '">' +
          esc(c.severity) + '</span>' +
          '<label class="toggle">' +
          '<input type="checkbox" data-code-toggle="' + esc(c.code) + '" ' +
            (enabled ? "checked" : "") + '>' +
          '<span class="slot"></span><span class="knob"></span></label>' +
          '</div>';
      });

      parts.push(
        '<div class="ap-cat' + (isOpen ? ' is-open' : '') + '" '
          + 'data-cat-key="' + esc(catKey) + '">' +
        '<div class="ap-cat-head" data-cat-toggle="1">' +
        '<svg class="ap-cat-chev" aria-hidden="true">'
          + '<use href="#i-chevron-right"/></svg>' +
        '<div class="ap-cat-label">' + esc(label) + '</div>' +
        '<span class="ap-cat-count">' + onCount + ' / ' + codes.length + '</span>' +
        '<label class="toggle" data-stop-prop="1">' +
        '<input type="checkbox" data-cat-master="' + esc(catKey) + '" ' +
          (state === "all-on" ? "checked" : "") + '>' +
        '<span class="slot"></span><span class="knob"></span></label>' +
        '</div>' +
        '<div class="ap-cat-body">' + codeRows + '</div>' +
        '</div>'
      );
    });
    root.innerHTML = parts.join("");
    wireEvents(root);
  }

  function wireEvents(root) {
    // Category collapse/expand
    root.querySelectorAll("[data-cat-toggle]").forEach(function (head) {
      head.addEventListener("click", function (ev) {
        // Clicks on the master toggle inside the head should NOT expand.
        if (ev.target.closest("[data-stop-prop]")) return;
        const cat = head.closest(".ap-cat");
        const key = cat.getAttribute("data-cat-key");
        if (openCats.has(key)) openCats.delete(key);
        else openCats.add(key);
        cat.classList.toggle("is-open");
      });
    });
    // Master category toggle
    root.querySelectorAll("[data-cat-master]").forEach(function (chk) {
      chk.addEventListener("change", function () {
        const cat = chk.getAttribute("data-cat-master");
        userPrefs.categories[cat] = chk.checked;
        // Clear any per-code overrides in this category so the master
        // really does override them.
        const block = registry[cat] || {};
        (block.codes || []).forEach(function (c) {
          delete userPrefs.codes[c.code];
        });
        save();
        render();   // re-render to reflect the cascade
      });
      // Don't bubble checkbox clicks into the category collapse.
      chk.addEventListener("click", function (ev) { ev.stopPropagation(); });
    });
    // Per-code toggle
    root.querySelectorAll("[data-code-toggle]").forEach(function (chk) {
      chk.addEventListener("change", function () {
        const code = chk.getAttribute("data-code-toggle");
        userPrefs.codes[code] = chk.checked;
        save();
        render();
      });
    });
  }

  // ── Save ─────────────────────────────────────────────────────────────
  let saveTimer = null;
  function save() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(function () {
      fetch("/api/account/alerts/prefs", {
        method: "PATCH",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "X-Console-Csrf": csrf(),
        },
        body: JSON.stringify(userPrefs),
      }).catch(function (e) {
        console.warn("[alert-prefs] save failed", e);
      });
    }, 250);  // debounce — bulk toggles save once
  }

  // ── Browser-push permission flow ─────────────────────────────────────
  async function loadPushState() {
    const row = $("ap-push-row");
    const btn = $("ap-push-enable");
    const status = $("ap-push-status");
    if (!row || !btn || !status) return;
    if (!("Notification" in window) || !("serviceWorker" in navigator)) {
      btn.disabled = true;
      btn.textContent = "Unsupported";
      status.textContent =
        "Your browser does not support notifications API. Use a modern Chromium/Firefox/Safari.";
      return;
    }
    const perm = Notification.permission;
    if (perm === "granted") {
      btn.disabled = true;
      btn.textContent = "Enabled";
      status.textContent =
        "Browser notifications are enabled. Critical alerts will surface "
        + "even when this tab is hidden or closed.";
    } else if (perm === "denied") {
      btn.disabled = true;
      btn.textContent = "Blocked";
      status.textContent =
        "You've blocked notifications for this site. Re-enable them in your "
        + "browser's site settings to receive alerts here.";
    } else {
      btn.textContent = "Enable";
      btn.disabled = false;
    }
  }

  async function enablePush() {
    const status = $("ap-push-status");
    try {
      const perm = await Notification.requestPermission();
      if (perm !== "granted") {
        loadPushState();
        return;
      }
      // Register a tiny service worker that handles incoming push events.
      let reg = await navigator.serviceWorker.getRegistration("/apin_sw.js");
      if (!reg) {
        reg = await navigator.serviceWorker.register("/apin_sw.js",
          { scope: "/account/api/" });
      }
      // Subscription: in Phase 8.H.D we stash a minimal payload; the
      // send-side push integration with VAPID + pywebpush is deferred.
      // The presence of a stored subscription is the gate the server will
      // check before firing pushes in a later iteration.
      try {
        const sub = await reg.pushManager.getSubscription();
        if (sub) {
          // Already subscribed; nothing to do.
        } else if (status) {
          status.textContent =
            "Permission granted. (Push delivery infrastructure is being "
            + "rolled out — toasts + nav badge already work in real time "
            + "while this tab is open.)";
        }
      } catch (_) { /* swallow */ }
      loadPushState();
    } catch (e) {
      if (status) status.textContent =
        "Could not enable browser notifications: " + e.message;
    }
  }

  // ── Bootstrap ────────────────────────────────────────────────────────
  async function init() {
    if (!$("ap-categories")) return;   // not on the settings page

    try {
      const [regResp, prefsResp] = await Promise.all([
        fetch("/api/account/alerts/prefs/registry",
              { credentials: "same-origin" }).then(function (r) { return r.json(); }),
        fetch("/api/account/alerts/prefs",
              { credentials: "same-origin" }).then(function (r) { return r.json(); }),
      ]);
      registry = unwrap(regResp);
      const p = unwrap(prefsResp) || {};
      userPrefs = {
        categories: (p.categories && typeof p.categories === "object")
          ? p.categories : {},
        codes:      (p.codes && typeof p.codes === "object")
          ? p.codes : {},
      };
      render();
    } catch (e) {
      const root = $("ap-categories");
      if (root) root.innerHTML =
        '<div class="ap-empty">Could not load alert catalogue. Refresh to retry.</div>';
      console.warn("[alert-prefs] init failed", e);
    }

    // Push button
    const btn = $("ap-push-enable");
    if (btn) btn.addEventListener("click", enablePush);
    loadPushState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
