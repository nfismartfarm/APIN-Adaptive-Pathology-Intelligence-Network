// Phase 8.G · Account chip + dropdown + session machinery (Console-only).
//
// Behaviour:
//   - On boot: fetch /auth/me, render avatar + name into the chip.
//   - Click chip: toggle dropdown open/close.
//   - Dropdown shows: identity strip (drifting leaves SVG + name + email),
//     cross-site jump (Inference site / API Console — highlighted = "here"),
//     Settings, Docs, Status, live session countdown, Extend, Sign out.
//   - Session: read expires_at from /auth/me (or /auth/session) and
//     count down. Idle detector (mousemove/keypress/scroll/click/touchstart)
//     resets a 48h idle timer. At expires_at − 5min, show warning modal.
//   - Cross-tab: BroadcastChannel('apin-auth') — if one tab logs out or
//     extends, all tabs sync.
//
// Loaded by console_nav.js (which is shipped on every Console page via the
// shared nav placeholder). This file lives at /static/console_account_chip.js.

(function () {
  "use strict";

  // ── DOM refs (resolved lazily — chip markup is injected by the nav) ──
  function $(id) { return document.getElementById(id); }

  // ── Tiny helpers ─────────────────────────────────────────────────────
  function escapeHTML(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fmtCountdown(msRemaining) {
    if (msRemaining <= 0) return "expired";
    var s = Math.floor(msRemaining / 1000);
    var d = Math.floor(s / 86400); s -= d * 86400;
    var h = Math.floor(s / 3600);  s -= h * 3600;
    var m = Math.floor(s / 60);
    if (d > 0) return d + "d " + h + "h";
    if (h > 0) return h + "h " + m + "m";
    if (m > 0) return m + "m";
    return Math.max(0, s) + "s";
  }

  // ── State ────────────────────────────────────────────────────────────
  var state = {
    user: null,
    expiresAt: null,       // ms epoch
    lastActivity: Date.now(),
    countdownTimer: null,
    idleWarningShown: false,
    bc: null,              // BroadcastChannel
  };

  // ── Render the chip itself (just avatar + name + caret) ──────────────
  function renderChip() {
    var avatarEl = $("apin-chip-avatar");
    var nameEl = $("apin-chip-name");
    if (!avatarEl || !nameEl) return;
    if (!state.user) {
      // Not signed in or /me errored. Render a sign-in link instead.
      var holder = $("apin-account-chip-holder");
      if (holder) {
        holder.innerHTML =
          '<a class="apin-chip apin-chip-signed-out" '
          + 'href="/dashboard?next=' + encodeURIComponent(location.pathname) + '">'
          + '<svg aria-hidden="true" style="width:14px;height:14px">'
          + '<use href="#i-user"/></svg>'
          + 'sign in</a>';
      }
      return;
    }
    var seed = (typeof state.user.pressed_leaf_seed === "number")
      ? state.user.pressed_leaf_seed : 0;
    if (window.APIN_pressedLeaf) {
      avatarEl.innerHTML = window.APIN_pressedLeaf(null, seed);
    } else {
      // pressed_leaf.js hasn't loaded yet — show user initial.
      var initial = (state.user.display_name || state.user.username || "?")
        .charAt(0).toUpperCase();
      avatarEl.innerHTML =
        '<span class="apin-chip-initial">' + escapeHTML(initial) + '</span>';
    }
    nameEl.textContent = state.user.display_name || state.user.username || "developer";
  }

  // ── Render the dropdown contents (identity strip + menu) ─────────────
  function renderDropdown() {
    var dd = $("apin-chip-dropdown");
    if (!dd || !state.user) return;
    var seed = (typeof state.user.pressed_leaf_seed === "number")
      ? state.user.pressed_leaf_seed : 0;
    var stripSVG = window.APIN_driftingLeavesStrip
      ? window.APIN_driftingLeavesStrip(seed)
      : '';
    var pathname = location.pathname;
    var onConsole = pathname.indexOf("/account/api") === 0;
    var displayName = escapeHTML(
      state.user.display_name || state.user.username || "developer");
    var email = escapeHTML(state.user.email || "");

    dd.innerHTML =
      '<div class="acd-strip">' + stripSVG + '</div>' +
      '<div class="acd-identity">' +
      '<div class="acd-name">' + displayName + '</div>' +
      (email ? '<div class="acd-email">' + email + '</div>' : '') +
      '</div>' +
      '<ul class="acd-menu" role="menu">' +
      '<li role="presentation"><a role="menuitem" href="/dashboard" '
      + 'class="acd-item' + (onConsole ? '' : ' acd-item-here') + '">'
      + '<span class="acd-icon"><svg><use href="#i-leaf"/></svg></span>'
      + '<span class="acd-label">Inference site</span>'
      + (onConsole ? '' : '<span class="acd-pill">here</span>')
      + '</a></li>' +
      '<li role="presentation"><a role="menuitem" href="/account/api/dashboard" '
      + 'class="acd-item' + (onConsole ? ' acd-item-here' : '') + '">'
      + '<span class="acd-icon"><svg><use href="#i-grid"/></svg></span>'
      + '<span class="acd-label">API Console</span>'
      + (onConsole ? '<span class="acd-pill">here</span>' : '')
      + '</a></li>' +
      '<li role="presentation" class="acd-sep"></li>' +
      '<li role="presentation"><a role="menuitem" href="/account/api/settings" class="acd-item">'
      + '<span class="acd-icon"><svg><use href="#i-settings"/></svg></span>'
      + '<span class="acd-label">Account settings</span></a></li>' +
      '<li role="presentation"><a role="menuitem" href="/docs" class="acd-item">'
      + '<span class="acd-icon"><svg><use href="#i-book"/></svg></span>'
      + '<span class="acd-label">API documentation</span></a></li>' +
      '<li role="presentation"><a role="menuitem" href="/status" class="acd-item">'
      + '<span class="acd-icon"><svg><use href="#i-activity"/></svg></span>'
      + '<span class="acd-label">System status</span></a></li>' +
      '<li role="presentation" class="acd-sep"></li>' +
      '<li role="presentation" class="acd-session">' +
      '<div class="acd-session-label">Session expires in</div>' +
      '<div class="acd-session-time" id="acd-countdown">&hellip;</div>' +
      '<button class="acd-extend-btn" id="acd-extend" type="button">' +
      '<svg aria-hidden="true"><use href="#i-refresh"/></svg> Extend session</button>' +
      '</li>' +
      '<li role="presentation" class="acd-sep"></li>' +
      '<li role="presentation"><button role="menuitem" type="button" id="acd-signout" '
      + 'class="acd-item acd-item-danger">'
      + '<span class="acd-icon"><svg><use href="#i-log-out"/></svg></span>'
      + '<span class="acd-label">Sign out</span></button></li>' +
      '</ul>';

    // Wire dropdown buttons
    var ext = $("acd-extend");
    if (ext) ext.addEventListener("click", extendSession);
    var so = $("acd-signout");
    if (so) so.addEventListener("click", signOut);

    // Admin-only: surface an "Admin console" entry (gated by whoami).
    ensureAdminLink(dd);
  }

  // ── Admin-console menu entry (only for admins) ───────────────────────
  // whoami is checked once and cached. A non-admin gets is_admin:false and
  // no link is injected; an admin gets the entry slotted after "API Console".
  var _adminChecked = false, _isAdmin = false;
  function injectAdminLink(dd) {
    if (!_isAdmin || !dd) return;
    var menu = dd.querySelector(".acd-menu");
    if (!menu || menu.querySelector(".acd-admin")) return;
    var consoleLink = menu.querySelector('a[href="/account/api/dashboard"]');
    var anchorLi = consoleLink ? consoleLink.parentNode : null;
    var li = document.createElement("li");
    li.setAttribute("role", "presentation");
    li.innerHTML =
      '<a role="menuitem" href="/account/api/admin" class="acd-item acd-admin">'
      + '<span class="acd-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
      + 'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
      + '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/>'
      + '<path d="M9.5 12l2 2 3.5-4"/></svg></span>'
      + '<span class="acd-label">Admin console</span></a>';
    if (anchorLi && anchorLi.nextSibling) menu.insertBefore(li, anchorLi.nextSibling);
    else menu.appendChild(li);
    // Intercept the click so we play the cinematic leaf-morph + flag the arrival,
    // matching the login → admin transition (the morph is otherwise IIFE-scoped
    // to the home page, so we ship a compact self-contained copy here).
    var a = li.querySelector("a.acd-admin");
    if (a) a.addEventListener("click", function (e) {
      e.preventDefault();
      try { sessionStorage.setItem("apin_admin_arrival", "1"); } catch (_) {}
      adminLeafMorph(function () { window.location.href = "/account/api/admin"; });
    });
  }

  // Compact cream→black leaf-flip morph (mirrors playAdminLeafMorph on the home
  // page). Self-contained: injects its own keyframes + overlay, then calls done.
  function adminLeafMorph(done) {
    var reduced = false;
    try { reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) {}
    var finished = false;
    function finish() { if (finished) return; finished = true; try { done(); } catch (e) { window.location.href = "/account/api/admin"; } }
    if (!document.getElementById("apin-morph-style")) {
      var st = document.createElement("style"); st.id = "apin-morph-style";
      st.textContent =
        "@keyframes apinMorphBg{0%{background:#fbf9f3}50%{background:#2c2a22}100%{background:#08080a}}"
        + "@keyframes apinMorphFade{to{opacity:1}}"
        + ".apin-morph{position:fixed;inset:0;z-index:99999;display:grid;place-items:center;background:#fbf9f3;overflow:hidden;animation:apinMorphBg 1.45s cubic-bezier(.7,0,.2,1) forwards}"
        + ".apin-morph .lw{perspective:1400px;width:200px;height:200px}"
        + ".apin-morph .l3{position:relative;width:100%;height:100%;transform-style:preserve-3d;transition:transform 1.35s cubic-bezier(.66,0,.2,1)}"
        + ".apin-morph .l3.flip{transform:rotateX(180deg)}"
        + ".apin-morph .lf{position:absolute;inset:0;backface-visibility:hidden;-webkit-backface-visibility:hidden;display:grid;place-items:center}"
        + ".apin-morph .lb{transform:rotateX(180deg)}"
        + ".apin-morph .lb svg{filter:drop-shadow(0 0 14px rgba(74,222,128,.45))}"
        + ".apin-morph .cap{position:absolute;bottom:17%;left:0;right:0;text-align:center;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:11.5px;letter-spacing:.22em;color:#9b9ba4;opacity:0;animation:apinMorphFade 1.2s ease .55s forwards}"
        + "@media (prefers-reduced-motion: reduce){.apin-morph{animation-duration:.25s}.apin-morph .l3{transition:none}}";
      document.head.appendChild(st);
    }
    var FRONT = '<svg viewBox="0 0 120 120" width="192" height="192" aria-hidden="true">'
      + '<path d="M60 8C30 35 28 80 60 112C92 80 90 35 60 8Z" fill="#dbe8d0" stroke="#2f6f3e" stroke-width="2.6" stroke-linejoin="round"/>'
      + '<path d="M60 14V106" stroke="#2f6f3e" stroke-width="2" fill="none"/></svg>';
    var BACK = '<svg viewBox="0 0 120 120" width="192" height="192" aria-hidden="true">'
      + '<path d="M60 8C30 35 28 80 60 112C92 80 90 35 60 8Z" fill="#0e1a12" stroke="#4ade80" stroke-width="2.6" stroke-linejoin="round"/>'
      + '<path d="M60 14V106" stroke="#4ade80" stroke-width="2" fill="none"/></svg>';
    var ov = document.createElement("div"); ov.className = "apin-morph";
    ov.innerHTML = '<div class="lw"><div class="l3" id="apin-leaf"><div class="lf">' + FRONT
      + '</div><div class="lf lb">' + BACK + '</div></div></div><div class="cap">elevating · admin console</div>';
    ov.addEventListener("click", finish);
    document.body.appendChild(ov);
    if (reduced) { setTimeout(finish, 280); return; }
    requestAnimationFrame(function () { requestAnimationFrame(function () {
      var l = document.getElementById("apin-leaf"); if (l) l.classList.add("flip");
    }); });
    setTimeout(finish, 1720);
  }
  function ensureAdminLink(dd) {
    if (_adminChecked) { injectAdminLink(dd); return; }
    _adminChecked = true;
    fetch("/api/account/admin/whoami",
          { credentials: "same-origin", headers: { "Accept": "application/json" } })
      .then(function (r) { return r.json(); })
      .then(function (j) { _isAdmin = !!(j && j.data && j.data.is_admin); injectAdminLink(dd); })
      .catch(function () {});
  }

  // ── Dropdown open/close ──────────────────────────────────────────────
  function openDropdown() {
    var dd = $("apin-chip-dropdown");
    var btn = $("apin-chip-button");
    if (!dd || !btn) return;
    renderDropdown();
    dd.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    refreshCountdown();   // immediate render
    // Outside-click + Esc handlers
    setTimeout(function () {
      document.addEventListener("click", outsideClick, true);
      document.addEventListener("keydown", escKey);
    }, 0);
  }
  function closeDropdown() {
    var dd = $("apin-chip-dropdown");
    var btn = $("apin-chip-button");
    if (!dd || !btn) return;
    dd.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    document.removeEventListener("click", outsideClick, true);
    document.removeEventListener("keydown", escKey);
  }
  function toggleDropdown(e) {
    if (e) e.stopPropagation();
    var dd = $("apin-chip-dropdown");
    if (dd && dd.hidden) openDropdown(); else closeDropdown();
  }
  function outsideClick(e) {
    var dd = $("apin-chip-dropdown");
    var btn = $("apin-chip-button");
    if (!dd || !btn) return;
    if (dd.contains(e.target) || btn.contains(e.target)) return;
    closeDropdown();
  }
  function escKey(e) { if (e.key === "Escape") closeDropdown(); }

  // ── Session countdown ────────────────────────────────────────────────
  function refreshCountdown() {
    var el = $("acd-countdown");
    if (!el || !state.expiresAt) return;
    var remaining = state.expiresAt - Date.now();
    el.textContent = fmtCountdown(remaining);
  }

  function startCountdownTimer() {
    if (state.countdownTimer) clearInterval(state.countdownTimer);
    state.countdownTimer = setInterval(function () {
      refreshCountdown();
      checkExpiryWarning();
    }, 30 * 1000);  // tick every 30 s — countdown granularity is minutes
  }

  // ── Idle detection + warning modal ───────────────────────────────────
  // "Idle" means no user input in the last IDLE_THRESHOLD ms.
  // Warning fires at expires_at − WARN_LEAD_MS, but ONLY if the user has
  // been idle for at least IDLE_THRESHOLD. Active users get sliding
  // renewal via /auth/extend (called on visibility change + a debounced
  // ping when input is detected after long quiet).
  var IDLE_THRESHOLD = 48 * 60 * 60 * 1000;     // 48 hours
  var WARN_LEAD_MS = 5 * 60 * 1000;             // 5 minutes (modal)
  var SOFT_WARN_LEAD_MS = 30 * 60 * 1000;       // 30 minutes (toast)

  function recordActivity() {
    state.lastActivity = Date.now();
    // If we already showed the warning AND the user is now interacting,
    // dismiss it (treat input as "extend").
    if (state.idleWarningShown) {
      hideExpiryWarning();
      extendSession();
    }
  }

  function checkExpiryWarning() {
    if (!state.expiresAt) return;
    var now = Date.now();
    var msUntilExpiry = state.expiresAt - now;
    var msSinceActivity = now - state.lastActivity;

    // ── Soft toast at T-30min (Phase 8.H · session.expiring_soon) ───
    // Fires once per session, regardless of idle state. Non-blocking.
    // The modal still fires later at T-5min if the user goes idle.
    if (!state.softWarnShown
        && msUntilExpiry > 0
        && msUntilExpiry <= SOFT_WARN_LEAD_MS
        && msUntilExpiry > WARN_LEAD_MS) {
      state.softWarnShown = true;
      if (window.APIN && window.APIN.toast) {
        var mins = Math.max(1, Math.round(msUntilExpiry / 60000));
        window.APIN.toast.showLocal({
          id: "session-expiring-" + state.expiresAt,
          severity: "warn",
          code: "session.expiring_soon",
          title: "Session expires soon",
          body: "Your session ends in about " + mins + " minutes. "
              + "Click Extend to keep working.",
          details: { action: { kind: "extend_session" } },
        });
      }
    }

    if (state.idleWarningShown) return;
    if (msUntilExpiry > WARN_LEAD_MS) return;
    if (msUntilExpiry <= 0) {
      // Already expired — show the hard "session ended" modal instead.
      showSessionEndedModal();
      return;
    }
    // Modal only fires when user has been idle long enough.
    if (msSinceActivity < IDLE_THRESHOLD) return;
    showExpiryWarning();
  }

  function attachIdleListeners() {
    var events = ["mousemove", "keypress", "scroll", "click", "touchstart"];
    var debouncer = null;
    var handler = function () {
      // Debounce — input events fire many times per second.
      if (debouncer) return;
      debouncer = setTimeout(function () { debouncer = null; }, 1000);
      recordActivity();
    };
    events.forEach(function (ev) {
      document.addEventListener(ev, handler, { passive: true });
    });
  }

  // ── Warning + ended modals ───────────────────────────────────────────
  function ensureModalMount() {
    var mount = $("apin-session-modal-root");
    if (mount) return mount;
    mount = document.createElement("div");
    mount.id = "apin-session-modal-root";
    document.body.appendChild(mount);
    return mount;
  }

  function showExpiryWarning() {
    state.idleWarningShown = true;
    var mount = ensureModalMount();
    var remaining = Math.max(0, state.expiresAt - Date.now());
    mount.innerHTML =
      '<div class="apin-modal-backdrop" id="apin-warn-backdrop" role="dialog" '
      + 'aria-modal="true" aria-labelledby="apin-warn-title">' +
      '<div class="apin-modal-card">' +
      '<span class="apin-modal-icon"><svg><use href="#i-warning"/></svg></span>' +
      '<h2 id="apin-warn-title">Are you still there?</h2>' +
      '<p>Your session will expire in <strong id="apin-warn-time">'
      + fmtCountdown(remaining) + '</strong> due to inactivity.</p>' +
      '<div class="apin-modal-actions">' +
      '<button type="button" class="apin-btn apin-btn-secondary" '
      + 'id="apin-warn-logout">Sign out now</button>' +
      '<button type="button" class="apin-btn apin-btn-primary" '
      + 'id="apin-warn-stay">Stay signed in</button>' +
      '</div></div></div>';
    var stay = $("apin-warn-stay");
    var logout = $("apin-warn-logout");
    if (stay) stay.addEventListener("click", function () {
      hideExpiryWarning();
      extendSession();
    });
    if (logout) logout.addEventListener("click", signOut);
    // Live update the countdown inside the modal.
    var liveTimer = setInterval(function () {
      var t = $("apin-warn-time");
      if (!t) { clearInterval(liveTimer); return; }
      var r = Math.max(0, state.expiresAt - Date.now());
      t.textContent = fmtCountdown(r);
      if (r <= 0) {
        clearInterval(liveTimer);
        hideExpiryWarning();
        showSessionEndedModal();
      }
    }, 1000);
  }

  function hideExpiryWarning() {
    state.idleWarningShown = false;
    var bd = $("apin-warn-backdrop");
    if (bd && bd.parentNode) bd.parentNode.removeChild(bd);
  }

  function showSessionEndedModal() {
    var mount = ensureModalMount();
    mount.innerHTML =
      '<div class="apin-modal-backdrop" role="dialog" aria-modal="true" '
      + 'aria-labelledby="apin-ended-title">' +
      '<div class="apin-modal-card">' +
      '<span class="apin-modal-icon"><svg><use href="#i-lock"/></svg></span>' +
      '<h2 id="apin-ended-title">Session ended</h2>' +
      '<p>For your security we signed you out after a long period of '
      + 'inactivity. Sign back in to continue.</p>' +
      '<div class="apin-modal-actions">' +
      '<a class="apin-btn apin-btn-primary" '
      + 'href="/dashboard?next=' + encodeURIComponent(location.pathname) + '">'
      + 'Sign in</a>' +
      '</div></div></div>';
  }

  // ── /auth/me + /auth/extend + /auth/logout ───────────────
  function loadMe() {
    return fetch("/auth/me", { credentials: "same-origin" })
      .then(function (r) {
        if (r.status === 401) {
          state.user = null;
          renderChip();
          return null;
        }
        if (!r.ok) throw new Error("me " + r.status);
        return r.json();
      })
      .then(function (data) {
        if (!data) return;
        state.user = data;
        // expires_at lives on the session row, not /me — try to read it
        // from a separate endpoint, fall back to "last_seen + 7d" if absent.
        return fetch("/auth/session", { credentials: "same-origin" })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (sess) {
            if (sess && sess.expires_at) {
              state.expiresAt = new Date(sess.expires_at).getTime();
            } else {
              // Fall back to last_seen_at + 7d so we have *something*.
              var ls = data.last_seen_at
                ? new Date(data.last_seen_at).getTime()
                : Date.now();
              state.expiresAt = ls + 7 * 24 * 3600 * 1000;
            }
            renderChip();
          });
      })
      .catch(function (err) {
        console.warn("[chip] /me failed", err);
        state.user = null;
        renderChip();
      });
  }

  function extendSession() {
    return fetch("/auth/extend", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
    })
      .then(function (r) {
        if (!r.ok) return null;
        return r.json();
      })
      .then(function (data) {
        if (data && data.expires_at) {
          state.expiresAt = new Date(data.expires_at).getTime();
          // Allow a new soft toast on the next 30-min window.
          state.softWarnShown = false;
          refreshCountdown();
          state.lastActivity = Date.now();
          if (state.bc) state.bc.postMessage({ type: "extended", expires_at: data.expires_at });
        }
      })
      .catch(function (e) { console.warn("[chip] extend failed", e); });
  }

  function signOut() {
    fetch("/auth/logout", { method: "POST", credentials: "same-origin" })
      .catch(function () {})
      .then(function () {
        if (state.bc) state.bc.postMessage({ type: "logout" });
        location.href = "/dashboard";
      });
  }

  // ── Cross-tab broadcast ──────────────────────────────────────────────
  function setupBroadcastChannel() {
    if (typeof BroadcastChannel === "undefined") return;
    try {
      state.bc = new BroadcastChannel("apin-auth");
      state.bc.addEventListener("message", function (ev) {
        var msg = ev.data || {};
        if (msg.type === "logout") {
          // Another tab signed out — show the ended modal here too.
          showSessionEndedModal();
        } else if (msg.type === "extended" && msg.expires_at) {
          state.expiresAt = new Date(msg.expires_at).getTime();
          refreshCountdown();
          state.lastActivity = Date.now();
          hideExpiryWarning();
        }
      });
    } catch (e) { /* unavailable in older browsers */ }
  }

  // ── Boot ─────────────────────────────────────────────────────────────
  function init() {
    var btn = $("apin-chip-button");
    if (btn) btn.addEventListener("click", toggleDropdown);
    loadMe().then(function () {
      startCountdownTimer();
      attachIdleListeners();
      setupBroadcastChannel();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
