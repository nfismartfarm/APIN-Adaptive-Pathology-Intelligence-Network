// scripts/apin_v2/keys.js
// API Console — keys list page client.
//
// Extracted from inline <script> in keys.html as part of Phase 3.1 R1 fix
// bundle FX-A: CSP cannot allow inline scripts without per-request nonces,
// which keys.html does not have. Moving JS out lets the CSP use the strict
// `script-src 'self'` directive — see DEC-P31-FX-A in
// _qa_tmp/api_console_spec/decisions.md.
//
// This file is served by apin_server.py via /static/keys.js with ETag-based
// revalidation (same pattern as telemetry.js).
//
// Phase 3.2 will introduce the create-key flow; this file currently only
// handles the read-only list rendering.

(function () {
  'use strict';

  // ── DOM refs ────────────────────────────────────────────
  const $cards = document.getElementById('cards');
  const $ribbon = document.getElementById('ribbon');
  const $cActive = document.getElementById('count-active');
  const $cRotating = document.getElementById('count-rotating');
  const $cDisabled = document.getElementById('count-disabled');
  const $cOther = document.getElementById('count-other');
  const $fEnv = document.getElementById('f-env');
  const $fStatus = document.getElementById('f-status');
  const $fSearch = document.getElementById('f-search');
  const $pager = document.getElementById('pager');
  const $acct = document.getElementById('acct-label');
  const $sessionModal = document.getElementById('session-expired-modal');
  const $sessionSignIn = document.getElementById('session-expired-signin');

  // ── state ───────────────────────────────────────────────
  let allLoaded = [];
  let nextCursor = null;
  let isLoading = false;

  // ── helpers ─────────────────────────────────────────────
  const fmtDate = (iso) => {
    if (!iso) return '';
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    } catch (_) { return iso; }
  };

  const fmtRelative = (iso) => {
    if (!iso) return 'never used';
    try {
      const d = new Date(iso);
      const now = Date.now();
      const sec = Math.floor((now - d.getTime()) / 1000);
      if (sec < 60) return sec + 's ago';
      if (sec < 3600) return Math.floor(sec / 60) + 'm ago';
      if (sec < 86400) return Math.floor(sec / 3600) + 'h ago';
      const days = Math.floor(sec / 86400);
      if (days < 30) return days + 'd ago';
      return fmtDate(iso);
    } catch (_) { return iso; }
  };

  const esc = (s) => {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  };

  // ── session-expired modal (FX-E) ───────────────────────
  //
  // Replaces the previous broken-redirect path. The old code did
  //   window.location.href = '/dashboard?next=' + encodeURIComponent(...)
  // but the inference site's /dashboard handler discards `next=` and
  // sends the user back to /, so post-login they never returned to the
  // keys page (PDA-P3.1-R1 F01). Until the /dashboard handler honors
  // next=, we surface the expiry with an in-page modal and a "Sign in"
  // link that the user follows manually. Honest UX > a broken redirect.
  //
  // We stash the intended return path in sessionStorage so a future
  // /dashboard implementation can consume it without round-trip URLs.
  function showSessionExpiredModal() {
    try {
      sessionStorage.setItem('apin_post_login_return', window.location.pathname);
    } catch (_) { /* private mode etc.; non-fatal */ }
    if ($sessionModal) {
      $sessionModal.hidden = false;
      if ($sessionSignIn) $sessionSignIn.focus();
    } else {
      // Fallback if the modal element is somehow missing — go to /dashboard
      // (which surfaces the login modal). Users will need to navigate back
      // manually, but that's still better than a stuck/blank page.
      window.location.href = '/dashboard';
    }
  }

  if ($sessionSignIn) {
    $sessionSignIn.addEventListener('click', (ev) => {
      // Let the anchor's href handle navigation; we just ensure focus
      // is sensible. The href is /dashboard which renders the login modal.
      // No preventDefault — natural anchor click.
    });
  }

  // ── render a single key card ────────────────────────────
  function renderCard(key) {
    const status = key.status || 'unknown';
    const env = key.environment || 'live';
    const last4 = key.last_four || '????';
    const tokenPrefix = 'apin_' + env + '_';
    const scopes = Array.isArray(key.scopes) ? key.scopes : [];

    const lastUsed = key.last_used_at ? fmtRelative(key.last_used_at) : 'never used';
    const created = key.created_at ? fmtDate(key.created_at) : '';
    const expires = key.expires_at ? 'expires ' + fmtDate(key.expires_at) : 'no expiry';

    return (
      '<div class="card s-' + esc(status) + ' card-active" role="listitem"' +
      '     data-public-id="' + esc(key.public_id) + '"' +
      '     tabindex="0"' +
      '     aria-label="key ' + esc(key.name) + ' — ' + esc(status) + '">' +
      '  <div class="card-head">' +
      '    <span class="card-name">' + esc(key.name) + '</span>' +
      (env === 'test'
        ? '    <span class="env-pill env-test">test</span>'
        : '    <span class="env-pill">live</span>') +
      '    <span class="status-chip">' +
      '      <span class="status-dot" aria-hidden="true"></span>' +
      '      <span>' + esc(status) + '</span>' +
      '    </span>' +
      '  </div>' +
      '  <div class="token-mask">' +
      '    <span>' + esc(tokenPrefix) + '</span><span class="mask">******</span><span class="last4">' + esc(last4) + '</span>' +
      '  </div>' +
      '  <div class="meta-row">' +
      (created ? '<span>created ' + esc(created) + '</span><span class="pip">·</span>' : '') +
      '    <span>' + esc(expires) + '</span>' +
      '    <span class="pip">·</span>' +
      '    <span>last used ' + esc(lastUsed) + '</span>' +
      '  </div>' +
      (scopes.length
        ? ('<div class="scopes-row">' +
            scopes.slice(0, 6).map((s) => '<span class="scope-tag">' + esc(s) + '</span>').join('') +
            (scopes.length > 6 ? '<span class="scope-tag">+' + (scopes.length - 6) + ' more</span>' : '') +
          '</div>')
        : '') +
      '</div>'
    );
  }

  // ── empty / error placeholders ──────────────────────────
  function renderEmpty() {
    // Phase 8.G · two fixes:
    //   - was 🔑 emoji + ASCII "+" — now hand-drawn icons
    //   - was `disabled` with "lands in Phase 3.2" tooltip from when the
    //     create wizard didn't exist yet — wizard has shipped, so the
    //     empty-state CTA now opens it via the same path as the top-bar
    //     "+ new key" button (data-action="create-key" delegated below).
    return (
      '<div class="placeholder">' +
      '  <span class="hd-icon" aria-hidden="true">' +
      '    <svg><use href="#i-key"/></svg>' +
      '  </span>' +
      '  <h2>no API keys yet</h2>' +
      '  <p>Mint a key to start calling the API. Keys are shown <em>once</em>' +
      '     on creation — save them in your password manager immediately.</p>' +
      '  <div class="actions">' +
      '    <button class="btn btn-primary" data-action="create-key">' +
      '      <svg class="btn-icon" aria-hidden="true"><use href="#i-plus"/></svg>' +
      '      create your first key' +
      '    </button>' +
      '  </div>' +
      '</div>'
    );
  }

  // FX-A: replaced inline `onclick="window.location.reload()"` with a
  // data-action attribute + delegated listener (PDA-P3.1-R1 F04). CSP
  // `script-src 'self'` forbids inline event handlers.
  function renderError(message) {
    // Phase 8.G · ⚠ glyph → hand-drawn i-warning icon.
    return (
      '<div class="error-card">' +
      '  <span class="hd-icon hd-icon-warning" aria-hidden="true">' +
      '    <svg><use href="#i-warning"/></svg>' +
      '  </span>' +
      '  <div>' +
      '    <strong>Couldn\'t load your keys.</strong>' +
      '    <div style="color:var(--ink-soft);margin-top:2px">' + esc(message) + '</div>' +
      '  </div>' +
      '  <button class="btn" data-action="reload">retry</button>' +
      '</div>'
    );
  }

  function renderEmptyFiltered() {
    // Phase 8.G · 🔎 emoji → hand-drawn i-search icon.
    return (
      '<div class="placeholder">' +
      '  <span class="hd-icon" aria-hidden="true">' +
      '    <svg><use href="#i-search"/></svg>' +
      '  </span>' +
      '  <h2>no keys match your filters</h2>' +
      '  <p>Try removing a filter, or clearing the search.</p>' +
      '  <div class="actions">' +
      '    <button class="btn" id="btn-clear-filters">clear filters</button>' +
      '  </div>' +
      '</div>'
    );
  }

  // Delegated click handler on $cards — catches data-action buttons
  // inserted by renderError() and renderEmpty() (FX-A: replaces inline
  // onclick). 'reload' on error cards, 'create-key' on the empty-state
  // CTA (Phase 8.G fix — was previously disabled with a stale tooltip).
  $cards.addEventListener('click', (ev) => {
    const tgt = ev.target.closest('[data-action]');
    if (!tgt) return;
    const action = tgt.getAttribute('data-action');
    if (action === 'reload') {
      window.location.reload();
    } else if (action === 'create-key') {
      // Same path as the top-bar "+ new key" button. We dispatch a click
      // on $btnNew (rather than duplicating the handler body here) so any
      // future changes to the wizard-open flow stay in one place.
      const btn = document.getElementById('btn-new');
      if (btn) btn.click();
    }
  });

  // ── update the count ribbon ────────────────────────────
  function updateRibbon(items) {
    const c = { active: 0, rotating: 0, disabled: 0, other: 0 };
    for (const k of items) {
      const s = k.status || '';
      if (s === 'active') c.active++;
      else if (s === 'rotating') c.rotating++;
      else if (s === 'disabled') c.disabled++;
      else c.other++;
    }
    $cActive.textContent = c.active;
    $cRotating.textContent = c.rotating;
    $cDisabled.textContent = c.disabled;
    $cOther.textContent = c.other;
    $ribbon.hidden = items.length === 0;
  }

  // ── fetch keys from API ─────────────────────────────────
  async function fetchKeys({ reset = true } = {}) {
    if (isLoading) return;
    isLoading = true;
    $cards.setAttribute('aria-busy', 'true');

    const params = new URLSearchParams();
    params.set('env', $fEnv.value);
    params.set('status', $fStatus.value);
    if ($fSearch.value.trim()) params.set('search', $fSearch.value.trim());
    if (!reset && nextCursor !== null) params.set('cursor', String(nextCursor));
    params.set('limit', '20');

    try {
      const r = await fetch('/api/account/keys?' + params.toString(), {
        credentials: 'include',
        headers: { 'Accept': 'application/json' },
      });
      const body = await r.json();
      if (!r.ok || !body.ok) {
        const code = (body.error && body.error.code) || 'unknown';
        const msg = (body.error && body.error.message) || 'HTTP ' + r.status;
        if (code === 'invalid_or_missing_token') {
          // FX-E: don't bounce through /dashboard?next=... (broken). Show a
          // modal and let the user navigate to /dashboard via the anchor.
          showSessionExpiredModal();
          return;
        }
        renderResultsError(msg);
        return;
      }
      const data = body.data || {};
      const items = Array.isArray(data.items) ? data.items : [];
      if (reset) allLoaded = items;
      else allLoaded = allLoaded.concat(items);
      nextCursor = data.next_cursor;
      renderResults();
    } catch (err) {
      renderResultsError('Network error. Check your connection and try again.');
    } finally {
      isLoading = false;
      $cards.setAttribute('aria-busy', 'false');
    }
  }

  function renderResultsError(message) {
    $cards.innerHTML = renderError(message);
    $pager.hidden = true;
    updateRibbon([]);
  }

  function renderResults() {
    updateRibbon(allLoaded);
    if (allLoaded.length === 0) {
      const filtersActive = ($fEnv.value !== 'all') ||
        ($fStatus.value !== 'all') ||
        ($fSearch.value.trim() !== '');
      $cards.innerHTML = filtersActive ? renderEmptyFiltered() : renderEmpty();
      const clr = document.getElementById('btn-clear-filters');
      if (clr) clr.addEventListener('click', () => {
        $fEnv.value = 'all'; $fStatus.value = 'all'; $fSearch.value = '';
        fetchKeys({ reset: true });
      });
      $pager.hidden = true;
      return;
    }
    $cards.innerHTML = allLoaded.map(renderCard).join('');
    bindCardClicks();
    renderPager();
  }

  function renderPager() {
    if (nextCursor === null) {
      $pager.hidden = true;
      return;
    }
    $pager.hidden = false;
    $pager.innerHTML =
      '<div>showing ' + allLoaded.length + ' key' + (allLoaded.length === 1 ? '' : 's') + '</div>' +
      '<div class="controls">' +
      '  <button class="btn" id="btn-load-more">load 20 more</button>' +
      '</div>';
    document.getElementById('btn-load-more').addEventListener('click', () => {
      fetchKeys({ reset: false });
    });
  }

  function bindCardClicks() {
    for (const c of $cards.querySelectorAll('.card-active')) {
      // Phase 9.C: click navigates to the full per-key detail page (with
      // Overview/Usage/Requests/Audit/Settings tabs) instead of the
      // legacy in-page modal. Modal action flows (rotate/edit/delete)
      // still live on this page — the detail page's masthead buttons
      // round-trip back here via `#act=…&pid=…` for now.
      // Cmd/Ctrl-click and middle-click open in a new tab natively (we
      // emit a real <a>-like flow by setting location.href synchronously
      // only when no modifier is pressed).
      c.addEventListener('click', (ev) => {
        const pid = c.getAttribute('data-public-id');
        if (!pid) return;
        const url = '/account/api/keys/' + encodeURIComponent(pid) + '#overview';
        if (ev.metaKey || ev.ctrlKey || ev.shiftKey) {
          window.open(url, '_blank');
          return;
        }
        window.location.href = url;
      });
      c.addEventListener('auxclick', (ev) => {
        // Middle-click → new tab. Matches Vercel / Stripe behaviour.
        if (ev.button === 1) {
          const pid = c.getAttribute('data-public-id');
          if (pid) window.open(
            '/account/api/keys/' + encodeURIComponent(pid) + '#overview',
            '_blank');
        }
      });
      c.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          c.click();
        }
      });
    }
  }

  // ── filter debouncing ───────────────────────────────────
  let searchTimer = null;
  $fSearch.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => fetchKeys({ reset: true }), 250);
  });
  $fEnv.addEventListener('change', () => fetchKeys({ reset: true }));
  $fStatus.addEventListener('change', () => fetchKeys({ reset: true }));

  // ── keyboard shortcuts (§15.3 spec keymap) ─────────────
  document.addEventListener('keydown', (ev) => {
    // Don't trigger when typing in an input
    if (ev.target.matches('input, select, textarea')) return;
    if (ev.key === '/') {
      ev.preventDefault();
      $fSearch.focus();
    } else if (ev.key === 'n') {
      // Phase 3.2: open create wizard
      // ev.preventDefault();
      // window.location.href = '/account/api/keys/new';
    } else if (ev.key === 'Escape') {
      if ($sessionModal && !$sessionModal.hidden) {
        // Don't actually let the user dismiss without signing in — the
        // page is unusable until they re-auth. Re-focus the sign-in link.
        if ($sessionSignIn) $sessionSignIn.focus();
      }
    }
  });

  // ── Phase 3.4: create wizard + sudo step-up + one-time view ──────

  // DOM refs (modals)
  const $wizard          = document.getElementById('wizard-modal');
  const $wizardForm      = document.getElementById('wizard-form');
  const $wizardCancel    = document.getElementById('wizard-cancel');
  const $wizardSubmit    = document.getElementById('wizard-submit');
  const $wizardError     = document.getElementById('wizard-error');
  const $btnNew          = document.getElementById('btn-new');
  const $sudoModal       = document.getElementById('sudo-modal');
  const $sudoForm        = document.getElementById('sudo-form');
  const $sudoPassword    = document.getElementById('sudo-password');
  const $sudoCancel      = document.getElementById('sudo-cancel');
  const $sudoSubmit      = document.getElementById('sudo-submit');
  const $sudoError       = document.getElementById('sudo-error');
  const $onetime         = document.getElementById('onetime-modal');
  const $onetimeToken    = document.getElementById('onetime-token-value');
  const $onetimeCopyBtn  = document.getElementById('onetime-copy-btn');
  const $onetimeCopyFb   = document.getElementById('onetime-copy-feedback');
  const $onetimeMeta     = document.getElementById('onetime-meta');
  const $onetimeAck      = document.getElementById('onetime-ack');
  const $onetimeDone     = document.getElementById('onetime-done');

  // Modal helpers
  function openModal(el) { if (el) el.hidden = false; }
  function closeModal(el) { if (el) el.hidden = true; }
  function showError(el, msg) {
    if (!el) return;
    el.textContent = msg;
    el.hidden = false;
  }
  function clearError(el) {
    if (!el) return;
    el.textContent = '';
    el.hidden = true;
  }
  function setButtonLoading(btn, loading) {
    if (!btn) return;
    btn.disabled = loading;
    const lbl = btn.querySelector('.btn-label');
    const sp  = btn.querySelector('.btn-spinner');
    if (lbl) lbl.style.opacity = loading ? '0.4' : '1';
    if (sp)  sp.hidden = !loading;
  }

  // WI-P4-CSRF: real per-session CSRF token. Server seeds the value via
  // <meta name="csrf-token"> on every page render. We stash it in a mutable
  // variable because the value ROTATES on `POST /api/account/sudo` (per
  // spec §7.6 PDA-F44); the sudo response body's `data.csrf_token` carries
  // the new value, which we update here so subsequent mutations send the
  // rotated token. `_require_csrf` on the server uses constant-time compare
  // against `sessions.csrf_token` — a stale token = 401.
  let csrfToken = (function readInitialCsrf() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? (meta.content || '') : '';
  })();

  // FX-P5-2 (PDA-P5-R1 F02 + F03): unified pending-intent. A single object
  // {kind: 'mint'|'rotate', payload, publicId} replaces the previous pair
  // of (pendingAfterSudo: payload|null) + (pendingRotatePid: pid|null)
  // which racing via setTimeout(50ms). Single state = no cross-
  // contamination, no timing assumption. The sudo-success handler
  // dispatches on `pendingAfterSudo.kind` to retry the right operation.
  let pendingAfterSudo = null;

  // Open wizard
  $btnNew.addEventListener('click', () => {
    clearError($wizardError);
    $wizardForm.reset();
    // Re-check the default scope since reset clears it
    const defaultScope = $wizardForm.querySelector('input[name="scope"][value="predict:write"]');
    if (defaultScope) defaultScope.checked = true;
    openModal($wizard);
    document.getElementById('w-name').focus();
  });
  $wizardCancel.addEventListener('click', () => closeModal($wizard));

  // Wizard submit
  $wizardForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    clearError($wizardError);

    const fd = new FormData($wizardForm);
    const name = (fd.get('name') || '').trim();
    if (!name) { showError($wizardError, 'name is required'); return; }

    const environment = fd.get('environment') || 'live';
    const scopes = Array.from(
      $wizardForm.querySelectorAll('input[name="scope"]:checked')
    ).map((el) => el.value);
    if (scopes.length === 0) {
      showError($wizardError, 'pick at least one scope');
      return;
    }

    const payload = { name, environment, scopes };
    const expiresAt = (fd.get('expires_at') || '').trim();
    if (expiresAt) payload.expires_at = expiresAt + 'T00:00:00Z';
    const rate = parseInt(fd.get('rate_limit_per_min'), 10);
    if (!isNaN(rate) && rate > 0) payload.rate_limit_per_min = rate;
    const quota = parseInt(fd.get('quota_per_day'), 10);
    if (!isNaN(quota) && quota > 0) payload.quota_per_day = quota;
    const note = (fd.get('note') || '').trim();
    if (note) payload.note = note;

    await submitMint(payload);
  });

  async function submitMint(payload) {
    setButtonLoading($wizardSubmit, true);
    try {
      const r = await fetch('/api/account/keys', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'X-Console-Csrf': csrfToken,
        },
        body: JSON.stringify(payload),
      });
      const body = await r.json().catch(() => ({}));
      if (r.status === 201 && body.ok) {
        closeModal($wizard);
        showOneTimeView(body.data);
        fetchKeys({ reset: true });   // refresh list in background
        return;
      }
      const code = (body.error && body.error.code) || 'unknown';
      const msg  = (body.error && body.error.message) || 'HTTP ' + r.status;

      if (code === 'sudo_required') {
        // FX-P5-2: structured intent
        pendingAfterSudo = { kind: 'mint', payload };
        clearError($sudoError);
        $sudoPassword.value = '';
        closeModal($wizard);
        openModal($sudoModal);
        $sudoPassword.focus();
        return;
      }
      if (code === 'invalid_or_missing_token') {
        showSessionExpiredModal();
        return;
      }
      // Show field-specific error if present
      const field = body.error && body.error.details && body.error.details.field;
      showError($wizardError, field ? msg + ' (' + field + ')' : msg);
    } catch (err) {
      showError($wizardError, 'network error — try again');
    } finally {
      setButtonLoading($wizardSubmit, false);
    }
  }

  // Sudo step-up
  $sudoCancel.addEventListener('click', () => {
    pendingAfterSudo = null;
    closeModal($sudoModal);
  });
  $sudoForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    clearError($sudoError);
    const pw = $sudoPassword.value;
    if (!pw) { showError($sudoError, 'password required'); return; }

    setButtonLoading($sudoSubmit, true);
    try {
      const r = await fetch('/api/account/sudo', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'X-Console-Csrf': csrfToken,
        },
        body: JSON.stringify({ password: pw }),
      });
      // Zero out password ASAP regardless of result
      $sudoPassword.value = '';
      const body = await r.json().catch(() => ({}));
      if (r.status === 200 && body.ok) {
        // WI-P4-CSRF: server rotates session.csrf_token on sudo_started
        // (spec §7.6 PDA-F44). Stash the new token so subsequent mutations
        // send the rotated value. Without this, the retry POST would send
        // the pre-rotation token and fail with CSRF mismatch.
        const newCsrf = body.data && body.data.csrf_token;
        if (newCsrf) {
          csrfToken = newCsrf;
          // Also update the meta tag so any future page-script that reads
          // it sees the fresh value (defence in depth — we're the only
          // current reader, but consistency matters).
          const meta = document.querySelector('meta[name="csrf-token"]');
          if (meta) meta.content = newCsrf;
        }
        closeModal($sudoModal);
        // FX-P5-2: dispatch on intent kind. Replaces the previous
        // (pendingAfterSudo: payload-or-null) + (pendingRotatePid: pid-
        // or-null) + setTimeout race.
        if (pendingAfterSudo) {
          const intent = pendingAfterSudo;
          pendingAfterSudo = null;
          if (intent.kind === 'mint') {
            openModal($wizard);
            await submitMint(intent.payload);
          } else if (intent.kind === 'rotate') {
            openModal($detail);
            await submitRotate(intent.publicId);
          } else if (intent.kind === 'edit') {
            // Phase 6.A: re-open edit modal then retry PATCH
            openModal($edit);
            await submitEdit(intent.publicId, intent.payload);
          } else if (intent.kind === 'delete') {
            // Phase 6.A: re-open delete modal (user already confirmed
            // name; keep input populated so they can re-click confirm)
            openModal($del);
            await submitDelete(intent.publicId);
          }
        }
        return;
      }
      const code = (body.error && body.error.code) || 'unknown';
      const msg  = (body.error && body.error.message) || 'HTTP ' + r.status;
      if (code === 'invalid_or_missing_token') {
        showSessionExpiredModal();
        return;
      }
      // invalid_parameter on password field = wrong password
      showError($sudoError, msg);
    } catch (err) {
      showError($sudoError, 'network error — try again');
    } finally {
      setButtonLoading($sudoSubmit, false);
    }
  });

  // One-time-view modal
  function showOneTimeView(data) {
    const token = data && data.plaintext_token;
    if (!token) {
      // Should never happen — POST returns plaintext_token in data per spec
      showError($wizardError, 'mint succeeded but token was not in response');
      return;
    }
    $onetimeToken.textContent = token;
    $onetimeAck.checked = false;
    $onetimeDone.disabled = true;
    $onetimeCopyFb.hidden = true;

    // Populate meta
    const meta = [
      ['name',       data.name || ''],
      ['public id',  data.public_id || ''],
      ['environment', data.environment || ''],
      ['scopes',     (data.scopes || []).join(', ')],
      ['expires',    data.expires_at || 'no expiry'],
    ];
    $onetimeMeta.innerHTML = meta.map(([k, v]) =>
      '<div class="row"><span class="k">' + esc(k) + '</span><span>' + esc(v) + '</span></div>'
    ).join('');

    openModal($onetime);
  }

  // WI-P4-DL-SPLIT: download token as .env / .txt / .json
  // Helper: trigger a download of `content` as `filename` with `mime`.
  function triggerDownload(content, filename, mime) {
    const blob = new Blob([content], { type: mime + ';charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Free the blob URL on the next tick so the download has time to start
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }
  // Pull metadata from the most-recently-shown one-time view
  function currentOnetimeMeta() {
    const token = $onetimeToken.textContent || '';
    const rows = Array.from($onetimeMeta.querySelectorAll('.row'));
    const meta = {};
    for (const r of rows) {
      const ks = r.querySelectorAll('span');
      if (ks.length >= 2) meta[ks[0].textContent.trim()] = ks[1].textContent;
    }
    return { token, meta };
  }
  document.getElementById('dl-env-btn').addEventListener('click', () => {
    const { token, meta } = currentOnetimeMeta();
    if (!token) return;
    // Convention: variable name based on env ("APIN_KEY" for live, "APIN_TEST_KEY" for test)
    const env = (meta['environment'] || 'live').toUpperCase();
    const varName = env === 'TEST' ? 'APIN_TEST_KEY' : 'APIN_KEY';
    const body = '# APIN API key — created ' + new Date().toISOString() + '\n' +
                 '# name: ' + (meta['name'] || '') + '\n' +
                 '# DO NOT COMMIT THIS FILE\n' +
                 varName + '=' + token + '\n';
    triggerDownload(body, 'apin.env', 'text/plain');
  });
  document.getElementById('dl-txt-btn').addEventListener('click', () => {
    const { token, meta } = currentOnetimeMeta();
    if (!token) return;
    const body =
      'APIN API key\n' +
      '============\n\n' +
      'name:        ' + (meta['name'] || '') + '\n' +
      'public id:   ' + (meta['public id'] || '') + '\n' +
      'environment: ' + (meta['environment'] || '') + '\n' +
      'scopes:      ' + (meta['scopes'] || '') + '\n' +
      'expires:     ' + (meta['expires'] || 'no expiry') + '\n' +
      'created:     ' + new Date().toISOString() + '\n\n' +
      'token:\n' + token + '\n\n' +
      'IMPORTANT: this token is shown ONCE. Store it in a password\n' +
      'manager or secrets vault. APIN cannot recover it if lost.\n';
    triggerDownload(body, 'apin-' + (meta['name'] || 'key') + '.txt', 'text/plain');
  });
  document.getElementById('dl-json-btn').addEventListener('click', () => {
    const { token, meta } = currentOnetimeMeta();
    if (!token) return;
    const obj = {
      name: meta['name'] || '',
      public_id: meta['public id'] || '',
      environment: meta['environment'] || '',
      scopes: (meta['scopes'] || '').split(',').map((s) => s.trim()).filter(Boolean),
      expires: meta['expires'] || null,
      created: new Date().toISOString(),
      token: token,
    };
    triggerDownload(JSON.stringify(obj, null, 2), 'apin-' + (meta['name'] || 'key') + '.json', 'application/json');
  });

  // Copy to clipboard
  $onetimeCopyBtn.addEventListener('click', async () => {
    const token = $onetimeToken.textContent || '';
    if (!token) return;
    try {
      await navigator.clipboard.writeText(token);
      $onetimeCopyFb.hidden = false;
      // Hide feedback after 2 s
      setTimeout(() => { $onetimeCopyFb.hidden = true; }, 2000);
    } catch (_) {
      // Clipboard API failed (older browser, insecure context) — select
      // the text instead so user can Ctrl+C manually.
      const range = document.createRange();
      range.selectNode($onetimeToken);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });

  // Acknowledge gate — done button only enables when checkbox is ticked
  $onetimeAck.addEventListener('change', () => {
    $onetimeDone.disabled = !$onetimeAck.checked;
  });
  $onetimeDone.addEventListener('click', () => {
    // Zero out the token from the DOM before closing — defence in depth
    // against post-close inspection.
    $onetimeToken.textContent = '';
    closeModal($onetime);
  });

  // ── Phase 5.4: detail modal + rotate flow ────────────────────────

  const $detail          = document.getElementById('detail-modal');
  const $dTitle          = document.getElementById('detail-title');
  const $dStatusChip     = document.getElementById('detail-status-chip');
  const $dCloseX         = document.getElementById('detail-close-x');
  const $dClose          = document.getElementById('detail-close');
  const $dTokenPrefix    = document.getElementById('detail-token-prefix');
  const $dTokenLast4     = document.getElementById('detail-token-last4');
  const $dEnvPill        = document.getElementById('detail-env-pill');
  const $dPublicId       = document.getElementById('d-public-id');
  const $dCreated        = document.getElementById('d-created');
  const $dExpires        = document.getElementById('d-expires');
  const $dLastUsed       = document.getElementById('d-last-used');
  const $dRate           = document.getElementById('d-rate');
  const $dQuota          = document.getElementById('d-quota');
  const $dScopes         = document.getElementById('d-scopes');
  const $dIplist         = document.getElementById('d-iplist');
  const $dOrigList       = document.getElementById('d-origlist');
  const $dNoteSection    = document.getElementById('d-note-section');
  const $dNote           = document.getElementById('d-note');
  const $dError          = document.getElementById('detail-error');
  const $dRotateBtn      = document.getElementById('detail-rotate-btn');

  // The public_id of the key currently shown in the detail modal —
  // needed for rotate-after-sudo retry.
  let currentDetailPid = null;

  $dCloseX.addEventListener('click', () => closeModal($detail));
  $dClose.addEventListener('click', () => closeModal($detail));

  async function openDetailModal(publicId) {
    currentDetailPid = publicId;
    clearError($dError);
    // Reset content for the load — show "loading" placeholder by
    // emptying fields rather than leaving stale data from previous open.
    $dTitle.textContent = 'loading…';
    $dStatusChip.textContent = '';
    $dStatusChip.className = 'status-chip';
    $dPublicId.textContent = publicId;
    [$dCreated, $dExpires, $dLastUsed, $dRate, $dQuota].forEach((el) => {
      el.textContent = '';
    });
    $dScopes.innerHTML = '';
    $dIplist.innerHTML = '';
    $dOrigList.innerHTML = '';
    $dNoteSection.hidden = true;
    openModal($detail);

    try {
      const r = await fetch('/api/account/keys/' + encodeURIComponent(publicId), {
        credentials: 'include',
        headers: { 'Accept': 'application/json' },
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok || !body.ok) {
        const code = (body.error && body.error.code) || 'unknown';
        if (code === 'invalid_or_missing_token') {
          closeModal($detail);
          showSessionExpiredModal();
          return;
        }
        const msg = (body.error && body.error.message) || ('HTTP ' + r.status);
        $dTitle.textContent = 'error';
        showError($dError, msg);
        return;
      }
      renderDetail(body.data || {});
    } catch (err) {
      $dTitle.textContent = 'error';
      showError($dError, 'network error — try again');
    }
  }

  function renderDetail(d) {
    const status = d.status || 'unknown';
    const env    = d.environment || 'live';
    $dTitle.textContent = d.name || 'unnamed key';
    $dStatusChip.className = 'status-chip s-' + esc(status);
    $dStatusChip.innerHTML =
      '<span class="status-dot" aria-hidden="true"></span><span>' +
      esc(status) + '</span>';

    $dTokenPrefix.textContent = 'apin_' + env + '_';
    $dTokenLast4.textContent = d.last_four || '????';
    $dEnvPill.textContent = env;
    $dEnvPill.className = 'detail-env-pill' + (env === 'test' ? ' env-test' : '');

    $dPublicId.textContent = d.public_id || '';
    $dCreated.textContent  = d.created_at ? fmtDate(d.created_at) : '';
    $dExpires.textContent  = d.expires_at ? fmtDate(d.expires_at) : 'no expiry';
    $dLastUsed.textContent = d.last_used_at ? fmtRelative(d.last_used_at) : 'never used';
    $dRate.textContent     = d.rate_limit_per_min != null ? String(d.rate_limit_per_min) : 'default';
    $dQuota.textContent    = d.quota_per_day      != null ? String(d.quota_per_day)      : 'default';

    const scopes = Array.isArray(d.scopes) ? d.scopes : [];
    $dScopes.innerHTML = scopes.length === 0
      ? '<span class="placeholder-text">no scopes</span>'
      : scopes.map((s) => '<span class="scope-tag">' + esc(s) + '</span>').join('');

    const iplist = Array.isArray(d.ip_allowlist) ? d.ip_allowlist : [];
    $dIplist.innerHTML = iplist.length === 0
      ? '<span class="placeholder-text">any IP allowed</span>'
      : iplist.map((ip) => '<span>' + esc(ip) + '</span>').join('<span class="pip">·</span>');

    const orig = Array.isArray(d.origin_allowlist) ? d.origin_allowlist : [];
    $dOrigList.innerHTML = orig.length === 0
      ? '<span class="placeholder-text">any origin allowed</span>'
      : orig.map((o) => '<span>' + esc(o) + '</span>').join('<span class="pip">·</span>');

    if (d.note) {
      $dNote.textContent = d.note;
      $dNoteSection.hidden = false;
    } else {
      $dNoteSection.hidden = true;
    }

    // Rotate is meaningful only for active or rotating keys
    $dRotateBtn.disabled = !(status === 'active' || status === 'rotating');
  }

  // Rotate handler: tries POST /rotate; on 403 sudo_required, stashes
  // intent in the unified `pendingAfterSudo` (FX-P5-2 — was a separate
  // `pendingRotatePid` + setTimeout race). On 200/201, shows one-time-view.
  $dRotateBtn.addEventListener('click', async () => {
    if (!currentDetailPid) return;
    await submitRotate(currentDetailPid);
  });

  // FX-P5-4 (VER-P5-R1 5.4a): show/hide custom grace input
  const $graceCustomRow = document.getElementById('grace-custom-row');
  const $graceCustomHours = document.getElementById('grace-custom-hours');
  document.querySelectorAll('input[name="grace"]').forEach((radio) => {
    radio.addEventListener('change', (ev) => {
      $graceCustomRow.hidden = ev.target.value !== 'custom';
      if (ev.target.value === 'custom') $graceCustomHours.focus();
    });
  });

  // FX-P5-4: build {grace_seconds: N} payload from current radio selection.
  // Returns null on invalid custom input (so caller can show error).
  function readGraceSeconds() {
    const selected = document.querySelector('input[name="grace"]:checked');
    if (!selected) return 172800;   // default 48h
    if (selected.value === 'custom') {
      const hours = parseInt($graceCustomHours.value, 10);
      if (isNaN(hours) || hours < 0 || hours > 720) {
        return null;   // signal invalid
      }
      return hours * 3600;
    }
    const sec = parseInt(selected.value, 10);
    return isNaN(sec) ? 172800 : sec;
  }

  async function submitRotate(publicId) {
    clearError($dError);
    // FX-P5-4: validate grace input before submit
    const graceSeconds = readGraceSeconds();
    if (graceSeconds === null) {
      showError($dError, 'custom grace period must be 0–720 hours');
      return;
    }
    setButtonLoading($dRotateBtn, true);
    try {
      const r = await fetch(
        '/api/account/keys/' + encodeURIComponent(publicId) + '/rotate',
        {
          method: 'POST',
          credentials: 'include',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Console-Csrf': csrfToken,
          },
          body: JSON.stringify({ grace_seconds: graceSeconds }),
        }
      );
      const body = await r.json().catch(() => ({}));
      // FX-P5-1: accept 200 OR 201. Server now returns 201 (added
      // status_code=201 to the rotate decorator for consistency with
      // mint), but accept 200 too in case a future spec change reverts
      // to the FastAPI default — defensive forward-compat.
      if ((r.status === 200 || r.status === 201) && body.ok) {
        closeModal($detail);
        showOneTimeView(body.data);
        fetchKeys({ reset: true });   // refresh list — status may have changed
        return;
      }
      const code = (body.error && body.error.code) || 'unknown';
      const msg  = (body.error && body.error.message) || ('HTTP ' + r.status);

      if (code === 'sudo_required') {
        // FX-P5-2: structured intent (kind=rotate); unified handler in
        // the sudo POST success branch will dispatch back to submitRotate.
        pendingAfterSudo = { kind: 'rotate', publicId };
        clearError($sudoError);
        $sudoPassword.value = '';
        closeModal($detail);   // close detail so user focuses on sudo
        openModal($sudoModal);
        $sudoPassword.focus();
        return;
      }
      if (code === 'invalid_or_missing_token') {
        closeModal($detail);
        showSessionExpiredModal();
        return;
      }
      showError($dError, msg);
    } catch (err) {
      showError($dError, 'network error — try again');
    } finally {
      setButtonLoading($dRotateBtn, false);
    }
  }

  // FX-P5-2: the second `$sudoForm` submit listener with the 50ms
  // setTimeout race is REMOVED. The unified `pendingAfterSudo` intent
  // is dispatched directly inside the existing sudo POST success
  // handler (see above). No timing assumptions; no cross-contamination.

  // ── Phase 6.A: Edit + Delete handlers ───────────────────────────

  const $edit          = document.getElementById('edit-modal');
  const $editForm      = document.getElementById('edit-form');
  const $editCancel    = document.getElementById('edit-cancel');
  const $editSubmit    = document.getElementById('edit-submit');
  const $editError     = document.getElementById('edit-error');
  const $editPid       = document.getElementById('e-public-id');
  const $editName      = document.getElementById('e-name');
  const $editRate      = document.getElementById('e-rate');
  const $editQuota     = document.getElementById('e-quota');
  const $editNote      = document.getElementById('e-note');
  const $editBtn       = document.getElementById('detail-edit-btn');

  const $del           = document.getElementById('delete-modal');
  const $delName       = document.getElementById('del-name-display');
  const $delInput      = document.getElementById('del-confirm-input');
  const $delCancel     = document.getElementById('delete-cancel');
  const $delConfirm    = document.getElementById('delete-confirm-btn');
  const $delError      = document.getElementById('delete-error');
  const $delBtn        = document.getElementById('detail-delete-btn');

  // Track the detail's currently-loaded key data so edit can pre-fill
  // and delete can show the name. Set by renderDetail.
  let currentDetailData = null;
  const _origRenderDetail = renderDetail;
  renderDetail = function (d) {
    currentDetailData = d;
    return _origRenderDetail(d);
  };

  // Edit button → open modal pre-filled with current values
  $editBtn.addEventListener('click', () => {
    if (!currentDetailData) return;
    clearError($editError);
    $editForm.reset();
    $editPid.value   = currentDetailData.public_id || '';
    $editName.value  = currentDetailData.name || '';
    $editRate.value  = currentDetailData.rate_limit_per_min ?? '';
    $editQuota.value = currentDetailData.quota_per_day ?? '';
    $editNote.value  = currentDetailData.note || '';
    // Re-check scopes
    const have = new Set(currentDetailData.scopes || []);
    $editForm.querySelectorAll('input[name="e-scope"]').forEach((cb) => {
      cb.checked = have.has(cb.value);
    });
    closeModal($detail);
    openModal($edit);
    $editName.focus();
  });
  $editCancel.addEventListener('click', () => {
    closeModal($edit);
    if (currentDetailData) openModal($detail);
  });

  $editForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    clearError($editError);
    const pid = $editPid.value;
    if (!pid) return;

    // Build PATCH payload with only CHANGED fields (smaller, safer).
    const orig = currentDetailData || {};
    const payload = {};
    const newName = $editName.value.trim();
    if (newName !== (orig.name || '')) payload.name = newName;

    const newScopes = Array.from(
      $editForm.querySelectorAll('input[name="e-scope"]:checked')
    ).map((cb) => cb.value).sort();
    const origScopes = (orig.scopes || []).slice().sort();
    if (JSON.stringify(newScopes) !== JSON.stringify(origScopes)) {
      if (newScopes.length === 0) {
        showError($editError, 'at least one scope required');
        return;
      }
      payload.scopes = newScopes;
    }

    const newRate = $editRate.value === '' ? null : parseInt($editRate.value, 10);
    if (newRate !== (orig.rate_limit_per_min ?? null)) payload.rate_limit_per_min = newRate;
    const newQuota = $editQuota.value === '' ? null : parseInt($editQuota.value, 10);
    if (newQuota !== (orig.quota_per_day ?? null)) payload.quota_per_day = newQuota;
    const newNote = $editNote.value.trim() || null;
    if (newNote !== (orig.note || null)) payload.note = newNote;

    if (Object.keys(payload).length === 0) {
      showError($editError, 'nothing changed');
      return;
    }

    await submitEdit(pid, payload);
  });

  async function submitEdit(pid, payload) {
    setButtonLoading($editSubmit, true);
    try {
      const r = await fetch('/api/account/keys/' + encodeURIComponent(pid), {
        method: 'PATCH',
        credentials: 'include',
        headers: {'Accept':'application/json','Content-Type':'application/json',
                   'X-Console-Csrf': csrfToken},
        body: JSON.stringify(payload),
      });
      const body = await r.json().catch(() => ({}));
      if (r.status === 200 && body.ok) {
        closeModal($edit);
        openModal($detail);
        renderDetail(body.data);   // update with server-returned canonical state
        fetchKeys({reset: true});  // refresh list
        return;
      }
      const code = (body.error && body.error.code) || 'unknown';
      if (code === 'sudo_required') {
        pendingAfterSudo = { kind: 'edit', publicId: pid, payload };
        closeModal($edit);
        openModal($sudoModal);
        $sudoPassword.focus();
        return;
      }
      if (code === 'invalid_or_missing_token') {
        closeModal($edit);
        showSessionExpiredModal();
        return;
      }
      const field = body.error && body.error.details && body.error.details.field;
      showError($editError, field ? body.error.message + ' (' + field + ')'
                                  : (body.error.message || 'HTTP ' + r.status));
    } catch (err) {
      showError($editError, 'network error — try again');
    } finally {
      setButtonLoading($editSubmit, false);
    }
  }

  // Delete button → open confirmation modal
  $delBtn.addEventListener('click', () => {
    if (!currentDetailData) return;
    clearError($delError);
    $delName.textContent = currentDetailData.name || '(unnamed)';
    $delInput.value = '';
    $delConfirm.disabled = true;
    closeModal($detail);
    openModal($del);
    $delInput.focus();
  });
  $delCancel.addEventListener('click', () => {
    closeModal($del);
    if (currentDetailData) openModal($detail);
  });
  $delInput.addEventListener('input', () => {
    $delConfirm.disabled = ($delInput.value !== (currentDetailData?.name || ''));
  });
  $delConfirm.addEventListener('click', async () => {
    if (!currentDetailData) return;
    await submitDelete(currentDetailData.public_id);
  });

  // Phase 8.H fix · server requires the key to be 'disabled' or 'expired'
  // before hard-delete (no accidental delete of an active production key).
  // The previous UI didn't honour that precondition — user typed name,
  // clicked delete forever, got the wall-of-text "not in a deletable
  // status" error. Now we transparently disable first when the status
  // is 'active' or 'rotating', then proceed with delete. Status is
  // taken from `currentDetailData` which is loaded when the detail
  // modal opens (the delete dialog is always reached via that modal).
  async function _disableKeyIfNeeded(pid) {
    if (!currentDetailData) return true;   // no info, let server decide
    const status = String(currentDetailData.status || '').toLowerCase();
    if (status === 'disabled' || status === 'expired') return true;
    // Not in a deletable state — try disable first.
    const r = await fetch('/api/account/keys/' + encodeURIComponent(pid)
                          + '/disable', {
      method: 'POST', credentials: 'include',
      headers: {'Accept':'application/json','X-Console-Csrf': csrfToken},
    });
    const body = await r.json().catch(() => ({}));
    if (r.status === 200 && body.ok) {
      currentDetailData.status = 'disabled';  // keep local copy fresh
      return true;
    }
    // Bubble the failure up to the caller — sudo / network / etc.
    const code = (body.error && body.error.code) || 'unknown';
    if (code === 'sudo_required') {
      pendingAfterSudo = { kind: 'delete', publicId: pid };
      closeModal($del);
      openModal($sudoModal);
      $sudoPassword.focus();
      return false;
    }
    if (code === 'invalid_or_missing_token') {
      closeModal($del);
      showSessionExpiredModal();
      return false;
    }
    showError($delError, 'Could not disable before delete: '
                        + (body.error?.message || 'HTTP ' + r.status));
    return false;
  }

  async function submitDelete(pid) {
    clearError($delError);
    setButtonLoading($delConfirm, true);
    try {
      // Phase 8.H · transparent auto-disable for active/rotating keys.
      // Spec requires status ∈ {disabled, expired} before hard-delete.
      const ok = await _disableKeyIfNeeded(pid);
      if (!ok) return;   // helper already surfaced the right error / sudo

      const r = await fetch('/api/account/keys/' + encodeURIComponent(pid), {
        method: 'DELETE',
        credentials: 'include',
        headers: {'Accept':'application/json','X-Console-Csrf': csrfToken},
      });
      const body = await r.json().catch(() => ({}));
      // DELETE returns 200 with envelope on success
      if (r.status === 200 && body.ok) {
        closeModal($del);
        // Don't re-open detail — the key no longer exists
        currentDetailData = null;
        fetchKeys({reset: true});
        return;
      }
      const code = (body.error && body.error.code) || 'unknown';
      if (code === 'sudo_required') {
        pendingAfterSudo = { kind: 'delete', publicId: pid };
        closeModal($del);
        openModal($sudoModal);
        $sudoPassword.focus();
        return;
      }
      if (code === 'invalid_or_missing_token') {
        closeModal($del);
        showSessionExpiredModal();
        return;
      }
      // If we get here, the auto-disable succeeded but the DELETE still
      // failed (e.g. concurrent re-enable, race condition). Surface raw.
      showError($delError, body.error?.message || 'HTTP ' + r.status);
    } catch (err) {
      showError($delError, 'network error — try again');
    } finally {
      setButtonLoading($delConfirm, false);
    }
  }

  // ── initial load ────────────────────────────────────────
  fetchKeys({ reset: true });
})();
