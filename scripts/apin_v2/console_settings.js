(function () {
  'use strict';

  // ── State ────────────────────────────────────────────────────────────
  // FX-P7-F04: mutable so we can adopt the rotated CSRF token after a
  // successful sudo step-up. Prior `const csrf` made the second save
  // after sudo fail with "session expired" (mismatched token).
  let liveCsrf = document.querySelector('meta[name="csrf-token"]').content;
  const $form = document.getElementById('settings-form');
  const $toast = document.getElementById('toast');
  const $actionBar = document.getElementById('action-bar');
  const $abStatus = document.getElementById('ab-status');
  const $saveBtn = document.getElementById('save-btn');
  const $discardBtn = document.getElementById('discard-btn');
  const $loading = document.getElementById('loading-veil');
  const $helpPop = document.getElementById('help-pop');

  // sudo modal refs
  const $sudoModal  = document.getElementById('sudo-modal');
  const $sudoPw     = document.getElementById('sudo-pw');
  const $sudoErr    = document.getElementById('sudo-error');
  const $sudoSubmit = document.getElementById('sudo-submit');
  const $sudoCancel = document.getElementById('sudo-cancel');
  const $sudoClose  = document.getElementById('sudo-close');
  const $sudoTtlHint = document.getElementById('sudo-ttl-hint');

  // server canonical state vs working draft
  let serverState = {};
  let pendingRetry = null;   // payload to retry after sudo completes

  // ── Utility ──────────────────────────────────────────────────────────
  function showToast(msg, level = 'ok') {
    $toast.textContent = msg;
    $toast.className = 'toast show ' + (level === 'err' ? 'err' : level === 'ok' ? 'ok' : '');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => { $toast.className = 'toast'; }, 2800);
  }

  function $$(sel) { return document.querySelectorAll(sel); }
  function $(sel) { return document.querySelector(sel); }

  // ── Field <-> DOM read/write ─────────────────────────────────────────

  // For each [data-field] return its current DOM value cast to the right type.
  function readFieldValue(input) {
    const f = input.dataset.field;
    if (input.type === 'checkbox') return input.checked;
    if (input.type === 'number') {
      if (input.value === '') {
        // Empty number: for nullable threshold this means "disable", otherwise leave alone
        return (f === 'notify_on_auth_failures_threshold') ? null : undefined;
      }
      return parseInt(input.value, 10);
    }
    return input.value;
  }

  function writeFieldValue(input, val) {
    if (input.type === 'checkbox') {
      input.checked = !!val;
    } else if (input.type === 'number') {
      input.value = (val === null || val === undefined) ? '' : String(val);
    } else {
      input.value = (val == null) ? '' : String(val);
    }
    input.classList.remove('dirty', 'error');
  }

  // Chip group for sudo_required_for is special — read/write a Set.
  function readChips() {
    return [...$$('#f-sudo-actions .chip.on')].map(c => c.dataset.action);
  }
  function writeChips(arr) {
    const set = new Set(arr || []);
    $$('#f-sudo-actions .chip').forEach(c => {
      if (set.has(c.dataset.action)) c.classList.add('on');
      else c.classList.remove('on');
    });
  }

  // ── Dirty-state tracking ─────────────────────────────────────────────

  function computeDirty() {
    const dirtyFields = [];
    // Standard inputs
    $$('[data-field]').forEach(input => {
      const f = input.dataset.field;
      const cur = readFieldValue(input);
      if (cur === undefined) return;
      const server = serverState[f];
      if (JSON.stringify(cur) !== JSON.stringify(server)) {
        dirtyFields.push(f);
        input.classList.add('dirty');
      } else {
        input.classList.remove('dirty');
      }
    });
    // Chips
    const chipsNow = readChips().sort();
    const chipsServer = (serverState.sudo_required_for || []).slice().sort();
    if (JSON.stringify(chipsNow) !== JSON.stringify(chipsServer)) {
      dirtyFields.push('sudo_required_for');
    }
    return dirtyFields;
  }

  function refreshActionBar() {
    const dirty = computeDirty();
    if (dirty.length === 0) {
      $actionBar.classList.remove('show');
      $abStatus.innerHTML = 'No changes yet.';
      return;
    }
    $actionBar.classList.add('show');
    if (dirty.length === 1) {
      $abStatus.innerHTML = '<strong>1</strong> unsaved change in <strong>'
                            + sectionForField(dirty[0]) + '</strong>.';
    } else {
      $abStatus.innerHTML = '<strong>' + dirty.length
                            + '</strong> unsaved changes.';
    }
  }

  function sectionForField(f) {
    const map = {
      'sudo_session_length_seconds': 'Sudo step-up',
      'sudo_max_uses': 'Sudo step-up',
      'sudo_required_for': 'Sudo step-up',
      'default_key_expiry_days': 'New-key defaults',
      'default_scope_template': 'New-key defaults',
      'require_ip_allowlist': 'New-key defaults',
      'notify_on_key_created': 'Notifications',
      'notify_on_first_use_from_new_ip': 'Notifications',
      'notify_on_quota_exceeded': 'Notifications',
      'notify_on_auth_failures_threshold': 'Notifications',
      'ip_truncation_enabled': 'Privacy & retention',
      'request_log_retention_days': 'Privacy & retention',
      'audit_log_retention_days': 'Privacy & retention',
      'webhook_delivery_retention_days': 'Privacy & retention',
      'org_aggregate_rate_limit': 'Rate limits & quotas',
      'org_aggregate_quota_per_day': 'Rate limits & quotas',
      'sandbox_rate_per_min': 'Rate limits & quotas',
      'sandbox_max_upload_mb_per_min': 'Rate limits & quotas',
      'max_webhooks_per_user': 'Rate limits & quotas',
      'auto_revoke_on_partner_detection': 'Advanced',
    };
    return map[f] || 'this section';
  }

  // ── Load / save ──────────────────────────────────────────────────────

  function fillForm(data) {
    serverState = Object.assign({}, data);
    $$('[data-field]').forEach(input => writeFieldValue(input, data[input.dataset.field]));
    writeChips(data.sudo_required_for);
    if (data.sudo_session_length_seconds) {
      const m = Math.round(data.sudo_session_length_seconds / 60);
      $sudoTtlHint.textContent = m + ' minute' + (m === 1 ? '' : 's');
    }
    refreshActionBar();
    clearFieldErrors();
  }

  function clearFieldErrors() {
    $$('.field-control .field-error').forEach(e => e.remove());
    $$('[data-field].error').forEach(i => i.classList.remove('error'));
  }

  function showFieldError(field, msg) {
    const input = document.querySelector('[data-field="' + field + '"]');
    if (!input) {
      showToast(msg, 'err'); return;
    }
    input.classList.add('error');
    const wrap = input.parentElement;
    let span = wrap.querySelector('.field-error');
    if (!span) {
      span = document.createElement('span');
      span.className = 'field-error';
      wrap.appendChild(span);
    }
    span.textContent = msg;
    input.scrollIntoView({behavior:'smooth', block:'center'});
    input.focus();
  }

  async function loadSettings() {
    try {
      const r = await fetch('/api/account/settings', {credentials:'include',
        headers:{'Accept':'application/json'}});
      const body = await r.json();
      if (r.status === 401 || (body.error && body.error.code === 'invalid_or_missing_token')) {
        showToast('Session expired. Redirecting to dashboard…', 'err');
        setTimeout(() => { window.location.href = '/dashboard'; }, 1200);
        return;
      }
      if (!body.ok) {
        showToast(body.error?.message || 'Failed to load settings', 'err');
        return;
      }
      fillForm(body.data || {});
    } catch (err) {
      showToast('Network error while loading settings.', 'err');
    } finally {
      $loading.hidden = true;
    }
  }

  function buildPayload(dirtyOnly = true) {
    const out = {};
    const dirty = dirtyOnly ? new Set(computeDirty()) : null;
    $$('[data-field]').forEach(input => {
      const f = input.dataset.field;
      if (dirty && !dirty.has(f)) return;
      const v = readFieldValue(input);
      if (v === undefined) return;
      out[f] = v;
    });
    if (!dirty || dirty.has('sudo_required_for')) {
      out.sudo_required_for = readChips();
    }
    return out;
  }

  async function saveChanges() {
    clearFieldErrors();
    const payload = buildPayload(true);
    if (Object.keys(payload).length === 0) {
      showToast('No changes to save.', 'err');
      return;
    }
    $saveBtn.disabled = true;
    try {
      const r = await fetch('/api/account/settings', {
        method: 'PATCH', credentials: 'include',
        headers: {'Accept':'application/json','Content-Type':'application/json',
                  'X-Console-Csrf': liveCsrf},
        body: JSON.stringify(payload),
      });
      const body = await r.json();

      if (body.ok) {
        fillForm(body.data || {});
        $actionBar.classList.add('save-pulse');
        setTimeout(() => $actionBar.classList.remove('save-pulse'), 900);
        showToast('Settings saved.', 'ok');
        return;
      }
      const code = body.error?.code || 'unknown';
      const msg  = body.error?.message || ('HTTP ' + r.status);
      if (code === 'invalid_or_missing_token') {
        showToast('Session expired. Redirecting…', 'err');
        setTimeout(() => { window.location.href = '/dashboard'; }, 1200);
        return;
      }
      if (code === 'sudo_required') {
        pendingRetry = payload;
        openSudoModal();
        return;
      }
      if (code === 'invalid_parameter') {
        // Map the message back to the field it mentions if we can
        const fields = Object.keys(payload);
        const hit = fields.find(f => msg.indexOf(f) !== -1);
        if (hit) showFieldError(hit, msg);
        else showToast(msg, 'err');
        return;
      }
      showToast(msg, 'err');
    } catch (err) {
      showToast('Network error while saving. Retry in a moment.', 'err');
    } finally {
      $saveBtn.disabled = false;
    }
  }

  // ── Embedded sudo modal ──────────────────────────────────────────────

  function openSudoModal() {
    $sudoErr.hidden = true; $sudoErr.textContent = '';
    $sudoPw.value = '';
    $sudoModal.hidden = false;
    setTimeout(() => $sudoPw.focus(), 30);
  }
  function closeSudoModal() {
    $sudoModal.hidden = true;
    $sudoPw.value = '';
  }

  $sudoCancel.addEventListener('click', () => {
    closeSudoModal(); pendingRetry = null;
    showToast('Cancelled. Your changes are still here unsaved.', 'err');
  });
  $sudoClose.addEventListener('click', () => { $sudoCancel.click(); });
  $sudoPw.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); $sudoSubmit.click(); }
    if (e.key === 'Escape') { $sudoCancel.click(); }
  });

  $sudoSubmit.addEventListener('click', async () => {
    const pw = $sudoPw.value;
    if (!pw) { $sudoErr.textContent = 'Enter your password.'; $sudoErr.hidden = false; return; }
    $sudoSubmit.disabled = true;
    try {
      const r = await fetch('/api/account/sudo', {
        method: 'POST', credentials: 'include',
        headers: {'Accept':'application/json','Content-Type':'application/json',
                  'X-Console-Csrf': liveCsrf},
        body: JSON.stringify({password: pw}),
      });
      const body = await r.json();
      if (body.ok) {
        // FX-P7-F04: adopt the rotated CSRF token so subsequent saves
        // in the same page lifetime use the fresh value.
        const newCsrf = body.data && body.data.csrf_token;
        if (newCsrf) {
          liveCsrf = newCsrf;
          document.querySelector('meta[name="csrf-token"]').content = newCsrf;
        }
        closeSudoModal();
        if (pendingRetry) {
          const payload = pendingRetry;
          pendingRetry = null;
          await retryPatch(payload);
        }
        return;
      }
      const code = body.error?.code || 'unknown';
      if (code === 'invalid_password') {
        $sudoErr.textContent = 'Password did not match. Try again.';
        $sudoErr.hidden = false;
        $sudoPw.select();
      } else if (code === 'rate_limited') {
        $sudoErr.textContent = body.error.message
          || 'Too many attempts. Wait a few minutes and try again.';
        $sudoErr.hidden = false;
      } else {
        $sudoErr.textContent = body.error?.message || ('HTTP ' + r.status);
        $sudoErr.hidden = false;
      }
    } catch (err) {
      $sudoErr.textContent = 'Network error. Check your connection.';
      $sudoErr.hidden = false;
    } finally {
      $sudoSubmit.disabled = false;
    }
  });

  async function retryPatch(payload) {
    $saveBtn.disabled = true;
    try {
      // FX-P7-F04 (cont.): read liveCsrf at call time. The closure sees
      // the rotated value the sudo handler just wrote, no parameter needed.
      const r = await fetch('/api/account/settings', {
        method: 'PATCH', credentials: 'include',
        headers: {'Accept':'application/json','Content-Type':'application/json',
                  'X-Console-Csrf': liveCsrf},
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      if (body.ok) {
        fillForm(body.data || {});
        $actionBar.classList.add('save-pulse');
        setTimeout(() => $actionBar.classList.remove('save-pulse'), 900);
        showToast('Settings saved.', 'ok');
      } else {
        showToast(body.error?.message || 'Save failed after sudo.', 'err');
      }
    } catch (err) {
      showToast('Network error after sudo. Retry.', 'err');
    } finally {
      $saveBtn.disabled = false;
    }
  }

  // ── Event wiring ─────────────────────────────────────────────────────

  $$('[data-field]').forEach(input => {
    const evt = (input.type === 'checkbox' || input.tagName === 'SELECT') ? 'change' : 'input';
    input.addEventListener(evt, refreshActionBar);
  });
  $$('#f-sudo-actions .chip').forEach(c => {
    c.addEventListener('click', () => {
      c.classList.toggle('on');
      refreshActionBar();
    });
  });

  $saveBtn.addEventListener('click', saveChanges);
  $discardBtn.addEventListener('click', () => {
    fillForm(serverState);
    showToast('Reverted to last-saved values.', 'ok');
  });

  // Keyboard shortcut: cmd/ctrl-S to save when bar is showing
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 's') {
      if ($actionBar.classList.contains('show')) {
        e.preventDefault();
        saveChanges();
      }
    }
  });

  // ── TOC scroll-spy ───────────────────────────────────────────────────
  const tocLinks = [...$$('.toc a[data-toc-target]')];
  const sections = tocLinks.map(a => document.getElementById(a.dataset.tocTarget));
  function activateToc() {
    let activeIdx = 0;
    for (let i = 0; i < sections.length; i++) {
      const r = sections[i].getBoundingClientRect();
      if (r.top < 120) activeIdx = i;
    }
    tocLinks.forEach((a, i) => a.classList.toggle('active', i === activeIdx));
  }
  window.addEventListener('scroll', activateToc, {passive:true});

  // ── Help popover ────────────────────────────────────────────────────
  $$('.help[data-help]').forEach(h => {
    h.addEventListener('mouseenter', (e) => positionHelp(e.target));
    h.addEventListener('focus',      (e) => positionHelp(e.target));
    h.addEventListener('mouseleave', hideHelp);
    h.addEventListener('blur',       hideHelp);
    h.setAttribute('tabindex', '0');
    h.setAttribute('role', 'button');
    h.setAttribute('aria-label', 'Help: ' + h.dataset.help.slice(0, 60));
  });
  function positionHelp(el) {
    $helpPop.textContent = el.dataset.help;
    $helpPop.hidden = false;
    requestAnimationFrame(() => {
      const r = el.getBoundingClientRect();
      const pw = $helpPop.offsetWidth;
      const ph = $helpPop.offsetHeight;
      let left = r.left - 12;
      if (left + pw > window.innerWidth - 16) left = window.innerWidth - pw - 16;
      $helpPop.style.left = (left + window.scrollX) + 'px';
      $helpPop.style.top  = (r.top - ph - 10 + window.scrollY) + 'px';
      $helpPop.classList.add('show');
    });
  }
  function hideHelp() {
    $helpPop.classList.remove('show');
    setTimeout(() => { if (!$helpPop.classList.contains('show')) $helpPop.hidden = true; }, 150);
  }

  // ── Warn on unload if dirty ─────────────────────────────────────────
  window.addEventListener('beforeunload', (e) => {
    if (computeDirty().length > 0) {
      e.preventDefault(); e.returnValue = '';
    }
  });

  // ── Bootstrap ────────────────────────────────────────────────────────
  loadSettings();

})();