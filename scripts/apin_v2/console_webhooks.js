(function () {
  'use strict';

  const csrf = (function(){ const m = document.querySelector('meta[name="csrf-token"]'); return m ? m.content : ''; })();
  let liveCsrf = csrf;   // rotated after each sudo step-up

  // ── DOM ──────────────────────────────────────────────────────────────
  const $list = document.getElementById('wh-list');
  const $capPill = document.getElementById('cap-pill');
  const $svcDown = document.getElementById('svc-down-banner');
  const $toast = document.getElementById('toast');

  const $newBtn = document.getElementById('new-wh-btn');
  const $whModal = document.getElementById('wh-modal');
  const $whModalTitle = document.getElementById('wh-modal-title');
  const $whModalIntro = document.getElementById('wh-modal-intro');
  const $whModalSubmit = document.getElementById('wh-modal-submit');
  const $whModalErr = document.getElementById('wh-modal-error');
  const $whUrl = document.getElementById('wh-url');
  const $whDesc = document.getElementById('wh-desc');
  const $whEvents = document.getElementById('wh-events');
  const $whAllowSelfSigned = document.getElementById('wh-allow-self-signed');
  const $whAllowInternal = document.getElementById('wh-allow-internal');

  const $secretModal = document.getElementById('secret-modal');
  const $secretTitle = document.getElementById('secret-title');
  const $secretValue = document.getElementById('secret-value');
  const $secretCopy = document.getElementById('secret-copy');
  const $secretGraceNote = document.getElementById('secret-grace-note');

  const $deleteModal = document.getElementById('delete-modal');
  const $delUrlDisplay = document.getElementById('del-url-display');
  const $delConfirmInput = document.getElementById('del-confirm-input');
  const $delConfirmBtn = document.getElementById('delete-confirm-btn');
  const $delErr = document.getElementById('delete-error');

  const $testModal = document.getElementById('test-modal');
  const $testUrlDisplay = document.getElementById('test-url-display');
  const $testFireBtn = document.getElementById('test-fire-btn');
  const $testResultArea = document.getElementById('test-result-area');
  const $testErr = document.getElementById('test-error');

  const $rotateModal = document.getElementById('rotate-modal');
  const $rotateGrace = document.getElementById('rotate-grace');
  const $rotateConfirmBtn = document.getElementById('rotate-confirm-btn');
  const $rotateErr = document.getElementById('rotate-error');

  const $sudoModal = document.getElementById('sudo-modal');
  const $sudoPw = document.getElementById('sudo-pw');
  const $sudoSubmit = document.getElementById('sudo-submit');
  const $sudoErr = document.getElementById('sudo-error');

  // ── State ────────────────────────────────────────────────────────────
  let webhooks = [];
  let cap = 50;
  let editingId = null;   // null = create, else id of webhook being edited
  let deletingId = null;
  let testingId = null;
  let rotatingId = null;
  let pendingRetry = null;   // function to re-run after sudo

  // ── Helpers ──────────────────────────────────────────────────────────
  function showToast(msg, level) {
    $toast.textContent = msg;
    $toast.className = 'toast show ' + (level === 'err' ? 'err' : 'ok');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => { $toast.className = 'toast'; }, 2400);
  }
  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function fmtAgo(iso) {
    if (!iso) return 'never';
    const ms = Date.now() - new Date(iso).getTime();
    if (isNaN(ms)) return iso;
    if (ms < 60_000) return Math.round(ms/1000) + 's ago';
    if (ms < 3600_000) return Math.round(ms/60_000) + 'm ago';
    if (ms < 86400_000) return Math.round(ms/3600_000) + 'h ago';
    return new Date(iso).toLocaleDateString();
  }
  function openModal(m) { m.hidden = false; }
  function closeModal(m) { m.hidden = true; }
  // Wire generic close buttons
  document.querySelectorAll('[data-close]').forEach(btn => {
    btn.addEventListener('click', () => {
      const m = btn.closest('.modal-veil');
      if (m) closeModal(m);
    });
  });
  // Esc to close any modal
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      [$whModal, $secretModal, $deleteModal, $testModal, $rotateModal, $sudoModal]
        .forEach(m => { if (!m.hidden) closeModal(m); });
    }
  });

  // ── Load list ────────────────────────────────────────────────────────
  async function loadList() {
    try {
      const r = await fetch('/api/account/webhooks', {credentials:'include'});
      if (r.status === 401) { window.location.href = '/dashboard'; return; }
      const body = await r.json();
      if (r.status === 503 || body.error?.code === 'service_unavailable') {
        renderServiceOffline(body.error?.message || 'Webhook service is offline.');
        return;
      }
      if (!body.ok) { showToast(body.error?.message || 'Failed to load webhooks', 'err'); return; }
      webhooks = body.data.items || [];
      cap = body.data.cap || 50;
      $svcDown.innerHTML = '';
      render();
    } catch (err) {
      showToast('Network error while loading webhooks.', 'err');
    }
  }

  function renderServiceOffline(msg) {
    $svcDown.innerHTML = '<div class="svc-down"><strong>Webhook service unavailable.</strong> ' + escHtml(msg) + '</div>';
    $list.innerHTML = '';
    $capPill.textContent = '—';
    $newBtn.disabled = true;
  }

  // ── Render ───────────────────────────────────────────────────────────
  function render() {
    $capPill.innerHTML = '<strong>' + webhooks.length + '</strong> / ' + cap + ' webhooks';
    $newBtn.disabled = (webhooks.length >= cap);

    if (webhooks.length === 0) {
      $list.innerHTML = ''
        + '<div class="list-empty">'
        + '<h3>No webhooks yet</h3>'
        + '<p>A webhook receives signed HTTP POSTs whenever events you subscribe to fire. Use them to notify Slack, page your team on quota incidents, or kick off downstream pipelines when a key is rotated.</p>'
        + '<button type="button" class="btn btn-primary" id="empty-new-btn">+ create your first webhook</button>'
        + '</div>';
      document.getElementById('empty-new-btn').addEventListener('click', () => openCreate());
      return;
    }

    $list.innerHTML = webhooks.map(renderCard).join('');
    wireCards();
  }

  function renderCard(w) {
    const isAutoDisabled = !!w.auto_disabled_at;
    const stateCls = isAutoDisabled ? 'auto-disabled' : (w.active ? 'active' : 'disabled');
    const stateLbl = isAutoDisabled ? 'auto-disabled' : (w.active ? 'active' : 'disabled');
    const eventsHtml = (w.events || []).map(e =>
      '<span class="event-chip">' + escHtml(e) + '</span>').join('');
    return '<div class="wh-card ' + stateCls + '" data-wh-id="' + escHtml(w.id) + '">'
      + '<div class="wh-head">'
        + '<div>'
          + '<div class="wh-url">' + escHtml(w.url) + '</div>'
          + (w.description ? '<div class="wh-desc">' + escHtml(w.description) + '</div>' : '')
        + '</div>'
        + '<div class="wh-status-area">'
          + '<span class="wh-pill ' + stateCls + '">' + stateLbl + '</span>'
        + '</div>'
      + '</div>'
      + '<div class="wh-events">' + eventsHtml + '</div>'
      + '<dl class="wh-meta">'
        + '<div><dt>id</dt><dd>' + escHtml(w.id) + '</dd></div>'
        + '<div><dt>created</dt><dd>' + fmtAgo(w.created_at) + '</dd></div>'
        + '<div><dt>last delivery</dt><dd>' + fmtAgo(w.last_delivery_at)
            + (w.last_delivery_status
                ? ' · HTTP ' + w.last_delivery_status
                : '')
            + '</dd></div>'
        + '<div><dt>consecutive failures</dt><dd>' + (w.consecutive_failure_count || 0) + '</dd></div>'
      + '</dl>'
      + '<div class="wh-actions">'
        + '<button type="button" class="btn btn-sm" data-act="test">test ping</button>'
        + '<button type="button" class="btn btn-sm" data-act="rotate">rotate secret</button>'
        + '<button type="button" class="btn btn-sm" data-act="edit">edit</button>'
        + '<button type="button" class="btn btn-sm" data-act="deliveries">show deliveries</button>'
        + '<span style="flex:1"></span>'
        + '<button type="button" class="btn btn-sm btn-ghost" data-act="delete">delete</button>'
      + '</div>'
      + '<div class="wh-deliveries" data-deliveries></div>'
      + '</div>';
  }

  function wireCards() {
    $list.querySelectorAll('.wh-card').forEach(card => {
      const id = card.dataset.whId;
      card.querySelectorAll('[data-act]').forEach(btn => {
        btn.addEventListener('click', () => {
          const act = btn.dataset.act;
          if (act === 'edit')        openEdit(id);
          else if (act === 'delete') openDelete(id);
          else if (act === 'test')   openTest(id);
          else if (act === 'rotate') openRotate(id);
          else if (act === 'deliveries') toggleDeliveries(card, id);
        });
      });
    });
  }

  // ── Deliveries drawer ────────────────────────────────────────────────
  async function toggleDeliveries(card, id) {
    const drawer = card.querySelector('[data-deliveries]');
    if (drawer.classList.contains('open')) {
      drawer.classList.remove('open');
      card.querySelector('[data-act="deliveries"]').textContent = 'show deliveries';
      return;
    }
    drawer.innerHTML = '<h4>Recent deliveries</h4><p style="font-family:Fraunces,serif;font-style:italic;color:var(--ink-soft);font-size:12px">loading&hellip;</p>';
    drawer.classList.add('open');
    card.querySelector('[data-act="deliveries"]').textContent = 'hide deliveries';
    try {
      const r = await fetch('/api/account/webhooks/' + id + '/deliveries?limit=25', {credentials:'include'});
      const body = await r.json();
      if (!body.ok) { drawer.innerHTML = '<h4>Recent deliveries</h4><p style="color:var(--accent-crimson)">' + escHtml(body.error?.message || 'load failed') + '</p>'; return; }
      const items = body.data.items || [];
      if (items.length === 0) {
        drawer.innerHTML = '<h4>Recent deliveries</h4><p style="font-family:Fraunces,serif;font-style:italic;color:var(--ink-soft);font-size:12px">No deliveries yet. Fire a test ping to seed one.</p>';
        return;
      }
      drawer.innerHTML = '<h4>Recent deliveries (last ' + items.length + ')</h4>'
        + '<table class="dlv-table"><thead><tr>'
        + '<th>queued</th><th>event</th><th>status</th><th>attempts</th><th>HTTP</th><th>error</th>'
        + '</tr></thead><tbody>'
        + items.map(d => '<tr>'
            + '<td>' + fmtAgo(d.queued_at) + '</td>'
            + '<td>' + escHtml(d.event_type) + '</td>'
            + '<td><span class="dlv-status ' + escHtml(d.status) + '">' + escHtml(d.status) + '</span></td>'
            + '<td>' + (d.attempts || 0) + '</td>'
            + '<td>' + (d.response_status || '—') + '</td>'
            + '<td>' + escHtml(d.error_text || '—').slice(0, 80) + '</td>'
          + '</tr>').join('')
        + '</tbody></table>';
    } catch (err) {
      drawer.innerHTML = '<h4>Recent deliveries</h4><p style="color:var(--accent-crimson)">network error</p>';
    }
  }

  // ── Create / edit ────────────────────────────────────────────────────
  $newBtn.addEventListener('click', () => openCreate());

  function openCreate() {
    editingId = null;
    $whModalTitle.textContent = 'New webhook';
    $whModalIntro.textContent = 'A webhook receives signed HTTP POSTs whenever the events you subscribe to fire. You will get the signing secret ONCE on create — save it then.';
    $whModalSubmit.textContent = 'Create';
    $whUrl.value = ''; $whDesc.value = '';
    $whEvents.querySelectorAll('input').forEach(c => c.checked = false);
    $whAllowSelfSigned.checked = false; $whAllowInternal.checked = false;
    $whModalErr.hidden = true;
    openModal($whModal);
    setTimeout(() => $whUrl.focus(), 30);
  }
  function openEdit(id) {
    const w = webhooks.find(x => x.id === id);
    if (!w) return;
    editingId = id;
    $whModalTitle.textContent = 'Edit webhook';
    $whModalIntro.textContent = 'Update URL, description, subscribed events, or advanced flags. The signing secret is unchanged — use Rotate to mint a new one.';
    $whModalSubmit.textContent = 'Save changes';
    $whUrl.value = w.url || '';
    $whDesc.value = w.description || '';
    const set = new Set(w.events || []);
    $whEvents.querySelectorAll('input').forEach(c => c.checked = set.has(c.value));
    $whAllowSelfSigned.checked = !!w.allow_self_signed;
    $whAllowInternal.checked = !!w.allow_internal_target;
    $whModalErr.hidden = true;
    openModal($whModal);
  }
  $whModalSubmit.addEventListener('click', () => {
    $whModalErr.hidden = true;
    const url = $whUrl.value.trim();
    const desc = $whDesc.value.trim();
    const events = [...$whEvents.querySelectorAll('input:checked')].map(c => c.value);
    if (!url || !events.length) {
      $whModalErr.hidden = false;
      $whModalErr.textContent = 'URL and at least one event are required.'; return;
    }
    const payload = {
      url,
      events,
      description: desc || null,
      allow_self_signed: $whAllowSelfSigned.checked,
      allow_internal_target: $whAllowInternal.checked,
    };
    if (editingId) submitEdit(editingId, payload);
    else submitCreate(payload);
  });

  async function submitCreate(payload) {
    $whModalSubmit.disabled = true;
    try {
      const r = await fetch('/api/account/webhooks', {
        method: 'POST', credentials: 'include',
        headers: {'Accept':'application/json','Content-Type':'application/json',
                  'X-Console-Csrf': liveCsrf},
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      if (r.status === 201 && body.ok) {
        closeModal($whModal);
        revealSecret(body.data.signing_secret, false);
        loadList();
        return;
      }
      handleApiErr(body, () => submitCreate(payload), $whModalErr);
    } catch (err) {
      $whModalErr.hidden = false; $whModalErr.textContent = 'Network error.';
    } finally { $whModalSubmit.disabled = false; }
  }

  async function submitEdit(id, payload) {
    $whModalSubmit.disabled = true;
    try {
      const r = await fetch('/api/account/webhooks/' + id, {
        method: 'PATCH', credentials: 'include',
        headers: {'Accept':'application/json','Content-Type':'application/json',
                  'X-Console-Csrf': liveCsrf},
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      if (body.ok) {
        closeModal($whModal);
        showToast('Webhook updated.');
        loadList();
        return;
      }
      handleApiErr(body, () => submitEdit(id, payload), $whModalErr);
    } catch (err) {
      $whModalErr.hidden = false; $whModalErr.textContent = 'Network error.';
    } finally { $whModalSubmit.disabled = false; }
  }

  // ── Delete ───────────────────────────────────────────────────────────
  function openDelete(id) {
    const w = webhooks.find(x => x.id === id);
    if (!w) return;
    deletingId = id;
    $delUrlDisplay.textContent = w.url;
    $delConfirmInput.value = ''; $delConfirmBtn.disabled = true;
    $delErr.hidden = true;
    openModal($deleteModal);
    setTimeout(() => $delConfirmInput.focus(), 30);
  }
  $delConfirmInput.addEventListener('input', () => {
    $delConfirmBtn.disabled = ($delConfirmInput.value.trim() !== $delUrlDisplay.textContent.trim());
  });
  $delConfirmBtn.addEventListener('click', () => submitDelete(deletingId));
  async function submitDelete(id) {
    $delConfirmBtn.disabled = true;
    try {
      const r = await fetch('/api/account/webhooks/' + id, {
        method: 'DELETE', credentials: 'include',
        headers: {'Accept':'application/json','X-Console-Csrf': liveCsrf},
      });
      const body = await r.json();
      if (body.ok) {
        closeModal($deleteModal);
        showToast('Webhook deleted.');
        loadList(); return;
      }
      handleApiErr(body, () => submitDelete(id), $delErr);
    } catch (err) {
      $delErr.hidden = false; $delErr.textContent = 'Network error.';
    } finally { $delConfirmBtn.disabled = false; }
  }

  // ── Test ─────────────────────────────────────────────────────────────
  function openTest(id) {
    const w = webhooks.find(x => x.id === id);
    if (!w) return;
    testingId = id;
    $testUrlDisplay.textContent = w.url;
    $testResultArea.innerHTML = '';
    $testErr.hidden = true;
    openModal($testModal);
  }
  $testFireBtn.addEventListener('click', () => submitTest(testingId));
  async function submitTest(id) {
    $testFireBtn.disabled = true;
    $testFireBtn.textContent = 'Firing…';
    $testResultArea.innerHTML = '';
    try {
      const r = await fetch('/api/account/webhooks/' + id + '/test', {
        method: 'POST', credentials: 'include',
        headers: {'Accept':'application/json','X-Console-Csrf': liveCsrf},
      });
      const body = await r.json();
      if (body.ok) {
        const d = body.data;
        const isOk = d.status === 'delivered';
        $testResultArea.innerHTML = '<div class="test-result ' + (isOk ? 'ok' : 'bad') + '">'
          + '<h4>' + (isOk ? 'Delivered' : 'Failed') + '</h4>'
          + '<div class="res-meta">HTTP ' + (d.response_status || '?')
              + ' · ' + d.elapsed_ms + ' ms · event_id ' + escHtml(d.event_id) + '</div>'
          + (d.error_text ? '<div class="res-meta" style="margin-top:4px">' + escHtml(d.error_text) + '</div>' : '')
          + '</div>';
        loadList();
        return;
      }
      handleApiErr(body, () => submitTest(id), $testErr);
    } catch (err) {
      $testErr.hidden = false; $testErr.textContent = 'Network error firing test.';
    } finally {
      $testFireBtn.disabled = false; $testFireBtn.textContent = 'Fire test event';
    }
  }

  // ── Rotate ───────────────────────────────────────────────────────────
  function openRotate(id) {
    rotatingId = id;
    $rotateGrace.value = '86400';
    $rotateErr.hidden = true;
    openModal($rotateModal);
    setTimeout(() => $rotateGrace.focus(), 30);
  }
  $rotateConfirmBtn.addEventListener('click', () => submitRotate(rotatingId));
  async function submitRotate(id) {
    const grace = parseInt($rotateGrace.value, 10);
    if (isNaN(grace) || grace < 0) {
      $rotateErr.hidden = false; $rotateErr.textContent = 'Grace seconds must be a non-negative integer.'; return;
    }
    $rotateConfirmBtn.disabled = true;
    try {
      const r = await fetch('/api/account/webhooks/' + id + '/rotate-secret', {
        method: 'POST', credentials: 'include',
        headers: {'Accept':'application/json','Content-Type':'application/json',
                  'X-Console-Csrf': liveCsrf},
        body: JSON.stringify({grace_seconds: grace}),
      });
      const body = await r.json();
      if (body.ok) {
        closeModal($rotateModal);
        revealSecret(body.data.signing_secret, true, grace);
        loadList(); return;
      }
      handleApiErr(body, () => submitRotate(id), $rotateErr);
    } catch (err) {
      $rotateErr.hidden = false; $rotateErr.textContent = 'Network error.';
    } finally { $rotateConfirmBtn.disabled = false; }
  }

  // ── Secret reveal ────────────────────────────────────────────────────
  function revealSecret(secret, isRotation, graceSec) {
    $secretTitle.textContent = isRotation ? 'Save your NEW signing secret' : 'Save your signing secret';
    $secretValue.textContent = secret;
    $secretGraceNote.innerHTML = isRotation
      ? '<p>The old secret stays valid for <strong>' + graceSec
        + ' seconds</strong> (' + Math.round(graceSec/3600) + ' h). After that, only this new secret is accepted.</p>'
      : '';
    openModal($secretModal);
  }
  $secretCopy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($secretValue.textContent);
      $secretCopy.classList.add('copied'); $secretCopy.textContent = 'copied';
      setTimeout(() => { $secretCopy.classList.remove('copied'); $secretCopy.textContent = 'copy'; }, 1400);
    } catch (_) { showToast('Clipboard blocked. Select & copy manually.', 'err'); }
  });

  // ── Sudo flow ────────────────────────────────────────────────────────
  function handleApiErr(body, retryFn, errEl) {
    const code = body.error?.code;
    const msg  = body.error?.message || 'Failed';
    if (code === 'sudo_required') {
      pendingRetry = retryFn;
      openSudo();
      return;
    }
    if (code === 'invalid_or_missing_token') {
      showToast('Session expired. Redirecting…', 'err');
      setTimeout(() => { window.location.href = '/dashboard'; }, 1200);
      return;
    }
    if (errEl) { errEl.hidden = false; errEl.textContent = msg; }
    else showToast(msg, 'err');
  }
  function openSudo() {
    $sudoPw.value = ''; $sudoErr.hidden = true;
    openModal($sudoModal);
    setTimeout(() => $sudoPw.focus(), 30);
  }
  $sudoPw.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); $sudoSubmit.click(); }
  });
  $sudoSubmit.addEventListener('click', async () => {
    const pw = $sudoPw.value;
    if (!pw) { $sudoErr.hidden = false; $sudoErr.textContent = 'Enter your password.'; return; }
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
        if (body.data?.csrf_token) liveCsrf = body.data.csrf_token;
        closeModal($sudoModal);
        if (pendingRetry) {
          const fn = pendingRetry; pendingRetry = null;
          await fn();
        }
        return;
      }
      const code = body.error?.code;
      if (code === 'invalid_password') {
        $sudoErr.hidden = false; $sudoErr.textContent = 'Password did not match.'; $sudoPw.select();
      } else if (code === 'rate_limited') {
        $sudoErr.hidden = false; $sudoErr.textContent = body.error?.message || 'Too many attempts.';
      } else {
        $sudoErr.hidden = false; $sudoErr.textContent = body.error?.message || ('HTTP ' + r.status);
      }
    } catch (err) {
      $sudoErr.hidden = false; $sudoErr.textContent = 'Network error.';
    } finally { $sudoSubmit.disabled = false; }
  });

  // ── Bootstrap ────────────────────────────────────────────────────────
  loadList();
})();