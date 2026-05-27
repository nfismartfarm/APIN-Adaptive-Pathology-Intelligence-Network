(function () {
  'use strict';

  // ── Endpoint catalog ──────────────────────────────────────────────────
  // Each entry: { method, path (with {param}), title, desc, auth, body, queryHints }
  //   auth: 'session' (cookie), 'session+csrf' (cookie + X-Console-Csrf),
  //         'bearer' (predict/full only), 'public' (none)
  //   body: null | 'json' | 'file' (multipart with image=)
  const ENDPOINTS = [
    // Keys
    { group: 'Keys', method: 'GET',    path: '/api/account/keys',
      title: 'List keys',
      desc: 'Returns all keys belonging to the signed-in account, with last_4 + scopes. Cursor pagination via `?cursor=`.',
      auth: 'session', body: null,
      queryHints: ['cursor', 'page_size', 'environment', 'status'] },
    { group: 'Keys', method: 'POST',   path: '/api/account/keys',
      title: 'Mint key',
      desc: 'Creates a new key. The plaintext value is returned ONCE. Requires sudo.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"name":"sandbox-test","scopes":["predict:write"],"environment":"test","expires_in_days":30} },
    { group: 'Keys', method: 'GET',    path: '/api/account/keys/{public_id}',
      title: 'Fetch one key',
      desc: 'Returns the full record for one key. last_4 only, never the plaintext.',
      auth: 'session', body: null },
    { group: 'Keys', method: 'PATCH',  path: '/api/account/keys/{public_id}',
      title: 'Edit key',
      desc: 'Update name, scopes, rate, quota, allowlist, or note. Requires sudo. Send only the changed fields.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"name":"updated-name","note":"used by backend"} },
    { group: 'Keys', method: 'POST',   path: '/api/account/keys/{public_id}/rotate',
      title: 'Rotate key',
      desc: 'Mint a new value for an existing key; old value valid for grace_seconds (default 86400). Requires sudo.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"grace_seconds":86400} },
    { group: 'Keys', method: 'DELETE', path: '/api/account/keys/{public_id}',
      title: 'Delete key',
      desc: 'Hard-delete. The key must already be disabled. Requires sudo.',
      auth: 'session+csrf', body: null },
    // Sudo
    { group: 'Sudo', method: 'GET',    path: '/api/account/sudo',
      title: 'Sudo state',
      desc: 'Returns whether you currently hold a sudo session, expires_at, and uses_remaining.',
      auth: 'session', body: null },
    { group: 'Sudo', method: 'POST',   path: '/api/account/sudo',
      title: 'Step up to sudo',
      desc: 'Confirm with password to mint a sudo cookie. Rate-limited 5/5min/IP.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"password":"your_password_here"} },
    { group: 'Sudo', method: 'POST',   path: '/api/account/sudo/revoke',
      title: 'Revoke sudo',
      desc: 'Tear down the current sudo session immediately.',
      auth: 'session+csrf', body: null },
    // Settings
    { group: 'Settings', method: 'GET',   path: '/api/account/settings',
      title: 'Get settings',
      desc: 'Returns the full account_settings row (with schema defaults filled in).',
      auth: 'session', body: null },
    { group: 'Settings', method: 'PATCH', path: '/api/account/settings',
      title: 'Update settings',
      desc: 'Update any subset of the 20 editable fields. Requires sudo.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"sudo_max_uses":60,"notify_on_quota_exceeded":true} },
    // Webhooks
    { group: 'Webhooks', method: 'GET',    path: '/api/account/webhooks',
      title: 'List webhooks',
      desc: 'Active + disabled webhooks with cap usage.',
      auth: 'session', body: null },
    { group: 'Webhooks', method: 'POST',   path: '/api/account/webhooks',
      title: 'Create webhook',
      desc: 'Mint a webhook + signing secret (returned ONCE). Requires sudo.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"url":"https://example.com/hook","events":["account.alert_raised"],"description":"sandbox test"} },
    { group: 'Webhooks', method: 'GET',    path: '/api/account/webhooks/{webhook_id}',
      title: 'Fetch webhook',
      desc: 'Single webhook record. Secret is never returned.',
      auth: 'session', body: null },
    { group: 'Webhooks', method: 'PATCH',  path: '/api/account/webhooks/{webhook_id}',
      title: 'Edit webhook',
      desc: 'Update URL, events, description, active, or allow_* flags. Requires sudo.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"description":"updated","active":false} },
    { group: 'Webhooks', method: 'DELETE', path: '/api/account/webhooks/{webhook_id}',
      title: 'Delete webhook',
      desc: 'Hard-delete; cascades to delivery rows. Requires sudo.',
      auth: 'session+csrf', body: null },
    { group: 'Webhooks', method: 'POST',   path: '/api/account/webhooks/{webhook_id}/rotate-secret',
      title: 'Rotate signing secret',
      desc: 'Mint a new signing secret; old secret stays valid for grace_seconds. Requires sudo.',
      auth: 'session+csrf', body: 'json',
      bodyExample: {"grace_seconds":86400} },
    { group: 'Webhooks', method: 'POST',   path: '/api/account/webhooks/{webhook_id}/test',
      title: 'Test-ping webhook',
      desc: 'Fires ONE synchronous POST to the URL. Returns status + ms. Requires sudo.',
      auth: 'session+csrf', body: null },
    { group: 'Webhooks', method: 'GET',    path: '/api/account/webhooks/{webhook_id}/deliveries',
      title: 'Delivery log',
      desc: 'Recent delivery attempts for one webhook, newest first.',
      auth: 'session', body: null,
      queryHints: ['limit'] },
    // Alerts
    { group: 'Alerts', method: 'GET',    path: '/api/account/alerts',
      title: 'List alerts',
      desc: 'Alerts feed with filters. Cursor pagination.',
      auth: 'session', body: null,
      queryHints: ['severity', 'code', 'only_unread', 'limit', 'cursor'] },
    { group: 'Alerts', method: 'GET',    path: '/api/account/alerts/unread-count',
      title: 'Unread count',
      desc: 'Cheap query for the nav bell badge.',
      auth: 'session', body: null },
    { group: 'Alerts', method: 'GET',    path: '/api/account/alerts/{alert_id}',
      title: 'Fetch alert',
      desc: 'Single alert record with parsed details.',
      auth: 'session', body: null },
    { group: 'Alerts', method: 'PATCH',  path: '/api/account/alerts/{alert_id}/read',
      title: 'Mark alert read',
      desc: 'Idempotent. Sets read_at.',
      auth: 'session+csrf', body: null },
    { group: 'Alerts', method: 'POST',   path: '/api/account/alerts/{alert_id}/restore',
      title: 'Restore alert',
      desc: 'Reverses a prior dismiss.',
      auth: 'session+csrf', body: null },
    { group: 'Alerts', method: 'DELETE', path: '/api/account/alerts/{alert_id}',
      title: 'Dismiss alert',
      desc: 'Soft-delete. Sets dismissed_at; reversible via /restore.',
      auth: 'session+csrf', body: null },
    // Predict
    { group: 'Predict', method: 'POST',  path: '/api/predict/full',
      title: 'Predict (full pipeline)',
      desc: 'Routes the image to the right specialist and returns the diagnosis. Uses Bearer auth with one of your live keys; session cookie is NOT honored.',
      auth: 'bearer', body: 'file' },
  ];

  // ── State ─────────────────────────────────────────────────────────────
  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  let currentEp = ENDPOINTS[0];
  let pathParams = {};
  let queryParams = [];   // [{k, v}]
  let bodyValue = '';
  let fileBlob = null;
  let bearerKey = null;
  let userKeys = [];

  // ── DOM refs ──────────────────────────────────────────────────────────
  const $epSel  = document.getElementById('endpoint-select');
  const $methodPill = document.getElementById('method-pill');
  const $epDesc = document.getElementById('endpoint-desc');
  const $pathSec = document.getElementById('path-section');
  const $pathParams = document.getElementById('path-params');
  const $querySec = document.getElementById('query-section');
  const $queryRows = document.getElementById('query-rows');
  const $addQueryBtn = document.getElementById('add-query-btn');
  const $fileSec = document.getElementById('file-section');
  const $filePick = document.getElementById('file-pick');
  const $fileInput = document.getElementById('file-input');
  const $fileLabel = document.getElementById('file-pick-label');
  const $keySec = document.getElementById('key-section');
  const $keySel = document.getElementById('key-select');
  const $bodySec = document.getElementById('body-section');
  const $bodyBadge = document.getElementById('body-badge');
  const $bodyEditor = document.getElementById('body-editor');
  const $bodyHint = document.getElementById('body-hint');
  const $curlPreview = document.getElementById('curl-preview');
  const $copyCurlBtn = document.getElementById('copy-curl-btn');
  const $sendBtn = document.getElementById('send-btn');
  const $respStatus = document.getElementById('response-status');
  const $respMeta = document.getElementById('resp-meta');
  const $respBody = document.getElementById('response-body');
  const $respHeaders = document.getElementById('response-headers');
  const $histList = document.getElementById('history-list');
  const $histCount = document.getElementById('hist-count');
  const $clearHistBtn = document.getElementById('clear-hist-btn');
  const $toast = document.getElementById('toast');

  // ── Helpers ───────────────────────────────────────────────────────────
  function showToast(msg, level) {
    $toast.textContent = msg;
    $toast.className = 'toast show ' + (level === 'err' ? 'err' : 'ok');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => { $toast.className = 'toast'; }, 2400);
  }
  function escHtml(s) {
    return String(s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function syntaxHighlightJson(obj) {
    let json;
    try { json = JSON.stringify(obj, null, 2); } catch (_) { return escHtml(String(obj)); }
    return json
      .replace(/(&|<|>)/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))
      .replace(/"([^"\\]*(\\.[^"\\]*)*)"(\s*:)/g, '<span class="j-key">"$1"</span>$3')
      .replace(/: "([^"\\]*(\\.[^"\\]*)*)"/g, ': <span class="j-str">"$1"</span>')
      .replace(/: (true|false)\b/g, ': <span class="j-bool">$1</span>')
      .replace(/: (null)\b/g, ': <span class="j-null">$1</span>')
      .replace(/: (-?\d+\.?\d*)/g, ': <span class="j-num">$1</span>');
  }
  function fmtAgo(ts) {
    const ms = Date.now() - ts;
    if (ms < 60_000) return Math.round(ms/1000) + 's ago';
    if (ms < 3600_000) return Math.round(ms/60_000) + 'm ago';
    if (ms < 86400_000) return Math.round(ms/3600_000) + 'h ago';
    return new Date(ts).toLocaleDateString();
  }
  function statusClass(s) {
    if (s >= 200 && s < 300) return 's-2xx';
    if (s >= 300 && s < 400) return 's-3xx';
    if (s >= 400 && s < 500) return 's-4xx';
    return 's-5xx';
  }

  // ── Endpoint dropdown ─────────────────────────────────────────────────
  function buildEndpointDropdown() {
    // Group as optgroups
    const groups = {};
    ENDPOINTS.forEach((ep, idx) => {
      if (!groups[ep.group]) groups[ep.group] = [];
      groups[ep.group].push({ep, idx});
    });
    $epSel.innerHTML = '';
    Object.keys(groups).forEach(g => {
      const og = document.createElement('optgroup');
      og.label = g;
      groups[g].forEach(({ep, idx}) => {
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.textContent = ep.method.padEnd(7, ' ') + ' ' + ep.path + '  · ' + ep.title;
        og.appendChild(opt);
      });
      $epSel.appendChild(og);
    });
  }

  // ── Reactive: when endpoint changes, rebuild compose form ─────────────
  function setEndpoint(idx) {
    currentEp = ENDPOINTS[idx];
    pathParams = {};
    queryParams = [];
    bodyValue = '';
    fileBlob = null;
    $bodyEditor.value = '';
    $bodyEditor.classList.remove('invalid');
    $bodyHint.classList.remove('err');
    $bodyHint.textContent = 'Will be sent as application/json. Sandbox parses the JSON locally to surface syntax errors before sending.';
    $fileInput.value = '';
    $filePick.classList.remove('has-file');
    $fileLabel.textContent = 'click or drop a JPEG / PNG / WebP leaf photo here';

    $methodPill.className = 'method-pill method-' + currentEp.method;
    $methodPill.textContent = currentEp.method;
    $epDesc.textContent = currentEp.desc;

    // Path params
    const placeholders = [...currentEp.path.matchAll(/\{([^}]+)\}/g)].map(m => m[1]);
    if (placeholders.length) {
      $pathSec.hidden = false;
      $pathParams.innerHTML = '';
      placeholders.forEach(name => {
        const row = document.createElement('div');
        row.className = 'path-param-row';
        row.innerHTML = '<label for="pp-' + name + '">{' + name + '}</label>'
          + '<input type="text" id="pp-' + name + '" class="path-input" data-path-param="' + name + '" placeholder="' + name + ' value">';
        $pathParams.appendChild(row);
      });
      $pathParams.querySelectorAll('input[data-path-param]').forEach(inp => {
        inp.addEventListener('input', () => {
          pathParams[inp.dataset.pathParam] = inp.value;
          refreshCurl();
        });
      });
    } else {
      $pathSec.hidden = true;
    }

    // Query params
    if (currentEp.method === 'GET' || (currentEp.queryHints && currentEp.queryHints.length)) {
      $querySec.hidden = false;
      $queryRows.innerHTML = '';
      // Seed with hints as placeholders, but unfilled
      if (currentEp.queryHints) {
        $queryRows.innerHTML = '<p style="font-family:Fraunces,serif;font-style:italic;'
          + 'font-size:11.5px;color:var(--ink-soft);margin-bottom:6px">hints: '
          + currentEp.queryHints.map(h => '<code style="font-family:JetBrains Mono,monospace;'
              + 'font-size:11px;background:var(--paper-deep);padding:1px 5px;border-radius:3px">'
              + h + '</code>').join(' ')
          + '</p>';
      }
    } else {
      $querySec.hidden = true;
    }

    // Body editor (JSON only for non-file)
    if (currentEp.body === 'json') {
      $bodySec.hidden = false;
      $bodyBadge.textContent = 'JSON';
      $bodyBadge.className = 'badge';
      if (currentEp.bodyExample) {
        $bodyEditor.value = JSON.stringify(currentEp.bodyExample, null, 2);
        bodyValue = $bodyEditor.value;
      }
    } else {
      $bodySec.hidden = true;
    }

    // File picker (only for /predict/full)
    $fileSec.hidden = (currentEp.body !== 'file');

    // Key picker (only for bearer auth)
    $keySec.hidden = (currentEp.auth !== 'bearer');
    if (currentEp.auth === 'bearer' && userKeys.length === 0) {
      loadUserKeys();
    }

    refreshCurl();
  }

  // ── Query rows ────────────────────────────────────────────────────────
  $addQueryBtn.addEventListener('click', () => {
    const row = document.createElement('div');
    row.className = 'kv-row';
    row.innerHTML = '<input type="text" placeholder="key" data-q-key>'
      + '<input type="text" placeholder="value" data-q-val>'
      + '<button type="button" class="btn btn-ghost btn-sm" data-remove>&times;</button>';
    $queryRows.appendChild(row);
    const kInp = row.querySelector('[data-q-key]');
    const vInp = row.querySelector('[data-q-val]');
    kInp.addEventListener('input', updateQueryFromDom);
    vInp.addEventListener('input', updateQueryFromDom);
    row.querySelector('[data-remove]').addEventListener('click', () => {
      row.remove(); updateQueryFromDom();
    });
    kInp.focus();
  });
  function updateQueryFromDom() {
    queryParams = [];
    $queryRows.querySelectorAll('.kv-row').forEach(row => {
      const k = row.querySelector('[data-q-key]').value.trim();
      const v = row.querySelector('[data-q-val]').value;
      if (k) queryParams.push([k, v]);
    });
    refreshCurl();
  }

  // ── File picker ───────────────────────────────────────────────────────
  // Phase 8.H fix · DO NOT forward click to $fileInput here. The HTML
  // wraps the input in a <label class="file-pick">, and native label
  // semantics already dispatch a click to the input. Calling
  // $fileInput.click() *also* opened a SECOND file dialog right after
  // the first closed — that was the "file upload happens twice" bug.
  $filePick.addEventListener('dragover', (e) => { e.preventDefault(); $filePick.style.background = '#e8f5ee'; });
  $filePick.addEventListener('dragleave', () => { $filePick.style.background = ''; });
  $filePick.addEventListener('drop', (e) => {
    e.preventDefault(); $filePick.style.background = '';
    const f = e.dataTransfer.files[0];
    if (f) onFilePicked(f);
  });
  $fileInput.addEventListener('change', () => {
    if ($fileInput.files[0]) onFilePicked($fileInput.files[0]);
  });
  function onFilePicked(f) {
    fileBlob = f;
    $filePick.classList.add('has-file');
    $fileLabel.textContent = f.name + ' · ' + (f.size/1024).toFixed(1) + ' KB · ' + (f.type || 'image');
    refreshCurl();
  }

  // ── Bearer-paste modal (Phase 8.H · replaces window.prompt) ─────────
  // Returns a Promise<string|null>. Resolves with the entered token on
  // "use key" + non-empty input, or null on cancel / Esc / outside-click.
  // The input is type=password so the token doesn't bleed into the DOM
  // as plain text. We trim whitespace and reject pure whitespace as if
  // the user had cancelled.
  function askBearerToken(keyLabel) {
    return new Promise(function (resolve) {
      const modal = document.getElementById('bearer-modal');
      const nameEl = document.getElementById('bearer-key-name');
      const input  = document.getElementById('bearer-input');
      const useBtn = document.getElementById('bearer-use');
      const cancelBtn = document.getElementById('bearer-cancel');
      if (!modal || !input || !useBtn || !cancelBtn) {
        // Defensive: if the modal markup isn't present, fall back to a
        // plain prompt so the user can still proceed (degraded but works).
        const fallback = window.prompt(
          'Paste your API key for ' + (keyLabel || '') + ':');
        resolve(fallback || null);
        return;
      }
      if (nameEl) nameEl.textContent = keyLabel || 'this key';
      input.value = '';
      useBtn.disabled = true;
      modal.hidden = false;
      setTimeout(function () { input.focus(); }, 30);

      function cleanup() {
        modal.hidden = true;
        input.value = '';
        input.removeEventListener('input', onInput);
        input.removeEventListener('keydown', onKey);
        useBtn.removeEventListener('click', onUse);
        cancelBtn.removeEventListener('click', onCancel);
        modal.removeEventListener('click', onBackdrop);
      }
      function onInput() {
        useBtn.disabled = input.value.trim().length === 0;
      }
      function onKey(e) {
        if (e.key === 'Enter' && !useBtn.disabled) { e.preventDefault(); onUse(); }
        if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
      }
      function onUse() {
        const v = input.value.trim();
        cleanup();
        resolve(v || null);
      }
      function onCancel() {
        cleanup();
        resolve(null);
      }
      function onBackdrop(e) {
        // Click on backdrop (outside the card) = cancel.
        if (e.target === modal) onCancel();
      }
      input.addEventListener('input', onInput);
      input.addEventListener('keydown', onKey);
      useBtn.addEventListener('click', onUse);
      cancelBtn.addEventListener('click', onCancel);
      modal.addEventListener('click', onBackdrop);
    });
  }

  // ── Body editor ───────────────────────────────────────────────────────
  $bodyEditor.addEventListener('input', () => {
    bodyValue = $bodyEditor.value;
    // Live-validate JSON
    if (bodyValue.trim() === '') {
      $bodyEditor.classList.remove('invalid');
      $bodyHint.classList.remove('err');
      $bodyHint.textContent = 'Empty body. Will send {} if required.';
    } else {
      try {
        JSON.parse(bodyValue);
        $bodyEditor.classList.remove('invalid');
        $bodyHint.classList.remove('err');
        $bodyHint.textContent = 'Valid JSON.';
      } catch (e) {
        $bodyEditor.classList.add('invalid');
        $bodyHint.classList.add('err');
        $bodyHint.textContent = 'Invalid JSON: ' + e.message;
      }
    }
    refreshCurl();
  });

  // ── User keys (for bearer picker) ─────────────────────────────────────
  async function loadUserKeys() {
    try {
      const r = await fetch('/api/account/keys?status=active', {credentials:'include'});
      const body = await r.json();
      if (body.ok) {
        userKeys = (body.data && body.data.items) || [];
        $keySel.innerHTML = '';
        if (userKeys.length === 0) {
          const o = document.createElement('option');
          o.value = ''; o.textContent = '(no active keys — mint one first)';
          $keySel.appendChild(o);
        } else {
          userKeys.forEach(k => {
            const o = document.createElement('option');
            o.value = k.public_id;
            const last4 = k.last_4 || '????';
            o.textContent = (k.name || '(unnamed)') + ' · ' + k.environment + ' · ******' + last4;
            $keySel.appendChild(o);
          });
          bearerKey = '__SESSION_PROXIED__';   // we can't read plaintext from list; use proxy
        }
      }
    } catch (_) {
      $keySel.innerHTML = '<option value="">(failed to load — refresh)</option>';
    }
  }

  // ── Build request from current state ─────────────────────────────────
  function resolvedPath() {
    let p = currentEp.path;
    Object.keys(pathParams).forEach(k => {
      p = p.replace('{' + k + '}', encodeURIComponent(pathParams[k] || ''));
    });
    return p;
  }
  function resolvedUrl() {
    const p = resolvedPath();
    if (!queryParams.length) return p;
    const qs = queryParams.map(([k, v]) =>
      encodeURIComponent(k) + '=' + encodeURIComponent(v)).join('&');
    return p + '?' + qs;
  }

  // ── cURL preview ─────────────────────────────────────────────────────
  // Phase 8.H · current language tab. Switching tabs just re-renders.
  let currentLang = 'curl';

  function escSh(s) { return String(s).replace(/'/g, "'\\''"); }
  function escPy(s) { return JSON.stringify(String(s)); }
  function escGo(s) { return JSON.stringify(String(s)); }
  function fileName() { return fileBlob ? fileBlob.name : 'leaf.jpg'; }

  // ── cURL ────────────────────────────────────────────────
  function renderCurl() {
    const lines = ['curl -X ' + currentEp.method,
      '  https://YOUR_HOST' + resolvedUrl()];
    if (currentEp.auth === 'bearer') {
      lines.push('  -H "Authorization: Bearer $APIN_KEY"');
    } else if (currentEp.auth.indexOf('session') === 0) {
      lines.push('  -H "Cookie: apin_v2_session=YOUR_SESSION_COOKIE"');
      if (currentEp.auth.indexOf('csrf') !== -1) {
        lines.push('  -H "X-Console-Csrf: YOUR_CSRF_TOKEN"');
      }
    }
    if (currentEp.body === 'json' && bodyValue.trim()) {
      lines.push('  -H "Content-Type: application/json"');
      lines.push("  -d '" + escSh(bodyValue) + "'");
    } else if (currentEp.body === 'file') {
      lines.push('  -F "file=@' + fileName() + '"');
    }
    return lines.join(' \\\n');
  }

  // ── Python (requests) ───────────────────────────────────
  function renderPython() {
    const lines = ['import requests', '', 'url = "https://YOUR_HOST' + resolvedUrl() + '"'];
    const hdrs = [];
    if (currentEp.auth === 'bearer') {
      hdrs.push('"Authorization": f"Bearer {APIN_KEY}"');
    } else if (currentEp.auth.indexOf('session') === 0) {
      hdrs.push('"Cookie": "apin_v2_session=YOUR_SESSION_COOKIE"');
      if (currentEp.auth.indexOf('csrf') !== -1) {
        hdrs.push('"X-Console-Csrf": "YOUR_CSRF_TOKEN"');
      }
    }
    if (currentEp.body === 'file') {
      lines.push('headers = {' + hdrs.join(', ') + '}');
      lines.push('with open(' + escPy(fileName()) + ', "rb") as f:');
      lines.push('    files = {"file": f}');
      lines.push('    r = requests.' + currentEp.method.toLowerCase() + '(url, headers=headers, files=files)');
    } else if (currentEp.body === 'json' && bodyValue.trim()) {
      hdrs.push('"Content-Type": "application/json"');
      lines.push('headers = {' + hdrs.join(', ') + '}');
      lines.push('payload = ' + bodyValue.trim());
      lines.push('r = requests.' + currentEp.method.toLowerCase() + '(url, headers=headers, json=payload)');
    } else {
      lines.push('headers = {' + hdrs.join(', ') + '}');
      lines.push('r = requests.' + currentEp.method.toLowerCase() + '(url, headers=headers)');
    }
    lines.push('r.raise_for_status()');
    lines.push('print(r.json())');
    return lines.join('\n');
  }

  // ── Browser JavaScript (fetch) ──────────────────────────
  function renderJS() {
    const url = '"https://YOUR_HOST' + resolvedUrl() + '"';
    const lines = [];
    const initParts = ['method: "' + currentEp.method + '"'];
    const hdrs = [];
    if (currentEp.auth === 'bearer') {
      hdrs.push('"Authorization": `Bearer ${APIN_KEY}`');
    } else if (currentEp.auth.indexOf('session') === 0) {
      initParts.push('credentials: "include"');
      if (currentEp.auth.indexOf('csrf') !== -1) {
        hdrs.push('"X-Console-Csrf": "YOUR_CSRF_TOKEN"');
      }
    }
    if (currentEp.body === 'file') {
      lines.push('// `fileInput` is an <input type="file"> element');
      lines.push('const fd = new FormData();');
      lines.push('fd.append("file", fileInput.files[0]);');
      initParts.push('headers: { ' + hdrs.join(', ') + ' }');
      initParts.push('body: fd');
    } else if (currentEp.body === 'json' && bodyValue.trim()) {
      hdrs.push('"Content-Type": "application/json"');
      initParts.push('headers: { ' + hdrs.join(', ') + ' }');
      initParts.push('body: JSON.stringify(' + bodyValue.trim() + ')');
    } else {
      initParts.push('headers: { ' + hdrs.join(', ') + ' }');
    }
    lines.push('const r = await fetch(' + url + ', {');
    initParts.forEach((p, i) => {
      lines.push('  ' + p + (i < initParts.length - 1 ? ',' : ''));
    });
    lines.push('});');
    lines.push('if (!r.ok) throw new Error("HTTP " + r.status);');
    lines.push('const data = await r.json();');
    lines.push('console.log(data);');
    return lines.join('\n');
  }

  // ── Node.js (built-in fetch on Node 18+, FormData) ─────
  function renderNode() {
    const lines = ['// Node 18+ — fetch + FormData are global.'];
    if (currentEp.body === 'file') {
      lines.push('import fs from "node:fs";');
      lines.push('');
      lines.push('const fd = new FormData();');
      lines.push('fd.append("file", new Blob([fs.readFileSync(' + escPy(fileName()) + ')]), ' + escPy(fileName()) + ');');
    } else {
      lines.push('');
    }
    const hdrs = [];
    if (currentEp.auth === 'bearer') hdrs.push('"Authorization": `Bearer ${process.env.APIN_KEY}`');
    if (currentEp.auth.indexOf('csrf') !== -1) hdrs.push('"X-Console-Csrf": "YOUR_CSRF_TOKEN"');
    if (currentEp.body === 'json' && bodyValue.trim()) hdrs.push('"Content-Type": "application/json"');

    lines.push('const r = await fetch("https://YOUR_HOST' + resolvedUrl() + '", {');
    lines.push('  method: "' + currentEp.method + '",');
    lines.push('  headers: { ' + hdrs.join(', ') + ' },');
    if (currentEp.body === 'file') {
      lines.push('  body: fd,');
    } else if (currentEp.body === 'json' && bodyValue.trim()) {
      lines.push('  body: JSON.stringify(' + bodyValue.trim() + '),');
    }
    lines.push('});');
    lines.push('console.log(await r.json());');
    return lines.join('\n');
  }

  // ── Go (net/http + mime/multipart) ──────────────────────
  function renderGo() {
    const lines = ['package main', '', 'import (',
      '\t"bytes"',
      '\t"encoding/json"',
      '\t"fmt"',
      '\t"io"',
      '\t"mime/multipart"',
      '\t"net/http"',
      '\t"os"',
      ')',
      '',
      'func main() {'];
    if (currentEp.body === 'file') {
      lines.push('\tvar buf bytes.Buffer');
      lines.push('\tw := multipart.NewWriter(&buf)');
      lines.push('\tfw, _ := w.CreateFormFile("file", ' + escGo(fileName()) + ')');
      lines.push('\tf, _ := os.Open(' + escGo(fileName()) + ')');
      lines.push('\tdefer f.Close()');
      lines.push('\tio.Copy(fw, f)');
      lines.push('\tw.Close()');
      lines.push('\treq, _ := http.NewRequest("' + currentEp.method + '", "https://YOUR_HOST' + resolvedUrl() + '", &buf)');
      lines.push('\treq.Header.Set("Content-Type", w.FormDataContentType())');
    } else if (currentEp.body === 'json' && bodyValue.trim()) {
      lines.push('\tpayload, _ := json.Marshal(' + JSON.stringify(safeJsonParse(bodyValue)) + ')');
      lines.push('\treq, _ := http.NewRequest("' + currentEp.method + '", "https://YOUR_HOST' + resolvedUrl() + '", bytes.NewReader(payload))');
      lines.push('\treq.Header.Set("Content-Type", "application/json")');
    } else {
      lines.push('\treq, _ := http.NewRequest("' + currentEp.method + '", "https://YOUR_HOST' + resolvedUrl() + '", nil)');
    }
    if (currentEp.auth === 'bearer') {
      lines.push('\treq.Header.Set("Authorization", "Bearer " + os.Getenv("APIN_KEY"))');
    } else if (currentEp.auth.indexOf('csrf') !== -1) {
      lines.push('\treq.Header.Set("X-Console-Csrf", "YOUR_CSRF_TOKEN")');
    }
    lines.push('\tresp, err := http.DefaultClient.Do(req)');
    lines.push('\tif err != nil { panic(err) }');
    lines.push('\tdefer resp.Body.Close()');
    lines.push('\tbody, _ := io.ReadAll(resp.Body)');
    lines.push('\tfmt.Println(string(body))');
    lines.push('}');
    return lines.join('\n');
  }

  function safeJsonParse(s) {
    try { return JSON.parse(s); } catch (_) { return null; }
  }

  function refreshCurl() {
    let code;
    if (currentLang === 'python') code = renderPython();
    else if (currentLang === 'js') code = renderJS();
    else if (currentLang === 'node') code = renderNode();
    else if (currentLang === 'go') code = renderGo();
    else code = renderCurl();
    $curlPreview.textContent = code;
  }

  // Wire language tabs (delegated — works after re-render too).
  document.querySelectorAll('.lang-tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      document.querySelectorAll('.lang-tab').forEach(function (t) {
        t.classList.toggle('active', t === tab);
      });
      currentLang = tab.getAttribute('data-lang') || 'curl';
      refreshCurl();
    });
  });

  $copyCurlBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($curlPreview.textContent);
      $copyCurlBtn.classList.add('copied');
      $copyCurlBtn.textContent = 'copied';
      setTimeout(() => {
        $copyCurlBtn.classList.remove('copied');
        $copyCurlBtn.textContent = 'copy';
      }, 1400);
    } catch (_) { showToast('Clipboard blocked. Manually select the text.', 'err'); }
  });

  // ── Send ──────────────────────────────────────────────────────────────
  $sendBtn.addEventListener('click', sendRequest);

  async function sendRequest() {
    // Validate before sending
    const missingPath = [...resolvedPath().matchAll(/\{|%7B/g)];
    if (missingPath.length) {
      showToast('Fill in all path parameters first.', 'err'); return;
    }
    if (currentEp.body === 'file' && !fileBlob) {
      showToast('Pick an image file first.', 'err'); return;
    }
    if (currentEp.auth === 'bearer' && (!$keySel.value)) {
      showToast('Pick an API key — needed for /predict/full.', 'err'); return;
    }
    if (currentEp.body === 'json' && bodyValue.trim()) {
      try { JSON.parse(bodyValue); }
      catch (_) { showToast('Fix the JSON body before sending.', 'err'); return; }
    }

    $sendBtn.disabled = true;
    $sendBtn.textContent = 'Sending…';
    $respStatus.querySelector('.status-pill').className = 'status-pill idle';
    $respStatus.querySelector('.status-pill').textContent = '…';
    $respMeta.textContent = 'request in flight';

    const startedAt = performance.now();
    let req;
    try {
      // Phase 8.H fix · was buildFetchInit() — now must await because the
      // bearer-modal flow returns a Promise. sendRequest is already async.
      req = await buildFetchInit();
    } catch (err) {
      $sendBtn.disabled = false;
      $sendBtn.textContent = 'Send';
      showToast(err.message || 'Could not build request.', 'err');
      return;
    }
    // User cancelled the bearer modal — buildFetchInit returns null in
    // that case (preferred path: we threw in the old code, now we
    // gracefully back out).
    if (!req) {
      $sendBtn.disabled = false;
      $sendBtn.textContent = 'Send';
      return;
    }

    let resp, respText, respJson, respHeadersObj = {};
    try {
      resp = await fetch(req.url, req.init);
      respHeadersObj = {};
      resp.headers.forEach((v, k) => { respHeadersObj[k] = v; });
      respText = await resp.text();
      try { respJson = JSON.parse(respText); } catch (_) { respJson = null; }
    } catch (err) {
      $sendBtn.disabled = false; $sendBtn.textContent = 'Send';
      $respStatus.querySelector('.status-pill').className = 'status-pill s-5xx';
      $respStatus.querySelector('.status-pill').textContent = 'network';
      $respMeta.textContent = 'network error: ' + (err.message || err);
      $respBody.classList.remove('empty');
      $respBody.textContent = String(err.message || err);
      return;
    }
    const elapsed = Math.round(performance.now() - startedAt);

    $sendBtn.disabled = false; $sendBtn.textContent = 'Send';
    const sp = $respStatus.querySelector('.status-pill');
    sp.className = 'status-pill ' + statusClass(resp.status);
    sp.textContent = String(resp.status);
    $respMeta.textContent = (resp.statusText || '') + ' · ' + elapsed + ' ms · ' + respText.length + ' B';

    // Body
    if (respJson !== null) {
      $respBody.classList.remove('empty');
      $respBody.innerHTML = syntaxHighlightJson(respJson);
    } else if (respText) {
      $respBody.classList.remove('empty');
      $respBody.textContent = respText;
    } else {
      $respBody.classList.add('empty');
      $respBody.textContent = '(empty body)';
    }
    // Headers
    const tbody = document.createElement('tbody');
    Object.keys(respHeadersObj).sort().forEach(k => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + escHtml(k) + '</td><td>' + escHtml(respHeadersObj[k]) + '</td>';
      tbody.appendChild(tr);
    });
    $respHeaders.innerHTML = ''; $respHeaders.appendChild(tbody);

    // Record history
    pushHistory({
      ts: Date.now(), method: currentEp.method, url: req.url,
      status: resp.status, elapsed_ms: elapsed,
      preview: respText.slice(0, 400),
    });
  }

  // Phase 8.H fix · MUST be async — askBearerToken returns a Promise and
  // we await it below. Top-level await is illegal in non-module scripts,
  // so the await landed outside an async fn and bricked the entire script
  // (Uncaught SyntaxError → console_sandbox.js fails to parse → nothing
  // on the sandbox page wires up). Adding `async` here fixes parsing.
  async function buildFetchInit() {
    const url = resolvedUrl();
    const headers = {'Accept': 'application/json'};
    let body = undefined;

    if (currentEp.auth === 'bearer') {
      // Phase 8.H · replaced window.prompt() with the bearer-modal flow.
      // APIN does not store plaintext bearer tokens, so the user must paste
      // the original. We cache in sessionStorage keyed by public_id so the
      // user only types it once per tab.
      const pasted = sessionStorage.getItem('sandbox_bearer_for_' + $keySel.value);
      let token = pasted;
      if (!token) {
        token = await askBearerToken(
          $keySel.options[$keySel.selectedIndex].textContent);
        if (!token) throw new Error('Bearer token required for /predict/full.');
        sessionStorage.setItem('sandbox_bearer_for_' + $keySel.value, token);
      }
      headers['Authorization'] = 'Bearer ' + token;
    } else if (currentEp.auth.indexOf('csrf') !== -1) {
      headers['X-Console-Csrf'] = csrf;
    }

    if (currentEp.body === 'json' && bodyValue.trim()) {
      headers['Content-Type'] = 'application/json';
      body = bodyValue;
    } else if (currentEp.body === 'file') {
      const fd = new FormData();
      // Phase 8.H fix · the FastAPI handler binds the multipart field name
      // `file` (see scripts/apin/section8_apin_server.py). The earlier
      // code sent `image` which gave a 400 "Field required" for
      // `body.file`. The cURL/Python/JS/Go renderers already write
      // `file=@…` correctly; this line was the one source-of-truth drift.
      fd.append('file', fileBlob);
      body = fd;
    }
    return {
      url,
      init: {
        method: currentEp.method,
        credentials: 'include',
        headers,
        body,
      }
    };
  }

  // ── History (localStorage) ───────────────────────────────────────────
  const HIST_KEY = 'apin_sandbox_history_v1';
  let history = [];
  try { history = JSON.parse(localStorage.getItem(HIST_KEY) || '[]'); }
  catch (_) { history = []; }

  function pushHistory(entry) {
    history.unshift(entry);
    history = history.slice(0, 20);
    try { localStorage.setItem(HIST_KEY, JSON.stringify(history)); } catch (_) {}
    renderHistory();
  }
  function renderHistory() {
    if (history.length === 0) {
      $histList.innerHTML = '<p class="hist-empty">Recent sandbox calls appear here.</p>';
      $histCount.textContent = 'empty';
      return;
    }
    $histCount.textContent = history.length + ' call' + (history.length === 1 ? '' : 's');
    $histList.innerHTML = history.map(h => {
      const meth = '<span class="h-method method-' + h.method + '">' + h.method + '</span>';
      const stat = '<span class="h-status ' + statusClass(h.status) + '">' + h.status + '</span>';
      return '<div class="hist-row">' + meth
        + '<span class="h-path">' + escHtml(h.url) + '</span>'
        + stat + '<span class="h-time">' + fmtAgo(h.ts) + '</span></div>';
    }).join('');
  }
  $clearHistBtn.addEventListener('click', () => {
    history = []; localStorage.removeItem(HIST_KEY); renderHistory();
    showToast('History cleared.');
  });

  // ── Response tabs ────────────────────────────────────────────────────
  document.querySelectorAll('.response-tabs .tab').forEach(t => {
    t.addEventListener('click', () => {
      document.querySelectorAll('.response-tabs .tab').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      document.querySelectorAll('.response-pane').forEach(p =>
        p.classList.toggle('active', p.dataset.respPane === t.dataset.respTab));
    });
  });

  // ── Bootstrap ────────────────────────────────────────────────────────
  buildEndpointDropdown();
  $epSel.addEventListener('change', () => setEndpoint(parseInt($epSel.value, 10)));
  setEndpoint(0);
  renderHistory();

})();