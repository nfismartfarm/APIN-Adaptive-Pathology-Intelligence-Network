/* 9.O.2 · APIN.modal — dedicated in-page modal for key actions.
 *
 * A focus-trapped, ESC/backdrop-dismissable dialog rendered into a portal at
 * <body>. NO URL hash (unlike the chart lightbox), so it never collides with
 * the #lightbox= flow and never lands in browser history. Paper-ink styling,
 * hand-drawn sprite icons.
 *
 *   const m = APIN.modal.open({
 *     icon: 'i-refresh', title: 'Rotate token', subtitle: '…', danger: false,
 *     body: (el) => { … build form … },            // string | Node | fn(el)
 *     actions: [
 *       { label: 'Cancel', kind: 'ghost' },         // closeOnClick default true
 *       { label: 'Rotate', kind: 'primary', busyLabel: 'Rotating…',
 *         onClick: async (ctx) => { … ; ctx.close(); } }
 *     ],
 *     onClose: () => {}
 *   });
 *
 * ctx passed to onClick: { root, close, setError, setBusy, modal }.
 * Helpers: APIN.modal.confirm({…}) and APIN.modal.form({…}).
 */
(function () {
  'use strict';
  const M = {};
  let _stack = [];           // open modals (supports nesting)

  function injectCSS() {
    if (document.getElementById('apin-modal-css')) return;
    const s = document.createElement('style'); s.id = 'apin-modal-css';
    s.textContent = `
.apm-backdrop{position:fixed;inset:0;z-index:1000;display:flex;align-items:center;justify-content:center;background:rgba(26,22,18,.42);backdrop-filter:blur(2px);opacity:0;transition:opacity .18s ease;padding:20px}
.apm-backdrop.in{opacity:1}
.apm{width:100%;max-width:480px;max-height:calc(100vh - 48px);display:flex;flex-direction:column;background:var(--paper,#efe7d4);border:1px solid var(--paper-edge,#d8cdb2);border-radius:14px;box-shadow:0 24px 64px rgba(20,16,12,.34);transform:translateY(10px) scale(.98);opacity:0;transition:transform .2s cubic-bezier(.2,.9,.3,1.1),opacity .2s ease;overflow:hidden}
.apm.in{transform:none;opacity:1}
.apm.apm-danger{border-color:#d9a99a}
.apm-head{display:flex;align-items:center;gap:11px;padding:16px 18px 12px}
.apm-ic{width:34px;height:34px;flex:none;display:flex;align-items:center;justify-content:center;border-radius:9px;background:var(--paper-deep,#e7dcc4);color:var(--ink,#1a1612)}
.apm-danger .apm-ic{background:#f3ddd6;color:#b3402f}
.apm-ic svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.apm-ht{flex:1;min-width:0}
.apm-title{font:600 16px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin:0}
.apm-sub{font:11.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:2px}
.apm-x{flex:none;width:28px;height:28px;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);border-radius:7px;cursor:pointer;color:var(--ink-soft,#5b5446);display:flex;align-items:center;justify-content:center}
.apm-x:hover{background:#fff}.apm-x svg{width:13px;height:13px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round}
.apm-body{padding:4px 18px 16px;overflow:auto;font:13px/1.55 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}
.apm-body p{margin:0 0 10px}.apm-body strong{color:var(--ink,#1a1612)}
.apm-err{display:none;margin:0 18px 12px;background:#f6e3dd;border:1px solid #d9a99a;color:#92301f;border-radius:8px;padding:8px 11px;font:11.5px 'JetBrains Mono',monospace}
.apm-err.show{display:block}
.apm-foot{display:flex;align-items:center;justify-content:flex-end;gap:9px;padding:12px 18px 16px;border-top:1px solid var(--paper-edge,#e3d9c0)}
.apm-btn{font:600 12.5px 'JetBrains Mono',monospace;padding:8px 16px;border-radius:9px;cursor:pointer;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);color:var(--ink,#1a1612);display:inline-flex;align-items:center;gap:7px}
.apm-btn:hover{background:#fff}
.apm-btn.primary{background:var(--ink,#1a1612);color:#efe7d4;border-color:var(--ink,#1a1612)}
.apm-btn.primary:hover{background:var(--accent-deep,#2d6a4f);border-color:var(--accent-deep,#2d6a4f)}
.apm-btn.danger{background:#b3402f;color:#fff;border-color:#b3402f}.apm-btn.danger:hover{background:#9a3526}
.apm-btn[disabled]{opacity:.45;pointer-events:none}
.apm-btn .apm-spin{width:13px;height:13px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:apmSpin .7s linear infinite}
@keyframes apmSpin{to{transform:rotate(360deg)}}
/* form field helpers */
.apm-field{margin-bottom:13px}
.apm-field label{display:block;font:600 11px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);text-transform:uppercase;letter-spacing:.04em;margin-bottom:5px}
.apm-field .hint{font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:4px}
.apm-input,.apm-ta,.apm-sel{width:100%;font:13px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);background:#fff;border:1px solid var(--paper-edge,#d8cdb2);border-radius:8px;padding:8px 10px}
.apm-input:focus,.apm-ta:focus,.apm-sel:focus{outline:none;border-color:var(--accent,#52b788);box-shadow:0 0 0 3px rgba(82,183,136,.18)}
.apm-ta{min-height:74px;resize:vertical;line-height:1.5}
.apm-secret{display:flex;align-items:center;gap:8px;background:var(--ink,#1a1612);border-radius:9px;padding:10px 12px;margin:6px 0}
.apm-secret code{flex:1;min-width:0;color:#8fe0b4;font:12px 'JetBrains Mono',monospace;word-break:break-all}
.apm-copy{flex:none;border:1px solid #3a4a3f;background:#222;color:#cfe;border-radius:7px;padding:6px 9px;cursor:pointer;font:11px 'JetBrains Mono',monospace;display:inline-flex;gap:5px;align-items:center}
.apm-copy svg{width:12px;height:12px;fill:none;stroke:currentColor;stroke-width:2}
.apm-warn{background:#fbeccd;border:1px solid #e0c98c;color:#8a6d1f;border-radius:8px;padding:8px 11px;font:11.5px 'JetBrains Mono',monospace;margin-bottom:12px}
/* one-time token reveal */
.apm-ott-warn{background:#fcf3da;border:1px solid #e6d199;border-radius:9px;padding:10px 12px;font:11.5px/1.55 'JetBrains Mono',monospace;color:#7a4e0a;margin-bottom:12px}
.apm-ott-warn b{color:#92301f}
.apm-ott-saveas{display:flex;align-items:center;gap:8px;margin:10px 0 4px}
.apm-ott-lbl{font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);text-transform:uppercase;letter-spacing:.04em}
.apm-ott-fmt{border:1.2px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);border-radius:8px;padding:5px 13px;font:600 11.5px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);cursor:pointer;transition:background .15s,border-color .15s,transform .12s}
.apm-ott-fmt:hover{background:#fff;border-color:var(--ink-mute,#8b8273);transform:translateY(-1px)}
.apm-ott-meta{margin-top:13px;border-top:1px solid var(--paper-edge,#e3d9c0);padding-top:11px;display:flex;flex-direction:column;gap:5px}
.apm-ott-row{display:grid;grid-template-columns:96px 1fr;gap:12px;font:11.5px 'JetBrains Mono',monospace}
.apm-ott-row span:first-child{color:var(--ink-mute,#9a917d);text-transform:uppercase;letter-spacing:.03em;font-size:10.5px}
.apm-ott-row span:last-child{color:var(--ink,#1a1612);word-break:break-word}
.apm-ott-ack{display:flex;align-items:center;gap:9px;margin-right:auto;font:11.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);cursor:pointer}
@media (prefers-reduced-motion:reduce){.apm,.apm-backdrop{transition:none}.apm-spin{animation-duration:0s}}
`;
    document.head.appendChild(s);
  }

  function svg(id) { return '<svg viewBox="0 0 24 24" aria-hidden="true"><use href="#' + id + '"/></svg>'; }

  function open(opts) {
    injectCSS();
    opts = opts || {};
    const prevFocus = document.activeElement;
    const backdrop = document.createElement('div');
    backdrop.className = 'apm-backdrop';
    const dialog = document.createElement('div');
    dialog.className = 'apm' + (opts.danger ? ' apm-danger' : '');
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    const titleId = 'apm-t-' + Math.random().toString(36).slice(2, 7);
    dialog.setAttribute('aria-labelledby', titleId);
    if (opts.width) dialog.style.maxWidth = (typeof opts.width === 'number' ? opts.width + 'px' : opts.width);

    dialog.innerHTML =
      '<div class="apm-head">' +
      (opts.icon ? '<span class="apm-ic">' + svg(opts.icon) + '</span>' : '') +
      '<div class="apm-ht"><h3 class="apm-title" id="' + titleId + '">' + esc(opts.title || '') + '</h3>' +
      (opts.subtitle ? '<div class="apm-sub">' + esc(opts.subtitle) + '</div>' : '') + '</div>' +
      '<button class="apm-x" aria-label="Close">' + svg('i-x') + '</button>' +
      '</div>' +
      '<div class="apm-err" role="alert"></div>' +
      '<div class="apm-body"></div>' +
      '<div class="apm-foot"></div>';
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);

    const bodyEl = dialog.querySelector('.apm-body');
    const footEl = dialog.querySelector('.apm-foot');
    const errEl = dialog.querySelector('.apm-err');

    // body: string | Node | fn(el)
    const b = opts.body;
    if (typeof b === 'function') { try { b(bodyEl); } catch (e) { bodyEl.textContent = 'error'; } }
    else if (b instanceof Node) bodyEl.appendChild(b);
    else if (typeof b === 'string') bodyEl.innerHTML = b;

    let busy = false;
    const ctrl = {
      root: bodyEl, dialog: dialog,
      close: () => destroy(),
      setError: (msg) => { if (msg) { errEl.textContent = msg; errEl.classList.add('show'); } else { errEl.classList.remove('show'); } },
      setBusy: (v) => { busy = !!v; updateBtns(); },
      query: (sel) => bodyEl.querySelector(sel),
      footEl: footEl,
    };

    const btns = [];
    (opts.actions || []).forEach((a, i) => {
      const btn = document.createElement('button');
      btn.className = 'apm-btn ' + (a.kind || 'ghost');
      btn.dataset.idx = i;
      btn.innerHTML = esc(a.label || 'OK');
      btn._spec = a;
      btn.addEventListener('click', async () => {
        if (busy) return;
        if (a.onClick) {
          ctrl.setError(null);
          const spin = (a.kind === 'primary' || a.kind === 'danger');
          if (spin) { btn._label = btn.innerHTML; btn.innerHTML = '<span class="apm-spin"></span>' + esc(a.busyLabel || a.label || ''); }
          ctrl.setBusy(true);
          try {
            const r = await a.onClick(ctrl);
            ctrl.setBusy(false); if (spin) btn.innerHTML = btn._label;
            if (r !== false && a.closeOnClick !== false) destroy();
          } catch (e) {
            ctrl.setBusy(false); if (spin) btn.innerHTML = btn._label;
            ctrl.setError((e && e.message) ? e.message : 'Something went wrong.');
          }
        } else if (a.closeOnClick !== false) { destroy(); }
      });
      footEl.appendChild(btn); btns.push(btn);
    });
    function updateBtns() { btns.forEach(b => { const a = b._spec || {}; b.disabled = busy || (a.disabled && a.disabled(ctrl)); }); }
    ctrl.refreshButtons = updateBtns;
    updateBtns();

    dialog.querySelector('.apm-x').addEventListener('click', () => { if (!busy) destroy(); });
    backdrop.addEventListener('mousedown', (e) => { if (e.target === backdrop && !busy) destroy(); });

    function onKey(e) {
      if (e.key === 'Escape' && !busy) { e.preventDefault(); destroy(); return; }
      if (e.key === 'Tab') {
        const f = dialog.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        const list = Array.prototype.filter.call(f, el => !el.disabled && el.offsetParent !== null);
        if (!list.length) return;
        const first = list[0], last = list[list.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    }
    document.addEventListener('keydown', onKey, true);

    let destroyed = false;
    function destroy() {
      if (destroyed) return; destroyed = true;
      document.removeEventListener('keydown', onKey, true);
      backdrop.classList.remove('in'); dialog.classList.remove('in');
      setTimeout(() => { try { backdrop.remove(); } catch (e) { } }, 200);
      _stack = _stack.filter(x => x !== ctrl);
      try { if (prevFocus && prevFocus.focus) prevFocus.focus(); } catch (e) { }
      if (opts.onClose) try { opts.onClose(); } catch (e) { }
    }

    requestAnimationFrame(() => {
      backdrop.classList.add('in'); dialog.classList.add('in');
      const focusable = dialog.querySelector('.apm-body input, .apm-body select, .apm-body textarea, .apm-foot .primary, .apm-foot .danger, .apm-x');
      if (focusable) try { focusable.focus(); } catch (e) { }
    });
    _stack.push(ctrl);
    return ctrl;
  }

  // ── confirm helper ────────────────────────────────────────────────────────
  function confirm(opts) {
    opts = opts || {};
    const requireText = opts.requireText;
    let inputEl = null;
    const m = open({
      icon: opts.icon || (opts.danger ? 'i-warning' : 'i-help-circle'),
      title: opts.title || 'Are you sure?',
      subtitle: opts.subtitle,
      danger: opts.danger,
      body: (el) => {
        el.innerHTML = (opts.message ? '<p>' + opts.message + '</p>' : '');
        if (requireText) {
          const f = document.createElement('div'); f.className = 'apm-field';
          f.innerHTML = '<label>Type <strong>' + esc(requireText) + '</strong> to confirm</label>' +
            '<input class="apm-input" autocomplete="off" spellcheck="false" placeholder="' + esc(requireText) + '">';
          el.appendChild(f);
          inputEl = f.querySelector('input');
          inputEl.addEventListener('input', () => m && m.refreshButtons && m.refreshButtons());
        }
      },
      actions: [
        { label: opts.cancelLabel || 'Cancel', kind: 'ghost' },
        {
          label: opts.confirmLabel || (opts.danger ? 'Delete' : 'Confirm'),
          kind: opts.danger ? 'danger' : 'primary',
          busyLabel: opts.busyLabel,
          disabled: () => requireText ? (!inputEl || inputEl.value.trim() !== requireText) : false,
          onClick: async (ctx) => { if (opts.onConfirm) return await opts.onConfirm(ctx); },
        },
      ],
      onClose: opts.onClose,
    });
    return m;
  }

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  // ── Shared sudo-aware writer ────────────────────────────────────────────
  // Reads the CSRF token LIVE from <meta> each call (so it never goes stale
  // across modules), does the sudo step-up modal on 403 sudo_required,
  // captures the rotated token, updates <meta>, and retries once.
  function _metaCsrf() {
    const m = document.querySelector('meta[name="csrf-token"]');
    return m ? m.content : '';
  }
  function _requireSudo() {
    return new Promise((resolve) => {
      let settled = false;
      const finish = (v) => { if (!settled) { settled = true; resolve(v); } };
      open({
        icon: 'i-shield-alert', title: 'Confirm it’s you',
        subtitle: 'A quick password check protects this change',
        body: (el) => {
          el.innerHTML = '<div class="apm-field"><label>Account password</label>' +
            '<input class="apm-input" id="apm-sudo-pw" type="password" autocomplete="current-password" placeholder="••••••••"></div>' +
            '<div class="hint">You stay signed in. This just re-confirms it’s really you.</div>';
          const inp = el.querySelector('#apm-sudo-pw'); if (inp) setTimeout(() => inp.focus(), 30);
        },
        onClose: () => finish(false),
        actions: [
          { label: 'Cancel', kind: 'ghost' },
          { label: 'Confirm', kind: 'primary', busyLabel: 'Verifying…', closeOnClick: false,
            onClick: async (ctx) => {
              const pw = (ctx.query('#apm-sudo-pw') || {}).value || '';
              if (!pw) { ctx.setError('Password required.'); return false; }
              const r = await fetch('/api/account/sudo', { method: 'POST', credentials: 'include',
                headers: { 'Accept': 'application/json', 'Content-Type': 'application/json', 'X-Console-Csrf': _metaCsrf() },
                body: JSON.stringify({ password: pw }) });
              const b = await r.json().catch(() => ({}));
              if (r.status === 200 && b.ok) {
                const nc = b.data && b.data.csrf_token;
                if (nc) { const m = document.querySelector('meta[name="csrf-token"]'); if (m) m.content = nc; }
                settled = true; resolve(true); ctx.close(); return true;
              }
              ctx.setError((b.error && b.error.message) || 'That password did not match.');
              return false;
            } },
        ],
      });
    });
  }
  async function sudoFetch(url, method, payload, _retried) {
    const opts = { method: method, credentials: 'include',
      headers: { 'X-Console-Csrf': _metaCsrf(), 'Accept': 'application/json' } };
    if (payload !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(payload);
    }
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (r.status === 403 && body && body.error && body.error.code === 'sudo_required' && !_retried) {
      const ok = await _requireSudo();
      if (ok) return sudoFetch(url, method, payload, true);
    }
    // A successful mutation may have produced an alert — pop it now rather than
    // waiting for the toast's 20s poll.
    if (r.ok && method && method.toUpperCase() !== 'GET') {
      try { window.dispatchEvent(new CustomEvent('apin:alerts:refresh')); } catch (e) {}
    }
    return { status: r.status, body: body };
  }
  function writeErr(body) { return (body && body.error && body.error.message) || 'Request failed. Please try again.'; }

  // ── One-time token reveal ───────────────────────────────────────────────
  // Rewrites an open modal's body+footer into the "save this token now" view
  // used after minting OR rotating a key. opts: {name, env, public_id, scopes,
  // secret, expires, rotating, onDone}. Reused so mint / rotate / create-in-
  // group all look identical.
  function oneTimeToken(ctx, opts) {
    opts = opts || {};
    const sec = opts.secret || '';
    const rows = [
      ['name', opts.name || ''],
      ['public id', opts.public_id || ''],
      ['environment', opts.env || ''],
      ['scopes', (opts.scopes || []).join(', ') || '—'],
      ['expires', opts.expires || 'no expiry'],
    ];
    ctx.root.innerHTML =
      '<div class="apm-ott">' +
      (opts.rotating ? '<div class="apm-warn">Rotated — the previous token keeps working during the grace window, then stops.</div>' : '') +
      '<div class="apm-ott-warn"><b>This is the only time it will be shown.</b> After you close this dialog the plaintext is gone forever — paste it into your password manager or secrets vault now.</div>' +
      '<div class="apm-secret"><code>' + esc(sec || '(token hidden by server)') + '</code>' +
      '<button class="apm-copy" id="apm-ott-copy" type="button">' + svg('i-clipboard') + 'copy</button></div>' +
      '<div class="apm-ott-saveas"><span class="apm-ott-lbl">save as</span>' +
      '<button class="apm-ott-fmt" data-fmt="env" type="button">.env</button>' +
      '<button class="apm-ott-fmt" data-fmt="txt" type="button">.txt</button>' +
      '<button class="apm-ott-fmt" data-fmt="json" type="button">.json</button></div>' +
      '<div class="apm-ott-meta">' + rows.map(r =>
        '<div class="apm-ott-row"><span>' + esc(r[0]) + '</span><span>' + esc(r[1]) + '</span></div>').join('') +
      '</div></div>';
    const cp = ctx.root.querySelector('#apm-ott-copy');
    if (cp && sec) cp.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(sec); cp.innerHTML = svg('i-check') + 'copied'; setTimeout(() => { cp.innerHTML = svg('i-clipboard') + 'copy'; }, 1400); } catch (e) {}
    });
    ctx.root.querySelectorAll('.apm-ott-fmt').forEach(b => b.addEventListener('click', () => {
      const fmt = b.getAttribute('data-fmt');
      const base = (opts.name || 'apin-key').replace(/[^a-zA-Z0-9._-]+/g, '-');
      let content, fname;
      if (fmt === 'env') { content = 'APIN_API_KEY=' + sec + '\n'; fname = base + '.env'; }
      else if (fmt === 'json') { content = JSON.stringify({ name: opts.name, public_id: opts.public_id, environment: opts.env, token: sec }, null, 2); fname = base + '.json'; }
      else { content = sec + '\n'; fname = base + '.txt'; }
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = fname;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }));
    ctx.footEl.innerHTML =
      '<label class="apm-ott-ack"><input type="checkbox" class="apin-check" id="apm-ott-ack"> I’ve saved my token in a secure place.</label>' +
      '<button class="apm-btn primary" id="apm-ott-done" disabled>Done</button>';
    const ack = ctx.footEl.querySelector('#apm-ott-ack');
    const done = ctx.footEl.querySelector('#apm-ott-done');
    ack.addEventListener('change', () => { done.disabled = !ack.checked; });
    done.addEventListener('click', () => { ctx.close(); if (opts.onDone) try { opts.onDone(); } catch (e) {} });
  }

  M.open = open; M.confirm = confirm; M.svg = svg; M._esc = esc;
  M.sudoFetch = sudoFetch; M.writeErr = writeErr; M.oneTimeToken = oneTimeToken;
  window.APIN = window.APIN || {};
  window.APIN.modal = M;
  window.APIN.sudoFetch = sudoFetch;
})();
