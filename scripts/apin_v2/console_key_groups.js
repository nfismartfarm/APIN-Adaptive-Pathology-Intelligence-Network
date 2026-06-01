/* console_key_groups.js — Phase 2 · API key groups UI for the keys page.
 *
 * Mounts a "New group" button + a Groups panel above the key cards. Uses the
 * shared APIN.modal + APIN.sudoFetch (sudo step-up + CSRF rotation handled
 * there). Self-contained: does nothing if the keys page markup isn't present.
 */
(function () {
  'use strict';
  if (!document.getElementById('cards')) return;          // not the keys page
  const MD = window.APIN && window.APIN.modal;
  if (!MD) { console.warn('[groups] APIN.modal missing'); return; }
  const esc = MD._esc;
  const sudoFetch = window.APIN.sudoFetch;

  const SCOPE_GROUPS = [
    { key: 'predictions', label: 'Predictions', scopes: ['predict:read', 'predict:write'] },
    { key: 'reports', label: 'Reports', scopes: ['reports:read', 'reports:write'] },
    { key: 'models', label: 'Models', scopes: ['models:read'] },
    { key: 'feedback', label: 'Feedback', scopes: ['feedback:write'] },
    { key: 'account', label: 'Usage & account', scopes: ['usage:read', 'account:read'] },
  ];
  const SCOPE_DESC = {
    'predict:read': 'read inference results', 'predict:write': 'run new predictions',
    'reports:read': 'read reports', 'reports:write': 'create & export reports',
    'models:read': 'read model cards', 'feedback:write': 'submit corrections',
    'usage:read': 'read analytics & usage', 'account:read': 'read key & account info',
  };
  const KNOWN = SCOPE_GROUPS.reduce((a, g) => a.concat(g.scopes), []);

  injectCSS();

  // ── data ──────────────────────────────────────────────────────────────
  async function gGet(url) {
    const r = await fetch(url, { credentials: 'include', headers: { 'Accept': 'application/json' } });
    const j = await r.json().catch(() => ({}));
    return j && j.data ? j.data : {};
  }

  // ── scope checklist ─────────────────────────────────────────────────────
  // Category-level by default: one row per category with a select-all checkbox.
  // Each category can EXPAND to reveal its individual permissions for granular
  // control. The category row is always shown (never hidden) — only the
  // detailed scopes expand. selected: granted scopes. lockedOn: the group floor
  // (always-checked + disabled) inside a ceiling editor.
  function scopeChecklistHTML(selected, lockedOn) {
    selected = selected || []; lockedOn = lockedOn || [];
    const has = (s) => selected.indexOf(s) >= 0 || lockedOn.indexOf(s) >= 0;
    const lock = (s) => lockedOn.indexOf(s) >= 0;
    return '<div class="kg-scopes">' + SCOPE_GROUPS.map(g => {
      const on = g.scopes.filter(has).length;
      const single = g.scopes.length === 1;
      return '<div class="kg-grp" data-grp="' + esc(g.key) + '">' +
        '<div class="kg-grp-h">' +
          '<label class="kg-grp-label"><input type="checkbox" class="apin-check" data-grp-check="' + esc(g.key) + '"> <b>' + esc(g.label) + '</b></label>' +
          (single ? '<span class="kg-grp-sum">1 permission</span>'
                  : '<button type="button" class="kg-expand" aria-expanded="false"><span class="kg-grp-sum">' + on + '/' + g.scopes.length + '</span><span class="kg-xt">individual permissions</span>' + MD.svg('i-chevron-right') + '</button>') +
        '</div>' +
        '<div class="kg-grp-body" hidden>' +
          g.scopes.map(s => '<label class="kg-row"><input type="checkbox" class="apin-check" data-scope="' + esc(s) + '"' +
            (has(s) ? ' checked' : '') + (lock(s) ? ' disabled' : '') + '> <code>' + esc(s) + '</code>' +
            '<span class="kg-d">' + esc(SCOPE_DESC[s] || '') + '</span></label>').join('') +
        '</div></div>';
    }).join('') + '</div>';
  }
  function wireScopeChecklist(root) {
    root.querySelectorAll('.kg-grp').forEach(grp => {
      const exp = grp.querySelector('.kg-expand');
      const body = grp.querySelector('.kg-grp-body');
      const parent = grp.querySelector('input[data-grp-check]');
      const sum = grp.querySelector('.kg-grp-sum');
      const kids = Array.prototype.slice.call(grp.querySelectorAll('input[data-scope]'));
      const enabled = kids.filter(k => !k.disabled);
      if (exp) exp.addEventListener('click', () => {
        const willOpen = body.hidden; body.hidden = !willOpen;
        grp.classList.toggle('open', willOpen);
        exp.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
      });
      const sync = () => {
        const on = enabled.filter(k => k.checked).length;
        parent.checked = enabled.length > 0 && on === enabled.length;
        parent.indeterminate = on > 0 && on < enabled.length;
        if (sum && kids.length > 1) sum.textContent = kids.filter(k => k.checked).length + '/' + kids.length;
      };
      sync();
      parent.addEventListener('change', () => { enabled.forEach(k => k.checked = parent.checked); sync(); });
      kids.forEach(k => k.addEventListener('change', sync));
    });
  }
  function collectScopes(root) {
    return Array.prototype.map.call(root.querySelectorAll('input[data-scope]:checked'),
      i => i.getAttribute('data-scope'));
  }

  // ── panel render ────────────────────────────────────────────────────────
  function ensureMount() {
    let panel = document.getElementById('kg-panel');
    if (panel) return panel;
    panel = document.createElement('section');
    panel.id = 'kg-panel'; panel.className = 'kg-panel';
    const cards = document.getElementById('cards');
    cards.parentNode.insertBefore(panel, cards);
    return panel;
  }

  const KG_TIP = 'A group shares one permission set across its keys. Members inherit it (locked) or override within a ceiling (special).';

  // The "+ new group" button lives in the action bar next to "+ new key"
  // (same primary style). The explanation rides along as a hover tooltip.
  function mountButton() {
    if (document.getElementById('kg-new-btn')) return;
    const bar = document.querySelector('.action-bar');
    if (!bar) return;
    const btn = document.createElement('button');
    btn.className = 'btn btn-primary'; btn.id = 'kg-new-btn'; btn.type = 'button';
    btn.innerHTML = '<span class="icon">+</span> new group';
    btn.addEventListener('click', openCreate);
    const after = document.getElementById('btn-new');
    if (after && after.parentNode === bar) bar.insertBefore(btn, after.nextSibling);
    else bar.appendChild(btn);
    let tip = document.getElementById('kg-tip');
    if (!tip) { tip = document.createElement('div'); tip.id = 'kg-tip'; tip.className = 'kg-tip'; tip.textContent = KG_TIP; document.body.appendChild(tip); }
    const show = () => { const r = btn.getBoundingClientRect(); tip.style.left = (r.left + window.scrollX) + 'px'; tip.style.top = (r.bottom + window.scrollY + 8) + 'px'; tip.classList.add('show'); };
    const hide = () => tip.classList.remove('show');
    btn.addEventListener('mouseenter', show); btn.addEventListener('mouseleave', hide);
    btn.addEventListener('focus', show); btn.addEventListener('blur', hide);
  }

  // Keep the ribbon group-count and the keys-page group filter in sync with
  // the current group set (both live in keys.html, owned by keys.js).
  function syncChrome(groups) {
    const cnt = document.getElementById('count-groups');
    if (cnt) cnt.textContent = groups.length;
    const sel = document.getElementById('f-group');
    if (sel) {
      const cur = sel.value;
      sel.innerHTML = '<option value="all">all groups</option>' +
        '<option value="__none__">ungrouped</option>' +
        groups.map(g => '<option value="' + esc(g.name) + '">' + esc(g.name) + '</option>').join('');
      // restore selection if still valid
      if (Array.prototype.some.call(sel.options, o => o.value === cur)) sel.value = cur;
    }
  }

  async function renderPanel() {
    mountButton();
    const panel = ensureMount();
    let groups = [];
    try { groups = (await gGet('/api/account/key-groups')).groups || []; }
    catch (e) { panel.innerHTML = ''; panel.style.display = 'none'; syncChrome([]); return; }
    syncChrome(groups);
    if (!groups.length) { panel.innerHTML = ''; panel.style.display = 'none'; return; }
    panel.style.display = '';
    panel.innerHTML = '<div class="kg-cap">API groups</div><div id="kg-list" class="kg-list">' +
      groups.map(g =>
        '<button class="kg-card" data-gid="' + g.id + '">' +
        '<div class="kg-card-top"><span class="kg-name">' + esc(g.name) + '</span>' +
        '<span class="kg-count">' + g.member_count + ' key' + (g.member_count === 1 ? '' : 's') + '</span></div>' +
        '<div class="kg-chips">' + (g.scopes.length ? g.scopes.map(s => '<code>' + esc(s) + '</code>').join('') :
          '<span class="kg-d">no scopes</span>') + '</div></button>').join('') + '</div>';
    panel.querySelectorAll('.kg-card').forEach(c => c.addEventListener('click', () => {
      openEditor(parseInt(c.getAttribute('data-gid'), 10));
    }));
  }

  // ── create group ────────────────────────────────────────────────────────
  function openCreate() {
    MD.open({
      icon: 'i-shield-alert', title: 'New API group', subtitle: 'Shared permission set',
      width: 540,
      body: (el) => {
        el.innerHTML = '<div class="apm-field"><label>Group name</label>' +
          '<input class="apin-input" id="kg-cname" placeholder="e.g. mobile-app, analytics, partners" maxlength="80"></div>' +
          '<div class="apm-field"><label>Permissions</label>' + scopeChecklistHTML([], []) + '</div>';
        wireScopeChecklist(el);
        setTimeout(() => { const i = el.querySelector('#kg-cname'); if (i) i.focus(); }, 30);
      },
      actions: [
        { label: 'Cancel', kind: 'ghost' },
        { label: 'Create group', kind: 'primary', busyLabel: 'Creating…', closeOnClick: false,
          onClick: async (ctx) => {
            const name = (ctx.query('#kg-cname').value || '').trim();
            if (!name) { ctx.setError('Group name is required.'); return false; }
            const scopes = collectScopes(ctx.root);
            const { status, body } = await sudoFetch('/api/account/key-groups', 'POST', { name, scopes });
            if (!body.ok) { ctx.setError(MD.writeErr(body)); return false; }
            renderPanel();
            return true;
          } },
      ],
    });
  }

  // ── group editor ─────────────────────────────────────────────────────────
  async function openEditor(gid) {
    let g;
    try { g = await gGet('/api/account/key-groups/' + gid); } catch (e) { return; }
    if (!g || !g.id) return;
    MD.open({
      icon: 'i-shield-alert', title: g.name, subtitle: 'API group · ' + (g.member_count || 0) + ' members',
      width: 600,
      body: (el) => { renderEditorBody(el, g); },
      actions: [{ label: 'Done', kind: 'ghost' }],
    });
  }

  function renderEditorBody(el, g) {
    el.innerHTML =
      '<div class="kg-sec"><div class="kg-sec-h"><b>Name &amp; permissions</b>' +
      '<button class="apin-btn ghost kg-mini" id="kg-edit-group">Edit</button></div>' +
      '<div id="kg-scope-view">' + (g.scopes.length ? g.scopes.map(s => '<code>' + esc(s) + '</code>').join(' ') :
        '<span class="kg-d">no scopes — locked members can do nothing</span>') + '</div></div>' +
      '<div class="kg-sec"><div class="kg-sec-h"><b>Members</b><span class="kg-mini-actions">' +
      '<button class="apin-btn ghost kg-mini" id="kg-new-key">+ new key</button>' +
      '<button class="apin-btn ghost kg-mini" id="kg-add-member">+ add existing</button></span></div>' +
      '<div id="kg-members">' + renderMembers(g) + '</div></div>' +
      '<div class="kg-sec kg-danger"><div class="kg-sec-h"><b>Delete group</b></div>' +
      '<div class="kg-d">Remove the group and decide what happens to its keys.</div>' +
      '<button class="apin-btn danger kg-mini" id="kg-del-group" style="margin-top:8px">Delete group</button></div>';

    el.querySelector('#kg-edit-group').addEventListener('click', () => openEditGroup(g, el));
    el.querySelector('#kg-new-key').addEventListener('click', () => openCreateKeyInGroup(g, el));
    el.querySelector('#kg-add-member').addEventListener('click', () => openAddMember(g, el));
    el.querySelector('#kg-del-group').addEventListener('click', () => openDeleteGroup(g));
    wireMemberActions(el, g);
  }

  function renderMembers(g) {
    const ms = g.members || [];
    if (!ms.length) return '<div class="kg-d">No keys in this group yet.</div>';
    return ms.map(m =>
      '<div class="kg-member" data-pid="' + esc(m.public_id) + '">' +
      '<div class="kg-m-main"><span class="kg-m-name">' + esc(m.name || m.public_id) + '</span>' +
      '<span class="kg-role kg-role-' + (m.group_role || 'locked') + '">' + (m.group_role || 'locked') + '</span></div>' +
      '<div class="kg-m-actions">' +
      (m.group_role === 'special'
        ? '<button class="kg-link" data-act="demote">make locked</button>'
        : '<button class="kg-link" data-act="promote">make special</button>') +
      '<button class="kg-link danger" data-act="remove">remove</button></div></div>').join('');
  }

  function wireMemberActions(el, g) {
    el.querySelectorAll('.kg-member').forEach(row => {
      const pid = row.getAttribute('data-pid');
      row.querySelectorAll('[data-act]').forEach(btn => btn.addEventListener('click', async () => {
        const act = btn.getAttribute('data-act');
        if (act === 'remove') {
          const { body } = await sudoFetch('/api/account/key-groups/' + g.id + '/members/' + encodeURIComponent(pid), 'DELETE', undefined);
          if (body.ok) refreshEditor(el, g.id);
          else alert(MD.writeErr(body));
        } else if (act === 'demote') {
          const { body } = await sudoFetch('/api/account/key-groups/' + g.id + '/members/' + encodeURIComponent(pid), 'PATCH', { role: 'locked' });
          if (body.ok) refreshEditor(el, g.id);
          else alert(MD.writeErr(body));
        } else if (act === 'promote') {
          openPromote(g, pid, el);
        }
      }));
    });
  }

  async function refreshEditor(el, gid) {
    const g = await gGet('/api/account/key-groups/' + gid);
    renderEditorBody(el, g);
    renderPanel();
  }

  // edit group NAME + scopes
  function openEditGroup(g, el) {
    MD.open({
      icon: 'i-shield-alert', title: 'Edit group', subtitle: g.name, width: 540,
      body: (b) => {
        b.innerHTML = '<div class="apm-field"><label>Group name</label>' +
          '<input class="apin-input" id="kg-ename" maxlength="80" value="' + esc(g.name) + '"></div>' +
          '<div class="apm-field"><label>Permissions</label>' + scopeChecklistHTML(g.scopes, []) + '</div>';
        wireScopeChecklist(b);
      },
      actions: [
        { label: 'Cancel', kind: 'ghost' },
        { label: 'Save', kind: 'primary', busyLabel: 'Saving…', closeOnClick: false,
          onClick: async (ctx) => {
            const name = (ctx.query('#kg-ename').value || '').trim();
            if (!name) { ctx.setError('Group name is required.'); return false; }
            const scopes = collectScopes(ctx.root);
            const payload = { scopes };
            if (name !== g.name) payload.name = name;
            const { body } = await sudoFetch('/api/account/key-groups/' + g.id, 'PATCH', payload);
            if (!body.ok) { ctx.setError(MD.writeErr(body)); return false; }
            refreshEditor(el, g.id);
            return true;
          } },
      ],
    });
  }

  // create a brand-new key directly inside this group (locked member, inherits
  // the group's scopes). Shows the one-time token reveal like minting.
  function openCreateKeyInGroup(g, el) {
    MD.open({
      icon: 'i-key', title: 'New key in group', subtitle: g.name, width: 520,
      body: (b) => {
        b.innerHTML = '<div class="apm-field"><label>Key name</label>' +
          '<input class="apin-input" id="kg-kname" placeholder="e.g. mobile-prod" maxlength="80"></div>' +
          '<div class="apm-field"><label>Environment</label><select class="apin-select" id="kg-kenv">' +
          '<option value="live">live</option><option value="test">test</option></select></div>' +
          '<div class="kg-d">It joins <b>' + esc(g.name) + '</b> as a <b>locked</b> member and inherits its ' +
          g.scopes.length + ' permission(s): ' + (g.scopes.map(s => '<code>' + esc(s) + '</code>').join(' ') || '—') + '</div>';
        setTimeout(() => { const i = b.querySelector('#kg-kname'); if (i) i.focus(); }, 30);
      },
      actions: [
        { label: 'Cancel', kind: 'ghost' },
        { label: 'Create key', kind: 'primary', busyLabel: 'Minting…', closeOnClick: false,
          onClick: async (ctx) => {
            const name = (ctx.query('#kg-kname').value || '').trim();
            if (!name) { ctx.setError('Key name is required.'); return false; }
            const env = (ctx.query('#kg-kenv') || {}).value || 'live';
            // 1. mint the key with the group's scopes
            const mint = await sudoFetch('/api/account/keys', 'POST',
              { name: name, environment: env, scopes: g.scopes });
            if (!mint.body.ok) { ctx.setError(MD.writeErr(mint.body)); return false; }
            const nk = mint.body.data || {};
            const pid = nk.public_id;
            const secret = nk.plaintext_token || nk.token || nk.secret || '';
            // 2. attach it to the group as locked (best-effort; key still exists if this fails)
            if (pid) { try { await sudoFetch('/api/account/key-groups/' + g.id + '/members', 'POST', { public_id: pid, role: 'locked' }); } catch (e) {} }
            // 3. one-time token reveal in the same modal
            MD.oneTimeToken(ctx, { name: name, env: env, public_id: pid, scopes: g.scopes, secret: secret, expires: 'no expiry' });
            refreshEditor(el, g.id);
            return false;   // keep modal open to show the secret
          } },
      ],
    });
  }

  // add an existing key to the group (locked)
  async function openAddMember(g, el) {
    let keys = [];
    try { keys = (await gGet('/api/account/keys?limit=100')).keys || (await gGet('/api/account/keys?limit=100')).items || []; }
    catch (e) {}
    // keys not already in THIS group, not deleted
    const inGroup = new Set((g.members || []).map(m => m.public_id));
    const avail = keys.filter(k => !inGroup.has(k.public_id) && k.status !== 'deleted');
    MD.open({
      icon: 'i-plus', title: 'Add key to group', subtitle: g.name, width: 520,
      body: (b) => {
        if (!avail.length) { b.innerHTML = '<div class="kg-d">No eligible keys. Every key is already in this group (a key can belong to one group).</div>'; return; }
        b.innerHTML = '<div class="apm-field"><label>Key</label><select class="apin-select" id="kg-pick">' +
          avail.map(k => '<option value="' + esc(k.public_id) + '">' + esc(k.name || k.public_id) + '</option>').join('') +
          '</select></div><div class="kg-d">It joins as a <b>locked</b> member (inherits the group’s scopes). You can promote it to special afterwards.</div>';
      },
      actions: [
        { label: 'Cancel', kind: 'ghost' },
        { label: 'Add key', kind: 'primary', busyLabel: 'Adding…', closeOnClick: false,
          disabled: () => !avail.length,
          onClick: async (ctx) => {
            const pid = (ctx.query('#kg-pick') || {}).value;
            if (!pid) return false;
            const { body } = await sudoFetch('/api/account/key-groups/' + g.id + '/members', 'POST', { public_id: pid, role: 'locked' });
            if (!body.ok) { ctx.setError(MD.writeErr(body)); return false; }
            refreshEditor(el, g.id);
            return true;
          } },
      ],
    });
  }

  // promote a member to special, choosing its ceiling
  function openPromote(g, pid, el) {
    MD.open({
      icon: 'i-shield-alert', title: 'Make special key', subtitle: g.name, width: 540,
      body: (b) => {
        b.innerHTML = '<div class="apm-warn">A special key sets its own scopes within a <b>ceiling</b>. The group’s scopes are always allowed; tick any <b>extra</b> scopes this key may also hold.</div>' +
          scopeChecklistHTML(g.scopes, g.scopes);   // group scopes locked-on; extras selectable
        wireScopeChecklist(b);
      },
      actions: [
        { label: 'Cancel', kind: 'ghost' },
        { label: 'Make special', kind: 'primary', busyLabel: 'Updating…', closeOnClick: false,
          onClick: async (ctx) => {
            const ceiling = collectScopes(ctx.root);   // group ∪ extras (backend unions anyway)
            const { body } = await sudoFetch('/api/account/key-groups/' + g.id + '/members/' + encodeURIComponent(pid), 'PATCH',
              { role: 'special', scope_ceiling: ceiling });
            if (!body.ok) { ctx.setError(MD.writeErr(body)); return false; }
            refreshEditor(el, g.id);
            return true;
          } },
      ],
    });
  }

  // delete group with member policy
  // Staged delete: (1) pick the member policy with custom controls, then
  // (2) reveal the type-to-confirm. The group is only deleted after BOTH the
  // policy is chosen AND the name is typed — never on type-match alone.
  function openDeleteGroup(g) {
    const ms = g.members || [];
    let keep = [];
    let m;
    const POLICY_LABELS = {
      ungroup: 'keep all keys (ungroup them)',
      delete_all: 'delete all keys',
      keep_special: 'keep only special keys',
      choose: 'keep the selected keys, delete the rest',
    };
    function opt(val, title, desc, checked) {
      return '<label class="kg-opt"><input type="radio" name="kg-policy" class="apin-radio" value="' + val + '"' + (checked ? ' checked' : '') + '>' +
        '<span class="kg-opt-t"><b>' + esc(title) + '</b><span>' + esc(desc) + '</span></span></label>';
    }
    function renderPolicy(el) {
      el.innerHTML =
        '<p class="kg-del-lead">Choose what happens to the <b>' + ms.length + '</b> key' + (ms.length === 1 ? '' : 's') + ' in this group.</p>' +
        '<div class="kg-policy">' +
          opt('ungroup', 'Keep all keys', 'detach every key — they become ungrouped and keep their current scopes', true) +
          opt('delete_all', 'Delete all keys', 'permanently disable + delete every key in the group') +
          opt('keep_special', 'Keep only special keys', 'detach special keys; delete locked keys') +
          opt('choose', 'Choose which to keep', 'pick the keys to keep; the rest are deleted') +
        '</div>' +
        '<div id="kg-keep" class="kg-keep" hidden>' + ms.map(mem =>
          '<label class="kg-row"><input type="checkbox" class="apin-check" data-keep="' + esc(mem.public_id) + '"> <span>' +
          esc(mem.name || mem.public_id) + ' <span class="kg-role kg-role-' + (mem.group_role || 'locked') + '">' + (mem.group_role || 'locked') + '</span></span></label>').join('') + '</div>' +
        '<div class="kg-del-foot"><button class="apin-btn danger" id="kg-del-next" type="button">Continue</button></div>';
      el.querySelectorAll('input[name="kg-policy"]').forEach(r => r.addEventListener('change', () => {
        el.querySelector('#kg-keep').hidden = (el.querySelector('input[name="kg-policy"]:checked').value !== 'choose');
      }));
      el.querySelector('#kg-del-next').addEventListener('click', () => {
        const sel = el.querySelector('input[name="kg-policy"]:checked');
        const policy = sel ? sel.value : 'ungroup';
        if (policy === 'choose') keep = Array.prototype.map.call(el.querySelectorAll('input[data-keep]:checked'), i => i.getAttribute('data-keep'));
        renderConfirm(el, policy);
      });
    }
    function renderConfirm(el, policy) {
      el.innerHTML =
        '<div class="apm-warn">This permanently deletes the group. Keys: <b>' + esc(POLICY_LABELS[policy]) + '</b>. This cannot be undone.</div>' +
        '<div class="apm-field"><label>Type <strong>' + esc(g.name) + '</strong> to confirm</label>' +
        '<input class="apin-input" id="kg-del-confirm" autocomplete="off" spellcheck="false" placeholder="' + esc(g.name) + '"></div>' +
        '<div class="apm-err kg-del-err"></div>' +
        '<div class="kg-del-foot"><button class="apin-btn ghost" id="kg-del-back" type="button">Back</button>' +
        '<button class="apin-btn danger" id="kg-del-go" type="button" disabled>Delete group</button></div>';
      const inp = el.querySelector('#kg-del-confirm');
      const go = el.querySelector('#kg-del-go');
      const err = el.querySelector('.kg-del-err');
      inp.addEventListener('input', () => { go.disabled = inp.value.trim() !== g.name; });
      setTimeout(() => inp.focus(), 30);
      el.querySelector('#kg-del-back').addEventListener('click', () => renderPolicy(el));
      go.addEventListener('click', async () => {
        go.disabled = true; go.textContent = 'Deleting…'; err.classList.remove('show');
        const { body } = await sudoFetch('/api/account/key-groups/' + g.id, 'DELETE', { member_policy: policy, keep_public_ids: keep });
        if (!body.ok) { go.disabled = false; go.textContent = 'Delete group'; err.textContent = MD.writeErr(body); err.classList.add('show'); return; }
        if (m) m.close();
        renderPanel();
      });
    }
    m = MD.open({
      icon: 'i-trash', danger: true, title: 'Delete "' + g.name + '"', width: 560,
      body: (el) => renderPolicy(el),
      actions: [{ label: 'Cancel', kind: 'ghost' }],
    });
  }

  // ── styles ────────────────────────────────────────────────────────────────
  function injectCSS() {
    if (document.getElementById('kg-css')) return;
    const s = document.createElement('style'); s.id = 'kg-css';
    s.textContent = `
.kg-panel{margin:6px 0 22px}
.kg-cap{font:600 11px 'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-mute,#8b8273);margin-bottom:11px}
.kg-tip{position:absolute;z-index:3000;max-width:300px;background:var(--ink,#1a1612);color:var(--paper,#fbf9f3);font:11px/1.55 'JetBrains Mono',monospace;padding:9px 12px;border-radius:9px;box-shadow:0 10px 28px rgba(20,16,12,.28);opacity:0;transform:translateY(-5px);transition:opacity .15s ease,transform .15s ease;pointer-events:none}
.kg-tip.show{opacity:1;transform:none}
.kg-tip::before{content:"";position:absolute;top:-5px;left:18px;border:5px solid transparent;border-top:0;border-bottom-color:var(--ink,#1a1612)}
.kg-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.kg-card{text-align:left;border:1px solid var(--paper-edge,#c7bca9);background:var(--paper,#fbf9f3);border-radius:13px;padding:13px 15px;cursor:pointer;transition:transform .12s,box-shadow .15s,border-color .15s;display:flex;flex-direction:column;gap:9px}
.kg-card:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(20,16,12,.09);border-color:var(--green,#2f6f3e)}
.kg-card-top{display:flex;align-items:center;justify-content:space-between;gap:8px}
.kg-name{font:600 13px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.kg-count{font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273);background:var(--paper-deep,#e9e2d1);border-radius:20px;padding:2px 9px;white-space:nowrap}
.kg-chips{display:flex;flex-wrap:wrap;gap:5px}
.kg-chips code{font:10.5px 'JetBrains Mono',monospace;background:var(--paper-deep,#e9e2d1);border-radius:5px;padding:1px 7px;color:var(--ink-soft,#5a5246)}
.kg-d{font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273)}
.kg-scopes{display:flex;flex-direction:column;gap:9px;max-height:48vh;overflow:auto;padding-right:4px;margin-top:9px}
.kg-grp{border:1.2px solid var(--paper-edge,#e3d9c0);border-radius:11px;background:var(--paper-soft,#f4efe6);overflow:hidden}
.kg-grp.open{border-color:var(--paper-edge,#c7bca9);background:var(--paper,#fbf9f3)}
.kg-grp-h{display:flex;align-items:center;gap:10px;padding:11px 13px}
.kg-grp-label{display:flex;align-items:center;gap:10px;cursor:pointer;flex:1;min-width:0}
.kg-grp-label b{font:600 12.5px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.kg-expand{display:inline-flex;align-items:center;gap:7px;border:0;background:none;cursor:pointer;color:var(--ink-mute,#8b8273);font:10.5px 'JetBrains Mono',monospace;padding:3px 4px;border-radius:7px;transition:color .15s,background .15s}
.kg-expand:hover{color:var(--green-deep,#1f5b32);background:rgba(47,111,62,.07)}
.kg-expand .kg-xt{font-weight:600}
.kg-expand svg{width:13px;height:13px;fill:none;stroke:currentColor;stroke-width:2;transition:transform .18s ease}
.kg-grp.open .kg-expand svg{transform:rotate(90deg)}
.kg-grp-sum{font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273)}
.kg-grp-body{display:flex;flex-direction:column;gap:3px;padding:2px 14px 11px 38px}
.kg-row{display:flex;align-items:center;gap:9px;padding:4px 0;font:11.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5246);cursor:pointer}
.kg-row code{color:var(--ink,#1a1612)}.kg-row .kg-d{color:var(--ink-mute,#8b8273)}
.kg-sec{padding:13px 0;border-top:1px solid var(--paper-edge,#efe6cf)}
.kg-sec:first-child{border-top:0;padding-top:2px}
.kg-sec-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}
.kg-sec-h b{font:600 12.5px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
#kg-scope-view code{font:11px 'JetBrains Mono',monospace;background:var(--paper-deep,#e9e2d1);border-radius:5px;padding:1px 7px;margin:0 3px 4px 0;display:inline-block}
.kg-mini{padding:5px 11px;font-size:11px}
.kg-mini-actions{display:inline-flex;gap:7px}
.kg-del-lead{font:12px 'JetBrains Mono',monospace;color:var(--ink-soft,#5a5246);margin-bottom:13px;line-height:1.5}
.kg-del-lead b{color:var(--ink,#1a1612)}
.kg-del-foot{display:flex;justify-content:flex-end;gap:9px;margin-top:16px}
.kg-del-err{margin:12px 0 0}
.kg-opt input{margin-top:2px}
.kg-member{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 0;border-top:1px solid var(--paper-edge,#efe6cf)}
.kg-member:first-child{border-top:0}
.kg-m-main{display:flex;align-items:center;gap:9px;min-width:0}
.kg-m-name{font:12px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);overflow:hidden;text-overflow:ellipsis}
.kg-role{font:600 9.5px 'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.04em;border-radius:20px;padding:2px 8px}
.kg-role-locked{background:var(--paper-deep,#e9e2d1);color:var(--ink-mute,#8b8273)}
.kg-role-special{background:var(--ochre-soft,#fbe3c2);color:var(--ochre-deep,#7a4e0a)}
.kg-m-actions{display:flex;gap:10px;flex:none}
.kg-link{border:0;background:none;cursor:pointer;font:600 11px 'JetBrains Mono',monospace;color:var(--green-deep,#1f5b32);padding:2px}
.kg-link:hover{text-decoration:underline}.kg-link.danger{color:#b3402f}
.kg-danger .kg-sec-h b{color:#b3402f}
.kg-policy{display:flex;flex-direction:column;gap:8px;margin:0 0 14px}
.kg-opt{display:flex;align-items:flex-start;gap:10px;border:1.2px solid var(--paper-edge,#c7bca9);border-radius:10px;padding:10px 12px;cursor:pointer;transition:border-color .15s,background .15s}
.kg-opt:hover{border-color:var(--ink-mute,#8b8273)}
.kg-opt input{margin-top:3px}
.kg-opt-t{display:flex;flex-direction:column;gap:2px}.kg-opt-t b{font:600 12px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}.kg-opt-t span{font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#8b8273)}
.kg-keep{display:flex;flex-direction:column;gap:5px;padding:8px 10px;border:1px dashed var(--paper-edge,#c7bca9);border-radius:9px;margin-bottom:6px}
`;
    document.head.appendChild(s);
  }

  // The keys page doesn't ship the icon sprite — inject it so modal/button
  // glyphs (i-plus, i-shield-alert, i-x, i-trash…) resolve.
  async function ensureSprite() {
    if (document.getElementById('i-plus') || document.getElementById('i-x')) return;
    try {
      const r = await fetch('/static/console_icons.svg');
      if (!r.ok) return;
      const div = document.createElement('div');
      div.style.display = 'none';
      div.innerHTML = await r.text();
      document.body.appendChild(div);
    } catch (e) {}
  }

  async function boot() { await ensureSprite(); renderPanel(); }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
  window.APIN.keyGroups = { refresh: renderPanel };
})();
