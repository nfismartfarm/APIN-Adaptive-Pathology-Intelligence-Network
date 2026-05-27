(function () {
  'use strict';

  // ── DOM ──────────────────────────────────────────────────────────────
  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  const $feed = document.getElementById('alert-feed');
  const $bellCount = document.getElementById('bell-count');
  const $loadMore = document.getElementById('load-more-btn');
  // Phase 8.H fix · removed select-all + mark-read + dismiss toolbar
  // buttons; replaced by one "Mark all visible read" button + always-
  // visible per-row icon actions.
  const $markAllReadBtn = document.getElementById('mark-all-read-btn');
  const $toast = document.getElementById('toast');

  // ── State ────────────────────────────────────────────────────────────
  let filters = { sev: '', state: 'active' };  // state: active | unread | all
  let items = [];
  let nextCursor = null;
  let pollTimer = null;
  const POLL_MS = 30_000;

  // ── Helpers ──────────────────────────────────────────────────────────
  function showToast(msg, level) {
    $toast.textContent = msg;
    $toast.className = 'toast show ' + (level === 'err' ? 'err' : 'ok');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => { $toast.className = 'toast'; }, 2200);
  }
  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function fmtAgo(iso) {
    const ms = Date.now() - new Date(iso).getTime();
    if (isNaN(ms)) return iso;
    if (ms < 60_000) return Math.round(ms/1000) + 's ago';
    if (ms < 3600_000) return Math.round(ms/60_000) + 'm ago';
    if (ms < 86400_000) return Math.round(ms/3600_000) + 'h ago';
    if (ms < 7*86400_000) return Math.round(ms/86400_000) + 'd ago';
    return new Date(iso).toLocaleDateString();
  }

  // ── Fetch ────────────────────────────────────────────────────────────
  function buildQuery(opts) {
    const qs = new URLSearchParams();
    qs.set('limit', '30');
    if (filters.sev) qs.set('severity', filters.sev);
    if (filters.state === 'unread') qs.set('only_unread', 'true');
    if (filters.state === 'all') qs.set('include_dismissed', 'true');
    if (opts && opts.cursor) qs.set('cursor', opts.cursor);
    return qs.toString();
  }

  async function loadFeed(opts) {
    opts = opts || {};
    try {
      const r = await fetch('/api/account/alerts?' + buildQuery(opts), {credentials:'include'});
      if (r.status === 401) { window.location.href = '/dashboard'; return; }
      const body = await r.json();
      if (!body.ok) { showToast(body.error?.message || 'Failed to load alerts', 'err'); return; }
      const newItems = (body.data && body.data.items) || [];
      if (opts.append) items = items.concat(newItems);
      else items = newItems;
      nextCursor = (body.data && body.data.next_cursor) || null;
      render();
    } catch (err) {
      showToast('Network error while loading alerts.', 'err');
    }
  }

  async function loadUnreadCount() {
    try {
      const r = await fetch('/api/account/alerts/unread-count', {credentials:'include'});
      const body = await r.json();
      if (body.ok) {
        const n = body.data?.unread || 0;
        $bellCount.textContent = n;
        $bellCount.style.color = n > 0 ? 'var(--accent-ochre)' : 'var(--ink-soft)';
      }
    } catch (_) { /* silent */ }
  }

  // ── Render ───────────────────────────────────────────────────────────
  function render() {
    if (items.length === 0) {
      $feed.innerHTML = ''
        + '<div class="feed-empty">'
        + '<h3>All quiet here</h3>'
        + '<p>You will see alerts surface here whenever something noteworthy happens to your keys, webhooks, or quotas.</p>'
        + '<ul>'
        + '<li>A new key is minted in your account</li>'
        + '<li>A key is used from an IP it has never seen before</li>'
        + '<li>A webhook delivery fails or is auto-disabled</li>'
        + '<li>You hit a daily quota</li>'
        + '<li>Repeated auth failures appear from a single IP</li>'
        + '</ul>'
        + '<p style="margin-top:14px">Configure which of these fire in <a href="/account/api/settings#sec-notif">Settings → Notifications</a>.</p>'
        + '</div>';
      $loadMore.hidden = true;
      return;
    }
    $feed.innerHTML = items.map(renderCard).join('');
    $loadMore.hidden = !nextCursor;
    wireCards();
    // Phase 8.H fix · `refreshBulkButtons()` removed — was a leftover from
    // the now-deleted bulk-select toolbar. The TypeError it raised was
    // being swallowed by the outer try/catch around loadAlerts() and
    // mislabelled as "Network error while loading alerts." (the toast
    // you saw despite the cards rendering fine).
  }

  function renderCard(a) {
    const isUnread = !a.read_at;
    const isDismissed = !!a.dismissed_at;
    const cls = ['alert-card', 'sev-' + a.severity,
      isUnread ? 'unread' : '', isDismissed ? 'dismissed' : ''
    ].filter(Boolean).join(' ');
    const bumped = a.occurrence_count > 1
      ? '<span class="alert-count bumped">×' + a.occurrence_count + '</span>' : '';
    return '<div class="' + cls + '" data-alert-id="' + a.id + '">'
      + '<div class="alert-main">'
        + '<div class="alert-title">'
          + (isUnread ? '<span class="unread-dot" aria-label="unread"></span>' : '')
          + escHtml(a.title)
        + '</div>'
        + '<div class="alert-body">' + escHtml(a.body) + '</div>'
        + '<div class="alert-meta-row">'
          + '<span class="alert-sev sev-' + a.severity + '">' + a.severity + '</span>'
          + '<span class="alert-code">' + escHtml(a.code) + '</span>'
          + bumped
          + '<span class="alert-time">' + fmtAgo(a.updated_at || a.created_at) + '</span>'
          + (a.key_id ? '<span>· key <a href="/account/api/keys#' + escHtml(a.key_id) + '">'
              + escHtml(a.key_id.slice(0,12)) + '…</a></span>' : '')
        + '</div>'
        + (a.details
          ? '<div class="alert-details" hidden data-details>'
            + escHtml(JSON.stringify(a.details, null, 2)) + '</div>'
          : '')
      + '</div>'
      + '<div class="alert-actions">'
        + (isUnread
            ? '<button type="button" class="ic-btn success" data-act="read" '
              + 'aria-label="Mark read" title="Mark read">'
              + '<svg aria-hidden="true"><use href="#i-check"/></svg></button>'
            : '')
        + (isDismissed
            ? '<button type="button" class="ic-btn" data-act="restore" '
              + 'aria-label="Restore" title="Restore">'
              + '<svg aria-hidden="true"><use href="#i-refresh"/></svg></button>'
            : '<button type="button" class="ic-btn danger" data-act="dismiss" '
              + 'aria-label="Dismiss" title="Dismiss">'
              + '<svg aria-hidden="true"><use href="#i-trash"/></svg></button>')
      + '</div>'
      + '</div>';
  }

  function wireCards() {
    $feed.querySelectorAll('.alert-card').forEach(card => {
      const id = card.dataset.alertId;
      // Click anywhere except actions = expand. Phase 8.H · removed
      // the `input` selector (no more checkboxes on rows).
      card.addEventListener('click', (e) => {
        if (e.target.closest('.ic-btn, a')) return;
        card.classList.toggle('expanded');
        const details = card.querySelector('[data-details]');
        if (details) details.hidden = !card.classList.contains('expanded');
        // Optimistically mark read on expand (engagement = read).
        if (card.classList.contains('expanded') && card.classList.contains('unread')) {
          markRead([id]);
        }
      });
      const readBtn = card.querySelector('[data-act="read"]');
      if (readBtn) readBtn.addEventListener('click', (e) => { e.stopPropagation(); markRead([id]); });
      const disBtn  = card.querySelector('[data-act="dismiss"]');
      if (disBtn)  disBtn.addEventListener('click',  (e) => { e.stopPropagation(); dismissAlerts([id]); });
      const resBtn  = card.querySelector('[data-act="restore"]');
      if (resBtn)  resBtn.addEventListener('click',  (e) => { e.stopPropagation(); restoreAlert(id); });
    });
  }

  // ── Bulk action: mark all visible read ───────────────────────────────
  if ($markAllReadBtn) {
    $markAllReadBtn.addEventListener('click', () => {
      const ids = [...$feed.querySelectorAll('.alert-card.unread')]
        .map(c => c.dataset.alertId);
      if (ids.length === 0) {
        showToast('Nothing to mark read');
        return;
      }
      markRead(ids);
    });
  }

  // ── Mutations ────────────────────────────────────────────────────────
  async function markRead(ids) {
    if (!ids.length) return;
    for (const id of ids) {
      // Optimistic UI
      const card = $feed.querySelector('[data-alert-id="' + id + '"]');
      if (card) { card.classList.remove('unread');
        const dot = card.querySelector('.unread-dot'); if (dot) dot.remove();
        const btn = card.querySelector('[data-act="read"]'); if (btn) btn.remove(); }
      try {
        const r = await fetch('/api/account/alerts/' + id + '/read', {
          method: 'PATCH', credentials: 'include',
          headers: {'Accept':'application/json','X-Console-Csrf': csrf},
        });
        const body = await r.json();
        if (!body.ok) showToast(body.error?.message || 'Could not mark read', 'err');
      } catch (_) { showToast('Network error', 'err'); }
    }
    // refresh badge
    loadUnreadCount();
    // Phase 8.H · same-tab signal for nav badge + dashboard tile.
    try {
      window.dispatchEvent(new CustomEvent("apin:alerts:changed",
        { detail: { type: "read", ids: ids } }));
    } catch (_) { /* defensive */ }
    showToast(ids.length + ' marked read');
  }

  async function dismissAlerts(ids) {
    if (!ids.length) return;
    for (const id of ids) {
      const card = $feed.querySelector('[data-alert-id="' + id + '"]');
      if (card) card.classList.add('dismissed');
      try {
        const r = await fetch('/api/account/alerts/' + id, {
          method: 'DELETE', credentials: 'include',
          headers: {'Accept':'application/json','X-Console-Csrf': csrf},
        });
        const body = await r.json();
        if (!body.ok) showToast(body.error?.message || 'Could not dismiss', 'err');
      } catch (_) { showToast('Network error', 'err'); }
    }
    // Update items array, re-render if filter is "active" (dismissed should vanish)
    if (filters.state === 'active') {
      items = items.filter(a => !ids.includes(String(a.id)));
      render();
    }
    loadUnreadCount();
    showToast(ids.length + ' dismissed');
  }

  async function restoreAlert(id) {
    try {
      const r = await fetch('/api/account/alerts/' + id + '/restore', {
        method: 'POST', credentials: 'include',
        headers: {'Accept':'application/json','X-Console-Csrf': csrf},
      });
      const body = await r.json();
      if (!body.ok) { showToast(body.error?.message || 'Could not restore', 'err'); return; }
      loadFeed();
      loadUnreadCount();
      showToast('Restored');
    } catch (_) { showToast('Network error', 'err'); }
  }

  // ── Filter wiring ────────────────────────────────────────────────────
  document.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const f = chip.dataset.filter;
      const v = chip.dataset.val;
      // Single-select within group
      document.querySelectorAll('.filter-chip[data-filter="' + f + '"]').forEach(c =>
        c.classList.toggle('on', c === chip));
      filters[f] = v;
      loadFeed();
    });
  });

  // ── Load more ────────────────────────────────────────────────────────
  $loadMore.addEventListener('click', () => loadFeed({append: true, cursor: nextCursor}));

  // ── Auto-poll ────────────────────────────────────────────────────────
  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      loadUnreadCount();
      // Only auto-refresh the feed if user is at the top (no scroll position lost)
      if (window.scrollY < 80) loadFeed();
    }, POLL_MS);
  }
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      loadUnreadCount(); loadFeed();
      startPolling();
    } else {
      if (pollTimer) clearInterval(pollTimer);
    }
  });

  // ── Bootstrap ────────────────────────────────────────────────────────
  loadFeed();
  loadUnreadCount();
  startPolling();
})();