// 9.N.T · Per-key TRAFFIC tab — widget module (revision 2).
//
// Owns #pane-traffic. One /traffic fetch feeds: hero (status-stacked blocks
// over time, two-line local labels, brush-scrubber + zoom, click→request
// container), four expandable KPI tiles (Total · Peak/min · Busiest · Data),
// a compact GitHub activity calendar, a Method×Status honeycomb HIVE, a polar
// traffic clock (self-driving hand, click-hour→60-min breakdown), and a
// bytes-flow mirror. Everything updates per-SSE-event in real time (cards AND
// open expanded states) — no manual refresh. Drills open an inline request
// container (lightbox + filter chips); each row opens the existing request
// drawer (no tab jump). Times are device-local. Paper-ink SVG, hand-drawn
// sprite icons, no emoji.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtNum = (n) => n == null ? "—" : Number(n).toLocaleString();
  const fmtBytes = (b) => b == null ? "—" : b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : b < 1073741824 ? (b / 1048576).toFixed(1) + " MB" : (b / 1073741824).toFixed(2) + " GB";
  // Clear directional ratio (the larger side : smaller side). Avoids the
  // "out:in 0×" artifact when uploads make bytes-in dominate bytes-out.
  function _ratioStr(tin, tout) {
    if (!tin || !tout) return "";
    const r = tout >= tin ? tout / tin : tin / tout;
    const lbl = tout >= tin ? "out:in" : "in:out";
    return lbl + " " + (r < 10 ? r.toFixed(1) : Math.round(r)) + "×";
  }
  const TZOFF = (window.APIN && APIN.time) ? APIN.time.offsetMin() : -new Date().getTimezoneOffset();
  const PID_META = document.querySelector('meta[name="key-public-id"]');
  function api(path) {
    return fetch(path, { headers: { "Accept": "application/json" }, credentials: "same-origin" })
      .then(r => r.json().then(b => ({ ok: r.ok, body: b })).catch(() => ({ ok: false, body: null })))
      .catch(() => ({ ok: false, body: null }));
  }
  function _C() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n, d) => (cs.getPropertyValue(n).trim() || d);
    return { ok: g("--c-ok", "#2f6f3e"), amber: g("--c-amber", "#c98a2b"),
             danger: g("--c-danger", "#b3402f"), ink: g("--ink", "#1a1612"),
             soft: g("--ink-soft", "#6b6453"), mute: g("--ink-mute", "#9a907a"),
             edge: g("--paper-edge", "#c7bca9"), accent: g("--c-accent", "#52b788") };
  }
  // shared floating tooltip
  let _tip = null;
  function tip(show, x, y, html) {
    if (!_tip) { _tip = document.createElement("div"); _tip.className = "ov-tip"; document.body.appendChild(_tip); }
    if (!show) { _tip.classList.remove("show"); return; }
    _tip.innerHTML = html;
    _tip.style.left = Math.min(x + 12, window.innerWidth - 250) + "px";
    _tip.style.top = (y + 14) + "px";
    _tip.classList.add("show");
  }
  // slot-machine integer (odometer) with count-up fallback
  function slot(el, val) {
    if (!el) return;
    const s = fmtNum(val);
    if (/\d/.test(s) && window.APIN && APIN.odometer) { el.classList.add("apin-odometer"); APIN.odometer.roll(el, s); }
    else { el.textContent = s; }
  }
  // UTC 'YYYY-MM-DD HH:MM:SS' from epoch ms (matches the DB timestamp column)
  const _utcStr = (ms) => new Date(ms).toISOString().replace("T", " ").slice(0, 19);
  const _local = (iso) => (window.APIN && APIN.time) ? APIN.time.local(iso) : iso;
  const GRAN_MS = { hour: 3600e3, day: 86400e3, week: 7 * 86400e3, month: 31 * 86400e3 };
  const GRAN_LABEL = { hour: "last 24h", day: "last 30d", week: "last 12w", month: "last 12mo" };
  // window (ms) the clock aggregates over, per granularity — matches backend _TRAFFIC_GRAN
  const GRAN_WIN_MS = { hour: 1440 * 60e3, day: 43200 * 60e3, week: 120960 * 60e3, month: 535680 * 60e3 };
  // 12-hour labels — "7:00 AM", "6:00 PM", "12:00 AM" (midnight), "12:00 PM" (noon)
  const fmt12 = (h) => { h = ((+h) % 24 + 24) % 24; return ((h % 12) || 12) + ":00 " + (h < 12 ? "AM" : "PM"); };
  // current wall-clock with minutes — "7:42 PM"
  const fmtClockNow = (d) => { const h = d.getHours(); return ((h % 12) || 12) + ":" + String(d.getMinutes()).padStart(2, "0") + " " + (h < 12 ? "AM" : "PM"); };

  // ── state ───────────────────────────────────────────────────────────────
  let PID = null, DATA = null, _active = false, _wired = false, _introDone = false;
  let GRAN = (function () { try { return sessionStorage.getItem("tf_gran") || "hour"; } catch (_) { return "hour"; } })();
  let LIVE = true, _refreshSeq = 0, _liveTimer = null;
  let _clockTick = null, _rerenderRaf = null, _dirty = {};
  let _openExpand = null;   // {kind, update} when a lightbox is open
  let _zoom = null;         // {i0,i1} bucket-index window for the hero (null = full)
  let _calMode = "volume";  // calendar heatmap mode (volume | error)
  let _calPhraseIdx = 0;    // which interactive phrase is shown (click cycles)
  let _calSweep = false;    // request the today cell to light-sweep on next render
  let _byMode = "ring";     // bandwidth widget view: ring | river
  let _byPhraseIdx = 0;     // bytes-insight phrase index (click cycles)
  let _rateWin = [];        // rolling [{t, bytes}] for the live MB/s readout
  let _byPulse = 0;         // pending live throughput pulses to flush
  let _rateTimer = null;    // 1 s interval that decays the live rate
  let _hivePulseMethod = null;   // colony to pulse on the next render frame

  // ════════════════════ DATA FETCH ════════════════════════════════════════
  async function refresh() {
    if (!PID) return;
    const myGran = GRAN, mySeq = ++_refreshSeq;
    const { ok, body } = await api(`/api/account/keys/${encodeURIComponent(PID)}/traffic?granularity=${myGran}&tz_off=${TZOFF}`);
    if (mySeq !== _refreshSeq || myGran !== GRAN) return;   // superseded
    if (!ok || !body || body.ok === false) return;
    DATA = body.data || body;
    _zoom = null;
    const stack = $("tf-stack"); if (stack) stack.classList.remove("ov-loading");
    const intro = !_introDone;
    renderAll(intro);
    _introDone = true;
  }
  function renderAll(intro) {
    renderHero(intro); renderKpis(); renderCalendar(intro); renderHive(intro);
    renderClock(intro); renderBytes(intro);
  }

  // ════════════════════ ⓪ REUSABLE REQUEST CONTAINER (drill target) ═══════
  // A lightbox listing every request in a time window, with method/status/
  // endpoint filter chips. Row click → the existing request drawer (no tab
  // jump). Used by hero bars, calendar days, clock hours, KPI/hive drills.
  function openRequestContainer(opts) {
    if (!window.APIN || !APIN.lightbox) { _drillFallback(opts); return; }
    const sinceISO = _utcStr(opts.sinceMs), untilISO = _utcStr(opts.untilMs);
    APIN.lightbox.open({
      title: "Requests · " + (opts.label || "window"),
      subtitle: _local(sinceISO) + "  →  " + _local(untilISO),
      sourceCard: opts.sourceEl || null,
      hashKey: "tfreq",
      build: (panel) => buildRequestContainer(panel, sinceISO, untilISO, opts),
    });
  }
  function _drillFallback(opts) {
    if (window.APIN && APIN.keyDetail && APIN.keyDetail.filterRequests)
      APIN.keyDetail.filterRequests(_utcStr(opts.sinceMs), _utcStr(opts.untilMs));
    else location.hash = "#requests";
  }
  function _methClass(m) { m = (m || "").toUpperCase(); return ["GET","POST","PUT","PATCH","DELETE"].includes(m) ? "meth-" + m : ""; }
  function _statClass(s) { s = +s || 0; return s >= 500 ? "stat-5xx" : s >= 400 ? "stat-4xx" : s >= 300 ? "stat-3xx" : s >= 200 ? "stat-2xx" : ""; }
  async function buildRequestContainer(panel, sinceISO, untilISO, opts) {
    opts = opts || {};
    panel.innerHTML =
      `<div class="tf-rc">
        <div class="tf-rc-bar">
          <div class="tf-rc-chips" id="tf-rc-meth" data-grp="method"></div>
          <div class="tf-rc-chips" id="tf-rc-stat" data-grp="status"></div>
          <input class="tf-rc-search" id="tf-rc-q" type="text" placeholder="filter path…" aria-label="Filter by path">
        </div>
        <div class="tf-rc-meta" id="tf-rc-meta">loading…</div>
        <div class="tf-rc-tablewrap">
          <table class="tf-rc-table"><thead><tr>
            <th>when</th><th>method</th><th>path</th><th>status</th><th style="text-align:right">latency</th><th style="text-align:right">out</th>
          </tr></thead><tbody id="tf-rc-tbody"></tbody></table>
        </div>
      </div>`;
    let rows;
    if (opts.presetRows) {
      // caller already computed the exact row set (e.g. a per-block slice)
      rows = opts.presetRows.slice();
    } else {
      // Time-of-day filters (clock drills) are applied SERVER-SIDE so they span
      // the whole window correctly (the 200-row cap can't miss the target hour).
      let qs = `since=${encodeURIComponent(sinceISO)}&until=${encodeURIComponent(untilISO)}&limit=200`;
      if (opts.localHour != null) qs += `&local_hour=${+opts.localHour}&tz_off=${TZOFF}`;
      if (opts.localWeekday != null) qs += `&local_weekday=${+opts.localWeekday}&tz_off=${TZOFF}`;
      const r = await api(`/api/account/keys/${encodeURIComponent(PID)}/requests?${qs}`);
      rows = (r.body && r.body.data && r.body.data.items) || [];
    }
    let fM = (opts.method || "ALL"), fS = (opts.status || "ALL"), fEp = "";
    const methods = ["ALL"].concat(Array.from(new Set(rows.map(x => (x.method || "").toUpperCase()).filter(Boolean))));
    const statuses = ["ALL", "2xx", "4xx", "5xx"];
    const mHost = panel.querySelector("#tf-rc-meth"), sHost = panel.querySelector("#tf-rc-stat");
    mHost.innerHTML = methods.map(m => `<button class="tf-chip" data-v="${esc(m)}"${m === fM ? ' aria-pressed="true"' : ""}>${esc(m)}</button>`).join("");
    sHost.innerHTML = statuses.map(s => `<button class="tf-chip tf-chip-${s}" data-v="${esc(s)}"${s === fS ? ' aria-pressed="true"' : ""}>${esc(s)}</button>`).join("");
    const tbody = panel.querySelector("#tf-rc-tbody"), meta = panel.querySelector("#tf-rc-meta");
    function inStat(sc) { sc = +sc || 0; return fS === "ALL" || (fS === "2xx" && sc < 400) || (fS === "4xx" && sc >= 400 && sc < 500) || (fS === "5xx" && sc >= 500); }
    function paint() {
      const f = rows.filter(x => (fM === "ALL" || (x.method || "").toUpperCase() === fM) && inStat(x.status_code) && (!fEp || (x.path || "").toLowerCase().includes(fEp)));
      meta.innerHTML = `<b>${fmtNum(f.length)}</b> of ${fmtNum(rows.length)} request(s) in window · <span style="font-style:italic;color:var(--ink-mute)">click a row for full detail</span>`;
      if (!f.length) { tbody.innerHTML = `<tr><td colspan="6" style="padding:22px;text-align:center;color:var(--ink-mute);font-style:italic">no requests match these filters</td></tr>`; return; }
      tbody.innerHTML = f.map(x =>
        `<tr class="tf-rc-row" data-rid="${esc(x.id)}">
          <td title="${esc(x.timestamp)}">${esc(_local(x.timestamp))}</td>
          <td><span class="meth ${_methClass(x.method)}">${esc(x.method || "")}</span></td>
          <td class="tf-rc-path">${esc(x.path || "")}</td>
          <td><span class="stat ${_statClass(x.status_code)}">${esc(x.status_code || "")}</span></td>
          <td style="text-align:right">${x.latency_ms != null ? x.latency_ms + " ms" : "·"}</td>
          <td style="text-align:right">${x.bytes_out != null ? fmtBytes(x.bytes_out) : "·"}</td>
        </tr>`).join("");
    }
    mHost.querySelectorAll(".tf-chip").forEach(b => b.addEventListener("click", () => { fM = b.getAttribute("data-v"); mHost.querySelectorAll(".tf-chip").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); paint(); }));
    sHost.querySelectorAll(".tf-chip").forEach(b => b.addEventListener("click", () => { fS = b.getAttribute("data-v"); sHost.querySelectorAll(".tf-chip").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); paint(); }));
    const q = panel.querySelector("#tf-rc-q");
    if (q) q.addEventListener("input", () => { fEp = q.value.trim().toLowerCase(); paint(); });
    tbody.addEventListener("click", (e) => {
      const tr = e.target.closest(".tf-rc-row"); if (!tr) return;
      const rid = tr.getAttribute("data-rid");
      if (rid && window.APIN && APIN.keyDetail && APIN.keyDetail.openRequest) APIN.keyDetail.openRequest(rid);
    });
    paint();
  }

  // ════════════════════ ① HERO — segmented blocks + draggable scrubber ════
  const HERO_W = 760, HERO_H = 200, MAXBLOCKS = 14, BLK_H = 9, BLK_GAP = 3, BASE_OFF = 30;
  // a slice's local time-range label, e.g. "2:00 PM – 2:08 PM" or, across days,
  // "May 27 11:00 PM – May 28 1:00 AM".
  function _sliceTip(sinceMs, untilMs) {
    const d1 = new Date(sinceMs), d2 = new Date(untilMs);
    const t = (d) => { const h = d.getHours(); return ((h % 12) || 12) + ":" + String(d.getMinutes()).padStart(2, "0") + " " + (h < 12 ? "AM" : "PM"); };
    const day = (d) => d.toLocaleDateString([], { month: "short", day: "numeric" });
    return d1.toDateString() === d2.toDateString() ? `${t(d1)} – ${t(d2)}` : `${day(d1)} ${t(d1)} – ${day(d2)} ${t(d2)}`;
  }
  // Click a block → fetch its bucket, sort by time, take the k-th equal-COUNT
  // chunk, and open the container with that exact row set + its real time-span.
  async function _drillSlice(b, k, blocks) {
    if (!b || !b.total) return;
    const dur = DATA.bucket_ms || GRAN_MS[GRAN] || 3600e3;
    const r = await api(`/api/account/keys/${encodeURIComponent(PID)}/requests?since=${encodeURIComponent(_utcStr(b.t_ms))}&until=${encodeURIComponent(_utcStr(b.t_ms + dur))}&limit=200`);
    let rows = (r.body && r.body.data && r.body.data.items) || [];
    rows.sort((x, y) => (x.timestamp < y.timestamp ? -1 : x.timestamp > y.timestamp ? 1 : 0));  // ascending
    const N = rows.length;
    let lo = Math.floor(k * N / blocks), hi = Math.floor((k + 1) * N / blocks);
    if (hi <= lo && lo < N) hi = lo + 1;
    const chunk = rows.slice(lo, hi);
    const base = b.label + (b.sub ? " " + b.sub : "");
    let label = base + " · slice " + (k + 1) + "/" + blocks, sMs = b.t_ms, uMs = b.t_ms + dur;
    if (chunk.length) {
      const ts = (s) => new Date((s || "").replace(" ", "T") + "Z").getTime();
      sMs = ts(chunk[0].timestamp); uMs = ts(chunk[chunk.length - 1].timestamp) + 1000;
      label = base + " · " + _sliceTip(sMs, uMs);
    }
    openRequestContainer({ presetRows: chunk, sinceMs: sMs, untilMs: uMs, label });
  }
  function heroSvg(buckets, max, off) {
    const col = _C(), n = buckets.length || 1;
    const slotW = HERO_W / n, bw = Math.max(6, Math.min(30, slotW - 7));
    const baseY = HERO_H - BASE_OFF;
    const stride = Math.max(1, Math.ceil(n / 12));
    let bars = "", labels = "", topHits = "", sliceHits = "";
    buckets.forEach((b, i) => {
      const ai = i + (off || 0);              // absolute bucket index
      const cx = i * slotW + slotW / 2, x = cx - bw / 2, slotX = i * slotW;
      const blocks = b.total > 0 ? Math.max(1, Math.round(b.total / (max || 1) * MAXBLOCKS)) : 0;
      const g = Math.round(blocks * (b.n2 / (b.total || 1)));
      const a = Math.round(blocks * (b.n4 / (b.total || 1)));
      // colored blocks (k=0 is the bottom block = earliest slice of the window)
      for (let k = 0; k < blocks; k++) {
        const yTop = baseY - (k + 1) * BLK_H;
        const st = k < g ? "n2" : k < g + a ? "n4" : "n5";
        const color = st === "n2" ? col.ok : st === "n4" ? col.amber : col.danger;
        bars += `<rect class="tf-blk" data-i="${ai}" data-k="${k}" x="${x.toFixed(1)}" y="${yTop.toFixed(1)}" width="${bw.toFixed(1)}" height="${BLK_H - BLK_GAP}" rx="2" fill="${color}"/>`;
        // a transparent hit slice over this block (covers the gap too)
        sliceHits += `<rect class="tf-slice" data-i="${ai}" data-k="${k}" data-blocks="${blocks}" x="${slotX.toFixed(1)}" y="${yTop.toFixed(1)}" width="${slotW.toFixed(1)}" height="${BLK_H}" fill="transparent"/>`;
      }
      if (i % stride === 0) {
        labels += `<text x="${cx.toFixed(1)}" y="${baseY + 13}" text-anchor="middle" style="font:9.5px 'JetBrains Mono',monospace;fill:var(--ink-soft)">${esc(b.label)}</text>`;
        if (b.sub) labels += `<text x="${cx.toFixed(1)}" y="${baseY + 24}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${esc(b.sub)}</text>`;
      }
      // whole-bucket hit covers ONLY the empty area above the bar
      const barTop = baseY - blocks * BLK_H;
      topHits += `<rect class="tf-bar" data-i="${ai}" x="${slotX.toFixed(1)}" y="0" width="${slotW.toFixed(1)}" height="${Math.max(0, barTop).toFixed(1)}" fill="transparent"/>`;
    });
    return `<svg class="tf-hero-svg" viewBox="0 0 ${HERO_W} ${HERO_H}" preserveAspectRatio="none" style="height:${HERO_H}px">
      <line x1="0" y1="${baseY}" x2="${HERO_W}" y2="${baseY}" stroke="var(--paper-edge)" stroke-width="1"/>${bars}${labels}${topHits}${sliceHits}</svg>`;
  }
  // scrubber strip — handles drawn at local x=0 and positioned via transform so
  // they can be repositioned in place during a drag (no SVG rebuild).
  const BR_W = 760, BR_H = 36;
  function brushSvg(buckets, max, z) {
    const col = _C(), n = buckets.length || 1, sw = BR_W / n;
    let bars = "";
    buckets.forEach((b, i) => {
      const h = b.total > 0 ? Math.max(2, (b.total / (max || 1)) * (BR_H - 8)) : 1;
      const er = b.total ? (b.n4 + b.n5) / b.total : 0;
      const fill = b.total === 0 ? "rgba(120,110,90,.14)" : er > 0.1 ? col.danger : er > 0 ? col.amber : col.accent;
      bars += `<rect x="${(i * sw + 1).toFixed(1)}" y="${(BR_H - h).toFixed(1)}" width="${Math.max(1, sw - 2).toFixed(1)}" height="${h.toFixed(1)}" rx="1" fill="${fill}" fill-opacity="${b.total ? 0.7 : 1}"/>`;
    });
    const x0 = (z.i0 / n) * BR_W, x1 = ((z.i1 + 1) / n) * BR_W;
    const shade = `<rect class="tf-brush-shade-l" x="0" y="0" width="${x0.toFixed(1)}" height="${BR_H}" fill="var(--paper)" fill-opacity=".6" pointer-events="none"/>
      <rect class="tf-brush-shade-r" x="${x1.toFixed(1)}" y="0" width="${(BR_W - x1).toFixed(1)}" height="${BR_H}" fill="var(--paper)" fill-opacity=".6" pointer-events="none"/>`;
    const gripDots = [0.32, 0.5, 0.68].map(f => `<circle cx="0" cy="${(BR_H * f).toFixed(1)}" r="1.1" fill="var(--paper)"/>`).join("");
    const handle = (edge, gx) => `<g class="tf-brush-h" data-edge="${edge}" transform="translate(${gx.toFixed(1)},0)" style="cursor:ew-resize"><rect x="-4.5" y="0" width="9" height="${BR_H}" rx="2" fill="var(--ink)"/>${gripDots}</g>`;
    const win = `<rect class="tf-brush-win" x="${x0.toFixed(1)}" y="0" width="${(x1 - x0).toFixed(1)}" height="${BR_H}" fill="rgba(82,183,136,.12)" stroke="var(--ink)" stroke-width="1.3" style="cursor:grab"/>`;
    return `<svg class="tf-brush-svg" viewBox="0 0 ${BR_W} ${BR_H}" preserveAspectRatio="none" style="height:${BR_H}px;width:100%">${bars}${shade}${win}${handle("0", x0)}${handle("1", x1)}</svg>`;
  }
  function _heroWindow(total) {
    const n = total;
    if (!_zoom) return { i0: 0, i1: n - 1 };
    return { i0: Math.max(0, Math.min(_zoom.i0, n - 1)), i1: Math.max(0, Math.min(_zoom.i1, n - 1)) };
  }
  // Build the shell once; main chart + brush are then updated in place.
  function heroChart(host, intro) {
    const h = DATA.hero || { buckets: [], max: 0 };
    if (!h.buckets.length || h.max === 0) { host.innerHTML = `<div class="tf-empty">No traffic in this range yet.</div>`; return; }
    const z = _heroWindow(h.buckets.length);
    host.innerHTML =
      `<div class="tf-hero-main"></div>
       <div class="tf-brush-wrap"><div class="tf-brush-cap"></div>${brushSvg(h.buckets, h.max, z)}</div>`;
    _renderHeroMain(host, intro);
    _wireBrush(host, h.buckets);
  }
  // Re-render ONLY the main chart + cap (cheap; safe to call every drag frame).
  function _renderHeroMain(host, intro) {
    const h = DATA.hero || { buckets: [] };
    const z = _heroWindow(h.buckets.length);
    const slice = h.buckets.slice(z.i0, z.i1 + 1);
    const sliceMax = Math.max(1, ...slice.map(b => b.total));
    const main = host.querySelector(".tf-hero-main"); if (!main) return;
    main.innerHTML = heroSvg(slice, sliceMax, z.i0);
    _wireHeroMain(main, h.buckets, intro);
    const cap = host.querySelector(".tf-brush-cap");
    if (cap) {
      const zoomed = z.i0 > 0 || z.i1 < h.buckets.length - 1;
      cap.innerHTML = zoomed
        ? `showing ${z.i0 + 1}–${z.i1 + 1} of ${h.buckets.length} · <button class="tf-brush-reset" id="tf-brush-reset">reset</button>`
        : `drag the edge handles to zoom · grab the window to slide`;
      const rs = cap.querySelector("#tf-brush-reset");
      if (rs) rs.addEventListener("click", () => { _zoom = null; _renderHeroMain(host, false); _positionBrush(host, _heroWindow(h.buckets.length), h.buckets.length); });
    }
  }
  // Reposition the brush window/handles/shades by mutating attributes only.
  function _positionBrush(host, z, n) {
    const svg = host.querySelector(".tf-brush-svg"); if (!svg) return;
    const x0 = (z.i0 / n) * BR_W, x1 = ((z.i1 + 1) / n) * BR_W;
    const win = svg.querySelector(".tf-brush-win"); if (win) { win.setAttribute("x", x0.toFixed(1)); win.setAttribute("width", Math.max(0, x1 - x0).toFixed(1)); }
    const sl = svg.querySelector(".tf-brush-shade-l"); if (sl) sl.setAttribute("width", x0.toFixed(1));
    const sr = svg.querySelector(".tf-brush-shade-r"); if (sr) { sr.setAttribute("x", x1.toFixed(1)); sr.setAttribute("width", Math.max(0, BR_W - x1).toFixed(1)); }
    const h0 = svg.querySelector('.tf-brush-h[data-edge="0"]'); if (h0) h0.setAttribute("transform", `translate(${x0.toFixed(1)},0)`);
    const h1 = svg.querySelector('.tf-brush-h[data-edge="1"]'); if (h1) h1.setAttribute("transform", `translate(${x1.toFixed(1)},0)`);
  }
  function _wireHeroMain(main, buckets, intro) {
    const svg = main.querySelector("svg"); if (!svg) return;
    const dur = DATA.bucket_ms || GRAN_MS[GRAN] || 3600e3;
    // per-block SLICE: each block ≈ an equal share of the bar's requests.
    // Hover shows the approx count; click opens that chronological chunk with
    // its real time-span (robust to bursty traffic — never an empty slice).
    svg.querySelectorAll(".tf-slice").forEach(sl => {
      const ai = +sl.getAttribute("data-i"), k = +sl.getAttribute("data-k"), blocks = +sl.getAttribute("data-blocks");
      sl.style.cursor = "pointer";
      sl.addEventListener("mousemove", (e) => {
        const b = buckets[ai]; if (!b) return;
        const approx = Math.max(1, Math.round(b.total / blocks));
        tip(true, e.clientX, e.clientY, `<b>${esc(b.label)}${b.sub ? " · " + esc(b.sub) : ""}</b><br>slice ${k + 1} of ${blocks} · ≈${fmtNum(approx)} requests<br><span style="opacity:.7">click → this slice of requests</span>`);
        svg.querySelectorAll(`.tf-blk[data-i="${ai}"]`).forEach(blk => { blk.style.opacity = (+blk.getAttribute("data-k") === k) ? "1" : "0.3"; });
      });
      sl.addEventListener("mouseleave", () => { tip(false); svg.querySelectorAll(".tf-blk").forEach(blk => blk.style.opacity = "1"); });
      sl.addEventListener("click", (e) => { e.stopPropagation(); _drillSlice(buckets[ai], k, blocks); });
    });
    // empty area above a bar → drill the whole bucket
    svg.querySelectorAll(".tf-bar").forEach(hrect => {
      const ai = +hrect.getAttribute("data-i");
      hrect.addEventListener("mousemove", (e) => {
        const b = buckets[ai]; if (!b) return;
        const errp = b.total ? Math.round(100 * (b.n4 + b.n5) / b.total) : 0;
        tip(true, e.clientX, e.clientY, `<b>${esc(b.label)}${b.sub ? " · " + esc(b.sub) : ""}</b><br>${fmtNum(b.total)} req · 2xx ${b.n2} · 4xx ${b.n4} · 5xx ${b.n5} · ${errp}% err<br><span style="opacity:.7">click → all requests · click a block for a time slice</span>`);
      });
      hrect.addEventListener("mouseleave", () => tip(false));
      hrect.addEventListener("click", () => {
        const b = buckets[ai];
        if (b && b.total) openRequestContainer({ sinceMs: b.t_ms, untilMs: b.t_ms + dur, label: b.label + (b.sub ? " " + b.sub : "") });
      });
    });
    if (intro && window.APIN && APIN.fx) svg.querySelectorAll(".tf-blk").forEach((r, k) => {
      try { r.animate([{ transform: "scaleY(0)", transformOrigin: "center bottom" }, { transform: "scaleY(1)", transformOrigin: "center bottom" }], { duration: 320, delay: Math.min(600, k * 4), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {}
    });
  }
  // Real scrubber: drag an edge handle to RESIZE, grab the window to PAN.
  // Updates happen IN PLACE (main re-render + brush attr mutation) so the brush
  // SVG stays attached and getBoundingClientRect keeps working mid-drag.
  function _wireBrush(host, buckets) {
    const svg = host.querySelector(".tf-brush-svg"); if (!svg) return;
    const n = buckets.length;
    const idxAt = (clientX) => {
      const r = svg.getBoundingClientRect();
      if (!r.width) return 0;
      return Math.max(0, Math.min(n - 1, Math.floor(((clientX - r.left) / r.width) * n)));
    };
    const apply = () => { _renderHeroMain(host, false); _positionBrush(host, _heroWindow(n), n); };
    let mode = null, grabIdx = 0, origI0 = 0, origI1 = n - 1;
    const onMove = (e) => {
      if (!mode) return;
      const i = idxAt(e.clientX), cur = _zoom || { i0: 0, i1: n - 1 };
      if (mode === "e0") _zoom = { i0: Math.min(i, cur.i1), i1: cur.i1 };
      else if (mode === "e1") _zoom = { i0: cur.i0, i1: Math.max(i, cur.i0) };
      else if (mode === "pan") { const w = origI1 - origI0 + 1; const ni0 = Math.max(0, Math.min(n - w, origI0 + (i - grabIdx))); _zoom = { i0: ni0, i1: ni0 + w - 1 }; }
      apply();
    };
    const onUp = () => { mode = null; window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); document.body.style.userSelect = ""; const w = svg.querySelector(".tf-brush-win"); if (w) w.style.cursor = "grab"; };
    svg.addEventListener("mousedown", (e) => {
      const edge = e.target.closest(".tf-brush-h");
      const cur = _heroWindow(n); origI0 = cur.i0; origI1 = cur.i1;
      if (edge) { mode = edge.getAttribute("data-edge") === "0" ? "e0" : "e1"; _zoom = { i0: cur.i0, i1: cur.i1 }; }
      else {
        mode = "pan"; grabIdx = idxAt(e.clientX);
        // grab outside the window → recenter on the cursor first, then pan
        if (grabIdx < cur.i0 || grabIdx > cur.i1) {
          const w = cur.i1 - cur.i0 + 1, ni0 = Math.max(0, Math.min(n - w, grabIdx - Math.floor(w / 2)));
          origI0 = ni0; origI1 = ni0 + w - 1; grabIdx = ni0 + Math.floor(w / 2); _zoom = { i0: origI0, i1: origI1 };
        }
        const win = svg.querySelector(".tf-brush-win"); if (win) win.style.cursor = "grabbing";
      }
      document.body.style.userSelect = "none";
      window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
      e.preventDefault(); apply();
    });
  }
  function renderHero(intro) {
    const host = $("tf-hero-body"); if (!host || !DATA) return;
    const aux = $("tf-hero-aux"); if (aux) aux.textContent = GRAN_LABEL[GRAN] || "";
    heroChart(host, intro);
  }

  // ════════════════════ ② KPI TILES (4 · expandable) ══════════════════════
  const KPI_DEFS = [
    { k: "total", tfx: "kpiTotal", lbl: "total requests", icon: "i-activity" },
    { k: "peak",  tfx: "kpiPeak",  lbl: "peak / min",     icon: "i-zap" },
    { k: "busy",  tfx: "kpiBusy",  lbl: "busiest bucket", icon: "i-bar-chart" },
    { k: "data",  tfx: "kpiData",  lbl: "data transferred", icon: "i-download" },
  ];
  function _kpiValue(k) {
    const s = DATA.stats || {}, by = DATA.bytes || {};
    if (k === "total") return { val: fmtNum(s.total || 0), num: true, sub: `${s.error_pct || 0}% errors` };
    if (k === "peak")  return { val: fmtNum(s.peak_per_min || 0), num: true, sub: "requests / minute" };
    if (k === "busy")  return { val: esc(s.busiest_label || "—"), num: false, sub: `${fmtNum(s.busiest_count || 0)} requests` };
    if (k === "data")  return { val: fmtBytes((by.total_in || 0) + (by.total_out || 0)), num: false, sub: `${fmtBytes(by.avg_out || 0)}/req out` };
    return { val: "—", num: false, sub: "" };
  }
  function renderKpis() {
    const host = $("tf-kpis"); if (!host || !DATA) return;
    if (!host.dataset.built) {
      host.innerHTML = KPI_DEFS.map(d =>
        `<div class="ov-card tf-kpi" data-kpi="${d.k}">
          <button class="ov-expand" data-tfx="${d.tfx}" aria-label="Expand ${esc(d.lbl)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><use href="#i-expand"/></svg></button>
          <div class="tf-kpi-head"><svg class="icon" aria-hidden="true"><use href="#${d.icon}"/></svg><span>${esc(d.lbl)}</span></div>
          <div class="tf-kpi-val" data-num>—</div>
          <div class="tf-kpi-sub" data-sub></div>
        </div>`).join("");
      host.dataset.built = "1";
    }
    KPI_DEFS.forEach(d => {
      const tile = host.querySelector(`.tf-kpi[data-kpi="${d.k}"]`); if (!tile) return;
      const v = _kpiValue(d.k);
      const numEl = tile.querySelector("[data-num]"), subEl = tile.querySelector("[data-sub]");
      if (v.num) slot(numEl, Number(String(v.val).replace(/,/g, "")) || 0);
      else { numEl.classList.remove("apin-odometer"); numEl.textContent = v.val; }
      subEl.textContent = v.sub;
    });
  }

  // ════════════════════ ③ CALENDAR — GitHub heatmap ═══════════════════════
  function _calGrid(days, weeks) {
    const byDate = {}; (days || []).forEach(d => byDate[d.date] = d);
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const cells = [];
    const start = new Date(today); start.setDate(start.getDate() - (weeks * 7 - 1));
    const sd = (start.getDay() + 6) % 7; start.setDate(start.getDate() - sd);
    let max = 0;
    for (let c = 0; ; c++) {
      let any = false;
      for (let r = 0; r < 7; r++) {
        const dt = new Date(start); dt.setDate(start.getDate() + c * 7 + r);
        if (dt > today) break;
        any = true;
        const key = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`;
        const rec = byDate[key];
        const n = rec ? rec.n : 0, e = rec ? rec.e : 0;
        if (n > max) max = n;
        cells.push({ c, r, key, n, e, dt: dt.getTime(), label: dt.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" }) });
      }
      if (!any) break;
      if (c > weeks + 2) break;
    }
    return { cells, max, ncols: cells.reduce((m, x) => Math.max(m, x.c), 0) + 1 };
  }
  // ── colour ramps (single-hue sequential for volume, ordered for error) ──
  const _MIX = (a, b, t) => `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
  const _VOL_PALE = [216, 238, 226], _VOL_ACC = [82, 183, 136], _VOL_DEEP = [21, 56, 40];   // pale → accent → forest-ink
  const _ERR_OK = [47, 111, 62], _ERR_AM = [201, 138, 43], _ERR_DA = [179, 64, 47];          // ok → amber → danger
  function _volColor(t) { t = Math.max(0, Math.min(1, t)); return t < 0.55 ? _MIX(_VOL_PALE, _VOL_ACC, t / 0.55) : _MIX(_VOL_ACC, _VOL_DEEP, (t - 0.55) / 0.45); }
  function _errColor(r) { r = Math.max(0, Math.min(1, r)); return r < 0.5 ? _MIX(_ERR_OK, _ERR_AM, r / 0.5) : _MIX(_ERR_AM, _ERR_DA, (r - 0.5) / 0.5); }
  const _calAvg = () => { const a = ((DATA.calendar || {}).days || []).filter(d => d.n > 0); return a.length ? a.reduce((s, d) => s + d.n, 0) / a.length : 0; };

  function calSvg(days, weeks, mode, cell) {
    const g = _calGrid(days, weeks);
    cell = cell || 18; const gap = 4, ox = 26, oy = 18;
    const W = ox + g.ncols * (cell + gap), H = oy + 7 * (cell + gap);
    const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const todayKey = (() => { const t = new Date(); return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`; })();
    let cells = "", monLbls = "", lastMon = -1;
    let hits = "";
    g.cells.forEach(c => {
      const x = ox + c.c * (cell + gap), y = oy + c.r * (cell + gap), live = c.key === todayKey;
      const liveRing = live ? `<rect class="tf-cal-livering" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="4" fill="none" stroke="var(--ink)" stroke-width="1.5" pointer-events="none"/>` : "";
      if (c.n > 0) {
        let fill, op = 1, glow = 0;
        if (mode === "error") { const rate = c.e / c.n, cn = Math.min(1, c.n / (g.max || 1)); fill = _errColor(rate); op = 0.34 + 0.66 * cn; glow = rate > 0.08 ? (0.35 + 0.65 * cn) : 0; }
        else { const t = Math.pow(c.n / (g.max || 1), 0.7); fill = _volColor(t); glow = t > 0.6 ? (t - 0.6) / 0.4 : 0; }
        const hot = glow > 0.04;
        cells += `<rect class="tf-cal-cell${hot ? " tf-cal-hot" : ""}${live ? " tf-cal-live" : ""}" data-key="${c.key}" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="4" fill="${fill}" fill-opacity="${op.toFixed(2)}" style="--gc:${fill}${hot ? `;--g:${(1.5 + glow * 6).toFixed(1)}px` : ""}" pointer-events="none"/>`
          + `<rect class="tf-cal-sheen" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="4" fill="url(#tfSheen)" pointer-events="none"/>` + liveRing;
      } else {
        cells += `<rect class="tf-cal-cell tf-cal-empty${live ? " tf-cal-live" : ""}" data-key="${c.key}" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="4" fill="var(--paper-deep,#e9e2d1)" fill-opacity="0.45" stroke="rgba(26,22,18,0.07)" stroke-width="1" pointer-events="none"/>`
          + `<circle cx="${(x + cell / 2).toFixed(1)}" cy="${(y + cell / 2).toFixed(1)}" r="${Math.max(1, cell * 0.08).toFixed(1)}" fill="rgba(26,22,18,0.13)" pointer-events="none"/>` + liveRing;
      }
      // topmost transparent hit target — guarantees hover/click land here
      hits += `<rect class="tf-cal-hit" data-key="${c.key}" data-ms="${c.dt}" data-n="${c.n}" data-e="${c.e}" data-label="${esc(c.label)}" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="4" fill="transparent" style="cursor:${c.n ? "pointer" : "default"}"/>`;
      if (c.r === 0) { const mo = new Date(c.dt).getMonth(); if (mo !== lastMon) { lastMon = mo; monLbls += `<text x="${x}" y="12" style="font:9px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${MON[mo]}</text>`; } }
    });
    const wd = ["", "Mon", "", "Wed", "", "Fri", ""].map((d, r) => d ? `<text x="0" y="${oy + r * (cell + gap) + cell - 3}" style="font:8.5px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${d}</text>` : "").join("");
    const defs = `<defs><linearGradient id="tfSheen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff" stop-opacity="0.24"/><stop offset="0.5" stop-color="#ffffff" stop-opacity="0.05"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient></defs>`;
    return `<svg class="tf-cal-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="max-width:100%;max-height:${H + 4}px;overflow:visible">${defs}${monLbls}${wd}${cells}${hits}</svg>`;
  }
  function _wireCal(host) {
    host.querySelectorAll(".tf-cal-hit").forEach(h => {
      const key = h.getAttribute("data-key");
      const vis = host.querySelector('.tf-cal-cell[data-key="' + key + '"]');
      h.addEventListener("mousemove", (e) => {
        const n = +h.getAttribute("data-n"), er = +h.getAttribute("data-e"), label = h.getAttribute("data-label");
        if (!n) { tip(true, e.clientX, e.clientY, `<b>${esc(label)}</b><br><span style="opacity:.7">no traffic</span>`); return; }
        const errp = Math.round(100 * er / n), avg = _calAvg();
        const cmp = avg ? `<br>${(n / avg) >= 1 ? (n / avg).toFixed(1) + "× your daily average" : Math.round(100 * n / avg) + "% of your daily average"}` : "";
        tip(true, e.clientX, e.clientY, `<b>${esc(label)}</b><br>${fmtNum(n)} requests · ${errp}% errors${cmp}`);
        if (vis && !vis.classList.contains("tf-cal-empty")) vis.classList.add("tf-cal-cell-hover");
      });
      h.addEventListener("mouseleave", () => { tip(false); if (vis) vis.classList.remove("tf-cal-cell-hover"); });
      h.addEventListener("click", () => { const ms = +h.getAttribute("data-ms"), n = +h.getAttribute("data-n"); if (ms && n) openRequestContainer({ sinceMs: ms, untilMs: ms + 86400e3, label: new Date(ms).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" }) }); });
    });
  }
  // ── interactive phrase engine — many data-driven, personal variations ──
  function _calInsights() {
    const C = DATA.calendar || { days: [] }, S = DATA.stats || {}, CL = DATA.clock || {};
    const days = (C.days || []).slice().sort((a, b) => a.date < b.date ? -1 : 1);
    const active = days.filter(d => d.n > 0);
    if (!active.length) return ["Fresh key — your traffic map starts here."];
    const WDl = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const fmtD = (s) => new Date(s + "T00:00:00").toLocaleDateString([], { month: "short", day: "numeric" });
    const wdOf = (s) => new Date(s + "T00:00:00").getDay();
    const todayKey = (() => { const t = new Date(); return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`; })();
    const today = days.find(d => d.date === todayKey) || { date: todayKey, n: 0, e: 0 };
    const total = active.reduce((s, d) => s + d.n, 0), avg = total / active.length;
    const busiest = active.reduce((m, d) => d.n > m.n ? d : m, active[0]);
    const out = [], push = (sc, t) => out.push({ sc, t });
    let streak = 0; { const t0 = new Date(); t0.setHours(0, 0, 0, 0); for (let i = 0; ; i++) { const dt = new Date(t0); dt.setDate(dt.getDate() - i); const k = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`; const r = days.find(d => d.date === k); if (r && r.n > 0) streak++; else break; } }
    // today
    const tWd = wdOf(today.date), sameWd = active.filter(d => wdOf(d.date) === tWd);
    if (today.n > 0) {
      const errp = Math.round(100 * today.e / today.n);
      if (sameWd.length >= 2 && today.n >= Math.max(...sameWd.map(d => d.n))) push(96, `${fmtNum(today.n)} requests today — your busiest ${WDl[tWd]} yet.`);
      else if (today.n >= 2 * avg) push(90, `Today's already ${(today.n / avg).toFixed(1)}× a typical day — ${fmtNum(today.n)} requests.`);
      else if (today.n < 0.5 * avg) push(58, `Quiet so far today — just ${fmtNum(today.n)} request${today.n === 1 ? "" : "s"}.`);
      else push(54, `${fmtNum(today.n)} requests today · ${errp}% errors.`);
    } else push(38, `No traffic yet today — the grid's waiting.`);
    // streak
    if (streak >= 3) push(82, `${streak}-day active streak${streak >= active.length ? " — your longest yet" : ""}.`);
    else if (streak === 2) push(46, `Two days running.`);
    // peak
    const daysAgo = Math.round((Date.now() - new Date(busiest.date + "T00:00:00").getTime()) / 864e5);
    if (daysAgo <= 2) push(86, `New high-water mark: ${fmtNum(busiest.n)} requests on ${fmtD(busiest.date)}.`);
    else push(73, `Last peak was ${daysAgo} days ago — ${fmtD(busiest.date)} · ${fmtNum(busiest.n)} requests.`);
    // weekday dominance + weekend share
    const wt = [0, 0, 0, 0, 0, 0, 0]; active.forEach(d => wt[wdOf(d.date)] += d.n);
    const dom = wt.indexOf(Math.max(...wt)), domPct = Math.round(100 * wt[dom] / total);
    if (domPct >= 35) push(67, `${WDl[dom]}s carry ${domPct}% of your traffic.`);
    const wkndPct = Math.round(100 * (wt[0] + wt[6]) / total);
    if (wkndPct <= 8) push(57, `Weekends stay quiet — ${wkndPct}% of all volume.`);
    else if (wkndPct >= 45) push(63, `Weekend-heavy — ${wkndPct}% lands on Sat/Sun.`);
    // week over week
    const wsum = (off) => { let s = 0; const t0 = new Date(); t0.setHours(0, 0, 0, 0); for (let i = 0; i < 7; i++) { const dt = new Date(t0); dt.setDate(dt.getDate() - (off * 7 + i)); const k = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`; const r = days.find(d => d.date === k); if (r) s += r.n; } return s; };
    const w0 = wsum(0), w1 = wsum(1);
    if (w1 > 0) { const ch = Math.round(100 * (w0 - w1) / w1); if (ch >= 15) push(71, `This week is up ${ch}% on last.`); else if (ch <= -15) push(65, `This week is down ${Math.abs(ch)}% on last.`); }
    // errors
    const totErr = active.reduce((s, d) => s + d.e, 0);
    if (totErr === 0 && active.length >= 3) push(75, `Clean sheet — zero errors across ${active.length} active days.`);
    const errDay = active.slice().sort((a, b) => (b.e / (b.n || 1)) - (a.e / (a.n || 1)))[0];
    if (errDay && errDay.n >= 10 && errDay.e / errDay.n >= 0.2) push(77, `Error spike on ${fmtD(errDay.date)} — ${Math.round(100 * errDay.e / errDay.n)}% of ${fmtNum(errDay.n)} requests.`);
    // clock rhythm
    if (CL.busiest_h != null) { const h = CL.busiest_h, band = (h >= 22 || h < 6) ? `Night owl — busiest around ${fmt12(h)}` : h < 12 ? `Morning rush — peak around ${fmt12(h)}` : h < 17 ? `Afternoon peak around ${fmt12(h)}` : `Evening peak around ${fmt12(h)}`; push(61, band + "."); }
    // milestone + breadth
    if (total >= 1000) push(49, `${(total / 1000).toFixed(total >= 10000 ? 0 : 1)}k+ requests mapped here.`);
    push(34, `${active.length} active day${active.length > 1 ? "s" : ""} · ${fmtNum(total)} requests in view.`);
    out.sort((a, b) => b.sc - a.sc);
    return out.map(o => o.t);
  }
  function _renderCalPhrase() {
    const ph = $("tf-cal-phrase"); if (!ph) return;
    const list = _calInsights(); if (!list.length) { ph.innerHTML = ""; return; }
    const idx = ((_calPhraseIdx % list.length) + list.length) % list.length;
    ph.innerHTML = `<span class="tf-cal-phrase-txt">${esc(list[idx])}</span>` + (list.length > 1 ? `<svg class="tf-cal-phrase-cyc" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><use href="#i-refresh"/></svg>` : "");
    ph.dataset.count = list.length;
  }
  function renderCalendar(intro, modeOverride, weeks, cell) {
    const host = $("tf-cal-body"); if (!host || !DATA) return;
    const mode = modeOverride || _calMode;
    const swatch = (t) => `<i style="background:${mode === "error" ? _errColor(t) : _volColor(t)}"></i>`;
    host.innerHTML =
      `<div class="tf-cal-phrase" id="tf-cal-phrase" role="button" tabindex="0" aria-label="Cycle insight"></div>
       <div class="tf-cal-grid">${calSvg((DATA.calendar || {}).days, weeks || 18, mode, cell)}</div>
       <div class="tf-cal-foot">
         <span class="tf-cal-legend">less ${[0.06, 0.3, 0.55, 0.8, 1].map(swatch).join("")} more</span>
         <div class="ov-range" id="tf-cal-mode"><button data-m="volume"${mode === "volume" ? ' aria-pressed="true"' : ""}>volume</button><button data-m="error"${mode === "error" ? ' aria-pressed="true"' : ""}>error rate</button></div>
       </div>`;
    _wireCal(host);
    _renderCalPhrase();
    const ph = $("tf-cal-phrase");
    if (ph) ph.addEventListener("click", () => { _calPhraseIdx++; _renderCalPhrase(); });
    host.querySelectorAll("#tf-cal-mode button").forEach(b => b.addEventListener("click", () => { if (b.getAttribute("data-m") === _calMode) return; _calMode = b.getAttribute("data-m"); renderCalendar(false); }));
    if (intro && window.APIN && APIN.fx) host.querySelectorAll(".tf-cal-cell").forEach((c, k) => { try { c.animate([{ opacity: 0, transform: "scale(.6)" }, { opacity: 1, transform: "scale(1)" }], { duration: 280, delay: Math.min(700, k * 5), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
    if (_calSweep) { _calSweep = false; const lc = host.querySelector(".tf-cal-live:not(.tf-cal-empty)") || host.querySelector(".tf-cal-live"); if (lc) { lc.classList.add("tf-cal-sweep"); setTimeout(() => lc.classList.remove("tf-cal-sweep"), 760); } }
  }

  // ════════════════════ ④ HIVE — Method × Status honeycomb ════════════════
  function _hexPath(cx, cy, r) {
    let p = "";
    for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 90); p += (i ? "L" : "M") + (cx + r * Math.cos(a)).toFixed(1) + " " + (cy + r * Math.sin(a)).toFixed(1) + " "; }
    return p + "Z";
  }
  const _kc = (n) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n);
  // ── Connected Adaptive Hive ────────────────────────────────────────────
  // Each METHOD is a colony: a circular core node (◉ total) connected to its
  // 2xx/4xx/5xx status cells in a small fan. Encodings:
  //   hex size  = request volume        color   = status category
  //   glow      = live activity         border  = error severity / anomaly
  //   texture   = latency class (per-cell stripe density, per-method hatch angle)
  // Zero cells are rendered as faint DORMANT hexes (asleep, not missing).
  const _STAT = [{ k: "n2", sb: "2", lbl: "2xx", cv: "ok" }, { k: "n4", sb: "4", lbl: "4xx", cv: "amber" }, { k: "n5", sb: "5", lbl: "5xx", cv: "danger" }];
  function _hatchAngle(method) {            // stable per-method hatch angle (colony identity)
    let h = 0; const s = String(method || "?"); for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return 20 + (h % 5) * 30;               // 20,50,80,110,140°
  }
  function _latBand(ms) { return ms == null ? null : ms < 100 ? "fast" : ms < 500 ? "med" : "slow"; }
  function _cellLatBand(cell) {             // dominant latency band of a cell from its bands
    if (!cell) return null;
    if (cell.lat != null) return _latBand(cell.lat);
    const b = { fast: cell.lf || 0, med: cell.lm || 0, slow: cell.ls || 0 };
    return Object.keys(b).sort((x, y) => b[y] - b[x])[0];
  }
  // latency texture: stripe spacing + weight per band; fast reads dense (fine),
  // slow reads boldly striped. Hatch angle is the colony's shared identity.
  const _BAND_TEX = { fast: { sp: 3, w: 0.8, o: 0.16 }, med: { sp: 5, w: 1.4, o: 0.34 }, slow: { sp: 8, w: 2.2, o: 0.52 } };
  function _hiveDefs(matrix) {
    let defs = "";
    const seen = {};
    matrix.forEach(m => {
      const ang = _hatchAngle(m.method);
      ["fast", "med", "slow"].forEach(band => {
        const id = `tfhx-${_methSafe(m.method)}-${band}`; if (seen[id]) return; seen[id] = 1;
        const t = _BAND_TEX[band];
        defs += `<pattern id="${id}" patternUnits="userSpaceOnUse" width="${t.sp}" height="${t.sp}" patternTransform="rotate(${ang})">`
          + `<line x1="0" y1="0" x2="0" y2="${t.sp}" stroke="#fff" stroke-width="${t.w}" stroke-opacity="${t.o}"/></pattern>`;
      });
    });
    return `<defs>${defs}</defs>`;
  }
  const _methSafe = (m) => String(m || "x").replace(/[^A-Za-z0-9]/g, "");
  function _methHealth(m) {
    const t = m.total || 0, err = (m.n4 || 0) + (m.n5 || 0);
    const succ = t ? Math.round(100 * (m.n2 || 0) / t) : 0;
    const errPct = t ? Math.round(100 * err / t) : 0;
    let state = "stable";
    if (t >= 3) { if (m.n5 > 0 && errPct >= 30) state = "critical"; else if (errPct >= 20) state = "critical"; else if (errPct >= 5) state = "degraded"; }
    if (succ === 0 && t >= 3) state = "critical";
    return { succ, errPct, state, latBand: _latBand(m.lat_avg), lat: m.lat_avg };
  }
  function hiveColonySvg(matrix, opts) {
    opts = opts || {}; const EXP = !!opts.expanded;
    const col = _C();
    const rows = (matrix || []).filter(m => m.total > 0);
    if (!rows.length) return `<div class="tf-empty">No requests to map yet — colonies emerge with traffic.</div>`;
    const cv = { ok: col.ok, amber: col.amber, danger: col.danger };
    const allMax = Math.max(1, ...rows.flatMap(m => [m.n2, m.n4, m.n5]));
    const methodMax = Math.max(1, ...rows.map(m => m.total));
    const busiest = rows[0] && rows[0].method;
    const G = EXP
      ? { rowH: 116, coreX: 104, R: 104, ang: 24, crMax: 30, crMin: 15, maxR: 33, minR: 11, dormR: 8, lblX: 30, W: 360 }
      : { rowH: 80, coreX: 70, R: 66, ang: 19, crMax: 19, crMin: 10, maxR: 22, minR: 9, dormR: 6.5, lblX: 8, W: 250 };
    const oy = EXP ? 30 : 24;
    const H = oy + rows.length * G.rowH + 22;
    const angs = { "2": -G.ang * Math.PI / 180, "4": 0, "5": G.ang * Math.PI / 180 };
    let body = "";
    rows.forEach((m, ri) => {
      const cy = oy + ri * G.rowH + G.rowH / 2 - 6;
      const cr = Math.max(G.crMin, G.crMax * Math.sqrt((m.total || 0) / methodMax));
      const ang = _hatchAngle(m.method);
      const hl = _methHealth(m);
      let conns = "", cells = "";
      _STAT.forEach(st => {
        const v = m[st.k] || 0;
        const cellX = G.coreX + G.R * Math.cos(angs[st.sb]);
        const cellY = cy + G.R * Math.sin(angs[st.sb]);
        const dormant = v === 0;
        const r = dormant ? G.dormR : Math.max(G.minR, G.maxR * Math.sqrt(v / allMax));
        const cell = (m.cells && m.cells[st.sb]) || null;
        const band = _cellLatBand(cell);
        // connector core → cell (opacity ∝ share); dormant = faint dotted
        const share = m.total ? v / m.total : 0;
        const mx = (G.coreX + cellX) / 2, my = (cy + cellY) / 2 - 6;
        conns += `<path class="tf-hive-conn${dormant ? " tf-hive-conn-dorm" : ""}" d="M ${G.coreX.toFixed(1)} ${cy.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${cellX.toFixed(1)} ${cellY.toFixed(1)}" fill="none" stroke="${dormant ? "var(--ink-mute)" : cv[st.cv]}" stroke-width="${dormant ? 1 : (1 + 2.4 * share).toFixed(2)}" stroke-opacity="${dormant ? 0.18 : (0.25 + 0.5 * share).toFixed(2)}"${dormant ? ' stroke-dasharray="2 3"' : ""}/>`;
        if (dormant) {
          cells += `<g class="tf-hive-cellg tf-hive-dormant"><path d="${_hexPath(cellX, cellY, r)}" fill="rgba(120,110,90,.05)" stroke="var(--ink-mute)" stroke-width="0.7" stroke-opacity="0.3" stroke-dasharray="1.5 2.5"/><circle cx="${cellX.toFixed(1)}" cy="${cellY.toFixed(1)}" r="1" fill="var(--ink-mute)" fill-opacity="0.3"/></g>`;
        } else {
          const errShare = (st.sb === "4" || st.sb === "5") ? share : 0;
          const sw = (st.sb === "2" ? 0.8 : 1 + (st.sb === "5" ? 3.4 : 2.4) * errShare).toFixed(2);
          const pctM = m.total ? Math.round(100 * v / m.total) : 0;
          const lblBig = r > 15;
          cells += `<g class="tf-hive-cellg" style="--gc:${cv[st.cv]}">`
            + `<path class="tf-hive-cell-base" d="${_hexPath(cellX, cellY, r)}" fill="${cv[st.cv]}" fill-opacity="0.9" stroke="${st.sb === "2" ? "var(--ink)" : cv[st.cv]}" stroke-width="${sw}" stroke-opacity="${st.sb === "2" ? 0.45 : 0.95}"/>`
            + (band ? `<path d="${_hexPath(cellX, cellY, r)}" fill="url(#tfhx-${_methSafe(m.method)}-${band})" pointer-events="none"/>` : "")
            + `<text x="${cellX.toFixed(1)}" y="${(cellY + (lblBig ? -1 : 3)).toFixed(1)}" text-anchor="middle" pointer-events="none" style="font:600 ${lblBig ? 9.5 : 8}px 'JetBrains Mono',monospace;fill:#fff">${_kc(v)}</text>`
            + (lblBig ? `<text x="${cellX.toFixed(1)}" y="${(cellY + 9).toFixed(1)}" text-anchor="middle" pointer-events="none" style="font:7px 'JetBrains Mono',monospace;fill:#fff;fill-opacity:.85">${pctM}%</text>` : "")
            + `<path class="tf-hive-cell tf-hive-hit" data-m="${esc(m.method)}" data-s="${st.lbl}" data-v="${v}" data-pct="${pctM}" data-lat="${cell && cell.lat != null ? cell.lat : ""}" data-band="${band || ""}" d="${_hexPath(cellX, cellY, r + 3)}" fill="transparent"/></g>`;
        }
      });
      // core node (◉ total) + method label + state dot
      const stCol = hl.state === "critical" ? cv.danger : hl.state === "degraded" ? cv.amber : cv.ok;
      const core = `<g class="tf-hive-coreg${m.method === busiest ? " tf-hive-breathe" : ""}" style="--gc:${stCol}">`
        + `<circle class="tf-hive-core-ring" cx="${G.coreX}" cy="${cy.toFixed(1)}" r="${(cr + 3).toFixed(1)}" fill="none" stroke="${stCol}" stroke-width="1.4" stroke-opacity="0.5"/>`
        + `<circle class="tf-hive-core-fill" cx="${G.coreX}" cy="${cy.toFixed(1)}" r="${cr.toFixed(1)}" fill="var(--ink)" fill-opacity="0.9"/>`
        + `<text x="${G.coreX}" y="${(cy + 3.5).toFixed(1)}" text-anchor="middle" pointer-events="none" style="font:600 ${cr > 15 ? 11 : 9}px 'JetBrains Mono',monospace;fill:var(--paper)">${_kc(m.total)}</text>`
        + `<text x="${G.coreX}" y="${(cy - cr - 6).toFixed(1)}" text-anchor="middle" pointer-events="none" class="meth ${_methClass(m.method)}" style="font:600 ${EXP ? 12 : 10.5}px 'JetBrains Mono',monospace">${esc(m.method)}</text>`
        + `<circle class="tf-hive-hit" data-m="${esc(m.method)}" data-core="1" cx="${G.coreX}" cy="${cy.toFixed(1)}" r="${(cr + 4).toFixed(1)}" fill="transparent"/></g>`;
      body += `<g class="tf-hive-colony" data-colony="${esc(m.method)}">${conns}${cells}${core}</g>`;
    });
    const legend = `<text x="${G.lblX}" y="${H - 6}" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">size=volume · color=status · texture=latency · ring=health · ${EXP ? "click a colony to inspect" : "hover a colony · click → requests"}</text>`;
    return `<svg class="tf-hive-svg" viewBox="0 0 ${G.W} ${H}" preserveAspectRatio="xMidYMid meet" style="max-height:${H}px;width:100%;overflow:visible">${_hiveDefs(rows)}${body}${legend}</svg>`;
  }
  // recent-spike / trend read from a method's status-over-time series
  function _seriesSpike(series) {
    const act = (series || []).filter(s => s.n > 0); if (act.length < 2) return null;
    const avg = act.reduce((a, s) => a + s.n, 0) / act.length, last = series[series.length - 1] || { n: 0 };
    if (last.n > avg * 1.8 && last.n >= 3) return "spiking now";
    const h = Math.floor(series.length / 2);
    const a = series.slice(0, h).reduce((s, x) => s + x.n, 0), b = series.slice(h).reduce((s, x) => s + x.n, 0);
    if (a && b > a * 1.5) return "trending up"; if (b && a > b * 1.5) return "cooling down"; return "steady";
  }
  function _hiveCellTip(m, st) {
    const v = m[st === "2xx" ? "n2" : st === "4xx" ? "n4" : "n5"] || 0;
    const pct = m.total ? Math.round(100 * v / m.total) : 0;
    const cell = m.cells && m.cells[st[0]];
    const lat = cell && cell.lat != null ? cell.lat : null;
    const band = _cellLatBand(cell);
    const bn = band === "slow" ? "slow" : band === "med" ? "moderate" : band === "fast" ? "fast" : "—";
    return `<b>${esc(m.method)} · ${esc(st)}</b><br>${fmtNum(v)} req · ${pct}% of ${esc(m.method)}`
      + (lat != null ? `<br>latency ${fmtNum(lat)} ms · ${bn}` : "")
      + `<br><span style="opacity:.6">click → these requests</span>`;
  }
  function _hiveCoreTip(m) {
    const hl = _methHealth(m), spike = _seriesSpike(m.series);
    const top = (m.endpoints || [])[0];
    return `<b>${esc(m.method)} colony</b> · ${fmtNum(m.total)} req<br>`
      + `<span style="color:${hl.state === "critical" ? "var(--c-danger,#b3402f)" : hl.state === "degraded" ? "var(--c-amber,#c98a2b)" : "var(--c-ok,#2f6f3e)"}">${hl.succ}% success · ${hl.state}</span>`
      + (hl.lat != null ? `<br>~${fmtNum(hl.lat)} ms typical` : "")
      + (top ? `<br>top · ${esc(top.path)} (${top.pct}%)` : "")
      + (spike ? `<br><span style="opacity:.7">${spike}</span>` : "")
      + `<br><span style="opacity:.6">click → inspect colony</span>`;
  }
  function _wireHiveColony(host, opts) {
    opts = opts || {};
    const svg = host.querySelector(".tf-hive-svg"); if (!svg) return;
    const byMethod = {}; (DATA.matrix || []).forEach(m => byMethod[m.method] = m);
    const focus = (method) => { svg.classList.add("tf-hive-diffuse"); svg.querySelectorAll(".tf-hive-colony").forEach(g => g.classList.toggle("tf-hive-focus", g.getAttribute("data-colony") === method)); };
    const unfocus = () => { svg.classList.remove("tf-hive-diffuse"); svg.querySelectorAll(".tf-hive-focus").forEach(g => g.classList.remove("tf-hive-focus")); };
    host.querySelectorAll(".tf-hive-hit").forEach(h => {
      const method = h.getAttribute("data-m"), isCore = h.getAttribute("data-core"), st = h.getAttribute("data-s");
      h.style.cursor = "pointer";
      h.addEventListener("mousemove", (e) => {
        const m = byMethod[method]; if (!m) return;
        tip(true, e.clientX, e.clientY, isCore ? _hiveCoreTip(m) : _hiveCellTip(m, st));
        focus(method);
        const g = h.closest(".tf-hive-cellg, .tf-hive-coreg"); if (g) g.classList.add("tf-hive-pop");
      });
      h.addEventListener("mouseleave", () => { tip(false); unfocus(); const g = h.closest(".tf-hive-pop"); if (g) g.classList.remove("tf-hive-pop"); });
      h.addEventListener("click", () => {
        const m = byMethod[method]; if (!m) return;
        if (isCore) {
          if (opts.onColony) opts.onColony(method);
          else _openExpander("hive", host.closest(".ov-card"));
          return;
        }
        const v = +h.getAttribute("data-v"); if (!v || !DATA.hero || !DATA.hero.buckets.length) return;
        const first = DATA.hero.buckets[0], last = DATA.hero.buckets[DATA.hero.buckets.length - 1];
        openRequestContainer({ sinceMs: first.t_ms, untilMs: last.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: method + " · " + st, method, status: st });
      });
    });
  }
  // live pulse on a colony (no rebuild — keeps the layout/animations stable)
  function _hivePulse(method) {
    document.querySelectorAll(`.tf-hive-colony[data-colony="${(method || "").replace(/"/g, "")}"]`).forEach(g => {
      g.classList.remove("tf-hive-ping"); void g.offsetWidth; g.classList.add("tf-hive-ping");
      const ring = g.querySelector(".tf-hive-core-ring");
      if (ring) { try { ring.animate([{ strokeOpacity: 0.9, strokeWidth: 3 }, { strokeOpacity: 0.5, strokeWidth: 1.4 }], { duration: 620, easing: "ease-out" }); } catch (_) {} }
    });
  }
  function renderHive(intro) {
    const host = $("tf-hive-body"); if (!host || !DATA) return;
    host.innerHTML = hiveColonySvg(DATA.matrix || [], { expanded: false });
    _wireHiveColony(host, {});
    if (intro && window.APIN && APIN.fx) {
      host.querySelectorAll(".tf-hive-colony").forEach((g, i) => { try { g.animate([{ opacity: 0, transform: "scale(.72)" }, { opacity: 1, transform: "scale(1)" }], { duration: 460, delay: Math.min(640, i * 90), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
    }
  }

  // ════════════════════ ⑤ TRAFFIC CLOCK — polar local-hour dial ═══════════
  function _polar(cx, cy, r, frac) { const a = -Math.PI / 2 + frac * 2 * Math.PI; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; }
  function _wedgePath(cx, cy, r0, r1, a0, a1) {
    const p0 = [cx + r0 * Math.cos(a0), cy + r0 * Math.sin(a0)], p1 = [cx + r1 * Math.cos(a0), cy + r1 * Math.sin(a0)];
    const p2 = [cx + r1 * Math.cos(a1), cy + r1 * Math.sin(a1)], p3 = [cx + r0 * Math.cos(a1), cy + r0 * Math.sin(a1)];
    const large = (a1 - a0) > Math.PI ? 1 : 0;
    return `M ${p0[0].toFixed(1)} ${p0[1].toFixed(1)} L ${p1[0].toFixed(1)} ${p1[1].toFixed(1)} A ${r1} ${r1} 0 ${large} 1 ${p2[0].toFixed(1)} ${p2[1].toFixed(1)} L ${p3[0].toFixed(1)} ${p3[1].toFixed(1)} A ${r0} ${r0} 0 ${large} 0 ${p0[0].toFixed(1)} ${p0[1].toFixed(1)} Z`;
  }
  function clockSvg(clock, size, mode) {
    size = size || 300; const cx = size / 2, cy = size / 2, r0 = size * 0.16, rmax = size * 0.44;
    const col = _C(), hours = clock.hours || [], max = clock.max || 1;
    let wedges = "", rings = "", labels = "";
    [0.25, 0.5, 0.75, 1].forEach(rr => { rings += `<circle cx="${cx}" cy="${cy}" r="${(r0 + (rmax - r0) * rr).toFixed(1)}" fill="none" stroke="var(--paper-edge)" stroke-width="1" opacity="0.4"/>`; });
    hours.forEach((h, i) => {
      const a0 = -Math.PI / 2 + (i / 24) * 2 * Math.PI + 0.012, a1 = -Math.PI / 2 + ((i + 1) / 24) * 2 * Math.PI - 0.012;
      const r1 = r0 + (rmax - r0) * (h.n / max);
      let fill = col.accent, op = 0.85;
      if (h.n === 0) { fill = "rgba(120,110,90,.10)"; op = 1; }
      else if (mode === "error") { fill = h.err_pct === 0 ? col.ok : h.err_pct < 10 ? col.amber : col.danger; }
      const rr = h.n === 0 ? r0 + 2 : r1;
      wedges += `<path class="tf-wedge" data-h="${i}" d="${_wedgePath(cx, cy, r0, rr, a0, a1)}" fill="${fill}" fill-opacity="${op}"/>`;
    });
    // cardinal labels in 12-hour form (12 AM / 6 AM / 12 PM / 6 PM)
    [0, 6, 12, 18].forEach(hh => { const [lx, ly] = _polar(cx, cy, rmax + 14, hh / 24); labels += `<text x="${lx.toFixed(1)}" y="${(ly + 3).toFixed(1)}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-soft)">${((hh % 12) || 12) + (hh < 12 ? " AM" : " PM")}</text>`; });
    const peakShort = clock.busiest_h != null ? (((clock.busiest_h % 12) || 12) + " " + (clock.busiest_h < 12 ? "AM" : "PM")) : "";
    return `<svg class="tf-clock-svg" viewBox="0 0 ${size} ${size}" style="max-width:${size}px;margin:0 auto;overflow:visible">
      ${rings}${wedges}${labels}
      <line id="tf-clock-hand" x1="${cx}" y1="${cy}" x2="${cx}" y2="${(cy - rmax - 4).toFixed(1)}" stroke="${col.ink}" stroke-width="2" stroke-linecap="round" transform="rotate(0 ${cx} ${cy})"/>
      <rect id="tf-clock-now-bg" rx="9" width="0" height="0" fill="var(--ink)"/>
      <text id="tf-clock-now" text-anchor="middle" style="font:600 9px 'JetBrains Mono',monospace;fill:var(--paper)"></text>
      <circle cx="${cx}" cy="${cy}" r="${(r0 - 4).toFixed(1)}" fill="var(--paper)" stroke="var(--paper-edge)"/>
      <text x="${cx}" y="${(cy - 2).toFixed(1)}" text-anchor="middle" style="font:7.5px 'JetBrains Mono',monospace;fill:var(--ink-mute)">peak</text>
      <text x="${cx}" y="${(cy + 9).toFixed(1)}" text-anchor="middle" style="font:600 9px 'JetBrains Mono',monospace;fill:var(--ink-soft)">${peakShort}</text></svg>`;
  }
  function _startClockHand() {
    if (_clockTick) clearInterval(_clockTick);
    const move = () => {
      const hand = $("tf-clock-hand"); if (!hand) return;
      const now = new Date(); const frac = (now.getHours() + now.getMinutes() / 60 + now.getSeconds() / 3600) / 24;
      const cx = +hand.getAttribute("x1"), cy = +hand.getAttribute("y1"), y2 = +hand.getAttribute("y2");
      hand.style.transition = "transform 1s linear";
      hand.setAttribute("transform", `rotate(${(frac * 360).toFixed(2)} ${cx} ${cy})`);
      // live time chip riding the hand's tip — a pill sized to fit the text
      const lbl = $("tf-clock-now"), bg = $("tf-clock-now-bg");
      if (lbl) {
        const tipR = (cy - y2) + 15, ang = frac * 2 * Math.PI;
        const tx = cx + tipR * Math.sin(ang), ty = cy - tipR * Math.cos(ang);
        const txt = fmtClockNow(now);            // e.g. "11:32 PM"
        const w = txt.length * 5.4 + 12, hgt = 16;
        lbl.textContent = txt;
        lbl.style.transition = "all 1s linear";
        lbl.setAttribute("x", tx.toFixed(1)); lbl.setAttribute("y", (ty + 3.2).toFixed(1));
        if (bg) {
          bg.style.transition = "all 1s linear";
          bg.setAttribute("x", (tx - w / 2).toFixed(1)); bg.setAttribute("y", (ty - hgt / 2).toFixed(1));
          bg.setAttribute("width", w.toFixed(1)); bg.setAttribute("height", hgt);
        }
      }
    };
    move(); _clockTick = setInterval(move, 1000);
  }
  // local 'that hour' window today (or yesterday if still in the future) → ms
  function _hourWindow(hh) {
    const d = new Date(); d.setHours(hh, 0, 0, 0); if (d > new Date()) d.setDate(d.getDate() - 1);
    return { sinceMs: d.getTime(), untilMs: d.getTime() + 3600e3 };
  }
  // the window the clock aggregates over (matches the active granularity)
  function _clockWindow() {
    const win = GRAN_WIN_MS[GRAN] || GRAN_WIN_MS.day;
    const now = Date.now();
    return { sinceMs: now - win, untilMs: now };
  }
  // drill a clock hour: the clock is hour-of-day across the whole window, so the
  // container filters every request in that LOCAL hour (not just today's).
  function _drillHourOfDay(hh) {
    const w = _clockWindow();
    openRequestContainer({ sinceMs: w.sinceMs, untilMs: w.untilMs, label: fmt12(hh) + " · hour of day", localHour: hh });
  }
  // drill a weekday×hour cell: target the most-recent matching weekday that
  // actually has traffic (from the calendar), then filter to that local hour —
  // reliable even though the 28-day window exceeds the 200-row request fetch.
  function _drillWeekdayHour(wd, hh) {
    const WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const days = ((DATA.calendar || {}).days || []).filter(d => d.n > 0)
      .filter(d => { const dt = new Date(d.date + "T00:00:00"); return ((dt.getDay() + 6) % 7) === wd; })
      .sort((a, b) => a.date < b.date ? 1 : -1);
    let ms;
    if (days.length) ms = new Date(days[0].date + "T00:00:00").getTime();
    else { const t = new Date(); t.setHours(0, 0, 0, 0); ms = t.getTime() - ((((t.getDay() + 6) % 7) - wd + 7) % 7) * 86400e3; }
    openRequestContainer({ sinceMs: ms, untilMs: ms + 86400e3, label: WD[wd] + " " + new Date(ms).toLocaleDateString([], { month: "short", day: "numeric" }) + " · " + fmt12(hh), localHour: hh });
  }
  function _wireClock(host, intro, onWedge) {
    const svg = host.querySelector("svg"); if (!svg) return;
    svg.querySelectorAll(".tf-wedge").forEach(w => {
      w.addEventListener("mousemove", (e) => {
        const h = (DATA.clock.hours || [])[+w.getAttribute("data-h")]; if (!h) return;
        tip(true, e.clientX, e.clientY, `<b>${fmt12(h.h)} – ${fmt12((h.h + 1) % 24)}</b><br>${fmtNum(h.n)} req · ${h.err_pct}% err`);
        svg.querySelectorAll(".tf-wedge").forEach(o => o.style.opacity = o === w ? "1" : "0.4");
      });
      w.addEventListener("mouseleave", () => { tip(false); svg.querySelectorAll(".tf-wedge").forEach(o => o.style.opacity = "1"); });
      w.addEventListener("click", () => { const hh = +w.getAttribute("data-h"); if (onWedge) onWedge(hh); else _drillHourOfDay(hh); });
    });
    _startClockHand();
    if (intro) svg.querySelectorAll(".tf-wedge").forEach((w, i) => { try { w.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 360, delay: Math.min(820, i * 32), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
  }
  function renderClock(intro, mode) {
    const host = $("tf-clock-body"); if (!host || !DATA) return;
    const c = DATA.clock || { hours: [], max: 0 };
    if (!c.max) { host.innerHTML = `<div class="tf-empty">Rhythm emerges as traffic accrues.</div>`; return; }
    host.innerHTML = `<div class="tf-clock-wrap">${clockSvg(c, 270, mode || "volume")}</div>
      <div class="tf-clock-hint">device-local time (${esc((window.APIN && APIN.time) ? APIN.time.zone : "")}) · click an hour for its 60-min detail</div>`;
    _wireClock(host, intro, null);
  }

  // ════════════════════ ⑥ BYTES FLOW — mirror area ════════════════════════
  // ════════════════════ ⑥ BANDWIDTH — radial ring · river · channels ══════
  const ING = "#52b788";              // ingress = accent green (inner ring)
  const EGR = "#2c6e63";              // egress  = deep teal-ink (outer ring)
  const _rateStr = (bps) => bps <= 0 ? "idle" : bps >= 1048576 ? (bps / 1048576).toFixed(1) + " MB/s" : bps >= 1024 ? (bps / 1024).toFixed(0) + " KB/s" : Math.round(bps) + " B/s";
  function _liveRate() {
    const now = Date.now(); _rateWin = _rateWin.filter(e => now - e.t < 3000);
    if (!_rateWin.length) return 0;
    return _rateWin.reduce((s, e) => s + e.bytes, 0) / 3;   // bytes per second over 3s window
  }
  // ── radial bandwidth ring: clockwise time, inner=ingress, outer=egress ──
  function bandwidthRing(by, size) {
    size = size || 210; const cx = size / 2, cy = size / 2;
    const hubR = size * 0.205, inB = hubR + 5, inLen = size * 0.115, outB = inB + inLen + 7, outLen = size * 0.115;
    const buckets = by.buckets || [], n = buckets.length || 1;
    const maxIn = Math.max(1, ...buckets.map(b => b.bin || 0)), maxOut = Math.max(1, ...buckets.map(b => b.bout || 0));
    const step = 2 * Math.PI / n, gap = Math.min(0.045, step * 0.2);
    let guides = "", inSeg = "", outSeg = "", hits = "";
    [inB, outB + outLen].forEach(r => { guides += `<circle cx="${cx}" cy="${cy}" r="${r.toFixed(1)}" fill="none" stroke="rgba(26,22,18,0.06)" stroke-width="1"/>`; });
    buckets.forEach((b, i) => {
      const a0 = -Math.PI / 2 + i * step + gap, a1 = -Math.PI / 2 + (i + 1) * step - gap;
      const inN = (b.bin || 0) / maxIn, outN = (b.bout || 0) / maxOut;
      const inR1 = inB + Math.max(b.bin ? 0.1 : 0.02, inN) * inLen, outR1 = outB + Math.max(b.bout ? 0.1 : 0.02, outN) * outLen;
      inSeg += `<path class="tf-ring-seg${inN > 0.6 ? " tf-ring-hot" : ""}" d="${_wedgePath(cx, cy, inB, inR1, a0, a1)}" fill="${ING}" fill-opacity="${(0.4 + 0.6 * inN).toFixed(2)}" style="--gc:${ING}"/>`;
      outSeg += `<path class="tf-ring-seg${outN > 0.6 ? " tf-ring-hot" : ""}" d="${_wedgePath(cx, cy, outB, outR1, a0, a1)}" fill="${EGR}" fill-opacity="${(0.4 + 0.6 * outN).toFixed(2)}" style="--gc:${EGR}"/>`;
      hits += `<path class="tf-ring-hit" data-i="${i}" data-dir="in" d="${_wedgePath(cx, cy, inB - 2, inB + inLen + 2, a0, a1)}" fill="transparent"/>`;
      hits += `<path class="tf-ring-hit" data-i="${i}" data-dir="out" d="${_wedgePath(cx, cy, outB - 2, outB + outLen + 3, a0, a1)}" fill="transparent"/>`;
    });
    const orbit = (rmid, cls, count, dur, dir) => {
      let dots = "";
      for (let k = 0; k < count; k++) { const ang = (k / count) * 2 * Math.PI, px = cx + rmid * Math.cos(ang), py = cy + rmid * Math.sin(ang); dots += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="1.5" fill="#fff" fill-opacity="0.85"/>`; }
      return `<g class="${cls}">${dots}<animateTransform attributeName="transform" type="rotate" from="${dir > 0 ? 0 : 360} ${cx} ${cy}" to="${dir > 0 ? 360 : 0} ${cx} ${cy}" dur="${dur}s" repeatCount="indefinite"/></g>`;
    };
    const particles = orbit(inB + inLen * 0.5, "tf-orbit-in", 6, 9, 1) + orbit(outB + outLen * 0.5, "tf-orbit-out", 7, 13, -1);
    const sweep = `<g class="tf-ring-sweep"><path d="${_wedgePath(cx, cy, inB, outB + outLen, -Math.PI / 2, -Math.PI / 2 + 1.0)}" fill="url(#tfSweep)"/><animateTransform attributeName="transform" type="rotate" from="0 ${cx} ${cy}" to="360 ${cx} ${cy}" dur="7s" repeatCount="indefinite"/></g>`;
    const defs = `<defs><radialGradient id="tfSweep"><stop offset="0.4" stop-color="${ING}" stop-opacity="0"/><stop offset="1" stop-color="${ING}" stop-opacity="0.16"/></radialGradient></defs>`;
    const pulse = `<circle class="tf-ring-pulse" cx="${cx}" cy="${cy}" r="${inB}" fill="none" stroke="${ING}" stroke-width="2" opacity="0"/>`;
    return `<svg class="tf-ring-svg" viewBox="0 0 ${size} ${size}" data-in-base="${inB}" data-out-edge="${(outB + outLen).toFixed(1)}" data-cx="${cx}" style="max-width:${size}px;width:100%;overflow:visible">${defs}${guides}${sweep}${inSeg}${outSeg}${particles}${pulse}${hits}</svg>`;
  }
  function _ringPulse() {
    // Pulse EVERY live ring (card + open expand) — IDs would collide, so class.
    document.querySelectorAll(".tf-ring-pulse").forEach(p => {
      const svg = p.closest("svg"); const r0 = +(svg && svg.getAttribute("data-in-base") || 30), r1 = +(svg && svg.getAttribute("data-out-edge") || 100);
      try { p.animate([{ r: r0, opacity: 0.55, strokeWidth: 3 }, { r: r1 + 8, opacity: 0, strokeWidth: 1 }], { duration: 720, easing: "cubic-bezier(.22,1,.36,1)" }); } catch (_) {}
    });
  }
  // ── river / stream graph: mirrored, smoothed, gradient-filled ──────────
  function _smoothPath(pts) {
    if (pts.length < 2) return pts.length ? `M ${pts[0][0]} ${pts[0][1]}` : "";
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6, c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
    }
    return d;
  }
  function riverSvg(buckets) {
    const W = 520, H = 152, mid = H / 2, pad = 10, n = buckets.length || 1;
    const maxIn = Math.max(1, ...buckets.map(b => b.bin || 0)), maxOut = Math.max(1, ...buckets.map(b => b.bout || 0));
    const X = (i) => pad + (n <= 1 ? (W - 2 * pad) / 2 : (i / (n - 1)) * (W - 2 * pad));
    const upY = (v) => mid - (v / maxIn) * (mid - 14), dnY = (v) => mid + (v / maxOut) * (mid - 14);
    const inLine = _smoothPath(buckets.map((b, i) => [X(i), upY(b.bin || 0)]));
    const outLine = _smoothPath(buckets.map((b, i) => [X(i), dnY(b.bout || 0)]));
    const inArea = `${inLine} L ${X(n - 1).toFixed(1)} ${mid} L ${X(0).toFixed(1)} ${mid} Z`;
    const outArea = `${outLine} L ${X(n - 1).toFixed(1)} ${mid} L ${X(0).toFixed(1)} ${mid} Z`;
    const stride = Math.max(1, Math.ceil(n / 6)); let xlab = "";
    buckets.forEach((b, i) => { if (i % stride === 0) xlab += `<text x="${X(i).toFixed(1)}" y="${H - 2}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${esc(b.label)}</text>`; });
    return `<svg class="tf-river-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:${H}px;overflow:visible">
      <defs><linearGradient id="tfRivIn" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="${ING}" stop-opacity="0.04"/><stop offset="1" stop-color="${ING}" stop-opacity="0.42"/></linearGradient>
      <linearGradient id="tfRivOut" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${EGR}" stop-opacity="0.04"/><stop offset="1" stop-color="${EGR}" stop-opacity="0.42"/></linearGradient></defs>
      <path class="tf-river-band" d="${inArea}" fill="url(#tfRivIn)"/><path d="${inLine}" fill="none" stroke="${ING}" stroke-width="1.7"/>
      <path class="tf-river-band" d="${outArea}" fill="url(#tfRivOut)"/><path d="${outLine}" fill="none" stroke="${EGR}" stroke-width="1.7"/>
      <line x1="${pad}" y1="${mid}" x2="${W - pad}" y2="${mid}" stroke="var(--paper-edge)" stroke-width="1"/>
      <text x="2" y="11" style="font:8px 'JetBrains Mono',monospace;fill:${ING}">in ▲</text><text x="2" y="${H - 12}" style="font:8px 'JetBrains Mono',monospace;fill:${EGR}">out ▼</text>
      <line class="tf-river-cursor" id="tf-river-cursor" x1="0" y1="0" x2="0" y2="${H}" stroke="var(--ink)" stroke-opacity="0" stroke-width="1" stroke-dasharray="3 3"/>
      ${xlab}<rect id="tf-river-hit" x="0" y="0" width="${W}" height="${H}" fill="transparent"/></svg>`;
  }
  function _wireRiver(host, by) {
    const hit = host.querySelector("#tf-river-hit"), cur = host.querySelector("#tf-river-cursor"); if (!hit) return;
    const n = (by.buckets || []).length || 1;
    hit.addEventListener("mousemove", (e) => {
      const r = hit.getBoundingClientRect(); if (!r.width) return;
      const i = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1))));
      const b = by.buckets[i]; if (!b) return;
      const x = 10 + (n <= 1 ? 250 : (i / (n - 1)) * 500);
      if (cur) { cur.setAttribute("x1", x); cur.setAttribute("x2", x); cur.setAttribute("stroke-opacity", "0.5"); }
      tip(true, e.clientX, e.clientY, `<b>${esc(b.label)}${b.sub ? " · " + esc(b.sub) : ""}</b><br><span style="color:${ING}">▲ in ${fmtBytes(b.bin)}</span> · <span style="color:${EGR}">▼ out ${fmtBytes(b.bout)}</span>`);
    });
    hit.addEventListener("mouseleave", () => { tip(false); if (cur) cur.setAttribute("stroke-opacity", "0"); });
  }
  function _wireRing(host, by) {
    host.querySelectorAll(".tf-ring-hit").forEach(h => {
      const i = +h.getAttribute("data-i"), dir = h.getAttribute("data-dir");
      h.style.cursor = "pointer";
      h.addEventListener("mousemove", (e) => {
        const b = by.buckets[i]; if (!b) return;
        const headline = dir === "in" ? `<span style="color:${ING}">▲ ingress ${fmtBytes(b.bin)}</span>` : `<span style="color:${EGR}">▼ egress ${fmtBytes(b.bout)}</span>`;
        tip(true, e.clientX, e.clientY, `<b>${esc(b.label)}${b.sub ? " · " + esc(b.sub) : ""}</b><br>${headline}<br><span style="opacity:.6">in ${fmtBytes(b.bin)} · out ${fmtBytes(b.bout)}</span>`);
        host.querySelectorAll(".tf-ring-svg").forEach(s => s.classList.add("tf-ring-dim"));
        host.querySelectorAll(`.tf-ring-hit[data-i="${i}"]`).forEach(x => x.classList.add("tf-ring-hit-on"));
      });
      h.addEventListener("mouseleave", () => { tip(false); host.querySelectorAll(".tf-ring-svg").forEach(s => s.classList.remove("tf-ring-dim")); h.classList.remove("tf-ring-hit-on"); host.querySelectorAll(".tf-ring-hit-on").forEach(x => x.classList.remove("tf-ring-hit-on")); });
      h.addEventListener("click", () => { const b = by.buckets[i]; if (b) openRequestContainer({ sinceMs: b.t_ms, untilMs: b.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: b.label + (b.sub ? " " + b.sub : "") }); });
    });
  }
  function _bytesLive() {
    // Update ALL rings on screen (card + open expand) — class, not id.
    const rate = _liveRate();
    document.querySelectorAll(".tf-ring-hub-rate").forEach(re => {
      re.textContent = _rateStr(rate); re.classList.toggle("tf-rate-on", rate > 0);
      const wrap = re.closest(".tf-ring-wrap"), svg = wrap && wrap.querySelector(".tf-ring-svg");
      if (svg) svg.style.setProperty("--rglow", Math.min(9, rate / 180000 * 8).toFixed(1) + "px");
    });
    if (DATA && DATA.bytes) {
      const tot = fmtBytes((DATA.bytes.total_in || 0) + (DATA.bytes.total_out || 0));
      document.querySelectorAll(".tf-ring-hub-total").forEach(t => { t.textContent = tot; });
    }
  }
  function renderBytes(intro) {
    const host = $("tf-bytes-body"); if (!host || !DATA) return;
    const by = DATA.bytes || { buckets: [] };
    if (!by.buckets.length || (by.total_in === 0 && by.total_out === 0)) { host.innerHTML = `<div class="tf-empty">No payload data in this range.</div>`; return; }
    const toggle = `<div class="ov-range tf-by-toggle" id="tf-by-toggle"><button data-m="ring"${_byMode === "ring" ? ' aria-pressed="true"' : ""}>ring</button><button data-m="river"${_byMode === "river" ? ' aria-pressed="true"' : ""}>river</button></div>`;
    const legend = `<div class="tf-bytes-legend"><span style="color:${ING}">▲ in ${fmtBytes(by.total_in)} · ${fmtBytes(by.avg_in)}/req</span><span style="color:${EGR}">▼ out ${fmtBytes(by.total_out)} · ${fmtBytes(by.avg_out)}/req</span>${_ratioStr(by.total_in, by.total_out) ? `<span>${_ratioStr(by.total_in, by.total_out)}</span>` : ""}</div>`;
    if (_byMode === "ring") {
      host.innerHTML = toggle +
        `<div class="tf-ring-wrap">${bandwidthRing(by, 210)}<div class="tf-ring-hub"><div class="tf-ring-hub-total">${fmtBytes((by.total_in || 0) + (by.total_out || 0))}</div><div class="tf-ring-hub-rate">idle</div></div></div>` + legend;
      _wireRing(host, by); _bytesLive();
      if (intro && window.APIN && APIN.fx) host.querySelectorAll(".tf-ring-seg").forEach((s, k) => { try { s.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 340, delay: Math.min(520, k * 9), easing: "ease", fill: "backwards" }); } catch (_) {} });
    } else {
      host.innerHTML = toggle + `<div class="tf-river-wrap">${riverSvg(by.buckets)}</div>` + legend;
      _wireRiver(host, by);
      if (intro && window.APIN && APIN.fx) host.querySelectorAll(".tf-river-band").forEach(p => { try { p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 480, easing: "ease" }); } catch (_) {} });
    }
    host.querySelectorAll("#tf-by-toggle button").forEach(b => b.addEventListener("click", () => { if (b.getAttribute("data-m") === _byMode) return; _byMode = b.getAttribute("data-m"); renderBytes(false); }));
  }

  // ── endpoint channels: animated directional flow pipes per path ─────────
  // ingress marches rightward (▶), egress marches leftward (◀). Lane fill
  // length ∝ that direction's share of the busiest endpoint; marching speed
  // ∝ intensity so heavier pipes visibly flow faster.
  function endpointChannels(by) {
    const eps = (by.by_endpoint || []).slice()
      .sort((a, b) => (b.bin + b.bout) - (a.bin + a.bout)).slice(0, 8);
    if (!eps.length) return `<div class="tf-empty">no endpoint payload data</div>`;
    const maxIn = Math.max(1, ...eps.map(e => e.bin)), maxOut = Math.max(1, ...eps.map(e => e.bout));
    const spd = (v, mx) => (2.4 - 1.8 * Math.min(1, v / mx)).toFixed(2);   // s · faster when heavier
    const lane = (cls, v, mx, col) => {
      const w = (Math.max(v ? 5 : 0, (v / mx) * 100)).toFixed(1);
      return `<span class="tf-chan-track"><i class="tf-chan-flow ${cls}" style="width:${w}%;--spd:${spd(v, mx)}s;--fc:${col}"></i>`
        + `<b class="tf-chan-node ${v ? "tf-chan-node-on" : ""}" style="--fc:${col}"></b></span>`
        + `<span class="tf-chan-val" style="color:${col}">${fmtBytes(v)}</span>`;
    };
    return `<div class="tf-chan-wrap">` + eps.map(e =>
      `<div class="tf-chan" data-path="${esc(e.path)}" data-in="${e.bin}" data-out="${e.bout}" data-n="${e.n || 0}">
        <span class="tf-chan-path" title="${esc(e.path)}">${esc(e.path)}</span>
        <span class="tf-chan-dir" style="color:${ING}">IN ▶</span>${lane("tf-chan-in", e.bin, maxIn, ING)}
        <span class="tf-chan-dir" style="color:${EGR}">OUT ◀</span>${lane("tf-chan-out", e.bout, maxOut, EGR)}
      </div>`).join("") + `</div>`;
  }
  function _wireChannels(host, by) {
    host.querySelectorAll(".tf-chan").forEach(row => {
      const path = row.getAttribute("data-path"), bin = +row.getAttribute("data-in"), bout = +row.getAttribute("data-out"), n = +row.getAttribute("data-n");
      const inSh = by.total_in ? Math.round(100 * bin / by.total_in) : 0, outSh = by.total_out ? Math.round(100 * bout / by.total_out) : 0;
      row.addEventListener("mousemove", (e) => {
        tip(true, e.clientX, e.clientY, `<b>${esc(path)}</b><br><span style="color:${ING}">▶ in ${fmtBytes(bin)}</span> · ${inSh}% of ingress<br><span style="color:${EGR}">◀ out ${fmtBytes(bout)}</span> · ${outSh}% of egress<br><span style="opacity:.6">${fmtNum(n)} request(s) · click → these</span>`);
        host.querySelectorAll(".tf-chan").forEach(o => o.classList.toggle("tf-chan-dim", o !== row));
      });
      row.addEventListener("mouseleave", () => { tip(false); host.querySelectorAll(".tf-chan").forEach(o => o.classList.remove("tf-chan-dim")); });
      row.style.cursor = "pointer";
      row.addEventListener("click", () => {
        if (!DATA.hero || !DATA.hero.buckets.length) return;
        const first = DATA.hero.buckets[0], last = DATA.hero.buckets[DATA.hero.buckets.length - 1];
        openRequestContainer({ sinceMs: first.t_ms, untilMs: last.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: path, path });
      });
    });
  }

  // ── bytes insight engine: scored, 50+ phrasings across 10 families ──────
  function _bytesInsights(by) {
    const eps = (by.by_endpoint || []).slice();
    const buckets = (by.buckets || []).filter(b => (b.bin + b.bout) > 0);
    const tin = by.total_in || 0, tout = by.total_out || 0, tot = tin + tout;
    if (!tot) return [{ score: 1, text: "No payload has crossed this key in the selected window yet." }];
    // deterministic-but-personal variant picker (seeded by this key's totals)
    const seed = ((tin % 97) * 31 + (tout % 89) * 17 + eps.length * 7) >>> 0;
    const pick = (arr, salt) => arr[(seed + (salt || 0)) % arr.length];
    const out = [];
    const topIn = eps.slice().sort((a, b) => b.bin - a.bin)[0];
    const topOut = eps.slice().sort((a, b) => b.bout - a.bout)[0];
    // ① ingress concentration
    if (topIn && tin) {
      const sh = Math.round(100 * topIn.bin / tin);
      out.push({ score: 30 + sh, text: pick([
        `Ingress concentrated heavily on <b>${esc(topIn.path)}</b> — ${sh}% of everything uploaded.`,
        `<b>${esc(topIn.path)}</b> absorbs ${sh}% of inbound payload, the dominant intake route.`,
        `Most uploads (${sh}%) funnel through <b>${esc(topIn.path)}</b>.`,
        `${sh}% of bytes received arrived at <b>${esc(topIn.path)}</b>.`,
        `<b>${esc(topIn.path)}</b> is the heaviest ingress endpoint, drawing ${sh}% of input.`,
      ], sh) });
    }
    // ② egress concentration
    if (topOut && tout) {
      const sh = Math.round(100 * topOut.bout / tout);
      out.push({ score: 28 + sh, text: pick([
        `<b>${esc(topOut.path)}</b> dominates egress, returning ${sh}% of all bytes sent.`,
        `Outbound traffic centres on <b>${esc(topOut.path)}</b> — ${sh}% of responses by size.`,
        `${sh}% of egress flows out through <b>${esc(topOut.path)}</b>.`,
        `The bulk of downloads (${sh}%) leaves via <b>${esc(topOut.path)}</b>.`,
        `<b>${esc(topOut.path)}</b> is the primary response payload source (${sh}% of out).`,
      ], sh + 1) });
    }
    // ③ in/out balance
    const ratio = tin ? tout / tin : (tout ? Infinity : 0);
    if (ratio >= 4) out.push({ score: 60, text: pick([
      `Downloads dominate: egress outweighs ingress about ${ratio.toFixed(1)}× — a read-heavy workload.`,
      `Outbound dwarfs inbound (${ratio.toFixed(1)}×); clients pull far more than they push.`,
      `Egress runs ${ratio.toFixed(1)}× ingress — responses are much larger than requests.`,
      `This key is download-skewed — ${ratio.toFixed(1)}× more bytes leave than arrive.`,
    ], 2) });
    else if (ratio > 0 && ratio <= 0.25) out.push({ score: 60, text: pick([
      `Outbound traffic unusually low relative to input — uploads dominate (${(1 / ratio).toFixed(1)}× ingress).`,
      `Ingress outweighs egress ${(1 / ratio).toFixed(1)}× — a write/upload-heavy pattern.`,
      `Clients push far more than they receive (${(1 / ratio).toFixed(1)}× more in than out).`,
      `Responses are tiny against the payloads sent — ${(1 / ratio).toFixed(1)}× upload skew.`,
    ], 3) });
    else if (ratio >= 0.7 && ratio <= 1.4) out.push({ score: 34, text: pick([
      `Ingress and egress are well balanced — roughly a byte out for every byte in.`,
      `Symmetric flow: inbound and outbound payloads track each other closely.`,
      `In and out volumes mirror one another — an echo-shaped transfer profile.`,
    ], 4) });
    // ④ peak timing
    if (buckets.length) {
      const pk = buckets.slice().sort((a, b) => (b.bin + b.bout) - (a.bin + a.bout))[0];
      const lbl = esc(pk.label) + (pk.sub ? " " + esc(pk.sub) : "");
      out.push({ score: 26, text: pick([
        `Traffic peaked sharply around <b>${lbl}</b> with ${fmtBytes(pk.bin + pk.bout)} moved.`,
        `The heaviest transfer window was <b>${lbl}</b> (${fmtBytes(pk.bin + pk.bout)}).`,
        `Most data crossed near <b>${lbl}</b> — a clear volume spike.`,
        `Bandwidth crested at <b>${lbl}</b>, ${fmtBytes(pk.bin + pk.bout)} in that bucket alone.`,
      ], 5) });
    }
    // ⑤ largest single request
    if (by.largest) {
      const lg = by.largest, big = (lg.bin || 0) + (lg.bout || 0);
      out.push({ score: 22, text: pick([
        `One request stands out: <b>${esc(lg.method || "")} ${esc(lg.path || "")}</b> moved ${fmtBytes(big)} on its own.`,
        `Heaviest single call was <b>${esc(lg.method || "")} ${esc(lg.path || "")}</b> at ${fmtBytes(big)}.`,
        `A lone <b>${esc(lg.path || "")}</b> request carried ${fmtBytes(big)} — the biggest payload seen.`,
      ], 6) });
    }
    // ⑥ average payload size
    const avg = by.avg_in || 0, avgo = by.avg_out || 0;
    if (avg || avgo) out.push({ score: 16, text: pick([
      `Typical exchange: ${fmtBytes(avg)} in, ${fmtBytes(avgo)} out per request.`,
      `Per call this key averages ${fmtBytes(avg)} received and ${fmtBytes(avgo)} returned.`,
      `Mean payload sits at ${fmtBytes(avg)} inbound / ${fmtBytes(avgo)} outbound.`,
    ], 7) });
    // ⑦ payload spread (burstiness)
    if (by.pct && by.pct.in_p95 && by.pct.in_p50 && by.pct.in_p95 > by.pct.in_p50 * 4) {
      out.push({ score: 24, text: pick([
        `Upload sizes are bursty — p95 (${fmtBytes(by.pct.in_p95)}) towers over the median (${fmtBytes(by.pct.in_p50)}).`,
        `Most requests are small (${fmtBytes(by.pct.in_p50)} median) but the top 5% reach ${fmtBytes(by.pct.in_p95)}.`,
        `A long tail of large uploads: median ${fmtBytes(by.pct.in_p50)}, p95 ${fmtBytes(by.pct.in_p95)}.`,
      ], 8) });
    }
    // ⑧ quiet windows
    const zero = (by.buckets || []).length - buckets.length;
    if (zero > 0 && (by.buckets || []).length) {
      const pctQ = Math.round(100 * zero / by.buckets.length);
      if (pctQ >= 35) out.push({ score: 18, text: pick([
        `Transfer is intermittent — ${pctQ}% of buckets carried no payload at all.`,
        `Quiet for much of the window: ${pctQ}% of slots saw zero bytes.`,
        `Traffic arrives in pockets; ${pctQ}% of the timeline was idle.`,
      ], 9) });
    }
    // ⑨ trend (first vs second half)
    if (buckets.length >= 4) {
      const all = by.buckets, h = Math.floor(all.length / 2);
      const a = all.slice(0, h).reduce((s, b) => s + b.bin + b.bout, 0);
      const c = all.slice(h).reduce((s, b) => s + b.bin + b.bout, 0);
      if (a && c) {
        if (c > a * 1.6) out.push({ score: 20, text: pick([`Bandwidth is climbing — the latter half moved ${(c / a).toFixed(1)}× the data of the first.`, `Transfer is accelerating toward the present (${(c / a).toFixed(1)}× ramp).`, `Recent buckets are heavier — a rising data trend.`], 10) });
        else if (a > c * 1.6) out.push({ score: 20, text: pick([`Bandwidth is tapering — recent buckets moved ${(c / a * 100).toFixed(0)}% of the early volume.`, `Transfer is cooling off after a heavier opening stretch.`, `Data flow has been winding down through the window.`], 11) });
      }
    }
    // ⑩ headline volume (always available, low score)
    out.push({ score: 8, text: pick([
      `${fmtBytes(tot)} crossed this key — ${fmtBytes(tin)} in, ${fmtBytes(tout)} out.`,
      `Total throughput for the window: ${fmtBytes(tot)} (${fmtBytes(tin)} ▲ / ${fmtBytes(tout)} ▼).`,
      `This key moved ${fmtBytes(tot)} of payload across the period.`,
    ], 12) });
    return out.sort((a, b) => b.score - a.score);
  }

  // ════════════════════ LIVE (per-SSE-event, rAF-coalesced) ═══════════════
  let _es = null;
  function startSSE() {
    if (_es || !window.EventSource) return;
    try { _es = new EventSource("/api/account/usage/stream"); } catch (_) { _es = null; return; }
    _es.onmessage = (e) => {
      if (!LIVE || !_active || !DATA) return;
      let ev; try { ev = JSON.parse(e.data); } catch (_) { return; }
      if (!ev || ev.type !== "request" || (ev.key_id && PID && ev.key_id !== PID)) return;
      _applyEvent(ev);
    };
    _es.onerror = () => {};
  }
  function stopSSE() { if (_es) { try { _es.close(); } catch (_) {} _es = null; } }
  function _statusBucket(sc) { sc = +sc || 0; return sc >= 500 ? "n5" : sc >= 400 ? "n4" : "n2"; }
  function _applyEvent(ev) {
    const sb = _statusBucket(ev.status_code), isErr = sb !== "n2";
    if (DATA.hero && DATA.hero.buckets.length) {
      const last = DATA.hero.buckets[DATA.hero.buckets.length - 1];
      last[sb] = (last[sb] || 0) + 1; last.total = last.n2 + last.n4 + last.n5;
      if (last.total > DATA.hero.max) DATA.hero.max = last.total;
    }
    if (DATA.stats) {
      const errs = Math.round((DATA.stats.error_pct || 0) / 100 * (DATA.stats.total || 0)) + (isErr ? 1 : 0);
      DATA.stats.total = (DATA.stats.total || 0) + 1;
      DATA.stats.error_pct = DATA.stats.total ? Math.round(1000 * errs / DATA.stats.total) / 10 : 0;
      // keep busiest in sync with the (live) last bucket if it overtakes
      if (DATA.hero && DATA.hero.buckets.length) {
        const last = DATA.hero.buckets[DATA.hero.buckets.length - 1];
        if (last.total > (DATA.stats.busiest_count || 0)) { DATA.stats.busiest_count = last.total; DATA.stats.busiest_label = last.label; }
      }
    }
    // hive matrix: bump method × status cell + enriched colony fields
    if (DATA.matrix) {
      const m = (ev.method || "?").toUpperCase();
      let row = DATA.matrix.find(x => x.method === m);
      if (!row) { row = { method: m, n2: 0, n4: 0, n5: 0, total: 0, lat_avg: null, lat_bands: { fast: 0, med: 0, slow: 0 }, cells: {}, endpoints: [], series: [] }; DATA.matrix.push(row); }
      row[sb] += 1; row.total = row.n2 + row.n4 + row.n5;
      // per-cell count + latency band (sb is "n2"/"n4"/"n5" → cell key "2"/"4"/"5")
      const sk = sb.slice(1), lat = +ev.latency_ms;
      const cell = (row.cells = row.cells || {})[sk] || (row.cells[sk] = { n: 0, lat: null, lf: 0, lm: 0, ls: 0 });
      cell.n += 1;
      if (!isNaN(lat)) {
        cell.lat = cell.lat == null ? lat : Math.round(cell.lat + (lat - cell.lat) / cell.n);
        const bb = lat < 100 ? "lf" : lat < 500 ? "lm" : "ls"; cell[bb] = (cell[bb] || 0) + 1;
        const mb = lat < 100 ? "fast" : lat < 500 ? "med" : "slow";
        (row.lat_bands = row.lat_bands || { fast: 0, med: 0, slow: 0 })[mb] += 1;
      }
      // last series bucket (aligned to hero grid)
      if (Array.isArray(row.series) && row.series.length) {
        const lb = row.series[row.series.length - 1]; lb[sb] = (lb[sb] || 0) + 1; lb.n = (lb.n || 0) + 1;
      }
      // endpoint contributors
      if (ev.path) {
        const ep = (row.endpoints = row.endpoints || []).find(e => e.path === ev.path);
        if (ep) { ep.n += 1; if (isErr) ep.e = (ep.e || 0) + 1; } else row.endpoints.push({ path: ev.path, n: 1, e: isErr ? 1 : 0, pct: 0 });
        row.endpoints.sort((a, b) => b.n - a.n);
        row.endpoints.forEach(e => e.pct = row.total ? Math.round(100 * e.n / row.total) : 0);
      }
      DATA.matrix.sort((a, b) => b.total - a.total);
      _hivePulseMethod = m;   // pulse this colony on the next render frame
    }
    if (DATA.clock) {
      const lh = new Date().getHours(); const hr = (DATA.clock.hours || [])[lh];
      if (hr) { hr.n += 1; if (isErr) hr.e += 1; hr.err_pct = hr.n ? Math.round(1000 * hr.e / hr.n) / 10 : 0; if (hr.n > DATA.clock.max) DATA.clock.max = hr.n; }
    }
    if (DATA.calendar) {
      const t = new Date(); const key = `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`;
      let day = (DATA.calendar.days || []).find(d => d.date === key);
      if (!day) { day = { date: key, n: 0, e: 0 }; (DATA.calendar.days = DATA.calendar.days || []).push(day); }
      day.n += 1; if (isErr) day.e += 1; if (day.n > DATA.calendar.max) DATA.calendar.max = day.n;
      _calSweep = true;   // light-sweep today's cell on the next calendar render
    }
    if (DATA.bytes && DATA.bytes.buckets.length) {
      const lb = DATA.bytes.buckets[DATA.bytes.buckets.length - 1];
      lb.bin += (+ev.bytes_in || 0); lb.bout += (+ev.bytes_out || 0);
      DATA.bytes.total_in += (+ev.bytes_in || 0); DATA.bytes.total_out += (+ev.bytes_out || 0);
      const tr = DATA.stats ? DATA.stats.total : 0;
      DATA.bytes.avg_in = tr ? Math.round(DATA.bytes.total_in / tr) : 0;
      DATA.bytes.avg_out = tr ? Math.round(DATA.bytes.total_out / tr) : 0;
      DATA.bytes.ratio = DATA.bytes.total_in ? Math.round(10 * DATA.bytes.total_out / DATA.bytes.total_in) / 10 : null;
      _rateWin.push({ t: Date.now(), bytes: (+ev.bytes_in || 0) + (+ev.bytes_out || 0) });
      _byPulse++;   // emit a throughput ring-pulse on the next render frame
    }
    _dirty = { hero: 1, kpis: 1, clock: 1, cal: 1, bytes: 1, hive: 1 };
    if (!_rerenderRaf) _rerenderRaf = requestAnimationFrame(_flushRender);
    _scheduleReconcile();
  }
  function _flushRender() {
    _rerenderRaf = null;
    if (_dirty.hero) renderHero(false);
    if (_dirty.kpis) renderKpis();
    if (_dirty.clock) renderClock(false);
    if (_dirty.cal) renderCalendar(false);
    // Hive: pulse the active colony (glow propagation) without a full rebuild,
    // so connectors/breathing/textures don't restart. Sizes reconcile on the
    // 1.8 s debounced refresh().
    if (_dirty.hive) { if (_hivePulseMethod) { _hivePulse(_hivePulseMethod); _hivePulseMethod = null; } }
    // Bytes: tick the live counters + emit a throughput pulse, but DON'T rebuild
    // the SVG — a full rebuild would restart every SMIL orbit/sweep + marching
    // pipe from frame zero, producing the "stop-motion" stutter we're avoiding.
    // Segment lengths reconcile on the 1.8 s debounced refresh() instead.
    if (_dirty.bytes) { _bytesLive(); if (_byPulse) { _ringPulse(); _byPulse = 0; } }
    _dirty = {};
    if (_openExpand && _openExpand.update) {
      const open = !(window.APIN && APIN.lightbox && APIN.lightbox.isOpen) || APIN.lightbox.isOpen();
      if (open) { try { _openExpand.update(); } catch (_) {} }
      else _openExpand = null;
    }
  }
  function _scheduleReconcile() {
    if (_liveTimer) clearTimeout(_liveTimer);
    _liveTimer = setTimeout(() => { if (LIVE && _active) refresh(); }, 1800);
  }

  // ════════════════════ EXPANDED STATES ═══════════════════════════════════
  const _sec = (t) => `<div class="tfx-sec">${esc(t)}</div>`;
  const _hbars = (items, color) => {
    const max = Math.max(1, ...items.map(i => i.v));
    return `<div class="tfx-hbars">${items.map(i =>
      `<div class="tfx-hbar"${i.click ? ' data-click="' + esc(i.click) + '" style="cursor:pointer"' : ""}><span class="tfx-hbar-lbl" title="${esc(i.label)}">${esc(i.label)}</span><span class="tfx-hbar-track"><i style="width:${(i.v / max * 100).toFixed(1)}%;background:${color || "var(--c-accent,#52b788)"}"></i></span><span class="tfx-hbar-val">${esc(i.vl != null ? i.vl : fmtNum(i.v))}</span></div>`).join("")}</div>`;
  };

  function _compositionStrip(buckets) {
    const col = _C(), n = buckets.length || 1;
    let s = "";
    buckets.forEach((b, i) => {
      if (!b.total) { s += `<rect x="${i}" y="0" width="0.9" height="100" fill="rgba(120,110,90,.10)"/>`; return; }
      const g = 100 * b.n2 / b.total, a = 100 * b.n4 / b.total;
      s += `<rect x="${i}" y="${(100 - g).toFixed(1)}" width="0.9" height="${g.toFixed(1)}" fill="${col.ok}"/>`;
      s += `<rect x="${i}" y="${(100 - g - a).toFixed(1)}" width="0.9" height="${a.toFixed(1)}" fill="${col.amber}"/>`;
      s += `<rect x="${i}" y="0" width="0.9" height="${(100 - g - a).toFixed(1)}" fill="${col.danger}"/>`;
    });
    return `<svg viewBox="0 0 ${n} 100" preserveAspectRatio="none" style="width:100%;height:42px;border-radius:4px">${s}</svg>`;
  }
  function _readingInsight(buckets, s) {
    const act = buckets.filter(b => b.total > 0);
    const tot = act.reduce((a, b) => a + b.total, 0) || 1;
    if (!act.length) return "No traffic in this window yet.";
    const sorted = act.slice().sort((a, b) => b.total - a.total);
    const k = Math.min(3, sorted.length);
    const topShare = Math.round(100 * sorted.slice(0, k).reduce((a, b) => a + b.total, 0) / tot);
    const worst = act.slice().filter(b => b.total >= 5).sort((a, b) => (b.n4 + b.n5) / b.total - (a.n4 + a.n5) / a.total)[0];
    let txt = `${topShare}% of ${GRAN_LABEL[GRAN].replace("last ", "")} volume landed in the busiest ${k} ${GRAN === "hour" ? "hour(s)" : "bucket(s)"}.`;
    if (worst) { const ep = Math.round(100 * (worst.n4 + worst.n5) / worst.total); if (ep >= 20) txt += ` Errors peaked at ${ep}% on ${esc(worst.label)}${worst.sub ? " " + esc(worst.sub) : ""}.`; }
    return txt;
  }
  function expandHero(panel) {
    panel.innerHTML =
      `<div class="tfx-controls">
        <div class="ov-range" id="tfx-gran">${["hour", "day", "week", "month"].map(g => `<button data-g="${g}"${g === GRAN ? ' aria-pressed="true"' : ""}>${g[0].toUpperCase() + g.slice(1)}</button>`).join("")}</div>
        <div class="ov-range" id="tfx-ov"><button data-ov="none" aria-pressed="true">bars</button><button data-ov="err">+ error %</button></div>
      </div>
      <div id="tfx-hero"></div>
      <div class="tfx-legend"><span><i class="rdot s-2xx"></i> 2xx</span><span><i class="rdot s-4xx"></i> 4xx</span><span><i class="rdot s-5xx"></i> 5xx</span><span style="margin-left:auto;font-style:italic">drag the scrubber to zoom · click a colour band → that status</span></div>
      ${_sec("reading this window")}<p class="tfx-insight" id="tfx-hero-insight"></p>
      ${_sec("composition over time")}<div id="tfx-hero-comp"></div>
      ${_sec("top buckets · click → requests")}<div id="tfx-hero-tb"></div>
      ${_sec("throughput")}<div id="tfx-hero-stats" class="tfx-facts"></div>`;
    let ov = "none";
    const paint = () => {
      const host = $("tfx-hero"); heroChart(host, false);
      if (ov === "err") _overlayErr(host.querySelector(".tf-hero-main"));
      const s = DATA.stats || {}, buckets = (DATA.hero || { buckets: [] }).buckets;
      $("tfx-hero-insight").innerHTML = _readingInsight(buckets, s);
      $("tfx-hero-comp").innerHTML = _compositionStrip(buckets);
      // top buckets table
      const ranked = buckets.map((b, i) => ({ b, i })).filter(x => x.b.total > 0).sort((a, b) => b.b.total - a.b.total).slice(0, 8);
      $("tfx-hero-tb").innerHTML = ranked.length
        ? `<div class="tfx-tb"><div class="tfx-tb-row tfx-tb-head"><span>#</span><span>bucket</span><span>reqs</span><span>2xx</span><span>4xx</span><span>5xx</span><span>err%</span></div>`
          + ranked.map((x, k) => `<div class="tfx-tb-row" data-i="${x.i}"><span>${k + 1}</span><span class="tfx-tb-lbl">${esc(x.b.label)}${x.b.sub ? " " + esc(x.b.sub) : ""}</span><span>${fmtNum(x.b.total)}</span><span>${x.b.n2}</span><span>${x.b.n4}</span><span>${x.b.n5}</span><span>${x.b.total ? Math.round(100 * (x.b.n4 + x.b.n5) / x.b.total) : 0}%</span></div>`).join("")
          + `</div>`
        : `<div class="tf-empty">no data</div>`;
      $("tfx-hero-tb").querySelectorAll(".tfx-tb-row[data-i]").forEach(row => row.addEventListener("click", () => {
        const x = buckets[+row.getAttribute("data-i")]; if (x && x.total) openRequestContainer({ sinceMs: x.t_ms, untilMs: x.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: x.label + (x.sub ? " " + x.sub : "") });
      }));
      $("tfx-hero-stats").innerHTML =
        `<span>total <b>${fmtNum(s.total)}</b></span><span>busiest <b>${esc(s.busiest_label)}</b> (${fmtNum(s.busiest_count)})</span><span>peak <b>${fmtNum(s.peak_per_min)}</b>/min</span><span>errors <b>${s.error_pct}%</b></span>`;
    };
    paint();
    panel.querySelectorAll("#tfx-gran button").forEach(b => b.addEventListener("click", async () => {
      panel.querySelectorAll("#tfx-gran button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true");
      GRAN = b.getAttribute("data-g"); try { sessionStorage.setItem("tf_gran", GRAN); } catch (_) {}
      syncGranButtons(); await refresh(); paint();
    }));
    panel.querySelectorAll("#tfx-ov button").forEach(b => b.addEventListener("click", () => {
      panel.querySelectorAll("#tfx-ov button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true");
      ov = b.getAttribute("data-ov"); paint();
    }));
    _openExpand = { kind: "hero", update: paint };
  }
  function _overlayErr(main) {
    const svg = main && main.querySelector("svg"); if (!svg) return;
    const h = DATA.hero || { buckets: [] }; const z = _heroWindow(h.buckets.length);
    const slice = h.buckets.slice(z.i0, z.i1 + 1); const n = slice.length || 1;
    const baseY = HERO_H - BASE_OFF; const X = (i) => (i + 0.5) * (HERO_W / n);
    const pts = slice.map((b, i) => { const er = b.total ? (b.n4 + b.n5) / b.total : 0; return `${X(i).toFixed(1)},${(baseY - er * (baseY - 8)).toFixed(1)}`; }).join(" ");
    svg.insertAdjacentHTML("beforeend", `<polyline points="${pts}" fill="none" stroke="${_C().danger}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.85"/>`);
  }

  // KPI: Total → status split donut + requests-over-time + facts
  // fetch a window's requests once (small windows) for client-side breakdowns
  function _fetchWindow(sinceMs, untilMs) {
    return api(`/api/account/keys/${encodeURIComponent(PID)}/requests?since=${encodeURIComponent(_utcStr(sinceMs))}&until=${encodeURIComponent(_utcStr(untilMs))}&limit=200`)
      .then(({ body }) => (body && body.data && body.data.items) || []);
  }
  // minute histogram SVG (colour by error rate) — used by Peak deep-dive
  function _minHist(mins, merr) {
    const col = _C(), max = Math.max(1, ...mins);
    return `<svg viewBox="0 0 600 96" preserveAspectRatio="none" style="width:100%;height:96px">
      ${mins.map((v, i) => { const x = (i / mins.length) * 600, w = 600 / mins.length - 1.5, hh = v ? Math.max(2, v / max * 80) : 0; const er = v ? merr[i] / v : 0; const f = v === 0 ? "rgba(120,110,90,.12)" : er > 0.1 ? col.danger : er > 0 ? col.amber : col.accent; return `<rect x="${x.toFixed(1)}" y="${(86 - hh).toFixed(1)}" width="${w.toFixed(1)}" height="${hh ? hh.toFixed(1) : 1}" rx="1" fill="${f}"><title>:${String(i).padStart(2, "0")} · ${v} req</title></rect>`; }).join("")}
      <line x1="0" y1="86" x2="600" y2="86" stroke="var(--paper-edge)"/></svg>`;
  }
  // KPI: Total → status donut + method mix + over-time + insight
  function expandKpiTotal(panel) {
    const paint = () => {
      const col = _C(), s = DATA.stats || {}, h = DATA.hero || { buckets: [] };
      const n2 = h.buckets.reduce((a, b) => a + b.n2, 0), n4 = h.buckets.reduce((a, b) => a + b.n4, 0), n5 = h.buckets.reduce((a, b) => a + b.n5, 0);
      const meth = (DATA.matrix || []).slice().sort((a, b) => b.total - a.total);
      const grand = meth.reduce((a, m) => a + m.total, 0) || 1;
      const getShare = Math.round(100 * ((meth.find(m => m.method === "GET") || {}).total || 0) / grand);
      const writeShare = Math.round(100 * meth.filter(m => ["POST", "PUT", "PATCH", "DELETE"].includes(m.method)).reduce((a, m) => a + m.total, 0) / grand);
      const insight = getShare >= 60 ? `Read-heavy key — ${getShare}% of calls are GET; writes are ${writeShare}%.`
        : writeShare >= 60 ? `Write-heavy key — ${writeShare}% are writes; ${getShare}% GET.`
        : `Mixed workload — ${getShare}% GET, ${writeShare}% writes.`;
      panel.innerHTML =
        `<div class="tfx-bignum"><span class="apin-odometer" id="tfx-tot-num">${fmtNum(s.total)}</span><span class="tfx-bignum-lbl">requests · ${GRAN_LABEL[GRAN]}</span></div>
         ${_sec("status mix")}<div class="tfx-split">${_donut([{ l: "2xx", v: n2, c: col.ok }, { l: "4xx", v: n4, c: col.amber }, { l: "5xx", v: n5, c: col.danger }])}<div class="tfx-facts" style="flex-direction:column;gap:6px"><span><i class="rdot s-2xx"></i> 2xx <b>${fmtNum(n2)}</b> · ${Math.round(100 * n2 / (s.total || 1))}%</span><span><i class="rdot s-4xx"></i> 4xx <b>${fmtNum(n4)}</b> · ${Math.round(100 * n4 / (s.total || 1))}%</span><span><i class="rdot s-5xx"></i> 5xx <b>${fmtNum(n5)}</b> · ${Math.round(100 * n5 / (s.total || 1))}%</span><span>error rate <b>${s.error_pct}%</b></span></div></div>
         ${_sec("method mix")}${meth.length ? _hbars(meth.map(m => ({ label: m.method, v: m.total, vl: fmtNum(m.total) + " · " + Math.round(100 * m.total / grand) + "%" })), col.accent) : `<div class="tf-empty">no data</div>`}
         ${_sec("over time")}<div id="tfx-tot-hero"></div>
         ${_sec("insight")}<p class="tfx-insight">${insight}</p>`;
      const o = $("tfx-tot-num"); if (window.APIN && APIN.odometer) APIN.odometer.roll(o, fmtNum(s.total));
      heroChart($("tfx-tot-hero"), false);
    };
    paint(); _openExpand = { kind: "kpiTotal", update: paint };
  }
  // KPI: Peak/min → peak-minute callout + per-minute histogram + verdict + busiest hours
  function expandKpiPeak(panel) {
    const s = DATA.stats || {}, c = DATA.clock || { hours: [] };
    const ranked = (c.hours || []).filter(h => h.n > 0).sort((a, b) => b.n - a.n).slice(0, 6)
      .map(h => ({ label: fmt12(h.h), v: h.n, click: "h" + h.h}));
    panel.innerHTML =
      `<div class="tfx-bignum"><span>${fmtNum(s.peak_per_min)}</span><span class="tfx-bignum-lbl">peak requests in a single minute</span></div>
       <div class="tfx-facts"><span>busiest hour <b>${c.busiest_h != null ? fmt12(c.busiest_h) : "—"}</b></span><span>busiest bucket <b>${esc(s.busiest_label)}</b> (${fmtNum(s.busiest_count)})</span></div>
       ${_sec("per-minute · busiest hour")}<div id="tfx-peak-min" class="tf-empty">loading…</div>
       ${_sec("busiest hours · click → requests")}${ranked.length ? _hbars(ranked, _C().accent) : `<div class="tf-empty">no data</div>`}
       ${_sec("verdict")}<p class="tfx-insight" id="tfx-peak-verdict">…</p>`;
    panel.querySelectorAll(".tfx-hbar[data-click]").forEach(b => b.addEventListener("click", () => { const hh = +b.getAttribute("data-click").slice(1); _drillHourOfDay(hh); }));
    if (c.busiest_h != null) {
      const w = _hourWindow(c.busiest_h);
      _fetchWindow(w.sinceMs, w.untilMs).then(rows => {
        const mins = new Array(60).fill(0), merr = new Array(60).fill(0);
        rows.forEach(r => { const d = new Date((r.timestamp || "").replace(" ", "T") + "Z"); if (isNaN(d)) return; const m = d.getMinutes(); mins[m]++; if ((+r.status_code || 0) >= 400) merr[m]++; });
        const host = $("tfx-peak-min"); if (host) { host.className = ""; host.innerHTML = _minHist(mins, merr); }
        const active = mins.filter(v => v > 0).sort((a, b) => a - b);
        const median = active.length ? active[Math.floor(active.length / 2)] : 0;
        const burst = median ? (s.peak_per_min / median).toFixed(1) : "—";
        const v = $("tfx-peak-verdict");
        if (v) v.innerHTML = median >= 1 && s.peak_per_min / median > 4
          ? `<b>Spiky</b> — peak ${fmtNum(s.peak_per_min)}/min is ${burst}× the median active minute (${median}/min). Bursty traffic.`
          : `<b>Steady</b> — peak ${fmtNum(s.peak_per_min)}/min vs ${median}/min median active minute (${burst}×).`;
      });
    } else { const v = $("tfx-peak-verdict"); if (v) v.textContent = "Not enough data yet."; }
    _openExpand = { kind: "kpiPeak", update: null };
  }
  // KPI: Busiest → anatomy + top endpoints/methods (client-side) + top buckets + insight
  function expandKpiBusy(panel) {
    const col = _C(), h = DATA.hero || { buckets: [] }, s = DATA.stats || {};
    const ranked = h.buckets.map((b, i) => ({ b, i })).filter(x => x.b.total > 0).sort((a, b) => b.b.total - a.b.total);
    const top = ranked.slice(0, 10).map(x => ({ label: x.b.label + (x.b.sub ? " " + x.b.sub : ""), v: x.b.total, click: String(x.i) }));
    const totals = ranked.map(x => x.b.total);
    const median = totals.length ? totals.slice().sort((a, b) => a - b)[Math.floor(totals.length / 2)] : 0;
    const mult = median ? ((s.busiest_count || 0) / median).toFixed(1) : "—";
    const bb = ranked.length ? ranked[0].b : null;
    panel.innerHTML =
      `<div class="tfx-bignum"><span>${esc(s.busiest_label || "—")}</span><span class="tfx-bignum-lbl">busiest ${GRAN === "hour" ? "hour" : "bucket"} · ${fmtNum(s.busiest_count || 0)} requests</span></div>
       ${bb ? `${_sec("anatomy")}<div class="tfx-facts"><span><i class="rdot s-2xx"></i> 2xx <b>${fmtNum(bb.n2)}</b></span><span><i class="rdot s-4xx"></i> 4xx <b>${fmtNum(bb.n4)}</b></span><span><i class="rdot s-5xx"></i> 5xx <b>${fmtNum(bb.n5)}</b></span><span>errors <b>${bb.total ? Math.round(100 * (bb.n4 + bb.n5) / bb.total) : 0}%</b></span></div>` : ""}
       ${_sec("top endpoints · busiest bucket")}<div id="tfx-busy-ep" class="tf-empty">loading…</div>
       ${_sec("top methods · busiest bucket")}<div id="tfx-busy-meth" class="tf-empty">loading…</div>
       ${_sec("top buckets · click → requests")}${top.length ? _hbars(top, col.accent) : `<div class="tf-empty">no data</div>`}
       ${_sec("insight")}<p class="tfx-insight">Busiest ${GRAN === "hour" ? "hour" : "bucket"} ran <b>${fmtNum(s.busiest_count || 0)}</b> requests — ${mult}× the median active ${GRAN === "hour" ? "hour" : "bucket"} (${fmtNum(median)}).</p>`;
    panel.querySelectorAll(".tfx-hbar[data-click]").forEach(b => b.addEventListener("click", () => { const i = +b.getAttribute("data-click"); const x = h.buckets[i]; if (x) openRequestContainer({ sinceMs: x.t_ms, untilMs: x.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: x.label + (x.sub ? " " + x.sub : "") }); }));
    if (bb) {
      _fetchWindow(bb.t_ms, bb.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN])).then(rows => {
        const eps = {}, mts = {};
        rows.forEach(r => { eps[r.path || "?"] = (eps[r.path || "?"] || 0) + 1; mts[(r.method || "?").toUpperCase()] = (mts[(r.method || "?").toUpperCase()] || 0) + 1; });
        const epItems = Object.entries(eps).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => ({ label: k, v }));
        const mtItems = Object.entries(mts).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: k, v }));
        const e1 = $("tfx-busy-ep"); if (e1) { e1.className = ""; e1.innerHTML = epItems.length ? _hbars(epItems, col.accent) : `<div class="tf-empty">no data</div>`; }
        const e2 = $("tfx-busy-meth"); if (e2) { e2.className = ""; e2.innerHTML = mtItems.length ? _hbars(mtItems, col.soft) : `<div class="tf-empty">no data</div>`; }
      });
    }
    _openExpand = { kind: "kpiBusy", update: null };
  }
  // KPI: Data → full bytes deep-dive (shared)
  function expandKpiData(panel) { expandBytes(panel); }

  // tiny donut primitive
  function _donut(items) {
    const total = items.reduce((a, i) => a + i.v, 0) || 1, R = 52, r = 32, cx = 64, cy = 64, C = 2 * Math.PI * 42;
    let off = 0, arcs = "";
    items.forEach(it => { const frac = it.v / total; const len = frac * C; arcs += `<circle cx="${cx}" cy="${cy}" r="42" fill="none" stroke="${it.c}" stroke-width="20" stroke-dasharray="${len.toFixed(2)} ${(C - len).toFixed(2)}" stroke-dashoffset="${(-off).toFixed(2)}" transform="rotate(-90 ${cx} ${cy})"/>`; off += len; });
    return `<svg viewBox="0 0 128 128" style="width:128px;height:128px">${arcs}<text x="${cx}" y="${cy - 2}" text-anchor="middle" style="font:600 18px 'Fraunces',serif;fill:var(--ink)">${fmtNum(total)}</text><text x="${cx}" y="${cy + 14}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">requests</text></svg>`;
  }

  function expandCalendar(panel) {
    panel.innerHTML =
      `<div class="ov-range" id="tfx-calmode" style="margin-bottom:10px"><button data-m="volume" aria-pressed="true">volume</button><button data-m="error">error rate</button></div>
       <div id="tfx-cal" style="overflow-x:auto"></div>
       <div class="tfx-facts" id="tfx-calstats"></div>
       ${_sec("active days · click → requests")}<div id="tfx-cal-table"></div>
       ${_sec("by weekday (click → requests)")}<div id="tfx-wkprofile"></div>
       ${_sec("by month")}<div id="tfx-moprofile"></div>`;
    let mode = "volume";
    const WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const paint = () => {
      $("tfx-cal").innerHTML = calSvg((DATA.calendar || {}).days, 52, mode, 14);
      _wireCal($("tfx-cal"));
      const days = ((DATA.calendar || {}).days || []).slice().sort((a, b) => a.date < b.date ? -1 : 1);
      const active = days.filter(d => d.n > 0);
      const busiest = active.reduce((m, d) => d.n > (m ? m.n : -1) ? d : m, null);
      let streak = 0; const today = new Date(); today.setHours(0, 0, 0, 0);
      for (let i = 0; ; i++) { const dt = new Date(today); dt.setDate(dt.getDate() - i); const k = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`; const r = days.find(d => d.date === k); if (r && r.n > 0) streak++; else break; }
      const tot = active.reduce((a, d) => a + d.n, 0);
      $("tfx-calstats").innerHTML = `<span>${active.length} active day(s)</span><span>current streak <b>${streak}d</b></span><span>busiest <b>${busiest ? esc(busiest.date) : "—"}</b> (${busiest ? fmtNum(busiest.n) : 0})</span><span>total <b>${fmtNum(tot)}</b></span>`;
      // active-days table (most recent first), row → that day's requests
      const at = active.slice().sort((a, b) => a.date < b.date ? 1 : -1).slice(0, 14);
      $("tfx-cal-table").innerHTML = at.length
        ? `<div class="tfx-tb tfx-tb-3"><div class="tfx-tb-row tfx-tb-head"><span>date</span><span>reqs</span><span>err%</span></div>`
          + at.map(d => `<div class="tfx-tb-row tfx-cal-trow" data-d="${esc(d.date)}"><span class="tfx-tb-lbl">${esc(d.date)}</span><span>${fmtNum(d.n)}</span><span>${d.n ? Math.round(100 * d.e / d.n) : 0}%</span></div>`).join("")
          + `</div>`
        : `<div class="tf-empty">no active days</div>`;
      $("tfx-cal-table").querySelectorAll(".tfx-cal-trow").forEach(row => row.addEventListener("click", () => {
        const dstr = row.getAttribute("data-d"); const ms = new Date(dstr + "T00:00:00").getTime();
        openRequestContainer({ sinceMs: ms, untilMs: ms + 86400e3, label: dstr });
      }));
      // weekday profile (clickable → that weekday's most-recent day window)
      const wk = [0, 0, 0, 0, 0, 0, 0], wkLast = [null, null, null, null, null, null, null];
      days.forEach(d => { const dt = new Date(d.date + "T00:00:00"); const wi = (dt.getDay() + 6) % 7; wk[wi] += d.n; if (!wkLast[wi] || d.date > wkLast[wi]) wkLast[wi] = d.date; });
      $("tfx-wkprofile").innerHTML = _hbars(WD.map((nm, i) => ({ label: nm, v: wk[i], click: "wk" + i })), _C().accent);
      $("tfx-wkprofile").querySelectorAll(".tfx-hbar[data-click]").forEach(b => b.addEventListener("click", () => {
        const wi = +b.getAttribute("data-click").slice(2); const dstr = wkLast[wi]; if (!dstr) return;
        const ms = new Date(dstr + "T00:00:00").getTime(); openRequestContainer({ sinceMs: ms, untilMs: ms + 86400e3, label: WD[wi] + " " + dstr });
      }));
      // month profile
      const mo = {}; days.forEach(d => { const key = d.date.slice(0, 7); mo[key] = (mo[key] || 0) + d.n; });
      const moItems = Object.keys(mo).sort().map(k => ({ label: k, v: mo[k] }));
      $("tfx-moprofile").innerHTML = moItems.length ? _hbars(moItems, _C().soft) : `<div class="tf-empty">no data</div>`;
    };
    paint();
    panel.querySelectorAll("#tfx-calmode button").forEach(b => b.addEventListener("click", () => { panel.querySelectorAll("#tfx-calmode button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); mode = b.getAttribute("data-m"); paint(); }));
    _openExpand = { kind: "calendar", update: paint };
  }

  // ── colony health strip: last ≤12 buckets, colored by mode, recent pulses ─
  function _colonyHealthStrip(series, mode) {
    const buckets = (series || []).slice(-12);
    if (!buckets.length || !buckets.some(b => b.n > 0)) return `<span class="tfx-hs-empty">no recent activity</span>`;
    const col = _C(), maxN = Math.max(1, ...buckets.map(b => b.n));
    const lastActive = buckets.map(b => b.n > 0).lastIndexOf(true);
    return `<span class="tfx-hs">` + buckets.map((b, i) => {
      if (!b.n) return `<span class="tfx-hs-cell tfx-hs-dorm" title="quiet"></span>`;
      let fill, op = 1;
      if (mode === "volume") { fill = col.accent; op = 0.25 + 0.75 * (b.n / maxN); }
      else {
        const ep = b.n ? (b.n4 + b.n5) / b.n : 0;
        if (mode === "success") { const sp = 1 - ep; fill = sp >= 0.95 ? col.ok : sp >= 0.8 ? col.amber : col.danger; }
        else { fill = ep < 0.05 ? col.ok : ep < 0.2 ? col.amber : col.danger; }
      }
      const ping = i === lastActive ? " tfx-hs-ping" : "";
      return `<span class="tfx-hs-cell${ping}" style="background:${fill};opacity:${op.toFixed(2)};--gc:${fill}" title="${b.n} req · ${Math.round(100 * (b.n4 + b.n5) / b.n)}% err"></span>`;
    }).join("") + `</span>`;
  }
  // ── method intelligence: smart observability bullets (no table) ──────────
  function _methodIntel(m) {
    const hl = _methHealth(m), bands = m.lat_bands || {}, lt = (bands.fast || 0) + (bands.med || 0) + (bands.slow || 0);
    const slowSh = lt ? (bands.slow || 0) / lt : 0, fastSh = lt ? (bands.fast || 0) / lt : 0;
    const top = (m.endpoints || [])[0], bullets = [];
    if (hl.succ === 0 && m.total >= 3) {
      bullets.push({ t: "0 successful requests", k: "danger" });
      bullets.push({ t: m.n4 > m.n5 ? "consistent 4xx — likely auth / bad path" : "server-side 5xx failures", k: "danger" });
    } else {
      bullets.push({ t: `${hl.succ}% success`, k: hl.succ >= 90 ? "ok" : hl.succ >= 70 ? "amber" : "danger" });
      if (m.n5 > 0) bullets.push({ t: `${fmtNum(m.n5)} server error${m.n5 > 1 ? "s" : ""} (5xx)`, k: "danger" });
      else if (m.n4 > 0 && hl.errPct >= 10) bullets.push({ t: `${hl.errPct}% client errors (4xx)`, k: "amber" });
    }
    if (hl.lat != null) {
      if (slowSh >= 0.4) bullets.push({ t: `elevated latency · ~${fmtNum(hl.lat)} ms`, k: "amber" });
      else if (fastSh >= 0.7) bullets.push({ t: `fast & stable · ~${fmtNum(hl.lat)} ms`, k: "ok" });
      else bullets.push({ t: `~${fmtNum(hl.lat)} ms typical`, k: "neutral" });
    }
    if (top) bullets.push({ t: `top route · ${top.path} (${top.pct}%)`, k: "neutral" });
    return { hl, bullets };
  }
  // ── endpoint contributors as flow-lanes (not rows) ───────────────────────
  function _endpointLanes(m) {
    const eps = (m.endpoints || []).slice(0, 6);
    if (!eps.length) return `<div class="tf-empty">no endpoint data</div>`;
    const col = _C(), max = Math.max(1, ...eps.map(e => e.n));
    return `<div class="tfx-lanes">` + eps.map(e => {
      const w = (e.n / max * 100).toFixed(1), errPct = e.n ? Math.round(100 * (e.e || 0) / e.n) : 0;
      const fc = errPct >= 20 ? col.danger : errPct > 0 ? col.amber : col.accent;
      return `<div class="tfx-lane" data-path="${esc(e.path)}" data-m="${esc(m.method)}">
        <span class="tfx-lane-path" title="${esc(e.path)}">${esc(e.path)}</span>
        <span class="tfx-lane-track"><i class="tfx-lane-flow" style="width:${w}%;--fc:${fc}"></i></span>
        <span class="tfx-lane-pct">${e.pct}%</span><span class="tfx-lane-n">${fmtNum(e.n)}</span></div>`;
    }).join("") + `</div>`;
  }
  // ── temporal stream: stacked ok/amber/danger + scrub cursor + hit rects ──
  function _temporalStream(series) {
    const s = series || []; const n = s.length; if (n < 2) return `<div class="tf-empty">not enough history</div>`;
    const col = _C(), W = 520, H = 92, pad = 8, maxN = Math.max(1, ...s.map(b => b.n));
    const X = (i) => pad + (i / (n - 1)) * (W - 2 * pad), Y = (v) => H - pad - (v / maxN) * (H - 2 * pad);
    const area = (key, fill, op) => { let d = `M ${X(0).toFixed(1)} ${(H - pad).toFixed(1)}`; s.forEach((b, i) => { d += ` L ${X(i).toFixed(1)} ${Y(key(b)).toFixed(1)}`; }); d += ` L ${X(n - 1).toFixed(1)} ${(H - pad).toFixed(1)} Z`; return `<path class="tfx-stream-band" d="${d}" fill="${fill}" fill-opacity="${op}"/>`; };
    let spikes = "", hits = "";
    const bw = (W - 2 * pad) / n;
    s.forEach((b, i) => {
      if (b.n5 > 0) spikes += `<circle class="tfx-stream-spike" cx="${X(i).toFixed(1)}" cy="${Y(b.n).toFixed(1)}" r="2.6" fill="${col.danger}"/>`;
      hits += `<rect class="tfx-stream-hit" data-i="${i}" x="${(X(i) - bw / 2).toFixed(1)}" y="0" width="${bw.toFixed(1)}" height="${H}" fill="transparent"/>`;
    });
    return `<svg class="tfx-stream-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:${H}px;overflow:visible">
      ${area(b => b.n, col.ok, 0.5)}${area(b => b.n4 + b.n5, col.amber, 0.62)}${area(b => b.n5, col.danger, 0.72)}
      <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="var(--paper-edge)"/>${spikes}
      <line class="tfx-stream-cursor" id="tfx-stream-cursor" x1="0" y1="0" x2="0" y2="${H}" stroke="var(--ink)" stroke-opacity="0" stroke-width="1" stroke-dasharray="3 3"/>${hits}</svg>`;
  }
  // ── status evolution mini-hive: one hex per bucket (errors spread/recover) ─
  function _statusEvolution(series) {
    const s = (series || []).filter(b => b.n >= 0); const act = s.filter(b => b.n > 0);
    if (!act.length) return `<div class="tf-empty">no evolution yet</div>`;
    const col = _C(), maxN = Math.max(1, ...s.map(b => b.n)), R = 9, gap = 4, ox = R + 2, oy = R + 3;
    let cells = "";
    s.forEach((b, i) => {
      const cx = ox + i * (R * 1.5 + gap), cy = oy + (i % 2 ? R * 0.5 : 0);
      if (!b.n) { cells += `<g class="tfx-evo-cell" data-i="${i}"><path d="${_hexPath(cx, cy, R * 0.62)}" fill="none" stroke="var(--ink-mute)" stroke-opacity="0.25" stroke-width="0.7" stroke-dasharray="1.5 2"/><path class="tfx-evo-hit" data-i="${i}" d="${_hexPath(cx, cy, R + 2)}" fill="transparent"/></g>`; return; }
      const ep = (b.n4 + b.n5) / b.n, fill = b.n5 > 0 && ep >= 0.2 ? col.danger : ep >= 0.2 ? col.amber : ep > 0 ? col.amber : col.ok;
      const op = 0.4 + 0.6 * (b.n / maxN);
      cells += `<g class="tfx-evo-cell" data-i="${i}" style="--gc:${fill}"><path class="tfx-evo-hex" d="${_hexPath(cx, cy, R)}" fill="${fill}" fill-opacity="${op.toFixed(2)}" stroke="${fill}" stroke-width="${b.n5 > 0 ? 1.4 : 0.6}" stroke-opacity="0.75"/><path class="tfx-evo-hit" data-i="${i}" d="${_hexPath(cx, cy, R + 2)}" fill="transparent"/></g>`;
    });
    const W = ox + s.length * (R * 1.5 + gap) + R, H = oy + R * 1.5 + 5;
    return `<svg class="tfx-evo-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMinYMid meet" style="max-height:${H}px;width:100%;overflow:visible">${cells}</svg>`;
  }
  // ── insights for the inspection temporal visuals ─────────────────────────
  function _streamInsight(series, buckets) {
    const s = series || []; const act = s.map((b, i) => ({ b, i })).filter(x => x.b.n > 0);
    if (!act.length) return "No activity recorded for this colony in the window.";
    const lbl = (i) => buckets && buckets[i] ? (buckets[i].label + (buckets[i].sub ? " " + buckets[i].sub : "")) : "bucket " + (i + 1);
    const peak = act.slice().sort((a, b) => b.b.n - a.b.n)[0];
    const errBk = act.filter(x => (x.b.n4 + x.b.n5) > 0).sort((a, b) => (b.b.n4 + b.b.n5) - (a.b.n4 + a.b.n5))[0];
    const sev = act.filter(x => x.b.n5 > 0);
    let t = `Busiest at <b>${esc(lbl(peak.i))}</b> (${fmtNum(peak.b.n)} req).`;
    if (sev.length) t += ` Server errors surfaced ${sev.length === 1 ? "once, around" : "across " + sev.length + " buckets near"} <b>${esc(lbl(sev[sev.length - 1].i))}</b>.`;
    else if (errBk && (errBk.b.n4 + errBk.b.n5) / errBk.b.n >= 0.2) t += ` Client errors peaked around <b>${esc(lbl(errBk.i))}</b> (${Math.round(100 * (errBk.b.n4 + errBk.b.n5) / errBk.b.n)}%).`;
    else t += " No error spikes — clean throughout.";
    return t;
  }
  function _evoInsight(series) {
    const s = (series || []).filter(b => b.n > 0); if (!s.length) return "";
    const errBuckets = s.filter(b => (b.n4 + b.n5) > 0).length;
    const tail = s.slice(-3), tailErr = tail.some(b => (b.n4 + b.n5) / b.n >= 0.2);
    const head = s.slice(0, Math.max(1, s.length - 3)), headErr = head.some(b => (b.n4 + b.n5) / b.n >= 0.2);
    if (!errBuckets) return "Status held healthy across every active bucket.";
    if (headErr && !tailErr) return "Errors appeared earlier and the colony recovered — stabilising now.";
    if (!headErr && tailErr) return "Recently degrading — errors are concentrated in the latest buckets.";
    if (s.some(b => b.n5 > 0)) return "Server errors recur intermittently — not yet stable.";
    return "Client errors scattered through the window; no sustained outage.";
  }

  function expandHive(panel) {
    let sel = null, hmode = "error";   // selected colony · health-strip mode
    const rowsNow = () => (DATA.matrix || []).filter(m => m.total > 0);
    const overallState = () => {
      const rows = rowsNow(); if (!rows.length) return { txt: "no traffic", k: "neutral" };
      const t = rows.reduce((a, m) => a + m.total, 0), e = rows.reduce((a, m) => a + m.n4 + m.n5, 0), sev = rows.reduce((a, m) => a + m.n5, 0);
      const ep = t ? e / t : 0;
      if (sev > 0 && ep >= 0.15) return { txt: "critical", k: "danger" };
      if (ep >= 0.1) return { txt: "degraded", k: "amber" };
      return { txt: "stable", k: "ok" };
    };
    const buildShell = () => {
      const rows = rowsNow(), st = overallState();
      const total = rows.reduce((a, m) => a + m.total, 0);
      panel.innerHTML =
        `<div class="tfx-hv2-head">
           <span class="tfx-hv2-live"><i></i>live</span>
           <span class="tfx-hv2-period">${esc(GRAN_LABEL[GRAN] || "")} · ${fmtNum(total)} requests</span>
           <span class="tfx-hv2-state tfx-hv2-${st.k}">${st.txt}</span>
         </div>
         <div id="tfx-hive" class="tfx-hv2-hive"></div>
         <div class="tfx-sec-row">${_sec("method intelligence · click a colony to inspect")}
           <div class="ov-range tfx-hs-mode" id="tfx-hs-mode"><button data-m="error" aria-pressed="true">error</button><button data-m="success">success</button><button data-m="volume">volume</button></div></div>
         <div class="tfx-intel" id="tfx-intel"></div>
         <div id="tfx-inspect"></div>`;
      // hero hive (built once; live pulses target it via _hivePulse)
      $("tfx-hive").innerHTML = hiveColonySvg(DATA.matrix || [], { expanded: true });
      _wireHiveColony($("tfx-hive"), { onColony: (mm) => { sel = mm; renderIntel(); renderInspect(); } });
      $("tfx-hs-mode").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
        $("tfx-hs-mode").querySelectorAll("button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true");
        hmode = b.getAttribute("data-m"); renderIntel();
      }));
      if (!sel && rows.length) sel = rows[0].method;
      renderIntel(); renderInspect();
    };
    const renderIntel = () => {
      const host = $("tfx-intel"); if (!host) return;
      const rows = rowsNow();
      host.innerHTML = rows.map(m => {
        const { hl, bullets } = _methodIntel(m);
        return `<div class="tfx-card tfx-card-${hl.state}${m.method === sel ? " tfx-card-sel" : ""}" data-m="${esc(m.method)}">
          <div class="tfx-card-head"><span class="meth ${_methClass(m.method)}">${esc(m.method)}</span><span class="tfx-card-total">◉ ${fmtNum(m.total)}</span></div>
          <ul class="tfx-card-bul">${bullets.map(b => `<li class="tfx-bul-${b.k}">${esc(b.t)}</li>`).join("")}</ul>
          <div class="tfx-card-strip">${_colonyHealthStrip(m.series, hmode)}</div></div>`;
      }).join("") || `<div class="tf-empty">no methods</div>`;
      host.querySelectorAll(".tfx-card[data-m]").forEach(c => c.addEventListener("click", () => { sel = c.getAttribute("data-m"); renderIntel(); renderInspect(); }));
    };
    const renderInspect = () => {
      const host = $("tfx-inspect"); if (!host) return;
      const m = rowsNow().find(x => x.method === sel);
      if (!m) { host.innerHTML = ""; return; }
      const h = DATA.hero || { buckets: [] };
      const bk = h.buckets || [];
      host.innerHTML =
        `${_sec("colony inspection · " + esc(m.method))}
         <div class="tfx-insp-grid">
           <div><div class="tfx-insp-lbl">endpoint contributors</div>${_endpointLanes(m)}</div>
           <div><div class="tfx-insp-lbl">temporal activity · hover to scrub</div>${_temporalStream(m.series)}
             <p class="tfx-insight tfx-insp-ins">${_streamInsight(m.series, bk)}</p>
             <div class="tfx-insp-lbl" style="margin-top:8px">status evolution</div>${_statusEvolution(m.series)}
             <p class="tfx-insight tfx-insp-ins">${esc(_evoInsight(m.series))}</p></div>
         </div>
         <button class="tfx-drill-btn" id="tfx-insp-all">open all ${esc(m.method)} requests →</button>`;
      const bLabel = (i) => bk[i] ? (esc(bk[i].label) + (bk[i].sub ? " " + esc(bk[i].sub) : "")) : "bucket " + (i + 1);
      const bucketDrill = (i) => {
        const b = bk[i]; if (!b) return;
        openRequestContainer({ sinceMs: b.t_ms, untilMs: b.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: m.method + " · " + bLabel(i), method: m.method });
      };
      const seriesTip = (e, i) => {
        const s = (m.series || [])[i]; if (!s) return;
        const err = s.n4 + s.n5, ep = s.n ? Math.round(100 * err / s.n) : 0;
        tip(true, e.clientX, e.clientY, `<b>${bLabel(i)}</b><br>${fmtNum(s.n)} req · <span style="color:var(--c-ok,#2f6f3e)">${s.n2} ok</span>`
          + (s.n4 ? ` · <span style="color:var(--c-amber,#c98a2b)">${s.n4} 4xx</span>` : "")
          + (s.n5 ? ` · <span style="color:var(--c-danger,#b3402f)">${s.n5} 5xx</span>` : "")
          + `<br>${ep}% errors${s.n ? " · click → requests" : ""}`);
      };
      // temporal stream: scrub cursor + tooltip + click drill
      const cursor = host.querySelector("#tfx-stream-cursor");
      host.querySelectorAll(".tfx-stream-hit").forEach(r => {
        const i = +r.getAttribute("data-i");
        r.style.cursor = "pointer";
        r.addEventListener("mousemove", (e) => {
          seriesTip(e, i);
          if (cursor) { const x = r.getAttribute("x"), w = +r.getAttribute("width"); const cx = (+x + w / 2).toFixed(1); cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx); cursor.setAttribute("stroke-opacity", "0.5"); }
        });
        r.addEventListener("mouseleave", () => { tip(false); if (cursor) cursor.setAttribute("stroke-opacity", "0"); });
        r.addEventListener("click", () => bucketDrill(i));
      });
      // status evolution: hover glow + tooltip + click drill
      host.querySelectorAll(".tfx-evo-hit").forEach(hit => {
        const i = +hit.getAttribute("data-i");
        hit.style.cursor = "pointer";
        hit.addEventListener("mousemove", (e) => { seriesTip(e, i); const g = hit.closest(".tfx-evo-cell"); if (g) g.classList.add("tfx-evo-pop"); });
        hit.addEventListener("mouseleave", () => { tip(false); const g = hit.closest(".tfx-evo-pop"); if (g) g.classList.remove("tfx-evo-pop"); });
        hit.addEventListener("click", () => bucketDrill(i));
      });
      host.querySelectorAll(".tfx-lane[data-path]").forEach(l => l.addEventListener("click", () => {
        if (!h.buckets.length) return;
        openRequestContainer({ sinceMs: h.buckets[0].t_ms, untilMs: h.buckets[h.buckets.length - 1].t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: m.method + " · " + l.getAttribute("data-path"), method: m.method, path: l.getAttribute("data-path") });
      }));
      host.querySelectorAll(".tfx-lane[data-path]").forEach(l => {
        l.addEventListener("mousemove", (e) => {
          const p = l.getAttribute("data-path"); const ep = (m.endpoints || []).find(x => x.path === p); if (!ep) return;
          const errPct = ep.n ? Math.round(100 * (ep.e || 0) / ep.n) : 0;
          tip(true, e.clientX, e.clientY, `<b>${esc(p)}</b><br>${fmtNum(ep.n)} req · ${ep.pct}% of ${esc(m.method)}${errPct ? ` · ${errPct}% err` : ""}<br><span style="opacity:.6">click → these requests</span>`);
        });
        l.addEventListener("mouseleave", () => tip(false));
      });
      const allBtn = $("tfx-insp-all");
      if (allBtn) allBtn.addEventListener("click", () => { if (!h.buckets.length) return; openRequestContainer({ sinceMs: h.buckets[0].t_ms, untilMs: h.buckets[h.buckets.length - 1].t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: m.method + " · all", method: m.method }); });
    };
    buildShell();
    // live update: refresh header/cards/inspection text without rebuilding the
    // hive SVG (keeps breathing/diffuse stable); per-event glow via _hivePulse.
    _openExpand = { kind: "hive", update: () => {
      const st = overallState(); const sEl = panel.querySelector(".tfx-hv2-state");
      if (sEl) { sEl.className = "tfx-hv2-state tfx-hv2-" + st.k; sEl.textContent = st.txt; }
      const pEl = panel.querySelector(".tfx-hv2-period"); if (pEl) pEl.textContent = `${GRAN_LABEL[GRAN] || ""} · ${fmtNum(rowsNow().reduce((a, m) => a + m.total, 0))} requests`;
      renderIntel(); renderInspect();
    } };
  }

  // weekday × hour heatmap (7 rows Mon–Sun × 24 cols) — consumes DATA.clock.wkhr
  function _wkhrHeatmap(m) {
    const col = _C(), WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const max = Math.max(1, ...m.flat());
    const cell = 13, gap = 2, ox = 30, oy = 14, W = ox + 24 * (cell + gap), H = oy + 7 * (cell + gap) + 4;
    let cells = "", lbls = "";
    WD.forEach((d, r) => { lbls += `<text x="0" y="${oy + r * (cell + gap) + cell - 2}" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${d}</text>`; });
    [0, 6, 12, 18, 23].forEach(hh => { lbls += `<text x="${ox + hh * (cell + gap) + cell / 2}" y="10" text-anchor="middle" style="font:7px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${hh}</text>`; });
    for (let r = 0; r < 7; r++) for (let h = 0; h < 24; h++) {
      const v = (m[r] && m[r][h]) || 0, x = ox + h * (cell + gap), y = oy + r * (cell + gap);
      const op = v ? 0.2 + 0.8 * Math.min(1, v / max) : 1, fill = v ? col.accent : "rgba(120,110,90,.10)";
      cells += `<rect class="tf-wkhr-cell" data-r="${r}" data-h="${h}" data-v="${v}" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="2" fill="${fill}" fill-opacity="${op.toFixed(2)}" style="${v ? "cursor:pointer" : ""}"/>`;
    }
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMinYMid meet" style="max-height:${H}px;width:100%">${lbls}${cells}</svg>`;
  }
  function expandClock(panel) {
    panel.innerHTML =
      `<div class="tfx-controls"><div class="ov-range" id="tfx-clkmode"><button data-m="volume" aria-pressed="true">volume</button><button data-m="error">error rate</button></div><span class="tfx-clk-tz">device-local · ${esc((window.APIN && APIN.time) ? APIN.time.zone : "")}</span></div>
       <div class="tfx-clk-grid"><div id="tfx-clock"></div><div id="tfx-clkdetail" class="tfx-clk-detail"></div></div>`;
    let mode = "volume";
    const overview = () => {
      const c = DATA.clock || {}, hours = c.hours || [];
      const quiet = hours.filter(h => h.n > 0).reduce((m, h) => h.n < (m ? m.n : 1e9) ? h : m, null);
      const biz = hours.slice(9, 17).reduce((s, h) => s + h.n, 0), tot = hours.reduce((s, h) => s + h.n, 0);
      const bizPct = tot ? Math.round(100 * biz / tot) : 0;
      const topHours = hours.filter(h => h.n > 0).slice().sort((a, b) => b.n - a.n).slice(0, 5);
      const wkhr = c.wkhr;   // [7][24] from backend (T15)
      const insight = c.busiest_h != null
        ? `Traffic clusters around <b>${fmt12(c.busiest_h)}</b> — ${bizPct >= 60 ? "mostly business hours" : bizPct <= 30 ? "mostly off-hours" : "spread across the day"} (${bizPct}% within 9–17).`
        : "Rhythm emerges as traffic accrues.";
      $("tfx-clkdetail").innerHTML =
        `<div class="tfx-facts" style="flex-direction:column;gap:7px"><span>busiest hour <b>${c.busiest_h != null ? fmt12(c.busiest_h) : "—"}</b></span><span>quietest active <b>${quiet ? fmt12(quiet.h) : "—"}</b></span><span>business hrs (9–17) <b>${bizPct}%</b></span><span>total <b>${fmtNum(tot)}</b></span></div>
         ${_sec("top hours · click → requests")}${topHours.length ? _hbars(topHours.map(h => ({ label: fmt12(h.h), v: h.n, click: "h" + h.h})), _C().accent) : `<div class="tf-empty">no data</div>`}
         ${wkhr ? `${_sec("weekday × hour")}<div id="tfx-wkhr"></div>` : ""}
         ${_sec("insight")}<p class="tfx-insight">${insight}</p>
         <p class="tfx-clk-hint">click an hour wedge for its 60-minute breakdown.</p>`;
      $("tfx-clkdetail").querySelectorAll(".tfx-hbar[data-click]").forEach(b => b.addEventListener("click", () => { const hh = +b.getAttribute("data-click").slice(1); _drillHourOfDay(hh); }));
      if (wkhr && $("tfx-wkhr")) {
        const host = $("tfx-wkhr"); host.innerHTML = _wkhrHeatmap(wkhr);
        const WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        host.querySelectorAll(".tf-wkhr-cell").forEach(cell => {
          const rr = +cell.getAttribute("data-r"), hh = +cell.getAttribute("data-h"), v = +cell.getAttribute("data-v");
          cell.addEventListener("mousemove", (e) => { tip(true, e.clientX, e.clientY, `<b>${WD[rr]} ${fmt12(hh)}</b><br>${fmtNum(v)} request(s)${v ? "<br><span style=\"opacity:.7\">click → these requests</span>" : ""}`); cell.style.opacity = "0.75"; });
          cell.addEventListener("mouseleave", () => { tip(false); cell.style.opacity = ""; });
          if (v) cell.addEventListener("click", () => _drillWeekdayHour(rr, hh));
        });
      }
    };
    const detailHour = (hh) => {
      const h = (DATA.clock.hours || [])[hh]; if (!h) return;
      // 60-min breakdown for that hour-of-day, aggregated across the whole
      // clock window (the clock is hour-of-day, not a single date).
      const win = _clockWindow();
      $("tfx-clkdetail").innerHTML =
        `<div class="tfx-hour-head"><b>${fmt12(hh)} – ${fmt12((hh + 1) % 24)}</b><button class="tfx-back" id="tfx-clk-back">← all hours</button></div>
         <div class="tfx-facts"><span>${fmtNum(h.n)} requests · ${GRAN_LABEL[GRAN]}</span><span>${h.err_pct}% errors</span></div>
         ${_sec("per-minute · within this hour")}<div id="tfx-min" class="tf-empty">loading…</div>
         <button class="tfx-drill-btn" id="tfx-clk-drill">open request container →</button>`;
      $("tfx-clk-back").addEventListener("click", overview);
      $("tfx-clk-drill").addEventListener("click", () => _drillHourOfDay(hh));
      api(`/api/account/keys/${encodeURIComponent(PID)}/requests?since=${encodeURIComponent(_utcStr(win.sinceMs))}&until=${encodeURIComponent(_utcStr(win.untilMs))}&local_hour=${hh}&tz_off=${TZOFF}&limit=200`).then(({ body }) => {
        const rows = (body && body.data && body.data.items) || [];
        const mins = new Array(60).fill(0), merr = new Array(60).fill(0);
        rows.forEach(r => { const d = new Date((r.timestamp || "").replace(" ", "T") + "Z"); if (isNaN(d)) return; const m = d.getMinutes(); mins[m]++; if ((+r.status_code || 0) >= 400) merr[m]++; });
        const max = Math.max(1, ...mins); const host = $("tfx-min"); if (!host) return;
        host.className = ""; host.innerHTML =
          `<svg viewBox="0 0 600 110" preserveAspectRatio="none" style="width:100%;height:110px">
            ${mins.map((v, i) => { const x = (i / 60) * 600, w = 600 / 60 - 1.5, hh2 = v ? Math.max(2, v / max * 86) : 0; const er = v ? merr[i] / v : 0; const f = v === 0 ? "rgba(120,110,90,.12)" : er > 0.1 ? _C().danger : er > 0 ? _C().amber : _C().accent; return `<rect x="${x.toFixed(1)}" y="${(96 - hh2).toFixed(1)}" width="${w.toFixed(1)}" height="${hh2 ? hh2.toFixed(1) : 1}" rx="1" fill="${f}"><title>:${String(i).padStart(2, "0")} · ${v} req</title></rect>`; }).join("")}
            <line x1="0" y1="96" x2="600" y2="96" stroke="var(--paper-edge)"/>
            <text x="0" y="108" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">:00</text><text x="300" y="108" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">:30</text><text x="600" y="108" text-anchor="end" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">:59</text>
          </svg>`;
      });
    };
    const paint = () => {
      $("tfx-clock").innerHTML = clockSvg(DATA.clock || { hours: [], max: 0 }, 320, mode);
      _wireClock($("tfx-clock"), false, detailHour);
      if (!panel.querySelector(".tfx-hour-head")) overview();
    };
    paint();
    panel.querySelectorAll("#tfx-clkmode button").forEach(b => b.addEventListener("click", () => { panel.querySelectorAll("#tfx-clkmode button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); mode = b.getAttribute("data-m"); $("tfx-clock").innerHTML = clockSvg(DATA.clock || { hours: [], max: 0 }, 320, mode); _wireClock($("tfx-clock"), false, detailHour); }));
    _openExpand = { kind: "clock", update: paint };
  }

  function expandBytes(panel) {
    panel.innerHTML =
      `<div class="tfx-by-top">
         <div class="tfx-by-ringcol">
           <div class="tf-ring-wrap tf-ring-wrap-lg" id="tfx-by-ring"></div>
           <div class="tf-bytes-legend" id="tfx-by-legend"></div>
         </div>
         <div class="tfx-by-rivercol">${_sec("flow over time · river")}<div class="tf-river-wrap" id="tfx-by-river"></div></div>
       </div>
       ${_sec("endpoint channels · ingress ▶ · egress ◀")}<div id="tfx-by-chan"></div>
       <div id="tfx-by-extra"></div>
       ${_sec("insight · click to cycle")}<p class="tfx-insight tf-by-phrase" id="tfx-by-phrase" title="click for another reading"></p>`;
    const renderPhrase = (by) => {
      const ins = _bytesInsights(by);
      if (_byPhraseIdx >= ins.length) _byPhraseIdx = 0;
      const el = $("tfx-by-phrase"); if (el) el.innerHTML = ins[_byPhraseIdx].text + (ins.length > 1 ? ` <span class="tf-by-phrase-dot">${_byPhraseIdx + 1}/${ins.length}</span>` : "");
    };
    const paint = () => {
      const by = DATA.bytes || { buckets: [], by_endpoint: [] };
      // radial ring (larger) + center hub
      const ring = $("tfx-by-ring");
      if (ring) {
        ring.innerHTML = `${bandwidthRing(by, 300)}<div class="tf-ring-hub"><div class="tf-ring-hub-total">${fmtBytes((by.total_in || 0) + (by.total_out || 0))}</div><div class="tf-ring-hub-rate">idle</div></div>`;
        _wireRing(ring, by); _bytesLive();
      }
      const lg2 = $("tfx-by-legend");
      if (lg2) lg2.innerHTML = `<span style="color:${ING}">▲ in ${fmtBytes(by.total_in)} · ${fmtBytes(by.avg_in)}/req</span><span style="color:${EGR}">▼ out ${fmtBytes(by.total_out)} · ${fmtBytes(by.avg_out)}/req</span>${_ratioStr(by.total_in, by.total_out) ? `<span>${_ratioStr(by.total_in, by.total_out)}</span>` : ""}`;
      // river
      const riv = $("tfx-by-river");
      if (riv) { riv.innerHTML = riverSvg(by.buckets || []); _wireRiver(riv, by); }
      // endpoint channels
      const ch = $("tfx-by-chan");
      if (ch) { ch.innerHTML = endpointChannels(by); _wireChannels(ch, by); }
      // largest single request + payload percentiles
      const lg = by.largest, pct = by.pct;
      const ex = $("tfx-by-extra");
      if (ex) {
        ex.innerHTML =
          (lg ? `${_sec("largest single request")}<div class="tfx-largest"><b>${esc(lg.method || "")} ${esc(lg.path || "")}</b> · <span style="color:${ING}">in ${fmtBytes(lg.bin)}</span> · <span style="color:${EGR}">out ${fmtBytes(lg.bout)}</span> · ${esc(_local(lg.ts || ""))}<button class="tfx-drill-btn" id="tfx-lg-open">open request →</button></div>` : "")
          + (pct ? `${_sec("payload sizes")}<div class="tfx-facts"><span>in p50 <b>${fmtBytes(pct.in_p50)}</b></span><span>in p95 <b>${fmtBytes(pct.in_p95)}</b></span><span>out p50 <b>${fmtBytes(pct.out_p50)}</b></span></div>` : "");
        const lo = $("tfx-lg-open");
        if (lo && lg) lo.addEventListener("click", () => { if (window.APIN && APIN.requestDrawer && lg.id != null) APIN.requestDrawer.open(lg.id); else if (window.APIN && APIN.keyDetail && APIN.keyDetail.openRequest) APIN.keyDetail.openRequest(lg.id); });
      }
      renderPhrase(by);
    };
    paint();
    const ph = $("tfx-by-phrase");
    if (ph) ph.addEventListener("click", () => { _byPhraseIdx++; renderPhrase(DATA.bytes || { buckets: [] }); });
    _openExpand = { kind: "bytes", update: () => { _bytesLive(); } };   // live: tick counters only, keep SMIL/marching alive
  }

  const EXPANDERS = {
    hero:     { title: "Requests over time", build: expandHero },
    kpiTotal: { title: "Total requests",     build: expandKpiTotal },
    kpiPeak:  { title: "Peak throughput",    build: expandKpiPeak },
    kpiBusy:  { title: "Busiest periods",    build: expandKpiBusy },
    kpiData:  { title: "Data transfer",      build: expandKpiData },
    calendar: { title: "Activity calendar",  build: expandCalendar },
    hive:     { title: "Method × status hive", build: expandHive },
    clock:    { title: "Traffic rhythm",     build: expandClock },
    bytes:    { title: "Data transfer",      build: expandBytes },
  };

  // ════════════════════ CONTROLS / WIRING ═════════════════════════════════
  function syncGranButtons() {
    document.querySelectorAll("#tf-gran button").forEach(b => {
      if (b.getAttribute("data-gran") === GRAN) b.setAttribute("aria-pressed", "true");
      else b.removeAttribute("aria-pressed");
    });
  }
  function _openExpander(tfx, sourceCard) {
    const ex = EXPANDERS[tfx]; if (!ex || !window.APIN || !APIN.lightbox) return;
    APIN.lightbox.open({
      title: ex.title + (DATA && DATA.key && DATA.key.name ? " · " + DATA.key.name : ""),
      subtitle: GRAN_LABEL[GRAN] || "",
      sourceCard: sourceCard || null,
      hashKey: "tfx-" + tfx,
      build: ex.build,
    });
  }
  function wire() {
    if (_wired) return; _wired = true;
    document.querySelectorAll("#tf-gran button").forEach(b => b.addEventListener("click", () => {
      if (b.getAttribute("data-gran") === GRAN) return;
      GRAN = b.getAttribute("data-gran"); try { sessionStorage.setItem("tf_gran", GRAN); } catch (_) {}
      syncGranButtons();
      const st = $("tf-stack"); if (st) st.classList.add("ov-loading");
      if (_liveTimer) { clearTimeout(_liveTimer); _liveTimer = null; }
      refresh();
    }));
    const live = $("tf-live");
    if (live) live.addEventListener("click", () => {
      LIVE = !LIVE; live.setAttribute("data-on", LIVE ? "true" : "false");
      const lbl = live.querySelector(".ov-live-label"); if (lbl) lbl.textContent = LIVE ? "live" : "paused";
      if (LIVE) startSSE(); else stopSSE();
    });
    const rf = $("tf-refresh");
    if (rf) rf.addEventListener("click", () => { rf.classList.add("is-spinning"); setTimeout(() => rf.classList.remove("is-spinning"), 760); refresh(); });
    // Delegated expand handling (KPI tiles render dynamically).
    const stack = $("tf-stack");
    if (stack) stack.addEventListener("click", (e) => {
      const btn = e.target.closest(".ov-expand[data-tfx]"); if (!btn) return;
      _openExpander(btn.getAttribute("data-tfx"), btn.closest(".ov-card"));
    });
    window.addEventListener("resize", () => { if (_active && DATA) { renderHero(false); renderClock(false); renderBytes(false); renderHive(false); } });
  }

  // ════════════════════ ACTIVATE / DEACTIVATE ═════════════════════════════
  function activate(pid) {
    PID = pid || PID || (PID_META ? PID_META.content : null); _active = true;
    if (!$("tf-stack")) return;
    wire(); syncGranButtons();
    if (DATA) renderAll(false);   // instant from cache (no re-intro)
    refresh();
    if (LIVE) startSSE();
    // decay the live MB/s readout once a second even when no events arrive,
    // so the rate gracefully falls back to "idle" and the glow fades out.
    if (!_rateTimer) _rateTimer = setInterval(() => { if (_active) _bytesLive(); }, 1000);
  }
  function deactivate() {
    _active = false;
    stopSSE();
    if (_clockTick) { clearInterval(_clockTick); _clockTick = null; }
    if (_liveTimer) { clearTimeout(_liveTimer); _liveTimer = null; }
    if (_rateTimer) { clearInterval(_rateTimer); _rateTimer = null; }
    if (_rerenderRaf) { cancelAnimationFrame(_rerenderRaf); _rerenderRaf = null; }
  }

  window.APIN = window.APIN || {};
  window.APIN.keyTraffic = { activate, deactivate };
})();
