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

  // ── state ───────────────────────────────────────────────────────────────
  let PID = null, DATA = null, _active = false, _wired = false, _introDone = false;
  let GRAN = (function () { try { return sessionStorage.getItem("tf_gran") || "hour"; } catch (_) { return "hour"; } })();
  let LIVE = true, _refreshSeq = 0, _liveTimer = null;
  let _clockTick = null, _rerenderRaf = null, _dirty = {};
  let _openExpand = null;   // {kind, update} when a lightbox is open
  let _zoom = null;         // {i0,i1} bucket-index window for the hero (null = full)

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
    const r = await api(`/api/account/keys/${encodeURIComponent(PID)}/requests?since=${encodeURIComponent(sinceISO)}&until=${encodeURIComponent(untilISO)}&limit=200`);
    let rows = (r.body && r.body.data && r.body.data.items) || [];
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

  // ════════════════════ ① HERO — segmented blocks + brush/zoom ════════════
  const HERO_W = 760, HERO_H = 200, MAXBLOCKS = 14, BLK_H = 9, BLK_GAP = 3, BASE_OFF = 30;
  function heroSvg(buckets, max, off) {
    const col = _C(), n = buckets.length || 1;
    const slotW = HERO_W / n, bw = Math.max(6, Math.min(30, slotW - 7));
    const baseY = HERO_H - BASE_OFF;
    const stride = Math.max(1, Math.ceil(n / 12));
    let bars = "", labels = "", hits = "";
    buckets.forEach((b, i) => {
      const cx = i * slotW + slotW / 2, x = cx - bw / 2;
      const blocks = b.total > 0 ? Math.max(1, Math.round(b.total / (max || 1) * MAXBLOCKS)) : 0;
      const g = Math.round(blocks * (b.n2 / (b.total || 1)));
      const a = Math.round(blocks * (b.n4 / (b.total || 1)));
      let y = baseY;
      for (let k = 0; k < blocks; k++) {
        y -= BLK_H; const color = k < g ? col.ok : k < g + a ? col.amber : col.danger;
        bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${BLK_H - BLK_GAP}" rx="2" fill="${color}"/>`;
        y -= BLK_GAP;
      }
      if (i % stride === 0) {
        labels += `<text x="${cx.toFixed(1)}" y="${baseY + 13}" text-anchor="middle" style="font:9.5px 'JetBrains Mono',monospace;fill:var(--ink-soft)">${esc(b.label)}</text>`;
        if (b.sub) labels += `<text x="${cx.toFixed(1)}" y="${baseY + 24}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${esc(b.sub)}</text>`;
      }
      hits += `<rect class="tf-bar" data-i="${i + (off || 0)}" x="${(i * slotW).toFixed(1)}" y="0" width="${slotW.toFixed(1)}" height="${baseY}" fill="transparent"/>`;
    });
    return `<svg class="tf-hero-svg" viewBox="0 0 ${HERO_W} ${HERO_H}" preserveAspectRatio="none" style="height:${HERO_H}px">
      <line x1="0" y1="${baseY}" x2="${HERO_W}" y2="${baseY}" stroke="var(--paper-edge)" stroke-width="1"/>${bars}${labels}${hits}</svg>`;
  }
  // brush strip: a thin overview of ALL buckets with a draggable window.
  const BR_W = 760, BR_H = 34;
  function brushSvg(buckets, max, z) {
    const col = _C(), n = buckets.length || 1, sw = BR_W / n;
    let bars = "";
    buckets.forEach((b, i) => {
      const h = b.total > 0 ? Math.max(2, (b.total / (max || 1)) * (BR_H - 6)) : 1;
      const er = b.total ? (b.n4 + b.n5) / b.total : 0;
      const fill = b.total === 0 ? "rgba(120,110,90,.14)" : er > 0.1 ? col.danger : er > 0 ? col.amber : col.accent;
      bars += `<rect x="${(i * sw + 1).toFixed(1)}" y="${(BR_H - h).toFixed(1)}" width="${Math.max(1, sw - 2).toFixed(1)}" height="${h.toFixed(1)}" rx="1" fill="${fill}" fill-opacity="${b.total ? 0.7 : 1}"/>`;
    });
    const x0 = (z.i0 / n) * BR_W, x1 = ((z.i1 + 1) / n) * BR_W;
    const win = `<rect class="tf-brush-win" x="${x0.toFixed(1)}" y="0" width="${(x1 - x0).toFixed(1)}" height="${BR_H}" fill="rgba(82,183,136,.14)" stroke="var(--ink)" stroke-width="1.2"/>
      <rect class="tf-brush-h" data-edge="0" x="${(x0 - 3).toFixed(1)}" y="0" width="6" height="${BR_H}" fill="var(--ink)" fill-opacity=".0" style="cursor:ew-resize"/>
      <rect class="tf-brush-h" data-edge="1" x="${(x1 - 3).toFixed(1)}" y="0" width="6" height="${BR_H}" fill="var(--ink)" fill-opacity=".0" style="cursor:ew-resize"/>`;
    return `<svg class="tf-brush-svg" viewBox="0 0 ${BR_W} ${BR_H}" preserveAspectRatio="none" style="height:${BR_H}px;width:100%;cursor:crosshair">${bars}${win}</svg>`;
  }
  function _heroWindow(total) {
    const n = total;
    if (!_zoom) return { i0: 0, i1: n - 1 };
    return { i0: Math.max(0, Math.min(_zoom.i0, n - 1)), i1: Math.max(0, Math.min(_zoom.i1, n - 1)) };
  }
  function heroChart(host, intro) {
    const h = DATA.hero || { buckets: [], max: 0 };
    if (!h.buckets.length || h.max === 0) { host.innerHTML = `<div class="tf-empty">No traffic in this range yet.</div>`; return; }
    const z = _heroWindow(h.buckets.length);
    const slice = h.buckets.slice(z.i0, z.i1 + 1);
    const sliceMax = Math.max(1, ...slice.map(b => b.total));
    const zoomed = z.i0 > 0 || z.i1 < h.buckets.length - 1;
    host.innerHTML =
      `<div class="tf-hero-main">${heroSvg(slice, sliceMax, z.i0)}</div>
       <div class="tf-brush-wrap"><div class="tf-brush-cap">${zoomed ? `showing ${z.i0 + 1}–${z.i1 + 1} of ${h.buckets.length} · <button class="tf-brush-reset" id="tf-brush-reset">reset zoom</button>` : `drag below to zoom`}</div>${brushSvg(h.buckets, h.max, z)}</div>`;
    _wireHeroMain(host.querySelector(".tf-hero-main"), h.buckets, intro);
    _wireBrush(host.querySelector(".tf-brush-wrap"), h.buckets, () => heroChart(host, false));
    const rs = host.querySelector("#tf-brush-reset");
    if (rs) rs.addEventListener("click", () => { _zoom = null; heroChart(host, false); });
  }
  function _wireHeroMain(main, buckets, intro) {
    const svg = main.querySelector("svg"); if (!svg) return;
    svg.querySelectorAll(".tf-bar").forEach(hrect => {
      hrect.addEventListener("mousemove", (e) => {
        const b = buckets[+hrect.getAttribute("data-i")]; if (!b) return;
        const errp = b.total ? Math.round(100 * (b.n4 + b.n5) / b.total) : 0;
        tip(true, e.clientX, e.clientY, `<b>${esc(b.label)}${b.sub ? " · " + esc(b.sub) : ""}</b><br>${fmtNum(b.total)} req · 2xx ${b.n2} · 4xx ${b.n4} · 5xx ${b.n5} · ${errp}% err`);
      });
      hrect.addEventListener("mouseleave", () => tip(false));
      hrect.addEventListener("click", () => {
        const b = buckets[+hrect.getAttribute("data-i")];
        if (b && b.total) openRequestContainer({ sinceMs: b.t_ms, untilMs: b.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN] || 3600e3), label: b.label + (b.sub ? " " + b.sub : "") });
      });
    });
    if (intro && window.APIN && APIN.fx) svg.querySelectorAll("rect[rx]").forEach((r, k) => {
      try { r.animate([{ transform: "scaleY(0)", transformOrigin: "center bottom" }, { transform: "scaleY(1)", transformOrigin: "center bottom" }], { duration: 320, delay: Math.min(600, k * 4), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {}
    });
  }
  function _wireBrush(wrap, buckets, rerender) {
    const svg = wrap.querySelector(".tf-brush-svg"); if (!svg) return;
    const n = buckets.length;
    const idxAt = (clientX) => {
      const r = svg.getBoundingClientRect();
      return Math.max(0, Math.min(n - 1, Math.floor(((clientX - r.left) / r.width) * n)));
    };
    let mode = null, anchor = 0;
    const onMove = (e) => {
      if (!mode) return;
      const i = idxAt(e.clientX);
      if (mode === "new") { _zoom = { i0: Math.min(anchor, i), i1: Math.max(anchor, i) }; }
      else if (mode === "e0") { _zoom = { i0: Math.min(i, _zoom.i1), i1: _zoom.i1 }; }
      else if (mode === "e1") { _zoom = { i0: _zoom.i0, i1: Math.max(i, _zoom.i0) }; }
      rerender();
    };
    const onUp = () => { mode = null; window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
    svg.addEventListener("mousedown", (e) => {
      const edge = e.target.closest(".tf-brush-h");
      if (edge) { mode = edge.getAttribute("data-edge") === "0" ? "e0" : "e1"; if (!_zoom) _zoom = { i0: 0, i1: n - 1 }; }
      else { mode = "new"; anchor = idxAt(e.clientX); _zoom = { i0: anchor, i1: anchor }; }
      window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp); rerender();
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
  function calSvg(days, weeks, mode, cell) {
    const col = _C(), g = _calGrid(days, weeks);
    cell = cell || 13; const gap = 3, ox = 22, oy = 16;
    const W = ox + g.ncols * (cell + gap), H = oy + 7 * (cell + gap);
    let rects = "", monLbls = "";
    const todayKey = (() => { const t = new Date(); return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`; })();
    let lastMon = -1;
    g.cells.forEach(c => {
      const x = ox + c.c * (cell + gap), y = oy + c.r * (cell + gap);
      let fill = "rgba(120,110,90,.10)", op = 1;
      if (c.n > 0) {
        if (mode === "error") { const er = c.e / c.n; fill = er === 0 ? col.ok : er < 0.1 ? col.amber : col.danger; }
        else { fill = col.accent; op = 0.2 + 0.8 * Math.min(1, c.n / (g.max || 1)); }
      }
      rects += `<rect class="tf-cal-cell" data-key="${c.key}" data-ms="${c.dt}" data-n="${c.n}" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="3" fill="${fill}" fill-opacity="${op.toFixed(2)}"${c.key === todayKey ? ' stroke="var(--ink)" stroke-width="1.3"' : ""}><title>${esc(c.label)} · ${c.n} req · ${c.n ? Math.round(100 * c.e / c.n) : 0}% err</title></rect>`;
      if (c.r === 0) { const mo = new Date(c.dt).getMonth(); if (mo !== lastMon) { lastMon = mo; monLbls += `<text x="${x}" y="11" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][mo]}</text>`; } }
    });
    const wd = ["", "Mon", "", "Wed", "", "Fri", ""].map((d, r) => d ? `<text x="0" y="${oy + r * (cell + gap) + cell - 2}" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${d}</text>` : "").join("");
    return `<svg class="tf-cal-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMinYMid meet" style="max-height:${H}px">${monLbls}${wd}${rects}</svg>`;
  }
  function _wireCal(host) {
    host.querySelectorAll(".tf-cal-cell").forEach(c => {
      c.addEventListener("mousemove", (e) => { const t = c.querySelector("title"); tip(true, e.clientX, e.clientY, esc(t ? t.textContent : "")); });
      c.addEventListener("mouseleave", () => tip(false));
      c.addEventListener("click", () => { const ms = +c.getAttribute("data-ms"); const n = +c.getAttribute("data-n"); if (ms && n) openRequestContainer({ sinceMs: ms, untilMs: ms + 86400e3, label: new Date(ms).toLocaleDateString([], { month: "short", day: "numeric" }) }); });
    });
  }
  function renderCalendar(intro, mode, weeks, cell) {
    const host = $("tf-cal-body"); if (!host || !DATA) return;
    host.innerHTML = calSvg((DATA.calendar || {}).days, weeks || 18, mode || "volume", cell);
    _wireCal(host);
    if (intro && window.APIN && APIN.fx) host.querySelectorAll(".tf-cal-cell").forEach(c => { try { c.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, delay: Math.min(640, (+c.getAttribute("data-ms")) % 640), easing: "ease", fill: "backwards" }); } catch (_) {} });
  }

  // ════════════════════ ④ HIVE — Method × Status honeycomb ════════════════
  function _hexPath(cx, cy, r) {
    let p = "";
    for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 90); p += (i ? "L" : "M") + (cx + r * Math.cos(a)).toFixed(1) + " " + (cy + r * Math.sin(a)).toFixed(1) + " "; }
    return p + "Z";
  }
  function hiveSvg(matrix, size) {
    const col = _C();
    const cols = [{ k: "n2", lbl: "2xx", c: col.ok }, { k: "n4", lbl: "4xx", c: col.amber }, { k: "n5", lbl: "5xx", c: col.danger }];
    const rows = matrix.filter(m => m.total > 0);
    if (!rows.length) return `<div class="tf-empty">No requests to map yet.</div>`;
    const allMax = Math.max(1, ...rows.flatMap(m => [m.n2, m.n4, m.n5]));
    const rH = 58, hexR = 22, ox = 64, oy = 30, colW = 70;
    const W = ox + cols.length * colW + 30, H = oy + rows.length * rH + 10;
    let body = "", head = "";
    cols.forEach((c, ci) => { head += `<text x="${ox + ci * colW + (ci % 2 ? rH * 0 : 0)}" y="16" text-anchor="middle" style="font:600 10px 'JetBrains Mono',monospace;fill:${c.c}">${c.lbl}</text>`; });
    rows.forEach((m, ri) => {
      const cy = oy + ri * rH + hexR;
      body += `<text x="${ox - 24}" y="${cy + 4}" text-anchor="end" style="font:600 11px 'JetBrains Mono',monospace;fill:var(--ink)">${esc(m.method)}</text>`;
      cols.forEach((c, ci) => {
        const v = m[c.k] || 0;
        const cx = ox + ci * colW + (ri % 2 ? 14 : 0);   // honeycomb offset on alt rows
        const r = v > 0 ? Math.max(7, hexR * Math.sqrt(v / allMax)) : 4;
        const fill = v > 0 ? c.c : "rgba(120,110,90,.10)";
        body += `<path class="tf-hex" data-m="${esc(m.method)}" data-s="${c.lbl}" data-v="${v}" d="${_hexPath(cx, cy, r)}" fill="${fill}" fill-opacity="${v > 0 ? 0.82 : 1}" stroke="var(--paper)" stroke-width="1.5"/>`;
        if (v > 0) body += `<text x="${cx}" y="${cy + 3}" text-anchor="middle" pointer-events="none" style="font:600 9px 'JetBrains Mono',monospace;fill:var(--paper)">${v >= 1000 ? (v / 1000).toFixed(1) + "k" : v}</text>`;
      });
    });
    return `<svg class="tf-hive-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="max-height:${H}px;width:100%">${head}${body}</svg>`;
  }
  function _wireHive(host) {
    host.querySelectorAll(".tf-hex").forEach(h => {
      h.addEventListener("mousemove", (e) => { const v = +h.getAttribute("data-v"); if (!v) return; tip(true, e.clientX, e.clientY, `<b>${esc(h.getAttribute("data-m"))} · ${esc(h.getAttribute("data-s"))}</b><br>${fmtNum(v)} request(s)`); h.style.filter = "brightness(1.12)"; });
      h.addEventListener("mouseleave", () => { tip(false); h.style.filter = ""; });
      h.addEventListener("click", () => {
        const v = +h.getAttribute("data-v"); if (!v || !DATA.hero || !DATA.hero.buckets.length) return;
        const first = DATA.hero.buckets[0], last = DATA.hero.buckets[DATA.hero.buckets.length - 1];
        openRequestContainer({ sinceMs: first.t_ms, untilMs: last.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: h.getAttribute("data-m") + " · " + h.getAttribute("data-s"), method: h.getAttribute("data-m"), status: h.getAttribute("data-s") });
      });
    });
  }
  function renderHive(intro) {
    const host = $("tf-hive-body"); if (!host || !DATA) return;
    host.innerHTML = hiveSvg(DATA.matrix || [], 0);
    _wireHive(host);
    if (intro && window.APIN && APIN.fx) host.querySelectorAll(".tf-hex").forEach((h, i) => { try { h.animate([{ opacity: 0, transform: "scale(.4)" }, { opacity: 1, transform: "scale(1)" }], { duration: 360, delay: Math.min(700, i * 36), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
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
    [0, 6, 12, 18].forEach(hh => { const [lx, ly] = _polar(cx, cy, rmax + 12, hh / 24); labels += `<text x="${lx.toFixed(1)}" y="${(ly + 3).toFixed(1)}" text-anchor="middle" style="font:9px 'JetBrains Mono',monospace;fill:var(--ink-soft)">${String(hh).padStart(2, "0")}</text>`; });
    const hubTxt = clock.busiest_h != null ? `peak ${String(clock.busiest_h).padStart(2, "0")}h` : "";
    return `<svg class="tf-clock-svg" viewBox="0 0 ${size} ${size}" style="max-width:${size}px;margin:0 auto">
      ${rings}${wedges}${labels}
      <line id="tf-clock-hand" x1="${cx}" y1="${cy}" x2="${cx}" y2="${(cy - rmax - 4).toFixed(1)}" stroke="${col.ink}" stroke-width="2" stroke-linecap="round" transform="rotate(0 ${cx} ${cy})"/>
      <circle cx="${cx}" cy="${cy}" r="${(r0 - 4).toFixed(1)}" fill="var(--paper)" stroke="var(--paper-edge)"/>
      <text x="${cx}" y="${(cy + 3).toFixed(1)}" text-anchor="middle" style="font:9px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${hubTxt}</text></svg>`;
  }
  function _startClockHand() {
    if (_clockTick) clearInterval(_clockTick);
    const move = () => {
      const hand = $("tf-clock-hand"); if (!hand) return;
      const now = new Date(); const frac = (now.getHours() + now.getMinutes() / 60 + now.getSeconds() / 3600) / 24;
      const cx = hand.getAttribute("x1");
      hand.style.transition = "transform 1s linear";
      hand.setAttribute("transform", `rotate(${(frac * 360).toFixed(2)} ${cx} ${cx})`);
    };
    move(); _clockTick = setInterval(move, 1000);
  }
  // local 'that hour' window today (or yesterday if still in the future) → ms
  function _hourWindow(hh) {
    const d = new Date(); d.setHours(hh, 0, 0, 0); if (d > new Date()) d.setDate(d.getDate() - 1);
    return { sinceMs: d.getTime(), untilMs: d.getTime() + 3600e3 };
  }
  function _wireClock(host, intro, onWedge) {
    const svg = host.querySelector("svg"); if (!svg) return;
    svg.querySelectorAll(".tf-wedge").forEach(w => {
      w.addEventListener("mousemove", (e) => {
        const h = (DATA.clock.hours || [])[+w.getAttribute("data-h")]; if (!h) return;
        tip(true, e.clientX, e.clientY, `<b>${String(h.h).padStart(2, "0")}:00–${String((h.h + 1) % 24).padStart(2, "0")}:00</b><br>${fmtNum(h.n)} req · ${h.err_pct}% err`);
        svg.querySelectorAll(".tf-wedge").forEach(o => o.style.opacity = o === w ? "1" : "0.4");
      });
      w.addEventListener("mouseleave", () => { tip(false); svg.querySelectorAll(".tf-wedge").forEach(o => o.style.opacity = "1"); });
      w.addEventListener("click", () => { const hh = +w.getAttribute("data-h"); if (onWedge) onWedge(hh); else { const win = _hourWindow(hh); openRequestContainer({ sinceMs: win.sinceMs, untilMs: win.untilMs, label: String(hh).padStart(2, "0") + ":00" }); } });
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
  const BY_W = 760, BY_H = 168, BY_PAD = 22;
  function bytesSvg(buckets) {
    const col = _C(), n = buckets.length || 1, mid = BY_H / 2;
    const maxIn = Math.max(1, ...buckets.map(b => b.bin)), maxOut = Math.max(1, ...buckets.map(b => b.bout));
    const X = (i) => BY_PAD + (n <= 1 ? (BY_W - 2 * BY_PAD) / 2 : (i / (n - 1)) * (BY_W - 2 * BY_PAD));
    const upY = (v) => mid - (v / maxIn) * (mid - 14), dnY = (v) => mid + (v / maxOut) * (mid - 14);
    const inLine = buckets.map((b, i) => `${X(i).toFixed(1)},${upY(b.bin).toFixed(1)}`).join(" ");
    const outLine = buckets.map((b, i) => `${X(i).toFixed(1)},${dnY(b.bout).toFixed(1)}`).join(" ");
    const inArea = `${BY_PAD},${mid} ` + inLine + ` ${BY_W - BY_PAD},${mid}`, outArea = `${BY_PAD},${mid} ` + outLine + ` ${BY_W - BY_PAD},${mid}`;
    // peak markers + independent-axis labels
    const iMax = buckets.reduce((m, b, i) => b.bin > buckets[m].bin ? i : m, 0);
    const oMax = buckets.reduce((m, b, i) => b.bout > buckets[m].bout ? i : m, 0);
    const outLabY = Math.min(dnY(maxOut) + 11, BY_H - 18);
    const peak = `<text x="${X(iMax).toFixed(1)}" y="${(upY(maxIn) - 4).toFixed(1)}" text-anchor="middle" style="font:8.5px 'JetBrains Mono',monospace;fill:${col.accent}">${fmtBytes(maxIn)}</text>
      <text x="${X(oMax).toFixed(1)}" y="${outLabY.toFixed(1)}" text-anchor="middle" style="font:8.5px 'JetBrains Mono',monospace;fill:${col.ok}">${fmtBytes(maxOut)}</text>`;
    const labStride = Math.max(1, Math.ceil(n / 6));
    let xlab = "";
    buckets.forEach((b, i) => { if (i % labStride === 0) xlab += `<text x="${X(i).toFixed(1)}" y="${BY_H - 3}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${esc(b.label)}</text>`; });
    return `<svg class="tf-bytes-svg" viewBox="0 0 ${BY_W} ${BY_H}" preserveAspectRatio="none" style="height:${BY_H}px">
      <defs><linearGradient id="tfIn" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="${col.accent}" stop-opacity="0"/><stop offset="1" stop-color="${col.accent}" stop-opacity=".32"/></linearGradient>
      <linearGradient id="tfOut" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${col.ok}" stop-opacity="0"/><stop offset="1" stop-color="${col.ok}" stop-opacity=".32"/></linearGradient></defs>
      <polygon points="${inArea}" fill="url(#tfIn)"/><polyline points="${inLine}" fill="none" stroke="${col.accent}" stroke-width="1.6"/>
      <polygon points="${outArea}" fill="url(#tfOut)"/><polyline points="${outLine}" fill="none" stroke="${col.ok}" stroke-width="1.6"/>
      <line x1="${BY_PAD}" y1="${mid}" x2="${BY_W - BY_PAD}" y2="${mid}" stroke="var(--paper-edge)" stroke-width="1"/>
      <text x="2" y="14" style="font:8px 'JetBrains Mono',monospace;fill:${col.accent}">in ▲</text>
      <text x="2" y="${BY_H - 14}" style="font:8px 'JetBrains Mono',monospace;fill:${col.ok}">out ▼</text>
      ${peak}${xlab}
      <rect id="tf-bytes-hit" x="0" y="0" width="${BY_W}" height="${BY_H}" fill="transparent"/></svg>`;
  }
  function renderBytes(intro) {
    const host = $("tf-bytes-body"); if (!host || !DATA) return;
    const by = DATA.bytes || { buckets: [] };
    if (!by.buckets.length || (by.total_in === 0 && by.total_out === 0)) { host.innerHTML = `<div class="tf-empty">No payload data in this range.</div>`; return; }
    host.innerHTML = bytesSvg(by.buckets) +
      `<div class="tf-bytes-legend"><span style="color:${_C().accent}">▲ in ${fmtBytes(by.total_in)} · ${fmtBytes(by.avg_in)}/req</span><span style="color:${_C().ok}">▼ out ${fmtBytes(by.total_out)} · ${fmtBytes(by.avg_out)}/req</span>${_ratioStr(by.total_in, by.total_out) ? `<span>${_ratioStr(by.total_in, by.total_out)}</span>` : ""}</div>`;
    const hit = $("tf-bytes-hit");
    if (hit) {
      hit.addEventListener("mousemove", (e) => {
        const r = hit.getBoundingClientRect(); const i = Math.round(((e.clientX - r.left) / r.width) * (by.buckets.length - 1));
        const b = by.buckets[Math.max(0, Math.min(by.buckets.length - 1, i))]; if (!b) return;
        tip(true, e.clientX, e.clientY, `<b>${esc(b.label)}${b.sub ? " · " + esc(b.sub) : ""}</b><br>in ${fmtBytes(b.bin)} · out ${fmtBytes(b.bout)}`);
      });
      hit.addEventListener("mouseleave", () => tip(false));
    }
    if (intro && window.APIN && APIN.fx) host.querySelectorAll("polyline,polygon").forEach(p => { try { p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 420, easing: "ease" }); } catch (_) {} });
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
    // hive matrix: bump method × status cell
    if (DATA.matrix) {
      const m = (ev.method || "?").toUpperCase();
      let row = DATA.matrix.find(x => x.method === m);
      if (!row) { row = { method: m, n2: 0, n4: 0, n5: 0, total: 0 }; DATA.matrix.push(row); }
      row[sb] += 1; row.total = row.n2 + row.n4 + row.n5;
      DATA.matrix.sort((a, b) => b.total - a.total);
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
    }
    if (DATA.bytes && DATA.bytes.buckets.length) {
      const lb = DATA.bytes.buckets[DATA.bytes.buckets.length - 1];
      lb.bin += (+ev.bytes_in || 0); lb.bout += (+ev.bytes_out || 0);
      DATA.bytes.total_in += (+ev.bytes_in || 0); DATA.bytes.total_out += (+ev.bytes_out || 0);
      const tr = DATA.stats ? DATA.stats.total : 0;
      DATA.bytes.avg_in = tr ? Math.round(DATA.bytes.total_in / tr) : 0;
      DATA.bytes.avg_out = tr ? Math.round(DATA.bytes.total_out / tr) : 0;
      DATA.bytes.ratio = DATA.bytes.total_in ? Math.round(10 * DATA.bytes.total_out / DATA.bytes.total_in) / 10 : null;
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
    if (_dirty.hive) renderHive(false);
    if (_dirty.bytes) renderBytes(false);
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

  function expandHero(panel) {
    panel.innerHTML =
      `<div class="tfx-controls">
        <div class="ov-range" id="tfx-gran">${["hour", "day", "week", "month"].map(g => `<button data-g="${g}"${g === GRAN ? ' aria-pressed="true"' : ""}>${g[0].toUpperCase() + g.slice(1)}</button>`).join("")}</div>
        <div class="ov-range" id="tfx-ov"><button data-ov="none" aria-pressed="true">bars</button><button data-ov="err">+ error %</button></div>
      </div>
      <div id="tfx-hero"></div>
      <div class="tfx-legend"><span><i class="rdot s-2xx"></i> 2xx</span><span><i class="rdot s-4xx"></i> 4xx</span><span><i class="rdot s-5xx"></i> 5xx</span><span style="margin-left:auto;font-style:italic">drag the strip to zoom · click a bar → requests</span></div>
      ${_sec("at a glance")}<div id="tfx-hero-stats" class="tfx-facts"></div>`;
    let ov = "none";
    const paint = () => {
      const host = $("tfx-hero"); heroChart(host, false);
      if (ov === "err") _overlayErr(host.querySelector(".tf-hero-main"));
      const s = DATA.stats || {};
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
  function expandKpiTotal(panel) {
    const paint = () => {
      const s = DATA.stats || {}, h = DATA.hero || { buckets: [] };
      const n2 = h.buckets.reduce((a, b) => a + b.n2, 0), n4 = h.buckets.reduce((a, b) => a + b.n4, 0), n5 = h.buckets.reduce((a, b) => a + b.n5, 0);
      panel.innerHTML =
        `<div class="tfx-bignum"><span class="apin-odometer" id="tfx-tot-num">${fmtNum(s.total)}</span><span class="tfx-bignum-lbl">requests · ${GRAN_LABEL[GRAN]}</span></div>
         ${_sec("status mix")}<div class="tfx-split">${_donut([{ l: "2xx", v: n2, c: _C().ok }, { l: "4xx", v: n4, c: _C().amber }, { l: "5xx", v: n5, c: _C().danger }])}<div class="tfx-facts" style="flex-direction:column;gap:6px"><span><i class="rdot s-2xx"></i> 2xx <b>${fmtNum(n2)}</b></span><span><i class="rdot s-4xx"></i> 4xx <b>${fmtNum(n4)}</b></span><span><i class="rdot s-5xx"></i> 5xx <b>${fmtNum(n5)}</b></span><span>error rate <b>${s.error_pct}%</b></span></div></div>
         ${_sec("over time")}<div id="tfx-tot-hero"></div>`;
      const o = $("tfx-tot-num"); if (window.APIN && APIN.odometer) APIN.odometer.roll(o, fmtNum(s.total));
      heroChart($("tfx-tot-hero"), false);
    };
    paint(); _openExpand = { kind: "kpiTotal", update: paint };
  }
  // KPI: Peak/min → busiest hour rank + clock + facts
  function expandKpiPeak(panel) {
    const paint = () => {
      const s = DATA.stats || {}, c = DATA.clock || { hours: [] };
      const ranked = (c.hours || []).filter(h => h.n > 0).sort((a, b) => b.n - a.n).slice(0, 6)
        .map(h => ({ label: String(h.h).padStart(2, "0") + ":00", v: h.n, click: "h" + h.h }));
      panel.innerHTML =
        `<div class="tfx-bignum"><span id="tfx-peak-num">${fmtNum(s.peak_per_min)}</span><span class="tfx-bignum-lbl">peak requests / minute</span></div>
         <div class="tfx-facts"><span>busiest bucket <b>${esc(s.busiest_label)}</b> (${fmtNum(s.busiest_count)})</span><span>busiest hour <b>${c.busiest_h != null ? String(c.busiest_h).padStart(2, "0") + ":00" : "—"}</b></span></div>
         ${_sec("busiest hours (click → requests)")}${ranked.length ? _hbars(ranked, _C().accent) : `<div class="tf-empty">no data</div>`}`;
      panel.querySelectorAll(".tfx-hbar[data-click]").forEach(b => b.addEventListener("click", () => { const hh = +b.getAttribute("data-click").slice(1); const w = _hourWindow(hh); openRequestContainer({ sinceMs: w.sinceMs, untilMs: w.untilMs, label: String(hh).padStart(2, "0") + ":00" }); }));
    };
    paint(); _openExpand = { kind: "kpiPeak", update: paint };
  }
  // KPI: Busiest → top buckets ranked, click → request container
  function expandKpiBusy(panel) {
    const paint = () => {
      const h = DATA.hero || { buckets: [] };
      const ranked = h.buckets.map((b, i) => ({ b, i })).filter(x => x.b.total > 0).sort((a, b) => b.b.total - a.b.total).slice(0, 10)
        .map(x => ({ label: x.b.label + (x.b.sub ? " " + x.b.sub : ""), v: x.b.total, click: String(x.i) }));
      panel.innerHTML =
        `<div class="tfx-bignum"><span>${esc((DATA.stats || {}).busiest_label || "—")}</span><span class="tfx-bignum-lbl">busiest ${GRAN === "hour" ? "hour" : "bucket"} · ${fmtNum((DATA.stats || {}).busiest_count || 0)} requests</span></div>
         ${_sec("top buckets (click → requests)")}${ranked.length ? _hbars(ranked, _C().accent) : `<div class="tf-empty">no data</div>`}`;
      panel.querySelectorAll(".tfx-hbar[data-click]").forEach(b => b.addEventListener("click", () => { const i = +b.getAttribute("data-click"); const bb = h.buckets[i]; if (bb) openRequestContainer({ sinceMs: bb.t_ms, untilMs: bb.t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: bb.label + (bb.sub ? " " + bb.sub : "") }); }));
    };
    paint(); _openExpand = { kind: "kpiBusy", update: paint };
  }
  // KPI: Data → bytes mirror + by-endpoint + ratio (shared with bytes expand)
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

  function expandHive(panel) {
    const paint = () => {
      const rows = (DATA.matrix || []).filter(m => m.total > 0);
      const grand = rows.reduce((a, m) => a + m.total, 0) || 1;
      panel.innerHTML =
        `<div id="tfx-hive"></div>
         ${_sec("by method")}${_hbars(rows.map(m => ({ label: m.method, v: m.total, vl: fmtNum(m.total) + " · " + Math.round(100 * m.total / grand) + "%", click: "m" + m.method })), _C().accent)}
         ${_sec("error share by method")}${_hbars(rows.map(m => ({ label: m.method, v: m.n4 + m.n5, vl: (m.total ? Math.round(100 * (m.n4 + m.n5) / m.total) : 0) + "%" })), _C().danger)}
         <div class="tfx-legend" style="margin-top:10px"><span style="font-style:italic">hex size = volume · click a cell or bar → requests for that method</span></div>`;
      $("tfx-hive").innerHTML = hiveSvg(DATA.matrix || [], 0);
      _wireHive($("tfx-hive"));
      panel.querySelectorAll(".tfx-hbar[data-click]").forEach(b => b.addEventListener("click", () => {
        const m = b.getAttribute("data-click").slice(1); const h = DATA.hero || { buckets: [] }; if (!h.buckets.length) return;
        openRequestContainer({ sinceMs: h.buckets[0].t_ms, untilMs: h.buckets[h.buckets.length - 1].t_ms + (DATA.bucket_ms || GRAN_MS[GRAN]), label: m + " · all", method: m });
      }));
    };
    paint(); _openExpand = { kind: "hive", update: paint };
  }

  function expandClock(panel) {
    panel.innerHTML =
      `<div class="tfx-controls"><div class="ov-range" id="tfx-clkmode"><button data-m="volume" aria-pressed="true">volume</button><button data-m="error">error rate</button></div><span class="tfx-clk-tz">device-local · ${esc((window.APIN && APIN.time) ? APIN.time.zone : "")}</span></div>
       <div class="tfx-clk-grid"><div id="tfx-clock"></div><div id="tfx-clkdetail" class="tfx-clk-detail"></div></div>`;
    let mode = "volume";
    const overview = () => {
      const c = DATA.clock || {}; const quiet = (c.hours || []).filter(h => h.n > 0).reduce((m, h) => h.n < (m ? m.n : 1e9) ? h : m, null);
      const biz = (c.hours || []).slice(9, 17).reduce((s, h) => s + h.n, 0); const tot = (c.hours || []).reduce((s, h) => s + h.n, 0);
      $("tfx-clkdetail").innerHTML =
        `<div class="tfx-facts" style="flex-direction:column;gap:7px"><span>busiest hour <b>${c.busiest_h != null ? String(c.busiest_h).padStart(2, "0") + ":00" : "—"}</b></span><span>quietest active <b>${quiet ? String(quiet.h).padStart(2, "0") + ":00" : "—"}</b></span><span>business hrs (9–17) <b>${tot ? Math.round(100 * biz / tot) : 0}%</b></span><span>total <b>${fmtNum(tot)}</b></span></div><p class="tfx-clk-hint">click an hour wedge for its 60-minute breakdown.</p>`;
    };
    const detailHour = (hh) => {
      const h = (DATA.clock.hours || [])[hh]; if (!h) return;
      // 60-min breakdown for that hour (today/yesterday) from the requests log
      const win = _hourWindow(hh);
      $("tfx-clkdetail").innerHTML =
        `<div class="tfx-hour-head"><b>${String(hh).padStart(2, "0")}:00 – ${String((hh + 1) % 24).padStart(2, "0")}:00</b><button class="tfx-back" id="tfx-clk-back">← all hours</button></div>
         <div class="tfx-facts"><span>${fmtNum(h.n)} requests</span><span>${h.err_pct}% errors</span></div>
         ${_sec("per-minute")}<div id="tfx-min" class="tf-empty">loading…</div>
         <button class="tfx-drill-btn" id="tfx-clk-drill">open request container →</button>`;
      $("tfx-clk-back").addEventListener("click", overview);
      $("tfx-clk-drill").addEventListener("click", () => openRequestContainer({ sinceMs: win.sinceMs, untilMs: win.untilMs, label: String(hh).padStart(2, "0") + ":00" }));
      api(`/api/account/keys/${encodeURIComponent(PID)}/requests?since=${encodeURIComponent(_utcStr(win.sinceMs))}&until=${encodeURIComponent(_utcStr(win.untilMs))}&limit=200`).then(({ body }) => {
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
    const paint = () => {
      const by = DATA.bytes || { buckets: [], by_endpoint: [] };
      const epOutMax = Math.max(1, ...(by.by_endpoint || []).map(e => e.bout));
      const epInMax = Math.max(1, ...(by.by_endpoint || []).map(e => e.bin));
      panel.innerHTML =
        `<div id="tfx-bytes"></div>
         <div class="tfx-facts"><span style="color:${_C().accent}">▲ in <b>${fmtBytes(by.total_in)}</b> · ${fmtBytes(by.avg_in)}/req</span><span style="color:${_C().ok}">▼ out <b>${fmtBytes(by.total_out)}</b> · ${fmtBytes(by.avg_out)}/req</span>${_ratioStr(by.total_in, by.total_out) ? `<span><b>${_ratioStr(by.total_in, by.total_out)}</b></span>` : ""}</div>
         ${_sec("out by endpoint")}${_hbars((by.by_endpoint || []).map(e => ({ label: e.path, v: e.bout, vl: fmtBytes(e.bout) })), _C().ok)}
         ${_sec("in by endpoint")}${_hbars((by.by_endpoint || []).map(e => ({ label: e.path, v: e.bin, vl: fmtBytes(e.bin) })), _C().accent)}`;
      $("tfx-bytes").innerHTML = by.buckets && by.buckets.length ? bytesSvg(by.buckets) : `<div class="tf-empty">no payload data</div>`;
      const hit = $("tf-bytes-hit");
      if (hit) { hit.addEventListener("mousemove", (e) => { const r = hit.getBoundingClientRect(); const i = Math.round(((e.clientX - r.left) / r.width) * (by.buckets.length - 1)); const b = by.buckets[Math.max(0, Math.min(by.buckets.length - 1, i))]; if (b) tip(true, e.clientX, e.clientY, `<b>${esc(b.label)}</b><br>in ${fmtBytes(b.bin)} · out ${fmtBytes(b.bout)}`); }); hit.addEventListener("mouseleave", () => tip(false)); }
    };
    paint(); _openExpand = { kind: "bytes", update: paint };
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
  }
  function deactivate() {
    _active = false;
    stopSSE();
    if (_clockTick) { clearInterval(_clockTick); _clockTick = null; }
    if (_liveTimer) { clearTimeout(_liveTimer); _liveTimer = null; }
    if (_rerenderRaf) { cancelAnimationFrame(_rerenderRaf); _rerenderRaf = null; }
  }

  window.APIN = window.APIN || {};
  window.APIN.keyTraffic = { activate, deactivate };
})();
