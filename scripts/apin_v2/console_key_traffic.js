// 9.N.T · Per-key TRAFFIC tab — widget module.
//
// Owns #pane-traffic. One /traffic fetch feeds five widgets: hero (segmented
// status-stacked blocks over time), stats rail (slot-machine numbers), GitHub
// activity calendar, traffic clock (polar local-hour dial with a self-driving
// hand), and a bytes-flow mirror. Everything updates per-SSE-event in real
// time (cards AND open expanded states) — no manual refresh, no poll lag.
// Times are device-local (APIN.time / tz_off). Paper-ink SVG, hand-drawn
// sprite icons, no emoji.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtNum = (n) => n == null ? "—" : Number(n).toLocaleString();
  const fmtBytes = (b) => b == null ? "—" : b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : b < 1073741824 ? (b / 1048576).toFixed(1) + " MB" : (b / 1073741824).toFixed(2) + " GB";
  const TZOFF = (window.APIN && APIN.time) ? APIN.time.offsetMin() : -new Date().getTimezoneOffset();
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
  const GRAN_MS = { hour: 3600e3, day: 86400e3, week: 7 * 86400e3, month: 31 * 86400e3 };

  // ── state ───────────────────────────────────────────────────────────────
  let PID = null, DATA = null, _active = false, _wired = false;
  let GRAN = (function () { try { return sessionStorage.getItem("tf_gran") || "hour"; } catch (_) { return "hour"; } })();
  let LIVE = true, _refreshSeq = 0, _liveTimer = null, _pollTimer = null;
  let _clockTick = null, _rerenderRaf = null, _dirty = {};
  let _openExpand = null;   // {kind, update} when a lightbox is open

  // ════════════════════ DATA FETCH ════════════════════════════════════════
  async function refresh() {
    if (!PID) return;
    const myGran = GRAN, mySeq = ++_refreshSeq;
    const { ok, body } = await api(`/api/account/keys/${encodeURIComponent(PID)}/traffic?granularity=${myGran}&tz_off=${TZOFF}`);
    if (mySeq !== _refreshSeq || myGran !== GRAN) return;   // superseded
    if (!ok || !body || body.ok === false) return;
    DATA = body.data || body;
    const stack = $("tf-stack"); if (stack) stack.classList.remove("ov-loading");
    renderHero(true); renderStats(); renderCalendar(true); renderClock(true); renderBytes(true);
  }

  // ════════════════════ ① HERO — segmented stacked blocks ═════════════════
  const HERO_W = 720, HERO_H = 188, MAXBLOCKS = 14, BLK_H = 9, BLK_GAP = 3;
  function heroSvg(buckets, max) {
    const col = _C(), n = buckets.length || 1;
    const slotW = HERO_W / n, bw = Math.max(6, Math.min(26, slotW - 6));
    const baseY = HERO_H - 16;
    let bars = "", labels = "", hits = "";
    buckets.forEach((b, i) => {
      const cx = i * slotW + slotW / 2, x = cx - bw / 2;
      const blocks = b.total > 0 ? Math.max(1, Math.round(b.total / (max || 1) * MAXBLOCKS)) : 0;
      const g = Math.round(blocks * (b.n2 / (b.total || 1)));
      const a = Math.round(blocks * (b.n4 / (b.total || 1)));
      const r = Math.max(0, blocks - g - a);
      let y = baseY;
      for (let k = 0; k < blocks; k++) {
        y -= BLK_H; const color = k < g ? col.ok : k < g + a ? col.amber : col.danger;
        bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${BLK_H - BLK_GAP}" rx="2" fill="${color}"/>`;
        y -= BLK_GAP;
      }
      if (i % Math.ceil(n / 8) === 0)
        labels += `<text x="${cx.toFixed(1)}" y="${HERO_H - 2}" text-anchor="middle" style="font:9px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${esc(b.label)}</text>`;
      hits += `<rect class="tf-bar" data-i="${i}" x="${(i * slotW).toFixed(1)}" y="0" width="${slotW.toFixed(1)}" height="${baseY}" fill="transparent"/>`;
    });
    return `<svg class="tf-hero-svg" viewBox="0 0 ${HERO_W} ${HERO_H}" preserveAspectRatio="none" style="height:${HERO_H}px">
      <line x1="0" y1="${baseY}" x2="${HERO_W}" y2="${baseY}" stroke="var(--paper-edge)" stroke-width="1"/>${bars}${labels}${hits}</svg>`;
  }
  function _wireHero(host, intro) {
    const svg = host.querySelector("svg"); if (!svg) return;
    svg.querySelectorAll(".tf-bar").forEach(h => {
      h.addEventListener("mousemove", (e) => {
        const b = DATA.hero.buckets[+h.getAttribute("data-i")]; if (!b) return;
        const errp = b.total ? Math.round(100 * (b.n4 + b.n5) / b.total) : 0;
        tip(true, e.clientX, e.clientY, `${esc(b.label)} · ${fmtNum(b.total)} req<br>2xx ${b.n2} · 4xx ${b.n4} · 5xx ${b.n5} · ${errp}% err`);
      });
      h.addEventListener("mouseleave", () => tip(false));
      h.addEventListener("click", () => {
        const b = DATA.hero.buckets[+h.getAttribute("data-i")];
        if (b) _drill(b.t_ms, b.t_ms + (GRAN_MS[GRAN] || 3600e3));
      });
    });
    if (intro && window.APIN && APIN.fx) {
      svg.querySelectorAll("rect[rx]").forEach((r, k) => {
        try { r.animate([{ transform: "scaleY(0)", transformOrigin: "center bottom" }, { transform: "scaleY(1)", transformOrigin: "center bottom" }], { duration: 320, delay: Math.min(600, k * 4), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {}
      });
    }
  }
  function renderHero(intro) {
    const host = $("tf-hero-body"); if (!host || !DATA) return;
    const h = DATA.hero || { buckets: [], max: 0 };
    const aux = $("tf-hero-aux"); if (aux) aux.textContent = GRAN === "hour" ? "last 24h" : GRAN === "day" ? "last 30d" : GRAN === "week" ? "last 12w" : "last 12mo";
    if (!h.buckets.length || h.max === 0) { host.innerHTML = `<div class="tf-empty">No traffic in this range yet.</div>`; return; }
    host.innerHTML = heroSvg(h.buckets, h.max);
    _wireHero(host, intro);
  }

  // ════════════════════ ② STATS RAIL ══════════════════════════════════════
  function renderStats() {
    const host = $("tf-stats"); if (!host || !DATA) return;
    const s = DATA.stats || {};
    host.innerHTML = [
      ["total", "requests", fmtNum(s.total), false],
      ["peak", "peak / min", fmtNum(s.peak_per_min), false],
      ["busy", "busiest " + (GRAN === "hour" ? "hour" : "bucket"), `${esc(s.busiest_label || "—")} (${fmtNum(s.busiest_count || 0)})`, true],
      ["err", "error rate", (s.error_pct != null ? s.error_pct : 0) + "%", false],
    ].map(([k, lbl, val, click]) =>
      `<div class="tf-stat${click ? " clickable" : ""}" data-stat="${k}"><span class="tf-stat-val" data-num>${esc(val)}</span><span class="tf-stat-lbl">${esc(lbl)}</span></div>`).join("");
    // slot-machine the pure integers
    slot(host.querySelector('.tf-stat[data-stat="total"] [data-num]'), s.total || 0);
    slot(host.querySelector('.tf-stat[data-stat="peak"] [data-num]'), s.peak_per_min || 0);
    const busy = host.querySelector('.tf-stat[data-stat="busy"]');
    if (busy) busy.addEventListener("click", () => { _flashHeroBar(s.busiest_label); });
  }
  function _flashHeroBar(label) {
    if (!DATA) return; const i = (DATA.hero.buckets || []).findIndex(b => b.label === label);
    const svg = $("tf-hero-body") && $("tf-hero-body").querySelector("svg");
    if (i < 0 || !svg) return;
    const hit = svg.querySelector(`.tf-bar[data-i="${i}"]`);
    if (hit && window.APIN && APIN.fx) APIN.fx.pulse(hit.closest(".ov-card") || svg);
  }

  // ════════════════════ ③ CALENDAR — GitHub heatmap ═══════════════════════
  function _calGrid(days, weeks) {
    // build [col][row] grid ending today (local); rows Mon(0)..Sun(6)
    const byDate = {}; (days || []).forEach(d => byDate[d.date] = d);
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const cells = [];
    const start = new Date(today); start.setDate(start.getDate() - (weeks * 7 - 1));
    // align start to Monday
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
    return { cells, max };
  }
  function calSvg(days, weeks, mode) {
    const col = _C(), g = _calGrid(days, weeks);
    const cell = 13, gap = 3, ox = 4, oy = 4;
    const ncols = g.cells.reduce((m, x) => Math.max(m, x.c), 0) + 1;
    const W = ox + ncols * (cell + gap), H = oy + 7 * (cell + gap) + 14;
    let rects = "";
    const todayKey = (() => { const t = new Date(); return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`; })();
    g.cells.forEach(c => {
      const x = ox + c.c * (cell + gap), y = oy + c.r * (cell + gap);
      let fill = "rgba(120,110,90,.10)";
      if (c.n > 0) {
        if (mode === "error") {
          const er = c.e / c.n; fill = er === 0 ? col.ok : er < 0.1 ? col.amber : col.danger;
        } else {
          const t = Math.min(1, c.n / (g.max || 1)); fill = col.accent;
          rects += `<rect class="tf-cal-cell" data-key="${c.key}" data-ms="${c.dt}" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="3" fill="${fill}" fill-opacity="${(0.2 + 0.8 * t).toFixed(2)}"${c.key === todayKey ? ' stroke="var(--ink)" stroke-width="1.2"' : ''}><title>${esc(c.label)} · ${c.n} req · ${c.n ? Math.round(100 * c.e / c.n) : 0}% err</title></rect>`;
          return;
        }
      }
      rects += `<rect class="tf-cal-cell" data-key="${c.key}" data-ms="${c.dt}" x="${x}" y="${y}" width="${cell}" height="${cell}" rx="3" fill="${fill}"${c.key === todayKey ? ' stroke="var(--ink)" stroke-width="1.2"' : ''}><title>${esc(c.label)} · ${c.n} req · ${c.n ? Math.round(100 * c.e / c.n) : 0}% err</title></rect>`;
    });
    const labels = ["Mon", "", "Wed", "", "Fri", "", ""].map((d, r) => d ? `<text x="0" y="${oy + r * (cell + gap) + cell - 2}" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${d}</text>` : "").join("");
    return `<svg class="tf-cal-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMinYMid meet" style="max-height:${H}px">${rects}</svg>`;
  }
  function _wireCal(host, intro) {
    host.querySelectorAll(".tf-cal-cell").forEach((c, k) => {
      c.addEventListener("mousemove", (e) => { const t = c.querySelector("title"); tip(true, e.clientX, e.clientY, esc(t ? t.textContent : "")); });
      c.addEventListener("mouseleave", () => tip(false));
      c.addEventListener("click", () => { const ms = +c.getAttribute("data-ms"); if (ms) _drill(ms, ms + 86400e3); });
    });
    if (intro && window.APIN && APIN.fx) host.querySelectorAll(".tf-cal-cell").forEach(c => {
      try { c.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 280, delay: Math.min(700, (+c.getAttribute("data-ms")) % 700), easing: "ease", fill: "backwards" }); } catch (_) {}
    });
  }
  function renderCalendar(intro, mode, weeks) {
    const host = $("tf-cal-body"); if (!host || !DATA) return;
    host.innerHTML = calSvg((DATA.calendar || {}).days, weeks || 17, mode || "volume");
    _wireCal(host, intro);
  }

  // ════════════════════ ④ TRAFFIC CLOCK — polar local-hour dial ═══════════
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
      const m = hand.getAttribute("x1"); const cx = m, cy = hand.getAttribute("y1");
      hand.style.transition = "transform 1s linear";
      hand.setAttribute("transform", `rotate(${(frac * 360).toFixed(2)} ${cx} ${cy})`);
    };
    move(); _clockTick = setInterval(move, 1000);
  }
  function _wireClock(host, intro) {
    const svg = host.querySelector("svg"); if (!svg) return;
    svg.querySelectorAll(".tf-wedge").forEach(w => {
      w.addEventListener("mousemove", (e) => {
        const h = (DATA.clock.hours || [])[+w.getAttribute("data-h")]; if (!h) return;
        tip(true, e.clientX, e.clientY, `${String(h.h).padStart(2, "0")}:00 · ${fmtNum(h.n)} req · ${h.err_pct}% err`);
        svg.querySelectorAll(".tf-wedge").forEach(o => o.style.opacity = o === w ? "1" : "0.4");
      });
      w.addEventListener("mouseleave", () => { tip(false); svg.querySelectorAll(".tf-wedge").forEach(o => o.style.opacity = "1"); });
      w.addEventListener("click", () => { const hh = +w.getAttribute("data-h"); _drillHour(hh); });
    });
    _startClockHand();
    if (intro) svg.querySelectorAll(".tf-wedge").forEach((w, i) => { try { w.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 360, delay: Math.min(820, i * 32), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
  }
  function renderClock(intro, mode) {
    const host = $("tf-clock-body"); if (!host || !DATA) return;
    const c = DATA.clock || { hours: [], max: 0 };
    if (!c.max) { host.innerHTML = `<div class="tf-empty">Rhythm emerges as traffic accrues.</div>`; return; }
    // In-tab: colour by VOLUME (clean green rhythm). Error-rate hotspots are a
    // toggle in the expanded view, so the card never reads as all-red alarm.
    host.innerHTML = `<div class="tf-clock-wrap">${clockSvg(c, 280, mode || "volume")}</div>`;
    _wireClock(host, intro);
  }

  // ════════════════════ ⑤ BYTES FLOW — mirror area ════════════════════════
  const BY_W = 720, BY_H = 150;
  function bytesSvg(buckets) {
    const col = _C(), n = buckets.length || 1, mid = BY_H / 2;
    const maxIn = Math.max(1, ...buckets.map(b => b.bin)), maxOut = Math.max(1, ...buckets.map(b => b.bout));
    const X = (i) => (n <= 1 ? BY_W / 2 : (i / (n - 1)) * BY_W);
    const upY = (v) => mid - (v / maxIn) * (mid - 6), dnY = (v) => mid + (v / maxOut) * (mid - 6);
    const inLine = buckets.map((b, i) => `${X(i).toFixed(1)},${upY(b.bin).toFixed(1)}`).join(" ");
    const outLine = buckets.map((b, i) => `${X(i).toFixed(1)},${dnY(b.bout).toFixed(1)}`).join(" ");
    const inArea = `0,${mid} ` + inLine + ` ${BY_W},${mid}`, outArea = `0,${mid} ` + outLine + ` ${BY_W},${mid}`;
    return `<svg class="tf-bytes-svg" viewBox="0 0 ${BY_W} ${BY_H}" preserveAspectRatio="none" style="height:${BY_H}px">
      <defs><linearGradient id="tfIn" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="${col.accent}" stop-opacity="0"/><stop offset="1" stop-color="${col.accent}" stop-opacity=".32"/></linearGradient>
      <linearGradient id="tfOut" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${col.ok}" stop-opacity="0"/><stop offset="1" stop-color="${col.ok}" stop-opacity=".32"/></linearGradient></defs>
      <polygon points="${inArea}" fill="url(#tfIn)"/><polyline points="${inLine}" fill="none" stroke="${col.accent}" stroke-width="1.5"/>
      <polygon points="${outArea}" fill="url(#tfOut)"/><polyline points="${outLine}" fill="none" stroke="${col.ok}" stroke-width="1.5"/>
      <line x1="0" y1="${mid}" x2="${BY_W}" y2="${mid}" stroke="var(--paper-edge)" stroke-width="1"/>
      <rect id="tf-bytes-hit" x="0" y="0" width="${BY_W}" height="${BY_H}" fill="transparent"/></svg>`;
  }
  function renderBytes(intro) {
    const host = $("tf-bytes-body"); if (!host || !DATA) return;
    const by = DATA.bytes || { buckets: [] };
    if (!by.buckets.length || (by.total_in === 0 && by.total_out === 0)) { host.innerHTML = `<div class="tf-empty">No payload data in this range.</div>`; return; }
    host.innerHTML = bytesSvg(by.buckets) +
      `<div class="tf-bytes-legend" style="display:flex;justify-content:space-between;font:11px 'JetBrains Mono',monospace;color:var(--ink-soft);padding:4px 6px 0">
        <span style="color:${_C().accent}">▲ in · ${fmtBytes(by.avg_in)}/req</span>
        <span style="color:${_C().ok}">▼ out · ${fmtBytes(by.avg_out)}/req</span></div>`;
    const hit = $("tf-bytes-hit");
    if (hit) {
      hit.addEventListener("mousemove", (e) => {
        const r = hit.getBoundingClientRect(); const i = Math.round(((e.clientX - r.left) / r.width) * (by.buckets.length - 1));
        const b = by.buckets[Math.max(0, Math.min(by.buckets.length - 1, i))]; if (!b) return;
        tip(true, e.clientX, e.clientY, `${esc(b.label)}<br>in ${fmtBytes(b.bin)} · out ${fmtBytes(b.bout)}`);
      });
      hit.addEventListener("mouseleave", () => tip(false));
    }
    if (intro && window.APIN && APIN.fx) host.querySelectorAll("polyline,polygon").forEach(p => { try { p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 420, easing: "ease" }); } catch (_) {} });
  }

  // ════════════════════ DRILL → Requests tab (time-filtered) ══════════════
  function _drill(sinceMs, untilMs) {
    if (window.APIN && APIN.keyDetail && typeof APIN.keyDetail.filterRequests === "function") {
      APIN.keyDetail.filterRequests(_utcStr(sinceMs), _utcStr(untilMs));
    } else { location.hash = "#requests"; }
  }
  function _drillHour(hh) {
    // most-recent local day's that-hour window → UTC
    const d = new Date(); d.setHours(hh, 0, 0, 0); if (d > new Date()) d.setDate(d.getDate() - 1);
    _drill(d.getTime(), d.getTime() + 3600e3);
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
    // mutate in-memory DATA so the visible change is instant (sub-second).
    const sb = _statusBucket(ev.status_code), isErr = sb !== "n2";
    // hero: the current (last) bucket
    if (DATA.hero && DATA.hero.buckets.length) {
      const last = DATA.hero.buckets[DATA.hero.buckets.length - 1];
      last[sb] = (last[sb] || 0) + 1; last.total = last.n2 + last.n4 + last.n5;
      if (last.total > DATA.hero.max) DATA.hero.max = last.total;
    }
    // stats
    if (DATA.stats) {
      DATA.stats.total = (DATA.stats.total || 0) + 1;
      const errs = Math.round((DATA.stats.error_pct || 0) / 100 * (DATA.stats.total - 1)) + (isErr ? 1 : 0);
      DATA.stats.error_pct = DATA.stats.total ? Math.round(1000 * errs / DATA.stats.total) / 10 : 0;
    }
    // clock: current local hour
    if (DATA.clock) {
      const lh = new Date().getHours(); const hr = (DATA.clock.hours || [])[lh];
      if (hr) { hr.n += 1; if (isErr) hr.e += 1; hr.err_pct = hr.n ? Math.round(1000 * hr.e / hr.n) / 10 : 0; if (hr.n > DATA.clock.max) DATA.clock.max = hr.n; }
    }
    // calendar: today
    if (DATA.calendar) {
      const t = new Date(); const key = `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`;
      let day = (DATA.calendar.days || []).find(d => d.date === key);
      if (!day) { day = { date: key, n: 0, e: 0 }; (DATA.calendar.days = DATA.calendar.days || []).push(day); }
      day.n += 1; if (isErr) day.e += 1; if (day.n > DATA.calendar.max) DATA.calendar.max = day.n;
    }
    // bytes: current bucket
    if (DATA.bytes && DATA.bytes.buckets.length) {
      const lb = DATA.bytes.buckets[DATA.bytes.buckets.length - 1];
      lb.bin += (+ev.bytes_in || 0); lb.bout += (+ev.bytes_out || 0);
      DATA.bytes.total_in += (+ev.bytes_in || 0); DATA.bytes.total_out += (+ev.bytes_out || 0);
      const tr = DATA.stats ? DATA.stats.total : 0;
      DATA.bytes.avg_in = tr ? Math.round(DATA.bytes.total_in / tr) : 0;
      DATA.bytes.avg_out = tr ? Math.round(DATA.bytes.total_out / tr) : 0;
    }
    _dirty = { hero: 1, stats: 1, clock: 1, cal: 1, bytes: 1 };
    if (!_rerenderRaf) _rerenderRaf = requestAnimationFrame(_flushRender);
    _scheduleReconcile();
  }
  function _flushRender() {
    _rerenderRaf = null;
    if (_dirty.hero) renderHero(false);
    if (_dirty.stats) renderStats();
    if (_dirty.clock) renderClock(false);
    if (_dirty.cal) renderCalendar(false);
    if (_dirty.bytes) renderBytes(false);
    _dirty = {};
    // keep an open expanded view live too; drop the subscription once closed.
    if (_openExpand && _openExpand.update) {
      const open = !(window.APIN && APIN.lightbox && APIN.lightbox.isOpen) || APIN.lightbox.isOpen();
      if (open) { try { _openExpand.update(); } catch (_) {} }
      else _openExpand = null;
    }
  }
  function _scheduleReconcile() {
    if (_liveTimer) clearTimeout(_liveTimer);
    _liveTimer = setTimeout(() => { if (LIVE && _active) refresh(); }, 1500);
  }

  // ════════════════════ EXPANDED STATES ═══════════════════════════════════
  const _kxSec = (t) => `<div style="font:500 italic 11px 'Fraunces',serif;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-soft);margin:16px 0 9px;padding-bottom:6px;border-bottom:1px solid var(--paper-edge)">${esc(t)}</div>`;

  function expandHero(panel) {
    const col = _C();
    panel.innerHTML =
      `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px;font:12px 'JetBrains Mono',monospace">
        <div class="ov-range" id="tfx-gran">${["hour", "day", "week", "month"].map(g => `<button data-g="${g}"${g === GRAN ? ' aria-pressed="true"' : ""}>${g[0].toUpperCase() + g.slice(1)}</button>`).join("")}</div>
        <div class="ov-range" id="tfx-ov"><button data-ov="none" aria-pressed="true">bars</button><button data-ov="err">+ error %</button></div>
      </div>
      <div id="tfx-hero"></div>
      <div style="display:flex;gap:16px;font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:4px"><span><i class="rdot s-2xx"></i> 2xx</span><span><i class="rdot s-4xx"></i> 4xx</span><span><i class="rdot s-5xx"></i> 5xx</span><span style="margin-left:auto;font-style:italic">click a bar → requests at that time</span></div>
      <div id="tfx-hero-stats" style="font:12px/1.7 'JetBrains Mono',monospace;color:var(--ink-soft);margin-top:8px"></div>`;
    let ov = "none";
    const paint = () => {
      const host = $("tfx-hero"); const h = DATA.hero || { buckets: [], max: 0 };
      host.innerHTML = heroSvg(h.buckets, h.max);
      _wireHero(host, false);
      if (ov === "err") _overlayErr(host, h.buckets);
      const s = DATA.stats || {};
      $("tfx-hero-stats").innerHTML = `total ${fmtNum(s.total)} · busiest ${esc(s.busiest_label)} (${fmtNum(s.busiest_count)}) · peak ${fmtNum(s.peak_per_min)}/min · err ${s.error_pct}%`;
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
  function _overlayErr(host, buckets) {
    const svg = host.querySelector("svg"); if (!svg) return; const n = buckets.length || 1;
    const X = (i) => (i + 0.5) * (HERO_W / n); const baseY = HERO_H - 16;
    const pts = buckets.map((b, i) => { const er = b.total ? (b.n4 + b.n5) / b.total : 0; return `${X(i).toFixed(1)},${(baseY - er * (baseY - 8)).toFixed(1)}`; }).join(" ");
    svg.insertAdjacentHTML("beforeend", `<polyline points="${pts}" fill="none" stroke="${_C().danger}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.85"/>`);
  }

  function expandCalendar(panel) {
    panel.innerHTML =
      `<div class="ov-range" id="tfx-calmode" style="margin-bottom:10px"><button data-m="volume" aria-pressed="true">volume</button><button data-m="error">error rate</button></div>
       <div id="tfx-cal"></div>
       <div id="tfx-calstats" style="font:12px/1.7 'JetBrains Mono',monospace;color:var(--ink-soft);margin-top:12px"></div>
       ${_kxSec("weekday profile")}<div id="tfx-wkprofile"></div>`;
    let mode = "volume";
    const paint = () => {
      $("tfx-cal").innerHTML = calSvg((DATA.calendar || {}).days, 52, mode);
      _wireCal($("tfx-cal"), false);
      // streaks + busiest + weekday profile
      const days = ((DATA.calendar || {}).days || []).slice().sort((a, b) => a.date < b.date ? -1 : 1);
      const active = days.filter(d => d.n > 0);
      const busiest = active.reduce((m, d) => d.n > (m ? m.n : -1) ? d : m, null);
      // current streak (consecutive days up to today with traffic)
      let streak = 0; const today = new Date(); today.setHours(0, 0, 0, 0);
      for (let i = 0; ; i++) { const dt = new Date(today); dt.setDate(dt.getDate() - i); const k = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`; const r = days.find(d => d.date === k); if (r && r.n > 0) streak++; else break; }
      $("tfx-calstats").innerHTML = `${active.length} active day(s) · current streak ${streak}d · busiest ${busiest ? esc(busiest.date) + " (" + fmtNum(busiest.n) + ")" : "—"}`;
      // weekday profile from clock-independent: aggregate days by weekday
      const wk = [0, 0, 0, 0, 0, 0, 0]; const names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
      days.forEach(d => { const dt = new Date(d.date + "T00:00:00"); wk[(dt.getDay() + 6) % 7] += d.n; });
      const wkmax = Math.max(1, ...wk);
      $("tfx-wkprofile").innerHTML = `<div style="display:flex;gap:6px;align-items:flex-end;height:60px">${wk.map((v, i) => `<div style="flex:1;text-align:center"><div style="height:${Math.round(v / wkmax * 48)}px;background:var(--c-accent,#52b788);border-radius:3px 3px 0 0"></div><div style="font:9px 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:3px">${names[i]}</div></div>`).join("")}</div>`;
    };
    paint();
    panel.querySelectorAll("#tfx-calmode button").forEach(b => b.addEventListener("click", () => { panel.querySelectorAll("#tfx-calmode button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); mode = b.getAttribute("data-m"); paint(); }));
    _openExpand = { kind: "calendar", update: paint };
  }

  function expandClock(panel) {
    panel.innerHTML =
      `<div class="ov-range" id="tfx-clkmode" style="margin-bottom:8px"><button data-m="error" aria-pressed="true">error rate</button><button data-m="volume">volume</button></div>
       <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap"><div id="tfx-clock"></div><div id="tfx-clkdetail" style="flex:1;min-width:200px;font:12px/1.8 'JetBrains Mono',monospace;color:var(--ink-soft)"></div></div>`;
    let mode = "error";
    const paint = () => {
      $("tfx-clock").innerHTML = clockSvg(DATA.clock || { hours: [], max: 0 }, 320, mode);
      const svg = $("tfx-clock").querySelector("svg");
      svg && svg.querySelectorAll(".tf-wedge").forEach(w => {
        w.addEventListener("mousemove", (e) => { const h = (DATA.clock.hours || [])[+w.getAttribute("data-h")]; if (h) tip(true, e.clientX, e.clientY, `${String(h.h).padStart(2, "0")}:00 · ${fmtNum(h.n)} · ${h.err_pct}% err`); });
        w.addEventListener("mouseleave", () => tip(false));
        w.addEventListener("click", () => showDetail(+w.getAttribute("data-h")));
      });
      _startClockHand();
      const c = DATA.clock || {}; const quiet = (c.hours || []).filter(h => h.n > 0).reduce((m, h) => h.n < (m ? m.n : 1e9) ? h : m, null);
      const biz = (c.hours || []).slice(9, 17).reduce((s, h) => s + h.n, 0); const tot = (c.hours || []).reduce((s, h) => s + h.n, 0);
      $("tfx-clkdetail").innerHTML = `busiest ${c.busiest_h != null ? String(c.busiest_h).padStart(2, "0") + ":00" : "—"}<br>quietest ${quiet ? String(quiet.h).padStart(2, "0") + ":00" : "—"}<br>${tot ? Math.round(100 * biz / tot) : 0}% in business hrs (9–17)<br><span style="font-style:italic;color:var(--ink-mute)">click a wedge for detail</span>`;
    };
    const showDetail = (hh) => {
      const h = (DATA.clock.hours || [])[hh]; if (!h) return;
      $("tfx-clkdetail").innerHTML = `<b style="font:600 14px 'Fraunces',serif;color:var(--ink)">${String(hh).padStart(2, "0")}:00–${String((hh + 1) % 24).padStart(2, "0")}:00</b><br>${fmtNum(h.n)} requests · ${h.err_pct}% errors<br><button class="btn-link" id="tfx-clkdrill" style="background:none;border:none;color:var(--c-accent,#52b788);cursor:pointer;padding:6px 0;font:inherit">view in Requests →</button>`;
      const d = $("tfx-clkdrill"); if (d) d.addEventListener("click", () => _drillHour(hh));
    };
    paint();
    panel.querySelectorAll("#tfx-clkmode button").forEach(b => b.addEventListener("click", () => { panel.querySelectorAll("#tfx-clkmode button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); mode = b.getAttribute("data-m"); paint(); }));
    _openExpand = { kind: "clock", update: paint };
  }

  function expandBytes(panel) {
    const paint = () => {
      const by = DATA.bytes || { buckets: [], by_endpoint: [] };
      const epMax = Math.max(1, ...(by.by_endpoint || []).map(e => e.bout));
      panel.innerHTML =
        bytesSvg(by.buckets) +
        `<div style="display:flex;gap:24px;font:12px 'JetBrains Mono',monospace;color:var(--ink-soft);margin-top:8px">
          <span style="color:${_C().accent}">▲ in total ${fmtBytes(by.total_in)} · ${fmtBytes(by.avg_in)}/req</span>
          <span style="color:${_C().ok}">▼ out total ${fmtBytes(by.total_out)} · ${fmtBytes(by.avg_out)}/req</span>
          ${by.ratio != null ? `<span>out:in ${by.ratio}×</span>` : ""}</div>
        ${_kxSec("bytes by endpoint")}
        ${(by.by_endpoint || []).map(e => `<div style="display:grid;grid-template-columns:150px 1fr 78px;gap:10px;align-items:center;margin:5px 0;font:11.5px 'JetBrains Mono',monospace"><span style="color:var(--ink-soft);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(e.path)}">${esc(e.path)}</span><span style="height:13px;background:rgba(120,110,90,.16);border-radius:3px;overflow:hidden;display:block"><i style="display:block;height:100%;width:${(e.bout / epMax * 100).toFixed(1)}%;background:${_C().ok};border-radius:3px"></i></span><span style="text-align:right;color:var(--ink)">${fmtBytes(e.bout)}</span></div>`).join("") || `<div style="font:italic 12px 'Fraunces',serif;color:var(--ink-mute)">no payload data</div>`}`;
    };
    paint();
    _openExpand = { kind: "bytes", update: paint };
  }

  const EXPANDERS = { hero: { title: "Requests over time", build: expandHero },
                      calendar: { title: "Activity calendar", build: expandCalendar },
                      clock: { title: "Traffic rhythm", build: expandClock },
                      bytes: { title: "Data transfer", build: expandBytes } };

  // ════════════════════ CONTROLS / WIRING ═════════════════════════════════
  function syncGranButtons() {
    document.querySelectorAll("#tf-gran button").forEach(b => {
      if (b.getAttribute("data-gran") === GRAN) b.setAttribute("aria-pressed", "true");
      else b.removeAttribute("aria-pressed");
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
      live.querySelector(".ov-live-label").textContent = LIVE ? "live" : "paused";
      if (LIVE) startSSE(); else stopSSE();
    });
    const rf = $("tf-refresh");
    if (rf) rf.addEventListener("click", () => { rf.classList.add("is-spinning"); setTimeout(() => rf.classList.remove("is-spinning"), 760); refresh(); });
    document.querySelectorAll(".ov-expand[data-tfx]").forEach(btn => btn.addEventListener("click", () => {
      const ex = EXPANDERS[btn.getAttribute("data-tfx")];
      if (!ex || !window.APIN || !APIN.lightbox) return;
      const card = btn.closest(".ov-card");
      APIN.lightbox.open({ title: ex.title + (DATA && DATA.key ? " · " + (DATA.key.name || "") : ""), sourceEl: card,
        build: ex.build, onClose: () => { _openExpand = null; } });
    }));
    window.addEventListener("resize", () => { if (_active && DATA) { renderHero(false); renderClock(false); renderBytes(false); } });
  }

  // ════════════════════ ACTIVATE / DEACTIVATE ═════════════════════════════
  function activate(pid) {
    PID = pid || PID; _active = true;
    if (!$("tf-stack")) return;
    wire(); syncGranButtons();
    if (DATA) { renderHero(true); renderStats(); renderCalendar(true); renderClock(true); renderBytes(true); }  // instant from cache
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
