// 9.N.T19/20 · Per-key ENDPOINTS tab — widget module.
//
// Owns #pane-endpoints. One /endpoints fetch feeds four widgets in the
// "flow + proportion" archetype:
//   ① Endpoint treemap   — size = volume, tint = error rate
//   ② Method × status     — compact heat-matrix (rows = method, cols = status)
//   ③ Endpoint affinity   — call-sequence Sankey (what-follows-what)
//   ④ Sortable table      — n / p50 / p95 / err% / volume sparkline
// Range toggle (1h/24h/7d, persisted) cascades to all widgets. Live via SSE.
// Every widget drills into a filtered request list; each row → the shared
// full request drawer. Paper-ink SVG, custom tooltips, no emoji.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtNum = (n) => { n = +n || 0; return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); };
  const fmtBytes = (b) => b == null ? "—" : b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : b < 1073741824 ? (b / 1048576).toFixed(1) + " MB" : (b / 1073741824).toFixed(2) + " GB";
  const fmtMs = (ms) => ms == null ? "—" : ms < 1 ? "<1ms" : ms < 1000 ? Math.round(ms) + "ms" : (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + "s";
  const TZOFF = (window.APIN && APIN.time) ? APIN.time.offsetMin() : -new Date().getTimezoneOffset();
  const PID_META = document.querySelector('meta[name="key-public-id"]');
  const _pathTail = (p) => { p = String(p || "?"); const seg = p.split("/").filter(Boolean); return seg.length ? "/" + seg.slice(-2).join("/") : p; };
  const _short = (p) => { const seg = String(p || "?").split("/").filter(Boolean); return seg.length ? seg[seg.length - 1] : p; };
  // animated "waiting for data" placeholder — the hand-drawn leaf from the live tail
  let _emptyN = 0;
  function _emptyState(caption, sub) {
    const u = "ep-empty-" + (++_emptyN);   // unique ids so multiple instances don't collide
    return `<div class="ls-empty ep-empty">`
      + `<svg viewBox="0 0 120 140" class="ls-empty-leaf" aria-hidden="true">`
      + `<defs><filter id="${u}-w" x="-2%" y="-2%" width="104%" height="104%">`
      + `<feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="5" result="t"/>`
      + `<feDisplacementMap in="SourceGraphic" in2="t" scale="0.5" xChannelSelector="R" yChannelSelector="G"/></filter></defs>`
      + `<line x1="60" y1="135" x2="60" y2="110" stroke="var(--ink-soft,#6b6453)" stroke-width="1.5" stroke-linecap="round" filter="url(#${u}-w)"/>`
      + `<path d="M60 110 C 30 100, 18 60, 60 18 C 102 60, 90 100, 60 110 Z" fill="none" stroke="var(--ink,#1a1612)" stroke-width="1.4" stroke-linejoin="round" filter="url(#${u}-w)"/>`
      + `<path id="${u}-v" d="M60 108 L60 22" fill="none" stroke="var(--ink-soft,#6b6453)" stroke-width="0.7" filter="url(#${u}-w)"/>`
      + `<path d="M60 90 L40 75 M60 90 L80 75 M60 70 L36 55 M60 70 L84 55 M60 50 L42 40 M60 50 L78 40" fill="none" stroke="var(--ink-soft,#6b6453)" stroke-width="0.6" opacity="0.7" filter="url(#${u}-w)"/>`
      + `<circle r="2.4" fill="var(--c-ok,#2f6f3e)" opacity="0.85">`
      + `<animateMotion dur="3.5s" repeatCount="indefinite" rotate="auto"><mpath href="#${u}-v"/></animateMotion>`
      + `<animate attributeName="opacity" values="0;0.9;0.9;0" dur="3.5s" repeatCount="indefinite"/></circle>`
      + `</svg>`
      + `<div class="ls-empty-caption">${esc(caption)}</div>`
      + `<div class="ls-empty-sub">${esc(sub)}</div></div>`;
  }

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
             edge: g("--paper-edge", "#c7bca9"), accent: g("--c-accent", "#52b788"),
             paper: g("--paper", "#f4efe4") };
  }
  let _tip = null;
  function tip(show, x, y, html) {
    if (!_tip) { _tip = document.createElement("div"); _tip.className = "ov-tip"; document.body.appendChild(_tip); }
    if (!show) { _tip.classList.remove("show"); return; }
    _tip.innerHTML = html;
    _tip.style.left = Math.min(x + 12, window.innerWidth - 260) + "px";
    _tip.style.top = (y + 14) + "px";
    _tip.classList.add("show");
  }
  // hex colour lerp + 3-stop error ramp (green → amber → red)
  const _hex = (c) => { c = c.replace("#", ""); if (c.length === 3) c = c.split("").map(x => x + x).join(""); return [parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)]; };
  const _mix = (a, b, t) => { const A = _hex(a), B = _hex(b); return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`; };
  function _errColor(rate) {
    const col = _C(); const t = Math.min(1, (rate || 0) / 0.2);   // 20% err = full danger
    return t <= 0.5 ? _mix(col.ok, col.amber, t / 0.5) : _mix(col.amber, col.danger, (t - 0.5) / 0.5);
  }
  // soft pastel tint for treemap cells (reference aesthetic): light sage→sand→clay
  // by error rate, with a touch of depth by volume share. Ink text reads on these.
  const _lerp = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const SAGE = [171, 206, 168], SAND = [226, 201, 138], CLAY = [210, 144, 122];   // a notch richer (relates to the 3D jade→gold→ember)
  function _lightTint(rate, vol) {
    const t = Math.min(1, (rate || 0) / 0.2);
    let c = t <= 0.5 ? _lerp(SAGE, SAND, t / 0.5) : _lerp(SAND, CLAY, (t - 0.5) / 0.5);
    c = _lerp(c, [58, 48, 36], Math.min(0.16, (vol || 0) * 0.55));   // subtle depth for big cells
    return `rgb(${c.join(",")})`;
  }
  // fit a label to a pixel width at a given mono font size, ellipsizing
  function _ellip(s, px, fontpx) { s = String(s || ""); const max = Math.max(3, Math.floor(px / (fontpx * 0.6))); return s.length > max ? s.slice(0, max - 1) + "…" : s; }
  const _utcStr = (ms) => new Date(ms).toISOString().replace("T", " ").slice(0, 19);
  const _ago = (iso) => (window.APIN && APIN.time && iso) ? APIN.time.ago(iso) : "";
  const _methClass = (m) => { m = (m || "").toUpperCase(); return ["GET", "POST", "PUT", "PATCH", "DELETE"].includes(m) ? "meth-" + m : ""; };

  // ── state ────────────────────────────────────────────────────────────────
  const WIN_MS = { "1h": 3600e3, "24h": 86400e3, "7d": 7 * 86400e3 };
  const WIN_LABEL = { "1h": "last hour", "24h": "last 24 hours", "7d": "last 7 days" };
  let PID = null, DATA = null, _active = false, _wired = false, _introDone = false;
  let WIN = (function () { try { return sessionStorage.getItem("ep_win") || "24h"; } catch (_) { return "24h"; } })();
  let LIVE = true, _seq = 0, _liveTimer = null, _es = null, _dirty = false;
  let _sort = { col: "n", dir: -1 };   // dir: -1 desc, +1 asc
  // Living Endpoint Terrain (expanded modal) state
  let _terrRAF = null, _terrState = null, _terrMode = "terrain", _terrSel = null, _terrZoom = null, _terrReflowTimer = null;
  let _terr3d = null, _3dLoad = null;   // WebGL 3D instance + lazy-load promise
  let _terrCore = null, _coreStrata = "traffic";   // per-hill cutaway state
  // ── focus bus (cross-widget "living ecosystem" linking) ──
  let _focus = null, _pinned = false, _panelGn = null;   // genome handle in panel
  let _intelNumT = 0, _intelGnT = 0;   // live-update throttles (numbers vs genome rebuild)
  const TONE_OF = { Calm: "calm", Steady: "ok", Busy: "busy", Strained: "warn", Critical: "crit", Dormant: "mute" };
  const _terr3dResize = () => { if (_terr3d) try { _terr3d.resize(); } catch (_) {} };
  function _webglOK() { try { const c = document.createElement("canvas"); return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl"))); } catch (_) { return false; } }
  function _loadScript(src) { return new Promise((res, rej) => { if (document.querySelector(`script[data-eps="${src}"]`)) { res(); return; } const s = document.createElement("script"); s.src = src; s.dataset.eps = src; s.onload = () => res(); s.onerror = () => rej(new Error("load " + src)); document.head.appendChild(s); }); }
  function _ensureThree() {
    if (window.THREE && window.APIN.terrain3d) return Promise.resolve();
    if (_3dLoad) return _3dLoad;
    _3dLoad = _loadScript("/static/three.min.js?v=149").then(() => _loadScript("/static/console_key_terrain3d.js?v=9e5"));
    return _3dLoad;
  }
  function _setTerrHint(m) { const h = $("ept-mode-hint"); if (h) h.textContent = m === "flow" ? "request movement · sparks travel call → call · hover a hill" : m === "health" ? "error propagation · ember glows where requests fail" : m === "core" ? "x-ray · terrain turns translucent · click a hill to open its core" : "volume terrain · height = traffic · colour = errors · contours = elevation"; }

  // ════════════════════ DATA FETCH ════════════════════════════════════════
  async function refresh() {
    if (!PID) return;
    const myWin = WIN, mySeq = ++_seq;
    const { ok, body } = await api(`/api/account/keys/${encodeURIComponent(PID)}/endpoints?window=${myWin}&tz_off=${TZOFF}`);
    if (mySeq !== _seq || myWin !== WIN) return;
    if (!ok || !body || body.ok === false) return;
    DATA = body.data || body;
    const stack = $("ep-stack"); if (stack) stack.classList.remove("ov-loading");
    renderAll(!_introDone);
    _introDone = true;
  }
  function renderAll(intro) {
    renderTreemap(intro);
    // full panel rebuild only on intro/first paint; live reconciles update in
    // place so the genome's breathing animation never restarts mid-cycle
    if (intro || !document.getElementById("epi-gn")) renderIntel(intro); else _repaintIntelLive();
    renderAffinity(intro); renderTable(intro);
    _publishFocus(true);   // re-apply cross-widget focus highlights after re-render
    const aux = $("ep-aux");
    if (aux && DATA) aux.textContent = `${fmtNum(DATA.total || 0)} requests · ${(DATA.endpoints || []).length} endpoints · ${WIN_LABEL[WIN]}`;
  }

  // ════════════════════ ① ENDPOINT TREEMAP ════════════════════════════════
  // squarified treemap (Bruls/Huizing). items sorted desc by value.
  function _squarify(items, x, y, w, h) {
    const out = [], total = items.reduce((s, i) => s + i.v, 0) || 1, area = w * h;
    const norm = items.map(i => ({ it: i, a: i.v / total * area }));
    const worst = (row, len) => { const s = row.reduce((a, b) => a + b.a, 0); const mx = Math.max(...row.map(r => r.a)), mn = Math.min(...row.map(r => r.a)); return Math.max((len * len * mx) / (s * s), (s * s) / (len * len * mn)); };
    let rx = x, ry = y, rw = w, rh = h, i = 0;
    while (i < norm.length) {
      const len = Math.min(rw, rh) || 1; let row = [norm[i]]; i++;
      while (i < norm.length && worst(row, len) >= worst(row.concat([norm[i]]), len)) { row.push(norm[i]); i++; }
      const s = row.reduce((a, b) => a + b.a, 0);
      if (rw >= rh) { const cw = s / rh || 0; let cy = ry; row.forEach(r => { const ch = r.a / (cw || 1); out.push({ it: r.it, x: rx, y: cy, w: cw, h: ch }); cy += ch; }); rx += cw; rw -= cw; }
      else { const ch = s / rw || 0; let cx = rx; row.forEach(r => { const cw2 = r.a / (ch || 1); out.push({ it: r.it, x: cx, y: ry, w: cw2, h: ch }); cx += cw2; }); ry += ch; rh -= ch; }
    }
    return out;
  }
  // Card treemap = the lighter terrain: clustered by band, soft pastel tint,
  // ink labels, generous paper gaps — the reference aesthetic, calm + on-brand.
  function treemapSvg(eps, W, H) {
    const layout = _terrainLayout(eps, W, H);
    if (!layout) return _emptyState("no endpoint traffic yet", "send a request to see it grow");
    const col = _C(), total = DATA && DATA.total ? DATA.total : Math.max(1, ...layout.cells.map(c => c.e.n));
    let bands = "", body = "";
    layout.bands.forEach(b => { if (b.h > 26 && b.w > 60) bands += `<text x="${(b.x + 5).toFixed(1)}" y="${(b.y + 9).toFixed(1)}" pointer-events="none" style="font:600 6px 'JetBrains Mono',monospace;fill:${col.mute};letter-spacing:.14em">${esc(b.label)}</text>`; });
    layout.cells.forEach(c => {
      const e = c.e, fill = _lightTint(e.err_rate, e.n / total), big = c.w > 50 && c.h > 26, mid = c.w > 30 && c.h > 18;
      const ec = e.err_pct >= 5 ? col.danger : e.err_pct > 0 ? col.amber : col.soft;
      body += `<g class="epx-tm-cell" data-path="${esc(e.path)}" style="--gc:${fill}">`
        + `<rect class="epx-tm-rect" x="${c.x.toFixed(1)}" y="${c.y.toFixed(1)}" width="${c.w.toFixed(1)}" height="${c.h.toFixed(1)}" rx="5" fill="${fill}" stroke="var(--paper)" stroke-width="2"/>`
        + (big
          ? `<text x="${(c.x + 7).toFixed(1)}" y="${(c.y + 15).toFixed(1)}" pointer-events="none" style="font:600 9px 'JetBrains Mono',monospace;fill:${col.ink}">${esc(_ellip(_pathTail(e.path), c.w - 14, 9))}</text>`
            + `<text x="${(c.x + 7).toFixed(1)}" y="${(c.y + 26).toFixed(1)}" pointer-events="none" style="font:8px 'JetBrains Mono',monospace;fill:${col.soft}">${fmtNum(e.n)}${e.err_pct ? ` · <tspan fill="${ec}">${e.err_pct}%</tspan>` : ""}</text>`
          : mid ? `<text x="${(c.x + 5).toFixed(1)}" y="${(c.y + 13).toFixed(1)}" pointer-events="none" style="font:7.5px 'JetBrains Mono',monospace;fill:${col.ink}">${esc(_ellip(_pathTail(e.path), c.w - 9, 7.5))}</text>` : "")
        + `</g>`;
    });
    return `<svg class="epx-tm-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:${H}px">${bands}${body}</svg>`;
  }
  function renderTreemap(intro) {
    const host = $("ep-treemap-body"); if (!host || !DATA) return;
    const W = Math.max(280, Math.round(host.clientWidth || 520));   // real px ⇒ viewBox 1:1, no stretch
    const H = Math.max(280, Math.round(host.clientHeight || 300));  // fill the card → matches the intelligence panel height
    host.innerHTML = treemapSvg(DATA.endpoints, W, H);
    _wireTreemap(host);
    if (intro) host.querySelectorAll(".epx-tm-cell").forEach((g, i) => { try { g.style.transformBox = "fill-box"; g.style.transformOrigin = "center"; g.animate([{ opacity: 0, transform: "scale(.85)" }, { opacity: 1, transform: "none" }], { duration: 420, delay: Math.min(560, i * 32), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
  }
  function _wireTreemap(host) {
    const byPath = {}; (DATA.endpoints || []).forEach(e => byPath[e.path] = e);
    host.querySelectorAll(".epx-tm-cell").forEach(g => {
      const path = g.getAttribute("data-path"), e = byPath[path]; if (!e) return;
      g.style.cursor = "pointer";
      g.addEventListener("mousemove", (ev) => {
        tip(true, ev.clientX, ev.clientY, `<b>${esc(e.path)}</b><br>${fmtNum(e.n)} req · ${e.err_pct}% err<br>p50 ${fmtMs(e.p50)} · p95 ${fmtMs(e.p95)}<br>${fmtBytes(e.bytes_in + e.bytes_out)} total<br><span style="opacity:.6">click → requests · hover → intelligence</span>`);
        host.querySelectorAll(".epx-tm-cell").forEach(o => o.classList.toggle("epx-dim", o !== g));
        _setFocus(path);
      });
      g.addEventListener("mouseleave", () => { tip(false); host.querySelectorAll(".epx-tm-cell").forEach(o => o.classList.remove("epx-dim")); });
      g.addEventListener("click", () => _setFocus(path, { pin: true }));     // click = pin focus
      g.addEventListener("dblclick", () => { _setFocus(path, { pin: true }); _openExpand("profile"); });
    });
  }

  // ════════════════════ ② ENDPOINT INTELLIGENCE PANEL ═════════════════════
  // Always reflects the focused endpoint (hover anywhere updates it; click
  // pins it). Four reads — weather · genome · fanout · instruments — in our
  // paper-ink hand-drawn language. No emoji.

  function _pearson(a, b) {
    const n = (a || []).length; if (!n || (b || []).length !== n) return 0;
    let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
    return (da && db) ? num / Math.sqrt(da * db) : 0;
  }
  function _outLinksOf(path) { return ((DATA.affinity || {}).links || []).filter(l => l.from === path); }
  function _inLinksOf(path) { return ((DATA.affinity || {}).links || []).filter(l => l.to === path); }

  // Client mirror of the backend intelligence — keeps the panel correct under
  // live SSE (which mutates counts without a refetch). Uses server fields when
  // present and not live-dirtied.
  function _clientIntel(e) {
    const total = DATA.total || 0;
    const maxN = Math.max(1, ...(DATA.endpoints || []).map(x => x.n || 0));
    const T = _SLO[e.cls] || 2000;
    const n = e.n || 0, err = e.err_rate || 0, p95 = e.p95 || 0, avg = e.avg_bytes || 0;
    const retry = e.retry || 0, burst = e.burst || 0;
    const branchN = Object.keys(e.methods || {}).length + _outLinksOf(e.path).length;
    const coupling = Math.abs(_pearson(e.spark || [], e.spark_err || []));
    const sp = e.spark || [], sumSp = sp.reduce((a, b) => a + b, 0), q = Math.max(1, sp.length >> 2);
    const recent = sumSp ? sp.slice(-q).reduce((a, b) => a + b, 0) / sumSp : 0;
    const share = total ? n / total : 0, latR = T ? p95 / T : 0;
    const reasons = [];
    if (n) reasons.push(`${fmtNum(n)} requests · ${(100 * share).toFixed(1)}% of this key's traffic`);
    if (err > 0) reasons.push(`${(100 * err).toFixed(1)}% errors over the window`);
    if (p95) reasons.push(`p95 ${fmtMs(p95)} = ${latR.toFixed(1)}× the ${fmtMs(T)} target`);
    if (burst >= 0.4) reasons.push(`peak bucket ${(1 + burst * 4).toFixed(1)}× the average (bursty)`);
    let state;
    if (n < 3 || (recent <= 0 && share < 0.05 && n < 25)) { state = "Dormant"; reasons.unshift("Little or no recent traffic"); }
    else if (err >= 0.10 || latR >= 4) state = "Critical";
    else if (err >= 0.03 || latR >= 2 || burst >= 0.6) state = "Strained";
    else if (share >= 0.25) state = "Busy";
    else if (share < 0.05 && err < 0.01 && latR < 1) state = "Calm";
    else state = "Steady";
    const half = (sp.length >> 1) || 1, first = sp.slice(0, half).reduce((a, b) => a + b, 0), second = sp.slice(half).reduce((a, b) => a + b, 0);
    const trend = second > first * 1.4 ? "Traffic rising" : second < first * 0.6 ? "Traffic cooling" : "Traffic steady";
    const se = e.spark_err || [];
    const errTrend = se.slice(half).reduce((a, b) => a + b, 0) > se.slice(0, half).reduce((a, b) => a + b, 0) ? "Error trend ↑"
      : se.slice(0, half).reduce((a, b) => a + b, 0) > se.slice(half).reduce((a, b) => a + b, 0) ? "Error trend ↓" : "Errors flat";
    const intel = [trend, errTrend, latR < 1 ? "Latency normal" : latR < 2 ? "Latency elevated" : "Latency high"];
    const genome = {
      seed: e.path, traffic: maxN ? Math.min(1, Math.pow(n / maxN, 0.6)) : 0,
      err_break: Math.min(1, err / 0.2), latency_osc: T ? Math.min(1, p95 / (T * 4)) : 0,
      branches: Math.max(2, Math.min(8, branchN)), payload_density: Math.min(1, Math.sqrt(avg / (256 * 1024))),
      retry_frag: n ? Math.min(1, (retry / n) * 8) : 0, burst: Math.min(1, burst), coupling,
    };
    const heavy = avg >= 256 * 1024, hub = _outLinksOf(e.path).length >= 3 || (e.fanout || 0) >= 40;
    const archetype = e.cls === "heavy_inference" ? "Compute Heavy" : e.cls === "quick_inference" ? "Inference"
      : e.cls === "metadata" ? (hub ? "Gateway" : "Metadata") : (hub ? "Gateway" : "Utility");
    const dens = heavy ? "High" : avg < 8 * 1024 ? "Low" : "Medium";
    const flow = Math.round(100 * (1 - Math.min(1, burst)) * (0.6 + 0.4 * (1 - Math.min(1, err / 0.2))));
    return {
      weather: { state, tone: TONE_OF[state] || "ok", reasons, intel },
      genome, archetype,
      archetype_traits: [heavy ? "High payloads" : avg < 8192 ? "Light payloads" : "Moderate payloads",
        (e.cls === "heavy_inference" || e.cls === "quick_inference") ? "Long execution" : "Fast execution",
        _outLinksOf(e.path).length ? `${_outLinksOf(e.path).length} downstream` : ((e.fanout || 0) >= 40 ? "Wide reach" : "Low fanout")],
      traffic_share: +(100 * share).toFixed(1),
      traits: { branch_complexity: genome.branches, payload_density: dens, flow_stability: flow },
    };
  }
  function _intelOf(e) {
    if (!e) return null;
    if (e._live || !e.weather || !e.genome) return _clientIntel(e);
    return { weather: e.weather, genome: e.genome, archetype: e.archetype,
             archetype_traits: e.archetype_traits || [], traffic_share: e.traffic_share, traits: e.traits || {} };
  }

  // weather "operational sky" — a tiny hand-drawn motif that changes per state
  function _wxMotif(state) {
    const m = {
      Calm: `<path d="M6 26 Q24 20 42 26" /><path d="M10 31 Q24 27 38 31" opacity=".5"/>`,
      Steady: `<path d="M6 24 Q24 24 42 24"/><path d="M6 30 Q24 30 42 30" opacity=".5"/>`,
      Busy: `<path d="M5 22 Q14 16 24 22 Q34 28 43 22"/><path d="M5 29 Q14 23 24 29 Q34 35 43 29" opacity=".55"/>`,
      Strained: `<path d="M5 28 Q15 18 24 26 Q33 34 43 22"/><path d="M7 33 L13 28 L19 33 L25 27 L31 33 L37 28 L43 33" opacity=".6"/>`,
      Critical: `<path d="M6 31 L12 20 L17 27 L23 17 L29 28 L35 19 L42 30"/><path d="M22 14 L18 24 L25 24 L20 34" stroke-width="1.6"/>`,
      Dormant: `<path d="M8 26 Q24 24 40 26" opacity=".4" stroke-dasharray="2 3"/>`,
    };
    return `<svg class="wx-motif" viewBox="0 0 48 40" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${m[state] || m.Steady}</svg>`;
  }

  // fingerprint hovercard text from the genome legend
  function _genomeTip(gn, e) {
    const lg = (gn && gn.legend) ? gn.legend() : [];
    return `<b>Fingerprint — ${esc(_pathTail(e.path))}</b><br>`
      + `<span style="opacity:.7">a unique seal drawn from this endpoint's metrics</span><br>`
      + lg.map(l => `· <b>${esc(l.k)}</b> — ${esc(l.v)}`).join("<br>");
  }

  // ambient animated weather backdrop (operational, in our ink style — not literal)
  function _wxBackdrop(state) {
    return `<div class="epi-wx-bg" aria-hidden="true">${"<span></span>".repeat(8)}</div>`;
  }
  // weather explain-why hovercard (idempotent — uses onX so live rebinds don't stack)
  function _bindWeather(wxEl, wx) {
    if (!wxEl) return;
    const card = `<b>${esc(wx.state)} — operational weather</b><br>`
      + (wx.reasons || []).map(r => "· " + esc(r)).join("<br>")
      + `<br><span style="opacity:.65">${esc((wx.intel || []).join(" · "))}</span>`;
    wxEl.onmousemove = (ev) => tip(true, ev.clientX, ev.clientY, card);
    wxEl.onmouseleave = () => tip(false);
    wxEl.onfocus = () => { const r = wxEl.getBoundingClientRect(); tip(true, r.right - 40, r.top + 10, card); };
    wxEl.onblur = () => tip(false);
  }

  // visible filled-area sparkline with a baseline track (reads clearly even when sparse)
  function _areaSpark(arr, w, h, color, opts) {
    opts = opts || {};
    const n = (arr || []).length || 1, max = Math.max(1, ...(arr || [0]));
    const X = (i) => (n <= 1 ? w / 2 : (i / (n - 1)) * w);
    const Y = (v) => h - (v ? Math.max(2, (v / max) * (h - 2)) : 0);
    let line = "", area = `M 0 ${h}`;
    (arr || []).forEach((v, i) => { const x = X(i).toFixed(1), y = Y(v).toFixed(1); line += (i ? "L" : "M") + ` ${x} ${y} `; area += ` L ${x} ${y}`; });
    area += ` L ${w} ${h} Z`;
    const cur = opts.scrub ? `<line class="ic-cursor" x1="-1" y1="0" x2="-1" y2="${h}" stroke="${color}" stroke-width="1" opacity="0"/>` : "";
    return `<svg class="ic-area${opts.scrub ? " ic-scrub" : ""}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="width:100%;height:${h}px">`
      + `<line x1="0" y1="${h - 0.5}" x2="${w}" y2="${h - 0.5}" stroke="var(--paper-edge)" stroke-width="1"/>`
      + `<path d="${area}" fill="${color}" fill-opacity="0.16"/>`
      + `<path d="${line}" fill="none" stroke="${color}" stroke-width="1.6"/>${cur}</svg>`;
  }
  // wire a scrub cursor + readout + click over an _areaSpark with series metadata
  function _wireSparkScrub(svgEl, series, startMs, bucketMs, onClick, readEl) {
    if (!svgEl) return;
    const n = (series || []).length || 1, cur = svgEl.querySelector(".ic-cursor");
    const max = Math.max(1, ...(series || [0]));
    const at = (cx) => { const r = svgEl.getBoundingClientRect(); return Math.max(0, Math.min(n - 1, Math.round((cx - r.left) / r.width * (n - 1)))); };
    const show = (i, ev) => {
      const v = series[i] || 0, ms = startMs + i * bucketMs;
      if (cur) { const x = (n <= 1 ? 0.5 : i / (n - 1)) * 100; cur.setAttribute("x1", x + "%"); cur.setAttribute("x2", x + "%"); cur.setAttribute("opacity", "0.6"); }
      const iso = _utcStr(ms);
      if (readEl) readEl.innerHTML = `${esc(_ago(iso) || "—")} · <b>${fmtNum(v)}</b> req`;
      if (ev) tip(true, ev.clientX, ev.clientY, `<b>${esc(_ago(iso) || "moment")}</b><br>${fmtNum(v)} request${v === 1 ? "" : "s"}${v ? `<br><span style="opacity:.6">click → these requests</span>` : ""}`);
    };
    svgEl.style.cursor = "ew-resize";
    svgEl.addEventListener("mousemove", (ev) => show(at(ev.clientX), ev));
    svgEl.addEventListener("mouseleave", () => { tip(false); if (cur) cur.setAttribute("opacity", "0"); });
    svgEl.addEventListener("click", (ev) => { const i = at(ev.clientX); if ((series[i] || 0) && onClick) onClick(startMs + i * bucketMs, startMs + (i + 1) * bucketMs); });
  }

  // mini instrument cluster — visual-first; traffic spans the full width so it reads clearly
  function _instrumentCluster(e) {
    const col = _C(), T = _SLO[e.cls] || 2000;
    const errPct = e.err_pct || 0;
    const errFill = errPct >= 10 ? 5 : errPct >= 5 ? 4 : errPct >= 2 ? 3 : errPct > 0.5 ? 2 : errPct > 0 ? 1 : 0;
    const errSquares = Array.from({ length: 5 }, (_, i) => `<span class="ic-sq${i < errFill ? " on" : ""}"></span>`).join("");
    const dens = Math.min(4, Math.round((e.avg_bytes || 0) / (64 * 1024)));
    const payBlocks = Array.from({ length: 4 }, (_, i) => `<span class="ic-blk${i < dens ? " on" : ""}"></span>`).join("");
    const latR = Math.min(1, (e.p95 || 0) / (T * 2));
    return `<div class="ic-stack">
      <div class="ic ic-wide"><span class="ic-lbl">traffic</span><div class="ic-spark">${_areaSpark(e.spark || [], 200, 34, col.accent)}</div></div>
      <div class="ic-row3">
        <div class="ic"><span class="ic-lbl">latency</span><div class="ic-gauge"><div class="ic-gauge-fill" style="width:${(latR * 100).toFixed(0)}%"></div></div><b class="ic-val">${fmtMs(e.p95)}<span>p95</span></b></div>
        <div class="ic"><span class="ic-lbl">errors</span><div class="ic-sqs">${errSquares}</div><b class="ic-val">${errPct}%</b></div>
        <div class="ic"><span class="ic-lbl">payload</span><div class="ic-blks">${payBlocks}</div><b class="ic-val">${fmtBytes(e.avg_bytes)}<span>/req</span></b></div>
      </div>
    </div>`;
  }

  // fanout-mini — roots/branches with live flow particles
  function _fanoutMini(e) {
    const out = _outLinksOf(e.path).slice(0, 3), inn = _inLinksOf(e.path).length;
    const rows = out.length ? out.map(l =>
      `<div class="fm-branch" data-path="${esc(l.to)}"><span class="fm-line"><i class="fm-dot"></i></span><span class="fm-tail">${esc(_pathTail(l.to))}</span><span class="fm-ct">${l.count}×</span></div>`).join("")
      : `<div class="fm-empty">no downstream calls</div>`;
    return `<div class="fm"><div class="fm-head"><span class="fm-in">${inn ? inn + " upstream" : "entry point"}</span><span class="fm-self">${esc(_pathTail(e.path))}</span></div>${rows}</div>`;
  }

  function renderIntel(intro) {
    const host = $("ep-intel-body"); if (!host || !DATA) return;
    const eps = DATA.endpoints || [];
    if (!eps.length) { host.innerHTML = _emptyState("no endpoint traffic yet", "intelligence appears with the first request"); return; }
    if (!_focus || !eps.find(x => x.path === _focus)) _focus = eps[0].path;
    const e = eps.find(x => x.path === _focus) || eps[0];
    const it = _intelOf(e);
    const wx = it.weather, tone = wx.tone || "ok";
    const fa = $("ep-intel-focus"); if (fa) fa.textContent = _pinned ? "pinned" : "hover an endpoint";
    host.innerHTML =
      `<div class="epi" data-tone="${tone}">
         <div class="epi-head">
           <span class="meth ${_methClass(e.method)}">${esc(e.method)}</span>
           <b class="epi-path" title="open profile · ${esc(e.path)}">${esc(_pathTail(e.path))}</b>
           <button class="epi-pin${_pinned ? " on" : ""}" id="epi-pin" title="${_pinned ? "unpin" : "pin this endpoint"}" aria-pressed="${_pinned}"><svg class="icon" aria-hidden="true"><use href="#i-lock"/></svg></button>
         </div>
         <div class="epi-sub">${esc(it.archetype)} · ${fmtNum(e.n)} req · <span class="epi-err">${e.err_pct}% err</span></div>
         <div class="epi-grid">
           <div class="epi-cell epi-weather wx-${tone}" id="epi-weather" data-state="${esc(wx.state)}" tabindex="0">
             ${_wxBackdrop(wx.state)}
             ${_wxMotif(wx.state)}
             <div class="epi-wx-txt"><b>${esc(wx.state)}</b><span>${esc((wx.intel || [])[0] || "")}</span></div>
           </div>
           <div class="epi-cell epi-genome" id="epi-genome"><div class="epi-gn" id="epi-gn"></div><span class="epi-cell-lbl">fingerprint</span></div>
           <div class="epi-cell epi-fanout">${_fanoutMini(e)}<span class="epi-cell-lbl">fanout</span></div>
           <div class="epi-cell epi-instr">${_instrumentCluster(e)}</div>
         </div>
       </div>`;
    // mount genome + fingerprint hovercard (explains every stroke)
    const gnHost = $("epi-gn");
    if (gnHost && window.APIN && APIN.genome) {
      if (_panelGn) { try { _panelGn.destroy(); } catch (_) {} }
      _panelGn = APIN.genome.mount(gnHost, it.genome, { tone, strokeBase: 1.8 });
      const gc = $("epi-genome");
      if (gc) { const txt = _genomeTip(_panelGn, e);
        gc.addEventListener("mousemove", (ev) => tip(true, ev.clientX, ev.clientY, txt));
        gc.addEventListener("mouseleave", () => tip(false)); }
    }
    // weather explain-why hovercard (Wikipedia-style)
    _bindWeather($("epi-weather"), wx);
    // fanout branches → drill to that neighbour's focus (pinned)
    host.querySelectorAll(".fm-branch[data-path]").forEach(b =>
      b.addEventListener("click", () => _setFocus(b.getAttribute("data-path"), { pin: true })));
    // pin toggle
    const pinBtn = $("epi-pin");
    if (pinBtn) pinBtn.addEventListener("click", () => { _pinned = !_pinned; renderIntel(false); _publishFocus(true); });
    // path text → open the full profile
    const pe = host.querySelector(".epi-path");
    if (pe) { pe.style.cursor = "pointer"; pe.addEventListener("click", () => _openExpand("profile")); }
    if (intro) { try { host.querySelector(".epi").animate([{ opacity: 0 }, { opacity: 1 }], { duration: 360, easing: "ease" }); } catch (_) {} }
  }

  // light in-place panel update (numbers + instruments + weather) — no genome
  // teardown, so the sigil's breathing animation never restarts here
  function _repaintIntelLive() {
    const host = $("ep-intel-body"); const e = (DATA.endpoints || []).find(x => x.path === _focus);
    if (!host || !e) return;
    const epi = host.querySelector(".epi"); if (!epi) { renderIntel(false); return; }
    const it = _intelOf(e), wx = it.weather, tone = wx.tone || "ok";
    const sub = epi.querySelector(".epi-sub");
    if (sub) sub.innerHTML = `${esc(it.archetype)} · ${fmtNum(e.n)} req · <span class="epi-err">${e.err_pct}% err</span>`;
    const instr = epi.querySelector(".epi-instr");
    if (instr) instr.innerHTML = _instrumentCluster(e);
    // weather can change live; only rebuild the cell when the STATE flips
    epi.setAttribute("data-tone", tone);
    const wxCell = $("epi-weather");
    if (wxCell) {
      if (wxCell.getAttribute("data-state") !== wx.state) {
        wxCell.setAttribute("data-state", wx.state);
        wxCell.className = "epi-cell epi-weather wx-" + tone;
        wxCell.innerHTML = _wxBackdrop(wx.state) + _wxMotif(wx.state)
          + `<div class="epi-wx-txt"><b>${esc(wx.state)}</b><span>${esc((wx.intel || [])[0] || "")}</span></div>`;
        _bindWeather(wxCell, wx);
      } else {
        const sp = wxCell.querySelector(".epi-wx-txt span"); if (sp) sp.textContent = (wx.intel || [])[0] || "";
      }
    }
  }

  // ── focus bus ──
  function _setFocus(path, opts) {
    opts = opts || {};
    if (!path) return;
    if (opts.pin) { _pinned = !(_pinned && _focus === path); _focus = path; }
    else { if (_pinned || _focus === path) return; _focus = path; }   // hovers ignored while pinned / unchanged
    _publishFocus();
  }
  function _publishFocus(silent) {
    if (!DATA) return;
    if (!silent) renderIntel(false);
    // table rows
    document.querySelectorAll("#ep-table-body .epx-row").forEach(r =>
      r.classList.toggle("epx-row-focus", r.getAttribute("data-path") === _focus));
    // treemap cells
    document.querySelectorAll("#ep-treemap-body .epx-tm-cell").forEach(g =>
      g.classList.toggle("epx-tm-focus", g.getAttribute("data-path") === _focus));
    // constellation: re-centre the orbital on the focused endpoint + strip
    if (_galaxy && _galaxy.setFocus) { try { _galaxy.setFocus(_focus); } catch (_) {} }
    _cstInsight();
    // 3D terrain region (if mounted + supports focus)
    if (_terr3d && _terr3d.focus) { try { _terr3d.focus(_focus); } catch (_) {} }
  }

  // ════════════════════ ③ CONSTELLATION (API Galaxy, compact) ═════════════
  // A focused-hub orbital: the hub follows the focus bus; neighbours orbit;
  // neural edges + watershed particles. Replaces the old Sankey.
  let _galaxy = null, _galaxyRef = null, _routesLoad = null;
  function _galaxyEdges() {
    // session-derived edges if loaded (richer), else the affinity links
    return (DATA._edges && DATA._edges.length) ? DATA._edges : ((DATA.affinity || {}).links || []);
  }
  function renderAffinity(intro) {
    const host = $("ep-affinity-body"); if (!host || !DATA) return;
    const eps = DATA.endpoints || [];
    if (!eps.length) { if (_galaxy) { try { _galaxy.dispose(); } catch (_) {} _galaxy = null; } host.innerHTML = _emptyState("the constellation is dark", "call sequences will light it up"); return; }
    if (!_focus || !eps.find(x => x.path === _focus)) _focus = eps[0].path;
    if (!APIN.galaxy) { host.innerHTML = `<div class="tf-empty">galaxy module unavailable.</div>`; return; }
    const stage = $("ep-cst-stage");
    const live = _galaxy && stage;
    // 1.5s reconcile mutates the SAME endpoint objects (ref-equal) → the galaxy
    // is already live via its own rAF + shared refs; just refresh the strip.
    if (live && _galaxyRef === eps) { _cstInsight(); return; }
    // window change swaps in fresh arrays (ref differs) → re-seed in place.
    if (live) { _galaxy.setData({ endpoints: eps, edges: _galaxyEdges(), hub: DATA._hub }); _galaxy.setFocus(_focus); _galaxyRef = eps; _ensureRoutes(); _cstInsight(); return; }
    if (_galaxy) { try { _galaxy.dispose(); } catch (_) {} }
    host.innerHTML = `<div class="cst-stage" id="ep-cst-stage"></div><div class="cst-strip" id="ep-cst-strip"></div>`;
    _galaxy = APIN.galaxy.create($("ep-cst-stage"), {
      mode: "compact", endpoints: eps, edges: _galaxyEdges(), focus: _focus, hub: DATA._hub,
      onHoverNode: (path, x, y) => { const e = eps.find(z => z.path === path); if (!e) return; const it = _intelOf(e);
        tip(true, x, y, `<b>${esc(e.path)}</b><br>${fmtNum(e.n)} req · ${e.err_pct}% err · p95 ${fmtMs(e.p95)}<br><span style="opacity:.6">${esc((it.weather || {}).state || "")} · click → focus here</span>`);
        _setFocus(path); },
      onHoverEdge: (rec, x, y) => tip(true, x, y, `<b>${esc(_pathTail(rec.from))}</b> → <b>${esc(_pathTail(rec.to))}</b><br>${rec.count}× in sequence`),
      onLeave: () => tip(false),
      onClickNode: (path) => _setFocus(path, { pin: true }),
    });
    _galaxyRef = eps;
    _ensureRoutes();
    _cstInsight();
  }
  // compact insights strip under the constellation (so the card isn't empty)
  function _cstInsight() {
    const strip = $("ep-cst-strip"); if (!strip || !DATA) return;
    const eps = DATA.endpoints || [], bits = [];
    if (DATA._hub) { const deg = (DATA._edges || []).filter(l => l.from === DATA._hub || l.to === DATA._hub).length; bits.push(`<b>${esc(_short(DATA._hub))}</b> hub · ${deg} pathways`); }
    const top = (DATA._routes || [])[0]; if (top) bits.push(`hot route ${top.seq.map(_short).join(" → ")}`);
    const f = eps.find(e => e.path === _focus); if (f) bits.push(`focus <b>${esc(_short(f.path))}</b> · ${fmtNum(f.n)} req · ${f.err_pct}%`);
    if (DATA._sessions) bits.push(`${fmtNum(DATA._sessions)} sessions`);
    strip.innerHTML = bits.map(b => `<span class="cst-ins">${b}</span>`).join("");
  }
  // fetch the session-reconstructed journey graph (edges + routes + hub) once
  // per window; re-seed the compact galaxy with the richer edges when it lands.
  function _ensureRoutes(force) {
    if (!PID) return Promise.resolve(null);
    if (DATA._routesWin === WIN && !force) return Promise.resolve(DATA);
    if (_routesLoad && !force) return _routesLoad;
    const myWin = WIN;
    _routesLoad = api(`/api/account/keys/${encodeURIComponent(PID)}/routes?window=${myWin}&tz_off=${TZOFF}`)
      .then(({ ok, body }) => {
        _routesLoad = null;
        if (!ok || !body || myWin !== WIN) return null;
        const d = body.data || body;
        DATA._routes = d.routes || []; DATA._edges = d.edges || []; DATA._hub = d.hub || null;
        DATA._sessions = d.sessions_total || 0; DATA._routesWin = myWin;
        if (_galaxy && _galaxy.setData) try { _galaxy.setData({ endpoints: DATA.endpoints, edges: _galaxyEdges(), hub: DATA._hub }); _galaxy.setFocus(_focus); _galaxyRef = DATA.endpoints; } catch (_) {}
        return DATA;
      });
    return _routesLoad;
  }

  // ════════════════════ ④ SORTABLE ENDPOINT TABLE ═════════════════════════
  function _sparkBars(arr, w, h, color) {
    const n = (arr || []).length || 1, max = Math.max(1, ...(arr || [0]));
    const bw = w / n;
    let bars = "";
    (arr || []).forEach((v, i) => { const bh = v ? Math.max(1, v / max * h) : 0; bars += `<rect x="${(i * bw).toFixed(1)}" y="${(h - bh).toFixed(1)}" width="${Math.max(0.6, bw - 0.5).toFixed(1)}" height="${bh.toFixed(1)}" rx="0.5" fill="${color}"/>`; });
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="width:${w}px;height:${h}px;vertical-align:middle">${bars}</svg>`;
  }
  function _sortEps(eps) {
    const c = _sort.col, d = _sort.dir;
    return (eps || []).slice().sort((a, b) => {
      let av = a[c], bv = b[c];
      if (c === "path" || c === "method") { av = String(av || ""); bv = String(bv || ""); return d * av.localeCompare(bv); }
      return d * ((+av || 0) - (+bv || 0));
    });
  }
  function renderTable(intro) {
    const host = $("ep-table-body"); if (!host || !DATA) return;
    const col = _C(), eps = _sortEps(DATA.endpoints);
    if (!eps.length) { host.innerHTML = _emptyState("no endpoints yet", "send a request to populate the table"); return; }
    const cols = [["path", "endpoint", "l"], ["n", "n", "r"], ["p50", "p50", "r"], ["p95", "p95", "r"], ["err_pct", "err%", "r"], ["spark", "trend", "c"]];
    const head = `<div class="epx-tr epx-th">` + cols.map(([k, lbl, al]) =>
      `<span class="epx-c-${al}${k !== "spark" ? " epx-sortable" : ""}" ${k !== "spark" ? `data-col="${k}"` : ""}>${esc(lbl)}${_sort.col === k ? `<i class="epx-sort">${_sort.dir < 0 ? "▾" : "▴"}</i>` : ""}</span>`).join("") + `</div>`;
    const rows = eps.map(e => {
      const ec = e.err_pct >= 5 ? col.danger : e.err_pct > 0 ? col.amber : col.soft;
      return `<div class="epx-tr epx-row" data-path="${esc(e.path)}">`
        + `<span class="epx-c-l epx-ep-path"><b class="meth ${_methClass(e.method)}">${esc(e.method)}</b> ${esc(e.path)}</span>`
        + `<span class="epx-c-r">${fmtNum(e.n)}</span>`
        + `<span class="epx-c-r">${fmtMs(e.p50)}</span>`
        + `<span class="epx-c-r">${fmtMs(e.p95)}</span>`
        + `<span class="epx-c-r" style="color:${ec}">${e.err_pct}%</span>`
        + `<span class="epx-c-c">${_sparkBars(e.spark, 56, 18, _errColor(e.err_rate))}</span></div>`;
    }).join("");
    host.innerHTML = `<div class="epx-table">${head}${rows}</div>`;
    if (intro) host.querySelectorAll(".epx-row").forEach((r, i) => { try { r.animate([{ opacity: 0, transform: "translateX(-6px)" }, { opacity: 1, transform: "none" }], { duration: 300, delay: Math.min(560, i * 34), easing: "ease", fill: "backwards" }); } catch (_) {} });
    host.querySelectorAll(".epx-sortable").forEach(h => h.addEventListener("click", () => {
      const c = h.getAttribute("data-col");
      if (_sort.col === c) _sort.dir = -_sort.dir; else { _sort.col = c; _sort.dir = (c === "path" || c === "method") ? 1 : -1; }
      renderTable();
    }));
    host.querySelectorAll(".epx-row[data-path]").forEach(r => {
      r.addEventListener("click", () => { const p = r.getAttribute("data-path"); openReqList({ label: p, path: p }); });
      r.addEventListener("mousemove", (e) => { const ep = (DATA.endpoints || []).find(x => x.path === r.getAttribute("data-path")); if (!ep) return; _setFocus(ep.path); tip(true, e.clientX, e.clientY, `<b>${esc(ep.path)}</b><br>${fmtNum(ep.n)} req · p50 ${fmtMs(ep.p50)} · p95 ${fmtMs(ep.p95)}<br>${fmtBytes(ep.avg_bytes)}/req · ${ep.err_pct}% err<br><span style="opacity:.6">click → requests</span>`); });
      r.addEventListener("mouseleave", () => tip(false));
    });
  }

  // ════════════════════ FILTERED REQUEST LIST (→ drawer) ══════════════════
  function _reqRow(r) {
    const sc = +r.status_code || 0, k = sc >= 500 ? "danger" : sc >= 400 ? "amber" : "ok";
    return `<div class="epx-req" data-rid="${esc(r.id)}">`
      + `<span class="meth ${_methClass(r.method)}">${esc(r.method || "")}</span>`
      + `<span class="epx-req-path" title="${esc(r.path)}">${esc(r.path || "")}</span>`
      + `<span class="epx-req-sc epx-sc-${k}">${sc || "—"}</span>`
      + `<span class="epx-req-lat">${fmtMs(r.latency_ms)}</span>`
      + `<span class="epx-req-ago">${esc(_ago(r.timestamp))}</span></div>`;
  }
  function openReqList(opts) {
    if (!window.APIN || !APIN.lightbox) return;
    const sinceMs = opts.sinceMs != null ? opts.sinceMs : Date.now() - (WIN_MS[WIN] || 86400e3);
    const untilMs = opts.untilMs != null ? opts.untilMs : Date.now() + 1000;
    let curStatus = opts.status || "all";
    APIN.lightbox.open({
      title: "Requests · " + (opts.label || "endpoint"),
      subtitle: WIN_LABEL[WIN],
      hashKey: "ep-reqs",
      sourceCard: $("pane-endpoints"),
      build: async (panel) => {
        panel.innerHTML = `<div class="epx-reqs-chips" id="epx-chips">${["all", "2xx", "4xx", "5xx"].map(s => `<button data-s="${s}"${s === curStatus ? ' aria-pressed="true"' : ""}>${s}</button>`).join("")}</div><div id="epx-reqs-wrap"><div class="tf-empty">loading…</div></div>`;
        const since = _utcStr(sinceMs), until = _utcStr(untilMs);
        const { body } = await api(`/api/account/keys/${encodeURIComponent(PID)}/requests?since=${encodeURIComponent(since)}&until=${encodeURIComponent(until)}&limit=200&tz_off=${TZOFF}`);
        const all = (body && (body.items || (body.data && body.data.items))) || [];
        const draw = () => {
          let rows = all.filter(r => {
            if (opts.path && (r.path || "") !== opts.path) return false;
            if (opts.method && (r.method || "").toUpperCase() !== opts.method) return false;
            const sc = +r.status_code || 0, b = sc >= 500 ? "5xx" : sc >= 400 ? "4xx" : "2xx";
            if (opts.status && b !== opts.status) return false;
            if (curStatus !== "all" && b !== curStatus) return false;
            return true;
          });
          const wrap = $("epx-reqs-wrap"); if (!wrap) return;
          wrap.innerHTML = rows.length
            ? `<div class="epx-reqs-head"><span>${rows.length} request${rows.length > 1 ? "s" : ""}</span><span style="opacity:.6">click a row → full drawer</span></div><div class="epx-reqs">${rows.map(_reqRow).join("")}</div>`
            : `<div class="tf-empty">no matching requests in this window</div>`;
          wrap.querySelectorAll(".epx-req[data-rid]").forEach(el => el.addEventListener("click", () => { if (window.APIN && APIN.requestDrawer) APIN.requestDrawer.open(el.getAttribute("data-rid")); }));
        };
        panel.querySelectorAll("#epx-chips button").forEach(b => b.addEventListener("click", () => { curStatus = b.getAttribute("data-s"); panel.querySelectorAll("#epx-chips button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); draw(); }));
        draw();
      },
    });
  }

  // ════════════════════ LIVING ENDPOINT TERRAIN (treemap expand) ══════════
  // Layered "world map of the API": canvas weather/flow under SVG regions.
  // Two-level squarify: semantic bands (MODEL/CONTENT/SYSTEM) → endpoint cells.
  const TERR_W = 1000, TERR_H = 560;   // logical coordinate space
  const _BANDS = [
    { key: "model", label: "MODEL", test: (p) => /predict|scan|batch|infer/.test(p) },
    { key: "content", label: "CONTENT", test: (p) => /disease|info|model\/card|doc|class|stat/.test(p) },
    { key: "system", label: "SYSTEM", test: (p) => /version|feedback|health|warmup|bench|account|usage|key|webhook|alert/.test(p) },
    { key: "other", label: "OTHER", test: () => true },
  ];
  function _domain(path) { const p = String(path || "").toLowerCase(); for (const b of _BANDS) if (b.test(p)) return b.key; return "other"; }
  function _bandLabel(k) { const b = _BANDS.find(x => x.key === k); return b ? b.label : "OTHER"; }
  // p95 ÷ p50 dispersion → latency instability 0..1
  function _instability(e) { const r = e.p50 > 0 ? e.p95 / e.p50 : 1; return Math.max(0, Math.min(1, (r - 1) / 7)); }
  function _terrainLayout(eps, W, H) {
    W = W || TERR_W; H = H || TERR_H;
    const live = (eps || []).filter(e => e.n > 0);
    if (!live.length) return null;
    const m = Math.max(4, Math.min(8, W * 0.012)), gap = W > 500 ? 2.5 : 1.5;
    const byBand = {};
    live.forEach(e => { const d = _domain(e.path); (byBand[d] = byBand[d] || []).push(e); });
    const bandItems = _BANDS.filter(b => byBand[b.key]).map(b => ({ v: byBand[b.key].reduce((s, e) => s + e.n, 0), key: b.key, label: b.label, eps: byBand[b.key] }));
    const bandRects = _squarify(bandItems, m, m, W - m * 2, H - m * 2);
    const bands = [], cells = [];
    bandRects.forEach(br => {
      const b = br.it; bands.push({ key: b.key, label: b.label, x: br.x, y: br.y, w: br.w, h: br.h, total: b.v });
      const pad = W > 500 ? 7 : 3, headH = br.h > 60 ? 16 : (br.h > 30 ? 10 : 3);
      const inner = _squarify(b.eps.map(e => ({ v: e.n, e })),
        br.x + pad, br.y + headH, Math.max(1, br.w - pad * 2), Math.max(1, br.h - headH - pad));
      inner.forEach(c => cells.push({ e: c.it.e, band: b.key, x: c.x, y: c.y, w: Math.max(0, c.w - gap), h: Math.max(0, c.h - gap) }));
    });
    return { bands, cells };
  }
  // elevation tier by volume share (0=flat dormant .. 3=raised)
  function _elev(e, max) { const t = e.n / (max || 1); return t > 0.5 ? 3 : t > 0.22 ? 2 : t > 0.06 ? 1 : 0; }
  // z1 — cells + elevation + sheen + hit zones (viewBox = real px ⇒ no stretch)
  function _terrainCellsSvg(layout, W, H) {
    const col = _C(), maxN = Math.max(1, ...layout.cells.map(c => c.e.n));
    const total = DATA && DATA.total ? DATA.total : maxN;
    const busiest = layout.cells.slice().sort((a, b) => b.e.n - a.e.n)[0];
    let defs = `<defs>`
      + [1, 2, 3].map(l => `<filter id="eptElev${l}" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="${l + 1}" stdDeviation="${l * 2.0}" flood-color="#1a1612" flood-opacity="${0.10 + l * 0.07}"/></filter>`).join("")
      + `<linearGradient id="eptSheen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0.28"/><stop offset="0.45" stop-color="#fff" stop-opacity="0.03"/><stop offset="1" stop-color="#1a1612" stop-opacity="0.13"/></linearGradient></defs>`;
    let bandLayer = "", cellLayer = "";
    layout.bands.forEach(b => {
      bandLayer += `<rect x="${b.x.toFixed(1)}" y="${b.y.toFixed(1)}" width="${b.w.toFixed(1)}" height="${b.h.toFixed(1)}" rx="12" fill="none" stroke="${col.edge}" stroke-opacity="0.5" stroke-dasharray="2 5"/>`;
    });
    layout.cells.forEach(c => {
      const e = c.e, fill = _lightTint(e.err_rate, e.n / total), lvl = _elev(e, maxN), inst = _instability(e), big = c.w > 56 && c.h > 30;
      const rx = Math.min(11, c.w / 4, c.h / 4).toFixed(1);
      cellLayer += `<g class="ept-cell${busiest && c === busiest ? " ept-breathe" : ""}" data-path="${esc(e.path)}" style="--gc:${_errColor(e.err_rate)}">`
        + `<rect class="ept-cell-rect" x="${c.x.toFixed(1)}" y="${c.y.toFixed(1)}" width="${c.w.toFixed(1)}" height="${c.h.toFixed(1)}" rx="${rx}" fill="${fill}"${lvl ? ` filter="url(#eptElev${lvl})"` : ""}/>`
        + `<rect class="ept-cell-sheen" x="${c.x.toFixed(1)}" y="${c.y.toFixed(1)}" width="${c.w.toFixed(1)}" height="${c.h.toFixed(1)}" rx="${rx}" fill="url(#eptSheen)" pointer-events="none"/>`
        + (inst > 0.45 && big ? `<rect class="ept-cell-edge" x="${c.x.toFixed(1)}" y="${c.y.toFixed(1)}" width="${c.w.toFixed(1)}" height="${c.h.toFixed(1)}" rx="${rx}" fill="none" stroke="${_errColor(e.err_rate)}" stroke-width="1.4" pointer-events="none"/>` : "")
        + `<path class="ept-hit" data-path="${esc(e.path)}" d="M ${c.x.toFixed(1)} ${c.y.toFixed(1)} h ${c.w.toFixed(1)} v ${c.h.toFixed(1)} h ${(-c.w).toFixed(1)} Z" fill="transparent"/></g>`;
    });
    return `<svg class="ept-cells-svg" id="ept-cells" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${defs}${bandLayer}${cellLayer}</svg>`;
  }
  // z3 — crisp labels on top of the canvas weather (never stretched)
  function _terrainLabelsSvg(layout, W, H) {
    const col = _C();
    let s = "";
    layout.bands.forEach(b => { if (b.h > 40 && b.w > 60) s += `<text class="ept-band-lbl" x="${(b.x + 10).toFixed(1)}" y="${(b.y + 14).toFixed(1)}" style="font:600 9px 'JetBrains Mono',monospace;fill:${col.mute};letter-spacing:.16em">${esc(b.label)}</text>`; });
    layout.cells.forEach(c => {
      const e = c.e, big = c.w > 56 && c.h > 30, ec = e.err_pct >= 5 ? col.danger : e.err_pct > 0 ? col.amber : col.soft;
      if (!big) return;
      const fs = c.w > 150 ? 12.5 : 11;
      s += `<text class="ept-lbl ept-lbl-main" data-path="${esc(e.path)}" x="${(c.x + 10).toFixed(1)}" y="${(c.y + 19).toFixed(1)}" style="font:600 ${fs}px 'JetBrains Mono',monospace;fill:${col.ink}">${esc(_ellip(_pathTail(e.path), c.w - 20, fs))}</text>`
        + `<text class="ept-lbl ept-lbl-sub" data-path="${esc(e.path)}" x="${(c.x + 10).toFixed(1)}" y="${(c.y + 33).toFixed(1)}" style="font:9.5px 'JetBrains Mono',monospace;fill:${col.soft}">${fmtNum(e.n)}${e.err_pct ? ` · <tspan fill="${ec}">${e.err_pct}%</tspan>` : ""}</text>`;
    });
    return `<svg class="ept-labels-svg" id="ept-labels" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${s}</svg>`;
  }

  // ── canvas weather/flow layer (ON TOP of cells, under labels) ────────────
  function _roundRect(ctx, x, y, w, h, r) {
    if (w < 2 || h < 2) return; r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function _terrainStartCanvas(canvas, layout, W, H) {
    const ctx = canvas.getContext("2d");
    const maxN = Math.max(1, ...layout.cells.map(c => c.e.n));
    const parts = [];
    layout.cells.forEach((c, ci) => {
      const cnt = Math.min(12, Math.round(1 + (c.e.n / maxN) * 11));
      for (let k = 0; k < cnt; k++) parts.push({ ci, x: c.x + Math.random() * c.w, y: c.y + Math.random() * c.h, a: Math.random() * Math.PI * 2, sp: 0.25 + Math.random() * 0.55 });
    });
    _terrState = { canvas, ctx, layout, parts, W, H, maxN, t0: performance.now(), pings: [] };
    const loop = () => {
      if (!_active || !APIN.lightbox || !APIN.lightbox.isOpen() || !canvas.isConnected) { _terrRAF = null; return; }
      _terrainPaint();
      _terrRAF = requestAnimationFrame(loop);
    };
    _terrRAF = requestAnimationFrame(loop);
  }
  function _terrainPaint() {
    const st = _terrState; if (!st) return;
    const { ctx, canvas, layout, W, H, maxN } = st;
    const cw = canvas.clientWidth || 1, ch = canvas.clientHeight || 1, dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(cw * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
    const sx = cw / W, sy = ch / H, col = _C(), now = performance.now(), tt = (now - st.t0) / 1000;
    ctx.setTransform(dpr * sx, 0, 0, dpr * sy, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const cellByPath = {}; layout.cells.forEach(c => cellByPath[c.e.path] = c);
    // ── topographic contour rings (always) + per-mode weather ──
    layout.cells.forEach(c => {
      const e = c.e, lvl = _elev(e, maxN), cx = c.x + c.w / 2, cy = c.y + c.h / 2;
      const rings = 1 + lvl;                              // raised cells = more contour lines
      const breathe = (c.e.burst > 0.3) ? Math.sin(tt * 1.1 + c.x * 0.02) * 0.8 : 0;
      for (let k = 1; k <= rings; k++) {
        const step = Math.max(6, Math.min(c.w, c.h) / (rings + 2));
        const inset = 4 + k * step;
        if (c.w - inset * 2 < 6 || c.h - inset * 2 < 6) break;
        ctx.strokeStyle = _rgba(col.ink, 0.05 + (rings - k) * 0.018);
        ctx.lineWidth = 1;
        _roundRect(ctx, c.x + inset, c.y + inset + breathe * (k / rings), c.w - inset * 2, c.h - inset * 2, 6);
        ctx.stroke();
      }
      if (_terrMode === "health" && e.err_rate > 0) {
        const a = Math.min(0.5, e.err_rate * (0.4 + 0.2 * Math.sin(tt * 1.6 + cx)));
        const g = ctx.createRadialGradient(cx, cy, 2, cx, cy, Math.max(c.w, c.h) * 0.62);
        g.addColorStop(0, _rgba(col.danger, a)); g.addColorStop(1, _rgba(col.danger, 0));
        ctx.fillStyle = g; _roundRect(ctx, c.x, c.y, c.w, c.h, 8); ctx.fill();
      } else if (_terrMode === "terrain" && e.burst > 0.4 && c.w > 36) {
        const ph = (tt * 0.45 + c.x * 0.01) % 1, r = ph * Math.max(c.w, c.h) * 0.5;
        ctx.strokeStyle = _rgba(col.accent, 0.22 * (1 - ph)); ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.ellipse(cx, cy, r, r * 0.72, 0, 0, Math.PI * 2); ctx.stroke();
      }
    });
    // ── flow particles (flow mode) ──
    if (_terrMode === "flow") {
      ctx.lineCap = "round";
      st.parts.forEach(p => {
        const c = layout.cells[p.ci]; if (!c) return;
        p.a += Math.sin(tt * 0.3 + p.x * 0.02) * 0.05;
        p.x += Math.cos(p.a) * p.sp * (0.6 + c.e.n / 70);
        p.y += Math.sin(p.a) * p.sp * 0.6;
        if (p.x < c.x || p.x > c.x + c.w || p.y < c.y || p.y > c.y + c.h) { p.x = c.x + Math.random() * c.w; p.y = c.y + Math.random() * c.h; }
        ctx.strokeStyle = _rgba(_errColor(c.e.err_rate), 0.62);
        ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x - Math.cos(p.a) * 5, p.y - Math.sin(p.a) * 5); ctx.stroke();
      });
    }
    // ── hover pathways (affinity) ──
    if (_terrSel) {
      const src = cellByPath[_terrSel]; const links = ((DATA.affinity || {}).links || []).filter(l => l.from === _terrSel);
      if (src) links.forEach(l => {
        const dst = cellByPath[l.to]; if (!dst) return;
        const x1 = src.x + src.w / 2, y1 = src.y + src.h / 2, x2 = dst.x + dst.w / 2, y2 = dst.y + dst.h / 2;
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - 44, dash = (tt * 60) % 16;
        ctx.strokeStyle = _rgba(col.ink, 0.55); ctx.lineWidth = Math.max(1.2, Math.min(5, l.count / 6)); ctx.setLineDash([6, 10]); ctx.lineDashOffset = -dash;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(mx, my, x2, y2); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = _rgba(col.accent, 0.9); ctx.beginPath(); ctx.arc(x2, y2, 3.2, 0, Math.PI * 2); ctx.fill();
      });
    }
    // ── live pings (edge pulse on a fresh request) ──
    st.pings = st.pings.filter(pg => now - pg.t < 750);
    st.pings.forEach(pg => { const c = cellByPath[pg.path]; if (!c) return; const k = (now - pg.t) / 750; ctx.strokeStyle = _rgba(_errColor(c.e.err_rate), 0.7 * (1 - k)); ctx.lineWidth = 2; _roundRect(ctx, c.x - k * 7, c.y - k * 7, c.w + k * 14, c.h + k * 14, 10); ctx.stroke(); });
  }
  function _rgba(c, a) { if (c[0] !== "#") { const m = c.match(/\d+/g); return m ? `rgba(${m[0]},${m[1]},${m[2]},${a})` : c; } c = c.replace("#", ""); if (c.length === 3) c = c.split("").map(x => x + x).join(""); return `rgba(${parseInt(c.slice(0, 2), 16)},${parseInt(c.slice(2, 4), 16)},${parseInt(c.slice(4, 6), 16)},${a})`; }

  // ── bottom intelligence strip ────────────────────────────────────────────
  function _dnaStrip(e, eps) {
    const maxN = Math.max(1, ...eps.map(x => x.n)), maxB = Math.max(1, ...eps.map(x => x.avg_bytes));
    const Tcls = { quick_inference: 8000, heavy_inference: 15000, metadata: 300, default: 1000 };
    const latT = Tcls[e.cls] || 1000;
    const bars = [
      ["traffic", e.n / maxN, fmtNum(e.n)],
      ["latency", Math.min(1, (e.p95 || 0) / (latT * 2)), fmtMs(e.p95)],
      ["errors", Math.min(1, e.err_rate / 0.3), e.err_pct + "%"],
      ["payload", (e.avg_bytes || 0) / maxB, fmtBytes(e.avg_bytes)],
      ["bursting", e.burst || 0, Math.round((e.burst || 0) * 100) + "%"],
    ];
    return `<div class="ept-dna">` + bars.map(([k, v, lbl]) =>
      `<div class="ept-dna-row"><span class="ept-dna-k">${k}</span><span class="ept-dna-track"><i style="width:${Math.round(Math.max(0, Math.min(1, v)) * 100)}%"></i></span><span class="ept-dna-v">${esc(lbl)}</span></div>`).join("") + `</div>`;
  }
  function _behaviorTags(e, eps) {
    const medFan = (() => { const f = eps.map(x => x.fanout).sort((a, b) => a - b); return f[Math.floor(f.length / 2)] || 1; })();
    const tags = [];
    if (e.burst >= 0.5) tags.push(["burst-prone", "amber"]);
    if (e.fanout >= Math.max(3, medFan * 2)) tags.push(["high fan-out", "amber"]);
    if (e.retry >= 3) tags.push(["retry-heavy", "danger"]);
    if (e.p50 > 0 && e.p95 / e.p50 >= 4) tags.push(["latency-unstable", "amber"]);
    if (e.err_pct >= 10) tags.push(["error-prone", "danger"]);
    if ((e.methods && (e.methods.GET || 0) / e.n >= 0.8)) tags.push(["read-mostly", "ok"]);
    if (e.err_pct === 0 && (e.p50 === 0 || e.p95 / Math.max(1, e.p50) < 3)) tags.push(["healthy", "ok"]);
    if (!tags.length) tags.push(["nominal", "neutral"]);
    return `<div class="ept-tags">` + tags.slice(0, 5).map(([t, k]) => `<span class="ept-tag ept-tag-${k}">${esc(t)}</span>`).join("") + `</div>`;
  }
  function _rhythmClock(e) {
    const hours = e.hours || [], max = Math.max(1, ...hours), S = 150, cx = S / 2, cy = S / 2, r0 = 20, r1 = 68;
    const col = _C(); let w = "", ticks = "";
    [0.5, 0.75, 1].forEach(f => { w += `<circle cx="${cx}" cy="${cy}" r="${(r0 + (r1 - r0) * f).toFixed(1)}" fill="none" stroke="${col.edge}" stroke-opacity="0.3" stroke-width="0.6"/>`; });
    for (let h = 0; h < 24; h++) {
      const a0 = -Math.PI / 2 + (h / 24) * 2 * Math.PI + 0.02, a1 = -Math.PI / 2 + ((h + 1) / 24) * 2 * Math.PI - 0.02;
      const rr = r0 + (r1 - r0) * ((hours[h] || 0) / max);
      const x0 = cx + r0 * Math.cos(a0), y0 = cy + r0 * Math.sin(a0), x1 = cx + rr * Math.cos(a0), y1 = cy + rr * Math.sin(a0);
      const x2 = cx + rr * Math.cos(a1), y2 = cy + rr * Math.sin(a1), x3 = cx + r0 * Math.cos(a1), y3 = cy + r0 * Math.sin(a1);
      w += `<path d="M ${x0.toFixed(1)} ${y0.toFixed(1)} L ${x1.toFixed(1)} ${y1.toFixed(1)} A ${rr} ${rr} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)} L ${x3.toFixed(1)} ${y3.toFixed(1)} Z" fill="var(--c-accent,#52b788)" fill-opacity="${hours[h] ? 0.88 : 0.10}"/>`;
    }
    [0, 6, 12, 18].forEach(hh => { const a = -Math.PI / 2 + (hh / 24) * 2 * Math.PI, lx = cx + (r1 + 7) * Math.cos(a), ly = cy + (r1 + 7) * Math.sin(a); ticks += `<text x="${lx.toFixed(1)}" y="${(ly + 3).toFixed(1)}" text-anchor="middle" style="font:7px 'JetBrains Mono',monospace;fill:${col.mute}">${hh === 0 ? "12a" : hh === 12 ? "12p" : hh > 12 ? (hh - 12) + "p" : hh + "a"}</text>`; });
    return `<svg viewBox="0 0 ${S} ${S}" style="width:148px;height:148px;overflow:visible">${w}${ticks}<circle cx="${cx}" cy="${cy}" r="${r0 - 3}" fill="var(--paper)" stroke="${col.edge}" stroke-width="1"/></svg>`;
  }
  function _relationships(e) {
    const links = ((DATA.affinity || {}).links || []).filter(l => l.from === e.path).slice(0, 4);
    if (!links.length) return `<div class="ept-rel-empty">no common next-hop yet</div>`;
    return `<div class="ept-rel">` + links.map(l => `<div class="ept-rel-row"><span class="ept-rel-arrow">↳</span><span class="ept-rel-path">${esc(_pathTail(l.to))}</span><span class="ept-rel-n">${l.count}×</span></div>`).join("") + `</div>`;
  }
  function _terrainInsight(e, eps) {
    const share = DATA.total ? Math.round(100 * e.n / DATA.total) : 0;
    const Tcls = { quick_inference: 8000, heavy_inference: 15000, metadata: 300, default: 1000 };
    const latT = Tcls[e.cls] || 1000, slow = e.p95 > latT;
    let s = `<b>${esc(e.path)}</b> holds ${share}% of this key's traffic`;
    if (e.err_pct >= 10) s += `, and it's <b style="color:var(--c-danger,#b3402f)">erroring at ${e.err_pct}%</b>`;
    else if (slow) s += `, running <b>slow for a ${_bandLabel(_domain(e.path)).toLowerCase()} route</b> (p95 ${fmtMs(e.p95)})`;
    else if (e.burst >= 0.5) s += ` in <b>bursty</b> spikes`;
    else s += ` and looks healthy`;
    s += ".";
    return s;
  }
  function _renderStrip(path) {
    const host = $("ept-strip"); if (!host || !DATA) return;
    const eps = (DATA.endpoints || []).filter(x => x.n > 0);
    const e = eps.find(x => x.path === path) || eps[0]; if (!e) { host.innerHTML = ""; return; }
    host.innerHTML =
      `<div class="ept-strip-head"><span class="meth ${_methClass(e.method)}">${esc(e.method)}</span> <b>${esc(e.path)}</b><span class="ept-strip-cls">${esc(_bandLabel(_domain(e.path)))}</span></div>
       <div class="ept-strip-grid">
         <div><div class="ept-strip-lbl">endpoint DNA</div>${_dnaStrip(e, eps)}</div>
         <div><div class="ept-strip-lbl">behaviour</div>${_behaviorTags(e, eps)}
           <div class="ept-strip-lbl" style="margin-top:9px">common next-hop</div>${_relationships(e)}</div>
         <div class="ept-strip-rhythm"><div class="ept-strip-lbl">daily rhythm</div>${_rhythmClock(e)}</div>
       </div>
       <p class="tfx-insight ept-strip-ins">${_terrainInsight(e, eps)}</p>
       <button class="tfx-drill-btn" id="ept-strip-req">view ${fmtNum(e.n)} ${esc(e.method)} ${esc(_pathTail(e.path))} requests →</button>`;
    const rb = $("ept-strip-req");
    if (rb) rb.addEventListener("click", () => openReqList({ label: e.path, path: e.path }));
  }

  function _buildTerrain(panel) {
    panel.innerHTML =
      `<div class="ept-controls">
         <div class="ov-range ept-mode" id="ept-mode">
           <button data-m="terrain" aria-pressed="true"><svg class="icon" aria-hidden="true"><use href="#i-grid"/></svg> terrain</button>
           <button data-m="flow"><svg class="icon" aria-hidden="true"><use href="#i-route"/></svg> flow</button>
           <button data-m="health"><svg class="icon" aria-hidden="true"><use href="#i-activity"/></svg> health</button>
           <button data-m="core"><svg class="icon" aria-hidden="true"><use href="#i-scan"/></svg> x-ray</button>
         </div>
         <span class="ept-mode-hint" id="ept-mode-hint">volume dominance · size = traffic · tint = errors · contours = elevation</span>
       </div>
       <div class="ept-stage" id="ept-stage"></div>
       <div class="ept-strip" id="ept-strip"></div>`;
    const stage = $("ept-stage"); if (!stage) return;
    _terrMode = "terrain"; _terrSel = null; _terrZoom = null; _terr3d = null; _terrCore = null; _coreStrata = "traffic";
    const W = Math.max(360, Math.round(stage.clientWidth || 900));
    const H = Math.max(260, Math.round(stage.clientHeight || 520));
    const layout = _terrainLayout(DATA.endpoints, W, H);
    if (!layout) { stage.innerHTML = `<div class="tf-empty" style="padding:60px">No endpoint traffic yet — terrain emerges with requests.</div>`; return; }
    _renderStrip((DATA.endpoints[0] || {}).path);
    if (_webglOK()) _mount3D(stage, layout, W, H);
    else _build2DStage(stage, layout, W, H);
    // smooth dominance drift: re-lay every 2.5s while open & not zoomed (3D or 2D)
    if (_terrReflowTimer) clearInterval(_terrReflowTimer);
    _terrReflowTimer = setInterval(() => {
      if (!(_active && APIN.lightbox && APIN.lightbox.isOpen() && _terrZoom == null)) return;
      const lay = _terrainLayout(DATA.endpoints, W, H); if (!lay) return;
      if (_terr3d) _terr3d.setLayout(lay, DATA.affinity); else _terrReflow(W, H);
    }, 2500);
  }
  // ── WebGL 3D world (lazy Three.js) ──
  async function _mount3D(stage, layout, W, H) {
    stage.classList.add("ept-stage-3d");
    stage.innerHTML = `<canvas class="ept3-canvas" id="ept3-canvas"></canvas><div class="ept3-labels" id="ept3-labels"></div><div class="ept-core-panel" id="ept-core-panel" hidden></div><div class="ept-zoom" id="ept-zoom" hidden></div><div class="ept3-hint">drag to orbit · scroll to zoom · click a hill to open its core</div>`;
    const panel = stage.parentElement;
    panel.querySelectorAll("#ept-mode button").forEach(b => b.addEventListener("click", () => {
      const m = b.getAttribute("data-m"); _terrMode = m;
      panel.querySelectorAll("#ept-mode button").forEach(x => { if (x.getAttribute("data-m") === m) x.setAttribute("aria-pressed", "true"); else x.removeAttribute("aria-pressed"); });
      _setTerrHint(m);
      if (_terrCore) _exitCore();                 // switching the terrain mode leaves any open cutaway
      if (_terr3d) _terr3d.setMode(m);
    }));
    try {
      await _ensureThree();
      if (!window.APIN.terrain3d || !window.THREE) throw new Error("three unavailable");
      if (!stage.isConnected) return;   // modal closed mid-load
      _terr3d = window.APIN.terrain3d.create($("ept3-canvas"), {
        layout, W, H, affinity: DATA.affinity, labelHost: $("ept3-labels"),
        onHover: (path, x, y) => { const e = (DATA.endpoints || []).find(z => z.path === path); if (!e) return; _terrSel = path; _setFocus(path); tip(true, x, y, `<b>${esc(e.path)}</b><br>${fmtNum(e.n)} req · ${e.err_pct}% err · p95 ${fmtMs(e.p95)}<br><span style="opacity:.6">click → open its core</span>`); _renderStrip(path); },
        onLeave: () => { tip(false); },
        onClick: (path) => { const e = (DATA.endpoints || []).find(z => z.path === path); if (e) { _setFocus(path, { pin: true }); _enterCore(e); } },
      });
      _terr3d.setMode(_terrMode);
      window.addEventListener("resize", _terr3dResize);
    } catch (err) { stage.classList.remove("ept-stage-3d"); _build2DStage(stage, layout, W, H); }
  }
  // ── 2D fallback stage (SVG cells + canvas weather + labels) ──
  function _build2DStage(stage, layout, W, H) {
    stage.innerHTML = _terrainCellsSvg(layout, W, H)
      + `<canvas class="ept-canvas" id="ept-canvas"></canvas>`
      + _terrainLabelsSvg(layout, W, H)
      + `<div class="ept-zoom" id="ept-zoom" hidden></div>`;
    _terrainStartCanvas($("ept-canvas"), layout, W, H);
    _wireTerrain(stage, layout, W, H);
    stage.querySelectorAll(".ept-cell").forEach((g, i) => { try { g.style.transformBox = "fill-box"; g.style.transformOrigin = "center"; g.animate([{ opacity: 0, transform: "scale(.9)" }, { opacity: 1, transform: "none" }], { duration: 480, delay: Math.min(640, i * 28), easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
    stage.querySelectorAll(".ept-labels-svg .ept-lbl").forEach((t, i) => { try { t.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: Math.min(700, 200 + i * 24), easing: "ease", fill: "backwards" }); } catch (_) {} });
  }
  function _wireTerrain(stage, layout, W, H) {
    const setMode = (m) => {
      _terrMode = m;
      stage.parentElement.querySelectorAll("#ept-mode button").forEach(b => { if (b.getAttribute("data-m") === m) b.setAttribute("aria-pressed", "true"); else b.removeAttribute("aria-pressed"); });
      const hint = $("ept-mode-hint");
      if (hint) hint.textContent = m === "flow" ? "request movement · particles drift by volume · hover → call pathways" : m === "health" ? "error propagation · red bleeds where requests fail" : "volume dominance · size = traffic · tint = errors · contours = elevation";
    };
    stage.parentElement.querySelectorAll("#ept-mode button").forEach(b => b.addEventListener("click", () => setMode(b.getAttribute("data-m"))));
    const byPath = {}; (DATA.endpoints || []).forEach(e => byPath[e.path] = e);
    stage.querySelectorAll(".ept-cells-svg .ept-hit").forEach(h => {
      const path = h.getAttribute("data-path"), e = byPath[path]; if (!e) return;
      h.style.cursor = "pointer";
      h.addEventListener("mousemove", (ev) => {
        _terrSel = path;
        stage.classList.add("ept-focusing");
        const tgts = ((DATA.affinity || {}).links || []).filter(l => l.from === path).map(l => l.to);
        stage.querySelectorAll(".ept-cell").forEach(g => { const p = g.getAttribute("data-path"); g.classList.toggle("ept-cell-on", p === path); g.classList.toggle("ept-cell-linked", tgts.includes(p)); });
        stage.querySelectorAll(".ept-lbl").forEach(t => { const p = t.getAttribute("data-path"); t.classList.toggle("ept-lbl-on", p === path || tgts.includes(p)); });
        tip(true, ev.clientX, ev.clientY, `<b>${esc(e.path)}</b><br>${fmtNum(e.n)} req · ${e.err_pct}% err · p95 ${fmtMs(e.p95)}<br><span style="opacity:.6">click → enter region</span>`);
        _renderStrip(path);
      });
      h.addEventListener("mouseleave", () => { tip(false); _terrSel = null; stage.classList.remove("ept-focusing"); stage.querySelectorAll(".ept-cell-on,.ept-cell-linked,.ept-lbl-on").forEach(g => g.classList.remove("ept-cell-on", "ept-cell-linked", "ept-lbl-on")); });
      h.addEventListener("click", () => _enterRegion(e));
    });
  }
  // smooth dominance drift — recompute layout, tween cell rects + labels in place
  function _terrReflow(W, H) {
    const stage = $("ept-stage"), cellsSvg = $("ept-cells"), labelsSvg = $("ept-labels");
    if (!stage || !cellsSvg || !labelsSvg || !_terrState) return;
    const next = _terrainLayout(DATA.endpoints, W, H); if (!next) return;
    const cur = _terrState.layout;
    if (next.cells.length !== cur.cells.length) {   // structure changed → clean rebuild
      const sel = _terrSel;
      cellsSvg.outerHTML = _terrainCellsSvg(next, W, H);
      labelsSvg.outerHTML = _terrainLabelsSvg(next, W, H);
      _terrState.layout = next; _terrState.maxN = Math.max(1, ...next.cells.map(c => c.e.n));
      _wireTerrain(stage, next, W, H);
      return;
    }
    const tgt = {}; next.cells.forEach(c => tgt[c.e.path] = c);
    const items = [];
    cur.cells.forEach(c => {
      const t = tgt[c.e.path]; if (!t) return;
      const g = cellsSvg.querySelector(`.ept-cell[data-path="${c.e.path}"]`);
      const lm = labelsSvg.querySelector(`.ept-lbl-main[data-path="${c.e.path}"]`);
      const ls = labelsSvg.querySelector(`.ept-lbl-sub[data-path="${c.e.path}"]`);
      items.push({ c, from: { x: c.x, y: c.y, w: c.w, h: c.h }, to: { x: t.x, y: t.y, w: t.w, h: t.h },
                   rect: g && g.querySelector(".ept-cell-rect"), sheen: g && g.querySelector(".ept-cell-sheen"), edge: g && g.querySelector(".ept-cell-edge"), hit: g && g.querySelector(".ept-hit"), lm, ls });
    });
    if (items.length !== cur.cells.length) return;
    const t0 = performance.now(), dur = 700;
    const tween = () => {
      const k = Math.min(1, (performance.now() - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      items.forEach(it => {
        const x = it.from.x + (it.to.x - it.from.x) * e, y = it.from.y + (it.to.y - it.from.y) * e,
              w = it.from.w + (it.to.w - it.from.w) * e, h = it.from.h + (it.to.h - it.from.h) * e;
        // mutate the live layout cell so the canvas (contours/weather) stays in sync
        it.c.x = x; it.c.y = y; it.c.w = w; it.c.h = h;
        const rx = Math.min(11, w / 4, h / 4).toFixed(1);
        [it.rect, it.sheen, it.edge].forEach(r => { if (r) { r.setAttribute("x", x.toFixed(1)); r.setAttribute("y", y.toFixed(1)); r.setAttribute("width", w.toFixed(1)); r.setAttribute("height", h.toFixed(1)); r.setAttribute("rx", rx); } });
        if (it.hit) it.hit.setAttribute("d", `M ${x.toFixed(1)} ${y.toFixed(1)} h ${w.toFixed(1)} v ${h.toFixed(1)} h ${(-w).toFixed(1)} Z`);
        if (it.lm) { it.lm.setAttribute("x", (x + 10).toFixed(1)); it.lm.setAttribute("y", (y + 19).toFixed(1)); }
        if (it.ls) { it.ls.setAttribute("x", (x + 10).toFixed(1)); it.ls.setAttribute("y", (y + 33).toFixed(1)); }
      });
      if (k < 1) requestAnimationFrame(tween);
    };
    requestAnimationFrame(tween);
  }
  // cinematic region zoom → micro-topology sub-view
  function _catmull(pts) {
    if (pts.length < 2) return pts.length ? `M ${pts[0][0]} ${pts[0][1]}` : "";
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6, c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
    }
    return d;
  }
  // ── per-hill "earth core" cutaway (3D) + strata side panel ──
  // _SLO shared with the Intelligence Panel; default matches backend (2000ms).
  const _SLO = { quick_inference: 8000, heavy_inference: 15000, metadata: 300, default: 2000 };
  function _coreStrataInfo(e, strata) {
    const T = _SLO[e.cls] || 1000;
    const pc = (s) => s >= 0.8 ? "#299e6b" : s >= 0.5 ? "#d19e33" : "#c74f33";
    if (strata === "status") {
      return { title: "status mix · shell → core", bands: [
        { lbl: "2xx success", val: fmtNum(e.n2 || 0), col: "#299e6b" },
        { lbl: "4xx client", val: fmtNum(e.n4 || 0), col: "#d19e33" },
        { lbl: "5xx server", val: fmtNum(e.n5 || 0), col: "#c74f33", core: true }] };
    }
    if (strata === "pillars") {
      const rel = 1 - (e.err_rate || 0);
      const perf = 1 - Math.min(1, (e.p95 || 0) / (T * 4));
      const stab = 1 - Math.min(1, (e.p50 > 0 ? e.p95 / e.p50 - 1 : 0) / 7);
      return { title: "health pillars · shell → core", bands: [
        { lbl: "reliability", val: Math.round(rel * 100) + "%", col: pc(rel) },
        { lbl: "performance", val: Math.round(perf * 100) + "%", col: pc(perf) },
        { lbl: "stability", val: Math.round(stab * 100) + "%", col: pc(stab), core: true }] };
    }
    const latS = 1 - Math.min(1, (e.p95 || 0) / (T * 4));
    return { title: "signal layers · shell → core", bands: [
      { lbl: "traffic", val: fmtNum(e.n) + " req", col: "#299e6b" },
      { lbl: "latency p95", val: fmtMs(e.p95), col: pc(latS) },
      { lbl: "error rate", val: e.err_pct + "%", col: e.err_rate > 0 ? "#c74f33" : "#299e6b", core: true }] };
  }
  function _enterCore(e) {
    if (!_terr3d) { _enterRegion(e); return; }   // 2D fallback has no cutaway → river
    _terrCore = e.path; _coreStrata = "traffic";
    try { _terr3d.setStrata("traffic"); _terr3d.enterCore(e); } catch (_) {}
    _buildCorePanel(e, true);
  }
  function _buildCorePanel(e, animate) {
    const panel = $("ept-core-panel"); if (!panel) return;
    const info = _coreStrataInfo(e, _coreStrata);
    const subs = [["traffic", "signals"], ["status", "status"], ["pillars", "health"]];
    const coreLbl = info.bands[info.bands.length - 1].lbl;
    panel.innerHTML =
      `<div class="ept-core-head">
         <span class="meth ${_methClass(e.method)}">${esc(e.method)}</span>
         <b title="${esc(e.path)}">${esc(_pathTail(e.path))}</b>
         <button class="ept-core-x" id="ept-core-x" title="back to terrain"><svg class="icon" aria-hidden="true"><use href="#i-x"/></svg></button>
       </div>
       <div class="ov-range ept-core-strata" id="ept-core-strata">
         ${subs.map(([k, l]) => `<button data-st="${k}"${k === _coreStrata ? ' aria-pressed="true"' : ""}>${l}</button>`).join("")}
       </div>
       <div class="ept-core-title">${esc(info.title)}</div>
       <div class="ept-core-legend">
         ${info.bands.map((b, i) => `<div class="ept-core-band${b.core ? " is-core" : ""}"><span class="ept-core-dot" style="background:${b.col}"></span><span class="ept-core-bl">${esc(b.lbl)}</span><b>${esc(b.val)}</b><span class="ept-core-ring">ring ${i + 1}</span></div>`).join("")}
       </div>
       <p class="ept-core-note">drag the wheel to spin · innermost ring = <b>${esc(coreLbl)}</b></p>
       <div class="ept-core-stats"><span>p50 <b>${fmtMs(e.p50)}</b></span><span>p95 <b>${fmtMs(e.p95)}</b></span><span>IPs <b>${e.fanout}</b></span></div>
       <div class="ept-core-actions">
         <button class="tfx-drill-btn" id="ept-core-river"><svg class="icon" aria-hidden="true"><use href="#i-waterfall"/></svg> traffic river</button>
         <button class="tfx-drill-btn ghost" id="ept-core-back">← terrain</button>
       </div>`;
    panel.hidden = false;
    if (animate) { try { panel.animate([{ opacity: 0, transform: "translateX(18px)" }, { opacity: 1, transform: "none" }], { duration: 300, easing: "cubic-bezier(.22,1,.36,1)" }); } catch (_) {} }
    panel.querySelectorAll("#ept-core-strata button").forEach(b => b.addEventListener("click", () => {
      _coreStrata = b.getAttribute("data-st");
      if (_terr3d) try { _terr3d.setStrata(_coreStrata); } catch (_) {}
      _buildCorePanel(e, false);
    }));
    const x = $("ept-core-x"), bk = $("ept-core-back"), rv = $("ept-core-river");
    if (x) x.addEventListener("click", _exitCore);
    if (bk) bk.addEventListener("click", _exitCore);
    if (rv) rv.addEventListener("click", () => { _exitCore(); _enterRegion(e); });
    _renderStrip(e.path);
  }
  function _exitCore() {
    const e = (DATA.endpoints || []).find(z => z.path === _terrCore);
    _terrCore = null;
    if (_terr3d) try { _terr3d.exitCore(); } catch (_) {}
    const panel = $("ept-core-panel");
    if (panel) { try { panel.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 180, easing: "ease" }).finished.then(() => { panel.hidden = true; panel.innerHTML = ""; }); } catch (_) { panel.hidden = true; panel.innerHTML = ""; } }
  }

  function _enterRegion(e) {
    if (_terrCore) _exitCore();
    const zoom = $("ept-zoom"); if (!zoom) return;
    _terrZoom = e.path;
    const col = _C(), spark = e.spark || [], sparkErr = e.spark_err || [], n = spark.length || 1, max = Math.max(1, ...spark);
    const W = 1000, H = 320, padX = 18, baseY = H - 38, topY = 20, plotH = baseY - topY;
    const winMs = WIN_MS[WIN] || 86400e3, dur = winMs / n, startMs = Date.now() - winMs;
    const X = (i) => padX + (n <= 1 ? 0.5 : i / (n - 1)) * (W - 2 * padX);
    const Y = (v) => baseY - (v / max) * plotH;
    const pts = spark.map((v, i) => [X(i), Y(v)]);
    const lineD = _catmull(pts);
    const areaD = lineD + ` L ${X(n - 1).toFixed(1)} ${baseY} L ${X(0).toFixed(1)} ${baseY} Z`;
    let dots = "", xlab = ""; const stride = Math.max(1, Math.ceil(n / 5));
    spark.forEach((v, i) => {
      if (sparkErr[i]) dots += `<circle class="ept-river-dot" cx="${X(i).toFixed(1)}" cy="${(Y(v) - 7).toFixed(1)}" r="3.4" fill="${col.danger}"/>`;
      if (i % stride === 0) { const iso = new Date(startMs + i * dur).toISOString().slice(0, 19).replace("T", " "); xlab += `<text x="${X(i).toFixed(1)}" y="${H - 20}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${esc((_ago(iso) || "").replace(" ago", ""))}</text>`; }
    });
    const links = ((DATA.affinity || {}).links || []).filter(l => l.from === e.path).slice(0, 5);
    const flows = links.length ? links.map(l => `<div class="ept-zoom-flow" data-path="${esc(l.to)}"><span class="ept-zoom-arrow">→</span><span>${esc(_pathTail(l.to))}</span><span class="ept-zoom-fn">${l.count}×</span></div>`).join("") : `<div class="ept-rel-empty">no downstream calls</div>`;
    zoom.innerHTML =
      `<div class="ept-zoom-head"><button class="ept-zoom-back" id="ept-zoom-back"><svg class="icon" aria-hidden="true" style="transform:rotate(180deg)"><use href="#i-chevron-right"/></svg> terrain</button>
         <span class="meth ${_methClass(e.method)}">${esc(e.method)}</span><b>${esc(e.path)}</b>
         <span class="ept-zoom-cls">${esc(_bandLabel(_domain(e.path)))} · ${fmtNum(e.n)} req</span></div>
       <div class="ept-zoom-body">
         <div class="ept-zoom-river">
           <div class="ept-strip-lbl"><span>traffic river · drag the scrubber · click a slice → requests</span><span class="ept-river-read" id="ept-river-read"></span></div>
           <svg class="ept-river-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:100%;min-height:280px">
             <defs><linearGradient id="eptRiv" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${col.accent}" stop-opacity="0.5"/><stop offset="1" stop-color="${col.accent}" stop-opacity="0.04"/></linearGradient></defs>
             <line x1="${padX}" y1="${baseY}" x2="${W - padX}" y2="${baseY}" stroke="var(--paper-edge)"/>
             <path class="ept-river-area" id="ept-river-area" d="${areaD}" fill="url(#eptRiv)"/>
             <path class="ept-river-line" id="ept-river-line" d="${lineD}" fill="none" stroke="${col.accent}" stroke-width="2.2" stroke-linejoin="round"/>
             ${dots}${xlab}
             <line class="ept-river-cursor" id="ept-river-cursor" x1="${X(n - 1).toFixed(1)}" y1="${topY}" x2="${X(n - 1).toFixed(1)}" y2="${baseY}" stroke="var(--ink)" stroke-opacity="0.4" stroke-width="1" stroke-dasharray="3 3"/>
             <g id="ept-river-knob" style="cursor:ew-resize"><line x1="${X(n - 1).toFixed(1)}" y1="${baseY}" x2="${X(n - 1).toFixed(1)}" y2="${baseY + 14}" stroke="var(--ink)" stroke-width="1"/><circle cx="${X(n - 1).toFixed(1)}" cy="${baseY + 16}" r="6" fill="var(--ink)"/></g>
             <rect class="ept-river-scrub" id="ept-river-scrub" x="0" y="0" width="${W}" height="${H}" fill="transparent" style="cursor:ew-resize"/>
           </svg>
         </div>
         <div class="ept-zoom-side">
           <div class="ept-strip-lbl">connected flows</div>${flows}
           <div class="ept-zoom-stats"><span>p50 <b>${fmtMs(e.p50)}</b></span><span>p95 <b>${fmtMs(e.p95)}</b></span><span>err <b>${e.err_pct}%</b></span><span>IPs <b>${e.fanout}</b></span></div>
           <button class="tfx-drill-btn" id="ept-zoom-req">view ${fmtNum(e.n)} requests →</button>
         </div>
       </div>`;
    zoom.hidden = false;
    try { zoom.animate([{ opacity: 0, transform: "scale(.94)" }, { opacity: 1, transform: "none" }], { duration: 320, easing: "cubic-bezier(.22,1,.36,1)" }); } catch (_) {}
    const back = $("ept-zoom-back"); if (back) back.addEventListener("click", _exitRegion);
    const rq = $("ept-zoom-req"); if (rq) rq.addEventListener("click", () => openReqList({ label: e.path, path: e.path }));
    zoom.querySelectorAll(".ept-zoom-flow[data-path]").forEach(f => f.addEventListener("click", () => { const ne = (DATA.endpoints || []).find(x => x.path === f.getAttribute("data-path")); if (ne) _enterRegion(ne); }));
    // intro: line draws in, area + dots fade up
    const lineEl = zoom.querySelector("#ept-river-line"), areaEl = zoom.querySelector("#ept-river-area");
    try { const L = lineEl.getTotalLength(); lineEl.style.strokeDasharray = L; lineEl.animate([{ strokeDashoffset: L }, { strokeDashoffset: 0 }], { duration: 900, easing: "cubic-bezier(.22,1,.36,1)" }).finished.then(() => { lineEl.style.strokeDasharray = ""; }).catch(() => { lineEl.style.strokeDasharray = ""; }); } catch (_) {}
    try { areaEl.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 800, easing: "ease" }); } catch (_) {}
    zoom.querySelectorAll(".ept-river-dot").forEach((d, i) => { try { d.animate([{ opacity: 0, transform: "scale(0)" }, { opacity: 1, transform: "scale(1)" }], { duration: 360, delay: 500 + i * 60, easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }); } catch (_) {} });
    // scrubber: drag (or hover) → cursor + knob + readout; click a slice → drill
    const svg = zoom.querySelector(".ept-river-svg"), cursor = zoom.querySelector("#ept-river-cursor"), knob = zoom.querySelector("#ept-river-knob"), read = $("ept-river-read"), scrub = $("ept-river-scrub");
    let dragging = false;
    const _bucketAt = (clientX) => { const r = svg.getBoundingClientRect(); const fx = (clientX - r.left) / r.width; return Math.max(0, Math.min(n - 1, Math.round(fx * (n - 1)))); };
    const _scrubTo = (i, ev, showTip) => {
      const x = X(i), v = spark[i] || 0, er = sparkErr[i] || 0;
      cursor.setAttribute("x1", x.toFixed(1)); cursor.setAttribute("x2", x.toFixed(1)); cursor.setAttribute("stroke-opacity", "0.55");
      knob.querySelector("line").setAttribute("x1", x.toFixed(1)); knob.querySelector("line").setAttribute("x2", x.toFixed(1)); knob.querySelector("circle").setAttribute("cx", x.toFixed(1));
      const iso = new Date(startMs + i * dur).toISOString().slice(0, 19).replace("T", " ");
      if (read) read.innerHTML = `${esc(_ago(iso) || "—")} · <b>${fmtNum(v)}</b> req${er ? ` · <span style="color:var(--c-danger,#b3402f)">${er} err</span>` : ""}`;
      if (showTip && ev) tip(true, ev.clientX, ev.clientY, `<b>${esc(_ago(iso) || "bucket " + (i + 1))}</b><br>${fmtNum(v)} request${v === 1 ? "" : "s"}${er ? ` · <span style="color:var(--c-danger,#b3402f)">${er} error${er === 1 ? "" : "s"}</span>` : ""}${v ? `<br><span style="opacity:.6">click → these requests</span>` : ""}`);
    };
    _scrubTo(n - 1, null, false);
    scrub.addEventListener("pointerdown", (ev) => { dragging = true; scrub.setPointerCapture(ev.pointerId); _scrubTo(_bucketAt(ev.clientX), ev, true); });
    scrub.addEventListener("pointermove", (ev) => { _scrubTo(_bucketAt(ev.clientX), ev, true); });
    scrub.addEventListener("pointerup", (ev) => { const i = _bucketAt(ev.clientX); dragging = false; if (spark[i]) { const s = startMs + i * dur; openReqList({ label: e.path + " · slice", path: e.path, sinceMs: s, untilMs: s + dur }); } });
    scrub.addEventListener("pointerleave", () => { tip(false); });
    _renderStrip(e.path);
  }
  function _exitRegion() {
    const zoom = $("ept-zoom"); if (!zoom) return; _terrZoom = null;
    try { zoom.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 200, easing: "ease" }).finished.then(() => { zoom.hidden = true; }); } catch (_) { zoom.hidden = true; }
  }

  // ════════════════════ EXPANDS (lightbox) ════════════════════════════════
  function _openExpand(kind) {
    if (!window.APIN || !APIN.lightbox || !DATA) return;
    const focusEp = (DATA.endpoints || []).find(x => x.path === _focus);
    const titles = { treemap: "Living endpoint terrain", profile: "Endpoint profile" + (focusEp ? " · " + _pathTail(focusEp.path) : ""), galaxy: "API galaxy" };
    APIN.lightbox.open({
      title: titles[kind] + (kind !== "profile" && DATA.key && DATA.key.name ? " · " + DATA.key.name : ""),
      subtitle: WIN_LABEL[WIN], hashKey: "epx-" + kind, sourceCard: $("pane-endpoints"),
      build: (panel) => {
        if (kind === "treemap") _buildTerrain(panel);
        else if (kind === "profile") _buildProfile(panel);
        else _buildGalaxy(panel);
      },
    });
  }

  // ════════════════════ EXPANDED API GALAXY ═══════════════════════════════
  // Centre = full orbital network (zoom/pan) · left = Common Routes ·
  // right = Behavior Inspector · top = view toggles + Route Replay.
  let _gx = null;
  function _buildGalaxy(panel) {
    _gxReplaying = false; _gxSel = null;
    panel.innerHTML =
      `<div class="gxx" id="gxx">
         <div class="gxx-top">
           <div class="ov-range gxx-modes" id="gxx-modes">
             <button data-m="flow" aria-pressed="true"><svg class="icon" aria-hidden="true"><use href="#i-route"/></svg> flow</button>
             <button data-m="health"><svg class="icon" aria-hidden="true"><use href="#i-activity"/></svg> health</button>
             <button data-m="latency"><svg class="icon" aria-hidden="true"><use href="#i-gauge"/></svg> latency</button>
           </div>
           <span class="gxx-hint" id="gxx-hint">drag to pan · scroll to zoom · hover a node · click a route to replay</span>
         </div>
         <div class="gxx-body">
           <div class="gxx-stage" id="gxx-stage"></div>
           <aside class="gxx-side">
             <div class="gxx-inspector" id="gxx-inspector"><div class="gxx-col-h">Behavior</div><div class="gxx-insp-empty">hover or select a node</div></div>
             <div class="gxx-routes" id="gxx-routes"><div class="gxx-col-h">Routes</div><div class="tf-empty" style="padding:18px 6px">reading journeys…</div></div>
           </aside>
         </div>
         <div class="gxx-insight" id="gxx-insight"></div>
         <div class="gxx-replay" id="gxx-replay" hidden></div>
       </div>`;
    const stage = $("gxx-stage"); if (!stage) return;
    const eps = DATA.endpoints || [];
    _ensureRoutes().then(() => {
      if (!stage.isConnected) return;
      _gx = APIN.galaxy.create(stage, {
        mode: "full", endpoints: eps, edges: _galaxyEdges(), hub: DATA._hub, focus: _focus,
        onHoverNode: (path, x, y) => { const e = eps.find(z => z.path === path); if (!e) return; const it = _intelOf(e);
          tip(true, x, y, `<b>${esc(e.path)}</b><br>${fmtNum(e.n)} req · ${e.err_pct}% err · p95 ${fmtMs(e.p95)}<br><span style="opacity:.6">${esc((it.weather || {}).state || "")}</span>`); _gxInspect(path); },
        onHoverEdge: (rec, x, y) => tip(true, x, y, `<b>${esc(_pathTail(rec.from))}</b> → <b>${esc(_pathTail(rec.to))}</b><br>${rec.count}× in sequence`),
        onLeave: () => tip(false),
        onClickNode: (path) => { _gxSel = path; _gxInspect(path); const e = eps.find(z => z.path === path); if (e) { /* double-acts as drill via inspector button */ } },
      });
      _gxRenderRoutes();
      _gxInsight();
    });
    panel.querySelectorAll("#gxx-modes button").forEach(b => b.addEventListener("click", () => {
      const m = b.getAttribute("data-m");
      panel.querySelectorAll("#gxx-modes button").forEach(x => { if (x.getAttribute("data-m") === m) x.setAttribute("aria-pressed", "true"); else x.removeAttribute("aria-pressed"); });
      const h = $("gxx-hint"); if (h) h.textContent = m === "health" ? "node colour = error rate · stress halos spread from erroring endpoints" : m === "latency" ? "tint = p95 vs SLO · find the slow paths" : "particles flow by volume · click a route to replay";
      if (_gx) _gx.setMode(m);
    }));
  }
  let _gxSel = null;
  function _gxInspect(path) {
    const host = $("gxx-inspector"); const e = (DATA.endpoints || []).find(x => x.path === path); if (!host || !e) return;
    const it = _intelOf(e), wx = it.weather, tone = wx.tone || "ok";
    const T = ({ metadata: 300, quick_inference: 8000, heavy_inference: 15000, default: 2000 })[e.cls] || 2000;
    const flowClass = e.n > 0 ? (it.traffic_share >= 25 ? "river" : it.traffic_share >= 8 ? "stream" : "trickle") : "dry";
    const latBand = (e.p95 || 0) < T ? "within SLO" : (e.p95 || 0) < T * 2 ? "elevated" : "over SLO";
    host.innerHTML = `<div class="gxx-col-h">Behavior</div>
      <div class="gxx-insp" data-tone="${tone}">
        <div class="gxx-insp-path"><span class="meth ${_methClass(e.method)}">${esc(e.method)}</span> ${esc(_pathTail(e.path))}</div>
        <div class="gxx-insp-row"><span>route health</span><b class="wx-${tone}">${esc(wx.state)}</b></div>
        <div class="gxx-insp-row"><span>traffic flow</span><b>${esc(flowClass)} · ${fmtNum(e.n)}</b></div>
        <div class="gxx-insp-row"><span>latency</span><b>${fmtMs(e.p95)} · ${esc(latBand)}</b></div>
        <div class="gxx-insp-row"><span>errors</span><b>${e.err_pct}%</b></div>
        <p class="gxx-insp-note">${esc((wx.reasons || [])[0] || "")}</p>
        <button class="tfx-drill-btn" id="gxx-insp-prof">open profile →</button>
        <button class="tfx-drill-btn ghost" id="gxx-insp-req">view requests →</button>
      </div>`;
    const pf = $("gxx-insp-prof"), rq = $("gxx-insp-req");
    if (pf) pf.addEventListener("click", () => { _focus = path; _pinned = true; APIN.lightbox && APIN.lightbox.close && APIN.lightbox.close(); setTimeout(() => _openExpand("profile"), 180); });
    if (rq) rq.addEventListener("click", () => openReqList({ label: e.path, path: e.path }));
  }
  function _gxRenderRoutes() {
    const host = $("gxx-routes"); if (!host) return;
    const routes = DATA._routes || [];
    if (!routes.length) { host.innerHTML = `<div class="gxx-col-h">Routes</div><div class="tf-empty" style="padding:18px 6px">No multi-step journeys yet — routes appear once callers chain requests.</div>`; return; }
    const maxShare = Math.max(1, ...routes.map(r => r.share));
    host.innerHTML = `<div class="gxx-col-h">Routes <span class="gxx-col-aux">${fmtNum(DATA._sessions || 0)} sessions</span></div>`
      + routes.map((r, i) => {
        const gaps = (r.steps || []).map(s => s.dt_ms).filter(d => d > 0);
        const avg = gaps.length ? Math.round(gaps.reduce((a, b) => a + b, 0) / gaps.length) : 0;
        const expl = `A ${r.seq.length}-step journey · ${r.share}% of sessions (${r.count}×)${avg ? " · avg gap " + fmtMs(avg) : ""}`;
        const steps = (r.steps && r.steps.length) ? r.steps : r.seq.map(p => ({ path: p }));
        return `<div class="gxx-route" data-i="${i}">
          <button class="gxx-route-head">
            <div class="gxx-route-seq">${r.seq.map((p, j) => `${j ? '<span class="gxx-arrow">→</span>' : ""}<span>${esc(_short(p))}</span>`).join("")}</div>
            <div class="gxx-route-bar"><span style="width:${(r.share / maxShare * 100).toFixed(0)}%"></span></div>
            <div class="gxx-route-meta">${expl}</div>
          </button>
          <div class="gxx-route-exp" hidden>
            <ol class="gxx-route-steps">${steps.map((s, j) => `<li data-j="${j}"><span class="gxx-step-n">${j + 1}</span><span class="gxx-step-p">${esc(_short(s.path))}</span>${s.status ? `<span class="gxx-step-sc epx-sc-${s.status >= 500 ? "danger" : s.status >= 400 ? "amber" : "ok"}">${s.status}</span>` : ""}${s.latency_ms != null ? `<span class="gxx-step-lat">${fmtMs(s.latency_ms)}</span>` : ""}${s.dt_ms ? `<span class="gxx-step-dt">+${fmtMs(s.dt_ms)}</span>` : ""}</li>`).join("")}</ol>
            <button class="tfx-drill-btn gxx-route-replay"><svg class="icon" aria-hidden="true"><use href="#i-play"/></svg> replay this route</button>
          </div>
        </div>`;
      }).join("");
    host.querySelectorAll(".gxx-route[data-i]").forEach(div => {
      const r = routes[+div.getAttribute("data-i")];
      const head = div.querySelector(".gxx-route-head"), exp = div.querySelector(".gxx-route-exp");
      const steps = (r.steps && r.steps.length) ? r.steps : r.seq.map(p => ({ path: p }));
      head.addEventListener("mouseenter", () => { if (_gx && !_gxReplaying) { _gx.highlightRoute(r.seq); _gx.traceRoute(steps); } });
      head.addEventListener("mouseleave", () => { if (_gx && !_gxReplaying) _gx.clearHighlight(); });
      head.addEventListener("click", () => {
        const open = !exp.hidden;
        host.querySelectorAll(".gxx-route-exp").forEach(e => e.hidden = true);
        host.querySelectorAll(".gxx-route").forEach(d => d.classList.remove("is-open"));
        if (!open) { exp.hidden = false; div.classList.add("is-open"); if (_gx) _gx.highlightRoute(r.seq); }
      });
      exp.querySelectorAll(".gxx-route-steps li[data-j]").forEach(li => li.addEventListener("click", () => {
        const s = steps[+li.getAttribute("data-j")]; if (s) openReqList({ label: r.seq.map(_short).join("→") + " · " + _short(s.path), path: s.path });
      }));
      const rp = exp.querySelector(".gxx-route-replay");
      if (rp) rp.addEventListener("click", (ev) => { ev.stopPropagation(); _gxReplay(r); });
    });
  }
  function _gxInsight() {
    const host = $("gxx-insight"); if (!host) return;
    const eps = DATA.endpoints || [], routes = DATA._routes || [];
    const bits = [];
    if (DATA._hub) { const hub = eps.find(e => e.path === DATA._hub); const deg = (DATA._edges || []).filter(l => l.from === DATA._hub || l.to === DATA._hub).length; if (hub) bits.push(`<b>${esc(_pathTail(DATA._hub))}</b> is the hub — ${deg} connected pathways`); }
    if (routes[0]) bits.push(`hottest route: ${routes[0].seq.map(_short).join(" → ")} (${routes[0].share}% of sessions)`);
    const worst = eps.slice().filter(e => e.err_rate > 0.05).sort((a, b) => b.err_rate - a.err_rate)[0];
    if (worst) bits.push(`errors concentrate on <b>${esc(_pathTail(worst.path))}</b> (${worst.err_pct}%)`);
    host.innerHTML = bits.length ? bits.map(b => `<span class="gxx-ins">${b}</span>`).join("") : "";
  }
  // ── route replay (real session, time-compressed) ──
  let _gxReplaying = false;
  function _gxReplay(route) {
    if (!_gx) return;
    const steps = (route.steps && route.steps.length >= 2) ? route.steps : route.seq.map(p => ({ path: p, dt_ms: 0 }));
    _gxReplaying = true;
    const rc = $("gxx-replay"); if (rc) rc.hidden = false;
    let speed = 1;
    const renderControls = (idx) => {
      if (!rc) return;
      rc.innerHTML = `<div class="gxx-rp-head"><b>Replaying</b> ${route.seq.map(_short).join(" → ")}<button class="gxx-rp-x" id="gxx-rp-x"><svg class="icon" aria-hidden="true"><use href="#i-x"/></svg></button></div>
        <div class="gxx-rp-steps">${steps.map((s, j) => `<span class="gxx-rp-step${j <= idx ? " on" : ""}" data-j="${j}">${esc(_short(s.path))}${s.status ? ` · ${s.status}` : ""}</span>${j < steps.length - 1 ? '<i class="gxx-arrow">→</i>' : ""}`).join("")}</div>
        <div class="gxx-rp-ctl">
          <input type="range" id="gxx-rp-scrub" min="0" max="1000" value="${Math.round((idx / Math.max(1, steps.length - 1)) * 1000)}">
          <div class="gxx-rp-speed" id="gxx-rp-speed">${[0.5, 1, 2].map(s => `<button data-s="${s}"${s === speed ? ' aria-pressed="true"' : ""}>${s}×</button>`).join("")}</div>
        </div>`;
      const x = $("gxx-rp-x"); if (x) x.addEventListener("click", _gxStopReplay);
      const scrub = $("gxx-rp-scrub"); if (scrub) scrub.addEventListener("input", () => { if (_gx) _gx.setReplayPos(+scrub.value / 1000); });
      rc.querySelectorAll("#gxx-rp-speed button").forEach(b => b.addEventListener("click", () => { speed = +b.getAttribute("data-s"); rc.querySelectorAll("#gxx-rp-speed button").forEach(x => x.removeAttribute("aria-pressed")); b.setAttribute("aria-pressed", "true"); if (_gx) _gx.setReplaySpeed(speed); }));
      rc.querySelectorAll(".gxx-rp-step[data-j]").forEach(s => s.addEventListener("click", () => { const j = +s.getAttribute("data-j"); const st = steps[j]; if (st) openReqList({ label: route.seq.map(_short).join("→") + " · " + _short(st.path), path: st.path }); }));
    };
    renderControls(0);
    _gx.replay(steps, { speed, onStep: (idx) => renderControls(idx), onDone: () => {} });
  }
  function _gxStopReplay() { _gxReplaying = false; if (_gx) { _gx.stopReplay(); _gx.clearHighlight(); } const rc = $("gxx-replay"); if (rc) { rc.hidden = true; rc.innerHTML = ""; } }

  // ════════════════════ EXPANDED ENDPOINT PROFILE ═════════════════════════
  // 3-zone identity view: Hero (genome + weather) · Identity + Behavior ·
  // Constellation + Life Story. Neighbour click morphs the profile in place.
  let _profGn = null;
  function _buildProfile(panel) {
    panel.innerHTML = `<div class="epp" id="epp"><div class="tf-empty" style="padding:60px">reading endpoint profile…</div></div>`;
    _profileLoad(panel, _focus);
  }
  async function _profileLoad(panel, path) {
    const host = panel.querySelector("#epp") || panel;
    try { host.animate([{ opacity: 1 }, { opacity: 0.3 }], { duration: 140 }); } catch (_) {}
    const { ok, body } = await api(`/api/account/keys/${encodeURIComponent(PID)}/endpoint-profile?path=${encodeURIComponent(path)}&window=${WIN}&tz_off=${TZOFF}`);
    const prof = (body && (body.data || body)) || {};
    if (!ok || prof.not_found || prof.empty) {
      host.innerHTML = `<div class="tf-empty" style="padding:60px">No traffic for ${esc(_pathTail(path))} in ${esc(WIN_LABEL[WIN])}.</div>`;
      return;
    }
    const listEp = (DATA.endpoints || []).find(x => x.path === path) || prof;
    const it = _intelOf(listEp);
    _renderProfile(panel, prof, it);
  }
  function _renderProfile(panel, p, it) {
    const host = panel.querySelector("#epp") || panel;
    const wx = it.weather, tone = wx.tone || "ok";
    const traits = it.traits || {};
    host.innerHTML =
      `<div class="epp-wrap" data-tone="${tone}">
        <div class="epp-hero">
          <div class="epp-hero-gn" id="epp-gn"></div>
          <div class="epp-hero-meta">
            <div class="epp-hero-path"><span class="meth ${_methClass(p.method)}">${esc(p.method)}</span> ${esc(p.path)}</div>
            <div class="epp-hero-tags">
              <span class="epp-tag">${esc(it.archetype || "")}</span>
              <span class="epp-tag epp-tag-wx wx-${tone}" id="epp-wx">${_wxMotif(wx.state)} ${esc(wx.state)}</span>
              <span class="epp-tag">${it.traffic_share != null ? it.traffic_share + "% of traffic" : ""}</span>
            </div>
            <p class="epp-insight">${esc(_profileInsight(p, it))}</p>
          </div>
        </div>
        <div class="epp-cols">
          <div class="epp-col epp-identity">
            <h4>Identity</h4>
            <div class="epp-blk"><span class="epp-blk-h">${esc(wx.state)} — operational weather</span>
              ${(wx.intel || []).map(s => `<div class="epp-li">${esc(s)}</div>`).join("")}</div>
            <div class="epp-blk"><span class="epp-blk-h">Archetype · ${esc(it.archetype || "")}</span>
              ${(it.archetype_traits || []).map(s => `<div class="epp-li">${esc(s)}</div>`).join("")}</div>
            <div class="epp-blk"><span class="epp-blk-h">Fingerprint traits</span>
              <div class="epp-li">Branch complexity <b>${traits.branch_complexity ?? "—"}</b></div>
              <div class="epp-li">Payload density <b>${esc(traits.payload_density || "—")}</b></div>
              <div class="epp-li">Flow stability <b>${traits.flow_stability ?? "—"}%</b></div>
            </div>
            <div class="epp-blk" id="epp-gn-legend"><span class="epp-blk-h">What the sigil shows</span></div>
          </div>
          <div class="epp-col epp-behavior">
            <h4>Behavior</h4>
            ${_profileInstruments(p)}
            <div class="epp-blk"><span class="epp-blk-h">Top callers</span>
              ${(p.callers || []).map(c => `<div class="epp-caller"><span>${esc(c.name)}</span><b>${fmtNum(c.n)}</b><i>${c.err_pct}%</i></div>`).join("") || `<div class="epp-li">no caller data</div>`}
            </div>
          </div>
        </div>
        <div class="epp-constellation">
          <h4>Constellation <span class="epp-h-aux">what calls this · what it calls</span></h4>
          <div class="epp-const-stage" id="epp-const"></div>
        </div>
        <div class="epp-story">
          <h4>Life story <span class="epp-h-aux">${esc(WIN_LABEL[WIN])} · click a moment → its requests</span></h4>
          <div class="epp-story-stage" id="epp-story"></div>
        </div>
      </div>`;
    // hero genome (large)
    const gnHost = host.querySelector("#epp-gn");
    if (gnHost && APIN.genome) {
      if (_profGn) { try { _profGn.destroy(); } catch (_) {} }
      _profGn = APIN.genome.mount(gnHost, it.genome, { tone, strokeBase: 2.0 });
      const lg = host.querySelector("#epp-gn-legend");
      if (lg && _profGn.legend) _profGn.legend().forEach(l => { const d = document.createElement("div"); d.className = "epp-li"; d.innerHTML = `<b>${esc(l.k)}</b> — ${esc(l.v)}`; lg.appendChild(d); });
      const txt = _genomeTip(_profGn, p);
      gnHost.addEventListener("mousemove", (ev) => tip(true, ev.clientX, ev.clientY, txt));
      gnHost.addEventListener("mouseleave", () => tip(false));
    }
    // throughput scrub
    const thru = host.querySelector("#epp-thru .ic-area"), s0 = p.series || {};
    _wireSparkScrub(thru, s0.n || [], s0.start_ms || Date.now(), s0.bucket_ms || 60000,
      (a, b) => openReqList({ label: p.path + " · slice", path: p.path, sinceMs: a, untilMs: b }),
      host.querySelector("#epp-thru-read"));
    // weather hovercard
    const wxEl = host.querySelector("#epp-wx");
    if (wxEl) { const card = `<b>${esc(wx.state)} — operational weather</b><br>` + (wx.reasons || []).map(r => "· " + esc(r)).join("<br>");
      wxEl.addEventListener("mousemove", (ev) => tip(true, ev.clientX, ev.clientY, card));
      wxEl.addEventListener("mouseleave", () => tip(false)); }
    _renderConstellation(host.querySelector("#epp-const"), p, panel);
    _renderLifeStory(host.querySelector("#epp-story"), p);
    try { host.querySelector(".epp-wrap").animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 240, easing: "ease" }); } catch (_) {}
  }
  function _profileInsight(p, it) {
    const wx = it.weather || {}, bits = [];
    if (wx.state) bits.push(`${wx.state.toLowerCase()} right now`);
    if (p.p95) bits.push(`p95 ${fmtMs(p.p95)}`);
    if (p.err_pct) bits.push(`${p.err_pct}% errors`);
    const ev = (p.events || [])[((p.events || []).length - 1)];
    if (ev) bits.push(`last event: ${ev.label.toLowerCase()}`);
    return bits.length ? bits.join(" · ") : "steady, unremarkable traffic — a healthy endpoint.";
  }
  function _profileInstruments(p) {
    const col = _C(), T = _SLO[p.cls] || 2000;
    const errFill = p.err_pct >= 10 ? 5 : p.err_pct >= 5 ? 4 : p.err_pct >= 2 ? 3 : p.err_pct > 0.5 ? 2 : p.err_pct > 0 ? 1 : 0;
    const sq = Array.from({ length: 5 }, (_, i) => `<span class="ic-sq${i < errFill ? " on" : ""}"></span>`).join("");
    const latR = Math.min(1, (p.p95 || 0) / (T * 2));
    return `<div class="epp-thru">
        <div class="ic-lbl epp-thru-h"><span>throughput</span><span class="epp-thru-read" id="epp-thru-read"><b>${fmtNum(p.n)}</b> req total</span></div>
        <div class="epp-thru-chart" id="epp-thru">${_areaSpark((p.series || {}).n || [], 640, 56, col.accent, { scrub: true })}</div>
      </div>
      <div class="ic-grid epp-ic">
        <div class="ic"><span class="ic-lbl">latency</span><div class="ic-gauge"><div class="ic-gauge-fill" style="width:${(latR * 100).toFixed(0)}%"></div></div><b class="ic-val">${fmtMs(p.p50)} · ${fmtMs(p.p95)} · ${fmtMs(p.p99)}<span>p50·p95·p99</span></b></div>
        <div class="ic"><span class="ic-lbl">errors</span><div class="ic-sqs">${sq}</div><b class="ic-val">${p.err_pct}%</b></div>
        <div class="ic"><span class="ic-lbl">payload</span><b class="ic-val">${fmtBytes(p.bytes_in)} in · ${fmtBytes(p.bytes_out)} out</b></div>
      </div>`;
  }
  function _renderConstellation(host, p, panel) {
    if (!host) return;
    const col = _C(), c = p.constellation || { incoming: [], outgoing: [], callers: 0 };
    const W = 560, H = 240, cx = W / 2, cy = H / 2;
    const inc = (c.incoming || []).slice(0, 4), out = (c.outgoing || []).slice(0, 4);
    const place = (arr, side) => arr.map((l, i) => {
      const span = Math.max(1, arr.length);
      const y = (H / (span + 1)) * (i + 1);
      const x = side < 0 ? 70 : W - 70;
      return { x, y, l };
    });
    const L = place(inc, -1), R = place(out, 1);
    let lines = "", nodes = "", packets = "";
    const mkLine = (x, y, toRight) => `<path d="M ${toRight ? cx + 30 : x} ${toRight ? cy : y} Q ${(x + cx) / 2} ${(y + cy) / 2 - 18} ${toRight ? x : cx - 30} ${toRight ? y : cy}" fill="none" stroke="${col.edge}" stroke-width="1.4" opacity="0.6"/>`;
    L.forEach(n => { lines += mkLine(n.x, n.y, false); });
    R.forEach(n => { lines += mkLine(n.x, n.y, true); });
    const node = (x, y, label, sub, path) => `<g class="epp-cn-node" data-path="${esc(path || "")}" style="cursor:${path ? "pointer" : "default"}">`
      + `<circle cx="${x}" cy="${y}" r="6" fill="${col.accent}"/>`
      + `<text x="${x}" y="${y - 11}" text-anchor="middle" style="font:600 9.5px 'JetBrains Mono',monospace;fill:${col.ink}">${esc(_pathTail(label))}</text>`
      + `<text x="${x}" y="${y + 17}" text-anchor="middle" style="font:8px 'JetBrains Mono',monospace;fill:${col.mute}">${esc(sub)}</text></g>`;
    L.forEach(n => { nodes += node(n.x, n.y, n.l.path, n.l.count + "× →", n.l.path); });
    R.forEach(n => { nodes += node(n.x, n.y, n.l.path, "→ " + n.l.count + "×", n.l.path); });
    // centre (this endpoint) + callers aggregate
    nodes += `<g><circle cx="${cx}" cy="${cy}" r="13" fill="${col.accent}" fill-opacity="0.16" stroke="${col.accent}" stroke-width="2"/>`
      + `<text x="${cx}" y="${cy + 3.5}" text-anchor="middle" style="font:600 9px 'JetBrains Mono',monospace;fill:${col.ink}">${esc(_pathTail(p.path))}</text></g>`;
    if (!inc.length) nodes += `<text x="70" y="${cy}" text-anchor="middle" style="font:9px 'JetBrains Mono',monospace;fill:${col.mute}">${c.callers} caller IPs</text>`;
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;max-height:260px">${lines}${nodes}</svg>`;
    // travelling packets along each line (WAAPI loop)
    [...L.map(n => [n.x, n.y, false]), ...R.map(n => [n.x, n.y, true])].forEach(([x, y, toRight], k) => {
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("r", "2.4"); dot.setAttribute("fill", col.accent);
      host.querySelector("svg").appendChild(dot);
      const from = toRight ? [cx + 30, cy] : [x, y], to = toRight ? [x, y] : [cx - 30, cy];
      try {
        dot.animate([{ cx: from[0], cy: from[1], opacity: 0 }, { offset: 0.15, opacity: 1 }, { offset: 0.85, opacity: 1 }, { cx: to[0], cy: to[1], opacity: 0 }],
          { duration: 1800 + k * 120, iterations: Infinity, delay: k * 220 });
      } catch (_) {}
    });
    host.querySelectorAll(".epp-cn-node[data-path]").forEach(nd => {
      const path = nd.getAttribute("data-path"); if (!path) return;
      nd.addEventListener("click", () => { _focus = path; _profileLoad(panel, path); });   // morph in place
    });
  }
  function _renderLifeStory(host, p) {
    if (!host) return;
    const col = _C(), s = p.series || { n: [], err: [], bucket_ms: 60000, start_ms: Date.now() };
    const series = s.n || [], nB = series.length || 1;
    const W = 720, H = 132, padX = 14, baseY = H - 18, topY = 16, plotH = baseY - topY;
    const max = Math.max(1, ...series);
    const X = (i) => padX + (nB <= 1 ? 0.5 : i / (nB - 1)) * (W - 2 * padX);
    const Y = (v) => baseY - (v / max) * plotH;
    let line = "", area = `M ${X(0).toFixed(1)} ${baseY}`;
    series.forEach((v, i) => { const x = X(i).toFixed(1), y = Y(v).toFixed(1); line += (i ? "L" : "M") + ` ${x} ${y} `; area += ` L ${x} ${y}`; });
    area += ` L ${X(nB - 1).toFixed(1)} ${baseY} Z`;

    // ── build an honest activity log: anomalies + derived milestones ──
    const tsOf = (i) => s.start_ms + i * s.bucket_ms;
    const acts = (p.events || []).slice();
    const sumAll = series.reduce((a, b) => a + b, 0);
    if (sumAll > 0) {
      const firstI = series.findIndex(v => v > 0);
      let lastI = -1; for (let i = nB - 1; i >= 0; i--) if (series[i] > 0) { lastI = i; break; }
      const peakI = series.indexOf(Math.max(...series));
      if (firstI >= 0) acts.push({ ts: tsOf(firstI), kind: "start", label: "First activity", detail: `traffic begins · ${fmtNum(series[firstI])} req` });
      if (peakI >= 0) acts.push({ ts: tsOf(peakI), kind: "peak", label: "Busiest moment", detail: `${fmtNum(series[peakI])} requests — the window's peak` });
      if (lastI >= 0 && lastI !== firstI) acts.push({ ts: tsOf(lastI), kind: "recent", label: "Most recent activity", detail: `${fmtNum(series[lastI])} req` });
    }
    acts.sort((a, b) => a.ts - b.ts);
    const merged = [];
    acts.forEach(a => { const l = merged[merged.length - 1]; if (l && l.kind === a.kind && Math.abs(l.ts - a.ts) < s.bucket_ms * 1.5) return; merged.push(a); });
    const toneOf = (k) => k === "error" ? col.danger : k === "latency" ? col.amber : k === "recovery" ? col.soft : k === "peak" ? col.accent : k === "spike" ? col.accent : col.mute;
    const evX = (ts) => Math.max(padX, Math.min(W - padX, padX + ((ts - s.start_ms) / (nB * s.bucket_ms || 1)) * (W - 2 * padX)));
    let marks = "";
    merged.forEach((ev, i) => { const x = evX(ev.ts), t = toneOf(ev.kind);
      marks += `<g class="epp-ev" data-i="${i}" style="cursor:pointer"><line x1="${x.toFixed(1)}" y1="${topY}" x2="${x.toFixed(1)}" y2="${baseY}" stroke="${t}" stroke-width="1" stroke-dasharray="2 2" opacity="0.55"/><circle cx="${x.toFixed(1)}" cy="${topY}" r="4" fill="${t}"/></g>`; });

    host.innerHTML =
      `<div class="epp-story-top"><span class="epp-story-read" id="epp-story-read">drag across the timeline</span></div>`
      + `<svg class="epp-story-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:${H}px">`
      + `<line x1="${padX}" y1="${baseY}" x2="${W - padX}" y2="${baseY}" stroke="var(--paper-edge)"/>`
      + `<path d="${area}" fill="${col.accent}" fill-opacity="0.12"/>`
      + `<path d="${line}" fill="none" stroke="${col.accent}" stroke-width="1.6"/>`
      + marks
      + `<line class="epp-story-cursor" id="epp-story-cursor" x1="-1" y1="${topY}" x2="-1" y2="${baseY}" stroke="${col.ink}" stroke-opacity="0.5" stroke-width="1" stroke-dasharray="3 3"/>`
      + `<rect class="epp-story-hit" id="epp-story-hit" x="0" y="0" width="${W}" height="${H}" fill="transparent" style="cursor:ew-resize"/>`
      + `</svg>`
      + `<div class="epp-log">` + (merged.length ? merged.map((ev, i) =>
        `<button class="epp-log-row" data-i="${i}"><span class="epp-log-dot epp-ev-${ev.kind}"></span><span class="epp-log-k">${esc(ev.label)}</span><span class="epp-log-d">${esc(ev.detail)}</span><span class="epp-log-t">${esc(_ago(_utcStr(ev.ts)) || "")}</span></button>`).join("")
        : `<div class="epp-li" style="padding:8px 4px">a calm window — no recorded activity.</div>`) + `</div>`;

    const drill = (ts) => openReqList({ label: p.path + " · slice", path: p.path, sinceMs: ts - s.bucket_ms, untilMs: ts + s.bucket_ms });
    host.querySelectorAll(".epp-ev[data-i], .epp-log-row[data-i]").forEach(el =>
      el.addEventListener("click", () => { const ev = merged[+el.getAttribute("data-i")]; if (ev) drill(ev.ts); }));
    // scrubber across the timeline
    const svg = host.querySelector(".epp-story-svg"), cur = host.querySelector("#epp-story-cursor"), read = host.querySelector("#epp-story-read"), hit = host.querySelector("#epp-story-hit");
    if (svg && hit) {
      const at = (cx) => { const r = svg.getBoundingClientRect(); return Math.max(0, Math.min(nB - 1, Math.round((cx - r.left) / r.width * (nB - 1)))); };
      const show = (i, ev) => { const x = X(i), v = series[i] || 0, ms = tsOf(i);
        if (cur) { cur.setAttribute("x1", x.toFixed(1)); cur.setAttribute("x2", x.toFixed(1)); cur.setAttribute("stroke-opacity", "0.6"); }
        if (read) read.innerHTML = `${esc(_ago(_utcStr(ms)) || "—")} · <b>${fmtNum(v)}</b> req`;
        if (ev) tip(true, ev.clientX, ev.clientY, `<b>${esc(_ago(_utcStr(ms)) || "moment")}</b><br>${fmtNum(v)} request${v === 1 ? "" : "s"}${v ? `<br><span style="opacity:.6">click → these requests</span>` : ""}`); };
      hit.addEventListener("mousemove", (ev) => show(at(ev.clientX), ev));
      hit.addEventListener("mouseleave", () => { tip(false); if (cur) cur.setAttribute("stroke-opacity", "0"); });
      hit.addEventListener("click", (ev) => { const i = at(ev.clientX); if (series[i]) drill(tsOf(i)); });
    }
  }

  // ════════════════════ LIVE (SSE) ════════════════════════════════════════
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
  function _applyEvent(ev) {
    const path = ev.path || "?", method = (ev.method || "?").toUpperCase(), sc = +ev.status_code || 0;
    const sk = sc >= 500 ? "n5" : sc >= 400 ? "n4" : "n2";
    let e = (DATA.endpoints || []).find(x => x.path === path);
    if (!e) { e = { path, method, methods: {}, cls: "default", n: 0, n2: 0, n4: 0, n5: 0, err_pct: 0, err_rate: 0, p50: +ev.latency_ms || 0, p95: +ev.latency_ms || 0, bytes_in: 0, bytes_out: 0, avg_bytes: 0, spark: new Array(24).fill(0), spark_err: new Array(24).fill(0), hours: new Array(24).fill(0), fanout: 0, retry: 0, burst: 0 }; (DATA.endpoints = DATA.endpoints || []).push(e); }
    e.n += 1; e[sk] += 1; e.methods[method] = (e.methods[method] || 0) + 1;
    e.bytes_in += (+ev.bytes_in || 0); e.bytes_out += (+ev.bytes_out || 0);
    e.avg_bytes = Math.round((e.bytes_in + e.bytes_out) / e.n);
    e.err_rate = (e.n4 + e.n5) / e.n; e.err_pct = Math.round(1000 * e.err_rate) / 10;
    if (e.spark && e.spark.length) { e.spark[e.spark.length - 1] += 1; if (sk !== "n2" && e.spark_err) e.spark_err[e.spark_err.length - 1] += 1; }
    if (e.hours) e.hours[new Date().getHours()] += 1;
    let mm = (DATA.matrix || []).find(x => x.method === method);
    if (!mm) { mm = { method, n2: 0, n4: 0, n5: 0, total: 0 }; (DATA.matrix = DATA.matrix || []).push(mm); }
    mm[sk] += 1; mm.total = mm.n2 + mm.n4 + mm.n5;
    DATA.total = (DATA.total || 0) + 1;
    e._live = true;   // panel self-derives intelligence from live counts
    // ── immediate "alive" ticks (realtime, not refresh-based) ──
    if (path === _focus) {
      if (_panelGn) { try { _panelGn.ripple(); if (sk !== "n2") _panelGn.flicker(); } catch (_) {} }
      const tnow = performance.now();
      // numbers/instruments/weather update almost immediately (throttled lightly)
      if (tnow - _intelNumT > 140) { _intelNumT = tnow; _repaintIntelLive(); }
      // the sigil itself grows / fractures live — slower throttle so ripples finish
      if (tnow - _intelGnT > 850 && _panelGn) {
        _intelGnT = tnow; const it2 = _intelOf(e);
        try { _panelGn.setMetrics(it2.genome, { tone: (it2.weather && it2.weather.tone) || "ok", strokeBase: 1.8 }); } catch (_) {}
      }
    }
    if (_terr3d) _terr3d.pulse(path);                                                            // 3D hill pulse
    if (_galaxy) { try { _galaxy.pulse(path); if (sk !== "n2") _galaxy.restyle(); } catch (_) {} } // constellation flow + recolor
    if (_terrState && _terrState.pings) _terrState.pings.push({ path, t: performance.now() });   // 2D edge pulse
    const tmCell = document.querySelector(`#ep-treemap-body .epx-tm-cell[data-path="${(path || "").replace(/"/g, "")}"]`);
    if (tmCell) { tmCell.classList.remove("epx-tm-ping"); void tmCell.offsetWidth; tmCell.classList.add("epx-tm-ping"); }
    // Card reconciles on a slower cadence (re-squarify is fine for the glance
    // card). The OPEN terrain modal deliberately keeps its layout stable — its
    // "alive" comes from the canvas pings/weather (live rAF) + strip numbers —
    // so the map never reshuffles under the user's cursor.
    if (_terrSel && APIN.lightbox && APIN.lightbox.isOpen()) _renderStrip(_terrSel);
    _dirty = true;
    if (!_liveTimer) _liveTimer = setTimeout(() => { _liveTimer = null; if (_dirty && _active) { _dirty = false; renderAll(false); } }, 1500);
  }

  // ════════════════════ WIRING / ACTIVATE ═════════════════════════════════
  function syncWinButtons() {
    document.querySelectorAll("#ep-range button").forEach(b => { if (b.getAttribute("data-win") === WIN) b.setAttribute("aria-pressed", "true"); else b.removeAttribute("aria-pressed"); });
  }
  function wire() {
    if (_wired) return; _wired = true;
    document.querySelectorAll("#ep-range button").forEach(b => b.addEventListener("click", () => {
      const w = b.getAttribute("data-win"); if (w === WIN) return;
      WIN = w; try { sessionStorage.setItem("ep_win", WIN); } catch (_) {}
      syncWinButtons(); const st = $("ep-stack"); if (st) st.classList.add("ov-loading"); refresh();
    }));
    const live = $("ep-live");
    if (live) live.addEventListener("click", () => { LIVE = !LIVE; live.setAttribute("data-on", LIVE ? "true" : "false"); const l = live.querySelector(".ov-live-label"); if (l) l.textContent = LIVE ? "live" : "paused"; if (LIVE) startSSE(); else stopSSE(); });
    const rf = $("ep-refresh");
    if (rf) rf.addEventListener("click", () => { rf.classList.add("is-spinning"); setTimeout(() => rf.classList.remove("is-spinning"), 760); refresh(); });
    const stack = $("ep-stack");
    if (stack) stack.addEventListener("click", (e) => { const b = e.target.closest(".ov-expand[data-epx]"); if (b) _openExpand(b.getAttribute("data-epx")); });
  }
  function activate(pid) {
    PID = pid || PID || (PID_META ? PID_META.content : null); _active = true;
    if (!$("ep-stack")) return;
    wire(); syncWinButtons();
    if (DATA) renderAll(false);
    refresh();
    if (LIVE) startSSE();
  }
  function deactivate() {
    _active = false; stopSSE();
    if (_liveTimer) { clearTimeout(_liveTimer); _liveTimer = null; }
    if (_terrRAF) { cancelAnimationFrame(_terrRAF); _terrRAF = null; }
    if (_terrReflowTimer) { clearInterval(_terrReflowTimer); _terrReflowTimer = null; }
    if (_terr3d) { try { _terr3d.dispose(); } catch (_) {} _terr3d = null; window.removeEventListener("resize", _terr3dResize); }
    if (_galaxy) { try { _galaxy.dispose(); } catch (_) {} _galaxy = null; _galaxyRef = null; }
    _terrState = null; _terrCore = null;
  }

  window.APIN = window.APIN || {};
  window.APIN.keyEndpoints = { activate, deactivate };
})();
