// 9.N.9 · Per-key Bento Overview — widget module.
//
// Owns the #ov-bento Overview tab. Renders 6 widgets from a single
// /api/account/keys/{id}/overview fetch, wires hover + click-to-expand
// (FLIP-grow lightbox), and a tiny OverviewBus pub/sub for the
// cross-widget "alive" layer (hover a ribbon tick → pulse the matching
// spark-grid row, etc.).
//
// Built to the per-key-dashboard.md spec, in the paper-ink UI language.
(function () {
  "use strict";

  // ── tiny event bus for cross-widget linking ──────────────────────────
  const Bus = (function () {
    const subs = {};
    return {
      on(ev, fn) { (subs[ev] = subs[ev] || []).push(fn); },
      emit(ev, payload) { (subs[ev] || []).forEach(fn => { try { fn(payload); } catch (_) {} }); },
    };
  })();

  // ── state ─────────────────────────────────────────────────────────────
  let PID = null;
  let RANGE = (function () { try { return sessionStorage.getItem("ov_range") || "24h"; } catch (_) { return "24h"; } })();
  let LIVE = true;
  let DATA = null;
  let _pollTimer = null;
  let _wired = false;
  let _active = false;

  // ── helpers ────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtNum = (n) => n == null ? "—" : Number(n).toLocaleString();
  const fmtMs = (n) => n == null ? "—" : (n < 1000 ? Math.round(n) + "ms" : (n / 1000).toFixed(2) + "s");
  const fmtPct = (n) => n == null ? "—" : Number(n).toFixed(n >= 99 ? 1 : 1) + "%";
  const statusBucket = (s) => s >= 500 ? "5xx" : s >= 400 ? "4xx" : "2xx";
  function api(path) {
    return fetch(path, { headers: { "Accept": "application/json" }, credentials: "same-origin" })
      .then(r => r.json().then(b => ({ ok: r.ok, body: b })).catch(() => ({ ok: false, body: null })))
      .catch(() => ({ ok: false, body: null }));
  }
  function fmtAgo(iso) {
    if (!iso) return "—";
    const t = Date.parse(String(iso).replace(" ", "T"));
    if (isNaN(t)) return "—";
    const s = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (s < 60) return s + "s ago";
    if (s < 3600) return Math.floor(s / 60) + "m ago";
    if (s < 86400) return Math.floor(s / 3600) + "h ago";
    return Math.floor(s / 86400) + "d ago";
  }

  // shared floating tooltip
  let _tipEl = null;
  function tip(show, x, y, html) {
    if (!_tipEl) { _tipEl = document.createElement("div"); _tipEl.className = "ov-tip"; document.body.appendChild(_tipEl); }
    if (!show) { _tipEl.classList.remove("show"); return; }
    _tipEl.innerHTML = html;
    _tipEl.style.left = Math.min(x + 12, window.innerWidth - 290) + "px";
    _tipEl.style.top = (y + 14) + "px";
    _tipEl.classList.add("show");
  }

  // ── data fetch + dispatch ────────────────────────────────────────────
  function _showError(msg) {
    // Replace skeletons with a visible error rather than spinning forever.
    [["ov-health-body", "gauge"], ["ov-personality-body", ""], ["ov-sparkgrid-body", ""], ["ov-insights-body", ""]].forEach(([id]) => {
      const el = $(id);
      if (el) el.innerHTML = `<div style="font:italic 12.5px/1.5 'Fraunces',serif;color:var(--c-danger,#b3402f);padding:14px 4px">${esc(msg)}</div>`;
    });
  }
  async function refresh() {
    if (!PID) return;
    const { ok, body } = await api(`/api/account/keys/${encodeURIComponent(PID)}/overview?window=${RANGE}`);
    if (!ok || !body || body.ok === false) {
      const detail = (body && body.error && body.error.message) || (body && body.detail) || "could not load overview data";
      _showError(detail);
      return;
    }
    // @api_endpoint wraps the route's return in {ok, data:{...}} — unwrap.
    const d = body.data || body;
    DATA = d;
    _liveReqCount = d.kpis && d.kpis.requests ? d.kpis.requests.value : null;  // re-sync live counter
    renderHealth(d.health);
    renderKpis(d.kpis);
    renderRibbon(d.ribbon);
    renderPersonality(d.personality);
    renderSparkGrid(d.spark_grid);
    renderInsights(d.insights);
  }

  // ════════════════════ WIDGET 1 · HEALTH SCORE ════════════════════════
  function arcPath(cx, cy, r, frac) {
    const a = Math.PI * 2 * Math.max(0, Math.min(1, frac));
    const x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
    const large = a > Math.PI ? 1 : 0;
    return `M ${cx + r} ${cy} A ${r} ${r} 0 ${large} 1 ${x} ${y}`;
  }
  let _healthFrac = 0;          // last drawn gauge fraction (for arc tween)
  let _healthArcRaf = null;
  const _toneColor = (t) => ({ great: "#2f6f3e", ok: "#7a9a3e", warn: "#c98a2b", bad: "#b3402f", nodata: "#9a907a" }[t] || "#7a9a3e");
  function _tweenArc(pathEl, from, to, color) {
    if (_healthArcRaf) cancelAnimationFrame(_healthArcRaf);
    const t0 = performance.now(), dur = 500;
    pathEl.setAttribute("stroke", color);
    (function step(now) {
      const k = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);          // easeOutCubic
      const f = from + (to - from) * eased;
      pathEl.setAttribute("d", arcPath(70, 70, 58, f));
      if (k < 1) _healthArcRaf = requestAnimationFrame(step);
      else _healthFrac = to;
    })(t0);
  }
  function renderHealth(h) {
    const host = $("ov-health-body");
    if (!host || !h) return;
    if (h.insufficient || h.composite == null) {
      host.innerHTML = `<div class="ov-gauge-wrap"><div class="ov-gauge">
        <svg width="140" height="140"><circle cx="70" cy="70" r="58" fill="none" stroke="var(--paper-edge)" stroke-width="10"/></svg>
        <div class="ov-gauge-num"><b>—</b></div></div></div>
        <div class="ov-health-headline">${esc(h.headline || "Awaiting first requests.")}</div>`;
      host._built = false; _healthFrac = 0;
      return;
    }
    const frac = h.composite / 100;
    const tone = h.tone || "ok";
    const color = _toneColor(tone);
    const P = h.pillars || {};
    const pscore = (k) => (P[k] && P[k].score != null) ? Math.round(P[k].score) : null;
    const ptone = (v) => v == null ? "nodata" : v >= 90 ? "great" : v >= 75 ? "ok" : v >= 60 ? "warn" : "bad";

    // Build-once; subsequent refreshes UPDATE in place so the arc tweens
    // and the number odometer-rolls instead of hard-redrawing.
    if (!host._built) {
      const pill = (k, lbl) => {
        const v = pscore(k);
        return `<div class="ov-pillar s-${ptone(v)}" data-pk="${k}"><div class="ov-pillar-lbl">${lbl}</div>
          <div class="ov-pillar-val">${v == null ? "—" : v}</div>
          <div class="ov-pillar-bar"><i style="width:${v || 0}%"></i></div></div>`;
      };
      host.innerHTML = `<div class="ov-gauge-wrap"><div class="ov-gauge" id="ov-gauge" tabindex="0">
          <svg width="140" height="140">
            <circle cx="70" cy="70" r="58" fill="none" stroke="var(--paper-edge)" stroke-width="10"/>
            <path id="ov-gauge-arc" d="${arcPath(70, 70, 58, 0)}" fill="none" stroke="${color}" stroke-width="10" stroke-linecap="round"/>
          </svg>
          <div class="ov-gauge-num"><b id="ov-gauge-num">${h.composite}</b><span id="ov-gauge-grade">${esc(h.grade)}${h.provisional ? " ~" : ""}</span></div>
        </div></div>
        <div class="ov-pillars">${pill("reliability", "REL")}${pill("performance", "PERF")}${pill("capacity", "CAP")}${pill("hygiene", "HYG")}</div>
        <div class="ov-health-headline" id="ov-health-headline">${esc(h.headline || "")}</div>`;
      host._built = true;
      const g = $("ov-gauge");
      if (g) {
        g.addEventListener("mousemove", (e) => {
          const r = P.reliability, p = P.performance, c = P.capacity, hy = P.hygiene;
          tip(true, e.clientX, e.clientY,
            `reliability ${r && r.score != null ? r.score : "—"} · performance ${p && p.score != null ? p.score : "—"}<br>capacity ${c && c.score != null ? c.score : "—"} · hygiene ${hy && hy.score != null ? hy.score : "—"}`);
        });
        g.addEventListener("mouseleave", () => tip(false));
      }
    }
    // Update dynamic bits
    const arc = $("ov-gauge-arc");
    if (arc) _tweenArc(arc, _healthFrac, frac, color);
    const numEl = $("ov-gauge-num");
    if (numEl) { if (window.APIN && APIN.odometer) APIN.odometer.roll(numEl, h.composite); else numEl.textContent = h.composite; }
    const gradeEl = $("ov-gauge-grade");
    if (gradeEl) gradeEl.textContent = esc(h.grade) + (h.provisional ? " ~" : "");
    ["reliability", "performance", "capacity", "hygiene"].forEach(k => {
      const el = host.querySelector(`.ov-pillar[data-pk="${k}"]`);
      if (!el) return;
      const v = pscore(k);
      el.className = "ov-pillar s-" + ptone(v);
      const val = el.querySelector(".ov-pillar-val"); if (val) val.textContent = v == null ? "—" : v;
      const bar = el.querySelector(".ov-pillar-bar i"); if (bar) bar.style.width = (v || 0) + "%";
    });
    const hl = $("ov-health-headline"); if (hl) hl.textContent = h.headline || "";
  }

  function expandHealth(panel) {
    const h = (DATA && DATA.health) || {};
    const P = h.pillars || {};
    const sec = (window.APIN && APIN.lightbox && APIN.lightbox.section);
    const bullet = (lbl, score, detail) => {
      const v = score == null ? 0 : score;
      const t = v >= 90 ? "great" : v >= 75 ? "ok" : v >= 60 ? "warn" : "bad";
      return `<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;font:600 12px/1.6 'JetBrains Mono',monospace;color:var(--ink-soft)"><span>${esc(lbl)}</span><span>${score == null ? "—" : score}</span></div>
        <div class="ov-pillar-bar s-${t}" style="height:8px"><i style="width:${v}%"></i></div>
        <div style="font:11px/1.5 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:3px">${detail}</div></div>`;
    };
    const rel = P.reliability || {}, perf = P.performance || {}, cap = P.capacity || {}, hyg = P.hygiene || {};
    let perfDetail = "";
    Object.entries(perf.per_class || {}).forEach(([cls, d]) => {
      perfDetail += `${cls.replace("_", " ")}: Apdex ${d.apdex} (n=${d.n})  `;
    });
    panel.innerHTML =
      `<div style="display:flex;align-items:center;gap:20px;margin-bottom:8px">
        <div class="ov-gauge" style="width:110px;height:110px"><svg width="110" height="110" style="transform:rotate(-90deg)">
          <circle cx="55" cy="55" r="46" fill="none" stroke="var(--paper-edge)" stroke-width="9"/>
          <path d="${arcPath(55, 55, 46, (h.composite || 0) / 100)}" fill="none" stroke="var(--c-ok)" stroke-width="9" stroke-linecap="round"/></svg>
          <div class="ov-gauge-num"><b style="font-size:28px">${h.composite == null ? "—" : h.composite}</b><span>${esc(h.grade || "")}</span></div></div>
        <div style="font:13px/1.7 'Fraunces',serif;color:var(--ink-soft)">window ${esc(h.window || "")} · ${fmtNum(h.sample_size)} requests${h.provisional ? " · <b>provisional</b>" : ""}${h.cold_start_excluded ? `<br><span style="font-size:11.5px;color:var(--ink-mute)">${h.cold_start_excluded} cold-start request(s) excluded from latency scoring</span>` : ""}</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px">
        <div>${bullet("Reliability " + (rel.score != null ? rel.score : ""), rel.score, `5xx ${rel.rate_5xx != null ? rel.rate_5xx + "%" : "—"} · 4xx ${rel.rate_4xx != null ? rel.rate_4xx + "%" : "—"} · success(Wilson) ${rel.success_wilson != null ? rel.success_wilson + "%" : "—"}`)}
          ${bullet("Capacity " + (cap.score != null ? cap.score : ""), cap.score, `rate-limited ${cap.rate_limited != null ? cap.rate_limited : 0} · quota ${esc(cap.quota_label || "—")}`)}</div>
        <div>${bullet("Performance " + (perf.score != null ? perf.score : ""), perf.score, perfDetail || "no latency data")}
          ${bullet("Hygiene " + (hyg.score != null ? hyg.score : ""), hyg.score, (hyg.penalties && hyg.penalties.length) ? hyg.penalties.map(p => p.detail).join(" · ") : "clean — no penalties")}</div>
      </div>
      <div class="ov-health-headline" style="margin-top:16px"><b>What's capping you:</b> ${esc(h.headline || "")}</div>
      <div style="font:11px/1.6 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:10px">
        WEIGHTING · Reliability 35 · Performance 30 · Capacity 20 · Hygiene 15<br>
        GRADE · A+≥97 A≥93 A-≥90 B+≥87 B≥83 B-≥80 C+≥77 C≥73 C-≥70 D≥60 F&lt;60</div>`;
  }

  // ════════════════════ WIDGET 2 · REQUEST RIBBON ══════════════════════
  let _ribbonGeo = [];   // [{x,w,row}] for hit-testing
  function drawRibbon(canvas, rows, opts) {
    opts = opts || {};
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 600, cssH = canvas.clientHeight || 90;
    canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    _ribbonGeo = [];
    if (!rows || !rows.length) return;
    const cs = getComputedStyle(document.documentElement);
    const col = {
      "2xx": cs.getPropertyValue("--c-ok").trim() || "#2f6f3e",
      "4xx": cs.getPropertyValue("--c-amber").trim() || "#c98a2b",
      "5xx": cs.getPropertyValue("--c-danger").trim() || "#b3402f",
    };
    const n = rows.length;
    const gap = 1;
    const w = Math.max(2, (cssW - gap * (n - 1)) / n);
    const baseY = cssH - 4;
    const maxLog = Math.log10(30000);   // 30s ceiling
    rows.forEach((r, i) => {
      const lat = Math.max(1, r.latency_ms || 1);
      const frac = Math.min(1, Math.log10(lat) / maxLog);
      const hgt = Math.max(3, frac * (cssH - 8));
      const x = i * (w + gap);
      const bucket = statusBucket(r.status_code || 0);
      ctx.fillStyle = (opts.dimId && r.id !== opts.dimId && opts.dimMatchPath && r.path !== opts.dimMatchPath) ? "rgba(0,0,0,.08)" : col[bucket];
      if (opts.highlightId && r.id === opts.highlightId) { ctx.fillStyle = "var(--ink)"; ctx.fillStyle = cs.getPropertyValue("--ink").trim() || "#1a1612"; }
      ctx.fillRect(x, baseY - hgt, w, hgt);
      _ribbonGeo.push({ x, w: w + gap, row: r });
    });
  }
  let _ribbonRows = [];   // live source of truth for the ribbon
  function renderRibbon(rows) {
    const canvas = $("ov-ribbon-canvas"), empty = $("ov-ribbon-empty"), aux = $("ov-ribbon-aux");
    if (!canvas) return;
    _ribbonRows = (rows || []).slice();
    if (!_ribbonRows.length) { canvas.style.display = "none"; if (empty) empty.hidden = false; return; }
    canvas.style.display = "block"; if (empty) empty.hidden = true;
    if (aux) aux.textContent = "last " + _ribbonRows.length;
    drawRibbon(canvas, _ribbonRows);
    canvas._rows = _ribbonRows;
    if (!canvas._wired) {
      canvas._wired = true;
      canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const hit = _ribbonGeo.find(g => x >= g.x && x < g.x + g.w);
        if (hit) {
          const r = hit.row;
          tip(true, e.clientX, e.clientY,
            `${esc(r.method || "")} ${esc(r.path || "")}<br>${r.status_code} · ${fmtMs(r.latency_ms)} · ${fmtAgo(r.timestamp)}`);
          Bus.emit("hover:endpoint", r.path);
        } else { tip(false); Bus.emit("hover:endpoint", null); }
      });
      canvas.addEventListener("mouseleave", () => { tip(false); Bus.emit("hover:endpoint", null); });
      canvas.addEventListener("click", (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const hit = _ribbonGeo.find(g => x >= g.x && x < g.x + g.w);
        if (hit && hit.row.id != null) openRequestDrawer(hit.row.id);
      });
    }
  }
  function openRequestDrawer(rid) {
    // Reuse the existing per-key request drawer if console_key_detail exposes it
    if (window.APIN && APIN.keyDetail && APIN.keyDetail.openRequest) { APIN.keyDetail.openRequest(rid); return; }
    // else navigate to the requests tab
    location.hash = "#requests";
  }
  function expandRibbon(panel) {
    const rows = (DATA && DATA.ribbon) || [];
    panel.innerHTML =
      `<div style="margin-bottom:10px;font:12px/1.5 'JetBrains Mono',monospace;color:var(--ink-soft)">${rows.length} requests · hover a tick for detail · click → request drawer</div>
       <canvas id="ov-ribbon-big" style="width:100%;height:200px;display:block;cursor:crosshair"></canvas>
       <div class="ov-ribbon-foot" style="padding-left:0"><span><i class="rdot s-2xx"></i>2xx</span><span><i class="rdot s-4xx"></i>4xx</span><span><i class="rdot s-5xx"></i>5xx</span><span class="ov-ribbon-hint">height = latency (log scale)</span></div>`;
    setTimeout(() => {
      const c = $("ov-ribbon-big");
      if (c) { drawRibbon(c, rows); c.addEventListener("mousemove", (e) => {
        const rect = c.getBoundingClientRect(); const x = e.clientX - rect.left;
        const hit = _ribbonGeo.find(g => x >= g.x && x < g.x + g.w);
        if (hit) { const r = hit.row; tip(true, e.clientX, e.clientY, `${esc(r.method)} ${esc(r.path)}<br>${r.status_code} · ${fmtMs(r.latency_ms)} · ${fmtAgo(r.timestamp)}`); } else tip(false);
      }); c.addEventListener("mouseleave", () => tip(false)); }
    }, 60);
  }

  // ════════════════════ WIDGET 3 · KPI TILES ═══════════════════════════
  function renderKpis(k) {
    if (!k) return;
    setKpi("requests", fmtNum(k.requests.value), k.requests.delta_pct, false);
    setKpi("success_rate", k.success_rate.value == null ? "—" : k.success_rate.value + "%", k.success_rate.delta_pct, false);
    setKpi("p50_ms", k.p50_ms.value == null ? "—" : fmtMs(k.p50_ms.value), k.p50_ms.delta_pct, true);
    setKpi("rate_limited", fmtNum(k.rate_limited.value), null, true);
  }
  function setKpi(name, valStr, delta, goodIfDown) {
    const tile = document.querySelector(`.ov-kpi[data-kpi="${name}"]`);
    if (!tile) return;
    const numEl = tile.querySelector("[data-num]"), dEl = tile.querySelector("[data-delta]");
    if (numEl) {
      // Odometer-roll pure-integer values (requests, rate-limited); set
      // mixed strings (e.g. "96.4%", "287ms") directly via cross-fade.
      const intVal = /^[\d,]+$/.test(valStr) ? Number(valStr.replace(/,/g, "")) : null;
      if (intVal != null && window.APIN && APIN.odometer) {
        APIN.odometer.roll(numEl, intVal);
      } else if (window.APIN && APIN.fx && APIN.fx.fadeReplace && numEl.textContent !== valStr && numEl.textContent !== "·") {
        APIN.fx.fadeReplace(numEl, () => { numEl.textContent = valStr; });
      } else numEl.textContent = valStr;
    }
    if (dEl) {
      if (delta == null) { dEl.textContent = ""; dEl.className = "ov-kpi-delta neutral"; }
      else {
        const up = delta > 0;
        const good = goodIfDown ? !up : up;
        dEl.textContent = (up ? "▴" : delta < 0 ? "▾" : "") + " " + Math.abs(delta) + "%";
        dEl.className = "ov-kpi-delta " + (delta === 0 ? "neutral" : good ? "up" : "down");
      }
    }
  }
  function expandKpi(name, panel) {
    const k = (DATA && DATA.kpis) || {};
    const titles = { requests: "Requests", success_rate: "Success rate", p50_ms: "p50 latency", rate_limited: "Rate-limited / quota" };
    const d = k[name] || {};
    let body = `<div style="font:600 40px/1 'Fraunces',serif;color:var(--ink)">${name === "p50_ms" ? fmtMs(d.value) : name === "success_rate" ? (d.value == null ? "—" : d.value + "%") : fmtNum(d.value)}</div>`;
    body += `<div style="font:12px/1.6 'JetBrains Mono',monospace;color:var(--ink-soft);margin-top:6px">previous period: ${name === "p50_ms" ? fmtMs(d.prev) : fmtNum(d.prev)}${d.delta_pct != null ? ` · Δ ${d.delta_pct > 0 ? "+" : ""}${d.delta_pct}%` : ""}</div>`;
    if (name === "success_rate" && DATA) {
      const h = DATA.health || {}; const rel = (h.pillars || {}).reliability || {};
      body += `<div style="margin-top:18px;font:13px/1.7 'Fraunces',serif;color:var(--ink-soft)">Status breakdown drives this metric.<br>5xx rate ${rel.rate_5xx != null ? rel.rate_5xx + "%" : "—"} · 4xx rate ${rel.rate_4xx != null ? rel.rate_4xx + "%" : "—"}</div>`;
    } else if (name === "p50_ms" && DATA) {
      const perf = ((DATA.health || {}).pillars || {}).performance || {};
      body += `<div style="margin-top:18px;font:12px/1.7 'JetBrains Mono',monospace;color:var(--ink-soft)">Apdex by endpoint class:<br>`;
      Object.entries(perf.per_class || {}).forEach(([c, v]) => { body += `${c.replace("_", " ")}: ${v.apdex} (T=${v.t_ms}ms, n=${v.n})<br>`; });
      body += `</div>`;
    }
    panel.innerHTML = body;
    return titles[name] || name;
  }

  // ════════════════════ WIDGET 4 · PERSONALITY ═════════════════════════
  function renderPersonality(p) {
    const host = $("ov-personality-body");
    if (!host || !p) return;
    const tags = p.tags || [];
    if (!tags.length) { host.innerHTML = `<div style="font:italic 13px/1.5 'Fraunces',serif;color:var(--ink-mute);padding:14px 0">Personality emerges after ~20 requests.</div>`; return; }
    host.innerHTML = tags.map(t => {
      const pct = Math.round((t.value || 0) * 100);
      return `<div class="ov-pbar" title="${esc(t.signal)}">
        <div class="ov-pbar-top"><span>${esc(t.name)}</span><span>${pct}%</span></div>
        <div class="ov-pbar-track" data-sig="${esc(t.signal)}"><div class="ov-pbar-fill" style="width:${pct}%"></div></div></div>`;
    }).join("");
    host.querySelectorAll(".ov-pbar-track").forEach(tr => {
      tr.addEventListener("mousemove", (e) => tip(true, e.clientX, e.clientY, esc(tr.getAttribute("data-sig"))));
      tr.addEventListener("mouseleave", () => tip(false));
    });
  }
  function expandPersonality(panel) {
    const tags = ((DATA && DATA.personality) || {}).tags || [];
    panel.innerHTML = `<div style="font:13px/1.7 'Fraunces',serif;color:var(--ink-soft);margin-bottom:14px">Behavioural fingerprint derived from this key's request mix and timing.</div>` +
      tags.map(t => {
        const pct = Math.round((t.value || 0) * 100);
        return `<div style="margin:14px 0"><div style="display:flex;justify-content:space-between;font:600 13px/1.6 'JetBrains Mono',monospace;color:var(--ink)"><span>${esc(t.name)}</span><span>${pct}%</span></div>
          <div class="ov-pbar-track" style="height:10px"><div class="ov-pbar-fill" style="width:${pct}%"></div></div>
          <div style="font:11.5px/1.5 'Fraunces',serif;font-style:italic;color:var(--ink-mute);margin-top:4px">${esc(t.signal)}</div></div>`;
      }).join("");
  }

  // ════════════════════ WIDGET 5 · SPARK-GRID ══════════════════════════
  function sparkSvg(buckets, w, h) {
    w = w || 90; h = h || 24;
    const max = Math.max(1, ...buckets);
    const n = buckets.length;
    const step = w / Math.max(1, n - 1);
    const pts = buckets.map((v, i) => `${(i * step).toFixed(1)},${(h - (v / max) * (h - 2) - 1).toFixed(1)}`).join(" ");
    return `<svg class="ov-spark-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="var(--c-ok)" stroke-width="1.5" stroke-linejoin="round"/></svg>`;
  }
  function renderSparkGrid(grid) {
    const host = $("ov-sparkgrid-body");
    if (!host) return;
    if (!grid || !grid.length) { host.innerHTML = `<div style="font:italic 13px/1.5 'Fraunces',serif;color:var(--ink-mute);padding:16px">No endpoint traffic yet.</div>`; return; }
    host.innerHTML = grid.map(s =>
      `<div class="ov-spark-row" data-path="${esc(s.path)}">
        <div class="ov-spark-path">${esc(s.path)}</div>
        ${sparkSvg(s.buckets || [])}
        <div class="ov-spark-meta">${fmtNum(s.count)} · p95 ${fmtMs(s.p95)}</div></div>`).join("");
    host.querySelectorAll(".ov-spark-row").forEach(row => {
      const path = row.getAttribute("data-path");
      row.addEventListener("mouseenter", () => Bus.emit("hover:endpoint", path));
      row.addEventListener("mouseleave", () => Bus.emit("hover:endpoint", null));
      row.addEventListener("click", () => { location.hash = "#requests"; });
    });
  }
  function expandSparkGrid(panel) {
    const grid = (DATA && DATA.spark_grid) || [];
    const metricKey = { requests: "buckets", latency: "buckets_lat", errors: "buckets_err" };
    function paint(metric) {
      const bk = metricKey[metric];
      const color = metric === "errors" ? "var(--c-danger)" : metric === "latency" ? "var(--c-amber)" : "var(--c-ok)";
      rowsHost.innerHTML = grid.map(s => {
        const buckets = s[bk] || [];
        const metaTxt = metric === "latency" ? "p95 " + fmtMs(s.p95)
          : metric === "errors" ? (s.buckets_err || []).reduce((a, b) => a + b, 0) + " err"
          : fmtNum(s.count) + " reqs";
        return `<div class="ov-spark-row" style="grid-template-columns:1fr 120px auto">
          <div class="ov-spark-path">${esc(s.path)}</div>
          ${sparkSvg(buckets, 120, 28).replace("var(--c-ok)", color)}
          <div class="ov-spark-meta">${metaTxt}</div></div>`;
      }).join("");
    }
    panel.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font:12px/1.5 'JetBrains Mono',monospace;color:var(--ink-soft)">top endpoints · 24h</span>
        <div class="ov-range" id="ov-spark-metric"><button data-m="requests" aria-pressed="true">requests</button><button data-m="latency">latency</button><button data-m="errors">errors</button></div>
      </div><div id="ov-spark-rows"></div>`;
    const rowsHost = panel.querySelector("#ov-spark-rows");
    paint("requests");
    panel.querySelectorAll("#ov-spark-metric button").forEach(b => {
      b.addEventListener("click", () => {
        panel.querySelectorAll("#ov-spark-metric button").forEach(x => x.removeAttribute("aria-pressed"));
        b.setAttribute("aria-pressed", "true");
        paint(b.getAttribute("data-m"));
      });
    });
  }

  // ════════════════════ WIDGET 6 · INSIGHTS ════════════════════════════
  // (d) infer which bento card an insight refers to (for hover-glow + click)
  function _insightRef(text) {
    const t = (text || "").toLowerCase();
    if (/latency|p95|p50|slow|fast|degrad|improv/.test(t)) return "ov-health";
    if (/predict|inference|read|get|burst|personalit/.test(t)) return "ov-personality";
    if (/error|5xx|4xx|fail|grade|health/.test(t)) return "ov-health";
    if (/endpoint|\/api\//.test(t)) return "ov-sparkgrid";
    if (/ip|integration|shared|leak/.test(t)) return "ov-ribbon";
    return null;
  }
  function renderInsights(list) {
    const host = $("ov-insights-body");
    if (!host) return;
    if (!list || !list.length) { host.innerHTML = `<div style="font:italic 13px/1.5 'Fraunces',serif;color:var(--ink-mute)">Operating cleanly — no notable signals.</div>`; return; }
    host.innerHTML = list.map(i => {
      const ref = _insightRef(i.text);
      return `<div class="ov-insight ${esc(i.tone || "info")}" data-ref="${ref || ""}"${ref ? ' style="cursor:pointer"' : ""}><i class="ins-icon"></i><span>${esc(i.text)}</span></div>`;
    }).join("");
    host.querySelectorAll(".ov-insight[data-ref]").forEach(el => {
      const ref = el.getAttribute("data-ref");
      if (!ref) return;
      el.addEventListener("mouseenter", () => { const c = $(ref); if (c) c.classList.add("is-glow"); });
      el.addEventListener("mouseleave", () => { const c = $(ref); if (c) c.classList.remove("is-glow"); });
      el.addEventListener("click", () => {
        const c = $(ref); if (!c) return;
        c.scrollIntoView({ behavior: "smooth", block: "center" });
        if (window.APIN && APIN.fx && APIN.fx.pulse) APIN.fx.pulse(c);
        else { c.classList.add("is-glow"); setTimeout(() => c.classList.remove("is-glow"), 1200); }
      });
    });
    if (window.APIN && APIN.fx) host.querySelectorAll(".ov-insight").forEach((el, i) => setTimeout(() => APIN.fx.enter(el), i * 60));
  }
  function expandInsights(panel) {
    const list = (DATA && DATA.insights) || [];
    const byTone = { great: 0, warn: 0, info: 0 };
    list.forEach(i => byTone[i.tone] = (byTone[i.tone] || 0) + 1);
    panel.innerHTML = `<div style="font:12px/1.6 'JetBrains Mono',monospace;color:var(--ink-soft);margin-bottom:10px">${byTone.warn || 0} warning · ${byTone.info || 0} info · ${byTone.great || 0} positive</div>` +
      list.map(i => `<div class="ov-insight ${esc(i.tone || "info")}"><i class="ins-icon"></i><span>${esc(i.text)}</span></div>`).join("");
  }

  // ── cross-widget linking: hover an endpoint → highlight spark-grid row ─
  Bus.on("hover:endpoint", (path) => {
    document.querySelectorAll(".ov-spark-row").forEach(r => {
      r.classList.toggle("is-linked", !!path && r.getAttribute("data-path") === path);
    });
  });

  // ── expand dispatcher (FLIP-grow lightbox) ────────────────────────────
  const EXPANDERS = {
    health: { title: "Health Score", build: expandHealth },
    ribbon: { title: "Request Ribbon", build: expandRibbon },
    personality: { title: "Key Personality", build: expandPersonality },
    sparkgrid: { title: "Top Endpoints", build: expandSparkGrid },
    insights: { title: "Insights", build: expandInsights },
  };
  function wireExpands() {
    document.querySelectorAll(".ov-expand[data-expand]").forEach(btn => {
      if (btn._wired) return; btn._wired = true;
      btn.addEventListener("click", (e) => {
        const which = btn.getAttribute("data-expand");
        const ex = EXPANDERS[which];
        if (!ex || !window.APIN || !APIN.lightbox) return;
        const card = btn.closest(".ov-card");
        APIN.lightbox.open({ title: ex.title + (DATA && DATA.key ? " · " + (DATA.key.name || "") : ""), sourceEl: card, build: ex.build });
      });
    });
    // KPI tiles → expand
    document.querySelectorAll(".ov-kpi[data-kpi]").forEach(tile => {
      if (tile._wired) return; tile._wired = true;
      tile.addEventListener("click", () => {
        const name = tile.getAttribute("data-kpi");
        if (!window.APIN || !APIN.lightbox) return;
        APIN.lightbox.open({ title: name, sourceEl: tile, build: (panel) => { const t = expandKpi(name, panel); } });
      });
    });
  }

  // ── controls ──────────────────────────────────────────────────────────
  function wireControls() {
    if (_wired) return; _wired = true;
    document.querySelectorAll(".ov-range button[data-range]").forEach(b => {
      if (b.getAttribute("data-range") === RANGE) b.setAttribute("aria-pressed", "true");
      else b.removeAttribute("aria-pressed");
      b.addEventListener("click", () => {
        RANGE = b.getAttribute("data-range");
        try { sessionStorage.setItem("ov_range", RANGE); } catch (_) {}
        document.querySelectorAll(".ov-range button").forEach(x => x.removeAttribute("aria-pressed"));
        b.setAttribute("aria-pressed", "true");
        refresh();
      });
    });
    const live = $("ov-live");
    if (live) live.addEventListener("click", () => {
      LIVE = !LIVE; live.setAttribute("data-on", LIVE ? "true" : "false");
      live.querySelector(".ov-live-label").textContent = LIVE ? "live" : "paused";
      _schedulePoll();
      if (LIVE) startSSE(); else stopSSE();
    });
    const rf = $("ov-refresh");
    if (rf) rf.addEventListener("click", () => { rf.classList.add("is-spinning"); setTimeout(() => rf.classList.remove("is-spinning"), 760); refresh(); });
    window.addEventListener("resize", () => {
      const c = $("ov-ribbon-canvas"); if (c && c._rows) drawRibbon(c, c._rows);
    });
  }
  function _schedulePoll() {
    if (_pollTimer) clearInterval(_pollTimer);
    if (LIVE && _active) _pollTimer = setInterval(refresh, 15000);
  }

  // ── (b) Per-event SSE — live ribbon ticks + KPI bump ──────────────────
  let _es = null, _liveReqCount = null;
  function startSSE() {
    if (_es || !window.EventSource) return;
    try { _es = new EventSource("/api/account/usage/stream"); }
    catch (_) { _es = null; return; }
    _es.onmessage = (e) => {
      if (!LIVE || !_active) return;
      let ev; try { ev = JSON.parse(e.data); } catch (_) { return; }
      if (!ev || ev.type !== "request") return;
      if (ev.key_id && PID && ev.key_id !== PID) return;   // not this key
      // Append the new request to the ribbon (newest on the right) + cap.
      _ribbonRows.push({
        id: ev.id || null, timestamp: ev.timestamp,
        method: ev.method, path: ev.path,
        status_code: ev.status_code, latency_ms: ev.latency_ms,
      });
      if (_ribbonRows.length > 120) _ribbonRows.shift();
      const canvas = $("ov-ribbon-canvas"), empty = $("ov-ribbon-empty"), aux = $("ov-ribbon-aux");
      if (canvas) {
        canvas.style.display = "block"; if (empty) empty.hidden = true;
        if (aux) aux.textContent = "last " + _ribbonRows.length + " · live";
        canvas._rows = _ribbonRows;
        drawRibbon(canvas, _ribbonRows);
      }
      // Bump the requests KPI live (poll reconciles the rest within 15s).
      if (DATA && DATA.kpis && DATA.kpis.requests) {
        if (_liveReqCount == null) _liveReqCount = DATA.kpis.requests.value || 0;
        _liveReqCount += 1;
        setKpi("requests", fmtNum(_liveReqCount), DATA.kpis.requests.delta_pct, false);
      }
      // cross-link: flash matching spark-grid row
      Bus.emit("hover:endpoint", ev.path);
      setTimeout(() => Bus.emit("hover:endpoint", null), 600);
    };
    _es.onerror = () => { /* EventSource auto-reconnects */ };
  }
  function stopSSE() { if (_es) { try { _es.close(); } catch (_) {} _es = null; } }

  // ── public init / activate ────────────────────────────────────────────
  function activate(pid) {
    PID = pid; _active = true;
    if (!$("ov-bento")) return;
    wireControls(); wireExpands();
    refresh();
    _schedulePoll();
    if (LIVE) startSSE();
  }
  function deactivate() { _active = false; if (_pollTimer) clearInterval(_pollTimer); stopSSE(); }

  window.APIN = window.APIN || {};
  window.APIN.keyOverview = { activate, deactivate, Bus };

  // auto-activate if overview is the visible pane on load
  document.addEventListener("DOMContentLoaded", () => {
    const pane = $("pane-overview");
    if (pane && pane.getAttribute("aria-hidden") === "false") {
      // PID is provided by console_key_detail.js via activate(); but if it
      // loads first, derive PID from the URL path as a fallback.
      const m = location.pathname.match(/\/keys\/([^/?#]+)/);
      if (m) setTimeout(() => activate(decodeURIComponent(m[1])), 50);
    }
  });
})();
