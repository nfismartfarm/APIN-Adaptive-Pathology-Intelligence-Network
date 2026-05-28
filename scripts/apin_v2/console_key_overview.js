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
  // Generic per-element count-up. Robust replacement for the odometer (which
  // needs a CSS contract absent on this page and garbled the gauge centre).
  // State lives on the element (el._cuVal / el._cuRaf) so the health gauge
  // and the four KPI tiles animate independently without clobbering frames.
  //   opts: { dur, from, fmt }  fmt(value)→string (default rounded integer)
  function countUp(el, to, opts) {
    if (!el) return;
    opts = opts || {};
    const dur = opts.dur || 600;
    const fmt = opts.fmt || ((v) => String(Math.round(v)));
    const from = (opts.from != null) ? opts.from
               : (el._cuVal != null ? el._cuVal : 0);
    if (el._cuRaf) cancelAnimationFrame(el._cuRaf);
    if (el._cuTimer) clearTimeout(el._cuTimer);
    // Safety net: rAF is paused/throttled in hidden or backgrounded tabs,
    // which would otherwise freeze the number on its first frame (the
    // "stuck number" class of bug). Guarantee the final value lands even if
    // not a single animation frame ever runs. No-op on a visible tab.
    el._cuTimer = setTimeout(() => { el.textContent = fmt(to); el._cuVal = to; }, dur + 160);
    const t0 = performance.now();
    (function step(now) {
      const k = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      const v = from + (to - from) * eased;
      el.textContent = fmt(v); el._cuVal = v;
      if (k < 1) el._cuRaf = requestAnimationFrame(step);
      else { el.textContent = fmt(to); el._cuVal = to; if (el._cuTimer) { clearTimeout(el._cuTimer); el._cuTimer = null; } }
    })(t0);
  }
  const _toneColor = (t) => ({ great: "#2f6f3e", ok: "#7a9a3e", warn: "#c98a2b", bad: "#b3402f", nodata: "#9a907a" }[t] || "#7a9a3e");
  let _healthArcTimer = null;
  function _tweenArc(pathEl, from, to, color) {
    if (_healthArcRaf) cancelAnimationFrame(_healthArcRaf);
    if (_healthArcTimer) clearTimeout(_healthArcTimer);
    const t0 = performance.now(), dur = 500;
    pathEl.setAttribute("stroke", color);
    // Safety net (see countUp): land the final arc even if rAF never fires.
    _healthArcTimer = setTimeout(() => { pathEl.setAttribute("d", arcPath(70, 70, 58, to)); _healthFrac = to; }, dur + 160);
    (function step(now) {
      const k = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);          // easeOutCubic
      const f = from + (to - from) * eased;
      pathEl.setAttribute("d", arcPath(70, 70, 58, f));
      if (k < 1) _healthArcRaf = requestAnimationFrame(step);
      else { _healthFrac = to; if (_healthArcTimer) { clearTimeout(_healthArcTimer); _healthArcTimer = null; } }
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
          <div class="ov-gauge-num"><b id="ov-gauge-num">0</b><span id="ov-gauge-grade">${esc(h.grade)}${h.provisional ? " ~" : ""}</span></div>
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
    if (numEl) {
      // Count-up from the previously shown value (0 on first launch → score).
      const first = (numEl._cuVal == null);
      countUp(numEl, Math.round(h.composite), { dur: first ? 700 : 450 });
    }
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

  // mini bullet bar used inside pillar cards
  function _hbar(score, color) {
    const v = score == null ? 0 : score;
    return `<span style="display:block;height:7px;background:rgba(120,110,90,.16);border-radius:3px;overflow:hidden"><i style="display:block;height:100%;width:${v}%;background:${color};border-radius:3px"></i></span>`;
  }
  function expandHealth(panel) {
    const h = (DATA && DATA.health) || {};
    const P = h.pillars || {};
    const col = _C();
    const rel = P.reliability || {}, perf = P.performance || {}, cap = P.capacity || {}, hyg = P.hygiene || {};
    const tone = (v) => v == null ? col.mute : v >= 90 ? col.ok : v >= 75 ? "#7a9a3e" : v >= 60 ? col.amber : col.danger;
    const cardHead = (key, label, score) =>
      `<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
         <span style="font:600 11px 'JetBrains Mono',monospace;letter-spacing:.08em;color:var(--ink-soft)">${label}</span>
         <span style="font:700 20px 'Fraunces',serif;color:${tone(score)}">${score == null ? "—" : score}</span></div>`;
    const subbar = (label, val, score, color) =>
      `<div style="margin:5px 0"><div style="display:flex;justify-content:space-between;font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute)"><span>${label}</span><span>${val}</span></div>${_hbar(score, color)}</div>`;
    // Reliability card
    const relCard = cardHead("reliability", "RELIABILITY", rel.score) +
      subbar("5xx", (rel.rate_5xx != null ? rel.rate_5xx + "%" : "—"), rel.rate_5xx != null ? 100 - Math.min(100, rel.rate_5xx * 200) : 0, col.danger) +
      subbar("4xx", (rel.rate_4xx != null ? rel.rate_4xx + "%" : "—"), rel.rate_4xx != null ? 100 - Math.min(100, rel.rate_4xx * 4) : 0, col.amber) +
      `<div style="font:10.5px/1.6 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:6px">success (Wilson) ${rel.success_wilson != null ? rel.success_wilson + "%" : "—"}<br>error budget: ${rel.error_budget_pct != null ? rel.error_budget_pct + "% remaining" : "—"}</div>`;
    // Performance card — Apdex by class bars
    let perfRows = "";
    Object.entries(perf.per_class || {}).forEach(([cls, d]) => {
      perfRows += subbar(cls.replace(/_/g, " "), `${d.apdex} (n=${d.n})`, d.apdex * 100, d.apdex >= 0.85 ? col.ok : d.apdex >= 0.7 ? col.amber : col.danger);
    });
    const trendTxt = perf.trend_pct == null ? "" : (perf.trend_pct < 0 ? `▾ improving ${Math.abs(perf.trend_pct)}%` : `▴ degrading ${perf.trend_pct}%`);
    const perfCard = cardHead("performance", "PERFORMANCE", perf.score) +
      (perfRows || `<div style="font:11px 'Fraunces',serif;font-style:italic;color:var(--ink-mute)">no latency data</div>`) +
      (trendTxt ? `<div style="font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:6px">p95 trend · ${trendTxt}</div>` : "");
    // Capacity card
    const capCard = cardHead("capacity", "CAPACITY", cap.score) +
      `<div style="font:11px/1.7 'JetBrains Mono',monospace;color:var(--ink-soft)">rate-limited · ${cap.rate_limited != null ? cap.rate_limited : 0}<br>quota · ${esc(cap.quota_label || "unlimited")}</div>` +
      `<div style="font:10.5px 'Fraunces',serif;font-style:italic;color:var(--ink-mute);margin-top:6px">${cap.rate_limited ? "throttling observed this window" : "no throttling this window"}</div>`;
    // Hygiene card
    const tick = (ok) => ok ? `<span style="color:${col.ok}">✓</span>` : `<span style="color:${col.danger}">⚠</span>`;
    const hygCard = cardHead("hygiene", "HYGIENE", hyg.score) +
      `<div style="font:11px/1.8 'JetBrains Mono',monospace;color:var(--ink-soft)">
        expires · ${esc(hyg.expires_label || "never")} ${tick((hyg.expires_label || "never") === "never" || !/^expired/.test(hyg.expires_label || ""))}<br>
        age · ${esc(hyg.age_label || "—")} ${tick(true)}<br>
        scopes ${tick(hyg.scope_ok !== false)} ${hyg.scope_ok === false ? "over-permissioned" : "match usage"}<br>
        IPs · ${hyg.distinct_ips != null ? hyg.distinct_ips : "—"} ${tick(!hyg.ip_fanout)} ${hyg.ip_fanout ? "fan-out!" : "stable"}</div>`;
    const cardWrap = (key, inner) =>
      `<div class="ov-hx-card" data-pillar="${key}" style="border:1px solid var(--paper-edge);border-radius:10px;padding:13px 15px;cursor:pointer;transition:background .12s,border-color .12s">${inner}</div>`;

    panel.innerHTML =
      `<div style="display:flex;align-items:center;gap:20px;margin-bottom:14px">
        <svg width="92" height="92" viewBox="0 0 92 92" style="transform:rotate(-90deg);flex:none">
          <circle cx="46" cy="46" r="38" fill="none" stroke="var(--paper-edge)" stroke-width="8"/>
          <path d="${arcPath(46, 46, 38, (h.composite || 0) / 100)}" fill="none" stroke="${tone(h.composite)}" stroke-width="8" stroke-linecap="round"/></svg>
        <div><div style="font:700 30px/1 'Fraunces',serif;color:var(--ink)">${h.composite == null ? "—" : h.composite} <span style="font-size:18px;color:${tone(h.composite)}">${esc(h.grade || "")}</span></div>
          <div style="font:12px/1.6 'JetBrains Mono',monospace;color:var(--ink-soft);margin-top:4px">window ${esc(h.window || "")} · ${fmtNum(h.sample_size)} requests${h.provisional ? " · provisional" : ""}</div>
          ${h.cold_start_excluded ? `<div style="font:10.5px 'Fraunces',serif;font-style:italic;color:var(--ink-mute)">${h.cold_start_excluded} cold-start request(s) excluded from latency scoring</div>` : ""}</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        ${cardWrap("reliability", relCard)}${cardWrap("performance", perfCard)}
        ${cardWrap("capacity", capCard)}${cardWrap("hygiene", hygCard)}</div>
      <div id="ov-hx-drill" hidden style="margin-top:12px;border:1px solid var(--paper-edge);border-radius:10px;padding:13px 15px;background:rgba(120,110,90,.05)"></div>
      <div style="margin-top:16px"><div style="font:500 italic 11px 'Fraunces',serif;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-soft);margin-bottom:6px">30-day composite trend</div>
        <div id="ov-hx-trend"></div><div id="ov-hx-trend-read" style="font:11px 'JetBrains Mono',monospace;color:var(--ink-mute);height:14px;text-align:center"></div></div>
      <div class="ov-health-headline" style="margin-top:14px"><b>What's capping you:</b> ${esc(h.headline || "")}</div>
      <details style="margin-top:10px"><summary style="cursor:pointer;font:11px 'JetBrains Mono',monospace;color:var(--ink-soft)">weighting &amp; grade scale</summary>
        <div style="font:10.5px/1.7 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:8px">
          HEALTH = 0.35·Reliability + 0.30·Performance + 0.20·Capacity + 0.15·Hygiene<br>
          A+≥97 · A≥93 · A-≥90 · B+≥87 · B≥83 · B-≥80 · C+≥77 · C≥73 · C-≥70 · D≥60 · F&lt;60</div></details>`;

    // 30-day trend (hover-scrub)
    trendChart(panel.querySelector("#ov-hx-trend"), (h.trend || []).map(t => ({ label: t.day, value: t.composite })), panel.querySelector("#ov-hx-trend-read"));

    // pillar click-to-drill
    const drill = panel.querySelector("#ov-hx-drill");
    const drillBuild = {
      performance: () => {
        const rows = Object.entries(perf.per_class || {});
        if (!rows.length) return "no per-class latency data.";
        return `<div style="font:600 11px 'JetBrains Mono',monospace;color:var(--ink-soft);margin-bottom:8px">PERFORMANCE · per-endpoint-class Apdex (T = satisfaction threshold)</div>` +
          rows.map(([c, d]) => `<div style="margin:7px 0"><div style="display:flex;justify-content:space-between;font:11px 'JetBrains Mono',monospace;color:var(--ink)"><span>${esc(c.replace(/_/g, " "))}</span><span>Apdex ${d.apdex} · T=${d.t_ms}ms · n=${d.n}</span></div>${_hbar(d.apdex * 100, d.apdex >= 0.85 ? col.ok : d.apdex >= 0.7 ? col.amber : col.danger)}</div>`).join("");
      },
      reliability: () => `<div style="font:11px/1.8 'JetBrains Mono',monospace;color:var(--ink-soft)">5xx ${rel.rate_5xx != null ? rel.rate_5xx + "%" : "—"} (${rel.n_5xx || 0} reqs) · 4xx ${rel.rate_4xx != null ? rel.rate_4xx + "%" : "—"} (${rel.n_4xx || 0})<br>success(Wilson lower bound) ${rel.success_wilson != null ? rel.success_wilson + "%" : "—"}<br>server-error score ${rel.server_score != null ? rel.server_score : "—"} (70%) · client-error score ${rel.client_score != null ? rel.client_score : "—"} (30%)<br>error budget ${rel.error_budget_pct != null ? rel.error_budget_pct + "% remaining" : "—"} (99.9% SLO)</div>`,
      capacity: () => `<div style="font:11px/1.8 'JetBrains Mono',monospace;color:var(--ink-soft)">rate-limit pressure score ${cap.rate_limit_score != null ? cap.rate_limit_score : "—"} (60%)<br>quota headroom score ${cap.quota_score != null ? cap.quota_score : "—"} (40%)<br>rate-limited ${cap.rate_limited || 0} · quota ${esc(cap.quota_label || "unlimited")}</div>`,
      hygiene: () => {
        const pens = hyg.penalties || [];
        return `<div style="font:11px/1.8 'JetBrains Mono',monospace;color:var(--ink-soft)">expires ${esc(hyg.expires_label || "never")} · age ${esc(hyg.age_label || "—")}<br>distinct IPs ${hyg.distinct_ips != null ? hyg.distinct_ips : "—"}${hyg.ip_fanout ? " (fan-out flagged)" : ""}<br>${pens.length ? "penalties: " + pens.map(p => esc(p.detail)).join(" · ") : "clean — no penalties applied"}</div>`;
      },
    };
    panel.querySelectorAll(".ov-hx-card").forEach(card => {
      card.addEventListener("mouseenter", () => card.style.background = "rgba(120,110,90,.06)");
      card.addEventListener("mouseleave", () => card.style.background = "");
      card.addEventListener("click", () => {
        const pk = card.getAttribute("data-pillar");
        drill.innerHTML = (drillBuild[pk] ? drillBuild[pk]() : "");
        drill.hidden = false;
        panel.querySelectorAll(".ov-hx-card").forEach(c => c.style.borderColor = "var(--paper-edge)");
        card.style.borderColor = tone(P[pk] && P[pk].score);
      });
    });
  }

  // ════════════════════ WIDGET 2 · REQUEST RIBBON ══════════════════════
  function _ribbonColors() {
    const cs = getComputedStyle(document.documentElement);
    return {
      "2xx": cs.getPropertyValue("--c-ok").trim() || "#2f6f3e",
      "4xx": cs.getPropertyValue("--c-amber").trim() || "#c98a2b",
      "5xx": cs.getPropertyValue("--c-danger").trim() || "#b3402f",
      ink:   cs.getPropertyValue("--ink").trim() || "#1a1612",
    };
  }
  // Draw the ribbon into a canvas. Per-canvas geometry stored on canvas._geo
  // (so the bento + expanded canvases don't clobber each other's hit-test).
  // opts.progress (0..1) scales bar heights for the intro draw-in animation.
  function drawRibbon(canvas, rows, opts) {
    opts = opts || {};
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 600, cssH = canvas.clientHeight || 90;
    canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const geo = [];
    canvas._geo = geo;
    if (!rows || !rows.length) return;
    const col = _ribbonColors();
    const n = rows.length, gap = 1;
    const w = Math.max(2, (cssW - gap * (n - 1)) / n);
    const baseY = cssH - 4, maxLog = Math.log10(30000);
    const prog = opts.progress == null ? 1 : opts.progress;
    const hi = opts.highlightPath;
    rows.forEach((r, i) => {
      const lat = Math.max(1, r.latency_ms || 1);
      const frac = Math.min(1, Math.log10(lat) / maxLog);
      const hgt = Math.max(3, frac * (cssH - 8)) * prog;
      const x = i * (w + gap);
      const bucket = statusBucket(r.status_code || 0);
      ctx.globalAlpha = (hi && r.path !== hi) ? 0.25 : 1;
      ctx.fillStyle = col[bucket];
      ctx.fillRect(x, baseY - hgt, w, hgt);
      ctx.globalAlpha = 1;
      geo.push({ x, w: w + gap, row: r });
    });
  }
  // Intro draw-in: animate progress 0→1 over ~520ms (staggered feel via easeOut).
  function drawRibbonIntro(canvas, rows) {
    const t0 = performance.now(), dur = 520;
    // Safety net (see countUp): if rAF never fires the bars would be stuck at
    // progress 0 (invisible). Force a full-height draw after the intro window.
    const safety = setTimeout(() => drawRibbon(canvas, rows, { progress: 1 }), dur + 160);
    (function step(now) {
      const k = Math.min(1, (now - t0) / dur);
      drawRibbon(canvas, rows, { progress: 1 - Math.pow(1 - k, 3) });
      if (k < 1) requestAnimationFrame(step);
      else clearTimeout(safety);
    })(t0);
  }
  function _wireRibbonHover(canvas, opts) {
    opts = opts || {};
    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const hit = (canvas._geo || []).find(g => x >= g.x && x < g.x + g.w);
      if (hit) {
        const r = hit.row;
        tip(true, e.clientX, e.clientY,
          `${esc(r.method || "")} ${esc(r.path || "")}<br>${r.status_code} · ${fmtMs(r.latency_ms)} · ${fmtAgo(r.timestamp)}`);
        if (opts.bus !== false) Bus.emit("hover:endpoint", r.path);
      } else { tip(false); if (opts.bus !== false) Bus.emit("hover:endpoint", null); }
    });
    canvas.addEventListener("mouseleave", () => { tip(false); if (opts.bus !== false) Bus.emit("hover:endpoint", null); });
    canvas.addEventListener("click", (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const hit = (canvas._geo || []).find(g => x >= g.x && x < g.x + g.w);
      if (hit) openRequestDrawer(hit.row.id, hit.row);
    });
  }
  let _ribbonRows = [];      // full set (up to 240) — live source of truth
  let _ribbonIntroDone = false;
  function renderRibbon(rows) {
    const canvas = $("ov-ribbon-canvas"), empty = $("ov-ribbon-empty"), aux = $("ov-ribbon-aux");
    if (!canvas) return;
    _ribbonRows = (rows || []).slice();
    if (!_ribbonRows.length) { canvas.style.display = "none"; if (empty) empty.hidden = false; return; }
    canvas.style.display = "block"; if (empty) empty.hidden = true;
    // Bento shows the most recent 120.
    const shown = _ribbonRows.slice(-120);
    if (aux) aux.textContent = "last " + shown.length;
    canvas._rows = shown;
    if (!_ribbonIntroDone) { _ribbonIntroDone = true; drawRibbonIntro(canvas, shown); }
    else drawRibbon(canvas, shown);
    if (!canvas._wired) { canvas._wired = true; _wireRibbonHover(canvas); }
  }
  function openRequestDrawer(rid, row) {
    // Live-streamed rows have no DB id yet (id===null) — they aren't in the
    // request log until the buffer flushes. Show a brief note instead of a
    // broken drawer.
    if (rid == null) {
      if (window.APIN && APIN.toast) APIN.toast.show("This request is still being recorded — try again in a few seconds.");
      return;
    }
    if (window.APIN && APIN.keyDetail && typeof APIN.keyDetail.openRequest === "function") {
      APIN.keyDetail.openRequest(rid); return;
    }
    location.hash = "#requests";
  }
  // Mini density histogram (req/sec) drawn into a small canvas.
  function drawDensity(canvas, buckets) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 600, cssH = canvas.clientHeight || 34;
    canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const max = Math.max(1, ...buckets), n = buckets.length, w = cssW / n;
    const col = _ribbonColors();
    buckets.forEach((v, i) => {
      const h = (v / max) * (cssH - 2);
      ctx.fillStyle = col["2xx"]; ctx.globalAlpha = 0.5;
      ctx.fillRect(i * w, cssH - h, Math.max(1, w - 1), h);
    });
    ctx.globalAlpha = 1;
  }
  function expandRibbon(panel) {
    const allRows = (DATA && DATA.ribbon) || [];
    const ts = (DATA && DATA.timeseries) || { req: [] };
    panel.innerHTML =
      `<div class="ov-rb-controls" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px;font:12px 'JetBrains Mono',monospace">
         <div class="ov-range" id="ov-rb-status"><button data-s="all" aria-pressed="true">all</button><button data-s="2xx">2xx</button><button data-s="4xx">4xx</button><button data-s="5xx">5xx</button></div>
         <select id="ov-rb-method" style="padding:5px 8px;border:1px solid var(--paper-edge);background:var(--paper);font:inherit;color:var(--ink);border-radius:7px"><option value="">any method</option><option>GET</option><option>POST</option><option>PUT</option><option>DELETE</option></select>
         <input id="ov-rb-ep" type="search" placeholder="endpoint contains…" style="padding:5px 8px;border:1px solid var(--paper-edge);background:var(--paper);font:inherit;color:var(--ink);border-radius:7px;flex:1;min-width:140px">
         <button id="ov-rb-pause" class="ov-livebtn" data-on="true"><span class="ov-live-dot"></span><span class="ov-live-label">live</span></button>
         <span id="ov-rb-count" style="color:var(--ink-mute)">${allRows.length} requests</span>
       </div>
       <div style="font:italic 10.5px 'Fraunces',serif;color:var(--ink-mute);margin-bottom:2px">density · req per bucket</div>
       <canvas id="ov-rb-density" style="width:100%;height:30px;display:block;margin-bottom:6px"></canvas>
       <canvas id="ov-rb-big" style="width:100%;height:190px;display:block;cursor:crosshair"></canvas>
       <div style="margin-top:8px"><input id="ov-rb-brush" type="range" min="0" max="100" value="100" style="width:100%"></div>
       <div style="font:italic 11px 'Fraunces',serif;color:var(--ink-mute);text-align:center">drag to pan the window · showing <span id="ov-rb-window"></span></div>
       <div class="ov-ribbon-foot" style="padding-left:0"><span><i class="rdot s-2xx"></i>2xx</span><span><i class="rdot s-4xx"></i>4xx</span><span><i class="rdot s-5xx"></i>5xx</span><span class="ov-ribbon-hint">height = latency (log) · hover → detail · click → request drawer</span></div>`;
    let live = true;
    const WINDOW = 80;                 // ticks visible at once
    function filtered() {
      const sf = panel.querySelector('#ov-rb-status button[aria-pressed="true"]').getAttribute("data-s");
      const mf = panel.querySelector("#ov-rb-method").value;
      const ef = panel.querySelector("#ov-rb-ep").value.toLowerCase();
      return allRows.filter(r => {
        if (sf !== "all" && statusBucket(r.status_code || 0) !== sf) return false;
        if (mf && (r.method || "").toUpperCase() !== mf) return false;
        if (ef && !(r.path || "").toLowerCase().includes(ef)) return false;
        return true;
      });
    }
    function paint() {
      const c = $("ov-rb-big"); if (!c) return;
      const rows = filtered();
      const brush = Number(panel.querySelector("#ov-rb-brush").value);
      const maxStart = Math.max(0, rows.length - WINDOW);
      const start = Math.round((brush / 100) * maxStart);
      const win = rows.slice(start, start + WINDOW);
      drawRibbon(c, win.length ? win : rows.slice(-WINDOW));
      const wEl = $("ov-rb-window");
      if (wEl) wEl.textContent = rows.length <= WINDOW ? `all ${rows.length}` : `${start + 1}–${Math.min(start + WINDOW, rows.length)} of ${rows.length}`;
      const cnt = $("ov-rb-count"); if (cnt) cnt.textContent = rows.length + " requests";
    }
    setTimeout(() => {
      drawDensity($("ov-rb-density"), ts.req || []);
      const big = $("ov-rb-big");
      // donut cross-link: pre-press a status chip if opened via _openRibbonFiltered
      if (_ribbonPreset) {
        const pb = panel.querySelector(`#ov-rb-status button[data-s="${_ribbonPreset}"]`);
        if (pb) { panel.querySelectorAll("#ov-rb-status button").forEach(x => x.removeAttribute("aria-pressed")); pb.setAttribute("aria-pressed", "true"); }
        _ribbonPreset = null;
      }
      paint();
      _wireRibbonHover(big, { bus: false });
      panel.querySelectorAll("#ov-rb-status button").forEach(b => b.addEventListener("click", () => {
        panel.querySelectorAll("#ov-rb-status button").forEach(x => x.removeAttribute("aria-pressed"));
        b.setAttribute("aria-pressed", "true"); paint();
      }));
      panel.querySelector("#ov-rb-method").addEventListener("change", paint);
      panel.querySelector("#ov-rb-ep").addEventListener("input", paint);
      panel.querySelector("#ov-rb-brush").addEventListener("input", paint);
      const pause = $("ov-rb-pause");
      pause.addEventListener("click", () => { live = !live; pause.setAttribute("data-on", live ? "true" : "false"); pause.querySelector(".ov-live-label").textContent = live ? "live" : "paused"; });
    }, 60);
  }

  // ── shared expanded-view chart primitives (inline SVG, token-aware) ───
  function _C() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n, d) => (cs.getPropertyValue(n).trim() || d);
    return { ok: g("--c-ok", "#2f6f3e"), amber: g("--c-amber", "#c98a2b"),
             danger: g("--c-danger", "#b3402f"), ink: g("--ink", "#1a1612"),
             soft: g("--ink-soft", "#6b6453"), mute: g("--ink-mute", "#9a907a"),
             edge: g("--paper-edge", "#c7bca9") };
  }
  // section subheader inside a lightbox panel (matches lightbox.section style)
  const _kxSec = (t) => `<div style="font:500 italic 11px 'Fraunces',serif;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-soft);margin:16px 0 9px;padding-bottom:6px;border-bottom:1px solid var(--paper-edge)">${esc(t)}</div>`;
  // horizontal labelled bars (endpoint breakdown / tables)
  function hbars(items, opts) {
    opts = opts || {};
    const max = Math.max(1, ...items.map(i => i.value));
    const c = opts.color || _C().ok;
    return items.map(i => {
      const pct = (i.value / max) * 100;
      return `<div style="display:grid;grid-template-columns:140px 1fr 70px;gap:10px;align-items:center;margin:6px 0;font:11.5px 'JetBrains Mono',monospace">
        <span style="color:var(--ink-soft);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(i.label)}">${esc(i.label)}</span>
        <span style="height:13px;background:rgba(120,110,90,.16);border-radius:3px;overflow:hidden;display:block"><i style="display:block;height:100%;width:${pct.toFixed(1)}%;background:${c};border-radius:3px;transition:width .4s cubic-bezier(.22,1,.36,1)"></i></span>
        <span style="color:var(--ink);text-align:right">${esc(i.sub != null ? i.sub : i.value)}</span></div>`;
    }).join("");
  }
  // stacked status bars over time (timeseries)
  function stackedBarsSvg(ts, w, h) {
    w = w || 560; h = h || 150;
    const req = ts.req || [], n = req.length || 1;
    const max = Math.max(1, ...req);
    const bw = w / n, gap = Math.min(2.5, bw * 0.22), col = _C();
    let bars = "";
    for (let i = 0; i < n; i++) {
      let y = h;
      const seg = (v, c) => { if (!v) return ""; const sh = (v / max) * (h - 4); y -= sh; return `<rect x="${(i * bw + gap / 2).toFixed(1)}" y="${y.toFixed(1)}" width="${(bw - gap).toFixed(1)}" height="${sh.toFixed(1)}" fill="${c}"/>`; };
      bars += seg(ts.s2xx ? ts.s2xx[i] : 0, col.ok)
            + seg(ts.s4xx ? ts.s4xx[i] : 0, col.amber)
            + seg(ts.s5xx ? ts.s5xx[i] : 0, col.danger);
    }
    return `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block;height:${h}px">${bars}</svg>`;
  }
  // donut for status mix; parts:[{key,label,value,color}] — slices are clickable
  function donutSvg(parts, size, thick) {
    size = size || 156; thick = thick || 26;
    const r = (size - thick) / 2, cx = size / 2, cy = size / 2, circ = 2 * Math.PI * r;
    const total = parts.reduce((a, p) => a + p.value, 0) || 1;
    let off = 0;
    const segs = parts.filter(p => p.value > 0).map(p => {
      const len = (p.value / total) * circ, dash = `${len.toFixed(2)} ${(circ - len).toFixed(2)}`;
      const s = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${p.color}" stroke-width="${thick}" stroke-dasharray="${dash}" stroke-dashoffset="${(-off).toFixed(2)}" data-slice="${p.key}" style="cursor:pointer;transition:stroke-width .14s" transform="rotate(-90 ${cx} ${cy})"><title>${esc(p.label)}: ${p.value}</title></circle>`;
      off += len; return s;
    }).join("");
    const pct = Math.round(100 * (parts[0] ? parts[0].value : 0) / total);
    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--paper-edge)" stroke-width="${thick}" opacity="0.35"/>
      ${segs}
      <text x="${cx}" y="${cy - 1}" text-anchor="middle" style="font:700 26px 'Fraunces',serif;fill:var(--ink)">${pct}%</text>
      <text x="${cx}" y="${cy + 17}" text-anchor="middle" style="font:9.5px 'JetBrains Mono',monospace;fill:var(--ink-mute);letter-spacing:.08em">SUCCESS</text></svg>`;
  }
  // log-binned latency histogram with edge ticks
  function histSvg(lh, w, h) {
    w = w || 560; h = h || 150;
    const edges = lh.edges || [], bins = lh.bins || [], n = bins.length || 1;
    const max = Math.max(1, ...bins), bw = w / n, col = _C();
    let bars = "", ticks = "";
    for (let i = 0; i < n; i++) {
      const bh = (bins[i] / max) * (h - 4);
      bars += `<rect x="${(i * bw + 1).toFixed(1)}" y="${(h - bh).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${bh.toFixed(1)}" fill="${col.ok}" opacity="${(0.35 + 0.55 * (bins[i] / max)).toFixed(2)}"><title>${edges[i]}–${edges[i + 1]}ms: ${bins[i]}</title></rect>`;
    }
    for (let i = 0; i <= n; i += 2) {
      const e = edges[i]; if (e == null) continue;
      const lbl = e >= 1000 ? (e / 1000) + "s" : e + "ms";
      ticks += `<text x="${(i * bw).toFixed(1)}" y="${h + 12}" text-anchor="middle" style="font:9px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${lbl}</text>`;
    }
    return `<svg width="100%" viewBox="0 0 ${w} ${h + 16}" preserveAspectRatio="none" style="display:block;height:${h + 16}px;overflow:visible">${bars}${ticks}</svg>`;
  }
  // radial burndown gauge
  function radialSvg(frac, label, sub, size) {
    size = size || 144; const r = (size - 22) / 2, cx = size / 2, cy = size / 2, col = _C();
    const c = frac >= 0.9 ? col.danger : frac >= 0.7 ? col.amber : col.ok;
    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--paper-edge)" stroke-width="11" opacity="0.35"/>
      <path d="${arcPath(cx, cy, r, frac)}" fill="none" stroke="${c}" stroke-width="11" stroke-linecap="round" transform="rotate(-90 ${cx} ${cy})"/>
      <text x="${cx}" y="${cy - 1}" text-anchor="middle" style="font:700 21px 'Fraunces',serif;fill:var(--ink)">${esc(label)}</text>
      <text x="${cx}" y="${cy + 15}" text-anchor="middle" style="font:9px 'JetBrains Mono',monospace;fill:var(--ink-mute)">${esc(sub || "")}</text></svg>`;
  }

  // 6-axis radar — this key (solid fill) vs account average (dashed outline)
  function radarSvg(dims, labels, vecA, vecB, size) {
    size = size || 300;
    const cx = size / 2, cy = size / 2, R = size / 2 - 42, n = dims.length, col = _C();
    const ang = (i) => (-Math.PI / 2) + i * (2 * Math.PI / n);
    const pt = (i, r) => [cx + Math.cos(ang(i)) * R * r, cy + Math.sin(ang(i)) * R * r];
    let grid = "", spokes = "", labs = "";
    [0.25, 0.5, 0.75, 1].forEach(rr => {
      grid += `<polygon points="${dims.map((_, i) => pt(i, rr).map(v => v.toFixed(1)).join(",")).join(" ")}" fill="none" stroke="var(--paper-edge)" stroke-width="1" opacity="0.45"/>`;
    });
    dims.forEach((k, i) => {
      const [ex, ey] = pt(i, 1);
      spokes += `<line x1="${cx}" y1="${cy}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" stroke="var(--paper-edge)" stroke-width="1" opacity="0.45"/>`;
      const [lx, ly] = pt(i, 1.17);
      const anchor = Math.abs(lx - cx) < 12 ? "middle" : (lx < cx ? "end" : "start");
      labs += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${anchor}" dominant-baseline="middle" style="font:9px 'JetBrains Mono',monospace;fill:var(--ink-soft)">${esc(labels[i])}</text>`;
    });
    const poly = (vec, stroke, fill, dash) =>
      `<polygon points="${dims.map((k, i) => pt(i, Math.max(0, Math.min(1, vec[k] || 0))).map(v => v.toFixed(1)).join(",")).join(" ")}" fill="${fill}" stroke="${stroke}" stroke-width="2" stroke-linejoin="round"${dash ? ' stroke-dasharray="5 4"' : ""}/>`;
    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">${grid}${spokes}${labs}
      ${vecB ? poly(vecB, col.mute, "none", true) : ""}
      ${poly(vecA, col.ok, "rgba(47,111,62,0.16)", false)}</svg>`;
  }
  // percentile fan over time: shaded p50–p99 band + p50/p95/p99 lines
  function fanSvg(p50, p95, p99, w, h) {
    w = w || 560; h = h || 140; const col = _C();
    const n = (p99 || []).length || 1;
    const max = Math.max(1, ...(p99 || [1]), ...(p95 || [1]), ...(p50 || [1]));
    const X = (i) => (n <= 1 ? 0 : (i / (n - 1)) * w);
    const Y = (v) => h - (v / max) * (h - 6) - 3;
    const path = (arr) => (arr || []).map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    const band = () => (p99 || []).map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`)
      .concat((p50 || []).map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).reverse()).join(" ");
    return `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block;height:${h}px">
      <polygon points="${band()}" fill="${col.amber}" opacity="0.13"/>
      <polyline points="${path(p99)}" fill="none" stroke="${col.danger}" stroke-width="1.2" opacity="0.7" stroke-dasharray="4 3"/>
      <polyline points="${path(p95)}" fill="none" stroke="${col.amber}" stroke-width="1.5"/>
      <polyline points="${path(p50)}" fill="none" stroke="${col.ok}" stroke-width="1.6"/></svg>`;
  }
  // line chart with hover-scrub crosshair; points:[{label,value}]. Renders into
  // `host` and wires a mousemove readout into `readoutEl`.
  function trendChart(host, points, readoutEl, fmtVal) {
    fmtVal = fmtVal || ((v) => Math.round(v));
    const col = _C(), w = 560, h = 120;
    if (!points || !points.length) { host.innerHTML = `<div style="font:italic 12px 'Fraunces',serif;color:var(--ink-mute);padding:12px 0">Trend builds over the coming days — one snapshot is recorded per day.</div>`; return; }
    const vals = points.map(p => p.value).filter(v => v != null);
    const max = Math.max(100, ...vals), min = Math.min(0, ...vals);
    const X = (i) => (points.length <= 1 ? w / 2 : (i / (points.length - 1)) * w);
    const Y = (v) => h - ((v - min) / (max - min || 1)) * (h - 10) - 5;
    const line = points.map((p, i) => `${X(i).toFixed(1)},${Y(p.value).toFixed(1)}`).join(" ");
    const dots = points.map((p, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(p.value).toFixed(1)}" r="2.4" fill="${col.ok}"/>`).join("");
    host.innerHTML = `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block;height:${h}px;cursor:crosshair;overflow:visible">
      <polyline points="${line}" fill="none" stroke="${col.ok}" stroke-width="1.8" stroke-linejoin="round"/>${dots}
      <line id="ov-tr-cross" x1="0" y1="0" x2="0" y2="${h}" stroke="${col.ink}" stroke-width="1" opacity="0" stroke-dasharray="3 3"/></svg>`;
    const svg = host.querySelector("svg"), cross = host.querySelector("#ov-tr-cross");
    svg.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect();
      const i = Math.round(((e.clientX - r.left) / r.width) * (points.length - 1));
      const p = points[Math.max(0, Math.min(points.length - 1, i))];
      if (!p) return;
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", "0.4");
      if (readoutEl) readoutEl.textContent = `${p.label}: ${fmtVal(p.value)}`;
    });
    svg.addEventListener("mouseleave", () => { cross.setAttribute("opacity", "0"); if (readoutEl) readoutEl.textContent = ""; });
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
      // Pure-integer values (requests, rate-limited) count-up cleanly via the
      // per-element rAF (odometer needs a CSS contract absent on this page and
      // garbled the gauge). Mixed strings ("96.4%", "287ms") cross-fade.
      const intVal = /^[\d,]+$/.test(valStr) ? Number(valStr.replace(/,/g, "")) : null;
      if (intVal != null) {
        countUp(numEl, intVal, { fmt: (v) => Math.round(v).toLocaleString() });
      } else if (window.APIN && APIN.fx && APIN.fx.fadeReplace && numEl.textContent !== valStr && numEl.textContent !== "·") {
        APIN.fx.fadeReplace(numEl, () => { numEl.textContent = valStr; });
        numEl._cuVal = null;
      } else { numEl.textContent = valStr; numEl._cuVal = null; }
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
    const titles = { requests: "Requests", success_rate: "Success rate", p50_ms: "p50 latency", rate_limited: "Rate-limit / quota" };
    const d = (DATA && DATA.kpis && DATA.kpis[name]) || {};
    if (name === "requests") _kpiRequests(panel, d);
    else if (name === "success_rate") _kpiSuccess(panel, d);
    else if (name === "p50_ms") _kpiLatency(panel, d);
    else _kpiRateLimit(panel, d);
    return titles[name] || name;
  }
  // REQUESTS — range toggle + stacked-by-status timeseries + busiest bucket
  //            + endpoint bars + vs-account-average.
  function _kpiRequests(panel, d) {
    const acct = (DATA && DATA.account) || {};
    const vsAvg = acct.this_vs_avg;
    const dl = d.delta_pct;
    panel.innerHTML =
      `<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap">
        <span id="ov-rq-num" style="font:700 42px/1 'Fraunces',serif;color:var(--ink)">${fmtNum(d.value)}</span>
        <span style="font:12px 'JetBrains Mono',monospace;color:var(--ink-soft)">vs previous period ${fmtNum(d.prev)}${dl != null ? ` · <b style="color:${dl >= 0 ? "var(--c-ok)" : "var(--c-danger)"}">${dl > 0 ? "+" : ""}${dl}%</b>` : ""}</span>
        ${vsAvg != null ? `<span style="font:12px 'JetBrains Mono',monospace;color:var(--ink-mute)">· ${vsAvg}× account average (${fmtNum(acct.avg_requests_per_key)}/key)</span>` : ""}
      </div>
      <div style="display:flex;justify-content:flex-end;margin-top:6px"><div class="ov-range" id="ov-rq-range"><button data-r="1h">1h</button><button data-r="24h">24h</button><button data-r="7d">7d</button></div></div>
      ${_kxSec("requests over time · stacked by status")}
      <div id="ov-rq-ts"></div>
      <div style="display:flex;gap:16px;font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute);margin:5px 0 2px"><span><i class="rdot s-2xx"></i> 2xx</span><span><i class="rdot s-4xx"></i> 4xx</span><span><i class="rdot s-5xx"></i> 5xx</span></div>
      ${_kxSec("busiest bucket")}<div id="ov-rq-busy" style="font:13px/1.6 'Fraunces',serif;color:var(--ink-soft)"></div>
      ${_kxSec("endpoint breakdown")}<div id="ov-rq-eps"></div>`;
    function paint(ts, grid, total) {
      ts = ts || { req: [] }; grid = grid || [];
      $("ov-rq-ts").innerHTML = stackedBarsSvg(ts);
      let bi = -1, bmax = -1;
      (ts.req || []).forEach((v, i) => { if (v > bmax) { bmax = v; bi = i; } });
      let busy = "no traffic in this window";
      if (bi >= 0 && bmax > 0 && ts.t0_ms && ts.bucket_ms) {
        const tmid = ts.t0_ms + (bi + 0.5) * ts.bucket_ms;
        busy = new Date(tmid).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + ` · ${bmax} requests`;
      }
      $("ov-rq-busy").textContent = busy;
      const eps = grid.slice(0, 6).map(s => ({ label: s.path, value: s.count, sub: fmtNum(s.count) }));
      $("ov-rq-eps").innerHTML = eps.length ? hbars(eps) : `<div style="font:italic 12px 'Fraunces',serif;color:var(--ink-mute)">no endpoint data</div>`;
      const numEl = $("ov-rq-num"); if (numEl && total != null) numEl.textContent = fmtNum(total);
    }
    // initial paint from the already-loaded window
    const rng = panel.querySelector("#ov-rq-range");
    rng.querySelectorAll("button").forEach(b => { if (b.getAttribute("data-r") === RANGE) b.setAttribute("aria-pressed", "true"); });
    paint(DATA && DATA.timeseries, DATA && DATA.spark_grid, d.value);
    rng.querySelectorAll("button").forEach(b => b.addEventListener("click", async () => {
      const w = b.getAttribute("data-r");
      rng.querySelectorAll("button").forEach(x => x.removeAttribute("aria-pressed"));
      b.setAttribute("aria-pressed", "true");
      const { ok, body } = await api(`/api/account/keys/${encodeURIComponent(PID)}/overview?window=${w}`);
      if (!ok || !body) return;
      const dd = body.data || body;
      paint(dd.timeseries, dd.spark_grid, (dd.kpis && dd.kpis.requests) ? dd.kpis.requests.value : null);
    }));
  }
  // SUCCESS — status donut + table; click a slice/row → ribbon filtered
  function _kpiSuccess(panel, d) {
    const sc = (DATA && DATA.status_counts) || { n_2xx: 0, n_4xx: 0, n_5xx: 0, total: 0 };
    const col = _C(), tot = sc.total || 0;
    const parts = [
      { key: "2xx", label: "2xx success", value: sc.n_2xx, color: col.ok },
      { key: "4xx", label: "4xx client", value: sc.n_4xx, color: col.amber },
      { key: "5xx", label: "5xx server", value: sc.n_5xx, color: col.danger },
    ];
    const rowPct = (v) => tot ? ((100 * v / tot).toFixed(1) + "%") : "—";
    panel.innerHTML =
      `<div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap">
        <div>${donutSvg(parts)}</div>
        <div style="flex:1;min-width:210px">
          ${[["2xx", sc.n_2xx, col.ok], ["4xx", sc.n_4xx, col.amber], ["5xx", sc.n_5xx, col.danger]].map(([k, v, c]) =>
            `<div class="ov-kx-statrow" data-slice="${k}" style="display:grid;grid-template-columns:13px 50px 1fr 60px;gap:10px;align-items:center;padding:8px 7px;border-radius:7px;cursor:pointer;font:12px 'JetBrains Mono',monospace;transition:background .12s">
              <span style="width:11px;height:11px;border-radius:3px;background:${c}"></span>
              <span style="color:var(--ink)">${k}</span>
              <span style="color:var(--ink-mute)">${fmtNum(v)} requests</span>
              <span style="color:var(--ink-soft);text-align:right">${rowPct(v)}</span></div>`).join("")}
          <div style="font:italic 11px/1.5 'Fraunces',serif;color:var(--ink-mute);margin-top:12px;padding:0 7px">click a status → request ribbon filtered to it</div>
        </div>
      </div>`;
    panel.querySelectorAll("[data-slice]").forEach(el => {
      el.addEventListener("click", () => _openRibbonFiltered(el.getAttribute("data-slice")));
      if (el.classList.contains("ov-kx-statrow")) {
        el.addEventListener("mouseenter", () => el.style.background = "rgba(120,110,90,.10)");
        el.addEventListener("mouseleave", () => el.style.background = "");
      }
    });
  }
  // p50 LATENCY — histogram + p50/p95/p99 + bimodal callout + slowest endpoint
  function _kpiLatency(panel, d) {
    const lh = (DATA && DATA.latency_hist) || { edges: [], bins: [], p50: null, p95: null, p99: null };
    const slow = (DATA && DATA.slowest_endpoint) || null;
    const bins = lh.bins || [], max = Math.max(1, ...bins), peaks = [];
    for (let i = 1; i < bins.length - 1; i++) {
      if (bins[i] >= max * 0.35 && bins[i] >= bins[i - 1] && bins[i] >= bins[i + 1]) peaks.push(i);
    }
    const bimodal = peaks.length >= 2 && (peaks[peaks.length - 1] - peaks[0] >= 3);
    panel.innerHTML =
      `<div style="display:flex;gap:22px;align-items:baseline">
        ${[["p50", lh.p50], ["p95", lh.p95], ["p99", lh.p99]].map(([k, v]) =>
          `<div><span style="font:700 27px/1 'Fraunces',serif;color:var(--ink)">${fmtMs(v)}</span><div style="font:10px 'JetBrains Mono',monospace;color:var(--ink-mute);letter-spacing:.12em;margin-top:3px">${k}</div></div>`).join("")}
      </div>
      ${_kxSec("latency distribution · log-binned")}
      ${histSvg(lh)}
      ${bimodal ? `<div style="font:italic 12px/1.5 'Fraunces',serif;color:var(--c-amber);margin-top:10px">Bimodal — fast metadata responses and slow inference form two distinct clusters.</div>` : ""}
      ${(() => { const ts = (DATA && DATA.timeseries) || {}; return (ts.lat_p95 || []).some(v => v > 0)
        ? `${_kxSec("percentile fan over time")}${fanSvg(ts.lat_p50, ts.lat_p95, ts.lat_p99)}
           <div style="display:flex;gap:16px;font:10px 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:4px"><span style="color:${_C().ok}">— p50</span><span style="color:${_C().amber}">— p95</span><span style="color:${_C().danger}">- - p99</span></div>` : ""; })()}
      ${slow ? `${_kxSec("slowest endpoint (by p95)")}<div style="font:13px/1.6 'JetBrains Mono',monospace;color:var(--ink-soft)">${esc(slow.path)} · p95 ${fmtMs(slow.p95)}</div>` : ""}`;
  }
  // RATE-LIMIT / QUOTA — events + quota burndown radial
  function _kpiRateLimit(panel, d) {
    const cap = (((DATA && DATA.health) || {}).pillars || {}).capacity || {};
    const rl = d.value || 0, ql = cap.quota_label || "unlimited";
    let radial, m = /^(\d+)\s*\/\s*(\d+)$/.exec(ql);
    if (m) {
      const consumed = +m[1], quota = +m[2], frac = quota ? Math.min(1, consumed / quota) : 0;
      radial = radialSvg(frac, Math.round(frac * 100) + "%", fmtNum(consumed) + " / " + fmtNum(quota));
    } else radial = radialSvg(0, "∞", "no daily cap");
    const events = (DATA && DATA.rate_limit_events) || [];
    let timeline = "";
    if (events.length) {
      const max = Math.max(1, ...events.map(e => e.count));
      timeline = `${_kxSec("rate-limit events")}
        <div style="display:flex;align-items:flex-end;gap:2px;height:60px;border-bottom:1px solid var(--paper-edge);padding-bottom:2px">
          ${events.slice(-80).map(e => `<span title="${esc(e.minute)} · ${e.count}" style="flex:1;min-width:2px;height:${Math.max(6, (e.count / max) * 56)}px;background:var(--c-amber);border-radius:1px"></span>`).join("")}</div>
        <div style="font:10px 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:4px">${events.length} minute(s) with throttling · most recent ${esc(events[events.length - 1].minute)}</div>`;
    }
    panel.innerHTML =
      `<div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap">
        <div>${radial}</div>
        <div style="flex:1;min-width:210px">
          <div style="font:700 38px/1 'Fraunces',serif;color:${rl ? "var(--c-amber)" : "var(--ink)"}">${fmtNum(rl)}</div>
          <div style="font:12px 'JetBrains Mono',monospace;color:var(--ink-soft);margin-top:5px">rate-limit events · ${esc(DATA ? DATA.window : "")}</div>
          <div style="font:13px/1.6 'Fraunces',serif;color:var(--ink-soft);margin-top:15px">${rl ? `${rl} throttle hit(s) this window — the integration is bumping the rate limit.` : "0 events — healthy. No throttling this window."}</div>
          <div style="font:11px 'JetBrains Mono',monospace;color:var(--ink-mute);margin-top:11px">quota · ${esc(String(ql))}</div>
        </div>
      </div>${timeline}`;
  }
  // open the ribbon lightbox pre-filtered to a status bucket (donut cross-link)
  let _ribbonPreset = null;
  function _openRibbonFiltered(status) {
    if (!window.APIN || !APIN.lightbox) return;
    _ribbonPreset = status;
    const title = "Request Ribbon" + (DATA && DATA.key ? " · " + (DATA.key.name || "") : "");
    const reopen = () => APIN.lightbox.open({ title, build: expandRibbon });
    if (APIN.lightbox.isOpen && APIN.lightbox.isOpen()) { APIN.lightbox.close({ skipAnim: true }); setTimeout(reopen, 70); }
    else reopen();
  }

  // ════════════════════ WIDGET 4 · PERSONALITY ═════════════════════════
  function renderPersonality(p) {
    const host = $("ov-personality-body");
    if (!host || !p) return;
    // Bento shows the 3 headline tags; the radar + full 6-dim table live in
    // the expanded view (⤢).
    const tags = (p.tags || []).slice(0, 3);
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
    const p = (DATA && DATA.personality) || {};
    const tags = p.tags || [];
    const vec = p.vector || {};
    const acct = p.account_vector || null;
    const similar = p.similar_keys || [];
    const col = _C();
    const DIMS = ["predict_heavy", "read_mostly", "write_heavy", "bursty", "error_tolerant", "steady"];
    const LBL = { predict_heavy: "predict-heavy", read_mostly: "read-mostly", write_heavy: "write-heavy", bursty: "bursty", error_tolerant: "error-tolerant", steady: "steady" };
    const sigByName = {}; tags.forEach(t => sigByName[t.name] = t.signal);
    const hasAcct = !!acct && DIMS.some(k => (acct[k] || 0) > 0);
    panel.innerHTML =
      `<div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap">
        <div style="flex:none">${radarSvg(DIMS, DIMS.map(k => LBL[k]), vec, hasAcct ? acct : null, 300)}</div>
        <div style="flex:1;min-width:230px">
          <div style="display:flex;gap:14px;font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute);margin-bottom:8px">
            <span><i style="display:inline-block;width:10px;height:10px;background:rgba(47,111,62,0.5);border:1.5px solid ${col.ok};vertical-align:middle"></i> this key</span>
            ${hasAcct ? `<span><i style="display:inline-block;width:10px;height:0;border-top:2px dashed ${col.mute};vertical-align:middle"></i> account avg</span>` : ""}</div>
          <div style="font:13px/1.7 'Fraunces',serif;color:var(--ink-soft)">Behavioural fingerprint from this key's request mix and timing${hasAcct ? ", overlaid on your account average" : ""}.</div>
        </div>
      </div>
      <div style="margin-top:16px;font:500 italic 11px 'Fraunces',serif;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-soft);border-bottom:1px solid var(--paper-edge);padding-bottom:6px">dimensions</div>
      <div style="display:grid;grid-template-columns:120px 60px 60px 1fr;gap:6px 10px;align-items:center;font:11px 'JetBrains Mono',monospace;margin-top:10px">
        <span style="color:var(--ink-mute)">DIMENSION</span><span style="color:var(--ink-mute);text-align:right">THIS</span><span style="color:var(--ink-mute);text-align:right">ACCT</span><span style="color:var(--ink-mute)">SIGNAL</span>
        ${DIMS.map(k => `<span style="color:var(--ink)">${LBL[k]}</span>
          <span style="text-align:right;color:var(--ink-soft)">${Math.round((vec[k] || 0) * 100)}%</span>
          <span style="text-align:right;color:var(--ink-mute)">${hasAcct ? Math.round((acct[k] || 0) * 100) + "%" : "—"}</span>
          <span style="color:var(--ink-mute);font-size:10px">${esc(sigByName[LBL[k]] || "")}</span>`).join("")}
      </div>
      <div style="margin-top:16px;font:500 italic 11px 'Fraunces',serif;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-soft);border-bottom:1px solid var(--paper-edge);padding-bottom:6px">similar keys</div>
      <div id="ov-pz-similar" style="margin-top:10px">${
        similar.length
          ? similar.map(s => `<div class="ov-pz-row" data-pid="${esc(s.public_id)}" style="display:flex;justify-content:space-between;align-items:center;padding:7px 8px;border-radius:7px;cursor:pointer;font:12px 'JetBrains Mono',monospace;transition:background .12s"><span style="color:var(--ink)">${esc(s.name)}</span><span style="color:var(--ink-soft)">${s.match}% match</span></div>`).join("")
          : `<div style="font:italic 12px 'Fraunces',serif;color:var(--ink-mute)">No peer keys with traffic to compare against yet.</div>`
      }</div>`;
    panel.querySelectorAll(".ov-pz-row").forEach(r => {
      r.addEventListener("mouseenter", () => r.style.background = "rgba(120,110,90,.10)");
      r.addEventListener("mouseleave", () => r.style.background = "");
      r.addEventListener("click", () => { location.href = "/account/api/keys/" + encodeURIComponent(r.getAttribute("data-pid")); });
    });
  }

  // ════════════════════ WIDGET 5 · SPARK-GRID ══════════════════════════
  let _sparkUid = 0;
  // Rich sparkline: gradient area-fill under a line, with a dot on the latest
  // point. opts.color overrides the stroke (default --c-ok). Each call gets a
  // unique gradient id so multiple sparklines on one page don't collide.
  function sparkSvg(buckets, w, h, opts) {
    w = w || 90; h = h || 24; opts = opts || {};
    const stroke = opts.color || "var(--c-ok)";
    const n = buckets.length;
    if (!n) return `<svg class="ov-spark-svg" viewBox="0 0 ${w} ${h}"></svg>`;
    const max = Math.max(1, ...buckets);
    const step = w / Math.max(1, n - 1);
    const xy = buckets.map((v, i) => [i * step, h - (v / max) * (h - 3) - 1.5]);
    const line = xy.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    const area = `0,${h} ` + line + ` ${w},${h}`;
    const uid = "sg" + (++_sparkUid), last = xy[xy.length - 1];
    return `<svg class="ov-spark-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <defs><linearGradient id="${uid}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${stroke}" stop-opacity="0.28"/>
        <stop offset="1" stop-color="${stroke}" stop-opacity="0"/></linearGradient></defs>
      <polygon points="${area}" fill="url(#${uid})" stroke="none"/>
      <polyline points="${line}" fill="none" stroke="${stroke}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="1.8" fill="${stroke}"/></svg>`;
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
    const col = _C();
    const COLS = "1.6fr 112px 50px 56px 56px 48px 66px";
    const metricKey = { requests: "buckets", latency: "buckets_lat", errors: "buckets_err", bytes: "buckets_bytes" };
    const metricColor = { requests: col.ok, latency: col.amber, errors: col.danger, bytes: col.soft };
    const fmtBytes = (b) => b == null ? "—" : b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : (b / 1048576).toFixed(1) + " MB";
    let metric = "requests", sortKey = "count", sortDir = -1;
    const sortVal = (s, k) => k === "count" ? s.count : k === "p50" ? (s.p50 || 0) : k === "p95" ? (s.p95 || 0) : k === "err" ? (s.err_pct || 0) : (s.bytes_total || 0);
    const sorted = () => grid.slice().sort((a, b) => (sortVal(a, sortKey) - sortVal(b, sortKey)) * sortDir);
    if (!grid.length) { panel.innerHTML = `<div style="font:italic 13px/1.6 'Fraunces',serif;color:var(--ink-mute);padding:20px">No endpoint traffic in this window yet.</div>`; return; }
    panel.innerHTML =
      `<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px">
        <span style="font:12px 'JetBrains Mono',monospace;color:var(--ink-soft)">top endpoints · ${esc(DATA ? DATA.window : "")}</span>
        <div class="ov-range" id="ov-ep-metric"><button data-m="requests" aria-pressed="true">requests</button><button data-m="latency">latency</button><button data-m="errors">errors</button><button data-m="bytes">bytes</button></div>
      </div>
      <div id="ov-ep-head" style="display:grid;grid-template-columns:${COLS};gap:8px;padding:0 8px 7px;border-bottom:1px solid var(--paper-edge);font:600 10px 'JetBrains Mono',monospace;letter-spacing:.05em;color:var(--ink-mute)">
        <span>ENDPOINT</span><span style="text-align:center">TREND</span>
        <span data-sort="count" data-base="n" style="text-align:right;cursor:pointer">n</span>
        <span data-sort="p50" data-base="p50" style="text-align:right;cursor:pointer">p50</span>
        <span data-sort="p95" data-base="p95" style="text-align:right;cursor:pointer">p95</span>
        <span data-sort="err" data-base="err%" style="text-align:right;cursor:pointer">err%</span>
        <span data-sort="bytes" data-base="bytes" style="text-align:right;cursor:pointer">bytes</span></div>
      <div id="ov-ep-rows" style="margin-top:4px"></div>`;
    const rowsHost = panel.querySelector("#ov-ep-rows");
    function paint() {
      const c = metricColor[metric], bk = metricKey[metric];
      rowsHost.innerHTML = sorted().map(s =>
        `<div class="ov-ep-row" data-path="${esc(s.path)}" style="display:grid;grid-template-columns:${COLS};gap:8px;align-items:center;padding:7px 8px;border-radius:7px;cursor:pointer;font:11.5px 'JetBrains Mono',monospace;transition:background .12s">
          <span style="color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(s.path)}">${esc(s.path)}</span>
          <span style="height:26px">${sparkSvg(s[bk] || [], 112, 26, { color: c })}</span>
          <span style="color:var(--ink-soft);text-align:right">${fmtNum(s.count)}</span>
          <span style="color:var(--ink-soft);text-align:right">${fmtMs(s.p50)}</span>
          <span style="color:var(--ink-soft);text-align:right">${fmtMs(s.p95)}</span>
          <span style="text-align:right;color:${s.err_pct > 0 ? "var(--c-danger)" : "var(--ink-mute)"}">${s.err_pct != null ? s.err_pct + "%" : "—"}</span>
          <span style="color:var(--ink-mute);text-align:right">${fmtBytes(s.bytes_total)}</span></div>`).join("");
      rowsHost.querySelectorAll(".ov-ep-row").forEach(r => {
        const path = r.getAttribute("data-path");
        r.addEventListener("mouseenter", () => { r.style.background = "rgba(120,110,90,.10)"; Bus.emit("hover:endpoint", path); });
        r.addEventListener("mouseleave", () => { r.style.background = ""; Bus.emit("hover:endpoint", null); });
        r.addEventListener("click", () => { if (window.APIN && APIN.lightbox) APIN.lightbox.close(); location.hash = "#requests"; });
      });
      panel.querySelectorAll("#ov-ep-head [data-sort]").forEach(thx => {
        const active = thx.getAttribute("data-sort") === sortKey;
        thx.style.color = active ? "var(--ink)" : "var(--ink-mute)";
        thx.textContent = thx.getAttribute("data-base") + (active ? (sortDir < 0 ? " ▾" : " ▴") : "");
      });
    }
    paint();
    panel.querySelectorAll("#ov-ep-metric button").forEach(b => b.addEventListener("click", () => {
      panel.querySelectorAll("#ov-ep-metric button").forEach(x => x.removeAttribute("aria-pressed"));
      b.setAttribute("aria-pressed", "true"); metric = b.getAttribute("data-m"); paint();
    }));
    panel.querySelectorAll("#ov-ep-head [data-sort]").forEach(thx => thx.addEventListener("click", () => {
      const k = thx.getAttribute("data-sort");
      if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = -1; }
      paint();
    }));
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
  // Build a supporting mini-chart that "proves" an insight, keyed off its text.
  function _insightChart(text) {
    const t = (text || "").toLowerCase();
    const col = _C();
    if (/p95|p50|latency|slow|fast|degrad|improv/.test(t)) {
      const ts = (DATA && DATA.timeseries) || {};
      if ((ts.lat_p95 || []).some(v => v > 0)) return fanSvg(ts.lat_p50, ts.lat_p95, ts.lat_p99, 460, 90);
    }
    if (/error|5xx|4xx|fail/.test(t)) {
      const sc = (DATA && DATA.status_counts) || {};
      return donutSvg([
        { key: "2xx", label: "2xx", value: sc.n_2xx || 0, color: col.ok },
        { key: "4xx", label: "4xx", value: sc.n_4xx || 0, color: col.amber },
        { key: "5xx", label: "5xx", value: sc.n_5xx || 0, color: col.danger }], 120, 20);
    }
    if (/predict|inference/.test(t)) {
      const grid = (DATA && DATA.spark_grid) || [];
      if (grid.length) return hbars(grid.slice(0, 5).map(s => ({ label: s.path, value: s.count, sub: fmtNum(s.count) })));
    }
    if (/ip|integration|shared|leak/.test(t)) {
      const hyg = (((DATA && DATA.health) || {}).pillars || {}).hygiene || {};
      return `<div style="font:11.5px/1.7 'JetBrains Mono',monospace;color:var(--ink-soft)">distinct IPs this window: ${hyg.distinct_ips != null ? hyg.distinct_ips : "—"}${hyg.ip_fanout ? " · <b style='color:var(--c-danger)'>fan-out flagged</b>" : " · stable"}</div>`;
    }
    if (/healthy|grade|health/.test(t)) {
      const P = (((DATA && DATA.health) || {}).pillars) || {};
      return hbars([["reliability", "RELIABILITY"], ["performance", "PERFORMANCE"], ["capacity", "CAPACITY"], ["hygiene", "HYGIENE"]]
        .map(([k, L]) => ({ label: L, value: (P[k] && P[k].score) || 0, sub: (P[k] && P[k].score != null) ? P[k].score : "—" })));
    }
    return null;
  }
  function expandInsights(panel) {
    const list = (DATA && DATA.insights) || [];
    const byTone = { great: 0, warn: 0, info: 0 };
    list.forEach(i => byTone[i.tone] = (byTone[i.tone] || 0) + 1);
    if (!list.length) { panel.innerHTML = `<div style="font:italic 13px/1.6 'Fraunces',serif;color:var(--ink-mute);padding:16px">Operating cleanly — no notable signals this window.</div>`; return; }
    panel.innerHTML = `<div style="font:12px/1.6 'JetBrains Mono',monospace;color:var(--ink-soft);margin-bottom:12px">⚠ ${byTone.warn || 0} warning · ℹ ${byTone.info || 0} info · ✓ ${byTone.great || 0} positive</div>` +
      list.map((i, idx) => {
        const hasChart = !!_insightChart(i.text);
        return `<div class="ov-insight ${esc(i.tone || "info")}" style="display:block;margin-bottom:8px">
          <div style="display:flex;align-items:flex-start;gap:6px"><i class="ins-icon"></i><span style="flex:1">${esc(i.text)}</span>
          ${hasChart ? `<button class="ov-ins-toggle" data-i="${idx}" style="background:none;border:none;cursor:pointer;color:var(--ink-soft);font:13px 'JetBrains Mono',monospace">▸</button>` : ""}</div>
          ${hasChart ? `<div class="ov-ins-chart" data-i="${idx}" hidden style="margin:8px 0 2px 22px"></div>` : ""}</div>`;
      }).join("");
    panel.querySelectorAll(".ov-ins-toggle").forEach(btn => {
      btn.addEventListener("click", () => {
        const idx = btn.getAttribute("data-i");
        const box = panel.querySelector(`.ov-ins-chart[data-i="${idx}"]`);
        if (!box) return;
        if (box.hidden) { box.innerHTML = _insightChart(list[idx].text) || ""; box.hidden = false; btn.textContent = "▾"; }
        else { box.hidden = true; btn.textContent = "▸"; }
      });
    });
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
