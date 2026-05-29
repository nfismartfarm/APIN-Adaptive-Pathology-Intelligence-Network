// 9.N.T29 · Usage empty-state — a hand-drawn "seed → sprout" hero for the
// account-wide Usage page, in our paper-ink language (no emoji).
//
// Three states, chosen by the caller from the lifetime probe + filters:
//   new       — zero traffic ever in scope → onboarding (self-typing terminal
//               demo + docs/quickstart/sandbox CTAs + copy curl + test ping)
//   dormant   — traffic exists but this window is empty → last-request line
//               (clickable) + a window-ladder with data-dots + jump button
//   filtered  — endpoint/status filters are hiding everything → clear filters
//
// Ghost teaser charts fill the small cards so they hint at value instead of
// going blank. All motion respects prefers-reduced-motion (CSS-frozen).
//
// APIN.usageEmpty.hero(hostEl, ctx) -> { destroy }
// APIN.usageEmpty.ghost(hostEl, kind)
(function () {
  "use strict";
  if (!window.APIN) window.APIN = {};
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let _uid = 0;

  // ── a small almond leaf centred near (x, y), pointing up ──
  function _leafPath(x, y, s) {
    const f = (n) => n.toFixed(1);
    return `M${f(x)} ${f(y)} C ${f(x - s)} ${f(y - s * 0.4)}, ${f(x - s * 0.5)} ${f(y - s * 1.7)}, ${f(x)} ${f(y - s * 1.9)} `
      + `C ${f(x + s * 0.5)} ${f(y - s * 1.7)}, ${f(x + s)} ${f(y - s * 0.4)}, ${f(x)} ${f(y)} Z`;
  }

  // ── hand-drawn tree (gentle ink wobble); dormant/filtered shed leaves in a
  //    loop, new shows a young sapling. ──
  function _sprout(state, u) {
    const young = (state === "new");
    const GD = "var(--green-deep,#2d6a4f)", AC = "var(--c-accent,#52b788)";
    const filt = `<filter id="${u}-w" x="-6%" y="-6%" width="112%" height="112%">`
      + `<feTurbulence type="fractalNoise" baseFrequency="0.03" numOctaves="1" seed="6" result="t"/>`
      + `<feDisplacementMap in="SourceGraphic" in2="t" scale="0.6" xChannelSelector="R" yChannelSelector="G"/></filter>`;
    // soil
    const soil = `<path d="M42 160 Q120 154 198 160" stroke="var(--ink-soft,#6b6453)" stroke-width="1.3" fill="none" stroke-linecap="round" filter="url(#${u}-w)"/>`
      + `<path d="M58 165 L182 165" stroke="var(--ink-mute,#9a907a)" stroke-width="0.8" stroke-dasharray="1 5" opacity="0.5"/>`;
    // trunk + branches
    const trunk = `<g filter="url(#${u}-w)" stroke="${GD}" fill="none" stroke-linecap="round">`
      + `<path d="M120 158 C 117 140 122 120 120 100" stroke-width="${young ? 2 : 3}"/>`
      + (young ? ""
        : `<path d="M120 120 C 112 114 106 108 100 101" stroke-width="1.8"/>`
        + `<path d="M120 112 C 129 107 137 101 145 94" stroke-width="1.8"/>`
        + `<path d="M120 104 C 116 96 118 88 120 80" stroke-width="1.8"/>`)
      + `</g>`;
    // canopy
    const canopy = young
      ? `<g class="us-canopy" filter="url(#${u}-w)"><path d="M120 96 C 103 88 101 62 120 54 C 139 62 137 88 120 96 Z" fill="${AC}" fill-opacity="0.2" stroke="${GD}" stroke-width="1.6"/><path d="M120 92 L120 58" stroke="${GD}" stroke-width="0.7" opacity="0.6"/></g>`
      : `<g class="us-canopy" filter="url(#${u}-w)" stroke="${GD}" stroke-width="1.4">`
      + `<ellipse cx="120" cy="60" rx="31" ry="25" fill="${AC}" fill-opacity="0.15"/>`
      + `<ellipse cx="97" cy="76" rx="22" ry="18" fill="${AC}" fill-opacity="0.12"/>`
      + `<ellipse cx="143" cy="76" rx="22" ry="18" fill="${AC}" fill-opacity="0.12"/>`
      + `</g>`;
    // falling leaves (dormant/filtered) — each flutters differently via CSS vars
    const COL = [AC, "var(--ochre-deep,#b6822a)", GD, "var(--ink-mute,#9a907a)", AC, "var(--ochre-deep,#b6822a)", GD];
    const seeds = [[104, 58, 5, 0.0, 6.4, 20, -12], [134, 56, 4, 1.5, 7.2, -18, 14], [118, 50, 5, 2.7, 5.8, 24, -8],
                   [95, 74, 4, 3.7, 6.8, -24, 12], [146, 72, 5, 0.9, 7.6, 16, -16], [120, 64, 4, 4.7, 6.0, -12, 20], [110, 70, 5, 2.1, 8.0, 26, -6]];
    const leaves = young ? "" : `<g class="us-fall">`
      + seeds.map((s, i) => {
        const [x, y, sz, delay, dur, lx1, lx2] = s;
        return `<path class="us-leaf-fall" d="${_leafPath(x, y, sz)}" fill="${COL[i % COL.length]}" fill-opacity="0.82" `
          + `style="--lx1:${lx1}px;--lx2:${lx2}px;animation-delay:${delay}s;animation-duration:${dur}s"/>`;
      }).join("") + `</g>`;
    return `<svg class="us-sprout" viewBox="0 0 240 180" fill="none" aria-hidden="true"><defs>${filt}</defs>`
      + soil + trunk + canopy + leaves + `</svg>`;
  }

  // ── faint "ghost" teaser shapes for the small cards ──
  function _ghostSvg(kind) {
    const c = "var(--ink-mute,#9a907a)";
    if (kind === "donut") {
      return `<svg viewBox="0 0 90 90" aria-hidden="true"><circle cx="45" cy="45" r="30" fill="none" stroke="${c}" stroke-width="11" opacity="0.16"/>`
        + `<circle cx="45" cy="45" r="30" fill="none" stroke="${c}" stroke-width="11" stroke-dasharray="60 200" opacity="0.26" transform="rotate(-90 45 45)"/></svg>`;
    }
    if (kind === "bars") {
      const hs = [22, 40, 30, 52, 34, 46, 26, 38];
      return `<svg viewBox="0 0 160 60" preserveAspectRatio="none" aria-hidden="true">`
        + hs.map((h, i) => `<rect x="${4 + i * 19}" y="${58 - h}" width="12" height="${h}" rx="2" fill="${c}" opacity="0.14"/>`).join("") + `</svg>`;
    }
    if (kind === "rows") {
      return `<svg viewBox="0 0 160 70" preserveAspectRatio="none" aria-hidden="true">`
        + [0, 1, 2, 3].map(i => `<rect x="4" y="${4 + i * 17}" width="${120 - i * 22}" height="9" rx="3" fill="${c}" opacity="${0.18 - i * 0.03}"/>`).join("") + `</svg>`;
    }
    if (kind === "treemap") {
      const cells = [[4, 4, 70, 50], [78, 4, 78, 28], [78, 36, 38, 30], [120, 36, 36, 30], [4, 58, 46, 8], [54, 58, 46, 8]];
      return `<svg viewBox="0 0 160 70" preserveAspectRatio="none" aria-hidden="true">`
        + cells.map((r, i) => `<rect x="${r[0]}" y="${r[1]}" width="${r[2]}" height="${r[3]}" rx="3" fill="${c}" opacity="${0.16 - (i % 3) * 0.03}"/>`).join("") + `</svg>`;
    }
    // default: a soft line
    return `<svg viewBox="0 0 160 56" preserveAspectRatio="none" aria-hidden="true">`
      + `<path d="M2 44 C 30 30, 50 38, 78 22 C 104 8, 130 30, 158 18" fill="none" stroke="${c}" stroke-width="2" opacity="0.2"/></svg>`;
  }

  // ── window-ladder: which standard windows contain the last request ──
  function _ladder(ctx) {
    const ranges = ctx.ranges || [];
    const lastMs = ctx.lastTs ? Date.parse(_isoUtc(ctx.lastTs)) : null;
    const now = ctx.now || Date.now();
    return ranges.map(r => ({ id: r.id, label: r.label, has: (lastMs != null && (now - lastMs) <= r.ms) }));
  }
  function _isoUtc(s) { s = String(s).trim().replace(" ", "T"); return /[zZ]$|[+-]\d\d:?\d\d$/.test(s) ? s : s + "Z"; }
  function _suggest(ladder) { const f = ladder.find(x => x.has); return f ? f.id : null; }
  function _fmtWhen(ts) {
    if (window.APIN && APIN.time && APIN.time.localFull) return APIN.time.localFull(ts);
    return String(ts || "");
  }
  function _ago(ts) { return (window.APIN && APIN.time && APIN.time.ago) ? APIN.time.ago(ts) : ""; }

  // ── self-typing terminal (returns a stop fn) ──
  function _typewriter(el, lines) {
    if (!el) return () => {};
    let alive = true, timers = [];
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function run() {
      if (!alive) return;
      el.innerHTML = "";
      if (reduce) { el.innerHTML = lines.map(l => `<div class="us-term-line ${l.cls || ""}">${esc(l.t)}</div>`).join(""); return; }
      let li = 0;
      function typeLine() {
        if (!alive || li >= lines.length) { timers.push(setTimeout(run, 3600)); return; }   // loop
        const line = lines[li], row = document.createElement("div");
        row.className = "us-term-line " + (line.cls || ""); el.appendChild(row);
        let ci = 0;
        function ch() {
          if (!alive) return;
          row.textContent = line.t.slice(0, ci) + (ci < line.t.length ? "▏" : "");
          if (ci <= line.t.length) { ci++; timers.push(setTimeout(ch, 18 + Math.random() * 26)); }
          else { row.textContent = line.t; li++; timers.push(setTimeout(typeLine, line.pause || 360)); }
        }
        ch();
      }
      typeLine();
    }
    run();
    return () => { alive = false; timers.forEach(clearTimeout); };
  }

  function _btn(cls, icon, label, attrs) {
    return `<button class="us-cta ${cls}" ${attrs || ""}>`
      + (icon ? `<svg class="icon" aria-hidden="true"><use href="#${icon}"/></svg>` : "")
      + esc(label) + `</button>`;
  }

  function hero(host, ctx) {
    if (!host) return { destroy() {} };
    ctx = ctx || {};
    const u = "us" + (++_uid);
    const state = ctx.state || "dormant";
    let stop = () => {};

    let body = "";
    if (state === "new") {
      const sample = ctx.sampleCurl || 'curl -H "Authorization: Bearer apin_live_…" https://your-host/api/version';
      body =
        `<div class="us-copy">`
        + `<h3 class="us-title">No requests yet</h3>`
        + `<p class="us-sub">your dashboard grows as traffic flows — make your first call</p>`
        + `<div class="us-term" id="${u}-term"></div>`
        + `<div class="us-curl"><code id="${u}-curl">${esc(sample)}</code>`
        + `<button class="us-copybtn" id="${u}-copy" title="copy"><svg class="icon" aria-hidden="true"><use href="#i-clipboard"/></svg></button></div>`
        + `<div class="us-ctas">`
        + _btn("primary", "i-book", "Read the docs", `data-href="${esc(ctx.docsUrl || "/docs")}"`)
        + _btn("", "i-flask", "Quickstart", `data-href="${esc(ctx.quickstartUrl || "/account/api/quickstart")}"`)
        + _btn("", "i-zap", "Go to Sandbox", `data-href="${esc(ctx.sandboxUrl || "/account/api/sandbox")}"`)
        + _btn("ghost", "i-activity", "Send a test ping", `id="${u}-ping"`)
        + `</div>`
        + `<p class="us-listening" id="${u}-listen"><span class="us-dot"></span> listening — your first request appears here live</p>`
        + `</div>`;
    } else if (state === "filtered") {
      const chips = (ctx.filterChips || []).map(c => `<span class="us-chip">${esc(c)}</span>`).join("");
      const lad = _ladder(ctx), sug = _suggest(lad);
      body =
        `<div class="us-copy">`
        + `<h3 class="us-title">No matches for these filters</h3>`
        + `<p class="us-sub">nothing in <b>${esc(ctx.rangeLabel || "this window")}</b> matches:</p>`
        + `<div class="us-chips">${chips}</div>`
        + `<div class="us-ctas">`
        + _btn("primary", "i-x", "Clear filters", `id="${u}-clear"`)
        + (sug ? _btn("", "i-clock", "Show last " + sug, `data-range="${sug}"`) : "")
        + `</div></div>`;
    } else {  // dormant
      const lad = _ladder(ctx), sug = _suggest(lad);
      const lt = ctx.lifetime || {};
      const lastLine = lt.last_ts
        ? `last request · <b class="us-meth">${esc(lt.last_method || "")}</b> ${esc(lt.last_path || "")}`
        + (lt.last_status != null ? ` · <span class="us-sc us-sc-${(+lt.last_status >= 500 ? "danger" : +lt.last_status >= 400 ? "amber" : "ok")}">${esc(lt.last_status)}</span>` : "")
        : "";
      const whenLine = lt.last_ts ? `${esc(_fmtWhen(lt.last_ts))} · ${esc(_ago(lt.last_ts))}` : "";
      const ladderHtml = lad.length
        ? `<div class="us-ladder">` + lad.map(x =>
          `<button class="us-rung${x.has ? " has" : ""}${x.id === ctx.range ? " cur" : ""}" data-range="${x.id}"><i></i>${esc(x.label)}</button>`).join("") + `</div>`
        : "";
      const xhint = ctx.metricEmptyButOthers
        ? `<p class="us-hint">no ${esc(ctx.metricEmptyButOthers.metric)} samples here — but <b>${esc(String(ctx.metricEmptyButOthers.otherCount))}</b> ${esc(ctx.metricEmptyButOthers.other)} in this window. <button class="us-link" data-metric="${esc(ctx.metricEmptyButOthers.other)}">switch →</button></p>`
        : "";
      const keyhint = ctx.keyScope
        ? `<p class="us-hint"><b>${esc(ctx.keyScope.name)}</b> has no requests yet — your account has <b>${esc(String(ctx.keyScope.accountTotal))}</b> across other keys. <button class="us-link" id="${u}-allkeys">view all keys →</button></p>`
        : "";
      const dctas = [
        (sug && sug !== ctx.range) ? _btn("primary", "i-clock", "Show last " + sug, `data-range="${sug}"`) : "",
        _btn("", "i-activity", "Send a test ping", `id="${u}-ping"`),
        _btn("", "i-zap", "Go to Sandbox", `data-href="${esc(ctx.sandboxUrl || "/account/api/sandbox")}"`),
        ctx.filtersActive ? _btn("ghost", "i-x", "Clear filters", `id="${u}-clear"`) : "",
      ].filter(Boolean).join("");
      body =
        `<div class="us-copy">`
        + `<h3 class="us-title">Quiet in ${esc(ctx.rangeLabel || "this window")}</h3>`
        + (lastLine ? `<p class="us-last"${lt.last_id ? ` data-rid="${esc(lt.last_id)}"` : ""}>${lastLine}</p>` : "")
        + (whenLine ? `<p class="us-when">${whenLine}</p>` : "")
        + ladderHtml
        + (dctas ? `<div class="us-ctas">${dctas}</div>` : "")
        + xhint + keyhint
        + `<p class="us-listening" id="${u}-listen"><span class="us-dot"></span> listening — new requests appear here live</p>`
        + `</div>`;
    }

    host.innerHTML = `<div class="us-empty us-${state}">${_sprout(state, u)}${body}</div>`;

    // ── wiring ──
    const $ = (id) => document.getElementById(id);
    host.querySelectorAll(".us-cta[data-href]").forEach(b =>
      b.addEventListener("click", () => { location.href = b.getAttribute("data-href"); }));
    host.querySelectorAll("[data-range]").forEach(b =>
      b.addEventListener("click", () => { if (ctx.onSetRange) ctx.onSetRange(b.getAttribute("data-range")); }));
    host.querySelectorAll(".us-link[data-metric]").forEach(b =>
      b.addEventListener("click", () => { if (ctx.onSetMetric) ctx.onSetMetric(b.getAttribute("data-metric")); }));
    const clr = $(u + "-clear"); if (clr) clr.addEventListener("click", () => { if (ctx.onClearFilters) ctx.onClearFilters(); });
    const allk = $(u + "-allkeys"); if (allk) allk.addEventListener("click", () => { if (ctx.onClearFilters) ctx.onClearFilters(); });
    const lastEl = host.querySelector(".us-last[data-rid]");
    if (lastEl) { lastEl.style.cursor = "pointer"; lastEl.addEventListener("click", () => { if (ctx.onOpenRequest) ctx.onOpenRequest(lastEl.getAttribute("data-rid")); }); }
    const copyBtn = $(u + "-copy"); if (copyBtn) copyBtn.addEventListener("click", () => {
      const t = ($(u + "-curl") || {}).textContent || "";
      try { navigator.clipboard.writeText(t); copyBtn.classList.add("ok"); setTimeout(() => copyBtn.classList.remove("ok"), 1200); } catch (_) {}
    });
    const ping = $(u + "-ping"); if (ping) ping.addEventListener("click", () => { if (ctx.onTestPing) ctx.onTestPing(ping); });

    if (state === "new") stop = _typewriter($(u + "-term"), [
      { t: "$ curl -H \"Authorization: Bearer apin_live_…\" \\", cls: "cmd", pause: 200 },
      { t: "       https://your-host/api/version", cls: "cmd", pause: 520 },
      { t: "← 200 · 14ms", cls: "ok", pause: 240 },
      { t: "  { \"version\": \"1.4.0\", \"status\": \"ok\" }", cls: "json", pause: 1400 },
    ]);

    return { destroy() { try { stop(); } catch (_) {} } };
  }

  function ghost(host, kind) {
    if (!host) return;
    host.innerHTML = `<div class="us-ghost">${_ghostSvg(kind)}<span class="us-ghost-lbl">awaiting traffic</span></div>`;
  }

  window.APIN.usageEmpty = { hero, ghost };
})();
