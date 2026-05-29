// 9.N.T28 · API Galaxy — a living endpoint ecosystem (hand-drawn SVG, 2D).
//
// Three fused layers, our paper-ink language, no emoji:
//   • Orbital constellation  — deterministic BFS-ring layout around a hub.
//     hub centre · ring = graph hop-distance · radius nudged by affinity ·
//     node size ∝ traffic. Stable (no force jitter) + gentle celestial drift.
//   • Neural pathways         — soft hand-drawn bézier strands (seeded wobble),
//     width ∝ transition count; brighten on hover / flow.
//   • Watershed flow          — ink particles travel hub→edge: idle trickle by
//     volume + live pulses on SSE events, with fading trails. Error states
//     bleed jade→gold→ember and raise stress halos on connected nodes.
//
// One requestAnimationFrame loop drives drift + particles. Exposes:
//   APIN.galaxy.create(hostEl, opts) -> handle {
//     setData, setFocus, setMode, highlightRoute, clearHighlight,
//     replay, stopReplay, pulse, pulseEdge, resize, dispose }
(function () {
  "use strict";
  if (!window.APIN) window.APIN = {};
  const SVGNS = "http://www.w3.org/2000/svg";

  // warm health ramp (jade → gold → ember) — shared with terrain/genome
  const JADE = [0.16, 0.62, 0.42], GOLD = [0.82, 0.62, 0.20], EMBER = [0.78, 0.31, 0.20];
  const _l3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const _rgb = (c) => `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
  function _ramp(err) { const t = Math.min(1, (err || 0) / 0.2); return t <= 0.5 ? _l3(JADE, GOLD, t / 0.5) : _l3(GOLD, EMBER, (t - 0.5) / 0.5); }
  function _hash(s) { let h = 2166136261 >>> 0; s = String(s || "?"); for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
  const _esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const _tail = (p) => { p = String(p || "?"); const s = p.split("/").filter(Boolean); return s.length ? "/" + s.slice(-2).join("/") : p; };
  const _short = (p) => { const s = String(p || "?").split("/").filter(Boolean); return s.length ? s[s.length - 1] : p; };
  const _fmtN = (n) => { n = +n || 0; return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); };
  const SLO = { metadata: 300, quick_inference: 8000, heavy_inference: 15000, default: 2000 };

  // hand-drawn (wobbled) closed circle path through `segs` points
  function _wobbleCircle(cx, cy, r, seed, jit) {
    let a = seed >>> 0; const rng = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const segs = 14, pts = [];
    for (let i = 0; i < segs; i++) { const ang = (i / segs) * Math.PI * 2; const rr = r * (1 + (rng() - 0.5) * (jit || 0.06)); pts.push([cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr]); }
    let d = "";
    for (let i = 0; i < segs; i++) { const p = pts[i], n = pts[(i + 1) % segs], n2 = pts[(i + 2) % segs]; const mx = (n[0] + n2[0]) / 2, my = (n[1] + n2[1]) / 2; if (i === 0) d += `M ${((p[0] + n[0]) / 2).toFixed(1)} ${((p[1] + n[1]) / 2).toFixed(1)} `; d += `Q ${n[0].toFixed(1)} ${n[1].toFixed(1)} ${mx.toFixed(1)} ${my.toFixed(1)} `; }
    return d + "Z";
  }

  function create(host, opts) {
    opts = opts || {};
    const mode0 = opts.mode || "compact";
    let endpoints = opts.endpoints || [];
    let edges = opts.edges || [];
    let hub = opts.hub || null;
    let focus = opts.focus || null;
    let viewMode = "flow";                       // flow | health | latency
    const enablePan = mode0 === "full";

    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("class", "gx-svg");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    host.appendChild(svg);
    const view = document.createElementNS(SVGNS, "g"); view.setAttribute("class", "gx-view"); svg.appendChild(view);
    const gEdges = document.createElementNS(SVGNS, "g"); gEdges.setAttribute("class", "gx-edges"); view.appendChild(gEdges);
    const gParts = document.createElementNS(SVGNS, "g"); gParts.setAttribute("class", "gx-particles"); view.appendChild(gParts);
    const gNodes = document.createElementNS(SVGNS, "g"); gNodes.setAttribute("class", "gx-nodes"); view.appendChild(gNodes);

    let W = 0, H = 0;
    let nodes = {};        // path -> { e, x, y, bx, by, r, hop, phase, el, ring, halo, core, label, drift }
    let elist = [];        // [{from,to,count,strength,pts,el,hit,baseW}]
    let adj = {};          // undirected adjacency path->Set
    let _hoverPath = null, _selPath = null, _routeSet = null, _routeEdgeSet = null;
    let zoom = 1, panX = 0, panY = 0, tZoom = 1, tPanX = 0, tPanY = 0;

    function _byPath() { const m = {}; endpoints.forEach(e => m[e.path] = e); return m; }

    let _layoutRetry = 0;
    function _computeLayout() {
      const r = svg.getBoundingClientRect(); W = r.width || host.clientWidth || 0; H = r.height || host.clientHeight || 0;
      if (W < 40 || H < 40) {          // not laid out yet — retry (backstop to ResizeObserver)
        if (_layoutRetry < 40 && !disposed) { _layoutRetry++; requestAnimationFrame(_computeLayout); }
        return;
      }
      _layoutRetry = 0;
      gEdges.innerHTML = ""; gParts.innerHTML = ""; gNodes.innerHTML = "";
      if (parts) parts.length = 0;     // drop orphaned particle refs (their DOM was just cleared)
      _replay = null;                  // any in-flight replay references stale edges
      nodes = {}; elist = []; adj = {};
      const em = _byPath();
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const cx = W / 2, cy = H / 2;
      if (!endpoints.length) return;

      // adjacency (undirected for placement)
      edges.forEach(l => { (adj[l.from] = adj[l.from] || new Set()).add(l.to); (adj[l.to] = adj[l.to] || new Set()).add(l.from); });
      const center = (mode0 === "compact" ? focus : (hub || focus)) || endpoints[0].path;

      let placed;            // [{path, hop}]
      if (mode0 === "compact") {
        // hub + its direct neighbours only (single ring), ranked by edge weight
        const wgt = {};
        edges.forEach(l => { if (l.from === center) wgt[l.to] = (wgt[l.to] || 0) + l.count; if (l.to === center) wgt[l.from] = (wgt[l.from] || 0) + l.count; });
        const neigh = Object.keys(wgt).filter(p => em[p]).sort((a, b) => wgt[b] - wgt[a]).slice(0, 8);
        placed = [{ path: center, hop: 0 }].concat(neigh.map(p => ({ path: p, hop: 1 })));
      } else {
        // BFS hop distance from hub; unreachable → outer field
        const dist = { [center]: 0 }; const q = [center];
        while (q.length) { const u = q.shift(); (adj[u] || new Set()).forEach(v => { if (dist[v] == null && em[v]) { dist[v] = dist[u] + 1; q.push(v); } }); }
        placed = endpoints.map(e => ({ path: e.path, hop: dist[e.path] != null ? Math.min(3, dist[e.path]) : 3 }));
      }

      const maxN = Math.max(1, ...endpoints.map(e => e.n || 0));
      const ringStep = Math.min(W, H) * (mode0 === "compact" ? 0.30 : 0.165);
      const byRing = {}; placed.forEach(p => (byRing[p.hop] = byRing[p.hop] || []).push(p));
      Object.keys(byRing).forEach(hopK => {
        const arr = byRing[hopK]; const hop = +hopK;
        arr.sort((a, b) => _hash(a.path) - _hash(b.path));   // stable angular order
        arr.forEach((p, i) => {
          const e = em[p.path] || { n: 0, err_rate: 0 };
          let x, y;
          if (hop === 0) { x = cx; y = cy; }
          else {
            const ang = (i / arr.length) * Math.PI * 2 + hop * 0.6;   // offset rings so they don't align
            // affinity strength to centre nudges radius inward
            let strg = 0; edges.forEach(l => { if ((l.from === center && l.to === p.path) || (l.to === center && l.from === p.path)) strg += l.count; });
            const pull = Math.min(0.18, strg / (maxN || 1) * 0.4);
            const rad = ringStep * hop * (1 - pull);
            x = cx + Math.cos(ang) * rad; y = cy + Math.sin(ang) * rad;
          }
          const nr = (hop === 0 ? 16 : 7) + Math.pow((e.n || 0) / maxN, 0.5) * (hop === 0 ? 8 : 12);
          nodes[p.path] = { e, x, y, bx: x, by: y, r: nr, hop, phase: _hash(p.path) % 1000 / 1000 * 6.28, drift: hop === 0 ? 0 : 1 };
        });
      });

      // ── edges ──
      edges.forEach(l => {
        const a = nodes[l.from], b = nodes[l.to]; if (!a || !b) return;
        const maxC = Math.max(1, ...edges.map(x => x.count));
        const baseW = 0.8 + (l.count / maxC) * 3.2;
        const path = document.createElementNS(SVGNS, "path");
        path.setAttribute("class", "gx-edge"); path.setAttribute("fill", "none"); path.setAttribute("stroke-width", baseW.toFixed(2)); path.setAttribute("stroke-linecap", "round");
        gEdges.appendChild(path);
        const hit = document.createElementNS(SVGNS, "path");
        hit.setAttribute("class", "gx-edge-hit"); hit.setAttribute("fill", "none"); hit.setAttribute("stroke", "transparent"); hit.setAttribute("stroke-width", "12");
        gEdges.appendChild(hit);
        const rec = { from: l.from, to: l.to, count: l.count, strength: l.count / maxC, el: path, hit, baseW, pts: [], emit: 0, gap: Math.max(420, 2600 - (l.count / maxC) * 2200) };
        hit.addEventListener("mousemove", (ev) => { if (opts.onHoverEdge) opts.onHoverEdge(rec, ev.clientX, ev.clientY); _emphasizeEdge(rec, true); });
        hit.addEventListener("mouseleave", () => { if (opts.onLeave) opts.onLeave(); _emphasizeEdge(rec, false); });
        elist.push(rec);
      });

      // ── nodes ──
      placed.forEach(p => {
        const nd = nodes[p.path], e = nd.e;
        const g = document.createElementNS(SVGNS, "g"); g.setAttribute("class", "gx-node" + (p.hop === 0 ? " gx-hub" : "")); g.style.cursor = "pointer";
        const halo = document.createElementNS(SVGNS, "circle"); halo.setAttribute("class", "gx-halo"); halo.setAttribute("r", (nd.r + 6).toFixed(1)); g.appendChild(halo);
        const stress = document.createElementNS(SVGNS, "circle"); stress.setAttribute("class", "gx-stress"); stress.setAttribute("r", (nd.r + 10).toFixed(1)); stress.setAttribute("fill", "none"); g.appendChild(stress);
        const ring = document.createElementNS(SVGNS, "path"); ring.setAttribute("class", "gx-ring"); ring.setAttribute("fill", "var(--paper,#f4efe4)"); ring.setAttribute("d", _wobbleCircle(0, 0, nd.r, _hash(p.path), e.err_rate > 0.08 ? 0.18 : 0.06)); g.appendChild(ring);
        const core = document.createElementNS(SVGNS, "circle"); core.setAttribute("class", "gx-core"); core.setAttribute("r", Math.max(1.5, nd.r * 0.32).toFixed(1)); g.appendChild(core);
        const isMin = p.hop !== 0 && (e.n || 0) < maxN * 0.12;   // declutter tiny nodes
        const label = document.createElementNS(SVGNS, "text"); label.setAttribute("class", "gx-label" + (p.hop === 0 ? " gx-label-hub" : "") + (isMin ? " gx-label-min" : "")); label.setAttribute("text-anchor", "middle"); label.setAttribute("y", (nd.r + 13).toFixed(1)); label.textContent = _short(p.path); g.appendChild(label);
        gNodes.appendChild(g);
        nd.el = g; nd.halo = halo; nd.ring = ring; nd.core = core; nd.label = label; nd.stress = stress;
        // pulse rate by recent activity
        const sp = e.spark || []; const recent = sp.slice(-4).reduce((s, v) => s + v, 0);
        halo.style.animationDuration = (recent > 0 ? Math.max(1.4, 3.2 - recent * 0.1) : 4.5) + "s";
        g.addEventListener("mousemove", (ev) => { _hoverPath = p.path; if (opts.onHoverNode) opts.onHoverNode(p.path, ev.clientX, ev.clientY); _emphasizeNode(p.path); });
        g.addEventListener("mouseleave", () => { _hoverPath = null; if (opts.onLeave) opts.onLeave(); _restyle(); });
        g.addEventListener("click", () => { if (opts.onClickNode) opts.onClickNode(p.path); });
      });
      _restyle();
    }

    // colour an edge by the current view mode
    function _edgeColor(rec) {
      const em = _byPath(), a = em[rec.from], b = em[rec.to];
      if (viewMode === "health") { const er = Math.max(a ? a.err_rate : 0, b ? b.err_rate : 0); return _rgb(_ramp(er)); }
      if (viewMode === "latency") { const T = SLO[(b && b.cls) || "default"] || 2000; const lr = Math.min(1, (b ? b.p95 : 0) / (T * 2)); return _rgb(_l3(JADE, EMBER, lr)); }
      return "var(--ink-soft,#6b6453)";
    }
    function _nodeColor(e) {
      if (viewMode === "latency") { const T = SLO[e.cls || "default"] || 2000; const lr = Math.min(1, (e.p95 || 0) / (T * 2)); return _rgb(_l3(JADE, EMBER, lr)); }
      return _rgb(_ramp(e.err_rate));
    }
    function _restyle() {
      const relNodes = _routeSet || (_hoverPath ? new Set([_hoverPath, ...(adj[_hoverPath] || [])]) : null);
      elist.forEach(rec => {
        const col = _edgeColor(rec);
        const inRoute = _routeEdgeSet ? _routeEdgeSet.has(rec.from + "→" + rec.to) : null;
        const onHover = _hoverPath && (rec.from === _hoverPath || rec.to === _hoverPath);
        let op = 0.22 + rec.strength * 0.4;
        if (inRoute === true) op = 0.95; else if (_routeEdgeSet) op = 0.06;
        else if (onHover) op = 0.9; else if (_hoverPath) op = 0.08;
        rec.el.setAttribute("stroke", col); rec.el.setAttribute("opacity", op.toFixed(2));
      });
      Object.keys(nodes).forEach(p => {
        const nd = nodes[p], e = nd.e;
        nd.core.setAttribute("fill", _nodeColor(e));
        nd.ring.setAttribute("stroke", _nodeColor(e)); nd.ring.setAttribute("stroke-width", (nd.hop === 0 ? 2.4 : 1.8).toFixed(1));
        nd.halo.setAttribute("fill", _nodeColor(e));
        const dim = (relNodes && !relNodes.has(p)) ? 0.26 : 1;
        nd.el.setAttribute("opacity", dim.toFixed(2));
        nd.el.classList.toggle("gx-rel", !!(relNodes && relNodes.has(p)));   // reveal its label
        // stress halo (health mode): connected to a high-error node
        let stress = 0;
        if (viewMode === "health") {
          (adj[p] || new Set()).forEach(v => { const ne = _byPath()[v]; if (ne && ne.err_rate >= 0.08) stress = Math.max(stress, ne.err_rate); });
          if (e.err_rate >= 0.08) stress = Math.max(stress, e.err_rate);
        }
        nd.stress.setAttribute("stroke", stress ? _rgb(EMBER) : "none");
        nd.stress.setAttribute("opacity", stress ? (0.2 + stress).toFixed(2) : "0");
        nd.stress.classList.toggle("gx-stress-on", !!stress);
      });
    }
    function _emphasizeNode(p) { _hoverPath = p; _restyle(); }
    function _emphasizeEdge(rec, on) { rec.el.setAttribute("opacity", on ? "0.95" : (0.22 + rec.strength * 0.4).toFixed(2)); rec.el.setAttribute("stroke-width", on ? (rec.baseW + 1).toFixed(2) : rec.baseW.toFixed(2)); }

    // ── particles ──
    const parts = [];     // {rec, t0, dur, el}
    function _spawn(rec, fast) {
      if (parts.length > 90) return;
      const c = document.createElementNS(SVGNS, "circle"); c.setAttribute("r", (1.6 + rec.strength * 1.6).toFixed(1)); c.setAttribute("class", "gx-part"); gParts.appendChild(c);
      parts.push({ rec, t0: performance.now(), dur: fast ? 620 : (900 + (1 - rec.strength) * 900), el: c });
    }
    function pulse(path) { elist.forEach(rec => { if (rec.from === path || rec.to === path) _spawn(rec, true); }); }
    function pulseEdge(from, to) { const rec = elist.find(r => r.from === from && r.to === to); if (rec) _spawn(rec, true); }

    // ── replay ──
    let _replay = null;    // {steps, i, t0, speed, dur, onStep, onDone, dims}
    function replay(steps, ro) {
      ro = ro || {}; if (!steps || steps.length < 2) return;
      stopReplay();
      // compress real dt: sqrt-scale + clamp so an 85s pause doesn't stall
      const cdt = steps.map((s, i) => i === 0 ? 300 : Math.max(280, Math.min(1600, Math.sqrt(s.dt_ms || 0) * 24)));
      _replay = { steps, cdt, i: 0, speed: ro.speed || 1, t0: performance.now(), onStep: ro.onStep, onDone: ro.onDone, total: cdt.reduce((a, b) => a + b, 0) };
      _routeSet = new Set(steps.map(s => s.path));
      _routeEdgeSet = new Set(); for (let k = 0; k < steps.length - 1; k++) _routeEdgeSet.add(steps[k].path + "→" + steps[k + 1].path);
      _restyle();
    }
    function stopReplay() { _replay = null; }
    function setReplayPos(frac) { if (_replay) { _replay.scrub = Math.max(0, Math.min(1, frac)); } }
    function setReplaySpeed(s) { if (_replay) _replay.speed = s; }

    // one-shot directional "explain" trace (hover a route): fire a particle
    // down each hop in order, once — shows the route's direction without the
    // full replay UI.
    const _traceTimers = [];
    function traceRoute(steps) {
      _traceTimers.forEach(clearTimeout); _traceTimers.length = 0;
      if (!steps || steps.length < 2) return;
      let acc = 0;
      for (let k = 0; k < steps.length - 1; k++) {
        const a = steps[k].path, b = steps[k + 1].path;
        const rec = elist.find(x => x.from === a && x.to === b);
        if (rec) { _traceTimers.push(setTimeout(() => _spawn(rec, true), acc)); acc += 240; }
      }
    }
    function highlightRoute(seq) {
      if (!seq || !seq.length) { clearHighlight(); return; }
      _routeSet = new Set(seq); _routeEdgeSet = new Set();
      for (let k = 0; k < seq.length - 1; k++) _routeEdgeSet.add(seq[k] + "→" + seq[k + 1]);
      _restyle();
    }
    function clearHighlight() { _routeSet = null; _routeEdgeSet = null; _restyle(); }

    // ── pan / zoom (full only) ──
    if (enablePan) {
      svg.addEventListener("wheel", (e) => { e.preventDefault(); tZoom = Math.max(0.6, Math.min(3, tZoom * (1 - Math.sign(e.deltaY) * 0.12))); }, { passive: false });
      let dragging = false, lx = 0, ly = 0;
      svg.addEventListener("pointerdown", (e) => { if (e.target.closest(".gx-node") || e.target.closest(".gx-edge-hit")) return; dragging = true; lx = e.clientX; ly = e.clientY; svg.setPointerCapture(e.pointerId); svg.style.cursor = "grabbing"; });
      svg.addEventListener("pointermove", (e) => { if (!dragging) return; tPanX += e.clientX - lx; tPanY += e.clientY - ly; lx = e.clientX; ly = e.clientY; });
      svg.addEventListener("pointerup", (e) => { dragging = false; svg.style.cursor = "grab"; });
    }

    // ── animation loop ──
    let raf = null, disposed = false;
    function frame() {
      if (disposed) return;
      if (!host.isConnected) { dispose(); return; }
      if (W < 40 || H < 40) { raf = requestAnimationFrame(frame); return; }   // never position before layout
      try { _frameBody(); } catch (_) { /* one bad frame must never kill the loop */ }
      raf = requestAnimationFrame(frame);
    }
    function _frameBody() {
      const now = performance.now(), t = now / 1000;
      // drift + place nodes (stable base positions + gentle celestial drift)
      Object.keys(nodes).forEach(p => {
        const nd = nodes[p];
        const dx = nd.drift ? Math.sin(t * 0.32 + nd.phase) * 2.2 : 0;
        const dy = nd.drift ? Math.cos(t * 0.27 + nd.phase * 1.3) * 2.0 : 0;
        nd.x = nd.bx + dx; nd.y = nd.by + dy;
        nd.el.setAttribute("transform", `translate(${nd.x.toFixed(1)},${nd.y.toFixed(1)})`);
      });
      // edges follow drifted centres (organic bow)
      elist.forEach(rec => {
        const a = nodes[rec.from], b = nodes[rec.to]; if (!a || !b) return;
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const nx = -(b.y - a.y), ny = (b.x - a.x), len = Math.hypot(nx, ny) || 1;
        const bow = Math.min(34, Math.hypot(b.x - a.x, b.y - a.y) * 0.16);
        const ctrlX = mx + nx / len * bow, ctrlY = my + ny / len * bow;
        const d = `M ${a.x.toFixed(1)} ${a.y.toFixed(1)} Q ${ctrlX.toFixed(1)} ${ctrlY.toFixed(1)} ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
        rec.el.setAttribute("d", d); rec.hit.setAttribute("d", d);
        rec.ctrl = [ctrlX, ctrlY];
        // idle trickle (skip during route highlight / replay of other edges)
        const muted = _routeEdgeSet && !_routeEdgeSet.has(rec.from + "→" + rec.to);
        if (!muted && viewMode !== "latency" && now - rec.emit > rec.gap) { rec.emit = now; _spawn(rec, false); }
      });
      // replay stepping
      if (_replay) {
        const r = _replay; const elapsed = (now - r.t0) * r.speed;
        let acc = 0, idx = 0;
        for (let k = 0; k < r.cdt.length; k++) { if (elapsed >= acc) idx = k; acc += r.cdt[k]; }
        if (r.scrub != null) idx = Math.floor(r.scrub * (r.steps.length - 1));
        if (idx !== r.i) {
          r.i = idx;
          if (idx > 0) { const rec = elist.find(x => x.from === r.steps[idx - 1].path && x.to === r.steps[idx].path); if (rec) _spawn(rec, true); }
          if (r.onStep) r.onStep(idx, r.steps[idx]);
        }
        if (r.scrub == null && elapsed > r.total + 400) { if (r.onDone) r.onDone(); }
      }
      // advance particles
      for (let i = parts.length - 1; i >= 0; i--) {
        const pt = parts[i], age = (now - pt.t0) / pt.dur, rec = pt.rec, a = nodes[rec.from], b = nodes[rec.to];
        if (age >= 1 || !a || !b) { if (pt.el.parentNode) pt.el.parentNode.removeChild(pt.el); parts.splice(i, 1); continue; }
        const ctrl = rec.ctrl || [(a.x + b.x) / 2, (a.y + b.y) / 2], u = age, iu = 1 - u;
        const x = iu * iu * a.x + 2 * iu * u * ctrl[0] + u * u * b.x;
        const y = iu * iu * a.y + 2 * iu * u * ctrl[1] + u * u * b.y;
        pt.el.setAttribute("cx", x.toFixed(1)); pt.el.setAttribute("cy", y.toFixed(1));
        pt.el.setAttribute("opacity", (Math.sin(age * Math.PI)).toFixed(2));
      }
      // ease zoom/pan
      if (enablePan) { zoom += (tZoom - zoom) * 0.15; panX += (tPanX - panX) * 0.15; panY += (tPanY - panY) * 0.15; view.setAttribute("transform", `translate(${panX.toFixed(1)},${panY.toFixed(1)}) scale(${zoom.toFixed(3)})`); }
    }

    function resize() { _computeLayout(); }
    function setData(d) { if (d.endpoints) endpoints = d.endpoints; if (d.edges) edges = d.edges; if (d.hub !== undefined) hub = d.hub; _computeLayout(); }
    function setFocus(p) { if (mode0 === "compact" && p && p !== focus) { focus = p; _computeLayout(); } else { focus = p; } }
    function setMode(m) { viewMode = m; _restyle(); }
    function dispose() { disposed = true; if (raf) cancelAnimationFrame(raf); if (_ro) { try { _ro.disconnect(); } catch (_) {} } try { host.removeChild(svg); } catch (_) {} }

    // ResizeObserver makes layout robust to when the host gets its box
    // (modal open, tab switch, window resize) without a polling loop.
    let _ro = null, _roRaf = 0;
    if (window.ResizeObserver) {
      _ro = new ResizeObserver(() => { if (_roRaf) return; _roRaf = requestAnimationFrame(() => { _roRaf = 0; if (!disposed) _computeLayout(); }); });
      try { _ro.observe(host); } catch (_) {}
    }
    _computeLayout();
    raf = requestAnimationFrame(frame);
    return { setData, setFocus, setMode, restyle: _restyle, highlightRoute, clearHighlight, traceRoute, replay, stopReplay, setReplayPos, setReplaySpeed, pulse, pulseEdge, resize, dispose,
             get focus() { return focus; }, get mode() { return viewMode; } };
  }

  window.APIN.galaxy = { create };
})();
