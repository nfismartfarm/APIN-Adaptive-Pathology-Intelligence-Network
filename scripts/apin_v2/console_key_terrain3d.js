// 9.N.T25 · Living Endpoint Terrain — WebGL 3D world (Three.js r149, UMD global THREE).
//
// A deep warm-ink topographic world for one API key's endpoints. The ground
// plan is the squarified treemap (area ∝ volume — data-accurate); each cell is
// raised into a smooth rounded dome (height ∝ volume) so the surface reads as
// rolling, contoured terrain. Colour = green→gold→ember by error rate. Lit with
// a warm key light, height-band contour iso-lines, valley shading, warm grain.
//
// Real-time: live events only rewrite the height/colour buffers + uniforms and
// lerp toward them each frame — geometry is never rebuilt, so the world grows
// like video (no stop-motion). Orbit/zoom, raycast hover + click, and HTML
// labels projected each frame keep it interactive while it morphs.
//
// Exposes APIN.terrain3d.create(canvas, opts) → instance. Falls back to the 2D
// terrain (handled by the caller) when WebGL is unavailable.
(function () {
  "use strict";
  if (!window.APIN) window.APIN = {};

  // warm luminous ramp (jade → gold → ember) — vibrant but not neon
  const JADE = [0.16, 0.62, 0.42], GOLD = [0.82, 0.62, 0.20], EMBER = [0.78, 0.31, 0.20];
  function _lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function _ramp(err) { const t = Math.min(1, (err || 0) / 0.2); return t <= 0.5 ? _lerp3(JADE, GOLD, t / 0.5) : _lerp3(GOLD, EMBER, (t - 0.5) / 0.5); }

  const VERT = `
    attribute vec3 aColor;
    uniform float uTime;
    varying vec3 vColor; varying float vH; varying vec3 vNormalW; varying vec3 vWorld;
    void main(){
      vColor = aColor;
      // subtle always-on "breathing" — hills gently rise/settle (region motion)
      float breath = 1.0 + 0.025 * sin(uTime * 1.1 + position.x * 1.7 + position.z * 1.3) * smoothstep(0.04, 0.3, position.y);
      vec3 p = vec3(position.x, position.y * breath, position.z);
      vH = p.y;
      vNormalW = normalize(mat3(modelMatrix) * normal);
      vec4 wp = modelMatrix * vec4(p, 1.0);
      vWorld = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }`;
  const FRAG = `
    precision highp float;
    varying vec3 vColor; varying float vH; varying vec3 vNormalW; varying vec3 vWorld;
    uniform vec3 uLightDir; uniform float uMaxH; uniform float uContourFreq;
    uniform vec3 uContourCol; uniform float uTime; uniform vec3 uHover; uniform vec3 uPulse;
    uniform float uMode; uniform vec3 uCamPos; uniform float uAlpha;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }
    void main(){
      vec3 N = normalize(vNormalW);
      vec3 L = normalize(uLightDir);
      vec3 V = normalize(uCamPos - vWorld);
      float hn = clamp(vH / max(uMaxH, 0.001), 0.0, 1.0);
      float diff = clamp(dot(N, L), 0.0, 1.0);
      // soft specular sheen + warm fresnel rim — gives form / "rendered" feel
      vec3 Hh = normalize(L + V);
      float spec = pow(clamp(dot(N, Hh), 0.0, 1.0), 26.0) * 0.30;
      float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0) * 0.40;
      float ao = mix(0.62, 1.0, hn);                          // valleys occluded/darker
      vec3 base = vColor * mix(0.62, 1.16, hn);
      vec3 col = base * (0.30 * ao + diff * 0.90);
      col += spec * vec3(1.0, 0.95, 0.82);                    // sheen highlight
      col += fres * vColor * 1.15;                            // rim light
      // fine glowing contour iso-lines (thin, not chunky bands)
      float ph = vH * uContourFreq;
      float t = abs(fract(ph) - 0.5);
      float lineCore = smoothstep(0.5, 0.46, t);              // narrow band near integer
      float shimmer = (uMode < 0.5) ? (0.5 + 0.5 * sin(ph * 6.2831 - uTime * 1.4)) : 1.0;
      col += uContourCol * lineCore * (0.18 + 0.42 * hn) * shimmer;
      // micro surface variation so it isn't flat-matte
      col *= 1.0 + (hash(floor(vWorld.xz * 90.0)) - 0.5) * 0.05;
      // HEALTH mode (uMode 2) — ember glows + pulses where requests fail
      if (uMode > 1.5 && uMode < 2.5) {
        float heat = clamp((vColor.r - vColor.g) * 2.2, 0.0, 1.0);
        col += vec3(0.95, 0.32, 0.16) * heat * (0.28 + 0.34 * (0.5 + 0.5 * sin(uTime * 3.0)));
        col *= mix(0.78, 1.0, heat);                          // damp healthy so errors pop
      }
      // CORE / X-RAY mode (uMode 3) — translucent skin + bold bright strata shells
      if (uMode > 2.5) {
        col += uContourCol * lineCore * 0.9 * (0.4 + 0.6 * hn);
        col += vColor * 0.25;                                 // glow the bands by health
      }
      // hover glow
      col += vColor * uHover.z * smoothstep(0.6, 0.0, distance(vWorld.xz, uHover.xy)) * 0.7;
      // live pulse ring (uPulse = x, z, age 0..1)
      if (uPulse.z > 0.0) {
        float ring = smoothstep(0.06, 0.0, abs(distance(vWorld.xz, uPulse.xy) - uPulse.z * 0.9));
        col += vec3(0.95, 0.82, 0.5) * ring * (1.0 - uPulse.z) * 0.9;
      }
      col += (hash(gl_FragCoord.xy + uTime) - 0.5) * 0.016;   // warm grain
      gl_FragColor = vec4(col, uAlpha);
    }`;

  function create(canvas, opts) {
    opts = opts || {};
    let layout = opts.layout, W = opts.W || 1000, H = opts.H || 600;
    const labelHost = opts.labelHost || null;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x16110c);                 // deep warm ink
    scene.fog = new THREE.FogExp2(0x16110c, 0.12);
    const aspect = (canvas.clientWidth || 800) / (canvas.clientHeight || 500);
    const camera = new THREE.PerspectiveCamera(42, aspect, 0.1, 100);
    // plane sized to the treemap aspect, centred at origin, Y up
    const A = (W >= H) ? 3.2 : 3.2 * (W / H), B = (W >= H) ? 3.2 * (H / W) : 3.2;
    const GX = 132, GZ = 96, HMAX = 0.62;   // ~12.7k verts (was 16.5k) — smoother on weak GPUs, contours are shader-based so detail is unaffected
    const geo = new THREE.PlaneGeometry(A * 2, B * 2, GX - 1, GZ - 1);
    geo.rotateX(-Math.PI / 2);                                    // lie flat on XZ
    const nVerts = geo.attributes.position.count;
    const aColor = new Float32Array(nVerts * 3);
    geo.setAttribute("aColor", new THREE.BufferAttribute(aColor, 3));
    const curH = new Float32Array(nVerts), tgtH = new Float32Array(nVerts);
    const curC = new Float32Array(nVerts * 3), tgtC = new Float32Array(nVerts * 3);
    const uniforms = {
      uLightDir: { value: new THREE.Vector3(0.5, 1.0, 0.35) },
      uMaxH: { value: HMAX }, uContourFreq: { value: 9.0 },
      uContourCol: { value: new THREE.Color(0xf0e0b8) },
      uTime: { value: 0 }, uHover: { value: new THREE.Vector3(99, 99, 0) },
      uPulse: { value: new THREE.Vector3(0, 0, 0) },
      uMode: { value: 0 }, uCamPos: { value: new THREE.Vector3() },
      uAlpha: { value: 1.0 },
    };
    const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);

    // map a pixel-space point (px,py in [0,W]×[0,H]) → terrain world xz
    const toWorld = (px, py) => [(px / W * 2 - 1) * A, (py / H * 2 - 1) * B];
    const fromWorld = (x, z) => [(x / A + 1) / 2 * W, (z / B + 1) / 2 * H];

    // evaluate the heightfield + colour from the layout into tgtH / tgtC.
    // ampFn(cellIndex) → 0..1+ multiplier (used by the intro drop-in animation).
    function evalLayout(lay, ampFn) {
      const cells = lay.cells, maxN = Math.max(1, ...cells.map(c => c.e.n));
      const pos = geo.attributes.position;
      const baseAmp = new Float32Array(cells.length), af = new Float32Array(cells.length);
      for (let k = 0; k < cells.length; k++) { baseAmp[k] = Math.pow(cells[k].e.n / maxN, 0.6) * HMAX; af[k] = ampFn ? ampFn(k) : 1; }
      for (let i = 0; i < nVerts; i++) {
        const wx = pos.getX(i), wz = pos.getZ(i);
        const px = (wx / A + 1) / 2 * W, py = (wz / B + 1) / 2 * H;
        let h = 0, cr = 0.16, cg = 0.16, cb = 0.16, best = 0;
        for (let k = 0; k < cells.length; k++) {
          const c = cells[k], cx = c.x + c.w / 2, cy = c.y + c.h / 2;
          const nx = (px - cx) / (c.w / 2 + 0.001), ny = (py - cy) / (c.h / 2 + 0.001);
          const r = Math.pow(Math.pow(Math.abs(nx), 2.4) + Math.pow(Math.abs(ny), 2.4), 1 / 2.4);
          const amp = baseAmp[k] * af[k];
          const dome = amp * (1 - _smooth(0.55, 1.12, r));     // rounded plateau, soft edge
          if (dome > 0) {
            h += dome;
            const w = (1 - _smooth(0.55, 1.12, r));             // dominance weight (geometry, not amp)
            if (w > best) { best = w; const rc = _ramp(c.e.err_rate); cr = rc[0]; cg = rc[1]; cb = rc[2]; }
          }
        }
        tgtH[i] = h; tgtC[i * 3] = cr; tgtC[i * 3 + 1] = cg; tgtC[i * 3 + 2] = cb;
      }
    }
    function _smooth(e0, e1, x) { const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); }

    let tweening = 0;
    function applyImmediate() { for (let i = 0; i < nVerts; i++) { curH[i] = tgtH[i]; } curC.set(tgtC); commit(); }
    function commit() {
      const pos = geo.attributes.position;
      for (let i = 0; i < nVerts; i++) pos.setY(i, curH[i]);
      pos.needsUpdate = true;
      aColor.set(curC); geo.attributes.aColor.needsUpdate = true;
      geo.computeVertexNormals();
      let mx = 0; for (let i = 0; i < nVerts; i++) if (curH[i] > mx) mx = curH[i];
      uniforms.uMaxH.value = Math.max(0.2, mx);
    }
    // ── intro: hills drop in from the sky, staggered (biggest first), bounce
    //    on landing + a warm dust ring on impact ──
    const dustGroup = new THREE.Group(); scene.add(dustGroup); const dusts = [];
    function _easeOutBack(t) { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); }
    function _spawnDust(cell) {
      const [wx, wz] = toWorld(cell.x + cell.w / 2, cell.y + cell.h / 2);
      const g = new THREE.RingGeometry(0.03, 0.08, 32); g.rotateX(-Math.PI / 2);
      const ring = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xdcc08a, transparent: true, opacity: 0.85, depthWrite: false }));
      ring.position.set(wx, 0.015, wz); dustGroup.add(ring); dusts.push({ ring, t0: performance.now() });
    }
    evalLayout(layout);                       // tgtH = full heights, tgtC = colours
    for (let i = 0; i < nVerts; i++) curH[i] = 0;
    curC.set(tgtC); commit();                 // flat ground, colours ready
    const _introOrder = layout.cells.map((c, k) => k).sort((a, b) => layout.cells[b].e.n - layout.cells[a].e.n);
    const introDelay = new Array(layout.cells.length).fill(0);
    _introOrder.forEach((k, rank) => { introDelay[k] = rank * 150; });
    let intro = { t0: performance.now(), dur: 640, last: introDelay.length ? Math.max(...introDelay) : 0, landed: new Array(layout.cells.length).fill(false) };

    // ── labels (HTML, projected each frame) ──
    const labels = [];
    function buildLabels() {
      if (!labelHost) return;
      labelHost.innerHTML = ""; labels.length = 0;
      layout.cells.slice().sort((a, b) => b.e.n - a.e.n).forEach((c, idx) => {
        const el = document.createElement("div");
        el.className = "ept3-label" + (idx < 9 ? " ept3-label-major" : "");
        el.innerHTML = `<b>${_esc(_tail(c.e.path))}</b><span>${_kc(c.e.n)}${c.e.err_pct ? " · " + c.e.err_pct + "%" : ""}</span>`;
        labelHost.appendChild(el);
        const [wx, wz] = toWorld(c.x + c.w / 2, c.y + c.h / 2);
        labels.push({ el, c, wx, wz, major: idx < 9, rank: idx, vi: _nearestVert(c.x + c.w / 2, c.y + c.h / 2) });
      });
    }
    buildLabels();

    // ── affinity flow curves ──
    let _mode = "terrain";   // declared before buildFlow() reads it (avoid TDZ)
    let flowGroup = new THREE.Group(); scene.add(flowGroup);
    function buildFlow() {
      flowGroup.clear();
      const links = (opts.affinity && opts.affinity.links) || [];
      const byPath = {}; layout.cells.forEach(c => byPath[c.e.path] = c);
      links.slice(0, 10).forEach(l => {
        const a = byPath[l.from], b = byPath[l.to]; if (!a || !b) return;
        const [ax, az] = toWorld(a.x + a.w / 2, a.y + a.h / 2), [bx, bz] = toWorld(b.x + b.w / 2, b.y + b.h / 2);
        const ah = _peakH(a), bh = _peakH(b);
        const mid = new THREE.Vector3((ax + bx) / 2, Math.max(ah, bh) + 0.5, (az + bz) / 2);
        const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(ax, ah + 0.05, az), mid, new THREE.Vector3(bx, bh + 0.05, bz));
        const pts = curve.getPoints(24);
        const g = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xe8d6ad, transparent: true, opacity: 0.18 }));
        line.userData.curve = curve; line.userData.count = l.count;
        flowGroup.add(line);
        // travelling spark
        const spark = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 8), new THREE.MeshBasicMaterial({ color: 0x8fe0b4 }));
        spark.userData.curve = curve; spark.userData.off = Math.random();
        flowGroup.add(spark);
      });
      flowGroup.visible = (_mode === "flow");
    }
    function _peakH(c) { const i = _nearestVert(c.x + c.w / 2, c.y + c.h / 2); return curH[i] || 0.2; }
    function _nearestVert(px, py) {
      const [wx, wz] = toWorld(px, py); const pos = geo.attributes.position; let bi = 0, bd = 1e9;
      for (let i = 0; i < nVerts; i++) { const dx = pos.getX(i) - wx, dz = pos.getZ(i) - wz, d = dx * dx + dz * dz; if (d < bd) { bd = d; bi = i; } }
      return bi;
    }
    buildFlow();

    // ── camera controls (hand-rolled orbit / zoom / pan) ──
    const ctr = new THREE.Vector3(0, 0.05, 0), tCtr = new THREE.Vector3(0, 0.05, 0);
    const PANX = A * 0.55, PANZ = B * 0.55;          // pan stays within the terrain
    let theta = -0.5, phi = 0.92, R = Math.max(A, B) * 2.0;
    let tTheta = theta, tPhi = phi, tR = R, dragging = false, panning = false, lx = 0, ly = 0, moved = false;
    function placeCam() {
      camera.position.set(ctr.x + R * Math.sin(phi) * Math.cos(theta), ctr.y + R * Math.cos(phi), ctr.z + R * Math.sin(phi) * Math.sin(theta));
      camera.lookAt(ctr);
    }
    placeCam();
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => { dragging = true; panning = (e.shiftKey || e.button === 2 || e.button === 1); moved = false; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointermove", (e) => {
      if (dragging) {
        const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        if (_coreSel) {
          // inside the cutaway, drag spins the core wheel (camera stays front-on)
          _coreDragRot += dx * 0.01;
        } else if (panning) {
          // pan along the camera's ground plane, clamped to a bounded region
          const sp = R * 0.0016;
          const rx = -Math.sin(theta), rz = Math.cos(theta);   // screen-right in world XZ
          const fx = Math.cos(theta), fz = Math.sin(theta);    // screen-forward in world XZ
          tCtr.x = Math.max(-PANX, Math.min(PANX, tCtr.x - dx * sp * rx + dy * sp * fx));
          tCtr.z = Math.max(-PANZ, Math.min(PANZ, tCtr.z - dx * sp * rz + dy * sp * fz));
        } else { tTheta += dx * 0.006; tPhi = Math.max(0.18, Math.min(1.32, tPhi - dy * 0.005)); }
      } else { _raycastHover(e); }
    });
    const endDrag = () => { dragging = false; panning = false; };
    canvas.addEventListener("pointerup", (e) => { const wasPan = panning; endDrag(); if (!moved && !wasPan) _raycastClick(e); });
    canvas.addEventListener("pointerleave", () => { dragging = false; panning = false; uniforms.uHover.value.set(99, 99, 0); if (opts.onLeave) opts.onLeave(); });
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); tR = Math.max(Math.max(A, B) * 0.8, Math.min(Math.max(A, B) * 4, tR * (1 + Math.sign(e.deltaY) * 0.1))); }, { passive: false });

    // ── raycasting hover / click ──
    const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    function _pick(e) {
      const r = canvas.getBoundingClientRect();
      ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1; ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObject(mesh, false)[0]; if (!hit) return null;
      const [px, py] = fromWorld(hit.point.x, hit.point.z);
      let best = null, bestD = 1e9;
      layout.cells.forEach(c => { if (px >= c.x && px <= c.x + c.w && py >= c.y && py <= c.y + c.h) { const d = Math.abs(px - (c.x + c.w / 2)) + Math.abs(py - (c.y + c.h / 2)); if (d < bestD) { bestD = d; best = c; } } });
      return best ? { cell: best, world: hit.point } : null;
    }
    let _hoverPath = null;
    function _raycastHover(e) {
      if (_coreSel) { canvas.style.cursor = "grab"; return; }   // no terrain picking in cutaway
      const r = _pick(e);
      if (r) { uniforms.uHover.value.set(r.world.x, r.world.z, 1); _hoverPath = r.cell.e.path; if (opts.onHover) opts.onHover(r.cell.e.path, e.clientX, e.clientY); canvas.style.cursor = "pointer"; }
      else { uniforms.uHover.value.set(99, 99, 0); _hoverPath = null; canvas.style.cursor = "grab"; if (opts.onLeave) opts.onLeave(); }
    }
    function _raycastClick(e) { if (_coreSel) return; const r = _pick(e); if (r && opts.onClick) opts.onClick(r.cell.e.path); }

    // external focus (from the page focus bus): glow a region without entering it
    function focus(path) {
      if (_coreSel) return;
      const c = (layout.cells || []).find(x => x.e.path === path);
      if (!c) { uniforms.uHover.value.set(99, 99, 0); return; }
      const [wx, wz] = toWorld(c.x + c.w / 2, c.y + c.h / 2);
      uniforms.uHover.value.set(wx, wz, 1); _hoverPath = path;
    }

    // double-click a hill → contextual zoom toward it (toggle back on repeat)
    let _zoomedPath = null;
    canvas.addEventListener("dblclick", (e) => {
      if (_coreSel) return;
      const r = _pick(e); if (!r) { return; }
      const c = r.cell;
      if (_zoomedPath === c.e.path) {              // toggle back out
        _zoomedPath = null; tCtr.set(0, 0.05, 0); tR = Math.max(A, B) * 2.0; tPhi = 0.92;
      } else {
        _zoomedPath = c.e.path;
        const [wx, wz] = toWorld(c.x + c.w / 2, c.y + c.h / 2);
        tCtr.set(Math.max(-PANX, Math.min(PANX, wx)), 0.05, Math.max(-PANZ, Math.min(PANZ, wz)));
        tR = Math.max(A, B) * 1.05; tPhi = 0.78;   // closer + lower angle = "into" the region
      }
    });

    // ── live pulse ──
    let _pulse = null;   // { x, z, t0 }
    function pulse(path) {
      if (_coreSel && _coreSel.path === path) { _corePulseT = performance.now(); return; }  // flash the core strata
      const c = (layout.cells || []).find(x => x.e.path === path); if (!c) return;
      const [wx, wz] = toWorld(c.x + c.w / 2, c.y + c.h / 2); _pulse = { x: wx, z: wz, t0: performance.now() };
    }

    // ── core cutaway (per-hill "earth core" view) ──
    let _coreSel = null, coreGroup = null, _strata = "traffic", _coreDragRot = 0, _corePulseT = 0, _savedCam = null;
    const _T_SLO = { quick_inference: 8000, heavy_inference: 15000, metadata: 300, default: 1000 };
    function _scoreCol(s) { return s >= 0.8 ? [0.16, 0.62, 0.42] : s >= 0.5 ? [0.82, 0.62, 0.20] : [0.78, 0.31, 0.20]; }
    function _coreBands(e, strata) {
      if (strata === "status") {
        const t = Math.max(1, e.n);
        return [{ frac: (e.n2 || 0) / t, col: [0.16, 0.62, 0.42] },
                { frac: (e.n4 || 0) / t, col: [0.82, 0.62, 0.20] },
                { frac: (e.n5 || 0) / t, col: [0.78, 0.31, 0.20], core: true }];
      }
      if (strata === "pillars") {
        const T = _T_SLO[e.cls] || 1000;
        const rel = 1 - (e.err_rate || 0);
        const perf = 1 - Math.min(1, (e.p95 || 0) / (T * 4));
        const stab = 1 - Math.min(1, (e.p50 > 0 ? e.p95 / e.p50 - 1 : 0) / 7);
        return [{ frac: 0.34, col: _scoreCol(rel) }, { frac: 0.33, col: _scoreCol(perf) }, { frac: 0.33, col: _scoreCol(stab), core: true }];
      }
      // traffic → latency → error (default)
      const T = _T_SLO[e.cls] || 1000, latS = 1 - Math.min(1, (e.p95 || 0) / (T * 4));
      const errC = (e.err_rate > 0) ? _ramp(e.err_rate) : [0.16, 0.62, 0.42];
      return [{ frac: 0.34, col: [0.16, 0.62, 0.42] }, { frac: 0.33, col: _scoreCol(latS) }, { frac: 0.33, col: errC, core: true }];
    }
    function _buildCoreGroup(e, strata) {
      const g = new THREE.Group(), bands = _coreBands(e, strata), R = 1.25;
      const total = bands.reduce((s, b) => s + Math.max(0.07, b.frac), 0);
      let outer = R;
      bands.forEach(b => {
        const thick = (Math.max(0.07, b.frac) / total) * R, inner = Math.max(0, outer - thick);
        const c = new THREE.Color(b.col[0], b.col[1], b.col[2]);
        const front = new THREE.Mesh(new THREE.RingGeometry(inner, outer, 96, 1, 0, Math.PI), new THREE.MeshBasicMaterial({ color: c }));
        front.position.z = 0.07; g.add(front);
        const back = new THREE.Mesh(new THREE.RingGeometry(inner, outer, 96, 1, 0, Math.PI), new THREE.MeshBasicMaterial({ color: c.clone().multiplyScalar(0.4) }));
        back.position.z = -0.07; g.add(back);
        const wall = new THREE.Mesh(new THREE.CylinderGeometry(outer, outer, 0.14, 64, 1, true, 0, Math.PI), new THREE.MeshBasicMaterial({ color: c.clone().multiplyScalar(0.7), side: THREE.DoubleSide }));
        wall.rotation.x = Math.PI / 2; g.add(wall);     // curved shell wall = thickness
        if (b.core) { g.userData.core = front; g.userData.coreBack = back; }
        outer = inner;
      });
      g.position.y = R * 0.5;
      return g;
    }
    function _disposeGroup(grp) { grp.children.forEach(o => { o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); }); }
    function enterCore(e) {
      _coreSel = e; mat.visible = true;
      mesh.visible = false; flowGroup.visible = false; labels.forEach(L => { L.el.style.opacity = "0"; });
      if (coreGroup) { scene.remove(coreGroup); _disposeGroup(coreGroup); }
      coreGroup = _buildCoreGroup(e, _strata); scene.add(coreGroup);
      _coreDragRot = 0;
      _savedCam = { theta, phi, R, cx: tCtr.x, cz: tCtr.z };
      tCtr.set(0, coreGroup.position.y, 0); ctr.copy(tCtr);
      tTheta = Math.PI / 2; tPhi = 1.25; tR = 4.2;       // front-on framing
    }
    function exitCore() {
      _coreSel = null;
      if (coreGroup) { scene.remove(coreGroup); _disposeGroup(coreGroup); coreGroup = null; }
      mesh.visible = true; flowGroup.visible = (_mode === "flow");
      if (_savedCam) { tTheta = _savedCam.theta; tPhi = _savedCam.phi; tR = _savedCam.R; tCtr.set(_savedCam.cx, 0.05, _savedCam.cz); }
    }
    function setStrata(s) { _strata = s; if (_coreSel) { if (coreGroup) { scene.remove(coreGroup); _disposeGroup(coreGroup); } coreGroup = _buildCoreGroup(_coreSel, _strata); scene.add(coreGroup); } }

    // ── modes ──
    function setMode(m) {
      _mode = m; flowGroup.visible = (m === "flow");
      uniforms.uMode.value = m === "flow" ? 1 : m === "health" ? 2 : m === "core" ? 3 : 0;
      uniforms.uContourFreq.value = (m === "health") ? 7.0 : (m === "core") ? 6.0 : 9.0;
      uniforms.uAlpha.value = (m === "core") ? 0.6 : 1.0;     // x-ray translucency
      mat.transparent = (m === "core"); mat.depthWrite = (m !== "core"); mat.needsUpdate = true;
    }

    // ── reflow (smooth drift) ──
    function setLayout(lay, affinity) {
      if (intro || _coreSel) return;   // let the drop-in finish; never reflow while in the cutaway
      layout = lay; if (affinity) opts.affinity = affinity;
      evalLayout(lay); tweening = performance.now();
      // reposition labels IN PLACE (no DOM rebuild → no blink) unless the
      // endpoint set actually changed
      const byPath = {}; lay.cells.forEach(c => byPath[c.e.path] = c);
      const same = labels.length === lay.cells.length && labels.every(L => byPath[L.c.e.path]);
      if (same) labels.forEach(L => { const c = byPath[L.c.e.path]; L.c = c; const [wx, wz] = toWorld(c.x + c.w / 2, c.y + c.h / 2); L.wx = wx; L.wz = wz; L.vi = _nearestVert(c.x + c.w / 2, c.y + c.h / 2); });
      else buildLabels();
      buildFlow();
    }

    // ── animation loop ──
    let raf = null, disposed = false;
    function frame() {
      if (disposed) return;
      if (!canvas.isConnected) { dispose(); return; }   // modal closed → self-clean
      const now = performance.now(), t = now / 1000; uniforms.uTime.value = t;
      // camera ease (orbit + zoom + pan target)
      theta += (tTheta - theta) * 0.12; phi += (tPhi - phi) * 0.12; R += (tR - R) * 0.12;
      ctr.x += (tCtr.x - ctr.x) * 0.15; ctr.z += (tCtr.z - ctr.z) * 0.15; placeCam();
      uniforms.uCamPos.value.copy(camera.position);
      // CORE cutaway view — gentle sway + drag-rotate + pulsing error core
      if (_coreSel && coreGroup) {
        coreGroup.rotation.y = _coreDragRot + Math.sin(t * 0.5) * 0.3;
        const core = coreGroup.userData.core;
        if (core) { const pf = (_corePulseT && now - _corePulseT < 600) ? (1 - (now - _corePulseT) / 600) : 0; const s = 1 + 0.05 * Math.sin(t * 3.0) + pf * 0.3; core.scale.set(s, s, 1); }
      } else
      // intro drop-in (staggered, bounce, dust) — overrides reflow tween while active
      if (intro) {
        for (let k = 0; k < layout.cells.length; k++) { if (!intro.landed[k] && now - intro.t0 >= introDelay[k]) { intro.landed[k] = true; _spawnDust(layout.cells[k]); } }
        evalLayout(layout, (k) => { const lt = (now - intro.t0 - introDelay[k]) / intro.dur; return lt <= 0 ? 0 : (lt >= 1 ? 1 : _easeOutBack(lt)); });
        curH.set(tgtH); commit();
        if (now - intro.t0 > intro.last + intro.dur + 60) { evalLayout(layout); curH.set(tgtH); commit(); intro = null; }
      } else if (tweening) {
        const k = Math.min(1, (now - tweening) / 750);
        for (let i = 0; i < nVerts; i++) {
          curH[i] += (tgtH[i] - curH[i]) * 0.14;
          curC[i * 3] += (tgtC[i * 3] - curC[i * 3]) * 0.14;
          curC[i * 3 + 1] += (tgtC[i * 3 + 1] - curC[i * 3 + 1]) * 0.14;
          curC[i * 3 + 2] += (tgtC[i * 3 + 2] - curC[i * 3 + 2]) * 0.14;
        }
        commit();
        if (k >= 1) { curH.set(tgtH); curC.set(tgtC); commit(); tweening = 0; }
      }
      // pulse uniform
      if (_pulse) { const age = (now - _pulse.t0) / 800; if (age >= 1) _pulse = null; else uniforms.uPulse.value.set(_pulse.x, _pulse.z, age); }
      else uniforms.uPulse.value.set(0, 0, 0);
      // flow sparks
      if (flowGroup.visible) flowGroup.children.forEach(o => { if (o.userData.curve && o.geometry.type === "SphereGeometry") { o.userData.off = (o.userData.off + 0.004) % 1; const p = o.userData.curve.getPoint(o.userData.off); o.position.copy(p); } });
      // dust rings (intro landings) — expand + fade
      for (let i = dusts.length - 1; i >= 0; i--) {
        const d = dusts[i], age = (now - d.t0) / 620;
        if (age >= 1) { dustGroup.remove(d.ring); d.ring.geometry.dispose(); d.ring.material.dispose(); dusts.splice(i, 1); continue; }
        const s = 0.4 + age * 3.4; d.ring.scale.set(s, s, s); d.ring.material.opacity = 0.8 * (1 - age);
      }
      // labels — cached vertex index (no per-frame scan); biggest-first dedupe
      if (labelHost && labels.length && !_coreSel) {
        const r = canvas.getBoundingClientRect(), placed = [];
        const flat = Math.cos(phi);   // ~1 looking straight down, ~0 from the side
        labels.forEach(L => {
          // lift the anchor more when viewing top-down so close peaks separate on screen
          const v = new THREE.Vector3(L.wx, (curH[L.vi] || 0.2) + 0.14 + flat * 0.10, L.wz).project(camera);
          const onScreen = v.z < 1 && v.x > -1.08 && v.x < 1.08 && v.y > -1.08 && v.y < 1.12;
          const sx = (v.x * 0.5 + 0.5) * r.width, sy = (-v.y * 0.5 + 0.5) * r.height;
          let show = onScreen && (L.major || _hoverPath === L.c.e.path);
          // majors are ALWAYS shown (no overlap-hiding) so no key label disappears;
          // only the minor hover-revealed labels participate in collision culling
          if (show && !L.major && _hoverPath !== L.c.e.path) {
            for (let p = 0; p < placed.length; p++) { if (Math.abs(placed[p][0] - sx) < 72 && Math.abs(placed[p][1] - sy) < 18) { show = false; break; } }
          }
          if (show) placed.push([sx, sy]);
          L.el.style.opacity = show ? (L.major ? "1" : "0.96") : "0";
          L.el.style.zIndex = L.major ? "2" : "1";
          L.el.style.transform = `translate(-50%,-120%) translate(${sx.toFixed(1)}px,${sy.toFixed(1)}px)`;
        });
      }
      renderer.render(scene, camera);
      raf = requestAnimationFrame(frame);
    }
    function resize() {
      const w = canvas.clientWidth || 800, h = canvas.clientHeight || 500;
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    resize(); raf = requestAnimationFrame(frame);

    function dispose() {
      disposed = true; if (raf) cancelAnimationFrame(raf);
      try { geo.dispose(); mat.dispose(); flowGroup.children.forEach(o => { o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); }); renderer.dispose(); } catch (_) {}
      if (labelHost) labelHost.innerHTML = "";
    }
    return { setLayout, setMode, pulse, enterCore, exitCore, setStrata, focus, resize, dispose };
  }

  const _esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const _tail = (p) => { p = String(p || "?"); const s = p.split("/").filter(Boolean); return s.length ? "/" + s.slice(-2).join("/") : p; };
  const _kc = (n) => n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);

  window.APIN.terrain3d = { create };
})();
