/* 9.N.T32 · Living Agricultural Observatory — expanded Inference Field.
 *
 * A navigable miniature farm world (procedural three.js) instead of a chart.
 * Crop DISTRICTS (okra · broccoli · tomato) are built from /analytics/field;
 * every PLANT is a real scan (from /analytics/field/scans), so hovering reads
 * the diagnosis and clicking opens a Prediction Inspector with the real image.
 *
 * Performance (the "mix"): procedural geometry only (no GLB download), with
 * InstancedMesh for the bulk — tiles, plant tops, stems, fences, rocks each
 * render in ONE draw call regardless of count, and the layer toggles recolour
 * via per-instance colour buffers (no rebuilds). Lazy-loaded only when the
 * expand opens; three.js is already present from the compact donut.
 *
 * Layers: Crops · Disease · Confidence · Growth (instanceColor recolour).
 * Live: SSE scan events drop a seed → sprout on the right district.
 * Lazy module — registers window.APIN.analyticsObservatory.open(el, opts).
 */
(function () {
  'use strict';
  const O = {};
  let T = null;            // THREE
  let OBS = null;          // single live instance

  // paper-ink palette
  const PAPER = 0xefe7d4, INK = 0x1a1612, GREEN = 0x52b788, GREEN_D = 0x2d6a4f;
  const OCHRE = 0xb6822a, AMBER = 0xc98a2b, RED = 0xb3402f, SOIL = 0x6b5b43, SAGE = 0x6cbf94;
  const GRASS = 0x9bbf73, GRASS_D = 0x86a862, PATH = 0xcabf9f, WATER = 0x7fb0c4, MUTE = 0xb9b3a3;
  const ROSTER = [
    { key: 'okra', label: 'Okra', accent: '#52b788', leaf: 0x6cbf94, soil: 0x88a85f, land: 'windmill' },
    { key: 'brassica', label: 'Broccoli', accent: '#2f6a43', leaf: 0x4f8f5a, soil: 0x7e9a55, land: 'barn' },
    { key: 'tomato', label: 'Tomato', accent: '#c0563b', leaf: 0x6f9a52, soil: 0x8a8270, land: 'greenhouse' },
  ];
  const HEALTHY = { okra_healthy: 1, brassica_healthy: 1, tomato_healthy: 1 };
  const LAYERS = [['crops', 'Crops'], ['disease', 'Disease'], ['confidence', 'Confidence'], ['growth', 'Growth']];
  const RANGES = ['7d', '30d', '90d'];

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
  function titleCase(s) { return String(s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }
  function fmt(n) { n = +n || 0; return Math.abs(n) >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n)); }
  function api(p) { const f = window.APIN && window.APIN.analyticsApi; return f ? f(p) : Promise.reject(new Error('no api')); }
  function ctxRange() { try { return ((window.APIN.analyticsCtx && window.APIN.analyticsCtx()) || {}).range || '30d'; } catch (e) { return '30d'; } }
  function ipAgo(iso) {
    try { if (window.APIN.time && window.APIN.time.ago) return window.APIN.time.ago(iso); } catch (e) { }
    if (!iso) return ''; const ms = Date.now() - new Date(iso).getTime(); if (isNaN(ms)) return '';
    if (ms < 6e4) return Math.max(1, Math.round(ms / 1e3)) + 's ago'; if (ms < 36e5) return Math.round(ms / 6e4) + 'm ago';
    if (ms < 864e5) return Math.round(ms / 36e5) + 'h ago'; return Math.round(ms / 864e5) + 'd ago';
  }
  function localTime(iso) { try { return new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); } catch (e) { return iso || ''; } }

  function open(el, opts) {
    injectCSS();
    if (OBS) { try { OBS.destroy(); } catch (e) { } OBS = null; }
    T = window.THREE;
    if (!T) { el.innerHTML = '<div class="an-ph an-ph-empty">3D unavailable</div>'; return; }
    OBS = makeObservatory(el, opts || {});
  }

  // ── shell ──────────────────────────────────────────────────────────────
  function shell() {
    const layers = LAYERS.map((l, i) => '<button data-layer="' + l[0] + '"' + (i === 0 ? ' aria-pressed="true"' : '') + '>' + l[1] + '</button>').join('');
    const ranges = RANGES.map(r => '<button data-range="' + r + '"' + (r === '30d' ? ' aria-pressed="true"' : '') + '>' + r + '</button>').join('');
    return '<div class="an-obs">' +
      '<div class="an-obs-head"><h3>Living Agricultural Observatory</h3>' +
      '<div class="an-obs-tools"><div class="an-obs-seg" id="obs-layer">' + layers + '</div>' +
      '<div class="an-obs-seg" id="obs-range">' + ranges + '</div></div></div>' +
      '<div class="an-obs-world" id="obs-world"><div class="an-obs-ins" id="obs-ins" hidden></div>' +
      '<div class="an-obs-crumb" id="obs-crumb" hidden></div>' +
      '<div class="an-obs-insight" id="obs-insight"></div></div>' +
      '<div class="an-obs-panels">' +
      '<div class="an-obs-panel" id="obs-crop"><div class="an-obs-ph">Crop Intel</div><div class="an-obs-body" id="obs-crop-b"></div></div>' +
      '<div class="an-obs-panel" id="obs-disease"><div class="an-obs-ph">Disease Lab</div><div class="an-obs-body" id="obs-disease-b"></div></div>' +
      '<div class="an-obs-panel" id="obs-feed"><div class="an-obs-ph">Prediction Feed <span class="an-obs-live"><i></i>live</span></div><div class="an-obs-body" id="obs-feed-b"></div></div>' +
      '</div>' +
      '<div class="an-obs-timeline" id="obs-timeline"></div>' +
      '</div>';
  }

  function makeObservatory(el, opts) {
    el.innerHTML = shell();
    const node = el.querySelector('#obs-world');
    let scene, cam, renderer, ro, raf, destroyed = false;
    let tilesMesh, tileMap = [], plantTop, plantStem, plantMap = [], plantBase = [];
    let landmarks = [], clouds = [], fences = null;
    let districts = {};     // key -> {center, tiles, scans, data, count, dormant}
    let _data = null, _scans = [], _layer = 'crops', _range = (RANGES.indexOf(ctxRange()) >= 0 ? ctxRange() : '30d');
    let _focus = null, _diseaseFilter = null;
    // ── replay scrubber state ───────────────────────────────────────────
    // The farm starts blank; plants appear at their real timestamp as the
    // scrub head advances. Media controls drive _replay.t in [0,1].
    let _baseTop = [], _baseStem = [];      // captured target matrices per plant
    let _plantTime = [];                    // ms timestamp per plant instance
    let _appear = [], _revealed = [];       // per-plant pop progress + reveal flag
    let _tMin = 0, _tMax = 0;               // window time bounds (ms)
    let _replay = { t: 1, playing: false, speed: 1, durMs: 9000 };
    let _popActive = false, _scrubbing = false, _lastFrame = 0;
    let raycaster, mouse, _hoverTile = -1, _hoverPlant = -1;
    // orbit
    let orb = { theta: 0.5, phi: 0.82, radius: 9, target: null }, orbAnim = null, drag = null;

    // ── three setup ─────────────────────────────────────────────────────
    scene = new T.Scene();
    const w = node.clientWidth || 800, h = node.clientHeight || 480;
    cam = new T.PerspectiveCamera(42, w / h, 0.1, 200);
    orb.target = new T.Vector3(0, 0, 0); applyOrbit();
    renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(w, h); renderer.setClearColor(0x000000, 0);
    node.appendChild(renderer.domElement);
    scene.add(new T.HemisphereLight(0xfff6e6, 0x6b5b43, 1.0));
    const dir = new T.DirectionalLight(0xffffff, 0.5); dir.position.set(4, 8, 3); scene.add(dir);
    raycaster = new T.Raycaster(); mouse = new T.Vector2();
    const dom = renderer.domElement; dom.style.touchAction = 'none';
    dom.addEventListener('pointerdown', onDown); dom.addEventListener('pointermove', onPMove);
    dom.addEventListener('pointerup', onUp); dom.addEventListener('pointerleave', () => { hideTip(); if (!drag) node.style.cursor = 'grab'; });
    dom.addEventListener('wheel', onWheel, { passive: false });
    dom.addEventListener('contextmenu', e => e.preventDefault());
    node.style.cursor = 'grab';
    ro = new ResizeObserver(resize); ro.observe(node);

    // shared geometries / materials (reused → batching)
    const tileGeo = new T.BoxGeometry(0.96, 0.22, 0.96);
    const tileMat = new T.MeshToonMaterial({ vertexColors: false });
    const topGeo = new T.IcosahedronGeometry(0.17, 0);
    const stemGeo = new T.CylinderGeometry(0.02, 0.035, 0.2, 5);
    const topMat = new T.MeshToonMaterial();
    const stemMat = new T.MeshToonMaterial({ color: GREEN_D });

    // ── wire shell ──────────────────────────────────────────────────────
    el.querySelector('#obs-layer').addEventListener('click', e => {
      const b = e.target.closest('[data-layer]'); if (!b) return; _layer = b.getAttribute('data-layer');
      el.querySelectorAll('#obs-layer [data-layer]').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
      recolorPlants(); renderInsight(); renderDisease(); renderCrop();   // tiles follow the active layer
    });
    el.querySelector('#obs-range').addEventListener('click', e => {
      const b = e.target.closest('[data-range]'); if (!b) return; _range = b.getAttribute('data-range');
      el.querySelectorAll('#obs-range [data-range]').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
      reload();   // the bottom replay scrubber rebuilds with the new window in build()
    });

    // ── data ────────────────────────────────────────────────────────────
    function reload() {
      Promise.all([api('field?range=' + _range), api('field/scans?range=' + _range + '&limit=600')])
        .then(([fd, sc]) => { if (destroyed) return; _data = fd; _scans = (sc && sc.items) || []; build(); renderPanels(); })
        .catch(() => { });
    }
    reload();

    // ── build the farm ──────────────────────────────────────────────────
    function build() {
      // wipe previous world meshes
      [tilesMesh, plantTop, plantStem, fences].forEach(m => { if (m) { scene.remove(m); m.geometry && m.geometry.dispose && 0; } });
      landmarks.forEach(g => scene.remove(g)); landmarks = []; clouds.forEach(c => scene.remove(c)); clouds = [];
      tileMap = []; plantMap = []; plantBase = []; districts = {};
      const crops = (_data && _data.crops) || []; const byKey = {}; crops.forEach(c => byKey[c.crop] = c);
      // district centres in a gentle row
      const span = 6.2; const centres = { okra: [-span, 0], brassica: [0, span * 0.55], tomato: [span, 0] };
      const tiles = [];
      ROSTER.forEach(r => {
        const c = byKey[r.key] || { crop: r.key, count: 0, diseases: [], pct: 0 };
        const cnt = c.count || 0, dormant = cnt === 0;
        const cell = centres[r.key]; const cx = cell[0], cz = cell[1];
        const gridN = dormant ? 2 : Math.max(2, Math.min(6, Math.round(Math.sqrt(cnt) / 2) + 2));
        const tileC = [];
        for (let gx = 0; gx < gridN; gx++) for (let gz = 0; gz < gridN; gz++) {
          const px = cx + (gx - (gridN - 1) / 2) * 1.0, pz = cz + (gz - (gridN - 1) / 2) * 1.0;
          tileC.push([px, pz]);
          tiles.push({ x: px, z: pz, key: r.key, dormant: dormant });
        }
        districts[r.key] = { roster: r, center: [cx, cz], tiles: tileC, data: c, count: cnt, dormant: dormant, scans: [] };
      });
      // tiles instanced
      tilesMesh = new T.InstancedMesh(tileGeo, tileMat, tiles.length);
      const m4 = new T.Matrix4(), col = new T.Color();
      tiles.forEach((t, i) => {
        m4.makeTranslation(t.x, 0.11, t.z); tilesMesh.setMatrixAt(i, m4);
        col.setHex(t.dormant ? 0xb6ad97 : ((i % 2) ? GRASS : GRASS_D)); tilesMesh.setColorAt(i, col);
        tileMap[i] = t.key;
      });
      tilesMesh.instanceMatrix.needsUpdate = true; if (tilesMesh.instanceColor) tilesMesh.instanceColor.needsUpdate = true;
      tilesMesh.userData = { tiles: true }; scene.add(tilesMesh);
      // plants = scans
      const placed = [];
      _scans.forEach(s => {
        const d = districts[s.crop]; if (!d || d.dormant) return;
        const tc = d.tiles[(Math.random() * d.tiles.length) | 0];
        const px = tc[0] + (Math.random() - 0.5) * 0.7, pz = tc[1] + (Math.random() - 0.5) * 0.7;
        placed.push({ x: px, z: pz, scan: s, key: s.crop });
        d.scans.push(s);
      });
      const N = placed.length;
      plantTop = new T.InstancedMesh(topGeo, topMat, Math.max(1, N));
      plantStem = new T.InstancedMesh(stemGeo, stemMat, Math.max(1, N));
      const mt = new T.Matrix4(), ms = new T.Matrix4();
      placed.forEach((p, i) => {
        const hh = 0.2 + Math.random() * 0.08;
        ms.makeTranslation(p.x, 0.22 + 0.1, p.z); plantStem.setMatrixAt(i, ms);
        mt.makeTranslation(p.x, 0.22 + 0.2 + hh, p.z); plantTop.setMatrixAt(i, mt);
        plantMap[i] = p.scan; plantBase[i] = { x: p.x, z: p.z, y: 0.22 + 0.2 + hh };
      });
      if (N === 0) { plantTop.count = 0; plantStem.count = 0; }
      plantTop.instanceMatrix.needsUpdate = true; plantStem.instanceMatrix.needsUpdate = true;
      plantTop.userData = { plants: true }; scene.add(plantTop); scene.add(plantStem);
      recolorPlants();
      // landmarks + fences + pond + clouds
      ROSTER.forEach(r => { const d = districts[r.key]; const g = landmark(r.land, d.center[0], d.center[1], d.dormant, r); if (g) { scene.add(g); landmarks.push(g); } });
      addPond(0, -span * 0.55);
      addClouds();
      // replay: capture base matrices, then play the farm from blank → full
      setupReplay();
      // frame the whole farm
      orbitTo(0, 0, 0, 9, 0.5, 0.82);
      renderInsight(); renderCrumb(); renderReplay();
    }

    // ── replay subsystem ────────────────────────────────────────────────
    function setupReplay() {
      _baseTop = []; _baseStem = []; _plantTime = []; _appear = []; _revealed = [];
      const n = (plantTop && plantTop.count) || 0;
      const tmp = new T.Matrix4();
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < n; i++) {
        plantTop.getMatrixAt(i, tmp); _baseTop.push(tmp.clone());
        plantStem.getMatrixAt(i, tmp); _baseStem.push(tmp.clone());
        const s = plantMap[i]; const ms = new Date((s && (s.processed_at || s.captured_at)) || 0).getTime();
        const t = isNaN(ms) ? 0 : ms; _plantTime.push(t);
        if (t < mn) mn = t; if (t > mx) mx = t;
        _appear.push(0); _revealed.push(false);
      }
      _tMin = isFinite(mn) ? mn : 0; _tMax = isFinite(mx) ? mx : 0;
      if (_tMax <= _tMin) _tMax = _tMin + 1;        // guard zero-width window
      // Start blank and auto-play the growth once (the "seed → sprout" intro).
      _replay.t = 0; _replay.playing = n > 0; _lastFrame = performance.now();
      applyReplay(true);
    }
    function easeOut(x) { return 1 - (1 - x) * (1 - x); }
    function writePlantMatrices() {
      if (!plantTop || !plantTop.count) return;
      const pos = new T.Vector3(), q = new T.Quaternion(), sc = new T.Vector3();
      const mT = new T.Matrix4(), mS = new T.Matrix4();
      for (let i = 0; i < plantTop.count; i++) {
        const a = _revealed[i] ? easeOut(_appear[i]) : 0.0001;
        _baseTop[i].decompose(pos, q, sc); mT.compose(pos, q, new T.Vector3(a, a, a)); plantTop.setMatrixAt(i, mT);
        _baseStem[i].decompose(pos, q, sc); mS.compose(pos, q, new T.Vector3(1, a, 1)); plantStem.setMatrixAt(i, mS);
      }
      plantTop.instanceMatrix.needsUpdate = true; plantStem.instanceMatrix.needsUpdate = true;
    }
    function replayCutoff() { return _tMin + _replay.t * (_tMax - _tMin); }
    // animate=true → newly revealed plants pop in; false → instant (scrubbing)
    function applyReplay(animate) {
      if (!plantTop || !plantTop.count) { updateReplayUI(); return; }
      const cutoff = replayCutoff();
      for (let i = 0; i < plantTop.count; i++) {
        const vis = _plantTime[i] <= cutoff;
        if (vis && !_revealed[i]) { _revealed[i] = true; _appear[i] = animate ? 0 : 1; }
        else if (!vis && _revealed[i]) { _revealed[i] = false; _appear[i] = 0; }
        else if (vis && !animate) { _appear[i] = 1; }
      }
      _popActive = true;            // let the main loop settle pops / write matrices
      writePlantMatrices();
      updateReplayUI();
    }
    function replayTick(dtMs) {
      let changed = false;
      if (_replay.playing) {
        _replay.t = Math.min(1, _replay.t + (dtMs / _replay.durMs) * _replay.speed);
        // reveal any plants whose time is now under the head (pop them in)
        const cutoff = replayCutoff();
        for (let i = 0; i < plantTop.count; i++) {
          if (!_revealed[i] && _plantTime[i] <= cutoff) { _revealed[i] = true; _appear[i] = 0; }
        }
        if (_replay.t >= 1) { _replay.playing = false; }
        changed = true;
      }
      // advance pops
      let popping = false;
      for (let i = 0; i < plantTop.count; i++) {
        if (_revealed[i] && _appear[i] < 1) { _appear[i] = Math.min(1, _appear[i] + dtMs / 340); popping = true; }
      }
      if (changed || popping) { writePlantMatrices(); updateReplayUI(); }
      _popActive = popping || _replay.playing;
    }
    function replayClock() { try { return new Date(replayCutoff()).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); } catch (e) { return ''; } }
    function replayRevealedCount() { let c = 0; for (let i = 0; i < _revealed.length; i++) if (_revealed[i]) c++; return c; }
    function updateReplayUI() {
      const host = el.querySelector('#obs-timeline'); if (!host) return;
      const rg = host.querySelector('.an-obs-rp-range'); if (rg && !_scrubbing) rg.value = String(Math.round(_replay.t * 1000));
      const fill = host.querySelector('.an-obs-rp-fill'); if (fill) fill.style.width = (_replay.t * 100) + '%';
      const clk = host.querySelector('.an-obs-rp-clock'); if (clk) clk.textContent = replayClock();
      const cnt = host.querySelector('.an-obs-rp-count'); if (cnt) cnt.textContent = replayRevealedCount() + ' / ' + (plantTop ? plantTop.count : 0) + ' plants';
      const pb = host.querySelector('.an-obs-rp-play use'); if (pb) pb.setAttribute('href', _replay.playing ? '#i-pause' : '#i-play');
    }
    function playReplay() { if (_replay.t >= 1) { restartReplay(); return; } _replay.playing = true; _popActive = true; _lastFrame = performance.now(); updateReplayUI(); }
    function pauseReplay() { _replay.playing = false; updateReplayUI(); }
    function restartReplay() { _replay.t = 0; for (let i = 0; i < _revealed.length; i++) { _revealed[i] = false; _appear[i] = 0; } _replay.playing = true; _popActive = true; _lastFrame = performance.now(); applyReplay(true); }
    function seekReplay(t) { _replay.playing = false; _replay.t = Math.max(0, Math.min(1, t)); applyReplay(false); }

    // ── plant colour per layer ──────────────────────────────────────────
    function recolorPlants() {
      if (!plantTop || !plantTop.count) return;
      const col = new T.Color(); const now = Date.now();
      for (let i = 0; i < plantTop.count; i++) {
        const s = plantMap[i]; let hex = SAGE;
        const dim = (_focus && s.crop !== _focus) || (_diseaseFilter && s.diagnosis !== _diseaseFilter);
        if (_layer === 'crops') { const r = ROSTER.find(x => x.key === s.crop); hex = r ? r.leaf : SAGE; }
        else if (_layer === 'disease') { hex = HEALTHY[s.diagnosis] ? GREEN : (s.severity === 'severe' ? RED : s.severity === 'moderate' ? AMBER : 0xd9c24e); }
        else if (_layer === 'confidence') { const c = s.confidence || 0; hex = c >= 0.85 ? GREEN : c >= 0.65 ? SAGE : c >= 0.45 ? AMBER : RED; }
        else if (_layer === 'growth') { const age = (now - new Date(s.processed_at).getTime()) / 864e5; hex = age < 1 ? GREEN : age < 3 ? SAGE : age < 7 ? 0xbfcf8a : 0xb6ad97; }
        col.setHex(hex); if (dim) col.lerp(new T.Color(PAPER), 0.55);
        plantTop.setColorAt(i, col);
      }
      if (plantTop.instanceColor) plantTop.instanceColor.needsUpdate = true;
    }

    // ── landmarks (procedural) ──────────────────────────────────────────
    function box(w, h, d, color, x, y, z) { const m = new T.Mesh(new T.BoxGeometry(w, h, d), new T.MeshToonMaterial({ color: color })); m.position.set(x || 0, y || 0, z || 0); return m; }
    function landmark(kind, cx, cz, dormant, r) {
      const g = new T.Group(); g.position.set(cx, 0.22, cz - 0.2);
      const tone = dormant ? 0x9b927e : SOIL;
      if (kind === 'windmill') {
        g.add(box(0.5, 0.9, 0.5, dormant ? 0xb8b09c : 0xe7dcc4, 0, 0.45, 0));
        const roof = new T.Mesh(new T.ConeGeometry(0.42, 0.45, 4), new T.MeshToonMaterial({ color: dormant ? 0x9b927e : RED })); roof.position.y = 1.1; roof.rotation.y = Math.PI / 4; g.add(roof);
        const hub = box(0.12, 0.12, 0.12, INK, 0, 0.95, 0.28); g.add(hub);
        const blades = new T.Group(); blades.position.set(0, 0.95, 0.34);
        for (let i = 0; i < 4; i++) { const bl = box(0.08, 0.6, 0.02, dormant ? 0x9b927e : 0x8a6d3e, 0, 0.32, 0); bl.rotation.z = i * Math.PI / 2; const w2 = new T.Group(); w2.add(bl); w2.rotation.z = i * Math.PI / 2; blades.add(w2); }
        blades.userData = { spin: dormant ? 0 : 0.012 }; g.add(blades); g.userData.blades = blades;
      } else if (kind === 'barn') {
        g.add(box(0.8, 0.5, 0.6, dormant ? 0xa89a7e : 0x9a6a3e, 0, 0.25, 0));
        const roof = new T.Mesh(new T.CylinderGeometry(0.001, 0.46, 0.36, 4), new T.MeshToonMaterial({ color: dormant ? 0x8a8270 : 0x5b4632 })); roof.position.y = 0.66; roof.rotation.y = Math.PI / 4; g.add(roof);
        g.add(box(0.22, 0.3, 0.02, dormant ? 0x6b6552 : 0x3b2c1e, 0, 0.16, 0.31));
      } else {
        // greenhouse / shed (tomato — usually dormant)
        const gh = box(0.7, 0.5, 0.5, dormant ? 0xb6ad97 : 0xbfe0e8, 0, 0.25, 0); gh.material.transparent = true; gh.material.opacity = dormant ? 1 : 0.6; g.add(gh);
        const roof = new T.Mesh(new T.CylinderGeometry(0.001, 0.4, 0.3, 4), new T.MeshToonMaterial({ color: dormant ? 0x9b927e : 0x8aa0a6 })); roof.position.y = 0.6; roof.rotation.y = Math.PI / 4; g.add(roof);
      }
      return g;
    }
    function addPond(cx, cz) {
      const g = new T.Group(); g.position.set(cx, 0.22, cz);
      const water = new T.Mesh(new T.CircleGeometry(1.1, 28), new T.MeshToonMaterial({ color: WATER, transparent: true, opacity: 0.9 }));
      water.rotation.x = -Math.PI / 2; water.position.y = 0.01; g.add(water);
      // ring of rocks (instanced)
      const rg = new T.IcosahedronGeometry(0.16, 0); const rm = new T.MeshToonMaterial({ color: 0x9a8f7a });
      const rocks = new T.InstancedMesh(rg, rm, 18); const m = new T.Matrix4();
      for (let i = 0; i < 18; i++) { const a = i / 18 * Math.PI * 2; m.makeTranslation(cx + Math.cos(a) * 1.18, 0.24, cz + Math.sin(a) * 1.18); rocks.setMatrixAt(i, m); }
      rocks.instanceMatrix.needsUpdate = true; scene.add(rocks); landmarks.push(rocks);
      scene.add(g); landmarks.push(g);
    }
    function addClouds() {
      for (let i = 0; i < 2; i++) {
        const c = new T.Mesh(new T.CircleGeometry(2.4, 20), new T.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.05 }));
        c.rotation.x = -Math.PI / 2; c.position.set((i ? 3 : -4), 0.12, (i ? -2 : 3)); c.userData = { drift: (i ? 0.004 : 0.0028) }; scene.add(c); clouds.push(c);
      }
    }

    // ── orbit ───────────────────────────────────────────────────────────
    function applyOrbit() { const o = orb, sp = Math.sin(o.phi), cp = Math.cos(o.phi); cam.position.set(o.target.x + o.radius * sp * Math.sin(o.theta), o.target.y + o.radius * cp, o.target.z + o.radius * sp * Math.cos(o.theta)); cam.lookAt(o.target.x, o.target.y, o.target.z); }
    function orbitTo(tx, ty, tz, radius, theta, phi) { orbAnim = { ft: orb.target.clone(), tt: new T.Vector3(tx, ty, tz), fr: orb.radius, tr: radius, fth: orb.theta, tth: theta == null ? orb.theta : theta, fph: orb.phi, tph: phi == null ? orb.phi : phi, start: performance.now() }; }
    function panBy(dx, dy) { const o = orb; const fwd = new T.Vector3(o.target.x - cam.position.x, o.target.y - cam.position.y, o.target.z - cam.position.z).normalize(); const right = new T.Vector3().crossVectors(fwd, new T.Vector3(0, 1, 0)).normalize(); const up = new T.Vector3().crossVectors(right, fwd).normalize(); const sp = o.radius * 0.0016; o.target.x += (-dx * sp) * right.x + (dy * sp) * up.x; o.target.y += (-dx * sp) * right.y + (dy * sp) * up.y; o.target.z += (-dx * sp) * right.z + (dy * sp) * up.z; }
    function onDown(ev) { orbAnim = null; const pan = ev.button === 2 || ev.shiftKey || ev.metaKey || ev.ctrlKey; drag = { mode: pan ? 'pan' : 'rotate', x: ev.clientX, y: ev.clientY, moved: 0 }; node.style.cursor = 'grabbing'; try { dom.setPointerCapture(ev.pointerId); } catch (e) { } }
    function onPMove(ev) { if (drag) { const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y; drag.x = ev.clientX; drag.y = ev.clientY; drag.moved += Math.abs(dx) + Math.abs(dy); hideTip(); if (drag.mode === 'rotate') { orb.theta -= dx * 0.006; orb.phi = Math.max(0.18, Math.min(1.45, orb.phi - dy * 0.006)); } else panBy(dx, dy); applyOrbit(); return; } onHover(ev); }
    function onUp(ev) { if (!drag) return; const click = drag.moved < 6; drag = null; node.style.cursor = 'grab'; try { dom.releasePointerCapture(ev.pointerId); } catch (e) { } if (click) onClick(ev); }
    function onWheel(ev) { ev.preventDefault(); orbAnim = null; orb.radius = Math.max(2.4, Math.min(16, orb.radius * (1 + (ev.deltaY > 0 ? 1 : -1) * 0.09))); applyOrbit(); }

    // ── picking ─────────────────────────────────────────────────────────
    function ndc(ev) { const r = dom.getBoundingClientRect(); mouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1; mouse.y = -((ev.clientY - r.top) / r.height) * 2 + 1; raycaster.setFromCamera(mouse, cam); }
    function pickPlant(ev) { ndc(ev); if (!plantTop || !plantTop.count) return -1; const h = raycaster.intersectObject(plantTop, false); return (h.length && h[0].instanceId != null) ? h[0].instanceId : -1; }
    function pickTile(ev) { ndc(ev); if (!tilesMesh) return -1; const h = raycaster.intersectObject(tilesMesh, false); return (h.length && h[0].instanceId != null) ? h[0].instanceId : -1; }
    function onHover(ev) {
      const pi = pickPlant(ev);
      if (pi >= 0) { node.style.cursor = 'pointer'; const s = plantMap[pi]; raisePlant(pi);
        showTip(ev.clientX, ev.clientY, '<b>' + esc(titleCase(s.diagnosis)) + '</b><br>' + esc(titleCase(s.crop)) + ' · ' + (s.severity || '·') + (s.confidence ? '<br>conf ' + (s.confidence).toFixed(2) : '') + (s.geo_state ? '<br>' + esc(s.geo_state) : '') + '<br><i>click → inspect this prediction</i>'); return; }
      if (_hoverPlant >= 0) { lowerPlant(_hoverPlant); _hoverPlant = -1; }
      const ti = pickTile(ev);
      if (ti >= 0) { node.style.cursor = 'pointer'; const k = tileMap[ti]; const d = districts[k]; if (!d) { hideTip(); return; }
        glowDistrict(k);
        if (d.dormant) showTip(ev.clientX, ev.clientY, '<b>' + d.roster.label + ' Valley</b><br>dormant · no scans<br><i>supported — awaiting first scan</i>');
        else { const top = (d.data.diseases || [])[0]; showTip(ev.clientX, ev.clientY, '<b>' + d.roster.label + '</b><br>' + d.count + ' predictions · ' + (d.data.pct || 0) + '%' + (top ? '<br>top: ' + titleCase(top.disease) + ' (' + top.count + ')' : '') + '<br><i>click → fly into ' + d.roster.label + '</i>'); }
        return; }
      glowDistrict(null); hideTip(); node.style.cursor = drag ? 'grabbing' : 'grab';
    }
    function onClick(ev) {
      const pi = pickPlant(ev);
      if (pi >= 0) { openInspector(plantMap[pi]); return; }
      const ti = pickTile(ev);
      if (ti >= 0) { const k = tileMap[ti]; const d = districts[k]; if (d && !d.dormant) flyToDistrict(k); return; }
      if (_focus) clearFocus();
    }
    function raisePlant(i) { if (_hoverPlant === i) return; if (_hoverPlant >= 0) lowerPlant(_hoverPlant); _hoverPlant = i; movePlantY(i, 0.12); }
    function lowerPlant(i) { movePlantY(i, 0); }
    function movePlantY(i, dy) { if (!plantTop || i < 0 || i >= plantTop.count) return; const b = plantBase[i]; const m = new T.Matrix4().makeTranslation(b.x, b.y + dy, b.z); plantTop.setMatrixAt(i, m); plantTop.instanceMatrix.needsUpdate = true; }
    let _glowKey = null;
    function glowDistrict(k) {
      if (_glowKey === k) return; _glowKey = k;
      if (!tilesMesh) return; const col = new T.Color();
      for (let i = 0; i < tileMap.length; i++) { const tk = tileMap[i]; const d = districts[tk];
        const base = d && d.dormant ? 0xb6ad97 : ((i % 2) ? GRASS : GRASS_D);
        col.setHex(base); if (k && tk === k) col.lerp(new T.Color(GREEN), 0.28); tilesMesh.setColorAt(i, col); }
      if (tilesMesh.instanceColor) tilesMesh.instanceColor.needsUpdate = true;
    }

    // ── focus / fly-in ──────────────────────────────────────────────────
    function flyToDistrict(k) { const d = districts[k]; if (!d) return; _focus = k; recolorPlants(); orbitTo(d.center[0], 0, d.center[1], 5.2, orb.theta, 0.95); renderCrumb(); renderPanels(); renderInsight(); }
    function clearFocus() { _focus = null; _diseaseFilter = null; recolorPlants(); orbitTo(0, 0, 0, 9, orb.theta, 0.82); renderCrumb(); renderPanels(); renderInsight(); }
    function renderCrumb() { const c = el.querySelector('#obs-crumb'); if (!c) return; if (_focus) { const d = districts[_focus]; c.hidden = false; c.innerHTML = '<button id="obs-back">← all crops</button> <b>' + (d ? d.roster.label : '') + ' District</b>'; const b = el.querySelector('#obs-back'); if (b) b.addEventListener('click', clearFocus); } else c.hidden = true; }

    // ── prediction inspector ────────────────────────────────────────────
    function openInspector(s) {
      const ins = el.querySelector('#obs-ins'); if (!ins) return; ins.hidden = false;
      ins.innerHTML = '<div class="an-obs-ins-c"><button class="an-obs-ins-x" aria-label="Close">✕</button><div class="an-ph">loading prediction&hellip;</div></div>';
      ins.querySelector('.an-obs-ins-x').addEventListener('click', () => { ins.hidden = true; });
      api('scan/' + encodeURIComponent(s.scan_uid)).then(d => {
        if (destroyed || ins.hidden) return; if (!d || !d.found) { ins.querySelector('.an-obs-ins-c').innerHTML = '<button class="an-obs-ins-x">✕</button><div class="an-ph">prediction unavailable</div>'; bindX(); return; }
        const img = d.has_image ? '<img class="an-obs-ins-img" src="/api/account/analytics/scan/' + encodeURIComponent(s.scan_uid) + '/image" alt="leaf">' : '<div class="an-obs-ins-noimg"><span class="an-obs-leaf"></span>no image stored</div>';
        const geo = [d.geo_district, d.geo_state, d.geo_cc].filter(Boolean).join(', ') || 'unknown';
        // The inspector location is the photo's GPS capture point (lat/lon reverse-
        // geocoded), NOT the request IP origin shown by the Geographic Origin widget.
        const hasGps = (d.latitude != null && d.longitude != null);
        const gpsLine = hasGps ? '<div><dt>GPS</dt><dd>' + Number(d.latitude).toFixed(3) + ', ' + Number(d.longitude).toFixed(3) + '</dd></div>' : '';
        const payload = (d.payload && typeof d.payload === 'object') ? d.payload : null;
        ins.innerHTML = '<div class="an-obs-ins-c"><button class="an-obs-ins-x" aria-label="Close">✕</button>' +
          '<div class="an-obs-ins-h">Prediction Inspector</div>' + img +
          '<div class="an-obs-ins-diag ' + (HEALTHY[d.diagnosis] ? 'ok' : (d.severity === 'severe' ? 'bad' : 'warn')) + '">' + esc(titleCase(d.diagnosis)) + '</div>' +
          '<dl class="an-obs-ins-dl">' +
          '<div><dt>Confidence</dt><dd>' + (d.confidence != null ? Math.round(d.confidence * 100) + '%' : '·') + '</dd></div>' +
          '<div><dt>Severity</dt><dd>' + esc(d.severity || '·') + (d.is_ood ? ' · OOD' : '') + '</dd></div>' +
          '<div><dt>Crop</dt><dd>' + esc(titleCase(d.crop)) + '</dd></div>' +
          '<div><dt>Capture site</dt><dd>' + esc(geo) + '</dd></div>' +
          gpsLine +
          '<div><dt>Captured</dt><dd>' + esc(localTime(d.captured_at)) + '</dd></div>' +
          (d.processing_ms ? '<div><dt>Inference</dt><dd>' + d.processing_ms + 'ms</dd></div>' : '') +
          '</dl>' +
          (payload ? '<button class="an-obs-ins-pbtn" type="button" aria-expanded="false">See payload</button><pre class="an-obs-ins-pre" hidden></pre>' : '') +
          '<div class="an-obs-ins-uid">' + esc(d.scan_uid) + '</div></div>';
        bindX();
        if (payload) {
          const btn = ins.querySelector('.an-obs-ins-pbtn'), pre = ins.querySelector('.an-obs-ins-pre');
          btn.addEventListener('click', () => {
            const show = pre.hidden;
            if (show && !pre.dataset.filled) { pre.textContent = JSON.stringify(payload, null, 2); pre.dataset.filled = '1'; }
            pre.hidden = !show; btn.setAttribute('aria-expanded', String(show)); btn.textContent = show ? 'Hide payload' : 'See payload';
          });
        }
      }).catch(() => { });
      function bindX() { const x = ins.querySelector('.an-obs-ins-x'); if (x) x.addEventListener('click', () => { ins.hidden = true; }); }
    }

    // ── panels ──────────────────────────────────────────────────────────
    function renderPanels() { renderCrop(); renderDisease(); renderFeed(); }
    function renderInsight() {
      const host = el.querySelector('#obs-insight'); if (!host || !_data) return;
      const total = _data.total || 0; const crops = (_data.crops || []);
      const lead = crops.slice().sort((a, b) => b.count - a.count)[0];
      const allDis = []; crops.forEach(c => (c.diseases || []).forEach(dd => { if (!HEALTHY[dd.disease]) allDis.push(dd); }));
      allDis.sort((a, b) => b.count - a.count);
      const layerNote = _layer === 'disease' ? 'Disease layer: green healthy · amber/red infected.' : _layer === 'confidence' ? 'Confidence layer: lush = sure, sparse/red = uncertain.' : _layer === 'growth' ? 'Growth layer: bright = fresh scans, faded = older.' : 'Crops layer: each district is a supported crop.';
      host.innerHTML = '<b>' + fmt(total) + '</b> predictions' + (lead ? ' · ' + esc(titleCase(lead.crop)) + ' leads (' + (lead.pct || 0) + '%)' : '') + (allDis[0] ? ' · top disease ' + esc(titleCase(allDis[0].disease)) + ' (' + allDis[0].count + ')' : '') + ' <span class="an-obs-note">' + layerNote + '</span>';
    }
    // Per-crop rollup used by the layer-aware Crop Intel tiles.
    function cropAgg(key) {
      const c = ((_data && _data.crops) || []).find(x => x.crop === key);
      const cnt = c ? (c.count || 0) : 0;
      let infected = 0;
      (c && c.diseases || []).forEach(dd => { if (!HEALTHY[dd.disease]) infected += (dd.count || 0); });
      infected = Math.min(infected, cnt);
      const mine = _scans.filter(s => s.crop === key);
      const now = Date.now();
      let cSum = 0, cN = 0, fresh1 = 0, fresh3 = 0;
      mine.forEach(s => {
        if (s.confidence != null) { cSum += s.confidence; cN++; }
        const age = (now - new Date(s.processed_at || s.captured_at).getTime()) / 864e5;
        if (age < 1) fresh1++; if (age < 3) fresh3++;
      });
      const avgConf = cN ? cSum / cN : 0;
      return {
        cnt: cnt, infected: infected, healthy: Math.max(0, cnt - infected),
        infectionRate: cnt ? infected / cnt : 0,
        avgConf: avgConf, fresh1: fresh1, fresh3: fresh3,
        freshShare: mine.length ? fresh3 / mine.length : 0,
      };
    }

    const CROP_PANEL_TITLE = { crops: 'Crop Intel', disease: 'Infection by Crop', confidence: 'Confidence by Crop', growth: 'Growth by Crop' };
    function renderCrop() {
      const host = el.querySelector('#obs-crop-b'); if (!host || !_data) return;
      const ph = el.querySelector('#obs-crop .an-obs-ph'); if (ph) ph.textContent = CROP_PANEL_TITLE[_layer] || 'Crop Intel';
      const total = _data.total || 1;
      host.innerHTML = ROSTER.map(r => {
        const a = cropAgg(r.key); const dorm = !a.cnt;
        let val, barW, barCol, sub;
        if (_layer === 'disease') {
          val = dorm ? 'dormant' : a.infected + ' infected';
          barW = dorm ? 0 : Math.round(a.infectionRate * 100); barCol = a.infectionRate >= 0.5 ? '#b3402f' : a.infectionRate >= 0.2 ? '#c98a2b' : '#52b788';
          sub = dorm ? '' : a.healthy + ' healthy · ' + Math.round(a.infectionRate * 100) + '% sick';
        } else if (_layer === 'confidence') {
          val = dorm ? 'dormant' : Math.round(a.avgConf * 100) + '%';
          barW = dorm ? 0 : Math.round(a.avgConf * 100); barCol = a.avgConf >= 0.8 ? '#52b788' : a.avgConf >= 0.6 ? '#6cbf94' : a.avgConf >= 0.45 ? '#c98a2b' : '#b3402f';
          sub = dorm ? '' : 'avg confidence across ' + a.cnt + ' scans';
        } else if (_layer === 'growth') {
          val = dorm ? 'dormant' : a.fresh3 + ' fresh';
          barW = dorm ? 0 : Math.round(a.freshShare * 100); barCol = '#52b788';
          sub = dorm ? '' : a.fresh1 + ' today · ' + a.fresh3 + ' in 3d';
        } else { // crops
          val = dorm ? 'dormant' : a.cnt;
          barW = dorm ? 0 : Math.round(a.cnt / total * 100); barCol = r.accent;
          sub = dorm ? '' : Math.round(a.cnt / total * 100) + '% of all scans';
        }
        return '<div class="an-obs-card' + (dorm ? ' dorm' : '') + (_focus === r.key ? ' on' : '') + '" data-crop="' + r.key + '"' + (dorm ? '' : ' tabindex="0"') + '>' +
          '<div class="an-obs-card-h"><i style="background:' + r.accent + '"></i>' + r.label + '<b>' + val + '</b></div>' +
          '<div class="an-obs-card-bar"><span style="width:' + barW + '%;background:' + barCol + '"></span></div>' +
          (sub ? '<div class="an-obs-card-sub">' + esc(sub) + '</div>' : '') + '</div>';
      }).join('');
      host.querySelectorAll('.an-obs-card[data-crop]').forEach(cd => {
        const k = cd.getAttribute('data-crop'); const d = districts[k];
        cd.addEventListener('mouseenter', () => glowDistrict(d && !d.dormant ? k : null));
        cd.addEventListener('mouseleave', () => glowDistrict(_focus));
        cd.addEventListener('click', () => { if (d && !d.dormant) flyToDistrict(k); });
      });
    }
    function renderDisease() {
      const host = el.querySelector('#obs-disease-b'); if (!host || !_data) return;
      const all = []; (_data.crops || []).forEach(c => (c.diseases || []).forEach(dd => { if (!HEALTHY[dd.disease]) all.push(dd); }));
      all.sort((a, b) => b.count - a.count); const max = Math.max(1, ...all.map(d => d.count));
      host.innerHTML = all.slice(0, 6).map(d => '<button class="an-obs-drow' + (_diseaseFilter === d.disease ? ' on' : '') + '" data-dis="' + esc(d.disease) + '"><span class="an-obs-dnm">' + esc(titleCase(d.disease)) + '</span><span class="an-obs-dbar"><i style="width:' + Math.round(d.count / max * 100) + '%"></i></span><span class="an-obs-dn">' + d.count + '</span></button>').join('') || '<div class="an-obs-mut">no diseases — all healthy</div>';
      host.querySelectorAll('.an-obs-drow[data-dis]').forEach(b => { const dis = b.getAttribute('data-dis');
        b.addEventListener('mouseenter', () => { _diseaseFilter = dis; recolorPlants(); });
        b.addEventListener('mouseleave', () => { if (el.querySelector('.an-obs-drow.on') == null) { _diseaseFilter = null; recolorPlants(); } });
        b.addEventListener('click', () => { _diseaseFilter = (_diseaseFilter === dis ? null : dis); _layer = 'disease'; el.querySelectorAll('#obs-layer [data-layer]').forEach(x => x.setAttribute('aria-pressed', x.getAttribute('data-layer') === 'disease' ? 'true' : 'false')); recolorPlants(); renderDisease(); renderInsight(); });
      });
    }
    function renderFeed() {
      const host = el.querySelector('#obs-feed-b'); if (!host) return;
      const list = _scans.slice(0, 18);
      host.innerHTML = list.length ? list.map(feedRow).join('') : '<div class="an-obs-mut">no predictions in this window</div>';
      bindFeed(host);
    }
    function feedRow(s) {
      const cls = HEALTHY[s.diagnosis] ? 'ok' : (s.severity === 'severe' ? 'bad' : 'warn');
      return '<button class="an-obs-feedrow" data-uid="' + esc(s.scan_uid) + '"><span class="an-obs-fdot ' + cls + '"></span>' +
        '<span class="an-obs-fnm">' + esc(titleCase(s.diagnosis)) + '<small>' + esc(s.geo_state || s.geo_cc || '') + (s.confidence ? ' · ' + Math.round(s.confidence * 100) + '%' : '') + '</small></span>' +
        '<span class="an-obs-fago">' + esc(ipAgo(s.processed_at)) + '</span></button>';
    }
    function bindFeed(host) {
      host.querySelectorAll('.an-obs-feedrow[data-uid]').forEach(b => b.addEventListener('click', () => {
        const uid = b.getAttribute('data-uid'); const idx = plantMap.findIndex(p => p && p.scan_uid === uid);
        if (idx >= 0) { const bse = plantBase[idx]; flyToPoint(bse.x, bse.z); pulsePlant(idx); }
        const s = plantMap[idx] || _scans.find(x => x.scan_uid === uid); if (s) openInspector(s);
      }));
    }
    function flyToPoint(x, z) { orbitTo(x, 0, z, 3.6, orb.theta, 1.0); }
    function pulsePlant(i) { raisePlant(i); setTimeout(() => lowerPlant(i), 800); }

    // ── replay scrubber (media controls) ────────────────────────────────
    const SPEEDS = [0.5, 1, 2, 4];
    function svgIcon(id) { return '<svg viewBox="0 0 24 24" aria-hidden="true"><use href="#' + id + '"/></svg>'; }
    function renderReplay() {
      const host = el.querySelector('#obs-timeline'); if (!host) return;
      const spd = SPEEDS.map(s => '<button class="an-obs-rp-spd' + (s === _replay.speed ? ' on' : '') + '" data-spd="' + s + '">' + (s === 1 ? '1×' : s + '×') + '</button>').join('');
      host.innerHTML =
        '<div class="an-obs-rp-ctrls">' +
          '<button class="an-obs-rp-btn an-obs-rp-restart" title="Restart" aria-label="Restart">' + svgIcon('i-skip-back') + '</button>' +
          '<button class="an-obs-rp-btn an-obs-rp-play" title="Play / pause" aria-label="Play or pause">' + svgIcon('i-play') + '</button>' +
          '<span class="an-obs-rp-speed">' + spd + '</span>' +
        '</div>' +
        '<div class="an-obs-rp-scrub">' +
          '<i class="an-obs-rp-fill"></i>' +
          '<input type="range" class="an-obs-rp-range" min="0" max="1000" step="1" value="' + Math.round(_replay.t * 1000) + '" aria-label="Replay scrubber">' +
        '</div>' +
        '<div class="an-obs-rp-meta"><span class="an-obs-rp-clock"></span><span class="an-obs-rp-count"></span></div>';
      host.querySelector('.an-obs-rp-restart').addEventListener('click', restartReplay);
      host.querySelector('.an-obs-rp-play').addEventListener('click', () => { _replay.playing ? pauseReplay() : playReplay(); });
      host.querySelectorAll('.an-obs-rp-spd[data-spd]').forEach(b => b.addEventListener('click', () => {
        _replay.speed = parseFloat(b.getAttribute('data-spd'));
        host.querySelectorAll('.an-obs-rp-spd').forEach(x => x.classList.toggle('on', x === b));
      }));
      const rg = host.querySelector('.an-obs-rp-range');
      rg.addEventListener('pointerdown', () => { _scrubbing = true; _replay.playing = false; });
      rg.addEventListener('input', () => { seekReplay(parseInt(rg.value, 10) / 1000); });
      const stop = () => { _scrubbing = false; }; rg.addEventListener('pointerup', stop); rg.addEventListener('pointercancel', stop); rg.addEventListener('blur', stop);
      updateReplayUI();
    }

    // ── live ────────────────────────────────────────────────────────────
    function onScan(e) {
      if (destroyed || !e) return; const key = (e.crop || (e.diagnosis || '').split('_')[0] || '').toLowerCase();
      const d = districts[key]; if (!d || d.dormant) return;
      // seed particle → sprout at a random tile of the district
      const tc = d.tiles[(Math.random() * d.tiles.length) | 0]; const px = tc[0] + (Math.random() - 0.5) * 0.7, pz = tc[1] + (Math.random() - 0.5) * 0.7;
      const seed = new T.Mesh(new T.SphereGeometry(0.06, 8, 8), new T.MeshToonMaterial({ color: OCHRE })); seed.position.set(px, 2.2, pz); scene.add(seed);
      const start = performance.now();
      (function fall() { if (destroyed) { scene.remove(seed); return; } const t = Math.min(1, (performance.now() - start) / 600); seed.position.y = 2.2 - t * 2.0 + 0.42; if (t < 1) requestAnimationFrame(fall); else { scene.remove(seed); sprout(px, pz, key); } })();
      // prepend to feed
      const host = el.querySelector('#obs-feed-b'); if (host && e.scan_uid) { const s = { scan_uid: e.scan_uid, diagnosis: e.diagnosis, crop: key, severity: e.severity, confidence: e.confidence, geo_state: e.geo_state, geo_cc: e.geo_cc, processed_at: new Date().toISOString() }; _scans.unshift(s); const div = document.createElement('div'); div.innerHTML = feedRow(s); const row = div.firstChild; row.classList.add('fresh'); host.insertBefore(row, host.firstChild); bindFeed(host); while (host.children.length > 24) host.removeChild(host.lastChild); }
    }
    function sprout(x, z, key) {
      const r = ROSTER.find(k => k.key === key) || ROSTER[0];
      const g = new T.Group(); const st = new T.Mesh(stemGeo, new T.MeshToonMaterial({ color: GREEN_D })); st.position.set(x, 0.32, z);
      const tp = new T.Mesh(topGeo, new T.MeshToonMaterial({ color: r.leaf })); tp.position.set(x, 0.5, z); g.add(st); g.add(tp); g.scale.setScalar(0.001); scene.add(g); landmarks.push(g);
      const start = performance.now(); (function grow() { if (destroyed) return; const t = Math.min(1, (performance.now() - start) / 500); g.scale.setScalar(t); if (t < 1) requestAnimationFrame(grow); })();
    }

    // ── tooltip ─────────────────────────────────────────────────────────
    let tipEl = null;
    function tip() { if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'an-obs-tip'; tipEl.style.display = 'none'; document.body.appendChild(tipEl); } return tipEl; }
    function showTip(x, y, html) { const t = tip(); t.innerHTML = html; t.style.display = 'block'; t.style.left = Math.min(x + 14, window.innerWidth - 220) + 'px'; t.style.top = (y + 14) + 'px'; }
    function hideTip() { if (tipEl) tipEl.style.display = 'none'; }

    // ── loop ────────────────────────────────────────────────────────────
    function loop() {
      raf = requestAnimationFrame(loop); if (destroyed || !renderer) return; const now = performance.now();
      const dt = Math.min(60, now - (_lastFrame || now)); _lastFrame = now;
      if (_popActive || _replay.playing) replayTick(dt);
      landmarks.forEach(g => { if (g.userData && g.userData.blades) g.userData.blades.rotation.z += g.userData.blades.userData.spin; });
      clouds.forEach(c => { c.position.x += c.userData.drift; if (c.position.x > 8) c.position.x = -8; });
      if (orbAnim) { const t = Math.min(1, (now - orbAnim.start) / 900), e = t * t * (3 - 2 * t); orb.target.lerpVectors(orbAnim.ft, orbAnim.tt, e); orb.radius = orbAnim.fr + (orbAnim.tr - orbAnim.fr) * e; orb.theta = orbAnim.fth + (orbAnim.tth - orbAnim.fth) * e; orb.phi = orbAnim.fph + (orbAnim.tph - orbAnim.fph) * e; if (t >= 1) orbAnim = null; }
      applyOrbit(); renderer.render(scene, cam);
    }
    loop();
    function resize() { if (!renderer || !node) return; const w = node.clientWidth, h = node.clientHeight; if (w < 40 || h < 40) return; cam.aspect = w / h; cam.updateProjectionMatrix(); renderer.setSize(w, h); }
    function destroy() {
      destroyed = true; if (raf) cancelAnimationFrame(raf); if (ro) try { ro.disconnect(); } catch (e) { }
      try { renderer && renderer.dispose && renderer.dispose(); } catch (e) { }
      try { scene && scene.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m && m.dispose && m.dispose()); }); } catch (e) { }
      hideTip();
    }
    return { onScan: onScan, destroy: destroy, alive: () => !destroyed && el.isConnected };
  }

  // ── CSS ──────────────────────────────────────────────────────────────
  function injectCSS() {
    if (document.getElementById('an-obs-css')) return; const s = document.createElement('style'); s.id = 'an-obs-css';
    s.textContent = `
.an-obs{display:flex;flex-direction:column;gap:10px;height:80vh;min-height:520px}
.an-obs-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.an-obs-head h3{font:600 17px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin:0}
.an-obs-tools{display:flex;gap:10px;flex-wrap:wrap}
.an-obs-seg{display:inline-flex;border:1px solid var(--paper-edge,#d8cdb2);border-radius:8px;overflow:hidden}
.an-obs-seg button{font:600 11px 'JetBrains Mono',monospace;border:0;background:var(--paper,#efe7d4);color:var(--ink-soft,#5b5446);padding:5px 11px;cursor:pointer}
.an-obs-seg button[aria-pressed=true]{background:var(--ink,#1a1612);color:#efe7d4}
.an-obs-world{position:relative;flex:1;min-height:280px;border-radius:12px;overflow:hidden;border:1px solid var(--paper-edge,#d8cdb2);background:#f3ecd9;background-image:radial-gradient(circle,rgba(26,22,18,.10) 1px,transparent 1px);background-size:22px 22px;-webkit-user-select:none;user-select:none;touch-action:none}
.an-obs-world .maplibregl-canvas,.an-obs-world canvas{outline:none}
.an-obs-crumb{position:absolute;left:12px;top:12px;z-index:3;font:11px 'JetBrains Mono',monospace;background:rgba(239,231,212,.9);border:1px solid var(--paper-edge,#d8cdb2);border-radius:8px;padding:5px 9px;display:flex;align-items:center;gap:8px}
.an-obs-crumb b{font-family:'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.an-obs-crumb button{border:0;background:none;color:var(--accent-deep,#2d6a4f);font:inherit;cursor:pointer;font-weight:600}
.an-obs-insight{position:absolute;left:12px;bottom:12px;right:12px;z-index:2;font:12px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);background:rgba(239,231,212,.82);border-radius:8px;padding:7px 11px;backdrop-filter:blur(2px)}
.an-obs-insight b{font-weight:600}.an-obs-note{color:var(--ink-mute,#9a917d);font:10.5px 'JetBrains Mono',monospace;margin-left:8px}
.an-obs-tip{position:fixed;z-index:9999;pointer-events:none;max-width:210px;background:var(--ink,#1a1612);color:#efe7d4;font:11px 'JetBrains Mono',monospace;line-height:1.45;padding:7px 9px;border-radius:7px;box-shadow:0 4px 16px rgba(0,0,0,.3)}
.an-obs-tip b{color:#8fe0b4}.an-obs-tip i{color:#b9b09c;font-style:normal}
/* inspector */
.an-obs-ins{position:absolute;right:12px;top:12px;z-index:5;width:230px;max-height:calc(100% - 24px);overflow:auto;background:var(--paper,#efe7d4);border:1px solid var(--paper-edge,#d8cdb2);border-radius:12px;box-shadow:0 10px 32px rgba(20,16,12,.25);animation:anObsIn .22s ease}
@keyframes anObsIn{from{opacity:0;transform:translateX(10px)}to{opacity:1;transform:none}}
.an-obs-ins-c{padding:12px;position:relative}
.an-obs-ins-x{position:absolute;right:8px;top:8px;width:24px;height:24px;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);border-radius:6px;cursor:pointer;font-size:11px}
.an-obs-ins-h{font:600 12px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin-bottom:9px}
.an-obs-ins-img{width:100%;border-radius:8px;display:block;margin-bottom:9px}
.an-obs-ins-noimg{height:90px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;background:var(--paper-deep,#e7dcc4);border-radius:8px;font:10.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-bottom:9px}
.an-obs-leaf{width:16px;height:16px;border-radius:0 50% 0 50%;background:var(--accent,#52b788);opacity:.6;display:inline-block}
.an-obs-ins-diag{font:700 13px 'Fraunces',Georgia,serif;padding:5px 9px;border-radius:7px;margin-bottom:9px}
.an-obs-ins-diag.ok{background:#d8efe0;color:#2d6a4f}.an-obs-ins-diag.warn{background:#fbeccd;color:#8a6d1f}.an-obs-ins-diag.bad{background:#f3ddd6;color:#b3402f}
.an-obs-ins-dl{margin:0}.an-obs-ins-dl>div{display:flex;justify-content:space-between;gap:10px;font:11px 'JetBrains Mono',monospace;padding:3px 0;border-bottom:1px solid var(--paper-edge,#d8cdb2)}
.an-obs-ins-dl dt{color:var(--ink-mute,#9a917d)}.an-obs-ins-dl dd{margin:0;color:var(--ink,#1a1612);text-align:right}
.an-obs-ins-pbtn{margin-top:9px;width:100%;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper-deep,#e7dcc4);color:var(--accent-deep,#2d6a4f);font:600 10.5px 'JetBrains Mono',monospace;padding:6px 8px;border-radius:7px;cursor:pointer}
.an-obs-ins-pbtn:hover{background:var(--paper,#efe7d4)}
.an-obs-ins-pre{margin:8px 0 0;max-height:180px;overflow:auto;background:var(--ink,#1a1612);color:#d8e8d8;font:9.5px/1.5 'JetBrains Mono',monospace;padding:9px;border-radius:8px;white-space:pre;word-break:normal}
.an-obs-ins-uid{font:9px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:8px;word-break:break-all}
/* panels */
.an-obs-panels{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.an-obs-panel{background:var(--paper-deep,#e7dcc4);border:1px solid var(--paper-edge,#d8cdb2);border-radius:10px;padding:11px 12px;min-height:150px;max-height:210px;overflow:auto}
.an-obs-ph{font:600 12px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin-bottom:8px;display:flex;align-items:center;justify-content:space-between}
.an-obs-live{font:9px 'JetBrains Mono',monospace;color:var(--accent-deep,#2d6a4f);display:inline-flex;align-items:center;gap:4px}
.an-obs-live i{width:6px;height:6px;border-radius:50%;background:var(--accent,#52b788);animation:anObsBlink 1.6s infinite}
@keyframes anObsBlink{50%{opacity:.3}}
.an-obs-card{padding:6px 7px;margin:0 -7px;border-radius:7px;cursor:pointer;transition:background .12s}
.an-obs-card:hover,.an-obs-card:focus,.an-obs-card.on{background:var(--paper,#efe7d4);outline:none}
.an-obs-card.dorm{cursor:default;opacity:.6}
.an-obs-card-h{display:flex;align-items:center;gap:6px;font:11.5px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.an-obs-card-h i{width:9px;height:9px;border-radius:2px}.an-obs-card-h b{margin-left:auto;color:var(--ink-soft,#5b5446)}
.an-obs-card-bar{height:7px;background:var(--paper,#efe7d4);border-radius:4px;overflow:hidden;margin-top:5px}
.an-obs-card-bar span{display:block;height:100%;border-radius:4px;transition:width .4s,background .3s}
.an-obs-card-sub{font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:4px}
.an-obs-drow{display:grid;grid-template-columns:1fr 56px 28px;align-items:center;gap:7px;width:100%;padding:5px;margin:0 -5px;border:0;background:none;border-radius:6px;cursor:pointer;font:11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);transition:background .12s}
.an-obs-drow:hover,.an-obs-drow.on{background:var(--paper,#efe7d4)}
.an-obs-dnm{text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--ink-soft,#5b5446)}
.an-obs-dbar{height:7px;background:var(--paper,#efe7d4);border-radius:4px;overflow:hidden}.an-obs-dbar i{display:block;height:100%;background:var(--c-danger,#b3402f);border-radius:4px}
.an-obs-dn{text-align:right;font-weight:600;color:var(--ink-soft,#5b5446)}
.an-obs-feedrow{display:flex;align-items:center;gap:8px;width:100%;padding:5px;margin:0 -5px;border:0;background:none;border-radius:6px;cursor:pointer;text-align:left;transition:background .12s}
.an-obs-feedrow:hover{background:var(--paper,#efe7d4)}
.an-obs-feedrow.fresh{animation:anObsFresh 1.2s ease}
@keyframes anObsFresh{0%{background:#d8efe0}100%{background:transparent}}
.an-obs-fdot{width:8px;height:8px;border-radius:50%;flex:none}.an-obs-fdot.ok{background:#52b788}.an-obs-fdot.warn{background:#c98a2b}.an-obs-fdot.bad{background:#b3402f}
.an-obs-fnm{flex:1;min-width:0;font:11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;flex-direction:column}
.an-obs-fnm small{font-size:9px;color:var(--ink-mute,#9a917d)}
.an-obs-fago{font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);flex:none}
.an-obs-mut{font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
/* replay scrubber */
.an-obs-timeline{display:flex;align-items:center;gap:14px;background:var(--paper-deep,#e7dcc4);border:1px solid var(--paper-edge,#d8cdb2);border-radius:10px;padding:9px 16px}
.an-obs-rp-ctrls{display:flex;align-items:center;gap:8px;flex:none}
.an-obs-rp-btn{width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);border-radius:8px;cursor:pointer;color:var(--ink,#1a1612)}
.an-obs-rp-btn:hover{background:#fff;border-color:var(--accent,#52b788)}
.an-obs-rp-btn svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.an-obs-rp-play{background:var(--ink,#1a1612);color:#efe7d4;border-color:var(--ink,#1a1612)}
.an-obs-rp-play:hover{background:var(--accent-deep,#2d6a4f);border-color:var(--accent-deep,#2d6a4f)}
.an-obs-rp-speed{display:inline-flex;border:1px solid var(--paper-edge,#d8cdb2);border-radius:7px;overflow:hidden;margin-left:2px}
.an-obs-rp-spd{border:0;background:var(--paper,#efe7d4);color:var(--ink-soft,#5b5446);font:600 10px 'JetBrains Mono',monospace;padding:5px 7px;cursor:pointer}
.an-obs-rp-spd.on{background:var(--ink,#1a1612);color:#efe7d4}
.an-obs-rp-scrub{position:relative;flex:1;height:24px;display:flex;align-items:center}
.an-obs-rp-scrub:before{content:'';position:absolute;left:0;right:0;height:4px;background:var(--paper-edge,#cfc2a3);border-radius:2px}
.an-obs-rp-fill{position:absolute;left:0;height:4px;background:var(--accent-deep,#2d6a4f);border-radius:2px;pointer-events:none}
.an-obs-rp-range{position:relative;width:100%;margin:0;-webkit-appearance:none;appearance:none;background:none;height:24px;cursor:pointer}
.an-obs-rp-range::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:var(--paper,#efe7d4);border:2px solid var(--accent-deep,#2d6a4f);box-shadow:0 1px 4px rgba(20,16,12,.3);cursor:grab}
.an-obs-rp-range::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--paper,#efe7d4);border:2px solid var(--accent-deep,#2d6a4f);cursor:grab}
.an-obs-rp-range::-moz-range-track{background:transparent}
.an-obs-rp-meta{display:flex;flex-direction:column;align-items:flex-end;gap:1px;flex:none;min-width:120px}
.an-obs-rp-clock{font:600 11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612)}
.an-obs-rp-count{font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
@media (max-width:820px){.an-obs-timeline{flex-wrap:wrap;gap:8px}.an-obs-rp-meta{min-width:0}}
@media (max-width:820px){.an-obs-panels{grid-template-columns:1fr}}
`;
    document.head.appendChild(s);
  }

  O.open = open;
  O.onScan = function (e) { if (OBS && OBS.alive && OBS.alive()) OBS.onScan(e); };
  O.alive = function () { return !!(OBS && OBS.alive && OBS.alive()); };
  window.APIN = window.APIN || {};
  window.APIN.analyticsObservatory = O;
})();
