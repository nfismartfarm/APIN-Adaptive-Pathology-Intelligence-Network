/* 9.N.T31 · Crop Donut Field — 3D inference field (procedural three.js).
 *
 * A circular "donut of land" split into wedges by crop share, planted with
 * procedural low-poly plants. Fixed roster of the 3 supported crops — okra,
 * broccoli (the brassica class), tomato — each filled from /analytics/field.
 * A crop with no scans this window renders DORMANT (bare soil, "no scans yet"),
 * which is honest: the crop is supported, just unused by this key so far.
 *
 *   · donut split by proportion (min-arc so all crops stay visible)
 *   · center medallion (total scans) in the donut hole
 *   · dot-grid sketchbook background
 *   · top-right stats block (split bar · severity · confidence · top diseases)
 *   · per-WEDGE and per-PLANT hover + click, custom paper-ink tooltips
 *   · click a wedge → focus that crop · click a plant → drill into its disease
 *
 * Registered on APIN.analyticsField. No external mesh dependency.
 */
(function () {
  'use strict';
  const F = {};
  let THREE = null, host = null, insightHost = null, ro = null, raf = null;
  let scene, cam, renderer, rootGroup, ground, raycaster, mouse, medalEl = null;
  let _data = null, _mode = 'crop', _paused = false, _focusCrop = null;
  let _entries = [], _hover = null, _hoverPlant = null;
  let _onPickDisease = null, _onPickCrop = null;

  const GREEN = 0x2d6a4f, ACCENT = 0x52b788, DANGER = 0xb3402f, AMBER = 0xc98a2b;
  const rIn = 0.62, rOut = 1.55, DEPTH = 0.14;

  // fixed roster of supported crops (label · stat accent · plant leaf · soil)
  const ROSTER = [
    { key: 'okra', label: 'Okra', accent: '#52b788', leaf: 0x6cbf94, soil: 0x6b5b43 },
    { key: 'brassica', label: 'Broccoli', accent: '#2f6a43', leaf: 0x4f8f5a, soil: 0x665a44 },
    { key: 'tomato', label: 'Tomato', accent: '#c0563b', leaf: 0x6f9a52, soil: 0x6e5a46 },
  ];
  const DORMANT_SOIL = 0x8c8270;

  // ── data shaping ─────────────────────────────────────────────────────────
  function buildEntries() {
    const data = _data || {}; const crops = data.crops || [];
    const byKey = {}; crops.forEach(c => { byKey[c.crop] = c; });
    const total = data.total || 0;
    const raw = ROSTER.map(r => {
      const c = byKey[r.key] || { crop: r.key, count: 0, pct: 0, diseases: [] };
      return { roster: r, data: c, count: c.count || 0, pct: c.pct || 0, dormant: (c.count || 0) === 0 };
    });
    const MINF = 0.10, restF = 1 - MINF * ROSTER.length;
    raw.forEach(e => { e.frac = total > 0 ? MINF + (e.count / total) * restF : 1 / ROSTER.length; });
    const sum = raw.reduce((s, e) => s + e.frac, 0) || 1; raw.forEach(e => { e.frac /= sum; });
    return { entries: raw, total: total };
  }
  function cropSeverity(c) {
    const m = { mild: 0, moderate: 0, severe: 0 };
    (c.diseases || []).forEach(d => { const s = d.severity_mix || {}; m.mild += s.mild || 0; m.moderate += s.moderate || 0; m.severe += s.severe || 0; });
    return m;
  }
  function cropConf(c) { let n = 0, s = 0; (c.diseases || []).forEach(d => { n += d.count || 0; s += (d.avg_confidence || 0) * (d.count || 0); }); return n ? s / n : 0; }
  function pickDisease(c) {
    const ds = c.diseases || []; if (!ds.length) return null;
    const tot = ds.reduce((s, d) => s + (d.count || 0), 0) || 1; let x = Math.random() * tot;
    for (const d of ds) { x -= (d.count || 0); if (x <= 0) return d; }
    return ds[ds.length - 1];
  }
  function diseaseSeverity(d) { const s = (d && d.severity_mix) || {}; return s.severe ? 'severe' : s.moderate ? 'moderate' : 'mild'; }

  // ── meshes ───────────────────────────────────────────────────────────────
  function plantMesh(leafColor, height, wilt) {
    const g = new THREE.Group();
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.02, height, 5),
      new THREE.MeshToonMaterial({ color: GREEN }));
    stem.position.y = height / 2; g.add(stem);
    const leafMat = new THREE.MeshToonMaterial({ color: leafColor });
    for (let i = 0; i < 3; i++) {
      const leaf = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.14, 5), leafMat);
      const a = (i / 3) * Math.PI * 2;
      leaf.position.set(Math.cos(a) * 0.05, height - 0.05 - i * 0.04, Math.sin(a) * 0.05);
      leaf.rotation.z = Math.PI / 2.4; leaf.rotation.y = a; g.add(leaf);
    }
    if (wilt) g.rotation.z = 0.22 + Math.random() * 0.18;
    return g;
  }
  function wedgeMesh(a0, a1, soilColor) {
    // ring-sector shape extruded, then geometry rotated flat. A shape point at
    // angle θ maps to world (r·cosθ, DEPTH, −r·sinθ) — used for plant placement.
    const sh = new THREE.Shape();
    sh.absarc(0, 0, rOut, a0, a1, false);
    sh.absarc(0, 0, rIn, a1, a0, true);
    const geo = new THREE.ExtrudeGeometry(sh, { depth: DEPTH, bevelEnabled: false, curveSegments: 64 });
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshToonMaterial({ color: soilColor });
    return new THREE.Mesh(geo, mat);
  }
  function disposeGroup(g) {
    if (!g) return;
    g.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m && m.dispose && m.dispose()); });
  }

  // ── build the donut ──────────────────────────────────────────────────────
  function buildDonut() {
    if (!scene) return;
    if (rootGroup) { scene.remove(rootGroup); disposeGroup(rootGroup); }
    rootGroup = new THREE.Group(); scene.add(rootGroup);
    const { entries, total } = buildEntries(); _entries = entries;
    let ang = -Math.PI / 2; const gap = 0.035;
    entries.forEach(e => {
      const span = e.frac * Math.PI * 2;
      const wa0 = ang + gap / 2, wa1 = ang + span - gap / 2;
      e.a0 = wa0; e.a1 = wa1; e.mid = (wa0 + wa1) / 2;
      const wedge = wedgeMesh(wa0, wa1, e.dormant ? DORMANT_SOIL : e.roster.soil);
      wedge.userData = { wedge: true, entry: e };
      rootGroup.add(wedge); e.mesh = wedge; e.plants = [];
      if (!e.dormant) {
        const dense = Math.max(3, Math.min(40, Math.round((e.count / Math.max(1, total)) * 110) + 3));
        for (let p = 0; p < dense; p++) {
          // uniform-area sample within the wedge
          const t = wa0 + Math.random() * (wa1 - wa0);
          const r = Math.sqrt(rIn * rIn + Math.random() * (rOut * rOut - rIn * rIn));
          const dis = pickDisease(e.data); const sev = diseaseSeverity(dis);
          const h = 0.15 + Math.random() * 0.12 + (sev === 'severe' ? -0.05 : 0.03);
          const pm = plantMesh(e.roster.leaf, h, sev === 'severe' && Math.random() < 0.5);
          pm.position.set(r * Math.cos(t), DEPTH, -r * Math.sin(t));
          const sc = 0.62 + Math.random() * 0.5;
          pm.userData = { plant: true, entry: e, grow: sc, base: sc, disease: dis ? dis.disease : (e.roster.key + '_?'), severity: sev, confidence: dis ? dis.avg_confidence : 0 };
          pm.scale.setScalar(0.001);
          rootGroup.add(pm); e.plants.push(pm);
        }
      } else {
        // dormant marker — a faint ring of dashes handled by soil tint + tooltip
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 8), new THREE.MeshToonMaterial({ color: 0xb9b3a3 }));
        dot.position.set((rIn + rOut) / 2 * Math.cos(e.mid), DEPTH + 0.02, -(rIn + rOut) / 2 * Math.sin(e.mid));
        dot.userData = { wedge: true, entry: e };
        rootGroup.add(dot);
      }
      ang += span;
    });
    // center medallion (3D disc) — label is a CSS overlay over the hole
    const med = new THREE.Mesh(new THREE.CylinderGeometry(rIn - 0.04, rIn - 0.04, 0.1, 48),
      new THREE.MeshToonMaterial({ color: 0xefe7d4 }));
    med.position.y = 0.05; med.userData = { center: true }; rootGroup.add(med);
    growIn();
    updateMedallion(total);
    renderStats();
  }
  function growIn() {
    const start = performance.now();
    function step() {
      const t = Math.min(1, (performance.now() - start) / 850);
      rootGroup && rootGroup.children.forEach(o => { if (o.userData && o.userData.grow != null) o.scale.setScalar(o.userData.grow * (t * t)); });
      if (t < 1) requestAnimationFrame(step);
    }
    step();
  }

  // ── center medallion (CSS overlay) ────────────────────────────────────────
  function ctxRange() { try { return ((window.APIN.analyticsCtx && window.APIN.analyticsCtx()) || {}).range || ''; } catch (e) { return ''; } }
  function updateMedallion(total) {
    if (!medalEl) return;
    const live = _entries.filter(e => !e.dormant).sort((a, b) => b.count - a.count);
    const dom = live[0];
    medalEl.innerHTML = '<b>' + (total || 0) + '</b><span>scans' + (ctxRange() ? ' · ' + ctxRange() : '') + '</span>' +
      (dom ? '<i>' + dom.roster.label.toLowerCase() + ' leads ' + Math.round(dom.pct) + '%</i>' : '');
  }

  // ── stats block (top-right) ───────────────────────────────────────────────
  function pctOf(n, t) { return t ? Math.round(n / t * 100) : 0; }
  function renderStats() {
    if (!insightHost) return;
    const total = (_data && _data.total) || 0;
    const splitBar = _entries.map(e => e.count > 0
      ? '<i title="' + e.roster.label + ' ' + (e.pct || 0) + '%" style="width:' + (e.pct || 0) + '%;background:' + e.roster.accent + '"></i>' : '').join('');
    const legend = _entries.map(e =>
      '<span class="an-fld-leg' + (e.dormant ? ' dorm' : '') + '" data-crop="' + e.roster.key + '"><i style="background:' + (e.dormant ? '#c7bfad' : e.roster.accent) + '"></i>' +
      e.roster.label + ' <b>' + (e.dormant ? '—' : (e.pct || 0) + '%') + '</b></span>').join('');
    const cropRows = _entries.map(e => {
      if (e.dormant) return '<div class="an-fld-crow dorm" data-crop="' + e.roster.key + '"><span class="an-fld-cnm">' + e.roster.label + '</span><span class="an-fld-cdorm">dormant · no scans yet</span></div>';
      const sev = cropSeverity(e.data); const st = (sev.mild + sev.moderate + sev.severe) || 1;
      const conf = cropConf(e.data);
      return '<div class="an-fld-crow" data-crop="' + e.roster.key + '" tabindex="0">' +
        '<span class="an-fld-cnm"><i style="background:' + e.roster.accent + '"></i>' + e.roster.label + '</span>' +
        '<span class="an-fld-cn">' + e.count + '</span>' +
        '<span class="an-fld-sev" title="mild ' + sev.mild + ' · moderate ' + sev.moderate + ' · severe ' + sev.severe + '">' +
        '<i style="width:' + pctOf(sev.mild, st) + '%;background:#52b788"></i><i style="width:' + pctOf(sev.moderate, st) + '%;background:#c98a2b"></i><i style="width:' + pctOf(sev.severe, st) + '%;background:#b3402f"></i></span>' +
        '<span class="an-fld-conf">' + conf.toFixed(2) + '</span></div>';
    }).join('');
    // top diseases across all crops
    const allDis = [];
    _entries.forEach(e => (e.data.diseases || []).forEach(d => allDis.push({ d: d, crop: e.roster })));
    allDis.sort((a, b) => (b.d.count || 0) - (a.d.count || 0));
    const maxD = Math.max(1, ...allDis.map(x => x.d.count || 0));
    const topDis = allDis.slice(0, 5).map(x =>
      '<button class="an-fld-drow" data-dis="' + x.d.disease + '"><span class="an-fld-dnm">' + (x.d.disease || '').replace(/_/g, ' ') + '</span>' +
      '<span class="an-fld-dbar"><i style="width:' + Math.round((x.d.count || 0) / maxD * 100) + '%;background:' + x.crop.accent + '"></i></span>' +
      '<span class="an-fld-dn">' + (x.d.count || 0) + '</span></button>').join('') || '<div class="an-fld-mut">no diagnoses yet</div>';
    insightHost.innerHTML =
      '<div class="an-fld-stats">' +
      '<div class="an-fld-sh">Field stats</div>' +
      '<div class="an-fld-total"><b>' + total + '</b><span>scans' + (ctxRange() ? ' · ' + ctxRange() : '') + '</span></div>' +
      '<div class="an-fld-splitbar">' + splitBar + '</div>' +
      '<div class="an-fld-legrow">' + legend + '</div>' +
      '<div class="an-fld-sub">crops</div><div class="an-fld-crops">' + cropRows + '</div>' +
      '<div class="an-fld-sub">top diagnoses <small>click → drill</small></div><div class="an-fld-tops">' + topDis + '</div>' +
      '</div>';
    insightHost.querySelectorAll('.an-fld-crow[data-crop],.an-fld-leg[data-crop]').forEach(r =>
      r.addEventListener('click', () => { const k = r.getAttribute('data-crop'); const e = _entries.find(x => x.roster.key === k); if (e && !e.dormant) focusEntry(e); }));
    insightHost.querySelectorAll('.an-fld-drow[data-dis]').forEach(b =>
      b.addEventListener('click', () => { if (_onPickDisease) _onPickDisease(b.getAttribute('data-dis')); }));
  }

  // ── focus a crop ──────────────────────────────────────────────────────────
  function focusEntry(e) {
    if (!e) return; _focusCrop = e.roster.key;
    if (_onPickCrop) _onPickCrop(e.data.crop || e.roster.key);
    // dim the others
    _entries.forEach(x => { (x.plants || []).forEach(p => { p.visible = (x === e); }); if (x.mesh) x.mesh.material.opacity = (x === e ? 1 : 0.5), x.mesh.material.transparent = (x !== e); });
    const cx = (rIn + rOut) / 2 * Math.cos(e.mid), cz = -(rIn + rOut) / 2 * Math.sin(e.mid);
    // dolly in toward the wedge: move the orbit target onto it + pull the radius in
    orbitTo(cx, 0, cz, 2.1, _orbit ? _orbit.theta : 0, 0.95);
  }
  function unfocus() {
    _focusCrop = null;
    _entries.forEach(x => { (x.plants || []).forEach(p => { p.visible = true; }); if (x.mesh) { x.mesh.material.opacity = 1; x.mesh.material.transparent = false; } });
    orbitTo(0, 0, 0, DEF_R, _orbit ? _orbit.theta : 0, DEF_PHI);
  }

  // ── orbit controls (rotate · zoom · pan) ─────────────────────────────────
  let _orbit = null, _orbAnim = null, _drag = null;
  const DEF_R = 3.68, DEF_PHI = 0.83;
  function applyOrbit() {
    if (!cam || !_orbit) return;
    const o = _orbit, sp = Math.sin(o.phi), cp = Math.cos(o.phi);
    cam.position.set(o.target.x + o.radius * sp * Math.sin(o.theta), o.target.y + o.radius * cp, o.target.z + o.radius * sp * Math.cos(o.theta));
    cam.lookAt(o.target.x, o.target.y, o.target.z);
  }
  function orbitTo(tx, ty, tz, radius, theta, phi) {
    if (!_orbit) return;
    _orbAnim = { ft: _orbit.target.clone(), tt: new THREE.Vector3(tx, ty, tz), fr: _orbit.radius, tr: radius,
      fth: _orbit.theta, tth: theta == null ? _orbit.theta : theta, fph: _orbit.phi, tph: phi == null ? _orbit.phi : phi, start: performance.now() };
  }
  function panBy(dx, dy) {
    const o = _orbit;
    const fwd = new THREE.Vector3(o.target.x - cam.position.x, o.target.y - cam.position.y, o.target.z - cam.position.z).normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    const sp = o.radius * 0.0016;
    o.target.x += (-dx * sp) * right.x + (dy * sp) * up.x;
    o.target.y += (-dx * sp) * right.y + (dy * sp) * up.y;
    o.target.z += (-dx * sp) * right.z + (dy * sp) * up.z;
  }
  function onDown(ev) {
    _orbAnim = null;
    const pan = (ev.button === 2) || ev.shiftKey || ev.metaKey || ev.ctrlKey;
    _drag = { mode: pan ? 'pan' : 'rotate', x: ev.clientX, y: ev.clientY, moved: 0 };
    host.style.cursor = 'grabbing';
    try { renderer.domElement.setPointerCapture(ev.pointerId); } catch (e) { }
  }
  function onPointerMove(ev) {
    if (_drag) {
      const dx = ev.clientX - _drag.x, dy = ev.clientY - _drag.y; _drag.x = ev.clientX; _drag.y = ev.clientY; _drag.moved += Math.abs(dx) + Math.abs(dy);
      hideTip();
      if (_drag.mode === 'rotate') { _orbit.theta -= dx * 0.006; _orbit.phi = Math.max(0.12, Math.min(1.52, _orbit.phi - dy * 0.006)); }
      else panBy(dx, dy);
      applyOrbit(); return;
    }
    onHover(ev);
  }
  function onUp(ev) {
    if (!_drag) return;
    const click = _drag.moved < 6; _drag = null; host.style.cursor = 'grab';
    try { renderer.domElement.releasePointerCapture(ev.pointerId); } catch (e) { }
    if (click) doClick(ev);
  }
  function onWheel(ev) {
    ev.preventDefault(); _orbAnim = null;
    _orbit.radius = Math.max(1.5, Math.min(9, _orbit.radius * (1 + (ev.deltaY > 0 ? 1 : -1) * 0.09)));
    applyOrbit();
  }
  function projectMedallion() {
    if (!medalEl || !cam || !renderer) return;
    const p = new THREE.Vector3(0, 0.06, 0).project(cam);
    if (p.z > 1) { medalEl.style.display = 'none'; return; }
    medalEl.style.display = 'block';
    medalEl.style.left = ((p.x * 0.5 + 0.5) * (host.clientWidth || 1)) + 'px';
    medalEl.style.top = ((-p.y * 0.5 + 0.5) * (host.clientHeight || 1)) + 'px';
  }

  // ── tooltip ───────────────────────────────────────────────────────────────
  let _tip = null;
  function tipEl() { if (!_tip) { _tip = document.createElement('div'); _tip.className = 'an-ftip'; _tip.style.display = 'none'; document.body.appendChild(_tip); } return _tip; }
  function showTip(x, y, html) { const t = tipEl(); t.innerHTML = html; t.style.display = 'block'; t.style.left = Math.min(x + 14, window.innerWidth - 220) + 'px'; t.style.top = (y + 14) + 'px'; }
  function hideTip() { if (_tip) _tip.style.display = 'none'; }

  // ── mount ─────────────────────────────────────────────────────────────────
  function mount(hostEl, opts) {
    host = hostEl; opts = opts || {};
    insightHost = opts.insightHost || null;
    _onPickDisease = opts.onPickDisease || null; _onPickCrop = opts.onPickCrop || null;
    THREE = window.THREE;
    injectCSS();
    if (!hostEl || !THREE) { if (hostEl) hostEl.innerHTML = '<div class="an-ph an-ph-empty">field unavailable</div>'; return; }
    hostEl.innerHTML = ''; hostEl.classList.add('an-fld-host');
    scene = new THREE.Scene();
    const w = hostEl.clientWidth || 600, h = hostEl.clientHeight || 320;
    cam = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    _orbit = { theta: 0, phi: DEF_PHI, radius: DEF_R, target: new THREE.Vector3(0, 0, 0) };
    applyOrbit();
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(w, h); renderer.setClearColor(0x000000, 0);
    hostEl.appendChild(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xfff6e6, 0x6b5b43, 0.95));
    const dir = new THREE.DirectionalLight(0xffffff, 0.55); dir.position.set(2, 4, 2); scene.add(dir);
    // soft ground shadow disc under the donut
    ground = new THREE.Mesh(new THREE.CircleGeometry(rOut + 0.2, 48), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.05 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.001; scene.add(ground);
    raycaster = new THREE.Raycaster(); mouse = new THREE.Vector2();
    const dom = renderer.domElement; dom.style.touchAction = 'none';
    dom.addEventListener('pointerdown', onDown);
    dom.addEventListener('pointermove', onPointerMove);
    dom.addEventListener('pointerup', onUp);
    dom.addEventListener('pointerleave', () => { hideTip(); if (!_drag) host.style.cursor = 'grab'; });
    dom.addEventListener('wheel', onWheel, { passive: false });
    dom.addEventListener('contextmenu', e => e.preventDefault());
    host.style.cursor = 'grab';
    // center medallion overlay
    medalEl = document.createElement('div'); medalEl.className = 'an-fld-medal'; hostEl.appendChild(medalEl);
    ro = new ResizeObserver(resize); ro.observe(hostEl);
    if (_data) buildDonut();
    loop();
  }
  function resize() { if (!renderer || !host) return; const w = host.clientWidth, h = host.clientHeight; if (w < 40 || h < 40) return; cam.aspect = w / h; cam.updateProjectionMatrix(); renderer.setSize(w, h); }

  // ── picking ───────────────────────────────────────────────────────────────
  function pick(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, cam);
    const hits = raycaster.intersectObjects(rootGroup ? rootGroup.children : [], true);
    for (const hit of hits) {
      let o = hit.object;
      while (o && !(o.userData && (o.userData.plant || o.userData.wedge || o.userData.center))) o = o.parent;
      if (o && o.userData) { if (o.userData.center) return null; return o; }
    }
    return null;
  }
  function onHover(ev) {
    const o = pick(ev);
    // restore previous hover emphasis
    if (_hoverPlant && (!o || o !== _hoverPlant)) { _hoverPlant.scale.setScalar(_hoverPlant.userData.base); _hoverPlant = null; }
    if (_hover && (!o || o !== _hover) && _hover.userData && _hover.userData.wedge) { try { _hover.material.emissive && _hover.material.emissive.setHex(0x000000); } catch (e) { } _hover = null; }
    if (!o) { hideTip(); host.style.cursor = 'grab'; return; }
    host.style.cursor = 'pointer';
    if (o.userData.plant) {
      _hoverPlant = o; o.scale.setScalar(o.userData.base * 1.35);
      const u = o.userData;
      showTip(ev.clientX, ev.clientY, '<b>' + (u.disease || '').replace(/_/g, ' ') + '</b><br>' + u.entry.roster.label + ' · ' + u.severity +
        (u.confidence ? '<br>conf ' + (u.confidence).toFixed(2) : '') + '<br><i>click → drill this diagnosis</i>');
    } else if (o.userData.wedge) {
      _hover = o; try { o.material.emissive && o.material.emissive.setHex(0x2a2419); } catch (e) { }
      const e = o.userData.entry;
      if (e.dormant) showTip(ev.clientX, ev.clientY, '<b>' + e.roster.label + '</b><br>dormant · no scans this window<br><i>supported — awaiting first scan</i>');
      else { const top = (e.data.diseases || [])[0]; showTip(ev.clientX, ev.clientY, '<b>' + e.roster.label + '</b><br>' + e.count + ' scans · ' + (e.pct || 0) + '%' + (top ? '<br>top: ' + (top.disease || '').replace(/_/g, ' ') + ' (' + top.count + ')' : '') + '<br><i>click → focus this crop</i>'); }
    }
  }
  function doClick(ev) {
    const o = pick(ev);
    if (!o) { if (_focusCrop) unfocus(); return; }
    if (o.userData.plant) { hideTip(); if (_onPickDisease) _onPickDisease(o.userData.disease); }
    else if (o.userData.wedge) { const e = o.userData.entry; if (!e.dormant) focusEntry(e); }
  }

  // ── loop ──────────────────────────────────────────────────────────────────
  function loop() {
    raf = requestAnimationFrame(loop);
    if (_paused || !renderer) return;
    const now = performance.now();
    if (rootGroup) rootGroup.children.forEach((o, i) => { if (o.userData && o.userData.plant && o.visible) o.rotation.x = Math.sin(now / 900 + i) * 0.04; });
    if (_orbAnim) {
      const t = Math.min(1, (now - _orbAnim.start) / 900), e = t * t * (3 - 2 * t);
      _orbit.target.lerpVectors(_orbAnim.ft, _orbAnim.tt, e);
      _orbit.radius = _orbAnim.fr + (_orbAnim.tr - _orbAnim.fr) * e;
      _orbit.theta = _orbAnim.fth + (_orbAnim.tth - _orbAnim.fth) * e;
      _orbit.phi = _orbAnim.fph + (_orbAnim.tph - _orbAnim.fph) * e;
      if (t >= 1) _orbAnim = null;
    }
    applyOrbit();
    projectMedallion();
    renderer.render(scene, cam);
  }

  // ── public API ────────────────────────────────────────────────────────────
  function setData(d) { _data = d; if (scene) buildDonut(); }
  function setMode(m) { _mode = m; if (m === 'mosaic') orbitTo(0, 0, 0, 3.2, _orbit ? _orbit.theta : 0, 0.12); else { _focusCrop = null; unfocus(); } }
  function focusDisease(dis) {
    const e = _entries.find(x => (x.data.diseases || []).some(d => d.disease === dis));
    if (e && !e.dormant) focusEntry(e);
  }
  function onScan(e) {
    if (!e) return; const key = (e.crop || (e.diagnosis || '').split('_')[0] || '').toLowerCase();
    const ent = _entries.find(x => x.roster.key === key);
    if (ent && ent.mesh) { ent.mesh.scale.setScalar(1.05); setTimeout(() => ent.mesh && ent.mesh.scale.setScalar(1), 240); }
    // forward to the Observatory (if the expanded world is open) for seed→sprout
    if (window.APIN && window.APIN.analyticsObservatory && window.APIN.analyticsObservatory.onScan) window.APIN.analyticsObservatory.onScan(e);
  }
  function renderExpanded(el, d) {
    // The expanded state is the heavy "Living Agricultural Observatory" — a
    // separate module lazy-loaded ONLY when the field is first expanded, so the
    // ~25KB world never touches the initial analytics load (production speed).
    const launch = () => { try { window.APIN.analyticsObservatory.open(el, { data: d }); } catch (e) { el.innerHTML = '<div class="an-ph an-ph-empty">observatory failed to start</div>'; } };
    if (window.APIN && window.APIN.analyticsObservatory) { launch(); return; }
    el.innerHTML = '<div class="an-ph">loading observatory&hellip;</div>';
    const s = document.createElement('script'); s.src = '/static/console_analytics_observatory.js?v=9t32a'; s.async = true;
    s.onload = () => { if (window.APIN && window.APIN.analyticsObservatory) launch(); else el.innerHTML = '<div class="an-ph an-ph-empty">observatory unavailable</div>'; };
    s.onerror = () => { el.innerHTML = '<div class="an-ph an-ph-empty">observatory unavailable</div>'; };
    document.head.appendChild(s);
  }
  function pause() { _paused = true; }
  function resume() { _paused = false; resize(); }
  function setRange() { }

  // ── injected CSS ──────────────────────────────────────────────────────────
  function injectCSS() {
    if (document.getElementById('an-fld-css')) return;
    const s = document.createElement('style'); s.id = 'an-fld-css';
    s.textContent = `
.an-fld-host{position:relative;background:#f5efe0;background-image:radial-gradient(circle,rgba(26,22,18,.16) 1.1px,transparent 1.1px);background-size:20px 20px;border-radius:10px;overflow:hidden}
.an-fld-medal{position:absolute;left:50%;top:52%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;font-family:'JetBrains Mono',monospace;z-index:2}
.an-fld-medal b{display:block;font:700 26px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);line-height:1}
.an-fld-medal span{display:block;font-size:9px;color:var(--ink-mute,#9a917d);margin-top:1px}
.an-fld-medal i{display:block;font-style:normal;font-size:9px;color:var(--ink-soft,#5b5446);margin-top:3px}
.an-ftip{position:fixed;z-index:9999;pointer-events:none;max-width:210px;background:var(--ink,#1a1612);color:#efe7d4;font:11px 'JetBrains Mono',monospace;line-height:1.45;padding:7px 9px;border-radius:7px;box-shadow:0 4px 16px rgba(0,0,0,.3)}
.an-ftip b{color:#8fe0b4}.an-ftip i{color:#b9b09c;font-style:normal}
/* stats block (top-right) */
.an-fld-stats{font-family:'JetBrains Mono',monospace}
.an-fld-sh{font:600 13px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin-bottom:8px}
.an-fld-total{display:flex;align-items:baseline;gap:6px;margin-bottom:8px}
.an-fld-total b{font:700 26px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.an-fld-total span{font-size:10px;color:var(--ink-mute,#9a917d)}
.an-fld-splitbar{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--paper-deep,#e7dcc4);margin-bottom:6px}
.an-fld-splitbar i{display:block;height:100%}
.an-fld-legrow{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.an-fld-leg{font-size:10px;color:var(--ink-soft,#5b5446);cursor:pointer;display:inline-flex;align-items:center;gap:4px}
.an-fld-leg.dorm{opacity:.55;cursor:default}
.an-fld-leg i{width:8px;height:8px;border-radius:2px;display:inline-block}
.an-fld-leg b{color:var(--ink,#1a1612)}
.an-fld-sub{font:600 9px 'JetBrains Mono',monospace;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-mute,#9a917d);margin:10px 0 5px}
.an-fld-sub small{text-transform:none;letter-spacing:0;font-weight:400;opacity:.8}
.an-fld-crow{display:grid;grid-template-columns:1fr 28px 46px 34px;align-items:center;gap:7px;padding:4px 5px;margin:0 -5px;border-radius:6px;cursor:pointer;font-size:11px;transition:background .12s}
.an-fld-crow:hover,.an-fld-crow:focus{background:var(--paper,#efe7d4);outline:none}
.an-fld-crow.dorm{cursor:default;opacity:.6;grid-template-columns:1fr auto}
.an-fld-cnm{display:flex;align-items:center;gap:5px;color:var(--ink,#1a1612);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.an-fld-cnm i{width:8px;height:8px;border-radius:2px;flex:none}
.an-fld-cn{text-align:right;font-weight:600;color:var(--ink-soft,#5b5446)}
.an-fld-cdorm{font-size:9.5px;color:var(--ink-mute,#9a917d);text-align:right}
.an-fld-sev{display:flex;height:7px;border-radius:4px;overflow:hidden;background:var(--paper-deep,#e7dcc4)}
.an-fld-sev i{display:block;height:100%}
.an-fld-conf{text-align:right;color:var(--ink-mute,#9a917d)}
.an-fld-drow{display:grid;grid-template-columns:1fr 56px 28px;align-items:center;gap:7px;width:100%;padding:4px 5px;margin:0 -5px;border:0;background:none;border-radius:6px;cursor:pointer;font:11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);transition:background .12s}
.an-fld-drow:hover{background:var(--paper,#efe7d4)}
.an-fld-dnm{text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--ink-soft,#5b5446)}
.an-fld-dbar{height:7px;background:var(--paper-deep,#e7dcc4);border-radius:4px;overflow:hidden}
.an-fld-dbar i{display:block;height:100%;border-radius:4px}
.an-fld-dn{text-align:right;font-weight:600;color:var(--ink-soft,#5b5446)}
.an-fld-mut{font-size:10.5px;color:var(--ink-mute,#9a917d)}
/* expanded layout */
.an-fld-exp{display:grid;grid-template-columns:1fr 300px;gap:14px;height:70vh;min-height:440px}
.an-fld-exp-canvas{height:100%;min-height:300px}
.an-fld-exp-stats{background:var(--paper-deep,#e7dcc4);border:1px solid var(--paper-edge,#d8cdb2);border-radius:10px;padding:14px;overflow:auto}
@media (max-width:760px){.an-fld-exp{grid-template-columns:1fr;height:auto}.an-fld-exp-canvas{height:46vh}}
`;
    document.head.appendChild(s);
  }

  F.mount = mount; F.setData = setData; F.setMode = setMode; F.setRange = setRange;
  F.focusDisease = focusDisease; F.onScan = onScan; F.renderExpanded = renderExpanded;
  F.pause = pause; F.resume = resume;
  window.APIN = window.APIN || {};
  window.APIN.analyticsField = F;
})();
