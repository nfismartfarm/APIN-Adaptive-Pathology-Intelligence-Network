/* ADM-T · Living Globe — Geography section.
 *
 * A hand-built Three.js globe (vendored r149 UMD THREE). Real country borders
 * are projected from geo_world_countries.json (Natural Earth, 177 features);
 * real scan-origin districts (admin_geo → genuine lat/lon) are plotted as glowing
 * markers sized by scan count and coloured by dominant crop. Everything on this
 * globe is real geometry — no fabricated points. API request IPs are private/
 * local and are NOT plotted (the Traffic section states that truth instead).
 *
 * window.ADM_GLOBE.create(canvas, geo, opts) → { setData, resize, dispose } or
 * null when WebGL is unavailable (caller keeps a 2D fallback list).
 */
(function () {
  'use strict';
  var THREE = window.THREE;

  var CROP_COL = { okra: 0x4ade80, brassica: 0xe0b341, tomato: 0xe0584a, other: 0x7da7ff };
  var R = 2;                         // globe radius
  var _bordersCache = null;          // parsed country-border segments (fetched once)
  var _statesCache = null;           // parsed state/province-border segments
  var _haloTexCache = null;          // soft radial sprite texture (shared)

  // soft circular glow texture — without this, Sprites render as hard squares
  function haloTex() {
    if (_haloTexCache) return _haloTexCache;
    var c = document.createElement('canvas'); c.width = c.height = 64;
    var ctx = c.getContext('2d');
    var g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.35, 'rgba(255,255,255,0.55)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(32, 32, 32, 0, 6.2832); ctx.fill();
    _haloTexCache = new THREE.CanvasTexture(c);
    return _haloTexCache;
  }

  function llToVec(lat, lon, r) {
    var phi = (90 - lat) * Math.PI / 180, theta = (lon + 180) * Math.PI / 180;
    return new THREE.Vector3(
      -r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta)
    );
  }

  // GeoJSON FeatureCollection → flat Float32Array of border line segments
  function bordersToSegments(geojson) {
    var seg = [];
    function ring(coords) {
      for (var i = 0; i < coords.length - 1; i++) {
        var a = llToVec(coords[i][1], coords[i][0], R * 1.002);
        var b = llToVec(coords[i + 1][1], coords[i + 1][0], R * 1.002);
        seg.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }
    }
    (geojson.features || []).forEach(function (f) {
      var g = f.geometry; if (!g) return;
      if (g.type === 'Polygon') g.coordinates.forEach(ring);
      else if (g.type === 'MultiPolygon') g.coordinates.forEach(function (poly) { poly.forEach(ring); });
    });
    return new Float32Array(seg);
  }

  function create(canvas, geo, opts) {
    if (!THREE || !canvas) return null;
    opts = opts || {};
    var renderer;
    try { renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true, preserveDrawingBuffer: true }); }
    catch (e) { return null; }
    if (!renderer || !renderer.getContext()) return null;
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    var W = opts.W || canvas.clientWidth || 700, H = opts.H || canvas.clientHeight || 420;
    renderer.setSize(W, H, false); renderer.setClearColor(0x000000, 0);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(38, W / H, 0.1, 100);
    camera.position.set(0, 1.4, 6.2);
    camera.lookAt(0, 0, 0);
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    var key = new THREE.DirectionalLight(0xbfe8d0, 0.7); key.position.set(5, 3, 5); scene.add(key);

    var world = new THREE.Group(); scene.add(world);

    // ocean sphere — deep teal with a faint fresnel-ish rim
    var ocean = new THREE.Mesh(
      new THREE.SphereGeometry(R, 64, 64),
      new THREE.MeshPhongMaterial({ color: 0x0c1a18, emissive: 0x07110f, shininess: 8, specular: 0x123 })
    );
    world.add(ocean);
    // atmosphere halo (slightly larger, back-side, additive)
    var atmo = new THREE.Mesh(
      new THREE.SphereGeometry(R * 1.06, 48, 48),
      new THREE.MeshBasicMaterial({ color: 0x2e6a4f, transparent: true, opacity: 0.10, side: THREE.BackSide, blending: THREE.AdditiveBlending })
    );
    world.add(atmo);

    var bordersMesh = null, statesMesh = null;
    function addBorders(segs) {
      if (bordersMesh) { world.remove(bordersMesh); bordersMesh.geometry.dispose(); }
      var g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(segs, 3));
      bordersMesh = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x3a6f5a, transparent: true, opacity: 0.6 }));
      world.add(bordersMesh);
    }
    function addStates(segs) {
      if (statesMesh) { world.remove(statesMesh); statesMesh.geometry.dispose(); }
      var g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(segs, 3));
      // fainter + thinner than country borders so the hierarchy reads clearly
      statesMesh = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x2c5446, transparent: true, opacity: 0.32 }));
      world.add(statesMesh);
    }

    // ── markers (real districts) ───────────────────────────────────────────
    var markers = [], markerGroup = new THREE.Group(); world.add(markerGroup);
    function clearMarkers() {
      markers.forEach(function (m) { markerGroup.remove(m.dot); markerGroup.remove(m.halo); markerGroup.remove(m.spike); });
      markers = [];
    }
    function buildMarkers(districts) {
      clearMarkers();
      var maxN = districts.reduce(function (a, d) { return Math.max(a, d.count || 0); }, 1);
      var tex = haloTex();
      districts.forEach(function (d) {
        if (d.lat == null || d.lon == null) return;
        var col = opts.colorOf ? opts.colorOf(d) : (CROP_COL[d.crop] || CROP_COL.other);
        var frac = Math.sqrt((d.count || 0) / maxN);
        var sz = 0.009 + 0.022 * frac;     // smaller, tighter dots
        var pos = llToVec(d.lat, d.lon, R * 1.01);
        var dot = new THREE.Mesh(new THREE.SphereGeometry(sz, 14, 14),
          new THREE.MeshBasicMaterial({ color: col }));
        dot.position.copy(pos); dot.userData = d; markerGroup.add(dot);
        // soft circular halo (CanvasTexture → no square edge)
        var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: col, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true }));
        halo.position.copy(pos); var base = sz * 4.5; halo.scale.setScalar(base); markerGroup.add(halo);
        // thin spike from surface
        var tip = llToVec(d.lat, d.lon, R * (1.01 + 0.08 * frac));
        var sg = new THREE.BufferGeometry().setFromPoints([llToVec(d.lat, d.lon, R), tip]);
        var spike = new THREE.Line(sg, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.45 }));
        markerGroup.add(spike);
        markers.push({ dot: dot, halo: halo, spike: spike, base: base, d: d });
      });
    }

    function setData(geo2) {
      buildMarkers((geo2 && geo2.districts) || []);
      // recentre on the densest cluster (India) for a meaningful first frame
      var ds = (geo2 && geo2.districts) || [];
      if (ds.length) {
        var c = ds[0];
        targetYaw = -(c.lon + 180) * Math.PI / 180 + Math.PI / 2;
        targetPitch = (c.lat) * Math.PI / 180 * 0.7;
      }
    }

    // ── interaction ─────────────────────────────────────────────────────────
    var ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    var dragging = false, lx = 0, ly = 0, yaw = 0.4, pitch = 0.2, targetYaw = null, targetPitch = null;
    var autorot = true, raf = null, disposed = false, t = 0;

    function pickMarker(px, py) {
      var rect = canvas.getBoundingClientRect();
      ndc.x = ((px - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((py - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      var hit = ray.intersectObjects(markers.map(function (m) { return m.dot; }), false)[0];
      if (!hit) return null;
      // occlusion: ignore markers behind the globe (closer ocean hit)
      var oc = ray.intersectObject(ocean, false)[0];
      if (oc && oc.distance < hit.distance - 0.02) return null;
      return hit.object.userData;
    }
    function onMove(e) {
      if (dragging) { yaw += (e.clientX - lx) * 0.006; pitch += (e.clientY - ly) * 0.006; pitch = Math.max(-1.2, Math.min(1.2, pitch)); lx = e.clientX; ly = e.clientY; autorot = false; targetYaw = null; return; }
      var d = pickMarker(e.clientX, e.clientY);
      if (d) { canvas.style.cursor = 'pointer'; if (opts.onHover) opts.onHover(d, e.clientX, e.clientY); }
      else { canvas.style.cursor = 'grab'; if (opts.onLeave) opts.onLeave(); }
    }
    function onDown(e) { dragging = true; lx = e.clientX; ly = e.clientY; canvas.style.cursor = 'grabbing'; }
    function onUp(e) {
      if (dragging && Math.abs(e.clientX - lx) < 3 && Math.abs(e.clientY - ly) < 3) {
        var d = pickMarker(e.clientX, e.clientY); if (d && opts.onClick) opts.onClick(d);
      }
      dragging = false; canvas.style.cursor = 'grab';
    }
    function onLeave() { dragging = false; if (opts.onLeave) opts.onLeave(); }
    function onWheel(e) { e.preventDefault(); camera.position.z = Math.max(3.6, Math.min(9, camera.position.z + (e.deltaY > 0 ? 0.4 : -0.4))); }
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointerleave', onLeave);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    function frame() {
      if (disposed) return;
      raf = requestAnimationFrame(frame);
      if (canvas.offsetParent === null || document.hidden) return;
      t += 0.016;
      if (targetYaw != null) { yaw += (targetYaw - yaw) * 0.06; pitch += (targetPitch - pitch) * 0.06; if (Math.abs(targetYaw - yaw) < 0.002) targetYaw = null; }
      else if (autorot) yaw += 0.0014;
      world.rotation.y = yaw; world.rotation.x = pitch;
      // pulse halos
      for (var i = 0; i < markers.length; i++) { var s = markers[i].base * (1 + 0.12 * Math.sin(t * 2 + i)); markers[i].halo.scale.setScalar(s); }
      renderer.render(scene, camera);
    }

    // fetch borders once (cached across instances), then go
    function boot(geoData) {
      if (_bordersCache) addBorders(_bordersCache);
      else {
        fetch('/static/geo_world_countries.json').then(function (r) { return r.json(); })
          .then(function (j) { _bordersCache = bordersToSegments(j); if (!disposed) addBorders(_bordersCache); })
          .catch(function () {/* borders optional — markers still render */ });
      }
      if (_statesCache) addStates(_statesCache);
      else {
        fetch('/static/geo_world_states.json').then(function (r) { return r.json(); })
          .then(function (j) { _statesCache = bordersToSegments(j); if (!disposed) addStates(_statesCache); })
          .catch(function () {/* state borders optional */ });
      }
      setData(geoData);
      frame();
    }
    boot(geo);

    return {
      setData: setData,
      resize: function (w, h) { W = w || canvas.clientWidth; H = h || canvas.clientHeight; renderer.setSize(W, H, false); camera.aspect = W / H; camera.updateProjectionMatrix(); },
      dispose: function () {
        disposed = true; if (raf) cancelAnimationFrame(raf);
        canvas.removeEventListener('pointermove', onMove); canvas.removeEventListener('pointerdown', onDown);
        window.removeEventListener('pointerup', onUp); canvas.removeEventListener('pointerleave', onLeave);
        canvas.removeEventListener('wheel', onWheel);
        try { renderer.dispose(); } catch (e) {}
      },
    };
  }

  window.ADM_GLOBE = { create: create, ok: !!THREE, CROP_COL: CROP_COL };
})();
