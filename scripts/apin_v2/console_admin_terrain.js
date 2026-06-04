/* ADM-T · Traffic Terrain — WebGL 3D hero for the Traffic › API tab.
 *
 * Renders the 7-day × 24-hour request matrix (traffic_api_terrain) as a 3D
 * heightfield: vertex height ∝ request volume, colour ∝ error rate (jade →
 * gold → ember). Grows in from a flat plane, auto-orbits, drag to rotate,
 * raycast hover → cell tooltip. Uses the vendored THREE (r149 UMD global).
 *
 * Exposes window.ADM_TERRAIN.create(canvas, data, opts) → { setData, resize,
 * dispose } or null when WebGL is unavailable (caller keeps a 2D fallback).
 */
(function () {
  'use strict';
  var THREE = window.THREE;

  var JADE = [0.16, 0.62, 0.42], GOLD = [0.82, 0.62, 0.20], EMBER = [0.80, 0.30, 0.22];
  function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function ramp(err) {
    var t = Math.min(1, (err || 0) / 0.25);
    return t <= 0.5 ? lerp3(JADE, GOLD, t / 0.5) : lerp3(GOLD, EMBER, (t - 0.5) / 0.5);
  }

  var HOURS = 24;
  var PLANE_W = 17, PLANE_D = 7, MAX_H = 3.4;

  function create(canvas, data, opts) {
    if (!THREE || !canvas) return null;
    opts = opts || {};
    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    } catch (e) { return null; }
    if (!renderer || !renderer.getContext()) return null;
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

    var W = opts.W || canvas.clientWidth || 800;
    var H = opts.H || canvas.clientHeight || 340;
    renderer.setSize(W, H, false);
    renderer.setClearColor(0x000000, 0);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 200);
    scene.add(new THREE.AmbientLight(0xffffff, 0.58));
    var key = new THREE.DirectionalLight(0xfff1da, 1.15); key.position.set(7, 14, 7); scene.add(key);
    var rim = new THREE.DirectionalLight(0x4ade80, 0.40); rim.position.set(-8, 6, -7); scene.add(rim);

    var grid, days, maxN, geo, mesh, pos, colors, targetH, meta, cellN, rows;
    var _settling = false;   // live bumps lerp vertices toward targetH

    function _recolor(i) {
      var n = cellN[i], hN = n / maxN, err = n > 0 ? (meta[i].e || 0) / n : 0;
      var c = ramp(err), lum = 0.34 + 0.66 * hN;
      colors[i * 3] = c[0] * lum; colors[i * 3 + 1] = c[1] * lum; colors[i * 3 + 2] = c[2] * lum;
    }

    function build(d) {
      days = (d && d.days) || [];
      grid = (d && d.grid) || [];
      rows = Math.max(1, days.length || grid.length || 7);
      maxN = 1;
      var i, j;
      for (i = 0; i < grid.length; i++) for (j = 0; j < (grid[i] || []).length; j++) maxN = Math.max(maxN, (grid[i][j] || {}).n || 0);

      if (geo) { scene.remove(mesh); geo.dispose(); }
      geo = new THREE.PlaneGeometry(PLANE_W, PLANE_D, HOURS - 1, rows - 1);
      geo.rotateX(-Math.PI / 2);
      pos = geo.attributes.position;
      colors = new Float32Array(pos.count * 3);
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      targetH = new Float32Array(pos.count);
      cellN = new Float32Array(pos.count);
      meta = new Array(pos.count);
      for (i = 0; i < pos.count; i++) {
        var col = i % HOURS;
        var row = Math.floor(i / HOURS);
        var cell = (grid[row] && grid[row][col]) || { n: 0, e: 0, lat: 0 };
        var n = cell.n || 0;
        cellN[i] = n;
        targetH[i] = (n / maxN) * MAX_H;
        var err = n > 0 ? (cell.e || 0) / n : 0;
        meta[i] = { day: days[row] || '', hour: col, n: n, e: cell.e || 0, lat: cell.lat || 0, err: err };
        _recolor(i);
        pos.setY(i, 0); // start flat → grows in
      }
      pos.needsUpdate = true;
      var mat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.82, metalness: 0.06, side: THREE.DoubleSide });
      mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      // frame the camera to the plane
      camera.position.set(0, 8.5, 12.5);
      camera.lookAt(0, 1.1, 0);
      _grow = 0;
    }

    // ── interaction: auto-orbit + drag, raycast hover ──────────────────────
    var ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    var dragging = false, lastX = 0, yaw = 0.5, autorot = true, _grow = 0, raf = null, disposed = false;

    function pointToCell(px, py) {
      var rect = canvas.getBoundingClientRect();
      ndc.x = ((px - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((py - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      var hit = ray.intersectObject(mesh, false)[0];
      if (!hit) return null;
      // invert local X,Z → hour, day
      var lp = mesh.worldToLocal(hit.point.clone());
      var hour = Math.round((lp.x + PLANE_W / 2) / PLANE_W * (HOURS - 1));
      var rows = days.length || 7;
      var day = Math.round((lp.z + PLANE_D / 2) / PLANE_D * (rows - 1));
      hour = Math.max(0, Math.min(HOURS - 1, hour));
      day = Math.max(0, Math.min(rows - 1, day));
      var cell = (grid[day] && grid[day][hour]) || { n: 0, e: 0, lat: 0 };
      return { day: days[day] || '', hour: hour, n: cell.n || 0, e: cell.e || 0, lat: cell.lat || 0 };
    }

    function onMove(e) {
      if (dragging) { yaw += (e.clientX - lastX) * 0.01; lastX = e.clientX; autorot = false; return; }
      var c = pointToCell(e.clientX, e.clientY);
      if (c && c.n > 0) { canvas.style.cursor = 'pointer'; if (opts.onHover) opts.onHover(c, e.clientX, e.clientY); }
      else { canvas.style.cursor = 'grab'; if (opts.onLeave) opts.onLeave(); }
    }
    function onDown(e) { dragging = true; lastX = e.clientX; canvas.style.cursor = 'grabbing'; }
    function onUp() { dragging = false; canvas.style.cursor = 'grab'; }
    function onLeave() { dragging = false; if (opts.onLeave) opts.onLeave(); }
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointerleave', onLeave);

    function frame() {
      if (disposed) return;
      raf = requestAnimationFrame(frame);
      // pause when removed from layout (e.g. Website tab). The perpetual idle
      // auto-rotate also pauses while the tab is backgrounded — BUT the one-time
      // grow-in always completes so the hero is painted (and frame persists via
      // preserveDrawingBuffer) the moment the tab is shown again.
      if (canvas.offsetParent === null) return;
      if (document.hidden && _grow >= 1 && !_settling && !dragging) return;
      if (autorot) yaw += 0.0016;
      // grow-in: lerp vertex Y toward target (global cinematic ease)
      if (_grow < 1) {
        _grow = Math.min(1, _grow + 0.018);
        var e = 1 - Math.pow(1 - _grow, 3);
        for (var i = 0; i < pos.count; i++) pos.setY(i, targetH[i] * e);
        pos.needsUpdate = true;
        geo.computeVertexNormals();
      } else if (_settling) {
        // live mode: ease every vertex toward its (possibly bumped) target
        var done = true, k;
        for (k = 0; k < pos.count; k++) {
          var cur = pos.getY(k), d = targetH[k] - cur;
          if (Math.abs(d) > 0.002) { pos.setY(k, cur + d * 0.16); done = false; }
          else if (cur !== targetH[k]) pos.setY(k, targetH[k]);
        }
        pos.needsUpdate = true;
        geo.computeVertexNormals();
        if (done) _settling = false;
      }
      mesh.rotation.y = yaw;
      renderer.render(scene, camera);
    }

    build(data);
    frame();

    return {
      setData: function (d) { build(d); },
      // Live bump: a request just landed in `hour` (today = last day row). Raise
      // that cell's height + recolour, animating via the settling lerp. If it
      // overtakes maxN, renormalise the whole field so heights stay comparable.
      bump: function (hour, isErr) {
        if (!cellN || _grow < 1) return;
        var h = Math.max(0, Math.min(HOURS - 1, hour | 0));
        var i = (rows - 1) * HOURS + h;
        if (i < 0 || i >= cellN.length) return;
        cellN[i] += 1; meta[i].n = cellN[i]; if (isErr) meta[i].e = (meta[i].e || 0) + 1;
        if (cellN[i] > maxN) { maxN = cellN[i]; for (var k = 0; k < cellN.length; k++) { targetH[k] = (cellN[k] / maxN) * MAX_H; _recolor(k); } }
        else { targetH[i] = (cellN[i] / maxN) * MAX_H; _recolor(i); }
        if (colors) geo.attributes.color.needsUpdate = true;
        _settling = true;
      },
      resize: function (w, h) {
        W = w || canvas.clientWidth; H = h || canvas.clientHeight;
        renderer.setSize(W, H, false); camera.aspect = W / H; camera.updateProjectionMatrix();
      },
      dispose: function () {
        disposed = true;
        if (raf) cancelAnimationFrame(raf);
        canvas.removeEventListener('pointermove', onMove);
        canvas.removeEventListener('pointerdown', onDown);
        window.removeEventListener('pointerup', onUp);
        canvas.removeEventListener('pointerleave', onLeave);
        try { if (geo) geo.dispose(); if (mesh) mesh.material.dispose(); renderer.dispose(); } catch (e) {}
      },
    };
  }

  window.ADM_TERRAIN = { create: create, ok: !!THREE };
})();
