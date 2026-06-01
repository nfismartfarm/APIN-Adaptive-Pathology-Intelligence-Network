/* 9.N.T31 · Geographic origin widget — MapLibre GL JS (vector-tile map).
 *
 * Compact card: a paper-ink globe (sea-glass ocean) with request/scan origins
 * as a heatmap when zoomed out and animated "sonar" dots when zoomed in. A
 * bottom-right preview box flips between the 3D globe and a flat 2D map while
 * keeping zoom/pan/all layers intact. Google-Maps-style zoom LOD: country
 * borders far out, India district borders as you zoom in (viewport-culled).
 *
 * Expanded console (renderExpanded): a full geographic observatory —
 *   B · region dossier  (trend, endpoints, errors, latency, clients, first-seen)
 *   C · metric overlay   (volume / errors / latency / OOD recolor + scale + insight)
 *   A · replay scrubber  (play/pause, 1–8×, per-frame pulses, day/night terminator)
 *   D · anomaly pins     (surge / drop / new — from compare windows)
 *   E · flight arcs      (origin → API edge, animated dashes)
 *   G · compare windows  (this vs previous, per-region delta)
 *
 * Backend (auth_db.compute_analytics_geo*) is unchanged in shape; the per-region
 * metrics (error_rate / avg_latency / ood_rate / delta_pct) and the
 * /geo/region + /geo/replay endpoints power the expanded views.
 */
(function () {
  'use strict';
  const G = {};

  // ── palette / config ──────────────────────────────────────────────────
  const PAPER = '#efe7d4';     // land / paper
  const SEA = '#b8d4cf';       // sea-glass ocean (vibrant, in-language)
  const COAST = 'rgba(26,22,18,0.50)';
  const INK = '#1a1612';
  const GREEN = '#52b788', GREEN_D = '#2d6a4f', OCHRE = '#b6822a';
  const AMBER = '#c98a2b', RED = '#b3402f', MUTE = '#b9b3a3';
  const STYLE = 'https://tiles.openfreemap.org/styles/positron';
  // API edge location (arcs terminate here). Kerala deployment target → Kochi.
  const EDGE = { lon: 76.27, lat: 9.93, label: 'API edge · Kochi' };
  const REDUCE = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);

  // ── compact-map state ─────────────────────────────────────────────────
  let map = null, host = null, ro = null, _onPick = null, _onLevel = null, popup = null;
  let _source = 'requests', _layer = 'origins', _paused = false, _ready = false, _tier = 'state';
  let _data = { country: null, state: null, district: null };
  let _proj = 'globe', _pings = [], _pingTimer = null, _layCap = null;
  let X = null;   // expanded console controller (single instance)

  // ── small helpers ─────────────────────────────────────────────────────
  const $ = (p, el) => (el || document).querySelector(p);
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function fmt(n) {
    if (n == null || isNaN(n)) return '·';
    n = Number(n);
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return String(Math.round(n));
  }
  function titleCase(s) { return String(s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }
  function _hex(h) { h = h.replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function lerpHex(a, b, t) {
    const A = _hex(a), B = _hex(b); t = Math.max(0, Math.min(1, t));
    return 'rgb(' + A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',') + ')';
  }
  function volColor(c, max) {
    const t = Math.min(1, (c || 0) / Math.max(1, (max || 1) * 0.5));
    return t < 0.5 ? lerpHex(GREEN, GREEN_D, t * 2) : lerpHex(GREEN_D, OCHRE, (t - 0.5) * 2);
  }
  function api(path) {
    const f = window.APIN && window.APIN.analyticsApi;
    return f ? f(path) : Promise.reject(new Error('no analyticsApi'));
  }
  // most granular regions that carry lat/lon (the dot layer). `store` lets the
  // expanded console use its own (possibly compare-enriched) snapshot.
  function pointRegions(store) {
    const s = store || _data;
    const d = (s.district && s.district.regions && s.district.regions.length)
      ? s.district : (s.state || s.country);
    return (d && d.regions) || [];
  }

  function pointsGeoJSON(store) {
    const cd = pointRegions(store);
    return {
      type: 'FeatureCollection',
      features: cd.filter(r => r.lat != null && r.lon != null).map(r => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [r.lon, r.lat] },
        properties: {
          name: r.district || r.state || r.cc || '?',
          sub: [r.state, r.cc].filter(Boolean).join(' · '),
          cc: r.cc || '', state: r.state || '', district: r.district || '',
          count: r.count || 0, top: r.top_disease || '', crop: r.top_crop || '',
          error_rate: r.error_rate == null ? -1 : r.error_rate,
          avg_latency: r.avg_latency == null ? -1 : r.avg_latency,
          ood_rate: r.ood_rate == null ? -1 : r.ood_rate,
          avg_confidence: r.avg_confidence == null ? -1 : r.avg_confidence,
          delta: r.delta_pct == null ? null : r.delta_pct,
          prev: r.prev_count == null ? null : r.prev_count
        }
      }))
    };
  }

  // ── basemap recolour (paper land · sea-glass ocean · ink coastlines) ───
  function paperInk(m) {
    let layers = [];
    try { layers = m.getStyle().layers || []; } catch (e) { return; }
    layers.forEach(L => {
      const id = L.id, t = L.type;
      try {
        // In Positron the BACKGROUND layer is the land base; water is drawn as
        // fill polygons on top. So land = paper, water = sea-glass.
        if (t === 'background') m.setPaintProperty(id, 'background-color', PAPER);
        else if (/water|ocean|sea|bathym|marine/i.test(id) && t === 'fill') m.setPaintProperty(id, 'fill-color', SEA);
        else if (/(coast|shoreline)/i.test(id) && t === 'line') { m.setPaintProperty(id, 'line-color', COAST); m.setPaintProperty(id, 'line-opacity', 0.7); }
        else if (/(boundary|admin)/i.test(id) && t === 'line') { m.setPaintProperty(id, 'line-color', 'rgba(26,22,18,0.5)'); m.setPaintProperty(id, 'line-opacity', 0.55); }
        else if (/(building|road|bridge|tunnel|transportation|highway|aeroway|rail)/i.test(id)) {
          if (t === 'fill') m.setPaintProperty(id, 'fill-opacity', 0.05);
          else if (t === 'line') m.setPaintProperty(id, 'line-opacity', 0.08);
        } else if (/(land|landuse|landcover|park|wood|grass|forest|earth)/i.test(id) && t === 'fill') {
          m.setPaintProperty(id, 'fill-color', PAPER); m.setPaintProperty(id, 'fill-opacity', 1);
        } else if (t === 'symbol') {
          try { m.setPaintProperty(id, 'text-color', '#5b5446'); m.setPaintProperty(id, 'text-halo-color', PAPER); } catch (e) { }
        }
      } catch (e) { }
    });
  }

  // Zoom-aware radius so dots stay legible at world view AND scale up on zoom.
  function radiusExpr() {
    return ['interpolate', ['linear'], ['zoom'],
      0, ['interpolate', ['linear'], ['get', 'count'], 1, 2.6, 40, 8],
      4, ['interpolate', ['linear'], ['get', 'count'], 1, 4, 40, 13],
      8, ['interpolate', ['linear'], ['get', 'count'], 1, 5, 40, 18]];
  }
  function metricColorExpr(metric) {
    if (metric === 'errors')
      return ['case', ['<', ['get', 'error_rate'], 0], MUTE,
        ['interpolate', ['linear'], ['get', 'error_rate'], 0, GREEN, 0.1, AMBER, 0.4, RED]];
    if (metric === 'latency')
      return ['case', ['<', ['get', 'avg_latency'], 0], MUTE,
        ['interpolate', ['linear'], ['get', 'avg_latency'], 0, GREEN, 400, AMBER, 1500, RED]];
    if (metric === 'ood')
      return ['case', ['<', ['get', 'ood_rate'], 0], MUTE,
        ['interpolate', ['linear'], ['get', 'ood_rate'], 0, GREEN, 0.1, AMBER, 0.35, RED]];
    if (metric === 'crop')
      return ['match', ['slice', ['get', 'crop'], 0, 4], 'okra', GREEN, 'bras', OCHRE, GREEN_D];
    // volume (default)
    return ['interpolate', ['linear'], ['get', 'count'], 1, GREEN, 12, GREEN_D, 40, OCHRE];
  }

  // Categorical palettes (built from the data each setData) so the crop/disease
  // layers paint a distinct mosaic rather than the same volume ramp.
  const CAT_PAL = ['#52b788', '#b6822a', '#2d6a4f', '#c98a2b', '#6a8cae', '#a0563b',
    '#7d9b6a', '#9a6a8c', '#4f7a6a', '#b0894a', '#5b7d8c', '#8a6a3e'];
  let _ccExpr = null, _disExpr = null;
  function buildCatExprs(store) {
    const regs = pointRegions(store);
    const ccs = []; const diss = [];
    regs.forEach(r => { if (r.cc && ccs.indexOf(r.cc) < 0) ccs.push(r.cc); });
    regs.forEach(r => { if (r.top_disease && diss.indexOf(r.top_disease) < 0) diss.push(r.top_disease); });
    _ccExpr = ccs.length
      ? ['match', ['get', 'cc']].concat(ccs.reduce((a, c, i) => a.concat([c, CAT_PAL[i % CAT_PAL.length]]), [])).concat([MUTE])
      : MUTE;
    _disExpr = diss.length
      ? ['match', ['get', 'top']].concat(diss.reduce((a, d, i) => a.concat([d, CAT_PAL[i % CAT_PAL.length]]), [])).concat([MUTE])
      : MUTE;
  }
  // Per-layer colour encoding (each toggle = a genuinely different map).
  function layerColorExpr(layer) {
    if (layer === 'crop') return _source === 'scans' ? metricColorExpr('crop') : (_ccExpr || MUTE);
    if (layer === 'disease') return _source === 'scans' ? (_disExpr || MUTE) : metricColorExpr('errors');
    return metricColorExpr('volume');   // origins / density
  }
  function circleColor() { return layerColorExpr(_layer); }
  function layerCaption(L) {
    if (L === 'density') return 'density — request heat cloud';
    if (L === 'crop') return _source === 'scans' ? 'crop mix — okra vs brassica' : 'origin mosaic — by country';
    if (L === 'disease') return _source === 'scans' ? 'disease mix — by top diagnosis' : 'health — by error rate';
    return 'origins — graduated by volume';
  }
  // Apply the active layer to a map: recolour dots, swap dots↔heat-cloud for
  // 'density', and update the on-map caption. Paint transitions animate it.
  function applyLayer(m, capEl) {
    if (!m || !m.getLayer || !m.getLayer('origins-pt')) return;
    const dense = _layer === 'density';
    try {
      m.setPaintProperty('origins-pt', 'circle-color', layerColorExpr(_layer));
      m.setPaintProperty('origins-pt', 'circle-opacity', dense ? 0 : 0.9);
      m.setPaintProperty('origins-pt', 'circle-stroke-opacity', dense ? 0 : 0.7);
      m.setPaintProperty('origins-heat', 'heatmap-opacity', dense
        ? ['interpolate', ['linear'], ['zoom'], 0, 0.92, 11, 0.85]
        : ['interpolate', ['linear'], ['zoom'], 0, 0.5, 4.5, 0.42, 5.6, 0]);
      m.setPaintProperty('origins-heat', 'heatmap-radius', dense
        ? ['interpolate', ['linear'], ['zoom'], 0, 18, 5, 48]
        : ['interpolate', ['linear'], ['zoom'], 0, 10, 5, 34]);
    } catch (e) { }
    if (capEl) capEl.textContent = layerCaption(_layer);
  }

  // ── base data layers (shared by compact + expanded) ────────────────────
  function addBaseLayers(m) {
    if (m.getSource('origins')) return;
    m.addSource('origins', { type: 'geojson', data: pointsGeoJSON() });
    m.addLayer({
      id: 'origins-heat', type: 'heatmap', source: 'origins',
      paint: {
        'heatmap-weight': ['interpolate', ['linear'], ['get', 'count'], 0, 0.2, 40, 1],
        'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'],
          0, 'rgba(82,183,136,0)', 0.3, 'rgba(82,183,136,0.45)', 0.6, 'rgba(45,106,79,0.7)', 1, 'rgba(182,130,42,0.92)'],
        'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 10, 5, 34],
        // Subtle low-zoom glow that fades out as the graduated dots take over.
        'heatmap-opacity': ['interpolate', ['linear'], ['zoom'], 0, 0.5, 4.5, 0.42, 5.6, 0],
        'heatmap-opacity-transition': { duration: 500, delay: 0 },
        'heatmap-radius-transition': { duration: 500, delay: 0 }
      }
    });
    m.addLayer({
      id: 'origins-pt', type: 'circle', source: 'origins', minzoom: 0,
      paint: {
        'circle-radius': radiusExpr(),
        'circle-color': circleColor(),
        'circle-opacity': _layer === 'density' ? 0 : 0.9,
        'circle-stroke-width': 1, 'circle-stroke-color': 'rgba(26,22,18,0.45)',
        'circle-stroke-opacity': _layer === 'density' ? 0 : 0.7,
        'circle-color-transition': { duration: 450, delay: 0 },
        'circle-radius-transition': { duration: 350, delay: 0 },
        'circle-opacity-transition': { duration: 400, delay: 0 }
      }
    });
    // India district borders — our GeoJSON, only at higher zoom (viewport-culled).
    m.addSource('dist-in', { type: 'geojson', data: '/static/geo_district_IN.json?v=9t31a' });
    m.addLayer({
      id: 'dist-in-line', type: 'line', source: 'dist-in', minzoom: 5,
      paint: { 'line-color': 'rgba(26,22,18,0.32)', 'line-width': ['interpolate', ['linear'], ['zoom'], 5, 0.4, 9, 0.9] }
    });
  }

  // ── animated "sonar" dots (DOM markers, CSS-driven, cheap) ─────────────
  function clearPings(store) { store.forEach(mk => { try { mk.remove(); } catch (e) { } }); store.length = 0; }
  function buildPings(m, store, dataStore) {
    clearPings(store);
    if (typeof maplibregl === 'undefined') return;
    const feats = pointsGeoJSON(dataStore).features
      .slice().sort((a, b) => b.properties.count - a.properties.count).slice(0, 56);
    if (!feats.length) return;
    const max = feats[0].properties.count || 1;
    feats.forEach(f => {
      const c = f.properties.count || 0;
      const col = volColor(c, max);
      const sz = 9 + Math.min(20, Math.round((c / max) * 20));
      const el = document.createElement('div');
      el.className = 'an-ping' + (REDUCE ? ' is-static' : '');
      el.style.setProperty('--c', col);
      el.style.setProperty('--sz', sz + 'px');
      el.style.animationDelay = (Math.random() * 1.6).toFixed(2) + 's';
      const mk = new maplibregl.Marker({ element: el }).setLngLat(f.geometry.coordinates).addTo(m);
      store.push(mk);
    });
  }

  // ── globe ↔ 2D preview toggle (bottom-right) ───────────────────────────
  function projThumb(target) {
    // render a tiny preview of the mode you'd switch TO
    if (target === 'mercator') {
      return '<svg viewBox="0 0 44 30" aria-hidden="true">' +
        '<rect x="2.5" y="3.5" width="39" height="23" rx="2.5" fill="' + PAPER + '" stroke="' + INK + '" stroke-width="1.2"/>' +
        '<path d="M2.5 11H41.5M2.5 19H41.5M15 3.5V26.5M29 3.5V26.5" stroke="rgba(26,22,18,.28)" stroke-width=".7"/>' +
        '<path d="M9 16q4-5 8-2t7-1 6 3" fill="none" stroke="' + GREEN_D + '" stroke-width="1.4" stroke-linecap="round"/></svg>';
    }
    return '<svg viewBox="0 0 44 30" aria-hidden="true">' +
      '<circle cx="22" cy="15" r="11.5" fill="' + PAPER + '" stroke="' + INK + '" stroke-width="1.2"/>' +
      '<ellipse cx="22" cy="15" rx="11.5" ry="4.4" fill="none" stroke="rgba(26,22,18,.3)" stroke-width=".7"/>' +
      '<ellipse cx="22" cy="15" rx="4.4" ry="11.5" fill="none" stroke="rgba(26,22,18,.3)" stroke-width=".7"/>' +
      '<path d="M14 13q4 3 8 1t8 .5" fill="none" stroke="' + GREEN_D + '" stroke-width="1.4" stroke-linecap="round"/></svg>';
  }
  function mountProjToggle(m, container, getProj, setProj) {
    const box = document.createElement('button');
    box.type = 'button'; box.className = 'an-projtoggle';
    const render = () => {
      const cur = getProj();
      const target = cur === 'globe' ? 'mercator' : 'globe';
      box.innerHTML = projThumb(target) + '<span class="an-projtoggle-lbl">' + (target === 'mercator' ? '2D map' : '3D globe') + '</span>';
      box.title = 'Switch to ' + (target === 'mercator' ? '2D map' : '3D globe');
    };
    box.addEventListener('click', (e) => { e.stopPropagation(); setProj(); render(); });
    container.appendChild(box);
    render();
    return { render };
  }

  // ── tooltip HTML ───────────────────────────────────────────────────────
  function tipHTML(p) {
    let extra = '';
    if (_source === 'scans') {
      if (p.top) extra += '<br>top: ' + esc(titleCase(p.top));
      if (p.ood_rate >= 0) extra += '<br>OOD ' + (p.ood_rate * 100).toFixed(0) + '%';
    } else {
      if (p.top) extra += '<br>top: ' + esc(p.top);
      if (p.error_rate >= 0) extra += '<br>err ' + (p.error_rate * 100).toFixed(1) + '%';
      if (p.avg_latency >= 0) extra += ' · ' + fmt(p.avg_latency) + 'ms';
    }
    if (p.delta != null) {
      const up = p.delta >= 0;
      extra += '<br><span style="color:' + (up ? '#8fe0b4' : '#e2a08f') + '">' + (up ? '▲' : '▼') + ' ' + Math.abs(p.delta).toFixed(0) + '% vs prev</span>';
    }
    return '<b>' + esc(p.name) + '</b> <small>' + esc(p.sub) + '</small><br>' +
      fmt(p.count) + ' ' + (_source === 'scans' ? 'scans' : 'requests') + extra;
  }

  function wireInteractions(m, pop, onClick) {
    m.on('mousemove', 'origins-pt', e => {
      const f = e.features[0]; m.getCanvas().style.cursor = 'pointer';
      pop.setLngLat(e.lngLat).setHTML(tipHTML(f.properties)).addTo(m);
    });
    m.on('mouseleave', 'origins-pt', () => { m.getCanvas().style.cursor = ''; if (pop) pop.remove(); });
    m.on('click', 'origins-pt', e => {
      const f = e.features[0], c = f.geometry.coordinates;
      m.flyTo({ center: c, zoom: Math.max(m.getZoom(), 6.5), speed: 1.2 });
      if (onClick) onClick(f.properties, c);
    });
  }

  // ── compact map ────────────────────────────────────────────────────────
  function tierForZoom(z) { return z < 3.2 ? 'country' : z < 5.5 ? 'state' : 'district'; }
  function onZoom() {
    if (!map) return;
    const t = tierForZoom(map.getZoom());
    if (t !== _tier) { _tier = t; if (_onLevel) _onLevel(t, _data[t]); }
  }

  function mount(hostEl, opts) {
    injectCSS();
    host = hostEl; opts = opts || {};
    _onPick = opts.onPickRegion || null; _onLevel = opts.onLevel || null;
    if (!hostEl || typeof window.maplibregl === 'undefined') {
      if (hostEl) hostEl.innerHTML = '<div class="an-ph an-ph-empty">map unavailable</div>'; return;
    }
    hostEl.innerHTML = '';
    map = new maplibregl.Map({
      container: hostEl, style: STYLE,
      center: [40, 20], zoom: 1.6, minZoom: 0.6, maxZoom: 11,
      attributionControl: false, dragRotate: false
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: 'OpenFreeMap · OSM · DB-IP' }), 'bottom-right');
    popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: 'an-mlp', offset: 12 });
    map.on('style.load', () => {
      try { map.setProjection({ type: 'globe' }); } catch (e) { }
      paperInk(map); addBaseLayers(map); _ready = true;
      // Compact map: clicking a region dot opens the request list (→ drawer)
      // for that origin. Scans have no request rows, so they keep focus-only.
      wireInteractions(map, popup, (p) => {
        if (_source !== 'scans') openRegionRequests(p);
        if (_onPick) _onPick(p.name);
      });
      // on-map caption describing the active layer encoding
      _layCap = document.createElement('div'); _layCap.className = 'an-laycap';
      hostEl.appendChild(_layCap);
      buildCatExprs(_data); applyLayer(map, _layCap);
      if (pointRegions().length) buildPings(map, _pings);
      mountProjToggle(map, hostEl, () => _proj, () => {
        _proj = _proj === 'globe' ? 'mercator' : 'globe';
        try { map.setProjection({ type: _proj }); } catch (e) { }
      });
      onZoom();
      requestAnimationFrame(() => { try { map.resize(); } catch (e) { } });
    });
    map.on('zoom', onZoom);
    ro = new ResizeObserver(() => { try { map.resize(); } catch (e) { } }); ro.observe(hostEl);
  }

  function setData(all) {
    _data = all || _data;
    if (_ready && map && map.getSource('origins')) {
      map.getSource('origins').setData(pointsGeoJSON());
      buildCatExprs(_data); applyLayer(map, _layCap);
      if (!_paused) buildPings(map, _pings);
    }
    if (X && X.alive()) X.setData(_data);
  }
  function setSource(s) {
    _source = s;
    if (_ready && map) { buildCatExprs(_data); applyLayer(map, _layCap); }
    if (X && X.alive()) X.setSource(s);
  }
  function setLayer(l) {
    _layer = l;
    if (_ready && map) applyLayer(map, _layCap);
  }
  function setRange() { }
  function focusRegion(name) {
    if (!map || !name) return;
    let r = null;
    ['district', 'state', 'country'].forEach(t => {
      if (r || !_data[t]) return;
      r = _data[t].regions.find(x => [x.district, x.state, x.cc].some(v => v && String(v).toLowerCase() === String(name).toLowerCase()));
    });
    if (r && r.lat != null && r.lon != null) map.flyTo({ center: [r.lon, r.lat], zoom: 6, speed: 1.2 });
  }
  function spawnRipple(m, lnglat, scale) {
    try {
      const el = document.createElement('div'); el.className = 'an-ml-pulse';
      if (scale) el.style.transform = 'scale(' + scale + ')';
      const mk = new maplibregl.Marker({ element: el }).setLngLat(lnglat).addTo(m);
      setTimeout(() => { try { mk.remove(); } catch (e) { } }, 1600);
    } catch (e) { }
  }
  function onScan(e) {
    if (e == null || e.lat == null || e.lon == null) return;
    if (map && _ready && !_paused) spawnRipple(map, [e.lon, e.lat]);
    if (X && X.alive()) X.onScan(e);
  }
  function pause() { _paused = true; }
  function resume() { _paused = false; if (map) { try { map.resize(); } catch (e) { } if (_ready) buildPings(map, _pings); } }

  // ════════════════════════════════════════════════════════════════════════
  //  EXPANDED CONSOLE
  // ════════════════════════════════════════════════════════════════════════
  function renderExpanded(el, all, opts) {
    injectCSS();
    if (all) _data = all;
    if (opts && opts.source) _source = opts.source;
    if (typeof window.maplibregl === 'undefined') { el.innerHTML = '<div class="an-ph">map unavailable</div>'; return; }
    if (X && X.destroy) { try { X.destroy(); } catch (e) { } X = null; }
    X = makeExpanded(el);
  }

  function sparkline(vals, w, h, col, fill) {
    vals = vals || []; w = w || 180; h = h || 34;
    if (!vals.length) return '<svg width="' + w + '" height="' + h + '"></svg>';
    const max = Math.max(1, ...vals);
    const step = vals.length > 1 ? w / (vals.length - 1) : w;
    const pts = vals.map((v, i) => [i * step, h - 2 - (v / max) * (h - 4)]);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
    const area = fill ? '<path d="' + d + ' L' + w + ' ' + h + ' L0 ' + h + ' Z" fill="' + fill + '" opacity=".18"/>' : '';
    return '<svg width="' + w + '" height="' + h + '" class="an-spark">' + area +
      '<path d="' + d + '" fill="none" stroke="' + (col || GREEN_D) + '" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  }

  // ── shared interaction helpers (tooltip · cross-links · interactive spark) ─
  let _tipEl = null;
  function _tip() {
    if (!_tipEl) { _tipEl = document.createElement('div'); _tipEl.className = 'an-tip'; _tipEl.style.display = 'none'; document.body.appendChild(_tipEl); }
    return _tipEl;
  }
  function showTip(x, y, html) {
    const t = _tip(); t.innerHTML = html; t.style.display = 'block';
    const vw = window.innerWidth, tw = 220;
    t.style.left = Math.min(x + 14, vw - tw) + 'px'; t.style.top = (y + 14) + 'px';
  }
  function hideTip() { if (_tipEl) _tipEl.style.display = 'none'; }
  function closeAll() {
    hideTip();
    try { if (CMP) { CMP.remove(); CMP = null; } } catch (e) { }
    try { if (window.APIN.lightbox && window.APIN.lightbox.close) window.APIN.lightbox.close(); } catch (e) { }
  }
  function crossTime(since, until) {
    try { if (window.APIN.keyDetail && window.APIN.keyDetail.filterRequests) { window.APIN.keyDetail.filterRequests(since, until); closeAll(); } } catch (e) { }
  }
  function crossEndpoint(path) {
    try { if (window.APIN.keyDetail && window.APIN.keyDetail.filterEndpoint) { window.APIN.keyDetail.filterEndpoint(path); closeAll(); } } catch (e) { }
  }
  // bucket i (oldest→newest) → its [since,until] clock window + a label
  function bucketRange(i, total, bsec) {
    const now = Date.now();
    const until = new Date(now - (total - 1 - i) * bsec * 1000);
    const since = new Date(until.getTime() - bsec * 1000);
    return { since: since.toISOString(), until: until.toISOString(),
      label: since.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) };
  }
  // interactive sparkline (per-bucket hover + click). Returns SVG; wire after insert.
  function sparkI(vals, w, h, col, fill) {
    vals = vals || []; w = w || 240; h = h || 40;
    if (!vals.length) return '<svg width="' + w + '" height="' + h + '"></svg>';
    const max = Math.max(1, ...vals), step = vals.length > 1 ? w / (vals.length - 1) : w;
    const pts = vals.map((v, i) => [i * step, h - 2 - (v / max) * (h - 4)]);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
    const area = fill ? '<path d="' + d + ' L' + w + ' ' + h + ' L0 ' + h + ' Z" fill="' + fill + '" opacity=".18"/>' : '';
    const dots = pts.map(p => '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="1.8" fill="' + (col || GREEN_D) + '"/>').join('');
    return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '" class="an-spk" data-n="' + vals.length + '">' +
      area + '<path d="' + d + '" fill="none" stroke="' + (col || GREEN_D) + '" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>' +
      dots + '<rect class="an-spk-cur" x="-9" width="0.8" height="' + h + '" fill="rgba(26,22,18,.45)" opacity="0"/></svg>';
  }
  function wireSpark(svg, vals, bsec, label) {
    if (!svg || !vals || !vals.length) return;
    const n = vals.length, cur = svg.querySelector('.an-spk-cur');
    const W = svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal.width : 240;
    const idxAt = ev => { const r = svg.getBoundingClientRect(); const step = n > 1 ? r.width / (n - 1) : r.width; return Math.max(0, Math.min(n - 1, Math.round((ev.clientX - r.left) / step))); };
    svg.style.cursor = 'pointer';
    svg.addEventListener('mousemove', ev => {
      const i = idxAt(ev), br = bucketRange(i, n, bsec);
      if (cur) { cur.setAttribute('x', (n > 1 ? i * W / (n - 1) : 0).toFixed(1)); cur.setAttribute('opacity', '1'); }
      showTip(ev.clientX, ev.clientY, '<b>' + esc(br.label) + '</b><br>' + label(vals[i]) + '<br><i>click → see these requests</i>');
    });
    svg.addEventListener('mouseleave', () => { hideTip(); if (cur) cur.setAttribute('opacity', '0'); });
    svg.addEventListener('click', ev => { const i = idxAt(ev), br = bucketRange(i, n, bsec); crossTime(br.since, br.until); });
  }
  // ── narrative insight builders ─────────────────────────────────────────
  function cmpFactor(now, prev) {
    if (!prev) return now ? 'new' : '0%';
    const dp = Math.round(100 * (now - prev) / prev);
    return Math.abs(dp) >= 200 ? (now / Math.max(1, prev)).toFixed(1) + '×' : (dp >= 0 ? '+' : '') + dp + '%';
  }
  function globalInsight(gl, source) {
    const now = gl.total || 0, prev = gl.prev_total || 0, noun = source === 'scans' ? 'scans' : 'requests';
    const regs = gl.regions || [];
    const movers = regs.filter(r => (r.prev_count || 0) > 0).map(r => ({ r: r, ab: (r.count || 0) - (r.prev_count || 0) }));
    const risers = movers.filter(m => m.ab > 0).sort((a, b) => b.ab - a.ab);
    const fallers = movers.filter(m => m.ab < 0).sort((a, b) => a.ab - b.ab);
    const fresh = regs.filter(r => (r.prev_count || 0) === 0 && (r.count || 0) > 0);
    const gone = gl.gone || [];
    let s = prev
      ? 'Traffic ' + (now >= prev ? 'grew' : 'fell') + ' <b>' + cmpFactor(now, prev) + '</b> vs the previous window (' + fmt(prev) + ' → ' + fmt(now) + ' ' + noun + ').'
      : fmt(now) + ' ' + noun + ' this window — no prior-window traffic to compare against.';
    if (risers.length) s += ' Led by <b>' + esc(regNm(risers[0].r)) + '</b> (+' + risers[0].ab + ')' + (risers[1] ? ' and ' + esc(regNm(risers[1].r)) + ' (+' + risers[1].ab + ')' : '') + '.';
    if (fresh.length) s += ' ' + fresh.length + ' new origin' + (fresh.length > 1 ? 's' : '') + ' appeared';
    s += gone.length ? '; ' + gone.length + ' went quiet.' : (fresh.length ? '.' : '');
    if (fallers.length) s += ' Biggest drop: <b>' + esc(regNm(fallers[0].r)) + '</b> (' + fallers[0].ab + ').';
    return s;
  }
  function regionInsight(rd, name, source) {
    const now = rd.count || 0, prev = rd.prev_count || 0, noun = source === 'scans' ? 'scans' : 'requests';
    let s = '<b>' + esc(name) + '</b> sent ' + fmt(now) + ' ' + noun + ' this window';
    s += prev ? ' (<b>' + cmpFactor(now, prev) + '</b> vs ' + fmt(prev) + ' before).' : (prev === 0 ? ' — a brand-new origin.' : '.');
    if (source !== 'scans') {
      if (rd.prev_p95_latency != null && rd.p95_latency != null && rd.p95_latency !== rd.prev_p95_latency)
        s += ' p95 latency ' + (rd.p95_latency < rd.prev_p95_latency ? 'improved' : 'rose') + ' to ' + fmt(rd.p95_latency) + 'ms.';
      if (rd.prev_error_rate != null) { const ne = rd.error_rate * 100, pe = rd.prev_error_rate * 100; if (Math.abs(ne - pe) >= 0.5) s += ' Error rate ' + (ne < pe ? 'fell' : 'rose') + ' to ' + ne.toFixed(1) + '%.'; }
    } else if (rd.prev_ood_rate != null) {
      const no = rd.ood_rate * 100, po = rd.prev_ood_rate * 100; if (Math.abs(no - po) >= 0.5) s += ' OOD rate ' + (no < po ? 'fell' : 'rose') + ' to ' + no.toFixed(0) + '%.';
    }
    return s;
  }

  // great-circle samples between two [lng,lat]
  function greatCircle(a, b, steps) {
    steps = steps || 64;
    const R = Math.PI / 180, D = 180 / Math.PI;
    const lat1 = a[1] * R, lon1 = a[0] * R, lat2 = b[1] * R, lon2 = b[0] * R;
    const d = 2 * Math.asin(Math.sqrt(Math.sin((lat2 - lat1) / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin((lon2 - lon1) / 2) ** 2));
    if (!d) return [a, b];
    const out = [];
    for (let i = 0; i <= steps; i++) {
      const f = i / steps;
      const A = Math.sin((1 - f) * d) / Math.sin(d), B = Math.sin(f * d) / Math.sin(d);
      const x = A * Math.cos(lat1) * Math.cos(lon1) + B * Math.cos(lat2) * Math.cos(lon2);
      const y = A * Math.cos(lat1) * Math.sin(lon1) + B * Math.cos(lat2) * Math.sin(lon2);
      const z = A * Math.sin(lat1) + B * Math.sin(lat2);
      out.push([Math.atan2(y, x) * D, Math.atan2(z, Math.sqrt(x * x + y * y)) * D]);
    }
    return out;
  }
  // day/night terminator polygon for a given Date (approx sub-solar point)
  function terminator(date) {
    const R = Math.PI / 180, D = 180 / Math.PI;
    const jd = date / 86400000 + 2440587.5, n = jd - 2451545.0;
    const Ls = (280.46 + 0.9856474 * n) % 360, g = (357.528 + 0.9856003 * n) % 360;
    const lam = (Ls + 1.915 * Math.sin(g * R) + 0.02 * Math.sin(2 * g * R)) * R;
    const eps = 23.439 * R;
    const dec = Math.asin(Math.sin(eps) * Math.sin(lam)) * D;
    const utch = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
    const slng = 180 - utch * 15;
    const pts = [];
    for (let lng = -180; lng <= 180; lng += 3) {
      const tlat = Math.atan(-Math.cos((lng - slng) * R) / Math.tan((dec || 1e-6) * R)) * D;
      pts.push([lng, tlat]);
    }
    const dark = dec > 0 ? -90 : 90;
    pts.push([180, dark], [-180, dark], pts[0]);
    return { type: 'Feature', geometry: { type: 'Polygon', coordinates: [pts] }, properties: {} };
  }

  function makeExpanded(el) {
    el.innerHTML =
      '<div class="an-gx" data-proj="globe">' +
      '  <div class="an-gx-main">' +
      '    <div class="an-gx-bar">' +
      '      <div class="an-seg an-gx-metric" role="group" aria-label="Overlay metric">' +
      '        <button data-m="volume" aria-pressed="true">Volume</button>' +
      '        <button data-m="errors">Errors</button>' +
      '        <button data-m="latency">Latency</button>' +
      '        <button data-m="ood">OOD</button>' +
      '      </div>' +
      '      <button class="an-gx-tg" data-tg="anom">Anomalies</button>' +
      '      <button class="an-gx-tg" data-tg="arcs">Arcs</button>' +
      '      <button class="an-gx-tg" data-tg="replay">Replay</button>' +
      '      <button class="an-gx-cmp" id="angx-cmp" title="Open Compare mode — this window vs previous">⇄ Compare</button>' +
      '      <span class="an-gx-flex"></span>' +
      '      <span class="an-gx-kpi" id="angx-kpi"></span>' +
      '    </div>' +
      '    <div class="an-gx-map" id="angx-map"></div>' +
      '    <div class="an-gx-scale" id="angx-scale"></div>' +
      '    <div class="an-gx-insight" id="angx-insight"></div>' +
      '    <div class="an-gx-scrub" id="angx-scrub" hidden></div>' +
      '  </div>' +
      '  <aside class="an-gx-side" id="angx-side">' +
      '    <div class="an-gx-mode" id="angx-mode"></div>' +
      '    <div class="an-gx-dosshost" id="angx-doss"><div class="an-gx-doss-empty">Click a region on the map for its dossier — volume, top endpoints, errors, latency, clients and trend.</div></div>' +
      '  </aside>' +
      '</div>';

    const node = $('#angx-map', el);
    const wrap = $('.an-gx', el);
    let m2 = null, pop2 = null, pings2 = [], proj = 'globe', destroyed = false, ro2 = null;
    let metric = 'volume', tgl = { anom: false, arcs: false, replay: false };
    function curTier() { return (liveStore.district && liveStore.district.regions && liveStore.district.regions.length) ? 'district' : 'state'; }
    let anomMarkers = [], arcRAF = null, arcPhase = 0, focusArc = null;
    let replay = null;  // {frames, idx, playing, timer, speed, bucket}
    let liveStore = _data;
    let watchdog = null;

    function alive() { return !destroyed && el.isConnected; }
    // The lightbox just removes the DOM on close — detect detachment and free
    // the WebGL context so repeated open/close never leaks GPU memory.
    watchdog = setInterval(() => {
      if (!el.isConnected && !destroyed) { destroy(); if (X && X.alive === alive) X = null; }
    }, 1200);

    m2 = new maplibregl.Map({
      container: node, style: STYLE, center: [40, 20], zoom: 1.7,
      minZoom: 0.6, maxZoom: 12, attributionControl: false, dragRotate: false
    });
    m2.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    pop2 = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: 'an-mlp', offset: 12 });

    m2.on('style.load', () => {
      try { m2.setProjection({ type: 'globe' }); } catch (e) { }
      paperInk(m2); addBaseLayers(m2);
      // night-shade (replay only), under the dots
      m2.addSource('night', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      m2.addLayer({ id: 'night-fill', type: 'fill', source: 'night', paint: { 'fill-color': 'rgba(20,18,28,0.32)', 'fill-opacity': 0 } }, 'origins-heat');
      // flight arcs
      m2.addSource('arcs', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      m2.addLayer({
        id: 'arcs-line', type: 'line', source: 'arcs',
        layout: { 'line-cap': 'round' },
        paint: { 'line-color': OCHRE, 'line-width': 1.6, 'line-opacity': 0.85, 'line-dasharray': [0, 4, 3] }
      });
      // edge marker
      const eEl = document.createElement('div'); eEl.className = 'an-edge';
      eEl.innerHTML = '<span></span>'; eEl.title = EDGE.label;
      new maplibregl.Marker({ element: eEl }).setLngLat([EDGE.lon, EDGE.lat]).addTo(m2);
      wireInteractions(m2, pop2, (p, c) => openDossier(p, c));
      buildPings(m2, pings2, liveStore);
      mountProjToggle(m2, node, () => proj, () => {
        proj = proj === 'globe' ? 'mercator' : 'globe';
        wrap.setAttribute('data-proj', proj);
        try { m2.setProjection({ type: proj }); } catch (e) { }
      });
      applyMetric(); refreshKPIs(); renderInsight(); renderMode();
      requestAnimationFrame(() => { try { m2.resize(); } catch (e) { } intro(); });
      try { ro2 = new ResizeObserver(() => { try { m2 && m2.resize(); } catch (e) { } }); ro2.observe(node); } catch (e) { }
    });

    // intro animation — gentle fly-in
    function intro() {
      try { m2.easeTo({ zoom: 2.1, duration: 1100, essential: true }); } catch (e) { }
    }

    // ── metric overlay ───────────────────────────────────────────────────
    function applyMetric() {
      if (!m2.getLayer('origins-pt')) return;
      // expanded always shows graduated dots (never the density-hidden state)
      m2.setPaintProperty('origins-pt', 'circle-opacity', 0.9);
      m2.setPaintProperty('origins-pt', 'circle-stroke-opacity', 0.7);
      m2.setPaintProperty('origins-pt', 'circle-color', metricColorExpr(metric === 'volume' ? (_layer === 'crop' && _source === 'scans' ? 'crop' : 'volume') : metric));
      renderScale();
    }
    function renderScale() {
      const sc = $('#angx-scale', el);
      const defs = {
        volume: ['Requests / region', GREEN, GREEN_D, OCHRE, ['low', 'mid', 'high']],
        errors: ['Error rate', GREEN, AMBER, RED, ['0%', '10%', '40%+']],
        latency: ['Avg latency', GREEN, AMBER, RED, ['fast', '400ms', '1.5s+']],
        ood: ['OOD rate', GREEN, AMBER, RED, ['0%', '10%', '35%+']]
      };
      const d = defs[metric] || defs.volume;
      sc.innerHTML = '<span class="an-gx-scale-t">' + d[0] + '</span>' +
        '<i class="an-gx-scale-bar" style="background:linear-gradient(90deg,' + d[1] + ',' + d[2] + ',' + d[3] + ')"></i>' +
        '<span class="an-gx-scale-k">' + d[4].map(esc).join('<i></i>') + '</span>';
    }

    // ── KPI + insight ─────────────────────────────────────────────────────
    function curTierData() {
      return (liveStore.district && liveStore.district.regions && liveStore.district.regions.length)
        ? liveStore.district : (liveStore.state || liveStore.country || { regions: [], total: 0 });
    }
    function refreshKPIs() {
      const d = curTierData();
      const total = d.total || 0;
      const kpi = $('#angx-kpi', el);
      let bits = [fmt(total) + ' ' + (_source === 'scans' ? 'scans' : 'requests'), (d.regions || []).length + ' regions'];
      if (_source === 'scans') { if (d.ood_rate != null) bits.push('OOD ' + (d.ood_rate * 100).toFixed(0) + '%'); }
      else { if (d.error_rate != null) bits.push('err ' + (d.error_rate * 100).toFixed(1) + '%'); if (d.avg_latency != null) bits.push(fmt(d.avg_latency) + 'ms'); }
      kpi.innerHTML = bits.map(b => '<span>' + esc(b) + '</span>').join('');
    }
    function renderInsight() {
      const d = curTierData(); const regs = (d.regions || []).slice();
      const ins = $('#angx-insight', el); if (!regs.length) { ins.innerHTML = ''; return; }
      const busiest = regs[0];
      let errHot = null, latHot = null;
      regs.forEach(r => {
        if (_source !== 'scans') {
          if (r.error_rate > 0 && (!errHot || r.error_rate > errHot.error_rate)) errHot = r;
          if (r.avg_latency != null && (!latHot || r.avg_latency > latHot.avg_latency)) latHot = r;
        } else {
          if (r.ood_rate > 0 && (!errHot || r.ood_rate > errHot.ood_rate)) errHot = r;
        }
      });
      const nm = r => r.district || r.state || r.cc || '?';
      let parts = ['<b>Busiest:</b> ' + esc(nm(busiest)) + ' (' + fmt(busiest.count) + ')'];
      if (errHot) parts.push(_source === 'scans'
        ? '<b>OOD hotspot:</b> ' + esc(nm(errHot)) + ' (' + (errHot.ood_rate * 100).toFixed(0) + '%)'
        : '<b>Error hotspot:</b> ' + esc(nm(errHot)) + ' (' + (errHot.error_rate * 100).toFixed(0) + '%)');
      if (latHot && _source !== 'scans') parts.push('<b>Slowest:</b> ' + esc(nm(latHot)) + ' (' + fmt(latHot.avg_latency) + 'ms)');
      ins.innerHTML = parts.join('<span class="an-gx-dot">·</span>');
    }

    // ── mode-info panel (explains the active overlay + its KPIs) ───────────
    const regionName = r => r.district || r.state || r.cc || '?';
    function haversineKm(a, b) {
      const R = 6371, toR = Math.PI / 180;
      const dLat = (b[1] - a[1]) * toR, dLon = (b[0] - a[0]) * toR;
      const s = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * toR) * Math.cos(b[1] * toR) * Math.sin(dLon / 2) ** 2;
      return 2 * R * Math.asin(Math.sqrt(s));
    }
    function fmtDur(sec) {
      sec = sec || 0;
      if (sec < 3600) return Math.max(1, Math.round(sec / 60)) + ' min';
      if (sec < 86400) return (sec / 3600).toFixed(sec % 3600 ? 1 : 0) + ' h';
      return (sec / 86400).toFixed(sec % 86400 ? 1 : 0) + ' d';
    }
    function anomKind(r) {
      if ((r.prev_count === 0 || r.prev_count == null) && r.count >= 3) return 'new';
      if (r.delta_pct != null && r.delta_pct >= 60 && r.count >= 4) return 'surge';
      if (r.delta_pct != null && r.delta_pct <= -50 && r.prev_count >= 4) return 'drop';
      return null;
    }
    function modeCard(title, desc, kpis, accent) {
      return '<div class="an-gx-modecard" style="--ac:' + (accent || GREEN_D) + '">' +
        '<div class="an-gx-mode-h">' + esc(title) + '</div>' +
        '<p class="an-gx-mode-d">' + esc(desc) + '</p>' +
        '<div class="an-gx-mode-k">' + kpis.map(k =>
          '<div><b>' + esc(String(k[1])) + '</b><span>' + esc(k[0]) + '</span></div>').join('') + '</div></div>';
    }
    function renderMode() {
      const host = $('#angx-mode', el); if (!host) return;
      const d = curTierData(); const regs = (d.regions || []).slice();
      const edgeName = EDGE.label.replace('API edge · ', '');
      let html = '';
      if (tgl.replay) {
        const win = replay ? replay.frames.length * replay.bucket : 0;
        html = modeCard('Replay',
          'Re-plays ' + (_source === 'scans' ? 'scan' : 'request') + ' arrivals across the window, oldest → newest. Each origin pulses as its traffic landed; the shaded band is night (the sun line sweeps west). Use ▶, the scrubber, or speed.',
          [['window', win ? fmtDur(win) : '…'], ['frames', replay ? replay.frames.length : '…'], ['per frame', replay ? fmtDur(replay.bucket) : '…']], OCHRE);
      } else if (tgl.anom) {
        let s = 0, dp = 0, nw = 0;
        regs.forEach(r => { const k = anomKind(r); if (k === 'surge') s++; else if (k === 'drop') dp++; else if (k === 'new') nw++; });
        html = modeCard('Anomaly pins',
          'Flags origins whose volume deviates sharply from the previous window: ▲ surge (≥60% up), ▼ drop (≥50% down), ✦ new origin. Hover any pin for the number.',
          [['▲ surge', s], ['▼ drop', dp], ['✦ new', nw]], RED);
      } else if (tgl.arcs) {
        const origins = pointRegions(liveStore).filter(r => r.lat != null && r.lon != null);
        const top = origins.slice().sort((a, b) => b.count - a.count)[0];
        let far = null, farD = 0;
        origins.forEach(r => { const km = haversineKm([r.lon, r.lat], [EDGE.lon, EDGE.lat]); if (km > farD) { farD = km; far = r; } });
        html = modeCard('Flight arcs',
          'Each arc traces ' + (_source === 'scans' ? 'scans' : 'requests') + ' from their origin to your API edge (' + edgeName + ', Kerala). Dashes flow origin → edge; click a region to isolate just its arc.',
          [['edge', edgeName], ['top origin', top ? regionName(top) + ' (' + fmt(top.count) + ')' : '—'], ['farthest', far ? Math.round(farD) + ' km' : '—']], OCHRE);
      } else {
        html = '<div class="an-gx-modehint"><b>Overlays</b>' +
          '<ul><li><span class="an-gx-sw" style="background:' + RED + '"></span><b>Anomalies</b> — surge / drop / new origins</li>' +
          '<li><span class="an-gx-sw" style="background:' + OCHRE + '"></span><b>Arcs</b> — request paths to the API edge</li>' +
          '<li><span class="an-gx-sw" style="background:' + INK + '"></span><b>Replay</b> — watch traffic arrive over time</li>' +
          '<li><span class="an-gx-sw" style="background:' + GREEN_D + '"></span><b>⇄ Compare</b> — full this-window vs previous report</li></ul>' +
          'Or hover a dot for a tooltip · click for a full dossier.</div>';
      }
      host.innerHTML = html;
    }

    // ── region dossier (B) ────────────────────────────────────────────────
    function openDossier(p, c) {
      const side = $('#angx-doss', el);
      side.innerHTML = '<div class="an-gx-doss"><div class="an-gx-doss-h"><b>' + esc(p.name) + '</b><small>' + esc(p.sub) + '</small></div><div class="an-ph">loading dossier&hellip;</div></div>';
      if (tgl.arcs) { focusArc = c; drawArcs(); renderMode(); }
      const dtier = p.district ? 'district' : (p.state ? 'state' : 'country');
      const q = 'geo/region?source=' + encodeURIComponent(_source) + '&tier=' + dtier + '&compare=1' +
        '&cc=' + encodeURIComponent(p.cc || '') +
        (p.state ? '&state=' + encodeURIComponent(p.state) : '') +
        (p.district ? '&district=' + encodeURIComponent(p.district) : '');
      api(q).then(d => { if (alive()) renderDossier(p, d); }).catch(() => {
        if (alive()) side.innerHTML = '<div class="an-gx-doss"><div class="an-gx-doss-h"><b>' + esc(p.name) + '</b></div><div class="an-ph">dossier unavailable</div></div>';
      });
    }
    function renderDossier(p, d) {
      const side = $('#angx-doss', el);
      const isScan = _source === 'scans';
      const up = (d.delta_pct || 0) >= 0;
      const deltaTxt = d.delta_pct == null ? 'new' : (up ? '▲' : '▼') + ' ' + Math.abs(d.delta_pct).toFixed(0) + '%';
      const topMax = Math.max(1, (d.top[0] || {}).count || 1);
      const topRows = (d.top || []).map((t, i) => {
        const w = Math.round((t.count / topMax) * 100);
        return '<div class="an-gx-top is-click" data-ep="' + esc(t.key) + '" data-cnt="' + t.count + '" tabindex="0">' +
          '<span class="an-gx-top-k">' + esc(isScan ? titleCase(t.key) : t.key) + '</span>' +
          '<span class="an-gx-top-bar"><i style="width:' + w + '%"></i></span>' +
          '<span class="an-gx-top-n">' + fmt(t.count) + '</span></div>';
      }).join('') || '<div class="an-gx-mut">no data</div>';
      const metricsHTML = isScan
        ? '<div class="an-gx-stat"><b>' + ((d.ood_rate * 100) || 0).toFixed(0) + '%</b><span>OOD rate</span></div>' +
        '<div class="an-gx-stat"><b>' + (d.avg_confidence != null ? d.avg_confidence.toFixed(2) : '·') + '</b><span>avg conf</span></div>'
        : '<div class="an-gx-stat"><b>' + ((d.error_rate * 100) || 0).toFixed(1) + '%</b><span>error rate</span></div>' +
        '<div class="an-gx-stat"><b>' + (d.p95_latency != null ? fmt(d.p95_latency) + 'ms' : '·') + '</b><span>p95</span></div>' +
        '<div class="an-gx-stat"><b>' + fmt(d.clients) + '</b><span>clients</span></div>';
      const sumTop = (d.top || []).reduce((s, t) => s + t.count, 0) || 1;
      side.innerHTML =
        '<div class="an-gx-doss">' +
        '  <div class="an-gx-doss-h"><b>' + esc(p.name) + '</b><small>' + esc(p.sub) + '</small>' +
        '    <span class="an-gx-delta ' + (d.delta_pct == null ? 'is-new' : up ? 'is-up' : 'is-dn') + '">' + deltaTxt + '</span></div>' +
        '  <div class="an-gx-big">' + fmt(d.count) + '<span>' + (isScan ? 'scans' : 'requests') + ' · prev ' + fmt(d.prev_count) + '</span></div>' +
        '  <div class="an-gx-dossins">' + regionInsight(d, p.name, _source) + '</div>' +
        '  <div class="an-gx-sec"><label>Volume trend</label><div class="an-spkwrap" id="angx-spk-v"></div></div>' +
        '  <div class="an-gx-stats">' + metricsHTML + '</div>' +
        (isScan ? '' : '<div class="an-gx-sec"><label>Error trend</label><div class="an-spkwrap" id="angx-spk-e"></div></div>') +
        '  <div class="an-gx-sec"><label>' + (isScan ? 'Top diagnoses' : 'Top endpoints') + (isScan ? '' : ' <small class="an-gx-hint">click → filter requests</small>') + '</label>' + topRows + '</div>' +
        '  <div class="an-gx-meta">first seen ' + esc((d.first_seen || '·').replace('T', ' ').slice(0, 16)) + '</div>' +
        '  <button class="an-gx-cta" id="angx-filter">Filter page to this region →</button>' +
        '  <button class="an-gx-cta an-gx-cta2" id="angx-cmpr">⇄ Compare mode</button>' +
        '</div>';
      // interactive sparklines (hover bucket → tooltip · click → Requests time-filter)
      const bsec = d.bucket_seconds || 3600;
      const vWrap = $('#angx-spk-v', el);
      if (vWrap) { vWrap.innerHTML = sparkI(d.trend, 232, 40, GREEN_D, GREEN); wireSpark(vWrap.querySelector('svg'), d.trend, bsec, v => v + (isScan ? ' scans' : ' requests')); }
      const eWrap = $('#angx-spk-e', el);
      if (eWrap) { eWrap.innerHTML = sparkI(d.err_trend, 232, 30, RED, RED); wireSpark(eWrap.querySelector('svg'), d.err_trend, bsec, v => v + ' errors'); }
      // clickable top rows (endpoints → cross-link · diagnoses → focus)
      Array.from(el.querySelectorAll('.an-gx-doss .an-gx-top.is-click')).forEach(row => {
        const ep = row.getAttribute('data-ep'), cnt = +row.getAttribute('data-cnt');
        row.addEventListener('mousemove', ev => showTip(ev.clientX, ev.clientY, '<b>' + esc(isScan ? titleCase(ep) : ep) + '</b><br>' + fmt(cnt) + ' · ' + Math.round(cnt / sumTop * 100) + '% of top<br><i>click → ' + (isScan ? 'focus this diagnosis' : 'see these requests') + '</i>'));
        row.addEventListener('mouseleave', hideTip);
        row.addEventListener('click', () => {
          hideTip();
          if (isScan) { if (window.APIN.analyticsFocus) window.APIN.analyticsFocus({ disease: ep }); closeAll(); }
          else crossEndpoint(ep);
        });
      });
      const btn = $('#angx-filter', el);
      if (btn) btn.addEventListener('click', () => {
        if (window.APIN && window.APIN.analyticsFocus) window.APIN.analyticsFocus({ region: p.name });
        if (window.APIN && window.APIN.lightbox && window.APIN.lightbox.close) window.APIN.lightbox.close();
      });
      const cbtn = $('#angx-cmpr', el);
      if (cbtn) cbtn.addEventListener('click', () => openCompare({
        source: _source, tier: curTier(),
        region: { cc: p.cc, state: p.state, district: p.district, name: p.name, sub: p.sub }
      }));
    }

    // ── anomalies (D) ──────────────────────────────────────────────────────
    function clearAnoms() { anomMarkers.forEach(m => { try { m.remove(); } catch (e) { } }); anomMarkers = []; }
    function loadAnoms() {
      const q = 'geo?tier=district&source=' + encodeURIComponent(_source) + '&compare=1';
      api(q).then(d => {
        if (!alive() || !tgl.anom) return;
        // reuse the compare-enriched data so the KPI counts match the pins
        liveStore = Object.assign({}, liveStore, { district: d });
        if (m2.getSource('origins')) m2.getSource('origins').setData(pointsGeoJSON(liveStore));
        drawAnoms((d && d.regions) || []);
        renderMode();
      }).catch(() => { });
    }
    function drawAnoms(regs) {
      clearAnoms();
      regs.forEach(r => {
        if (r.lat == null || r.lon == null) return;
        let kind = null;
        if ((r.prev_count === 0 || r.prev_count == null) && r.count >= 3) kind = 'new';
        else if (r.delta_pct != null && r.delta_pct >= 60 && r.count >= 4) kind = 'surge';
        else if (r.delta_pct != null && r.delta_pct <= -50 && r.prev_count >= 4) kind = 'drop';
        if (!kind) return;
        const el2 = document.createElement('div');
        el2.className = 'an-anom an-anom-' + kind;
        el2.innerHTML = '<span>' + (kind === 'surge' ? '▲' : kind === 'drop' ? '▼' : '✦') + '</span>';
        el2.title = (r.district || r.state || r.cc) + ' · ' + kind + (r.delta_pct != null ? ' ' + r.delta_pct.toFixed(0) + '%' : '');
        const mk = new maplibregl.Marker({ element: el2 }).setLngLat([r.lon, r.lat]).addTo(m2);
        anomMarkers.push(mk);
      });
    }

    // ── flight arcs (E) ────────────────────────────────────────────────────
    function arcFeatures() {
      const feats = [];
      if (focusArc) feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: greatCircle(focusArc, [EDGE.lon, EDGE.lat]) }, properties: {} });
      else {
        const regs = pointRegions(liveStore).filter(r => r.lat != null && r.lon != null).slice(0, 8);
        regs.forEach(r => feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: greatCircle([r.lon, r.lat], [EDGE.lon, EDGE.lat]) }, properties: {} }));
      }
      return { type: 'FeatureCollection', features: feats };
    }
    function drawArcs() {
      if (!m2.getSource('arcs')) return;
      m2.getSource('arcs').setData(tgl.arcs ? arcFeatures() : { type: 'FeatureCollection', features: [] });
      if (tgl.arcs && !arcRAF && !REDUCE) animateArcs();
      if (!tgl.arcs && arcRAF) { cancelAnimationFrame(arcRAF); arcRAF = null; }
    }
    // canonical MapLibre "marching ants" dash sequence (leading-gap grows)
    const ARC_SEQ = [[0, 4, 3], [0.5, 4, 2.5], [1, 4, 2], [1.5, 4, 1.5], [2, 4, 1], [2.5, 4, 0.5],
      [3, 4, 0], [0, 0.5, 3, 3.5], [0, 1, 3, 3], [0, 1.5, 3, 2.5], [0, 2, 3, 2], [0, 2.5, 3, 1.5], [0, 3, 3, 1], [0, 3.5, 3, 0.5]];
    function animateArcs() {
      let last = 0;
      const tick = (ts) => {
        if (!alive() || !tgl.arcs) { arcRAF = null; return; }
        if (ts - last > 70) {
          last = ts; arcPhase = (arcPhase + 1) % ARC_SEQ.length;
          try { m2.setPaintProperty('arcs-line', 'line-dasharray', ARC_SEQ[arcPhase]); } catch (e) { }
        }
        arcRAF = requestAnimationFrame(tick);
      };
      arcRAF = requestAnimationFrame(tick);
    }

    // ── replay (A) ─────────────────────────────────────────────────────────
    function openReplay() {
      const scrub = $('#angx-scrub', el);
      scrub.hidden = false;
      scrub.innerHTML = '<div class="an-ph">loading replay&hellip;</div>';
      const dtier = (liveStore.district && liveStore.district.regions && liveStore.district.regions.length) ? 'district' : 'state';
      api('geo/replay?source=' + encodeURIComponent(_source) + '&tier=' + dtier + '&frames=120')
        .then(d => { if (alive()) buildReplay(d); }).catch(() => { if (alive()) scrub.innerHTML = '<div class="an-ph">replay unavailable</div>'; });
    }
    function closeReplay() {
      if (replay && replay.timer) clearInterval(replay.timer);
      replay = null;
      const scrub = $('#angx-scrub', el); scrub.hidden = true; scrub.innerHTML = '';
      if (m2.getLayer('night-fill')) m2.setPaintProperty('night-fill', 'fill-opacity', 0);
      renderMode();
    }
    function buildReplay(d) {
      const frames = (d && d.frames) || [];
      replay = { frames, idx: 0, playing: false, timer: null, speed: 2, bucket: d.bucket_seconds || 600 };
      const scrub = $('#angx-scrub', el);
      const maxV = Math.max(1, ...frames.map(f => f.total || 0));
      const bars = frames.map((f, i) => '<i data-i="' + i + '" style="height:' + (4 + (f.total / maxV) * 26).toFixed(0) + 'px"></i>').join('');
      scrub.innerHTML =
        '<div class="an-gx-rphead">' +
        '  <span class="an-gx-clock" id="angx-clock">--:--:--</span>' +
        '  <span class="an-gx-comm" id="angx-comm">press ▶ to replay traffic</span>' +
        '</div>' +
        '<div class="an-gx-rprow">' +
        '  <button class="an-gx-play" id="angx-play" aria-label="Play">▶</button>' +
        '  <div class="an-gx-bars" id="angx-bars">' + bars + '</div>' +
        '  <input class="an-gx-range" id="angx-rng" type="range" min="0" max="' + Math.max(0, frames.length - 1) + '" value="0">' +
        '  <select class="an-gx-spd" id="angx-spd" aria-label="Playback speed"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select>' +
        '</div>';
      const play = $('#angx-play', el), rng = $('#angx-rng', el), spd = $('#angx-spd', el), barsEl = $('#angx-bars', el);
      play.addEventListener('click', () => replay.playing ? pauseReplay() : playReplay());
      rng.addEventListener('input', () => { seekReplay(+rng.value, false); });
      spd.addEventListener('change', () => { replay.speed = +spd.value; if (replay.playing) { pauseReplay(); playReplay(); } });
      barsEl.addEventListener('click', e => { const i = e.target && e.target.getAttribute('data-i'); if (i != null) { rng.value = i; seekReplay(+i, false); } });
      renderMode();
      seekReplay(0, false);
    }
    // narration for a frame, prefixed with the local-device time it happened
    function frameNarr(f, when) {
      const pts = (f && f.points) || [];
      const tot = pts.reduce((s, p) => s + (p.count || 0), 0);
      const noun = _source === 'scans' ? 'scans' : 'requests';
      const ts = when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      if (!tot) return '<b>' + ts + '</b> — quiet, no ' + noun + ' this interval';
      const top = pts.slice().sort((a, b) => b.count - a.count).slice(0, 3)
        .map(p => fmt(p.count) + ' from ' + esc(p.label || '?')).join(' · ');
      return '<b>' + ts + '</b> — ' + fmt(tot) + ' ' + noun + ': ' + top;
    }
    function playReplay() {
      if (!replay) return; replay.playing = true;
      const play = $('#angx-play', el); if (play) play.textContent = '❚❚';
      replay.timer = setInterval(() => {
        if (!alive()) return closeReplay();
        let ni = replay.idx + 1;
        if (ni >= replay.frames.length) ni = 0;
        seekReplay(ni, true);
      }, Math.max(90, 700 / replay.speed));
    }
    function pauseReplay() {
      if (!replay) return; replay.playing = false;
      if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
      const play = $('#angx-play', el); if (play) play.textContent = '▶';
    }
    function seekReplay(i, emit) {
      if (!replay) return;
      replay.idx = i;
      const rng = $('#angx-rng', el); if (rng && +rng.value !== i) rng.value = i;
      const bars = $('#angx-bars', el);
      if (bars) Array.from(bars.children).forEach((b, k) => b.classList.toggle('on', k <= i));
      const ago = (replay.frames.length - 1 - i) * replay.bucket;
      const when = new Date(Date.now() - ago * 1000);
      const f = replay.frames[i];
      // clock: local device time, moving with the replay
      const clock = $('#angx-clock', el);
      if (clock) clock.innerHTML = when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
        '<small>' + when.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' · your time</small>';
      // live commentary narrating what happened this minute
      const comm = $('#angx-comm', el); if (comm) comm.innerHTML = frameNarr(f, when);
      // night terminator
      try {
        if (m2.getSource('night')) { m2.getSource('night').setData(terminator(when)); m2.setPaintProperty('night-fill', 'fill-opacity', 1); }
      } catch (e) { }
      // pulse frame points
      if (f && emit) (f.points || []).forEach(pt => spawnRipple(m2, [pt.lon, pt.lat], 0.5 + Math.min(1.3, (pt.count || 1) / 6)));
    }

    // ── toolbar wiring ─────────────────────────────────────────────────────
    $('.an-gx-metric', el).addEventListener('click', e => {
      const b = e.target.closest('[data-m]'); if (!b) return;
      metric = b.getAttribute('data-m');
      Array.from(el.querySelectorAll('.an-gx-metric [data-m]')).forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
      applyMetric();
    });
    el.querySelectorAll('.an-gx-tg').forEach(b => b.addEventListener('click', () => {
      const k = b.getAttribute('data-tg'); tgl[k] = !tgl[k];
      b.classList.toggle('on', tgl[k]);
      if (k === 'anom') { if (tgl.anom) loadAnoms(); else clearAnoms(); }
      else if (k === 'arcs') { if (!tgl.arcs) focusArc = null; drawArcs(); }
      else if (k === 'replay') { if (tgl.replay) openReplay(); else closeReplay(); }
      renderMode();
    }));
    // Compare is now a full-window mode (not a recolour toggle).
    const cmpBtn = $('#angx-cmp', el);
    if (cmpBtn) cmpBtn.addEventListener('click', () =>
      openCompare({ source: _source, tier: curTier(), region: null }));

    // ── live + data updates ────────────────────────────────────────────────
    function setData(all) {
      liveStore = all || liveStore;
      if (m2 && m2.getSource('origins')) { m2.getSource('origins').setData(pointsGeoJSON(liveStore)); buildPings(m2, pings2, liveStore); }
      refreshKPIs(); renderInsight(); renderMode();
      if (tgl.arcs) drawArcs();
    }
    function setSource(s) { applyMetric(); refreshKPIs(); renderInsight(); renderMode(); }
    function onScan(e) { if (m2 && e && e.lat != null && (!replay || !replay.playing)) spawnRipple(m2, [e.lon, e.lat]); }

    function destroy() {
      destroyed = true;
      if (watchdog) { clearInterval(watchdog); watchdog = null; }
      if (ro2) { try { ro2.disconnect(); } catch (e) { } ro2 = null; }
      if (arcRAF) cancelAnimationFrame(arcRAF);
      if (replay && replay.timer) clearInterval(replay.timer);
      clearPings(pings2); clearAnoms();
      try { if (m2) m2.remove(); } catch (e) { }
      m2 = null;
    }

    return { alive, setData, setSource, onScan, destroy };
  }

  // ════════════════════════════════════════════════════════════════════════
  //  COMPARE MODE — dedicated full-window this-vs-previous report
  // ════════════════════════════════════════════════════════════════════════
  let CMP = null;
  const RANGE_SEC = { '15m': 900, '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800, '30d': 2592000 };
  function cmpCtx() { try { return (window.APIN.analyticsCtx && window.APIN.analyticsCtx()) || {}; } catch (e) { return {}; } }
  function fmtPct(p) { return p == null ? '' : (p >= 0 ? '+' : '') + Math.round(p) + '%'; }
  function regNm(r) { return r.district || r.state || r.cc || '?'; }
  function regSub(r) { const n = regNm(r); return [r.state, r.cc].filter(x => x && x !== n).join(' · '); }
  function windowLabels() {
    const ctx = cmpCtx(); const sec = RANGE_SEC[ctx.range || '7d'] || 604800; const now = Date.now();
    const fmtD = t => new Date(t).toLocaleDateString([], { month: 'short', day: 'numeric' });
    return { range: ctx.range || '7d', now: fmtD(now - sec * 1000) + ' – ' + fmtD(now), prev: fmtD(now - sec * 2000) + ' – ' + fmtD(now - sec * 1000) };
  }
  function dualSpark(now, prev, w, h) {
    now = now || []; prev = prev || []; w = w || 240; h = h || 46;
    const max = Math.max(1, ...now, ...prev);
    const path = vals => { if (!vals.length) return ''; const step = vals.length > 1 ? w / (vals.length - 1) : w; return vals.map((v, i) => (i ? 'L' : 'M') + (i * step).toFixed(1) + ' ' + (h - 2 - (v / max) * (h - 4)).toFixed(1)).join(' '); };
    return '<svg width="' + w + '" height="' + h + '" class="an-spark">' +
      '<path d="' + path(prev) + '" fill="none" stroke="' + MUTE + '" stroke-width="1.4" stroke-dasharray="3 2" opacity=".85"/>' +
      '<path d="' + path(now) + '" fill="none" stroke="' + GREEN_D + '" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  }
  function deltaMapSVG(regions, gone) {
    const W = 360, H = 180, proj = (lon, lat) => [(lon + 180) / 360 * W, (90 - lat) / 180 * H];
    let grat = '';
    for (let lo = -120; lo <= 120; lo += 60) { const x = (lo + 180) / 360 * W; grat += '<line x1="' + x.toFixed(1) + '" y1="0" x2="' + x.toFixed(1) + '" y2="' + H + '" stroke="rgba(26,22,18,.08)" stroke-width=".6"/>'; }
    for (let la = -60; la <= 60; la += 30) { const y = (90 - la) / 180 * H; grat += '<line x1="0" y1="' + y.toFixed(1) + '" x2="' + W + '" y2="' + y.toFixed(1) + '" stroke="rgba(26,22,18,.08)" stroke-width=".6"/>'; }
    const all = (regions || []).filter(r => r.lat != null && r.lon != null);
    const maxV = Math.max(1, ...all.map(r => Math.max(r.count || 0, r.prev_count || 0)));
    let dots = '';
    const dd = r => 'data-cc="' + esc(r.cc || '') + '" data-state="' + esc(r.state || '') + '" data-district="' + esc(r.district || '') +
      '" data-nm="' + esc(regNm(r)) + '" data-sub="' + esc(regSub(r)) + '" data-now="' + (r.count || 0) + '" data-prev="' + (r.prev_count || 0) + '"';
    all.forEach(r => {
      const c = proj(r.lon, r.lat), v = Math.max(r.count || 0, r.prev_count || 0), rad = (2 + Math.sqrt(v / maxV) * 7).toFixed(1);
      const col = r.delta_pct == null ? '#6a8cae' : r.delta_pct > 0 ? GREEN_D : r.delta_pct < 0 ? RED : MUTE;
      dots += '<circle class="an-cmp-dot" ' + dd(r) + ' cx="' + c[0].toFixed(1) + '" cy="' + c[1].toFixed(1) + '" r="' + rad + '" fill="' + col + '" fill-opacity=".82" stroke="rgba(26,22,18,.4)" stroke-width=".5"></circle>';
    });
    (gone || []).forEach(r => { if (r.lat == null) return; const c = proj(r.lon, r.lat); dots += '<circle class="an-cmp-dot" ' + dd(r) + ' cx="' + c[0].toFixed(1) + '" cy="' + c[1].toFixed(1) + '" r="4" fill="none" stroke="' + RED + '" stroke-width="1.1" stroke-dasharray="2 1.5"></circle>'; });
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" class="an-cmp-map" preserveAspectRatio="xMidYMid meet"><rect width="' + W + '" height="' + H + '" rx="6" fill="#f3ecd9"/>' + grat + dots + '</svg>';
  }
  function rowData(r) {
    return ' data-cc="' + esc(r.cc || '') + '" data-state="' + esc(r.state || '') + '" data-district="' + esc(r.district || '') +
      '" data-nm="' + esc(regNm(r)) + '" data-sub="' + esc(regSub(r)) + '"';
  }
  function moverRow(r, maxAbs) {
    const ab = (r.count || 0) - (r.prev_count || 0), up = ab >= 0;
    const pct = (r.prev_count >= 3 && r.delta_pct != null) ? ' <em>' + fmtPct(r.delta_pct) + '</em>' : '';
    const w = Math.round(Math.min(1, Math.abs(ab) / Math.max(1, maxAbs)) * 100);
    return '<div class="an-cmp-row is-click"' + rowData(r) + ' tabindex="0"><span class="an-cmp-row-k">' + esc(regNm(r)) + '<small>' + esc(regSub(r)) + '</small></span>' +
      '<span class="an-cmp-row-v">' + fmt(r.prev_count || 0) + ' → ' + fmt(r.count || 0) + pct + '</span>' +
      '<span class="an-cmp-bar ' + (up ? 'up' : 'dn') + '"><i style="width:' + w + '%"></i></span></div>';
  }
  function statDiff(label, nowV, prevV, unit, lowerBetter) {
    const fmtv = v => v == null ? '·' : (unit === '%' ? (v * 100).toFixed(1) + '%' : unit === 'ms' ? fmt(v) + 'ms' : fmt(v));
    let arrow = '—', cls = 'is-flat';
    if (nowV != null && prevV != null && nowV !== prevV) {
      if (nowV > prevV) { arrow = '▲'; cls = lowerBetter ? 'is-dn' : 'is-up'; }
      else { arrow = '▼'; cls = lowerBetter ? 'is-up' : 'is-dn'; }
    }
    return '<div class="an-cmp-stat"><span class="an-cmp-stat-l">' + esc(label) + '</span><b>' + fmtv(nowV) + '</b>' +
      '<span class="an-cmp-stat-d ' + cls + '">' + arrow + ' from ' + fmtv(prevV) + '</span></div>';
  }
  function openCompare(opts) {
    injectCSS(); opts = opts || {};
    if (CMP) { try { CMP.remove(); } catch (e) { } CMP = null; }
    const root = document.createElement('div'); root.className = 'an-cmp-root';
    root.innerHTML = '<div class="an-cmp-backdrop"></div><div class="an-cmp" role="dialog" aria-modal="true"><div class="an-cmp-load">building comparison&hellip;</div></div>';
    document.body.appendChild(root); CMP = root;
    const panel = root.querySelector('.an-cmp');
    const close = () => { try { root.remove(); } catch (e) { } if (CMP === root) CMP = null; document.removeEventListener('keydown', onKey); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    root.querySelector('.an-cmp-backdrop').addEventListener('click', close);
    const st = { source: opts.source || 'requests', tier: opts.tier || 'state', region: opts.region || null };
    function header(title, sub) {
      const wl = windowLabels();
      return '<div class="an-cmp-head"><div class="an-cmp-htitle"><div class="an-cmp-title">' + esc(title) + '</div>' +
        (sub ? '<div class="an-cmp-sub">' + esc(sub) + '</div>' : '') + '</div>' +
        '<div class="an-cmp-wins"><span><i class="dot now"></i>this window <b>' + esc(wl.now) + '</b></span>' +
        '<span><i class="dot prev"></i>previous <b>' + esc(wl.prev) + '</b></span></div>' +
        '<button class="an-cmp-x" aria-label="Close">✕</button></div>';
    }
    function wireClose() { const x = panel.querySelector('.an-cmp-x'); if (x) x.addEventListener('click', close); }
    function fail() { panel.innerHTML = '<div class="an-cmp-load">comparison unavailable</div>'; }
    function load() {
      panel.innerHTML = '<div class="an-cmp-load">building comparison&hellip;</div>';
      if (st.region) {
        const reg = st.region, dt = reg.district ? 'district' : reg.state ? 'state' : 'country';
        const q = 'geo/region?source=' + encodeURIComponent(st.source) + '&tier=' + dt + '&compare=1&cc=' + encodeURIComponent(reg.cc || '') +
          (reg.state ? '&state=' + encodeURIComponent(reg.state) : '') + (reg.district ? '&district=' + encodeURIComponent(reg.district) : '');
        api(q).then(rd => renderRegion(reg, rd)).catch(fail);
      } else {
        api('geo?tier=' + st.tier + '&source=' + encodeURIComponent(st.source) + '&compare=1').then(renderGlobal).catch(fail);
      }
    }
    function renderGlobal(gl) {
      const noun = st.source === 'scans' ? 'scans' : 'requests';
      const now = gl.total || 0, prev = gl.prev_total || 0;
      const delta = prev ? Math.round(100 * (now - prev) / prev) : (now ? 100 : 0);
      const regs = gl.regions || [];
      const movers = regs.filter(r => (r.prev_count || 0) > 0).map(r => ({ r: r, ab: (r.count || 0) - (r.prev_count || 0) }));
      const risers = movers.filter(m => m.ab > 0).sort((a, b) => b.ab - a.ab).slice(0, 8).map(m => m.r);
      const fallers = movers.filter(m => m.ab < 0).sort((a, b) => a.ab - b.ab).slice(0, 8).map(m => m.r);
      const fresh = regs.filter(r => (r.prev_count || 0) === 0 && (r.count || 0) > 0).sort((a, b) => b.count - a.count).slice(0, 8);
      const gone = (gl.gone || []).slice(0, 8);
      const maxAbs = Math.max(1, ...movers.map(m => Math.abs(m.ab)));
      const barMax = Math.max(now, prev, 1);
      const list = (rows, none) => rows.length ? rows : '<div class="an-cmp-none">' + none + '</div>';
      const simpleRow = (r, tail) => '<div class="an-cmp-row is-click"' + rowData(r) + ' tabindex="0"><span class="an-cmp-row-k">' + esc(regNm(r)) + '<small>' + esc(regSub(r)) + '</small></span><span class="an-cmp-row-v">' + tail + '</span></div>';
      panel.innerHTML = header('Compare · all origins', regs.length + ' active origins · ' + noun) +
        '<div class="an-cmp-body">' +
        '<div class="an-cmp-insight">' + globalInsight(gl, st.source) + '</div>' +
        '<div class="an-cmp-summary">' +
        '<div class="an-cmp-totals"><div class="an-cmp-big"><b>' + fmt(now) + '</b><span>this window</span></div>' +
        '<div class="an-cmp-delta ' + (delta >= 0 ? 'is-up' : 'is-dn') + '">' + (delta >= 0 ? '▲' : '▼') + ' ' + Math.abs(delta) + '%</div>' +
        '<div class="an-cmp-big alt"><b>' + fmt(prev) + '</b><span>previous</span></div></div>' +
        '<div class="an-cmp-cmpbars"><div class="an-cmp-cb"><label>now</label><span><i class="now" style="width:' + (now / barMax * 100).toFixed(0) + '%"></i></span></div>' +
        '<div class="an-cmp-cb"><label>prev</label><span><i class="prev" style="width:' + (prev / barMax * 100).toFixed(0) + '%"></i></span></div></div>' +
        '</div>' +
        '<div class="an-cmp-mapwrap"><label>Δ map · green grew · red shrank · blue new · dashed went quiet · <small class="an-gx-hint">hover a dot · click → region compare</small></label>' + deltaMapSVG(regs, gl.gone) + '</div>' +
        '<div class="an-cmp-cols">' +
        '<div class="an-cmp-col"><h4 class="up">▲ Risers</h4>' + list(risers.map(r => moverRow(r, maxAbs)).join(''), 'none') + '</div>' +
        '<div class="an-cmp-col"><h4 class="dn">▼ Fallers</h4>' + list(fallers.map(r => moverRow(r, maxAbs)).join(''), 'none') + '</div>' +
        '</div>' +
        '<div class="an-cmp-cols">' +
        '<div class="an-cmp-col"><h4 class="new">✦ New origins</h4>' + list(fresh.map(r => simpleRow(r, fmt(r.count) + ' new')).join(''), 'none') + '</div>' +
        '<div class="an-cmp-col"><h4 class="quiet">◌ Went quiet</h4>' + list(gone.map(r => simpleRow(r, 'was ' + fmt(r.prev_count))).join(''), 'none') + '</div>' +
        '</div></div>';
      wireClose();
      wireDrill();
    }
    // hover + click on any [data-cc] element (dots & rows) → that region's compare
    function wireDrill() {
      Array.from(panel.querySelectorAll('.an-cmp-dot, .an-cmp-row.is-click')).forEach(node => {
        const ds = node.dataset || {};
        if (!ds.cc) return;
        const drill = () => { hideTip(); st.region = { cc: ds.cc, state: ds.state || null, district: ds.district || null, name: ds.nm || ds.cc, sub: ds.sub || '' }; load(); };
        node.addEventListener('click', drill);
        node.addEventListener('keydown', e => { if (e.key === 'Enter') drill(); });
        node.addEventListener('mousemove', ev => showTip(ev.clientX, ev.clientY,
          '<b>' + esc(ds.nm || ds.cc) + '</b>' + (ds.sub ? ' <small>' + esc(ds.sub) + '</small>' : '') +
          (ds.now != null && ds.prev != null ? '<br>' + fmt(+ds.prev) + ' → ' + fmt(+ds.now) : '') +
          '<br><i>click → region compare</i>'));
        node.addEventListener('mouseleave', hideTip);
      });
    }
    function renderRegion(reg, rd) {
      const isScan = st.source === 'scans';
      const now = rd.count || 0, prev = rd.prev_count || 0, delta = rd.delta_pct;
      const diffs = isScan
        ? statDiff('OOD rate', rd.ood_rate, rd.prev_ood_rate, '%', true) + statDiff('avg confidence', rd.avg_confidence, rd.prev_avg_confidence, '', false)
        : statDiff('error rate', rd.error_rate, rd.prev_error_rate, '%', true) + statDiff('p95 latency', rd.p95_latency, rd.prev_p95_latency, 'ms', true) +
          statDiff('avg latency', rd.avg_latency, rd.prev_avg_latency, 'ms', true) + statDiff('clients', rd.clients, rd.prev_clients, '', false);
      const topRows = (rd.top || []).map(t => '<div class="an-cmp-row' + (isScan ? '' : ' is-clickep') + '" data-ep="' + esc(t.key) + '"' + (isScan ? '' : ' tabindex="0"') + '><span class="an-cmp-row-k">' + esc(isScan ? titleCase(t.key) : t.key) + '</span><span class="an-cmp-row-v">' + fmt(t.count) + '</span></div>').join('') || '<div class="an-cmp-none">no data</div>';
      panel.innerHTML = header('Compare · ' + (reg.name || regNm(reg)), reg.sub || '') +
        '<div class="an-cmp-body">' +
        '<div class="an-cmp-insight">' + regionInsight(rd, reg.name || regNm(reg), st.source) + '</div>' +
        '<div class="an-cmp-summary">' +
        '<div class="an-cmp-totals"><div class="an-cmp-big"><b>' + fmt(now) + '</b><span>this window</span></div>' +
        '<div class="an-cmp-delta ' + ((delta || 0) >= 0 ? 'is-up' : 'is-dn') + '">' + (delta == null ? 'new' : (delta >= 0 ? '▲ ' : '▼ ') + Math.abs(delta) + '%') + '</div>' +
        '<div class="an-cmp-big alt"><b>' + fmt(prev) + '</b><span>previous</span></div></div>' +
        '<div class="an-cmp-trend"><label>volume · now (solid) vs previous (dashed)</label>' + dualSpark(rd.trend, rd.prev_trend, 300, 56) + '</div>' +
        '</div>' +
        '<div class="an-cmp-section"><label>what changed</label><div class="an-cmp-stats">' + diffs + '</div></div>' +
        '<div class="an-cmp-section"><label>' + (isScan ? 'top diagnoses' : 'top endpoints') + ' · this window' + (isScan ? '' : ' <small class="an-gx-hint">click → filter requests</small>') + '</label>' + topRows + '</div>' +
        '<button class="an-cmp-allbtn">← all origins</button>' +
        '</div>';
      wireClose();
      Array.from(panel.querySelectorAll('.an-cmp-row.is-clickep')).forEach(row => {
        const ep = row.getAttribute('data-ep');
        row.addEventListener('click', () => crossEndpoint(ep));
        row.addEventListener('mousemove', ev => showTip(ev.clientX, ev.clientY, '<b>' + esc(ep) + '</b><br><i>click → see these requests</i>'));
        row.addEventListener('mouseleave', hideTip);
      });
      const ab = panel.querySelector('.an-cmp-allbtn');
      if (ab) ab.addEventListener('click', () => { st.region = null; load(); });
    }
    load();
  }

  // ════════════════════════════════════════════════════════════════════════
  //  REGION REQUEST LIST — click a map dot → filtered requests → full drawer
  // ════════════════════════════════════════════════════════════════════════
  function agoTime(iso) {
    try { if (window.APIN && window.APIN.time && window.APIN.time.ago) return window.APIN.time.ago(iso); } catch (e) { }
    if (!iso) return '';
    const ms = Date.now() - new Date(iso).getTime();
    if (isNaN(ms)) return '';
    if (ms < 60000) return Math.max(1, Math.round(ms / 1000)) + 's';
    if (ms < 3600000) return Math.round(ms / 60000) + 'm';
    if (ms < 86400000) return Math.round(ms / 3600000) + 'h';
    return Math.round(ms / 86400000) + 'd';
  }
  function reqRow(r) {
    const sc = +r.status_code || 0, k = sc >= 500 ? 'danger' : sc >= 400 ? 'amber' : 'ok';
    const m = (r.method || '').toUpperCase();
    return '<div class="an-rqx" data-rid="' + esc(r.id) + '" tabindex="0">' +
      '<span class="an-rqx-m meth-' + esc(m) + '">' + esc(m) + '</span>' +
      '<span class="an-rqx-path" title="' + esc(r.path || '') + '">' + esc(r.path || '') + '</span>' +
      '<span class="an-rqx-sc an-sc-' + k + '">' + (sc || '—') + '</span>' +
      '<span class="an-rqx-lat">' + (r.latency_ms != null ? fmt(r.latency_ms) + 'ms' : '·') + '</span>' +
      '<span class="an-rqx-ago">' + esc(agoTime(r.timestamp)) + '</span></div>';
  }
  function openRegionRequests(p) {
    if (!(window.APIN && window.APIN.lightbox)) return;
    const dt = p.district ? 'district' : (p.state ? 'state' : 'country');
    const q = 'geo/region/requests?tier=' + dt + '&cc=' + encodeURIComponent(p.cc || '') +
      (p.state ? '&state=' + encodeURIComponent(p.state) : '') + (p.district ? '&district=' + encodeURIComponent(p.district) : '');
    window.APIN.lightbox.open({
      title: 'Requests · ' + p.name, subtitle: p.sub || '', hashKey: 'an-georeq',
      build: async (panel) => {
        panel.innerHTML = '<div class="an-rqx-chips" id="an-rqx-chips">' +
          ['all', '2xx', '4xx', '5xx'].map(s => '<button data-s="' + s + '"' + (s === 'all' ? ' aria-pressed="true"' : '') + '>' + s + '</button>').join('') +
          '</div><div id="an-rqx-wrap"><div class="an-ph">loading requests&hellip;</div></div>';
        let all = [], cur = 'all';
        try { const d = await api(q); all = (d && d.items) || []; } catch (e) { }
        const draw = () => {
          const wrap = panel.querySelector('#an-rqx-wrap'); if (!wrap) return;
          const rows = all.filter(r => { const sc = +r.status_code || 0, b = sc >= 500 ? '5xx' : sc >= 400 ? '4xx' : '2xx'; return cur === 'all' || b === cur; });
          wrap.innerHTML = rows.length
            ? '<div class="an-rqx-head"><span>' + rows.length + ' request' + (rows.length > 1 ? 's' : '') + '</span><span class="an-gx-hint">click a row → full drawer</span></div><div class="an-rqx-list">' + rows.map(reqRow).join('') + '</div>'
            : '<div class="an-ph an-ph-empty">no requests from this region in the window</div>';
          wrap.querySelectorAll('.an-rqx[data-rid]').forEach(elr => elr.addEventListener('click', () => {
            const rid = elr.getAttribute('data-rid');
            if (window.APIN.requestDrawer && window.APIN.requestDrawer.open) window.APIN.requestDrawer.open(rid);
            else if (window.APIN.keyDetail && window.APIN.keyDetail.openRequest) window.APIN.keyDetail.openRequest(rid);
          }));
        };
        panel.querySelectorAll('#an-rqx-chips button').forEach(b => b.addEventListener('click', () => {
          cur = b.getAttribute('data-s'); panel.querySelectorAll('#an-rqx-chips button').forEach(x => x.removeAttribute('aria-pressed')); b.setAttribute('aria-pressed', 'true'); draw();
        }));
        draw();
      }
    });
  }

  // ── injected CSS (self-contained, versioned with this file) ────────────
  function injectCSS() {
    if (document.getElementById('an-geo-css')) return;
    const s = document.createElement('style'); s.id = 'an-geo-css';
    s.textContent = `
.an-ping{width:var(--sz,12px);height:var(--sz,12px);pointer-events:none}
.an-ping::before{content:'';position:absolute;inset:0;border-radius:50%;border:1.5px solid var(--c,#52b788);opacity:.0;animation:anPing 2.4s ease-out infinite}
.an-ping.is-static::before{animation:none;opacity:.5}
@keyframes anPing{0%{transform:scale(.35);opacity:.7}70%{opacity:.12}100%{transform:scale(2.3);opacity:0}}
.an-projtoggle{position:absolute;right:10px;bottom:30px;z-index:3;display:flex;align-items:center;gap:7px;background:var(--paper,#efe7d4);border:1.4px solid var(--ink,#1a1612);border-radius:9px;padding:5px 8px 5px 5px;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.18);transition:transform .12s ease,box-shadow .12s ease}
.an-projtoggle:hover{transform:translateY(-1px);box-shadow:0 5px 16px rgba(0,0,0,.24)}
.an-projtoggle svg{width:40px;height:27px;display:block;border-radius:4px}
.an-projtoggle-lbl{font:600 11px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);padding-right:3px}
.an-edge{width:14px;height:14px;border-radius:50%;background:var(--ink,#1a1612);border:2px solid #efe7d4;box-shadow:0 0 0 0 rgba(26,22,18,.4);animation:anEdge 2.6s ease-out infinite}
.an-edge>span{display:none}
@keyframes anEdge{0%{box-shadow:0 0 0 0 rgba(26,22,18,.35)}100%{box-shadow:0 0 0 16px rgba(26,22,18,0)}}
.an-anom{width:22px;height:22px;display:flex;align-items:center;justify-content:center;font:700 12px 'JetBrains Mono',monospace;border-radius:50%;border:1.6px solid var(--ink,#1a1612);background:var(--paper,#efe7d4);cursor:help}
.an-anom-surge{color:#b3402f}.an-anom-drop{color:#3b6ea5}.an-anom-new{color:#2d6a4f}
/* ── expanded console ── */
.an-gx{display:grid;grid-template-columns:1fr 300px;gap:14px;height:74vh;min-height:460px}
.an-gx-main{position:relative;display:flex;flex-direction:column;min-width:0}
.an-gx-bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.an-gx .an-seg{display:inline-flex;border:1px solid var(--paper-edge,#d8cdb2);border-radius:8px;overflow:hidden}
.an-gx .an-seg button{font:600 11px 'JetBrains Mono',monospace;border:0;background:var(--paper,#efe7d4);color:var(--ink-soft,#5b5446);padding:5px 9px;cursor:pointer}
.an-gx .an-seg button[aria-pressed=true]{background:var(--ink,#1a1612);color:#efe7d4}
.an-gx-tg{font:600 11px 'JetBrains Mono',monospace;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);color:var(--ink-soft,#5b5446);border-radius:8px;padding:5px 10px;cursor:pointer;transition:all .12s ease}
.an-gx-tg:hover{border-color:var(--ink,#1a1612)}
.an-gx-tg.on{background:var(--accent,#52b788);border-color:var(--accent,#52b788);color:#11241b}
.an-gx-flex{flex:1}
.an-gx-kpi{display:inline-flex;gap:6px;flex-wrap:wrap}
.an-gx-kpi span{font:600 10.5px 'JetBrains Mono',monospace;background:var(--paper-deep,#e7dcc4);color:var(--ink-soft,#5b5446);padding:3px 7px;border-radius:6px}
.an-gx-map{flex:1;position:relative;border-radius:10px;overflow:hidden;border:1px solid var(--paper-edge,#d8cdb2);-webkit-user-select:none;user-select:none;touch-action:none;min-height:240px}
.an-gx-map .maplibregl-canvas{outline:none}
.an-gx-scale{display:flex;align-items:center;gap:8px;margin-top:8px;font:600 10px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}
.an-gx-scale-bar{height:8px;width:150px;border-radius:5px;display:inline-block}
.an-gx-scale-k{display:inline-flex;align-items:center;gap:4px}.an-gx-scale-k i{display:inline-block;width:34px}
.an-gx-insight{margin-top:6px;font:12px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.an-gx-insight b{font-weight:600}.an-gx-dot{margin:0 7px;color:var(--ink-mute,#9a917d)}
.an-gx-scrub{display:flex;flex-direction:column;gap:7px;margin-top:8px;padding:9px 11px;background:var(--paper-deep,#e7dcc4);border-radius:10px}
.an-gx-rphead{display:flex;align-items:center;gap:12px}
.an-gx-clock{font:700 16px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);line-height:1;flex:none;display:flex;flex-direction:column;align-items:flex-start}
.an-gx-clock small{font:9px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);font-weight:400;margin-top:2px}
.an-gx-comm{flex:1;font:12.5px 'Fraunces',Georgia,serif;color:var(--ink-soft,#5b5446);min-width:0}
.an-gx-comm b{color:var(--ink,#1a1612);font-weight:600}
.an-gx-rprow{display:flex;align-items:center;gap:10px}
.an-gx-play{width:30px;height:30px;border-radius:50%;border:1.4px solid var(--ink,#1a1612);background:var(--paper,#efe7d4);cursor:pointer;font-size:11px;flex:none}
.an-gx-bars{flex:1;display:flex;align-items:flex-end;gap:1px;height:30px;overflow:hidden}
.an-gx-bars i{flex:1;min-width:1px;background:var(--paper-edge,#cfc2a3);border-radius:1px 1px 0 0;cursor:pointer;transition:background .15s}
.an-gx-bars i.on{background:var(--accent,#52b788)}
.an-gx-range{flex:none;width:120px;accent-color:var(--accent-deep,#2d6a4f)}
.an-gx-spd{font:600 11px 'JetBrains Mono',monospace;border:1px solid var(--paper-edge,#d8cdb2);border-radius:6px;background:var(--paper,#efe7d4);padding:3px 5px}
/* on-map layer caption (compact card) */
.an-laycap{position:absolute;left:10px;bottom:10px;z-index:2;font:600 10px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);background:rgba(239,231,212,.86);border:1px solid var(--paper-edge,#d8cdb2);border-radius:7px;padding:4px 8px;pointer-events:none;backdrop-filter:blur(2px)}
/* mode-info card + dossier host */
.an-gx-side{background:var(--paper-deep,#e7dcc4);border:1px solid var(--paper-edge,#d8cdb2);border-radius:10px;padding:14px;overflow:auto;display:flex;flex-direction:column;gap:12px}
.an-gx-modecard{background:var(--paper,#efe7d4);border:1px solid var(--paper-edge,#d8cdb2);border-left:3px solid var(--ac,#2d6a4f);border-radius:9px;padding:11px 12px;animation:anGxFade .3s ease}
.an-gx-mode-h{font:600 13px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin-bottom:4px}
.an-gx-mode-d{font:11.5px/1.5 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);margin:0 0 9px}
.an-gx-mode-k{display:flex;gap:7px;flex-wrap:wrap}
.an-gx-mode-k>div{flex:1;min-width:60px;text-align:center;background:var(--paper-deep,#e7dcc4);border-radius:7px;padding:6px 4px}
.an-gx-mode-k b{display:block;font:700 14px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.an-gx-mode-k span{font:9px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-gx-modehint{font:11.5px/1.6 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}
.an-gx-modehint>b{display:block;font:600 12px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);margin-bottom:6px}
.an-gx-modehint ul{list-style:none;margin:0 0 8px;padding:0}
.an-gx-modehint li{margin-bottom:5px;display:flex;align-items:baseline;gap:6px}
.an-gx-modehint li b{color:var(--ink,#1a1612)}
.an-gx-sw{width:9px;height:9px;border-radius:3px;flex:none;display:inline-block;transform:translateY(1px)}
.an-gx-dosshost{flex:1}
@keyframes anGxFade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.an-gx-doss-empty{font:12.5px 'Fraunces',Georgia,serif;color:var(--ink-mute,#9a917d);line-height:1.6}
.an-gx-doss-h{display:flex;align-items:baseline;gap:7px;flex-wrap:wrap;margin-bottom:6px}
.an-gx-doss-h b{font:600 15px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.an-gx-doss-h small{font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-gx-delta{margin-left:auto;font:700 11px 'JetBrains Mono',monospace;padding:2px 7px;border-radius:6px}
.an-gx-delta.is-up{background:#d8efe0;color:#2d6a4f}.an-gx-delta.is-dn{background:#f3ddd6;color:#b3402f}.an-gx-delta.is-new{background:#efe6cf;color:#8a6d1f}
.an-gx-big{font:700 30px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);line-height:1.1;margin:4px 0 12px}
.an-gx-big span{display:block;font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);font-weight:400}
.an-gx-sec{margin:12px 0}
.an-gx-sec>label{display:block;font:600 10px 'JetBrains Mono',monospace;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-mute,#9a917d);margin-bottom:5px}
.an-spark{display:block}
.an-gx-stats{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
.an-gx-stat{flex:1;min-width:64px;background:var(--paper,#efe7d4);border-radius:8px;padding:8px 6px;text-align:center}
.an-gx-stat b{display:block;font:700 16px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.an-gx-stat span{font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-gx-top{display:flex;align-items:center;gap:7px;margin-bottom:4px}
.an-gx-top-k{flex:none;width:118px;font:11px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.an-gx-top-bar{flex:1;height:7px;background:var(--paper,#efe7d4);border-radius:4px;overflow:hidden}
.an-gx-top-bar i{display:block;height:100%;background:var(--accent-deep,#2d6a4f);border-radius:4px}
.an-gx-top-n{flex:none;font:600 10.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);width:34px;text-align:right}
.an-gx-mut{font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-gx-meta{font:10px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin:10px 0 8px}
.an-gx-cta{width:100%;font:600 12px 'JetBrains Mono',monospace;border:1.4px solid var(--ink,#1a1612);background:var(--paper,#efe7d4);color:var(--ink,#1a1612);border-radius:8px;padding:8px;cursor:pointer;transition:all .12s}
.an-gx-cta:hover{background:var(--ink,#1a1612);color:#efe7d4}
@media (max-width:760px){.an-gx{grid-template-columns:1fr;height:auto}.an-gx-map{height:46vh}}
/* compare action button (toolbar) + secondary dossier CTA */
.an-gx-cmp{font:700 11px 'JetBrains Mono',monospace;border:1.4px solid var(--accent-deep,#2d6a4f);background:var(--accent-deep,#2d6a4f);color:#fff;border-radius:8px;padding:5px 11px;cursor:pointer;transition:filter .12s}
.an-gx-cmp:hover{filter:brightness(1.08)}
.an-gx-cta2{margin-top:7px;border-color:var(--accent-deep,#2d6a4f);color:var(--accent-deep,#2d6a4f)}
.an-gx-cta2:hover{background:var(--accent-deep,#2d6a4f);color:#fff}
/* ── COMPARE WINDOW ── */
.an-cmp-root{position:fixed;inset:0;z-index:9600;display:flex;align-items:center;justify-content:center}
.an-cmp-backdrop{position:absolute;inset:0;background:rgba(20,16,12,.5);animation:anGxFade .2s ease}
.an-cmp{position:relative;width:min(1060px,94vw);max-height:90vh;display:flex;flex-direction:column;background:var(--paper,#efe7d4);border:1px solid var(--paper-edge,#d8cdb2);border-radius:14px;box-shadow:0 24px 70px rgba(20,16,12,.32);overflow:hidden;animation:anCmpIn .28s cubic-bezier(.22,1,.36,1)}
@keyframes anCmpIn{from{opacity:0;transform:scale(.96) translateY(10px)}to{opacity:1;transform:none}}
.an-cmp-load{padding:60px;text-align:center;font:13px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-cmp-head{display:flex;align-items:center;gap:16px;padding:16px 20px;border-bottom:1px solid var(--paper-edge,#d8cdb2);background:var(--paper-deep,#e7dcc4);flex:none}
.an-cmp-htitle{flex:none}
.an-cmp-title{font:600 18px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612)}
.an-cmp-sub{font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-top:1px}
.an-cmp-wins{flex:1;display:flex;gap:18px;justify-content:center;flex-wrap:wrap}
.an-cmp-wins span{font:11px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);display:flex;align-items:center;gap:5px}
.an-cmp-wins b{color:var(--ink,#1a1612)}
.an-cmp-wins .dot{width:9px;height:9px;border-radius:3px}
.an-cmp-wins .dot.now{background:var(--accent-deep,#2d6a4f)}
.an-cmp-wins .dot.prev{background:#b9b3a3;border:1px dashed #8a8474}
.an-cmp-x{flex:none;width:30px;height:30px;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);border-radius:8px;cursor:pointer;font-size:13px;color:var(--ink,#1a1612)}
.an-cmp-x:hover{background:var(--ink,#1a1612);color:#efe7d4}
.an-cmp-body{padding:18px 20px 22px;overflow-y:auto}
.an-cmp-summary{display:flex;gap:22px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.an-cmp-totals{display:flex;align-items:center;gap:16px}
.an-cmp-big{text-align:center}
.an-cmp-big b{display:block;font:700 34px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);line-height:1}
.an-cmp-big.alt b{color:var(--ink-mute,#9a917d)}
.an-cmp-big span{font:10px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-cmp-delta{font:700 15px 'JetBrains Mono',monospace;padding:5px 11px;border-radius:9px}
.an-cmp-delta.is-up{background:#d8efe0;color:#2d6a4f}.an-cmp-delta.is-dn{background:#f3ddd6;color:#b3402f}
.an-cmp-cmpbars{flex:1;min-width:200px;display:flex;flex-direction:column;gap:6px}
.an-cmp-cb{display:flex;align-items:center;gap:8px;font:10px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-cmp-cb label{width:30px;flex:none}
.an-cmp-cb span{flex:1;height:13px;background:var(--paper-deep,#e7dcc4);border-radius:4px;overflow:hidden}
.an-cmp-cb i{display:block;height:100%;border-radius:4px;transition:width .5s cubic-bezier(.22,1,.36,1)}
.an-cmp-cb i.now{background:var(--accent-deep,#2d6a4f)}.an-cmp-cb i.prev{background:#b9b3a3}
.an-cmp-mapwrap{margin:6px 0 16px}
.an-cmp-mapwrap>label,.an-cmp-section>label,.an-cmp-trend>label{display:block;font:600 10px 'JetBrains Mono',monospace;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-mute,#9a917d);margin-bottom:6px}
.an-cmp-map{width:100%;height:auto;max-height:240px;border:1px solid var(--paper-edge,#d8cdb2);border-radius:8px;display:block}
.an-cmp-trend{flex:1;min-width:240px}
.an-cmp-cols{display:grid;grid-template-columns:1fr 1fr;gap:14px 22px;margin-bottom:14px}
.an-cmp-col h4{font:600 12px 'Fraunces',Georgia,serif;margin:0 0 8px;padding-bottom:5px;border-bottom:1px solid var(--paper-edge,#d8cdb2)}
.an-cmp-col h4.up{color:#2d6a4f}.an-cmp-col h4.dn{color:#b3402f}.an-cmp-col h4.new{color:#3b6ea5}.an-cmp-col h4.quiet{color:#8a6d1f}
.an-cmp-row{display:flex;align-items:center;gap:10px;padding:4px 0}
.an-cmp-row-k{flex:1;min-width:0;font:12px 'JetBrains Mono',monospace;color:var(--ink,#1a1612);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;flex-direction:column}
.an-cmp-row-k small{font:9.5px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d)}
.an-cmp-row-v{flex:none;font:600 11.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446);white-space:nowrap}
.an-cmp-row-v em{font-style:normal;color:var(--ink-mute,#9a917d);margin-left:3px}
.an-cmp-bar{flex:none;width:60px;height:7px;background:var(--paper-deep,#e7dcc4);border-radius:4px;overflow:hidden}
.an-cmp-bar i{display:block;height:100%;border-radius:4px}
.an-cmp-bar.up i{background:#2d6a4f}.an-cmp-bar.dn i{background:#b3402f}
.an-cmp-none{font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);padding:4px 0}
.an-cmp-section{margin:14px 0}
.an-cmp-stats{display:flex;flex-direction:column;gap:6px}
.an-cmp-stat{display:flex;align-items:baseline;gap:10px}
.an-cmp-stat-l{flex:1;font:11.5px 'JetBrains Mono',monospace;color:var(--ink-soft,#5b5446)}
.an-cmp-stat b{font:700 14px 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);min-width:64px;text-align:right}
.an-cmp-stat-d{flex:none;width:120px;font:10.5px 'JetBrains Mono',monospace;text-align:right}
.an-cmp-stat-d.is-up{color:#2d6a4f}.an-cmp-stat-d.is-dn{color:#b3402f}.an-cmp-stat-d.is-flat{color:var(--ink-mute,#9a917d)}
.an-cmp-allbtn{margin-top:8px;font:600 12px 'JetBrains Mono',monospace;border:1.4px solid var(--ink,#1a1612);background:var(--paper,#efe7d4);color:var(--ink,#1a1612);border-radius:8px;padding:7px 14px;cursor:pointer}
.an-cmp-allbtn:hover{background:var(--ink,#1a1612);color:#efe7d4}
@media (max-width:680px){.an-cmp-cols{grid-template-columns:1fr}.an-cmp-summary{gap:12px}}
/* ── shared interactivity: tooltip · clickable rows/dots/sparks · insights ── */
.an-tip{position:fixed;z-index:9999;pointer-events:none;max-width:220px;background:var(--ink,#1a1612);color:#efe7d4;font:11px 'JetBrains Mono',monospace;line-height:1.45;padding:7px 9px;border-radius:7px;box-shadow:0 4px 16px rgba(0,0,0,.3)}
.an-tip b{color:#8fe0b4}.an-tip i{color:#b9b09c;font-style:normal}.an-tip small{color:#b9b09c}
.an-spk{display:block;border-radius:4px}.an-spk:hover{background:rgba(45,106,79,.05)}
.an-spkwrap{position:relative}
.an-gx-hint{font:9px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);text-transform:none;letter-spacing:0;font-weight:400}
.an-gx-top.is-click{cursor:pointer;border-radius:5px;padding:3px 4px;margin:0 -4px;transition:background .12s}
.an-gx-top.is-click:hover,.an-gx-top.is-click:focus{background:var(--paper,#efe7d4);outline:none}
.an-gx-dossins{font:11.5px/1.55 'Fraunces',Georgia,serif;color:var(--ink-soft,#5b5446);background:var(--paper,#efe7d4);border-left:3px solid var(--accent-deep,#2d6a4f);border-radius:0 7px 7px 0;padding:8px 10px;margin:2px 0 12px}
.an-gx-dossins b{color:var(--ink,#1a1612);font-weight:600}
.an-cmp-insight{font:13px/1.6 'Fraunces',Georgia,serif;color:var(--ink,#1a1612);background:var(--paper-deep,#e7dcc4);border-left:3px solid var(--accent-deep,#2d6a4f);border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:16px}
.an-cmp-insight b{font-weight:600}
.an-cmp-dot{cursor:pointer;transition:fill-opacity .12s}
.an-cmp-dot:hover{fill-opacity:1;stroke-width:1.2}
.an-cmp-row.is-click,.an-cmp-row.is-clickep{cursor:pointer;border-radius:6px;margin:0 -6px;padding:4px 6px;transition:background .12s}
.an-cmp-row.is-click:hover,.an-cmp-row.is-click:focus,.an-cmp-row.is-clickep:hover{background:var(--paper-deep,#e7dcc4);outline:none}
/* region request list (click a dot → requests → drawer) */
.an-rqx-chips{display:flex;gap:6px;margin-bottom:14px}
.an-rqx-chips button{font:600 11px 'JetBrains Mono',monospace;border:1px solid var(--paper-edge,#d8cdb2);background:var(--paper,#efe7d4);color:var(--ink-soft,#5b5446);border-radius:7px;padding:4px 12px;cursor:pointer}
.an-rqx-chips button[aria-pressed=true]{background:var(--ink,#1a1612);color:#efe7d4;border-color:var(--ink,#1a1612)}
.an-rqx-head{display:flex;justify-content:space-between;align-items:center;font:11px 'JetBrains Mono',monospace;color:var(--ink-mute,#9a917d);margin-bottom:8px}
.an-rqx-list{display:flex;flex-direction:column;gap:2px;max-height:64vh;overflow:auto}
.an-rqx{display:grid;grid-template-columns:52px 1fr 46px 64px 40px;align-items:center;gap:10px;padding:7px 9px;border-radius:7px;cursor:pointer;font:12px 'JetBrains Mono',monospace;transition:background .12s}
.an-rqx:hover,.an-rqx:focus{background:var(--paper-deep,#e7dcc4);outline:none}
.an-rqx-m{font-weight:700;font-size:10px}
.an-rqx-path{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink,#1a1612)}
.an-rqx-sc{font-weight:600;text-align:center}
.an-sc-ok{color:#2d6a4f}.an-sc-amber{color:#c98a2b}.an-sc-danger{color:#b3402f}
.an-rqx-lat{text-align:right;color:var(--ink-soft,#5b5446)}
.an-rqx-ago{text-align:right;color:var(--ink-mute,#9a917d)}
`;
    document.head.appendChild(s);
  }

  // ── public API ─────────────────────────────────────────────────────────
  G.mount = mount; G.setData = setData; G.setLayer = setLayer; G.setSource = setSource;
  G.setRange = setRange; G.focusRegion = focusRegion; G.onScan = onScan;
  G.renderExpanded = renderExpanded; G.pause = pause; G.resume = resume;
  G.openCompare = openCompare; G.openRegionRequests = openRegionRequests;
  window.APIN = window.APIN || {};
  window.APIN.analyticsGeo = G;
})();
