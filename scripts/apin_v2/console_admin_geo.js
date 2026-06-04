/* ADM-T · Geography — globe-centric, three lenses on real data.
 *
 *   SCAN      where scan-REQUESTS originate (real GPS) · colour = crop
 *   INFERENCE what the model DETECTED, mapped · colour = disease · severity/conf/OOD
 *   API       developer API origins · honest local/private network + per-host
 *
 * Every page has the Living Globe (window.ADM_GLOBE) — that is the point of the
 * section. All data is real (admin_geo / admin_inference_geo / admin_origins);
 * the API page never invents map pins for private IPs. Uses the window.ADM bridge.
 */
(function () {
  'use strict';
  var A = window.ADM || {};
  var esc = A.esc || function (s) { return String(s == null ? '' : s); };
  var fmtInt = A.fmtInt || function (n) { return String(n); };

  var CROP_LABEL = { okra: 'Okra', brassica: 'Brassica', tomato: 'Tomato', other: 'Other' };
  var CROP_HEX = { okra: '#4ade80', brassica: '#e0b341', tomato: '#e0584a', other: '#7da7ff' };
  var DISEASE_HEX = {
    okra_yvmv: '#e0b341', okra_powdery_mildew: '#cdd6a6', okra_cercospora: '#d98a4a',
    okra_enation: '#c77be0', okra_healthy: '#4ade80',
    brassica_black_rot: '#9b4a4a', brassica_downy_mildew: '#7da7ff',
    brassica_alternaria: '#e0584a', brassica_clubroot: '#b06a3a', brassica_healthy: '#52b788',
  };
  var SEV_HEX = { mild: '#4ade80', moderate: '#e0b341', severe: '#e0584a' };
  var COUNTRY = { IN: 'India', GB: 'United Kingdom', US: 'United States', KE: 'Kenya', LK: 'Sri Lanka',
    BD: 'Bangladesh', NP: 'Nepal', PK: 'Pakistan', AE: 'UAE', SG: 'Singapore', AU: 'Australia',
    CA: 'Canada', DE: 'Germany', FR: 'France', NL: 'Netherlands', PH: 'Philippines', NG: 'Nigeria' };
  function countryName(cc) { return COUNTRY[cc] || cc || 'Unknown'; }
  function dxHex(dx) { return DISEASE_HEX[dx] || '#8a9a93'; }
  function hexInt(h) { return parseInt((h || '#888').slice(1), 16); }
  function dxLabel(dx) { return dx ? dx.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); }) : '—'; }

  var st = { sub: 'scan', globe: null, mounted: false, data: {}, rendered: {} };

  function fetchW(widget) {
    var url = '/api/account/admin/traffic?widget=' + widget + '&window=all';
    if (A.adminFetch) return A.adminFetch(url);
    var csrf = (document.querySelector('meta[name=csrf-token]') || {}).content || '';
    return fetch(url, { credentials: 'include', headers: { 'X-Console-Csrf': csrf } })
      .then(function (r) { return r.json(); }).then(function (j) { return j.data || j; }).catch(function () { return null; });
  }

  // ── floating tooltip ───────────────────────────────────────────────────────
  var _tip = null;
  function tip(html, x, y) {
    if (!_tip) { _tip = document.createElement('div'); _tip.className = 'trf-tip'; document.body.appendChild(_tip); }
    _tip.innerHTML = html; _tip.style.display = 'block';
    var r = _tip.getBoundingClientRect();
    _tip.style.left = Math.min(window.innerWidth - r.width - 8, Math.max(8, x - r.width / 2)) + 'px';
    _tip.style.top = Math.max(8, y - r.height - 14) + 'px';
  }
  function hideTip() { if (_tip) _tip.style.display = 'none'; }

  // ── shared visual bits ───────────────────────────────────────────────────────
  function barList(items, opts) {
    opts = opts || {}; items = items || [];
    if (!items.length) return '<div class="trf-empty">' + (opts.empty || 'no data') + '</div>';
    var max = items.reduce(function (a, b) { return Math.max(a, b.count || 0); }, 0) || 1;
    return '<div class="trf-barlist">' + items.map(function (it) {
      return '<div class="bl-row' + (opts.drill ? ' clickable' : '') + '"' + (opts.drill ? ' data-k="' + esc(opts.id(it)) + '" role="button" tabindex="0"' : '') + '>'
        + '<span class="bl-l">' + esc(opts.label(it)) + (opts.sub ? '<small>' + esc(opts.sub(it)) + '</small>' : '') + '</span>'
        + '<span class="bl-bar"><i style="width:' + Math.round(100 * (it.count || 0) / max) + '%' + (opts.color ? ';background:' + opts.color(it) : '') + '"></i></span>'
        + '<span class="bl-v">' + (opts.fmt ? opts.fmt(it) : fmtInt(it.count || 0)) + '</span></div>';
    }).join('') + '</div>';
  }
  function stat(big, label, tone) { return '<div class="geo-stat' + (tone ? ' ' + tone : '') + '"><b>' + big + '</b><span>' + esc(label) + '</span></div>'; }
  function panel(label, body, extra) { return '<div class="geo-panel"><div class="k-label">' + esc(label) + (extra || '') + '</div>' + body + '</div>'; }

  function sizeCanvas(cv) { var r = cv.getBoundingClientRect(); cv.width = Math.max(360, r.width | 0); cv.height = Math.max(300, r.height | 0); }
  function disposeGlobe() { if (st.globe) { try { st.globe.dispose(); } catch (e) {} st.globe = null; } }

  // globe-card scaffold shared by all three lenses
  function globeCard(title, cov, legend, hint) {
    return '<div class="geo-globe-card"><div class="geo-card-h"><span class="k-label">' + esc(title) + '</span>'
      + (cov ? '<span class="geo-cov">' + cov + '</span>' : '')
      + '<button class="trf-x" data-exp="globe" aria-label="expand">⤢</button></div>'
      + '<div class="geo-stage"><canvas class="geo-globe-cv"></canvas>'
      + '<div class="geo-legend">' + legend + '<span class="geo-hint">' + esc(hint || 'drag to orbit · scroll to zoom') + '</span></div></div></div>';
  }
  function legendChips(items) { return items.map(function (it) { return '<span class="geo-chip"><i style="background:' + it.color + '"></i>' + esc(it.label) + (it.count != null ? ' <b>' + fmtInt(it.count) + '</b>' : '') + '</span>'; }).join(''); }

  // ════════════════════════════════════════════════════════════════════════════
  // SCAN lens — scan-request origins, colour = crop
  // ════════════════════════════════════════════════════════════════════════════
  function renderScan(pane, g) {
    var cov = fmtInt(g.geolocated || 0) + ' / ' + fmtInt(g.total_scans || 0) + ' geolocated · ' + (g.coverage_pct || 0) + '%';
    var legend = legendChips((g.crops || []).map(function (c) { return { color: CROP_HEX[c.crop] || CROP_HEX.other, label: CROP_LABEL[c.crop] || c.crop, count: c.count }; }));
    var side = '<div class="geo-side">'
      + panel('coverage', '<div class="geo-cov-big"><b>' + (g.coverage_pct || 0) + '%</b><span>' + fmtInt(g.geolocated || 0) + ' of ' + fmtInt(g.total_scans || 0) + ' scans carry real GPS</span></div>')
      + panel('by country', barList(g.countries, { label: function (c) { return countryName(c.cc); } }))
      + panel('by crop', barList(g.crops, { label: function (c) { return CROP_LABEL[c.crop] || c.crop; }, color: function (c) { return CROP_HEX[c.crop] || CROP_HEX.other; } }))
      + panel('top districts', barList((g.districts || []).slice(0, 10), { drill: true, id: function (d) { return d.district; }, label: function (d) { return d.district; }, sub: function (d) { return (d.state || '') + ' · ' + dxLabel(d.top_diagnosis); }, color: function (d) { return CROP_HEX[d.crop] || CROP_HEX.other; } }))
      + '</div>';
    pane.innerHTML = '<div class="geo-grid">' + globeCard('scan-origin globe · real GPS coordinates', cov, legend, 'drag to orbit · scroll to zoom · hover a district') + side + '</div>';
    mountGlobe(pane, g, {
      onHover: function (d, x, y) { tip('<b>' + esc(d.district) + '</b><span>' + esc(d.state || '') + ' · ' + fmtInt(d.count) + ' scans · ' + dxLabel(d.top_diagnosis) + ' · conf ' + Math.round((d.avg_confidence || 0) * 100) + '%</span>', x, y); },
    });
    wireSideDrill(pane);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // INFERENCE lens — what the model detected, colour = disease
  // ════════════════════════════════════════════════════════════════════════════
  function renderInference(pane, g) {
    var cov = fmtInt(g.geolocated || 0) + ' / ' + fmtInt(g.total || 0) + ' geolocated · ' + (g.coverage_pct || 0) + '%';
    var topDx = (g.diseases || []).slice(0, 6);
    var legend = legendChips(topDx.slice(0, 5).map(function (d) { return { color: dxHex(d.diagnosis), label: dxLabel(d.diagnosis) }; }));
    var sevTotal = (g.severities || []).reduce(function (a, s) { return a + s.count; }, 0) || 1;
    var sevBar = '<div class="geo-sevbar">' + (g.severities || []).map(function (s) { return '<span style="flex:' + s.count + ';background:' + (SEV_HEX[s.label] || '#888') + '" title="' + esc(s.label) + ' ' + s.count + '" data-sev="' + esc(s.label) + '"></span>'; }).join('') + '</div>'
      + '<div class="geo-sevleg">' + (g.severities || []).map(function (s) { return '<span><i style="background:' + (SEV_HEX[s.label] || '#888') + '"></i>' + esc(s.label) + ' <b>' + s.count + '</b> · ' + Math.round(100 * s.count / sevTotal) + '%</span>'; }).join('') + '</div>';
    var side = '<div class="geo-side">'
      + '<div class="geo-panel"><div class="geo-tot">' + stat((g.avg_confidence != null ? Math.round(g.avg_confidence * 100) + '%' : '—'), 'avg confidence') + stat(fmtInt(g.ood_total || 0), 'OOD flags', (g.ood_total ? 'warn' : '')) + stat(fmtInt((g.diseases || []).length), 'diseases') + '</div></div>'
      + panel('disease distribution', barList(g.diseases, { drill: true, id: function (d) { return d.diagnosis; }, label: function (d) { return dxLabel(d.diagnosis); }, sub: function (d) { return 'conf ' + Math.round((d.avg_confidence || 0) * 100) + '%'; }, color: function (d) { return dxHex(d.diagnosis); } }))
      + panel('severity spectrum', sevBar)
      + panel('top regions by detection', barList((g.districts || []).slice(0, 8), { drill: true, id: function (d) { return d.district; }, label: function (d) { return d.district; }, sub: function (d) { return dxLabel(d.top_diagnosis) + (d.ood ? ' · ' + d.ood + ' OOD' : ''); }, color: function (d) { return dxHex(d.top_diagnosis); } }))
      + '</div>';
    pane.innerHTML = '<div class="geo-grid">' + globeCard('inference globe · disease detected per region', cov, legend, 'colour = dominant disease · hover a region') + side + '</div>';
    mountGlobe(pane, g, {
      colorOf: function (d) { return hexInt(dxHex(d.top_diagnosis)); },
      onHover: function (d, x, y) {
        var sev = d.severity_mix || {}; var sevStr = ['mild', 'moderate', 'severe'].filter(function (k) { return sev[k]; }).map(function (k) { return sev[k] + ' ' + k; }).join(' · ');
        tip('<b>' + esc(d.district) + ' · ' + dxLabel(d.top_diagnosis) + '</b><span>' + fmtInt(d.count) + ' scans · conf ' + Math.round((d.avg_confidence || 0) * 100) + '%' + (d.ood ? ' · ' + d.ood + ' OOD' : '') + (sevStr ? '<br>' + sevStr : '') + '</span>', x, y);
      },
    });
    wireSideDrill(pane);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // API lens — honest origin network (all local/private), globe present + network
  // ════════════════════════════════════════════════════════════════════════════
  function renderApi(pane, o) {
    var pct = o.private_pct || 0;
    var legend = legendChips([{ color: 'var(--ochre)', label: 'private/LAN' }, { color: 'var(--accent)', label: 'public (geolocatable)' }]);
    var side = '<div class="geo-side">'
      + '<div class="geo-panel"><div class="geo-tot">' + stat(fmtInt(o.total || 0), 'requests') + stat(pct + '%', 'private', 'warn') + stat(fmtInt((o.hosts || []).length), 'hosts') + '</div>'
      + '<div class="geo-note">' + (pct >= 99.9 ? 'Every request comes from the local / private network (localhost + LAN). These IPs are <b>not geolocatable</b> — the globe shows no public origins by design.' : pct + '% of traffic is local/private; only public IPs can be placed on the globe.') + '</div></div>'
      + panel('origin composition · over time', compositionChart(o.series))
      + panel('per-host requests', hostTable(o.hosts))
      + '</div>';
    pane.innerHTML = '<div class="geo-grid">' + globeCard('api origin globe · ' + (o.public || 0) + ' public · ' + (o.private || 0) + ' local', null, legend, 'all origins local — network detail at right')
      + side + '</div>';
    // globe shows borders only (no real geo origins) + honest overlay
    mountGlobe(pane, { districts: [] }, {});
    var stage = pane.querySelector('.geo-stage');
    if (stage) { var ov = document.createElement('div'); ov.className = 'geo-api-ov'; ov.innerHTML = '<div class="geo-net">' + nodeGraph(o) + '</div><div class="geo-api-cap">internal origin network · ' + fmtInt(o.total || 0) + ' requests, all local</div>'; stage.appendChild(ov); }
    wireApiSide(pane, o);
  }

  // stacked area of origin composition over time (SVG)
  function compositionChart(series) {
    series = (series || []).filter(function (s) { return (s.loopback + s.lan + s.public) > 0; });
    if (!series.length) return '<div class="trf-empty">no traffic in window</div>';
    var w = 280, h = 70, n = series.length, dx = w / (n - 1 || 1);
    var max = series.reduce(function (a, s) { return Math.max(a, s.loopback + s.lan + s.public); }, 1);
    function band(key, base) {
      var top = series.map(function (s, i) { return [i * dx, h - (base(s) + s[key]) / max * (h - 4)]; });
      var bot = series.map(function (s, i) { return [i * dx, h - base(s) / max * (h - 4)]; }).reverse();
      var pts = top.concat(bot).map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ');
      return pts;
    }
    var bLoop = function () { return 0; };
    var bLan = function (s) { return s.loopback; };
    var bPub = function (s) { return s.loopback + s.lan; };
    return '<svg class="geo-comp" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">'
      + '<polygon points="' + band('loopback', bLoop) + '" fill="#c98b32" opacity="0.85"></polygon>'
      + '<polygon points="' + band('lan', bLan) + '" fill="#e0b341" opacity="0.7"></polygon>'
      + '<polygon points="' + band('public', bPub) + '" fill="#4ade80" opacity="0.8"></polygon></svg>'
      + '<div class="geo-comp-leg"><span><i style="background:#c98b32"></i>localhost</span><span><i style="background:#e0b341"></i>LAN</span><span><i style="background:#4ade80"></i>public</span></div>';
  }

  // internal network node-graph (localhost hub + LAN hosts) — SVG
  function nodeGraph(o) {
    var hosts = (o.hosts || []).slice(0, 9);
    if (!hosts.length) return '';
    var W = 240, H = 240, cx = W / 2, cy = H / 2;
    var hub = hosts.find(function (h) { return h.class === 'loopback'; }) || hosts[0];
    var ring = hosts.filter(function (h) { return h !== hub; });
    var max = hosts.reduce(function (a, h) { return Math.max(a, h.count || 0); }, 1);
    function rad(h) { return 6 + 16 * Math.sqrt((h.count || 0) / max); }
    var edges = '', nodes = '';
    ring.forEach(function (h, i) {
      var ang = (i / Math.max(1, ring.length)) * Math.PI * 2 - Math.PI / 2;
      var x = cx + Math.cos(ang) * 88, y = cy + Math.sin(ang) * 88;
      h._x = x; h._y = y;
      edges += '<line x1="' + cx + '" y1="' + cy + '" x2="' + x.toFixed(1) + '" y2="' + y.toFixed(1) + '" class="gn-edge"/>';
      var col = h.class === 'public' ? '#4ade80' : '#e0b341';
      nodes += '<g class="gn-node" data-ip="' + esc(h.ip) + '"><circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + rad(h).toFixed(1) + '" fill="' + col + '"/>'
        + '<text x="' + x.toFixed(1) + '" y="' + (y + rad(h) + 9).toFixed(1) + '" class="gn-label">' + esc(h.ip.replace(/^10\.16\./, '·')) + '</text></g>';
    });
    nodes += '<g class="gn-node" data-ip="' + esc(hub.ip) + '"><circle cx="' + cx + '" cy="' + cy + '" r="' + rad(hub).toFixed(1) + '" fill="#7da7ff"/>'
      + '<text x="' + cx + '" y="' + (cy + rad(hub) + 10).toFixed(1) + '" class="gn-label gn-hub">localhost</text></g>';
    return '<svg class="geo-nodegraph" viewBox="0 0 ' + W + ' ' + H + '">' + edges + nodes + '</svg>';
  }

  function hostTable(hosts) {
    hosts = hosts || [];
    if (!hosts.length) return '<div class="trf-empty">no hosts</div>';
    var max = hosts.reduce(function (a, h) { return Math.max(a, h.count || 0); }, 1);
    return '<div class="geo-hosts">' + hosts.map(function (h) {
      var ec = (h.error_rate || 0) >= 20 ? 'bad' : ((h.error_rate || 0) >= 5 ? 'warn' : '');
      var top = (h.top_paths || [])[0];
      return '<div class="geo-host clickable" data-ip="' + esc(h.ip) + '" role="button" tabindex="0">'
        + '<span class="gh-cls gh-' + h.class + '"></span>'
        + '<span class="gh-ip mono">' + esc(h.ip) + '</span>'
        + '<span class="gh-bar"><i style="width:' + Math.round(100 * (h.count || 0) / max) + '%"></i></span>'
        + '<span class="gh-n mono">' + fmtInt(h.count) + '</span>'
        + '<span class="gh-e mono ' + ec + '">' + (h.error_rate || 0) + '%</span>'
        + (top ? '<span class="gh-top mono">' + esc(top.path) + '</span>' : '') + '</div>';
    }).join('') + '</div>';
  }

  // ── globe mount (shared) ─────────────────────────────────────────────────────
  function mountGlobe(pane, data, gopts) {
    var cv = pane.querySelector('.geo-globe-cv'); if (!cv) return;
    sizeCanvas(cv);
    if (window.ADM_GLOBE && window.ADM_GLOBE.ok) {
      gopts = gopts || {};
      gopts.W = cv.width; gopts.H = cv.height; gopts.onLeave = hideTip;
      gopts.onClick = function (d) { if (A.openDrillList) A.openDrillList('scans', { district: d.district }, d.district); };
      st.globe = window.ADM_GLOBE.create(cv, data, gopts);
    }
    if (!st.globe) { cv.style.display = 'none'; }
  }

  function wireSideDrill(pane) {
    var side = pane.querySelector('.geo-side'); if (!side) return;
    side.addEventListener('mousemove', function (e) { var r = e.target.closest && e.target.closest('.bl-row.clickable[data-k]'); if (!r) { hideTip(); return; } });
    side.addEventListener('mouseleave', hideTip);
    side.addEventListener('click', function (e) {
      var r = e.target.closest && e.target.closest('.bl-row.clickable[data-k]'); if (!r || !A.openDrillList) return;
      var k = r.getAttribute('data-k');
      // disease keys look like okra_*/brassica_*; districts are plain names
      if (/^(okra|brassica|tomato)_/.test(k)) A.openDrillList('scans', { diagnosis: k }, dxLabel(k));
      else A.openDrillList('scans', { district: k }, k);
    });
  }

  function wireApiSide(pane, o) {
    var side = pane.querySelector('.geo-side');
    if (side) side.addEventListener('click', function (e) {
      var h = e.target.closest && e.target.closest('.geo-host.clickable[data-ip]');
      if (h && A.openDrillList) A.openDrillList('requests', { ip: h.getAttribute('data-ip') }, h.getAttribute('data-ip'));
    });
    // node-graph hover/click
    var net = pane.querySelector('.geo-net');
    if (net) {
      net.addEventListener('mousemove', function (e) {
        var g = e.target.closest && e.target.closest('.gn-node[data-ip]'); if (!g) { hideTip(); return; }
        var ip = g.getAttribute('data-ip'); var h = (o.hosts || []).find(function (x) { return x.ip === ip; });
        if (h) tip('<b>' + esc(h.ip) + '</b><span>' + fmtInt(h.count) + ' req · ' + (h.error_rate || 0) + '% err · ' + h.class + '</span>', e.clientX, e.clientY);
      });
      net.addEventListener('mouseleave', hideTip);
      net.addEventListener('click', function (e) {
        var g = e.target.closest && e.target.closest('.gn-node[data-ip]');
        if (g && A.openDrillList) A.openDrillList('requests', { ip: g.getAttribute('data-ip') }, g.getAttribute('data-ip'));
      });
    }
    var x = pane.querySelector('.trf-x[data-exp=globe]'); if (x) x.addEventListener('click', openTheatre);
  }

  // ── expand theatre (re-mounts a fresh globe for the active lens) ──────────────
  function openTheatre() {
    var back = document.createElement('div'); back.className = 'trf-lb-back';
    var lb = document.createElement('div'); lb.className = 'trf-lb trf-lb-terrain';
    var ttl = st.sub === 'inference' ? 'Inference Globe · disease detected per region' : (st.sub === 'api' ? 'API Origin Network' : 'Scan-origin Globe');
    lb.innerHTML = '<div class="trf-lb-h"><b>' + esc(ttl) + '</b><button class="trf-lb-x" aria-label="close">×</button></div>'
      + '<div class="trf-lb-b"><div class="geo-stage-lg"><canvas class="geo-globe-cv-lg"></canvas></div>'
      + '<p class="trf-lb-note">' + (st.sub === 'api' ? 'API request origins are private/local (localhost + LAN) and are deliberately not geolocated — the network is shown by host instead.' : 'Every marker is a real scan GPS coordinate resolved to its district. Country + state borders are Natural Earth geometry.') + '</p></div>';
    document.body.appendChild(back); document.body.appendChild(lb);
    requestAnimationFrame(function () { back.classList.add('on'); lb.classList.add('on'); });
    var inst = null;
    setTimeout(function () {
      var cv = lb.querySelector('.geo-globe-cv-lg'); if (!cv) return;
      var r = cv.getBoundingClientRect(); cv.width = Math.max(480, r.width | 0); cv.height = Math.max(360, r.height | 0);
      var data = st.sub === 'inference' ? st.data.inference : (st.sub === 'api' ? { districts: [] } : st.data.scan);
      var gopts = { W: cv.width, H: cv.height, onLeave: hideTip };
      if (st.sub === 'inference') { gopts.colorOf = function (d) { return hexInt(dxHex(d.top_diagnosis)); }; gopts.onHover = function (d, x, y) { tip('<b>' + esc(d.district) + ' · ' + dxLabel(d.top_diagnosis) + '</b><span>' + fmtInt(d.count) + ' scans</span>', x, y); }; }
      else gopts.onHover = function (d, x, y) { tip('<b>' + esc(d.district) + '</b><span>' + fmtInt(d.count) + ' scans · ' + dxLabel(d.top_diagnosis) + '</span>', x, y); };
      gopts.onClick = function (d) { if (A.openDrillList) A.openDrillList('scans', { district: d.district }, d.district); };
      if (window.ADM_GLOBE && window.ADM_GLOBE.ok) inst = window.ADM_GLOBE.create(cv, data || { districts: [] }, gopts);
    }, 60);
    function close() { back.classList.remove('on'); lb.classList.remove('on'); if (inst) { try { inst.dispose(); } catch (e) {} } setTimeout(function () { back.remove(); lb.remove(); }, 260); }
    back.addEventListener('click', close);
    lb.querySelector('.trf-lb-x').addEventListener('click', close);
    document.addEventListener('keydown', function ek(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', ek); } });
  }

  // ── sub-nav + lifecycle ──────────────────────────────────────────────────────
  function tab(sub, label) { return '<button type="button" class="trf-tab' + (st.sub === sub ? ' on' : '') + '" data-sub="' + sub + '" role="tab">' + esc(label) + '</button>'; }
  function shell() {
    return '<div class="trf-subnav geo-subnav" id="geo-subnav" role="tablist">'
      + tab('scan', 'Scan') + tab('inference', 'Inference') + tab('api', 'API')
      + '<span class="trf-tab-ink" id="geo-ink"></span></div>'
      + '<div class="geo-pane" id="geo-p-scan"></div>'
      + '<div class="geo-pane" id="geo-p-inference" hidden></div>'
      + '<div class="geo-pane" id="geo-p-api" hidden></div>';
  }
  function moveInk() {
    var nav = document.getElementById('geo-subnav'), ink = document.getElementById('geo-ink'), on = nav && nav.querySelector('.trf-tab.on');
    if (nav && ink && on) { ink.style.width = on.offsetWidth + 'px'; ink.style.transform = 'translateX(' + on.offsetLeft + 'px)'; }
  }
  function showSub() {
    ['scan', 'inference', 'api'].forEach(function (s) { var p = document.getElementById('geo-p-' + s); if (p) p.hidden = s !== st.sub; });
    moveInk();
    if (st.rendered[st.sub]) return;       // already built → instant
    st.rendered[st.sub] = true;
    disposeGlobe();                         // one globe at a time
    var pane = document.getElementById('geo-p-' + st.sub);
    pane.innerHTML = '<div class="geo-loading">loading ' + st.sub + ' lens…</div>';
    if (st.sub === 'scan') {
      (st.data.scan ? Promise.resolve(st.data.scan) : fetchW('geo')).then(function (g) { st.data.scan = g = g || {}; if ((g.districts || []).length || g.total_scans) renderScan(pane, g); else pane.innerHTML = '<div class="trf-empty">no geolocated scans yet.</div>'; });
    } else if (st.sub === 'inference') {
      (st.data.inference ? Promise.resolve(st.data.inference) : fetchW('inference_geo')).then(function (g) { st.data.inference = g = g || {}; renderInference(pane, g); });
    } else {
      (st.data.origins ? Promise.resolve(st.data.origins) : fetchW('origins')).then(function (o) { st.data.origins = o = o || {}; renderApi(pane, o); });
    }
  }
  function wireNav() {
    var nav = document.getElementById('geo-subnav');
    if (nav) nav.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('.trf-tab[data-sub]'); if (!b) return;
      [].forEach.call(nav.querySelectorAll('.trf-tab'), function (x) { x.classList.toggle('on', x === b); });
      // moving to a different lens: the globe belongs to whichever pane is visible,
      // so re-mount happens inside renderX. Dispose current first.
      st.sub = b.getAttribute('data-sub');
      // a cached pane keeps its DOM but its globe was disposed on leave → re-mount
      if (st.rendered[st.sub]) { st.rendered[st.sub] = false; }
      showSub();
    });
    window.addEventListener('resize', moveInk);
  }

  window.ADM_GEO = {
    mount: function () {
      var body = document.getElementById('geo-body'); if (!body) return;
      st.mounted = true; st.rendered = {};
      body.innerHTML = shell();
      wireNav(); showSub();
    },
    dispose: function () { st.mounted = false; disposeGlobe(); hideTip(); },
  };
})();
