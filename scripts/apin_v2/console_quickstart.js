(function(){
  'use strict';

  // Tab switching
  document.querySelectorAll('.tabs').forEach((tabs) => {
    const block = tabs.nextElementSibling;
    tabs.querySelectorAll('.tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        tabs.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        block.querySelectorAll('.lang').forEach((p) => p.classList.remove('active'));
        const target = block.querySelector('.lang[data-lang="' + btn.dataset.lang + '"]');
        if (target) target.classList.add('active');
      });
    });
  });

  // Copy buttons
  document.querySelectorAll('.code-block .copy').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const block = document.getElementById(btn.dataset.copy);
      const active = block.querySelector('.lang.active');
      if (!active) return;
      const txt = active.textContent.replace(/^[A-Z][A-Z ·\-]+\n/, '');
      try {
        await navigator.clipboard.writeText(txt);
        btn.classList.add('copied');
        btn.textContent = 'copied!';
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.textContent = 'copy';
        }, 1500);
      } catch (_) {
        const r = document.createRange();
        r.selectNode(active);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(r);
      }
    });
  });

  // Annotated JSON viewer
  const FIELD_DOCS = {
    api_version: {title: 'api_version', body: '<p>Envelope: which version of the API contract this response uses. Currently <code>1.0</code>. Key your typed client on this if you maintain one.</p>'},
    request_id: {title: 'request_id', body: '<p>Envelope: per-request unique ID (also returned in <code>X-Request-Id</code>). Quote this in support tickets.</p>'},
    endpoint: {title: 'endpoint', body: '<p>Envelope: which route handled the request. Useful when proxies rewrite paths.</p>'},
    ok: {title: 'ok', body: '<p>Envelope: <code>true</code> on success, <code>false</code> on error. Always check before reading <code>data</code>.</p>'},
    processed_at: {title: 'processed_at', body: '<p>Envelope: server-side timestamp (UTC, ISO-8601, ms precision).</p>'},
    processing_time_ms: {title: 'processing_time_ms', body: '<p>Envelope: total request time as seen by the API edge. Includes upload + inference, excludes client-side round-trip.</p>'},
    status_code: {title: 'status_code', body: '<p>Envelope: HTTP status mirrored into the body for proxies that hide it.</p>'},
    data: {title: 'data', body: '<p>Envelope: the actual payload. For <code>/api/predict/full</code> this is a 35-field dict; the highlighted keys are the ones most callers read.</p>'},
    diagnosis: {title: 'data.diagnosis', body: '<p>Top-1 predicted class label. String, e.g. <code>brassica_alternaria</code>. <code>null</code> when <code>tier</code> is <code>ROUTER_REJECTED</code> or any error tier.</p>'},
    confidence: {title: 'data.confidence', body: '<p>Posterior probability for the winning class. Float in [0, 1]. <strong>NOT calibrated</strong>: prefer the length of <code>conformal_prediction_set</code> as a calibrated confidence signal.</p>'},
    tier: {title: 'data.tier', body: '<p>Qualitative bucket. <code>1A</code> (high-confidence specialist), <code>2A</code>/<code>3A</code>/<code>4A</code> (lower confidence, surface alternatives), <code>ROUTER_REJECTED</code>, <code>TOMATO_UNAVAILABLE</code>, <code>TOMATO_INFERENCE_ERROR</code>.</p>'},
    conformal_prediction_set: {title: 'data.conformal_prediction_set', body: '<p>Smallest set of class labels guaranteed to contain the true label with calibrated 90&percnt; probability. Length 1 = model is sure; 2-3 = surface alternatives; &gt;3 = surface uncertainty prominently.</p>'},
    is_ood: {title: 'data.is_ood', body: '<p>Boolean. <code>true</code> when the image is far from the training feature distribution (Mahalanobis-distance based). Treat diagnosis as a hint, not an answer.</p>'},
    uncertainty_aleatoric: {title: 'data.uncertainty_aleatoric', body: '<p>Float in [0, 1]. How ambiguous the IMAGE is: blur, glare, partial leaf, atypical angle. High &rarr; ask the user to retake.</p>'},
    uncertainty_epistemic: {title: 'data.uncertainty_epistemic', body: '<p>Float in [0, 1]. How much the MODEL disagrees with itself across signals. High &rarr; consult an expert; the model knows it doesnt know.</p>'},
    output_message: {title: 'data.output_message', body: '<p>Human-readable summary the website uses verbatim. Pre-formatted for end-user display.</p>'},
    gradcam_b64_png: {title: 'data.gradcam_b64_png', body: '<p>Base64-encoded PNG of the Grad-CAM heatmap overlay, ~200&nbsp;KB typical. Decode with <code>atob</code> + render to canvas to show users which pixels drove the diagnosis.</p>'},
    signal_predictions: {title: 'data.signal_predictions', body: '<p>Per-sub-model breakdown: <code>{model2: {argmax, top_prob}, efficientnet: ..., psv: ..., dinov2_head: ...}</code>. Useful for explainability dashboards.</p>'},
    api_scope: {title: 'data.api_scope', body: '<p>Which crop families this endpoint handles in the current deployment. Today: <code>okra,brassica</code>. Tomato has its own future endpoint.</p>'},
    meta: {title: 'meta', body: '<p>Envelope: server-supplied metadata (warnings, pagination, deprecation notices). May be empty.</p>'},
  };

  const $panel = document.getElementById('resp-panel');
  document.querySelectorAll('#resp-viewer .json-key').forEach((k) => {
    k.addEventListener('click', () => {
      document.querySelectorAll('#resp-viewer .json-key').forEach((x) => x.classList.remove('active'));
      k.classList.add('active');
      const doc = FIELD_DOCS[k.dataset.key];
      if (!doc) {
        $panel.innerHTML = '<h5>(undocumented)</h5><p class="empty">No docs for this field yet.</p>';
        return;
      }
      $panel.innerHTML = '<h5>' + doc.title + '</h5>' + doc.body;
    });
  });

  // TOC scroll-spy
  const tocLinks = [...document.querySelectorAll('.toc a[data-toc-target]')];
  const sections = tocLinks.map(a => document.getElementById(a.dataset.tocTarget));
  function activateToc() {
    let idx = 0;
    for (let i = 0; i < sections.length; i++) {
      const r = sections[i].getBoundingClientRect();
      if (r.top < 140) idx = i;
    }
    tocLinks.forEach((a, i) => a.classList.toggle('active', i === idx));
  }
  window.addEventListener('scroll', activateToc, {passive:true});

})();