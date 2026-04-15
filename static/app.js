// static/app.js
// UI with: top-2 display, confidence language, try-again for OOD,
// crop probabilities, guided photography, download report, feedback pipeline

const CLASS_NAMES = [
  'okra_yvmv','okra_powdery_mildew','okra_cercospora','okra_enation','okra_healthy',
  'brassica_black_rot','brassica_downy_mildew','brassica_alternaria',
  'brassica_clubroot','brassica_healthy',
  'tomato_bacterial_spot','tomato_early_blight','tomato_late_blight',
  'tomato_leaf_mold','tomato_septoria_leaf_spot','tomato_target_spot',
  'tomato_mosaic_virus','tomato_yellow_leaf_curl_virus','tomato_healthy',
  'chilli_anthracnose','chilli_cercospora_leaf_spot','chilli_leaf_curl',
  'chilli_healthy'
];

const CROP_DISPLAY = {
  'okra': 'Okra (Ladies Finger)',
  'brassica': 'Broccoli',
  'tomato': 'Tomato',
  'chilli': 'Chilli',
};

function formatClassName(cls) {
  if (!cls) return 'Unknown';
  return cls.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function getConfidenceLanguage(confidence, diseases) {
  if (!diseases || diseases.length === 0) return '';
  const name = formatClassName(diseases[0]);
  if (confidence > 0.70) return 'This leaf shows ' + name;
  if (confidence > 0.50) return 'This leaf likely has ' + name;
  if (confidence > 0.35) return 'This leaf may have ' + name + ' — consider consulting an expert';
  return 'Diagnosis uncertain — please try a clearer photo or consult an expert';
}

// ── State ─────────────────────────────────────────────────────────────────
let currentResult = null;
let currentFile = null;

// ── Element refs ──────────────────────────────────────────────────────────
const uploadArea      = document.getElementById('upload-area');
const fileInput       = document.getElementById('file-input');
const previewSection  = document.getElementById('preview-section');
const previewImg      = document.getElementById('preview-img');
const changeBtn       = document.getElementById('change-btn');
const uploadSection   = document.getElementById('upload-section');
const resultSection   = document.getElementById('result-section');
const spinner         = document.getElementById('spinner');
const errorBox        = document.getElementById('error-box');

const heatmapImg      = document.getElementById('heatmap-img');
const cropBadge       = document.getElementById('crop-badge');
const severityBadge   = document.getElementById('severity-badge');
const urgencyBadge    = document.getElementById('urgency-badge');
const oodBadge        = document.getElementById('ood-badge');
const diseaseHeading  = document.getElementById('disease-heading');
const confidenceText  = document.getElementById('confidence-text');
const confidenceBar   = document.getElementById('confidence-bar');
const treatmentList   = document.getElementById('treatment-list');
const preventionList  = document.getElementById('prevention-list');

const thumbsUpBtn         = document.getElementById('thumbs-up-btn');
const thumbsDownBtn       = document.getElementById('thumbs-down-btn');
const correctionForm      = document.getElementById('correction-form');
const correctionSelect    = document.getElementById('correction-select');
const submitCorrectionBtn = document.getElementById('submit-correction-btn');
const feedbackThanks      = document.getElementById('feedback-thanks');

// ── Upload handling ────────────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
uploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  uploadArea.style.background = '#e8f5ee';
});
uploadArea.addEventListener('dragleave', () => {
  uploadArea.style.background = '';
});
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.style.background = '';
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) processFile(fileInput.files[0]);
});
changeBtn.addEventListener('click', resetToUpload);

function resetToUpload() {
  fileInput.value     = '';
  previewSection.hidden = true;
  resultSection.hidden  = true;
  errorBox.hidden       = true;
  uploadSection.hidden  = false;
  currentResult = null;
  currentFile = null;
}

function processFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewSection.hidden = false;
  uploadSection.hidden  = true;
  resultSection.hidden  = true;
  errorBox.hidden       = true;
  submitImage(file, false);
}

// ── API call ───────────────────────────────────────────────────────────────
async function submitImage(file, forcePredict) {
  spinner.hidden = false;
  errorBox.hidden = true;

  const fd = new FormData();
  fd.append('file', file);

  const url = forcePredict ? '/predict?force_predict=true' : '/predict';

  try {
    const resp = await fetch(url, { method: 'POST', body: fd });
    const data = await resp.json();

    if (!resp.ok) {
      showError(data.detail || 'Prediction failed. Please try another image.');
      return;
    }

    currentResult = data;
    renderResult(data);
  } catch (err) {
    showError('Network error — is the server running?');
  } finally {
    spinner.hidden = true;
  }
}

// ── Render ─────────────────────────────────────────────────────────────────
function renderResult(d) {
  // ── OOD handling with Try Again button ──────────────────────────────────
  if (d.ood_flagged) {
    document.getElementById('heatmap-container').hidden = true;
    document.getElementById('badges').hidden = true;
    diseaseHeading.textContent = '';
    confidenceText.textContent = '';
    confidenceBar.style.width = '0%';
    treatmentList.innerHTML = '';
    preventionList.innerHTML = '';
    document.getElementById('treatment-details').hidden = true;
    document.getElementById('prevention-details').hidden = true;
    document.getElementById('feedback-section').hidden = true;

    // Show OOD message with crop probabilities and Try Again button
    let oodHtml = '<div class="ood-message">';
    oodHtml += '<p>' + (d.ood_reason || 'Image not recognized as a supported crop leaf.') + '</p>';

    // Show crop probabilities if available
    if (d.crop_probabilities) {
      oodHtml += '<p style="margin-top:8px;font-size:0.85rem;color:var(--color-subtle)">Crop probabilities: ';
      const probs = Object.entries(d.crop_probabilities)
        .sort((a,b) => b[1] - a[1])
        .map(([crop, conf]) => (CROP_DISPLAY[crop] || crop) + ' ' + Math.round(conf*100) + '%');
      oodHtml += probs.join(', ') + '</p>';
    }

    oodHtml += '<button id="try-again-btn" class="btn-primary" style="margin-top:12px">';
    oodHtml += 'Try Again (Force Predict)</button>';
    oodHtml += '</div>';

    oodBadge.hidden = false;
    oodBadge.innerHTML = oodHtml;
    oodBadge.className = 'badge-ood-block';

    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Attach Try Again handler
    document.getElementById('try-again-btn').addEventListener('click', () => {
      if (currentFile) {
        submitImage(currentFile, true);
      }
    });
    return;
  }

  // ── Normal result ───────────────────────────────────────────────────────
  // Clear any previous OOD message
  oodBadge.hidden = true;
  oodBadge.innerHTML = '';
  oodBadge.className = 'badge badge-ood';

  document.getElementById('heatmap-container').hidden = false;
  document.getElementById('badges').hidden = false;
  document.getElementById('treatment-details').hidden = false;
  document.getElementById('prevention-details').hidden = false;
  document.getElementById('feedback-section').hidden = false;

  // Heatmap
  if (d.heatmap_b64) {
    heatmapImg.src = 'data:image/png;base64,' + d.heatmap_b64;
    heatmapImg.parentElement.hidden = false;
  } else {
    heatmapImg.parentElement.hidden = true;
  }

  // Crop badge
  const cropDisplay = CROP_DISPLAY[d.crop] || d.crop;
  cropBadge.textContent = cropDisplay;

  // Severity badge
  const sev = (d.severity || 'mild').toLowerCase();
  severityBadge.textContent = sev.charAt(0).toUpperCase() + sev.slice(1);
  severityBadge.className = 'badge badge-' + sev;

  // Urgency badge
  const urg = (d.urgency || 'Low').toLowerCase();
  urgencyBadge.textContent = d.urgency + ' Urgency';
  urgencyBadge.className = 'badge badge-' + urg;

  // OOD badge hidden
  oodBadge.hidden = true;

  // Disease heading with confidence language (Issue 17: handle co-infections)
  const diseases = d.diseases || ['unknown'];
  if (diseases.length > 1) {
    // Co-infection: show both disease names with confidence qualifier
    const names = diseases.map(formatClassName).join(' + ');
    if (d.confidence > 0.70) diseaseHeading.textContent = 'This leaf shows ' + names;
    else if (d.confidence > 0.50) diseaseHeading.textContent = 'This leaf likely has ' + names;
    else diseaseHeading.textContent = 'This leaf may have ' + names;
  } else {
    const confLang = getConfidenceLanguage(d.confidence, diseases);
    diseaseHeading.textContent = confLang || formatClassName(diseases[0]);
  }

  // Confidence with traffic light
  const pct = Math.round((d.confidence || 0) * 100);
  const uncPct = Math.round((d.uncertainty || 0) * 100);
  let confIcon = '';
  if (d.confidence_level === 'high') confIcon = '🟢';
  else if (d.confidence_level === 'moderate') confIcon = '🟡';
  else confIcon = '🔴';
  confidenceText.textContent = confIcon + ' Confidence: ' + pct + '%  ·  Uncertainty: ' + uncPct + '%';
  confidenceBar.style.width = pct + '%';

  // Top-2 predictions
  const top2El = document.getElementById('top2-predictions');
  if (top2El && d.top2_diseases && d.top2_diseases.length > 0) {
    let top2Html = '<p style="font-size:0.85rem;color:var(--color-subtle);margin-top:6px">Top predictions: ';
    top2Html += d.top2_diseases.map(t =>
      formatClassName(t.class) + ' (' + Math.round(t.confidence * 100) + '%)'
    ).join(' · ');
    top2Html += '</p>';
    top2El.innerHTML = top2Html;
    top2El.hidden = false;
  }

  // Uncertainty tooltip
  const uncTip = document.getElementById('uncertainty-tooltip');
  if (uncTip) {
    uncTip.title = 'Uncertainty reflects how consistent the model is across multiple predictions. Low = confident. High = consider consulting an expert.';
  }

  // Treatment
  treatmentList.innerHTML = '';
  (d.treatment || []).forEach(t => {
    const li = document.createElement('li');
    li.textContent = t;
    treatmentList.appendChild(li);
  });

  // Prevention
  preventionList.innerHTML = '';
  (d.prevention || []).forEach(p => {
    const li = document.createElement('li');
    li.textContent = p;
    preventionList.appendChild(li);
  });

  // Reset feedback
  correctionForm.hidden = true;
  feedbackThanks.hidden = true;
  thumbsUpBtn.disabled  = false;
  thumbsDownBtn.disabled= false;

  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Feedback ───────────────────────────────────────────────────────────────
thumbsUpBtn.addEventListener('click', async () => {
  if (!currentResult) return;
  thumbsUpBtn.disabled  = true;
  thumbsDownBtn.disabled= true;
  await sendFeedback({ thumbs_up: true, crop: currentResult.crop,
                       diseases: currentResult.diseases });
  feedbackThanks.textContent = 'Thank you! Your feedback helps improve the model.';
  feedbackThanks.hidden = false;
});

thumbsDownBtn.addEventListener('click', () => {
  thumbsDownBtn.disabled = true;
  thumbsUpBtn.disabled   = true;
  correctionSelect.innerHTML = '<option value="">-- Select correct class --</option>';
  CLASS_NAMES.forEach(cls => {
    const opt = document.createElement('option');
    opt.value = cls;
    opt.textContent = formatClassName(cls);
    correctionSelect.appendChild(opt);
  });
  correctionForm.hidden = false;
});

submitCorrectionBtn.addEventListener('click', async () => {
  const correction = correctionSelect.value;
  if (!correction) {
    alert('Please select the correct class.');
    return;
  }
  await sendFeedback({
    thumbs_up  : false,
    crop       : currentResult ? currentResult.crop : '',
    diseases   : currentResult ? currentResult.diseases : [],
    correction : correction,
  });
  correctionForm.hidden = true;
  feedbackThanks.textContent = 'Thank you! Your photo and correction have been saved to improve the model.';
  feedbackThanks.hidden = false;
});

async function sendFeedback(payload) {
  try {
    await fetch('/feedback', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(payload),
    });
  } catch (e) {
    // silent
  }
}

// ── Download Report (Item 27) ──────────────────────────────────────────────
function downloadReport() {
  if (!currentResult) return;
  const d = currentResult;
  const diseases = (d.diseases || []).map(formatClassName).join(', ');
  const crop = CROP_DISPLAY[d.crop] || d.crop || 'Unknown';
  const now = new Date().toLocaleString();

  let report = 'PLANT DISEASE DIAGNOSIS REPORT\n';
  report += '=' .repeat(40) + '\n';
  report += 'Date: ' + now + '\n';
  report += 'Crop: ' + crop + '\n';
  report += 'Disease(s): ' + diseases + '\n';
  report += 'Confidence: ' + Math.round((d.confidence||0)*100) + '%\n';
  report += 'Severity: ' + (d.severity || 'N/A') + '\n';
  report += 'Urgency: ' + (d.urgency || 'N/A') + '\n\n';

  report += 'TREATMENT\n' + '-'.repeat(20) + '\n';
  (d.treatment || []).forEach((t,i) => { report += (i+1) + '. ' + t + '\n'; });

  report += '\nPREVENTION\n' + '-'.repeat(20) + '\n';
  (d.prevention || []).forEach((p,i) => { report += (i+1) + '. ' + p + '\n'; });

  report += '\n---\nGenerated by Plant Disease Detector\n';

  const blob = new Blob([report], {type: 'text/plain'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'diagnosis_report_' + Date.now() + '.txt';
  a.click();
}

// ── Error display ──────────────────────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden      = false;
  spinner.hidden       = true;
  previewSection.hidden= false;
  uploadSection.hidden = true;
}
