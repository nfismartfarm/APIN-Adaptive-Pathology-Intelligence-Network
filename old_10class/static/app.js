// static/app.js

const CLASS_NAMES = [
  'okra_yvmv','okra_powdery_mildew','okra_cercospora','okra_enation','okra_healthy',
  'brassica_black_rot','brassica_downy_mildew','brassica_alternaria',
  'brassica_clubroot','brassica_healthy'
];

// ── State ─────────────────────────────────────────────────────────────────
let currentResult = null;

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

// Result elements
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

// Feedback elements
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
}

function processFile(file) {
  // Show preview
  const url         = URL.createObjectURL(file);
  previewImg.src    = url;
  previewSection.hidden = false;
  uploadSection.hidden  = true;
  resultSection.hidden  = true;
  errorBox.hidden       = true;
  // Submit
  submitImage(file);
}

// ── API call ───────────────────────────────────────────────────────────────
async function submitImage(file) {
  spinner.hidden = false;
  errorBox.hidden = true;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
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
  // Heatmap
  if (d.heatmap_b64) {
    heatmapImg.src              = 'data:image/png;base64,' + d.heatmap_b64;
    heatmapImg.parentElement.hidden = false;
  } else {
    heatmapImg.parentElement.hidden = true;
  }

  // Crop badge
  cropBadge.textContent = d.crop.toUpperCase();

  // Severity badge
  const sev = (d.severity || 'mild').toLowerCase();
  severityBadge.textContent = sev.charAt(0).toUpperCase() + sev.slice(1);
  severityBadge.className   = 'badge badge-' + sev;

  // Urgency badge
  const urg = (d.urgency || 'Low').toLowerCase();
  urgencyBadge.textContent = d.urgency + ' Urgency';
  urgencyBadge.className   = 'badge badge-' + urg;

  // OOD
  oodBadge.hidden = !d.ood_flagged;

  // Disease heading
  const diseases = d.diseases || ['unknown'];
  diseaseHeading.textContent = diseases
    .map(c => c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))
    .join(' + ');

  // Confidence
  const pct = Math.round((d.confidence || 0) * 100);
  confidenceText.textContent =
    `Confidence: ${pct}%  ·  Uncertainty: ${Math.round((d.uncertainty || 0) * 100)}%`;
  confidenceBar.style.width = pct + '%';

  // Per-class probabilities
  const probsContainer = document.getElementById('probs-container');
  if (probsContainer && d.all_probabilities) {
    probsContainer.innerHTML = '';
    const sorted = Object.entries(d.all_probabilities).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([cls, prob]) => {
      const pctVal = Math.round(prob * 100);
      const color = cls.includes('healthy') ? '#27ae60' :
                    pctVal > 50 ? '#c0392b' : pctVal > 20 ? '#e67e22' : '#95a5a6';
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = '<span class="prob-name">' + cls.replace(/_/g, ' ') + '</span>' +
        '<div class="prob-bar-bg"><div class="prob-bar" style="width:' + pctVal +
        '%;background:' + color + '"></div></div>' +
        '<span class="prob-pct">' + pctVal + '%</span>';
      probsContainer.appendChild(row);
    });
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

  // Show result section
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Feedback ───────────────────────────────────────────────────────────────
// [FIX GAP 50] Thumbs-up: send positive feedback and show thanks
thumbsUpBtn.addEventListener('click', async () => {
  if (!currentResult) return;
  thumbsUpBtn.disabled  = true;
  thumbsDownBtn.disabled= true;
  await sendFeedback({ thumbs_up: true, crop: currentResult.crop,
                       diseases: currentResult.diseases });
  feedbackThanks.hidden = false;
});

// [FIX GAP 49] Thumbs-down: reveal correction form with class options
thumbsDownBtn.addEventListener('click', () => {
  thumbsDownBtn.disabled = true;
  thumbsUpBtn.disabled   = true;
  // Populate correction select with all class names
  correctionSelect.innerHTML = '<option value="">— Select correct class —</option>';
  CLASS_NAMES.forEach(cls => {
    const opt = document.createElement('option');
    opt.value       = cls;
    opt.textContent = cls.replace(/_/g, ' ');
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
    // Feedback failure is silent — never show error for this
  }
}

// ── Error display ──────────────────────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden      = false;
  spinner.hidden       = true;
  previewSection.hidden= false;
  uploadSection.hidden = true;
}
