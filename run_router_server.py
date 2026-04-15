"""
Router Model Server — Crop classifier on localhost:8003
Upload a leaf image → identifies okra / brassica / tomato / chilli with confidence.

Usage: python run_router_server.py
Open: http://localhost:8003
"""

import os
import sys
import io
import json
import time

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))

from app.config_router import (
    BACKBONE_NAME, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM,
    NUM_CLASSES, CLASS_NAMES,
)

# ── Model loading ─────────────────────────────────────────────────────

def load_router(device='cuda'):
    """Load frozen DINOv2 backbone + trained head from checkpoint."""
    backbone = timm.create_model(BACKBONE_NAME, pretrained=True,
                                  num_classes=0, img_size=DINOV2_IMG_SIZE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head = nn.Linear(DINOV2_EMBED_DIM, NUM_CLASSES)

    ckpt_path = ROOT / 'models' / 'router' / 'router_best.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'head_state_dict' in ckpt:
            head.load_state_dict(ckpt['head_state_dict'])
            print(f'Loaded router head from {ckpt_path}')
            print(f'  Best F1: {ckpt.get("best_f1", "?")}, Epoch: {ckpt.get("epoch", "?")}')
        elif 'model_state_dict' in ckpt:
            # Full model checkpoint — extract head weights
            sd = ckpt['model_state_dict']
            head_sd = {k.replace('head.', ''): v for k, v in sd.items() if k.startswith('head.')}
            if head_sd:
                head.load_state_dict(head_sd)
                print(f'Loaded router head from full checkpoint')
    else:
        print(f'WARNING: No checkpoint at {ckpt_path} — using random head!')

    backbone = backbone.to(device)
    head = head.to(device)
    return backbone, head


def get_transform():
    return A.Compose([
        A.Resize(DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── FastAPI app ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone, head = load_router(device)
    app.state.backbone = backbone
    app.state.head = head
    app.state.device = device
    app.state.transform = get_transform()
    # Load conformal thresholds
    thresh_path = ROOT / 'data' / 'specialist' / 'router' / 'conformal_thresholds.json'
    if thresh_path.exists():
        with open(thresh_path) as f:
            app.state.thresholds = json.load(f)
        print(f'Conformal thresholds loaded: {app.state.thresholds}')
    else:
        app.state.thresholds = {c: 0.6 for c in CLASS_NAMES}
    print(f'Router server ready on {device}')
    yield

app = FastAPI(title='Crop Router — Kerala Agriculture', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


CROP_INFO = {
    'okra': {'emoji': '🫛', 'name': 'Okra (Ladies Finger)', 'color': '#27ae60', 'specialist': 'Model 2'},
    'brassica': {'emoji': '🥦', 'name': 'Brassica (Broccoli/Cabbage)', 'color': '#2980b9', 'specialist': 'Model 2'},
    'tomato': {'emoji': '🍅', 'name': 'Tomato', 'color': '#e74c3c', 'specialist': 'Model 3'},
    'chilli': {'emoji': '🌶️', 'name': 'Chilli', 'color': '#d35400', 'specialist': 'Model 3'},
}


@app.get('/', response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crop Router — Kerala Agriculture</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0f1923; --surface: #1a2733; --surface2: #243447;
  --accent: #4ecdc4; --accent2: #45b7aa; --text: #e8edf2;
  --subtle: #8899aa; --border: #2d4052; --radius: 14px;
}
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg); color: var(--text); min-height: 100vh; }
header { background: linear-gradient(135deg, #1a2733 0%, #243447 100%);
  padding: 24px 20px; text-align: center; border-bottom: 1px solid var(--border); }
header h1 { font-size: 1.5rem; font-weight: 700; color: var(--accent);
  letter-spacing: -0.02em; }
header p { color: var(--subtle); font-size: 0.85rem; margin-top: 4px; }
main { max-width: 640px; margin: 28px auto; padding: 0 16px; }

#upload-area { border: 2px dashed var(--border); border-radius: var(--radius);
  background: var(--surface); padding: 48px 24px; text-align: center;
  cursor: pointer; transition: all 0.25s ease; }
#upload-area:hover, #upload-area.dragover { border-color: var(--accent);
  background: var(--surface2); }
#upload-area .icon { font-size: 3rem; margin-bottom: 12px; }
#upload-area .label { color: var(--accent); font-weight: 600; font-size: 1rem; }
#upload-area .hint { color: var(--subtle); font-size: 0.8rem; margin-top: 6px; }
input[type=file] { display: none; }

#preview { margin-top: 16px; text-align: center; display: none; }
#preview img { max-width: 100%; max-height: 280px; border-radius: var(--radius);
  border: 1px solid var(--border); }
#change-btn { margin-top: 10px; background: transparent; color: var(--accent);
  border: 1px solid var(--accent); border-radius: 8px; padding: 6px 16px;
  cursor: pointer; font-size: 0.85rem; }

#spinner { display: none; text-align: center; padding: 40px; }
.ring { width: 40px; height: 40px; border: 4px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
@keyframes spin { to { transform: rotate(360deg); } }

#result { display: none; margin-top: 20px; }
.result-card { background: var(--surface); border-radius: var(--radius);
  border: 1px solid var(--border); padding: 28px; }
.crop-header { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
.crop-emoji { font-size: 2.8rem; }
.crop-name { font-size: 1.3rem; font-weight: 700; }
.crop-specialist { font-size: 0.8rem; color: var(--subtle); margin-top: 2px; }

.confidence-section { margin-bottom: 18px; }
.conf-label { font-size: 0.85rem; color: var(--subtle); margin-bottom: 6px; }
.conf-bar-bg { height: 10px; background: var(--surface2); border-radius: 5px; overflow: hidden; }
.conf-bar { height: 100%; border-radius: 5px; transition: width 0.6s ease; }
.conf-value { font-size: 0.9rem; font-weight: 600; margin-top: 4px; }

.all-crops { margin-top: 18px; border-top: 1px solid var(--border); padding-top: 14px; }
.all-crops h3 { font-size: 0.85rem; color: var(--subtle); margin-bottom: 10px; }
.crop-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.crop-row .name { width: 90px; font-size: 0.85rem; }
.crop-row .bar-bg { flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; }
.crop-row .bar { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.crop-row .pct { width: 45px; text-align: right; font-size: 0.8rem; color: var(--subtle); }

.latency { text-align: center; color: var(--subtle); font-size: 0.78rem; margin-top: 14px; }
#error { display: none; background: #2d1b1b; border: 1px solid #5c3030; color: #e88;
  border-radius: var(--radius); padding: 14px; margin-top: 16px; font-size: 0.9rem; }
</style>
</head>
<body>
<header>
  <h1>Crop Router</h1>
  <p>DINOv2-Small+Registers — identifies crop type from leaf photo</p>
</header>
<main>
  <div id="upload-area" tabindex="0">
    <div class="icon">🌿</div>
    <div class="label">Drop a leaf image here or click to upload</div>
    <div class="hint">JPEG / PNG / WebP</div>
  </div>
  <input type="file" id="file-input" accept="image/*">

  <div id="preview"><img id="preview-img"><br><button id="change-btn">Change photo</button></div>
  <div id="spinner"><div class="ring"></div><p style="color:var(--subtle)">Classifying...</p></div>
  <div id="error"></div>

  <div id="result">
    <div class="result-card">
      <div class="crop-header">
        <span class="crop-emoji" id="r-emoji"></span>
        <div>
          <div class="crop-name" id="r-name"></div>
          <div class="crop-specialist" id="r-specialist"></div>
        </div>
      </div>
      <div class="confidence-section">
        <div class="conf-label">Confidence</div>
        <div class="conf-bar-bg"><div class="conf-bar" id="r-bar"></div></div>
        <div class="conf-value" id="r-conf"></div>
      </div>
      <div class="all-crops">
        <h3>All crop probabilities</h3>
        <div id="r-all-crops"></div>
      </div>
      <div id="r-routing" style="margin-top:14px;padding:12px;border-radius:10px;text-align:center;font-weight:600;font-size:0.9rem;"></div>
      <div class="latency" id="r-latency"></div>
    </div>
  </div>
</main>
<script>
const ua = document.getElementById('upload-area');
const fi = document.getElementById('file-input');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');

ua.addEventListener('click', () => fi.click());
ua.addEventListener('dragover', e => { e.preventDefault(); ua.classList.add('dragover'); });
ua.addEventListener('dragleave', () => ua.classList.remove('dragover'));
ua.addEventListener('drop', e => { e.preventDefault(); ua.classList.remove('dragover');
  if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]); });
fi.addEventListener('change', () => { if (fi.files[0]) processFile(fi.files[0]); });
document.getElementById('change-btn').addEventListener('click', () => {
  fi.value=''; preview.style.display='none'; ua.style.display='';
  document.getElementById('result').style.display='none';
  document.getElementById('error').style.display='none';
});

function processFile(file) {
  previewImg.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  ua.style.display = 'none';
  document.getElementById('result').style.display = 'none';
  document.getElementById('error').style.display = 'none';
  submitImage(file);
}

async function submitImage(file) {
  const spinner = document.getElementById('spinner');
  spinner.style.display = 'block';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Prediction failed');
    renderResult(data);
  } catch(e) {
    document.getElementById('error').textContent = e.message;
    document.getElementById('error').style.display = 'block';
  } finally { spinner.style.display = 'none'; }
}

const CROP_META = {
  okra:     { emoji:'🫛', name:'Okra (Ladies Finger)', color:'#27ae60', spec:'→ Model 2 Specialist' },
  brassica: { emoji:'🥦', name:'Brassica (Broccoli/Cabbage)', color:'#2980b9', spec:'→ Model 2 Specialist' },
  tomato:   { emoji:'🍅', name:'Tomato', color:'#e74c3c', spec:'→ Model 3 Specialist' },
  chilli:   { emoji:'🌶️', name:'Chilli', color:'#d35400', spec:'→ Model 3 Specialist' },
};

function renderResult(d) {
  const meta = CROP_META[d.predicted_crop] || CROP_META.okra;
  document.getElementById('r-emoji').textContent = meta.emoji;
  document.getElementById('r-name').textContent = meta.name;
  document.getElementById('r-specialist').textContent = meta.spec;
  const pct = Math.round(d.confidence * 100);
  document.getElementById('r-bar').style.width = pct + '%';
  document.getElementById('r-bar').style.background = meta.color;
  document.getElementById('r-conf').textContent = pct + '%';
  document.getElementById('r-conf').style.color = meta.color;

  const allDiv = document.getElementById('r-all-crops');
  allDiv.innerHTML = '';
  for (const [crop, prob] of Object.entries(d.all_probabilities)) {
    const cm = CROP_META[crop] || {color:'#888'};
    const p = Math.round(prob * 100);
    allDiv.innerHTML += '<div class="crop-row">' +
      '<span class="name">' + crop + '</span>' +
      '<div class="bar-bg"><div class="bar" style="width:'+p+'%;background:'+cm.color+'"></div></div>' +
      '<span class="pct">'+p+'%</span></div>';
  }
  // Routing decision
  const routeDiv = document.getElementById('r-routing');
  if (d.routed) {
    routeDiv.style.background = '#1a3a2a';
    routeDiv.style.border = '1px solid #2d6a4f';
    routeDiv.style.color = '#4ecdc4';
    routeDiv.innerHTML = '&#10003; ROUTED to ' + d.specialist +
      ' <span style="color:#8899aa;font-weight:400">(conf ' +
      Math.round(d.confidence*100) + '% &ge; threshold ' +
      Math.round(d.threshold*100) + '%)</span>';
  } else {
    routeDiv.style.background = '#3a2a1a';
    routeDiv.style.border = '1px solid #6a4f2d';
    routeDiv.style.color = '#e67e22';
    routeDiv.innerHTML = '&#9888; ABSTAINED — confidence too low ' +
      '<span style="color:#8899aa;font-weight:400">(conf ' +
      Math.round(d.confidence*100) + '% &lt; threshold ' +
      Math.round(d.threshold*100) + '%)</span>';
  }

  document.getElementById('r-latency').textContent = 'Inference: ' + d.latency_ms + ' ms';
  document.getElementById('result').style.display = 'block';
}
</script>
</body>
</html>"""


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, 'File too large (max 10 MB)')

    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
    except Exception:
        raise HTTPException(400, 'Could not open image')

    transform = app.state.transform
    backbone = app.state.backbone
    head = app.state.head
    device = app.state.device

    tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        features = backbone(tensor)
        logits = head(features)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
    latency = int((time.time() - t0) * 1000)

    top_idx = int(np.argmax(probs))
    predicted_crop = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])

    # Conformal routing decision
    threshold = app.state.thresholds.get(predicted_crop, 0.6)
    routed = confidence >= threshold
    specialist = CROP_INFO.get(predicted_crop, {}).get('specialist', '?')

    return JSONResponse({
        'predicted_crop': predicted_crop,
        'confidence': round(confidence, 4),
        'threshold': round(threshold, 4),
        'routed': routed,
        'specialist': specialist,
        'all_probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
        'latency_ms': latency,
    })


@app.get('/health')
async def health():
    return {'status': 'ok', 'model': 'Router DINOv2-Small+Registers',
            'classes': CLASS_NAMES, 'device': str(app.state.device)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("run_router_server:app", host="0.0.0.0", port=8003, reload=False)
