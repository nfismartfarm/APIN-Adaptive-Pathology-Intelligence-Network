"""
Model 2 Specialist Server -- Okra + Brassica disease classifier on localhost:8004
Upload a leaf image -> identifies disease with confidence.

Usage: python run_model2_server.py
Open: http://localhost:8004
"""

import os
import sys
import io
import json
import time
import base64

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))

from app.config_model2 import NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
from scripts.models import Model2ConvNeXt


def load_model(device='cuda'):
    model = Model2ConvNeXt(num_classes=NUM_CLASSES, pretrained=False)
    ckpt_path = ROOT / 'models' / 'model2_specialist' / 'model2_production.pt'
    if not ckpt_path.exists():
        ckpt_path = ROOT / 'models' / 'model2_specialist' / 'model2_best.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded model from {ckpt_path.name}, F1={ckpt.get("val_f1", ckpt.get("best_f1", "?"))}')
    else:
        print('WARNING: No checkpoint found!')
    model.to(device).eval()
    return model


def get_transform():
    return A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


DISEASE_INFO = {
    'okra_yvmv': {
        'emoji': '🟡', 'name': 'Yellow Vein Mosaic Virus',
        'severity': 'High', 'color': '#e74c3c',
        'desc': 'Viral disease spread by whitefly. Yellow vein network on leaves.',
    },
    'okra_powdery_mildew': {
        'emoji': '🤍', 'name': 'Powdery Mildew',
        'severity': 'Medium', 'color': '#e67e22',
        'desc': 'White powdery coating on leaf surface. Thrives in warm dry conditions.',
    },
    'okra_cercospora': {
        'emoji': '🟤', 'name': 'Cercospora Leaf Spot',
        'severity': 'Medium', 'color': '#e67e22',
        'desc': 'Brown spots with grey center and yellow halo.',
    },
    'okra_enation': {
        'emoji': '🔴', 'name': 'Enation Leaf Curl',
        'severity': 'High', 'color': '#e74c3c',
        'desc': 'Severe leaf curling with enations on vein undersides. No cure.',
    },
    'okra_healthy': {
        'emoji': '💚', 'name': 'Healthy Okra',
        'severity': 'None', 'color': '#27ae60',
        'desc': 'No disease detected. Leaf appears healthy.',
    },
    'brassica_black_rot': {
        'emoji': '⬛', 'name': 'Black Rot',
        'severity': 'High', 'color': '#e74c3c',
        'desc': 'V-shaped lesions from leaf margins. Bacterial, spreads via rain splash.',
    },
    'brassica_downy_mildew': {
        'emoji': '🔵', 'name': 'Downy Mildew',
        'severity': 'Medium', 'color': '#e67e22',
        'desc': 'Yellow patches above, white sporulation below. Thrives in cool humid conditions.',
    },
    'brassica_alternaria': {
        'emoji': '🎯', 'name': 'Alternaria Leaf Spot',
        'severity': 'Medium', 'color': '#e67e22',
        'desc': 'Dark concentric ring spots (target pattern). Spread by airborne spores.',
    },
    'brassica_healthy': {
        'emoji': '💚', 'name': 'Healthy Brassica',
        'severity': 'None', 'color': '#27ae60',
        'desc': 'No disease detected. Leaf appears healthy.',
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    app.state.model = load_model(device)
    app.state.device = device
    app.state.transform = get_transform()
    thresh_path = ROOT / 'data' / 'specialist' / 'model2' / 'conformal_thresholds.json'
    if thresh_path.exists():
        with open(thresh_path) as f:
            app.state.thresholds = json.load(f)
    else:
        app.state.thresholds = {c: 0.5 for c in CLASS_NAMES}
    print(f'Model 2 server ready on {device}')
    yield

app = FastAPI(title='Model 2 -- Okra & Brassica Disease Detector', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


@app.get('/', response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Disease Detector -- Okra & Brassica</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a1628;--surface:#111d33;--surface2:#1a2d4a;--accent:#f59e0b;
--accent2:#d97706;--text:#e8edf2;--subtle:#7a8ba0;--border:#1e3352;--radius:14px}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
header{background:linear-gradient(135deg,#111d33,#1a2d4a);padding:24px 20px;text-align:center;border-bottom:1px solid var(--border)}
header h1{font-size:1.5rem;font-weight:700;color:var(--accent)}
header p{color:var(--subtle);font-size:0.85rem;margin-top:4px}
main{max-width:680px;margin:28px auto;padding:0 16px}
#upload-area{border:2px dashed var(--border);border-radius:var(--radius);background:var(--surface);padding:48px 24px;text-align:center;cursor:pointer;transition:0.25s}
#upload-area:hover{border-color:var(--accent);background:var(--surface2)}
#upload-area .icon{font-size:3rem;margin-bottom:12px}
#upload-area .label{color:var(--accent);font-weight:600}
#upload-area .hint{color:var(--subtle);font-size:0.8rem;margin-top:6px}
input[type=file]{display:none}
#preview{margin-top:16px;text-align:center;display:none}
#preview img{max-width:100%;max-height:280px;border-radius:var(--radius);border:1px solid var(--border)}
#change-btn{margin-top:10px;background:transparent;color:var(--accent);border:1px solid var(--accent);border-radius:8px;padding:6px 16px;cursor:pointer;font-size:0.85rem}
#spinner{display:none;text-align:center;padding:40px}
.ring{width:40px;height:40px;border:4px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}
#result{display:none;margin-top:20px}
.result-card{background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);padding:28px}
.disease-header{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.disease-emoji{font-size:2.5rem}
.disease-name{font-size:1.3rem;font-weight:700}
.disease-desc{color:var(--subtle);font-size:0.85rem;margin-top:4px}
.severity-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;text-transform:uppercase;margin-top:8px}
.sev-high{background:#3a1a1a;color:#e74c3c;border:1px solid #5c2020}
.sev-medium{background:#3a2a1a;color:#e67e22;border:1px solid #5c3a20}
.sev-none{background:#1a3a2a;color:#27ae60;border:1px solid #2d6a4f}
.conf-section{margin:18px 0}
.conf-label{font-size:0.85rem;color:var(--subtle);margin-bottom:6px}
.conf-bar-bg{height:10px;background:var(--surface2);border-radius:5px;overflow:hidden}
.conf-bar{height:100%;border-radius:5px;transition:width 0.6s}
.conf-value{font-size:0.9rem;font-weight:600;margin-top:4px}
.all-probs{margin-top:18px;border-top:1px solid var(--border);padding-top:14px}
.all-probs h3{font-size:0.85rem;color:var(--subtle);margin-bottom:10px}
.prob-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.prob-row .name{width:160px;font-size:0.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.prob-row .bar-bg{flex:1;height:6px;background:var(--surface2);border-radius:3px}
.prob-row .bar{height:100%;border-radius:3px;transition:width 0.4s}
.prob-row .pct{width:40px;text-align:right;font-size:0.78rem;color:var(--subtle)}
.routing{margin-top:14px;padding:12px;border-radius:10px;text-align:center;font-weight:600;font-size:0.85rem}
.latency{text-align:center;color:var(--subtle);font-size:0.78rem;margin-top:14px}
#error{display:none;background:#2d1b1b;border:1px solid #5c3030;color:#e88;border-radius:var(--radius);padding:14px;margin-top:16px;font-size:0.9rem}
</style>
</head>
<body>
<header>
  <h1>Okra & Brassica Disease Detector</h1>
  <p>Model 2 Specialist -- DINOv3-ConvNeXt-Small -- 9 diseases -- F1=0.946</p>
</header>
<main>
  <div id="upload-area" tabindex="0">
    <div class="icon">🌿</div>
    <div class="label">Drop a leaf image here or click to upload</div>
    <div class="hint">JPEG / PNG / WebP -- Okra or Brassica leaves only</div>
  </div>
  <input type="file" id="file-input" accept="image/*">
  <div id="preview"><img id="preview-img"><br><button id="change-btn">Change photo</button></div>
  <div id="spinner"><div class="ring"></div><p style="color:var(--subtle)">Analyzing disease...</p></div>
  <div id="error"></div>
  <div id="result">
    <div class="result-card">
      <div class="disease-header">
        <span class="disease-emoji" id="r-emoji"></span>
        <div>
          <div class="disease-name" id="r-name"></div>
          <div class="disease-desc" id="r-desc"></div>
        </div>
      </div>
      <span class="severity-badge" id="r-severity"></span>
      <div id="heatmap-container" style="margin:16px 0;text-align:center;display:none">
        <p style="font-size:0.8rem;color:var(--subtle);margin-bottom:8px">GradCAM++ Disease Heatmap (red = high activation)</p>
        <img id="r-heatmap" style="max-width:100%;max-height:300px;border-radius:10px;border:1px solid var(--border)">
      </div>
      <div class="conf-section">
        <div class="conf-label">Confidence</div>
        <div class="conf-bar-bg"><div class="conf-bar" id="r-bar"></div></div>
        <div class="conf-value" id="r-conf"></div>
      </div>
      <div class="all-probs">
        <h3>All disease probabilities</h3>
        <div id="r-all-probs"></div>
      </div>
      <div class="routing" id="r-routing"></div>
      <div class="latency" id="r-latency"></div>
    </div>
  </div>
</main>
<script>
const DISEASE_META = {
  okra_yvmv:{emoji:'🟡',name:'Yellow Vein Mosaic Virus',color:'#e74c3c',sev:'High'},
  okra_powdery_mildew:{emoji:'🤍',name:'Powdery Mildew',color:'#e67e22',sev:'Medium'},
  okra_cercospora:{emoji:'🟤',name:'Cercospora Leaf Spot',color:'#e67e22',sev:'Medium'},
  okra_enation:{emoji:'🔴',name:'Enation Leaf Curl',color:'#e74c3c',sev:'High'},
  okra_healthy:{emoji:'💚',name:'Healthy Okra',color:'#27ae60',sev:'None'},
  brassica_black_rot:{emoji:'⬛',name:'Black Rot',color:'#e74c3c',sev:'High'},
  brassica_downy_mildew:{emoji:'🔵',name:'Downy Mildew',color:'#e67e22',sev:'Medium'},
  brassica_alternaria:{emoji:'🎯',name:'Alternaria Leaf Spot',color:'#e67e22',sev:'Medium'},
  brassica_healthy:{emoji:'💚',name:'Healthy Brassica',color:'#27ae60',sev:'None'},
};
const ua=document.getElementById('upload-area'),fi=document.getElementById('file-input');
const preview=document.getElementById('preview'),previewImg=document.getElementById('preview-img');
ua.addEventListener('click',()=>fi.click());
ua.addEventListener('dragover',e=>{e.preventDefault();ua.style.borderColor='var(--accent)'});
ua.addEventListener('dragleave',()=>{ua.style.borderColor=''});
ua.addEventListener('drop',e=>{e.preventDefault();ua.style.borderColor='';if(e.dataTransfer.files[0])processFile(e.dataTransfer.files[0])});
fi.addEventListener('change',()=>{if(fi.files[0])processFile(fi.files[0])});
document.getElementById('change-btn').addEventListener('click',()=>{
  fi.value='';preview.style.display='none';ua.style.display='';
  document.getElementById('result').style.display='none';document.getElementById('error').style.display='none'});
function processFile(f){previewImg.src=URL.createObjectURL(f);preview.style.display='block';ua.style.display='none';
  document.getElementById('result').style.display='none';document.getElementById('error').style.display='none';submitImage(f)}
async function submitImage(f){const sp=document.getElementById('spinner');sp.style.display='block';
  const fd=new FormData();fd.append('file',f);
  try{const r=await fetch('/predict',{method:'POST',body:fd});const d=await r.json();
    if(!r.ok)throw new Error(d.detail||'Failed');renderResult(d)}
  catch(e){document.getElementById('error').textContent=e.message;document.getElementById('error').style.display='block'}
  finally{sp.style.display='none'}}
function renderResult(d){
  const m=DISEASE_META[d.predicted_disease]||{emoji:'?',name:d.predicted_disease,color:'#888',sev:'?'};
  document.getElementById('r-emoji').textContent=m.emoji;
  document.getElementById('r-name').textContent=m.name;
  document.getElementById('r-desc').textContent=d.description||'';
  // Heatmap
  const hc=document.getElementById('heatmap-container');
  if(d.heatmap_b64){
    document.getElementById('r-heatmap').src='data:image/png;base64,'+d.heatmap_b64;
    hc.style.display='block'}else{hc.style.display='none'}
  const sb=document.getElementById('r-severity');
  sb.textContent=m.sev+' severity';
  sb.className='severity-badge sev-'+(m.sev==='High'?'high':m.sev==='Medium'?'medium':'none');
  const pct=Math.round(d.confidence*100);
  document.getElementById('r-bar').style.width=pct+'%';
  document.getElementById('r-bar').style.background=m.color;
  document.getElementById('r-conf').textContent=pct+'%';
  document.getElementById('r-conf').style.color=m.color;
  const ap=document.getElementById('r-all-probs');ap.innerHTML='';
  for(const[cls,prob] of Object.entries(d.all_probabilities)){
    const cm=DISEASE_META[cls]||{color:'#888'};const p=Math.round(prob*100);
    ap.innerHTML+='<div class="prob-row"><span class="name">'+cls.replace(/_/g,' ')+'</span>'+
      '<div class="bar-bg"><div class="bar" style="width:'+p+'%;background:'+cm.color+'"></div></div>'+
      '<span class="pct">'+p+'%</span></div>'}
  const rd=document.getElementById('r-routing');
  if(d.confident){rd.style.background='#1a3a2a';rd.style.border='1px solid #2d6a4f';rd.style.color='#4ecdc4';
    rd.innerHTML='&#10003; Confident diagnosis ('+pct+'% >= '+Math.round(d.threshold*100)+'% threshold)'}
  else{rd.style.background='#3a2a1a';rd.style.border='1px solid #6a4f2d';rd.style.color='#e67e22';
    rd.innerHTML='&#9888; Low confidence -- consider expert consultation'}
  document.getElementById('r-latency').textContent='Inference: '+d.latency_ms+' ms';
  document.getElementById('result').style.display='block'}
</script>
</body>
</html>"""


def generate_gradcam_heatmap(model, input_tensor, original_img_np, target_class=None):
    """
    Generate GradCAM++ heatmap overlay on the original image.
    Returns base64-encoded PNG string.
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layer = model.get_gradcam_target_layer()
    if target_layer is None:
        return ''

    try:
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Resize original image to match and create overlay
        orig_resized = cv2.resize(original_img_np, (384, 384))
        orig_float = orig_resized.astype(np.float32) / 255.0
        overlay = show_cam_on_image(orig_float, grayscale_cam, use_rgb=True)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f'GradCAM++ failed: {e}', flush=True)
        return ''


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
    model = app.state.model
    device = app.state.device

    tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

    t0 = time.time()
    # Need gradients for GradCAM++ -- don't use no_grad
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
        logits = model(tensor)
    probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    predicted = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])
    threshold = app.state.thresholds.get(predicted, 0.5)

    # Generate GradCAM++ heatmap
    heatmap_b64 = generate_gradcam_heatmap(model, tensor, img_np, target_class=top_idx)

    latency = int((time.time() - t0) * 1000)

    info = DISEASE_INFO.get(predicted, {})

    return JSONResponse({
        'predicted_disease': predicted,
        'confidence': round(confidence, 4),
        'threshold': round(threshold, 4),
        'confident': confidence >= threshold,
        'description': info.get('desc', ''),
        'severity': info.get('severity', 'Unknown'),
        'all_probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
        'heatmap_b64': heatmap_b64,
        'latency_ms': latency,
    })


@app.get('/health')
async def health():
    return {'status': 'ok', 'model': 'Model 2 Specialist (Okra+Brassica)',
            'classes': CLASS_NAMES, 'num_classes': NUM_CLASSES,
            'device': str(app.state.device), 'f1': 0.9464}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("run_model2_server:app", host="0.0.0.0", port=8004, reload=False)
