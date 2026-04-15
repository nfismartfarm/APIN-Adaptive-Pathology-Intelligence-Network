# app/main.py

import os
import sys
import json
import sqlite3
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io

from app.config import DEVICE, BEST_MODEL, ROOT
from app.model import load_model_for_inference
from app.validator import validate_image
from app.inference import run_inference


DB_PATH = os.path.join(ROOT, 'feedback.db')


def init_db():
    # [FIX GAP inline] check_same_thread=False required for multi-threaded FastAPI
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT,
            crop       TEXT,
            diseases   TEXT,
            thumbs_up  INTEGER,
            correction TEXT
        )
    ''')
    conn.commit()
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_for_inference(BEST_MODEL, DEVICE)
    app.state.db    = init_db()
    yield
    app.state.db.close()


app = FastAPI(title='Plant Disease Detection — Kerala', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

templates = Jinja2Templates(directory=os.path.join(ROOT, 'templates'))
app.mount('/static', StaticFiles(directory=os.path.join(ROOT, 'static')), name='static')


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')


@app.get('/health')
async def health():
    from app.inference import get_sam_mask_generator
    sam_available = get_sam_mask_generator() is not None
    return {
        'status': 'ok',
        'device': str(DEVICE),
        'sam_available': sam_available,
        'model': 'Swin-Tiny + FPN + AttPool + CLN + MoE (23 classes, 4 crops)',
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...), force_predict: bool = False):
    contents = await file.read()

    validation = validate_image(contents)
    if not validation['valid']:
        raise HTTPException(status_code=400, detail=validation['reason'])

    image_np = validation['image']
    model    = app.state.model
    loop     = asyncio.get_running_loop()  # Issue 14: deprecated get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_inference(model, image_np, force_predict=force_predict)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference failed: {e}')

    # Store image bytes temporarily for feedback saving
    app.state.last_upload = contents

    # OOD returns 200 with ood_flagged=True — NOT 422
    return JSONResponse(content=result)


@app.post('/feedback')
async def feedback(request: Request):
    import re
    body      = await request.json()
    timestamp = datetime.utcnow().isoformat()
    # Sanitize correction input
    correction = body.get('correction', '')
    correction = re.sub(r'<[^>]+>', '', str(correction)).strip()[:200]
    thumbs_up = 1 if body.get('thumbs_up') else 0
    db        = app.state.db
    db.execute(
        'INSERT INTO feedback (timestamp, crop, diseases, thumbs_up, correction) '
        'VALUES (?, ?, ?, ?, ?)',
        (timestamp,
         body.get('crop', ''),
         json.dumps(body.get('diseases', [])),
         thumbs_up,
         correction)
    )
    db.commit()

    # Save image when user gives feedback (Item 25: feedback pipeline)
    last_upload = getattr(app.state, 'last_upload', None)
    if last_upload and correction:
        try:
            feedback_dir = os.path.join(ROOT, 'data', 'feedback', 'corrected', correction)
            os.makedirs(feedback_dir, exist_ok=True)
            fname = f"fb_{timestamp.replace(':', '-').replace('.', '-')}.jpg"
            with open(os.path.join(feedback_dir, fname), 'wb') as f:
                f.write(last_upload)
        except Exception:
            pass  # feedback image saving is best-effort
    elif last_upload and thumbs_up:
        try:
            crop = body.get('crop', 'unknown')
            diseases = body.get('diseases', [])
            label = diseases[0] if diseases else crop
            feedback_dir = os.path.join(ROOT, 'data', 'feedback', 'confirmed', label)
            os.makedirs(feedback_dir, exist_ok=True)
            fname = f"fb_{timestamp.replace(':', '-').replace('.', '-')}.jpg"
            with open(os.path.join(feedback_dir, fname), 'wb') as f:
                f.write(last_upload)
        except Exception:
            pass

    return {'status': 'saved'}


@app.get('/api/classes')
async def get_classes():
    """Returns all 23 class names for the correction dropdown."""
    from app.config import CLASS_NAMES
    return {'classes': CLASS_NAMES}


# ── Phase 5: Gallery endpoint ────────────────────────────────────────────

import pandas as pd

GALLERY_MANIFEST = None
GALLERY_MANIFEST_PATH = os.path.join(ROOT, 'static', 'gallery', 'gallery_manifest.csv')

@app.on_event('startup')
async def load_gallery():
    global GALLERY_MANIFEST
    if os.path.exists(GALLERY_MANIFEST_PATH):
        GALLERY_MANIFEST = pd.read_csv(GALLERY_MANIFEST_PATH)
        print(f'Gallery loaded: {len(GALLERY_MANIFEST)} reference images')

@app.get('/gallery/{class_name}')
async def get_gallery(class_name: str):
    """Returns up to 3 reference image paths for the given disease class."""
    from app.config import CLASS_NAMES
    if GALLERY_MANIFEST is None:
        return JSONResponse({'images': [], 'error': 'Gallery not available'})
    if class_name not in CLASS_NAMES:
        return JSONResponse({'images': [], 'error': f'Unknown class: {class_name}'})
    class_images = GALLERY_MANIFEST[GALLERY_MANIFEST['class_name'] == class_name]
    images = [{'url': f'/static/{row["image_file"]}', 'confidence': float(row['confidence'])}
              for _, row in class_images.iterrows()]
    return JSONResponse({
        'class_name': class_name, 'images': images, 'count': len(images),
        'attribution': 'PlantVillage Dataset (Hughes et al., 2016), CC BY 4.0',
    })


# ── Phase 5: Quality check endpoint ──────────────────────────────────────

@app.post('/quality-check')
async def quality_check_endpoint(file: UploadFile = File(...)):
    """Pre-submission image quality check."""
    from app.quality_check import check_image_quality
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_np = np.array(img)
    quality = check_image_quality(img_np)
    return JSONResponse({
        'overall_score': quality['overall_score'],
        'passed': quality['passed'],
        'feedback': quality['feedback'],
        'issues': quality['issues'],
        'dimensions': quality['dimensions'],
    })


@app.get('/api/feedback/stats')
async def feedback_stats():
    """Returns feedback statistics for monitoring."""
    db = app.state.db
    cursor = db.execute('SELECT COUNT(*) FROM feedback')
    total = cursor.fetchone()[0]
    cursor = db.execute('SELECT COUNT(*) FROM feedback WHERE thumbs_up = 1')
    thumbs_up = cursor.fetchone()[0]
    cursor = db.execute('SELECT COUNT(*) FROM feedback WHERE thumbs_up = 0')
    thumbs_down = cursor.fetchone()[0]
    cursor = db.execute(
        'SELECT correction, COUNT(*) as cnt FROM feedback '
        'WHERE correction != "" GROUP BY correction ORDER BY cnt DESC LIMIT 10'
    )
    top_corrections = [{'class': row[0], 'count': row[1]} for row in cursor.fetchall()]
    return {
        'total': total,
        'thumbs_up': thumbs_up,
        'thumbs_down': thumbs_down,
        'top_corrections': top_corrections,
    }
