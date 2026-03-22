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
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/health')
async def health():
    return {'status': 'ok', 'device': str(DEVICE)}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    validation = validate_image(contents)
    if not validation['valid']:
        raise HTTPException(status_code=400, detail=validation['reason'])

    image_np = validation['image']
    model    = app.state.model
    loop     = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_inference(model, image_np)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference failed: {e}')

    # OOD returns 200 with ood_flagged=True — NOT 422
    return JSONResponse(content=result)


@app.post('/feedback')
async def feedback(request: Request):
    body      = await request.json()
    timestamp = datetime.utcnow().isoformat()
    db        = app.state.db
    db.execute(
        'INSERT INTO feedback (timestamp, crop, diseases, thumbs_up, correction) '
        'VALUES (?, ?, ?, ?, ?)',
        (timestamp,
         body.get('crop', ''),
         json.dumps(body.get('diseases', [])),
         1 if body.get('thumbs_up') else 0,
         body.get('correction', ''))
    )
    db.commit()
    return {'status': 'saved'}
