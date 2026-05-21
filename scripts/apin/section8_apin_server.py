"""Section 8 — APIN Server Integration.

FastAPI server exposing the APIN ensemble at /predict/apin while keeping
the legacy Model 2-only /predict endpoint for backward compatibility.

Endpoints:
  GET  /                — minimal HTML upload form
  GET  /health          — model load state, GPU VRAM, version
  POST /predict         — Model 2 only (legacy, fast)
  POST /predict/apin    — full APIN ensemble (slower, more accurate)
  POST /feedback        — store farmer correction with metadata
  GET  /feedback/stats  — counts of corrections received
  GET  /apin/info       — APIN config (gate weights, reliability matrix,
                          calibration timestamps)

Run:
  python scripts/apin/section8_apin_server.py --port 8005
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Eager FastAPI imports — pydantic v2 needs these resolved at module load
# time, not lazily inside make_app(). Without this, /predict and /predict/full
# raise PydanticUserError "UploadFile is not fully defined" on first call.
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
FEEDBACK_DIR = APIN_DIR / "feedback"
FEEDBACK_DIR.mkdir(exist_ok=True)
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_buffer.jsonl"
INFERENCE_LOG = APIN_DIR / "inference_log.jsonl"

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apin.server")

# Lazy-loaded singletons. Lock prevents concurrent first-callers from each
# constructing APINInference (which loads ~800MB of model weights into GPU
# memory). On the RTX 4060 8GB target, double-loading risks CUDA OOM.
_apin = None
_apin_lock = threading.Lock()


def get_apin():
    global _apin
    if _apin is None:
        with _apin_lock:
            # Double-checked locking: re-test under the lock, since another
            # thread may have completed construction while we waited.
            if _apin is None:
                from scripts.apin.inference import APINInference
                logger.info("Loading APIN inference (one-time)...")
                _apin = APINInference(verbose=False)
    return _apin


def make_app():
    app = FastAPI(title="Plant Disease APIN Server",
                   version="1.0",
                   description="APIN — Adaptive Pathological Intelligence Network")

    app.add_middleware(CORSMiddleware,
                        allow_origins=["*"], allow_credentials=True,
                        allow_methods=["*"], allow_headers=["*"])

    # HTML template is loaded from disk at server-construct time so
    # the UI can be iterated without touching server logic.
    _HTML_PATH = APIN_DIR / "ui_template.html"
    try:
        HTML = _HTML_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        HTML = f"<h1>ui_template.html missing</h1><p>Expected at {_HTML_PATH}</p>"


    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML

    @app.get("/warmup/status")
    async def warmup_status():
        """Per-model load state + cumulative stage log.
        Frontend polls this during cold-start to drive the growing-branch
        loader. Each stage reports whether its artifact is resident yet.
        """
        import torch
        entry = {
            "apin_constructed": _apin is not None,
            "router_loaded": False,
            "model2_loaded": False,
            "efficientnet_loaded": False,
            "psv_ready": False,
            "dinov2_loaded": False,
            "all_ready": False,
            "gpu_vram_gb": (
                round(torch.cuda.memory_allocated() / 1e9, 2)
                if torch.cuda.is_available() else 0.0
            ),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        # Router lives in make_app closure state
        entry["router_loaded"] = bool(_router_state.get("loaded") and
                                        _router_state.get("backbone") is not None)
        if _apin is not None:
            entry["model2_loaded"]       = (_apin._model2 is not None)
            entry["efficientnet_loaded"] = (_apin._efficientnet is not None)
            entry["dinov2_loaded"]       = (_apin._dinov2_backbone is not None)
            entry["psv_ready"]           = (_apin.psv_calibration is not None)
            # Round-1 audit fix K: expose OOD threshold so frontend doesn't
            # have to hardcode 34.53. Backend is the source of truth; if it
            # ever retrains, the UI chart auto-syncs.
            entry["ood_threshold"] = (
                float(_apin._ood_detector["threshold"])
                if _apin._ood_detector is not None else None
            )
        # Round-2 audit fix: separate `apin_ready` (the 4-signal ensemble +
        # PSV) from `all_ready` (which also requires the router). The router
        # is OPTIONAL — a deployment that doesn't ship `app/config_router.py`
        # or the trained head should still be usable on `/predict/apin` and
        # `/predict`. Without this split, the cold-start loader would poll
        # forever waiting for a router that's permanently unavailable.
        entry["apin_ready"] = (
            entry["model2_loaded"] and entry["efficientnet_loaded"]
            and entry["dinov2_loaded"] and entry["psv_ready"]
        )
        entry["all_ready"] = entry["apin_ready"] and entry["router_loaded"]
        return entry

    @app.get("/health")
    async def health():
        import torch
        cuda_avail = torch.cuda.is_available()
        vram = (torch.cuda.memory_allocated() / 1e9) if cuda_avail else 0
        return {"status": "ok",
                "device": "cuda" if cuda_avail else "cpu",
                "gpu_vram_gb": round(vram, 2),
                "version": "apin-1.0",
                "apin_loaded": _apin is not None,
                "timestamp": datetime.utcnow().isoformat()}

    @app.get("/apin/info")
    async def apin_info():
        a = get_apin()
        return {"n_signals": int(a.n_signals),
                "use_psv": bool(a.use_psv),
                "class_order": a.class_order,
                "stacking_mlp_gate_mean": a.stacking_mlp_gate_mean,
                "reliability_matrix": a.reliability_matrix.tolist(),
                "cold_start_active": bool(a.cold_start_active),
                "per_class_temperatures":
                    a.per_class_temps.tolist(),
                "conformal_thresholds":
                    a.conformal_thresholds.tolist()}

    def parse_image(file_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    # ── Router-to-APIN integration (Gap 7 audit fix) ─────────────────────
    # The crop router (run_router_server.py at port 8003) classifies into
    # okra / brassica / tomato / chilli. APIN handles okra + brassica only.
    # This /predict/full endpoint provides a single farmer-facing entry that:
    #   1. Routes the image to the crop classifier (in-process; no HTTP hop)
    #   2. If okra or brassica → run full APIN ensemble (this server)
    #   3. If tomato or chilli → return a "use a different specialist"
    #      payload with a router_only diagnosis so the farmer sees something
    #      while we wait for those specialists to be built
    #
    # Crop confidence threshold: if router conf < CROP_CONF_MIN, treat as
    # "uncertain crop" and run APIN anyway (the farmer's intent is dominant).
    CROP_CONF_MIN = 0.40
    APIN_HANDLED_CROPS = {"okra", "brassica"}
    _router_state = {"backbone": None, "head": None, "transform": None,
                       "device": None, "loaded": False}
    _router_lock = threading.Lock()

    def _ensure_router():
        if _router_state["loaded"]:
            return _router_state
        with _router_lock:
            if _router_state["loaded"]:
                return _router_state
            try:
                import torch
                from app.config_router import (
                    BACKBONE_NAME, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM,
                    NUM_CLASSES, CLASS_NAMES,
                )
                import timm
                import albumentations as A
                from albumentations.pytorch import ToTensorV2
                from torch import nn
                device = "cuda" if torch.cuda.is_available() else "cpu"
                backbone = timm.create_model(
                    BACKBONE_NAME, pretrained=True,
                    num_classes=0, img_size=DINOV2_IMG_SIZE,
                ).eval().to(device)
                for p in backbone.parameters(): p.requires_grad = False
                head = nn.Linear(DINOV2_EMBED_DIM, NUM_CLASSES).to(device)
                ckpt_path = PROJECT_ROOT / "models" / "router" / "router_best.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    if "head_state_dict" in ckpt:
                        head.load_state_dict(ckpt["head_state_dict"])
                    elif "model_state_dict" in ckpt:
                        sd = ckpt["model_state_dict"]
                        head_sd = {k.replace("head.", ""): v for k, v in sd.items()
                                     if k.startswith("head.")}
                        if head_sd:
                            head.load_state_dict(head_sd)
                    head.eval()
                else:
                    logger.warning(f"Router checkpoint missing: {ckpt_path}")
                transform = A.Compose([
                    A.Resize(DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
                _router_state.update(backbone=backbone, head=head,
                                       transform=transform, device=device,
                                       class_names=CLASS_NAMES, loaded=True)
                logger.info("Router loaded (in-process)")
            except Exception as e:
                logger.warning(f"Router load failed; /predict/full will skip routing: {e}")
                _router_state["loaded"] = True   # mark loaded-with-failure
        return _router_state

    def _classify_crop(img_np):
        """Returns (crop_name, conf) or (None, 0.0) if router unavailable."""
        st = _ensure_router()
        if st.get("backbone") is None:
            return None, 0.0
        import torch
        try:
            tens = st["transform"](image=img_np)["image"].unsqueeze(0).to(st["device"])
            with torch.no_grad():
                feat = st["backbone"](tens)
                logits = st["head"](feat)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            argmax = int(probs.argmax())
            return st["class_names"][argmax], float(probs[argmax])
        except Exception as e:
            logger.warning(f"Router inference failed: {e}")
            return None, 0.0

    @app.post("/predict/full")
    async def predict_full(file: UploadFile = File(...)):
        """Unified entry: router → APIN (okra/brassica) or specialist-not-available
        message (tomato/chilli). The farmer should hit THIS endpoint by default
        rather than calling /predict/apin directly."""
        try:
            data = await file.read()
            img = parse_image(data)
            crop, conf = _classify_crop(img)
            routing = {"router_crop": crop, "router_confidence": round(conf, 4),
                         "router_handled": crop in APIN_HANDLED_CROPS,
                         "low_router_confidence": (conf < CROP_CONF_MIN)}
            if crop is not None and crop not in APIN_HANDLED_CROPS and conf >= CROP_CONF_MIN:
                # Tomato or chilli — APIN doesn't handle these (Model 2 scope)
                return {
                    "routing": routing,
                    "diagnosis": None,
                    "message": (
                        f"Detected crop: {crop} (confidence {conf:.2f}). "
                        f"This is outside Model 2 APIN's scope (okra + brassica). "
                        f"A {crop} specialist is planned but not yet deployed. "
                        f"Please retake with an okra or brassica leaf, or "
                        f"contact your agricultural extension officer."
                    ),
                    "tier": "ROUTER_REJECTED",
                }
            # Okra or brassica (or low-confidence — APIN runs anyway)
            apin_engine = get_apin()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: apin_engine.predict(img))
            from dataclasses import asdict
            try:
                out = asdict(result)
            except Exception:
                out = {k: v for k, v in vars(result).items() if not k.startswith("_")}
            def _convert(o):
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (np.float32, np.float64)): return float(o)
                if isinstance(o, (np.int32, np.int64)): return int(o)
                if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
                if isinstance(o, list): return [_convert(v) for v in o]
                return o
            out = _convert(out)
            out["routing"] = routing
            return out
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("predict_full failed")
            raise HTTPException(500, str(e))

    @app.post("/predict")
    async def predict_legacy(file: UploadFile = File(...)):
        """Model 2 only (faster, legacy)."""
        try:
            data = await file.read()
            img = parse_image(data)
            a = get_apin()
            # Use just signal 1 (Model 2)
            t0 = time.time()
            probs = a._infer_model2(img)
            elapsed = (time.time() - t0) * 1000
            argmax = int(probs.argmax())
            return {"diagnosis": a.class_order[argmax],
                     "confidence": float(probs[argmax]),
                     "all_probabilities": {
                         a.class_order[i]: float(probs[i]) for i in range(9)
                     },
                     "processing_time_ms": round(elapsed, 1),
                     "mode": "model2_only_legacy"}
        except Exception as e:
            logger.exception("predict_legacy failed")
            raise HTTPException(500, str(e))

    @app.post("/predict/apin")
    async def predict_apin(file: UploadFile = File(...)):
        """Full APIN ensemble inference."""
        try:
            data = await file.read()
            img = parse_image(data)
            a = get_apin()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: a.predict(img))

            # Convert APINResult → JSON-safe dict
            from dataclasses import asdict
            try:
                out = asdict(result)
            except Exception:
                out = {k: v for k, v in vars(result).items()
                       if not k.startswith("_")}

            # Convert any np types
            def _convert(o):
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (np.float32, np.float64)): return float(o)
                if isinstance(o, (np.int32, np.int64)): return int(o)
                if isinstance(o, dict):
                    return {k: _convert(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_convert(v) for v in o]
                return o
            out = _convert(out)

            # Log to inference_log.jsonl
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "tier": out.get("tier"),
                "diagnosis": out.get("diagnosis"),
                "confidence": out.get("confidence"),
                "conflict_type": out.get("conflict_type"),
                "processing_time_ms": out.get("processing_time_ms"),
                "is_ood": out.get("is_ood"),
                "gate_weights": out.get("gate_weights"),
            }
            with open(INFERENCE_LOG, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            return out
        except Exception as e:
            logger.exception("predict_apin failed")
            raise HTTPException(500, str(e))

    # ── Feedback loop (Section 10 — Memory & Continuous Learning) ────────
    # Validates corrections, applies recency weighting, gates retraining
    # signals on minimum-sample requirements.

    # Architecture spec (Section 10):
    #   - 10+ new corrections before retrain consideration
    #   - 3+ independent corrections per class before that class influences retrain
    #   - Recency weights: 5x ≤7d, 3x 8-30d, 1.5x 31-90d, 1.0x older

    from scripts.apin.constants import MODEL2_CLASS_ORDER as _CLASSES
    _VALID_CLASSES = set(_CLASSES)
    FEEDBACK_RECENCY_BUCKETS = [
        (7, 5.0),     # ≤ 7 days: 5x
        (30, 3.0),    # 8-30 days: 3x
        (90, 1.5),    # 31-90 days: 1.5x
        (10**9, 1.0), # older: 1.0x
    ]

    def _validate_feedback(payload: dict) -> tuple:
        """Returns (is_valid, reason). Architecture: feedback for an
        already-correct case must be rejected so it cannot poison the buffer."""
        if not isinstance(payload, dict):
            return False, "payload must be a JSON object"
        if "is_correct" not in payload:
            return False, "missing required field: is_correct (bool)"
        if not isinstance(payload["is_correct"], bool):
            return False, "is_correct must be boolean"
        if not payload["is_correct"]:
            cc = payload.get("correct_class")
            if not cc:
                return False, "correct_class required when is_correct=false"
            if cc not in _VALID_CLASSES:
                return False, (
                    f"correct_class '{cc}' not in canonical class set "
                    f"(must be one of: {sorted(_VALID_CLASSES)})"
                )
            # Reject feedback for already-correct case (12.4.20)
            predicted = payload.get("predicted_class")
            if predicted == cc:
                return False, (
                    "predicted_class matches correct_class but is_correct=false "
                    "— ambiguous feedback, rejected"
                )
        return True, ""

    def _recency_weight(timestamp_iso: str) -> float:
        # Always operate in UTC-aware datetime space. Old code used
        # datetime.utcnow() (naive) when input was naive — subtracting an
        # aware-now from an aware-then works, but the else branch could
        # silently drift if a client submitted a non-UTC timestamp without
        # offset info. Force everything to UTC-aware.
        try:
            t = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            if t.tzinfo is None:
                # Treat naive timestamps as UTC (consistent with our writer)
                t = t.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days = max(0.0, (now - t).total_seconds() / 86400.0)
        except Exception:
            return 1.0
        for cutoff, w in FEEDBACK_RECENCY_BUCKETS:
            if days <= cutoff:
                return w
        return 1.0

    @app.post("/feedback")
    async def feedback(payload: dict = Body(...)):
        """Record farmer correction. Schema:
            prediction_id    (str, optional)  — id from prior /predict response
            is_correct       (bool, required)
            correct_class    (str, required if is_correct=False)
            predicted_class  (str, optional)  — for already-correct rejection
            geographic_hint  (str, optional)  — for clonal-amplification weighting
        """
        ok, reason = _validate_feedback(payload)
        if not ok:
            raise HTTPException(400, reason)
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **payload,
        }
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        return {"status": "recorded", "validated": True}

    # ── Retrain trigger (Gap 7 audit fix) ────────────────────────────────
    # The architecture spec requires that when feedback accumulates past
    # the safety thresholds (≥10 wrong + ≥3 classes with ≥3 corrections),
    # the stacking MLP gets retrained on the new evidence. The trigger is
    # gated by the same readiness function that /feedback/stats reports,
    # plus a minimum-interval rate-limit so a burst of corrections doesn't
    # spawn many retraining jobs back-to-back.
    #
    # Implementation: when readiness is True AND no retrain has run in the
    # last RETRAIN_MIN_INTERVAL_S seconds, dispatch Section 4 in a daemon
    # thread (so the request returns immediately). The retrain output is
    # logged and the next health-check exposes it.
    RETRAIN_MIN_INTERVAL_S = 6 * 3600   # 6 hours
    RETRAIN_LOG = APIN_DIR / "retrain_history.jsonl"
    _retrain_lock = threading.Lock()

    def _last_retrain_ts():
        if not RETRAIN_LOG.exists():
            return None
        try:
            with open(RETRAIN_LOG) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            if not lines:
                return None
            t = lines[-1].get("timestamp", "")
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            return None

    # F1 regression safety gate: if the retrained MLP's val_macro_f1 drops
    # below (old - this delta), reject the new checkpoint and roll back.
    # Architecture spec calls for 0.005 floor.
    RETRAIN_F1_DROP_REJECT = 0.005
    CKPT_PATH = (PROJECT_ROOT / "scripts" / "apin" / "caches" /
                  "apin_stacking_mlp.pt")

    def _spawn_retrain(reason: str):
        """Run Section 4 retrain in a background daemon thread.
        Backs up the existing checkpoint, retrains, recalibrates, then
        compares val_macro_f1: if the new MLP's F1 is worse by more than
        RETRAIN_F1_DROP_REJECT, restores the backup and logs a rejection.
        Otherwise resets the singleton so the next /predict/apin call
        picks up the new checkpoint."""
        import shutil
        import subprocess
        import torch

        def _read_val_f1(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu",
                                   weights_only=False)
                return float(ckpt.get("val_macro_f1", 0.0))
            except Exception:
                return None

        def _runner():
            # Round-3 audit fix (CRITICAL): acquire `_retrain_lock` HERE for
            # the full duration of the retrain (subprocess + F1 gate +
            # rollback). The previous design released the lock in the
            # endpoint handler immediately after dispatching the thread,
            # which meant a second concurrent /feedback/retrain request
            # would see a free lock and start a second retrain — two
            # subprocesses racing to write the same checkpoint.
            #
            # We try blocking=False so a stale dispatch path doesn't deadlock.
            # If we can't acquire, the endpoint must have already screened
            # this; bail with a logged warning.
            if not _retrain_lock.acquire(blocking=False):
                logger.warning("  Retrain runner could not acquire lock; "
                                "another retrain in progress. Aborting.")
                # Round-4 audit fix: write a skipped-concurrent record to
                # RETRAIN_LOG so it appears in /feedback/retrain/history.
                # Without this the operator only sees the dispatch 200 OK
                # and a Python warning that's invisible to API clients.
                try:
                    skip_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                        "reason": reason,
                        "status": "skipped_concurrent",
                    }
                    with open(RETRAIN_LOG, "a") as f:
                        f.write(json.dumps(skip_entry) + "\n")
                except Exception:
                    pass
                return
            try:
                _runner_body()
            finally:
                _retrain_lock.release()

        def _runner_body():
            ts = datetime.now(timezone.utc).isoformat() + "Z"
            entry = {"timestamp": ts, "reason": reason, "status": "started"}
            with open(RETRAIN_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Pre-flight: capture old F1 + back up current checkpoint
            old_f1 = _read_val_f1(CKPT_PATH) if CKPT_PATH.exists() else None
            backup_path = None
            if CKPT_PATH.exists():
                backup_path = CKPT_PATH.with_suffix(
                    f".backup_{ts.replace(':','').replace('-','')}.pt"
                )
                try:
                    shutil.copy2(CKPT_PATH, backup_path)
                except Exception as e:
                    logger.warning(f"  Retrain backup failed: {e}")
                    backup_path = None

            python_exe = sys.executable
            project_root = str(PROJECT_ROOT)
            try:
                for script in ("scripts/apin/section4_stacking_mlp.py",
                                 "scripts/apin/section5_calibration.py"):
                    r = subprocess.run(
                        [python_exe, script],
                        cwd=project_root,
                        capture_output=True, text=True, timeout=3600,
                    )
                    if r.returncode != 0:
                        raise RuntimeError(
                            f"{script} failed: {r.stderr[-500:]}"
                        )

                # F1 safety gate: compare new vs old
                new_f1 = _read_val_f1(CKPT_PATH)
                if (old_f1 is not None and new_f1 is not None
                        and new_f1 < (old_f1 - RETRAIN_F1_DROP_REJECT)):
                    # Rollback: restore the backup
                    if backup_path is not None and backup_path.exists():
                        shutil.copy2(backup_path, CKPT_PATH)
                    raise RuntimeError(
                        f"F1 regression rejected: new={new_f1:.4f} < "
                        f"old={old_f1:.4f} - {RETRAIN_F1_DROP_REJECT}. "
                        f"Rolled back to backup."
                    )

                # Accepted: force singleton to reload the new checkpoint
                global _apin
                with _apin_lock:
                    _apin = None
                done_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "started": ts,
                    "reason": reason,
                    "status": "complete",
                    "old_val_macro_f1": old_f1,
                    "new_val_macro_f1": new_f1,
                    "f1_delta": (None if (old_f1 is None or new_f1 is None)
                                  else round(new_f1 - old_f1, 4)),
                    "backup_path": (str(backup_path.relative_to(PROJECT_ROOT))
                                     if backup_path else None),
                }
            except Exception as e:
                done_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "started": ts,
                    "reason": reason,
                    "status": "failed",
                    "error": str(e)[:500],
                    "old_val_macro_f1": old_f1,
                    "backup_path": (str(backup_path.relative_to(PROJECT_ROOT))
                                     if backup_path else None),
                }
            with open(RETRAIN_LOG, "a") as f:
                f.write(json.dumps(done_entry) + "\n")

        t = threading.Thread(target=_runner, daemon=True, name="apin-retrain")
        t.start()
        return t

    @app.post("/feedback/retrain")
    async def feedback_retrain():
        """Manually trigger a retrain if readiness is met and rate-limit allows.
        Returns 202 if dispatched, 400 if not ready, 429 if rate-limited."""
        # Compute readiness inline (mirrors /feedback/stats logic)
        if not FEEDBACK_LOG.exists():
            raise HTTPException(400, "No feedback log present")
        wrong = 0
        by_class: dict = {}
        with open(FEEDBACK_LOG) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("is_correct"):
                        wrong += 1
                        cc = r.get("correct_class", "unknown")
                        by_class[cc] = by_class.get(cc, 0) + 1
                except Exception:
                    continue
        classes_with_min = sum(1 for v in by_class.values() if v >= 3)
        if not (wrong >= 10 and classes_with_min >= 3):
            raise HTTPException(
                400,
                f"Not ready: have {wrong} wrong corrections across "
                f"{classes_with_min} classes with ≥3 each "
                f"(need ≥10 and ≥3 respectively)."
            )
        # Rate limit
        last = _last_retrain_ts()
        if last is not None:
            now = datetime.now(timezone.utc)
            elapsed = (now - last).total_seconds()
            if elapsed < RETRAIN_MIN_INTERVAL_S:
                wait = RETRAIN_MIN_INTERVAL_S - elapsed
                raise HTTPException(
                    429,
                    f"Rate-limited: last retrain was {int(elapsed)}s ago; "
                    f"wait {int(wait)}s more (minimum interval "
                    f"{RETRAIN_MIN_INTERVAL_S}s)."
                )
        # Dispatch (lock ensures only one retrain in flight at a time).
        # Round-3 audit fix: the lock is acquired and held by `_runner`
        # itself for the FULL duration of the retrain (not just the
        # microsecond of dispatch). Here we just check availability —
        # if locked, another retrain is already in progress.
        if _retrain_lock.locked():
            raise HTTPException(429, "Retrain already in progress")
        _spawn_retrain(reason=f"manual /feedback/retrain (wrong={wrong}, "
                                f"classes_with_min={classes_with_min})")
        return {"status": "dispatched", "wrong_corrections": wrong,
                "classes_with_3_or_more": classes_with_min}

    @app.get("/feedback/retrain/history")
    async def feedback_retrain_history():
        if not RETRAIN_LOG.exists():
            return {"events": [], "last_complete": None}
        events = []
        last_complete = None
        with open(RETRAIN_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    events.append(e)
                    if e.get("status") == "complete":
                        last_complete = e.get("timestamp")
                except Exception:
                    continue
        return {"events": events[-20:], "last_complete": last_complete,
                "rate_limit_seconds": RETRAIN_MIN_INTERVAL_S}

    @app.get("/feedback/stats")
    async def feedback_stats():
        """Aggregate stats with recency-weighted counts per class.

        ready_for_retrain: True when total wrong corrections ≥ 10 AND at
        least 3 independent classes have ≥ 3 wrong corrections each. This
        is the trigger architecture Section 10.2/10.3 specifies before
        retraining the stacking MLP on new data.
        """
        if not FEEDBACK_LOG.exists():
            return {"total": 0, "correct": 0, "wrong": 0, "by_class": {},
                    "weighted_by_class": {}, "ready_for_retrain": False}
        total = correct = wrong = 0
        by_class: dict = {}
        weighted_by_class: dict = {}
        with open(FEEDBACK_LOG) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                total += 1
                if r.get("is_correct"):
                    correct += 1
                    continue
                wrong += 1
                cc = r.get("correct_class", "unknown")
                by_class[cc] = by_class.get(cc, 0) + 1
                w = _recency_weight(r.get("timestamp", ""))
                weighted_by_class[cc] = round(
                    weighted_by_class.get(cc, 0.0) + w, 4
                )
        # Retrain readiness gate
        classes_with_min = sum(1 for v in by_class.values() if v >= 3)
        ready = (wrong >= 10) and (classes_with_min >= 3)
        return {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "by_class": by_class,
            "weighted_by_class": weighted_by_class,
            "ready_for_retrain": ready,
            "retrain_gate_reason": (
                "ready" if ready
                else f"need ≥10 wrong (have {wrong}) and ≥3 classes with ≥3 "
                     f"corrections (have {classes_with_min})"
            ),
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preload", action="store_true",
                         help="Load APIN at startup (vs lazy on first request)")
    args = parser.parse_args()

    import uvicorn
    app = make_app()
    if args.preload:
        # Full warmup: construct APINInference AND eagerly load all lazy
        # backbones (Model 2 ~800MB, EfficientNet ~80MB, DINOv2 ~90MB). This
        # front-loads the ~3-minute first-call cost to server startup so
        # the first real user upload runs in ~600ms-1.5s (warm), not 3 min.
        logger.info("Warming APIN (Model 2 + EfficientNet + DINOv2 + PSV)...")
        a = get_apin()
        import time
        t0 = time.time()
        try: a._lazy_load_model2();          logger.info(f"  Model 2 warm ({time.time()-t0:.1f}s)")
        except Exception as e: logger.warning(f"  Model 2 warmup failed: {e}")
        t0 = time.time()
        try: a._lazy_load_efficientnet();    logger.info(f"  EfficientNet warm ({time.time()-t0:.1f}s)")
        except Exception as e: logger.warning(f"  EfficientNet warmup failed: {e}")
        t0 = time.time()
        try: a._lazy_load_dinov2();          logger.info(f"  DINOv2 warm ({time.time()-t0:.1f}s)")
        except Exception as e: logger.warning(f"  DINOv2 warmup failed: {e}")
        logger.info("APIN fully warm — first user request will be fast")
    logger.info(f"Starting APIN server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
