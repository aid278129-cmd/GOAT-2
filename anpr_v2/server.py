"""
anpr_v2/server.py
FastAPI Inference Microservice for ANPR V2 (Port 5001).

Exact drop-in replacement for scripts/anpr_server.py:
- Exposes /detect, /detect-file, /health, /config, and /debug endpoints.
- Forwards confirmed events to Node.js backend at https://127.0.0.1:3000/api/detections.
- Fully compatible with MultiCameraQueueManager in server.js.
"""

import os
import sys
import io
import time
import base64
import requests
import urllib3
import cv2
import numpy as np

# Ensure parent directory is in sys.path
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from anpr_v2.config import CONFIG, BASE_DIR, DEBUG_DIR, DEBUG_CROPS_DIR, DEBUG_FRAMES_DIR
from anpr_v2.pipeline import get_pipeline, ANPRPipelineV2

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="City-Wide Indian ANPR V2 Intelligence Server (BEL SIH)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount debug static directory so dashboard can inspect crops and frames
app.mount("/debug_output", StaticFiles(directory=DEBUG_DIR), name="debug_output")

# Telemetry History Cache
RECENT_DEBUG_HISTORY: List[Dict[str, Any]] = []
LATEST_DEBUG_PER_CAM: Dict[int, Dict[str, Any]] = {}
SERVER_STATS = {
    "total_frames_processed": 0,
    "plates_detected_count": 0,
    "confirmed_count": 0,
    "rejection_reasons": {}
}

class Base64DetectRequest(BaseModel):
    image: str
    cameraId: int = 1
    cameraName: str = "Camera 1"
    forwardToDashboard: bool = True
    manualScan: bool = False
    developerMode: bool = False
    keyframeSharpness: Optional[float] = None
    keyframeWindowMs: Optional[int] = None
    keyframeCandidates: Optional[int] = None
    frameCapturedAt: Optional[float] = None
    keyframeSelectedAt: Optional[float] = None
    requestQueuedAt: Optional[float] = None
    workerDispatchedAt: Optional[float] = None
    queueWaitMs: Optional[float] = None

def decode_base64_image(b64_str: str) -> Optional[np.ndarray]:
    """Decodes data:image/...;base64,... string to OpenCV BGR numpy array."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ANPR V2] Base64 decode error: {e}")
        return None

@app.get("/health")
def health():
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "version": "v2",
        "detector_backend": CONFIG.detector_backend,
        "ocr_backend": CONFIG.ocr_backend,
        "detector_ready": pipeline.detector.is_ready,
        "recognizer_ready": pipeline.recognizer.is_ready,
        "debug_output_ready": os.path.exists(DEBUG_CROPS_DIR) and os.path.exists(DEBUG_FRAMES_DIR),
        "config": {
            "plate_conf_threshold": CONFIG.plate_conf_threshold,
            "live_plate_conf_threshold": CONFIG.live_plate_conf_threshold,
            "temporal_consensus": CONFIG.temporal_consensus_enabled,
            "save_debug_crops": CONFIG.save_debug_crops
        }
    }

@app.get("/config")
def get_config():
    return {
        "anpr_version": CONFIG.anpr_version,
        "detector_backend": CONFIG.detector_backend,
        "ocr_backend": CONFIG.ocr_backend,
        "plate_conf_threshold": CONFIG.plate_conf_threshold,
        "live_plate_conf_threshold": CONFIG.live_plate_conf_threshold,
        "min_ocr_plate_conf": CONFIG.min_ocr_plate_conf,
        "temporal_consensus_enabled": CONFIG.temporal_consensus_enabled,
        "temporal_window_sec": CONFIG.temporal_window_sec,
        "save_debug_crops": CONFIG.save_debug_crops
    }

@app.post("/config")
def update_config(updates: Dict[str, Any]):
    for k, v in updates.items():
        if hasattr(CONFIG, k):
            setattr(CONFIG, k, v)
    return {"success": True, "updated_config": get_config()}

@app.get("/debug/last")
def debug_last(cameraId: Optional[int] = Query(None)):
    if cameraId and cameraId in LATEST_DEBUG_PER_CAM:
        return LATEST_DEBUG_PER_CAM[cameraId]
    return RECENT_DEBUG_HISTORY[-1] if RECENT_DEBUG_HISTORY else {}

@app.get("/debug/history")
def debug_history(limit: int = Query(20, ge=1, le=100)):
    return RECENT_DEBUG_HISTORY[-limit:]

@app.get("/debug/stats")
def debug_stats():
    return SERVER_STATS

@app.post("/detect")
def detect_plate(req: Base64DetectRequest):
    t_req_start = time.perf_counter()
    SERVER_STATS["total_frames_processed"] += 1

    img_bgr = decode_base64_image(req.image)
    if img_bgr is None:
        return {"success": False, "error": "Invalid base64 image data"}

    pipeline = get_pipeline()
    extra_telemetry = {
        "frameCapturedAt": req.frameCapturedAt,
        "keyframeSelectedAt": req.keyframeSelectedAt,
        "requestQueuedAt": req.requestQueuedAt,
        "workerDispatchedAt": req.workerDispatchedAt,
        "queueWaitMs": req.queueWaitMs
    }

    result = pipeline.process_frame(
        frame_bgr=img_bgr,
        camera_id=req.cameraId,
        camera_name=req.cameraName,
        manual_scan=req.manualScan,
        extra_telemetry=extra_telemetry
    )

    if result.get("detected"):
        SERVER_STATS["plates_detected_count"] += 1
        SERVER_STATS["confirmed_count"] += 1

    # Record telemetry cache
    dbg = result.get("debug", {})
    LATEST_DEBUG_PER_CAM[req.cameraId] = dbg
    RECENT_DEBUG_HISTORY.append(dbg)
    if len(RECENT_DEBUG_HISTORY) > 100:
        RECENT_DEBUG_HISTORY.pop(0)

    # Forward confirmed event to Node.js backend if requested
    confirmed_det = result.get("detection")
    if req.forwardToDashboard and confirmed_det and confirmed_det.get("plate"):
        try:
            requests.post(
                CONFIG.node_server_url,
                json={
                    "plate": confirmed_det["plate"],
                    "cameraId": req.cameraId,
                    "confidence": confirmed_det["confidence"],
                    "vehicleType": "car",
                    "manualScan": bool(req.manualScan)
                },
                timeout=1.5,
                verify=False
            )
        except Exception as e:
            print(f"[ANPR V2] Dashboard forward warning: {e}")

    return result

@app.post("/detect-file")
async def detect_plate_file(
    file: UploadFile = File(...),
    cameraId: int = Form(1),
    cameraName: str = Form("Manual Scan"),
    forwardToDashboard: bool = Form(True)
):
    try:
        contents = await file.read()
        b64 = base64.b64encode(contents).decode("utf-8")
        req = Base64DetectRequest(
            image=f"data:image/jpeg;base64,{b64}",
            cameraId=cameraId,
            cameraName=cameraName,
            forwardToDashboard=forwardToDashboard,
            manualScan=True
        )
        return detect_plate(req)
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_server(port: int = 5001):
    print(f"Starting ANPR V2 Inference Microservice on http://127.0.0.1:{port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    run_server()
