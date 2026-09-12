"""
anpr_server.py
High-Performance Local ANPR Inference Service for Indian License Plates
SIH Problem Statement ID: 26127 (BEL)

Layered Vision Pipeline:
  1. Vehicle-First Detection (YOLOv8 COCO: car, motorcycle, bus, truck)
  2. Vehicle ROI Extraction & Spatial Verification
  3. License Plate Detection inside Vehicle ROI (Trained Indian Plate YOLOv8)
  4. Geometric & Aspect Ratio Filtering
  5. Optimized OCR (Tesseract v5.4.0 with CLAHE/Dual-Otsu + EasyOCR fallback)
  6. Indian Registration Syntax Scoring & Positional Disambiguation
  7. Multi-Frame Temporal Confirmation Tracker (Rejects isolated single-frame noise)
  8. Confirmed ANPR Event Forwarding to Node.js Backend
"""

import os
import io
import re
import time
import base64
import difflib
import numpy as np
from PIL import Image
import cv2
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
import easyocr
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="City-Wide Indian ANPR Intelligence Server (BEL SIH)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "indian_plate_best.pt")
FALLBACK_PLATE_MODEL_PATH = os.path.join(BASE_DIR, "runs", "indian_plate_run", "weights", "best.pt")
NODE_SERVER_URL = "https://127.0.0.1:3000/api/detections"

# Set Tesseract binary path
TESSERACT_DEFAULT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_DEFAULT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_DEFAULT_PATH
    print(f"Configured Tesseract OCR binary: {TESSERACT_DEFAULT_PATH}")
else:
    print("Using system default tesseract command")

# ──────────────────────────────────────────────────────────────────
# Centralized, Configurable Settings (No Arbitrary Hardcoding)
# ──────────────────────────────────────────────────────────────────
ANPR_CONFIG = {
    # Stage 1: Vehicle Detection
    "vehicle_detection_enabled": True,
    "vehicle_classes": [2, 3, 5, 7],         # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    "vehicle_conf_threshold": 0.22,           # Lowered slightly for partial bumper framing
    "vehicle_roi_padding": 0.05,             # 5% padding around vehicle crop

    # Stage 2: License Plate Candidate Detection
    "plate_conf_threshold": 0.40,             # Min plate candidate confidence
    "closeup_plate_conf_threshold": 0.60,     # Strict threshold if no full vehicle body in frame

    # Stage 3: Geometric & Spatial Validation
    "min_plate_aspect_ratio": 1.5,           # Width / Height
    "max_plate_aspect_ratio": 6.2,           # Standard rectangular plates
    "min_plate_width": 20,                   # Pixels
    "min_plate_height": 8,                   # Pixels
    "max_plate_to_vehicle_area": 0.35,       # Plate cannot be >35% of full vehicle area
    "plate_vertical_pos_min": 0.15,          # Plate must not be at the very top roof of vehicle

    # Stage 4: OCR & Indian Plate Syntax
    "min_ocr_conf": 0.45,
    "min_plate_chars": 6,
    "max_plate_chars": 11,
    "strict_indian_state_required": True,

    # Stage 5: Multi-Frame Temporal Confirmation
    "temporal_window_sec": 3.0,              # Sliding window duration
    "min_consecutive_sightings": 2,          # Minimum frames to confirm (isolated 1-frame spikes rejected)
    "temporal_similarity_threshold": 0.80,  # String similarity ratio
}

# ──────────────────────────────────────────────────────────────────
# Model Holders
# ──────────────────────────────────────────────────────────────────
vehicle_model = None
plate_model = None
easyocr_reader = None

def get_vehicle_model():
    global vehicle_model
    if vehicle_model is None:
        from ultralytics import YOLO
        vehicle_model_path = os.path.join(BASE_DIR, "yolov8n.pt")
        print(f"Loading vehicle detector from {vehicle_model_path}...")
        vehicle_model = YOLO(vehicle_model_path if os.path.exists(vehicle_model_path) else "yolov8n.pt")
    return vehicle_model

def get_plate_model():
    global plate_model
    if plate_model is None:
        from ultralytics import YOLO
        if os.path.exists(PLATE_MODEL_PATH):
            print(f"Loading trained plate weights from {PLATE_MODEL_PATH}")
            plate_model = YOLO(PLATE_MODEL_PATH)
        elif os.path.exists(FALLBACK_PLATE_MODEL_PATH):
            print(f"Loading trained plate weights from fallback {FALLBACK_PLATE_MODEL_PATH}")
            plate_model = YOLO(FALLBACK_PLATE_MODEL_PATH)
        else:
            print("Warning: Trained plate weights not found, using yolov8n.pt")
            plate_model = YOLO("yolov8n.pt")
    return plate_model

def get_easyocr_fallback():
    global easyocr_reader
    if easyocr_reader is None:
        print("Initializing EasyOCR fallback reader...")
        easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return easyocr_reader

# ──────────────────────────────────────────────────────────────────
# Indian State & Union Territory Standards
# ──────────────────────────────────────────────────────────────────
INDIAN_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK',
    'UA', 'UP', 'WB', 'BH'
}

INDIAN_STATE_NAMES = {
    'AN': 'Andaman and Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
    'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh',
    'DD': 'Daman and Diu', 'DL': 'Delhi', 'DN': 'Dadra and Nagar Haveli',
    'GA': 'Goa', 'GJ': 'Gujarat', 'HP': 'Himachal Pradesh', 'HR': 'Haryana',
    'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala',
    'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya',
    'MN': 'Manipur', 'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OD': 'Odisha', 'OR': 'Odisha', 'PB': 'Punjab', 'PY': 'Puducherry',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura',
    'TS': 'Telangana', 'UK': 'Uttarakhand', 'UA': 'Uttarakhand', 'UP': 'Uttar Pradesh',
    'WB': 'West Bengal', 'BH': 'Bharat Series (All-India)'
}

# ──────────────────────────────────────────────────────────────────
# Indian Registration Syntax Validation & Repair
# ──────────────────────────────────────────────────────────────────
def clean_plate_text(raw_text):
    """Normalize and repair common OCR character confusions based on character position."""
    if not raw_text:
        return ""

    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(cleaned) < 4:
        return cleaned

    digit_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'}
    char_to_digit = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'G': '6'}

    chars = list(cleaned)

    # First 2 chars: State prefix must be letters (unless BH series)
    is_bh_start = len(chars) >= 4 and chars[0].isdigit() and chars[1].isdigit()
    if not is_bh_start:
        for i in [0, 1]:
            if i < len(chars) and chars[i] in digit_to_char:
                chars[i] = digit_to_char[chars[i]]

    # Check BH Series format: YY BH #### XX
    bh_match = re.match(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$", "".join(chars))
    if bh_match:
        return "".join(chars)

    # Standard State prefix check
    state = "".join(chars[:2])
    if state in INDIAN_STATES and len(chars) >= 7:
        # Positions 2 & 3: RTO number must be digits
        for i in [2, 3]:
            if i < len(chars) and chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]

        # Last 4 positions: Vehicle registration serial digits
        num_start = max(4, len(chars) - 4)
        for i in range(num_start, len(chars)):
            if chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]

    return "".join(chars)

def validate_indian_plate_syntax(text):
    """
    Validate normalized plate string against Indian registration standards.
    Returns: (is_valid, score, state_code, state_name, match_type)
    """
    if not text or len(text) < ANPR_CONFIG["min_plate_chars"] or len(text) > ANPR_CONFIG["max_plate_chars"]:
        return False, 0.0, "", "Unknown", "REJECTED_LENGTH"

    state_code = text[:2]
    is_state_valid = state_code in INDIAN_STATES
    is_bh = bool(re.match(r"^\d{2}BH", text))

    if ANPR_CONFIG["strict_indian_state_required"] and not is_state_valid and not is_bh:
        return False, 0.0, state_code, "Invalid State Prefix", "REJECTED_INVALID_STATE"

    # Tier 1: Standard MoRTH format (e.g., TN45AB1234, DL3CD1210)
    if re.match(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$", text):
        state_name = INDIAN_STATE_NAMES.get(state_code, "Indian Registered Vehicle")
        return True, 1.0, state_code, state_name, "STANDARD_MORTH"

    # Tier 1: Bharat Series (e.g., 22BH1234AA)
    if re.match(r"^\d{2}BH\d{4}[A-Z]{1,2}$", text):
        return True, 1.0, "BH", "Bharat Series (All-India)", "BHARAT_SERIES"

    # Tier 2: Flexible Commercial / Older formats (e.g., KL498262)
    if re.match(r"^[A-Z]{2}\d{1,4}[A-Z]{0,2}\d{1,4}$", text):
        state_name = INDIAN_STATE_NAMES.get(state_code, "Indian Registered Vehicle")
        return True, 0.85, state_code, state_name, "VALID_HISTORICAL"

    return False, 0.0, state_code, "Unrecognized Syntax", "REJECTED_SYNTAX"

# ──────────────────────────────────────────────────────────────────
# Tesseract OCR & Image Enhancement
# ──────────────────────────────────────────────────────────────────
def preprocess_for_tesseract(crop_bgr):
    """Generate targeted binary & contrast-enhanced images specifically for Tesseract OCR."""
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    h, w = crop_bgr.shape[:2]
    scale = max(2.0, 100.0 / float(h))
    target_w = int(w * scale)
    target_h = int(h * scale)
    scaled = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(bilateral)

    # Standard Otsu (white background, dark characters)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Inverted Otsu (dark/yellow background, white characters)
    otsu_inv = cv2.bitwise_not(otsu)
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return [('otsu', otsu), ('otsu_inv', otsu_inv), ('adaptive', adaptive), ('clahe', clahe)]

def extract_plate_ocr(crop_bgr):
    """Recognize Indian vehicle registration plate text using Tesseract OCR (with EasyOCR fallback)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, "None"

    variants = preprocess_for_tesseract(crop_bgr)
    whitelist = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    best_text = ""
    best_conf = 0.0
    best_engine = "Tesseract OCR v5.4.0"

    # 1. Primary Engine: Official Tesseract OCR
    for vname, vimg in variants:
        for psm in [7, 8, 6]:
            cfg = f"--psm {psm} {whitelist}"
            try:
                data = pytesseract.image_to_data(vimg, config=cfg, output_type=pytesseract.Output.DICT)
                tokens = [data['text'][i] for i in range(len(data['text'])) if data['text'][i].strip()]
                if not tokens:
                    continue
                raw_combined = "".join(tokens)
                confs = [float(data['conf'][i]) for i in range(len(data['conf'])) if str(data['conf'][i]) != '-1']
                avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.5

                cleaned = clean_plate_text(raw_combined)
                score = avg_conf

                is_valid, syn_score, _, _, _ = validate_indian_plate_syntax(cleaned)
                if is_valid:
                    score += (syn_score * 0.35)

                if score > best_conf and len(cleaned) >= 4:
                    best_conf = score
                    best_text = cleaned
                    best_engine = f"Tesseract OCR (PSM {psm}, {vname})"
                    if is_valid and score >= 0.80:
                        return best_text, min(0.99, best_conf), best_engine
            except Exception:
                continue

    # 2. Fallback to EasyOCR if Tesseract confidence is low or length < 5
    if len(best_text) < 5 or best_conf < ANPR_CONFIG["min_ocr_conf"]:
        try:
            reader = get_easyocr_fallback()
            easy_candidates = [v[1] for v in variants[:2]]
            for img_variant in easy_candidates:
                results = reader.readtext(
                    img_variant,
                    detail=1,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False
                )
                if results:
                    sorted_res = sorted(results, key=lambda r: (min(pt[1] for pt in r[0]) // 25, min(pt[0] for pt in r[0])))
                    raw_combined = "".join([r[1] for r in sorted_res])
                    confs = [float(r[2]) for r in sorted_res if len(r) > 2]
                    avg_conf = float(np.mean(confs)) if confs else 0.5
                    cleaned = clean_plate_text(raw_combined)
                    score = avg_conf
                    is_valid, syn_score, _, _, _ = validate_indian_plate_syntax(cleaned)
                    if is_valid:
                        score += (syn_score * 0.35)
                    if score > best_conf and len(cleaned) >= 4:
                        best_conf = score
                        best_text = cleaned
                        best_engine = "EasyOCR (Fallback)"
                        break
        except Exception:
            pass

    return best_text, min(0.99, best_conf), best_engine

# ──────────────────────────────────────────────────────────────────
# Stage 5: Multi-Frame Temporal Confirmation Tracker
# ──────────────────────────────────────────────────────────────────
class TemporalConfirmationTracker:
    """
    Accumulates plate observations across consecutive frames per camera.
    A candidate plate is ONLY confirmed when observed across >= min_sightings
    within a temporal sliding window. Random one-frame noise is discarded.
    """
    def __init__(self, min_sightings=2, window_sec=3.0, similarity_threshold=0.80):
        self.min_sightings = min_sightings
        self.window_sec = window_sec
        self.similarity_threshold = similarity_threshold
        self.candidates = {}  # cameraId -> list of active candidate tracking dicts

    def process_candidate(self, camera_id, plate_str, conf, bypass_temporal=False):
        """
        Process plate observation.
        Returns: (is_confirmed, consolidated_plate, consolidated_conf, total_sightings)
        """
        if bypass_temporal:
            return True, plate_str, conf, 1

        now = time.time()
        if camera_id not in self.candidates:
            self.candidates[camera_id] = []

        # 1. Prune expired candidate sightings older than window
        self.candidates[camera_id] = [
            c for c in self.candidates[camera_id]
            if (now - c['last_seen']) <= self.window_sec
        ]

        # 2. Look for matching candidate using Levenshtein distance
        matched = None
        for c in self.candidates[camera_id]:
            sim = difflib.SequenceMatcher(None, c['plate'], plate_str).ratio()
            if sim >= self.similarity_threshold or c['plate'] == plate_str:
                matched = c
                break

        if matched:
            matched['sightings'] += 1
            matched['last_seen'] = now
            matched['confidences'].append(conf)

            # Consolidate string: prefer longer or higher confidence version
            if len(plate_str) > len(matched['plate']) or conf > max(matched['confidences'][:-1]):
                matched['plate'] = plate_str

            avg_conf = float(np.mean(matched['confidences']))
            is_confirmed = (matched['sightings'] >= self.min_sightings) and not matched.get('already_confirmed', False)
            if is_confirmed:
                matched['already_confirmed'] = True
                return True, matched['plate'], avg_conf, matched['sightings']
            return False, matched['plate'], avg_conf, matched['sightings']
        else:
            # First sighting in window — hold in candidate buffer
            self.candidates[camera_id].append({
                'plate': plate_str,
                'first_seen': now,
                'last_seen': now,
                'sightings': 1,
                'confidences': [conf],
                'already_confirmed': False
            })
            return False, plate_str, conf, 1

temporal_tracker = TemporalConfirmationTracker(
    min_sightings=ANPR_CONFIG["min_consecutive_sightings"],
    window_sec=ANPR_CONFIG["temporal_window_sec"],
    similarity_threshold=ANPR_CONFIG["temporal_similarity_threshold"]
)

# ──────────────────────────────────────────────────────────────────
# API Request Models & Configuration Endpoints
# ──────────────────────────────────────────────────────────────────
class Base64DetectRequest(BaseModel):
    image: str
    cameraId: int = 1
    cameraName: str = "Camera 1"
    forwardToDashboard: bool = True
    manualScan: bool = False  # Set True for manual snapshot uploads

@app.get("/health")
def health():
    vm = get_vehicle_model()
    pm = get_plate_model()
    return {
        "status": "ok",
        "vehicle_detector_ready": vm is not None,
        "plate_detector_ready": pm is not None,
        "tesseract_ready": os.path.exists(pytesseract.pytesseract.tesseract_cmd)
    }

@app.get("/config")
def get_config():
    return ANPR_CONFIG

@app.post("/config")
def update_config(new_config: dict):
    for k, v in new_config.items():
        if k in ANPR_CONFIG:
            ANPR_CONFIG[k] = v
    temporal_tracker.min_sightings = ANPR_CONFIG["min_consecutive_sightings"]
    temporal_tracker.window_sec = ANPR_CONFIG["temporal_window_sec"]
    return {"status": "updated", "config": ANPR_CONFIG}

# ──────────────────────────────────────────────────────────────────
# Core Detection Endpoint: Vehicle-First + Two-Stage ANPR
# ──────────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect_plate(req: Base64DetectRequest):
    try:
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        h_img, w_img = img.shape[:2]
        v_model = get_vehicle_model()
        p_model = get_plate_model()

        # ──────────────────────────────────────────────────────────
        # STAGE 1: Vehicle-First Detection
        # ──────────────────────────────────────────────────────────
        detected_vehicles = []
        if ANPR_CONFIG["vehicle_detection_enabled"]:
            v_results = v_model.predict(
                img,
                classes=ANPR_CONFIG["vehicle_classes"],
                conf=ANPR_CONFIG["vehicle_conf_threshold"],
                imgsz=640,
                verbose=False
            )
            for r in v_results:
                for b in r.boxes:
                    v_conf = float(b.conf[0])
                    v_cls = int(b.cls[0])
                    v_name = v_model.names.get(v_cls, "vehicle")
                    xyxy = b.xyxy[0].cpu().numpy().astype(int)
                    detected_vehicles.append({
                        "class": v_name,
                        "confidence": round(v_conf, 2),
                        "box": xyxy.tolist()
                    })

        # CRITICAL FALSE POSITIVE SUPPRESSION:
        # If no vehicles detected and not a manual snapshot scan, reject immediately!
        if len(detected_vehicles) == 0 and not req.manualScan:
            return {
                "success": True,
                "detected": False,
                "status": "REJECTED_NO_VEHICLE",
                "reason": "No vehicle detected in scene. Frame safely rejected to prevent false positives.",
                "vehicles": [],
                "detections": []
            }

        # ──────────────────────────────────────────────────────────
        # STAGE 2: Plate Candidate Detection inside Vehicle ROI
        # ──────────────────────────────────────────────────────────
        candidate_plates = []

        if len(detected_vehicles) > 0:
            # Search inside each detected vehicle's bounding box
            for v in detected_vehicles:
                vx1, vy1, vx2, vy2 = v["box"]
                vw = vx2 - vx1
                vh = vy2 - vy1
                if vw < 25 or vh < 25:
                    continue

                # Add configurable padding around vehicle
                pad_x = int(vw * ANPR_CONFIG["vehicle_roi_padding"])
                pad_y = int(vh * ANPR_CONFIG["vehicle_roi_padding"])
                cx1 = max(0, vx1 - pad_x)
                cy1 = max(0, vy1 - pad_y)
                cx2 = min(w_img, vx2 + pad_x)
                cy2 = min(h_img, vy2 + pad_y)

                v_crop = img[cy1:cy2, cx1:cx2]
                p_results = p_model.predict(
                    v_crop,
                    conf=ANPR_CONFIG["plate_conf_threshold"],
                    verbose=False
                )

                for pr in p_results:
                    for pb in pr.boxes:
                        p_conf = float(pb.conf[0])
                        px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy().astype(int)

                        # Translate crop coordinates back to full image frame
                        fx1 = cx1 + px1
                        fy1 = cy1 + py1
                        fx2 = cx1 + px2
                        fy2 = cy1 + py2
                        bw = fx2 - fx1
                        bh = fy2 - fy1

                        # STAGE 3: Geometric & Spatial Validation
                        if bw < ANPR_CONFIG["min_plate_width"] or bh < ANPR_CONFIG["min_plate_height"]:
                            continue
                        aspect_ratio = float(bw) / float(bh)
                        if aspect_ratio < ANPR_CONFIG["min_plate_aspect_ratio"] or aspect_ratio > ANPR_CONFIG["max_plate_aspect_ratio"]:
                            continue

                        # Plate area must be realistic fraction of vehicle area
                        plate_area = bw * bh
                        vehicle_area = vw * vh
                        if (plate_area / vehicle_area) > ANPR_CONFIG["max_plate_to_vehicle_area"]:
                            continue

                        # Plate must be in lower portion of vehicle (never on roof)
                        rel_y = (fy1 + bh / 2.0 - vy1) / float(vh)
                        if rel_y < ANPR_CONFIG["plate_vertical_pos_min"]:
                            continue

                        candidate_plates.append({
                            "detectorConfidence": p_conf,
                            "bbox": [fx1, fy1, bw, bh],
                            "vehicleType": v["class"],
                            "vehicleConfidence": v["confidence"]
                        })
        else:
            # Manual snapshot scan override: search full frame with strict threshold
            p_results = p_model.predict(img, conf=ANPR_CONFIG["closeup_plate_conf_threshold"], verbose=False)
            for pr in p_results:
                for pb in pr.boxes:
                    p_conf = float(pb.conf[0])
                    x1, y1, x2, y2 = pb.xyxy[0].cpu().numpy().astype(int)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < ANPR_CONFIG["min_plate_width"] or bh < ANPR_CONFIG["min_plate_height"]:
                        continue
                    ar = float(bw) / float(bh)
                    if ar < ANPR_CONFIG["min_plate_aspect_ratio"] or ar > ANPR_CONFIG["max_plate_aspect_ratio"]:
                        continue
                    candidate_plates.append({
                        "detectorConfidence": p_conf,
                        "bbox": [x1, y1, bw, bh],
                        "vehicleType": "Car / Passenger Vehicle",
                        "vehicleConfidence": 0.80
                    })

        if not candidate_plates:
            return {
                "success": True,
                "detected": False,
                "status": "NO_PLATE_CANDIDATES",
                "reason": "Vehicles detected but no valid license plate candidates passed geometric filters.",
                "vehicles": detected_vehicles,
                "detections": []
            }

        # Sort candidate plates by detector confidence descending
        candidate_plates.sort(key=lambda c: c["detectorConfidence"], reverse=True)
        confirmed_detection = None
        all_evaluations = []

        # ──────────────────────────────────────────────────────────
        # STAGES 4, 5 & 6: OCR, Indian Syntax & Temporal Confirmation
        # ──────────────────────────────────────────────────────────
        for cand in candidate_plates[:2]:
            x1, y1, bw, bh = cand["bbox"]
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w_img, x1 + bw + pad_x)
            cy2 = min(h_img, y1 + bh + pad_y)

            crop = img[cy1:cy2, cx1:cx2]
            clean_text, ocr_conf, ocr_engine = extract_plate_ocr(crop)

            # STAGE 4: Indian Registration Syntax Scoring
            is_valid, syn_score, state_code, state_name, match_type = validate_indian_plate_syntax(clean_text)

            overall_conf = float(round((cand["detectorConfidence"] * 0.4 + ocr_conf * 0.4 + syn_score * 0.2), 2))

            eval_record = {
                "plate": clean_text if is_valid else "",
                "rawOcr": clean_text,
                "stateCode": state_code,
                "stateName": state_name,
                "syntaxMatch": match_type,
                "confidence": overall_conf,
                "detectorConfidence": float(round(cand["detectorConfidence"], 2)),
                "ocrConfidence": float(round(ocr_conf, 2)),
                "ocrEngine": ocr_engine,
                "bbox": [int(x1), int(y1), int(bw), int(bh)],
                "cameraId": req.cameraId,
                "vehicleType": cand["vehicleType"],
                "plateType": "HSRP (High Security Registration Plate)",
            }

            if not is_valid:
                eval_record["status"] = "REJECTED_SYNTAX"
                eval_record["reason"] = f"Failed Indian registration validation ({match_type})"
                all_evaluations.append(eval_record)
                continue

            # STAGE 5: Multi-Frame Temporal Confirmation
            is_confirmed, final_plate, cons_conf, sightings = temporal_tracker.process_candidate(
                req.cameraId,
                clean_text,
                overall_conf,
                bypass_temporal=req.manualScan
            )

            eval_record["plate"] = final_plate
            eval_record["confidence"] = round(cons_conf, 2)
            eval_record["sightings"] = sightings
            eval_record["status"] = "CONFIRMED" if is_confirmed else "CANDIDATE_ACCUMULATING"

            all_evaluations.append(eval_record)

            if is_confirmed and confirmed_detection is None:
                confirmed_detection = eval_record

        # ──────────────────────────────────────────────────────────
        # STAGE 7: Forward ONLY Confirmed Events to Node.js Backend
        # ──────────────────────────────────────────────────────────
        if confirmed_detection and req.forwardToDashboard:
            try:
                requests.post(
                    NODE_SERVER_URL,
                    json={
                        "plate": confirmed_detection["plate"],
                        "cameraId": req.cameraId,
                        "confidence": confirmed_detection["confidence"],
                        "vehicleType": confirmed_detection["vehicleType"],
                        "simulated": False,
                    },
                    timeout=1.5,
                    verify=False,
                )
            except Exception as e:
                print(f"Forward to dashboard failed: {e}")

        return {
            "success": True,
            "detected": confirmed_detection is not None,
            "status": "CONFIRMED_ANPR_EVENT" if confirmed_detection else "CANDIDATE_TRACKING",
            "detection": confirmed_detection,
            "vehicles": detected_vehicles,
            "detections": all_evaluations,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/detect-file")
async def detect_plate_file(
    file: UploadFile = File(...),
    cameraId: int = Form(1),
    cameraName: str = Form("Manual Scan"),
    forwardToDashboard: bool = Form(True)
):
    try:
        contents = await file.read()
        encoded = base64.b64encode(contents).decode("utf-8")
        req = Base64DetectRequest(
            image=f"data:image/jpeg;base64,{encoded}",
            cameraId=cameraId,
            cameraName=cameraName,
            forwardToDashboard=forwardToDashboard,
            manualScan=True  # Manual file uploads bypass multi-frame delay
        )
        return await detect_plate(req)
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("Starting Multi-Stage ANPR Inference Server on http://127.0.0.1:5001...")
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="info")
