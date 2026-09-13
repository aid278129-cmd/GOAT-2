"""
anpr_server.py
High-Performance Local ANPR Inference Service for Indian License Plates
SIH Problem Statement ID: 26127 (BEL)

Layered Vision Pipeline with Comprehensive 16-Point Per-Frame Diagnostics:
  1. Vehicle-First Detection (YOLOv8 COCO: car, motorcycle, bus, truck)
  2. Vehicle ROI Extraction & Spatial Verification
  3. License Plate Candidate Detection inside Vehicle ROI (Trained Indian Plate YOLOv8)
  4. Geometric & Aspect Ratio Filtering
  5. Optimized OCR (Tesseract v5.4.0 with CLAHE/Dual-Otsu + EasyOCR fallback)
  6. Indian Registration Syntax Scoring & Positional Disambiguation
  7. Multi-Frame Temporal Confirmation Tracker (Rejects isolated single-frame noise)
  8. Confirmed ANPR Event Forwarding to Node.js Backend
  9. Full Frame & Plate Crop Archival + Real-time 16-point Diagnostics Telemetry
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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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
FALLBACK_PLATE_MODEL_PATH = PLATE_MODEL_PATH
NODE_SERVER_URL = "https://127.0.0.1:3000/api/detections"

# Debug output directories
DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")
DEBUG_FRAMES_DIR = os.path.join(DEBUG_DIR, "frames")
DEBUG_CROPS_DIR = os.path.join(DEBUG_DIR, "crops")

os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)

# Mount debug static directory so client/browser can inspect saved frames and crops
app.mount("/debug_output", StaticFiles(directory=DEBUG_DIR), name="debug_output")

# Set Tesseract binary path
TESSERACT_DEFAULT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_DEFAULT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_DEFAULT_PATH
    print(f"Configured Tesseract OCR binary: {TESSERACT_DEFAULT_PATH}")
else:
    print("Using system default tesseract command")

# ──────────────────────────────────────────────────────────────────
# Centralized, Configurable Settings (Runtime Configurable)
# ──────────────────────────────────────────────────────────────────
ANPR_CONFIG = {
    # Stage 1: Vehicle Detection
    "vehicle_detection_enabled": True,
    "vehicle_classes": [2, 3, 5, 7],         # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    "vehicle_conf_threshold": 0.22,           # Lowered for partial bumper framing
    "vehicle_roi_padding": 0.05,             # 5% padding around vehicle crop

    # Stage 2: License Plate Candidate Detection
    "plate_conf_threshold": 0.40,             # Min plate candidate confidence
    "closeup_plate_conf_threshold": 0.45,     # Threshold if searching full frame
    "allow_fullframe_fallback": True,         # Fall back to full-frame plate search if 0 vehicles detected

    # Stage 3: Geometric & Spatial Validation
    "min_plate_aspect_ratio": 1.1,           # Width / Height (1.1 allows MoRTH Rule 50 square/two-row plates)
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
    "temporal_window_sec": 10.0,             # Sliding window duration (10s handles CPU inference and traversal)
    "min_consecutive_sightings": 2,          # Minimum frames to confirm (isolated 1-frame spikes rejected)
    "temporal_similarity_threshold": 0.80,  # String similarity ratio

    # Diagnostics & Storage
    "save_debug_frames": True,               # Save exact incoming WebRTC frame to disk
    "save_debug_crops": True,                # Save pre-OCR plate crops to disk
    "max_saved_frames": 200,                 # Frame cache limit

    # Phase 2: Intelligent Keyframe Selection & Sharpness Filtering
    "keyframe_selection": True,              # Select sharpest frame in temporal window
    "minimum_sharpness": 60.0,               # Minimum Laplacian variance to reject blurry frames
    "keyframe_window_ms": 1000,              # Candidate evaluation rolling window (ms)
    "anpr_sampling_interval": 1000,          # Interval between ANPR inference dispatches (ms)
}

# ──────────────────────────────────────────────────────────────────
# Diagnostic Memory Store
# ──────────────────────────────────────────────────────────────────
RECENT_DEBUG_LOGS: List[Dict[str, Any]] = []
LATEST_DEBUG_PER_CAM: Dict[int, Dict[str, Any]] = {}
DEBUG_STATS = {
    "total_frames_processed": 0,
    "vehicle_detected_count": 0,
    "plate_attempted_count": 0,
    "plate_candidates_found_count": 0,
    "ocr_attempted_count": 0,
    "syntax_valid_count": 0,
    "confirmed_count": 0,
    "rejection_reasons_breakdown": {}
}

def record_rejection_stat(reason_code: str):
    prefix = reason_code.split(":")[0].strip() if ":" in reason_code else reason_code
    DEBUG_STATS["rejection_reasons_breakdown"][prefix] = (
        DEBUG_STATS["rejection_reasons_breakdown"].get(prefix, 0) + 1
    )

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
STATE_REPAIRS = {
    'MG': 'MH', 'MN': 'MH', 'NH': 'MH',
    'OL': 'DL', 'D1': 'DL', 'DI': 'DL',
    'K1': 'KL', 'KI': 'KL',
    'TM': 'TN', 'TI': 'TN',
    'HR': 'HR', 'HA': 'HR',
    'GJ': 'GJ', 'CJ': 'GJ',
    'VP': 'UP', 'UF': 'UP',
    'AP': 'AP', 'AF': 'AP',
    'TS': 'TS', 'T5': 'TS',
    'WB': 'WB', 'WE': 'WB',
    'PB': 'PB', 'P8': 'PB',
    'RJ': 'RJ', 'R1': 'RJ',
}

def repair_inverted_two_row(text):
    """
    On Indian square / 2-row plates (Rule 50 MoRTH), OCR frequently reads the bottom row 
    (registration number) before the top row (State + District RTO code).
    E.g. '0074AHAP29' -> 'AP29AH0074'.
    """
    if len(text) < 7:
        return text
    # Check if a valid Indian state prefix is embedded inside or at the end
    for i in range(1, len(text) - 1):
        st = text[i:i+2]
        repaired_st = STATE_REPAIRS.get(st, st)
        if repaired_st in INDIAN_STATES:
            rto_match = re.match(r"^([A-Z]{2}\d{1,2})", repaired_st + text[i+2:])
            if rto_match:
                state_rto = rto_match.group(1)
                rem = text[:i] + text[i+len(state_rto):]
                rem_letters = re.findall(r"[A-Z]+", rem)
                rem_digits = re.findall(r"\d+", rem)
                return f"{state_rto}{''.join(rem_letters)}{''.join(rem_digits)}"
    return text

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

    # Check BH Series format: YY BH #### XX
    is_bh_candidate = len(chars) >= 6 and chars[0].isdigit() and chars[1].isdigit() and "".join(chars[2:4]) in ['BH', '8H']
    if is_bh_candidate:
        chars[2] = 'B'
        chars[3] = 'H'
        for i in range(4, min(8, len(chars))):
            if chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]
        return "".join(chars)

    # Check if inverted 2-row plate before character substitutions
    if "".join(chars[:2]) not in INDIAN_STATES:
        reordered = repair_inverted_two_row("".join(chars))
        if reordered[:2] in INDIAN_STATES:
            chars = list(reordered)

    # First 2 chars: State prefix must be letters
    for i in [0, 1]:
        if i < len(chars) and chars[i] in digit_to_char:
            chars[i] = digit_to_char[chars[i]]

    # Standard State prefix check & optical repair
    state = "".join(chars[:2])
    if state not in INDIAN_STATES and len(chars) >= 6:
        if state in STATE_REPAIRS:
            repaired = STATE_REPAIRS[state]
            chars[0] = repaired[0]
            chars[1] = repaired[1]
            state = repaired
        else:
            reordered = repair_inverted_two_row("".join(chars))
            if reordered[:2] in INDIAN_STATES:
                chars = list(reordered)
                state = "".join(chars[:2])

    if state in INDIAN_STATES and len(chars) >= 6:
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
        return False, 0.0, "", "Unknown", f"REJECTED_LENGTH (got {len(text)} chars, expected {ANPR_CONFIG['min_plate_chars']}-{ANPR_CONFIG['max_plate_chars']})"

    state_code = text[:2]
    is_state_valid = state_code in INDIAN_STATES
    is_bh = bool(re.match(r"^\d{2}BH", text))

    if ANPR_CONFIG["strict_indian_state_required"] and not is_state_valid and not is_bh:
        return False, 0.0, state_code, "Invalid State Prefix", f"REJECTED_INVALID_STATE (prefix '{state_code}' not a recognised Indian State/UT code)"

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

    return False, 0.0, state_code, "Unrecognized Syntax", "REJECTED_SYNTAX (Pattern mismatch against MoRTH/BH series)"

# ──────────────────────────────────────────────────────────────────
# Perspective & Skew Correction (Phase 1)
# ──────────────────────────────────────────────────────────────────
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def rectify_plate_perspective(crop_bgr):
    """
    Detect plate boundary or line orientation and apply 4-point perspective
    transformation or rotation to rectify skewed / angled license plates.
    Returns: (rectified_crop, skew_degrees, status_str)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr, 0.0, "SKIPPED_EMPTY"

    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 20:
        return crop_bgr, 0.0, "SKIPPED_TOO_SMALL"

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Method 1: Quadrilateral Contour Detection (4-point Homography)
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 40, 160)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.20 * (w * h):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.035 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype("float32")
                rect = order_points(pts)
                (tl, tr, br, bl) = rect

                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                if maxWidth < 25 or maxHeight < 10:
                    continue

                ar = maxWidth / float(max(1, maxHeight))
                if ar < 1.0 or ar > 7.0:
                    continue

                dx = tr[0] - tl[0]
                dy = tr[1] - tl[1]
                skew_deg = float(np.degrees(np.arctan2(dy, dx)))

                if abs(skew_deg) < 1.5:
                    return crop_bgr, float(round(skew_deg, 2)), "SKIPPED_NEGLIGIBLE_SKEW"

                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                rectified = cv2.warpPerspective(crop_bgr, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rectified, float(round(skew_deg, 2)), "APPLIED_4PT_HOMOGRAPHY"
    except Exception:
        pass

    # Method 2: Hough Line Skew Analysis (Detect plate borders & text baseline)
    try:
        edged = cv2.Canny(gray, 40, 160)
        min_len = max(20, int(w * 0.25))
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=15)
        weighted_angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.ravel()
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                deg = float(np.degrees(np.arctan2(dy, dx)))
                length = np.sqrt(dx * dx + dy * dy)
                if abs(deg) <= 20.0:
                    weighted_angles.append((deg, length))

        if len(weighted_angles) >= 2:
            weighted_angles.sort(key=lambda x: x[1], reverse=True)
            top_angles = [a[0] for a in weighted_angles[:10]]
            median_skew = float(np.median(top_angles))
            if abs(median_skew) >= 1.5:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_skew, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                rectified = cv2.warpAffine(crop_bgr, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rectified, float(round(median_skew, 2)), "APPLIED_HOUGH_ROTATION"
            else:
                return crop_bgr, float(round(median_skew, 2)), "SKIPPED_NEGLIGIBLE_SKEW"
    except Exception:
        pass

    return crop_bgr, 0.0, "SKIPPED_NO_SIGNIFICANT_SKEW"

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
    """
    Recognize Indian vehicle registration plate text using Tesseract OCR (with EasyOCR fallback).
    Returns: (cleaned_text, raw_text, ocr_conf, ocr_engine)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", "", 0.0, "None"

    variants = preprocess_for_tesseract(crop_bgr)
    whitelist = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    best_text = ""
    best_raw = ""
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
                    best_raw = raw_combined
                    best_engine = f"Tesseract OCR (PSM {psm}, {vname})"
                    if is_valid and score >= 0.80:
                        return best_text, best_raw, min(0.99, best_conf), best_engine
            except Exception:
                continue

    # 1b. Multi-Row Stacked Line OCR for Two-Row Indian Plates (AR < 2.0)
    h_c, w_c = crop_bgr.shape[:2]
    crop_ar = w_c / float(max(1, h_c))
    if crop_ar < 2.0:
        try:
            top_half = crop_bgr[:int(h_c * 0.55), :]
            bot_half = crop_bgr[int(h_c * 0.45):, :]
            top_vars = preprocess_for_tesseract(top_half)
            bot_vars = preprocess_for_tesseract(bot_half)
            if top_vars and bot_vars:
                t_data = pytesseract.image_to_data(top_vars[0][1], config=f"--psm 7 {whitelist}", output_type=pytesseract.Output.DICT)
                b_data = pytesseract.image_to_data(bot_vars[0][1], config=f"--psm 7 {whitelist}", output_type=pytesseract.Output.DICT)
                t_tokens = [t_data['text'][i] for i in range(len(t_data['text'])) if t_data['text'][i].strip()]
                b_tokens = [b_data['text'][i] for i in range(len(b_data['text'])) if b_data['text'][i].strip()]
                if t_tokens and b_tokens:
                    t_str = "".join(t_tokens)
                    b_str = "".join(b_tokens)
                    stacked_raw = f"{t_str}{b_str}"
                    stacked_clean = clean_plate_text(stacked_raw)
                    is_valid, syn_score, _, _, _ = validate_indian_plate_syntax(stacked_clean)
                    if is_valid:
                        return stacked_clean, stacked_raw, 0.88, "Tesseract OCR (Two-Row Sliced)"
        except Exception:
            pass

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
                    h_v, w_v = img_variant.shape[:2]
                    var_ar = w_v / float(max(1, h_v))
                    if var_ar >= 2.0:
                        # Single-row horizontal plate: sort left-to-right by X
                        sorted_res = sorted(results, key=lambda r: min(pt[0] for pt in r[0]))
                    else:
                        # Two-row square plate: sort by row then by X
                        row_h = max(20, h_v // 2)
                        sorted_res = sorted(results, key=lambda r: (min(pt[1] for pt in r[0]) // row_h, min(pt[0] for pt in r[0])))
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
                        best_raw = raw_combined
                        best_engine = "EasyOCR (Fallback)"
                        break
        except Exception:
            pass

    return best_text, best_raw, min(0.99, best_conf), best_engine

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
    developerMode: bool = False
    keyframeSharpness: Optional[float] = None
    keyframeWindowMs: Optional[int] = None
    keyframeCandidates: Optional[int] = None

@app.get("/health")
def health():
    vm = get_vehicle_model()
    pm = get_plate_model()
    return {
        "status": "ok",
        "vehicle_detector_ready": vm is not None,
        "plate_detector_ready": pm is not None,
        "tesseract_ready": os.path.exists(pytesseract.pytesseract.tesseract_cmd),
        "debug_output_ready": os.path.exists(DEBUG_FRAMES_DIR) and os.path.exists(DEBUG_CROPS_DIR),
        "config": ANPR_CONFIG
    }

@app.get("/config")
def get_config():
    """Expose runtime-configurable values."""
    return ANPR_CONFIG

@app.post("/config")
def update_config(new_config: dict):
    """Update runtime-configurable values without restarting server."""
    for k, v in new_config.items():
        if k in ANPR_CONFIG:
            ANPR_CONFIG[k] = type(ANPR_CONFIG[k])(v)
    temporal_tracker.min_sightings = ANPR_CONFIG["min_consecutive_sightings"]
    temporal_tracker.window_sec = ANPR_CONFIG["temporal_window_sec"]
    return {"status": "updated", "config": ANPR_CONFIG}

@app.get("/debug/last")
def get_debug_last(cameraId: Optional[int] = None):
    """Return latest 16-point debug record for specified camera or all cameras."""
    if cameraId is not None:
        return LATEST_DEBUG_PER_CAM.get(cameraId, {"message": f"No debug frames yet for Camera {cameraId}"})
    return LATEST_DEBUG_PER_CAM

@app.get("/debug/history")
def get_debug_history(limit: int = 20):
    """Return history of recent per-frame debug logs."""
    return RECENT_DEBUG_LOGS[-min(limit, 100):]

@app.get("/debug/stats")
def get_debug_stats():
    """Return comparative pipeline funnel diagnostics statistics."""
    total = DEBUG_STATS["total_frames_processed"]
    return {
        **DEBUG_STATS,
        "pass_rates": {
            "vehicle_detection_rate": round(DEBUG_STATS["vehicle_detected_count"] / max(1, total), 3),
            "plate_localization_rate": round(DEBUG_STATS["plate_candidates_found_count"] / max(1, total), 3),
            "syntax_validation_rate": round(DEBUG_STATS["syntax_valid_count"] / max(1, total), 3),
            "final_confirmation_rate": round(DEBUG_STATS["confirmed_count"] / max(1, total), 3),
        }
    }

@app.get("/debug/frames")
def list_debug_frames(limit: int = 20):
    """List recently saved incoming frames and plate crops."""
    frames = sorted(os.listdir(DEBUG_FRAMES_DIR), reverse=True)[:limit]
    crops = sorted(os.listdir(DEBUG_CROPS_DIR), reverse=True)[:limit]
    return {
        "frames": [f"/debug_output/frames/{f}" for f in frames],
        "crops": [f"/debug_output/crops/{c}" for c in crops]
    }

# ──────────────────────────────────────────────────────────────────
# Core Detection Endpoint: Comprehensive 16-Point Diagnostics
# ──────────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect_plate(req: Base64DetectRequest):
    try:
        DEBUG_STATS["total_frames_processed"] += 1
        now_ts = int(time.time() * 1000)
        frame_id = f"cam{req.cameraId}_{now_ts}"

        # Decode incoming frame
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        h_img, w_img = img.shape[:2]

        # ── EXACT INCOMING FRAME ARCHIVAL ───────────────────────────
        # Save exact incoming frame from mobile WebRTC stream before ANY processing
        frame_filename = f"frame_{frame_id}.jpg"
        frame_saved_path = os.path.join(DEBUG_FRAMES_DIR, frame_filename)
        frame_relative_url = f"/debug_output/frames/{frame_filename}"

        if ANPR_CONFIG.get("save_debug_frames", True):
            cv2.imwrite(frame_saved_path, img)

        # Cap max frame dimension to 1280 for real-time inference latency while preserving full detail
        max_dim = 1280
        if max(h_img, w_img) > max_dim:
            scale = max_dim / float(max(h_img, w_img))
            img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)), interpolation=cv2.INTER_AREA)
            h_img, w_img = img.shape[:2]

        # Calculate Image Quality Metrics (sharpness / blur / brightness)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        min_sharpness = float(ANPR_CONFIG.get("minimum_sharpness", 60.0))
        is_blurry = laplacian_var < min_sharpness

        v_model = get_vehicle_model()
        p_model = get_plate_model()

        # ──────────────────────────────────────────────────────────
        # POINT 1, 2, 3: Vehicle Detection
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

        vehicle_detected = len(detected_vehicles) > 0
        if vehicle_detected:
            DEBUG_STATS["vehicle_detected_count"] += 1

        # ──────────────────────────────────────────────────────────
        # POINT 4: Plate Detection Attempt Evaluation
        # ──────────────────────────────────────────────────────────
        allow_fallback = ANPR_CONFIG.get("allow_fullframe_fallback", False)
        attempt_plate_detection = vehicle_detected or req.manualScan or (not ANPR_CONFIG["vehicle_detection_enabled"]) or allow_fallback

        if attempt_plate_detection:
            DEBUG_STATS["plate_attempted_count"] += 1
            if vehicle_detected:
                attempt_reason = f"Attempted plate detection inside {len(detected_vehicles)} vehicle ROI(s)"
            elif req.manualScan:
                attempt_reason = "Attempted full-frame plate detection (manual scan override)"
            elif not ANPR_CONFIG["vehicle_detection_enabled"]:
                attempt_reason = "Attempted full-frame plate detection (vehicle gating disabled)"
            else:
                attempt_reason = "Attempted full-frame plate detection (fallback enabled)"
        else:
            attempt_reason = f"SKIPPED: Zero vehicles detected in scene (threshold={ANPR_CONFIG['vehicle_conf_threshold']}). Vehicle gating active."

        # If vehicle gating rejects frame before plate attempt
        if not attempt_plate_detection:
            rejection_reason = f"REJECTED_NO_VEHICLE: No vehicle detected above confidence threshold ({ANPR_CONFIG['vehicle_conf_threshold']}). Plate detection skipped."
            record_rejection_stat("REJECTED_NO_VEHICLE")

            frame_telemetry = {
                "frameId": frame_id,
                "timestamp": now_ts,
                "cameraId": req.cameraId,
                "cameraName": req.cameraName,
                "frameSavedPath": frame_saved_path,
                "frameUrl": frame_relative_url,
                "resolution": f"{w_img}x{h_img}",
                "sharpnessScore": round(laplacian_var, 1),
                "isBlurry": is_blurry,
                "keyframeStatus": "BLURRY" if is_blurry else "ACCEPTED",
                "clientKeyframeSharpness": req.keyframeSharpness,
                "keyframeCandidates": req.keyframeCandidates,
                "meanBrightness": round(mean_brightness, 1),

                # 16-point diagnostic records
                "1_vehicleDetected": False,
                "2_vehicleClassAndConf": [],
                "3_vehicleBoundingBoxes": [],
                "4_plateDetectionAttempted": False,
                "4_plateAttemptReason": attempt_reason,
                "5_plateCandidateCount": 0,
                "6_plateConfidences": [],
                "7_plateBoundingBoxesAndAR": [],
                "8_candidatesValidationPassed": [],
                "9_candidateRejectionReasons": [rejection_reason],
                "10_plateCropsSaved": [],
                "11_ocrRawResults": [],
                "12_ocrConfidences": [],
                "13_normalizedOcrResults": [],
                "14_indianFormatValidationResults": [],
                "15_multiFrameConfirmationStatus": "SKIPPED",
                "16_finalDecision": "REJECTED",

                "summary": {
                    "status": "REJECTED_NO_VEHICLE",
                    "reason": rejection_reason,
                    "failureStage": "Stage 1: Vehicle Detection",
                },
                "candidates": []
            }

            LATEST_DEBUG_PER_CAM[req.cameraId] = frame_telemetry
            RECENT_DEBUG_LOGS.append(frame_telemetry)
            if len(RECENT_DEBUG_LOGS) > 100:
                RECENT_DEBUG_LOGS.pop(0)

            return {
                "success": True,
                "detected": False,
                "status": "REJECTED_NO_VEHICLE",
                "reason": rejection_reason,
                "vehicles": [],
                "detections": [],
                "debug": frame_telemetry
            }

        # ──────────────────────────────────────────────────────────
        # POINT 5 to 9: Plate Candidate Detection & Geometric Validation
        # ──────────────────────────────────────────────────────────
        raw_candidates_evaluated = []
        valid_candidates_to_ocr = []

        if vehicle_detected:
            # Search inside each detected vehicle's bounding box
            for v_idx, v in enumerate(detected_vehicles):
                vx1, vy1, vx2, vy2 = v["box"]
                vw = vx2 - vx1
                vh = vy2 - vy1
                if vw < 20 or vh < 20:
                    continue

                pad_x = int(vw * ANPR_CONFIG["vehicle_roi_padding"])
                pad_y = int(vh * ANPR_CONFIG["vehicle_roi_padding"])
                cx1 = max(0, vx1 - pad_x)
                cy1 = max(0, vy1 - pad_y)
                cx2 = min(w_img, vx2 + pad_x)
                cy2 = min(h_img, vy2 + pad_y)

                v_crop = img[cy1:cy2, cx1:cx2]
                p_results = p_model.predict(
                    v_crop,
                    conf=0.15,  # Run lower detector threshold internally so we can log rejected confidence candidates
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

                        aspect_ratio = float(bw) / float(max(1, bh))
                        plate_area = bw * bh
                        vehicle_area = max(1, vw * vh)
                        area_ratio = plate_area / float(vehicle_area)
                        rel_y = (fy1 + bh / 2.0 - vy1) / float(vh)

                        # Check validation criteria
                        reject_reason = None
                        passed_val = True

                        if p_conf < ANPR_CONFIG["plate_conf_threshold"]:
                            passed_val = False
                            reject_reason = f"REJECTED_BELOW_PLATE_CONF: confidence {p_conf:.2f} < {ANPR_CONFIG['plate_conf_threshold']:.2f}"
                        elif bw < ANPR_CONFIG["min_plate_width"] or bh < ANPR_CONFIG["min_plate_height"]:
                            passed_val = False
                            reject_reason = f"REJECTED_TOO_SMALL: size {bw}x{bh}px < min {ANPR_CONFIG['min_plate_width']}x{ANPR_CONFIG['min_plate_height']}px"
                        elif aspect_ratio < ANPR_CONFIG["min_plate_aspect_ratio"] or aspect_ratio > ANPR_CONFIG["max_plate_aspect_ratio"]:
                            passed_val = False
                            reject_reason = f"REJECTED_ASPECT_RATIO: AR {aspect_ratio:.2f} not in [{ANPR_CONFIG['min_plate_aspect_ratio']}, {ANPR_CONFIG['max_plate_aspect_ratio']}]"
                        elif area_ratio > ANPR_CONFIG["max_plate_to_vehicle_area"]:
                            passed_val = False
                            reject_reason = f"REJECTED_AREA_RATIO: plate area {area_ratio*100:.1f}% exceeds max {ANPR_CONFIG['max_plate_to_vehicle_area']*100:.0f}% of vehicle"
                        elif rel_y < ANPR_CONFIG["plate_vertical_pos_min"]:
                            passed_val = False
                            reject_reason = f"REJECTED_VERTICAL_POSITION: rel_y {rel_y:.2f} < min {ANPR_CONFIG['plate_vertical_pos_min']} (mounted too high)"

                        cand_obj = {
                            "candidateIndex": int(len(raw_candidates_evaluated)),
                            "detectorConfidence": float(round(p_conf, 3)),
                            "bbox": [int(fx1), int(fy1), int(bw), int(bh)],
                            "aspectRatio": float(round(aspect_ratio, 2)),
                            "size": f"{int(bw)}x{int(bh)}",
                            "passedValidation": bool(passed_val),
                            "rejectionReason": str(reject_reason) if reject_reason else None,
                            "vehicleClass": str(v["class"]),
                            "vehicleConfidence": float(v["confidence"]),
                            "vehicleBox": [int(x) for x in v["box"]]
                        }
                        raw_candidates_evaluated.append(cand_obj)
                        if passed_val:
                            valid_candidates_to_ocr.append(cand_obj)
        else:
            # Full-frame search (manual scan or fallback)
            conf_thresh = ANPR_CONFIG["closeup_plate_conf_threshold"] if not req.manualScan else 0.25
            p_results = p_model.predict(img, conf=0.15, verbose=False)
            for pr in p_results:
                for pb in pr.boxes:
                    p_conf = float(pb.conf[0])
                    x1, y1, x2, y2 = pb.xyxy[0].cpu().numpy().astype(int)
                    bw = x2 - x1
                    bh = y2 - y1
                    aspect_ratio = float(bw) / float(max(1, bh))

                    reject_reason = None
                    passed_val = True

                    if p_conf < conf_thresh:
                        passed_val = False
                        reject_reason = f"REJECTED_BELOW_PLATE_CONF: confidence {p_conf:.2f} < {conf_thresh:.2f}"
                    elif bw < ANPR_CONFIG["min_plate_width"] or bh < ANPR_CONFIG["min_plate_height"]:
                        passed_val = False
                        reject_reason = f"REJECTED_TOO_SMALL: size {bw}x{bh}px < min {ANPR_CONFIG['min_plate_width']}x{ANPR_CONFIG['min_plate_height']}px"
                    elif aspect_ratio < ANPR_CONFIG["min_plate_aspect_ratio"] or aspect_ratio > ANPR_CONFIG["max_plate_aspect_ratio"]:
                        passed_val = False
                        reject_reason = f"REJECTED_ASPECT_RATIO: AR {aspect_ratio:.2f} not in [{ANPR_CONFIG['min_plate_aspect_ratio']}, {ANPR_CONFIG['max_plate_aspect_ratio']}]"

                    cand_obj = {
                        "candidateIndex": int(len(raw_candidates_evaluated)),
                        "detectorConfidence": float(round(p_conf, 3)),
                        "bbox": [int(x1), int(y1), int(bw), int(bh)],
                        "aspectRatio": float(round(aspect_ratio, 2)),
                        "size": f"{int(bw)}x{int(bh)}",
                        "passedValidation": bool(passed_val),
                        "rejectionReason": str(reject_reason) if reject_reason else None,
                        "vehicleClass": "Vehicle (Full-Frame)",
                        "vehicleConfidence": 0.80,
                        "vehicleBox": [0, 0, int(w_img), int(h_img)]
                    }
                    raw_candidates_evaluated.append(cand_obj)
                    if passed_val:
                        valid_candidates_to_ocr.append(cand_obj)

        if len(raw_candidates_evaluated) > 0:
            DEBUG_STATS["plate_candidates_found_count"] += 1

        # If no candidates or all candidates failed geometric validation
        if not valid_candidates_to_ocr:
            primary_reason = (
                raw_candidates_evaluated[0]["rejectionReason"]
                if raw_candidates_evaluated
                else "NO_PLATE_CANDIDATES: Plate detector found 0 candidate boxes inside vehicle ROI."
            )
            record_rejection_stat(primary_reason)

            frame_telemetry = {
                "frameId": frame_id,
                "timestamp": now_ts,
                "cameraId": req.cameraId,
                "cameraName": req.cameraName,
                "frameSavedPath": frame_saved_path,
                "frameUrl": frame_relative_url,
                "resolution": f"{w_img}x{h_img}",
                "sharpnessScore": float(round(laplacian_var, 1)),
                "isBlurry": bool(is_blurry),
                "meanBrightness": float(round(mean_brightness, 1)),

                "1_vehicleDetected": bool(vehicle_detected),
                "2_vehicleClassAndConf": [{"class": str(v["class"]), "confidence": float(v["confidence"])} for v in detected_vehicles],
                "3_vehicleBoundingBoxes": [[int(x) for x in v["box"]] for v in detected_vehicles],
                "4_plateDetectionAttempted": True,
                "4_plateAttemptReason": str(attempt_reason),
                "5_plateCandidateCount": int(len(raw_candidates_evaluated)),
                "6_plateConfidences": [float(c["detectorConfidence"]) for c in raw_candidates_evaluated],
                "7_plateBoundingBoxesAndAR": [{"bbox": [int(x) for x in c["bbox"]], "size": str(c["size"]), "aspectRatio": float(c["aspectRatio"])} for c in raw_candidates_evaluated],
                "8_candidatesValidationPassed": [bool(c["passedValidation"]) for c in raw_candidates_evaluated],
                "9_candidateRejectionReasons": [str(c["rejectionReason"]) for c in raw_candidates_evaluated if c["rejectionReason"]],
                "10_plateCropsSaved": [],
                "11_ocrRawResults": [],
                "12_ocrConfidences": [],
                "13_normalizedOcrResults": [],
                "14_indianFormatValidationResults": [],
                "15_multiFrameConfirmationStatus": "SKIPPED_NO_VALID_GEOMETRY",
                "16_finalDecision": "REJECTED",

                "summary": {
                    "status": "REJECTED_GEOMETRY" if raw_candidates_evaluated else "REJECTED_NO_PLATES",
                    "reason": primary_reason,
                    "failureStage": "Stage 2: Plate Localization / Geometric Validation",
                },
                "candidates": raw_candidates_evaluated
            }

            LATEST_DEBUG_PER_CAM[req.cameraId] = frame_telemetry
            RECENT_DEBUG_LOGS.append(frame_telemetry)
            if len(RECENT_DEBUG_LOGS) > 100:
                RECENT_DEBUG_LOGS.pop(0)

            return {
                "success": True,
                "detected": False,
                "status": "NO_PLATE_CANDIDATES",
                "reason": primary_reason,
                "vehicles": detected_vehicles,
                "detections": [],
                "debug": frame_telemetry
            }

        # Sort valid candidates by detector confidence descending
        valid_candidates_to_ocr.sort(key=lambda c: c["detectorConfidence"], reverse=True)
        confirmed_detection = None
        all_evaluations = []

        # ──────────────────────────────────────────────────────────
        # POINT 10 to 16: Plate Crop Archival, OCR, Syntax & Multi-Frame Confirmation
        # ──────────────────────────────────────────────────────────
        DEBUG_STATS["ocr_attempted_count"] += 1

        for c_idx, cand in enumerate(valid_candidates_to_ocr[:2]):
            x1, y1, bw, bh = cand["bbox"]
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w_img, x1 + bw + pad_x)
            cy2 = min(h_img, y1 + bh + pad_y)

            crop = img[cy1:cy2, cx1:cx2]

            # Phase 1: Perspective / Skew Rectification (Confidence-Aware Dual Candidate)
            rectified_crop, skew_deg, persp_status = rectify_plate_perspective(crop)
            persp_applied = persp_status.startswith("APPLIED")

            # POINT 10: Save plate crop before OCR
            crop_filename = f"crop_{frame_id}_cand{c_idx}.jpg"
            crop_orig_filename = f"crop_{frame_id}_cand{c_idx}_orig.jpg"
            crop_rect_filename = f"crop_{frame_id}_cand{c_idx}_rect.jpg"

            crop_saved_path = os.path.join(DEBUG_CROPS_DIR, crop_filename)
            crop_orig_path = os.path.join(DEBUG_CROPS_DIR, crop_orig_filename)
            crop_rect_path = os.path.join(DEBUG_CROPS_DIR, crop_rect_filename)
            crop_relative_url = f"/debug_output/crops/{crop_filename}"

            if ANPR_CONFIG.get("save_debug_crops", True) and crop.size > 0:
                cv2.imwrite(crop_saved_path, crop)
                cv2.imwrite(crop_orig_path, crop)
                if persp_applied and rectified_crop is not None and rectified_crop.size > 0:
                    cv2.imwrite(crop_rect_path, rectified_crop)

            # POINT 11, 12, 13: OCR Recognition (Dual-Candidate Evaluation)
            orig_clean, orig_raw, orig_conf, orig_engine = extract_plate_ocr(crop)
            orig_valid, orig_syn_score, orig_state, orig_sname, orig_match = validate_indian_plate_syntax(orig_clean)

            selected_source = "ORIGINAL"
            clean_text = orig_clean
            raw_ocr_text = orig_raw
            ocr_conf = orig_conf
            ocr_engine = orig_engine
            is_valid_syntax = orig_valid
            syn_score = orig_syn_score
            state_code = orig_state
            state_name = orig_sname
            match_type = orig_match
            rect_raw = ""
            rect_clean = ""

            if persp_applied and rectified_crop is not None and rectified_crop.size > 0:
                rect_clean, rect_raw, rect_conf, rect_engine = extract_plate_ocr(rectified_crop)
                rect_valid, rect_syn_score, rect_state, rect_sname, rect_match = validate_indian_plate_syntax(rect_clean)

                orig_composite = (1.0 if orig_valid else 0.0) * 0.40 + orig_conf * 0.40 + (orig_syn_score * 0.20)
                rect_composite = (1.0 if rect_valid else 0.0) * 0.40 + rect_conf * 0.40 + (rect_syn_score * 0.20)

                # Select rectified only if it produces valid syntax or higher composite evidence
                if (rect_valid and not orig_valid) or (rect_valid == orig_valid and rect_composite > orig_composite):
                    selected_source = "RECTIFIED"
                    clean_text = rect_clean
                    raw_ocr_text = rect_raw
                    ocr_conf = rect_conf
                    ocr_engine = f"{rect_engine} [Perspective {skew_deg:+.1f}°]"
                    is_valid_syntax = rect_valid
                    syn_score = rect_syn_score
                    state_code = rect_state
                    state_name = rect_sname
                    match_type = rect_match
                    if ANPR_CONFIG.get("save_debug_crops", True):
                        cv2.imwrite(crop_saved_path, rectified_crop)

            # POINT 14: Indian registration format validation result
            if is_valid_syntax:
                DEBUG_STATS["syntax_valid_count"] += 1

            overall_conf = float(round((cand["detectorConfidence"] * 0.4 + ocr_conf * 0.4 + syn_score * 0.2), 2))

            # POINT 15: Multi-frame confirmation status
            if is_valid_syntax:
                is_confirmed, final_plate, cons_conf, sightings = temporal_tracker.process_candidate(
                    req.cameraId,
                    clean_text,
                    overall_conf,
                    bypass_temporal=req.manualScan
                )
                conf_status = "CONFIRMED" if is_confirmed else "CANDIDATE_ACCUMULATING"
                if is_confirmed:
                    DEBUG_STATS["confirmed_count"] += 1
            else:
                is_confirmed = False
                final_plate = clean_text
                cons_conf = overall_conf
                sightings = 0
                conf_status = "REJECTED_SYNTAX"

            # POINT 16: Final ACCEPTED / REJECTED result
            if is_confirmed:
                final_decision = "ACCEPTED"
                rejection_reason = None
            elif not is_valid_syntax:
                final_decision = "REJECTED"
                rejection_reason = match_type
                record_rejection_stat(match_type)
            elif ocr_conf < ANPR_CONFIG["min_ocr_conf"]:
                final_decision = "REJECTED"
                rejection_reason = f"REJECTED_OCR_CONF: OCR confidence {ocr_conf:.2f} < {ANPR_CONFIG['min_ocr_conf']:.2f}"
                record_rejection_stat("REJECTED_OCR_CONF")
            else:
                final_decision = "CANDIDATE_ACCUMULATING"
                rejection_reason = f"Awaiting multi-frame confirmation ({sightings}/{ANPR_CONFIG['min_consecutive_sightings']} sightings)"

            eval_record = {
                "candidateIndex": c_idx,
                "plate": final_plate if is_valid_syntax else "",
                "rawOcr": raw_ocr_text,
                "normalizedOcr": clean_text,
                "ocrConfidence": float(round(ocr_conf, 2)),
                "ocrEngine": ocr_engine,
                "stateCode": state_code,
                "stateName": state_name,
                "syntaxMatch": match_type,
                "syntaxValid": is_valid_syntax,
                "detectorConfidence": float(round(cand["detectorConfidence"], 2)),
                "confidence": float(round(cons_conf, 2)),
                "sightings": sightings,
                "requiredSightings": ANPR_CONFIG["min_consecutive_sightings"],
                "confirmationStatus": conf_status,
                "finalDecision": final_decision,
                "rejectionReason": rejection_reason,
                "cropSavedPath": crop_saved_path,
                "cropUrl": crop_relative_url,
                "perspectiveCorrection": "APPLIED" if persp_applied else "SKIPPED",
                "perspectiveStatus": persp_status,
                "estimatedSkewDegrees": float(round(skew_deg, 2)),
                "perspectiveSelected": selected_source,
                "originalOcr": orig_raw,
                "rectifiedOcr": rect_raw if persp_applied else "N/A",
                "bbox": [int(x1), int(y1), int(bw), int(bh)],
                "cameraId": req.cameraId,
                "vehicleType": str(cand.get("vehicleClass", "car")),
                "plateType": "HSRP (High Security Registration Plate)",
                "status": "CONFIRMED" if is_confirmed else ("REJECTED_SYNTAX" if not is_valid_syntax else "CANDIDATE_ACCUMULATING")
            }

            all_evaluations.append(eval_record)
            if is_confirmed and confirmed_detection is None:
                confirmed_detection = eval_record

        # Construct comprehensive 16-point diagnostic payload for this frame
        primary_eval = all_evaluations[0] if all_evaluations else {}
        frame_telemetry = {
            "frameId": frame_id,
            "timestamp": now_ts,
            "cameraId": req.cameraId,
            "cameraName": req.cameraName,
            "frameSavedPath": frame_saved_path,
            "frameUrl": frame_relative_url,
            "resolution": f"{w_img}x{h_img}",
            "sharpnessScore": float(round(laplacian_var, 1)),
            "isBlurry": bool(is_blurry),
            "meanBrightness": float(round(mean_brightness, 1)),

            # The 16 requested diagnostic points:
            "1_vehicleDetected": bool(vehicle_detected),
            "2_vehicleClassAndConf": [{"class": str(v["class"]), "confidence": float(v["confidence"])} for v in detected_vehicles],
            "3_vehicleBoundingBoxes": [[int(x) for x in v["box"]] for v in detected_vehicles],
            "4_plateDetectionAttempted": True,
            "4_plateAttemptReason": str(attempt_reason),
            "5_plateCandidateCount": int(len(raw_candidates_evaluated)),
            "6_plateConfidences": [float(c["detectorConfidence"]) for c in raw_candidates_evaluated],
            "7_plateBoundingBoxesAndAR": [{"bbox": [int(x) for x in c["bbox"]], "size": str(c["size"]), "aspectRatio": float(c["aspectRatio"])} for c in raw_candidates_evaluated],
            "8_candidatesValidationPassed": [bool(c["passedValidation"]) for c in raw_candidates_evaluated],
            "9_candidateRejectionReasons": [str(c["rejectionReason"]) for c in raw_candidates_evaluated if c["rejectionReason"]],
            "10_plateCropsSaved": [str(e["cropUrl"]) for e in all_evaluations],
            "11_ocrRawResults": [str(e["rawOcr"]) for e in all_evaluations],
            "12_ocrConfidences": [float(e["ocrConfidence"]) for e in all_evaluations],
            "13_normalizedOcrResults": [str(e["normalizedOcr"]) for e in all_evaluations],
            "14_indianFormatValidationResults": [{"valid": bool(e["syntaxValid"]), "syntax": str(e["syntaxMatch"]), "state": str(e["stateCode"])} for e in all_evaluations],
            "15_multiFrameConfirmationStatus": str(primary_eval.get("confirmationStatus", "NONE")),
            "16_finalDecision": "ACCEPTED" if confirmed_detection else str(primary_eval.get("finalDecision", "REJECTED")),
            "perspectiveCorrection": str(primary_eval.get("perspectiveCorrection", "SKIPPED")),
            "estimatedSkewDegrees": float(primary_eval.get("estimatedSkewDegrees", 0.0)),
            "perspectiveSelected": str(primary_eval.get("perspectiveSelected", "NONE")),
            "originalOcr": str(primary_eval.get("originalOcr", "")),
            "rectifiedOcr": str(primary_eval.get("rectifiedOcr", "N/A")),
            "keyframeSharpness": float(round(laplacian_var, 1)),
            "clientKeyframeSharpness": req.keyframeSharpness,
            "keyframeCandidatesEvaluated": req.keyframeCandidates,
            "keyframeStatus": "BLURRY" if is_blurry else "ACCEPTED",

            "summary": {
                "status": "CONFIRMED_ANPR_EVENT" if confirmed_detection else primary_eval.get("status", "NO_DETECTION"),
                "reason": primary_eval.get("rejectionReason") if not confirmed_detection else f"Confirmed plate {confirmed_detection['plate']}",
                "failureStage": "None (Passed)" if confirmed_detection else (
                    "Stage 5: Multi-Frame Temporal Window" if primary_eval.get("confirmationStatus") == "CANDIDATE_ACCUMULATING" else "Stage 4: OCR / Syntax Validation"
                )
            },
            "candidates": raw_candidates_evaluated,
            "evaluations": all_evaluations
        }

        LATEST_DEBUG_PER_CAM[req.cameraId] = frame_telemetry
        RECENT_DEBUG_LOGS.append(frame_telemetry)
        if len(RECENT_DEBUG_LOGS) > 100:
            RECENT_DEBUG_LOGS.pop(0)

        # ──────────────────────────────────────────────────────────
        # POINT 16: Forward ONLY Confirmed Events to Node.js Backend
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
            "status": "CONFIRMED_ANPR_EVENT" if confirmed_detection else primary_eval.get("status", "CANDIDATE_TRACKING"),
            "detection": confirmed_detection,
            "vehicles": detected_vehicles,
            "detections": all_evaluations,
            "debug": frame_telemetry,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
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
