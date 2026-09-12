"""
anpr_server.py
High-Performance Local ANPR Inference Service for Indian License Plates
Runs YOLOv8 plate detector + optimized OCR on live video frames.
"""

import os
import io
import re
import base64
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

app = FastAPI(title="Indian ANPR Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "indian_plate_best.pt")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "runs", "indian_plate_run", "weights", "best.pt")
NODE_SERVER_URL = "https://127.0.0.1:3000/api/detections"

# Set Tesseract binary path
TESSERACT_DEFAULT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_DEFAULT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_DEFAULT_PATH
    print(f"Configured Tesseract OCR binary: {TESSERACT_DEFAULT_PATH}")
else:
    print("Using system default tesseract command")

yolo_model = None
easyocr_reader = None

def get_easyocr_fallback():
    global easyocr_reader
    if easyocr_reader is None:
        print("Initializing EasyOCR fallback reader...")
        easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return easyocr_reader

def get_model():
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    from ultralytics import YOLO
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained weights from {MODEL_PATH}")
        yolo_model = YOLO(MODEL_PATH)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        print(f"Loading trained weights from fallback {FALLBACK_MODEL_PATH}")
        yolo_model = YOLO(FALLBACK_MODEL_PATH)
    else:
        print("Warning: Trained weights not found yet, initializing pretrained yolov8n.pt")
        yolo_model = YOLO("yolov8n.pt")
    return yolo_model

INDIAN_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK',
    'UA', 'UP', 'WB', 'BH'
}

INDIAN_STATE_NAMES = {
    'AN': 'Andaman and Nicobar',
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CG': 'Chhattisgarh',
    'CH': 'Chandigarh',
    'DD': 'Daman and Diu',
    'DL': 'Delhi',
    'DN': 'Dadra and Nagar Haveli',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HP': 'Himachal Pradesh',
    'HR': 'Haryana',
    'JH': 'Jharkhand',
    'JK': 'Jammu and Kashmir',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'LA': 'Ladakh',
    'LD': 'Lakshadweep',
    'MH': 'Maharashtra',
    'ML': 'Meghalaya',
    'MN': 'Manipur',
    'MP': 'Madhya Pradesh',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OD': 'Odisha',
    'OR': 'Odisha',
    'PB': 'Punjab',
    'PY': 'Puducherry',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TR': 'Tripura',
    'TS': 'Telangana',
    'UK': 'Uttarakhand',
    'UA': 'Uttarakhand',
    'UP': 'Uttar Pradesh',
    'WB': 'West Bengal',
    'BH': 'Bharat Series (All-India)'
}

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

    # Standard Otsu
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Inverted Otsu (for light text on dark plate)
    otsu_inv = cv2.bitwise_not(otsu)
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return [('otsu', otsu), ('otsu_inv', otsu_inv), ('adaptive', adaptive), ('clahe', clahe)]

def clean_plate_text(raw_text):
    """Normalize and repair Indian vehicle registration text."""
    if not raw_text:
        return ""

    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(cleaned) < 4:
        return cleaned

    digit_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'}
    char_to_digit = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'G': '6'}

    chars = list(cleaned)
    for i in [0, 1]:
        if i < len(chars) and chars[i] in digit_to_char:
            chars[i] = digit_to_char[chars[i]]

    bh_match = re.match(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$", "".join(chars))
    if bh_match:
        return "".join(chars)

    state = "".join(chars[:2])
    if state in INDIAN_STATES and len(chars) >= 7:
        for i in [2, 3]:
            if i < len(chars) and chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]

        num_start = max(4, len(chars) - 4)
        for i in range(num_start, len(chars)):
            if chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]

    return "".join(chars)

def extract_plate_ocr(crop_bgr):
    """Recognize Indian vehicle registration plate text using Tesseract OCR (with fallback)."""
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

                if len(cleaned) >= 6 and cleaned[:2] in INDIAN_STATES:
                    score += 0.40
                if 8 <= len(cleaned) <= 10:
                    score += 0.25

                if score > best_conf and len(cleaned) >= 4:
                    best_conf = score
                    best_text = cleaned
                    best_engine = f"Tesseract OCR (PSM {psm}, {vname})"
                    # Early termination if valid Indian plate discovered
                    if score >= 0.80 and len(cleaned) >= 7 and cleaned[:2] in INDIAN_STATES:
                        return best_text, min(0.99, best_conf), best_engine
            except Exception:
                continue

    # 2. Fallback to EasyOCR if Tesseract confidence is low or length < 4
    if len(best_text) < 5 or best_conf < 0.45:
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
                    if len(cleaned) >= 6 and cleaned[:2] in INDIAN_STATES:
                        score += 0.35
                    if score > best_conf and len(cleaned) >= 4:
                        best_conf = score
                        best_text = cleaned
                        best_engine = "EasyOCR (Fallback)"
                        break
        except Exception:
            pass

    return best_text, min(0.99, best_conf), best_engine

class Base64DetectRequest(BaseModel):
    image: str
    cameraId: int = 1
    cameraName: str = "Camera 1"
    forwardToDashboard: bool = True

@app.get("/health")
def health():
    m = get_model()
    return {"status": "ok", "model_ready": m is not None}

@app.post("/detect")
async def detect_plate(req: Base64DetectRequest):
    try:
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        model = get_model()
        results = model.predict(img, conf=0.25, imgsz=640, verbose=False)

        best_detection = None
        all_detections = []
        highest_score = 0.0

        detected_boxes = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                detected_boxes.append((conf, xyxy))

        # Sort by box confidence descending and inspect top 2 candidates
        detected_boxes.sort(key=lambda x: x[0], reverse=True)
        top_candidates = detected_boxes[:2]

        h_img, w_img = img.shape[:2]
        for conf, (x1, y1, x2, y2) in top_candidates:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            bw, bh = x2 - x1, y2 - y1

            if bw < 20 or bh < 8:
                continue

            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w_img, x2 + pad_x)
            cy2 = min(h_img, y2 + pad_y)

            crop = img[cy1:cy2, cx1:cx2]
            clean_text, ocr_conf, ocr_engine = extract_plate_ocr(crop)

            # Determine State
            state_code = clean_text[:2] if len(clean_text) >= 2 else ""
            state_name = INDIAN_STATE_NAMES.get(state_code, "Indian Registered Vehicle")

            display_plate = clean_text if len(clean_text) >= 4 else "IND PLATE DETECTED"
            overall_conf = float(round((conf * 0.4 + max(ocr_conf, 0.4) * 0.6), 2))

            det_item = {
                "plate": display_plate,
                "stateCode": state_code,
                "stateName": state_name,
                "confidence": overall_conf,
                "detectorConfidence": float(round(conf, 2)),
                "ocrConfidence": float(round(ocr_conf, 2)),
                "ocrEngine": ocr_engine,
                "bbox": [int(x1), int(y1), int(bw), int(bh)],
                "cameraId": req.cameraId,
                "vehicleType": "Car / Passenger Vehicle",
                "plateType": "HSRP (High Security Registration Plate)",
            }
            all_detections.append(det_item)

            item_score = conf + (1.0 if len(clean_text) >= 6 else 0.3)
            if item_score > highest_score:
                highest_score = item_score
                best_detection = det_item

        if best_detection and req.forwardToDashboard:
            try:
                requests.post(
                    NODE_SERVER_URL,
                    json={
                        "plate": best_detection["plate"],
                        "cameraId": req.cameraId,
                        "confidence": best_detection["confidence"],
                        "vehicleType": best_detection["vehicleType"],
                        "simulated": False,
                    },
                    timeout=1.5,
                    verify=False,
                )
            except Exception as e:
                print(f"Forward to dashboard failed: {e}")

        return {
            "success": True,
            "detected": best_detection is not None,
            "detection": best_detection,
            "detections": all_detections,
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
            forwardToDashboard=forwardToDashboard
        )
        return await detect_plate(req)
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("Starting ANPR Inference Server on http://127.0.0.1:5001...")
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="info")
