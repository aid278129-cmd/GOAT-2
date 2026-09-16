# ANPR REBUILD PLAN: Modular ANPR V2 Architecture
**Project:** City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban Traffic Analytics  
**SIH Problem Statement ID:** 26127 (Bharat Electronics Limited - BEL)  
**Document:** ANPR Pipeline Rebuild & Modernization Plan  
**Target:** Replace unreliable ANPR/OCR recognition with a clean, modern, maintainable implementation.

---

## 1. Executive Summary & Core Mandate

The existing project has a functioning full-stack architecture:
- **WebRTC multi-camera ingestion & streaming** (`server.js`, `public/camera.html`, `public/mobile.html`)
- **Intelligent keyframe selection & canvas dispatch** (`public/js/anpr-engine.js`)
- **Multi-Camera Concurrency Queue Manager** (`server.js`, maxConcurrency: 2, stale frame dropping)
- **GIS interactive Leaflet map & trajectory stitching** (`public/dashboard.html`, `/api/detections/trajectory/:plate`)
- **Watchlist stolen/wanted vehicle alerts** (`data/watchlist.json`, `/api/alerts`)
- **Urban traffic analytics & density metrics** (`/api/detections/analytics`)
- **Single-command launcher** (`run.py`, `start_system.bat`, `start_system.ps1`)

**CRITICAL DIRECTIVE:**
We will **NOT** rebuild the application, redesign the frontend, or touch working WebRTC, GIS, trajectory tracking, watchlist, or analytics code.
We will **REPLACE ONLY** the unreliable ANPR/OCR recognition pipeline with a clean, modular **ANPR V2** engine.

```
CAMERA / IMAGE
      ↓
LICENSE PLATE DETECTION (Direct Full-Frame, RF-DETR Small / YOLO baseline)
      ↓
BEST PLATE CROP
      ↓
PERSPECTIVE RECTIFICATION (Minimal & Non-Destructive)
      ↓
PP-OCR RECOGNITION (PaddleOCR PP-OCRv4 via ONNX Runtime)
      ↓
MULTI-FRAME OCR CONSENSUS (Character-Level Confidence-Weighted Voting)
      ↓
INDIAN REGISTRATION FORMAT VALIDATION (Pure Validator: VALID/INVALID/UNCERTAIN; NO Text Rewriting)
      ↓
CONFIRMED EXACT PLATE
      ↓
EXISTING TRAJECTORY / GIS / WATCHLIST / ANALYTICS
```

---

## 2. Current Pipeline vs. ANPR V2 Pipeline

| Component | Current Pipeline (V1) | ANPR V2 Pipeline |
| :--- | :--- | :--- |
| **Detection Strategy** | Two-stage: Vehicle-first (COCO YOLO) $\rightarrow$ Plate inside vehicle ROI. Bypassed or fails when camera is close to plate or vehicle body is cropped. | **Direct Full-Frame Detection**: Locates license plates directly in the complete camera frame without requiring vehicle cabin/wheels. Supports cars, bikes, buses, trucks, autos. |
| **Detector Model** | YOLOv8n (`models/indian_plate_best.pt` / `.onnx`) | **RF-DETR Small** evaluated and benchmarked against existing **YOLO plate detector** as baseline. Selected only if benchmark justifies it. |
| **Crop Preprocessing** | Heavy pipeline: LAB CLAHE, bilateral anti-moiré, adaptive unsharp masking, dual-Otsu binarization, HSRP color masking. Can distort stroke topology. | **Minimal & Non-Destructive**: Crisp crop, optional 4-point perspective rectification, resolution normalization. Crop visually resembles the original plate. |
| **OCR Engine** | CRNN ONNX (`models/crnn_plate_best.onnx`) $\rightarrow$ fallback to Tesseract v5 / EasyOCR. Fallback caused 2+ second latency spikes. | **PaddleOCR PP-OCR Recognition** (PP-OCRv4 via ONNX Runtime). Single high-accuracy engine (5-15 ms on CPU), eliminating slow fallback chains. |
| **Character Disambiguation** | **Aggressive Regex Rewriting**: Mutated characters (`8` $\rightarrow$ `B`, `0` $\rightarrow$ `O`, `1` $\rightarrow$ `I`, `MG` $\rightarrow$ `MH`) to force matches to Indian regexes. Produced hallucinations (e.g. `TN046978` from invalid crops). | **REMOVED AGGRESSIVE REWRITING**: Format rules act purely as a **VALIDATOR**, never an OCR text generator. If uncertain, system requests another frame. |
| **Temporal Tracking** | Frame-level string matching in sliding window. | **Character-Level Confidence-Weighted Temporal Voting**: Tracks plate across frames; votes per character position weighted by visual confidence. |
| **Best Frame Selection** | Client-side only (Laplacian sharpness). | **Candidate Crop Scoring**: Scores crops by detector confidence, sharpness, plate size, and viewing angle; OCRs only the highest-quality candidates. |
| **Validation Output** | Regex match string modification. | Strictly returns enum: `VALID`, `INVALID`, `UNCERTAIN`. Does not rewrite characters. |

---

## 3. Component Categorization

### A. Files to KEEP (Unchanged / Reused)
- `server.js`: Node.js server, HTTPS self-signed certificates, WebSocket/Socket.io, WebRTC signaling, `MultiCameraQueueManager`, REST APIs, GIS, watchlist, analytics.
- `public/dashboard.html`: Command center, vehicle tracking, analytics charts, camera network, watchlist, Leaflet map, HUD overlay.
- `public/camera.html` & `public/mobile.html`: WebRTC camera nodes.
- `public/js/anpr-engine.js`: Keyframe sharpness filter, canvas scaling, REST dispatch to `/api/anpr/detect`.
- `public/css/style.css`: Dashboard styling.
- `data/detections.json`, `data/watchlist.json`, `data/alerts.json`: Storage files.
- `dataset/`: Existing annotated Indian plates dataset (`images/train`, `images/val`, `labels/`).
- `Indian_Number_Plates/Sample_Images/` & `Annotations/`: Real challenge vehicle images.
- `number_plate_images_ocr/` & `number_plate_annos_ocr/`: Real vehicle images with VOC ground-truth annotations.
- `models/indian_plate_best.pt` & `models/indian_plate_best.onnx`: Kept as benchmark baseline detector.

### B. Files to REFACTOR / ADAPT
- `scripts/anpr_server.py`: Provide backward-compatible FastAPI adapter routing to `anpr_v2.pipeline` while preserving endpoints `/detect`, `/detect-file`, `/health`, `/config`, and debug routes.
- `run.py`: Startup launcher updated to load ANPR V2 by default.
- `requirements.txt`: Pin clean, compatible dependencies (`onnxruntime`, `rapidocr_onnxruntime` or PP-OCR ONNX, `rfdetr`, `torch`, `ultralytics`).

### C. Files to DEPRECATE (Replaced by V2 modules)
- `scripts/crnn_model.py`: Deprecated in primary inference path.
- `scripts/crnn_ocr.py`: Deprecated in primary inference path (its aggressive character-swapping functions are removed).
- `scripts/evaluate_tesseract_ocr.py`: Deprecated.

---

## 4. Proposed ANPR V2 Modular File Structure

```
d:\college work\Hackaton projects\new zyn\
├── anpr_v2/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Centralized settings (ANPR_VERSION, DETECTOR, OCR, DEVICE, thresholds)
│   ├── detector.py          # Direct Plate Detector (RF-DETR Small + YOLO baseline adapter)
│   ├── rectifier.py         # 4-Point homography perspective rectification & crop extraction
│   ├── recognizer.py        # PaddleOCR PP-OCRv4 recognition engine (ONNX Runtime)
│   ├── tracker.py           # Multi-frame character-level confidence-weighted temporal voting
│   ├── validator.py         # Pure Indian registration validator (VALID / INVALID / UNCERTAIN)
│   ├── pipeline.py          # Unified ANPR V2 pipeline orchestrator
│   └── server.py            # FastAPI service adapter (Port 5001) for Node.js backend
├── training/
│   ├── detector/            # RF-DETR fine-tuning scripts and COCO/YOLO converter
│   └── ocr/                 # PP-OCR evaluation & fine-tuning scripts
├── benchmarks/
│   ├── anpr_v2/             # Automated benchmark runner (detection + OCR exact match)
│   └── regression_test.py   # Critical regression test (MH01AV8669 vs TN046978)
├── datasets/
│   └── README.md            # Dataset sources, licenses, 80/10/10 split documentation
├── models/
│   ├── rfdetr_plate_best.*  # RF-DETR plate detector weights
│   ├── ppocr_rec_v4.onnx    # PP-OCRv4 recognition ONNX model
│   ├── ppocr_keys_v1.txt    # PP-OCR character dictionary
│   ├── indian_plate_best.*  # Baseline YOLO plate detector
│   └── crnn_plate_best.*    # Baseline CRNN model
├── ANPR_REBUILD_PLAN.md     # This document
├── DATASET_ANALYSIS.md      # Dataset distribution analysis (Phase 4)
└── ANPR_V2_REPORT.md        # Final benchmark report (Phase 22)
```

---

## 5. Model & Runtime Specifications

### A. License Plate Detector
- **Primary:** RF-DETR Small fine-tuned for `license_plate`. Full camera frame input $\rightarrow$ bounding box `[x1, y1, x2, y2]` with confidence.
- **Baseline:** YOLOv8 Indian Plate Detector (`models/indian_plate_best.pt` / `.onnx`).
- **Benchmark:** Compare on identical held-out test split: Precision, Recall, mAP50, mAP50-95, False Positives, CPU latency, GPU latency. Selected only if supported by empirical benchmark.

### B. License Plate OCR Recognizer
- **Primary:** PaddleOCR PP-OCRv4 recognition model (`en_PP-OCRv4_rec` / `ch_PP-OCRv4_rec`) executed via **ONNX Runtime**.
  - **Why ONNX Runtime?** Python 3.14 on Windows has no prebuilt `paddlepaddle` wheels, but `onnxruntime 1.30.0` is already installed and runs PP-OCR ONNX models in 5–15 ms on CPU with zero compilation issues and high cross-platform stability.
- **Dictionary:** Character vocabulary mapping including `0-9`, `A-Z`, and special tokens.
- **Baseline:** Existing CRNN (`models/crnn_plate_best.onnx`) and Tesseract/EasyOCR kept strictly for comparative benchmarking.

---

## 6. Dataset Strategy & Leakage Prevention

1. **Source Datasets:**
   - Pascal VOC annotated Indian vehicle dataset (`number_plate_images_ocr/` + `number_plate_annos_ocr/`)
   - DataCluster Indian Number Plates (`dataset/`, `Indian_Number_Plates/`)
2. **Unified Dataset Splits:**
   - **Train:** 80%
   - **Validation:** 10%
   - **Test (Held-out Benchmark):** 10%
3. **Leakage Rule:**
   - Images from the same vehicle or continuous video capture must reside in the **same split**.
   - Held-out test images are strictly **never** used during training.
   - The critical regression sample `MH01AV8669` is reserved strictly as a test case.

---

## 7. Implementation Sequence (Step-by-Step)

```
[x] Step 1: Repository Inspection & Architectural Rebuild Plan (ANPR_REBUILD_PLAN.md)
[ ] Step 2: Dataset Preparation & Distribution Audit (datasets/README.md, DATASET_ANALYSIS.md)
[ ] Step 3: Clean Modular ANPR V2 Scaffolding (config.py, validator.py, rectifier.py)
[ ] Step 4: PP-OCR Recognition Engine with ONNX Runtime (recognizer.py)
[ ] Step 5: Character-Level Temporal Consensus Tracker & Best Frame Selection (tracker.py)
[ ] Step 6: License Plate Detector Adapter (RF-DETR Small & YOLO baseline) (detector.py)
[ ] Step 7: Unified Pipeline Orchestrator (pipeline.py) & Regression Testing (MH01AV8669)
[ ] Step 8: Comprehensive Benchmarking Suite & A/B Comparison (benchmarks/anpr_v2/)
[ ] Step 9: Production Integration with server.js & run.py
[ ] Step 10: Automated Unit & Integration Tests (scripts/test_anpr_v2.py)
[ ] Step 11: Final Report & Documentation (ANPR_V2_REPORT.md)
```

---

## 8. Critical Regression Verification

- **Target Plate:** `MH01AV8669`
- **Expected Output:** `MH01AV8669` (Exact Match)
- **Catastrophic Failure Definition:** Levenshtein distance $\ge 3$, or wrong state prefix (`TN...`) combined with character errors.
- **Rule:** The system must reject ambiguous inputs rather than hallucinating an invalid or incorrect plate registration.
