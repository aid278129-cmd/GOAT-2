# Multi-Camera ANPR Traffic Intelligence Platform 🚗🔍

> High-performance Automatic Number Plate Recognition (ANPR) and traffic monitoring system designed specifically for Indian vehicle registration plates using **YOLOv8** plate detection, **Tesseract OCR v5.4.0**, and real-time **WebRTC** camera streaming.

---

## 🌟 Key Features

- **🎯 Dedicated Indian Plate Detector (YOLOv8)**:
  - Custom-trained on real-world Indian license plate datasets across cars, motorcycles, auto-rickshaws, buses, and trucks.
  - Achieves **98.4% Box Precision** and **73.7% mAP@50**.
  - ONNX runtime export for client-side / web inference.

- **🔤 Tesseract OCR Engine Integration**:
  - Official **Tesseract v5.4.0** optical character recognition.
  - Multi-variant binarization: Bilateral Filtering, CLAHE, Dual-Mode Otsu (normal + inverted for commercial/EV plates), and Adaptive Gaussian thresholding.
  - Targeted Page Segmentation Modes (`--psm 7`, `--psm 8`, `--psm 6`) with alphanumeric whitelist.
  - Syntactic normalization for all 36 Indian States and Union Territories plus Bharat (`BH`) Series.

- **📹 Real-Time WebRTC Multi-Camera Streaming**:
  - Connects up to 4 simultaneous live camera feeds (Webcam + Mobile devices paired via dynamic QR codes).
  - Live in-canvas glowing neon bounding boxes, confidence tags, and HSRP badges.

- **📸 Interactive Car Plate Scanner**:
  - Upload car photos on-demand or snapshot any live camera stream.
  - Instant dual-pane inspection HUD with authentic embossed Indian HSRP license plate rendering.

- **📊 Vehicle Tracking & Analytics**:
  - Reconstructs vehicle routes and travel times between camera nodes.
  - Real-time watchlist hit notifications and traffic telemetry logging.

---

## 🏗️ Architecture

```
[ Camera Nodes / Phone Streams / Uploads ]
                    │
                    ▼
[ WebRTC / Socket.IO Node.js Server (Port 3000) ]
                    │
                    ▼
[ Python ANPR Inference Server (Port 5001) ]
         ┌──────────┴──────────┐
         ▼                     ▼
[ YOLOv8 Plate Detector ]   [ Tesseract OCR v5.4.0 ]
         └──────────┬──────────┘
                    ▼
   [ Normalization & State Parsing ]
                    │
                    ▼
 [ Dashboard HUD Overlay & detections.json ]
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Node.js**: v18+
- **Python**: v3.10+
- **Tesseract OCR v5+**: Installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (or system PATH)

### 2. Install Dependencies

```bash
# Node dependencies
npm install

# Python dependencies
pip install -r requirements.txt
```

### 3. Launch Services

**Terminal 1 - Web Dashboard:**
```bash
node server.js
```
*Access dashboard at `https://localhost:3000/dashboard.html`*

**Terminal 2 - ANPR Inference Server:**
```bash
python scripts/anpr_server.py
```
*Runs on `http://127.0.0.1:5001`*

---

## 📁 Repository Structure

```
├── Annotations/                  # Ground truth Pascal VOC annotations
├── Indian_Number_Plates/         # Sample Indian vehicle dataset
├── data/                         # Persistent JSON logs (detections, watchlist, alerts)
├── docs/                         # Technical documentation & training roadmaps
│   └── TESSERACT_TRAINING_ROADMAP.md
├── models/                       # Trained PyTorch & ONNX weights
│   ├── indian_plate_best.pt
│   └── tesseract_benchmark_results.json
├── public/                       # Frontend web assets, UI dashboard, and styles
│   ├── dashboard.html
│   ├── camera.html
│   ├── js/anpr-engine.js
│   └── style-dashboard.css
├── scripts/                      # Training, inference, and benchmarking scripts
│   ├── anpr_server.py
│   ├── evaluate_tesseract_ocr.py
│   ├── prepare_dataset.py
│   ├── train_plate_model.py
│   └── verify_live_detection.py
└── server.js                     # Express & Socket.IO WebRTC streaming server
```

---

## 📄 Documentation

- [Tesseract OCR Training Roadmap](docs/TESSERACT_TRAINING_ROADMAP.md): Detailed blueprint for training custom LSTM weights using `text2image` and `lstmtraining`.
- [Dataset Evaluation Results](models/tesseract_benchmark_results.json): Empirical benchmark accuracy across ground-truth annotated plates.

---

## 📜 License
MIT License.
