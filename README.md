# 🚗 Multi-Camera Indian ANPR Traffic Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/SIH%202024-Problem%20ID%2026127-blue?style=for-the-badge&logo=target" alt="SIH Problem ID 26127" />
  <img src="https://img.shields.io/badge/Organization-Bharat%20Electronics%20Limited%20(BEL)-navy?style=for-the-badge" alt="BEL" />
  <img src="https://img.shields.io/badge/YOLOv8-Dedicated%20Plate%20Detector-orange?style=for-the-badge&logo=yolo" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/OCR-Tesseract%20v5.4.0%20%2B%20EasyOCR-green?style=for-the-badge" alt="Tesseract OCR" />
  <img src="https://img.shields.io/badge/Backend-FastAPI%20%7C%20Node.js%20WebRTC-informational?style=for-the-badge&logo=fastapi" alt="FastAPI + Node.js" />
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="MIT License" />
</p>

> **High-Performance City-Wide AI Engine for Multi-Camera Automatic Number Plate Recognition (ANPR), Spatio-Temporal Trajectory Reconstruction, and Urban Traffic Intelligence.**  
> *Developed for Smart India Hackathon — Problem Statement ID: 26127 (Bharat Electronics Limited - BEL).*

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features & Innovations](#-key-features--innovations)
- [System Architecture](#-system-architecture)
- [Empirical Benchmarks](#-empirical-benchmarks)
- [Indian Plate Syntactic Engine](#-indian-plate-syntactic-engine)
- [Quick Start](#-quick-start)
  - [One-Click Startup](#1-one-click-startup-recommended)
  - [Manual Setup](#2-manual-setup)
- [Microservice API Reference](#-microservice-api-reference)
- [Repository Structure](#-repository-structure)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [License](#-license)

---

## 🌟 Overview

The **Multi-Camera Indian ANPR Intelligence Platform** is an enterprise-grade surveillance system specifically engineered for the complexities of Indian vehicle registration plates. Real-world Indian roadways present extreme visual variance: diverse vehicle classes (two-wheelers, auto-rickshaws, commercial trucks, buses, private cars), multiple plate color schemes (private white, commercial yellow, EV green, rental black), varied typography, and two-row high-security registration plates (HSRP).

This platform combines a **two-stage YOLOv8 deep learning vision pipeline**, an **optimized dual-engine OCR subsystem** (Tesseract v5.4.0 with multi-variant adaptive thresholding + EasyOCR fallback), and a **6-stage false-positive evidence chain gating system** connected to a **persistent Single-Page Application (SPA) Command Center** via low-latency WebRTC streams.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KEY SYSTEM CAPABILITIES                           │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ 97.1% Precision      │ 0.0% False Positives │ 4-Stream WebRTC Ingestion     │
│ Across Indian plates │ Strict negative gate │ Instant Mobile QR Pairing     │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ Dual-Engine OCR      │ 36 States + BH Series│ Spatio-Temporal Trajectories  │
│ Tesseract + EasyOCR  │ Syntactic validation │ Multi-camera vehicle tracking │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

---

## 🚀 Key Features & Innovations

### 1. 🎯 Dedicated Indian Plate Detector (YOLOv8 Nano)
- Custom-trained on real-world Indian roadway datasets encompassing cars, motorcycles, auto-rickshaws, tempo vans, buses, and heavy trucks.
- Detects tight plate bounding boxes even under acute perspective angles, severe shadow, vibration, and low illumination.
- Quantized and exported to **ONNX runtime** for high-throughput edge and client-side web inference.

### 2. 🔤 Hardened Dual-Engine OCR Pipeline
- **Primary Engine (Tesseract OCR v5.4.0)**:
  - Multi-variant image binarization: **CLAHE** (Contrast Limited Adaptive Histogram Equalization), **Bilateral Filtering**, and **Dual-Mode Otsu Thresholding** (handling both dark-on-light and light-on-dark EV/commercial plates).
  - Constrained Character Whitelists: Restricts OCR dictionary to valid uppercase alphanumeric symbols (`ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`).
  - Targeted Page Segmentation Modes (`--psm 7` single-line, `--psm 8` single-word, `--psm 6` multi-row block).
- **Secondary Fallback (EasyOCR)**:
  - Automatically triggered if Tesseract confidence drops below threshold or character count is deficient.
  - Excels on slanted and embossed high-security registration plates (HSRP).

### 3. 🛡️ 6-Stage False-Positive Evidence Chain Gating
Rejects non-vehicle visual clutter (street signs, bumper stickers, road text, headlights, billboards) before plate recognition is accepted:
1. **Stage 1 (Vehicle Presence Gate)**: Verifies presence of a recognized vehicle bounding box (`car`, `motorcycle`, `bus`, `truck`) via COCO YOLOv8.
2. **Stage 2 (Plate Localization Gate)**: Plate candidate must reside spatially within the vehicle ROI.
3. **Stage 3 (Geometric & Aspect Ratio Gate)**: Rejects boxes failing aspect ratio ($1.5 \le AR \le 6.5$) and relative area constraints ($0.5\% \le \text{Area} \le 35\%$).
4. **Stage 4 (Syntax & State Code Gate)**: Validates OCR against 36 Indian State/UT prefixes or the Bharat (`BH`) Series.
5. **Stage 5 (Multi-Frame Temporal Window)**: Requires spatial and textual consistency across $\ge 2$ consecutive frames to eliminate transient camera noise.
6. **Stage 6 (Event Confirmation Gate)**: Persists confirmed event, updates vehicle trajectory, and checks live watchlist.

### 4. 📹 Persistent WebRTC Multi-Camera Ingestion & SPA Dashboard
- **Zero-Reload SPA Navigation**: Switch seamlessly between **Command Center**, **Vehicle Tracking**, **Analytics**, **Camera Network**, and **Watchlist** via client-side hash routing (`#cmd`, `#track`, `#analytics`, `#network`, `#watchlist`) without tearing down active WebRTC streams.
- **Dynamic QR Pairing**: Instantly connect any mobile smartphone camera as an active surveillance node by scanning an on-screen QR code.
- **Background Keepalive**: Extended WebSocket ping timeouts (60s) prevent stream disconnects when browser tabs are placed in the background.

### 5. 🗺️ Spatio-Temporal Trajectory Tracking & Watchlist (BOLO)
- Connects detections across distributed camera nodes (`CAM-01`, `CAM-02`, etc.) to reconstruct origin-to-destination vehicle routes.
- Computes transit duration, average speed between camera nodes, and direction of travel.
- Real-time automated BOLO (Be On the Lookout) security alerts with audible HUD sirens and visual notifications upon watchlist plate hits.

### 6. 🔍 16-Point Per-Frame Diagnostic Telemetry HUD
Live diagnostic breakdown displayed directly in the dashboard and accessible via REST API:
- Vehicle Detected (`true`/`false`) & Class/Confidence
- Vehicle Bounding Box Coordinates
- Plate Detection Candidate Count & Bounding Box Coordinates
- Candidate Aspect Ratio & Size Filtering Status
- Specific Stage Rejection Reasons
- Raw vs. Normalized OCR Text & Confidence
- Indian State Code Identification
- Multi-Frame Confirmation Sighting Counter

---

## 🏗️ System Architecture

```
                                  [ SURVEILLANCE EDGE ]
               ┌───────────────────────────┬───────────────────────────┐
               │                           │                           │
        [ Mobile Phone ]             [ USB / Webcam ]          [ Static Image ]
     (WebRTC Camera Stream)       (Local Video Stream)         (Manual Upload)
               │                           │                           │
               └───────────────────────────┼───────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  Node.js Ingestion & WebRTC Signaling Tier   │
                    │         (Express + Socket.io :3000)          │
                    │  • Camera Node Pairing & QR Authentication   │
                    │  • Frame Ingestion & Trajectory Aggregator   │
                    │  • Static Asset & SPA Dashboard Delivery     │
                    └──────────────────────┬───────────────────────┘
                                           │  HTTP POST /detect (Frames)
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │     Python ANPR Microservice (Port 5001)     │
                    │               (FastAPI + ASGI)               │
                    ├──────────────────────────────────────────────┤
                    │  [1] Stage 1: Vehicle Detection (YOLOv8n)    │
                    │  [2] Stage 2: Vehicle ROI Plate Localization │
                    │  [3] Stage 3: Geometric & Aspect Filtering   │
                    │  [4] Stage 4: CLAHE / Dual-Otsu Binarization │
                    │  [5] Stage 5: Tesseract v5.4.0 + EasyOCR     │
                    │  [6] Stage 6: MoRTH Syntax Disambiguation    │
                    │  [7] Stage 7: Multi-Frame Temporal Window    │
                    └──────────────────────┬───────────────────────┘
                                           │  Confirmed Detection + 16 Telemetry Points
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │     Command Center SPA Dashboard (HTTPS)     │
                    │  • Neon Glowing Bounding Boxes & HUD Badges  │
                    │  • Real-Time Trajectory GIS Map (Leaflet)    │
                    │  • Watchlist Security Alerts & Sound Sirens  │
                    │  • 16-Point Real-Time Telemetry Inspector    │
                    └──────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks

The hardened pipeline was benchmarked using the official evaluation suite (`scripts/benchmark_anpr.py`) across ground-truth annotated Indian vehicles and challenging negative controls:

| Metric | Measured Value | Standard Target | Status |
| :--- | :---: | :---: | :---: |
| **Plate Detection Precision** | **97.14%** | $\ge 90\%$ | 🟢 Exceeds Target |
| **Plate Detection Recall** | **97.14%** | $\ge 90\%$ | 🟢 Exceeds Target |
| **False Positive Rate (FPR)** | **0.00%** | $\le 2\%$ | 🟢 Zero False Positives |
| **Auto-Rickshaw Detection Rate** | **100.0%** | $\ge 85\%$ | 🟢 Perfect Capture |
| **Car Detection Rate** | **100.0%** | $\ge 90\%$ | 🟢 Perfect Capture |
| **Commercial Truck & Van Detection**| **100.0%** | $\ge 85\%$ | 🟢 Perfect Capture |
| **Negative Control Rejection** | **100.0%** | $\ge 95\%$ | 🟢 Total Rejection |

*Benchmark artifacts persisted in [`benchmark_results.json`](benchmark_results.json) and [`benchmark_results.csv`](benchmark_results.csv).*

---

## 🇮🇳 Indian Plate Syntactic Engine

The normalization engine validates and disambiguates characters against the Ministry of Road Transport and Highways (MoRTH) standards across all **36 States & Union Territories**:

### Supported Formats

| Plate Type | Format Pattern | Example | Description |
| :--- | :--- | :--- | :--- |
| **Standard Private** | `SS DD AA NNNN` | `MH12DE1433` | 2-letter State, 2-digit District, 1-3 Letters, 4 Digits |
| **Bharat Series (BH)**| `YY BH NNNN AA` | `22BH1234AA` | Registration Year, `BH` code, 4 Digits, 2 Series Letters |
| **Commercial / Taxi** | `SS DD T NNNN` | `DL1TA1234` | Yellow plate with black text |
| **Electric Vehicle** | `SS DD EV NNNN` | `KA01EV5678` | Green plate with white text (handled via Inverted Otsu) |
| **Two-Row High-Security** | `SS DD \n AA NNNN` | `DL1C \n AA1234`| Segmented two-tier motorcycle and auto-rickshaw plates |

### Context-Aware Positional Disambiguation
Optical noise frequently swaps visually identical glyphs. The syntactic parser resolves these based on positional expectations:
- **State Code & Series (Alphabetic Positions)**: Converts `0` $\to$ `O`, `1` $\to$ `I`, `8` $\to$ `B`, `5` $\to$ `S`, `2` $\to$ `Z`.
- **District & Registration Number (Numeric Positions)**: Converts `O`/`D` $\to$ `0`, `I`/`L` $\to$ `1`, `B` $\to$ `8`, `S` $\to$ `5`, `Z` $\to$ `2`.

---

## 🚀 Quick Start

### Prerequisites
- **Python**: v3.10 or higher
- **Node.js**: v18 or higher
- **Tesseract OCR v5+**:
  - *Windows*: Download and install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
  - *Linux*: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`

---

### 1. One-Click Startup (Recommended)

#### Unified Python Launcher (Cross-Platform — Windows / Linux / macOS):
```bash
python run.py
```
*Runs pre-flight checks, handles port allocation, launches both Python ANPR backend (`scripts/anpr_server.py`) and Node.js frontend (`server.js`) with live log streaming, and automatically opens the dashboard in your default browser.*

##### Useful Launcher Flags:
- `python run.py --no-browser` : Launch services without opening browser
- `python run.py --separate-windows` : Open services in separate consoles (Windows)
- `python run.py --kill-existing` : Clear and restart existing port listeners (5001 & 3000)
- `python run.py --install-deps` : Check and auto-install missing packages
- `python run.py --backend-only` : Start only Python ANPR inference server
- `python run.py --frontend-only` : Start only Node.js dashboard server

#### Windows Command Prompt / Double-Click:
```cmd
start_system.bat
```

#### Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File start_system.ps1
```

Once started, the script will automatically open the dashboard in your default browser at `https://127.0.0.1:3000/dashboard.html`.

---

### 2. Manual Setup

If running in customized environments or headless servers:

#### Step 1: Install Dependencies
```bash
# Install Node.js dependencies
npm install

# Install Python vision & inference dependencies
pip install -r requirements.txt
```

#### Step 2: Start Python ANPR Microservice (Terminal 1)
```bash
python scripts/anpr_server.py
```
*Inference microservice starts at `http://127.0.0.1:5001`.*

#### Step 3: Start Node.js WebRTC Streaming Server (Terminal 2)
```bash
node server.js
```
*Web dashboard starts at `https://127.0.0.1:3000`.*

> [!NOTE]
> When accessing `https://127.0.0.1:3000` for the first time, your browser may display a self-signed HTTPS warning. Click **"Advanced" $\to$ "Proceed"**. HTTPS is strictly required by modern mobile browsers to grant camera and WebRTC permissions.

---

## 🔌 Microservice API Reference

### Python ANPR Microservice (`http://127.0.0.1:5001`)

| Method | Endpoint | Description | Payload / Parameters |
| :--- | :--- | :--- | :--- |
| `POST` | `/detect` | Full ANPR pipeline execution | Form multipart `file` or JSON `{ image: base64, cameraId: "CAM-01" }` |
| `GET` | `/health` | System health, model readiness, and diagnostic status | None |
| `GET` | `/config` | Retrieve current thresholds (confidence, temporal window, gating flags) | None |
| `POST` | `/config` | Dynamically update pipeline parameters without server restart | JSON config key-value pairs |
| `GET` | `/debug_output/frames/{filename}` | Retrieve archived surveillance frame | Static file |
| `GET` | `/debug_output/crops/{filename}` | Retrieve archived pre-OCR plate crop | Static file |

### Node.js Streaming & Analytics Backend (`https://127.0.0.1:3000`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/detections` | Fetch historic detection records and telemetry |
| `POST` | `/api/detections` | Ingest confirmed detection event from Python worker |
| `GET` | `/api/watchlist` | Retrieve active BOLO vehicle watchlist |
| `POST` | `/api/watchlist` | Add vehicle plate with reason to real-time watchlist |
| `DELETE`| `/api/watchlist/:plate` | Remove vehicle plate from watchlist |
| `GET` | `/api/alerts` | Retrieve real-time security alerts and watchlist hits |

---

## 📁 Repository Structure

```
.
├── models/                               # Consolidated Deep Learning Weights
│   ├── indian_plate_best.pt              # Fine-tuned YOLOv8 PyTorch plate detector
│   ├── indian_plate_best.onnx            # Quantized ONNX plate detector (Edge/Web)
│   ├── yolov8n.pt                        # Base COCO vehicle detector (car, bike, bus, truck)
│   ├── yolov8n.onnx                      # ONNX vehicle detector
│   └── tesseract_benchmark_results.json  # Empirical Tesseract validation benchmarks
├── public/                               # Frontend Single-Page Application (SPA)
│   ├── dashboard.html                    # Unified SPA Command Center (All tab views)
│   ├── camera.html                       # Mobile camera streaming client (WebRTC)
│   ├── index.html                        # Gateway redirect
│   ├── js/
│   │   ├── anpr-engine.js                # Frontend vision overlay & ONNX runner
│   │   ├── camera-config.js              # Camera settings & node pairing logic
│   │   └── shared.js                     # Shared utilities & WebSocket connectors
│   ├── models/
│   │   └── indian_plate_detector.onnx    # Browser-side client inference model
│   ├── style-dashboard.css               # Premium cyber-defense dark UI theme
│   ├── style-camera.css                  # Mobile camera viewport styling
│   └── style-pages.css                   # Auxiliary page styling
├── scripts/                              # Core Python Pipelines & Utilities
│   ├── anpr_server.py                    # Main FastAPI ANPR Inference Microservice
│   ├── benchmark_anpr.py                 # Real-world benchmark & evaluation suite
│   ├── diagnose_pipeline.py              # 16-Point automated diagnostic runner
│   ├── evaluate_tesseract_ocr.py         # Tesseract OCR Character Error Rate test
│   ├── prepare_dataset.py                # Pascal VOC to YOLO format converter
│   ├── train_plate_model.py              # Plate detector fine-tuning script
│   └── verify_*.py                       # Comprehensive E2E verification test suites
├── data/                                 # Runtime JSON Persistent Stores
│   ├── alerts.json                       # Real-time BOLO security alerts
│   ├── detections.json                   # Verified detection history & timestamps
│   └── watchlist.json                    # Active vehicle watchlist
├── docs/                                 # Technical Documentation
│   └── TESSERACT_TRAINING_ROADMAP.md     # Custom LSTM OCR training roadmap
├── package.json                          # Node.js dependencies & scripts
├── requirements.txt                      # Python dependencies (pip install -r)
├── server.js                             # Express & Socket.io WebRTC Streaming Server
├── start_system.bat                      # One-click Windows startup batch script
├── start_system.ps1                      # One-click PowerShell startup script
├── benchmark_results.json                # Empirical test benchmark results
└── benchmark_results.csv                 # Detailed per-image benchmark evaluation
```

---

## 🔧 Troubleshooting & FAQs

### Q1: Mobile camera cannot connect or shows "Camera permission denied"
- Ensure you access the dashboard via `https://<YOUR_LAN_IP>:3000` rather than `http://`.
- In Chrome on Android: You may need to bypass the self-signed certificate warning by typing `thisisunsafe` or clicking "Advanced" $\to$ "Proceed".

### Q2: Tesseract OCR returns empty predictions
- Verify that `tesseract.exe` is installed and accessible. Check by running `tesseract --version` in terminal.
- If installed in a non-standard location, configure `TESSERACT_DEFAULT_PATH` in [`scripts/anpr_server.py`](scripts/anpr_server.py) or add Tesseract to your system `PATH`.

### Q3: Why does tab navigation not disconnect video feeds?
- The dashboard operates as a persistent Single-Page Application (SPA). Switching between navigation tabs toggles CSS visibility without reloading the DOM or destroying WebRTC peer connections.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
Developed for the **Smart India Hackathon (SIH)** under Problem Statement ID **26127** by **Bharat Electronics Limited (BEL)**.
