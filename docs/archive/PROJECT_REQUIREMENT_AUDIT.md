# Project Requirement Audit & Compliance Mapping

**Project:** City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking & Urban Traffic Analytics  
**SIH Problem Statement ID:** 26127  
**Client / Organization:** Bharat Electronics Limited (BEL)  
**Audit Date:** September 2026  
**Audit Status:** Baseline Audited  

---

## 1. Audit Framework & Rating Criteria

Each requirement from SIH Problem Statement ID 26127 is rigorously assessed against the actual active implementation in this repository according to the following statuses:
- **COMPLETE**: Fully implemented, empirically verified through automated test suites or benchmarks, and integrated into live system pipelines.
- **PARTIAL**: Implemented in code, but lacks complete real-world capability, fails target threshold, or requires dedicated improvement/fine-tuning.
- **MISSING**: Not implemented or absent from codebase.
- **UNVERIFIED**: Code exists but lacks formal empirical tests or quantitative evidence.

---

## 2. Requirement Traceability Matrix

| Problem Statement Requirement | Current Implementation in Codebase | Verified Status | Test / Evidence | Major Gap Discovered | Action Required |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **1. High-Accuracy License Plate Recognition (>90%)** | `anpr_v2/recognizer.py` (PP-OCRv4) & `anpr_v2/fastplate_recognizer.py` (CCT-S-v2) | **PARTIAL** | Empirical Benchmark on 228 held-out crops (`benchmarks/anpr_v2/ab_benchmark_results.json`) | PP-OCRv4 exact accuracy is **32.02%**; Pretrained FastPlateOCR exact accuracy is **20.61%**. Both fall far short of the >90% statutory mandate. | Fine-tune FastPlateOCR CCT-S-v2 on extensive Indian plate dataset (including two-line format & digit focus); benchmark with post-processing and temporal consensus. |
| **2. Robust Plate Detection Under Diverse Angles & Occlusion** | `anpr_v2/detector.py` (RF-DETR Small & YOLOv8 ONNX fallback) | **COMPLETE** | `scripts/benchmark_anpr.py` & `scripts/test_anpr_v2.py` | 97.14% Precision, 97.14% Recall, 0% FPR. Bypasses vehicle-first gating failures on tight camera crops. | Preserve detector weights (`models/indian_plate_best.onnx`); verify no regression. |
| **3. Non-Destructive Plate Rectification & Crop Normalization** | `anpr_v2/rectifier.py` (10% padding + 4-point homography unskewing) | **COMPLETE** | `test_anpr_v2.py` (Test 04) & Benchmark Test B | Corrects perspective up to 45°. Rectification measurably improves FastPlateOCR accuracy (+1.76%). | Maintain minimal 10% padding and aspect-ratio-preserving normalization. |
| **4. Pure Indian Registration Syntax Validation** | `anpr_v2/validator.py` (MoRTH Rule 50, Bharat Series 'BH', Commercial) | **COMPLETE** | `test_anpr_v2.py` (Test 05) & `benchmarks/regression_test.py` | Validates 36 States/UTs, BH series, and historic plates without character-morphing hallucinations. | Keep validator strictly as a validator; do not fabricate characters. |
| **5. Multi-Frame Temporal Consensus & Tracklet Voting** | `anpr_v2/tracker.py` (Laplacian sharpness scoring + confidence voting) | **COMPLETE** | `test_anpr_v2.py` (Test 06) | Aggregates multi-frame sightings; awards instant confirmation for single-image manual scans. | Benchmark temporal consensus gain over single-crop OCR. |
| **6. Multi-Camera Low-Latency WebRTC Video Ingestion** | `server.js` (Express + Socket.io + WebRTC Signaling on Port 3000 HTTPS) | **COMPLETE** | `scripts/verify_system_e2e.py` & Live Browser Streaming | Low-latency WebRTC streams from mobile phones, IP cameras, and local webcams with QR pairing. | Maintain WebRTC signaling and keepalive timeouts. |
| **7. Multi-Camera Concurrency Queue Management** | `MultiCameraQueueManager` in `server.js` (maxConcurrency: 2, depth-1 queue) | **COMPLETE** | `server.js` queue tests & Socket.io telemetry | Semaphore-bounded CPU execution; drops stale frames to preserve real-time live video feeds. | Maintain queue telemetry and bounded concurrency. |
| **8. Chronological Vehicle Trajectory Reconstruction** | `services/trajectoryService.js` & `services/geoService.js` | **COMPLETE** | `tests/test_phase_b_trajectory.js` (13/13 PASS) | Stitches multi-camera sightings, computes leg distance (Haversine), travel time, leg speeds, and journey stats. | Verify trajectory persistence after OCR improvements. |
| **9. Impossible Travel / Speed Anomaly Detection** | `services/trajectoryService.js` (`IMPOSSIBLE_TRAVEL` flag for >180 km/h) | **COMPLETE** | `test_phase_b_trajectory.js` (Edge case test) | Detects cloned plates or timestamp corruptions exceeding 180 km/h. | Expand with additional deterministic anomaly rules (loops, restricted zone). |
| **10. Macro Urban Traffic Density Analytics** | `services/trafficAnalytics.js` (LOW, MEDIUM, HIGH, SEVERE tiers) | **COMPLETE** | `tests/test_phase_c_analytics.js` (18/18 PASS) | 12s deduplication window; computes per-camera and zone traffic volume and density classification. | Preserve existing analytics service architecture. |
| **11. Camera-to-Camera Route Density Analytics** | `services/routeAnalytics.js` | **COMPLETE** | `test_phase_c_analytics.js` | Identifies busiest corridors and cross-camera transitions across time windows (15m, 30m, 1h, today). | Maintain API compatibility with dashboard tables. |
| **12. Network-Wide & Route-Specific Average Speed** | `services/speedAnalytics.js` | **COMPLETE** | `test_phase_c_analytics.js` | Computes weighted average network speed and leg-by-leg speed distribution. | Maintain compatibility with speed analytics endpoints. |
| **13. Origin-Destination (OD) Travel Matrix** | `services/odAnalytics.js` | **COMPLETE** | `test_phase_c_analytics.js` | Computes square origin-destination journey matrix across camera network nodes. | Preserve `/api/analytics/origin-destination` API. |
| **14. Deterministic Congestion Detection** | `services/congestionAnalytics.js` | **COMPLETE** | `test_phase_c_analytics.js` | Compares current route speed and volume against baseline parameters (`data/analytics_config.json`). | Preserve `/api/analytics/congestion` API. |
| **15. Time-Series Traffic Flow & Speed Trends** | `services/trendAnalytics.js` | **COMPLETE** | `test_phase_c_analytics.js` | Aggregates volume and speed into uniform time buckets (5m, 15m, 30m, 1h). | Preserve `/api/analytics/trends` API. |
| **16. GIS Interactive Map & Traffic Heatmaps** | `public/js/trafficMap.js` & `public/js/leaflet-heat.js` (Leaflet.js) | **COMPLETE** | `tests/test_phase_d_gis.js` (12/12 PASS) | 4 dynamic map overlays: Camera Markers, Density Heatmap, Route Flow Polylines, and Trajectory paths. | Verify interactive layer toggles and auto-refresh sync. |
| **17. Real-Time Watchlist & BOLO Hotlist Alerting** | `server.js` (`/api/watchlist`, `/api/alerts`) & `data/watchlist.json` | **COMPLETE** | `scripts/verify_trajectory_and_alerts.py` | Instant BOLO matching, priority tiers (CRITICAL, HIGH, MEDIUM, LOW), and audio HUD siren broadcasts. | Verify alert dispatching on newly recognized plates. |
| **18. Mobile Smartphone Camera QR Code Pairing** | `server.js` (`/api/qrcode/:id`) & `public/camera.html` | **COMPLETE** | `scripts/verify_system_e2e.py` | Generates TLS-secured QR codes for rapid field pairing of phone cameras over LAN/Wi-Fi. | Maintain QR generation endpoint. |
| **19. Operator Command Center SPA Dashboard** | `public/dashboard.html` & `public/style-dashboard.css` (5-Tab SPA) | **COMPLETE** | Manual and programmatic E2E testing | 5 persistent tabs (`#cmd`, `#track`, `#analytics`, `#network`, `#watchlist`) with zero-reload hash routing. | Ensure frontend renders updated OCR telemetry seamlessly. |
| **20. Suspicious Route Anomaly Layer (Extended)** | `services/trajectoryService.js` (Currently has speed anomaly only) | **PARTIAL** | Speed check in `test_phase_b_trajectory.js` | Missing explicit rules for: Route Loop detection, Restricted Camera Zone Entry, Rapid Multi-Camera Sequences. | Implement deterministic anomaly detector module with configurable rules. |

---

## 3. Summary of Major Gaps & Priority Focus

1. **OCR Accuracy (Critical Blocker):**
   - Both PP-OCRv4 (32.02%) and pretrained FastPlateOCR (20.61%) fail the >90% exact-match target.
   - FastPlateOCR has an immense speed advantage (20.9 ms vs. 209.5 ms) and high series/state recognition, but suffers from tail digit omission (`6 -> [OMIT]`, `4 -> [OMIT]`) and two-line plate collapse.
   - **Primary Action:** Fine-tune FastPlateOCR CCT-S-v2 on extensive, balanced Indian license plate data with explicit two-row plate handling, followed by temporal consensus and post-processing evaluation.

2. **Route Anomalies (Secondary Gap):**
   - Trajectory tracking correctly flags impossible speed (>180 km/h).
   - **Action:** Formalize a lightweight, deterministic anomaly rule evaluator for repeated route loops and restricted zone entry.

3. **Core Functionality Preservation:**
   - Detectors, WebRTC, Node.js backend, Leaflet GIS, and Analytics are fully functional and pass 100% of their test suites (43 Node.js tests, 15 Python tests). These must remain intact.
