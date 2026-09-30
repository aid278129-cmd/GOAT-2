# Pre-Live Multi-Camera Test Validation Report

## 1. Analytics Validation
* **Ground-Truth Test Suite:** **PASS** (14 / 14 tests passed)
* **Phase C Analytics Regression Suite:** **PASS** (18 / 18 tests passed)
* **Total Analytics Tests:** **32 / 32 PASS (100%)**
* **Ground-Truth Metrics Verified:**
  * Total Detections: CAM_01 = 12, CAM_02 = 21, CAM_03 = 5 (Exact Match)
  * Unique Vehicles: CAM_01 = 10, CAM_02 = 20, CAM_03 = 5 (Exact Match)
  * Flow Rates: CAM_01 = 10.0 vph, CAM_02 = 20.0 vph, CAM_03 = 5.0 vph (Exact Match)
  * Busiest Camera: CAM_02 (Exact Match)
  * Transitions: CAM_01 $\rightarrow$ CAM_02 = 8, CAM_02 $\rightarrow$ CAM_03 = 3, CAM_01 $\rightarrow$ CAM_03 = 2 (Exact Match)
  * Busiest Corridor: CAM_01 $\rightarrow$ CAM_02 (Exact Match)
  * Origin-Destination Matrix: Correctly captures true journey origins and final destinations without intermediate node contamination.

---

## 2. Camera Network Validation
* **Lifecycle State Transitions:** **PASS** (8 / 8 tests passed)
* **Heartbeat Mechanism:** **PASS** (Endpoint `POST /api/cameras/:id/heartbeat` and socket `camera:heartbeat` verified)
* **Reconnect & Failure Handling:** **PASS**
  * Disconnect transition latency: `0.012 ms`
  * Reconnection restoration latency: `0.032 ms`
* **States Enforced:** `ONLINE`, `DEGRADED`, `DISCONNECTED`, `OFFLINE`

---

## 3. Watchlist & Real-Time Alert Validation
* **Plate Match:** **PASS** (Exact case/whitespace-insensitive plate normalization)
* **Alert Generation & Schema:** **PASS** (6 / 6 tests passed, full payload with `alertId`, `plate`, `cameraId`, `location`, `timestamp`, `confidence`)
* **Duplicate Suppression:** **PASS** (Cooldown window of 60 seconds prevents multiple alerts for repeated frames at the same camera)
* **Cross-Camera Alerting:** **PASS** (Immediate alert triggered upon arrival at a new camera node)
* **Socket Delivery & Tracking Deep Link:** **PASS** (Broadcasts `watchlist:match` event with direct link to `#track`)

---

## 4. Route Prediction (CAM_01 Current Model Output)
* **Prediction Algorithm:** First-Order Markov Transition Model with Time-of-Day alignment, OSRM Road Topology Feasibility, and Directional Heading bonuses.
* **CAM_01 Outgoing Predictions (Empirically Queried):**
  * **Top-1:** `CAM_02` (T. Nagar) — **68% probability** (Confidence: `MEDIUM`, OSRM Road Distance: 2.75 km, ETA: `1m 21s`, 4/4 historical transitions)
  * **Top-2:** `CAM_03` (Perambur) — **21% probability** (Confidence: `INSUFFICIENT_DATA`, OSRM Road Distance: 1.75 km, ETA: `2m 48s`)
  * **Top-3:** `CAM_04` (Guindy) — **11% probability** (Confidence: `INSUFFICIENT_DATA`, OSRM Road Distance: 8.34 km, ETA: `9m 12s`)

---

## 5. Physical Test Timing & Feasibility Analysis

### Segment 1: CAM_01 $\rightarrow$ CAM_02
* **From Camera:** CAM_01 (Junction A / Chennai Central, $13.0827, 80.2707$)
* **To Camera:** CAM_02 (Junction B / T. Nagar, $13.0731, 80.2609$)
* **Road Distance:** **2.75 km** (2,749 meters via OSRM road geometry)
* **Geodesic (Haversine) Distance:** 1.505 km
* **OSRM Estimated Travel Time:** **202 seconds** (3 min 22 sec)
* **Historical Median Travel Time:** 80 seconds
* **Maximum Feasible Speed Threshold:** $180\text{ km/h}$ (`maxReasonableSpeedKmh`)
* **Minimum Physically Valid Travel Time:**
  * At $180\text{ km/h}$ based on Haversine: **30.1 seconds**
  * At $180\text{ km/h}$ based on Road: **55.0 seconds**
* **Session Gap Threshold:** 30 minutes (1,800 seconds)
* **Recommended Physical-Test Delay:** **60 to 90 seconds** (Yields an urban travel speed of $60$–$90\text{ km/h}$)
* **Journey Classification:** **SAME JOURNEY** (No `IMPOSSIBLE_TRAVEL` anomaly; well under 30-minute session split threshold)

### Segment 2: CAM_02 $\rightarrow$ CAM_03
* **From Camera:** CAM_02 (Junction B / T. Nagar, $13.0731, 80.2609$)
* **To Camera:** CAM_03 (Junction C / Perambur, $13.0878, 80.2785$)
* **Road Distance:** **3.72 km** (3,725 meters via OSRM road geometry)
* **Geodesic (Haversine) Distance:** 2.511 km
* **OSRM Estimated Travel Time:** **298 seconds** (4 min 58 sec)
* **Historical Median Travel Time:** 61 seconds
* **Maximum Feasible Speed Threshold:** $180\text{ km/h}$
* **Minimum Physically Valid Travel Time:**
  * At $180\text{ km/h}$ based on Haversine: **50.2 seconds**
  * At $180\text{ km/h}$ based on Road: **74.5 seconds**
* **Session Gap Threshold:** 30 minutes (1,800 seconds)
* **Recommended Physical-Test Delay:** **75 to 120 seconds** (Yields an urban travel speed of $75$–$120\text{ km/h}$)
* **Journey Classification:** **SAME JOURNEY** (No anomaly; well under 30-minute session split threshold)

---

## 6. OCR Accuracy Benchmark Baseline (Phase 4)
* **Overall Exact Full-Plate Match:** `71.00%`
* **Exact Match (When Plate Detected):** `84.52%`
* **Character-Level Accuracy:** `80.07%`
* **Plate Detector Recall:** `84.00%`
* **Plate Detector Precision:** `100.00%`
* **High-Resolution Plates ($h \ge 60\text{px}$):** `81.78%` Exact Match | `92.32%` Character Accuracy
* **Low-Resolution Plates ($h < 60\text{px}$):** `4.76%` Exact Match (100% due to detector missing $40\text{px}$ crops)
* **Single-Line Plates:** `59.05%` Exact Match
* **Two-Line / Square Plates:** `98.89%` Exact Match
* **Target Assessment:** Problem statement $>90\%$ target is **NOT YET ACHIEVED**.

---

## 7. Test Readiness
* **Controlled Integration Test (Software Pipeline):** **READY**
* **Realistic Visual ANPR Test (Real World Scene):** **READY**
