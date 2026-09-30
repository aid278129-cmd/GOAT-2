# ROUTE DATASET AUDIT & DATA SUFFICIENCY REPORT

**Author:** Antigravity AI Engineering Team  
**Date:** September 28, 2026  
**System:** ANPR Traffic Intelligence Platform  
**Status:** AUDITED & CLASSIFIED AS `INSUFFICIENT_REAL_ROUTE_DATA`  

---

## 1. REAL PRODUCTION / DEMO DATASET METRICS

Analysis of the active dataset (`data/detections.json`):

| Metric | Real Dataset Count | Notes |
| :--- | :--- | :--- |
| **Total ANPR Detections** | 37 | Recorded across 4 surveillance cameras in Chennai corridor |
| **Unique License Plates** | 17 | Passenger cars, commercial vehicles, and motorcycles |
| **Total Journey Sessions** | 24 | Formed using 30-minute journey sessionization rule |
| **Single-Camera Journeys** | 19 | Detections at single locations without onward observation |
| **Multi-Camera Journeys** | 5 | Valid multi-hop trips suitable for trajectory analysis |
| **Total Valid Transitions** | 12 | Sequential camera-to-camera movements |
| **Unique Directed Camera Pairs** | 4 | `CAM_01->CAM_02`, `CAM_02->CAM_03`, `CAM_03->CAM_04`, `CAM_04->CAM_01` |
| **Median Transitions per Edge** | 3.0 | Min: 1 (`CAM_04->CAM_01`), Max: 5 (`CAM_02->CAM_03`) |
| **Testable Prediction Events** | 12 | Total target next-camera transition events |

---

## 2. TRANSITIONS BREAKDOWN BY CAMERA

### Outgoing Transitions:
- **CAM_01 (Junction A):** 3 outgoing transitions $\rightarrow$ 100% to `CAM_02` (3/3)
- **CAM_02 (Junction B):** 5 outgoing transitions $\rightarrow$ 100% to `CAM_03` (5/5)
- **CAM_03 (Junction C):** 3 outgoing transitions $\rightarrow$ 100% to `CAM_04` (3/3)
- **CAM_04 (Highway Entry):** 1 outgoing transition $\rightarrow$ 100% to `CAM_01` (1/1)

### Corridor / 2nd-Order Transitions:
- `CAM_01 -> CAM_02` $\rightarrow$ `CAM_03`: 3 observations
- `CAM_02 -> CAM_03` $\rightarrow$ `CAM_04`: 3 observations
- `CAM_03 -> CAM_04` $\rightarrow$ `CAM_01`: 1 observation

---

## 3. DATA SUFFICIENCY CLASSIFICATION

### Formal Verdict:
```
STATUS: INSUFFICIENT_REAL_ROUTE_DATA
```

### Rationale:
1. **Sample Size:** With only 12 total transitions across 5 multi-camera journeys, splitting into a 70% chronological train (8 transitions) and 30% chronological test (4 transitions) produces test sets with very high variance (1 correct prediction = 25% accuracy swing).
2. **Lack of Branching Diversity:** Every camera in the active dataset has historically transitioned to only **one** destination camera (100% deterministic corridor). There are zero empirical forks/intersections where a vehicle turned into an alternate corridor in the recorded real dataset.
3. **Statistical Validity:** While a first-order Markov model on this dataset will achieve 100% accuracy on its linear sequence, this does not represent true statistical route selection in an urban traffic network.

### Action Plan:
As mandated in Phase 8 and Phase 26:
1. Perform the full benchmark on the **Real Dataset** (reporting exact chronological train/test numbers without fabrication).
2. Create an isolated, clearly labeled **Simulation Environment** in `data/simulation/` with 12 camera nodes and realistic branching traffic corridors (200+ multi-camera trips, origin-destination distribution, and peak/off-peak variations).
3. Report real results and simulation results strictly in separate sections.
