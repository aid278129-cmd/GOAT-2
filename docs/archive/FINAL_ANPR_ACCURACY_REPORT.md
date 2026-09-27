# Final ANPR Accuracy & Benchmark Report
## Indian License Plate Recognition Engine Evaluation
**Evaluation Dataset:** 228 Untouched Held-Out Indian License Plate Crops (`benchmarks/anpr_v2/crops/`)  
**Hardware Platform:** Intel Core CPU (4 Logical Cores), 7.8 GB RAM, Windows 11 x64  
**Date of Verification:** September 2026  
**Audited Models:** PP-OCRv4 (RapidOCR ONNX), FastPlateOCR Pretrained CCT-S-v2 (Global), FastPlateOCR Fine-Tuned CCT-S-v2 (Indian), Adaptive Fallback Pipeline  

---

### 1. License Plate Detector Performance
Evaluated using the dedicated YOLOv8 Indian Plate Detector (`models/indian_plate_best.onnx`) on standard surveillance imagery and negative control backgrounds:

| Metric | Measured Value | Statutory Target | Status |
| :--- | :--- | :--- | :--- |
| **Precision** | **97.14%** | > 95.0% | **PASS** |
| **Recall** | **97.14%** | > 95.0% | **PASS** |
| **False Positive Rate (FPR)** | **0.00%** | < 2.0% | **PASS** |
| **Negative Control Rejection** | **100.00%** (35/35) | 100.0% | **PASS** |
| **Inference Latency** | **18.42 ms** | < 50.0 ms | **PASS** |

*Note: The detector maintained high precision and zero false positives. In accordance with Phase 15 rules, the detector was frozen to prevent unnecessary regressions.*

---

### 2. Comprehensive OCR Recognizer Comparison
Evaluated on the **228 untouched, frozen Indian license plate test set** (zero test-set leakage, SHA-256 hashed in `frozen_test_set_manifest.json`):

| Model / Pipeline | Raw Full-Plate Exact Match | Character Accuracy | Mean Edit Distance | Median Edit Distance | Perfect Matches (ED=0) | 1-Char Errors (ED=1) | 2-Char Errors (ED=2) | ≥3 Errors (ED≥3) | Mean Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: PP-OCRv4 (Baseline)** | 32.02% (73/228) | 67.94% | 3.38 | 2.0 | 73 | 28 | 29 | 98 | 347.8 ms | 2.9 FPS |
| **Model B: Pretrained FastPlateOCR CCT-S-v2** | 20.61% (47/228) | 69.15% | 2.98 | 1.0 | 47 | 73 | 24 | 84 | **52.3 ms** | **19.1 FPS** |
| **Model C: Indian Fine-Tuned CCT-S-v2** | **43.42%** (99/228) | **74.35%** | **2.47** | **1.0** | **99** | 44 | 9 | 76 | 168.1 ms | 6.0 FPS |
| **Pipeline: Adaptive Fallback (CCT + PP-OCR)** | **51.32%** (117/228) | **76.50%** | **2.29** | **0.0** | **117** | 40 | 3 | 68 | 215.4 ms | 4.6 FPS |
| **Theoretical Oracle Union (Upper Bound)** | **53.95%** (123/228) | — | — | — | 123 | — | — | — | — | — |

#### Key Performance Takeaways:
1. **Model C (Indian Fine-Tuned CCT-S-v2)** more than **doubled** the exact accuracy of Pretrained FastPlateOCR (**43.42% vs. 20.61%**, +110.7% relative gain).
2. **Model C surpassed PP-OCRv4** by **+11.40% absolute** (**43.42% vs. 32.02%**), while reducing mean edit distance from 3.38 to 2.47.
3. The **Adaptive Fallback Pipeline** achieved **51.32% exact full-plate accuracy**, successfully unlocking complementary correct detections from both engines.

---

### 3. Pipeline Hierarchy & Post-Processing Metrics

| Pipeline Tier | Exact Full-Plate Recognition | Character Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **Raw Single-Crop Exact** | **43.42%** (99/228) | 74.35% | Direct transformer prediction without heuristic modification |
| **Post-Processed Single-Crop** | **43.42%** (99/228) | 74.35% | Indian MoRTH Rule 50 syntax validator with anti-hallucination policy |
| **Multi-Frame Temporal Consensus** | **34.09%** (15/44 vehicle chunks) | 71.12% | Grouped voting simulation over consecutive surveillance frames |
| **Adaptive Complementary Fallback** | **51.32%** (117/228) | 76.50% | High-confidence CCT primary with calibrated PP-OCRv4 format fallback |

---

### 4. Layout Breakdown: Single-Line vs. Two-Line Plates

Two-line (square/stacked) plates represent 47.4% of the held-out benchmark (108/228 crops). The previous FastPlateOCR baseline suffered catastrophic failure on two-line plates.

| Plate Layout Format | Total Test Samples | Fine-Tuned CCT-S-v2 Exact Matches | Exact Accuracy (%) | Pretrained CCT-S-v2 Accuracy | Net Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Line (Standard Strip)** | 120 | 45 | **37.50%** | 39.17% | -1.67% (format trade-off) |
| **Two-Line (Square / Stacked)** | 108 | 54 | **50.00%** | **0.00%** | **+50.00% (MASSIVE LEAP)** |
| **Total Test Cohort** | 228 | 99 | **43.42%** | 20.61% | **+22.81%** |

*Crucial Finding:* The fine-tuned transformer learned to parse two-line plate topologies directly in its native (128, 64) receptive field, leaping from **0% to 50.00%** exact recognition on two-line motorcycle and commercial crops.

---

### 5. Real-World Environmental Condition Breakdown

Evaluated across the 6 condition partitions present in the frozen benchmark:

| Environmental Condition | Sample Count | Exact Matches | Exact Accuracy (%) | Mean Character Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| **High Contrast / Glare** | 38 | 19 | **50.00%** | **76.28%** |
| **Clean Daylight** | 38 | 17 | **44.74%** | **74.91%** |
| **Motion / Focus Blur** | 38 | 17 | **44.74%** | **73.58%** |
| **Perspective Skew (≤45°)** | 38 | 17 | **44.74%** | **74.48%** |
| **Low Light / Night Dim** | 38 | 15 | **39.47%** | **72.87%** |
| **Heavy JPEG Compression** | 38 | 14 | **36.84%** | **73.98%** |

---

### 6. Complementarity Analysis (Phase 13)

| Prediction Category | Sample Count | % of Test Set | Description |
| :--- | :--- | :--- | :--- |
| **Both Models Correct** | 49 | 21.49% | Concordant exact match by both CCT-S-v2 and PP-OCRv4 |
| **Fine-Tuned CCT Only Correct** | 50 | 21.93% | CCT succeeded where PP-OCRv4 failed (predominantly 2-row plates) |
| **PP-OCRv4 Only Correct** | 24 | 10.53% | PP-OCRv4 succeeded where CCT failed (predominantly degraded single-line) |
| **Neither Model Correct** | 105 | 46.05% | Severe degradation, occlusions, or illegal character fonts |
| **Oracle Union (Upper Bound)** | **123** | **53.95%** | Theoretical ceiling if an omniscient selector chose the right model |

---

### 7. Character Confusion Analysis

The top 12 character confusion pairs observed on the fine-tuned model:

| Ground Truth Character | Predicted Character | Error Count | Underlying Root Cause |
| :--- | :--- | :--- | :--- |
| **8** | **0** | 14 | Inner loop closure in low-resolution / compressed crops |
| **P** | **A** | 13 | Top horizontal bar and diagonal leg similarity in square plates |
| **6** | **4** | 12 | Open loop geometry in thin aftermarket fonts |
| **4** | **A** | 11 | Triangular apex resemblance under acute perspective |
| **1** | **8** | 11 | Smudged vertical stroke or screw hole proximity |
| **L** | **4** | 10 | Corner stroke ambiguity in two-line bottom row |
| **J** | **0** | 10 | Tail curvature loss in low-light conditions |
| **3** | **4** | 9 | Mid-stroke horizontal overlap in condensed plates |
| **A** | **4** | 9 | Horizontal crossbar ambiguity |
| **2** | **5** | 9 | Reversible curve geometry in blurred frames |
| **3** | **1** | 9 | Loss of middle/bottom loops in dirty crops |
| **2** | **8** | 9 | Loop closure caused by road dirt or mounting screws |

---

### 8. Latency & Throughput Profile

Measured on host CPU (Intel 4 cores, batch size 1):

| Engine / Component | Mean Latency | Median Latency | P95 Latency | Single-Thread Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Plate Detector (YOLOv8)** | 18.4 ms | 17.1 ms | 26.5 ms | 54.3 FPS |
| **Perspective Rectification** | 4.8 ms | 4.2 ms | 8.1 ms | 208.3 FPS |
| **Pretrained CCT-S-v2 (ONNX)** | 52.3 ms | 45.7 ms | 95.5 ms | 19.1 FPS |
| **Indian Fine-Tuned CCT-S-v2 (PyTorch)** | 168.1 ms | 150.5 ms | 321.8 ms | 6.0 FPS |
| **PP-OCRv4 Multi-Line (ONNX)** | 347.8 ms | 315.5 ms | 1119.5 ms | 2.9 FPS |
| **Adaptive Fallback Pipeline** | 215.4 ms | 185.0 ms | 450.2 ms | 4.6 FPS |

---

### 9. Statutory >90% Target Compliance Verdict

- **Problem Statement Statutory Requirement:** > 90% License-Plate Recognition Accuracy across realistic conditions.
- **Strict Verification Metric:** `exact_correct / total_test_images > 0.90` on untouched test set.
- **Actual Measured Result:**
  - Raw Indian Fine-Tuned CCT-S-v2: **43.42%** (99 / 228)
  - Adaptive Fallback Pipeline: **51.32%** (117 / 228)
  - Character-level Accuracy: **74.35%** (CCT) / **76.50%** (Fallback)
- **Compliance Verdict:** **NOT YET COMPLIANT (>90% TARGET NOT REACHED)**.

Per the master project instructions, this result is reported with complete scientific honesty without metric manipulation, data filtering, or rounding. Evidence-based next steps to bridge the remaining gap are provided in `NEXT_RECOMMENDED_ACTIONS.md`.
