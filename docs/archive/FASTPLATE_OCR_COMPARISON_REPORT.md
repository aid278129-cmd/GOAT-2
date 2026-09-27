# FastPlateOCR vs. PP-OCRv4 Benchmark & Comparative Evaluation Report

**Project:** City-Wide Multi-Camera ANPR Traffic Intelligence Platform  
**Problem Statement ID:** 26127 (Bharat Electronics Limited - BEL)  
**Evaluation Date:** September 2026  
**Document Status:** Complete Empirical Benchmark  

---

## 1. Executive Summary

We conducted a rigorous, isolated comparative evaluation benchmarking the candidate **FastPlateOCR** (`cct-s-v2-global-model`, Compact Convolutional Transformer) against the production baseline **PP-OCRv4** (`RapidOCR` ONNX runtime) on the identical 228 held-out real Indian license plate crops dataset. The empirical results demonstrate that while FastPlateOCR delivers exceptional sub-21ms CPU inference throughput (47.6 plates/sec vs. 4.8 plates/sec, a ~10x speedup) and strong high-level character alignment (69.15% character accuracy, 1.0 median edit distance, 66.67% state code accuracy), its **exact full-plate accuracy drops by 11.41 percentage points (20.61% vs. 32.02%)**, classifying this pretrained checkpoint as a **REGRESSION** under our engineering evaluation criteria. FastPlateOCR's primary points of failure stem from severe character omission on the final registration number digits (20.18% number accuracy vs. 39.04%) and a lack of multi-line spatial decomposition for two-row square plates (25.64% vs. 50.00% for PP-OCRv4). Consequently, neither model meets the 90% production target (FastPlateOCR exhibits a 69.39 percentage point deficit), and our definitive recommendation is to **FINE-TUNE FASTPLATEOCR ON INDIAN PLATES BEFORE PRODUCTION USE** while retaining PP-OCRv4 as the active production baseline.

---

## 2. Test Environment

| Attribute | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Home / Pro (Build 10.0.26200-SP0, x64) |
| **CPU Processor** | Intel64 Family 6 Model 140 Stepping 1 (2 Physical Cores, 4 Logical Threads) |
| **System RAM** | 7.80 GB DDR4 |
| **GPU Acceleration** | None (Evaluated strictly on CPUExecutionProvider) |
| **Python Version** | Python 3.14.3 (64-bit AMD64) |
| **ONNX Runtime** | v1.30.0 (`onnxruntime` native binary) |
| **Execution Provider** | `CPUExecutionProvider` |
| **Candidate Package** | `fast-plate-ocr` v1.1.0 (Official GitHub release) |
| **Candidate Model** | `cct-s-v2-global-model` (Compact Convolutional Transformer - Small v2) |
| **Model Weight Paths** | ONNX: `~/.cache/fast-plate-ocr/cct-s-v2-global-model/cct_s_v2_global.onnx`<br>Config: `~/.cache/fast-plate-ocr/cct-s-v2-global-model/cct_s_v2_global_plate_config.yaml` |
| **Candidate Init Latency** | 199.16 ms (One-time initialization) |
| **Baseline Init Latency** | 398.90 ms (One-time initialization) |

---

## 3. Dataset Characteristics & Ground Truth

- **Total Test Samples:** 228 real Indian license plate crops.
- **Ground Truth Source:** Held-out validation index (`benchmarks/anpr_v2/ground_truth_benchmark.json`).
  - Derived from verified Pascal VOC OCR annotations (25 base vehicle scenes), Challenge Dataset collection (7 base vehicles), and Live Surveillance stream sequences (6 multi-frame crops of `MH01AV8866`).
  - Expanded into 6 controlled testing conditions (38 base + 190 condition variants: Clean, Mild Blur, Low Light, Contrast, JPEG Compression, and Skew Angle).
- **Plate Normalization:** All ground truth and prediction tokens were normalized identically using `clean_ocr_raw_tokens` (uppercase alphanumeric only, whitespace/hyphens stripped, leading blue-strip 'IND' stripped if length $\ge$ 8). Zero character morphing or validator guessing was applied to ground truth strings.
- **Row Layout Distribution:**
  - **Single-Line (Horizontal) Plates:** 150 crops (65.8%, Aspect Ratio $\ge$ 2.0).
  - **Two-Line (Stacked Square) Plates:** 78 crops (34.2%, Aspect Ratio $<$ 2.0, primarily auto-rickshaws and motorcycles).
- **Important Limitations:** Dataset contains high visual variance, skewed camera perspectives (up to 40°), varying illumination, and HSRP embossed fonts.

---

## 4. Models Evaluated

### Baseline Model: Current PP-OCRv4 / RapidOCR
- **Architecture:** PaddleOCR PP-OCRv4 text detection and recognition pipeline (`en_PP-OCRv4_rec`).
- **Runtime Wrapper:** `rapidocr_onnxruntime`.
- **Special Capabilities:** Automatic line-aware spatial segmentation (detects separate bounding boxes for multi-line text and concatenates top-to-bottom), character whitelist restriction.
- **Role:** Production Baseline.

### Candidate Model: FastPlateOCR CCT-S-v2 Global
- **Architecture:** Compact Convolutional Transformer (CCT-S-v2) trained on diverse global license plate datasets across 65+ jurisdictions.
- **Model Parameters:** Resizes all inputs to fixed dimension $128 \times 64$ RGB (`PlateConfig`: `max_plate_slots=10`, `keep_aspect_ratio=False`).
- **Runtime:** ONNX Runtime via `fast_plate_ocr.LicensePlateRecognizer`.
- **Role:** Isolated Candidate under evaluation.

---

## 5. Main Benchmark Results

Evaluated across all 228 benchmark crops after model warm-up (Test A: Raw Crops):

| METRIC | CURRENT (PP-OCRv4) | FASTPLATEOCR (CCT-S-v2) | DIFFERENCE | STATUS / IMPACT |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Plate Accuracy** | **32.02%** (73/228) | **20.61%** (47/228) | **-11.41%** | 🔴 **REGRESSION** |
| **Post-Processed Exact Accuracy** | **32.02%** (73/228) | **18.42%** (42/228) | **-13.60%** | 🔴 **Severe Gap** |
| **Mean Character Accuracy** | 67.94% | **69.15%** | **+1.21%** | 🟢 Minor Gain |
| **Average Edit Distance** | 3.382 | **2.978** | **-0.404** | 🟢 Closer Proximity |
| **Median Edit Distance** | 2.0 chars | **1.0 char** | **-1.0 char** | 🟢 High Near-Matches |
| **1-Character Error Rate** | 12.28% (28/228) | **32.02%** (73/228) | **+19.74%** | 🟡 Dominant Failure Mode |
| **2-Character Error Rate** | 12.72% (29/228) | **10.53%** (24/228) | **-2.19%** | 🟢 Lower Multi-Error |
| **3+ Character Error Rate** | 42.98% (98/228) | **36.84%** (84/228) | **-6.14%** | 🟢 Fewer Disasters |
| **Empty Predictions** | 7 (3.07%) | **1 (0.44%)** | **-6** | 🟢 Almost Zero Empty |
| **Invalid Indian Plates** | 116 (50.88%) | **83 (36.40%)** | **-33** | 🟢 Higher Indian Validity |
| **Mean Inference Latency** | 209.51 ms | **20.99 ms** | **-188.52 ms** | 🟢 **9.98x Faster** |
| **Median Inference Latency** | 212.26 ms | **19.86 ms** | **-192.40 ms** | 🟢 **10.69x Faster** |
| **P95 Tail Latency** | 586.28 ms | **25.87 ms** | **-560.41 ms** | 🟢 **22.66x Tighter Tail** |
| **Throughput (Plates / Sec)** | 4.8 / sec | **47.6 / sec** | **+42.8 / sec** | 🟢 **Massive Scaling** |

---

## 6. Exact Full-Plate Accuracy Analysis

```
CURRENT (PP-OCRv4):       73 / 228  (32.02%)
FASTPLATEOCR (CCT-S-v2):   47 / 228  (20.61%)
─────────────────────────────────────────────
Absolute Difference:      -26 plates (-11.41 percentage points)
Evaluation Outcome:       REGRESSION (> 3.0 percentage point deficit)
```

FastPlateOCR achieves exact matches on only 47 plates compared to PP-OCRv4's 73 plates. Crucially, a prediction was counted as correct **strictly** when `predicted_text == ground_truth_text`. No "almost correct" or near-match leniency was permitted.

---

## 7. Indian Plate Component Accuracy

Decomposing Indian registrations into their 4 statutory components reveals exactly where FastPlateOCR succeeds and where it breaks down:

| Plate Component | Example | CURRENT (PP-OCRv4) | FASTPLATEOCR | Component Delta |
| :--- | :--- | :---: | :---: | :---: |
| **State Code** | `TN`, `DL`, `KL`, `MH` | 53.07% | **66.67%** | 🟢 **+13.60%** (Superior) |
| **District / RTO Code** | `45`, `01`, `26`, `3` | 44.30% | **54.82%** | 🟢 **+10.52%** (Superior) |
| **Series Letters** | `AB`, `CBL`, `MPL`, `DY` | 35.53% | **50.00%** | 🟢 **+14.47%** (Superior) |
| **Registration Number** | `1234`, `6855`, `8866` | **39.04%** | **20.18%** | 🔴 **-18.86% (Critical Deficit)** |

### Key Diagnostic Discovery:
FastPlateOCR outperforms PP-OCRv4 on the first three sections of the license plate (State, District, Series). However, its accuracy plummets on the final 4-digit registration number (**20.18% vs. 39.04%**). Because a full plate requires all 4 components to be correct simultaneously, this failure on the final digits drags down the entire exact-match rate.

---

## 8. Confusion Analysis

Character alignments were computed via dynamic programming (Needleman-Wunsch / Levenshtein alignment) across all 228 plates:

### Top Character Confusions for FastPlateOCR:
1. `6 -> [OMIT]` (41 occurrences) — Truncated or lost digit '6'.
2. `4 -> [OMIT]` (23 occurrences) — Truncated or lost digit '4'.
3. `8 -> [OMIT]` (23 occurrences) — Truncated or lost digit '8'.
4. `2 -> [OMIT]` (23 occurrences) — Truncated or lost digit '2'.
5. `H -> [OMIT]` (18 occurrences) — Truncated or lost letter 'H'.
6. `0 -> O` (15 occurrences) — Optical confusion between digit zero and letter O.
7. `9 -> [OMIT]` (14 occurrences) — Truncated or lost digit '9'.
8. `A -> [OMIT]` (13 occurrences) — Letter omission.
9. `4 -> A` (12 occurrences) — Slanted '4' misclassified as 'A'.
10. `6 -> 4` (12 occurrences) — Digit confusion.
11. `1 -> 2` (11 occurrences) — Font confusion.
12. `8 -> 4` (10 occurrences) — Stroke confusion.
13. `6 -> 0` (10 occurrences) — Curved glyph confusion.
14. `A -> 0` (8 occurrences) — Triangular stroke confusion.
15. `1 -> 6` (8 occurrences) — Serif confusion.

### Comparison with PP-OCRv4 Confusions:
- **PP-OCRv4 Primary Confusions:** `3 -> [OMIT]` (34), `0 -> O` (29), `6 -> G` (28), `4 -> [OMIT]` (21), `V -> Y` (19), `4 -> 2` (14), `K -> X` (9), `8 -> 2` (9).
- FastPlateOCR exhibits **significantly more omission errors on numeric digits** at the tail of the plate. This directly explains why FastPlateOCR achieves a high 1-character error rate (32.02%): it reads 9 out of 10 characters correctly, but drops the last digit due to the global model's `max_plate_slots=10` slot-attention ceiling on crowded Indian plates.

---

## 9. Error Distribution

| Error Category | CURRENT (PP-OCRv4) | FASTPLATEOCR | Shift / Difference |
| :--- | :---: | :---: | :---: |
| **Exact Match (0 Errors)** | 73 (32.02%) | 47 (20.61%) | -26 plates (-11.41%) |
| **1 Character Wrong** | 28 (12.28%) | **73 (32.02%)** | **+45 plates (+19.74%)** |
| **2 Characters Wrong** | 29 (12.72%) | 24 (10.53%) | -5 plates (-2.19%) |
| **3+ Characters Wrong (Catastrophic)** | 98 (42.98%) | **84 (36.84%)** | **-14 plates (-6.14%)** |
| **Total Test Samples** | 228 (100.0%) | 228 (100.0%) | — |

**Takeaway:** FastPlateOCR shifts errors dramatically into the **1-character-wrong bucket** (32.02% of all crops). Combining exact matches and 1-character errors:
- FastPlateOCR achieves $\le 1$ error on **52.63%** of all plates (120 / 228).
- PP-OCRv4 achieves $\le 1$ error on **44.30%** of all plates (101 / 228).

---

## 10. Complementarity & Oracle Theoretical Maximum

Analyzing per-image win/loss outcomes reveals whether an ensemble or fallback strategy could offer value:

```
┌─────────────────────────────────────────────────────────────┐
│                 SAMPLE PREDICTION COMPLEMENTARITY           │
├─────────────────────────────────────────────────────────────┤
│ Correct ONLY by CURRENT (PP-OCRv4):       39 plates (17.1%) │
│ Correct ONLY by FASTPLATEOCR:             13 plates  (5.7%) │
│ Correct by BOTH Models:                   34 plates (14.9%) │
│ Failed by BOTH Models:                   142 plates (62.3%) │
├─────────────────────────────────────────────────────────────┤
│ THEORETICAL ORACLE MAXIMUM:               86 plates (37.72%)│
└─────────────────────────────────────────────────────────────┘
```

### Theoretical Analysis:
If an "oracle" selector could perfectly choose the correct prediction whenever *either* model succeeded, the system would achieve **37.72% exact accuracy** (86 / 228). This represents only a modest **+5.7 percentage point gain** over PP-OCRv4 alone (32.02%), while 62.3% of crops remain unrecognized by both models. Thus, a simple parallel fallback would not bridge the gap to 90%.

---

## 11. Preprocessing Comparison (Test A vs. Test B)

We evaluated both models on **Test A (Raw Crops)** and **Test B (Perspective Rectified + Height Normalized Crops)** using the project's existing [`anpr_v2/rectifier.py`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/anpr_v2/rectifier.py):

| Model Evaluated | Test A (Raw Crop) | Test B (Rectified Crop) | Accuracy Delta | Impact on Model |
| :--- | :---: | :---: | :---: | :---: |
| **FASTPLATEOCR** | 20.61% (47/228) | **22.37% (51/228)** | **+1.76%** | 🟢 **IMPROVES** |
| **CURRENT (PP-OCRv4)** | 32.02% (73/228) | 32.02% (73/228) | $\pm 0.00\%$ | ⚪ No Change in Exact Match (Char Acc improved from 67.9% to 70.6%) |

**Assessment:**
Preprocessing **IMPROVES** FastPlateOCR accuracy by **+1.76 percentage points** (bringing exact matches from 47 to 51). Because FastPlateOCR relies on a global spatial transformer without an internal line detector, 4-point perspective unskewing helps align characters along the expected horizontal axes.

---

## 12. Two-Line vs. Single-Line Plate Breakdown

Indian vehicles feature both standard horizontal plates (private cars, commercial trucks) and stacked square plates (motorcycles, auto-rickshaws, tempo cabs):

| Plate Form Factor | Aspect Ratio | Samples | CURRENT (PP-OCRv4) | FASTPLATEOCR | Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Line (Horizontal)** | $\ge 2.0$ | 150 crops | 34 (22.67%) | 27 (18.00%) | -4.67% |
| **Two-Line (Stacked Square)** | $< 2.0$ | 78 crops | **39 (50.00%)** | 20 (25.64%) | **-24.36% (Severe)** |

### Architectural Root Cause:
PP-OCRv4 employs `RapidOCR` with an integrated text-detection stage that isolates individual text lines and reads them top-to-bottom. FastPlateOCR's `cct-s-v2-global-model` squashes the entire input into a $128 \times 64$ tensor without line decomposition, causing top and bottom characters to bleed together into a single 10-slot sequence.

---

## 13. Performance Under Difficult Environmental Conditions

| Condition Category | Benchmark Crops | CURRENT Exact % | FASTPLATE Exact % | Leading Model |
| :--- | :---: | :---: | :---: | :---: |
| **Clean / Standard Daylight** | 38 | 31.58% (12/38) | 18.42% (7/38) | CURRENT (+13.16%) |
| **Mild Motion Blur** | 38 | 34.21% (13/38) | 21.05% (8/38) | CURRENT (+13.16%) |
| **Low-Light / Night Simulation** | 38 | 31.58% (12/38) | 21.05% (8/38) | CURRENT (+10.53%) |
| **High Contrast / Lighting** | 38 | 28.95% (11/38) | 23.68% (9/38) | CURRENT (+5.27%) |
| **JPEG Compression Artifacts** | 38 | 36.84% (14/38) | 18.42% (7/38) | CURRENT (+18.42%) |
| **Acute Skew Angle ($>15^\circ$)** | 38 | 28.95% (11/38) | 21.05% (8/38) | CURRENT (+7.90%) |

PP-OCRv4 maintains higher resilience across all simulated environmental stress conditions.

---

## 14. Speed & Computational Resource Comparison

Benchmark measured after warm-up on Intel Core x64 (2 Physical Cores / 4 Logical Threads, 7.8 GB RAM):

| Latency Metric | CURRENT (PP-OCRv4) | FASTPLATEOCR | Latency Reduction |
| :--- | :---: | :---: | :---: |
| **Mean Inference Latency** | 209.51 ms | **20.99 ms** | **9.98x faster** |
| **Median Inference Latency** | 212.26 ms | **19.86 ms** | **10.69x faster** |
| **95th Percentile (P95) Latency**| 586.28 ms | **25.87 ms** | **22.66x faster tail** |
| **Throughput (FPS on CPU)** | 4.77 plates/sec | **47.64 plates/sec** | **+42.87 plates/sec** |
| **Model Memory Footprint** | ~48.2 MB | **~5.1 MB** | **9.4x smaller footprint** |

FastPlateOCR is **immensely fast**, providing consistent sub-25ms response times on entry-level CPU hardware.

---

## 15. The 90% Requirement Assessment

| Evaluation Mode | Required Threshold | Measured Value | Formal Status |
| :--- | :---: | :---: | :---: |
| **CURRENT RAW PP-OCRv4** | $\ge 90.00\%$ | 32.02% | ❌ **FAIL** |
| **FASTPLATEOCR RAW** | $\ge 90.00\%$ | 20.61% | ❌ **FAIL** |
| **FASTPLATEOCR POST-PROCESSED**| $\ge 90.00\%$ | 18.42% | ❌ **FAIL** |

Neither model approaches the 90.0% accuracy bar in single-frame evaluation without multi-frame temporal voting.

---

## 16. Integration Compatibility Assessment

An isolated candidate adapter [`anpr_v2/fastplate_recognizer.py`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/anpr_v2/fastplate_recognizer.py) was built and verified. Its compatibility with existing components is summarized below:

- **`recognizer.py`**: Fully compatible. `FastPlateRecognizer.recognize(crop_bgr)` accepts BGR numpy arrays, auto-converts to RGB, and returns an `OCRResult` object matching the existing dataclass schema.
- **`pipeline.py` & `server.py`**: Can be swapped or toggled via config flag without altering any downstream data structures.
- **`tracker.py` (Temporal Consensus)**: Fully compatible. Character-level confidence arrays are exposed as `character_confidences`, allowing multi-frame voting.
- **`validator.py`**: Fully compatible with MoRTH Rule 50 syntax checking.
- **CPU Inference**: Fully supported via Microsoft ONNX Runtime `CPUExecutionProvider` with zero native C++ compile dependencies.

---

## 17. Engineering Recommendation

### Measured Recommendation:
👉 **FINE-TUNE FASTPLATEOCR BEFORE PRODUCTION USE**  
*(Maintain PP-OCRv4 as the primary production baseline in the interim).*

### Detailed Technical Rationale:
1. **Pretrained FastPlateOCR Causes an Immediate Accuracy Regression:**
   Deploying pretrained `cct-s-v2-global-model` today would reduce exact full-plate accuracy from **32.02% down to 20.61%** (an 11.41 percentage point drop).
2. **High Latent Potential Due to 1-Character Error Concentration:**
   FastPlateOCR already gets 1 character wrong on **32.02%** of plates and outperforms PP-OCRv4 on State (66.7% vs 53.1%), District (54.8% vs 44.3%), and Series (50.0% vs 35.5%). Its exact-match failures are overwhelmingly caused by font mismatch on the final numeric digits (`6 -> [OMIT]`, `4 -> A`, `8 -> 4`).
3. **Unmatched 10x Latency Advantage:**
   Executing in 20.9 ms on CPU (vs. 209.5 ms) with a 5.1 MB memory footprint gives FastPlateOCR the exact operational profile needed for multi-camera video streams.
4. **Actionable Next Step:**
   Fine-tune `cct-s-v2-global-model` on Indian plate crops (using `training/ocr/train_ocr.py` or synthetic Indian license plate generators). Once fine-tuned to fix digit omissions and two-row plate representations, FastPlateOCR is poised to significantly surpass PP-OCRv4 in both accuracy and speed.

---

## 18. Remaining Accuracy Gap

To achieve the 90.0% exact full-plate recognition objective:

```
Production Target Accuracy:       90.00%
Current FastPlateOCR Raw Result:   20.61%
─────────────────────────────────────────────
REMAINING ACCURACY GAP:            69.39 percentage points
```

---

## 19. Files Created During This Task

1. [`anpr_v2/fastplate_recognizer.py`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/anpr_v2/fastplate_recognizer.py): Isolated candidate adapter wrapping `fast_plate_ocr.LicensePlateRecognizer` with BGR-to-RGB conversion, ONNX Runtime CPU execution, and `OCRResult` compatibility.
2. [`benchmarks/anpr_v2/benchmark_ocr_comparison.py`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/benchmarks/anpr_v2/benchmark_ocr_comparison.py): Comprehensive automated benchmark runner evaluating both models on all 228 crops across 4 testing modes.
3. [`ocr_comparison_results/summary.json`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/ocr_comparison_results/summary.json): Complete machine-readable JSON metrics artifact.
4. [`ocr_comparison_results/comparison.csv`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/ocr_comparison_results/comparison.csv): Per-sample prediction, latency, accuracy, and win/loss comparison table.
5. [`ocr_comparison_results/errors_current.csv`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/ocr_comparison_results/errors_current.csv): Failure records for Current PP-OCRv4 (155 errors).
6. [`ocr_comparison_results/errors_fastplate.csv`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/ocr_comparison_results/errors_fastplate.csv): Failure records for FastPlateOCR (181 errors).
7. [`ocr_comparison_results/confusion_analysis.csv`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/ocr_comparison_results/confusion_analysis.csv): Character confusion frequency rankings.
8. [`FASTPLATE_OCR_COMPARISON_REPORT.md`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/FASTPLATE_OCR_COMPARISON_REPORT.md): This comprehensive benchmark report.

---

## 20. Files Modified

1. [`PROJECT_OVERVIEW.txt`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/PROJECT_OVERVIEW.txt): Updated in preceding step to reflect current platform version v2.4.

*No production vision code, detectors, validators, trackers, FastAPI endpoints, Node.js services, or frontend dashboards were modified.*

================================================================================
END OF BENCHMARK REPORT
================================================================================
