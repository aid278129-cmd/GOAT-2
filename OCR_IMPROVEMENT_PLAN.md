# OCR / ANPR Targeted Improvement Plan

## 1. Executive Summary

The Phase 4 untouched validation benchmark established the following ground-truth performance across 300 test samples (0% train/val overlap):

| Metric | Measured Baseline | Requirement Target | Status |
| :--- | :--- | :--- | :--- |
| **Plate Detector Precision** | **100.00%** | > 95% | **EXCEEDED** |
| **Plate Detector Recall** | **84.00%** | > 90% | **GAP (-6.0%)** |
| **OCR Exact Match (Detected Only)** | **84.52%** | > 90% | **GAP (-5.48%)** |
| **OCR Character Accuracy** | **80.07%** | > 90% | **GAP (-9.93%)** |
| **End-to-End ANPR Exact Match** | **71.00%** | > 90% | **NOT YET ACHIEVED (-19.0%)** |
| **High Resolution Exact Match** | **81.78%** | > 80% | **EXCEEDED** |
| **Two-Line / Square Plate Match** | **98.89%** | > 90% | **OUTSTANDING** |
| **Single-Line Plate Match** | **59.05%** | > 90% | **MAJOR GAP (-30.95%)** |
| **Low Resolution ($h < 60\text{px}$)** | **4.76%** | > 90% | **CRITICAL FAILURE (-85.24%)** |

> [!IMPORTANT]
> Two-line / square plate recognition is already operating at state-of-the-art levels (**98.89% exact match, 99.11% character accuracy**). All improvements must strictly preserve two-line feature representations.

---

## 2. Root Cause Investigation

### 2.1 Low-Resolution Breakdown ($h < 60\text{px}$)
Evaluation of the 42 low-resolution test samples ($40 \times 175\text{px}$ crops) revealed:
* **Detection Failure (Plate not detected):** **42 / 42 (100.0%)**
* **OCR Recognition Failure:** **0 / 42 (0.0%)** (OCR was never invoked because detector returned 0 bounding boxes)
* **Root Cause:** The YOLO plate detector (`indian_plate_best.onnx`) downsamples inputs through stride 32. For a $40\text{px}$ high plate, the feature map representation is $\approx 1.25$ grid cells, falling below the detection anchor / box regression threshold ($conf < 0.25$).

### 2.2 Single-Line Plate Breakdown (Aspect Ratio > 2.5)
Evaluation of the 210 single-line standard plates revealed:
* **Exact Match:** 59.05%
* **Character Accuracy:** 71.90%
* **Dominant Confusion Pairs:**
  * `9` $\leftrightarrow$ `0` (Rounded loop misinterpretation in low contrast)
  * `3` $\leftrightarrow$ `0` (Middle crossbar dropout)
  * `B` $\leftrightarrow$ `8` (Loop fusion)
  * `V` $\leftrightarrow$ `Y` (Stem truncation)
  * `7` $\leftrightarrow$ `6` / `1` $\leftrightarrow$ `I`
* **Root Cause:** Standard Indian single-line plates have tightly spaced 10-character sequences (`SS NN AA NNNN`). Character segmentation in the compact CRNN/CCT model struggles when character aspect ratios deviate slightly without positional syntax masking.

---

## 3. Evidence-Ranked Improvement Roadmap (No Training Yet)

Ranked strictly by expected accuracy gain versus implementation risk:

### Priority 1: Multi-Scale / Super-Resolution Inference for Small Plates
* **Method:** Apply Test-Time Augmentation (TTA) with $1.5\times$ bicubic upscaling and unsharp masking for frames where no plate is found on the primary pass.
* **Target:** Resolve the 100% detection drop on $h < 60\text{px}$ inputs without modifying model weights.
* **Expected Gain:** $+10\%$ to $+14\%$ overall detector recall.

### Priority 2: Indian MoRTH Syntax-Aware Positional Post-Processor
* **Method:** Indian license plates follow a rigid standard:
  $$\underbrace{\text{State}}_{\text{Positions 1-2: Alpha}} \quad \underbrace{\text{RTO}}_{\text{Positions 3-4: Numeric}} \quad \underbrace{\text{Series}}_{\text{Positions 5-6: Alpha}} \quad \underbrace{\text{Number}}_{\text{Positions 7-10: Numeric}}$$
* **Action:** When character confusion occurs at position 3-4 (e.g. `O` vs `0`), syntactically enforce numeric digit. When confusion occurs at position 1-2, syntactically enforce state code.
* **Expected Gain:** $+8\%$ to $+12\%$ exact match on single-line plates with zero risk of regression.

### Priority 3: Multi-Frame Temporal Consensus Voting
* **Method:** Real camera streams deliver 15–30 FPS. Aggregate 3–5 consecutive frames using per-character probability voting.
* **Target:** Transient motion blur or single-frame character dropouts are corrected by adjacent frames.
* **Expected Gain:** $+10\%$ to $+15\%$ in live multi-camera deployment.

### Priority 4: Targeted Single-Line Fine-Tuning (Deferred)
* **Method:** Fine-tune only the final dense layers of `cct_s_v2_real_best.keras` on single-line crops with synthetic character spacing augmentations. Freeze the vertical line segmentation module to protect the 98.89% two-line accuracy.
