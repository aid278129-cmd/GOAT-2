# Next Recommended Actions
## Roadmap to Bridge the License Plate OCR Accuracy Gap to >90%
**Status:** Current Exact Full-Plate Recognition on Untouched Test Set is **43.42% (CCT)** and **51.32% (Adaptive Fallback)** against the **>90% statutory target**.

---

### Top 3 Root Causes Identified from Empirical Error Analysis

#### 1. Low Resolution & Small Crop Geometry (Root Cause #1)
- **Empirical Evidence:** In `final_test_predictions.csv`, 64% of missed plates (67/105) have crop heights below 28 pixels (down to 12-16px). Resizing a 16px high crop into 128x64 causes extreme bilinear blurring where distinct character strokes (e.g., '8' vs '0', '3' vs '1') merge into indistinguishable blobs.
- **Top Observed Confusions:** `8->0` (14 occurrences), `1->8` (11 occurrences), `3->1` (9 occurrences), `2->8` (9 occurrences).

#### 2. Aftermarket Non-Standard Font Geometries & Kerning (Root Cause #2)
- **Empirical Evidence:** Real Indian plates frequently exhibit localized regional fonts, italicized digits, thin-stroke characters, or handwritten numbers that deviate significantly from official DIN 1451 / MoRTH mandated fonts.
- **Top Observed Confusions:** `P->A` (13 occurrences), `6->4` (12 occurrences), `4->A` (11 occurrences), `L->4` (10 occurrences). While synthetic data solved the global digit 4/6 omission problem, non-standard thin fonts still trigger confusions.

#### 3. Real-World Road Clutter, Mud, & Fastener Interference (Root Cause #3)
- **Empirical Evidence:** 23.8% of failed benchmark test images feature physical black bolt screws, mounting rivets, yellow decorative stickers, or dirt smudges placed directly through characters. For instance, a screw through a '1' transforms its bounding contours into an '8' or 'B'.

---

### Minimal Evidence-Based Action Plan to Reach >90%

#### Action 1: Super-Resolution / Bicubic Edge Sharpening for Low-Resolution Crops
- **Why:** 67 failed images suffered from stroke-merging due to low resolution.
- **Action:** Introduce an ultra-lightweight ESRGAN / Real-ESRGAN or unsharp masking pre-filter for crops with $h < 32\text{px}$ before feeding into the 128x64 CCT input.
- **Estimated Gain:** +8.0% to +12.0% exact match recovery on small distant crops.

#### Action 2: Hard-Negative Font Mining & Semi-Supervised Real Indian Plate Fine-Tuning
- **Why:** The current fine-tuned checkpoint was trained on 4,000 synthetic crops + 500 val crops using standard TTF fonts (`arialbd.ttf`). It needs exposure to the distribution of real road imagery.
- **Action:** Fine-tune CCT-S-v2 on 15,000–25,000 real Indian road crops from open repositories (e.g., Indian Driving Dataset - IDD, Datacluster Indian License Plates, Kaggle Indian Number Plates) using pseudo-labeling with PP-OCRv4 consensus.
- **Estimated Gain:** +18.0% to +24.0% exact match on unconstrained road conditions.

#### Action 3: Tight Margin Detector Crop Inflation & Fastener Inpainting
- **Why:** In several benchmark crops, the YOLO detector crop sliced within 1-2 pixels of the first or last character, causing edge characters (e.g., State prefix 'M' or last registration digit '4') to be cut off.
- **Action:** Add a dynamic 8% padding margin on all detector bounding box crops before perspective rectification. Implement morphological black-hat filtering to suppress high-contrast screw artifacts.
- **Estimated Gain:** +5.0% to +7.0% exact match improvement.

#### Action 4: ONNX Export with INT8 Quantization for Edge Deployment
- **Why:** The PyTorch fine-tuned model currently runs in 168 ms on CPU. Converting `cct_s_v2_indian_best.keras` into an optimized ONNX graph will reduce inference latency to ~15-25 ms, enabling 40+ FPS real-time throughput.
- **Action:** Export model via `torch.onnx.export()` with ONNX Runtime optimizations.

---

### Summary Projection

| Pipeline Milestone | Raw Exact Match | Fallback Exact Match | Status |
| :--- | :--- | :--- | :--- |
| **Initial Pretrained Baseline** | 20.61% | 32.02% | Historical Baseline |
| **Current Verified State (Phase 10-14)** | **43.42%** | **51.32%** | **Current State** |
| **With Action 1 (Super-Resolution Pre-Filter)** | ~54.0% | ~61.0% | Next Step |
| **With Action 2 (Real Dataset Fine-Tuning)** | ~78.0% | ~84.0% | Planned Milestone |
| **With Action 3 (8% Crop Padding + Inpainting)** | **~88.0%** | **>92.0%** | **Target Met (>90%)** |
