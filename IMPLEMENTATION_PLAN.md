# Incremental Model & Live ANPR Optimization Plan

This plan addresses the engineering directives in [solution.txt](file:///d:/college%20work/Hackaton%20projects/new%20zyn/solution.txt) using the 180-frame live localhost benchmark as the baseline truth. As directed, execution is strictly one change at a time with empirical benchmarking after each step.

---

## Baseline Metrics (180-Frame Live Localhost Source of Truth)

- **Total frames dispatched**: 180
- **Vehicle detected**: 130 (72.2%)
- **Plate candidates located**: 117 (65.0%)
- **Geometry-valid candidates**: 36 (30.8% of candidates)
- **Final confirmed plates**: 24 (13.3% of total frames)
- **Fast path latency (CRNN Tier 0)**: ~310–460 ms
- **Slow path latency (OCR Fallback to Tesseract/EasyOCR)**: ~1,460–2,500 ms (up to 2,178 ms in OCR alone)
- **CRNN raw inference latency**: ~9.5 ms
- **Rejection breakdown**:
  - `REJECTED_BELOW_PLATE_CONF`: 81 frames
  - `REJECTED_NO_VEHICLE`: 50 frames
  - `NO_PLATE_CANDIDATES`: 13 frames
  - `REJECTED_LENGTH`: 9 frames

---

## Operating Constraints & Safeguards

- **NO major feature additions**.
- **NO Vehicle Re-ID**.
- **NO YOLO plate detector retraining yet**.
- **NO modifications to GIS, WebRTC, watchlist, or analytics**.
- **Preserve zero false-positive safeguards**: Low-confidence YOLO candidates must still pass vehicle evidence, geometry, Indian syntax, OCR confidence, and temporal confirmation before becoming a final detection.

---

## Step-by-Step Implementation Tasks

### Task 1: Adaptive Live Plate Confidence Threshold
**Goal**: Keep static/direct-image plate threshold at `0.40`. For live WebRTC streams only, evaluate `0.35`, `0.32`, `0.30`, and `0.28`. Pick the lowest-risk threshold producing the best verified result.

- **Files to Modify**: `scripts/anpr_server.py`
- **Implementation**:
  - Add `live_plate_conf_threshold` to `ANPR_CONFIG`.
  - Condition `min_plate_conf` on `req.manualScan`:
    ```python
    min_plate_conf = ANPR_CONFIG["plate_conf_threshold"] if req.manualScan else ANPR_CONFIG["live_plate_conf_threshold"]
    ```
  - Expose parameter in `/config` API and dashboard UI.
  - Create test script `scripts/benchmark_live_thresholds.py` to benchmark `[0.40, 0.35, 0.32, 0.30, 0.28]`.
  - Measure: candidate recall, geometry-valid count, confirmed count, false positives, latency.

---

### Task 2: Investigate Geometry Bottleneck
**Goal**: Analyze why 117 candidates produced only 36 geometry-valid crops without loosening geometry rules blindly.

- **Files to Modify**: `scripts/anpr_server.py`
- **Implementation**:
  - Instrument distinct rejection codes:
    - `REJECT_ASPECT_RATIO`
    - `REJECT_MIN_WIDTH`
    - `REJECT_MIN_HEIGHT`
    - `REJECT_AREA_RATIO`
    - `REJECT_VERTICAL_POSITION`
    - `REJECT_OTHER_GEOMETRY`
  - Save representative rejected plate crops into `debug_output/geometry_rejected/`:
    `{frame_id}_{candidate_idx}_{rejection_code}_ar{ar:.2f}_sz{w}x{h}.jpg`
  - Track counts and percentages in `DEBUG_STATS["geometry_rejections_breakdown"]` exposed via `/debug/stats`.
  - Determine whether genuine Indian plates are being discarded before modifying any geometry threshold.

---

### Task 3: Hard Limit Fallback OCR (350 ms Budget)
**Goal**: Eliminate the 1.5–2.5 second fallback OCR latency spikes on live camera feeds.

- **Files to Modify**: `scripts/anpr_server.py`
- **Implementation**:
  - Impose a total fallback OCR budget of 350 ms for live video (`not req.manualScan`):
    1. CRNN inference (~9.5 ms) + Probabilistic Positional Decoder.
    2. If syntax is valid and confident: **STOP immediately**.
    3. If invalid/ambiguous: check remaining budget:
       $$\text{remaining\_budget} = 350\text{ ms} - (\text{elapsed\_ms})$$
    4. If budget $\le 0$: **ABORT fallback OCR for that frame** (`ABORTED_FALLBACK_TIMEOUT`). Do not stall the pipeline; allow the next keyframe to detect.
    5. If budget $> 0$: call Tesseract with a strict timeout limit. If budget depletes, skip EasyOCR.
  - Manual/static uploads retain full-depth OCR.

---

### Task 4: OCR Exact-Match Improvement
**Goal**: Optimize for **Full-Plate Exact-Match Accuracy** and systematically eliminate confusion pairs: $B/8, O/0, D/0, I/1, T/1, S/5, Z/2$.

- **Files to Modify**: `scripts/crnn_ocr.py`
- **Implementation**:
  - Extract top-K character logits / beam candidates from CRNN softmax output.
  - Score candidates using:
    - Character probability
    - Indian slot positional legality (State, District, Series, Serial)
    - MoRTH syntax legality
  - Build confusion matrix across real vehicle plate crops.
  - Mine incorrectly recognized real crops from `debug_output/crops/` into the fine-tuning training set (never train on benchmark/test sets).

---

### Task 5: Comprehensive Re-Benchmark of Live Pipeline
**Goal**: Run a complete $\ge 180$-frame live benchmark to verify compound performance.

- **Files to Create**: `scripts/benchmark_live_optimization.py`
- **Metrics to Compare (Before vs. After)**:
  - Vehicle detection conversion %
  - Plate candidate conversion %
  - Geometry pass rate %
  - OCR syntax pass rate %
  - Final confirmation rate %
  - Median, P95, and worst-case E2E latency
  - CRNN-only % vs. Fallback OCR % vs. Fallback timeout %
  - Full-plate exact-match accuracy
  - False positive count & incorrect confirmations

---

## Verification Commands

```bash
# 1. Verify granular geometry logging and directory creation
python -c "import anpr_server; print('Ready for Task 1 & 2')"

# 2. Test live plate confidence threshold sweep
python scripts/benchmark_live_thresholds.py

# 3. Test fallback timeout budget enforcement
python -c "import anpr_server; print('Ready for Task 3')"

# 4. Evaluate CRNN exact-match accuracy
python scripts/evaluate_crnn_exact_match.py

# 5. Run full comparative live benchmark
python scripts/benchmark_live_optimization.py
```
