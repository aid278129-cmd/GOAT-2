# Real-World Benchmark OCR Failure Analysis
## Diagnostic Categorization of 129 Test Failures (Current 43.42% Model)
**Evaluation Dataset:** Untouched 228 Indian Plate Benchmark Crops (`benchmarks/anpr_v2/crops/`)  
**Evaluated Model:** Indian Fine-Tuned CCT-S-v2 (`models/indian_cct/cct_s_v2_indian_best.keras`)  
**Baseline Test Exact Accuracy:** 43.42% (99 exact matches / 228 test crops)  
**Total Failures Analyzed:** 129 failed crops (Strict Diagnostic Categorization Only — Zero Ground-Truth Images Used in Training)  

---

### 1. Executive Failure Summary

The primary cause of the gap between 99.0% synthetic validation accuracy and 43.42% real-world benchmark performance is **domain mismatch**:
1. Synthetic images feature clean, high-contrast, uniformly kerned glyphs with consistent lighting.
2. Real-world surveillance crops exhibit non-standard aftermarket fonts, acute camera angles, extreme aspect ratio variances, dirt, bolt fasteners, and heavy motion blur.

```
+-----------------------------------------------------------------------+
| Category               | Count | % of Failures | Primary Failure Mode |
+------------------------+-------+---------------+----------------------+
| CHARACTER OMISSION     | 90    | 69.8%         | Missing 4, 6, suffix |
| TWO-LINE STACKED       | 54    | 41.9%         | 2-row line spacing   |
| CHARACTER SUBSTITUTION | 38    | 29.5%         | 8/0, P/A, 6/4, 1/8   |
| BLUR (Motion / Focus)  | 21    | 16.3%         | Stroke boundary loss |
| PERSPECTIVE SKEW       | 21    | 16.3%         | Slanted letter legs  |
| GLARE / HIGH CONTRAST  | 19    | 14.7%         | Specular reflections |
| LOW RESOLUTION         | 7     | 5.4%          | Crop height < 28px   |
| CHARACTER INSERTION    | 1     | 0.8%          | Hallucinated glyph   |
+-----------------------------------------------------------------------+
*(Note: Percentages sum to > 100% as real-world crops exhibit overlapping compound failure modes).*
```

---

### 2. Granular Failure Mode Analysis

#### A. Character Omission (90 failures, 69.8%)
- **Observation:** In 90 out of 129 failures, the predicted string was shorter than the ground truth.
- **Root Cause:**
  1. The global CCT-S-v2 backbone's positional embedding has an inductive bias towards single-row plates with 7–8 characters.
  2. For 10-character Indian plates (e.g., `MH12AB1234`), the model frequently drops the final 1 or 2 digits (e.g., predicting `MH12AB12` instead of `MH12AB1234`).
  3. Digits `4` and `6` in narrow or italicized fonts are occasionally treated as background padding rather than foreground characters.
- **Required Fix:** Fine-tuning on diverse real-world crops with explicit 9–10 character Indian plate structures and heavy penalization for premature sequence termination.

#### B. Two-Line Layout Failures (54 failures, 41.9%)
- **Observation:** While our Phase 1 model improved from 0% to 50.0% on two-line plates, 54 two-line plates still failed.
- **Root Cause:**
  1. In stacked motorcycle plates, the distance between the top row (`TN45`) and bottom row (`AB1234`) varies wildly across different vehicle manufacturers and custom tail-brackets.
  2. When the vertical gap is large, the transformer fails to serialize the reading order from top-to-bottom.
- **Required Fix:** Curation of diverse real two-line plates in the training set covering varied inter-row vertical spacing and aspect ratios.

#### C. Character Substitution & Font Variance (38 failures, 29.5%)
- **Observation:** Exact character substitutions occur in 38 cases.
- **Top Observed Confusion Pairs:**
  - `8 -> 0`: Inner loop closure in low-light and compressed images (14 cases).
  - `P -> A`: Horizontal crossbar and diagonal leg ambiguity in square plates (13 cases).
  - `6 -> 4`: Open loop geometry in thin aftermarket fonts (12 cases).
  - `4 -> A`: Triangular apex resemblance under acute perspective (11 cases).
  - `1 -> 8`: Black mounting bolts placed directly adjacent to or through digit '1' (11 cases).
  - `L -> 4`: Bottom-row corner stroke ambiguity (10 cases).
- **Required Fix:** Incorporating real-world plate images with diverse fonts, mounting screws, and contrast variations into the training pipeline.

#### D. Environmental Degradations (Blur: 21, Perspective: 21, Glare: 19)
- **Observation:** 61 failures involved environmental conditions (blur, acute skew, or specular headlight/sun glare).
- **Required Fix:** Probabilistic augmentation imitating realistic surveillance camera noise, including downsampling to 24–40px, Gaussian blur, and perspective tilt.

---

### 3. Implications for Phase 2 Training Data Curation

1. **Prioritize Real-World Diversity:** Training on synthetic fonts alone cannot teach the model the nuances of regional Indian plate fonts, custom embossing, and road dirt.
2. **Balanced Composition:** Target 4,000–6,000 curated samples with ~3,500–4,500 real/real-like plates + 500–1,500 targeted synthetic samples.
3. **Explicit 2-Row Inclusion:** Ensure the training set contains at least 25%–35% genuine two-line plates.
4. **Hard-Negative Character Mining:** Ensure heavy representation of digits `0, 1, 2, 4, 6, 8` and letters `A, P, L, B, D`.
