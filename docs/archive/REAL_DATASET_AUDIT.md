# REAL DATASET AUDIT & PROVENANCE REPORT

## 1. Primary Dataset Verification
- **Dataset Title**: Indian License Plate Images
- **Kaggle Identifier**: `umar1103/final-licence`
- **Creator**: Umar Farooq (`umar1103`)
- **Dataset URL**: https://www.kaggle.com/datasets/umar1103/final-licence
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Download Date**: September 26, 2026
- **Total Files**: 30,450 images
- **Total Download Size**: 590 MB (618,724,352 bytes)
- **File Format**: 100% PNG images (RGB / 8-bit palette)

## 2. Dataset Structure & Ground Truth Verification
- **Label Encoding**: Plate ground truth is encoded directly into filenames (e.g., `KA 61 MH 5551.png`, `AA06YO1523.png`, `GJ32AP8867.png`).
- **Filename Normalization**: Spaces, hyphens, and extension artifacts stripped; converted to uppercase alphanumeric strings `[A-Z0-9]`.
- **Label Validation Results**:
  - **Valid Indian Registrations**: 26,455 images (86.88%) conform strictly to Indian MoRTH standards (State code, RTO district code, series, registration number) or recognized BH-series formats.
  - **Rejected Labels**: 3,995 images (13.12%) rejected due to invalid syntax (e.g., random synthetic generator strings such as `0079RMZTR8.png`, non-Indian alphabets, or malformed lengths). Logged completely in `rejected_labels.csv`.
- **Test-Set Leakage Verification**:
  - Exact SHA-256 and perceptual pHash cross-matching performed against the frozen 228 held-out test crops.
  - **Zero Leakage**: Exactly 0 overlapping or near-duplicate images detected between `umar1103/final-licence` and the frozen 228 test crops. Logged in `dataset_leakage_report.csv`.

## 3. Resolution & Layout Cohort Analysis
The primary dataset exhibits distinct resolution and aspect ratio (AR = width / height) clusters:
1. **Stacked Two-Line Plates** (`AR < 2.3`, predominantly 512x256 and 200x100):
   - ~10,380 images (34.1%).
   - True stacked Indian plates (e.g., two-wheelers, auto-rickshaws, commercial trucks with state/RTO on top line and series/digits on bottom line).
2. **Surveillance Low-Resolution Real Crops** (`175x40`, `AR ≈ 4.38`, height <= 48px):
   - ~10,050 images (33.0%).
   - Authentic low-resolution crops extracted from surveillance cameras displaying real-world sensor noise, slight blur, and compression artifacts.
3. **Standard Single-Line Plates** (`512x128`, `AR ≈ 4.00`):
   - ~6,025 images (19.8%).
   - Single-line passenger car and commercial vehicle plates with varied real backgrounds and lighting.
4. **Synthetic Non-Standard Images** (`200x100`, `AR = 2.00`):
   - ~4,000 images (13.1%).
   - Clean synthetic plates with arbitrary letter strings; filtered out during label validation.

## 4. Visual Condition & Edge Case Assessment
- **Single-Line Plates**: Well-represented across multiple states (DL, MH, KA, TN, UP, GJ, HR, WB, etc.).
- **Two-Line Plates**: Abundantly present (~34%), addressing the exact architectural failure of the previous model.
- **Lighting Conditions**: Daylight, evening, headlight glare, low-contrast shadows.
- **Perspective Distortion**: Variable angles (-15° to +20° skew) from real camera mountings.
- **Plate Backgrounds**: Both white (private passenger vehicles) and yellow (commercial taxis, trucks, autos).
- **Dirt & Degradation**: Mild road grime, edge clipping, and natural camera motion blur present in low-res cohort.

## 5. Secondary Dataset Audit & Decision
### A. `kedarsai/indian-license-plates-with-labels`
- **Identifier**: `kedarsai/indian-license-plates-with-labels`
- **Audit Findings**: Contains image files accompanied by YOLO `.txt` annotation files.
- **Label Content**: Annotations provide only bounding boxes (`<class> <x_center> <y_center> <width> <height>`).
- **OCR Text**: Zero text transcriptions provided in filenames or metadata.
- **Decision**: **REJECTED** per Master Rule ("If a dataset only provides x, y, width, height, then it is detection data, NOT directly usable recognition data").

### B. `Indian vehicle license plate dataset (~1.5k)`
- **Audit Findings**: Unverified licensing and inconsistent ground truth transcriptions.
- **Decision**: **SKIPPED** to maintain strict CC BY 4.0 licensing compliance and avoid data contamination.

## 6. Curated Dataset Composition (5,000 Samples)
To prevent model degradation and reduce the synthetic domain gap:
- **Curated Real Two-Line Cohort**: 1,400 samples
- **Curated Real Low-Res Surveillance Crops (<=48px)**: 1,400 samples
- **Curated Real Single-Line Cohort**: 1,200 samples
- **Targeted Synthetic Edge Supplement**: 1,000 samples (specifically targeted for underrepresented digits `4` and `6`, rare characters, extreme blur, and perspective)
- **Total Curated Pool**: 5,000 balanced samples
- **Train/Val Split**: Deterministic 85% Train (~4,250) / 15% Validation (~750), grouped by plate string to guarantee zero train-to-val leakage.
- **Frozen Test Set**: Exactly 228 untouched real benchmark plates (100% isolated).
