# Tesseract OCR Integration & Custom LSTM Training Roadmap
**Project:** Indian Vehicle License Plate Detection & Recognition (ANPR)  
**Model Architecture:** YOLOv8 (Plate Detector) + Tesseract OCR v5.4.0 (Text Recognizer)  
**Repository Reference:** [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

---

## 1. Executive Summary

This document establishes the architecture, active implementation, and future roadmap for training and deploying **Tesseract OCR** as the primary optical character recognition engine for Indian vehicle number plates.

### Key Highlights
- **Active Implementation (Completed):**
  - Official **Tesseract v5.4.0** engine installed and configured (`C:\Program Files\Tesseract-OCR\tesseract.exe`).
  - Integrated directly with `pytesseract` in `scripts/anpr_server.py`.
  - Multi-variant license plate binarization pipeline (Bilateral Filter + CLAHE + Otsu Binary + Inverted Otsu + Adaptive Gaussian).
  - Constrained Page Segmentation Modes (`--psm 7`, `--psm 8`, `--psm 6`) and alphanumeric whitelist (`ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`).
  - Real-time Indian registration number parsing and syntax validation (State Codes: `DL`, `MH`, `KA`, `UP`, `WB`, `KL`, `TN`, `GJ`, `BH-Series`).
  - Automated evaluation harness (`scripts/evaluate_tesseract_ocr.py`) benchmarking against real-world ground-truth annotated plates.

- **Future Roadmap (Implementation for Later):**
  - Retraining and fine-tuning the Tesseract LSTM neural network (`lstmtraining`) specifically on Indian license plate fonts (`FE-Schrift`, `Mandatory`, `DIN 1451`).
  - Automated synthetic plate generation using `text2image`.
  - Packaging the resulting model as `ind_plate.traineddata`.

---

## 2. Active Implementation Details

### 2.1 Preprocessing Pipeline for License Plates
License plate images captured from moving vehicles or surveillance cameras frequently suffer from noise, uneven illumination, shadows, and angle distortions. Tesseract OCR delivers optimal character recognition when fed properly binarized images:

1. **Aspect Ratio & Resolution Normalization:**
   - Minimum height scaled to 100px using bicubic interpolation (`cv2.INTER_CUBIC`) to preserve thin character strokes.
2. **Edge-Preserving Noise Reduction:**
   - `cv2.bilateralFilter(gray, 9, 75, 75)` removes camera sensor noise while maintaining sharp character edges.
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE):**
   - `clipLimit=3.0`, `tileGridSize=(8, 8)` to balance headlights glare or shadow gradients across the plate.
4. **Dual Binarization (Otsu + Otsu Invert):**
   - **Otsu Normal:** White plate background with dark characters (standard private vehicles).
   - **Otsu Inverted:** Black/dark plate background with yellow/white characters (commercial, EV, or military plates).
5. **Adaptive Gaussian Thresholding:**
   - Fallback binarization for severely non-uniform lighting conditions.

### 2.2 Tesseract Engine Configuration
```python
# Tesseract License Plate Extraction Config
whitelist = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Single-line plates (e.g., DL 3C D 1210)
cfg_line = f"--psm 7 {whitelist}"

# Single-word plates
cfg_word = f"--psm 8 {whitelist}"

# Multi-line stacked plates (e.g., two-wheelers, trucks, auto-rickshaws)
cfg_multiline = f"--psm 6 {whitelist}"
```

---

## 3. Future Roadmap: Custom Tesseract LSTM Retraining

To push accuracy to 99%+ on low-resolution, damaged, or customized font plates, the Tesseract LSTM neural network can be fine-tuned on custom plate characters.

### 3.1 Prerequisite Requirements
1. **Tesseract Floating-Point Base Model:**
   - The default `eng.traineddata` included with standard distributions is an integer quantized model (`tessdata_fast`), which cannot be trained directly.
   - Retraining requires the unquantized float model from `tessdata_best`:
     ```bash
     curl -L -o tessdata_best/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
     ```
2. **Tesseract Training Binaries:**
   - `combine_tessdata.exe`
   - `text2image.exe`
   - `lstmtraining.exe`
   - `lstmeval.exe`

### 3.2 Step-by-Step Retraining Workflow

#### Step A: Extract Existing LSTM Model
Extract the LSTM layer from the base model:
```bash
combine_tessdata -e tessdata_best/eng.traineddata training_dir/eng.lstm
```

#### Step B: Generate Synthetic Indian Plate Dataset
Generate training line images and box files using Indian vehicle plate fonts (`FE-Schrift`, `Mandatory`, `Arial Bold`):
```bash
text2image \
  --text=training_texts/indian_plates.txt \
  --outputbase=training_data/eng.ind_font.exp0 \
  --font="FE-Schrift" \
  --fonts_dir=fonts/ \
  --ptsize=36 \
  --margin=12
```

#### Step C: Generate `.lstmf` Training Files
Compile the image and box files into Tesseract LSTM format:
```bash
tesseract training_data/eng.ind_font.exp0.tif training_data/eng.ind_font.exp0 --psm 6 lstm.train
```

#### Step D: Run `lstmtraining`
Fine-tune the neural network weights on the dataset:
```bash
lstmtraining \
  --continue_from training_dir/eng.lstm \
  --traineddata tessdata_best/eng.traineddata \
  --train_listfile training_data/train_list.txt \
  --eval_listfile training_data/val_list.txt \
  --model_output models/ind_plate_checkpoints/ind_plate \
  --max_iterations 5000 \
  --learning_rate 0.001 \
  --target_error_rate 0.01
```

#### Step E: Evaluate Model Performance
Validate the trained checkpoint using `lstmeval`:
```bash
lstmeval \
  --model models/ind_plate_checkpoints/ind_plate_checkpoint \
  --traineddata tessdata_best/eng.traineddata \
  --eval_listfile training_data/val_list.txt
```

#### Step F: Package Final `.traineddata`
Convert the checkpoint into an optimized, deployable runtime model:
```bash
lstmtraining \
  --stop_training \
  --continue_from models/ind_plate_checkpoints/ind_plate_checkpoint \
  --traineddata tessdata_best/eng.traineddata \
  --convert_to_int \
  --model_output models/ind_plate.traineddata
```

#### Step G: Deploy in ANPR Inference Server
In `scripts/anpr_server.py`, configure:
```python
custom_tessdata_dir = os.path.join(BASE_DIR, "models")
cfg = f"--tessdata-dir '{custom_tessdata_dir}' -l ind_plate --psm 7 {whitelist}"
```

---

## 4. Verification & Testing Instructions

### Run Evaluation Benchmark
```powershell
python scripts/evaluate_tesseract_ocr.py
```

### Test Single Image with Tesseract OCR
```powershell
python scripts/test_inference.py
```

### Run Live ANPR Server
```powershell
python scripts/anpr_server.py
```
