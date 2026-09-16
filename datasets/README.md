# Indian License Plate Datasets & Partitioning Protocol

**Project:** City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban Traffic Analytics  
**SIH Problem Statement ID:** 26127 (Bharat Electronics Limited - BEL)  
**Standard:** Ministry of Road Transport and Highways (MoRTH Rule 50) & Bharat Series (BH)

---

## 1. Legal Usability & Public Source Documentation

In strict accordance with Phase 3 directives, only legally accessible public research datasets and self-collected surveillance feeds are utilized.

| Dataset Name | Source Repository | Public License | Source Image Count | Annotation Format | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DataCluster Indian License Plates** | [DataCluster Labs](https://github.com/datacluster-labs/Indian-Number-Plates-Dataset) | CC BY-NC 4.0 / Public Research | 27 Full Vehicle Images | Pascal VOC XML (BBox + Plate Text) | Real-world Indian traffic scenes across diverse states (cars, tempos, trucks, autos). |
| **VOC Indian Vehicle LPR** | Kaggle Indian License Plates Dataset | Open Public / Kaggle Research | 20 Full Vehicle Images | Pascal VOC XML + BBoxes | High-resolution multi-angle Indian vehicle photos with verified ground-truth strings. |
| **YOLOv8 Prepared Plates** | Local Unified Partition (`dataset/`) | Project Internal | 47 Pre-processed Images | YOLO normalized `[class x_center y_center width height]` | Pre-converted bounding box dataset for detector training. |
| **Real Live Surveillance Crops** | Local WebRTC Camera Engine (`debug_output/crops/`) | Internal Hackathon Feeds | 1,877 Plate Crops | Image Crop (`.jpg`) | Real-world live streaming camera captures across multiple angles, illumination levels, and distances. |
| **Held-out Regression Test Case** | Reserved Benchmarking Suite | Internal Test Ground Truth | 1 High-Precision Sample | Image Crop (`MH01AV8669`) | Critical negative/positive regression benchmark verifying anti-hallucination policy. |

---

## 2. Unified Dataset Schema & Format

### A. Detection Dataset (Full Frame $\rightarrow$ Bounding Box)
Each detection sample consists of:
- **Image:** Full vehicle frame (`.jpg` or `.png`).
- **Annotation:** Bounding box for class `0` (`number_plate`):
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```
  Normalized relative to image width and height in range `[0.0, 1.0]`.

### B. Recognition Dataset (Plate Crop $\rightarrow$ Ground-Truth Text)
Each recognition sample consists of:
- **Image:** Rectified license plate crop (`.jpg`).
- **Target:** Exact alphanumeric registration string conforming to Indian state standards (e.g., `MH01AV8669`, `DL3CD1210`, `KL35F4337`, `UP84AE9889`).

---

## 3. Dataset Splits & Leakage Prevention Protocol

The unified dataset is split strictly by vehicle identity to prevent train/test contamination:

```
Total Vehicle Samples
         │
  ┌──────┴─────────────────────────┐
  │                                │
80% Train Split              20% Held-Out
(Training & Augmentation)          │
                            ┌──────┴──────┐
                            │             │
                       10% Validation  10% Test Benchmark
                       (Early Stop)    (Unseen Ground Truth)
```

### Strict Anti-Leakage Rules
1. **Vehicle Identity Partitioning:** Multiple camera frames or sequential crops of the same vehicle/plate MUST remain in the same partition.
2. **Zero Test-Set Contamination:** Images in the `test` split must NEVER be viewed by the detector or recognizer during training or hyperparameter tuning.
3. **Regression Immunity:** The known regression sample `MH01AV8669` is completely isolated from all training pipelines to prevent overfitting on benchmark questions.
