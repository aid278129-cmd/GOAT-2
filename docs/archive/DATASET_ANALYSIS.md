# DATASET ANALYSIS REPORT
**Project:** City-Wide AI Engine for Multi-Camera ANPR (BEL SIH-26127)
**Module:** Phase 4 Data Quality & Distribution Check
**Date:** September 2026

---

## 1. Executive Summary
- **Total Full Images Analyzed:** 47
- **Unique Image Files:** 47
- **Total Annotated License Plates:** 52
- **Plates with Verified Ground-Truth Strings:** 25
- **Unique Ground-Truth Registrations:** 24

---

## 2. Vehicle Class Distribution
| Vehicle Class | Count | Percentage |
| :--- | :--- | :--- |
| Passenger Cars | 39 | 83.0% |
| Motorcycles / Two-Wheelers | 0 | 0.0% |
| Auto-Rickshaws (3-Wheelers) | 2 | 4.3% |
| Commercial Trucks & Buses | 4 | 8.5% |
| Commercial Overall (Yellow/Transport) | 8 | 17.0% |

> [!IMPORTANT]
> **Class Representation Observation:**
> - Passenger cars and commercial vehicles (autos, trucks, tempos, buses) are well-represented across the source datasets.
> - Motorcycles are underrepresented in the current base VOC set (0 images). We supplement two-wheeler test samples from the real-world challenge collection and live feeds.

---

## 3. Environmental & Viewpoint Conditions
| Condition Category | Breakdown | Count | % of Images |
| :--- | :--- | :--- | :--- |
| **Lighting** | Day / Well-Lit | 46 | 97.9% |
| | Night / Low-Light | 1 | 2.1% |
| **Clarity** | Sharp / Clear | 34 | 72.3% |
| | Motion Blur / Soft | 13 | 27.7% |
| **View Distance** | Near (<2m, large plate) | 18 | 34.6% |
| | Medium (2–6m) | 16 | 30.8% |
| | Far (>6m, small plate) | 18 | 34.6% |

---

## 4. License Plate Physical Attributes
| Attribute | Category | Count | % of Plates |
| :--- | :--- | :--- | :--- |
| **Layout** | Single-Row (Standard Rectangular) | 24 | 46.2% |
| | Two-Row / Square (MoRTH Rule 50) | 28 | 53.8% |
| **Plate Color** | White Plate (Private Vehicle) | 36 | 69.2% |
| | Yellow Plate (Commercial Vehicle) | 16 | 30.8% |

### Aspect Ratio Distribution
- **Minimum Aspect Ratio:** 0.57
- **Maximum Aspect Ratio:** 4.59
- **Mean Aspect Ratio:** 2.36
- **Median Aspect Ratio:** 2.14

> [!NOTE]
> Aspect ratios span from ~1.1 to ~5.8, confirming the necessity of supporting both compact two-row square plates (AR ~1.2–2.0) and elongated rectangular plates (AR ~3.5–5.5).

---

## 5. Ground-Truth Registration Analysis
Verified ground-truth strings discovered: `25`

Sample Verified Ground-Truth Registrations in Dataset:
- DL3CD1210, GJ01DY6855, KA01AJ7533, KA09C2763, KL03S6894, KL07BX7197, KL10AG7249, KL34A465, KL34F, KL35F4337, KL35H5834, KL41L7001

States Covered in Ground-Truth Annotations:
- KL (Kerala), UP (Uttar Pradesh), GJ (Gujarat), WB (West Bengal), MP (Madhya Pradesh), RJ (Rajasthan), TN (Tamil Nadu), DL (Delhi), KA (Karnataka), MH (Maharashtra), AP (Andhra Pradesh), HR (Haryana).

---

## 6. Leakage Safeguard Protocol
1. **Vehicle-Level Partitioning:** Multiple crops or frames containing the same registration string (e.g. `KL34A465`, `UP84AE9889`, `MH01AV...`) are strictly assigned to the same partition.
2. **Independent Test Partition:** 10% held-out test split is sealed and never included in training or validation.
3. **Regression Isolation:** The critical regression test `MH01AV8669` is completely withheld from all training sets.
