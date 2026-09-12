"""
evaluate_tesseract_ocr.py
Automated benchmark and evaluation of Tesseract OCR on Indian Vehicle License Plates.
Tests ground-truth cropped plates from dataset and computes accuracy and Character Error Rate (CER).
"""

import os
import glob
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytesseract
import difflib

# Ensure Tesseract binary is linked
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

from anpr_server import extract_plate_ocr, clean_plate_text, INDIAN_STATES

def compute_cer(reference, hypothesis):
    """Compute Character Error Rate (edit distance / len(ref))."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    similarity = matcher.ratio()
    return round(1.0 - similarity, 3)

def run_evaluation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml_dir = os.path.join(base_dir, "number_plate_annos_ocr", "number_plate_annos_ocr")
    img_dir = os.path.join(base_dir, "number_plate_images_ocr", "number_plate_images_ocr")

    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    print(f"Discovered {len(xml_files)} ground-truth XML annotation files.")

    results = []
    total_plates = 0
    state_matches = 0
    exact_matches = 0
    total_cer = 0.0

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            fname_elem = root.find("filename")
            if fname_elem is None or not fname_elem.text:
                continue

            img_name = fname_elem.text.strip()
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                # Try finding alternative extension
                bname = os.path.splitext(img_name)[0]
                candidates = glob.glob(os.path.join(img_dir, bname + ".*"))
                if candidates:
                    img_path = candidates[0]
                else:
                    continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            for obj in root.findall("object"):
                # Find ground truth text
                gt_text = ""
                attrs = obj.find("attributes")
                if attrs is not None:
                    for attr in attrs.findall("attribute"):
                        name_elem = attr.find("name")
                        if name_elem is not None and name_elem.text == "number_plate_text":
                            val_elem = attr.find("value")
                            if val_elem is not None and val_elem.text:
                                gt_text = val_elem.text.strip().upper()

                if not gt_text or len(gt_text) < 4:
                    continue

                bnd = obj.find("bndbox")
                if bnd is None:
                    continue

                xmin = int(float(bnd.find("xmin").text))
                ymin = int(float(bnd.find("ymin").text))
                xmax = int(float(bnd.find("xmax").text))
                ymax = int(float(bnd.find("ymax").text))

                h, w = image.shape[:2]
                xmin = max(0, min(xmin, w))
                ymin = max(0, min(ymin, h))
                xmax = max(0, min(xmax, w))
                ymax = max(0, min(ymax, h))

                if (xmax - xmin) < 15 or (ymax - ymin) < 8:
                    continue

                # Add a 5% margin around the plate crop
                pad_x = int((xmax - xmin) * 0.05)
                pad_y = int((ymax - ymin) * 0.05)
                cx1 = max(0, xmin - pad_x)
                cy1 = max(0, ymin - pad_y)
                cx2 = min(w, xmax + pad_x)
                cy2 = min(h, ymax + pad_y)

                crop = image[cy1:cy2, cx1:cx2]
                pred_text, conf, engine = extract_plate_ocr(crop)

                gt_clean = clean_plate_text(gt_text)
                pred_clean = pred_text

                cer = compute_cer(gt_clean, pred_clean)
                is_exact = (gt_clean == pred_clean)
                is_state_match = (len(gt_clean) >= 2 and len(pred_clean) >= 2 and gt_clean[:2] == pred_clean[:2])

                total_plates += 1
                if is_exact:
                    exact_matches += 1
                if is_state_match:
                    state_matches += 1
                total_cer += cer

                results.append({
                    "image": img_name,
                    "groundTruth": gt_clean,
                    "predicted": pred_clean,
                    "confidence": round(conf, 2),
                    "ocrEngine": engine,
                    "exactMatch": is_exact,
                    "stateMatch": is_state_match,
                    "cer": cer
                })

                print(f"[{total_plates:02d}] GT: {gt_clean:<12} | Pred: {pred_clean:<12} | Conf: {conf:.2f} | CER: {cer:.2f} | Engine: {engine}")

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    if total_plates > 0:
        avg_cer = round(total_cer / total_plates, 3)
        state_acc = round((state_matches / total_plates) * 100, 1)
        exact_acc = round((exact_matches / total_plates) * 100, 1)
    else:
        avg_cer, state_acc, exact_acc = 0.0, 0.0, 0.0

    summary = {
        "totalPlatesTested": total_plates,
        "exactMatchCount": exact_matches,
        "exactMatchAccuracyPercent": exact_acc,
        "stateMatchCount": state_matches,
        "stateMatchAccuracyPercent": state_acc,
        "averageCER": avg_cer,
        "tesseractVersion": "5.4.0",
        "detailedResults": results
    }

    print("\n=======================================================")
    print("           TESSERACT OCR BENCHMARK SUMMARY             ")
    print("=======================================================")
    print(f"Total License Plates Evaluated : {total_plates}")
    print(f"State Code Recognition Rate     : {state_acc}% ({state_matches}/{total_plates})")
    print(f"Exact Plate String Accuracy     : {exact_acc}% ({exact_matches}/{total_plates})")
    print(f"Average Character Error Rate    : {avg_cer}")
    print("=======================================================\n")

    out_json = os.path.join(base_dir, "models", "tesseract_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark results to {out_json}")

if __name__ == "__main__":
    run_evaluation()
