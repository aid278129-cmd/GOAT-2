import os
import glob
import base64
import json
import cv2
from anpr_server import get_model, extract_plate_ocr

def test_full_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model = get_model()

    images = glob.glob(os.path.join(base_dir, "number_plate_images_ocr", "number_plate_images_ocr", "*.jpg"))
    sample_images = glob.glob(os.path.join(base_dir, "Indian_Number_Plates", "Sample_Images", "*.jpg"))
    test_set = images[:4] + sample_images[:4]

    print(f"Testing full ANPR pipeline on {len(test_set)} real vehicle images...\n")

    results_summary = []
    for img_path in test_set:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        preds = model.predict(img, conf=0.20, imgsz=640, verbose=False)

        found = False
        for r in preds:
            for box in r.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Crop plate region with padding
                bw, bh = x2 - x1, y2 - y1
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w_img, x2 + pad_x)
                cy2 = min(h_img, y2 + pad_y)

                crop = img[cy1:cy2, cx1:cx2]
                text, ocr_conf, engine = extract_plate_ocr(crop)

                if len(text) >= 4:
                    found = True
                    results_summary.append({
                        "file": img_name,
                        "detected_plate": text,
                        "plate_confidence": round(conf * 100, 1),
                        "ocr_confidence": round(ocr_conf * 100, 1),
                        "engine": engine,
                        "box": [int(x1), int(y1), int(bw), int(bh)]
                    })
                    print(f"✅ {img_name} -> Plate: '{text}' (YOLO: {conf*100:.1f}%, OCR: {ocr_conf*100:.1f}%, Engine: {engine})")
                    break
        if not found:
            print(f"⚠️  {img_name} -> No clean plate extracted")

    print("\nSummary of detections:")
    print(json.dumps(results_summary, indent=2))

if __name__ == "__main__":
    test_full_pipeline()
