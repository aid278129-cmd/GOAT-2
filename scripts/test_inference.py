import os
import glob
import cv2
from ultralytics import YOLO
from anpr_server import clean_plate_text
import easyocr

def test_inference():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "indian_plate_best.pt")

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Loading EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)

    sample_images = glob.glob(os.path.join(base_dir, "Indian_Number_Plates", "Sample_Images", "*.jpg"))
    ocr_images = glob.glob(os.path.join(base_dir, "number_plate_images_ocr", "number_plate_images_ocr", "*.jpg"))

    test_candidates = (sample_images[:3] + ocr_images[:3])
    print(f"Testing on {len(test_candidates)} sample images...\n")

    for img_path in test_candidates:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model.predict(img, conf=0.25, imgsz=640, verbose=False)
        detected_plates = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Crop plate region with padding
                h_img, w_img = img.shape[:2]
                bw, bh = x2 - x1, y2 - y1
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w_img, x2 + pad_x)
                cy2 = min(h_img, y2 + pad_y)

                crop = img[cy1:cy2, cx1:cx2]
                ocr_results = reader.readtext(crop, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                raw_text = "".join([res[1] for res in ocr_results])
                clean_text = clean_plate_text(raw_text)

                detected_plates.append((conf, [x1, y1, bw, bh], clean_text))

        print(f"🖼️  Image: {img_name}")
        if detected_plates:
            for conf, bbox, text in detected_plates:
                print(f"   ✅ Detected Plate: '{text}' | Confidence: {conf*100:.1f}% | Box: {bbox}")
        else:
            print("   ❌ No plate detected")
        print("-" * 50)

if __name__ == "__main__":
    test_inference()
