import os
import io
import time
import requests
import numpy as np
import cv2

ANPR_URL = "http://127.0.0.1:5001/detect-file"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_negative_blank_image():
    print("\n--- Test 1: Negative Image (Blank / Solid Color) ---")
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', blank)
    res = requests.post(ANPR_URL, files={'file': ('blank.jpg', buf.tobytes(), 'image/jpeg')}, data={'cameraId': 1})
    data = res.json()
    print("Response:", data)
    assert data["success"] is True
    assert data["detected"] is False
    assert data.get("status") in ["REJECTED_NO_VEHICLE", "NO_PLATE_CANDIDATES"]
    print(">>> PASS: Negative blank image rejected successfully without false positive.")

def test_negative_indoor_noise():
    print("\n--- Test 2: Negative Image (Indoor / Desk / Random Noise) ---")
    noise = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    cv2.putText(noise, "OFFICE DESK COMPUTER ROOM", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', noise)
    res = requests.post(ANPR_URL, files={'file': ('noise.jpg', buf.tobytes(), 'image/jpeg')}, data={'cameraId': 1})
    data = res.json()
    print("Response:", data)
    assert data["success"] is True
    assert data["detected"] is False
    assert data.get("status") in ["REJECTED_NO_VEHICLE", "NO_PLATE_CANDIDATES"]
    print(">>> PASS: Indoor noise rejected successfully with 0 false positives.")

def test_positive_real_vehicles():
    print("\n--- Test 3: Real Vehicle Images from Dataset ---")
    import glob
    images = glob.glob(os.path.join(BASE_DIR, "number_plate_images_ocr", "number_plate_images_ocr", "*.jpg"))
    sample_images = glob.glob(os.path.join(BASE_DIR, "Indian_Number_Plates", "Sample_Images", "*.jpg"))
    files = (images[:3] + sample_images[:3])
    print(f"Testing on {len(files)} real vehicle samples...")
    detected_count = 0
    for fpath in files:
        with open(fpath, "rb") as f:
            res = requests.post(ANPR_URL, files={'file': (os.path.basename(fpath), f.read(), 'image/jpeg')}, data={'cameraId': 1, 'manualScan': True})
        data = res.json()
        print(f"File: {os.path.basename(fpath)} -> detected: {data.get('detected')}, status: {data.get('status')}")
        if data.get("detected"):
            det = data.get("detection")
            print(f"  Plate: {det.get('plate')}, Conf: {det.get('confidence')}, State: {det.get('stateName')}, Engine: {det.get('ocrEngine')}")
            detected_count += 1
        elif data.get("detections"):
            for d in data.get("detections"):
                print(f"  Candidate: {d.get('rawOcr')}, Status: {d.get('status')}, Reason: {d.get('reason')}")
    print(f">>> Result: {detected_count}/{len(files)} vehicle plates decoded.")

def test_temporal_confirmation():
    print("\n--- Test 4: Multi-Frame Temporal Confirmation via /detect ---")
    import base64
    import glob
    images = glob.glob(os.path.join(BASE_DIR, "number_plate_images_ocr", "number_plate_images_ocr", "dc_auto_image_000021*.jpg"))
    if not images:
        images = glob.glob(os.path.join(BASE_DIR, "Indian_Number_Plates", "Sample_Images", "*.jpg"))
    if not images:
        return
    with open(images[0], "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    
    url = "http://127.0.0.1:5001/detect"
    
    res1 = requests.post(url, json={
        "image": f"data:image/jpeg;base64,{b64}",
        "cameraId": 99,
        "manualScan": False,
        "forwardToDashboard": False
    }).json()
    print("Frame 1 Status:", res1.get("status"))
    
    time.sleep(0.3)
    
    res2 = requests.post(url, json={
        "image": f"data:image/jpeg;base64,{b64}",
        "cameraId": 99,
        "manualScan": False,
        "forwardToDashboard": False
    }).json()
    print("Frame 2 Status:", res2.get("status"))
    if res2.get("detected"):
        print(f">>> PASS: Multi-frame temporal confirmation promoted candidate to CONFIRMED: {res2.get('detection', {}).get('plate')}")
    else:
        print(">>> Candidate status:", res2.get("status"))

if __name__ == "__main__":
    test_negative_blank_image()
    test_negative_indoor_noise()
    test_positive_real_vehicles()
    test_temporal_confirmation()
    print("\nALL STAGE 2 & 3 PIPELINE TESTS COMPLETE!")
