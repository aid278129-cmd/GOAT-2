import os
import glob
import base64
import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def verify_end_to_end():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_img = os.path.join(base_dir, "number_plate_images_ocr", "number_plate_images_ocr", "dc_auto_image_000024_fqvRhfiO6i.jpg")

    print(f"Reading test image: {sample_img}")
    with open(sample_img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": f"data:image/jpeg;base64,{b64}",
        "cameraId": 1,
        "cameraName": "Junction A",
        "forwardToDashboard": True
    }

    print("Sending live frame through Node.js proxy -> Python ANPR service (POST https://127.0.0.1:3000/api/anpr/detect)...")
    resp = requests.post("https://127.0.0.1:3000/api/anpr/detect", json=payload, verify=False, timeout=10)
    print(f"Response Status: {resp.status_code}")
    data = resp.json()
    print("Response Data:")
    print(json.dumps(data, indent=2))

    print("\nQuerying Node.js detections API (GET https://127.0.0.1:3000/api/detections?limit=3)...")
    get_resp = requests.get("https://127.0.0.1:3000/api/detections?limit=3", verify=False, timeout=5)
    latest = get_resp.json()
    print("Latest 3 Detections in Dashboard:")
    print(json.dumps(latest, indent=2))

if __name__ == "__main__":
    verify_end_to_end()
