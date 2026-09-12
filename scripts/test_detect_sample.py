import requests
import base64
import json
import time

def test_detect():
    img_path = 'public/sample_cars/sample1.jpg'
    print(f"Reading {img_path}...")
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    
    print(f"Sending to http://127.0.0.1:5001/detect ({len(b64)} chars)...")
    t0 = time.time()
    res = requests.post(
        'http://127.0.0.1:5001/detect',
        json={'image': b64, 'cameraId': 1, 'forwardToDashboard': False},
        timeout=30
    )
    t1 = time.time()
    print(f"Status: {res.status_code}, Elapsed: {t1-t0:.2f}s")
    print("Response:", json.dumps(res.json(), indent=2))

if __name__ == '__main__':
    test_detect()
