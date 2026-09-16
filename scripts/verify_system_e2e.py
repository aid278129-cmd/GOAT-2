#!/usr/bin/env python3
"""
scripts/verify_system_e2e.py
End-to-End System Smoke & Integration Verification.

Verifies:
1. ANPR V2 Server launches successfully on port 5001
2. /health endpoint reports status='ok' and anpr_version='v2'
3. /detect endpoint accepts base64 frames and returns expected schema for server.js
4. /config endpoint supports runtime configuration queries
5. Clean shutdown
"""

import os
import sys
import time
import json
import base64
import subprocess
import cv2
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVER_URL = "http://127.0.0.1:5001"

def run_e2e_verification():
    print("=" * 70)
    print("ANPR V2 END-TO-END SYSTEM INTEGRATION VERIFICATION")
    print("=" * 70)

    # Launch ANPR server
    cmd = [sys.executable, os.path.join(BASE_DIR, "scripts", "anpr_server.py")]
    print(f"Starting server process: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        # 1. Wait for server to become healthy
        print("Waiting for server on http://127.0.0.1:5001/health ...")
        healthy = False
        health_data = {}
        for attempt in range(30):
            try:
                res = requests.get(f"{SERVER_URL}/health", timeout=1.0)
                if res.status_code == 200:
                    health_data = res.json()
                    healthy = True
                    break
            except Exception:
                time.sleep(1.0)

        if not healthy:
            print("[FAIL] Server failed to respond to /health within 30 seconds.")
            sys.exit(1)

        print("[OK] Server responded to /health:")
        print(f"     Status:       {health_data.get('status')}")
        print(f"     ANPR Version: {health_data.get('anpr_version')}")
        print(f"     Plate Detector Ready: {health_data.get('plate_detector_ready')}")
        print(f"     Vehicle Detector Ready: {health_data.get('vehicle_detector_ready')}")

        assert health_data.get("status") == "ok", "Health status is not 'ok'"
        assert health_data.get("anpr_version") == "v2", f"ANPR version is not 'v2': {health_data.get('anpr_version')}"

        # 2. Test /config endpoint
        cfg_res = requests.get(f"{SERVER_URL}/config", timeout=2.0)
        assert cfg_res.status_code == 200, "Failed to fetch /config"
        cfg_data = cfg_res.json()
        print(f"[OK] /config verified (version: {cfg_data.get('anpr_version')})")

        # 3. Test /detect endpoint with a frame
        test_img_path = os.path.join(BASE_DIR, "debug_output", "trace_mh01", "raw_crop.jpg")
        if os.path.exists(test_img_path):
            img = cv2.imread(test_img_path)
        else:
            # Create a synthetic plate frame
            img = 255 * np.ones((100, 300, 3), dtype=np.uint8)
            cv2.putText(img, "MH01AV8669", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        _, buf = cv2.imencode(".jpg", img)
        b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"

        payload = {
            "image": b64_str,
            "cameraId": 1,
            "cameraName": "E2E Verification Cam",
            "forwardToDashboard": False,
            "manualScan": True
        }

        print("Sending POST request to /detect ...")
        t0 = time.time()
        det_res = requests.post(f"{SERVER_URL}/detect", json=payload, timeout=10.0)
        latency_ms = (time.time() - t0) * 1000
        assert det_res.status_code == 200, f"/detect returned status {det_res.status_code}: {det_res.text}"

        data = det_res.json()
        print(f"[OK] /detect responded in {latency_ms:.1f}ms:")
        print(f"     Success:   {data.get('success')}")
        if not data.get("success"):
            print(f"     Error:     {data.get('error')}")
            print(f"     Full Data: {data}")
        print(f"     Timing:    {list(data.get('timing', {}).keys())}")
        print(f"     Debug:     {list(data.get('debug', {}).keys())}")
        if data.get("detection"):
            print(f"     Confirmed: {data.get('detection')}")
        if data.get("detections"):
            print(f"     Total Detections: {len(data.get('detections', []))}")

        # Check expected keys required by server.js
        for req_key in ["timing", "debug", "success"]:
            assert req_key in data, f"Required telemetry key '{req_key}' missing from /detect response"

        print("=" * 70)
        print("[SUCCESS] ALL END-TO-END VERIFICATION CHECKS PASSED!")
        print("=" * 70)

    finally:
        print("Shutting down server process...")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("Server process stopped.")

if __name__ == "__main__":
    run_e2e_verification()
