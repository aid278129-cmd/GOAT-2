import os
import time
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NODE_URL = "https://127.0.0.1:3000"

def test_multi_camera_trajectory():
    print("\n--- Test 1: Multi-Camera Trajectory Tracking ---")
    test_plate = "DL5SAF3210"
    cameras = [1, 2, 3, 4]
    
    # Ingest chronological sightings across 4 cameras
    for cid in cameras:
        res = requests.post(
            f"{NODE_URL}/api/detections",
            json={
                "plate": test_plate,
                "cameraId": cid,
                "confidence": 0.92,
                "vehicleType": "car",
                "simulated": False
            },
            verify=False,
            timeout=5
        )
        assert res.status_code == 200, f"Failed detection on camera {cid}: {res.text}"
        time.sleep(0.5)

    # Query trajectory
    traj_res = requests.get(f"{NODE_URL}/api/detections/trajectory/{test_plate}", verify=False, timeout=5)
    assert traj_res.status_code == 200, f"Failed to get trajectory: {traj_res.text}"
    traj = traj_res.json()
    print("Trajectory Result:", {
        "plate": traj.get("plate"),
        "found": traj.get("found"),
        "totalSightings": traj.get("totalSightings"),
        "cameras": traj.get("cameras"),
        "segments": len(traj.get("segments", []))
    })
    
    assert traj.get("found") is True
    assert traj.get("totalSightings") >= 4
    assert len(traj.get("cameras")) >= 4
    print(">>> PASS: Multi-camera trajectory successfully tracked across 4 cameras.")

def test_watchlist_alerting():
    print("\n--- Test 2: Watchlist Addition & Alert Triggering ---")
    wl_plate = "KA03CD5678"
    
    # 1. Add plate to watchlist
    wl_add_res = requests.post(
        f"{NODE_URL}/api/watchlist",
        json={"plate": wl_plate, "reason": "BOLO: Stolen Luxury Sedan (SIH Demo)"},
        verify=False,
        timeout=5
    )
    assert wl_add_res.status_code in [200, 201]
    wl_entries = wl_add_res.json()
    print(f"Watchlist size after addition: {len(wl_entries)}")
    
    # 2. Trigger detection for watchlisted plate
    det_res = requests.post(
        f"{NODE_URL}/api/detections",
        json={
            "plate": wl_plate,
            "cameraId": 2,
            "confidence": 0.94,
            "vehicleType": "car",
            "simulated": False
        },
        verify=False,
        timeout=5
    )
    assert det_res.status_code == 200, f"Detection failed: {det_res.text}"
    det = det_res.json()
    print("Detection Result:", {
        "plate": det.get("plate"),
        "cameraName": det.get("cameraName"),
        "isWatchlisted": det.get("isWatchlisted"),
        "watchlistReason": det.get("watchlistReason")
    })
    
    assert det.get("isWatchlisted") is True
    assert "BOLO" in det.get("watchlistReason")
    print(">>> PASS: Watchlist detection triggered and tagged with alert reason.")

def test_traffic_analytics():
    print("\n--- Test 3: City-Wide Urban Traffic Analytics ---")
    analytics_res = requests.get(f"{NODE_URL}/api/detections/analytics", verify=False, timeout=5)
    assert analytics_res.status_code == 200
    analytics = analytics_res.json()
    print("Analytics Summary:", {
        "totalDetections": analytics.get("totalDetections"),
        "last60min": analytics.get("last60min"),
        "vehiclesPerMin": analytics.get("vehiclesPerMin"),
        "busiestCamera": analytics.get("busiestCamera"),
        "topRoutesCount": len(analytics.get("topRoutes", []))
    })
    
    assert analytics.get("totalDetections") > 0
    assert analytics.get("busiestCamera") is not None
    assert len(analytics.get("perCamera", [])) >= 4
    print(">>> PASS: Urban traffic analytics aggregations active and consistent.")

if __name__ == "__main__":
    test_multi_camera_trajectory()
    test_watchlist_alerting()
    test_traffic_analytics()
    print("\nALL STAGE 6 & 7 TESTS PASSED SUCCESSFULLY!")
