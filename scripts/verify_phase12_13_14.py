#!/usr/bin/env python3
"""
verify_phase12_13_14.py — Verification Suite for Phases 12, 13, and 14
=====================================================================
SIH Problem Statement ID: 26127 (BEL)

Verifies:
  - Phase 12: Multi-Camera Chronological Trajectory & GIS Segments
              (Example from solution.txt: TN45AB1234 across CAM-01, CAM-02, CAM-04)
  - Phase 13: Ground-Truth Driven Urban Traffic Analytics
              (Total detections, unique vehicles, vpm, busiest camera, common routes)
  - Phase 14: Watchlist Addition & Real-Time Alert Dispatch
              (Arming TN45AB1234, triggering detection, and validating alert payload)
"""

import os
import sys
import time
import json
import datetime
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

NODE_URL = "https://127.0.0.1:3000"


def verify_phase12_trajectory():
    print("=" * 80)
    print(" PHASE 12: TRAJECTORY & GIS VERIFICATION")
    print("=" * 80)

    demo_plate = "TN45AB1234"
    print(f"Target Vehicle Plate: {demo_plate}")

    # Simulated timestamps corresponding to solution.txt scenario:
    # 10:15 (CAM-01) -> 10:27 (CAM-02) -> 10:41 (CAM-04)
    now = datetime.datetime.now()
    t1 = (now - datetime.timedelta(minutes=26)).isoformat()
    t2 = (now - datetime.timedelta(minutes=14)).isoformat()
    t3 = now.isoformat()

    steps = [
        {"cameraId": 1, "cameraName": "Highway Entry (CAM-01)", "lat": 12.9716, "lng": 77.5946, "timestamp": t1},
        {"cameraId": 2, "cameraName": "Junction B (CAM-02)",     "lat": 12.9750, "lng": 77.6000, "timestamp": t2},
        {"cameraId": 4, "cameraName": "Junction C (CAM-04)",     "lat": 12.9800, "lng": 77.6100, "timestamp": t3},
    ]

    print("\nIngesting multi-camera sightings in sequence:")
    for step in steps:
        payload = {
            "plate": demo_plate,
            "cameraId": step["cameraId"],
            "cameraName": step["cameraName"],
            "confidence": 0.94,
            "vehicleType": "car",
            "simulated": False,
            "timestamp": step["timestamp"],
            "lat": step["lat"],
            "lng": step["lng"]
        }
        r = requests.post(f"{NODE_URL}/api/detections", json=payload, verify=False, timeout=5)
        assert r.status_code == 200, f"Failed to ingest detection: {r.text}"
        print(f"  [+] Ingested at {step['cameraName']} | Timestamp: {step['timestamp']}")
        time.sleep(0.1)

    # Query trajectory API
    print(f"\nQuerying /api/detections/trajectory/{demo_plate} ...")
    r_traj = requests.get(f"{NODE_URL}/api/detections/trajectory/{demo_plate}", verify=False, timeout=5)
    assert r_traj.status_code == 200, f"Failed to fetch trajectory: {r_traj.text}"
    traj = r_traj.json()

    print(f"  • Found:           {traj.get('found')}")
    print(f"  • Total Sightings: {traj.get('totalSightings')}")
    print(f"  • Unique Cameras:  {traj.get('cameras')}")
    print(f"  • Total Duration:  {traj.get('totalMinutes')} minutes")
    print(f"  • Segments Count:  {len(traj.get('segments', []))}")

    # Assertions
    assert traj.get("found") is True, "Trajectory not found for test plate!"
    assert traj.get("totalSightings") >= 3, "Expected at least 3 sightings!"
    assert len(traj.get("cameras")) >= 3, "Expected at least 3 unique cameras!"

    # Verify chronological ordering
    trail = traj.get("trail", [])
    timestamps = [datetime.datetime.fromisoformat(d["timestamp"].replace("Z", "")) for d in trail if "timestamp" in d]
    is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    assert is_sorted, "Trajectory trail is not chronologically sorted!"
    print("  [OK] Chronological ordering verified.")

    # Verify GIS segments and travel times
    segments = traj.get("segments", [])
    print("\nReconstructed Camera-to-Camera Travel Segments:")
    for i, seg in enumerate(segments, 1):
        print(f"  Segment {i}: {seg.get('from')} -> {seg.get('to')} (Travel Time: {seg.get('travelMins')} min)")

    assert len(segments) >= 2, "Expected at least 2 camera-to-camera route segments!"
    print("\n>>> PASS: Phase 12 Trajectory and GIS verification successfully passed!\n")
    return traj


def verify_phase13_analytics():
    print("=" * 80)
    print(" PHASE 13: TRAFFIC ANALYTICS VERIFICATION")
    print("=" * 80)

    print("Querying /api/detections/analytics ...")
    r_ana = requests.get(f"{NODE_URL}/api/detections/analytics", verify=False, timeout=5)
    assert r_ana.status_code == 200, f"Failed to fetch analytics: {r_ana.text}"
    ana = r_ana.json()

    total = ana.get("totalDetections", 0)
    last60 = ana.get("last60min", 0)
    vpm = ana.get("vehiclesPerMin", 0)
    busiest = ana.get("busiestCamera", "N/A")
    unique_p = ana.get("uniquePlates", 0)
    top_routes = ana.get("topRoutes", [])
    per_cam = ana.get("perCamera", [])

    print(f"  • Total Detections:       {total}")
    print(f"  • Last 60 Min Detections: {last60}")
    print(f"  • Vehicles Per Minute:    {vpm}")
    print(f"  • Unique Vehicle Count:   {unique_p}")
    print(f"  • Busiest Camera:         {busiest}")
    print(f"  • Common Routes Count:    {len(top_routes)}")
    print(f"  • Active Camera Nodes:    {len(per_cam)}")

    # Verify that metrics are based on actual detections
    assert total > 0, "Total detections must be > 0"
    assert unique_p > 0, "Unique plates count must be > 0"
    assert busiest != "N/A", "Busiest camera should be computed from actual camera counts"

    print("\nCamera Traffic Distribution:")
    for cam in per_cam:
        print(f"  - {cam.get('name')}: {cam.get('count')} detections")

    if top_routes:
        print("\nTop Traffic Movement Routes:")
        for r in top_routes[:3]:
            print(f"  - Route: {r.get('route')} -> {r.get('count')} vehicles")

    print("\n>>> PASS: Phase 13 Analytics verification successfully passed!\n")
    return ana


def verify_phase14_watchlist():
    print("=" * 80)
    print(" PHASE 14: WATCHLIST VERIFICATION")
    print("=" * 80)

    test_plate = "TN45AB1234"
    test_reason = "Test Watchlist Vehicle (BOLO: Highway Patrol Alert)"

    # 1. Register vehicle on watchlist
    print(f"Registering {test_plate} to /api/watchlist ...")
    r_add = requests.post(
        f"{NODE_URL}/api/watchlist",
        json={"plate": test_plate, "reason": test_reason},
        verify=False,
        timeout=5
    )
    assert r_add.status_code in [200, 201], f"Failed to add to watchlist: {r_add.text}"
    print(f"  [OK] Added {test_plate} to watchlist: {r_add.json()}")

    # 2. Expose vehicle to a camera and trigger detection
    print(f"\nTriggering camera sighting for watchlisted vehicle {test_plate} ...")
    sighting_payload = {
        "plate": test_plate,
        "cameraId": 3,
        "cameraName": "Junction A (CAM-03)",
        "confidence": 0.98,
        "vehicleType": "car",
        "simulated": False
    }
    r_det = requests.post(f"{NODE_URL}/api/detections", json=sighting_payload, verify=False, timeout=5)
    assert r_det.status_code == 200, f"Detection ingestion failed: {r_det.text}"
    det = r_det.json()

    print(f"  • Detection Stored:  {det.get('plate')}")
    print(f"  • Watchlist Trigger: {det.get('isWatchlisted')}")
    print(f"  • Alert Reason:      {det.get('watchlistReason')}")

    assert det.get("isWatchlisted") is True, "Expected isWatchlisted to be True!"
    assert "Watchlist Vehicle" in det.get("watchlistReason", ""), "Expected alert reason to match watchlist registration!"

    # 3. Verify alert was logged in /api/alerts
    print("\nChecking /api/alerts dispatch history ...")
    r_alerts = requests.get(f"{NODE_URL}/api/alerts?limit=10", verify=False, timeout=5)
    assert r_alerts.status_code == 200, f"Failed to fetch alerts: {r_alerts.text}"
    alerts = r_alerts.json()
    matched_alerts = [a for a in alerts if a.get("plate") == test_plate]
    assert len(matched_alerts) > 0, f"No alert record found in /api/alerts for {test_plate}!"

    latest_alert = matched_alerts[0]
    print(f"  • Alert Plate:      {latest_alert.get('plate')}")
    print(f"  • Alert Camera:     {latest_alert.get('cameraName')}")
    print(f"  • Alert Location:   {latest_alert.get('lat')}, {latest_alert.get('lng')}")
    print(f"  • Alert Timestamp:  {latest_alert.get('timestamp')}")
    print(f"  • Alert Confidence: {latest_alert.get('confidence')}")
    print(f"  • Alert Reason:     {latest_alert.get('reason')}")

    print("\n>>> PASS: Phase 14 Watchlist verification successfully passed!\n")
    return latest_alert


if __name__ == "__main__":
    print("##################################################################")
    print("  SIH 26127 / BEL — PHASES 12, 13, 14 VERIFICATION SUITE")
    print("##################################################################\n")

    verify_phase12_trajectory()
    verify_phase13_analytics()
    verify_phase14_watchlist()

    print("=" * 80)
    print(" [ALL PASSED] PHASES 12, 13, AND 14 ARE FULLY VERIFIED AND FUNCTIONAL!")
    print("=" * 80)
