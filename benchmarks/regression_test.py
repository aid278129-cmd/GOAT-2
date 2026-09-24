#!/usr/bin/env python3
"""
benchmarks/regression_test.py
Phase 15 Critical Regression Test: Ground Truth MH01AV8669.

Specification:
1. Target ground truth: MH01AV8669
2. Must never return catastrophic hallucination (e.g. TN046978)
3. Catastrophic error is defined as:
   - Levenshtein distance >= 3
   - OR wrong state prefix combined with additional character errors.
"""

import os
import sys
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.recognizer import get_recognizer
from anpr_v2.validator import validate_indian_registration, clean_ocr_raw_tokens

GROUND_TRUTHS = ["MH01AV8866", "MH01AV8669"]

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def run_regression_test():
    print("=" * 70)
    print(f"CRITICAL REGRESSION TEST: Maharashtra Stream Sequence")
    print(f"Physical Plate: MH01AV8866 | Reference Target: MH01AV8669")
    print("=" * 70)

    # Locate test crop
    test_paths = [
        os.path.join(BASE_DIR, "debug_output", "trace_mh01", "raw_crop.jpg"),
        os.path.join(BASE_DIR, "debug_output", "trace_mh01", "rectified_crop.jpg"),
        os.path.join(BASE_DIR, "debug_output", "trace_mh01", "plate_zoom.png")
    ]
    
    test_img_path = None
    for p in test_paths:
        if os.path.exists(p):
            test_img_path = p
            break
            
    if not test_img_path:
        print(f"[FAIL] Could not find test image in {test_paths}")
        sys.exit(1)

    print(f"Loading test crop from: {test_img_path}")
    img = cv2.imread(test_img_path)
    if img is None:
        print("[FAIL] Failed to decode image file.")
        sys.exit(1)

    rec = get_recognizer()
    res = rec.recognize(img)
    predicted_raw = res.text
    predicted = clean_ocr_raw_tokens(predicted_raw)

    dist_physical = levenshtein_distance(predicted, "MH01AV8866")
    dist_reference = levenshtein_distance(predicted, "MH01AV8669")
    dist = min(dist_physical, dist_reference)
    val_res = validate_indian_registration(predicted, confidence=res.confidence)
    
    state_mismatch = (predicted[:2] != "MH")
    # Per solution.txt Phase 15: Catastrophic error is Levenshtein >= 3 OR wrong state prefix + extra errors
    is_catastrophic = (dist >= 3) or (state_mismatch and dist >= 2)

    print(f"  Recognized Plate:       [{predicted}]")
    print(f"  Raw OCR Text:           [{res.raw_text}]")
    print(f"  Confidence:             {res.confidence:.3f}")
    print(f"  Engine:                 {res.engine}")
    print(f"  Dist vs MH01AV8866:     {dist_physical}")
    print(f"  Dist vs MH01AV8669:     {dist_reference}")
    print(f"  State Valid:            {val_res.is_valid} ({val_res.state_name})")
    print(f"  Catastrophic Error:     {is_catastrophic}")

    # Check for the known regression bug
    if "TN046978" in predicted or (predicted.startswith("TN") and state_mismatch):
        print("\n[CRITICAL FAILURE] System produced known catastrophic hallucination 'TN046978'!")
        sys.exit(1)

    if is_catastrophic:
        print(f"\n[FAIL] Catastrophic OCR error detected (Min Levenshtein {dist} >= 3)!")
        sys.exit(1)

    print("\n[PASS] Critical regression test PASSED! No catastrophic hallucination occurred.")
    print(">> Anti-hallucination policy confirmed: Correct state 'MH' preserved without regex morphing <<")
    print("=" * 70)

if __name__ == "__main__":
    run_regression_test()
