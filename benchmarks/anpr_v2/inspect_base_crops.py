#!/usr/bin/env python3
"""
Inspect the 38 base crops in detail to establish verified ground truth.
"""
import os
import sys
import json
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.recognizer import get_recognizer
from anpr_v2.validator import clean_ocr_raw_tokens

BENCHMARK_INDEX = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ground_truth_benchmark.json")

def main():
    rec = get_recognizer("paddleocr")
    with open(BENCHMARK_INDEX, "r", encoding="utf-8") as f:
        samples = json.load(f)

    base_samples = [s for s in samples if not any(a in s.get("source", "") for a in ["MildBlur", "LowLight", "Contrast", "JPEGCompression", "SkewAngle"])]
    print(f"Inspecting {len(base_samples)} base crops:\n")

    for idx, s in enumerate(base_samples):
        p = s["crop_path"]
        im = cv2.imread(p)
        res = rec.recognize(im)
        print(f"[{idx+1:02d}] {s['crop_file']}")
        print(f"     Annotated GT : '{s['ground_truth']}'")
        print(f"     Raw OCR Text : '{res.raw_text}'")
        print(f"     Clean OCR    : '{res.text}' (conf: {res.confidence})")
        print(f"     Engine       : {res.engine}, 2-row: {res.is_two_row}")
        print()

if __name__ == "__main__":
    main()
