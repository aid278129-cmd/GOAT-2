#!/usr/bin/env python3
"""
Verify ground truth of all 38 base crops.
"""
import os
import sys
import json
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

BENCHMARK_INDEX = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ground_truth_benchmark.json")

def main():
    with open(BENCHMARK_INDEX, "r", encoding="utf-8") as f:
        samples = json.load(f)

    base_samples = [s for s in samples if not any(a in s.get("source", "") for a in ["MildBlur", "LowLight", "Contrast", "JPEGCompression", "SkewAngle"])]

    for idx, s in enumerate(base_samples):
        fn = s["crop_file"]
        gt = s["ground_truth"]
        p = s["crop_path"]
        im = cv2.imread(p)
        h, w = im.shape[:2] if im is not None else (0, 0)
        print(f"[{idx+1:02d}] {fn:50s} | GT: {gt:16s} | {w}x{h}")

if __name__ == "__main__":
    main()
