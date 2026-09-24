#!/usr/bin/env python3
"""
benchmarks/anpr_v2/audit_benchmark.py
Comprehensive Audit of the 228-sample Benchmark vs. ANPR V2 Production Engine.
"""

import os
import sys
import json
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.config import CONFIG
from anpr_v2.recognizer import get_recognizer, OCRResult
from anpr_v2.validator import clean_ocr_raw_tokens, validate_indian_registration
from anpr_v2.rectifier import rectify_plate_perspective, normalize_plate_resolution
from anpr_v2.tracker import compute_laplacian_sharpness

BENCHMARK_INDEX = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ground_truth_benchmark.json")

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = cur[j] + 1
            sub = prev[j] + (c1 != c2)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def character_accuracy(p: str, g: str) -> float:
    p_clean = clean_ocr_raw_tokens(p)
    g_clean = clean_ocr_raw_tokens(g)
    if not p_clean and not g_clean:
        return 1.0
    if not p_clean or not g_clean:
        return 0.0
    max_len = max(len(p_clean), len(g_clean))
    dist = levenshtein_distance(p_clean, g_clean)
    return max(0.0, 1.0 - (dist / float(max_len)))

def audit():
    rec = get_recognizer("paddleocr")
    with open(BENCHMARK_INDEX, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} benchmark samples.")
    base_samples = [s for s in samples if not any(a in s.get("source", "") for a in ["MildBlur", "LowLight", "Contrast", "JPEGCompression", "SkewAngle"])]
    
    print("\n" + "="*80)
    print("AUDITING 38 BASE (NON-AUGMENTED) SAMPLES")
    print("="*80)

    for i, s in enumerate(base_samples):
        p = s["crop_path"]
        gt = s["ground_truth"]
        im = cv2.imread(p)
        if im is None:
            print(f"Error loading {p}")
            continue
            
        h, w = im.shape[:2]
        ar = round(w / float(max(1, h)), 2)
        sharp = round(compute_laplacian_sharpness(im), 1)

        # Benchmark way (raw crop directly to recognizer)
        res_bench = rec.recognize(im)
        pred_bench = clean_ocr_raw_tokens(res_bench.text)
        dist_bench = levenshtein_distance(pred_bench, gt)

        # Production way (rectify + normalize resolution)
        rect_im, was_rect, skew = rectify_plate_perspective(im)
        norm_im = normalize_plate_resolution(rect_im)
        res_prod = rec.recognize(norm_im)
        pred_prod = clean_ocr_raw_tokens(res_prod.text)
        dist_prod = levenshtein_distance(pred_prod, gt)

        print(f"\nSample {i+1:2d}: {s['crop_file']}")
        print(f"  Shape: {w}x{h} (AR={ar}), Sharpness: {sharp}, Source: {s['source']}")
        print(f"  Ground Truth: [{gt}]")
        print(f"  Raw OCR Bench: [{res_bench.raw_text}] -> Pred: [{pred_bench}] (dist={dist_bench}, conf={res_bench.confidence}, eng={res_bench.engine})")
        if pred_bench != pred_prod or dist_bench != dist_prod:
            print(f"  Prod Preproc:  [{res_prod.raw_text}] -> Pred: [{pred_prod}] (dist={dist_prod}, conf={res_prod.confidence})")

if __name__ == "__main__":
    audit()
