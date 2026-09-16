#!/usr/bin/env python3
"""
benchmarks/anpr_v2/run_ab_benchmark.py
Executes Phase 15 & Phase 16: Empirical Benchmarking & A/B Testing.
Compares Old System (CRNN/Tesseract/EasyOCR baseline) vs. ANPR V2 (PP-OCRv4 ONNX).

Measures:
1. Full Plate Exact Match Rate
2. Character Accuracy (Mean Levenshtein Similarity)
3. 1-character Error Rate
4. 2-character Error Rate
5. 3+ character Error Rate (Catastrophic Error Rate)
6. Wrong State Code Rate
7. Median Latency & P95 Latency
8. Generates A/B comparison report table.
"""

import os
import sys
import json
import time
import csv
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.recognizer import get_recognizer
from anpr_v2.validator import validate_indian_registration, clean_ocr_raw_tokens

BENCHMARK_INDEX = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ground_truth_benchmark.json")
RESULTS_JSON = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ab_benchmark_results.json")
RESULTS_CSV = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ab_benchmark_results.csv")

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

def evaluate_pipeline(recognizer, samples, name="V2"):
    records = []
    latencies = []
    
    exact_matches = 0
    err_1_char = 0
    err_2_char = 0
    err_3plus_char = 0
    wrong_state_count = 0
    total_char_acc = 0.0

    print(f"\nEvaluating [{name}] on {len(samples)} benchmark crops...")
    for idx, item in enumerate(samples):
        img_path = item["crop_path"]
        gt = item["ground_truth"]
        
        im = cv2.imread(img_path)
        if im is None:
            continue

        t0 = time.perf_counter()
        res = recognizer.recognize(im)
        t_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_ms)

        pred = clean_ocr_raw_tokens(res.text)
        dist = levenshtein_distance(pred, gt)
        c_acc = character_accuracy(pred, gt)
        total_char_acc += c_acc

        is_exact = (pred == gt)
        if is_exact:
            exact_matches += 1
        elif dist == 1:
            err_1_char += 1
        elif dist == 2:
            err_2_char += 1
        else:
            err_3plus_char += 1

        gt_state = gt[:2]
        pred_state = pred[:2]
        state_mismatch = bool(pred and (pred_state != gt_state))
        if state_mismatch:
            wrong_state_count += 1

        records.append({
            "image": item["crop_file"],
            "ground_truth": gt,
            "predicted": pred,
            "raw_ocr": res.raw_text,
            "confidence": res.confidence,
            "exact_match": is_exact,
            "edit_distance": dist,
            "char_accuracy": round(c_acc, 4),
            "state_mismatch": state_mismatch,
            "latency_ms": round(t_ms, 2),
            "engine": res.engine
        })

    n = max(1, len(records))
    latencies_sorted = sorted(latencies)
    median_lat = np.median(latencies_sorted) if latencies_sorted else 0.0
    p95_lat = np.percentile(latencies_sorted, 95) if latencies_sorted else 0.0

    summary = {
        "pipeline_name": name,
        "total_samples": len(records),
        "exact_matches": exact_matches,
        "exact_match_pct": round((exact_matches / n) * 100, 2),
        "mean_character_accuracy_pct": round((total_char_acc / n) * 100, 2),
        "error_1_char_pct": round((err_1_char / n) * 100, 2),
        "error_2_char_pct": round((err_2_char / n) * 100, 2),
        "catastrophic_error_3plus_pct": round((err_3plus_char / n) * 100, 2),
        "wrong_state_pct": round((wrong_state_count / n) * 100, 2),
        "median_latency_ms": round(float(median_lat), 1),
        "p95_latency_ms": round(float(p95_lat), 1),
        "detailed_records": records
    }
    return summary

def run_ab_benchmark():
    print("=" * 80)
    print("    PHASE 16: SYSTEMATIC A/B BENCHMARK EVALUATION (OLD vs ANPR V2)")
    print("=" * 80)

    if not os.path.exists(BENCHMARK_INDEX):
        print(f"Error: Benchmark index {BENCHMARK_INDEX} not found. Running builder...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "build_benchmark_dataset.py")])

    with open(BENCHMARK_INDEX, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} held-out Indian number plate crops for evaluation.")

    # 1. Evaluate Old Baseline (CRNN Recognizer)
    old_recognizer = get_recognizer(backend="crnn")
    old_results = evaluate_pipeline(old_recognizer, samples, name="OLD_BASELINE (CRNN)")

    # 2. Evaluate ANPR V2 (PP-OCRv4 ONNX Recognizer)
    v2_recognizer = get_recognizer(backend="paddleocr")
    v2_results = evaluate_pipeline(v2_recognizer, samples, name="ANPR_V2 (PP-OCRv4 ONNX)")

    # Print comparative report table per Phase 16
    print("\n" + "=" * 80)
    print("                 PHASE 16 EMPIRICAL A/B BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<32} | {'OLD (CRNN)':<18} | {'ANPR V2 (PP-OCR)':<18} | {'Improvement':<16}")
    print("-" * 80)

    diff_exact = v2_results["exact_match_pct"] - old_results["exact_match_pct"]
    diff_char = v2_results["mean_character_accuracy_pct"] - old_results["mean_character_accuracy_pct"]
    diff_cat = old_results["catastrophic_error_3plus_pct"] - v2_results["catastrophic_error_3plus_pct"]
    diff_state = old_results["wrong_state_pct"] - v2_results["wrong_state_pct"]
    diff_med_lat = old_results["median_latency_ms"] - v2_results["median_latency_ms"]

    print(f"{'Full Plate Exact Match':<32} | {old_results['exact_match_pct']:>16.1f}% | {v2_results['exact_match_pct']:>16.1f}% | {diff_exact:>+15.1f}%")
    print(f"{'Character Accuracy':<32} | {old_results['mean_character_accuracy_pct']:>16.1f}% | {v2_results['mean_character_accuracy_pct']:>16.1f}% | {diff_char:>+15.1f}%")
    print(f"{'1-char Error Rate':<32} | {old_results['error_1_char_pct']:>16.1f}% | {v2_results['error_1_char_pct']:>16.1f}% | {old_results['error_1_char_pct'] - v2_results['error_1_char_pct']:>+15.1f}%")
    print(f"{'2-char Error Rate':<32} | {old_results['error_2_char_pct']:>16.1f}% | {v2_results['error_2_char_pct']:>16.1f}% | {old_results['error_2_char_pct'] - v2_results['error_2_char_pct']:>+15.1f}%")
    print(f"{'Catastrophic (3+ Char) Error':<32} | {old_results['catastrophic_error_3plus_pct']:>16.1f}% | {v2_results['catastrophic_error_3plus_pct']:>16.1f}% | {-diff_cat:>+15.1f}% (lower=better)")
    print(f"{'Wrong State Prefix Rate':<32} | {old_results['wrong_state_pct']:>16.1f}% | {v2_results['wrong_state_pct']:>16.1f}% | {-diff_state:>+15.1f}% (lower=better)")
    print(f"{'Median Latency':<32} | {old_results['median_latency_ms']:>16.1f}ms | {v2_results['median_latency_ms']:>16.1f}ms | {-diff_med_lat:>+15.1f}ms")
    print(f"{'P95 Latency':<32} | {old_results['p95_latency_ms']:>16.1f}ms | {v2_results['p95_latency_ms']:>16.1f}ms | {old_results['p95_latency_ms'] - v2_results['p95_latency_ms']:>+15.1f}ms")
    print("=" * 80)

    # Save to JSON
    output_payload = {
        "timestamp": time.time(),
        "total_test_samples": len(samples),
        "old_pipeline": old_results,
        "v2_pipeline": v2_results
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Save to CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Old_CRNN", "ANPR_V2_PPOCR", "Difference"])
        writer.writerow(["Exact_Match_Pct", old_results["exact_match_pct"], v2_results["exact_match_pct"], diff_exact])
        writer.writerow(["Char_Accuracy_Pct", old_results["mean_character_accuracy_pct"], v2_results["mean_character_accuracy_pct"], diff_char])
        writer.writerow(["Catastrophic_3plus_Pct", old_results["catastrophic_error_3plus_pct"], v2_results["catastrophic_error_3plus_pct"], -diff_cat])
        writer.writerow(["Wrong_State_Pct", old_results["wrong_state_pct"], v2_results["wrong_state_pct"], -diff_state])
        writer.writerow(["Median_Latency_Ms", old_results["median_latency_ms"], v2_results["median_latency_ms"], -diff_med_lat])
        writer.writerow(["P95_Latency_Ms", old_results["p95_latency_ms"], v2_results["p95_latency_ms"], old_results["p95_latency_ms"] - v2_results["p95_latency_ms"]])

    print(f"\nPersisted A/B results to {RESULTS_JSON} and {RESULTS_CSV}")

if __name__ == "__main__":
    run_ab_benchmark()
