#!/usr/bin/env python3
"""
benchmarks/anpr_v2/benchmark_ocr_comparison.py
Formal Benchmark & Evaluation: Current PP-OCRv4 (RapidOCR) vs. FastPlateOCR (CCT-S-v2 Global).

Evaluates on the exact 228 held-out Indian plate crops dataset under identical conditions.
Calculates all required metrics:
- Exact full-plate accuracy (Raw & Post-Processed)
- Mean character accuracy (Levenshtein similarity)
- Edit distance distribution (0, 1, 2, 3+)
- Empty and invalid Indian plate counts
- Inference latency (Mean, Median, P95, Plates/sec)
- Indian plate component accuracy (State, District, Series, Number)
- Character-level confusion matrix
- Complementarity and Oracle theoretical maximum
- Single-line vs. Two-line plate performance
- Condition-level breakdown (Clean, Blur, Dim, Contrast, JPEG, Skew)
- Test A (Original Crop) vs. Test B (Rectified / Preprocessed Crop)
"""

import os
import sys
import json
import time
import csv
import re
import platform
import psutil
import cv2
import numpy as np
from collections import Counter
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.recognizer import get_recognizer
from anpr_v2.fastplate_recognizer import FastPlateRecognizer
from anpr_v2.rectifier import rectify_plate_perspective, normalize_plate_resolution
from anpr_v2.validator import validate_indian_registration, clean_ocr_raw_tokens

BENCHMARK_INDEX = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ground_truth_benchmark.json")
CROPS_DIR = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "crops")
OUTPUT_DIR = os.path.join(BASE_DIR, "ocr_comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex pattern for Indian plate components
INDIAN_COMPONENT_PATTERN = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{0,4})$")

def get_system_hardware_info() -> Dict[str, Any]:
    try:
        import onnxruntime as ort
        ort_ver = ort.__version__
        ort_prov = ort.get_available_providers()
    except Exception:
        ort_ver = "Unknown"
        ort_prov = ["CPUExecutionProvider"]

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "platform": platform.platform(),
        "cpu_processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": sys.version.split()[0],
        "onnxruntime_version": ort_ver,
        "execution_providers": ort_prov,
        "gpu_available": "CUDAExecutionProvider" in ort_prov,
    }

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

def align_strings(s1: str, s2: str) -> Tuple[str, str]:
    """
    Standard Needleman-Wunsch / Levenshtein string alignment.
    Returns (aligned_s1, aligned_s2) with '-' for gaps.
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    a1, a2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if s1[i - 1] == s2[j - 1] else 1):
            a1.append(s1[i - 1])
            a2.append(s2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            a1.append(s1[i - 1])
            a2.append('-')
            i -= 1
        else:
            a1.append('-')
            a2.append(s2[j - 1])
            j -= 1
    return ''.join(reversed(a1)), ''.join(reversed(a2))

def parse_plate_components(plate_str: str) -> Dict[str, str]:
    clean = clean_ocr_raw_tokens(plate_str)
    m = INDIAN_COMPONENT_PATTERN.match(clean)
    if m:
        return {
            "state": m.group(1) or "",
            "district": m.group(2) or "",
            "series": m.group(3) or "",
            "number": m.group(4) or ""
        }
    return {
        "state": clean[:2] if len(clean) >= 2 else "",
        "district": "",
        "series": "",
        "number": ""
    }

def evaluate_model_on_dataset(
    recognizer,
    samples: List[Dict[str, Any]],
    model_name: str,
    use_rectification: bool = False
) -> Dict[str, Any]:
    """
    Evaluates recognizer over all samples.
    Measures latency per image (after warm-up).
    Computes exact accuracy, character accuracy, confusions, component accuracy, etc.
    """
    records = []
    latencies = []
    confusions = Counter()

    exact_matches = 0
    err_1 = 0
    err_2 = 0
    err_3plus = 0
    empty_count = 0
    invalid_indian_count = 0

    state_correct_count = 0
    dist_correct_count = 0
    series_correct_count = 0
    num_correct_count = 0
    valid_component_samples = 0

    post_processed_exact = 0

    print(f"\n[Benchmarking {model_name}] (Preprocessing: {'ENABLED' if use_rectification else 'RAW CROP'}) on {len(samples)} crops...")

    for idx, item in enumerate(samples):
        crop_file = item["crop_file"]
        crop_path = os.path.join(CROPS_DIR, crop_file)
        if not os.path.exists(crop_path):
            crop_path = item["crop_path"]

        im = cv2.imread(crop_path)
        if im is None:
            continue

        gt_raw = item["ground_truth"]
        gt_clean = clean_ocr_raw_tokens(gt_raw)

        # Apply preprocessing if specified (Test B)
        if use_rectification:
            im_rect, was_rect, _ = rectify_plate_perspective(im)
            im_input = normalize_plate_resolution(im_rect)
        else:
            im_input = im

        # Timed inference
        t0 = time.perf_counter()
        res = recognizer.recognize(im_input)
        t_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_ms)

        pred_clean = clean_ocr_raw_tokens(res.text)
        dist = levenshtein_distance(pred_clean, gt_clean)
        char_acc = character_accuracy(pred_clean, gt_clean)
        is_exact = (pred_clean == gt_clean)

        # Post-processed validation check
        val_res = validate_indian_registration(pred_clean, confidence=res.confidence)
        if val_res.is_valid and pred_clean == gt_clean:
            post_processed_exact += 1

        if not val_res.is_valid:
            invalid_indian_count += 1

        if len(pred_clean) == 0:
            empty_count += 1

        if is_exact:
            exact_matches += 1
        elif dist == 1:
            err_1 += 1
        elif dist == 2:
            err_2 += 1
        else:
            err_3plus += 1

        # Character confusion alignment
        aligned_gt, aligned_pred = align_strings(gt_clean, pred_clean)
        for c_gt, c_pred in zip(aligned_gt, aligned_pred):
            if c_gt != c_pred:
                if c_gt != '-' and c_pred != '-':
                    confusions[f"{c_gt} -> {c_pred}"] += 1
                elif c_gt != '-' and c_pred == '-':
                    confusions[f"{c_gt} -> [OMIT]"] += 1
                elif c_gt == '-' and c_pred != '-':
                    confusions[f"[INSERT] -> {c_pred}"] += 1

        # Component accuracy breakdown
        gt_comp = parse_plate_components(gt_clean)
        pred_comp = parse_plate_components(pred_clean)
        if gt_comp["state"]:
            valid_component_samples += 1
            if pred_comp["state"] == gt_comp["state"]:
                state_correct_count += 1
            if gt_comp["district"] and pred_comp["district"] == gt_comp["district"]:
                dist_correct_count += 1
            if gt_comp["series"] and pred_comp["series"] == gt_comp["series"]:
                series_correct_count += 1
            if gt_comp["number"] and pred_comp["number"] == gt_comp["number"]:
                num_correct_count += 1

        h, w = im.shape[:2]
        ar = round(w / float(max(1, h)), 2)
        is_two_row = ar < 2.0

        records.append({
            "crop_file": crop_file,
            "source": item.get("source", "Unknown"),
            "ground_truth": gt_clean,
            "predicted": pred_clean,
            "raw_text": res.raw_text,
            "confidence": round(res.confidence, 4),
            "exact_match": is_exact,
            "post_processed_valid": val_res.is_valid,
            "edit_distance": dist,
            "char_accuracy": round(char_acc, 4),
            "latency_ms": round(t_ms, 2),
            "aspect_ratio": ar,
            "is_two_row": is_two_row,
            "engine": res.engine
        })

    n = max(1, len(records))
    lat_sorted = sorted(latencies)
    mean_lat = np.mean(lat_sorted) if lat_sorted else 0.0
    median_lat = np.median(lat_sorted) if lat_sorted else 0.0
    p95_lat = np.percentile(lat_sorted, 95) if lat_sorted else 0.0
    fps = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

    edit_distances = [r["edit_distance"] for r in records]
    avg_edit_dist = round(float(np.mean(edit_distances)), 3)
    med_edit_dist = round(float(np.median(edit_distances)), 1)
    mean_char_acc = round(float(np.mean([r["char_accuracy"] for r in records])) * 100.0, 2)

    # Sub-category accuracy: Two-line vs Single-line
    two_row_recs = [r for r in records if r["is_two_row"]]
    single_row_recs = [r for r in records if not r["is_two_row"]]
    two_row_exact = sum(1 for r in two_row_recs if r["exact_match"])
    single_row_exact = sum(1 for r in single_row_recs if r["exact_match"])

    # Condition breakdown
    condition_stats = {}
    sources = set(r["source"] for r in records)
    for s in sorted(sources):
        sub = [r for r in records if r["source"] == s]
        s_exact = sum(1 for r in sub if r["exact_match"])
        s_char_acc = round(float(np.mean([r["char_accuracy"] for r in sub])) * 100.0, 2)
        condition_stats[s] = {
            "total": len(sub),
            "exact_matches": s_exact,
            "exact_pct": round((s_exact / max(1, len(sub))) * 100.0, 2),
            "mean_char_accuracy_pct": s_char_acc
        }

    return {
        "model_name": model_name,
        "use_rectification": use_rectification,
        "total_samples": n,
        "exact_matches": exact_matches,
        "exact_accuracy_pct": round((exact_matches / n) * 100.0, 2),
        "post_processed_exact": post_processed_exact,
        "post_processed_exact_pct": round((post_processed_exact / n) * 100.0, 2),
        "mean_character_accuracy_pct": mean_char_acc,
        "avg_edit_distance": avg_edit_dist,
        "median_edit_distance": med_edit_dist,
        "error_1_char_count": err_1,
        "error_1_char_pct": round((err_1 / n) * 100.0, 2),
        "error_2_char_count": err_2,
        "error_2_char_pct": round((err_2 / n) * 100.0, 2),
        "catastrophic_error_3plus_count": err_3plus,
        "catastrophic_error_3plus_pct": round((err_3plus / n) * 100.0, 2),
        "empty_predictions_count": empty_count,
        "empty_predictions_pct": round((empty_count / n) * 100.0, 2),
        "invalid_indian_plates_count": invalid_indian_count,
        "invalid_indian_plates_pct": round((invalid_indian_count / n) * 100.0, 2),
        "latency": {
            "mean_ms": round(float(mean_lat), 2),
            "median_ms": round(float(median_lat), 2),
            "p95_ms": round(float(p95_lat), 2),
            "plates_per_second": round(float(fps), 1)
        },
        "components": {
            "total_evaluated": valid_component_samples,
            "state_code_accuracy_pct": round((state_correct_count / max(1, valid_component_samples)) * 100.0, 2),
            "district_code_accuracy_pct": round((dist_correct_count / max(1, valid_component_samples)) * 100.0, 2),
            "series_accuracy_pct": round((series_correct_count / max(1, valid_component_samples)) * 100.0, 2),
            "number_accuracy_pct": round((num_correct_count / max(1, valid_component_samples)) * 100.0, 2),
        },
        "row_layout": {
            "single_line": {
                "count": len(single_row_recs),
                "exact": single_row_exact,
                "exact_pct": round((single_row_exact / max(1, len(single_row_recs))) * 100.0, 2)
            },
            "two_line": {
                "count": len(two_row_recs),
                "exact": two_row_exact,
                "exact_pct": round((two_row_exact / max(1, len(two_row_recs))) * 100.0, 2)
            }
        },
        "conditions": condition_stats,
        "top_confusions": confusions.most_common(25),
        "records": records
    }

def run_ocr_comparison():
    print("=" * 80)
    print("      OCR MODEL BENCHMARK: CURRENT PP-OCRv4 vs. CANDIDATE FASTPLATEOCR")
    print("=" * 80)

    # 1. System & Hardware Details
    sys_info = get_system_hardware_info()
    print("\n--- System & Hardware Information ---")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    # 2. Dataset Load
    if not os.path.exists(BENCHMARK_INDEX):
        raise FileNotFoundError(f"Benchmark index not found at {BENCHMARK_INDEX}")
    with open(BENCHMARK_INDEX, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"\nLoaded {len(samples)} benchmark samples from {BENCHMARK_INDEX}")

    # 3. Initialize Models
    print("\n--- Initializing OCR Models ---")
    t0_curr_init = time.perf_counter()
    current_recognizer = get_recognizer(backend="paddleocr")
    curr_init_ms = round((time.perf_counter() - t0_curr_init) * 1000.0, 2)

    t0_fp_init = time.perf_counter()
    fastplate_recognizer = FastPlateRecognizer(hub_ocr_model="cct-s-v2-global-model", device="cpu")
    fp_init_ms = round((time.perf_counter() - t0_fp_init) * 1000.0, 2)

    # 4. Warm-up Phase (10 inferences each)
    print("\n--- Running Warm-Up Phase (10 inferences per model) ---")
    dummy_img = cv2.imread(os.path.join(CROPS_DIR, samples[0]["crop_file"]))
    if dummy_img is None:
        dummy_img = np.zeros((48, 128, 3), dtype=np.uint8)
    for _ in range(10):
        current_recognizer.recognize(dummy_img)
        fastplate_recognizer.recognize(dummy_img)
    print("Warm-up complete. Steady-state performance initialized.")

    # 5. Run Evaluations
    # Test A: Raw Crops
    curr_raw_eval = evaluate_model_on_dataset(current_recognizer, samples, "CURRENT (PP-OCRv4)", use_rectification=False)
    fp_raw_eval = evaluate_model_on_dataset(fastplate_recognizer, samples, "FASTPLATEOCR (CCT-S-v2)", use_rectification=False)

    # Test B: Preprocessed / Rectified Crops
    curr_prep_eval = evaluate_model_on_dataset(current_recognizer, samples, "CURRENT (PP-OCRv4 + Rectified)", use_rectification=True)
    fp_prep_eval = evaluate_model_on_dataset(fastplate_recognizer, samples, "FASTPLATEOCR (CCT-S-v2 + Rectified)", use_rectification=True)

    # 6. Complementarity Analysis (on Raw Test A)
    curr_recs = curr_raw_eval["records"]
    fp_recs = fp_raw_eval["records"]

    curr_only_correct = 0
    fp_only_correct = 0
    both_correct = 0
    neither_correct = 0

    per_image_comparison = []
    errors_curr = []
    errors_fp = []

    for i in range(len(samples)):
        c_rec = curr_recs[i]
        f_rec = fp_recs[i]
        gt = c_rec["ground_truth"]

        c_ok = c_rec["exact_match"]
        f_ok = f_rec["exact_match"]

        if c_ok and f_ok:
            both_correct += 1
            winner = "TIE_CORRECT"
        elif c_ok and not f_ok:
            curr_only_correct += 1
            winner = "CURRENT"
        elif not c_ok and f_ok:
            fp_only_correct += 1
            winner = "FASTPLATE"
        else:
            neither_correct += 1
            winner = "TIE_WRONG"

        row = {
            "filename": c_rec["crop_file"],
            "source": c_rec["source"],
            "ground_truth": gt,
            "current_prediction": c_rec["predicted"],
            "current_correct": c_ok,
            "current_char_accuracy": c_rec["char_accuracy"],
            "current_latency_ms": c_rec["latency_ms"],
            "fastplate_prediction": f_rec["predicted"],
            "fastplate_correct": f_ok,
            "fastplate_char_accuracy": f_rec["char_accuracy"],
            "fastplate_latency_ms": f_rec["latency_ms"],
            "winner": winner
        }
        per_image_comparison.append(row)

        if not c_ok:
            errors_curr.append({
                "filename": c_rec["crop_file"],
                "source": c_rec["source"],
                "ground_truth": gt,
                "predicted": c_rec["predicted"],
                "edit_distance": c_rec["edit_distance"],
                "char_accuracy": c_rec["char_accuracy"]
            })
        if not f_ok:
            errors_fp.append({
                "filename": f_rec["crop_file"],
                "source": f_rec["source"],
                "ground_truth": gt,
                "predicted": f_rec["predicted"],
                "edit_distance": f_rec["edit_distance"],
                "char_accuracy": f_rec["char_accuracy"]
            })

    total_n = len(samples)
    oracle_count = both_correct + curr_only_correct + fp_only_correct
    oracle_pct = round((oracle_count / total_n) * 100.0, 2)

    # 7. Print Main Results Table
    diff_exact = round(fp_raw_eval["exact_accuracy_pct"] - curr_raw_eval["exact_accuracy_pct"], 2)
    diff_char = round(fp_raw_eval["mean_character_accuracy_pct"] - curr_raw_eval["mean_character_accuracy_pct"], 2)
    diff_edit = round(fp_raw_eval["avg_edit_distance"] - curr_raw_eval["avg_edit_distance"], 3)
    diff_lat = round(fp_raw_eval["latency"]["mean_ms"] - curr_raw_eval["latency"]["mean_ms"], 2)

    print("\n" + "=" * 80)
    print("                     MAIN BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'METRIC':<34} | {'CURRENT (PP-OCRv4)':<20} | {'FASTPLATEOCR':<20} | {'DIFFERENCE':<16}")
    print("-" * 80)
    print(f"{'Exact Plate Accuracy':<34} | {curr_raw_eval['exact_accuracy_pct']:>19.2f}% | {fp_raw_eval['exact_accuracy_pct']:>19.2f}% | {diff_exact:>+15.2f}%")
    print(f"{'Post-Processed Exact Accuracy':<34} | {curr_raw_eval['post_processed_exact_pct']:>19.2f}% | {fp_raw_eval['post_processed_exact_pct']:>19.2f}% | {round(fp_raw_eval['post_processed_exact_pct'] - curr_raw_eval['post_processed_exact_pct'], 2):>+15.2f}%")
    print(f"{'Mean Character Accuracy':<34} | {curr_raw_eval['mean_character_accuracy_pct']:>19.2f}% | {fp_raw_eval['mean_character_accuracy_pct']:>19.2f}% | {diff_char:>+15.2f}%")
    print(f"{'Average Edit Distance':<34} | {curr_raw_eval['avg_edit_distance']:>20.3f} | {fp_raw_eval['avg_edit_distance']:>20.3f} | {diff_edit:>+16.3f}")
    print(f"{'Median Edit Distance':<34} | {curr_raw_eval['median_edit_distance']:>20.1f} | {fp_raw_eval['median_edit_distance']:>20.1f} | {round(fp_raw_eval['median_edit_distance'] - curr_raw_eval['median_edit_distance'], 1):>+16.1f}")
    print(f"{'1-Character Error Rate':<34} | {curr_raw_eval['error_1_char_pct']:>19.2f}% | {fp_raw_eval['error_1_char_pct']:>19.2f}% | {round(fp_raw_eval['error_1_char_pct'] - curr_raw_eval['error_1_char_pct'], 2):>+15.2f}%")
    print(f"{'2-Character Error Rate':<34} | {curr_raw_eval['error_2_char_pct']:>19.2f}% | {fp_raw_eval['error_2_char_pct']:>19.2f}% | {round(fp_raw_eval['error_2_char_pct'] - curr_raw_eval['error_2_char_pct'], 2):>+15.2f}%")
    print(f"{'3+ Character Error Rate':<34} | {curr_raw_eval['catastrophic_error_3plus_pct']:>19.2f}% | {fp_raw_eval['catastrophic_error_3plus_pct']:>19.2f}% | {round(fp_raw_eval['catastrophic_error_3plus_pct'] - curr_raw_eval['catastrophic_error_3plus_pct'], 2):>+15.2f}%")
    print(f"{'Empty Predictions':<34} | {curr_raw_eval['empty_predictions_count']:>20} | {fp_raw_eval['empty_predictions_count']:>20} | {fp_raw_eval['empty_predictions_count'] - curr_raw_eval['empty_predictions_count']:>+16}")
    print(f"{'Invalid Indian Plates':<34} | {curr_raw_eval['invalid_indian_plates_count']:>20} | {fp_raw_eval['invalid_indian_plates_count']:>20} | {fp_raw_eval['invalid_indian_plates_count'] - curr_raw_eval['invalid_indian_plates_count']:>+16}")
    print(f"{'Mean Latency (ms)':<34} | {curr_raw_eval['latency']['mean_ms']:>18.2f}ms | {fp_raw_eval['latency']['mean_ms']:>18.2f}ms | {diff_lat:>+14.2f}ms")
    print(f"{'Median Latency (ms)':<34} | {curr_raw_eval['latency']['median_ms']:>18.2f}ms | {fp_raw_eval['latency']['median_ms']:>18.2f}ms | {round(fp_raw_eval['latency']['median_ms'] - curr_raw_eval['latency']['median_ms'], 2):>+14.2f}ms")
    print(f"{'P95 Latency (ms)':<34} | {curr_raw_eval['latency']['p95_ms']:>18.2f}ms | {fp_raw_eval['latency']['p95_ms']:>18.2f}ms | {round(fp_raw_eval['latency']['p95_ms'] - curr_raw_eval['latency']['p95_ms'], 2):>+14.2f}ms")
    print(f"{'Plates / Second (Throughput)':<34} | {curr_raw_eval['latency']['plates_per_second']:>18.1f}/s | {fp_raw_eval['latency']['plates_per_second']:>18.1f}/s | {round(fp_raw_eval['latency']['plates_per_second'] - curr_raw_eval['latency']['plates_per_second'], 1):>+14.1f}/s")
    print("=" * 80)

    # 8. Category & 90% Requirement
    if diff_exact >= 20.0:
        category = "MAJOR IMPROVEMENT"
    elif diff_exact >= 10.0:
        category = "MODERATE IMPROVEMENT"
    elif diff_exact >= 3.0:
        category = "SMALL IMPROVEMENT"
    elif diff_exact >= -3.0:
        category = "NO MEANINGFUL IMPROVEMENT"
    else:
        category = "REGRESSION"

    req_curr_raw_pass = curr_raw_eval["exact_accuracy_pct"] >= 90.0
    req_fp_raw_pass = fp_raw_eval["exact_accuracy_pct"] >= 90.0
    req_fp_post_pass = fp_raw_eval["post_processed_exact_pct"] >= 90.0

    print("\n--- Evaluation Criteria & 90% Requirement ---")
    print(f"  Engineering Category: {category}")
    print(f"  CURRENT RAW (>= 90%): {'PASS' if req_curr_raw_pass else 'FAIL'} ({curr_raw_eval['exact_accuracy_pct']}%)")
    print(f"  FASTPLATE RAW (>= 90%): {'PASS' if req_fp_raw_pass else 'FAIL'} ({fp_raw_eval['exact_accuracy_pct']}%)")
    print(f"  FASTPLATE POST-PROCESSED (>= 90%): {'PASS' if req_fp_post_pass else 'FAIL'} ({fp_raw_eval['post_processed_exact_pct']}%)")

    # 9. Complementarity Summary
    print("\n--- Complementarity & Oracle Upper Bound ---")
    print(f"  Correct only by CURRENT:   {curr_only_correct} ({round(curr_only_correct/total_n*100, 1)}%)")
    print(f"  Correct only by FASTPLATE: {fp_only_correct} ({round(fp_only_correct/total_n*100, 1)}%)")
    print(f"  Correct by BOTH:           {both_correct} ({round(both_correct/total_n*100, 1)}%)")
    print(f"  Wrong by BOTH:             {neither_correct} ({round(neither_correct/total_n*100, 1)}%)")
    print(f"  Oracle Theoretical Max:    {oracle_count} / {total_n} ({oracle_pct}%)")

    # 10. Preprocessing Comparison (Test A vs Test B)
    prep_diff_fp = round(fp_prep_eval["exact_accuracy_pct"] - fp_raw_eval["exact_accuracy_pct"], 2)
    if prep_diff_fp > 1.0:
        prep_impact_fp = "IMPROVES"
    elif prep_diff_fp < -1.0:
        prep_impact_fp = "DEGRADES"
    else:
        prep_impact_fp = "DOES NOT SIGNIFICANTLY CHANGE"

    print("\n--- Preprocessing Impact (Test A: Raw vs. Test B: Rectified/Preprocessed) ---")
    print(f"  CURRENT: Raw={curr_raw_eval['exact_accuracy_pct']}% vs Rectified={curr_prep_eval['exact_accuracy_pct']}% (diff={round(curr_prep_eval['exact_accuracy_pct'] - curr_raw_eval['exact_accuracy_pct'], 2):+g}%)")
    print(f"  FASTPLATE: Raw={fp_raw_eval['exact_accuracy_pct']}% vs Rectified={fp_prep_eval['exact_accuracy_pct']}% (diff={prep_diff_fp:+g}% -> {prep_impact_fp})")

    # 11. Save CSV and JSON Artifacts
    summary_data = {
        "timestamp": time.time(),
        "system_info": sys_info,
        "models": {
            "current": {
                "name": "PP-OCRv4 / RapidOCR",
                "init_time_ms": curr_init_ms,
                "raw_evaluation": {k: v for k, v in curr_raw_eval.items() if k != "records"},
                "preprocessed_evaluation": {k: v for k, v in curr_prep_eval.items() if k != "records"}
            },
            "candidate": {
                "name": "FastPlateOCR (cct-s-v2-global-model)",
                "init_time_ms": fp_init_ms,
                "execution_providers": getattr(fastplate_recognizer, "execution_providers", []),
                "onnx_model_path": getattr(fastplate_recognizer, "onnx_model_path", ""),
                "plate_config_path": getattr(fastplate_recognizer, "plate_config_path", ""),
                "raw_evaluation": {k: v for k, v in fp_raw_eval.items() if k != "records"},
                "preprocessed_evaluation": {k: v for k, v in fp_prep_eval.items() if k != "records"}
            }
        },
        "comparison": {
            "exact_accuracy_diff_pct": diff_exact,
            "mean_char_acc_diff_pct": diff_char,
            "avg_edit_distance_diff": diff_edit,
            "mean_latency_diff_ms": diff_lat,
            "category": category,
            "target_90_pct": {
                "current_raw": "PASS" if req_curr_raw_pass else "FAIL",
                "fastplate_raw": "PASS" if req_fp_raw_pass else "FAIL",
                "fastplate_post_processed": "PASS" if req_fp_post_pass else "FAIL",
                "fastplate_gap_percentage_points": round(90.0 - fp_raw_eval["exact_accuracy_pct"], 2)
            },
            "complementarity": {
                "current_only_correct": curr_only_correct,
                "fastplate_only_correct": fp_only_correct,
                "both_correct": both_correct,
                "neither_correct": neither_correct,
                "oracle_theoretical_count": oracle_count,
                "oracle_theoretical_pct": oracle_pct
            },
            "preprocessing_impact": {
                "fastplate_diff_pct": prep_diff_fp,
                "assessment": prep_impact_fp
            }
        }
    }

    # Save summary.json
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved {summary_path}")

    # Save comparison.csv
    comp_csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename", "source", "ground_truth",
            "current_prediction", "current_correct", "current_char_accuracy", "current_latency_ms",
            "fastplate_prediction", "fastplate_correct", "fastplate_char_accuracy", "fastplate_latency_ms",
            "winner"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_comparison)
    print(f"Saved {comp_csv_path}")

    # Save errors_current.csv
    err_curr_path = os.path.join(OUTPUT_DIR, "errors_current.csv")
    with open(err_curr_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "source", "ground_truth", "predicted", "edit_distance", "char_accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors_curr)
    print(f"Saved {err_curr_path} ({len(errors_curr)} errors)")

    # Save errors_fastplate.csv
    err_fp_path = os.path.join(OUTPUT_DIR, "errors_fastplate.csv")
    with open(err_fp_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "source", "ground_truth", "predicted", "edit_distance", "char_accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors_fp)
    print(f"Saved {err_fp_path} ({len(errors_fp)} errors)")

    # Save confusion_analysis.csv
    conf_csv_path = os.path.join(OUTPUT_DIR, "confusion_analysis.csv")
    with open(conf_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Current_Confusion", "Current_Count", "FastPlate_Confusion", "FastPlate_Count"])
        c_top = curr_raw_eval["top_confusions"]
        f_top = fp_raw_eval["top_confusions"]
        max_rows = max(len(c_top), len(f_top))
        for r in range(max_rows):
            c_pair = c_top[r] if r < len(c_top) else ("", "")
            f_pair = f_top[r] if r < len(f_top) else ("", "")
            writer.writerow([r + 1, c_pair[0], c_pair[1], f_pair[0], f_pair[1]])
    print(f"Saved {conf_csv_path}")

    return summary_data

if __name__ == "__main__":
    run_ocr_comparison()
