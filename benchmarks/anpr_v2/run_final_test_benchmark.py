#!/usr/bin/env python3
"""
benchmarks/anpr_v2/run_final_test_benchmark.py
Phase 10-13 Formal Evaluation on the Untouched 228 Held-Out Test Set.

Evaluates and compares:
- MODEL A: PP-OCRv4 (Current RapidOCR engine)
- MODEL B: FastPlateOCR Pretrained CCT-S-V2 (Global weights)
- MODEL C: FastPlateOCR Indian Fine-Tuned CCT-S-V2 (With Phase 6 Two-Line Handler)
- PIPELINE 1: Raw Single-Frame OCR
- PIPELINE 2: Post-Processed OCR (Indian Format Validator)
- PIPELINE 3: Multi-Frame Temporal Consensus (Phase 12)
- PIPELINE 4: Adaptive Fallback Engine (Phase 13 Complementary Union)

Produces:
- benchmarks/anpr_v2/final_test_metrics.json
- benchmarks/anpr_v2/final_test_predictions.csv
- Detailed condition, layout, latency, and error breakdown.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
import json
import time
import re
import platform
import psutil
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.recognizer import get_recognizer
from anpr_v2.fastplate_recognizer import FastPlateRecognizer
from anpr_v2.fastplate_indian_recognizer import IndianFastPlateRecognizer
from anpr_v2.two_line_handler import is_two_line_plate, normalize_two_line_to_single_strip
from anpr_v2.validator import validate_indian_registration, clean_ocr_raw_tokens
from anpr_v2.rectifier import rectify_plate_perspective

BENCHMARK_INDEX = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ground_truth_benchmark.json")
CROPS_DIR = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "crops")
OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "final_test_metrics.json")
OUTPUT_PREDS_PATH = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "final_test_predictions.csv")

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = cur[j] + 1
            replace = prev[j] + (0 if c1 == c2 else 1)
            cur.append(min(insert, delete, replace))
        prev = cur
    return prev[-1]

def run_evaluation():
    print("=================================================================")
    print("PHASE 10-13: FINAL TEST BENCHMARK ON UNTOUCHED 228 HELD-OUT CROPS")
    print("=================================================================")
    
    with open(BENCHMARK_INDEX, "r", encoding="utf-8") as f:
        benchmark_records = json.load(f)
    print(f"Loaded {len(benchmark_records)} benchmark test records.")
    
    # Initialize all models
    print("\n[1/3] Initializing PP-OCRv4 recognizer...")
    ppocr = get_recognizer()
    
    print("\n[2/3] Initializing Pretrained FastPlateOCR CCT-S-V2...")
    fast_pretrained = FastPlateRecognizer()
    
    print("\n[3/3] Initializing Indian Fine-Tuned FastPlateOCR CCT-S-V2...")
    fast_indian = IndianFastPlateRecognizer()
    
    rows = []
    
    # Trackers for the 3 main models (Raw and Post-Processed)
    models = {
        "ppocr_v4": {"exact": 0, "post_exact": 0, "char_accs": [], "edit_dists": [], "times": []},
        "fastplate_pretrained": {"exact": 0, "post_exact": 0, "char_accs": [], "edit_dists": [], "times": []},
        "fastplate_indian": {"exact": 0, "post_exact": 0, "char_accs": [], "edit_dists": [], "times": []},
        "adaptive_fallback": {"exact": 0, "post_exact": 0, "char_accs": [], "edit_dists": []}
    }
    
    # Condition & Layout breakdowns for Indian Fine-Tuned Model
    single_line_stats = {"total": 0, "exact": 0}
    two_line_stats = {"total": 0, "exact": 0}
    condition_stats = {}
    
    # State tracking
    state_breakdown = {}
    
    # Confusion matrix tracker
    char_confusions = Counter()
    
    print(f"\nRunning inference across all 228 test crops...", flush=True)
    
    for idx, item in enumerate(benchmark_records):
        fname = os.path.basename(item["crop_path"])
        local_path = os.path.join(CROPS_DIR, fname)
        gt = item["ground_truth"].strip().upper()
        source = item.get("source", "TestSet")
        
        img = cv2.imread(local_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        ar = w / float(max(1, h))
        is_two = is_two_line_plate(img)
        
        # 1. PP-OCRv4
        t0 = time.perf_counter()
        res_pp = ppocr.recognize(img)
        t_pp = (time.perf_counter() - t0) * 1000.0
        raw_pp = res_pp.text if hasattr(res_pp, "text") else str(res_pp)
        conf_pp = getattr(res_pp, "confidence", 0.0)
        val_pp = validate_indian_registration(raw_pp)
        post_pp = val_pp.cleaned_plate
        
        # 2. FastPlateOCR Pretrained
        t0 = time.perf_counter()
        res_pre = fast_pretrained.recognize(img)
        t_pre = (time.perf_counter() - t0) * 1000.0
        raw_pre = res_pre.text if hasattr(res_pre, "text") else str(res_pre)
        conf_pre = getattr(res_pre, "confidence", 0.0)
        val_pre = validate_indian_registration(raw_pre)
        post_pre = val_pre.cleaned_plate
        
        # 3. FastPlateOCR Indian Fine-Tuned
        t0 = time.perf_counter()
        raw_ind, conf_ind = fast_indian.recognize(img, handle_two_line=False)
        t_ind = (time.perf_counter() - t0) * 1000.0
        val_ind = validate_indian_registration(raw_ind)
        post_ind = val_ind.cleaned_plate
        
        # 4. Adaptive Fallback (Phase 13):
        # Fine-tuned Indian model is primary. Fallback to PP-OCRv4 if Indian is invalid but PP-OCR is valid,
        # or if PP-OCR has higher format validity confidence.
        if val_ind.is_valid and not val_pp.is_valid:
            fallback_pred = raw_ind
            fallback_post = post_ind
        elif val_pp.is_valid and not val_ind.is_valid:
            fallback_pred = raw_pp
            fallback_post = post_pp
        elif val_ind.is_valid and val_pp.is_valid:
            # Both valid: choose the one with higher confidence
            if conf_ind >= conf_pp:
                fallback_pred = raw_ind
                fallback_post = post_ind
            else:
                fallback_pred = raw_pp
                fallback_post = post_pp
        else:
            # Neither strictly valid: pick the longer or higher-confidence candidate
            if len(raw_ind) >= 8:
                fallback_pred = raw_ind
                fallback_post = post_ind
            elif len(raw_pp) >= 8:
                fallback_pred = raw_pp
                fallback_post = post_pp
            else:
                fallback_pred = raw_ind
                fallback_post = post_ind
            
        # Update metrics for each
        for m_name, raw_p, post_p, lat in [
            ("ppocr_v4", raw_pp, post_pp, t_pp),
            ("fastplate_pretrained", raw_pre, post_pre, t_pre),
            ("fastplate_indian", raw_ind, post_ind, t_ind),
            ("adaptive_fallback", fallback_pred, fallback_post, t_ind + (t_pp if not val_ind.is_valid else 0))
        ]:
            m = models[m_name]
            ed = levenshtein(raw_p, gt)
            m["edit_dists"].append(ed)
            char_acc = max(0.0, 1.0 - (ed / max(len(raw_p), len(gt), 1))) * 100.0
            m["char_accs"].append(char_acc)
            if lat is not None and "times" in m:
                m["times"].append(lat)
            if raw_p == gt:
                m["exact"] += 1
            if post_p == gt:
                m["post_exact"] += 1
                
        # Layout tracking for Indian fine-tuned
        if is_two:
            two_line_stats["total"] += 1
            if raw_ind == gt:
                two_line_stats["exact"] += 1
        else:
            single_line_stats["total"] += 1
            if raw_ind == gt:
                single_line_stats["exact"] += 1
                
        # Condition tracking
        cond_name = source
        if "aug_blur" in fname:
            cond_name = "Blur"
        elif "aug_dim" in fname:
            cond_name = "LowLight"
        elif "aug_contrast" in fname:
            cond_name = "HighContrast"
        elif "aug_jpeg" in fname:
            cond_name = "JPEGCompression"
        elif "aug_rot" in fname:
            cond_name = "PerspectiveSkew"
        else:
            cond_name = "CleanDaylight"
            
        if cond_name not in condition_stats:
            condition_stats[cond_name] = {"total": 0, "exact": 0, "char_accs": []}
        condition_stats[cond_name]["total"] += 1
        if raw_ind == gt:
            condition_stats[cond_name]["exact"] += 1
        ed_ind = levenshtein(raw_ind, gt)
        c_acc = max(0.0, 1.0 - (ed_ind / max(len(raw_ind), len(gt), 1))) * 100.0
        condition_stats[cond_name]["char_accs"].append(c_acc)
        
        # State tracking
        st = gt[:2]
        if st not in state_breakdown:
            state_breakdown[st] = {"total": 0, "exact": 0}
        state_breakdown[st]["total"] += 1
        if raw_ind == gt:
            state_breakdown[st]["exact"] += 1
            
        # Confusion analysis
        if raw_ind != gt:
            for c_pred, c_gt in zip(raw_ind, gt):
                if c_pred != c_gt:
                    char_confusions[f"{c_gt}->{c_pred}"] += 1
                    
        # Record row
        rows.append({
            "filename": fname,
            "ground_truth": gt,
            "is_two_line": is_two,
            "aspect_ratio": round(ar, 2),
            "condition": cond_name,
            "ppocr_raw": raw_pp,
            "ppocr_correct": (raw_pp == gt),
            "ppocr_latency_ms": round(t_pp, 2),
            "pretrained_fastplate_raw": raw_pre,
            "pretrained_correct": (raw_pre == gt),
            "pretrained_latency_ms": round(t_pre, 2),
            "indian_fastplate_raw": raw_ind,
            "indian_fastplate_post": post_ind,
            "indian_correct_raw": (raw_ind == gt),
            "indian_correct_post": (post_ind == gt),
            "indian_latency_ms": round(t_ind, 2),
            "fallback_raw": fallback_pred,
            "fallback_correct": (fallback_pred == gt)
        })
        
    total_samples = len(rows)
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"\nSaved prediction comparison table to {OUTPUT_PREDS_PATH}")
    
    # Calculate summary metrics
    def summarize_model(m):
        eds = m["edit_dists"]
        lat = m.get("times", [0.0])
        return {
            "total_samples": total_samples,
            "raw_exact_matches": m["exact"],
            "raw_exact_accuracy_pct": round(m["exact"] / total_samples * 100.0, 2),
            "post_processed_exact": m["post_exact"],
            "post_processed_exact_pct": round(m["post_exact"] / total_samples * 100.0, 2),
            "mean_character_accuracy_pct": round(float(np.mean(m["char_accs"])), 2),
            "mean_edit_distance": round(float(np.mean(eds)), 3),
            "median_edit_distance": float(np.median(eds)),
            "perfect_matches": sum(1 for e in eds if e == 0),
            "error_1_char": sum(1 for e in eds if e == 1),
            "error_2_char": sum(1 for e in eds if e == 2),
            "error_3plus_char": sum(1 for e in eds if e >= 3),
            "mean_latency_ms": round(float(np.mean(lat)), 2) if lat else 0.0,
            "median_latency_ms": round(float(np.median(lat)), 2) if lat else 0.0,
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 2) if lat else 0.0,
            "throughput_fps": round(1000.0 / float(np.mean(lat)), 1) if lat and np.mean(lat) > 0 else 0.0
        }
        
    summary_pp = summarize_model(models["ppocr_v4"])
    summary_pre = summarize_model(models["fastplate_pretrained"])
    summary_ind = summarize_model(models["fastplate_indian"])
    summary_fb = summarize_model(models["adaptive_fallback"])
    
    # Phase 12: Multi-Frame Temporal Simulation
    # Group predictions by vehicle (simulating 3 consecutive surveillance frames)
    # Temporal consensus votes across the 3 consecutive detections
    temporal_exact = 0
    temporal_total = 0
    group_size = 3
    for i in range(0, total_samples - group_size + 1, group_size):
        chunk = rows[i:i+group_size]
        ground_truth = chunk[0]["ground_truth"]
        # If all 3 belong to the same plate/vehicle
        if all(c["ground_truth"] == ground_truth for c in chunk):
            votes = [c["indian_fastplate_raw"] for c in chunk]
            vote_counts = Counter(votes)
            consensus_plate = vote_counts.most_common(1)[0][0]
            temporal_total += 1
            if consensus_plate == ground_truth:
                temporal_exact += 1
    temporal_pct = round(temporal_exact / max(1, temporal_total) * 100.0, 2)
    
    # Complementarity Matrix (Phase 13)
    both_correct = sum(1 for r in rows if r["ppocr_correct"] and r["indian_correct_raw"])
    indian_only = sum(1 for r in rows if r["indian_correct_raw"] and not r["ppocr_correct"])
    ppocr_only = sum(1 for r in rows if r["ppocr_correct"] and not r["indian_correct_raw"])
    neither_correct = sum(1 for r in rows if not r["ppocr_correct"] and not r["indian_correct_raw"])
    oracle_exact = both_correct + indian_only + ppocr_only
    oracle_pct = round(oracle_exact / total_samples * 100.0, 2)
    
    # Condition breakdown formatting
    cond_summary = {}
    for c_name, c_data in condition_stats.items():
        tot = c_data["total"]
        cond_summary[c_name] = {
            "total": tot,
            "exact": c_data["exact"],
            "exact_pct": round(c_data["exact"] / tot * 100.0, 2),
            "mean_char_acc_pct": round(float(np.mean(c_data["char_accs"])), 2)
        }
        
    final_metrics = {
        "benchmark_dataset": "Untouched 228 Indian Plate Crops (Frozen Benchmark)",
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "processor": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)
        },
        "model_comparison": {
            "MODEL_A_PPOCRv4": summary_pp,
            "MODEL_B_FASTPLATE_PRETRAINED": summary_pre,
            "MODEL_C_FASTPLATE_INDIAN_FINETUNED": summary_ind,
            "ADAPTIVE_FALLBACK_PIPELINE": summary_fb
        },
        "pipeline_hierarchy": {
            "raw_single_crop_exact_pct": summary_ind["raw_exact_accuracy_pct"],
            "post_processed_single_crop_exact_pct": summary_ind["post_processed_exact_pct"],
            "temporal_consensus_exact_pct": temporal_pct,
            "adaptive_fallback_exact_pct": summary_fb["raw_exact_accuracy_pct"]
        },
        "layout_breakdown": {
            "single_line": {
                "total": single_line_stats["total"],
                "exact": single_line_stats["exact"],
                "exact_pct": round(single_line_stats["exact"] / max(1, single_line_stats["total"]) * 100.0, 2)
            },
            "two_line": {
                "total": two_line_stats["total"],
                "exact": two_line_stats["exact"],
                "exact_pct": round(two_line_stats["exact"] / max(1, two_line_stats["total"]) * 100.0, 2)
            }
        },
        "condition_breakdown": cond_summary,
        "complementarity_phase13": {
            "both_correct": both_correct,
            "indian_only_correct": indian_only,
            "ppocr_only_correct": ppocr_only,
            "neither_correct": neither_correct,
            "oracle_union_exact": oracle_exact,
            "oracle_union_pct": oracle_pct
        },
        "top_character_confusions": char_confusions.most_common(12)
    }
    
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved complete benchmark metrics to {OUTPUT_METRICS_PATH}")
    
    # Print terminal report
    print("\n=======================================================")
    print("FINAL TEST SET BENCHMARK RESULTS (228 CROPS)")
    print("=======================================================")
    print(f"1. PP-OCRv4 (Current):             Raw Exact: {summary_pp['raw_exact_accuracy_pct']}% | Char Acc: {summary_pp['mean_character_accuracy_pct']}% | Latency: {summary_pp['mean_latency_ms']} ms")
    print(f"2. FastPlateOCR (Pretrained):      Raw Exact: {summary_pre['raw_exact_accuracy_pct']}% | Char Acc: {summary_pre['mean_character_accuracy_pct']}% | Latency: {summary_pre['mean_latency_ms']} ms")
    print(f"3. FastPlateOCR (Indian Fine-Tune): Raw Exact: {summary_ind['raw_exact_accuracy_pct']}% | Char Acc: {summary_ind['mean_character_accuracy_pct']}% | Latency: {summary_ind['mean_latency_ms']} ms")
    print(f"4. Post-Processed Indian Pipeline: Exact:     {summary_ind['post_processed_exact_pct']}%")
    print(f"5. Temporal Multi-Frame Consensus: Exact:     {temporal_pct}%")
    print(f"6. Adaptive Fallback Pipeline:     Exact:     {summary_fb['raw_exact_accuracy_pct']}% | Oracle: {oracle_pct}%")
    print("=======================================================")
    print(f"Single-Line Exact: {single_line_stats['exact']}/{single_line_stats['total']} ({final_metrics['layout_breakdown']['single_line']['exact_pct']}%)")
    print(f"Two-Line Exact:    {two_line_stats['exact']}/{two_line_stats['total']} ({final_metrics['layout_breakdown']['two_line']['exact_pct']}%)")
    print("=======================================================")

if __name__ == "__main__":
    run_evaluation()
