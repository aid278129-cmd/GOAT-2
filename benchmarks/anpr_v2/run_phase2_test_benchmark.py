#!/usr/bin/env python3
"""
benchmarks/anpr_v2/run_phase2_test_benchmark.py
Phase 2 Final Evaluation on the Untouched 228 Held-Out Test Set.

Evaluates and compares:
- MODEL A: PP-OCRv4 (Current RapidOCR engine)
- MODEL B: FastPlateOCR Pretrained CCT-S-V2 (Global weights)
- MODEL C: FastPlateOCR Indian Fine-Tuned v1 (cct_s_v2_indian_best.keras)
- MODEL D: FastPlateOCR Indian Real-Data Fine-Tuned v2 (cct_s_v2_real_best.keras)
- MODEL E: Adaptive Fallback (Model D + PP-OCRv4 complementary union)

STRICTLY: Zero test set data is exposed to training at any point.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
import json
import time
import platform
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.recognizer import get_recognizer
from anpr_v2.fastplate_recognizer import FastPlateRecognizer
from anpr_v2.fastplate_indian_recognizer import IndianFastPlateRecognizer
from anpr_v2.two_line_handler import is_two_line_plate
from anpr_v2.validator import validate_indian_registration, clean_ocr_raw_tokens

BENCHMARK_INDEX = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "ground_truth_benchmark.json")
CROPS_DIR = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "crops")
REAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "indian_cct", "cct_s_v2_real_best.keras")
PREV_MODEL_PATH = os.path.join(BASE_DIR, "models", "indian_cct", "cct_s_v2_indian_best.keras")

OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "phase2_test_metrics.json")
OUTPUT_PREDS_PATH = os.path.join(BASE_DIR, "benchmarks", "anpr_v2", "phase2_test_predictions.csv")

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

def apply_lanczos_upscale(img_bgr: np.ndarray, target_h: int = 48) -> np.ndarray:
    """Upscale low-res crops to improve OCR on very small plates."""
    h, w = img_bgr.shape[:2]
    if h < target_h:
        scale = target_h / float(max(1, h))
        new_w = max(16, int(w * scale))
        return cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return img_bgr

def run_phase2_evaluation():
    print("=================================================================")
    print("PHASE 2: FINAL TEST BENCHMARK ON UNTOUCHED 228 HELD-OUT CROPS")
    print("=================================================================")

    with open(BENCHMARK_INDEX, "r", encoding="utf-8") as f:
        benchmark_records = json.load(f)
    total_samples = len(benchmark_records)
    print(f"Loaded {total_samples} benchmark test records.")

    # Initialize all models
    print("\n[1/5] Initializing PP-OCRv4...")
    ppocr = get_recognizer()

    print("\n[2/5] Initializing FastPlateOCR Pretrained CCT-S-V2...")
    fast_pretrained = FastPlateRecognizer()

    print("\n[3/5] Initializing Indian Fine-Tuned v1 (cct_s_v2_indian_best)...")
    fast_v1 = IndianFastPlateRecognizer(model_path=PREV_MODEL_PATH)

    print("\n[4/5] Initializing Real-Data Fine-Tuned v2 (cct_s_v2_real_best)...")
    if os.path.exists(REAL_MODEL_PATH):
        fast_v2 = IndianFastPlateRecognizer(model_path=REAL_MODEL_PATH)
    else:
        print(f"  WARNING: Real-data model not found at {REAL_MODEL_PATH}. Using v1 as stand-in.")
        fast_v2 = fast_v1

    print("\n[5/5] Starting inference on 228 test crops...")

    models = {
        "ppocr_v4":             {"exact": 0, "char_accs": [], "edit_dists": [], "times": []},
        "fastplate_pretrained": {"exact": 0, "char_accs": [], "edit_dists": [], "times": []},
        "fastplate_v1_indian":  {"exact": 0, "char_accs": [], "edit_dists": [], "times": []},
        "fastplate_v2_real":    {"exact": 0, "char_accs": [], "edit_dists": [], "times": []},
        "adaptive_v2_fallback": {"exact": 0, "char_accs": [], "edit_dists": [], "times": []},
    }

    rows = []
    single_line = {"total": 0, "exact_v1": 0, "exact_v2": 0}
    two_line =    {"total": 0, "exact_v1": 0, "exact_v2": 0}
    low_res =     {"total": 0, "exact_v1": 0, "exact_v2": 0}

    for idx, item in enumerate(benchmark_records):
        fname = os.path.basename(item["crop_path"])
        local_path = os.path.join(CROPS_DIR, fname)
        gt = item["ground_truth"].strip().upper()

        img = cv2.imread(local_path)
        if img is None:
            print(f"  WARNING: Cannot read {local_path}, skipping.")
            continue

        h, w = img.shape[:2]
        is_two = is_two_line_plate(img)
        is_low_res = (h <= 48)

        # Apply Lanczos upscale for small plates
        img_proc = apply_lanczos_upscale(img) if is_low_res else img

        # 1. PP-OCRv4
        t0 = time.perf_counter()
        res_pp = ppocr.recognize(img)
        t_pp = (time.perf_counter() - t0) * 1000.0
        raw_pp = res_pp.text if hasattr(res_pp, "text") else str(res_pp)
        conf_pp = getattr(res_pp, "confidence", 0.0)
        val_pp = validate_indian_registration(raw_pp)

        # 2. FastPlateOCR Pretrained
        t0 = time.perf_counter()
        res_pre = fast_pretrained.recognize(img)
        t_pre = (time.perf_counter() - t0) * 1000.0
        raw_pre = res_pre.text if hasattr(res_pre, "text") else str(res_pre)
        val_pre = validate_indian_registration(raw_pre)

        # 3. FastPlateOCR Indian v1
        t0 = time.perf_counter()
        raw_v1, conf_v1 = fast_v1.recognize(img_proc)
        t_v1 = (time.perf_counter() - t0) * 1000.0
        val_v1 = validate_indian_registration(raw_v1)

        # 4. FastPlateOCR Real-Data v2
        t0 = time.perf_counter()
        raw_v2, conf_v2 = fast_v2.recognize(img_proc)
        t_v2 = (time.perf_counter() - t0) * 1000.0
        val_v2 = validate_indian_registration(raw_v2)

        # 5. Adaptive v2 Fallback: v2 primary, PP-OCRv4 fallback
        if val_v2.is_valid and not val_pp.is_valid:
            raw_adap, t_adap = raw_v2, t_v2
        elif val_pp.is_valid and not val_v2.is_valid:
            raw_adap, t_adap = raw_pp, t_pp + t_v2
        elif val_v2.is_valid and val_pp.is_valid:
            raw_adap = raw_v2 if conf_v2 >= conf_pp else raw_pp
            t_adap = t_v2
        else:
            raw_adap = raw_v2 if len(raw_v2) >= 8 else raw_pp
            t_adap = t_v2

        # Update tracker
        for m_name, raw_p, lat in [
            ("ppocr_v4", raw_pp, t_pp),
            ("fastplate_pretrained", raw_pre, t_pre),
            ("fastplate_v1_indian", raw_v1, t_v1),
            ("fastplate_v2_real", raw_v2, t_v2),
            ("adaptive_v2_fallback", raw_adap, t_adap),
        ]:
            m = models[m_name]
            ed = levenshtein(raw_p, gt)
            m["edit_dists"].append(ed)
            char_acc = max(0.0, 1.0 - (ed / max(len(raw_p), len(gt), 1))) * 100.0
            m["char_accs"].append(char_acc)
            m["times"].append(lat)
            if raw_p == gt:
                m["exact"] += 1

        # Layout breakdown for v1 and v2
        if is_two:
            two_line["total"] += 1
            if raw_v1 == gt: two_line["exact_v1"] += 1
            if raw_v2 == gt: two_line["exact_v2"] += 1
        else:
            single_line["total"] += 1
            if raw_v1 == gt: single_line["exact_v1"] += 1
            if raw_v2 == gt: single_line["exact_v2"] += 1

        if is_low_res:
            low_res["total"] += 1
            if raw_v1 == gt: low_res["exact_v1"] += 1
            if raw_v2 == gt: low_res["exact_v2"] += 1

        rows.append({
            "crop_file": fname,
            "ground_truth": gt,
            "ppocr_raw": raw_pp,
            "pretrained_raw": raw_pre,
            "v1_indian_raw": raw_v1,
            "v2_real_raw": raw_v2,
            "adaptive_raw": raw_adap,
            "ppocr_correct": raw_pp == gt,
            "pretrained_correct": raw_pre == gt,
            "v1_correct": raw_v1 == gt,
            "v2_correct": raw_v2 == gt,
            "adaptive_correct": raw_adap == gt,
            "is_two_line": is_two,
            "is_low_res": is_low_res,
            "h": h, "w": w
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{total_samples} test crops...", flush=True)

    # Build summary metrics
    print("\n\n" + "="*70)
    print("PHASE 2 FINAL TEST BENCHMARK RESULTS (228 HELD-OUT CROPS)")
    print("="*70)

    metrics_out = {}
    model_labels = {
        "ppocr_v4":             "MODEL A: PP-OCRv4 (RapidOCR)",
        "fastplate_pretrained": "MODEL B: FastPlateOCR Pretrained",
        "fastplate_v1_indian":  "MODEL C: Indian Fine-Tuned v1",
        "fastplate_v2_real":    "MODEL D: Real-Data Fine-Tuned v2",
        "adaptive_v2_fallback": "MODEL E: Adaptive v2+PP Fallback",
    }

    for m_name, label in model_labels.items():
        m = models[m_name]
        exact_pct = round(m["exact"] / total_samples * 100.0, 2)
        char_acc = round(float(np.mean(m["char_accs"])), 2)
        avg_ed = round(float(np.mean(m["edit_dists"])), 2)
        avg_lat = round(float(np.mean(m["times"])), 1)
        print(f"{label:38s} | Exact: {exact_pct:6.2f}% ({m['exact']:3d}/228) | Char: {char_acc:5.2f}% | Avg ED: {avg_ed:.2f} | Latency: {avg_lat:.1f}ms")
        metrics_out[m_name] = {
            "exact_pct": exact_pct,
            "exact_count": m["exact"],
            "char_acc_pct": char_acc,
            "avg_edit_dist": avg_ed,
            "avg_latency_ms": avg_lat
        }

    # Layout breakdown
    print("\n--- LAYOUT BREAKDOWN (v1 vs v2) ---")
    print(f"Single-Line: v1={round(single_line['exact_v1']/max(1,single_line['total'])*100,2)}% ({single_line['exact_v1']}/{single_line['total']}) | v2={round(single_line['exact_v2']/max(1,single_line['total'])*100,2)}% ({single_line['exact_v2']}/{single_line['total']})")
    print(f"Two-Line:    v1={round(two_line['exact_v1']/max(1,two_line['total'])*100,2)}% ({two_line['exact_v1']}/{two_line['total']}) | v2={round(two_line['exact_v2']/max(1,two_line['total'])*100,2)}% ({two_line['exact_v2']}/{two_line['total']})")
    print(f"Low-Res:     v1={round(low_res['exact_v1']/max(1,low_res['total'])*100,2)}% ({low_res['exact_v1']}/{low_res['total']}) | v2={round(low_res['exact_v2']/max(1,low_res['total'])*100,2)}% ({low_res['exact_v2']}/{low_res['total']})")

    # Complementarity
    v2_only = sum(1 for r in rows if r["v2_correct"] and not r["ppocr_correct"])
    pp_only = sum(1 for r in rows if r["ppocr_correct"] and not r["v2_correct"])
    both_correct = sum(1 for r in rows if r["v2_correct"] and r["ppocr_correct"])
    oracle = both_correct + v2_only + pp_only
    oracle_pct = round(oracle / total_samples * 100.0, 2)
    print(f"\n--- COMPLEMENTARITY ---")
    print(f"Oracle (v2 OR ppocr correct): {oracle_pct}% ({oracle}/{total_samples})")
    print(f"v2 Only correct: {v2_only} | PP Only correct: {pp_only} | Both: {both_correct}")

    metrics_out["layout_breakdown"] = {
        "single_line": single_line,
        "two_line": two_line,
        "low_res": low_res
    }
    metrics_out["oracle_pct"] = oracle_pct
    metrics_out["oracle_count"] = oracle

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved metrics to {OUTPUT_METRICS_PATH}")

    pd.DataFrame(rows).to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PREDS_PATH}")

    # Terminal Summary Block
    v2_exact = metrics_out["fastplate_v2_real"]["exact_pct"]
    adap_exact = metrics_out["adaptive_v2_fallback"]["exact_pct"]
    pp_exact = metrics_out["ppocr_v4"]["exact_pct"]
    v1_exact = metrics_out["fastplate_v1_indian"]["exact_pct"]

    print("\n" + "="*70)
    print("PHASE 2 FINAL TERMINAL SUMMARY")
    print("="*70)
    print(f"  Frozen Test Set: 228 real Indian plate crops (immutable, never seen during training)")
    print(f"  PP-OCRv4 (Baseline):                   {pp_exact:.2f}% exact full-plate")
    print(f"  FastPlateOCR Indian v1 (Phase 1):      {v1_exact:.2f}% exact full-plate")
    print(f"  FastPlateOCR Real-Data v2 (Phase 2):   {v2_exact:.2f}% exact full-plate")
    print(f"  Adaptive v2+PP Fallback:               {adap_exact:.2f}% exact full-plate")
    print(f"  Oracle Upper Bound:                    {oracle_pct:.2f}% exact full-plate")
    print("="*70)

if __name__ == "__main__":
    run_phase2_evaluation()
