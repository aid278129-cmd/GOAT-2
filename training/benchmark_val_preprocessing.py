"""
training/benchmark_val_preprocessing.py
High-Speed Batched Preprocessing and Low-Res Upscaling A/B Testing on Validation Data ONLY.

Evaluates on the curated validation set (training/dataset_v2/val):
1. RAW (native model resize)
2. CURRENT RECTIFICATION (homography deskew)
3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. RECTIFICATION + CLAHE
5. LANCZOS UPSCALE (upscale small crops <48px to 48px)
6. LANCZOS + MILD SHARPENING

Strict Rule: Uses ONLY validation data. Zero test set exposure.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import sys
import time
import json
import pathlib
import cv2
import numpy as np
import pandas as pd
import torch
import keras
from typing import Dict, Any, List, Tuple

import fast_plate_ocr.train.model.layers
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.rectifier import rectify_plate_perspective

VAL_DIR = os.path.join(BASE_DIR, "training", "dataset_v2", "val")
VAL_CSV = os.path.join(BASE_DIR, "training", "dataset_v2", "val_annotations.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "indian_cct", "cct_s_v2_indian_best.keras")
CONFIG_YAML = os.path.expanduser(r"~/.cache/fast-plate-ocr/cct-s-v2-global-model/cct_s_v2_global_plate_config.yaml")

def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_unsharp_mask(img_bgr: np.ndarray, sigma: float = 1.0, strength: float = 0.4) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def apply_lanczos_upscale(img_bgr: np.ndarray, target_h: int = 48) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h < target_h:
        scale = target_h / float(max(1, h))
        new_w = max(16, int(w * scale))
        return cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return img_bgr

def preprocess_image(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "RAW":
        return img_bgr
    elif mode == "RECTIFICATION":
        rect, _, _ = rectify_plate_perspective(img_bgr)
        return rect
    elif mode == "CLAHE":
        return apply_clahe(img_bgr)
    elif mode == "RECTIFICATION_CLAHE":
        rect, _, _ = rectify_plate_perspective(img_bgr)
        return apply_clahe(rect)
    elif mode == "LANCZOS_UPSCALE":
        return apply_lanczos_upscale(img_bgr, target_h=48)
    elif mode == "LANCZOS_SHARPEN":
        up = apply_lanczos_upscale(img_bgr, target_h=48)
        return apply_unsharp_mask(up, sigma=0.8, strength=0.35)
    else:
        return img_bgr

def decode_predictions(logits: torch.Tensor, alphabet: List[str], pad_char: str = '_') -> List[str]:
    preds = []
    indices = torch.argmax(logits, dim=-1).cpu().numpy()
    for row in indices:
        chars = [alphabet[idx] for idx in row]
        plate = "".join(chars).replace(pad_char, "")
        preds.append(plate)
    return preds

def run_preprocessing_ab_test():
    if not os.path.exists(VAL_CSV):
        print(f"Validation CSV not found at {VAL_CSV}")
        return

    df_val = pd.read_csv(VAL_CSV)
    print(f"Loaded {len(df_val)} validation samples for Preprocessing A/B Test.", flush=True)

    plate_cfg = load_plate_config_from_yaml(pathlib.Path(CONFIG_YAML))
    alphabet = plate_cfg.alphabet
    pad_char = plate_cfg.pad_char

    print(f"Loading Indian CCT model from {MODEL_PATH}...", flush=True)
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.eval()

    pipelines = [
        "RAW",
        "RECTIFICATION",
        "CLAHE",
        "RECTIFICATION_CLAHE",
        "LANCZOS_UPSCALE",
        "LANCZOS_SHARPEN"
    ]

    # Preload raw images into memory
    print("Preloading validation images into RAM...", flush=True)
    val_items = []
    for _, row in df_val.iterrows():
        p = os.path.join(VAL_DIR, row["crop_file"])
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        val_items.append({
            "img": img,
            "gt": str(row["plate_text"]).strip().upper(),
            "is_two_line": bool(row.get("is_two_line", False)) or (w / float(max(1, h)) < 2.3),
            "is_low_res": (h <= 48)
        })

    print(f"Preloaded {len(val_items)} valid images into RAM.", flush=True)
    batch_size = 64
    summary_rows = []

    print("\n" + "="*80, flush=True)
    print("RUNNING BATCHED PREPROCESSING & LOW-RES A/B TEST (VALIDATION SET ONLY)", flush=True)
    print("="*80, flush=True)

    for pipe in pipelines:
        t0 = time.time()
        # 1. Preprocess & build tensor batch
        tensors = []
        for item in val_items:
            proc = preprocess_image(item["img"], pipe)
            resized = cv2.resize(proc, (128, 64), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensors.append(rgb)

        tensor_arr = np.array(tensors, dtype=np.uint8)
        all_preds = []

        # 2. Batched PyTorch inference
        with torch.no_grad():
            for b_start in range(0, len(tensor_arr), batch_size):
                b_end = min(len(tensor_arr), b_start + batch_size)
                xb = torch.from_numpy(tensor_arr[b_start:b_end])
                out = model(xb)
                preds = decode_predictions(out, alphabet, pad_char)
                all_preds.extend(preds)

        # 3. Evaluate metrics
        exact = 0
        total = len(all_preds)
        lr_exact = 0
        lr_total = 0
        tl_exact = 0
        tl_total = 0

        for pred, item in zip(all_preds, val_items):
            is_match = (pred == item["gt"])
            if is_match:
                exact += 1
            if item["is_low_res"]:
                lr_total += 1
                if is_match:
                    lr_exact += 1
            if item["is_two_line"]:
                tl_total += 1
                if is_match:
                    tl_exact += 1

        exact_pct = round(exact / max(1, total) * 100.0, 2)
        lr_exact_pct = round(lr_exact / max(1, lr_total) * 100.0, 2)
        tl_exact_pct = round(tl_exact / max(1, tl_total) * 100.0, 2)
        elapsed = round(time.time() - t0, 2)

        print(f"Pipeline: {pipe:20s} | Exact: {exact_pct:6.2f}% ({exact}/{total}) | Low-Res: {lr_exact_pct:6.2f}% ({lr_exact}/{lr_total}) | Two-Line: {tl_exact_pct:6.2f}% ({tl_exact}/{tl_total}) | Latency: {elapsed}s", flush=True)

        summary_rows.append({
            "pipeline": pipe,
            "overall_exact_pct": exact_pct,
            "low_res_exact_pct": lr_exact_pct,
            "two_line_exact_pct": tl_exact_pct,
            "exact_count": exact,
            "total_count": total,
            "eval_time_sec": elapsed
        })

    out_csv = os.path.join(BASE_DIR, "validation_preprocessing_ab_results.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nSaved detailed preprocessing results to {out_csv}", flush=True)

if __name__ == "__main__":
    run_preprocessing_ab_test()
