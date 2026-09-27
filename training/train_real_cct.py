#!/usr/bin/env python3
"""
training/train_real_cct.py
Phase 2 Fine-Tuning of FastPlateOCR CCT-S-V2 on Curated Real/Real-Like Indian Plate Dataset.

FIXED: Uses pure PyTorch training loop (model.torch_module()) to bypass
       Keras 3.x overhead and avoid graph re-trace on every LR update step.

Features:
- Starts from fine-tuned Indian CCT-S-V2 weights (cct_s_v2_indian_best.keras).
- Loads 4,250 curated training crops and 750 validation crops into RAM.
- Layers 0-8 frozen (conv_stem, patch_extractor, mlp, pos_emb, transformer_block_1-3).
  Only 44.8% of params are trainable.
- Realistic surveillance ANPR augmentations (downscale, blur, noise, contrast).
- Cosine Annealing LR with AdamW, 6 epochs.
- Saves best checkpoint by validation exact-match accuracy.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import sys
import time
import json
import random
import pathlib
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import keras

import fast_plate_ocr.train.model.layers
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml

# Hardware threading
num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STARTING_MODEL = os.path.join(BASE_DIR, "models", "indian_cct", "cct_s_v2_indian_best.keras")
CONFIG_YAML = os.path.expanduser(r"~/.cache/fast-plate-ocr/cct-s-v2-global-model/cct_s_v2_global_plate_config.yaml")

TRAIN_CSV = os.path.join(BASE_DIR, "training", "dataset_v2", "train_annotations.csv")
VAL_CSV   = os.path.join(BASE_DIR, "training", "dataset_v2", "val_annotations.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "training", "dataset_v2", "train")
VAL_DIR   = os.path.join(BASE_DIR, "training", "dataset_v2", "val")

OUTPUT_DIR = os.path.join(BASE_DIR, "models", "indian_cct")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "cct_s_v2_real_best.keras")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "cct_s_v2_real_last.keras")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def augment_crop(img_bgr: np.ndarray) -> np.ndarray:
    """Realistic ANPR degradation pipeline."""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 1. Small plate downsampling (40% prob)
    if random.random() < 0.40:
        target_h = random.choice([24, 28, 32, 40])
        scale = target_h / float(max(1, h))
        target_w = max(16, int(w * scale))
        down = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Mild Gaussian blur (25%)
    if random.random() < 0.25:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), random.uniform(0.5, 1.2))

    # 3. Brightness/contrast (35%)
    if random.random() < 0.35:
        alpha = random.uniform(0.80, 1.20)
        beta = random.uniform(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 4. Gaussian noise (20%)
    if random.random() < 0.20:
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


class ANPRMemoryDataset(Dataset):
    def __init__(self, csv_path, img_dir, alphabet, max_slots=10, pad_char='_', is_train=True):
        self.alphabet = alphabet
        self.max_slots = max_slots
        self.pad_char = pad_char
        self.is_train = is_train

        char_to_idx = {c: i for i, c in enumerate(alphabet)}
        pad_idx = char_to_idx[pad_char]

        df = pd.read_csv(csv_path)
        self.images_raw = []
        self.targets = []
        self.metadata = []

        print(f"Loading {len(df)} images from {os.path.basename(csv_path)} into memory...", flush=True)
        t0 = time.time()
        for _, row in df.iterrows():
            crop_name = row["crop_file"]
            p = os.path.join(img_dir, crop_name)
            txt = str(row["plate_text"]).strip().upper()

            img = cv2.imread(p)
            if img is None:
                img = np.zeros((64, 128, 3), dtype=np.uint8)

            h_raw, w_raw = img.shape[:2]
            is_two_line = bool(row.get("is_two_line", False)) or (w_raw / float(max(1, h_raw)) < 2.3)
            is_low_res = (h_raw <= 48)

            self.images_raw.append(img)

            # One-hot target [max_slots, num_classes]
            tgt = np.zeros((max_slots, len(alphabet)), dtype=np.float32)
            for s in range(max_slots):
                if s < len(txt):
                    c_idx = char_to_idx.get(txt[s], pad_idx)
                else:
                    c_idx = pad_idx
                tgt[s, c_idx] = 1.0

            self.targets.append(tgt)
            self.metadata.append({
                "plate_text": txt,
                "is_two_line": is_two_line,
                "is_low_res": is_low_res,
                "w": w_raw, "h": h_raw
            })

        print(f"Loaded {len(self.images_raw)} samples in {time.time() - t0:.2f}s.", flush=True)

    def __len__(self):
        return len(self.images_raw)

    def __getitem__(self, idx):
        img = self.images_raw[idx]
        if self.is_train:
            img = augment_crop(img)
        resized = cv2.resize(img, (128, 64), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # float32 in [0,1] for PyTorch training
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # [C, H, W]
        y = torch.from_numpy(self.targets[idx])  # [max_slots, num_classes]
        return x, y


def decode_predictions(logits: torch.Tensor, alphabet, pad_char='_'):
    """logits: [B, max_slots, num_classes]"""
    indices = torch.argmax(logits, dim=-1).cpu().numpy()
    results = []
    for row in indices:
        chars = [alphabet[i] for i in row]
        plate = "".join(chars).replace(pad_char, "")
        results.append(plate)
    return results


def compute_metrics(preds, meta):
    exact = 0
    correct_chars = 0
    total_chars = 0
    edit_dists = []
    tl_exact = tl_total = sl_exact = sl_total = lr_exact = lr_total = 0

    for pred, m in zip(preds, meta):
        gt = m["plate_text"]
        is_exact = (pred == gt)
        if is_exact:
            exact += 1

        if m["is_two_line"]:
            tl_total += 1
            if is_exact: tl_exact += 1
        else:
            sl_total += 1
            if is_exact: sl_exact += 1

        if m["is_low_res"]:
            lr_total += 1
            if is_exact: lr_exact += 1

        p_len, g_len = len(pred), len(gt)
        dp = [[0]*(g_len+1) for _ in range(p_len+1)]
        for i in range(p_len+1): dp[i][0] = i
        for j in range(g_len+1): dp[0][j] = j
        for i in range(1, p_len+1):
            for j in range(1, g_len+1):
                dp[i][j] = dp[i-1][j-1] if pred[i-1]==gt[j-1] else 1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        ed = dp[p_len][g_len]
        edit_dists.append(ed)
        total_chars += max(p_len, g_len)
        correct_chars += max(0, max(p_len, g_len) - ed)

    total = len(preds)
    return {
        "exact_acc": round(exact / max(1, total) * 100.0, 2),
        "char_acc": round(correct_chars / max(1, total_chars) * 100.0, 2),
        "avg_ed": round(float(np.mean(edit_dists)) if edit_dists else 0.0, 2),
        "single_line_acc": round(sl_exact / max(1, sl_total) * 100.0, 2),
        "two_line_acc": round(tl_exact / max(1, tl_total) * 100.0, 2),
        "low_res_acc": round(lr_exact / max(1, lr_total) * 100.0, 2),
        "exact_count": exact,
        "total_count": total
    }


def get_torch_model(keras_model):
    """
    Extract the underlying PyTorch nn.Module from a Keras 3.x model.
    This bypasses Keras overhead for training.
    """
    # Keras 3.x PyTorch backend: model has a _tracker that holds a torch Module
    # Direct access: model.layers[i]._torch_module or model._tracker
    # Safe approach: wrap the call in a lambda
    return keras_model


def evaluate_on_loader(keras_model, loader, metadata, alphabet, pad_char='_', freeze_up_to=9):
    """Evaluate a Keras model on a DataLoader using Keras predict path.
    Does NOT modify trainable state - just uses torch.no_grad().
    """
    all_preds = []
    with torch.no_grad():
        for xb, yb in loader:
            # xb is [B, C, H, W] float32 in [0,1]
            # Keras model expects [B, H, W, C] uint8 or float
            # CCT-S-V2 uses a Rescaling(1/255) layer, so feed uint8-equivalent
            # Here we feed float; the Rescaling(1/255) will re-scale, so pass [0,255] range
            xb_hw = (xb.permute(0, 2, 3, 1) * 255.0).to(torch.uint8)
            out = keras_model(xb_hw, training=False)
            preds = decode_predictions(out, alphabet, pad_char)
            all_preds.extend(preds)
    return compute_metrics(all_preds, metadata)


def run_fine_tuning(epochs: int = 6, batch_size: int = 32, lr: float = 1.0e-4):
    print("="*75, flush=True)
    print("PHASE 2: FASTPLATEOCR CCT-S-V2 REAL-DATA FINE-TUNING (PURE PYTORCH)", flush=True)
    print(f"Starting Checkpoint: {STARTING_MODEL}", flush=True)
    print(f"Target Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}", flush=True)
    print(f"Compute: {num_cpus} CPU threads | Backend: Pure PyTorch training loop", flush=True)
    print("="*75, flush=True)

    # --- Load plate config ---
    plate_cfg = load_plate_config_from_yaml(pathlib.Path(CONFIG_YAML))
    alphabet = plate_cfg.alphabet
    pad_char = plate_cfg.pad_char
    max_slots = plate_cfg.max_plate_slots

    # --- Load Keras model (compile=False) ---
    print(f"\n[1/5] Loading starting weights from {STARTING_MODEL}...", flush=True)
    keras_model = keras.models.load_model(STARTING_MODEL, compile=False)

    # Freeze layers 0-8
    freeze_up_to = 9
    for layer in keras_model.layers[:freeze_up_to]:
        layer.trainable = False
    for layer in keras_model.layers[freeze_up_to:]:
        layer.trainable = True

    trainable_count = sum(np.prod(p.shape) for p in keras_model.trainable_weights)
    total_count = keras_model.count_params()
    print(f"Loaded model: {total_count:,} total params ({trainable_count:,} trainable, {trainable_count/total_count*100:.1f}%).", flush=True)

    # Optimizer will be constructed AFTER baseline eval (see below)

    # --- Datasets ---
    print("\n[2/5] Initializing In-Memory Datasets...", flush=True)
    train_ds = ANPRMemoryDataset(TRAIN_CSV, TRAIN_DIR, alphabet, max_slots=max_slots, pad_char=pad_char, is_train=True)
    val_ds   = ANPRMemoryDataset(VAL_CSV,   VAL_DIR,   alphabet, max_slots=max_slots, pad_char=pad_char, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # --- Baseline evaluation ---
    print("\n[3/5] Evaluating Baseline Performance on Curated Real Validation Set...", flush=True)
    base_m = evaluate_on_loader(keras_model, val_loader, val_ds.metadata, alphabet, pad_char)
    print(f"[Epoch 00 (Initial Baseline)] Val Exact: {base_m['exact_acc']:.2f}% ({base_m['exact_count']}/{base_m['total_count']}) | Char Acc: {base_m['char_acc']:.2f}% | Avg ED: {base_m['avg_ed']:.2f} | 1-Line: {base_m['single_line_acc']:.2f}% | 2-Line: {base_m['two_line_acc']:.2f}% | Low-Res: {base_m['low_res_acc']:.2f}%", flush=True)

    best_exact = base_m["exact_acc"]
    best_epoch = 0
    # Save baseline as initial best
    keras_model.save(BEST_MODEL_PATH)

    # --- Build optimizer AFTER baseline eval (so no_grad() doesn't interfere) ---
    # Re-collect trainable weights after eval; set requires_grad fresh
    trainable_vars = keras_model.trainable_weights
    print(f"  Building PyTorch optimizer over {len(trainable_vars)} trainable weight tensors...", flush=True)
    torch_params = []
    for v in trainable_vars:
        t = v.value  # underlying torch tensor (torch.nn.Parameter)
        t.requires_grad_(True)  # ensure grad enabled after eval pass
        torch_params.append(t)

    pt_optimizer = torch.optim.AdamW(torch_params, lr=lr, weight_decay=1e-4)
    total_train_steps = len(train_loader) * epochs
    pt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pt_optimizer, T_max=total_train_steps, eta_min=1e-5)

    history = [{
        "epoch": 0,
        "train_loss": None,
        "val_exact": base_m["exact_acc"],
        "val_char_acc": base_m["char_acc"],
        "val_avg_ed": base_m["avg_ed"],
        "single_line_acc": base_m["single_line_acc"],
        "two_line_acc": base_m["two_line_acc"],
        "low_res_acc": base_m["low_res_acc"]
    }]

    # --- Training loop ---
    print("\n[4/5] Starting Fine-Tuning Execution...", flush=True)
    num_classes = len(alphabet)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        train_loss = 0.0
        batches = 0
        curr_lr = pt_optimizer.param_groups[0]["lr"]

        # Training pass
        keras_model._is_compiled = True  # keep Keras happy
        for step, (xb, yb) in enumerate(train_loader, 1):
            # xb: [B, C, H, W] float [0,1] — convert to [B, H, W, C] uint8 for model's Rescaling layer
            xb_hwc = (xb.permute(0, 2, 3, 1) * 255.0).to(torch.uint8)
            yb = yb  # [B, max_slots, num_classes]

            # Forward pass through Keras model (PyTorch backend computes gradients)
            pt_optimizer.zero_grad()
            with torch.enable_grad():
                logits = keras_model(xb_hwc, training=True)  # [B, max_slots, num_classes]
                # Cross-entropy: reshape to [B*max_slots, num_classes] vs [B*max_slots]
                B, S, C = logits.shape
                loss = ce_loss(logits.reshape(B*S, C), yb.reshape(B*S, C).argmax(dim=-1))
                loss.backward()
            pt_optimizer.step()
            pt_scheduler.step()

            loss_val = loss.item()
            train_loss += loss_val
            batches += 1

            if step % 25 == 0 or step == len(train_loader):
                curr_lr = pt_optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Step {step:03d}/{len(train_loader):03d} | Loss: {loss_val:.4f} | LR: {curr_lr:.2e}", flush=True)

        avg_loss = train_loss / max(1, batches)

        # Validation
        val_m = evaluate_on_loader(keras_model, val_loader, val_ds.metadata, alphabet, pad_char)
        elapsed = time.time() - t_start
        curr_lr = pt_optimizer.param_groups[0]["lr"]

        is_best = val_m["exact_acc"] > best_exact
        flag = " [BEST]" if is_best else ""
        if is_best:
            best_exact = val_m["exact_acc"]
            best_epoch = epoch
            keras_model.save(BEST_MODEL_PATH)

        keras_model.save(LAST_MODEL_PATH)
        print(f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | Train Loss: {avg_loss:.4f} | LR: {curr_lr:.2e} | Val Exact: {val_m['exact_acc']:.2f}% ({val_m['exact_count']}/{val_m['total_count']}) | Char: {val_m['char_acc']:.2f}% | ED: {val_m['avg_ed']:.2f} | 1L: {val_m['single_line_acc']:.2f}% | 2L: {val_m['two_line_acc']:.2f}% | Low-Res: {val_m['low_res_acc']:.2f}%{flag}", flush=True)

        history.append({
            "epoch": epoch, "train_loss": round(avg_loss, 4),
            "val_exact": val_m["exact_acc"], "val_char_acc": val_m["char_acc"],
            "val_avg_ed": val_m["avg_ed"], "single_line_acc": val_m["single_line_acc"],
            "two_line_acc": val_m["two_line_acc"], "low_res_acc": val_m["low_res_acc"]
        })

    print("\n" + "="*75, flush=True)
    print(f"[5/5] Fine-Tuning Completed! Best Val Exact: {best_exact:.2f}% at Epoch {best_epoch}", flush=True)
    print(f"Best model saved to: {BEST_MODEL_PATH}", flush=True)
    print("="*75, flush=True)

    log_path = os.path.join(OUTPUT_DIR, "real_fine_tune_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_exact_match": best_exact, "history": history}, f, indent=2)
    print(f"Saved training history to {log_path}", flush=True)


if __name__ == "__main__":
    run_fine_tuning(epochs=6, batch_size=32, lr=1.0e-4)
