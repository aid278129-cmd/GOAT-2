#!/usr/bin/env python3
"""
training/train_indian_cct.py
Fast and Optimized Fine-Tuning of FastPlateOCR CCT-S-V2 for Indian License Plates.

Key Optimizations:
- Pre-loads training crops into RAM (only ~30 MB for 1,200 crops), eliminating disk I/O bottlenecks.
- Freezes early feature extractor (conv_stem, patch_extractor, early transformer blocks 1-3).
- Focuses optimization on TokenReducer, post-reducer blocks, and VocabularyProjection head.
- Sets PyTorch multi-threading for maximum CPU parallelism.
- Explicit sys.stdout.flush() for real-time progress logging.
- Evaluates Exact Full-Plate Accuracy, Character Accuracy, and Edit Distance after every epoch.
- Saves best checkpoint based on validation exact match.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
import time
import json
import pathlib
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import keras
from fast_plate_ocr.train.utilities.utils import load_keras_model
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml

# Optimize PyTorch CPU threads
num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRETRAINED_KERAS = os.path.join(BASE_DIR, "models", "pretrained", "cct_s_v2_global.keras")
CONFIG_YAML = os.path.join(r"C:\Users\iyers\.cache\fast-plate-ocr\cct-s-v2-global-model\cct_s_v2_global_plate_config.yaml")

TRAIN_CSV = os.path.join(BASE_DIR, "training", "dataset", "train_annotations.csv")
VAL_CSV = os.path.join(BASE_DIR, "training", "dataset", "val_annotations.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "indian_cct")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data_into_memory(csv_path, alphabet, max_samples=1200, max_slots=10, pad_char='_'):
    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        df = df.iloc[:max_samples]
        
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    pad_idx = char_to_idx[pad_char]
    
    images = []
    targets = []
    plate_strings = []
    
    print(f"Loading {len(df)} images into RAM from {os.path.basename(csv_path)}...", flush=True)
    t0 = time.time()
    for _, row in df.iterrows():
        img_p = row['image_path']
        p_str = str(row['plate_text']).strip().upper()
        
        img = cv2.imread(img_p)
        if img is None:
            img = np.zeros((64, 128, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (128, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        # Target one-hot (max_slots, len(alphabet))
        tgt = np.zeros((max_slots, len(alphabet)), dtype=np.float32)
        for s in range(max_slots):
            if s < len(p_str):
                ch = p_str[s]
                c_idx = char_to_idx.get(ch, pad_idx)
            else:
                c_idx = pad_idx
            tgt[s, c_idx] = 1.0
        targets.append(tgt)
        plate_strings.append(p_str)
        
    x_tensor = torch.from_numpy(np.array(images, dtype=np.uint8))
    y_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))
    print(f"Loaded in {time.time() - t0:.2f}s. Memory shape: {x_tensor.shape}, {y_tensor.shape}", flush=True)
    return x_tensor, y_tensor, plate_strings

def compute_metrics(preds, targets_str):
    exact_matches = 0
    total_chars = 0
    correct_chars = 0
    edit_dists = []
    
    for pred, gt in zip(preds, targets_str):
        if pred == gt:
            exact_matches += 1
            
        m, n = len(pred), len(gt)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == gt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        ed = dp[m][n]
        edit_dists.append(ed)
        
        total_chars += max(m, n)
        correct_chars += max(0, max(m, n) - ed)
        
    exact_acc = (exact_matches / len(targets_str)) * 100.0 if targets_str else 0.0
    char_acc = (correct_chars / max(1, total_chars)) * 100.0
    avg_ed = float(np.mean(edit_dists)) if edit_dists else 0.0
    return exact_acc, char_acc, avg_ed

def decode_predictions(logits, alphabet, pad_char='_'):
    preds = []
    indices = torch.argmax(logits, dim=-1).cpu().numpy()
    for row in indices:
        chars = [alphabet[idx] for idx in row]
        plate = "".join(chars).replace(pad_char, "")
        preds.append(plate)
    return preds

def train(epochs=6, batch_size=32, lr=1.5e-4):
    print("=======================================================", flush=True)
    print("FastPlateOCR CCT-S-V2 Fine-Tuning for Indian ANPR", flush=True)
    print(f"Target Epochs: {epochs}, Batch Size: {batch_size}, Initial LR: {lr}", flush=True)
    print(f"Hardware: {num_cpus} CPU threads active", flush=True)
    print("=======================================================", flush=True)
    
    plate_cfg = load_plate_config_from_yaml(pathlib.Path(CONFIG_YAML))
    alphabet = plate_cfg.alphabet
    
    # 1. Load pretrained model
    print(f"Loading pretrained base: {PRETRAINED_KERAS}...", flush=True)
    full_model = load_keras_model(pathlib.Path(PRETRAINED_KERAS), plate_cfg)
    plate_output = full_model.get_layer('plate').output
    model = keras.Model(inputs=full_model.inputs, outputs=plate_output, name="FastPlateOCR_Indian")
    
    # Freeze layers 0 to 8 (conv_stem, patch_extractor, mlp, pos_emb, transformer_block_1..3)
    freeze_up_to = 9
    for i, layer in enumerate(model.layers[:freeze_up_to]):
        layer.trainable = False
    for layer in model.layers[freeze_up_to:]:
        layer.trainable = True
        
    trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
    total_count = model.count_params()
    print(f"Model architecture: Total params: {total_count:,}, Trainable params: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)", flush=True)
    
    # 2. In-Memory Datasets
    x_train, y_train, train_gts = load_data_into_memory(TRAIN_CSV, alphabet, max_samples=1200)
    x_val, y_val, val_gts = load_data_into_memory(VAL_CSV, alphabet, max_samples=300)
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Optimizer & Loss
    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    # 4. Initial Baseline Evaluation
    print("\n--- Zero-Shot Pretrained Baseline on Validation Set ---", flush=True)
    val_loss = 0.0
    val_preds = []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb, training=False)
            l = loss_fn(yb, out)
            val_loss += float(l.detach().cpu())
            preds = decode_predictions(out, alphabet)
            val_preds.extend(preds)
            
    val_loss /= len(val_loader)
    exact_acc, char_acc, avg_ed = compute_metrics(val_preds, val_gts)
    print(f"[Epoch 00 (Pretrained Zero-Shot)]: Val Loss: {val_loss:.4f} | Exact Match: {exact_acc:.2f}% | Char Acc: {char_acc:.2f}% | Avg ED: {avg_ed:.2f}", flush=True)
    
    history = [{
        "epoch": 0,
        "train_loss": None,
        "val_loss": round(val_loss, 4),
        "val_exact_acc": round(exact_acc, 2),
        "val_char_acc": round(char_acc, 2),
        "val_avg_ed": round(avg_ed, 2)
    }]
    
    best_exact_acc = exact_acc
    best_epoch = 0
    best_weights_path = os.path.join(OUTPUT_DIR, "cct_s_v2_indian_best.keras")
    last_weights_path = os.path.join(OUTPUT_DIR, "cct_s_v2_indian_last.keras")
    
    # 5. Training Loop
    total_steps_per_epoch = len(train_loader)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_train_loss = 0.0
        
        for step, (xb, yb) in enumerate(train_loader, 1):
            loss_val = model.train_on_batch(xb, yb)
            loss_num = float(loss_val[0] if isinstance(loss_val, list) else loss_val)
            running_train_loss += loss_num
            if step % 10 == 0 or step == total_steps_per_epoch:
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Step {step:02d}/{total_steps_per_epoch:02d} | Step Loss: {loss_num:.4f}", flush=True)
                
        train_loss = running_train_loss / total_steps_per_epoch
        
        # Validation Evaluation
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb, training=False)
                l = loss_fn(yb, out)
                val_loss += float(l.detach().cpu())
                preds = decode_predictions(out, alphabet)
                val_preds.extend(preds)
                
        val_loss /= len(val_loader)
        exact_acc, char_acc, avg_ed = compute_metrics(val_preds, val_gts)
        epoch_time = time.time() - t0
        
        print(f">>> [Epoch {epoch:02d}/{epochs:02d}] Finished in {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Exact: {exact_acc:.2f}% | Char Acc: {char_acc:.2f}% | Avg ED: {avg_ed:.2f}", flush=True)
        
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_exact_acc": round(exact_acc, 2),
            "val_char_acc": round(char_acc, 2),
            "val_avg_ed": round(avg_ed, 2),
            "epoch_time_s": round(epoch_time, 1)
        })
        
        # Save checkpoints
        model.save(last_weights_path)
        if exact_acc > best_exact_acc:
            best_exact_acc = exact_acc
            best_epoch = epoch
            model.save(best_weights_path)
            print(f"  *** NEW BEST MODEL SAVED to {best_weights_path} (Val Exact: {best_exact_acc:.2f}%) ***", flush=True)
            
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "FastPlateOCR CCT-S-V2 (Indian Fine-Tuned)",
            "pretrained_base": PRETRAINED_KERAS,
            "best_epoch": best_epoch,
            "best_val_exact_acc": best_exact_acc,
            "history": history
        }, f, indent=2)
        
    print("\n=======================================================", flush=True)
    print("Fine-Tuning Finished Successfully!", flush=True)
    print(f"Best Epoch: {best_epoch} with Val Exact Accuracy: {best_exact_acc:.2f}%", flush=True)
    print(f"Best checkpoint: {best_weights_path}", flush=True)
    print(f"Training history saved to {history_path}", flush=True)
    print("=======================================================", flush=True)

if __name__ == "__main__":
    train(epochs=6, batch_size=32, lr=1.5e-4)
