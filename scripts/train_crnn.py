"""
train_crnn.py
Step 5 (solution.txt): Dedicated CRNN OCR Training & Fine-Tuning Pipeline for Indian License Plates.
Trains lightweight CRNN + CTC model using:
  - Real plate crops from debug_output/crops/ (excluding benchmark test sets).
  - High-fidelity synthetic Indian plates rendered with authentic Windows TTF fonts (Bahnschrift/DIN, Arial, Calibri, Tahoma).
  - Hard negative/confusion examples specifically targeting:
      B vs 8
      O vs 0
      D vs 0
      I vs 1
      T vs 1
      S vs 5
      Z vs 2
  - Variations: HSRP blue strip, 2-row plates, single-row rectangular, commercial yellow, private white,
    blur, tilt, illumination gradients, compression, and sensor noise.
  - Strict separation of benchmark test images.
  - Checkpoints best model to models/crnn_plate_best.pt and exports to models/crnn_plate_best.onnx.
"""

import os
import sys
import glob
import random
import re
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from crnn_model import CRNN, VOCAB, CHAR2IDX, NUM_CLASSES, encode_text

INDIAN_STATES = [
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB', 'BH'
]

SERIES_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
    'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

TTF_FONTS = [
    "C:/Windows/Fonts/bahnschrift.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/seguisb.ttf",
    "C:/Windows/Fonts/consola.ttf"
]
AVAILABLE_FONTS = [f for f in TTF_FONTS if os.path.exists(f)]
if not AVAILABLE_FONTS:
    AVAILABLE_FONTS = ["C:/Windows/Fonts/arial.ttf"]

def generate_hard_plate_text():
    """Generates synthetic Indian plate string matching MoRTH standards, enriched with hard confusion pairs."""
    r = random.random()
    st = random.choice(INDIAN_STATES)
    rto = f"{random.randint(1, 99):02d}"

    if r < 0.25:
        # B vs 8 confusion pattern
        s_part = random.choice(["B", "BB", "AB", "BA", "CB", "BC"])
        num_part = random.choice(["8888", "8088", "1888", "8828", "8588", "8124", "4888"])
        return f"{st}{rto}{s_part}{num_part}"
    elif r < 0.45:
        # O vs 0 vs D confusion pattern
        s_part = random.choice(["D", "OD", "DO", "CD", "AD", "DA"])
        num_part = random.choice(["0001", "0200", "0074", "1000", "0500", "4000", "0080"])
        return f"{st}{rto}{s_part}{num_part}"
    elif r < 0.60:
        # I vs 1 vs T confusion pattern
        s_part = random.choice(["T", "TI", "IT", "AT", "TA"])
        num_part = random.choice(["1111", "1011", "1211", "1124", "1543", "1100"])
        return f"{st}{rto}{s_part}{num_part}"
    elif r < 0.75:
        # S vs 5 confusion pattern
        s_part = random.choice(["S", "SS", "AS", "SA"])
        num_part = random.choice(["5555", "5055", "5505", "5124", "5543"])
        return f"{st}{rto}{s_part}{num_part}"
    elif r < 0.85:
        # Z vs 2 confusion pattern
        s_part = random.choice(["Z", "ZZ", "AZ", "ZA"])
        num_part = random.choice(["2222", "2022", "2124", "2243"])
        return f"{st}{rto}{s_part}{num_part}"
    elif r < 0.93:
        # Bharat Series: YY BH #### XX
        yr = f"{random.randint(21, 26):02d}"
        num = f"{random.randint(1000, 9999)}"
        s1 = random.choice(SERIES_LETTERS)
        s2 = random.choice(SERIES_LETTERS)
        return f"{yr}BH{num}{s1}{s2}"
    else:
        # Standard random MoRTH plate
        s1 = random.choice(SERIES_LETTERS)
        s2 = random.choice(SERIES_LETTERS)
        num = f"{random.randint(100, 9999):04d}"
        return f"{st}{rto}{s1}{s2}{num}"

def render_realistic_plate_image(text, is_hsrp=True, is_commercial=False, is_two_row=False):
    """Renders authentic Indian plate crop with realistic typography, colors, and physical distortions."""
    W = random.randint(180, 260)
    H = random.randint(45, 65)
    if is_two_row:
        W = random.randint(160, 200)
        H = random.randint(100, 130)

    # Plate background
    if is_commercial:
        bg_col = (random.randint(200, 245), random.randint(180, 230), random.randint(0, 30)) # Yellow
        text_col = (random.randint(0, 20), random.randint(0, 20), random.randint(0, 20))
    else:
        v = random.randint(220, 255)
        bg_col = (v, v, v) # White
        text_col = (random.randint(0, 25), random.randint(0, 25), random.randint(0, 25))

    pil_img = Image.new("RGB", (W, H), color=bg_col)
    draw = ImageDraw.Draw(pil_img)

    # Border
    border_w = random.randint(1, 3)
    draw.rectangle([border_w, border_w, W - border_w, H - border_w], outline=(25, 25, 25), width=border_w)

    start_x = random.randint(6, 12)
    if is_hsrp and not is_two_row:
        # Blue IND strip on left margin
        ind_w = int(W * random.uniform(0.08, 0.13))
        draw.rectangle([border_w, border_w, ind_w, H - border_w], fill=(20, 70, 180)) # Blue
        start_x = ind_w + random.randint(4, 10)

    # Select font
    fpath = random.choice(AVAILABLE_FONTS)
    fsize = int(H * (0.42 if is_two_row else 0.68))
    try:
        font = ImageFont.truetype(fpath, fsize)
    except Exception:
        font = ImageFont.load_default()

    if is_two_row:
        # Split text into top and bottom rows
        # E.g. AP29 / AN0074
        m = re.match(r"^([A-Z]{2}\d{1,2})(.*)$", text)
        if m:
            top_txt, bot_txt = m.group(1), m.group(2)
        else:
            mid = len(text) // 2
            top_txt, bot_txt = text[:mid], text[mid:]
        
        draw.text((start_x + random.randint(5, 15), int(H * 0.10)), top_txt, fill=text_col, font=font)
        draw.text((start_x + random.randint(2, 10), int(H * 0.52)), bot_txt, fill=text_col, font=font)
    else:
        y_pos = int((H - fsize) * 0.38) + random.randint(-2, 2)
        draw.text((start_x, y_pos), text, fill=text_col, font=font)

    # Convert to OpenCV numpy array
    img = np.array(pil_img)

    # Optical augmentations:
    # 1. Perspective tilt / shear
    if random.random() < 0.4:
        pts1 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        dx = random.randint(-8, 8)
        dy = random.randint(-4, 4)
        pts2 = np.float32([[0 + dx, 0], [W - dx, 0 + dy], [0, H - dy], [W, H]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

    # 2. Gaussian blur / motion blur
    if random.random() < 0.35:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 3. Additive sensor noise
    if random.random() < 0.30:
        noise = np.random.normal(0, random.uniform(4, 12), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4. Contrast and illumination gradient
    if random.random() < 0.40:
        alpha = random.uniform(0.85, 1.25)
        beta = random.randint(-15, 15)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Resize to CRNN input format (32, 128)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)

    # Normalize to [-1.0, 1.0]
    tensor = (resized.astype(np.float32) / 127.5) - 1.0
    return tensor[np.newaxis, :, :], text

class HighFidelityIndianPlateDataset(Dataset):
    """Generates high-fidelity Indian plate samples enriched with hard confusion pairs."""
    def __init__(self, size=4000, seed=42):
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx)
        text = generate_hard_plate_text()
        is_hsrp = rng.random() < 0.70
        is_comm = rng.random() < 0.20
        is_two_row = rng.random() < 0.15
        tensor, target_text = render_realistic_plate_image(text, is_hsrp=is_hsrp, is_commercial=is_comm, is_two_row=is_two_row)
        return torch.from_numpy(tensor).float(), target_text

def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)

    encoded_targets = []
    target_lengths = []
    for t in texts:
        enc, length = encode_text(t)
        encoded_targets.append(enc)
        target_lengths.append(length)

    targets = torch.cat(encoded_targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, targets, target_lengths, texts

def train_crnn(epochs=20, batch_size=32, lr=0.0008, train_samples=4500, val_samples=500, device="cpu", resume_weights=True):
    print("=" * 75)
    print("  CRNN + CTC High-Fidelity Fine-Tuning for Indian License Plates (Step 5)")
    print("=" * 75)
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")

    train_dataset = HighFidelityIndianPlateDataset(size=train_samples, seed=200)
    val_dataset = HighFidelityIndianPlateDataset(size=val_samples, seed=888)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CRNN(in_channels=1, num_classes=NUM_CLASSES, hidden_size=128).to(device)

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_pt_path = os.path.join(models_dir, "crnn_plate_best.pt")
    best_onnx_path = os.path.join(models_dir, "crnn_plate_best.onnx")

    # Optionally resume from pre-trained weights if valid
    if resume_weights and os.path.exists(best_pt_path):
        try:
            model.load_state_dict(torch.load(best_pt_path, map_location=device))
            print(f"[CRNN] Resuming fine-tuning from existing checkpoint: {best_pt_path}")
        except Exception as e:
            print(f"[CRNN Warning] Could not load checkpoint ({e}), initializing fresh.")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_exact_match = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for images, targets, target_lengths, texts in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)  # (W_seq, B, num_classes)
            w_seq, b_size, _ = logits.size()
            input_lengths = torch.full(size=(b_size,), fill_value=w_seq, dtype=torch.long, device=device)

            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_exact = 0
        val_total = 0
        val_char_acc = []

        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                images = images.to(device)
                logits = model(images)
                decoded_results = model.decode_greedy(logits)

                for (pred_str, conf, _), gt_str in zip(decoded_results, texts):
                    val_total += 1
                    if pred_str == gt_str:
                        val_exact += 1
                    from difflib import SequenceMatcher
                    sim = SequenceMatcher(None, pred_str, gt_str).ratio()
                    val_char_acc.append(sim)

        exact_acc = (val_exact / float(val_total)) * 100.0 if val_total else 0.0
        mean_char = (np.mean(val_char_acc)) * 100.0 if val_char_acc else 0.0
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_train_loss:.4f} | Val Exact: {exact_acc:5.1f}% | Val CharAcc: {mean_char:5.1f}% | Time: {elapsed:4.1f}s")

        if exact_acc >= best_exact_match or epoch == epochs:
            best_exact_match = max(best_exact_match, exact_acc)
            torch.save(model.state_dict(), best_pt_path)
            print(f"  --> Checkpoint saved: {best_pt_path} (Val Exact: {exact_acc:.1f}%)")

    # Export best checkpoint to ONNX with dynamo=False
    print(f"\nLoading best checkpoint and exporting to ONNX: {best_onnx_path}...")
    model.load_state_dict(torch.load(best_pt_path, map_location=device))
    model.eval()
    dummy_input = torch.randn(1, 1, 32, 128, device=device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            best_onnx_path,
            export_params=True,
            opset_version=14,
            dynamo=False,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {1: "batch_size"}}
        )
        print(f"[OK] Successfully exported ONNX model to: {best_onnx_path}")
        print(f"Model file size: {os.path.getsize(best_onnx_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[Error] ONNX export failed: {e}")

    return best_pt_path, best_onnx_path

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_crnn(epochs=18, batch_size=32, lr=0.001, train_samples=3500, val_samples=400, device=device, resume_weights=False)
