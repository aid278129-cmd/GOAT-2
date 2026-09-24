"""
crnn_model.py
Lightweight CRNN (Convolutional Recurrent Neural Network) with CTC for Indian License Plate Recognition.
Architecture:
  - Input: (B, 1, 32, 128) Normalized Grayscale Plate Crop
  - 5-layer CNN Backbone with Batch Normalization and ReLU
  - Map-to-Sequence layer
  - 2-layer Bidirectional LSTM (hidden_size=128)
  - Linear Projection to 37 classes (Blank + 10 Digits + 26 Letters)
  - CTC Loss / Greedy Decoding with Character-Level Probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Vocabulary: Index 0 is CTC Blank, followed by Digits and Uppercase Letters
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB) + 1  # 37 (index 0 is CTC blank)

class CRNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, hidden_size=128):
        super(CRNN, self).__init__()

        # CNN Feature Extractor
        # Target: from (B, 1, 32, 128) -> (B, 256, 1, 31)
        self.cnn = nn.Sequential(
            # Block 1: 32x128 -> 16x64
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x64 -> 8x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x32 -> 4x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 4: 4x32 -> 2x32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 5: 2x32 -> 1x31
            nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # Sequence modeling with BiLSTM
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

        # Linear classifier: 2*hidden_size (due to bidirectional) -> num_classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, 32, 128)
        features = self.cnn(x)  # (B, 256, 1, W_seq)
        b, c, h, w = features.size()
        assert h == 1, f"Expected height 1 after CNN, got {h}"

        features = features.squeeze(2)          # (B, 256, W_seq)
        features = features.permute(2, 0, 1)    # (W_seq, B, 256) for PyTorch RNN

        rnn_out, _ = self.rnn(features)         # (W_seq, B, 2*hidden_size)
        logits = self.fc(rnn_out)               # (W_seq, B, num_classes)
        return logits

    def decode_greedy(self, logits):
        """
        Greedy CTC decoding of network logits.
        logits: (W_seq, B, num_classes) or (W_seq, num_classes)
        Returns: list of (decoded_string, avg_confidence, per_char_confidences)
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)

        w_seq, batch_size, num_classes = logits.size()
        probs = F.softmax(logits, dim=2)  # (W_seq, B, num_classes)
        max_probs, max_indices = torch.max(probs, dim=2)  # (W_seq, B)

        results = []
        for b in range(batch_size):
            char_list = []
            conf_list = []
            prev_idx = 0

            for t in range(w_seq):
                idx = int(max_indices[t, b].item())
                prob = float(max_probs[t, b].item())

                # CTC Rule: Collapse repeats and ignore blank token 0
                if idx != 0 and idx != prev_idx:
                    char_list.append(IDX2CHAR.get(idx, ""))
                    conf_list.append(prob)
                prev_idx = idx

            decoded_str = "".join(char_list)
            avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
            results.append((decoded_str, avg_conf, conf_list))

        return results


def encode_text(text: str):
    """Encodes a plate string into a tensor of character indices (excluding blank)."""
    clean_str = "".join(c for c in str(text).upper() if c in CHAR2IDX)
    indices = [CHAR2IDX[c] for c in clean_str]
    return torch.tensor(indices, dtype=torch.long), len(indices)


def build_crnn_model(pretrained_path=None, device="cpu"):
    """Instantiates the CRNN model and optionally loads pre-trained weights."""
    model = CRNN(in_channels=1, num_classes=NUM_CLASSES, hidden_size=128)
    if pretrained_path:
        import os
        if os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state)
            print(f"[CRNN] Loaded weights from {pretrained_path}")
    model.to(device)
    return model
