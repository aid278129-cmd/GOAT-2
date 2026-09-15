"""
crnn_ocr.py
Step 4 (solution.txt): Dedicated CRNN OCR Recognizer for Indian License Plates.
Operates on rectified plate crops and predicts the registration string directly.
Supports both ONNX Runtime (fast CPU inference) and PyTorch backends.
Integrated with the Position-Aware Indian Plate Decoder.
"""

import os
import re
import cv2
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CRNN_ONNX_PATH = os.path.join(BASE_DIR, "models", "crnn_plate_best.onnx")
CRNN_PT_PATH = os.path.join(BASE_DIR, "models", "crnn_plate_best.pt")

VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX2CHAR = {i + 1: c for i, c in enumerate(VOCAB)}

INDIAN_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK',
    'UA', 'UP', 'WB', 'BH'
}

STATE_REPAIRS = {
    'MG': 'MH', 'MN': 'MH', 'NH': 'MH',
    'OL': 'DL', 'D1': 'DL', 'DI': 'DL',
    'K1': 'KL', 'KI': 'KL',
    'TM': 'TN', 'TI': 'TN', 'TF': 'TN', 'TH': 'TN',
    'HR': 'HR', 'HA': 'HR',
    'GJ': 'GJ', 'CJ': 'GJ',
    'VP': 'UP', 'UF': 'UP',
    'AP': 'AP', 'AF': 'AP',
    'TS': 'TS', 'T5': 'TS',
    'WB': 'WB', 'WE': 'WB',
    'PB': 'PB', 'P8': 'PB',
    'RJ': 'RJ', 'R1': 'RJ',
    'KH': 'KA',
}

DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'}
LETTER_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'U': '0', 'I': '1', 'L': '1', 'T': '1', 'J': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'P': '9'}

def decode_positional_syntax(raw_text: str):
    """
    Step 3 (solution.txt): Position-Aware Indian Plate Decoder.
    Contextually disambiguates characters based strictly on structural position.
    """
    if not raw_text:
        return ""

    s = re.sub(r"[^A-Z0-9]", "", str(raw_text).upper())
    if len(s) < 4:
        return s

    # Strip leading HSRP 'IND' / '1ND' artifact if present
    for ind_prefix in ["IND", "1ND", "LND", "TND"]:
        if s.startswith(ind_prefix) and len(s) >= len(ind_prefix) + 5:
            cand_after = s[len(ind_prefix):]
            if cand_after[:2] in INDIAN_STATES or cand_after[:2] in STATE_REPAIRS or cand_after[1:3] in INDIAN_STATES:
                s = cand_after
                break

    # Strip single-char border/screw artifact if followed by a valid state code (e.g. MMP42 -> MP42, 0DL3C -> DL3C)
    if len(s) >= 7 and s[:2] not in INDIAN_STATES and s[:2] not in STATE_REPAIRS:
        if s[1:3] in INDIAN_STATES or s[1:3] in STATE_REPAIRS:
            s = s[1:]

    # Check two-row inversion: e.g. '0074AP29AN' -> 'AP29AN0074'
    if s[:2] not in INDIAN_STATES and s[:2] not in STATE_REPAIRS:
        for i in range(2, len(s) - 3):
            st = s[i:i+2]
            rep_st = STATE_REPAIRS.get(st, st)
            if rep_st in INDIAN_STATES:
                m_rto = re.match(r"^([A-Z]{2}\d{1,2})", rep_st + s[i+2:])
                if m_rto:
                    state_rto = m_rto.group(1)
                    rem = s[:i] + s[i+len(state_rto):]
                    rem_letters = "".join(re.findall(r"[A-Z]+", rem))
                    rem_digits = "".join(re.findall(r"\d+", rem))
                    s = f"{state_rto}{rem_letters}{rem_digits}"
                    break

    # Check BH series: YY BH #### XX
    if len(s) >= 8:
        cand_bh = s[2:4]
        if cand_bh in ['BH', '8H', '81', 'B1', '8N']:
            c0 = LETTER_TO_DIGIT.get(s[0], s[0])
            c1 = LETTER_TO_DIGIT.get(s[1], s[1])
            if c0.isdigit() and c1.isdigit():
                mid_digits = "".join(LETTER_TO_DIGIT.get(c, c) for c in s[4:min(8, len(s))])
                tail_letters = "".join(DIGIT_TO_LETTER.get(c, c) for c in s[8:])
                return f"{c0}{c1}BH{mid_digits}{tail_letters}"

    chars = list(s)

    # Position 0 & 1: State prefix must be letters
    for i in [0, 1]:
        if i < len(chars) and chars[i].isdigit():
            chars[i] = DIGIT_TO_LETTER.get(chars[i], chars[i])

    state = "".join(chars[:2])
    if state in STATE_REPAIRS:
        state = STATE_REPAIRS[state]
        chars[0] = state[0]
        chars[1] = state[1]

    if state not in INDIAN_STATES:
        # If not a valid state yet, try repairing second char
        if len(chars) >= 2 and chars[1].isdigit():
            rep_c1 = DIGIT_TO_LETTER.get(chars[1], chars[1])
            cand = chars[0] + rep_c1
            if cand in INDIAN_STATES:
                chars[1] = rep_c1
                state = cand

    if state in INDIAN_STATES and len(chars) >= 6:
        # Position 2 must be digit
        if len(chars) > 2 and not chars[2].isdigit():
            chars[2] = LETTER_TO_DIGIT.get(chars[2], chars[2])

        # Trailing 4 positions: Serial number must strictly be digits
        tail_len = min(4, len(chars) - 4)
        if tail_len > 0:
            for i in range(len(chars) - tail_len, len(chars)):
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

        if len(chars) == 10:
            # Pos 0, 1: State letters (handled above)
            # Pos 2: District digit (1-9)
            # Pos 4, 5: Series letters (e.g. AB or BL)
            chars[4] = DIGIT_TO_LETTER.get(chars[4], chars[4])
            chars[5] = DIGIT_TO_LETTER.get(chars[5], chars[5])
            # Pos 3: Can be 2nd district digit (MH12AB1234) OR 1st series letter (DL3CBL1234)
            # If chars[3] is letter (e.g. 'C') keep it; if digit, keep it
            if chars[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                pass  # Keep as digit (e.g. MH12...)
            elif chars[3] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
                pass  # Keep as series letter (e.g. DL3CBL...)
            else:
                # Disambiguate based on confusable pairs
                chars[3] = DIGIT_TO_LETTER.get(chars[3], chars[3])
        elif len(chars) == 9:
            # Check if pos 3 is letter or digit
            if chars[3] in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z'] and chars[4].isalpha():
                chars[3] = DIGIT_TO_LETTER.get(chars[3], chars[3])
                chars[4] = DIGIT_TO_LETTER.get(chars[4], chars[4])
            else:
                chars[3] = LETTER_TO_DIGIT.get(chars[3], chars[3])
                chars[4] = DIGIT_TO_LETTER.get(chars[4], chars[4])
        elif len(chars) == 8:
            chars[2] = LETTER_TO_DIGIT.get(chars[2], chars[2])
            chars[3] = LETTER_TO_DIGIT.get(chars[3], chars[3])

    res = "".join(chars)
    # Check optical screw/hologram '0' artifact in 5-digit tail
    m_ext = re.match(r"^([A-Z]{2}\d{1,2}[A-Z]{1,3})0(\d{4})$", res)
    if m_ext:
        res = f"{m_ext.group(1)}{m_ext.group(2)}"

    return res


CONFUSABLE_PAIRS = {
    'B': ['8'], '8': ['B'],
    'O': ['0', 'D', 'Q'], '0': ['O', 'D', 'Q'],
    'D': ['0', 'O'], 'Q': ['0', 'O'],
    'I': ['1', 'T', 'L', 'J'], '1': ['I', 'T', 'L', 'J'],
    'T': ['1', 'I'], 'L': ['1', 'I'], 'J': ['1', 'I'],
    'S': ['5'], '5': ['S'],
    'Z': ['2'], '2': ['Z'],
    'G': ['6'], '6': ['G'],
    'A': ['4'], '4': ['A'],
}

def extract_ctc_segments(probs):
    """
    Extracts collapsed non-blank character segment emission peaks and their probability distributions.
    probs: (W_seq, 37) numpy array of softmax probabilities across time steps.
    Returns: list of slot probability dicts {char: prob} for each emitted character.
    """
    w_seq = probs.shape[0]
    max_idx = np.argmax(probs, axis=1)

    segments = []
    curr_t = []
    curr_char_idx = 0

    for t in range(w_seq):
        idx = max_idx[t]
        if idx == 0:
            if curr_t:
                segments.append((curr_char_idx, curr_t))
                curr_t = []
                curr_char_idx = 0
        else:
            if idx == curr_char_idx:
                curr_t.append(t)
            else:
                if curr_t:
                    segments.append((curr_char_idx, curr_t))
                curr_t = [t]
                curr_char_idx = idx
    if curr_t:
        segments.append((curr_char_idx, curr_t))

    slot_probs = []
    for char_idx, t_steps in segments:
        max_p_per_char = {}
        for c_idx in range(1, len(VOCAB) + 1):
            ch = IDX2CHAR[c_idx]
            max_p_per_char[ch] = float(np.max(probs[t_steps, c_idx]))

        total_p = sum(max_p_per_char.values()) + 1e-9
        norm_p = {c: p / total_p for c, p in max_p_per_char.items()}
        slot_probs.append(norm_p)

    return slot_probs

def decode_probabilistic_syntax(slot_probs):
    """
    Step 6 (solution.txt): Probabilistic Positional Decoding for Indian Plates.
    Uses OCR character probabilities plus Indian registration position constraints.
    Generates several legal candidate strings and scores them using:
      - CRNN character probability
      - position validity
      - MoRTH syntax validity
      - state code validity
    Returns: (best_plate, confidence, has_unresolved_ambiguity, top_candidates)
    """
    N = len(slot_probs)
    if N < 4:
        greedy = "".join(max(sp.items(), key=lambda x: x[1])[0] for sp in slot_probs)
        return greedy, 0.5, False, []

    slot_candidates = []
    for i, sp in enumerate(slot_probs):
        sorted_chars = sorted(sp.items(), key=lambda x: x[1], reverse=True)
        top_chars = dict(sorted_chars[:4])
        primary_char = sorted_chars[0][0]

        if primary_char in CONFUSABLE_PAIRS:
            for alt in CONFUSABLE_PAIRS[primary_char]:
                if alt not in top_chars:
                    top_chars[alt] = sp.get(alt, 1e-4)

        slot_candidates.append(top_chars)

    def evaluate_candidate_string(cand_str):
        log_lik = 0.0
        mismatches = 0

        for i, c in enumerate(cand_str):
            p = slot_candidates[i].get(c, 1e-5)
            log_lik += np.log(max(1e-5, p))

        state_code = cand_str[:2]
        is_state_valid = state_code in INDIAN_STATES or state_code in STATE_REPAIRS

        is_morth = bool(re.match(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$", cand_str))
        is_bh = bool(re.match(r"^\d{2}BH\d{4}[A-Z]{1,2}$", cand_str))

        if len(cand_str) == 10:
            for i in [0, 1, 4, 5]:
                if not cand_str[i].isalpha():
                    mismatches += 1
            for i in [2, 6, 7, 8, 9]:
                if not cand_str[i].isdigit():
                    mismatches += 1
            # Note: cand_str[3] can be 2nd district digit (MH12AB1234) or 1st series letter (DL3CBL1234). Both are valid MoRTH!
        elif len(cand_str) == 9:
            for i in [0, 1]:
                if not cand_str[i].isalpha():
                    mismatches += 1
            for i in range(len(cand_str) - 4, len(cand_str)):
                if not cand_str[i].isdigit():
                    mismatches += 1
        elif len(cand_str) == 8:
            for i in [0, 1]:
                if not cand_str[i].isalpha():
                    mismatches += 1
            for i in range(2, 4):
                if not cand_str[i].isdigit():
                    mismatches += 1
            for i in range(4, 8):
                if not cand_str[i].isdigit():
                    mismatches += 1

        score = log_lik
        if is_morth or is_bh:
            score += 8.0
        elif re.match(r"^[A-Z]{2}\d{1,4}[A-Z]{0,2}\d{1,4}$", cand_str):
            score += 4.0

        if is_state_valid:
            score += 4.0
        else:
            score -= 6.0

        score -= mismatches * 5.0
        return score, is_morth or is_bh, is_state_valid, mismatches

    beam = [("", 0.0)]
    beam_width = 16

    for i in range(N):
        new_beam = []
        for prefix, prefix_score in beam:
            for ch, p in slot_candidates[i].items():
                cand_score = prefix_score + np.log(max(1e-5, p))
                if i in [0, 1] and ch.isdigit():
                    cand_score -= 3.0
                elif i in [2] and ch.isalpha():
                    cand_score -= 3.0
                elif i >= N - 4 and ch.isalpha():
                    cand_score -= 3.0
                new_beam.append((prefix + ch, cand_score))
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    evaluated_candidates = []
    for cand_str, _ in beam:
        score, is_valid_syntax, is_valid_state, mismatches = evaluate_candidate_string(cand_str)
        avg_prob = float(np.mean([slot_candidates[i].get(cand_str[i], 0.0) for i in range(N)]))
        evaluated_candidates.append({
            "plate": cand_str,
            "score": score,
            "confidence": round(avg_prob, 3),
            "valid_syntax": is_valid_syntax,
            "valid_state": is_valid_state,
            "mismatches": mismatches
        })

    evaluated_candidates.sort(key=lambda x: x["score"], reverse=True)
    best = evaluated_candidates[0]

    has_unresolved = False
    if len(evaluated_candidates) > 1:
        runner_up = evaluated_candidates[1]
        if abs(best["score"] - runner_up["score"]) < 0.6 and best["valid_syntax"] == runner_up["valid_syntax"]:
            has_unresolved = True

    return best["plate"], best["confidence"], has_unresolved, evaluated_candidates[:5]

class CRNNOcrRecognizer:
    """
    Dedicated CRNN License Plate Recognition Engine.
    Operates on rectified plate crops and predicts the registration string directly.
    """
    def __init__(self, backend="onnx"):
        self.backend = backend.lower()
        self.session = None
        self.torch_model = None
        self._init_engine()

    def _init_engine(self):
        if self.backend == "onnx" and os.path.exists(CRNN_ONNX_PATH):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(CRNN_ONNX_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
                print(f"[CRNN OCR] Initialized ONNX Runtime session: {CRNN_ONNX_PATH}")
                return
            except Exception as e:
                print(f"[CRNN OCR Warning] ONNX Runtime failed ({e}), falling back to PyTorch.")

        if os.path.exists(CRNN_PT_PATH):
            from crnn_model import build_crnn_model
            self.torch_model = build_crnn_model(pretrained_path=CRNN_PT_PATH, device="cpu")
            self.torch_model.eval()
            print(f"[CRNN OCR] Initialized PyTorch model: {CRNN_PT_PATH}")
            self.backend = "pytorch"
        else:
            print(f"[CRNN OCR Warning] Neither ONNX nor PT model found at {CRNN_ONNX_PATH} / {CRNN_PT_PATH}")

    def preprocess_crop(self, crop_bgr):
        """Standardizes plate crop to (1, 1, 32, 128) normalized tensor."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
        tensor = (resized.astype(np.float32) / 127.5) - 1.0
        return tensor[np.newaxis, np.newaxis, :, :]  # (1, 1, 32, 128)

    def predict(self, crop_bgr, return_details=False):
        """
        Runs CRNN inference on plate crop and decodes with probabilistic positional Indian decoder.
        Returns: (cleaned_plate, raw_plate, confidence, [has_unresolved, top_candidates])
        """
        inp = self.preprocess_crop(crop_bgr)
        if inp is None:
            if return_details:
                return "", "", 0.0, True, []
            return "", "", 0.0

        raw_str = ""
        avg_conf = 0.0
        probs = None

        if self.session is not None:
            try:
                ort_inputs = {self.session.get_inputs()[0].name: inp}
                logits = self.session.run(None, ort_inputs)[0]  # (W_seq, 1, num_classes)
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
                if probs.ndim == 3:
                    probs = probs[:, 0, :]  # (W_seq, num_classes)

                max_indices = np.argmax(probs, axis=-1).squeeze()
                max_probs = np.max(probs, axis=-1).squeeze()

                char_list = []
                conf_list = []
                prev = 0
                for t in range(len(max_indices)):
                    idx = int(max_indices[t])
                    prob = float(max_probs[t])
                    if idx != 0 and idx != prev:
                        char_list.append(IDX2CHAR.get(idx, ""))
                        conf_list.append(prob)
                    prev = idx

                raw_str = "".join(char_list)
                avg_conf = float(np.mean(conf_list)) if conf_list else 0.5
            except Exception as e:
                pass

        elif self.torch_model is not None:
            try:
                tensor_t = torch.from_numpy(inp).float()
                with torch.no_grad():
                    logits = self.torch_model(tensor_t)
                    results = self.torch_model.decode_greedy(logits)
                    raw_str, avg_conf, _ = results[0]
                    t_probs = torch.softmax(logits, dim=-1).squeeze(1).cpu().numpy()
                    probs = t_probs
            except Exception as e:
                pass

        # Apply Probabilistic Positional Decoding if probs are available
        has_unresolved = False
        candidates = []
        cleaned_str = decode_positional_syntax(raw_str)

        if probs is not None:
            try:
                slot_probs = extract_ctc_segments(probs)
                if len(slot_probs) >= 4:
                    prob_plate, prob_conf, unresolved, candidates = decode_probabilistic_syntax(slot_probs)
                    has_unresolved = unresolved

                    # Check if probabilistic plate is valid
                    is_prob_valid = bool(re.match(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$", prob_plate)) or bool(re.match(r"^\d{2}BH\d{4}[A-Z]{1,2}$", prob_plate))
                    is_rule_valid = bool(re.match(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$", cleaned_str)) or bool(re.match(r"^\d{2}BH\d{4}[A-Z]{1,2}$", cleaned_str))

                    if is_prob_valid:
                        cleaned_str = prob_plate
                        avg_conf = max(avg_conf, prob_conf)
                    elif not is_rule_valid and prob_plate:
                        cleaned_str = prob_plate
            except Exception:
                pass

        if return_details:
            return cleaned_str, raw_str, round(avg_conf, 2), has_unresolved, candidates
        return cleaned_str, raw_str, round(avg_conf, 2)


_CRNN_RECOGNIZER = None

def get_crnn_recognizer():
    global _CRNN_RECOGNIZER
    if _CRNN_RECOGNIZER is None:
        _CRNN_RECOGNIZER = CRNNOcrRecognizer(backend="onnx")
    return _CRNN_RECOGNIZER
