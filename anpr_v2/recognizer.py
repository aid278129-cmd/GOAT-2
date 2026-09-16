"""
anpr_v2/recognizer.py
PaddleOCR PP-OCR Recognition Engine via ONNX Runtime (Phase 7 & 8).

Optimized execution:
1. Fast-path direct recognition (8-12ms on CPU) for standard single-row plates (aspect ratio >= 2.2).
2. Line-aware detection & multi-line reading for two-row square plates (MoRTH Rule 50).
3. Clean fallback adapter for baseline CRNN model during A/B comparative testing.
"""

import os
import time
import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from anpr_v2.config import CONFIG
from anpr_v2.validator import clean_ocr_raw_tokens

@dataclass
class OCRResult:
    text: str
    raw_text: str
    confidence: float
    character_confidences: List[float] = field(default_factory=list)
    engine: str = "PP-OCRv4 (ONNX)"
    inference_ms: float = 0.0
    is_two_row: bool = False
    lines_detected: int = 1

class PPOCRRecognizer:
    def __init__(self):
        self._engine = None
        self._is_ready = False
        self._init_engine()

    def _init_engine(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._is_ready = True
            print("[ANPR V2] RapidOCR / PP-OCRv4 ONNX Recognition Engine initialized successfully.")
        except Exception as e:
            print(f"[ANPR V2 Error] Failed to initialize PP-OCR recognizer: {e}")
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def recognize(self, crop_bgr: np.ndarray) -> OCRResult:
        """
        Runs PP-OCR recognition on the plate crop.
        Uses fast direct recognition (sub-10ms) for single-row plates,
        and line-aware decomposition for square / 2-row plates.
        """
        if not self._is_ready or crop_bgr is None or crop_bgr.size == 0:
            return OCRResult(
                text="",
                raw_text="",
                confidence=0.0,
                engine="None",
                inference_ms=0.0
            )

        t0 = time.perf_counter()
        h, w = crop_bgr.shape[:2]
        aspect_ratio = w / float(max(1, h))
        
        try:
            # 1. If aspect ratio >= 2.2, attempt ultra-fast direct recognition first (~8-12ms)
            if aspect_ratio >= 2.2 and hasattr(self._engine, "text_recognizer"):
                rec_res, _ = self._engine.text_recognizer([crop_bgr])
                if rec_res and len(rec_res) > 0:
                    raw_str, score = rec_res[0]
                    cleaned = clean_ocr_raw_tokens(raw_str)
                    conf = float(score)
                    
                    # If high confidence and valid length, accept immediately
                    if len(cleaned) >= 6 and conf >= 0.60:
                        t_ms = (time.perf_counter() - t0) * 1000.0
                        return OCRResult(
                            text=cleaned,
                            raw_text=str(raw_str),
                            confidence=round(conf, 3),
                            character_confidences=[conf] * len(cleaned),
                            engine="PP-OCRv4 Fast (ONNX)",
                            inference_ms=round(t_ms, 2),
                            is_two_row=False,
                            lines_detected=1
                        )

            # 2. Line-aware decomposition (handles two-row plates or difficult single-row crops)
            result, _ = self._engine(crop_bgr)
            t_ms = (time.perf_counter() - t0) * 1000.0

            if result and len(result) > 0:
                # Sort lines top-to-bottom
                sorted_lines = sorted(result, key=lambda item: item[0][0][1])
                
                raw_text_parts = []
                scores = []
                
                for item in sorted_lines:
                    text_line = item[1].strip()
                    score = float(item[2])
                    if text_line:
                        raw_text_parts.append(text_line)
                        scores.append(score)
                
                full_raw = " ".join(raw_text_parts)
                cleaned = clean_ocr_raw_tokens("".join(raw_text_parts))
                avg_score = float(np.mean(scores)) if scores else 0.0
                is_two_row = len(sorted_lines) >= 2
                
                return OCRResult(
                    text=cleaned,
                    raw_text=full_raw,
                    confidence=round(avg_score, 3),
                    character_confidences=[avg_score] * len(cleaned),
                    engine="PP-OCRv4 Multi-Line (ONNX)",
                    inference_ms=round(t_ms, 2),
                    is_two_row=is_two_row,
                    lines_detected=len(sorted_lines)
                )

            # 3. Direct recognition fallback
            if hasattr(self._engine, "text_recognizer"):
                rec_res, _ = self._engine.text_recognizer([crop_bgr])
                t_ms = (time.perf_counter() - t0) * 1000.0
                if rec_res and len(rec_res) > 0:
                    raw_str, score = rec_res[0]
                    cleaned = clean_ocr_raw_tokens(raw_str)
                    conf = float(score)
                    return OCRResult(
                        text=cleaned,
                        raw_text=str(raw_str),
                        confidence=round(conf, 3),
                        character_confidences=[conf] * len(cleaned),
                        engine="PP-OCRv4 Direct (ONNX)",
                        inference_ms=round(t_ms, 2),
                        is_two_row=False,
                        lines_detected=1
                    )

        except Exception as e:
            print(f"[ANPR V2 Warning] OCR recognition exception: {e}")

        t_ms = (time.perf_counter() - t0) * 1000.0
        return OCRResult(
            text="",
            raw_text="",
            confidence=0.0,
            engine="PP-OCRv4 (ONNX)",
            inference_ms=round(t_ms, 2),
            is_two_row=False,
            lines_detected=0
        )


class BaselineCRNNRecognizer:
    """Baseline CRNN Recognizer for A/B benchmarking comparison against V2."""
    def __init__(self):
        self._session = None
        self._is_ready = False
        self._init_session()

    def _init_session(self):
        if os.path.exists(CONFIG.crnn_plate_onnx_path):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    CONFIG.crnn_plate_onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                self._is_ready = True
            except Exception as e:
                print(f"[CRNN Baseline Error] {e}")
                self._is_ready = False

    def recognize(self, crop_bgr: np.ndarray) -> OCRResult:
        if not self._is_ready or crop_bgr is None or crop_bgr.size == 0:
            return OCRResult(text="", raw_text="", confidence=0.0, engine="CRNN Baseline (Unavailable)")
            
        t0 = time.perf_counter()
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (160, 32))
            norm = (resized.astype(np.float32) / 127.5) - 1.0
            inp = np.expand_dims(np.expand_dims(norm, axis=0), axis=0)
            
            ort_inputs = {self._session.get_inputs()[0].name: inp}
            preds = self._session.run(None, ort_inputs)[0]
            
            vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            idx2char = {i + 1: c for i, c in enumerate(vocab)}
            
            if preds.ndim == 3 and preds.shape[1] == 1:
                logits = preds[:, 0, :]
            else:
                logits = preds[0]
                
            char_indices = np.argmax(logits, axis=1)
            raw_chars = []
            prev = 0
            for idx in char_indices:
                if idx != 0 and idx != prev:
                    raw_chars.append(idx2char.get(idx, ""))
                prev = idx
                
            raw_str = "".join(raw_chars)
            t_ms = (time.perf_counter() - t0) * 1000.0
            return OCRResult(
                text=raw_str,
                raw_text=raw_str,
                confidence=0.75 if raw_str else 0.0,
                engine="CRNN Baseline",
                inference_ms=round(t_ms, 2)
            )
        except Exception as e:
            return OCRResult(text="", raw_text="", confidence=0.0, engine=f"CRNN Error: {e}")

_RECOGNIZER_INSTANCE = None

def get_recognizer(backend: Optional[str] = None):
    global _RECOGNIZER_INSTANCE
    b = (backend or CONFIG.ocr_backend).lower()
    if b == "crnn":
        return BaselineCRNNRecognizer()
    if _RECOGNIZER_INSTANCE is None:
        _RECOGNIZER_INSTANCE = PPOCRRecognizer()
    return _RECOGNIZER_INSTANCE
