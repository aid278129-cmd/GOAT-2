"""
anpr_v2/fastplate_recognizer.py
Candidate OCR Adapter for FastPlateOCR (CCT-S-v2-global-model) via ONNX Runtime.

ISOLATED CANDIDATE ADAPTER — Does NOT replace production PP-OCRv4 recognizer.
Provides a clean interface compatible with the existing ANPR V2 recognizer concept:
recognize(crop_bgr) -> OCRResult or {"text": ..., "confidence": ..., "latencyMs": ...}
"""

import os
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

from anpr_v2.validator import clean_ocr_raw_tokens

@dataclass
class FastPlateOCRResult:
    text: str
    raw_text: str
    confidence: float
    character_confidences: List[float] = field(default_factory=list)
    engine: str = "FastPlateOCR CCT-S-v2 (ONNX)"
    inference_ms: float = 0.0
    is_two_row: bool = False
    lines_detected: int = 1
    region: Optional[str] = None
    region_prob: Optional[float] = None

class FastPlateRecognizer:
    """
    Isolated candidate recognizer wrapping FastPlateOCR's cct-s-v2-global-model.
    Loads the ONNX model once upon initialization.
    """
    def __init__(
        self,
        hub_ocr_model: str = "cct-s-v2-global-model",
        device: str = "cpu"
    ):
        self.model_name = hub_ocr_model
        self.device = device
        self._model = None
        self._is_ready = False
        self.init_time_ms = 0.0
        self.execution_providers = []
        self.onnx_model_path = ""
        self.plate_config_path = ""
        self.config = None
        self._init_engine()

    def _init_engine(self):
        t0 = time.perf_counter()
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            from fast_plate_ocr.inference import hub

            # Identify model paths
            try:
                m_path, c_path = hub.download_model(self.model_name)
                self.onnx_model_path = str(m_path)
                self.plate_config_path = str(c_path)
            except Exception:
                pass

            self._model = LicensePlateRecognizer(
                hub_ocr_model=self.model_name,
                device=self.device
            )
            self._is_ready = True
            self.execution_providers = getattr(self._model, "providers", ["CPUExecutionProvider"])
            self.config = getattr(self._model, "config", None)
            self.init_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            print(f"[FastPlateRecognizer] Loaded {self.model_name} in {self.init_time_ms} ms. Providers: {self.execution_providers}")
        except Exception as e:
            print(f"[FastPlateRecognizer Error] Failed to initialize FastPlateOCR: {e}")
            self._is_ready = False
            self.init_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def recognize(
        self,
        crop_bgr: np.ndarray,
        return_dict: bool = False
    ) -> Union[FastPlateOCRResult, Dict[str, Any]]:
        """
        Runs FastPlateOCR recognition on an input BGR image.
        Converts BGR -> RGB as required by FastPlateOCR PlateConfig.
        """
        if not self._is_ready or crop_bgr is None or crop_bgr.size == 0:
            empty_res = FastPlateOCRResult(
                text="",
                raw_text="",
                confidence=0.0,
                character_confidences=[],
                engine=f"FastPlateOCR ({self.model_name})",
                inference_ms=0.0
            )
            if return_dict:
                return {
                    "text": "",
                    "raw_text": "",
                    "confidence": 0.0,
                    "latencyMs": 0.0,
                    "char_probs": [],
                    "region": None,
                    "region_prob": None,
                    "engine": empty_res.engine
                }
            return empty_res

        t0 = time.perf_counter()

        # Handle color channels: OpenCV BGR -> RGB
        if len(crop_bgr.shape) == 2:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2RGB)
        elif crop_bgr.shape[2] == 4:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2RGB)
        else:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        try:
            pred = self._model.run_one(crop_rgb, return_confidence=True)
            t_ms = (time.perf_counter() - t0) * 1000.0

            raw_str = pred.plate or ""
            cleaned = clean_ocr_raw_tokens(raw_str)

            # Character probabilities
            probs = [float(p) for p in pred.char_probs] if pred.char_probs is not None else []
            # Calculate overall confidence as mean of character probs (or 0.0 if empty)
            conf = float(np.mean(probs)) if probs else (0.85 if cleaned else 0.0)

            result = FastPlateOCRResult(
                text=cleaned,
                raw_text=raw_str,
                confidence=round(conf, 4),
                character_confidences=[round(p, 4) for p in probs],
                engine=f"FastPlateOCR ({self.model_name})",
                inference_ms=round(t_ms, 2),
                is_two_row=False,
                lines_detected=1,
                region=pred.region,
                region_prob=round(float(pred.region_prob), 4) if pred.region_prob is not None else None
            )

            if return_dict:
                return {
                    "text": cleaned,
                    "raw_text": raw_str,
                    "confidence": round(conf, 4),
                    "latencyMs": round(t_ms, 2),
                    "char_probs": result.character_confidences,
                    "region": result.region,
                    "region_prob": result.region_prob,
                    "engine": result.engine
                }
            return result

        except Exception as e:
            t_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[FastPlateRecognizer] Inference error: {e}")
            err_res = FastPlateOCRResult(
                text="",
                raw_text="",
                confidence=0.0,
                character_confidences=[],
                engine=f"FastPlateOCR ({self.model_name})",
                inference_ms=round(t_ms, 2)
            )
            if return_dict:
                return {
                    "text": "",
                    "raw_text": "",
                    "confidence": 0.0,
                    "latencyMs": round(t_ms, 2),
                    "char_probs": [],
                    "region": None,
                    "region_prob": None,
                    "engine": err_res.engine,
                    "error": str(e)
                }
            return err_res
