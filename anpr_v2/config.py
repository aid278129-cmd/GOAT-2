"""
anpr_v2/config.py
Centralized Configuration for ANPR V2 Engine.
All runtime settings, models, device selections, and thresholds are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Base Project Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")
DEBUG_CROPS_DIR = os.path.join(DEBUG_DIR, "crops")
DEBUG_FRAMES_DIR = os.path.join(DEBUG_DIR, "frames")

os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)

@dataclass
class ANPRConfig:
    # ── Pipeline Version & Active Backends ──
    anpr_version: str = "v2"
    detector_backend: str = "rfdetr"       # "rfdetr" (primary) or "yolo" (baseline)
    ocr_backend: str = os.environ.get("OCR_ENGINE", "fastplate_indian")  # "fastplate_indian", "fastplate_fallback", "paddleocr", "crnn"
    device: str = "auto"                   # "auto" (cuda if available else cpu), "cuda", "cpu"
    
    # ── Model Checkpoint Paths ──
    rfdetr_model_path: str = os.path.join(MODELS_DIR, "rfdetr_plate_best.pt")
    rfdetr_onnx_path: str = os.path.join(MODELS_DIR, "rfdetr_plate_best.onnx")
    yolo_plate_pt_path: str = os.path.join(MODELS_DIR, "indian_plate_best.pt")
    yolo_plate_onnx_path: str = os.path.join(MODELS_DIR, "indian_plate_best.onnx")
    
    ppocr_rec_onnx_path: str = os.path.join(MODELS_DIR, "ppocr_rec_v4.onnx")
    ppocr_keys_path: str = os.path.join(MODELS_DIR, "ppocr_keys_v1.txt")
    crnn_plate_onnx_path: str = os.path.join(MODELS_DIR, "crnn_plate_best.onnx")
    
    # ── Full-Frame License Plate Detector Thresholds ──
    plate_conf_threshold: float = 0.35      # Confidence threshold for direct plate detection
    live_plate_conf_threshold: float = 0.28 # Relaxed threshold for live video feeds
    detector_iou_threshold: float = 0.45    # NMS IOU threshold
    
    # ── Geometric Plate Crop Filtering ──
    min_aspect_ratio: float = 1.05         # Allows MoRTH Rule 50 square / 2-row plates (~1.1-2.0)
    max_aspect_ratio: float = 6.50         # Allows standard single-row rectangular plates (~3.5-5.5)
    min_plate_width: int = 24              # Minimum width in pixels
    min_plate_height: int = 10             # Minimum height in pixels
    max_plate_area_ratio: float = 0.65     # Max % of entire image plate can occupy
    
    # ── Perspective Rectification & Crop Preprocessing ──
    enable_rectification: bool = True      # 4-point homography warping for angled plates
    max_skew_angle_deg: float = 45.0       # Max skew angle to attempt rectification
    ocr_target_height: int = 48            # Standard normalized height for single-row OCR
    ocr_target_height_two_row: int = 80    # Standard normalized height for two-row OCR
    
    # ── OCR Recognition & Confidence Thresholds ──
    min_ocr_char_conf: float = 0.35        # Minimum single-character confidence
    min_ocr_plate_conf: float = 0.50       # Minimum overall plate OCR confidence
    min_plate_chars: int = 6               # Minimum valid characters
    max_plate_chars: int = 11              # Maximum valid characters
    
    # ── Anti-Hallucination & Validation Policy ──
    # CRITICAL: Pure validation without text morphing or aggressive rewriting!
    strict_indian_state_required: bool = True
    allow_aggressive_ocr_correction: bool = False  # Strictly FALSE: No converting 8->B or fabricating plates
    
    # ── Temporal OCR Consensus Tracker ──
    temporal_consensus_enabled: bool = True
    temporal_window_sec: float = 6.0       # Active window for multi-frame voting
    min_confirm_sightings: int = 2         # Medium confidence requires 2 frames
    high_conf_fast_confirm: float = 0.85   # Immediate confirmation if valid format + conf >= 0.85
    tracker_iou_threshold: float = 0.30    # Spatial box tracking overlap
    
    # ── Best Frame Selection ──
    best_frame_selection_enabled: bool = True
    min_sharpness_score: float = 50.0      # Laplacian variance threshold
    
    # ── Diagnostic Archival & Endpoints ──
    save_debug_crops: bool = True          # Save plate crops to debug_output/crops/
    save_debug_frames: bool = False        # Save full frames
    node_server_url: str = "https://127.0.0.1:3000/api/detections"
    anpr_port: int = 5001

# Global singleton configuration instance
CONFIG = ANPRConfig()

def get_resolved_device(device_setting: Optional[str] = None) -> str:
    """Resolves 'auto', 'cuda', or 'cpu' to an actual torch / onnx runtime device."""
    target = (device_setting or CONFIG.device).lower()
    if target == "cuda":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    if target == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"
