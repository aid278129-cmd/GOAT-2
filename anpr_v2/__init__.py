"""
ANPR V2 Package
Clean, modular, and empirically validated Automatic Number Plate Recognition engine for Indian plates.
"""

from anpr_v2.config import CONFIG
from anpr_v2.validator import validate_indian_registration, ValidationStatus, ValidationResult
from anpr_v2.rectifier import extract_plate_crop, rectify_plate_perspective, normalize_plate_resolution
from anpr_v2.recognizer import get_recognizer, OCRResult
from anpr_v2.tracker import TemporalConsensusTracker, character_level_temporal_voting
from anpr_v2.detector import get_detector, DetectionBox
from anpr_v2.pipeline import get_pipeline, ANPRPipelineV2

__all__ = [
    "CONFIG",
    "validate_indian_registration",
    "ValidationStatus",
    "ValidationResult",
    "extract_plate_crop",
    "rectify_plate_perspective",
    "normalize_plate_resolution",
    "get_recognizer",
    "OCRResult",
    "TemporalConsensusTracker",
    "character_level_temporal_voting",
    "get_detector",
    "DetectionBox",
    "get_pipeline",
    "ANPRPipelineV2"
]
