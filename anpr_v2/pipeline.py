"""
anpr_v2/pipeline.py
Unified ANPR V2 Vision & Intelligence Pipeline.

Connects:
  Frame -> Plate Detector -> Best Crop -> Perspective Rectification ->
  PP-OCR -> Multi-Frame Temporal Voting -> Pure Indian Validator -> Confirmed Event.

Maintains 100% backward compatibility with Node.js server.js and Dashboard Telemetry HUD.
"""

import time
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

from anpr_v2.config import CONFIG, DEBUG_CROPS_DIR, DEBUG_FRAMES_DIR
from anpr_v2.detector import get_detector, DetectionBox
from anpr_v2.rectifier import extract_plate_crop, rectify_plate_perspective, normalize_plate_resolution, save_debug_crops_artifact
from anpr_v2.recognizer import get_recognizer, OCRResult
from anpr_v2.tracker import TemporalConsensusTracker, compute_laplacian_sharpness
from anpr_v2.validator import validate_indian_registration, ValidationStatus

class ANPRPipelineV2:
    def __init__(self):
        self.detector = get_detector()
        self.recognizer = get_recognizer()
        self.tracker = TemporalConsensusTracker()
        print("[ANPR V2 Pipeline] Initialized successfully.")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: int = 1,
        camera_name: str = "Camera 1",
        manual_scan: bool = False,
        extra_telemetry: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Executes full ANPR V2 vision pipeline on a single frame.
        Returns complete diagnostic payload compatible with server.js & frontend HUD.
        """
        t_start = time.perf_counter()
        now_ts = time.time()
        frame_id = f"v2_cam{camera_id}_{int(now_ts * 1000)}"

        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "success": False,
                "detected": False,
                "status": "REJECTED_EMPTY_FRAME",
                "error": "Empty frame input"
            }

        h_img, w_img = frame_bgr.shape[:2]
        laplacian_var = compute_laplacian_sharpness(frame_bgr)
        is_blurry = laplacian_var < CONFIG.min_sharpness_score

        # ── Step 1: Direct Full-Frame License Plate Detection ──
        t0_det = time.perf_counter()
        conf_thresh = CONFIG.plate_conf_threshold if manual_scan else CONFIG.live_plate_conf_threshold
        plate_boxes: List[DetectionBox] = self.detector.detect(frame_bgr, conf_threshold=conf_thresh)
        t_det_ms = (time.perf_counter() - t0_det) * 1000.0

        # Direct Plate Crop Fallback: if detector finds no bounding boxes (e.g. uploaded image is already a crop)
        if not plate_boxes:
            crop_rect_fb, was_rect_fb, skew_deg_fb = rectify_plate_perspective(frame_bgr)
            crop_norm_fb = normalize_plate_resolution(crop_rect_fb)
            t0_ocr_fb = time.perf_counter()
            ocr_res_fb = self.recognizer.recognize(crop_norm_fb)
            t_ocr_fb = (time.perf_counter() - t0_ocr_fb) * 1000.0
            val_res_fb = validate_indian_registration(ocr_res_fb.text, confidence=ocr_res_fb.confidence)

            if val_res_fb.is_valid or (ocr_res_fb.text and ocr_res_fb.confidence >= 0.35 and len(ocr_res_fb.text) >= 5):
                ar_fb = round(w_img / float(max(1, h_img)), 2)
                plate_boxes = [DetectionBox(
                    bbox=[0, 0, w_img, h_img],
                    confidence=0.95 if manual_scan else 0.80,
                    class_name="license_plate",
                    aspect_ratio=ar_fb,
                    area=w_img * h_img
                )]

        if not plate_boxes:
            t_total_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "success": True,
                "detected": False,
                "status": "NO_PLATE_DETECTED",
                "detection": None,
                "vehicles": [],
                "detections": [],
                "timing": {
                    "plateDetectionMs": round(t_det_ms, 1),
                    "ocrTotalMs": 0.0,
                    "totalProcessingMs": round(t_total_ms, 1),
                    "detector": CONFIG.detector_backend,
                    "ocr": CONFIG.ocr_backend
                },
                "timeline": {
                    "pyStartMs": round(t_start * 1000.0, 1),
                    "totalProcessingMs": round(t_total_ms, 1)
                },
                "debug": {
                    "frameId": frame_id,
                    "sharpness": round(laplacian_var, 1),
                    "platesFound": 0,
                    "summary": {"status": "NO_PLATE_DETECTED"}
                }
            }

        # ── Step 2 & 3: Crop Extraction, Rectification & OCR Recognition ──
        t_persp_total = 0.0
        t_ocr_total = 0.0
        candidate_sightings = []
        raw_candidates_telemetry = []

        for idx, pbox in enumerate(plate_boxes):
            # Extract crop
            t0_p = time.perf_counter()
            # If bounding box is the full frame (from crop fallback), use frame directly
            if pbox.bbox == [0, 0, w_img, h_img]:
                crop_raw = frame_bgr.copy()
                clamped_box = [0, 0, w_img, h_img]
            else:
                crop_raw, clamped_box = extract_plate_crop(frame_bgr, pbox.bbox)
            if crop_raw.size == 0:
                continue

            # Rectification
            crop_rect, was_rect, skew_deg = rectify_plate_perspective(crop_raw)
            crop_norm = normalize_plate_resolution(crop_rect)
            t_persp_total += (time.perf_counter() - t0_p) * 1000.0

            # PP-OCR Recognition
            t0_ocr = time.perf_counter()
            ocr_res: OCRResult = self.recognizer.recognize(crop_norm)
            t_ocr_total += (time.perf_counter() - t0_ocr) * 1000.0

            # Save debug crops if configured
            crop_paths = {}
            if CONFIG.save_debug_crops and idx == 0:
                crop_paths = save_debug_crops_artifact(
                    raw_frame=frame_bgr if CONFIG.save_debug_frames else None,
                    detected_crop=crop_raw,
                    rectified_crop=crop_rect if was_rect else None,
                    ocr_input=crop_norm,
                    prefix=f"{frame_id}_cand{idx}"
                )

            candidate_sightings.append({
                "bbox": clamped_box,
                "crop": crop_norm,
                "ocr_result": ocr_res,
                "det_conf": pbox.confidence,
                "was_rectified": was_rect,
                "skew_deg": skew_deg,
                "crop_paths": crop_paths
            })

            raw_candidates_telemetry.append({
                "candidateIndex": idx,
                "bbox": clamped_box,
                "detectorConfidence": pbox.confidence,
                "aspectRatio": pbox.aspect_ratio,
                "rawOcr": ocr_res.raw_text,
                "cleanedOcr": ocr_res.text,
                "ocrConfidence": ocr_res.confidence,
                "ocrEngine": ocr_res.engine,
                "rectified": was_rect,
                "skewDeg": skew_deg
            })

        # ── Step 4 & 5: Temporal Voting & Validation ──
        t0_trk = time.perf_counter()
        evaluations = self.tracker.update_tracks_and_vote(
            camera_id=camera_id,
            candidates=candidate_sightings,
            manual_scan=manual_scan
        )
        t_trk_ms = (time.perf_counter() - t0_trk) * 1000.0

        # Select primary confirmed event
        confirmed_detection = None
        for ev in evaluations:
            if ev.get("isConfirmed") and ev.get("plate"):
                confirmed_detection = ev
                break

        # If not confirmed yet, promote strongest valid candidate
        if confirmed_detection is None and evaluations:
            valid_cands = [e for e in evaluations if e.get("syntaxValid") and e.get("plate")]
            if valid_cands:
                confirmed_detection = max(valid_cands, key=lambda e: e.get("confidence", 0.0))
            elif manual_scan:
                confirmed_detection = max(evaluations, key=lambda e: e.get("confidence", 0.0))
                if confirmed_detection and not confirmed_detection.get("plate") and confirmed_detection.get("rawOcr"):
                    confirmed_detection["plate"] = confirmed_detection["rawOcr"]

        primary_eval = confirmed_detection if confirmed_detection else (evaluations[0] if evaluations else {})
        is_detected = confirmed_detection is not None and bool(confirmed_detection.get("plate"))

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        timing_payload = {
            "plateDetectionMs": round(t_det_ms, 1),
            "perspectiveMs": round(t_persp_total, 1),
            "ocrTotalMs": round(t_ocr_total, 1),
            "temporalTrackerMs": round(t_trk_ms, 2),
            "totalProcessingMs": round(t_total_ms, 1),
            "detectorBackend": CONFIG.detector_backend,
            "ocrBackend": CONFIG.ocr_backend
        }

        timeline_payload = {
            "pyStartMs": round(t_start * 1000.0, 1),
            "plateDetectionMs": round(t_det_ms, 1),
            "ocrTotalMs": round(t_ocr_total, 1),
            "totalProcessingMs": round(t_total_ms, 1)
        }
        if extra_telemetry:
            timeline_payload.update(extra_telemetry)

        frame_telemetry = {
            "frameId": frame_id,
            "cameraId": camera_id,
            "cameraName": camera_name,
            "sharpness": round(laplacian_var, 1),
            "isBlurry": bool(is_blurry),
            "candidatesFound": len(plate_boxes),
            "candidates": raw_candidates_telemetry,
            "evaluations": evaluations,
            "summary": {
                "status": "CONFIRMED_ANPR_EVENT" if is_detected else primary_eval.get("status", "TRACKING"),
                "confirmedPlate": confirmed_detection["plate"] if confirmed_detection else None,
                "detector": CONFIG.detector_backend,
                "ocr": CONFIG.ocr_backend
            }
        }

        return {
            "success": True,
            "detected": is_detected,
            "status": "CONFIRMED_ANPR_EVENT" if is_detected else primary_eval.get("status", "CANDIDATE_TRACKING"),
            "detection": confirmed_detection,
            "vehicles": [],
            "detections": evaluations,
            "timing": timing_payload,
            "timeline": timeline_payload,
            "debug": frame_telemetry
        }

_PIPELINE_INSTANCE = None

def get_pipeline() -> ANPRPipelineV2:
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = ANPRPipelineV2()
    return _PIPELINE_INSTANCE
