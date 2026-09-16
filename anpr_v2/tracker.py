"""
anpr_v2/tracker.py
Multi-Frame Temporal Consensus & Best-Frame Candidate Selection (Phase 11, 12, 14).

Features:
1. Spatial Bounding-Box Track Association across frames.
2. Best-Frame Candidate Scoring (sharpness, detector confidence, plate area, aspect ratio).
3. Character-Level Confidence-Weighted Temporal Voting (visual evidence over format guessing).
4. Adaptive Confirmation Policy (Fast path for >=0.85 valid, 2 frames for medium confidence).
"""

import time
import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from anpr_v2.config import CONFIG
from anpr_v2.validator import validate_indian_registration, ValidationStatus, ValidationResult

@dataclass
class CandidateCrop:
    crop_bgr: np.ndarray
    bbox: List[int]
    detector_confidence: float
    sharpness: float
    quality_score: float
    timestamp: float

@dataclass
class SightingRecord:
    plate_text: str
    confidence: float
    character_confidences: List[float]
    detector_confidence: float
    timestamp: float
    validation: ValidationResult

@dataclass
class TrackedPlate:
    track_id: str
    camera_id: int
    bbox: List[int] # [x1, y1, x2, y2]
    created_at: float
    last_updated: float
    sightings: List[SightingRecord] = field(default_factory=list)
    best_candidate: Optional[CandidateCrop] = None
    confirmed: bool = False
    confirmed_plate: Optional[str] = None
    confirmed_validation: Optional[ValidationResult] = None
    confirmed_confidence: float = 0.0

def compute_box_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection over Union (IoU) between [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    boxA_area = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter_area / float(boxA_area + boxB_area - inter_area)

def compute_laplacian_sharpness(img_bgr: np.ndarray) -> float:
    """Computes variance of Laplacian as a proxy for edge sharpness."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def score_candidate_crop(
    crop_bgr: np.ndarray,
    det_conf: float,
    bbox: List[int]
) -> Tuple[float, float]:
    """
    Scores a plate candidate crop based on:
      1. Detector confidence (40%)
      2. Laplacian sharpness (30%)
      3. Plate area / resolution (20%)
      4. Aspect ratio sanity (10%)
    Returns: (quality_score, sharpness)
    """
    sharpness = compute_laplacian_sharpness(crop_bgr)
    h, w = crop_bgr.shape[:2]
    area = w * h
    ar = w / float(max(1, h))

    # Normalized metrics in [0.0, 1.0]
    sharp_norm = min(1.0, sharpness / 150.0)
    area_norm = min(1.0, area / 20000.0)
    
    # Ideal aspect ratio penalty (penalize extreme skews)
    if 1.1 <= ar <= 5.5:
        ar_norm = 1.0
    else:
        ar_norm = max(0.0, 1.0 - abs(ar - 3.5) / 4.0)

    quality_score = (
        0.40 * float(det_conf) +
        0.30 * sharp_norm +
        0.20 * area_norm +
        0.10 * ar_norm
    )
    return float(round(quality_score, 3)), float(round(sharpness, 1))

def character_level_temporal_voting(sightings: List[SightingRecord]) -> Tuple[str, float]:
    """
    Performs character-level confidence-weighted voting across multi-frame sightings.
    Adheres strictly to Phase 11:
      VISUAL EVIDENCE FROM MULTIPLE FRAMES > FORMAT-BASED CHARACTER CORRECTION.
    """
    if not sightings:
        return "", 0.0

    if len(sightings) == 1:
        return sightings[0].plate_text, sightings[0].confidence

    # 1. Group sightings by string length
    length_votes = defaultdict(float)
    for s in sightings:
        if s.plate_text:
            length_votes[len(s.plate_text)] += s.confidence

    if not length_votes:
        return "", 0.0

    target_length = max(length_votes.keys(), key=lambda l: length_votes[l])

    # Filter sightings matching the target length
    filtered_sightings = [s for s in sightings if len(s.plate_text) == target_length]
    if not filtered_sightings:
        filtered_sightings = sightings

    # 2. Position-wise character accumulation
    consensus_chars = []
    char_confidences = []

    for pos in range(target_length):
        char_weights = defaultdict(float)
        total_weight = 0.0

        for s in filtered_sightings:
            if pos < len(s.plate_text):
                char = s.plate_text[pos]
                # Weight by combination of OCR confidence and detector confidence
                weight = s.confidence * (0.8 + 0.2 * s.detector_confidence)
                char_weights[char] += weight
                total_weight += weight

        if char_weights:
            best_char = max(char_weights.keys(), key=lambda c: char_weights[c])
            best_weight = char_weights[best_char]
            char_confidence = best_weight / max(1e-6, total_weight)
            consensus_chars.append(best_char)
            char_confidences.append(char_confidence)
        else:
            consensus_chars.append("?")
            char_confidences.append(0.0)

    consensus_str = "".join(consensus_chars)
    overall_conf = float(np.mean(char_confidences)) if char_confidences else 0.0
    return consensus_str, round(overall_conf, 3)


class TemporalConsensusTracker:
    def __init__(self, window_sec: Optional[float] = None):
        self.window_sec = window_sec or CONFIG.temporal_window_sec
        self.tracks: Dict[str, TrackedPlate] = {}
        self._next_id = 1

    def update_tracks_and_vote(
        self,
        camera_id: int,
        candidates: List[dict], # list of dicts: {"bbox", "crop", "ocr_result", "det_conf"}
        manual_scan: bool = False
    ) -> List[dict]:
        """
        Updates multi-camera tracks with new candidate sightings,
        performs character-level temporal voting, and applies the Phase 14 confirmation policy.
        """
        now = time.time()
        self._purge_stale_tracks(now)

        evaluations = []

        for cand in candidates:
            bbox = cand["bbox"]
            crop_bgr = cand["crop"]
            ocr_res = cand["ocr_result"]
            det_conf = cand["det_conf"]

            quality_score, sharpness = score_candidate_crop(crop_bgr, det_conf, bbox)
            cand_crop_obj = CandidateCrop(
                crop_bgr=crop_bgr,
                bbox=bbox,
                detector_confidence=det_conf,
                sharpness=sharpness,
                quality_score=quality_score,
                timestamp=now
            )

            # Match to existing track for this camera
            matched_track = self._match_track(camera_id, bbox)

            if matched_track is None:
                track_id = f"trk_{camera_id}_{self._next_id}_{int(now*1000)%100000}"
                self._next_id += 1
                matched_track = TrackedPlate(
                    track_id=track_id,
                    camera_id=camera_id,
                    bbox=bbox,
                    created_at=now,
                    last_updated=now,
                    best_candidate=cand_crop_obj
                )
                self.tracks[track_id] = matched_track
            else:
                matched_track.last_updated = now
                matched_track.bbox = bbox
                if (matched_track.best_candidate is None or 
                    quality_score > matched_track.best_candidate.quality_score):
                    matched_track.best_candidate = cand_crop_obj

            # Validate current frame reading
            val_res = validate_indian_registration(ocr_res.text, confidence=ocr_res.confidence)
            
            sighting = SightingRecord(
                plate_text=ocr_res.text,
                confidence=ocr_res.confidence,
                character_confidences=ocr_res.character_confidences,
                detector_confidence=det_conf,
                timestamp=now,
                validation=val_res
            )
            matched_track.sightings.append(sighting)

            # Character-level temporal voting across track's sightings
            consensus_plate, consensus_conf = character_level_temporal_voting(matched_track.sightings)
            consensus_val = validate_indian_registration(consensus_plate, confidence=consensus_conf)

            # Phase 14 Confirmation Policy
            num_sightings = len(matched_track.sightings)
            is_confirmed = False
            status_code = "CANDIDATE_ACCUMULATING"

            if consensus_val.is_valid:
                # Manual scan or single image upload confirms immediately
                if manual_scan:
                    is_confirmed = True
                    status_code = "CONFIRMED_ANPR_EVENT"
                # Immediate confirmation for high confidence (>= 0.65)
                elif consensus_conf >= 0.65:
                    is_confirmed = True
                    status_code = "CONFIRMED_ANPR_EVENT"
                # Medium Path: 2+ sightings with valid syntax
                elif num_sightings >= CONFIG.min_confirm_sightings and consensus_conf >= CONFIG.min_ocr_plate_conf:
                    is_confirmed = True
                    status_code = "CONFIRMED_ANPR_EVENT"
                else:
                    status_code = f"CANDIDATE_ACCUMULATING ({num_sightings}/{CONFIG.min_confirm_sightings} frames)"
            elif manual_scan and ocr_res.text and ocr_res.confidence >= 0.40:
                # Fallback for manual single-image scan with readable text
                is_confirmed = True
                status_code = "CONFIRMED_ANPR_EVENT"
            else:
                status_code = f"INVALID_SYNTAX ({consensus_val.reason})"

            if is_confirmed and not matched_track.confirmed:
                matched_track.confirmed = True
                matched_track.confirmed_plate = consensus_plate or ocr_res.text
                matched_track.confirmed_validation = consensus_val
                matched_track.confirmed_confidence = consensus_conf or ocr_res.confidence

            # Compute standard [x, y, w, h] bbox for dashboard Canvas strokeRect
            x1, y1, x2, y2 = bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            active_plate_str = consensus_plate if consensus_val.is_valid else (ocr_res.text if (manual_scan and is_confirmed) else "")

            eval_record = {
                "id": f"det-{int(now * 1000) % 1000000}",
                "trackId": matched_track.track_id,
                "plate": active_plate_str,
                "rawOcr": ocr_res.raw_text,
                "confidence": round(float(consensus_conf or ocr_res.confidence), 3),
                "detectorConfidence": round(float(det_conf), 3),
                "ocrConfidence": round(float(ocr_res.confidence), 3),
                "stateCode": consensus_val.state_code,
                "stateName": consensus_val.state_name,
                "formatType": consensus_val.format_type,
                "syntaxValid": consensus_val.is_valid,
                "status": status_code,
                "isConfirmed": is_confirmed,
                "sightings": num_sightings,
                "bbox": [int(x1), int(y1), int(bw), int(bh)],
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "sharpness": sharpness,
                "qualityScore": quality_score,
                "isTwoRow": ocr_res.is_two_row,
                "engine": ocr_res.engine,
                "inferenceMs": ocr_res.inference_ms,
                "vehicleType": "Car"
            }
            evaluations.append(eval_record)

        return evaluations

    def _match_track(self, camera_id: int, bbox: List[int]) -> Optional[TrackedPlate]:
        best_track = None
        best_iou = 0.0
        for track in self.tracks.values():
            if track.camera_id == camera_id:
                iou = compute_box_iou(track.bbox, bbox)
                if iou >= CONFIG.tracker_iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track = track
        return best_track

    def _purge_stale_tracks(self, now: float):
        stale_keys = [
            k for k, trk in self.tracks.items()
            if (now - trk.last_updated) > self.window_sec
        ]
        for k in stale_keys:
            del self.tracks[k]
