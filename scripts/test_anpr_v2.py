#!/usr/bin/env python3
"""
scripts/test_anpr_v2.py
Automated Comprehensive Test Suite for ANPR V2 Engine (Phase 21).

Tests covered:
1.  detector loading (YOLO and RF-DETR adapter)
2.  OCR loading (PP-OCRv4 ONNX)
3.  plate crop extraction & clamping
4.  rectification (4-point homography & skew angle)
5.  validator (standard MoRTH, Bharat series, historical, invalid state, out-of-bounds)
6.  temporal consensus (confidence-weighted voting across multi-frame sequence)
7.  pipeline empty frame handling
8.  pipeline no-plate frame handling
9.  pipeline multiple plates handling
10. two-row plate reading & decomposition
11. yellow commercial plate handling
12. motorcycle plate crop handling
13. invalid text rejection
14. low-confidence OCR handling
15. API integration & payload schema validation
"""

import os
import sys
import unittest
import numpy as np
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from anpr_v2.config import CONFIG
from anpr_v2.detector import get_detector, YOLOPlateDetector, RFDETRPlateDetector
from anpr_v2.recognizer import get_recognizer, PPOCRRecognizer
from anpr_v2.rectifier import extract_plate_crop, rectify_plate_perspective, normalize_plate_resolution
from anpr_v2.validator import validate_indian_registration, ValidationStatus, clean_ocr_raw_tokens
from anpr_v2.tracker import TemporalConsensusTracker, SightingRecord, character_level_temporal_voting
from anpr_v2.pipeline import get_pipeline, ANPRPipelineV2

class TestANPRV2(unittest.TestCase):

    def setUp(self):
        self.pipeline = get_pipeline()

    # 1. Detector Loading
    def test_01_detector_loading(self):
        yolo_det = YOLOPlateDetector()
        self.assertTrue(yolo_det.is_ready, "YOLO Baseline Detector failed to load.")
        rf_det = RFDETRPlateDetector()
        self.assertTrue(rf_det.is_ready, "RF-DETR Detector failed to load.")

    # 2. OCR Loading
    def test_02_ocr_loading(self):
        rec = get_recognizer("paddleocr")
        self.assertTrue(rec.is_ready, "PP-OCRv4 Recognizer failed to load.")

    # 3. Plate Crop Extraction
    def test_03_plate_crop_extraction(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = [100, 150, 300, 200]
        crop, clamped = extract_plate_crop(dummy_img, bbox, padding=0.05)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)
        self.assertLessEqual(clamped[2], 640)
        self.assertLessEqual(clamped[3], 480)

    # 4. Rectification
    def test_04_rectification(self):
        # Create a sample synthetic plate image
        plate = np.full((60, 200, 3), 240, dtype=np.uint8)
        cv2.putText(plate, "MH01AV8669", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
        rect, was_rect, skew = rectify_plate_perspective(plate)
        self.assertIsNotNone(rect)
        self.assertGreater(rect.shape[0], 10)
        norm = normalize_plate_resolution(rect, target_height=48)
        self.assertEqual(norm.shape[0], 48)

    # 5. Validator
    def test_05_validator_rules(self):
        # Valid standard MoRTH
        r1 = validate_indian_registration("MH01AV8669")
        self.assertEqual(r1.status, ValidationStatus.VALID)
        self.assertEqual(r1.state_code, "MH")
        self.assertEqual(r1.cleaned_plate, "MH01AV8669")

        # Valid Bharat Series
        r2 = validate_indian_registration("22BH1234AA")
        self.assertEqual(r2.status, ValidationStatus.VALID)
        self.assertEqual(r2.state_code, "BH")

        # Valid Historical / Commercial
        r3 = validate_indian_registration("KL498262")
        self.assertEqual(r3.status, ValidationStatus.VALID)
        self.assertEqual(r3.state_code, "KL")

        # Invalid State Code (never mutate!)
        r4 = validate_indian_registration("ZZ99ZZ9999")
        self.assertNotEqual(r4.status, ValidationStatus.VALID)

        # Length out of bounds
        r5 = validate_indian_registration("ABC")
        self.assertEqual(r5.status, ValidationStatus.INVALID)

    # 6. Temporal Consensus
    def test_06_temporal_consensus_voting(self):
        v = validate_indian_registration("MH01AV8669")
        sightings = [
            SightingRecord("MH01AV8669", 0.85, [0.85]*10, 0.9, 100.0, v),
            SightingRecord("MH01AV8668", 0.50, [0.50]*10, 0.9, 101.0, v),
            SightingRecord("MH01AV8669", 0.90, [0.90]*10, 0.9, 102.0, v),
        ]
        cons_str, conf = character_level_temporal_voting(sightings)
        self.assertEqual(cons_str, "MH01AV8669", "Temporal voting failed to resolve majority character.")
        self.assertGreater(conf, 0.80)

    # 7. Empty Frame
    def test_07_empty_frame(self):
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        res = self.pipeline.process_frame(empty_img)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "REJECTED_EMPTY_FRAME")

    # 8. No Plate
    def test_08_no_plate(self):
        # Pure blue gradient image with zero plates
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        blank[:, :] = (150, 80, 20)
        res = self.pipeline.process_frame(blank)
        self.assertTrue(res["success"])
        self.assertFalse(res["detected"])
        self.assertEqual(res["status"], "NO_PLATE_DETECTED")

    # 9. Multiple Plates
    def test_09_multiple_plates(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw 2 distinct plates
        cv2.rectangle(frame, (100, 200), (320, 270), (250, 250, 250), -1)
        cv2.putText(frame, "DL3CD1210", (110, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
        cv2.rectangle(frame, (700, 300), (920, 370), (250, 250, 250), -1)
        cv2.putText(frame, "TN58D5353", (710, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
        res = self.pipeline.process_frame(frame)
        self.assertTrue(res["success"])
        # Pipeline correctly extracts candidate crops without crashing
        self.assertIn("timing", res)

    # 10. Two-Row Plate Handling
    def test_10_two_row_plate(self):
        square_plate = np.full((120, 160, 3), 245, dtype=np.uint8)
        cv2.putText(square_plate, "KL34", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 3)
        cv2.putText(square_plate, "A465", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 3)
        rec = get_recognizer()
        ocr_res = rec.recognize(square_plate)
        self.assertIsNotNone(ocr_res.text)

    # 11. Yellow Plate Handling
    def test_11_yellow_plate(self):
        yellow_plate = np.zeros((50, 200, 3), dtype=np.uint8)
        yellow_plate[:, :] = (30, 220, 240) # BGR Yellow
        cv2.putText(yellow_plate, "UP84AE9889", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        rec = get_recognizer()
        res = rec.recognize(yellow_plate)
        self.assertIn("UP84", res.text)

    # 12. Motorcycle Plate Handling
    def test_12_motorcycle_plate(self):
        # Motorcycle plates are often narrow or 2-row
        moto_crop = np.full((80, 130, 3), 240, dtype=np.uint8)
        cv2.putText(moto_crop, "MH12", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(moto_crop, "5678", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        rec = get_recognizer()
        res = rec.recognize(moto_crop)
        self.assertIsNotNone(res)

    # 13. Invalid Text Rejection
    def test_13_invalid_text_rejection(self):
        gibberish = "X8!@#9"
        val = validate_indian_registration(gibberish)
        self.assertFalse(val.is_valid)

    # 14. Low Confidence OCR
    def test_14_low_confidence_ocr(self):
        val = validate_indian_registration("MH01AV8669", confidence=0.30)
        self.assertTrue(val.is_valid) # syntax is valid, but confidence tracker handles thresholding

    # 15. API Integration Payload
    def test_15_api_payload_schema(self):
        sample_frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        res = self.pipeline.process_frame(sample_frame, camera_id=2, camera_name="Test CAM")
        self.assertIn("success", res)
        self.assertIn("detected", res)
        self.assertIn("status", res)
        self.assertIn("timing", res)
        self.assertIn("debug", res)
        self.assertIn("plateDetectionMs", res["timing"])
        self.assertIn("ocrTotalMs", res["timing"])

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestANPRV2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
