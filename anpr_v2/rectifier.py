"""
anpr_v2/rectifier.py
Minimal & Non-Destructive Plate Crop Processing per Phase 6.

1. Crop extraction with slight padding.
2. Optional 4-point perspective rectification for angled plates.
3. Resolution normalization (preserving aspect ratio and natural stroke topology).
4. Debug crop saving to debug_output/.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, List
from anpr_v2.config import CONFIG, DEBUG_DIR, DEBUG_CROPS_DIR, DEBUG_FRAMES_DIR

def extract_plate_crop(
    image: np.ndarray,
    bbox: List[float],
    padding: float = 0.10
) -> Tuple[np.ndarray, List[int]]:
    """
    Extracts license plate crop from image using bounding box [x1, y1, x2, y2]
    with configurable spatial padding.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    bw = x2 - x1
    bh = y2 - y1
    
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w_img, int(x2 + pad_x))
    cy2 = min(h_img, int(y2 + pad_y))
    
    crop = image[cy1:cy2, cx1:cx2].copy()
    clamped_box = [cx1, cy1, cx2, cy2]
    return crop, clamped_box

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # bottom-left has largest difference
    return rect

def rectify_plate_perspective(crop_bgr: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """
    Estimates 4-point perspective homography on plate crop.
    Returns: (rectified_crop, was_rectified, estimated_skew_degrees)
    Keeps processing minimal and non-destructive.
    """
    if not CONFIG.enable_rectification or crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr, False, 0.0

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 24:
        return crop_bgr, False, 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_bgr, False, 0.0

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    best_quad = None
    max_quad_area = 0.0

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # Area must cover at least 30% of crop to be the outer plate boundary
            if area > (0.30 * w * h) and area > max_quad_area:
                max_quad_area = area
                best_quad = approx.reshape(4, 2)

    if best_quad is None:
        return crop_bgr, False, 0.0

    pts = order_quad_points(best_quad.astype("float32"))
    (tl, tr, br, bl) = pts

    # Compute width and height of transformed quad
    w_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    w_bot = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_w = max(int(w_top), int(w_bot))

    h_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(h_right), int(h_left))

    if max_w < 20 or max_h < 8:
        return crop_bgr, False, 0.0

    # Skew angle estimation (angle between top edge and horizontal)
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    skew_angle = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0.0

    # Only rectify if skew is meaningful (> 2 degrees) and within bounds (< max_skew_angle_deg)
    if abs(skew_angle) < 2.0 or abs(skew_angle) > CONFIG.max_skew_angle_deg:
        return crop_bgr, False, float(skew_angle)

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    rectified = cv2.warpPerspective(crop_bgr, M, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rectified, True, float(skew_angle)

def normalize_plate_resolution(
    crop_bgr: np.ndarray,
    target_height: Optional[int] = None
) -> np.ndarray:
    """
    Normalizes crop height while strictly preserving aspect ratio.
    Target height is typically 48px for PP-OCR single-row and 80px for two-row plates.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr

    ar = w / float(h)
    if target_height is None:
        target_height = CONFIG.ocr_target_height_two_row if ar < 2.3 else CONFIG.ocr_target_height

    # Calculate proportional width
    target_width = max(32, int(round(target_height * ar)))
    # Avoid extreme resizing if already very close
    if abs(h - target_height) <= 2:
        return crop_bgr

    interpolation = cv2.INTER_AREA if target_height < h else cv2.INTER_CUBIC
    resized = cv2.resize(crop_bgr, (target_width, target_height), interpolation=interpolation)
    return resized

def save_debug_crops_artifact(
    raw_frame: Optional[np.ndarray],
    detected_crop: Optional[np.ndarray],
    rectified_crop: Optional[np.ndarray],
    ocr_input: Optional[np.ndarray],
    prefix: str = "frame"
) -> dict:
    """
    Saves debug crops according to Phase 6 specifications:
      debug_output/
        raw_frame.jpg
        detected_crop.jpg
        rectified_crop.jpg
        ocr_input.jpg
    """
    saved_paths = {}
    if not CONFIG.save_debug_crops:
        return saved_paths

    try:
        if raw_frame is not None and CONFIG.save_debug_frames:
            p = os.path.join(DEBUG_DIR, f"{prefix}_raw_frame.jpg")
            cv2.imwrite(p, raw_frame)
            saved_paths["raw_frame"] = p

        if detected_crop is not None:
            p = os.path.join(DEBUG_DIR, f"{prefix}_detected_crop.jpg")
            cv2.imwrite(p, detected_crop)
            saved_paths["detected_crop"] = p

        if rectified_crop is not None:
            p = os.path.join(DEBUG_DIR, f"{prefix}_rectified_crop.jpg")
            cv2.imwrite(p, rectified_crop)
            saved_paths["rectified_crop"] = p

        if ocr_input is not None:
            p = os.path.join(DEBUG_DIR, f"{prefix}_ocr_input.jpg")
            cv2.imwrite(p, ocr_input)
            saved_paths["ocr_input"] = p
    except Exception as e:
        print(f"[Warning] Failed to save debug crops: {e}")

    return saved_paths
