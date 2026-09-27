"""
anpr_v2/two_line_handler.py
Explicit Two-Line License Plate Detection & Processing per Phase 6.

Indian license plates conform to MoRTH Rule 50:
- Single-row plates (cars, commercial vehicles): Aspect Ratio ~3.5 to 5.5
- Two-row plates (motorcycles, auto-rickshaws, SUVs/rear bumpers): Aspect Ratio ~1.2 to 2.2

When a two-row plate is naively resized to 128x64, characters vertically overlap,
causing severe character omission and confusion.

This module provides:
1. is_two_line_plate(): Robust heuristic combining Aspect Ratio and Horizontal Projection.
2. normalize_two_line_to_single_strip(): Slices top & bottom rows and stitches them horizontally.
3. recognize_two_line_modular(): Runs OCR on top and bottom rows independently and concatenates.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Callable

def is_two_line_plate(crop_bgr: np.ndarray, ar_threshold: float = 2.25) -> bool:
    """
    Detects whether a license plate crop has a two-line layout.
    Uses Aspect Ratio (width / height) and horizontal projection analysis.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
        
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return False
        
    aspect_ratio = w / float(h)
    
    # Decisive single-line: wide rectangular strip
    if aspect_ratio >= 2.5:
        return False
        
    # Decisive two-line: square or near-square
    if aspect_ratio <= 1.8:
        return True
        
    # Ambiguous zone (1.8 <= AR < 2.5): verify with horizontal projection profile
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # Gradient magnitude along Y
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel = np.abs(sobel_y)
        
        # Horizontal projection (sum along width)
        proj = np.sum(abs_sobel[:, int(w * 0.15):int(w * 0.85)], axis=1)
        if len(proj) < 10:
            return aspect_ratio < ar_threshold
            
        # Check for two distinct energy peaks separated by a dip in the middle 30%-70%
        mid_start, mid_end = int(h * 0.35), int(h * 0.65)
        top_peak = np.max(proj[:mid_start]) if mid_start > 0 else 0
        bot_peak = np.max(proj[mid_end:]) if mid_end < h else 0
        mid_valley = np.min(proj[mid_start:mid_end]) if mid_start < mid_end else 1
        
        # If both top and bottom have strong text energy and the middle dips
        if top_peak > 0 and bot_peak > 0 and mid_valley < (0.85 * min(top_peak, bot_peak)):
            return True
    except Exception:
        pass
        
    return aspect_ratio < ar_threshold

def normalize_two_line_to_single_strip(
    crop_bgr: np.ndarray,
    target_height: int = 64,
    overlap_margin: float = 0.08
) -> np.ndarray:
    """
    Transforms a two-line plate crop into a normalized single-line wide strip.
    Splits into Top Row (0% to ~55%) and Bottom Row (~45% to 100%) and stitches them horizontally.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
        
    h, w = crop_bgr.shape[:2]
    if h < 16 or w < 20:
        return crop_bgr
        
    split_top = int(h * (0.50 + overlap_margin))
    split_bot = int(h * (0.50 - overlap_margin))
    
    top_slice = crop_bgr[0:split_top, :].copy()
    bot_slice = crop_bgr[split_bot:h, :].copy()
    
    # Scale each slice to target_height preserving aspect ratio
    h_top, w_top = top_slice.shape[:2]
    h_bot, w_bot = bot_slice.shape[:2]
    
    new_w_top = max(16, int(w_top * (target_height / float(max(1, h_top)))))
    new_w_bot = max(16, int(w_bot * (target_height / float(max(1, h_bot)))))
    
    top_resized = cv2.resize(top_slice, (new_w_top, target_height), interpolation=cv2.INTER_LINEAR)
    bot_resized = cv2.resize(bot_slice, (new_w_bot, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Optional small spacer between rows
    spacer_w = max(4, int(target_height * 0.10))
    # Fill spacer with median background color from edges
    bg_color = [int(x) for x in np.median(top_slice[-2:, :], axis=(0, 1))]
    spacer = np.full((target_height, spacer_w, 3), bg_color, dtype=np.uint8)
    
    # Horizontally stitch: [Top Row | Spacer | Bottom Row]
    stitched = np.hstack([top_resized, spacer, bot_resized])
    return stitched

def recognize_two_line_modular(
    ocr_func: Callable[[np.ndarray], Tuple[str, float]],
    crop_bgr: np.ndarray,
    overlap_margin: float = 0.08
) -> Tuple[str, float]:
    """
    Processes a two-line plate by running OCR on the top row and bottom row separately,
    then concatenating the results (e.g., 'TN45' + 'AB1234' -> 'TN45AB1234').
    """
    h, w = crop_bgr.shape[:2]
    split_top = int(h * (0.50 + overlap_margin))
    split_bot = int(h * (0.50 - overlap_margin))
    
    top_slice = crop_bgr[0:split_top, :]
    bot_slice = crop_bgr[split_bot:h, :]
    
    text_top, conf_top = ocr_func(top_slice)
    text_bot, conf_bot = ocr_func(bot_slice)
    
    combined_text = (text_top + text_bot).replace(" ", "").upper()
    combined_conf = round(float((conf_top + conf_bot) / 2.0), 3)
    return combined_text, combined_conf
