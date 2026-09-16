"""
anpr_v2/validator.py
Pure Indian Registration Format Validator.
Supports standard MoRTH Rule 50 formats, Two-Row plates, and Bharat Series (BH).

CRITICAL REQUIREMENT (Phase 10 & Phase 13):
- Strictly acts as a VALIDATOR, NEVER an OCR generator.
- Returns status: "VALID", "INVALID", or "UNCERTAIN".
- Does NOT mutate, rewrite, or morph characters (NO converting 8->B or fabricating plates).
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNCERTAIN = "UNCERTAIN"

# Recognized 36 Indian State and Union Territory codes
INDIAN_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK',
    'UA', 'UP', 'WB', 'BH'
}

INDIAN_STATE_NAMES = {
    'AN': 'Andaman and Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
    'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh',
    'DD': 'Daman and Diu', 'DL': 'Delhi', 'DN': 'Dadra and Nagar Haveli',
    'GA': 'Goa', 'GJ': 'Gujarat', 'HP': 'Himachal Pradesh', 'HR': 'Haryana',
    'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala',
    'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya',
    'MN': 'Manipur', 'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OD': 'Odisha', 'OR': 'Odisha', 'PB': 'Punjab', 'PY': 'Puducherry',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura',
    'TS': 'Telangana', 'UK': 'Uttarakhand', 'UA': 'Uttarakhand', 'UP': 'Uttar Pradesh',
    'WB': 'West Bengal', 'BH': 'Bharat Series (All-India)'
}

# Standard regex patterns according to MoRTH Rule 50
# 1. Standard MoRTH: State (2 letters) + RTO (1-2 digits) + Series (0-3 letters) + Number (4 digits)
#    E.g. MH01AV8669, DL3CD1210, TN45AB1234, KA03MN8821
PATTERN_STANDARD_MORTH = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$")

# 2. Bharat Series (BH): Year (2 digits) + BH + Number (4 digits) + Letters (1-2 letters)
#    E.g. 22BH1234AA
PATTERN_BHARAT_SERIES = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

# 3. Vintage / Older Indian format: State (2 letters) + Digits (1-4) + Digits (1-4)
#    E.g. KL498262, KL34A465
PATTERN_HISTORICAL = re.compile(r"^[A-Z]{2}\d{1,4}[A-Z]{0,2}\d{1,4}$")

# 4. Diplomatic & Military Formats:
#    E.g. 23D12345 (Diplomatic), ↑ 02D 12345 (Defence)
PATTERN_DIPLOMATIC = re.compile(r"^\d{2}[CD]\d{4,5}$")

@dataclass
class ValidationResult:
    status: ValidationStatus
    cleaned_plate: str
    state_code: str
    state_name: str
    format_type: str
    reason: str
    is_valid: bool = False

def clean_ocr_raw_tokens(raw_text: str) -> str:
    """
    Removes whitespace, hyphens, dots, and non-alphanumeric noise from OCR raw output.
    Does NOT rewrite letters or digits.
    """
    if not raw_text:
        return ""
    # Strip any leading HSRP 'IND' emblem if OCR detected the blue strip text
    text = re.sub(r"[^A-Za-z0-9]", "", str(raw_text)).upper()
    if text.startswith("IND") and len(text) >= 8:
        # Check if text without IND has a valid state prefix
        remainder = text[3:]
        if remainder[:2] in INDIAN_STATES:
            text = remainder
    return text

def validate_indian_registration(ocr_text: str, confidence: float = 1.0) -> ValidationResult:
    """
    Validates a raw or normalized OCR string against Indian registration standards.
    Strictly adheres to Phase 10: does NOT modify or substitute characters.
    """
    cleaned = clean_ocr_raw_tokens(ocr_text)
    
    if not cleaned:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            cleaned_plate="",
            state_code="",
            state_name="",
            format_type="EMPTY",
            reason="Empty plate text",
            is_valid=False
        )
        
    if len(cleaned) < 5 or len(cleaned) > 11:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            cleaned_plate=cleaned,
            state_code=cleaned[:2] if len(cleaned) >= 2 else "",
            state_name="",
            format_type="LENGTH_OUT_OF_BOUNDS",
            reason=f"Length {len(cleaned)} out of bounds (expected 6-11 characters)",
            is_valid=False
        )
        
    # Check Bharat Series: YY BH #### XX
    if PATTERN_BHARAT_SERIES.match(cleaned):
        return ValidationResult(
            status=ValidationStatus.VALID,
            cleaned_plate=cleaned,
            state_code="BH",
            state_name="Bharat Series (All-India)",
            format_type="BHARAT_SERIES",
            reason="Conforms to MoRTH Bharat Series standard",
            is_valid=True
        )
        
    # Standard prefix check
    state_code = cleaned[:2]
    if state_code not in INDIAN_STATES:
        # If state code is invalid, return INVALID or UNCERTAIN based on confidence
        if confidence < 0.65:
            return ValidationResult(
                status=ValidationStatus.UNCERTAIN,
                cleaned_plate=cleaned,
                state_code=state_code,
                state_name="Uncertain State Code",
                format_type="UNKNOWN_PREFIX",
                reason=f"Prefix '{state_code}' is not a recognized Indian State/UT code (confidence {confidence:.2f})",
                is_valid=False
            )
        return ValidationResult(
            status=ValidationStatus.INVALID,
            cleaned_plate=cleaned,
            state_code=state_code,
            state_name="Invalid State Code",
            format_type="INVALID_PREFIX",
            reason=f"Prefix '{state_code}' is not a recognized Indian State/UT code",
            is_valid=False
        )
        
    state_name = INDIAN_STATE_NAMES.get(state_code, "Indian Registered Vehicle")
    
    # Check Standard MoRTH format (MH01AV8669, TN45AB1234, DL3CD1210)
    if PATTERN_STANDARD_MORTH.match(cleaned):
        return ValidationResult(
            status=ValidationStatus.VALID,
            cleaned_plate=cleaned,
            state_code=state_code,
            state_name=state_name,
            format_type="STANDARD_MORTH",
            reason="Conforms to standard MoRTH Rule 50 syntax",
            is_valid=True
        )
        
    # Check Historical / Commercial / 2-row formats (e.g. KL498262, KL34A465)
    if PATTERN_HISTORICAL.match(cleaned):
        return ValidationResult(
            status=ValidationStatus.VALID,
            cleaned_plate=cleaned,
            state_code=state_code,
            state_name=state_name,
            format_type="HISTORICAL_OR_COMMERCIAL",
            reason="Conforms to valid historical or commercial Indian registration syntax",
            is_valid=True
        )
        
    # Check Diplomatic format
    if PATTERN_DIPLOMATIC.match(cleaned):
        return ValidationResult(
            status=ValidationStatus.VALID,
            cleaned_plate=cleaned,
            state_code="CD",
            state_name="Diplomatic / Consular Vehicle",
            format_type="DIPLOMATIC",
            reason="Conforms to Diplomatic registration syntax",
            is_valid=True
        )
        
    # If it starts with a valid state code but structure is slightly irregular
    # e.g. OCR missed one digit or read an extra stroke
    return ValidationResult(
        status=ValidationStatus.UNCERTAIN,
        cleaned_plate=cleaned,
        state_code=state_code,
        state_name=state_name,
        format_type="SYNTAX_MISMATCH",
        reason=f"Plate '{cleaned}' starts with valid state '{state_code}' but does not match standard registration patterns",
        is_valid=False
    )
