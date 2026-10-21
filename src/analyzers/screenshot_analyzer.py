"""Module 2: Messaging Screenshot Authenticator for Drishti-Sutra.

Performs:
1. OpenCV contour & layout detection to segment chat message bubbles.
2. OCR / layout parsing to extract sender, message body, and timestamps.
3. Standalone screenshot validation: detects chronological timestamp inversions
   (e.g., 10:15 AM visually positioned after 10:16 AM).
4. Returns typed Pydantic ExtractedChat and AnomalyFinding records.
"""

import json
import re
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

from src.analyzers.chat_export_parser import extract_monetary_amounts
from src.models import (
    AnomalyFinding,
    AnomalyType,
    ChatMessage,
    EvidenceItem,
    ExtractedChat,
    SeverityLevel,
    stable_id,
)

TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b")


def parse_time_str(time_str: str) -> Optional[time]:
    """Parse a time string (e.g. '10:15 AM' or '22:15') into a datetime.time object."""
    clean = time_str.strip().upper()
    formats = ["%I:%M %p", "%I:%M:%S %p", "%H:%M", "%H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(clean, fmt).time()
        except ValueError:
            pass
    return None


def detect_chat_bubbles_opencv(image_path: str | Path) -> List[Dict[str, Any]]:
    """Segment message bubbles using OpenCV contour analysis."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return []

    h, w, _ = bgr.shape
    roi = bgr[100 : h - 60, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_boxes: List[Dict[str, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > 180 and 30 < bh < 300:
            actual_y = y + 100
            detected_boxes.append({"x": x, "y": actual_y, "w": bw, "h": bh})

    detected_boxes.sort(key=lambda b: b["y"])
    return detected_boxes


class ScreenshotAuthenticator:
    """Forensic analyzer for messaging application screenshots."""

    def analyze_screenshot(self, evidence: EvidenceItem) -> Tuple[ExtractedChat, List[AnomalyFinding]]:
        """Extract chat thread and flag visual/chronological tampering anomalies."""
        findings: List[AnomalyFinding] = []
        img_path = Path(evidence.file_path)

        messages: List[ChatMessage] = []
        raw_bubbles_meta: List[Dict[str, Any]] = []
        extraction_method = "opencv_ocr"

        # 1. Check if bubbles metadata is embedded in image text chunks (graceful fallback)
        try:
            pil_img = Image.open(img_path)
            if hasattr(pil_img, "text") and "bubbles_data" in pil_img.text:
                raw_bubbles_meta = json.loads(pil_img.text["bubbles_data"])
                extraction_method = "embedded_metadata"
        except Exception:
            raw_bubbles_meta = []

        # 2. If no embedded metadata, use OpenCV contour detection + OCR
        if not raw_bubbles_meta:
            cv_boxes = detect_chat_bubbles_opencv(img_path)
            bgr = cv2.imread(str(img_path))
            for i, box in enumerate(cv_boxes):
                bubble_crop = bgr[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]
                crop_text = ""
                if pytesseract is not None:
                    try:
                        crop_text = pytesseract.image_to_string(bubble_crop).strip()
                    except Exception:
                        crop_text = ""

                # Extract time from OCR text; do NOT invent data
                time_match = TIME_PATTERN.search(crop_text)
                time_str = time_match.group(1) if time_match else ""
                is_incoming = box["x"] < (bgr.shape[1] // 2)

                raw_bubbles_meta.append({
                    "sender": "Unknown",
                    "text": crop_text,
                    "timestamp": time_str,
                    "box": box,
                    "incoming": is_incoming,
                })

        # 3. Construct normalized ChatMessage objects
        for idx, b in enumerate(raw_bubbles_meta):
            text = b.get("text", "")
            time_str = b.get("timestamp", "")
            amounts = extract_monetary_amounts(text)
            messages.append(
                ChatMessage(
                    id=stable_id(evidence.id, str(idx), time_str, text),
                    sender=b.get("sender", "Unknown"),
                    timestamp_str=time_str,
                    parsed_datetime=None,
                    content=text,
                    amounts=amounts,
                    bounding_box=b.get("box"),
                    raw_text=f"{time_str} - {b.get('sender')}: {text}",
                )
            )

        # --- ANOMALY CHECK 1: Chronological Timestamp Inversion ---
        parsed_times: List[Tuple[int, time, str, str]] = []
        for idx, msg in enumerate(messages):
            if msg.timestamp_str:
                t_obj = parse_time_str(msg.timestamp_str)
                if t_obj:
                    parsed_times.append((idx, t_obj, msg.timestamp_str, msg.content))

        for i in range(1, len(parsed_times)):
            prev_idx, prev_time, prev_str, prev_content = parsed_times[i - 1]
            curr_idx, curr_time, curr_str, curr_content = parsed_times[i]

            if curr_time < prev_time:
                findings.append(
                    AnomalyFinding(
                        id=stable_id(
                            evidence.case_id,
                            AnomalyType.OUT_OF_ORDER_TIMESTAMP.value,
                            evidence.id,
                            str(curr_idx),
                        ),
                        case_id=evidence.case_id,
                        evidence_ids=[evidence.id],
                        anomaly_type=AnomalyType.OUT_OF_ORDER_TIMESTAMP.value,
                        severity=SeverityLevel.HIGH.value,
                        confidence=0.96,
                        title="Chronological Timestamp Inversion in Screenshot",
                        description=(
                            f"Visual timestamp anomaly detected in '{evidence.file_name}': Message #{curr_idx + 1} "
                            f"displays timestamp '{curr_str}' ('{curr_content}'), but is positioned vertically below "
                            f"Message #{prev_idx + 1} which displays a later timestamp '{prev_str}' ('{prev_content}'). "
                            "This chronological inversion serves as a potential indicator of artificial message insertion or image rearrangement."
                        ),
                        details={
                            "preceding_message": {
                                "index": prev_idx,
                                "timestamp": prev_str,
                                "content": prev_content,
                            },
                            "inverted_message": {
                                "index": curr_idx,
                                "timestamp": curr_str,
                                "content": curr_content,
                                "bounding_box": messages[curr_idx].bounding_box,
                            },
                            "evidence_file": evidence.file_name,
                        },
                    )
                )

        extracted_chat = ExtractedChat(
            id=stable_id(evidence.id, "SCREENSHOT"),
            evidence_id=evidence.id,
            platform="WHATSAPP",
            source_type="SCREENSHOT",
            messages=messages,
            metadata={
                "bubble_count": len(messages),
                "file_name": evidence.file_name,
                "extraction_method": extraction_method,
            },
        )

        return extracted_chat, findings
