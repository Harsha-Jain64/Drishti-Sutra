"""Parser for standard plain-text chat exports (WhatsApp, Telegram, Signal)."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import ChatMessage, EvidenceItem, ExtractedChat, stable_id

# Regex patterns matching common chat export formats
WHATSAPP_EXPORT_PATTERN_1 = re.compile(
    r"^\[(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]\s+([^:]+):\s+(.*)$"
)
WHATSAPP_EXPORT_PATTERN_2 = re.compile(
    r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\s*-\s*([^:]+):\s+(.*)$"
)
AMOUNT_PATTERN = re.compile(
    r"(?:[\u20b9$€£]|Rs\.?|INR)\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)",
    re.IGNORECASE,
)


def extract_monetary_amounts(text: str) -> List[float]:
    """Find all monetary values mentioned in message text."""
    amounts: List[float] = []
    for match in AMOUNT_PATTERN.finditer(text):
        num_str = match.group(1).replace(",", "")
        try:
            amounts.append(float(num_str))
        except ValueError:
            pass
    return amounts


def parse_timestamp_to_iso(date_str: str, time_str: str) -> Optional[str]:
    """Attempt parsing varied date & time string formats to standardized ISO format."""
    combined = f"{date_str.strip()} {time_str.strip()}"
    candidates = [
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%Y %I:%M %p",
        "%d-%m-%Y %I:%M:%S %p",
        "%d-%m-%Y %I:%M %p",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%y %I:%M %p",
        "%m/%d/%Y %I:%M:%S %p",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(combined, fmt)
            return dt.isoformat()
        except ValueError:
            pass
    return None


def parse_chat_export_file(evidence: EvidenceItem) -> ExtractedChat:
    """Parse a plain text chat export file into normalized ExtractedChat object."""
    path = Path(evidence.file_path)
    messages: List[ChatMessage] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_msg: Optional[ChatMessage] = None
    msg_idx = 0

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        match1 = WHATSAPP_EXPORT_PATTERN_1.match(line_clean)
        match2 = WHATSAPP_EXPORT_PATTERN_2.match(line_clean) if not match1 else None
        match = match1 or match2

        if match:
            if current_msg:
                messages.append(current_msg)

            date_part = match.group(1)
            time_part = match.group(2)
            sender = match.group(3).strip()
            content = match.group(4).strip()

            iso_dt = parse_timestamp_to_iso(date_part, time_part)
            amounts = extract_monetary_amounts(content)
            ts_str = f"{date_part} {time_part}"

            current_msg = ChatMessage(
                id=stable_id(evidence.id, str(msg_idx), ts_str, content),
                sender=sender,
                timestamp_str=ts_str,
                parsed_datetime=iso_dt,
                content=content,
                amounts=amounts,
                raw_text=line_clean,
            )
            msg_idx += 1
        else:
            if current_msg:
                current_msg.content += f"\n{line_clean}"
                new_amounts = extract_monetary_amounts(line_clean)
                current_msg.amounts.extend(new_amounts)

    if current_msg:
        messages.append(current_msg)

    return ExtractedChat(
        id=stable_id(evidence.id, "EXPORT_TXT"),
        evidence_id=evidence.id,
        platform="WHATSAPP",
        source_type="EXPORT_TXT",
        messages=messages,
        metadata={
            "total_messages": len(messages),
            "file_name": evidence.file_name,
            "extraction_method": "regex_export",
        },
    )
