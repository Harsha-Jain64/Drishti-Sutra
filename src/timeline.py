"""Module 3: Timeline Reconstructor for Drishti-Sutra.

Merges, chronologically aligns, and scores events across documents, chat communications,
and system audit logs into a unified, tamper-evident timeline.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from src import config
from src.database import DatabaseManager
from src.models import (
    ChatMessage,
    Contradiction,
    EventLogRecord,
    ExtractedChat,
    TimelineEvent,
    stable_id,
)

SOURCE_PRIORITY = {
    "system_events.log": 1,
    "whatsapp_chat_export.txt": 2,
    "whatsapp_screenshot.png": 3,
}


def parse_chat_time(ts_str: str) -> Optional[Tuple[int, int]]:
    """Parse hour and minute from time strings like '10:15 AM'."""
    from src.truth_engine import parse_time_only
    t = parse_time_only(ts_str)
    return (t.hour, t.minute) if t else None


def build_timeline(case_id: str, db: DatabaseManager) -> List[TimelineEvent]:
    """Reconstruct a unified, scored chronological timeline from persisted case records.

    Merges chat export entries, screenshot bubbles with timestamps (date inferred),
    and parsed event log records. Persists generated events to SQLite.
    """
    evidence_items = {e.id: e for e in db.list_evidence(case_id)}
    chats: List[ExtractedChat] = db.list_extracted_chats(case_id)
    logs: List[EventLogRecord] = db.list_event_log_records_for_case(case_id)
    contradictions: List[Contradiction] = db.list_contradictions(case_id)
    findings = db.list_findings(case_id)

    # Determine dominant conversation date from export or logs
    dominant_date: Optional[str] = None
    for chat in chats:
        if chat.source_type == "EXPORT_TXT":
            for msg in chat.messages:
                if msg.parsed_datetime:
                    dominant_date = msg.parsed_datetime.split("T")[0]
                    break
        if dominant_date:
            break

    if not dominant_date:
        for r in logs:
            if r.parsed_datetime:
                dominant_date = r.parsed_datetime.split("T")[0]
                break

    if not dominant_date:
        dominant_date = datetime.now().strftime("%Y-%m-%d")

    # Collect anomalous timestamps/sources from contradictions and indicators
    anomaly_timestamps: Set[str] = set()
    for c in contradictions:
        for claim in (c.claim_a, c.claim_b):
            if isinstance(claim, dict) and "timestamp" in claim and claim["timestamp"]:
                anomaly_timestamps.add(str(claim["timestamp"]).strip())
    for f in findings:
        if f.details and "timestamp" in f.details:
            anomaly_timestamps.add(str(f.details["timestamp"]).strip())

    events: List[TimelineEvent] = []

    # 1. Integrate System Log Records
    for r in logs:
        if r.status != "PARSED" or not r.parsed_datetime:
            continue
        ev = evidence_items.get(r.evidence_id)
        fn = ev.file_name if ev else "system_events.log"

        is_anom = r.level in ("WARN", "ERROR") or any(
            ts in r.timestamp_str for ts in anomaly_timestamps
        )

        events.append(
            TimelineEvent(
                id=stable_id(case_id, "TIMELINE_LOG", r.id),
                case_id=case_id,
                evidence_id=r.evidence_id,
                timestamp_str=r.timestamp_str,
                parsed_datetime=r.parsed_datetime,
                event_type="SYSTEM_LOG",
                actor=r.actor or "System",
                description=f"[{r.component.upper()}] {r.message}",
                source_reference=f"{fn}:L{r.line_no}",
                confidence=1.0,
                is_anomaly=is_anom,
                metadata={
                    "level": r.level,
                    "component": r.component,
                    "kv": r.kv,
                    "extraction_method": "log_parser",
                    "date_inferred": False,
                    "timezone_assumed": True,
                },
            )
        )

    # 2. Integrate Chat Export Messages
    for chat in chats:
        if chat.source_type != "EXPORT_TXT":
            continue
        ev = evidence_items.get(chat.evidence_id)
        fn = ev.file_name if ev else "chat_export.txt"

        for idx, msg in enumerate(chat.messages):
            is_anom = any(ts in msg.timestamp_str for ts in anomaly_timestamps)
            amt_info = f" [Amounts: {msg.amounts}]" if msg.amounts else ""

            events.append(
                TimelineEvent(
                    id=stable_id(case_id, "TIMELINE_EXPORT", chat.evidence_id, str(idx)),
                    case_id=case_id,
                    evidence_id=chat.evidence_id,
                    timestamp_str=msg.timestamp_str,
                    parsed_datetime=msg.parsed_datetime,
                    event_type="CHAT_MESSAGE",
                    actor=msg.sender,
                    description=f"{msg.sender}: {msg.content}{amt_info}",
                    source_reference=f"{fn}:Msg{idx + 1}",
                    confidence=0.98,
                    is_anomaly=is_anom,
                    metadata={
                        "extraction_method": "regex_export",
                        "date_inferred": False,
                        "amounts": msg.amounts,
                        "timezone_assumed": True,
                    },
                )
            )

    # 3. Integrate Screenshot Messages (with inferred full date)
    for chat in chats:
        if chat.source_type != "SCREENSHOT":
            continue
        ev = evidence_items.get(chat.evidence_id)
        fn = ev.file_name if ev else "screenshot.png"

        for idx, msg in enumerate(chat.messages):
            if not msg.timestamp_str:
                continue

            hm = parse_chat_time(msg.timestamp_str)
            iso_dt: Optional[str] = None
            if hm:
                h, m = hm
                iso_dt = f"{dominant_date}T{h:02d}:{m:02d}:00"

            is_anom = any(ts in msg.timestamp_str for ts in anomaly_timestamps)
            amt_info = f" [Amounts: {msg.amounts}]" if msg.amounts else ""

            # Score adjustments
            conf = 0.90 - config.PENALTY_INFERRED_DATE
            if chat.metadata.get("extraction_method") == "opencv_ocr":
                conf -= config.PENALTY_OCR_EXTRACTION

            events.append(
                TimelineEvent(
                    id=stable_id(case_id, "TIMELINE_SCREENSHOT", chat.evidence_id, str(idx)),
                    case_id=case_id,
                    evidence_id=chat.evidence_id,
                    timestamp_str=f"{dominant_date} {msg.timestamp_str}",
                    parsed_datetime=iso_dt,
                    event_type="CHAT_MESSAGE",
                    actor=msg.sender,
                    description=f"{msg.sender}: {msg.content}{amt_info}",
                    source_reference=f"{fn}:Bubble{idx + 1}",
                    confidence=round(conf, 2),
                    is_anomaly=is_anom,
                    metadata={
                        "extraction_method": chat.metadata.get("extraction_method", "opencv_ocr"),
                        "date_inferred": True,
                        "amounts": msg.amounts,
                        "bounding_box": msg.bounding_box,
                        "timezone_assumed": True,
                    },
                )
            )

    # Deterministic chronological sort: (parsed_datetime, source priority, source_reference)
    def sort_key(e: TimelineEvent) -> Tuple[str, int, str]:
        prio = 99
        for k, v in SOURCE_PRIORITY.items():
            if k in e.source_reference:
                prio = v
                break
        return (e.parsed_datetime or "9999", prio, e.source_reference)

    events.sort(key=sort_key)

    # Flag anomalous gap intervals between successive events
    for k in range(1, len(events)):
        prev_e = events[k - 1]
        curr_e = events[k]
        if prev_e.parsed_datetime and curr_e.parsed_datetime:
            try:
                dt_p = datetime.fromisoformat(prev_e.parsed_datetime)
                dt_c = datetime.fromisoformat(curr_e.parsed_datetime)
                diff_min = (dt_c - dt_p).total_seconds() / 60.0
                if diff_min > config.ANOMALOUS_GAP_MINUTES:
                    curr_e.metadata["anomalous_gap_after_previous_minutes"] = round(diff_min, 1)
            except ValueError:
                pass

    # Persist all timeline events
    for e in events:
        db.insert_timeline_event(e)

    return events
