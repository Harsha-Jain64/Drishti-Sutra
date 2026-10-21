"""Module: Event Log Parser for Drishti-Sutra.

Parses structured device/system event logs of the form:
    YYYY-MM-DD HH:MM:SS LEVEL [component] message

All lines are processed; unparseable lines are stored as SKIPPED records
with a reason — no line is dropped silently.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import EventLogRecord, EvidenceItem, stable_id

# Primary log line regex
LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(INFO|DEBUG|WARN|WARNING|ERROR|CRITICAL)\s+"
    r"\[([^\]]+)\]\s+"
    r"(.+)$",
    re.IGNORECASE,
)

# KV pairs: key='quoted value' or key=unquoted_value
KV_PATTERN = re.compile(r"(\w+)=\'([^\']+)\'|(\w+)=(\S+)")

# Quoted username: User 'aarav.sharma'
ACTOR_PATTERN = re.compile(r"User\s+'([^']+)'")

# File path pattern
FILE_PATH_PATTERN = re.compile(r"(/[\w/.\-]+|[A-Za-z]:[\\][\w\\/.\-]+)")


def parse_datetime(ts: str) -> Optional[str]:
    """Parse 'YYYY-MM-DD HH:MM:SS' into ISO 8601 string."""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.isoformat()
    except ValueError:
        return None


def extract_kv(message: str) -> Dict[str, Any]:
    """Extract key=value pairs from a log message, handling quoted and unquoted values."""
    kv: Dict[str, Any] = {}
    for m in KV_PATTERN.finditer(message):
        if m.group(1) and m.group(2) is not None:
            key, val = m.group(1), m.group(2)
        else:
            key, val = m.group(3), m.group(4)

        # Parse numeric types if possible
        try:
            if "." in val:
                kv[key] = float(val)
            else:
                kv[key] = int(val)
        except (ValueError, TypeError):
            kv[key] = val
    return kv


def parse_event_log_file(evidence: EvidenceItem) -> List[EventLogRecord]:
    """Parse all lines of a structured event log into EventLogRecord objects.

    Unparseable and blank lines are stored as SKIPPED with a reason.
    Returns list of all records.
    """
    path = Path(evidence.file_path)
    records: List[EventLogRecord] = []

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        raw_content = f.read()

    lines = raw_content.splitlines()

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            records.append(
                EventLogRecord(
                    id=stable_id(evidence.id, str(line_no), "empty"),
                    evidence_id=evidence.id,
                    line_no=line_no,
                    timestamp_str="",
                    raw_line=raw_line,
                    status="SKIPPED",
                    skip_reason="Empty line",
                )
            )
            continue

        m = LOG_LINE_RE.match(stripped)
        if not m:
            records.append(
                EventLogRecord(
                    id=stable_id(evidence.id, str(line_no), stripped[:40]),
                    evidence_id=evidence.id,
                    line_no=line_no,
                    timestamp_str="",
                    raw_line=raw_line,
                    status="SKIPPED",
                    skip_reason="Does not match expected log format (YYYY-MM-DD HH:MM:SS LEVEL [component] message)",
                )
            )
            continue

        ts_str = m.group(1)
        level = m.group(2).upper().replace("WARNING", "WARN")
        component = m.group(3)
        message = m.group(4)

        parsed_dt = parse_datetime(ts_str)
        kv = extract_kv(message)

        # Extract actor if available
        actor_m = ACTOR_PATTERN.search(message)
        if actor_m:
            actor = actor_m.group(1)
        elif "Payee" in kv and isinstance(kv["Payee"], str):
            actor = kv["Payee"]
        elif "payee" in kv and isinstance(kv["payee"], str):
            actor = kv["payee"]
        else:
            actor = ""

        # Extract file path if mentioned
        fp_m = FILE_PATH_PATTERN.search(message)
        file_path = fp_m.group(1) if fp_m else ""

        records.append(
            EventLogRecord(
                id=stable_id(evidence.id, str(line_no), ts_str, message[:60]),
                evidence_id=evidence.id,
                line_no=line_no,
                timestamp_str=ts_str,
                parsed_datetime=parsed_dt,
                level=level,
                component=component,
                message=message,
                kv=kv,
                actor=actor,
                file_path=file_path,
                raw_line=raw_line,
                status="PARSED",
            )
        )

    return records
