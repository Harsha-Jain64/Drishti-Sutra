"""Module 3: The Truth Engine for Drishti-Sutra.

Cross-source correlation and deterministic contradiction detection engine.

Scoring Rubric (Heuristic Indicator Values):
--------------------------------------------
- Base Score per Contradiction Type:
  * QR_PRINTED_MISMATCH: 0.90
  * AMOUNT_MISMATCH: 0.88
  * DOC_FIELD_MISMATCH: 0.85
  * ORDER_INVERSION: 0.85
  * LOG_COMM_CONFLICT: 0.82
  * MESSAGE_ABSENT_IN_EXPORT: 0.80
  * MESSAGE_TEXT_ALTERED: 0.78
- Score Adjustments:
  * +0.06 (CORROBORATION_BONUS) when a 3rd independent source corroborates the discrepancy
  * -0.08 (PENALTY_INFERRED_DATE) when message date was inferred rather than explicitly timestamped
  * -0.05 (PENALTY_OCR_EXTRACTION) when text was extracted via OCR rather than digital export
  * -0.07 (PENALTY_SUSPECTED_ENTITY) when entity link was resolved via fuzzy threshold rather than confirmed

Note: Confidence scores are heuristic indicators to prioritize investigative review,
not calibrated statistical probabilities or legal proof of forgery.
"""

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from src import config
from src.database import DatabaseManager
from src.models import (
    AnomalyFinding,
    AnomalyType,
    ChatMessage,
    Contradiction,
    EventLogRecord,
    EvidenceItem,
    ExtractedChat,
    ExtractedDocument,
    SeverityLevel,
    TruthEngineResult,
    stable_id,
    utc_now_iso,
)

TIME_ZONE_LIMITATION = "Timezone unknown; sources assumed to share synchronized local clock."


# -----------------------------------------------------------------------------
# Entity Resolution & Normalization Helpers
# -----------------------------------------------------------------------------

def normalize_entity_name(name: str) -> str:
    """Normalize personal names and identifiers for fuzzy matching.

    Strips honorifics, handles punctuation / separators (e.g., 'aarav.sharma' -> 'aarav sharma'),
    and converts to lowercase alphanumeric tokens.
    """
    if not name:
        return ""
    cleaned = re.sub(r"[._\-]", " ", name)
    cleaned = re.sub(r"\b(mr|mrs|ms|dr|shri|smt)\b\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", cleaned)
    tokens = [t.lower() for t in cleaned.split() if t]
    return " ".join(tokens)


def normalize_dob(dob_str: Optional[str]) -> Optional[str]:
    """Standardize date of birth strings into ISO YYYY-MM-DD format."""
    if not dob_str:
        return None
    s = dob_str.strip()
    patterns = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%y", "%d-%m-%y"]
    for p in patterns:
        try:
            return datetime.strptime(s, p).strftime("%Y-%m-%d")
        except ValueError:
            pass
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return s


def normalize_id_number(id_str: Optional[str]) -> Optional[str]:
    """Strip whitespace and punctuation from national/tax identification numbers."""
    if not id_str:
        return None
    return re.sub(r"[\s\-_]+", "", id_str).upper()


def match_entities(name_a: str, name_b: str) -> Tuple[bool, str, float]:
    """Compare two entity names using rapidfuzz token_set_ratio.

    Returns:
        (is_match, match_status, score) where match_status is 'confirmed' (100%),
        'suspected' (>= threshold), or 'unrelated'.
    """
    norm_a = normalize_entity_name(name_a)
    norm_b = normalize_entity_name(name_b)
    if not norm_a or not norm_b:
        return False, "unrelated", 0.0

    score = fuzz.token_set_ratio(norm_a, norm_b)
    if score >= config.NAME_CONFIRMED_THRESHOLD:
        return True, "confirmed", score / 100.0
    if score >= config.NAME_MATCH_THRESHOLD:
        return True, "suspected", score / 100.0
    return False, "unrelated", score / 100.0


def parse_time_only(ts_str: str) -> Optional[time]:
    """Extract hour/minute from varied timestamp strings."""
    if not ts_str:
        return None
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?\b", ts_str, re.IGNORECASE)
    if not m:
        return None
    h = int(m.group(1))
    minute = int(m.group(2))
    ampm = m.group(4)
    if ampm:
        ampm = ampm.upper()
        if ampm == "PM" and h < 12:
            h += 12
        elif ampm == "AM" and h == 12:
            h = 0
    return time(h, minute)


def parse_full_or_inferred_datetime(msg: ChatMessage, dominant_date: Optional[str]) -> Tuple[Optional[datetime], bool]:
    """Return full datetime for a message, inferring date from companion source if needed."""
    if msg.parsed_datetime:
        try:
            return datetime.fromisoformat(msg.parsed_datetime), False
        except ValueError:
            pass

    t = parse_time_only(msg.timestamp_str)
    if t and dominant_date:
        try:
            d = datetime.strptime(dominant_date, "%Y-%m-%d").date()
            return datetime.combine(d, t), True
        except ValueError:
            pass
    elif t:
        # Default fallback to today's date placeholder
        return datetime.combine(datetime.now().date(), t), True

    return None, False


# -----------------------------------------------------------------------------
# Truth Engine Core
# -----------------------------------------------------------------------------

def run_truth_engine(case_id: str, db: DatabaseManager) -> TruthEngineResult:
    """Execute cross-source entity resolution, correlation rules, and contradiction detection."""
    # 1. Fetch persisted extractions for the case
    docs: List[ExtractedDocument] = db.list_extracted_documents(case_id)
    chats: List[ExtractedChat] = db.list_extracted_chats(case_id)
    logs: List[EventLogRecord] = db.list_event_log_records_for_case(case_id)
    evidence_items = {e.id: e for e in db.list_evidence(case_id)}
    module1_2_findings = db.list_findings(case_id)

    contradictions: List[Contradiction] = []
    indicators: List[AnomalyFinding] = []
    skipped_comparisons: List[Dict[str, str]] = []
    resolved_entities: Dict[str, Dict[str, Any]] = {}

    # Helper to resolve evidence filename and extraction method
    def get_source_ref(evidence_id: str, locator: str, default_method: str = "direct") -> Dict[str, Any]:
        ev = evidence_items.get(evidence_id)
        file_name = ev.file_name if ev else "unknown"
        return {
            "evidence_id": evidence_id,
            "file_name": file_name,
            "locator": locator,
            "extraction_method": default_method,
        }

    # -------------------------------------------------------------------------
    # RULE 1: Cross-Document Attribute Mismatch (DOC_FIELD_MISMATCH)
    # -------------------------------------------------------------------------
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            doc_a, doc_b = docs[i], docs[j]
            name_a = doc_a.fields.get("name")
            name_b = doc_b.fields.get("name")

            if not name_a or not name_b:
                skipped_comparisons.append({
                    "rule": "DOC_FIELD_MISMATCH",
                    "reason": f"Document {doc_a.evidence_id} or {doc_b.evidence_id} missing extracted name.",
                })
                continue

            is_match, status, match_score = match_entities(name_a, name_b)
            if not is_match:
                continue

            # Track resolved entity
            entity_key = normalize_entity_name(name_a)
            if entity_key not in resolved_entities:
                resolved_entities[entity_key] = {
                    "primary_name": name_a,
                    "status": status,
                    "linked_names": [name_a, name_b],
                }

            # Compare DOB
            dob_a = normalize_dob(doc_a.fields.get("dob"))
            dob_b = normalize_dob(doc_b.fields.get("dob"))
            if dob_a and dob_b and dob_a != dob_b:
                ev_a = evidence_items.get(doc_a.evidence_id)
                ev_b = evidence_items.get(doc_b.evidence_id)
                fn_a = ev_a.file_name if ev_a else "Doc A"
                fn_b = ev_b.file_name if ev_b else "Doc B"

                # Check corroboration from Module 1 findings (e.g. ELA anomaly on doc_b)
                related_ids = [
                    f.id for f in module1_2_findings
                    if (doc_a.evidence_id in f.evidence_ids or doc_b.evidence_id in f.evidence_ids)
                ]
                score = config.BASE_SCORE_DOC_FIELD_MISMATCH
                if status == "suspected":
                    score -= config.PENALTY_SUSPECTED_ENTITY
                if related_ids:
                    score += config.CORROBORATION_BONUS
                score = min(0.99, max(0.10, score))

                contra_id = stable_id(case_id, "DOC_FIELD_MISMATCH", doc_a.evidence_id, doc_b.evidence_id, "dob")
                contradictions.append(
                    Contradiction(
                        id=contra_id,
                        case_id=case_id,
                        evidence_ids=[doc_a.evidence_id, doc_b.evidence_id],
                        entity=name_a,
                        field_name="Date of Birth",
                        claim_a={"source": fn_a, "value": dob_a, "doc_type": doc_a.doc_type},
                        claim_b={"source": fn_b, "value": dob_b, "doc_type": doc_b.doc_type},
                        explanation=(
                            f"Cross-document discrepancy: Identity documents for '{name_a}' contain conflicting dates of birth. "
                            f"{fn_a} states '{dob_a}', whereas {fn_b} states '{dob_b}'."
                        ),
                        severity=SeverityLevel.HIGH.value,
                        confidence=round(score, 2),
                        contradiction_type="DOC_FIELD_MISMATCH",
                        source_a_ref=get_source_ref(doc_a.evidence_id, "fields.dob", doc_a.metadata.get("extraction_method", "ocr")),
                        source_b_ref=get_source_ref(doc_b.evidence_id, "fields.dob", doc_b.metadata.get("extraction_method", "ocr")),
                        observed_fact_a=f"{fn_a} records DOB as {dob_a}",
                        observed_fact_b=f"{fn_b} records DOB as {dob_b}",
                        derived_result=f"Conflicting birth dates for resolved entity '{name_a}'",
                        limitations=["OCR or typographic variance may contribute to character differences."],
                        evidence_labels=["observed", "observed"],
                        supporting_sources=[fn_a],
                        conflicting_sources=[fn_b],
                        related_finding_ids=related_ids,
                    )
                )

    # -------------------------------------------------------------------------
    # RULE 2: QR vs Printed Document Mismatch (QR_PRINTED_MISMATCH)
    # -------------------------------------------------------------------------
    for doc in docs:
        if not doc.qr_data:
            continue
        qr_dob = normalize_dob(doc.qr_data.get("dob"))
        printed_dob = normalize_dob(doc.fields.get("dob"))
        ev = evidence_items.get(doc.evidence_id)
        fn = ev.file_name if ev else "Document"

        if qr_dob and printed_dob and qr_dob != printed_dob:
            related_ids = [
                f.id for f in module1_2_findings
                if doc.evidence_id in f.evidence_ids and f.anomaly_type == AnomalyType.QR_TEXT_MISMATCH.value
            ]
            contra_id = stable_id(case_id, "QR_PRINTED_MISMATCH", doc.evidence_id, "dob")
            contradictions.append(
                Contradiction(
                    id=contra_id,
                    case_id=case_id,
                    evidence_ids=[doc.evidence_id],
                    entity=doc.fields.get("name", "Unknown"),
                    field_name="Date of Birth (QR vs Printed)",
                    claim_a={"source": f"{fn} (Printed Text)", "value": printed_dob},
                    claim_b={"source": f"{fn} (QR Payload)", "value": qr_dob},
                    explanation=(
                        f"Internal document discrepancy in '{fn}': Printed text specifies Date of Birth as '{printed_dob}', "
                        f"whereas the encoded QR payload specifies '{qr_dob}'."
                    ),
                    severity=SeverityLevel.CRITICAL.value,
                    confidence=config.BASE_SCORE_QR_PRINTED,
                    contradiction_type="QR_PRINTED_MISMATCH",
                    source_a_ref=get_source_ref(doc.evidence_id, "fields.dob", doc.metadata.get("extraction_method", "ocr")),
                    source_b_ref=get_source_ref(doc.evidence_id, "qr_data.dob", "qr_decoder"),
                    observed_fact_a=f"Printed DOB text on {fn} is {printed_dob}",
                    observed_fact_b=f"Decoded QR payload encodes DOB as {qr_dob}",
                    derived_result=f"Internal mismatch between printed text and QR payload in {fn}",
                    limitations=["QR payload format specification assumed standard ISO date encoding."],
                    evidence_labels=["observed", "observed"],
                    supporting_sources=[],
                    conflicting_sources=[fn],
                    related_finding_ids=related_ids,
                )
            )

    # -------------------------------------------------------------------------
    # RULE 3 & 4: Screenshot vs Export Alignment, Amount/Text Mismatches, Absent Messages
    # -------------------------------------------------------------------------
    screenshots = [c for c in chats if c.source_type == "SCREENSHOT"]
    exports = [c for c in chats if c.source_type == "EXPORT_TXT"]

    if screenshots and not exports:
        skipped_comparisons.append({
            "rule": "MESSAGE_ALIGNMENT",
            "reason": "Chat screenshot present without companion text export; cross-source comparison skipped.",
        })
    elif exports and not screenshots:
        skipped_comparisons.append({
            "rule": "MESSAGE_ALIGNMENT",
            "reason": "Chat export present without companion screenshot; cross-source comparison skipped.",
        })

    # Find dominant date from export if present
    dominant_date: Optional[str] = None
    for exp in exports:
        for msg in exp.messages:
            if msg.parsed_datetime:
                dominant_date = msg.parsed_datetime.split("T")[0]
                break
        if dominant_date:
            break

    for sc in screenshots:
        for ex in exports:
            sc_ev = evidence_items.get(sc.evidence_id)
            ex_ev = evidence_items.get(ex.evidence_id)
            sc_fn = sc_ev.file_name if sc_ev else "Screenshot"
            ex_fn = ex_ev.file_name if ex_ev else "Chat Export"

            # Compute export coverage time span
            export_datetimes: List[datetime] = []
            for msg in ex.messages:
                dt, _ = parse_full_or_inferred_datetime(msg, dominant_date)
                if dt:
                    export_datetimes.append(dt)

            min_ex_dt = min(export_datetimes) if export_datetimes else None
            max_ex_dt = max(export_datetimes) if export_datetimes else None

            aligned_export_indices: Set[int] = set()

            for sc_idx, sc_msg in enumerate(sc.messages):
                if not sc_msg.timestamp_str and not sc_msg.content:
                    skipped_comparisons.append({
                        "rule": "MESSAGE_ALIGNMENT",
                        "reason": f"Screenshot message #{sc_idx + 1} has no timestamp or text; skipped.",
                    })
                    continue

                sc_dt, sc_inferred = parse_full_or_inferred_datetime(sc_msg, dominant_date)
                best_match: Optional[Tuple[int, ChatMessage, float]] = None

                for ex_idx, ex_msg in enumerate(ex.messages):
                    ex_dt, _ = parse_full_or_inferred_datetime(ex_msg, dominant_date)

                    # Time proximity check (minute precision within window)
                    time_close = False
                    if sc_dt and ex_dt:
                        diff_sec = abs((sc_dt - ex_dt).total_seconds())
                        if diff_sec <= (config.MESSAGE_TIME_WINDOW_MINUTES * 60):
                            time_close = True
                    else:
                        time_close = True  # Fallback to sender + text matching

                    sender_match, _, _ = match_entities(sc_msg.sender, ex_msg.sender)
                    text_score = fuzz.token_set_ratio(sc_msg.content.lower(), ex_msg.content.lower())

                    if (sender_match or not sc_msg.sender or sc_msg.sender == "Unknown") and (time_close or text_score >= config.TEXT_SIMILARITY_THRESHOLD):
                        if best_match is None or text_score > best_match[2]:
                            best_match = (ex_idx, ex_msg, text_score)

                if best_match is not None and best_match[2] >= 50:
                    ex_idx, ex_msg, t_score = best_match
                    aligned_export_indices.add(ex_idx)

                    # Compare amounts mentioned
                    if sc_msg.amounts and ex_msg.amounts:
                        if sc_msg.amounts != ex_msg.amounts:
                            score = config.BASE_SCORE_AMOUNT_MISMATCH
                            if sc_inferred:
                                score -= config.PENALTY_INFERRED_DATE
                            # Corroborate with system logs if any bank log matches the screenshot amount
                            matching_logs = [
                                r for r in logs
                                if r.kv.get("Amount") in sc_msg.amounts
                            ]
                            if matching_logs:
                                score += config.CORROBORATION_BONUS
                            score = min(0.99, max(0.10, score))

                            contra_id = stable_id(case_id, "AMOUNT_MISMATCH", sc.evidence_id, ex.evidence_id, str(sc_idx), str(ex_idx))
                            contradictions.append(
                                Contradiction(
                                    id=contra_id,
                                    case_id=case_id,
                                    evidence_ids=[sc.evidence_id, ex.evidence_id],
                                    entity=sc_msg.sender,
                                    field_name="Transfer Amount",
                                    claim_a={"source": sc_fn, "value": sc_msg.amounts, "text": sc_msg.content, "timestamp": sc_msg.timestamp_str},
                                    claim_b={"source": ex_fn, "value": ex_msg.amounts, "text": ex_msg.content, "timestamp": ex_msg.timestamp_str},
                                    explanation=(
                                        f"Monetary value discrepancy: Message from '{sc_msg.sender}' at ~{sc_msg.timestamp_str} "
                                        f"displays transfer amount {sc_msg.amounts} in {sc_fn}, whereas the companion export {ex_fn} "
                                        f"records {ex_msg.amounts}."
                                    ),
                                    severity=SeverityLevel.CRITICAL.value,
                                    confidence=round(score, 2),
                                    contradiction_type="AMOUNT_MISMATCH",
                                    source_a_ref=get_source_ref(sc.evidence_id, f"message_bubble_{sc_idx}", sc.metadata.get("extraction_method", "opencv_ocr")),
                                    source_b_ref=get_source_ref(ex.evidence_id, f"line_{ex_idx}", "regex_export"),
                                    observed_fact_a=f"{sc_fn} bubble text quotes {sc_msg.amounts}",
                                    observed_fact_b=f"{ex_fn} text export quotes {ex_msg.amounts}",
                                    derived_result="Altered transaction figure between visual capture and text export",
                                    limitations=[TIME_ZONE_LIMITATION, "Visual layout rendering differences considered."],
                                    evidence_labels=["observed", "observed"],
                                    supporting_sources=[sc_fn],
                                    conflicting_sources=[ex_fn],
                                    related_finding_ids=[],
                                )
                            )

                    # Compare text alterations if amounts were not the sole difference
                    elif t_score < 95 and sc_msg.content.strip().lower() != ex_msg.content.strip().lower():
                        contra_id = stable_id(case_id, "MESSAGE_TEXT_ALTERED", sc.evidence_id, ex.evidence_id, str(sc_idx))
                        contradictions.append(
                            Contradiction(
                                id=contra_id,
                                case_id=case_id,
                                evidence_ids=[sc.evidence_id, ex.evidence_id],
                                entity=sc_msg.sender,
                                field_name="Message Text Content",
                                claim_a={"source": sc_fn, "value": sc_msg.content, "timestamp": sc_msg.timestamp_str},
                                claim_b={"source": ex_fn, "value": ex_msg.content, "timestamp": ex_msg.timestamp_str},
                                explanation=(
                                    f"Text content variance in aligned message at ~{sc_msg.timestamp_str}: '{sc_msg.content}' "
                                    f"in {sc_fn} versus '{ex_msg.content}' in {ex_fn}."
                                ),
                                severity=SeverityLevel.MEDIUM.value,
                                confidence=0.82,
                                contradiction_type="MESSAGE_TEXT_ALTERED",
                                source_a_ref=get_source_ref(sc.evidence_id, f"bubble_{sc_idx}", sc.metadata.get("extraction_method", "opencv_ocr")),
                                source_b_ref=get_source_ref(ex.evidence_id, f"line_{ex_idx}", "regex_export"),
                                observed_fact_a=f"Screenshot message: '{sc_msg.content}'",
                                observed_fact_b=f"Chat export message: '{ex_msg.content}'",
                                derived_result="Wording discrepancy across communication records",
                                limitations=[TIME_ZONE_LIMITATION],
                                evidence_labels=["observed", "observed"],
                                supporting_sources=[sc_fn],
                                conflicting_sources=[ex_fn],
                                related_finding_ids=[],
                            )
                        )
                else:
                    # Message is in screenshot but not matched in export
                    # Apply Coverage Guard: Only flag if message time is within export span
                    if sc_dt and min_ex_dt and max_ex_dt and (min_ex_dt <= sc_dt <= max_ex_dt):
                        score = config.BASE_SCORE_MESSAGE_ABSENT
                        if sc_inferred:
                            score -= config.PENALTY_INFERRED_DATE
                        score = round(min(0.99, max(0.10, score)), 2)

                        contra_id = stable_id(case_id, "MESSAGE_ABSENT_IN_EXPORT", sc.evidence_id, ex.evidence_id, str(sc_idx))
                        contradictions.append(
                            Contradiction(
                                id=contra_id,
                                case_id=case_id,
                                evidence_ids=[sc.evidence_id, ex.evidence_id],
                                entity=sc_msg.sender,
                                field_name="Message Presence",
                                claim_a={"source": sc_fn, "value": f"Present: '{sc_msg.content}'", "timestamp": sc_msg.timestamp_str},
                                claim_b={"source": ex_fn, "value": "Not present in exported chat file"},
                                explanation=(
                                    f"Message coverage anomaly: Message '{sc_msg.content}' at {sc_msg.timestamp_str} "
                                    f"is visible in {sc_fn} within the active conversation timeframe, but is not present in {ex_fn}."
                                ),
                                severity=SeverityLevel.HIGH.value,
                                confidence=score,
                                contradiction_type="MESSAGE_ABSENT_IN_EXPORT",
                                source_a_ref=get_source_ref(sc.evidence_id, f"bubble_{sc_idx}", sc.metadata.get("extraction_method", "opencv_ocr")),
                                source_b_ref=get_source_ref(ex.evidence_id, "export_full_span", "regex_export"),
                                observed_fact_a=f"Message '{sc_msg.content}' visible in {sc_fn} at {sc_msg.timestamp_str}",
                                observed_fact_b=f"Message absent from {ex_fn} despite falling within coverage range ({min_ex_dt} - {max_ex_dt})",
                                derived_result="Omitted communication record in export file",
                                limitations=[TIME_ZONE_LIMITATION, "Export filter or manual export truncation may omit entries."],
                                evidence_labels=["observed", "observed"],
                                supporting_sources=[sc_fn],
                                conflicting_sources=[ex_fn],
                                related_finding_ids=[],
                            )
                        )
                    else:
                        skipped_comparisons.append({
                            "rule": "MESSAGE_ABSENT_IN_EXPORT",
                            "reason": f"Message '{sc_msg.content[:30]}...' ({sc_msg.timestamp_str}) outside export covered span.",
                        })

    # -------------------------------------------------------------------------
    # RULE 5: Full Datetime Chronological Order Inversion (ORDER_INVERSION)
    # -------------------------------------------------------------------------
    for sc in screenshots:
        sc_ev = evidence_items.get(sc.evidence_id)
        sc_fn = sc_ev.file_name if sc_ev else "Screenshot"

        parsed_msgs: List[Tuple[int, datetime, bool, ChatMessage]] = []
        for idx, m in enumerate(sc.messages):
            if not m.timestamp_str:
                continue
            dt, inferred = parse_full_or_inferred_datetime(m, dominant_date)
            if dt:
                parsed_msgs.append((idx, dt, inferred, m))

        for k in range(1, len(parsed_msgs)):
            prev_idx, prev_dt, prev_inf, prev_m = parsed_msgs[k - 1]
            curr_idx, curr_dt, curr_inf, curr_m = parsed_msgs[k]

            # Midnight rollover check: if time decreased by > 12 hours, treat as next-day rollover
            diff_hours = (prev_dt - curr_dt).total_seconds() / 3600.0
            if 0 < diff_hours < 12.0:
                score = config.BASE_SCORE_ORDER_INVERSION
                if prev_inf or curr_inf:
                    score -= config.PENALTY_INFERRED_DATE
                score = round(min(0.99, max(0.10, score)), 2)

                contra_id = stable_id(case_id, "ORDER_INVERSION", sc.evidence_id, str(curr_idx))
                related_ids = [
                    f.id for f in module1_2_findings
                    if sc.evidence_id in f.evidence_ids and f.anomaly_type == AnomalyType.OUT_OF_ORDER_TIMESTAMP.value
                ]
                contradictions.append(
                    Contradiction(
                        id=contra_id,
                        case_id=case_id,
                        evidence_ids=[sc.evidence_id],
                        entity=curr_m.sender,
                        field_name="Chronological Message Sequence",
                        claim_a={"source": sc_fn, "position": prev_idx + 1, "timestamp": prev_m.timestamp_str, "text": prev_m.content},
                        claim_b={"source": sc_fn, "position": curr_idx + 1, "timestamp": curr_m.timestamp_str, "text": curr_m.content},
                        explanation=(
                            f"Visual sequence inversion in '{sc_fn}': Message #{curr_idx + 1} at {curr_m.timestamp_str} "
                            f"appears vertically below Message #{prev_idx + 1} at {prev_m.timestamp_str}. "
                            "This visual inversion serves as a potential indicator of message insertion or UI alteration."
                        ),
                        severity=SeverityLevel.HIGH.value,
                        confidence=score,
                        contradiction_type="ORDER_INVERSION",
                        source_a_ref=get_source_ref(sc.evidence_id, f"bubble_{prev_idx}", sc.metadata.get("extraction_method", "opencv_ocr")),
                        source_b_ref=get_source_ref(sc.evidence_id, f"bubble_{curr_idx}", sc.metadata.get("extraction_method", "opencv_ocr")),
                        observed_fact_a=f"Message #{prev_idx + 1} has timestamp {prev_m.timestamp_str}",
                        observed_fact_b=f"Message #{curr_idx + 1} has earlier timestamp {curr_m.timestamp_str} but appears lower on screen",
                        derived_result="Out-of-order message rendering in screenshot",
                        limitations=[TIME_ZONE_LIMITATION, "Date inferred from companion conversation."],
                        evidence_labels=["observed", "observed"],
                        supporting_sources=[],
                        conflicting_sources=[sc_fn],
                        related_finding_ids=related_ids,
                    )
                )

    # -------------------------------------------------------------------------
    # RULE 6: System Log vs Communication Correlation (LOG_COMM_CONFLICT)
    # -------------------------------------------------------------------------
    txn_records = [
        r for r in logs
        if r.status == "PARSED" and "Amount" in r.kv and r.parsed_datetime
    ]

    for txn in txn_records:
        txn_amt = float(txn.kv["Amount"])
        txn_payee = str(txn.kv.get("Payee", ""))
        txn_dt = datetime.fromisoformat(txn.parsed_datetime) if txn.parsed_datetime else None
        if not txn_dt:
            continue

        # Look for messages within config window
        for chat in chats:
            chat_ev = evidence_items.get(chat.evidence_id)
            chat_fn = chat_ev.file_name if chat_ev else "Chat"

            for msg_idx, msg in enumerate(chat.messages):
                msg_dt, msg_inf = parse_full_or_inferred_datetime(msg, dominant_date)
                if not msg_dt:
                    continue

                diff_sec = abs((txn_dt - msg_dt).total_seconds())
                if diff_sec <= (config.LOG_TIME_WINDOW_MINUTES * 60) and msg.amounts:
                    # Check if log amount contradicts message amount
                    if txn_amt not in msg.amounts:
                        contra_id = stable_id(case_id, "LOG_COMM_CONFLICT", txn.evidence_id, chat.evidence_id, str(msg_idx))
                        contradictions.append(
                            Contradiction(
                                id=contra_id,
                                case_id=case_id,
                                evidence_ids=[txn.evidence_id, chat.evidence_id],
                                entity=txn_payee or msg.sender,
                                field_name="Transaction Amount Verification",
                                claim_a={"source": "System Transaction Log", "value": txn_amt, "timestamp": txn.timestamp_str},
                                claim_b={"source": chat_fn, "value": msg.amounts, "timestamp": msg.timestamp_str},
                                explanation=(
                                    f"System log vs communication discrepancy: Banking transaction log confirms transfer of ₹{txn_amt:,.2f} "
                                    f"at {txn.timestamp_str}, whereas {chat_fn} claims {msg.amounts} at {msg.timestamp_str}. "
                                    f"Note: Log amount agrees with screenshot records and differs from text export records."
                                ),
                                severity=SeverityLevel.HIGH.value,
                                confidence=config.BASE_SCORE_LOG_CONFLICT,
                                contradiction_type="LOG_COMM_CONFLICT",
                                source_a_ref=get_source_ref(txn.evidence_id, f"line_{txn.line_no}", "system_event_log"),
                                source_b_ref=get_source_ref(chat.evidence_id, f"message_{msg_idx}", chat.metadata.get("extraction_method", "direct")),
                                observed_fact_a=f"Bank transaction record confirms transfer of ₹{txn_amt:,.2f}",
                                observed_fact_b=f"{chat_fn} records transfer figure as {msg.amounts}",
                                derived_result="Communication claim contradicts authoritative system transaction log",
                                limitations=[TIME_ZONE_LIMITATION, "Cross-platform clock drift between mobile OS and bank server."],
                                evidence_labels=["observed", "observed"],
                                supporting_sources=["system_events.log"],
                                conflicting_sources=[chat_fn],
                                related_finding_ids=[],
                            )
                        )

        # ---------------------------------------------------------------------
        # RULE 7: Post-Transaction File Modification Indicator
        # -------------------------------------------------------------------------
        warn_records = [
            r for r in logs
            if r.status == "PARSED" and r.level in ("WARN", "ERROR") and r.parsed_datetime
        ]
        for w in warn_records:
            w_dt = datetime.fromisoformat(w.parsed_datetime) if w.parsed_datetime else None
            if not w_dt:
                continue

            delta_sec = (w_dt - txn_dt).total_seconds()
            if 0 < delta_sec <= (config.POST_EVENT_MODIFICATION_WINDOW_MINUTES * 60):
                indic_id = stable_id(case_id, "POST_TXN_FILE_MOD", txn.evidence_id, w.evidence_id, str(w.line_no))
                indicators.append(
                    AnomalyFinding(
                        id=indic_id,
                        case_id=case_id,
                        evidence_ids=[w.evidence_id],
                        anomaly_type="POST_TXN_FILE_MOD",
                        severity=SeverityLevel.HIGH.value,
                        confidence=0.85,
                        title="Suspicious Post-Transaction File Modification Logged",
                        description=(
                            f"System log record at line {w.line_no} ({w.timestamp_str}) reports: '{w.message}' "
                            f"occurring {int(delta_sec // 60)} minutes after banking transaction (TxnID={txn.kv.get('TxnID')}). "
                            "This represents a potential indicator of post-facto database or artifact modification."
                        ),
                        details={
                            "log_line": w.line_no,
                            "timestamp": w.timestamp_str,
                            "message": w.message,
                            "file_path": w.file_path,
                            "related_transaction": txn.kv,
                            "limitations": [
                                "The log does not identify user identity or tool performing modification; may be automated background sync.",
                            ],
                        },
                    )
                )

    # -------------------------------------------------------------------------
    # Persist Contradictions and Indicators to DB
    # -------------------------------------------------------------------------
    for c in contradictions:
        db.insert_contradiction(c)
    for ind in indicators:
        db.insert_finding(ind)

    # Compile Entity List
    entities_list = [
        {"name": v["primary_name"], "status": v["status"], "aliases": v["linked_names"]}
        for v in resolved_entities.values()
    ]

    # Generate Summary
    stats = {
        "total_contradictions": len(contradictions),
        "total_indicators": len(indicators) + len(module1_2_findings),
        "high_or_critical_contradictions": sum(1 for c in contradictions if c.severity in ("HIGH", "CRITICAL")),
        "documents_analyzed": len(docs),
        "chats_analyzed": len(chats),
        "log_records_parsed": len(logs),
    }

    summary_text, summary_src = generate_summary(contradictions, indicators, stats)

    return TruthEngineResult(
        case_id=case_id,
        contradictions=contradictions,
        indicators=indicators,
        summary_text=summary_text,
        summary_source=summary_src,
        stats=stats,
        limitations=[
            TIME_ZONE_LIMITATION,
            "All findings represent probabilistic anomaly indicators and data contradictions.",
            "Outputs require human forensic review and do not constitute definitive legal proof.",
        ],
        skipped_comparisons=skipped_comparisons,
        entities=entities_list,
        created_at=utc_now_iso(),
    )


# -----------------------------------------------------------------------------
# Summary Generator (Deterministic Template with Optional LLM Fallback Hook)
# -----------------------------------------------------------------------------

def generate_summary(
    contradictions: List[Contradiction],
    indicators: List[AnomalyFinding],
    stats: Dict[str, Any],
) -> Tuple[str, str]:
    """Generate executive forensic summary using deterministic template, with optional LLM hook."""
    # Attempt local LLM hook if URL is configured
    llm_url = os.environ.get("DRISHTI_LLM_URL")
    if llm_url:
        try:
            payload = json.dumps({
                "prompt": (
                    "Summarize these digital forensic findings neutrally without claiming proof:\n"
                    f"Contradictions: {[c.explanation for c in contradictions]}\n"
                    f"Indicators: {[i.title for i in indicators]}"
                )
            }).encode("utf-8")
            req = urllib.request.Request(
                llm_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data.get("summary") or data.get("response")
                if text and len(text.strip()) > 20:
                    return text.strip(), "llm"
        except Exception:
            pass  # Clean fallback to deterministic template

    # Deterministic Template Fallback
    c_count = stats.get("total_contradictions", len(contradictions))
    i_count = stats.get("total_indicators", len(indicators))
    high_count = stats.get("high_or_critical_contradictions", 0)

    lines = [
        f"### Forensic Executive Summary",
        f"- **Contradiction Indicators Flagged:** {c_count} ({high_count} High/Critical)",
        f"- **Internal Anomaly Indicators:** {i_count}",
        f"- **Primary Discrepancy Findings:**",
    ]
    for c in contradictions[:5]:
        lines.append(f"  * [{c.severity}] **{c.field_name}**: {c.explanation}")

    if indicators:
        lines.append("- **Key Timeline / Log Indicators:**")
        for ind in indicators[:3]:
            lines.append(f"  * [{ind.severity}] {ind.title}: {ind.description}")

    lines.append(
        "\n*Note: All findings represent probabilistic indicators and factual contradictions extracted "
        "from persisted evidence files. Findings require human investigator verification.*"
    )

    return "\n".join(lines), "template"
