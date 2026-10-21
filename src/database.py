"""SQLite storage manager for case metadata, evidence provenance, audit trails, and structured forensic findings."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.models import (
    AnomalyFinding,
    CaseSummary,
    ChatMessage,
    Contradiction,
    EvidenceItem,
    EventLogRecord,
    ExtractedChat,
    ExtractedDocument,
    TimelineEvent,
    utc_now_iso,
)


ANALYSIS_MODULES = ("document", "screenshot", "truth_engine")


class DatabaseManager:
    """Manages forensic state and tamper-evident audit logs using SQLite."""

    def __init__(self, db_path: str | Path = "data/drishti_sutra.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
        self._migrate_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE'
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    evidence_id TEXT,
                    action TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    case_id TEXT NOT NULL,
                    module TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'not_run',
                    started_at TEXT,
                    completed_at TEXT,
                    evidence_fingerprint TEXT,
                    PRIMARY KEY (case_id, module),
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS extracted_documents (
                    id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL UNIQUE,
                    doc_type TEXT NOT NULL,
                    text_content TEXT,
                    fields_json TEXT NOT NULL DEFAULT '{}',
                    qr_data_json TEXT,
                    raw_ocr_confidence REAL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS extracted_chats (
                    id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL UNIQUE,
                    platform TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    messages_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS contradictions (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    entity TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    claim_a_json TEXT NOT NULL,
                    claim_b_json TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'HIGH',
                    confidence REAL NOT NULL DEFAULT 0.9,
                    created_at TEXT NOT NULL,
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS timeline_events (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    evidence_id TEXT,
                    timestamp_str TEXT NOT NULL,
                    parsed_datetime TEXT,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_reference TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    is_anomaly INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS event_log_records (
                    id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    line_no INTEGER NOT NULL,
                    timestamp_str TEXT NOT NULL,
                    parsed_datetime TEXT,
                    level TEXT,
                    component TEXT,
                    message TEXT,
                    kv_json TEXT NOT NULL DEFAULT '{}',
                    actor TEXT,
                    file_path TEXT,
                    raw_line TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PARSED',
                    skip_reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence(case_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_hash ON evidence(sha256_hash);
                CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_logs(case_id);
                CREATE INDEX IF NOT EXISTS idx_findings_case ON findings(case_id);
                CREATE INDEX IF NOT EXISTS idx_contradictions_case ON contradictions(case_id);
                CREATE INDEX IF NOT EXISTS idx_timeline_case ON timeline_events(case_id);
                CREATE INDEX IF NOT EXISTS idx_event_log_evidence ON event_log_records(evidence_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_runs_status ON analysis_runs(status);
                """
            )
            conn.commit()

    def _migrate_db(self) -> None:
        """Apply additive schema migrations so existing databases continue functioning."""
        with self._get_connection() as conn:
            cols = conn.execute("PRAGMA table_info(contradictions)").fetchall()
            col_names = [c["name"] for c in cols]
            if "extra_json" not in col_names:
                conn.execute("ALTER TABLE contradictions ADD COLUMN extra_json TEXT NOT NULL DEFAULT '{}' ")
            for case in conn.execute("SELECT case_id FROM cases").fetchall():
                for module in ANALYSIS_MODULES:
                    conn.execute(
                        "INSERT OR IGNORE INTO analysis_runs (case_id, module, status) VALUES (?, ?, 'not_run')",
                        (case["case_id"], module),
                    )
            conn.commit()

    def create_case(self, case_id: str, title: str, description: str = "") -> None:
        """Create a case and its module status rows, preserving existing cases."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO cases (case_id, title, description, created_at, status)
                VALUES (?, ?, ?, ?, 'ACTIVE')
                """,
                (case_id, title, description, utc_now_iso()),
            )
            created = cursor.rowcount == 1
            if created:
                conn.executemany(
                    "INSERT INTO analysis_runs (case_id, module, status) VALUES (?, ?, 'not_run')",
                    [(case_id, module) for module in ANALYSIS_MODULES],
                )
            conn.commit()
        if created:
            self.log_audit(case_id=case_id, action="CASE_CREATED", details={"title": title, "description": description})

    def log_audit(
        self,
        case_id: str,
        action: str,
        evidence_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append an immutable audit entry."""
        log_id = str(uuid4())
        details_str = json.dumps(details or {})
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (id, case_id, evidence_id, action, timestamp, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (log_id, case_id, evidence_id, action, utc_now_iso(), details_str),
            )
            conn.commit()
        return log_id

    def insert_evidence(self, item: EvidenceItem) -> None:
        """Insert a new evidence item with provenance tracking."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence (
                    id, case_id, file_path, file_name, file_type,
                    sha256_hash, file_size_bytes, ingested_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.case_id,
                    item.file_path,
                    item.file_name,
                    item.file_type,
                    item.sha256_hash,
                    item.file_size_bytes,
                    item.ingested_at,
                    json.dumps(item.metadata),
                ),
            )
            conn.commit()
        details = {"file_name": item.file_name, "sha256_hash": item.sha256_hash, "file_type": item.file_type}
        self.log_audit(case_id=item.case_id, evidence_id=item.id, action="EVIDENCE_UPLOADED", details=details)
        self.log_audit(case_id=item.case_id, evidence_id=item.id, action="EVIDENCE_INGESTED", details=details)
        self.mark_stale_analysis_runs(item.case_id)

    def find_evidence_by_hash(self, case_id: str, sha256: str) -> Optional[EvidenceItem]:
        """Find an existing evidence item by case + SHA-256 hash for deduplication."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM evidence WHERE case_id = ? AND sha256_hash = ?",
                (case_id, sha256),
            ).fetchone()
            if not row:
                return None
            return EvidenceItem(
                id=row["id"],
                case_id=row["case_id"],
                file_path=row["file_path"],
                file_name=row["file_name"],
                file_type=row["file_type"],
                sha256_hash=row["sha256_hash"],
                file_size_bytes=row["file_size_bytes"],
                ingested_at=row["ingested_at"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )

    def update_evidence_location(self, item: EvidenceItem) -> None:
        """Refresh source location without replacing linked analysis records."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE evidence SET file_path = ?, file_name = ?, file_type = ?, metadata_json = ? WHERE id = ?",
                (item.file_path, item.file_name, item.file_type, json.dumps(item.metadata), item.id),
            )
            conn.commit()
        self.mark_stale_analysis_runs(item.case_id)

    def list_evidence(self, case_id: str) -> List[EvidenceItem]:
        """Fetch all evidence items registered for a case."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM evidence WHERE case_id = ? ORDER BY ingested_at ASC", (case_id,))
            rows = cursor.fetchall()
            return [
                EvidenceItem(
                    id=row["id"],
                    case_id=row["case_id"],
                    file_path=row["file_path"],
                    file_name=row["file_name"],
                    file_type=row["file_type"],
                    sha256_hash=row["sha256_hash"],
                    file_size_bytes=row["file_size_bytes"],
                    ingested_at=row["ingested_at"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
                for row in rows
            ]

    def _row_to_extracted_document(self, row: sqlite3.Row) -> ExtractedDocument:
        return ExtractedDocument(
            id=row["id"],
            evidence_id=row["evidence_id"],
            doc_type=row["doc_type"],
            text_content=row["text_content"] or "",
            fields=json.loads(row["fields_json"] or "{}"),
            qr_data=json.loads(row["qr_data_json"]) if row["qr_data_json"] else None,
            raw_ocr_confidence=row["raw_ocr_confidence"] or 1.0,
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _row_to_extracted_chat(self, row: sqlite3.Row) -> ExtractedChat:
        raw_msgs = json.loads(row["messages_json"] or "[]")
        return ExtractedChat(
            id=row["id"],
            evidence_id=row["evidence_id"],
            platform=row["platform"],
            source_type=row["source_type"],
            messages=[ChatMessage(**m) for m in raw_msgs],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def insert_extracted_document(self, doc: ExtractedDocument) -> None:
        """Store extracted document content and key-value fields."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO extracted_documents (
                    id, evidence_id, doc_type, text_content, fields_json,
                    qr_data_json, raw_ocr_confidence, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.evidence_id,
                    doc.doc_type,
                    doc.text_content,
                    json.dumps(doc.fields),
                    json.dumps(doc.qr_data) if doc.qr_data else None,
                    doc.raw_ocr_confidence,
                    json.dumps(doc.metadata),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def get_extracted_document(self, evidence_id: str) -> Optional[ExtractedDocument]:
        """Retrieve extracted document data for a specific evidence item."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM extracted_documents WHERE evidence_id = ?", (evidence_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_extracted_document(row)

    def list_extracted_documents(self, case_id: str) -> List[ExtractedDocument]:
        """Fetch all ExtractedDocuments for a case via join through evidence."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT ed.* FROM extracted_documents ed
                JOIN evidence e ON ed.evidence_id = e.id
                WHERE e.case_id = ?
                """,
                (case_id,),
            )
            return [self._row_to_extracted_document(row) for row in cursor.fetchall()]

    def insert_extracted_chat(self, chat: ExtractedChat) -> None:
        """Store parsed chat thread messages."""
        messages_data = [msg.model_dump() for msg in chat.messages]
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO extracted_chats (
                    id, evidence_id, platform, source_type, messages_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat.id,
                    chat.evidence_id,
                    chat.platform,
                    chat.source_type,
                    json.dumps(messages_data),
                    json.dumps(chat.metadata),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def get_extracted_chat(self, evidence_id: str) -> Optional[ExtractedChat]:
        """Retrieve parsed chat messages for a specific evidence item."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM extracted_chats WHERE evidence_id = ?", (evidence_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_extracted_chat(row)

    def list_extracted_chats(self, case_id: str) -> List[ExtractedChat]:
        """Fetch all ExtractedChats for a case via join through evidence."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT ec.* FROM extracted_chats ec
                JOIN evidence e ON ec.evidence_id = e.id
                WHERE e.case_id = ?
                """,
                (case_id,),
            )
            return [self._row_to_extracted_chat(row) for row in cursor.fetchall()]

    def insert_finding(self, finding: AnomalyFinding) -> None:
        """Record an anomaly finding."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO findings (
                    id, case_id, evidence_ids_json, anomaly_type, severity,
                    confidence, title, description, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding.id,
                    finding.case_id,
                    json.dumps(finding.evidence_ids),
                    finding.anomaly_type,
                    finding.severity,
                    finding.confidence,
                    finding.title,
                    finding.description,
                    json.dumps(finding.details),
                    finding.created_at,
                ),
            )
            conn.commit()
        self.log_audit(
            finding.case_id,
            "FINDING_GENERATED",
            evidence_id=finding.evidence_ids[0] if finding.evidence_ids else None,
            details={"finding_id": finding.id, "anomaly_type": finding.anomaly_type, "title": finding.title},
        )

    def list_findings(self, case_id: str) -> List[AnomalyFinding]:
        """List all findings for a given case."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM findings WHERE case_id = ? ORDER BY confidence DESC", (case_id,))
            rows = cursor.fetchall()
            return [
                AnomalyFinding(
                    id=row["id"],
                    case_id=row["case_id"],
                    evidence_ids=json.loads(row["evidence_ids_json"] or "[]"),
                    anomaly_type=row["anomaly_type"],
                    severity=row["severity"],
                    confidence=row["confidence"],
                    title=row["title"],
                    description=row["description"],
                    details=json.loads(row["details_json"] or "{}"),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def insert_contradiction(self, contradiction: Contradiction) -> None:
        """Record a cross-source contradiction including extended Truth Engine metadata."""
        extra = {
            "contradiction_type": contradiction.contradiction_type,
            "source_a_ref": contradiction.source_a_ref,
            "source_b_ref": contradiction.source_b_ref,
            "observed_fact_a": contradiction.observed_fact_a,
            "observed_fact_b": contradiction.observed_fact_b,
            "derived_result": contradiction.derived_result,
            "limitations": contradiction.limitations,
            "evidence_labels": contradiction.evidence_labels,
            "supporting_sources": contradiction.supporting_sources,
            "conflicting_sources": contradiction.conflicting_sources,
            "related_finding_ids": contradiction.related_finding_ids,
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO contradictions (
                    id, case_id, evidence_ids_json, entity, field_name,
                    claim_a_json, claim_b_json, explanation, severity, confidence, created_at, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contradiction.id,
                    contradiction.case_id,
                    json.dumps(contradiction.evidence_ids),
                    contradiction.entity,
                    contradiction.field_name,
                    json.dumps(contradiction.claim_a),
                    json.dumps(contradiction.claim_b),
                    contradiction.explanation,
                    contradiction.severity,
                    contradiction.confidence,
                    contradiction.created_at,
                    json.dumps(extra),
                ),
            )
            conn.commit()
        self.log_audit(
            contradiction.case_id,
            "CONTRADICTION_GENERATED",
            evidence_id=contradiction.evidence_ids[0] if contradiction.evidence_ids else None,
            details={"contradiction_id": contradiction.id, "contradiction_type": contradiction.contradiction_type,
                     "field_name": contradiction.field_name},
        )

    def list_contradictions(self, case_id: str) -> List[Contradiction]:
        """List all detected cross-source contradictions including extended metadata."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM contradictions WHERE case_id = ?", (case_id,))
            rows = cursor.fetchall()
            result = []
            for row in rows:
                extra = json.loads(row["extra_json"]) if ("extra_json" in row.keys() and row["extra_json"]) else {}
                result.append(
                    Contradiction(
                        id=row["id"],
                        case_id=row["case_id"],
                        evidence_ids=json.loads(row["evidence_ids_json"] or "[]"),
                        entity=row["entity"],
                        field_name=row["field_name"],
                        claim_a=json.loads(row["claim_a_json"]),
                        claim_b=json.loads(row["claim_b_json"]),
                        explanation=row["explanation"],
                        severity=row["severity"],
                        confidence=row["confidence"],
                        created_at=row["created_at"],
                        contradiction_type=extra.get("contradiction_type"),
                        source_a_ref=extra.get("source_a_ref"),
                        source_b_ref=extra.get("source_b_ref"),
                        observed_fact_a=extra.get("observed_fact_a"),
                        observed_fact_b=extra.get("observed_fact_b"),
                        derived_result=extra.get("derived_result"),
                        limitations=extra.get("limitations", []),
                        evidence_labels=extra.get("evidence_labels", []),
                        supporting_sources=extra.get("supporting_sources", []),
                        conflicting_sources=extra.get("conflicting_sources", []),
                        related_finding_ids=extra.get("related_finding_ids", []),
                    )
                )
            return result

    def insert_timeline_event(self, event: TimelineEvent) -> None:
        """Insert a single timeline event."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO timeline_events (
                    id, case_id, evidence_id, timestamp_str, parsed_datetime,
                    event_type, actor, description, source_reference,
                    confidence, is_anomaly, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.case_id,
                    event.evidence_id,
                    event.timestamp_str,
                    event.parsed_datetime,
                    event.event_type,
                    event.actor,
                    event.description,
                    event.source_reference,
                    event.confidence,
                    1 if event.is_anomaly else 0,
                    json.dumps(event.metadata),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list_timeline_events(self, case_id: str) -> List[TimelineEvent]:
        """Fetch all chronologically ordered timeline events."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM timeline_events
                WHERE case_id = ?
                ORDER BY parsed_datetime ASC, timestamp_str ASC
                """,
                (case_id,),
            )
            rows = cursor.fetchall()
            return [
                TimelineEvent(
                    id=row["id"],
                    case_id=row["case_id"],
                    evidence_id=row["evidence_id"],
                    timestamp_str=row["timestamp_str"],
                    parsed_datetime=row["parsed_datetime"],
                    event_type=row["event_type"],
                    actor=row["actor"],
                    description=row["description"],
                    source_reference=row["source_reference"],
                    confidence=row["confidence"],
                    is_anomaly=bool(row["is_anomaly"]),
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
                for row in rows
            ]

    def insert_event_log_record(self, record: EventLogRecord) -> None:
        """Persist a single parsed event log record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO event_log_records (
                    id, evidence_id, line_no, timestamp_str, parsed_datetime,
                    level, component, message, kv_json, actor, file_path,
                    raw_line, status, skip_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.evidence_id,
                    record.line_no,
                    record.timestamp_str,
                    record.parsed_datetime,
                    record.level,
                    record.component,
                    record.message,
                    json.dumps(record.kv),
                    record.actor,
                    record.file_path,
                    record.raw_line,
                    record.status,
                    record.skip_reason,
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list_event_log_records(self, evidence_id: str) -> List[EventLogRecord]:
        """Fetch all event log records for a given evidence item."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM event_log_records WHERE evidence_id = ? ORDER BY line_no ASC",
                (evidence_id,),
            ).fetchall()
            return [
                EventLogRecord(
                    id=r["id"],
                    evidence_id=r["evidence_id"],
                    line_no=r["line_no"],
                    timestamp_str=r["timestamp_str"],
                    parsed_datetime=r["parsed_datetime"],
                    level=r["level"] or "",
                    component=r["component"] or "",
                    message=r["message"] or "",
                    kv=json.loads(r["kv_json"] or "{}"),
                    actor=r["actor"] or "",
                    file_path=r["file_path"] or "",
                    raw_line=r["raw_line"],
                    status=r["status"],
                    skip_reason=r["skip_reason"] or "",
                )
                for r in rows
            ]

    def list_event_log_records_for_case(self, case_id: str) -> List[EventLogRecord]:
        """Fetch all event log records for a case via join through evidence."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT elr.* FROM event_log_records elr
                JOIN evidence e ON elr.evidence_id = e.id
                WHERE e.case_id = ?
                ORDER BY elr.parsed_datetime ASC, elr.line_no ASC
                """,
                (case_id,),
            ).fetchall()
            return [
                EventLogRecord(
                    id=r["id"],
                    evidence_id=r["evidence_id"],
                    line_no=r["line_no"],
                    timestamp_str=r["timestamp_str"],
                    parsed_datetime=r["parsed_datetime"],
                    level=r["level"] or "",
                    component=r["component"] or "",
                    message=r["message"] or "",
                    kv=json.loads(r["kv_json"] or "{}"),
                    actor=r["actor"] or "",
                    file_path=r["file_path"] or "",
                    raw_line=r["raw_line"],
                    status=r["status"],
                    skip_reason=r["skip_reason"] or "",
                )
                for r in rows
            ]

    def clear_analysis(self, case_id: str) -> None:
        """Delete analysis results for a case while preserving evidence rows and audit logs.

        Deletes: findings, contradictions, timeline_events, event_log_records,
        extracted_documents, and extracted_chats.
        """
        with self._get_connection() as conn:
            evidence_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM evidence WHERE case_id = ?", (case_id,)
                ).fetchall()
            ]
            if evidence_ids:
                placeholders = ",".join("?" * len(evidence_ids))
                conn.execute(
                    f"DELETE FROM event_log_records WHERE evidence_id IN ({placeholders})",
                    evidence_ids,
                )
                conn.execute(
                    f"DELETE FROM extracted_documents WHERE evidence_id IN ({placeholders})",
                    evidence_ids,
                )
                conn.execute(
                    f"DELETE FROM extracted_chats WHERE evidence_id IN ({placeholders})",
                    evidence_ids,
                )
            conn.execute("DELETE FROM findings WHERE case_id = ?", (case_id,))
            conn.execute("DELETE FROM contradictions WHERE case_id = ?", (case_id,))
            conn.execute("DELETE FROM timeline_events WHERE case_id = ?", (case_id,))
            conn.commit()
        self.log_audit(
            case_id=case_id,
            action="ANALYSIS_CLEARED",
            details={"note": "Findings, contradictions, timeline, event_logs, extracted docs/chats cleared"},
        )

    def get_case_summary(self, case_id: str) -> Optional[CaseSummary]:
        """Build an aggregated summary for the case."""
        with self._get_connection() as conn:
            c_row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if not c_row:
                return None

            ev_rows = conn.execute("SELECT file_type, count(*) as c FROM evidence WHERE case_id = ? GROUP BY file_type", (case_id,)).fetchall()
            type_counts = {r["file_type"]: r["c"] for r in ev_rows}
            total_ev = sum(type_counts.values())

            anom_count = conn.execute("SELECT count(*) as c FROM findings WHERE case_id = ?", (case_id,)).fetchone()["c"]
            contra_count = conn.execute("SELECT count(*) as c FROM contradictions WHERE case_id = ?", (case_id,)).fetchone()["c"]
            tl_count = conn.execute("SELECT count(*) as c FROM timeline_events WHERE case_id = ?", (case_id,)).fetchone()["c"]

            return CaseSummary(
                case_id=case_id,
                title=c_row["title"],
                description=c_row["description"] or "",
                created_at=c_row["created_at"],
                total_evidence_count=total_ev,
                document_count=type_counts.get("kyc_doc", 0) + type_counts.get("generic_document", 0),
                chat_count=type_counts.get("chat_screenshot", 0) + type_counts.get("chat_export", 0),
                log_count=type_counts.get("system_log", 0),
                anomaly_count=anom_count,
                contradiction_count=contra_count,
                timeline_event_count=tl_count,
                status=c_row["status"],
            )

    def get_audit_trail(self, case_id: str) -> List[Dict[str, Any]]:
        """Retrieve audit log sequence for chain of custody display."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_logs WHERE case_id = ? ORDER BY timestamp ASC",
                (case_id,),
            )
            return [
                {
                    "id": r["id"],
                    "case_id": r["case_id"],
                    "evidence_id": r["evidence_id"],
                    "action": r["action"],
                    "timestamp": r["timestamp"],
                    "details": json.loads(r["details_json"] or "{}"),
                }
                for r in cursor.fetchall()
            ]

    def evidence_fingerprint(self, case_id: str) -> str:
        """Return a stable digest for the complete evidence set."""
        records = [
            [item.id, item.sha256_hash, item.file_name, item.file_type, item.file_size_bytes]
            for item in self.list_evidence(case_id)
        ]
        payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def mark_stale_analysis_runs(self, case_id: str) -> None:
        fingerprint = self.evidence_fingerprint(case_id)
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE analysis_runs SET status='stale'
                   WHERE case_id=? AND status IN ('running','completed','partial')
                   AND evidence_fingerprint IS NOT NULL AND evidence_fingerprint != ?""",
                (case_id, fingerprint),
            )
            conn.commit()

    def get_analysis_runs(self, case_id: str) -> Dict[str, Dict[str, Any]]:
        self.mark_stale_analysis_runs(case_id)
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM analysis_runs WHERE case_id=?", (case_id,)).fetchall()
            return {row["module"]: dict(row) for row in rows}

    def analysis_is_current(self, case_id: str, module: str) -> bool:
        if module not in ANALYSIS_MODULES:
            raise ValueError(f"Unknown analysis module: {module}")
        run = self.get_analysis_runs(case_id).get(module)
        return bool(
            run
            and run["status"] in {"completed", "partial"}
            and run["evidence_fingerprint"] == self.evidence_fingerprint(case_id)
        )

    def start_analysis_run(self, case_id: str, module: str, fingerprint: Optional[str] = None) -> str:
        if module not in ANALYSIS_MODULES:
            raise ValueError(f"Unknown analysis module: {module}")
        started = utc_now_iso()
        fingerprint = fingerprint or self.evidence_fingerprint(case_id)
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO analysis_runs (case_id,module,status,started_at,completed_at,evidence_fingerprint)
                   VALUES (?,?,'running',?,NULL,?)
                   ON CONFLICT(case_id,module) DO UPDATE SET status='running',started_at=excluded.started_at,
                   completed_at=NULL,evidence_fingerprint=excluded.evidence_fingerprint""",
                (case_id, module, started, fingerprint),
            )
            conn.commit()
        self.log_audit(case_id, "ANALYSIS_STARTED", details={"module": module, "evidence_fingerprint": fingerprint})
        return started

    def finish_analysis_run(self, case_id: str, module: str, status: str, fingerprint: str) -> str:
        if status not in {"completed", "partial", "failed"}:
            raise ValueError(f"Invalid final analysis status: {status}")
        completed = utc_now_iso()
        saved_status = "stale" if fingerprint != self.evidence_fingerprint(case_id) else status
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE analysis_runs SET status=?,completed_at=?,evidence_fingerprint=?
                   WHERE case_id=? AND module=?""",
                (saved_status, completed, fingerprint, case_id, module),
            )
            conn.commit()
        action = "ANALYSIS_FAILED" if status == "failed" else "ANALYSIS_COMPLETED"
        self.log_audit(
            case_id,
            action,
            details={"module": module, "status": saved_status, "evidence_fingerprint": fingerprint},
        )
        return saved_status

    def invalidate_truth_engine(self, case_id: str, reason: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """UPDATE analysis_runs SET status='stale'
                   WHERE case_id=? AND module='truth_engine' AND status IN ('running','completed','partial')""",
                (case_id,),
            )
            conn.commit()
        if cursor.rowcount:
            self.log_audit(
                case_id,
                "ANALYSIS_INVALIDATED",
                details={"module": "truth_engine", "reason": reason},
            )

    def clear_evidence_results(
        self,
        case_id: str,
        evidence_ids: List[str],
        documents: bool = False,
        chats: bool = False,
        event_logs: bool = False,
    ) -> None:
        """Remove derived records for selected evidence before reprocessing that module."""
        if not evidence_ids:
            return
        placeholders = ",".join("?" for _ in evidence_ids)
        selected = set(evidence_ids)
        with self._get_connection() as conn:
            if documents:
                conn.execute(f"DELETE FROM extracted_documents WHERE evidence_id IN ({placeholders})", evidence_ids)
            if chats:
                conn.execute(f"DELETE FROM extracted_chats WHERE evidence_id IN ({placeholders})", evidence_ids)
            if event_logs:
                conn.execute(f"DELETE FROM event_log_records WHERE evidence_id IN ({placeholders})", evidence_ids)
            finding_rows = conn.execute("SELECT id,evidence_ids_json FROM findings WHERE case_id=?", (case_id,)).fetchall()
            remove_ids = [
                row["id"] for row in finding_rows
                if set(json.loads(row["evidence_ids_json"] or "[]")) & selected
            ]
            if remove_ids:
                conn.executemany("DELETE FROM findings WHERE id=?", [(item,) for item in remove_ids])
            conn.commit()

    def clear_truth_results(self, case_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM contradictions WHERE case_id=?", (case_id,))
            conn.execute("DELETE FROM timeline_events WHERE case_id=?", (case_id,))
            conn.execute("DELETE FROM findings WHERE case_id=? AND anomaly_type='POST_TXN_FILE_MOD'", (case_id,))
            conn.commit()

    def delete_evidence(self, case_id: str, evidence_id: str) -> bool:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT file_name FROM evidence WHERE id=? AND case_id=?", (evidence_id, case_id)
            ).fetchone()
        if not row:
            return False
        self.clear_evidence_results(case_id, [evidence_id], documents=True, chats=True, event_logs=True)
        with self._get_connection() as conn:
            conn.execute("DELETE FROM evidence WHERE id=? AND case_id=?", (evidence_id, case_id))
            conn.commit()
        self.log_audit(case_id, "EVIDENCE_REMOVED", evidence_id=evidence_id, details={"file_name": row["file_name"]})
        self.mark_stale_analysis_runs(case_id)
        return True

    def list_cases(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT c.case_id,c.title,c.description,c.created_at,c.status,
                   (SELECT COUNT(*) FROM evidence e WHERE e.case_id=c.case_id) AS evidence_count,
                   COALESCE((SELECT MAX(a.timestamp) FROM audit_logs a WHERE a.case_id=c.case_id),c.created_at) AS last_activity
                   FROM cases c ORDER BY last_activity DESC,c.created_at DESC"""
            ).fetchall()
            return [dict(row) for row in rows]

    def get_home_metrics(self) -> Dict[str, int]:
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        with self._get_connection() as conn:
            total_cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            evidence_files = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
            recent_activity = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE timestamp>=?", (cutoff,)).fetchone()[0]
        completed = sum(
            1
            for case in self.list_cases()
            for run in self.get_analysis_runs(case["case_id"]).values()
            if run["status"] in {"completed", "partial"}
        )
        return {
            "total_cases": total_cases,
            "evidence_files": evidence_files,
            "completed_analyses": completed,
            "recent_activity": recent_activity,
        }

    def search_audit_history(
        self, search: str = "", case_id: Optional[str] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if case_id:
            clauses.append("a.case_id=?")
            params.append(case_id)
        if search.strip():
            pattern = f"%{search.strip()}%"
            clauses.append("(a.action LIKE ? OR a.details_json LIKE ? OR a.evidence_id LIKE ? OR a.case_id LIKE ? OR c.title LIKE ?)")
            params.extend([pattern] * 5)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(int(limit), 5000)))
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT a.*,c.title AS case_title FROM audit_logs a
                   LEFT JOIN cases c ON c.case_id=a.case_id""" + where +
                " ORDER BY a.timestamp DESC LIMIT ?",
                params,
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "case_id": row["case_id"],
                    "case_title": row["case_title"] or "",
                    "evidence_id": row["evidence_id"],
                    "action": row["action"],
                    "timestamp": row["timestamp"],
                    "details": json.loads(row["details_json"] or "{}"),
                }
                for row in rows
            ]
