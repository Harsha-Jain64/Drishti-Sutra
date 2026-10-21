"""Strict Pydantic v2 data models for Drishti-Sutra."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4, uuid5
from pydantic import BaseModel, Field

# Fixed deterministic namespace for stable UUID5 generation
NS_DRISHTI = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalyType(str, Enum):
    ELA_SUSPECT = "ELA_SUSPECT"
    QR_TEXT_MISMATCH = "QR_TEXT_MISMATCH"
    OUT_OF_ORDER_TIMESTAMP = "OUT_OF_ORDER_TIMESTAMP"
    CHAT_CONTENT_TAMPERING = "CHAT_CONTENT_TAMPERING"
    MISSING_RECORD = "MISSING_RECORD"
    METADATA_DISCREPANCY = "METADATA_DISCREPANCY"
    SPACING_IRREGULARITY = "SPACING_IRREGULARITY"
    OCR_QUALITY_DEGRADATION = "OCR_QUALITY_DEGRADATION"


class EvidenceClassification(str, Enum):
    KYC_DOC = "kyc_doc"
    CHAT_SCREENSHOT = "chat_screenshot"
    CHAT_EXPORT = "chat_export"
    SYSTEM_LOG = "system_log"
    GENERIC_IMAGE = "generic_image"
    GENERIC_PDF = "generic_pdf"
    GENERIC_DOCUMENT = "generic_document"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: str) -> str:
    """Generate a deterministic UUID5 from ordered string parts using a fixed namespace.

    Two runs with identical parts produce identical IDs, ensuring pipeline idempotency.
    """
    combined = "|".join(str(p) for p in parts)
    return str(uuid5(NS_DRISHTI, combined))


class EvidenceItem(BaseModel):
    """Represents a discrete forensic artifact ingested into the platform."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str
    file_path: str
    file_name: str
    file_type: str = EvidenceClassification.UNKNOWN.value
    sha256_hash: str
    file_size_bytes: int
    ingested_at: str = Field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """Structured data extracted from KYC/identity/official documents."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_id: str
    doc_type: str = "GENERIC"  # PAN_CARD, ID_CARD, PASSPORT, etc.
    text_content: str = ""
    fields: Dict[str, Any] = Field(default_factory=dict)  # name, dob, id_number, etc.
    qr_data: Optional[Dict[str, Any]] = None
    raw_ocr_confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A single message entry inside a chat conversation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str
    timestamp_str: str
    parsed_datetime: Optional[str] = None
    content: str
    amounts: List[float] = Field(default_factory=list)
    bounding_box: Optional[Dict[str, int]] = None  # {x, y, w, h} if from screenshot
    raw_text: Optional[str] = None


class ExtractedChat(BaseModel):
    """Structured representation of a parsed chat thread from screenshot or text export."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_id: str
    platform: str = "WHATSAPP"
    source_type: str = "SCREENSHOT"  # SCREENSHOT or EXPORT_TXT
    messages: List[ChatMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnomalyFinding(BaseModel):
    """Probabilistic anomaly detected within an individual evidence item."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str
    evidence_ids: List[str] = Field(default_factory=list)
    anomaly_type: str
    severity: str = SeverityLevel.MEDIUM.value
    confidence: float = Field(ge=0.0, le=1.0)
    title: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class Contradiction(BaseModel):
    """Cross-source factual contradiction between two or more evidence items."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str
    evidence_ids: List[str] = Field(default_factory=list)
    entity: str
    field_name: str
    claim_a: Dict[str, Any]  # {"source": str, "value": Any, "timestamp": Optional[str]}
    claim_b: Dict[str, Any]  # {"source": str, "value": Any, "timestamp": Optional[str]}
    explanation: str
    severity: str = SeverityLevel.HIGH.value
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=utc_now_iso)

    # Extended fields (Truth Engine metadata; all optional for backward compatibility)
    contradiction_type: Optional[str] = None
    source_a_ref: Optional[Dict[str, Any]] = None  # {evidence_id, file_name, locator, extraction_method}
    source_b_ref: Optional[Dict[str, Any]] = None
    observed_fact_a: Optional[str] = None
    observed_fact_b: Optional[str] = None
    derived_result: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    evidence_labels: List[str] = Field(default_factory=list)
    supporting_sources: List[str] = Field(default_factory=list)
    conflicting_sources: List[str] = Field(default_factory=list)
    related_finding_ids: List[str] = Field(default_factory=list)


class EventLogRecord(BaseModel):
    """A single parsed line from a structured system/device event log."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_id: str
    line_no: int
    timestamp_str: str
    parsed_datetime: Optional[str] = None
    level: str = ""
    component: str = ""
    message: str = ""
    kv: Dict[str, Any] = Field(default_factory=dict)
    actor: str = ""
    file_path: str = ""
    raw_line: str = ""
    status: str = "PARSED"  # PARSED or SKIPPED
    skip_reason: str = ""


class TimelineEvent(BaseModel):
    """Discrete, timestamped event unified across documents, chats, and system logs."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str
    evidence_id: Optional[str] = None
    timestamp_str: str
    parsed_datetime: Optional[str] = None
    event_type: str  # SYSTEM_LOG, CHAT_MESSAGE, TRANSACTION, FILE_ACCESS, etc.
    actor: str
    description: str
    source_reference: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_anomaly: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TruthEngineResult(BaseModel):
    """Output of the Truth Engine for a single case analysis run."""
    case_id: str
    contradictions: List[Contradiction] = Field(default_factory=list)
    indicators: List[AnomalyFinding] = Field(default_factory=list)
    summary_text: str = ""
    summary_source: str = "template"  # 'template' or 'llm'
    stats: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    skipped_comparisons: List[Dict[str, str]] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class FileProcessingStatus(BaseModel):
    """Per-file status record in a pipeline run."""
    file_name: str
    file_type: str
    status: str  # 'processed', 'skipped', 'failed'
    reason: str = ""
    extraction_method: str = ""
    duration_seconds: float = 0.0
    finding_count: int = 0


class PipelineReport(BaseModel):
    """Full report from a pipeline run over a case directory or ZIP."""
    case_id: str
    case_dir: str
    started_at: str
    completed_at: str = ""
    files: List[FileProcessingStatus] = Field(default_factory=list)
    truth_engine_result: Optional[TruthEngineResult] = None
    total_evidence: int = 0
    total_findings: int = 0
    total_contradictions: int = 0
    total_timeline_events: int = 0
    success: bool = True
    errors: List[str] = Field(default_factory=list)


class CaseSummary(BaseModel):
    """Case-level aggregation metrics and audit statistics."""
    case_id: str
    title: str
    description: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    total_evidence_count: int = 0
    document_count: int = 0
    chat_count: int = 0
    log_count: int = 0
    anomaly_count: int = 0
    contradiction_count: int = 0
    timeline_event_count: int = 0
    status: str = "ACTIVE"
