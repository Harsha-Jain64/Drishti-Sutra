"""Ingestion engine: computes SHA-256 hashes, classifies forensic artifacts, and logs provenance."""

import hashlib
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from src.database import DatabaseManager
from src.models import EvidenceClassification, EvidenceItem, stable_id, utc_now_iso


def compute_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Compute cryptographic SHA-256 hash using streaming chunks for memory safety."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def classify_file(file_path: str | Path) -> str:
    """Classify the evidence item based on file extension, heuristics, and content inspection."""
    path = Path(file_path)
    ext = path.suffix.lower()
    name_lower = path.name.lower()

    # Image classifications
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        if any(k in name_lower for k in ["pan", "id", "aadhaar", "kyc", "card", "passport", "license"]):
            return EvidenceClassification.KYC_DOC.value
        if any(k in name_lower for k in ["chat", "screenshot", "whatsapp", "signal", "telegram"]):
            return EvidenceClassification.CHAT_SCREENSHOT.value
        return EvidenceClassification.GENERIC_IMAGE.value

    # PDF classification
    if ext == ".pdf":
        if any(k in name_lower for k in ["pan", "id", "aadhaar", "kyc", "statement"]):
            return EvidenceClassification.KYC_DOC.value
        return EvidenceClassification.GENERIC_PDF.value

    # Log classification
    if ext in {".log", ".syslog"}:
        return EvidenceClassification.SYSTEM_LOG.value

    # Text / Chat Export classification
    if ext in {".txt", ".csv"}:
        # Quick content heuristic: check first 2KB for chat timestamp patterns
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                snippet = f.read(2048)
                chat_pattern = re.compile(
                    r"(\[\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\]|"
                    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s+\d{1,2}:\d{2}\s*-\s*)"
                )
                if chat_pattern.search(snippet):
                    return EvidenceClassification.CHAT_EXPORT.value
                # Check for syslog / event log format
                log_pattern = re.compile(r"(\b(?:INFO|DEBUG|WARN|WARNING|ERROR|CRITICAL)\b|\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b)")
                if log_pattern.search(snippet):
                    return EvidenceClassification.SYSTEM_LOG.value
        except Exception:
            pass
        return EvidenceClassification.GENERIC_DOCUMENT.value

    # Archive
    if ext in {".zip", ".tar", ".gz"}:
        return EvidenceClassification.ARCHIVE.value

    return EvidenceClassification.UNKNOWN.value


class EvidenceIngestionService:
    """Service to ingest individual files, folders, or archives into a forensic case."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def ingest_file(self, case_id: str, file_path: str | Path) -> EvidenceItem:
        """Ingest a single forensic file, compute SHA-256, and persist in SQLite with deduplication."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Target evidence file does not exist: {path}")

        sha256_hash = compute_sha256(path)

        # Deduplication: reuse existing item if identical content already recorded for this case
        existing = self.db.find_evidence_by_hash(case_id, sha256_hash)
        if existing is not None:
            existing.file_path = str(path)
            existing.file_name = path.name
            existing.file_type = classify_file(path)
            existing.metadata.update({
                "extension": path.suffix.lower(),
                "absolute_path": str(path),
            })
            self.db.update_evidence_location(existing)
            self.db.log_audit(
                case_id,
                "EVIDENCE_UPLOADED",
                evidence_id=existing.id,
                details={"file_name": path.name, "sha256_hash": sha256_hash, "file_type": existing.file_type, "deduplicated": True},
            )
            return existing

        file_size = path.stat().st_size
        file_type = classify_file(path)

        item = EvidenceItem(
            id=stable_id(case_id, sha256_hash),
            case_id=case_id,
            file_path=str(path),
            file_name=path.name,
            file_type=file_type,
            sha256_hash=sha256_hash,
            file_size_bytes=file_size,
            ingested_at=utc_now_iso(),
            metadata={
                "extension": path.suffix.lower(),
                "absolute_path": str(path),
            },
        )
        self.db.insert_evidence(item)
        return item

    def ingest_directory(
        self, case_id: str, dir_path: str | Path, recursive: bool = True
    ) -> List[EvidenceItem]:
        """Scan and ingest all evidence files inside a directory."""
        dir_p = Path(dir_path).resolve()
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Target directory does not exist: {dir_p}")

        ingested_items: List[EvidenceItem] = []
        iterator = dir_p.rglob("*") if recursive else dir_p.glob("*")

        for file_entry in sorted(iterator):
            if file_entry.is_file():
                # Skip manifest.json from being treated as raw evidence
                if file_entry.name.lower() == "manifest.json":
                    continue
                # Skip temporary / hidden files or generated forensic overlays
                if file_entry.name.startswith(".") or file_entry.name.startswith("ela_"):
                    continue
                if "forensic_artifacts" in file_entry.parts:
                    continue
                item = self.ingest_file(case_id, file_entry)
                ingested_items.append(item)

        return ingested_items

    def ingest_zip(
        self, case_id: str, zip_path: str | Path, extract_base_dir: Optional[str | Path] = None
    ) -> List[EvidenceItem]:
        """Extract a ZIP archive securely (preventing Zip Slip) and ingest its contents."""
        zpath = Path(zip_path).resolve()
        if not zpath.is_file():
            raise FileNotFoundError(f"Target ZIP does not exist: {zpath}")

        extract_dir = (
            Path(extract_base_dir) if extract_base_dir else zpath.parent / f"extracted_{zpath.stem}"
        )
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zpath, "r") as zip_ref:
            for member in zip_ref.infolist():
                # Prevent Zip-Slip vulnerability
                target_path = (extract_dir / member.filename).resolve()
                if not str(target_path).startswith(str(extract_dir.resolve())):
                    raise ValueError(f"Malicious zip archive entry detected (Zip Slip): {member.filename}")
                zip_ref.extract(member, extract_dir)

        # Log extraction event
        self.db.log_audit(
            case_id=case_id,
            action="ZIP_EXTRACTED",
            details={"zip_path": str(zpath), "extracted_to": str(extract_dir)},
        )

        return self.ingest_directory(case_id, extract_dir)
