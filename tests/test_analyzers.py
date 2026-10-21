"""Automated test suite verifying Module 1 (Document Authenticator) and Module 2 (Screenshot Authenticator)."""

from pathlib import Path
import pytest

from src.database import DatabaseManager
from src.ingestion import EvidenceIngestionService
from src.analyzers.document_analyzer import DocumentAuthenticator
from src.analyzers.screenshot_analyzer import ScreenshotAuthenticator
from src.analyzers.chat_export_parser import parse_chat_export_file
from src.models import AnomalyType


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_forensics.db"
    db = DatabaseManager(db_file)
    db.create_case("Case_DS_001", "Financial Dispute Demo", "Automated Test Case")
    return db


def test_ingestion_and_hashing(test_db):
    ingestor = EvidenceIngestionService(test_db)
    items = ingestor.ingest_directory("Case_DS_001", "sample_data/Case_DS_001")
    assert len(items) >= 5

    names = {item.file_name: item for item in items}
    assert "synthetic_pan_card.png" in names
    assert "synthetic_id_card.png" in names
    assert "whatsapp_screenshot.png" in names
    assert "whatsapp_chat_export.txt" in names
    assert "system_events.log" in names

    # Verify SHA-256 is 64 hex characters
    for item in items:
        assert len(item.sha256_hash) == 64
        assert item.file_size_bytes > 0


def test_deduplicated_evidence_refreshes_current_path(test_db, tmp_path):
    source = Path("sample_data/Case_DS_001/synthetic_pan_card.png")
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())

    ingestor = EvidenceIngestionService(test_db)
    original = ingestor.ingest_file("Case_DS_001", source)
    current = ingestor.ingest_file("Case_DS_001", copied)

    assert current.id == original.id
    assert current.sha256_hash == original.sha256_hash
    assert current.file_path == str(copied.resolve())
    assert test_db.list_evidence("Case_DS_001")[0].file_path == str(copied.resolve())


def test_document_authenticator_pan_card(test_db):
    ingestor = EvidenceIngestionService(test_db)
    item = ingestor.ingest_file("Case_DS_001", "sample_data/Case_DS_001/synthetic_pan_card.png")

    analyzer = DocumentAuthenticator()
    extracted_doc, findings = analyzer.analyze_document(item)

    assert extracted_doc.doc_type == "PAN_CARD"
    assert extracted_doc.fields.get("name") == "Aarav Sharma"
    assert extracted_doc.fields.get("dob") == "1994-05-12"
    assert extracted_doc.fields.get("pan_number") == "ABCPS1234D"

    test_db.insert_extracted_document(extracted_doc)
    for f in findings:
        test_db.insert_finding(f)


def test_document_authenticator_id_card_anomalies(test_db):
    ingestor = EvidenceIngestionService(test_db)
    item = ingestor.ingest_file("Case_DS_001", "sample_data/Case_DS_001/synthetic_id_card.png")

    analyzer = DocumentAuthenticator()
    extracted_doc, findings = analyzer.analyze_document(item)

    assert extracted_doc.doc_type == "ID_CARD"
    assert extracted_doc.fields.get("dob") == "1994-08-12"
    assert extracted_doc.qr_data is not None
    assert extracted_doc.qr_data.get("dob") == "1994-05-12"

    # Confirm ELA file was created
    ela_path = Path(extracted_doc.metadata["ela_overlay_path"])
    assert ela_path.exists()

    # Verify anomalies detected
    anomaly_types = [f.anomaly_type for f in findings]
    assert AnomalyType.QR_TEXT_MISMATCH.value in anomaly_types
    assert AnomalyType.ELA_SUSPECT.value in anomaly_types

    test_db.insert_extracted_document(extracted_doc)
    for f in findings:
        test_db.insert_finding(f)


def test_screenshot_authenticator(test_db):
    ingestor = EvidenceIngestionService(test_db)
    item = ingestor.ingest_file("Case_DS_001", "sample_data/Case_DS_001/whatsapp_screenshot.png")

    analyzer = ScreenshotAuthenticator()
    extracted_chat, findings = analyzer.analyze_screenshot(item)

    assert len(extracted_chat.messages) == 4
    # Check amounts extracted
    amounts = [amt for msg in extracted_chat.messages for amt in msg.amounts]
    assert 50000.0 in amounts

    # Check chronological inversion finding
    anomaly_types = [f.anomaly_type for f in findings]
    assert AnomalyType.OUT_OF_ORDER_TIMESTAMP.value in anomaly_types

    test_db.insert_extracted_chat(extracted_chat)
    for f in findings:
        test_db.insert_finding(f)


def test_chat_export_parser(test_db):
    ingestor = EvidenceIngestionService(test_db)
    item = ingestor.ingest_file("Case_DS_001", "sample_data/Case_DS_001/whatsapp_chat_export.txt")

    extracted_chat = parse_chat_export_file(item)
    assert len(extracted_chat.messages) == 3

    # Check that the modified amount 5000 is extracted
    amounts = [amt for msg in extracted_chat.messages for amt in msg.amounts]
    assert 5000.0 in amounts
    assert 50000.0 not in amounts

    test_db.insert_extracted_chat(extracted_chat)
