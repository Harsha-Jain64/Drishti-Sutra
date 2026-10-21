from pathlib import Path

import pytest

from src.database import DatabaseManager
from src.ingestion import EvidenceIngestionService
from src.pipeline import run_document_analysis, run_screenshot_analysis, run_truth_engine


def test_staged_sample_workflow_isolated_and_fingerprint_gated(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    sample_dir = project_root / "sample_data" / "Case_DS_001"
    db_path = tmp_path / "workflow.db"
    db = DatabaseManager(db_path)

    db.create_case("EMPTY_CASE", "New empty case", "Created for workflow verification")
    assert db.list_evidence("EMPTY_CASE") == []
    assert {key: run["status"] for key, run in db.get_analysis_runs("EMPTY_CASE").items()} == {
        "document": "not_run",
        "screenshot": "not_run",
        "truth_engine": "not_run",
    }

    empty_doc_run = run_document_analysis("EMPTY_CASE", db_path)
    assert empty_doc_run["status"] == "completed"
    assert not empty_doc_run["files"]
    with pytest.raises(ValueError, match="requires current Document Analysis and Screenshot"):
        run_truth_engine("EMPTY_CASE", db_path)

    ingested = EvidenceIngestionService(db).ingest_directory("EMPTY_CASE", sample_dir)
    assert len(ingested) == 5
    assert all("manifest.json" != item.file_name for item in ingested)
    invalidated = db.get_analysis_runs("EMPTY_CASE")
    assert invalidated["document"]["status"] == "stale"
    assert invalidated["screenshot"]["status"] == "not_run"
    assert invalidated["truth_engine"]["status"] == "not_run"

    documents = run_document_analysis("EMPTY_CASE", db_path)
    assert documents["status"] == "completed"
    chats = run_screenshot_analysis("EMPTY_CASE", db_path)
    assert chats["status"] == "completed"

    evidence = {item.file_name: item for item in db.list_evidence("EMPTY_CASE")}
    methods = {name: item.metadata.get("extraction_method") for name, item in evidence.items()}
    assert methods == {
        "synthetic_id_card.png": "embedded_metadata",
        "synthetic_pan_card.png": "embedded_metadata",
        "system_events.log": "log_parser",
        "whatsapp_chat_export.txt": "regex_export",
        "whatsapp_screenshot.png": "embedded_metadata",
    }

    truth = run_truth_engine("EMPTY_CASE", db_path)
    assert len(truth.contradictions) == 6
    assert len(db.list_contradictions("EMPTY_CASE")) == 6
    assert len(db.list_timeline_events("EMPTY_CASE")) == 14
    assert all(db.analysis_is_current("EMPTY_CASE", module) for module in ("document", "screenshot", "truth_engine"))

    run_document_analysis("EMPTY_CASE", db_path)
    assert db.get_analysis_runs("EMPTY_CASE")["truth_engine"]["status"] == "stale"
    assert not db.analysis_is_current("EMPTY_CASE", "truth_engine")
    truth = run_truth_engine("EMPTY_CASE", db_path)
    assert len(truth.contradictions) == 6

    removed = evidence["synthetic_pan_card.png"]
    assert db.delete_evidence("EMPTY_CASE", removed.id)
    assert all(run["status"] == "stale" for run in db.get_analysis_runs("EMPTY_CASE").values())
    assert not db.analysis_is_current("EMPTY_CASE", "truth_engine")

    db.create_case("SECOND_CASE", "Second isolated case")
    assert db.list_evidence("SECOND_CASE") == []
    assert db.list_extracted_documents("SECOND_CASE") == []
    assert db.list_extracted_chats("SECOND_CASE") == []
    assert db.list_findings("SECOND_CASE") == []
    assert db.list_contradictions("SECOND_CASE") == []
    assert db.list_timeline_events("SECOND_CASE") == []

    actions = {entry["action"] for entry in db.get_audit_trail("EMPTY_CASE")}
    assert {"CASE_CREATED", "EVIDENCE_UPLOADED", "ANALYSIS_STARTED", "ANALYSIS_COMPLETED", "TRUTH_ENGINE_RUN"} <= actions
    assert "EVIDENCE_REMOVED" in actions
