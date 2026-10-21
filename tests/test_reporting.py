import re
from pathlib import Path

from src.database import DatabaseManager
from src.ingestion import EvidenceIngestionService
from src.pipeline import run_document_analysis, run_screenshot_analysis, run_truth_engine
from src.reporting import generate_html_report, record_report_export


def _resource_urls(report: str) -> list[str]:
    return re.findall(r'''(?:src|href)\s*=\s*["']https?://[^"']+''', report, re.IGNORECASE)


def test_html_report_has_schedule_draft_embedded_case_data_and_is_offline(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    case_dir = project_root / "sample_data" / "Case_DS_001"
    db_path = tmp_path / "report_test.db"
    db = DatabaseManager(db_path)
    db.create_case("Case_DS_001", "Synthetic sample case", "Report integration test")
    ingestor = EvidenceIngestionService(db)
    ingestor.ingest_directory("Case_DS_001", case_dir)

    document_result = run_document_analysis("Case_DS_001", db_path)
    screenshot_result = run_screenshot_analysis("Case_DS_001", db_path)
    truth_result = run_truth_engine("Case_DS_001", db_path)
    assert document_result["status"] == "completed"
    assert screenshot_result["status"] == "completed"
    assert truth_result is not None
    assert len(db.list_contradictions("Case_DS_001")) == 6

    report = generate_html_report("Case_DS_001", db)
    hashes = [item.sha256_hash for item in db.list_evidence("Case_DS_001")]
    contradiction_types = [item.contradiction_type for item in db.list_contradictions("Case_DS_001") if item.contradiction_type]

    assert hashes and all(value in report for value in hashes)
    assert contradiction_types and all(value in report for value in contradiction_types)
    assert report.index('<section class="certificate">') < report.index('<section class="cover">')
    assert '<header class="report-header">' in report
    assert "REPORT GENERATED (UTC)" in report
    assert "THE SCHEDULE" in report and "See Section 63(4)(c)" in report
    assert "PART A" in report and "PART B" in report
    assert "DRAFT FOR REVIEW AND COMPLETION. Not a legally executed certificate. Legal applicability must be verified." in report
    assert "[BLANK — NO SIGNATURE PROVIDED]" in report
    assert "Chronological timeline" in report or "Visual timeline and timeline table" in report
    assert "findings counted separately" in report.lower() or "findings and contradictions counted separately" in report.lower()
    assert "Potential anomalies" in report or "not proof of forgery or wrongdoing" in report
    assert "data:image/" in report
    assert not _resource_urls(report)

    # Exercise the same callback target used when the Streamlit download is clicked.
    record_report_export("Case_DS_001", db, "report_Case_DS_001.html")
    exported_report = generate_html_report("Case_DS_001", db)
    assert "REPORT_EXPORTED" in exported_report


def test_empty_case_report_is_isolated_and_labels_analysis_not_run(tmp_path):
    db = DatabaseManager(tmp_path / "empty_case.db")
    db.create_case("EMPTY_REPORT_CASE", "Empty report case", "No evidence loaded")
    report = generate_html_report("EMPTY_REPORT_CASE", db)

    assert '<section class="certificate">' in report
    assert "EMPTY_REPORT_CASE" in report
    assert "Unique evidence items</span>" in report
    assert "Not Run" in report
    assert "No evidence is recorded." in report
    assert "No current Truth Engine/timeline run" in report
    assert "Synthetic sample case" not in report
    assert "synthetic_id_card.png" not in report
    assert not _resource_urls(report)
    assert "href=\"https://" not in report and "src=\"https://" not in report
