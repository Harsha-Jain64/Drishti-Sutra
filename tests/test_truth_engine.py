"""Comprehensive test suite for Drishti-Sutra Module 3 (Truth Engine, Timeline, Graph)."""

import os
from pathlib import Path
import pytest

from src.analyzers.event_log_parser import parse_event_log_file
from src.database import DatabaseManager
from src.graph_builder import build_evidence_graph, build_plotly_figure
from src.models import (
    ChatMessage,
    EvidenceItem,
    ExtractedChat,
    TimelineEvent,
    stable_id,
)
from src.pipeline import run_case_pipeline
from src.timeline import build_timeline
from src.truth_engine import (
    generate_summary,
    match_entities,
    normalize_dob,
    normalize_entity_name,
    normalize_id_number,
    parse_full_or_inferred_datetime,
    run_truth_engine,
)

SAMPLE_CASE_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "Case_DS_001"


# -----------------------------------------------------------------------------
# End-to-End Pipeline & Ground-Truth Verification Tests
# -----------------------------------------------------------------------------

def test_pipeline_e2e_case_ds_001(tmp_path):
    """Verify all 6 ground-truth contradictions and log indicator on Case_DS_001."""
    db_file = tmp_path / "test_pipeline.db"
    case_id = "Case_DS_001_Test"

    report = run_case_pipeline(
        case_dir_or_zip=SAMPLE_CASE_DIR,
        db_path=db_file,
        case_id=case_id,
        clear_first=True,
    )

    assert report.success is True
    assert report.total_evidence >= 5
    assert report.total_contradictions == 6
    assert report.total_timeline_events > 0

    te = report.truth_engine_result
    assert te is not None

    types_found = {c.contradiction_type for c in te.contradictions}
    expected_types = {
        "DOC_FIELD_MISMATCH",
        "QR_PRINTED_MISMATCH",
        "MESSAGE_ABSENT_IN_EXPORT",
        "AMOUNT_MISMATCH",
        "ORDER_INVERSION",
        "LOG_COMM_CONFLICT",
    }
    assert expected_types.issubset(types_found)

    # Verify structured fields on every contradiction
    for c in te.contradictions:
        assert c.source_a_ref is not None
        assert "evidence_id" in c.source_a_ref
        assert "file_name" in c.source_a_ref
        assert c.observed_fact_a is not None
        assert c.observed_fact_b is not None
        assert 0.0 <= c.confidence <= 1.0

        # Forensic neutrality: ensure no claims of definitive proof or legal forgery
        lower_exp = c.explanation.lower()
        assert "proof of forgery" not in lower_exp
        assert "guilty" not in lower_exp
        assert "criminal" not in lower_exp

    # Check post-transaction file modification indicator
    indicator_types = {i.anomaly_type for i in te.indicators}
    assert "POST_TXN_FILE_MOD" in indicator_types

    # Verify timeline and graph generation
    db = DatabaseManager(db_file)
    events = db.list_timeline_events(case_id)
    assert len(events) >= 10

    graph = build_evidence_graph(case_id, db)
    assert graph.number_of_nodes() >= 15
    assert graph.number_of_edges() >= 20

    fig = build_plotly_figure(graph, case_id=case_id)
    assert fig is not None


def test_pipeline_idempotency(tmp_path):
    """Running the pipeline twice must produce identical IDs and row counts."""
    db_file = tmp_path / "idempotent.db"
    case_id = "Idempotency_Case"

    rep1 = run_case_pipeline(SAMPLE_CASE_DIR, db_file, case_id, clear_first=True)
    rep2 = run_case_pipeline(SAMPLE_CASE_DIR, db_file, case_id, clear_first=True)

    assert rep1.total_evidence == rep2.total_evidence
    assert rep1.total_contradictions == rep2.total_contradictions
    assert rep1.total_timeline_events == rep2.total_timeline_events

    ids1 = [c.id for c in rep1.truth_engine_result.contradictions]
    ids2 = [c.id for c in rep2.truth_engine_result.contradictions]
    assert sorted(ids1) == sorted(ids2)


# -----------------------------------------------------------------------------
# Unit Tests: Normalization, Matching & Scoring Logic
# -----------------------------------------------------------------------------

def test_entity_name_matching():
    """Verify fuzzy matching and normalization across different naming formats."""
    assert normalize_entity_name("Mr. Aarav Sharma") == "aarav sharma"
    assert normalize_entity_name("aarav.sharma") == "aarav sharma"
    assert normalize_entity_name("SHARMA, AARAV") == "sharma aarav"

    # Confirmed match
    match, status, score = match_entities("Aarav Sharma", "aarav.sharma")
    assert match is True
    assert status == "confirmed"

    # Suspected match with minor typo or extra token
    match, status, score = match_entities("Vikram Patel", "Vikram K Patel")
    assert match is True
    assert status == "confirmed"

    # Unrelated entity
    match, status, score = match_entities("Aarav Sharma", "Rahul Verma")
    assert match is False
    assert status == "unrelated"


def test_dob_normalization():
    """Verify normalization of various date formats to ISO YYYY-MM-DD."""
    assert normalize_dob("1994-05-12") == "1994-05-12"
    assert normalize_dob("12/05/1994") == "1994-05-12"
    assert normalize_dob("12-05-1994") == "1994-05-12"
    assert normalize_dob(None) is None


def test_id_normalization():
    """Verify whitespace and punctuation stripping from ID numbers."""
    assert normalize_id_number("9876 5432 1098") == "987654321098"
    assert normalize_id_number("ABCPS-1234-D") == "ABCPS1234D"
    assert normalize_id_number(None) is None


def test_coverage_guard_skips_out_of_bounds_message(tmp_path):
    """A message with timestamp outside export coverage range must be recorded in skipped_comparisons."""
    db = DatabaseManager(tmp_path / "cov_test.db")
    case_id = "Cov_Guard_Case"
    db.create_case(case_id, "Coverage Test")

    ev_sc = EvidenceItem(
        id=stable_id(case_id, "sc"), case_id=case_id, file_path="sc.png",
        file_name="sc.png", file_type="chat_screenshot", sha256_hash="h1", file_size_bytes=100
    )
    ev_ex = EvidenceItem(
        id=stable_id(case_id, "ex"), case_id=case_id, file_path="ex.txt",
        file_name="ex.txt", file_type="chat_export", sha256_hash="h2", file_size_bytes=100
    )
    db.insert_evidence(ev_sc)
    db.insert_evidence(ev_ex)

    # Export covers 10:00 to 10:05
    chat_ex = ExtractedChat(
        id=stable_id(ev_ex.id, "EXPORT_TXT"), evidence_id=ev_ex.id, platform="WHATSAPP", source_type="EXPORT_TXT",
        messages=[
            ChatMessage(id="m1", sender="Aarav", timestamp_str="20/10/2026, 10:00:00 AM", parsed_datetime="2026-10-20T10:00:00", content="Hi"),
            ChatMessage(id="m2", sender="Vikram", timestamp_str="20/10/2026, 10:05:00 AM", parsed_datetime="2026-10-20T10:05:00", content="Bye"),
        ]
    )
    # Screenshot has message at 11:30 AM (outside coverage)
    chat_sc = ExtractedChat(
        id=stable_id(ev_sc.id, "SCREENSHOT"), evidence_id=ev_sc.id, platform="WHATSAPP", source_type="SCREENSHOT",
        messages=[
            ChatMessage(id="m3", sender="Aarav", timestamp_str="11:30 AM", content="Late evening note"),
        ]
    )
    db.insert_extracted_chat(chat_ex)
    db.insert_extracted_chat(chat_sc)

    result = run_truth_engine(case_id, db)
    # Must NOT flag as absent in export
    absent_types = [c.contradiction_type for c in result.contradictions if c.contradiction_type == "MESSAGE_ABSENT_IN_EXPORT"]
    assert len(absent_types) == 0

    # Must be recorded in skipped comparisons
    skipped_reasons = [s["reason"] for s in result.skipped_comparisons]
    assert any("outside export covered span" in r for r in skipped_reasons)


def test_midnight_rollover_not_flagged_as_inversion():
    """A timestamp decrease across midnight (> 12 hours) must be treated as next-day rollover."""
    from src.truth_engine import parse_full_or_inferred_datetime
    msg1 = ChatMessage(id="1", sender="A", timestamp_str="11:58 PM", content="Night")
    msg2 = ChatMessage(id="2", sender="A", timestamp_str="12:02 AM", content="Morning")

    dt1, _ = parse_full_or_inferred_datetime(msg1, "2026-10-20")
    dt2, _ = parse_full_or_inferred_datetime(msg2, "2026-10-20")

    # Time delta between 23:58 and 00:02 on same date is ~23.9 hours
    diff_hours = (dt1 - dt2).total_seconds() / 3600.0
    assert diff_hours > 12.0  # Demonstrates > 12h decrease triggers midnight rollover guard


def test_missing_source_records_skipped_and_does_not_crash(tmp_path):
    """Case with only a screenshot and no export records skipped comparison cleanly."""
    db = DatabaseManager(tmp_path / "solo.db")
    case_id = "Solo_Case"
    db.create_case(case_id, "Solo Screenshot Case")

    ev = EvidenceItem(
        id=stable_id(case_id, "sc_only"), case_id=case_id, file_path="sc.png",
        file_name="sc.png", file_type="chat_screenshot", sha256_hash="h1", file_size_bytes=100
    )
    db.insert_evidence(ev)
    chat = ExtractedChat(
        id=stable_id(ev.id, "SCREENSHOT"), evidence_id=ev.id, platform="WHATSAPP", source_type="SCREENSHOT",
        messages=[ChatMessage(id="m1", sender="Aarav", timestamp_str="10:00 AM", content="Hello")]
    )
    db.insert_extracted_chat(chat)

    result = run_truth_engine(case_id, db)
    assert len(result.contradictions) == 0
    skipped_rules = [s["rule"] for s in result.skipped_comparisons]
    assert "MESSAGE_ALIGNMENT" in skipped_rules


def test_event_log_unparseable_lines_stored_as_skipped(tmp_path):
    """Unparseable log lines are recorded as SKIPPED, not dropped."""
    log_content = (
        "2026-10-20 10:10:02 INFO [auth] User 'aarav.sharma' authenticated\n"
        "MALFORMED LINE WITHOUT TIMESTAMP OR COMPONENT\n"
        "\n"
        "2026-10-20 10:15:15 INFO [transaction] Amount=50000.00 Payee='Vikram Patel'\n"
    )
    log_file = tmp_path / "test.log"
    log_file.write_text(log_content, encoding="utf-8")

    ev = EvidenceItem(
        id=stable_id("test", "test.log"), case_id="test", file_path=str(log_file),
        file_name="test.log", file_type="system_log", sha256_hash="abc", file_size_bytes=len(log_content)
    )
    records = parse_event_log_file(ev)
    assert len(records) == 4

    statuses = [r.status for r in records]
    assert statuses == ["PARSED", "SKIPPED", "SKIPPED", "PARSED"]
    assert records[1].skip_reason != ""
    assert records[2].skip_reason == "Empty line"


def test_llm_hook_fallback_on_unreachable_url(monkeypatch):
    """When DRISHTI_LLM_URL is unreachable, the engine falls back to the deterministic template."""
    monkeypatch.setenv("DRISHTI_LLM_URL", "http://127.0.0.1:9999/nonexistent")
    summary, src = generate_summary([], [], {"total_contradictions": 0, "total_indicators": 0})
    assert src == "template"
    assert "Forensic Executive Summary" in summary
