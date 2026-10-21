"""End-to-end automated forensic pipeline for Drishti-Sutra."""

import time
from pathlib import Path
from typing import Optional

from src.analyzers.chat_export_parser import parse_chat_export_file
from src.analyzers.document_analyzer import DocumentAuthenticator
from src.analyzers.event_log_parser import parse_event_log_file
from src.analyzers.screenshot_analyzer import ScreenshotAuthenticator
from src.database import DatabaseManager
from src.graph_builder import build_evidence_graph, build_plotly_figure
from src.ingestion import EvidenceIngestionService
from src.models import (
    FileProcessingStatus,
    PipelineReport,
    utc_now_iso,
)
from src.timeline import build_timeline
from src.truth_engine import run_truth_engine as _execute_truth_engine


def run_case_pipeline(
    case_dir_or_zip: str | Path,
    db_path: str | Path = "data/drishti_sutra.db",
    case_id: str = "Case_DS_001",
    title: Optional[str] = None,
    clear_first: bool = True,
) -> PipelineReport:
    """Run full automated forensic ingestion, analysis, truth engine, and timeline pipeline."""
    started_at = utc_now_iso()
    input_path = Path(case_dir_or_zip).resolve()
    db = DatabaseManager(db_path)
    ingestor = EvidenceIngestionService(db)

    case_title = title or f"Forensic Case {case_id}"
    db.create_case(case_id, case_title, f"Automated processing of {input_path.name}")

    if clear_first:
        db.clear_analysis(case_id)

    # 1. Ingestion Phase
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        evidence_items = ingestor.ingest_zip(case_id, input_path)
    elif input_path.is_file():
        evidence_items = [ingestor.ingest_file(case_id, input_path)]
    elif input_path.is_dir():
        evidence_items = ingestor.ingest_directory(case_id, input_path)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Set up dedicated artifacts directory for ELA overlays
    artifacts_dir = Path("data/artifacts") / case_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    doc_authenticator = DocumentAuthenticator(ela_output_dir=artifacts_dir)
    screenshot_authenticator = ScreenshotAuthenticator()

    file_statuses: list[FileProcessingStatus] = []
    errors: list[str] = []

    # 2. Analysis Phase
    for item in evidence_items:
        t0 = time.time()
        findings_count = 0
        status = "processed"
        reason = ""
        method = "none"

        try:
            if item.file_type == "kyc_doc":
                doc, findings = doc_authenticator.analyze_document(item)
                db.insert_extracted_document(doc)
                for f in findings:
                    db.insert_finding(f)
                findings_count = len(findings)
                method = doc.metadata.get("extraction_method", "ocr")

            elif item.file_type == "chat_screenshot":
                chat, findings = screenshot_authenticator.analyze_screenshot(item)
                db.insert_extracted_chat(chat)
                for f in findings:
                    db.insert_finding(f)
                findings_count = len(findings)
                method = chat.metadata.get("extraction_method", "opencv_ocr")

            elif item.file_type == "chat_export":
                chat = parse_chat_export_file(item)
                db.insert_extracted_chat(chat)
                method = chat.metadata.get("extraction_method", "regex_export")

            elif item.file_type == "system_log":
                records = parse_event_log_file(item)
                for r in records:
                    db.insert_event_log_record(r)
                method = "log_parser"

            else:
                status = "skipped"
                reason = f"No specialized analyzer configured for type: {item.file_type}"

        except Exception as e:
            status = "failed"
            reason = str(e)
            errors.append(f"Failed processing {item.file_name}: {e}")

        duration = round(time.time() - t0, 3)
        file_statuses.append(
            FileProcessingStatus(
                file_name=item.file_name,
                file_type=item.file_type,
                status=status,
                reason=reason,
                extraction_method=method,
                duration_seconds=duration,
                finding_count=findings_count,
            )
        )

    # 3. Truth Engine Phase
    truth_result = _execute_truth_engine(case_id, db)

    # 4. Timeline Reconstruction Phase
    timeline_events = build_timeline(case_id, db)

    completed_at = utc_now_iso()
    total_findings = sum(f.finding_count for f in file_statuses) + len(truth_result.indicators)

    return PipelineReport(
        case_id=case_id,
        case_dir=str(input_path),
        started_at=started_at,
        completed_at=completed_at,
        files=file_statuses,
        truth_engine_result=truth_result,
        total_evidence=len(evidence_items),
        total_findings=total_findings,
        total_contradictions=len(truth_result.contradictions),
        total_timeline_events=len(timeline_events),
        success=len(errors) == 0,
        errors=errors,
    )


def _run_tracked_stage(case_id: str, module: str, db: DatabaseManager, evidence_items: list, processor,
                       clear_options: dict) -> dict:
    """Run existing analyzers for one module and persist status/audit records."""
    import time as time_module

    fingerprint = db.evidence_fingerprint(case_id)
    if module in {"document", "screenshot"}:
        db.invalidate_truth_engine(case_id, f"{module} prerequisite analysis rerun")
    started_at = db.start_analysis_run(case_id, module, fingerprint)
    statuses: list[FileProcessingStatus] = []
    errors: list[str] = []
    successful = 0
    try:
        db.clear_evidence_results(case_id, [item.id for item in evidence_items], **clear_options)
        for item in evidence_items:
            t0 = time_module.time()
            try:
                method, findings_count = processor(item)
                successful += 1
                item.metadata["processing_status"] = "processed"
                item.metadata["extraction_method"] = method or "none"
                item.metadata["processing_reason"] = ""
                db.update_evidence_location(item)
                statuses.append(FileProcessingStatus(
                    file_name=item.file_name, file_type=item.file_type, status="processed", reason="",
                    extraction_method=method or "none", duration_seconds=round(time_module.time() - t0, 3),
                    finding_count=findings_count,
                ))
            except Exception as exc:
                reason = str(exc)
                errors.append(f"Failed processing {item.file_name}: {reason}")
                item.metadata["processing_status"] = "failed"
                item.metadata["extraction_method"] = "none"
                item.metadata["processing_reason"] = reason
                db.update_evidence_location(item)
                statuses.append(FileProcessingStatus(
                    file_name=item.file_name, file_type=item.file_type, status="failed", reason=reason,
                    extraction_method="none", duration_seconds=round(time_module.time() - t0, 3), finding_count=0,
                ))
        status = "failed" if errors and successful == 0 else "partial" if errors else "completed"
        saved_status = db.finish_analysis_run(case_id, module, status, fingerprint)
        return {
            "module": module, "status": saved_status, "started_at": started_at,
            "completed_at": utc_now_iso(), "files": statuses, "errors": errors,
        }
    except Exception:
        db.finish_analysis_run(case_id, module, "failed", fingerprint)
        raise


def run_document_analysis(case_id: str, db_path: str | Path = "data/drishti_sutra.db") -> dict:
    """Run the existing document analyzer over this case's KYC/document evidence."""
    db = DatabaseManager(db_path)
    evidence_items = [item for item in db.list_evidence(case_id) if item.file_type == "kyc_doc"]
    artifacts_dir = Path(db_path).resolve().parent / "artifacts" / case_id
    authenticator = DocumentAuthenticator(ela_output_dir=artifacts_dir)

    def process(item):
        doc, findings = authenticator.analyze_document(item)
        db.insert_extracted_document(doc)
        for finding in findings:
            db.insert_finding(finding)
        return doc.metadata.get("extraction_method", "ocr"), len(findings)

    return _run_tracked_stage(case_id, "document", db, evidence_items, process, {"documents": True})


def run_screenshot_analysis(case_id: str, db_path: str | Path = "data/drishti_sutra.db") -> dict:
    """Analyze screenshots and parse chat exports and event logs for one case."""
    db = DatabaseManager(db_path)
    evidence_items = [
        item for item in db.list_evidence(case_id)
        if item.file_type in {"chat_screenshot", "chat_export", "system_log"}
    ]
    authenticator = ScreenshotAuthenticator()

    def process(item):
        if item.file_type == "chat_screenshot":
            chat, findings = authenticator.analyze_screenshot(item)
            db.insert_extracted_chat(chat)
            for finding in findings:
                db.insert_finding(finding)
            return chat.metadata.get("extraction_method", "opencv_ocr"), len(findings)
        if item.file_type == "chat_export":
            chat = parse_chat_export_file(item)
            db.insert_extracted_chat(chat)
            return chat.metadata.get("extraction_method", "regex_export"), 0
        records = parse_event_log_file(item)
        for record in records:
            db.insert_event_log_record(record)
        return "log_parser", 0

    return _run_tracked_stage(
        case_id, "screenshot", db, evidence_items, process,
        {"chats": True, "event_logs": True},
    )


def run_truth_engine(case_id: str, db_path: str | Path = "data/drishti_sutra.db"):
    """Run Truth Engine and timeline reconstruction after fresh document and chat/log stages."""
    db = DatabaseManager(db_path)
    db.mark_stale_analysis_runs(case_id)
    missing = [
        module for module in ("document", "screenshot")
        if not db.analysis_is_current(case_id, module)
    ]
    if missing:
        raise ValueError(
            "Truth Engine requires current Document Analysis and Screenshot & Chat analysis first. "
            f"Missing or stale modules: {', '.join(missing)}."
        )

    fingerprint = db.evidence_fingerprint(case_id)
    started_at = db.start_analysis_run(case_id, "truth_engine", fingerprint)
    try:
        db.clear_truth_results(case_id)
        result = _execute_truth_engine(case_id, db)
        timeline_events = build_timeline(case_id, db)
        db.log_audit(
            case_id,
            "TRUTH_ENGINE_RUN",
            details={"contradictions": len(result.contradictions), "indicators": len(result.indicators),
                     "timeline_events": len(timeline_events)},
        )
        db.finish_analysis_run(case_id, "truth_engine", "completed", fingerprint)
        return result
    except Exception:
        db.finish_analysis_run(case_id, "truth_engine", "failed", fingerprint)
        raise
