"""Streamlit dashboard shell and demo-case views for Drishti-Sutra."""

import html
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.database import DatabaseManager
from src.graph_builder import build_evidence_graph, build_plotly_figure
from src.ingestion import EvidenceIngestionService
from src.models import AnomalyType
from src.pipeline import run_document_analysis, run_screenshot_analysis, run_truth_engine
from src.reporting import generate_html_report, record_report_export


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "drishti_sutra.db"
UPLOAD_ROOT = ROOT / "data" / "uploads"
NAV_ITEMS = [
    "1 · Overview",
    "2 · Evidence Intake",
    "3 · Document Analysis",
    "4 · Screenshot & Chat",
    "5 · Truth Engine",
    "6 · Timeline & Graph",
    "7 · Reports & Audit",
]

st.set_page_config(
    page_title="Drishti-Sutra | Evidence Analysis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _safe(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _valid_case_id(case_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case_id))


def _inject_theme() -> None:
    st.markdown("""
    <style>
    :root{--ink:#142b45;--muted:#62758a;--teal:#0f8a83;--amber:#c98618;--red:#bd3b3b;--green:#20805c;--paper:#f5f3ee;--line:#e2e6e5}
    .stApp{background:linear-gradient(180deg,#f7f6f2 0%,#f2f3f0 100%);color:var(--ink)}
    [data-testid="stHeader"]{background:rgba(247,246,242,.92)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#e9efed 0%,#e4eceb 100%);border-right:1px solid #d5e0dd}
    [data-testid="stSidebar"] h2{color:#12314b;letter-spacing:-.03em}
    [data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.48rem .65rem;border-radius:10px}
    h1,h2,h3,h4{color:#142b45!important;letter-spacing:-.025em}
    .brand-line{font-size:11px;font-weight:800;letter-spacing:.14em;color:#117b76;margin:.3rem 0 1.2rem}
    .brand-line span{color:#7d8c97;font-weight:600;letter-spacing:.08em;margin-left:.7rem}
    [data-testid="stMetric"]{background:#fff;border:1px solid #e2e7e4;border-radius:16px;padding:16px 18px;box-shadow:0 5px 18px rgba(20,43,69,.055)}
    [data-testid="stMetricLabel"]{color:#65798b!important}
    [data-testid="stMetricValue"]{color:#142b45!important}
    [data-testid="stVerticalBlockBorderWrapper"]{border-color:#e0e6e3!important;border-radius:18px!important;background:rgba(255,255,255,.82);box-shadow:0 8px 24px rgba(20,43,69,.06)}
    .stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid #d8e3e0;font-weight:650;transition:all .15s ease}
    .stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:#0f8a83;border-color:#0f8a83;color:#fff;box-shadow:0 5px 12px rgba(15,138,131,.18)}
    .stButton>button:hover,.stDownloadButton>button:hover{border-color:#0f8a83;color:#08756f;transform:translateY(-1px)}
    .stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover{background:#08756f;color:#fff}
    [data-testid="stAlert"]{border-radius:13px}
    [data-testid="stDataFrame"],.stTable{border:1px solid #e1e7e4;border-radius:13px;overflow:hidden}
    [data-testid="stImage"] img{border-radius:13px}
    .status-card{background:#fff;border:1px solid #e2e7e4;border-radius:14px;padding:13px 15px;box-shadow:0 4px 12px rgba(20,43,69,.045);height:100%}
    .status-card small,.muted{color:#697d90}.status-card strong{color:#142b45}
    .badge{display:inline-block;border-radius:999px;padding:4px 10px;font-size:11px;font-weight:750;letter-spacing:.03em}
    .badge-green{background:#e4f3eb;color:#1d7651}.badge-amber{background:#fff2d8;color:#98620b}.badge-red{background:#fde8e6;color:#ad302d}.badge-navy{background:#e9eef5;color:#3c5673}
    .field-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:.5rem 0 1rem}
    .field-card{border:1px solid #e1e7e4;border-radius:12px;background:#fff;padding:12px 14px;box-shadow:0 3px 12px rgba(20,43,69,.04)}
    .field-card .label{font-size:11px;color:#6c7f90;text-transform:uppercase;letter-spacing:.06em}.field-card .value{font-weight:700;color:#142b45;margin-top:5px;overflow-wrap:anywhere}
    .verifier-note{border-left:4px solid #d59a2a;background:#fff7e8;padding:10px 13px;border-radius:8px;color:#745817;font-size:13px;margin:.7rem 0}
    .chat-bubble{background:#fff;border:1px solid #e2e8e5;border-radius:14px;padding:11px 14px;margin:8px 0;box-shadow:0 3px 10px rgba(20,43,69,.04)}
    .chat-bubble.outgoing{background:#e7f5f0;margin-left:8%}.chat-meta{font-size:11px;color:#62758a;margin-bottom:4px}.chat-content{color:#19344c}
    .legend-row{display:flex;gap:16px;flex-wrap:wrap;color:#62758a;font-size:12px;margin:.35rem 0 .8rem}
    .legend-chip{display:flex;align-items:center;gap:6px}.legend-dot{width:11px;height:11px;border-radius:50%;display:inline-block}
    @media(max-width:760px){.field-grid{grid-template-columns:1fr}.brand-line span{display:block;margin:.25rem 0}}
    </style>""", unsafe_allow_html=True)


def _opencv_heatmap(image_path: str | Path) -> np.ndarray:
    """Compute an ELA-style colormap from current image pixels; heuristic only."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be read for heatmap generation")
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Image recompression failed")
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image, recompressed)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _anomaly_overlay(image_path: str | Path, messages: list[Any], finding: Any | None) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be read for overlay generation")
    boxes = []
    if finding:
        box = (finding.details.get("inverted_message") or {}).get("bounding_box")
        if box:
            boxes.append(box)
    for box in boxes:
        x, y = max(0, int(box.get("x", 0))), max(0, int(box.get("y", 0)))
        w, h = int(box.get("w", 0)), int(box.get("h", 0))
        if w > 0 and h > 0:
            cv2.rectangle(image, (x, y), (x + w, y + h), (20, 35, 220), 4)
            cv2.putText(image, "FLAGGED MESSAGE", (x, max(20, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 35, 220), 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _persist_processing_status(db: DatabaseManager, case_id: str, pipeline_report: Any) -> None:
    """Persist run outcome in existing evidence metadata so UI status is DB-backed."""
    status_by_name = {item.file_name: item for item in pipeline_report.files}
    for evidence in db.list_evidence(case_id):
        outcome = status_by_name.get(evidence.file_name)
        if outcome is None:
            continue
        evidence.metadata["processing_status"] = outcome.status
        evidence.metadata["extraction_method"] = outcome.extraction_method or "none"
        evidence.metadata["processing_reason"] = outcome.reason or ""
        db.update_evidence_location(evidence)


def _processing_display(evidence: Any, documents: dict[str, Any], chats: dict[str, Any], log_ids: set[str]) -> tuple[str, str]:
    saved_status = evidence.metadata.get("processing_status")
    saved_method = evidence.metadata.get("extraction_method")
    if saved_status:
        return str(saved_status), str(saved_method or "none")
    if evidence.id in documents:
        return "processed", str(documents[evidence.id].metadata.get("extraction_method") or "not recorded")
    if evidence.id in chats:
        return "processed", str(chats[evidence.id].metadata.get("extraction_method") or "not recorded")
    if evidence.id in log_ids:
        return "processed", "event-log parser"
    return "not recorded", "not recorded"


def _section_title(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def _run_stage_action(db: DatabaseManager, case_id: str, module: str, label: str, runner) -> bool:
    current = db.analysis_is_current(case_id, module)
    run = db.get_analysis_runs(case_id).get(module, {})
    if current:
        st.success(f"{label} is current for this evidence set.")
        return True
    status = run.get("status", "not_run")
    if status == "stale":
        st.warning(f"{label} is stale: re-run required because evidence was added, removed, or changed.")
    elif status == "failed":
        st.warning(f"The most recent {label} run failed. Review History, then rerun this module.")
    if st.button(f"Run {label}", type="primary", key=f"run_{module}_{case_id}"):
        try:
            with st.spinner(f"Running {label} against this case’s SQLite evidence…"):
                result = runner(case_id, DB_PATH)
            final_status = result.get("status", "completed")
            st.session_state["flash_message"] = ("error" if final_status in {"failed", "stale"} else "success", f"{label}: {final_status}.")
        except Exception as exc:
            st.session_state["flash_message"] = ("error", f"{label} could not complete: {exc}")
        st.rerun()
    if status in {"not_run", "running"}:
        st.caption(f"Status: {status.replace('_', ' ')}. Results appear after this module runs.")
    return False


def _load_case_data(db: DatabaseManager, case_id: str) -> dict[str, Any]:
    evidence = db.list_evidence(case_id)
    runs = db.get_analysis_runs(case_id)
    current = {module: db.analysis_is_current(case_id, module) for module in ("document", "screenshot", "truth_engine")}
    documents = db.list_extracted_documents(case_id) if current["document"] else []
    chats = db.list_extracted_chats(case_id) if current["screenshot"] else []
    evidence_types = {item.id: item.file_type for item in evidence}
    findings = []
    for finding in db.list_findings(case_id):
        types = {evidence_types.get(item_id) for item_id in finding.evidence_ids}
        if current["document"] and "kyc_doc" in types:
            findings.append(finding)
        elif current["screenshot"] and "chat_screenshot" in types:
            findings.append(finding)
        elif current["truth_engine"] and finding.anomaly_type == "POST_TXN_FILE_MOD":
            findings.append(finding)
    contradictions = db.list_contradictions(case_id) if current["truth_engine"] else []
    timeline = db.list_timeline_events(case_id) if current["truth_engine"] else []
    logs = db.list_event_log_records_for_case(case_id) if current["screenshot"] else []
    return {
        "summary": db.get_case_summary(case_id),
        "evidence": evidence,
        "documents": documents,
        "chats": chats,
        "findings": findings,
        "contradictions": contradictions,
        "timeline": timeline,
        "logs": logs,
        "audit": db.get_audit_trail(case_id),
        "runs": runs,
        "current": current,
    }


def _log_rejected_upload(db: DatabaseManager, case_id: str, file_name: str, reason: str) -> None:
    db.log_audit(case_id, "EVIDENCE_REJECTED", details={"file_name": file_name, "reason": reason})


def _ingest_uploaded_files(case_id: str, uploaded_files: list[Any], db: DatabaseManager) -> None:
    if not uploaded_files:
        st.warning("Choose a ZIP archive or one or more evidence files first.")
        return
    zip_files = [item for item in uploaded_files if Path(item.name.replace("\\", "/")).suffix.lower() == ".zip"]
    if zip_files and len(uploaded_files) != 1:
        reason = "Upload a ZIP archive by itself, or upload multiple individual files."
        for item in uploaded_files:
            _log_rejected_upload(db, case_id, Path(item.name).name, reason)
        st.error(reason)
        return

    run_dir = UPLOAD_ROOT / case_id / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    ingestor = EvidenceIngestionService(db)
    ingested = []
    errors = []
    for uploaded in uploaded_files:
        safe_name = Path(uploaded.name.replace("\\", "/")).name
        if not safe_name or safe_name in {".", ".."}:
            reason = "Invalid upload filename."
            _log_rejected_upload(db, case_id, str(uploaded.name), reason)
            errors.append(f"{uploaded.name}: {reason}")
            continue
        saved = run_dir / safe_name
        try:
            saved.write_bytes(uploaded.getvalue())
            if saved.suffix.lower() == ".zip":
                items = ingestor.ingest_zip(case_id, saved)
                if not items:
                    raise ValueError("The archive contained no ingestible evidence files.")
                ingested.extend(items)
            else:
                ingested.append(ingestor.ingest_file(case_id, saved))
        except Exception as exc:
            _log_rejected_upload(db, case_id, safe_name, str(exc))
            errors.append(f"{safe_name}: {exc}")

    if ingested:
        st.session_state["flash_message"] = (
            "success", f"Ingested {len(ingested)} evidence file(s). Analysis runs are now stale until rerun."
        )
        st.rerun()
    if errors:
        st.error("Evidence upload failed: " + " · ".join(errors))


def _ingest_sample_evidence(case_id: str, db: DatabaseManager) -> None:
    sample_dir = ROOT / "sample_data" / "Case_DS_001"
    try:
        items = EvidenceIngestionService(db).ingest_directory(case_id, sample_dir)
        if not items:
            raise ValueError("No evidence files were found in the synthetic sample folder.")
        st.session_state["flash_message"] = (
            "success", f"Loaded {len(items)} synthetic evidence file(s) through normal ingestion."
        )
        st.rerun()
    except Exception as exc:
        _log_rejected_upload(db, case_id, "sample_data/Case_DS_001", str(exc))
        st.error(f"Sample evidence could not be loaded: {exc}")


def _render_overview(data: dict[str, Any], case_id: str) -> None:
    _section_title("Case overview", "Current evidence and fresh analysis metrics are read from SQLite.")
    summary = data["summary"]
    if summary is None:
        st.info(f"No database record exists for case {case_id}. Create the case from Home first.")
        return
    st.markdown(
        f"**Case ID:** `{_safe(summary.case_id)}`　 ·　 **Status:** {_safe(summary.status)}"
        f"　 ·　 **Created:** {_safe(summary.created_at)}"
    )
    metric_cols = st.columns(4)
    metrics = [
        ("Evidence", len(data["evidence"])),
        ("Findings", len(data["findings"])),
        ("Contradictions", len(data["contradictions"])),
        ("Timeline events", len(data["timeline"])),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        col.metric(label, value)

    levels = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    st.subheader("Findings by severity")
    finding_counts = {level: sum((item.severity or "").upper() == level for item in data["findings"]) for level in levels}
    finding_cols = st.columns(len(levels))
    for col, level in zip(finding_cols, levels):
        col.metric(level.title(), finding_counts[level])
    st.subheader("Contradictions by severity")
    contradiction_counts = {level: sum((item.severity or "").upper() == level for item in data["contradictions"]) for level in levels}
    contradiction_cols = st.columns(len(levels))
    for col, level in zip(contradiction_cols, levels):
        col.metric(level.title(), contradiction_counts[level])
    if summary.description:
        st.caption(summary.description)


def _render_evidence(data: dict[str, Any], db: DatabaseManager, case_id: str) -> None:
    _section_title("Evidence intake", "Evidence is ingested into this case only; adding files marks prior analysis stale.")
    left, right = st.columns([3, 1])
    with left:
        uploaded_files = st.file_uploader(
            "Choose a ZIP archive or multiple evidence files",
            type=["zip", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "pdf", "txt", "csv", "log", "syslog"],
            accept_multiple_files=True,
            key=f"evidence_uploads_{case_id}",
        )
        upload_clicked = st.button("Add Evidence", type="primary", key=f"add_evidence_{case_id}")
    with right:
        st.write("")
        st.write("")
        sample_clicked = st.button("Load sample evidence (synthetic Case DS-001)", key=f"sample_evidence_{case_id}")
    if upload_clicked:
        _ingest_uploaded_files(case_id, uploaded_files or [], db)
    if sample_clicked:
        _ingest_sample_evidence(case_id, db)
    run = data["runs"].get("document", {})
    status = run.get("status", "not_run")
    if status == "stale":
        st.warning("Document analysis is stale: re-run required because the evidence set changed.")
    if not data["evidence"]:
        st.info("No evidence has been ingested for this case yet.")
        return
    documents = {item.evidence_id: item for item in data["documents"]}
    chats = {item.evidence_id: item for item in data["chats"]}
    log_ids = {item.evidence_id for item in data["logs"]}
    rows = []
    for item in data["evidence"]:
        status, method = _processing_display(item, documents, chats, log_ids)
        rows.append(
            "<tr>"
            f"<td>{_safe(item.file_name)}</td><td>{_safe(item.file_type)}</td>"
            f'<td class="evidence-hash">{_safe(item.sha256_hash)}</td>'
            f"<td>{item.file_size_bytes:,} bytes</td><td>{_safe(status)}</td><td>{_safe(method)}</td>"
            "</tr>"
        )
    st.markdown(
        '<style>.evidence-table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:12px;background:#fff;color:#19344c;border-radius:12px;overflow:hidden}'
        '.evidence-table th,.evidence-table td{padding:9px 7px;border-bottom:1px solid #e2e8e5;text-align:left;vertical-align:top;overflow-wrap:anywhere}'
        '.evidence-table th{color:#0f7772;font-size:10px;text-transform:uppercase;letter-spacing:.06em;background:#eff5f2}'
        '.evidence-table .evidence-hash{font-family:ui-monospace,monospace;font-size:10px;word-break:break-all}'
        '</style><table class="evidence-table"><thead><tr><th>File name</th><th>Type</th><th>SHA-256</th>'
        '<th>Size</th><th>Status</th><th>Extraction method</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption("A missing processing record is shown as ‘not recorded’ rather than inferred to be a failure.")


def _highlight_table(rows: list[tuple[str, Any, Any]]) -> None:
    if not rows:
        st.caption("No extracted fields are recorded.")
        return
    html_rows = []
    for label, printed, qr_value in rows:
        if printed is None:
            status, cls = "QR only", "single"
        elif qr_value is None:
            status, cls = "Printed only", "single"
        elif str(printed).strip().casefold() == str(qr_value).strip().casefold():
            status, cls = "Match", "match"
        else:
            status, cls = "Mismatch", "mismatch"
        html_rows.append(
            f'<tr><td>{_safe(label)}</td><td>{_safe(printed if printed is not None else "—")}</td>'
            f'<td>{_safe(qr_value if qr_value is not None else "—")}</td><td><span class="field-{cls}">{status}</span></td></tr>'
        )
    st.markdown(
        '<style>.field-mismatch{color:#bd3b3b;font-weight:700}.field-match{color:#20805c;font-weight:700}.field-single{color:#a66b0c;font-weight:700}</style>'
        '<div class="table-wrap"><table><thead><tr><th>Field</th><th>Printed</th><th>QR payload</th><th>Comparison</th></tr></thead><tbody>'
        + "".join(html_rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_documents(data: dict[str, Any], db: DatabaseManager, case_id: str) -> None:
    _section_title("Document verifier", "Inspect extracted document data and heuristic image indicators. These are review aids, not authenticity determinations.")
    if not _run_stage_action(db, case_id, "document", "Document Analysis", run_document_analysis):
        st.info("Document results are hidden until a current Document Analysis run completes.")
        return
    evidence_by_id = {item.id: item for item in data["evidence"]}
    docs = data["documents"]
    if not docs:
        st.info("No documents were analyzed. Add identity-document evidence in Evidence Intake, then run Document Analysis.")
        return
    doc_ids = {doc.evidence_id for doc in docs}
    doc_findings = [f for f in data["findings"] if any(eid in doc_ids for eid in f.evidence_ids)]
    finding_counts = {level: sum((f.severity or "").upper() == level for f in doc_findings) for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    summary_cols = st.columns(6)
    summary_cols[0].metric("Documents", len(docs))
    summary_cols[1].metric("Findings", len(doc_findings))
    for col, level in zip(summary_cols[2:], ("CRITICAL", "HIGH", "MEDIUM", "LOW")):
        col.metric(level.title(), finding_counts[level])
    st.markdown('<div class="verifier-note"><b>Interpretation:</b> field mismatches and ELA patterns are potential anomalies for review. They do not establish alteration or wrongdoing.</div>', unsafe_allow_html=True)
    for index, doc in enumerate(docs, start=1):
        item = evidence_by_id.get(doc.evidence_id)
        name = item.file_name if item else doc.evidence_id
        related_findings = [f for f in doc_findings if doc.evidence_id in f.evidence_ids]
        qr_data = doc.qr_data or {}
        all_fields = sorted(set(doc.fields) | set(qr_data))
        with st.container(border=True):
            st.subheader(f"{index}. {name}")
            st.markdown(f'<span class="badge badge-navy">{_safe(doc.doc_type)}</span> &nbsp; <span class="badge badge-green">OCR confidence {doc.raw_ocr_confidence:.0%} · heuristic</span> &nbsp; <span class="badge badge-amber">{len(related_findings)} finding(s)</span>', unsafe_allow_html=True)
            left, right = st.columns(2, gap="large")
            image_exists = bool(item and Path(item.file_path).is_file())
            if image_exists:
                try:
                    heatmap = _opencv_heatmap(item.file_path)
                except Exception as exc:
                    heatmap = None
                    st.warning(f"Could not compute the OpenCV heatmap: {exc}")
                with left:
                    st.image(str(item.file_path), caption="Original evidence", use_container_width=True)
                    with st.expander("Enlarge original document"):
                        st.image(str(item.file_path), caption=name, use_container_width=True)
                with right:
                    if heatmap is not None:
                        st.image(heatmap, caption="OpenCV ELA-style heatmap · heuristic indicator", use_container_width=True)
                        st.caption("Warm/brighter pixels show larger recompression differences; patterns can result from benign processing and are not proof of editing.")
            else:
                st.info("Original evidence preview is unavailable at the persisted path.")
            st.markdown("#### Extracted fields")
            if doc.fields:
                cards = []
                for key, value in doc.fields.items():
                    qr_value = qr_data.get(key)
                    if value in (None, ""):
                        icon, label, cls = "○", "Missing", "badge-amber"
                    elif qr_value is not None and str(value).strip().casefold() != str(qr_value).strip().casefold():
                        icon, label, cls = "!", "Potential mismatch", "badge-red"
                    elif qr_value is not None:
                        icon, label, cls = "✓", "Consistent", "badge-green"
                    else:
                        icon, label, cls = "•", "No QR comparison", "badge-navy"
                    cards.append(f'<div class="field-card"><div class="label">{_safe(key)}</div><div class="value">{_safe(value)}</div><span class="badge {cls}">{icon} {_safe(label)}</span></div>')
                st.markdown('<div class="field-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)
            else:
                st.info("No printed fields were extracted from this document.")
            st.markdown("#### QR decoded vs printed fields")
            _highlight_table([(key, doc.fields.get(key), qr_data.get(key)) for key in all_fields])
            if not qr_data:
                st.caption("No QR payload was decoded or persisted; printed-only status is not interpreted as a mismatch.")
            if related_findings:
                st.markdown("#### Findings for this document")
                for finding in related_findings:
                    sev = (finding.severity or "MEDIUM").upper()
                    badge = "badge-red" if sev in {"HIGH", "CRITICAL"} else "badge-amber" if sev == "MEDIUM" else "badge-green"
                    with st.expander(f"{finding.title} · {sev} · {finding.confidence:.0%} confidence (heuristic)"):
                        st.markdown(f'<span class="badge {badge}">{_safe(sev)}</span> &nbsp; <span class="badge badge-navy">{finding.confidence:.0%} confidence · heuristic</span>', unsafe_allow_html=True)
                        st.write(finding.description)
                        if finding.details:
                            st.caption("Recorded evidence details: " + _audit_detail_text(finding.details))


def _render_chat(data: dict[str, Any], db: DatabaseManager, case_id: str) -> None:
    _section_title("Screenshot verifier", "Review recorded chat evidence and image-derived indicators. The visualization is heuristic and is never proof of alteration.")
    if not _run_stage_action(db, case_id, "screenshot", "Screenshot & Chat Analysis", run_screenshot_analysis):
        st.info("Screenshot, chat-export, and event-log results are hidden until this module runs for current evidence.")
        return
    evidence_by_id = {item.id: item for item in data["evidence"]}
    screenshots = [chat for chat in data["chats"] if chat.source_type.upper() == "SCREENSHOT"]
    exports = [chat for chat in data["chats"] if chat.source_type.upper() != "SCREENSHOT"]
    inverted = [finding for finding in data["findings"] if finding.anomaly_type == AnomalyType.OUT_OF_ORDER_TIMESTAMP.value]
    st.markdown('<div class="verifier-note"><b>Heuristic indicator only.</b> Heatmap colours encode recompression differences; red boxes mark the message locator recorded by analysis. Neither visual feature proves editing or intent.</div>', unsafe_allow_html=True)
    if screenshots:
        for idx, chat in enumerate(screenshots):
            item = evidence_by_id.get(chat.evidence_id)
            st.subheader(f"Screenshot preview · {item.file_name if item else 'image source'}")
            if not item or not Path(item.file_path).is_file():
                st.warning("Screenshot record exists, but the original image is unavailable.")
                continue
            choice = st.radio("Preview mode", ["Original", "ELA heatmap", "Anomaly overlay"], horizontal=True, key=f"screenshot_view_{case_id}_{idx}")
            flagged = next((finding for finding in inverted if item.id in finding.evidence_ids), None)
            if choice == "Original":
                st.image(item.file_path, caption="Original evidence image", use_container_width=True)
            elif choice == "ELA heatmap":
                try:
                    st.image(_opencv_heatmap(item.file_path), caption="OpenCV ELA-style heatmap · heuristic indicator", use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not compute the heatmap: {exc}")
            else:
                try:
                    st.image(_anomaly_overlay(item.file_path, chat.messages, flagged), caption="Recorded flagged message bounding box · heuristic locator", use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not compute the anomaly overlay: {exc}")
            st.markdown('<div class="legend-row"><span class="legend-chip"><i class="legend-dot" style="background:#173d7a"></i> lower recompression difference</span><span class="legend-chip"><i class="legend-dot" style="background:#e74728"></i> higher recompression difference</span><span class="legend-chip"><i class="legend-dot" style="background:#bd2525"></i> flagged message locator</span></div>', unsafe_allow_html=True)
            st.caption("Heatmap is calculated from image pixels with OpenCV (JPEG recompression difference + colormap). Colour intensity is not a calibrated measure and can reflect benign image processing.")

    st.subheader("Parsed conversation · screenshot messages")
    if not screenshots:
        st.info("No screenshot conversation is available yet. Add screenshot evidence and run Screenshot & Chat Analysis.")
    else:
        for chat in screenshots:
            for message in chat.messages:
                outgoing = " outgoing" if message.bounding_box and message.bounding_box.get("x", 0) > 300 else ""
                st.markdown(f'<div class="chat-bubble{outgoing}"><div class="chat-meta">{_safe(message.sender)} · {_safe(message.timestamp_str)}</div><div class="chat-content">{_safe(message.content)}</div></div>', unsafe_allow_html=True)

    st.subheader("Screenshot vs chat-export comparison")
    screenshot_messages = [m for chat in screenshots for m in chat.messages]
    export_messages = [m for chat in exports for m in chat.messages]
    if not screenshot_messages or not export_messages:
        st.info("A comparison needs both parsed screenshot messages and a parsed chat export.")
    else:
        export_unused = set(range(len(export_messages)))
        compare_rows = []
        for msg in screenshot_messages:
            tokens = set(re.findall(r"[a-z0-9]+", msg.content.casefold()))
            candidates = []
            for j in export_unused:
                other = export_messages[j]
                other_tokens = set(re.findall(r"[a-z0-9]+", other.content.casefold()))
                overlap = len(tokens & other_tokens) / max(1, len(tokens | other_tokens))
                same_sender = msg.sender.casefold() == other.sender.casefold()
                candidates.append((overlap + (0.35 if same_sender else 0), j))
            score, match_idx = max(candidates, default=(0, -1))
            if match_idx >= 0 and score >= 0.25:
                export_unused.discard(match_idx)
                other = export_messages[match_idx]
                equal = re.sub(r"\s+", " ", msg.content).strip().casefold() == re.sub(r"\s+", " ", other.content).strip().casefold()
                compare_rows.append({"Screenshot time": msg.timestamp_str or "—", "Sender": msg.sender, "Screenshot text": msg.content, "Export time": other.timestamp_str or "—", "Export text": other.content, "Review": "Match" if equal else "Text differs"})
            else:
                compare_rows.append({"Screenshot time": msg.timestamp_str or "—", "Sender": msg.sender, "Screenshot text": msg.content, "Export time": "—", "Export text": "Not present in export", "Review": "Screenshot-only"})
        for j in sorted(export_unused):
            other = export_messages[j]
            compare_rows.append({"Screenshot time": "—", "Sender": other.sender, "Screenshot text": "Not present in screenshot parse", "Export time": other.timestamp_str or "—", "Export text": other.content, "Review": "Export-only"})
        comparison_html = []
        for row in compare_rows:
            cls = "match" if row["Review"] == "Match" else "diff"
            comparison_html.append(
                f'<tr class="{cls}"><td>{_safe(row["Screenshot time"])}</td><td>{_safe(row["Sender"])}</td>'
                f'<td>{_safe(row["Screenshot text"])}</td><td>{_safe(row["Export time"])}</td>'
                f'<td>{_safe(row["Export text"])}</td><td><b>{_safe(row["Review"])}</b></td></tr>'
            )
        st.markdown(
            '<style>.chat-compare{width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid #e1e7e4;border-radius:12px;overflow:hidden;font-size:12px}'
            '.chat-compare th,.chat-compare td{padding:10px 9px;border-bottom:1px solid #e5eae7;text-align:left;vertical-align:top;overflow-wrap:anywhere}'
            '.chat-compare th{background:#eef4f1;color:#0f7772;font-size:10px;text-transform:uppercase;letter-spacing:.05em}'
            '.chat-compare tr.diff td{background:#fff4e2;color:#694b14}.chat-compare tr.diff td:last-child{color:#a45d00}.chat-compare tr.match td:last-child{color:#20805c;font-weight:700}</style>'
            '<div style="overflow-x:auto"><table class="chat-compare"><thead><tr><th>Screenshot time</th><th>Sender</th><th>Screenshot text</th><th>Export time</th><th>Export text</th><th>Review</th></tr></thead><tbody>'
            + "".join(comparison_html) + "</tbody></table></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Timestamp order review")
    if inverted:
        for finding in inverted:
            details = finding.details
            msg = details.get("inverted_message", {})
            st.warning(f"Potential order inversion: {finding.title} · {finding.confidence:.0%} confidence (heuristic)")
        chart_messages = [m for m in screenshot_messages if m.timestamp_str]
        chart_x, chart_y, hover = [], [], []
        for idx, message in enumerate(chart_messages, 1):
            parsed = None
            for fmt in ("%I:%M %p", "%H:%M", "%I:%M:%S %p"):
                try:
                    parsed = datetime.strptime(message.timestamp_str.strip(), fmt)
                    break
                except ValueError:
                    pass
            if parsed:
                chart_x.append(idx)
                chart_y.append(parsed.hour * 60 + parsed.minute)
                hover.append(f"{message.timestamp_str} · {message.sender}: {message.content}")
        if chart_x:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_x, y=chart_y, mode="lines+markers+text", text=[m.timestamp_str for m in chart_messages if m.timestamp_str][:len(chart_x)], textposition="top center", customdata=hover, hovertemplate="Message %{x}<br>%{customdata}<extra></extra>", line={"color":"#0f8a83","width":3}, marker={"size":12,"color":"#0f8a83"}, name="Screenshot sequence"))
            for finding in inverted:
                details = finding.details
                prev = details.get("preceding_message", {})
                curr = details.get("inverted_message", {})
                try:
                    y0 = datetime.strptime(prev.get("timestamp", ""), "%I:%M %p")
                    y1 = datetime.strptime(curr.get("timestamp", ""), "%I:%M %p")
                    fig.add_annotation(x=curr.get("index", 0) + 1, y=y1.hour*60+y1.minute, text="timestamp goes backward", showarrow=True, arrowhead=2, arrowcolor="#bd3b3b", font={"color":"#bd3b3b"})
                except ValueError:
                    pass
            fig.update_layout(height=330, margin={"l":10,"r":15,"t":20,"b":10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.75)", font={"color":"#142b45"}, xaxis_title="Visual message order (top to bottom)", yaxis_title="Displayed time", yaxis={"tickmode":"array","tickvals":sorted(set(chart_y)),"ticktext":[f"{v//60:02d}:{v%60:02d}" for v in sorted(set(chart_y))]})
            st.plotly_chart(fig, use_container_width=True, key=f"timestamp_chart_{case_id}")
    amount_contradictions = [
        item for item in data["contradictions"]
        if "AMOUNT" in (item.contradiction_type or "").upper()
    ]
    if amount_contradictions:
        st.subheader("Potential amount differences")
        for item in amount_contradictions:
            st.warning(f"{item.field_name}: {item.observed_fact_a or item.claim_a} ↔ {item.observed_fact_b or item.claim_b}")
            st.caption(f"{item.explanation} · Confidence {item.confidence:.0%} (heuristic)")


def _claim_value(claim: dict[str, Any], observed: str | None) -> str:
    if observed:
        return str(observed)
    for key in ("value", "claim", "text", "statement"):
        if claim.get(key) is not None:
            return str(claim[key])
    return "; ".join(f"{key}: {value}" for key, value in claim.items()) or "Not recorded"


def _render_truth_engine(data: dict[str, Any], db: DatabaseManager, case_id: str) -> None:
    _section_title("Truth Engine", "Cross-source comparisons from the database. Confidence is heuristic, not a calibrated probability.")
    prerequisites = {"document": run_document_analysis, "screenshot": run_screenshot_analysis}
    missing = [module for module in prerequisites if not db.analysis_is_current(case_id, module)]
    if missing:
        names = {"document": "Document Analysis", "screenshot": "Screenshot & Chat Analysis"}
        st.warning("Truth Engine needs current results from both Document Analysis and Screenshot & Chat analysis first.")
        st.caption("Missing or stale prerequisites: " + ", ".join(names[module] for module in missing))
        if st.button("Run prerequisites first", type="primary", key=f"run_prerequisites_{case_id}"):
            results = []
            try:
                with st.spinner("Running the missing prerequisite analyzers…"):
                    for module in missing:
                        results.append(prerequisites[module](case_id, DB_PATH))
                failed = [item["module"] for item in results if item["status"] not in {"completed", "partial"}]
                if failed:
                    st.session_state["flash_message"] = ("error", "Prerequisite stages need attention: " + ", ".join(failed))
                else:
                    st.session_state["flash_message"] = ("success", "Prerequisites finished. Run Truth Engine to correlate the results.")
            except Exception as exc:
                st.session_state["flash_message"] = ("error", f"Prerequisites could not complete: {exc}")
            st.rerun()
        st.info("Truth Engine results remain hidden until both prerequisite modules are current.")
        return

    if not db.analysis_is_current(case_id, "truth_engine"):
        run = data["runs"].get("truth_engine", {})
        if run.get("status") == "stale":
            st.warning("Truth Engine is stale: rerun required because the evidence set changed.")
        if st.button("Run Evidence Correlation", type="primary", key=f"run_truth_engine_{case_id}"):
            try:
                with st.spinner("Correlating current case evidence and rebuilding the timeline…"):
                    result = run_truth_engine(case_id, DB_PATH)
                st.session_state["flash_message"] = (
                    "success",
                    f"Truth Engine completed: {len(result.contradictions)} contradictions; timeline rebuilt.",
                )
            except Exception as exc:
                st.session_state["flash_message"] = ("error", f"Truth Engine could not complete: {exc}")
            st.rerun()
        st.info("Contradictions and timeline indicators appear only after Truth Engine completes.")
        return
    contradictions = data["contradictions"]
    levels = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNSPECIFIED")
    st.subheader("Contradictions by severity")
    for col, level in zip(st.columns(len(levels)), levels):
        col.metric(level.title(), sum((item.severity or "UNSPECIFIED").upper() == level for item in contradictions))
    categories = sorted({item.contradiction_type or "Uncategorized" for item in contradictions})
    if categories:
        st.subheader("Contradictions by category")
        for start in range(0, len(categories), 6):
            for col, category in zip(st.columns(min(6, len(categories) - start)), categories[start:start + 6]):
                col.metric(category.replace("_", " ").title(), sum((item.contradiction_type or "Uncategorized") == category for item in contradictions))
    _render_relationship_graph(db, case_id)
    if not contradictions:
        st.info("No contradictions are recorded for this case.")

    severity_values = sorted({(item.severity or "UNSPECIFIED").upper() for item in contradictions})
    evidence_by_id = {item.id: item for item in data["evidence"]}
    evidence_types = sorted({evidence_by_id[eid].file_type for item in contradictions for eid in item.evidence_ids if eid in evidence_by_id})
    f1, f2, f3 = st.columns(3)
    selected_severity = f1.selectbox("Filter by severity", ["All severities", *severity_values], key=f"truth_severity_{case_id}")
    selected_category = f2.selectbox("Filter by category", ["All categories", *categories], key=f"truth_category_{case_id}")
    selected_type = f3.selectbox("Filter by evidence type", ["All evidence types", *evidence_types], key=f"truth_evidence_type_{case_id}")
    visible = []
    for item in contradictions:
        item_types = {evidence_by_id[eid].file_type for eid in item.evidence_ids if eid in evidence_by_id}
        if selected_severity != "All severities" and (item.severity or "UNSPECIFIED").upper() != selected_severity:
            continue
        if selected_category != "All categories" and (item.contradiction_type or "Uncategorized") != selected_category:
            continue
        if selected_type != "All evidence types" and selected_type not in item_types:
            continue
        visible.append(item)
    severity_colors = {
        "CRITICAL": "#ff5f66",
        "HIGH": "#ff945e",
        "MEDIUM": "#e4bd57",
        "LOW": "#5ac8a0",
        "UNSPECIFIED": "#8da2b8",
    }
    st.caption(f"Showing {len(visible)} of {len(contradictions)} contradictions; findings are counted separately in Overview.")
    for item in visible:
        fallback_a = evidence_by_id.get(item.evidence_ids[0]).file_name if item.evidence_ids and item.evidence_ids[0] in evidence_by_id else "Source A not recorded"
        fallback_b = evidence_by_id.get(item.evidence_ids[1]).file_name if len(item.evidence_ids) > 1 and item.evidence_ids[1] in evidence_by_id else "Source B not recorded"
        source_a = (item.source_a_ref or {}).get("file_name") or item.claim_a.get("source") or fallback_a
        source_b = (item.source_b_ref or {}).get("file_name") or item.claim_b.get("source") or fallback_b
        fact_a = _claim_value(item.claim_a, item.observed_fact_a)
        fact_b = _claim_value(item.claim_b, item.observed_fact_b)
        severity = (item.severity or "UNSPECIFIED").upper()
        color = severity_colors.get(severity, severity_colors["UNSPECIFIED"])
        limitations = item.limitations or ["No item-specific limitation was recorded for this comparison."]
        limits_html = "".join(f"<li>{_safe(value)}</li>" for value in limitations)
        card = (
            f'<article class="contra-card" style="--severity:{color}">'
            f'<div class="contra-top"><span class="contra-type">{_safe(item.contradiction_type or "Contradiction")}</span>'
            f'<span class="contra-severity">{_safe(severity)}</span>'
            f'<span class="contra-confidence">{item.confidence:.0%} confidence · heuristic</span></div>'
            f'<h3>{_safe(item.field_name)}</h3>'
            f'<div class="contra-sources"><div><small>Source A · {_safe(source_a)}</small><p>{_safe(fact_a)}</p></div>'
            f'<div><small>Source B · {_safe(source_b)}</small><p>{_safe(fact_b)}</p></div></div>'
            f'<p class="contra-explanation">{_safe(item.explanation)}</p>'
            f'<strong>Limitations</strong><ul>{limits_html}</ul></article>'
        )
        st.markdown(
            '<style>.contra-card{border:1px solid #e1e7e4;border-left:5px solid var(--severity);background:#fff;'
            'padding:18px 20px;border-radius:15px;margin:12px 0 16px;color:#19344c;box-shadow:0 7px 20px rgba(20,43,69,.055)}.contra-top{display:flex;gap:10px;'
            'align-items:center;flex-wrap:wrap}.contra-type{color:#0f7772;font-size:12px}.contra-severity{color:var(--severity);'
            'font-weight:700;font-size:12px}.contra-confidence, .contra-card small{color:#65798b;font-size:12px}.contra-card h3{margin:10px 0}'
            '.contra-sources{display:grid;grid-template-columns:1fr 1fr;gap:10px}.contra-sources>div{background:#f5f7f5;'
            'border:1px solid #e2e8e5;border-radius:11px;padding:12px;overflow-wrap:anywhere}.contra-sources p{margin:6px 0 0}'
            '.contra-explanation{margin:13px 0}.contra-card li{color:#586d80}@media(max-width:700px){.contra-sources{grid-template-columns:1fr}}</style>'
            + card,
            unsafe_allow_html=True,
        )


    _render_interactive_timeline(data["timeline"], case_id, key_prefix="truth")


def _render_relationship_graph(db: DatabaseManager, case_id: str) -> None:
    st.subheader("Evidence relationship graph")
    graph = build_evidence_graph(case_id, db)
    if graph.number_of_nodes() == 0:
        st.info("No graphable evidence relationships are recorded.")
        return
    st.caption(f"{graph.number_of_nodes()} nodes · {graph.number_of_edges()} relationships")
    st.markdown(
        '<div class="graph-legend"><span>● Person</span><span>■ Document</span><span>◆ Message</span><span>⬢ Event</span>'
        '<span><i style="background:#22a06b"></i> Confirmed / supports</span>'
        '<span><i class="dash" style="border-color:#d39116"></i> Possible match</span>'
        '<span><i style="background:#d34444"></i> Contradiction</span>'
        '<style>.graph-legend{display:flex;gap:15px;flex-wrap:wrap;color:#52697e;font-size:12px;margin:8px 0 14px}'
        '.graph-legend i{display:inline-block;width:22px;height:3px;margin:0 6px 3px 0;vertical-align:middle}'
        '.graph-legend i.dash{height:0;border-top:3px dashed #d39116;background:none}</style></div>',
        unsafe_allow_html=True,
    )
    figure = build_plotly_figure(graph, case_id)
    figure.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#f5f7f5",
        font={"color": "#142b45"}, title={"font": {"color": "#142b45"}},
        legend={"bgcolor": "rgba(255,255,255,.9)", "bordercolor": "#dfe7e3", "font": {"color": "#425a70"}},
    )
    for trace in figure.data:
        name = str(trace.name or "").upper()
        if name == "SAME_ENTITY":
            trace.update(name="Possible match", line={"color": "#d39116", "dash": "dash", "width": 2.5})
        elif name == "SUPPORTS":
            trace.update(name="Confirmed / supports", line={"color": "#22a06b", "width": 2.5})
        elif name.startswith("CONTRADICTS"):
            trace.update(name="Contradiction", line={"color": "#d34444", "width": 3.5})
        elif trace.mode and "text" in trace.mode:
            trace.update(textfont={"color": "#213c53", "size": 10})
    st.plotly_chart(figure, use_container_width=True, key=f"truth_graph_{case_id}")

    node_ids = list(graph.nodes)
    def label_for(node_id: str) -> str:
        attrs = graph.nodes[node_id]
        return f"{attrs.get('node_type', 'Evidence')} · {attrs.get('label', node_id)}"
    selected = st.selectbox("Inspect graph node", node_ids, format_func=label_for, key=f"truth_node_{case_id}")
    attrs = graph.nodes[selected]
    evidence_ids = [attrs.get("evidence_id")] if attrs.get("evidence_id") else []
    neighbors = set(graph.predecessors(selected)) | set(graph.successors(selected))
    evidence_ids.extend(graph.nodes[node].get("evidence_id") for node in neighbors if graph.nodes[node].get("evidence_id"))
    evidence_by_id = {item.id: item for item in db.list_evidence(case_id)}
    related = [evidence_by_id[eid] for eid in dict.fromkeys(evidence_ids) if eid in evidence_by_id]
    fields = [("Node type", attrs.get("node_type", "Evidence")), ("Label", attrs.get("label", selected))]
    for key in ("file_type", "sha256", "size_bytes", "sender", "timestamp", "source_type", "description", "content"):
        value = attrs.get(key)
        if value not in (None, ""):
            fields.append((key.replace("_", " ").title(), value))
    for evidence in related:
        fields.append(("Evidence source", f"{evidence.file_name} · {evidence.file_type} · {evidence.file_size_bytes:,} bytes · SHA-256 {evidence.sha256_hash}"))
    st.markdown('<div class="field-grid">' + "".join(
        f'<div class="field-card"><div class="label">{_safe(name)}</div><div class="value">{_safe(value)}</div></div>'
        for name, value in fields
    ) + "</div>", unsafe_allow_html=True)


def _render_interactive_timeline(timeline: list[Any], case_id: str, key_prefix: str) -> None:
    st.subheader("Timeline")
    if not timeline:
        st.info("No timeline events are recorded for this case.")
        return
    view = st.radio("Timeline view", ["Visual Timeline", "Timeline Table"], horizontal=True, key=f"{key_prefix}_timeline_mode_{case_id}")
    if view == "Visual Timeline":
        category_colors = {"SYSTEM_LOG": "#386c91", "CHAT_MESSAGE": "#8060a6", "TRANSACTION": "#19865e", "FILE_ACCESS": "#b27a14"}
        plotted = []
        for event in timeline:
            if not event.parsed_datetime:
                continue
            try:
                position = datetime.fromisoformat(event.parsed_datetime)
            except ValueError:
                continue
            inferred = bool(event.metadata.get("date_inferred"))
            uncertain = inferred or event.confidence < 0.8 or bool(event.metadata.get("timestamp_uncertain"))
            plotted.append((event, position, inferred, uncertain))
        fig = go.Figure()
        for category in sorted({record[0].event_type for record in plotted}):
            records = [record for record in plotted if record[0].event_type == category]
            fig.add_trace(go.Scatter(
                x=[record[1] for record in records], y=[category] * len(records), mode="markers",
                marker={
                    "size": [17 if record[0].is_anomaly else 11 for record in records],
                    "color": ["#d34444" if record[0].is_anomaly else category_colors.get(category, "#708399") for record in records],
                    "symbol": ["diamond-open" if record[3] else "circle" for record in records],
                    "line": {"width": 2, "color": ["#a12e2e" if record[0].is_anomaly else "#ffffff" for record in records]},
                },
                customdata=[[record[0].timestamp_str, record[0].actor, record[0].description, record[0].source_reference, record[0].confidence, "date inferred" if record[2] else ("uncertain" if record[3] else "recorded")] for record in records],
                hovertemplate="Parsed position: %{x}<br>Original source timestamp: %{customdata[0]}<br>Actor: %{customdata[1]}<br>%{customdata[2]}<br>Source: %{customdata[3]}<br>Confidence: %{customdata[4]:.0%}<br>Timestamp: %{customdata[5]}<extra>%{y}</extra>",
                name=category,
            ))
        fig.update_layout(
            height=370, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
            font={"color": "#142b45"}, margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis={"title": "Parsed chronology · original timestamp retained in hover", "type": "date"},
            yaxis={"title": "Event category", "autorange": "reversed"}, hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_timeline_plot_{case_id}", config={"scrollZoom": True, "displayModeBar": True})
        st.caption("Red, larger markers flag suspicious or contradiction-linked events; hollow diamonds mark inferred or uncertain timestamps. Chart positions use stored parsed values only; original source timestamps remain unchanged in hover and the table.")
        unplotted = sum(not event.parsed_datetime for event in timeline)
        if unplotted:
            st.info(f"{unplotted} event(s) without a stored parsed timestamp are not assigned an invented chart position; see Timeline Table.")
    else:
        search = st.text_input("Search timeline", key=f"{key_prefix}_timeline_search_{case_id}").casefold().strip()
        rows = []
        for event in timeline:
            quality = "Inferred date" if event.metadata.get("date_inferred") else ("Uncertain" if event.confidence < 0.8 or event.metadata.get("timestamp_uncertain") else "Recorded")
            row = {
                "Source timestamp (original)": event.timestamp_str,
                "Parsed timestamp": event.parsed_datetime or "Not parsed",
                "Timestamp basis": quality,
                "Category": event.event_type, "Actor": event.actor, "Event": event.description,
                "Source": event.source_reference, "Confidence": f"{event.confidence:.0%}",
                "Indicator": "Suspicious / contradiction-linked" if event.is_anomaly else "—",
            }
            if not search or search in " ".join(map(str, row.values())).casefold():
                rows.append(row)
        st.caption("Sortable table; original source timestamps are displayed unchanged. Search filters the stored timeline records.")
        st.dataframe(rows, use_container_width=True, hide_index=True, height=min(650, 44 + 42 * max(1, len(rows))))


def _is_post_transaction_modification(event: Any, timeline: list[Any]) -> bool:
    description = event.description.lower()
    is_modification = event.event_type.upper() == "SYSTEM_LOG" and any(
        term in description for term in ("modified", "modification", "file change")
    )
    if not is_modification or not event.is_anomaly or not event.parsed_datetime:
        return False
    return any(
        candidate.parsed_datetime
        and candidate.parsed_datetime < event.parsed_datetime
        and (
            str(candidate.metadata.get("component", "")).lower() == "transaction"
            or "transaction" in candidate.description.lower()
        )
        for candidate in timeline
    )


def _render_timeline_graph(data: dict[str, Any], db: DatabaseManager, case_id: str) -> None:
    _section_title("Timeline & Graph", "Chronological events and relationships are reconstructed from evidence persisted in SQLite.")
    if not db.analysis_is_current(case_id, "truth_engine"):
        st.info("Timeline and relationship graph are hidden until current prerequisites and Truth Engine analysis complete. Run it in section 5.")
        return
    _render_interactive_timeline(data["timeline"], case_id, key_prefix="timeline")
    _render_relationship_graph(db, case_id)


def _audit_detail_text(details: dict[str, Any]) -> str:
    rendered = []
    for key, value in details.items():
        if isinstance(value, dict):
            value_text = "; ".join(f"{subkey}={subvalue}" for subkey, subvalue in value.items())
        elif isinstance(value, (list, tuple)):
            value_text = ", ".join(str(item) for item in value)
        else:
            value_text = str(value)
        rendered.append(f"{key}: {value_text}")
    return " · ".join(rendered) or "No additional details recorded"


def _render_reports_audit(data: dict[str, Any], db: DatabaseManager, case_id: str) -> None:
    _section_title("Reports & Audit", "Generate a current self-contained report and search this case’s persisted history.")
    if db.analysis_is_current(case_id, "truth_engine"):
        try:
            report_html = generate_html_report(case_id, db)
        except Exception as exc:
            st.error(f"Report is unavailable for this case: {exc}")
        else:
            st.download_button(
                "Download HTML report",
                data=report_html,
                file_name=f"report_{case_id}.html",
                mime="text/html",
                type="primary",
                help="Downloads a self-contained report generated from SQLite records.",
                on_click=lambda: record_report_export(case_id, db, f"report_{case_id}.html"),
            )
    else:
        st.info("A report is available after document, screenshot/chat, and Truth Engine stages complete for current evidence.")

    st.subheader("Searchable case history")
    search = st.text_input("Search history", key=f"case_history_search_{case_id}")
    history = db.search_audit_history(search, case_id=case_id)
    if not history:
        st.info("No matching audit entries are recorded for this case.")
        return
    evidence_by_id = {item.id: item.file_name for item in data["evidence"]}
    rows = []
    for entry in history:
        rows.append({
            "Timestamp": entry.get("timestamp", ""),
            "Action": entry.get("action", ""),
            "Evidence": evidence_by_id.get(entry.get("evidence_id"), entry.get("evidence_id") or "—"),
            "Details": _audit_detail_text(entry.get("details") or {}),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=min(650, 42 + 44 * len(rows)))


def _history_table(db: DatabaseManager, search: str, case_id: str | None = None, key_prefix: str = "history") -> None:
    entries = db.search_audit_history(search, case_id=case_id)
    if not entries:
        st.info("No matching history entries.")
        return
    rows = [
        {
            "Timestamp": item["timestamp"],
            "Case": f"{item['case_id']} · {item['case_title']}",
            "Action": item["action"],
            "Details": _audit_detail_text(item["details"]),
        }
        for item in entries
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True, height=min(520, 42 + 42 * len(rows)))


def _render_home(db: DatabaseManager) -> None:
    _section_title("Home", "Create a case or open a workspace. Case content is not shown until you open that case.")
    metrics = db.get_home_metrics()
    cols = st.columns(4)
    for col, label, key in zip(
        cols,
        ("Total cases", "Evidence files", "Completed analyses", "Recent activity · 7 days"),
        ("total_cases", "evidence_files", "completed_analyses", "recent_activity"),
    ):
        col.metric(label, metrics[key])

    with st.form("create_case_form", clear_on_submit=True):
        st.subheader("Create New Case")
        case_id = st.text_input("Case ID", max_chars=80, placeholder="e.g. CASE_2026_01")
        title = st.text_input("Case name", max_chars=200, placeholder="Investigation name")
        description = st.text_area("Description", max_chars=2000)
        create_clicked = st.form_submit_button("Create New Case", type="primary")
    if create_clicked:
        case_id = case_id.strip()
        if not _valid_case_id(case_id):
            st.error("Use 1–80 letters, numbers, hyphens, or underscores for the case ID.")
        elif not title.strip():
            st.error("Enter a case name.")
        elif db.get_case_summary(case_id):
            st.error("That case ID already exists. Open it from the case gallery.")
        else:
            db.create_case(case_id, title.strip(), description.strip())
            st.session_state["active_case_id"] = case_id
            st.session_state["route"] = "case"
            st.session_state[f"workspace_{case_id}"] = NAV_ITEMS[0]
            st.rerun()

    st.subheader("Case gallery")
    cases = db.list_cases()
    if not cases:
        st.info("No cases yet. Create a new case above to get started.")
    else:
        for case in cases:
            with st.container(border=True):
                cols = st.columns([1.3, 2.2, 1.5, 1, 1, 2, 0.8])
                cols[0].markdown(f"**ID**\n\n`{_safe(case['case_id'])}`")
                cols[1].markdown(f"**Name**\n\n{_safe(case['title'])}")
                cols[2].markdown(f"**Created**\n\n{_safe(case['created_at'])}")
                cols[3].markdown(f"**Evidence**\n\n{case['evidence_count']}")
                cols[4].markdown(f"**Status**\n\n{_safe(case['status'])}")
                cols[5].markdown(f"**Last activity**\n\n{_safe(case['last_activity'])}")
                if cols[6].button("Open", key=f"open_case_{case['case_id']}"):
                    st.session_state["active_case_id"] = case["case_id"]
                    st.session_state["route"] = "case"
                    st.rerun()

    st.subheader("Recent activity · all cases")
    search = st.text_input("Search history", key="home_history_search")
    _history_table(db, search, key_prefix="home")


def main() -> None:
    _inject_theme()
    st.markdown("<div class='brand-line'>DRISHTI-SUTRA <span>OFFLINE FORENSIC WORKSPACE</span></div>", unsafe_allow_html=True)
    db = DatabaseManager(DB_PATH)
    st.session_state.setdefault("route", "home")
    st.session_state.setdefault("active_case_id", None)
    flash = st.session_state.pop("flash_message", None)

    if st.session_state["route"] != "case":
        with st.sidebar:
            st.markdown("## ◈ Drishti-Sutra")
            st.caption("Case management · local SQLite workspace")
        if flash:
            getattr(st, flash[0])(flash[1])
        _render_home(db)
        return

    case_id = st.session_state.get("active_case_id")
    summary = db.get_case_summary(case_id) if case_id else None
    if not summary:
        st.session_state["route"] = "home"
        st.session_state["active_case_id"] = None
        st.rerun()
        return

    with st.sidebar:
        st.markdown("## ◈ Drishti-Sutra")
        st.caption("Case workspace · local SQLite database")
        if st.button("← Back to Home", use_container_width=True, key="back_to_home"):
            st.session_state["route"] = "home"
            st.session_state["active_case_id"] = None
            st.rerun()
        st.divider()
        selected = st.radio("Workspace", NAV_ITEMS, key=f"workspace_{case_id}")

    st.header(summary.title)
    st.caption(f"Case `{case_id}` · Created {summary.created_at} · Status {summary.status}")
    if flash:
        getattr(st, flash[0])(flash[1])
    if shutil.which("tesseract") is None:
        st.warning(
            "Do not automatically declare any evidence as forged or fraudulent; report only detected anomalies and inconsistencies, as they do not necessarily prove forgery."
        )

    data = _load_case_data(db, case_id)
    if selected == NAV_ITEMS[0]:
        _render_overview(data, case_id)
    elif selected == NAV_ITEMS[1]:
        _render_evidence(data, db, case_id)
    elif selected == NAV_ITEMS[2]:
        _render_documents(data, db, case_id)
    elif selected == NAV_ITEMS[3]:
        _render_chat(data, db, case_id)
    elif selected == NAV_ITEMS[4]:
        _render_truth_engine(data, db, case_id)
    elif selected == NAV_ITEMS[5]:
        _render_timeline_graph(data, db, case_id)
    elif selected == NAV_ITEMS[6]:
        _render_reports_audit(data, db, case_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error(f"Dashboard encountered an issue while loading this view: {exc}")
