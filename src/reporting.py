"""Offline, print-ready HTML reports assembled exclusively from case-scoped DB records."""

from __future__ import annotations

import base64
import html
import mimetypes
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNSPECIFIED")
SEVERITY_COLORS = {"CRITICAL": "#b42318", "HIGH": "#d04437", "MEDIUM": "#b7791f", "LOW": "#26734d", "UNSPECIFIED": "#68788a"}
TYPE_COLORS = {"Person": "#5570b8", "Document": "#157a76", "Evidence": "#73849a", "Message": "#8055a6", "Event": "#c17d19"}
REL_COLORS = {"CONTRADICTS": "#c43d3d", "SUPPORTS": "#248253", "SAME_ENTITY": "#bf8015", "EXTRACTED_FROM": "#96a2ae", "MENTIONS": "#697d91"}


def _esc(value: Any, fallback: str = "Not recorded") -> str:
    if value is None or value == "":
        value = fallback
    return html.escape(str(value), quote=True)


def _status_text(run: dict[str, Any] | None, current: bool) -> str:
    if not run:
        return "Not run"
    status = str(run.get("status") or "not_run").replace("_", " ").title()
    return f"{status} · current" if current else ("Stale · rerun required" if status.lower() == "stale" else status)


def _safe_json_value(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{_esc(k)}: {_esc(v)}" for k, v in value.items()) or "No values recorded"
    if isinstance(value, (list, tuple)):
        return "; ".join(_esc(v) for v in value) or "No values recorded"
    return _esc(value)


def _image_data(path_value: Any) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _ela_data(path_value: Any) -> str:
    """Create a local JPEG-recompression difference overlay; never treat as proof."""
    try:
        import cv2
        import numpy as np
        path = Path(str(path_value))
        raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        original = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if original is None:
            return ""
        ok, compressed = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return ""
        recompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        difference = cv2.absdiff(original, recompressed)
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        scaled = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        blended = cv2.addWeighted(original, 0.28, heat, 0.72, 0)
        ok, encoded = cv2.imencode(".png", blended)
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""
    except Exception:
        return ""


def _anomaly_overlay_data(path_value: Any, messages: list[Any]) -> str:
    try:
        import cv2
        import numpy as np
        path = Path(str(path_value))
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return ""
        for message in messages:
            box = message.bounding_box or {}
            if not all(key in box for key in ("x", "y", "w", "h")):
                continue
            x, y = int(box["x"]), int(box["y"])
            cv2.rectangle(image, (x, y), (x + int(box["w"]), y + int(box["h"])), (35, 55, 220), 3)
        ok, encoded = cv2.imencode(".png", image)
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""
    except Exception:
        return ""


def _img(src: str, alt: str, cls: str = "preview") -> str:
    return f'<img class="{cls}" src="{src}" alt="{_esc(alt)}">' if src else '<div class="empty">Preview unavailable from the stored evidence path.</div>'


def _stage_banner(label: str, run: dict[str, Any] | None, current: bool) -> str:
    text = _status_text(run, current)
    css = "good" if current else "warn"
    return f'<p class="stage {css}"><b>{_esc(label)} status:</b> {_esc(text)}</p>'


def _bar_chart(title: str, values: list[tuple[str, int, str]], width: int = 720) -> str:
    height = max(90, 34 + len(values) * 34)
    label_width, bar_x, max_bar = 210, 220, 420
    max_value = max((value for _, value, _ in values), default=1) or 1
    rows = []
    for index, (label, value, color) in enumerate(values):
        y = 26 + index * 34
        bar_width = round(max_bar * value / max_value) if value else 0
        rows.append(
            f'<text x="{label_width}" y="{y + 15}" text-anchor="end" class="svg-label">{_esc(label)}</text>'
            f'<rect x="{bar_x}" y="{y}" width="{max(bar_width, 1)}" height="20" rx="5" fill="{color}" opacity=".88"/>'
            f'<text x="{bar_x + max(bar_width, 1) + 8}" y="{y + 15}" class="svg-value">{value}</text>'
        )
    return f'<figure class="chart"><figcaption>{_esc(title)}</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">{"".join(rows)}</svg></figure>'


def _severity_chart(findings: list[Any], contradictions: list[Any]) -> str:
    max_value = max((sum(1 for item in findings if (item.severity or "UNSPECIFIED").upper() == level) for level in SEVERITIES), default=0)
    max_value = max(max_value, *(sum(1 for item in contradictions if (item.severity or "UNSPECIFIED").upper() == level) for level in SEVERITIES), 1)
    width, left, bar_max, row_h = 760, 220, 190, 58
    height = 54 + len(SEVERITIES) * row_h
    out = ['<figure class="chart"><figcaption>Severity distribution · findings and contradictions counted separately</figcaption>', f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Severity distribution">']
    out += ['<rect x="220" y="8" width="12" height="12" fill="#246c89"/><text x="238" y="19" class="svg-value">Findings</text><rect x="330" y="8" width="12" height="12" fill="#b7791f"/><text x="348" y="19" class="svg-value">Contradictions</text>']
    for i, level in enumerate(SEVERITIES):
        y = 38 + i * row_h
        a = sum(1 for item in findings if (item.severity or "UNSPECIFIED").upper() == level)
        b = sum(1 for item in contradictions if (item.severity or "UNSPECIFIED").upper() == level)
        wa, wb = round(bar_max * a / max_value), round(bar_max * b / max_value)
        out.append(f'<text x="205" y="{y + 14}" text-anchor="end" class="svg-label">{level.title()}</text>')
        out.append(f'<rect x="220" y="{y}" width="{max(wa,1)}" height="18" rx="4" fill="#246c89"/><text x="{226+wa}" y="{y+14}" class="svg-value">{a}</text>')
        out.append(f'<rect x="220" y="{y+23}" width="{max(wb,1)}" height="18" rx="4" fill="#b7791f"/><text x="{226+wb}" y="{y+37}" class="svg-value">{b}</text>')
    out.append('</svg></figure>')
    return ''.join(out)


def _timeline_svg(timeline: list[Any]) -> str:
    dated = []
    for event in timeline:
        parsed = event.parsed_datetime
        try:
            dt = datetime.fromisoformat(parsed) if parsed else None
        except ValueError:
            dt = None
        if dt:
            dated.append((event, dt))
    if not dated:
        return '<p class="empty">No parsed timestamps are available for the visual timeline; original timestamp strings remain in the table.</p>'
    start, end = min(dt for _, dt in dated), max(dt for _, dt in dated)
    span = max((end - start).total_seconds(), 1)
    categories = list(dict.fromkeys(event.event_type for event, _ in dated))
    colors = ["#2b7582", "#7862a5", "#b7791f", "#3b7b55", "#b44a42", "#647b95"]
    y_for = {category: 40 + i * 48 for i, category in enumerate(categories)}
    width, height, left, plot_width = 900, 80 + max(len(categories), 1) * 48, 210, 650
    out = [f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Timeline by stored parsed timestamps">']
    for i, category in enumerate(categories):
        y = y_for[category]
        out.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" class="svg-label">{_esc(category)}</text><line x1="{left}" y1="{y}" x2="{left+plot_width}" y2="{y}" stroke="#d7e0e6"/>')
        color = colors[i % len(colors)]
        for event, dt in dated:
            if event.event_type != category:
                continue
            x = left + int((dt - start).total_seconds() / span * plot_width)
            is_uncertain = event.metadata.get("date_inferred") or event.metadata.get("timestamp_uncertain") or event.confidence < .8
            fill = "#c43d3d" if event.is_anomaly else color
            shape = f'<path d="M{x} {y-8} l8 8 -8 8 -8 -8 z" fill="white" stroke="{fill}" stroke-width="2"/>' if is_uncertain else f'<circle cx="{x}" cy="{y}" r="7" fill="{fill}"/>'
            title = f'{event.timestamp_str} | {event.actor} | {event.description} | {event.source_reference} | confidence {event.confidence:.0%}'
            out.append(f'<g><title>{_esc(title)}</title>{shape}</g>')
    out.append(f'<text x="{left}" y="{height-10}" class="svg-value">{_esc(start.isoformat())}</text><text x="{left+plot_width}" y="{height-10}" text-anchor="end" class="svg-value">{_esc(end.isoformat())}</text></svg>')
    out.append('<p class="legend"><span>● category event</span><span>◆ inferred/uncertain timestamp</span><span class="red">red marker = potential anomaly</span> · Original source timestamps are retained in the table and SVG hover titles.</p></div>')
    return ''.join(out)


def _graph_svg(case_id: str, db) -> str:
    try:
        from src.graph_builder import build_evidence_graph
        graph = build_evidence_graph(case_id, db)
    except Exception:
        return '<p class="empty">Relationship graph could not be generated from the stored records.</p>'
    nodes = list(graph.nodes(data=True))
    if not nodes:
        return '<p class="empty">No relationships are recorded for this case.</p>'
    width, height, cx, cy, radius = 900, 620, 450, 310, 230
    count = len(nodes)
    positions = {node_id: (cx + radius * __import__('math').cos(2*__import__('math').pi*i/count), cy + radius * __import__('math').sin(2*__import__('math').pi*i/count)) for i, (node_id, _) in enumerate(nodes)}
    out = [f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Evidence relationship graph">']
    for a, b, data in graph.edges(data=True):
        if a not in positions or b not in positions:
            continue
        x1,y1=positions[a]; x2,y2=positions[b]
        relationship=str(data.get('relationship',''))
        color=REL_COLORS.get(relationship,'#96a2ae')
        dash=' stroke-dasharray="7 5"' if relationship == 'SAME_ENTITY' else ''
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2" opacity=".65"{dash}><title>{_esc(relationship)}</title></line>')
    symbols={'Person':'circle','Document':'rect','Evidence':'rect','Message':'polygon','Event':'polygon'}
    for node_id, attrs in nodes:
        x,y=positions[node_id]; kind=str(attrs.get('node_type','Evidence')); color=TYPE_COLORS.get(kind,'#73849a')
        title=_esc(attrs.get('label',node_id)); label=_esc(str(attrs.get('label',node_id))[:23])
        if symbols.get(kind)=='circle': mark=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}"/>'
        elif kind in {'Document','Evidence'}: mark=f'<rect x="{x-11:.1f}" y="{y-11:.1f}" width="22" height="22" rx="3" fill="{color}"/>'
        elif kind=='Message': mark=f'<path d="M{x:.1f} {y-13:.1f} l13 13 -13 13 -13 -13 z" fill="{color}"/>'
        else: mark=f'<path d="M{x-11:.1f} {y-6:.1f} l6 -11 h10 l6 11 -6 11 h-10 z" fill="{color}"/>'
        out.append(f'<g><title>{title} · {kind}</title>{mark}<text x="{x:.1f}" y="{y+29:.1f}" text-anchor="middle" class="svg-value">{label}</text></g>')
    out.append('</svg><div class="legend">')
    out.extend(f'<span><i style="background:{color}"></i>{kind}</span>' for kind,color in TYPE_COLORS.items())
    out.extend(f'<span><i style="background:{color}"></i>{rel.replace("_"," ").title()}</span>' for rel,color in REL_COLORS.items())
    out.append('<span>Dashed amber = possible match</span></div></div>')
    return ''.join(out)


def _certificate(case_id: str, evidence: list[Any]) -> str:
    file_rows = ''.join(f'<tr><td>{_esc(item.file_name)}</td><td class="hash">{_esc(item.sha256_hash)}</td></tr>' for item in evidence)
    if not file_rows:
        file_rows = '<tr><td colspan="2">No electronic records are recorded for this case.</td></tr>'
    return f'''<section class="certificate">
<div class="cert-kicker">THE SCHEDULE · [See Section 63(4)(c)]</div><h1>CERTIFICATE</h1>
<div class="draft">DRAFT FOR REVIEW AND COMPLETION. Not a legally executed certificate. Legal applicability must be verified.</div>
<h2>PART A <small>(To be filled by the Party)</small></h2>
<p>Case identifier (verified from case database): <b>{_esc(case_id)}</b></p>
<p>I, <span class="blank">[BLANK — NAME]</span>, son/daughter/spouse of <span class="blank">[BLANK — RELATION / NAME]</span>, residing/employed at <span class="blank">[BLANK — ADDRESS / EMPLOYMENT]</span>, do hereby solemnly affirm and sincerely state and submit as follows:—</p>
<p>I have produced electronic record/output of the digital record taken from the following device/digital record source (tick mark):</p>
<p class="checklist">Computer / Storage Media □ &nbsp; DVR □ &nbsp; Mobile □ &nbsp; Flash Drive □ &nbsp; CD/DVD □ &nbsp; Server □ &nbsp; Cloud □ &nbsp; Other □</p>
<p>Other / specify: <span class="blank">[BLANK — DEVICE OR SOURCE DETAILS]</span><br>Make &amp; Model: <span class="blank">[BLANK]</span> &nbsp; Color: <span class="blank">[BLANK]</span><br>Serial Number: <span class="blank">[BLANK]</span><br>IMEI/UIN/UID/MAC/Cloud ID: <span class="blank">[BLANK]</span><br>Other relevant device/source information: <span class="blank">[BLANK]</span></p>
<p>Electronic records listed below (file names and SHA-256 values transcribed from this case’s database):</p>
<div class="table-wrap"><table><thead><tr><th>File name</th><th>SHA-256 hash value</th></tr></thead><tbody>{file_rows}</tbody></table></div>
<p>Device/source status (select as applicable): Owned □ &nbsp; Maintained □ &nbsp; Managed □ &nbsp; Operated □ &nbsp; <span class="blank">[BLANK — RESPONSIBLE PERSON’S ROLE]</span></p>
<p>Any statements concerning lawful control, regular use, proper operation, ordinary-course inputs, record accuracy, or the manner of production remain for the responsible person to verify and complete. No such fact is asserted by this draft.</p>
<p>Hash algorithm (select): SHA1 □ &nbsp; SHA256 □ &nbsp; MD5 □ &nbsp; Other □ &nbsp; <span class="blank">[BLANK — ALGORITHM / HASH REPORT ATTACHMENT]</span></p>
<p>Name and signature: <span class="blank">[BLANK — NO SIGNATURE PROVIDED]</span><br>Date (DD/MM/YYYY): <span class="blank">[BLANK]</span><br>Time (IST): <span class="blank">[BLANK]</span> &nbsp; Place: <span class="blank">[BLANK]</span></p>
<hr><h2>PART B <small>(To be filled by the Expert)</small></h2>
<p>I, <span class="blank">[BLANK — EXPERT NAME]</span>, son/daughter/spouse of <span class="blank">[BLANK]</span>, residing/employed at <span class="blank">[BLANK]</span>, do hereby solemnly affirm and sincerely state and submit as follows:—</p>
<p>The produced electronic record/output of the digital record was obtained from the following device/digital record source: <span class="blank">[BLANK — DEVICE / SOURCE DETAILS]</span></p>
<p>Make &amp; Model: <span class="blank">[BLANK]</span> &nbsp; Color: <span class="blank">[BLANK]</span><br>Serial Number: <span class="blank">[BLANK]</span><br>IMEI/UIN/UID/MAC/Cloud ID: <span class="blank">[BLANK]</span><br>Other relevant device/source information: <span class="blank">[BLANK]</span></p>
<p>Hash value(s) and algorithm verified by expert: <span class="blank">[BLANK — EXPERT TO COMPLETE]</span></p>
<p>Name, designation and signature: <span class="blank">[BLANK — NO SIGNATURE PROVIDED]</span><br>Date (DD/MM/YYYY): <span class="blank">[BLANK]</span><br>Time (IST): <span class="blank">[BLANK]</span> &nbsp; Place: <span class="blank">[BLANK]</span></p>
<p class="source-note">Template structure follows the Schedule to the Bharatiya Sakshya Adhiniyam, 2023. This generated draft is not legal advice, certification, or a representation that statutory conditions have been met.</p>
</section>'''


def record_report_export(case_id: str, db, file_name: str) -> None:
    """Persist the audit event when the UI's report download is actually clicked."""
    db.log_audit(case_id, "REPORT_EXPORTED", details={"file_name": file_name})


def generate_html_report(case_id: str, db) -> str:
    """Return a self-contained, light-themed report for exactly one SQLite case."""
    summary = db.get_case_summary(case_id)
    if summary is None:
        raise ValueError(f"No case found in database for case_id={case_id!r}")
    evidence = db.list_evidence(case_id)
    documents = db.list_extracted_documents(case_id)
    chats = db.list_extracted_chats(case_id)
    findings = db.list_findings(case_id)
    contradictions = db.list_contradictions(case_id)
    timeline = db.list_timeline_events(case_id)
    logs = db.list_event_log_records_for_case(case_id)
    audit = db.get_audit_trail(case_id)
    runs = db.get_analysis_runs(case_id)
    current = {module: db.analysis_is_current(case_id, module) for module in ("document", "screenshot", "truth_engine")}
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    evidence_by_id = {item.id: item for item in evidence}
    docs_by_id = {item.evidence_id: item for item in documents}
    chats_by_id = {item.evidence_id: item for item in chats}
    finding_by_evidence: dict[str, list[Any]] = {}
    for finding in findings:
        for eid in finding.evidence_ids:
            finding_by_evidence.setdefault(eid, []).append(finding)
    evidence_types = {item.id: item.file_type for item in evidence}
    current_findings = []
    for finding in findings:
        types = {evidence_types[eid] for eid in finding.evidence_ids if eid in evidence_types}
        if not types:
            if current["truth_engine"]:
                current_findings.append(finding)
        elif any(kind == "kyc_doc" and current["document"] for kind in types):
            current_findings.append(finding)
        elif any(kind == "chat_screenshot" and current["screenshot"] for kind in types):
            current_findings.append(finding)
        elif any(kind not in {"kyc_doc", "chat_screenshot"} for kind in types) and current["truth_engine"]:
            current_findings.append(finding)

    # Overview counts are disjoint: unique evidence records, individual findings, contradictions.
    evidence_type_counts = Counter(item.file_type for item in evidence)
    evidence_chart = _bar_chart("Unique evidence items by recorded type", [(name, evidence_type_counts[name], "#287a85") for name in sorted(evidence_type_counts)] or [("No evidence", 0, "#287a85")])
    severity_chart = _severity_chart(current_findings, contradictions if current["truth_engine"] else [])

    evidence_rows = []
    methodology_rows = []
    for item in evidence:
        doc, chat = docs_by_id.get(item.id), chats_by_id.get(item.id)
        item_logs = [record for record in logs if record.evidence_id == item.id]
        processing_status = item.metadata.get("processing_status", "Not recorded")
        method = item.metadata.get("extraction_method")
        if not method and doc:
            method = doc.metadata.get("extraction_method")
        if not method and chat:
            method = chat.metadata.get("extraction_method")
        if not method and item_logs:
            method = "Structured event-log parser" if current["screenshot"] else "Not recorded"
        method = method or "Not recorded"
        if item.file_type == "kyc_doc" and not current["document"]:
            processing_status = "Not analyzed / stale"
        elif item.file_type in {"chat_screenshot", "chat_export", "system_log"} and not current["screenshot"]:
            processing_status = "Not analyzed / stale"
        elif processing_status == "Not recorded" and (doc or chat or item_logs):
            processing_status = "Processed (recorded)"
        evidence_rows.append(f'<tr><td>{_esc(item.file_name)}</td><td>{_esc(item.file_type)}</td><td class="hash">{_esc(item.sha256_hash)}</td><td>{item.file_size_bytes:,} bytes</td><td>{_esc(processing_status)}</td><td>{_esc(method)}</td></tr>')
        methodology_rows.append(f'<tr><td>{_esc(item.file_name)}</td><td>{_esc(item.file_type)}</td><td>{_esc(method)}</td><td>{_esc(processing_status)}</td></tr>')
    evidence_tbody = ''.join(evidence_rows) or '<tr><td colspan="6" class="empty">No evidence is recorded.</td></tr>'
    method_tbody = ''.join(methodology_rows) or '<tr><td colspan="4" class="empty">No evidence files or extraction methods are recorded.</td></tr>'

    # Certificate gets the authoritative DB evidence hashes, but no identity/device/date/signature assertions.
    certificate = _certificate(case_id, evidence)

    # Document verification: fields and QR values shown only when the document stage is current.
    document_sections = []
    if current["document"]:
        for item in evidence:
            if item.file_type != "kyc_doc":
                continue
            doc = docs_by_id.get(item.id)
            original = _image_data(item.file_path)
            heat = _ela_data(item.file_path)
            findings_for_item = finding_by_evidence.get(item.id, []) if current["document"] else []
            field_rows = []
            if doc:
                all_keys = sorted(set(doc.fields) | set((doc.qr_data or {}).keys()))
                for key in all_keys:
                    printed = doc.fields.get(key)
                    qr = (doc.qr_data or {}).get(key)
                    if qr is None:
                        comparison = "No QR value recorded"
                    elif printed is None:
                        comparison = "No printed value recorded"
                    else:
                        comparison = "Consistent" if str(printed).strip().casefold() == str(qr).strip().casefold() else "Potential mismatch"
                    cls = "good" if comparison == "Consistent" else "warn" if "mismatch" in comparison.lower() else ""
                    field_rows.append(f'<tr><td>{_esc(key)}</td><td>{_safe_json_value(printed)}</td><td>{_safe_json_value(qr)}</td><td class="{cls}">{_esc(comparison)}</td></tr>')
                fields_table = ''.join(field_rows) or '<tr><td colspan="4">No extracted fields recorded.</td></tr>'
                extracted = ''.join(f'<div class="field"><b>{_esc(key)}</b><span>{_safe_json_value(value)}</span></div>' for key,value in sorted(doc.fields.items())) or '<p class="empty">No fields extracted.</p>'
                qr_text = _safe_json_value(doc.qr_data) if doc.qr_data else "No decoded QR payload recorded."
                method_text = doc.metadata.get("extraction_method", item.metadata.get("extraction_method", "Not recorded"))
            else:
                fields_table = '<tr><td colspan="4">No document extraction record.</td></tr>'
                extracted = '<p class="empty">No document extraction record.</p>'
                qr_text = "Not recorded"
                method_text = item.metadata.get("extraction_method", "Not recorded")
            finding_html = ''.join(f'<article class="finding"><div class="badges"><b>{_esc(f.anomaly_type)}</b><span class="sev">{_esc(f.severity)}</span><span>Confidence {_confidence(f.confidence)}</span></div><h4>{_esc(f.title)}</h4><p>{_esc(f.description)}</p><details><summary>Finding details and limitations</summary><p>{_safe_json_value(f.details)}</p></details></article>' for f in findings_for_item) or '<p class="empty">No document findings are recorded.</p>'
            document_sections.append(f'''<article class="panel doc-block"><h3>{_esc(item.file_name)}</h3><p>Extraction method: <b>{_esc(method_text)}</b> · SHA-256 <span class="hash">{_esc(item.sha256_hash)}</span></p>
<div class="image-grid"><div><h4>Original preview</h4>{_img(original,item.file_name)}</div><div><h4>ELA-style heatmap · heuristic indicator only</h4>{_img(heat,"ELA-style recompression heatmap for "+item.file_name)}</div></div>
<h4>Extracted fields</h4><div class="field-grid">{extracted}</div><h4>QR decoded versus printed fields</h4><div class="table-wrap"><table><thead><tr><th>Field</th><th>Printed / extracted</th><th>QR decoded</th><th>Comparison</th></tr></thead><tbody>{fields_table}</tbody></table></div><p><b>Decoded QR payload:</b> {_esc(qr_text)}</p><h4>Recorded findings</h4>{finding_html}</article>''')
    else:
        document_sections.append(f'<p class="empty">Document Analysis {_esc(_status_text(runs.get("document"), False))}. Results are omitted because this stage is not current for this evidence set.</p>')

    # Screenshot verifier includes source messages, embedded original/heatmap/boxes, and exact export comparison.
    screenshot_sections = []
    if current["screenshot"]:
        screenshot_items = [item for item in evidence if item.file_type == "chat_screenshot"]
        exports = [chats_by_id[item.id] for item in evidence if item.file_type == "chat_export" and item.id in chats_by_id]
        export_messages = [message for chat in exports for message in chat.messages]
        export_texts = {message.content.strip() for message in export_messages}
        for item in screenshot_items:
            chat = chats_by_id.get(item.id)
            messages = chat.messages if chat else []
            original = _image_data(item.file_path)
            heat = _ela_data(item.file_path)
            overlay = _anomaly_overlay_data(item.file_path, messages)
            parsed_times = [m.parsed_datetime for m in messages]
            inverted = set()
            for i in range(1, len(parsed_times)):
                if parsed_times[i-1] and parsed_times[i] and parsed_times[i] < parsed_times[i-1]:
                    inverted.update((i-1, i))
            message_html = ''.join(f'<article class="message {"flagged" if index in inverted else ""}"><div><b>{_esc(m.sender)}</b><span>{_esc(m.timestamp_str)}</span></div><p>{_esc(m.content)}</p>{"<small>Potential timestamp-order inversion from stored parsed timestamps</small>" if index in inverted else ""}</article>' for index,m in enumerate(messages)) or '<p class="empty">No screenshot messages are recorded.</p>'
            screenshot_findings = finding_by_evidence.get(item.id, [])
            screenshot_findings_html = ''.join(f'<article class="finding"><div class="badges"><b>{_esc(f.anomaly_type)}</b><span class="sev">{_esc(f.severity)}</span><span>Confidence {_confidence(f.confidence)}</span></div><h4>{_esc(f.title)}</h4><p>{_esc(f.description)}</p></article>' for f in screenshot_findings) or '<p class="empty">No screenshot-specific findings are recorded.</p>'
            compare_rows = []
            for index, message in enumerate(messages):
                matched = message.content.strip() in export_texts
                compare_rows.append(f'<tr class="{"match" if matched else "diff"}"><td>{_esc(message.timestamp_str)}</td><td>{_esc(message.content)}</td><td>{"Exact text present" if matched else "No exact text match in stored export"}</td></tr>')
            for message in export_messages:
                if message.content.strip() not in {m.content.strip() for m in messages}:
                    compare_rows.append(f'<tr class="diff"><td>{_esc(message.timestamp_str)}</td><td>—</td><td>Export-only: {_esc(message.content)}</td></tr>')
            compare_table = ''.join(compare_rows) or '<tr><td colspan="3">No screenshot/export message comparison is available.</td></tr>'
            screenshot_sections.append(f'''<article class="panel"><h3>{_esc(item.file_name)}</h3><p>Extraction method: <b>{_esc(chat.metadata.get("extraction_method", "Not recorded") if chat else "Not recorded")}</b> · SHA-256 <span class="hash">{_esc(item.sha256_hash)}</span></p>
<div class="image-grid"><div><h4>Original</h4>{_img(original,item.file_name)}</div><div><h4>ELA-style heatmap · heuristic indicator only</h4>{_img(heat,"ELA-style heatmap for "+item.file_name)}</div><div><h4>Stored message bounding boxes</h4>{_img(overlay,"Stored message boxes for "+item.file_name)}</div></div><p class="legend">Heatmap colours visualize recompression differences only; they are heuristic indicators, not proof of alteration. Red boxes show stored OCR message bounds.</p>
<h4>Screenshot conversation</h4><div class="conversation">{message_html}</div><h4>Screenshot findings</h4>{screenshot_findings_html}<h4>Screenshot versus chat-export comparison</h4><div class="table-wrap"><table><thead><tr><th>Screenshot time</th><th>Screenshot message</th><th>Export comparison</th></tr></thead><tbody>{compare_table}</tbody></table></div></article>''')
        if not screenshot_items:
            screenshot_sections.append('<p class="empty">No screenshot evidence is recorded. Stored chat exports are not visual screenshot sources.</p>')
    else:
        screenshot_sections.append(f'<p class="empty">Screenshot &amp; Chat Analysis {_esc(_status_text(runs.get("screenshot"), False))}. Results are omitted because this stage is not current for this evidence set.</p>')

    truth_current = current["truth_engine"]
    contra_cards = []
    if truth_current:
        for item in contradictions:
            source_a = (item.source_a_ref or {}).get("file_name") or item.claim_a.get("source") or _evidence_name(evidence_by_id, item.evidence_ids[0] if item.evidence_ids else None)
            source_b = (item.source_b_ref or {}).get("file_name") or item.claim_b.get("source") or _evidence_name(evidence_by_id, item.evidence_ids[1] if len(item.evidence_ids)>1 else None)
            val_a = item.observed_fact_a or item.claim_a.get("value") or item.claim_a
            val_b = item.observed_fact_b or item.claim_b.get("value") or item.claim_b
            limits = ''.join(f'<li>{_esc(value)}</li>' for value in item.limitations) or '<li>No item-specific limitation was recorded.</li>'
            contra_cards.append(f'''<article class="contradiction"><div class="badges"><b>{_esc(item.contradiction_type or "Contradiction")}</b><span class="sev">{_esc(item.severity)}</span><span>Confidence {_confidence(item.confidence)} · heuristic</span></div><h3>{_esc(item.field_name)}</h3><div class="compare"><div><small>Source A · {_esc(source_a)}</small><p>{_safe_json_value(val_a)}</p></div><div><small>Source B · {_esc(source_b)}</small><p>{_safe_json_value(val_b)}</p></div></div><p>{_esc(item.explanation)}</p><h4>Stated limitations</h4><ul>{limits}</ul></article>''')
    truth_section = ''.join(contra_cards) or ('<p class="empty">No contradictions are recorded for this current analysis.</p>' if truth_current else f'<p class="empty">Evidence Correlation {_esc(_status_text(runs.get("truth_engine"), False))}. Findings are omitted because the correlation stage is not current.</p>')

    if truth_current:
        timeline_rows=[]
        for event in timeline:
            basis = "Inferred date" if event.metadata.get("date_inferred") else ("Uncertain" if event.metadata.get("timestamp_uncertain") or event.confidence < .8 else "Recorded")
            indicator = "Potential anomaly" if event.is_anomaly else "—"
            timeline_rows.append(f'<tr class="{"diff" if event.is_anomaly else ""}"><td>{_esc(event.timestamp_str)}</td><td>{_esc(event.parsed_datetime,"Not parsed")}</td><td>{_esc(basis)}</td><td>{_esc(event.event_type)}</td><td>{_esc(event.actor)}</td><td>{_esc(event.description)}</td><td>{_esc(event.source_reference)}</td><td>{event.confidence:.0%}</td><td>{indicator}</td></tr>')
        timeline_table=''.join(timeline_rows) or '<tr><td colspan="9">No timeline events are recorded.</td></tr>'
        timeline_section = _timeline_svg(timeline) + f'<div class="table-wrap"><table><thead><tr><th>Original source timestamp</th><th>Parsed timestamp</th><th>Timestamp basis</th><th>Category</th><th>Actor</th><th>Event</th><th>Source</th><th>Confidence</th><th>Indicator</th></tr></thead><tbody>{timeline_table}</tbody></table></div>'
        graph_section = _graph_svg(case_id, db)
    else:
        timeline_section = '<p class="empty">No current Truth Engine/timeline run; no timeline events are presented as current.</p>'
        graph_section = '<p class="empty">No current Truth Engine run; no relationship graph is presented as current.</p>'

    severity_rows=[]
    for level in SEVERITIES:
        f_count=sum(1 for item in current_findings if (item.severity or "UNSPECIFIED").upper()==level)
        c_count=sum(1 for item in contradictions if (item.severity or "UNSPECIFIED").upper()==level) if truth_current else 0
        severity_rows.append(f'<tr><td>{_esc(level.title())}</td><td>{f_count}</td><td>{c_count}</td></tr>')
    severity_table=''.join(severity_rows)

    audit_rows=[]
    for entry in audit:
        item=evidence_by_id.get(entry.get("evidence_id"))
        detail=entry.get("details") or {}
        detail_text='; '.join(f'{_esc(k)}: {_safe_json_value(v)}' for k,v in detail.items()) or "—"
        audit_rows.append(f'<tr><td>{_esc(entry.get("timestamp"))}</td><td>{_esc(entry.get("action"))}</td><td>{_esc(item.file_name if item else "—")}</td><td>{detail_text}</td></tr>')
    audit_table=''.join(audit_rows) or '<tr><td colspan="4">No audit events are recorded.</td></tr>'

    executive=(f"This case contains {len(evidence)} unique evidence item(s), {len(current_findings)} current finding(s), {len(contradictions) if truth_current else 0} current contradiction(s), and {len(timeline) if truth_current else 0} current timeline event(s). These are indicators for review and do not establish authenticity, alteration, wrongdoing, or legal admissibility.")
    stage_summary=''.join(f'<tr><td>{_esc(label)}</td><td>{_esc(_status_text(runs.get(module),current[module]))}</td><td>{_esc(runs.get(module,{}).get("started_at","Not recorded"))}</td><td>{_esc(runs.get(module,{}).get("completed_at","Not recorded"))}</td></tr>' for module,label in (("document","Document Analysis"),("screenshot","Screenshot & Chat"),("truth_engine","Evidence Correlation / Truth Engine")))
    limitations = ["Automated extraction, OCR, QR decoding, recompression heatmaps, and cross-source comparisons can be incomplete or affected by source quality.", "An ELA-style heatmap is a heuristic visualization; local compression differences may have benign causes and are not proof of editing.", "Timestamp parsing may infer dates or rely on assumptions. Original source timestamp strings are preserved alongside parsed values.", "Confidence values are heuristic indicators, not calibrated probabilities.", "No official, government, identity-provider, bank, or messaging-platform verification is claimed."]
    for record in logs:
        if record.status.upper()=="SKIPPED" and record.skip_reason:
            limitations.append(f"{record.file_path or 'Event log'} line {record.line_no} skipped: {record.skip_reason}")
    limitation_html=''.join(f'<li>{_esc(item)}</li>' for item in dict.fromkeys(limitations))

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{_esc(summary.title)} · Drishti-Sutra report</title><style>
:root{{--ink:#1d3045;--navy:#183552;--teal:#176f6a;--paper:#fff;--wash:#f3f5f3;--line:#d9e0e2;--muted:#596b7c;--amber:#fff2d6;--red:#fce9e7;--green:#e6f4eb}}
*{{box-sizing:border-box}}html{{background:#e9eeec}}body{{margin:0;color:var(--ink);background:var(--paper);font:14px/1.5 Arial,Helvetica,sans-serif}}main{{max-width:1100px;margin:auto;padding:34px 48px 64px}}h1,h2,h3,h4{{color:var(--navy);line-height:1.2}}h1{{font-size:30px;margin:8px 0}}h2{{font-size:23px;border-bottom:2px solid #dce7e3;padding-bottom:8px;margin:0 0 18px}}h3{{font-size:17px;margin:12px 0 8px}}h4{{font-size:14px;margin:14px 0 6px}}p{{margin:7px 0}}small,.muted,.source-note{{color:var(--muted)}}
section{{margin:0 0 30px}}.certificate{{font-size:12px;line-height:1.38;border:1px solid #aeb8be;padding:22px 28px;background:white;min-height:950px}}.certificate h1{{text-align:center;font-size:20px;text-decoration:underline;margin:7px 0 12px}}.certificate h2{{font-size:15px;border:0;margin:12px 0 7px}}.certificate h2 small{{display:block;font-size:11px;font-weight:normal}}.cert-kicker{{text-align:center;text-transform:uppercase;letter-spacing:.08em;font-weight:bold}}.draft{{border:2px solid #a52a2a;color:#941b1b;text-align:center;font-weight:bold;padding:9px;margin:12px 0;background:#fff4f2}}.blank{{font-weight:bold;color:#5c6771;background:#f3f4f4;border-bottom:1px solid #404a50;padding:0 3px;white-space:normal}}.checklist{{line-height:1.9}}.certificate .table-wrap{{margin:8px 0}}.certificate table{{min-width:0}}.source-note{{font-size:10px;margin-top:16px}}
.cover{{min-height:880px;display:flex;flex-direction:column;justify-content:center;padding:40px 8px;page-break-before:always;break-before:page}}.brand{{letter-spacing:.14em;color:var(--teal);font-size:12px;font-weight:bold;text-transform:uppercase}}.cover h1{{font-size:42px;max-width:760px}}.cover .subtitle{{font-size:18px;color:var(--muted)}}.cover .meta{{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:16px 0;margin:30px 0;display:grid;grid-template-columns:1fr 1fr;gap:12px}}.report-section{{page-break-before:always;break-before:page;padding-top:8px}}.panel,.finding,.contradiction,.doc-block{{background:var(--wash);border:1px solid var(--line);border-radius:10px;padding:17px;margin:12px 0;break-inside:avoid}}.metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}}.metric{{border:1px solid var(--line);border-radius:8px;padding:14px;background:white}}.metric b{{display:block;font-size:26px;color:var(--teal)}}.metric span{{color:var(--muted);font-size:12px}}.chart{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:12px;margin:12px 0;break-inside:avoid}}.chart figcaption{{font-weight:bold;color:var(--navy);margin-bottom:8px}}svg{{max-width:100%;height:auto;display:block}}.svg-label{{font:12px Arial,sans-serif;fill:#31485d}}.svg-value{{font:11px Arial,sans-serif;fill:#536879}}.legend{{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin:8px 0}}.legend i{{display:inline-block;width:12px;height:12px;margin-right:5px;vertical-align:middle}}
.report-header{{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding:0 0 10px;margin-bottom:26px;color:var(--muted);font-size:11px;font-weight:bold;letter-spacing:.04em}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:10px 0}}table{{width:100%;border-collapse:collapse;min-width:690px}}th,td{{padding:8px 9px;text-align:left;vertical-align:top;border-bottom:1px solid #e3e8e9;overflow-wrap:anywhere}}th{{background:#eaf1ef;color:#234a56;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}tr:last-child td{{border-bottom:0}}tr{{break-inside:avoid}}.hash{{font:11px/1.45 monospace;word-break:break-all}}.badges{{display:flex;gap:8px;flex-wrap:wrap;font-size:11px;color:var(--muted)}}.badges b{{color:var(--teal)}}.sev{{background:var(--amber);border-radius:10px;padding:2px 7px;color:#795315}}.compare,.image-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:12px 0}}.image-grid{{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}}.compare>div{{background:#fff;border:1px solid var(--line);padding:12px;border-radius:8px}}.compare p{{overflow-wrap:anywhere}}.preview{{max-width:100%;max-height:360px;object-fit:contain;border:1px solid var(--line);background:white;border-radius:7px}}.field-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}}.field{{padding:9px;background:white;border:1px solid var(--line);border-radius:7px;display:grid;gap:3px;overflow-wrap:anywhere}}.field b{{color:var(--navy)}}.field span{{font-size:12px}}.stage{{padding:8px 11px;border-radius:7px}}.good,.match{{background:var(--green);color:#21643d}}.warn,.diff{{background:var(--amber);color:#725119}}.red{{color:#a52a2a}}.empty{{padding:14px;border:1px dashed #b8c4c6;border-radius:8px;color:var(--muted);background:#fafbfa}}.conversation{{display:grid;gap:8px}}.message{{padding:10px 13px;border-left:4px solid #9eb5c2;background:white;border-radius:0 7px 7px 0;break-inside:avoid}}.message>div{{display:flex;justify-content:space-between;gap:10px;font-size:12px}}.message p{{margin:4px 0}}.message.flagged{{border-color:#bd3c33;background:#fff3f1}}.message small{{color:#a52a2a}}details{{margin:8px 0}}summary{{cursor:pointer;color:var(--teal);font-weight:bold}}.disclaimer{{border-left:4px solid #c8952c;background:#fff7e8;padding:15px;border-radius:0 8px 8px 0}}.audit{{font-size:11px}}footer{{color:var(--muted);font-size:10px;margin-top:24px;border-top:1px solid var(--line);padding-top:8px}}
@page{{size:A4;margin:16mm 14mm 18mm}}@media print{{html,body{{background:white!important}}main{{max-width:none;padding:0}}.certificate{{min-height:0;page-break-after:always;break-after:page}}.cover{{page-break-before:avoid;break-before:avoid;min-height:250mm}}.report-section{{page-break-before:always;break-before:page}}.panel,.finding,.contradiction,.doc-block,.chart{{box-shadow:none;break-inside:avoid}}a{{color:inherit;text-decoration:none}}.preview{{max-height:105mm}}}}
@media(max-width:700px){{main{{padding:18px}}.metric-grid{{grid-template-columns:repeat(2,1fr)}}.compare{{grid-template-columns:1fr}}.cover h1{{font-size:34px}}}}
</style></head><body><main>
{certificate}
<section class="cover"><header class="report-header"><span>CASE ID · {_esc(case_id)}</span><span>REPORT GENERATED (UTC) · {_esc(generated)}</span></header><div class="brand">Drishti-Sutra · Offline forensic evidence analysis</div><h1>{_esc(summary.title)}</h1><p class="subtitle">Case report · indicators for review, not proof</p><div class="meta"><div><b>Case ID</b><br>{_esc(case_id)}</div><div><b>Report generated (UTC)</b><br>{_esc(generated)}</div><div><b>Case status</b><br>{_esc(summary.status)}</div><div><b>Case created</b><br>{_esc(summary.created_at)}</div></div><p>{_esc(summary.description,"No case description recorded.")}</p><p class="muted">All report content is assembled from this case's SQLite records and locally stored evidence referenced by those records. Images and charts are embedded in this file for offline viewing.</p></section>
<section class="report-section"><h2>Executive summary</h2><div class="panel"><p>{_esc(executive)}</p><p>Confidence values and image-difference visualizations are heuristic. Interpret the listed discrepancies in context and review the source artifacts.</p></div><div class="table-wrap"><table><thead><tr><th>Analysis module</th><th>Recorded status</th><th>Started</th><th>Completed</th></tr></thead><tbody>{stage_summary}</tbody></table></div></section>
<section class="report-section"><h2>Case overview</h2><div class="metric-grid"><div class="metric"><b>{len(evidence)}</b><span>Unique evidence items</span></div><div class="metric"><b>{len(current_findings)}</b><span>Current findings</span></div><div class="metric"><b>{len(contradictions) if truth_current else 0}</b><span>Current contradictions</span></div><div class="metric"><b>{len(timeline) if truth_current else 0}</b><span>Current timeline events</span></div></div>{evidence_chart}{severity_chart}<div class="table-wrap"><table><thead><tr><th>Severity</th><th>Findings</th><th>Contradictions</th></tr></thead><tbody>{severity_table}</tbody></table></div></section>
<section class="report-section"><h2>Evidence inventory</h2><p>Each row represents one unique evidence record in this selected case. SHA-256 values are reproduced from SQLite.</p><div class="table-wrap"><table><thead><tr><th>File name</th><th>Type</th><th>SHA-256</th><th>Size</th><th>Processing status</th><th>Extraction method</th></tr></thead><tbody>{evidence_tbody}</tbody></table></div></section>
<section class="report-section"><h2>Document results</h2>{_stage_banner("Document Analysis",runs.get("document"),current["document"])}{''.join(document_sections)}</section>
<section class="report-section"><h2>Screenshot results</h2>{_stage_banner("Screenshot & Chat Analysis",runs.get("screenshot"),current["screenshot"])}{''.join(screenshot_sections)}</section>
<section class="report-section"><h2>Truth Engine findings</h2>{_stage_banner("Evidence Correlation",runs.get("truth_engine"),truth_current)}<p>Comparisons are potential anomalies for review; they are not proof of alteration or wrongdoing.</p>{truth_section}</section>
<section class="report-section"><h2>Visual timeline and timeline table</h2>{timeline_section}</section>
<section class="report-section"><h2>Relationship graph</h2><p class="legend"><span>● Person</span><span>■ Document / evidence</span><span>◆ Message</span><span>⬢ Event</span><span class="red">Red: contradiction</span><span style="color:#248253">Green: supports</span><span style="color:#bf8015">Amber dashed: possible match</span></p>{graph_section}</section>
<section class="report-section"><h2>Methodology and limitations</h2><p>Per-file methods below are read from persisted evidence and extraction records. A method is shown as “Not recorded” when the database contains no method for that file; no method is inferred from a filename.</p><div class="table-wrap"><table><thead><tr><th>File</th><th>Type</th><th>Recorded extraction method</th><th>Status</th></tr></thead><tbody>{method_tbody}</tbody></table></div><div class="panel"><ul>{limitation_html}</ul></div></section>
<section class="report-section"><h2>Neutral conclusion</h2><div class="disclaimer">This report presents potential anomalies and cross-source differences as indicators for further review. They are not proof of forgery or wrongdoing. No official or government verification is claimed. This report does not determine authenticity, legal admissibility, or any person's intent.</div><p>Conclusions require examination of the original records, the circumstances of acquisition, and relevant technical and legal context.</p></section>
<section class="report-section"><h2>Audit-trail appendix</h2><p>Persisted audit events for case {_esc(case_id)}, in database order.</p><div class="table-wrap audit"><table><thead><tr><th>Timestamp</th><th>Action</th><th>Evidence</th><th>Recorded details</th></tr></thead><tbody>{audit_table}</tbody></table></div><footer>Case {_esc(case_id)} · Generated {_esc(generated)} UTC · Drishti-Sutra offline report</footer></section>
</main></body></html>'''


def _evidence_name(evidence_by_id: dict[str, Any], evidence_id: str | None) -> str:
    item = evidence_by_id.get(evidence_id or "")
    return item.file_name if item else (evidence_id or "Source not recorded")


def _confidence(value: float | None) -> str:
    return f"{float(value) * 100:.0f}% (heuristic)" if value is not None else "Not recorded"
