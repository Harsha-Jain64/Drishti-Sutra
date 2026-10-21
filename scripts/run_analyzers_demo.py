import json
import sys
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager
from src.ingestion import EvidenceIngestionService
from src.analyzers.document_analyzer import DocumentAuthenticator
from src.analyzers.screenshot_analyzer import ScreenshotAuthenticator
from src.analyzers.chat_export_parser import parse_chat_export_file


def main():
    print("=" * 70)
    print("DRISHTI-SUTRA: Running Forensic Ingestion & Analyzers (Modules 1 & 2)")
    print("=" * 70)

    db_path = Path("data/drishti_sutra.db")
    if db_path.exists():
        db_path.unlink()  # Start with fresh database

    db = DatabaseManager(db_path)
    case_id = "Case_DS_001"
    db.create_case(case_id, "Financial Dispute & Tampering Investigation", "Initial triage")

    # 1. Ingestion
    ingestor = EvidenceIngestionService(db)
    items = ingestor.ingest_directory(case_id, "sample_data/Case_DS_001")
    print(f"\n[+] Ingested {len(items)} evidence artifacts into database:")
    for item in items:
        print(f"    - {item.file_name:<28} | Type: {item.file_type:<15} | SHA-256: {item.sha256_hash[:16]}...")

    # 2. Module 1: Document Authenticator
    doc_analyzer = DocumentAuthenticator()
    print("\n" + "-" * 70)
    print("MODULE 1: DOCUMENT & KYC AUTHENTICATOR")
    print("-" * 70)

    for item in items:
        if item.file_type == "kyc_doc":
            print(f"\n[*] Analyzing Document: {item.file_name}")
            doc, findings = doc_analyzer.analyze_document(item)
            db.insert_extracted_document(doc)
            for f in findings:
                db.insert_finding(f)

            print(f"    Doc Type Classified: {doc.doc_type}")
            print(f"    Extracted Fields:     {doc.fields}")
            if doc.qr_data:
                print(f"    Decoded QR Payload:  {doc.qr_data}")
            print(f"    ELA Overlay Saved:   {doc.metadata.get('ela_overlay_path')}")
            print(f"    ELA Anomaly Metrics: {doc.metadata.get('ela_details')}")
            if findings:
                print(f"    [!] Anomalies Flagged ({len(findings)}):")
                for f in findings:
                    print(f"        -> [{f.severity}] {f.title} (Confidence: {f.confidence*100:.1f}%)")
                    print(f"           {f.description}")
            else:
                print("    [OK] No internal document anomalies detected.")

    # 3. Module 2: Screenshot Authenticator & Chat Export Parser
    screenshot_analyzer = ScreenshotAuthenticator()
    print("\n" + "-" * 70)
    print("MODULE 2: SCREENSHOT AUTHENTICATOR & CHAT PARSER")
    print("-" * 70)

    for item in items:
        if item.file_type == "chat_screenshot":
            print(f"\n[*] Analyzing Screenshot: {item.file_name}")
            chat, findings = screenshot_analyzer.analyze_screenshot(item)
            db.insert_extracted_chat(chat)
            for f in findings:
                db.insert_finding(f)

            print(f"    Total Messages Segmented: {len(chat.messages)}")
            for idx, msg in enumerate(chat.messages, 1):
                amt_str = f" [Amounts: {msg.amounts}]" if msg.amounts else ""
                print(f"      {idx}. [{msg.timestamp_str}] {msg.sender}: {msg.content}{amt_str}")

            if findings:
                print(f"    [!] Screenshot Anomalies Flagged ({len(findings)}):")
                for f in findings:
                    print(f"        -> [{f.severity}] {f.title} (Confidence: {f.confidence*100:.1f}%)")
                    print(f"           {f.description}")

        elif item.file_type == "chat_export":
            print(f"\n[*] Parsing Companion Chat Export: {item.file_name}")
            chat = parse_chat_export_file(item)
            db.insert_extracted_chat(chat)
            print(f"    Total Messages Parsed: {len(chat.messages)}")
            for idx, msg in enumerate(chat.messages, 1):
                amt_str = f" [Amounts: {msg.amounts}]" if msg.amounts else ""
                print(f"      {idx}. [{msg.timestamp_str}] {msg.sender}: {msg.content}{amt_str}")

    # Database Summary
    summary = db.get_case_summary(case_id)
    print("\n" + "=" * 70)
    print(f"DATABASE SUMMARY FOR {case_id}:")
    print(f"  Total Ingested Files: {summary.total_evidence_count}")
    print(f"  Total Anomaly Findings: {summary.anomaly_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
