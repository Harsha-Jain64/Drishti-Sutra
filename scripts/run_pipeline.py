"""CLI command-line interface for the Drishti-Sutra pipeline."""

import argparse
import sys
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager
from src.graph_builder import build_evidence_graph, build_plotly_figure
from src.pipeline import run_case_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Drishti-Sutra Forensic Pipeline")
    parser.add_argument("--case-dir", default="sample_data/Case_DS_001", help="Path to evidence folder or zip")
    parser.add_argument("--db", default="data/drishti_sutra.db", help="Path to SQLite database")
    parser.add_argument("--case-id", default="Case_DS_001", help="Forensic Case Identifier")
    parser.add_argument("--title", default="Case_DS_001 Investigation", help="Case Title")
    parser.add_argument("--graph-out", default="data/graph.html", help="Path to write Plotly HTML graph")
    args = parser.parse_args()

    print("=" * 75)
    print("DRISHTI-SUTRA FORENSIC PIPELINE EXECUTION")
    print("=" * 75)
    print(f"Case ID:        {args.case_id}")
    print(f"Evidence Path:  {args.case_dir}")
    print(f"Database File:  {args.db}")

    report = run_case_pipeline(
        case_dir_or_zip=args.case_dir,
        db_path=args.db,
        case_id=args.case_id,
        title=args.title,
        clear_first=True,
    )

    print("\n[+] INGESTION & ANALYSIS SUMMARY:")
    for f in report.files:
        print(f"    - {f.file_name:<28} | Type: {f.file_type:<15} | Status: {f.status:<9} | Method: {f.extraction_method}")

    te_res = report.truth_engine_result
    print("\n" + "-" * 75)
    print(f"TRUTH ENGINE CORRELATION FINDINGS ({len(te_res.contradictions)} Contradictions, {len(te_res.indicators)} Indicators):")
    print("-" * 75)

    for idx, c in enumerate(te_res.contradictions, 1):
        print(f"\n  {idx}. [{c.severity}] TYPE: {c.contradiction_type} (Confidence: {c.confidence * 100:.0f}%)")
        print(f"     Field:       {c.field_name}")
        print(f"     Claim A:     {c.observed_fact_a}")
        print(f"     Claim B:     {c.observed_fact_b}")
        print(f"     Derived:     {c.derived_result}")
        print(f"     Explanation: {c.explanation}")

    if te_res.indicators:
        print("\n  [+] Timeline & Log Indicators:")
        for idx, ind in enumerate(te_res.indicators, 1):
            print(f"     {idx}. [{ind.severity}] {ind.title}: {ind.description}")

    # Build and export graph
    db = DatabaseManager(args.db)
    graph = build_evidence_graph(args.case_id, db)
    graph_out_path = Path(args.graph_out)
    graph_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_plotly_figure(graph)
    fig.write_html(str(graph_out_path))

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION METRICS:")
    print(f"  Total Ingested Evidence: {report.total_evidence}")
    print(f"  Contradictions Flagged:  {report.total_contradictions}")
    print(f"  Timeline Events Built:   {report.total_timeline_events}")
    print(f"  Graph Node Count:        {graph.number_of_nodes()}")
    print(f"  Graph Edge Count:        {graph.number_of_edges()}")
    print(f"  Interactive Graph:       {graph_out_path.resolve()}")
    print(f"  Pipeline Status:         {'SUCCESS' if report.success else 'FAILED'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
