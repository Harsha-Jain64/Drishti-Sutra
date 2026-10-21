# Drishti-Sutra

Drishti-Sutra is an offline-first forensic evidence analysis demo. It ingests synthetic case artifacts, stores evidence provenance and analysis results in SQLite, correlates document/chat/log sources, and presents indicators neutrally for investigative review. Findings are potential anomalies, not proof of forgery or wrongdoing.

## Install and run

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL printed by the command. The app uses `data/drishti_sutra.db` and stores uploaded files beneath `data/uploads/<case_id>/`. Tesseract is optional; when it is unavailable, the dashboard displays a fallback-extraction notice. The exact extraction method for each item is shown in Evidence Intake.

## Six-step demo click-path

1. Start the app with `streamlit run app.py` and keep the default case ID `Case_DS_001`.
2. In the sidebar, choose the ZIP archive for `sample_data/Case_DS_001`.
3. Click **Run Analysis** and wait for the pipeline to finish.
4. Open **1 · Overview**, then **2 · Evidence Intake** to review DB-backed counts, severity totals, hashes, and processing methods.
5. Open **3 · Document Analysis** and **4 · Screenshot & Chat** to inspect extracted fields, QR comparisons, ELA overlays, chat differences, and timeline inversion indicators.
6. Open **5 · Truth Engine**, **6 · Timeline & Graph**, and **7 · Reports & Audit**; filter contradictions, inspect the timeline/graph, then download the self-contained HTML report.

All displayed case findings and metrics are loaded from SQLite. Confidence scores are heuristic and are not calibrated probabilities. The app does not claim official or government verification.
