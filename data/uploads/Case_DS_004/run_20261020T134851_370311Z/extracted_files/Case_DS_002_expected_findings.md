# Case DS-002 — Rental Deposit Dispute (SYNTHETIC)

Answer key for YOUR checking only. Do not put this file inside the evidence upload.
All people, IDs, numbers and bank details are fictional.

Upload: Case_DS_002.zip (or the 5 files inside the folder).

| # | Planted inconsistency | Where to look |
|---|---|---|
| 1 | DOB differs: PAN card 1991-03-07 vs ID card printed 1991-09-07 | Document Verifier / Truth Engine |
| 2 | QR on ID card encodes DOB 1991-03-07, printed DOB is 1991-09-07 | Document Verifier |
| 3 | Localized compression anomaly (ELA) over the DOB block of the ID card | Document Verifier heatmap |
| 4 | Screenshot order: 02:32 PM message appears above the 02:31 PM message | Screenshot Verifier |
| 5 | Amount: screenshot Rs 75,000, chat export Rs 7,500, bank log Amount=75000.00 | Truth Engine |
| 6 | "Okay, sending it now." is in the screenshot but absent from the export | Truth Engine |
| 7 | Log WARN: msgstore.db modified 14:35:40, after the 14:31:15 transfer | Timeline |

Expected: about the same set of contradiction types as DS-001 (6 contradictions).
These are potential anomalies, not proof of anything.

Notes
- The PNGs carry hidden text metadata like the DS-001 files, so the app's fallback
  works even without Tesseract. With Tesseract installed, real OCR runs first
  (checked here: both cards, the screenshot text, and the QR all read correctly).
- Amounts are written "Rs 75,000" (not the rupee glyph) because OCR misreads the glyph.
- Case ID suggestion: Case_DS_002. People: Meera Nair, Karan Mehta.
