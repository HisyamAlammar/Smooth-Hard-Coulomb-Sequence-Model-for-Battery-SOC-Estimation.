# Surgical Cleanup Final Report

## FIX 1
- Old text found at: Bagian I paragraph on Soft-PINN; Bagian III.A Soft-PINN retrospective paragraph; Bagian III.C paragraph before Gbr. 4.
- Grep confirming old text removed:
```text
FIX1A_OLD Ledger historis Sprint 44: 0
FIX1B_OLD Ledger Soft-PINN historis berhasil ditelusuri pada logs/sprint44_results_v3.json: 0
FIX1C_OLD Gbr. 4 memperlihatkan sapuan \u03b7 langsung dari eta_gamma_sweep.csv.: 0
FINAL_COUNT Ledger: 0
FINAL_COUNT Sprint 44: 0
FINAL_COUNT logs/sprint: 0
FINAL_COUNT eta_gamma_sweep.csv: 0
```
- Grep confirming new text present:
```text
FIX1A_NEW Evaluasi historis menunjukkan bahwa baseline Soft-PINN PI-TCN masih memiliki PVR 17,02%: 1
FIX1B_NEW Analisis retrospektif terhadap log eksperimen mengonfirmasi bahwa baseline Soft-PINN (PI-TCN) tetap mencapai PVR 17,02%: 1
FIX1C_NEW_ESC Gbr. 4 memperlihatkan sapuan \u03b7 berdasarkan data eksperimen kalibrasi.: 1
```
- Status: VERIFIED COMPLETE

## FIX 2
- Old text found at: Bagian II.A dataset paragraph; Gbr. 1 reference paragraph; Gbr. 1 caption; Bagian II.B feature paragraph.
- Grep confirming old text removed:
```text
FINAL_COUNT V_proxy: 0
FINAL_COUNT V_terminal: 0
FINAL_COUNT R_int: 0
FINAL_COUNT Q_actual: 0
FINAL_COUNT I_t: 0
```
- Grep confirming new text present:
```text
VERTALIGN_SUBSCRIPT_COUNT: 11
Paragraphs with <w:vertAlign w:val="subscript"/>: 26, 29, 31, 120
```
- Status: VERIFIED COMPLETE

## FIX 3
- Old text found at: Bagian II.B Protokol Zero Temporal Leakage paragraph.
- Grep confirming old text removed:
```text
FIX3_OLD train = 25?C, 10?C, validation = 0?C, test = 40?C, -10?C, -20?C: 0
```
- Grep confirming new text present:
```text
FIX3_NEW_ESC train = {25\u00b0C, 10\u00b0C}, validation = {0\u00b0C}, test = {40\u00b0C, -10\u00b0C, -20\u00b0C}: 1
```
- Status: VERIFIED COMPLETE

## FIX 4
- Old text found at: Bagian II.C fairness-baseline paragraph.
- Grep confirming old text removed:
```text
FIX4_OLD Guna menjaga keadilan pembandingan (fair baseline), model Vanilla LSTM dibangun: 0
```
- Grep confirming new text present:
```text
FIX4_NEW Guna menjaga keadilan pembandingan, model Vanilla LSTM dibangun: 1
```
- Status: VERIFIED COMPLETE

## OMML PROTECTION CHECK
- Pre-flight <m:oMath> count: 12
- Post-edit <m:oMath> count: 12
- Byte-diff result for all 12 equation blocks: IDENTICAL
- Status: EQUATIONS PROTECTED

## VISUAL VERIFICATION
- Screenshot(s) of equation pages: drafts/render_surgical_cleanup_final/page-03.png. All 12 equations still render with subscripts, superscripts, radical, hat accent, indicator symbol, and piecewise bracket.
- Screenshot/extract of fixed prose-variable locations: drafts/render_surgical_cleanup_final/page-02.png and page-03.png. R_int, V_proxy, V_terminal, Q_actual render with true subscript runs.
- Screenshot/extract of restored curly-brace sentence: drafts/render_surgical_cleanup_final/page-03.png. Set notation renders as train = {25?C, 10?C}, validation = {0?C}, test = {40?C, -10?C, -20?C}.
