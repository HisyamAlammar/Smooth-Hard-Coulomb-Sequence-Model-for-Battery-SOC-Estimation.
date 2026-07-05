import re
from pathlib import Path

f_in = Path('scripts/build_jnteti_v4.py')
text = f_in.read_text(encoding='utf-8')

# Dimension 1: De-jargonization
text = text.replace('Koreksi v5', 'Pra-Pemrosesan')
text = text.replace('Kampanye v5 merekonstruksi dataset', 'Evolusi pra-pemrosesan data dilakukan melalui tahapan evaluasi empiris')
text = text.replace('v4-v5', 'Protokol Dasar ke Protokol Final')
text = text.replace('representasi v4', 'representasi Protokol Dasar')
text = text.replace('Varian v5c (ohmic-corrected + mean-per-second decimation)', 'Varian Protokol Final (ohmic-corrected + mean-per-second decimation)')
text = text.replace('Tabel I. Varian dataset v4-v5 dan koreksi label/decimation.', 'Tabel I. Varian dataset dari Protokol Dasar ke Protokol Final. Pergeseran Label (Shift) diukur relatif terhadap Protokol Dasar untuk memverifikasi tidak ada zero temporal leakage.')
text = text.replace('data v5c', 'dataset Protokol Final')
text = text.replace('protokol v5', 'Protokol Final')
text = text.replace('checkpoint v5c', 'model terlatih representatif Skenario A (seed 42)')
text = text.replace('claims register v2', 'matriks verifikasi klaim')
text = text.replace('claims register:', 'matriks verifikasi klaim:')
text = text.replace('readiness gate Phase 10', 'tahapan evaluasi empiris')
text = text.replace('readiness gate', 'rekam jejak evaluasi empiris')
text = text.replace('artifact v5', 'rekam jejak evaluasi empiris (Protokol Final)')

# Dimension 2: Provenance
old_prov = "Penelitian ini menggunakan dataset publik LG 18650HG2 [8] yang berisi profil pengujian sel tunggal LG HG2 (kapasitas nominal Q_nom = 3,0 Ah) pada enam variasi suhu: -20°C, -10°C, 0°C, 10°C, 25°C, dan 40°C. Resistansi internal (R_int) diekstraksi dari data HPPC per temperatur: 16,51 mΩ (40°C), 19,86 mΩ (25°C), 28,75 mΩ (10°C), 40,08 mΩ (0°C), 62,19 mΩ (-10°C), 109,83 mΩ (-20°C) — rasio 6,65× antara suhu terpanas dan terdingin."
new_prov = "Dataset yang digunakan bersumber dari repositori publik Mendeley Data yang dipublikasikan oleh Kollmeyer dkk. (2020) [8], yang merupakan rujukan standar dalam literatur estimasi SOC berbasis deep learning. Objek pengujian adalah sel baterai litium-ion silindris komersial LG HG2 berformat 18650 dengan kimia katoda NMC (Nickel-Manganese-Cobalt) dan anoda grafit berkapasitas nominal 3,0 Ah. Pengambilan data dilakukan di dalam ruang termal (thermal chamber) terkendali pada enam kondisi isotermal tetap (-20°C, -10°C, 0°C, 10°C, 25°C, dan 40°C) guna mengisolasi pengaruh suhu terhadap resistansi internal secara presisi. Selain karakterisasi HPPC untuk ekstraksi resistansi internal, dataset memuat profil pembebanan dinamis yang merepresentasikan siklus berkendara kendaraan listrik nyata, meliputi siklus UDDS, LA92, dan US06. Data mentah didesimasi menjadi frekuensi operasional 1 Hz untuk menyelaraskan dengan laju sampling tipikal Battery Management System (BMS) tertanam. Resistansi internal (R_int) diekstraksi dari data HPPC per temperatur: 16,51 mΩ (40°C), 19,86 mΩ (25°C), 28,75 mΩ (10°C), 40,08 mΩ (0°C), 62,19 mΩ (-10°C), 109,83 mΩ (-20°C) — rasio 6,65× antara suhu terpanas dan terdingin."
text = text.replace(old_prov, new_prov)

# Multi-seed fixing
old_seed = "Seluruh model dilatih dengan 5 seed independen"
new_seed = "Seluruh model dilatih dengan 5 inisialisasi bobot acak berganda (independent random seeds)"
text = text.replace(old_seed, new_seed)

# Dimension 3: Recursive formulation
old_rec = "Pada inferensi windowed independen, setiap jendela memulai estimasi dari anchor baru. Pada carried inference, SOC akhir jendela sebelumnya menjadi state awal jendela berikutnya. Kalibrasi η* menskala envelope delta pada tahap inferensi tanpa retraining:"
new_rec = "Pada inferensi windowed independen (baseline worst-case di mana setiap jendela tidak memiliki memori), setiap jendela memulai estimasi dari anchor baru secara terisolasi. Sebaliknya, pada carried/recursive inference (merepresentasikan kontinuitas state deployment nyata), SOC akhir jendela sebelumnya menjadi state awal jendela berikutnya. Kalibrasi η* menskala envelope delta pada tahap inferensi tanpa retraining:"
text = text.replace(old_rec, new_rec)

# Dimension 5: Safety and Edge framing
old_safety1 = "PVR = 0,00% didefinisikan sebagai Mekanisme Keselamatan Algoritmik Deterministik — properti arsitektural yang terbukti, bukan pencapaian empiris. Kami secara eksplisit tidak mengklaim sertifikasi ASIL level-sistem, ketahanan terhadap fault sensor (characterized failure envelope saja), maupun \\\"functional safety\\\" sebagai klaim mandiri."
new_safety1 = "Klaim keselamatan dalam studi ini didefinisikan secara ketat pada ruang lingkup Kepatuhan Algoritmik (Algorithmic Admissibility). PVR bernilai presisi 0,00% dijamin secara inheren (by construction) sebagai Properti Arsitektural Termotivasi-Keselamatan, dengan asumsi integritas metrologi masukan. Apabila sensor arus mengalami kegagalan piranti keras, model akan menghasilkan trajektori deterministik yang tunduk secara konsisten terhadap masukan observasi yang cacat tersebut (fault laundering). Integrasi fusi sensor di level sistem berada di luar lingkup makalah ini."
text = text.replace(old_safety1, new_safety1)

old_edge = "Jejak parameter 54.626, flash INT8 ~56,7 KB, dan 5,25 MMAC/s mendukung kelayakan pada mikrokontroler Cortex-M4/M7 secara analitik. Namun lapisan konstrain bersifat non-kausal di dalam jendela: tidak ada WCET, pengukuran RAM target, maupun profiling CMSIS-NN. Seluruh klaim edge dilabelkan \\\"kelayakan level parameter saja\\\"."
new_edge = "Analisis kelayakan tingkat-parameter (parameter-level feasibility) menunjukkan jejak parameter 54.626, estimasi flash INT8 ~56,7 KB, dan beban komputasi 5,25 MMAC/s, yang mendukung kelayakan pada mikrokontroler kelas Cortex-M4/M7 secara analitik. Namun, arsitektur usulan mempertahankan sifat non-kausal di dalam jendela (intra-window non-causality), sehingga perumusan batas jaminan membutuhkan data historis jendela penuh. Realisasi perangkat keras bare-metal (termasuk pengukuran Worst-Case Execution Time dan profiling hardware-in-the-loop menggunakan CMSIS-NN) diserahkan sebagai subjek riset masa depan."
text = text.replace(old_edge, new_edge)

# Traceability string fixes
text = text.replace('drafts/JNTETI_SOC_Hard_Coulomb_Definitif_v4.docx', 'drafts/JNTETI_SOC_Hard_Coulomb_Definitif_Final.docx')
text = text.replace('drafts/JNTETI_SOC_Hard_Coulomb_Definitif_v4_traceability.md', 'drafts/JNTETI_SOC_Hard_Coulomb_Definitif_Final_traceability.md')
text = text.replace('drafts/JNTETI_Manuskrip_Definitif_v5.md', 'drafts/JNTETI_Manuskrip_Definitif_Final.md')
text = text.replace('Traceability v4', 'Traceability Final')
text = text.replace('v5 manuscript content', 'Final manuscript content')
text = text.replace('v5_traceability', 'Final_traceability')

Path('scripts/build_jnteti_final.py').write_text(text, encoding='utf-8')
print('Successfully wrote scripts/build_jnteti_final.py')
