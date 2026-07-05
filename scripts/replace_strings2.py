import re
from pathlib import Path

f = Path('scripts/build_jnteti_final.py')
text = f.read_text(encoding='utf-8')

# 3) Abstrak
old_abs = "Pelanggaran ini menjadikan luaran model physically inadmissible untuk BMS di bawah kerangka ISO 26262."
new_abs = "Pelanggaran ini merepresentasikan ketidaksesuaian luaran model dengan prinsip konsistensi arah aliran arus yang disyaratkan secara prinsip oleh kerangka keselamatan fungsional seperti ISO 26262 dan PAS 8800, meskipun penelitian ini tidak melakukan proses sertifikasi ASIL formal terhadap sistem yang diusulkan."
text = text.replace(old_abs, new_abs)

# 4) Bagian I
old_i = "secara eksplisit mensyaratkan bahwa estimasi SOC harus monotone-consistent"
new_i = "menetapkan prinsip bahwa estimasi SOC harus monotone-consistent"
text = text.replace(old_i, new_i)

# 5 & 6) Bagian II.C
old_eq4_para = '    add_equation(doc, "δ_t = -limit_t · σ(ℓ_t^δ) jika I_t < -τ;  +limit_t · σ(ℓ_t^δ) jika I_t > τ;  0 jika |I_t| ≤ τ", 4)'
new_eq4_para = '    add_equation(doc, "δ_t = \\\\begin{cases} -limit_t \\\\cdot \\\\sigma(ℓ_t^δ) & \\\\text{jika } I_t < -τ \\\\\\\\ +limit_t \\\\cdot \\\\sigma(ℓ_t^δ) & \\\\text{jika } I_t > τ \\\\\\\\ 0 & \\\\text{jika } |I_t| \\\\le τ \\\\end{cases}", 4)\n\n    add_para(doc, "Persamaan (4) mendefinisikan sign assignment delta SOC yang dipandu langsung oleh arah arus terukur (current-routed): tanda δ_t dipaksa mengikuti tanda arus I_t melalui percabangan keras, sementara magnitudonya tetap dipelajari dan dibatasi kontinu oleh σ(·) ∈ (0,1) dikalikan batas fisik limit_t.")'
text = text.replace(old_eq4_para, new_eq4_para)

old_eq3_para = '    add_equation(doc, "limit_t = |I_t| · η · γ,  γ = Δt / (Q_nom · 3600) = 9,259 × 10⁻⁵ SOC/A/s", 3)\n\n    add_para(doc, "Delta SOC dikonstrain oleh current-routed sign assignment:")'
new_eq3_para = '    add_equation(doc, "limit_t = |I_t| · η · γ,  γ = Δt / (Q_nom · 3600) = 9,259 × 10⁻⁵ SOC/A/s", 3)\n\n    add_para(doc, "dengan Δt = 1 detik merepresentasikan interval sampling data setelah desimasi (Bagian II.A), dan Q_nom = 3,0 Ah kapasitas nominal sel; γ dengan demikian mengonversi arus terukur (A) menjadi fraksi SOC yang berpindah dalam satu interval sampling, sesuai hukum Coulomb-counting: ΔSOC = I·Δt / (Q_nom · 3600).")\n\n    add_para(doc, "Delta SOC dikonstrain oleh current-routed sign assignment:")'
text = text.replace(old_eq3_para, new_eq3_para)

# 7) Bagian II.C Pembuktian
old_pembuktian = '    add_para(doc, "Karena σ(ℓ^a) ∈ (0,1), anchor selalu berada di dalam (lo, hi). Karena [lo, hi] didefinisikan agar SOC_anchor + min_t C_t ≥ 0 dan SOC_anchor + max_t C_t ≤ 1, seluruh trajektori dijamin berada dalam [0, 1]. Seluruh operasi bersifat terdiferensiasi-kontinu (smoothly differentiable), menghindari patologi gradien-nol dari pendekatan hard clamp.")'
new_pembuktian = '    add_para(doc, "Pembuktian: karena lo ≤ SOC_anchor ≤ hi berdasarkan Persamaan (7), dan berdasarkan definisi lo, hi pada Persamaan (6), untuk sembarang timestep t berlaku SOC_anchor + C_t ≥ lo + min_k C_k ≥ (-min_k C_k) + min_k C_k = 0, dan secara simetris SOC_anchor + C_t ≤ hi + max_k C_k ≤ (1 - max_k C_k) + max_k C_k = 1. Dengan demikian SOC_t ∈ [0,1] terjamin untuk seluruh t ∈ {1,...,T} tanpa memerlukan operasi clamping eksplisit pada keluaran akhir.")\n\n    add_para(doc, "Karena σ(ℓ^a) ∈ (0,1), anchor selalu berada di dalam (lo, hi). Karena [lo, hi] didefinisikan agar SOC_anchor + min_t C_t ≥ 0 dan SOC_anchor + max_t C_t ≤ 1, seluruh trajektori dijamin berada dalam [0, 1]. Seluruh operasi bersifat terdiferensiasi-kontinu (smoothly differentiable), menghindari patologi gradien-nol dari pendekatan hard clamp.")'
text = text.replace(old_pembuktian, new_pembuktian)

# 8 & 9) Bagian III.D
old_iiid = '        "Perbandingan like-for-like (keduanya kontinu, keduanya tanpa tuning pada data uji): HC terkalibrasi 4,43% vs. EKF terbaik 6,85%, dengan HC mempertahankan PVR ≡ 0,00%. Parameter EKF bersifat literature-like, bukan diidentifikasi dari sel ini — caveat yang dinyatakan secara eksplisit.",'
new_iiid = '        "Perbandingan like-for-like (keduanya kontinu, keduanya tanpa tuning pada data uji): HC terkalibrasi 4,43% vs. EKF terbaik 6,85%, dengan HC mempertahankan PVR ≡ 0,00%. Parameter EKF bersifat literature-like, bukan diidentifikasi dari sel ini — caveat yang dinyatakan secara eksplisit.",\n        "Perlu dicatat satu asimetri prosedural: hasil HC 4,43% melalui satu tahap kalibrasi tambahan tanpa retraining (η* = 2,0, disetel pada data validasi) yang tidak memiliki padanan langsung pada EKF, sehingga perbandingan ini bersifat indikatif terhadap potensi unggul arsitektural, bukan bukti definitif superioritas pada kondisi penyetelan yang identik untuk kedua metode.",\n        "Penting dicatat bahwa keunggulan HC atas EKF terletak pada jaminan konsistensi arah (PVR) dan performa recursive setelah kalibrasi, bukan pada imunitas terhadap degradasi suhu dingin itu sendiri: sebagaimana ditunjukkan pada Bagian III.C, komponen anchor HC juga mengalami degradasi signifikan pada -20°C sebelum penerapan carried inference dan kalibrasi η*.",'
text = text.replace(old_iiid, new_iiid)

# 10) Bagian III.A
old_iiia = '    section(doc, 1)\n    add_figure(doc, FIG_DIR / "fig02_research_evolution_flowchart.png",'
new_iiia = '    add_para(doc, "Perlu dicatat bahwa estimasi MaxE pada iterasi awal analisis sempat melaporkan nilai lebih tinggi akibat kontaminasi label oleh ohmic bias loaded-start (Bagian II.A); setelah koreksi label diterapkan, celah performa antara model HC dan baseline tetap signifikan (MaxE 36,50-46,47% vs 60,9%), mengonfirmasi temuan utama tidak bergantung pada artefak pelabelan tersebut.")\n\n    section(doc, 1)\n    add_figure(doc, FIG_DIR / "fig02_research_evolution_flowchart.png",'
text = text.replace(old_iiia, new_iiia)

# 11) Bagian III.E
old_iiie = '        "Invarian PVR terjaga pada tiga jalur kuantisasi benigna'
new_iiie = '        "Kami mendefinisikan jalur kuantisasi sebagai \'benign\' apabila operasi pembulatan bersifat simetris dan konsisten di seluruh timestep (tidak mengubah tanda maupun urutan relatif nilai delta SOC antar-timestep), sebagaimana berlaku pada dynamic int8, uint8 trajectory, dan float16 accumulation yang diuji.",\n        "Invarian PVR terjaga pada tiga jalur kuantisasi benigna'
text = text.replace(old_iiie, new_iiie)

# 12) Bagian IV
old_iv = 'dieliminasi secara struktural dan permanen melalui'
new_iv = 'dieliminasi secara struktural dan tidak bergantung pada penyetelan hyperparameter (by construction), untuk kelas arsitektur backbone sequential (LSTM dan TCN) yang divalidasi pada penelitian ini melalui'
text = text.replace(old_iv, new_iv)

# 13) Bagian II.A
old_iia_table = '    add_dataframe_table(\n        doc,\n        "Tabel I'
new_iia_table = '    add_para(doc, "Kolom \'Shift (%)\' pada Tabel I melaporkan persentase perubahan nilai SOC rata-rata akibat koreksi ohmic bias relatif terhadap label asli, sedangkan kolom \'Leak\' melaporkan jumlah timestamp yang teridentifikasi tumpang tindih antar partisi train/validation/test melalui enam asersi interseksi (Bagian II.B); nilai nol pada seluruh varian mengonfirmasi tidak adanya temporal leakage.")\n\n    add_dataframe_table(\n        doc,\n        "Tabel I'
text = text.replace(old_iia_table, new_iia_table)

f.write_text(text, encoding='utf-8')
print("Successfully updated build_jnteti_final.py")
