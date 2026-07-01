# Panduan Presentasi 12 Slide

Judul kerja: **Smooth Hard-Coulomb Sequence Model for Battery SOC Estimation**

Peran presentasi: Principal Research Advisor, MLOps Architect, Battery ML Auditor, dan Strategist untuk sidang akhir.

Tujuan deck: membuktikan bahwa kontribusi utama bukan sekadar akurasi prediksi SOC, tetapi **jaminan keselamatan struktural**: Physics Violation Rate (PVR) menjadi `0.00%` karena arsitektur, bukan karena keberuntungan training.

Durasi ideal: 12-15 menit. Target tempo: 55-75 detik per slide.

Kalimat tulang punggung presentasi:

> Baseline AI hanya memprediksi SOC; Soft-PINN menegosiasikan fisika; Smooth Hard-Coulomb membuat pelanggaran fisika tidak dapat dicapai oleh output model.

---

## Struktur Besar

| Bagian | Slide | Fungsi |
|---|---:|---|
| Masalah | 1-4 | Tunjukkan bahaya SOC estimation dan physics blindness. |
| Jalur kegagalan | 5-7 | Tunjukkan eksperimen gagal sebagai bukti kedalaman riset. |
| Terobosan | 8-10 | Jelaskan Smooth Hard-Coulomb dan sertifikasi PVR. |
| Kejujuran ilmiah | 11 | Jelaskan Anchor Trap dan limit observability pada -20 C. |
| Penutup | 12 | Ringkas kontribusi dan future work. |

---

## Slide 1 - Judul dan Klaim Utama

### Objective

Membuka presentasi dengan tesis yang tajam: penelitian ini adalah estimator SOC berbasis sequence model yang aman secara struktural.

### Visual/Data Content

Tampilkan:

- Judul: `Smooth Hard-Coulomb Sequence Model for Battery SOC Estimation`
- Subtitle: `Functional-safety-oriented SOC estimation with 0.00% Physics Violation Rate`
- Satu persamaan ringkas:

$$
\widehat{\mathrm{SOC}}_t = \widehat{\mathrm{SOC}}_{\mathrm{anchor}} + \sum_{\tau=1}^{t}\Delta \widehat{\mathrm{SOC}}_{\tau}
$$

$$
\mathrm{PVR} = 0.00\% \quad \text{by architectural invariant}
$$

Opsional visual kecil:

- `outputs/figures/ablation_studies/fig_11_hardcoulomb_lstm_vs_tcn_tradeoff.png`

### Key Narrative Points

"Penelitian ini bukan hanya membuat model SOC yang lebih akurat. Inti kontribusinya adalah membuat output model tidak bisa melanggar arah fisika saat baterai sedang discharge."

"Saya akan menunjukkan jalur kegagalan model biasa, lalu bagaimana Smooth Hard-Coulomb mengubah masalah ini dari optimasi biasa menjadi constraint struktural."

### Catatan Desain

Jangan penuh teks. Slide 1 harus bersih: judul, nama, pembimbing, institusi, dan satu klaim besar `0.00% PVR`.

---

## Slide 2 - SOC Error Adalah Hazard Keselamatan

### Objective

Mendefinisikan masalah keselamatan: RMSE kecil tidak cukup jika model masih memprediksi SOC naik saat discharge.

### Visual/Data Content

Tampilkan diagram sederhana:

```text
Current < 0  ->  Battery discharging  ->  SOC must not increase
```

Tampilkan definisi PVR:

$$
\mathrm{PVR} =
\frac{\#\{t : \Delta \widehat{\mathrm{SOC}}_t > 0 \land I_t < -I_{\mathrm{th}}\}}
{\#\{t : I_t < -I_{\mathrm{th}}\}}
\times 100\%
$$

Data pendukung dari log final:

- Vanilla LSTM Scenario A PVR: `49.9694%`
- Vanilla LSTM Scenario B PVR: `41.0552%`
- Hard-Coulomb LSTM Scenario A/B PVR: `0.00%`

Sumber data:

- `outputs/v7_final/sprint48_evaluation_results.json`

### Key Narrative Points

"Dalam Battery Management System, metrik rata-rata seperti RMSE tidak cukup. Jika model memprediksi SOC naik ketika arus menunjukkan discharge, model itu menghasilkan perilaku yang tidak fisik."

"Karena itu saya menggunakan PVR sebagai metrik keselamatan: bukan hanya seberapa dekat prediksi, tetapi apakah prediksi melanggar hukum arah energi."

### Catatan Desain

Gunakan warna merah untuk violation dan hijau untuk safe. Hindari terlalu banyak rumus.

---

## Slide 3 - Dataset Menunjukkan Masalah Sebenarnya

### Objective

Membuktikan bahwa tantangan terbesar berasal dari kondisi dingin ekstrem, bukan sekadar arsitektur neural network.

### Visual/Data Content

Tampilkan figure utama:

- `outputs/figures/fig_q1_observability_collapse.png`

Opsional inset kecil:

- `outputs/figures/fig_q1_transient_dynamic_profile.png`

Poin data yang perlu ditulis di slide:

- Bandingkan hubungan `Voltage vs SOC` pada `25 C` dan `-20 C`.
- Tekankan `-20 C` sebagai kondisi observability collapse.
- Tampilkan bahwa dataset berisi profil dinamis, bukan hanya constant-current lab test.

### Key Narrative Points

"Pada 25 C, terminal voltage masih membawa informasi SOC yang relatif stabil. Pada -20 C, polarisation dan internal resistance membuat voltage tidak lagi menjadi proxy SOC yang bersih."

"Artinya, model tidak gagal karena malas belajar. Model diberi sinyal observasi yang secara fisik jauh lebih buruk pada temperatur dingin."

### Catatan Desain

Slide ini harus menjadi bukti visual. Jangan jelaskan terlalu panjang; biarkan grafik `Voltage vs SOC` bekerja.

---

## Slide 4 - Vanilla Deep Learning Buta Fisika

### Objective

Membuktikan bahwa LSTM/TCN biasa tidak cocok untuk klaim keselamatan karena hanya mengoptimalkan MSE.

### Visual/Data Content

Tampilkan:

- `outputs/figures/ablation_studies/fig_02_vanilla_physics_blindness_pvr.png`

Angka utama:

| Model | Scenario A PVR | Scenario B PVR |
|---|---:|---:|
| Vanilla LSTM | 49.9694% | 41.0552% |
| Hard-Coulomb LSTM | 0.0000% | 0.0000% |

Sumber:

- `notebooks/ablation_studies/02_Vanilla_LSTM_TCN_Physics_Blindness.ipynb`
- `outputs/v7_final/sprint48_evaluation_results.json`

### Key Narrative Points

"Vanilla LSTM dapat terlihat masuk akal dari RMSE, tetapi secara fisika ia tetap berbahaya. Model ini memprediksi SOC meningkat selama discharge hampir separuh waktu pada Scenario A."

"Ini adalah physics blindness: loss function tidak memahami arah energi, sehingga prediksi tidak aman meskipun error rata-rata turun."

### Catatan Desain

Gunakan bar chart besar. Label `49.97%` dan `0.00%` harus terlihat jelas dari belakang ruangan.

---

## Slide 5 - Seq2Point Membuat Pseudo-Trajectory Artifact

### Objective

Menjelaskan kegagalan fase awal: prediksi point-by-point tidak punya struktur sequence untuk menjaga monotonicity dalam window.

### Visual/Data Content

Tampilkan:

- `outputs/figures/ablation_studies/fig_01_seq2point_pseudo_pvr.png`

Poin yang perlu tampil:

- Seq2Point menghasilkan satu nilai SOC per window.
- Overlapping window disambung menjadi pseudo-trajectory.
- Pseudo-trajectory dapat naik-turun walaupun discharge sebenarnya turun halus.

### Key Narrative Points

"Seq2Point bukan trajectory model. Ia menghasilkan tebakan lokal yang independen, lalu hasilnya dijahit menjadi seolah-olah trajectory."

"Karena tidak ada output sequence, tidak ada tempat alami untuk memaksakan monotonicity intra-window. Ini menghasilkan jitter yang terlihat seperti pelanggaran fisika."

### Catatan Desain

Tampilkan satu garis true SOC yang halus dan satu garis prediction yang jittery. Ini mudah dipahami penguji non-ML.

---

## Slide 6 - Soft-PINN Mengurangi Pelanggaran, Tetapi Tidak Menjamin Nol

### Objective

Menjawab pertanyaan reviewer: kenapa tidak cukup menambahkan physics penalty pada loss?

### Visual/Data Content

Tampilkan:

- `outputs/figures/ablation_studies/fig_03_soft_pinn_gradient_collision.png`

Rumus ringkas:

$$
\mathcal{L} = \mathrm{MSE}_{\mathrm{data}} + \lambda\,\mathcal{L}_{\mathrm{physics}}
$$

Angka penting:

- Soft-PINN Scenario A PVR plateau: `17.02%`
- Soft-PINN Scenario B PVR plateau: `43.63%`

Sumber:

- `notebooks/ablation_studies/03_Soft_PINN_Penalty_Gradient_Collision.ipynb`
- `logs/sprint44_results_v3.json`

### Key Narrative Points

"Soft-PINN hanya memberi penalti, bukan larangan. Optimizer tetap bisa memilih kompromi antara data loss dan physics loss."

"Untuk functional safety, kompromi seperti 17% PVR tetap gagal. Sistem aman tidak boleh bergantung pada harapan bahwa optimizer selalu memilih solusi fisik."

### Catatan Desain

Buat pesan hitam-putih: `Penalty != Guarantee`.

---

## Slide 7 - Hard Constraint Awal Juga Belum Cukup

### Objective

Menunjukkan evolusi desain: monotonic-only dan hard clamp menyelesaikan satu masalah tetapi menciptakan masalah lain.

### Visual/Data Content

Tampilkan dua panel:

- Kiri: `outputs/figures/ablation_studies/fig_04_direction_only_cumulative_drift.png`
- Kanan: `outputs/figures/ablation_studies/fig_05_clamp_vs_smooth_gradient_pathology.png`

Poin data:

- Direction-only constraint dapat mencapai MaxE sangat besar, sekitar `99.90%` pada historis Scenario A.
- Clamp lama menahan magnitude, tetapi menciptakan dead-gradient zone saat prediksi keluar envelope.

Sumber:

- `notebooks/ablation_studies/04_Direction_Only_Hard_Constraint_Cumulative_Drift.ipynb`
- `notebooks/ablation_studies/05_Hard_Clamp_vs_Smooth_Coulomb_Gradient_Pathology.ipynb`
- `logs/sprint45_results_v4_lstm.json`
- `logs/sprint46_results_v5_coulomb.json`

### Key Narrative Points

"Monotonic-only constraint benar pada arah, tetapi tidak membatasi besar delta. Akibatnya model bisa drift jauh lalu disembunyikan oleh clipping akhir."

"Hard clamp memperbaiki batas delta, tetapi gradien menjadi nol di luar envelope. Model menjadi sulit pulih ketika output mentah sudah terlempar terlalu jauh."

### Catatan Desain

Jadikan slide ini sebagai story of failure. Ini menunjukkan riset tidak langsung berhasil, tetapi melewati autopsi desain.

---

## Slide 8 - Smooth Hard-Coulomb: Solusi Final

### Objective

Memperkenalkan arsitektur final dan mengapa ia menyelesaikan masalah safety plus gradient.

### Visual/Data Content

Tampilkan diagram alur:

```text
Input sequence -> LSTM/TCN backbone -> delta logits + anchor logit
delta logits -> sigmoid magnitude -> Coulomb envelope -> cumulative update
anchor logit -> dynamic bound lo/hi -> SOC anchor
SOC anchor + cumulative Coulomb increment -> SOC sequence
```

Tampilkan equation block:

$$
L_t = |I_t| \cdot \frac{\Delta t}{Q_{\mathrm{nom}} \cdot 3600} \cdot \eta
$$

$$
m_t = \sigma\left(z^{\Delta}_t\right)
$$

$$
\Delta \widehat{\mathrm{SOC}}_t =
\begin{cases}
-L_t m_t, & I_t < -I_{\mathrm{th}} \\
\phantom{-}L_t m_t, & I_t > I_{\mathrm{th}} \\
0, & |I_t| \le I_{\mathrm{th}}
\end{cases}
$$

$$
C_t = \sum_{\tau=1}^{t} \Delta \widehat{\mathrm{SOC}}_{\tau}
$$

$$
w = \max(h_i - l_o, \epsilon), \qquad
\widehat{\mathrm{SOC}}_{\mathrm{anchor}} = l_o + w \cdot \sigma\left(z^{\mathrm{anchor}}\right)
$$

$$
\widehat{\mathrm{SOC}}_t = \widehat{\mathrm{SOC}}_{\mathrm{anchor}} + C_t
$$

Sumber code:

- `src/model_v5_coulomb.py`
- `src/model_v5_coulomb_tcn.py`
- `src/model_v6_contextual.py`

### Key Narrative Points

"Smooth Hard-Coulomb memisahkan learning dan safety. Neural network belajar magnitude dan anchor, tetapi arah dan batas delta ditentukan oleh arus dan hukum Coulomb."

"Karena magnitude memakai sigmoid, gradien tetap hidup. Karena arah memakai current-sign routing, PVR tetap nol secara struktural."

### Catatan Desain

Gunakan animasi bertahap kalau membuat PPT: tampilkan backbone dulu, lalu delta logits, lalu constraint layer, lalu output SOC.

---

## Slide 9 - Safety Certificate: PVR 0.00%

### Objective

Menampilkan hasil utama: semua model Hard-Coulomb final memiliki PVR nol.

### Visual/Data Content

Tampilkan:

- `outputs/figures/ablation_studies/fig_11_hardcoulomb_lstm_vs_tcn_tradeoff.png`

Tampilkan angka besar:

| Model | Scenario | $\mathrm{PVR}$ | Jumlah pelanggaran |
|---|---|---:|---:|
| Hard-Coulomb LSTM | A | $0.00\%$ | $0$ |
| Hard-Coulomb LSTM | B | $0.00\%$ | $0$ |
| Hard-Coulomb TCN | A | $0.00\%$ | $0$ |
| Hard-Coulomb TCN | B | $0.00\%$ | $0$ |

Sumber:

- `outputs/v7_final/sprint48_evaluation_results.json`
- `outputs/v8_tcn_redemption/sprint52/sprint52_tcn_redemption_results.json`

### Key Narrative Points

"Hasil terpenting adalah PVR nol lintas backbone. Ini bukan hasil karena model kebetulan belajar baik, tetapi karena output manifold tidak menyediakan jalur untuk melanggar arah fisika."

"Dengan kata lain, keselamatan tidak diletakkan di loss function. Keselamatan dipindahkan ke struktur arsitektur."

### Catatan Desain

Tuliskan `0.00%` sangat besar. Jangan biarkan RMSE mencuri perhatian dari klaim safety.

---

## Slide 10 - LSTM vs TCN: Backbone Menentukan Akurasi, Bukan Safety

### Objective

Menjawab apakah sequence backbone masih penting setelah constraint diterapkan.

### Visual/Data Content

Tampilkan lagi:

- `outputs/figures/ablation_studies/fig_11_hardcoulomb_lstm_vs_tcn_tradeoff.png`

Fokuskan pada RMSE, MaxE, parameter count.

Angka utama:

| Model | Scenario A RMSE | Scenario A MaxE | Scenario B RMSE | Scenario B MaxE | Params |
|---|---:|---:|---:|---:|---:|
| Hard-Coulomb LSTM | 12.7107% | 55.1126% | 8.5667% | 34.9985% | 54,626 |
| Hard-Coulomb TCN | 11.4587% | 46.7298% | 8.5823% | 39.4864% | 208,546 |

### Key Narrative Points

"Setelah Hard-Coulomb memastikan PVR nol, backbone tidak lagi menentukan safety. Backbone menentukan seberapa baik model membaca feature map di dalam ruang output yang aman."

"TCN lebih kuat pada Scenario A, tetapi menggunakan sekitar 3.82 kali parameter LSTM. Ini menjadi tradeoff antara akurasi dan biaya deployment."

### Catatan Desain

Buat satu callout: `Safety: same. Accuracy/cost: backbone-dependent.`

---

## Slide 11 - Anchor Trap: Kenapa Error -20 C Masih Tinggi?

### Objective

Menjelaskan kelemahan secara jujur: MaxE tinggi pada -20 C adalah observability bottleneck, bukan kegagalan constraint.

### Visual/Data Content

Tampilkan dua figure:

- `outputs/figures/ablation_studies/fig_08_anchor_trap_eta_non_cause.png`
- `outputs/figures/ablation_studies/fig_09_contextual_anchor_ocv_vs_history.png`

Equation utama:

$$
\widehat{\mathrm{SOC}}_t = \widehat{\mathrm{SOC}}_{\mathrm{anchor}} + \sum_{\tau=1}^{t}\Delta \widehat{\mathrm{SOC}}_{\tau}
$$

Poin data:

- Safety factor $\eta$ sweep tidak menghilangkan MaxE dingin.
- OCV-rest-only memperbaiki MaxE dibanding empty context.
- History-only memperburuk anchor: MaxE `62.1314%`.
- Gated context gagal karena rest evidence sparse, sekitar `10.38%` valid rest pada -20 C.

Sumber:

- `outputs/v7_final/sprint48_safety_ablation_results.json`
- `outputs/v5_contextual/sprint50_contextual/sprint50_contextual_results.json`
- `notebooks/ablation_studies/08_Anchor_Trap_and_Safety_Factor_NonCause.ipynb`
- `notebooks/ablation_studies/09_Contextual_Anchor_OCV_Rest_vs_History.ipynb`
- `notebooks/ablation_studies/10_Gated_Context_Negative_Result_Sparse_Rest_Validity.ipynb`

### Key Narrative Points

"Hard-Coulomb dapat menjamin arah dan magnitude delta, tetapi tidak bisa menebak ulang initial SOC jika anchor awal sudah salah. Ini saya sebut Anchor Trap."

"Pada -20 C, terminal voltage kehilangan hubungan yang bersih dengan SOC internal. Jadi error tinggi adalah bottleneck observability, bukan bukti bahwa constraint gagal."

### Catatan Desain

Ini slide kejujuran ilmiah. Penguji biasanya menghargai kalau limitation dijelaskan sebelum diserang.

---

## Slide 12 - Kesimpulan dan Future Work

### Objective

Menutup dengan kontribusi, bukti, dan arah riset berikutnya.

### Visual/Data Content

Tampilkan 3 kotak kontribusi:

```text
1. Failure forensics
   Vanilla, Seq2Point, Soft-PINN, clamp, dan context variants diaudit.

2. Smooth Hard-Coulomb architecture
   Sigmoid-scaled Coulomb envelope + dynamic bounded anchor.

3. Safety certificate
   Hard-Coulomb LSTM/TCN: PVR 0.00%, violations 0.
```

Future work:

- EKF/UKF/ECM baseline.
- Multi-seed statistical variance.
- TinyML quantization.
- Hardware-in-the-loop validation.
- WCET and latency profiling on embedded MCU.
- Current sensor bias and aging robustness.

### Key Narrative Points

"Kesimpulan utama penelitian ini: SOC estimator yang aman tidak cukup dilatih dengan data dan penalti. Output-nya harus dirancang agar pelanggaran fisika tidak mungkin terjadi."

"Langkah berikutnya adalah membawa bukti offline ini ke validasi embedded: quantization, WCET, dan hardware-in-the-loop testing."

### Catatan Desain

Tutup dengan kontribusi yang bisa diingat: `Forensics -> Architecture -> Certificate`.

---

## Backup Slide untuk Q&A

Jangan masukkan semua backup ke presentasi utama. Simpan setelah slide 12.

### Backup A - Data Leakage Defense

Gunakan:

- `outputs/figures/ablation_studies/fig_07_split_leakage_forensics.png`
- `notebooks/ablation_studies/07_Zero_Leakage_Split_Before_Windowing_Forensics.ipynb`

Pesan:

"Split dilakukan sebelum sliding window, sehingga tidak ada temporal overlap antara train, validation, dan test."

### Backup B - V_proxy Feature Defense

Gunakan:

- `outputs/figures/ablation_studies/fig_06_vproxy_feature_defense.png`

Pesan:

"V_proxy dipakai untuk mengurangi distorsi Ohmic drop dari terminal voltage, bukan untuk memasukkan label leakage."

### Backup C - Gated Context Negative Result

Gunakan:

- `outputs/figures/ablation_studies/fig_10_gated_context_sparse_rest_validity.png`
- `outputs/figures/ablation_studies/fig_10_gated_context_error_breakdown.png`

Pesan:

"Gating tidak gagal karena ide gating buruk, tetapi karena rest-state evidence terlalu sparse pada -20 C."

### Backup D - Exact Metric Ledger

Gunakan:

- `outputs/v7_final/sprint48_evaluation_results.json`
- `outputs/v8_tcn_redemption/sprint52/sprint52_tcn_redemption_results.json`
- `outputs/v5_contextual/sprint50_contextual/sprint50_contextual_results.json`

Pesan:

"Semua angka di slide berasal dari JSON hasil evaluasi, bukan angka manual."

---

## Urutan Narasi 1 Kalimat per Bagian

Gunakan ini sebagai transisi antarbagian.

1. Problem:
   "SOC estimation menjadi safety-critical ketika model boleh melanggar arah fisika."

2. Baseline failure:
   "Model deep learning biasa bisa akurat rata-rata tetapi tetap tidak aman secara fisika."

3. Constraint evolution:
   "Soft penalty, monotonic-only, dan hard clamp masing-masing memperbaiki sebagian masalah tetapi gagal memberi jaminan yang stabil."

4. Breakthrough:
   "Smooth Hard-Coulomb menggabungkan envelope Coulomb, current-sign routing, sigmoid logits, dan dynamic anchor bounding."

5. Honesty:
   "Error ekstrem pada -20 C berasal dari keterbatasan observability anchor, bukan dari pelanggaran constraint."

6. Close:
   "Kontribusi utama adalah safety by architecture, dengan arah future work menuju embedded validation."

---

## Daftar Figure Final yang Perlu Disiapkan

| Kebutuhan | File |
|---|---|
| Observability collapse | `outputs/figures/fig_q1_observability_collapse.png` |
| Transient dynamic profile | `outputs/figures/fig_q1_transient_dynamic_profile.png` |
| Seq2Point artifact | `outputs/figures/ablation_studies/fig_01_seq2point_pseudo_pvr.png` |
| Vanilla PVR gap | `outputs/figures/ablation_studies/fig_02_vanilla_physics_blindness_pvr.png` |
| Soft-PINN failure | `outputs/figures/ablation_studies/fig_03_soft_pinn_gradient_collision.png` |
| Direction-only drift | `outputs/figures/ablation_studies/fig_04_direction_only_cumulative_drift.png` |
| Clamp vs smooth | `outputs/figures/ablation_studies/fig_05_clamp_vs_smooth_gradient_pathology.png` |
| Eta sweep / Anchor Trap | `outputs/figures/ablation_studies/fig_08_anchor_trap_eta_non_cause.png` |
| OCV-rest vs history | `outputs/figures/ablation_studies/fig_09_contextual_anchor_ocv_vs_history.png` |
| Gated context sparse rest | `outputs/figures/ablation_studies/fig_10_gated_context_sparse_rest_validity.png` |
| LSTM vs TCN tradeoff | `outputs/figures/ablation_studies/fig_11_hardcoulomb_lstm_vs_tcn_tradeoff.png` |

---

## Design Rule untuk Slide

- Maksimal 1 grafik utama per slide, kecuali slide 7 dan 11 yang memang comparison slide.
- Font minimal 24 pt untuk isi, 32 pt untuk judul.
- Gunakan angka besar untuk `0.00% PVR`.
- Gunakan warna konsisten: merah untuk violation, biru untuk baseline, hijau untuk Hard-Coulomb/safe.
- Jangan membaca paragraf. Slide berisi bukti; narasi lisan menjelaskan sebab-akibat.
- Untuk sidang, lebih baik 12 slide kuat daripada 25 slide penuh tabel.

---

## Checklist Sebelum Sidang

- [ ] Semua figure terbuka jelas di PowerPoint/PDF.
- [ ] Label angka PVR terlihat jelas dari mode presenter.
- [ ] Slide 8 equation tidak terlalu kecil.
- [ ] Backup slide berisi leakage, V_proxy, gated context, dan JSON ledger.
- [ ] Siapkan jawaban untuk pertanyaan: "Kenapa tidak EKF/UKF?"
- [ ] Siapkan jawaban untuk pertanyaan: "Kenapa $\eta = 1.5$?"
- [ ] Siapkan jawaban untuk pertanyaan: "Kenapa MaxE -20 C masih tinggi?"
- [ ] Siapkan jawaban untuk pertanyaan: "Apakah PVR 0.00% berlaku jika sensor arus bias?"

---

## Jawaban Singkat untuk Pertanyaan Sulit

### Kenapa tidak cukup pakai RMSE?

RMSE adalah metrik rata-rata. Dalam sistem safety-critical, satu kelas error yang melanggar fisika tetap penting walaupun rata-rata error rendah. Karena itu PVR dipakai sebagai metrik safety.

### Kenapa Soft-PINN gagal?

Soft-PINN memberi penalti, bukan constraint. Optimizer dapat menukar sedikit physics violation untuk menurunkan data loss. Itu tidak cukup untuk klaim functional safety.

### Kenapa Smooth lebih baik dari clamp?

Clamp memberi batas keras tetapi gradien nol ketika output mentah berada di luar envelope. Sigmoid magnitude tetap membatasi output dalam envelope sambil menjaga gradien tetap hidup.

### Kenapa MaxE -20 C masih tinggi?

Karena pada -20 C terminal voltage terdistorsi oleh polarization dan resistance. Anchor awal menjadi sulit diamati. Hard-Coulomb menjaga trajectory tetap fisik, tetapi tidak bisa menciptakan informasi SOC absolut yang tidak ada pada fitur.

### Apakah ini siap untuk embedded?

Secara parameter count, LSTM sekitar 54k parameter dan TCN sekitar 208k parameter, sehingga feasible untuk arah TinyML. Tetapi validasi WCET, latency, quantization, dan HIL masih future work.
