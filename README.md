# Smooth Hard-Coulomb Sequence Model untuk Estimasi SOC Baterai

Repositori ini berisi kode penelitian untuk estimasi **State of Charge (SOC)** baterai lithium-ion menggunakan deep learning yang dibatasi oleh hukum fisika. Metode final penelitian ini adalah **Smooth Hard-Coulomb Sequence Model**, yaitu estimator sequence berbasis LSTM/TCN dengan output layer yang memaksa perubahan SOC tetap valid secara fisika selama proses discharge.

Repositori ini disusun untuk **koreksi proyek dan source code**. Draft manuskrip jurnal, dataset mentah, array hasil preprocessing, checkpoint model, virtual environment lokal, dan arsip eksperimen lama sengaja diabaikan melalui `.gitignore` agar dosen dapat memeriksa logika penelitian tanpa menerima workspace berukuran sangat besar.

## Ringkasan Penelitian

Estimasi SOC penting untuk Battery Management System (BMS). Masalahnya, model deep learning biasa dapat memiliki error rata-rata yang terlihat baik, tetapi tetap menghasilkan trajectory yang tidak fisik. Contoh pelanggaran yang berbahaya adalah model memprediksi SOC naik ketika baterai sedang discharge.

Penelitian ini mempelajari pola kegagalan tersebut dan mengusulkan perbaikan struktural:

> Fisika tidak hanya ditambahkan sebagai penalti pada loss function, tetapi dimasukkan langsung ke struktur output model sehingga pelanggaran fisika tidak dapat dicapai oleh prediksi model.

Smooth Hard-Coulomb layer mengarahkan perubahan SOC berdasarkan tanda arus dan membatasi magnitudenya menggunakan envelope Coulomb counting. Berbeda dari constraint lama berbasis `torch.clamp`, versi final memakai magnitude berbasis sigmoid sehingga tetap differentiable dan mengurangi masalah dead gradient.

## Hipotesis Utama

Hipotesis penelitian ini adalah:

1. Model sequence biasa seperti LSTM dan TCN dapat mengoptimalkan error prediksi, tetapi tidak menjamin keselamatan fisika.
2. Soft physics penalty atau Soft-PINN dapat mengurangi pelanggaran, tetapi tidak dapat menjamin Physics Violation Rate (PVR) menjadi nol.
3. Constraint struktural berbasis Coulomb dapat menjamin `0.00%` PVR selama asumsi arus dan timestep valid.
4. Error besar yang masih muncul pada `-20 C` lebih tepat dijelaskan sebagai masalah observability pada anchor SOC awal, bukan kegagalan constraint Coulomb.

## Ide Model Final

Pada setiap timestep, model tidak langsung memprediksi delta SOC bertanda. Model memprediksi raw logits, lalu Smooth Hard-Coulomb layer mengubah logits tersebut menjadi perubahan SOC yang diarahkan oleh arus dan dibatasi oleh envelope Coulomb.

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
+L_t m_t, & I_t > I_{\mathrm{th}} \\
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

Makna ringkas:

- $L_t$ adalah batas perubahan SOC berdasarkan arus dan kapasitas nominal.
- $m_t$ membuat magnitude tetap berada dalam envelope dan tetap memiliki gradient.
- Tanda $\Delta \widehat{\mathrm{SOC}}_t$ ditentukan oleh arus, sehingga trajectory tidak dapat melanggar arah fisika.
- $\widehat{\mathrm{SOC}}_{\mathrm{anchor}}$ dibatasi secara dinamis agar sequence SOC tetap berada dalam domain valid.

File implementasi utama:

- `src/model_v5_coulomb.py` - Smooth Hard-Coulomb LSTM dan constraint layer utama.
- `src/model_v5_coulomb_tcn.py` - Smooth Hard-Coulomb dengan backbone TCN.
- `src/model_v6_contextual.py` - varian contextual anchor.
- `src/preprocessing_v4.py` - pipeline preprocessing final tanpa contextual history.
- `src/preprocessing_v5_contextual.py` - pipeline preprocessing contextual dengan history kausal.

## Dataset dan Protokol Eksperimen

Dataset yang digunakan adalah data baterai lithium-ion bergaya LG HG2 dengan beberapa temperatur operasi: `40 C`, `25 C`, `10 C`, `0 C`, `-10 C`, dan `-20 C`. Untuk koreksi dosen, repository sebaiknya menyertakan **kode akuisisi/preprocessing**, `data/README.md`, metadata preprocessing kecil, audit table, dan figure hasil analisis. Dataset mentah penuh berukuran sekitar `835 MB`, sedangkan array hasil preprocessing sekitar `756 MB`; karena itu data besar sebaiknya tidak dikomit langsung ke Git biasa. Jika dosen perlu menjalankan ulang dari nol, data mentah dapat diletakkan secara lokal di `data/raw/` mengikuti instruksi pada `data/README.md`, atau dibagikan melalui Git LFS/Google Drive/OneDrive sebagai data bundle terpisah.

Fokus eksperimen adalah menguji apakah model SOC tetap akurat dan aman secara fisika ketika menghadapi kondisi temperatur yang berbeda. Karena itu penelitian memakai dua skenario pembagian data:

### Skenario A: Out-of-Distribution Temperature Generalization

Skenario A adalah skenario generalisasi temperatur ekstrem. Model hanya dilatih pada temperatur sedang/hangat tertentu, lalu diuji pada temperatur yang tidak muncul dalam training.

Pembagian Skenario A:

| Split | Temperatur | Tujuan |
|---|---|---|
| Train | `25 C`, `10 C` | Melatih model pada temperatur yang relatif informatif. |
| Validation | `25 C`, `10 C`, `0 C` | Memilih model terbaik dan menguji transisi ke suhu lebih dingin. |
| Test | `40 C`, `-10 C`, `-20 C` | Menguji generalisasi ke temperatur yang tidak dipakai training. |

Makna Skenario A:

- `-20 C` menjadi target paling sulit karena terminal voltage sangat terdistorsi oleh efek temperatur rendah.
- `40 C`, `-10 C`, dan `-20 C` tidak muncul di training, sehingga test benar-benar mengukur kemampuan OOD temperature generalization.
- Skenario A dipakai untuk membuktikan masalah utama penelitian: model biasa dapat gagal secara fisika ketika distribusi temperatur berubah.

### Skenario B: In-Distribution Multi-Temperature Evaluation

Skenario B adalah skenario evaluasi dalam distribusi yang lebih seimbang. Semua temperatur muncul pada train, validation, dan test, tetapi dipisahkan berdasarkan segmen waktu/window agar tidak terjadi overlap temporal.

Pembagian Skenario B:

| Split | Temperatur | Tujuan |
|---|---|---|
| Train | `40 C`, `25 C`, `10 C`, `0 C`, `-10 C`, `-20 C` | Melatih model pada semua kondisi temperatur. |
| Validation | `40 C`, `25 C`, `10 C`, `0 C`, `-10 C`, `-20 C` | Early stopping dan pemilihan model. |
| Test | `40 C`, `25 C`, `10 C`, `0 C`, `-10 C`, `-20 C` | Menguji performa pada semua temperatur dengan data yang tidak overlap. |

Makna Skenario B:

- Skenario B menguji apakah constraint Hard-Coulomb tetap menjaga PVR `0.00%` saat seluruh rentang temperatur masuk ke training.
- Karena semua temperatur muncul di training, Skenario B lebih mudah daripada Skenario A dari sisi generalisasi temperatur.
- Skenario B tetap penting karena menunjukkan apakah constraint bekerja konsisten pada distribusi yang lebih lengkap.

## Preprocessing dan Zero-Leakage Windowing

Pipeline final menggunakan `src/preprocessing_v4.py` untuk model non-contextual dan `src/preprocessing_v5_contextual.py` untuk model contextual. Protokol utama adalah **split-before-windowing**.

Urutan pipeline:

1. Membaca data mentah per file dan temperatur.
2. Membersihkan baris NaN dan duplikasi timestamp.
3. Menyeragamkan sampling ke `1 Hz`.
4. Membuat fitur utama: `V_proxy`, `Current`, `Temperature`, `dV_proxy_dt`, dan `dI_dt`.
5. Membagi data menjadi train, validation, dan test sesuai Skenario A/B.
6. Baru setelah split selesai, membuat sliding window dengan panjang `100` detik dan stride `10` detik.

Alasan split-before-windowing penting:

> Jika window dibuat sebelum split, dua window yang sangat mirip dapat masuk ke train dan test sekaligus. Itu menyebabkan temporal leakage karena sebagian besar sampel mentahnya overlap.

Pada pipeline ini, window dibuat setelah data dipisah. Metadata final menunjukkan overlap train/validation/test bernilai nol:

| Skenario | Train-Val overlap | Train-Test overlap | Val-Test overlap |
|---|---:|---:|---:|
| A | 0 | 0 | 0 |
| B | 0 | 0 | 0 |

Fitur `V_proxy` digunakan untuk mengurangi distorsi Ohmic drop pada terminal voltage:

$$
V_{\mathrm{proxy}} = V_{\mathrm{terminal}} - I \cdot R_{\mathrm{int}}(T)
$$

Nilai $R_{\mathrm{int}}(T)$ berbeda per temperatur dan diekstraksi dari profil HPPC/rest. Tujuannya bukan memasukkan label SOC, tetapi memperbaiki observabilitas voltage terhadap SOC.

## Komposisi Data dan Jumlah Window

Audit data mentah menunjukkan tidak ada pelanggaran batas voltage, spike arus ekstrem, atau anomali SOC sebelum clipping pada temperatur yang diaudit.

| Temperatur | File drive | Raw rows | Dropped/NaN rows | Voltage violation | Current spike | SOC anomaly |
|---|---:|---:|---:|---:|---:|---:|
| `40 C` | 12 | 633,806 | 22 | 0 | 0 | 0 |
| `25 C` | 12 | 927,095 | 23 | 0 | 0 | 0 |
| `10 C` | 11 | 847,097 | 34 | 0 | 0 | 0 |
| `0 C` | 11 | 778,005 | 21 | 0 | 0 | 0 |
| `-10 C` | 12 | 760,615 | 32 | 0 | 0 | 0 |
| `-20 C` | 12 | 552,061 | 12 | 0 | 0 | 0 |

Jumlah sliding window final untuk `v4`:

| Skenario | Temperatur | Train windows | Validation windows | Test windows |
|---|---|---:|---:|---:|
| A | `40 C` | 0 | 0 | 6,148 |
| A | `25 C` | 8,199 | 790 | 0 |
| A | `10 C` | 7,442 | 675 | 0 |
| A | `0 C` | 0 | 7,619 | 0 |
| A | `-10 C` | 0 | 0 | 7,387 |
| A | `-20 C` | 0 | 0 | 5,379 |
| B | `40 C` | 4,245 | 448 | 1,079 |
| B | `25 C` | 6,346 | 789 | 1,712 |
| B | `10 C` | 5,750 | 674 | 1,520 |
| B | `0 C` | 5,286 | 618 | 1,395 |
| B | `-10 C` | 5,107 | 597 | 1,342 |
| B | `-20 C` | 3,723 | 412 | 962 |

Total window `v4`:

| Skenario | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| A | 15,641 | 9,084 | 18,914 | 43,639 |
| B | 30,457 | 3,538 | 8,010 | 42,005 |

Untuk model contextual (`v5_contextual`), pipeline menambahkan konteks anchor 60 detik. Karena membutuhkan history penuh, jumlah window sedikit berbeda:

| Skenario | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| A contextual | 15,425 | 8,803 | 18,596 | 42,824 |
| B contextual | 29,740 | 3,030 | 7,406 | 40,176 |

Contextual features dihitung setelah split sehingga tidak mengambil informasi dari validation/test ke training. Metadata `v5_contextual` juga mencatat `context_computed_after_split: true` dan overlap key train/validation/test bernilai `0`.

## Hasil Utama

Hasil evaluasi final setelah refactor Smooth Hard-Coulomb disimpan dalam file JSON kecil di folder `outputs/`.

| Model | Skenario | RMSE (%) | MaxE (%) | PVR (%) | Parameter |
|---|---|---:|---:|---:|---:|
| Vanilla LSTM | A | 13.3712 | 51.0242 | 49.9694 | 53,569 |
| Hard-Coulomb LSTM | A | 12.7107 | 55.1126 | 0.0000 | 54,626 |
| Hard-Coulomb TCN | A | 11.4587 | 46.7298 | 0.0000 | 208,546 |
| Vanilla LSTM | B | 7.2806 | 48.7994 | 41.0552 | 53,569 |
| Hard-Coulomb LSTM | B | 8.5667 | 34.9985 | 0.0000 | 54,626 |
| Hard-Coulomb TCN | B | 8.5823 | 39.4864 | 0.0000 | 208,546 |

Interpretasi hasil:

- Vanilla LSTM masih memiliki PVR tinggi, yaitu sekitar `49.97%` pada Skenario A dan `41.06%` pada Skenario B.
- Hard-Coulomb LSTM dan Hard-Coulomb TCN mencapai `0.00%` PVR.
- TCN memberi peningkatan pada Skenario A, tetapi memakai parameter lebih banyak daripada LSTM.
- Error ekstrem pada `-20 C` masih ada dan dijelaskan sebagai masalah anchor/observability.

File metrik utama:

- `outputs/v7_final/sprint48_evaluation_results.json`
- `outputs/v7_final/sprint48_safety_ablation_results.json`
- `outputs/v5_contextual/sprint50_contextual/sprint50_contextual_results.json`
- `outputs/v8_tcn_redemption/sprint52/sprint52_tcn_redemption_results.json`

## Struktur Repositori

```text
data/
  README.md                         Panduan penempatan dataset mentah dan reproduksi data.
  raw/                              Folder lokal untuk dataset mentah LG HG2, tidak ikut Git biasa.
  processed/                        Folder lokal hasil preprocessing, dibuat ulang oleh script.

src/
  config.py                         Konfigurasi umum penelitian.
  hppc_rint_extractor.py            Dukungan fitur R_int dan V_proxy.
  preprocessing_v4.py               Pipeline non-contextual final dan zero leakage.
  preprocessing_v5_contextual.py     Pipeline contextual dengan history kausal.
  model_v5_coulomb.py               Smooth Hard-Coulomb LSTM.
  model_v5_coulomb_tcn.py           Smooth Hard-Coulomb TCN.
  model_v6_contextual.py            Model Contextual Hard-Coulomb.
  sprint48_train_scenario_A.py      Training Vanilla/Hard-Coulomb LSTM Skenario A.
  sprint48_train_scenario_B.py      Training Vanilla/Hard-Coulomb LSTM Skenario B.
  sprint48_evaluate_all.py          Evaluasi final LSTM dan sertifikasi PVR.
  sprint48_safety_ablation.py       Ablasi safety factor eta ($\eta$).
  sprint50_train_contextual.py      Eksperimen contextual anchor.
  sprint52_tcn_redemption.py        Evaluasi TCN dan contextual TCN.

notebooks/
  05_q1_eda_money_plots.py          Figure EDA untuk observability collapse.
  ablation_studies/                 Sebelas notebook ablasi forensik.

tools/
  generate_data_audit_tables.py     Tabel raw integrity dan scenario composition.

outputs/
  data_audit_tables.json            Ledger audit data berukuran kecil.
  figures/                          Figure pilihan untuk koreksi.
  v7_final/                         Metrik final LSTM.
  v5_contextual/                    Metrik contextual.
  v8_tcn_redemption/                Metrik final TCN.
```

Folder data besar, checkpoint model, draft manuskrip jurnal, dan archive eksperimen lama tidak disertakan dalam paket koreksi source code. Yang disertakan untuk audit adalah kode pipeline, metadata kecil, tabel audit, log metrik JSON/CSV, notebook ablasi, dan figure pilihan.

## Setup Environment

Versi Python yang disarankan: Python 3.10 atau lebih baru.

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline Reproduksi

Pipeline penuh membutuhkan dataset baterai bergaya LG HG2 di folder lokal `data/raw/`. Dataset mentah penuh dan array hasil preprocessing tidak dikomit melalui Git biasa karena ukurannya besar. Untuk koreksi, dosen tetap dapat melihat cara data ditangani melalui `data/README.md`, `tools/generate_data_audit_tables.py`, metadata preprocessing, dan tabel audit di `outputs/data_audit_tables.md`.

Rekomendasi pembagian data untuk koreksi:

| Jenis file | Status GitHub | Alasan |
|---|---|---|
| Kode akuisisi, preprocessing, training, evaluasi | Ikut GitHub | Diperlukan untuk memeriksa alur penelitian dari awal sampai akhir. |
| `data/README.md` | Ikut GitHub | Menjelaskan sumber data, struktur folder, dan cara menjalankan ulang pipeline. |
| Metadata preprocessing `metadata*.json` | Ikut GitHub | Bukti split-before-windowing, jumlah window, fitur, dan zero-overlap tanpa membawa array besar. |
| Audit table dan log metrik JSON/CSV | Ikut GitHub | Bukti hasil eksperimen dan validasi. |
| Dataset mentah penuh | Jangan commit Git biasa; gunakan Git LFS atau link data terpisah | Ukuran besar dan membuat clone repository berat. |
| Array `.npy` hasil preprocessing | Tidak perlu ikut GitHub | Dapat dibuat ulang dari script preprocessing. |
| Checkpoint `.pt/.pth` | Tidak perlu ikut GitHub | Besar dan semantik checkpoint mudah kadaluarsa setelah refactor model. |

### 1. Membuat Array Hasil Preprocessing

```bash
python src/preprocessing_v4.py
python src/preprocessing_v5_contextual.py
```

### 2. Melatih Model LSTM Final

```bash
python src/sprint48_train_scenario_A.py
python src/sprint48_train_scenario_B.py
```

### 3. Melatih Model Contextual Anchor

```bash
python src/sprint50_train_contextual.py
```

### 4. Melatih/Mengevaluasi Model TCN

```bash
python src/sprint52_tcn_redemption.py
```

### 5. Evaluasi Final

```bash
python src/sprint48_evaluate_all.py
```

### 6. Membuat Tabel Audit Data dan Figure EDA

```bash
python tools/generate_data_audit_tables.py
python notebooks/05_q1_eda_money_plots.py
```

## Studi Ablasi

Notebook ablasi menjelaskan mengapa arsitektur final diperlukan.

| Notebook | Tujuan |
|---|---|
| `01_Seq2Point_Windowing_Artifact_and_Pseudo_PVR.ipynb` | Menunjukkan artifact pseudo-trajectory dari estimasi SOC pointwise. |
| `02_Vanilla_LSTM_TCN_Physics_Blindness.ipynb` | Menunjukkan PVR tinggi pada model sequence tanpa constraint. |
| `03_Soft_PINN_Penalty_Gradient_Collision.ipynb` | Menunjukkan soft penalty mengurangi pelanggaran tetapi tidak menjamin nol. |
| `04_Direction_Only_Hard_Constraint_Cumulative_Drift.ipynb` | Menunjukkan constraint arah saja dapat drift jika magnitude tidak dibatasi. |
| `05_Hard_Clamp_vs_Smooth_Coulomb_Gradient_Pathology.ipynb` | Menjelaskan mengapa sigmoid magnitude lebih baik daripada clamp. |
| `06_Vproxy_HPPC_Rint_Feature_Defense.ipynb` | Membela fitur V_proxy dan kompensasi Ohmic drop. |
| `07_Zero_Leakage_Split_Before_Windowing_Forensics.ipynb` | Memverifikasi split-before-windowing dan nol overlap temporal. |
| `08_Anchor_Trap_and_Safety_Factor_NonCause.ipynb` | Menunjukkan MaxE dingin bukan terutama karena $\eta$ terlalu sempit. |
| `09_Contextual_Anchor_OCV_Rest_vs_History.ipynb` | Menunjukkan OCV-rest context lebih berguna daripada history-only context. |
| `10_Gated_Context_Negative_Result_Sparse_Rest_Validity.ipynb` | Menjelaskan kegagalan gated context akibat bukti rest yang sparse pada suhu dingin. |
| `11_HardCoulomb_LSTM_vs_TCN_Backbone_Tradeoff.ipynb` | Membandingkan backbone LSTM dan TCN dengan constraint yang sama. |

## Figure Penting untuk Koreksi

Figure pilihan yang disimpan untuk inspeksi cepat:

- `outputs/figures/fig_q1_observability_collapse.png`
- `outputs/figures/fig_q1_transient_dynamic_profile.png`
- `outputs/figures/ablation_studies/fig_02_vanilla_physics_blindness_pvr.png`
- `outputs/figures/ablation_studies/fig_05_clamp_vs_smooth_gradient_pathology.png`
- `outputs/figures/ablation_studies/fig_08_anchor_trap_eta_non_cause.png`
- `outputs/figures/ablation_studies/fig_11_hardcoulomb_lstm_vs_tcn_tradeoff.png`

## Limitasi Penelitian

Repositori ini siap untuk koreksi teknis, tetapi penelitian masih memiliki beberapa limitasi:

- Belum ada baseline observer klasik seperti EKF, UKF, atau ECM pada perbandingan final.
- Hasil final belum dilaporkan dengan variasi multi-seed.
- Jumlah parameter menunjukkan potensi embedded/TinyML, tetapi belum ada validasi latency, WCET, quantization, dan hardware-in-the-loop.
- Jaminan PVR bergantung pada asumsi bahwa arus, timestep, dan threshold arus valid.
- MaxE tinggi pada `-20 C` masih menjadi bottleneck observability karena anchor SOC sulit ditentukan dari terminal voltage pada suhu ekstrem.

## Panduan Koreksi untuk Dosen

Urutan file yang disarankan untuk diperiksa:

1. `src/model_v5_coulomb.py` - cek logika Smooth Hard-Coulomb.
2. `src/preprocessing_v4.py` - cek split-before-windowing dan zero leakage.
3. `src/sprint48_evaluate_all.py` - cek perhitungan RMSE, MaxE, dan PVR.
4. `notebooks/ablation_studies/02_Vanilla_LSTM_TCN_Physics_Blindness.ipynb` - lihat bukti kegagalan baseline vanilla.
5. `notebooks/ablation_studies/11_HardCoulomb_LSTM_vs_TCN_Backbone_Tradeoff.ipynb` - lihat perbandingan final LSTM dan TCN.

Draft manuskrip jurnal sengaja diabaikan pada tahap ini karena fokus koreksi adalah proyek dan source code terlebih dahulu.
