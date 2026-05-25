# Physics-Informed Temporal Convolutional Network Berbasis Sequence-to-Sequence untuk Estimasi State of Charge Baterai Li-ion pada Kondisi Suhu Ekstrem

Repositori ini berisi kode sumber, data, dan dokumentasi untuk penelitian mengenai estimasi **State of Charge (SOC)** baterai Lithium-ion menggunakan arsitektur **Physics-Informed Temporal Convolutional Network (PI-TCN)** dengan pendekatan **Sequence-to-Sequence**. Fokus utama dari penelitian ini adalah mengevaluasi dan meningkatkan performa model pada kondisi suhu ekstrem (Out-of-Distribution/OOD) yang menantang, seperti −20°C.

## 📌 Latar Belakang & Motivasi

Estimasi State of Charge (SOC) baterai Li-ion sangat krusial untuk Battery Management System (BMS). Namun, estimasi ini sangat rentan terhadap *dataset shift*, terutama ketika baterai dioperasikan pada suhu ekstrem yang belum pernah dilihat selama fase pelatihan (contoh: kondisi −20°C). 

Penelitian ini mengeksplorasi:
1. **Generalisasi Kondisi Suhu Ekstrem**: Mengevaluasi model baseline (LSTM dan TCN) saat berhadapan dengan OOD.
2. **Keterbatasan Model *Sequence-to-Point***: Menganalisis mengapa model tradisional berbasis *sequence-to-point* kesulitan mempertahankan konsistensi hukum fisika (monotonisitas SOC pada *pure discharge*).
3. **Pendekatan *Sequence-to-Sequence* + PI-TCN**: Mengusulkan arsitektur TCN berbasis *sequence-to-sequence* yang diintegrasikan dengan *Physics-Informed Learning*. Model ini memprediksi satu urutan (sequence) penuh dalam satu waktu, memungkinkan *constraint* monotonisitas fisika ditegakkan secara lebih koheren dan langsung antar *timestep*.

## 📁 Struktur Repositori

```text
├── data/
│   ├── raw/             # Data mentah benchmark (LG HG2 dataset)
│   └── processed/       # Data hasil pra-pemrosesan yang siap dilatih
├── logs/                # Log hasil eksperimen dan metrik pelatihan
├── notebooks/           # Jupyter notebooks untuk analisis data (EDA) & visualisasi
├── outputs/             # Hasil prediksi, checkpoint model, dan plot visualisasi
├── src/                 # Kode sumber utama
│   ├── config.py           # Konfigurasi parameter eksperimen dan hyperparameter
│   ├── preprocessing.py    # Skrip pipeline pra-pemrosesan data baterai
│   ├── model.py            # Definisi arsitektur model (LSTM, TCN, PI-TCN Seq2Seq)
│   ├── train.py            # Skrip utama untuk pelatihan model
│   ├── evaluate.py         # Skrip untuk evaluasi dan pengujian model
│   ├── sprint44_ablation.py # Skrip khusus untuk studi ablasi komprehensif
│   └── smoke_test_fixes.py # Skrip pengujian validitas struktural (Sanity Check)
├── JurnalDraft.md       # Draf naskah jurnal / paper penelitian
├── requirements.txt     # Daftar dependensi library Python
└── README.md            # Dokumentasi repositori ini
```

## ⚙️ Instalasi & Persiapan

1. Pastikan Anda telah menginstal Python 3.8 atau lebih baru.
2. Buat *virtual environment* (sangat disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```
3. Instal semua dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Penggunaan

### 1. Pra-pemrosesan Data
Jalankan skrip preprocessing untuk mengekstrak fitur, melakukan *windowing*, dan membagi dataset (Train, Validation, Test OOD).
```bash
python src/preprocessing.py
```

### 2. Pelatihan Model
Untuk memulai proses pelatihan model berdasarkan konfigurasi di `config.py`:
```bash
python src/train.py
```

### 3. Studi Ablasi (Eksperimen Utama)
Skrip ini akan menjalankan berbagai skenario pengujian (baseline vs PI-TCN) untuk menghasilkan bukti empiris penelitian:
```bash
python src/sprint44_ablation.py
```

### 4. Evaluasi & Visualisasi
Evaluasi hasil prediksi model pada set pengujian dan visualisasikan performa (RMSE, MAE, R², kurva SOC):
```bash
python src/evaluate.py
```

## 📊 Dataset Benchmark
Penelitian ini menggunakan **Dataset Baterai Li-ion LG HG2** yang dikumpulkan oleh McMaster University. Dataset ini mencakup berbagai profil berkendara dan profil pengisian daya pada berbagai suhu ambient (-20°C hingga 40°C).

Referensi Dataset: *Kollmeyer, P. J., Preindl, M., & Emadi, A. (2020). Lithium-ion battery dataset for state of charge estimation. Mendeley Data, v1.*

## 📝 Kontak
Untuk pertanyaan atau diskusi lebih lanjut mengenai penelitian ini, silakan ajukan di bagian *Issues* repositori ini.
