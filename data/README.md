# Panduan Dataset

Folder ini menjelaskan cara menempatkan dataset mentah dan membuat ulang array preprocessing untuk penelitian **Smooth Hard-Coulomb Sequence Model**.

## Kebijakan Data di GitHub

Dataset mentah penuh dan array hasil preprocessing tidak dikomit melalui Git biasa karena ukurannya besar.

Estimasi ukuran lokal saat audit:

| Komponen | Ukuran perkiraan | Status GitHub |
|---|---:|---|
| `data/raw/` | 835 MB | Lokal, atau Git LFS/data bundle terpisah |
| `data/processed/` | 756 MB | Dibuat ulang dari script |
| `metadata*.json` | Kecil | Boleh ikut GitHub untuk audit |
| `outputs/data_audit_tables.*` | Kecil | Ikut GitHub untuk bukti integritas data |

Alasan: dosen tetap dapat memeriksa cara data diproses dari kode dan metadata, tanpa harus melakukan clone repository berukuran sangat besar.

## Struktur Folder yang Diharapkan

Letakkan dataset mentah LG HG2 pada struktur berikut:

```text
data/
  raw/
    LG Dataset/
      LG_HG2_Original_Dataset/
        40degC/
        25degC/
        10degC/
        0degC/
        n10degC/
        n20degC/
      LG_HG2_Prepared_Dataset/
  processed/
```

Script utama mencari data original di:

```text
data/raw/LG Dataset/LG_HG2_Original_Dataset
```

Setiap folder temperatur berisi file CSV profil drive cycle dan HPPC/rest. File HPPC dipakai untuk estimasi resistansi internal per temperatur.

## Kolom Data yang Dibutuhkan

Pipeline membaca kolom utama berikut dari CSV:

| Kolom | Fungsi |
|---|---|
| `Voltage` | Tegangan terminal baterai |
| `Current` | Arus baterai |
| `Temperature` | Temperatur operasi |
| `Capacity` | Dasar rekonstruksi SOC Coulomb-counting |
| `Time Stamp` | Rekonstruksi waktu dan sampling |

Jika timestamp tidak valid, pipeline memakai fallback indeks waktu lokal. Setelah itu data diseragamkan ke `1 Hz`.

## Transformasi Fitur Utama

Fitur final model non-contextual adalah:

```text
[V_proxy, Current, Temperature, dV_proxy_dt, dI_dt]
```

Kompensasi Ohmic drop dihitung sebagai:

$$
V_{\mathrm{proxy}} = V_{\mathrm{terminal}} - I \cdot R_{\mathrm{int}}(T)
$$

Nilai $R_{\mathrm{int}}(T)$ diekstraksi dari file HPPC/rest menggunakan `src/hppc_rint_extractor.py` atau mapping final di `src/config.py`.

## Reproduksi Preprocessing

Dari root repository, jalankan:

```bash
python src/preprocessing_v4.py
python src/preprocessing_v5_contextual.py
```

Output dibuat ke:

```text
data/processed/v4_scenario_A/
data/processed/v4_scenario_B/
data/processed/v5_contextual/scenario_A/
data/processed/v5_contextual/scenario_B/
```

File `.npy` di folder tersebut adalah hasil generate, bukan sumber primer. Jika terhapus, file dapat dibuat ulang dari dataset mentah dan script preprocessing.

## Prinsip Zero Leakage

Pipeline memakai protokol **split-before-windowing**:

1. Data mentah dibaca per temperatur dan per profile.
2. Baris NaN dan duplikasi timestamp dibersihkan.
3. Sampling diseragamkan ke `1 Hz`.
4. Fitur fisika dibuat.
5. Data dibagi menjadi train, validation, dan test.
6. Sliding window `100 s` dengan stride `10 s` dibuat setelah split selesai.

Konsekuensi matematisnya: window train tidak berbagi timestep mentah dengan window validation/test. Metadata preprocessing mencatat overlap train/validation/test bernilai `0`.

## Skenario Data

Skenario A menguji generalisasi temperatur out-of-distribution:

| Split | Temperatur |
|---|---|
| Train | `25 C`, `10 C` |
| Validation | `25 C`, `10 C`, `0 C` |
| Test | `40 C`, `-10 C`, `-20 C` |

Skenario B menguji evaluasi multi-temperatur in-distribution:

| Split | Temperatur |
|---|---|
| Train | `40 C`, `25 C`, `10 C`, `0 C`, `-10 C`, `-20 C` |
| Validation | `40 C`, `25 C`, `10 C`, `0 C`, `-10 C`, `-20 C` |
| Test | `40 C`, `25 C`, `10 C`, `0 C`, `-10 C`, `-20 C` |

## Audit Data

Untuk membuat ulang tabel integritas data dan komposisi scenario:

```bash
python tools/generate_data_audit_tables.py
```

Output audit:

```text
outputs/data_audit_tables.md
outputs/data_audit_tables.json
```

## Jika Dataset Penuh Harus Dibagikan

Untuk koreksi internal, opsi paling rapi adalah membagikan dataset melalui Google Drive/OneDrive dan menyimpan link di pesan terpisah kepada dosen. Jika dataset penuh harus berada di GitHub, gunakan Git LFS untuk pola file besar:

```bash
git lfs track "data/raw/**/*.csv"
git lfs track "data/raw/**/*.mat"
git lfs track "data/raw/**/*.zip"
```

Jangan commit array `data/processed/**/*.npy` kecuali benar-benar diminta, karena seluruh array dapat dibuat ulang dari pipeline.
