# Sisa Umur Pakai Motor Induksi (RUL) - Kelompok 3 MKB 5A

**Kasus 3:** Sisa umur pakai motor induksi · **Aplikasi:** [tautan Streamlit] · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| [Nur Adila Putri] | [2402916] | EDA dan pembersihan data (notebook), API (FastAPI), aplikasi (Streamlit) |
| [Nafila Ajmal] | [2400721] | model dan evaluasi (notebook), tes, dan deploy |

## Masalah dan pengguna

Manajer pemeliharaan tiga pabrik (Cikarang, Karawang, Purwakarta) harus memesan motor pengganti 2–4 minggu sebelum
motor lama rusak. Model ini memperkirakan **berapa hari lagi** sebuah motor masih bisa bekerja dari pembacaan sensor
hari ini, lalu memberi status: **aman**, **pesan pengganti sekarang**, atau **segera ganti**.

## Data dan pembersihan

Data sintetis `data/raw.csv`: 300 motor dipantau harian sampai rusak (79.652 baris). Hasil bersih ada di `data/clean.csv`.

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| `motor_id` tidak seragam (`M-001`, `m-001`, `M001`, spasi); terbaca 1.147 motor, seharusnya 300 | 3.185 | dinormalkan jadi `M-###` |
| `tipe_motor` tidak seragam (`22KW`, `11 kW`, `5,5 kW`) | 1.588 | dinormalkan jadi `5.5kW` / `11kW` / `22kW` |
| Suhu belitan dalam Kelvin (Plant Purwakarta) | 25.355 | dikurangi 273,15 |
| Getaran bernilai -1 (sensor error) | 241 | diubah jadi NaN, diisi median di pipeline |
| Baris duplikat (`motor_id` + `hari_operasi`) | 493 | `drop_duplicates` |
| `arus_A` kosong | 1.588 | diisi median di pipeline |
| `suhu_belitan_C` kosong | 787 | diisi median di pipeline |
| Beban > 100% (aneh tapi mungkin benar) | 788 | dipertahankan |

Data dibagi **per `motor_id`** (70/15/15 = 210/45/45 motor), supaya motor di *test* tidak pernah muncul di *train*.
`hari_operasi` tidak dipakai sebagai fitur karena bukan masukan aplikasi.

## Hasil

| Model | MAE ≤ 60 hari *validation* | MAE ≤ 60 hari *test* | MAE semua baris *validation* | MAE semua baris *test* |
|---|---|---|---|---|
| *Baseline*: regresi linear (sensor asli) | [31.65] | [30.38] | [52.45] | [55.03] |
| Gradient boosting (`HistGradientBoostingRegressor`) + fitur turunan (final) | [12.08] | [11.62] | [46.47] | [51.88] |

Aturan status (dari waktu pengiriman motor pengganti 2–4 minggu): prediksi ≤ 14 hari → **segera ganti**;
≤ 35 hari (28 hari pengiriman + 7 hari cadangan) → **pesan pengganti sekarang**; selain itu → **aman**.

## Cara menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate                  # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                  # semua tes harus lolos
uvicorn app.api.main:app --reload       # API di http://127.0.0.1:8000/docs
streamlit run app/ui/streamlit_app.py   # aplikasi di http://localhost:8501
```

Pastikan versi library di `requirements.txt` sama dengan `artifacts/requirements_model.txt`.

## Struktur repositori

```
notebooks/      satu notebook (Colab): EDA, pembersihan, baseline, model, penyimpanan artifact
data/           raw.csv, clean.csv, contoh_batch.csv
src/features.py fungsi pembuat fitur + aturan status, dipakai saat pelatihan dan saat prediksi
artifacts/      model.joblib, metadata.json, requirements_model.txt, split_motor.json
app/            predictor.py, api/main.py (FastAPI), ui/streamlit_app.py (Streamlit)
tests/          smoke test API: normal, ekstrem, kosong, tidak valid
```

Struktur API dan antarmuka diadaptasi dari repositori praktisi [mkb-ml-p3](https://github.com/nurhikam/mkb-ml-p3).

## Keterbatasan

- Data sintetis dan hanya 300 motor; belum teruji di motor sungguhan.
- Model memakai pembacaan **satu hari**. Tren beberapa hari terakhir belum dipakai.
- Untuk motor yang masih jauh dari rusak (RUL > 120 hari) galatnya besar (puluhan hari). Model hanya andal untuk keputusan
  pemesanan, yaitu saat motor mendekati rusak (MAE pada data test untuk RUL ≤ 60 hari: 11,62 hari).
- Masukan di luar rentang data latih (mis. getaran > 8 mm/s) tetap dijawab, tetapi aplikasi menampilkan peringatan.
- Ambang status 14 dan 35 hari adalah asumsi dari waktu pengiriman, belum dihitung dari biaya sebenarnya.
- Pada data validation, [X]% motor yang sebenarnya bersisa umur ≤ 28 hari masih ditandai "aman" (terlewat), dan
  [Y]% motor dengan sisa umur > 60 hari ditandai bukan "aman" (alarm palsu).
