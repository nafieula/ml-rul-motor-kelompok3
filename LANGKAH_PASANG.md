# Cara memasang hasil notebook ke repo ini

1. Jalankan `notebooks/kasus03_eda_model_colab.ipynb` di Colab (Run all, unggah `raw.csv` saat diminta).
   Ini versi terbaru: metadata-nya berisi kunci `threshold` yang dibaca aplikasi.
2. Ekstrak `hasil_kasus03.zip` (terunduh otomatis) **di root repo ini**. Isinya otomatis menempati:
   `data/clean.csv`, `artifacts/*`, `src/features.py`, `src/__init__.py`.
3. Taruh `data/raw.csv`.
4. Buka `artifacts/requirements_model.txt` dan pastikan empat baris di blok "Model" pada `requirements.txt`
   sama persis (scikit-learn, pandas, numpy, joblib).
5. `pip install -r requirements-dev.txt` lalu `python -m pytest`. Semua tes lolos = siap.
6. Isi bagian `[...]` di README.md, hapus file ini, lalu commit dan push.
