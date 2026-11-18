"""Load artifact sekali, lalu prediksi. Dipakai FastAPI (app/api/main.py) dan Streamlit mode A (app/ui/streamlit_app.py)."""
import json
import sys
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# model.joblib menyimpan referensi ke src.features.tambah_fitur, jadi root repo harus bisa di-import
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "artifacts"

# Rentang sensor pada data latih (data bersih). Di luar rentang ini model masih menjawab, tapi kurang bisa dipercaya.
RENTANG_LATIH = {
    "beban_pct": (38.9, 110.0),
    "arus_A": (4.1, 47.3),
    "tegangan_V": (363.4, 396.8),
    "suhu_belitan_C": (39.6, 98.9),
    "getaran_rms_mm_s": (1.5, 8.0),
    "kecepatan_rpm": (1451.0, 1485.0),
}
# nama pustaka di metadata -> nama paket pip
_PAKET = {"sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy", "joblib": "joblib"}


def status_dari_rul(rul_hari: float, threshold: dict) -> str:
    """Aturan status dari metadata.json -> threshold (diisi saat training)."""
    if rul_hari <= threshold["segera_ganti_hari"]:
        return "segera ganti"
    if rul_hari <= threshold["pesan_pengganti_hari"]:
        return "pesan pengganti sekarang"
    return "aman"


@lru_cache(maxsize=1)
def load():
    """(model, metadata). lru_cache = di-load sekali per proses, bukan per request."""
    model_path, meta_path = ARTIFACT_DIR / "model.joblib", ARTIFACT_DIR / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} tidak ada. Sudah di-commit? Cek .gitignore.")
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return model, meta


def cek_versi() -> list[str]:
    """Bandingkan versi library saat training (metadata) dengan yang terpasang. Kosong = cocok."""
    _, meta = load()
    beda = []
    for kunci, paket in _PAKET.items():
        dilatih = meta.get("versi_library", {}).get(kunci)
        if dilatih is None:
            continue
        try:
            terpasang = version(paket)
        except PackageNotFoundError:
            terpasang = "tidak terpasang"
        if terpasang != dilatih:
            beda.append(f"{paket}: dilatih {dilatih}, terpasang {terpasang}")
    return beda


def _peringatan(row: dict) -> list[str]:
    pesan = []
    kosong = [k for k in RENTANG_LATIH if row.get(k) is None or pd.isna(row.get(k))]
    if kosong:
        pesan.append("Sensor kosong (" + ", ".join(kosong) + "): diisi nilai median data latih, hasil kurang pasti.")
    for k, (lo, hi) in RENTANG_LATIH.items():
        v = row.get(k)
        if v is not None and not pd.isna(v) and not (lo <= v <= hi):
            pesan.append(f"{k}={v} di luar rentang data latih ({lo}-{hi}): hasil kurang bisa dipercaya.")
    return pesan


def predict_rows(rows: list[dict]) -> list[dict]:
    """Prediksi beberapa baris sekaligus. Tiap baris berisi kolom fitur_input dari metadata."""
    model, meta = load()
    mulai = time.perf_counter()
    X = pd.DataFrame(rows)[meta["fitur_input"]]
    rul = np.clip(model.predict(X), 0, None)
    latency = round((time.perf_counter() - mulai) * 1000, 1)
    return [{
        "perkiraan_rul_hari": round(float(r), 1),
        "status": status_dari_rul(float(r), meta["threshold"]),
        "versi_model": meta["versi_model"],
        "peringatan": _peringatan(row),
        "latency_ms": latency,
    } for r, row in zip(rul, rows)]


def predict_one(row: dict) -> dict:
    return predict_rows([row])[0]
