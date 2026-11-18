"""Fitur turunan RUL motor. Dipakai saat pelatihan DAN saat prediksi (app)."""
import numpy as np
import pandas as pd

SENSOR = ["beban_pct", "arus_A", "tegangan_V", "suhu_belitan_C", "getaran_rms_mm_s", "kecepatan_rpm"]
TIPE = ["5.5kW", "11kW", "22kW"]
ARUS_NOMINAL = {"5.5kW": 8.1, "11kW": 16.2, "22kW": 31.5}   # median arus per tipe pada data bersih
RPM_SINKRON = 1500                                          # motor 4 kutub, 50 Hz


def buat_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """df berisi kolom tipe_motor + SENSOR. Mengembalikan tabel fitur numerik (NaN dibiarkan)."""
    X = df[SENSOR].astype(float).copy()
    for t in TIPE:
        X["tipe_" + t] = (df["tipe_motor"] == t).astype(int)
    nominal = df["tipe_motor"].map(ARUS_NOMINAL).astype(float)
    X["arus_rel"] = X["arus_A"] / nominal                         # arus dibanding arus normal tipenya
    X["arus_per_beban"] = X["arus_A"] / X["beban_pct"]            # arus naik tanpa beban naik = tanda masalah
    X["slip_rpm"] = RPM_SINKRON - X["kecepatan_rpm"]              # makin lambat = makin berat
    X["getaran_x_suhu"] = X["getaran_rms_mm_s"] * X["suhu_belitan_C"]
    return X
