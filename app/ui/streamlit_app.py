"""UI Streamlit untuk perencana pemeliharaan.

Jalankan dari root repo:
    streamlit run app/ui/streamlit_app.py

Dua mode:
- Mode A (API_URL kosong): model di-load langsung di sini. Paling gampang di-deploy.
- Mode B (API_URL diisi, mis. https://xxx.hf.space): memanggil FastAPI lewat HTTP.
  Isi lewat environment variable API_URL, atau di Streamlit Cloud: Settings → Secrets → API_URL = "https://..."
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Streamlit menjalankan file ini dari app/ui, jadi root repo ditambah manual
from app import predictor  # noqa: E402
from app.api.main import rapikan_tipe  # noqa: E402


def _api_url() -> str:
    try:
        url = st.secrets.get("API_URL", "")
    except Exception:  # tidak ada secrets.toml: wajar saat lokal
        url = ""
    return (url or os.getenv("API_URL", "")).rstrip("/")


API_URL = _api_url()
TIMEOUT = 60  # detik; free tier yang sedang tidur butuh waktu untuk bangun


def prediksi(rows: list[dict]) -> list[dict]:
    if not API_URL:
        return predictor.predict_rows(rows)
    path = "/predict" if len(rows) == 1 else "/predict/batch"
    r = requests.post(API_URL + path, json=rows[0] if len(rows) == 1 else rows, timeout=TIMEOUT)
    if r.status_code == 422:
        raise ValueError(f"Input ditolak API (422): {r.json().get('detail')}")
    r.raise_for_status()
    hasil = r.json()
    return [hasil] if len(rows) == 1 else hasil


def info_model() -> dict:
    if not API_URL:
        return predictor.load()[1]
    r = requests.get(API_URL + "/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Sisa Umur Pakai Motor Induksi", layout="centered")
st.title("Sisa umur pakai motor induksi")
st.caption("Isi tipe motor dan pembacaan sensor hari ini. Aplikasi memperkirakan berapa hari lagi motor masih bisa "
           "bekerja, supaya motor pengganti bisa dipesan sebelum motor lama rusak.")

try:
    META = info_model()
except Exception as e:
    st.error(f"Model belum siap: {e}")
    st.stop()

FITUR = META["fitur_input"]
SENSOR = [k for k in FITUR if k != "tipe_motor"]
MAE60 = (META.get("metrik_test") or {}).get("final_mae_rul60")
mode = f"Mode B · memanggil API {API_URL}" if API_URL else "Mode A · model di-load langsung"
st.info(f"{mode} · versi model {META['versi_model']}")
if not API_URL and (beda := predictor.cek_versi()):
    st.warning("Versi library berbeda dari saat model dilatih: " + "; ".join(beda))

STATUS_TEKS = {"segera ganti": ("error", "SEGERA GANTI"),
               "pesan pengganti sekarang": ("warning", "PESAN PENGGANTI SEKARANG"),
               "aman": ("success", "AMAN")}


def tampilkan(hasil: dict, row: dict):
    jenis, judul = STATUS_TEKS[hasil["status"]]
    rul = hasil["perkiraan_rul_hari"]
    pesan = f"### {judul}\nPerkiraan sisa umur: **{rul:.0f} hari**"
    if MAE60:
        pesan += f" (untuk motor yang mendekati rusak, rata-rata meleset sekitar ±{MAE60:.0f} hari)"
    getattr(st, jenis)(pesan)
    for p in hasil.get("peringatan", []):
        st.warning(p)
    with st.expander("Lihat request & response JSON"):
        st.json({"request": row, "response": hasil})


tab_satu, tab_batch = st.tabs(["Satu motor", "Banyak motor (CSV)"])

CONTOH = {
    "Isi manual": dict(tipe="11kW", beban=75.0, arus=16.0, tegangan=380.0, suhu=65.0, getar=2.5, rpm=1470.0),
    "Contoh: motor sehat": dict(tipe="11kW", beban=77.5, arus=17.0, tegangan=377.9, suhu=62.5, getar=2.2, rpm=1472.0),
    "Contoh: mendekati rusak": dict(tipe="11kW", beban=95.9, arus=22.4, tegangan=380.4, suhu=86.1, getar=7.1, rpm=1460.0),
}

with tab_satu:
    pilih = st.radio("Mulai dari", list(CONTOH), horizontal=True)
    c0 = CONTOH[pilih]
    with st.form("input"):
        c1, c2 = st.columns(2)
        tipe = c1.selectbox("Tipe motor (daya nominal)", ["5.5kW", "11kW", "22kW"],
                            index=["5.5kW", "11kW", "22kW"].index(c0["tipe"]), key=f"tipe_{pilih}")
        beban = c2.number_input("Beban (% dari daya nominal)", 0.0, 200.0, c0["beban"], 0.5, key=f"beban_{pilih}")
        arus = c1.number_input("Arus (A)", 0.0, 200.0, c0["arus"], 0.1, key=f"arus_{pilih}")
        tegangan = c2.number_input("Tegangan (V)", 200.0, 500.0, c0["tegangan"], 0.5, key=f"tegangan_{pilih}")
        suhu = c1.number_input("Suhu belitan (°C)", -20.0, 250.0, c0["suhu"], 0.5, key=f"suhu_{pilih}")
        getar = c2.number_input("Getaran RMS (mm/s)", 0.0, 100.0, c0["getar"], 0.1, key=f"getar_{pilih}")
        rpm = c1.number_input("Kecepatan (rpm)", 500.0, 1600.0, c0["rpm"], 1.0, key=f"rpm_{pilih}")
        rusak = st.multiselect("Sensor yang tidak terbaca / rusak (dikirim kosong)", SENSOR,
                               help="Model tetap menjawab dengan mengisi nilai tengah data latih, tapi hasilnya kurang pasti.")
        kirim = st.form_submit_button("Perkirakan sisa umur", type="primary", use_container_width=True)

    if kirim:
        row = {"tipe_motor": tipe, "beban_pct": beban, "arus_A": arus, "tegangan_V": tegangan,
               "suhu_belitan_C": suhu, "getaran_rms_mm_s": getar, "kecepatan_rpm": rpm}
        for k in rusak:
            row[k] = None
        try:
            hasil = prediksi([row])[0]
        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")
        else:
            tampilkan(hasil, row)

with tab_batch:
    st.write("Upload CSV dengan kolom: " + ", ".join(f"`{k}`" for k in FITUR))
    contoh = ROOT / "data" / "contoh_batch.csv"
    if contoh.exists():
        st.download_button("Unduh contoh CSV", contoh.read_bytes(), "contoh_batch.csv", "text/csv")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        kurang = [k for k in FITUR if k not in df.columns]
        if kurang:
            st.error(f"Kolom kurang: {kurang}")
        else:
            df["tipe_motor"] = df["tipe_motor"].map(rapikan_tipe)
            X = df[FITUR]
            rows = X.astype(object).where(X.notna(), None).to_dict("records")  # NaN → null supaya valid JSON
            try:
                hasil = pd.DataFrame(prediksi(rows))
            except Exception as e:
                st.error(f"Gagal memprediksi: {e}")
            else:
                out = pd.concat([df.reset_index(drop=True), hasil[["perkiraan_rul_hari", "status"]]], axis=1)
                out = out.sort_values("perkiraan_rul_hari")
                n1 = int((out["status"] == "segera ganti").sum())
                n2 = int((out["status"] == "pesan pengganti sekarang").sum())
                c1, c2 = st.columns(2)
                c1.metric("Segera ganti", f"{n1} dari {len(out)}")
                c2.metric("Pesan pengganti sekarang", f"{n2} dari {len(out)}")
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("Unduh hasil", out.to_csv(index=False).encode(), "hasil_prediksi.csv", "text/csv")
