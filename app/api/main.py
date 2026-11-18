"""FastAPI untuk model sisa umur pakai (RUL) motor induksi.

Jalankan dari root repo:
    uvicorn app.api.main:app --reload
Lalu buka http://127.0.0.1:8000/docs
"""
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # supaya `app` dan `src` bisa di-import
from app import predictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prediksi")

MAX_BATCH = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()  # 1 · load sekali saat startup, bukan per request
    yield


app = FastAPI(title="Prediksi Sisa Umur Pakai Motor Induksi", version="1.0.0", lifespan=lifespan)


def rapikan_tipe(v):
    """'22KW', '11 kW', '5,5 kW' -> '22kW', '11kW', '5.5kW' (sama dengan cleaning di notebook)."""
    if isinstance(v, str):
        return v.upper().replace(" ", "").replace(",", ".").replace("KW", "kW")
    return v


class InputData(BaseModel):
    """2 · Validasi: nilai mustahil secara fisika ditolak 422 sebelum menyentuh model.
    Sensor boleh null (sensor rusak): model mengisinya dengan median saat training."""
    tipe_motor: Literal["5.5kW", "11kW", "22kW"]
    beban_pct: float | None = Field(ge=0, le=200)
    arus_A: float | None = Field(ge=0, le=200)
    tegangan_V: float | None = Field(ge=200, le=500)
    suhu_belitan_C: float | None = Field(ge=-20, le=250)
    getaran_rms_mm_s: float | None = Field(ge=0, le=100)
    kecepatan_rpm: float | None = Field(ge=500, le=1600)

    @field_validator("tipe_motor", mode="before")
    @classmethod
    def _rapikan(cls, v):
        return rapikan_tipe(v)

    model_config = {"json_schema_extra": {"examples": [{
        "tipe_motor": "11kW", "beban_pct": 95.9, "arus_A": 22.4, "tegangan_V": 380.4,
        "suhu_belitan_C": 86.1, "getaran_rms_mm_s": 7.1, "kecepatan_rpm": 1460}]}}


class Prediksi(BaseModel):
    perkiraan_rul_hari: float
    status: Literal["segera ganti", "pesan pengganti sekarang", "aman"]
    versi_model: str
    peringatan: list[str]
    latency_ms: float


def _log(inputs: list[dict], outputs: list[dict]):
    """Setiap prediksi di-log: input + output, supaya bisa diaudit dan dipantau drift-nya."""
    for x, y in zip(inputs, outputs):
        log.info("prediksi %s", json.dumps({"input": x, "output": y, "ts": time.time()}))


@app.get("/health")
def health():
    _, meta = predictor.load()
    return {"status": "ok", "versi_model": meta["versi_model"], "threshold": meta["threshold"],
            "fitur_input": meta["fitur_input"], "metrik_test": meta.get("metrik_test"),
            "versi_library_beda": predictor.cek_versi()}


@app.post("/predict", response_model=Prediksi)
def predict(x: InputData):
    """3 · JSON masuk, prediksi keluar."""
    row = x.model_dump()
    hasil = predictor.predict_one(row)
    _log([row], [hasil])
    return hasil


@app.post("/predict/batch", response_model=list[Prediksi])
def predict_batch(xs: list[InputData]):
    if len(xs) > MAX_BATCH:
        raise HTTPException(413, f"Maksimal {MAX_BATCH} baris per request.")
    rows = [x.model_dump() for x in xs]
    if not rows:
        return []
    hasil = predictor.predict_rows(rows)
    _log(rows, hasil)
    return hasil
