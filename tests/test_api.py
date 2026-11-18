"""Smoke test API: input normal, ekstrem, kosong, dan tidak valid. Jalankan: pytest -q"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

SEHAT = {"tipe_motor": "11kW", "beban_pct": 77.5, "arus_A": 17.0, "tegangan_V": 377.9,
         "suhu_belitan_C": 62.5, "getaran_rms_mm_s": 2.2, "kecepatan_rpm": 1472}
RUSAK = {"tipe_motor": "11kW", "beban_pct": 95.9, "arus_A": 22.4, "tegangan_V": 380.4,
         "suhu_belitan_C": 86.1, "getaran_rms_mm_s": 7.1, "kecepatan_rpm": 1460}
STATUS = ("segera ganti", "pesan pengganti sekarang", "aman")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` menjalankan startup: model di-load sekali
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and len(r.json()["fitur_input"]) == 7


def test_normal_answers(client):
    r = client.post("/predict", json=SEHAT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["perkiraan_rul_hari"] >= 0 and body["status"] in STATUS


def test_healthy_motor_is_safe_and_failing_motor_is_urgent(client):
    sehat = client.post("/predict", json=SEHAT).json()
    rusak = client.post("/predict", json=RUSAK).json()
    assert sehat["status"] == "aman"
    assert rusak["status"] in ("segera ganti", "pesan pengganti sekarang")
    assert rusak["perkiraan_rul_hari"] < sehat["perkiraan_rul_hari"]


def test_status_consistent_with_rul(client):
    for row in (SEHAT, RUSAK):
        b = client.post("/predict", json=row).json()
        rul = b["perkiraan_rul_hari"]
        th = client.get("/health").json()["threshold"]
        assert b["status"] == ("segera ganti" if rul <= th["segera_ganti_hari"]
                               else "pesan pengganti sekarang" if rul <= th["pesan_pengganti_hari"] else "aman")


def test_out_of_range_but_valid_answers_with_warning(client):
    r = client.post("/predict", json={**RUSAK, "getaran_rms_mm_s": 40, "suhu_belitan_C": 150})
    assert r.status_code == 200 and len(r.json()["peringatan"]) >= 2


def test_broken_sensor_null_still_answers(client):
    r = client.post("/predict", json={**SEHAT, "getaran_rms_mm_s": None, "arus_A": None})
    assert r.status_code == 200 and any("kosong" in p for p in r.json()["peringatan"])


def test_messy_tipe_is_cleaned(client):
    for t in ("11 kW", "11KW", "5,5 kW"):
        assert client.post("/predict", json={**SEHAT, "tipe_motor": t}).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("tipe_motor", "7kW"), ("kecepatan_rpm", 9000), ("beban_pct", -5), ("getaran_rms_mm_s", -1),
    ("tegangan_V", 0), ("kecepatan_rpm", "cepat"),
])
def test_invalid_is_rejected_422(client, field, value):
    r = client.post("/predict", json={**SEHAT, field: value})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == field


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in SEHAT.items() if k != "arus_A"}
    assert client.post("/predict", json=body).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json=[SEHAT, RUSAK])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict/batch", json=[]).json() == []
