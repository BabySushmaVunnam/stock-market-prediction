"""
Tests for Phase 5 — FastAPI serving layer.

Uses TestClient (no real server needed) and monkeypatches predict_next_close
so tests run without requiring trained model artifacts on disk.
"""

import pytest
from fastapi.testclient import TestClient

from serving.main import app

client = TestClient(app)

MOCK_PREDICTION = {
    "ticker":          "AAPL",
    "model":           "xgboost",
    "last_date":       "2026-05-12",
    "last_close":      189.50,
    "predicted_close": 191.20,
    "change_pct":      0.897,
    "direction":       "UP",
    "train_end":       "2025-12-31",
    "test_mae":        6.35,
    "test_mape":       2.41,
}


@pytest.fixture(autouse=True)
def mock_predict(monkeypatch):
    """Patch predict_next_close and available_tickers for all tests."""
    import serving.main as mod
    monkeypatch.setattr(mod, "predict_next_close", lambda ticker: MOCK_PREDICTION)
    monkeypatch.setattr(mod, "available_tickers",  lambda: ["AAPL", "MSFT", "NVDA"])


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_health_lists_tickers():
    r = client.get("/health")
    assert "AAPL" in r.json()["available_models"]


# ── /tickers ──────────────────────────────────────────────────────────────────

def test_tickers_endpoint():
    r = client.get("/tickers")
    assert r.status_code == 200
    assert isinstance(r.json()["tickers"], list)


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_200():
    r = client.get("/predict?ticker=AAPL")
    assert r.status_code == 200

def test_predict_response_schema():
    r = client.get("/predict?ticker=AAPL")
    body = r.json()
    for field in ["ticker", "predicted_close", "last_close", "change_pct",
                  "direction", "test_mae", "test_mape"]:
        assert field in body, f"Missing field: {field}"

def test_predict_direction_values():
    r = client.get("/predict?ticker=AAPL")
    assert r.json()["direction"] in ("UP", "DOWN")

def test_predict_missing_ticker_returns_404(monkeypatch):
    import serving.main as mod
    monkeypatch.setattr(mod, "predict_next_close",
                        lambda ticker: (_ for _ in ()).throw(FileNotFoundError("no model")))
    r = client.get("/predict?ticker=XYZ")
    assert r.status_code == 404

def test_predict_requires_ticker_param():
    r = client.get("/predict")
    assert r.status_code == 422   # FastAPI validation error


# ── /predict/all ─────────────────────────────────────────────────────────────

def test_predict_all_returns_list():
    r = client.get("/predict/all")
    assert r.status_code == 200
    assert isinstance(r.json()["predictions"], list)

def test_predict_all_no_errors():
    r = client.get("/predict/all")
    assert r.json()["errors"] == {}
