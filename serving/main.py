"""
Phase 5 — FastAPI Prediction API

Endpoints:
  GET /health              — liveness check
  GET /tickers             — list tickers with trained models
  GET /predict?ticker=AAPL — next-day close prediction for a ticker
  GET /predict/all         — predictions for every available ticker

Run:
  python -m uvicorn serving.main:app --reload
  # Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from serving.predict import available_tickers, predict_next_close

app = FastAPI(
    title="Stock Market Prediction API",
    description="Next-day close price predictions powered by XGBoost + yfinance feature pipeline",
    version="1.0.0",
)


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    ticker:          str
    model:           str
    last_date:       str
    last_close:      float
    predicted_close: float
    change_pct:      float
    direction:       str
    train_end:       str
    test_mae:        float
    test_mape:       float


class HealthResponse(BaseModel):
    status:           str
    available_models: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Liveness check — also returns which tickers have trained models."""
    tickers = available_tickers()
    return {"status": "ok", "available_models": tickers}


@app.get("/tickers", tags=["Meta"])
def tickers():
    """List all tickers that have trained models ready for inference."""
    return {"tickers": available_tickers()}


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. AAPL", examples=["AAPL"])
):
    """
    Predict the next trading day's closing price for a single ticker.

    Returns the predicted price, percentage change vs last close,
    directional signal (UP/DOWN), and model accuracy metrics for reference.
    """
    try:
        result = predict_next_close(ticker.upper())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    return result


@app.get("/predict/all", tags=["Prediction"])
def predict_all():
    """Run predictions for every ticker that has a trained model."""
    results, errors = [], {}
    for ticker in available_tickers():
        try:
            results.append(predict_next_close(ticker))
        except Exception as e:
            errors[ticker] = str(e)
    return {"predictions": results, "errors": errors}
