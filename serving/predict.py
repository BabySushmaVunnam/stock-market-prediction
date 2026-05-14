"""
Prediction logic — loads a saved model artifact and runs inference
on the most recent row of the feature store for a given ticker.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

ARTIFACTS_DIR      = Path("models/artifacts")
PROCESSED_DATA_DIR = Path("data/processed")


def _load_artifact(ticker: str):
    model_path = ARTIFACTS_DIR / f"{ticker}_xgboost.pkl"
    meta_path  = ARTIFACTS_DIR / f"{ticker}_xgboost_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model for {ticker}. Train it first with models/train.py.")
    model = joblib.load(model_path)
    meta  = json.loads(meta_path.read_text())
    return model, meta


def predict_next_close(ticker: str) -> dict:
    """
    Returns a prediction dict for the next trading day's close price.
    Uses the most recent row in the processed feature store as input.
    """
    ticker = ticker.upper()
    model, meta = _load_artifact(ticker)
    feature_cols = meta["features"]

    parquet_path = PROCESSED_DATA_DIR / f"{ticker}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No processed features for {ticker}. Run pipeline/features.py first.")

    df = pd.read_parquet(parquet_path)
    df = df.sort_values("date").reset_index(drop=True)
    latest = df.iloc[[-1]]   # keep as DataFrame, not Series, for model.predict

    X = latest[feature_cols]
    predicted_close = float(model.predict(X)[0])
    last_close      = float(latest["close"].iloc[0])
    last_date       = str(latest["date"].iloc[0].date() if hasattr(latest["date"].iloc[0], "date") else latest["date"].iloc[0])

    change_pct = (predicted_close - last_close) / last_close * 100
    direction  = "UP" if predicted_close > last_close else "DOWN"

    logger.info(f"{ticker}: last_close={last_close:.2f}  predicted={predicted_close:.2f}  {direction} {change_pct:+.2f}%")

    return {
        "ticker":          ticker,
        "model":           "xgboost",
        "last_date":       last_date,
        "last_close":      round(last_close, 4),
        "predicted_close": round(predicted_close, 4),
        "change_pct":      round(change_pct, 4),
        "direction":       direction,
        "train_end":       meta["train_end"],
        "test_mae":        meta["metrics"]["mae"],
        "test_mape":       meta["metrics"]["mape"],
    }


def available_tickers() -> list[str]:
    return sorted(p.stem.replace("_xgboost", "") for p in ARTIFACTS_DIR.glob("*_xgboost.pkl"))
