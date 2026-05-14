"""
Phase 4 — Model Evaluation

Metrics used and why:
  MAE   — average dollar error; easy to explain ("off by $X on average")
  RMSE  — penalises large errors more than MAE; good for catching bad outlier predictions
  MAPE  — percentage error; comparable across tickers with different price scales
  Dir.Accuracy — "did we predict UP or DOWN correctly?" — more actionable than
                 raw price error for trading decisions

Usage:
  python models/evaluate.py --ticker AAPL
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

ARTIFACTS_DIR      = Path("models/artifacts")
PROCESSED_DATA_DIR = Path("data/processed")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    # Directional accuracy: did prediction move in the same direction as actual?
    # Compare predicted change to actual change (vs previous actual close).
    # We shift y_true by 1 to get "previous close" as a reference.
    actual_direction    = np.sign(np.diff(y_true))
    predicted_direction = np.sign(y_pred[1:] - y_true[:-1])
    dir_acc = float(np.mean(actual_direction == predicted_direction) * 100)

    return {
        "mae":                  round(mae, 4),
        "rmse":                 round(rmse, 4),
        "mape":                 round(mape, 4),
        "directional_accuracy": round(dir_acc, 4),
    }


def load_model_and_meta(ticker: str):
    model_path = ARTIFACTS_DIR / f"{ticker}_xgboost.pkl"
    meta_path  = ARTIFACTS_DIR / f"{ticker}_xgboost_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No model artifact for {ticker}. Run models/train.py first.")
    model = joblib.load(model_path)
    meta  = json.loads(meta_path.read_text())
    return model, meta


def evaluate(ticker: str) -> dict:
    model, meta = load_model_and_meta(ticker)
    feature_cols = meta["features"]

    df = pd.read_parquet(PROCESSED_DATA_DIR / f"{ticker}.parquet")
    df = df.sort_values("date").reset_index(drop=True)

    # Reconstruct the same test split using saved metadata
    test = df[df["date"].astype(str) >= meta["test_start"]].copy()
    X_test = test[feature_cols]
    y_test = test["target"].values

    preds   = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    logger.info(f"\n{'='*45}")
    logger.info(f"  Evaluation: {ticker} (XGBoost)")
    logger.info(f"  Test period: {meta['test_start']} → {meta['test_end']}")
    logger.info(f"  Samples: {len(test)}")
    logger.info(f"{'='*45}")
    logger.info(f"  MAE:                  ${metrics['mae']:.2f}")
    logger.info(f"  RMSE:                 ${metrics['rmse']:.2f}")
    logger.info(f"  MAPE:                 {metrics['mape']:.2f}%")
    logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
    logger.info(f"{'='*45}")

    # Show a sample of predictions vs actuals
    results_df = pd.DataFrame({
        "date":      test["date"].values,
        "actual":    y_test,
        "predicted": preds,
        "error":     np.abs(y_test - preds),
    })
    logger.info("\nSample predictions (last 10 rows):")
    logger.info("\n" + results_df.tail(10).to_string(index=False))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved XGBoost model")
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    evaluate(args.ticker)
