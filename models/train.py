"""
Phase 4 — Model Training (XGBoost)

Why XGBoost for stock prediction?
  - Handles tabular features well out of the box (no normalization needed)
  - Fast to train and retrain daily
  - Built-in feature importance tells you which indicators matter most
  - Widely used in industry for time-series regression baselines

Key design decisions:
  - Time-series split: never shuffle. Test set is always the LAST N% of
    rows chronologically — using random split would leak future data into
    training (look-ahead bias), which is the #1 mistake in financial ML.
  - Features excluded from input: date, ticker, close (raw price),
    and target itself.
  - Artifacts saved: model .pkl + metadata JSON (hyperparams, metrics,
    feature list, train/test date ranges) so predictions are reproducible.

Usage:
  python models/train.py --ticker AAPL
  python models/train.py --ticker AAPL --test-size 0.2 --n-estimators 300
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBRegressor

PROCESSED_DATA_DIR = Path("data/processed")
ARTIFACTS_DIR      = Path("models/artifacts")

# Columns that must never be used as model input features
NON_FEATURE_COLS = {"ticker", "date", "open", "high", "low", "close", "volume",
                    "is_outlier", "target"}


def load_features(ticker: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed data for {ticker}. Run pipeline/features.py first.")
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows for {ticker}")
    return df


def time_series_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split keeping chronological order — test set is always the last block.
    Never shuffle time-series data; that would leak future prices into training.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    logger.info(
        f"Train: {train['date'].iloc[0].date()} → {train['date'].iloc[-1].date()} ({len(train)} rows) | "
        f"Test:  {test['date'].iloc[0].date()} → {test['date'].iloc[-1].date()} ({len(test)} rows)"
    )
    return train, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def train_xgboost(
    train: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBRegressor:
    X_train = train[feature_cols]
    y_train = train["target"]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info(f"Trained XGBoost on {len(X_train)} samples with {len(feature_cols)} features")
    return model


def save_artifact(
    model: XGBRegressor,
    ticker: str,
    feature_cols: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
    hyperparams: dict,
) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / f"{ticker}_xgboost.pkl"
    meta_path  = ARTIFACTS_DIR / f"{ticker}_xgboost_meta.json"

    joblib.dump(model, model_path)

    meta = {
        "ticker":       ticker,
        "model":        "xgboost",
        "trained_at":   datetime.utcnow().isoformat(),
        "train_start":  str(train["date"].iloc[0].date()),
        "train_end":    str(train["date"].iloc[-1].date()),
        "test_start":   str(test["date"].iloc[0].date()),
        "test_end":     str(test["date"].iloc[-1].date()),
        "n_train":      len(train),
        "n_test":       len(test),
        "features":     feature_cols,
        "hyperparams":  hyperparams,
        "metrics":      metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved model → {model_path}")
    logger.info(f"Saved metadata → {meta_path}")
    return model_path


def top_features(model: XGBRegressor, feature_cols: list[str], n: int = 10) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(n)
    logger.info("Top feature importances:")
    for feat, score in top.items():
        logger.info(f"  {feat:<20} {score:.4f}")


def run(ticker: str, test_size: float = 0.2, **hyperparams) -> dict:
    df = load_features(ticker)
    train, test = time_series_split(df, test_size)
    feature_cols = get_feature_cols(df)

    model = train_xgboost(train, feature_cols, **hyperparams)

    # Import evaluate here to avoid circular import
    from models.evaluate import compute_metrics
    X_test = test[feature_cols]
    y_test = test["target"]
    preds  = model.predict(X_test)
    metrics = compute_metrics(y_test.values, preds)

    logger.info("Test set metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    top_features(model, feature_cols)

    save_artifact(
        model, ticker, feature_cols, train, test, metrics,
        hyperparams or {"n_estimators": 200, "max_depth": 6,
                        "learning_rate": 0.05, "subsample": 0.8,
                        "colsample_bytree": 0.8},
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model for a ticker")
    parser.add_argument("--ticker",          required=True, help="e.g. AAPL")
    parser.add_argument("--test-size",       type=float, default=0.2)
    parser.add_argument("--n-estimators",    type=int,   default=200)
    parser.add_argument("--max-depth",       type=int,   default=6)
    parser.add_argument("--learning-rate",   type=float, default=0.05)
    args = parser.parse_args()

    run(
        ticker=args.ticker,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
