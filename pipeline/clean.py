"""
Phase 2a — Cleaning

Reads raw Parquet files and produces a clean DataFrame ready for feature
engineering. Two problems are handled here:

  1. Missing values — market holidays create date gaps; forward-fill is the
     standard approach (last known price carries forward).
  2. Outliers — OHLCV data can have bad prints (fat-finger trades, split
     adjustments that yfinance missed). We flag rows where the daily return
     exceeds a z-score threshold rather than silently dropping them, so the
     analyst can inspect before deciding.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

RAW_DATA_DIR = Path("data/raw")


def load_raw(ticker: str) -> pd.DataFrame:
    """Load all monthly Parquet partitions for a ticker into one DataFrame."""
    ticker_dir = RAW_DATA_DIR / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"No raw data found for {ticker} at {ticker_dir}")

    parts = sorted(ticker_dir.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet files under {ticker_dir}")

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows for {ticker} ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill price columns; back-fill volume for leading NaNs."""
    price_cols = ["open", "high", "low", "close"]
    before = df[price_cols].isna().sum().sum()

    df[price_cols] = df[price_cols].ffill()
    df["volume"] = df["volume"].ffill().bfill()

    after = df[price_cols].isna().sum().sum()
    if before > 0:
        logger.info(f"  Filled {before - after} missing price values (ffill)")
    return df


def flag_outliers(df: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """
    Add an `is_outlier` column. A row is flagged when its daily return
    is more than z_threshold standard deviations from the mean.
    We flag rather than drop — the caller decides what to do.
    """
    df = df.copy()
    daily_return = df["close"].pct_change()
    mean = daily_return.mean()
    std  = daily_return.std()

    z_score = (daily_return - mean) / std
    df["is_outlier"] = z_score.abs() > z_threshold

    n_outliers = df["is_outlier"].sum()
    if n_outliers > 0:
        dates = df.loc[df["is_outlier"], "date"].dt.date.tolist()
        logger.warning(f"  Flagged {n_outliers} outlier rows: {dates}")
    return df


def clean(ticker: str) -> pd.DataFrame:
    """Full cleaning pipeline for one ticker. Returns a clean DataFrame."""
    df = load_raw(ticker)
    df = fill_missing_values(df)
    df = flag_outliers(df)
    logger.info(f"Cleaning complete for {ticker}: {len(df)} rows, {df['is_outlier'].sum()} outliers flagged")
    return df
