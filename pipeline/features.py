"""
Phase 2b — Feature Engineering

Computes all technical indicators used as model inputs.

Why these features?
  - Moving averages (MA / EMA): capture trend direction and momentum
  - MACD: momentum oscillator that shows trend changes early
  - RSI: overbought/oversold signal (value 0-100; >70 = overbought, <30 = oversold)
  - Bollinger Bands: volatility envelope around a 20-day MA
  - Volume change: unusual volume often precedes price moves
  - Lag features: give the model explicit memory of past prices
  - Daily return: percentage change — more stationary than raw price
  - target: next-day close — what we're trying to predict

Usage:
  python pipeline/features.py --tickers AAPL MSFT
  python pipeline/features.py          # all tickers in data/raw/
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands

from pipeline.clean import clean

PROCESSED_DATA_DIR = Path("data/processed")


# ── Individual indicator functions ────────────────────────────────────────────
# Each function takes a DataFrame with at least a `close` column and returns
# it with new columns appended. Keeping them separate makes unit testing easy.

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for window in [7, 21, 50]:
        df[f"ma_{window}"] = SMAIndicator(df["close"], window=window).sma_indicator()
    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    for window in [12, 26]:
        df[f"ema_{window}"] = EMAIndicator(df["close"], window=window).ema_indicator()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()   # histogram — direction of momentum shift
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_pband"]  = bb.bollinger_pband()   # % position within bands (0=lower, 1=upper)
    df["bb_wband"]  = bb.bollinger_wband()   # bandwidth — high = high volatility
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    vol_ma5 = df["volume"].rolling(5).mean()
    df["vol_change"] = (df["volume"] - vol_ma5) / vol_ma5   # % deviation from 5-day avg
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 5, 10]:
        df[f"lag_{lag}"] = df["close"].shift(lag)
    return df


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    df["daily_return"] = df["close"].pct_change()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    # Shift close back by 1 day — tomorrow's close is today's target
    df["target"] = df["close"].shift(-1)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

FEATURE_STEPS = [
    add_moving_averages,
    add_ema,
    add_macd,
    add_rsi,
    add_bollinger_bands,
    add_volume_features,
    add_lag_features,
    add_daily_return,
    add_target,
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature steps in sequence. Returns the enriched DataFrame."""
    for step in FEATURE_STEPS:
        df = step(df)

    # Drop rows where any feature is NaN — happens at the start of the series
    # before windows are full (e.g. first 50 rows for ma_50).
    # The last row is also dropped because its target is NaN (no tomorrow yet).
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"  Dropped {rows_before - len(df)} rows with NaN (window warmup + last row)")
    return df


def process_ticker(ticker: str) -> None:
    """Clean → feature engineer → write processed Parquet for one ticker."""
    logger.info(f"Processing {ticker}")
    df = clean(ticker)

    # Drop outlier rows before feature engineering so they don't distort indicators
    n_outliers = df["is_outlier"].sum()
    if n_outliers:
        logger.info(f"  Removing {n_outliers} outlier rows before feature engineering")
    df = df[~df["is_outlier"]].drop(columns=["is_outlier"]).reset_index(drop=True)

    df = engineer_features(df)

    out_path = PROCESSED_DATA_DIR / f"{ticker}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"  Saved {len(df)} rows → {out_path}")


def run(tickers: list[str]) -> None:
    logger.info(f"Feature engineering for {len(tickers)} tickers")
    for ticker in tickers:
        try:
            process_ticker(ticker)
        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")


def available_tickers() -> list[str]:
    return [p.name for p in Path("data/raw").iterdir() if p.is_dir()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute features from raw OHLCV data")
    parser.add_argument("--tickers", nargs="+", help="Tickers to process (default: all in data/raw)")
    args = parser.parse_args()

    tickers = args.tickers or available_tickers()
    run(tickers)
