"""
Phase 1 — Data Ingestion

Fetches OHLCV data from Yahoo Finance via yfinance and saves it as
Parquet files partitioned by ticker/year/month.

Key design decisions:
  - Incremental loading: if a Parquet partition already exists for a
    ticker+month, only missing dates within that month are appended.
    Re-running the script is idempotent — no duplicate rows.
  - Schema validation: every fetched row is validated via Pydantic
    before being written to disk.
  - Partition layout: data/raw/<TICKER>/<YEAR>/<MONTH>.parquet
    This mirrors how real lakehouses (S3/GCS) partition time-series data
    and makes range queries efficient.

Usage:
  python ingestion/fetch_stocks.py                        # all tickers, last 365 days
  python ingestion/fetch_stocks.py --tickers AAPL MSFT   # specific tickers
  python ingestion/fetch_stocks.py --days 730            # custom lookback
  python ingestion/fetch_stocks.py --start 2023-01-01 --end 2024-01-01
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from pydantic import ValidationError

from ingestion.schema import OHLCV_COLUMNS, OHLCVRecord

RAW_DATA_DIR = Path("data/raw")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_tickers(path: Path = Path("ingestion/tickers.txt")) -> list[str]:
    return [t.strip().upper() for t in path.read_text().splitlines() if t.strip()]


def partition_path(ticker: str, year: int, month: int) -> Path:
    return RAW_DATA_DIR / ticker / str(year) / f"{month:02d}.parquet"


def existing_dates(ticker: str, year: int, month: int) -> set[date]:
    """Return the set of dates already stored for this ticker/year/month."""
    path = partition_path(ticker, year, month)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["date"])
    return set(pd.to_datetime(df["date"]).dt.date)


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a clean DataFrame."""
    logger.info(f"Fetching {ticker} from {start} to {end}")
    raw = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=True,   # adjusts for splits/dividends — use adjusted prices
        progress=False,
    )
    if raw.empty:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()

    # yfinance returns a MultiIndex column when downloading a single ticker
    # with auto_adjust=True; flatten it.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    df = pd.DataFrame({
        "ticker": ticker,
        "date":   pd.to_datetime(raw["date"]).dt.date,
        "open":   raw["open"].astype(float),
        "high":   raw["high"].astype(float),
        "low":    raw["low"].astype(float),
        "close":  raw["close"].astype(float),
        "volume": raw["volume"].astype(int),
    })
    return df[OHLCV_COLUMNS]


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any rows that fail schema validation and log the failures."""
    valid_rows = []
    for row in df.itertuples(index=False):
        try:
            OHLCVRecord(
                ticker=row.ticker,
                date=row.date,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            valid_rows.append(row)
        except ValidationError as e:
            logger.warning(f"Skipping invalid row {row.ticker} {row.date}: {e}")
    return pd.DataFrame(valid_rows, columns=OHLCV_COLUMNS) if valid_rows else pd.DataFrame(columns=OHLCV_COLUMNS)


# ── Write ─────────────────────────────────────────────────────────────────────

def write_partition(df: pd.DataFrame, ticker: str, year: int, month: int) -> None:
    """Merge new rows with any existing Parquet partition and write back."""
    path = partition_path(ticker, year, month)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.date
        combined = combined.drop_duplicates(subset=["ticker", "date"]).sort_values("date")
    else:
        combined = df

    combined.to_parquet(path, index=False)
    logger.info(f"  Wrote {len(df)} new rows → {path}")


# ── Orchestrate ───────────────────────────────────────────────────────────────

def ingest_ticker(ticker: str, start: date, end: date) -> None:
    df = fetch_ticker(ticker, start, end)
    if df.empty:
        return

    df = validate_rows(df)
    if df.empty:
        logger.warning(f"All rows for {ticker} failed validation — nothing written")
        return

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Group by year/month and write each partition
    df["_year"]  = df["date"].apply(lambda d: d.year)
    df["_month"] = df["date"].apply(lambda d: d.month)

    for (year, month), group in df.groupby(["_year", "_month"]):
        already_stored = existing_dates(ticker, year, month)
        new_rows = group[~group["date"].isin(already_stored)].drop(columns=["_year", "_month"])
        if new_rows.empty:
            logger.debug(f"  {ticker} {year}-{month:02d}: all dates already stored, skipping")
            continue
        write_partition(new_rows, ticker, year, month)


def run(tickers: list[str], start: date, end: date) -> None:
    logger.info(f"Starting ingestion for {len(tickers)} tickers: {tickers}")
    logger.info(f"Date range: {start} → {end}")

    success, failed = [], []
    for ticker in tickers:
        try:
            ingest_ticker(ticker, start, end)
            success.append(ticker)
        except Exception as e:
            logger.error(f"Failed to ingest {ticker}: {e}")
            failed.append(ticker)

    logger.info(f"Done. Success: {success}")
    if failed:
        logger.error(f"Failed tickers: {failed}")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OHLCV stock data")
    parser.add_argument("--tickers", nargs="+", help="Override tickers from tickers.txt")
    parser.add_argument("--days",    type=int, default=365, help="Lookback window in days (default: 365)")
    parser.add_argument("--start",   type=str, help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end",     type=str, help="End date YYYY-MM-DD (default: today)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    end_date   = date.fromisoformat(args.end)   if args.end   else date.today()
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=args.days)
    tickers    = args.tickers or load_tickers()

    run(tickers, start_date, end_date)
