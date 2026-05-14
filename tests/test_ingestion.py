"""
Tests for Phase 1 — Data Ingestion.

We test the schema validation and partition logic without hitting the
network by constructing DataFrames directly.
"""

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ingestion.schema import OHLCVRecord, OHLCV_COLUMNS
from ingestion.fetch_stocks import (
    partition_path,
    existing_dates,
    validate_rows,
    write_partition,
)


@pytest.fixture(autouse=True)
def tmp_raw_dir(tmp_path, monkeypatch):
    """Redirect RAW_DATA_DIR to a temp folder for every test."""
    import ingestion.fetch_stocks as mod
    monkeypatch.setattr(mod, "RAW_DATA_DIR", tmp_path / "raw")
    yield tmp_path / "raw"


# ── Schema validation ─────────────────────────────────────────────────────────

def test_valid_record():
    rec = OHLCVRecord(ticker="AAPL", date=date(2024, 1, 2), open=180.0,
                      high=185.0, low=179.0, close=183.0, volume=50_000_000)
    assert rec.close == 183.0


def test_negative_price_rejected():
    with pytest.raises(Exception):
        OHLCVRecord(ticker="AAPL", date=date(2024, 1, 2), open=-1.0,
                    high=185.0, low=179.0, close=183.0, volume=50_000_000)


def test_high_less_than_low_rejected():
    with pytest.raises(Exception):
        OHLCVRecord(ticker="AAPL", date=date(2024, 1, 2), open=180.0,
                    high=100.0, low=200.0, close=183.0, volume=50_000_000)


def _sample_df(ticker="AAPL", dates=None):
    dates = dates or [date(2024, 1, 2), date(2024, 1, 3)]
    return pd.DataFrame([
        {"ticker": ticker, "date": d, "open": 180.0, "high": 185.0,
         "low": 179.0, "close": 183.0, "volume": 50_000_000}
        for d in dates
    ], columns=OHLCV_COLUMNS)


# ── Row validation ────────────────────────────────────────────────────────────

def test_validate_rows_passes_good_data():
    df = _sample_df()
    result = validate_rows(df)
    assert len(result) == len(df)


def test_validate_rows_drops_bad_rows():
    df = _sample_df()
    df.loc[0, "close"] = -1.0   # inject a bad row
    result = validate_rows(df)
    assert len(result) == 1


# ── Partition read/write ───────────────────────────────────────────────────────

def test_write_then_read_partition(tmp_raw_dir):
    df = _sample_df()
    write_partition(df, "AAPL", 2024, 1)
    stored = existing_dates("AAPL", 2024, 1)
    assert date(2024, 1, 2) in stored
    assert date(2024, 1, 3) in stored


def test_incremental_write_no_duplicates(tmp_raw_dir):
    df1 = _sample_df(dates=[date(2024, 1, 2)])
    df2 = _sample_df(dates=[date(2024, 1, 2), date(2024, 1, 3)])  # overlaps

    write_partition(df1, "AAPL", 2024, 1)
    write_partition(df2, "AAPL", 2024, 1)

    path = partition_path("AAPL", 2024, 1)
    result = pd.read_parquet(path)
    assert len(result) == 2   # no duplicate for Jan 2
