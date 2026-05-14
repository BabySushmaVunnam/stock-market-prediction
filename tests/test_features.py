"""
Tests for Phase 2 — Feature Engineering & Validation.

We build a synthetic 100-day price series (sine wave) so every
indicator has enough history to be non-NaN, and assert correctness
of each feature independently.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    add_bollinger_bands,
    add_daily_return,
    add_ema,
    add_lag_features,
    add_macd,
    add_moving_averages,
    add_rsi,
    add_target,
    add_volume_features,
    engineer_features,
)
from pipeline.validate import (
    check_close_positive,
    check_expected_columns,
    check_no_nulls,
    check_rsi_range,
)


@pytest.fixture
def sample_df():
    """100 trading days of synthetic OHLCV data."""
    n = 100
    t = np.linspace(0, 4 * np.pi, n)
    close = 150 + 20 * np.sin(t)   # oscillates between 130 and 170

    return pd.DataFrame({
        "ticker": "TEST",
        "date":   pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    })


# ── Moving averages ───────────────────────────────────────────────────────────

def test_moving_averages_columns(sample_df):
    df = add_moving_averages(sample_df.copy())
    for col in ["ma_7", "ma_21", "ma_50"]:
        assert col in df.columns

def test_ma7_shorter_warmup_than_ma50(sample_df):
    df = add_moving_averages(sample_df.copy())
    assert df["ma_7"].notna().sum() > df["ma_50"].notna().sum()


# ── EMA ───────────────────────────────────────────────────────────────────────

def test_ema_columns(sample_df):
    df = add_ema(sample_df.copy())
    assert "ema_12" in df.columns and "ema_26" in df.columns


# ── MACD ──────────────────────────────────────────────────────────────────────

def test_macd_columns(sample_df):
    df = add_macd(sample_df.copy())
    for col in ["macd", "macd_signal", "macd_diff"]:
        assert col in df.columns


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_range(sample_df):
    df = add_rsi(sample_df.copy()).dropna()
    assert (df["rsi_14"] >= 0).all() and (df["rsi_14"] <= 100).all()


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def test_bollinger_upper_gte_lower(sample_df):
    df = add_bollinger_bands(sample_df.copy()).dropna()
    assert (df["bb_upper"] >= df["bb_lower"]).all()


# ── Volume feature ────────────────────────────────────────────────────────────

def test_vol_change_column(sample_df):
    df = add_volume_features(sample_df.copy())
    assert "vol_change" in df.columns


# ── Lag features ─────────────────────────────────────────────────────────────

def test_lag_shift(sample_df):
    df = add_lag_features(sample_df.copy())
    # lag_1 on row 5 should equal close on row 4
    assert df.loc[5, "lag_1"] == pytest.approx(df.loc[4, "close"])


# ── Daily return ─────────────────────────────────────────────────────────────

def test_daily_return_first_row_nan(sample_df):
    df = add_daily_return(sample_df.copy())
    assert pd.isna(df.loc[0, "daily_return"])


# ── Target ────────────────────────────────────────────────────────────────────

def test_target_is_next_day_close(sample_df):
    df = add_target(sample_df.copy())
    assert df.loc[0, "target"] == pytest.approx(df.loc[1, "close"])

def test_target_last_row_nan(sample_df):
    df = add_target(sample_df.copy())
    assert pd.isna(df.iloc[-1]["target"])


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_engineer_features_no_nulls(sample_df):
    df = engineer_features(sample_df.copy())
    assert df.isnull().sum().sum() == 0

def test_engineer_features_has_all_columns(sample_df):
    df = engineer_features(sample_df.copy())
    expected = ["ma_7", "rsi_14", "macd", "bb_upper", "lag_1", "target"]
    for col in expected:
        assert col in df.columns


# ── Validation checks ─────────────────────────────────────────────────────────

def test_validation_no_nulls_passes(sample_df):
    result = check_no_nulls(sample_df.dropna())
    assert result["passed"]

def test_validation_no_nulls_fails_with_nulls(sample_df):
    df = sample_df.copy()
    df.loc[0, "close"] = None
    result = check_no_nulls(df)
    assert not result["passed"]

def test_validation_rsi_range(sample_df):
    df = add_rsi(sample_df.copy()).dropna()
    result = check_rsi_range(df)
    assert result["passed"]

def test_validation_close_positive(sample_df):
    result = check_close_positive(sample_df)
    assert result["passed"]
