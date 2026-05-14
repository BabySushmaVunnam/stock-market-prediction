"""
Tests for Phase 4 — Model Training & Evaluation.

Uses synthetic data so tests run fast without hitting disk or the network.
"""

import json
import numpy as np
import pandas as pd
import pytest

from models.evaluate import compute_metrics
from models.train import get_feature_cols, time_series_split, NON_FEATURE_COLS


@pytest.fixture
def sample_features():
    """Minimal feature DataFrame that mirrors the processed Parquet schema."""
    n = 100
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        "ticker":       "TEST",
        "date":         pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":         close * 0.99,
        "high":         close * 1.01,
        "low":          close * 0.98,
        "close":        close,
        "volume":       np.random.randint(1_000_000, 5_000_000, n),
        "ma_7":         close,
        "ma_21":        close,
        "ma_50":        close,
        "ema_12":       close,
        "ema_26":       close,
        "macd":         np.random.randn(n),
        "macd_signal":  np.random.randn(n),
        "macd_diff":    np.random.randn(n),
        "rsi_14":       np.random.uniform(30, 70, n),
        "bb_upper":     close * 1.02,
        "bb_lower":     close * 0.98,
        "bb_mid":       close,
        "bb_pband":     np.random.uniform(0, 1, n),
        "bb_wband":     np.random.uniform(0, 0.1, n),
        "vol_change":   np.random.randn(n),
        "lag_1":        np.roll(close, 1),
        "lag_5":        np.roll(close, 5),
        "lag_10":       np.roll(close, 10),
        "daily_return": np.random.randn(n) * 0.01,
        "target":       np.roll(close, -1),
    })
    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def test_time_series_split_sizes(sample_features):
    train, test = time_series_split(sample_features, test_size=0.2)
    assert len(train) == 80
    assert len(test) == 20

def test_time_series_split_no_overlap(sample_features):
    train, test = time_series_split(sample_features, test_size=0.2)
    assert train["date"].max() < test["date"].min()

def test_time_series_split_chronological(sample_features):
    train, test = time_series_split(sample_features, test_size=0.2)
    assert train["date"].is_monotonic_increasing
    assert test["date"].is_monotonic_increasing


# ── Feature selection ─────────────────────────────────────────────────────────

def test_get_feature_cols_excludes_non_features(sample_features):
    cols = get_feature_cols(sample_features)
    for excluded in NON_FEATURE_COLS:
        assert excluded not in cols

def test_get_feature_cols_includes_indicators(sample_features):
    cols = get_feature_cols(sample_features)
    for expected in ["ma_7", "rsi_14", "macd", "bb_upper", "lag_1"]:
        assert expected in cols


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_perfect_prediction_zero_error():
    y = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"]  == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0

def test_directional_accuracy_range():
    y_true = np.array([100.0, 102.0, 101.0, 105.0, 103.0])
    y_pred = np.array([100.5, 101.5, 101.5, 104.5, 102.5])
    metrics = compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["directional_accuracy"] <= 100.0

def test_metrics_keys():
    y = np.array([100.0, 101.0, 99.0, 102.0])
    metrics = compute_metrics(y, y + 1)
    assert set(metrics.keys()) == {"mae", "rmse", "mape", "directional_accuracy"}

def test_mae_less_than_rmse_for_uniform_error():
    y_true = np.array([100.0, 101.0, 102.0, 103.0])
    y_pred = y_true + 2.0   # constant error of 2
    metrics = compute_metrics(y_true, y_pred)
    # With uniform error, MAE == RMSE
    assert abs(metrics["mae"] - metrics["rmse"]) < 0.01
