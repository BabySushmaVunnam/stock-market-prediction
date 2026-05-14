"""
Phase 2c — Data Quality Validation

Runs checks on the processed feature store before any model training.
Think of this as a lightweight Great Expectations — explicit assertions
that catch data drift or pipeline bugs early.

Each check returns a dict: {"name": str, "passed": bool, "detail": str}
The run() function collects all results and raises if any check failed.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

PROCESSED_DATA_DIR = Path("data/processed")

EXPECTED_FEATURE_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume",
    "ma_7", "ma_21", "ma_50",
    "ema_12", "ema_26",
    "macd", "macd_signal", "macd_diff",
    "rsi_14",
    "bb_upper", "bb_lower", "bb_mid", "bb_pband", "bb_wband",
    "vol_change",
    "lag_1", "lag_5", "lag_10",
    "daily_return",
    "target",
]


def check_no_nulls(df: pd.DataFrame) -> dict:
    null_counts = df.isnull().sum()
    bad = null_counts[null_counts > 0]
    passed = bad.empty
    return {
        "name": "no_nulls",
        "passed": passed,
        "detail": "OK" if passed else f"Nulls found in: {bad.to_dict()}",
    }


def check_expected_columns(df: pd.DataFrame) -> dict:
    missing = set(EXPECTED_FEATURE_COLUMNS) - set(df.columns)
    passed = not missing
    return {
        "name": "expected_columns",
        "passed": passed,
        "detail": "OK" if passed else f"Missing columns: {missing}",
    }


def check_rsi_range(df: pd.DataFrame) -> dict:
    out_of_range = df[(df["rsi_14"] < 0) | (df["rsi_14"] > 100)]
    passed = out_of_range.empty
    return {
        "name": "rsi_range_0_100",
        "passed": passed,
        "detail": "OK" if passed else f"{len(out_of_range)} rows have RSI outside [0, 100]",
    }


def check_ma_ordering(df: pd.DataFrame) -> dict:
    """In a strong uptrend ma_7 > ma_21 > ma_50 — but this isn't always true.
    We only check that the columns are numeric and non-negative."""
    bad = df[(df["ma_7"] <= 0) | (df["ma_21"] <= 0) | (df["ma_50"] <= 0)]
    passed = bad.empty
    return {
        "name": "ma_positive",
        "passed": passed,
        "detail": "OK" if passed else f"{len(bad)} rows have non-positive MA values",
    }


def check_close_positive(df: pd.DataFrame) -> dict:
    bad = df[df["close"] <= 0]
    passed = bad.empty
    return {
        "name": "close_positive",
        "passed": passed,
        "detail": "OK" if passed else f"{len(bad)} rows have close <= 0",
    }


def check_min_rows(df: pd.DataFrame, min_rows: int = 30) -> dict:
    passed = len(df) >= min_rows
    return {
        "name": f"min_{min_rows}_rows",
        "passed": passed,
        "detail": f"{len(df)} rows" if passed else f"Only {len(df)} rows — need at least {min_rows}",
    }


ALL_CHECKS = [
    check_no_nulls,
    check_expected_columns,
    check_rsi_range,
    check_ma_ordering,
    check_close_positive,
    check_min_rows,
]


def validate_ticker(ticker: str) -> list[dict]:
    path = PROCESSED_DATA_DIR / f"{ticker}.parquet"
    if not path.exists():
        return [{"name": "file_exists", "passed": False, "detail": f"{path} not found"}]

    df = pd.read_parquet(path)
    results = [check(df) for check in ALL_CHECKS]

    passed = sum(r["passed"] for r in results)
    logger.info(f"{ticker}: {passed}/{len(results)} checks passed")
    for r in results:
        level = "info" if r["passed"] else "error"
        getattr(logger, level)(f"  [{'+' if r['passed'] else 'X'}] {r['name']}: {r['detail']}")

    return results


def run(tickers: list[str]) -> None:
    all_passed = True
    for ticker in tickers:
        results = validate_ticker(ticker)
        if any(not r["passed"] for r in results):
            all_passed = False

    if not all_passed:
        raise RuntimeError("Data quality validation failed — check logs above")
    logger.info("All validation checks passed")


def available_tickers() -> list[str]:
    return [p.stem for p in PROCESSED_DATA_DIR.glob("*.parquet")]


if __name__ == "__main__":
    tickers = available_tickers()
    if not tickers:
        logger.error("No processed Parquet files found in data/processed/")
    else:
        run(tickers)
