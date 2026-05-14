"""
Schema definitions for raw OHLCV data.

Pydantic validates every row at the ingestion boundary so bad data
never silently propagates into the feature store.
"""

from datetime import date
from pydantic import BaseModel, field_validator, model_validator


class OHLCVRecord(BaseModel):
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    @field_validator("open", "high", "low", "close")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def high_gte_low(self) -> "OHLCVRecord":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        return self

    @field_validator("volume")
    @classmethod
    def volume_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Volume must be >= 0, got {v}")
        return v


# Column order enforced when writing Parquet
OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]
