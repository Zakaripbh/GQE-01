"""
GQE-01 Market Data Module

Downloads and validates 5-minute SOL/USD market data from Coinbase.
The saved dataset is named SOLUSDT_5m.csv for consistency with the
GQE-01 research engine.

No missing candles are fabricated or forward-filled.
"""

from __future__ import annotations

from pathlib import Path

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


SOURCE_URL = "https://api.exchange.coinbase.com/products/SOL-USD/candles"

INTERVAL_SECONDS = 300
LOOKBACK_DAYS = 180
MAX_CANDLES_PER_REQUEST = 300

OUTPUT_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "SOLUSDT_5m.csv"
)


def download_candles(
    start_time: datetime,
    end_time: datetime,
) -> list[list]:
    """Download one Coinbase candle window."""

    params = {
        "granularity": INTERVAL_SECONDS,
        "start": start_time.astimezone(timezone.utc).isoformat(),
        "end": end_time.astimezone(timezone.utc).isoformat(),
    }

    response = requests.get(
        SOURCE_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError("Unexpected Coinbase API response.")

    return data


def download_history(
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Download historical 5-minute SOL/USD candles."""

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    all_candles = []
    current_start = start_time

    while current_start < end_time:
        current_end = min(
            current_start
            + timedelta(seconds=INTERVAL_SECONDS * MAX_CANDLES_PER_REQUEST),
            end_time,
        )

        candles = download_candles(current_start, current_end)
        all_candles.extend(candles)

        current_start = current_end

        time.sleep(0.15)

    if not all_candles:
        raise RuntimeError("No market data was returned.")

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "low",
            "high",
            "open",
            "close",
            "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .dropna()
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def validate_ohlcv(df: pd.DataFrame) -> dict:
    """Validate basic OHLCV data quality."""

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    gaps = (
        df["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(INTERVAL_SECONDS)
    )

    gap_count = int((gaps > 1).sum())

    return {
        "rows": len(df),
        "duplicate_timestamps": int(
            df["timestamp"].duplicated().sum()
        ),
        "missing_values": int(df.isna().sum().sum()),
        "chronological": bool(
            df["timestamp"].is_monotonic_increasing
        ),
        "gaps": gap_count,
        "start": df["timestamp"].iloc[0],
        "end": df["timestamp"].iloc[-1],
    }


def save_data(df: pd.DataFrame) -> None:
    """Save validated data to the GQE-01 data directory."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


if __name__ == "__main__":
    data = download_history()
    validation = validate_ohlcv(data)

    save_data(data)

    print("GQE-01 market data downloaded.")
    print(f"Rows: {validation['rows']:,}")
    print(f"Start: {validation['start']}")
    print(f"End:   {validation['end']}")
    print(f"Gaps:  {validation['gaps']}")
    print(f"Saved: {OUTPUT_FILE}")
