"""
GQE-01 Data Module

Downloads and validates SOL/USDT 5-minute spot market data
for research and paper-trading backtests.

No live orders are placed by this module.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


SYMBOL = "SOLUSDT"
INTERVAL = "5m"
LOOKBACK_DAYS = 180

BINANCE_URL = "https://api.binance.com/api/v3/klines"

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / f"{SYMBOL}_{INTERVAL}.csv"

INTERVAL_MS = 5 * 60 * 1000
LIMIT = 1000


def fetch_klines(start_time: int, end_time: int) -> list[list]:
    """Fetch one batch of candles from Binance public market data."""

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_time,
        "endTime": end_time,
        "limit": LIMIT,
    }

    response = requests.get(BINANCE_URL, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def download_data() -> pd.DataFrame:
    """Download approximately LOOKBACK_DAYS of 5-minute candles."""

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

    start_time = int(start_dt.timestamp() * 1000)
    end_time = int(end_dt.timestamp() * 1000)

    all_rows: list[list] = []

    print(f"Downloading {SYMBOL} {INTERVAL} data...")
    print(f"Start: {start_dt.isoformat()}")
    print(f"End:   {end_dt.isoformat()}")

    while start_time < end_time:
        rows = fetch_klines(start_time, end_time)

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        next_start_time = last_open_time + INTERVAL_MS

        if next_start_time <= start_time:
            break

        start_time = next_start_time

        print(f"Downloaded {len(all_rows):,} candles")

        time.sleep(0.15)

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(all_rows, columns=columns)

    if df.empty:
        raise ValueError("No market data was downloaded.")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize downloaded market data."""

    df = df.copy()

    df["open_time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"], unit="ms", utc=True
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("open_time")
    df = df.drop_duplicates(subset="open_time")
    df = df.reset_index(drop=True)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run integrity checks before saving market data."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not df["open_time"].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronological.")

    if df["open_time"].duplicated().any():
        raise ValueError("Duplicate candle timestamps detected.")

    if (df["open"] <= 0).any():
        raise ValueError("Invalid open prices detected.")

    if (df["high"] <= 0).any():
        raise ValueError("Invalid high prices detected.")

    if (df["low"] <= 0).any():
        raise ValueError("Invalid low prices detected.")

    if (df["close"] <= 0).any():
        raise ValueError("Invalid close prices detected.")

    if (df["volume"] < 0).any():
        raise ValueError("Negative volume detected.")

    invalid_ohlc = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC relationships detected.")

    gaps = df["open_time"].diff().dropna()

    expected_gap = pd.Timedelta(minutes=5)

    irregular_gaps = gaps[gaps != expected_gap]

    if not irregular_gaps.empty:
        print(
            f"WARNING: {len(irregular_gaps)} irregular "
            "5-minute candle gaps detected."
        )

    print("Data validation completed.")


def save_data(df: pd.DataFrame) -> None:
    """Save validated market data to the project data directory."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"Total candles: {len(df):,}")


def main() -> None:
    """Main data pipeline."""

    df = download_data()
    df = clean_data(df)

    validate_data(df)
    save_data(df)


if __name__ == "__main__":
    main()
