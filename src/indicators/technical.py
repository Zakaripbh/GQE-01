"""
GQE-01 Technical Indicators

Gap-aware technical indicator calculations.

Indicators:
- EMA20
- EMA50
- EMA200
- RSI14
- ATR14
- Volume ratio

Calculations restart after missing 5-minute candles so that
data from before a source gap does not contaminate the new segment.
"""

from __future__ import annotations

import pandas as pd


EXPECTED_INTERVAL = pd.Timedelta(minutes=5)


def _segment_data(df: pd.DataFrame) -> pd.Series:
    """Create contiguous segment IDs based on timestamp gaps."""

    gaps = df["timestamp"].diff() > EXPECTED_INTERVAL
    return gaps.cumsum()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate exponential moving average."""

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = average_gain / average_loss

    return 100 - (100 / (1 + rs))


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range."""

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def volume_ratio(
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Calculate current volume divided by rolling average volume."""

    average_volume = volume.rolling(
        period,
        min_periods=period,
    ).mean()

    return volume / average_volume


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all GQE-01 indicators to an OHLCV dataframe."""

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    result["segment"] = _segment_data(result)

    result["ema20"] = (
        result.groupby("segment", group_keys=False)["close"]
        .apply(lambda s: ema(s, 20))
        .reset_index(drop=True)
    )

    result["ema50"] = (
        result.groupby("segment", group_keys=False)["close"]
        .apply(lambda s: ema(s, 50))
        .reset_index(drop=True)
    )

    result["ema200"] = (
        result.groupby("segment", group_keys=False)["close"]
        .apply(lambda s: ema(s, 200))
        .reset_index(drop=True)
    )

    result["rsi14"] = (
        result.groupby("segment", group_keys=False)["close"]
        .apply(lambda s: rsi(s, 14))
        .reset_index(drop=True)
    )

    result["atr14"] = (
        result.groupby("segment", group_keys=False)
        .apply(
            lambda group: atr(
                group["high"],
                group["low"],
                group["close"],
                14,
            )
        )
        .reset_index(drop=True)
    )

    result["volume_ratio"] = (
        result.groupby("segment", group_keys=False)["volume"]
        .apply(lambda s: volume_ratio(s, 20))
        .reset_index(drop=True)
    )

    result.drop(columns=["segment"], inplace=True)

    return result
