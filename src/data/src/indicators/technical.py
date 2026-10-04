"""
GQE-01 Technical Indicators

Pure indicator calculations used by the research engine.

These functions do not place orders and do not make trading decisions.
"""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = average_gain / average_loss

    return 100 - (100 / (1 + rs))


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
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
        adjust=False
    ).mean()


def volume_ratio(
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """Compare current volume with its moving average."""

    average_volume = volume.rolling(period).mean()

    return volume / average_volume


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all GQE-01 baseline indicators to an OHLCV DataFrame.
    """

    result = df.copy()

    result["ema20"] = ema(result["close"], 20)
    result["ema50"] = ema(result["close"], 50)
    result["ema200"] = ema(result["close"], 200)

    result["rsi14"] = rsi(result["close"], 14)

    result["atr14"] = atr(
        result["high"],
        result["low"],
        result["close"],
        14,
    )

    result["volume_ratio"] = volume_ratio(
        result["volume"],
        20,
    )

    return result
