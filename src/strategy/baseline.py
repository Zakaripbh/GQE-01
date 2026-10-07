"""
GQE-01 Baseline Strategy

Transparent rules-based long-only strategy.

Score:
- Trend:       25 points
- Momentum:    20 points
- Volume:      20 points
- Volatility:  15 points
- Entry:       20 points

A long signal requires a minimum score of 75.
"""

from __future__ import annotations

import pandas as pd

MIN_SCORE = 75


def calculate_score(row: pd.Series) -> int:
    score = 0

    # Trend — 25 points
    if row["close"] > row["ema200"]:
        score += 25

    # Momentum — 20 points
    if row["ema20"] > row["ema50"]:
        score += 20

    # Volume — 20 points
    if row["volume_ratio"] > 1.0:
        score += 20

    # Volatility — 15 points
    if (
        pd.notna(row["atr_pct"])
        and pd.notna(row["atr_pct_median20"])
        and row["atr_pct"] > row["atr_pct_median20"]
    ):
        score += 15

    # Entry quality — 20 points
    if pd.notna(row["rsi14"]) and 50 <= row["rsi14"] <= 70:
        score += 20

    return score


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi14",
        "atr14",
        "volume_ratio",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    # ATR as percentage of price
    result["atr_pct"] = (
        result["atr14"] / result["close"]
    ) * 100

    # Relative volatility regime
    result["atr_pct_median20"] = (
        result["atr_pct"]
        .rolling(20, min_periods=20)
        .median()
    )

    result["score"] = result.apply(
        calculate_score,
        axis=1,
    )

    # Explicit long-entry conditions
    result["long_signal"] = (
        (result["score"] >= MIN_SCORE)
        & (result["close"] > result["ema200"])
        & (result["ema20"] > result["ema50"])
        & (result["rsi14"].between(50, 70))
        & (result["volume_ratio"] > 1.0)
        & (
            result["atr_pct"]
            > result["atr_pct_median20"]
        )
    )

    return result
