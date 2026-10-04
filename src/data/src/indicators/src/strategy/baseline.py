"""
GQE-01 Baseline Strategy

Rules-based baseline strategy for SOL/USDT spot.

This is a research hypothesis, not a proven profitable strategy.
"""

from __future__ import annotations

import pandas as pd


MIN_SCORE = 75


def calculate_score(row: pd.Series) -> int:
    """
    Calculate the baseline GQE-01 long score.

    Maximum score = 100.
    """

    score = 0

    # Trend: 25 points
    if row["close"] > row["ema200"]:
        score += 15

    if row["ema20"] > row["ema50"]:
        score += 10

    # Momentum: 20 points
    if 50 <= row["rsi14"] <= 70:
        score += 20

    # Volume: 20 points
    if row["volume_ratio"] > 1.0:
        score += 20

    # Volatility: 15 points
    if row["atr14"] > 0:
        atr_percentage = row["atr14"] / row["close"]

        # Avoid extremely compressed volatility.
        if atr_percentage >= 0.002:
            score += 15

    # Entry quality: 20 points
    if row["close"] > row["ema20"]:
        score += 10

    if row["close"] > row["open"]:
        score += 10

    return score


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate baseline long signals.

    Signals are generated only from the current CLOSED candle.
    """

    result = df.copy()

    result["score"] = result.apply(
        calculate_score,
        axis=1,
    )

    result["long_signal"] = (
        (result["close"] > result["ema200"])
        & (result["ema20"] > result["ema50"])
        & (result["rsi14"].between(50, 70))
        & (result["volume_ratio"] > 1.0)
        & (result["score"] >= MIN_SCORE)
    )

    return result
