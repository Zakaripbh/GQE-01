"""
GQE-01 Backtesting Engine

Tests the baseline strategy against historical OHLCV data.

Important:
- Signals use closed candles.
- Entry occurs on the next candle.
- Fees and slippage are modeled.
- If SL and TP occur in the same candle,
  SL is assumed to occur first (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.risk.position_sizing import (
    calculate_position_size,
    calculate_stop_and_target,
)


FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005


@dataclass
class Trade:
    entry_time: object
    exit_time: object

    entry_price: float
    exit_price: float

    quantity: float

    stop_price: float
    target_price: float

    pnl: float

    pnl_percent: float

    exit_reason: str


def apply_entry_slippage(price: float) -> float:
    """Apply conservative buy-side slippage."""

    return price * (1 + SLIPPAGE_RATE)


def apply_exit_slippage(price: float) -> float:
    """Apply conservative sell-side slippage."""

    return price * (1 - SLIPPAGE_RATE)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the baseline strategy backtest.

    Returns:
        trades_df
        equity_df
    """

    required_columns = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "atr14",
        "long_signal",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    equity = initial_capital

    trades: list[Trade] = []

    equity_history = []

    in_position = False

    entry_time = None
    entry_price = 0.0
    quantity = 0.0
    stop_price = 0.0
    target_price = 0.0

    for i in range(1, len(df)):
        previous = df.iloc[i - 1]
        current = df.iloc[i]

        # -------------------------------------------------
        # Manage existing position
        # -------------------------------------------------

        if in_position:

            exit_price = None
            exit_reason = None

            # Conservative assumption:
            # If both SL and TP are touched in the same
            # candle, assume SL happened first.

            if current["low"] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop_loss"

            elif current["high"] >= target_price:
                exit_price = target_price
                exit_reason = "take_profit"

            if exit_price is not None:

                execution_price = apply_exit_slippage(
                    exit_price
                )

                gross_exit_value = (
                    quantity * execution_price
                )

                exit_fee = (
                    gross_exit_value * FEE_RATE
                )

                entry_value = (
                    quantity * entry_price
                )

                entry_fee = (
                    entry_value * FEE_RATE
                )

                pnl = (
                    gross_exit_value
                    - exit_fee
                    - entry_value
                    - entry_fee
                )

                equity += pnl

                pnl_percent = (
                    pnl / entry_value
                ) * 100

                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=current["open_time"],
                        entry_price=entry_price,
                        exit_price=execution_price,
                        quantity=quantity,
                        stop_price=stop_price,
                        target_price=target_price,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        exit_reason=exit_reason,
                    )
                )

                in_position = False

        # -------------------------------------------------
        # Look for new entry
        # -------------------------------------------------

        if not in_position and previous["long_signal"]:

            # Entry happens on the current candle,
            # after the previous candle generated the signal.

            raw_entry_price = current["open"]

            execution_entry_price = apply_entry_slippage(
                raw_entry_price
            )

            stop, target = calculate_stop_and_target(
                execution_entry_price,
                previous["atr14"],
            )

            quantity = calculate_position_size(
                equity,
                execution_entry_price,
                stop,
            )

            if quantity <= 0:
                continue

            entry_time = current["open_time"]
            entry_price = execution_entry_price
            stop_price = stop
            target_price = target

            in_position = True

        equity_history.append(
            {
                "timestamp": current["open_time"],
                "equity": equity,
            }
        )

    trades_df = pd.DataFrame(
        [trade.__dict__ for trade in trades]
    )

    equity_df = pd.DataFrame(equity_history)

    return trades_df, equity_df
