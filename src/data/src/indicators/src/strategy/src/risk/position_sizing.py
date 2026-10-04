"""
GQE-01 Risk and Position Sizing Engine

Controls position size based on account equity and
maximum permitted loss.

This module does not place orders.
"""

from __future__ import annotations


INITIAL_CAPITAL = 50.00

RISK_PER_TRADE = 0.01

MIN_STOP_DISTANCE = 0.0035

REWARD_RISK_RATIO = 2.0


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """
    Calculate maximum position size in base asset units.

    Risk is limited to RISK_PER_TRADE of account equity.
    """

    if equity <= 0:
        raise ValueError("Equity must be greater than zero.")

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero.")

    if stop_price <= 0:
        raise ValueError("Stop price must be greater than zero.")

    if stop_price >= entry_price:
        raise ValueError(
            "For a long trade, stop price must be below entry price."
        )

    stop_distance = (
        entry_price - stop_price
    ) / entry_price

    if stop_distance < MIN_STOP_DISTANCE:
        stop_distance = MIN_STOP_DISTANCE

    maximum_risk = equity * RISK_PER_TRADE

    risk_per_unit = entry_price * stop_distance

    position_size = maximum_risk / risk_per_unit

    # Never allocate more than available equity.
    maximum_affordable_size = equity / entry_price

    return min(
        position_size,
        maximum_affordable_size,
    )


def calculate_stop_and_target(
    entry_price: float,
    atr_value: float,
) -> tuple[float, float]:
    """
    Calculate stop-loss and take-profit for a long trade.

    Stop = 1 ATR below entry.
    Target = 2R above entry.
    """

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero.")

    if atr_value <= 0:
        raise ValueError("ATR must be greater than zero.")

    stop_price = entry_price - atr_value

    minimum_stop = entry_price * (
        1 - MIN_STOP_DISTANCE
    )

    stop_price = min(
        stop_price,
        minimum_stop,
    )

    risk_per_unit = entry_price - stop_price

    target_price = (
        entry_price
        + risk_per_unit * REWARD_RISK_RATIO
    )

    return stop_price, target_price
