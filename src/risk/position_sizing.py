"""
GQE-01 Risk Engine

Spot-only position sizing and trade-level risk controls.

Rules:
- Initial paper capital: $50
- Risk per trade: 1% of current equity
- Minimum stop distance: 0.35%
- Reward/risk ratio: 1:2
- No leverage
"""

from __future__ import annotations

INITIAL_CAPITAL = 50.00
RISK_PER_TRADE = 0.01
MIN_STOP_DISTANCE = 0.0035
REWARD_RISK_RATIO = 2.0


def calculate_stop_and_target(
    entry_price: float,
    atr_value: float,
) -> tuple[float, float]:
    """Calculate an ATR-based stop and 2R target."""

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero.")

    if atr_value <= 0:
        raise ValueError("ATR must be greater than zero.")

    # Initial stop: 1 ATR below entry
    stop_price = entry_price - atr_value

    # Enforce minimum stop distance of 0.35%
    minimum_stop = entry_price * (1 - MIN_STOP_DISTANCE)
    stop_price = min(stop_price, minimum_stop)

    risk_per_unit = entry_price - stop_price

    target_price = (
        entry_price
        + risk_per_unit * REWARD_RISK_RATIO
    )

    return stop_price, target_price


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """Calculate position size using 1% risk and available cash."""

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

    stop_distance = max(
        stop_distance,
        MIN_STOP_DISTANCE,
    )

    maximum_risk = equity * RISK_PER_TRADE

    risk_per_unit = (
        entry_price * stop_distance
    )

    position_size = (
        maximum_risk / risk_per_unit
    )

    # Spot-only: cannot spend more than available cash
    maximum_affordable_size = (
        equity / entry_price
    )

    return min(
        position_size,
        maximum_affordable_size,
    )
