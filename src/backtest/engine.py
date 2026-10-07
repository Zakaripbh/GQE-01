"""
GQE-01 Backtest Engine

Conservative spot-only backtesting engine.

Rules:
- Initial capital: $50
- Entry on next candle open after a confirmed signal
- Buy slippage applied at entry
- Fees applied on entry and exit
- Stop loss: 1 ATR below entry, minimum 0.35%
- Target: 2R
- Same-candle SL/TP: SL assumed first
- Daily loss limit: 2%
- Pause after 3 consecutive losses
- Existing positions continue to be managed after new-entry locks
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.risk.position_sizing import (
    INITIAL_CAPITAL,
    RISK_PER_TRADE,
    MIN_STOP_DISTANCE,
    REWARD_RISK_RATIO,
    calculate_position_size,
    calculate_stop_and_target,
)


FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
DAILY_LOSS_LIMIT = 0.02
MAX_CONSECUTIVE_LOSSES = 3


def apply_buy_slippage(price: float) -> float:
    """Apply adverse slippage to a long entry."""

    return price * (1 + SLIPPAGE_RATE)


def apply_sell_slippage(price: float) -> float:
    """Apply adverse slippage to a long exit."""

    return price * (1 - SLIPPAGE_RATE)


def calculate_fee(notional: float) -> float:
    """Calculate trading fee."""

    return notional * FEE_RATE


def calculate_equity(
    cash: float,
    position: Position | None,
    mark_price: float,
) -> float:
    """Calculate current marked-to-market equity."""
    if position is None:
        return cash

    return cash + (
        position.quantity * mark_price
    )


def calculate_daily_loss(
    daily_start_equity: float,
    current_equity: float,
) -> float:
    """Return current daily equity loss as a fraction."""
    if daily_start_equity <= 0:
        return 0.0

    loss = (
        daily_start_equity - current_equity
    ) / daily_start_equity

    return max(loss, 0.0)


@dataclass
class Position:
    """Represents one open spot long position."""

    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop_price: float
    target_price: float
    entry_fee: float
    entry_value: float


@dataclass
class Trade:
    """Represents one completed trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    stop_price: float
    target_price: float
    entry_fee: float
    exit_fee: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str

def build_trade(
    position: Position,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
) -> Trade:
    """Convert an open position into a completed trade."""

    exit_value = position.quantity * exit_price
    exit_fee = calculate_fee(exit_value)

    gross_pnl = (
        (exit_price - position.entry_price)
        * position.quantity
    )

    net_pnl = (
        gross_pnl
        - position.entry_fee
        - exit_fee
    )

    capital_committed = position.entry_value + position.entry_fee

    return_pct = (
        net_pnl / capital_committed
    ) * 100

    return Trade(
        entry_time=position.entry_time,
        exit_time=exit_time,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
        stop_price=position.stop_price,
        target_price=position.target_price,
        entry_fee=position.entry_fee,
        exit_fee=exit_fee,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        return_pct=return_pct,
        exit_reason=exit_reason,
    )

def run_backtest(df: pd.DataFrame) -> tuple[list[Trade], pd.DataFrame]:
    """Run the conservative long-only GQE-01 backtest."""

    required_columns = {
        "timestamp",
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

    data = df.sort_values("timestamp").reset_index(drop=True).copy()

    cash = INITIAL_CAPITAL
    position: Position | None = None

    trades: list[Trade] = []
    equity_records: list[dict] = []

    current_day = None
    daily_start_equity = INITIAL_CAPITAL
    consecutive_losses = 0
    daily_locked = False
    consecutive_loss_locked = False

    for i in range(len(data)):
        row = data.iloc[i]

        timestamp = pd.Timestamp(row["timestamp"])
        trading_day = timestamp.date()

        # Reset only the daily loss control at the start of a
        # new UTC calendar day.
        #
        # The consecutive-loss streak intentionally persists
        # across calendar days until a winning trade occurs.
        if current_day != trading_day:
            current_day = trading_day

            if position is not None:
                mark_price = float(row["open"])
                daily_start_equity = (
                    cash
                    + position.quantity * mark_price
                )
            else:
                daily_start_equity = cash

            daily_locked = False

        # Prevent a position from being closed and re-opened
        # on the same candle using the same previous-candle signal.
        trade_closed_this_candle = False

        # ---------------------------------------------------------
        # 1. Manage an existing position.
        # ---------------------------------------------------------
        if position is not None:

            stop_hit = row["low"] <= position.stop_price
            target_hit = row["high"] >= position.target_price

            if stop_hit or target_hit:

                # Conservative assumption:
                # if both are touched during the same candle,
                # stop loss is assumed to occur first.
                if stop_hit:
                    exit_reason = "stop"
                    raw_exit_price = position.stop_price
                else:
                    exit_reason = "target"
                    raw_exit_price = position.target_price

                exit_price = apply_sell_slippage(raw_exit_price)

                trade = build_trade(
                    position=position,
                    exit_time=timestamp,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                )

                cash += (
                    position.quantity * exit_price
                    - trade.exit_fee
                )

                cash += 0.0  # Explicitly retain readability.

                trades.append(trade)

                if trade.net_pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                position = None
                trade_closed_this_candle = True

                # Daily loss lock based on realized equity.
                current_equity = calculate_equity(
                    cash=cash,
                    position=position,
                    mark_price=float(row["close"]),
                )

                daily_loss = calculate_daily_loss(
                    daily_start_equity=daily_start_equity,
                    current_equity=current_equity,
                )

                if daily_loss >= DAILY_LOSS_LIMIT:
                    daily_locked = True

                if (
                    consecutive_losses
                    >= MAX_CONSECUTIVE_LOSSES
                ):
                    consecutive_loss_locked = True

        # ---------------------------------------------------------
        # 2. Entry logic.
        #
        # Signal belongs to the previous CLOSED candle.
        # Entry occurs at the CURRENT candle open.
        # ---------------------------------------------------------
        if (
            position is None
            and not trade_closed_this_candle
            and i > 0
            and not daily_locked
            and not consecutive_loss_locked
        ):

            previous_row = data.iloc[i - 1]

            if bool(previous_row["long_signal"]):

                entry_price = apply_buy_slippage(
                    float(row["open"])
                )

                atr_value = float(previous_row["atr14"])

                if pd.notna(atr_value) and atr_value > 0:

                    stop_price, target_price = (
                        calculate_stop_and_target(
                            entry_price=entry_price,
                            atr_value=atr_value,
                        )
                    )

                    quantity = calculate_position_size(
                        equity=cash,
                        entry_price=entry_price,
                        stop_price=stop_price,
                    )

                    if quantity > 0:

                        # Make the cash cap fee-aware.
                        # The position-sizing function caps the
                        # notional at available cash, but entry
                        # fees also require cash.
                        max_fee_aware_quantity = (
                            cash
                            / (entry_price * (1 + FEE_RATE))
                        )

                        quantity = min(
                            quantity,
                            max_fee_aware_quantity,
                        )

                        entry_value = (
                            quantity * entry_price
                        )

                        entry_fee = calculate_fee(
                            entry_value
                        )

                        total_entry_cost = (
                            entry_value + entry_fee
                        )

                        # Allow a tiny floating-point tolerance
                        # when comparing the calculated cost to cash.
                        cash_tolerance = 1e-10

                        if (
                            quantity > 0
                            and total_entry_cost <= cash + cash_tolerance
                        ):

                            cash -= total_entry_cost
                            cash = max(cash, 0.0)

                            position = Position(
                                entry_time=timestamp,
                                entry_price=entry_price,
                                quantity=quantity,
                                stop_price=stop_price,
                                target_price=target_price,
                                entry_fee=entry_fee,
                                entry_value=entry_value,
                            )

                            # -------------------------------------------------
                            # Manage SL/TP on the same candle as entry.
                            #
                            # The position was entered at this candle's open,
                            # so its remaining high/low range is tradable.
                            # If both SL and TP are touched, assume SL first.
                            # -------------------------------------------------
                            same_candle_stop = (
                                float(row["low"])
                                <= position.stop_price
                            )
                            same_candle_target = (
                                float(row["high"])
                                >= position.target_price
                            )

                            if (
                                same_candle_stop
                                or same_candle_target
                            ):

                                if same_candle_stop:
                                    exit_reason = "stop"
                                    raw_exit_price = position.stop_price
                                else:
                                    exit_reason = "target"
                                    raw_exit_price = position.target_price

                                exit_price = apply_sell_slippage(
                                    raw_exit_price
                                )

                                trade = build_trade(
                                    position=position,
                                    exit_time=timestamp,
                                    exit_price=exit_price,
                                    exit_reason=exit_reason,
                                )

                                cash += (
                                    position.quantity * exit_price
                                    - trade.exit_fee
                                )

                                trades.append(trade)

                                if trade.net_pnl < 0:
                                    consecutive_losses += 1
                                else:
                                    consecutive_losses = 0

                                position = None
                                trade_closed_this_candle = True

                                current_equity = calculate_equity(
                                    cash=cash,
                                    position=position,
                                    mark_price=float(row["close"]),
                                )

                                daily_loss = calculate_daily_loss(
                                    daily_start_equity=daily_start_equity,
                                    current_equity=current_equity,
                                )

                                if daily_loss >= DAILY_LOSS_LIMIT:
                                    daily_locked = True

                                if (
                                    consecutive_losses
                                    >= MAX_CONSECUTIVE_LOSSES
                                ):
                                    consecutive_loss_locked = True

        # ---------------------------------------------------------
        # 3. Mark-to-market equity.
        # ---------------------------------------------------------
        if position is not None:
            mark_value = (
                position.quantity
                * float(row["close"])
            )
            equity = cash + mark_value
        else:
            equity = cash

        equity_records.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": cash,
                "in_position": position is not None,
                "daily_locked": daily_locked,
                "consecutive_loss_locked": (
                    consecutive_loss_locked
                ),
            }
        )

    # -------------------------------------------------------------
    # 4. End-of-data handling.
    #
    # Do NOT silently leave an open position unaccounted for.
    # Mark it to the final close and record an explicit exit.
    # -------------------------------------------------------------
    if position is not None:

        final_row = data.iloc[-1]

        final_timestamp = pd.Timestamp(
            final_row["timestamp"]
        )

        raw_exit_price = float(
            final_row["close"]
        )

        exit_price = apply_sell_slippage(
            raw_exit_price
        )

        trade = build_trade(
            position=position,
            exit_time=final_timestamp,
            exit_price=exit_price,
            exit_reason="end_of_data",
        )

        cash += (
            position.quantity * exit_price
            - trade.exit_fee
        )

        trades.append(trade)

        position = None

        # Update final equity record.
        if equity_records:
            equity_records[-1]["equity"] = cash
            equity_records[-1]["cash"] = cash
            equity_records[-1]["in_position"] = False

    equity_df = pd.DataFrame(equity_records)

    return trades, equity_df
