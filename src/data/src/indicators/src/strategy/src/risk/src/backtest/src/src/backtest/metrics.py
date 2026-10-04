"""
GQE-01 Backtest Performance Metrics

Calculates objective performance statistics from
backtest trades and equity history.
"""

from __future__ import annotations

import pandas as pd


def calculate_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float = 50.0,
) -> dict:
    """Calculate core backtest performance metrics."""

    if equity.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return_percent": 0.0,
            "max_drawdown_percent": 0.0,
            "net_profit": 0.0,
            "average_trade": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "max_consecutive_losses": 0,
        }

    ending_equity = float(
        equity["equity"].iloc[-1]
    )

    net_profit = (
        ending_equity - initial_capital
    )

    total_return_percent = (
        net_profit / initial_capital
    ) * 100

    total_trades = len(trades)

    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return_percent": total_return_percent,
            "max_drawdown_percent": 0.0,
            "net_profit": net_profit,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "max_consecutive_losses": 0,
        }

    winners = trades[
        trades["pnl"] > 0
    ]

    losers = trades[
        trades["pnl"] < 0
    ]

    win_rate = (
        len(winners) / total_trades
    ) * 100

    gross_profit = winners["pnl"].sum()
    gross_loss = abs(losers["pnl"].sum())

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = float("inf")

    equity_series = equity["equity"]

    running_peak = equity_series.cummax()

    drawdown = (
        (equity_series - running_peak)
        / running_peak
    ) * 100

    max_drawdown_percent = float(
        drawdown.min()
    )

    largest_win = (
        float(winners["pnl"].max())
        if not winners.empty
        else 0.0
    )

    largest_loss = (
        float(losers["pnl"].min())
        if not losers.empty
        else 0.0
    )

    average_winner = (
        float(winners["pnl"].mean())
        if not winners.empty
        else 0.0
    )

    average_loser = (
        float(losers["pnl"].mean())
        if not losers.empty
        else 0.0
    )

    max_consecutive_losses = 0
    current_losses = 0

    for pnl in trades["pnl"]:

        if pnl < 0:
            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses,
            )

        else:
            current_losses = 0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return_percent": total_return_percent,
        "max_drawdown_percent": max_drawdown_percent,
        "net_profit": net_profit,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "average_winner": average_winner,
        "average_loser": average_loser,
        "max_consecutive_losses": max_consecutive_losses,
    }


def print_metrics(metrics: dict) -> None:
    """Print metrics in a readable format."""

    print()
    print("=" * 50)
    print("GQE-01 PERFORMANCE REPORT")
    print("=" * 50)

    print(
        f"Total trades:          {metrics['total_trades']}"
    )

    print(
        f"Win rate:              {metrics['win_rate']:.2f}%"
    )

    print(
        f"Profit factor:         {metrics['profit_factor']:.2f}"
    )

    print(
        f"Total return:          {metrics['total_return_percent']:.2f}%"
    )

    print(
        f"Net profit:            ${metrics['net_profit']:.4f}"
    )

    print(
        f"Maximum drawdown:      {metrics['max_drawdown_percent']:.2f}%"
    )

    print(
        f"Largest win:           ${metrics['largest_win']:.4f}"
    )

    print(
        f"Largest loss:          ${metrics['largest_loss']:.4f}"
    )

    print(
        f"Average winner:        ${metrics['average_winner']:.4f}"
    )

    print(
        f"Average loser:         ${metrics['average_loser']:.4f}"
    )

    print(
        f"Max consecutive loss:  "
        f"{metrics['max_consecutive_losses']}"
    )

    print("=" * 50)
