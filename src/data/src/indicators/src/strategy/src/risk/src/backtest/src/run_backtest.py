"""
GQE-01 Backtest Runner

Loads historical SOL/USDT data, calculates indicators,
generates strategy signals, and runs the backtest.
"""

from pathlib import Path

import pandas as pd

from src.indicators.technical import add_indicators
from src.strategy.baseline import generate_signals
from src.backtest.engine import run_backtest


DATA_FILE = Path("data/SOLUSDT_5m.csv")
RESULTS_DIR = Path("results")


def main() -> None:

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Market data not found: {DATA_FILE}\n"
            "Run the data downloader first."
        )

    print("Loading market data...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["open_time", "close_time"],
    )

    print(f"Loaded {len(df):,} candles.")

    print("Calculating indicators...")

    df = add_indicators(df)

    print("Generating trading signals...")

    df = generate_signals(df)

    print("Running backtest...")

    trades, equity = run_backtest(
        df,
        initial_capital=50.0,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_file = RESULTS_DIR / "trades.csv"
    equity_file = RESULTS_DIR / "equity_curve.csv"

    trades.to_csv(
        trades_file,
        index=False,
    )

    equity.to_csv(
        equity_file,
        index=False,
    )

    print()
    print("=" * 50)
    print("GQE-01 BACKTEST COMPLETE")
    print("=" * 50)

    print(f"Trades: {len(trades)}")

    if not equity.empty:

        starting_equity = 50.0
        ending_equity = equity["equity"].iloc[-1]

        total_return = (
            (ending_equity / starting_equity) - 1
        ) * 100

        print(
            f"Starting equity: ${starting_equity:.2f}"
        )

        print(
            f"Ending equity:   ${ending_equity:.2f}"
        )

        print(
            f"Total return:    {total_return:.2f}%"
        )

    print()
    print(f"Trades saved to: {trades_file}")
    print(f"Equity saved to: {equity_file}")


if __name__ == "__main__":
    main()
