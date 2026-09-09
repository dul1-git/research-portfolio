"""
backtest_engine.py
==================

All the shared maths that every strategy in this project needs. I kept
rewriting Sharpe computation and drawdown logic in every file, so I pulled
it into one place.

The most important design decision here: all metrics run on a *daily*
mark-to-market equity series, not on per-trade returns. This matters a lot.

If you compute Sharpe by taking per-trade returns and multiplying by
sqrt(252), you're implicitly assuming the strategy is in a trade every
single trading day — which is false for something like RSI-2 that might
only enter 20 times a year. The resulting Sharpe is wildly inflated. I
made that mistake in v1 and it took comparing the number against SPY's
known Sharpe to notice something was off.

The fix: build a real calendar-day equity series. On days you're in a
position, mark to the current close. On days you're flat, the equity
just sits unchanged. Then pct_change() on that series gives you honest
daily returns that annualise correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


ANNUAL_RISK_FREE_RATE: float = 0.05  # using 5% — adjust if you're in a different rate environment
TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class BacktestResult:
    """Everything I want to know about a strategy run in one place."""

    label: str          # e.g. "JKH (CSE)" or "SPY (NYSE)"
    strategy_name: str

    # trade-level
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0

    # portfolio-level (daily MTM equity curve)
    cagr: float = 0.0
    max_drawdown: float = 0.0   # e.g. -0.15 means -15%
    sharpe: float = 0.0
    sortino: float = 0.0
    bh_cagr: float = 0.0        # buy-and-hold benchmark for the same period

    # raw series — not printed in the table, but useful for MC tests
    equity_curve: pd.Series = field(default_factory=pd.Series, repr=False)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


def compute_metrics(
    equity: pd.Series,
    trades: list[dict],
    close_series: pd.Series,
    start_capital: float = 100_000.0,
    label: str = "",
    strategy_name: str = "",
) -> BacktestResult:
    """
    Compute the full metric suite from a daily equity series.

    Parameters
    ----------
    equity : pd.Series
        Daily MTM portfolio value, DatetimeIndex.
    trades : list[dict]
        Each dict: entry_date, exit_date, return, hold_days, exit_type.
    close_series : pd.Series
        Raw close prices — used only for buy-and-hold CAGR.
    start_capital : float
    label, strategy_name : str
    """
    result = BacktestResult(label=label, strategy_name=strategy_name)

    if not trades:
        return result

    trades_df = pd.DataFrame(trades)
    result.trades_df = trades_df

    rets = trades_df["return"]
    wins   = trades_df[rets > 0]
    losses = trades_df[rets <= 0]

    result.total_trades  = len(trades_df)
    result.win_rate      = len(wins) / len(trades_df)
    result.avg_win_pct   = float(wins["return"].mean() * 100)   if len(wins)   > 0 else 0.0
    result.avg_loss_pct  = float(losses["return"].mean() * 100) if len(losses) > 0 else 0.0

    loss_sum = abs(losses["return"].sum())
    result.profit_factor = float(wins["return"].sum() / loss_sum) if loss_sum > 0 else float("inf")

    # daily equity metrics
    result.equity_curve = equity
    daily_rets = equity.pct_change().dropna()
    rf_daily   = ANNUAL_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess     = daily_rets - rf_daily

    if excess.std() > 1e-8:
        result.sharpe = float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))

    downside = excess[excess < 0]
    if len(downside) > 0 and downside.std() > 1e-8:
        result.sortino = float((excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))

    peak = equity.cummax()
    result.max_drawdown = float(((equity - peak) / peak).min())

    years = len(close_series) / TRADING_DAYS_PER_YEAR
    result.cagr    = float((equity.iloc[-1] / start_capital) ** (1 / years) - 1) if years > 0 else 0.0
    bh_total       = float(close_series.iloc[-1] / close_series.iloc[0])
    result.bh_cagr = float(bh_total ** (1 / years) - 1) if years > 0 else 0.0

    return result


def print_comparison_table(results: list[BacktestResult]) -> None:
    """Print a nicely aligned comparison table for all strategy runs."""
    headers = [
        "Strategy", "Market", "Trades", "Win%", "AvgW%", "AvgL%",
        "PF", "CAGR%", "B&H%", "MDD%", "Sharpe", "Sortino",
    ]
    rows = []
    for r in results:
        rows.append([
            r.strategy_name, r.label, r.total_trades,
            f"{r.win_rate*100:.1f}",
            f"+{r.avg_win_pct:.2f}", f"{r.avg_loss_pct:.2f}",
            f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "∞",
            f"{r.cagr*100:.2f}", f"{r.bh_cagr*100:.2f}", f"{r.max_drawdown*100:.2f}",
            f"{r.sharpe:.2f}", f"{r.sortino:.2f}",
        ])

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)

    print("\n" + fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()
