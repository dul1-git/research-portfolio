"""
strategies/night_effect.py
==========================

I stumbled on this reading about market microstructure. The basic question is:
does it matter *when* during the day a stock moves? Like, is the same asset
systematically going up overnight and giving it back during market hours?

Turns out for JKH on the CSE, the answer is extreme. Every single rupee of
cumulative gain over 8+ years happened between close and the next morning's open.
Trading during market hours — buying at open, selling at close — would have
destroyed a third of your capital over the same period.

The decomposition is simple: I just separate each day into two non-overlapping
return components and compound them independently to see where the money actually is.

The sad part: the overnight edge is real and the maths confirms it, but you can't
extract it at retail CSE commissions (2.24% round trip). The fee eats the entire
overnight return at daily trading frequency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestResult


def run(df: pd.DataFrame, label: str) -> tuple[BacktestResult, BacktestResult, BacktestResult]:
    """
    Decompose returns into overnight vs intraday components.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with DatetimeIndex, needs at least Open and Close.
    label : str
        Market label, e.g. "JKH (CSE)".

    Returns
    -------
    tuple of (intraday_result, overnight_result, buy_and_hold_result)
    """
    df = df.copy().dropna(subset=["Open", "Close"])
    df["Open"]  = pd.to_numeric(df["Open"],  errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Open", "Close"]).sort_index()

    # intraday: open → close, same session
    df["intraday_ret"]  = (df["Close"] - df["Open"]) / df["Open"]
    # overnight: previous close → today's open
    df["prev_close"]    = df["Close"].shift(1)
    df["overnight_ret"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
    df["daily_ret"]     = df["Close"].pct_change()
    df = df.dropna()

    start_capital = 100_000.0
    intraday_eq   = (1 + df["intraday_ret"]).cumprod()  * start_capital
    overnight_eq  = (1 + df["overnight_ret"]).cumprod() * start_capital
    bh_eq         = (1 + df["daily_ret"]).cumprod()     * start_capital

    def _build_result(eq: pd.Series, name: str, rets: pd.Series) -> BacktestResult:
        years    = len(df) / 252.0
        rf_daily = 0.05 / 252
        excess   = eq.pct_change().dropna() - rf_daily
        dn       = excess[excess < 0]

        r = BacktestResult(label=label, strategy_name=name)
        r.cagr         = float((eq.iloc[-1] / start_capital) ** (1 / years) - 1)
        r.max_drawdown = float(((eq - eq.cummax()) / eq.cummax()).min())
        r.sharpe       = float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0.0
        r.sortino      = float((excess.mean() / dn.std()) * np.sqrt(252)) if len(dn) > 0 and dn.std() > 0 else 0.0
        r.bh_cagr      = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1 / years) - 1)
        r.equity_curve = eq
        r.total_trades = len(df)  # one position per day
        r.win_rate     = float((rets > 0).mean())
        r.avg_win_pct  = float(rets[rets > 0].mean() * 100)  if (rets > 0).any()  else 0.0
        r.avg_loss_pct = float(rets[rets <= 0].mean() * 100) if (rets <= 0).any() else 0.0
        return r

    return (
        _build_result(intraday_eq,  "Night Effect — Intraday",  df["intraday_ret"]),
        _build_result(overnight_eq, "Night Effect — Overnight", df["overnight_ret"]),
        _build_result(bh_eq,        "Buy & Hold",               df["daily_ret"]),
    )
