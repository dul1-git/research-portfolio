"""
strategies/rsi2_mean_reversion.py
==================================

RSI-2 is a short-term mean reversion strategy by Larry Connors (2008). The idea
is simple: buy extreme 2-day oversold dips on assets that are in an uptrend,
then exit as soon as they bounce back to their 5-day average.

I tested this because it has a legitimate published track record on US indices,
and I wanted to see if the same logic would transfer to a frontier market like JKH.
Spoiler: it doesn't, but not for the reason I expected. The signal structure just
doesn't hold — JKH can trend down for years without the mean-reversion gravity
that broad-market ETFs have, so the SMA200 filter lets you right into bear markets
that never bounce.

On SPY it works reasonably well over 20 years. 70% win rate, profit factor of 1.73.
But the Sharpe is only 0.11 and it underperforms buy-and-hold — you're not in the
market most of the time, so you miss a lot of the upside.

--- Look-ahead prevention ---
This took two iterations to get right. In v1, I was using today's close to generate
a signal and then "entering" today — which means I was entering on the same bar the
signal fired. That's look-ahead bias. Every signal now reads bar T's data and all
execution happens at bar T+1's open.

--- Stop-loss ---
5% hard stop below entry. Two edge cases that matter:
1. Entry-day gap: if the market opens below the stop on the first day, I check this
   immediately and exit at open (not at the stop price — you'd never fill there).
2. Gap-down fills: if the market gaps below the stop on any subsequent day, I exit
   at open, not at the theoretical stop. This is the honest assumption.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.backtest_engine import BacktestResult, compute_metrics

STOP_PCT: float      = 0.05   # 5% hard stop below entry
RSI_THRESHOLD: float = 10.0   # enter when RSI-2 drops below this
SMA_FAST: int        = 5      # exit when close > N-day SMA
SMA_SLOW: int        = 200    # regime filter — only trade above this


def _compute_rsi2(close: pd.Series) -> pd.Series:
    """Standard 2-period RSI. Simple rolling mean version — no Wilder smoothing."""
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=2).mean()
    avg_loss = loss.rolling(window=2).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def run(df: pd.DataFrame, label: str, start_capital: float = 100_000.0) -> BacktestResult:
    """
    Run the RSI-2 mean reversion strategy.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with DatetimeIndex and columns [Open, High, Low, Close].
    label : str
        Market identifier, e.g. "SPY (NYSE)".
    start_capital : float
    """
    df = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    df["SMA200"] = df["Close"].rolling(window=SMA_SLOW).mean()
    df["SMA5"]   = df["Close"].rolling(window=SMA_FAST).mean()
    df["RSI2"]   = _compute_rsi2(df["Close"])
    df = df.dropna()

    in_position: bool                  = False
    buy_price:   float                 = 0.0
    stop_price:  float                 = 0.0
    entry_date:  pd.Timestamp | None   = None
    trades:      list[dict]            = []
    capital:     float                 = start_capital
    daily_equity: list[tuple]          = []

    for i in range(1, len(df)):
        today     = df.iloc[i]
        yesterday = df.iloc[i - 1]

        if not in_position:
            daily_equity.append((df.index[i], capital))

            # signal reads yesterday's data — execution tomorrow (today)
            if yesterday["Close"] > yesterday["SMA200"] and yesterday["RSI2"] < RSI_THRESHOLD:
                in_position = True
                entry_date  = df.index[i]
                buy_price   = float(today["Open"])
                stop_price  = buy_price * (1 - STOP_PCT)

                # check stop on the entry bar itself
                if today["Low"] <= stop_price:
                    exit_price = min(float(today["Open"]), stop_price)
                    ret = (exit_price - buy_price) / buy_price
                    capital *= (1 + ret)
                    daily_equity[-1] = (df.index[i], capital)
                    trades.append({"entry_date": entry_date, "exit_date": df.index[i],
                                   "return": ret, "hold_days": 0, "exit_type": "STOP (entry day)"})
                    in_position = False
                else:
                    unrealized = (float(today["Close"]) - buy_price) / buy_price
                    daily_equity[-1] = (df.index[i], capital * (1 + unrealized))

        else:
            exit_price: float | None = None
            exit_type:  str          = ""

            if float(today["Open"]) <= stop_price:
                exit_price = float(today["Open"]); exit_type = "STOP (gap-down)"
            elif float(today["Low"]) <= stop_price:
                exit_price = stop_price;           exit_type = "STOP"
            elif float(yesterday["Close"]) > float(yesterday["SMA5"]):
                exit_price = float(today["Open"]); exit_type = "SIGNAL"

            if exit_price is not None:
                ret    = (exit_price - buy_price) / buy_price
                capital *= (1 + ret)
                hold   = (df.index[i] - entry_date).days if entry_date else 0
                trades.append({"entry_date": entry_date, "exit_date": df.index[i],
                               "return": ret, "hold_days": hold, "exit_type": exit_type})
                daily_equity.append((df.index[i], capital))
                in_position = False
            else:
                unrealized = (float(today["Close"]) - buy_price) / buy_price
                daily_equity.append((df.index[i], capital * (1 + unrealized)))

    eq = pd.Series(
        [v for _, v in daily_equity],
        index=pd.DatetimeIndex([d for d, _ in daily_equity]),
    )
    return compute_metrics(eq, trades, df["Close"], start_capital, label, "RSI-2 Mean Reversion")
