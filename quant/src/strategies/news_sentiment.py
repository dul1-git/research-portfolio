"""
strategies/news_sentiment.py
=============================

I wanted to see if news headlines could predict short-term price moves on JKH.
The setup: pull recent headlines from Google News RSS, score them with VADER
(rule-based NLP, no API costs, no model), and use the sentiment signal to
trade the next day's return.

VADER is genuinely good at this kind of surface-level financial news scoring —
it catches words like "profit", "loss", "crisis", "record" without needing
fine-tuning. The limitation is it has no understanding of context or sarcasm,
so it can misread things like "record losses" as positive because it sees "record".

Results on JKH: strategy generates some gross alpha on the backtest window,
but the 2.24% round-trip CSE fee eats it entirely. Net result is -71%. Same
story as everything else on this exchange at daily frequency.

Signal logic:
- VADER compound score >= 0.05 → long (+1)
- VADER compound score <= -0.05 → short (-1)
- Otherwise → flat (0)
- Multiple headlines on the same day are averaged, then snapped to ternary
- Signal from day T drives a position entered at day T close, exited at T+1 close
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.backtest_engine import BacktestResult, compute_metrics

CSE_FEE_RATE: float = 0.0112   # 1.12% per leg → 2.24% round trip
US_FEE_RATE:  float = 0.0      # commission-free US brokers


def _load_headlines(json_path: Path) -> pd.DataFrame:
    """Score each headline with VADER and aggregate to a daily ternary signal."""
    analyzer = SentimentIntensityAnalyzer()

    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    rows = []
    for a in articles:
        try:
            date_str = pd.to_datetime(a["date"]).tz_localize(None).strftime("%Y-%m-%d")
            score    = analyzer.polarity_scores(a["headline"])["compound"]
            signal   = 1 if score >= 0.05 else (-1 if score <= -0.05 else 0)
            rows.append({"Date": date_str, "Signal": signal})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["Date", "Signal"])

    df    = pd.DataFrame(rows)
    daily = df.groupby("Date")["Signal"].mean().reset_index()
    # snap averaged signal back to ternary
    daily["Signal"] = daily["Signal"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return daily


def run(
    price_df: pd.DataFrame,
    label: str,
    news_json: Path,
    fee_per_leg: float = CSE_FEE_RATE,
    start_capital: float = 100_000.0,
) -> BacktestResult:
    """
    Run the news sentiment strategy.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV with either DatetimeIndex or a 'Date' column, needs 'Close'.
    label : str
    news_json : Path
        Path to the JSON file of headlines.
    fee_per_leg : float
        One-way commission rate. Applied twice (buy + sell) per trade.
    start_capital : float
    """
    sentiment = _load_headlines(news_json)

    if isinstance(price_df.index, pd.DatetimeIndex):
        price = price_df.copy().reset_index()
        price = price.rename(columns={price.columns[0]: "Date"})
        price["Date"] = price["Date"].dt.strftime("%Y-%m-%d")
    else:
        price = price_df.copy()
        price["Date"] = price["Date"].astype(str)

    merged = pd.merge(sentiment, price[["Date", "Close"]], on="Date", how="inner")
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values("Date").reset_index(drop=True)

    if merged.empty:
        return BacktestResult(label=label, strategy_name="News Sentiment (VADER)")

    # signal on T → enter at T close, exit at T+1 close → captures next-day return
    merged["next_ret"]     = merged["Close"].pct_change().shift(-1)
    merged["strategy_ret"] = merged["Signal"] * merged["next_ret"]
    merged["fee_cost"]     = merged["Signal"].abs() * fee_per_leg * 2
    merged["net_ret"]      = merged["strategy_ret"] - merged["fee_cost"]
    merged = merged.dropna()

    capital      = start_capital
    equity_vals: list[tuple] = []
    trades:      list[dict]  = []

    for _, row in merged.iterrows():
        if row["Signal"] != 0:
            ret = float(row["net_ret"])
            trades.append({"entry_date": row["Date"], "exit_date": row["Date"],
                           "return": ret, "hold_days": 1, "exit_type": "SIGNAL"})
            capital *= (1 + ret)
        equity_vals.append((row["Date"], capital))

    eq = pd.Series(
        [v for _, v in equity_vals],
        index=pd.DatetimeIndex([d for d, _ in equity_vals]),
    )
    return compute_metrics(eq, trades, merged.set_index("Date")["Close"],
                           start_capital, label, "News Sentiment (VADER)")
