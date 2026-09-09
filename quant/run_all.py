"""
run_all.py
==========

Single entry point. Loads both datasets, runs every strategy, prints the
comparison table, then runs Monte Carlo significance tests on all of them.

Usage:
    python run_all.py
    python run_all.py --skip-monte-carlo
    python run_all.py --simulations 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.backtest_engine import BacktestResult, print_comparison_table
from src.strategies import night_effect, rsi2_mean_reversion, news_sentiment
from src.strategies.monte_carlo import (
    run_rsi2_test, run_night_effect_test, run_sentiment_test, print_mc_result
)

DATA_DIR  = Path("data")
JKH_CSV   = DATA_DIR / "jkh_10_years.csv"
NEWS_JSON = DATA_DIR / "jkh_news.json"


def load_jkh() -> pd.DataFrame:
    """Load the CSE frontier-market OHLCV dataset from the static CSV."""
    df = pd.read_csv(JKH_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").set_index("Date").dropna(subset=["Open", "High", "Low", "Close"])


def load_spy() -> pd.DataFrame:
    """Download SPY from Yahoo Finance. Runs fresh every time."""
    print("  Downloading SPY...")
    spy = yf.download("SPY", start="2006-01-01", auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy[["Open", "High", "Low", "Close"]].dropna()


def main(skip_monte_carlo: bool = False, n_simulations: int = 300) -> None:
    print("\n" + "=" * 60)
    print("  Quantitative Trading Research — Full Results")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/4] Loading market data...")
    jkh = load_jkh()
    spy = load_spy()
    print(f"  JKH: {len(jkh)} days  ({jkh.index[0].date()} → {jkh.index[-1].date()})")
    print(f"  SPY: {len(spy)} days  ({spy.index[0].date()} → {spy.index[-1].date()})")

    all_results: list[BacktestResult] = []

    # ── Night Effect ──────────────────────────────────────────────
    print("\n[2/4] Night Effect decomposition...")
    night_results: dict = {}
    for df, lbl in [(jkh, "JKH (CSE)"), (spy, "SPY (NYSE)")]:
        intra, over, _ = night_effect.run(df, lbl)
        all_results.extend([intra, over])
        night_results[lbl] = (intra, over)
        print(f"  {lbl}: Overnight={over.cagr*100:.2f}% CAGR  Intraday={intra.cagr*100:.2f}% CAGR")

    # ── RSI-2 Mean Reversion ──────────────────────────────────────
    print("\n[3/4] RSI-2 Mean Reversion...")
    rsi2_results: dict[str, BacktestResult] = {}
    for df, lbl in [(jkh, "JKH (CSE)"), (spy, "SPY (NYSE)")]:
        r = rsi2_mean_reversion.run(df, lbl)
        rsi2_results[lbl] = r
        all_results.append(r)
        print(f"  {lbl}: {r.total_trades} trades  WinRate={r.win_rate*100:.1f}%  "
              f"Sharpe={r.sharpe:.2f}  CAGR={r.cagr*100:.2f}%")

    # ── News Sentiment (CSE only — news feed is JKH-specific) ─────
    sentiment_merged: pd.DataFrame | None = None
    sentiment_result: BacktestResult | None = None
    if NEWS_JSON.exists():
        print("\n  News Sentiment (JKH/CSE only)...")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from src.strategies.news_sentiment import _load_headlines, CSE_FEE_RATE
        # build the merged df once so the MC test can reuse it
        sentiment_df = _load_headlines(NEWS_JSON)
        price = jkh.reset_index().rename(columns={"Date": "Date"})
        price["Date"] = price["Date"].dt.strftime("%Y-%m-%d")
        merged = pd.merge(sentiment_df, price[["Date", "Close"]], on="Date", how="inner")
        merged["Date"] = pd.to_datetime(merged["Date"])
        merged = merged.sort_values("Date").reset_index(drop=True)
        merged["next_ret"] = merged["Close"].pct_change().shift(-1)
        merged = merged.dropna()
        sentiment_merged = merged

        ns = news_sentiment.run(jkh.reset_index(), "JKH (CSE)", NEWS_JSON, fee_per_leg=CSE_FEE_RATE)
        sentiment_result = ns
        all_results.append(ns)
        print(f"  JKH (CSE): {ns.total_trades} trades  Sharpe={ns.sharpe:.2f}  CAGR={ns.cagr*100:.2f}%")

    # ── Comparison table ──────────────────────────────────────────
    print("\n[4/4] Results")
    print_comparison_table(all_results)

    if skip_monte_carlo:
        print("  [Monte Carlo skipped — remove --skip-monte-carlo to run]\n")
        return

    # ── Monte Carlo significance tests — all strategies ───────────
    print("\n" + "─" * 60)
    print("  Monte Carlo Permutation Tests")
    print("─" * 60)

    # RSI-2 on both markets
    for lbl, r in rsi2_results.items():
        mc = run_rsi2_test(
            strategy_fn=lambda df, label: rsi2_mean_reversion.run(df, label),
            real_ohlcv=spy if "NYSE" in lbl else jkh,
            real_sharpe=r.sharpe,
            n_simulations=n_simulations,
            label=lbl,
        )
        print_mc_result(mc)

    # Night Effect on both markets
    for lbl, (intra, over) in night_results.items():
        mc = run_night_effect_test(
            real_df=spy if "NYSE" in lbl else jkh,
            real_overnight_sharpe=over.sharpe,
            real_intraday_sharpe=intra.sharpe,
            n_simulations=n_simulations,
            label=lbl,
        )
        print_mc_result(mc)

    # News Sentiment
    if sentiment_merged is not None and sentiment_result is not None:
        mc = run_sentiment_test(
            merged_df=sentiment_merged,
            real_sharpe=sentiment_result.sharpe,
            n_simulations=n_simulations,
            label="JKH (CSE)",
        )
        print_mc_result(mc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Research — Full Strategy Suite")
    parser.add_argument("--skip-monte-carlo", action="store_true",
                        help="Skip significance tests for a faster run.")
    parser.add_argument("--simulations", type=int, default=300,
                        help="Monte Carlo simulation count (default: 300).")
    args = parser.parse_args()
    main(skip_monte_carlo=args.skip_monte_carlo, n_simulations=args.simulations)
