"""
strategies/monte_carlo.py
==========================

After building the backtests, I wanted to know: how much of these results
could just be noise?

A 70% win rate sounds impressive until you ask whether a strategy with zero
real edge could produce that number by chance on this particular price history.
The answer is: maybe, especially if you looked at a lot of strategies before
landing on RSI-2.

This is what a Monte Carlo permutation test answers. The idea:
1. Take your real price series.
2. Generate hundreds of synthetic paths that have the same statistical properties
   (same volatility, same autocorrelation structure via block bootstrap) but where
   any genuine predictive signal has been destroyed by shuffling.
3. Run your exact strategy on every synthetic path.
4. Build a distribution of what the strategy achieves on pure noise.
5. See where your real result sits in that distribution.

If your real Sharpe is better than 95% of the noise runs → p < 0.05 → there's
real evidence of edge. If it's in the middle of the noise distribution → you
probably just got lucky on this particular path.

Reference: López de Prado, "Advances in Financial Machine Learning", Chapter 11.

--- Three different tests for three different strategies ---

RSI-2: block bootstrap on the full price series. Preserves local autocorrelation
(volatility clusters) while shuffling the overall sequence.

Night Effect: shuffles which day's (open, close) pair belongs to which row.
This tests: could the overnight-vs-intraday return gap arise from a random
arrangement of the same daily sessions?

News Sentiment: shuffles the signal column while keeping prices fixed. Tests:
does the actual news content matter, or would any random sequence of buy/sell
signals do equally well?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable


# ── Block bootstrap helpers ──────────────────────────────────────────────────

def _block_bootstrap_returns(
    returns: np.ndarray,
    block_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Resample a return series in contiguous blocks.

    Block size defaults to sqrt(T), which is the standard choice for
    preserving short-range autocorrelation without carrying over too much
    structure from the original series.
    """
    n = len(returns)
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))
    if rng is None:
        rng = np.random.default_rng()

    synthetic: list[float] = []
    while len(synthetic) < n:
        start = rng.integers(0, n - block_size + 1)
        synthetic.extend(returns[start : start + block_size].tolist())
    return np.array(synthetic[:n])


def _returns_to_ohlcv(returns: np.ndarray, start_price: float = 100.0) -> pd.DataFrame:
    """
    Turn a plain return series into a synthetic OHLCV DataFrame for the RSI-2 runner.

    I set Open = previous Close (no overnight gap simulation — synthetic paths
    don't model gaps, so gap-down stops won't trigger, which slightly flatters
    the synthetic Sharpes vs the real backtest. Worth noting as a limitation).
    """
    prices = start_price * np.cumprod(1 + returns)
    opens  = np.roll(prices, 1); opens[0] = start_price
    noise  = np.abs(np.random.normal(0, 0.001, size=len(prices)))
    highs  = np.maximum(opens, prices) * (1 + noise)
    lows   = np.minimum(opens, prices) * (1 - noise)
    idx    = pd.date_range(start="2006-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": prices}, index=idx)


def _sharpe_from_equity(equity: pd.Series, rf_daily: float = 0.05 / 252) -> float:
    """Annualised Sharpe from a daily equity series."""
    dr = equity.pct_change().dropna() - rf_daily
    return float((dr.mean() / dr.std()) * np.sqrt(252)) if dr.std() > 0 else 0.0


# ── RSI-2 permutation test ───────────────────────────────────────────────────

def run_rsi2_test(
    strategy_fn: Callable[[pd.DataFrame, str], object],
    real_ohlcv: pd.DataFrame,
    real_sharpe: float,
    n_simulations: int = 300,
    block_size: int | None = None,
    seed: int = 42,
    label: str = "",
) -> dict:
    """
    Block-bootstrap permutation test for the RSI-2 strategy.

    Parameters
    ----------
    strategy_fn : callable
        Function (df, label) → BacktestResult — typically rsi2_mean_reversion.run.
    real_ohlcv : pd.DataFrame
        The real market data used in the original backtest.
    real_sharpe : float
        Sharpe ratio from the real backtest.
    n_simulations : int
        Number of synthetic paths to run.
    seed : int
        RNG seed for reproducibility.
    label : str
        Display label.
    """
    rng          = np.random.default_rng(seed)
    real_returns = real_ohlcv["Close"].pct_change().dropna().to_numpy()
    sim_sharpes: list[float] = []

    print(f"\n  Running {n_simulations} Monte Carlo permutations on {label} (RSI-2)...")
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{n_simulations}]")
        syn_rets  = _block_bootstrap_returns(real_returns, block_size=block_size, rng=rng)
        syn_ohlcv = _returns_to_ohlcv(syn_rets)
        try:
            result = strategy_fn(syn_ohlcv, f"syn_{i}")
            sh = _sharpe_from_equity(result.equity_curve) if len(result.equity_curve) > 10 else 0.0
        except Exception:
            sh = 0.0
        sim_sharpes.append(sh)

    return _build_result_dict(label, "RSI-2 Mean Reversion", real_sharpe,
                              np.array(sim_sharpes), n_simulations)


# ── Night Effect permutation test ────────────────────────────────────────────

def run_night_effect_test(
    real_df: pd.DataFrame,
    real_overnight_sharpe: float,
    real_intraday_sharpe: float,
    n_simulations: int = 300,
    seed: int = 42,
    label: str = "",
) -> dict:
    """
    Permutation test for the Night Effect by shuffling daily OHLCV rows.

    The test statistic is the overnight Sharpe. I shuffle which day's
    (Open, Close) pair sits at which position in the sequence — this destroys
    any time-structure that causes the overnight edge while preserving the
    distribution of session-level returns.

    If the real overnight Sharpe is not in the tail of the shuffled distribution,
    the effect could just be a property of the individual daily sessions themselves,
    not of their ordering over time.
    """
    df = real_df.copy().dropna(subset=["Open", "Close"]).sort_index()
    df["Open"]  = pd.to_numeric(df["Open"],  errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Open", "Close"])

    rng = np.random.default_rng(seed)
    sim_overnight_sharpes: list[float] = []

    print(f"\n  Running {n_simulations} Monte Carlo permutations on {label} (Night Effect)...")
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{n_simulations}]")
        # shuffle the row indices → random assignment of daily sessions across time
        shuffled = df.sample(frac=1, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
        shuffled["prev_close"]    = shuffled["Close"].shift(1)
        shuffled["overnight_ret"] = (shuffled["Open"] - shuffled["prev_close"]) / shuffled["prev_close"]
        shuffled = shuffled.dropna()

        eq = (1 + shuffled["overnight_ret"]).cumprod() * 100_000.0
        eq.index = pd.RangeIndex(len(eq))
        eq_series = pd.Series(eq.values)
        sh = _sharpe_from_equity(eq_series)
        sim_overnight_sharpes.append(sh)

    result = _build_result_dict(label, "Night Effect — Overnight", real_overnight_sharpe,
                                np.array(sim_overnight_sharpes), n_simulations)
    result["real_intraday_sharpe"] = real_intraday_sharpe
    return result


# ── News Sentiment permutation test ─────────────────────────────────────────

def run_sentiment_test(
    merged_df: pd.DataFrame,
    real_sharpe: float,
    fee_per_leg: float = 0.0112,
    n_simulations: int = 300,
    seed: int = 42,
    label: str = "",
) -> dict:
    """
    Permutation test for the news sentiment strategy by shuffling signals.

    merged_df must contain 'Signal', 'next_ret' columns (pre-computed).
    Shuffling the Signal column destroys any relationship between news content
    and returns while keeping the fee structure and return distribution intact.
    """
    rng = np.random.default_rng(seed)
    signals = merged_df["Signal"].to_numpy().copy()
    sim_sharpes: list[float] = []

    print(f"\n  Running {n_simulations} Monte Carlo permutations on {label} (Sentiment)...")
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{n_simulations}]")
        rng.shuffle(signals)
        net_rets = signals * merged_df["next_ret"].to_numpy() - np.abs(signals) * fee_per_leg * 2
        eq = 100_000.0 * np.cumprod(1 + net_rets)
        eq_series = pd.Series(eq)
        sh = _sharpe_from_equity(eq_series)
        sim_sharpes.append(sh)

    return _build_result_dict(label, "News Sentiment (VADER)", real_sharpe,
                              np.array(sim_sharpes), n_simulations)


# ── Shared result builder + printer ─────────────────────────────────────────

def _build_result_dict(
    label: str, strategy: str, real_sharpe: float,
    sim_arr: np.ndarray, n_simulations: int,
) -> dict:
    p_value = float(np.mean(sim_arr >= real_sharpe))
    return {
        "label":             label,
        "strategy":          strategy,
        "real_sharpe":       real_sharpe,
        "simulated_sharpes": sim_arr,
        "p_value":           p_value,
        "is_significant":    p_value < 0.05,
        "percentile_rank":   float(np.mean(sim_arr < real_sharpe) * 100),
        "n_simulations":     n_simulations,
        "sim_mean":          float(sim_arr.mean()),
        "sim_std":           float(sim_arr.std()),
    }


def print_mc_result(r: dict) -> None:
    """Print a formatted summary of one permutation test result."""
    sig = "✓ SIGNIFICANT" if r["is_significant"] else "✗ NOT SIGNIFICANT"
    print(f"\n{'='*52}")
    print(f"  Monte Carlo — {r['strategy']} — {r['label']}")
    print(f"{'='*52}")
    print(f"  Simulations:      {r['n_simulations']}")
    print(f"  Real Sharpe:      {r['real_sharpe']:.3f}")
    print(f"  Sim. Mean Sharpe: {r['sim_mean']:.3f}  (Std: {r['sim_std']:.3f})")
    print(f"  Percentile Rank:  {r['percentile_rank']:.1f}%")
    print(f"  p-value:          {r['p_value']:.4f}")
    print(f"  Result:           {sig}  (α = 0.05)")
    print(f"{'='*52}")
