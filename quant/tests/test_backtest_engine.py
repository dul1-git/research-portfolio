"""
tests/test_backtest_engine.py
==============================

Unit tests for compute_metrics. I wanted to be sure the Sharpe/Sortino
computation and drawdown logic are correct on known synthetic inputs before
trusting the results on real data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import compute_metrics, BacktestResult


def _flat_equity(n: int = 252, start: float = 100_000.0) -> pd.Series:
    """Equity series that never moves — Sharpe should be ~0."""
    return pd.Series([start] * n, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _trending_equity(n: int = 252, daily_ret: float = 0.001, seed: int = 0) -> pd.Series:
    """
    Upward-trending equity with a small random noise component so daily returns
    have non-zero std and Sharpe is actually computable.
    """
    rng  = np.random.default_rng(seed)
    rets = daily_ret + rng.normal(0, daily_ret * 0.3, size=n)  # positive drift + noise
    vals = 100_000.0 * np.cumprod(1 + rets)
    return pd.Series(vals, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _dummy_trades(n_wins: int = 7, n_losses: int = 3) -> list[dict]:
    """Dummy trade list with controllable win/loss counts."""
    trades = []
    for i in range(n_wins):
        trades.append({"entry_date": pd.Timestamp("2020-01-02"),
                       "exit_date": pd.Timestamp("2020-01-03"),
                       "return": 0.01, "hold_days": 1, "exit_type": "SIGNAL"})
    for i in range(n_losses):
        trades.append({"entry_date": pd.Timestamp("2020-01-04"),
                       "exit_date": pd.Timestamp("2020-01-05"),
                       "return": -0.02, "hold_days": 1, "exit_type": "STOP"})
    return trades


def _dummy_close(n: int = 252, start: float = 100.0) -> pd.Series:
    return pd.Series([start] * n, index=pd.date_range("2020-01-01", periods=n, freq="B"))


class TestComputeMetrics:

    def test_empty_trades_returns_zeroed_result(self):
        eq = _trending_equity()
        result = compute_metrics(_flat_equity(), [], _dummy_close(),
                                 label="test", strategy_name="test")
        assert result.total_trades == 0
        assert result.sharpe == 0.0
        assert result.cagr == 0.0

    def test_win_rate_correct(self):
        trades = _dummy_trades(n_wins=7, n_losses=3)
        result = compute_metrics(_trending_equity(), trades, _dummy_close(),
                                 label="test", strategy_name="test")
        assert result.total_trades == 10
        assert abs(result.win_rate - 0.7) < 1e-9

    def test_profit_factor_correct(self):
        # 7 wins of 1%, 3 losses of -2% → PF = (7*0.01) / (3*0.02) = 0.07/0.06 ≈ 1.1667
        trades = _dummy_trades(n_wins=7, n_losses=3)
        result = compute_metrics(_trending_equity(), trades, _dummy_close(),
                                 label="test", strategy_name="test")
        expected_pf = (7 * 0.01) / (3 * 0.02)
        assert abs(result.profit_factor - expected_pf) < 1e-6

    def test_flat_equity_sharpe_near_negative(self):
        # flat equity → zero daily returns → excess return = -rf → negative Sharpe
        eq = _flat_equity()
        trades = _dummy_trades()
        result = compute_metrics(eq, trades, _dummy_close(), label="test", strategy_name="test")
        # std dev of flat returns is 0, so Sharpe returns 0.0 (not computable)
        assert result.sharpe == 0.0

    def test_drawdown_is_nonpositive(self):
        eq = _trending_equity()
        result = compute_metrics(eq, _dummy_trades(), _dummy_close(),
                                 label="test", strategy_name="test")
        assert result.max_drawdown <= 0.0

    def test_no_drawdown_on_monotone_equity(self):
        # strictly increasing equity → drawdown should be 0
        eq = _trending_equity()
        result = compute_metrics(eq, _dummy_trades(), _dummy_close(),
                                 label="test", strategy_name="test")
        assert abs(result.max_drawdown) < 1e-9

    def test_sharpe_positive_for_consistently_profitable_equity(self):
        eq = _trending_equity(daily_ret=0.005)  # strong daily return
        result = compute_metrics(eq, _dummy_trades(), _dummy_close(),
                                 label="test", strategy_name="test")
        assert result.sharpe > 0

    def test_cagr_roughly_correct(self):
        # 252 days at 0.1% per day → CAGR ≈ 28.4%
        daily_ret = 0.001
        eq = _trending_equity(n=252, daily_ret=daily_ret)
        close = _dummy_close(n=252)
        result = compute_metrics(eq, _dummy_trades(), close,
                                 label="test", strategy_name="test")
        expected_cagr = (1 + daily_ret) ** 252 - 1
        assert abs(result.cagr - expected_cagr) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
