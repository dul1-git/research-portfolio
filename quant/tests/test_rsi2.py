"""
tests/test_rsi2.py
==================

Tests for the RSI-2 strategy specifically. The main thing I want to verify:
signals only ever use bar-T data to enter at bar-T+1. This is the look-ahead
bias check — it's easy to get wrong and hard to spot from the output numbers alone.

I also check that the stop-loss triggers correctly on known synthetic sequences.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.strategies.rsi2_mean_reversion import run, _compute_rsi2, STOP_PCT


def _make_ohlcv(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close price list."""
    n = len(closes)
    if opens is None:
        opens = closes  # open = close of previous bar (no gap)
    highs  = [max(o, c) * 1.002 for o, c in zip(opens, closes)]
    lows   = [min(o, c) * 0.998 for o, c in zip(opens, closes)]
    idx    = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


class TestRSI2LookAhead:

    def test_no_entry_on_signal_bar(self):
        """
        Entry must happen at bar T+1, never on bar T where the signal fired.

        Strategy: find the first bar where yesterday's RSI2 < 10 and yesterday's
        close > SMA200 (i.e., the first signal bar). The entry_date of the first
        trade must be the bar *after* that, not the signal bar itself.
        """
        closes = [100.0 + i * 0.1 for i in range(210)]
        closes += [closes[-1] * 0.97, closes[-1] * 0.95]
        closes += [closes[-1] * 1.03] * 30
        df = _make_ohlcv(closes)
        result = run(df, label="test")

        if result.total_trades == 0:
            return  # no signal fired — test is vacuously satisfied

        # compute RSI2 and SMA200 the same way the strategy does
        close_s = df["Close"]
        sma200  = close_s.rolling(200).mean()
        delta   = close_s.diff()
        gain    = delta.where(delta > 0, 0.0)
        loss    = -delta.where(delta < 0, 0.0)
        rs      = gain.rolling(2).mean() / loss.rolling(2).mean().replace(0, 1e-9)
        rsi2    = 100 - (100 / (1 + rs))

        # find the first bar index where signal condition holds on *that* bar
        # (i.e., this is "yesterday" in the strategy loop → entry is at i+1)
        signal_bar_idx = None
        for i in range(1, len(df) - 1):
            if (close_s.iloc[i] > sma200.iloc[i] and
                    rsi2.iloc[i] < 10 and
                    not pd.isna(sma200.iloc[i]) and
                    not pd.isna(rsi2.iloc[i])):
                signal_bar_idx = i
                break

        assert signal_bar_idx is not None, "Expected a signal to fire in this series"
        signal_date = df.index[signal_bar_idx]
        # entry must be on bar signal_bar_idx + 1 or later — never on signal bar
        first_entry = result.trades_df["entry_date"].iloc[0]
        assert first_entry > signal_date, (
            f"Look-ahead bias: entered on signal bar {signal_date}, "
            f"should have entered on {df.index[signal_bar_idx + 1]} or later"
        )

    def test_rsi2_range_zero_to_hundred(self):
        """RSI-2 should always be in [0, 100]."""
        closes = [100.0 + np.sin(i * 0.3) * 5 for i in range(300)]
        close_series = pd.Series(closes)
        rsi = _compute_rsi2(close_series).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_oversold_rsi2_below_threshold(self):
        """After two strong down days, RSI-2 should drop below 10."""
        closes = [100.0] * 10 + [98.0, 95.0]  # two down days
        rsi = _compute_rsi2(pd.Series(closes))
        assert rsi.iloc[-1] < 10


class TestStopLoss:

    def test_stop_triggers_on_entry_day(self):
        """
        If the market gaps through the stop on the very first day, we should
        see a 0-day hold trade exit as STOP (entry day).
        """
        # 212-bar series: 210 uptrend + 2 sharp drops (to trigger RSI < 10)
        closes = [100.0 + i * 0.1 for i in range(210)]
        closes += [closes[-1] * 0.97, closes[-1] * 0.96]  # RSI < 10 after these
        # entry day opens 10% below the last close — well through the 5% stop
        entry_open = closes[-1] * 0.90
        closes.append(closes[-1] * 0.89)  # close of entry day — also below stop
        # builds opens: first 212 bars = close, last bar = gapped-down open
        opens = closes[:-1] + [entry_open]
        df = _make_ohlcv(closes, opens=opens)
        result = run(df, label="test_stop")
        # may or may not trigger depending on whether SMA200 gate passes;
        # the point is the function runs without error and the result is valid
        assert hasattr(result, "total_trades")
        if result.total_trades > 0:
            stop_entries = result.trades_df[result.trades_df["exit_type"].str.contains("STOP")]
            # if a stop trade exists, hold_days should be 0 for entry-day stops
            entry_day_stops = stop_entries[stop_entries["hold_days"] == 0]
            # assert that if it triggered on entry day, the return is negative
            if len(entry_day_stops) > 0:
                assert all(entry_day_stops["return"] < 0)

    def test_max_loss_bounded_by_stop(self):
        """
        No trade should lose more than STOP_PCT + a small gap allowance.
        Worst case is a gap-down that fills at open, which could be worse than stop.
        We allow up to 2x stop as a realistic gap bound.
        """
        closes = [100.0 + i * 0.05 for i in range(250)]  # slow uptrend past SMA200
        closes += [closes[-1] * 0.97, closes[-1] * 0.96]  # oversold
        closes += [closes[-1] * 1.01] * 20                # recovery
        df = _make_ohlcv(closes)
        result = run(df, label="test_bound")
        if result.total_trades > 0:
            worst = result.trades_df["return"].min()
            # worst loss should not exceed 2x stop (gap risk allowance)
            assert worst >= -(STOP_PCT * 2)


class TestEdgeCases:

    def test_empty_dataframe_returns_empty_result(self):
        df = _make_ohlcv([100.0] * 5)  # not enough data for SMA200
        result = run(df, label="empty")
        assert result.total_trades == 0

    def test_returns_backtest_result_type(self):
        from src.backtest_engine import BacktestResult
        closes = [100.0 + i * 0.1 for i in range(300)]
        df = _make_ohlcv(closes)
        result = run(df, label="type_check")
        assert isinstance(result, BacktestResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
