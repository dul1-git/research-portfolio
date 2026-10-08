# Algorithmic Trading Research — CSE vs NYSE

I got curious about whether short-term trading strategies that are documented
to work on liquid US markets would transfer to a small frontier exchange. The
Colombo Stock Exchange (CSE) in Sri Lanka is an interesting test case: it's
relatively illiquid, has a high retail broker fee structure, and goes through
extended closures during macro crises (the 2020 COVID shutdown, the 2022
economic collapse).

The short answer is: the strategies don't transfer.

---

## What I found

**The CSE's 2.24% round-trip broker fee renders moot any algorithmic edge
at any frequency above monthly trading.**

This is the core finding. Some of the signals are actually quite strong in                                            gross terms. The overnight drift anomaly on
JKH produces a theoretical CAGR of +52% per year. But at daily trading frequency,
the 2.24% round-trip fee consumes the entire return before it reaches the trader.

The same strategies on SPY, where commission-free execution is available:

| Strategy | CSE Net CAGR | SPY Net CAGR |
|---|---|---|
| Night Effect (Overnight only) | fee-destroyed | +7.87% |
| RSI-2 Mean Reversion | -9.89% | +5.69% |
| News Sentiment (VADER) | -71.02% | not tested (news feed is CSE-specific) |

---

## The overnight drift shows promise

Every single rupee of cumulative JKH price appreciation over 8+ years happened
between close and the next morning's open. Trading during market hours — buying
at open, selling at close — would have lost 32% over the same period.

The academic explanation is that institutional order flow concentrates in the
overnight session, away from the public order book, while retail selling pressure
absorbs intraday. The effect is clearly seen in the data, but cannot be extract                                     at retail commissions.

---

## tested methodology

**Look-ahead prevention:** Every signal reads bar T's closing data. All execution
happens at bar T+1's open. Nothing reads "the future."

**Mark-to-market equity curves:** Sharpe and Sortino are computed on a real
daily equity series — the portfolio value is marked to the current close while
in a position, and flat otherwise. 

Computing Sharpe on per-trade returns and multiplying by sqrt(252) inflates the number
significantly if you're only in 20–30 trades per year. My v1 backtest had this
bug and the inflated Sharpe looked great until I caught it.

**Realistic fill assumptions:** Stop-losses check the intraday Low on every bar.
Gap-down fills use the Opening price, not the theoretical stop — on crash days,
you don't get filled at your stop price.

**Monte Carlo significance testing:** For every strategy, I ran 300 synthetic
permutations to check whether the observed edge is real or noise. The permutation
method varies by strategy:
- RSI-2: block bootstrap on the price series (preserves volatility clustering)
- Night Effect: shuffle which daily session appears at which point in time
- News Sentiment: shuffle the signal column, keep prices fixed

---

## Results in full

```
Strategy                  Market      Trades  Win%  AvgW%  AvgL%  PF    CAGR%   B&H%    MDD%    Sharpe  Sortino
------------------------  ----------  ------  ----  -----  -----  ----  ------  ------  ------  ------  -------
Night Effect — Intraday   JKH (CSE)   2280    30.0  +1.10  -0.68  0.00  -32.11   3.53   -96.99   -2.10    -2.89
Night Effect — Overnight  JKH (CSE)   2280    47.6  +0.72  -0.33  0.00  +52.33   3.53   -25.27   +2.59    +3.02
Night Effect — Intraday   SPY (NYSE)  5201    53.9  +0.58  -0.65  0.00   +3.00  11.09   -46.67   -0.06    -0.08
Night Effect — Overnight  SPY (NYSE)  5201    55.7  +0.44  -0.48  0.00   +7.87  11.09   -29.41   +0.28    +0.32
RSI-2 Mean Reversion      JKH (CSE)    143    42.7  +1.48  -2.09  0.53   -9.89   2.39   -59.20   -1.41    -1.84
RSI-2 Mean Reversion      SPY (NYSE)   432    70.8  +0.89  -1.25  1.73   +5.69  11.09   -15.19   +0.11    +0.15
News Sentiment (VADER)    JKH (CSE)     24    37.5  +5.67  -5.03  0.68  -71.02   —      -38.23   -1.82    -2.22
```

---

## Monte Carlo results (300 simulations)

RSI-2 on SPY lands at the 44th percentile of synthetic runs — right in the
middle of the noise distribution. The 70% win rate and 1.73 profit factor are
real, but they don't clear the 95% significance threshold. The edge, if it
exists, is smaller than the raw numbers suggest.

RSI-2 on JKH lands at the 13th percentile — the strategy actively does worse
than most random noise configurations on this market. The fee structure isn't
just eating the edge, it's making things worse than random.

The Night Effect on JKH is the most statistically interesting result —
the overnight Sharpe of 2.59 is far outside what block-shuffled paths produce,
which means the time-of-day structure is real, not an artifact of the session
return distribution.

---

## Structure

```
quant/
├── run_all.py                      # single entry point
├── requirements.txt
├── data/
│   ├── jkh_10_years.csv            # static CSE dataset (~2,281 days)
│   ├── jkh_news.json               # ~100 recent headlines (Google News RSS)
│   └── data_sources.md
├── src/
│   ├── backtest_engine.py          # shared metrics logic
│   └── strategies/
│       ├── night_effect.py
│       ├── rsi2_mean_reversion.py
│       ├── news_sentiment.py
│       └── monte_carlo.py          # significance tests for all 3 strategies
└── tests/
    ├── test_backtest_engine.py
    └── test_rsi2.py
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run everything
python run_all.py

# skip Monte Carlo (faster)
python run_all.py --skip-monte-carlo

# run tests
pytest tests/ -v
```

---

## References

- Connors, L. & Alvarez, C. (2008). *Short-Term Trading Strategies That Work*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 11.
- Cont, R. (2001). Empirical properties of asset returns. *Quantitative Finance*, 1(2).
