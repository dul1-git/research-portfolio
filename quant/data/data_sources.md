# data provenance

## `jkh_10_years.csv`

Daily OHLCV data for a large-cap equity listed on the Colombo Stock Exchange (CSE), Sri Lanka.

- **Date range:** 2016 – 2026 (~2,281 trading days)
- **Fields:** Date, Open, High, Low, Close, Volume

Provided as a static dataset. Acquisition method is not documented in this repo.

### Periods worth knowing about

| Period | What happened |
|---|---|
| 2020-03 to 2020-06 | CSE suspended trading during COVID-19 lockdown |
| 2022-04 to 2022-08 | Circuit breaker halts during the Sri Lanka economic crisis |

These aren't gaps I cleaned out — I kept them because they're real market
conditions and any strategy that can't handle them isn't production-viable.

---

## `jkh_news.json`

~100 recent news headlines scraped from the Google News RSS endpoint
(`news.google.com/rss/search`) for the CSE company and exchange. Public endpoint,
no authentication. Fields: `date`, `headline`, `url`.

Covers only recent weeks — the sentiment backtest window is limited by this.
Re-fetch by running `python src/scripts/fetch_news.py` (not included in this repo —
it's a straightforward `requests` + `feedparser` script).

---

## SPY

Fetched live from Yahoo Finance via `yfinance` at runtime. No static file needed.
