# my research portfolio

Three independent research projects I built to explore things I got genuinely curious about.
Each one started as a question, not a project — the code is just how I went about answering it.

---

## [`quant/`](./quant/) — Algorithmic Trading on Frontier vs Liquid Markets

I wanted to know whether short-term trading strategies that work on the S&P 500 would
also work on a small frontier market. The answer is no, and the reason is interesting —
it's not the signal that fails, it's the fee structure.

Four strategies tested across two markets (CSE and NYSE), with a Monte Carlo significance
layer to check whether any of the edges were real or just noise.

**Key finding:** CSE's 2.24% round-trip broker fee structurally destroys every strategy
above monthly frequency. The overnight drift anomaly on JKH is the most extreme example —
52% annual CAGR in theory, completely unextractable in practice.

---

## [`physics/`](./physics/) — Market Microstructure as Turbulence

Ongoing. The statistical signatures of financial returns — fat tails, volatility clustering,
multifractal scaling — are nearly identical to those of turbulent fluid flows. I'm working
through the theoretical connection and building a Hurst exponent / multifractal volatility
regime detector to use as a filter on the quant strategies.

---

## [`automation/`](./automation/) — Automation & Media Processing Pipelines

End-to-end automation work: media processing pipelines, API integrations, scheduled
publishing systems. Built with Python, FFmpeg, and various platform APIs.
