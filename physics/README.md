# Physics — Market Microstructure as Turbulence

Working notes on the mathematical connection between financial returns and turbulent fluid dynamics.

## The observation

The statistical signatures of financial return series and turbulent fluid velocity increments are nearly identical:
- Fat-tailed distributions (leptokurtosis)
- Volatility clustering (long-range autocorrelation in |returns|)
- Multifractal scaling — structure functions scale as power laws with non-linear exponents

This isn't a coincidence or a loose analogy. The mathematical machinery developed for
turbulence (Kolmogorov scaling, Hurst exponents, multifractal formalism) translates
directly to price series analysis.

## What I'm building

A Hurst exponent / multifractal volatility regime detector. The idea: compute rolling
scaling exponents on the price series to classify the current market regime, then use
that classification as a filter on the quant strategies (enter RSI-2 only when the
market is in a mean-reverting regime, skip it during trending or turbulent phases).

## References

- Mandelbrot, B. (1997). *Fractals and Scaling in Finance.*
- Ghashghaie et al. (1996). Turbulent cascades in foreign exchange markets. *Nature*, 381.
- Calvet, L. & Fisher, A. (2002). Multifractality in asset returns. *Review of Economics and Statistics*.

*Active development — code coming.*
