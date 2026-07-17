# Implementation Plan: Phase 3 (The Quantitative Upgrade & 5000+ Strategies)

## Goal
To implement the advanced quantitative logic extracted from the 20 requested books (SuperSmoother, Fractional Differentiation, HMM Meta-Labeling, Optimal $f$) and to construct a system capable of discovering and testing thousands of novel, unseen algorithmic trading strategies.

## User Review Required
> [!CAUTION]
> **Regarding the 5,000 Strategies:** It is physically impossible to manually hardcode 5,000 completely unique Python trading strategies one by one. Furthermore, most "new" indicators on YouTube/Reddit are just variations of moving averages. 
> 
> **My Proposed Solution:** Instead of manual coding, I will build a **Combinatorial Strategy Generator** within `rbo_v2.py`. 
> I will code 30-50 highly advanced "Building Blocks" (e.g., Ehlers' DSP filters, Hurst Exponents, Fractal Dimensions, Machine Learning outputs). The Generator will dynamically combine these blocks (e.g., `If Hurst > 0.6 AND SuperSmoother crosses VWAP THEN Buy`) to automatically generate **10,000+ unique strategy permutations** during the search phase. 
> 
> **Do you approve of this dynamic generation approach?**

---

## Proposed Changes

### 1. Advanced Indicators Library (`indicators_library.py` [MODIFY])
I will implement the mathematical formulas extracted from the literature and state-of-the-art quantitative research into our library.
- **[NEW]** `calc_super_smoother()`: John Ehlers' 2-pole zero-lag filter.
- **[NEW]** `calc_fractional_diff()`: Marcos Lopez de Prado's method to make prices stationary while retaining memory.
- **[NEW]** `calc_hurst_exponent()`: To mathematically define if a market is trending, mean-reverting, or random walk.
- **[NEW]** `calc_directional_change()`: Tsang & Chen's event-based sampling indicator.

### 2. The Combinatorial Strategy Generator (`rbo_v2.py` [MODIFY])
Instead of 5 hardcoded strategies, I will rewrite `rbo_v2.py` to contain a `StrategyGenerator` class.
- It will define `Entry Conditions` (e.g., Breakouts, Mean Reversion bands, Momentum crosses).
- It will define `Regime Filters` (e.g., Hurst > 0.5, HMM State == 0).
- It will define `Exit Logic` (e.g., Time-based, Trailing ATR, Optimal $f$ ruin boundary).
- It will output a massive `GRIDS` dictionary containing thousands of unique logic combinations for `mass_search_sequential.py` to test.

### 3. HMM Meta-Labeling & Risk Management (`backtest_core.py` [MODIFY])
We will integrate the final pieces of the literature into the core engine.
- **[NEW]** **Leverage Space / Optimal $f$ Sizing:** Modifying the `run_backtest` function to dynamically scale position sizing based on Ralph Vince's math and current volatility.
- **[NEW]** **Meta-Labeling Veto:** Adding a mechanism where, even if the primary strategy fires a "BUY", if the HMM regime detects an incompatible market state, the trade is vetoed.

## Verification Plan
1. **Indicator Integrity:** I will write a quick validation script to ensure `SuperSmoother` actually lags less than an `EMA` and that `Fractional_Diff` passes the Augmented Dickey-Fuller (ADF) test for stationarity.
2. **Generator Load Test:** I will verify that `rbo_v2.py` successfully generates at least 5,000 valid, testable strategy definitions without crashing memory.
3. **Engine Run:** We will execute `mass_search_sequential.py` to ensure the core engine can ingest these 5,000 dynamic strategies and evaluate them across the 5 timeframes (Synthetic -> 1H -> 30m -> 15m -> 5m) as established in Phase 2.
