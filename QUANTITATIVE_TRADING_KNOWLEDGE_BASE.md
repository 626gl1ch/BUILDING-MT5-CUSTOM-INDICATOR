# 📚 QUANTITATIVE TRADING KNOWLEDGE BASE & ALGORITHMIC RESEARCH

This document is a comprehensive quantitative reference manual and book synthesis for algorithmic trading system development, regime detection, digital signal processing, statistical validation, and risk management.

---

## 1. EBOOK ACCESS & RESEARCH LINKS GUIDE

Below is a breakdown of legal access options, academic sources, open-source repositories, and official codebases for each book in your research curriculum:

### A. Modular Systems & Money Management
*   **Systematic Trading (Robert Carver)**
    *   *Core Concepts*: Portfolio volatility targeting, forecast scale factors, cash/futures leverage management, diversification multiplier.
    *   *Official Code*: [pysystemtrade GitHub Repository](https://github.com/pst-group/pysystemtrade) — The author's production backtest & execution engine.
    *   *Examples*: [systematictradingexamples GitHub](https://github.com/robcarver17/systematictradingexamples)
*   **The Leverage Space Trading Model & The Mathematics of Money Management (Ralph Vince)**
    *   *Core Concepts*: Optimal $f$, Leverage Space, Terminal Wealth Relative (TWR), Parametric risk distributions, serial dependency & arcsine loss laws.
    *   *Access*: Academic libraries, Wiley Online Library, Amazon.

### B. Machine Learning, Feature Engineering & Validation
*   **Advances in Financial Machine Learning (Marcos López de Prado)**
    *   *Core Concepts*: Triple-Barrier Labeling, Fractional Differentiation (balancing stationarity vs memory), Purged & Embargoed Cross-Validation, Feature Importance (MDI, MDA, SFI).
    *   *Open-Source Implementations*:
        *   [mlfinlab (Hudson & Thames)](https://github.com/hudson-and-thames/mlfinlab) — Production Python package for De Prado's algorithms.
        *   [boyboi86/AFML GitHub](https://github.com/boyboi86/AFML) — Exercise solutions & code implementations.
        *   [BlackArbsCEO/Adv_Fin_ML_Exercises GitHub](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises) — Jupyter notebook implementations.
*   **Testing and Tuning Market Trading Systems & Permutation Tests (Timothy Masters)**
    *   *Core Concepts*: Differential Evolution optimization, Permutation & Randomization Monte Carlo tests, Avoidance of in-sample trade overlap, Out-of-sample degradation gates.
    *   *Official Source Code*: [Timothy Masters C++ Repositories (GitHub)](https://github.com/topics/timothy-masters)

### C. Digital Signal Processing (DSP) & Cycle Analysis (John Ehlers)
*   **Rocket Science for Traders / Cybernetic Analysis / Cycle Analysis for Traders**
    *   *Core Concepts*: SuperSmoother Filter, Fractal Adaptive Moving Average (FRAMA), Hilbert Transform Phase Discriminator, Homodyne Discriminator, Empirical Mode Decomposition.
    *   *Official Papers & EasyLanguage/Python Code*: [MESA Software Papers](https://www.mesasoftware.com/papers.html)

### D. Regime Change & Pattern Recognition
*   **Detecting Regime Change in Computational Finance (Prodromos Tsinaslanidis & Achilles Zapranis)**
    *   *Core Concepts*: Directional Change (DC) intrinsic time sampling, Hidden Markov Models (HMM), Trend-Turn indicators, Volatility regime clustering.
    *   *References*: Directional Change research framework by Edward Tsang et al.

---

## 2. ADVANCED REGIME DETECTION: HMM & DIRECTIONAL CHANGE (DC)

### A. Directional Change (DC) Intrinsic Time Concept
Traditional fixed-time bars (5m, 15m, 1h) slice price by time intervals regardless of price activity. **Directional Change (DC)** samples data based on price movement magnitude $\delta$:

1.  **Directional Change Event**: A price reversal exceeding threshold $\delta = \frac{|P_t - P_{extreme}|}{P_{extreme}}$.
2.  **Overshoot (OS) Event**: The continuation of price movement in the direction of the DC event until the next reversal occurs.

$$\text{Total Intrinsic Movement} = \text{DC Event} + \text{OS Event}$$

### B. Hidden Markov Model (HMM) 3-State Classification
Market regimes are modeled as hidden states $S_t \in \{0: \text{Ranging/Choppy}, 1: \text{Bullish Trend}, 2: \text{Bearish Trend}\}$ generating observed feature vectors $X_t = [\text{Log Returns}, \text{Normalized ATR}]$.

```
             ┌─────────────────────────┐
             │   Ranging / High Noise  │
             └────────────┬────────────┘
                         / \
                        /   \
  ┌───────────────────┐/     \┌───────────────────┐
  │ Bullish Trend     │<═════>│ Bearish Trend     │
  └───────────────────┘       └───────────────────┘
```

---

## 3. DIGITAL SIGNAL PROCESSING (DSP) FORMULAS

### Ehlers 2-Pole SuperSmoother Filter
$$\gamma = \frac{\pi \cdot \sqrt{2}}{\text{Period}}, \quad \alpha = e^{-\gamma}, \quad \beta = 2 \alpha \cos(\gamma)$$
$$c_2 = \beta, \quad c_3 = -\alpha^2, \quad c_1 = 1 - c_2 - c_3$$
$$\text{SuperSmoother}_t = c_1 \cdot \left(\frac{P_t + P_{t-1}}{2}\right) + c_2 \cdot \text{SuperSmoother}_{t-1} + c_3 \cdot \text{SuperSmoother}_{t-2}$$

---

## 4. 3-LAYER VALIDATION PIPELINE ARCHITECTURE

1.  **Gate 1 — Standard Backtest**: Profit Factor $\ge 1.2$, Win Rate $\ge 50\%$, Expectancy $> 0$.
2.  **Gate 2 — 70/30 Purged Walk-Forward**: $Sharpe_{OOS} \ge 0.50 \cdot Sharpe_{IS}$ (Degradation Check).
3.  **Gate 3 — 200-Shuffle Permutation Test**: Shuffles trade returns 200 times. Monte Carlo $p\text{-value} < 0.10$.

---

## 5. STRATEGY GAUNTLET PIPELINE

```
Synthetic Data (1H) ──> [Gate 1 + 2 + 3] ──> PASS?
                                               │ (Yes)
                                               ▼
Live Data (1H, 30m, 15m, 5m) ──> [Gate 1 + 2 + 3 on ALL Timeframes & Assets]
                                               │ (Yes)
                                               ▼
                                      👑 GOD STRATEGY #X
```
