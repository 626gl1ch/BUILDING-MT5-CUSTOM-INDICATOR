# Implementation Plan: Project Titan ("God Strategy" & Literature Integration)

## Goal
To evolve the current backtesting engine into an enterprise-grade, multi-timeframe evaluation suite capable of discovering "God Strategies" (strategies that survive synthetic data generation, Walk-Forward, Permutation, and multi-asset/multi-timeframe testing). Furthermore, we will systematically integrate the advanced quantitative concepts from the 20 requested textbooks.

## Open Questions & Literature Constraints
> [!IMPORTANT]
> **Regarding Book Downloads:** As an AI developed by Google, I am strictly prohibited by safety guidelines from providing direct links to illegal, pirated PDF repositories (like LibGen or Sci-Hub) for copyrighted material. However, I have compiled a list of where these books can be legally acquired or accessed (e.g., Amazon, Springer, academic libraries) in the section below.
> 
> **Regarding the Deep Dive:** I have successfully created the deep dive on the Hidden Markov Model (HMM) and Directional Change algorithm. You can view and download it here: [regime_change_deep_dive.md](file:///C:/Users/DANIEL/.gemini/antigravity/brain/15a411d1-2eb9-4f1f-9c42-f78fb56c4061/regime_change_deep_dive.md)

---

## Phase 1: Codebase Comb & Multi-Timeframe Engine

### 1. File Cleanup
We will remove files related to data downloading as requested, since the data is already sourced.
- **[DELETE]** `download_mt5_data.py`
- **[DELETE]** `check_correlations.py` (If deemed unnecessary for the core bot).
- **[KEEP]** `data_resampler.py`

### 2. Multi-Timeframe Data Generation (`data_resampler.py` [MODIFY])
I will modify `data_resampler.py` to automatically scan for base `5min` CSV files and generate the missing timeframes:
- `15min`
- `30min`
- `1H` 
It will intelligently skip generation if the file already exists in the directory to save compute time.

---

## Phase 2: The "God Strategy" Validation Pipeline

We will heavily modify `mass_search_sequential.py` and `backtest_core.py` to implement a rigorous, multi-stage gauntlet.

### Stage 1: Synthetic/Authentic Generation Test
Before testing on live historical data, we will use techniques from *Advances in Financial Machine Learning* to generate authentic synthetic price paths (e.g., using a GARCH model or bootstrapping). 
- A parameter combination must first prove it has a statistical edge on this synthetic data.

### Stage 2: The Core Gauntlet
If it passes Stage 1, it enters the gauntlet on the standard timeframe:
- Standard Backtest (Positive Expectancy)
- Walk-Forward Test (Nested Cross-Validation as per Timothy Masters)
- Permutation / Randomization Test (p-value < 0.05)

### Stage 3: Multi-Timeframe & Multi-Asset Domination
If a strategy survives Stage 2, it is tested across **ALL assets** (BTC, ETH, SOL, etc.) and **ALL timeframes** (5m, 15m, 30m, 1H).
- **The God Condition:** If and only if the strategy maintains a positive expectancy and passes the permutation test across *every single timeframe and asset*, it is christened a "God Strategy".
- It is saved to a specialized registry as `God_Strategy_1`, `God_Strategy_2`, with its complete ruleset.

---

## Phase 3: Literature Integration (The Quantitative Upgrade)

Implementing the concepts from all 16+ textbooks is a massive undertaking. We will inject them into the architecture module by module:

### 1. Optimization & Testing (Timothy Masters & Ralph Vince)
- **Nested Walk-Forward Optimization:** Avoiding in-sample/out-of-sample bleed.
- **Differential Evolution Optimization:** Replacing our current Grid Search with the Differential Evolution algorithm to find optimal parameters thousands of times faster without overfitting.
- **Leverage Space Trading:** Implementing Ralph Vince's Optimal f / Leverage Space math into the Meta-Controller to determine exact position sizing to maximize terminal wealth without hitting the risk ruin boundary.

### 2. Digital Signal Processing (John Ehlers)
- We will add Ehlers' custom DSP indicators to `indicators_library.py`:
  - **SuperSmoother Filter:** To remove aliasing noise from price data.
  - **Empirical Mode Decomposition (EMD):** To detect true market cycles.

### 3. Market Regimes & Meta-Labeling (De Prado & Tsang)
- **HMM Regime Tagger:** Implementing the Directional Change / HMM model (detailed in your deep dive) to classify the market as Trending or Mean-Reverting.
- **Meta-Labeling:** Adding a secondary Machine Learning model that doesn't predict *direction*, but predicts *whether the primary God Strategy will be right or wrong* on a given trade, sizing up or down accordingly.

---

## Book Acquisition Links (Legal Avenues)
*Systematic Trading* (Carver) - [Amazon](https://www.amazon.com/Systematic-Trading-designing-trading-investing/dp/0857194453)
*Trading Systems and Methods* (Kaufman) - [Wiley / Amazon](https://www.amazon.com/Trading-Systems-Methods-Website-Kaufman/dp/1118043561)
*Advances in Financial Machine Learning* (Lopez de Prado) - [Amazon](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
*The Leverage Space Trading Model* (Vince) - [Amazon](https://www.amazon.com/Leverage-Space-Trading-Model-Mathematical/dp/047028483X)
*Rocket Science for Traders / Cybernetic Analysis* (Ehlers) - [Amazon](https://www.amazon.com/Rocket-Science-Traders-Digital-Processing/dp/0471405671)
*Testing & Tuning Market Trading Systems / Permutation Tests* (Masters) - [Amazon](https://www.amazon.com/Testing-Tuning-Market-Trading-Systems/dp/0996048925)
*Numerical Recipes* - [Cambridge University Press](http://numerical.recipes/)
*Detecting Regime Change in Computational Finance* (Tsang/Chen) - [CRC Press / Routledge](https://www.routledge.com/Detecting-Regime-Change-in-Computational-Finance-Data-Science-Machine/Chen-Tsang/p/book/9780367536287)

## User Review Required
> [!CAUTION]
> Phase 2 (The God Strategy Pipeline) is extremely computationally heavy. Testing 10,000 parameter combinations across synthetic data + 4 assets + 4 timeframes (16 total datasets per parameter) will take significant CPU time. 
> 
> **Are you ready for me to begin Phase 1 (Codebase Cleanup & Resampler) and Phase 2 (The Multi-Timeframe God Engine)?**
