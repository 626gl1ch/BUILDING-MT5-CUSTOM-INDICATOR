# Implementation Plan: Offline RL (IQL) Meta-Controller

## Goal
Pivot from the deeply flawed end-to-end PPO approach to a robust, interpretable **Offline RL Meta-Controller using Implicit Q-Learning (IQL)**. Instead of the RL agent picking entries and exits from raw indicators, it will act as a portfolio manager, allocating capital across our already-validated RBO strategies based on current market regimes.

## Why We Are Doing This (The Bugs Fixed)
The previous end-to-end PPO attempt had severe structural flaws that would have produced "silently wrong" backtests:
1.  **Signal Misalignment:** Dropping NaNs shifted the indicator dataframe relative to the raw price dataframe, corrupting all signals.
2.  **In-Sample Leakage:** The agent was trained on BTC but validated across all symbols (including its training data).
3.  **Path Memorization:** PPO trained on a single, deterministic historical path without episode randomization, leading to severe overfitting.
4.  **Dynamic Stop Loss:** ATR stops were floating with live volatility instead of locking at entry time.
5.  **Extrapolation Error:** Online Q-Learning/PPO suffers when it cannot explore the environment live. Offline RL (IQL) is mathematically designed for learning from fixed historical datasets.

---

## Architecture & Implementation Steps

### 1. The Strategy Wrapper Interface (`strategy_menu.py`)
We will create a clean API wrapper for our existing RBO strategies (EMA Pullback, Donchian Breakout, BB/VWAP Mean Reversion, etc.) so they can be plugged into the Meta-Controller.

Each wrapped strategy will expose:
*   `.signal(df, idx)`: Returns `1` (Long), `-1` (Short), or `0` (Flat).
*   `.rolling_sharpe(df, idx)`: Returns the strategy's recent risk-adjusted performance.
*   `.bar_return(df, idx, current_position, entry_price, entry_atr)`: Returns the realized PnL of the strategy for that specific 5m bar, strictly enforcing fixed ATR stop-losses and transaction costs.

### 2. The Regime Tagger (`regime_tagger.py`)
The Meta-Controller needs to know the current market state. We will build a function that classifies the market into 4 discrete regimes using our precomputed indicators:
1.  **Low Volatility / Ranging** (e.g., ADX < 20, ATR below moving average)
2.  **High Volatility / Ranging**
3.  **Low Volatility / Trending** (e.g., ADX > 25)
4.  **High Volatility / Trending**

### 3. The IQL Meta-Controller (`meta_rl_agent.py`)
We will implement the complete scaffold provided in the research:
*   **`MetaState`**: A highly compressed, 21-dimensional state vector (Regime + Signals + Recent Sharpe + Exposure + Time of Day).
*   **Action Space**: Combinatorial discrete allocation across strategies (e.g., 0.0, 0.5, 1.0 weights per strategy).
*   **`RewardShaper`**: Differential Sharpe Ratio (Moody & Saffell) to penalize volatility, plus explicit penalties for drawdowns and turnover (overtrading).
*   **`TradingMetaEnv`**: The Gym environment that wires the state to the `_get_bar_pnl` hook.
*   **`OfflineDatasetBuilder` & `IQL`**: The data collector and the Implicit Q-Learning neural networks.

### 4. Validation Integration (`train_meta_iql.py`)
We will replace the flawed PPO training script. The new script will:
1.  Precompute indicators for all CSVs.
2.  Initialize the 5 Strategy Wrappers.
3.  Collect the offline dataset by running the historical data through the `TradingMetaEnv`.
4.  Train the IQL agent on rolling Walk-Forward folds.
5.  **Crucial Step:** Run the Permutation Test directly on the IQL agent's Out-Of-Sample (OOS) equity curve. If the shuffled sequence curve beats the real curve, the agent failed.

---

## Finalized Configurations (Based on Review)

> [!NOTE]
> **1. Strategy Correlation Check Passed:**
> Computed the pairwise OOS equity curve correlation across the 5 strategies. Correlation between *Donchian Breakout* and *Volatility Squeeze* is `0.361`. Since this is well below the 0.6 threshold, they provide distinct edge and both remain in the menu.

> [!TIP]
> **2. Binary Action Space:**
> The Meta-Controller action space is reduced from `[0.0, 0.5, 1.0]` to `[0.0, 1.0]`. This shrinks the state-action space from 243 choices down to 32, allowing the Offline Dataset to properly sample and cover the distribution without extreme sparsity.

> [!TIP]
> **3. Dynamic Regime Thresholds:**
> Instead of static thresholds (e.g., ADX > 25), the regime tagger computes a **rolling 60-day 70th percentile** for both ADX and ATR per symbol. This ensures the Meta-Controller generalizes cleanly across assets with structurally different baseline volatility.

> [!IMPORTANT]
> **4. Explicit Friction Modeling:**
> The `RewardConfig` enforces strict separation of transaction costs:
> - `commission_bps = 5.0` (One-way taker fee benchmarked to Binance)
> - `slippage_bps = 2.0` (Stress-test execution friction)
> Both are rigorously applied directly within the `.bar_return()` signal wrapper, ensuring the agent learns under realistic conditions.
