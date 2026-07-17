# Implementation Plan: RBO → Meta-Controller Integration

## Goal
Now that the Offline RL (IQL) Meta-Controller architecture is verified, we need to transition it from using unoptimized placeholder parameters to using **genuinely validated strategy parameters**. We will run the RBO search engine to find true edges, and then pipe those parameters directly into the Meta-Controller's training loop.

## Architecture & Implementation Steps

### 1. Execute RBO Search (`rbo_v2.py`)
We will execute the `rbo_v2.py` master engine.
*   I have already filtered the search grid to focus strictly on the 5 strategy types used by our Meta-Controller (`ema_pullback`, `donchian_breakout`, `bb_vwap_mr`, `rsi_divergence`, `squeeze_breakout`).
*   The bot will sweep up to 120 combinations per strategy type across all 4 assets (`BTC`, `LTC`, `SOL`, `TRUMP`).
*   Strategies that survive the brutal 3-layer validation (Backtest + Walk-Forward + Permutation Test) will be saved to `strategies_confirmed.json`.

### 2. Dynamic Strategy Ingestion
We will modify `train_meta_iql.py` to dynamically read `strategies_confirmed.json`.
*   Instead of hardcoding `p_ema`, `p_donchian`, etc., the script will load the top 1 confirmed parameter set for each of the 5 strategy types.
*   *Fallback Logic*: If the RBO search bot fails to find a fully confirmed strategy for a specific type (because the safety gates are so strict), the Meta-Controller will skip that strategy or fallback to a historically stable default, ensuring the training loop doesn't crash.

### 3. Train the Meta-Controller
Once the validated parameters are plugged into the `StrategyMenu`, we will run the `train_meta_iql.py` pipeline again.
*   The IQL agent will collect offline data using the truly validated edges.
*   Because the underlying strategies now possess genuine edge (unlike the previous test), the Meta-Controller's OOS equity curve is expected to cleanly **pass the Permutation Test**.

---

## Open Questions for You

> [!IMPORTANT]
> **Strictness of RBO Validation Gates**
> The validation gates in `backtest_core.py` are notoriously strict. It is very possible that for some strategy types (like `squeeze_breakout`), the bot will scan 120 combinations and find **0** that pass all Walk-Forward and Permutation checks. 
> 
> **Question:** If the search bot fails to find a confirmed strategy for one of the 5 types, how would you like to handle it?
> 1. Allow the Meta-Controller to train with fewer than 5 strategies (e.g., a menu of only the 3 confirmed strategies).
> 2. Automatically lower the RBO validation thresholds slightly and re-run the search for that specific strategy until one passes.
