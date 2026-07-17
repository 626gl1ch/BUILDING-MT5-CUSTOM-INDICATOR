# RL-RBO Integration Plan (PPO)

## Overview
We will integrate a Proximal Policy Optimization (PPO) Reinforcement Learning agent directly into our existing RBO (Research, Backtest, Optimize) system. Instead of replacing the RBO system, the RL agent will act as an incredibly intelligent, dynamic "Strategy Generator." 

Once the RL agent is trained on our historical CSV data and indicators, it will output a series of trade signals. We will then pass those signals through our strict 3-layer RBO validation (Standard Backtest, Walk-Forward OOS, Permutation Test) to ensure the AI has actually found a robust edge and hasn't just memorized the data.

---

## 1. Environment Design (`rbo_rl_env.py`)
We will build a custom trading environment using `Gymnasium` (the modern standard for RL environments).

### Observation Space (The State)
The agent needs to "see" the market. We will feed it a normalized vector of precomputed indicators from our `indicators_library.py` for the current 5-minute candle:
*   **Trend & Momentum:** ADX, RSI, MACD Histogram
*   **Volatility & Distance:** Z-Score (distance from mean), Bollinger Band Width
*   **Position State:** Current open position (-1, 0, 1) and unrealized PnL.
*(All inputs will be scaled/normalized between -1 and 1 so the neural network can process them efficiently).*

### Action Space
A discrete action space for scalping:
*   `0`: **Hold / Stay Flat** (Do nothing, or close existing position)
*   `1`: **Buy / Go Long**
*   `2`: **Sell / Go Short**

### Reward Shaping
The reward function dictates what the agent learns. We will use a custom reward structure designed for day-trading/scalping:
*   **Step Reward:** Realized PnL of closed trades minus commissions and slippage (0.05% per side).
*   **Holding Penalty:** A tiny negative reward (e.g., -0.001) for every 5-minute candle it stays in a trade. This forces the agent to find quick scalps rather than holding for days.
*   **Drawdown Penalty:** Heavy negative reward if unrealized PnL dips below our 2.5 ATR stop-loss equivalent.

---

## 2. PPO Agent Implementation (`rbo_rl_agent.py`)
We will use `stable-baselines3`, the industry standard library for PPO.
*   **Network Architecture:** A Multi-Layer Perceptron (MLP) policy (e.g., two hidden layers of 64 or 128 neurons).
*   **Training Loop:** The agent will train iteratively on the In-Sample portion of our CSV data across all 4 assets (BTC, LTC, SOL, TRUMP).

---

## 3. RBO Validation Integration
Once the PPO agent finishes a training epoch, it will generate trade signals for the entire dataset. We will then treat the RL agent exactly like our other 7 grid-search strategies in `rbo_v2.py`:
1.  **Generate Signals:** Ask the trained model to predict actions across all historical bars.
2.  **RBO Evaluation:** Pass these signals into `BacktestCore.run_full_validation()`.
3.  **Strict Gates:** The RL model must pass the Expectancy Gate, Trade Frequency Gate (≥3 trades/day), Walk-Forward Retention Gate, and the Permutation Test (p-value < 0.10).

---

## 4. Execution Steps

### Step 1: Install Dependencies
We will need to install the RL libraries into your Python environment:
```bash
pip install stable-baselines3 gymnasium
```

### Step 2: Build the Gym Environment
Create `rbo_rl_env.py` to bridge our Pandas DataFrames with the Gymnasium API.

### Step 3: Build the Training & Validation Bridge
Create `train_rl.py` which will initialize the environment, train the PPO model, and then immediately pass the model's predictions into our `rbo_v2.py` pipeline for scoring.

---

## Open Questions for You

> [!IMPORTANT]
> **Q1: Training Time vs. Depth**
> RL training can be computationally intensive on a CPU. Do you want me to configure a "Fast" training run (e.g., 50,000 steps, ~5-10 minutes) for rapid prototyping, or a "Deep" training run (e.g., 500,000 steps, ~1-2 hours) to give the AI a real chance at mastering the data?
>
> *Default Plan: Start with a fast prototyping run to ensure the architecture works, then scale up.*

> [!IMPORTANT]
> **Q2: Neural Network Inputs**
> Are there any specific indicators from our massive library (e.g., SuperTrend, VWAP, Donchian) that you absolutely want the Neural Network to "see" in its observation space? 
>
> *Default Plan: I will select a robust mix of 8-10 diverse indicators (RSI, VWAP distance, ADX, Bollinger Band metrics).*
