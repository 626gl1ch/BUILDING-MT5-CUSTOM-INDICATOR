import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.abspath('.'))
from backtest_core import BacktestCore
from indicators_library import add_all_indicators
from rbo_rl_env import RboScalpingEnv

def rl_signal_wrapper(df, p):
    """
    This acts as a standard RBO signal generator, but instead of 
    calculating logic, it just reads the 'rl_action' column that 
    the trained neural network already populated.
    """
    if 'rl_action' not in df.columns:
        return pd.Series(0, index=df.index)
        
    # RL Actions: 0=Flat, 1=Long, 2=Short
    # RBO Signals: 0=Flat, 1=Long, -1=Short
    actions = df['rl_action'].copy()
    signals = pd.Series(0, index=df.index)
    signals[actions == 1] = 1
    signals[actions == 2] = -1
    return signals

def main():
    print("======================================================")
    print("  RL-RBO PPO Integration (Fast Prototyping Run)")
    print("======================================================")

    engine = BacktestCore()
    
    print("\n[1/4] Loading Data & Indicators...")
    all_data = engine.load_all_data()
    precomputed = {}
    for sym, df in all_data.items():
        print(f"  Computing indicators for {sym}...")
        precomputed[sym] = add_all_indicators(df)

    # Use BTC for training the agent (In-Sample)
    train_sym = 'BTCUSD_5M'
    if train_sym not in precomputed:
        train_sym = list(precomputed.keys())[0]
        
    train_df = precomputed[train_sym].copy()
    
    # We train on the first 70% of the data (In-Sample)
    split_idx = int(len(train_df) * 0.7)
    is_df = train_df.iloc[:split_idx].copy()
    
    print(f"\n[2/4] Setting up PPO Environment on {train_sym} (IS: {len(is_df)} candles)...")
    env = DummyVecEnv([lambda: RboScalpingEnv(is_df)])

    print("\n[3/4] Training PPO Agent (50,000 steps)...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=50000)
    
    print("\n[4/4] Generating Signals & Validating via RBO...")
    
    # Now we ask the model to predict actions for ALL assets (full datasets)
    for sym in precomputed.keys():
        df = precomputed[sym]
        eval_env = RboScalpingEnv(df)
        
        actions = []
        obs, _ = eval_env.reset()
        for _ in range(len(df)):
            action, _states = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
        df['rl_action'] = actions

    # Run the strict 3-layer validation
    p_config = {
        'model': 'PPO_50k_BTC',
        # SL/TP mapped back to ATR for the backtest engine
        'sl_atr': 2.5,
        'tp_atr': 5.0,
        'max_bars_hold': 12
    }
    
    print("\nExecuting RBO Validation Gates (Backtest -> Walk-Forward -> Permutation)...")
    result = engine.run_full_validation(
        precomputed, 
        rl_signal_wrapper, 
        p_config,
        min_trades_per_day=1.0, # Relaxed for prototype
        min_assets=4,
        n_permutations=200
    )
    
    agg = result['backtest']
    gates = result['gates']
    perm = result['permutation']
    
    print("\n=== RL MODEL VALIDATION RESULTS ===")
    print(f"  Expectancy : {agg['expectancy']:.4f}")
    print(f"  Win Rate   : {agg['win_rate']:.1f}%")
    print(f"  Sharpe Ratio: {agg['sharpe_ratio']:.3f}")
    print(f"  Trades/Day : {agg['trades_per_day']:.1f}")
    print("\n  [GATES PASS STATUS]")
    for k, v in gates.items():
        print(f"    {k}: {v}")
        
    print(f"\n  Final Result: {'PASSED ALL GATES! ★' if result['passed'] else 'Failed Validation'}")

if __name__ == "__main__":
    main()
