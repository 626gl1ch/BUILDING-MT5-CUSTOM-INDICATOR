import os
import sys
import json
import itertools
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from smc_enhanced_library import (
    generate_smc_baseline, generate_smc_rsi, 
    generate_smc_stoch_rsi, generate_smc_volume, 
    generate_smc_fvg, generate_smc_combined
)

def evaluate_strategy(engine, data_dict, fn, params, name):
    print(f"\n--- Evaluating: {name} ---")
    print(f"Params: {params}")
    
    # Standard Backtest
    sym_results, agg, _ = engine.run_multi_symbol(
        data_dict, fn, params, slippage_pct=0.0002, fee_pct=0.00055
    )
    
    # Walk-Forward Test (70/30)
    wf = engine.run_walkforward(
        data_dict, fn, params, split_pct=0.70, slippage_pct=0.0002, fee_pct=0.00055
    )
    
    is_agg = wf['in_sample']
    oos_agg = wf['out_of_sample']
    
    print(f"  [Standard Backtest] PF: {agg['profit_factor']:.2f} | WR: {agg['win_rate']:.1f}% | Sharpe: {agg['sharpe_ratio']:.3f} | Trades/Day: {agg['trades_per_day']:.2f}")
    print(f"  [Walk-Forward OOS]  PF: {oos_agg['profit_factor']:.2f} | WR: {oos_agg['win_rate']:.1f}% | Sharpe: {oos_agg['sharpe_ratio']:.3f} | Trades/Day: {oos_agg['trades_per_day']:.2f}")
    
    return {
        'name': name,
        'params': params,
        'backtest': agg,
        'walkforward': wf
    }

def main():
    print("=" * 80)
    print(" 5M SMC LIQUIDITY SWEEP RESEARCH & OPTIMIZATION")
    print("=" * 80)
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    print("\n[1/3] Loading 5m Historical Datasets...")
    data_5m = engine.load_all_data(suffix="5min_1year")
    
    print("\n[2/3] Precomputing Indicators...")
    precomputed = {}
    for sym, df in data_5m.items():
        print(f"  Precomputing indicators for {sym}...")
        precomputed[sym] = add_all_indicators(df)
        
    print("\n[3/3] Running Experiments...")
    
    experiments = [
        {'name': '1. Baseline SMC', 'fn': generate_smc_baseline, 'params': {'lookback': 20, 'sl_atr': 0.5, 'tp_atr': 2.0, 'max_bars_hold': 999, 'trailing': True}},
        {'name': '2. SMC + RSI', 'fn': generate_smc_rsi, 'params': {'lookback': 20, 'sl_atr': 0.5, 'tp_atr': 2.0, 'rsi_os': 40, 'rsi_ob': 60, 'max_bars_hold': 999, 'trailing': True}},
        {'name': '3. SMC + StochRSI', 'fn': generate_smc_stoch_rsi, 'params': {'lookback': 20, 'sl_atr': 0.5, 'tp_atr': 2.0, 'stoch_os': 25, 'stoch_ob': 75, 'max_bars_hold': 999, 'trailing': True}},
        {'name': '4. SMC + Volume', 'fn': generate_smc_volume, 'params': {'lookback': 20, 'sl_atr': 0.5, 'tp_atr': 2.0, 'rvol_thresh': 1.2, 'max_bars_hold': 999, 'trailing': True}},
        {'name': '5. SMC + FVG Filter', 'fn': generate_smc_fvg, 'params': {'lookback': 20, 'sl_atr': 0.5, 'tp_atr': 2.0, 'fvg_lookback': 5, 'max_bars_hold': 999, 'trailing': True}},
        {'name': '6. SMC Combined Synergy', 'fn': generate_smc_combined, 'params': {'lookback': 20, 'sl_atr': 0.5, 'tp_atr': 2.5, 'stoch_os': 30, 'stoch_ob': 70, 'rvol_thresh': 1.1, 'fvg_lookback': 4, 'max_bars_hold': 999, 'trailing': True}}
    ]
    
    results = []
    for exp in experiments:
        res = evaluate_strategy(engine, precomputed, exp['fn'], exp['params'], exp['name'])
        results.append(res)
        
    with open("smc_5m_research_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
        
    print("\n" + "=" * 80)
    print(" DONE. Results saved to smc_5m_research_results.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
