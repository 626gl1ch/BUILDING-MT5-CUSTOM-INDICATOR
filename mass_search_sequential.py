import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from rbo_v2 import GRIDS

def get_param_combos(param_dict, max_combos=5000):
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    all_combos = list(itertools.product(*values))
    if len(all_combos) > max_combos:
        rng = np.random.default_rng(777)
        idx = rng.choice(len(all_combos), size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in idx]
    return [dict(zip(keys, c)) for c in all_combos]

def main():
    print("======================================================")
    print("  MASSIVE SCALE RBO SEARCH (Sequential Memory-Safe)")
    print("======================================================")
    
    engine = BacktestCore()
    print("\n[1/3] Loading data...")
    all_data = engine.load_all_data()
    
    print("\n[2/3] Precomputing indicators (this takes ~3 min)...")
    t0 = time.time()
    precomputed = {}
    for sym, df in all_data.items():
        print(f"  Computing {sym}...")
        precomputed[sym] = add_all_indicators(df)
    print(f"  Done in {time.time() - t0:.0f}s")

    target_strategies = ['ema_pullback', 'donchian_breakout', 'bb_vwap_mr', 'rsi_divergence', 'squeeze_breakout']
    
    tasks = []
    for stype in target_strategies:
        combos = get_param_combos(GRIDS[stype]['params'], max_combos=2000) # 2000 combos per strategy = 10,000 total
        for c in combos:
            tasks.append((stype, c))
            
    total_combos = len(tasks)
    print(f"\n[3/3] Total Combinations to Evaluate: {total_combos}")
    
    confirmed_all = {stype: [] for stype in target_strategies}
    RESULTS_FILE = "strategies_confirmed.json"
    
    t0 = time.time()
    n_tested = 0
    total_found = 0
    
    for stype, c in tasks:
        n_tested += 1
        fn = GRIDS[stype]['fn']
        
        try:
            result = engine.run_full_validation(
                precomputed, fn, c,
                min_trades_per_day=3.0,
                min_assets=4,
                n_permutations=200
            )
        except Exception as e:
            result = None
            
        if n_tested % 50 == 0:
            elapsed = time.time() - t0
            rate = n_tested / elapsed if elapsed > 0 else 0
            eta = (total_combos - n_tested) / rate if rate > 0 else 0
            print(f"[{n_tested}/{total_combos}] ETA: {eta:.0f}s | Rate: {rate:.1f} tests/sec | Found: {total_found}")
            
        if result and result['passed']:
            print(f"\n  [SUCCESS] {stype} passed all gates! {c}")
            confirmed_all[stype].append(result)
            total_found += 1
            
            with open(RESULTS_FILE, 'w') as f:
                json.dump(confirmed_all, f, indent=2, default=str)
                
            # Keep detailed report updated
            with open("validated_strategies_report.txt", "w") as f:
                f.write("=== GENUINELY VALIDATED STRATEGIES REPORT ===\n")
                f.write("These strategies passed the Standard Backtest, Walk-Forward Test, and Permutation Test without lowering any safety gates.\n\n")
                for s_type, strats in confirmed_all.items():
                    f.write(f"--- {s_type.upper()} ({len(strats)} found) ---\n")
                    for i, s in enumerate(strats):
                        f.write(f"\n[{s_type}] Strategy #{i+1}\n")
                        f.write(f"Parameters: {s['params']}\n")
                        agg = s['backtest']
                        f.write(f"Performance (Aggregate): Expectancy={agg['expectancy']:.4f}, PF={agg['profit_factor']:.2f}, WR={agg['win_rate']:.1f}%, Sharpe={agg['sharpe_ratio']:.3f}, Trades/Day={agg['trades_per_day']:.1f}\n")
                        wf = s['walkforward']
                        f.write(f"Walk-Forward OOS: Expectancy={wf['out_of_sample']['expectancy']:.4f}, Sharpe={wf['out_of_sample']['sharpe_ratio']:.3f}\n")
                        perm = s['permutation']
                        f.write(f"Permutation Test: p_value={perm['p_value']:.4f}\n")
                        f.write("Per-Symbol Breakdown:\n")
                        for sym, m in s['symbol_results'].items():
                            f.write(f"  {sym}: Expectancy={m['expectancy']:.4f}, PF={m['profit_factor']:.2f}, WR={m['win_rate']:.1f}%, Sharpe={m['sharpe_ratio']:.3f}\n")
                        f.write("-" * 50 + "\n")

    print(f"\nSearch complete in {time.time() - t0:.0f} seconds.")
    print(f"Total Confirmed Strategies: {total_found}")

if __name__ == '__main__':
    main()
