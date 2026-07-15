import os
import sys
import json
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from rbo_v2 import GRIDS, signal_rsi_divergence

def get_param_combos(param_dict, max_combos=5000):
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    all_combos = list(itertools.product(*values))
    if len(all_combos) > max_combos:
        rng = np.random.default_rng(777)
        idx = rng.choice(len(all_combos), size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in idx]
    return [dict(zip(keys, c)) for c in all_combos]

# Global for worker processes
_engine = None
_precomputed = None

def worker_init():
    """Each worker loads and precomputes the data once."""
    global _engine, _precomputed
    _engine = BacktestCore()
    all_data = _engine.load_all_data()
    _precomputed = {}
    for sym, df in all_data.items():
        _precomputed[sym] = add_all_indicators(df)

def evaluate_combo(strategy_type, combo):
    """Worker task."""
    fn = GRIDS[strategy_type]['fn']
    try:
        result = _engine.run_full_validation(
            _precomputed, fn, combo,
            min_trades_per_day=3.0,
            min_assets=4,
            n_permutations=200
        )
        return strategy_type, combo, result
    except Exception as e:
        return strategy_type, combo, None

def main():
    print("======================================================")
    print("  MASSIVE SCALE RBO SEARCH (Multiprocessing)")
    print("======================================================")
    
    # We focus only on our 5 Meta-Controller strategies
    target_strategies = ['ema_pullback', 'donchian_breakout', 'bb_vwap_mr', 'rsi_divergence', 'squeeze_breakout']
    
    # Generate tasks
    tasks = []
    total_combos = 0
    for stype in target_strategies:
        combos = get_param_combos(GRIDS[stype]['params'], max_combos=3000) # 3000 combos per strategy
        for c in combos:
            tasks.append((stype, c))
        total_combos += len(combos)
        
    print(f"Total Combinations to Evaluate: {total_combos}")
    
    confirmed_all = {stype: [] for stype in target_strategies}
    RESULTS_FILE = "strategies_confirmed.json"
    
    t0 = time.time()
    n_tested = 0
    
    # Launch Process Pool
    # Windows spawn means worker_init takes ~3 mins per worker to precompute.
    max_workers = os.cpu_count() - 1 or 1
    print(f"Starting {max_workers} worker processes (Expect a 3-minute delay while workers precompute indicators)...")
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
        futures = {executor.submit(evaluate_combo, stype, c): (stype, c) for stype, c in tasks}
        
        for future in as_completed(futures):
            n_tested += 1
            stype, c, result = future.result()
            
            if n_tested % 100 == 0:
                elapsed = time.time() - t0
                rate = n_tested / elapsed
                eta = (total_combos - n_tested) / rate if rate > 0 else 0
                print(f"[{n_tested}/{total_combos}] ETA: {eta:.0f}s | Rate: {rate:.1f} tests/sec")
            
            if result and result['passed']:
                print(f"\n  [SUCCESS] {stype} passed all gates! {c}")
                confirmed_all[stype].append(result)
                
                # Save incrementally
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(confirmed_all, f, indent=2, default=str)
                    
    print(f"\nSearch complete in {time.time() - t0:.0f} seconds.")
    total_found = sum(len(v) for v in confirmed_all.values())
    print(f"Total Confirmed Strategies: {total_found}")
    
    # Write the detailed text report
    with open("validated_strategies_report.txt", "w") as f:
        f.write("=== GENUINELY VALIDATED STRATEGIES REPORT ===\n")
        f.write("These strategies passed the Standard Backtest, Walk-Forward Test, and Permutation Test without lowering any safety gates.\n\n")
        for stype, strats in confirmed_all.items():
            f.write(f"--- {stype.upper()} ({len(strats)} found) ---\n")
            for i, s in enumerate(strats):
                f.write(f"\n[{stype}] Strategy #{i+1}\n")
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

if __name__ == '__main__':
    main()
