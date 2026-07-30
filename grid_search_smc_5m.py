import os
import sys
import json
import itertools
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from smc_enhanced_library import generate_smc_fvg, generate_smc_combined, generate_smc_volume

def main():
    print("=" * 80)
    print(" 5M SMC - HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    print("\n[1/3] Loading 5m Historical Datasets...")
    data_5m = engine.load_all_data(suffix="5min_1year")
    
    print("\n[2/3] Precomputing Indicators...")
    precomputed = {}
    for sym, df in data_5m.items():
        precomputed[sym] = add_all_indicators(df)
        
    print("\n[3/3] Running Grid Search...")
    
    # We will test the FVG strategy and Combined strategy with wider ATR stops
    # because 0.5 ATR is likely smaller than the spread/fee on 5m TF.
    
    grid = {
        'lookback': [20],
        'sl_atr': [1.0, 1.5, 2.0],
        'tp_atr': [2.0, 3.0, 4.0],
        'trailing': [False],
        'fvg_lookback': [3, 5]
    }
    
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    
    best_pf = 0.0
    best_params = None
    
    results = []
    
    for idx, c in enumerate(combos):
        # We will use FVG strategy for this sweep as it's cleaner and maintains trade count
        print(f"\n--- Combo {idx+1}/{len(combos)} ---")
        print(f"Params: {c}")
        
        try:
            sym_results, agg, _ = engine.run_multi_symbol(
                precomputed, generate_smc_fvg, c, slippage_pct=0.0002, fee_pct=0.00055
            )
            
            pf = agg['profit_factor']
            wr = agg['win_rate']
            sharpe = agg['sharpe_ratio']
            tpd = agg['trades_per_day']
            
            print(f"PF: {pf:.2f} | WR: {wr:.1f}% | Sharpe: {sharpe:.3f} | TPD: {tpd:.2f}")
            
            res_entry = {
                'params': c,
                'pf': pf,
                'wr': wr,
                'sharpe': sharpe,
                'tpd': tpd
            }
            results.append(res_entry)
            
            if pf > best_pf and tpd > 0.5:
                best_pf = pf
                best_params = c
                
        except Exception as e:
            print(f"Error: {e}")
            
    print("\n==================================================")
    print(f"BEST PF: {best_pf} with Params: {best_params}")
    print("==================================================")

if __name__ == '__main__':
    main()
