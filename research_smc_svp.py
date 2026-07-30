import os
import sys
import itertools

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from smc_enhanced_library import generate_smc_vol_profile, generate_smc_baseline

def main():
    print("=" * 80)
    print(" 5M SMC - SESSION VOLUME PROFILE (POC) RESEARCH")
    print("=" * 80)
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    print("\n[1/3] Loading 5m Historical Datasets...")
    data_5m = engine.load_all_data(suffix="5min_1year")
    
    print("\n[2/3] Precomputing Indicators (including SVP)...")
    precomputed = {}
    for sym, df in data_5m.items():
        precomputed[sym] = add_all_indicators(df)
        
    print("\n[3/3] Running Grid Search...")
    
    grid = {
        'lookback': [20],
        'sl_atr': [2.0, 2.5],
        'tp_atr': [3.0, 4.0],
        'trailing': [False]
    }
    
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    
    print("\n--- BASELINE (Retail Friction) ---")
    _, agg_base, _ = engine.run_multi_symbol(precomputed, generate_smc_baseline, {'lookback': 20, 'sl_atr': 2.5, 'tp_atr': 4.0, 'trailing': False}, slippage_pct=0.0002, fee_pct=0.00055)
    print(f"PF: {agg_base['profit_factor']:.2f} | WR: {agg_base['win_rate']:.1f}% | Sharpe: {agg_base['sharpe_ratio']:.3f} | Trades/Day: {agg_base['trades_per_day']:.2f}")

    best_pf = 0.0
    best_params = None
    
    for idx, c in enumerate(combos):
        print(f"\n--- SVP POC Filter - Combo {idx+1}/{len(combos)} ---")
        print(f"Params: {c}")
        
        try:
            _, agg, _ = engine.run_multi_symbol(
                precomputed, generate_smc_vol_profile, c, slippage_pct=0.0002, fee_pct=0.00055
            )
            
            pf = agg['profit_factor']
            print(f"PF: {pf:.2f} | WR: {agg['win_rate']:.1f}% | Sharpe: {agg['sharpe_ratio']:.3f} | TPD: {agg['trades_per_day']:.2f}")
            
            if pf > best_pf and agg['trades_per_day'] > 0.1:
                best_pf = pf
                best_params = c
        except Exception as e:
            print(f"Error: {e}")
            
    print("\n==================================================")
    print(f"BEST PF (SVP Filter): {best_pf} | Params: {best_params}")
    print("==================================================")

if __name__ == '__main__':
    main()
