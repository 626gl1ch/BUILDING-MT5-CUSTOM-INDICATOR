import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from smc_enhanced_library import generate_smc_god_mode

def main():
    print("=" * 80)
    print(" 5M SMC - ZERO FRICTION PROOF OF EDGE")
    print("=" * 80)
    
    engine = BacktestCore(commission=0.0, initial_capital=10000.0, risk_pct=0.01)
    
    data_5m = engine.load_all_data(suffix="5min_1year")
    
    precomputed = {}
    for sym, df in data_5m.items():
        precomputed[sym] = add_all_indicators(df)
        
    c = {'lookback': 20, 'sl_atr': 2.5, 'tp_atr': 4.0, 'trailing': False, 'fvg_lookback': 5, 'rvol_thresh': 1.2, 'ema_p': 600}
    
    print(f"\nEvaluating Params: {c}")
    
    print("\n[WITH RETAIL FRICTION (0.055% fee, 0.02% slippage)]")
    _, agg_f, _ = engine.run_multi_symbol(precomputed, generate_smc_god_mode, c, slippage_pct=0.0002, fee_pct=0.00055)
    print(f"PF: {agg_f['profit_factor']:.2f} | WR: {agg_f['win_rate']:.1f}% | Sharpe: {agg_f['sharpe_ratio']:.3f} | Trades/Day: {agg_f['trades_per_day']:.2f}")

    print("\n[WITH ZERO FRICTION (0% fee, 0% slippage)]")
    _, agg_z, _ = engine.run_multi_symbol(precomputed, generate_smc_god_mode, c, slippage_pct=0.0, fee_pct=0.0)
    print(f"PF: {agg_z['profit_factor']:.2f} | WR: {agg_z['win_rate']:.1f}% | Sharpe: {agg_z['sharpe_ratio']:.3f} | Trades/Day: {agg_z['trades_per_day']:.2f}")

if __name__ == '__main__':
    main()
