import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from itertools import product

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import calc_alma, calc_adx, calc_swing_points, calc_sma
from backtest_core import BacktestCore

warnings.filterwarnings('ignore')

def precompute_alma_grid(df):
    res = df.copy()
    close = res['close']
    
    # ALMA multiple periods
    res['alma_9'] = calc_alma(close, 9)
    res['alma_14'] = calc_alma(close, 14)
    res['alma_21'] = calc_alma(close, 21)
    
    # Regime Filter
    adx, _, _ = calc_adx(res, 14)
    res['adx_14'] = adx
    
    # Sweep calculation
    sh, sl = calc_swing_points(res, 20)
    res['swing_high_20'] = sh
    res['swing_low_20'] = sl
    
    # Volume filter
    vol_sma = calc_sma(res['volume'], 20)
    res['vol_sma'] = vol_sma
    
    return res

def generate_alma_strategy(df, params):
    """
    Optimized Strategy Generator focused purely on ALMA sweeps.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Params
    alma_period = params.get('alma_period', 9)
    adx_threshold = params.get('adx_threshold', 25)
    require_vol_surge = params.get('require_vol_surge', False)
    
    swing_high = df.get('swing_high_20', close)
    swing_low = df.get('swing_low_20', close)
    alma = df.get(f'alma_{alma_period}', close)
    adx = df.get('adx_14', pd.Series(20, index=df.index))
    vol_sma = df.get('vol_sma', volume)
    
    # 1. SWEEP (Equal Lows / Highs)
    # The current bar closed back inside the range after breaking it
    sweep_bull = (low < swing_low.shift(1)) & (close > swing_low.shift(1)) & (close.shift(1) > swing_low.shift(1))
    sweep_bear = (high > swing_high.shift(1)) & (close < swing_high.shift(1)) & (close.shift(1) < swing_high.shift(1))
    
    # 2. ALMA CONFIRMATION
    alma_up = alma > alma.shift(1)
    alma_dn = alma < alma.shift(1)
    
    # 3. VOLUME FILTER
    if require_vol_surge:
        vol_ok = volume > vol_sma
    else:
        vol_ok = pd.Series(True, index=df.index)
        
    # 4. REGIME FILTER (Trend Strength)
    reg_ok = adx > adx_threshold
    
    long_cond = sweep_bull & alma_up & vol_ok & reg_ok
    short_cond = sweep_bear & alma_dn & vol_ok & reg_ok
    
    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    
    return signals, None, None

def main():
    print("=========================================================")
    print("  OPTIMIZING ALMA SWEEP STRATEGY (15M TIMEFRAME)")
    print("=========================================================")
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    print("\n[1/3] Loading 15M Datasets...")
    data_15m = engine.load_all_data(suffix="15min_1year")
    
    print("\n[2/3] Precomputing Indicators...")
    precomp_15m = {}
    for sym, df in data_15m.items():
        print(f"  Precomputing 15m indicators for {sym}...")
        precomp_15m[sym] = precompute_alma_grid(df)
        
    # Build Hyperparameter Grid
    alma_periods = [9, 14, 21]
    adx_thresholds = [20, 25]
    vol_surges = [True, False]
    sl_atrs = [1.5, 2.0]
    tp_atrs = [2.5, 3.0, 4.0]
    
    grid = list(product(alma_periods, adx_thresholds, vol_surges, sl_atrs, tp_atrs))
    strategies = []
    
    for i, (ap, adx, vol, sl, tp) in enumerate(grid):
        name = f"ALMA_Opt_{i}_P{ap}_ADX{adx}_Vol{vol}_SL{sl}_TP{tp}"
        params = {
            'alma_period': ap,
            'adx_threshold': adx,
            'require_vol_surge': vol,
            'sl_atr': sl,
            'tp_atr': tp,
            'max_bars_hold': 48,
            'risk_pct': 0.01
        }
        strategies.append({'name': name, 'params': params})
        
    print(f"\nCreated {len(strategies)} permutations for Grid Search.")
    
    print("\n[3/3] Running Optimization & Walk-Forward Validation...")
    
    results = []
    
    for s in strategies:
        res = engine.run_full_validation(
            precomp_15m, generate_alma_strategy, s['params'],
            min_trades_per_day=0.05, min_assets=1, n_permutations=20
        )
        agg = res['backtest']
        wf = res['walkforward']['out_of_sample']
        
        pf = agg['profit_factor']
        wr = agg['win_rate']
        sharpe = agg['sharpe_ratio']
        oos_pf = wf['profit_factor']
        
        results.append({
            'name': s['name'],
            'params': s['params'],
            'pf': pf,
            'wr': wr,
            'sharpe': sharpe,
            'total_trades': agg['total_trades'],
            'oos_pf': oos_pf
        })
        
        print(f"  {s['name']}: PF={pf:.2f} | OOS_PF={oos_pf:.2f} | Trades={agg['total_trades']}")
        
    # Sort by OOS Profit Factor (robustness)
    results.sort(key=lambda x: x['oos_pf'], reverse=True)
    
    # Save top 20 to file
    with open("optimized_alma_results.txt", "w") as f:
        f.write("=== OPTIMIZED ALMA STRATEGY RESULTS ===\n")
        f.write("Ranked by Out-Of-Sample (OOS) Profit Factor\n")
        f.write("=" * 60 + "\n\n")
        for i, w in enumerate(results[:20]):
            f.write(f"Rank {i+1}: {w['name']}\n")
            f.write(f"  Params:       {json.dumps(w['params'])}\n")
            f.write(f"  In-Sample PF: {w['pf']:.2f}\n")
            f.write(f"  OOS PF:       {w['oos_pf']:.2f}\n")
            f.write(f"  Sharpe Ratio: {w['sharpe']:.3f}\n")
            f.write(f"  Win Rate:     {w['wr']:.1f}%\n")
            f.write(f"  Total Trades: {w['total_trades']}\n")
            f.write("-" * 40 + "\n")
            
    print("\n=========================================================")
    print("  Optimization Complete! Saved to optimized_alma_results.txt")
    print("=========================================================")

if __name__ == '__main__':
    main()
