"""
Mean Reversion Strategy Search Bot
Endlessly pairs indicators to find a winning mean reversion strategy.
Saves any combination meeting or coming close to WR >= 70% and PF >= 1.8.
"""

import os
import sys
import json
import itertools
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from indicators_library import add_all_indicators
from backtest_core import BacktestCore

def generate_mr_signal(df, params):
    """
    Generates mean reversion buy/sell signals based on params.
    Ensures every strategy has OB/OS, Bands, Choppiness, and Volatility filters.
    """
    # Extract params
    ob_os_ind = params['ob_os_indicator'] # e.g. 'rsi_14', 'stoch_k_14', 'williams_14', 'cci_14'
    band_ind = params['band_indicator']   # e.g. 'bb_b_pct_20', 'keltner_b_pct_20', 'zscore_20'
    chop_ind = params['chop_indicator']   # e.g. 'chop_14', 'chop_7'
    vol_ind = params['vol_indicator']     # e.g. 'natr_14', 'bb_width_20'
    
    ob_val = params['overbought']
    os_val = params['oversold']
    band_high = params['band_high']
    band_low = params['band_low']
    chop_min = params['chop_min']
    vol_min = params['vol_min']
    
    close = df['close']
    
    # 1. OB/OS signals
    if 'williams' in ob_os_ind:
        # Williams %R is negative (-100 to 0)
        oversold_cond = df[ob_os_ind] < os_val
        overbought_cond = df[ob_os_ind] > ob_val
    elif 'cci' in ob_os_ind:
        oversold_cond = df[ob_os_ind] < os_val
        overbought_cond = df[ob_os_ind] > ob_val
    else:
        # RSI, Stoch
        oversold_cond = df[ob_os_ind] < os_val
        overbought_cond = df[ob_os_ind] > ob_val
        
    # 2. Band Extreme signals
    if 'zscore' in band_ind:
        band_low_cond = df[band_ind] < band_low
        band_high_cond = df[band_ind] > band_high
    else:
        # Bollinger Bands %B or Keltner %B (usually 0 to 1)
        band_low_cond = df[band_ind] < band_low
        band_high_cond = df[band_ind] > band_high
        
    # 3. Choppiness Index filter (ensure high choppiness/ranging)
    chop_cond = df[chop_ind] > chop_min
    
    # 4. Volatility filter (ensure enough volatility to move)
    vol_cond = df[vol_ind] > vol_min
    
    # Combine signals
    long_signals = oversold_cond & band_low_cond & chop_cond & vol_cond
    short_signals = overbought_cond & band_high_cond & chop_cond & vol_cond
    
    signals = pd.Series(0, index=df.index)
    signals[long_signals] = 1
    signals[short_signals] = -1
    
    return signals

def main():
    parser = argparse.ArgumentParser(description="Mean Reversion Strategy Bot")
    parser.add_argument('--test', action='store_true', help="Run 50 combinations in test mode")
    parser.add_argument('--limit', type=int, default=0, help="Limit the number of combinations to run (0 for unlimited)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STARTING MEAN REVERSION STRATEGY SEARCH BOT")
    print("=" * 80)
    
    # Initialize backtester
    backtester = BacktestCore()
    
    # Load all historical data
    print("Loading data...")
    all_data = backtester.load_all_data()
    if not all_data:
        print("Error: No data loaded. Exiting.")
        sys.exit(1)
        
    # Precompute all indicators on all files
    print("\nPrecomputing 150+ indicators on all files (this takes a moment)...")
    precomputed_data = {}
    for symbol, df in all_data.items():
        precomputed_data[symbol] = add_all_indicators(df)
        
    # Define Parameter Grid Pools
    ob_os_indicators = ['rsi_7', 'rsi_14', 'stoch_k_7', 'stoch_k_14', 'cci_14', 'williams_14']
    band_indicators = ['bb_b_pct_10', 'bb_b_pct_20', 'keltner_b_pct_20', 'zscore_20']
    chop_indicators = ['chop_7', 'chop_14', 'chop_21']
    vol_indicators = ['natr_14', 'bb_width_20']
    
    # Define parameter ranges
    oversold_vals = {
        'rsi_7': [20, 25, 30], 'rsi_14': [25, 30, 35],
        'stoch_k_7': [10, 15, 20], 'stoch_k_14': [15, 20, 25],
        'cci_14': [-120, -100, -80], 'williams_14': [-90, -80, -75]
    }
    overbought_vals = {
        'rsi_7': [80, 75, 70], 'rsi_14': [75, 70, 65],
        'stoch_k_7': [90, 85, 80], 'stoch_k_14': [85, 80, 75],
        'cci_14': [120, 100, 80], 'williams_14': [-10, -20, -25]
    }
    band_low_vals = {
        'bb_b_pct_10': [0.0, 0.05, 0.1], 'bb_b_pct_20': [0.0, 0.05, 0.1],
        'keltner_b_pct_20': [0.0, 0.05, 0.1], 'zscore_20': [-2.2, -2.0, -1.8]
    }
    band_high_vals = {
        'bb_b_pct_10': [1.0, 0.95, 0.9], 'bb_b_pct_20': [1.0, 0.95, 0.9],
        'keltner_b_pct_20': [1.0, 0.95, 0.9], 'zscore_20': [2.2, 2.0, 1.8]
    }
    chop_min_vals = [50, 52, 55, 58]
    vol_min_vals = {
        'natr_14': [0.0005, 0.001, 0.0015],
        'bb_width_20': [0.001, 0.002, 0.003]
    }
    
    atr_sl_vals = [1.5, 2.0, 2.5]
    atr_tp_vals = [2.0, 3.0, 4.0]
    max_hold_vals = [24, 36, 48, 72]
    
    # Save file
    save_file = 'mr_winning_strategies.json'
    winning_strategies = []
    
    # Load existing if exists
    if os.path.exists(save_file):
        try:
            with open(save_file, 'r') as f:
                winning_strategies = json.load(f)
            print(f"Loaded {len(winning_strategies)} existing winning strategies.")
        except Exception:
            pass
            
    # Generate all parameter combinations
    print("Generating search space...")
    all_combos = []
    for ob_os in ob_os_indicators:
        for band in band_indicators:
            for chop in chop_indicators:
                for vol in vol_indicators:
                    for os_v in oversold_vals[ob_os]:
                        ob_v = overbought_vals[ob_os][oversold_vals[ob_os].index(os_v)]
                        for bl_v in band_low_vals[band]:
                            bh_v = band_high_vals[band][band_low_vals[band].index(bl_v)]
                            for ch_v in chop_min_vals:
                                for vl_v in vol_min_vals[vol]:
                                    for sl in atr_sl_vals:
                                        for tp in atr_tp_vals:
                                            for hold in max_hold_vals:
                                                all_combos.append({
                                                    'ob_os_indicator': ob_os,
                                                    'band_indicator': band,
                                                    'chop_indicator': chop,
                                                    'vol_indicator': vol,
                                                    'oversold': os_v,
                                                    'overbought': ob_v,
                                                    'band_low': bl_v,
                                                    'band_high': bh_v,
                                                    'chop_min': ch_v,
                                                    'vol_min': vl_v,
                                                    'sl_atr': sl,
                                                    'tp_atr': tp,
                                                    'max_bars_hold': hold
                                                })
                                                
    random.shuffle(all_combos)
    total_combinations = len(all_combos)
    print(f"Total unique combinations in search space: {total_combinations:,}")
    
    if args.test:
        test_limit = 50
        print(f"\nRunning in TEST MODE: testing first {test_limit} combinations...")
        all_combos = all_combos[:test_limit]
    elif args.limit > 0:
        print(f"\nRunning in LIMITED SEARCH MODE: testing first {args.limit} combinations...")
        all_combos = all_combos[:args.limit]
    else:
        print("\nRunning in ENDLESS SEARCH MODE. Press Ctrl+C to stop.")
        
    pbar = tqdm(all_combos, desc="Searching MR Strategies")
    
    for idx, params in enumerate(pbar):
        try:
            # Run multi-symbol test
            results, agg = backtester.run_multi_symbol(precomputed_data, generate_mr_signal, params)
            
            # Check target limits
            wr = agg['win_rate']
            pf = agg['profit_factor']
            sharpe = agg['sharpe_ratio']
            total_tr = agg['total_trades']
            
            # Update progress bar description with current best
            pbar.set_postfix(WR=f"{wr}%", PF=pf, Sharpe=sharpe, Trades=total_tr)
            
            # "Comes close to" goal of 70% WR and 1.8 PF
            # We run walk-forward only on combinations that pass initial filters to stay fast
            if wr >= 60.0 and pf >= 1.45 and total_tr >= 30:
                wf_res = backtester.run_multi_symbol_walkforward(precomputed_data, generate_mr_signal, params)
                is_agg = wf_res['in_sample']['aggregate']
                oos_agg = wf_res['out_of_sample']['aggregate']
                
                oos_wr = oos_agg['win_rate']
                oos_pf = oos_agg['profit_factor']
                oos_tr = oos_agg['total_trades']
                
                # Walk-forward validation passes if OOS results don't completely break down
                if oos_pf >= 1.15 and oos_wr >= 50.0 and oos_tr >= 8:
                    is_perfect = (wr >= 70.0 and pf >= 1.8) and (oos_wr >= 65.0 and oos_pf >= 1.5)
                    status = "🏆 PERFECT WINNER (Passed WF)" if is_perfect else "⭐ CLOSE WINNER (Passed WF)"
                    
                    strategy_entry = {
                        'params': params,
                        'aggregate_metrics': agg,
                        'symbol_metrics': results,
                        'walkforward_metrics': wf_res,
                        'is_perfect': is_perfect
                    }
                    
                    # Check duplicates
                    if params not in [x['params'] for x in winning_strategies]:
                        winning_strategies.append(strategy_entry)
                        # Write to file immediately
                        with open(save_file, 'w') as f:
                            json.dump(winning_strategies, f, indent=2)
                            
                        print(f"\n[{status}] Found strategy! Full WR: {wr}%, Full PF: {pf} | OOS WR: {oos_wr}%, OOS PF: {oos_pf}, OOS Trades: {oos_tr}")
                        print(f"Params: {params}\n")
                    
        except KeyboardInterrupt:
            print("\nInterrupt received. Stopping search.")
            break
        except Exception as e:
            # Silently continue on errors during search to keep bot robust
            pass
            
    print("\n" + "="*80)
    print(f"MR strategy search finished. Total winning strategies saved: {len(winning_strategies)}")
    print(f"Results saved to {save_file}")
    print("="*80)

if __name__ == '__main__':
    main()
