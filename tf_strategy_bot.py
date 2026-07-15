"""
Trend Following Strategy Search Bot
Endlessly pairs indicators to find a winning trend following strategy.
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

def generate_tf_signal(df, params):
    """
    Generates trend following buy/sell signals based on params.
    Ensures every strategy has Trend Direction, Strength, Choppiness Index (low),
    Volatility, and OB/OS filter.
    """
    # Extract params
    trend_ind = params['trend_indicator']     # e.g. 'ema_20', 'macd_hist', 'lr_slope_14'
    strength_ind = params['strength_indicator'] # e.g. 'adx_14', 'vi_plus_14', 'aroon_osc_25'
    chop_ind = params['chop_indicator']       # e.g. 'chop_14', 'chop_7'
    vol_ind = params['vol_indicator']         # e.g. 'natr_14', 'bb_width_20'
    ob_os_ind = params['ob_os_indicator']     # e.g. 'rsi_14', 'stoch_k_14', 'cci_14'
    
    chop_max = params['chop_max']
    vol_min = params['vol_min']
    trend_filter_val = params['trend_filter_val']
    
    close = df['close']
    
    # 1. Trend Direction
    if 'ema' in trend_ind:
        # EMA alignment
        long_trend = df['close'] > df[trend_ind]
        short_trend = df['close'] < df[trend_ind]
    elif 'macd' in trend_ind:
        # MACD histogram positive/negative
        long_trend = df[trend_ind] > 0
        short_trend = df[trend_ind] < 0
    elif 'slope' in trend_ind:
        long_trend = df[trend_ind] > 0
        short_trend = df[trend_ind] < 0
    else:
        long_trend = df['close'] > df['sma_50']
        short_trend = df['close'] < df['sma_50']
        
    # 2. Trend Strength
    if 'adx' in strength_ind:
        strength_cond = df[strength_ind] > trend_filter_val
    elif 'vi' in strength_ind:
        # Vortex VI+ > VI- for longs
        strength_cond = df[strength_ind] > 1.0
    elif 'aroon' in strength_ind:
        strength_cond = df[strength_ind].abs() > trend_filter_val
    else:
        strength_cond = pd.Series(True, index=df.index)
        
    # 3. Choppiness Index filter (ensure low choppiness / trending market)
    chop_cond = df[chop_ind] < chop_max
    
    # 4. Volatility filter (ensure enough volatility to move)
    vol_cond = df[vol_ind] > vol_min
    
    # 5. OB/OS Pullback or Overbought filter
    # For longs, we don't want to buy if extremely overbought (e.g. rsi > 70)
    # For shorts, we don't want to short if extremely oversold (e.g. rsi < 30)
    if 'rsi' in ob_os_ind:
        ob_os_long = df[ob_os_ind] < 70
        ob_os_short = df[ob_os_ind] > 30
    elif 'stoch' in ob_os_ind:
        ob_os_long = df[ob_os_ind] < 80
        ob_os_short = df[ob_os_ind] > 20
    elif 'williams' in ob_os_ind:
        ob_os_long = df[ob_os_ind] < -20
        ob_os_short = df[ob_os_ind] > -80
    else:
        ob_os_long = pd.Series(True, index=df.index)
        ob_os_short = pd.Series(True, index=df.index)
        
    # Combine signals
    long_signals = long_trend & strength_cond & chop_cond & vol_cond & ob_os_long
    short_signals = short_trend & strength_cond & chop_cond & vol_cond & ob_os_short
    
    signals = pd.Series(0, index=df.index)
    signals[long_signals] = 1
    signals[short_signals] = -1
    
    return signals

def main():
    parser = argparse.ArgumentParser(description="Trend Following Strategy Bot")
    parser.add_argument('--test', action='store_true', help="Run 50 combinations in test mode")
    parser.add_argument('--limit', type=int, default=0, help="Limit the number of combinations to run (0 for unlimited)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STARTING TREND FOLLOWING STRATEGY SEARCH BOT")
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
    trend_indicators = ['ema_20', 'ema_50', 'sma_50', 'sma_200', 'macd_hist', 'lr_slope_14', 'lr_slope_50']
    strength_indicators = ['adx_14', 'adx_21', 'vi_plus_14', 'aroon_osc_25']
    chop_indicators = ['chop_7', 'chop_14', 'chop_21']
    vol_indicators = ['natr_14', 'bb_width_20']
    ob_os_indicators = ['rsi_14', 'stoch_k_14', 'williams_14']
    
    # Define parameter ranges
    chop_max_vals = [40, 42, 45, 48]
    vol_min_vals = {
        'natr_14': [0.0005, 0.001, 0.0015],
        'bb_width_20': [0.001, 0.002, 0.003]
    }
    trend_filter_vals = {
        'adx_14': [20, 25, 30], 'adx_21': [20, 25, 30],
        'vi_plus_14': [1.0], 'aroon_osc_25': [30, 50, 70]
    }
    
    atr_sl_vals = [2.0, 2.5, 3.0]
    atr_tp_vals = [3.0, 4.5, 6.0]
    max_hold_vals = [48, 72, 96, 120]
    
    # Save file
    save_file = 'tf_winning_strategies.json'
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
    for trend in trend_indicators:
        for strength in strength_indicators:
            for chop in chop_indicators:
                for vol in vol_indicators:
                    for ob_os in ob_os_indicators:
                        for ch_max in chop_max_vals:
                            for vl_v in vol_min_vals[vol]:
                                for tf_val in trend_filter_vals[strength]:
                                    for sl in atr_sl_vals:
                                        for tp in atr_tp_vals:
                                            for hold in max_hold_vals:
                                                all_combos.append({
                                                    'trend_indicator': trend,
                                                    'strength_indicator': strength,
                                                    'chop_indicator': chop,
                                                    'vol_indicator': vol,
                                                    'ob_os_indicator': ob_os,
                                                    'chop_max': ch_max,
                                                    'vol_min': vl_v,
                                                    'trend_filter_val': tf_val,
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
        
    pbar = tqdm(all_combos, desc="Searching TF Strategies")
    
    for idx, params in enumerate(pbar):
        try:
            # Run multi-symbol test
            results, agg = backtester.run_multi_symbol(precomputed_data, generate_tf_signal, params)
            
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
                wf_res = backtester.run_multi_symbol_walkforward(precomputed_data, generate_tf_signal, params)
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
    print(f"TF strategy search finished. Total winning strategies saved: {len(winning_strategies)}")
    print(f"Results saved to {save_file}")
    print("="*80)

if __name__ == '__main__':
    main()
