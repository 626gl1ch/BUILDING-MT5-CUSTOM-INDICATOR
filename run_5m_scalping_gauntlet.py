import os
import sys
import json
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from rbo_v2 import dynamic_signal_generator

def main():
    print("=" * 80)
    print(" 5-MINUTE SCALPING STRATEGY EVALUATION & VALIDATION GAUNTLET")
    print("=" * 80)
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    print("\n[1/3] Loading 5-Minute Historical CSV Datasets...")
    data_5m = engine.load_all_data(suffix="5min_1year")
    
    print("\n[2/3] Precomputing 150+ Technical Indicators...")
    precomputed_5m = {}
    for sym, df in data_5m.items():
        print(f"  Computing indicators for {sym} (5m)...")
        precomputed_5m[sym] = add_all_indicators(df)
        
    print("\n[3/3] Executing 5m Scalping Strategy Validation Gauntlet...")
    
    # Define targeted 5m scalping strategies across categories requested
    strategies_to_test = [
        # 1. EMA Pullback Strategy
        {
            'name': 'EMA Pullback Scalper',
            'category': 'Trend Following',
            'params': {
                'entry': 'ssf_cross',
                'regime': 'adx_strong',
                'sl_atr': 1.5,
                'tp_atr': 3.0,
                'max_bars_hold': 36,
                'modifier': 'fast',
                'risk_pct': 0.01,
                'trailing': True
            },
            'description': 'Trades SuperSmoother Filter trend momentum when ADX > 25 indicates strong trend.'
        },
        # 2. Donchian Volatility Breakout
        {
            'name': 'Donchian Volatility Breakout',
            'category': 'Volatility Breakout',
            'params': {
                'entry': 'donchian_breakout',
                'regime': 'chop_trending',
                'sl_atr': 1.5,
                'tp_atr': 3.5,
                'max_bars_hold': 48,
                'modifier': 'fast',
                'risk_pct': 0.01,
                'trailing': True
            },
            'description': 'Enters when 20-period Donchian Channel High/Low is broken during low Choppiness.'
        },
        # 3. Mean Reversion (BB + VWAP Filter)
        {
            'name': 'Bollinger VWAP Mean Reversion',
            'category': 'Filtered Mean Reversion',
            'params': {
                'entry': 'bb_vwap_mr',
                'regime': 'chop_choppy',
                'sl_atr': 2.0,
                'tp_atr': 2.5,
                'max_bars_hold': 24,
                'modifier': 'slow',
                'risk_pct': 0.01,
                'trailing': False
            },
            'description': 'Buys when price dips below lower BB and VWAP in a high Choppiness ranging regime.'
        },
        # 4. KAMA Adaptive Breakout
        {
            'name': 'KAMA Adaptive Trend Scalper',
            'category': 'Adaptive Moving Average',
            'params': {
                'entry': 'frama_breakout',
                'regime': 'hurst_trending',
                'sl_atr': 1.5,
                'tp_atr': 4.0,
                'max_bars_hold': 48,
                'modifier': 'fast',
                'risk_pct': 0.01,
                'trailing': True
            },
            'description': 'Uses Fractal Adaptive MA (FRAMA) & Hurst Exponent > 0.55 for trend riding.'
        },
        # 5. MACD Momentum Crossover
        {
            'name': 'MACD Zero-Lag Crossover',
            'category': 'Momentum Crossover',
            'params': {
                'entry': 'macd_cross',
                'regime': 'none',
                'sl_atr': 2.0,
                'tp_atr': 3.0,
                'max_bars_hold': 36,
                'modifier': 'fast',
                'risk_pct': 0.01,
                'trailing': False
            },
            'description': 'Enters on MACD line crossing Signal line below/above zero level.'
        },
        # 6. Stochastic RSI Mean Reversion
        {
            'name': 'Stochastic RSI Extreme Reversion',
            'category': 'Oscillator Mean Reversion',
            'params': {
                'entry': 'stoch_cross',
                'regime': 'chop_choppy',
                'sl_atr': 1.5,
                'tp_atr': 2.5,
                'max_bars_hold': 24,
                'modifier': 'slow',
                'risk_pct': 0.01,
                'trailing': False
            },
            'description': 'Triggers when StochRSI %K crosses %D in oversold (<20) or overbought (>80) zone.'
        },
        # 7. Keltner Channel Pullback
        {
            'name': 'Keltner Channel Reversion',
            'category': 'Channel Reversion',
            'params': {
                'entry': 'keltner_pullback',
                'regime': 'none',
                'sl_atr': 1.5,
                'tp_atr': 3.0,
                'max_bars_hold': 36,
                'modifier': 'fast',
                'risk_pct': 0.01,
                'trailing': False
            },
            'description': 'Buys when price touches lower Keltner Band with RSI < 30.'
        },
        # 8. Z-Score Extreme Reversion
        {
            'name': 'Statistical Z-Score Reversion',
            'category': 'Statistical Arbitrage',
            'params': {
                'entry': 'zscore_extreme',
                'regime': 'hurst_mean_reverting',
                'sl_atr': 2.0,
                'tp_atr': 2.5,
                'max_bars_hold': 24,
                'modifier': 'slow',
                'risk_pct': 0.01,
                'trailing': False
            },
            'description': 'Enters when 20-period price Z-score exceeds +/- 2.5 std deviations.'
        }
    ]
    
    results_summary = []
    
    print("\n" + "=" * 80)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    for s_info in strategies_to_test:
        name = s_info['name']
        cat = s_info['category']
        params = s_info['params']
        desc = s_info['description']
        
        print(f"\n---> Testing Strategy: {name} [{cat}]")
        
        # Run 3-Layer Validation
        res = engine.run_full_validation(
            precomputed_5m, dynamic_signal_generator, params,
            min_trades_per_day=0.2, min_assets=2, n_permutations=200,
            slippage_pct=0.0002, fee_pct=0.0005
        )
        
        agg = res['backtest']
        wf = res['walkforward']
        perm = res['permutation']
        passed = res['passed']
        
        status_str = "[PASSED (GOD SCALPER)]" if passed else "[FAILED GAUNTLET]"
        print(f"  Status: {status_str}")
        print(f"  Standard Backtest: Expectancy={agg['expectancy']:.4f} | PF={agg['profit_factor']:.2f} | WR={agg['win_rate']:.1f}% | Sharpe={agg['sharpe_ratio']:.3f} | Trades/Day={agg['trades_per_day']:.1f}")
        print(f"  Walk-Forward OOS: Expectancy={wf['out_of_sample']['expectancy']:.4f} | OOS Sharpe={wf['out_of_sample']['sharpe_ratio']:.3f} | OOS PF={wf['out_of_sample']['profit_factor']:.2f}")
        print(f"  Permutation Test: p-value={perm['p_value']:.4f}")
        
        entry_data = {
            'name': name,
            'category': cat,
            'description': desc,
            'passed': passed,
            'params': params,
            'aggregate_metrics': agg,
            'walkforward_metrics': wf,
            'permutation_metrics': perm,
            'symbol_metrics': res['symbol_results']
        }
        results_summary.append(entry_data)
        
    # Save full JSON report
    with open("scalping_5m_validation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
        
    print("\n" + "=" * 80)
    print(" Full 5m Scalping Results saved to scalping_5m_validation_results.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
