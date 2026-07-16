import pandas as pd
import numpy as np
import os
import glob
from backtest_core import BacktestCore
from rbo_v2 import signal_ema_pullback, signal_donchian_breakout
from logger_config import logger

def load_1h_data():
    """Loads all 1H data files"""
    symbols = ["BTCUSDT", "LTCUSDT", "SOLUSDT", "TRUMPUSDT"]
    data = {}
    for symbol in symbols:
        filepath = f"{symbol}_1H_1year.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            data[symbol] = df
    return data

def run_cost_stress_test(base_params: dict, strategy_fn, all_dfs: dict, 
                          base_slippage_pct=0.0002, base_fee_pct=0.0005,
                          multipliers=(1.0, 1.5, 2.0, 3.0)):
    """
    Runs the same strategy across different cost multipliers to see degradation curve.
    """
    engine = BacktestCore(commission=0) # Commission bypassed, handled by dynamic fees
    results = []
    
    logger.info(f"--- Running Cost-Stress Test for {strategy_fn.__name__} ---")
    
    for mult in multipliers:
        stressed_slippage = base_slippage_pct * mult
        stressed_fee = base_fee_pct * mult
        
        logger.info(f"Testing at {mult}x Cost (Slippage: {stressed_slippage*10000:.1f}bps, Fee: {stressed_fee*10000:.1f}bps)")
        
        res = engine.run_full_validation(
            all_dfs, strategy_fn, base_params,
            min_trades_per_day=0.5, # 1H data has fewer trades
            min_assets=2,
            n_permutations=200,
            slippage_pct=stressed_slippage,
            fee_pct=stressed_fee
        )
        
        # Extract metrics
        bt = res['backtest']
        results.append({
            "Cost Multiplier": mult,
            "Sharpe": bt.get('sharpe_ratio', 0),
            "Win Rate": bt.get('win_rate', 0),
            "Expectancy": bt.get('expectancy', 0),
            "Total Trades": bt.get('total_trades', 0),
            "Perm P-Value": res['permutation'].get('p_value', 1.0) if 'permutation' in res else 1.0,
            "Passed All Gates": res['passed']
        })
        
    df_res = pd.DataFrame(results)
    logger.info("Cost-Stress Test Results:\n" + df_res.to_string(index=False))
    return df_res

def run_sensitivity_sweep(base_params: dict, strategy_fn, all_dfs: dict, 
                          slippage_pct=0.0002, fee_pct=0.0005, shifts=(-0.2, -0.1, 0.0, 0.1, 0.2)):
    """
    Alters parameters by percentages to see if the edge sits on a plateau or a knife-edge.
    """
    engine = BacktestCore(commission=0)
    results = []
    
    logger.info(f"--- Running Parameter Sensitivity Sweep for {strategy_fn.__name__} ---")
    
    for shift in shifts:
        # Create shifted params
        p_shifted = {}
        for k, v in base_params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                new_v = v * (1.0 + shift)
                if isinstance(v, int):
                    new_v = int(round(new_v))
                    # Prevent going to 0 for periods
                    if new_v <= 0: new_v = 1
                p_shifted[k] = new_v
            else:
                p_shifted[k] = v
                
        logger.info(f"Testing Shift {shift*100:+.0f}%")
        
        res = engine.run_full_validation(
            all_dfs, strategy_fn, p_shifted,
            min_trades_per_day=0.5, 
            min_assets=2,
            n_permutations=50, # Lower permutations for speed during sweep
            slippage_pct=slippage_pct,
            fee_pct=fee_pct
        )
        
        bt = res['backtest']
        results.append({
            "Shift %": f"{shift*100:+.0f}%",
            "Sharpe": bt.get('sharpe_ratio', 0),
            "Win Rate": bt.get('win_rate', 0),
            "Total Trades": bt.get('total_trades', 0),
            "Passed All Gates": res['passed']
        })
        
    df_res = pd.DataFrame(results)
    logger.info("Sensitivity Sweep Results:\n" + df_res.to_string(index=False))
    return df_res


if __name__ == "__main__":
    logger.info("Loading 1H Data...")
    dfs = load_1h_data()
    
    # We will test a dummy or known parameter set for EMA Pullback.
    # Note: On 1H data, parameter ranges are different. A 50-period EMA on 5m is 250 mins (~4 hours).
    # We'll just test a theoretical parameter set.
    sample_params = {
        'fast_ema': 10,
        'slow_ema': 30,
        'trend_ema': 50,
        'trend_adx': 20,
        'pullback_len': 3,
        'sl_atr': 2.0,
        'tp_atr': 4.0,
        'max_bars_hold': 48
    }
    
    if len(dfs) > 0:
        run_cost_stress_test(sample_params, signal_ema_pullback, dfs)
        run_sensitivity_sweep(sample_params, signal_ema_pullback, dfs)
    else:
        logger.warning("No 1H data found. Please run data_resampler.py first.")
