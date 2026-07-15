import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from backtest_core import BacktestCore
from indicators_library import add_all_indicators
from rbo_v2 import (
    signal_ema_pullback, 
    signal_donchian_breakout, 
    signal_bb_vwap_mr, 
    signal_rsi_divergence, 
    signal_squeeze_breakout
)

def main():
    print("======================================================")
    print("  Checking Strategy Correlations")
    print("======================================================")

    engine = BacktestCore()
    all_data = engine.load_all_data()
    
    # We just need one symbol to check correlation, let's use BTC
    sym = 'BTCUSD_5M'
    if sym not in all_data:
        sym = list(all_data.keys())[0]
        
    print(f"\nComputing indicators for {sym}...")
    df = add_all_indicators(all_data[sym])
    
    # Drop NaNs to ensure clean signals
    df = df.dropna().reset_index(drop=True)

    # Define sensible placeholder parameters for the 5 strategies
    p_ema = {'fast_ema': 9, 'trend_ema': 50, 'adx_min': 20}
    p_donchian = {'period': 20, 'vol_min': 1.5}
    p_bbvwap = {'bb_period': 20, 'rsi_period': 14, 'rsi_os': 30, 'rsi_ob': 70}
    p_rsi_div = {'rsi_period': 14, 'zscore_thresh': 2.0, 'lookback': 10}
    p_squeeze = {'period': 20, 'vol_min': 1.5, 'squeeze_lb': 10}

    print("\nGenerating Signals...")
    signals_ema = signal_ema_pullback(df, p_ema)
    signals_donchian = signal_donchian_breakout(df, p_donchian)
    signals_bbvwap = signal_bb_vwap_mr(df, p_bbvwap)
    signals_rsidiv = signal_rsi_divergence(df, p_rsi_div)
    signals_squeeze = signal_squeeze_breakout(df, p_squeeze)

    # Create a dataframe of the signals
    sig_df = pd.DataFrame({
        'EMA_Pullback': signals_ema,
        'Donchian_Breakout': signals_donchian,
        'BB_VWAP_MR': signals_bbvwap,
        'RSI_Divergence': signals_rsidiv,
        'Vol_Squeeze': signals_squeeze
    })

    print("\nCalculating Signal Correlation Matrix (Pearson)...")
    corr_matrix = sig_df.corr()
    print("\n", corr_matrix)
    
    # Check if Donchian and Squeeze are highly correlated
    donchian_squeeze_corr = corr_matrix.loc['Donchian_Breakout', 'Vol_Squeeze']
    print(f"\nCorrelation between Donchian Breakout and Vol Squeeze: {donchian_squeeze_corr:.3f}")
    
    if donchian_squeeze_corr > 0.6:
        print("  -> WARNING: High correlation detected! Recommend merging or dropping one.")
    else:
        print("  -> Safe: Correlation is low enough to keep both.")

if __name__ == "__main__":
    main()
