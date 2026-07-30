import os
import glob
import pandas as pd
import numpy as np
import time

from backtest_core import BacktestCore
from indicators_library import (
    calc_alma, calc_stoch_rsi, calc_choppiness_index, calc_cmf, calc_adx,
    calc_vwap, calc_kama, calc_rsi, calc_supertrend, calc_bollinger_bands,
    calc_mfi, calc_ema, calc_atr
)

# ==========================================
# 5M TREND FOLLOWING PULLBACK STRATEGIES
# ==========================================

def strat_alma_stochrsi(df, p):
    # Indicators
    alma_fast = calc_alma(df['close'], period=3)
    alma_slow = calc_alma(df['close'], period=9)
    stoch_k, stoch_d = calc_stoch_rsi(df['close'], period=14, k=3, d=3)
    chop = calc_choppiness_index(df, period=14)
    cmf = calc_cmf(df, period=20)
    adx, _, _ = calc_adx(df, period=14)

    # Bias
    uptrend = (alma_fast > alma_slow)
    downtrend = (alma_fast < alma_slow)

    # Filters
    valid_market = (chop < 50) & (adx > 25)
    vol_bullish = (cmf > 0)
    vol_bearish = (cmf < 0)

    # Pullback (Oversold/Overbought)
    # StochRSI crosses back up from oversold (<20)
    pullback_long = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_d < 20)
    pullback_short = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_d > 80)

    # Signals
    signals = pd.Series(0, index=df.index)
    signals[uptrend & valid_market & vol_bullish & pullback_long] = 1
    signals[downtrend & valid_market & vol_bearish & pullback_short] = -1

    return signals

def strat_vwap_kama_rsi(df, p):
    # Indicators
    vwap = calc_vwap(df)
    kama = calc_kama(df['close'], period=10)
    rsi = calc_rsi(df['close'], period=14)
    chop = calc_choppiness_index(df, period=14)

    # Bias
    uptrend = (df['close'] > vwap) & (df['close'] > kama)
    downtrend = (df['close'] < vwap) & (df['close'] < kama)

    # Filters
    vwap_slope_up = (vwap > vwap.shift(1))
    vwap_slope_down = (vwap < vwap.shift(1))
    valid_market = (chop < 50)

    # Pullback
    # RSI drops below 40 for long, goes above 60 for short
    pullback_long = (rsi < 40)
    pullback_short = (rsi > 60)

    # Rejection (candle closes in direction of trend after pullback)
    reject_long = (df['close'] > df['open'])
    reject_short = (df['close'] < df['open'])

    # Signals
    signals = pd.Series(0, index=df.index)
    signals[uptrend & vwap_slope_up & valid_market & pullback_long & reject_long] = 1
    signals[downtrend & vwap_slope_down & valid_market & pullback_short & reject_short] = -1

    return signals

def strat_supertrend_bb(df, p):
    # Indicators
    st_val, st_dir = calc_supertrend(df, period=10, multiplier=3.0)
    bb_upper, bb_middle, bb_lower, _, _ = calc_bollinger_bands(df['close'], period=20, std=2.0)
    mfi = calc_mfi(df, period=14)
    adx, _, _ = calc_adx(df, period=14)

    # Bias
    uptrend = (st_dir == 1)
    downtrend = (st_dir == -1)

    # Filters
    valid_market = (adx > 25)

    # Pullback + Volume
    pullback_long = (df['low'] <= bb_lower) & (mfi < 30)
    pullback_short = (df['high'] >= bb_upper) & (mfi > 70)
    
    # Rejection
    reject_long = (df['close'] > bb_lower)
    reject_short = (df['close'] < bb_upper)

    # Signals
    signals = pd.Series(0, index=df.index)
    signals[uptrend & valid_market & pullback_long & reject_long] = 1
    signals[downtrend & valid_market & pullback_short & reject_short] = -1

    return signals

def strat_ema_pullback(df, p):
    # Indicators
    ema_50 = calc_ema(df['close'], period=50)
    ema_200 = calc_ema(df['close'], period=200)
    chop = calc_choppiness_index(df, period=14)
    atr = calc_atr(df, period=14)
    atr_sma = atr.rolling(window=14).mean()
    rsi = calc_rsi(df['close'], period=14)

    # Bias
    uptrend = (ema_50 > ema_200) & (df['close'] > ema_200)
    downtrend = (ema_50 < ema_200) & (df['close'] < ema_200)

    # Filters
    valid_market = (chop < 50) & (atr > atr_sma)

    # Pullback
    # Price touches or goes below EMA 50 during uptrend
    pullback_long = (df['low'] <= ema_50) & (rsi < 40)
    pullback_short = (df['high'] >= ema_50) & (rsi > 60)
    
    # Rejection
    reject_long = (df['close'] > ema_50)
    reject_short = (df['close'] < ema_50)

    signals = pd.Series(0, index=df.index)
    signals[uptrend & valid_market & pullback_long & reject_long] = 1
    signals[downtrend & valid_market & pullback_short & reject_short] = -1

    return signals

def load_data(timeframe="5min_1year"):
    csv_files = glob.glob(f"*_{timeframe}.csv")
    dfs = {}
    for f in csv_files:
        symbol = f.split('_')[0]
        try:
            df = pd.read_csv(f)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            dfs[symbol] = df
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return dfs

def main():
    print("Loading 5-Minute Datasets...")
    dfs = load_data("5min_1year")
    if not dfs:
        print("No 5m CSV files found.")
        return

    print(f"Loaded {len(dfs)} symbols: {list(dfs.keys())}")

    strategies = {
        "ALMA_StochRSI": strat_alma_stochrsi,
        "VWAP_KAMA_RSI": strat_vwap_kama_rsi,
        "Supertrend_BB": strat_supertrend_bb,
        "EMA_Pullback": strat_ema_pullback
    }
    
    # Widen stops to survive 5m noise
    params = {
        'sl_atr': 2.0, 
        'tp_atr': 4.0, 
        'max_bars_hold': 100, 
        'risk_pct': 0.01,
        'trailing': True
    }

    # Low friction Forex fees (0.02% round trip)
    fee_pct = 0.0002
    slippage_pct = 0.0000 

    engine = BacktestCore()

    print("\n=======================================================")
    print("  5-MINUTE TREND FOLLOWING PULLBACK GAUNTLET")
    print(f"  FEE: {fee_pct*100}% | TRAILING: {params['trailing']}")
    print("=======================================================\n")

    for name, fn in strategies.items():
        print(f"--- Testing {name} ---")
        
        # 1. Standard Backtest
        print("  Running Standard Backtest...")
        _, agg, _ = engine.run_multi_symbol(dfs, fn, params, slippage_pct=slippage_pct, fee_pct=fee_pct)
        print(f"    Standard PF: {agg['profit_factor']} | WR: {agg['win_rate']}% | Sharpe: {agg['sharpe_ratio']} | Trades: {agg['total_trades']}")
        
        if agg['profit_factor'] < 1.0 or agg['total_trades'] < 20:
            print("    [FAILED] Standard backtest unprofitable or too few trades.\n")
            continue
            
        # 2. Walk-Forward Test
        print("  Running Walk-Forward Test...")
        wf_results = engine.run_walkforward(dfs, fn, params, split_pct=0.70, slippage_pct=slippage_pct, fee_pct=fee_pct)
        if wf_results:
            is_pf = wf_results['in_sample']['profit_factor']
            oos_pf = wf_results['out_of_sample']['profit_factor']
            print(f"    In-Sample PF: {is_pf} | Out-Of-Sample PF: {oos_pf}")
            if oos_pf > 1.0:
                print("    [PASSED] Strategy survived Walk-Forward!\n")
            else:
                print("    [FAILED] Strategy overfit (OOS PF < 1.0).\n")
        else:
            print("    [FAILED] Walk-forward error.\n")

if __name__ == "__main__":
    main()
