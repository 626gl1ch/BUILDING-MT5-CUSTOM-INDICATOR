import os
import glob
import pandas as pd
import numpy as np
import time

from backtest_core import BacktestCore
from indicators_library import (
    calc_alma, calc_stoch_rsi, calc_choppiness_index, calc_cmf, calc_adx,
    calc_vwap, calc_kama, calc_rsi, calc_supertrend, calc_bollinger_bands,
    calc_mfi, calc_ema, calc_atr, calc_cvd, calc_bb_width_percentile
)

# Helper function for universal gates
def get_universal_gates(df):
    chop = calc_choppiness_index(df, period=14)
    atr = calc_atr(df, period=14)
    atr_sma = atr.rolling(window=50).mean()
    bb_pct = calc_bb_width_percentile(df, period=20, std=2.0, lookback=100)
    
    # RVOL = volume / volume_ma_20
    rvol = df['volume_ratio'] if 'volume_ratio' in df.columns else df['volume'] / df['volume'].rolling(20).mean()
    cvd = calc_cvd(df, period=5)
    
    # Universal regime gates
    # chop < 38.2
    # atr >= 0.9 * atr_sma
    regime_ok = (chop < 38.2) & (atr >= 0.9 * atr_sma)
    
    return chop, atr, bb_pct, rvol, cvd, regime_ok

# ==========================================
# 4 CORE STRATEGIES
# ==========================================

def strat_core_1(df, p):
    # KAMA Trend / Supertrend Confluence
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    
    kama = calc_kama(df['close'], period=10, fast=2, slow=30)
    st_val, st_dir = calc_supertrend(df, period=10, multiplier=3.0)
    stoch_k, stoch_d = calc_stoch_rsi(df['close'], period=14, k=3, d=3)
    
    # Trend bias holds for >= 3 candles
    bull_cond = (df['close'] > kama) & (st_dir == 1)
    bear_cond = (df['close'] < kama) & (st_dir == -1)
    
    uptrend = bull_cond.rolling(3).sum() >= 3
    downtrend = bear_cond.rolling(3).sum() >= 3
    
    # Gates
    regime = regime_ok & (bb_pct > 30)
    vol_bull = (cvd > cvd.shift(1)) & (rvol >= 1.0)
    vol_bear = (cvd < cvd.shift(1)) & (rvol >= 1.0)
    
    # OB/OS Timing
    # Dipped <= 20 within last 5 candles and now crossed back above 20
    stoch_was_os = (stoch_k.shift(1).rolling(5).min() <= 20)
    stoch_crossed_up = (stoch_k > 20) & (stoch_k.shift(1) <= 20)
    long_timing = stoch_was_os & stoch_crossed_up
    
    stoch_was_ob = (stoch_k.shift(1).rolling(5).max() >= 80)
    stoch_crossed_down = (stoch_k < 80) & (stoch_k.shift(1) >= 80)
    short_timing = stoch_was_ob & stoch_crossed_down
    
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime & vol_bull & long_timing] = 1
    signals[downtrend & regime & vol_bear & short_timing] = -1
    return signals

def strat_core_2(df, p):
    # ALMA Dual Cross / VWAP Confluence
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    
    alma9 = calc_alma(df['close'], period=9)
    alma21 = calc_alma(df['close'], period=21)
    vwap = calc_vwap(df)
    adx, _, _ = calc_adx(df, period=14)
    
    bull_cond = (alma9 > alma21) & (df['close'] > vwap)
    bear_cond = (alma9 < alma21) & (df['close'] < vwap)
    uptrend = bull_cond.rolling(2).sum() >= 2
    downtrend = bear_cond.rolling(2).sum() >= 2
    
    regime = regime_ok & (adx >= 20)
    vol_bull = (cvd > cvd.shift(1)) & (rvol >= 1.1)
    vol_bear = (cvd < cvd.shift(1)) & (rvol >= 1.1)
    
    # Pullback to VWAP or ALMA9 without chop > 55
    # Simplification: Low touches ALMA9 or VWAP
    target_long = pd.concat([vwap, alma9], axis=1).max(axis=1)
    target_short = pd.concat([vwap, alma9], axis=1).min(axis=1)
    
    long_timing = (df['low'] <= target_long) & (df['close'] > target_long) & (chop < 55)
    short_timing = (df['high'] >= target_short) & (df['close'] < target_short) & (chop < 55)
    
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime & vol_bull & long_timing] = 1
    signals[downtrend & regime & vol_bear & short_timing] = -1
    return signals

def strat_core_3(df, p):
    # Supertrend / Bollinger Squeeze-Break
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    
    st_val, st_dir = calc_supertrend(df, period=10, multiplier=3.0)
    bb_upper, _, bb_lower, _, _ = calc_bollinger_bands(df['close'], period=20, std=2.0)
    mfi = calc_mfi(df, period=14)
    
    uptrend = (st_dir == 1).rolling(3).sum() >= 3
    downtrend = (st_dir == -1).rolling(3).sum() >= 3
    
    # Squeeze condition
    had_squeeze = (bb_pct.shift(1).rolling(20).min() <= 20)
    expanding = (bb_pct > bb_pct.shift(3))
    regime = regime_ok & had_squeeze & expanding
    
    vol_bull = (cvd > cvd.shift(1)) & (rvol >= 1.3)
    vol_bear = (cvd < cvd.shift(1)) & (rvol >= 1.3)
    
    long_timing = (mfi.shift(1) < 30) & (mfi >= 30) & (df['close'] > bb_upper)
    short_timing = (mfi.shift(1) > 70) & (mfi <= 70) & (df['close'] < bb_lower)
    
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime & vol_bull & long_timing] = 1
    signals[downtrend & regime & vol_bear & short_timing] = -1
    return signals

def strat_core_4(df, p):
    # EMA Cross / VWAP Pullback
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    
    ema9 = calc_ema(df['close'], period=9)
    ema21 = calc_ema(df['close'], period=21)
    vwap = calc_vwap(df)
    adx, _, _ = calc_adx(df, period=14)
    rsi = calc_rsi(df['close'], period=10)
    
    bull_cond = (ema9 > ema21) & (df['close'] > vwap)
    bear_cond = (ema9 < ema21) & (df['close'] < vwap)
    uptrend = bull_cond.rolling(3).sum() >= 3
    downtrend = bear_cond.rolling(3).sum() >= 3
    
    regime = (chop < 35) & (atr >= 0.9 * atr.rolling(50).mean()) & (adx >= 22)
    vol_bull = (cvd > cvd.shift(1)) & (rvol >= 1.1)
    vol_bear = (cvd < cvd.shift(1)) & (rvol >= 1.1)
    
    # RSI 40-50 zone and price touches VWAP or EMA21
    long_timing = (rsi >= 40) & (rsi <= 50) & ((df['low'] <= vwap) | (df['low'] <= ema21)) & (df['close'] > vwap) & (df['close'] > ema21)
    short_timing = (rsi >= 50) & (rsi <= 60) & ((df['high'] >= vwap) | (df['high'] >= ema21)) & (df['close'] < vwap) & (df['close'] < ema21)
    
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime & vol_bull & long_timing] = 1
    signals[downtrend & regime & vol_bear & short_timing] = -1
    return signals

# ==========================================
# 8 PULLBACK VARIANTS
# ==========================================

def strat_p1(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    ema9, ema21 = calc_ema(df['close'], 9), calc_ema(df['close'], 21)
    stoch_k, stoch_d = calc_stoch_rsi(df['close'], 14, 3, 3)
    uptrend = (ema9 > ema21).rolling(3).sum() >= 3
    downtrend = (ema9 < ema21).rolling(3).sum() >= 3
    vol = (rvol >= 1.0) & (cvd > cvd.shift(1))
    vol_s = (rvol >= 1.0) & (cvd < cvd.shift(1))
    lt = (df['low'] <= ema21) & (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_d < 20)
    st = (df['high'] >= ema21) & (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_d > 80)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol_s & st] = -1
    return signals

def strat_p2(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    kama = calc_kama(df['close'], 10)
    rsi = calc_rsi(df['close'], 10)
    uptrend = (df['close'] > kama).rolling(3).sum() >= 3
    downtrend = (df['close'] < kama).rolling(3).sum() >= 3
    vol = (cvd > cvd.shift(1))
    vol_s = (cvd < cvd.shift(1))
    lt = (df['low'] <= kama) & (rsi >= 40) & (rsi <= 50) & (df['close'] > kama)
    st = (df['high'] >= kama) & (rsi >= 50) & (rsi <= 60) & (df['close'] < kama)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol_s & st] = -1
    return signals

def strat_p3(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    alma9, alma21 = calc_alma(df['close'], 9), calc_alma(df['close'], 21)
    stoch_k, stoch_d = calc_stoch_rsi(df['close'], 14, 3, 3)
    uptrend = (alma9 > alma21).rolling(3).sum() >= 3
    downtrend = (alma9 < alma21).rolling(3).sum() >= 3
    vol = (rvol >= 1.1)
    lt = (df['low'] <= alma9) & (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_d < 20)
    st = (df['high'] >= alma9) & (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_d > 80)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol & st] = -1
    return signals

def strat_p4(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    st_val, st_dir = calc_supertrend(df, 10, 3.0)
    mfi = calc_mfi(df, 14)
    uptrend = (st_dir == 1).rolling(3).sum() >= 3
    downtrend = (st_dir == -1).rolling(3).sum() >= 3
    vol = (rvol >= 1.0) & (cvd > cvd.shift(1))
    vol_s = (rvol >= 1.0) & (cvd < cvd.shift(1))
    lt = (df['low'] <= st_val) & (mfi >= 40) & (mfi <= 60) & (df['close'] > st_val)
    st = (df['high'] >= st_val) & (mfi >= 40) & (mfi <= 60) & (df['close'] < st_val)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol_s & st] = -1
    return signals

def strat_p5(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    vwap = calc_vwap(df)
    stoch_k, stoch_d = calc_stoch_rsi(df['close'], 14, 3, 3)
    uptrend = (df['close'] > vwap).rolling(3).sum() >= 3
    downtrend = (df['close'] < vwap).rolling(3).sum() >= 3
    vol = (cvd > cvd.shift(1))
    vol_s = (cvd < cvd.shift(1))
    lt = (df['low'] <= vwap) & (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_d < 20)
    st = (df['high'] >= vwap) & (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_d > 80)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol_s & st] = -1
    return signals

def strat_p6(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    ema9, ema21 = calc_ema(df['close'], 9), calc_ema(df['close'], 21)
    bb_u, _, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.0)
    rsi = calc_rsi(df['close'], 10)
    uptrend = (ema9 > ema21).rolling(3).sum() >= 3
    downtrend = (ema9 < ema21).rolling(3).sum() >= 3
    vol = (rvol >= 1.0)
    lt = (df['low'] <= bb_l) & (rsi > 40) & (rsi.shift(1) <= 40)
    st = (df['high'] >= bb_u) & (rsi < 60) & (rsi.shift(1) >= 60)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol & st] = -1
    return signals

def strat_p7(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    kama = calc_kama(df['close'], 10)
    vwap = calc_vwap(df)
    stoch_k, stoch_d = calc_stoch_rsi(df['close'], 14, 3, 3)
    uptrend = (df['close'] > kama).rolling(3).sum() >= 3
    downtrend = (df['close'] < kama).rolling(3).sum() >= 3
    vol = (rvol >= 1.1) & (cvd > cvd.shift(1))
    vol_s = (rvol >= 1.1) & (cvd < cvd.shift(1))
    lt = (df['low'] <= vwap) & (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_d < 20)
    st = (df['high'] >= vwap) & (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_d > 80)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol_s & st] = -1
    return signals

def strat_p8(df, p):
    chop, atr, bb_pct, rvol, cvd, regime_ok = get_universal_gates(df)
    st_val, st_dir = calc_supertrend(df, 10, 3.0)
    ema21 = calc_ema(df['close'], 21)
    rsi = calc_rsi(df['close'], 10)
    uptrend = (st_dir == 1).rolling(3).sum() >= 3
    downtrend = (st_dir == -1).rolling(3).sum() >= 3
    vol = (rvol >= 1.0)
    lt = (df['low'] <= ema21) & (rsi >= 40) & (rsi <= 50) & (df['close'] > ema21)
    st = (df['high'] >= ema21) & (rsi >= 50) & (rsi <= 60) & (df['close'] < ema21)
    signals = pd.Series(0, index=df.index)
    signals[uptrend & regime_ok & vol & lt] = 1
    signals[downtrend & regime_ok & vol & st] = -1
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
    dfs = load_data("5min_1year")
    if not dfs:
        print("No 5m CSV files found.")
        return

    strategies = {
        "Core_1_KAMA": strat_core_1,
        "Core_2_ALMA": strat_core_2,
        "Core_3_Supertrend_BB": strat_core_3,
        "Core_4_EMA_VWAP": strat_core_4,
        "P1_EMA_StochRSI": strat_p1,
        "P2_KAMA_RSI": strat_p2,
        "P3_ALMA_StochRSI": strat_p3,
        "P4_Supertrend_MFI": strat_p4,
        "P5_VWAP_StochRSI": strat_p5,
        "P6_EMA_BB_RSI": strat_p6,
        "P7_KAMA_VWAP_StochRSI": strat_p7,
        "P8_Supertrend_EMA21_RSI": strat_p8
    }
    
    # Using 1.5 ATR hard stop as requested, trailing for runners
    params = {
        'sl_atr': 1.5, 
        'tp_atr': 100.0, # Massive TP so trailing stop takes out the trade
        'max_bars_hold': 100, 
        'risk_pct': 0.01,
        'trailing': True
    }
    fee_pct = 0.0002

    engine = BacktestCore()
    results_list = []

    for name, fn in strategies.items():
        print(f"Evaluating {name}...")
        _, agg, _ = engine.run_multi_symbol(dfs, fn, params, slippage_pct=0, fee_pct=fee_pct)
        if agg['total_trades'] > 10:
            results_list.append({
                'Strategy': name,
                'Profit Factor': agg['profit_factor'],
                'Win Rate (%)': agg['win_rate'],
                'Sharpe': agg['sharpe_ratio'],
                'Trades': agg['total_trades']
            })

    # Convert to DataFrame to sort
    res_df = pd.DataFrame(results_list)
    if not res_df.empty:
        # Rank by PF, then WR, then Sharpe
        res_df = res_df.sort_values(by=['Profit Factor', 'Win Rate (%)', 'Sharpe'], ascending=False)
        print("\n=== FINAL RANKINGS ===")
        print(res_df.to_string(index=False))
        
        # Save to markdown
        with open("advanced_ranking.md", "w") as f:
            f.write("# Advanced 5m Trend Suite Rankings\n\n")
            f.write(res_df.to_markdown(index=False))
            
if __name__ == "__main__":
    main()
