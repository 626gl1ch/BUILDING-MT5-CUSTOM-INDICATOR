import os
import re

with open("indicators_library.py", "r") as f:
    content = f.read()

advanced_funcs = """
# ==========================================
# CATEGORY 12: ADVANCED SMC V2 INDICATORS
# ==========================================

def calc_vwap_stdev(df):
    '''
    Calculates VWAP and its Standard Deviation.
    '''
    tp = (df['high'] + df['low'] + df['close']) / 3
    volume = df['volume']
    
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df['datetime']).dt.date
        
    cum_pv = (tp * volume).groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    
    diff = tp - vwap
    diff_sq = diff ** 2
    cum_vol_diff_sq = (diff_sq * volume).groupby(dates).cumsum()
    variance = cum_vol_diff_sq / cum_vol.replace(0, np.nan)
    stdev = np.sqrt(variance)
    
    return vwap, stdev

def calc_order_blocks(df, lookback=20):
    '''
    Detects true unmitigated Order Blocks.
    '''
    atr = calc_atr(df, 14)
    bullish_disp = df['close'] > df['close'].shift(1) + 1.0 * atr.shift(1)
    bearish_disp = df['close'] < df['close'].shift(1) - 1.0 * atr.shift(1)
    
    is_bull_ob = bullish_disp.shift(-1).fillna(False)
    is_bear_ob = bearish_disp.shift(-1).fillna(False)
    
    import numpy as np
    import pandas as pd
    
    ob_bull_top = np.where(is_bull_ob, df['high'], np.nan)
    ob_bull_bottom = np.where(is_bull_ob, df['low'], np.nan)
    
    ob_bear_bottom = np.where(is_bear_ob, df['low'], np.nan)
    ob_bear_top = np.where(is_bear_ob, df['high'], np.nan)
    
    ob_bull_top_s = pd.Series(ob_bull_top, index=df.index).ffill()
    ob_bull_bottom_s = pd.Series(ob_bull_bottom, index=df.index).ffill()
    
    ob_bear_bottom_s = pd.Series(ob_bear_bottom, index=df.index).ffill()
    ob_bear_top_s = pd.Series(ob_bear_top, index=df.index).ffill()
    
    return ob_bull_top_s, ob_bull_bottom_s, ob_bear_bottom_s, ob_bear_top_s
"""

if "calc_vwap_stdev" not in content:
    content += advanced_funcs
    
    replacement = """
    # 10. Volume Profile
    res['svp_poc'] = calc_session_svp(res)
    
    # 11. SMC V2 Advanced
    vwap_base, vwap_std = calc_vwap_stdev(res)
    res['vwap_base'] = vwap_base
    res['vwap_std'] = vwap_std
    
    ob_bull_top, ob_bull_bot, ob_bear_bot, ob_bear_top = calc_order_blocks(res)
    res['ob_bull_top'] = ob_bull_top
    res['ob_bull_bot'] = ob_bull_bot
    res['ob_bear_bot'] = ob_bear_bot
    res['ob_bear_top'] = ob_bear_top
    
    # Clean NaN/Inf values
"""
    content = content.replace("    # 10. Volume Profile\n    res['svp_poc'] = calc_session_svp(res)\n    \n    # Clean NaN/Inf values\n", replacement)

    with open("indicators_library.py", "w") as f:
        f.write(content)
    print("Patched indicators_library.py for V2")
else:
    print("Already patched")
