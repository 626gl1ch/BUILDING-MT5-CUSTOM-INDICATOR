import pandas as pd
import numpy as np

def generate_smc_baseline(df, p):
    close = df['close']
    high = df['high']
    low = df['low']
    lookback = p.get('lookback', 20)
    
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    signals = pd.Series(0, index=df.index)
    
    sweep_high = (high > swing_high.shift(1)) & \
                 (close < swing_high.shift(1)) & \
                 (close.shift(1) < swing_high.shift(1))
    
    sweep_low = (low < swing_low.shift(1)) & \
                (close > swing_low.shift(1)) & \
                (close.shift(1) > swing_low.shift(1))
    
    signals[sweep_low] = 1
    signals[sweep_high] = -1
    return signals

def generate_smc_fvg(df, p):
    close = df['close']
    high = df['high']
    low = df['low']
    lookback = p.get('lookback', 20)
    fvg_lookback = p.get('fvg_lookback', 5)
    
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    bullish_fvg = low > high.shift(2)
    bearish_fvg = high < low.shift(2)
    recent_bearish_fvg = bearish_fvg.rolling(fvg_lookback).max() > 0
    recent_bullish_fvg = bullish_fvg.rolling(fvg_lookback).max() > 0
    
    signals = pd.Series(0, index=df.index)
    
    sweep_high = (high > swing_high.shift(1)) & \
                 (close < swing_high.shift(1)) & \
                 (close.shift(1) < swing_high.shift(1)) & \
                 (~recent_bullish_fvg)
    
    sweep_low = (low < swing_low.shift(1)) & \
                (close > swing_low.shift(1)) & \
                (close.shift(1) > swing_low.shift(1)) & \
                (~recent_bearish_fvg)
    
    signals[sweep_low] = 1
    signals[sweep_high] = -1
    return signals

def generate_smc_god_mode(df, p):
    """
    Ultimate Synergy:
    1. SMC Liquidity Sweep (Fake-out rejection)
    2. FVG check (don't sweep into heavy opposite displacement)
    3. RVOL check (ensure institutional volume presence)
    4. HTF Trend Alignment (don't fight the HTF trend)
    """
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']
    
    lookback = p.get('lookback', 20)
    fvg_lookback = p.get('fvg_lookback', 5)
    rvol_thresh = p.get('rvol_thresh', 1.0)
    ema_p = p.get('ema_p', 600)  # 600 on 5m = 50 on 1H
    
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    vol_sma = df.get('volume_sma_20', vol)
    rvol = vol / (vol_sma + 1e-9)
    
    ema_trend = df.get(f'ema_{ema_p}', close)
    
    bullish_fvg = low > high.shift(2)
    bearish_fvg = high < low.shift(2)
    recent_bearish_fvg = bearish_fvg.rolling(fvg_lookback).max() > 0
    recent_bullish_fvg = bullish_fvg.rolling(fvg_lookback).max() > 0
    
    signals = pd.Series(0, index=df.index)
    
    sweep_high = (high > swing_high.shift(1)) & \
                 (close < swing_high.shift(1)) & \
                 (close.shift(1) < swing_high.shift(1)) & \
                 (~recent_bullish_fvg) & \
                 (rvol > rvol_thresh) & \
                 (close < ema_trend)
    
    sweep_low = (low < swing_low.shift(1)) & \
                (close > swing_low.shift(1)) & \
                (close.shift(1) > swing_low.shift(1)) & \
                (~recent_bearish_fvg) & \
                (rvol > rvol_thresh) & \
                (close > ema_trend)
    
    signals[sweep_low] = 1
    signals[sweep_high] = -1
    return signals

def generate_smc_vol_profile(df, p):
    '''
    SMC Liquidity Sweep + Volume Profile (POC) Directional Filter
    '''
    close = df['close']
    high = df['high']
    low = df['low']
    
    lookback = p.get('lookback', 20)
    
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    poc = df.get('svp_poc', close)
    
    signals = pd.Series(0, index=df.index)
    
    sweep_high = (high > swing_high.shift(1)) &                  (close < swing_high.shift(1)) &                  (close.shift(1) < swing_high.shift(1)) &                  (close < poc)
    
    sweep_low = (low < swing_low.shift(1)) &                 (close > swing_low.shift(1)) &                 (close.shift(1) > swing_low.shift(1)) &                 (close > poc)
    
    signals[sweep_low] = 1
    signals[sweep_high] = -1
    return signals


def generate_smc_v2_vwap(df, p):
    '''
    Variant A: SMC + Institutional VWAP Deviations
    Sweep must occur while price is beyond the 2nd standard deviation of VWAP.
    '''
    close = df['close']
    high = df['high']
    low = df['low']
    lookback = p.get('lookback', 20)
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    poc = df.get('svp_poc', close)
    vwap_base = df.get('vwap_base', close)
    import pandas as pd
    vwap_std = df.get('vwap_std', pd.Series(0, index=df.index))
    
    import pandas as pd
    
    upper_band = vwap_base + (2.0 * vwap_std)
    lower_band = vwap_base - (2.0 * vwap_std)
    
    signals = pd.Series(0, index=df.index)
    
    sweep_high = (high > swing_high.shift(1)) &                  (close < swing_high.shift(1)) &                  (close.shift(1) < swing_high.shift(1)) &                  (close < poc) &                  (high >= upper_band)
                 
    sweep_low = (low < swing_low.shift(1)) &                 (close > swing_low.shift(1)) &                 (close.shift(1) > swing_low.shift(1)) &                 (close > poc) &                 (low <= lower_band)
                
    signals[sweep_low] = 1
    signals[sweep_high] = -1
    return signals

def generate_smc_v2_ob(df, p):
    '''
    Variant C: SMC + Multi-Timeframe Order Blocks (OB)
    Sweep must occur entirely within an unmitigated Order Block.
    '''
    close = df['close']
    high = df['high']
    low = df['low']
    lookback = p.get('lookback', 20)
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    poc = df.get('svp_poc', close)
    ob_bull_top = df.get('ob_bull_top', close)
    ob_bull_bot = df.get('ob_bull_bot', close)
    ob_bear_top = df.get('ob_bear_top', close)
    ob_bear_bot = df.get('ob_bear_bot', close)
    
    import pandas as pd
    
    signals = pd.Series(0, index=df.index)
    
    sweep_high = (high > swing_high.shift(1)) &                  (close < swing_high.shift(1)) &                  (close.shift(1) < swing_high.shift(1)) &                  (close < poc) &                  (high <= ob_bear_top) & (high >= ob_bear_bot)
                 
    sweep_low = (low < swing_low.shift(1)) &                 (close > swing_low.shift(1)) &                 (close.shift(1) > swing_low.shift(1)) &                 (close > poc) &                 (low <= ob_bull_top) & (low >= ob_bull_bot)
                
    signals[sweep_low] = 1
    signals[sweep_high] = -1
    return signals
