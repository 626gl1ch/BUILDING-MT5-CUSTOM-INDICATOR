import numpy as np
import pandas as pd
import indicators as ind
from volume_indicators import calc_volume_profile

def build_v3_indicators(df):
    """Computes all required oscillators and volume profile for V3 miner."""
    df = df.copy()
    
    # Core Oscillators
    df['alma9'] = ind.alma(df['close'], period=9)
    df['ema9'] = ind.ema(df['close'], period=9)
    df['ema50'] = ind.ema(df['close'], period=50) # For HTF trend
    
    df['adx_14'] = ind.adx(df['high'], df['low'], df['close'], period=14)
    df['chop_14'] = ind.choppiness_index(df['high'], df['low'], df['close'], period=14)
    
    df['rsi_14'] = ind.rsi(df['close'], period=14)
    df['rsi_3'] = ind.rsi(df['close'], period=3) # Fast RSI for scalping
    
    df['stoch_k'], df['stoch_d'] = ind.stoch_rsi(df['close'], rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
    
    df['macd_line'], df['macd_signal'], df['macd_hist'] = ind.macd(df['close'], fast=12, slow=26, signal=9)
    
    df['bb_upper'], _, df['bb_lower'], _ = ind.bollinger(df['close'], period=20, k=2.5)
    df['bb_upper_2'], _, df['bb_lower_2'], _ = ind.bollinger(df['close'], period=20, k=2.0)
    
    df['atr_14'] = ind.atr(df['high'], df['low'], df['close'], period=14)
    
    # Volume Profile (Rolling 200 bars ~ 16 hours of 5m data)
    poc, hvn_upper, hvn_lower = calc_volume_profile(df, window=200, num_bins=50)
    df['vp_poc'] = poc
    df['vp_hvn_up'] = hvn_upper
    df['vp_hvn_dn'] = hvn_lower
    
    # ALMA Distance in terms of ATR for exhaustion
    df['alma_dist_atr'] = (df['close'] - df['alma9']).abs() / df['atr_14']
    
    # Forward fill NaNs for Volume Profile just in case
    df['vp_poc'] = df['vp_poc'].ffill()
    df['vp_hvn_up'] = df['vp_hvn_up'].ffill()
    df['vp_hvn_dn'] = df['vp_hvn_dn'].ffill()
    
    return df.dropna()

def get_v3_signals(df, strategy_name, **params):
    """
    Returns signals (1 for long, -1 for short) and optionally dynamic exits.
    
    Regime Filters built-in:
    - Avoid low volatility: ADX > 15
    - Avoid tight ranging dead zones: Chop < 60
    """
    n = len(df)
    long_e = pd.Series(False, index=df.index)
    short_e = pd.Series(False, index=df.index)
    exit_long = pd.Series(False, index=df.index)
    exit_short = pd.Series(False, index=df.index)
    
    # Global Regime Filter: We want *some* directional movement or volatility to mean-revert against.
    # We avoid completely dead markets (ADX < 15 and Chop > 60).
    valid_regime = (df['adx_14'] > 15) & (df['chop_14'] < 60)
    
    # Trend alignment (Mean revert into the HTF trend)
    up_trend = df['close'] > df['ema50']
    dn_trend = df['close'] < df['ema50']
    
    if strategy_name == 'ALMA_POC_Bounce':
        # Strategy 1: Price drops far below ALMA 9, lands on Point of Control, StochRSI oversold.
        poc_tolerance = df['atr_14'] * 0.5 # Must be within 0.5 ATR of POC
        
        near_poc = (df['low'] <= (df['vp_poc'] + poc_tolerance)) & (df['close'] >= (df['vp_poc'] - poc_tolerance))
        near_poc_short = (df['high'] >= (df['vp_poc'] - poc_tolerance)) & (df['close'] <= (df['vp_poc'] + poc_tolerance))
        
        alma_stretched_dn = df['close'] < df['alma9'] - df['atr_14']
        alma_stretched_up = df['close'] > df['alma9'] + df['atr_14']
        
        long_e = near_poc & alma_stretched_dn & (df['stoch_k'] < 20) & up_trend & valid_regime
        short_e = near_poc_short & alma_stretched_up & (df['stoch_k'] > 80) & dn_trend & valid_regime
        
        # Dynamic Exit: Opposite ALMA cross or StochRSI extreme
        exit_long = (df['close'] > df['alma9']) | (df['stoch_k'] > 80)
        exit_short = (df['close'] < df['alma9']) | (df['stoch_k'] < 20)
        
    elif strategy_name == 'BB_Stoch_MACD_Exhaustion':
        # Strategy 2: Bollinger Band pierce + StochRSI cross + MACD momentum loss
        
        bb_pierce_dn = df['low'] < df['bb_lower']
        bb_pierce_up = df['high'] > df['bb_upper']
        
        macd_decel_dn = (df['macd_hist'] > df['macd_hist'].shift(1)) & (df['macd_hist'] < 0)
        macd_decel_up = (df['macd_hist'] < df['macd_hist'].shift(1)) & (df['macd_hist'] > 0)
        
        stoch_cross_up = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & (df['stoch_k'] < 30)
        stoch_cross_dn = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) & (df['stoch_k'] > 70)
        
        long_e = bb_pierce_dn & macd_decel_dn & stoch_cross_up & up_trend & valid_regime
        short_e = bb_pierce_up & macd_decel_up & stoch_cross_dn & dn_trend & valid_regime
        
        exit_long = df['close'] > df['ema9']
        exit_short = df['close'] < df['ema9']
        
    elif strategy_name == 'EMA9_HVN_Trap':
        # Strategy 3: Price loses EMA 9, hits a High Volume Node (HVN) below, and closes back above EMA 9 (Trap).
        
        hvn_tolerance = df['atr_14'] * 0.5
        hit_hvn_dn = (df['low'] <= (df['vp_hvn_dn'] + hvn_tolerance)) & (df['vp_hvn_dn'].notna())
        hit_hvn_up = (df['high'] >= (df['vp_hvn_up'] - hvn_tolerance)) & (df['vp_hvn_up'].notna())
        
        reclaim_ema9_up = (df['close'] > df['ema9']) & (df['close'].shift(1) < df['ema9'].shift(1))
        reclaim_ema9_dn = (df['close'] < df['ema9']) & (df['close'].shift(1) > df['ema9'].shift(1))
        
        hit_hvn_dn_recent = hit_hvn_dn.rolling(3).max() > 0
        hit_hvn_up_recent = hit_hvn_up.rolling(3).max() > 0
        
        long_e = hit_hvn_dn_recent & reclaim_ema9_up & up_trend & valid_regime
        short_e = hit_hvn_up_recent & reclaim_ema9_dn & dn_trend & valid_regime
        
        exit_long = df['rsi_14'] > 70
        exit_short = df['rsi_14'] < 30

    elif strategy_name == 'Fast_RSI_Chop_Scalp':
        # Strategy 4: High Choppiness (Price is trapped in a range), but we only trade the absolute edges
        chop_regime = df['chop_14'] > 60
        
        long_e = (df['rsi_3'] < 15) & (df['close'] < df['bb_lower_2']) & chop_regime
        short_e = (df['rsi_3'] > 85) & (df['close'] > df['bb_upper_2']) & chop_regime
        
        exit_long = df['rsi_3'] > 70
        exit_short = df['rsi_3'] < 30

    signals = pd.Series(0, index=df.index)
    signals.loc[long_e] = 1
    signals.loc[short_e] = -1
    
    exit_long = exit_long.fillna(False).astype(int)
    exit_short = exit_short.fillna(False).astype(int)
    
    return signals, exit_long, exit_short
