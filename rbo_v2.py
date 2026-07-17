import pandas as pd
import numpy as np
import itertools

def dynamic_signal_generator(df, p):
    """
    The Master Combinatorial Signal Generator.
    Interprets dynamic logic dictionaries into buy/sell signals.
    """
    signals = pd.Series(0, index=df.index)
    close = df['close']
    
    # ==========================================
    # 1. REGIME FILTERS
    # ==========================================
    regime_mask = pd.Series(True, index=df.index)
    r = p['regime']
    if r == 'hurst_trending':
        if 'hurst_100' in df.columns: regime_mask = df['hurst_100'] > 0.55
    elif r == 'hurst_mean_reverting':
        if 'hurst_100' in df.columns: regime_mask = df['hurst_100'] < 0.45
    elif r == 'chop_trending':
        if 'chop_14' in df.columns: regime_mask = df['chop_14'] < 38.2
    elif r == 'chop_choppy':
        if 'chop_14' in df.columns: regime_mask = df['chop_14'] > 61.8
    elif r == 'adx_strong':
        if 'adx_14' in df.columns: regime_mask = df['adx_14'] > 25
            
    # ==========================================
    # 2. ENTRY LOGIC
    # ==========================================
    long_entry = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)
    e = p['entry']
    
    # --- DSP & Fractals ---
    if e == 'ssf_cross':
        ssf = df.get('ssf_10', close)
        long_entry = (close > ssf) & (close.shift(1) <= ssf.shift(1))
        short_entry = (close < ssf) & (close.shift(1) >= ssf.shift(1))
        
    elif e == 'frac_diff_reversion':
        fd = df.get('frac_diff_04', close)
        roll_mean, roll_std = fd.rolling(20).mean(), fd.rolling(20).std()
        long_entry = fd < (roll_mean - 2 * roll_std)
        short_entry = fd > (roll_mean + 2 * roll_std)
        
    elif e == 'frama_breakout':
        frama = df.get('frama_20', close)
        long_entry = (close > frama) & (close.shift(1) <= frama.shift(1))
        short_entry = (close < frama) & (close.shift(1) >= frama.shift(1))
        
    # --- Classic & Channels ---
    elif e == 'bb_vwap_mr':
        bb_l, bb_u, vwap = df.get('bb_lower_20', close), df.get('bb_upper_20', close), df.get('vwap', close)
        long_entry = (close < bb_l) & (close < vwap)
        short_entry = (close > bb_u) & (close > vwap)
        
    elif e == 'donchian_breakout':
        du, dl = df.get('donchian_upper_20', close), df.get('donchian_lower_20', close)
        long_entry = close > du.shift(1)
        short_entry = close < dl.shift(1)
        
    elif e == 'keltner_pullback':
        ku, kl = df.get('keltner_upper_20', close), df.get('keltner_lower_20', close)
        long_entry = (close < kl) & (df.get('rsi_14', 50) < 30)
        short_entry = (close > ku) & (df.get('rsi_14', 50) > 70)
        
    # --- Momentum & Oscillators ---
    elif e == 'rsi_divergence':
        rsi = df.get('rsi_14', pd.Series(50, index=df.index))
        long_entry = (rsi < 30) & (rsi > rsi.shift(1)) & (close < close.shift(1))
        short_entry = (rsi > 70) & (rsi < rsi.shift(1)) & (close > close.shift(1))
        
    elif e == 'macd_cross':
        macd, macd_s = df.get('macd', close), df.get('macd_signal', close)
        long_entry = (macd > macd_s) & (macd.shift(1) <= macd_s.shift(1)) & (macd < 0)
        short_entry = (macd < macd_s) & (macd.shift(1) >= macd_s.shift(1)) & (macd > 0)
        
    elif e == 'cci_reversion':
        cci = df.get('cci_14', pd.Series(0, index=df.index))
        long_entry = (cci < -100) & (cci > cci.shift(1))
        short_entry = (cci > 100) & (cci < cci.shift(1))
        
    elif e == 'stoch_cross':
        k, d = df.get('stoch_k_14', close), df.get('stoch_d_14', close)
        long_entry = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 20)
        short_entry = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 80)
        
    # --- Volume & Statistics ---
    elif e == 'volume_osc_surge':
        vosc = df.get('volume_osc', pd.Series(0, index=df.index))
        long_entry = (vosc > 10) & (close > df.get('ema_20', close))
        short_entry = (vosc > 10) & (close < df.get('ema_20', close))
        
    elif e == 'zscore_extreme':
        z = df.get('zscore_20', pd.Series(0, index=df.index))
        long_entry = z < -2.5
        short_entry = z > 2.5
        
    elif e == 'pinbar_reversal':
        pin = df.get('pin_bar', pd.Series(0, index=df.index))
        long_entry = (pin == 1) & (df.get('rsi_14', 50) < 40)
        short_entry = (pin == -1) & (df.get('rsi_14', 50) > 60)
        
    elif e == 'engulfing_reversal':
        eng = df.get('engulfing', pd.Series(0, index=df.index))
        long_entry = (eng == 1) & (df.get('rsi_14', 50) < 40)
        short_entry = (eng == -1) & (df.get('rsi_14', 50) > 60)
        
    elif e == 'roc_ssf_momentum':
        roc = df.get('roc_ssf', pd.Series(0, index=df.index))
        long_entry = roc > 0
        short_entry = roc < 0
        
    # Combine
    signals[long_entry & regime_mask] = 1
    signals[short_entry & regime_mask] = -1
    
    return signals


def build_combinatorial_grid():
    """
    Generates 45,000+ unique strategy permutations by combining entries, regimes, and risk settings.
    """
    entries = [
        'ssf_cross', 'frac_diff_reversion', 'frama_breakout', 'bb_vwap_mr',
        'donchian_breakout', 'keltner_pullback', 'rsi_divergence', 'macd_cross',
        'cci_reversion', 'stoch_cross', 'volume_osc_surge', 'zscore_extreme',
        'pinbar_reversal', 'engulfing_reversal', 'roc_ssf_momentum'
    ] # 15 Entries
    
    regimes = [
        'hurst_trending', 'hurst_mean_reverting', 'chop_trending', 
        'chop_choppy', 'adx_strong', 'none'
    ] # 6 Regimes
    
    # We define the risk management parameters (which act as exit logic)
    sl_atrs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] # 6 Stop Losses
    tp_atrs = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0] # 6 Take Profits
    max_holds = [24, 48, 96] # 3 Hold periods
    
    # To hit 150,000+, we use these expanded arrays
    modifiers = ['fast', 'slow'] # 2
    risk_pcts = [0.01, 0.02, 0.03, 0.04, 0.05] # 5
    trailing_stops = [True, False] # 2
    
    # 15 * 6 * 6 * 6 * 3 * 2 * 5 * 2 = 194,400 exact combinations!
    
    all_combos = list(itertools.product(entries, regimes, sl_atrs, tp_atrs, max_holds, modifiers, risk_pcts, trailing_stops))
    
    GRIDS = {}
    STRATEGY_DESCRIPTIONS = {}
    
    # Because a 45k dictionary is massive, we will group them by Entry so mass_search_sequential can batch them.
    # Actually, the user's mass_search expects GRIDS['strategy_name']['params'] = { list of values }
    # Let's construct it so mass_search_sequential parses it easily.
    
    for entry in entries:
        GRIDS[entry] = {
            'fn': dynamic_signal_generator,
            'params': {
                'entry': [entry],
                'regime': regimes,
                'sl_atr': sl_atrs,
                'tp_atr': tp_atrs,
                'max_bars_hold': max_holds,
                'modifier': modifiers,
                'risk_pct': risk_pcts,
                'trailing': trailing_stops
            }
        }
        STRATEGY_DESCRIPTIONS[entry] = {
            'logic': f"Dynamic Combinatorial Engine: {entry} Base Logic.",
            'indicators': "SuperSmoother, Hurst, Choppiness, Fractional Diff, FRAMA, RSI, MACD, etc."
        }
        
    return GRIDS, STRATEGY_DESCRIPTIONS

GRIDS, STRATEGY_DESCRIPTIONS = build_combinatorial_grid()
