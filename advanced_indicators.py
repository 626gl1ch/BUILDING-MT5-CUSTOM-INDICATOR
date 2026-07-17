import numpy as np
import pandas as pd
from scipy.stats import linregress

def calc_super_smoother(price, period=10):
    """John Ehlers' 2-Pole Super Smoother Filter"""
    a1 = np.exp(-1.414 * 3.14159 / period)
    b1 = 2 * a1 * np.cos(1.414 * 180 / period)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    
    ssf = np.zeros_like(price, dtype=float)
    price_arr = price.values if isinstance(price, pd.Series) else price
    ssf[:2] = price_arr[:2]
    
    for i in range(2, len(price_arr)):
        ssf[i] = (c1 * (price_arr[i] + price_arr[i-1]) / 2) + (c2 * ssf[i-1]) + (c3 * ssf[i-2])
    return pd.Series(ssf, index=price.index)

def calc_hurst_exponent(price, window=100):
    """
    Rolling Hurst Exponent. 
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk
    H > 0.5: Trending
    """
    def hurst(p):
        if len(p) < 20: return 0.5
        lags = range(2, min(20, len(p)//2))
        tau = [np.sqrt(np.std(np.subtract(p[lag:], p[:-lag]))) for lag in lags]
        if np.any(np.array(tau) == 0): return 0.5
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]*2.0

    # Rolling apply is slow, so we approximate or use stride tricks if needed.
    # For a vectorized approximation, we'll calculate variance of returns at different lags.
    # To keep it performant for thousands of rows:
    res = np.full(len(price), 0.5)
    p_arr = price.values
    for i in range(window, len(p_arr)):
        if i % 5 == 0: # Calculate every 5 bars to save compute
            res[i] = hurst(p_arr[i-window:i])
        else:
            res[i] = res[i-1]
    return pd.Series(res, index=price.index)

def calc_choppiness_index(df, period=14):
    """Choppiness Index (CHOP). High = Choppy, Low = Trending"""
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(period).sum()
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop

def calc_frama(df, period=10):
    """Fractal Adaptive Moving Average (John Ehlers)"""
    w = period // 2
    hh1 = df['high'].rolling(w).max()
    ll1 = df['low'].rolling(w).min()
    hh2 = df['high'].shift(w).rolling(w).max()
    ll2 = df['low'].shift(w).rolling(w).min()
    hh = df['high'].rolling(period).max()
    ll = df['low'].rolling(period).min()
    
    n1 = (hh1 - ll1) / w
    n2 = (hh2 - ll2) / w
    n3 = (hh - ll) / period
    
    # Avoid div by 0
    d = np.log(2)
    dim = (np.log(n1 + n2) - np.log(n3.replace(0, np.nan))) / d
    alpha = np.exp(-4.6 * (dim - 1))
    alpha = np.clip(alpha, 0.01, 1.0)
    
    frama = np.zeros(len(df))
    close = df['close'].values
    alpha_arr = alpha.fillna(0.01).values
    frama[:period] = close[:period]
    
    for i in range(period, len(close)):
        frama[i] = alpha_arr[i] * close[i] + (1 - alpha_arr[i]) * frama[i-1]
        
    return pd.Series(frama, index=df.index)

def calc_fractional_diff(price, d=0.4, window=20):
    """
    Approximate Fractional Differentiation (Marcos Lopez de Prado).
    Keeps memory while achieving stationarity.
    """
    weights = [1.0]
    for k in range(1, window):
        w = -weights[-1] / k * (d - k + 1)
        weights.append(w)
    weights = np.array(weights[::-1])
    
    def apply_weights(x):
        if len(x) < window: return np.nan
        return np.dot(weights, x)
        
    return price.rolling(window).apply(apply_weights, raw=True)

def add_advanced_indicators(df):
    """Adds the massive suite of Phase 3 Advanced Indicators"""
    res = df.copy()
    
    # Ehlers DSP
    res['ssf_10'] = calc_super_smoother(res['close'], 10)
    res['ssf_20'] = calc_super_smoother(res['close'], 20)
    
    # Market Regimes & Fractals
    res['hurst_100'] = calc_hurst_exponent(res['close'], 100)
    res['chop_14'] = calc_choppiness_index(res, 14)
    res['frama_20'] = calc_frama(res, 20)
    
    # Fractional Differentiation (Stationary Price Proxy)
    res['frac_diff_04'] = calc_fractional_diff(res['close'], d=0.4)
    
    # Advanced Momentum
    res['roc_ssf'] = res['ssf_10'].diff(5)
    
    # Clean
    res = res.replace([np.inf, -np.inf], np.nan)
    return res
