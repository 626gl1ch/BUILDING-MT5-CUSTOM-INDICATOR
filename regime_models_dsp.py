"""
Quantitative Trading Modules: Regime Detection (HMM & Directional Change), DSP Filters & Optimization.
Reference implementations for advanced quantitative trading framework.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# Try importing hmmlearn if available
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


# ============================================================================
# 1. DIRECTIONAL CHANGE (DC) INTRINSIC TIME ALGORITHM
# ============================================================================

def calc_directional_change(prices: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Directional Change (DC) Intrinsic Time Sampling (Tsang et al. / Tsinaslanidis & Zapranis).
    
    Parameters:
        prices: Array of closing or tick prices.
        delta: Threshold percentage (e.g., 0.02 for 2%).
        
    Returns:
        events: Array of event markers (1 = Up DC, -1 = Down DC, 0 = Continuation/OS)
        extreme_prices: Array of extreme price levels at each bar.
    """
    n = len(prices)
    events = np.zeros(n, dtype=int)
    extremes = np.zeros(n, dtype=float)
    
    if n == 0:
        return events, extremes
        
    mode = 1  # 1: Searching for Peak/Up DC, -1: Searching for Trough/Down DC
    last_extreme = prices[0]
    
    for i in range(n):
        price = prices[i]
        extremes[i] = last_extreme
        
        if mode == 1:
            if price > last_extreme:
                last_extreme = price
                extremes[i] = last_extreme
            elif (last_extreme - price) / last_extreme >= delta:
                events[i] = -1  # Downward DC Event Confirmed
                mode = -1
                last_extreme = price
                extremes[i] = last_extreme
        else:
            if price < last_extreme:
                last_extreme = price
                extremes[i] = last_extreme
            elif (price - last_extreme) / last_extreme >= delta:
                events[i] = 1   # Upward DC Event Confirmed
                mode = 1
                last_extreme = price
                extremes[i] = last_extreme
                
    return events, extremes


# ============================================================================
# 2. GAUSSIAN HIDDEN MARKOV MODEL (HMM) REGIME CLASSIFIER
# ============================================================================

def fit_gaussian_hmm_regimes(df: pd.DataFrame, n_components: int = 3, lookback: int = 20) -> pd.Series:
    """
    Classifies Market Regimes using a Gaussian Hidden Markov Model (HMM).
    
    Features:
        - Log Returns
        - Normalized Volatility (ATR / Close)
        
    Returns:
        Series of Regime IDs (0: Ranging, 1: Bullish Trend, 2: Bearish Trend)
    """
    if not HMM_AVAILABLE:
        print("Warning: hmmlearn library not installed. Defaulting to Choppiness Index regime.")
        return pd.Series(0, index=df.index)
        
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    log_ret = np.zeros(len(close))
    log_ret[1:] = np.log(close[1:] / close[:-1])
    
    # ATR / Close
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(lookback).mean().values
    norm_vol = atr / np.where(close > 0, close, 1.0)
    
    X = np.column_stack([log_ret, norm_vol])
    valid_mask = ~np.isnan(X).any(axis=1)
    
    regimes = np.full(len(df), 0, dtype=int)
    if np.sum(valid_mask) < 100:
        return pd.Series(regimes, index=df.index)
        
    X_clean = X[valid_mask]
    
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=50, random_state=42)
    model.fit(X_clean)
    predicted = model.predict(X_clean)
    
    regimes[valid_mask] = predicted
    return pd.Series(regimes, index=df.index)


# ============================================================================
# 3. JOHN EHLERS DIGITAL SIGNAL PROCESSING (DSP) FILTERS
# ============================================================================

def calc_ehlers_supersmoother(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Ehlers 2-Pole SuperSmoother Filter (Rocket Science for Traders / Cybernetic Analysis).
    Removes high-frequency noise with zero lag delay.
    """
    vals = series.values
    n = len(vals)
    ss = np.zeros(n)
    
    if n < 3:
        return series
        
    f = np.sqrt(2.0) * np.pi / period
    a = np.exp(-f)
    b = 2.0 * a * np.cos(f)
    c2 = b
    c3 = -a * a
    c1 = 1.0 - c2 - c3
    
    ss[0] = vals[0]
    ss[1] = vals[1]
    
    for i in range(2, n):
        ss[i] = c1 * (vals[i] + vals[i-1]) / 2.0 + c2 * ss[i-1] + c3 * ss[i-2]
        
    return pd.Series(ss, index=series.index)


def calc_frama(df: pd.DataFrame, period: int = 16, FC: int = 1, SC: int = 200) -> pd.Series:
    """
    Fractal Adaptive Moving Average (FRAMA) by John Ehlers.
    Dynamically adjusts smoothing based on Fractal Dimension (D).
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(close)
    
    frama = np.copy(close)
    half_p = period // 2
    
    w_fast = 2.0 / (FC + 1.0)
    w_slow = 2.0 / (SC + 1.0)
    
    for i in range(period, n):
        h1 = np.max(high[i - period : i - half_p])
        l1 = np.min(low[i - period : i - half_p])
        
        h2 = np.max(high[i - half_p : i])
        l2 = np.min(low[i - half_p : i])
        
        h3 = np.max(high[i - period : i])
        l3 = np.min(low[i - period : i])
        
        n1 = (h1 - l1) / half_p if (h1 - l1) > 0 else 0
        n2 = (h2 - l2) / half_p if (h2 - l2) > 0 else 0
        n3 = (h3 - l3) / period if (h3 - l3) > 0 else 0
        
        if n1 + n2 > 0 and n3 > 0:
            D = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
        else:
            D = 1.0
            
        alpha = np.exp(-4.6 * (D - 1.0))
        alpha = np.clip(alpha, w_slow, w_fast)
        
        frama[i] = alpha * close[i] + (1.0 - alpha) * frama[i-1]
        
    return pd.Series(frama, index=df.index)


if __name__ == '__main__':
    print("Regime Models & DSP Module loaded successfully.")
