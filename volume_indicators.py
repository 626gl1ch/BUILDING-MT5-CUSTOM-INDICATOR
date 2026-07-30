import numpy as np
import pandas as pd
from numba import njit

@njit(cache=True)
def _rolling_poc_hvn_loop(close, volume, window, num_bins):
    n = len(close)
    poc = np.full(n, np.nan)
    hvn_upper = np.full(n, np.nan)
    hvn_lower = np.full(n, np.nan)
    
    for i in range(window, n):
        c_win = close[i-window:i]
        v_win = volume[i-window:i]
        
        min_c = np.min(c_win)
        max_c = np.max(c_win)
        
        if max_c == min_c:
            poc[i] = min_c
            continue
            
        bin_size = (max_c - min_c) / num_bins
        bins = np.zeros(num_bins + 1)
        for j in range(window):
            idx = int((c_win[j] - min_c) / bin_size)
            if idx > num_bins:
                idx = num_bins
            bins[idx] += v_win[j]
            
        max_idx = np.argmax(bins)
        poc[i] = min_c + (max_idx + 0.5) * bin_size
        
        # Find secondary High Volume Nodes (HVNs) above and below POC
        # Defined as distinct peaks with at least 30% of POC volume
        
        upper_hvn_idx = -1
        upper_hvn_vol = 0
        for b in range(max_idx + 2, num_bins + 1):
            if bins[b] > upper_hvn_vol and bins[b] > (bins[max_idx] * 0.3):
                upper_hvn_vol = bins[b]
                upper_hvn_idx = b
                
        if upper_hvn_idx != -1:
            hvn_upper[i] = min_c + (upper_hvn_idx + 0.5) * bin_size
            
        lower_hvn_idx = -1
        lower_hvn_vol = 0
        for b in range(0, max_idx - 1):
            if bins[b] > lower_hvn_vol and bins[b] > (bins[max_idx] * 0.3):
                lower_hvn_vol = bins[b]
                lower_hvn_idx = b
                
        if lower_hvn_idx != -1:
            hvn_lower[i] = min_c + (lower_hvn_idx + 0.5) * bin_size

    return poc, hvn_upper, hvn_lower

def calc_volume_profile(df, window=200, num_bins=50):
    """
    Calculates Rolling Visible Range Volume Profile.
    num_bins defines the resolution of the profile.
    Returns: POC (Point of Control), Upper HVN, and Lower HVN.
    """
    if 'volume' not in df.columns:
        return pd.Series(df['close'].rolling(window).mean(), index=df.index), \
               pd.Series(np.nan, index=df.index), \
               pd.Series(np.nan, index=df.index)
               
    close = df['close'].values
    volume = df['volume'].values
    
    poc, hvn_upper, hvn_lower = _rolling_poc_hvn_loop(close, volume, window, num_bins)
    
    return (pd.Series(poc, index=df.index), 
            pd.Series(hvn_upper, index=df.index),
            pd.Series(hvn_lower, index=df.index))
