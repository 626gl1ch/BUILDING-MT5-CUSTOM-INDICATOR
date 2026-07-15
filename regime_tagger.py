import numpy as np
import pandas as pd

def precompute_regime_thresholds(df: pd.DataFrame, window_bars: int = 17280): # 60 days * 288 bars
    """
    Precomputes trailing 70th percentile thresholds for ADX and ATR.
    This prevents the tagger from silently misclassifying regimes on assets with 
    structurally higher/lower volatility or trendiness than BTC.
    """
    if 'adx_14' in df.columns:
        # Use a rolling 70th percentile. We use min_periods=288 to get early readings
        df['adx_70_pct'] = df['adx_14'].rolling(window=window_bars, min_periods=288).quantile(0.70)
        # Forward fill the first 288 bars with the first computed value
        first_val = df['adx_70_pct'].dropna().iloc[0] if len(df['adx_70_pct'].dropna()) > 0 else 25.0
        df['adx_70_pct'] = df['adx_70_pct'].fillna(first_val)
    
    if 'atr_14' in df.columns:
        df['atr_70_pct'] = df['atr_14'].rolling(window=window_bars, min_periods=288).quantile(0.70)
        first_atr = df['atr_70_pct'].dropna().iloc[0] if len(df['atr_70_pct'].dropna()) > 0 else 1.0
        df['atr_70_pct'] = df['atr_70_pct'].fillna(first_atr)
        
    return df

def classify_market_regime(df: pd.DataFrame, idx: int) -> int:
    """
    Classifies the market into one of 4 discrete regimes at the given index using
    dynamic, symbol-specific trailing percentiles.
    
    Regimes:
    0: Low Volatility / Ranging
    1: High Volatility / Ranging
    2: Low Volatility / Trending
    3: High Volatility / Trending
    """
    if idx == 0:
        return 0
        
    adx_val = df['adx_14'].iloc[idx] if 'adx_14' in df.columns else 20.0
    atr_val = df['atr_14'].iloc[idx] if 'atr_14' in df.columns else (df['high'].iloc[idx] - df['low'].iloc[idx])
    
    # Dynamic thresholds (fallback to static if not precomputed)
    adx_thresh = df['adx_70_pct'].iloc[idx] if 'adx_70_pct' in df.columns else 25.0
    atr_thresh = df['atr_70_pct'].iloc[idx] if 'atr_70_pct' in df.columns else df['atr_sma_50'].iloc[idx]
    
    # Safety catch for extremely low volatility periods
    if pd.isna(atr_thresh) or atr_thresh == 0:
        atr_thresh = 1e-8
        
    is_trending = adx_val >= adx_thresh
    is_high_vol = atr_val >= atr_thresh
    
    if not is_trending and not is_high_vol:
        return 0 # Low Vol / Ranging
    elif not is_trending and is_high_vol:
        return 1 # High Vol / Ranging
    elif is_trending and not is_high_vol:
        return 2 # Low Vol / Trending
    else:
        return 3 # High Vol / Trending
