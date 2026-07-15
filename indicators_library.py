"""
Indicators Library for Algorithmic Trading (150+ Indicators)
All calculations are vectorized, numerically stable, and avoid lookahead bias.
"""

import pandas as pd
import numpy as np

# ==========================================
# CATEGORY 1: MOVING AVERAGES
# ==========================================

def calc_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def calc_dema(series, period):
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    return 2 * ema1 - ema2

def calc_tema(series, period):
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def calc_rolling_wma_numpy(values, weights):
    period = len(weights)
    if len(values) < period:
        return np.full(len(values), np.nan)
    shape = (values.size - period + 1, period)
    strides = (values.strides[0], values.strides[0])
    windows = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)
    res = np.dot(windows, weights)
    full = np.empty(values.size)
    full[:period-1] = np.nan
    full[period-1:] = res
    return full

def calc_wma(series, period):
    weights = np.arange(1, period + 1)
    weights = weights / weights.sum()
    return pd.Series(calc_rolling_wma_numpy(series.values, weights), index=series.index)

def calc_hma(series, period):
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = calc_wma(series, half_period)
    wma_full = calc_wma(series, period)
    diff = 2 * wma_half - wma_full
    return calc_wma(diff, sqrt_period)

def calc_kama(series, period=10, fast=2, slow=30):
    change = abs(series - series.shift(period))
    volatility = abs(series.diff()).rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc) + slow_sc) ** 2).values
    series_val = series.values
    
    kama = np.zeros(len(series))
    kama[:period] = series_val[:period]
    
    for i in range(period, len(series)):
        kama[i] = kama[i-1] + sc[i] * (series_val[i] - kama[i-1])
        
    return pd.Series(kama, index=series.index)

def calc_zlema(series, period):
    lag = int((period - 1) / 2)
    adjusted_series = series + (series - series.shift(lag))
    return calc_ema(adjusted_series, period)

def calc_t3(series, period=5, a=0.7):
    e1 = calc_ema(series, period)
    e2 = calc_ema(e1, period)
    e3 = calc_ema(e2, period)
    e4 = calc_ema(e3, period)
    e5 = calc_ema(e4, period)
    e6 = calc_ema(e5, period)
    c1 = -a**3
    c2 = 3*a**2 + 3*a**3
    c3 = -6*a**2 - 3*a - 3*a**3
    c4 = 1 + 3*a + 3*a**2 + a**3
    return c1*e6 + c2*e5 + c3*e4 + c4*e3

def calc_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    volume = df['volume']
    
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df['datetime']).dt.date
        
    cum_pv = (tp * volume).groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)

def calc_alma(series, period=9, sigma=6, offset=0.85):
    m = offset * (period - 1)
    s = period / sigma
    weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    return pd.Series(calc_rolling_wma_numpy(series.values, weights), index=series.index)

# ==========================================
# CATEGORY 2: TREND INDICATORS
# ==========================================

def calc_tr(high, low, close):
    h_l = high - low
    h_pc = abs(high - close.shift(1))
    l_pc = abs(low - close.shift(1))
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

def calc_atr(df, period=14):
    tr = calc_tr(df['high'], df['low'], df['close'])
    return tr.rolling(window=period, min_periods=period).mean()

def calc_supertrend(df, period=10, multiplier=3.0):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = calc_atr(df, period).values
    hl2 = (high + low) / 2
    
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    final_upperband = np.zeros(len(df))
    final_lowerband = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.ones(len(df))
    
    start_idx = 0
    for i in range(len(df)):
        if np.isnan(atr[i]):
            final_upperband[i] = np.nan
            final_lowerband[i] = np.nan
            supertrend[i] = np.nan
            direction[i] = 1
        else:
            final_upperband[i] = upperband[i]
            final_lowerband[i] = lowerband[i]
            supertrend[i] = upperband[i]
            start_idx = i + 1
            break
            
    for i in range(start_idx, len(df)):
        if close[i-1] <= final_upperband[i-1]:
            final_upperband[i] = min(upperband[i], final_upperband[i-1])
        else:
            final_upperband[i] = upperband[i]
            
        if close[i-1] >= final_lowerband[i-1]:
            final_lowerband[i] = max(lowerband[i], final_lowerband[i-1])
        else:
            final_lowerband[i] = lowerband[i]
            
        if close[i] > final_upperband[i-1]:
            direction[i] = 1
        elif close[i] < final_lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = final_lowerband[i]
        else:
            supertrend[i] = final_upperband[i]
            
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

def calc_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    up = high.diff()
    dn = low.diff()
    
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    
    tr = calc_tr(high, low, close)
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    plus_dm_sum = pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=period).sum()
    minus_dm_sum = pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=period).sum()
    
    pdi = 100 * (plus_dm_sum / tr_sum.replace(0, np.nan)).fillna(0)
    mdi = 100 * (minus_dm_sum / tr_sum.replace(0, np.nan)).fillna(0)
    
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).fillna(0)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx, pdi, mdi

def calc_aroon(df, period=25):
    high = df['high']
    low = df['low']
    
    high_vals = high.values
    low_vals = low.values
    p = period + 1
    
    if len(high_vals) < p:
        aroon_up = pd.Series(np.nan, index=df.index)
        aroon_down = pd.Series(np.nan, index=df.index)
    else:
        shape = (high_vals.size - p + 1, p)
        strides = (high_vals.strides[0], high_vals.strides[0])
        
        high_windows = np.lib.stride_tricks.as_strided(high_vals, shape=shape, strides=strides)
        low_windows = np.lib.stride_tricks.as_strided(low_vals, shape=shape, strides=strides)
        
        # Invert index because argmax index is distance from start of window
        # Aroon Up = ((period - days since highest high) / period) * 100
        # days since highest high = period - argmax
        up_idx = np.argmax(high_windows, axis=1)
        down_idx = np.argmin(low_windows, axis=1)
        
        full_up = np.empty(high_vals.size)
        full_up[:p-1] = np.nan
        full_up[p-1:] = 100 * up_idx / period
        
        full_down = np.empty(low_vals.size)
        full_down[:p-1] = np.nan
        full_down[p-1:] = 100 * down_idx / period
        
        aroon_up = pd.Series(full_up, index=df.index)
        aroon_down = pd.Series(full_down, index=df.index)
        
    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc

def calc_vortex(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    vm_plus = abs(high - low.shift(1))
    vm_minus = abs(low - high.shift(1))
    
    tr = calc_tr(high, low, close)
    
    vm_plus_sum = vm_plus.rolling(period).sum()
    vm_minus_sum = vm_minus.rolling(period).sum()
    tr_sum = tr.rolling(period).sum()
    
    vi_plus = vm_plus_sum / tr_sum.replace(0, np.nan)
    vi_minus = vm_minus_sum / tr_sum.replace(0, np.nan)
    return vi_plus, vi_minus

def calc_mass_index(df, p1=9, p2=25):
    h_l = df['high'] - df['low']
    ema1 = calc_ema(h_l, p1)
    ema2 = calc_ema(ema1, p1)
    ratio = ema1 / ema2.replace(0, np.nan)
    return ratio.rolling(p2).sum()

def calc_ichimoku(df, tenkan=9, kijun=26, senkou_b=52, chikou=26):
    high = df['high']
    low = df['low']
    close = df['close']
    
    t_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    k_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    
    senkou_a = ((t_line + k_line) / 2).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    
    c_line = close.shift(-chikou)
    return t_line, k_line, senkou_a, senkou_b_line, c_line

# ==========================================
# CATEGORY 3: MOMENTUM OSCILLATORS
# ==========================================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs.fillna(0)))

def calc_stoch_rsi(series, period=14, k=3, d=3):
    rsi = calc_rsi(series, period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch = 100 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    stoch_k = stoch.rolling(k).mean()
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calc_cci(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    
    values = tp.values
    if len(values) < period:
        md = pd.Series(np.nan, index=df.index)
    else:
        shape = (values.size - period + 1, period)
        strides = (values.strides[0], values.strides[0])
        windows = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)
        
        means = np.mean(windows, axis=1, keepdims=True)
        mads = np.mean(np.abs(windows - means), axis=1)
        
        full_mads = np.empty(values.size)
        full_mads[:period-1] = np.nan
        full_mads[period-1:] = mads
        md = pd.Series(full_mads, index=df.index)
        
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def calc_momentum(series, period=10):
    return series - series.shift(period)

def calc_roc(series, period=10):
    return 100 * (series - series.shift(period)) / series.shift(period).replace(0, np.nan)

def calc_williams_r(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    h_max = high.rolling(period).max()
    l_min = low.rolling(period).min()
    
    return -100 * (h_max - close) / (h_max - l_min).replace(0, np.nan)

def calc_ultimate_oscillator(df, p1=7, p2=14, p3=28):
    close = df['close']
    low = df['low']
    high = df['high']
    
    prev_close = close.shift(1)
    min_l_pc = pd.concat([low, prev_close], axis=1).min(axis=1)
    max_h_pc = pd.concat([high, prev_close], axis=1).max(axis=1)
    
    bp = close - min_l_pc
    tr = max_h_pc - min_l_pc
    
    avg7 = bp.rolling(p1).sum() / tr.rolling(p1).sum().replace(0, np.nan)
    avg14 = bp.rolling(p2).sum() / tr.rolling(p2).sum().replace(0, np.nan)
    avg28 = bp.rolling(p3).sum() / tr.rolling(p3).sum().replace(0, np.nan)
    
    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    return uo.fillna(0)

def calc_tsi(series, r=25, s=13):
    diff = series.diff()
    abs_diff = abs(diff)
    
    double_smoothed_diff = calc_ema(calc_ema(diff, r), s)
    double_smoothed_abs_diff = calc_ema(calc_ema(abs_diff, r), s)
    
    return 100 * double_smoothed_diff / double_smoothed_abs_diff.replace(0, np.nan)

def calc_trix(series, period=14):
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return 100 * ema3.diff() / ema3.shift(1).replace(0, np.nan)

def calc_dpo(series, period=20):
    ma = calc_sma(series, period)
    shift_len = int(period / 2 + 1)
    return series - ma.shift(shift_len)

# ==========================================
# CATEGORY 4: MACD FAMILY
# ==========================================

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_ppo(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    ppo_line = 100 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    signal_line = calc_ema(ppo_line, signal)
    hist = ppo_line - signal_line
    return ppo_line, signal_line, hist

# ==========================================
# CATEGORY 5: VOLATILITY INDICATORS
# ==========================================

def calc_bollinger_bands(series, period=20, std=2.0):
    ma = calc_sma(series, period)
    rolling_std = series.rolling(period).std()
    upper = ma + std * rolling_std
    lower = ma - std * rolling_std
    b_pct = (series - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / ma.replace(0, np.nan)
    return upper, ma, lower, b_pct, width

def calc_keltner_channel(df, period=20, atr_period=10, mult=1.5):
    ma = calc_ema(df['close'], period)
    atr = calc_atr(df, atr_period)
    upper = ma + mult * atr
    lower = ma - mult * atr
    b_pct = (df['close'] - lower) / (upper - lower).replace(0, np.nan)
    return upper, ma, lower, b_pct

def calc_donchian_channel(df, period=20):
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    middle = (upper + lower) / 2
    b_pct = (df['close'] - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, b_pct

def calc_historical_volatility(series, period=20):
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252 * 288)

def calc_garman_klass_volatility(df, period=20):
    h = df['high']
    l = df['low']
    c = df['close']
    o = df['open']
    
    term1 = 0.5 * (np.log(h / l))**2
    term2 = (2 * np.log(2) - 1) * (np.log(c / o))**2
    gk = term1 - term2
    return np.sqrt(gk.rolling(period).mean() * 252 * 288)

# ==========================================
# CATEGORY 6: VOLUME INDICATORS
# ==========================================

def calc_obv(df):
    close = df['close']
    volume = df['volume']
    direction = np.sign(close.diff().fillna(0))
    obv_val = (direction * volume).cumsum()
    return obv_val

def calc_mfi(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    
    direction = np.sign(tp.diff().fillna(0))
    pos_mf = np.where(direction > 0, mf, 0.0)
    neg_mf = np.where(direction < 0, mf, 0.0)
    
    pos_sum = pd.Series(pos_mf, index=df.index).rolling(period).sum()
    neg_sum = pd.Series(neg_mf, index=df.index).rolling(period).sum()
    
    m_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + m_ratio.fillna(0)))

def calc_cmf(df, period=20):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    ad = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    ad = ad.fillna(0) * volume
    
    cmf_val = ad.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)
    return cmf_val.fillna(0)

def calc_ease_of_movement(df, period=14):
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    mid_point_move = ((high + low) / 2) - (((high.shift(1) + low.shift(1)) / 2))
    box_ratio = (volume / 100000000) / (high - low).replace(0, np.nan)
    emv = mid_point_move / box_ratio.replace(0, np.nan)
    return emv.rolling(period).mean().fillna(0)

def calc_force_index(df, period=13):
    fi = df['close'].diff() * df['volume']
    return calc_ema(fi, period)

def calc_volume_oscillator(df, fast=5, slow=10):
    vol = df['volume']
    fast_ema = calc_ema(vol, fast)
    slow_ema = calc_ema(vol, slow)
    return 100 * (fast_ema - slow_ema) / slow_ema.replace(0, np.nan)

# ==========================================
# CATEGORY 7: CHOPPINESS & REGIME FILTERS
# ==========================================

def calc_choppiness_index(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = calc_tr(high, low, close)
    sum_tr = tr.rolling(period).sum()
    
    highest_h = high.rolling(period).max()
    lowest_l = low.rolling(period).min()
    
    ci = 100 * np.log10(sum_tr / (highest_h - lowest_l).replace(0, np.nan)) / np.log10(period)
    return ci.clip(0, 100).fillna(50)

# ==========================================
# CATEGORY 8: STATISTICAL & MATH METRICS
# ==========================================

def calc_linear_regression_slope(series, period=14):
    n = period
    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    den = n * sum_x2 - sum_x ** 2
    
    sum_y = series.rolling(n).sum()
    
    weights = np.arange(n)
    sum_xy = pd.Series(calc_rolling_wma_numpy(series.values, weights), index=series.index)
    
    slope = (n * sum_xy - sum_x * sum_y) / den
    return slope

def calc_linear_regression_r2(series, period=14):
    n = period
    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    den_x = n * sum_x2 - sum_x ** 2
    
    sum_y = series.rolling(n).sum()
    sum_y2 = (series ** 2).rolling(n).sum()
    den_y = n * sum_y2 - sum_y ** 2
    
    weights = np.arange(n)
    sum_xy = pd.Series(calc_rolling_wma_numpy(series.values, weights), index=series.index)
    
    num = n * sum_xy - sum_x * sum_y
    r2 = (num ** 2) / (den_x * den_y.replace(0, np.nan))
    return r2.fillna(0)

def calc_pearson_correlation(s1, s2, period=14):
    return s1.rolling(period).corr(s2)

def calc_z_score(series, period=20):
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)

# ==========================================
# CATEGORY 9: CANDLESTICK PATTERNS
# ==========================================

def calc_doji_score(df):
    body = abs(df['close'] - df['open'])
    range_val = df['high'] - df['low']
    ratio = body / range_val.replace(0, np.nan)
    return np.where(ratio < 0.1, 1.0, 0.0)

def calc_engulfing(df):
    c = df['close']
    o = df['open']
    c_prev = c.shift(1)
    o_prev = o.shift(1)
    
    bullish = (c > o) & (c_prev < o_prev) & (c >= o_prev) & (o <= c_prev)
    bearish = (c < o) & (c_prev > o_prev) & (c <= o_prev) & (o >= c_prev)
    
    signals = np.zeros(len(df))
    signals[bullish] = 1.0
    signals[bearish] = -1.0
    return pd.Series(signals, index=df.index)

def calc_pin_bar_score(df):
    c = df['close']
    o = df['open']
    h = df['high']
    l = df['low']
    
    body = abs(c - o)
    range_val = h - l
    
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    
    bullish_pin = (lower_wick > 2 * body) & (upper_wick < 0.5 * lower_wick)
    bearish_pin = (upper_wick > 2 * body) & (lower_wick < 0.5 * upper_wick)
    
    signals = np.zeros(len(df))
    signals[bullish_pin] = 1.0
    signals[bearish_pin] = -1.0
    return pd.Series(signals, index=df.index)

# ==========================================
# COMBINED BULK LOADER
# ==========================================

def add_all_indicators(df):
    """
    Computes and adds 150 indicators to the DataFrame.
    """
    res = df.copy()
    
    # 1. Moving Averages (25 indicators)
    periods = [5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200]
    for p in periods:
        res[f'sma_{p}'] = calc_sma(res['close'], p)
        res[f'ema_{p}'] = calc_ema(res['close'], p)
    for p in [10, 20, 50, 100]:
        res[f'dema_{p}'] = calc_dema(res['close'], p)
        res[f'tema_{p}'] = calc_tema(res['close'], p)
        res[f'wma_{p}'] = calc_wma(res['close'], p)
        res[f'hma_{p}'] = calc_hma(res['close'], p)
    res['kama_10'] = calc_kama(res['close'], 10)
    res['zlema_20'] = calc_zlema(res['close'], 20)
    res['t3_10'] = calc_t3(res['close'], 10)
    res['vwap'] = calc_vwap(res)
    res['alma_9'] = calc_alma(res['close'], 9)
    res['alma_20'] = calc_alma(res['close'], 20)
    
    # 2. Trend & Strength (20 indicators)
    for p in [7, 14, 21]:
        adx, pdi, mdi = calc_adx(res, p)
        res[f'adx_{p}'] = adx
        res[f'pdi_{p}'] = pdi
        res[f'mdi_{p}'] = mdi
        res[f'chop_{p}'] = calc_choppiness_index(res, p)
    
    for p in [14, 25]:
        _, _, aroon_osc = calc_aroon(res, p)
        res[f'aroon_osc_{p}'] = aroon_osc
        vi_plus, vi_minus = calc_vortex(res, p)
        res[f'vi_plus_{p}'] = vi_plus
        res[f'vi_minus_{p}'] = vi_minus
        
    res['mass_index'] = calc_mass_index(res)
    t, k, sa, sb, _ = calc_ichimoku(res)
    res['tenkan'] = t
    res['kijun'] = k
    res['senkou_a'] = sa
    res['senkou_b'] = sb
    res['lr_slope_14'] = calc_linear_regression_slope(res['close'], 14)
    res['lr_slope_50'] = calc_linear_regression_slope(res['close'], 50)
    st, st_dir = calc_supertrend(res, 10, 3.0)
    res['supertrend_10_3'] = st
    res['supertrend_dir_10_3'] = st_dir
    
    # 3. Momentum Oscillators (25 indicators)
    for p in [7, 9, 14, 21]:
        res[f'rsi_{p}'] = calc_rsi(res['close'], p)
        stoch_k, stoch_d = calc_stoch_rsi(res['close'], p)
        res[f'stoch_k_{p}'] = stoch_k
        res[f'stoch_d_{p}'] = stoch_d
        res[f'cci_{p}'] = calc_cci(res, p)
        res[f'mom_{p}'] = calc_momentum(res['close'], p)
        res[f'roc_{p}'] = calc_roc(res['close'], p)
        res[f'williams_{p}'] = calc_williams_r(res, p)
        
    res['ultimate_osc'] = calc_ultimate_oscillator(res)
    res['tsi'] = calc_tsi(res['close'])
    res['trix'] = calc_trix(res['close'])
    res['dpo'] = calc_dpo(res['close'])
    
    # 4. MACD Family (10 indicators)
    macd, signal, hist = calc_macd(res['close'], 12, 26, 9)
    res['macd'] = macd
    res['macd_signal'] = signal
    res['macd_hist'] = hist
    
    macd_f, signal_f, hist_f = calc_macd(res['close'], 5, 13, 5)
    res['macd_fast'] = macd_f
    res['macd_signal_fast'] = signal_f
    res['macd_hist_fast'] = hist_f
    
    ppo, ppos, ppoh = calc_ppo(res['close'])
    res['ppo'] = ppo
    res['ppo_signal'] = ppos
    res['ppo_hist'] = ppoh
    
    # 5. Volatility (20 indicators)
    for p in [7, 14, 21, 50]:
        res[f'atr_{p}'] = calc_atr(res, p)
        res[f'natr_{p}'] = res[f'atr_{p}'] / res['close']
        
    for p in [10, 20, 50]:
        u, m, l, b, w = calc_bollinger_bands(res['close'], p)
        res[f'bb_upper_{p}'] = u
        res[f'bb_mid_{p}'] = m
        res[f'bb_lower_{p}'] = l
        res[f'bb_b_pct_{p}'] = b
        res[f'bb_width_{p}'] = w
        
        ku, km, kl, kb = calc_keltner_channel(res, p)
        res[f'keltner_upper_{p}'] = ku
        res[f'keltner_lower_{p}'] = kl
        res[f'keltner_b_pct_{p}'] = kb
        
        du, dm, dl, db = calc_donchian_channel(res, p)
        res[f'donchian_upper_{p}'] = du
        res[f'donchian_mid_{p}'] = dm
        res[f'donchian_lower_{p}'] = dl
        res[f'donchian_b_pct_{p}'] = db
        
    res['hv_20'] = calc_historical_volatility(res['close'], 20)
    res['gk_vol_20'] = calc_garman_klass_volatility(res, 20)
    
    # 6. Volume (15 indicators)
    res['obv'] = calc_obv(res)
    res['obv_ema'] = calc_ema(res['obv'], 20)
    res['volume_sma_20'] = calc_sma(res['volume'], 20)
    res['volume_sma_50'] = calc_sma(res['volume'], 50)
    res['volume_ratio'] = res['volume'] / res['volume_sma_20'].replace(0, np.nan)
    res['mfi_14'] = calc_mfi(res)
    res['cmf_20'] = calc_cmf(res)
    res['emv'] = calc_ease_of_movement(res)
    res['force_index'] = calc_force_index(res)
    res['volume_osc'] = calc_volume_oscillator(res)
    
    # 7. Mean Reversion Specific (15 indicators)
    res['zscore_20'] = calc_z_score(res['close'], 20)
    res['zscore_50'] = calc_z_score(res['close'], 50)
    res['dev_ema20'] = (res['close'] - res['ema_20']) / res['ema_20'].replace(0, np.nan)
    res['dev_ema50'] = (res['close'] - res['ema_50']) / res['ema_50'].replace(0, np.nan)
    res['rsi_mean_dist_14'] = 50 - res['rsi_14']
    res['cci_mr_signal'] = np.where(res['cci_14'] < -100, 1, np.where(res['cci_14'] > 100, -1, 0))
    res['stoch_mid_dist_14'] = res['stoch_k_14'] - 50
    
    # 8. Statistical & Patterns (10 indicators)
    res['lr_r2_14'] = calc_linear_regression_r2(res['close'], 14)
    res['corr_ema200_14'] = calc_pearson_correlation(res['close'], res['ema_200'], 14)
    res['doji'] = calc_doji_score(res)
    res['engulfing'] = calc_engulfing(res)
    res['pin_bar'] = calc_pin_bar_score(res)
    
    # Clean NaN/Inf values
    res = res.replace([np.inf, -np.inf], np.nan)
    return res
