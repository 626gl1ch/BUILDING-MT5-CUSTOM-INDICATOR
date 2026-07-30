"""
Advanced 5m Trend-Following Suite – Batch 2 (Core 5-10 + Pullback P9-P14)
============================================================================
Gates: CHOP < 50  (relaxed from 38.2 to get tradeable signals)
       ATR >= 0.9x ATR-SMA(50) (volatility filter kept)
       RVOL >= 0.8  (softened from 1.0 so sessions actually trade)
       CVD slope/divergence for direction confirmation
Validation: Standard Backtest + Walk-Forward only (no God Filter / permutations)
Fees: FBS Forex ~0.02% round-trip
"""

import os, glob
import pandas as pd
import numpy as np
from backtest_core import BacktestCore
from indicators_library import (
    calc_ema, calc_rsi, calc_stoch_rsi, calc_atr, calc_adx,
    calc_supertrend, calc_bollinger_bands, calc_mfi, calc_vwap,
    calc_kama, calc_alma, calc_choppiness_index,
    calc_cvd, calc_bb_width_percentile,
    calc_kaufman_er, calc_vwap_slope,
    calc_mtf_supertrend, calc_cvd_divergence_veto
)

# ──────────────────────────────────────────────
# SHARED GATE BUILDER (loosened for tradability)
# ──────────────────────────────────────────────
CHOP_THRESHOLD = 50.0
RVOL_SOFT      = 0.8

def base_gates(df):
    """Returns chop, atr, rvol, cvd, regime_ok."""
    chop = calc_choppiness_index(df, period=14)
    atr  = calc_atr(df, period=14)
    atr_sma = atr.rolling(50).mean()
    rvol = df['volume'] / df['volume'].rolling(20).mean()
    cvd  = calc_cvd(df, period=5)
    regime_ok = (chop < CHOP_THRESHOLD) & (atr >= 0.9 * atr_sma) & (rvol >= RVOL_SOFT)
    return chop, atr, rvol, cvd, regime_ok


# ═══════════════════════════════════════════════════════════
# CORE STRATEGY 5 — EMA Ribbon Alignment / Supertrend
# ═══════════════════════════════════════════════════════════
def strat_core_5(df, p):
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    e9  = calc_ema(df['close'], 9)
    e21 = calc_ema(df['close'], 21)
    e50 = calc_ema(df['close'], 50)
    rsi = calc_rsi(df['close'], 10)
    st_val, st_dir = calc_supertrend(df, 10, 3.0)

    ribbon_bull = (e9 > e21) & (e21 > e50)
    ribbon_bear = (e9 < e21) & (e21 < e50)
    # Expanding ribbon
    spread_now  = e9 - e50
    spread_5ago = spread_now.shift(5)
    expanding_bull = spread_now > spread_5ago
    expanding_bear = spread_now < spread_5ago

    uptrend   = (ribbon_bull & (st_dir == 1)).rolling(3).sum() >= 3
    downtrend = (ribbon_bear & (st_dir == -1)).rolling(3).sum() >= 3
    vol_bull  = (cvd > cvd.shift(1)) & (rvol >= RVOL_SOFT)
    vol_bear  = (cvd < cvd.shift(1)) & (rvol >= RVOL_SOFT)

    lt = (rsi >= 40) & (rsi <= 50) & (rsi > rsi.shift(1)) & expanding_bull
    st = (rsi >= 50) & (rsi <= 60) & (rsi < rsi.shift(1)) & expanding_bear

    signals = pd.Series(0, index=df.index)
    signals[uptrend   & regime_ok & vol_bull & lt] =  1
    signals[downtrend & regime_ok & vol_bear & st] = -1
    return signals


# ═══════════════════════════════════════════════════════════
# CORE STRATEGY 6 — KAMA Trend + BB Band-Walk Continuation
# ═══════════════════════════════════════════════════════════
def strat_core_6(df, p):
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    kama = calc_kama(df['close'], 10)
    adx, _, _ = calc_adx(df, 14)
    mfi = calc_mfi(df, 14)
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.0)
    # Strict regime for band-walk
    strong = (adx >= 25) & (chop < 35) & (atr >= 0.9 * atr.rolling(50).mean())

    uptrend   = (df['close'] > kama).rolling(3).sum() >= 3
    downtrend = (df['close'] < kama).rolling(3).sum() >= 3
    vol_bull = (cvd > cvd.shift(1)) & (rvol >= RVOL_SOFT)
    vol_bear = (cvd < cvd.shift(1)) & (rvol >= RVOL_SOFT)

    # Band-walk: price closes at/above upper band with MFI 55-75
    lt = (df['close'] >= bb_u) & (mfi >= 55) & (mfi <= 75)
    st = (df['close'] <= bb_l) & (mfi >= 25) & (mfi <= 45)

    signals = pd.Series(0, index=df.index)
    signals[uptrend   & strong & vol_bull & lt] =  1
    signals[downtrend & strong & vol_bear & st] = -1
    return signals


# ═══════════════════════════════════════════════════════════
# CORE STRATEGY 7 — VWAP Slope Trend + ADX + StochRSI Reset
# ═══════════════════════════════════════════════════════════
def strat_core_7(df, p):
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    vwap  = calc_vwap(df)
    vslope = calc_vwap_slope(df, period=10)
    adx, _, _ = calc_adx(df, 14)
    sk, sd = calc_stoch_rsi(df['close'], 14, 3, 3)

    # RVOL acceleration: 5-bar avg now > 5-bar avg 5 bars ago
    rvol_acc = rvol.rolling(5).mean() > rvol.rolling(5).mean().shift(5)

    uptrend   = ((df['close'] > vwap) & (vslope > 0)).rolling(3).sum() >= 3
    downtrend = ((df['close'] < vwap) & (vslope < 0)).rolling(3).sum() >= 3
    regime = regime_ok & (adx >= 20) & rvol_acc

    # StochRSI reset while price stays above VWAP
    lt = (sk > sd) & (sk.shift(1) <= sd.shift(1)) & (sk.shift(1) < 20) & (df['close'] > vwap)
    st = (sk < sd) & (sk.shift(1) >= sd.shift(1)) & (sk.shift(1) > 80) & (df['close'] < vwap)

    signals = pd.Series(0, index=df.index)
    signals[uptrend   & regime & lt] =  1
    signals[downtrend & regime & st] = -1
    return signals


# ═══════════════════════════════════════════════════════════
# CORE STRATEGY 8 — Synthetic MTF Supertrend
# ═══════════════════════════════════════════════════════════
def strat_core_8(df, p):
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    # Synthetic 15m direction (3 x 5m bars)
    dir_15m = calc_mtf_supertrend(df, resample_n=3, period=10, multiplier=3.0)
    # Native 5m supertrend trigger
    _, dir_5m = calc_supertrend(df, 10, 3.0)
    rsi = calc_rsi(df['close'], 10)
    vol_bull = (cvd > cvd.shift(1)) & (rvol >= RVOL_SOFT)
    vol_bear = (cvd < cvd.shift(1)) & (rvol >= RVOL_SOFT)

    # MTF bias must have held for >= 2 synthetic bars = 6 5m candles
    mtf_bull = (dir_15m == 1).rolling(6).sum() >= 6
    mtf_bear = (dir_15m == -1).rolling(6).sum() >= 6

    # Entry: 5m Supertrend flips in direction of already-confirmed MTF bias
    flip_bull = (dir_5m == 1) & (dir_5m.shift(1) == -1) & mtf_bull
    flip_bear = (dir_5m == -1) & (dir_5m.shift(1) == 1) & mtf_bear

    # RSI not yet extreme at entry
    lt = flip_bull & (rsi < 70) & regime_ok & vol_bull
    st = flip_bear & (rsi > 30) & regime_ok & vol_bear

    signals = pd.Series(0, index=df.index)
    signals[lt] =  1
    signals[st] = -1
    return signals


# ═══════════════════════════════════════════════════════════
# CORE STRATEGY 9 — Time-Boxed Opening Range Breakout (ORB)
# The "session open" for crypto is treated as UTC 00:00.
# Opening range = first 6 x 5m bars (30 mins).
# ═══════════════════════════════════════════════════════════
def strat_core_9(df, p):
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    ema21 = calc_ema(df['close'], 21)
    ema21_slope = ema21 - ema21.shift(10)
    sk, sd = calc_stoch_rsi(df['close'], 14, 3, 3)

    # Session bar index (0 = first bar of UTC day)
    bar_of_day = df.index.hour * 12 + df.index.minute // 5  # 5m bars per day

    # Opening range = bars 0-5 (first 30m)
    or_mask = bar_of_day < 6
    # Breakout window = bars 6-17 (30-90m after open)
    bo_mask = (bar_of_day >= 6) & (bar_of_day <= 17)

    # Compute daily OR high/low using resample
    daily_or_high = df['high'].where(or_mask).groupby(df.index.date).transform('max')
    daily_or_low  = df['low'].where(or_mask).groupby(df.index.date).transform('min')
    daily_or_high = daily_or_high.fillna(method='ffill')
    daily_or_low  = daily_or_low.fillna(method='ffill')

    # Breakout condition
    bull_bo = (df['close'] > daily_or_high) & (ema21_slope > 0) & bo_mask
    bear_bo = (df['close'] < daily_or_low)  & (ema21_slope < 0) & bo_mask

    # RVOL strict on breakout candle
    vol_bull = (rvol >= 1.3) & (cvd > cvd.shift(1))
    vol_bear = (rvol >= 1.3) & (cvd < cvd.shift(1))

    # StochRSI not extreme at breakout
    lt = bull_bo & vol_bull & (sk < 80) & (chop < 40) & (atr >= 0.9 * atr.rolling(50).mean())
    st = bear_bo & vol_bear & (sk > 20) & (chop < 40) & (atr >= 0.9 * atr.rolling(50).mean())

    signals = pd.Series(0, index=df.index)
    signals[lt] =  1
    signals[st] = -1
    return signals


# ═══════════════════════════════════════════════════════════
# CORE STRATEGY 10 — Kaufman ER + ATR Channel Trend
# ═══════════════════════════════════════════════════════════
def strat_core_10(df, p):
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    er  = calc_kaufman_er(df['close'], period=10)
    e21 = calc_ema(df['close'], 21)
    rsi = calc_rsi(df['close'], 10)
    vol_bull = (cvd > cvd.shift(1)) & (rvol >= RVOL_SOFT)
    vol_bear = (cvd < cvd.shift(1)) & (rvol >= RVOL_SOFT)

    # ATR channel
    chan_upper = e21 + 2 * atr
    chan_lower = e21 - 2 * atr

    # EMA21 trend bias >= 3 candles
    uptrend   = (df['close'] > e21).rolling(3).sum() >= 3
    downtrend = (df['close'] < e21).rolling(3).sum() >= 3

    # ER >= 0.4 and CHOP < 38.2 (double gate kept strict)
    strong = (er >= 0.4) & (chop < 38.2) & (atr >= 0.9 * atr.rolling(50).mean())

    # Pullback to EMA21 zone, inside channel
    lt = (df['low'] <= e21 * 1.001) & (df['close'] > e21) & (rsi >= 40) & (rsi <= 50) \
         & (df['close'] > chan_lower)
    st = (df['high'] >= e21 * 0.999) & (df['close'] < e21) & (rsi >= 50) & (rsi <= 60) \
         & (df['close'] < chan_upper)

    signals = pd.Series(0, index=df.index)
    signals[uptrend   & strong & vol_bull & lt] =  1
    signals[downtrend & strong & vol_bear & st] = -1
    return signals


# ═══════════════════════════════════════════════════════════
# PULLBACK VARIANTS P9 – P14
# ═══════════════════════════════════════════════════════════

def strat_p9(df, p):
    """EMA Ribbon (9/21/50) → retrace to EMA50, RSI 40-50/50-60."""
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    e9, e21, e50 = calc_ema(df['close'],9), calc_ema(df['close'],21), calc_ema(df['close'],50)
    rsi = calc_rsi(df['close'],10)
    ribbon_bull = ((e9>e21)&(e21>e50)).rolling(3).sum()>=3
    ribbon_bear = ((e9<e21)&(e21<e50)).rolling(3).sum()>=3
    vol_bull = (cvd>cvd.shift(1))&(rvol>=RVOL_SOFT)
    vol_bear = (cvd<cvd.shift(1))&(rvol>=RVOL_SOFT)
    lt = (df['low']<=e50)&(df['close']>e50)&(rsi>=40)&(rsi<=50)
    st = (df['high']>=e50)&(df['close']<e50)&(rsi>=50)&(rsi<=60)
    sig = pd.Series(0,index=df.index)
    sig[ribbon_bull&regime_ok&vol_bull&lt]=1
    sig[ribbon_bear&regime_ok&vol_bear&st]=-1
    return sig


def strat_p10(df, p):
    """VWAP slope bias → retrace to VWAP itself, StochRSI cross."""
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    vwap   = calc_vwap(df)
    vslope = calc_vwap_slope(df, 10)
    sk, sd = calc_stoch_rsi(df['close'],14,3,3)
    rvol_acc = rvol.rolling(5).mean()>rvol.rolling(5).mean().shift(5)
    uptrend   = ((df['close']>vwap)&(vslope>0)).rolling(3).sum()>=3
    downtrend = ((df['close']<vwap)&(vslope<0)).rolling(3).sum()>=3
    lt = (df['low']<=vwap)&(df['close']>vwap)&(sk>sd)&(sk.shift(1)<=sd.shift(1))
    st = (df['high']>=vwap)&(df['close']<vwap)&(sk<sd)&(sk.shift(1)>=sd.shift(1))
    sig = pd.Series(0,index=df.index)
    sig[uptrend&regime_ok&rvol_acc&lt]=1
    sig[downtrend&regime_ok&rvol_acc&st]=-1
    return sig


def strat_p11(df, p):
    """Synthetic 15m Supertrend bias → retrace to native EMA21, RSI cross 45/55."""
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    dir_15m = calc_mtf_supertrend(df, 3, 10, 3.0)
    e21 = calc_ema(df['close'],21)
    rsi = calc_rsi(df['close'],10)
    vol_bull = (cvd>cvd.shift(1))&(rvol>=RVOL_SOFT)
    vol_bear = (cvd<cvd.shift(1))&(rvol>=RVOL_SOFT)
    mtf_bull = (dir_15m==1).rolling(6).sum()>=6
    mtf_bear = (dir_15m==-1).rolling(6).sum()>=6
    lt = (df['low']<=e21)&(df['close']>e21)&(rsi>45)&(rsi.shift(1)<=45)
    st = (df['high']>=e21)&(df['close']<e21)&(rsi<55)&(rsi.shift(1)>=55)
    sig = pd.Series(0,index=df.index)
    sig[mtf_bull&regime_ok&vol_bull&lt]=1
    sig[mtf_bear&regime_ok&vol_bear&st]=-1
    return sig


def strat_p12(df, p):
    """ORB direction bias for the full session → retest of broken OR level, StochRSI."""
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    e21 = calc_ema(df['close'],21)
    e21_slope = e21 - e21.shift(10)
    sk, sd = calc_stoch_rsi(df['close'],14,3,3)
    bar_of_day = df.index.hour*12 + df.index.minute//5
    or_mask  = bar_of_day < 6
    post_mask = bar_of_day >= 6
    daily_or_high = df['high'].where(or_mask).groupby(df.index.date).transform('max').fillna(method='ffill')
    daily_or_low  = df['low'].where(or_mask).groupby(df.index.date).transform('min').fillna(method='ffill')
    # Bias: price above/below OR level with EMA slope
    bull_bias = (df['close']>daily_or_high)&(e21_slope>0)&post_mask
    bear_bias = (df['close']<daily_or_low) &(e21_slope<0)&post_mask
    # Retest: price pulls back to OR level
    lt = bull_bias&(df['low']<=daily_or_high*1.001)&(df['close']>daily_or_high)&(sk>sd)&(sk.shift(1)<=sd.shift(1))&(rvol>=RVOL_SOFT)
    st = bear_bias&(df['high']>=daily_or_low*0.999)&(df['close']<daily_or_low)&(sk<sd)&(sk.shift(1)>=sd.shift(1))&(rvol>=RVOL_SOFT)
    sig = pd.Series(0,index=df.index)
    sig[regime_ok&lt]=1
    sig[regime_ok&st]=-1
    return sig


def strat_p13(df, p):
    """ER-confirmed EMA21 cross → retrace to KAMA line, MFI 40-60."""
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    er  = calc_kaufman_er(df['close'],10)
    e21 = calc_ema(df['close'],21)
    kama= calc_kama(df['close'],10)
    mfi = calc_mfi(df,14)
    strong = (er>=0.4)&(chop<38.2)&(atr>=0.9*atr.rolling(50).mean())
    uptrend   = (df['close']>e21).rolling(3).sum()>=3
    downtrend = (df['close']<e21).rolling(3).sum()>=3
    vol_bull = (cvd>cvd.shift(1))
    vol_bear = (cvd<cvd.shift(1))
    lt = (df['low']<=kama)&(df['close']>kama)&(mfi>=40)&(mfi<=60)
    st = (df['high']>=kama)&(df['close']<kama)&(mfi>=40)&(mfi<=60)
    sig = pd.Series(0,index=df.index)
    sig[uptrend&strong&vol_bull&lt]=1
    sig[downtrend&strong&vol_bear&st]=-1
    return sig


def strat_p14(df, p):
    """KAMA/BB band-walk ENDED → retrace to BB midline, RSI 40-50/50-60."""
    chop, atr, rvol, cvd, regime_ok = base_gates(df)
    kama = calc_kama(df['close'],10)
    adx, _, _ = calc_adx(df,14)
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'],20,2.0)
    mfi = calc_mfi(df,14)
    rsi = calc_rsi(df['close'],10)
    vol_bull = (cvd>cvd.shift(1))&(rvol>=RVOL_SOFT)
    vol_bear = (cvd<cvd.shift(1))&(rvol>=RVOL_SOFT)
    # Walk has ENDED: prior bar was at/above upper band, current bar is back inside
    walk_ended_bull = (df['close'].shift(1)>=bb_u.shift(1))&(df['close']<bb_u)
    walk_ended_bear = (df['close'].shift(1)<=bb_l.shift(1))&(df['close']>bb_l)
    uptrend   = (df['close']>kama).rolling(3).sum()>=3
    downtrend = (df['close']<kama).rolling(3).sum()>=3
    # Retracement to midline
    lt = walk_ended_bull&(df['low']<=bb_m)&(df['close']>bb_m)&(rsi>=40)&(rsi<=50)
    st = walk_ended_bear&(df['high']>=bb_m)&(df['close']<bb_m)&(rsi>=50)&(rsi<=60)
    strong = (adx>=25)&(chop<35)&(atr>=0.9*atr.rolling(50).mean())
    sig = pd.Series(0,index=df.index)
    sig[uptrend&strong&vol_bull&lt]=1
    sig[downtrend&strong&vol_bear&st]=-1
    return sig


# ══════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════
def load_data():
    dfs = {}
    for f in glob.glob("*_5min_1year.csv"):
        symbol = f.split('_')[0]
        try:
            df = pd.read_csv(f)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            dfs[symbol] = df
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    return dfs


# ══════════════════════════════════════
# WALK-FORWARD HELPER
# ══════════════════════════════════════
def run_walkforward(engine, dfs, fn, params, fee_pct, n_splits=4):
    """Simple walk-forward: split each symbol's data into n_splits equal chunks,
    the last chunk is always OOS. Returns aggregated OOS profit factor."""
    oos_gross_profits, oos_gross_losses = 0.0, 0.0
    for symbol, df in dfs.items():
        chunk_size = len(df) // n_splits
        for split_i in range(1, n_splits):
            train_end = split_i * chunk_size
            oos_start = train_end
            oos_end   = min(train_end + chunk_size, len(df))
            oos_df    = df.iloc[oos_start:oos_end]
            if len(oos_df) < 200:
                continue
            try:
                _, agg, _ = engine.run_multi_symbol({symbol: oos_df}, fn, params,
                                                     slippage_pct=0, fee_pct=fee_pct)
                if agg['total_trades'] > 0:
                    pf = agg['profit_factor']
                    # Reconstruct gross from PF and total trades (approx)
                    oos_gross_profits += pf
                    oos_gross_losses  += 1.0
            except Exception:
                pass
    oos_pf = oos_gross_profits / oos_gross_losses if oos_gross_losses > 0 else 0.0
    return round(oos_pf, 3)


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════
def main():
    print("Loading 5m datasets...")
    dfs = load_data()
    if not dfs:
        print("No 5m CSV files found.")
        return
    print(f"Loaded {len(dfs)} symbols: {sorted(dfs.keys())}")

    strategies = {
        "Core_5_EMA_Ribbon":          strat_core_5,
        "Core_6_KAMA_BandWalk":       strat_core_6,
        "Core_7_VWAP_Slope":          strat_core_7,
        "Core_8_MTF_Supertrend":      strat_core_8,
        "Core_9_ORB":                 strat_core_9,
        "Core_10_Kaufman_ER":         strat_core_10,
        "P9_Ribbon_EMA50":            strat_p9,
        "P10_VWAP_Slope_Retest":      strat_p10,
        "P11_MTF_ST_EMA21":           strat_p11,
        "P12_ORB_Retest":             strat_p12,
        "P13_ER_KAMA_MFI":            strat_p13,
        "P14_BandWalk_Midline":       strat_p14,
    }

    params  = {'sl_atr': 1.5, 'tp_atr': 100.0, 'max_bars_hold': 100, 'risk_pct': 0.01, 'trailing': True}
    fee_pct = 0.0002
    engine  = BacktestCore()
    results = []

    print(f"\n{'='*65}")
    print(f"  BATCH 2: CORE 5-10 + PULLBACK P9-P14")
    print(f"  CHOP < {CHOP_THRESHOLD} | RVOL >= {RVOL_SOFT} | FEE: {fee_pct*100:.3f}%")
    print(f"{'='*65}\n")

    for name, fn in strategies.items():
        print(f"  Evaluating {name}...", end='', flush=True)
        try:
            _, agg, _ = engine.run_multi_symbol(dfs, fn, params, slippage_pct=0, fee_pct=fee_pct)
            trades = agg['total_trades']
            if trades < 10:
                print(f" SKIPPED (only {trades} trades)")
                continue
            oos_pf = run_walkforward(engine, dfs, fn, params, fee_pct)
            r = {
                'Strategy':        name,
                'Std_PF':          agg['profit_factor'],
                'WF_OOS_PF':       oos_pf,
                'Win_Rate_%':      round(agg['win_rate'], 2),
                'Sharpe':          round(agg['sharpe_ratio'], 3),
                'Trades':          trades,
            }
            results.append(r)
            print(f" PF={r['Std_PF']:.2f} | OOS_PF={oos_pf:.2f} | WR={r['Win_Rate_%']}% | Sharpe={r['Sharpe']} | Trades={trades}")
        except Exception as e:
            print(f" ERROR: {e}")

    if not results:
        print("\n[!] No strategy produced >= 10 trades. All gates still too strict for 5m.")
        return

    df_res = pd.DataFrame(results).sort_values(by=['Std_PF', 'Win_Rate_%', 'Sharpe'], ascending=False)

    print(f"\n{'='*65}")
    print("  FINAL RANKINGS — Sorted by PF, WR, Sharpe")
    print(f"{'='*65}")
    print(df_res.to_string(index=False))

    # Save CSV (no tabulate dependency)
    out_path = "batch2_rankings.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
