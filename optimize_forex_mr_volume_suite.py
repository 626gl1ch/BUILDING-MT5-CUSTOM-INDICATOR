"""
Massive Volume Profile & Fast-Indicator MR Suite
======================================================================
Tests 5m and 15m Forex OHLCV data using a custom Volume Profile engine
to approximate institutional High Volume Nodes (HVNs) and Point of Control (POC).

Includes dynamic exit rules (e.g., exiting at the POC or on opposite signals)
to maximize trade frequency for scalping.
"""

import glob
import numpy as np
import pandas as pd
import gc
from backtest_core import BacktestCore
from indicators_library import (
    calc_ema, calc_rsi, calc_stoch_rsi, calc_atr, calc_mfi,
    calc_bollinger_bands, calc_vwap, calc_choppiness_index,
    calc_cvd, calc_adx, calc_macd,
    calc_volume_profile, calc_session_volume_profile
)

FEE_PCT = 0.0002

def calc_alma(series, period=9, sigma=6, offset=0.85):
    m = offset * (period - 1)
    s = period / sigma
    w = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
    w = w / w.sum()
    return series.rolling(period).apply(lambda x: np.dot(x, w), raw=True)

def load_tf(pattern):
    dfs = {}
    for f in glob.glob(pattern):
        symbol = f.split('_')[0]
        try:
            df = pd.read_csv(f)
            df.columns = [str(c).lower().strip() for c in df.columns]
            if 'datetime' not in df.columns and 'time' in df.columns:
                df.rename(columns={'time': 'datetime'}, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume', 'tick_volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.ffill(inplace=True)
            dfs[symbol] = df
        except Exception as e: pass
    return dfs

def run_walkforward(engine, dfs, fn, params, n_splits=5):
    all_pf = []
    for symbol, df in dfs.items():
        n = len(df)
        chunk = n // n_splits
        for i in range(1, n_splits):
            oos_df = df.iloc[i * chunk: (i + 1) * chunk]
            if len(oos_df) < 100: continue
            try:
                _, agg, _ = engine.run_multi_symbol({symbol: oos_df}, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
                if agg['total_trades'] >= 3:
                    all_pf.append(agg['profit_factor'])
            except Exception: pass
    return round(np.mean(all_pf), 3) if all_pf else 0.0

# ═══════════════════════════════════════════════════════════════
# VOLUME PROFILE SCALPING STRATEGIES
# ═══════════════════════════════════════════════════════════════

def strat_alma_poc(df, params):
    """
    1. ALMA + POC Reversion
    Price deviates from ALMA(9) but is pulled back toward the Session Point of Control (POC).
    Exit: When price touches the POC or opposite signal.
    """
    atr = calc_atr(df, 14)
    alma = calc_alma(df['close'], 9)
    spoc = calc_session_volume_profile(df, bins=50)
    stoch_k, stoch_d = calc_stoch_rsi(df, 9, 3, 3)
    chop = calc_choppiness_index(df, 14)
    
    # Deviation from ALMA
    dev_below = (df['close'].values < alma.values - 1.5 * atr.values) & (df['close'].values < spoc.values)
    dev_above = (df['close'].values > alma.values + 1.5 * atr.values) & (df['close'].values > spoc.values)
    
    # StochRSI Reversal
    turn_up = (stoch_k.values > stoch_k.shift(1).values) & (stoch_k.shift(1).values < 20)
    turn_dn = (stoch_k.values < stoch_k.shift(1).values) & (stoch_k.shift(1).values > 80)
    
    # Regimes
    vol_ok = atr.values >= 0.6 * atr.rolling(50).mean().values
    chop_ok = chop.values > 45
    
    signals = pd.Series(0, index=df.index)
    signals[dev_below & turn_up & vol_ok & chop_ok] = 1
    signals[dev_above & turn_dn & vol_ok & chop_ok] = -1
    
    # Dynamic Exits: Price reaches POC
    exit_long = (df['high'].values >= spoc.values) | (signals.values == -1)
    exit_short = (df['low'].values <= spoc.values) | (signals.values == 1)
    
    return signals, exit_long, exit_short


def strat_bb_hvn(df, params):
    """
    2. Bollinger Band + HVN Rejection
    Price spikes outside the BB(2.0) and rejects back toward the rolling Visible Range POC.
    Fast MACD histogram confirms momentum loss.
    """
    atr = calc_atr(df, 14)
    bb_u, _, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.0)
    vpoc = calc_volume_profile(df, lookback=100, bins=50)
    macd_l, macd_s, macd_h = calc_macd(df['close'], 5, 13, 4)
    adx_val, _, _ = calc_adx(df, 7)
    
    # BB Spike
    outside_dn = df['close'].values < bb_l.values
    outside_up = df['close'].values > bb_u.values
    
    # MACD Turn
    macd_up = (macd_h.values > macd_h.shift(1).values) & (macd_h.values < 0)
    macd_dn = (macd_h.values < macd_h.shift(1).values) & (macd_h.values > 0)
    
    # Regime: No massive trend
    adx_ok = adx_val.values < 25
    vol_ok = atr.values >= 0.6 * atr.rolling(50).mean().values
    
    signals = pd.Series(0, index=df.index)
    signals[outside_dn & macd_up & adx_ok & vol_ok] = 1
    signals[outside_up & macd_dn & adx_ok & vol_ok] = -1
    
    # Dynamic Exits: Price reverts to the rolling VPOC
    exit_long = (df['high'].values >= vpoc.values) | (signals.values == -1)
    exit_short = (df['low'].values <= vpoc.values) | (signals.values == 1)
    
    return signals, exit_long, exit_short


def strat_adx_vp_shift(df, params):
    """
    3. ADX Exhaustion + Volume Profile Shift
    Fast ADX(7) collapses, indicating trend is dead. 
    Price reverts to the nearest dense volume cluster (Session POC).
    """
    atr = calc_atr(df, 14)
    adx_val, _, _ = calc_adx(df, 7)
    spoc = calc_session_volume_profile(df, bins=50)
    rsi = calc_rsi(df['close'], 9)
    chop = calc_choppiness_index(df, 14)
    
    # ADX Collapse
    adx_collapse = (adx_val.values < adx_val.shift(2).values - 5) & (adx_val.shift(2).values > 25)
    
    # Stretch from SPOC
    stretch_dn = df['close'].values < (spoc.values - 1.0 * atr.values)
    stretch_up = df['close'].values > (spoc.values + 1.0 * atr.values)
    
    # RSI Balance
    rsi_ok_long = (rsi.values > 30) & (rsi.values < 50) & (rsi.values > rsi.shift(1).values)
    rsi_ok_short = (rsi.values < 70) & (rsi.values > 50) & (rsi.values < rsi.shift(1).values)
    
    vol_ok = atr.values >= 0.6 * atr.rolling(50).mean().values
    chop_ok = chop.values > 40
    
    signals = pd.Series(0, index=df.index)
    signals[adx_collapse & stretch_dn & rsi_ok_long & vol_ok & chop_ok] = 1
    signals[adx_collapse & stretch_up & rsi_ok_short & vol_ok & chop_ok] = -1
    
    # Exits: Revert to SPOC
    exit_long = (df['high'].values >= spoc.values) | (signals.values == -1)
    exit_short = (df['low'].values <= spoc.values) | (signals.values == 1)
    
    return signals, exit_long, exit_short

def strat_ema_alma_conv(df, params):
    """
    4. EMA9 vs ALMA9 Convergence
    Fast EMAs cross back toward the POC after a liquidity sweep (stoch extremes).
    """
    atr = calc_atr(df, 14)
    ema9 = calc_ema(df['close'], 9)
    alma9 = calc_alma(df['close'], 9)
    vpoc = calc_volume_profile(df, lookback=50, bins=24) # Faster 50-bar rolling POC
    stoch_k, stoch_d = calc_stoch_rsi(df, 14, 3, 3)
    chop = calc_choppiness_index(df, 14)
    
    # Convergence Signal
    bull_cross = (ema9.values > alma9.values) & (ema9.shift(1).values <= alma9.shift(1).values)
    bear_cross = (ema9.values < alma9.values) & (ema9.shift(1).values >= alma9.shift(1).values)
    
    # Below/Above POC
    below_poc = df['close'].values < vpoc.values
    above_poc = df['close'].values > vpoc.values
    
    # Oversold/Overbought confirmation from recent past
    was_os = stoch_k.rolling(5).min().values < 20
    was_ob = stoch_k.rolling(5).max().values > 80
    
    vol_ok = atr.values >= 0.6 * atr.rolling(50).mean().values
    chop_ok = chop.values > 45
    
    signals = pd.Series(0, index=df.index)
    signals[bull_cross & below_poc & was_os & vol_ok & chop_ok] = 1
    signals[bear_cross & above_poc & was_ob & vol_ok & chop_ok] = -1
    
    exit_long = (df['high'].values >= vpoc.values) | (signals.values == -1)
    exit_short = (df['low'].values <= vpoc.values) | (signals.values == 1)
    
    return signals, exit_long, exit_short


def main():
    engine = BacktestCore()
    dfs_5m  = load_tf("*_5min_1year.csv")
    dfs_15m = load_tf("*_15min_1year.csv")
    
    params = {'sl_atr': 1.5, 'tp_atr': 3.0, 'max_bars_hold': 40, 'risk_pct': 0.01, 'trailing': False}
    
    strats = {
        "VP1_ALMA_SPOC": strat_alma_poc,
        "VP2_BB_VPOC": strat_bb_hvn,
        "VP3_ADX_SPOC": strat_adx_vp_shift,
        "VP4_EMA_ALMA_VPOC": strat_ema_alma_conv,
    }
    
    results = []
    print("Running Massive Volume Profile Suite...\n")
    for name, fn in strats.items():
        for tf, dfs in [("5m", dfs_5m), ("15m", dfs_15m)]:
            try:
                _, agg, _ = engine.run_multi_symbol(dfs, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
                trades = agg['total_trades']
                if trades < 10:
                    print(f"[{tf}] {name}: Skipped (Only {trades} trades)")
                    continue
                pf = agg['profit_factor']
                oos = run_walkforward(engine, dfs, fn, params)
                print(f"[{tf}] {name} -> PF: {pf:.3f} | OOS: {oos:.3f} | WR: {agg['win_rate']:.1f}% | Trades: {trades}")
                results.append({'Strategy': name, 'TF': tf, 'PF': round(pf, 3), 'OOS': round(oos, 3), 'WR': round(agg['win_rate'], 1), 'Trades': trades})
            except Exception as e:
                import traceback
                print(f"[{tf}] {name}: Error - {e}")
                traceback.print_exc()
            finally:
                gc.collect()
                
    df_res = pd.DataFrame(results).sort_values('PF', ascending=False)
    print("\nMaster Volume Profile Rankings:")
    print(df_res.to_string(index=False))
    df_res.to_csv("volume_profile_rankings.csv", index=False)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
