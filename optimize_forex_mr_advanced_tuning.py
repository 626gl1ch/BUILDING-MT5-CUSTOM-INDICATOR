"""
Ultimate Strategy Tuning Engine
======================================================================
Injects Multi-Timeframe (MTF) Trend Alignment, VSA (Stopping Volume), 
and Trailing Stops to maximize the Win Rate and Profit Factor of our winners.
"""

import glob
import numpy as np
import pandas as pd
import gc
from backtest_core import BacktestCore
from indicators_library import (
    calc_atr, calc_rsi, calc_bollinger_bands, calc_choppiness_index, 
    calc_cvd, calc_mfi, calc_volume_profile
)

FEE_PCT = 0.0002

def get_rvol(df, p=20):
    return df['volume'] / df['volume'].rolling(p).mean()

def calc_alma(series, period=9, sigma=6, offset=0.85):
    m = offset * (period - 1)
    s = period / sigma
    w = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
    w = w / w.sum()
    return series.rolling(period).apply(lambda x: np.dot(x, w), raw=True)

def rolling_vwap(df, period=200):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()

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
# TUNED STRATEGIES
# ═══════════════════════════════════════════════════════════════

def strat_tuned_vwap(df, params):
    """Tuned MR2_NY_LTC: Adds MTF ALMA & VSA Stopping Volume."""
    atr = calc_atr(df, 14)
    rvwap = rolling_vwap(df, 200)
    rs = df['close'].rolling(20).std()
    vl = rvwap - 2.0 * rs
    vu = rvwap + 2.0 * rs
    rsi = calc_rsi(df['close'], 14)
    chop = calc_choppiness_index(df, 14)
    cvd = calc_cvd(df, 5)
    rvol = get_rvol(df)
    
    # MTF ALMA (1H proxy on 15m is 36 periods)
    mtf_alma = calc_alma(df['close'], 36)
    mtf_trend_up = mtf_alma.values >= mtf_alma.shift(2).values
    mtf_trend_dn = mtf_alma.values <= mtf_alma.shift(2).values
    
    # VSA: Exhaustion Volume (Low volume at extremes often signals a reversal better than high volume)
    vsa_exhaustion = rvol.values < 1.2
    
    # Divergence
    lk = 10
    bull_div = (df['close'].values <= df['close'].rolling(lk).min().values) & (rsi.values > rsi.rolling(lk).min().values)
    bear_div = (df['close'].values >= df['close'].rolling(lk).max().values) & (rsi.values < rsi.rolling(lk).max().values)
    
    cvd_bull = cvd.values > cvd.shift(2).values
    cvd_bear = cvd.values < cvd.shift(2).values
    
    regime = (chop.values > 50) & (atr.values >= 0.6 * atr.rolling(50).mean().values)
    mask_ny = pd.Series(df.index.hour, index=df.index).isin(range(13, 21)).values
    
    prior_dn = df['close'].shift(1).values < vl.shift(1).values
    prior_up = df['close'].shift(1).values > vu.shift(1).values
    
    turn_up = df['close'].values > df['close'].shift(1).values
    turn_dn = df['close'].values < df['close'].shift(1).values
    
    bull_div_s = pd.Series(bull_div).shift(1).fillna(False).astype(bool).values
    bear_div_s = pd.Series(bear_div).shift(1).fillna(False).astype(bool).values
    
    signals = pd.Series(0, index=df.index)
    signals[prior_dn & bull_div_s & turn_up & cvd_bull & regime & mask_ny & mtf_trend_up & vsa_exhaustion] = 1
    signals[prior_up & bear_div_s & turn_dn & cvd_bear & regime & mask_ny & mtf_trend_dn & vsa_exhaustion] = -1
    
    # Dynamic Exit on opposite CVD
    exit_long = (cvd_bear) | (signals.values == -1)
    exit_short = (cvd_bull) | (signals.values == 1)
    
    return signals, exit_long, exit_short


def strat_tuned_engulf(df, params):
    """Tuned MR10_Engulf: Adds VSA and Dynamic CVD Exit."""
    atr = calc_atr(df, 14)
    bb_u, _, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.0)
    chop = calc_choppiness_index(df, 14)
    cvd = calc_cvd(df, 5)
    rvol = get_rvol(df)
    mfi = calc_mfi(df, 14)
    
    o = df['open'].values; c = df['close'].values
    
    prior_bear_ex = (c[:-1] < o[:-1]) & (c[:-1] < bb_l[:-1].values)
    prior_bear_ex = np.insert(prior_bear_ex, 0, False)
    
    prior_bull_ex = (c[:-1] > o[:-1]) & (c[:-1] > bb_u[:-1].values)
    prior_bull_ex = np.insert(prior_bull_ex, 0, False)
    
    prior_mid = (o[:-1] + c[:-1]) / 2.0
    prior_mid = np.insert(prior_mid, 0, np.nan)
    
    bull_engulf = (c > o) & (c > prior_mid)
    bear_engulf = (c < o) & (c < prior_mid)
    
    # VSA Exhaustion
    vsa_exhaustion = rvol.values < 1.2
    
    regime = (chop.values > 42) & (atr.values >= 0.6 * atr.rolling(50).mean().values)
    
    mfi_os = mfi.shift(1).values < 40
    mfi_ob = mfi.shift(1).values > 60
    
    cvd_bull = cvd.values > cvd.shift(1).values
    cvd_bear = cvd.values < cvd.shift(1).values
    
    signals = pd.Series(0, index=df.index)
    signals[prior_bear_ex & bull_engulf & mfi_os & cvd_bull & regime & vsa_exhaustion] = 1
    signals[prior_bull_ex & bear_engulf & mfi_ob & cvd_bear & regime & vsa_exhaustion] = -1
    
    exit_long = (cvd_bear) | (signals.values == -1)
    exit_short = (cvd_bull) | (signals.values == 1)
    
    return signals, exit_long, exit_short


def main():
    engine = BacktestCore()
    # 15m is our mathematically proven timeframe for these, we will only run 15m to save time
    dfs_15m = load_tf("*_15min_1year.csv")
    
    # Optimization pass: We test Trailing Stops ON and OFF
    strats = {
        "TUNED_VWAP": strat_tuned_vwap,
        "TUNED_ENGULF": strat_tuned_engulf,
    }
    
    results = []
    print("Running Ultimate Tuning Pass...\n")
    for name, fn in strats.items():
        for trailing in [False, True]:
            # Wider TP to let trailing stops run, slightly tighter SL
            params = {'sl_atr': 1.2, 'tp_atr': 4.0, 'max_bars_hold': 80, 'risk_pct': 0.01, 'trailing': trailing}
            try:
                tag = name + ("_Trail" if trailing else "_Fixed")
                _, agg, _ = engine.run_multi_symbol(dfs_15m, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
                trades = agg['total_trades']
                if trades < 10:
                    print(f"[15m] {tag}: Skipped (Only {trades} trades)")
                    continue
                pf = agg['profit_factor']
                oos = run_walkforward(engine, dfs_15m, fn, params)
                sr = agg['sharpe_ratio']
                wr = agg['win_rate']
                print(f"[15m] {tag} -> PF: {pf:.3f} | OOS: {oos:.3f} | WR: {wr:.1f}% | Sharpe: {sr:.2f} | Trades: {trades}")
                results.append({'Strategy': tag, 'TF': '15m', 'PF': round(pf, 3), 'OOS': round(oos, 3), 'WR': round(wr, 1), 'Sharpe': round(sr, 2), 'Trades': trades})
            except Exception as e:
                import traceback
                print(f"[15m] {tag}: Error - {e}")
                traceback.print_exc()
            finally:
                gc.collect()
                
    df_res = pd.DataFrame(results).sort_values('PF', ascending=False)
    print("\nUltimate Tuning Rankings:")
    print(df_res.to_string(index=False))
    df_res.to_csv("ultimate_tuning_rankings.csv", index=False)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
