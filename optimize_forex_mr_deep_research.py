"""
Advanced MR Research Lab
======================================================================
Goal: Keep learning and find strategies that drastically outperform 
the current winners (MR2 NY, MR10 Engulf, FX_MR6 ADX Collapse).

Method: Formulate and test 3 highly advanced, logically sound hypotheses
combining the best elements of our winning setups, without resorting 
to brute-force random permutation testing.
"""

import glob
import numpy as np
import pandas as pd
from backtest_core import BacktestCore
from indicators_library import (
    calc_ema, calc_rsi, calc_stoch_rsi, calc_atr, calc_mfi,
    calc_bollinger_bands, calc_vwap, calc_choppiness_index,
    calc_cvd, calc_adx, calc_macd
)

FEE_PCT = 0.0002

def get_rvol(df, p=20):
    return df['volume'] / df['volume'].rolling(p).mean()

def calc_keltner_channels(df, n=20, atr_n=10, mult=2.0):
    ema = calc_ema(df['close'], n)
    atr = calc_atr(df, atr_n)
    upper = ema + (mult * atr)
    lower = ema - (mult * atr)
    return upper, ema, lower

def calc_alma(series, period=9, sigma=6, offset=0.85):
    m = offset * (period - 1)
    s = period / sigma
    w = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
    w = w / w.sum()
    return series.rolling(period).apply(lambda x: np.dot(x, w), raw=True)

def rolling_vwap(df, period=200):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()

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

def load_tf(pattern):
    dfs = {}
    for f in glob.glob(pattern):
        symbol = f.split('_')[0]
        try:
            df = pd.read_csv(f)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            dfs[symbol] = df
        except Exception as e: pass
    return dfs

# ═══════════════════════════════════════════════════════════════
# ADVANCED HYPOTHESES
# ═══════════════════════════════════════════════════════════════

def strat_adv1_keltner_mfi_div(df, p):
    """
    Hypothesis 1: Bollinger Bands are too volatile on 5m. Keltner Channels (ATR based) 
    provide structurally sounder support/resistance in choppy markets.
    Combine KC 2.5 deviation + MFI Divergence.
    """
    atr = calc_atr(df, 14)
    kc_u, kc_m, kc_l = calc_keltner_channels(df, 20, 14, 2.5)
    mfi = calc_mfi(df, 14)
    chop = calc_choppiness_index(df, 14)
    cvd = calc_cvd(df, 5)
    rvol = get_rvol(df)
    
    # Keltner Extreme
    at_kc_low = df['close'] < kc_l
    at_kc_high = df['close'] > kc_u
    
    # MFI Divergence (Lookback 10)
    lk = 10
    bull_div = at_kc_low & (df['close'] <= df['close'].rolling(lk).min()) & (mfi > mfi.rolling(lk).min())
    bear_div = at_kc_high & (df['close'] >= df['close'].rolling(lk).max()) & (mfi < mfi.rolling(lk).max())
    
    # Entry criteria
    regime = (chop > 48) & (atr >= 0.5 * atr.rolling(50).mean())
    vol_confirm = (rvol > 0.8)
    
    cvd_bull = cvd > cvd.shift(1)
    cvd_bear = cvd < cvd.shift(1)
    
    turn_up = (df['close'] > df['close'].shift(1))
    turn_dn = (df['close'] < df['close'].shift(1))
    
    signals = pd.Series(0, index=df.index)
    signals[bull_div.shift(1) & turn_up & regime & vol_confirm & cvd_bull] = 1
    signals[bear_div.shift(1) & turn_dn & regime & vol_confirm & cvd_bear] = -1
    return signals


def strat_adv2_alma_adx_snapback(df, p):
    """
    Hypothesis 2: ADX Collapse worked exceptionally well. Let's combine it with ALMA(9) deviation.
    When price snaps far from ALMA during an ADX collapse, the reversion should be near 100% reliable.
    """
    atr = calc_atr(df, 7)
    adx_val, _, _ = calc_adx(df, 7)
    alma = calc_alma(df['close'], 9)
    chop = calc_choppiness_index(df, 14)
    macd_line, macd_sig, macd_hist = calc_macd(df['close'], 5, 13, 4)
    
    # ADX Collapse
    adx_collapse = (adx_val < adx_val.shift(2) - 4) & (adx_val.shift(2) > 25)
    
    # ALMA Extreme Deviation
    alma_dev = (df['close'] - alma) / atr
    dev_below = alma_dev < -2.0
    dev_above = alma_dev > 2.0
    
    # MACD turn
    macd_up = (macd_hist > macd_hist.shift(1)) & (macd_hist < 0)
    macd_dn = (macd_hist < macd_hist.shift(1)) & (macd_hist > 0)
    
    regime = (chop > 45)
    
    signals = pd.Series(0, index=df.index)
    signals[adx_collapse & dev_below & macd_up & regime] = 1
    signals[adx_collapse & dev_above & macd_dn & regime] = -1
    return signals


def strat_adv3_vwap_ny_engulfing(df, p):
    """
    Hypothesis 3: Combine our top two absolute winners:
    MR2_NY (VWAP Divergence) + MR10_Engulf (Engulfing Confirmation).
    Trade only NY session, at VWAP 2.0 standard deviation, with a strict engulfing bar.
    """
    atr = calc_atr(df, 14)
    rvwap = rolling_vwap(df, 200)
    chop = calc_choppiness_index(df, 14)
    rvol = get_rvol(df)
    
    rs = df['close'].rolling(20).std()
    vl = rvwap - 2.0 * rs
    vu = rvwap + 2.0 * rs
    
    o = df['open']; c = df['close']
    
    # Setup at extreme
    prior_bear_low = (c.shift(1) < o.shift(1)) & (c.shift(1) < vl.shift(1))
    prior_bull_hi  = (c.shift(1) > o.shift(1)) & (c.shift(1) > vu.shift(1))
    
    # Strict Engulfing
    bull_engulf = (c > o) & (c >= o.shift(1)) & (o <= c.shift(1))
    bear_engulf = (c < o) & (c <= o.shift(1)) & (o >= c.shift(1))
    
    regime = (chop > 45) & (atr >= 0.6 * atr.rolling(50).mean())
    mask_ny = pd.Series(df.index.hour, index=df.index).isin(range(13, 21))
    vol_ok = rvol > 0.8
    
    signals = pd.Series(0, index=df.index)
    signals[prior_bear_low & bull_engulf & regime & mask_ny & vol_ok] = 1
    signals[prior_bull_hi  & bear_engulf & regime & mask_ny & vol_ok] = -1
    return signals


def main():
    engine = BacktestCore()
    dfs_5m  = load_tf("*_5min_1year.csv")
    dfs_15m = load_tf("*_15min_1year.csv")
    
    params = {'sl_atr': 1.5, 'tp_atr': 2.5, 'max_bars_hold': 60, 'risk_pct': 0.01, 'trailing': False}
    
    strats = {
        "ADV1_Keltner_MFI_Div": strat_adv1_keltner_mfi_div,
        "ADV2_ALMA_ADX_Snapback": strat_adv2_alma_adx_snapback,
        "ADV3_VWAP_NY_Engulfing": strat_adv3_vwap_ny_engulfing,
    }
    
    results = []
    print("Running Deep Advanced Research...\n")
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
                results.append({'Strategy': name, 'TF': tf, 'PF': round(pf, 3), 'OOS': round(oos, 3), 'WR': round(agg['win_rate'], 1)})
            except Exception as e:
                import traceback
                print(f"[{tf}] {name}: Error - {e}")
                traceback.print_exc()
                
    df_res = pd.DataFrame(results).sort_values('PF', ascending=False)
    df_res.to_csv("advanced_deep_research_results.csv", index=False)
    print("\nAdvanced research saved.")

if __name__ == "__main__":
    main()
