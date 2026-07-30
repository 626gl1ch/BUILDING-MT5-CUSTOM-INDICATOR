"""
15m Validation — Test Proven 5m Winners + Full Forex MR Suite
==============================================================
Winning strategies from 5m research (with full PASS):
  1. MR10_Relaxed_LTC: Engulf Reversal at BB extreme (LTCUSDT) — PF=1.02, OOS=1.038
  2. MR2_LTC_NY: VWAP Divergence (LTCUSDT, NY session) — PF=1.00, OOS=1.053

Test these on 15m TF across ALL symbols (15m is less noisy).
Also build a comprehensive Forex MR suite using: ALMA, ADX, ATR, RSI,
StochRSI, EMA9, Choppiness Index, MACD, Bollinger Bands —
optimized for 5m & 15m scalping.
"""

import glob
import numpy as np
import pandas as pd
from backtest_core import BacktestCore
from indicators_library import (
    calc_ema, calc_rsi, calc_stoch_rsi, calc_atr, calc_mfi,
    calc_bollinger_bands, calc_vwap, calc_choppiness_index,
    calc_cvd, calc_kama, calc_adx, calc_macd,
)

FEE_PCT = 0.0002   # FBS Forex 0.02% round-trip

def get_atr(df, p=14):
    return calc_atr(df, p)

def get_rvol(df, p=20):
    return df['volume'] / df['volume'].rolling(p).mean()

def rolling_vwap(df, period=200):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()

def calc_alma(series, period=9, sigma=6, offset=0.85):
    """ALMA — Arnaud Legoux Moving Average (optimized for 5m/15m: period=9)."""
    m  = offset * (period - 1)
    s  = period / sigma
    w  = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
    w  = w / w.sum()
    alma = series.rolling(period).apply(lambda x: np.dot(x, w), raw=True)
    return alma

def run_walkforward(engine, dfs, fn, params, n_splits=5):
    all_pf = []
    for symbol, df in dfs.items():
        n = len(df)
        chunk = n // n_splits
        for i in range(1, n_splits):
            oos_df = df.iloc[i * chunk: (i + 1) * chunk]
            if len(oos_df) < 100:
                continue
            try:
                _, agg, _ = engine.run_multi_symbol(
                    {symbol: oos_df}, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
                if agg['total_trades'] >= 3:
                    all_pf.append(agg['profit_factor'])
            except Exception:
                pass
    return round(np.mean(all_pf), 3) if all_pf else 0.0

def load_tf(pattern):
    """Load all CSVs matching a glob pattern (e.g. '*_15min_1year.csv')."""
    dfs = {}
    for f in glob.glob(pattern):
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


# ═══════════════════════════════════════════════════════════════
# PROVEN WINNERS (ported from 5m)
# ═══════════════════════════════════════════════════════════════

def strat_mr10_engulf(df, p):
    """Engulf Reversal at BB 2.0σ extreme — won on LTCUSDT 5m."""
    atr  = get_atr(df)
    chop = calc_choppiness_index(df, 14)
    rvol = get_rvol(df)
    cvd  = calc_cvd(df, 5)
    mfi  = calc_mfi(df, 14)
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.0)
    o = df['open']; c = df['close']
    prior_bear_low = (c.shift(1) < o.shift(1)) & (c.shift(1) < bb_l.shift(1))
    prior_bull_hi  = (c.shift(1) > o.shift(1)) & (c.shift(1) > bb_u.shift(1))
    prior_mid_up = (o.shift(1) + c.shift(1)) / 2
    prior_mid_dn = (o.shift(1) + c.shift(1)) / 2
    bull_q = (c > o) & (c > prior_mid_up)
    bear_q = (c < o) & (c < prior_mid_dn)
    vol_ok = rvol > 0.6
    regime = (chop > 42) & (atr >= 0.6 * atr.rolling(50).mean())
    signals = pd.Series(0, index=df.index)
    signals[prior_bear_low & bull_q & vol_ok & (cvd > cvd.shift(1)) & (mfi.shift(1)<40) & regime] =  1
    signals[prior_bull_hi  & bear_q & vol_ok & (cvd < cvd.shift(1)) & (mfi.shift(1)>60) & regime] = -1
    return signals

def make_mr2_ny(session_hours=range(13, 21)):
    def strat(df, p):
        atr   = get_atr(df)
        rsi   = calc_rsi(df['close'], 14)
        chop  = calc_choppiness_index(df, 14)
        rvwap = rolling_vwap(df, 200)
        rvol  = get_rvol(df)
        cvd   = calc_cvd(df, 5)
        rs    = df['close'].rolling(20).std()
        vl    = rvwap - 2.0 * rs
        vu    = rvwap + 2.0 * rs
        ext_d = df['close'] < vl
        ext_u = df['close'] > vu
        lk    = 10
        bull  = ext_d & (df['close'] <= df['close'].rolling(lk).min()) & (rsi > rsi.rolling(lk).min())
        bear  = ext_u & (df['close'] >= df['close'].rolling(lk).max()) & (rsi < rsi.rolling(lk).max())
        reg   = (chop > 50) & (atr >= 0.6 * atr.rolling(50).mean())
        el    = (df['close'] > df['close'].shift(1)) & ext_d.shift(1)
        es    = (df['close'] < df['close'].shift(1)) & ext_u.shift(1)
        signals = pd.Series(0, index=df.index)
        lc = bull.shift(1) & reg & el & (cvd > cvd.shift(2)) & (rvol > 0.7)
        sc = bear.shift(1) & reg & es & (cvd < cvd.shift(2)) & (rvol > 0.7)
        if session_hours is not None:
            mask = pd.Series(df.index.hour, index=df.index).isin(session_hours)
            lc = lc & mask; sc = sc & mask
        signals[lc] =  1; signals[sc] = -1
        return signals
    return strat


# ═══════════════════════════════════════════════════════════════
# NEW FOREX MR SUITE — Indicator-specified, optimized for 5m/15m
# Indicator settings tuned for fast timeframes:
#   ADX(7) — faster sensitivity
#   ATR(7) — shorter, reacts to recent volatility
#   CHOP(14) — standard but used strictly (>55 = choppy/range)
#   MACD(5,13,4) — faster than default 12,26,9 for scalping
#   BB(20,2.0) — standard
#   RSI(9) — faster (not 14) for 5m/15m
#   StochRSI(9,3,3) — faster period
#   EMA9 — fast signal EMA
#   ALMA(9) — ultra-fast adaptive MA
# ═══════════════════════════════════════════════════════════════

def forex_indicators(df):
    """Pre-compute all Forex MR indicators at optimal fast TF settings."""
    atr7   = calc_atr(df, 7)                            # Fast ATR
    atr14  = calc_atr(df, 14)
    adx_val, di_plus, di_minus = calc_adx(df, 7)       # Fast ADX — unpack tuple
    rsi9   = calc_rsi(df['close'], 9)                   # Fast RSI
    sk, sd = calc_stoch_rsi(df['close'], 9, 3, 3)       # Fast StochRSI
    chop14 = calc_choppiness_index(df, 14)
    macd_line, macd_sig, macd_hist = calc_macd(df['close'], 5, 13, 4)  # Fast MACD
    bb_u, bb_m, bb_l, _, bb_w = calc_bollinger_bands(df['close'], 20, 2.0)
    ema9   = calc_ema(df['close'], 9)
    ema21  = calc_ema(df['close'], 21)
    alma9  = calc_alma(df['close'], 9)
    rvol   = get_rvol(df, 20)
    cvd    = calc_cvd(df, 5)
    mfi14  = calc_mfi(df, 14)
    return dict(
        atr7=atr7, atr14=atr14, adx7=adx_val,
        rsi9=rsi9, sk=sk, sd=sd,
        chop=chop14,
        macd_l=macd_line, macd_s=macd_sig, macd_h=macd_hist,
        bb_u=bb_u, bb_m=bb_m, bb_l=bb_l, bb_w=bb_w,
        ema9=ema9, ema21=ema21, alma9=alma9,
        rvol=rvol, cvd=cvd, mfi=mfi14
    )

def regime_gate(i, adx_max=22, chop_min=50, vol_min_frac=0.6):
    """Universal regime gate: avoid strong trends + avoid dead markets."""
    # ADX < adx_max: market is NOT strongly trending (OK for MR)
    # CHOP > chop_min: market IS ranging/choppy (OK for MR)
    # ATR >= vol_min_frac × rolling mean: market has enough volatility
    no_trend = i['adx7'] < adx_max
    is_range = i['chop'] > chop_min
    has_vol  = i['atr7'] >= vol_min_frac * i['atr7'].rolling(50).mean()
    return no_trend & is_range & has_vol


# ─── FX_MR1: ALMA Mean Reversion + StochRSI Extreme ──────────
# ALMA(9) as fast adaptive mean. Price far from ALMA + StochRSI
# at extreme + MACD histogram reversing = high-prob reversion.
def strat_fx_mr1_alma_stoch(df, p):
    i = forex_indicators(df)
    gate = regime_gate(i)
    alma_dev = (df['close'] - i['alma9']) / i['atr7']
    # Extended below ALMA by > 1.5 ATR
    ext_below = alma_dev < -1.5
    ext_above = alma_dev >  1.5
    # StochRSI extreme + turning
    sk_os = (i['sk'] < 15) & (i['sk'] > i['sk'].shift(1))  # OS & turning up
    sk_ob = (i['sk'] > 85) & (i['sk'] < i['sk'].shift(1))  # OB & turning down
    # MACD histogram turning (momentum exhausting)
    macd_turn_up = (i['macd_h'] > i['macd_h'].shift(1)) & (i['macd_h'].shift(1) < 0)
    macd_turn_dn = (i['macd_h'] < i['macd_h'].shift(1)) & (i['macd_h'].shift(1) > 0)
    # RSI not yet crossed midline (early in the reversal)
    rsi_low  = i['rsi9'] < 40
    rsi_high = i['rsi9'] > 60
    signals = pd.Series(0, index=df.index)
    signals[gate & ext_below & sk_os & macd_turn_up & rsi_low ] =  1
    signals[gate & ext_above & sk_ob & macd_turn_dn & rsi_high] = -1
    return signals


# ─── FX_MR2: BB Band-Touch + RSI Divergence ──────────────────
# Classic BB touch with RSI divergence — updated for fast settings.
def strat_fx_mr2_bb_rsi_div(df, p):
    i = forex_indicators(df)
    gate = regime_gate(i, adx_max=25, chop_min=48)
    lk = 8
    price_low  = df['close'] == df['close'].rolling(lk).min()
    price_high = df['close'] == df['close'].rolling(lk).max()
    rsi_higher_low  = i['rsi9'] > i['rsi9'].rolling(lk).min()
    rsi_lower_high  = i['rsi9'] < i['rsi9'].rolling(lk).max()
    at_bb_low  = df['close'] <= i['bb_l']
    at_bb_high = df['close'] >= i['bb_u']
    bull_div = price_low & rsi_higher_low & at_bb_low
    bear_div = price_high & rsi_lower_high & at_bb_high
    entry_l = (df['close'] > df['close'].shift(1)) & bull_div.shift(1)
    entry_s = (df['close'] < df['close'].shift(1)) & bear_div.shift(1)
    cvd_ok_l = i['cvd'] > i['cvd'].shift(2)
    cvd_ok_s = i['cvd'] < i['cvd'].shift(2)
    signals = pd.Series(0, index=df.index)
    signals[gate & entry_l & cvd_ok_l] =  1
    signals[gate & entry_s & cvd_ok_s] = -1
    return signals


# ─── FX_MR3: EMA9/ALMA Cross Reversion ───────────────────────
# When fast EMA9 diverges far from ALMA9 (both very fast MAs),
# the spread tends to close quickly. Trade the convergence.
def strat_fx_mr3_ema_alma_cross(df, p):
    i = forex_indicators(df)
    gate = regime_gate(i, adx_max=20, chop_min=52)
    spread = (i['ema9'] - i['alma9']) / i['atr7']
    # EMA9 has diverged > 1.2 ATR from ALMA9
    spread_ext_below = spread < -1.2
    spread_ext_above = spread >  1.2
    # Spread is now narrowing (convergence started)
    narrowing_bull = (spread > spread.shift(1)) & spread_ext_below.shift(1)
    narrowing_bear = (spread < spread.shift(1)) & spread_ext_above.shift(1)
    # StochRSI not extreme against us
    srsi_ok_l = i['sk'] < 60
    srsi_ok_s = i['sk'] > 40
    # MACD confirming exhaustion
    macd_exhaust_l = i['macd_h'] > i['macd_h'].shift(2)
    macd_exhaust_s = i['macd_h'] < i['macd_h'].shift(2)
    signals = pd.Series(0, index=df.index)
    signals[gate & narrowing_bull & srsi_ok_l & macd_exhaust_l] =  1
    signals[gate & narrowing_bear & srsi_ok_s & macd_exhaust_s] = -1
    return signals


# ─── FX_MR4: StochRSI Double Extreme (Ultra-OS/OB) ───────────
# Adapted from MR11 but using fast 9-period StochRSI.
# ALL: RSI9, StochRSI9, MACD histogram — all at extreme simultaneously.
def strat_fx_mr4_triple_extreme(df, p):
    i = forex_indicators(df)
    gate = regime_gate(i, adx_max=28, chop_min=46)
    triple_os = (i['rsi9'] < 20) & (i['sk'] < 5) & (i['macd_h'] < 0) & (i['macd_h'] > i['macd_h'].shift(1))
    triple_ob = (i['rsi9'] > 80) & (i['sk'] > 95) & (i['macd_h'] > 0) & (i['macd_h'] < i['macd_h'].shift(1))
    bull_turn = (df['close'] > df['close'].shift(1)) & (i['cvd'] > i['cvd'].shift(1))
    bear_turn = (df['close'] < df['close'].shift(1)) & (i['cvd'] < i['cvd'].shift(1))
    signals = pd.Series(0, index=df.index)
    signals[triple_os.shift(1) & bull_turn] =  1
    signals[triple_ob.shift(1) & bear_turn] = -1
    return signals


# ─── FX_MR5: BB Engulf + MACD Histogram Turn ─────────────────
# Price closes beyond BB 2.0σ, next candle engulfs (or strong
# reversal bar), and MACD histogram has turned. Best of MR10 +
# MACD confirmation for Forex noise robustness.
def strat_fx_mr5_bb_engulf_macd(df, p):
    i = forex_indicators(df)
    gate = regime_gate(i, adx_max=25, chop_min=46)
    o = df['open']; c = df['close']
    at_bb_l = c.shift(1) < i['bb_l'].shift(1)
    at_bb_u = c.shift(1) > i['bb_u'].shift(1)
    pm = (o.shift(1) + c.shift(1)) / 2
    bull_bar = (c > o) & (c > pm)
    bear_bar = (c < o) & (c < pm)
    macd_turn_up = (i['macd_h'] > i['macd_h'].shift(1)) & (i['macd_h'] < 0)
    macd_turn_dn = (i['macd_h'] < i['macd_h'].shift(1)) & (i['macd_h'] > 0)
    rsi_not_ob = i['rsi9'] < 65
    rsi_not_os = i['rsi9'] > 35
    rvol_ok = i['rvol'] > 0.7
    signals = pd.Series(0, index=df.index)
    signals[gate & at_bb_l & bull_bar & macd_turn_up & rsi_not_ob & rvol_ok] =  1
    signals[gate & at_bb_u & bear_bar & macd_turn_dn & rsi_not_os & rvol_ok] = -1
    return signals


# ─── FX_MR6: ADX Collapse + BB Mid-Reversion ─────────────────
# When ADX was trending (>20) and then COLLAPSES (drops 5+ pts),
# the trend has exhausted and price reverts to BB middle.
def strat_fx_mr6_adx_collapse(df, p):
    i = forex_indicators(df)
    atr14 = i['atr14']
    # ADX collapse: was strong (>22) now falling back
    adx_collapse = (i['adx7'] < i['adx7'].shift(3) - 5) & (i['adx7'].shift(3) > 22)
    # Price is extended from BB midline
    dev_below = (i['bb_m'] - df['close']) > 1.0 * i['atr7']
    dev_above = (df['close'] - i['bb_m']) > 1.0 * i['atr7']
    # CHOP rising (transitioning to range)
    chop_rising = i['chop'] > i['chop'].shift(3)
    has_vol = atr14 >= 0.6 * atr14.rolling(50).mean()
    # RSI near midline (not at extreme — momentum is balanced)
    rsi_neutral = (i['rsi9'] > 35) & (i['rsi9'] < 65)
    # StochRSI reversing
    sk_up = (i['sk'] > i['sk'].shift(1)) & (i['sk'] < 50)
    sk_dn = (i['sk'] < i['sk'].shift(1)) & (i['sk'] > 50)
    signals = pd.Series(0, index=df.index)
    signals[adx_collapse & dev_below & chop_rising & has_vol & sk_up & rsi_neutral] =  1
    signals[adx_collapse & dev_above & chop_rising & has_vol & sk_dn & rsi_neutral] = -1
    return signals


# ─── FX_MR7: ALMA Slope Exhaustion Fade ──────────────────────
# The ALMA(9) slope tells us momentum direction and intensity.
# When slope has been running in one direction for 5+ bars but
# RSI is diverging (weakening momentum) and price hits BB outer,
# fade the exhausted move.
def strat_fx_mr7_alma_slope_exhaust(df, p):
    i = forex_indicators(df)
    gate = regime_gate(i, adx_max=28, chop_min=45)
    alma_slope = i['alma9'] - i['alma9'].shift(3)
    # ALMA has been sloping down/up for 5 bars but weakening
    slope_was_down = (alma_slope.rolling(5).sum() < 0)
    slope_was_up   = (alma_slope.rolling(5).sum() > 0)
    slope_weakening_up   = (alma_slope.abs() < alma_slope.shift(2).abs())
    slope_weakening_down = slope_weakening_up
    # Price at BB extreme while ALMA slope weakens
    at_low = (df['close'] < i['bb_l']) & slope_was_down & slope_weakening_up
    at_high = (df['close'] > i['bb_u']) & slope_was_up   & slope_weakening_down
    # RSI divergence
    rsi_bull = i['rsi9'] > i['rsi9'].shift(5)
    rsi_bear = i['rsi9'] < i['rsi9'].shift(5)
    # MACD histogram shrinking
    macd_shrink_l = i['macd_h'].abs() < i['macd_h'].shift(2).abs()
    signals = pd.Series(0, index=df.index)
    signals[gate & at_low  & rsi_bull & macd_shrink_l] =  1
    signals[gate & at_high & rsi_bear & macd_shrink_l] = -1
    return signals


# ─── FX_MR8: EMA9 vs EMA21 Channel Reversion ─────────────────
# EMA9 and EMA21 define a fast "channel". When price closes
# outside both EMAs AND BB outer band, trade the snap-back inside.
def strat_fx_mr8_ema_channel(df, p):
    i = forex_indicators(df)
    gate = regime_gate(i, adx_max=22, chop_min=50)
    # Price below both fast EMAs AND below lower BB
    below_both = (df['close'] < i['ema9']) & (df['close'] < i['ema21']) & (df['close'] < i['bb_l'])
    above_both = (df['close'] > i['ema9']) & (df['close'] > i['ema21']) & (df['close'] > i['bb_u'])
    # StochRSI at extreme and turning
    sk_os_turn = (i['sk'] < 20) & (i['sk'] > i['sk'].shift(1))
    sk_ob_turn = (i['sk'] > 80) & (i['sk'] < i['sk'].shift(1))
    # MACD histogram turning up/down
    hist_up = (i['macd_h'] > i['macd_h'].shift(1)) & (i['macd_h'] < 0)
    hist_dn = (i['macd_h'] < i['macd_h'].shift(1)) & (i['macd_h'] > 0)
    # Volume ok
    vol_ok = i['rvol'] > 0.6
    signals = pd.Series(0, index=df.index)
    signals[gate & below_both & sk_os_turn & hist_up & vol_ok] =  1
    signals[gate & above_both & sk_ob_turn & hist_dn & vol_ok] = -1
    return signals


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    engine = BacktestCore()

    dfs_5m  = load_tf("*_5min_1year.csv")
    dfs_15m = load_tf("*_15min_1year.csv")
    ltc_5m  = {'LTCUSDT': dfs_5m['LTCUSDT']} if 'LTCUSDT' in dfs_5m else {}
    ltc_15m = {'LTCUSDT': dfs_15m['LTCUSDT']} if 'LTCUSDT' in dfs_15m else {}

    print(f"5m  symbols: {sorted(dfs_5m.keys())}")
    print(f"15m symbols: {sorted(dfs_15m.keys())}\n")

    params_A = {'sl_atr': 1.5, 'tp_atr': 2.5, 'max_bars_hold': 60, 'risk_pct': 0.01, 'trailing': False}
    params_B = {'sl_atr': 2.0, 'tp_atr': 3.0, 'max_bars_hold': 80, 'risk_pct': 0.01, 'trailing': False}

    all_results = []

    def test(name, fn, dataset, params, tag=""):
        try:
            _, agg, _ = engine.run_multi_symbol(
                dataset, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
            trades = agg['total_trades']
            if trades < 20:
                print(f"  {name} [{tag}]: SKIP ({trades} trades)")
                return
            pf     = agg['profit_factor']
            wr     = agg['win_rate']
            sharpe = agg['sharpe_ratio']
            oos    = run_walkforward(engine, dataset, fn, params)
            status = "PASS" if pf >= 1.0 and oos >= 1.0 else ("STD_PASS" if pf >= 1.0 else "FAIL")
            flag   = " *** PASS ***" if status == "PASS" else (" * STD_PASS" if status == "STD_PASS" else "")
            print(f"  {name:40s} PF={pf:.3f} OOS={oos:.3f} WR={wr:.1f}% Sh={sharpe:.2f} T={trades} [{tag}]{flag}")
            all_results.append({'TF': tag, 'Strategy': name, 'Std_PF': round(pf,3),
                                 'OOS_PF': oos, 'WR_%': round(wr,2),
                                 'Sharpe': round(sharpe,3), 'Trades': trades, 'Status': status})
        except Exception as e:
            print(f"  {name} [{tag}]: ERROR {e}")

    # ── PART 1: PROVEN WINNERS ON 15m ─────────────────────────
    print("=" * 70)
    print("PART 1: PROVEN 5m WINNERS — TESTED ON 15m TF")
    print("=" * 70)
    for sym_lbl, dataset_15m in [("All", dfs_15m), ("LTC", ltc_15m)]:
        test(f"MR10_Engulf_{sym_lbl}", strat_mr10_engulf, dataset_15m, params_A, "15m")
        test(f"MR10_Engulf_{sym_lbl}", strat_mr10_engulf, dataset_15m, params_B, "15m")
        test(f"MR2_NY_{sym_lbl}",     make_mr2_ny(),      dataset_15m, params_A, "15m")
        test(f"MR2_NY_{sym_lbl}",     make_mr2_ny(),      dataset_15m, params_B, "15m")
        test(f"MR2_AllSess_{sym_lbl}",make_mr2_ny(None),  dataset_15m, params_A, "15m")

    # ── PART 2: FOREX MR SUITE — 5m ───────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: FOREX MR SUITE — 5m TF")
    print("=" * 70)
    fx_strategies = {
        "FX_MR1_ALMA_StochRSI":     strat_fx_mr1_alma_stoch,
        "FX_MR2_BB_RSI_Div":        strat_fx_mr2_bb_rsi_div,
        "FX_MR3_EMA_ALMA_Cross":    strat_fx_mr3_ema_alma_cross,
        "FX_MR4_Triple_Extreme":    strat_fx_mr4_triple_extreme,
        "FX_MR5_BB_Engulf_MACD":    strat_fx_mr5_bb_engulf_macd,
        "FX_MR6_ADX_Collapse":      strat_fx_mr6_adx_collapse,
        "FX_MR7_ALMA_SlopeExhaust": strat_fx_mr7_alma_slope_exhaust,
        "FX_MR8_EMA_Channel":       strat_fx_mr8_ema_channel,
    }
    for name, fn in fx_strategies.items():
        for p_lbl, params in [("A", params_A), ("B", params_B)]:
            test(f"{name}_{p_lbl}", fn, dfs_5m,  params, "5m")

    # ── PART 3: FOREX MR SUITE — 15m ──────────────────────────
    print("\n" + "=" * 70)
    print("PART 3: FOREX MR SUITE — 15m TF")
    print("=" * 70)
    for name, fn in fx_strategies.items():
        for p_lbl, params in [("A", params_A), ("B", params_B)]:
            test(f"{name}_{p_lbl}", fn, dfs_15m, params, "15m")

    # ── FINAL RANKINGS ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MASTER RANKINGS — All Strategies × Both TFs (Sorted by PF)")
    print("=" * 70)
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values('Std_PF', ascending=False)
        print(df_res.head(30).to_string(index=False))
        df_res.to_csv("forex_mr_rankings.csv", index=False)
        print("\nFull rankings saved to forex_mr_rankings.csv")

        passing = df_res[df_res['Std_PF'] >= 1.0]
        oos_passing = df_res[df_res['Status'] == 'PASS']
        print(f"\nStd PF >= 1.0: {len(passing)} strategies")
        print(f"Full PASS (Std + OOS >= 1.0): {len(oos_passing)} strategies")
        if not oos_passing.empty:
            print("\n*** FULL PASS STRATEGIES ***")
            print(oos_passing.to_string(index=False))


if __name__ == "__main__":
    main()
