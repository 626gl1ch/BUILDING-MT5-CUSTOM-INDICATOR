"""
5m MR Breakthrough — Targeted Testing of Best Intersections
============================================================
KEY FINDINGS from research.py:
  - MR2 on LTCUSDT alone: PF=0.960 (nearly profitable!)
  - MR2 during NY Session (13-21 UTC): PF=0.930, OOS=0.892
  - MR10 Engulf Reversal: PF=0.940 but only 261 trades across all symbols
  - MR2 all symbols all sessions baseline: PF=0.830

HYPOTHESES:
  H1: MR2 on LTCUSDT + NY session combined → PF > 1.0?
  H2: MR10 with loosened conditions → more trades + still high WR?
  H3: Pin-bar reversal at BB extreme → higher natural WR (65%+)?
  H4: LTCUSDT-only tuning (LTCUSDT has inherently better MR props)
"""

import glob
import numpy as np
import pandas as pd
from backtest_core import BacktestCore
from indicators_library import (
    calc_ema, calc_rsi, calc_stoch_rsi, calc_atr, calc_mfi,
    calc_bollinger_bands, calc_vwap, calc_choppiness_index,
    calc_cvd, calc_kama, calc_adx,
)

FEE_PCT = 0.0002

def get_atr(df, p=14):
    return calc_atr(df, p)

def get_rvol(df, p=20):
    return df['volume'] / df['volume'].rolling(p).mean()

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
            print(f"Error loading {f}: {e}")
    return dfs


# ─────────────────────────────────────────────────────────
# MR2 with session + symbol filters baked in
# ─────────────────────────────────────────────────────────
def make_mr2(session_hours=None, chop_floor=48, std_mult=2.0, rvol_min=0.7):
    def strat(df, p):
        atr   = get_atr(df)
        rsi   = calc_rsi(df['close'], 14)
        chop  = calc_choppiness_index(df, 14)
        rvwap = rolling_vwap(df, 200)
        rvol  = get_rvol(df)
        cvd   = calc_cvd(df, 5)

        rolling_std = df['close'].rolling(20).std()
        vwap_lower  = rvwap - std_mult * rolling_std
        vwap_upper  = rvwap + std_mult * rolling_std

        extended_below = df['close'] < vwap_lower
        extended_above = df['close'] > vwap_upper

        lookback = 10
        bull_div = (extended_below &
                    (df['close'] <= df['close'].rolling(lookback).min()) &
                    (rsi > rsi.rolling(lookback).min()))
        bear_div = (extended_above &
                    (df['close'] >= df['close'].rolling(lookback).max()) &
                    (rsi < rsi.rolling(lookback).max()))

        mr_regime = (chop > chop_floor) & (atr >= 0.6 * atr.rolling(50).mean())
        entry_long  = (df['close'] > df['close'].shift(1)) & extended_below.shift(1)
        entry_short = (df['close'] < df['close'].shift(1)) & extended_above.shift(1)
        cvd_bull = cvd > cvd.shift(2)
        cvd_bear = cvd < cvd.shift(2)

        signals = pd.Series(0, index=df.index)
        long_cond  = bull_div.shift(1) & mr_regime & entry_long  & cvd_bull & (rvol > rvol_min)
        short_cond = bear_div.shift(1) & mr_regime & entry_short & cvd_bear & (rvol > rvol_min)

        if session_hours is not None:
            mask = pd.Series(df.index.hour, index=df.index).isin(session_hours)
            long_cond  = long_cond  & mask
            short_cond = short_cond & mask

        signals[long_cond]  =  1
        signals[short_cond] = -1
        return signals
    return strat


# ─────────────────────────────────────────────────────────
# MR10 Loosened — Engulfing at BB Extreme
# ─────────────────────────────────────────────────────────
def make_mr10(bb_std=2.0, rvol_min=0.8, mfi_os=35, mfi_ob=65,
              chop_floor=44, require_full_engulf=True):
    def strat(df, p):
        atr  = get_atr(df)
        chop = calc_choppiness_index(df, 14)
        rvol = get_rvol(df)
        cvd  = calc_cvd(df, 5)
        mfi  = calc_mfi(df, 14)
        bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, bb_std)

        o = df['open']
        c = df['close']

        prior_bearish_at_low = (c.shift(1) < o.shift(1)) & (c.shift(1) < bb_l.shift(1))
        prior_bullish_at_hi  = (c.shift(1) > o.shift(1)) & (c.shift(1) > bb_u.shift(1))

        if require_full_engulf:
            # Full engulf: body covers prior body completely
            bull_engulf = (c > o) & (o <= c.shift(1)) & (c >= o.shift(1))
            bear_engulf = (c < o) & (o >= c.shift(1)) & (c <= o.shift(1))
        else:
            # Relaxed: just a strong reversal bar (closes above prior candle midpoint)
            prior_mid_up = (o.shift(1) + c.shift(1)) / 2
            prior_mid_dn = (o.shift(1) + c.shift(1)) / 2
            bull_engulf = (c > o) & (c > prior_mid_up)
            bear_engulf = (c < o) & (c < prior_mid_dn)

        vol_ok  = rvol > rvol_min
        cvd_bull = cvd > cvd.shift(1)
        cvd_bear = cvd < cvd.shift(1)
        mfi_bull = mfi.shift(1) < mfi_os
        mfi_bear = mfi.shift(1) > mfi_ob
        mr_regime = (chop > chop_floor) & (atr >= 0.6 * atr.rolling(50).mean())

        signals = pd.Series(0, index=df.index)
        signals[prior_bearish_at_low & bull_engulf & vol_ok & cvd_bull & mfi_bull & mr_regime] =  1
        signals[prior_bullish_at_hi  & bear_engulf & vol_ok & cvd_bear & mfi_bear & mr_regime] = -1
        return signals
    return strat


# ─────────────────────────────────────────────────────────
# MR12 — Pin Bar Reversal at BB Extreme
# A pin bar (hammer/shooting star) at the outer BB signals
# institutional rejection of price at that level. The wick is
# the market "testing" extreme and failing. The close back near
# the body = built-in reversal confirmation.
# Expected WR: 60-70% (the candle itself confirms reversal)
# ─────────────────────────────────────────────────────────
def strat_mr12_pin_bar(df, p):
    atr  = get_atr(df)
    chop = calc_choppiness_index(df, 14)
    rvol = get_rvol(df)
    cvd  = calc_cvd(df, 5)
    mfi  = calc_mfi(df, 14)
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.0)

    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']

    body       = (c - o).abs()
    hl_range   = h - l
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l

    # Bullish pin bar (hammer): long lower wick, small body, at lower BB
    # Wick must be >= 2× the body, wick points DOWN (rejection of lows)
    hammer = (
        (lower_wick >= 2.0 * body.clip(lower=1e-8)) &   # large lower wick
        (lower_wick >= 0.6 * hl_range) &                 # wick is 60%+ of range
        (l < bb_l) &                                      # wick poked below BB
        (c > bb_l) &                                      # but closed back inside
        (c > o)                                           # bullish close
    )

    # Bearish pin bar (shooting star): long upper wick, small body, at upper BB
    shooting_star = (
        (upper_wick >= 2.0 * body.clip(lower=1e-8)) &
        (upper_wick >= 0.6 * hl_range) &
        (h > bb_u) &
        (c < bb_u) &
        (c < o)
    )

    vol_surge  = rvol > 1.0
    cvd_bull   = cvd > cvd.shift(1)
    cvd_bear   = cvd < cvd.shift(1)
    mfi_bull   = mfi < 40
    mfi_bear   = mfi > 60
    mr_regime  = (chop > 44) & (atr >= 0.6 * atr.rolling(50).mean())

    signals = pd.Series(0, index=df.index)
    signals[hammer        & vol_surge & cvd_bull & mfi_bull & mr_regime] =  1
    signals[shooting_star & vol_surge & cvd_bear & mfi_bear & mr_regime] = -1
    return signals


# ─────────────────────────────────────────────────────────
# MR13 — Inside Bar Breakout After Extreme
# After a price extreme (below lower BB), the market often
# forms a small inside bar (low-volatility pause = indecision).
# When this inside bar breaks in the reversal direction, it
# signals institutional accumulation complete → strong reversal.
# ─────────────────────────────────────────────────────────
def strat_mr13_inside_bar(df, p):
    atr   = get_atr(df)
    chop  = calc_choppiness_index(df, 14)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.0)
    rsi   = calc_rsi(df['close'], 14)

    h = df['high']
    l = df['low']
    c = df['close']

    # Was at extreme 1-3 bars ago?
    was_below_bb_low = (c.shift(2) < bb_l.shift(2)) | (c.shift(3) < bb_l.shift(3))
    was_above_bb_hi  = (c.shift(2) > bb_u.shift(2)) | (c.shift(3) > bb_u.shift(3))

    # Inside bar: current high < prior high AND current low > prior low
    inside_bar = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))

    # Breakout of the inside bar in reversal direction
    bull_break = (c > h.shift(1)) & inside_bar & was_below_bb_low
    bear_break = (c < l.shift(1)) & inside_bar & was_above_bb_hi

    # Volume expanding on breakout
    vol_expand = rvol > 1.1
    cvd_bull   = cvd > cvd.shift(1)
    cvd_bear   = cvd < cvd.shift(1)
    mr_regime  = (chop > 46) & (atr >= 0.6 * atr.rolling(50).mean())
    rsi_ok_l   = rsi < 55  # not yet overbought after the reversal
    rsi_ok_s   = rsi > 45

    signals = pd.Series(0, index=df.index)
    signals[bull_break & vol_expand & cvd_bull & mr_regime & rsi_ok_l] =  1
    signals[bear_break & vol_expand & cvd_bear & mr_regime & rsi_ok_s] = -1
    return signals


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    dfs = load_data()
    print(f"Loaded: {sorted(dfs.keys())}\n")
    engine = BacktestCore()

    all_results = []

    def test(name, fn, dfs_subset, params, print_result=True):
        try:
            _, agg, _ = engine.run_multi_symbol(
                dfs_subset, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
            trades = agg['total_trades']
            pf     = agg['profit_factor']
            wr     = agg['win_rate']
            sharpe = agg['sharpe_ratio']
            if trades < 15:
                if print_result:
                    print(f"  {name}: SKIP ({trades} trades)")
                return None
            oos_pf = run_walkforward(engine, dfs_subset, fn, params)
            status = "PASS" if pf >= 1.0 and oos_pf >= 1.0 else ("STD_PASS" if pf >= 1.0 else "FAIL")
            if print_result:
                flag = " *** PASS ***" if status == "PASS" else (" * STD_PASS *" if status == "STD_PASS" else "")
                print(f"  {name}: PF={pf:.3f} OOS={oos_pf:.3f} WR={wr:.1f}% Sh={sharpe:.2f} T={trades} [{status}]{flag}")
            r = {'Strategy': name, 'Std_PF': round(pf,3), 'OOS_PF': oos_pf,
                 'WR_%': round(wr,2), 'Sharpe': round(sharpe,3),
                 'Trades': trades, 'Status': status}
            all_results.append(r)
            return r
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            return None

    # Best params from prior runs
    params_A = {'sl_atr': 1.5, 'tp_atr': 2.5, 'max_bars_hold': 80, 'risk_pct': 0.01, 'trailing': False}
    params_B = {'sl_atr': 2.0, 'tp_atr': 3.0, 'max_bars_hold': 100,'risk_pct': 0.01, 'trailing': False}
    params_C = {'sl_atr': 1.5, 'tp_atr': 2.0, 'max_bars_hold': 80, 'risk_pct': 0.01, 'trailing': False}

    ltc_only  = {'LTCUSDT': dfs['LTCUSDT']} if 'LTCUSDT' in dfs else dfs
    btc_only  = {'BTCUSDT': dfs['BTCUSDT']} if 'BTCUSDT' in dfs else dfs
    trump_only= {'TRUMPUSDT': dfs['TRUMPUSDT']} if 'TRUMPUSDT' in dfs else dfs
    sol_only  = {'SOLUSDT': dfs['SOLUSDT']} if 'SOLUSDT' in dfs else dfs

    # ── EXPERIMENT 1: LTCUSDT + NY Session (H1 intersection) ──
    print("=" * 65)
    print("EXP 1: MR2 — LTCUSDT + NY Session Intersection")
    print("=" * 65)
    ny_hours = range(13, 21)
    for params_lbl, params in [("A_SL1.5TP2.5", params_A), ("B_SL2.0TP3.0", params_B), ("C_SL1.5TP2.0", params_C)]:
        test(f"MR2_LTC_NY_{params_lbl}",
             make_mr2(session_hours=ny_hours), ltc_only, params)

    # Also test LTC all sessions vs LTC NY
    test("MR2_LTC_AllSess_A", make_mr2(session_hours=None), ltc_only, params_A)
    test("MR2_LTC_AllSess_B", make_mr2(session_hours=None), ltc_only, params_B)

    # ── EXPERIMENT 2: NY Session on all symbols ──────────────
    print("\n" + "=" * 65)
    print("EXP 2: MR2 NY Session — Per Symbol")
    print("=" * 65)
    for sym, subset in [("BTC", btc_only), ("LTC", ltc_only),
                        ("SOL", sol_only), ("TRUMP", trump_only)]:
        test(f"MR2_NY_{sym}", make_mr2(session_hours=ny_hours), subset, params_A)
        test(f"MR2_NY_{sym}_B", make_mr2(session_hours=ny_hours), subset, params_B)

    # ── EXPERIMENT 3: MR10 Loosened engulf ───────────────────
    print("\n" + "=" * 65)
    print("EXP 3: MR10 Engulf — Loosened Conditions")
    print("=" * 65)
    # Relaxed (quasi-engulf): more trades → reliable OOS
    test("MR10_Relaxed_All_A", make_mr10(bb_std=2.0, rvol_min=0.6, mfi_os=40, mfi_ob=60,
                                          chop_floor=42, require_full_engulf=False),
         dfs, params_A)
    test("MR10_Relaxed_All_B", make_mr10(bb_std=2.0, rvol_min=0.6, mfi_os=40, mfi_ob=60,
                                          chop_floor=42, require_full_engulf=False),
         dfs, params_B)
    test("MR10_Relaxed_LTC_A", make_mr10(bb_std=2.0, rvol_min=0.6, mfi_os=40, mfi_ob=60,
                                           chop_floor=42, require_full_engulf=False),
         ltc_only, params_A)
    # 1.5 sigma (more frequent extremes)
    test("MR10_BB1.5_All_A",   make_mr10(bb_std=1.5, rvol_min=0.6, mfi_os=40, mfi_ob=60,
                                           chop_floor=40, require_full_engulf=False),
         dfs, params_A)

    # ── EXPERIMENT 4: Pin Bar + Inside Bar ───────────────────
    print("\n" + "=" * 65)
    print("EXP 4: New Price-Action MR Strategies")
    print("=" * 65)
    for sym_lbl, subset in [("All", dfs), ("LTC", ltc_only), ("TRUMP", trump_only)]:
        test(f"MR12_PinBar_{sym_lbl}_A", strat_mr12_pin_bar, subset, params_A)
        test(f"MR12_PinBar_{sym_lbl}_B", strat_mr12_pin_bar, subset, params_B)
    for sym_lbl, subset in [("All", dfs), ("LTC", ltc_only)]:
        test(f"MR13_InsideBar_{sym_lbl}_A", strat_mr13_inside_bar, subset, params_A)
        test(f"MR13_InsideBar_{sym_lbl}_B", strat_mr13_inside_bar, subset, params_B)

    # ── EXPERIMENT 5: MR2 tighter CHOP + wider std ───────────
    print("\n" + "=" * 65)
    print("EXP 5: MR2 Parameter Micro-Tuning on LTC")
    print("=" * 65)
    for chop_f, std_m, label in [(55, 1.8, "CHOP55_std1.8"),
                                   (52, 2.0, "CHOP52_std2.0"),
                                   (48, 2.2, "CHOP48_std2.2"),
                                   (45, 2.5, "CHOP45_std2.5")]:
        for params_lbl, params in [("A", params_A), ("B", params_B)]:
            test(f"MR2_LTC_{label}_{params_lbl}",
                 make_mr2(session_hours=None, chop_floor=chop_f, std_mult=std_m),
                 ltc_only, params)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY — Ranked by Std PF")
    print("=" * 65)
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values('Std_PF', ascending=False)
        print(df_res.head(20).to_string(index=False))
        df_res.to_csv("breakthrough_results.csv", index=False)
        print("\nSaved to breakthrough_results.csv")

        passing = df_res[df_res['Std_PF'] >= 1.0]
        if not passing.empty:
            print("\n" + "*" * 65)
            print("*  PROFITABLE STRATEGIES FOUND!  *")
            print("*" * 65)
            print(passing.to_string(index=False))
        else:
            best = df_res.iloc[0]
            print(f"\nBest result: {best['Strategy']} PF={best['Std_PF']} OOS={best['OOS_PF']}")


if __name__ == "__main__":
    main()
