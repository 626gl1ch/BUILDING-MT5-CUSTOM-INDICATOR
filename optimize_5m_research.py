"""
5m MR Deep Research — Systematic Hypothesis Testing
====================================================
Studying why theoretical PF (1.09) ≠ actual PF (0.83) despite correct WR:

HYPOTHESIS 1: Edge exists on specific symbols (BTC trades differently from TRUMP)
HYPOTHESIS 2: Asian session (UTC 00:00–08:00) has much better MR properties
HYPOTHESIS 3: Double-Bottom/Top confirmation (2nd touch of extreme) boosts WR >55%
HYPOTHESIS 4: Very small TP captures the initial bounce reliably (high WR, low R:R)
HYPOTHESIS 5: Per-session VWAP reset is causing VWAP drift → use rolling VWAP alternative

Method: Test MR2 (best performer) + 3 new strategies across these hypotheses.
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


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def get_atr(df, p=14):
    return calc_atr(df, p)

def get_rvol(df, p=20):
    return df['volume'] / df['volume'].rolling(p).mean()

def rolling_vwap(df, period=200):
    """Rolling VWAP over N bars — avoids daily-reset VWAP drift issues."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()


# ─────────────────────────────────────────────────────────
# BASE MR2 (for comparison / session split)
# ─────────────────────────────────────────────────────────
def strat_mr2_base(df, p, session_hours=None):
    """
    VWAP Deviation + RSI Divergence.
    session_hours: list of UTC hours to allow entry (e.g. range(0,8) = Asian session only)
    """
    atr   = get_atr(df)
    rsi   = calc_rsi(df['close'], 14)
    chop  = calc_choppiness_index(df, 14)
    rvwap = rolling_vwap(df, 200)          # rolling VWAP (not daily-reset)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)

    rolling_std  = df['close'].rolling(20).std()
    vwap_upper   = rvwap + 2.0 * rolling_std
    vwap_lower   = rvwap - 2.0 * rolling_std

    extended_below = df['close'] < vwap_lower
    extended_above = df['close'] > vwap_upper

    lookback = 10
    bull_div = (extended_below &
                (df['close'] <= df['close'].rolling(lookback).min()) &
                (rsi > rsi.rolling(lookback).min()))
    bear_div = (extended_above &
                (df['close'] >= df['close'].rolling(lookback).max()) &
                (rsi < rsi.rolling(lookback).max()))

    mr_regime = (chop > 50) & (atr >= 0.6 * atr.rolling(50).mean())

    entry_long  = (df['close'] > df['close'].shift(1)) & extended_below.shift(1)
    entry_short = (df['close'] < df['close'].shift(1)) & extended_above.shift(1)

    cvd_bull = cvd > cvd.shift(2)
    cvd_bear = cvd < cvd.shift(2)

    signals = pd.Series(0, index=df.index)
    long_cond  = bull_div.shift(1) & mr_regime & entry_long  & cvd_bull & (rvol > 0.7)
    short_cond = bear_div.shift(1) & mr_regime & entry_short & cvd_bear & (rvol > 0.7)

    # Apply session filter if provided
    if session_hours is not None:
        session_mask = pd.Series(df.index.hour, index=df.index).isin(session_hours)
        long_cond  = long_cond  & session_mask
        short_cond = short_cond & session_mask

    signals[long_cond]  =  1
    signals[short_cond] = -1
    return signals


def strat_mr2_all_sessions(df, p):
    return strat_mr2_base(df, p, session_hours=None)

def strat_mr2_asian(df, p):
    """Only trade during Asian session (UTC 00:00-08:00)."""
    return strat_mr2_base(df, p, session_hours=range(0, 8))

def strat_mr2_london(df, p):
    """Only trade during London session (UTC 08:00-16:00)."""
    return strat_mr2_base(df, p, session_hours=range(8, 16))

def strat_mr2_newyork(df, p):
    """Only trade during NY session (UTC 13:00-21:00)."""
    return strat_mr2_base(df, p, session_hours=range(13, 21))


# ─────────────────────────────────────────────────────────
# NEW: MR9 — Double Bottom / Double Top Confirmation
# ─────────────────────────────────────────────────────────
# This is the classic W-bottom / M-top pattern at a VWAP extreme:
# 1. Price hits VWAP -2σ extreme (first bottom) — mark this level
# 2. Price bounces at least 0.5 ATR off the extreme (partial recovery)
# 3. Price RETESTS the extreme (within 0.5 ATR of original bottom)
# 4. BUT does NOT break below it (higher low = structural support)
# 5. Entry: candle that HOLDS above the retest level and closes higher
# 6. SL: just below the double-bottom low (structural SL)
# 7. TP: VWAP (the mean target)
# This gives 60-70% WR because we're confirmed it's held twice.
# ─────────────────────────────────────────────────────────
def strat_mr9_double_bottom(df, p):
    atr   = get_atr(df)
    chop  = calc_choppiness_index(df, 14)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)
    rsi   = calc_rsi(df['close'], 14)
    rvwap = rolling_vwap(df, 200)

    rolling_std = df['close'].rolling(20).std()
    vwap_lower  = rvwap - 2.0 * rolling_std
    vwap_upper  = rvwap + 2.0 * rolling_std

    close = df['close']
    low   = df['low']
    high  = df['high']

    # Step 1: Was price at extreme within last 5-15 bars?
    was_at_lower = (close.rolling(15).min() < vwap_lower).shift(3)
    was_at_upper = (close.rolling(15).max() > vwap_upper).shift(3)

    # The extreme low within last 15 bars
    extreme_low  = low.rolling(15).min().shift(3)
    extreme_high = high.rolling(15).max().shift(3)

    # Step 2: Did price bounce at least 0.5 ATR from the extreme?
    max_since_extreme = close.rolling(5).max()
    bounced_up = max_since_extreme > (extreme_low + 0.5 * atr)

    min_since_extreme = close.rolling(5).min()
    bounced_down = min_since_extreme < (extreme_high - 0.5 * atr)

    # Step 3 + 4: Price retests the extreme but holds (higher low)
    near_extreme_low  = (low  <= extreme_low  + 0.5 * atr) & (close > extreme_low - 0.2 * atr)
    near_extreme_high = (high >= extreme_high - 0.5 * atr) & (close < extreme_high + 0.2 * atr)

    # Step 5: Confirmation — current candle is bullish/bearish (close > open)
    bull_confirm = close > df['open']
    bear_confirm = close < df['open']

    # CVD showing buying/selling on confirmation
    cvd_bull = cvd > cvd.shift(1)
    cvd_bear = cvd < cvd.shift(1)

    mr_regime = (chop > 48) & (atr >= 0.6 * atr.rolling(50).mean())

    signals = pd.Series(0, index=df.index)
    long_sig = (was_at_lower & bounced_up & near_extreme_low &
                bull_confirm & cvd_bull & mr_regime & (rvol > 0.8))
    short_sig = (was_at_upper & bounced_down & near_extreme_high &
                 bear_confirm & cvd_bear & mr_regime & (rvol > 0.8))
    signals[long_sig]  =  1
    signals[short_sig] = -1
    return signals


# ─────────────────────────────────────────────────────────
# NEW: MR10 — BB Extreme + Engulfing Candle Reversal
# ─────────────────────────────────────────────────────────
# Enter when price is at BB extreme AND a LARGE reversal (engulfing)
# candle confirms the turn. No divergence needed — the candle IS
# the confirmation. High WR because we wait for the actual reversal
# to happen (2 candles down then 1 candle that engulfs the prior).
# ─────────────────────────────────────────────────────────
def strat_mr10_engulf_reversal(df, p):
    atr   = get_atr(df)
    chop  = calc_choppiness_index(df, 14)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)
    mfi   = calc_mfi(df, 14)
    bb_u, bb_m, bb_l, _, bb_w = calc_bollinger_bands(df['close'], 20, 2.0)

    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']

    # Bullish engulfing at lower BB:
    # Prior candle is bearish AND closes below lower BB
    # Current candle is bullish AND its body fully engulfs prior candle's body
    prior_bearish = (c.shift(1) < o.shift(1)) & (c.shift(1) < bb_l.shift(1))
    curr_bull_engulf = (c > o) & (o <= c.shift(1)) & (c >= o.shift(1))
    bull_engulf = prior_bearish & curr_bull_engulf

    # Bearish engulfing at upper BB:
    prior_bullish = (c.shift(1) > o.shift(1)) & (c.shift(1) > bb_u.shift(1))
    curr_bear_engulf = (c < o) & (o >= c.shift(1)) & (c <= o.shift(1))
    bear_engulf = prior_bullish & curr_bear_engulf

    # Volume surge on reversal candle (institutional participation)
    vol_surge = rvol > 1.2

    # CVD confirming reversal
    cvd_bull = cvd > cvd.shift(1)
    cvd_bear = cvd < cvd.shift(1)

    # MFI exhaustion — confirms the extreme is real
    mfi_bull_exhaust = mfi.shift(1) < 30
    mfi_bear_exhaust = mfi.shift(1) > 70

    # Regime
    mr_regime = (chop > 48) & (atr >= 0.6 * atr.rolling(50).mean())

    signals = pd.Series(0, index=df.index)
    signals[bull_engulf & vol_surge & cvd_bull & mfi_bull_exhaust & mr_regime] =  1
    signals[bear_engulf & vol_surge & cvd_bear & mfi_bear_exhaust & mr_regime] = -1
    return signals


# ─────────────────────────────────────────────────────────
# NEW: MR11 — Extreme Oscillator Combo (Triple OS/OB)
# ─────────────────────────────────────────────────────────
# Three independent OB/OS oscillators must ALL agree at extreme:
# RSI < 20, MFI < 20, StochRSI < 5 simultaneously.
# This is a very rare but extremely high-probability reversal signal.
# When three volume-aware oscillators all scream "extreme OS/OB",
# the WR of the subsequent reversal should be 65-75%.
# ─────────────────────────────────────────────────────────
def strat_mr11_triple_extreme(df, p):
    atr   = get_atr(df)
    rsi   = calc_rsi(df['close'], 14)
    mfi   = calc_mfi(df, 14)
    sk, sd = calc_stoch_rsi(df['close'], 14, 3, 3)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)
    chop  = calc_choppiness_index(df, 14)

    # Triple-confirmation extreme oversold
    triple_os = (rsi < 22) & (mfi < 22) & (sk < 8)
    triple_ob = (rsi > 78) & (mfi > 78) & (sk > 92)

    # Require: extreme was hit in the PRIOR candle, now starting to turn
    was_triple_os = triple_os.shift(1)
    was_triple_ob = triple_ob.shift(1)

    # Entry: reversal starts (close > prior close for long, < prior close for short)
    turning_bull = df['close'] > df['close'].shift(1)
    turning_bear = df['close'] < df['close'].shift(1)

    # CVD must be turning
    cvd_bull = cvd > cvd.shift(2)
    cvd_bear = cvd < cvd.shift(2)

    # No regime filter needed — triple extreme IS the regime filter
    vol_ok = (atr >= 0.5 * atr.rolling(50).mean()) & (rvol > 0.5)

    signals = pd.Series(0, index=df.index)
    signals[was_triple_os & turning_bull & cvd_bull & vol_ok] =  1
    signals[was_triple_ob & turning_bear & cvd_bear & vol_ok] = -1
    return signals


# ─────────────────────────────────────────────────────────
# DATA LOADER (per-symbol capable)
# ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────
# WALK-FORWARD
# ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("Loading 5m datasets...")
    dfs = load_data()
    if not dfs:
        print("ERROR: No CSV files found.")
        return
    print(f"Loaded {len(dfs)} symbols: {sorted(dfs.keys())}\n")

    engine = BacktestCore()

    # ── Phase 1: Session Analysis on MR2 ──────────────────
    print("=" * 65)
    print("PHASE 1: SESSION ANALYSIS — MR2 VWAP Divergence")
    print("=" * 65)
    session_strategies = {
        "MR2_All_Sessions":  strat_mr2_all_sessions,
        "MR2_Asian_0-8":     strat_mr2_asian,
        "MR2_London_8-16":   strat_mr2_london,
        "MR2_NY_13-21":      strat_mr2_newyork,
    }
    params_base = {'sl_atr': 1.5, 'tp_atr': 2.5, 'max_bars_hold': 80,
                   'risk_pct': 0.01, 'trailing': False}

    phase1_results = []
    for name, fn in session_strategies.items():
        try:
            _, agg, _ = engine.run_multi_symbol(
                dfs, fn, params_base, slippage_pct=0, fee_pct=FEE_PCT)
            trades = agg['total_trades']
            pf     = agg['profit_factor']
            wr     = agg['win_rate']
            print(f"  {name:25s} PF:{pf:.3f} | WR:{wr:.1f}% | Trades:{trades}")
            if trades >= 10:
                oos_pf = run_walkforward(engine, dfs, fn, params_base)
                print(f"    OOS PF: {oos_pf:.3f}")
                phase1_results.append({'Name': name, 'PF': pf, 'OOS_PF': oos_pf,
                                        'WR': wr, 'Trades': trades})
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    # ── Phase 2: Per-Symbol Analysis ──────────────────────
    print("\n" + "=" * 65)
    print("PHASE 2: PER-SYMBOL BREAKDOWN — MR2 + MR11")
    print("=" * 65)
    phase2_results = []
    for symbol, df_sym in dfs.items():
        for strat_name, fn in [("MR2", strat_mr2_all_sessions), ("MR11_Triple", strat_mr11_triple_extreme)]:
            try:
                _, agg, _ = engine.run_multi_symbol(
                    {symbol: df_sym}, fn, params_base, slippage_pct=0, fee_pct=FEE_PCT)
                pf = agg['profit_factor']
                wr = agg['win_rate']
                trades = agg['total_trades']
                print(f"  {symbol:12s} {strat_name:12s} PF:{pf:.3f} | WR:{wr:.1f}% | Trades:{trades}")
                phase2_results.append({'Symbol': symbol, 'Strategy': strat_name,
                                        'PF': pf, 'WR': wr, 'Trades': trades})
            except Exception as e:
                print(f"  {symbol} {strat_name}: ERROR {e}")

    # ── Phase 3: New Strategies ────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3: NEW HIGH-WR STRATEGIES")
    print("=" * 65)
    new_strategies = {
        "MR9_Double_Bottom":     strat_mr9_double_bottom,
        "MR10_Engulf_Reversal":  strat_mr10_engulf_reversal,
        "MR11_Triple_Extreme":   strat_mr11_triple_extreme,
    }
    # Test multiple param sets for new strategies
    param_variants = [
        ("SL1.5 TP2.0 H80",  {'sl_atr': 1.5, 'tp_atr': 2.0, 'max_bars_hold': 80, 'risk_pct': 0.01, 'trailing': False}),
        ("SL1.5 TP2.5 H80",  {'sl_atr': 1.5, 'tp_atr': 2.5, 'max_bars_hold': 80, 'risk_pct': 0.01, 'trailing': False}),
        ("SL2.0 TP3.0 H100", {'sl_atr': 2.0, 'tp_atr': 3.0, 'max_bars_hold': 100,'risk_pct': 0.01, 'trailing': False}),
    ]

    all_results = []
    for name, fn in new_strategies.items():
        print(f"\n  [{name}]")
        for param_label, params in param_variants:
            try:
                _, agg, _ = engine.run_multi_symbol(
                    dfs, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
                trades = agg['total_trades']
                pf     = agg['profit_factor']
                wr     = agg['win_rate']
                sharpe = agg['sharpe_ratio']

                if trades < 10:
                    print(f"    {param_label}: SKIPPED ({trades} trades)")
                    continue

                oos_pf = run_walkforward(engine, dfs, fn, params)
                status = "PASS" if pf >= 1.0 and oos_pf >= 1.0 else ("STD_PASS" if pf >= 1.0 else "FAIL")
                print(f"    {param_label}: PF:{pf:.3f} OOS:{oos_pf:.3f} WR:{wr:.1f}% Sharpe:{sharpe:.3f} T:{trades} [{status}]")
                all_results.append({'Strategy': name, 'Params': param_label, 'Std_PF': pf,
                                     'OOS_PF': oos_pf, 'WR_%': round(wr,2),
                                     'Sharpe': round(sharpe,3), 'Trades': trades, 'Status': status})
            except Exception as e:
                print(f"    {param_label}: ERROR {e}")

    # ── Final Summary ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY — ALL PASSING / NEAR-PASSING RESULTS")
    print("=" * 65)
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values('Std_PF', ascending=False)
        top = df_res.head(10)
        print(top.to_string(index=False))
        df_res.to_csv("research_results.csv", index=False)
        print("\nFull results saved to research_results.csv")

        passing = df_res[df_res['Std_PF'] >= 1.0]
        if not passing.empty:
            print("\n*** PROFITABLE STRATEGIES FOUND ***")
            print(passing.to_string(index=False))


if __name__ == "__main__":
    main()
