"""
5m Mean-Reversion Targeted Optimization v4
===========================================
Key insight from v1-v3:
  - WR = 45% at SL=1.5 ATR (wide enough to survive noise)
  - Theoretical PF at TP=2.0 should be 1.09 but actual is 0.83
  - Root cause: max_bars_hold exits at small loss BEFORE TP is reached
  - Fix: extend max_bars_hold to 100 bars (8.3 hours) to let MR complete
  - Also add 2 new strategies: Keltner Channel MR & Momentum Ignition Fade

Run both existing best (MR2) and new strategies across parameter sets.
"""

import glob
import numpy as np
import pandas as pd
from backtest_core import BacktestCore
from indicators_library import (
    calc_ema, calc_rsi, calc_stoch_rsi, calc_atr, calc_adx,
    calc_bollinger_bands, calc_mfi, calc_vwap, calc_kama,
    calc_choppiness_index, calc_cvd, calc_bb_width_percentile,
)

FEE_PCT = 0.0002


def get_atr(df, p=14):
    return calc_atr(df, p)

def get_rvol(df, period=20):
    return df['volume'] / df['volume'].rolling(period).mean()


# ─────────────────────────────────────────────────────────
# MR2 (Refined) — VWAP Deviation + RSI Divergence
# Best performer from v2/v3. Slightly tightened entry conditions.
# ─────────────────────────────────────────────────────────
def strat_mr2_refined(df, p):
    atr   = get_atr(df)
    rsi   = calc_rsi(df['close'], 14)
    chop  = calc_choppiness_index(df, 14)
    vwap  = calc_vwap(df)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)

    rolling_std  = df['close'].rolling(20).std()
    vwap_upper2  = vwap + 2.0 * rolling_std
    vwap_lower2  = vwap - 2.0 * rolling_std

    # Deviation gate
    extended_below = df['close'] < vwap_lower2
    extended_above = df['close'] > vwap_upper2

    # RSI divergence (price made new extreme but RSI didn't)
    lookback = 10
    bull_div = (extended_below &
                (df['close'] == df['close'].rolling(lookback).min()) &
                (rsi > rsi.rolling(lookback).min()))
    bear_div = (extended_above &
                (df['close'] == df['close'].rolling(lookback).max()) &
                (rsi < rsi.rolling(lookback).max()))

    # CVD shows buyers/sellers stepping in
    cvd_turning_bull = cvd > cvd.shift(2)
    cvd_turning_bear = cvd < cvd.shift(2)

    # Strict ranging regime
    mr_regime = (chop > 52) & (atr >= 0.7 * atr.rolling(50).mean())

    # Entry: price has started reverting (close is higher/lower than prior close)
    entry_long  = (df['close'] > df['close'].shift(1)) & extended_below.shift(1)
    entry_short = (df['close'] < df['close'].shift(1)) & extended_above.shift(1)

    signals = pd.Series(0, index=df.index)
    signals[bull_div.shift(1) & mr_regime & entry_long  & cvd_turning_bull & (rvol > 0.8)] =  1
    signals[bear_div.shift(1) & mr_regime & entry_short & cvd_turning_bear & (rvol > 0.8)] = -1
    return signals


# ─────────────────────────────────────────────────────────
# MR6 — Keltner Channel Mean Reversion
# Keltner Channels use ATR directly, making bands more adaptive
# to actual realized volatility than pure price-std BB.
# Entry: price closes beyond outer KC band, then reverses inside.
# This gives the structural SL naturally at the KC band level.
# ─────────────────────────────────────────────────────────
def strat_mr6_keltner(df, p):
    atr   = get_atr(df)
    ema20 = calc_ema(df['close'], 20)
    atr10 = get_atr(df, 10)
    mfi   = calc_mfi(df, 14)
    chop  = calc_choppiness_index(df, 14)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)

    # Keltner Channels: EMA20 ± 2×ATR(10)
    kc_upper = ema20 + 2.0 * atr10
    kc_lower = ema20 - 2.0 * atr10

    # Price touched outer band and CLOSED back inside (confirmed reversal candle)
    close_back_inside_bull = (df['close'].shift(1) < kc_lower.shift(1)) & (df['close'] > kc_lower)
    close_back_inside_bear = (df['close'].shift(1) > kc_upper.shift(1)) & (df['close'] < kc_upper)

    # MFI exhaustion at the extreme
    mfi_extreme_bull = mfi.shift(1) < 25
    mfi_extreme_bear = mfi.shift(1) > 75

    # Volume participated in the extreme (real move, not drift)
    vol_spike = rvol.shift(1) > 1.1

    # CVD turning
    cvd_bull = cvd > cvd.shift(3)
    cvd_bear = cvd < cvd.shift(3)

    # Ranging regime
    mr_regime = (chop > 50) & (atr >= 0.7 * atr.rolling(50).mean())

    signals = pd.Series(0, index=df.index)
    signals[close_back_inside_bull & mfi_extreme_bull & vol_spike & cvd_bull & mr_regime] =  1
    signals[close_back_inside_bear & mfi_extreme_bear & vol_spike & cvd_bear & mr_regime] = -1
    return signals


# ─────────────────────────────────────────────────────────
# MR7 — Momentum Ignition Fade ("Big Candle Trap")
# When a single 5m candle is huge (>1.5 ATR body) with a volume
# spike, it represents a momentum ignition / stop-hunt sweep.
# These routinely FAIL and reverse back to the pre-candle level.
# Entry: after 2 consecutive REVERSAL candles confirm the fade.
# SL: beyond the extreme of the big candle.
# TP: pre-big-candle open (gap fill = return to origin).
# ─────────────────────────────────────────────────────────
def strat_mr7_big_candle_fade(df, p):
    atr   = get_atr(df)
    chop  = calc_choppiness_index(df, 14)
    rvol  = get_rvol(df)
    cvd   = calc_cvd(df, 5)
    rsi   = calc_rsi(df['close'], 10)

    # Candle body size
    body = (df['close'] - df['open']).abs()
    # Wick ratio: wick is >= 40% of total range (shows rejection)
    hl_range = df['high'] - df['low']
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']

    # BIG BULLISH candle: large up-body + volume spike (potential bull trap)
    big_bull = (
        (df['close'] > df['open']) &          # bullish
        (body > 1.5 * atr) &                  # large body
        (rvol > 1.8) &                        # volume surge
        (upper_wick > 0.4 * hl_range)         # long upper wick (rejection at top)
    )

    # BIG BEARISH candle: large down-body + volume spike (potential bear trap)
    big_bear = (
        (df['close'] < df['open']) &          # bearish
        (body > 1.5 * atr) &                  # large body
        (rvol > 1.8) &                        # volume surge
        (lower_wick > 0.4 * hl_range)         # long lower wick (rejection at bottom)
    )

    # Confirmation: next 2 candles close in the opposite direction
    # (confirmed the big candle was a trap/fake-out)
    confirm_fade_bear = (
        big_bull.shift(2) &
        (df['close'].shift(1) < df['open'].shift(1)) &  # reversal candle 1
        (df['close'] < df['open'])                       # reversal candle 2
    )

    confirm_fade_bull = (
        big_bear.shift(2) &
        (df['close'].shift(1) > df['open'].shift(1)) &  # reversal candle 1
        (df['close'] > df['open'])                       # reversal candle 2
    )

    # CVD must NOT confirm the big candle direction (divergence = trap)
    cvd_not_confirming_bull = cvd < cvd.shift(3)  # CVD falling while price spiked up = trap
    cvd_not_confirming_bear = cvd > cvd.shift(3)  # CVD rising while price dropped = trap

    # RSI not at extreme against us (don't short when oversold, don't long when overbought)
    rsi_ok_short = rsi < 70
    rsi_ok_long  = rsi > 30

    signals = pd.Series(0, index=df.index)
    signals[confirm_fade_bear & cvd_not_confirming_bull & rsi_ok_short] = -1  # Short the bull trap
    signals[confirm_fade_bull & cvd_not_confirming_bear & rsi_ok_long]  =  1  # Long the bear trap
    return signals


# ─────────────────────────────────────────────────────────
# MR8 — SMC Liquidity Sweep (Refined for 5m)
# Same concept as MR3 but with:
#   - Must form a WICK candle (not just close back inside)
#   - Stronger volume requirement (RVOL > 2.0 = real institutional sweep)
#   - Wider sweep depth (0.7 ATR min)
#   - CVD must show immediate reversal
# ─────────────────────────────────────────────────────────
def strat_mr8_smc_sweep(df, p):
    atr  = get_atr(df)
    rvol = get_rvol(df)
    cvd  = calc_cvd(df, 5)
    mfi  = calc_mfi(df, 14)
    chop = calc_choppiness_index(df, 14)

    lookback = 12  # Swing detection lookback

    swing_high = df['high'].rolling(lookback).max().shift(1)
    swing_low  = df['low'].rolling(lookback).min().shift(1)

    # QUALITY sweep: wick went beyond swing, CLOSED back inside = trap candle
    # Plus volume surge (institutional stop-hunt)
    # Plus meaningful wick depth (0.7 ATR min to filter noise)
    
    bull_sweep_quality = (
        (df['low'] < swing_low) &                          # wick went below
        (df['close'] > swing_low) &                        # closed back above
        (rvol > 2.0) &                                     # strong volume
        ((swing_low - df['low']) > 0.7 * atr)              # meaningful wick
    )

    bear_sweep_quality = (
        (df['high'] > swing_high) &                        # wick went above
        (df['close'] < swing_high) &                       # closed back below
        (rvol > 2.0) &                                     # strong volume
        ((df['high'] - swing_high) > 0.7 * atr)            # meaningful wick
    )

    # CVD shows immediate reversal on the sweep candle itself
    cvd_reversing_bull = cvd > cvd.shift(1)   # CVD positive on the sweep candle
    cvd_reversing_bear = cvd < cvd.shift(1)   # CVD negative on the sweep candle

    # MFI exhaustion zone
    mfi_bull = mfi < 35
    mfi_bear = mfi > 65

    signals = pd.Series(0, index=df.index)
    signals[bull_sweep_quality & cvd_reversing_bull & mfi_bull] =  1
    signals[bear_sweep_quality & cvd_reversing_bear & mfi_bear] = -1
    return signals


# ─────────────────────────────────────────────────────────
# DATA LOADER
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
            if len(oos_df) < 200:
                continue
            try:
                _, agg, _ = engine.run_multi_symbol(
                    {symbol: oos_df}, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
                if agg['total_trades'] >= 5:
                    all_pf.append(agg['profit_factor'])
            except Exception:
                pass
    return round(np.mean(all_pf), 3) if all_pf else 0.0


# ─────────────────────────────────────────────────────────
# MAIN — Test with 2 parameter sets
# ─────────────────────────────────────────────────────────
def main():
    print("Loading 5m datasets...")
    dfs = load_data()
    if not dfs:
        print("ERROR: No CSV files found.")
        return
    print(f"Loaded {len(dfs)} symbols: {sorted(dfs.keys())}\n")

    strategies = {
        "MR2_VWAP_Divergence":    strat_mr2_refined,
        "MR6_Keltner_MR":         strat_mr6_keltner,
        "MR7_BigCandle_Fade":     strat_mr7_big_candle_fade,
        "MR8_SMC_Sweep_Quality":  strat_mr8_smc_sweep,
    }

    # Two parameter sets to test:
    # A: Wider hold time — let MR complete naturally
    # B: Wide SL to keep WR high + very long hold
    param_sets = {
        "A [SL=1.5 TP=2.0 hold=100]": {'sl_atr': 1.5, 'tp_atr': 2.0, 'max_bars_hold': 100, 'risk_pct': 0.01, 'trailing': False},
        "B [SL=1.5 TP=3.0 hold=100]": {'sl_atr': 1.5, 'tp_atr': 3.0, 'max_bars_hold': 100, 'risk_pct': 0.01, 'trailing': False},
    }

    engine  = BacktestCore()
    all_results = []

    for param_label, params in param_sets.items():
        print("=" * 65)
        print(f"  PARAMS: {param_label}")
        print("=" * 65)

        for name, fn in strategies.items():
            print(f"\n  [{name}]")
            try:
                _, agg, _ = engine.run_multi_symbol(
                    dfs, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
                trades = agg['total_trades']
                pf     = agg['profit_factor']
                wr     = agg['win_rate']
                sharpe = agg['sharpe_ratio']

                if trades < 20:
                    print(f"    SKIPPED ({trades} trades)")
                    continue

                print(f"    Std  -> PF:{pf:.3f} | WR:{wr:.1f}% | Sharpe:{sharpe:.3f} | Trades:{trades}")
                oos_pf = run_walkforward(engine, dfs, fn, params)
                status = "PASS" if pf >= 1.0 and oos_pf >= 1.0 else ("STD_PASS" if pf >= 1.0 else "FAIL")
                print(f"    WF   -> OOS_PF:{oos_pf:.3f}  |  Status: {status}")

                all_results.append({
                    'Params':      param_label,
                    'Strategy':    name,
                    'Std_PF':      round(pf, 3),
                    'WF_OOS_PF':   oos_pf,
                    'Win_Rate_%':  round(wr, 2),
                    'Sharpe':      round(sharpe, 3),
                    'Trades':      trades,
                    'Status':      status,
                })
            except Exception as e:
                print(f"    ERROR: {e}")

    print("\n" + "=" * 65)
    print("  MASTER RANKINGS — All strategies, all param sets")
    print("=" * 65)
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values(
            by=['Std_PF', 'Win_Rate_%', 'Sharpe'], ascending=False)
        print(df_res.to_string(index=False))
        df_res.to_csv("mr_v4_rankings.csv", index=False)
        print("\nSaved to mr_v4_rankings.csv")

        # Highlight passing strategies
        passing = df_res[df_res['Std_PF'] >= 1.0]
        if not passing.empty:
            print("\n*** PROFITABLE STRATEGIES FOUND ***")
            print(passing.to_string(index=False))
        else:
            print("\nNo strategy crossed PF >= 1.0 yet.")


if __name__ == "__main__":
    main()
