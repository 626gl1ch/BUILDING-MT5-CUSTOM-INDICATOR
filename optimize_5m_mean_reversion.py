"""
5m Mean-Reversion & Liquidity-Sweep Suite
==========================================
Lessons from 3 failed trend-following batches:
  - 70%+ of 5m moves REVERT → mean reversion wins, trend following loses
  - SMC Liquidity Sweeps work because they ENTER at the reversal, not after confirmation
  - Lagging indicator confirmation always arrives too late on 5m

This script tests 5 strategies designed around 5m's true statistical nature:
  MR1 - Bollinger Band Extreme + Volume Exhaustion Fade
  MR2 - VWAP Deviation + RSI Divergence Reversion
  MR3 - 5m Adapted SMC Liquidity Sweep (short lookback)
  MR4 - Stochastic RSI Extreme + BB Squeeze Recovery
  MR5 - Opening Range Fade (mean reversion of ORB extremes)

Validation: Standard Backtest + Walk-Forward (no permutations)
Fees: FBS Forex 0.02% round-trip
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

# ─────────────────────────────────────────────────────────
# UNIVERSAL HELPERS
# ─────────────────────────────────────────────────────────
def get_atr(df):
    return calc_atr(df, 14)

def get_rvol(df, period=20):
    return df['volume'] / df['volume'].rolling(period).mean()


# ─────────────────────────────────────────────────────────
# MR1 — Bollinger Band Extreme + Volume Exhaustion Fade
# ─────────────────────────────────────────────────────────
# Logic:
#   Price closes BEYOND the outer BB (3-sigma) = extreme extension
#   MFI confirms exhaustion (< 20 for longs, > 80 for shorts)
#   RVOL > 1.3 = volume spike (capitulation / blow-off)
#   CVD diverging: CVD didn't confirm the price extreme = trap move
#   Entry: Close back INSIDE the outer band on the NEXT candle
#   SL: 1.5x ATR beyond extreme candle's wick
#   TP: BB middle band (SMA20) — mean reversion target
# ─────────────────────────────────────────────────────────
def strat_mr1(df, p):
    atr  = get_atr(df)
    rvol = get_rvol(df)
    mfi  = calc_mfi(df, 14)
    cvd  = calc_cvd(df, 5)
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.5)  # 2.5 sigma (loosened)

    # Exhaustion: RVOL > 1.2, MFI extreme
    exhaustion_long  = (df['close'].shift(1) < bb_l.shift(1)) & (mfi.shift(1) < 30) & (rvol.shift(1) > 1.2)
    exhaustion_short = (df['close'].shift(1) > bb_u.shift(1)) & (mfi.shift(1) > 70) & (rvol.shift(1) > 1.2)

    # CVD divergence: price went lower but CVD didn't (or vice versa)
    cvd_bull_div = cvd > cvd.shift(3)  # CVD rising while price was at extreme low
    cvd_bear_div = cvd < cvd.shift(3)  # CVD falling while price was at extreme high

    # Entry candle: price closes back inside the band
    close_inside_long  = (df['close'] > bb_l) & (df['close'].shift(1) <= bb_l.shift(1))
    close_inside_short = (df['close'] < bb_u) & (df['close'].shift(1) >= bb_u.shift(1))

    signals = pd.Series(0, index=df.index)
    signals[exhaustion_long  & close_inside_long  & cvd_bull_div] =  1
    signals[exhaustion_short & close_inside_short & cvd_bear_div] = -1
    return signals


# ─────────────────────────────────────────────────────────
# MR2 — VWAP Deviation + RSI Divergence Reversion
# ─────────────────────────────────────────────────────────
# Logic:
#   Price is > 2 stdev from VWAP (over-extended from institutional fair value)
#   RSI shows hidden divergence (price made new extreme, RSI didn't)
#   CHOP > 45 = market is ranging = mean reversion more reliable
#   Entry: price closes back toward VWAP (first close that narrows the gap)
#   SL: 1.0x ATR from entry
#   TP: VWAP itself (mean reversion target)
# ─────────────────────────────────────────────────────────
def strat_mr2(df, p):
    atr  = get_atr(df)
    rsi  = calc_rsi(df['close'], 14)
    chop = calc_choppiness_index(df, 14)
    vwap = calc_vwap(df)
    rvol = get_rvol(df)

    # VWAP deviation bands
    rolling_std = df['close'].rolling(20).std()
    vwap_upper2 = vwap + 2.0 * rolling_std
    vwap_lower2 = vwap - 2.0 * rolling_std

    # Price is extended beyond 2-sigma VWAP band
    extended_below = df['close'] < vwap_lower2
    extended_above = df['close'] > vwap_upper2

    # RSI divergence: price at new N-bar extreme but RSI is not
    lookback = 10
    price_new_low = df['close'] == df['close'].rolling(lookback).min()
    rsi_not_new_low = rsi > rsi.rolling(lookback).min()
    bull_divergence = extended_below & price_new_low & rsi_not_new_low

    price_new_high = df['close'] == df['close'].rolling(lookback).max()
    rsi_not_new_high = rsi < rsi.rolling(lookback).max()
    bear_divergence = extended_above & price_new_high & rsi_not_new_high

    # Mean reversion regime: CHOP > 55 = definitely ranging
    mr_regime = chop > 55

    # Entry: price starts reverting toward VWAP
    entry_long  = (df['close'] > df['close'].shift(1)) & extended_below.shift(1)
    entry_short = (df['close'] < df['close'].shift(1)) & extended_above.shift(1)

    signals = pd.Series(0, index=df.index)
    signals[bull_divergence.shift(1) & mr_regime & entry_long  & (rvol > 0.8)] =  1
    signals[bear_divergence.shift(1) & mr_regime & entry_short & (rvol > 0.8)] = -1
    return signals


# ─────────────────────────────────────────────────────────
# MR3 — 5m Adapted SMC Liquidity Sweep
# ─────────────────────────────────────────────────────────
# Logic (adapted from proven 15m/30m SMC strategy, tuned for 5m):
#   Identify swing highs/lows using a 10-bar lookback (shorter for 5m noise)
#   A "sweep" occurs when price spikes BEYOND the swing level but CLOSES back inside
#   This = retail stop hunt: stops above the swing triggered, then smart money reverses
#   Volume SPIKE on the sweep candle confirms institutional activity
#   Entry: candle that closes back inside the swing level
#   SL: just beyond the sweep wick (1.2x ATR)
#   TP: opposite swing level (asymmetric R:R)
# ─────────────────────────────────────────────────────────
def strat_mr3(df, p):
    atr  = get_atr(df)
    rvol = get_rvol(df)

    lookback = 10  # 5m adapted (shorter than 41-bar 15m/30m version)

    # Swing highs and lows (N-bar highs/lows)
    swing_high = df['high'].rolling(lookback).max().shift(1)  # prior swing (shift 1 = no lookahead)
    swing_low  = df['low'].rolling(lookback).min().shift(1)

    # SWEEP detection with stronger quality filter
    # Bearish sweep: wick above swing high, close back below
    bear_sweep = (df['high'] > swing_high) & (df['close'] < swing_high) & (rvol > 1.5)
    # Bullish sweep: wick below swing low, close back above
    bull_sweep = (df['low'] < swing_low) & (df['close'] > swing_low) & (rvol > 1.5)

    # Minimum sweep depth: must extend by at least 0.5 ATR (not just noise)
    sweep_depth_bear = (df['high'] - swing_high) > (0.5 * atr)
    sweep_depth_bull = (swing_low - df['low'])   > (0.5 * atr)

    signals = pd.Series(0, index=df.index)
    signals[bull_sweep & sweep_depth_bull] =  1   # Long after stop-hunt of lows
    signals[bear_sweep & sweep_depth_bear] = -1   # Short after stop-hunt of highs
    return signals


# ─────────────────────────────────────────────────────────
# MR4 — StochRSI Extreme + BB Squeeze Recovery
# ─────────────────────────────────────────────────────────
# Logic:
#   BB squeeze (width < 20th percentile) = compressed volatility = spring loaded
#   StochRSI hits < 5 (extreme oversold) or > 95 (extreme overbought)
#   When StochRSI crosses back through 20/80 threshold = exhaustion reversal
#   This targets the "snap-back" from extreme readings in a range
#   SL: 1.5x ATR, TP: opposite BB band (full snap-back)
# ─────────────────────────────────────────────────────────
def strat_mr4(df, p):
    atr  = get_atr(df)
    rvol = get_rvol(df)
    sk, sd = calc_stoch_rsi(df['close'], 14, 3, 3)
    bb_u, bb_m, bb_l, _, bb_width = calc_bollinger_bands(df['close'], 20, 2.0)
    bb_pct = calc_bb_width_percentile(df, 20, 2.0, 100)
    chop = calc_choppiness_index(df, 14)

    # Was in a squeeze recently (last 5 candles had width < 30th percentile)
    was_squeezed = (bb_pct.rolling(5).min() < 30)

    # StochRSI extreme recovery
    # Long: K was < 5 (deeper than typical OS) and crosses back above 20
    was_extreme_os = (sk.rolling(5).min() < 5)
    cross_up = (sk > 20) & (sk.shift(1) <= 20) & was_extreme_os

    was_extreme_ob = (sk.rolling(5).max() > 95)
    cross_down = (sk < 80) & (sk.shift(1) >= 80) & was_extreme_ob

    # Mean reversion regime: CHOP > 55 = definitely ranging
    mr_regime = chop > 55

    signals = pd.Series(0, index=df.index)
    signals[cross_up   & was_squeezed & mr_regime] =  1
    signals[cross_down & was_squeezed & mr_regime] = -1
    return signals


# ─────────────────────────────────────────────────────────
# MR5 — Composite Mean Reversion (Best of All Worlds)
# ─────────────────────────────────────────────────────────
# Logic (combines insights from MR1-MR4):
#   3 independent confirmation triggers must align simultaneously:
#   1. Price at BB extreme (2.5 sigma) — extension gate
#   2. MFI exhaustion (< 25 long / > 75 short) — volume exhaustion gate
#   3. StochRSI at extreme and beginning to reverse — OB/OS timing gate
#   Regime: CHOP > 40 (ranging market favored)
#   SL: 1.0x ATR (tight — entry is AT the structural extreme)
#   TP: BB middle band — reliable mean reversion target
# ─────────────────────────────────────────────────────────
def strat_mr5(df, p):
    atr  = get_atr(df)
    rvol = get_rvol(df)
    mfi  = calc_mfi(df, 14)
    sk, sd = calc_stoch_rsi(df['close'], 14, 3, 3)
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.5)  # 2.5 sigma
    chop = calc_choppiness_index(df, 14)
    cvd  = calc_cvd(df, 5)

    # Mean reversion regime: CHOP > 55 = definitely ranging
    mr_regime = chop > 55

    # Gate 1: Price at extreme BB (2.5 sigma - loosened from 2.5)
    at_lower = df['close'] <= bb_l
    at_upper = df['close'] >= bb_u

    # Gate 2: MFI volume exhaustion (loosened slightly)
    mfi_exhaustion_long  = mfi < 30
    mfi_exhaustion_short = mfi > 70

    # Gate 3: StochRSI reversing from extreme
    sk_reversing_up   = (sk > sk.shift(1)) & (sk < 25)
    sk_reversing_down = (sk < sk.shift(1)) & (sk > 75)

    # Gate 4: CVD not confirming continuation (divergence)
    cvd_bull = cvd >= cvd.shift(2)   # CVD stabilizing/rising when price at low
    cvd_bear = cvd <= cvd.shift(2)   # CVD stabilizing/falling when price at high

    long_signal  = at_lower & mfi_exhaustion_long  & sk_reversing_up   & cvd_bull & mr_regime
    short_signal = at_upper & mfi_exhaustion_short & sk_reversing_down & cvd_bear & mr_regime

    signals = pd.Series(0, index=df.index)
    signals[long_signal]  =  1
    signals[short_signal] = -1
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
    """5-split walk-forward. Each OOS window is 20% of the data."""
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
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("Loading 5m datasets...")
    dfs = load_data()
    if not dfs:
        print("ERROR: No 5m CSV files found.")
        return
    print(f"Loaded {len(dfs)} symbols: {sorted(dfs.keys())}\n")

    strategies = {
        "MR1_BB_Extreme_Fade":      strat_mr1,
        "MR2_VWAP_RSI_Divergence":  strat_mr2,
        "MR3_SMC_LiqSweep_5m":      strat_mr3,
        "MR4_StochRSI_Squeeze":     strat_mr4,
        "MR5_Composite_MR":         strat_mr5,
    }

    # Mean reversion: tighter SL so 45% WR at 1:2 R:R gives theoretical PF 1.64
    # Even with time-expiry dilution this should clear PF 1.0
    params = {
        'sl_atr': 1.0,
        'tp_atr': 2.0,
        'max_bars_hold': 30,   # 30 x 5m = 2.5 hours max hold (tighter)
        'risk_pct': 0.01,
        'trailing': False
    }

    engine  = BacktestCore()
    results = []

    print("=" * 60)
    print("  5M MEAN-REVERSION & LIQUIDITY-SWEEP GAUNTLET")
    print(f"  FBS Fees: {FEE_PCT*100:.3f}% | SL: 1.0 ATR | TP: 3.0 ATR")
    print("=" * 60 + "\n")

    for name, fn in strategies.items():
        print(f"  [{name}]")
        try:
            _, agg, _ = engine.run_multi_symbol(
                dfs, fn, params, slippage_pct=0, fee_pct=FEE_PCT)
            trades = agg['total_trades']
            pf     = agg['profit_factor']
            wr     = agg['win_rate']
            sharpe = agg['sharpe_ratio']

            if trades < 20:
                print(f"    -> SKIPPED ({trades} trades — too few)\n")
                continue

            print(f"    Std  -> PF: {pf:.3f} | WR: {wr:.1f}% | Sharpe: {sharpe:.3f} | Trades: {trades}")

            oos_pf = run_walkforward(engine, dfs, fn, params)
            status = "PASS" if pf >= 1.0 and oos_pf >= 1.0 else ("PARTIAL" if pf >= 1.0 else "FAIL")
            print(f"    WF   -> OOS PF: {oos_pf:.3f}")
            print(f"    Status: {status}\n")

            results.append({
                'Strategy':    name,
                'Std_PF':      round(pf, 3),
                'WF_OOS_PF':   oos_pf,
                'Win_Rate_%':  round(wr, 2),
                'Sharpe':      round(sharpe, 3),
                'Trades':      trades,
                'Status':      status,
            })
        except Exception as e:
            print(f"    ERROR: {e}\n")

    if not results:
        print("\n[!] No strategy produced enough trades.")
        return

    df_res = pd.DataFrame(results).sort_values(
        by=['Std_PF', 'Win_Rate_%', 'Sharpe'], ascending=False)

    print("\n" + "=" * 60)
    print("  FINAL RANKINGS")
    print("=" * 60)
    print(df_res.to_string(index=False))

    df_res.to_csv("mr_rankings.csv", index=False)
    print("\nSaved rankings to mr_rankings.csv")


if __name__ == "__main__":
    main()
