"""
═══════════════════════════════════════════════════════════════════════════════
  COMPREHENSIVE MEAN REVERSION STRATEGY ENGINE - 5m & 15m Timeframes
  Version 1.0  |  Designed for High-Noise Scalping Environments
═══════════════════════════════════════════════════════════════════════════════

12 Unique Mean Reversion Strategies, each built with:
  ✓ Universal Market Condition Filters (blocks bad environments)
  ✓ Optimized indicator settings specifically tuned for 5m and 15m
  ✓ ATR-based SL/TP for adaptive risk management
  ✓ No lookahead bias (signal on close[i] -> entry at open[i+1])

MARKET CONDITION FILTERS (in every strategy):
  • ADX(7/10)   - blocks strong trending environments (ADX > threshold)
  • ATR(7/10)   - blocks dead/low-volatility markets (< 20th percentile)
  • CHOP(14)    - blocks pure directionless chop (> 61.8)
  • MACD(5,13,3 / 8,21,5) - blocks momentum continuation trades

STRATEGIES:
  MR-A  ALMA Bounce + BB Extreme + RSI Reset
  MR-B  StochRSI Snap + MACD Momentum Divergence
  MR-C  BB Squeeze + ALMA/EMA9 Crossover Pull
  MR-D  ADX-Filtered RSI Extreme + ATR Spike Reversal
  MR-E  MACD Fade + StochRSI Double Exhaustion
  MR-F  BB %B Extremity + ATR Blow-off + RSI Divergence
  MR-G  ALMA-EMA9 Channel Deviation Fade
  MR-H  RSI-ADX Precision Scalp (dual StochRSI confirm)
  MR-I  MACD Histogram Zero-Cross + BB Extreme Fade
  MR-J  Choppiness Regime Range Play + StochRSI
  MR-K  Triple Confluence: ALMA + BB + StochRSI
  MR-L  EMA9 Mean Reversion + MACD Divergence + ADX Gate

OUTPUT: mr_comprehensive_rankings.csv (ranked by PF desc -> WR desc)
"""

import glob
import warnings
import numpy as np
import pandas as pd
from backtest_core import BacktestCore
from indicators_library import (
    calc_alma, calc_ema, calc_rsi, calc_stoch_rsi,
    calc_atr, calc_adx, calc_bollinger_bands,
    calc_macd, calc_choppiness_index, calc_bb_width_percentile,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  TIMEFRAME PARAMETER SETS
# ═══════════════════════════════════════════════════════════════════════════

TF_PARAMS = {
    "5min": {
        "adx_period":    7,
        "atr_period":    7,
        "chop_period":   14,
        "macd_fast":     5,
        "macd_slow":     13,
        "macd_signal":   3,
        "rsi_period":    7,
        "srsi_period":   7,
        "srsi_k":        3,
        "srsi_d":        3,
        "bb_period":     15,
        "bb_std":        2.0,
        "alma_period":   9,
        "alma_sigma":    6,
        "alma_offset":   0.85,
        "ema_fast":      9,
        "ema_slow":      21,
        "adx_max_trend": 28,     # ADX above this = trending, avoid
        "adx_sweet_spot_lo": 12, # ADX below this = too flat/dead
        "chop_max":      61.8,   # Above = pure chop
        "chop_mid_lo":   38.0,   # Ranging zone lower bound
        "sl_atr":        1.2,
        "tp_atr":        2.4,
        "max_bars_hold": 30,     # 30 × 5m = 2.5 hours max
        "risk_pct":      0.01,
    },
    "15min": {
        "adx_period":    10,
        "atr_period":    10,
        "chop_period":   14,
        "macd_fast":     8,
        "macd_slow":     21,
        "macd_signal":   5,
        "rsi_period":    10,
        "srsi_period":   10,
        "srsi_k":        3,
        "srsi_d":        3,
        "bb_period":     20,
        "bb_std":        2.0,
        "alma_period":   9,
        "alma_sigma":    6,
        "alma_offset":   0.85,
        "ema_fast":      9,
        "ema_slow":      21,
        "adx_max_trend": 30,
        "adx_sweet_spot_lo": 10,
        "chop_max":      61.8,
        "chop_mid_lo":   40.0,
        "sl_atr":        1.5,
        "tp_atr":        3.0,
        "max_bars_hold": 20,     # 20 × 15m = 5 hours max
        "risk_pct":      0.01,
    },
}

FEE_PCT      = 0.0002   # Forex round-trip commission equivalent
SLIPPAGE_PCT = 0.0001   # Light slippage for liquid pairs


# ═══════════════════════════════════════════════════════════════════════════
#  UNIVERSAL MARKET FILTER HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_market_filters(df, p):
    """
    Returns a boolean Series: True = conditions are SAFE to trade MR.
    Blocks: strong trends, dead markets, pure chop, high MACD momentum.
    """
    adx, pdi, mdi  = calc_adx(df, p["adx_period"])
    atr             = calc_atr(df, p["atr_period"])
    chop            = calc_choppiness_index(df, p["chop_period"])
    macd_l, macd_s, macd_h = calc_macd(
        df["close"], p["macd_fast"], p["macd_slow"], p["macd_signal"]
    )

    # ATR percentile filter: avoid dead markets
    atr_floor = atr.rolling(50).quantile(0.20)
    vol_ok     = atr > atr_floor

    # ADX filter: avoid strong trends
    trend_ok   = (adx > p["adx_sweet_spot_lo"]) & (adx < p["adx_max_trend"])

    # Choppiness filter: avoid pure directionless chop (> 61.8) but also
    # avoid ultra-trending (< 25). Best MR zone is 38–61.8
    chop_ok    = (chop > p["chop_mid_lo"]) & (chop < p["chop_max"])

    # MACD momentum filter: avoid when histogram is strong & expanding
    hist_strong = abs(macd_h) > abs(macd_h).rolling(20).mean() * 1.5
    macd_ok     = ~hist_strong  # True = histogram NOT strongly expanding

    return vol_ok & trend_ok & chop_ok & macd_ok


def _atr(df, p):
    return calc_atr(df, p["atr_period"])


def _alma(df, p):
    return calc_alma(df["close"], p["alma_period"], p["alma_sigma"], p["alma_offset"])


def _ema9(df, p):
    return calc_ema(df["close"], p["ema_fast"])


def _ema21(df, p):
    return calc_ema(df["close"], p["ema_slow"])


def _rsi(df, p):
    return calc_rsi(df["close"], p["rsi_period"])


def _srsi(df, p):
    return calc_stoch_rsi(
        df["close"], p["srsi_period"], p["srsi_k"], p["srsi_d"]
    )


def _bb(df, p):
    return calc_bollinger_bands(df["close"], p["bb_period"], p["bb_std"])


def _macd(df, p):
    return calc_macd(df["close"], p["macd_fast"], p["macd_slow"], p["macd_signal"])


# ═══════════════════════════════════════════════════════════════════════════
#  12 MEAN REVERSION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# MR-A: ALMA Bounce + BB Extreme + RSI Reset
# ───────────────────────────────────────────────────────────────────────────
# Rationale: ALMA(9) is the "smart mean" - it weights recent bars more but
# smooths like a Gaussian filter. When price departs 2-sigma from BB AND RSI
# is extreme then starts reversing back toward ALMA, that is textbook MR.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_A(df, p):
    safe   = _get_market_filters(df, p)
    atr    = _atr(df, p)
    alma   = _alma(df, p)
    rsi    = _rsi(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)

    # Gate: price at BB extreme (beyond ±2 sigma)
    at_lower = df["close"] <= bb_l
    at_upper = df["close"] >= bb_u

    # Gate: RSI confirms oversold / overbought
    rsi_os = rsi < 25
    rsi_ob = rsi > 75

    # Trigger: RSI ticking back from extreme (first reversal bar)
    rsi_tick_up   = (rsi > rsi.shift(1)) & rsi_os.shift(1)
    rsi_tick_down = (rsi < rsi.shift(1)) & rsi_ob.shift(1)

    # Require price to be on the correct side of ALMA (extended away)
    below_alma = df["close"] < alma
    above_alma = df["close"] > alma

    signals = pd.Series(0, index=df.index)
    long_cond  = safe & at_lower.shift(1) & rsi_tick_up  & below_alma
    short_cond = safe & at_upper.shift(1) & rsi_tick_down & above_alma

    signals[long_cond]  =  1
    signals[short_cond] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= alma
    exit_short = df["low"] <= alma
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-B: StochRSI Snap + MACD Momentum Divergence
# ───────────────────────────────────────────────────────────────────────────
# Rationale: When StochRSI hits extreme (<10 / >90) AND the MACD histogram
# is diverging (price going to new extreme but MACD histogram shrinking), it
# signals momentum exhaustion. The cross of StochRSI back through 20/80 is
# the trigger. EMA9 is the TP target.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_B(df, p):
    safe          = _get_market_filters(df, p)
    sk, sd        = _srsi(df, p)
    macd_l, macd_s, macd_h = _macd(df, p)
    ema9          = _ema9(df, p)

    # StochRSI was at deep extreme recently (last 5 bars)
    was_extreme_os = sk.rolling(5).min() < 10
    was_extreme_ob = sk.rolling(5).max() > 90

    # StochRSI crosses back through 20 (long) / 80 (short)
    cross_up   = (sk > 20) & (sk.shift(1) <= 20) & was_extreme_os
    cross_down = (sk < 80) & (sk.shift(1) >= 80) & was_extreme_ob

    # MACD histogram divergence: histogram shrinking while price at extreme
    hist_shrinking_bull = (macd_h > macd_h.shift(1)) & (macd_h < 0)  # Turning from negative
    hist_shrinking_bear = (macd_h < macd_h.shift(1)) & (macd_h > 0)  # Turning from positive

    # Price stretched from EMA9
    below_ema9 = df["close"] < ema9 * 0.999
    above_ema9 = df["close"] > ema9 * 1.001

    signals = pd.Series(0, index=df.index)
    signals[safe & cross_up   & hist_shrinking_bull & below_ema9] =  1
    signals[safe & cross_down & hist_shrinking_bear & above_ema9] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= ema9
    exit_short = df["low"] <= ema9
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-C: BB Squeeze + ALMA/EMA9 Crossover Mean Pull
# ───────────────────────────────────────────────────────────────────────────
# Rationale: A BB squeeze compresses energy. When it releases with StochRSI
# extreme AND ALMA crosses EMA9 (faster MA pulling toward slower = mean pull),
# this is a high-probability snap-back to the BB midline.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_C(df, p):
    safe          = _get_market_filters(df, p)
    alma          = _alma(df, p)
    ema9          = _ema9(df, p)
    sk, sd        = _srsi(df, p)
    bb_u, bb_m, bb_l, bb_pct, bb_width = _bb(df, p)

    # BB squeeze: width below 20th percentile over 100 bars
    bb_pct_rank = calc_bb_width_percentile(df, p["bb_period"], p["bb_std"], 100)
    was_squeezed = bb_pct_rank.rolling(8).min() < 25

    # ALMA crosses above EMA9 (ALMA rising faster -> pull up)
    alma_cross_up   = (alma > ema9) & (alma.shift(1) <= ema9.shift(1))
    alma_cross_down = (alma < ema9) & (alma.shift(1) >= ema9.shift(1))

    # StochRSI extreme
    srsi_os = sk < 25
    srsi_ob = sk > 75

    # Price at or beyond BB band
    near_lower = bb_pct < 0.15
    near_upper = bb_pct > 0.85

    signals = pd.Series(0, index=df.index)
    signals[safe & was_squeezed & alma_cross_up   & srsi_os & near_lower] =  1
    signals[safe & was_squeezed & alma_cross_down & srsi_ob & near_upper] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= alma
    exit_short = df["low"] <= alma
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-D: ADX-Filtered RSI Extreme + ATR Expansion Spike Reversal
# ───────────────────────────────────────────────────────────────────────────
# Rationale: On small TFs, volatility spikes (ATR expansion) in a non-trending
# environment (ADX low) with RSI extreme = exhaustion spike. The next candle
# closing back inside BB confirms the reversal. This is the "wick fade".
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_D(df, p):
    safe          = _get_market_filters(df, p)
    atr           = _atr(df, p)
    adx, pdi, mdi = calc_adx(df, p["adx_period"])
    rsi           = _rsi(df, p)
    ema21         = _ema21(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)

    # ATR spike: current ATR > 1.25x of rolling 20-bar average
    atr_avg   = atr.rolling(20).mean()
    atr_spike = atr > atr_avg * 1.25

    # Very strong ADX override (not just the market filter)
    adx_ok    = adx < 22

    # RSI at extreme
    rsi_os = rsi < 22
    rsi_ob = rsi > 78

    # BB %B near extreme (0 = at lower, 1 = at upper)
    near_lower = bb_pct < 0.08
    near_upper = bb_pct > 0.92

    # Entry: next bar closes BACK inside BB
    close_inside_long  = (df["close"] > bb_l) & (df["close"].shift(1) <= bb_l.shift(1))
    close_inside_short = (df["close"] < bb_u) & (df["close"].shift(1) >= bb_u.shift(1))

    signals = pd.Series(0, index=df.index)
    long_setup  = (rsi_os.shift(1) & near_lower.shift(1) & atr_spike.shift(1) & adx_ok.shift(1))
    short_setup = (rsi_ob.shift(1) & near_upper.shift(1) & atr_spike.shift(1) & adx_ok.shift(1))

    signals[safe & long_setup  & close_inside_long]  =  1
    signals[safe & short_setup & close_inside_short] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= ema21
    exit_short = df["low"] <= ema21
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-E: MACD Fade + StochRSI Double Exhaustion
# ───────────────────────────────────────────────────────────────────────────
# Rationale: When MACD line crosses back through signal (momentum dying)
# AND StochRSI was OS/OB and is now returning toward neutral, both momentum
# systems confirm exhaustion. EMA9 is both the stretched reference and TP.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_E(df, p):
    safe             = _get_market_filters(df, p)
    macd_l, macd_s, macd_h = _macd(df, p)
    sk, sd           = _srsi(df, p)
    ema9             = _ema9(df, p)

    # MACD cross: line crosses back through signal (momentum reversal)
    macd_cross_up   = (macd_l > macd_s) & (macd_l.shift(1) <= macd_s.shift(1))
    macd_cross_down = (macd_l < macd_s) & (macd_l.shift(1) >= macd_s.shift(1))

    # StochRSI returning from extreme (was OS, now moving up; was OB, now down)
    was_os = sk.rolling(10).min() < 15
    was_ob = sk.rolling(10).max() > 85
    srsi_recovering_long  = (sk > sd) & (sk > sk.shift(2)) & was_os
    srsi_recovering_short = (sk < sd) & (sk < sk.shift(2)) & was_ob

    # Price must be stretched from EMA9
    dist_from_ema9 = (df["close"] - ema9) / ema9
    below_ema9 = dist_from_ema9 < -0.001
    above_ema9 = dist_from_ema9 > 0.001

    signals = pd.Series(0, index=df.index)
    signals[safe & macd_cross_up   & srsi_recovering_long  & below_ema9] =  1
    signals[safe & macd_cross_down & srsi_recovering_short & above_ema9] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= ema9
    exit_short = df["low"] <= ema9
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-F: BB %B Extremity + ATR Blow-off + RSI Divergence
# ───────────────────────────────────────────────────────────────────────────
# Rationale: When price glues to the outer BB (%B < 0.05 or > 0.95) with an
# ATR blow-off spike AND RSI diverges (price at new extreme but RSI is less
# extreme than its own prior reading), it's a classic exhaustion divergence.
# The reversal trigger is %B moving back inside 0.1–0.9.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_F(df, p):
    safe   = _get_market_filters(df, p)
    atr    = _atr(df, p)
    rsi    = _rsi(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)

    # %B extreme: hugging outer band
    at_lower_band = bb_pct < 0.05
    at_upper_band = bb_pct > 0.95

    # ATR blow-off: ATR spike beyond 1.3x rolling mean
    atr_avg   = atr.rolling(20).mean()
    blowoff   = atr > atr_avg * 1.3

    # RSI divergence: price at new N-bar extreme, RSI is not
    lookback = 12
    price_new_low = df["close"] == df["close"].rolling(lookback).min()
    rsi_not_new_low = rsi > rsi.rolling(lookback).min() + 3   # RSI making higher low

    price_new_high = df["close"] == df["close"].rolling(lookback).max()
    rsi_not_new_high = rsi < rsi.rolling(lookback).max() - 3  # RSI making lower high

    bull_divergence = at_lower_band & price_new_low & rsi_not_new_low
    bear_divergence = at_upper_band & price_new_high & rsi_not_new_high

    # Entry trigger: %B moves back inside (reversal confirmed)
    bb_move_inside_long  = (bb_pct > 0.10) & (bb_pct.shift(1) <= 0.10)
    bb_move_inside_short = (bb_pct < 0.90) & (bb_pct.shift(1) >= 0.90)

    signals = pd.Series(0, index=df.index)
    signals[safe & bull_divergence.shift(1) & blowoff.shift(1) & bb_move_inside_long]  =  1
    signals[safe & bear_divergence.shift(1) & blowoff.shift(1) & bb_move_inside_short] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= bb_m
    exit_short = df["low"] <= bb_m
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-G: ALMA-EMA9 Channel Deviation Fade
# ───────────────────────────────────────────────────────────────────────────
# Rationale: ALMA(9) is smoother than EMA9 (Gaussian weighting). When price
# deviates > 1.5x ATR from ALMA AND EMA9 is starting to revert (flat or
# turning), AND MACD histogram is losing power, price is over-extended and
# will snap back to ALMA.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_G(df, p):
    safe          = _get_market_filters(df, p)
    alma          = _alma(df, p)
    ema9          = _ema9(df, p)
    atr           = _atr(df, p)
    sk, sd        = _srsi(df, p)
    macd_l, macd_s, macd_h = _macd(df, p)

    # Price deviation from ALMA in ATR units
    deviation   = (df["close"] - alma) / atr.replace(0, np.nan)
    far_below   = deviation < -1.5
    far_above   = deviation >  1.5

    # EMA9 not strongly trending (flat slope)
    ema9_slope  = (ema9 - ema9.shift(3)) / ema9.shift(3)
    ema_flat    = ema9_slope.abs() < 0.002

    # MACD histogram shrinking (losing momentum)
    hist_shrink_bull = (macd_h > macd_h.shift(1)) & (macd_h < 0)   # Rising from negative -> toward 0
    hist_shrink_bear = (macd_h < macd_h.shift(1)) & (macd_h > 0)   # Falling from positive -> toward 0

    # StochRSI beginning reversal
    srsi_turning_up   = (sk > sk.shift(1)) & (sk < 40)
    srsi_turning_down = (sk < sk.shift(1)) & (sk > 60)

    signals = pd.Series(0, index=df.index)
    signals[safe & far_below & hist_shrink_bull & srsi_turning_up   & ema_flat] =  1
    signals[safe & far_above & hist_shrink_bear & srsi_turning_down & ema_flat] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= alma
    exit_short = df["low"] <= alma
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-H: RSI-ADX Precision Scalp (Double StochRSI Confirm)
# ───────────────────────────────────────────────────────────────────────────
# Rationale: Highest-precision setup. ADX must be in the "sweet spot" (15-25
# = mild drift, not trending). RSI extreme + BOTH K and D lines of StochRSI
# must confirm OS/OB. This dual confirmation dramatically reduces false signals
# on noisy 5m charts.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_H(df, p):
    atr           = _atr(df, p)
    adx, pdi, mdi = calc_adx(df, p["adx_period"])
    rsi           = _rsi(df, p)
    sk, sd        = _srsi(df, p)
    chop          = calc_choppiness_index(df, p["chop_period"])

    # ADX "sweet spot" - mild non-trending environment
    adx_sweet = (adx > 12) & (adx < 25)

    # ATR above 25th percentile (some volatility, not flat/dead)
    atr_floor = atr.rolling(50).quantile(0.25)
    vol_ok    = atr > atr_floor

    # CHOP not extreme chaos
    chop_ok   = chop < 61.8

    # RSI extended
    rsi_os = rsi < 30
    rsi_ob = rsi > 70

    # Double StochRSI confirm: BOTH K and D must agree on OS/OB
    srsi_both_os = (sk < 15) & (sd < 15)   # Double OS
    srsi_both_ob = (sk > 85) & (sd > 85)   # Double OB

    # StochRSI K turning (first sign of reversal)
    k_turning_up   = (sk > sk.shift(1)) & srsi_both_os.shift(1)
    k_turning_down = (sk < sk.shift(1)) & srsi_both_ob.shift(1)

    signals = pd.Series(0, index=df.index)
    signals[adx_sweet & vol_ok & chop_ok & rsi_os & k_turning_up]   =  1
    signals[adx_sweet & vol_ok & chop_ok & rsi_ob & k_turning_down] = -1
    # Dynamic Exit
    exit_long  = rsi >= 50
    exit_short = rsi <= 50
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-I: MACD Histogram Zero-Cross + BB Extreme Fade
# ───────────────────────────────────────────────────────────────────────────
# Rationale: When MACD histogram crosses zero (momentum direction change)
# while price is simultaneously at a BB extreme (over-extended), it captures
# the exact moment momentum shifts against the extension. RSI confirms the
# OB/OS state.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_I(df, p):
    safe             = _get_market_filters(df, p)
    rsi              = _rsi(df, p)
    macd_l, macd_s, macd_h = _macd(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)

    # MACD histogram crosses zero (momentum flip)
    hist_cross_up   = (macd_h > 0) & (macd_h.shift(1) <= 0)
    hist_cross_down = (macd_h < 0) & (macd_h.shift(1) >= 0)

    # Price at BB extreme (beyond 1.5 sigma = 75th percentile distance)
    # Use raw distance from midband in std units
    rolling_std = df["close"].rolling(p["bb_period"]).std()
    bb_mid      = df["close"].rolling(p["bb_period"]).mean()
    sigma_dist  = (df["close"] - bb_mid) / rolling_std.replace(0, np.nan)

    extended_low  = sigma_dist < -1.5
    extended_high = sigma_dist >  1.5

    # RSI on extreme side
    rsi_os = rsi < 35
    rsi_ob = rsi > 65

    signals = pd.Series(0, index=df.index)
    signals[safe & hist_cross_up   & extended_low  & rsi_os] =  1
    signals[safe & hist_cross_down & extended_high & rsi_ob] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= bb_m
    exit_short = df["low"] <= bb_m
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-J: Choppiness Regime Range Play + StochRSI Oscillator
# ───────────────────────────────────────────────────────────────────────────
# Rationale: When CHOP is in the 50-61.8 zone (clearly ranging but not pure
# chaos) AND ADX is below 20 (confirmed non-directional), the market IS in a
# proper mean-reverting range. StochRSI at extremes then becomes very reliable.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_J(df, p):
    atr           = _atr(df, p)
    adx, pdi, mdi = calc_adx(df, p["adx_period"])
    sk, sd        = _srsi(df, p)
    ema9          = _ema9(df, p)
    chop          = calc_choppiness_index(df, p["chop_period"])

    # Strong ranging regime: CHOP 50-61.8 + ADX < 20
    ranging_regime = (chop > 50) & (chop < 61.8) & (adx < 20)

    # ATR not too low (not completely flat)
    atr_floor = atr.rolling(50).quantile(0.20)
    vol_ok    = atr > atr_floor

    # EMA9 slope flat (confirms ranging)
    ema9_slope = (ema9 - ema9.shift(5)) / ema9.shift(5)
    ema_flat   = ema9_slope.abs() < 0.0015

    # StochRSI extremes with both K and D confirming
    srsi_os_trigger = (sk < 15) & (sk > sk.shift(1))   # OS + turning up
    srsi_ob_trigger = (sk > 85) & (sk < sk.shift(1))   # OB + turning down

    signals = pd.Series(0, index=df.index)
    signals[ranging_regime & vol_ok & ema_flat & srsi_os_trigger] =  1
    signals[ranging_regime & vol_ok & ema_flat & srsi_ob_trigger] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= ema9
    exit_short = df["low"] <= ema9
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-K: Triple Confluence (ALMA + BB Touch + StochRSI)
# ───────────────────────────────────────────────────────────────────────────
# Rationale: The most selective strategy - requires ALL three systems to agree:
# 1) ALMA above price (price below the smart mean) for longs
# 2) BB lower band touched (structural extreme)
# 3) StochRSI at the deepest extreme (<10) AND MACD histogram turning
# This is the "home-run" setup - rare but very high quality.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_K(df, p):
    safe             = _get_market_filters(df, p)
    alma             = _alma(df, p)
    atr              = _atr(df, p)
    sk, sd           = _srsi(df, p)
    macd_l, macd_s, macd_h = _macd(df, p)
    bb_u, bb_m, bb_l, bb_pct, bb_width = _bb(df, p)
    adx, pdi, mdi    = calc_adx(df, p["adx_period"])
    chop             = calc_choppiness_index(df, p["chop_period"])

    # Override: allow slightly wider ADX range (premium setup, less noise)
    adx_ok_k  = adx < 30

    # CHOP must be in ranging zone
    chop_ok_k = (chop > p["chop_mid_lo"]) & (chop < p["chop_max"])

    # ATR expanding from recent low (spring releasing from squeeze)
    atr_20_min = atr.rolling(20).min()
    atr_expanding = atr > atr_20_min * 1.15

    # Gate 1: ALMA above/below price (price extended from smart mean)
    alma_above_price = alma > df["close"] * 1.0005
    alma_below_price = alma < df["close"] * 0.9995

    # Gate 2: BB band touch
    at_bb_lower = bb_pct < 0.08
    at_bb_upper = bb_pct > 0.92

    # Gate 3: StochRSI at deepest extreme + turning
    deep_os = (sk < 10) & (sk > sk.shift(1))
    deep_ob = (sk > 90) & (sk < sk.shift(1))

    # Gate 4: MACD histogram turning toward zero
    hist_turning_bull = (macd_h > macd_h.shift(1)) & (macd_h < 0)
    hist_turning_bear = (macd_h < macd_h.shift(1)) & (macd_h > 0)

    signals = pd.Series(0, index=df.index)
    long_cond  = safe & adx_ok_k & chop_ok_k & atr_expanding & alma_above_price & at_bb_lower & deep_os & hist_turning_bull
    short_cond = safe & adx_ok_k & chop_ok_k & atr_expanding & alma_below_price & at_bb_upper & deep_ob & hist_turning_bear

    signals[long_cond]  =  1
    signals[short_cond] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= alma
    exit_short = df["low"] <= alma
    return signals, exit_long, exit_short


# ───────────────────────────────────────────────────────────────────────────
# MR-L: EMA9 Mean Reversion + MACD Divergence + ADX Gate
# ───────────────────────────────────────────────────────────────────────────
# Rationale: EMA9 is the primary dynamic mean. When price stretches > 2x ATR
# from EMA9 AND MACD diverges (price new extreme, MACD not matching) AND ADX
# is in a non-trending but not flat zone, the gravitational pull back to EMA9
# is powerful. Entry is on the crossback of price through EMA9.
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_L(df, p):
    safe             = _get_market_filters(df, p)
    ema9             = _ema9(df, p)
    atr              = _atr(df, p)
    rsi              = _rsi(df, p)
    macd_l, macd_s, macd_h = _macd(df, p)
    adx, pdi, mdi    = calc_adx(df, p["adx_period"])

    # ADX gate for MR-L: 14-28 (slightly active market, not dead)
    adx_gate = (adx > 14) & (adx < 28)

    # Price stretched > 2x ATR from EMA9
    dist = (df["close"] - ema9).abs()
    far_from_ema9 = dist > atr * 2.0
    below_ema9    = df["close"] < ema9
    above_ema9    = df["close"] > ema9

    # MACD divergence from price:
    # Long: price at new N-bar low, MACD_line is NOT at new low (higher)
    lookback = 15
    price_new_low    = df["close"] <= df["close"].rolling(lookback).min() * 1.002
    macd_not_new_low = macd_l > macd_l.rolling(lookback).min() + abs(macd_l.rolling(lookback).min()) * 0.05

    price_new_high    = df["close"] >= df["close"].rolling(lookback).max() * 0.998
    macd_not_new_high = macd_l < macd_l.rolling(lookback).max() - abs(macd_l.rolling(lookback).max()) * 0.05

    bull_divergence = price_new_low  & macd_not_new_low  & below_ema9
    bear_divergence = price_new_high & macd_not_new_high & above_ema9

    # Trigger: price crosses back through EMA9
    cross_above_ema9 = (df["close"] > ema9) & (df["close"].shift(1) <= ema9.shift(1))
    cross_below_ema9 = (df["close"] < ema9) & (df["close"].shift(1) >= ema9.shift(1))

    # RSI confirms
    rsi_os = rsi < 40
    rsi_ob = rsi > 60

    signals = pd.Series(0, index=df.index)
    signals[safe & adx_gate & far_from_ema9.shift(1) & bull_divergence.shift(1) & cross_above_ema9 & rsi_os] =  1
    signals[safe & adx_gate & far_from_ema9.shift(1) & bear_divergence.shift(1) & cross_below_ema9 & rsi_ob] = -1
    # Dynamic Exit
    exit_long  = df["high"] >= ema9
    exit_short = df["low"] <= ema9
    return signals, exit_long, exit_short


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "MR-A_ALMA_BB_RSI_Reset":        strategy_MR_A,
    "MR-B_StochRSI_MACD_Diverge":    strategy_MR_B,
    "MR-C_BB_Squeeze_ALMA_Pull":      strategy_MR_C,
    "MR-D_ADX_RSI_ATR_Spike_Fade":   strategy_MR_D,
    "MR-E_MACD_Fade_StochRSI_Exhaust":strategy_MR_E,
    "MR-F_BB_Pct_ATR_Blowoff_Div":   strategy_MR_F,
    "MR-G_ALMA_EMA_Deviation_Fade":   strategy_MR_G,
    "MR-H_RSI_ADX_Precision_Scalp":  strategy_MR_H,
    "MR-I_MACD_Zero_BB_Extreme":      strategy_MR_I,
    "MR-J_Choppiness_Range_StochRSI": strategy_MR_J,
    "MR-K_Triple_Confluence":         strategy_MR_K,
    "MR-L_EMA9_MACD_Diverge_ADX":    strategy_MR_L,
}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADER - reads ALL available 5min and 15min CSV files
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data():
    """
    Loads all *_5min_1year.csv and *_15min_1year.csv files found in the
    current directory. Returns:
        data_5m  : dict {symbol -> DataFrame}
        data_15m : dict {symbol -> DataFrame}
    """
    data_5m  = {}
    data_15m = {}

    required_cols = {"open", "high", "low", "close", "volume"}

    def _load_csv(filepath):
        df = pd.read_csv(filepath)
        # Normalise column names to lowercase
        df.columns = df.columns.str.lower().str.strip()

        # Parse datetime - support both named 'datetime' and generic timestamp cols
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
        elif "timestamp_ms" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
            df.set_index("datetime", inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

        df.sort_index(inplace=True)

        # Verify required OHLCV columns exist
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Ensure numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=list(required_cols), inplace=True)
        return df

    # ── 5m files ──
    for fpath in sorted(glob.glob("*_5min_1year.csv")):
        symbol = fpath.split("_")[0]
        try:
            df = _load_csv(fpath)
            if len(df) >= 500:
                data_5m[symbol] = df
                print(f"  [5m]  Loaded {fpath:35s} -> {len(df):>7,} rows")
            else:
                print(f"  [5m]  SKIPPED {fpath} - only {len(df)} rows")
        except Exception as exc:
            print(f"  [5m]  ERROR loading {fpath}: {exc}")

    # ── 15m files ──
    for fpath in sorted(glob.glob("*_15min_1year.csv")):
        symbol = fpath.split("_")[0]
        try:
            df = _load_csv(fpath)
            if len(df) >= 200:
                data_15m[symbol] = df
                print(f"  [15m] Loaded {fpath:35s} -> {len(df):>7,} rows")
            else:
                print(f"  [15m] SKIPPED {fpath} - only {len(df)} rows")
        except Exception as exc:
            print(f"  [15m] ERROR loading {fpath}: {exc}")

    return data_5m, data_15m


# ═══════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD RUNNER (5-split rolling OOS)
# ═══════════════════════════════════════════════════════════════════════════

def run_walkforward(engine, dfs, strategy_fn, params, n_splits=5):
    """
    Rolls through n_splits OOS windows (each = 1/n_splits of the data).
    Returns mean OOS PF and mean OOS WR across all windows and symbols.
    """
    all_pf  = []
    all_wr  = []
    all_trd = []

    for symbol, df in dfs.items():
        n     = len(df)
        chunk = n // n_splits

        for i in range(1, n_splits):
            oos_start = i * chunk
            oos_end   = (i + 1) * chunk
            oos_df    = df.iloc[oos_start:oos_end].copy()

            if len(oos_df) < 200:
                continue
            try:
                _, agg, _ = engine.run_multi_symbol(
                    {symbol: oos_df}, strategy_fn, params,
                    slippage_pct=SLIPPAGE_PCT, fee_pct=FEE_PCT
                )
                if agg["total_trades"] >= 5:
                    all_pf.append(agg["profit_factor"])
                    all_wr.append(agg["win_rate"])
                    all_trd.append(agg["total_trades"])
            except Exception:
                pass

    wf_pf  = round(np.mean(all_pf), 3)  if all_pf  else 0.0
    wf_wr  = round(np.mean(all_wr), 2)  if all_wr  else 0.0
    wf_trd = int(np.sum(all_trd))       if all_trd else 0
    return wf_pf, wf_wr, wf_trd


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN - TEST ALL STRATEGIES ON ALL TFs AND RANK
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 72)
    print("  COMPREHENSIVE MEAN REVERSION ENGINE -- 5m & 15m Timeframes")
    print("  12 Strategies x 2 TFs x All Available Symbols")
    print(f"  Fees: {FEE_PCT*100:.3f}% | Slippage: {SLIPPAGE_PCT*100:.4f}% | Risk: 1% per trade")
    print("=" * 72)
    print()

    print("Loading CSV data files ...")
    data_5m, data_15m = load_all_data()

    if not data_5m and not data_15m:
        print("\n[ERROR] No CSV files found. "
              "Ensure *_5min_1year.csv and *_15min_1year.csv are in the current directory.\n")
        return

    print(f"\n  5m  datasets : {sorted(data_5m.keys())}")
    print(f"  15m datasets : {sorted(data_15m.keys())}")
    print()

    engine  = BacktestCore()
    results = []

    tf_map = []
    if data_5m:
        tf_map.append(("5m",  data_5m,  TF_PARAMS["5min"]))
    if data_15m:
        tf_map.append(("15m", data_15m, TF_PARAMS["15min"]))

    total_combos = len(STRATEGIES) * len(tf_map)
    run_num      = 0

    for tf_label, dfs, p in tf_map:
        print("-" * 72)
        print(f"  TIMEFRAME: {tf_label}   "
              f"(ADX={p['adx_period']} | ATR={p['atr_period']} | "
              f"MACD={p['macd_fast']},{p['macd_slow']},{p['macd_signal']} | "
              f"RSI={p['rsi_period']} | SL={p['sl_atr']}xATR | TP={p['tp_atr']}xATR)")
        print("-" * 72)

        bt_params = p.copy()
        bt_params["trailing"] = False

        for strat_name, strategy_fn in STRATEGIES.items():
            run_num += 1
            label = f"[{run_num:02d}/{total_combos}] {tf_label} | {strat_name}"
            print(f"\n  {label}")

            try:
                # ── Standard Backtest ──
                _, agg, all_trades = engine.run_multi_symbol(
                    dfs, strategy_fn, bt_params,
                    slippage_pct=SLIPPAGE_PCT, fee_pct=FEE_PCT
                )

                trades  = agg["total_trades"]
                pf      = agg["profit_factor"]
                wr      = agg["win_rate"]
                sharpe  = agg["sharpe_ratio"]
                exp     = agg["expectancy"]
                tpd     = agg["trades_per_day"]
                symbols = agg["active_symbols"]

                if trades < 20:
                    print(f"    -> SKIPPED - only {trades} total trades across all symbols")
                    results.append({
                        "TF":          tf_label,
                        "Strategy":    strat_name,
                        "Status":      "SKIP_FEW_TRADES",
                        "Std_PF":      0.0,
                        "WF_OOS_PF":   0.0,
                        "Win_Rate_%":  0.0,
                        "WF_OOS_WR%":  0.0,
                        "Sharpe":      0.0,
                        "Expectancy":  0.0,
                        "Trades/Day":  0.0,
                        "Std_Trades":  trades,
                        "WF_Trades":   0,
                        "Symbols":     symbols,
                    })
                    continue

                print(f"    Std  -> PF: {pf:.3f} | WR: {wr:.1f}% | Sharpe: {sharpe:.3f} | "
                      f"Trades: {trades} | Exp: {exp:.4f} | TPD: {tpd:.2f}")

                # ── Walk-Forward ──
                wf_pf, wf_wr, wf_trd = run_walkforward(engine, dfs, strategy_fn, bt_params)
                print(f"    WF   -> OOS PF: {wf_pf:.3f} | OOS WR: {wf_wr:.1f}% | OOS Trades: {wf_trd}")

                # ── Status ──
                if pf >= 1.20 and wf_pf >= 1.10:
                    status = "[STRONG PASS]"
                elif pf >= 1.0 and wf_pf >= 1.0:
                    status = "[PASS]"
                elif pf >= 1.0 and wf_pf < 1.0:
                    status = "[IS-ONLY (overfit?)]"
                elif pf < 1.0:
                    status = "[FAIL]"
                else:
                    status = "[PARTIAL]"

                print(f"    Status: {status}")

                results.append({
                    "TF":          tf_label,
                    "Strategy":    strat_name,
                    "Status":      status,
                    "Std_PF":      round(pf, 3),
                    "WF_OOS_PF":   wf_pf,
                    "Win_Rate_%":  round(wr, 2),
                    "WF_OOS_WR%":  wf_wr,
                    "Sharpe":      round(sharpe, 3),
                    "Expectancy":  round(exp, 4),
                    "Trades/Day":  round(tpd, 2),
                    "Std_Trades":  trades,
                    "WF_Trades":   wf_trd,
                    "Symbols":     symbols,
                })

            except Exception as exc:
                import traceback
                print(f"    -> ERROR: {exc}")
                traceback.print_exc()
                results.append({
                    "TF":          tf_label,
                    "Strategy":    strat_name,
                    "Status":      f"ERROR: {str(exc)[:60]}",
                    "Std_PF":      0.0,
                    "WF_OOS_PF":   0.0,
                    "Win_Rate_%":  0.0,
                    "WF_OOS_WR%":  0.0,
                    "Sharpe":      0.0,
                    "Expectancy":  0.0,
                    "Trades/Day":  0.0,
                    "Std_Trades":  0,
                    "WF_Trades":   0,
                    "Symbols":     0,
                })

    # ═══════════════════════════════════════════════════════════════════════
    #  FINAL RANKINGS
    # ═══════════════════════════════════════════════════════════════════════

    df_res = pd.DataFrame(results)

    # Sort: PF desc -> WR desc -> WF PF desc (ignore skipped / errored entries)
    df_ranked = df_res[df_res["Std_PF"] > 0].copy()
    df_ranked.sort_values(
        by=["Std_PF", "Win_Rate_%", "WF_OOS_PF"],
        ascending=False,
        inplace=True
    )
    df_ranked.reset_index(drop=True, inplace=True)
    df_ranked.index = df_ranked.index + 1   # 1-based rank

    # Append skipped/failed at the bottom
    df_skip = df_res[df_res["Std_PF"] == 0].copy()
    df_full = pd.concat([df_ranked, df_skip], ignore_index=False)

    print()
    print("=" * 72)
    print("  FINAL RANKINGS -- sorted by Std PF (desc) -> Win Rate (desc)")
    print("=" * 72)

    # Pretty-print the leaderboard
    display_cols = [
        "TF", "Strategy", "Std_PF", "WF_OOS_PF",
        "Win_Rate_%", "WF_OOS_WR%", "Sharpe",
        "Std_Trades", "WF_Trades", "Status"
    ]
    if not df_ranked.empty:
        print(df_ranked[display_cols].to_string())
    else:
        print("  [!] No strategies produced enough trades. Check data files.")

    # ── Save to CSV ──
    out_file = "mr_comprehensive_rankings.csv"
    df_full[display_cols].to_csv(out_file)
    print(f"\n  Rankings saved to -> {out_file}")

    # ── Summary Banner ──
    passing = df_ranked[df_ranked["Std_PF"] >= 1.0]
    strong  = df_ranked[df_ranked["Std_PF"] >= 1.2]
    print()
    print("=" * 72)
    print("  SUMMARY")
    print(f"  Total strategies tested : {len(df_res)}")
    print(f"  Producing trades        : {len(df_ranked)}")
    print(f"  Profitable  (PF >= 1.0) : {len(passing)}")
    print(f"  Strong Pass (PF >= 1.2) : {len(strong)}")
    if not df_ranked.empty:
        best = df_ranked.iloc[0]
        print(f"\n  >> BEST STRATEGY: {best['Strategy']} [{best['TF']}]")
        print(f"     PF={best['Std_PF']:.3f} | WR={best['Win_Rate_%']:.1f}% "
              f"| OOS PF={best['WF_OOS_PF']:.3f} | Sharpe={best['Sharpe']:.3f}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
