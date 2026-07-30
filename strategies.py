import numpy as np
import pandas as pd
import indicators as ind


# ---------------------------------------------------------------------------
# Shared indicator build — computed once per (symbol, timeframe) and reused
# by every strategy so we don't recompute ADX/ATR/CHOP 8 times.
# ---------------------------------------------------------------------------
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["alma9"] = ind.alma(d["close"], period=9, offset=0.85, sigma=6.0)
    d["ema9"] = ind.ema(d["close"], 9)
    d["ema50"] = ind.ema(d["close"], 50)

    d["atr14"] = ind.atr(d["high"], d["low"], d["close"], 14)
    d["atr_pctile"] = ind.rolling_percentile_rank(d["atr14"], lookback=200)

    d["adx11"] = ind.adx(d["high"], d["low"], d["close"], 11)

    d["chop14"] = ind.choppiness_index(d["high"], d["low"], d["close"], 14)

    d["rsi7"] = ind.rsi(d["close"], 7)
    d["rsi3"] = ind.rsi(d["close"], 3)
    d["rsi14"] = ind.rsi(d["close"], 14)

    k, dd = ind.stoch_rsi(d["close"], rsi_period=8, stoch_period=8, k_smooth=3, d_smooth=3)
    d["stoch_k"] = k
    d["stoch_d"] = dd

    macd_line, macd_sig, macd_hist = ind.macd(d["close"], fast=6, slow=13, signal=5)
    d["macd"] = macd_line
    d["macd_sig"] = macd_sig
    d["macd_hist"] = macd_hist
    d["macd_hist_z"] = (macd_hist - macd_hist.rolling(100, min_periods=30).mean()) / \
                        macd_hist.rolling(100, min_periods=30).std(ddof=0).replace(0, np.nan)

    upper, mid, lower, pctb = ind.bollinger(d["close"], period=20, k=2.0)
    d["bb_upper"] = upper
    d["bb_mid"] = mid
    d["bb_lower"] = lower
    d["bb_pctb"] = pctb

    dist = (d["close"] - d["alma9"]).abs()
    d["alma_dist_atr"] = dist / d["atr14"].replace(0, np.nan)

    return d


# ---------------------------------------------------------------------------
# Regime filter — must be TRUE for any strategy to be allowed to enter.
# Rejects: strong trend (ADX high), dead/no-volatility market (low ATR
# percentile), volatility blow-outs (extreme ATR percentile / news spikes).
# ---------------------------------------------------------------------------
def tradeable_regime(d: pd.DataFrame,
                      adx_max: float = 20.0,
                      chop_min: float = 55.0,
                      atr_pctile_min: float = 25.0,
                      atr_pctile_max: float = 88.0) -> pd.Series:
    cond = (
        (d["adx11"] < adx_max) &
        (d["chop14"] > chop_min) &
        (d["atr_pctile"] >= atr_pctile_min) &
        (d["atr_pctile"] <= atr_pctile_max)
    )
    return cond.fillna(False)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
def strat_bb_rsi(d):
    """1. Bollinger Band + RSI(7) extreme reversion."""
    long_e = (d["close"] < d["bb_lower"]) & (d["rsi7"] < 25)
    short_e = (d["close"] > d["bb_upper"]) & (d["rsi7"] > 75)
    return long_e, short_e


def strat_stochrsi_alma_pullback(d):
    """2. StochRSI cross from oversold/overbought while price on the
    'wrong' side of ALMA9 (pullback-in-range reversion)."""
    k, kd = d["stoch_k"], d["stoch_d"]
    cross_up = (k > kd) & (k.shift(1) <= kd.shift(1)) & (k.shift(1) < 20)
    cross_dn = (k < kd) & (k.shift(1) >= kd.shift(1)) & (k.shift(1) > 80)
    long_e = cross_up & (d["close"] < d["alma9"])
    short_e = cross_dn & (d["close"] > d["alma9"])
    return long_e, short_e


def strat_macd_hist_fade(d):
    """3. MACD(6,13,5) histogram extreme (z-score) then rolling back
    toward zero -> fade the exhausted move."""
    z = d["macd_hist_z"]
    turning_up = (z.shift(1) < -1.8) & (d["macd_hist"] > d["macd_hist"].shift(1))
    turning_dn = (z.shift(1) > 1.8) & (d["macd_hist"] < d["macd_hist"].shift(1))
    long_e = turning_up
    short_e = turning_dn
    return long_e, short_e


def strat_rsi2_dip(d):
    """4. Connors-style RSI(3) extreme dip/rip inside a higher-TF trend
    filter (EMA50), i.e. buy dips in an up-drift range, sell rips in a
    down-drift range — but only inside the ranging regime filter."""
    long_e = (d["rsi3"] < 10) & (d["close"] > d["ema50"])
    short_e = (d["rsi3"] > 90) & (d["close"] < d["ema50"])
    return long_e, short_e


def strat_bb_pctb_chop(d):
    """5. Bollinger %B extreme + strong Choppiness Index confluence."""
    long_e = (d["bb_pctb"] < 0.03) & (d["chop14"] > 60)
    short_e = (d["bb_pctb"] > 0.97) & (d["chop14"] > 60)
    return long_e, short_e


def strat_double_oversold(d):
    """6. RSI(7) and StochRSI both extreme simultaneously (double
    confirmation, fewer but higher-quality signals)."""
    long_e = (d["rsi7"] < 30) & (d["stoch_k"] < 20)
    short_e = (d["rsi7"] > 70) & (d["stoch_k"] > 80)
    return long_e, short_e


def strat_alma_distance_reversion(d):
    """7. Price overextended from ALMA9 by > 2.2x ATR, first bar closing
    back toward the average = reversion trigger."""
    overext_up = (d["close"] > d["alma9"]) & (d["alma_dist_atr"] > 2.2)
    overext_dn = (d["close"] < d["alma9"]) & (d["alma_dist_atr"] > 2.2)
    reverting_dn = d["close"] < d["close"].shift(1)   # topped, rolling over
    reverting_up = d["close"] > d["close"].shift(1)   # bottomed, turning up
    short_e = overext_up.shift(1).fillna(False) & reverting_dn
    long_e = overext_dn.shift(1).fillna(False) & reverting_up
    return long_e, short_e


def strat_rsi_macd_divergence(d, lookback=8):
    """8. Simple bullish/bearish divergence: price makes a new local
    extreme but RSI(14) doesn't confirm it, combined with a MACD
    histogram uptick/downtick as trigger."""
    price_low = d["low"].rolling(lookback).min()
    price_high = d["high"].rolling(lookback).max()
    rsi_low = d["rsi14"].rolling(lookback).min()
    rsi_high = d["rsi14"].rolling(lookback).max()

    new_price_low = d["low"] <= price_low
    new_price_high = d["high"] >= price_high
    rsi_not_new_low = d["rsi14"] > rsi_low.shift(1)
    rsi_not_new_high = d["rsi14"] < rsi_high.shift(1)

    bull_div = new_price_low & rsi_not_new_low & (d["macd_hist"] > d["macd_hist"].shift(1))
    bear_div = new_price_high & rsi_not_new_high & (d["macd_hist"] < d["macd_hist"].shift(1))
    return bull_div.fillna(False), bear_div.fillna(False)


STRATEGIES = {
    "1_BB_RSI_Reversion": strat_bb_rsi,
    "2_StochRSI_ALMA_Pullback": strat_stochrsi_alma_pullback,
    "3_MACD_Hist_Fade": strat_macd_hist_fade,
    "4_RSI2_Dip_Rip": strat_rsi2_dip,
    "5_BB_PctB_Chop_Confluence": strat_bb_pctb_chop,
    "6_Double_Oversold_Confirm": strat_double_oversold,
    "7_ALMA_Distance_Reversion": strat_alma_distance_reversion,
    "8_RSI_MACD_Divergence": strat_rsi_macd_divergence,
}
