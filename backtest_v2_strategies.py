import pandas as pd
import numpy as np
import os
import glob
from backtest_core import BacktestCore
from mr_strategies_5m_15m import (
    TF_PARAMS, load_all_data, _atr, calc_adx, calc_choppiness_index, _rsi, _srsi, _alma, _bb
)

# ───────────────────────────────────────────────────────────────────────────
# MR-H_V2: RSI-ADX Precision Scalp + Volume Capitulation
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_H_V2(df, p):
    atr           = _atr(df, p)
    adx, pdi, mdi = calc_adx(df, p["adx_period"])
    rsi           = _rsi(df, p)
    sk, sd        = _srsi(df, p)
    chop          = calc_choppiness_index(df, p["chop_period"])

    # ADX "sweet spot" - mild non-trending environment
    adx_sweet = (adx > 12) & (adx < 25)
    atr_floor = atr.rolling(50).quantile(0.25)
    vol_ok    = atr > atr_floor
    chop_ok   = chop < 61.8

    rsi_os = rsi < 30
    rsi_ob = rsi > 70

    srsi_both_os = (sk < 15) & (sd < 15)
    srsi_both_ob = (sk > 85) & (sd > 85)

    k_turning_up   = (sk > sk.shift(1)) & srsi_both_os.shift(1)
    k_turning_down = (sk < sk.shift(1)) & srsi_both_ob.shift(1)

    # ADDED: Volume Capitulation Filter
    # Volume must be higher than the 20-period SMA of volume
    vol_sma = df['volume'].rolling(20).mean()
    vol_capitulation = df['volume'] > (vol_sma * 1.2)

    signals = pd.Series(0, index=df.index)
    # Require vol capitulation on the trigger bar OR the bar before it
    capitulation_zone = vol_capitulation | vol_capitulation.shift(1)

    signals[adx_sweet & vol_ok & chop_ok & rsi_os & k_turning_up & capitulation_zone]   =  1
    signals[adx_sweet & vol_ok & chop_ok & rsi_ob & k_turning_down & capitulation_zone] = -1

    return signals, None, None

# ───────────────────────────────────────────────────────────────────────────
# MR-A_V2: ALMA BB RSI Reset + ADX/CHOP Regime Filter
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_A_V2(df, p):
    atr    = _atr(df, p)
    alma   = _alma(df, p)
    rsi    = _rsi(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)
    
    # ADDED: Regime filters
    adx, pdi, mdi = calc_adx(df, p["adx_period"])
    chop          = calc_choppiness_index(df, p["chop_period"])
    adx_sweet = adx < 30 # Avoid strong trends
    chop_ok   = chop < 61.8 # Avoid extreme chop

    atr_floor = atr.rolling(50).quantile(0.25)
    vol_ok    = atr > atr_floor
    safe = adx_sweet & chop_ok & vol_ok

    at_lower = df["close"] <= bb_l
    at_upper = df["close"] >= bb_u

    rsi_os = rsi < 25
    rsi_ob = rsi > 75

    rsi_tick_up   = (rsi > rsi.shift(1)) & rsi_os.shift(1)
    rsi_tick_down = (rsi < rsi.shift(1)) & rsi_ob.shift(1)

    below_alma = df["close"] < alma
    above_alma = df["close"] > alma

    signals = pd.Series(0, index=df.index)
    long_cond  = safe & at_lower.shift(1) & rsi_tick_up  & below_alma
    short_cond = safe & at_upper.shift(1) & rsi_tick_down & above_alma

    signals[long_cond]  =  1
    signals[short_cond] = -1

    return signals, None, None


V2_STRATEGIES = {
    "MR-H_V2": strategy_MR_H_V2,
    "MR-A_V2": strategy_MR_A_V2
}

def run_v2_backtest():
    print("Loading datasets...")
    data_5m, data_15m = load_all_data()
    data_map = {"5m": data_5m, "15m": data_15m}
    engine = BacktestCore()

    # Base settings for evaluation
    SL = 1.2
    TP = 2.0

    print(f"\nEvaluating V2 Strategies (SL={SL}, TP={TP})...")
    
    for name, strat_fn in V2_STRATEGIES.items():
        for tf_label in ["5m", "15m"]:
            dfs = data_map.get(tf_label, {})
            if not dfs: continue
            
            params = TF_PARAMS[f"{tf_label}in"].copy()
            params["sl_atr"] = SL
            params["tp_atr"] = TP
            params["trailing"] = False

            results, agg, _ = engine.run_multi_symbol(
                dfs, strat_fn, params,
                slippage_pct=0.0001, fee_pct=0.0002
            )
            
            pf = agg.get("profit_factor", 0)
            wr = agg.get("win_rate", 0)
            trades = agg.get("total_trades", 0)
            sharpe = agg.get("sharpe_ratio", 0)
            
            print(f"[{tf_label}] {name:15s} -> PF: {pf:.2f} | WR: {wr:.2f}% | Sharpe: {sharpe:.2f} | Trades: {trades}")

if __name__ == "__main__":
    run_v2_backtest()
