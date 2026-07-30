import pandas as pd
import numpy as np
import os
import glob
from backtest_core import BacktestCore
from mr_strategies_5m_15m import (
    TF_PARAMS, load_all_data, _atr, calc_adx, calc_choppiness_index, _rsi, _srsi, _alma, _bb, _ema21
)

# Need a simple EMA 200
def _ema200(df):
    return df["close"].ewm(span=200, adjust=False).mean()

# ───────────────────────────────────────────────────────────────────────────
# MR-H_V3: RSI-ADX Precision Scalp + Macro Trend Alignment
# ───────────────────────────────────────────────────────────────────────────
# We only take mean reversion trades that pull back AGAINST the macro trend,
# so the reversion itself is WITH the trend.
def strategy_MR_H_V3(df, p):
    atr           = _atr(df, p)
    adx, pdi, mdi = calc_adx(df, p["adx_period"])
    rsi           = _rsi(df, p)
    sk, sd        = _srsi(df, p)
    chop          = calc_choppiness_index(df, p["chop_period"])
    ema200        = _ema200(df)

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

    # MACRO TREND ALIGNMENT
    # To take a LONG (mean revert up), the price must have pulled back BELOW a rising EMA200 or at least overall trend is up.
    # Actually, a simpler trend filter: only buy pullbacks if EMA200 is rising, OR just price > EMA200.
    # Often, a deep pullback goes below EMA200. So we can use EMA50 > EMA200 as the macro trend.
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    macro_bull = ema50 > ema200
    macro_bear = ema50 < ema200

    signals = pd.Series(0, index=df.index)
    
    signals[adx_sweet & vol_ok & chop_ok & rsi_os & k_turning_up & macro_bull]   =  1
    signals[adx_sweet & vol_ok & chop_ok & rsi_ob & k_turning_down & macro_bear] = -1

    return signals, None, None

# ───────────────────────────────────────────────────────────────────────────
# MR-A_V3: ALMA BB RSI Reset + Macro Trend Alignment
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_A_V3(df, p):
    atr    = _atr(df, p)
    alma   = _alma(df, p)
    rsi    = _rsi(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)
    
    # Trend filter
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    ema200 = df["close"].ewm(span=200, adjust=False).mean()
    macro_bull = ema50 > ema200
    macro_bear = ema50 < ema200

    at_lower = df["close"] <= bb_l
    at_upper = df["close"] >= bb_u

    rsi_os = rsi < 25
    rsi_ob = rsi > 75

    rsi_tick_up   = (rsi > rsi.shift(1)) & rsi_os.shift(1)
    rsi_tick_down = (rsi < rsi.shift(1)) & rsi_ob.shift(1)

    below_alma = df["close"] < alma
    above_alma = df["close"] > alma

    # Volatility gate (borrowed from MR-H)
    atr_floor = atr.rolling(50).quantile(0.25)
    vol_ok    = atr > atr_floor

    signals = pd.Series(0, index=df.index)
    long_cond  = vol_ok & at_lower.shift(1) & rsi_tick_up  & below_alma & macro_bull
    short_cond = vol_ok & at_upper.shift(1) & rsi_tick_down & above_alma & macro_bear

    signals[long_cond]  =  1
    signals[short_cond] = -1

    return signals, None, None


V3_STRATEGIES = {
    "MR-H_V3": strategy_MR_H_V3,
    "MR-A_V3": strategy_MR_A_V3
}

def run_v3_backtest():
    print("Loading datasets...")
    data_5m, data_15m = load_all_data()
    data_map = {"5m": data_5m, "15m": data_15m}
    engine = BacktestCore()

    # We will test standard TP/SL which worked well (SL 1.5, TP 2.5)
    # The pure fixed ATR mechanic outperforms the dynamic mechanic.
    SL = 1.5
    TP = 2.5

    print(f"\nEvaluating V3 Strategies (SL={SL}, TP={TP})...")
    
    for name, strat_fn in V3_STRATEGIES.items():
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
    run_v3_backtest()
