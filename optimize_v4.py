import pandas as pd
import numpy as np
import os
import glob
from backtest_core import BacktestCore
from mr_strategies_5m_15m import (
    TF_PARAMS, load_all_data, _atr, calc_adx, calc_choppiness_index, _rsi, _srsi, _alma, _bb, _ema9
)

# ───────────────────────────────────────────────────────────────────────────
# MR-H_V4: RSI-ADX Precision Scalp (Original, no dynamic exits)
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_H_V4(df, p):
    atr           = _atr(df, p)
    adx, pdi, mdi = calc_adx(df, p["adx_period"])
    rsi           = _rsi(df, p)
    sk, sd        = _srsi(df, p)
    chop          = calc_choppiness_index(df, p["chop_period"])

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

    signals = pd.Series(0, index=df.index)
    signals[adx_sweet & vol_ok & chop_ok & rsi_os & k_turning_up]   =  1
    signals[adx_sweet & vol_ok & chop_ok & rsi_ob & k_turning_down] = -1

    return signals, None, None

def run_v4_optimization():
    print("Loading datasets...")
    data_5m, data_15m = load_all_data()
    engine = BacktestCore()

    sl_grid = [1.2, 1.5, 2.0, 2.5]
    tp_grid = [2.0, 3.0, 4.0, 5.0]

    print("\n>> Optimizing Original MR-H (V4) on 15m")
    dfs = data_15m
    
    results_list = []
    for sl in sl_grid:
        for tp in tp_grid:
            params = TF_PARAMS["15min"].copy()
            params["sl_atr"] = sl
            params["tp_atr"] = tp
            params["trailing"] = True

            _, agg, _ = engine.run_multi_symbol(
                dfs, strategy_MR_H_V4, params,
                slippage_pct=0.0001, fee_pct=0.0002
            )
            
            pf = agg.get("profit_factor", 0)
            wr = agg.get("win_rate", 0)
            trades = agg.get("total_trades", 0)
            sharpe = agg.get("sharpe_ratio", 0)
            
            if trades > 0:
                results_list.append({
                    "SL": sl, "TP": tp, "PF": pf, "WR": wr, "Sharpe": sharpe, "Trades": trades
                })

    if results_list:
        df_res = pd.DataFrame(results_list).sort_values("PF", ascending=False)
        print(df_res.head(10).to_string(index=False))

if __name__ == "__main__":
    run_v4_optimization()
