import pandas as pd
import numpy as np
import os
from backtest_core import BacktestCore
from mr_strategies_5m_15m import (
    TF_PARAMS, load_all_data, _atr, _rsi, _bb, _get_market_filters
)

def _ema50(df): return df["close"].ewm(span=50, adjust=False).mean()
def _ema200(df): return df["close"].ewm(span=200, adjust=False).mean()

# ───────────────────────────────────────────────────────────────────────────
# MR-F_V2: Smart Money Capitulation Fade (Strict Volatility + Volume)
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_F_V2(df, p):
    safe   = _get_market_filters(df, p)
    atr    = _atr(df, p)
    rsi    = _rsi(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)

    at_lower_band = bb_pct < 0.05
    at_upper_band = bb_pct > 0.95

    # ATR blow-off: stricter (1.5x instead of 1.3x)
    atr_avg   = atr.rolling(20).mean()
    blowoff   = atr > atr_avg * 1.5

    # Volume Capitulation (New): Volume spike > 1.5x average
    vol_avg = df["volume"].rolling(20).mean()
    vol_capitulation = df["volume"] > vol_avg * 1.5

    lookback = 12
    price_new_low = df["close"] == df["close"].rolling(lookback).min()
    rsi_not_new_low = rsi > rsi.rolling(lookback).min() + 3

    price_new_high = df["close"] == df["close"].rolling(lookback).max()
    rsi_not_new_high = rsi < rsi.rolling(lookback).max() - 3

    bull_divergence = at_lower_band & price_new_low & rsi_not_new_low
    bear_divergence = at_upper_band & price_new_high & rsi_not_new_high

    bb_move_inside_long  = (bb_pct > 0.10) & (bb_pct.shift(1) <= 0.10)
    bb_move_inside_short = (bb_pct < 0.90) & (bb_pct.shift(1) >= 0.90)

    signals = pd.Series(0, index=df.index)
    signals[safe & bull_divergence.shift(1) & blowoff.shift(1) & vol_capitulation.shift(1) & bb_move_inside_long]  =  1
    signals[safe & bear_divergence.shift(1) & blowoff.shift(1) & vol_capitulation.shift(1) & bb_move_inside_short] = -1
    return signals, None, None

# ───────────────────────────────────────────────────────────────────────────
# MR-F_V3: Macro Pullback Reversion
# ───────────────────────────────────────────────────────────────────────────
def strategy_MR_F_V3(df, p):
    safe   = _get_market_filters(df, p)
    atr    = _atr(df, p)
    rsi    = _rsi(df, p)
    bb_u, bb_m, bb_l, bb_pct, _ = _bb(df, p)

    at_lower_band = bb_pct < 0.05
    at_upper_band = bb_pct > 0.95

    atr_avg   = atr.rolling(20).mean()
    blowoff   = atr > atr_avg * 1.3

    lookback = 12
    price_new_low = df["close"] == df["close"].rolling(lookback).min()
    rsi_not_new_low = rsi > rsi.rolling(lookback).min() + 3
    price_new_high = df["close"] == df["close"].rolling(lookback).max()
    rsi_not_new_high = rsi < rsi.rolling(lookback).max() - 3

    bull_divergence = at_lower_band & price_new_low & rsi_not_new_low
    bear_divergence = at_upper_band & price_new_high & rsi_not_new_high

    bb_move_inside_long  = (bb_pct > 0.10) & (bb_pct.shift(1) <= 0.10)
    bb_move_inside_short = (bb_pct < 0.90) & (bb_pct.shift(1) >= 0.90)

    # Macro trend filter (New)
    ema50 = _ema50(df)
    ema200 = _ema200(df)
    macro_bull = ema50 > ema200
    macro_bear = ema50 < ema200

    signals = pd.Series(0, index=df.index)
    signals[safe & bull_divergence.shift(1) & blowoff.shift(1) & bb_move_inside_long & macro_bull]  =  1
    signals[safe & bear_divergence.shift(1) & blowoff.shift(1) & bb_move_inside_short & macro_bear] = -1
    return signals, None, None


def run_v5_optimization():
    print("Loading datasets...")
    data_5m, data_15m = load_all_data()
    engine = BacktestCore()

    sl_grid = [1.2, 1.5, 2.0]
    tp_grid = [2.0, 3.0, 4.0]

    strategies = {
        "MR-F_V2_Capitulation": strategy_MR_F_V2,
        "MR-F_V3_MacroPullback": strategy_MR_F_V3
    }

    for name, strat_fn in strategies.items():
        print(f"\n>> Optimizing {name} on 15m")
        results_list = []
        for sl in sl_grid:
            for tp in tp_grid:
                params = TF_PARAMS["15min"].copy()
                params["sl_atr"] = sl
                params["tp_atr"] = tp
                params["trailing"] = True

                _, agg, _ = engine.run_multi_symbol(
                    data_15m, strat_fn, params,
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
            print(df_res.head(5).to_string(index=False))
        else:
            print("  No parameters produced trades.")

if __name__ == "__main__":
    run_v5_optimization()
