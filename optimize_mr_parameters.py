import pandas as pd
import numpy as np
import os
import glob
from backtest_core import BacktestCore
from mr_strategies_5m_15m import TF_PARAMS, STRATEGIES, load_all_data

def run_optimization():
    print("=" * 60)
    print("  MEAN REVERSION PARAMETER OPTIMIZER")
    print("=" * 60)
    
    engine = BacktestCore(initial_capital=10000)
    
    # We will test MR-A and MR-H on 5m and 15m
    targets = [
        ("5m", "MR-A_ALMA_BB_RSI_Reset"),
        ("15m", "MR-H_RSI_ADX_Precision_Scalp")
    ]
    
    sl_grid = [0.8, 1.2, 1.5, 2.0]
    tp_grid = [1.5, 2.0, 3.0, 4.0]
    
    print("Loading datasets...")
    data_5m, data_15m = load_all_data()
    data_map = {"5m": data_5m, "15m": data_15m}
    
    for tf, strat_name in targets:
        print(f"\n>> Optimizing {strat_name} on {tf}")
        dfs = data_map.get(tf, {})
        if not dfs:
            print(f"No data for {tf}. Skipping.")
            continue
            
        file_tf = "5min" if tf == "5m" else "15min"
        p = TF_PARAMS[file_tf]
        strategy_fn = STRATEGIES[strat_name]
        
        results_list = []
        
        for sl in sl_grid:
            for tp in tp_grid:
                if sl >= tp:
                    continue # Not sensible for MR
                
                bt_params = p.copy()
                bt_params["sl_atr"] = sl
                bt_params["tp_atr"] = tp
                bt_params["trailing"] = False
                
                _, agg, _ = engine.run_multi_symbol(
                    dfs, strategy_fn, bt_params,
                    slippage_pct=0.0001, fee_pct=0.0002
                )
                
                pf = agg.get("profit_factor", 0)
                wr = agg.get("win_rate", 0)
                trades = agg.get("total_trades", 0)
                
                if trades > 0:
                    results_list.append({
                        "SL": sl, "TP": tp, "PF": pf, "WR": wr, "Trades": trades
                    })
                else:
                    print(f"    [DEBUG] SL: {sl}, TP: {tp} -> PF: {pf}, Trades: {trades}")
        
        if results_list:
            df_res = pd.DataFrame(results_list).sort_values("PF", ascending=False).head(5)
            print("  Top 5 Parameters:")
            print(df_res.to_string(index=False))
        else:
            print("  No parameters produced >20 trades.")

if __name__ == "__main__":
    run_optimization()
