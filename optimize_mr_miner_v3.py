import itertools
import pandas as pd
import numpy as np
import time
from backtest_core import BacktestCore
from strategies_v3 import build_v3_indicators, get_v3_signals

def run_v3_miner():
    bc = BacktestCore(data_dir=".")
    
    # User requested to use the 5m and 15m TF available csv files
    files_to_load = ["5min_1year", "15min_1year"]
    
    results = []
    
    print("Starting Phase 3 Miner: Advanced Volume Profile Grid Search...")
    
    for suffix in files_to_load:
        print(f"\n--- Loading {suffix} data ---")
        dfs = bc.load_all_data(suffix=suffix)
        
        if not dfs:
            continue
            
        print("Building advanced V3 indicators (including Volume Profile) - this may take a moment...")
        for symbol in dfs:
            start_b = time.time()
            dfs[symbol] = build_v3_indicators(dfs[symbol])
            print(f"  {symbol} built in {time.time()-start_b:.2f}s")
            
        strategies = ['ALMA_POC_Bounce', 'BB_Stoch_MACD_Exhaustion', 'EMA9_HVN_Trap', 'Fast_RSI_Chop_Scalp']
        
        # SL/TP parameter grid
        sl_atrs = [1.5, 2.5, 4.0]
        tp_atrs = [0.5, 1.0, 2.0]
        
        combos = list(itertools.product(strategies, sl_atrs, tp_atrs))
        
        print(f"Scanning {len(combos)} combinations for {suffix}...")
        start_scan = time.time()
        
        for strategy_name, sl, tp in combos:
            def strategy_fn(df, params):
                return get_v3_signals(df, strategy_name, **params)
                
            bt_params = {
                'sl_atr': sl,
                'tp_atr': tp,
                'max_bars_hold': 24, # Increased time stop slightly due to 15m testing
                'risk_pct': 0.01,
                'trailing': False
            }
            
            # Using standard multi_symbol run with standard fee assumptions
            sym_results, agg, all_trades = bc.run_multi_symbol(
                dfs, strategy_fn, bt_params, slippage_pct=0.00015, fee_pct=0.00015
            )
            
            total_trades = agg.get('total_trades', 0)
            
            if total_trades > 50:
                results.append({
                    'Timeframe': suffix,
                    'Strategy': strategy_name,
                    'SL_ATR': sl,
                    'TP_ATR': tp,
                    'Trades': total_trades,
                    'WinRate': agg['win_rate'],
                    'ProfitFactor': agg['profit_factor'],
                    'Expectancy': agg['expectancy'],
                    'Sharpe': agg['sharpe_ratio']
                })
                
        print(f"Finished {suffix} scan in {time.time()-start_scan:.1f}s")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # Rank from best to worst by PF and WR
        df_res.sort_values(by=['ProfitFactor', 'WinRate'], ascending=[False, False], inplace=True)
        df_res.to_csv("mr_v3_rankings.csv", index=False)
        print("\n--- TOP 15 HIGH-FREQUENCY MEAN REVERSION STRATEGIES ---")
        print(df_res.head(15).to_string(index=False))
        print(f"\nSaved {len(df_res)} profitable/valid setups to mr_v3_rankings.csv")
    else:
        print("No strategy generated > 50 trades.")

if __name__ == "__main__":
    run_v3_miner()
