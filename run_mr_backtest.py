import pandas as pd
import numpy as np
from backtest_core import BacktestCore
from strategies import build_indicators, tradeable_regime, STRATEGIES

def create_strategy_wrapper(strat_func):
    def wrapper(df, params):
        long_e, short_e = strat_func(df)
        regime = tradeable_regime(df)
        
        signals = pd.Series(0, index=df.index)
        signals.loc[long_e & regime] = 1
        signals.loc[short_e & regime] = -1
        return signals
    return wrapper

def run_5_fold_wf(bc, dfs, wrapper, params, fee_pct, slippage_pct):
    folds = 5
    fold_pfs = []
    
    for i in range(folds):
        fold_dfs = {}
        for symbol, df in dfs.items():
            fold_size = len(df) // folds
            start = i * fold_size
            end = (i + 1) * fold_size if i < folds - 1 else len(df)
            fold_dfs[symbol] = df.iloc[start:end].copy()
            
        sym_res, agg, _ = bc.run_multi_symbol(fold_dfs, wrapper, params, slippage_pct=slippage_pct, fee_pct=fee_pct)
        fold_pfs.append(agg['profit_factor'])
        
    profitable_folds = sum(1 for pf in fold_pfs if pf > 1.0)
    min_pf = min(fold_pfs)
    return profitable_folds, min_pf

if __name__ == "__main__":
    bc = BacktestCore(data_dir=".")
    timeframes = ["5min_1year", "15min_1year"]
    results_list = []
    
    for tf in timeframes:
        print(f"\nLoading {tf} data...")
        dfs = bc.load_all_data(suffix=tf)
        
        for symbol, df in dfs.items():
            print(f"  Building indicators for {symbol}...")
            dfs[symbol] = build_indicators(df)
            
        for name, strat_func in STRATEGIES.items():
            print(f"Testing {name} on {tf}...")
            wrapper = create_strategy_wrapper(strat_func)
            
            params = {
                'sl_atr': 1.0,
                'tp_atr': 1.6,
                'max_bars_hold': 20,
                'risk_pct': 0.01,
                'trailing': False
            }
            
            sym_results, agg, all_trades = bc.run_multi_symbol(
                dfs, wrapper, params, slippage_pct=0.00015, fee_pct=0.00015
            )
            
            wf_prof, min_pf = run_5_fold_wf(bc, dfs, wrapper, params, fee_pct=0.00015, slippage_pct=0.00015)
            
            pf = agg['profit_factor']
            wr = agg['win_rate']
            sharpe = agg['sharpe_ratio']
            total_trades = agg['total_trades']
            
            results_list.append({
                'Timeframe': tf.split('_')[0],
                'Strategy': name,
                'PF': pf,
                'WR': wr,
                'Sharpe': sharpe,
                'Trades': total_trades,
                'wf_folds_profitable': f"{wf_prof}/5",
                'wf_min_fold_pf': min_pf
            })
            
    df_results = pd.DataFrame(results_list)
    df_results.sort_values(by=['Timeframe', 'PF', 'WR'], ascending=[True, False, False], inplace=True)
    
    print("\n--- Final Results ---")
    print(df_results.to_string(index=False))
    df_results.to_csv("mr_backtest_results.csv", index=False)
