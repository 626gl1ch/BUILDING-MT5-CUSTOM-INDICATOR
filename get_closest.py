import pandas as pd
import json
import warnings
import sys
from run_god_tier_gauntlet import STRATEGIES, load_data, ensure_columns, signal_capitulation, signal_macd_rev
from backtest_core import BacktestCore

warnings.filterwarnings('ignore')

def get_closest():
    # Only test the two that got closest
    target_names = ["ML_15MIN_TRIAL_53_CAPITULATION", "ML_30MIN_TRIAL_157_MACD_REV"]
    dfs = {
        '15min': load_data('15min_1year.csv'),
        '30min': load_data('30min_1year.csv'),
    }
    
    engine = BacktestCore(commission=0)
    
    results = []
    
    for strat in STRATEGIES:
        if strat['name'] not in target_names:
            continue
            
        tf_key = strat['tf'].split('_')[0]
        data_dict = dfs.get(tf_key, {})
        
        # Run base validation to get metrics
        res = engine.run_full_validation(
            data_dict,
            lambda df, p: strat['func'](df, p),
            strat['params'],
            min_trades_per_day=0.1,
            min_assets=1,
            n_permutations=200,
            slippage_pct=0.0002,
            fee_pct=0.0005
        )
        
        metrics = {
            "Strategy": strat['name'],
            "Base_PF": res['backtest'].get('profit_factor', 0),
            "Base_WR": res['backtest'].get('win_rate', 0),
            "Base_Trades": res['backtest'].get('total_trades', 0),
            "OOS_Sharpe": res.get('walk_forward', {}).get('oos_sharpe_mean', 0),
            "Perm_PValue": res.get('permutation', {}).get('p_value', 1.0)
        }
        
        # Now run at 1.5x cost to see what the metrics dropped to
        res_15x = engine.run_full_validation(
            data_dict,
            lambda df, p: strat['func'](df, p),
            strat['params'],
            min_trades_per_day=0.1,
            min_assets=1,
            n_permutations=20,
            slippage_pct=0.0002 * 1.5,
            fee_pct=0.0005 * 1.5
        )
        
        metrics["1.5x_Cost_PF"] = res_15x['backtest'].get('profit_factor', 0)
        metrics["1.5x_Cost_WR"] = res_15x['backtest'].get('win_rate', 0)
        
        results.append(metrics)
        
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    get_closest()
