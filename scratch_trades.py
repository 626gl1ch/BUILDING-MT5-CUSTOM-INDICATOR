from ml_demigod_5m_optimizer import MLDemigod5mOptimizer, generate_multi_strategy

def run():
    opt = MLDemigod5mOptimizer()
    p = {'strategy_type': 'CAPITULATION', 'b_lower': 0.022856045465724326, 'b_upper': 0.9027539828162178, 'b_trigger_l': 0.17766006089780167, 'b_trigger_s': 0.8770461854898064, 'vol_mult': 2.81629468260803, 'atr_mult': 2.678978532837026, 'sl_atr': 3.2, 'tp_atr': 2.6, 'max_bars_hold': 97}
    res = opt.core.run_full_validation(
        opt.all_dfs, 
        generate_multi_strategy, 
        p,
        min_trades_per_day=0.1,
        min_assets=1,
        n_permutations=20
    )
    print("TOTAL TRADES:", res['backtest']['total_trades'])

if __name__ == "__main__":
    run()
