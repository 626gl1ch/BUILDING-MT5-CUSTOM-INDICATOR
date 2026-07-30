import itertools
import pandas as pd
import numpy as np
import time
from backtest_core import BacktestCore
from strategies import build_indicators

def get_signals(df, regime_type, trigger_type, rsi_thresh, bb_std):
    """Generates signals based on regime and trigger."""
    long_e = pd.Series(False, index=df.index)
    short_e = pd.Series(False, index=df.index)

    # Contextual Filters (Regimes)
    if regime_type == 'HTF_Trend':
        regime_long = df['close'] > df['ema50']
        regime_short = df['close'] < df['ema50']
    elif regime_type == 'Chop_Squeeze':
        regime_long = df['chop14'] > 60
        regime_short = df['chop14'] > 60
    else: # No Filter
        regime_long = pd.Series(True, index=df.index)
        regime_short = pd.Series(True, index=df.index)

    # Entry Triggers
    if trigger_type == 'RSI_Extreme':
        long_e = (df['rsi3'] < rsi_thresh) & regime_long
        short_e = (df['rsi3'] > (100 - rsi_thresh)) & regime_short
    elif trigger_type == 'BB_Exhaustion':
        mid = df['bb_mid']
        std = (df['bb_upper'] - df['bb_mid']) / 2.0  # reconstruct std
        lower = mid - bb_std * std
        upper = mid + bb_std * std
        long_e = (df['close'] < lower) & regime_long
        short_e = (df['close'] > upper) & regime_short
    elif trigger_type == 'StochRSI_Cross':
        k, kd = df['stoch_k'], df['stoch_d']
        cross_up = (k > kd) & (k.shift(1) <= kd.shift(1)) & (k.shift(1) < 20)
        cross_dn = (k < kd) & (k.shift(1) >= kd.shift(1)) & (k.shift(1) > 80)
        long_e = cross_up & regime_long
        short_e = cross_dn & regime_short
    
    signals = pd.Series(0, index=df.index)
    signals.loc[long_e] = 1
    signals.loc[short_e] = -1
    return signals

def run_grid_search():
    bc = BacktestCore(data_dir=".")
    dfs = bc.load_all_data(suffix="5min_1year")
    
    for symbol in dfs:
        print(f"Building indicators for {symbol}...")
        dfs[symbol] = build_indicators(dfs[symbol])

    regimes = ['No_Filter', 'HTF_Trend', 'Chop_Squeeze']
    triggers = ['RSI_Extreme', 'BB_Exhaustion', 'StochRSI_Cross']
    
    rsi_thresholds = [10, 15, 20]
    bb_stds = [2.0, 2.5, 3.0]
    
    sl_atrs = [0.5, 1.0, 1.5, 2.0]
    tp_atrs = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    results = []
    
    print(f"Starting MR Miner Grid Search...")
    start_t = time.time()
    
    combos = list(itertools.product(regimes, triggers, sl_atrs, tp_atrs))
    total_combos = 0
    
    for regime, trigger, sl, tp in combos:
        if trigger == 'RSI_Extreme':
            params_list = [{'rsi_thresh': r, 'bb_std': 2.0} for r in rsi_thresholds]
        elif trigger == 'BB_Exhaustion':
            params_list = [{'rsi_thresh': 20, 'bb_std': b} for b in bb_stds]
        else:
            params_list = [{'rsi_thresh': 20, 'bb_std': 2.0}]
            
        for tp_param in params_list:
            total_combos += 1
            
            def strategy_fn(df, params):
                return get_signals(df, regime, trigger, tp_param['rsi_thresh'], tp_param['bb_std'])
                
            bt_params = {
                'sl_atr': sl,
                'tp_atr': tp,
                'max_bars_hold': 20,
                'risk_pct': 0.01,
                'trailing': False
            }
            
            sym_results, agg, all_trades = bc.run_multi_symbol(
                dfs, strategy_fn, bt_params, slippage_pct=0.00015, fee_pct=0.00015
            )
            
            pf = agg['profit_factor']
            wr = agg['win_rate']
            total_trades = agg['total_trades']
            expectancy = agg['expectancy']
            
            if total_trades > 0:
                results.append({
                    'Regime': regime,
                    'Trigger': trigger,
                    'Param': tp_param['rsi_thresh'] if trigger == 'RSI_Extreme' else (tp_param['bb_std'] if trigger == 'BB_Exhaustion' else 'N/A'),
                    'SL_ATR': sl,
                    'TP_ATR': tp,
                    'Trades': total_trades,
                    'WinRate': wr,
                    'ProfitFactor': pf,
                    'Expectancy': expectancy
                })
                
    df_res = pd.DataFrame(results)
    df_res.sort_values(by=['ProfitFactor', 'WinRate'], ascending=[False, False], inplace=True)
    df_res.to_csv("mr_miner_results.csv", index=False)
    
    print(f"\nMiner completed {total_combos} combinations in {time.time()-start_t:.1f}s")
    print("\nTop 10 High-Probability Setups:")
    print(df_res.head(10).to_string(index=False))

if __name__ == "__main__":
    run_grid_search()
