import itertools
import pandas as pd
import numpy as np
import time
from backtest_core import BacktestCore
from strategies import build_indicators

def get_signals_v2(df, trigger_type, bb_std, rsi_thresh):
    long_e = pd.Series(False, index=df.index)
    short_e = pd.Series(False, index=df.index)
    
    # Always use HTF Trend Filter for crypto Mean Reversion to avoid catching knives
    regime_long = df['close'] > df['ema50']
    regime_short = df['close'] < df['ema50']
    
    # Triggers
    if trigger_type == 'BB_RSI_Combo':
        # Combined trigger: Exhaustion beyond BB + Momentum extreme
        mid = df['bb_mid']
        std = (df['bb_upper'] - df['bb_mid']) / 2.0
        lower = mid - bb_std * std
        upper = mid + bb_std * std
        
        long_e = (df['close'] < lower) & (df['rsi3'] < rsi_thresh) & regime_long
        short_e = (df['close'] > upper) & (df['rsi3'] > (100 - rsi_thresh)) & regime_short
        
    elif trigger_type == 'ALMA_Distance':
        # Reversion to ALMA average when extremely stretched
        dist_thresh = bb_std # reusing variable name for stretch factor
        overext_up = (df['close'] > df['alma9']) & (df['alma_dist_atr'] > dist_thresh)
        overext_dn = (df['close'] < df['alma9']) & (df['alma_dist_atr'] > dist_thresh)
        reverting_dn = df['close'] < df['close'].shift(1)
        reverting_up = df['close'] > df['close'].shift(1)
        
        long_e = overext_dn.shift(1).fillna(False) & reverting_up & regime_long
        short_e = overext_up.shift(1).fillna(False) & reverting_dn & regime_short

    signals = pd.Series(0, index=df.index)
    signals.loc[long_e] = 1
    signals.loc[short_e] = -1
    return signals

def run_v2_grid():
    bc = BacktestCore(data_dir=".")
    dfs = bc.load_all_data(suffix="5min_1year")
    
    for symbol in dfs:
        print(f"Building indicators for {symbol}...")
        dfs[symbol] = build_indicators(dfs[symbol])

    triggers = ['BB_RSI_Combo', 'ALMA_Distance']
    
    bb_stds = [2.0, 2.5] # Also used as alma_dist_atr for ALMA strategy
    rsi_thresholds = [15, 20, 25]
    
    # Asymmetrical profiles (Wide SL, tight TP for high Win Rate)
    sl_atrs = [1.5, 2.0, 3.0, 4.0]
    tp_atrs = [0.5, 0.8, 1.0, 1.5]
    
    results = []
    
    print(f"Starting V2 MR Miner Grid Search (Asymmetrical Scalps)...")
    start_t = time.time()
    
    combos = list(itertools.product(triggers, bb_stds, rsi_thresholds, sl_atrs, tp_atrs))
    total_combos = 0
    
    for trigger, bb_std, rsi_thresh, sl, tp in combos:
        total_combos += 1
        
        def strategy_fn(df, params):
            return get_signals_v2(df, trigger, bb_std, rsi_thresh)
            
        bt_params = {
            'sl_atr': sl,
            'tp_atr': tp,
            'max_bars_hold': 15, # Tighten time stop, MR should play out fast
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
        
        if total_trades >= 50:
            results.append({
                'Trigger': trigger,
                'Param1_BB/Dist': bb_std,
                'Param2_RSI': rsi_thresh,
                'SL_ATR': sl,
                'TP_ATR': tp,
                'Trades': total_trades,
                'WinRate': wr,
                'ProfitFactor': pf,
                'Expectancy': expectancy
            })
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res.sort_values(by=['ProfitFactor', 'WinRate'], ascending=[False, False], inplace=True)
        df_res.to_csv("mr_miner_v2_results.csv", index=False)
        
        print(f"\nPhase 2 Miner completed {total_combos} combinations in {time.time()-start_t:.1f}s")
        print("\nTop 10 High-Probability Asymmetric Setups:")
        print(df_res.head(10).to_string(index=False))
    else:
        print("No strategy generated > 50 trades.")

if __name__ == "__main__":
    run_v2_grid()
