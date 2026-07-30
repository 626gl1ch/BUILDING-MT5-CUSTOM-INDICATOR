import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from smc_enhanced_library import generate_smc_vol_profile, generate_smc_god_mode

def main():
    print("=" * 80)
    print(" SMC LIQUIDITY SWEEP - HIGHER TIMEFRAME RESEARCH (15m, 30m, 1H)")
    print("=" * 80)
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    timeframes = [
        ("15-Minute", "15min_1year"),
        ("30-Minute", "30min_1year"),
        ("1-Hour", "1H_1year")
    ]
    
    params = {'lookback': 20, 'sl_atr': 2.5, 'tp_atr': 4.0, 'trailing': False, 'fvg_lookback': 5, 'rvol_thresh': 1.2, 'ema_p': 600}
    
    for tf_name, suffix in timeframes:
        print(f"\n[{tf_name} TIMEFRAME]")
        print("-" * 50)
        
        data = engine.load_all_data(suffix=suffix)
        if not data:
            print(f"No data found for {suffix}")
            continue
            
        print("Precomputing Indicators...")
        precomputed = {}
        for sym, df in data.items():
            precomputed[sym] = add_all_indicators(df)
            
        print("Evaluating God Mode (No SVP)...")
        _, agg_god, _ = engine.run_multi_symbol(precomputed, generate_smc_god_mode, params, slippage_pct=0.0002, fee_pct=0.00055)
        print(f"PF: {agg_god['profit_factor']:.2f} | WR: {agg_god['win_rate']:.1f}% | Sharpe: {agg_god['sharpe_ratio']:.3f} | Trades/Day: {agg_god['trades_per_day']:.2f}")

        print("Evaluating SVP Mode (With Session Volume Profile POC Filter)...")
        _, agg_svp, _ = engine.run_multi_symbol(precomputed, generate_smc_vol_profile, params, slippage_pct=0.0002, fee_pct=0.00055)
        print(f"PF: {agg_svp['profit_factor']:.2f} | WR: {agg_svp['win_rate']:.1f}% | Sharpe: {agg_svp['sharpe_ratio']:.3f} | Trades/Day: {agg_svp['trades_per_day']:.2f}")

if __name__ == '__main__':
    main()
