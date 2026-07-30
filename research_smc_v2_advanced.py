import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from smc_enhanced_library import generate_smc_vol_profile, generate_smc_v2_vwap, generate_smc_v2_ob

def main():
    print("=" * 80)
    print(" SMC ADVANCED GENERATION (V2) - 1H TIMEFRAME")
    print("=" * 80)
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    print("\n[1/3] Loading 1-Hour Historical Datasets...")
    data = engine.load_all_data(suffix="1H_1year")
    
    print("\n[2/3] Precomputing Indicators (SVP, VWAP, Order Blocks)...")
    precomputed = {}
    for sym, df in data.items():
        precomputed[sym] = add_all_indicators(df)
        
    print("\n[3/3] Evaluating Advanced Variants...")
    
    params = {'lookback': 20, 'sl_atr': 2.5, 'tp_atr': 4.0, 'trailing': False}
    
    print("\n--- BASELINE: SMC V1 (SVP Filter Only) ---")
    _, agg_base, _ = engine.run_multi_symbol(precomputed, generate_smc_vol_profile, params, slippage_pct=0.0002, fee_pct=0.00055)
    print(f"PF: {agg_base['profit_factor']:.2f} | WR: {agg_base['win_rate']:.1f}% | Sharpe: {agg_base['sharpe_ratio']:.3f} | Trades/Day: {agg_base['trades_per_day']:.2f}")

    print("\n--- VARIANT A: SMC + Institutional VWAP Deviations ---")
    _, agg_vwap, _ = engine.run_multi_symbol(precomputed, generate_smc_v2_vwap, params, slippage_pct=0.0002, fee_pct=0.00055)
    print(f"PF: {agg_vwap['profit_factor']:.2f} | WR: {agg_vwap['win_rate']:.1f}% | Sharpe: {agg_vwap['sharpe_ratio']:.3f} | Trades/Day: {agg_vwap['trades_per_day']:.2f}")

    print("\n--- VARIANT C: SMC + Multi-Timeframe Order Blocks ---")
    _, agg_ob, _ = engine.run_multi_symbol(precomputed, generate_smc_v2_ob, params, slippage_pct=0.0002, fee_pct=0.00055)
    print(f"PF: {agg_ob['profit_factor']:.2f} | WR: {agg_ob['win_rate']:.1f}% | Sharpe: {agg_ob['sharpe_ratio']:.3f} | Trades/Day: {agg_ob['trades_per_day']:.2f}")

if __name__ == '__main__':
    main()
