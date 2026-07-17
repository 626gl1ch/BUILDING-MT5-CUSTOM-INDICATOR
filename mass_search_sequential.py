import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore
from rbo_v2 import GRIDS, STRATEGY_DESCRIPTIONS
from logger_config import logger

def get_param_combos(param_dict, max_combos=5000):
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    all_combos = list(itertools.product(*values))
    if len(all_combos) > max_combos:
        rng = np.random.default_rng(777)
        idx = rng.choice(len(all_combos), size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in idx]
    return [dict(zip(keys, c)) for c in all_combos]

def main():
    logger.info("======================================================")
    logger.info("  MASSIVE SCALE RBO SEARCH (Sequential Memory-Safe)")
    logger.info("======================================================")
    
    engine = BacktestCore()
    logger.info("[1/3] Loading multi-timeframe and synthetic data...")
    
    raw_data = {
        'SYNTHETIC': engine.load_all_data(suffix="1H_1year_SYNTHETIC"),
        '1H': engine.load_all_data(suffix="1H_1year"),
        '30min': engine.load_all_data(suffix="30min_1year"),
        '15min': engine.load_all_data(suffix="15min_1year"),
        '5min': engine.load_all_data(suffix="5min_1year")
    }
    
    logger.info("[2/3] Precomputing indicators (this takes ~10 min)...")
    t0 = time.time()
    precomputed = {}
    for tf, sym_dict in raw_data.items():
        precomputed[tf] = {}
        for sym, df in sym_dict.items():
            logger.info(f"  Computing {sym} ({tf})...")
            precomputed[tf][sym] = add_all_indicators(df)
    logger.info(f"  Done in {time.time() - t0:.0f}s")

    target_strategies = ['ema_pullback', 'donchian_breakout', 'bb_vwap_mr', 'rsi_divergence', 'squeeze_breakout']
    
    tasks = []
    for stype in target_strategies:
        combos = get_param_combos(GRIDS[stype]['params'], max_combos=2000) # 2000 combos per strategy = 10,000 total
        for c in combos:
            tasks.append((stype, c))
            
    total_combos = len(tasks)
    logger.info(f"[3/3] Total Combinations to Evaluate: {total_combos}")
    
    confirmed_gods = {stype: [] for stype in target_strategies}
    RESULTS_FILE = "god_strategies.json"
    
    t0 = time.time()
    n_tested = 0
    total_found = 0
    
    for stype, c in tasks:
        n_tested += 1
        fn = GRIDS[stype]['fn']
        
        is_god = False
        god_results = {}
        
        try:
            # Stage 1: Synthetic Data Test
            synth_res = engine.run_full_validation(
                precomputed['SYNTHETIC'], fn, c,
                min_trades_per_day=0.2, min_assets=2, n_permutations=200
            )
            
            if synth_res['passed']:
                is_god = True
                god_results['SYNTHETIC'] = synth_res
                
                # Stage 2: Multi-Timeframe Gauntlet
                for tf in ['1H', '30min', '15min', '5min']:
                    tf_res = engine.run_full_validation(
                        precomputed[tf], fn, c,
                        min_trades_per_day=0.2, min_assets=2, n_permutations=200
                    )
                    if not tf_res['passed']:
                        is_god = False
                        break
                    god_results[tf] = tf_res
                    
        except Exception as e:
            logger.error(f"Error evaluating {stype} with params {c}: {str(e)}", exc_info=True)
            is_god = False
            
        if n_tested % 50 == 0:
            elapsed = time.time() - t0
            rate = n_tested / elapsed if elapsed > 0 else 0
            eta = (total_combos - n_tested) / rate if rate > 0 else 0
            logger.info(f"[{n_tested}/{total_combos}] ETA: {eta:.0f}s | Rate: {rate:.1f} tests/sec | God Strats Found: {total_found}")
            
        if is_god:
            logger.info(f"  [GOD STRATEGY FOUND!!!] {stype} passed ALL timeframes and Synthetic data! {c}")
            confirmed_gods[stype].append({'params': c, 'results': god_results})
            total_found += 1
            
            with open(RESULTS_FILE, 'w') as f:
                json.dump(confirmed_gods, f, indent=2, default=str)
                
            # Keep detailed report updated
            with open("god_strategies_report.txt", "w") as f:
                f.write("=== THE GOD STRATEGIES REPORT ===\n")
                f.write("These strategies passed the gauntlet: Synthetic Data, 1H, 30m, 15m, and 5m timeframes across multiple assets.\n\n")
                for s_type, strats in confirmed_gods.items():
                    f.write(f"--- {s_type.upper()} ({len(strats)} found) ---\n")
                    for i, s in enumerate(strats):
                        f.write(f"\n[{s_type}] God Strategy #{i+1}\n")
                        f.write(f"Parameters: {s['params']}\n")
                        for tf in ['SYNTHETIC', '1H', '30min', '15min', '5min']:
                            agg = s['results'][tf]['backtest']
                            wf = s['results'][tf]['walkforward']
                            perm = s['results'][tf]['permutation']
                            f.write(f"  [{tf}] Expectancy: {agg['expectancy']:.4f} | OOS Exp: {wf['out_of_sample']['expectancy']:.4f} | Perm p-val: {perm['p_value']:.4f}\n")
                        f.write("-" * 50 + "\n")
            
            # --- REAL-TIME MD DOCUMENTATION ---
            desc = STRATEGY_DESCRIPTIONS.get(stype, {'logic': 'N/A', 'indicators': 'N/A'})
            md_filename = f"god_strategy_{stype}_{total_found}.md"
            with open(md_filename, "w", encoding='utf-8') as f:
                f.write(f"# 👑 GOD STRATEGY: {stype.upper()}\n\n")
                f.write("## Strategy Logic\n")
                f.write(f"{desc['logic']}\n\n")
                f.write("## Indicators Used\n")
                f.write(f"{desc['indicators']}\n\n")
                f.write("## Exact Settings\n")
                f.write("```json\n")
                f.write(json.dumps(c, indent=2) + "\n")
                f.write("```\n\n")
                f.write("## Core Metrics (Walk-Forward OOS across Timeframes)\n")
                for tf in ['SYNTHETIC', '1H', '30min', '15min', '5min']:
                    tf_res = god_results[tf]
                    f.write(f"### {tf} Data\n")
                    f.write(f"- **Expectancy:** {tf_res['walkforward']['out_of_sample']['expectancy']:.4f}\n")
                    f.write(f"- **Sharpe Ratio:** {tf_res['walkforward']['out_of_sample']['sharpe_ratio']:.3f}\n")
                    f.write(f"- **Permutation p-value:** {tf_res['permutation']['p_value']:.4f}\n\n")

    logger.info(f"Search complete in {time.time() - t0:.0f} seconds.")
    logger.info(f"Total Confirmed Strategies: {total_found}")

if __name__ == '__main__':
    main()
