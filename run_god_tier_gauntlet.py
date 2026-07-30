import pandas as pd
import numpy as np
import glob
import os
import json
import warnings
from backtest_core import BacktestCore
from logger_config import logger
from indicators_library import (
    calc_adx, calc_atr, calc_rsi, calc_stoch_rsi,
    calc_bollinger_bands, calc_choppiness_index, calc_macd, calc_vwap, calc_sma
)

warnings.filterwarnings('ignore')

def load_data(tf_suffix):
    """Loads all data files matching a timeframe suffix, e.g. '5min_1year.csv'"""
    files = glob.glob(f"*{tf_suffix}")
    data = {}
    for f in files:
        if 'SYNTHETIC' in f: continue  # Skip synthetic to evaluate on raw assets
        sym = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        if 'volume' not in df.columns:
            df['volume'] = 1000
        data[sym] = df
    return data

# =====================================================================
# SIGNAL FUNCTIONS (MATHEMATICALLY EXACT FROM DEMI-GOD)
# =====================================================================
def ensure_columns(df):
    if 'high' not in df.columns: df['high'] = df['close']
    if 'low' not in df.columns: df['low'] = df['close']
    if 'open' not in df.columns: df['open'] = df['close']
    return df

def signal_macd_rev(df, p):
    df = ensure_columns(df)
    chop_thresh = p.get('chop_thresh', 61.8)
    sweep_lb = p.get('sweep_lookback', 20)
    
    swing_high = df['high'].rolling(sweep_lb).max()
    swing_low = df['low'].rolling(sweep_lb).min()
    
    sweep_low_cond = (df['low'] < swing_low.shift(1)) & (df['close'] > swing_low.shift(1))
    sweep_high_cond = (df['high'] > swing_high.shift(1)) & (df['close'] < swing_high.shift(1))
    
    macd_l, macd_s, macd_h = calc_macd(df['close'], fast=12, slow=26, signal=9)
    macd_up = macd_h > macd_h.shift(1)
    macd_down = macd_h < macd_h.shift(1)
    
    chop = calc_choppiness_index(df, 14)
    is_chop = chop > chop_thresh
    
    long_cond = sweep_low_cond & macd_up & is_chop
    short_cond = sweep_high_cond & macd_down & is_chop
    
    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals

def signal_capitulation(df, p):
    b_lower = p.get('b_lower', 0.05)
    b_upper = p.get('b_upper', 0.95)
    b_trigger_l = p.get('b_trigger_l', 0.10)
    b_trigger_s = p.get('b_trigger_s', 0.90)
    vol_mult = p.get('vol_mult', 1.5)
    atr_mult = p.get('atr_mult', 1.5)
    
    bb_u, bb_m, bb_l, bb_pb, bw = calc_bollinger_bands(df['close'], 20, 2.0)
    vol_50 = df['volume'].rolling(50).mean()
    atr_14 = calc_atr(df, 14)
    atr_50 = atr_14.rolling(50).mean()
    
    cap_long_setup = (bb_pb < b_lower).rolling(5).max() > 0
    cap_short_setup = (bb_pb > b_upper).rolling(5).max() > 0
    
    vol_spike = df['volume'] > (vol_50 * vol_mult)
    atr_spike = atr_14 > (atr_50 * atr_mult)
    
    long_cond = cap_long_setup & vol_spike & atr_spike & (bb_pb > b_trigger_l)
    short_cond = cap_short_setup & vol_spike & atr_spike & (bb_pb < b_trigger_s)
    
    if p.get('mrf_v2_extra', False):
        rsi = calc_rsi(df['close'], 14)
        price_low = df['close'] == df['close'].rolling(12).min()
        rsi_not_low = rsi > rsi.rolling(12).min() + 3
        long_cond = long_cond & (price_low.shift(1) & rsi_not_low.shift(1))
    
    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals

def signal_rsi_adx_chop(df, p):
    adx_min = p.get('adx_min', 12)
    adx_max = p.get('adx_max', 25)
    chop_max = p.get('chop_max', 61.8)
    rsi_os = p.get('rsi_os', 30)
    rsi_ob = p.get('rsi_ob', 70)
    
    adx, _, _ = calc_adx(df, 10 if p.get('mrh_extra', False) else 14)
    chop = calc_choppiness_index(df, 14)
    rsi = calc_rsi(df['close'], 14)
    
    stoch_k, stoch_d = calc_stoch_rsi(df['close'], 14, 3, 3)
    
    valid_regime = (adx > adx_min) & (adx < adx_max) & (chop < chop_max)
    
    stoch_k_cross_up = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
    stoch_k_cross_down = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))
    
    if p.get('mrh_extra', False):
        atr_14 = calc_atr(df, 14)
        atr_25p = atr_14.rolling(50).quantile(0.25)
        valid_regime = valid_regime & (atr_14 > atr_25p)
        stoch_k_cross_up = (stoch_k > stoch_k.shift(1)) & (stoch_k.shift(1) < 15)
        stoch_k_cross_down = (stoch_k < stoch_k.shift(1)) & (stoch_k.shift(1) > 85)
        
    long_cond = valid_regime & (rsi < rsi_os) & stoch_k_cross_up
    short_cond = valid_regime & (rsi > rsi_ob) & stoch_k_cross_down
    
    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals

def signal_trend_scalper(df, p):
    df = ensure_columns(df)
    regime = p.get('regime', 'any')
    use_sweep = p.get('use_sweep', False)
    sweep_lb = p.get('sweep_lookback', 20)
    use_poc = p.get('use_poc', True)
    
    is_green = df['close'] > df['open']
    is_red = df['close'] < df['open']
    
    long_c = is_green
    short_c = is_red
    
    if regime == 'trend':
        adx, _, _ = calc_adx(df, 14)
        long_c = long_c & (adx > 20)
        short_c = short_c & (adx > 20)
        
    if use_sweep:
        swing_high = df['high'].rolling(sweep_lb).max()
        swing_low = df['low'].rolling(sweep_lb).min()
        sweep_low_cond = (df['low'] < swing_low.shift(1)) & (df['close'] > swing_low.shift(1))
        sweep_high_cond = (df['high'] > swing_high.shift(1)) & (df['close'] < swing_high.shift(1))
        long_c = long_c & sweep_low_cond
        short_c = short_c & sweep_high_cond
        
    if use_poc:
        vwap = calc_vwap(df)
        poc = vwap.rolling(24).mean()
        long_c = long_c & (df['close'] > poc)
        short_c = short_c & (df['close'] < poc)
        
    signals = pd.Series(0, index=df.index)
    signals[long_c] = 1
    signals[short_c] = -1
    return signals

# =====================================================================
# THE STRATEGIES TO TEST
# =====================================================================
STRATEGIES = [
    # --- 1H / HTF Manual Demigods ---
    {
        "name": "RANK_128_MACD_MEAN_REVERSION",
        "tf": "1H_1year.csv",
        "func": lambda df, p: signal_macd_rev(df, p),
        "params": {"sl_atr": 1.5, "tp_atr": 3.0, "max_bars_hold": 99, "chop_thresh": 61.8, "sweep_lookback": 20}
    },
    {
        "name": "RANK_106_MR_H_RSI_ADX",
        "tf": "1H_1year.csv",
        "func": lambda df, p: signal_rsi_adx_chop(df, p),
        "params": {"sl_atr": 1.5, "tp_atr": 2.5, "max_bars_hold": 99, "mrh_extra": True, "adx_min": 12, "adx_max": 25, "chop_max": 61.8, "rsi_os": 30, "rsi_ob": 70}
    },
    # --- 30m Trend & Scalp ---
    {
        "name": "ML_TRIAL_152_TREND_FOLLOWER",
        "tf": "30min_1year.csv",
        "func": lambda df, p: signal_trend_scalper(df, p),
        "params": {'use_candle_color': True, 'regime': 'trend', 'use_sweep': False, 'use_poc': True, 'sl_atr': 4.6, 'tp_atr': 7.3, 'max_bars_hold': 94}
    },
    {
        "name": "ML_TRIAL_299_SCALPER",
        "tf": "30min_1year.csv",
        "func": lambda df, p: signal_trend_scalper(df, p),
        "params": {'use_candle_color': True, 'regime': 'any', 'use_sweep': True, 'sweep_lookback': 46, 'use_poc': True, 'sl_atr': 4.9, 'tp_atr': 1.5, 'max_bars_hold': 84}
    },
    # --- 5m Capitulation ---
    {
        "name": "RANK_87_MR_F_V2_CAPITULATION",
        "tf": "5min_1year.csv",
        "func": lambda df, p: signal_capitulation(df, p),
        "params": {"sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 99, "mrf_v2_extra": True, 'b_lower': 0.05, 'b_upper': 0.95, 'b_trigger_l': 0.10, 'b_trigger_s': 0.90, 'vol_mult': 1.5, 'atr_mult': 1.5}
    },
    {
        "name": "ML_5M_TRIAL_398_CAPITULATION",
        "tf": "5min_1year.csv",
        "func": lambda df, p: signal_capitulation(df, p),
        "params": {'b_lower': 0.0228, 'b_upper': 0.9027, 'b_trigger_l': 0.1776, 'b_trigger_s': 0.8770, 'vol_mult': 2.816, 'atr_mult': 2.678, 'sl_atr': 3.2, 'tp_atr': 2.6, 'max_bars_hold': 97}
    },
    {
        "name": "ML_5M_TRIAL_161_CAPITULATION",
        "tf": "5min_1year.csv",
        "func": lambda df, p: signal_capitulation(df, p),
        "params": {'b_lower': 0.0887, 'b_upper': 0.9660, 'b_trigger_l': 0.1779, 'b_trigger_s': 0.8391, 'vol_mult': 2.523, 'atr_mult': 2.874, 'sl_atr': 5.8, 'tp_atr': 4.4, 'max_bars_hold': 87}
    },
    # --- 15m RSI ADX & Capitulation ---
    {
        "name": "ML_15MIN_TRIAL_36_RSI_ADX_CHOP",
        "tf": "15min_1year.csv",
        "func": lambda df, p: signal_rsi_adx_chop(df, p),
        "params": {'adx_min': 19.69, 'adx_max': 21.16, 'chop_max': 50.38, 'rsi_os': 20.90, 'rsi_ob': 60.08, 'sl_atr': 4.8, 'tp_atr': 1.6, 'max_bars_hold': 46}
    },
    {
        "name": "ML_15MIN_TRIAL_53_CAPITULATION",
        "tf": "15min_1year.csv",
        "func": lambda df, p: signal_capitulation(df, p),
        "params": {'b_lower': 0.0111, 'b_upper': 0.9097, 'b_trigger_l': 0.1596, 'b_trigger_s': 0.9465, 'vol_mult': 2.987, 'atr_mult': 2.350, 'sl_atr': 2.2, 'tp_atr': 1.4, 'max_bars_hold': 25}
    },
    # --- 30m RSI ADX & MACD REV ---
    {
        "name": "ML_30MIN_TRIAL_101_RSI_ADX_CHOP",
        "tf": "30min_1year.csv",
        "func": lambda df, p: signal_rsi_adx_chop(df, p),
        "params": {'adx_min': 10.82, 'adx_max': 22.63, 'chop_max': 60.92, 'rsi_os': 27.17, 'rsi_ob': 77.01, 'sl_atr': 4.0, 'tp_atr': 0.6, 'max_bars_hold': 47}
    },
    {
        "name": "ML_30MIN_TRIAL_157_MACD_REV",
        "tf": "30min_1year.csv",
        "func": lambda df, p: signal_macd_rev(df, p),
        "params": {'chop_thresh': 64.85, 'sweep_lookback': 23, 'sl_atr': 2.2, 'tp_atr': 1.5, 'max_bars_hold': 48}
    }
]

# =====================================================================
# THE GOD TIER GAUNTLET ENGINE
# =====================================================================
def run_god_tier_gauntlet():
    logger.info("Initializing God Tier Gauntlet...")
    
    # Preload all data sets
    dfs = {
        '5min': load_data('5min_1year.csv'),
        '15min': load_data('15min_1year.csv'),
        '30min': load_data('30min_1year.csv'),
        '1H': load_data('1H_1year.csv')
    }
    
    survivors = []
    engine = BacktestCore(commission=0) # Base commission handled by our cost stress if needed
    
    for strat in STRATEGIES:
        tf_key = strat['tf'].split('_')[0]
        data_dict = dfs.get(tf_key, {})
        
        if not data_dict:
            logger.warning(f"No data loaded for {tf_key}. Skipping {strat['name']}.")
            continue
            
        logger.info(f"\n[{strat['name']}] Initiating Gauntlet...")
        
        # GATE 1: FULL VALIDATION (Walkforward + Monte Carlo)
        # Using tight gates to only allow true God Tier passes
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
        
        if not res['passed']:
            logger.info(f"[{strat['name']}] Failed Base Gauntlet (P-Value: {res['permutation'].get('p_value', 1.0)}). Eliminating.")
            continue
            
        logger.info(f"[{strat['name']}] Passed Base Gauntlet! Initiating Cost Stress Test...")
        
        # GATE 2: COST STRESS TEST
        cost_passed = False
        stressed_pf = 0
        for mult in [1.5, 2.0]:
            stress_res = engine.run_full_validation(
                data_dict,
                lambda df, p: strat['func'](df, p),
                strat['params'],
                min_trades_per_day=0.1,
                min_assets=1,
                n_permutations=20, # Fast check
                slippage_pct=0.0002 * mult,
                fee_pct=0.0005 * mult
            )
            if stress_res['passed']:
                cost_passed = True
                stressed_pf = stress_res['backtest'].get('profit_factor', 0)
            else:
                cost_passed = False
                break
                
        if not cost_passed:
            logger.info(f"[{strat['name']}] Crumbled under Cost Stress. Eliminating.")
            continue
            
        logger.info(f"[{strat['name']}] Passed Cost Stress! Initiating Sensitivity Sweep...")
        
        # GATE 3: SENSITIVITY SWEEP
        sens_passed = True
        for shift in [-0.15, 0.15]:
            p_shifted = {}
            for k, v in strat['params'].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    new_v = v * (1.0 + shift)
                    if isinstance(v, int): new_v = max(1, int(round(new_v)))
                    p_shifted[k] = new_v
                else:
                    p_shifted[k] = v
            
            sens_res = engine.run_full_validation(
                data_dict,
                lambda df, p: strat['func'](df, p),
                p_shifted,
                min_trades_per_day=0.1,
                min_assets=1,
                n_permutations=20, # Fast check
                slippage_pct=0.0002,
                fee_pct=0.0005
            )
            if not sens_res['passed']:
                sens_passed = False
                break
                
        if not sens_passed:
            logger.info(f"[{strat['name']}] Failed Sensitivity Sweep (Knife-Edge Overfit). Eliminating.")
            continue
            
        # GOD TIER SURVIVOR
        logger.info(f"[{strat['name']}] SURVIVED ALL GAUNTLETS. GOD TIER CERTIFIED.")
        survivors.append({
            "Strategy": strat['name'],
            "Base_PF": res['backtest'].get('profit_factor', 0),
            "Base_WR": res['backtest'].get('win_rate', 0),
            "Perm_PValue": res['permutation'].get('p_value', 1.0),
            "OOS_Sharpe": res['walk_forward'].get('oos_sharpe_mean', 0),
            "Cost_Stressed_PF": stressed_pf
        })
        
    # Write report
    df_surv = pd.DataFrame(survivors)
    with open("GOD_TIER_SURVIVORS.md", "w") as f:
        f.write("# God Tier Gauntlet Survivors\n\n")
        if df_surv.empty:
            f.write("No strategies survived the ultimate gauntlet. The market is brutal.\n")
        else:
            df_surv = df_surv.sort_values(by="Base_PF", ascending=False)
            f.write(df_surv.to_markdown(index=False))
            f.write("\n\nAll of these strategies passed the 3-Layer Gauntlet, Cost Stress (2x), and Parameter Sensitivity Sweep (+-15%).")
            
    logger.info("Gauntlet Complete. Results saved to GOD_TIER_SURVIVORS.md.")

if __name__ == "__main__":
    run_god_tier_gauntlet()
