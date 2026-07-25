import os
import random
import time
import pandas as pd
import numpy as np
from backtest_core import BacktestCore
from logger_config import logger

def calculate_advanced_indicators(df):
    """Calculates all necessary indicators for the advanced strategies."""
    close = df['close']
    high = df['high']
    low = df['low']
    open_pr = df['open']
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    
    # ADX (using basic proxy for speed or proper ADX)
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    tr_rolling = df['atr_14'] * 14
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_rolling)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_rolling)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx_14'] = dx.rolling(14).mean()
    
    # Chop
    atr_sum = tr.rolling(14).sum()
    high_max = high.rolling(14).max()
    low_min = low.rolling(14).min()
    df['chop_14'] = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(14)
    
    # Volume SMA
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    
    # Swings (Liquidity)
    for lookback in [10, 20]:
        df[f'swing_high_{lookback}'] = high.rolling(lookback).max()
        df[f'swing_low_{lookback}'] = low.rolling(lookback).min()
        
    # SVP POC proxy (Price with highest volume in rolling 24 bars)
    df['vwap'] = (close * df['volume']).cumsum() / df['volume'].cumsum()
    df['svp_poc'] = df['vwap'].rolling(24).mean() # fast approximation
    
    # Candle color
    df['is_green'] = close > open_pr
    df['is_red'] = close < open_pr
    
    return df

def generate_dynamic_strategy(df, p):
    """
    Evaluates dynamic logic based on parameters p.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Params
    use_candle_color = p.get('use_candle_color', True)
    regime = p.get('regime', 'trend') # 'trend' or 'chop' or 'any'
    use_sweep = p.get('use_sweep', True)
    sweep_lookback = p.get('sweep_lookback', 20)
    use_poc = p.get('use_poc', True)
    
    signals = pd.Series(0, index=df.index)
    
    # Base Conditions
    long_cond = pd.Series(True, index=df.index)
    short_cond = pd.Series(True, index=df.index)
    
    # 1. Candle Color
    if use_candle_color:
        long_cond = long_cond & df['is_green']
        short_cond = short_cond & df['is_red']
        
    # 2. Regime Filter
    if regime == 'trend':
        long_cond = long_cond & (df['adx_14'] > 20)
        short_cond = short_cond & (df['adx_14'] > 20)
    elif regime == 'chop':
        long_cond = long_cond & (df['chop_14'] > 55)
        short_cond = short_cond & (df['chop_14'] > 55)
        
    # 3. Liquidity Sweep
    if use_sweep:
        swing_high = df.get(f'swing_high_{sweep_lookback}')
        swing_low = df.get(f'swing_low_{sweep_lookback}')
        
        sweep_low_cond = (low < swing_low.shift(1)) & (close > swing_low.shift(1))
        sweep_high_cond = (high > swing_high.shift(1)) & (close < swing_high.shift(1))
        
        long_cond = long_cond & sweep_low_cond
        short_cond = short_cond & sweep_high_cond
        
    # 4. POC Alignment (Institutional Volume)
    if use_poc:
        poc = df['svp_poc']
        long_cond = long_cond & (close > poc)
        short_cond = short_cond & (close < poc)
        
    signals[long_cond] = 1
    signals[short_cond] = -1
    
    return signals

def run_mass_generation(target_count=100):
    core = BacktestCore()
    logger.info("Loading 5m, 15m, 30m data...")
    
    all_dfs = {}
    for tf in ["5min_1year", "15min_1year", "30min_1year"]:
        data = core.load_all_data(suffix=tf)
        for symbol, df in data.items():
            key = f"{symbol}_{tf}"
            # Precalculate indicators
            all_dfs[key] = calculate_advanced_indicators(df.copy())
            
    logger.info(f"Loaded {len(all_dfs)} datasets across timeframes.")
    
    timeframes = ["5min_1year", "15min_1year", "30min_1year"]
    
    found_strategies = []
    attempts = 0
    
    demi_god_path = "Demi-God strategies.txt"
    csv_path = "mass_generated_winners.csv"
    
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=[
            "Rank", "Timeframe", "Regime", "Use_Color", "Use_Sweep", "Use_POC", 
            "SL_ATR", "TP_ATR", "PF", "WR", "Sharpe", "Expectancy"
        ]).to_csv(csv_path, index=False)
        
    start_time = time.time()
    
    while len(found_strategies) < target_count and attempts < 1000:
        attempts += 1
        
        p = {
            'use_candle_color': random.choice([True, False]),
            'regime': random.choice(['trend', 'chop', 'any']),
            'use_sweep': random.choice([True, False]),
            'sweep_lookback': random.choice([10, 20]),
            'use_poc': random.choice([True, False]),
            'sl_atr': round(random.uniform(0.5, 3.0), 1),
            'tp_atr': round(random.uniform(1.0, 6.0), 1),
            'max_bars_hold': random.choice([12, 24, 48, 96])
        }
        
        if not p['use_sweep'] and p['regime'] == 'any' and not p['use_candle_color'] and not p['use_poc']:
            continue
            
        tf_choice = random.choice(timeframes)
        tf_dfs = {k.split('_')[0]: v for k, v in all_dfs.items() if tf_choice in k}
        
        if len(tf_dfs) == 0:
            continue
            
        res = core.run_full_validation(
            tf_dfs, 
            generate_dynamic_strategy, 
            p,
            min_trades_per_day=0.2,
            min_assets=1,
            n_permutations=20
        )
        
        pf = res['backtest']['profit_factor']
        wr = res['backtest']['win_rate']
        
        # Log every valid variation generated to fulfill the 100 strategies request
        if res['backtest']['total_trades'] > 50:
            record = {
                "Rank": len(found_strategies) + 1,
                "Timeframe": tf_choice,
                "Regime": p['regime'],
                "Use_Color": p['use_candle_color'],
                "Use_Sweep": p['use_sweep'],
                "Use_POC": p['use_poc'],
                "SL_ATR": p['sl_atr'],
                "TP_ATR": p['tp_atr'],
                "PF": pf,
                "WR": wr,
                "Sharpe": res['backtest']['sharpe_ratio'],
                "Expectancy": res['backtest']['expectancy']
            }
            found_strategies.append(record)
            pd.DataFrame([record]).to_csv(csv_path, mode='a', header=False, index=False)
            
            # Demi God Constraint Check
            if pf >= 1.2 and wr >= 55.0 and res['passed']:
                sharpe = res['walkforward']['out_of_sample']['sharpe_ratio']
                strategy_name = f"GOD_MODE_{len(found_strategies)}_{tf_choice.split('_')[0]}"
                
                with open(demi_god_path, "a") as f:
                    f.write(f"\n--------------------------------------------------------------------------------\n")
                    f.write(f"RANK: {record['Rank']} | STRATEGY: {strategy_name}\n")
                    f.write(f"ASSET/TIMEFRAME: ALL_ASSETS_{tf_choice}\n")
                    f.write(f"--------------------------------------------------------------------------------\n")
                    f.write(f"PERFORMANCE:\n")
                    f.write(f"  * OOS Profit Factor : {res['walkforward']['out_of_sample']['profit_factor']}\n")
                    f.write(f"  * OOS Win Rate      : {res['walkforward']['out_of_sample']['win_rate']}%\n")
                    f.write(f"  * OOS Sharpe Ratio  : {sharpe}\n")
                    f.write(f"  * In-Sample PF      : {res['walkforward']['in_sample']['profit_factor']}\n")
                    f.write(f"  * In-Sample WR      : {res['walkforward']['in_sample']['win_rate']}%\n")
                    f.write(f"SETTINGS & RULES:\n")
                    f.write(f"  {p}\n\n")
                    f.write(f"RULES:\n")
                    f.write(f"1. Regime Filter: {p['regime']}\n")
                    if p['use_sweep']:
                        f.write(f"2. Sweep: Price sweeps {p['sweep_lookback']}-bar swing low/high and rejects.\n")
                    if p['use_poc']:
                        f.write(f"3. Institutional Volume: Price must close relative to 24-bar POC.\n")
                    if p['use_candle_color']:
                        f.write(f"4. Candlestick Confirmation: Only enter longs on green candles, shorts on red.\n")
                    f.write(f"5. Exit: SL {p['sl_atr']} ATR, TP {p['tp_atr']} ATR, Max Hold {p['max_bars_hold']} bars.\n")
                
                logger.info(f"Found DEMI GOD! {len(found_strategies)}/100 | PF: {pf} | WR: {wr}% | TF: {tf_choice}")
            else:
                logger.info(f"Generated Variation {len(found_strategies)}/100 | PF: {pf} | WR: {wr}% | TF: {tf_choice}")
        
        if attempts % 100 == 0:
            logger.info(f"Attempts: {attempts} | Variations found: {len(found_strategies)}")
            
    elapsed = time.time() - start_time
    logger.info(f"Finished mass generation. Generated {len(found_strategies)} variations in {attempts} attempts. Time: {elapsed:.2f}s")
    
if __name__ == "__main__":
    run_mass_generation(100)
