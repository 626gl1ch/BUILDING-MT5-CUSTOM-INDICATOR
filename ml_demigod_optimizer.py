import os
import time
import pandas as pd
import numpy as np
import optuna
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
    
    # ADX (proxy)
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
    
    # SVP POC proxy
    df['vwap'] = (close * df['volume']).cumsum() / df['volume'].cumsum()
    df['svp_poc'] = df['vwap'].rolling(24).mean()
    
    # Candle color
    df['is_green'] = close > open_pr
    df['is_red'] = close < open_pr
    
    return df

def generate_dynamic_strategy(df, p):
    close = df['close']
    high = df['high']
    low = df['low']
    
    use_candle_color = p.get('use_candle_color', True)
    regime = p.get('regime', 'trend')
    use_sweep = p.get('use_sweep', True)
    sweep_lookback = p.get('sweep_lookback', 20)
    use_poc = p.get('use_poc', True)
    
    signals = pd.Series(0, index=df.index)
    long_cond = pd.Series(True, index=df.index)
    short_cond = pd.Series(True, index=df.index)
    
    if use_candle_color:
        long_cond = long_cond & df['is_green']
        short_cond = short_cond & df['is_red']
        
    if regime == 'trend':
        long_cond = long_cond & (df['adx_14'] > 20)
        short_cond = short_cond & (df['adx_14'] > 20)
    elif regime == 'chop':
        long_cond = long_cond & (df['chop_14'] > 55)
        short_cond = short_cond & (df['chop_14'] > 55)
        
    if use_sweep:
        swing_high = high.rolling(sweep_lookback).max()
        swing_low = low.rolling(sweep_lookback).min()
        
        sweep_low_cond = (low < swing_low.shift(1)) & (close > swing_low.shift(1))
        sweep_high_cond = (high > swing_high.shift(1)) & (close < swing_high.shift(1))
        
        long_cond = long_cond & sweep_low_cond
        short_cond = short_cond & sweep_high_cond
        
    if use_poc:
        poc = df['svp_poc']
        long_cond = long_cond & (close > poc)
        short_cond = short_cond & (close < poc)
        
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals

class MLStrategyOptimizer:
    def __init__(self, timeframes=["30min_1year", "15min_1year", "5min_1year"]):
        self.core = BacktestCore()
        self.timeframes = timeframes
        self.all_dfs = {}
        self.load_data()
        self.demi_god_path = "Demi-God strategies.txt"
        self.csv_path = "ml_optimized_winners.csv"
        self.best_score = -999
        
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=[
                "Trial", "Timeframe", "Regime", "Use_Color", "Use_Sweep", "Use_POC", 
                "SL_ATR", "TP_ATR", "PF", "WR", "Sharpe", "Score"
            ]).to_csv(self.csv_path, index=False)
            
    def load_data(self):
        logger.info("Loading and calculating indicators for data...")
        for tf in self.timeframes:
            data = self.core.load_all_data(suffix=tf)
            for symbol, df in data.items():
                key = f"{symbol}_{tf}"
                self.all_dfs[key] = calculate_advanced_indicators(df.copy())
        logger.info(f"Loaded {len(self.all_dfs)} datasets.")
        
    def objective(self, trial):
        p = {
            'use_candle_color': trial.suggest_categorical('use_candle_color', [True, False]),
            'regime': trial.suggest_categorical('regime', ['trend', 'chop', 'any']),
            'use_sweep': trial.suggest_categorical('use_sweep', [True, False]),
            'sweep_lookback': trial.suggest_int('sweep_lookback', 5, 50),
            'use_poc': trial.suggest_categorical('use_poc', [True, False]),
            'sl_atr': round(trial.suggest_float('sl_atr', 0.5, 5.0, step=0.1), 1),
            'tp_atr': round(trial.suggest_float('tp_atr', 1.0, 10.0, step=0.1), 1),
            'max_bars_hold': trial.suggest_int('max_bars_hold', 5, 100)
        }
        
        tf_choice = trial.suggest_categorical('timeframe', self.timeframes)
        
        # Prune invalid logic trees
        if not p['use_sweep'] and p['regime'] == 'any' and not p['use_candle_color'] and not p['use_poc']:
            raise optuna.TrialPruned()
            
        tf_dfs = {k.split('_')[0]: v for k, v in self.all_dfs.items() if tf_choice in k}
        
        res = self.core.run_full_validation(
            tf_dfs, 
            generate_dynamic_strategy, 
            p,
            min_trades_per_day=0.1,
            min_assets=1,
            n_permutations=20
        )
        
        if res['backtest']['total_trades'] < 30:
            return -10.0 # Heavy penalty for no trades
            
        pf = res['backtest']['profit_factor']
        wr = res['backtest']['win_rate']
        sharpe = res['walkforward']['out_of_sample']['sharpe_ratio'] if 'walkforward' in res else 0.0
        
        # Composite score
        # PF is king, but we need WR and Sharpe
        score = (pf * 2.0) + (wr / 20.0) + sharpe
        
        if res['passed']:
            score += 5.0 # Massive bonus for passing permutation/WF gates
            
            # Check Demi-God status
            if pf >= 1.2 and wr >= 55.0 and sharpe > 0:
                self.save_demi_god(trial.number, tf_choice, p, res, sharpe)
        else:
            # Penalize slightly if it failed gates but don't zero it so the ML can learn
            score -= 2.0
            
        # Log to CSV if it's decent
        if score > self.best_score or score > 4.0:
            self.best_score = max(self.best_score, score)
            record = {
                "Trial": trial.number,
                "Timeframe": tf_choice,
                "Regime": p['regime'],
                "Use_Color": p['use_candle_color'],
                "Use_Sweep": p['use_sweep'],
                "Use_POC": p['use_poc'],
                "SL_ATR": p['sl_atr'],
                "TP_ATR": p['tp_atr'],
                "PF": pf,
                "WR": wr,
                "Sharpe": sharpe,
                "Score": round(score, 3)
            }
            pd.DataFrame([record]).to_csv(self.csv_path, mode='a', header=False, index=False)
            
        return score

    def save_demi_god(self, trial_num, tf_choice, p, res, sharpe):
        strategy_name = f"ML_GOD_MODE_TRIAL_{trial_num}_{tf_choice.split('_')[0]}"
        pf = res['backtest']['profit_factor']
        wr = res['backtest']['win_rate']
        
        with open(self.demi_god_path, "a") as f:
            f.write(f"\n================================================================================\n")
            f.write(f"RANK: ML_TRIAL_{trial_num} | STRATEGY: {strategy_name}\n")
            f.write(f"ASSET/TIMEFRAME: ALL_ASSETS_{tf_choice}\n")
            f.write(f"================================================================================\n")
            f.write(f"PERFORMANCE (FULL AGGREGATE):\n")
            f.write(f"  * Profit Factor : {pf}\n")
            f.write(f"  * Win Rate      : {wr}%\n")
            f.write(f"  * OOS Sharpe    : {sharpe}\n")
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
            f.write(f"================================================================================\n")
            
        logger.info(f"*** FOUND DEMI-GOD! Trial {trial_num} | PF: {pf} | WR: {wr}% | TF: {tf_choice} ***")

    def run(self, n_trials=300):
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        logger.info(f"Starting ML Bayesian Optimization for {n_trials} trials...")
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info("Optimization Complete!")
        logger.info(f"Best Trial Score: {study.best_value}")
        logger.info(f"Best Params: {study.best_params}")

if __name__ == "__main__":
    opt = MLStrategyOptimizer()
    opt.run(n_trials=300)
