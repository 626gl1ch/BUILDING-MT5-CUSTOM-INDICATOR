import os
import time
import pandas as pd
import numpy as np
import optuna
from backtest_core import BacktestCore
from logger_config import logger

def calculate_advanced_indicators_5m(df):
    """Calculates deep indicators for the 5m ML Demigod optimization."""
    close = df['close']
    high = df['high']
    low = df['low']
    open_pr = df['open']
    volume = df['volume']
    
    # --- ATR ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_50_mean'] = df['atr_14'].rolling(50).mean()
    
    # --- ADX ---
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
    
    # --- Chop ---
    atr_sum = tr.rolling(14).sum()
    high_max = high.rolling(14).max()
    low_min = low.rolling(14).min()
    df['chop_14'] = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(14)
    
    # --- SVP POC ---
    df['vwap'] = (close * volume).cumsum() / volume.cumsum()
    df['svp_poc'] = df['vwap'].rolling(24).mean()
    
    # --- MACD ---
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # --- Bollinger Bands (%B) ---
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = bb_sma + (bb_std * 2)
    df['bb_lower'] = bb_sma - (bb_std * 2)
    df['bb_pb'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # --- Volume ---
    df['vol_50_mean'] = volume.rolling(50).mean()
    
    # --- RSI & StochRSI ---
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    rsi_min = df['rsi_14'].rolling(14).min()
    rsi_max = df['rsi_14'].rolling(14).max()
    stoch_rsi = (df['rsi_14'] - rsi_min) / (rsi_max - rsi_min)
    df['stoch_k'] = stoch_rsi.rolling(3).mean() * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # --- Candle color ---
    df['is_green'] = close > open_pr
    df['is_red'] = close < open_pr
    
    return df

def generate_multi_strategy(df, p):
    close = df['close']
    high = df['high']
    low = df['low']
    
    signals = pd.Series(0, index=df.index)
    long_cond = pd.Series(False, index=df.index)
    short_cond = pd.Series(False, index=df.index)
    
    strategy_type = p.get('strategy_type', 'TREND_SCALP')
    
    if strategy_type == 'MACD_REV':
        chop_thresh = p.get('chop_thresh', 61.8)
        sweep_lb = p.get('sweep_lookback', 20)
        
        swing_high = high.rolling(sweep_lb).max()
        swing_low = low.rolling(sweep_lb).min()
        sweep_low_cond = (low < swing_low.shift(1)) & (close > swing_low.shift(1))
        sweep_high_cond = (high > swing_high.shift(1)) & (close < swing_high.shift(1))
        
        macd_up = df['macd_hist'] > df['macd_hist'].shift(1)
        macd_down = df['macd_hist'] < df['macd_hist'].shift(1)
        
        is_chop = df['chop_14'] > chop_thresh
        
        long_cond = sweep_low_cond & macd_up & is_chop
        short_cond = sweep_high_cond & macd_down & is_chop
        
    elif strategy_type == 'CAPITULATION':
        b_lower = p.get('b_lower', 0.05)
        b_upper = p.get('b_upper', 0.95)
        b_trigger_l = p.get('b_trigger_l', 0.10)
        b_trigger_s = p.get('b_trigger_s', 0.90)
        vol_mult = p.get('vol_mult', 1.5)
        atr_mult = p.get('atr_mult', 1.5)
        
        cap_long_setup = (df['bb_pb'] < b_lower).rolling(5).max() > 0
        cap_short_setup = (df['bb_pb'] > b_upper).rolling(5).max() > 0
        
        vol_spike = df['volume'] > (df['vol_50_mean'] * vol_mult)
        atr_spike = df['atr_14'] > (df['atr_50_mean'] * atr_mult)
        
        # Divergence proxy: price makes lower low, RSI makes higher low
        # We simplify to extreme conditions matching setup
        
        long_cond = cap_long_setup & vol_spike & atr_spike & (df['bb_pb'] > b_trigger_l)
        short_cond = cap_short_setup & vol_spike & atr_spike & (df['bb_pb'] < b_trigger_s)
        
    elif strategy_type == 'RSI_ADX_CHOP':
        adx_min = p.get('adx_min', 12)
        adx_max = p.get('adx_max', 25)
        chop_max = p.get('chop_max', 61.8)
        rsi_os = p.get('rsi_os', 30)
        rsi_ob = p.get('rsi_ob', 70)
        
        valid_regime = (df['adx_14'] > adx_min) & (df['adx_14'] < adx_max) & (df['chop_14'] < chop_max)
        
        stoch_k_cross_up = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        stoch_k_cross_down = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        
        long_cond = valid_regime & (df['rsi_14'] < rsi_os) & stoch_k_cross_up
        short_cond = valid_regime & (df['rsi_14'] > rsi_ob) & stoch_k_cross_down
        
    elif strategy_type == 'TREND_SCALP':
        regime = p.get('regime', 'trend')
        use_candle_color = p.get('use_candle_color', True)
        use_sweep = p.get('use_sweep', True)
        sweep_lb = p.get('sweep_lookback', 20)
        use_poc = p.get('use_poc', True)
        
        long_c = pd.Series(True, index=df.index)
        short_c = pd.Series(True, index=df.index)
        
        if use_candle_color:
            long_c = long_c & df['is_green']
            short_c = short_c & df['is_red']
            
        if regime == 'trend':
            long_c = long_c & (df['adx_14'] > 20)
            short_c = short_c & (df['adx_14'] > 20)
        elif regime == 'chop':
            long_c = long_c & (df['chop_14'] > 55)
            short_c = short_c & (df['chop_14'] > 55)
            
        if use_sweep:
            swing_high = high.rolling(sweep_lb).max()
            swing_low = low.rolling(sweep_lb).min()
            sweep_low_cond = (low < swing_low.shift(1)) & (close > swing_low.shift(1))
            sweep_high_cond = (high > swing_high.shift(1)) & (close < swing_high.shift(1))
            long_c = long_c & sweep_low_cond
            short_c = short_c & sweep_high_cond
            
        if use_poc:
            poc = df['svp_poc']
            long_c = long_c & (close > poc)
            short_c = short_c & (close < poc)
            
        long_cond = long_c
        short_cond = short_c
        
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals

class MLDemigod5mOptimizer:
    def __init__(self):
        self.core = BacktestCore()
        self.timeframe = "5min_1year"
        self.all_dfs = {}
        self.load_data()
        self.demi_god_path = "Demi-God strategies.txt"
        self.csv_path = "ml_5m_winners.csv"
        self.best_score = -999
        
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=[
                "Trial", "Strategy", "PF", "WR", "Sharpe", "Score", "Params"
            ]).to_csv(self.csv_path, index=False)
            
    def load_data(self):
        logger.info("Loading 5m data and calculating complex indicators...")
        data = self.core.load_all_data(suffix=self.timeframe)
        for symbol, df in data.items():
            self.all_dfs[symbol] = calculate_advanced_indicators_5m(df.copy())
        logger.info(f"Loaded {len(self.all_dfs)} datasets for 5m TF.")
        
    def objective(self, trial):
        strategy_type = trial.suggest_categorical('strategy_type', ['MACD_REV', 'CAPITULATION', 'RSI_ADX_CHOP', 'TREND_SCALP'])
        
        p = {'strategy_type': strategy_type}
        
        if strategy_type == 'MACD_REV':
            p['chop_thresh'] = trial.suggest_float('chop_thresh', 50.0, 70.0)
            p['sweep_lookback'] = trial.suggest_int('sweep_lookback', 10, 50)
            
        elif strategy_type == 'CAPITULATION':
            p['b_lower'] = trial.suggest_float('b_lower', 0.0, 0.1)
            p['b_upper'] = trial.suggest_float('b_upper', 0.9, 1.0)
            p['b_trigger_l'] = trial.suggest_float('b_trigger_l', 0.05, 0.2)
            p['b_trigger_s'] = trial.suggest_float('b_trigger_s', 0.8, 0.95)
            p['vol_mult'] = trial.suggest_float('vol_mult', 1.1, 3.0)
            p['atr_mult'] = trial.suggest_float('atr_mult', 1.1, 3.0)
            
        elif strategy_type == 'RSI_ADX_CHOP':
            p['adx_min'] = trial.suggest_float('adx_min', 5.0, 20.0)
            p['adx_max'] = trial.suggest_float('adx_max', 20.0, 35.0)
            p['chop_max'] = trial.suggest_float('chop_max', 50.0, 70.0)
            p['rsi_os'] = trial.suggest_float('rsi_os', 20.0, 40.0)
            p['rsi_ob'] = trial.suggest_float('rsi_ob', 60.0, 80.0)
            
        elif strategy_type == 'TREND_SCALP':
            p['use_candle_color'] = trial.suggest_categorical('use_candle_color', [True, False])
            p['regime'] = trial.suggest_categorical('regime', ['trend', 'chop', 'any'])
            p['use_sweep'] = trial.suggest_categorical('use_sweep', [True, False])
            p['sweep_lookback'] = trial.suggest_int('ts_sweep_lookback', 5, 50)
            p['use_poc'] = trial.suggest_categorical('use_poc', [True, False])
            
            if not p['use_sweep'] and p['regime'] == 'any' and not p['use_candle_color'] and not p['use_poc']:
                raise optuna.TrialPruned()
        
        p['sl_atr'] = round(trial.suggest_float('sl_atr', 0.5, 6.0, step=0.1), 1)
        p['tp_atr'] = round(trial.suggest_float('tp_atr', 0.5, 8.0, step=0.1), 1)
        p['max_bars_hold'] = trial.suggest_int('max_bars_hold', 10, 100)
        
        res = self.core.run_full_validation(
            self.all_dfs, 
            generate_multi_strategy, 
            p,
            min_trades_per_day=0.1,
            min_assets=1,
            n_permutations=20
        )
        
        if res['backtest']['total_trades'] < 20:
            return -10.0 # Heavy penalty for no trades
            
        pf = res['backtest']['profit_factor']
        wr = res['backtest']['win_rate']
        sharpe = res['walkforward']['out_of_sample']['sharpe_ratio'] if 'walkforward' in res else 0.0
        
        # 5m TF specific scoring: prioritize PF > 1.2 and WR > 55%
        score = (pf * 3.0) + (wr / 10.0) + sharpe
        
        if pf >= 1.2 and wr >= 55.0 and sharpe > 0 and res['passed']:
            score += 20.0 # Huge bonus for hitting Demi-God limits on 5m
            self.save_demi_god(trial.number, p, res, sharpe)
        elif res['passed']:
            score += 5.0
        else:
            score -= 2.0
            
        if score > self.best_score or score > 5.0:
            self.best_score = max(self.best_score, score)
            record = {
                "Trial": trial.number,
                "Strategy": strategy_type,
                "PF": pf,
                "WR": wr,
                "Sharpe": sharpe,
                "Score": round(score, 3),
                "Params": str(p)
            }
            pd.DataFrame([record]).to_csv(self.csv_path, mode='a', header=False, index=False)
            
        return score

    def save_demi_god(self, trial_num, p, res, sharpe):
        strategy_name = f"ML_GOD_MODE_5m_{p['strategy_type']}_TRIAL_{trial_num}"
        pf = res['backtest']['profit_factor']
        wr = res['backtest']['win_rate']
        
        with open(self.demi_god_path, "a") as f:
            f.write(f"\n================================================================================\n")
            f.write(f"RANK: ML_5M_TRIAL_{trial_num} | STRATEGY: {strategy_name}\n")
            f.write(f"ASSET/TIMEFRAME: ALL_ASSETS_5min_1year\n")
            f.write(f"================================================================================\n")
            f.write(f"PERFORMANCE (FULL AGGREGATE):\n")
            f.write(f"  * Profit Factor : {pf}\n")
            f.write(f"  * Win Rate      : {wr}%\n")
            f.write(f"  * OOS Sharpe    : {sharpe}\n")
            f.write(f"SETTINGS & RULES:\n")
            f.write(f"  {p}\n")
            f.write(f"================================================================================\n")
            
        logger.info(f"*** FOUND 5M DEMI-GOD! Trial {trial_num} | PF: {pf} | WR: {wr}% ***")

    def run(self, n_trials=500):
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        logger.info(f"Starting 5m ML Bayesian Optimization for {n_trials} trials...")
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info("Optimization Complete!")
        logger.info(f"Best Trial Score: {study.best_value}")
        logger.info(f"Best Params: {study.best_params}")

if __name__ == "__main__":
    opt = MLDemigod5mOptimizer()
    opt.run(n_trials=500)
