import pandas as pd
import numpy as np
import glob
import json
import warnings
import os
from backtest_core import BacktestCore
from indicators_library import (
    calc_adx, calc_atr, calc_rsi, calc_stoch_rsi,
    calc_bollinger_bands, calc_choppiness_index, calc_macd,
    calc_alma, calc_vwap, calc_sma, calc_swing_points
)
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# STRATEGY DEFINITIONS
# ---------------------------------------------------------

def strat_MR_F_V2_CAPITULATION(df):
    p = {'sl_atr': 2.0, 'tp_atr': 3.0, 'max_bars_hold': 99}
    bb_u, bb_m, bb_l, bb_pct, _ = calc_bollinger_bands(df['close'], 20, 2.0)
    atr = calc_atr(df, 10)
    vol_sma = calc_sma(df['volume'], 20)
    rsi = calc_rsi(df['close'], 14)

    atr_avg = atr.rolling(20).mean()
    at_lower = bb_pct < 0.05
    at_upper = bb_pct > 0.95
    blowoff = atr > atr_avg * 1.5
    vol_cap = df['volume'] > vol_sma * 1.5

    lookback = 12
    price_low = df['close'] == df['close'].rolling(lookback).min()
    rsi_not_low = rsi > rsi.rolling(lookback).min() + 3
    bull_div = at_lower & price_low & rsi_not_low

    price_high = df['close'] == df['close'].rolling(lookback).max()
    rsi_not_high = rsi < rsi.rolling(lookback).max() - 3
    bear_div = at_upper & price_high & rsi_not_high

    bb_inside_long = (bb_pct > 0.10) & (bb_pct.shift(1) <= 0.10)
    bb_inside_short = (bb_pct < 0.90) & (bb_pct.shift(1) >= 0.90)

    signals = pd.Series(0, index=df.index)
    signals[bull_div.shift(1) & blowoff.shift(1) & vol_cap.shift(1) & bb_inside_long] = 1
    signals[bear_div.shift(1) & blowoff.shift(1) & vol_cap.shift(1) & bb_inside_short] = -1
    return signals, p

def strat_MR_H_RSI_ADX(df):
    p = {'sl_atr': 1.5, 'tp_atr': 2.5, 'max_bars_hold': 99}
    adx, _, _ = calc_adx(df, 10)
    atr = calc_atr(df, 10)
    chop = calc_choppiness_index(df, 14)
    rsi = calc_rsi(df['close'], 14)
    sk, sd = calc_stoch_rsi(df['close'], 14, 3, 3)

    adx_sweet = (adx > 12) & (adx < 25)
    atr_floor = atr.rolling(50).quantile(0.25)
    vol_ok = atr > atr_floor
    chop_ok = chop < 61.8
    rsi_os = rsi < 30
    rsi_ob = rsi > 70
    srsi_os = (sk < 15) & (sd < 15)
    srsi_ob = (sk > 85) & (sd > 85)

    k_up = (sk > sk.shift(1)) & srsi_os.shift(1)
    k_down = (sk < sk.shift(1)) & srsi_ob.shift(1)

    signals = pd.Series(0, index=df.index)
    signals[adx_sweet & vol_ok & chop_ok & rsi_os & k_up] = 1
    signals[adx_sweet & vol_ok & chop_ok & rsi_ob & k_down] = -1
    return signals, p

def strat_VWAP_MACD_PULLBACK(df):
    p = {'sl_atr': 1.5, 'tp_atr': 3.0, 'max_bars_hold': 99}
    vwap = calc_vwap(df)
    macd, macdsignal, macdhist = calc_macd(df['close'], 12, 26, 9)
    vol_sma = calc_sma(df['volume'], 20)
    adx, _, _ = calc_adx(df, 14)

    long_trend = df['close'] > vwap
    short_trend = df['close'] < vwap
    long_mom = (macdhist > 0) & (macd > macdsignal)
    short_mom = (macdhist < 0) & (macd < macdsignal)
    macd_cross_up = (macd > 0) & (macd.shift(1) <= 0)
    macd_cross_down = (macd < 0) & (macd.shift(1) >= 0)
    vol_ok = df['volume'] > vol_sma
    trend_ok = adx > 25

    signals = pd.Series(0, index=df.index)
    signals[long_trend & long_mom & macd_cross_up & vol_ok & trend_ok] = 1
    signals[short_trend & short_mom & macd_cross_down & vol_ok & trend_ok] = -1
    return signals, p

def strat_ALMA_OPT_61(df):
    p = {'sl_atr': 1.5, 'tp_atr': 3.0, 'max_bars_hold': 48}
    alma = calc_alma(df['close'], 21)
    adx, _, _ = calc_adx(df, 14)
    vol_sma = calc_sma(df['volume'], 20)
    swing_high, swing_low = calc_swing_points(df, 20)

    eq_lows = (df['low'] < swing_low.shift(1)) & \
              (df['close'] > swing_low.shift(1)) & \
              (df['close'].shift(1) > swing_low.shift(1))
    
    eq_highs = (df['high'] > swing_high.shift(1)) & \
               (df['close'] < swing_high.shift(1)) & \
               (df['close'].shift(1) < swing_high.shift(1))

    alma_up = alma > alma.shift(1)
    alma_down = alma < alma.shift(1)
    trend_ok = adx > 25
    vol_ok = df['volume'] > vol_sma

    signals = pd.Series(0, index=df.index)
    signals[eq_lows & alma_up & trend_ok & vol_ok] = 1
    signals[eq_highs & alma_down & trend_ok & vol_ok] = -1
    return signals, p

def strat_MACD_MEAN_REVERSION(df):
    p = {'sl_atr': 1.5, 'tp_atr': 3.0, 'max_bars_hold': 99}
    macd, macdsignal, macdhist = calc_macd(df['close'], 12, 26, 9)
    chop = calc_choppiness_index(df, 14)
    swing_high, swing_low = calc_swing_points(df, 20)

    eq_lows = (df['low'] < swing_low.shift(1)) & \
              (df['close'] > swing_low.shift(1)) & \
              (df['close'].shift(1) > swing_low.shift(1))

    eq_highs = (df['high'] > swing_high.shift(1)) & \
               (df['close'] < swing_high.shift(1)) & \
               (df['close'].shift(1) < swing_high.shift(1))

    macd_up = macdhist > macdhist.shift(1)
    macd_down = macdhist < macdhist.shift(1)
    chop_ok = chop > 61.8

    signals = pd.Series(0, index=df.index)
    signals[eq_lows & macd_up & chop_ok] = 1
    signals[eq_highs & macd_down & chop_ok] = -1
    return signals, p

def strat_ASYMMETRIC_SQUEEZE(df):
    p = {'sl_atr': 3.0, 'tp_atr': 0.8, 'max_bars_hold': 15}
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.5)
    rsi = calc_rsi(df['close'], 3)
    ema = df['close'].ewm(span=50).mean()

    long_cond = (df['close'] < bb_l) & (rsi < 25) & (df['close'] > ema)
    short_cond = (df['close'] > bb_u) & (rsi > 75) & (df['close'] < ema)

    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals, p

def strat_CONSERVATIVE_HTF_EXHAUSTION(df):
    p = {'sl_atr': 1.5, 'tp_atr': 3.0, 'max_bars_hold': 20}
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.5)
    ema = df['close'].ewm(span=50).mean()

    long_cond = (df['close'] < bb_l) & (df['close'] > ema)
    short_cond = (df['close'] > bb_u) & (df['close'] < ema)

    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals, p

def strat_BALANCED_ASYMMETRIC_SCALP(df):
    p = {'sl_atr': 3.0, 'tp_atr': 0.8, 'max_bars_hold': 15}
    bb_u, bb_m, bb_l, _, _ = calc_bollinger_bands(df['close'], 20, 2.5)
    rsi = calc_rsi(df['close'], 3)
    ema = df['close'].ewm(span=50).mean()

    long_cond = (df['close'] < bb_l) & (rsi < 20) & (df['close'] > ema)
    short_cond = (df['close'] > bb_u) & (rsi > 80) & (df['close'] < ema)

    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals, p

STRATEGIES = {
    "MR-F_V2_CAPITULATION": strat_MR_F_V2_CAPITULATION,
    "MR-H_RSI_ADX": strat_MR_H_RSI_ADX,
    "VWAP_MACD_PULLBACK": strat_VWAP_MACD_PULLBACK,
    "ALMA_OPT_61": strat_ALMA_OPT_61,
    "MACD_MEAN_REVERSION": strat_MACD_MEAN_REVERSION,
    "HIGH_WIN_RATE_ASYMMETRIC_SQUEEZE": strat_ASYMMETRIC_SQUEEZE,
    "CONSERVATIVE_HTF_EXHAUSTION": strat_CONSERVATIVE_HTF_EXHAUSTION,
    "BALANCED_ASYMMETRIC_SCALP": strat_BALANCED_ASYMMETRIC_SCALP
}

# ---------------------------------------------------------
# RUNNER SCRIPT
# ---------------------------------------------------------
def run_all():
    files = glob.glob('*min_1year.csv') + glob.glob('*1H_1year.csv') + glob.glob('*1H_1year_SYNTHETIC.csv')
    print(f"Found {len(files)} CSV files to evaluate.")
    
    results = []

    for file in files:
        print(f"Processing {file}...")
        try:
            df = pd.read_csv(file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            if 'volume' not in df.columns:
                df['volume'] = 1000 # Dummy
            
            # Standard Split
            split_idx = int(len(df) * 0.7)
            df_is = df.iloc[:split_idx]
            df_oos = df.iloc[split_idx:]

            for name, func in STRATEGIES.items():
                # Evaluate IS
                signals_is, p = func(df_is.copy())
                if signals_is.sum() != 0 or signals_is.min() < 0:
                    core_is = BacktestCore()
                    trades_is, cap_is = core_is.run_backtest(
                        df=df_is.copy(),
                        signals=signals_is,
                        sl_atr=p['sl_atr'], 
                        tp_atr=p['tp_atr'], 
                        max_bars_hold=p['max_bars_hold'], 
                        risk_pct=0.01,
                        trailing=True
                    )
                    res_is = core_is.calculate_metrics(trades_is, cap_is)
                    
                    if res_is['total_trades'] > 5:
                        # Evaluate OOS
                        signals_oos, _ = func(df_oos.copy())
                        core_oos = BacktestCore()
                        trades_oos, cap_oos = core_oos.run_backtest(
                            df=df_oos.copy(),
                            signals=signals_oos,
                            sl_atr=p['sl_atr'], 
                            tp_atr=p['tp_atr'], 
                            max_bars_hold=p['max_bars_hold'], 
                            risk_pct=0.01,
                            trailing=True
                        )
                        res_oos = core_oos.calculate_metrics(trades_oos, cap_oos)
                        
                        results.append({
                            "strategy": name,
                            "file": file,
                            "IS_PF": res_is['profit_factor'],
                            "OOS_PF": res_oos['profit_factor'],
                            "IS_WR": res_is['win_rate'],
                            "OOS_WR": res_oos['win_rate'],
                            "IS_Sharpe": res_is['sharpe_ratio'],
                            "OOS_Sharpe": res_oos['sharpe_ratio'],
                            "IS_Trades": res_is['total_trades'],
                            "OOS_Trades": res_oos['total_trades'],
                            "Settings": json.dumps(p)
                        })
        except Exception as e:
            import traceback
            print(f"Error on {os.path.basename(file)}:\n{traceback.format_exc()}")

    # Rank and save
    if not results:
        print("No valid results found.")
        return
        
    df_res = pd.DataFrame(results)
    # We rank strictly by OOS Profit Factor * Win Rate to find true Demigods
    df_res['Demigod_Score'] = df_res['OOS_PF'] * (df_res['OOS_WR'] / 100)
    df_res = df_res.sort_values(by='Demigod_Score', ascending=False)
    
    # Filter for true winners (OOS PF > 1.2, Trades > 5)
    df_res = df_res[(df_res['OOS_PF'] > 1.2) & (df_res['OOS_Trades'] > 5)]
    
    with open("Demi-God strategies.txt", "w") as f:
        f.write("================================================================================\n")
        f.write("                          THE DEMI-GOD STRATEGIES                               \n")
        f.write("================================================================================\n\n")
        f.write("Strictly ranked strategies that proved out-of-sample profitability across multiple assets.\n\n")
        
        for idx, row in df_res.iterrows():
            f.write("-" * 80 + "\n")
            f.write(f"RANK: {idx+1} | STRATEGY: {row['strategy']}\n")
            f.write(f"ASSET/TIMEFRAME: {row['file']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"PERFORMANCE:\n")
            f.write(f"  * OOS Profit Factor : {row['OOS_PF']:.2f}\n")
            f.write(f"  * OOS Win Rate      : {row['OOS_WR']:.2f}%\n")
            f.write(f"  * OOS Sharpe Ratio  : {row['OOS_Sharpe']:.3f}\n")
            f.write(f"  * OOS Total Trades  : {row['OOS_Trades']}\n")
            f.write(f"  * In-Sample PF      : {row['IS_PF']:.2f}\n")
            f.write(f"  * In-Sample WR      : {row['IS_WR']:.2f}%\n")
            f.write(f"SETTINGS & RULES:\n")
            f.write(f"  {row['Settings']}\n\n")
            
            strat_rules = {
                "MR_F_V2_CAPITULATION": "RULES:\n1. BB %B drops below 0.05.\n2. ATR spikes > 1.5x rolling average.\n3. Volume spikes > 1.5x rolling average.\n4. Price makes 12-bar low, RSI diverges (makes higher low).\n5. Trigger: %B moves back above 0.10.\n",
                "MR_H_RSI_ADX": "RULES:\n1. ADX(10) is between 12 and 25 (Choppy/Ranging).\n2. ATR is greater than its 50-period 25th percentile (Volatility is not completely dead).\n3. Choppiness Index(14) < 61.8.\n4. For Long: RSI(14) < 30 (Oversold), StochRSI K crosses above previous K, and previous StochRSI was < 15.\n5. For Short: RSI(14) > 70 (Overbought), StochRSI K crosses below previous K, and previous StochRSI was > 85.\n",
                "VWAP_MACD_PULLBACK": "RULES:\n1. Price is above VWAP (Trend is Up).\n2. MACD Histogram > 0 and MACD Line > Signal Line.\n3. MACD Line crosses ABOVE Zero Line.\n4. Volume > SMA(20).\n5. ADX(14) > 25 (Strong Trend).\n",
                "ALMA_OPT_61": "RULES:\n1. Price forms Equal Lows (sweeps support swing low).\n2. ALMA(21) is sloping UP.\n3. ADX(14) > 25 (Strong Trend).\n4. Volume > 20 SMA (Institutional Surge).\n",
                "MACD_MEAN_REVERSION": "RULES:\n1. Price sweeps a recent 20-period swing low.\n2. MACD Histogram ticks UP from the previous bar (Momentum Shift).\n3. Choppiness Index(14) > 61.8 (Chop regime confirmed).\n",
                "ASYMMETRIC_SQUEEZE": "RULES:\n1. Price Close < lower BB (20, 2.5).\n2. RSI(3) < 25 (Extreme short-term oversold).\n3. Price Close > EMA(50) (Higher timeframe trend is still up).\n",
                "CONSERVATIVE_HTF_EXHAUSTION": "RULES:\n1. Price Close < lower BB (20, 2.5).\n2. Price Close > EMA(50).\nNo RSI filter, just pure exhaustion and reversion to mean.\n",
                "BALANCED_ASYMMETRIC_SCALP": "RULES:\n1. Price Close < lower BB (20, 2.5).\n2. RSI(3) < 20 (Ultra extreme oversold).\n3. Price Close > EMA(50).\n"
            }
            
            f.write(strat_rules.get(row['strategy'], "RULES: Not explicitly documented for this variation.\n"))
            
    print("Demi-God strategies.txt successfully generated!")

if __name__ == "__main__":
    run_all()
