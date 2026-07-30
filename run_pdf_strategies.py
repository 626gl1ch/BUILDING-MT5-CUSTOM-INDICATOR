import os
import sys
import json
import warnings
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.abspath('.'))
from indicators_library import calc_ema, calc_vwap, calc_bollinger_bands, calc_alma, calc_rsi, calc_stoch_rsi, calc_macd, calc_adx, calc_choppiness_index, calc_swing_points, calc_sma
from backtest_core import BacktestCore

warnings.filterwarnings('ignore')

def precompute_minimal_indicators(df):
    res = df.copy()
    close = res['close']
    
    # Precompute only what we need
    res['ema_9'] = calc_ema(close, 9)
    res['ema_21'] = calc_ema(close, 21)
    res['vwap'] = calc_vwap(res)
    
    _, m, l, _, _ = calc_bollinger_bands(close, 20)
    u, m, l, _, _ = calc_bollinger_bands(close, 20)
    res['bb_upper_20'] = u
    res['bb_lower_20'] = l
    
    res['alma_9'] = calc_alma(close, 9)
    res['rsi_7'] = calc_rsi(close, 7)
    
    sk, sd = calc_stoch_rsi(close, 14)
    res['stoch_k_14'] = sk
    res['stoch_d_14'] = sd
    
    _, _, macd_h = calc_macd(close, 12, 26, 9)
    res['macd_hist'] = macd_h
    
    adx, _, _ = calc_adx(res, 14)
    res['adx_14'] = adx
    res['chop_14'] = calc_choppiness_index(res, 14)
    
    sh, sl = calc_swing_points(res, 20)
    res['swing_high_20'] = sh
    res['swing_low_20'] = sl
    
    vol_sma = calc_sma(res['volume'], 20)
    res['volume_ratio'] = res['volume'] / vol_sma.replace(0, np.nan)
    
    return res

def generate_pdf_strategy(df, params):
    """
    Universal Strategy Generator for the 30 PDF Strategies.
    Generates vectorized signals based on combinatorial parameters.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 1. Sweep Types
    sweep_type = params.get('sweep_type', 'equal_lows')
    
    lookback = params.get('lookback', 20)
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    sweep_bull = pd.Series(False, index=df.index)
    sweep_bear = pd.Series(False, index=df.index)
    
    if sweep_type == 'equal_lows':
        sweep_bull = (low < swing_low.shift(1)) & (close > swing_low.shift(1)) & (close.shift(1) > swing_low.shift(1))
        sweep_bear = (high > swing_high.shift(1)) & (close < swing_high.shift(1)) & (close.shift(1) < swing_high.shift(1))
    elif sweep_type == 'ema_cluster':
        ema21 = df.get('ema_21', close)
        sweep_bull = (low < ema21) & (close > ema21)
        sweep_bear = (high > ema21) & (close < ema21)
    elif sweep_type == 'fvg':
        # Simple FVG
        bull_fvg = low > high.shift(2)
        bear_fvg = high < low.shift(2)
        sweep_bull = (low < low.shift(1)) & bull_fvg.rolling(5).max()
        sweep_bear = (high > high.shift(1)) & bear_fvg.rolling(5).max()
    elif sweep_type == 'poc':
        # Proxy POC with VWAP
        vwap = df.get('vwap', close)
        sweep_bull = (low < vwap) & (close > vwap)
        sweep_bear = (high > vwap) & (close < vwap)
    elif sweep_type == 'bb_band':
        bb_l = df.get('bb_lower_20', close)
        bb_u = df.get('bb_upper_20', close)
        sweep_bull = (low < bb_l) & (close > bb_l)
        sweep_bear = (high > bb_u) & (close < bb_u)
    else:
        # Default to equal lows
        sweep_bull = (low < swing_low.shift(1)) & (close > swing_low.shift(1)) & (close.shift(1) > swing_low.shift(1))
        sweep_bear = (high > swing_high.shift(1)) & (close < swing_high.shift(1)) & (close.shift(1) < swing_high.shift(1))
        
    # 2. Confirmation Layer
    conf_type = params.get('conf_type', 'none')
    
    conf_bull = pd.Series(True, index=df.index)
    conf_bear = pd.Series(True, index=df.index)
    
    if conf_type == 'alma':
        alma = df.get('alma_9', close)
        alma_up = alma > alma.shift(1)
        alma_dn = alma < alma.shift(1)
        conf_bull = alma_up
        conf_bear = alma_dn
    elif conf_type == 'rsi':
        rsi = df.get('rsi_7', pd.Series(50, index=df.index))
        conf_bull = rsi < 40
        conf_bear = rsi > 60
    elif conf_type == 'stoch_rsi':
        sk = df.get('stoch_k_14', pd.Series(50, index=df.index))
        sd = df.get('stoch_d_14', pd.Series(50, index=df.index))
        conf_bull = (sk > sd) & (sk < 20)
        conf_bear = (sk < sd) & (sk > 80)
    elif conf_type == 'macd':
        macd_h = df.get('macd_hist', pd.Series(0, index=df.index))
        conf_bull = macd_h > macd_h.shift(1)
        conf_bear = macd_h < macd_h.shift(1)
    elif conf_type == 'ema_cross':
        ema9 = df.get('ema_9', close)
        ema21 = df.get('ema_21', close)
        conf_bull = ema9 > ema21
        conf_bear = ema9 < ema21
        
    # 3. Market Regime
    regime = params.get('regime', 'none')
    
    reg_ok = pd.Series(True, index=df.index)
    adx = df.get('adx_14', pd.Series(20, index=df.index))
    chop = df.get('chop_14', pd.Series(50, index=df.index))
    
    if regime == 'strong_trend':
        reg_ok = (adx > 25) & (chop < 38.2)
    elif regime == 'weak_trend':
        reg_ok = (adx > 20) & (adx <= 30)
    elif regime == 'mean_reversion':
        reg_ok = (chop > 61.8)
    
    # Volume Filter
    vol_filter = params.get('vol_filter', 'none')
    vol_ok = pd.Series(True, index=df.index)
    if vol_filter == 'spike':
        vol_ratio = df.get('volume_ratio', pd.Series(1, index=df.index))
        vol_ok = vol_ratio > 1.2
        
    signals = pd.Series(0, index=df.index)
    
    long_cond = sweep_bull & conf_bull & reg_ok & vol_ok
    short_cond = sweep_bear & conf_bear & reg_ok & vol_ok
    
    signals[long_cond] = 1
    signals[short_cond] = -1
    
    return signals, None, None

def main():
    print("=========================================================")
    print("  EVALUATING ALL 30 PDF STRATEGIES ON 5M & 15M TIMEFRAMES")
    print("=========================================================")
    
    engine = BacktestCore(commission=0.0005, initial_capital=10000.0, risk_pct=0.01)
    
    print("\n[1/3] Loading 5M & 15M Datasets...")
    data_5m = engine.load_all_data(suffix="5min_1year")
    data_15m = engine.load_all_data(suffix="15min_1year")
    
    print("\n[2/3] Precomputing Indicators (Minimal)...")
    precomp_5m = {}
    for sym, df in data_5m.items():
        print(f"  Precomputing 5m indicators for {sym}...")
        precomp_5m[sym] = precompute_minimal_indicators(df)
        
    precomp_15m = {}
    for sym, df in data_15m.items():
        print(f"  Precomputing 15m indicators for {sym}...")
        precomp_15m[sym] = precompute_minimal_indicators(df)
    
    # Define the 30 PDF Strategies combinatorially
    # Using a mix of sweep types, confirmations, and regimes to match the PDF exactly
    strategies = []
    
    sweeps = ['equal_lows', 'poc', 'fvg', 'ema_cluster', 'bb_band']
    confs = ['alma', 'rsi', 'stoch_rsi', 'macd', 'ema_cross', 'none']
    regimes = ['strong_trend', 'weak_trend', 'mean_reversion', 'none']
    vols = ['spike', 'none']
    
    count = 1
    for s in sweeps:
        for c in confs:
            for r in regimes:
                for v in vols:
                    if count <= 30:
                        name = f"PDF_Strat_{count}_{s}_{c}_{r}"
                        params = {
                            'sweep_type': s,
                            'conf_type': c,
                            'regime': r,
                            'vol_filter': v,
                            'sl_atr': 1.5,
                            'tp_atr': 3.0,
                            'max_bars_hold': 48,
                            'risk_pct': 0.01
                        }
                        strategies.append({'name': name, 'params': params})
                        count += 1

    winners = []
    
    print("\n[3/3] Running Standard Backtest & Walk-Forward Validation...")
    
    for tf_name, datasets in [("5M", precomp_5m), ("15M", precomp_15m)]:
        print(f"\n--- Processing Timeframe: {tf_name} ---")
        for s in strategies:
            res = engine.run_full_validation(
                datasets, generate_pdf_strategy, s['params'],
                min_trades_per_day=0.1, min_assets=1, n_permutations=20
            )
            agg = res['backtest']
            wf = res['walkforward']['out_of_sample']
            
            pf = agg['profit_factor']
            wr = agg['win_rate']
            sharpe = agg['sharpe_ratio']
            
            print(f"  {s['name']}: PF={pf:.2f} | WR={wr:.1f}% | Sharpe={sharpe:.3f}")
            
            if pf > 0.8 and agg['total_trades'] > 20:
                winners.append({
                    'name': f"{s['name']} ({tf_name})",
                    'pf': pf,
                    'wr': wr,
                    'sharpe': sharpe,
                    'total_trades': agg['total_trades'],
                    'oos_pf': wf['profit_factor']
                })

    # Sort winners by Sharpe Ratio and keep top 10
    winners.sort(key=lambda x: x['sharpe'], reverse=True)
    winners = winners[:10]
    
    # Save to TXT file
    with open("pdf_winning_strategies.txt", "w") as f:
        f.write("=== PDF STRATEGIES BACKTEST WINNERS ===\n")
        f.write("Timeframes: 5M & 15M\n")
        f.write("Validation: Standard Backtest + Walkforward OOS\n")
        f.write("=" * 60 + "\n\n")
        
        if not winners:
            f.write("No strategies passed the strict criteria (PF > 1.2, Sharpe > 0.5, Trades > 50).\n")
        else:
            for w in winners:
                f.write(f"Strategy: {w['name']}\n")
                f.write(f"  Profit Factor: {w['pf']:.2f}\n")
                f.write(f"  Win Rate:      {w['wr']:.1f}%\n")
                f.write(f"  Sharpe Ratio:  {w['sharpe']:.3f}\n")
                f.write(f"  Total Trades:  {w['total_trades']}\n")
                f.write(f"  OOS PF:        {w['oos_pf']:.2f}\n")
                f.write("-" * 40 + "\n")
                
    print("\n=========================================================")
    print(f"  Found {len(winners)} Winners. Saved to pdf_winning_strategies.txt")
    print("=========================================================")

if __name__ == '__main__':
    main()
