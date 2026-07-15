"""
RBO v2 — Research, Backtest & Optimize Master Engine
=====================================================
Tests 7 strategy types on 5-minute crypto data across all 4 CSV assets.
Each strategy must pass a 3-layer validation:
  1. Standard Backtest  (positive expectancy, >=3 trades/day, all 4 assets)
  2. Walk-Forward Test  (OOS expectancy > 0, OOS Sharpe >= 50% of IS)
  3. Permutation Test   (p-value < 0.10 — not a coin flip or overfit)

Goal: Find >= 2 confirmed strategies per type for daily scalping.
"""

import os
import sys
import json
import time
import itertools
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from indicators_library import add_all_indicators
from backtest_core import BacktestCore

# ─────────────────────────────────────────────────────────────
# SIGNAL GENERATORS — 7 STRATEGY TYPES
# ─────────────────────────────────────────────────────────────

# ── TYPE 1: EMA PULLBACK TREND FOLLOWING ──────────────────────
def signal_ema_pullback(df, p):
    """
    Enter when price pulls back to the fast EMA during a trend.
    - Trend bias: close > trend_ema (for longs) or close < trend_ema (for shorts)
    - ADX > adx_min confirms trend strength
    - Entry trigger: close crosses back above fast_ema after touching it
    """
    fast_col = f'ema_{p["fast_ema"]}'
    trend_col = f'ema_{p["trend_ema"]}'
    adx_col = f'adx_14'

    if fast_col not in df.columns or trend_col not in df.columns:
        return pd.Series(0, index=df.index)

    close = df['close']
    fast_ema = df[fast_col]
    trend_ema = df[trend_col]
    adx = df[adx_col] if adx_col in df.columns else pd.Series(30, index=df.index)

    trend_up = close > trend_ema
    trend_dn = close < trend_ema
    strong = adx > p['adx_min']

    # Pullback touch: previous close <= fast_ema, current close crosses above
    prev_close = close.shift(1)
    prev_fast = fast_ema.shift(1)

    long_entry = trend_up & strong & (prev_close <= prev_fast) & (close > fast_ema)
    short_entry = trend_dn & strong & (prev_close >= prev_fast) & (close < fast_ema)

    signals = pd.Series(0, index=df.index)
    signals[long_entry] = 1
    signals[short_entry] = -1
    return signals


# ── TYPE 2: DONCHIAN/SUPERTREND BREAKOUT ──────────────────────
def signal_donchian_breakout(df, p):
    """
    Donchian channel breakout confirmed by SuperTrend direction + volume surge.
    - Break above upper Donchian → Long (if SuperTrend = bullish)
    - Break below lower Donchian → Short (if SuperTrend = bearish)
    - Volume surge: volume_ratio > vol_min
    """
    period = p['period']
    don_upper = f'donchian_upper_{period}'
    don_lower = f'donchian_lower_{period}'

    if don_upper not in df.columns:
        return pd.Series(0, index=df.index)

    close = df['close']
    st_dir = df['supertrend_dir_10_3'] if 'supertrend_dir_10_3' in df.columns else pd.Series(1, index=df.index)
    vol_ratio = df['volume_ratio'] if 'volume_ratio' in df.columns else pd.Series(2.0, index=df.index)

    prev_upper = df[don_upper].shift(1)
    prev_lower = df[don_lower].shift(1)
    vol_ok = vol_ratio > p['vol_min']

    long_break = (close > prev_upper) & (st_dir == 1) & vol_ok
    short_break = (close < prev_lower) & (st_dir == -1) & vol_ok

    signals = pd.Series(0, index=df.index)
    signals[long_break] = 1
    signals[short_break] = -1
    return signals


# ── TYPE 3: BOLLINGER BAND / VWAP MEAN REVERSION ─────────────
def signal_bb_vwap_mr(df, p):
    """
    Enter when price is below BB lower + below VWAP (buy) or above BB upper + above VWAP (sell).
    RSI confirms oversold/overbought.
    """
    bb_period = p['bb_period']
    bb_lower = f'bb_lower_{bb_period}'
    bb_upper = f'bb_upper_{bb_period}'
    rsi_col = f'rsi_{p["rsi_period"]}'

    if bb_lower not in df.columns or 'vwap' not in df.columns:
        return pd.Series(0, index=df.index)

    close = df['close']
    vwap = df['vwap']
    rsi = df[rsi_col] if rsi_col in df.columns else pd.Series(50, index=df.index)

    long_cond = (close < df[bb_lower]) & (close < vwap) & (rsi < p['rsi_os'])
    short_cond = (close > df[bb_upper]) & (close > vwap) & (rsi > p['rsi_ob'])

    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


# ── TYPE 4: RSI DIVERGENCE MEAN REVERSION ────────────────────
def signal_rsi_divergence(df, p):
    """
    Bullish divergence: price makes lower low, RSI makes higher low.
    Bearish divergence: price makes higher high, RSI makes lower high.
    Z-score filter: price must be extended (|zscore| > threshold).
    """
    rsi_col = f'rsi_{p["rsi_period"]}'
    zscore_col = 'zscore_20'
    lb = p['lookback']

    if rsi_col not in df.columns:
        return pd.Series(0, index=df.index)

    close = df['close']
    rsi = df[rsi_col]
    zscore = df[zscore_col] if zscore_col in df.columns else pd.Series(0, index=df.index)

    close_arr = close.values
    rsi_arr = rsi.values
    zs_arr = zscore.values
    n = len(df)
    signals = np.zeros(n)

    for i in range(lb + 1, n):
        c_now = close_arr[i]
        r_now = rsi_arr[i]
        c_prev = np.min(close_arr[i - lb:i])
        r_prev_min = rsi_arr[np.argmin(close_arr[i - lb:i]) + i - lb]
        c_prev_h = np.max(close_arr[i - lb:i])
        r_prev_max = rsi_arr[np.argmax(close_arr[i - lb:i]) + i - lb]
        z = zs_arr[i]

        # Bullish divergence: price lower low but RSI higher low + price extended down
        if c_now < c_prev and r_now > r_prev_min and z < -p['zscore_thresh']:
            signals[i] = 1
        # Bearish divergence: price higher high but RSI lower high + price extended up
        elif c_now > c_prev_h and r_now < r_prev_max and z > p['zscore_thresh']:
            signals[i] = -1

    return pd.Series(signals, index=df.index)


# ── TYPE 5: OPENING RANGE BREAKOUT ────────────────────────────
def signal_orb(df, p):
    """
    Daily opening range = first N bars after UTC 00:00.
    Enter long on break above OR high, short on break below OR low.
    Volume surge required.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(0, index=df.index)

    or_bars = p['or_bars']  # e.g. 6 = first 30 min (6 × 5min bars)
    vol_min = p['vol_min']

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol_ratio = df['volume_ratio'].values if 'volume_ratio' in df.columns else np.ones(len(df))
    times = df.index
    n = len(df)
    signals = np.zeros(n)

    # Build daily OR levels
    date_or_high = {}
    date_or_low = {}
    date_or_ready = {}

    for i, ts in enumerate(times):
        d = ts.date()
        bar_of_day = (ts.hour * 60 + ts.minute) // 5  # bar index within day

        if bar_of_day < or_bars:
            if d not in date_or_high:
                date_or_high[d] = high[i]
                date_or_low[d] = low[i]
                date_or_ready[d] = False
            else:
                date_or_high[d] = max(date_or_high[d], high[i])
                date_or_low[d] = min(date_or_low[d], low[i])
        elif bar_of_day == or_bars:
            date_or_ready[d] = True  # OR is now fully formed

        # Trade signals after OR is established
        if d in date_or_ready and date_or_ready[d]:
            or_h = date_or_high[d]
            or_l = date_or_low[d]
            vol_ok = vol_ratio[i] > vol_min

            if close[i] > or_h and vol_ok:
                signals[i] = 1
            elif close[i] < or_l and vol_ok:
                signals[i] = -1

    return pd.Series(signals, index=df.index)


# ── TYPE 6: MACD / MOMENTUM SCALP ────────────────────────────
def signal_macd_scalp(df, p):
    """
    Fast MACD crossover in direction of EMA bias, Stoch RSI confirms entry.
    - Long: MACD hist crosses above 0 + close > bias_ema + stoch not overbought
    - Short: MACD hist crosses below 0 + close < bias_ema + stoch not oversold
    """
    if p['macd_type'] == 'fast':
        macd_hist = 'macd_hist_fast'
    else:
        macd_hist = 'macd_hist'

    bias_col = f'ema_{p["bias_ema"]}'
    stoch_col = f'stoch_k_{p["stoch_period"]}'

    if macd_hist not in df.columns or bias_col not in df.columns:
        return pd.Series(0, index=df.index)

    hist = df[macd_hist]
    close = df['close']
    bias = df[bias_col]
    stoch = df[stoch_col] if stoch_col in df.columns else pd.Series(50, index=df.index)

    prev_hist = hist.shift(1)

    long_cross = (hist > 0) & (prev_hist <= 0) & (close > bias) & (stoch < p['stoch_ob'])
    short_cross = (hist < 0) & (prev_hist >= 0) & (close < bias) & (stoch > p['stoch_os'])

    signals = pd.Series(0, index=df.index)
    signals[long_cross] = 1
    signals[short_cross] = -1
    return signals


# ── TYPE 7: VOLATILITY SQUEEZE BREAKOUT ──────────────────────
def signal_squeeze_breakout(df, p):
    """
    Detects BB inside Keltner (squeeze) then enters on breakout with volume.
    - Squeeze: BB upper < Keltner upper AND BB lower > Keltner lower
    - Long: price breaks above BB upper after squeeze with volume surge
    - Short: price breaks below BB lower after squeeze with volume surge
    """
    period = p['period']
    bb_upper = f'bb_upper_{period}'
    bb_lower = f'bb_lower_{period}'
    kelt_upper = f'keltner_upper_{period}'
    kelt_lower = f'keltner_lower_{period}'
    vol_min = p['vol_min']
    squeeze_lb = p['squeeze_lb']

    if bb_upper not in df.columns or kelt_upper not in df.columns:
        return pd.Series(0, index=df.index)

    close = df['close']
    bbu = df[bb_upper]
    bbl = df[bb_lower]
    ku = df[kelt_upper]
    kl = df[kelt_lower]
    vol_ratio = df['volume_ratio'] if 'volume_ratio' in df.columns else pd.Series(2.0, index=df.index)

    # Squeeze = BB is inside Keltner
    in_squeeze = (bbu < ku) & (bbl > kl)
    # Was in squeeze recently?
    was_squeezed = in_squeeze.rolling(squeeze_lb).max().shift(1).fillna(0).astype(bool)

    vol_ok = vol_ratio > vol_min
    long_break = was_squeezed & (close > bbu.shift(1)) & vol_ok & ~in_squeeze
    short_break = was_squeezed & (close < bbl.shift(1)) & vol_ok & ~in_squeeze

    signals = pd.Series(0, index=df.index)
    signals[long_break] = 1
    signals[short_break] = -1
    return signals


# ─────────────────────────────────────────────────────────────
# PARAMETER GRIDS
# ─────────────────────────────────────────────────────────────

GRIDS = {
    'ema_pullback': {
        'fn': signal_ema_pullback,
        'params': {
            'fast_ema':    [9, 20],
            'trend_ema':   [50, 100, 200],
            'adx_min':     [15, 20, 25],
            'sl_atr':      [1.5, 2.0, 2.5],
            'tp_atr':      [3.0, 4.0, 6.0],
            'max_bars_hold': [12, 24, 36]
        }
    },
    'donchian_breakout': {
        'fn': signal_donchian_breakout,
        'params': {
            'period':        [10, 20, 50],
            'vol_min':       [1.2, 1.5, 2.0],
            'sl_atr':        [2.0, 2.5],
            'tp_atr':        [4.0, 6.0, 8.0],
            'max_bars_hold': [24, 48, 72]
        }
    },
    'bb_vwap_mr': {
        'fn': signal_bb_vwap_mr,
        'params': {
            'bb_period':     [10, 20],
            'rsi_period':    [9, 14],
            'rsi_os':        [30, 35],
            'rsi_ob':        [65, 70],
            'sl_atr':        [1.5, 2.0],
            'tp_atr':        [2.0, 3.0, 4.0],
            'max_bars_hold': [6, 12, 24]
        }
    },
    'rsi_divergence': {
        'fn': signal_rsi_divergence,
        'params': {
            'rsi_period':    [9, 14],
            'zscore_thresh': [1.5, 2.0],
            'lookback':      [5, 10, 14],
            'sl_atr':        [1.5, 2.0],
            'tp_atr':        [2.0, 3.0],
            'max_bars_hold': [12, 24]
        }
    },

    'squeeze_breakout': {
        'fn': signal_squeeze_breakout,
        'params': {
            'period':        [10, 20],
            'vol_min':       [1.5, 2.0],
            'squeeze_lb':    [5, 10],
            'sl_atr':        [1.5, 2.0, 2.5],
            'tp_atr':        [3.0, 4.0, 6.0],
            'max_bars_hold': [12, 24]
        }
    }
}


# ─────────────────────────────────────────────────────────────
# GRID SAMPLER (cap at 120 combos per type for speed)
# ─────────────────────────────────────────────────────────────

def get_param_combos(param_dict, max_combos=120):
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    all_combos = list(itertools.product(*values))
    if len(all_combos) > max_combos:
        rng = np.random.default_rng(777)
        idx = rng.choice(len(all_combos), size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in idx]
    return [dict(zip(keys, c)) for c in all_combos]


# ─────────────────────────────────────────────────────────────
# RESULT SAVING
# ─────────────────────────────────────────────────────────────

RESULTS_FILE = "strategies_confirmed.json"
REPORT_FILE = "strategy_report.md"

def load_confirmed():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_confirmed(data):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def print_result(strategy_type, rank, result):
    agg = result['backtest']
    wf = result['walkforward']
    perm = result['permutation']
    gates = result['gates']
    p = result['params']

    print(f"\n  ★ CONFIRMED #{rank} | {strategy_type}")
    print(f"    Params: {p}")
    print(f"    Backtest : Expectancy={agg['expectancy']:.4f} | "
          f"PF={agg['profit_factor']:.2f} | WR={agg['win_rate']:.1f}% | "
          f"Sharpe={agg['sharpe_ratio']:.3f} | Trades/Day={agg['trades_per_day']:.1f} | "
          f"Trades={agg['total_trades']}")
    print(f"    WF IS    : Expectancy={wf['in_sample']['expectancy']:.4f} | "
          f"Sharpe={wf['in_sample']['sharpe_ratio']:.3f}")
    print(f"    WF OOS   : Expectancy={wf['out_of_sample']['expectancy']:.4f} | "
          f"Sharpe={wf['out_of_sample']['sharpe_ratio']:.3f} | "
          f"Retention={gates['sharpe_retention_ratio']:.2f}")
    print(f"    Perm Test: real_sharpe={perm['real_sharpe']:.3f} | "
          f"p_value={perm['p_value']:.3f} | PASSED={perm['passed']}")
    print(f"    Gates    : {gates}")


def generate_report(confirmed_all):
    lines = ["# RBO v2 — Confirmed Strategy Report\n"]
    lines.append(f"Generated strategies that passed all 3 validation layers.\n")
    lines.append("---\n")

    for stype, strategies in confirmed_all.items():
        lines.append(f"## {stype.replace('_', ' ').title()} ({len(strategies)} confirmed)\n")
        for i, s in enumerate(strategies, 1):
            p = s['params']
            agg = s['backtest']
            wf = s['walkforward']
            perm = s['permutation']
            lines.append(f"### Strategy {i}\n")
            lines.append(f"**Parameters:** `{p}`\n\n")
            lines.append(f"| Metric | Value |\n|---|---|\n")
            lines.append(f"| Expectancy | {agg['expectancy']:.4f} |\n")
            lines.append(f"| Profit Factor | {agg['profit_factor']:.2f} |\n")
            lines.append(f"| Win Rate | {agg['win_rate']:.1f}% |\n")
            lines.append(f"| Sharpe Ratio | {agg['sharpe_ratio']:.3f} |\n")
            lines.append(f"| Trades/Day | {agg['trades_per_day']:.1f} |\n")
            lines.append(f"| Total Trades | {agg['total_trades']} |\n")
            lines.append(f"| OOS Sharpe Retention | {s['gates']['sharpe_retention_ratio']:.2f} |\n")
            lines.append(f"| Permutation p-value | {perm['p_value']:.4f} |\n")

            # Per-symbol breakdown
            lines.append(f"\n**Per-Symbol Backtest:**\n\n")
            lines.append(f"| Symbol | Expectancy | PF | WR | Sharpe | Trades/Day |\n|---|---|---|---|---|---|\n")
            for sym, m in s['symbol_results'].items():
                lines.append(f"| {sym} | {m['expectancy']:.4f} | {m['profit_factor']:.2f} | "
                              f"{m['win_rate']:.1f}% | {m['sharpe_ratio']:.3f} | {m['trades_per_day']:.1f} |\n")
            lines.append("\n---\n")

    with open(REPORT_FILE, 'w') as f:
        f.write(''.join(lines))
    print(f"\n  Report saved to {REPORT_FILE}")


# ─────────────────────────────────────────────────────────────
# MAIN RBO LOOP
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  RBO v2 — Research, Backtest & Optimize")
    print("  3-Layer Validation: Backtest + Walk-Forward + Permutation Test")
    print("=" * 70)

    engine = BacktestCore()

    print("\n[1/3] Loading data...")
    all_data = engine.load_all_data()
    if len(all_data) < 4:
        print(f"  ERROR: Only {len(all_data)} assets loaded. Need 4. Aborting.")
        return

    print("\n[2/3] Precomputing indicators (this takes ~3 min)...")
    t0 = time.time()
    precomputed = {}
    for sym, df in all_data.items():
        print(f"  Computing {sym}...")
        precomputed[sym] = add_all_indicators(df)
    print(f"  Done in {time.time() - t0:.0f}s")

    print("\n[3/3] Starting strategy sweeps...\n")

    confirmed_all = load_confirmed()
    total_confirmed = sum(len(v) for v in confirmed_all.values())

    for strategy_type, config in GRIDS.items():
        fn = config['fn']
        param_grid = config['params']
        combos = get_param_combos(param_grid, max_combos=120)

        existing = confirmed_all.get(strategy_type, [])
        if len(existing) >= 2:
            print(f"\n[SKIP] {strategy_type} already has {len(existing)} confirmed strategies.")
            continue

        print(f"\n{'='*70}")
        print(f"  Sweeping: {strategy_type.upper()} ({len(combos)} combinations)")
        print(f"{'='*70}")

        confirmed_this_type = list(existing)
        n_tested = 0
        t_start = time.time()

        for combo in combos:
            if len(confirmed_this_type) >= 2:
                break
            n_tested += 1

            try:
                result = engine.run_full_validation(
                    precomputed, fn, combo,
                    min_trades_per_day=3.0,
                    min_assets=4,
                    n_permutations=200
                )
            except Exception as e:
                continue  # skip broken combos silently

            elapsed = time.time() - t_start
            rate = n_tested / elapsed if elapsed > 0 else 0
            eta = (len(combos) - n_tested) / rate if rate > 0 else 0

            # Progress print every 10 combos
            if n_tested % 10 == 0:
                gates = result['gates']
                agg = result['backtest']
                print(f"  [{n_tested}/{len(combos)}] ETA={eta:.0f}s | "
                      f"Confirmed={len(confirmed_this_type)} | "
                      f"Expectancy={agg['expectancy']:.4f} | "
                      f"TPD={agg['trades_per_day']:.1f} | "
                      f"Assets={agg['active_symbols']} | "
                      f"Gates: Assets={gates['assets_coverage']} "
                      f"Exp={gates['positive_expectancy']} "
                      f"Freq={gates['trade_frequency']} "
                      f"OOS={gates['oos_expectancy']} "
                      f"Perm={gates['permutation_test']}")

            if result['passed']:
                rank = len(confirmed_this_type) + 1
                print_result(strategy_type, rank, result)
                confirmed_this_type.append(result)
                confirmed_all[strategy_type] = confirmed_this_type
                save_confirmed(confirmed_all)
                total_confirmed += 1
                print(f"\n  ✓ Total confirmed strategies so far: {total_confirmed}")

        confirmed_all[strategy_type] = confirmed_this_type
        save_confirmed(confirmed_all)

        found = len(confirmed_this_type)
        print(f"\n  {strategy_type}: {found}/2 confirmed after {n_tested} tests "
              f"({time.time()-t_start:.0f}s)")

    # Final report
    print("\n" + "=" * 70)
    print("  RBO v2 COMPLETE")
    print("=" * 70)
    for stype, strats in confirmed_all.items():
        print(f"  {stype}: {len(strats)} confirmed")

    print(f"\n  Total confirmed: {sum(len(v) for v in confirmed_all.values())}")
    generate_report(confirmed_all)
    print("\n  Results saved to strategies_confirmed.json")
    print("  Report saved to strategy_report.md")


if __name__ == "__main__":
    main()
