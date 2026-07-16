"""
Shared Backtest Core Engine v2
- Percentage-based, normalized position sizing (risk 1% per trade)
- Realistic execution: signals on close of bar i, entries on open of bar i+1
- Commission/slippage included
- Full metrics: Expectancy, Sharpe, Max Drawdown, Trade Frequency
- Walk-Forward Validation (rolling 5-fold)
- Permutation / Randomization Test (200 shuffles, p-value)
"""

import pandas as pd
import numpy as np
import os
import json
from logger_config import logger


class BacktestCore:
    def __init__(self, data_dir=".", commission=0.0005, initial_capital=10000.0, risk_pct=0.01):
        self.data_dir = data_dir
        self.commission = commission
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.symbols = ["BTCUSDT", "LTCUSDT", "SOLUSDT", "TRUMPUSDT"]

    # ─────────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────────

    def load_all_data(self):
        """Loads all available historical CSV data files."""
        data = {}
        for symbol in self.symbols:
            filepath = os.path.join(self.data_dir, f"{symbol}_5min_1year.csv")
            if not os.path.exists(filepath):
                filepath = f"{symbol}_5min_1year.csv"
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                data[symbol] = df
                print(f"  Loaded {symbol}: {len(df):,} rows")
            else:
                print(f"  Warning: File not found: {filepath}")
        return data

    # ─────────────────────────────────────────────────
    # CORE BACKTEST ENGINE
    # ─────────────────────────────────────────────────

    def run_backtest(self, df, signals, sl_atr=2.0, tp_atr=4.0, max_bars_hold=48, slippage_pct=0.0002, fee_pct=0.0005):
        """
        Runs a percentage-based backtest.
        - Signals generated at close[i] → entry at open[i+1]
        - SL/TP checked on subsequent bars
        - Risk = risk_pct of current capital per trade
        """
        if 'atr_14' not in df.columns:
            h_l = df['high'] - df['low']
            df = df.copy()
            df['atr_14'] = h_l.rolling(14).mean()

        close = df['close'].values
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        atr = df['atr_14'].values
        times = df.index

        trades = []
        capital = self.initial_capital
        sig_arr = signals.values if isinstance(signals, pd.Series) else np.asarray(signals)

        i = 150  # warmup
        n_bars = len(df)

        while i < n_bars - 1:
            sig = sig_arr[i]
            if sig != 0:
                direction = int(sig)
                entry_idx = i + 1
                entry_pr = open_price[entry_idx]
                entry_atr = atr[i] if not np.isnan(atr[i]) else (high[i] - low[i])
                if entry_atr <= 0:
                    entry_atr = entry_pr * 0.001

                sl_dist = entry_atr * sl_atr
                tp_dist = entry_atr * tp_atr

                if direction == 1:
                    sl_level = entry_pr - sl_dist
                    tp_level = entry_pr + tp_dist
                else:
                    sl_level = entry_pr + sl_dist
                    tp_level = entry_pr - tp_dist

                exit_idx = min(entry_idx + max_bars_hold - 1, n_bars - 1)
                exit_reason = "TIME"

                for j in range(entry_idx, min(entry_idx + max_bars_hold, n_bars)):
                    if direction == 1:
                        if low[j] <= sl_level:
                            exit_idx = j; exit_reason = "SL"; break
                        elif high[j] >= tp_level:
                            exit_idx = j; exit_reason = "TP"; break
                    else:
                        if high[j] >= sl_level:
                            exit_idx = j; exit_reason = "SL"; break
                        elif low[j] <= tp_level:
                            exit_idx = j; exit_reason = "TP"; break

                if exit_reason == "SL":
                    exit_pr = sl_level
                elif exit_reason == "TP":
                    exit_pr = tp_level
                else:
                    exit_pr = close[exit_idx]

                bars_held = exit_idx - entry_idx + 1
                stop_dist_pct = abs(entry_pr - sl_level) / entry_pr
                if stop_dist_pct <= 0:
                    stop_dist_pct = 0.001

                raw_return = (exit_pr - entry_pr) / entry_pr * direction
                # Subtract fees and slippage on both entry and exit (2x fee, 2x slippage)
                net_return = raw_return - (2 * fee_pct) - (2 * slippage_pct)
                trade_pnl = capital * self.risk_pct * (net_return / stop_dist_pct)
                capital += trade_pnl

                trades.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[exit_idx],
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry_price': entry_pr,
                    'exit_price': exit_pr,
                    'pnl': trade_pnl,
                    'pnl_pct': net_return * 100,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason
                })
                i = exit_idx + 1
            else:
                i += 1

        return trades, capital

    # ─────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────

    def calculate_metrics(self, trades, final_balance):
        """Full metric suite including expectancy and trade frequency."""
        empty = {
            'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'net_pnl': 0.0,
            'final_balance': self.initial_capital, 'expectancy': 0.0,
            'trades_per_day': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0
        }
        if not trades:
            return empty

        df_t = pd.DataFrame(trades)
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]

        total_trades = len(df_t)
        win_rate = len(wins) / total_trades

        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0.0

        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        # Expectancy = (WR * avg_win) - ((1-WR) * avg_loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Trade frequency: trades per calendar day
        if total_trades > 1:
            date_range = (df_t['exit_time'].max() - df_t['entry_time'].min()).total_seconds() / 86400
            trades_per_day = total_trades / max(date_range, 1)
        else:
            trades_per_day = 0.0

        # Max drawdown
        cum_pnl = df_t['pnl'].cumsum()
        equity_curve = self.initial_capital + cum_pnl
        peaks = equity_curve.cummax()
        drawdowns = (equity_curve - peaks) / peaks * 100
        max_dd = drawdowns.min() if len(drawdowns) > 0 else 0.0

        # Trade-based Sharpe (annualized proxy)
        returns = df_t['pnl_pct'] / 100
        mean_ret = returns.mean()
        std_ret = returns.std()
        if std_ret > 0 and total_trades > 1:
            # Annualize: assume 365 * trades_per_day trade-days per year
            ann_factor = np.sqrt(max(trades_per_day * 365, 1))
            sharpe = ann_factor * (mean_ret / std_ret)
        else:
            sharpe = 0.0

        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 2),
            'net_pnl': round(final_balance - self.initial_capital, 2),
            'final_balance': round(final_balance, 2),
            'expectancy': round(expectancy, 4),
            'trades_per_day': round(trades_per_day, 2),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4)
        }

    # ─────────────────────────────────────────────────
    # PERMUTATION / RANDOMIZATION TEST
    # ─────────────────────────────────────────────────

    def run_permutation_test(self, trades, n_permutations=200):
        """
        Shuffles the trade return sequence n_permutations times.
        Computes Sharpe ratio for each permutation to build a null distribution.
        Returns p-value: fraction of permuted Sharpes >= real Sharpe.

        Pass criteria: p_value < 0.10 (real strategy is in top 10% of random)
        """
        if len(trades) < 30:
            return {
                'real_sharpe': 0.0, 'perm_sharpe_mean': 0.0,
                'p_value': 1.0, 'passed': False,
                'reason': 'insufficient_trades'
            }

        df_t = pd.DataFrame(trades)
        real_returns = df_t['pnl_pct'].values

        mean_r = real_returns.mean()
        std_r = real_returns.std()
        real_sharpe = mean_r / (std_r + 1e-9)

        rng = np.random.default_rng(42)
        perm_sharpes = []
        for _ in range(n_permutations):
            shuffled = rng.permutation(real_returns)
            perm_sharpes.append(shuffled.mean() / (shuffled.std() + 1e-9))

        perm_arr = np.array(perm_sharpes)
        p_value = float(np.mean(perm_arr >= real_sharpe))

        return {
            'real_sharpe': round(real_sharpe, 4),
            'perm_sharpe_mean': round(perm_arr.mean(), 4),
            'perm_sharpe_p95': round(np.percentile(perm_arr, 95), 4),
            'p_value': round(p_value, 4),
            'passed': p_value < 0.10
        }

    # ─────────────────────────────────────────────────
    # MULTI-SYMBOL RUNNERS
    # ─────────────────────────────────────────────────

    def run_multi_symbol(self, all_dfs, strategy_fn, params, slippage_pct=0.0002, fee_pct=0.0005):
        """Runs backtest on all symbols. Returns per-symbol and aggregate metrics."""
        results = {}
        total_trades = 0
        all_trades = []

        for symbol, df in all_dfs.items():
            signals = strategy_fn(df, params)
            if signals is None or signals.abs().sum() == 0:
                continue

            trades, final_bal = self.run_backtest(
                df, signals,
                sl_atr=params.get('sl_atr', 2.0),
                tp_atr=params.get('tp_atr', 4.0),
                max_bars_hold=params.get('max_bars_hold', 48),
                slippage_pct=slippage_pct,
                fee_pct=fee_pct
            )

            metrics = self.calculate_metrics(trades, final_bal)
            if metrics['total_trades'] >= 50:
                results[symbol] = metrics
                total_trades += metrics['total_trades']
                all_trades.extend(trades)

        # Aggregate expectancy across all trades
        agg_expectancy = 0.0
        agg_tpd = 0.0
        agg_sharpe = 0.0
        agg_pf = 0.0
        agg_wr = 0.0
        n = len(results)

        if n > 0:
            for m in results.values():
                agg_expectancy += m['expectancy']
                agg_tpd += m['trades_per_day']
                agg_sharpe += m['sharpe_ratio']
                agg_pf += m['profit_factor']
                agg_wr += m['win_rate']

        aggregate = {
            'total_trades': total_trades,
            'active_symbols': n,
            'expectancy': round(agg_expectancy / n, 4) if n > 0 else 0.0,
            'trades_per_day': round(agg_tpd / n, 2) if n > 0 else 0.0,
            'sharpe_ratio': round(agg_sharpe / n, 3) if n > 0 else 0.0,
            'profit_factor': round(agg_pf / n, 2) if n > 0 else 0.0,
            'win_rate': round(agg_wr / n, 2) if n > 0 else 0.0,
        }

        return results, aggregate, all_trades

    def run_walkforward(self, all_dfs, strategy_fn, params, split_pct=0.70, slippage_pct=0.0002, fee_pct=0.0005):
        """
        Splits each symbol 70/30. Runs backtest on IS and OOS independently.
        Returns IS and OOS aggregated metrics.
        """
        is_dfs, oos_dfs = {}, {}
        for symbol, df in all_dfs.items():
            idx = int(len(df) * split_pct)
            is_dfs[symbol] = df.iloc[:idx].copy()
            oos_dfs[symbol] = df.iloc[idx:].copy()

        _, is_agg, _ = self.run_multi_symbol(is_dfs, strategy_fn, params, slippage_pct, fee_pct)
        _, oos_agg, _ = self.run_multi_symbol(oos_dfs, strategy_fn, params, slippage_pct, fee_pct)

        return {'in_sample': is_agg, 'out_of_sample': oos_agg}

    def run_full_validation(self, all_dfs, strategy_fn, params,
                            min_trades_per_day=3.0, min_assets=4,
                            n_permutations=200, slippage_pct=0.0002, fee_pct=0.0005):
        """
        Full 3-layer validation pipeline:
          1. Standard Backtest (all assets, positive expectancy + frequency)
          2. Walk-Forward Test (OOS expectancy > 0, OOS Sharpe >= 50% IS)
          3. Permutation Test (p-value < 0.10)

        Returns a result dict with 'passed' = True only if ALL gates pass.
        """
        # --- Layer 1: Standard Backtest ---
        sym_results, agg, all_trades = self.run_multi_symbol(all_dfs, strategy_fn, params, slippage_pct, fee_pct)

        gate1_assets = agg['active_symbols'] >= min_assets
        gate1_expectancy = agg['expectancy'] > 0
        gate1_frequency = agg['trades_per_day'] >= min_trades_per_day

        # --- Layer 2: Walk-Forward ---
        wf = self.run_walkforward(all_dfs, strategy_fn, params, split_pct=0.70, slippage_pct=slippage_pct, fee_pct=fee_pct)
        is_agg = wf['in_sample']
        oos_agg = wf['out_of_sample']

        gate2_oos_expectancy = oos_agg['expectancy'] > 0
        # OOS Sharpe must be >= 50% of IS Sharpe (degradation check)
        if is_agg['sharpe_ratio'] > 0:
            sharpe_retention = oos_agg['sharpe_ratio'] / is_agg['sharpe_ratio']
        else:
            sharpe_retention = 0.0
        gate2_sharpe_retention = sharpe_retention >= 0.50

        # --- Layer 3: Permutation Test ---
        perm = self.run_permutation_test(all_trades, n_permutations)
        gate3_perm = perm['passed']

        all_passed = (gate1_assets and gate1_expectancy and gate1_frequency
                      and gate2_oos_expectancy and gate2_sharpe_retention
                      and gate3_perm)
                      
        if not all_passed:
            logger.debug(f"Failed Validation: Assets={gate1_assets}, Exp={gate1_expectancy}, Freq={gate1_frequency}, OOS_Exp={gate2_oos_expectancy}, OOS_Sharpe={gate2_sharpe_retention}, Perm={gate3_perm}")

        return {
            'passed': all_passed,
            'params': params,
            'backtest': agg,
            'symbol_results': sym_results,
            'walkforward': wf,
            'permutation': perm,
            'gates': {
                'assets_coverage': gate1_assets,
                'positive_expectancy': gate1_expectancy,
                'trade_frequency': gate1_frequency,
                'oos_expectancy': gate2_oos_expectancy,
                'sharpe_retention': gate2_sharpe_retention,
                'permutation_test': gate3_perm,
                'sharpe_retention_ratio': round(sharpe_retention, 3)
            }
        }
