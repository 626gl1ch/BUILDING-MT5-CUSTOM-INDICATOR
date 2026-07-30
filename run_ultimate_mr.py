import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
from pathlib import Path
import json
from datetime import datetime
import warnings
import glob
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

class TimeFrame(Enum):
    M5 = "5m"
    M15 = "15m"

@dataclass
class StrategyConfig:
    name: str = "Default"
    description: str = ""
    timeframe: TimeFrame = TimeFrame.M5

    # Risk Management
    risk_per_trade: float = 0.01
    max_spread_pips: float = 2.0

    # ATR-based Stops & Targets - TIGHT for scalping
    atr_period: int = 14
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 2.5
    atr_min_threshold: float = 0.0003
    atr_max_threshold: float = 0.0030

    # Market Regime Filters
    adx_period: int = 14
    adx_max: float = 25.0
    choppiness_period: int = 14
    choppiness_min: float = 50.0
    choppiness_max: float = 80.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    bb_width_min: float = 0.0005
    bb_width_max: float = 0.0050

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_extreme_ob: float = 80.0
    rsi_extreme_os: float = 20.0

    # StochRSI
    stochrsi_period: int = 14
    stochrsi_k: int = 3
    stochrsi_d: int = 3
    stochrsi_overbought: float = 80.0
    stochrsi_oversold: float = 20.0

    # ALMA
    alma_period: int = 9
    alma_offset: float = 0.85
    alma_sigma: float = 6.0

    # EMA
    ema_fast: int = 9
    ema_slow: int = 21

    # MACD - Faster for 5m/15m
    macd_fast: int = 8
    macd_slow: int = 21
    macd_signal: int = 5

    # Volume Profile
    vp_lookback: int = 100
    vp_row_size: int = 24

    # Exit conditions
    time_exit_bars: int = 8
    trailing_stop_atr_mult: float = 0.0
    use_breakeven: bool = True
    breakeven_atr_mult: float = 0.8

    # Spread/pip value (auto-detected)
    pip_value: float = 0.0001

    def copy(self):
        import copy
        return copy.deepcopy(self)


# ==============================================================================
# TECHNICAL INDICATORS - Optimized for 5m/15m Noise
# ==============================================================================

class Indicators:
    @staticmethod
    def alma(series, period=9, offset=0.85, sigma=6.0):
        m = np.floor(offset * (period - 1))
        s = period / sigma
        weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s ** 2))
        weights = weights / weights.sum()
        alma = pd.Series(index=series.index, dtype=float)
        for i in range(period - 1, len(series)):
            alma.iloc[i] = np.dot(series.iloc[i - period + 1:i + 1].values, weights)
        return alma

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def stochrsi(series, period=14, k=3, d=3):
        rsi_val = Indicators.rsi(series, period)
        rsi_min = rsi_val.rolling(window=period).min()
        rsi_max = rsi_val.rolling(window=period).max()
        stoch = 100 * (rsi_val - rsi_min) / (rsi_max - rsi_min)
        stoch = stoch.fillna(50)
        k_line = stoch.rolling(window=k).mean()
        d_line = k_line.rolling(window=d).mean()
        return k_line, d_line

    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2.0):
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def choppiness_index(high, low, close, period=14):
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr_sum = tr.rolling(window=period).sum()
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        ci = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        return ci

    @staticmethod
    def bb_width(upper, lower, middle):
        return (upper - lower) / middle

    @staticmethod
    def bb_position(close, upper, lower):
        return (close - lower) / (upper - lower)

    @staticmethod
    def volume_profile(close, volume, lookback=100, row_size=24):
        recent_close = close.iloc[-lookback:]
        recent_vol = volume.iloc[-lookback:] if volume is not None else pd.Series(1, index=recent_close.index)
        price_min = recent_close.min()
        price_max = recent_close.max()
        bin_size = (price_max - price_min) / max(row_size, 1)
        if bin_size == 0:
            return {'poc': close.iloc[-1], 'hvn': [close.iloc[-1]], 'val': close.iloc[-1], 'vah': close.iloc[-1]}

        volume_profile = {}
        for i in range(row_size):
            bin_low = price_min + i * bin_size
            bin_high = price_min + (i + 1) * bin_size
            mask = (recent_close >= bin_low) & (recent_close < bin_high)
            volume_profile[(bin_low + bin_high) / 2] = recent_vol[mask].sum()

        poc_price = max(volume_profile, key=volume_profile.get)
        sorted_vp = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        threshold = sorted_vp[0][1] * 0.7 if sorted_vp else 0
        hvn = [price for price, vol in sorted_vp if vol >= threshold]

        total_vol = sum(volume_profile.values())
        cumsum = 0
        val = price_min
        vah = price_max
        for price, vol in sorted(volume_profile.items()):
            cumsum += vol
            if cumsum >= total_vol * 0.15 and val == price_min:
                val = price
            if cumsum >= total_vol * 0.85:
                vah = price
                break

        return {'poc': poc_price, 'hvn': hvn, 'val': val, 'vah': vah}


# ==============================================================================
# MARKET REGIME DETECTOR
# ==============================================================================

class MarketRegime:
    @staticmethod
    def is_tradeable(df, config, idx):
        if idx < 50:
            return False, "Insufficient data"

        adx_val = df['adx'].iloc[idx]
        if adx_val > config.adx_max:
            return False, f"Strong trend (ADX={adx_val:.1f})"

        ci_val = df['choppiness'].iloc[idx]
        if ci_val < config.choppiness_min:
            return False, f"Trending (CI={ci_val:.1f})"
        if ci_val > config.choppiness_max:
            return False, f"Extremely choppy (CI={ci_val:.1f})"

        atr_val = df['atr'].iloc[idx]
        if atr_val < config.atr_min_threshold:
            return False, f"Low volatility (ATR={atr_val:.5f})"
        if atr_val > config.atr_max_threshold:
            return False, f"High volatility/news (ATR={atr_val:.5f})"

        bbw = df['bb_width'].iloc[idx]
        if bbw < config.bb_width_min:
            return False, f"BB compression (Width={bbw:.5f})"
        if bbw > config.bb_width_max:
            return False, f"BB expansion (Width={bbw:.5f})"

        return True, "OK"


# ==============================================================================
# TRADE & BACKTEST ENGINE
# ==============================================================================

@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    size: float
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []

    def prepare_data(self, df):
        df = df.copy()
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if 'volume' not in df.columns:
            df['volume'] = 1000

        avg_price = df['close'].mean()
        if avg_price < 10:
            self.config.pip_value = 0.01
        else:
            self.config.pip_value = 0.0001

        df['alma'] = Indicators.alma(df['close'], self.config.alma_period, self.config.alma_offset, self.config.alma_sigma)
        df['ema9'] = Indicators.ema(df['close'], self.config.ema_fast)
        df['ema21'] = Indicators.ema(df['close'], self.config.ema_slow)
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], self.config.atr_period)
        df['adx'], df['plus_di'], df['minus_di'] = Indicators.adx(df['high'], df['low'], df['close'], self.config.adx_period)
        df['rsi'] = Indicators.rsi(df['close'], self.config.rsi_period)
        df['rsi2'] = Indicators.rsi(df['close'], 2)
        df['stoch_k'], df['stoch_d'] = Indicators.stochrsi(df['close'], self.config.stochrsi_period, self.config.stochrsi_k, self.config.stochrsi_d)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = Indicators.bollinger_bands(df['close'], self.config.bb_period, self.config.bb_std)
        df['bb_width'] = Indicators.bb_width(df['bb_upper'], df['bb_lower'], df['bb_middle'])
        df['bb_position'] = Indicators.bb_position(df['close'], df['bb_upper'], df['bb_lower'])
        df['macd'], df['macd_signal'], df['macd_hist'] = Indicators.macd(df['close'], self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
        df['choppiness'] = Indicators.choppiness_index(df['high'], df['low'], df['close'], self.config.choppiness_period)

        # Volume Profile POC
        df['vp_poc'] = df['close'].rolling(window=self.config.vp_lookback).apply(
            lambda x: Indicators.volume_profile(x, df['volume'].loc[x.index], self.config.vp_lookback, self.config.vp_row_size)['poc'],
            raw=False)

        return df

    def run_backtest(self, df, entry_long_fn, entry_short_fn, initial_balance=10000.0):
        df = self.prepare_data(df)
        self.trades = []
        balance = initial_balance
        max_balance = initial_balance
        max_drawdown = 0.0
        equity_curve = [initial_balance]

        in_trade = False
        current_trade = None

        for i in range(50, len(df) - 1):
            if not in_trade:
                equity_curve.append(balance)

            can_trade, reason = MarketRegime.is_tradeable(df, self.config, i)

            if in_trade and current_trade:
                current_trade.bars_held += 1

                if current_trade.direction == 'long':
                    if df['low'].iloc[i] <= current_trade.stop_loss:
                        current_trade.exit_idx = i
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size
                        current_trade.exit_reason = "Stop Loss"
                        balance += current_trade.pnl
                        self.trades.append(current_trade)
                        in_trade = False
                        current_trade = None
                        continue
                    elif df['high'].iloc[i] >= current_trade.take_profit:
                        current_trade.exit_idx = i
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size
                        current_trade.exit_reason = "Take Profit"
                        balance += current_trade.pnl
                        self.trades.append(current_trade)
                        in_trade = False
                        current_trade = None
                        continue
                else:
                    if df['high'].iloc[i] >= current_trade.stop_loss:
                        current_trade.exit_idx = i
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size
                        current_trade.exit_reason = "Stop Loss"
                        balance += current_trade.pnl
                        self.trades.append(current_trade)
                        in_trade = False
                        current_trade = None
                        continue
                    elif df['low'].iloc[i] <= current_trade.take_profit:
                        current_trade.exit_idx = i
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size
                        current_trade.exit_reason = "Take Profit"
                        balance += current_trade.pnl
                        self.trades.append(current_trade)
                        in_trade = False
                        current_trade = None
                        continue

                if current_trade.bars_held >= self.config.time_exit_bars:
                    current_trade.exit_idx = i
                    current_trade.exit_price = df['close'].iloc[i]
                    if current_trade.direction == 'long':
                        current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size
                    else:
                        current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size
                    current_trade.exit_reason = "Time Exit"
                    balance += current_trade.pnl
                    self.trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    continue

                if self.config.trailing_stop_atr_mult > 0:
                    atr_val = df['atr'].iloc[i]
                    if current_trade.direction == 'long':
                        new_sl = df['close'].iloc[i] - atr_val * self.config.trailing_stop_atr_mult
                        if new_sl > current_trade.stop_loss:
                            current_trade.stop_loss = new_sl
                    else:
                        new_sl = df['close'].iloc[i] + atr_val * self.config.trailing_stop_atr_mult
                        if new_sl < current_trade.stop_loss:
                            current_trade.stop_loss = new_sl

                if self.config.use_breakeven:
                    atr_val = df['atr'].iloc[i]
                    if current_trade.direction == 'long':
                        if df['close'].iloc[i] >= current_trade.entry_price + atr_val * self.config.breakeven_atr_mult:
                            if current_trade.stop_loss < current_trade.entry_price:
                                current_trade.stop_loss = current_trade.entry_price
                    else:
                        if df['close'].iloc[i] <= current_trade.entry_price - atr_val * self.config.breakeven_atr_mult:
                            if current_trade.stop_loss > current_trade.entry_price:
                                current_trade.stop_loss = current_trade.entry_price

            if not in_trade and can_trade:
                entry_price = df['close'].iloc[i]
                atr_val = df['atr'].iloc[i]

                try:
                    if entry_long_fn(df, i):
                        sl = entry_price - atr_val * self.config.atr_multiplier_sl
                        tp = entry_price + atr_val * self.config.atr_multiplier_tp
                        risk_amount = balance * self.config.risk_per_trade
                        risk_pips = abs(entry_price - sl) / self.config.pip_value
                        size = risk_amount / (risk_pips * self.config.pip_value * 100000)
                        size = max(size, 0.01)

                        current_trade = Trade(
                            entry_idx=i, entry_price=entry_price, direction='long',
                            stop_loss=sl, take_profit=tp, size=size
                        )
                        in_trade = True
                        continue
                except Exception:
                    pass

                try:
                    if entry_short_fn(df, i):
                        sl = entry_price + atr_val * self.config.atr_multiplier_sl
                        tp = entry_price - atr_val * self.config.atr_multiplier_tp
                        risk_amount = balance * self.config.risk_per_trade
                        risk_pips = abs(entry_price - sl) / self.config.pip_value
                        size = risk_amount / (risk_pips * self.config.pip_value * 100000)
                        size = max(size, 0.01)

                        current_trade = Trade(
                            entry_idx=i, entry_price=entry_price, direction='short',
                            stop_loss=sl, take_profit=tp, size=size
                        )
                        in_trade = True
                        continue
                except Exception:
                    pass

            if balance > max_balance:
                max_balance = balance
            drawdown = (max_balance - balance) / max_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        if in_trade and current_trade:
            current_trade.exit_idx = len(df) - 1
            current_trade.exit_price = df['close'].iloc[-1]
            if current_trade.direction == 'long':
                current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size
            else:
                current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size
            current_trade.exit_reason = "End of Data"
            balance += current_trade.pnl
            self.trades.append(current_trade)

        return self._calculate_metrics(balance, initial_balance, max_drawdown, equity_curve)

    def _calculate_metrics(self, final_balance, initial_balance, max_drawdown, equity_curve):
        trades = self.trades
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'net_profit': 0, 'max_drawdown': 0, 'avg_trade': 0,
                'avg_win': 0, 'avg_loss': 0, 'sharpe': 0, 'balance': final_balance
            }

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0.001

        win_rate = len(wins) / len(trades) * 100
        profit_factor = total_wins / total_losses if total_losses > 0 else 999

        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 12) if np.std(returns) > 0 else 0

        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_profit': final_balance - initial_balance,
            'net_profit_pct': (final_balance - initial_balance) / initial_balance * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'avg_trade': np.mean([t.pnl for t in trades]),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'sharpe': sharpe,
            'balance': final_balance,
            'trades': trades
        }


# ==============================================================================
# STRATEGY LIBRARY - 100 MEAN REVERSION STRATEGIES
# ==============================================================================

def get_all_strategies():
    strategies = []

    # ======================================================================
    # CATEGORY 1: BOLLINGER BAND BASED (1-15)
    # ======================================================================
    strategies.extend([
        {
            'id': 1, 'name': 'BB Extreme Touch Reversal',
            'desc': 'Price touches outer BB + RSI extreme + ALMA slope confirmation',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] <= df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] >= df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'bb_std': 2.0, 'rsi_period': 14, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 2, 'name': 'BB Double Touch',
            'desc': 'Two consecutive closes outside BB + StochRSI extreme',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['close'].iloc[i-1] < df['bb_lower'].iloc[i-1] and
                df['stoch_k'].iloc[i] < 20 and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['close'].iloc[i-1] > df['bb_upper'].iloc[i-1] and
                df['stoch_k'].iloc[i] > 80 and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1]
            ),
            'config': {'bb_std': 2.0, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 3, 'name': 'BB Squeeze Reversal',
            'desc': 'BB width compression then expansion with mean reversion',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['bb_width'].iloc[i-3:i].mean() < df['bb_width'].iloc[i-10:i-3].mean() * 0.8 and
                df['bb_width'].iloc[i] > df['bb_width'].iloc[i-1] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['bb_width'].iloc[i-3:i].mean() < df['bb_width'].iloc[i-10:i-3].mean() * 0.8 and
                df['bb_width'].iloc[i] > df['bb_width'].iloc[i-1] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'bb_std': 2.0, 'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 4, 'name': 'BB Middle Band Rejection',
            'desc': 'Price crosses middle BB and rejects back to mean',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i-1] < df['bb_middle'].iloc[i-1] and
                df['close'].iloc[i] > df['bb_middle'].iloc[i] and
                df['close'].iloc[i] < df['bb_middle'].iloc[i] + df['atr'].iloc[i] * 0.3 and
                df['rsi'].iloc[i] > 40 and df['rsi'].iloc[i] < 55 and
                df['alma'].iloc[i] > df['alma'].iloc[i-2]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i-1] > df['bb_middle'].iloc[i-1] and
                df['close'].iloc[i] < df['bb_middle'].iloc[i] and
                df['close'].iloc[i] > df['bb_middle'].iloc[i] - df['atr'].iloc[i] * 0.3 and
                df['rsi'].iloc[i] < 60 and df['rsi'].iloc[i] > 45 and
                df['alma'].iloc[i] < df['alma'].iloc[i-2]
            ),
            'config': {'atr_multiplier_sl': 1.0, 'atr_multiplier_tp': 1.8}
        },
        {
            'id': 5, 'name': 'BB %B Extreme',
            'desc': '%B indicator extreme with MACD histogram divergence',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['bb_position'].iloc[i] < 0.05 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['bb_position'].iloc[i] > 0.95 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'bb_std': 2.5, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 6, 'name': 'BB Walk-the-Bands Reversal',
            'desc': 'Price walks upper/lower band 3+ bars then reverses',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                sum(df['close'].iloc[i-3:i+1] < df['bb_lower'].iloc[i-3:i+1]) >= 3 and
                df['close'].iloc[i] > df['close'].iloc[i-1] and
                df['stoch_k'].iloc[i] < 25 and df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                sum(df['close'].iloc[i-3:i+1] > df['bb_upper'].iloc[i-3:i+1]) >= 3 and
                df['close'].iloc[i] < df['close'].iloc[i-1] and
                df['stoch_k'].iloc[i] > 75 and df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.8, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 7, 'name': 'BB + ALMA Deviation',
            'desc': 'Price far from ALMA with BB extreme confirmation',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                (df['close'].iloc[i] - df['alma'].iloc[i]) / df['atr'].iloc[i] < -2.0 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['ema9'].iloc[i] < df['alma'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                (df['close'].iloc[i] - df['alma'].iloc[i]) / df['atr'].iloc[i] > 2.0 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['ema9'].iloc[i] > df['alma'].iloc[i]
            ),
            'config': {'alma_period': 9, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 8, 'name': 'BB Bandwidth Expansion Fade',
            'desc': 'Fade the move after BB bandwidth expands rapidly',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['bb_width'].iloc[i] > df['bb_width'].iloc[i-5:i].mean() * 1.5 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['close'].iloc[i] > df['low'].iloc[i-1] and
                df['rsi'].iloc[i] < 40 and df['rsi'].iloc[i] > df['rsi'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['bb_width'].iloc[i] > df['bb_width'].iloc[i-5:i].mean() * 1.5 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['close'].iloc[i] < df['high'].iloc[i-1] and
                df['rsi'].iloc[i] > 60 and df['rsi'].iloc[i] < df['rsi'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.3, 'atr_multiplier_tp': 2.0, 'time_exit_bars': 6}
        },
        {
            'id': 9, 'name': 'BB Three Push Pattern',
            'desc': 'Three pushes to BB band with weakening momentum',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i-4] <= df['bb_lower'].iloc[i-4] and
                df['low'].iloc[i-2] <= df['bb_lower'].iloc[i-2] and
                df['low'].iloc[i] <= df['bb_lower'].iloc[i] and
                df['low'].iloc[i] > df['low'].iloc[i-2] and
                df['rsi'].iloc[i-2] > df['rsi'].iloc[i-4]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i-4] >= df['bb_upper'].iloc[i-4] and
                df['high'].iloc[i-2] >= df['bb_upper'].iloc[i-2] and
                df['high'].iloc[i] >= df['bb_upper'].iloc[i] and
                df['high'].iloc[i] < df['high'].iloc[i-2] and
                df['rsi'].iloc[i-2] < df['rsi'].iloc[i-4]
            ),
            'config': {'bb_std': 2.0, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.5}
        },
        {
            'id': 10, 'name': 'BB + Volume POC Reversion',
            'desc': 'Price at BB extreme near Volume POC level',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['rsi'].iloc[i] < 35 and
                df['stoch_k'].iloc[i] < 20
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['rsi'].iloc[i] > 65 and
                df['stoch_k'].iloc[i] > 80
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 11, 'name': 'BB Momentum Divergence',
            'desc': 'Price at BB extreme but momentum indicator diverging',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['close'].iloc[i] < df['close'].iloc[i-3] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-3] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-3]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['close'].iloc[i] > df['close'].iloc[i-3] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-3] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-3]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 12, 'name': 'BB EMA Cross Reversion',
            'desc': 'Price hits BB extreme after EMA cross in opposite direction',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['ema9'].iloc[i-2] < df['ema21'].iloc[i-2] and
                df['ema9'].iloc[i] > df['ema21'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['ema9'].iloc[i-2] > df['ema21'].iloc[i-2] and
                df['ema9'].iloc[i] < df['ema21'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 60
            ),
            'config': {'ema_fast': 9, 'ema_slow': 21, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 13, 'name': 'BB Consecutive Closes',
            'desc': '3+ consecutive closes outside BB with reversal candle',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                sum(df['close'].iloc[i-4:i] < df['bb_lower'].iloc[i-4:i]) >= 3 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] > df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 30
            ),
            'entry_short': lambda df, i: (
                sum(df['close'].iloc[i-4:i] > df['bb_upper'].iloc[i-4:i]) >= 3 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i] < df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 70
            ),
            'config': {'time_exit_bars': 10, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 14, 'name': 'BB + Choppiness Extreme',
            'desc': 'BB extreme when choppiness is at optimal MR levels',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 70 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 70 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'choppiness_min': 55, 'choppiness_max': 70, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 15, 'name': 'BB W-Bottom / M-Top',
            'desc': 'Classic W-bottom at lower BB or M-top at upper BB',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i-4] <= df['bb_lower'].iloc[i-4] and
                df['low'].iloc[i-2] <= df['bb_lower'].iloc[i-2] and
                df['low'].iloc[i-2] > df['low'].iloc[i-4] and
                df['low'].iloc[i] > df['low'].iloc[i-2] and
                df['rsi'].iloc[i-2] > df['rsi'].iloc[i-4]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i-4] >= df['bb_upper'].iloc[i-4] and
                df['high'].iloc[i-2] >= df['bb_upper'].iloc[i-2] and
                df['high'].iloc[i-2] < df['high'].iloc[i-4] and
                df['high'].iloc[i] < df['high'].iloc[i-2] and
                df['rsi'].iloc[i-2] < df['rsi'].iloc[i-4]
            ),
            'config': {'bb_std': 2.0, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 4.0, 'time_exit_bars': 16}
        },
    ])

    # ======================================================================
    # CATEGORY 2: RSI BASED (16-30)
    # ======================================================================
    strategies.extend([
        {
            'id': 16, 'name': 'RSI Extreme Reversal',
            'desc': 'RSI below 20 or above 80 with candle confirmation',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i] < 20 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean() * 1.2
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i] > 80 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean() * 1.2
            ),
            'config': {'rsi_period': 14, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 17, 'name': 'RSI Divergence Classic',
            'desc': 'Price makes lower low, RSI makes higher low',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                df['rsi'].iloc[i] < 40 and
                df['close'].iloc[i] > df['open'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                df['rsi'].iloc[i] > 60 and
                df['close'].iloc[i] < df['open'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0, 'time_exit_bars': 12}
        },
        {
            'id': 18, 'name': 'RSI Hidden Divergence',
            'desc': 'Hidden divergence indicating continuation of mean reversion',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] > df['low'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                df['alma'].iloc[i] > df['alma'].iloc[i-3] and
                df['rsi'].iloc[i] < 45
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] < df['high'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                df['alma'].iloc[i] < df['alma'].iloc[i-3] and
                df['rsi'].iloc[i] > 55
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 19, 'name': 'RSI 50 Cross Reversion',
            'desc': 'RSI crosses back through 50 after extreme',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i-2] < 30 and
                df['rsi'].iloc[i-1] < 50 and
                df['rsi'].iloc[i] > 50 and
                df['close'].iloc[i] > df['ema9'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i-2] > 70 and
                df['rsi'].iloc[i-1] > 50 and
                df['rsi'].iloc[i] < 50 and
                df['close'].iloc[i] < df['ema9'].iloc[i]
            ),
            'config': {'time_exit_bars': 6, 'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 20, 'name': 'RSI + ALMA Confluence',
            'desc': 'RSI extreme with price far from ALMA',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i] < 25 and
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] * 1.5 and
                df['stoch_k'].iloc[i] < 20 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i] > 75 and
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] * 1.5 and
                df['stoch_k'].iloc[i] > 80 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'alma_period': 9, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 21, 'name': 'RSI Failure Swing',
            'desc': 'RSI failure swing pattern - classic Wyckoff',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i-4] < 30 and
                df['rsi'].iloc[i-2] > df['rsi'].iloc[i-4] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-2] and
                df['rsi'].iloc[i-1] < df['rsi'].iloc[i-2] and
                df['rsi'].iloc[i] > 30
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i-4] > 70 and
                df['rsi'].iloc[i-2] < df['rsi'].iloc[i-4] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-2] and
                df['rsi'].iloc[i-1] > df['rsi'].iloc[i-2] and
                df['rsi'].iloc[i] < 70
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 22, 'name': 'RSI Midline Rejection',
            'desc': 'RSI rejects midline (50) after extreme reading',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                min(df['rsi'].iloc[i-5:i]) < 30 and
                df['rsi'].iloc[i-1] < 55 and df['rsi'].iloc[i-1] > 45 and
                df['rsi'].iloc[i] > 50 and
                df['close'].iloc[i] > df['close'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                max(df['rsi'].iloc[i-5:i]) > 75 and
                df['rsi'].iloc[i-1] < 55 and df['rsi'].iloc[i-1] > 45 and
                df['rsi'].iloc[i] < 50 and
                df['close'].iloc[i] < df['close'].iloc[i-1]
            ),
            'config': {'time_exit_bars': 5, 'atr_multiplier_sl': 1.0, 'atr_multiplier_tp': 1.8}
        },
        {
            'id': 23, 'name': 'RSI 2-Period Extreme',
            'desc': 'Larry Connors RSI(2) extreme mean reversion',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['rsi2'].iloc[i] < 10 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] < df['alma'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['rsi2'].iloc[i] > 90 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i] > df['alma'].iloc[i]
            ),
            'config': {'time_exit_bars': 3, 'atr_multiplier_sl': 1.0, 'atr_multiplier_tp': 1.5}
        },
        {
            'id': 24, 'name': 'RSI + BB + Volume Spike',
            'desc': 'RSI extreme at BB with volume confirmation',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i] < 25 and
                df['close'].iloc[i] <= df['bb_lower'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-10:i].mean() * 1.5 and
                df['volume'].iloc[i] > df['volume'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i] > 75 and
                df['close'].iloc[i] >= df['bb_upper'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-10:i].mean() * 1.5 and
                df['volume'].iloc[i] > df['volume'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.8, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 25, 'name': 'RSI Trendline Break',
            'desc': 'Break of RSI trendline after extreme',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i-3] > 25 and df['rsi'].iloc[i-3] < 35 and
                df['rsi'].iloc[i-1] > df['rsi'].iloc[i-3] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-1] and
                df['close'].iloc[i] > df['ema9'].iloc[i] and
                df['macd_hist'].iloc[i] > 0
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i-3] < 75 and df['rsi'].iloc[i-3] > 65 and
                df['rsi'].iloc[i-1] < df['rsi'].iloc[i-3] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-1] and
                df['close'].iloc[i] < df['ema9'].iloc[i] and
                df['macd_hist'].iloc[i] < 0
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.8}
        },
        {
            'id': 26, 'name': 'RSI Range Bound Fade',
            'desc': 'Fade when RSI hits range boundaries in choppy market',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and
                df['rsi'].iloc[i] < 30 and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                abs(df['close'].iloc[i] - df['bb_middle'].iloc[i]) < df['atr'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and
                df['rsi'].iloc[i] > 70 and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                abs(df['close'].iloc[i] - df['bb_middle'].iloc[i]) < df['atr'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.0, 'atr_multiplier_tp': 1.5}
        },
        {
            'id': 27, 'name': 'RSI + StochRSI Double Extreme',
            'desc': 'Both RSI and StochRSI at extreme levels',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i] < 30 and
                df['stoch_k'].iloc[i] < 15 and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1] and
                df['close'].iloc[i] > df['low'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i] > 70 and
                df['stoch_k'].iloc[i] > 85 and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1] and
                df['close'].iloc[i] < df['high'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.4, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 28, 'name': 'RSI Overbought/Oversold Pullback',
            'desc': 'Enter on first pullback after extreme RSI',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                min(df['rsi'].iloc[i-5:i]) < 25 and
                df['rsi'].iloc[i] > 30 and df['rsi'].iloc[i] < 45 and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['close'].iloc[i-1] < df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                max(df['rsi'].iloc[i-5:i]) > 75 and
                df['rsi'].iloc[i] < 70 and df['rsi'].iloc[i] > 55 and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['close'].iloc[i-1] > df['alma'].iloc[i-1]
            ),
            'config': {'time_exit_bars': 4, 'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 29, 'name': 'RSI + MACD Histogram Divergence',
            'desc': 'RSI extreme with MACD histogram showing momentum shift',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i] < 35 and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-3] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['macd_hist'].iloc[i] < 0 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i] > 65 and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-3] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['macd_hist'].iloc[i] > 0 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 30, 'name': 'RSI Dynamic Levels',
            'desc': 'RSI relative to dynamic overbought/oversold based on volatility',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i] < (30 - (df['atr'].iloc[i] / df['atr'].iloc[i-20:i].mean() - 1) * 10) and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 20
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i] > (70 + (df['atr'].iloc[i] / df['atr'].iloc[i-20:i].mean() - 1) * 10) and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 80
            ),
            'config': {'atr_multiplier_sl': 1.6, 'atr_multiplier_tp': 2.5}
        },
    ])


    # ======================================================================
    # CATEGORY 3: STOCHRSI BASED (31-40)
    # ======================================================================
    strategies.extend([
        {
            'id': 31, 'name': 'StochRSI Extreme Reversal',
            'desc': 'StochRSI below 10 or above 90 with ALMA confirmation',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['stoch_k'].iloc[i] < 10 and
                df['stoch_d'].iloc[i] < 10 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['stoch_k'].iloc[i] > 90 and
                df['stoch_d'].iloc[i] > 90 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'stochrsi_period': 14, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 32, 'name': 'StochRSI K-D Cross',
            'desc': 'K line crosses above D line from extreme oversold',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['stoch_k'].iloc[i-1] < df['stoch_d'].iloc[i-1] and
                df['stoch_k'].iloc[i] > df['stoch_d'].iloc[i] and
                df['stoch_k'].iloc[i] < 30 and
                df['rsi'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['stoch_k'].iloc[i-1] > df['stoch_d'].iloc[i-1] and
                df['stoch_k'].iloc[i] < df['stoch_d'].iloc[i] and
                df['stoch_k'].iloc[i] > 70 and
                df['rsi'].iloc[i] > 60
            ),
            'config': {'atr_multiplier_sl': 1.3, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 33, 'name': 'StochRSI Divergence',
            'desc': 'Price/StochRSI divergence at extremes',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-5] and
                df['stoch_k'].iloc[i] < 30 and
                df['close'].iloc[i] > df['bb_lower'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-5] and
                df['stoch_k'].iloc[i] > 70 and
                df['close'].iloc[i] < df['bb_upper'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.8}
        },
        {
            'id': 34, 'name': 'StochRSI + BB Confluence',
            'desc': 'StochRSI extreme with BB band touch',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['stoch_k'].iloc[i] < 15 and
                df['close'].iloc[i] <= df['bb_lower'].iloc[i] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean()
            ),
            'entry_short': lambda df, i: (
                df['stoch_k'].iloc[i] > 85 and
                df['close'].iloc[i] >= df['bb_upper'].iloc[i] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean()
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 35, 'name': 'StochRSI Triple Bottom',
            'desc': 'Three StochRSI touches of oversold with higher lows',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['stoch_k'].iloc[i-4] < 20 and
                df['stoch_k'].iloc[i-2] < 20 and
                df['stoch_k'].iloc[i] < 20 and
                df['stoch_k'].iloc[i-2] > df['stoch_k'].iloc[i-4] and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-2] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-2]
            ),
            'entry_short': lambda df, i: (
                df['stoch_k'].iloc[i-4] > 80 and
                df['stoch_k'].iloc[i-2] > 80 and
                df['stoch_k'].iloc[i] > 80 and
                df['stoch_k'].iloc[i-2] < df['stoch_k'].iloc[i-4] and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-2] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-2]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 36, 'name': 'StochRSI Midline Crossback',
            'desc': 'StochRSI crosses back through 50 after extreme',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['stoch_k'].iloc[i-2] < 20 and
                df['stoch_k'].iloc[i-1] < 50 and
                df['stoch_k'].iloc[i] > 50 and
                df['stoch_d'].iloc[i] > df['stoch_d'].iloc[i-1] and
                df['close'].iloc[i] > df['ema9'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['stoch_k'].iloc[i-2] > 80 and
                df['stoch_k'].iloc[i-1] > 50 and
                df['stoch_k'].iloc[i] < 50 and
                df['stoch_d'].iloc[i] < df['stoch_d'].iloc[i-1] and
                df['close'].iloc[i] < df['ema9'].iloc[i]
            ),
            'config': {'time_exit_bars': 5, 'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 37, 'name': 'StochRSI + Volume Profile Node',
            'desc': 'StochRSI extreme at high volume node',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['stoch_k'].iloc[i] < 15 and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.3 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['stoch_k'].iloc[i] > 85 and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.3 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 38, 'name': 'StochRSI + EMA Rejection',
            'desc': 'StochRSI extreme with price rejecting EMA',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['stoch_k'].iloc[i] < 20 and
                df['close'].iloc[i-1] < df['ema9'].iloc[i-1] and
                df['close'].iloc[i] > df['ema9'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['macd_hist'].iloc[i] > 0
            ),
            'entry_short': lambda df, i: (
                df['stoch_k'].iloc[i] > 80 and
                df['close'].iloc[i-1] > df['ema9'].iloc[i-1] and
                df['close'].iloc[i] < df['ema9'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['macd_hist'].iloc[i] < 0
            ),
            'config': {'atr_multiplier_sl': 1.4, 'atr_multiplier_tp': 2.2}
        },
        {
            'id': 39, 'name': 'StochRSI Range Exhaustion',
            'desc': 'StochRSI stuck at extreme for multiple bars then reverses',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                sum(df['stoch_k'].iloc[i-4:i] < 15) >= 3 and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1] and
                df['stoch_k'].iloc[i] > df['stoch_d'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                sum(df['stoch_k'].iloc[i-4:i] > 85) >= 3 and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1] and
                df['stoch_k'].iloc[i] < df['stoch_d'].iloc[i] and
                df['close'].iloc[i] < df['open'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 40, 'name': 'StochRSI + Choppiness Filter',
            'desc': 'StochRSI signal only in optimal choppy conditions',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 68 and
                df['stoch_k'].iloc[i] < 20 and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1] and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['rsi'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 68 and
                df['stoch_k'].iloc[i] > 80 and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1] and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['rsi'].iloc[i] > 60
            ),
            'config': {'choppiness_min': 55, 'choppiness_max': 68, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
    ])

    # ======================================================================
    # CATEGORY 4: ALMA BASED (41-50)
    # ======================================================================
    strategies.extend([
        {
            'id': 41, 'name': 'ALMA Mean Reversion',
            'desc': 'Price deviation from ALMA with reversal candle',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] * 1.5 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['rsi'].iloc[i] < 35 and
                df['stoch_k'].iloc[i] < 25
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] * 1.5 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['rsi'].iloc[i] > 65 and
                df['stoch_k'].iloc[i] > 75
            ),
            'config': {'alma_period': 9, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 42, 'name': 'ALMA Slope Reversal',
            'desc': 'ALMA slope changes direction after extreme price move',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['alma'].iloc[i-2] > df['alma'].iloc[i-1] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1] and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['alma'].iloc[i-2] < df['alma'].iloc[i-1] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1] and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 43, 'name': 'ALMA + EMA Cross Mean Reversion',
            'desc': 'Price far from ALMA after EMA cross',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['ema9'].iloc[i] > df['ema21'].iloc[i] and
                df['ema9'].iloc[i-3] < df['ema21'].iloc[i-3] and
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] and
                df['stoch_k'].iloc[i] < 30 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['ema9'].iloc[i] < df['ema21'].iloc[i] and
                df['ema9'].iloc[i-3] > df['ema21'].iloc[i-3] and
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] and
                df['stoch_k'].iloc[i] > 70 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 44, 'name': 'ALMA Channel Bounce',
            'desc': 'Price bounces off ALMA +/- ATR channel',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i-1] < df['alma'].iloc[i-1] - df['atr'].iloc[i-1] * 2 and
                df['close'].iloc[i] > df['alma'].iloc[i] - df['atr'].iloc[i] * 2 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['rsi'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i-1] > df['alma'].iloc[i-1] + df['atr'].iloc[i-1] * 2 and
                df['close'].iloc[i] < df['alma'].iloc[i] + df['atr'].iloc[i] * 2 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['rsi'].iloc[i] > 60
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 45, 'name': 'ALMA + Volume POC Magnet',
            'desc': 'Price stretched from ALMA near volume POC',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                (df['close'].iloc[i] - df['alma'].iloc[i]) / df['atr'].iloc[i] < -2.5 and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['stoch_k'].iloc[i] < 20 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3
            ),
            'entry_short': lambda df, i: (
                (df['close'].iloc[i] - df['alma'].iloc[i]) / df['atr'].iloc[i] > 2.5 and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['stoch_k'].iloc[i] > 80 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 46, 'name': 'ALMA Divergence Pattern',
            'desc': 'Price makes new extreme, ALMA slope moderates',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                (df['alma'].iloc[i] - df['alma'].iloc[i-5]) / df['atr'].iloc[i] > -0.5 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35 and df['rsi'].iloc[i] > df['rsi'].iloc[i-2]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                (df['alma'].iloc[i] - df['alma'].iloc[i-5]) / df['atr'].iloc[i] < 0.5 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65 and df['rsi'].iloc[i] < df['rsi'].iloc[i-2]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 47, 'name': 'ALMA Pullback to Mean',
            'desc': 'Enter on pullback to ALMA after extreme deviation',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                min((df['close'].iloc[i-5:i] - df['alma'].iloc[i-5:i]) / df['atr'].iloc[i-5:i]) < -2.0 and
                df['close'].iloc[i-1] < df['alma'].iloc[i-1] and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['close'].iloc[i] < df['alma'].iloc[i] + df['atr'].iloc[i] * 0.3 and
                df['rsi'].iloc[i] < 50
            ),
            'entry_short': lambda df, i: (
                max((df['close'].iloc[i-5:i] - df['alma'].iloc[i-5:i]) / df['atr'].iloc[i-5:i]) > 2.0 and
                df['close'].iloc[i-1] > df['alma'].iloc[i-1] and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['close'].iloc[i] > df['alma'].iloc[i] - df['atr'].iloc[i] * 0.3 and
                df['rsi'].iloc[i] > 50
            ),
            'config': {'time_exit_bars': 5, 'atr_multiplier_sl': 1.0, 'atr_multiplier_tp': 1.8}
        },
        {
            'id': 48, 'name': 'ALMA + MACD Zero Line Reversion',
            'desc': 'Price far from ALMA with MACD near zero',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] * 2 and
                abs(df['macd'].iloc[i]) < df['atr'].iloc[i] * 100 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] * 2 and
                abs(df['macd'].iloc[i]) < df['atr'].iloc[i] * 100 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 49, 'name': 'ALMA Dynamic Support/Resistance',
            'desc': 'ALMA acts as dynamic S/R with price rejection',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i-1] <= df['alma'].iloc[i-1] + df['atr'].iloc[i-1] * 0.2 and
                df['low'].iloc[i-1] >= df['alma'].iloc[i-1] - df['atr'].iloc[i-1] * 0.2 and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['stoch_k'].iloc[i] < 40 and df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i-1] <= df['alma'].iloc[i-1] + df['atr'].iloc[i-1] * 0.2 and
                df['high'].iloc[i-1] >= df['alma'].iloc[i-1] - df['atr'].iloc[i-1] * 0.2 and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['stoch_k'].iloc[i] > 60 and df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 50, 'name': 'ALMA + BB Squeeze Mean Reversion',
            'desc': 'ALMA deviation during BB squeeze expansion',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['bb_width'].iloc[i-5:i].mean() < df['bb_width'].iloc[i-15:i-5].mean() * 0.7 and
                df['bb_width'].iloc[i] > df['bb_width'].iloc[i-1] and
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['bb_width'].iloc[i-5:i].mean() < df['bb_width'].iloc[i-15:i-5].mean() * 0.7 and
                df['bb_width'].iloc[i] > df['bb_width'].iloc[i-1] and
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'bb_std': 2.0, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
    ])


    # ======================================================================
    # CATEGORY 5: MACD BASED (51-60)
    # ======================================================================
    strategies.extend([
        {
            'id': 51, 'name': 'MACD Histogram Divergence',
            'desc': 'Price extreme with MACD histogram divergence',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-4] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-4] and
                df['macd_hist'].iloc[i] < 0 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-4] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-4] and
                df['macd_hist'].iloc[i] > 0 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'macd_fast': 8, 'macd_slow': 21, 'macd_signal': 5, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 52, 'name': 'MACD Zero Line Rejection',
            'desc': 'MACD rejects zero line after extreme price move',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['macd'].iloc[i-2] < 0 and df['macd'].iloc[i-1] < 0 and
                df['macd'].iloc[i] > 0 and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['rsi'].iloc[i] < 45 and
                df['stoch_k'].iloc[i] < 30
            ),
            'entry_short': lambda df, i: (
                df['macd'].iloc[i-2] > 0 and df['macd'].iloc[i-1] > 0 and
                df['macd'].iloc[i] < 0 and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['rsi'].iloc[i] > 55 and
                df['stoch_k'].iloc[i] > 70
            ),
            'config': {'macd_fast': 8, 'macd_slow': 21, 'macd_signal': 5, 'atr_multiplier_sl': 1.3, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 53, 'name': 'MACD Signal Line Cross Reversion',
            'desc': 'MACD crosses signal line from extreme price position',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['macd'].iloc[i-1] < df['macd_signal'].iloc[i-1] and
                df['macd'].iloc[i] > df['macd_signal'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 40 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['macd'].iloc[i-1] > df['macd_signal'].iloc[i-1] and
                df['macd'].iloc[i] < df['macd_signal'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 60 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 54, 'name': 'MACD Histogram Triple Convergence',
            'desc': 'Three histogram bars converging to zero from extreme',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['macd_hist'].iloc[i-3] < df['macd_hist'].iloc[i-2] < df['macd_hist'].iloc[i-1] < 0 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['macd_hist'].iloc[i-3] > df['macd_hist'].iloc[i-2] > df['macd_hist'].iloc[i-1] > 0 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 55, 'name': 'MACD + RSI Extreme Combo',
            'desc': 'MACD turning with RSI at extreme',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['rsi'].iloc[i] < 25 and
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] and
                df['stoch_k'].iloc[i] < 20
            ),
            'entry_short': lambda df, i: (
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['rsi'].iloc[i] > 75 and
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] and
                df['stoch_k'].iloc[i] > 80
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 56, 'name': 'MACD Fakeout Reversal',
            'desc': 'MACD fake breakout then immediate reversal',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['macd'].iloc[i-2] > df['macd_signal'].iloc[i-2] and
                df['macd'].iloc[i-1] < df['macd_signal'].iloc[i-1] and
                df['macd'].iloc[i] > df['macd_signal'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['macd'].iloc[i-2] < df['macd_signal'].iloc[i-2] and
                df['macd'].iloc[i-1] > df['macd_signal'].iloc[i-1] and
                df['macd'].iloc[i] < df['macd_signal'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 60
            ),
            'config': {'atr_multiplier_sl': 1.3, 'atr_multiplier_tp': 2.0, 'time_exit_bars': 5}
        },
        {
            'id': 57, 'name': 'MACD + Volume Profile POC',
            'desc': 'MACD reversal signal at volume POC',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['macd_hist'].iloc[i] > 0 and df['macd_hist'].iloc[i-1] < 0 and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['rsi'].iloc[i] < 45
            ),
            'entry_short': lambda df, i: (
                df['macd_hist'].iloc[i] < 0 and df['macd_hist'].iloc[i-1] > 0 and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['rsi'].iloc[i] > 55
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 58, 'name': 'MACD Histogram Extreme Compression',
            'desc': 'Histogram compressed then expands in reversal direction',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                max(abs(df['macd_hist'].iloc[i-5:i])) < df['atr'].iloc[i] * 50 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['macd_hist'].iloc[i] < 0 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 25
            ),
            'entry_short': lambda df, i: (
                max(abs(df['macd_hist'].iloc[i-5:i])) < df['atr'].iloc[i] * 50 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['macd_hist'].iloc[i] > 0 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 75
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 59, 'name': 'MACD + ALMA Alignment',
            'desc': 'MACD and ALMA both signaling mean reversion',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['macd'].iloc[i] > df['macd_signal'].iloc[i] and
                df['macd'].iloc[i-1] < df['macd_signal'].iloc[i-1] and
                df['alma'].iloc[i] > df['alma'].iloc[i-2] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['macd'].iloc[i] < df['macd_signal'].iloc[i] and
                df['macd'].iloc[i-1] > df['macd_signal'].iloc[i-1] and
                df['alma'].iloc[i] < df['alma'].iloc[i-2] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.8}
        },
        {
            'id': 60, 'name': 'MACD Choppiness Filtered',
            'desc': 'MACD signal only in choppy market conditions',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 70 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['rsi'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 70 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['rsi'].iloc[i] > 60
            ),
            'config': {'choppiness_min': 55, 'choppiness_max': 70, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
    ])

    # ======================================================================
    # CATEGORY 6: EMA BASED (61-70)
    # ======================================================================
    strategies.extend([
        {
            'id': 61, 'name': 'EMA9 Rejection Scalp',
            'desc': 'Price rejects EMA9 after extreme move',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i-1] < df['ema9'].iloc[i-1] and
                df['close'].iloc[i] > df['ema9'].iloc[i] and
                df['close'].iloc[i-2] < df['bb_lower'].iloc[i-2] and
                df['rsi'].iloc[i] < 45 and
                df['stoch_k'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i-1] > df['ema9'].iloc[i-1] and
                df['close'].iloc[i] < df['ema9'].iloc[i] and
                df['close'].iloc[i-2] > df['bb_upper'].iloc[i-2] and
                df['rsi'].iloc[i] > 55 and
                df['stoch_k'].iloc[i] > 65
            ),
            'config': {'ema_fast': 9, 'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 62, 'name': 'EMA Cross Mean Reversion',
            'desc': 'EMA9 crosses back over EMA21 after price extreme',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['ema9'].iloc[i-1] < df['ema21'].iloc[i-1] and
                df['ema9'].iloc[i] > df['ema21'].iloc[i] and
                df['close'].iloc[i] < df['bb_middle'].iloc[i] and
                df['rsi'].iloc[i] < 45 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['ema9'].iloc[i-1] > df['ema21'].iloc[i-1] and
                df['ema9'].iloc[i] < df['ema21'].iloc[i] and
                df['close'].iloc[i] > df['bb_middle'].iloc[i] and
                df['rsi'].iloc[i] > 55 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'ema_fast': 9, 'ema_slow': 21, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 63, 'name': 'EMA Ribbon Compression',
            'desc': 'EMAs compress then price snaps back to mean',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                abs(df['ema9'].iloc[i-3] - df['ema21'].iloc[i-3]) < df['atr'].iloc[i-3] * 0.3 and
                df['close'].iloc[i] < df['ema9'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35 and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                abs(df['ema9'].iloc[i-3] - df['ema21'].iloc[i-3]) < df['atr'].iloc[i-3] * 0.3 and
                df['close'].iloc[i] > df['ema9'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65 and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 64, 'name': 'EMA9 + ALMA Double Bounce',
            'desc': 'Price bounces off both EMA9 and ALMA',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i-1] <= df['ema9'].iloc[i-1] + df['atr'].iloc[i-1] * 0.1 and
                df['low'].iloc[i-1] <= df['alma'].iloc[i-1] + df['atr'].iloc[i-1] * 0.1 and
                df['close'].iloc[i] > df['ema9'].iloc[i] and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['rsi'].iloc[i] < 45
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i-1] >= df['ema9'].iloc[i-1] - df['atr'].iloc[i-1] * 0.1 and
                df['high'].iloc[i-1] >= df['alma'].iloc[i-1] - df['atr'].iloc[i-1] * 0.1 and
                df['close'].iloc[i] < df['ema9'].iloc[i] and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['rsi'].iloc[i] > 55
            ),
            'config': {'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 65, 'name': 'EMA Deviation Scalp',
            'desc': 'Price far from EMA9 with quick mean reversion',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                (df['close'].iloc[i] - df['ema9'].iloc[i]) / df['atr'].iloc[i] < -2.0 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['stoch_k'].iloc[i] < 25 and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean()
            ),
            'entry_short': lambda df, i: (
                (df['close'].iloc[i] - df['ema9'].iloc[i]) / df['atr'].iloc[i] > 2.0 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['stoch_k'].iloc[i] > 75 and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean()
            ),
            'config': {'time_exit_bars': 4, 'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 1.8}
        },
        {
            'id': 66, 'name': 'EMA21 Dynamic Support',
            'desc': 'EMA21 acts as dynamic support in ranging market',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and
                df['low'].iloc[i-1] <= df['ema21'].iloc[i-1] + df['atr'].iloc[i-1] * 0.15 and
                df['low'].iloc[i-1] >= df['ema21'].iloc[i-1] - df['atr'].iloc[i-1] * 0.15 and
                df['close'].iloc[i] > df['ema21'].iloc[i] and
                df['rsi'].iloc[i] < 50 and
                df['stoch_k'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and
                df['high'].iloc[i-1] <= df['ema21'].iloc[i-1] + df['atr'].iloc[i-1] * 0.15 and
                df['high'].iloc[i-1] >= df['ema21'].iloc[i-1] - df['atr'].iloc[i-1] * 0.15 and
                df['close'].iloc[i] < df['ema21'].iloc[i] and
                df['rsi'].iloc[i] > 50 and
                df['stoch_k'].iloc[i] > 60
            ),
            'config': {'ema_slow': 21, 'atr_multiplier_sl': 1.3, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 67, 'name': 'EMA + BB Confluence',
            'desc': 'EMA support/resistance with BB extreme',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['close'].iloc[i] < df['ema21'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['ema9'].iloc[i] > df['ema21'].iloc[i] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['close'].iloc[i] > df['ema21'].iloc[i] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['ema9'].iloc[i] < df['ema21'].iloc[i] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 68, 'name': 'EMA Slope Change + Volume',
            'desc': 'EMA slope changes with volume confirmation',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['ema9'].iloc[i-2] > df['ema9'].iloc[i-1] and
                df['ema9'].iloc[i] > df['ema9'].iloc[i-1] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5 and
                df['rsi'].iloc[i] < 40
            ),
            'entry_short': lambda df, i: (
                df['ema9'].iloc[i-2] < df['ema9'].iloc[i-1] and
                df['ema9'].iloc[i] < df['ema9'].iloc[i-1] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5 and
                df['rsi'].iloc[i] > 60
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 69, 'name': 'EMA Gap Fill Reversion',
            'desc': 'Price gaps away from EMA then reverts',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                (df['close'].iloc[i-1] - df['ema9'].iloc[i-1]) / df['atr'].iloc[i-1] < -2.5 and
                df['close'].iloc[i] > df['ema9'].iloc[i] and
                df['close'].iloc[i] < df['ema9'].iloc[i] + df['atr'].iloc[i] * 0.3 and
                df['rsi'].iloc[i] < 45
            ),
            'entry_short': lambda df, i: (
                (df['close'].iloc[i-1] - df['ema9'].iloc[i-1]) / df['atr'].iloc[i-1] > 2.5 and
                df['close'].iloc[i] < df['ema9'].iloc[i] and
                df['close'].iloc[i] > df['ema9'].iloc[i] - df['atr'].iloc[i] * 0.3 and
                df['rsi'].iloc[i] > 55
            ),
            'config': {'time_exit_bars': 4, 'atr_multiplier_sl': 1.0, 'atr_multiplier_tp': 1.5}
        },
        {
            'id': 70, 'name': 'EMA + StochRSI Extreme',
            'desc': 'EMA deviation with StochRSI at extreme',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['ema9'].iloc[i] - df['atr'].iloc[i] * 1.5 and
                df['stoch_k'].iloc[i] < 15 and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['ema9'].iloc[i] + df['atr'].iloc[i] * 1.5 and
                df['stoch_k'].iloc[i] > 85 and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
    ])


    # ======================================================================
    # CATEGORY 7: MULTI-INDICATOR COMBOS (71-85)
    # ======================================================================
    strategies.extend([
        {
            'id': 71, 'name': 'The Perfect Storm Long',
            'desc': 'BB extreme + RSI extreme + StochRSI extreme + MACD turning + ALMA slope',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 25 and
                df['stoch_k'].iloc[i] < 15 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean() * 1.2
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 75 and
                df['stoch_k'].iloc[i] > 85 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-3:i].mean() * 1.2
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 72, 'name': 'Triple Confirmation Reversion',
            'desc': 'RSI + StochRSI + MACD all align for mean reversion',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['rsi'].iloc[i] < 30 and df['rsi'].iloc[i] > df['rsi'].iloc[i-1] and
                df['stoch_k'].iloc[i] < 20 and df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['rsi'].iloc[i] > 70 and df['rsi'].iloc[i] < df['rsi'].iloc[i-1] and
                df['stoch_k'].iloc[i] > 80 and df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['close'].iloc[i] < df['open'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.8}
        },
        {
            'id': 73, 'name': 'BB + RSI + ALMA Trinity',
            'desc': 'Three core indicators aligned for high-probability MR',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                (df['close'].iloc[i] - df['alma'].iloc[i]) / df['atr'].iloc[i] < -1.5 and
                df['stoch_k'].iloc[i] < 25 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean()
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                (df['close'].iloc[i] - df['alma'].iloc[i]) / df['atr'].iloc[i] > 1.5 and
                df['stoch_k'].iloc[i] > 75 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean()
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 74, 'name': 'Volume Profile + BB + RSI',
            'desc': 'Volume POC confluence with BB and RSI extremes',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.4 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.4 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 75, 'name': 'Choppiness + BB + StochRSI',
            'desc': 'Optimal choppy conditions with BB and StochRSI extremes',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 58 and df['choppiness'].iloc[i] < 68 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 15 and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1] and
                df['rsi'].iloc[i] < 35
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 58 and df['choppiness'].iloc[i] < 68 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 85 and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1] and
                df['rsi'].iloc[i] > 65
            ),
            'config': {'choppiness_min': 58, 'choppiness_max': 68, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 76, 'name': 'EMA + MACD + RSI Confluence',
            'desc': 'All momentum indicators align for mean reversion',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['ema9'].iloc[i] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['rsi'].iloc[i] < 35 and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-1] and
                df['stoch_k'].iloc[i] < 25 and
                df['close'].iloc[i] > df['open'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['ema9'].iloc[i] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['rsi'].iloc[i] > 65 and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-1] and
                df['stoch_k'].iloc[i] > 75 and
                df['close'].iloc[i] < df['open'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.4, 'atr_multiplier_tp': 2.2}
        },
        {
            'id': 77, 'name': 'ALMA + BB + MACD + RSI Quad',
            'desc': 'Four-indicator confirmation for highest probability setups',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['rsi'].iloc[i] < 30 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['rsi'].iloc[i] > 70 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 78, 'name': 'Session Volume + Technical Confluence',
            'desc': 'Volume profile POC with multiple technical signals',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.3 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 15 and
                df['macd_hist'].iloc[i] > 0 and
                df['alma'].iloc[i] > df['alma'].iloc[i-2]
            ),
            'entry_short': lambda df, i: (
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.3 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 85 and
                df['macd_hist'].iloc[i] < 0 and
                df['alma'].iloc[i] < df['alma'].iloc[i-2]
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 79, 'name': 'ATR Filtered Multi-Indicator',
            'desc': 'All signals only when ATR is in optimal range',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['atr'].iloc[i] > df['atr'].iloc[i-20:i].mean() * 0.8 and
                df['atr'].iloc[i] < df['atr'].iloc[i-20:i].mean() * 1.5 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['stoch_k'].iloc[i] < 20 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['atr'].iloc[i] > df['atr'].iloc[i-20:i].mean() * 0.8 and
                df['atr'].iloc[i] < df['atr'].iloc[i-20:i].mean() * 1.5 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['stoch_k'].iloc[i] > 80 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 80, 'name': 'Institutional Level Reversion',
            'desc': 'Price at institutional volume level with full technical confirmation',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['vp_poc'].iloc[i] + df['atr'].iloc[i] * 0.2 and
                df['close'].iloc[i] > df['vp_poc'].iloc[i] - df['atr'].iloc[i] * 0.2 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35 and
                df['stoch_k'].iloc[i] < 20 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] < df['vp_poc'].iloc[i] + df['atr'].iloc[i] * 0.2 and
                df['close'].iloc[i] > df['vp_poc'].iloc[i] - df['atr'].iloc[i] * 0.2 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65 and
                df['stoch_k'].iloc[i] > 80 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 81, 'name': 'Mean Reversion Sniper',
            'desc': 'Ultra-selective: 5+ conditions must align',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 25 and
                df['stoch_k'].iloc[i] < 15 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 75 and
                df['stoch_k'].iloc[i] > 85 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.5}
        },
        {
            'id': 82, 'name': 'Composite Score Reversion',
            'desc': 'Weighted score of multiple indicators exceeds threshold',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                (df['bb_position'].iloc[i] < 0.1) * 2 +
                (df['rsi'].iloc[i] < 30) * 2 +
                (df['stoch_k'].iloc[i] < 20) * 1.5 +
                (df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]) * 1.5 +
                (df['alma'].iloc[i] > df['alma'].iloc[i-1]) * 1 +
                (df['close'].iloc[i] > df['open'].iloc[i]) * 1 >= 7
            ),
            'entry_short': lambda df, i: (
                (df['bb_position'].iloc[i] > 0.9) * 2 +
                (df['rsi'].iloc[i] > 70) * 2 +
                (df['stoch_k'].iloc[i] > 80) * 1.5 +
                (df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]) * 1.5 +
                (df['alma'].iloc[i] < df['alma'].iloc[i-1]) * 1 +
                (df['close'].iloc[i] < df['open'].iloc[i]) * 1 >= 7
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 83, 'name': 'Divergence Stack',
            'desc': 'Multiple divergences align: RSI + MACD + Price',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-5] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 25
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-5] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 75
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.5}
        },
        {
            'id': 84, 'name': 'Range Extreme with Momentum Shift',
            'desc': 'Price at range extreme with clear momentum reversal',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-2] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['stoch_k'].iloc[i] < 20 and df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-2] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['stoch_k'].iloc[i] > 80 and df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 85, 'name': 'Smart Money Reversion',
            'desc': 'Volume profile + price action + momentum alignment',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5 and
                df['rsi'].iloc[i] < 35 and
                df['macd_hist'].iloc[i] > 0
            ),
            'entry_short': lambda df, i: (
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5 and
                df['rsi'].iloc[i] > 65 and
                df['macd_hist'].iloc[i] < 0
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
    ])

    # ======================================================================
    # CATEGORY 8: DIVERGENCE BASED (86-95)
    # ======================================================================
    strategies.extend([
        {
            'id': 86, 'name': 'RSI-MACD Double Divergence',
            'desc': 'Both RSI and MACD diverge from price at extremes',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-5] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 25
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-5] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 75
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.5}
        },
        {
            'id': 87, 'name': 'Price-ALMA Divergence',
            'desc': 'Price diverges from ALMA slope direction',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-4] and
                df['alma'].iloc[i] > df['alma'].iloc[i-4] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-4] and
                df['alma'].iloc[i] < df['alma'].iloc[i-4] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 88, 'name': 'Volume-Price Divergence',
            'desc': 'Volume divergence at price extremes',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-3] and
                df['volume'].iloc[i] < df['volume'].iloc[i-3] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 35 and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-3] and
                df['volume'].iloc[i] < df['volume'].iloc[i-3] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 65 and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 89, 'name': 'Triple Divergence Master',
            'desc': 'Price, RSI, MACD, and StochRSI all diverge',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-5] and
                df['stoch_k'].iloc[i] > df['stoch_k'].iloc[i-5] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-5] and
                df['stoch_k'].iloc[i] < df['stoch_k'].iloc[i-5] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 4.0}
        },
        {
            'id': 90, 'name': 'Hidden Divergence Pro',
            'desc': 'Hidden divergence with trend continuation for MR',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] > df['low'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                df['alma'].iloc[i] > df['alma'].iloc[i-3] and
                df['close'].iloc[i] < df['alma'].iloc[i] and
                df['stoch_k'].iloc[i] < 30
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] < df['high'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                df['alma'].iloc[i] < df['alma'].iloc[i-3] and
                df['close'].iloc[i] > df['alma'].iloc[i] and
                df['stoch_k'].iloc[i] > 70
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 91, 'name': 'Divergence + Volume POC',
            'desc': 'Divergence at institutional volume level',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.4 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.4 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i]
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 92, 'name': 'Divergence Exhaustion',
            'desc': 'Multiple divergences showing exhaustion',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] < df['low'].iloc[i-3] and df['low'].iloc[i-3] < df['low'].iloc[i-6] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-3] and df['rsi'].iloc[i-3] > df['rsi'].iloc[i-6] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-3] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 20
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] > df['high'].iloc[i-3] and df['high'].iloc[i-3] > df['high'].iloc[i-6] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-3] and df['rsi'].iloc[i-3] < df['rsi'].iloc[i-6] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-3] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 80
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.5}
        },
        {
            'id': 93, 'name': 'BB Position Divergence',
            'desc': 'Price at BB extreme with internal divergence',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['bb_position'].iloc[i] < 0.05 and
                df['bb_position'].iloc[i] > df['bb_position'].iloc[i-3] and
                df['close'].iloc[i] < df['close'].iloc[i-3] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-3] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-3]
            ),
            'entry_short': lambda df, i: (
                df['bb_position'].iloc[i] > 0.95 and
                df['bb_position'].iloc[i] < df['bb_position'].iloc[i-3] and
                df['close'].iloc[i] > df['close'].iloc[i-3] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-3] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-3]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
        {
            'id': 94, 'name': 'Momentum Divergence Scalp',
            'desc': 'Quick divergence-based scalp on 5m',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['low'].iloc[i] <= df['low'].iloc[i-2] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-2] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-2] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['stoch_k'].iloc[i] < 25
            ),
            'entry_short': lambda df, i: (
                df['high'].iloc[i] >= df['high'].iloc[i-2] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-2] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-2] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['stoch_k'].iloc[i] > 75
            ),
            'config': {'time_exit_bars': 4, 'atr_multiplier_sl': 1.2, 'atr_multiplier_tp': 2.0}
        },
        {
            'id': 95, 'name': 'Divergence + Choppiness Confluence',
            'desc': 'Divergence only in optimal choppy conditions',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 70 and
                df['low'].iloc[i] < df['low'].iloc[i-5] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-5] and
                df['close'].iloc[i] < df['bb_lower'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['choppiness'].iloc[i] > 55 and df['choppiness'].iloc[i] < 70 and
                df['high'].iloc[i] > df['high'].iloc[i-5] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5] and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-5] and
                df['close'].iloc[i] > df['bb_upper'].iloc[i]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 3.0}
        },
    ])

    # ======================================================================
    # CATEGORY 9: ADVANCED/VOLATILITY ADJUSTED (96-100)
    # ======================================================================
    strategies.extend([
        {
            'id': 96, 'name': 'Adaptive ATR Mean Reversion',
            'desc': 'Dynamic SL/TP based on current vs historical ATR',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['atr'].iloc[i] < df['atr'].iloc[i-20:i].mean() * 1.2 and
                df['stoch_k'].iloc[i] < 20 and
                df['alma'].iloc[i] > df['alma'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['atr'].iloc[i] < df['atr'].iloc[i-20:i].mean() * 1.2 and
                df['stoch_k'].iloc[i] > 80 and
                df['alma'].iloc[i] < df['alma'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 97, 'name': 'Volatility Regime Switcher',
            'desc': 'Different logic for low vs normal volatility',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['atr'].iloc[i] < df['atr'].iloc[i-20:i].mean() * 0.8 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 25 and
                df['stoch_k'].iloc[i] < 15 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1]
            ),
            'entry_short': lambda df, i: (
                df['atr'].iloc[i] < df['atr'].iloc[i-20:i].mean() * 0.8 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 75 and
                df['stoch_k'].iloc[i] > 85 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1]
            ),
            'config': {'atr_multiplier_sl': 1.0, 'atr_multiplier_tp': 1.5, 'time_exit_bars': 4}
        },
        {
            'id': 98, 'name': 'News Avoidance + MR',
            'desc': 'Mean reversion with strict volatility filtering',
            'timeframe': ['M5', 'M15'],
            'entry_long': lambda df, i: (
                df['atr'].iloc[i] < df['atr'].iloc[i-10:i].mean() * 1.3 and
                df['atr'].iloc[i] > df['atr'].iloc[i-10:i].mean() * 0.7 and
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['stoch_k'].iloc[i] < 20 and
                df['volume'].iloc[i] < df['volume'].iloc[i-5:i].mean() * 2.0
            ),
            'entry_short': lambda df, i: (
                df['atr'].iloc[i] < df['atr'].iloc[i-10:i].mean() * 1.3 and
                df['atr'].iloc[i] > df['atr'].iloc[i-10:i].mean() * 0.7 and
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['stoch_k'].iloc[i] > 80 and
                df['volume'].iloc[i] < df['volume'].iloc[i-5:i].mean() * 2.0
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 99, 'name': 'Session-Aware Mean Reversion',
            'desc': 'MR strategy with session-based adjustments',
            'timeframe': ['M5'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['rsi'].iloc[i] < 30 and
                df['stoch_k'].iloc[i] < 20 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-10:i].mean() * 0.8
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['rsi'].iloc[i] > 70 and
                df['stoch_k'].iloc[i] > 80 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-10:i].mean() * 0.8
            ),
            'config': {'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 2.5}
        },
        {
            'id': 100, 'name': 'The Ultimate Mean Reversion',
            'desc': 'Maximum confluence: 6+ indicators, volume profile, strict filters',
            'timeframe': ['M15'],
            'entry_long': lambda df, i: (
                df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                df['close'].iloc[i] < df['alma'].iloc[i] - df['atr'].iloc[i] * 1.5 and
                df['rsi'].iloc[i] < 25 and
                df['stoch_k'].iloc[i] < 15 and
                df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] > df['macd_hist'].iloc[i-2] and
                df['alma'].iloc[i] > df['alma'].iloc[i-1] and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3 and
                df['close'].iloc[i] > df['open'].iloc[i]
            ),
            'entry_short': lambda df, i: (
                df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                df['close'].iloc[i] > df['alma'].iloc[i] + df['atr'].iloc[i] * 1.5 and
                df['rsi'].iloc[i] > 75 and
                df['stoch_k'].iloc[i] > 85 and
                df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] < df['macd_hist'].iloc[i-2] and
                df['alma'].iloc[i] < df['alma'].iloc[i-1] and
                abs(df['close'].iloc[i] - df['vp_poc'].iloc[i]) < df['atr'].iloc[i] * 0.5 and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.3 and
                df['close'].iloc[i] < df['open'].iloc[i]
            ),
            'config': {'vp_row_size': 48, 'atr_multiplier_sl': 1.5, 'atr_multiplier_tp': 4.0, 'time_exit_bars': 12}
        },
    ])

    return strategies


# ==============================================================================
# CSV DATA LOADER
# ==============================================================================

class DataLoader:
    @staticmethod
    def load_csv(filepath):
        """Load and standardize CSV data for backtesting"""
        df = pd.read_csv(filepath)

        # Standardize column names (handle common variations)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['open', 'o']:
                col_map[col] = 'open'
            elif col_lower in ['high', 'h', 'hi']:
                col_map[col] = 'high'
            elif col_lower in ['low', 'l', 'lo']:
                col_map[col] = 'low'
            elif col_lower in ['close', 'c', 'last']:
                col_map[col] = 'close'
            elif col_lower in ['volume', 'vol', 'tick_volume', 'real_volume']:
                col_map[col] = 'volume'
            elif col_lower in ['datetime', 'date', 'time', 'timestamp', 'date_time']:
                col_map[col] = 'datetime'

        df = df.rename(columns=col_map)

        # Parse datetime if present
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {filepath}. Found: {list(df.columns)}")

        # Ensure numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        print(f"Loaded {filepath}: {len(df)} rows, columns: {list(df.columns)}")
        return df

    @staticmethod
    def find_csv_files(directory='.'):
        """Find all CSV files in directory"""
        patterns = ['*.csv', '*_5m*.csv', '*_15m*.csv', '*M5*.csv', '*M15*.csv']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(str(Path(directory) / pattern)))

        # Remove duplicates and sort
        files = sorted(list(set(files)))

        if not files:
            print(f"WARNING: No CSV files found in {directory}")
            print("Looking in current directory and subdirectories...")
            files = glob.glob('**/*.csv', recursive=True)
            files = sorted(list(set(files)))

        return files


# ==============================================================================
# WALK-FORWARD ANALYSIS
# ==============================================================================

class WalkForwardAnalysis:
    """Walk-forward analysis to prevent overfitting"""

    @staticmethod
    def split_data(df, train_pct=0.7, n_windows=3):
        """Split data into training and testing windows"""
        n = len(df)
        window_size = n // n_windows
        splits = []

        for i in range(n_windows):
            start = i * window_size
            mid = start + int(window_size * train_pct)
            end = min((i + 1) * window_size, n)

            if end - start < 100:
                continue

            train_df = df.iloc[start:mid].copy()
            test_df = df.iloc[mid:end].copy()
            splits.append((train_df, test_df, i+1))

        return splits

    @staticmethod
    def run_walk_forward(df, strategy, config, initial_balance=10000.0):
        """Run walk-forward analysis for a single strategy"""
        splits = WalkForwardAnalysis.split_data(df)
        all_results = []

        for train_df, test_df, window_num in splits:
            if len(test_df) < 50:
                continue

            engine = BacktestEngine(config)
            result = engine.run_backtest(
                test_df, 
                strategy['entry_long'], 
                strategy['entry_short'],
                initial_balance
            )
            result['window'] = window_num
            result['train_size'] = len(train_df)
            result['test_size'] = len(test_df)
            all_results.append(result)

        # Aggregate results
        if not all_results:
            return None

        total_trades = sum(r['total_trades'] for r in all_results)
        if total_trades == 0:
            return None

        avg_wr = np.mean([r['win_rate'] for r in all_results if r['total_trades'] > 0])
        avg_pf = np.mean([r['profit_factor'] for r in all_results if r['total_trades'] > 0])
        total_profit = sum(r['net_profit'] for r in all_results)
        avg_drawdown = np.mean([r['max_drawdown_pct'] for r in all_results])

        # Consistency score (how consistent across windows)
        pf_values = [r['profit_factor'] for r in all_results if r['total_trades'] > 0]
        consistency = 1.0 - (np.std(pf_values) / np.mean(pf_values)) if np.mean(pf_values) > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': avg_wr,
            'profit_factor': avg_pf,
            'net_profit': total_profit,
            'max_drawdown_pct': avg_drawdown,
            'consistency_score': max(0, consistency) * 100,
            'window_results': all_results
        }


# ==============================================================================
# STRATEGY TESTING ORCHESTRATOR
# ==============================================================================

class StrategyTester:
    """Orchestrates testing of all strategies across all data files"""

    def __init__(self, data_directory='.', initial_balance=10000.0):
        self.data_directory = data_directory
        self.initial_balance = initial_balance
        self.results = []

    def run_all_tests(self, use_walk_forward=True):
        """Run all 100 strategies on all available CSV files"""

        # Find all CSV files
        csv_files = DataLoader.find_csv_files(self.data_directory)

        if not csv_files:
            print("\n" + "="*80)
            print("ERROR: No CSV files found!")
            print("="*80)
            print("Please place your CSV data files in the same directory as this script.")
            print("Expected format: columns = [datetime, open, high, low, close, volume]")
            print("="*80 + "\n")
            return []

        print(f"\nFound {len(csv_files)} CSV file(s):")
        for f in csv_files:
            print(f"  - {f}")
        print()

        # Get all strategies
        strategies = get_all_strategies()
        print(f"Loaded {len(strategies)} mean reversion strategies\n")

        all_results = []

        for csv_file in csv_files:
            try:
                df = DataLoader.load_csv(csv_file)
                if len(df) < 200:
                    print(f"Skipping {csv_file}: insufficient data ({len(df)} rows)")
                    continue

                # Detect timeframe from filename or data
                tf = self._detect_timeframe(csv_file, df)

                print(f"\n{'='*80}")
                print(f"Testing on: {csv_file} (Detected: {tf})")
                print(f"{'='*80}")

                for strategy in strategies:
                    # Skip if strategy not designed for this timeframe
                    if tf not in strategy['timeframe']:
                        continue

                    # Create config with strategy overrides
                    config = StrategyConfig()
                    config.name = strategy['name']
                    config.description = strategy['desc']
                    config.timeframe = TimeFrame.M5 if tf == 'M5' else TimeFrame.M15

                    # Apply strategy-specific config overrides
                    for key, value in strategy.get('config', {}).items():
                        setattr(config, key, value)

                    try:
                        if use_walk_forward:
                            result = WalkForwardAnalysis.run_walk_forward(
                                df, strategy, config, self.initial_balance
                            )
                        else:
                            engine = BacktestEngine(config)
                            result = engine.run_backtest(
                                df, strategy['entry_long'], strategy['entry_short'],
                                self.initial_balance
                            )

                        if result and result['total_trades'] > 5:
                            result['strategy_id'] = strategy['id']
                            result['strategy_name'] = strategy['name']
                            result['strategy_desc'] = strategy['desc']
                            result['timeframe'] = tf
                            result['file'] = csv_file
                            all_results.append(result)

                            print(f"  Strategy {strategy['id']:3d}: {strategy['name'][:40]:<40} | "
                                  f"Trades: {result['total_trades']:4d} | "
                                  f"WR: {result['win_rate']:5.1f}% | "
                                  f"PF: {result['profit_factor']:5.2f} | "
                                  f"Profit: ${result['net_profit']:8.2f}")
                    except Exception as e:
                        print(f"  Strategy {strategy['id']:3d}: ERROR - {str(e)[:50]}")

            except Exception as e:
                print(f"ERROR loading {csv_file}: {e}")
                continue

        self.results = all_results
        return all_results

    def _detect_timeframe(self, filepath, df):
        """Detect timeframe from filename or data characteristics"""
        fp_lower = filepath.lower()
        if '5m' in fp_lower or 'm5' in fp_lower or '_5' in fp_lower:
            return 'M5'
        elif '15m' in fp_lower or 'm15' in fp_lower or '_15' in fp_lower:
            return 'M15'

        # Try to detect from data frequency
        if len(df) > 1 and hasattr(df.index, 'freq') and df.index.freq:
            freq = str(df.index.freq)
            if '5' in freq:
                return 'M5'
            elif '15' in freq:
                return 'M15'

        # Default based on row count heuristic
        if len(df) > 5000:
            return 'M5'
        return 'M15'

    def rank_results(self):
        """Rank all results by Profit Factor and Win Rate"""
        if not self.results:
            print("No results to rank. Run tests first.")
            return pd.DataFrame()

        df_results = pd.DataFrame(self.results)

        # Filter out strategies with too few trades
        df_results = df_results[df_results['total_trades'] >= 10]

        if len(df_results) == 0:
            print("No strategies with sufficient trades to rank.")
            return pd.DataFrame()

        # Composite score: weighted combination of PF, WR, and profit
        df_results['pf_score'] = df_results['profit_factor'].clip(0, 10) / 10 * 40
        df_results['wr_score'] = df_results['win_rate'].clip(0, 100) / 100 * 30
        df_results['profit_score'] = (df_results['net_profit'] / df_results['net_profit'].abs().max()).clip(-1, 1) * 20 + 10

        if 'consistency_score' in df_results.columns:
            df_results['consistency_score'] = df_results['consistency_score'].fillna(0)
            df_results['composite_score'] = (df_results['pf_score'] + df_results['wr_score'] + 
                                            df_results['profit_score'] + df_results['consistency_score'] * 0.1)
        else:
            df_results['composite_score'] = df_results['pf_score'] + df_results['wr_score'] + df_results['profit_score']

        # Sort by composite score
        df_results = df_results.sort_values('composite_score', ascending=False)

        return df_results

    def print_rankings(self, top_n=20):
        """Print ranked results"""
        df_ranked = self.rank_results()

        if df_ranked.empty:
            return

        print("\n" + "="*120)
        print("STRATEGY RANKINGS - TOP PERFORMERS")
        print("="*120)
        print(f"{'Rank':<6}{'ID':<6}{'Strategy Name':<45}{'TF':<5}{'Trades':<8}{'WR%':<8}{'PF':<8}{'Profit':<12}{'DD%':<8}{'Score':<8}")
        print("-"*120)

        for idx, (_, row) in enumerate(df_ranked.head(top_n).iterrows(), 1):
            print(f"{idx:<6}{row['strategy_id']:<6}{row['strategy_name'][:44]:<45}{row['timeframe']:<5}"
                  f"{row['total_trades']:<8}{row['win_rate']:<8.1f}{row['profit_factor']:<8.2f}"
                  f"${row['net_profit']:<11.2f}{row['max_drawdown_pct']:<8.1f}{row['composite_score']:<8.1f}")

        print("="*120)

        # Save to CSV
        output_file = 'strategy_rankings.csv'
        df_ranked.to_csv(output_file, index=False)
        print(f"\nFull rankings saved to: {output_file}")
        
        # ADDITION: SAVE ALL WINNERS IN A TXT FILE
        # Filter "winners" based on positive net profit, PF > 1, etc.
        winners = df_ranked[
            (df_ranked['net_profit'] > 0) & 
            (df_ranked['profit_factor'] > 1.0) & 
            (df_ranked['win_rate'] >= 40.0)
        ]
        
        with open('all_winning_strategies.txt', 'w') as f:
            f.write("================================================================================\n")
            f.write("ALL WINNING STRATEGIES (Walk-Forward on 5M & 15M)\n")
            f.write("Criteria: Net Profit > 0, PF > 1.0, WR >= 40%\n")
            f.write("================================================================================\n\n")
            
            for idx, row in winners.iterrows():
                f.write(f"ID: {row['strategy_id']} | Name: {row['strategy_name']}\n")
                f.write(f"Description: {row['strategy_desc']}\n")
                f.write(f"Timeframe: {row['timeframe']} | File: {row['file']}\n")
                f.write(f"Win Rate: {row['win_rate']:.1f}% | Profit Factor: {row['profit_factor']:.2f}\n")
                f.write(f"Net Profit: ${row['net_profit']:.2f}\n")
                # Add sharpe ratio if available or simulated
                sharpe = row.get('sharpe', 0.0)
                f.write(f"Sharpe Ratio: {sharpe:.2f} (approx)\n")
                f.write("-" * 80 + "\n")
        
        print(f"Saved {len(winners)} winning strategies to all_winning_strategies.txt")

        # Save top strategy details
        if len(df_ranked) > 0:
            top = df_ranked.iloc[0]
            print(f"\n🏆 TOP STRATEGY: #{top['strategy_id']} - {top['strategy_name']}")
            print(f"   Description: {top['strategy_desc']}")
            print(f"   Timeframe: {top['timeframe']}")
            print(f"   Total Trades: {top['total_trades']}")
            print(f"   Win Rate: {top['win_rate']:.1f}%")
            print(f"   Profit Factor: {top['profit_factor']:.2f}")
            print(f"   Net Profit: ${top['net_profit']:.2f}")
            print(f"   Max Drawdown: {top['max_drawdown_pct']:.1f}%")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("="*80)
    print("ULTIMATE MEAN REVERSION BACKTESTING ENGINE")
    print("100 Strategies | 5m & 15m Forex Scalping")
    print("="*80)

    # Configuration
    DATA_DIRECTORY = '.'  # Current directory - place CSV files here
    INITIAL_BALANCE = 10000.0
    USE_WALK_FORWARD = True

    # Run tests
    tester = StrategyTester(DATA_DIRECTORY, INITIAL_BALANCE)
    tester.run_all_tests(use_walk_forward=USE_WALK_FORWARD)

    # Rank and display results
    tester.print_rankings(top_n=25)

    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)
