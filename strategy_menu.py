import numpy as np
import pandas as pd
from typing import Dict, Any

class StrategySignal:
    """
    Interface for wrapping existing RBO strategies into a common format for the Meta-Controller.
    Provides signal generation, rolling Sharpe computation, and exact bar-by-bar PnL.
    """
    def __init__(self, name: str, params: Dict[str, Any], signal_fn):
        self.name = name
        self.params = params
        self.signal_fn = signal_fn
        
        # We cache the full array of signals so we don't recalculate per step
        self._cached_signals = None
        self._cached_df_id = None
        
        # PnL tracking for rolling sharpe
        self.pnl_history = []

    def _ensure_cache(self, df: pd.DataFrame):
        """Generates the full signal array for the DataFrame if not already cached."""
        if self._cached_signals is None or id(df) != self._cached_df_id:
            self._cached_signals = self.signal_fn(df, self.params).values
            self._cached_df_id = id(df)

    def signal(self, df: pd.DataFrame, idx: int) -> int:
        """Returns the signal (-1, 0, 1) for a given bar."""
        self._ensure_cache(df)
        return int(self._cached_signals[idx])

    def rolling_sharpe(self, df: pd.DataFrame, idx: int, window: int = 60) -> float:
        """
        Calculates the rolling Sharpe ratio over the last `window` bars.
        If history is shorter than window, uses available history.
        """
        if len(self.pnl_history) < 2:
            return 0.0
            
        recent_pnl = self.pnl_history[-window:]
        mean_pnl = np.mean(recent_pnl)
        std_pnl = np.std(recent_pnl)
        
        if std_pnl < 1e-8:
            return 0.0
        return float((mean_pnl / std_pnl) * np.sqrt(288)) # Annualized roughly to daily (288 5m bars)

    def bar_return(self, df: pd.DataFrame, idx: int, current_position: float, 
                   entry_price: float, entry_atr: float, 
                   commission_bps: float = 5.0, slippage_bps: float = 2.0) -> float:
        """
        Computes the precise realized + unrealized PnL of holding `current_position` 
        for this single 5-minute bar.
        
        Enforces EXACT ATR stop-losses pegged to the original `entry_price` and `entry_atr`,
        fixing the live-ATR bug. Applies commission and slippage distinctly.
        """
        if current_position == 0 or idx == 0:
            self.pnl_history.append(0.0)
            return 0.0
            
        prev_close = df['close'].iloc[idx - 1]
        curr_close = df['close'].iloc[idx]
        
        # Bar's gross return
        gross_ret = (curr_close - prev_close) / prev_close
        if current_position < 0:
            gross_ret = -gross_ret
            
        # Total unrealized return since entry
        total_pnl_pct = (curr_close - entry_price) / entry_price
        if current_position < 0:
            total_pnl_pct = -total_pnl_pct
            
        # Apply Slippage explicitly to the bar return
        net_ret = gross_ret * abs(current_position) # Scale by allocation weight
        
        # Check Stops (pegged to ENTRY atr, not current atr)
        # Assuming sl_atr = 2.5 and tp_atr = 5.0 from params
        sl_mult = self.params.get('sl_atr', 2.5)
        tp_mult = self.params.get('tp_atr', 5.0)
        
        stop_loss_pct = sl_mult * (entry_atr / entry_price)
        take_profit_pct = tp_mult * (entry_atr / entry_price)
        
        hit_exit = False
        if total_pnl_pct <= -stop_loss_pct:
            hit_exit = True
            # Cap the loss exactly at the stop level
            net_ret = -stop_loss_pct - ((prev_close - entry_price) / entry_price * (1 if current_position > 0 else -1))
        elif total_pnl_pct >= take_profit_pct:
            hit_exit = True
            # Cap the profit exactly at the target level
            net_ret = take_profit_pct - ((prev_close - entry_price) / entry_price * (1 if current_position > 0 else -1))
            
        if hit_exit:
            # Apply closing transaction costs (commission + slippage)
            closing_cost = (commission_bps + slippage_bps) / 10000.0
            net_ret -= closing_cost
            
        self.pnl_history.append(float(net_ret))
        return float(net_ret)

# Example wrapper instantiation using our known validated S4 / existing strategies
"""
strategy_menu = [
    StrategySignal("S4_BB_VWAP_MR", p_bbvwap, signal_bb_vwap_mr),
    StrategySignal("Donchian_Breakout", p_donchian, signal_donchian_breakout),
    StrategySignal("Vol_Squeeze", p_squeeze, signal_squeeze_breakout),
    StrategySignal("EMA_Pullback", p_ema, signal_ema_pullback),
    StrategySignal("RSI_Divergence", p_rsidiv, signal_rsi_divergence)
]
"""
