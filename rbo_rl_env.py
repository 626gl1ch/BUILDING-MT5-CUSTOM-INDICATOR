import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RboScalpingEnv(gym.Env):
    """
    Custom Environment for 5-minute Crypto Scalping that follows gym interface.
    Integrates directly with the precomputed DataFrames from indicators_library.py.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission=0.0005,
                 holding_penalty=0.0001, sl_atr_mult=2.5, tp_atr_mult=5.0):
        super(RboScalpingEnv, self).__init__()

        self.df = df.copy()
        # Drop rows with NaNs to ensure clean state
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        self.commission = commission
        self.holding_penalty = holding_penalty
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # Actions: 0: Flat, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)

        # Build feature matrix
        self._build_features()
        self.n_features = self.features.shape[1] + 2 # +2 for current position & unrealized PnL

        # Observation space boundaries (-inf, inf for safety, though scaled)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

        self.reset()

    def _build_features(self):
        """Extracts and normalizes features from the dataframe."""
        df = self.df
        features = pd.DataFrame(index=df.index)

        # Price momentum
        features['ret_1'] = df['close'].pct_change().fillna(0) * 100
        features['ret_5'] = df['close'].pct_change(5).fillna(0) * 100
        
        # Volatility / Range
        features['zscore'] = df['zscore_20'].fillna(0) / 3.0  # normalize ~[-1, 1]
        
        # Trend / Momentum Indicators
        if 'rsi_14' in df.columns:
            features['rsi'] = (df['rsi_14'] - 50) / 50.0
        else:
            features['rsi'] = 0.0
            
        if 'adx_14' in df.columns:
            features['adx'] = df['adx_14'] / 100.0
        else:
            features['adx'] = 0.0
            
        if 'macd_hist_fast' in df.columns:
            # Scale MACD hist relative to price
            features['macd'] = (df['macd_hist_fast'] / df['close']) * 1000
        else:
            features['macd'] = 0.0

        if 'vwap' in df.columns:
            features['vwap_dist'] = (df['close'] - df['vwap']) / df['close'] * 100
        else:
            features['vwap_dist'] = 0.0

        if 'supertrend_dir_10_3' in df.columns:
            features['st_dir'] = df['supertrend_dir_10_3'] # already 1 or -1
        else:
            features['st_dir'] = 0.0

        if 'atr_14' in df.columns:
            self.atr_arr = df['atr_14'].values
        else:
            self.atr_arr = (df['high'] - df['low']).values

        # Save prices for PnL calculation
        self.close_arr = df['close'].values
        self.features = features.values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0 # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.trades = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Current row features
        obs = self.features[self.current_step].copy()
        
        # Add position info
        unrealized_pnl = 0.0
        if self.position != 0:
            current_price = self.close_arr[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            if self.position == -1:
                unrealized_pnl = -unrealized_pnl
            
            # Normalize PnL roughly to ATR terms
            atr_pct = self.atr_arr[self.current_step] / current_price
            if atr_pct > 0:
                unrealized_pnl = unrealized_pnl / atr_pct
        
        # Append internal state (position logic)
        obs = np.append(obs, [self.position, unrealized_pnl])
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0.0
        current_price = self.close_arr[self.current_step]
        atr = self.atr_arr[self.current_step]
        
        # Map gym action (0,1,2) to trading position (0, 1, -1)
        target_pos = 0
        if action == 1: target_pos = 1
        elif action == 2: target_pos = -1

        # Check for SL/TP hits BEFORE taking new action if we are already in a position
        # (simplified: checking at close price for speed)
        if self.position != 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if self.position == -1:
                pnl_pct = -pnl_pct
            
            # Stop Loss
            if pnl_pct < -self.sl_atr_mult * (atr / self.entry_price):
                target_pos = 0 # Force exit
            # Take Profit
            elif pnl_pct > self.tp_atr_mult * (atr / self.entry_price):
                target_pos = 0 # Force exit

        # Execute Position Change
        if target_pos != self.position:
            # Close existing position
            if self.position != 0:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if self.position == -1:
                    pnl_pct = -pnl_pct
                
                # Apply commission on closing
                pnl_pct -= self.commission
                reward += pnl_pct * 100 # scale reward up slightly
            
            # Open new position
            if target_pos != 0:
                # Apply commission on opening
                reward -= self.commission * 100
                self.entry_price = current_price
                self.trades += 1

            self.position = target_pos
        else:
            # Holding penalty to discourage waiting forever
            if self.position != 0:
                reward -= self.holding_penalty

        self.current_step += 1
        
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        # Prevent index out of bounds on the final step
        if terminated:
            self.current_step = self.n_steps - 1
            
        obs = self._get_observation()
        info = {'position': self.position, 'trades': self.trades}
        
        return obs, float(reward), terminated, truncated, info
