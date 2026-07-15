import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional
from collections import namedtuple
import pandas as pd

from strategy_menu import StrategySignal
from regime_tagger import classify_market_regime

# ---------------------------------------------------------------------------
# 1. STATE SPEC
# ---------------------------------------------------------------------------

@dataclass
class MetaState:
    regime_id: int              
    strategy_signals: np.ndarray  
    strategy_recent_sharpe: np.ndarray  
    atr_normalized: float        
    current_exposure: float      
    bars_since_last_trade: int
    time_of_day_bucket: int      

    N_REGIMES: int = 4           
    N_STRATEGIES: int = 5        
    N_TOD_BUCKETS: int = 4

    def to_vector(self) -> np.ndarray:
        regime_oh = np.eye(self.N_REGIMES)[self.regime_id]
        tod_oh = np.eye(self.N_TOD_BUCKETS)[self.time_of_day_bucket]
        return np.concatenate([
            regime_oh,
            self.strategy_signals,
            self.strategy_recent_sharpe,
            [self.atr_normalized],
            [self.current_exposure],
            [np.tanh(self.bars_since_last_trade / 20.0)],
            tod_oh,
        ]).astype(np.float32)

    @classmethod
    def dim(cls) -> int:
        return cls.N_REGIMES + cls.N_STRATEGIES * 2 + 3 + cls.N_TOD_BUCKETS


# ---------------------------------------------------------------------------
# 2. ACTION SPEC
# ---------------------------------------------------------------------------
ALLOCATION_LEVELS = np.array([0.0, 1.0]) # Changed to binary to shrink action space from 243 to 32
N_STRATEGIES = 5
N_ACTIONS = len(ALLOCATION_LEVELS) ** N_STRATEGIES 

def decode_action(action_idx: int) -> np.ndarray:
    idxs = []
    n = action_idx
    for _ in range(N_STRATEGIES):
        idxs.append(n % len(ALLOCATION_LEVELS))
        n //= len(ALLOCATION_LEVELS)
    return ALLOCATION_LEVELS[idxs]


# ---------------------------------------------------------------------------
# 3. REWARD SPEC
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    sharpe_window: int = 60          
    drawdown_penalty_weight: float = 2.0
    commission_bps: float = 5.0      
    slippage_bps: float = 2.0
    turnover_penalty_weight: float = 0.1
    use_differential_sharpe: bool = True

class RewardShaper:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.A = 0.0  
        self.B = 0.0  
        self.eta = 1.0 / cfg.sharpe_window

    def reset(self):
        self.A = 0.0
        self.B = 0.0

    def step(self, pnl_return: float, prev_alloc: np.ndarray, new_alloc: np.ndarray,
              running_drawdown: float) -> float:
        cfg = self.cfg
        
        # Turnover penalty
        turnover = np.abs(new_alloc - prev_alloc).sum()
        
        # Note: Slippage and Commission are already handled strictly inside 
        # StrategySignal.bar_return() for accurate PnL. 
        # But we still penalize the meta-controller for excessive swapping.
        net_return = pnl_return 

        if cfg.use_differential_sharpe:
            dA = net_return - self.A
            dB = net_return ** 2 - self.B
            denom = (self.B - self.A ** 2) ** 1.5
            dsr = 0.0 if abs(denom) < 1e-8 else (self.B * dA - 0.5 * self.A * dB) / denom
            self.A += self.eta * dA
            self.B += self.eta * dB
            reward = dsr
        else:
            reward = net_return

        reward -= cfg.drawdown_penalty_weight * max(0.0, running_drawdown)
        reward -= cfg.turnover_penalty_weight * turnover
        return float(reward)


# ---------------------------------------------------------------------------
# 4. ENVIRONMENT 
# ---------------------------------------------------------------------------

class TradingMetaEnv:
    def __init__(self, historical_data: pd.DataFrame, strategy_menu: list[StrategySignal], 
                 reward_cfg: RewardConfig, regime_tagger: Callable, episode_len: int = 2000):
        self.data = historical_data.copy()
        self.data.reset_index(drop=True, inplace=True) # Ensure clean index for idx lookups
        
        self.strategy_menu = strategy_menu    
        self.regime_tagger = regime_tagger    
        self.reward_shaper = RewardShaper(reward_cfg)
        self.episode_len = episode_len
        self.reset()

    def reset(self, start_idx: Optional[int] = None):
        self.reward_shaper.reset()
        self.idx = start_idx if start_idx is not None else np.random.randint(
            0, len(self.data) - self.episode_len)
        self.start_idx = self.idx
        self.current_alloc = np.zeros(N_STRATEGIES)
        
        # Track entry prices and ATRs per strategy for accurate ATR stops
        self.strategy_entries = [{'price': 0.0, 'atr': 0.0, 'pos': 0.0} for _ in range(N_STRATEGIES)]
        
        self.equity_curve = [1.0]
        self.peak_equity = 1.0
        self.bars_since_trade = 0
        return self._build_state()

    def _get_time_of_day_bucket(self, idx: int) -> int:
        if isinstance(self.data.index, pd.DatetimeIndex):
            hour = self.data.index[idx].hour
        elif 'time' in self.data.columns and isinstance(self.data['time'].iloc[0], pd.Timestamp):
            hour = self.data['time'].iloc[idx].hour
        else:
            return 0 # Fallback
            
        if hour < 6: return 0
        elif hour < 12: return 1
        elif hour < 18: return 2
        else: return 3

    def _build_state(self) -> MetaState:
        regime_id = self.regime_tagger(self.data, self.idx)
        signals = np.array([s.signal(self.data, self.idx) for s in self.strategy_menu])
        sharpes = np.array([s.rolling_sharpe(self.data, self.idx) for s in self.strategy_menu])
        
        atr = self.data['atr_14'].iloc[self.idx] if 'atr_14' in self.data.columns else 1.0
        close = self.data['close'].iloc[self.idx]
        atr_norm = atr / close if close > 0 else 0.0
        
        tod_bucket = self._get_time_of_day_bucket(self.idx)
        
        return MetaState(
            regime_id=regime_id,
            strategy_signals=signals,
            strategy_recent_sharpe=sharpes,
            atr_normalized=atr_norm,
            current_exposure=float(self.current_alloc.sum() / N_STRATEGIES),
            bars_since_last_trade=self.bars_since_trade,
            time_of_day_bucket=tod_bucket,
        )

    def _get_bar_pnl(self, new_alloc: np.ndarray) -> float:
        total_pnl = 0.0
        curr_close = self.data['close'].iloc[self.idx]
        curr_atr = self.data['atr_14'].iloc[self.idx] if 'atr_14' in self.data.columns else 1.0
        
        for i, strat in enumerate(self.strategy_menu):
            # Check if position changed to set entry anchors
            if new_alloc[i] != self.strategy_entries[i]['pos']:
                if new_alloc[i] > 0: # Opening/updating
                    self.strategy_entries[i]['price'] = curr_close
                    self.strategy_entries[i]['atr'] = curr_atr
                self.strategy_entries[i]['pos'] = new_alloc[i]
                
            pnl = strat.bar_return(
                df=self.data, 
                idx=self.idx, 
                current_position=new_alloc[i],
                entry_price=self.strategy_entries[i]['price'],
                entry_atr=self.strategy_entries[i]['atr'],
                commission_bps=self.reward_shaper.cfg.commission_bps,
                slippage_bps=self.reward_shaper.cfg.slippage_bps
            )
            total_pnl += pnl
            
        return float(total_pnl / max(new_alloc.sum(), 1e-8) if new_alloc.sum() > 0 else 0.0)

    def step(self, action_idx: int):
        new_alloc = decode_action(action_idx)
        pnl = self._get_bar_pnl(new_alloc)

        self.equity_curve.append(self.equity_curve[-1] * (1 + pnl))
        self.peak_equity = max(self.peak_equity, self.equity_curve[-1])
        drawdown = 1 - self.equity_curve[-1] / self.peak_equity

        reward = self.reward_shaper.step(pnl, self.current_alloc, new_alloc, drawdown)

        self.bars_since_trade = 0 if not np.array_equal(new_alloc, self.current_alloc) \
            else self.bars_since_trade + 1
        self.current_alloc = new_alloc
        self.idx += 1

        done = (self.idx - self.start_idx) >= self.episode_len or self.idx >= len(self.data) - 1
        next_state = self._build_state() if not done else None
        info = {"pnl": pnl, "drawdown": drawdown, "equity": self.equity_curve[-1]}
        return next_state, reward, done, info


# ---------------------------------------------------------------------------
# 5. OFFLINE DATASET BUILDER
# ---------------------------------------------------------------------------

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class OfflineDatasetBuilder:
    def __init__(self, env: TradingMetaEnv, behavior_policy: Callable[[MetaState], int]):
        self.env = env
        self.behavior_policy = behavior_policy

    def collect(self, n_episodes: int) -> list[Transition]:
        transitions = []
        for ep in range(n_episodes):
            state = self.env.reset(start_idx=self.env.episode_len * ep)  
            done = False
            while not done:
                action = self.behavior_policy(state)
                next_state, reward, done, info = self.env.step(action)
                transitions.append(Transition(state.to_vector(), action, reward,
                                               next_state.to_vector() if next_state else None,
                                               done))
                state = next_state
        return transitions

def random_behavior_policy(state: MetaState) -> int:
    return np.random.randint(0, N_ACTIONS)


# ---------------------------------------------------------------------------
# 6. IQL (Implicit Q-Learning)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class IQLConfig:
    state_dim: int
    n_actions: int
    gamma: float = 0.99
    tau: float = 0.7          
    beta: float = 3.0         
    lr: float = 3e-4
    hidden: int = 128

class IQL:
    def __init__(self, cfg: IQLConfig, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.q_net = MLP(cfg.state_dim, cfg.n_actions, cfg.hidden).to(device)
        self.q_target = MLP(cfg.state_dim, cfg.n_actions, cfg.hidden).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.v_net = MLP(cfg.state_dim, 1, cfg.hidden).to(device)
        self.policy_net = MLP(cfg.state_dim, cfg.n_actions, cfg.hidden).to(device)

        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=cfg.lr)
        self.pi_opt = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

    @staticmethod
    def expectile_loss(diff, tau):
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * diff ** 2).mean()

    def update(self, batch):
        s, a, r, s_next, done = batch
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        
        s_next_valid = torch.as_tensor(
            np.where(done.cpu().numpy()[:, None] == 1, 0, s_next if s_next is not None else 0),
            dtype=torch.float32, device=self.device
        ) if s_next is not None else torch.zeros_like(s)

        with torch.no_grad():
            q_target_all = self.q_target(s)
            q_sa = q_target_all.gather(1, a.unsqueeze(1)).squeeze(1)

        # Value update
        v_pred = self.v_net(s).squeeze(1)
        v_loss = self.expectile_loss(q_sa - v_pred, self.cfg.tau)
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # Q update
        with torch.no_grad():
            v_next = self.v_net(s_next_valid).squeeze(1)
            q_target_val = r + self.cfg.gamma * (1 - done) * v_next
        q_all = self.q_net(s)
        q_pred = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_pred, q_target_val)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # Policy update
        with torch.no_grad():
            adv = q_sa - v_pred
            weights = torch.exp(self.cfg.beta * adv).clamp(max=100.0)
        logits = self.policy_net(s)
        log_probs = F.log_softmax(logits, dim=1).gather(1, a.unsqueeze(1)).squeeze(1)
        pi_loss = -(weights * log_probs).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        # Soft target update
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                tp.data.mul_(0.995).add_(0.005 * p.data)

        return {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "pi_loss": pi_loss.item()}

    def act(self, state_vec: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.policy_net(s)
            return int(torch.argmax(logits, dim=1).item())

def make_batches(transitions: list[Transition], batch_size: int, state_dim: int):
    n = len(transitions)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    for start in range(0, n - batch_size, batch_size):
        batch_idx = idxs[start:start + batch_size]
        s = np.stack([transitions[i].state for i in batch_idx])
        a = np.array([transitions[i].action for i in batch_idx])
        r = np.array([transitions[i].reward for i in batch_idx], dtype=np.float32)
        s_next = np.stack([
            transitions[i].next_state if transitions[i].next_state is not None
            else np.zeros(state_dim, dtype=np.float32)
            for i in batch_idx
        ])
        done = np.array([transitions[i].done for i in batch_idx], dtype=np.float32)
        yield s, a, r, s_next, done
