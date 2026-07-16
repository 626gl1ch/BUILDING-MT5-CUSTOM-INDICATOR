from mass_search_sequential import main
from rbo_v2 import GRIDS
import sys

# Force a crash in one strategy
def broken_strategy(df, p):
    raise ValueError("INTENTIONAL CRASH FOR LOGGING TEST!")

# Force a pass in another strategy by monkeypatching BacktestCore
from backtest_core import BacktestCore
original_run_full = BacktestCore.run_full_validation

def forced_pass_validation(self, all_dfs, strategy_fn, params, *args, **kwargs):
    if strategy_fn.__name__ == 'signal_ema_pullback':
        return {
            'passed': True,
            'params': params,
            'backtest': {'expectancy': 0.5, 'profit_factor': 2.0, 'win_rate': 60.0, 'sharpe_ratio': 1.5, 'trades_per_day': 1.0, 'total_trades': 100},
            'symbol_results': {},
            'walkforward': {'in_sample': {}, 'out_of_sample': {'expectancy': 0.4, 'sharpe_ratio': 1.2}},
            'permutation': {'p_value': 0.01},
            'gates': {}
        }
    return original_run_full(self, all_dfs, strategy_fn, params, *args, **kwargs)

# Inject
GRIDS['donchian_breakout']['fn'] = broken_strategy
BacktestCore.run_full_validation = forced_pass_validation

print("Running test sweep. We expect donchian to log an error and ema_pullback to instantly generate an MD file.")
# Temporarily shrink max_combos inside mass_search_sequential so it runs instantly
import mass_search_sequential
mass_search_sequential.get_param_combos = lambda d, max_combos=1: [list(mass_search_sequential.itertools.product(*[d[k] for k in d.keys()]))[0] for _ in range(1)]

try:
    mass_search_sequential.main()
except Exception as e:
    print(e)
