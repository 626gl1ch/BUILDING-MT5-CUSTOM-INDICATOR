import os
import json
import pandas as pd
import hashlib

class StrategyKnowledgeBase:
    """
    Persistent ML Strategy Database.
    Logs parameters and synthetic test outcomes to generate a massive training
    dataset for a future Meta-Labeling Machine Learning Model.
    """
    def __init__(self, db_path="strategy_learning_db.csv"):
        self.db_path = db_path
        self.cache = set()
        self._load_db()

    def _generate_hash(self, params):
        # Create a deterministic string from the dictionary
        # Convert lists to tuples to be hashable if needed, though they shouldn't be lists here
        # Parameters passed to backtest are scalar.
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                df = pd.read_csv(self.db_path)
                if 'param_hash' in df.columns:
                    self.cache = set(df['param_hash'].tolist())
            except Exception as e:
                print(f"[KnowledgeBase] Error loading DB: {e}")

    def has_been_tested(self, params):
        """Returns True if this exact parameter combination has been tested before."""
        phash = self._generate_hash(params)
        return phash in self.cache

    def log_result(self, params, metrics, passed_synthetic):
        """Logs the strategy parameters and its synthetic test results."""
        phash = self._generate_hash(params)
        
        # Don't log duplicates
        if phash in self.cache:
            return
            
        row = {
            'param_hash': phash,
            'entry': params.get('entry', ''),
            'regime': params.get('regime', ''),
            'sl_atr': params.get('sl_atr', 0),
            'tp_atr': params.get('tp_atr', 0),
            'risk_pct': params.get('risk_pct', 0),
            'trailing': params.get('trailing', False),
            'passed_synthetic': passed_synthetic,
            'expectancy': metrics.get('expectancy', 0),
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'total_trades': metrics.get('total_trades', 0),
        }
        
        df = pd.DataFrame([row])
        if not os.path.exists(self.db_path):
            df.to_csv(self.db_path, index=False)
        else:
            df.to_csv(self.db_path, mode='a', header=False, index=False)
            
        self.cache.add(phash)
