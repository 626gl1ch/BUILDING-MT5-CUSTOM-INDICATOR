import pandas as pd
import numpy as np
import os
import glob
from logger_config import logger

def generate_synthetic_data(source_csv: str, output_csv: str, method: str = 'bootstrap'):
    """
    Generates an 'Authentic Synthetic' price path.
    Method 'bootstrap': Calculates log returns from the source asset, randomly samples them 
    with replacement, and reconstructs a completely new, randomized price path that shares 
    the exact same statistical properties (volatility, drift) as the original asset.
    """
    logger.info(f"Generating synthetic data from {source_csv} using {method}...")
    df = pd.read_csv(source_csv)
    
    if 'datetime' in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    elif 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    if method == 'bootstrap':
        # Calculate log returns
        log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Randomly sample log returns with replacement
        np.random.seed(42) # Fixed seed for reproducibility in testing
        sampled_returns = np.random.choice(log_returns, size=len(df)-1, replace=True)
        
        # Reconstruct price path
        synthetic_close = [df['close'].iloc[0]]
        for ret in sampled_returns:
            synthetic_close.append(synthetic_close[-1] * np.exp(ret))
            
        # Reconstruct OHLC based on the original bar's relative volatility
        # We find the relative O, H, L to C in the original and apply to the synthetic
        # To avoid negative prices in extreme cases, we use ratios
        open_ratio = df['open'] / df['close']
        high_ratio = df['high'] / df['close']
        low_ratio = df['low'] / df['close']
        
        # Shuffle these ratios independently to randomize bar shapes
        np.random.shuffle(open_ratio.values)
        np.random.shuffle(high_ratio.values)
        np.random.shuffle(low_ratio.values)
        
        synth_df = pd.DataFrame({
            'datetime': df['timestamp'],
            'close': synthetic_close,
        })
        
        synth_df['open'] = synth_df['close'] * open_ratio
        synth_df['high'] = synth_df['close'] * high_ratio
        synth_df['low'] = synth_df['close'] * low_ratio
        
        # Fix highs and lows to be true max/min of the bar
        synth_df['real_high'] = synth_df[['open', 'close', 'high']].max(axis=1)
        synth_df['real_low'] = synth_df[['open', 'close', 'low']].min(axis=1)
        
        synth_df['high'] = synth_df['real_high']
        synth_df['low'] = synth_df['real_low']
        synth_df = synth_df.drop(columns=['real_high', 'real_low'])
        
        # Keep original volume for simplicity (or shuffle it)
        shuffled_volume = df['volume'].values.copy()
        np.random.shuffle(shuffled_volume)
        synth_df['volume'] = shuffled_volume
        
        if 'source_bar_count' in df.columns:
            synth_df['source_bar_count'] = df['source_bar_count']
        if 'is_low_confidence' in df.columns:
            synth_df['is_low_confidence'] = df['is_low_confidence']
            
        synth_df.to_csv(output_csv, index=False)
        logger.info(f"Saved synthetic dataset to {output_csv}")
    else:
        logger.error(f"Unknown generation method: {method}")

if __name__ == '__main__':
    # Generate synthetic 1H data for testing
    source_files = glob.glob("*_1H_*.csv")
    for f in source_files:
        if "SYNTHETIC" in f: continue
        out_name = f.replace(".csv", "_SYNTHETIC.csv")
        if not os.path.exists(out_name):
            generate_synthetic_data(f, out_name)
        else:
            logger.info(f"Skipping {out_name}, already exists.")
