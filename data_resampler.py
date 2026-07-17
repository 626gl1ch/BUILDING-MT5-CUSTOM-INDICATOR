import pandas as pd
import numpy as np
import os
import glob
from logger_config import logger

def validate_data(df: pd.DataFrame, file_name: str) -> dict:
    """
    Scans CSV data for duplicate timestamps, missing bars, and bad prices before resampling.
    """
    df = df.copy()
    if 'datetime' in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    df = df.sort_values("timestamp")
    
    # 1. Check for duplicates
    duplicate_count = df.duplicated(subset=["timestamp"]).sum()
    
    # 2. Check for missing bars (assuming 5m data)
    # Calculate expected bars
    time_diffs = df["timestamp"].diff().dt.total_seconds() / 60
    missing_gaps = time_diffs[time_diffs > 5]
    
    # 3. Check for bad prices
    bad_prices = ((df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)).sum()
    
    # Extreme outliers (e.g., high is 10x the median close)
    median_close = df["close"].median()
    extreme_outliers = (df["high"] > median_close * 10) | (df["low"] < median_close * 0.1)
    
    report = {
        "file_name": file_name,
        "date_range": f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}",
        "total_bars": len(df),
        "duplicates": int(duplicate_count),
        "gaps_longer_than_5m": len(missing_gaps),
        "max_gap_minutes": float(missing_gaps.max()) if len(missing_gaps) > 0 else 0,
        "bad_prices_zeros_or_neg": int(bad_prices),
        "extreme_outliers": int(extreme_outliers.sum())
    }
    return report

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples lower timeframe OHLCV data to strict bars of the specified timeframe (e.g., '15min', '30min', '1H').
    Drops any incomplete bars to prevent lookahead bias.
    """
    df = df.copy()
    if 'datetime' in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    elif 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    df = df.set_index("timestamp").sort_index()

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }

    # Resample strictly to the target timeframe, labeled by the left boundary
    # In pandas, '1H' works for hours, '15min' for 15 minutes, etc.
    if timeframe == '1H':
        pandas_tf = '1H'
        expected_bars = 12 # 12 * 5min = 1H
    elif timeframe == '30min':
        pandas_tf = '30min'
        expected_bars = 6
    elif timeframe == '15min':
        pandas_tf = '15min'
        expected_bars = 3
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    resampled = df.resample(pandas_tf, label="left", closed="left").agg(agg_rules)

    # Count the number of source bars per resampled bar to drop incomplete ones
    bar_counts = df['close'].resample(pandas_tf, label="left", closed="left").count()
    
    # Keep the bar if it has ANY data, but we drop rows with NaNs.
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    
    # Add the source bar count for reconciliation
    resampled['source_bar_count'] = bar_counts
    
    # Flag low confidence bars (missing too many source 5m bars)
    min_required = int(expected_bars * 0.8) # 80% of expected bars required
    resampled['is_low_confidence'] = resampled['source_bar_count'] < min_required

    return resampled.reset_index()

def reconcile_resample(source_df: pd.DataFrame, resampled_df: pd.DataFrame, timeframe: str) -> bool:
    """
    Cross-validates a random 1hr bar by manually aggregating the source data.
    """
    if len(resampled_df) == 0:
        return False
        
    # Pick a random bar that has full data
    valid_bars = resampled_df[resampled_df['is_low_confidence'] == False]
    if len(valid_bars) == 0:
        sample_time = resampled_df.iloc[len(resampled_df) // 2]['timestamp']
    else:
        sample_time = valid_bars.sample(1).iloc[0]['timestamp']
        
    window_start = sample_time
    if timeframe == '1H':
        window_end = sample_time + pd.Timedelta(hours=1)
    elif timeframe == '30min':
        window_end = sample_time + pd.Timedelta(minutes=30)
    elif timeframe == '15min':
        window_end = sample_time + pd.Timedelta(minutes=15)
    
    if 'datetime' in source_df.columns:
        source_ts = pd.to_datetime(source_df["datetime"])
    else:
        source_ts = pd.to_datetime(source_df["timestamp"])

    source_slice = source_df[(source_ts >= window_start) & (source_ts < window_end)]

    if len(source_slice) == 0:
        return False

    expected_open = source_slice.iloc[0]["open"]
    expected_high = source_slice["high"].max()
    expected_low = source_slice["low"].min()
    expected_close = source_slice.iloc[-1]["close"]
    expected_vol = source_slice["volume"].sum()
    
    actual = resampled_df[resampled_df["timestamp"] == sample_time].iloc[0]
    
    match = (
        np.isclose(expected_open, actual["open"]) and
        np.isclose(expected_high, actual["high"]) and
        np.isclose(expected_low, actual["low"]) and
        np.isclose(expected_close, actual["close"]) and
        np.isclose(expected_vol, actual["volume"])
    )
    
    if not match:
        logger.warning(f"Reconciliation FAILED for {sample_time}")
        logger.warning(f"Expected: O={expected_open}, H={expected_high}, L={expected_low}, C={expected_close}, V={expected_vol}")
        logger.warning(f"Actual:   O={actual['open']}, H={actual['high']}, L={actual['low']}, C={actual['close']}, V={actual['volume']}")
    return match

def run_pipeline():
    files = glob.glob("*_5min_*.csv")
    if not files:
        logger.warning("No 5min CSV files found.")
        return

    reports = []
    
    logger.info("=== DATA VALIDATION ===")
    for f in files:
        logger.info(f"Validating {f}...")
        df = pd.read_csv(f)
        report = validate_data(df, f)
        reports.append(report)
        for k, v in report.items():
            logger.debug(f"  {k}: {v}")
            
        target_tfs = ['15min', '30min', '1H']
        for tf in target_tfs:
            out_name = f.replace("5min", tf)
            if os.path.exists(out_name):
                logger.info(f"  Skipping {tf} for {f} (already exists).")
                continue
                
            logger.info(f"  Resampling {f} to {tf}...")
            resampled_df = resample_data(df, tf)
            
            # Reconciliation
            passed = reconcile_resample(df, resampled_df, tf)
            logger.info(f"  Reconciliation Check ({tf}): {'PASS' if passed else 'FAIL'}")
            
            # Save output
            save_df = resampled_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source_bar_count', 'is_low_confidence']].copy()
            save_df = save_df.rename(columns={'timestamp': 'datetime'})
            save_df.to_csv(out_name, index=False)
            logger.info(f"  Saved {out_name}\n")
        
    # Write validation report to file
    report_df = pd.DataFrame(reports)
    report_df.to_csv("data_validation_report.csv", index=False)
    logger.info("Validation report saved to data_validation_report.csv")

if __name__ == "__main__":
    run_pipeline()
