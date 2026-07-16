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

def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples lower timeframe OHLCV data to 1H strict bars.
    Drops any incomplete hour bars to prevent lookahead bias.
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

    # Resample strictly to 1H, labeled by the left boundary (e.g., 00:00-00:59 is labeled 00:00)
    hourly = df.resample("1H", label="left", closed="left").agg(agg_rules)

    # Count the number of source bars per hour to drop incomplete hours
    # (assuming 5m data, a complete hour should have 12 bars)
    bar_counts = df['close'].resample("1H", label="left", closed="left").count()
    
    # Keep the hour if it has ANY data, but we drop rows with NaNs.
    # Note: If an hour has 0 bars, dropna removes it.
    hourly = hourly.dropna(subset=["open", "high", "low", "close"])
    
    # Add the source bar count for reconciliation
    hourly['source_bar_count'] = bar_counts
    
    # Flag low confidence hours (e.g., fewer than 10 out of 12 5-min bars)
    hourly['is_low_confidence'] = hourly['source_bar_count'] < 10

    return hourly.reset_index()

def reconcile_resample(source_df: pd.DataFrame, hourly_df: pd.DataFrame) -> bool:
    """
    Cross-validates a random 1hr bar by manually aggregating the source data.
    """
    if len(hourly_df) == 0:
        return False
        
    # Pick a random hour that has full data
    valid_hours = hourly_df[hourly_df['is_low_confidence'] == False]
    if len(valid_hours) == 0:
        sample_hour = hourly_df.iloc[len(hourly_df) // 2]['timestamp']
    else:
        sample_hour = valid_hours.sample(1).iloc[0]['timestamp']
        
    window_start = sample_hour
    window_end = sample_hour + pd.Timedelta(hours=1)
    
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
    
    actual = hourly_df[hourly_df["timestamp"] == sample_hour].iloc[0]
    
    match = (
        np.isclose(expected_open, actual["open"]) and
        np.isclose(expected_high, actual["high"]) and
        np.isclose(expected_low, actual["low"]) and
        np.isclose(expected_close, actual["close"]) and
        np.isclose(expected_vol, actual["volume"])
    )
    
        if not match:
            logger.warning(f"Reconciliation FAILED for {sample_hour}")
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
            
        logger.info(f"Resampling {f} to 1hr...")
        hourly_df = resample_to_1h(df)
        
        # Reconciliation
        passed = reconcile_resample(df, hourly_df)
        logger.info(f"  Reconciliation Check: {'PASS' if passed else 'FAIL'}")
        
        # Save output
        out_name = f.replace("5min", "1H")
        # Save only the required columns to match expected engine format
        save_df = hourly_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source_bar_count', 'is_low_confidence']].copy()
        # Rename timestamp back to datetime for the indicators library expectations
        save_df = save_df.rename(columns={'timestamp': 'datetime'})
        save_df.to_csv(out_name, index=False)
        logger.info(f"  Saved {out_name}\n")
        
    # Write validation report to file
    report_df = pd.DataFrame(reports)
    report_df.to_csv("data_validation_report.csv", index=False)
    logger.info("Validation report saved to data_validation_report.csv")

if __name__ == "__main__":
    run_pipeline()
