
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- MT5 Login Details ---
MT5_LOGIN = 106298009
MT5_PASSWORD = "NhAl-4Ru"
MT5_SERVER = "FBS-Demo"

# --- Assets to Download ---
ASSETS = ["USDJPY", "EURUSD", "BTCUSD", "XAUUSD", "GBPUSD"]

# --- Timeframe ---
TIMEFRAME = mt5.TIMEFRAME_M5  # 5-minute timeframe
YEARS_TO_DOWNLOAD = 1

def download_data(symbol, timeframe, years):
    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        datetime.now() - timedelta(days=years * 365),
        datetime.now()
    )
    if rates is None:
        print(f"Error downloading data for {symbol}: {mt5.last_error()}")
        return None
    return pd.DataFrame(rates)

def calculate_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()

    # Rename columns for consistency
    df = df.rename(columns={'time': 'timestamp_ms', 'tick_volume': 'volume'})
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='s')
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time

    # Basic OHLCV derived metrics
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_pct'] = (df['body'] / df['range']).fillna(0) * 100
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Placeholder for 'turnover' and 'dollar_volume' as MT5 doesn't provide them directly for all assets
    df['turnover'] = df['volume'] * df['close'] # Approximation
    df['dollar_volume'] = df['volume'] * df['close'] # Approximation

    # Volume analytics (placeholders)
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20'] # Approximation

    # ATR (using a simple calculation for now)
    # This is a simplified ATR. A more robust implementation would be needed for a real EA.
    df['atr_14'] = df['range'].rolling(window=14).mean()

    # RSI (placeholder)
    df['rsi_14'] = 50 # Placeholder

    # MACD (placeholder)
    df['macd'] = 0 # Placeholder
    df['macd_signal'] = 0 # Placeholder
    df['macd_hist'] = 0 # Placeholder

    # Bollinger Bands (placeholder)
    df['bb_upper_20'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
    df['bb_lower_20'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()

    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['is_weekend'] = ((df['day_of_week'] == 5) | (df['day_of_week'] == 6)).astype(int)
    df['bar_index'] = df.index # Simple bar index
    df['session'] = 'unknown' # Placeholder for session

    # Moving Averages (placeholders)
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['sma_14'] = df['close'].rolling(window=14).mean()
    df['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    return df

def main():
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return

    # Attempt to login
    authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        print(f"Failed to connect to MT5 account {MT5_LOGIN} on {MT5_SERVER}, error code: {mt5.last_error()}")
        mt5.shutdown()
        return

    print(f"Successfully connected to MT5 account {MT5_LOGIN} on {MT5_SERVER}")

    for asset in ASSETS:
        print(f"Downloading data for {asset}...")
        df = download_data(asset, TIMEFRAME, YEARS_TO_DOWNLOAD)

        if df is not None:
            # Calculate indicators and format data
            df = calculate_indicators(df)

            # Reorder columns to match the reference CSV
            # This list should be updated if the reference CSV columns change significantly
            ordered_columns = [
                'timestamp_ms', 'datetime', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'turnover', 
                'range', 'body', 'upper_wick', 'lower_wick', 'body_pct', 'returns', 'log_returns', 'volume_ma_20', 
                'volume_ratio', 'dollar_volume', 'atr_14', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
                'bb_upper_20', 'bb_lower_20', 'hour', 'day_of_week', 'week_of_year', 'month', 'year', 
                'is_weekend', 'bar_index', 'session', 'sma_7', 'ema_7', 'sma_14', 'ema_14', 'sma_21', 'ema_21', 
                'sma_50', 'ema_50', 'sma_100', 'ema_100', 'sma_200', 'ema_200'
            ]
            
            # Ensure all ordered_columns exist in df, add missing ones with NaN if necessary
            for col in ordered_columns:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[ordered_columns]

            output_filename = f"{asset}_5min_{YEARS_TO_DOWNLOAD}year.csv"
            df.to_csv(output_filename, index=False)
            print(f"Successfully downloaded and saved {asset} data to {output_filename}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
