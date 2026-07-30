import os

new_func = """
def generate_smc_vol_profile(df, p):
    '''
    SMC Liquidity Sweep + Volume Profile (POC) Directional Filter
    '''
    close = df['close']
    high = df['high']
    low = df['low']
    
    lookback = p.get('lookback', 20)
    
    swing_high = df.get(f'swing_high_{lookback}', close)
    swing_low = df.get(f'swing_low_{lookback}', close)
    
    poc = df.get('svp_poc', close)
    
    signals = pd.Series(0, index=df.index)
    
    sweep_high = (high > swing_high.shift(1)) & \
                 (close < swing_high.shift(1)) & \
                 (close.shift(1) < swing_high.shift(1)) & \
                 (close < poc)
    
    sweep_low = (low < swing_low.shift(1)) & \
                (close > swing_low.shift(1)) & \
                (close.shift(1) > swing_low.shift(1)) & \
                (close > poc)
    
    signals[sweep_low] = 1
    signals[sweep_high] = -1
    return signals
"""

with open("smc_enhanced_library.py", "r") as f:
    content = f.read()

if "generate_smc_vol_profile" not in content:
    with open("smc_enhanced_library.py", "a") as f:
        f.write(new_func)
    print("Patched smc_enhanced_library.py")
else:
    print("Already patched")
