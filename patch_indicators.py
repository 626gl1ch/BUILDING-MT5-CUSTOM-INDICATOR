import os
import re

with open("indicators_library.py", "r") as f:
    content = f.read()

if "calc_session_svp" not in content:
    svp_func = """

# ==========================================
# CATEGORY 11: VOLUME PROFILE INDICATORS
# ==========================================

def calc_session_svp(df, rows=50):
    '''
    Calculates Daily Session Volume Profile and Point of Control (POC).
    To optimize for backtesting, we group by Day, compute the profile, 
    and use the *previous* day's POC as the directional filter for the current day.
    '''
    df_copy = df.copy()
    df_copy['date_only'] = df_copy.index.date
    
    daily_poc = {}
    import numpy as np
    
    for date, group in df_copy.groupby('date_only'):
        min_p = group['low'].min()
        max_p = group['high'].max()
        
        if max_p == min_p:
            daily_poc[date] = min_p
            continue
            
        bins = np.linspace(min_p, max_p, rows + 1)
        vol_profile = np.zeros(rows)
        
        tp = (group['high'] + group['low'] + group['close']) / 3
        bin_indices = np.digitize(tp, bins) - 1
        bin_indices = np.clip(bin_indices, 0, rows - 1)
        
        for i in range(len(group)):
            vol_profile[bin_indices.iloc[i]] += group['volume'].iloc[i]
            
        poc_idx = np.argmax(vol_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2.0
        daily_poc[date] = poc_price
        
    unique_dates = sorted(list(daily_poc.keys()))
    shifted_poc = {unique_dates[i]: daily_poc[unique_dates[i-1]] for i in range(1, len(unique_dates))}
    if len(unique_dates) > 0:
        shifted_poc[unique_dates[0]] = daily_poc[unique_dates[0]] 
        
    poc_series = df_copy['date_only'].map(shifted_poc)
    return poc_series.ffill()
"""

    content += svp_func
    
    replacement = """
    # 10. Volume Profile
    res['svp_poc'] = calc_session_svp(res)
    
    # Clean NaN/Inf values
    res = res.replace([np.inf, -np.inf], np.nan)
    return res.copy()
"""
    content = content.replace("    # Clean NaN/Inf values\n    res = res.replace([np.inf, -np.inf], np.nan)\n    return res.copy()\n", replacement)

    with open("indicators_library.py", "w") as f:
        f.write(content)
    print("Patched indicators_library.py")
else:
    print("Already patched")
