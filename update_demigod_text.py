import re

with open("Demi-God strategies.txt", "r") as f:
    content = f.read()

# 1. Update Rank 128 MACD_MEAN_REVERSION
mac_rep = """RULES:
1. Sweep (Long): Price sweeps a recent 20-period swing low (Current Low < Prev 20-bar Low AND Current Close > Prev 20-bar Low).
   Sweep (Short): Price sweeps a recent 20-period swing high (Current High > Prev 20-bar High AND Current Close < Prev 20-bar High).
2. Momentum (Long): MACD(12, 26, 9) Histogram ticks UP from the previous bar.
   Momentum (Short): MACD(12, 26, 9) Histogram ticks DOWN from the previous bar.
3. Regime: Choppiness Index(14) > 61.8 (Chop regime confirmed)."""
content = re.sub(r"RULES:\n1\. Price sweeps a recent 20-period swing low.\n2\. MACD Histogram ticks UP from the previous bar \(Momentum Shift\).\n3\. Choppiness Index\(14\) > 61.8 \(Chop regime confirmed\).", mac_rep, content)


# 2. Update Rank 87 MR-F_V2_CAPITULATION
mrf_rep = """RULES:
1. BB(20, 2.0) %B drops below 0.05 within the last 5 bars (Washout).
2. ATR(14) spikes > 1.5x its 50-period rolling average.
3. Volume spikes > 1.5x its 50-period rolling average.
4. Price makes 12-bar low, RSI(14) diverges (makes higher low).
5. Trigger: BB(20, 2.0) %B moves back above 0.10 on the current bar (Reclaim)."""
content = re.sub(r"RULES:\n1\. BB\(20, 2\.0\) %B drops below 0\.05.\n2\. ATR\(14\) spikes > 1\.5x its 50-period rolling average.\n3\. Volume spikes > 1\.5x its 50-period rolling average.\n4\. Price makes 12-bar low, RSI\(14\) diverges \(makes higher low\).\n5\. Trigger: BB\(20, 2\.0\) %B moves back above 0\.10.", mrf_rep, content)


# 3. Update POC Definition
content = content.replace("24-bar Volume Weighted Average (SVP POC)", "24-period SMA of a rolling VWAP line (SVP POC approximation)")
content = content.replace("24-bar Volume POC", "24-period SMA of a rolling VWAP line (SVP POC approximation)")


# 4. Update StochRSI Definition
stoch_str = "StochRSI(14, 3, 3) (K is a 3-period SMA of raw stochastic, D is a 3-period SMA of K)"
content = content.replace("StochRSI K crosses above D", f"{stoch_str} %K crosses above %D")
content = content.replace("StochRSI K crosses below D", f"{stoch_str} %K crosses below %D")


# 5. Add LOGIC & CONDITIONS for the 5m ML trials
def add_5m_logic(match):
    full_str = match.group(0)
    # Extract params using regex
    b_lower = float(re.search(r"'b_lower': ([\d\.]+)", full_str).group(1))
    b_upper = float(re.search(r"'b_upper': ([\d\.]+)", full_str).group(1))
    b_trigger_l = float(re.search(r"'b_trigger_l': ([\d\.]+)", full_str).group(1))
    b_trigger_s = float(re.search(r"'b_trigger_s': ([\d\.]+)", full_str).group(1))
    vol_mult = float(re.search(r"'vol_mult': ([\d\.]+)", full_str).group(1))
    atr_mult = float(re.search(r"'atr_mult': ([\d\.]+)", full_str).group(1))
    sl = float(re.search(r"'sl_atr': ([\d\.]+)", full_str).group(1))
    tp = float(re.search(r"'tp_atr': ([\d\.]+)", full_str).group(1))
    mb = int(re.search(r"'max_bars_hold': (\d+)", full_str).group(1))
    
    logic_str = f"""
LOGIC & CONDITIONS:
1. Setup (Long): Bollinger Band (20, 2.0) %B drops below {b_lower:.3f} within the last 5 bars.
2. Setup (Short): Bollinger Band (20, 2.0) %B spikes above {b_upper:.3f} within the last 5 bars.
3. Volume Filter: Volume must spike > {vol_mult:.2f}x the 50-period average.
4. Volatility Filter: ATR(14) must spike > {atr_mult:.2f}x the 50-period average.
5. Trigger (Long): %B crosses back inside to > {b_trigger_l:.3f} on the current bar.
6. Trigger (Short): %B crosses back inside to < {b_trigger_s:.3f} on the current bar.

TRIGGERS & EXITS:
1. Stop Loss: {sl} * ATR(14)
2. Take Profit: {tp} * ATR(14)
3. Time Stop: Exit after {mb} bars."""
    
    # We append this right after the SETTINGS & RULES dict
    return full_str + logic_str + "\n"

# The regex matches SETTINGS & RULES: \n {dict} \n
content = re.sub(r"SETTINGS & RULES:\n  \{'strategy_type': 'CAPITULATION',.*?'max_bars_hold': \d+\}", add_5m_logic, content)

# 6. Update MACD_REV (Trial 157)
macd_rev_t157_rep = """LOGIC & CONDITIONS:
1. Sweep (Long): Price sweeps a recent 23-bar swing low (Current Low < Prev 23-bar Low AND Current Close > Prev 23-bar Low).
   Sweep (Short): Price sweeps a recent 23-bar swing high (Current High > Prev 23-bar High AND Current Close < Prev 23-bar High).
2. Momentum (Long): MACD(12, 26, 9) Histogram ticks UP from the previous bar.
   Momentum (Short): MACD(12, 26, 9) Histogram ticks DOWN from the previous bar.
3. Regime: Choppiness Index (14) must be > 64.9 (Ranging/Choppy market)."""
content = re.sub(r"LOGIC & CONDITIONS:\n1\. Liquidity Sweep: Price must sweep a 23-bar swing high/low.\n2\. Momentum: MACD Histogram must shift direction \(tick up for longs, tick down for shorts\).\n3\. Regime: Choppiness Index \(14\) must be > 64\.9 \(Ranging/Choppy market\).", macd_rev_t157_rep, content)


with open("Demi-God strategies.txt", "w") as f:
    f.write(content)

print("Updates successful.")
