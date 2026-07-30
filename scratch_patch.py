import re

def update_strategies(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Define the replacements as a list of (function_name, exit_logic_str)
    replacements = [
        ("strategy_MR_A", '    # Dynamic Exit\n    exit_long  = df["high"] >= alma\n    exit_short = df["low"] <= alma\n    return signals, exit_long, exit_short'),
        ("strategy_MR_B", '    # Dynamic Exit\n    exit_long  = df["high"] >= ema9\n    exit_short = df["low"] <= ema9\n    return signals, exit_long, exit_short'),
        ("strategy_MR_C", '    # Dynamic Exit\n    exit_long  = df["high"] >= alma\n    exit_short = df["low"] <= alma\n    return signals, exit_long, exit_short'),
        ("strategy_MR_D", '    # Dynamic Exit\n    exit_long  = df["high"] >= ema21\n    exit_short = df["low"] <= ema21\n    return signals, exit_long, exit_short'),
        ("strategy_MR_E", '    # Dynamic Exit\n    exit_long  = df["high"] >= ema9\n    exit_short = df["low"] <= ema9\n    return signals, exit_long, exit_short'),
        ("strategy_MR_F", '    # Dynamic Exit\n    exit_long  = df["high"] >= bb_m\n    exit_short = df["low"] <= bb_m\n    return signals, exit_long, exit_short'),
        ("strategy_MR_G", '    # Dynamic Exit\n    exit_long  = df["high"] >= alma\n    exit_short = df["low"] <= alma\n    return signals, exit_long, exit_short'),
        ("strategy_MR_H", '    # Dynamic Exit\n    exit_long  = rsi >= 50\n    exit_short = rsi <= 50\n    return signals, exit_long, exit_short'),
        ("strategy_MR_I", '    # Dynamic Exit\n    exit_long  = df["high"] >= bb_m\n    exit_short = df["low"] <= bb_m\n    return signals, exit_long, exit_short'),
        ("strategy_MR_J", '    # Dynamic Exit\n    exit_long  = df["high"] >= ema9\n    exit_short = df["low"] <= ema9\n    return signals, exit_long, exit_short'),
        ("strategy_MR_K", '    # Dynamic Exit\n    exit_long  = df["high"] >= alma\n    exit_short = df["low"] <= alma\n    return signals, exit_long, exit_short'),
        ("strategy_MR_L", '    # Dynamic Exit\n    exit_long  = df["high"] >= ema9\n    exit_short = df["low"] <= ema9\n    return signals, exit_long, exit_short')
    ]

    for func, exit_code in replacements:
        # We find the specific return signals string inside the function block
        # A simple way: find "def func_name(" and then the first "return signals"
        
        # Regex to match the return signals for the specific function
        pattern = re.compile(rf'(def {func}\(df, p\):.*?)(    return signals\n)', re.DOTALL)
        content = pattern.sub(rf'\1{exit_code}\n', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Updated successfully.")

update_strategies("mr_strategies_5m_15m.py")
