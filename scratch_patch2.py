with open("mr_strategies_5m_15m.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("→", "->")
content = content.replace("—", "-")

with open("mr_strategies_5m_15m.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed unicode arrows in mr_strategies_5m_15m.py")
