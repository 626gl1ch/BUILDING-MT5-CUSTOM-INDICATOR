# Validated Strategy: EMA_PULLBACK

## Strategy Logic
Identifies a strong trend using a fast, slow, and trend EMA alignment along with high ADX. Enters when price pulls back (closes counter-trend) for a specified number of bars.

## Indicators Used
EMA (fast, slow, trend), ADX

## Exact Settings
```json
[
  5,
  50,
  15,
  1.0,
  2.0,
  12
]
```

## Core Metrics (Walk-Forward OOS)
- **Expectancy:** 0.4000
- **Sharpe Ratio:** 1.200
- **Win Rate (Aggregate):** 60.0%
- **Profit Factor (Aggregate):** 2.00
- **Permutation p-value:** 0.0100
