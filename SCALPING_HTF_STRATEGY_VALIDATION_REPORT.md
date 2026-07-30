=== GENUINELY VALIDATED STRATEGIES REPORT ===
These strategies passed the Standard Backtest, Walk-Forward Test, and Permutation Test without lowering any safety gates.

--- EMA PULLBACK SCALPER (Strategy #1) ---
Status: [FAILED GAUNTLET]
Parameters: {"fast_ema": 5, "slow_ema": 50, "rsi_period": 14, "rsi_pullback": 40, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true, "ema_trend_p": 200}
Performance (Aggregate): Expectancy=-7.8816, PF=0.79, WR=33.6%, Sharpe=-1.400, Trades/Day=0.5
Walk-Forward OOS: Expectancy=-7.6369, Sharpe=-1.477, PF=0.81
Permutation Test: p_value=0.9600
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-5.9440, PF=0.82, WR=36.2%, Sharpe=-1.202
  LTCUSDT_15m: Expectancy=-16.1979, PF=0.58, WR=27.9%, Sharpe=-3.318
  SOLUSDT_15m: Expectancy=-8.3513, PF=0.77, WR=34.9%, Sharpe=-1.711
  TRUMPUSDT_15m: Expectancy=-7.3898, PF=0.80, WR=31.8%, Sharpe=-1.314
  BTCUSDT_30m: Expectancy=-9.3499, PF=0.73, WR=37.7%, Sharpe=-1.465
  LTCUSDT_30m: Expectancy=-10.4524, PF=0.75, WR=31.2%, Sharpe=-1.209
  SOLUSDT_30m: Expectancy=-6.3435, PF=0.82, WR=33.8%, Sharpe=-1.028
  TRUMPUSDT_30m: Expectancy=0.9764, PF=1.03, WR=35.5%, Sharpe=0.044
--------------------------------------------------

--- VOLATILITY EXPANSION BREAKOUT (Strategy #2) ---
Status: [FAILED GAUNTLET]
Parameters: {"donchian_p": 20, "bb_width_max": 0.04, "chop_max": 45.0, "sl_atr": 1.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true, "ema_trend_p": 200}
Performance (Aggregate): Expectancy=-4.8329, PF=0.70, WR=32.9%, Sharpe=-4.457, Trades/Day=3.6
Walk-Forward OOS: Expectancy=-6.8654, Sharpe=-3.756, PF=0.78
Permutation Test: p_value=1.0000
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-5.3090, PF=0.53, WR=30.7%, Sharpe=-8.266
  LTCUSDT_15m: Expectancy=-5.0822, PF=0.51, WR=31.2%, Sharpe=-8.037
  SOLUSDT_15m: Expectancy=-5.0992, PF=0.60, WR=32.8%, Sharpe=-5.974
  TRUMPUSDT_15m: Expectancy=-4.8256, PF=0.69, WR=32.7%, Sharpe=-3.540
  BTCUSDT_30m: Expectancy=-1.5146, PF=0.95, WR=36.5%, Sharpe=-0.610
  LTCUSDT_30m: Expectancy=-8.5076, PF=0.60, WR=31.5%, Sharpe=-5.555
  SOLUSDT_30m: Expectancy=-6.6462, PF=0.75, WR=34.0%, Sharpe=-3.096
  TRUMPUSDT_30m: Expectancy=-1.6785, PF=0.95, WR=34.0%, Sharpe=-0.578
--------------------------------------------------

--- FILTERED MEAN REVERSION (Strategy #3) ---
Status: [FAILED GAUNTLET]
Parameters: {"bb_p": 20, "z_thresh": 2.0, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true, "ema_trend_p": 50}
Performance (Aggregate): Expectancy=0.6186, PF=1.02, WR=35.7%, Sharpe=0.019, Trades/Day=0.2
Walk-Forward OOS: Expectancy=0.0000, Sharpe=0.000, PF=0.00
Permutation Test: p_value=0.8450
Per-Symbol Breakdown:
  SOLUSDT_15m: Expectancy=0.6186, PF=1.02, WR=35.7%, Sharpe=0.019
--------------------------------------------------

--- SMA CROSSOVER SCALPER (Strategy #4) ---
Status: [FAILED GAUNTLET]
Parameters: {"ma_type": "sma", "fast_p": 10, "slow_p": 50, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-9.4641, PF=0.70, WR=34.1%, Sharpe=-3.186, Trades/Day=1.1
Walk-Forward OOS: Expectancy=-12.6822, Sharpe=-4.007, PF=0.65
Permutation Test: p_value=0.9600
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-9.5264, PF=0.71, WR=33.6%, Sharpe=-4.009
  LTCUSDT_15m: Expectancy=-9.8713, PF=0.63, WR=32.5%, Sharpe=-4.603
  SOLUSDT_15m: Expectancy=-9.8810, PF=0.64, WR=33.8%, Sharpe=-4.264
  TRUMPUSDT_15m: Expectancy=-7.5286, PF=0.76, WR=35.6%, Sharpe=-3.016
  BTCUSDT_30m: Expectancy=-8.8261, PF=0.71, WR=35.0%, Sharpe=-2.277
  LTCUSDT_30m: Expectancy=-12.1046, PF=0.65, WR=34.1%, Sharpe=-3.125
  SOLUSDT_30m: Expectancy=-10.0933, PF=0.70, WR=32.9%, Sharpe=-2.506
  TRUMPUSDT_30m: Expectancy=-7.8817, PF=0.80, WR=35.6%, Sharpe=-1.692
--------------------------------------------------

--- EMA CROSSOVER SCALPER (Strategy #5) ---
Status: [FAILED GAUNTLET]
Parameters: {"ma_type": "ema", "fast_p": 10, "slow_p": 50, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-8.9763, PF=0.72, WR=35.4%, Sharpe=-2.948, Trades/Day=1.0
Walk-Forward OOS: Expectancy=-15.5563, Sharpe=-4.769, PF=0.58
Permutation Test: p_value=0.4100
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-10.9298, PF=0.66, WR=34.7%, Sharpe=-4.661
  LTCUSDT_15m: Expectancy=-10.3434, PF=0.63, WR=34.9%, Sharpe=-4.137
  SOLUSDT_15m: Expectancy=-10.6422, PF=0.65, WR=31.4%, Sharpe=-4.414
  TRUMPUSDT_15m: Expectancy=-9.2232, PF=0.69, WR=35.5%, Sharpe=-3.606
  BTCUSDT_30m: Expectancy=-10.4915, PF=0.70, WR=33.7%, Sharpe=-2.497
  LTCUSDT_30m: Expectancy=-6.5516, PF=0.80, WR=37.6%, Sharpe=-1.294
  SOLUSDT_30m: Expectancy=-7.9721, PF=0.76, WR=36.0%, Sharpe=-1.850
  TRUMPUSDT_30m: Expectancy=-5.6565, PF=0.85, WR=39.3%, Sharpe=-1.128
--------------------------------------------------

--- MACD STRATEGY (Strategy #6) ---
Status: [FAILED GAUNTLET]
Parameters: {"sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-7.7259, PF=0.66, WR=33.8%, Sharpe=-4.824, Trades/Day=2.4
Walk-Forward OOS: Expectancy=-9.8614, Sharpe=-4.196, PF=0.70
Permutation Test: p_value=1.0000
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-7.1349, PF=0.56, WR=33.6%, Sharpe=-6.243
  LTCUSDT_15m: Expectancy=-7.0654, PF=0.65, WR=31.6%, Sharpe=-6.542
  SOLUSDT_15m: Expectancy=-7.0462, PF=0.65, WR=33.2%, Sharpe=-6.186
  TRUMPUSDT_15m: Expectancy=-6.1445, PF=0.67, WR=33.8%, Sharpe=-4.137
  BTCUSDT_30m: Expectancy=-8.8155, PF=0.68, WR=36.2%, Sharpe=-3.690
  LTCUSDT_30m: Expectancy=-10.8510, PF=0.56, WR=31.8%, Sharpe=-5.812
  SOLUSDT_30m: Expectancy=-9.1031, PF=0.67, WR=33.8%, Sharpe=-4.147
  TRUMPUSDT_30m: Expectancy=-5.6467, PF=0.81, WR=36.0%, Sharpe=-1.833
--------------------------------------------------

--- STOCH RSI STRATEGY (Strategy #7) ---
Status: [FAILED GAUNTLET]
Parameters: {"os_lvl": 20, "ob_lvl": 80, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-6.1248, PF=0.72, WR=32.9%, Sharpe=-5.166, Trades/Day=3.1
Walk-Forward OOS: Expectancy=-10.1256, Sharpe=-5.868, PF=0.68
Permutation Test: p_value=0.8400
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-6.1988, PF=0.63, WR=32.0%, Sharpe=-7.664
  LTCUSDT_15m: Expectancy=-5.6801, PF=0.66, WR=32.0%, Sharpe=-6.969
  SOLUSDT_15m: Expectancy=-5.3978, PF=0.74, WR=32.1%, Sharpe=-5.591
  TRUMPUSDT_15m: Expectancy=-5.9896, PF=0.63, WR=32.1%, Sharpe=-7.714
  BTCUSDT_30m: Expectancy=-9.4961, PF=0.63, WR=30.3%, Sharpe=-5.908
  LTCUSDT_30m: Expectancy=-2.9482, PF=0.90, WR=35.7%, Sharpe=-1.195
  SOLUSDT_30m: Expectancy=-6.9262, PF=0.77, WR=34.8%, Sharpe=-3.454
  TRUMPUSDT_30m: Expectancy=-6.3620, PF=0.78, WR=34.2%, Sharpe=-2.830
--------------------------------------------------

--- KAMA STRATEGY (Strategy #8) ---
Status: [FAILED GAUNTLET]
Parameters: {"sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-8.0749, PF=0.71, WR=33.7%, Sharpe=-3.576, Trades/Day=1.6
Walk-Forward OOS: Expectancy=-7.0842, Sharpe=-2.526, PF=0.80
Permutation Test: p_value=0.4750
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-6.8036, PF=0.72, WR=34.5%, Sharpe=-3.233
  LTCUSDT_15m: Expectancy=-10.1903, PF=0.56, WR=30.7%, Sharpe=-7.275
  SOLUSDT_15m: Expectancy=-7.7500, PF=0.69, WR=35.5%, Sharpe=-4.281
  TRUMPUSDT_15m: Expectancy=-7.1050, PF=0.73, WR=33.5%, Sharpe=-3.596
  BTCUSDT_30m: Expectancy=-11.8863, PF=0.60, WR=30.5%, Sharpe=-4.295
  LTCUSDT_30m: Expectancy=-6.9975, PF=0.78, WR=34.4%, Sharpe=-1.975
  SOLUSDT_30m: Expectancy=-6.4183, PF=0.81, WR=36.8%, Sharpe=-1.781
  TRUMPUSDT_30m: Expectancy=-7.4482, PF=0.77, WR=33.7%, Sharpe=-2.168
--------------------------------------------------

--- CONNORS RSI STRATEGY (Strategy #9) ---
Status: [FAILED GAUNTLET]
Parameters: {"sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true, "ema_trend_p": 200}
Performance (Aggregate): Expectancy=-10.5923, PF=0.72, WR=33.0%, Sharpe=-1.829, Trades/Day=0.5
Walk-Forward OOS: Expectancy=-13.2801, Sharpe=-3.239, PF=0.66
Permutation Test: p_value=0.7950
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-7.6198, PF=0.79, WR=34.7%, Sharpe=-1.498
  LTCUSDT_15m: Expectancy=-11.1109, PF=0.68, WR=32.3%, Sharpe=-2.827
  SOLUSDT_15m: Expectancy=-3.9176, PF=0.89, WR=40.3%, Sharpe=-0.856
  TRUMPUSDT_15m: Expectancy=-8.7719, PF=0.77, WR=35.1%, Sharpe=-1.712
  BTCUSDT_30m: Expectancy=-14.0731, PF=0.66, WR=29.9%, Sharpe=-1.972
  LTCUSDT_30m: Expectancy=-14.7687, PF=0.58, WR=31.1%, Sharpe=-2.367
  SOLUSDT_30m: Expectancy=-6.3334, PF=0.83, WR=35.7%, Sharpe=-0.960
  TRUMPUSDT_30m: Expectancy=-18.1428, PF=0.58, WR=24.7%, Sharpe=-2.439
--------------------------------------------------

--- DC STRATEGY (Strategy #10) ---
Status: [FAILED GAUNTLET]
Parameters: {"donchian_p": 20, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-6.8394, PF=0.64, WR=32.6%, Sharpe=-6.810, Trades/Day=3.4
Walk-Forward OOS: Expectancy=-12.1876, Sharpe=-7.577, PF=0.60
Permutation Test: p_value=0.7750
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-5.7132, PF=0.58, WR=30.8%, Sharpe=-9.610
  LTCUSDT_15m: Expectancy=-5.5867, PF=0.63, WR=31.1%, Sharpe=-8.460
  SOLUSDT_15m: Expectancy=-5.5102, PF=0.66, WR=34.5%, Sharpe=-7.479
  TRUMPUSDT_15m: Expectancy=-5.6701, PF=0.60, WR=32.6%, Sharpe=-7.100
  BTCUSDT_30m: Expectancy=-8.8965, PF=0.59, WR=33.4%, Sharpe=-6.144
  LTCUSDT_30m: Expectancy=-9.6189, PF=0.56, WR=31.8%, Sharpe=-7.536
  SOLUSDT_30m: Expectancy=-8.9472, PF=0.61, WR=32.9%, Sharpe=-6.420
  TRUMPUSDT_30m: Expectancy=-4.7720, PF=0.85, WR=33.5%, Sharpe=-1.727
--------------------------------------------------

--- MFI STRATEGY (Strategy #11) ---
Status: [FAILED GAUNTLET]
Parameters: {"sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true, "ema_trend_p": 50}
Performance (Aggregate): Expectancy=-9.8145, PF=0.73, WR=35.4%, Sharpe=-1.298, Trades/Day=0.2
Walk-Forward OOS: Expectancy=0.0000, Sharpe=0.000, PF=0.00
Permutation Test: p_value=0.6900
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-10.5019, PF=0.74, WR=31.4%, Sharpe=-0.971
  LTCUSDT_15m: Expectancy=-14.7490, PF=0.59, WR=33.3%, Sharpe=-2.227
  SOLUSDT_15m: Expectancy=-15.7109, PF=0.60, WR=30.4%, Sharpe=-1.959
  TRUMPUSDT_15m: Expectancy=-11.3438, PF=0.70, WR=35.0%, Sharpe=-1.683
  LTCUSDT_30m: Expectancy=3.4635, PF=1.10, WR=42.0%, Sharpe=0.245
  TRUMPUSDT_30m: Expectancy=-10.0450, PF=0.67, WR=40.4%, Sharpe=-1.192
--------------------------------------------------

--- SMASIG STRATEGY (Strategy #12) ---
Status: [FAILED GAUNTLET]
Parameters: {"sma_p": 20, "sig_p": 5, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-8.2486, PF=0.61, WR=32.1%, Sharpe=-6.487, Trades/Day=2.8
Walk-Forward OOS: Expectancy=-13.0008, Sharpe=-7.039, PF=0.61
Permutation Test: p_value=0.7650
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-6.5399, PF=0.66, WR=32.3%, Sharpe=-6.751
  LTCUSDT_15m: Expectancy=-6.3701, PF=0.58, WR=30.9%, Sharpe=-8.458
  SOLUSDT_15m: Expectancy=-6.3596, PF=0.59, WR=33.6%, Sharpe=-7.064
  TRUMPUSDT_15m: Expectancy=-6.3072, PF=0.61, WR=33.3%, Sharpe=-6.888
  BTCUSDT_30m: Expectancy=-10.1158, PF=0.64, WR=32.7%, Sharpe=-5.335
  LTCUSDT_30m: Expectancy=-10.1883, PF=0.63, WR=31.3%, Sharpe=-6.141
  SOLUSDT_30m: Expectancy=-10.6004, PF=0.54, WR=30.4%, Sharpe=-6.546
  TRUMPUSDT_30m: Expectancy=-9.5078, PF=0.64, WR=32.1%, Sharpe=-4.712
--------------------------------------------------

--- SMA CLOSE/EXIT SIGNALS (Strategy #13) ---
Status: [FAILED GAUNTLET]
Parameters: {"sma_p": 20, "sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-6.3509, PF=0.67, WR=33.9%, Sharpe=-6.046, Trades/Day=3.4
Walk-Forward OOS: Expectancy=-10.2568, Sharpe=-6.013, PF=0.68
Permutation Test: p_value=0.9650
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-5.5791, PF=0.67, WR=33.1%, Sharpe=-6.815
  LTCUSDT_15m: Expectancy=-5.2592, PF=0.59, WR=31.3%, Sharpe=-9.459
  SOLUSDT_15m: Expectancy=-5.2878, PF=0.69, WR=33.5%, Sharpe=-6.405
  TRUMPUSDT_15m: Expectancy=-5.6763, PF=0.54, WR=31.6%, Sharpe=-9.566
  BTCUSDT_30m: Expectancy=-8.1562, PF=0.72, WR=34.9%, Sharpe=-4.586
  LTCUSDT_30m: Expectancy=-5.9142, PF=0.78, WR=36.5%, Sharpe=-3.079
  SOLUSDT_30m: Expectancy=-7.5497, PF=0.70, WR=36.6%, Sharpe=-4.479
  TRUMPUSDT_30m: Expectancy=-7.3848, PF=0.70, WR=33.7%, Sharpe=-3.979
--------------------------------------------------

--- OVUN STRATEGY (Strategy #14) ---
Status: [FAILED GAUNTLET]
Parameters: {"sl_atr": 2.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-5.5165, PF=0.65, WR=33.7%, Sharpe=-7.040, Trades/Day=4.5
Walk-Forward OOS: Expectancy=-9.8452, Sharpe=-7.463, PF=0.66
Permutation Test: p_value=0.6450
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-5.4895, PF=0.65, WR=32.9%, Sharpe=-8.019
  LTCUSDT_15m: Expectancy=-4.2726, PF=0.56, WR=32.6%, Sharpe=-9.399
  SOLUSDT_15m: Expectancy=-4.1773, PF=0.64, WR=33.1%, Sharpe=-10.801
  TRUMPUSDT_15m: Expectancy=-4.0377, PF=0.63, WR=34.6%, Sharpe=-6.611
  BTCUSDT_30m: Expectancy=-6.5278, PF=0.70, WR=36.3%, Sharpe=-4.732
  LTCUSDT_30m: Expectancy=-7.1677, PF=0.65, WR=31.9%, Sharpe=-6.680
  SOLUSDT_30m: Expectancy=-6.6324, PF=0.64, WR=34.2%, Sharpe=-6.331
  TRUMPUSDT_30m: Expectancy=-5.8270, PF=0.75, WR=34.1%, Sharpe=-3.751
--------------------------------------------------

--- FX REPLAY SESSION BREAKOUT (Strategy #15) ---
Status: [FAILED GAUNTLET]
Parameters: {"sl_atr": 1.0, "tp_atr": 3.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=-3.8082, PF=0.86, WR=34.4%, Sharpe=-1.376, Trades/Day=1.4
Walk-Forward OOS: Expectancy=-3.9349, Sharpe=-1.038, PF=0.90
Permutation Test: p_value=0.5800
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=-7.9395, PF=0.67, WR=34.3%, Sharpe=-3.142
  LTCUSDT_15m: Expectancy=-6.4748, PF=0.76, WR=34.1%, Sharpe=-2.216
  SOLUSDT_15m: Expectancy=-5.3755, PF=0.81, WR=32.3%, Sharpe=-1.722
  TRUMPUSDT_15m: Expectancy=-5.5469, PF=0.81, WR=33.5%, Sharpe=-1.451
  BTCUSDT_30m: Expectancy=-2.6217, PF=0.93, WR=35.1%, Sharpe=-0.513
  LTCUSDT_30m: Expectancy=-8.8613, PF=0.72, WR=34.8%, Sharpe=-1.957
  SOLUSDT_30m: Expectancy=-2.5722, PF=0.93, WR=35.1%, Sharpe=-0.562
  TRUMPUSDT_30m: Expectancy=8.9261, PF=1.21, WR=36.3%, Sharpe=0.558
--------------------------------------------------

--- SMC LIQUIDITY SWEEP (Strategy #16) ---
Status: [PASSED GAUNTLET]
Parameters: {"lookback": 20, "sl_atr": 0.5, "tp_atr": 2.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=5.6362, PF=1.12, WR=42.4%, Sharpe=0.636, Trades/Day=1.5
Walk-Forward OOS: Expectancy=0.1583, Sharpe=-0.220, PF=1.01
Permutation Test: p_value=0.0250
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=3.1669, PF=1.07, WR=41.2%, Sharpe=0.542
  LTCUSDT_15m: Expectancy=2.1798, PF=1.04, WR=41.4%, Sharpe=0.352
  SOLUSDT_15m: Expectancy=1.3519, PF=1.04, WR=43.3%, Sharpe=0.255
  TRUMPUSDT_15m: Expectancy=3.9929, PF=1.09, WR=42.6%, Sharpe=0.481
  BTCUSDT_30m: Expectancy=27.6683, PF=1.54, WR=44.0%, Sharpe=2.565
  LTCUSDT_30m: Expectancy=7.0743, PF=1.17, WR=45.8%, Sharpe=1.141
  SOLUSDT_30m: Expectancy=0.6512, PF=1.02, WR=41.7%, Sharpe=0.029
  TRUMPUSDT_30m: Expectancy=-0.9956, PF=0.97, WR=39.3%, Sharpe=-0.276
--------------------------------------------------

--- SMC LIQUIDITY SWEEP (Strategy #17) ---
Status: [PASSED GAUNTLET]
Parameters: {"lookback": 20, "sl_atr": 0.5, "tp_atr": 5.0, "max_bars_hold": 9999, "risk_pct": 0.01, "trailing": true}
Performance (Aggregate): Expectancy=5.6362, PF=1.12, WR=42.4%, Sharpe=0.636, Trades/Day=1.5
Walk-Forward OOS: Expectancy=0.1583, Sharpe=-0.220, PF=1.01
Permutation Test: p_value=0.0250
Per-Symbol Breakdown:
  BTCUSDT_15m: Expectancy=3.1669, PF=1.07, WR=41.2%, Sharpe=0.542
  LTCUSDT_15m: Expectancy=2.1798, PF=1.04, WR=41.4%, Sharpe=0.352
  SOLUSDT_15m: Expectancy=1.3519, PF=1.04, WR=43.3%, Sharpe=0.255
  TRUMPUSDT_15m: Expectancy=3.9929, PF=1.09, WR=42.6%, Sharpe=0.481
  BTCUSDT_30m: Expectancy=27.6683, PF=1.54, WR=44.0%, Sharpe=2.565
  LTCUSDT_30m: Expectancy=7.0743, PF=1.17, WR=45.8%, Sharpe=1.141
  SOLUSDT_30m: Expectancy=0.6512, PF=1.02, WR=41.7%, Sharpe=0.029
  TRUMPUSDT_30m: Expectancy=-0.9956, PF=0.97, WR=39.3%, Sharpe=-0.276
--------------------------------------------------

