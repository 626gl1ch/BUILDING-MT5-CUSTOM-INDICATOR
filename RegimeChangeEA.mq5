
//+------------------------------------------------------------------+
//|                                              RegimeChangeEA.mq5 |
//|                                             Daniel's Trading Co. |
//|                                       https://www.danielstrading.com |
//+------------------------------------------------------------------+
#property copyright "Daniel's Trading Co."
#property link      "https://www.danielstrading.com"
#property version   "1.00"
#property description "Expert Advisor for regime-switching strategy (MR & Trend Following)"
#property strict

// --- Input Parameters ---
input group "=== Regime Detection ==="
input int    ChopPeriod = 14;       // Choppiness Index period
input int    ADXPeriod = 14;        // ADX period
input int    ATRPeriod = 14;        // ATR period
input double TrendThreshold = 25;   // ADX trending threshold
input double ChopThreshold = 50;    // Choppiness ranging threshold

input group "=== Mean Reversion (Ranging) ==="
input int    StochK = 14;           // Stochastic %K period
input int    StochD = 3;            // Stochastic %D period
input int    StochSlowing = 3;      // Stochastic slowing
input int    RsiPeriod = 14;        // RSI period
input double OBLevel = 80;          // Overbought level
input double OSLevel = 20;          // Oversold level

input group "=== Trend Following (Trending) ==="
input int    FastEMA = 20;          // Fast EMA for trend
input int    SlowEMA = 50;          // Slow EMA for trend
input int    BreakoutPeriod = 20;   // Breakout lookback
input double ATRMultiplier = 2.0;   // ATR trailing stop multiplier

input group "=== Risk Management ==="
input double RiskPercent = 1.0;     // Risk per trade (%)
input double FixedSL_ATR = 1.5;     // Stop loss in ATR
input double FixedTP_ATR = 3.0;     // Take profit in ATR

// --- Global Variables ---
int OnInit()
  {
   //--- Initialization code here
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   //--- Deinitialization code here
  }

void OnTick()
  {
   //--- Check for new bar
   static datetime last_bar_time = 0;
   MqlRates rates[];
   if(CopyRates(Symbol(),Period(),0,1,rates) < 1) return;
   if(rates[0].time == last_bar_time) return;
   last_bar_time = rates[0].time;

   //--- Calculate Indicators ---
   // Choppiness Index
   double chop_buffer[];
   int chop_handle = iBandsOnArray(rates,0,ChopPeriod,0,0,MODE_SMA,PRICE_CLOSE,chop_buffer);
   if(chop_handle < 0) { Print("Error creating Chop Index"); return; }
   CopyBuffer(chop_handle,0,0,1,chop_buffer);
   double choppiness = chop_buffer[0]; // Simplified for now - needs proper CI calculation

   // ADX
   double adx_main_buffer[], adx_plus_buffer[], adx_minus_buffer[];
   int adx_handle = iADX(Symbol(),Period(),ADXPeriod);
   if(adx_handle < 0) { Print("Error creating ADX"); return; }
   CopyBuffer(adx_handle,MAIN_LINE,0,1,adx_main_buffer);
   CopyBuffer(adx_handle,PLUSDI_LINE,0,1,adx_plus_buffer);
   CopyBuffer(adx_handle,MINUSDI_LINE,0,1,adx_minus_buffer);
   double adx_value = adx_main_buffer[0];
   double adx_plus_di = adx_plus_buffer[0];
   double adx_minus_di = adx_minus_buffer[0];

   // ATR
   double atr_buffer[];
   int atr_handle = iATR(Symbol(),Period(),ATRPeriod);
   if(atr_handle < 0) { Print("Error creating ATR"); return; }
   CopyBuffer(atr_handle,0,0,1,atr_buffer);
   double atr_value = atr_buffer[0];

   // Stochastic RSI
   double stoch_k_buffer[], stoch_d_buffer[];
   int stoch_handle = iStochastic(Symbol(),Period(),StochK,StochD,StochSlowing,MODE_SMA,STO_LOWHIGH);
   if(stoch_handle < 0) { Print("Error creating Stochastic"); return; }
   CopyBuffer(stoch_handle,MAIN_LINE,0,1,stoch_k_buffer);
   CopyBuffer(stoch_handle,SIGNAL_LINE,0,1,stoch_d_buffer);
   double stoch_k = stoch_k_buffer[0];
   double stoch_d = stoch_d_buffer[0];

   // RSI
   double rsi_buffer[];
   int rsi_handle = iRSI(Symbol(),Period(),RsiPeriod,PRICE_CLOSE);
   if(rsi_handle < 0) { Print("Error creating RSI"); return; }
   CopyBuffer(rsi_handle,0,0,1,rsi_buffer);
   double rsi_value = rsi_buffer[0];

   // EMAs
   double fast_ema_buffer[], slow_ema_buffer[];
   int fast_ema_handle = iMA(Symbol(),Period(),FastEMA,0,MODE_EMA,PRICE_CLOSE);
   int slow_ema_handle = iMA(Symbol(),Period(),SlowEMA,0,MODE_EMA,PRICE_CLOSE);
   if(fast_ema_handle < 0 || slow_ema_handle < 0) { Print("Error creating EMAs"); return; }
   CopyBuffer(fast_ema_handle,0,0,1,fast_ema_buffer);
   CopyBuffer(slow_ema_handle,0,0,1,slow_ema_buffer);
   double fast_ema = fast_ema_buffer[0];
   double slow_ema = slow_ema_buffer[0];

   //--- Regime Detection ---
   bool is_ranging = (choppiness > ChopThreshold && adx_value < TrendThreshold);
   bool is_trending = (choppiness < ChopThreshold && adx_value > TrendThreshold);

   //--- Trading Logic ---
   if (is_ranging)
     {
      // Mean Reversion Strategy
      if (stoch_k < OSLevel && rsi_value < 50) // Oversold conditions for BUY
        {
         // Place BUY order
         // Simplified order placement - needs proper implementation with risk management
         Print("Ranging Market: BUY Signal");
        }
      else if (stoch_k > OBLevel && rsi_value > 50) // Overbought conditions for SELL
        {
         // Place SELL order
         // Simplified order placement - needs proper implementation with risk management
         Print("Ranging Market: SELL Signal");
        }
     }
   else if (is_trending)
     {
      // Trend Following Strategy
      if (rates[0].close > fast_ema && fast_ema > slow_ema && adx_plus_di > adx_minus_di) // Uptrend
        {
         // Place BUY order
         Print("Trending Market: BUY Signal");
        }
      else if (rates[0].close < fast_ema && fast_ema < slow_ema && adx_minus_di > adx_plus_di) // Downtrend
        {
         // Place SELL order
         Print("Trending Market: SELL Signal");
        }
     }
   else
     {
      Print("Neutral Market: Waiting for clear regime...");
     }
  }

//+------------------------------------------------------------------+
