//+------------------------------------------------------------------+
//|                                       RegimeChangeIndicator.mq5 |
//|                                             Daniel's Trading Co. |
//|                                       https://www.danielstrading.com |
//+------------------------------------------------------------------+
#property copyright "Daniel's Trading Co."
#property link      "https://www.danielstrading.com"
#property version   "2.00"
#property description "MT5 Custom Indicator for Regime-Switching Mean Reversion & Trend Following"
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   2

// --- Plot Buy Signals ---
#property indicator_label1  "Buy Signal"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrLime
#property indicator_width1  2

// --- Plot Sell Signals ---
#property indicator_label2  "Sell Signal"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrRed
#property indicator_width2  2

// --- Input Parameters ---
input group "=== Regime Detection ==="
input int    ChopPeriod = 14;       // Choppiness Index period
input int    ADXPeriod = 14;        // ADX period
input int    ATRPeriod = 14;        // ATR period
input double TrendThreshold = 25;   // ADX trending threshold
input double ChopThreshold = 50;    // Choppiness ranging threshold
input int    EmaHTFPeriod = 600;    // Macro Trend Filter (600-EMA)

input group "=== Mean Reversion ==="
input int    BBPeriod = 20;         // Bollinger Bands period
input double BBDeviation = 2.0;     // Bollinger Bands deviation
input int    StochPeriod = 14;      // Stochastic RSI period
input double OSLevel = 15.0;        // StochRSI Oversold level
input double OBLevel = 85.0;        // StochRSI Overbought level
input int    RsiPeriod = 14;        // RSI Period

input group "=== Risk Management ==="
input double SL_ATR_Mult = 3.0;     // Stop Loss multiplier (ATR)

input group "=== New Indicators ==="
input int    KamaPeriod = 10;       // KAMA Period
input int    KamaFast = 2;          // KAMA Fast EMA Constant
input int    KamaSlow = 30;         // KAMA Slow EMA Constant
input int    AlmaPeriod = 9;        // ALMA Period
input double AlmaOffset = 0.85;     // ALMA Offset
input double AlmaSigma = 6.0;       // ALMA Sigma
input int    DonchianPeriod = 20;   // Donchian Channel Period
input int    SuperTrendPeriod = 10; // SuperTrend ATR Period
input double SuperTrendMult = 3.0;  // SuperTrend Multiplier

// --- Indicator Buffers ---
double BuyBuffer[];
double SellBuffer[];
double TPBuffer[];
double SLBuffer[];

// --- Handles ---
int adx_handle;
int atr_handle;
int rsi_handle;
int ema_handle;
int bb_handle;
int kama_handle;

// --- Global variables ---
int min_bars_required;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//|------------------------------------------------------------------+
int OnInit()
  {
   // Bind buffers
   SetIndexBuffer(0, BuyBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, SellBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, TPBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(3, SLBuffer, INDICATOR_CALCULATIONS);

   // Configure arrow styles
   PlotIndexSetInteger(0, PLOT_ARROW, 233); // Up arrow
   PlotIndexSetInteger(1, PLOT_ARROW, 234); // Down arrow

   // Empty values
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, 0.0);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, 0.0);

   // Get Indicator Handles
   adx_handle = iADX(Symbol(), Period(), ADXPeriod);
   atr_handle = iATR(Symbol(), Period(), ATRPeriod);
   rsi_handle = iRSI(Symbol(), Period(), RsiPeriod, PRICE_CLOSE);
   ema_handle = iMA(Symbol(), Period(), EmaHTFPeriod, 0, MODE_EMA, PRICE_CLOSE);
   bb_handle  = iBands(Symbol(), Period(), BBPeriod, 0, BBDeviation, PRICE_CLOSE);
   kama_handle = iAMA(Symbol(), Period(), KamaPeriod, KamaFast, KamaSlow, 0, PRICE_CLOSE);

   if(adx_handle == INVALID_HANDLE || atr_handle == INVALID_HANDLE || 
      rsi_handle == INVALID_HANDLE || ema_handle == INVALID_HANDLE || 
      bb_handle == INVALID_HANDLE || kama_handle == INVALID_HANDLE)
     {
      Print("Error creating indicator handles.");
      return(INIT_FAILED);
     }

   min_bars_required = EmaHTFPeriod + 50;
   IndicatorSetString(INDICATOR_SHORTNAME, "Regime-Switching Indicator (V2)");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//|------------------------------------------------------------------|
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(rates_total < min_bars_required)
      return(0);

   int limit = rates_total - prev_calculated;
   if(limit > 0)
      limit = rates_total - min_bars_required;

   // Arrays for copying data
   double adx_main[], adx_plus[], adx_minus[];
   double atr_val[];
   double rsi_val[];
   double ema_val[];
   double bb_mid[], bb_up[], bb_low[];
   double kama_val[];

   ArraySetAsSeries(adx_main, true);
   ArraySetAsSeries(adx_plus, true);
   ArraySetAsSeries(adx_minus, true);
   ArraySetAsSeries(atr_val, true);
   ArraySetAsSeries(rsi_val, true);
   ArraySetAsSeries(ema_val, true);
   ArraySetAsSeries(bb_mid, true);
   ArraySetAsSeries(bb_up, true);
   ArraySetAsSeries(bb_low, true);
   ArraySetAsSeries(kama_val, true);

   static double st_upper[];
   static double st_lower[];
   static double st_val[];
   static double st_dir[];
   
   if(ArraySize(st_upper) != rates_total)
     {
      ArrayResize(st_upper, rates_total);
      ArrayResize(st_lower, rates_total);
      ArrayResize(st_val, rates_total);
      ArrayResize(st_dir, rates_total);
     }

   // Loop from oldest historical bar to current
   for(int i = limit; i >= 0; i--)
     {
      BuyBuffer[i] = 0.0;
      SellBuffer[i] = 0.0;

      // Copy values for bar i
      if(CopyBuffer(adx_handle, 0, i, 1, adx_main) < 1) continue;
      if(CopyBuffer(adx_handle, 1, i, 1, adx_plus) < 1) continue;
      if(CopyBuffer(adx_handle, 2, i, 1, adx_minus) < 1) continue;
      if(CopyBuffer(atr_handle, 0, i, 1, atr_val) < 1) continue;
      if(CopyBuffer(bb_handle, 0, i, 1, bb_mid) < 1) continue;
      if(CopyBuffer(bb_handle, 1, i, 1, bb_up) < 1) continue;
      if(CopyBuffer(bb_handle, 2, i, 1, bb_low) < 1) continue;
      if(CopyBuffer(ema_handle, 0, i, 1, ema_val) < 1) continue;
      if(CopyBuffer(kama_handle, 0, i, 1, kama_val) < 1) continue;

      // Calculate new indicators for research/dashboard use
      double alma = CalculateALMA(i, rates_total, AlmaPeriod, AlmaOffset, AlmaSigma, close);
      double vwap = CalculateVWAP(i, rates_total, high, low, close, volume, time);
      double donchian_upper, donchian_lower, donchian_middle;
      CalculateDonchian(i, rates_total, DonchianPeriod, high, low, donchian_upper, donchian_lower, donchian_middle);
      
      // Calculate SuperTrend
      int offset = rates_total - 1 - i;
      double hl2 = (high[offset] + low[offset]) / 2.0;
      double upperband = hl2 + SuperTrendMult * atr_val[0];
      double lowerband = hl2 - SuperTrendMult * atr_val[0];
      
      if(offset == 0)
        {
         st_upper[offset] = upperband;
         st_lower[offset] = lowerband;
         st_val[offset] = upperband;
         st_dir[offset] = 1.0;
        }
      else
        {
         if(close[offset-1] <= st_upper[offset-1])
            st_upper[offset] = MathMin(upperband, st_upper[offset-1]);
         else
            st_upper[offset] = upperband;
            
         if(close[offset-1] >= st_lower[offset-1])
            st_lower[offset] = MathMax(lowerband, st_lower[offset-1]);
         else
            st_lower[offset] = lowerband;
            
         if(close[offset] > st_upper[offset-1])
            st_dir[offset] = 1.0;
         else if(close[offset] < st_lower[offset-1])
            st_dir[offset] = -1.0;
         else
            st_dir[offset] = st_dir[offset-1];
            
         if(st_dir[offset] == 1.0)
            st_val[offset] = st_lower[offset];
         else
            st_val[offset] = st_upper[offset];
        }
      double supertrend = st_val[offset];
      double supertrend_dir = st_dir[offset];
      double kama = kama_val[0];

      // Calculate Choppiness Index manually
      double chop = CalculateChoppiness(i, rates_total, high, low, close);
      if(chop < 0) continue;

      // Calculate Stochastic RSI manually
      double stoch_k = CalculateStochRSI(i, rates_total, close);
      if(stoch_k < 0) continue;

      // Calculate standard RSI
      double rsi_buffer[];
      if(CopyBuffer(rsi_handle, 0, i, 1, rsi_buffer) < 1) continue;
      double rsi = rsi_buffer[0];

      // Regime check
      bool is_ranging = (chop > ChopThreshold || adx_main[0] < TrendThreshold);
      bool is_uptrend = (close[rates_total - 1 - i] > ema_val[0]);
      bool is_downtrend = (close[rates_total - 1 - i] < ema_val[0]);

      if(is_ranging)
        {
         // Long MR setup: Price <= BB Lower, StochRSI K < 15, RSI < 40, Trend is UP
         if(is_uptrend && close[rates_total - 1 - i] <= bb_low[0] && stoch_k < OSLevel && rsi < 40)
           {
            BuyBuffer[i] = low[rates_total - 1 - i] - 2 * atr_val[0];
            TPBuffer[i] = bb_mid[0];
            SLBuffer[i] = close[rates_total - 1 - i] - (atr_val[0] * SL_ATR_Mult);
           }
         // Short MR setup: Price >= BB Upper, StochRSI K > 85, RSI > 60, Trend is DOWN
         else if(is_downtrend && close[rates_total - 1 - i] >= bb_up[0] && stoch_k > OBLevel && rsi > 60)
           {
            SellBuffer[i] = high[rates_total - 1 - i] + 2 * atr_val[0];
            TPBuffer[i] = bb_mid[0];
            SLBuffer[i] = close[rates_total - 1 - i] + (atr_val[0] * SL_ATR_Mult);
           }
        }
     }

   // Update HUD Dashboard
   DrawDashboard(rates_total);

   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Calculate Choppiness Index for bar index i                       |
//|------------------------------------------------------------------|
double CalculateChoppiness(int i, int rates_total, const double &high[], const double &low[], const double &close[])
  {
   int offset = rates_total - 1 - i;
   if(offset < ChopPeriod) return(-1.0);

   double sum_tr = 0.0;
   double max_high = high[offset];
   double min_low = low[offset];

   for(int k = 0; k < ChopPeriod; k++)
     {
      int idx = offset - k;
      // True range calculation
      double tr = high[idx] - low[idx];
      if(idx < rates_total - 1)
        {
         double tr2 = MathAbs(high[idx] - close[idx + 1]);
         double tr3 = MathAbs(low[idx] - close[idx + 1]);
         tr = MathMax(tr, MathMax(tr2, tr3));
        }
      sum_tr += tr;

      if(high[idx] > max_high) max_high = high[idx];
      if(low[idx] < min_low) min_low = low[idx];
     }

   double range = max_high - min_low;
   if(range == 0.0) return(0.0);

   double ci = 100.0 * MathLog10(sum_tr / range) / MathLog10(ChopPeriod);
   return(ci);
  }

//+------------------------------------------------------------------+
//| Calculate Stochastic RSI K line                                  |
//|------------------------------------------------------------------|
double CalculateStochRSI(int i, int rates_total, const double &close[])
  {
   int offset = rates_total - 1 - i;
   if(offset < StochPeriod * 2) return(-1.0);

   double rsi_vals[];
   ArrayResize(rsi_vals, StochPeriod);
   
   for(int k = 0; k < StochPeriod; k++)
     {
      double temp[];
      if(CopyBuffer(rsi_handle, 0, i + k, 1, temp) < 1) return(-1.0);
      rsi_vals[k] = temp[0];
     }

   double min_rsi = rsi_vals[0];
   double max_rsi = rsi_vals[0];

   for(int k = 1; k < StochPeriod; k++)
     {
      if(rsi_vals[k] < min_rsi) min_rsi = rsi_vals[k];
      if(rsi_vals[k] > max_rsi) max_rsi = rsi_vals[k];
     }

   double diff = max_rsi - min_rsi;
   if(diff == 0.0) return(50.0);

   double stoch_k = 100.0 * (rsi_vals[0] - min_rsi) / diff;
   return(stoch_k);
  }

//+------------------------------------------------------------------+
//| Draw graphical user interface dashboard                          |
//|------------------------------------------------------------------|
void DrawDashboard(int rates_total)
  {
   double adx_vals[], chop_val, stoch_k, rsi_val[];
   double close_last[], ema_last[];
   
   if(CopyBuffer(adx_handle, 0, 0, 1, adx_vals) < 1) return;
   if(CopyBuffer(ema_handle, 0, 0, 1, ema_last) < 1) return;
   
   // Temporary copy close
   MqlRates rates[];
   if(CopyRates(Symbol(), Period(), 0, 1, rates) < 1) return;
   double current_close = rates[0].close;

   // Get indicator arrays for manual calculations
   double high_arr[], low_arr[], close_arr[];
   CopyHigh(Symbol(), Period(), 0, ChopPeriod + 2, high_arr);
   CopyLow(Symbol(), Period(), 0, ChopPeriod + 2, low_arr);
   CopyClose(Symbol(), Period(), 0, StochPeriod * 2 + 2, close_arr);

   ArraySetAsSeries(high_arr, true);
   ArraySetAsSeries(low_arr, true);
   ArraySetAsSeries(close_arr, true);

   chop_val = CalculateChoppiness(0, rates_total, high_arr, low_arr, close_arr);
   stoch_k = CalculateStochRSI(0, rates_total, close_arr);

   bool is_ranging = (chop_val > ChopThreshold || adx_vals[0] < TrendThreshold);
   string regime_str = is_ranging ? "RANGING (MR MODE)" : "TRENDING (TF MODE)";
   color  regime_col = is_ranging ? clrGold : clrDeepSkyBlue;

   string macro_trend = (current_close > ema_last[0]) ? "BULLISH (UPTREND)" : "BEARISH (DOWNTREND)";
   color  trend_col = (current_close > ema_last[0]) ? clrLime : clrRed;

   string label_text = "=== SYSTEM METRICS ===\n"+
                       "Regime: " + regime_str + "\n" +
                       "HTF Trend: " + macro_trend + "\n" +
                       "Choppiness Index: " + DoubleToString(chop_val, 1) + "\n" +
                       "ADX Value: " + DoubleToString(adx_vals[0], 1) + "\n" +
                       "StochRSI: " + DoubleToString(stoch_k, 1);

    Comment(label_text);
  }

//+------------------------------------------------------------------+
//| Calculate ALMA manually using Gaussian weights                   |
//+------------------------------------------------------------------+
double CalculateALMA(int i, int rates_total, int period, double offset, double sigma, const double &price[])
  {
   int offset_idx = rates_total - 1 - i;
   if(offset_idx < period) return(price[offset_idx]);
   
   double m = offset * (period - 1);
   double s = period / sigma;
   double weight_sum = 0.0;
   double sum = 0.0;
   
   for(int k = 0; k < period; k++)
     {
      double weight = MathExp(-MathPow(k - m, 2.0) / (2.0 * s * s));
      sum += price[offset_idx - (period - 1 - k)] * weight;
      weight_sum += weight;
     }
   
   if(weight_sum == 0.0) return(price[offset_idx]);
   return(sum / weight_sum);
  }

//+------------------------------------------------------------------+
//| Calculate VWAP manually (cumulative since start of day)          |
//+------------------------------------------------------------------+
double CalculateVWAP(int i, int rates_total, const double &high[], const double &low[], const double &close[], const long &volume[], const datetime &time[])
  {
   int offset = rates_total - 1 - i;
   if(offset < 0) return(0.0);
   
   MqlDateTime dt;
   TimeToStruct(time[offset], dt);
   
   double sum_pv = 0.0;
   double sum_vol = 0.0;
   
   for(int k = offset; k >= 0; k--)
     {
      MqlDateTime temp_dt;
      TimeToStruct(time[k], temp_dt);
      
      if(temp_dt.day != dt.day || temp_dt.mon != dt.mon || temp_dt.year != dt.year)
         break;
         
      double typical_price = (high[k] + low[k] + close[k]) / 3.0;
      sum_pv += typical_price * (double)volume[k];
      sum_vol += (double)volume[k];
     }
   
   if(sum_vol == 0.0) return(close[offset]);
   return(sum_pv / sum_vol);
  }

//+------------------------------------------------------------------+
//| Calculate Donchian Channel manually                              |
//+------------------------------------------------------------------+
void CalculateDonchian(int i, int rates_total, int period, const double &high[], const double &low[], double &upper, double &lower, double &middle)
  {
   int offset = rates_total - 1 - i;
   if(offset < period)
     {
      upper = high[offset];
      lower = low[offset];
      middle = (upper + lower) / 2.0;
      return;
     }
   
   upper = high[offset];
   lower = low[offset];
   
   for(int k = 0; k < period; k++)
     {
      int idx = offset - k;
      if(high[idx] > upper) upper = high[idx];
      if(low[idx] < lower)  lower = low[idx];
     }
   
   middle = (upper + lower) / 2.0;
  }
//+------------------------------------------------------------------+
