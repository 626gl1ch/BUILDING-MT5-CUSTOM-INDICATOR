//+------------------------------------------------------------------+
//|                                        SMC_Liquidity_Sweep.mq5   |
//|                                  Copyright 2026, Advanced Coder  |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Advanced Coder"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   4

//--- plot SwingHigh
#property indicator_label1  "Swing High"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrCrimson
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot SwingLow
#property indicator_label2  "Swing Low"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDodgerBlue
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- plot BuyArrow
#property indicator_label3  "Buy Signal"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrLime
#property indicator_width3  4

//--- plot SellArrow
#property indicator_label4  "Sell Signal"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrRed
#property indicator_width4  4

//--- Input Parameters
input int    InpLookback  = 20;    // Swing Lookback Period
input int    InpATRPeriod = 14;    // ATR Period
input double InpSL_ATR    = 0.5;   // Stop Loss (in ATR)
input double InpTP_ATR    = 2.0;   // Take Profit (in ATR)
input bool   InpDrawZones = true;  // Draw SL/TP Zones (Rectangles)

//--- Indicator Buffers
double SwingHighBuffer[];
double SwingLowBuffer[];
double BuyArrowBuffer[];
double SellArrowBuffer[];

//--- Internal Variables
int atrHandle;
double atrBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0, SwingHighBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, SwingLowBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, BuyArrowBuffer, INDICATOR_DATA);
   SetIndexBuffer(3, SellArrowBuffer, INDICATOR_DATA);

//--- set empty values for lines
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, 0.0);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, 0.0);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, 0.0);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, 0.0);

//--- set arrow codes (233=Up, 234=Down)
   PlotIndexSetInteger(2, PLOT_ARROW, 233);
   PlotIndexSetInteger(3, PLOT_ARROW, 234);

//--- get ATR handle
   atrHandle = iATR(_Symbol, _Period, InpATRPeriod);
   if(atrHandle == INVALID_HANDLE)
     {
      Print("Error creating ATR indicator");
      return(INIT_FAILED);
     }
     
   ArraySetAsSeries(atrBuffer, true);
   ArraySetAsSeries(SwingHighBuffer, true);
   ArraySetAsSeries(SwingLowBuffer, true);
   ArraySetAsSeries(BuyArrowBuffer, true);
   ArraySetAsSeries(SellArrowBuffer, true);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
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
//--- Set arrays as series (index 0 is newest candle)
   ArraySetAsSeries(time, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);

//--- Check for insufficient data
   if(rates_total < InpLookback * 2 + 1)
      return(0);

//--- Amount of data to copy
   int limit = rates_total - prev_calculated;
   if(prev_calculated == 0)
     {
      limit = rates_total - (InpLookback * 2 + 1) - 1;
      // Initialize arrays with 0
      ArrayInitialize(SwingHighBuffer, 0.0);
      ArrayInitialize(SwingLowBuffer, 0.0);
      ArrayInitialize(BuyArrowBuffer, 0.0);
      ArrayInitialize(SellArrowBuffer, 0.0);
     }
   
   if(CopyBuffer(atrHandle, 0, 0, limit + 1, atrBuffer) <= 0)
      return(0);

//--- Main Calculation Loop
// We process from oldest to newest (limit down to 0) to propagate the swing lines properly
   for(int i = limit; i >= 0 && !IsStopped(); i--)
     {
      BuyArrowBuffer[i] = 0.0;
      SellArrowBuffer[i] = 0.0;
      
      // Carry forward previous swing values (from older candle to newer)
      if(i < rates_total - 1)
        {
         SwingHighBuffer[i] = SwingHighBuffer[i+1];
         SwingLowBuffer[i] = SwingLowBuffer[i+1];
        }

      // Check if a swing high formed `InpLookback` bars ago (i + InpLookback)
      // A swing point at index T is confirmed if T is the extreme in [T - InpLookback, T + InpLookback]
      int window = 2 * InpLookback + 1;
      
      // Ensure we don't go out of bounds backwards
      if(i + window < rates_total)
        {
         // Find extreme index in the window starting from `i` (newest in window) to `i+window` (oldest)
         int max_idx = i;
         int min_idx = i;
         double max_val = high[i];
         double min_val = low[i];
         
         for(int k = 0; k < window; k++)
           {
            if(high[i+k] > max_val)
              {
               max_val = high[i+k];
               max_idx = i+k;
              }
            if(low[i+k] < min_val)
              {
               min_val = low[i+k];
               min_idx = i+k;
              }
           }
           
         // If the peak occurred exactly in the middle of the window (which is i + InpLookback), we confirm it NOW at index i
         if(max_idx == i + InpLookback)
           {
            SwingHighBuffer[i] = max_val;
           }
           
         if(min_idx == i + InpLookback)
           {
            SwingLowBuffer[i] = min_val;
           }
        }
        
      // SIGNAL LOGIC (SMC Sweep + Rejection)
      // We check if current candle `i` sweeps and rejects the swing point established by `i+1`
      if(i < rates_total - 2 && SwingHighBuffer[i+1] > 0 && SwingLowBuffer[i+1] > 0)
        {
         double prev_sh = SwingHighBuffer[i+1];
         double prev_sl = SwingLowBuffer[i+1];
         
         // LONG: Sweep of Swing Low
         // 1. Low[i] < prev_sl (Sweep)
         // 2. Close[i] > prev_sl (Rejection)
         // 3. Close[i+1] > prev_sl (Ensure it wasn't already broken)
         if(low[i] < prev_sl && close[i] > prev_sl && close[i+1] > prev_sl)
           {
            BuyArrowBuffer[i] = low[i] - (atrBuffer[i] * 0.5); // Place arrow below low
            
            if(InpDrawZones)
              {
               DrawRiskZone("LongZone_" + IntegerToString(i) + "_" + TimeToString(time[i]), time[i], close[i], true, atrBuffer[i]);
              }
           }
           
         // SHORT: Sweep of Swing High
         // 1. High[i] > prev_sh (Sweep)
         // 2. Close[i] < prev_sh (Rejection)
         // 3. Close[i+1] < prev_sh
         if(high[i] > prev_sh && close[i] < prev_sh && close[i+1] < prev_sh)
           {
            SellArrowBuffer[i] = high[i] + (atrBuffer[i] * 0.5); // Place arrow above high
            
            if(InpDrawZones)
              {
               DrawRiskZone("ShortZone_" + IntegerToString(i) + "_" + TimeToString(time[i]), time[i], close[i], false, atrBuffer[i]);
              }
           }
        }
     }
     
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, "LongZone_");
   ObjectsDeleteAll(0, "ShortZone_");
  }

//+------------------------------------------------------------------+
//| Draw Risk Zone Rectangles                                        |
//+------------------------------------------------------------------+
void DrawRiskZone(string name, datetime t1, double entry_price, bool is_long, double atr)
  {
   // Draw width: 6 candles forward to visually represent the trade duration without cluttering
   datetime t2 = t1 + (PeriodSeconds() * 6);
   
   double sl_price = is_long ? (entry_price - (atr * InpSL_ATR)) : (entry_price + (atr * InpSL_ATR));
   double tp_price = is_long ? (entry_price + (atr * InpTP_ATR)) : (entry_price - (atr * InpTP_ATR));
   
   string sl_name = name + "_SL";
   string tp_name = name + "_TP";
   
   // Create SL Rectangle (Red-ish)
   ObjectCreate(0, sl_name, OBJ_RECTANGLE, 0, t1, entry_price, t2, sl_price);
   ObjectSetInteger(0, sl_name, OBJPROP_COLOR, clrLightPink);
   ObjectSetInteger(0, sl_name, OBJPROP_BACK, true);
   ObjectSetInteger(0, sl_name, OBJPROP_FILL, true);
   ObjectSetInteger(0, sl_name, OBJPROP_HIDDEN, true); // Hide from object list
   ObjectSetString(0, sl_name, OBJPROP_TOOLTIP, "Stop Loss Zone");
   
   // Create TP Rectangle (Green-ish)
   ObjectCreate(0, tp_name, OBJ_RECTANGLE, 0, t1, entry_price, t2, tp_price);
   ObjectSetInteger(0, tp_name, OBJPROP_COLOR, clrPaleGreen);
   ObjectSetInteger(0, tp_name, OBJPROP_BACK, true);
   ObjectSetInteger(0, tp_name, OBJPROP_FILL, true);
   ObjectSetInteger(0, tp_name, OBJPROP_HIDDEN, true);
   ObjectSetString(0, tp_name, OBJPROP_TOOLTIP, "Take Profit Zone");
  }
//+------------------------------------------------------------------+
