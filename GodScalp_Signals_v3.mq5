//+------------------------------------------------------------------+
//|                                            GodScalp_Signals_v3.mq5|
//|                                   GOD SCALP v3 - Glitch EnterPrice|
//+------------------------------------------------------------------+
#property copyright "Glitch EnterPrice"
#property version   "1.00"
#property strict
#property indicator_chart_window
#property indicator_buffers 10
#property indicator_plots   10

#property indicator_label1  "ALMA 5m"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrMediumPurple
#property indicator_width1  2

#property indicator_label2  "BB Basis"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDimGray
#property indicator_style2  STYLE_DASH
#property indicator_width2  1

#property indicator_label3  "BB Upper"
#property indicator_type3   DRAW_NONE
#property indicator_color3  clrTeal
#property indicator_width3  1

#property indicator_label4  "BB Lower"
#property indicator_type4   DRAW_NONE
#property indicator_color4  clrTeal
#property indicator_width4  1

#property indicator_label5  "Long Entry Signal"
#property indicator_type5   DRAW_ARROW
#property indicator_color5  clrLime
#property indicator_width5  3

#property indicator_label6  "Short Entry Signal"
#property indicator_type6   DRAW_ARROW
#property indicator_color6  clrRed
#property indicator_width6  3

#property indicator_label7  "Stop Loss"
#property indicator_type7   DRAW_ARROW
#property indicator_color7  clrCrimson
#property indicator_width7  2

#property indicator_label8  "First Target"
#property indicator_type8   DRAW_ARROW
#property indicator_color8  clrLimeGreen
#property indicator_width8  2

#property indicator_label9  "Runner Target"
#property indicator_type9   DRAW_ARROW
#property indicator_color9  clrDodgerBlue
#property indicator_width9  2

#property indicator_label10 "ALMA 50m"
#property indicator_type10  DRAW_LINE
#property indicator_color10 clrOrange
#property indicator_width10 2

input group "=== ALMA (Direction) ==="
input int    InpAlmaLen                   = 9;
input int    InpAlmaSlowLen               = 50;
input double InpAlmaOffset                = 0.85;
input double InpAlmaSigma                 = 6.0;

input group "=== Bollinger Bands (Zone) ==="
input int    InpBBLen                     = 20;
input double InpBBStdDev                  = 2.0;

input group "=== MACD (Trigger) ==="
input int    InpMacdFast                  = 5;
input int    InpMacdSlow                  = 13;
input int    InpMacdSignal                = 6;
input int    InpMacdFlipConfirmBars       = 2;       // consecutive against-bars before exit alert

input group "=== ATR (Volatility) ==="
input int    InpAtrFast                   = 14;
input double InpAtrSpikeExitMult          = 1.6;      // ATR14 >= this x entry ATR, against you = exit

input group "=== Choppiness Index (Regime) ==="
input int    InpChopLen                   = 14;
input double InpChopTrendTh               = 38.2;
input double InpChopRangeTh               = 51.5;

input group "=== Entry State Machine ==="
input int    InpPendingMaxBars            = 50;       // bars a pending setup stays armed
input int    InpFrozenMaxBars             = 25;       // bars a setup stays frozen after a bad trigger candle
input int    InpTrendStructLookback       = 6;        // bars back for pullback-low/rally-high
input int    InpMRWickLookback            = 4;        // bars back for sweep-wick extreme
input double InpTrendRRFloor              = 0.55;
input double InpMRRRFloor                 = 0.28;

input group "=== Stop-Loss Placement ==="
input double InpTrendSLAtrMult            = 1.0;
input double InpTrendSLFloorAtr           = 0.30;
input double InpTrendSLStructBufAtr       = 0.10;
input double InpMRSLAtrMult               = 1.2;
input double InpMRSLStructBufAtr          = 0.10;
input double InpMRSLEnvMin                = 0.5;      // MR stop distance floor, x ATR-mult distance
input double InpMRSLEnvMax                = 1.3;      // MR stop distance ceiling, x ATR-mult distance

input group "=== First Target Placement ==="
input double InpTrendTPAtrMult            = 1.5;
input double InpTrendTPFloorAtr           = 0.50;

input group "=== Partial + Runner (v2 exit engine) ==="
input double InpPartialFraction           = 0.35;     // fraction to bank at first target
input double InpRunnerTrendAtrMult        = 2.5;       // runner extends this far beyond trend TP
input double InpRunnerMRBandFrac          = 0.85;      // runner = this fraction of the way to opp band

input group "=== Trade Management ==="
input double InpBreakevenAtR              = 1.0;
input double InpTrailStartR               = 1.5;
input double InpTrailAtrMult              = 0.8;
input double InpCostBufferPoints          = 0;         // 0 = auto (2x current spread)

input group "=== Risk Controls ==="
input int    InpCooldownBars              = 6;
input double InpDailyLossLimitPct         = 3.0;

input group "=== Alerts ==="
input bool   InpEnablePopupAlert          = true;
input bool   InpEnableSoundAlert          = true;
input string InpAlertSound                = "alert.wav";

//--- Plotted buffers
double BufAlma5[], BufBBBasis[], BufBBUpper[], BufBBLower[], BufArrowUp[], BufArrowDown[];
double BufSL[], BufTP[], BufRunner[], BufAlma50[];

//--- Internal (non-plotted) computed series
double g_atrFast[];
double g_emaFast[], g_emaSlow[], g_macdLine[], g_macdSignal[], g_macdHist[];
double g_chop[];
int    g_regime[];
int    g_trendBias[];              


//--- Local copies of price series
datetime g_time[];
double   g_open[], g_high[], g_low[], g_close[];

//--- Pending-setup state machine
bool   g_pendActive   = false;
int    g_pendDir      = 0;
int    g_pendMode     = 0;      
int    g_pendBarsLeft = 0;

//--- Frozen state (bad candle delayed entry)
bool   g_frozenActive   = false;
int    g_frozenDir      = 0;
int    g_frozenMode     = 0;
int    g_frozenBarsLeft = 0;

//--- Last fired signal
int      g_lastSignalDir  = 0;
int      g_lastSignalMode = 0;
datetime g_lastSignalTime = 0;
double   g_lastSignalPrice= 0;

//--- Live active-trade tracking
bool     g_hasActiveTrade    = false;
ulong    g_tradeTicket       = 0;
int      g_tradeDir          = 0;
int      g_tradeMode         = -1;      
double   g_tradeEntryPrice   = 0;
double   g_tradeSL           = 0;
double   g_tradeT1           = 0;
double   g_tradeRunner       = 0;
bool     g_tradePartialTaken = false;
bool     g_tradeBreakevenMoved = false;
bool     g_tradeTrailingActive = false;
double   g_tradeATRatEntry   = 0;
int      g_tradeHistFlipStreak = 0;

//--- Risk-control state
datetime g_cooldownUntilBarTime = 0;
datetime g_currentDay           = 0;
double   g_dayStartEquity       = 0;
bool     g_dailyHalt            = false;

//--- Signal-quality counters
int g_cntArmed      = 0;
int g_cntFired      = 0;
int g_cntExpired    = 0;
int g_cntRejectedRR = 0;

int      g_lastProcessedIdx = -1;

//+------------------------------------------------------------------+
//| Forward declarations                                             |
//+------------------------------------------------------------------+
double TrueRange(int i);
void   UpdateATR(int i,int period,double &arr[]);
void   UpdateBB(int i);
void   UpdateEMA(int i,int len,double &emaArr[],const double &src[]);
void   UpdateCHOP(int i);
void   UpdateRegime(int i);
double CalcALMA(int i,int length,double offset,double sigma,const double &price[]);
double RecentExtreme(int i,int dir,int lookback);
void   ComputeTrendStopTarget(int dir,double entry,double structExt,double oppBand,double atrF,double &sl,double &tp);
double ComputeMRStop(int dir,double entry,double wickExt,double atrF);
double BreakevenPrice(int dir,double entry);
string ModeStr(int mode);
string RegimeStr(int r);
void   AlertMsg(string msg);
void   ProcessClosedBar(int i,bool isLive);
bool   FireEntry(int i,int dir,int mode,bool isLive);
void   OnNewBarTradeUpdate(int i);
void   ManageActiveTrade();
void   InitNewTrade(ulong ticket);
bool   DetermineLastTradeWasLoss(ulong posTicket);
void   CheckDailyCircuitBreaker();
void   CheckPriceExitTriggers();
void   DrawTradeLines();
void   SetHLine(string name,double price,color c,ENUM_LINE_STYLE style);
void   ClearTradeLines();
void   UpdateStatusComment();
string CounterKey(string suffix);
void   LoadCounters();
void   SaveCounter(string suffix,int val);
void   IncArmed();
void   IncFired();
void   IncExpired();
void   IncRejectedRR();

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufAlma5,     INDICATOR_DATA);
   SetIndexBuffer(1, BufBBBasis,   INDICATOR_DATA);
   SetIndexBuffer(2, BufBBUpper,   INDICATOR_DATA);
   SetIndexBuffer(3, BufBBLower,   INDICATOR_DATA);
   SetIndexBuffer(4, BufArrowUp,   INDICATOR_DATA);
   SetIndexBuffer(5, BufArrowDown, INDICATOR_DATA);
   SetIndexBuffer(6, BufSL,        INDICATOR_DATA);
   SetIndexBuffer(7, BufTP,        INDICATOR_DATA);
   SetIndexBuffer(8, BufRunner,    INDICATOR_DATA);
   SetIndexBuffer(9, BufAlma50,    INDICATOR_DATA);

   PlotIndexSetInteger(4, PLOT_ARROW, 233);
   PlotIndexSetInteger(5, PLOT_ARROW, 234);
   PlotIndexSetDouble(4, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(5, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   PlotIndexSetInteger(6, PLOT_ARROW, 175); 
   PlotIndexSetInteger(7, PLOT_ARROW, 175); 
   PlotIndexSetInteger(8, PLOT_ARROW, 175); 
   PlotIndexSetDouble(6, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(7, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(8, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   IndicatorSetString(INDICATOR_SHORTNAME, "GOD SCALP Signals v3");
   IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

   g_pendActive = false;
   g_hasActiveTrade = false;
   g_dailyHalt = false;
   g_currentDay = 0;
   g_cooldownUntilBarTime = 0;
   g_lastProcessedIdx = -1;
   LoadCounters();

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Comment("");
   ObjectsDeleteAll(0, "GS_Vis_");
   ClearTradeLines();
}

//+------------------------------------------------------------------+
//| Helper calc functions                                            |
//+------------------------------------------------------------------+
double TrueRange(int i)
{
   if(i<=0) return g_high[i]-g_low[i];
   double hl = g_high[i]-g_low[i];
   double hc = MathAbs(g_high[i]-g_close[i-1]);
   double lc = MathAbs(g_low[i]-g_close[i-1]);
   return MathMax(hl, MathMax(hc,lc));
}

void UpdateATR(int i,int period,double &arr[])
{
   if(i < period-1){ arr[i]=0; return; }
   if(i == period-1)
   {
      double sum=0;
      for(int k=0;k<period;k++) sum += TrueRange(k);
      arr[i] = sum/period;
      return;
   }
   double tr = TrueRange(i);
   arr[i] = (arr[i-1]*(period-1)+tr)/period;
}

void UpdateBB(int i)
{
   int period = InpBBLen;
   if(i<period-1){ BufBBBasis[i]=g_close[i]; BufBBUpper[i]=g_close[i]; BufBBLower[i]=g_close[i]; return; }
   double sum=0;
   for(int k=i-period+1;k<=i;k++) sum+=g_close[k];
   double mean = sum/period;
   double sq=0;
   for(int k=i-period+1;k<=i;k++){ double d=g_close[k]-mean; sq += d*d; }
   double sd = MathSqrt(sq/period);
   BufBBBasis[i]=mean;
   BufBBUpper[i]=mean+InpBBStdDev*sd;
   BufBBLower[i]=mean-InpBBStdDev*sd;
}

void UpdateEMA(int i,int len,double &emaArr[],const double &src[])
{
   double alpha = 2.0/((double)len+1.0);
   if(i==0){ emaArr[i]=src[i]; return; }
   emaArr[i] = alpha*src[i] + (1.0-alpha)*emaArr[i-1];
}

void UpdateCHOP(int i)
{
   int period = InpChopLen;
   if(i<period-1){ g_chop[i]=50.0; return; }
   double trSum=0;
   for(int k=i-period+1;k<=i;k++) trSum += TrueRange(k);
   double hh=g_high[i-period+1], ll=g_low[i-period+1];
   for(int k=i-period+1;k<=i;k++){ if(g_high[k]>hh) hh=g_high[k]; if(g_low[k]<ll) ll=g_low[k]; }
   double rng = hh-ll;
   if(rng<=0){ g_chop[i] = (i>0)? g_chop[i-1] : 50.0; return; }
   g_chop[i] = 100.0*MathLog10(trSum/rng)/MathLog10((double)period);
}

void UpdateRegime(int i)
{
   int prev = (i>0)? g_regime[i-1] : 1; // Default to trend (1) initially
   if(g_chop[i] < InpChopTrendTh) g_regime[i]=1;
   else if(g_chop[i] > InpChopRangeTh) g_regime[i]=2;
   else g_regime[i]=prev; // Hysteresis band
}

void UpdateTrendBias(int i)
{
   if(i<1) { g_trendBias[i] = 0; return; }
   bool crossUp   = (BufAlma5[i-1] <= BufAlma50[i-1] && BufAlma5[i] > BufAlma50[i]);
   bool crossDown = (BufAlma5[i-1] >= BufAlma50[i-1] && BufAlma5[i] < BufAlma50[i]);
   
   if(crossUp)        g_trendBias[i] = 1;
   else if(crossDown) g_trendBias[i] = -1;
   else               g_trendBias[i] = g_trendBias[i-1];
}

double CalcALMA(int i,int length,double offset,double sigma,const double &price[])
{
   if(i<length-1) return price[i];
   double m = MathFloor(offset*(length-1));
   double s = (double)length/sigma;
   double wsum=0, vsum=0;
   for(int j=0;j<length;j++)
   {
      double w = MathExp(-((j-m)*(j-m))/(2.0*s*s));
      wsum += w;
      vsum += w*price[i-length+1+j];
   }
   if(wsum==0) return price[i];
   return vsum/wsum;
}

double RecentExtreme(int i,int dir,int lookback)
{
   int from = MathMax(0, i-lookback);
   double ext = (dir==1)? g_low[i] : g_high[i];
   for(int k=from;k<=i;k++)
   {
      if(dir==1) ext = MathMin(ext, g_low[k]);
      else       ext = MathMax(ext, g_high[k]);
   }
   return ext;
}

void ComputeTrendStopTarget(int dir,double entry,double structExt,double oppBand,double atrF,double &sl,double &tp)
{
   double buf = InpTrendSLStructBufAtr*atrF;
   double structDist;
   if(dir==1) structDist = MathMax(entry-structExt,0.0)+buf;
   else       structDist = MathMax(structExt-entry,0.0)+buf;
   double atrDist = InpTrendSLAtrMult*atrF;
   double slDist = MathMin(structDist, atrDist);
   slDist = MathMax(slDist, InpTrendSLFloorAtr*atrF);

   double structTpDist = MathAbs(oppBand-entry);
   double atrTpDist = InpTrendTPAtrMult*atrF;
   double tpDist = MathMin(structTpDist, atrTpDist);
   tpDist = MathMax(tpDist, InpTrendTPFloorAtr*atrF);

   if(dir==1){ sl = entry-slDist; tp = entry+tpDist; }
   else      { sl = entry+slDist; tp = entry-tpDist; }
}

double ComputeMRStop(int dir,double entry,double wickExt,double atrF)
{
   double buf = InpMRSLStructBufAtr*atrF;
   double structDist;
   if(dir==1) structDist = MathMax(entry-wickExt,0.0)+buf;
   else       structDist = MathMax(wickExt-entry,0.0)+buf;
   double atrDist = InpMRSLAtrMult*atrF;
   double slDist = MathMin(MathMax(structDist, InpMRSLEnvMin*atrDist), InpMRSLEnvMax*atrDist);
   return (dir==1)? entry-slDist : entry+slDist;
}

double BreakevenPrice(int dir,double entry)
{
   double bufPts = InpCostBufferPoints;
   if(bufPts<=0)
   {
      long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      bufPts = 2.0*(double)spread;
   }
   double bufPrice = bufPts*_Point;
   return (dir==1)? entry+bufPrice : entry-bufPrice;
}

string ModeStr(int mode)
{
   if(mode==0) return "TREND";
   if(mode==1) return "MEAN-REVERSION";
   return "UNCLASSIFIED";
}

string RegimeStr(int r)
{
   if(r==1) return "TREND";
   if(r==2) return "RANGE";
   return "TRANSITIONAL";
}

void AlertMsg(string msg)
{
   string full = "[GOD SCALP] "+_Symbol+" "+EnumToString((ENUM_TIMEFRAMES)_Period)+" - "+msg;
   if(InpEnablePopupAlert) Alert(full);
   if(InpEnableSoundAlert) PlaySound(InpAlertSound);
   Print(full);
}

string CounterKey(string suffix)
{
   return "GS_"+_Symbol+"_"+EnumToString((ENUM_TIMEFRAMES)_Period)+"_"+suffix;
}

void LoadCounters()
{
   g_cntArmed      = GlobalVariableCheck(CounterKey("Armed"))      ? (int)GlobalVariableGet(CounterKey("Armed"))      : 0;
   g_cntFired      = GlobalVariableCheck(CounterKey("Fired"))      ? (int)GlobalVariableGet(CounterKey("Fired"))      : 0;
   g_cntExpired    = GlobalVariableCheck(CounterKey("Expired"))    ? (int)GlobalVariableGet(CounterKey("Expired"))    : 0;
   g_cntRejectedRR = GlobalVariableCheck(CounterKey("RejectedRR")) ? (int)GlobalVariableGet(CounterKey("RejectedRR")) : 0;
}

void SaveCounter(string suffix,int val)
{
   GlobalVariableSet(CounterKey(suffix), (double)val);
}

void IncArmed()      { g_cntArmed++;      SaveCounter("Armed", g_cntArmed); }
void IncFired()      { g_cntFired++;      SaveCounter("Fired", g_cntFired); }
void IncExpired()    { g_cntExpired++;    SaveCounter("Expired", g_cntExpired); }
void IncRejectedRR() { g_cntRejectedRR++; SaveCounter("RejectedRR", g_cntRejectedRR); }

//+------------------------------------------------------------------+
//| Bad Candle (Indecision) Filter                                   |
//+------------------------------------------------------------------+
bool IsBadCandle(int i, int dir)
{
   if(i<1) return false;
   double hl = g_high[i] - g_low[i];
   double body = MathAbs(g_close[i] - g_open[i]);
   double upperWick = g_high[i] - MathMax(g_open[i], g_close[i]);
   double lowerWick = MathMin(g_open[i], g_close[i]) - g_low[i];
   
   if(hl < _Point) return true; // Dash / Four-price doji
   
   double bodyPct = body / hl;
   if(bodyPct < 0.10) return true; // Doji
   if(bodyPct < 0.30 && upperWick > body && lowerWick > body) return true; // Spinning top
   if(i>1 && g_high[i] <= g_high[i-1] && g_low[i] >= g_low[i-1]) return true; // Inside Bar
   if(body > g_atrFast[i] * 2.5) return true; // Exhaustion breakout
   
   // Direction-specific rejection candles
   if(dir == 1) {
      // For LONG, a massive upper wick (shooting star rejection) is dangerous
      if(upperWick > body * 2.0 && upperWick > lowerWick * 2.0) return true; 
   } else if(dir == -1) {
      // For SHORT, a massive lower wick (hammer rejection) is dangerous
      if(lowerWick > body * 2.0 && lowerWick > upperWick * 2.0) return true;
   }
   
   return false;
}

void ProcessAction(int i, int dir, int mode, bool isLive)
{
   if(IsBadCandle(i, dir))
   {
       g_frozenActive = true;
       g_frozenDir = dir;
       g_frozenMode = mode;
       g_frozenBarsLeft = InpFrozenMaxBars;
       if(isLive) Print("[GOD SCALP] Setup FROZEN due to Bad Candle at ", TimeToString(g_time[i]));
   }
   else
   {
       bool ok = FireEntry(i, dir, mode, isLive);
       if(isLive){ if(ok) IncFired(); else IncRejectedRR(); }
   }
}

//+------------------------------------------------------------------+
//| Entry state machine - runs once per newly closed bar             |
//+------------------------------------------------------------------+
void ProcessClosedBar(int i,bool isLive)
{
   if(i<1) return;
   
   //--- Step 0: Process Frozen Setups
   if(g_frozenActive)
   {
      // Note: Regime is intentionally not re-checked here. The regime at arm-time governs the trade.
      g_frozenBarsLeft--;
      bool isHealthy = !IsBadCandle(i, g_frozenDir);
      
      bool supportsDir = false;
      if(g_frozenDir == 1) {
         bool isBullish = (g_close[i] > g_open[i]);
         bool macdStillUp = (g_macdLine[i] > g_macdSignal[i]);
         supportsDir = (isBullish && macdStillUp && g_trendBias[i] == 1);
         if(g_macdLine[i] < g_macdSignal[i]) g_frozenBarsLeft = 0; // MACD flipped back
      } else {
         bool isBearish = (g_close[i] < g_open[i]);
         bool macdStillDown = (g_macdLine[i] < g_macdSignal[i]);
         supportsDir = (isBearish && macdStillDown && g_trendBias[i] == -1);
         if(g_macdLine[i] > g_macdSignal[i]) g_frozenBarsLeft = 0; // MACD flipped back
      }
      
      if(isHealthy && supportsDir)
      {
         bool ok = FireEntry(i, g_frozenDir, g_frozenMode, isLive);
         if(isLive){ if(ok) IncFired(); else IncRejectedRR(); }
         g_frozenActive = false;
         return; 
      }
      
      if(g_frozenBarsLeft <= 0)
      {
         g_frozenActive = false;
         if(isLive) IncExpired();
      }
      return; // Skip normal processing while frozen
   }

   double atrF = g_atrFast[i];



   bool macdCrossUp   = (g_macdLine[i-1]<=g_macdSignal[i-1] && g_macdLine[i]>g_macdSignal[i]);
   bool macdCrossDown = (g_macdLine[i-1]>=g_macdSignal[i-1] && g_macdLine[i]<g_macdSignal[i]);
   bool macdIsUp      = (g_macdLine[i]>g_macdSignal[i]);
   bool macdIsDown    = (g_macdLine[i]<g_macdSignal[i]);

   //--- Step 1: try to fire an already-armed pending setup
   if(g_pendActive)
   {
      // Note: Regime is intentionally not re-checked here. The regime at arm-time governs the trade.
      g_pendBarsLeft--;
      bool stillValid = true;
      // ALMA direction gate applies to BOTH trend and MR pending setups (persistent bias)
      if(g_pendDir==1) stillValid = (g_trendBias[i] == 1);
      else             stillValid = (g_trendBias[i] == -1);
      if(stillValid)
      {
          if(g_pendDir==1 && macdCrossUp)
          {
             ProcessAction(i, 1, g_pendMode, isLive);
             g_pendActive=false; return;
          }
          if(g_pendDir==-1 && macdCrossDown)
          {
             ProcessAction(i, -1, g_pendMode, isLive);
             g_pendActive=false; return;
          }
      }
      if(!stillValid || g_pendBarsLeft<=0)
      {
         g_pendActive=false;
         if(isLive) IncExpired();
      }
   }

   if(g_pendActive) return;

   //--- Step 2: look for a new setup to arm
   int regime = g_regime[i];

   if(regime==1) // trend
   {
      bool ltfUp   = (g_trendBias[i] == 1);
      bool ltfDown = (g_trendBias[i] == -1);

      if(ltfUp)
      {
         bool touched  = (g_low[i]<=BufAlma5[i] || g_low[i]<=BufBBBasis[i] || g_low[i]<=BufBBLower[i]);
         bool candleOk = (g_close[i]>g_open[i]);
         if(touched && candleOk)
         {
            if(macdCrossUp) // Aligned with pending standard (fresh cross)
            {
               ProcessAction(i, 1, 0, isLive);
            }
            else { g_pendActive=true; g_pendDir=1; g_pendMode=0; g_pendBarsLeft=InpPendingMaxBars; if(isLive) IncArmed(); }
         }
         return;
      }
      if(ltfDown)
      {
         bool touched  = (g_high[i]>=BufAlma5[i] || g_high[i]>=BufBBBasis[i] || g_high[i]>=BufBBUpper[i]);
         bool candleOk = (g_close[i]<g_open[i]);
         if(touched && candleOk)
         {
            if(macdCrossDown) // Aligned with pending standard (fresh cross)
            {
               ProcessAction(i, -1, 0, isLive);
            }
            else { g_pendActive=true; g_pendDir=-1; g_pendMode=0; g_pendBarsLeft=InpPendingMaxBars; if(isLive) IncArmed(); }
         }
         return;
      }
   }
   else if(regime==2) // range
   {
      // ALMA direction gate: only take MR longs in an uptrend structure, MR shorts in a downtrend
      bool almaUp   = (g_trendBias[i] == 1);
      bool almaDown = (g_trendBias[i] == -1);

      bool touchedDown = false;
      bool touchedUp = false;
      for(int k=0; k<=2; k++) {
         if(i-k > 0) {
            if(g_low[i-k] <= BufBBLower[i-k]) touchedDown = true;
            if(g_high[i-k] >= BufBBUpper[i-k]) touchedUp = true;
         }
      }
      
      bool reclaimedDown = (g_close[i] > BufBBLower[i]);
      bool reclaimedUp   = (g_close[i] < BufBBUpper[i]);
      bool candleOkLong  = (g_close[i] > g_open[i]);
      bool candleOkShort = (g_close[i] < g_open[i]);

      // Only arm/fire a LONG bounce if ALMA9 > ALMA50 (uptrend structure)
      if(almaUp && touchedDown && reclaimedDown && candleOkLong)
      {
         if(macdCrossUp)
         {
            ProcessAction(i, 1, 1, isLive);
         }
         else { g_pendActive=true; g_pendDir=1; g_pendMode=1; g_pendBarsLeft=InpPendingMaxBars; if(isLive) IncArmed(); }
         return;
      }
      // Only arm/fire a SHORT bounce if ALMA9 < ALMA50 (downtrend structure)
      if(almaDown && touchedUp && reclaimedUp && candleOkShort)
      {
         if(macdCrossDown)
         {
            ProcessAction(i, -1, 1, isLive);
         }
         else { g_pendActive=true; g_pendDir=-1; g_pendMode=1; g_pendBarsLeft=InpPendingMaxBars; if(isLive) IncArmed(); }
         return;
      }
   }
}

//+------------------------------------------------------------------+
//| Fire an entry: R:R filter, draw arrow, alert if live              |
//+------------------------------------------------------------------+
bool FireEntry(int i,int dir,int mode,bool isLive)
{
   double entry = g_close[i];
   double atrF  = g_atrFast[i];
   double sl, t1, runner;

   if(mode==0)
   {
      double structExt = RecentExtreme(i, dir, InpTrendStructLookback);
      double oppBand = (dir==1)? BufBBUpper[i] : BufBBLower[i];
      ComputeTrendStopTarget(dir, entry, structExt, oppBand, atrF, sl, t1);
      runner = (dir==1)? t1 + InpRunnerTrendAtrMult*atrF : t1 - InpRunnerTrendAtrMult*atrF;
   }
   else
   {
      double wickExt = RecentExtreme(i, dir, InpMRWickLookback);
      sl = ComputeMRStop(dir, entry, wickExt, atrF);
      t1 = BufBBBasis[i];
      double opp = (dir==1)? BufBBUpper[i] : BufBBLower[i];
      if(dir==1) runner = entry + InpRunnerMRBandFrac*(opp-entry);
      else       runner = entry - InpRunnerMRBandFrac*(entry-opp);
   }

   if(dir==1 && !(sl<entry && entry<t1)) return false;
   if(dir==-1 && !(t1<entry && entry<sl)) return false;

   double riskDist = MathAbs(entry-sl);
   double rewardDist = MathAbs(t1-entry);
   if(riskDist<=0) return false;
   double rr = rewardDist/riskDist;
   double rrFloor = (mode==0)? InpTrendRRFloor : InpMRRRFloor;
   if(rr < rrFloor) return false;

   if(dir==1) BufArrowUp[i]   = g_low[i]  - 0.15*atrF;
   else       BufArrowDown[i] = g_high[i] + 0.15*atrF;

   // Assign to buffers for immediate visual dash at the signal bar
   BufSL[i] = sl;
   BufTP[i] = t1;
   BufRunner[i] = runner;

   // Also draw short trendlines spanning forward so the user clearly sees the zone
   if(isLive || i > ArraySize(g_time) - 1000)
   {
      string objPrefix = "GS_Vis_"+IntegerToString(i)+"_";
      datetime tStart = g_time[i];
      datetime tEnd = tStart + 15 * PeriodSeconds(PERIOD_CURRENT);
      
      ObjectCreate(0, objPrefix+"SLBox", OBJ_RECTANGLE, 0, tStart, entry, tEnd, sl);
      ObjectSetInteger(0, objPrefix+"SLBox", OBJPROP_COLOR, clrMaroon);
      ObjectSetInteger(0, objPrefix+"SLBox", OBJPROP_FILL, true);
      ObjectSetInteger(0, objPrefix+"SLBox", OBJPROP_BACK, true);
      
      ObjectCreate(0, objPrefix+"TPBox", OBJ_RECTANGLE, 0, tStart, entry, tEnd, t1);
      ObjectSetInteger(0, objPrefix+"TPBox", OBJPROP_COLOR, clrDarkGreen);
      ObjectSetInteger(0, objPrefix+"TPBox", OBJPROP_FILL, true);
      ObjectSetInteger(0, objPrefix+"TPBox", OBJPROP_BACK, true);
      
      ObjectCreate(0, objPrefix+"EntryLine", OBJ_TREND, 0, tStart, entry, tEnd, entry);
      ObjectSetInteger(0, objPrefix+"EntryLine", OBJPROP_COLOR, clrSilver);
      ObjectSetInteger(0, objPrefix+"EntryLine", OBJPROP_STYLE, STYLE_DOT);
      ObjectSetInteger(0, objPrefix+"EntryLine", OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, objPrefix+"EntryLine", OBJPROP_BACK, true);
      
      ObjectCreate(0, objPrefix+"Runner", OBJ_TREND, 0, tStart, runner, tEnd, runner);
      ObjectSetInteger(0, objPrefix+"Runner", OBJPROP_COLOR, clrDodgerBlue);
      ObjectSetInteger(0, objPrefix+"Runner", OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, objPrefix+"Runner", OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, objPrefix+"Runner", OBJPROP_BACK, true);
   }

   g_lastSignalDir   = dir;
   g_lastSignalMode  = mode;
   g_lastSignalTime  = g_time[i];
   g_lastSignalPrice = entry;

   if(isLive)
   {
      bool cooldownActive = (TimeCurrent() < g_cooldownUntilBarTime);
      if(g_dailyHalt)
         Print("[GOD SCALP] Signal suppressed (arrow still drawn) - daily loss circuit breaker active.");
      else if(cooldownActive)
         Print("[GOD SCALP] Signal suppressed (arrow still drawn) - cooldown active after last loss.");
      else
      {
         string dirS  = (dir==1)?"LONG":"SHORT";
         string modeS = ModeStr(mode);
         AlertMsg(dirS+" entry signal ("+modeS+") @ "+DoubleToString(entry,_Digits)+
                  "  SL "+DoubleToString(sl,_Digits)+"  First target "+DoubleToString(t1,_Digits)+
                  "  R:R "+DoubleToString(rr,2));
      }
   }
   return true;
}

//+------------------------------------------------------------------+
//| Per-new-bar management of an already-tracked live trade          |
//+------------------------------------------------------------------+
void OnNewBarTradeUpdate(int i)
{
   if(!g_hasActiveTrade) return;

   // 1. Check for dangerous structural candles
   if(IsBadCandle(i, g_tradeDir))
   {
      AlertMsg("⚠️ WITHDRAW SIGNAL: Dangerous reversal candle detected! Consider closing your "+(g_tradeDir==1?"LONG":"SHORT")+" immediately.");
      string objName = "GS_Exit_" + IntegerToString(i);
      ObjectCreate(0, objName, OBJ_TEXT, 0, g_time[i], g_tradeDir==1 ? g_high[i] + g_atrFast[i]*0.5 : g_low[i] - g_atrFast[i]*0.5);
      ObjectSetString(0, objName, OBJPROP_TEXT, "⚠️ EXIT");
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrYellow);
      ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, 12);
      ObjectSetInteger(0, objName, OBJPROP_ANCHOR, g_tradeDir==1 ? ANCHOR_LOWER : ANCHOR_UPPER);
   }

   double atrF = g_atrFast[i];
   double hist = g_macdHist[i];
   bool against = (g_tradeDir==1 && hist<0) || (g_tradeDir==-1 && hist>0);
   g_tradeHistFlipStreak = against ? g_tradeHistFlipStreak+1 : 0;
   if(g_tradeHistFlipStreak >= InpMacdFlipConfirmBars)
   {
      AlertMsg("EXIT SIGNAL: MACD histogram flipped against your "+(g_tradeDir==1?"LONG":"SHORT")+
                " for "+IntegerToString(InpMacdFlipConfirmBars)+" bars - momentum may be dying. Consider closing.");
      string objNameM = "GS_Exit_MACD_" + IntegerToString(i);
      ObjectCreate(0, objNameM, OBJ_TEXT, 0, g_time[i], g_tradeDir==1 ? g_high[i] + g_atrFast[i]*0.5 : g_low[i] - g_atrFast[i]*0.5);
      ObjectSetString(0, objNameM, OBJPROP_TEXT, "⚠️ EXIT (MACD)");
      ObjectSetInteger(0, objNameM, OBJPROP_COLOR, clrOrange);
      ObjectSetInteger(0, objNameM, OBJPROP_FONTSIZE, 10);
      ObjectSetInteger(0, objNameM, OBJPROP_ANCHOR, g_tradeDir==1 ? ANCHOR_LOWER : ANCHOR_UPPER);
      g_tradeHistFlipStreak = 0;
   }

   if(g_tradeATRatEntry>0 && atrF > InpAtrSpikeExitMult*g_tradeATRatEntry)
   {
      bool badMove = (g_tradeDir==1 && g_close[i]<g_tradeEntryPrice) || (g_tradeDir==-1 && g_close[i]>g_tradeEntryPrice);
      if(badMove) {
         AlertMsg("EXIT SIGNAL: ATR spiked against your position ("+DoubleToString(atrF,_Digits)+
                   " vs "+DoubleToString(g_tradeATRatEntry,_Digits)+" at entry) - elevated reversal risk.");
         string objNameA = "GS_Exit_ATR_" + IntegerToString(i);
         ObjectCreate(0, objNameA, OBJ_TEXT, 0, g_time[i], g_tradeDir==1 ? g_high[i] + g_atrFast[i]*0.5 : g_low[i] - g_atrFast[i]*0.5);
         ObjectSetString(0, objNameA, OBJPROP_TEXT, "⚠️ EXIT (ATR)");
         ObjectSetInteger(0, objNameA, OBJPROP_COLOR, clrOrange);
         ObjectSetInteger(0, objNameA, OBJPROP_FONTSIZE, 10);
         ObjectSetInteger(0, objNameA, OBJPROP_ANCHOR, g_tradeDir==1 ? ANCHOR_LOWER : ANCHOR_UPPER);
      }
   }

   double rDist = MathAbs(g_tradeEntryPrice - g_tradeSL);
   if(rDist<=0) rDist=_Point;
   double favorable = (g_tradeDir==1)? (g_close[i]-g_tradeEntryPrice) : (g_tradeEntryPrice-g_close[i]);
   double rNow = favorable/rDist;

   if(!g_tradeBreakevenMoved && rNow>=InpBreakevenAtR)
   {
      g_tradeSL = BreakevenPrice(g_tradeDir, g_tradeEntryPrice);
      g_tradeBreakevenMoved = true;
      AlertMsg("Stop moved to cost-aware breakeven ("+DoubleToString(g_tradeSL,_Digits)+"). Remainder is now risk-free.");
   }
   if(rNow>=InpTrailStartR) g_tradeTrailingActive=true;
   if(g_tradeTrailingActive)
   {
      double trailDist = InpTrailAtrMult*atrF;
      if(g_tradeDir==1) g_tradeSL = MathMax(g_tradeSL, g_close[i]-trailDist);
      else              g_tradeSL = MathMin(g_tradeSL, g_close[i]+trailDist);
   }

   if(g_tradeMode==1 && !g_tradePartialTaken) g_tradeT1 = BufBBBasis[i];

   DrawTradeLines();
}

//+------------------------------------------------------------------+
//| Tick-based: detect a new/closed real position (read-only)         |
//+------------------------------------------------------------------+
void ManageActiveTrade()
{
   bool posExists = PositionSelect(_Symbol);
   if(posExists)
   {
      ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
      if(!g_hasActiveTrade || g_tradeTicket!=ticket)
         InitNewTrade(ticket);
   }
   else
   {
      if(g_hasActiveTrade)
      {
         bool wasLoss = DetermineLastTradeWasLoss(g_tradeTicket);
         if(wasLoss)
         {
            datetime baseT = (g_lastProcessedIdx>=0 && g_lastProcessedIdx<ArraySize(g_time)) ? g_time[g_lastProcessedIdx] : TimeCurrent();
            g_cooldownUntilBarTime = baseT + (datetime)(InpCooldownBars*PeriodSeconds(PERIOD_CURRENT));
            AlertMsg("Trade closed at a loss. Cooldown active for "+IntegerToString(InpCooldownBars)+" bars.");
         }
         else
            AlertMsg("Trade closed.");
         g_hasActiveTrade=false;
         ClearTradeLines();
      }
   }
   CheckDailyCircuitBreaker();
}

void InitNewTrade(ulong ticket)
{
   double posOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   long posType = PositionGetInteger(POSITION_TYPE);
   int dir = (posType==POSITION_TYPE_BUY)?1:-1;

   g_hasActiveTrade  = true;
   g_tradeTicket     = ticket;
   g_tradeDir        = dir;
   g_tradeEntryPrice = posOpenPrice;

   int refIdx = (g_lastProcessedIdx>=0)? g_lastProcessedIdx : 0;
   datetime nowBarT = (refIdx<ArraySize(g_time))? g_time[refIdx] : TimeCurrent();

   if(g_lastSignalDir==dir && g_lastSignalTime>0 && (nowBarT-g_lastSignalTime)<=(datetime)(10*PeriodSeconds(PERIOD_CURRENT)))
      g_tradeMode = g_lastSignalMode;
   else
      g_tradeMode = -1;

   double atrF = (refIdx<ArraySize(g_atrFast))? g_atrFast[refIdx] : 0;
   g_tradeATRatEntry = atrF;

   if(g_tradeMode==0)
   {
      double structExt = RecentExtreme(refIdx, dir, InpTrendStructLookback);
      double oppBand = (dir==1)? BufBBUpper[refIdx] : BufBBLower[refIdx];
      ComputeTrendStopTarget(dir, posOpenPrice, structExt, oppBand, atrF, g_tradeSL, g_tradeT1);
   }
   else if(g_tradeMode==1)
   {
      double wickExt = RecentExtreme(refIdx, dir, InpMRWickLookback);
      g_tradeSL = ComputeMRStop(dir, posOpenPrice, wickExt, atrF);
      g_tradeT1 = (refIdx<ArraySize(BufBBBasis))? BufBBBasis[refIdx] : posOpenPrice;
   }
   else
   {
      // Position doesn't match a recent signal (manual entry) - generic ATR fallback
      g_tradeSL = posOpenPrice - dir*1.0*atrF;
      g_tradeT1 = posOpenPrice + dir*1.5*atrF;
   }

   g_tradePartialTaken   = false;
   g_tradeBreakevenMoved = false;
   g_tradeTrailingActive = false;
   g_tradeHistFlipStreak = 0;
   g_tradeRunner = g_tradeT1;

   DrawTradeLines();
   AlertMsg("Position detected - now tracking exits. Mode="+ModeStr(g_tradeMode)+
             "  SL="+DoubleToString(g_tradeSL,_Digits)+"  First target="+DoubleToString(g_tradeT1,_Digits));
}

bool DetermineLastTradeWasLoss(ulong posTicket)
{
   if(posTicket==0) return false;
   if(!HistorySelect(TimeCurrent()-86400*3, TimeCurrent())) return false;
   double totalProfit=0; bool found=false;
   int total = HistoryDealsTotal();
   for(int i=total-1;i>=0;i--)
   {
      ulong dealTicket = HistoryDealGetTicket(i);
      if(dealTicket==0) continue;
      ulong posId = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
      if(posId == posTicket)
      {
         totalProfit += HistoryDealGetDouble(dealTicket, DEAL_PROFIT)
                       + HistoryDealGetDouble(dealTicket, DEAL_SWAP)
                       + HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
         found=true;
      }
   }
   if(!found) return false;
   return (totalProfit<0);
}

void CheckDailyCircuitBreaker()
{
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   dt.hour=0; dt.min=0; dt.sec=0;
   datetime dayStart = StructToTime(dt);
   if(dayStart != g_currentDay)
   {
      g_currentDay = dayStart;
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      g_dailyHalt = false;
   }
   if(!g_dailyHalt && g_dayStartEquity>0)
   {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      double ddPct = 100.0*(eq-g_dayStartEquity)/g_dayStartEquity;
      if(ddPct <= -InpDailyLossLimitPct)
      {
         g_dailyHalt = true;
         AlertMsg("Daily loss circuit breaker tripped ("+DoubleToString(ddPct,2)+"%). No new entry alerts for the rest of today.");
      }
   }
}

//+------------------------------------------------------------------+
//| Tick-based: price crossing SL / first-target / runner-target      |
//+------------------------------------------------------------------+
void CheckPriceExitTriggers()
{
   if(!g_hasActiveTrade) return;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double cur = (g_tradeDir==1)? bid : ask;

   bool slHit = (g_tradeDir==1)? (cur<=g_tradeSL) : (cur>=g_tradeSL);
   if(slHit)
   {
      AlertMsg("EXIT SIGNAL: STOP LOSS level reached ("+DoubleToString(g_tradeSL,_Digits)+"). Close the position.");
      return;
   }

   if(!g_tradePartialTaken)
   {
      bool t1Hit = (g_tradeDir==1)? (cur>=g_tradeT1) : (cur<=g_tradeT1);
      if(t1Hit)
      {
         g_tradePartialTaken = true;
         g_tradeSL = BreakevenPrice(g_tradeDir, g_tradeEntryPrice);
         g_tradeBreakevenMoved = true;

         if(g_tradeMode==0)
         {
            g_tradeRunner = (g_tradeDir==1)? g_tradeT1 + InpRunnerTrendAtrMult*g_tradeATRatEntry
                                            : g_tradeT1 - InpRunnerTrendAtrMult*g_tradeATRatEntry;
         }
         else
         {
            int lastIdx = ArraySize(BufBBUpper)-1;
            double opp = (g_tradeDir==1)? BufBBUpper[lastIdx] : BufBBLower[lastIdx];
            if(g_tradeDir==1) g_tradeRunner = g_tradeEntryPrice + InpRunnerMRBandFrac*(opp-g_tradeEntryPrice);
            else              g_tradeRunner = g_tradeEntryPrice - InpRunnerMRBandFrac*(g_tradeEntryPrice-opp);
         }

         AlertMsg("FIRST TARGET HIT @ "+DoubleToString(g_tradeT1,_Digits)+" - close "+
                   DoubleToString(InpPartialFraction*100,0)+"% here, stop now at cost-aware breakeven, let the rest run to "+
                   DoubleToString(g_tradeRunner,_Digits)+".");
         DrawTradeLines();
      }
   }
   else
   {
      bool runHit = (g_tradeDir==1)? (cur>=g_tradeRunner) : (cur<=g_tradeRunner);
      if(runHit)
         AlertMsg("RUNNER TARGET HIT @ "+DoubleToString(g_tradeRunner,_Digits)+" - close the remainder.");
   }
}

//+------------------------------------------------------------------+
//| Chart object helpers                                             |
//+------------------------------------------------------------------+
void SetHLine(string name,double price,color c,ENUM_LINE_STYLE style)
{
   if(ObjectFind(0,name)<0)
   {
      ObjectCreate(0,name,OBJ_HLINE,0,0,price);
      ObjectSetInteger(0,name,OBJPROP_COLOR,c);
      ObjectSetInteger(0,name,OBJPROP_STYLE,style);
      ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
      ObjectSetString(0,name,OBJPROP_TEXT,name);
   }
   else
      ObjectSetDouble(0,name,OBJPROP_PRICE,price);
}

void DrawTradeLines()
{
   if(!g_hasActiveTrade) return;
   
   datetime tStart = g_lastSignalTime;
   if(tStart == 0) tStart = TimeCurrent();
   datetime tEnd = TimeCurrent() + PeriodSeconds(PERIOD_CURRENT)*5; 

   // Entry Line
   if(ObjectFind(0, "GS_Entry") < 0) {
      ObjectCreate(0, "GS_Entry", OBJ_TREND, 0, tStart, g_tradeEntryPrice, tEnd, g_tradeEntryPrice);
      ObjectSetInteger(0, "GS_Entry", OBJPROP_COLOR, clrSilver);
      ObjectSetInteger(0, "GS_Entry", OBJPROP_STYLE, STYLE_DOT);
      ObjectSetInteger(0, "GS_Entry", OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, "GS_Entry", OBJPROP_RAY_RIGHT, true);
      ObjectSetInteger(0, "GS_Entry", OBJPROP_BACK, true);
   } else {
      ObjectSetInteger(0, "GS_Entry", OBJPROP_TIME, 0, tStart);
      ObjectSetInteger(0, "GS_Entry", OBJPROP_TIME, 1, tEnd);
      ObjectSetDouble(0, "GS_Entry", OBJPROP_PRICE, 0, g_tradeEntryPrice);
      ObjectSetDouble(0, "GS_Entry", OBJPROP_PRICE, 1, g_tradeEntryPrice);
   }

   // SL Box
   if(ObjectFind(0, "GS_SLBoxLive") < 0) {
      ObjectCreate(0, "GS_SLBoxLive", OBJ_RECTANGLE, 0, tStart, g_tradeEntryPrice, tEnd, g_tradeSL);
      ObjectSetInteger(0, "GS_SLBoxLive", OBJPROP_COLOR, clrMaroon);
      ObjectSetInteger(0, "GS_SLBoxLive", OBJPROP_FILL, true);
      ObjectSetInteger(0, "GS_SLBoxLive", OBJPROP_BACK, true);
   } else {
      ObjectSetInteger(0, "GS_SLBoxLive", OBJPROP_TIME, 0, tStart);
      ObjectSetInteger(0, "GS_SLBoxLive", OBJPROP_TIME, 1, tEnd);
      ObjectSetDouble(0, "GS_SLBoxLive", OBJPROP_PRICE, 0, g_tradeEntryPrice);
      ObjectSetDouble(0, "GS_SLBoxLive", OBJPROP_PRICE, 1, g_tradeSL);
   }

   // TP Box
   if(!g_tradePartialTaken)
   {
      if(ObjectFind(0, "GS_TPBoxLive") < 0) {
         ObjectCreate(0, "GS_TPBoxLive", OBJ_RECTANGLE, 0, tStart, g_tradeEntryPrice, tEnd, g_tradeT1);
         ObjectSetInteger(0, "GS_TPBoxLive", OBJPROP_COLOR, clrDarkGreen);
         ObjectSetInteger(0, "GS_TPBoxLive", OBJPROP_FILL, true);
         ObjectSetInteger(0, "GS_TPBoxLive", OBJPROP_BACK, true);
      } else {
         ObjectSetInteger(0, "GS_TPBoxLive", OBJPROP_TIME, 0, tStart);
         ObjectSetInteger(0, "GS_TPBoxLive", OBJPROP_TIME, 1, tEnd);
         ObjectSetDouble(0, "GS_TPBoxLive", OBJPROP_PRICE, 0, g_tradeEntryPrice);
         ObjectSetDouble(0, "GS_TPBoxLive", OBJPROP_PRICE, 1, g_tradeT1);
      }
      ObjectDelete(0,"GS_Runner");
   }
   else
   {
      ObjectDelete(0,"GS_TPBoxLive");
      if(ObjectFind(0, "GS_Runner") < 0) {
         ObjectCreate(0, "GS_Runner", OBJ_TREND, 0, tStart, g_tradeRunner, tEnd, g_tradeRunner);
         ObjectSetInteger(0, "GS_Runner", OBJPROP_COLOR, clrDodgerBlue);
         ObjectSetInteger(0, "GS_Runner", OBJPROP_STYLE, STYLE_DASHDOT);
         ObjectSetInteger(0, "GS_Runner", OBJPROP_WIDTH, 2);
         ObjectSetInteger(0, "GS_Runner", OBJPROP_RAY_RIGHT, true);
         ObjectSetInteger(0, "GS_Runner", OBJPROP_BACK, true);
      } else {
         ObjectSetInteger(0, "GS_Runner", OBJPROP_TIME, 0, tStart);
         ObjectSetInteger(0, "GS_Runner", OBJPROP_TIME, 1, tEnd);
         ObjectSetDouble(0, "GS_Runner", OBJPROP_PRICE, 0, g_tradeRunner);
         ObjectSetDouble(0, "GS_Runner", OBJPROP_PRICE, 1, g_tradeRunner);
      }
   }
}

void ClearTradeLines()
{
   ObjectDelete(0,"GS_Entry");
   ObjectDelete(0,"GS_SLBoxLive");
   ObjectDelete(0,"GS_TPBoxLive");
   ObjectDelete(0,"GS_Runner");
}

void UpdateStatusComment()
{
   string s = "GOD SCALP v3 Signals\n";
   int lastIdx = ArraySize(g_regime)-1;
   if(lastIdx>=0)
      s += "Regime: "+RegimeStr(g_regime[lastIdx])+"\n";

   if(g_hasActiveTrade)
   {
      s += "ACTIVE TRADE: "+(g_tradeDir==1?"LONG":"SHORT")+" ("+ModeStr(g_tradeMode)+")\n";
      s += "Entry "+DoubleToString(g_tradeEntryPrice,_Digits)+"   SL "+DoubleToString(g_tradeSL,_Digits)+"\n";
      s += g_tradePartialTaken ? ("Runner target "+DoubleToString(g_tradeRunner,_Digits)+" (partial already banked)")
                                : ("First target "+DoubleToString(g_tradeT1,_Digits));
   }
   else if(g_dailyHalt)
      s += "DAILY CIRCUIT BREAKER ACTIVE - no new entry alerts today.";
   else if(TimeCurrent()<g_cooldownUntilBarTime)
      s += "COOLDOWN active after last loss.";
   else if(g_pendActive)
      s += "Setup ARMED ("+ModeStr(g_pendMode)+" "+(g_pendDir==1?"long":"short")+") - waiting on MACD trigger, "+IntegerToString(g_pendBarsLeft)+" bars left.";
   else
      s += "Flat. Watching for a setup.";

   double expiryRate = (g_cntArmed>0)? 100.0*g_cntExpired/g_cntArmed : 0;
   s += "\n\n-- signal stats (this symbol, since first attached) --\n";
   s += "Armed:"+IntegerToString(g_cntArmed)+"  Fired:"+IntegerToString(g_cntFired)+
        "  Expired:"+IntegerToString(g_cntExpired)+" ("+DoubleToString(expiryRate,0)+"%)"+
        "  RR-rejected:"+IntegerToString(g_cntRejectedRR);

   Comment(s);
}

//+------------------------------------------------------------------+
//| OnCalculate                                                      |
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
   int total = rates_total;
   int warmupBars = MathMax(MathMax(InpBBLen, InpChopLen), InpMacdSlow) + 5;
   if(total < warmupBars+5) return 0;

   //--- keep local copies of price series for use by all helper functions
   ArrayResize(g_time,  total);
   ArrayResize(g_open,  total);
   ArrayResize(g_high,  total);
   ArrayResize(g_low,   total);
   ArrayResize(g_close, total);
   ArrayCopy(g_time,  time);
   ArrayCopy(g_open,  open);
   ArrayCopy(g_high,  high);
   ArrayCopy(g_low,   low);
   ArrayCopy(g_close, close);

   ArrayResize(g_atrFast,total);
   ArrayResize(g_emaFast,total);  ArrayResize(g_emaSlow,total);
   ArrayResize(g_macdLine,total); ArrayResize(g_macdSignal,total); ArrayResize(g_macdHist,total);
   ArrayResize(g_chop,total);     ArrayResize(g_regime,total);     ArrayResize(g_trendBias,total);


   int start = (prev_calculated>1) ? prev_calculated-1 : 0;

   for(int i=start;i<total;i++)
   {
      BufArrowUp[i]   = EMPTY_VALUE;
      BufArrowDown[i] = EMPTY_VALUE;
      BufSL[i]        = EMPTY_VALUE;
      BufTP[i]        = EMPTY_VALUE;
      BufRunner[i]    = EMPTY_VALUE;

      UpdateATR(i, InpAtrFast, g_atrFast);
      UpdateBB(i);
      UpdateEMA(i, InpMacdFast, g_emaFast, g_close);
      UpdateEMA(i, InpMacdSlow, g_emaSlow, g_close);
      g_macdLine[i] = g_emaFast[i]-g_emaSlow[i];
      UpdateEMA(i, InpMacdSignal, g_macdSignal, g_macdLine);
      g_macdHist[i] = g_macdLine[i]-g_macdSignal[i];
      UpdateCHOP(i);
      UpdateRegime(i);
      BufAlma5[i] = CalcALMA(i, InpAlmaLen, InpAlmaOffset, InpAlmaSigma, g_close);
      BufAlma50[i] = CalcALMA(i, InpAlmaSlowLen, InpAlmaOffset, InpAlmaSigma, g_close);
      UpdateTrendBias(i);
   }

   //--- Signal state machine: process each newly closed 5m bar exactly once
   int lastClosedIdx = total-2;
   if(lastClosedIdx >= warmupBars)
   {
      if(prev_calculated==0)
      {
         g_pendActive = false;
         g_lastProcessedIdx = -1;
         for(int k=warmupBars;k<=lastClosedIdx;k++)
            ProcessClosedBar(k, false);
         g_lastProcessedIdx = lastClosedIdx;
         if(g_hasActiveTrade) OnNewBarTradeUpdate(lastClosedIdx);
      }
      else if(lastClosedIdx > g_lastProcessedIdx)
      {
         for(int k=g_lastProcessedIdx+1; k<=lastClosedIdx; k++)
         {
            bool live = (k==lastClosedIdx);
            ProcessClosedBar(k, live);
            if(live) OnNewBarTradeUpdate(k);
         }
         g_lastProcessedIdx = lastClosedIdx;
      }
   }

   //--- Live, tick-based trade tracking (independent of bar close)
   ManageActiveTrade();
   CheckPriceExitTriggers();
   UpdateStatusComment();

   return total;
}
//+------------------------------------------------------------------+
