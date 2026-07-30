import os
import asyncio
import logging
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from pybit.unified_trading import HTTP

# Internal imports for MT5 indicators
import indicators_library as ind

# --- Configuration ---
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Initialize Pybit Client
session = HTTP(
    testnet=True, # Default to testnet
    api_key=API_KEY,
    api_secret=API_SECRET,
)

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# State management
managed_positions = {}
active_strategy = "TF" # Default
alert_chat_id = None # Captured from /start

# --- Telegram Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alert_chat_id
    alert_chat_id = update.effective_chat.id
    await update.message.reply_text(
        "Welcome to the Bybit Position Manager!\n\n"
        "I will automatically monitor your open Bybit positions and manage them according to your selected MT5 strategy.\n\n"
        "Commands:\n"
        "/status - View current positions\n"
        "/set_strategy <MR|TF> - Switch between Mean Reversion and Trend Following\n"
        "/exit_now - Market close all positions immediately"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        positions = session.get_positions(category="linear", settleCoin="USDT")['result']['list']
        open_pos = [p for p in positions if float(p['size']) > 0]
        
        if not open_pos:
            await update.message.reply_text("No open positions found on Bybit.")
            return

        msg = "📊 **Current Open Positions:**\n\n"
        for p in open_pos:
            msg += f"Symbol: {p['symbol']}\nSide: {p['side']}\nSize: {p['size']}\nEntry: {p['avgPrice']}\nUnrealized PNL: {p['unrealisedPnl']}\n\n"
            
        msg += f"Active Strategy: {active_strategy}"
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"Error fetching status: {str(e)}")

async def set_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global active_strategy
    if not context.args:
        await update.message.reply_text("Please provide a strategy. Example: /set_strategy MR")
        return
        
    strat = context.args[0].upper()
    if strat not in ['MR', 'TF']:
        await update.message.reply_text("Invalid strategy. Choose 'MR' (Mean Reversion) or 'TF' (Trend Following).")
        return
        
    active_strategy = strat
    await update.message.reply_text(f"Strategy updated to: {active_strategy}")

async def exit_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        positions = session.get_positions(category="linear", settleCoin="USDT")['result']['list']
        open_pos = [p for p in positions if float(p['size']) > 0]
        
        if not open_pos:
            await update.message.reply_text("No open positions to exit.")
            return
            
        for p in open_pos:
            side = "Sell" if p['side'] == "Buy" else "Buy"
            session.place_order(
                category="linear",
                symbol=p['symbol'],
                side=side,
                orderType="Market",
                qty=p['size'],
                reduceOnly=True
            )
        await update.message.reply_text("✅ All open positions have been market closed.")
        managed_positions.clear()
    except Exception as e:
        await update.message.reply_text(f"Error closing positions: {str(e)}")

# --- Background Task: Position Monitoring & Logic ---

async def fetch_klines(symbol, interval="5", limit=200):
    res = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)['result']['list']
    # Bybit returns: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(res, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

async def monitor_positions(context: ContextTypes.DEFAULT_TYPE):
    """Periodically check positions and apply MT5 strategy exit/TP rules."""
    if not alert_chat_id:
        return
        
    try:
        positions = session.get_positions(category="linear", settleCoin="USDT")['result']['list']
        open_pos = [p for p in positions if float(p['size']) > 0]
        
        current_symbols = []
        
        for p in open_pos:
            symbol = p['symbol']
            side = p['side']
            size = float(p['size'])
            entry = float(p['avgPrice'])
            current_symbols.append(symbol)
            
            # If position is newly discovered, alert Telegram
            if symbol not in managed_positions:
                managed_positions[symbol] = {'entry': entry, 'side': side, 'size': size}
                await context.bot.send_message(
                    chat_id=alert_chat_id,
                    text=f"🚀 Detected new position on Bybit: {side} {size} {symbol} @ {entry}. Managing via {active_strategy} strategy."
                )
            
            # Fetch 5m data
            df = await fetch_klines(symbol, interval="5", limit=200)
            
            # Use MT5 Indicators Library
            atr = ind.calc_atr(df, 14).iloc[-2]
            
            # Dynamic TP/SL Logic based on Active Strategy
            if active_strategy == "TF":
                # Trend Following: Wide Trailing (e.g. 3 ATR)
                # Setting actual TP on Bybit
                if side == "Buy":
                    sl_price = entry - (3 * atr)
                    tp_price = entry + (5 * atr)
                else:
                    sl_price = entry + (3 * atr)
                    tp_price = entry - (5 * atr)
                    
                session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    takeProfit=str(round(tp_price, 2)),
                    stopLoss=str(round(sl_price, 2)),
                    tpslMode="Full",
                    positionIdx=0
                )
                
            elif active_strategy == "MR":
                # Mean Reversion: Tight TP at Bollinger Mid
                upper, mid, lower, _, _ = ind.calc_bollinger_bands(df['close'], 20, 2.0)
                
                if side == "Buy":
                    tp_price = mid.iloc[-2]
                    sl_price = entry - (2 * atr)
                else:
                    tp_price = mid.iloc[-2]
                    sl_price = entry + (2 * atr)
                    
                session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    takeProfit=str(round(tp_price, 2)),
                    stopLoss=str(round(sl_price, 2)),
                    tpslMode="Full",
                    positionIdx=0
                )
                
        # Clean up closed positions
        closed_symbols = [s for s in managed_positions.keys() if s not in current_symbols]
        for s in closed_symbols:
            del managed_positions[s]
            await context.bot.send_message(
                chat_id=alert_chat_id,
                text=f"🏁 Position closed for {s}. No longer managing."
            )
                
    except Exception as e:
        logger.error(f"Error in monitor_positions: {e}")

def main():
    if not TELEGRAM_TOKEN or not API_KEY or not API_SECRET:
        print("Please set BYBIT_API_KEY, BYBIT_API_SECRET, and TELEGRAM_BOT_TOKEN environment variables.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("set_strategy", set_strategy))
    app.add_handler(CommandHandler("exit_now", exit_now))

    # Start the background job every 60 seconds
    job_queue = app.job_queue
    job_queue.run_repeating(monitor_positions, interval=60, first=10)
    
    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
