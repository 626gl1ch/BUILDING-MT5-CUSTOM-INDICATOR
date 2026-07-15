"""
Launcher script to run both Mean Reversion and Trend Following search bots simultaneously.
Can be run with --test flag for a quick validation run.
"""
import subprocess
import sys
import time

def main():
    print("=" * 80)
    print("LAUNCHING BOTH MEAN REVERSION & TREND FOLLOWING SEARCH BOTS")
    print("=" * 80)
    
    # Forward any arguments (like --test) to the search bots
    args = sys.argv[1:]
    
    # Launch Mean Reversion Bot
    mr_cmd = [sys.executable, 'mr_strategy_bot.py'] + args
    print(f"Starting Mean Reversion Search Bot: {' '.join(mr_cmd)}")
    mr_proc = subprocess.Popen(mr_cmd)
    
    # Launch Trend Following Bot
    tf_cmd = [sys.executable, 'tf_strategy_bot.py'] + args
    print(f"Starting Trend Following Search Bot: {' '.join(tf_cmd)}")
    tf_proc = subprocess.Popen(tf_cmd)
    
    print("\nBoth search bots are now running in parallel. Press Ctrl+C to terminate both.")
    
    try:
        mr_running = True
        tf_running = True
        while mr_running or tf_running:
            if mr_running:
                mr_status = mr_proc.poll()
                if mr_status is not None:
                    print(f"\nMean Reversion search bot exited with code {mr_status}")
                    mr_running = False
            
            if tf_running:
                tf_status = tf_proc.poll()
                if tf_status is not None:
                    print(f"\nTrend Following search bot exited with code {tf_status}")
                    tf_running = False
                    
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTermination signal received. Terminating both search processes...")
        mr_proc.terminate()
        tf_proc.terminate()
        mr_proc.wait()
        tf_proc.wait()
        print("Both processes terminated cleanly.")

if __name__ == '__main__':
    main()
