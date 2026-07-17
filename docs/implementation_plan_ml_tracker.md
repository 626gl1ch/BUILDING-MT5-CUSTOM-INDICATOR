# Implementation Plan: Strategy Learning Database & ML Meta-Filter

## Goal
To prevent the engine from wasting computational resources on useless strategies by building a persistent "Strategy Learning Database." This system will log the precise parameters and synthetic performance of every strategy tested. By storing this offline, the system builds an enormous dataset of what works and what fails in randomized synthetic markets, which can be used to train an ML model to pre-filter garbage strategies instantly.

## Proposed Changes

### 1. The Strategy Knowledge Base (`knowledge_base.py` [NEW])
I will create a new dedicated script to handle the persistent storage of strategy outcomes. 
- It will maintain a master CSV file: `strategy_learning_db.csv`.
- Every time a strategy is evaluated against the Synthetic Data, its exact DNA (Entry Logic, Regime Filter, Risk %, Stop Loss, Take Profit) and its exact performance metrics (Sharpe Ratio, Expectancy, Win Rate, Permutation P-Value) will be appended to this database.
- Even if a strategy fails, it is recorded. This is critical for Machine Learning, as an ML model needs both negative and positive examples to learn the boundary of profitability.

### 2. ML Meta-Filter Integration (`mass_search_sequential.py` [MODIFY])
I will update the main search loop to integrate this knowledge base.
- **Pre-Filtering:** Before running the expensive backtest on a combination, the bot will check the `strategy_learning_db.csv`. If this exact combination was previously tested and failed the synthetic data test, the bot will instantly skip it, saving massive amounts of time.
- **Data Logging:** After the Synthetic Test stage, the bot will call the `knowledge_base.py` module to log the results. If it passes the synthetic test, it continues to the multi-timeframe gauntlet as normal. 

## User Review Required
> [!CAUTION]
> **Data Size Consideration:** With 45,000 strategies, the `strategy_learning_db.csv` will grow to roughly 5-10 MB. This is perfectly fine and highly optimized for Pandas/Scikit-Learn to read offline. 
> 
> **Are you okay with this architecture?** We are essentially building the ultimate training dataset for your future ML Meta-Labeler. Every run of the bot will make it "smarter" by permanently recording what parameter combinations survive statistical noise.
