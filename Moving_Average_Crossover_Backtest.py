# trading_strategy_backtest.py

import yfinance as yf          # To download historical stock data
import pandas as pd            # For data manipulation (tables, calculations)
import numpy as np             # For numerical calculations (mean, std, etc.)
import matplotlib.pyplot as plt # For plotting charts
import os as os

def download_data(ticker, start, end):
    """Download historical data from Yahoo Finance""" #called docstring, like comment at beginning of function but doesn't affect code
# 3 quotations means you can write multi-line strings
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

def generate_signals(data, short_window=50, long_window=200):
    """Generate buy/sell signals using moving average crossover"""
    data["SMA50"] = data["Close"].rolling(short_window).mean()
#this adds a new column to the excel called SMA50
    data["SMA200"] = data["Close"].rolling(long_window).mean()
#the rolling function is used in conjunction with the mean function to take the mean of the last x values

    data["Signal"] = 0
    data.loc[data["SMA50"] > data["SMA200"], "Signal"] = 1   # long
    data.loc[data["SMA50"] < data["SMA200"], "Signal"] = -1  # short
# .loc makes it look at each row and compares the column values, then adds value to empty Signal column (comes from pandas package)
    return data

def backtest_strategy(data):
    """Run backtest and calculate strategy performance"""
    data["Return"] = data["Close"].pct_change()
#looks at % change between current and previous value in column, then adds % change to Return column (comes from numpy package)
    data["Strategy"] = data["Signal"].shift(1) * data["Return"]
#since you don't know the strategy until the end of the day, whether to go long or short, as you don't have closing prices until end of trading_strategy_backtest
#the shift goes down 1 row in the column and then multiplies it by the return column
    data = data.dropna(subset=["SMA200"]).copy()
#this drops the observations until the average 200 kicks in
    cumulative_strategy = (1 + data["Strategy"]).cumprod()
    cumulative_benchmark = (1 + data["Return"]).cumprod()
#cumprod() multiplies each cell in column by each other

    sharpe_ratio = np.sqrt(252) * data["Strategy"].mean() / data["Strategy"].std()
    return cumulative_strategy, cumulative_benchmark, sharpe_ratio, data

def plot_results(cumulative_strategy, cumulative_benchmark):
    """Plot strategy performance vs Buy & Hold benchmark"""
    plt.figure(figsize=(10, 6))  # Set chart size
    plt.plot(cumulative_benchmark, label="Buy & Hold")  # Plot benchmark
    plt.plot(cumulative_strategy, label="Strategy")     # Plot strategy
    plt.title("Trading Strategy vs Buy & Hold")         # Chart title
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Growth of $1)")
    plt.legend()                                        # Show labels
    plt.show()                                          # Display chart

# Parameters
# Ask for user inputs
ticker = input("Enter ticker (e.g. AAPL, MSFT, TSLA): ").upper()
start = input("Enter start date (YYYY-MM-DD): ")
end = input("Enter end date (YYYY-MM-DD): ")

# Run pipeline
data = download_data(ticker, start, end)             # Step 1: Get data

data = generate_signals(data)                        # Step 2: Generate buy/sell signals
data.to_csv("historical data.csv")
print("\nColumns in data before backtest:", data.columns.tolist())
cumulative_strategy, cumulative_benchmark, sharpe_ratio, data_valid = backtest_strategy(data)
#do this as function returns tuple, so your just directly assigning variables instead of having to reference tuple going forward
# Print results
print(f"\nTicker: {ticker}")
#the f ensures that the variable ticker is picked up and put in the sentence as a string
print(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
#the 2f turns the sharpe ratio into a floating point number with 2 decimals
print("Last few rows of data with signals:")
print(data.tail(10))

# Plot results
plot_results(cumulative_strategy, cumulative_benchmark)
