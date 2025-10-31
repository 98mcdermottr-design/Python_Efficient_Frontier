import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px

# Parameters

x = "YES"
tickers = []
data = []
while x == "YES":
    ticker = input("Enter ticker (e.g. AAPL, MSFT, TSLA): ").strip().upper()
    tickers.append(ticker)
    x = input("Do you have more stocks you want added to the portfolio: YES or NO: ").strip().upper()
start = input("Enter start date (YYYY-MM-DD): ")
end = input("Enter end date (YYYY-MM-DD): ")


# Data

data = yf.download(tickers, start=start, end=end)['Close']

# Remove tickers that failed to download (all-NaN columns)
failed_tickers = [ticker for ticker in tickers if data[ticker].isna().all()]
if failed_tickers:
    print("Warning: Below tickers had no data and were removed:")
    print(failed_tickers)
    for ticker in failed_tickers:
        tickers.remove(ticker)
    data = data.drop(columns=failed_tickers)

# If all tickers failed, stop the script
if len(tickers) == 0:
    raise ValueError("No valid tickers left after removing failed downloads. Please check your inputs.")

returns = data.pct_change().dropna()
mean_returns = returns.mean().values  # 1D array
cov_matrix = returns.cov().values     # 2D matrix



# Portfolio Simulation

num_portfolios = 20000
#empty vector to store all of the results, 3 rows return, volatility and sharpe ratio
results = np.zeros((3, num_portfolios))
#what the weights will be in each stock for each sim
weights_record = []

correlation_matrix = returns.corr().values  # Get correlation matrix

penalty_strength = 0.5  # Adjust this value to control how strongly correlation is penalized

for i in range(num_portfolios):
    #creates random weights for each sim portfolio
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)

    # Annualized portfolio return and volatility
    portfolio_return = np.sum(weights * mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    # Calculate weighted average pairwise correlation
    weighted_corr = 0
    for j in range(len(tickers)):
        for k in range(len(tickers)):
            if j != k:
                weighted_corr += weights[j] * weights[k] * correlation_matrix[j][k]
    # Normalize by number of pairs
    num_pairs = len(tickers) * (len(tickers) - 1)
    avg_weighted_corr = weighted_corr / num_pairs

    # Penalized Sharpe Ratio
    sharpe_ratio = portfolio_return / portfolio_volatility - penalty_strength * avg_weighted_corr

    results[0, i] = portfolio_volatility
    results[1, i] = portfolio_return
    results[2, i] = sharpe_ratio


# Max Sharpe & Min Volatility Portfolios

max_sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[0])

max_sharpe_portfolio = results[:, max_sharpe_idx]
min_vol_portfolio = results[:, min_vol_idx]


# Plot Efficient Frontier

results_df = pd.DataFrame({
    'Volatility': results[0, :],
    'Return': results[1, :],
    'Sharpe Ratio': results[2, :],
})
weights_df = pd.DataFrame(weights_record, columns=tickers)
results_df = pd.concat([results_df, weights_df], axis=1)

fig = px.scatter(
    results_df,
    x='Volatility',
    y='Return',
    color='Sharpe Ratio',
    color_continuous_scale='viridis',
    hover_data=tickers,
    title='Interactive Efficient Frontier'
)

fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.update_layout(width=900, height=600)
fig.write_html("efficient_frontier.html", auto_open=True)


# Portfolio Details

print("\nMaximum Sharpe Ratio Portfolio Allocation\n")
print("Annualised Return:", round(max_sharpe_portfolio[1], 2))
print("Annualised Volatility:", round(max_sharpe_portfolio[0], 2))
print("\nWeights:")
for ticker, weight in zip(tickers, weights_record[max_sharpe_idx]):
    print(f"{ticker}: {weight:.2%}")

print("\nMinimum Volatility Portfolio Allocation\n")
print("Annualised Return:", round(min_vol_portfolio[1], 2))
print("Annualised Volatility:", round(min_vol_portfolio[0], 2))
print("\nWeights:")
for ticker, weight in zip(tickers, weights_record[min_vol_idx]):
    print(f"{ticker}: {weight:.2%}")
