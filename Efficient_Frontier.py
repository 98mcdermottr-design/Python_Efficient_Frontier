import numpy as np
# for numerical computations
import pandas as pd
# for tables
import yfinance as yf
# gets data from yahoo finance
import plotly.express as px
# for interactive charts

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
# the above is a loop, where you enter in tickers and it keeps asking you if you want to add more tickers to
# the list of stocks your considering in your portfolio. Once you say no (or anything outside of yes for that matter)
# the code then moves on

# Data

data = yf.download(tickers, start=start, end=end)['Close']
# just downloads closing prices for each of the tickers specified from yahoo finance

failed_tickers = [ticker for ticker in tickers if data[ticker].isna().all()]
# a variable created from a loop, looks for the tickers in the tickers list that came up as NA after the yahoo finance download
if failed_tickers:
    print("Warning: Below tickers had no data and were removed:")
    print(failed_tickers)
    for ticker in failed_tickers:
        tickers.remove(ticker)
    # removes NA tickers from tickers list
    data = data.drop(columns=failed_tickers)
    # then removes the failed tickers from the data set
# the idea behind the above is that sometimes the tickers given in the above loop are not valid, and it messes up the below code if there are invalid tickers


if len(tickers) == 0:
    raise ValueError("No valid tickers left after removing failed downloads. Please check your inputs.")
# checks if any tickers left after removing failed tickers, if there are none then just stops the code

returns = data.pct_change().dropna()
# returns a dataframe with the percentage change for each ticker, day on day
mean_returns = returns.mean().values
# calculates the mean of the daily returns for each ticker
# .values turns it into a 1D Numpy array rather than a panda series which is what the returns dataframe is
cov_matrix = returns.cov().values
# then calculates the covariance matrix (nxn matrix, where n is the number of tickers)



# Portfolio Simulation

num_portfolios = 20000
# you're going to simulate 20,000 portfolios with random weights on each of the tickers
results = np.zeros((3, num_portfolios))
# creates blank array with 3 rows and the number of portfolios of columns
weights_record = []
# blank list to be populated with what the weights will be in each stock for each sim

correlation_matrix = returns.corr().values
# calculates the correlation matrix, and enters into a numoy array
# put it into a numpy array to ensure easier mathematical calculations going forward

penalty_strength = 0.5
# adjust this value to control how strongly correlation is penalized

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)
    # loop that creates random weights for each sim portfolio and then adds these random weights to the blank list created earlier

    portfolio_return = np.sum(weights * mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    # calculates the annualized portfolio return and volatility for the simulated portfolio currently being run in the loop

    weighted_corr = 0
    for j in range(len(tickers)):
        for k in range(len(tickers)):
            if j != k:
                weighted_corr += weights[j] * weights[k] * correlation_matrix[j][k]
    # loop that goes through each of the tickers and applies the weights on the current simulated portfolio to
    # the correlation matrix and then multiplies them and adds them to the weighted-corr variavke
    num_pairs = len(tickers) * (len(tickers) - 1)
    avg_weighted_corr = weighted_corr / num_pairs
    # finds the average weighted correlation among the tickers in the simulated portfolio

    sharpe_ratio = portfolio_return / portfolio_volatility - penalty_strength * avg_weighted_corr
    # sharpe ration is just annualised return divided by annualised standard deviation
    #  but then you subtract the concentration penalty to discourage portfolios with a lack of diversification
    # the diversification penalty is not an exact science here, but it helps to cut weaken portfolios that doen't have effective diversification in the results

    results[0, i] = portfolio_volatility
    results[1, i] = portfolio_return
    results[2, i] = sharpe_ratio
#adds the sharpe ratios, volatility and the returns to the results array

# Max Sharpe & Min Volatility Portfolios

max_sharpe_idx = np.argmax(results[2])
# finds the location of the column/simulated portfolio with the highest sharpe ratio
# argmax returns an index, e.g. 3, which means the simmed portfolio with the highest sharpe ratio is in the 3rd column
min_vol_idx = np.argmin(results[0])
# finds the location of the column/simulated portfolio with the lowest volatility

max_sharpe_portfolio = results[:, max_sharpe_idx]
min_vol_portfolio = results[:, min_vol_idx]
# creates two new arrays with the highest sharpe ratio portfolio and the lowest volatility portfolio


# Plot Efficient Frontier

results_df = pd.DataFrame({
    'Volatility': results[0, :],
    'Return': results[1, :],
    'Sharpe Ratio': results[2, :],
})
weights_df = pd.DataFrame(weights_record, columns=tickers)
results_df = pd.concat([results_df, weights_df], axis=1)
# turn the results back into a pandas dataframe, which helps with plotting

fig = px.scatter(
    results_df,
    x='Volatility',
    y='Return',
    color='Sharpe Ratio',
    color_continuous_scale='green',
    hover_data=tickers,
    title='Interactive Efficient Frontier'
)
# makes interactive chart
fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.update_layout(width=900, height=600)
# adjusts size of interactive chart
fig.write_html("efficient_frontier.html", auto_open=True)
# makes html for interactive chart and saves it down in folder


# Portfolio Details

print("\nMaximum Sharpe Ratio Portfolio Allocation\n")
print("Annualised Return:", round(max_sharpe_portfolio[1], 2))
# gets annuakised return of max sharpe portfolio and rounds to two decimal places
print("Annualised Volatility:", round(max_sharpe_portfolio[0], 2))
# gets annuakised volatility of max sharpe portfolio and rounds to two decimal places
print("\nWeights:")
for ticker, weight in zip(tickers, weights_record[max_sharpe_idx]):
# essentially a loop that connects each of the tickers with the weights from the max sharpe ratio portfolio
    print(f"{ticker}: {weight:.2%}")
# just prints as string (f does this) and weights in 2 decimal places (.2%)

print("\nMinimum Volatility Portfolio Allocation\n")
print("Annualised Return:", round(min_vol_portfolio[1], 2))
print("Annualised Volatility:", round(min_vol_portfolio[0], 2))
print("\nWeights:")
for ticker, weight in zip(tickers, weights_record[min_vol_idx]):
    print(f"{ticker}: {weight:.2%}")
