# Efficient Frontier with Diversification Penalty

This project simulates and visualizes efficient portfolios using historical stock data while applying a diversification penalty to discourage highly correlated holdings.

---

## 🧩 Features
- Fetches stock price data from Yahoo Finance.
- Simulates thousands of random portfolios using Monte Carlo methods.
- Computes annualized return, volatility, and a *penalized* Sharpe ratio.
- Identifies:
  - Maximum Sharpe Ratio portfolio
  - Minimum Volatility portfolio
- Interactive visualization of the Efficient Frontier using Plotly.

---

## 🧠 The Diversification Penalty
Traditional portfolio optimization ignores correlation between assets.  
Here, the Sharpe ratio is penalized for portfolios with high pairwise correlations:

This encourages selecting portfolios with uncorrelated assets, promoting diversification.

Adjust the diversification penalty on line 53 of the attached code. Set right now at 0.5, but can be adjusted up or down depending on how much you want to penalise excessive concentration.

---

## ⚙️ Requirements
Install dependencies:
```bash
pip install numpy pandas yfinance matplotlib plotly
