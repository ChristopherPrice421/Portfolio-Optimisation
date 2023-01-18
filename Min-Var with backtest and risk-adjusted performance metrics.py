import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import yfinance as yf


tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']

# Get the stock data from Yahoo Finance
data = yf.download(tickers, start='2020-01-01', end='2022-01-01', group_by='ticker')


# Calculate the returns of the assets
returns = data.pct_change().dropna()

# Define the optimization function to minimize the portfolio variance
def portfolio_variance(w, returns):
    cov = np.cov(returns.T)
    return np.dot(np.dot(w, cov), w) + 1e-6  # Add small value to variance

# Define the optimization constraint that the weights add up to 1
def constraint_sum(w):
    return np.sum(w) - 1

# Define the optimization bounds for the weights
bounds = [(0, 1) for i in range(len(returns.columns))]

# Create an array to store the backtested portfolio returns
backtest_returns = []

# Run the backtest
for i in range(1, len(returns)):
    # Select the historical data up to the current date
    hist_returns = returns.iloc[:i]

    # Run the optimization to find the minimum variance portfolio
    w0 = [1/len(returns.columns) for i in range(len(returns.columns))]
    opt = sco.minimize(portfolio_variance, w0, args=(hist_returns,), bounds=bounds, constraints={'type':'eq', 'fun':constraint_sum}, method='SLSQP')

    # Calculate the backtested portfolio return
    backtest_returns.append(np.dot(opt.x, returns.iloc[i]))

# Plot the backtested portfolio returns
plt.plot(backtest_returns)
plt.xlabel('Time')
plt.ylabel('Portfolio return')
plt.show()


# Calculate the annualized portfolio return
portfolio_return = np.mean(backtest_returns) * 252

# Calculate the annualized portfolio volatility
portfolio_volatility = np.std(backtest_returns) * np.sqrt(252)

# Calculate the risk-free rate
risk_free_rate = 0.03

# Calculate the Sharpe Ratio
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
print("Sharpe Ratio:", sharpe_ratio)

# Calculate the downside deviation
downside_returns = [r for r in backtest_returns if r < risk_free_rate]
downside_deviation = np.std(downside_returns) * np.sqrt(252)

# Calculate the Sortino Ratio
sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation
print("Sortino Ratio:", sortino_ratio)

# Calculate the Maximum Drawdown
portfolio_cumulative_returns = np.cumsum(backtest_returns)
max_drawdown = (np.max(portfolio_cumulative_returns) - np.min(portfolio_cumulative_returns)) / np.max(portfolio_cumulative_returns)
print("Maximum Drawdown:", max_drawdown)
