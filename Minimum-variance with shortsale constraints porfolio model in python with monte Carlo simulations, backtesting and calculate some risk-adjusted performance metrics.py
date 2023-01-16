import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
import yfinance as yf

# Define the stock tickers you want to get data for
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']

# Get the stock data from Yahoo Finance
data = yf.download(tickers, start='2020-01-01', end='2022-01-01', group_by='ticker')

# Print the data
print(data)

# Define the expected returns and covariance matrix of the assets
returns = np.array([0.1, 0.2, 0.15])
cov = np.array([[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]])

# Define the optimization function to minimize the portfolio variance
def portfolio_variance(w, returns, cov):
    return np.dot(np.dot(w, cov), w)

# Define the optimization constraint that the weights add up to 1
def constraint_sum(w):
    return np.sum(w) - 1

# Define the constraint that the weights are greater than or equal to 0
def constraint_short(w):
    return w

# Define the optimization bounds for the weights
bounds = [(0, None) for i in range(len(returns))]

# Set the number of Monte Carlo simulations
num_sims = 1000

# Create an array to store the simulated portfolio returns
sim_returns = np.zeros(num_sims)

# Run the Monte Carlo simulations
for i in range(num_sims):
    # Generate random returns for the assets
    rand_returns = np.random.multivariate_normal(returns, cov)

    # Run the optimization to find the minimum variance portfolio with short sale constraint
    w0 = [1/len(returns) for i in range(len(returns))]
    opt = sco.minimize(portfolio_variance, w0, args=(rand_returns, cov), bounds=bounds, constraints=[{'type':'eq', 'fun':constraint_sum}, {'type':'ineq', 'fun':constraint_short}], method='SLSQP')

    # Store the simulated portfolio return
    sim_returns[i] = np.dot(opt.x, rand_returns)

# Plot the histogram of the simulated portfolio returns
plt.hist(sim_returns, bins=50)
plt.xlabel('Portfolio return')
plt.ylabel('Frequency')
plt.show()


# Calculate the returns of the assets
returns = data.pct_change().dropna()

# Create an array to store the backtested portfolio returns
backtest_returns = []

# Run the backtest
for i in range(1, len(returns)):
    # Select the historical data up to the current date
    hist_returns = returns.iloc[:i]

    # Run the optimization to find the minimum variance portfolio with short sale constraint
    w0 = [1/len(returns.columns) for i in range(len(returns.column))]
    opt = sco.minimize(portfolio_variance, w0, args=(hist_returns,), bounds=bounds, constraints=[{'type':'eq', 'fun':constraint_sum}, {'type':'ineq', 'fun':constraint_short}], method='SLSQP')

    # Calculate the backtested portfolio return
    backtest_returns.append(np.dot(opt.x, returns.iloc[i]))

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