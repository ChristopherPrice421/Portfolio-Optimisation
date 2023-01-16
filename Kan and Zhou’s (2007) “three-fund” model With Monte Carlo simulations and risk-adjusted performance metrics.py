import numpy as np
import pandas as pd
import matplotlib as plt

# Define the expected returns, standard deviation and correlation of the three assets
stock_return = 0.15
stock_volatility = 0.20

bond_return = 0.05
bond_volatility = 0.10

cash_return = 0.03
cash_volatility = 0.02

stock_bond_corr = 0.3
stock_cash_corr = -0.2
bond_cash_corr = 0.1

# Define the covariance matrix
cov = np.array([[stock_volatility**2, stock_bond_corr*stock_volatility*bond_volatility, stock_cash_corr*stock_volatility*cash_volatility], 
               [stock_bond_corr*stock_volatility*bond_volatility, bond_volatility**2, bond_cash_corr*bond_volatility*cash_volatility],
               [stock_cash_corr*stock_volatility*cash_volatility, bond_cash_corr*bond_volatility*cash_volatility, cash_volatility**2]])

# Set the number of Monte Carlo simulations
num_sims = 1000

# Create an array to store the simulated portfolio returns
sim_returns = np.zeros(num_sims)

# Run the Monte Carlo simulations
for i in range(num_sims):
    # Generate random returns for the assets
    rand_returns = np.random.multivariate_normal([stock_return, bond_return, cash_return], cov)

    # Allocate 60% of the portfolio to stocks, 30% to bonds, and 10% to cash
    sim_returns[i] = 0.6*rand_returns[0] + 0.3*rand_returns[1] + 0.1*rand_returns[2]

# Plot the histogram of the simulated portfolio returns
plt.hist(sim_returns, bins=50)
plt.xlabel('Portfolio return')
plt.ylabel('Frequency')
plt.show()



# Load the historical data of the assets
data = pd.read_csv('asset_prices.csv', index_col='Date')

# Calculate the returns of the assets
returns = data.pct_change().dropna()

# Create an array to store the backtested portfolio returns
backtest_returns = []

# Run the backtest
for i in range(1, len(returns)):
    # Select the historical data up to the current date
    hist_returns = returns.iloc[:i]

    # Allocate 60% of the portfolio to stocks, 30% to bonds, and 10% to cash
    stock_weight = 0.6
    bond_weight = 0.3
    cash_weight = 0.1
    backtest_returns.append(stock_weight*hist_returns['stock'] + bond_weight*hist_returns['bond'] + cash_weight*hist_returns['cash'])

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
