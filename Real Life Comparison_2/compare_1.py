import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockdata = yf.download(stocks, start=start, end=end)
    stockdata = stockdata['Close']
    stockdata = stockdata.dropna(axis=1, how='all')   
    returns = stockdata.pct_change()   # <-- Simple daily returns
    meanreturns = returns.mean()
    covmatrix = returns.cov()
    return meanreturns, covmatrix, stockdata

# Stock list
stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList] 

# Parameters
initialportfolio = 1000
mc_sims = 500
T = 252

# 1. Train on 2023
train_start = dt.datetime(2023,1,1)
train_end = dt.datetime(2023,12,31)

meanreturns, covariancematrix, stock_prices_2023 = get_data(stocks, train_start, train_end)

# 2. Random Weights
weights = np.random.random(len(meanreturns))
weights /= np.sum(weights)

# 3. Monte Carlo simulation using simple returns & normal distribution
meanM = np.full(shape=(T, len(weights)), fill_value=meanreturns).T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

for m in range(mc_sims):
    Z = np.random.normal(size=(T, len(weights)))   # Normal distribution
    L = np.linalg.cholesky(covariancematrix)
    simulated_daily_returns = meanM + np.inner(L, Z)
    
    # Cumulative simple returns
    portfolio_cum_returns = np.cumprod(1 + np.inner(weights, simulated_daily_returns.T)) * initialportfolio
    portfolio_sims[:, m] = portfolio_cum_returns

# 4. Actual 2024 returns
test_start = dt.datetime(2024,1,1)
test_end = dt.datetime(2024,12,31)

_, _, stock_prices_2024 = get_data(stocks, test_start, test_end)

actual_returns = stock_prices_2024.pct_change().dropna()
actual_portfolio_returns = actual_returns.dot(weights)
actual_cum_returns = (1 + actual_portfolio_returns).cumprod() * initialportfolio


plt.figure(figsize=(12,6))
plt.plot(portfolio_sims, color='lightgrey', alpha=0.3)
mean_simulation = portfolio_sims.mean(axis=1)
plt.plot(mean_simulation, color='green', linewidth=2, label='Average Monte Carlo Simulation')
plt.plot(actual_cum_returns.values, color='red', linewidth=2, label='Actual Portfolio 2024')
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Simulated 2024 vs Actual 2024 (Simple Returns, Normal Distribution)')
plt.legend()
plt.grid(True)

# 6. Histogram of final values
final_simulated = portfolio_sims[-1,:]

plt.figure(figsize=(10,6))
plt.hist(final_simulated, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(actual_cum_returns.values[-1], color='red', linestyle='dashed', linewidth=2, label='Actual 2024 Final Value')
plt.title('Distribution of Simulated Final Portfolio Values vs Actual')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)


# 7. Error calculation
actual_values = actual_cum_returns.values
actual_values[actual_values == 0] = 1e-8  

percent_error = np.abs(mean_simulation - actual_values) / actual_values * 100

average_percent_error = np.mean(percent_error)
max_percent_error = np.max(percent_error)

print(f"Average Percentage Error: {average_percent_error:.2f}%")
print(f"Maximum Percentage Error: {max_percent_error:.2f}%")
