#We are comparing how this montecarlo works with real past data 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockdata = yf.download(stocks, start=start, end=end)
    stockdata = stockdata['Close']
    stockdata = stockdata.dropna(axis=1, how='all')   
    returns = np.log(stockdata / stockdata.shift(1))
    meanreturns = returns.mean()
    covmatrix = returns.cov()
    return meanreturns, covmatrix, stockdata

# Stock list
stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList] 

# Backtest parameters
initialportfolio = 1000
mc_sims = 500
T = 252  # Trading days in 1 year

# 1. Train model on 2023
train_start = dt.datetime(2023,1,1)
train_end = dt.datetime(2023,12,31)

meanreturns, covariancematrix, stock_prices_2023 = get_data(stocks, train_start, train_end)

# 2. Use same weights for real and simulated
weights = np.random.random(len(meanreturns))
weights /= np.sum(weights)

# 3. Monte Carlo simulation for 2024
meanM = np.full(shape=(T,len(weights)), fill_value=meanreturns)
meanM = meanM.T
portfolio_sims = np.full(shape=(T,mc_sims), fill_value=0.0)

for m in range(0, mc_sims):
    df = 5  # Student-t for fat tails
    Z = np.random.standard_t(df, size=(T, len(weights)))
    L = np.linalg.cholesky(covariancematrix)
    daily_log_returns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.exp(np.cumsum(np.inner(weights, daily_log_returns.T))) * initialportfolio


# 4. Get actual 2024 returns
test_start = dt.datetime(2024,1,1)
test_end = dt.datetime(2024,12,31)

_, _, stock_prices_2024 = get_data(stocks, test_start, test_end)

actual_log_returns = np.log(stock_prices_2024 / stock_prices_2024.shift(1)).dropna()
actual_portfolio_log_returns = actual_log_returns.dot(weights)
actual_cum_returns = np.exp(actual_portfolio_log_returns.cumsum()) * initialportfolio


#5.Plot
plt.figure(figsize=(12,6))

# Plot all simulations in light grey
plt.plot(portfolio_sims, color='lightgrey', alpha=0.3)

# Compute and plot the average of Monte Carlo simulations
mean_simulation = portfolio_sims.mean(axis=1)
plt.plot(mean_simulation, color='green', linewidth=2, label='Average Monte Carlo Simulation')

# Plot actual 2024 portfolio
plt.plot(actual_cum_returns.values, color='red', linewidth=2, label='Actual Portfolio 2024')

plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Simulated 2024 vs Actual 2024 Portfolio Performance')
plt.legend()
plt.grid(True)



# 6. Histogram of final portfolio values
final_simulated = portfolio_sims[-1,:]

plt.figure(figsize=(10,6))
plt.hist(final_simulated, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(actual_cum_returns.values[-1], color='red', linestyle='dashed', linewidth=2, label='Actual 2024 Final Value')
plt.title('Distribution of Simulated Final Portfolio Values vs Actual')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)


#7. Error calculation
# Avoid division by zero in actual_cum_returns
actual_values = actual_cum_returns.values
# Replace zeros with small number to avoid divide by zero
actual_values[actual_values == 0] = 1e-8  

# Absolute Percentage Error
percent_error = np.abs(mean_simulation - actual_values) / actual_values * 100

average_percent_error = np.mean(percent_error)
max_percent_error = np.max(percent_error)

print(f"Average Percentage Error: {average_percent_error:.2f}%")
print(f"Maximum Percentage Error: {max_percent_error:.2f}%")
