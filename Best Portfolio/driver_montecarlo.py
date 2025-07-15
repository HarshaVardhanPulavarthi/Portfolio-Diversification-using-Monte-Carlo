import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from tabulate import tabulate  # For pretty console tables
from main_montecarlo import get_data, monte_carlo_portfolio

def run_driver():
    # Stock list
    stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
    stocks = [stock + '.AX' for stock in stockList] 

    # Dates
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=366)

    # Get data
    meanreturns, covmatrix = get_data(stocks, startDate, endDate)

    # Run 10000 portfolios
    portfolios = []
    for i in range(10000):
        weights = np.random.random(len(meanreturns))
        weights /= np.sum(weights)
        result = monte_carlo_portfolio(meanreturns, covmatrix, weights)
        portfolios.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(portfolios)

    # Pretty Table in console
    print("\n========= Portfolio Summary =========\n")
    table_data = []
    for idx, row in df.iterrows():
        weight_str = ', '.join([f"{w*100:.1f}%" for w in row['Weights']])
        table_data.append([idx+1, f"{row['Return']:.4f}", f"{row['Risk']:.4f}", f"{row['Sharpe']:.4f}", 
                           f"{row['VaR_95']:.4f}", f"{row['CVaR_95']:.4f}", weight_str])
    
    headers = ["Portfolio", "Return", "Risk", "Sharpe", "VaR 95%", "CVaR 95%", "Weights (%)"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    # Plot Return vs Risk
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(df['Risk'], df['Return'], c=df['Sharpe'], cmap='viridis', s=50)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Monte Carlo Portfolios: Risk vs Return')
    plt.grid(True)
    plt.show()

    # Find the best Sharpe ratio
    best_idx = df['Sharpe'].idxmax()
    best_portfolio = df.loc[best_idx]

    print("\n===== Best Portfolio =====\n")
    print(f"Sharpe Ratio: {best_portfolio['Sharpe']:.2f}")
    print(f"Expected Return: {best_portfolio['Return']:.2f}")
    print(f"Risk (Std Dev): {best_portfolio['Risk']:.2f}")
    print(f"VaR 95%: {best_portfolio['VaR_95']:.2f}")
    print(f"CVaR 95%: {best_portfolio['CVaR_95']:.2f}")

    # Pie Chart of Best Portfolio
    plt.figure(figsize=(8,8))
    weights = best_portfolio['Weights']
    labels = [f"{stockList[i]} ({weights[i]*100:.1f}%)" for i in range(len(stockList))]
    plt.pie(weights, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Best Portfolio Weights (Max Sharpe Ratio)')
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_driver()
