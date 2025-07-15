# main_montecarlo.py

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockdata = yf.download(stocks, start=start, end=end)
    stockdata = stockdata['Close']
    stockdata = stockdata.dropna(axis=1, how='all')   
    returns = stockdata.pct_change()
    meanreturns = returns.mean()
    covmatrix = returns.cov()
    return meanreturns, covmatrix

def monte_carlo_portfolio(meanreturns, covmatrix, weights, mc_sims=100, T=200, initialportfolio=1000):
    meanM = np.full(shape=(T, len(weights)), fill_value=meanreturns).T
    portfolio_sims = np.zeros((T, mc_sims))

    for m in range(mc_sims):
        df = 5  # t-distribution degrees of freedom (fat tails)
        Z = np.random.standard_t(df, size=(T, len(weights)))
        L = np.linalg.cholesky(covmatrix)
        dailyreturns = meanM + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyreturns.T) + 1) * initialportfolio

    final_values = portfolio_sims[-1, :]

    # Calculate metrics
    final_returns = (final_values / initialportfolio) - 1
    mean_return = np.mean(final_returns) * (252 / T)  # Annualized return
    risk = np.std(final_returns) * np.sqrt(252 / T)   # Annualized std deviation (Risk)

    var_95 = np.percentile(final_returns, 5)  # VaR at 95%
    cvar_95 = np.mean(final_returns[final_returns <= var_95])  # CVaR at 95%

    sharpe_ratio = mean_return / risk if risk != 0 else 0

    return {
        'Return': mean_return,
        'Risk': risk,
        'VaR_95': var_95,
        'CVaR_95': cvar_95,
        'Sharpe': sharpe_ratio,
        'Weights': weights
    }
