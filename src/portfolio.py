import pandas as pd
import numpy as np
from scipy.optimize import minimize


# -------------------------------------------------
# Calculate Daily Returns
# -------------------------------------------------

def calculate_returns(price_data):
    """
    Calculate daily percentage returns
    """

    returns = price_data.pct_change().dropna()

    return returns


# -------------------------------------------------
# Portfolio Recommendation Based on Risk Level
# -------------------------------------------------

def recommend_portfolio(returns, investment_amount, risk_level):
    """
    Recommend portfolio allocation based on risk level
    """

    num_assets = len(returns.columns)

    # Simple weight allocation depending on risk level
    if risk_level == "Low":
        weights = np.repeat(1 / num_assets, num_assets)

    elif risk_level == "Medium":
        weights = np.repeat(1 / num_assets, num_assets)

    else:  # High risk
        weights = np.repeat(1 / num_assets, num_assets)

    allocation = pd.DataFrame({
        "Stock": returns.columns,
        "Weight": weights
    })

    allocation["Investment"] = allocation["Weight"] * investment_amount

    return allocation


# -------------------------------------------------
# Portfolio Performance Calculation
# -------------------------------------------------

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio return and volatility
    """

    returns = np.dot(weights, mean_returns)

    volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    return returns, volatility


# -------------------------------------------------
# Negative Sharpe Ratio (for optimization)
# -------------------------------------------------

def negative_sharpe(weights, mean_returns, cov_matrix):

    returns, volatility = portfolio_performance(
        weights,
        mean_returns,
        cov_matrix
    )

    return -returns / volatility


# -------------------------------------------------
# Optimize Portfolio (Maximum Sharpe Ratio)
# -------------------------------------------------

def optimize_portfolio(returns):
    """
    Optimize portfolio weights using Sharpe Ratio
    """

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_assets = len(mean_returns)

    constraints = ({
        'type': 'eq',
        'fun': lambda x: np.sum(x) - 1
    })

    bounds = tuple((0, 1) for _ in range(num_assets))

    init_guess = num_assets * [1 / num_assets]

    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x

    allocation = pd.DataFrame({
        "Stock": mean_returns.index,
        "Optimal Weight": optimal_weights
    })

    return allocation